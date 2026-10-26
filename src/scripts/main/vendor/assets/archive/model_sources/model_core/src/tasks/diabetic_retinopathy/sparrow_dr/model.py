from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tasks.diabetic_retinopathy.models.clinical_risk.crrl_model import ClinicalRiskRepresentationLearningModel
from tasks.diabetic_retinopathy.models.distributional_progression.ordinal_gaussian_decoder import grade_probs_to_ordinal_logits
from .destyle import ResidualDestylizer
from .severity import OrderedSeverityAxis


class SparrowDRModel(nn.Module):
    """Canonical B3 classifier with training-only SPARROW feature operators."""

    def __init__(
        self,
        base: ClinicalRiskRepresentationLearningModel,
        *,
        arm_id: str,
        method_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.arm_id = str(arm_id).upper()
        self.method_enabled = bool(method_enabled)
        if self.arm_id not in {"C0", "C1", "C2", "M3"}:
            raise ValueError(f"unknown SPARROW arm: {arm_id}")
        self.axis_stage3 = OrderedSeverityAxis(48)
        self.axis_stage4 = OrderedSeverityAxis(136)
        self.destylize_stage3 = ResidualDestylizer(48)
        self.destylize_stage4 = ResidualDestylizer(136)

    @property
    def is_enabled(self) -> bool:
        return self.method_enabled and self.arm_id != "C0"

    def _transform(
        self,
        feature: torch.Tensor,
        axis: OrderedSeverityAxis,
        destylizer: ResidualDestylizer,
        *,
        scope: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        parallel, residual = axis.decompose(feature)
        score = axis.score(feature)
        if self.arm_id == "C1":
            transformed = feature
        elif self.arm_id == "C2":
            # Parameter-matched full-feature control: the learned axis remains
            # active for L_ord, but the de-stylizer sees the complete feature.
            transformed = destylizer(feature)
            parallel = torch.zeros_like(feature)
            residual = feature
        elif self.arm_id == "M3":
            transformed = parallel + destylizer(residual)
        else:  # pragma: no cover - C0 exits before this path
            transformed = feature
        return transformed, parallel, residual, score

    def _build_from_features(self, stage3: torch.Tensor, stage4: torch.Tensor) -> dict[str, torch.Tensor]:
        transformed3, parallel3, residual3, score3 = self._transform(
            stage3, self.axis_stage3, self.destylize_stage3, scope="stage3"
        )
        transformed4, parallel4, residual4, score4 = self._transform(
            stage4, self.axis_stage4, self.destylize_stage4, scope="stage4"
        )

        local_features = self.base.local_branch(transformed3)
        z_local = self.base.local_project(local_features["local_embedding_raw"])
        z_global = self.base.global_branch(transformed4)
        local_mu, local_sigma = self.base.local_head(z_local)
        global_mu, global_sigma = self.base.global_head(z_global)
        if self.base.mode == "global_only":
            fused_mu, fused_sigma = global_mu, global_sigma
            local_weight = torch.zeros_like(global_mu)
            global_weight = torch.ones_like(global_mu)
            within_variance = global_sigma.float().pow(2)
            disagreement_variance = torch.zeros_like(global_mu)
        else:
            fused = self.base.fusion(
                local_mu=local_mu,
                local_sigma=local_sigma,
                global_mu=global_mu,
                global_sigma=global_sigma,
            )
            fused_mu, fused_sigma = fused.mean, fused.std
            local_weight, global_weight = fused.local_weight, fused.global_weight
            within_variance, disagreement_variance = fused.within_variance, fused.disagreement_variance

        grade_probs = self.base._probs(fused_mu, fused_sigma)
        local_grade_probs = self.base._probs(local_mu, local_sigma)
        global_grade_probs = self.base._probs(global_mu, global_sigma)
        binary_prob = grade_probs[:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        ordinal_logits = grade_probs_to_ordinal_logits(grade_probs)
        embedding = self.base.embedding_project(
            torch.cat([z_local, z_global, torch.abs(z_local - z_global), z_local * z_global], dim=1)
        )
        output: dict[str, torch.Tensor] = {
            "binary_logits": torch.logit(binary_prob),
            "ordinal_logits": ordinal_logits,
            "logits": torch.logit(binary_prob),
            "embedding": embedding,
            "local_embedding": z_local,
            "global_embedding": z_global,
            "grade_probs": grade_probs,
            "local_grade_probs": local_grade_probs,
            "global_grade_probs": global_grade_probs,
            "binary_prob": binary_prob,
            "severity_score": fused_mu,
            "fused_mu": fused_mu,
            "fused_sigma": fused_sigma,
            "local_mu": local_mu,
            "local_sigma": local_sigma,
            "global_mu": global_mu,
            "global_sigma": global_sigma,
            "local_weight": local_weight,
            "global_weight": global_weight,
            "within_variance": within_variance,
            "disagreement_variance": disagreement_variance,
            "total_variance": fused_sigma.float().pow(2),
            "severity_scores_stage3": score3,
            "severity_scores_stage4": score4,
            "parallel_stage3": parallel3,
            "parallel_stage4": parallel4,
            "residual_stage3": residual3,
            "residual_stage4": residual4,
            "transformed_stage3": transformed3,
            "transformed_stage4": transformed4,
            "aux_cumulative_stage3": self.axis_stage3.cumulative(stage3),
            "aux_cumulative_stage4": self.axis_stage4.cumulative(stage4),
        }
        disease_state = self.base._disease_state_features(output)
        risk_embedding, risk_uncertainty = self.base.risk_encoder(disease_state.state_vector)
        screening_logits, screening_prob = self.base.risk_alignment(risk_embedding)
        screening_prob = screening_prob.clamp(1e-6, 1.0 - 1e-6)
        risk_alignment_error = torch.abs(screening_prob - disease_state.cumulative_risk)
        output.update(
            {
                "binary_logits": screening_logits,
                "logits": screening_logits,
                "binary_prob": screening_prob,
                "screening_logits": screening_logits,
                "screening_prob": screening_prob,
                "screening_probability": screening_prob,
                "clinical_risk_embedding": risk_embedding,
                "risk_uncertainty": risk_uncertainty,
                "cumulative_risk": disease_state.cumulative_risk,
                "expected_grade": disease_state.expected_grade,
                "severity_distribution_variance": disease_state.severity_variance,
                "risk_alignment_error": risk_alignment_error,
                "risk_alignment_delta": risk_alignment_error,
                "risk_order_score": screening_prob,
            }
        )
        output.update(local_features)
        return output

    def forward(self, image: torch.Tensor, *, orbit_view: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if not self.is_enabled:
            # This early return is deliberate: no new module is evaluated in
            # method-off mode, so canonical B0 tensors are bitwise preserved.
            return self.base(image)
        source_image = image if orbit_view is None else orbit_view
        stage3, stage4 = self.base.feature_pyramid(source_image)
        return self._build_from_features(stage3, stage4)


def build_sparrow_arm(arm_id: str, b3: ClinicalRiskRepresentationLearningModel, method_enabled: bool = True) -> SparrowDRModel:
    return SparrowDRModel(b3, arm_id=arm_id, method_enabled=method_enabled)


__all__ = ["SparrowDRModel", "build_sparrow_arm"]
