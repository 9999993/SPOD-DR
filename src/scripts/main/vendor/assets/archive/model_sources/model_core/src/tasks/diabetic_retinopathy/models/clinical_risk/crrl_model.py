from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tasks.diabetic_retinopathy.models.clinical_risk.risk_alignment import ClinicalRiskAlignmentHead
from tasks.diabetic_retinopathy.models.clinical_risk.risk_encoder import ClinicalRiskEncoder
from tasks.diabetic_retinopathy.models.distributional_progression.distributional_progression_model import (
    DisagreementAwareProgressionModel,
)


@dataclass(frozen=True)
class DiseaseStateFeatures:
    expected_grade: torch.Tensor
    severity_variance: torch.Tensor
    cumulative_risk: torch.Tensor
    disagreement: torch.Tensor
    state_vector: torch.Tensor


class ClinicalRiskRepresentationLearningModel(DisagreementAwareProgressionModel):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        backbone_img_size: int | None = None,
        pooling: str = "gem",
        local_query_count: int = 6,
        feature_dim: int = 256,
        attention_heads: int = 4,
        mode: str = "local_global",
        beta: float = 1.0,
        crrl_mode: str = "risk_only",
        risk_hidden_dim: int = 256,
        risk_embedding_dim: int = 256,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            backbone_img_size=backbone_img_size,
            pooling=pooling,
            local_query_count=local_query_count,
            feature_dim=feature_dim,
            attention_heads=attention_heads,
            mode=mode,
            beta=beta,
        )
        resolved_mode = str(crrl_mode).strip().lower()
        if resolved_mode not in {"risk_only", "consistency", "full"}:
            raise ValueError(f"Unsupported CRRL mode: {crrl_mode}")
        self.crrl_mode = resolved_mode
        self.risk_encoder = ClinicalRiskEncoder(
            input_dim=9,
            hidden_dim=int(risk_hidden_dim),
            embedding_dim=int(risk_embedding_dim),
        )
        self.risk_alignment = ClinicalRiskAlignmentHead(embedding_dim=int(risk_embedding_dim))

    def _disease_state_features(self, output: dict[str, torch.Tensor]) -> DiseaseStateFeatures:
        grade_probs = output["grade_probs"].float()
        grade_axis = torch.arange(grade_probs.shape[1], device=grade_probs.device, dtype=grade_probs.dtype)
        expected_grade = (grade_probs * grade_axis.unsqueeze(0)).sum(dim=1)
        severity_variance = ((grade_axis.unsqueeze(0) - expected_grade.unsqueeze(1)) ** 2 * grade_probs).sum(dim=1)
        cumulative_risk = grade_probs[:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        disagreement = output["disagreement_variance"].float()
        state_vector = torch.cat(
            [
                grade_probs,
                cumulative_risk.unsqueeze(1),
                expected_grade.unsqueeze(1),
                severity_variance.unsqueeze(1),
                disagreement.unsqueeze(1),
            ],
            dim=1,
        )
        return DiseaseStateFeatures(
            expected_grade=expected_grade,
            severity_variance=severity_variance,
            cumulative_risk=cumulative_risk,
            disagreement=disagreement,
            state_vector=state_vector,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(image)
        disease_state = self._disease_state_features(output)
        risk_embedding, risk_uncertainty = self.risk_encoder(disease_state.state_vector)
        screening_logits, screening_prob = self.risk_alignment(risk_embedding)
        screening_prob = screening_prob.clamp(1e-6, 1.0 - 1e-6)
        risk_alignment_error = torch.abs(screening_prob - disease_state.cumulative_risk)
        output.update(
            {
                "binary_logits": screening_logits,
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
        return output


def build_clinical_risk_representation_learning_model(config: dict[str, Any]) -> nn.Module:
    return ClinicalRiskRepresentationLearningModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        feature_dim=int(config.get("feature_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        mode=str(config.get("distribution_mode", "local_global")),
        beta=float(config.get("disagreement_beta", 1.0)),
        crrl_mode=str(config.get("crrl_mode", "risk_only")),
        risk_hidden_dim=int(config.get("risk_hidden_dim", 256)),
        risk_embedding_dim=int(config.get("risk_embedding_dim", 256)),
    )
