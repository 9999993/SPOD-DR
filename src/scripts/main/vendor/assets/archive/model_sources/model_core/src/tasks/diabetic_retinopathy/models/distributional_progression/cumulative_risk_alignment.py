from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tasks.diabetic_retinopathy.models.distributional_progression.distributional_progression_model import (
    DisagreementAwareProgressionModel,
)


@dataclass(frozen=True)
class RiskFeatureBundle:
    cumulative_risk: torch.Tensor
    expected_grade: torch.Tensor
    severity_variance: torch.Tensor
    disagreement: torch.Tensor


class MonotonicRiskCalibrationLayer(nn.Module):
    def __init__(
        self,
        *,
        init_risk_weight: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        risk_weight = torch.tensor(float(init_risk_weight), dtype=torch.float32)
        self.risk_weight_unconstrained = nn.Parameter(torch.log(torch.expm1(risk_weight.clamp_min(1e-4))))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def positive_risk_weight(self) -> torch.Tensor:
        return F.softplus(self.risk_weight_unconstrained)

    def forward(
        self,
        *,
        cumulative_risk: torch.Tensor,
    ) -> torch.Tensor:
        return self.positive_risk_weight() * cumulative_risk.float() + self.bias


class BoundedRiskResidualLayer(nn.Module):
    def __init__(
        self,
        *,
        max_delta: float = 0.2,
        init_variance_weight: float = 0.0,
        init_disagreement_weight: float = 0.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.variance_weight = nn.Parameter(torch.tensor(float(init_variance_weight), dtype=torch.float32))
        self.disagreement_weight = nn.Parameter(torch.tensor(float(init_disagreement_weight), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(
        self,
        *,
        cumulative_risk: torch.Tensor,
        severity_variance: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        correction_signal = (
            self.variance_weight * torch.log1p(severity_variance.float().clamp_min(0.0))
            + self.disagreement_weight * torch.log1p(disagreement.float().clamp_min(0.0))
            + self.bias
        )
        delta = self.max_delta * torch.tanh(correction_signal)
        screening_prob = (cumulative_risk.float() + delta).clamp(1e-6, 1.0 - 1e-6)
        return screening_prob, delta


class CumulativeRiskAlignedProgressionModel(DisagreementAwareProgressionModel):
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
        beta: float = 0.0,
        cra_mode: str = "none",
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
        resolved_mode = str(cra_mode).strip().lower()
        if resolved_mode not in {"none", "cumulative", "disagreement"}:
            raise ValueError(f"Unsupported CRA mode: {cra_mode}")
        self.cra_mode = resolved_mode
        self.risk_calibration = MonotonicRiskCalibrationLayer()
        self.risk_residual = BoundedRiskResidualLayer(max_delta=0.2)

    def _risk_features(self, output: dict[str, torch.Tensor]) -> RiskFeatureBundle:
        grade_probs = output["grade_probs"].float()
        grade_axis = torch.arange(grade_probs.shape[1], device=grade_probs.device, dtype=grade_probs.dtype)
        expected_grade = (grade_probs * grade_axis.unsqueeze(0)).sum(dim=1)
        severity_variance = ((grade_axis.unsqueeze(0) - expected_grade.unsqueeze(1)) ** 2 * grade_probs).sum(dim=1)
        cumulative_risk = grade_probs[:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        disagreement = output["disagreement_variance"].float()
        return RiskFeatureBundle(
            cumulative_risk=cumulative_risk,
            expected_grade=expected_grade,
            severity_variance=severity_variance,
            disagreement=disagreement,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(image)
        risk = self._risk_features(output)
        if self.cra_mode == "none":
            screening_prob = risk.cumulative_risk
            screening_logits = torch.logit(screening_prob)
            cra_delta = torch.zeros_like(screening_prob)
        else:
            if self.cra_mode == "cumulative":
                screening_logits = self.risk_calibration(cumulative_risk=risk.cumulative_risk)
                screening_prob = torch.sigmoid(screening_logits).clamp(1e-6, 1.0 - 1e-6)
                cra_delta = screening_prob - risk.cumulative_risk
            else:
                screening_prob, cra_delta = self.risk_residual(
                    cumulative_risk=risk.cumulative_risk,
                    severity_variance=risk.severity_variance,
                    disagreement=risk.disagreement,
                )
                screening_logits = torch.logit(screening_prob)
        local_prob = output["local_grade_probs"][:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        global_prob = output["global_grade_probs"][:, 2:].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        zero_like = torch.zeros_like(screening_prob)
        output.update(
            {
                "binary_logits": screening_logits,
                "binary_prob": screening_prob,
                "screening_logits": screening_logits,
                "screening_prob": screening_prob,
                "cumulative_risk": risk.cumulative_risk,
                "expected_grade": risk.expected_grade,
                "severity_distribution_variance": risk.severity_variance,
                "risk_alignment_delta": torch.abs(screening_prob - risk.cumulative_risk),
                "cra_signed_delta": cra_delta,
                "local_prob_pos": local_prob,
                "global_prob_pos": global_prob,
                "local_binary_logits": torch.logit(local_prob),
                "global_binary_logits": torch.logit(global_prob),
                "cra_positive_risk_weight": self.risk_calibration.positive_risk_weight().expand_as(screening_prob),
                "cra_bias": self.risk_calibration.bias.expand_as(screening_prob),
                "cra_variance_weight": self.risk_residual.variance_weight.expand_as(screening_prob),
                "cra_disagreement_weight": self.risk_residual.disagreement_weight.expand_as(screening_prob),
                "cra_max_delta": screening_prob.new_full(screening_prob.shape, self.risk_residual.max_delta),
                "cra_expected_grade_weight": zero_like,
            }
        )
        return output


def build_cumulative_risk_alignment_model(config: dict[str, Any]) -> nn.Module:
    return CumulativeRiskAlignedProgressionModel(
        backbone_name=str(config.get("backbone", "efficientnet_b3")),
        pretrained=bool(config.get("pretrained", True)),
        backbone_img_size=(int(config["backbone_img_size"]) if config.get("backbone_img_size") is not None else None),
        pooling=str(config.get("pooling", "gem")),
        local_query_count=int(config.get("local_query_count", 6)),
        feature_dim=int(config.get("feature_dim", 256)),
        attention_heads=int(config.get("attention_heads", 4)),
        mode=str(config.get("distribution_mode", "local_global")),
        beta=float(config.get("disagreement_beta", 0.0)),
        cra_mode=str(config.get("cra_mode", "none")),
    )
