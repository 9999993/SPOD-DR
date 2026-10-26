from __future__ import annotations

import torch
from torch import nn

from tasks.diabetic_retinopathy.models.distributional_progression.outputs import FusedSeverityDistribution


class DisagreementFusion(nn.Module):
    def __init__(self, *, beta: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(
        self,
        *,
        local_mu: torch.Tensor,
        local_sigma: torch.Tensor,
        global_mu: torch.Tensor,
        global_sigma: torch.Tensor,
    ) -> FusedSeverityDistribution:
        local_var = local_sigma.float().pow(2)
        global_var = global_sigma.float().pow(2)
        local_precision = 1.0 / (local_var + self.eps)
        global_precision = 1.0 / (global_var + self.eps)
        precision_sum = local_precision + global_precision
        local_weight = local_precision / precision_sum
        global_weight = global_precision / precision_sum
        mean = local_weight * local_mu.float() + global_weight * global_mu.float()
        within_variance = local_weight * local_var + global_weight * global_var
        disagreement_variance = local_weight * global_weight * (local_mu.float() - global_mu.float()).pow(2)
        variance = (within_variance + self.beta * disagreement_variance).clamp_min(self.eps)
        return FusedSeverityDistribution(
            mean=mean,
            variance=variance,
            std=torch.sqrt(variance),
            local_weight=local_weight,
            global_weight=global_weight,
            within_variance=within_variance,
            disagreement_variance=disagreement_variance,
        )
