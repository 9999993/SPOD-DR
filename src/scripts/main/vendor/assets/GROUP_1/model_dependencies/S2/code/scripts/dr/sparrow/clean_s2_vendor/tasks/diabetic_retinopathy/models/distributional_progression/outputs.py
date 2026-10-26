from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FusedSeverityDistribution:
    mean: torch.Tensor
    variance: torch.Tensor
    std: torch.Tensor
    local_weight: torch.Tensor
    global_weight: torch.Tensor
    within_variance: torch.Tensor
    disagreement_variance: torch.Tensor
