from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class OrderedSeverityAxis(nn.Module):
    """Learn one ordered scalar severity axis for a feature scale."""

    def __init__(self, channels: int, *, cutpoint_scale: float = 0.5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.u = nn.Parameter(torch.randn(self.channels) * 0.02)
        self.cutpoints = nn.Parameter(torch.linspace(-1.0, 1.0, 4) * float(cutpoint_scale))

    def unit(self) -> torch.Tensor:
        return F.normalize(self.u.float(), dim=0, eps=1e-12).to(dtype=self.u.dtype)

    def decompose(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4 or int(feature.shape[1]) != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(feature.shape)}")
        unit = self.unit().to(device=feature.device, dtype=feature.dtype)
        scalar = torch.sum(feature * unit.view(1, -1, 1, 1), dim=1, keepdim=True)
        parallel = scalar * unit.view(1, -1, 1, 1)
        residual = feature - parallel
        return parallel, residual

    def score(self, feature: torch.Tensor) -> torch.Tensor:
        unit = self.unit().to(device=feature.device, dtype=feature.dtype)
        pooled = feature.mean(dim=(-2, -1))
        return torch.sum(pooled * unit.unsqueeze(0), dim=1)

    def cumulative(self, feature: torch.Tensor) -> torch.Tensor:
        score = self.score(feature).float().unsqueeze(1)
        # Registered cutpoints are kept ordered by cumulative softplus gaps.
        first = self.cutpoints[0]
        gaps = F.softplus(self.cutpoints[1:]) + 1e-4
        points = torch.cat([first.reshape(1), first.reshape(1) + torch.cumsum(gaps, dim=0)])
        return torch.sigmoid(score - points.float().view(1, -1))

    def reconstruction_error(self, feature: torch.Tensor) -> float:
        parallel, residual = self.decompose(feature)
        return float((parallel + residual - feature).abs().max().detach().cpu())


@dataclass(frozen=True)
class SeverityDecomposition:
    parallel: torch.Tensor
    residual: torch.Tensor
    transformed: torch.Tensor
    score: torch.Tensor


__all__ = ["OrderedSeverityAxis", "SeverityDecomposition"]
