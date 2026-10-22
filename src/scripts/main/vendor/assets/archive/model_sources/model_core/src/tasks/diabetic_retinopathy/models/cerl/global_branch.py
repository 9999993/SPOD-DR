from __future__ import annotations

import torch
from torch import nn

from tasks.diabetic_retinopathy.models.common import GeMPooling


class GlobalRetinaBranch(nn.Module):
    def __init__(self, *, stage4_dim: int, fusion_dim: int, pooling: str) -> None:
        super().__init__()
        self.pool = GeMPooling() if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(stage4_dim, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, stage4: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(stage4)
        if pooled.ndim == 4:
            pooled = pooled.flatten(1)
        return self.project(pooled)
