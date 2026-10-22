from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClinicalRiskEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )
        self.uncertainty_head = nn.Linear(int(embedding_dim), 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.projection(features.float())
        uncertainty = F.softplus(self.uncertainty_head(embedding).squeeze(1))
        return embedding, uncertainty
