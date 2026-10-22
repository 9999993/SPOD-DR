from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SeverityDistributionHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int | None = None,
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
    ) -> None:
        super().__init__()
        resolved_hidden = int(hidden_dim or in_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), resolved_hidden),
            nn.GELU(),
            nn.LayerNorm(resolved_hidden),
        )
        self.mu_head = nn.Linear(resolved_hidden, 1)
        self.sigma_head = nn.Linear(resolved_hidden, 1)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(embedding)
        mu = 4.0 * torch.sigmoid(self.mu_head(hidden).squeeze(1))
        sigma = self.sigma_min + F.softplus(self.sigma_head(hidden).squeeze(1))
        sigma = sigma.clamp(max=self.sigma_max)
        return mu, sigma
