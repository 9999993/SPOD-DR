from __future__ import annotations

import torch
from torch import nn


class ResidualDestylizer(nn.Module):
    """Channel-statistics normalization used on a registered feature scope."""

    def __init__(self, channels: int, *, initial_lambda: float = 0.9) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gamma = nn.Parameter(torch.ones(self.channels))
        self.beta = nn.Parameter(torch.zeros(self.channels))
        initial_lambda = min(max(float(initial_lambda), 1e-4), 1.0 - 1e-4)
        self.rho = nn.Parameter(torch.tensor(initial_lambda).logit())

    @property
    def lambda_value(self) -> torch.Tensor:
        return torch.sigmoid(self.rho)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4 or int(residual.shape[1]) != self.channels:
            raise ValueError(f"expected [B,{self.channels},H,W], got {tuple(residual.shape)}")
        mean = residual.mean(dim=(-2, -1), keepdim=True)
        variance = residual.var(dim=(-2, -1), keepdim=True, unbiased=False)
        normalized = (residual - mean) / torch.sqrt(variance + 1e-5)
        affine = normalized * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        lam = self.lambda_value.to(device=residual.device, dtype=residual.dtype)
        return lam * residual + (1.0 - lam) * affine


__all__ = ["ResidualDestylizer"]
