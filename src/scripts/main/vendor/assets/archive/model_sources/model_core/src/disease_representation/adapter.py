from __future__ import annotations

from torch import nn


class DiseaseRepresentationAdapter(nn.Module):
    """Lightweight MLP that maps backbone features into a shared disease space."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

    def forward(self, feature):
        return self.net(feature)
