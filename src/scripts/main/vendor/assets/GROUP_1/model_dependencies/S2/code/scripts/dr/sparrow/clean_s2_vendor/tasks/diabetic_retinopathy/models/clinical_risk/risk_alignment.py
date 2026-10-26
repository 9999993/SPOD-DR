from __future__ import annotations

import torch
from torch import nn


class ClinicalRiskAlignmentHead(nn.Module):
    def __init__(self, *, embedding_dim: int = 256) -> None:
        super().__init__()
        self.logit_head = nn.Linear(int(embedding_dim), 1)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logit_head(embedding.float()).squeeze(1)
        probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        return logits, probability
