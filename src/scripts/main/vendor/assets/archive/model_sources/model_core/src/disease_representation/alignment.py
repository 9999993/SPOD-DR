from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_alignment_loss(*embeddings: torch.Tensor) -> torch.Tensor:
    """Mean pairwise L2 alignment loss for later cross-backbone experiments."""

    if len(embeddings) < 2:
        raise ValueError("At least two embeddings are required for alignment.")
    losses: list[torch.Tensor] = []
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1 :]:
            losses.append(F.mse_loss(left.float(), right.float()))
    return torch.stack(losses).mean()
