from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiseasePrototypeHead(nn.Module):
    """Prototype scaffold for a later UDR progression-geometry phase."""

    def __init__(self, *, dim: int = 256, num_grades: int = 5) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(int(num_grades), int(dim)) * 0.02)

    def prototype_loss(self, embedding: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
        target = self.prototypes[grades.to(dtype=torch.long)]
        return F.mse_loss(embedding.float(), target.float())

    def distances(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.cdist(embedding.float(), self.prototypes.float())

    def severity_score(self, embedding: torch.Tensor) -> torch.Tensor:
        distances = self.distances(embedding)
        grade_axis = torch.arange(
            self.prototypes.shape[0],
            device=embedding.device,
            dtype=embedding.dtype,
        )
        weights = torch.softmax(-distances, dim=1)
        return (weights * grade_axis.unsqueeze(0)).sum(dim=1)

    def order_margin_loss(self, margin: float = 0.25) -> torch.Tensor:
        if self.prototypes.shape[0] < 3:
            return self.prototypes.sum() * 0.0
        adjacent = torch.linalg.vector_norm(self.prototypes[1:] - self.prototypes[:-1], dim=1)
        if adjacent.numel() < 2:
            return adjacent.sum() * 0.0
        violations = F.relu(float(margin) - adjacent[1:] + adjacent[:-1])
        return violations.mean()
