from __future__ import annotations

import torch
from torch.nn import functional as F


def ordinal_ranking_loss(*, severity_score: torch.Tensor, grades: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    grades = grades.to(dtype=torch.long)
    lower_higher = grades.unsqueeze(1) < grades.unsqueeze(0)
    if not torch.any(lower_higher):
        return severity_score.sum() * 0.0
    lower_scores = severity_score.unsqueeze(1).expand_as(lower_higher)
    higher_scores = severity_score.unsqueeze(0).expand_as(lower_higher)
    violations = F.relu(float(margin) - higher_scores[lower_higher] + lower_scores[lower_higher])
    return violations.mean()
