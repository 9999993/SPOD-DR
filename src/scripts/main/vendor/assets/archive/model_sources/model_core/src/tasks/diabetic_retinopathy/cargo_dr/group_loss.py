from __future__ import annotations

import torch


def compute_grade_conditioned_soft_cvar_loss(
    *,
    per_sample_loss: torch.Tensor,
    grades: torch.Tensor,
    group_probs: torch.Tensor,
    num_grades: int = 5,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    if per_sample_loss.ndim != 1:
        raise ValueError(f"per_sample_loss must be 1D, got shape={tuple(per_sample_loss.shape)}")
    if grades.ndim != 1 or grades.shape[0] != per_sample_loss.shape[0]:
        raise ValueError("grades must be a 1D tensor aligned with per_sample_loss")
    if group_probs.ndim != 2 or group_probs.shape[0] != per_sample_loss.shape[0]:
        raise ValueError("group_probs must be a [B,G] tensor aligned with per_sample_loss")

    alpha_value = min(max(float(alpha), 0.0), 1.0)
    group_risks: list[torch.Tensor] = []
    for grade_index in range(int(num_grades)):
        grade_mask = grades == int(grade_index)
        if not bool(grade_mask.any()):
            continue
        grade_losses = per_sample_loss[grade_mask]
        grade_membership = group_probs[grade_mask]
        membership_mass = grade_membership.sum(dim=0).clamp_min(float(eps))
        weighted_losses = (grade_membership * grade_losses.unsqueeze(1)).sum(dim=0) / membership_mass
        sorted_risks, _ = torch.sort(weighted_losses, descending=True)
        tail_count = max(int(round(sorted_risks.numel() * max(alpha_value, 1e-6))), 1)
        group_risks.append(sorted_risks[:tail_count].mean())
    if not group_risks:
        return per_sample_loss.mean() * 0.0
    return torch.stack(group_risks).mean()
