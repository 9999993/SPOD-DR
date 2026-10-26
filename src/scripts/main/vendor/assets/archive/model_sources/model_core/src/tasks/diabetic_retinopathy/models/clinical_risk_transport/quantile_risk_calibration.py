from __future__ import annotations

import torch
import torch.nn.functional as F


def quantile_risk_calibration_loss(
    *,
    cumulative_risk: torch.Tensor,
    transport_cumulative_risk: torch.Tensor,
    max_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    risk = cumulative_risk.float().reshape(-1)
    transport_risk = transport_cumulative_risk.float().reshape(-1)
    if risk.numel() < 2:
        zero = risk.sum() * 0.0
        return zero, torch.tensor(0.0, device=risk.device)

    ranking_mask = risk.unsqueeze(1) > risk.unsqueeze(0)
    ranking_mask.fill_diagonal_(False)
    candidate_pairs = torch.nonzero(ranking_mask, as_tuple=False)
    if candidate_pairs.numel() == 0:
        zero = risk.sum() * 0.0
        return zero, torch.tensor(0.0, device=risk.device)

    max_pairs = max(int(max_pairs), 0)
    if max_pairs == 0:
        zero = risk.sum() * 0.0
        return zero, torch.tensor(0.0, device=risk.device)
    if candidate_pairs.shape[0] > max_pairs:
        pair_indices = torch.randperm(candidate_pairs.shape[0], device=risk.device)[:max_pairs]
        candidate_pairs = candidate_pairs[pair_indices]

    left_tensor = candidate_pairs[:, 0]
    right_tensor = candidate_pairs[:, 1]
    weak_prob = torch.sigmoid(risk[left_tensor] - risk[right_tensor]).detach().clamp(1e-6, 1.0 - 1e-6)
    transport_logit = (transport_risk[left_tensor] - transport_risk[right_tensor]).float()
    loss = F.binary_cross_entropy_with_logits(transport_logit, weak_prob.float())
    return loss, torch.tensor(float(candidate_pairs.shape[0]), device=risk.device)
