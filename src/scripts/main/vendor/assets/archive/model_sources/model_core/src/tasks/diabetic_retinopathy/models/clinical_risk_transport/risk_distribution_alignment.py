from __future__ import annotations

import torch
import torch.nn.functional as F


def risk_distribution_alignment_loss(
    *,
    risk_embedding: torch.Tensor,
    transport_risk_embedding: torch.Tensor,
    cumulative_risk: torch.Tensor,
    transport_cumulative_risk: torch.Tensor,
) -> torch.Tensor:
    embed_loss = F.mse_loss(risk_embedding.float(), transport_risk_embedding.float())
    risk_loss = F.l1_loss(cumulative_risk.float(), transport_cumulative_risk.float())
    return embed_loss + risk_loss
