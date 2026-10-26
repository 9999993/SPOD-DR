from __future__ import annotations

import torch
from torch.nn import functional as F


def binary_kl_agreement(
    *,
    local_logits: torch.Tensor,
    global_logits: torch.Tensor,
    fusion_logits: torch.Tensor,
) -> torch.Tensor:
    local_logprob = F.log_softmax(torch.stack([torch.zeros_like(local_logits), local_logits], dim=1), dim=1)
    global_logprob = F.log_softmax(torch.stack([torch.zeros_like(global_logits), global_logits], dim=1), dim=1)
    fusion_logprob = F.log_softmax(torch.stack([torch.zeros_like(fusion_logits), fusion_logits], dim=1), dim=1)
    local_prob = local_logprob.exp()
    global_prob = global_logprob.exp()
    return 0.5 * (
        F.kl_div(fusion_logprob, local_prob, reduction="batchmean")
        + F.kl_div(fusion_logprob, global_prob, reduction="batchmean")
        + F.kl_div(local_logprob, fusion_logprob.exp(), reduction="batchmean")
        + F.kl_div(global_logprob, fusion_logprob.exp(), reduction="batchmean")
    )
