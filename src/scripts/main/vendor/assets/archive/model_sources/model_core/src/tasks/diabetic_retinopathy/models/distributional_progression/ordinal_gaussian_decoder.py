from __future__ import annotations

import torch


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / (2.0**0.5)))


def compute_ordinal_gaussian_probs(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Map continuous DR severity N(mu, sigma^2) to fixed grade bins 0..4."""
    mu_f = mu.float()
    sigma_f = sigma.float().clamp_min(1e-6)
    boundaries = torch.tensor([0.5, 1.5, 2.5, 3.5], device=mu_f.device, dtype=torch.float32)
    cdf = _normal_cdf((boundaries.unsqueeze(0) - mu_f.unsqueeze(1)) / sigma_f.unsqueeze(1))
    probs = torch.cat(
        [
            cdf[:, :1],
            cdf[:, 1:2] - cdf[:, 0:1],
            cdf[:, 2:3] - cdf[:, 1:2],
            cdf[:, 3:4] - cdf[:, 2:3],
            1.0 - cdf[:, 3:4],
        ],
        dim=1,
    )
    probs = probs.clamp_min(1e-7)
    return probs / probs.sum(dim=1, keepdim=True)


def grade_probs_to_ordinal_logits(grade_probs: torch.Tensor) -> torch.Tensor:
    cumulative_greater = torch.stack([grade_probs[:, threshold + 1 :].sum(dim=1) for threshold in range(4)], dim=1)
    return torch.logit(cumulative_greater.clamp(1e-6, 1.0 - 1e-6))
