from __future__ import annotations

import torch
import torch.nn.functional as F


def build_ordinal_decision_maps(
    *,
    feature_map: torch.Tensor,
    ordinal_weights: torch.Tensor,
    output_size: int = 14,
) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must be [B,C,H,W], got shape={tuple(feature_map.shape)}")
    if ordinal_weights.ndim != 2:
        raise ValueError(f"ordinal_weights must be [K,C], got shape={tuple(ordinal_weights.shape)}")
    if feature_map.shape[1] != ordinal_weights.shape[1]:
        raise ValueError(
            f"Channel mismatch between feature_map={feature_map.shape[1]} and ordinal_weights={ordinal_weights.shape[1]}"
        )
    projected = torch.einsum("bchw,kc->bkhw", feature_map, ordinal_weights)
    return F.interpolate(
        projected,
        size=(int(output_size), int(output_size)),
        mode="bilinear",
        align_corners=False,
    )
