from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _clamp_std(value: float) -> float:
    return max(float(value), 0.0)


def _sample_normal_like(
    image: torch.Tensor,
    shape: tuple[int, ...],
    std: float,
) -> torch.Tensor:
    if std <= 0.0:
        return torch.zeros(shape, dtype=image.dtype, device=image.device)
    return torch.randn(shape, dtype=image.dtype, device=image.device) * std


def _random_translate(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    if max_pixels <= 0:
        return image

    max_pixels = int(max_pixels)
    dy = int(torch.randint(-max_pixels, max_pixels + 1, size=(1,), device=image.device).item())
    dx = int(torch.randint(-max_pixels, max_pixels + 1, size=(1,), device=image.device).item())
    if dy == 0 and dx == 0:
        return image

    shifted = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0.0
    elif dy < 0:
        shifted[..., dy:, :] = 0.0
    if dx > 0:
        shifted[..., :, :dx] = 0.0
    elif dx < 0:
        shifted[..., :, dx:] = 0.0
    return shifted


def _random_phase_drop(
    image: torch.Tensor,
    phase_mask: torch.Tensor | None,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    probability = float(probability)
    if probability <= 0.0:
        return image, phase_mask

    batch_size, num_phases = image.shape[:2]
    if phase_mask is None:
        updated_mask = torch.ones(batch_size, num_phases, dtype=image.dtype, device=image.device)
    else:
        updated_mask = phase_mask.clone().to(dtype=image.dtype, device=image.device)

    updated_image = image.clone()
    drop_flags = torch.rand(batch_size, device=image.device) < probability
    for index in torch.nonzero(drop_flags, as_tuple=False).flatten().tolist():
        active = torch.nonzero(updated_mask[index] > 0, as_tuple=False).flatten()
        if active.numel() <= 1:
            continue
        phase_index = active[torch.randint(0, active.numel(), size=(1,), device=image.device)].item()
        updated_image[index, phase_index] = 0.0
        updated_mask[index, phase_index] = 0.0
    return updated_image, updated_mask


def make_robust_view(
    image: torch.Tensor,
    phase_mask: torch.Tensor | None,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Create a second training view that simulates center/protocol style shift.

    Inputs are fixed-HU normalized ROI tensors in [0, 1]. The perturbations are
    intentionally applied only inside the training loop, never during validation
    or external MCT benchmarking.
    """

    augment_probability = float(getattr(config, "augment_probability", 1.0))
    if augment_probability <= 0.0 or torch.rand((), device=image.device).item() > augment_probability:
        return image.clone(), None if phase_mask is None else phase_mask.clone()

    view = image.clone()
    batch_size, num_phases = view.shape[:2]

    global_scale_std = _clamp_std(getattr(config, "global_scale_std", 0.0))
    global_shift_std = _clamp_std(getattr(config, "global_shift_std", 0.0))
    phase_scale_std = _clamp_std(getattr(config, "phase_scale_std", 0.0))
    phase_shift_std = _clamp_std(getattr(config, "phase_shift_std", 0.0))

    global_scale = 1.0 + _sample_normal_like(view, (batch_size, 1, 1, 1), global_scale_std)
    global_shift = _sample_normal_like(view, (batch_size, 1, 1, 1), global_shift_std)
    phase_scale = 1.0 + _sample_normal_like(view, (batch_size, num_phases, 1, 1), phase_scale_std)
    phase_shift = _sample_normal_like(view, (batch_size, num_phases, 1, 1), phase_shift_std)
    view = view * global_scale * phase_scale + global_shift + phase_shift

    gamma_min = float(getattr(config, "gamma_min", 1.0))
    gamma_max = float(getattr(config, "gamma_max", 1.0))
    if gamma_min > 0.0 and gamma_max > 0.0 and abs(gamma_max - gamma_min) > 1e-8:
        lower, upper = sorted((gamma_min, gamma_max))
        gamma = torch.empty(batch_size, 1, 1, 1, dtype=view.dtype, device=view.device).uniform_(lower, upper)
        view = view.clamp(0.0, 1.0).pow(gamma)

    noise_std = _clamp_std(getattr(config, "noise_std", 0.0))
    if noise_std > 0.0:
        view = view + torch.randn_like(view) * noise_std

    blur_probability = float(getattr(config, "blur_probability", 0.0))
    if blur_probability > 0.0 and torch.rand((), device=view.device).item() < blur_probability:
        view = F.avg_pool2d(F.pad(view, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)

    view = _random_translate(view, int(getattr(config, "translate_pixels", 0)))
    view = view.clamp(0.0, 1.0)

    view, updated_mask = _random_phase_drop(
        view,
        phase_mask,
        probability=float(getattr(config, "view_phase_drop_prob", 0.0)),
    )
    return view, updated_mask


def feature_consistency_loss(
    clean_feature: torch.Tensor | None,
    augmented_feature: torch.Tensor | None,
) -> torch.Tensor:
    if clean_feature is None or augmented_feature is None:
        if augmented_feature is not None:
            return torch.zeros((), dtype=augmented_feature.dtype, device=augmented_feature.device)
        if clean_feature is not None:
            return torch.zeros((), dtype=clean_feature.dtype, device=clean_feature.device)
        return torch.zeros(())
    clean = F.normalize(clean_feature.detach().float(), dim=-1)
    augmented = F.normalize(augmented_feature.float(), dim=-1)
    return F.mse_loss(augmented, clean)


def logit_consistency_loss(
    clean_logits: torch.Tensor | None,
    augmented_logits: torch.Tensor | None,
) -> torch.Tensor:
    if clean_logits is None or augmented_logits is None:
        if augmented_logits is not None:
            return torch.zeros((), dtype=augmented_logits.dtype, device=augmented_logits.device)
        if clean_logits is not None:
            return torch.zeros((), dtype=clean_logits.dtype, device=clean_logits.device)
        return torch.zeros(())
    teacher = F.softmax(clean_logits.detach().float(), dim=-1)
    student = F.log_softmax(augmented_logits.float(), dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean")
