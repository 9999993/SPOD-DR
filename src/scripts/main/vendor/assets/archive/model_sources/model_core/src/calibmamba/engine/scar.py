from __future__ import annotations

import inspect
import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


PHASE_SUBSET_MASKS: dict[str, tuple[float, float, float, float]] = {
    "full": (1.0, 1.0, 1.0, 1.0),
    "missing_plain": (0.0, 1.0, 1.0, 1.0),
    "missing_arterial": (1.0, 0.0, 1.0, 1.0),
    "missing_portal_venous": (1.0, 1.0, 0.0, 1.0),
    "missing_delayed": (1.0, 1.0, 1.0, 0.0),
    "only_plain": (1.0, 0.0, 0.0, 0.0),
    "only_arterial": (0.0, 1.0, 0.0, 0.0),
    "only_portal_venous": (0.0, 0.0, 1.0, 0.0),
    "only_delayed": (0.0, 0.0, 0.0, 1.0),
    "pair_plain_arterial": (1.0, 1.0, 0.0, 0.0),
    "pair_arterial_portal_venous": (0.0, 1.0, 1.0, 0.0),
    "pair_portal_venous_delayed": (0.0, 0.0, 1.0, 1.0),
}


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


def gradient_reverse(input_tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
    return _GradientReverse.apply(input_tensor, float(lambda_value))


class DomainAdversarialHead(nn.Module):
    """Small GRL domain classifier attached only during training."""

    def __init__(self, feature_dim: int, num_domains: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_domains = max(int(num_domains), 1)
        hidden = int(hidden_dim or max(32, min(256, self.feature_dim)))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.num_domains),
        )

    def forward(self, feature: torch.Tensor, lambda_value: float = 1.0) -> torch.Tensor:
        return self.classifier(gradient_reverse(feature, lambda_value))


def build_style_pseudo_domain_tensor(
    image: torch.Tensor,
    phase_mask: torch.Tensor | None,
    num_domains: int,
) -> torch.Tensor:
    """Bucket normalized ROI style statistics into deterministic pseudo-domains."""
    num_domains = max(int(num_domains), 1)
    if num_domains <= 1:
        return torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
    if image.ndim != 4:
        raise ValueError(f"Expected image shaped [B,C,H,W], got {tuple(image.shape)}")

    with torch.no_grad():
        mask = torch.ones(image.shape[:2], dtype=image.dtype, device=image.device)
        if phase_mask is not None:
            mask = phase_mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        phase_mean = image.float().mean(dim=(-2, -1))
        phase_std = image.float().std(dim=(-2, -1), unbiased=False)
        masked_mean = (phase_mean * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        masked_std = (phase_std * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if image.shape[1] >= 4:
            arterial_delta = phase_mean[:, 1] - phase_mean[:, 0]
            portal_delta = phase_mean[:, 2] - phase_mean[:, 0]
            delayed_delta = phase_mean[:, 3] - phase_mean[:, 2]
            enhancement_score = 0.35 * arterial_delta + 0.45 * portal_delta + 0.20 * delayed_delta
        else:
            enhancement_score = torch.zeros_like(masked_mean)
        style_score = 0.55 * masked_mean + 0.25 * masked_std + 0.20 * enhancement_score
        style_score = style_score.clamp(0.0, 1.0 - 1e-6)
        return torch.floor(style_score * float(num_domains)).to(dtype=torch.long).clamp(0, num_domains - 1)

def domain_lambda_for_epoch(epoch: int, warmup_epochs: int) -> float:
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs <= 0:
        return 1.0
    progress = min(max(float(epoch + 1) / float(warmup_epochs), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def supervised_contrastive_loss(
    feature: torch.Tensor | None,
    target: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if feature is None:
        return torch.zeros((), device=target.device)
    if feature.shape[0] <= 1:
        return torch.zeros((), dtype=feature.dtype, device=feature.device)
    temperature = max(float(temperature), 1e-6)
    normalized = F.normalize(feature.float(), dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = target.view(-1, 1).eq(target.view(1, -1)) & ~self_mask
    valid = positive_mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return torch.zeros((), dtype=feature.dtype, device=feature.device)
    logits_masked = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    positive_log_prob = positive_log_prob / positive_mask.sum(dim=1).clamp_min(1)
    return -positive_log_prob[valid].mean().to(dtype=feature.dtype)


def _resolve_subset_mask(name: str, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = name.strip().lower()
    if key not in PHASE_SUBSET_MASKS:
        raise ValueError(f"Unsupported SCAR phase subset: {name}")
    values = PHASE_SUBSET_MASKS[key]
    if len(values) != channels:
        raise ValueError(f"SCAR phase subset {name} expects {len(values)} channels, got {channels}.")
    return torch.tensor(values, dtype=dtype, device=device)


def select_phase_comp_subsets(
    subset_names: list[str],
    max_views: int,
    epoch: int,
) -> list[str]:
    names = [name for name in subset_names if name.strip()]
    if not names:
        return []
    max_views = int(max_views)
    if max_views <= 0 or max_views >= len(names):
        return names
    offset = int(epoch) % len(names)
    return [names[(offset + index) % len(names)] for index in range(max_views)]


def apply_phase_subset(
    image: torch.Tensor,
    phase_mask: torch.Tensor | None,
    subset_name: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if image.ndim != 4:
        raise ValueError(f"Expected image shaped [B,C,H,W], got {tuple(image.shape)}")
    mask = _resolve_subset_mask(subset_name, image.shape[1], image.device, image.dtype)
    subset_image = image * mask.view(1, -1, 1, 1)
    if phase_mask is None:
        subset_phase_mask = mask.view(1, -1).expand(image.shape[0], -1).clone()
    else:
        subset_phase_mask = phase_mask.to(device=image.device, dtype=image.dtype) * mask.view(1, -1)
    return subset_image, subset_phase_mask


def _model_signature(model: nn.Module):
    module = getattr(model, "module", model)
    try:
        return inspect.signature(module.forward)
    except (TypeError, ValueError):
        return None


def _model_accepts_argument(model: nn.Module, argument_name: str) -> bool:
    signature = _model_signature(model)
    return bool(signature is not None and argument_name in signature.parameters)


def phase_compositional_consistency_loss(
    model: nn.Module,
    image: torch.Tensor,
    context_image: torch.Tensor | None,
    phase_mask: torch.Tensor | None,
    clean_output: dict[str, torch.Tensor | None],
    subset_names: list[str],
    max_views: int,
    epoch: int,
    logit_weight: float,
    feature_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logit_weight = max(float(logit_weight), 0.0)
    feature_weight = max(float(feature_weight), 0.0)
    if logit_weight <= 0.0 and feature_weight <= 0.0:
        return torch.zeros((), device=image.device), {"views": 0.0, "logit": 0.0, "feature": 0.0}

    clean_logits = clean_output.get("logits")
    clean_feature = clean_output.get("feature")
    if clean_logits is None and clean_feature is None:
        return torch.zeros((), device=image.device), {"views": 0.0, "logit": 0.0, "feature": 0.0}

    selected_subsets = select_phase_comp_subsets(subset_names, max_views=max_views, epoch=epoch)
    if not selected_subsets:
        return torch.zeros((), device=image.device), {"views": 0.0, "logit": 0.0, "feature": 0.0}

    temperature = max(float(temperature), 1e-6)
    total_loss = torch.zeros((), device=image.device)
    logit_loss_total = torch.zeros((), device=image.device)
    feature_loss_total = torch.zeros((), device=image.device)
    views = 0

    for subset_name in selected_subsets:
        subset_image, subset_phase_mask = apply_phase_subset(image, phase_mask, subset_name)
        subset_kwargs: dict[str, torch.Tensor | None] = {}
        if _model_accepts_argument(model, "phase_mask"):
            subset_kwargs["phase_mask"] = subset_phase_mask
        if context_image is not None and _model_accepts_argument(model, "context_image"):
            subset_context_image, _ = apply_phase_subset(context_image, phase_mask, subset_name)
            subset_kwargs["context_image"] = subset_context_image
        subset_output = model(subset_image, **subset_kwargs)
        subset_loss = torch.zeros((), device=image.device)

        subset_logits = subset_output.get("logits")
        if logit_weight > 0.0 and clean_logits is not None and subset_logits is not None:
            teacher_prob = F.softmax(clean_logits.detach().float() / temperature, dim=-1)
            student_log_prob = F.log_softmax(subset_logits.float() / temperature, dim=-1)
            logit_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature * temperature)
            if torch.isfinite(logit_loss.detach()):
                subset_loss = subset_loss + logit_weight * logit_loss
                logit_loss_total = logit_loss_total + logit_loss.detach()

        subset_feature = subset_output.get("feature")
        if feature_weight > 0.0 and clean_feature is not None and subset_feature is not None:
            clean_normalized = F.normalize(clean_feature.detach().float(), dim=-1)
            subset_normalized = F.normalize(subset_feature.float(), dim=-1)
            feature_loss = F.mse_loss(subset_normalized, clean_normalized)
            if torch.isfinite(feature_loss.detach()):
                subset_loss = subset_loss + feature_weight * feature_loss
                feature_loss_total = feature_loss_total + feature_loss.detach()

        if torch.isfinite(subset_loss.detach()):
            total_loss = total_loss + subset_loss
            views += 1

    if views <= 0:
        return torch.zeros((), device=image.device), {"views": 0.0, "logit": 0.0, "feature": 0.0}

    total_loss = total_loss / float(views)
    stats = {
        "views": float(views),
        "logit": float((logit_loss_total / float(views)).detach().cpu()),
        "feature": float((feature_loss_total / float(views)).detach().cpu()),
    }
    return total_loss, stats
