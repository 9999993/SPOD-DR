from __future__ import annotations

from contextlib import nullcontext
import inspect
from typing import Any

import torch
from torch import nn

from calibmamba.engine.robustness import (
    feature_consistency_loss,
    logit_consistency_loss,
    make_robust_view,
)
from calibmamba.engine.high_specificity import negative_tail_loss
from calibmamba.engine.scar import (
    build_style_pseudo_domain_tensor,
    domain_lambda_for_epoch,
    phase_compositional_consistency_loss,
    supervised_contrastive_loss,
)


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type not in {"cuda", "npu"}:
        return nullcontext()
    if amp_dtype.lower() in {"none", "off", "fp32"}:
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
    try:
        return torch.autocast(device_type=device.type, dtype=dtype)
    except Exception:
        return nullcontext()


def _compute_loss(
    output_dict: dict[str, torch.Tensor | None],
    target: torch.Tensor,
    criterion: Any,
    epoch: int,
) -> dict[str, torch.Tensor]:
    if hasattr(criterion, "__class__") and criterion.__class__.__name__ in {
        "EvidentialLoss",
        "OriginalEvidentialLoss",
        "ReEvidentialLoss",
    }:
        return criterion(output_dict, target, epoch)

    logits = output_dict["logits"]
    if logits is None:
        raise ValueError("Cross-entropy training requires logits.")
    loss = criterion(logits, target)
    return {
        "loss": loss,
        "loss_nll": loss.detach(),
        "loss_kl": torch.zeros((), device=loss.device),
    }


def _model_signature(model: nn.Module):
    module = getattr(model, "module", model)
    try:
        return inspect.signature(module.forward)
    except (TypeError, ValueError):
        return None


def _model_accepts_argument(model: nn.Module, argument_name: str) -> bool:
    signature = _model_signature(model)
    return bool(signature is not None and argument_name in signature.parameters)


def _forward_model(
    model: nn.Module,
    image: torch.Tensor,
    phase_mask: torch.Tensor | None = None,
    context_image: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    hard_negative_mask: torch.Tensor | None = None,
    hard_negative_weight: torch.Tensor | None = None,
):
    kwargs: dict[str, torch.Tensor | None] = {}
    if _model_accepts_argument(model, "phase_mask"):
        kwargs["phase_mask"] = phase_mask
    if context_image is not None and _model_accepts_argument(model, "context_image"):
        kwargs["context_image"] = context_image
    if target is not None and _model_accepts_argument(model, "target"):
        kwargs["target"] = target
    if hard_negative_mask is not None and _model_accepts_argument(model, "hard_negative_mask"):
        kwargs["hard_negative_mask"] = hard_negative_mask
    if hard_negative_weight is not None and _model_accepts_argument(model, "hard_negative_weight"):
        kwargs["hard_negative_weight"] = hard_negative_weight
    return model(image, **kwargs)


def _merge_model_auxiliary_outputs(
    output: dict[str, Any],
    loss: torch.Tensor,
    loss_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    auxiliary_losses = output.get("auxiliary_losses") or {}
    auxiliary_stats = output.get("auxiliary_stats") or {}
    for aux_name, aux_value in auxiliary_losses.items():
        if aux_value is None:
            continue
        if not torch.is_tensor(aux_value):
            aux_value = torch.tensor(float(aux_value), device=loss.device, dtype=loss.dtype)
        if aux_value.ndim > 0:
            aux_value = aux_value.mean()
        aux_value = aux_value.to(device=loss.device, dtype=loss.dtype)
        if not torch.isfinite(aux_value.detach()):
            aux_value = torch.zeros((), device=loss.device, dtype=loss.dtype)
        loss = loss + aux_value
        loss_dict[aux_name] = aux_value
    for stat_name, stat_value in auxiliary_stats.items():
        if stat_value is None:
            continue
        if not torch.is_tensor(stat_value):
            stat_value = torch.tensor(float(stat_value), device=loss.device, dtype=loss.dtype)
        if stat_value.ndim > 0:
            stat_value = stat_value.float().mean()
        stat_value = stat_value.to(device=loss.device, dtype=loss.dtype)
        if not torch.isfinite(stat_value.detach()):
            stat_value = torch.zeros((), device=loss.device, dtype=loss.dtype)
        loss_dict[stat_name] = stat_value.detach()
    loss_dict["loss"] = loss
    return loss, loss_dict


def _merge_external_auxiliary_outputs(
    payload: dict[str, Any] | None,
    loss: torch.Tensor,
    loss_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not payload:
        loss_dict["loss"] = loss
        return loss, loss_dict
    auxiliary_losses = payload.get("auxiliary_losses") or {}
    auxiliary_stats = payload.get("auxiliary_stats") or {}
    for aux_name, aux_value in auxiliary_losses.items():
        if aux_value is None:
            continue
        if not torch.is_tensor(aux_value):
            aux_value = torch.tensor(float(aux_value), device=loss.device, dtype=loss.dtype)
        if aux_value.ndim > 0:
            aux_value = aux_value.mean()
        aux_value = aux_value.to(device=loss.device, dtype=loss.dtype)
        if not torch.isfinite(aux_value.detach()):
            aux_value = torch.zeros((), device=loss.device, dtype=loss.dtype)
        loss = loss + aux_value
        loss_dict[aux_name] = aux_value
    for stat_name, stat_value in auxiliary_stats.items():
        if stat_value is None:
            continue
        if not torch.is_tensor(stat_value):
            stat_value = torch.tensor(float(stat_value), device=loss.device, dtype=loss.dtype)
        if stat_value.ndim > 0:
            stat_value = stat_value.float().mean()
        stat_value = stat_value.to(device=loss.device, dtype=loss.dtype)
        if not torch.isfinite(stat_value.detach()):
            stat_value = torch.zeros((), device=loss.device, dtype=loss.dtype)
        loss_dict[stat_name] = stat_value.detach()
    loss_dict["loss"] = loss
    return loss, loss_dict


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            if parameter is None:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def make_phase_masked_view(
    image: torch.Tensor,
    phase_mask: torch.Tensor | None,
    probability: float,
    mode: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Mask one semantic CECT phase per selected sample for PETAL-V0 training."""
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0:
        return image, phase_mask, torch.zeros((), device=image.device)
    if image.ndim != 4:
        raise ValueError(f"Expected image shaped [B,C,H,W], got {tuple(image.shape)}")

    batch_size, channels = image.shape[:2]
    selected = torch.rand(batch_size, device=image.device) < probability
    selected_count = selected.float().sum()
    if float(selected_count.item()) <= 0.0:
        return image, phase_mask, selected_count

    dropped_phase = torch.randint(low=0, high=channels, size=(batch_size,), device=image.device)
    masked_image = image.clone()

    if mode == "zero":
        masked_image[selected, dropped_phase[selected]] = 0.0
    elif mode == "batch_mean":
        fill = image.mean(dim=(0, 2, 3), keepdim=False)
        masked_image[selected, dropped_phase[selected]] = fill[dropped_phase[selected]].view(-1, 1, 1)
    else:
        raise ValueError(f"Unsupported phase mask mode: {mode}")

    masked_phase_mask = None
    if phase_mask is not None:
        masked_phase_mask = phase_mask.clone()
        if masked_phase_mask.ndim == 1:
            masked_phase_mask = masked_phase_mask.unsqueeze(0).expand(batch_size, -1).clone()
        masked_phase_mask[selected, dropped_phase[selected]] = 0.0

    return masked_image, masked_phase_mask, selected_count


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    amp_dtype: str = "bf16",
    robust_config: Any | None = None,
    phase_mask_consistency_weight: float = 0.0,
    phase_mask_probability: float = 0.0,
    phase_mask_mode: str = "zero",
    phase_mask_temperature: float = 2.0,
    scar_config: Any | None = None,
    scar_domain_head: nn.Module | None = None,
    high_specificity_config: Any | None = None,
    paper_method_adapter: nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    if scar_domain_head is not None:
        scar_domain_head.train()
    if paper_method_adapter is not None:
        paper_method_adapter.train()
    running: dict[str, torch.Tensor] = {}
    steps = 0
    skipped_nonfinite_loss_steps = 0
    skipped_nonfinite_grad_steps = 0
    robust_enabled = bool(getattr(robust_config, "enabled", False))
    scar_enabled = bool(getattr(scar_config, "enabled", False))
    high_specificity_enabled = bool(getattr(high_specificity_config, "enabled", False))
    phase_mask_consistency_weight = max(float(phase_mask_consistency_weight), 0.0)
    phase_mask_probability = min(max(float(phase_mask_probability), 0.0), 1.0)
    phase_mask_temperature = max(float(phase_mask_temperature), 1e-6)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        context_image = batch.get("context_image")
        if context_image is not None:
            context_image = context_image.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        phase_mask = batch.get("phase_mask")
        if phase_mask is not None:
            phase_mask = phase_mask.to(device, non_blocking=True)
        hard_negative_mask = batch.get("hard_negative_mask")
        if hard_negative_mask is not None:
            hard_negative_mask = hard_negative_mask.to(device, non_blocking=True)
        hard_negative_weight = batch.get("hard_negative_weight")
        if hard_negative_weight is not None:
            hard_negative_weight = hard_negative_weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_dtype):
            output = _forward_model(
                model,
                image,
                phase_mask=phase_mask,
                context_image=context_image,
                target=target,
                hard_negative_mask=hard_negative_mask,
                hard_negative_weight=hard_negative_weight,
            )
            loss_dict = _compute_loss(output, target, criterion, epoch)
            clean_loss = loss_dict["loss"]
            loss = clean_loss

            phase_mask_loss = torch.zeros((), device=loss.device)
            phase_mask_selected = torch.zeros((), device=loss.device)
            if phase_mask_consistency_weight > 0.0 and phase_mask_probability > 0.0:
                masked_image, masked_phase_mask, phase_mask_selected = make_phase_masked_view(
                    image=image,
                    phase_mask=phase_mask,
                    probability=phase_mask_probability,
                    mode=phase_mask_mode,
                )
                if float(phase_mask_selected.item()) > 0.0:
                    masked_output = _forward_model(
                        model,
                        masked_image,
                        phase_mask=masked_phase_mask,
                        context_image=context_image,
                    )
                    student_logits = output.get("logits")
                    masked_logits = masked_output.get("logits")
                    if student_logits is None or masked_logits is None:
                        raise ValueError("Phase-masked consistency requires student logits.")
                    full_prob = torch.nn.functional.softmax(
                        student_logits.detach().float() / phase_mask_temperature,
                        dim=-1,
                    )
                    masked_log_prob = torch.nn.functional.log_softmax(
                        masked_logits.float() / phase_mask_temperature,
                        dim=-1,
                    )
                    phase_mask_loss = torch.nn.functional.kl_div(
                        masked_log_prob,
                        full_prob,
                        reduction="batchmean",
                    ) * (phase_mask_temperature * phase_mask_temperature)
                    if not torch.isfinite(phase_mask_loss.detach()):
                        phase_mask_loss = torch.zeros((), device=loss.device)
                    loss = loss + phase_mask_consistency_weight * phase_mask_loss

            if robust_enabled:
                augmented_supervised_weight = float(getattr(robust_config, "augmented_supervised_weight", 0.0))
                feature_weight = float(getattr(robust_config, "feature_consistency_weight", 0.0))
                logit_weight = float(getattr(robust_config, "logit_consistency_weight", 0.0))
                augmented_image, augmented_phase_mask = make_robust_view(image, phase_mask, robust_config)
                augmented_output = _forward_model(
                    model,
                    augmented_image,
                    phase_mask=augmented_phase_mask,
                    context_image=context_image,
                )
                augmented_loss_dict = _compute_loss(augmented_output, target, criterion, epoch)

                feature_loss = torch.zeros((), device=loss.device)
                logit_loss = torch.zeros((), device=loss.device)
                if feature_weight > 0.0:
                    feature_loss = feature_consistency_loss(output.get("feature"), augmented_output.get("feature"))
                    if not torch.isfinite(feature_loss.detach()):
                        feature_loss = torch.zeros((), device=loss.device)
                if logit_weight > 0.0:
                    logit_loss = logit_consistency_loss(output.get("logits"), augmented_output.get("logits"))
                    if not torch.isfinite(logit_loss.detach()):
                        logit_loss = torch.zeros((), device=loss.device)

                loss = (
                    clean_loss
                    + augmented_supervised_weight * augmented_loss_dict["loss"]
                    + feature_weight * feature_loss
                    + logit_weight * logit_loss
                )
                loss_dict = {
                    "loss": loss,
                    "loss_nll": loss_dict["loss_nll"]
                    + augmented_supervised_weight * augmented_loss_dict["loss_nll"].detach(),
                    "loss_kl": loss_dict["loss_kl"]
                    + augmented_supervised_weight * augmented_loss_dict["loss_kl"].detach(),
                    "loss_clean": clean_loss,
                    "loss_augmented": augmented_loss_dict["loss"],
                    "loss_feature_consistency": feature_loss,
                    "loss_logit_consistency": logit_loss,
                    "loss_phase_mask_consistency": phase_mask_loss,
                    "phase_mask_applied_frac": phase_mask_selected / max(float(image.shape[0]), 1.0),
                }
            else:
                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_clean": clean_loss,
                    "loss_phase_mask_consistency": phase_mask_loss,
                    "phase_mask_applied_frac": phase_mask_selected / max(float(image.shape[0]), 1.0),
                }

            hs_loss = torch.zeros((), device=loss.device)
            if high_specificity_enabled:
                hs_weight = max(float(getattr(high_specificity_config, "hard_positive_weight", 0.0)), 0.0) + max(
                    float(getattr(high_specificity_config, "negative_tail_weight", 0.0)),
                    0.0,
                )
                if hs_weight > 0.0:
                    hs_loss = negative_tail_loss(
                        output.get("logits"),
                        target,
                        positive_margin=float(getattr(high_specificity_config, "positive_margin", 0.65)),
                        negative_margin=float(getattr(high_specificity_config, "negative_margin", 0.25)),
                    )
                    if not torch.isfinite(hs_loss.detach()):
                        hs_loss = torch.zeros((), device=loss.device)
                    loss = loss + hs_weight * hs_loss
                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_high_specificity_tail": hs_loss,
                }

            if scar_enabled:
                domain_loss = torch.zeros((), device=loss.device)
                supcon_loss = torch.zeros((), device=loss.device)
                phase_comp_loss = torch.zeros((), device=loss.device)
                phase_comp_stats = {"views": 0.0, "logit": 0.0, "feature": 0.0}
                feature = output.get("feature")
                domain_weight = max(float(getattr(scar_config, "domain_adversarial_weight", 0.0)), 0.0)
                num_domains = max(int(getattr(scar_config, "pseudo_domain_count", 4)), 1)
                if domain_weight > 0.0 and scar_domain_head is not None and feature is not None and num_domains > 1:
                    lambda_value = domain_lambda_for_epoch(
                        epoch=epoch,
                        warmup_epochs=int(getattr(scar_config, "domain_warmup_epochs", 10)),
                    )
                    domain_id = build_style_pseudo_domain_tensor(image, phase_mask, num_domains=num_domains)
                    domain_logits = scar_domain_head(feature, lambda_value=lambda_value)
                    domain_loss = torch.nn.functional.cross_entropy(domain_logits.float(), domain_id)
                    if not torch.isfinite(domain_loss.detach()):
                        domain_loss = torch.zeros((), device=loss.device)
                    loss = loss + domain_weight * lambda_value * domain_loss

                supcon_weight = max(float(getattr(scar_config, "supcon_weight", 0.0)), 0.0)
                if supcon_weight > 0.0:
                    supcon_loss = supervised_contrastive_loss(
                        feature=feature,
                        target=target,
                        temperature=float(getattr(scar_config, "supcon_temperature", 0.2)),
                    )
                    if not torch.isfinite(supcon_loss.detach()):
                        supcon_loss = torch.zeros((), device=loss.device)
                    loss = loss + supcon_weight * supcon_loss

                phase_comp_warmup = max(int(getattr(scar_config, "phase_comp_warmup_epochs", 0)), 0)
                phase_comp_weight = max(float(getattr(scar_config, "phase_comp_weight", 0.0)), 0.0)
                phase_comp_feature_weight = max(float(getattr(scar_config, "phase_comp_feature_weight", 0.0)), 0.0)
                if epoch + 1 >= phase_comp_warmup and (phase_comp_weight > 0.0 or phase_comp_feature_weight > 0.0):
                    phase_comp_loss, phase_comp_stats = phase_compositional_consistency_loss(
                        model=model,
                        image=image,
                        context_image=context_image,
                        phase_mask=phase_mask,
                        clean_output=output,
                        subset_names=list(getattr(scar_config, "phase_comp_subsets", [])),
                        max_views=int(getattr(scar_config, "phase_comp_max_views", 1)),
                        epoch=epoch,
                        logit_weight=phase_comp_weight,
                        feature_weight=phase_comp_feature_weight,
                        temperature=float(getattr(scar_config, "phase_comp_temperature", 4.0)),
                    )
                    if not torch.isfinite(phase_comp_loss.detach()):
                        phase_comp_loss = torch.zeros((), device=loss.device)
                    loss = loss + phase_comp_loss

                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_scar_domain": domain_loss,
                    "loss_scar_supcon": supcon_loss,
                    "loss_scar_phase_comp": phase_comp_loss,
                    "scar_phase_comp_views": torch.tensor(
                        phase_comp_stats["views"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                    "scar_phase_comp_logit": torch.tensor(
                        phase_comp_stats["logit"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                    "scar_phase_comp_feature": torch.tensor(
                        phase_comp_stats["feature"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                }

            if paper_method_adapter is not None:
                payload = paper_method_adapter.compute_loss(batch=batch, outputs=output, state={"epoch": epoch})
                loss, loss_dict = _merge_external_auxiliary_outputs(payload, loss, loss_dict)
            loss, loss_dict = _merge_model_auxiliary_outputs(output, loss, loss_dict)

        if not torch.isfinite(loss.detach()):
            skipped_nonfinite_loss_steps += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), grad_clip)
            if not torch.isfinite(grad_norm):
                skipped_nonfinite_grad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
        optimizer.step()

        for key, value in loss_dict.items():
            detached = value.detach()
            running[key] = running.get(key, torch.zeros((), device=detached.device)) + detached
        steps += 1

    stats = {key: float((value / max(steps, 1)).cpu()) for key, value in running.items()}
    stats.setdefault("loss", float("nan"))
    stats.setdefault("loss_nll", float("nan"))
    stats.setdefault("loss_kl", float("nan"))
    stats["loss_nonfinite_steps"] = float(skipped_nonfinite_loss_steps)
    stats["grad_nonfinite_steps"] = float(skipped_nonfinite_grad_steps)
    return stats


def train_one_epoch_with_teacher(
    model: nn.Module,
    teacher_model: nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    amp_dtype: str = "bf16",
    robust_config: Any | None = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 2.0,
    phase_mask_consistency_weight: float = 0.0,
    phase_mask_probability: float = 0.0,
    phase_mask_mode: str = "zero",
    scar_config: Any | None = None,
    scar_domain_head: nn.Module | None = None,
    high_specificity_config: Any | None = None,
    paper_method_adapter: nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    if scar_domain_head is not None:
        scar_domain_head.train()
    if paper_method_adapter is not None:
        paper_method_adapter.train()
    teacher_model.eval()
    running: dict[str, torch.Tensor] = {}
    steps = 0
    skipped_nonfinite_loss_steps = 0
    skipped_nonfinite_grad_steps = 0
    robust_enabled = bool(getattr(robust_config, "enabled", False))
    scar_enabled = bool(getattr(scar_config, "enabled", False))
    high_specificity_enabled = bool(getattr(high_specificity_config, "enabled", False))
    distill_alpha = max(float(distill_alpha), 0.0)
    distill_temperature = max(float(distill_temperature), 1e-6)
    phase_mask_consistency_weight = max(float(phase_mask_consistency_weight), 0.0)
    phase_mask_probability = min(max(float(phase_mask_probability), 0.0), 1.0)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        context_image = batch.get("context_image")
        if context_image is not None:
            context_image = context_image.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        phase_mask = batch.get("phase_mask")
        if phase_mask is not None:
            phase_mask = phase_mask.to(device, non_blocking=True)
        hard_negative_mask = batch.get("hard_negative_mask")
        if hard_negative_mask is not None:
            hard_negative_mask = hard_negative_mask.to(device, non_blocking=True)
        hard_negative_weight = batch.get("hard_negative_weight")
        if hard_negative_weight is not None:
            hard_negative_weight = hard_negative_weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_dtype):
            output = _forward_model(
                model,
                image,
                phase_mask=phase_mask,
                context_image=context_image,
                target=target,
                hard_negative_mask=hard_negative_mask,
                hard_negative_weight=hard_negative_weight,
            )
            loss_dict = _compute_loss(output, target, criterion, epoch)
            clean_loss = loss_dict["loss"]
            loss = clean_loss

            teacher_loss = torch.zeros((), device=loss.device)
            if distill_alpha > 0.0:
                with torch.no_grad():
                    teacher_output = _forward_model(
                        teacher_model,
                        image,
                        phase_mask=phase_mask,
                        context_image=context_image,
                    )
                student_logits = output.get("logits")
                teacher_logits = teacher_output.get("logits")
                if student_logits is None or teacher_logits is None:
                    raise ValueError("Teacher distillation requires student and teacher logits.")
                student_log_prob = torch.nn.functional.log_softmax(
                    student_logits.float() / distill_temperature,
                    dim=-1,
                )
                teacher_prob = torch.nn.functional.softmax(
                    teacher_logits.float() / distill_temperature,
                    dim=-1,
                )
                teacher_loss = torch.nn.functional.kl_div(
                    student_log_prob,
                    teacher_prob,
                    reduction="batchmean",
                ) * (distill_temperature * distill_temperature)
                if not torch.isfinite(teacher_loss.detach()):
                    teacher_loss = torch.zeros((), device=loss.device)
                loss = loss + distill_alpha * teacher_loss

            phase_mask_loss = torch.zeros((), device=loss.device)
            phase_mask_selected = torch.zeros((), device=loss.device)
            if phase_mask_consistency_weight > 0.0 and phase_mask_probability > 0.0:
                masked_image, masked_phase_mask, phase_mask_selected = make_phase_masked_view(
                    image=image,
                    phase_mask=phase_mask,
                    probability=phase_mask_probability,
                    mode=phase_mask_mode,
                )
                if float(phase_mask_selected.item()) > 0.0:
                    masked_output = _forward_model(
                        model,
                        masked_image,
                        phase_mask=masked_phase_mask,
                        context_image=context_image,
                    )
                    student_logits = output.get("logits")
                    masked_logits = masked_output.get("logits")
                    if student_logits is None or masked_logits is None:
                        raise ValueError("Phase-masked consistency requires student logits.")
                    full_prob = torch.nn.functional.softmax(
                        student_logits.detach().float() / distill_temperature,
                        dim=-1,
                    )
                    masked_log_prob = torch.nn.functional.log_softmax(
                        masked_logits.float() / distill_temperature,
                        dim=-1,
                    )
                    phase_mask_loss = torch.nn.functional.kl_div(
                        masked_log_prob,
                        full_prob,
                        reduction="batchmean",
                    ) * (distill_temperature * distill_temperature)
                    if not torch.isfinite(phase_mask_loss.detach()):
                        phase_mask_loss = torch.zeros((), device=loss.device)
                    loss = loss + phase_mask_consistency_weight * phase_mask_loss

            if robust_enabled:
                augmented_supervised_weight = float(getattr(robust_config, "augmented_supervised_weight", 0.0))
                feature_weight = float(getattr(robust_config, "feature_consistency_weight", 0.0))
                logit_weight = float(getattr(robust_config, "logit_consistency_weight", 0.0))
                augmented_image, augmented_phase_mask = make_robust_view(image, phase_mask, robust_config)
                augmented_output = _forward_model(
                    model,
                    augmented_image,
                    phase_mask=augmented_phase_mask,
                    context_image=context_image,
                )
                augmented_loss_dict = _compute_loss(augmented_output, target, criterion, epoch)

                feature_loss = torch.zeros((), device=loss.device)
                logit_loss = torch.zeros((), device=loss.device)
                if feature_weight > 0.0:
                    feature_loss = feature_consistency_loss(output.get("feature"), augmented_output.get("feature"))
                    if not torch.isfinite(feature_loss.detach()):
                        feature_loss = torch.zeros((), device=loss.device)
                if logit_weight > 0.0:
                    logit_loss = logit_consistency_loss(output.get("logits"), augmented_output.get("logits"))
                    if not torch.isfinite(logit_loss.detach()):
                        logit_loss = torch.zeros((), device=loss.device)

                loss = (
                    loss
                    + augmented_supervised_weight * augmented_loss_dict["loss"]
                    + feature_weight * feature_loss
                    + logit_weight * logit_loss
                )
                loss_dict = {
                    "loss": loss,
                    "loss_nll": loss_dict["loss_nll"]
                    + augmented_supervised_weight * augmented_loss_dict["loss_nll"].detach(),
                    "loss_kl": loss_dict["loss_kl"]
                    + augmented_supervised_weight * augmented_loss_dict["loss_kl"].detach(),
                    "loss_clean": clean_loss,
                    "loss_augmented": augmented_loss_dict["loss"],
                    "loss_feature_consistency": feature_loss,
                    "loss_logit_consistency": logit_loss,
                    "loss_teacher_distill": teacher_loss,
                    "loss_phase_mask_consistency": phase_mask_loss,
                    "phase_mask_applied_frac": phase_mask_selected / max(float(image.shape[0]), 1.0),
                }
            else:
                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_clean": clean_loss,
                    "loss_teacher_distill": teacher_loss,
                    "loss_phase_mask_consistency": phase_mask_loss,
                    "phase_mask_applied_frac": phase_mask_selected / max(float(image.shape[0]), 1.0),
                }

            hs_loss = torch.zeros((), device=loss.device)
            if high_specificity_enabled:
                hs_weight = max(float(getattr(high_specificity_config, "hard_positive_weight", 0.0)), 0.0) + max(
                    float(getattr(high_specificity_config, "negative_tail_weight", 0.0)),
                    0.0,
                )
                if hs_weight > 0.0:
                    hs_loss = negative_tail_loss(
                        output.get("logits"),
                        target,
                        positive_margin=float(getattr(high_specificity_config, "positive_margin", 0.65)),
                        negative_margin=float(getattr(high_specificity_config, "negative_margin", 0.25)),
                    )
                    if not torch.isfinite(hs_loss.detach()):
                        hs_loss = torch.zeros((), device=loss.device)
                    loss = loss + hs_weight * hs_loss
                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_high_specificity_tail": hs_loss,
                }

            if scar_enabled:
                domain_loss = torch.zeros((), device=loss.device)
                supcon_loss = torch.zeros((), device=loss.device)
                phase_comp_loss = torch.zeros((), device=loss.device)
                phase_comp_stats = {"views": 0.0, "logit": 0.0, "feature": 0.0}
                feature = output.get("feature")
                domain_weight = max(float(getattr(scar_config, "domain_adversarial_weight", 0.0)), 0.0)
                num_domains = max(int(getattr(scar_config, "pseudo_domain_count", 4)), 1)
                if domain_weight > 0.0 and scar_domain_head is not None and feature is not None and num_domains > 1:
                    lambda_value = domain_lambda_for_epoch(
                        epoch=epoch,
                        warmup_epochs=int(getattr(scar_config, "domain_warmup_epochs", 10)),
                    )
                    domain_id = build_style_pseudo_domain_tensor(image, phase_mask, num_domains=num_domains)
                    domain_logits = scar_domain_head(feature, lambda_value=lambda_value)
                    domain_loss = torch.nn.functional.cross_entropy(domain_logits.float(), domain_id)
                    if not torch.isfinite(domain_loss.detach()):
                        domain_loss = torch.zeros((), device=loss.device)
                    loss = loss + domain_weight * lambda_value * domain_loss

                supcon_weight = max(float(getattr(scar_config, "supcon_weight", 0.0)), 0.0)
                if supcon_weight > 0.0:
                    supcon_loss = supervised_contrastive_loss(
                        feature=feature,
                        target=target,
                        temperature=float(getattr(scar_config, "supcon_temperature", 0.2)),
                    )
                    if not torch.isfinite(supcon_loss.detach()):
                        supcon_loss = torch.zeros((), device=loss.device)
                    loss = loss + supcon_weight * supcon_loss

                phase_comp_warmup = max(int(getattr(scar_config, "phase_comp_warmup_epochs", 0)), 0)
                phase_comp_weight = max(float(getattr(scar_config, "phase_comp_weight", 0.0)), 0.0)
                phase_comp_feature_weight = max(float(getattr(scar_config, "phase_comp_feature_weight", 0.0)), 0.0)
                if epoch + 1 >= phase_comp_warmup and (phase_comp_weight > 0.0 or phase_comp_feature_weight > 0.0):
                    phase_comp_loss, phase_comp_stats = phase_compositional_consistency_loss(
                        model=model,
                        image=image,
                        context_image=context_image,
                        phase_mask=phase_mask,
                        clean_output=output,
                        subset_names=list(getattr(scar_config, "phase_comp_subsets", [])),
                        max_views=int(getattr(scar_config, "phase_comp_max_views", 1)),
                        epoch=epoch,
                        logit_weight=phase_comp_weight,
                        feature_weight=phase_comp_feature_weight,
                        temperature=float(getattr(scar_config, "phase_comp_temperature", 4.0)),
                    )
                    if not torch.isfinite(phase_comp_loss.detach()):
                        phase_comp_loss = torch.zeros((), device=loss.device)
                    loss = loss + phase_comp_loss

                loss_dict = {
                    **loss_dict,
                    "loss": loss,
                    "loss_scar_domain": domain_loss,
                    "loss_scar_supcon": supcon_loss,
                    "loss_scar_phase_comp": phase_comp_loss,
                    "scar_phase_comp_views": torch.tensor(
                        phase_comp_stats["views"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                    "scar_phase_comp_logit": torch.tensor(
                        phase_comp_stats["logit"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                    "scar_phase_comp_feature": torch.tensor(
                        phase_comp_stats["feature"],
                        dtype=loss.dtype,
                        device=loss.device,
                    ),
                }

            if paper_method_adapter is not None:
                payload = paper_method_adapter.compute_loss(batch=batch, outputs=output, state={"epoch": epoch})
                loss, loss_dict = _merge_external_auxiliary_outputs(payload, loss, loss_dict)
            loss, loss_dict = _merge_model_auxiliary_outputs(output, loss, loss_dict)

        if not torch.isfinite(loss.detach()):
            skipped_nonfinite_loss_steps += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), grad_clip)
            if not torch.isfinite(grad_norm):
                skipped_nonfinite_grad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
        optimizer.step()

        for key, value in loss_dict.items():
            detached = value.detach()
            running[key] = running.get(key, torch.zeros((), device=detached.device)) + detached
        steps += 1

    stats = {key: float((value / max(steps, 1)).cpu()) for key, value in running.items()}
    stats.setdefault("loss", float("nan"))
    stats.setdefault("loss_nll", float("nan"))
    stats.setdefault("loss_kl", float("nan"))
    stats["loss_nonfinite_steps"] = float(skipped_nonfinite_loss_steps)
    stats["grad_nonfinite_steps"] = float(skipped_nonfinite_grad_steps)
    return stats


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    amp_dtype: str = "bf16",
    paper_method_adapter: nn.Module | None = None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    if paper_method_adapter is not None:
        paper_method_adapter.eval()
    running: dict[str, torch.Tensor] = {}
    steps = 0
    records: list[dict[str, object]] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        context_image = batch.get("context_image")
        if context_image is not None:
            context_image = context_image.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        phase_mask = batch.get("phase_mask")
        if phase_mask is not None:
            phase_mask = phase_mask.to(device, non_blocking=True)

        with _amp_context(device, amp_dtype):
            output = _forward_model(model, image, phase_mask=phase_mask, context_image=context_image)
            loss_dict = _compute_loss(output, target, criterion, epoch)
            loss = loss_dict["loss"]
            if paper_method_adapter is not None:
                payload = paper_method_adapter.compute_loss(
                    batch=batch,
                    outputs=output,
                    state={"epoch": epoch, "mode": "eval"},
                )
                loss, loss_dict = _merge_external_auxiliary_outputs(payload, loss, loss_dict)
            loss, loss_dict = _merge_model_auxiliary_outputs(output, loss, loss_dict)

        prob = output["prob"]
        logits = output.get("logits")
        uncertainty = output["uncertainty"]
        evidence = output.get("evidence")
        alpha = output.get("alpha")
        if prob is None:
            raise ValueError("Model output must include probabilities.")

        prob_pos = prob[:, 1].detach().cpu().tolist()
        logit_pos = logits[:, 1].detach().cpu().tolist() if logits is not None and logits.shape[1] >= 2 else [None] * len(prob_pos)
        uncertainty_list = (
            uncertainty.squeeze(-1).detach().cpu().tolist() if uncertainty is not None else [None] * len(prob_pos)
        )
        evidence_strength_list = (
            evidence.sum(dim=-1).detach().cpu().tolist() if evidence is not None else [None] * len(prob_pos)
        )
        alpha_strength_list = (
            alpha.sum(dim=-1).detach().cpu().tolist() if alpha is not None else [None] * len(prob_pos)
        )

        for idx in range(len(prob_pos)):
            records.append(
                {
                    "patient_id": batch["patient_id"][idx],
                    "lesion_id": batch["lesion_id"][idx],
                    "slice_index": int(batch["slice_index"][idx]),
                    "label": int(batch["label"][idx]),
                    "prob_pos": float(prob_pos[idx]),
                    "logit_pos": None if logit_pos[idx] is None else float(logit_pos[idx]),
                    "uncertainty": None if uncertainty_list[idx] is None else float(uncertainty_list[idx]),
                    "evidence_strength": None
                    if evidence_strength_list[idx] is None
                    else float(evidence_strength_list[idx]),
                    "alpha_strength": None if alpha_strength_list[idx] is None else float(alpha_strength_list[idx]),
                }
            )

        for key, value in loss_dict.items():
            detached = value.detach()
            running[key] = running.get(key, torch.zeros((), device=detached.device)) + detached
        steps += 1

    return {key: float((value / max(steps, 1)).cpu()) for key, value in running.items()}, records
