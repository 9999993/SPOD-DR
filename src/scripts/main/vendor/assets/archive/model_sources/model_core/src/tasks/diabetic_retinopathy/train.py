from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from calibmamba.utils.io import ensure_dir
from calibmamba.utils.seeding import seed_everything
from tasks.diabetic_retinopathy.dataset import DiabeticRetinopathyDataset
from tasks.diabetic_retinopathy.losses import BinaryOrdinalLoss, build_coral_targets
from tasks.diabetic_retinopathy.metrics import summarize_predictions
from tasks.diabetic_retinopathy.models import build_dr_model
from tasks.diabetic_retinopathy.thresholds import select_threshold_for_min_sensitivity, select_youden_threshold


def _resolve_memory_format(*, device: torch.device, channels_last: bool) -> torch.memory_format:
    if bool(channels_last) and device.type == "cuda":
        return torch.channels_last
    return torch.contiguous_format


def _resolve_amp_dtype(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {value}")


def _to_numpy_float(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: str):
    if not amp_enabled or device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_resolve_amp_dtype(amp_dtype))


def _dataloader_kwargs(
    *,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory) and device.type == "cuda",
    }
    if int(num_workers) > 0:
        resolved["persistent_workers"] = bool(persistent_workers)
        resolved["prefetch_factor"] = max(int(prefetch_factor), 1)
    return resolved


def _build_loader_seed_state(*, seed: int, num_workers: int) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        seed_everything(int(worker_seed))

    state: dict[str, Any] = {"generator": generator}
    if int(num_workers) > 0:
        state["worker_init_fn"] = _seed_worker
    return state


def _should_run_validation(*, epoch: int, total_epochs: int, validation_interval: int) -> bool:
    resolved_interval = max(int(validation_interval), 1)
    return int(epoch) == int(total_epochs) or int(epoch) % resolved_interval == 0


def _prediction_save_path(
    out_dir: Path,
    *,
    epoch: int,
    prediction_save_policy: str,
    is_best: bool,
) -> Path | None:
    normalized = str(prediction_save_policy).strip().lower()
    if normalized == "all":
        return out_dir / f"val_predictions_epoch_{int(epoch):03d}.csv"
    if normalized == "best":
        return (out_dir / "val_predictions_best.csv") if is_best else None
    if normalized == "none":
        return None
    raise ValueError(f"Unsupported prediction_save_policy: {prediction_save_policy}")


def _load_yaml_config(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runner_parity_mode(*, method_cfg: dict[str, Any], loss_cfg: dict[str, Any]) -> bool:
    return bool(method_cfg.get("runner_parity_mode", False) or loss_cfg.get("lar_orrc_parity_mode", False))


def _resolve_runtime_controls(
    *,
    train_cfg: dict[str, Any],
    device: torch.device,
    runner_parity_mode: bool,
) -> dict[str, Any]:
    num_workers = int(train_cfg.get("num_workers", 0))
    runtime = {
        "runner_parity_mode": bool(runner_parity_mode),
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": bool(train_cfg.get("persistent_workers", num_workers > 0)),
        "prefetch_factor": int(train_cfg.get("prefetch_factor", 2)),
        "amp_enabled": bool(train_cfg.get("amp", True)) and device.type == "cuda",
        "amp_dtype": str(train_cfg.get("amp_dtype", "float16")),
        "cudnn_benchmark": bool(train_cfg.get("cudnn_benchmark", True)),
        "matmul_precision": str(train_cfg.get("matmul_precision", "high")),
        "allow_tf32": True,
        "use_deterministic_algorithms": False,
    }
    if runner_parity_mode:
        runtime.update(
            {
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "amp_enabled": False,
                "cudnn_benchmark": False,
                "matmul_precision": "highest",
                "allow_tf32": False,
                "use_deterministic_algorithms": True,
            }
        )
    return runtime


@contextmanager
def _runtime_backend_context(
    *,
    device: torch.device,
    cudnn_benchmark: bool,
    matmul_precision: str,
    allow_tf32: bool,
    use_deterministic_algorithms: bool,
):
    old_matmul_precision = torch.get_float32_matmul_precision()
    old_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    old_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_cudnn_benchmark = None
    old_cudnn_allow_tf32 = None
    old_cuda_matmul_allow_tf32 = None
    if hasattr(torch.backends, "cudnn"):
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        old_cudnn_allow_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        old_cuda_matmul_allow_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    try:
        if use_deterministic_algorithms:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if device.type == "cuda":
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
                if old_cudnn_allow_tf32 is not None:
                    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
            if old_cuda_matmul_allow_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.set_float32_matmul_precision(str(matmul_precision))
        torch.use_deterministic_algorithms(bool(use_deterministic_algorithms))
        yield
    finally:
        torch.use_deterministic_algorithms(bool(old_deterministic_algorithms))
        torch.set_float32_matmul_precision(old_matmul_precision)
        if hasattr(torch.backends, "cudnn") and old_cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = bool(old_cudnn_benchmark)
            if old_cudnn_allow_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = bool(old_cudnn_allow_tf32)
        if old_cuda_matmul_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(old_cuda_matmul_allow_tf32)
        if old_cublas_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas_workspace


def _compute_pos_weight(dataset: DiabeticRetinopathyDataset) -> tuple[torch.Tensor, torch.Tensor]:
    binary = np.asarray([row.label for row in dataset.rows], dtype=np.int64)
    grades = np.asarray([row.grade for row in dataset.rows], dtype=np.int64)
    negatives = int((binary == 0).sum())
    positives = int((binary == 1).sum())
    pos_weight = torch.tensor(float(negatives / max(positives, 1)), dtype=torch.float32)

    coral_targets = np.stack([(grades > threshold).astype(np.int64) for threshold in range(4)], axis=1)
    ordinal_weights = []
    for threshold in range(coral_targets.shape[1]):
        threshold_pos = int(coral_targets[:, threshold].sum())
        threshold_neg = int(coral_targets.shape[0] - threshold_pos)
        ordinal_weights.append(float(threshold_neg / max(threshold_pos, 1)))
    return pos_weight, torch.tensor(ordinal_weights, dtype=torch.float32)


def _ordinal_predictions(ordinal_logits: np.ndarray) -> np.ndarray:
    ordinal_prob = 1.0 / (1.0 + np.exp(-np.asarray(ordinal_logits, dtype=np.float64)))
    return (ordinal_prob >= 0.5).sum(axis=1).astype(np.int64)


def _save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _criterion_kwargs_from_output(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    kwargs: dict[str, torch.Tensor] = {}
    for source_key, target_key in (
        ("local_binary_logits", "local_binary_logits"),
        ("global_binary_logits", "global_binary_logits"),
        ("severity_score", "severity_score"),
        ("stability_target_score", "stability_target_score"),
        ("prototype_distances", "prototype_distances"),
        ("prototype_order_margin_loss", "prototype_order_margin_loss"),
        ("grade_probs", "grade_probs"),
        ("local_grade_probs", "local_grade_probs"),
        ("global_grade_probs", "global_grade_probs"),
        ("screening_prob", "screening_prob"),
        ("cumulative_risk", "cumulative_risk"),
        ("risk_order_score", "risk_order_score"),
        ("clinical_risk_embedding", "clinical_risk_embedding"),
        ("evidence_diversity_loss", "evidence_diversity_loss"),
        ("progression_score", "progression_score"),
        ("decision_maps", "decision_maps"),
    ):
        if source_key in output:
            kwargs[target_key] = output[source_key]
    return kwargs


def _score_from_output(output: dict[str, torch.Tensor]) -> torch.Tensor:
    severity_score = output.get("severity_score")
    if severity_score is not None:
        return severity_score
    return output["ordinal_logits"].float().sum(dim=1)


def _imagenet_appearance_perturbation(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    image_01 = (image * std + mean).clamp(0.0, 1.0)
    batch_size = image.shape[0]
    brightness = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.85, 1.15)
    contrast = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.85, 1.15)
    gamma = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.85, 1.15)
    center = image_01.mean(dim=(2, 3), keepdim=True)
    shifted = ((image_01 - center) * contrast + center).clamp(0.0, 1.0)
    shifted = (shifted * brightness).clamp(0.0, 1.0).pow(gamma)
    return (shifted - mean) / std


def _transport_appearance_perturbation(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    image_01 = (image * std + mean).clamp(0.0, 1.0)
    batch_size = image.shape[0]
    brightness = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.80, 1.20)
    contrast = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.80, 1.20)
    gamma = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.80, 1.20)
    color_scale = torch.empty(batch_size, 3, 1, 1, device=image.device, dtype=image.dtype).uniform_(0.90, 1.10)
    center = image_01.mean(dim=(2, 3), keepdim=True)
    shifted = ((image_01 - center) * contrast + center).clamp(0.0, 1.0)
    shifted = (shifted * brightness).clamp(0.0, 1.0).pow(gamma)
    shifted = (shifted * color_scale).clamp(0.0, 1.0)
    quality = torch.empty(batch_size, 1, 1, 1, device=image.device, dtype=image.dtype).uniform_(32.0, 96.0)
    shifted = torch.round(shifted * quality) / quality.clamp_min(1.0)
    return (shifted - mean) / std


def _prediction_extra_columns(output: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key in (
        "local_prob_pos",
        "global_prob_pos",
        "fusion_prob_pos",
        "evidence_prob_pos",
        "evidence_disagreement",
        "local_global_abs_diff",
        "local_global_cosine",
        "severity_score",
        "evidence_token_norm_mean",
        "evidence_token_norm_std",
        "evidence_token_variance",
        "query_attention_entropy",
        "query_gate_entropy",
        "query_output_variance",
        "prototype_distance_grade0",
        "prototype_distance_grade1",
        "prototype_distance_grade2",
        "prototype_distance_grade3",
        "prototype_distance_grade4",
        "fused_mu",
        "fused_sigma",
        "local_mu",
        "local_sigma",
        "global_mu",
        "global_sigma",
        "local_weight",
        "global_weight",
        "within_variance",
        "disagreement_variance",
        "total_variance",
        "screening_prob",
        "screening_probability",
        "cumulative_risk",
        "expected_grade",
        "severity_distribution_variance",
        "risk_uncertainty",
        "risk_alignment_error",
        "risk_alignment_delta",
        "risk_original",
        "risk_transport",
        "delta_risk",
        "embedding_distance",
        "screen_probability_delta",
        "cra_signed_delta",
        "cra_positive_risk_weight",
        "cra_bias",
        "cra_expected_grade_weight",
        "cra_variance_weight",
        "cra_disagreement_weight",
        "cra_max_delta",
        "evidence_state_mean",
        "evidence_state_std",
        "query_attention_overlap_mean",
    ):
        value = output.get(key)
        if value is not None and value.ndim >= 1:
            columns[key] = float(value[index].detach().float().cpu().item())
    risk_embedding = output.get("clinical_risk_embedding")
    if risk_embedding is not None and risk_embedding.ndim == 2:
        columns["clinical_risk_embedding"] = json.dumps(
            risk_embedding[index].detach().float().cpu().tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    evidence_state = output.get("evidence_state")
    if evidence_state is not None and evidence_state.ndim == 2:
        columns["evidence_state"] = json.dumps(
            evidence_state[index].detach().float().cpu().tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for evidence_index in range(int(evidence_state.shape[1])):
            columns[f"evidence_state_{evidence_index}"] = float(
                evidence_state[index, evidence_index].detach().float().cpu().item()
            )
    for key in ("local_binary_logits", "global_binary_logits"):
        value = output.get(key)
        if value is not None and value.ndim >= 1:
            columns[key] = float(value[index].detach().float().cpu().item())
    grade_probs = output.get("grade_probs")
    if grade_probs is not None and grade_probs.ndim == 2:
        for grade_index in range(min(grade_probs.shape[1], 5)):
            columns[f"grade_prob_{grade_index}"] = float(grade_probs[index, grade_index].detach().float().cpu().item())
    distances = output.get("prototype_distances")
    if distances is not None and distances.ndim == 2:
        for grade_index in range(min(distances.shape[1], 5)):
            columns[f"prototype_distance_grade{grade_index}"] = float(
                distances[index, grade_index].detach().float().cpu().item()
            )
    return columns


def _train_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BinaryOrdinalLoss,
    device: torch.device,
    pos_weight: torch.Tensor,
    ordinal_pos_weight: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: str,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    accumulation_steps: int,
    memory_format: torch.memory_format,
    transport_enabled: bool,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_components: dict[str, float] = {}
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step_index, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True).contiguous(memory_format=memory_format)
        label = batch["label"].to(device, non_blocking=True)
        grade = batch["grade"].to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled, amp_dtype):
            output = model(image)
            criterion_kwargs = _criterion_kwargs_from_output(output)
            if float(getattr(criterion, "stability_weight", 0.0)) > 0:
                perturbed = _imagenet_appearance_perturbation(image)
                perturbed_output = model(perturbed)
                criterion_kwargs["stability_target_score"] = _score_from_output(perturbed_output)
            if float(getattr(criterion, "risk_invariance_weight", 0.0)) > 0:
                perturbed = _imagenet_appearance_perturbation(image)
                perturbed_output = model(perturbed)
                if "clinical_risk_embedding" in perturbed_output:
                    criterion_kwargs["stability_target_embedding"] = perturbed_output["clinical_risk_embedding"]
            if transport_enabled:
                transport_output = model(_transport_appearance_perturbation(image))
                if "clinical_risk_embedding" in transport_output:
                    criterion_kwargs["transport_risk_embedding"] = transport_output["clinical_risk_embedding"]
                if "cumulative_risk" in transport_output:
                    criterion_kwargs["transport_cumulative_risk"] = transport_output["cumulative_risk"]
            losses = criterion(
                binary_logits=output["binary_logits"],
                ordinal_logits=output["ordinal_logits"],
                binary_target=label,
                grade_target=grade,
                pos_weight=pos_weight.to(device),
                ordinal_pos_weight=ordinal_pos_weight.to(device),
                cargo_group_probs=batch.get("cargo_group_probs").to(device) if batch.get("cargo_group_probs") is not None else None,
                cargo_sample_weight=batch.get("cargo_sample_weight").to(device) if batch.get("cargo_sample_weight") is not None else None,
                **criterion_kwargs,
            )
            loss = losses["loss"] / max(accumulation_steps, 1)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step_index % max(accumulation_steps, 1) == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(losses["loss"].detach().cpu().item())
        for key, value in losses.items():
            running_components[key] = running_components.get(key, 0.0) + float(value.detach().cpu().item())
        steps += 1
    if steps % max(accumulation_steps, 1) != 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    stats = {"loss": running_loss / max(steps, 1)}
    for key in (
        "loss_binary",
        "loss_ordinal",
        "loss_agreement",
        "loss_rank",
        "loss_stability",
        "loss_prototype",
        "loss_prototype_order",
        "loss_distribution_grade",
        "loss_branch",
        "loss_rps",
        "loss_risk_consistency",
        "loss_risk_ranking",
        "loss_risk_invariance",
        "loss_risk_distribution_alignment",
        "loss_quantile_risk_calibration",
        "loss_cargo_group",
        "loss_evidence_diversity",
        "loss_progression_direction",
        "loss_odic_consistency",
        "loss_odic_relation",
        "transport_pair_count",
    ):
        stats[key] = running_components.get(key, float("nan")) / max(steps, 1)
    return stats


def _apply_trainability_controls(
    *,
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> bool:
    freeze_backbone = bool(train_cfg.get("freeze_backbone", model_cfg.get("freeze_backbone", False)))
    if not freeze_backbone:
        return False
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(False)
        return True
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return False
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return True


def _evaluate_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BinaryOrdinalLoss,
    device: torch.device,
    pos_weight: torch.Tensor,
    ordinal_pos_weight: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: str,
    memory_format: torch.memory_format,
    transport_audit: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    loss_components: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True).contiguous(memory_format=memory_format)
            label = batch["label"].to(device, non_blocking=True)
            grade = batch["grade"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled, amp_dtype):
                output = model(image)
                transport_output = model(_transport_appearance_perturbation(image)) if transport_audit else None
                criterion_kwargs = _criterion_kwargs_from_output(output)
                if transport_output is not None:
                    if "clinical_risk_embedding" in transport_output:
                        criterion_kwargs["transport_risk_embedding"] = transport_output["clinical_risk_embedding"]
                    if "cumulative_risk" in transport_output:
                        criterion_kwargs["transport_cumulative_risk"] = transport_output["cumulative_risk"]
                loss_dict = criterion(
                    binary_logits=output["binary_logits"],
                    ordinal_logits=output["ordinal_logits"],
                    binary_target=label,
                    grade_target=grade,
                    pos_weight=pos_weight.to(device),
                    ordinal_pos_weight=ordinal_pos_weight.to(device),
                    cargo_group_probs=batch.get("cargo_group_probs").to(device) if batch.get("cargo_group_probs") is not None else None,
                    cargo_sample_weight=batch.get("cargo_sample_weight").to(device) if batch.get("cargo_sample_weight") is not None else None,
                    **criterion_kwargs,
                )
            losses.append(float(loss_dict["loss"].detach().cpu().item()))
            for key, value in loss_dict.items():
                loss_components.setdefault(key, []).append(float(value.detach().cpu().item()))
            prob_pos = _to_numpy_float(torch.sigmoid(output["binary_logits"]))
            ordinal_logits = _to_numpy_float(output["ordinal_logits"])
            pred_grade = _ordinal_predictions(ordinal_logits)
            grade_probs = output.get("grade_probs")
            if grade_probs is not None:
                pred_grade = _to_numpy_float(grade_probs).argmax(axis=1).astype(np.int64)
            for index in range(len(prob_pos)):
                row = {
                        "patient_id": batch["patient_id"][index],
                        "image_id": batch["image_id"][index],
                        "dataset": batch["dataset"][index],
                        "split_role": batch["split_role"][index],
                        "label": int(batch["label"][index].item()),
                        "grade": int(batch["grade"][index].item()),
                        "prob_pos": float(prob_pos[index]),
                        "logit_pos": float(output["binary_logits"][index].detach().float().cpu().item()),
                        "pred_grade": int(pred_grade[index]),
                        "ordinal_logit_0": float(ordinal_logits[index, 0]),
                        "ordinal_logit_1": float(ordinal_logits[index, 1]),
                        "ordinal_logit_2": float(ordinal_logits[index, 2]),
                        "ordinal_logit_3": float(ordinal_logits[index, 3]),
                }
                row.update(_prediction_extra_columns(output, index))
                if transport_output is not None:
                    original_risk = float(output["cumulative_risk"][index].detach().float().cpu().item())
                    transport_risk = float(transport_output["cumulative_risk"][index].detach().float().cpu().item())
                    row["risk_original"] = original_risk
                    row["risk_transport"] = transport_risk
                    row["delta_risk"] = abs(original_risk - transport_risk)
                    row["embedding_distance"] = float(
                        torch.norm(
                            output["clinical_risk_embedding"][index].detach().float()
                            - transport_output["clinical_risk_embedding"][index].detach().float(),
                            p=2,
                        ).cpu().item()
                    )
                    row["screen_probability_delta"] = abs(
                        float(output["screening_probability"][index].detach().float().cpu().item())
                        - float(transport_output["screening_probability"][index].detach().float().cpu().item())
                    )
                rows.append(row)
    stats = {"loss": float(np.mean(losses)) if losses else float("nan")}
    for key in (
        "loss_binary",
        "loss_ordinal",
        "loss_agreement",
        "loss_rank",
        "loss_stability",
        "loss_prototype",
        "loss_prototype_order",
        "loss_distribution_grade",
        "loss_branch",
        "loss_rps",
        "loss_risk_consistency",
        "loss_risk_distribution_alignment",
        "loss_quantile_risk_calibration",
        "loss_cargo_group",
        "loss_evidence_diversity",
        "loss_progression_direction",
        "loss_odic_consistency",
        "loss_odic_relation",
    ):
        values = loss_components.get(key, [])
        stats[key] = float(np.mean(values)) if values else float("nan")
    return stats, rows


def run_formal_training(
    *,
    config_path: str | Path,
    train_index: str | Path,
    val_index: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    epochs_override: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = _load_yaml_config(Path(config_path))
    train_cfg = dict(config.get("training", {}))
    data_cfg = dict(config.get("data", {}))
    method_cfg = dict(config.get("method", {}))
    cache_mode = str(train_cfg.get("cache_mode", "numpy"))
    memory_cache_size = int(train_cfg.get("memory_cache_size", 0))
    local_cache_root = data_cfg.get("local_cache_root")
    train_dataset = DiabeticRetinopathyDataset(
        train_index,
        require_cache=True,
        cache_mode=cache_mode,
        preload_count=int(train_cfg.get("train_preload_count", 0)),
        memory_cache_size=memory_cache_size,
        local_cache_root=local_cache_root,
    )
    val_dataset = DiabeticRetinopathyDataset(
        val_index,
        require_cache=True,
        cache_mode=cache_mode,
        preload_count=int(train_cfg.get("val_preload_count", 0)),
        memory_cache_size=int(train_cfg.get("val_memory_cache_size", memory_cache_size)),
        local_cache_root=local_cache_root,
    )
    pos_weight, ordinal_pos_weight = _compute_pos_weight(train_dataset)

    model_cfg = dict(config.get("model", {}))
    loss_cfg = dict(config.get("loss", {}))
    eval_cfg = dict(config.get("evaluation", {}))
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    progress_path = out_dir / "progress.json"

    seed_everything(int(seed))

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runtime_controls = _resolve_runtime_controls(
        train_cfg=train_cfg,
        device=resolved_device,
        runner_parity_mode=_runner_parity_mode(method_cfg=method_cfg, loss_cfg=loss_cfg),
    )
    memory_format = _resolve_memory_format(
        device=resolved_device,
        channels_last=bool(train_cfg.get("channels_last", False)),
    )
    batch_size = int(train_cfg.get("per_device_batch", 8))
    num_workers = int(runtime_controls["num_workers"])
    accumulation_steps = int(train_cfg.get("gradient_accumulation", 1))
    epochs = int(epochs_override if epochs_override is not None else train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 8))
    grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    amp_enabled = bool(runtime_controls["amp_enabled"])
    amp_dtype = str(runtime_controls["amp_dtype"])
    validation_interval = int(train_cfg.get("validation_interval", 1))
    prediction_save_policy = str(train_cfg.get("prediction_save_policy", "all"))
    transport_enabled = (
        float(loss_cfg.get("risk_distribution_alignment_weight", 0.0)) > 0
        or float(loss_cfg.get("quantile_risk_calibration_weight", 0.0)) > 0
    )
    transport_audit = bool(eval_cfg.get("transport_audit", False))
    history: list[dict[str, Any]] = []
    _write_json_payload(
        progress_path,
        {
            "status": "starting",
            "updated_at": _utc_now_iso(),
            "pid": os.getpid(),
            "config_path": Path(config_path).as_posix(),
            "out_dir": out_dir.as_posix(),
            "seed": int(seed),
            "total_epochs": epochs,
            "epochs_completed": 0,
            "percent_complete": 0.0,
            "validation_interval": validation_interval,
        },
    )

    try:
        with _runtime_backend_context(
            device=resolved_device,
            cudnn_benchmark=bool(runtime_controls["cudnn_benchmark"]),
            matmul_precision=str(runtime_controls["matmul_precision"]),
            allow_tf32=bool(runtime_controls["allow_tf32"]),
            use_deterministic_algorithms=bool(runtime_controls["use_deterministic_algorithms"]),
        ):
            model = build_dr_model(model_cfg).to(resolved_device)
            if memory_format == torch.channels_last:
                model = model.to(memory_format=memory_format)
            backbone_frozen = _apply_trainability_controls(model=model, train_cfg=train_cfg, model_cfg=model_cfg)
            criterion = BinaryOrdinalLoss(
                binary_weight=float(loss_cfg.get("binary_weight", 1.0)),
                ordinal_weight=float(loss_cfg.get("ordinal_weight", 0.25)),
                agreement_weight=float(loss_cfg.get("agreement_weight", 0.0)),
                rank_weight=float(loss_cfg.get("rank_weight", 0.0)),
                stability_weight=float(loss_cfg.get("stability_weight", 0.0)),
                prototype_weight=float(loss_cfg.get("prototype_weight", 0.0)),
                prototype_order_weight=float(loss_cfg.get("prototype_order_weight", 0.0)),
                distribution_grade_weight=float(loss_cfg.get("distribution_grade_weight", 0.0)),
                branch_weight=float(loss_cfg.get("branch_weight", 0.0)),
                rps_weight=float(loss_cfg.get("rps_weight", 0.0)),
                risk_consistency_weight=float(loss_cfg.get("risk_consistency_weight", 0.0)),
                risk_ranking_weight=float(loss_cfg.get("risk_ranking_weight", 0.0)),
                risk_invariance_weight=float(loss_cfg.get("risk_invariance_weight", 0.0)),
                risk_distribution_alignment_weight=float(loss_cfg.get("risk_distribution_alignment_weight", 0.0)),
                quantile_risk_calibration_weight=float(loss_cfg.get("quantile_risk_calibration_weight", 0.0)),
                cargo_group_weight=float(loss_cfg.get("cargo_group_weight", 0.0)),
                cargo_cvar_alpha=float(loss_cfg.get("cargo_cvar_alpha", 0.5)),
                evidence_diversity_weight=float(loss_cfg.get("evidence_diversity_weight", 0.0)),
                progression_direction_weight=float(loss_cfg.get("progression_direction_weight", 0.0)),
                odic_consistency_weight=float(loss_cfg.get("odic_consistency_weight", 0.0)),
                odic_relation_weight=float(loss_cfg.get("odic_relation_weight", 0.0)),
                odic_num_grades=int(loss_cfg.get("odic_num_grades", 5)),
                odic_momentum=float(loss_cfg.get("odic_momentum", 0.9)),
                transport_pair_sample_limit=int(loss_cfg.get("transport_pair_sample_limit", train_cfg.get("transport_pair_sample_limit", 128))),
                rank_margin=float(loss_cfg.get("rank_margin", 0.25)),
            )
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(train_cfg.get("lr", 3e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            )
            use_grad_scaler = amp_enabled and _resolve_amp_dtype(amp_dtype) == torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

            loader_kwargs = _dataloader_kwargs(
                device=resolved_device,
                num_workers=num_workers,
                pin_memory=bool(runtime_controls["pin_memory"]),
                persistent_workers=bool(runtime_controls["persistent_workers"]),
                prefetch_factor=int(runtime_controls["prefetch_factor"]),
            )
            train_seed_state = _build_loader_seed_state(seed=int(seed), num_workers=num_workers)
            val_seed_state = _build_loader_seed_state(seed=int(seed) + 1, num_workers=num_workers)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                **loader_kwargs,
                **train_seed_state,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
                **val_seed_state,
            )

            best_score = -math.inf
            best_state: dict[str, Any] | None = None
            best_summary: dict[str, Any] | None = None
            stale_epochs = 0

            for epoch in range(1, epochs + 1):
                train_stats = _train_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=resolved_device,
                    pos_weight=pos_weight,
                    ordinal_pos_weight=ordinal_pos_weight,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    scaler=scaler,
                    grad_clip=grad_clip,
                    accumulation_steps=accumulation_steps,
                    memory_format=memory_format,
                    transport_enabled=transport_enabled,
                )
                if not _should_run_validation(
                    epoch=epoch,
                    total_epochs=epochs,
                    validation_interval=validation_interval,
                ):
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": float(train_stats["loss"]),
                            "val_loss": float("nan"),
                            "validation_skipped": True,
                        }
                    )
                    latest = history[-1]
                    _write_json_payload(
                        progress_path,
                        {
                            "status": "running",
                            "updated_at": _utc_now_iso(),
                            "pid": os.getpid(),
                            "epoch": int(epoch),
                            "total_epochs": epochs,
                            "epochs_completed": int(epoch),
                            "percent_complete": round(100.0 * float(epoch) / float(max(epochs, 1)), 1),
                            "validation_skipped": True,
                            "train_loss": float(latest["train_loss"]),
                            "best_epoch": (int(best_summary["epoch"]) if best_summary is not None else None),
                            "best_qwk": (float(best_summary["qwk"]) if best_summary is not None and "qwk" in best_summary else None),
                            "best_auc": (float(best_summary["auc"]) if best_summary is not None and "auc" in best_summary else None),
                        },
                    )
                    continue

                val_stats, prediction_rows = _evaluate_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=resolved_device,
                    pos_weight=pos_weight,
                    ordinal_pos_weight=ordinal_pos_weight,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    memory_format=memory_format,
                    transport_audit=transport_audit,
                )

                y_true_binary = np.asarray([int(row["label"]) for row in prediction_rows], dtype=np.int64)
                prob_pos = np.asarray([float(row["prob_pos"]) for row in prediction_rows], dtype=np.float64)
                logits_pos = np.asarray([float(row["logit_pos"]) for row in prediction_rows], dtype=np.float64)
                y_true_grade = np.asarray([int(row["grade"]) for row in prediction_rows], dtype=np.int64)
                pred_grade = np.asarray([int(row["pred_grade"]) for row in prediction_rows], dtype=np.int64)

                screen_threshold = select_threshold_for_min_sensitivity(
                    y_true_binary,
                    prob_pos,
                    target_sensitivity=float(eval_cfg.get("screening_sensitivity_target", 0.90)),
                )
                youden_threshold = select_youden_threshold(y_true_binary, prob_pos)
                metrics = summarize_predictions(
                    y_true_binary=y_true_binary,
                    prob_pos=prob_pos,
                    logits_pos=logits_pos,
                    y_true_grade=y_true_grade,
                    pred_grade=pred_grade,
                    threshold=screen_threshold,
                )
                metrics["youden_threshold"] = float(youden_threshold)
                metrics["fixed_threshold_0_5"] = 0.5
                summary = {
                    "epoch": epoch,
                    "train_loss": float(train_stats["loss"]),
                    "val_loss": float(val_stats["loss"]),
                    "validation_skipped": False,
                    "train_loss_binary": float(train_stats.get("loss_binary", float("nan"))),
                    "train_loss_ordinal": float(train_stats.get("loss_ordinal", float("nan"))),
                    "train_loss_agreement": float(train_stats.get("loss_agreement", float("nan"))),
                    "train_loss_rank": float(train_stats.get("loss_rank", float("nan"))),
                    "train_loss_stability": float(train_stats.get("loss_stability", float("nan"))),
                    "train_loss_prototype": float(train_stats.get("loss_prototype", float("nan"))),
                    "train_loss_prototype_order": float(train_stats.get("loss_prototype_order", float("nan"))),
                    "train_loss_distribution_grade": float(train_stats.get("loss_distribution_grade", float("nan"))),
                    "train_loss_branch": float(train_stats.get("loss_branch", float("nan"))),
                    "train_loss_rps": float(train_stats.get("loss_rps", float("nan"))),
                    "train_loss_risk_consistency": float(train_stats.get("loss_risk_consistency", float("nan"))),
                    "train_loss_risk_distribution_alignment": float(train_stats.get("loss_risk_distribution_alignment", float("nan"))),
                    "train_loss_quantile_risk_calibration": float(train_stats.get("loss_quantile_risk_calibration", float("nan"))),
                    "train_loss_evidence_diversity": float(train_stats.get("loss_evidence_diversity", float("nan"))),
                    "train_loss_progression_direction": float(train_stats.get("loss_progression_direction", float("nan"))),
                    "val_loss_binary": float(val_stats.get("loss_binary", float("nan"))),
                    "val_loss_ordinal": float(val_stats.get("loss_ordinal", float("nan"))),
                    "val_loss_agreement": float(val_stats.get("loss_agreement", float("nan"))),
                    "val_loss_rank": float(val_stats.get("loss_rank", float("nan"))),
                    "val_loss_stability": float(val_stats.get("loss_stability", float("nan"))),
                    "val_loss_prototype": float(val_stats.get("loss_prototype", float("nan"))),
                    "val_loss_prototype_order": float(val_stats.get("loss_prototype_order", float("nan"))),
                    "val_loss_distribution_grade": float(val_stats.get("loss_distribution_grade", float("nan"))),
                    "val_loss_branch": float(val_stats.get("loss_branch", float("nan"))),
                    "val_loss_rps": float(val_stats.get("loss_rps", float("nan"))),
                    "val_loss_risk_consistency": float(val_stats.get("loss_risk_consistency", float("nan"))),
                    "val_loss_risk_distribution_alignment": float(val_stats.get("loss_risk_distribution_alignment", float("nan"))),
                    "val_loss_quantile_risk_calibration": float(val_stats.get("loss_quantile_risk_calibration", float("nan"))),
                    "val_loss_evidence_diversity": float(val_stats.get("loss_evidence_diversity", float("nan"))),
                    "val_loss_progression_direction": float(val_stats.get("loss_progression_direction", float("nan"))),
                    **metrics,
                }
                history.append(summary)

                checkpoint_primary = str(eval_cfg.get("checkpoint_primary", "auc"))
                checkpoint_tie_breaker = str(eval_cfg.get("checkpoint_tie_breaker", "none"))
                primary_value = float(summary.get(checkpoint_primary, float("nan")))
                tie_value = float(summary.get(checkpoint_tie_breaker, float("nan")))
                score = primary_value
                if checkpoint_tie_breaker == "brier" and np.isfinite(tie_value):
                    score = primary_value - 1e-3 * tie_value
                is_best = bool(np.isfinite(score) and score > best_score)
                prediction_path = _prediction_save_path(
                    out_dir,
                    epoch=epoch,
                    prediction_save_policy=prediction_save_policy,
                    is_best=is_best,
                )
                if prediction_path is not None:
                    _save_predictions(prediction_path, prediction_rows)
                if is_best:
                    best_score = score
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    best_summary = {
                        **summary,
                        "screening_threshold": float(screen_threshold),
                        "prediction_path": prediction_path.as_posix() if prediction_path is not None else None,
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                _write_json_payload(
                    progress_path,
                    {
                        "status": "running",
                        "updated_at": _utc_now_iso(),
                        "pid": os.getpid(),
                        "epoch": int(epoch),
                        "total_epochs": epochs,
                        "epochs_completed": int(epoch),
                        "percent_complete": round(100.0 * float(epoch) / float(max(epochs, 1)), 1),
                        "validation_skipped": False,
                        "train_loss": float(summary["train_loss"]),
                        "val_loss": float(summary["val_loss"]),
                        "qwk": float(summary["qwk"]),
                        "auc": float(summary["auc"]),
                        "specificity": float(summary["specificity"]),
                        "best_epoch": (int(best_summary["epoch"]) if best_summary is not None else None),
                        "best_qwk": (float(best_summary["qwk"]) if best_summary is not None else None),
                        "best_auc": (float(best_summary["auc"]) if best_summary is not None else None),
                    },
                )
                print(
                    f"[progress] epoch {epoch}/{epochs} "
                    f"qwk={summary['qwk']:.4f} auc={summary['auc']:.4f} "
                    f"spec={summary['specificity']:.4f}",
                    flush=True,
                )
                if stale_epochs >= patience:
                    break

        if best_state is None or best_summary is None:
            raise RuntimeError("Formal DR training produced no valid validation epoch.")

        best_ckpt = out_dir / "best.pt"
        torch.save(best_state, best_ckpt)
        history_path = out_dir / "history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_path = out_dir / "summary.json"
        payload = {
            "decision": "DR_FORMAL_TRAINING_PASS",
            "config_path": Path(config_path).as_posix(),
            "train_index": Path(train_index).as_posix(),
            "val_index": Path(val_index).as_posix(),
            "out_dir": out_dir.as_posix(),
            "seed": int(seed),
            "device": str(resolved_device),
            "epochs_ran": len(history),
            "runtime": {
                "per_device_batch": batch_size,
                "gradient_accumulation": accumulation_steps,
                "effective_batch": batch_size * accumulation_steps,
                "num_workers": num_workers,
                "runner_parity_mode": bool(runtime_controls["runner_parity_mode"]),
                "deterministic_algorithms": bool(runtime_controls["use_deterministic_algorithms"]),
                "allow_tf32": bool(runtime_controls["allow_tf32"]),
                "amp_enabled": amp_enabled,
                "amp_dtype": amp_dtype if amp_enabled else None,
                "hardware_profile": train_cfg.get("hardware_profile"),
                "validation_interval": validation_interval,
                "prediction_save_policy": prediction_save_policy,
                "channels_last": memory_format == torch.channels_last,
                "local_cache_root": local_cache_root,
                "freeze_backbone": bool(backbone_frozen),
            },
            "best_checkpoint": best_ckpt.as_posix(),
            **best_summary,
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_json_payload(
            progress_path,
            {
                "status": "complete",
                "updated_at": _utc_now_iso(),
                "pid": os.getpid(),
                "epoch": int(best_summary["epoch"]),
                "total_epochs": epochs,
                "epochs_completed": len(history),
                "percent_complete": 100.0,
                "best_epoch": int(best_summary["epoch"]),
                "best_qwk": float(best_summary["qwk"]),
                "best_auc": float(best_summary["auc"]),
                "summary_path": summary_path.as_posix(),
                "best_checkpoint": best_ckpt.as_posix(),
            },
        )
        return payload
    except Exception as exc:
        _write_json_payload(
            progress_path,
            {
                "status": "failed",
                "updated_at": _utc_now_iso(),
                "pid": os.getpid(),
                "total_epochs": epochs,
                "epochs_completed": int(history[-1]["epoch"]) if history else 0,
                "percent_complete": round(100.0 * float(history[-1]["epoch"]) / float(max(epochs, 1)), 1) if history else 0.0,
                "error": repr(exc),
            },
        )
        raise
