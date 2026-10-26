#!/usr/bin/env python3
"""Locked CAL, evaluation, and paired-cluster analysis for SPARROW V1 P4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.dr.sparrow import p4_execution as p4


RUN_ROOT = p4.RUN_ROOT
ARMS = p4.ARMS
SEEDS = p4.SEEDS
GPU_COUNT = p4.MAX_GPUS
ANALYSIS_ROOT = RUN_ROOT / "analysis"
MODEL_LOCK_PATH = RUN_ROOT / "FINAL_MODELS_LOCKED.json"
CAL_INPUT = RUN_ROOT / "inputs/cal_eye_manifest.csv"
DEV_INPUT = RUN_ROOT / "inputs/dev_manifest.csv"
EVAL_INPUT_DIR = RUN_ROOT / "evaluation_inputs"
CAL_THRESHOLD_LOCK = RUN_ROOT / "CAL_THRESHOLDS_LOCKED.json"
CAL_RULE_CORRECTION_ID = "cal_threshold_rule_correction_20261002"
CORRECTION_BASELINE_ANALYSIS_SHA256 = "b4b580776ee71507a7658a5446ac8c731d3199ca090445ddd0110ed0c9537661"
CAL_THRESHOLD_CRITERION = (
    "highest unique q2 candidate in unique([0.0] + q2) with image sensitivity "
    "one-sided 90% Wilson lower bound >= 0.90; no predicted-negative requirement"
)


def configure_run_root(run_root: Path) -> None:
    """Bind this process to the arms and paths in one frozen execution contract."""
    global RUN_ROOT, ARMS, SEEDS, ANALYSIS_ROOT, MODEL_LOCK_PATH, CAL_INPUT, DEV_INPUT
    global EVAL_INPUT_DIR, CAL_THRESHOLD_LOCK
    RUN_ROOT = Path(run_root).resolve()
    contract_path = RUN_ROOT / "EXECUTION_CONTRACT.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        ARMS = tuple(str(x) for x in contract["arms"])
        SEEDS = tuple(int(x) for x in contract["seeds"])
    ANALYSIS_ROOT = RUN_ROOT / "analysis"
    MODEL_LOCK_PATH = RUN_ROOT / "FINAL_MODELS_LOCKED.json"
    CAL_INPUT = RUN_ROOT / "inputs/cal_eye_manifest.csv"
    DEV_INPUT = RUN_ROOT / "inputs/dev_manifest.csv"
    EVAL_INPUT_DIR = RUN_ROOT / "evaluation_inputs"
    CAL_THRESHOLD_LOCK = RUN_ROOT / "CAL_THRESHOLDS_LOCKED.json"
    p4.RUN_ROOT = RUN_ROOT
    p4.ARMS = ARMS
    p4.SEEDS = SEEDS

COHORT_EXPECTED_N = {"Retino": 1392, "SetB": 400}
CLINICAL_METRICS = (
    "false_referral_grade0", "false_referral_grade1", "false_referral_grade01",
    "miss_grade2", "miss_grade3", "miss_grade4", "miss_grade234",
    "sensitivity", "specificity",
)
PRIMARY_METRICS = ("qwk", "auroc_q2")
PLANNED_CONTRASTS = (
    ("M3", "C2"), ("M3", "C0"), ("M3", "C1"), ("M3", "SNR-B3-adapted"),
    ("C2", "C0"), ("C2", "C1"), ("C2", "SNR-B3-adapted"),
    ("S-X", "C2"), ("M3", "F-R"), ("F-R", "C2"), ("M3", "S-X"),
)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_frozen_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        return [dict(row) for row in reader]


def _write_frozen_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _model_key(arm: str, seed: int) -> str:
    return f"{arm.replace('-', '_')}__seed_{int(seed)}"


def lock_final_models(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    scheduler_path = run_root / "FORMAL_SCHEDULER.json"
    if not scheduler_path.is_file():
        raise FileNotFoundError("formal scheduler receipt missing")
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if scheduler.get("status") != "FORMAL_QUEUE_COMPLETE" or scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("all frozen formal jobs must complete before locking any terminal model")
    contract_sha = p4._contract_hash_from_run(run_root)
    models: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            arm_dir = run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}"
            receipt_path = arm_dir / "run_receipt.json"
            model_path = arm_dir / "final_model.pt"
            if not receipt_path.is_file() or not model_path.is_file():
                raise FileNotFoundError(f"terminal model missing for {arm}/{seed}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "FORMAL_TRAINING_COMPLETE" or receipt.get("contract_sha256") != contract_sha:
                raise RuntimeError(f"formal receipt invalid for {arm}/{seed}")
            digest = p4.sha256_file(model_path)
            if digest != receipt.get("final_model_sha256") or receipt.get("logical_updates_requested") != p4.FORMAL_UPDATES or receipt.get("logical_next_update") != p4.FORMAL_UPDATES:
                raise RuntimeError(f"final model identity or update count invalid for {arm}/{seed}")
            models.append({
                "arm": arm, "seed": seed, "model_path": str(model_path),
                "model_sha256": digest, "model_bytes": model_path.stat().st_size,
                "run_receipt_path": str(receipt_path), "run_receipt_sha256": p4.sha256_file(receipt_path),
                "logical_updates": receipt["logical_updates_requested"],
                "physical_updates_total_for_run": receipt["physical_updates_total_for_run"],
                "training_manifest_sha256": receipt["training_manifest_sha256"],
                "contract_sha256": contract_sha,
            })
    lock = {
        "status": "ALL_TERMINAL_MODELS_LOCKED",
        "contract_sha256": contract_sha,
        "formal_scheduler_sha256": p4.sha256_file(scheduler_path),
        "expected_models": len(ARMS) * len(SEEDS),
        "models": models,
        "locked_utc": p4.utc_now(),
        "calibration_or_evaluation_read_before_lock": False,
    }
    path = run_root / "FINAL_MODELS_LOCKED.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("models") != models:
            raise FileExistsError("a different terminal-model lock already exists")
        return old
    p4.write_json(path, lock)
    return lock


class _EvalDataset:
    def __init__(self, rows: Sequence[dict[str, str]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        row = self.rows[index]
        cache_path = str(row.get("cache_path", "")).strip()
        array = np.load(cache_path, allow_pickle=False) if cache_path else _load_registered_retino_image(str(row.get("image_path", "")))
        if array.shape != (3, 512, 512) or array.dtype != np.float32:
            raise ValueError(f"evaluation cache metadata mismatch: {row['row_index']}")
        return {
            "image": torch.from_numpy(np.asarray(array, dtype=np.float32)),
            "grade": int(row["grade"]), "label": int(row["label"]),
            "component_hash": row["component_hash"], "row_index": int(row["row_index"]),
        }


def _load_registered_retino_image(image_path_value: str) -> np.ndarray:
    """Match the existing SPARROW external evaluator's registered Retino transform."""
    image_path = Path(image_path_value)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        cropped = rgb.crop((left, top, left + crop_size, top + crop_size))
        resized = cropped.resize((512, 512), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def _predict_rows(model: Any, rows: Sequence[dict[str, str]], device: Any) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader
    dataset = _EvalDataset(rows)
    loader = DataLoader(dataset, batch_size=p4.BATCH_SIZE, shuffle=False, num_workers=p4.WORKER_COUNT, pin_memory=True, persistent_workers=p4.WORKER_COUNT > 0)
    outputs: dict[str, list[np.ndarray]] = {key: [] for key in ("grade_probs", "grade_pred", "q2", "screening_prob")}
    labels: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    component_hashes: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                result = model(image)
                if hasattr(model, "feature_pyramid") and hasattr(model.feature_pyramid, "pop_aux"):
                    model.feature_pyramid.pop_aux()
                probs = result["grade_probs"].float().detach().cpu().numpy()
                q2 = probs[:, 2:].sum(axis=1, dtype=np.float64).astype(np.float32)
                outputs["grade_probs"].append(probs.astype(np.float32, copy=False))
                outputs["grade_pred"].append(probs.argmax(axis=1).astype(np.int8))
                outputs["q2"].append(q2)
                screen = result.get("screening_prob", result.get("binary_prob"))
                outputs["screening_prob"].append(screen.float().detach().cpu().numpy().astype(np.float32))
                labels.append(batch["grade"].numpy().astype(np.int8))
                row_indices.append(batch["row_index"].numpy().astype(np.int32))
                component_hashes.append(np.asarray(batch["component_hash"], dtype="U64"))
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    return {
        **{key: np.concatenate(value, axis=0) for key, value in outputs.items()},
        "grade": np.concatenate(labels), "row_index": np.concatenate(row_indices),
        "component_hash": np.concatenate(component_hashes),
    }


def _load_model_for_run(arm: str, seed: int, run_root: Path = RUN_ROOT) -> tuple[Any, dict[str, Any], Any]:
    import torch
    model, init_receipt = p4.build_p4_model(arm, seed)
    device = torch.device("cuda", torch.cuda.current_device())
    model.to(device)
    model_path = run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}" / "final_model.pt"
    lock = json.loads((run_root / "FINAL_MODELS_LOCKED.json").read_text(encoding="utf-8"))
    model_lock = next((item for item in lock["models"] if item["arm"] == arm and int(item["seed"]) == int(seed)), None)
    if model_lock is None or p4.sha256_file(model_path) != model_lock["model_sha256"]:
        raise RuntimeError(f"terminal model changed after lock: {arm}/{seed}")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return model, init_receipt, device


def _cal_rows(run_root: Path = RUN_ROOT) -> list[dict[str, str]]:
    if not CAL_INPUT.is_file():
        _, rows = p4._read_csv(run_root / "inputs/cal_manifest.csv")
        rows = [r for r in rows if r["source"] == "EyePACS"]
        _write_frozen_csv(CAL_INPUT, rows, ("source", "row_index", "source_row", "component_hash", "grade", "label", "role", "cache_path"))
    rows = _read_frozen_csv(CAL_INPUT)
    if any(row.get("source") != "EyePACS" or row.get("role") != "CAL" for row in rows):
        raise RuntimeError("CAL prediction input is not the frozen EyePACS CAL partition")
    return rows


def _calibration_worker(arm: str, seed: int, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    import torch
    if not MODEL_LOCK_PATH.is_file():
        raise RuntimeError("all terminal models must be locked before CAL")
    arm_dir = ANALYSIS_ROOT / "calibration" / arm.replace("-", "_") / f"seed_{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    output_path = arm_dir / "cal_predictions.npz"
    receipt_path = arm_dir / "run_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    rows = _cal_rows(run_root)
    started = time.perf_counter()
    cpu_started = p4._process_cpu_seconds("self")
    child_cpu_started = p4._process_cpu_seconds("children")
    model, init_receipt, device = _load_model_for_run(arm, seed, run_root)
    predictions = _predict_rows(model, rows, device)
    _write_npz_atomic(output_path, **predictions)
    import torch as _torch
    _torch.cuda.synchronize(device)
    receipt = {
        "status": "CAL_PREDICTIONS_COMPLETE", "stage": "calibration", "arm": arm, "seed": int(seed),
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "model_sha256": p4.sha256_file(run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}" / "final_model.pt"),
        "cal_manifest_sha256": p4.sha256_file(CAL_INPUT), "cal_image_count": len(rows),
        "prediction_path": str(output_path), "prediction_sha256": p4.sha256_file(output_path),
        "allocated_gpu_wall_seconds": time.perf_counter() - started,
        "cpu_core_seconds": max(
            0.0,
            (p4._process_cpu_seconds("self") - cpu_started)
            + (p4._process_cpu_seconds("children") - child_cpu_started),
        ),
        "gpu_uuid": os.environ.get("P4_GPU_UUID", "not_injected"),
        "completed_utc": p4.utc_now(),
    }
    p4.write_json(receipt_path, receipt)
    return receipt


def _lower_wilson(successes: int, total: int, z: float = 1.2815515655) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt(max(p * (1.0 - p) / total + z * z / (4.0 * total * total), 0.0))
    return max(0.0, (center - spread) / denom)


def select_calibration_threshold(q2: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Apply the adopted CAL-only q2 policy without a predicted-negative constraint."""
    scores = np.asarray(q2, dtype=np.float64)
    y = np.asarray(labels)
    if scores.ndim != 1 or y.ndim != 1 or len(scores) != len(y):
        raise ValueError("CAL scores and binary labels must be aligned one-dimensional arrays")
    if not np.all(np.isin(y, (0, 1))):
        return {
            "status": "INVALID_CAL_LABEL",
            "tau_q2": None,
            "cal_positive_images": int(np.sum(y == 1)),
            "cal_negative_images": int(np.sum(y == 0)),
            "candidate_count": 0,
            "curve": [],
            "reason": "CAL label contains values outside {0,1}",
        }
    y = y.astype(np.int8, copy=False)
    positive_total = int(np.sum(y == 1))
    negative_total = int(np.sum(y == 0))
    if not np.all(np.isfinite(scores)) or np.any(scores < -1e-7) or np.any(scores > 1.0 + 1e-6):
        return {
            "status": "INVALID_CAL_SCORE",
            "tau_q2": None,
            "cal_positive_images": positive_total,
            "cal_negative_images": negative_total,
            "candidate_count": 0,
            "curve": [],
            "reason": "CAL q2 contains non-finite values or values outside probability bounds",
        }
    if positive_total < 15 or negative_total < 1:
        return {
            "status": "NOT_ESTIMABLE",
            "tau_q2": None,
            "cal_positive_images": positive_total,
            "cal_negative_images": negative_total,
            "candidate_count": 0,
            "curve": [],
            "reason": "CAL requires at least 15 true positives and one true negative",
        }

    candidates = np.unique(np.concatenate(([0.0], scores)))
    curve: list[dict[str, Any]] = []
    feasible: list[float] = []
    for tau in candidates:
        referral = scores >= float(tau)
        predicted_positive = int(referral.sum())
        predicted_negative = int(len(referral) - predicted_positive)
        true_positive = int((referral & (y == 1)).sum())
        lower = _lower_wilson(true_positive, positive_total)
        valid = lower >= 0.90
        point = {
            "tau": float(tau),
            "true_positive": true_positive,
            "n_positive": positive_total,
            "n_negative": negative_total,
            "predicted_positive": predicted_positive,
            "predicted_negative": predicted_negative,
            "sensitivity": true_positive / positive_total,
            "wilson_lower_90": lower,
            "valid": valid,
        }
        curve.append(point)
        if valid:
            feasible.append(float(tau))
    if feasible:
        tau = max(feasible)
        selected = next(row for row in curve if row["tau"] == tau)
        status = "THRESHOLD_LOCKED"
        reason = None
    else:
        tau = None
        selected = None
        status = "CAL_INSUFFICIENT"
        reason = "no unique CAL q2 candidate meets the one-sided Wilson sensitivity bound"
    return {
        "status": status,
        "tau_q2": tau,
        "cal_positive_images": positive_total,
        "cal_negative_images": negative_total,
        "candidate_count": int(len(candidates)),
        "selected": selected,
        "curve": curve,
        "criterion": CAL_THRESHOLD_CRITERION,
        "wilson_z": 1.2815515655,
        "reason": reason,
    }


def lock_calibration_thresholds(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    model_lock = json.loads((run_root / "FINAL_MODELS_LOCKED.json").read_text(encoding="utf-8"))
    cal_rows = _cal_rows(run_root)
    labels = np.asarray([int(r["label"]) for r in cal_rows], dtype=np.int8)
    positive_total = int(labels.sum())
    negative_total = int(len(labels) - positive_total)
    models: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            arm_dir = ANALYSIS_ROOT / "calibration" / arm.replace("-", "_") / f"seed_{seed}"
            receipt_path = arm_dir / "run_receipt.json"
            predictions_path = arm_dir / "cal_predictions.npz"
            if not receipt_path.is_file() or not predictions_path.is_file():
                raise FileNotFoundError(f"CAL prediction missing for {arm}/{seed}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            pred = np.load(predictions_path, allow_pickle=False)
            q2 = pred["q2"].astype(np.float64)
            if len(q2) != len(labels) or not np.array_equal(pred["grade"], np.asarray([int(r["grade"]) for r in cal_rows], dtype=np.int8)):
                raise RuntimeError(f"CAL output row/label alignment failed for {arm}/{seed}")
            selection = select_calibration_threshold(q2, labels)
            status, tau = selection["status"], selection["tau_q2"]
            models.append({
                "arm": arm, "seed": seed, "status": status, "tau_q2": tau,
                "cal_positive_images": positive_total, "cal_negative_images": negative_total,
                "criterion": CAL_THRESHOLD_CRITERION,
                "wilson_z": selection.get("wilson_z", 1.2815515655),
                "prediction_sha256": receipt["prediction_sha256"],
                "candidate_count": selection["candidate_count"], "curve": selection["curve"],
            })
    lock = {
        "status": "ALL_CAL_THRESHOLDS_LOCKED_OR_EXPLICITLY_NOT_ESTIMABLE",
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "model_lock_sha256": p4.sha256_file(run_root / "FINAL_MODELS_LOCKED.json"),
        "cal_manifest_sha256": p4.sha256_file(CAL_INPUT),
        "cal_image_count": len(cal_rows), "cal_positive_images": positive_total, "cal_negative_images": negative_total,
        "score": "q2=p2+p3+p4", "positive_rule": "q2 >= tau_q2",
        "models": models, "locked_utc": p4.utc_now(),
        "thresholds_chosen_using_only_eyePACS_CAL": True,
        "test_labels_or_predictions_used": False,
    }
    if CAL_THRESHOLD_LOCK.exists():
        previous = json.loads(CAL_THRESHOLD_LOCK.read_text(encoding="utf-8"))
        if previous.get("models") != models:
            raise FileExistsError("a different CAL threshold lock already exists")
        return previous
    p4.write_json(CAL_THRESHOLD_LOCK, lock)
    return lock


def _external_component_value(row: dict[str, str], cohort: str) -> str:
    for key in ("patient_id", "patient", "patient_key", "cluster_id", "component_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    if cohort == "Retino":
        value = str(row.get("image_id", "")).strip()
        if value:
            return value
    raise ValueError(f"registered cluster/component key missing from {cohort} manifest")


def _freeze_external_manifest(cohort: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields, source_rows = p4._read_csv(path)
    required = {"grade", "cache_path"}
    if not required.issubset(fields):
        raise ValueError(f"external manifest missing required fields for {cohort}: {sorted(required - set(fields))}")
    rows: list[dict[str, Any]] = []
    image_ids_seen: set[str] = set()
    identity = hashlib.sha256()
    for source_index, raw in enumerate(source_rows):
        grade = int(raw["grade"])
        label = int(raw.get("label", int(grade >= 2)))
        if grade not in range(5) or label != int(grade >= 2):
            raise ValueError(f"external grade/binary mapping mismatch in {cohort}, row={source_index}")
        image_id = str(raw.get("image_id", raw.get("image_path", source_index))).strip()
        if not image_id or image_id in image_ids_seen:
            raise ValueError(f"external image identity is missing or duplicated in {cohort}")
        image_ids_seen.add(image_id)
        component_raw = _external_component_value(raw, cohort)
        component_hash = p4.sha256_bytes(("sparrow-v1-p4-evaluation-component|" + cohort + "|" + component_raw).encode("utf-8"))
        raw_cache_path = str(raw.get("cache_path", "")).strip()
        raw_image_path = str(raw.get("image_path", "")).strip()
        cache_path = ""
        image_path = ""
        if raw_cache_path:
            mapped_cache = p4._mapped_path(raw_cache_path)
            if not mapped_cache.is_file():
                raise FileNotFoundError(f"registered evaluation cache missing after explicit migration mapping: {mapped_cache}")
            shape, dtype = p4._npy_metadata(mapped_cache)
            if shape != (3, 512, 512) or dtype != "float32":
                raise ValueError(f"evaluation cache metadata mismatch in {cohort} row {source_index}: {shape}/{dtype}")
            cache_path = str(mapped_cache)
        elif cohort == "Retino" and raw_image_path:
            mapped_image = p4._mapped_path(raw_image_path)
            if not mapped_image.is_file():
                raise FileNotFoundError(f"registered Retino image missing after explicit migration mapping: {mapped_image}")
            image_path = str(mapped_image)
        else:
            raise ValueError(f"no registered image input is available in {cohort} row {source_index}")
        identity.update(f"{source_index}\0{component_hash}\0{grade}\0{label}\0{image_id}\n".encode("utf-8"))
        rows.append({
            "cohort": cohort, "row_index": source_index, "component_hash": component_hash,
            "grade": grade, "label": label, "cache_path": cache_path, "image_path": image_path,
        })
    if len(rows) != COHORT_EXPECTED_N[cohort]:
        raise RuntimeError(f"registered {cohort} cohort size changed: expected {COHORT_EXPECTED_N[cohort]}, got {len(rows)}")
    summary = {
        "cohort": cohort, "manifest_path": str(path), "manifest_sha256": p4.sha256_file(path),
        "manifest_bytes": path.stat().st_size, "row_count": len(rows),
        "component_count": len({row["component_hash"] for row in rows}),
        "grades": {str(k): sum(row["grade"] == k for row in rows) for k in range(5)},
        "registered_identity_sha256": identity.hexdigest(),
        "component_field_used": "registered patient/cluster/component key; anonymized with a cohort-scoped SHA256",
        "data_paths_migration_mapped": True,
        "input_mode": "image_path_with_existing_registered_Retino_transform" if cohort == "Retino" else "registered_npy_cache",
    }
    return rows, summary


def freeze_evaluation_inputs(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    if not CAL_THRESHOLD_LOCK.is_file():
        raise RuntimeError("all CAL threshold decisions must be locked before opening DEV or external evaluation manifests")
    if not MODEL_LOCK_PATH.is_file():
        raise RuntimeError("terminal models must be locked first")
    freeze_started = time.perf_counter()
    freeze_cpu_started = p4._process_cpu_seconds("self")
    retino_smoke_wall_seconds: float | None = None
    retino_smoke_cpu_core_seconds: float | None = None
    dest = run_root / "evaluation_inputs"
    partial_files: list[dict[str, Any]] = []
    if dest.exists() and any(dest.iterdir()):
        receipt_path = dest / "EVALUATION_INPUT_RECEIPT.json"
        if receipt_path.is_file():
            return json.loads(receipt_path.read_text(encoding="utf-8"))
        existing = sorted(path for path in dest.iterdir() if path.is_file())
        allowed = {"DEV.csv", "Retino.csv", "SetB.csv"}
        if len(existing) != len(list(dest.iterdir())) or any(path.name not in allowed for path in existing):
            raise FileExistsError(f"unexpected partial evaluation input freeze exists: {dest}")
        partial_files = [
            {"name": path.name, "sha256": p4.sha256_file(path), "bytes": path.stat().st_size}
            for path in existing
        ]
    dest.mkdir(parents=True, exist_ok=True)
    _, dev_rows = p4._read_csv(run_root / "inputs/dev_manifest.csv")
    if not dev_rows or any(row.get("role") != "DEV" for row in dev_rows):
        raise RuntimeError("DEV manifest is not the frozen DEV partition")
    _write_frozen_csv(dest / "DEV.csv", dev_rows, ("source", "row_index", "source_row", "component_hash", "grade", "label", "role", "cache_path"))
    manifest_summaries: dict[str, Any] = {
        "DEV": {
            "manifest_path": str(run_root / "inputs/dev_manifest.csv"),
            "manifest_sha256": p4.sha256_file(run_root / "inputs/dev_manifest.csv"),
            "row_count": len(dev_rows),
            "component_count": len({row["component_hash"] for row in dev_rows}),
            "grades": {str(k): sum(int(row["grade"]) == k for row in dev_rows) for k in range(5)},
        }
    }
    retino_preprocessor = Path(__file__).resolve().with_name("evaluate_sparrow_external.py")
    for cohort, path in p4.EXTERNAL_MANIFESTS.items():
        rows, summary = _freeze_external_manifest(cohort, path)
        _write_frozen_csv(dest / f"{cohort}.csv", rows, ("cohort", "row_index", "component_hash", "grade", "label", "cache_path", "image_path"))
        if cohort == "Retino":
            retino_smoke_started = time.perf_counter()
            retino_smoke_cpu_started = p4._process_cpu_seconds("self")
            sample = _load_registered_retino_image(rows[0]["image_path"])
            if sample.shape != (3, 512, 512) or sample.dtype != np.float32:
                raise ValueError(f"registered Retino transform returned {sample.shape}/{sample.dtype}")
            retino_smoke_wall_seconds = time.perf_counter() - retino_smoke_started
            retino_smoke_cpu_core_seconds = max(0.0, p4._process_cpu_seconds("self") - retino_smoke_cpu_started)
        manifest_summaries[cohort] = summary
    receipt = {
        "status": "LOCKED_DEV_AND_EXISTING_EXTERNAL_EVALUATION_INPUTS",
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "model_lock_sha256": p4.sha256_file(MODEL_LOCK_PATH),
        "cal_threshold_lock_sha256": p4.sha256_file(CAL_THRESHOLD_LOCK),
        "manifests": manifest_summaries,
        "retino_true_patient_mapping": "not asserted; bootstrap unit is the registered image component",
        "setb_bootstrap_unit": "registered patient key",
        "retino_preprocessing": {
            "description": "existing evaluator: RGB, centered square crop, bicubic 512 resize, ImageNet mean/std normalization",
            "source_path": str(retino_preprocessor),
            "source_sha256": p4.sha256_file(retino_preprocessor),
            "smoke_shape": [3, 512, 512], "smoke_dtype": "float32",
            "smoke_wall_seconds": retino_smoke_wall_seconds,
            "smoke_cpu_core_seconds": retino_smoke_cpu_core_seconds,
        },
        "input_freeze_wall_seconds": time.perf_counter() - freeze_started,
        "input_freeze_cpu_core_seconds": max(0.0, p4._process_cpu_seconds("self") - freeze_cpu_started),
        "partial_freeze_recovery": {"recovered": bool(partial_files), "previous_partial_files": partial_files},
        "setc_opened": False,
        "created_utc": p4.utc_now(),
    }
    for cohort in ("DEV", "Retino", "SetB"):
        receipt["frozen_" + cohort + "_csv_sha256"] = p4.sha256_file(dest / f"{cohort}.csv")
    p4.write_json(dest / "EVALUATION_INPUT_RECEIPT.json", receipt)
    return receipt


def _referral_metrics(grade: np.ndarray, q2: np.ndarray, tau: float | None) -> dict[str, Any]:
    y = np.asarray(grade, dtype=np.int64)
    if tau is None:
        return {name: {"numerator": None, "denominator": None, "rate": None} for name in CLINICAL_METRICS}
    referral = np.asarray(q2, dtype=np.float64) >= float(tau)
    definitions = {
        "false_referral_grade0": (y == 0, referral),
        "false_referral_grade1": (y == 1, referral),
        "false_referral_grade01": (y <= 1, referral),
        "miss_grade2": (y == 2, ~referral),
        "miss_grade3": (y == 3, ~referral),
        "miss_grade4": (y == 4, ~referral),
        "miss_grade234": (y >= 2, ~referral),
        "sensitivity": (y >= 2, referral),
        "specificity": (y <= 1, ~referral),
    }
    out: dict[str, Any] = {}
    for name, (denominator_mask, numerator_mask) in definitions.items():
        denominator = int(denominator_mask.sum())
        numerator = int((denominator_mask & numerator_mask).sum())
        out[name] = {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}
    return out


def _metrics_for_predictions(pred: dict[str, np.ndarray], tau: float | None) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, roc_auc_score
    y = np.asarray(pred["grade"], dtype=np.int64)
    probs = np.asarray(pred["grade_probs"], dtype=np.float64)
    pred_grade = np.asarray(pred["grade_pred"], dtype=np.int64)
    q2 = np.asarray(pred["q2"], dtype=np.float64)
    screen = np.asarray(pred["screening_prob"], dtype=np.float64)
    positive = (y >= 2).astype(np.int8)
    q2_auc = float(roc_auc_score(positive, q2)) if len(np.unique(positive)) == 2 else None
    screen_auc = float(roc_auc_score(positive, screen)) if len(np.unique(positive)) == 2 else None
    q2_auprc = float(average_precision_score(positive, q2)) if positive.any() and (~positive.astype(bool)).any() else None
    q2_brier = float(np.mean((q2 - positive) ** 2))
    grade_nll = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    qwk_value = cohen_kappa_score(y, pred_grade, labels=[0, 1, 2, 3, 4], weights="quadratic")
    cm = confusion_matrix(y, pred_grade, labels=[0, 1, 2, 3, 4]).astype(int).tolist()
    return {
        "n_images": int(len(y)),
        "n_registered_components": int(len(np.unique(pred["component_hash"]))),
        "grade_support": {str(k): int((y == k).sum()) for k in range(5)},
        "qwk": float(qwk_value) if np.isfinite(qwk_value) else None,
        "accuracy": float((y == pred_grade).mean()),
        "auroc_q2": q2_auc,
        "auprc_q2": q2_auprc,
        "auroc_screening_probability": screen_auc,
        "brier_q2": q2_brier,
        "nll_five_grade": grade_nll,
        "q2_threshold": tau,
        "clinical_counts": _referral_metrics(y, q2, tau),
        "grade_confusion_matrix_true_rows_pred_columns": cm,
    }


def _worker_for_eval(arm: str, seed: int, stage: str, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    if arm not in ARMS or int(seed) not in SEEDS:
        raise ValueError(f"worker identity is not in the frozen contract: {arm}/{seed}")
    if not MODEL_LOCK_PATH.is_file():
        raise RuntimeError("terminal-model lock missing")
    if stage == "calibration":
        return _calibration_worker(arm, seed, run_root)
    if stage not in {"reload-check", "evaluation"}:
        raise ValueError(stage)
    if not CAL_THRESHOLD_LOCK.is_file():
        raise RuntimeError("locked source-CAL threshold receipt missing")
    eval_receipt_path = run_root / "evaluation_inputs/EVALUATION_INPUT_RECEIPT.json"
    if not eval_receipt_path.is_file():
        raise RuntimeError("frozen evaluation-input receipt missing")
    eval_input_receipt = json.loads(eval_receipt_path.read_text(encoding="utf-8"))
    if (
        eval_input_receipt.get("contract_sha256") != p4._contract_hash_from_run(run_root)
        or eval_input_receipt.get("model_lock_sha256") != p4.sha256_file(MODEL_LOCK_PATH)
        or eval_input_receipt.get("cal_threshold_lock_sha256") != p4.sha256_file(CAL_THRESHOLD_LOCK)
    ):
        raise RuntimeError("evaluation inputs are not bound to the current model/threshold locks")
    if stage == "evaluation":
        reload_scheduler = run_root / "analysis/RELOAD_CHECK_SCHEDULER.json"
        if not reload_scheduler.is_file() or json.loads(reload_scheduler.read_text(encoding="utf-8")).get("jobs_completed") != len(ARMS) * len(SEEDS):
            raise RuntimeError("every model must pass CAL reload identity before any DEV/external scoring")
    output_dir = ANALYSIS_ROOT / stage.replace("-", "_") / arm.replace("-", "_") / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "run_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    cpu_started = p4._process_cpu_seconds("self")
    child_cpu_started = p4._process_cpu_seconds("children")
    wall_started = time.perf_counter()
    model, init_receipt, device = _load_model_for_run(arm, seed, run_root)
    if stage == "reload-check":
        cal_rows = _cal_rows(run_root)
        cal_saved = np.load(ANALYSIS_ROOT / "calibration" / arm.replace("-", "_") / f"seed_{seed}" / "cal_predictions.npz", allow_pickle=False)
        check_count = min(32, len(cal_rows))
        reload_result = _predict_rows(model, cal_rows[:check_count], device)
        cal_probs = cal_saved["grade_probs"][:check_count]
        cal_q2 = cal_saved["q2"][:check_count]
        max_prob_diff = float(np.max(np.abs(reload_result["grade_probs"] - cal_probs))) if check_count else 0.0
        max_q2_diff = float(np.max(np.abs(reload_result["q2"] - cal_q2))) if check_count else 0.0
        if not np.allclose(reload_result["grade_probs"], cal_probs, rtol=1e-6, atol=1e-7) or not np.allclose(reload_result["q2"], cal_q2, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"CAL model reload predictions differ for {arm}/{seed}: max_prob={max_prob_diff}, max_q2={max_q2_diff}")
        receipt = {
            "status": "CAL_RELOAD_CHECK_PASS", "stage": stage, "arm": arm, "seed": int(seed),
            "contract_sha256": p4._contract_hash_from_run(run_root),
            "model_sha256": p4.sha256_file(run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}" / "final_model.pt"),
            "cal_threshold_lock_sha256": p4.sha256_file(CAL_THRESHOLD_LOCK),
            "cal_input_count": check_count, "cal_prob_max_abs_difference": max_prob_diff,
            "cal_q2_max_abs_difference": max_q2_diff,
            "reload_evaluation_forward_inputs": check_count,
        }
    else:
        reload_receipt_path = ANALYSIS_ROOT / "reload_check" / arm.replace("-", "_") / f"seed_{seed}" / "run_receipt.json"
        if not reload_receipt_path.is_file():
            raise RuntimeError(f"CAL reload-check receipt missing for {arm}/{seed}")
        reload_receipt = json.loads(reload_receipt_path.read_text(encoding="utf-8"))
        model_sha256 = p4.sha256_file(run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}" / "final_model.pt")
        if reload_receipt.get("status") != "CAL_RELOAD_CHECK_PASS" or reload_receipt.get("contract_sha256") != p4._contract_hash_from_run(run_root) or reload_receipt.get("model_sha256") != model_sha256 or reload_receipt.get("cal_threshold_lock_sha256") != p4.sha256_file(CAL_THRESHOLD_LOCK):
            raise RuntimeError(f"CAL reload-check receipt identity mismatch for {arm}/{seed}")
        check_count = int(reload_receipt["cal_input_count"])
        max_prob_diff = float(reload_receipt["cal_prob_max_abs_difference"])
        max_q2_diff = float(reload_receipt["cal_q2_max_abs_difference"])
        eval_thresholds = json.loads(CAL_THRESHOLD_LOCK.read_text(encoding="utf-8"))
        threshold_row = next(x for x in eval_thresholds["models"] if x["arm"] == arm and int(x["seed"]) == int(seed))
        tau = threshold_row["tau_q2"] if threshold_row["status"] == "THRESHOLD_LOCKED" else None
        eval_rows_by_cohort: dict[str, list[dict[str, str]]] = {}
        for cohort in ("DEV", "Retino", "SetB"):
            eval_rows_by_cohort[cohort] = _read_frozen_csv(run_root / "evaluation_inputs" / f"{cohort}.csv")
        result_metrics: dict[str, Any] = {}
        result_files: dict[str, Any] = {}
        for cohort, rows in eval_rows_by_cohort.items():
            predictions = _predict_rows(model, rows, device)
            prediction_path = output_dir / f"{cohort}_predictions.npz"
            _write_npz_atomic(prediction_path, **predictions)
            result_files[cohort] = {"path": str(prediction_path), "sha256": p4.sha256_file(prediction_path), "n_images": len(rows)}
            result_metrics[cohort] = _metrics_for_predictions(predictions, tau)
        receipt = {
            "status": "LOCKED_EVALUATION_COMPLETE", "stage": stage, "arm": arm, "seed": int(seed),
            "contract_sha256": p4._contract_hash_from_run(run_root),
            "model_sha256": p4.sha256_file(run_root / "formal" / arm.replace("-", "_") / f"seed_{seed}" / "final_model.pt"),
            "cal_threshold_lock_sha256": p4.sha256_file(CAL_THRESHOLD_LOCK),
            "tau_q2": tau, "threshold_status": threshold_row["status"],
            "cal_reload_check": {"input_count": check_count, "max_prob_abs_difference": max_prob_diff, "max_q2_abs_difference": max_q2_diff},
            "prediction_files": result_files, "metrics": result_metrics,
            "evaluation_forward_inputs": sum(len(rows) for rows in eval_rows_by_cohort.values()),
        }
    torch = __import__("torch")
    torch.cuda.synchronize(device)
    receipt.update({
        "allocated_gpu_wall_seconds": time.perf_counter() - wall_started,
        "cpu_core_seconds": max(
            0.0,
            (p4._process_cpu_seconds("self") - cpu_started)
            + (p4._process_cpu_seconds("children") - child_cpu_started),
        ),
        "gpu_uuid": os.environ.get("P4_GPU_UUID", "not_injected"),
        "completed_utc": p4.utc_now(),
    })
    p4.write_json(receipt_path, receipt)
    return receipt


def _stage_receipt_path(stage: str, arm: str, seed: int) -> Path:
    folder = "calibration" if stage == "calibration" else stage.replace("-", "_")
    return ANALYSIS_ROOT / folder / arm.replace("-", "_") / f"seed_{seed}" / "run_receipt.json"


def schedule_analysis_stage(stage: str, gpu_indices: Sequence[int], run_root: Path = RUN_ROOT, *, resume: bool = False) -> dict[str, Any]:
    if stage not in {"calibration", "reload-check", "evaluation"}:
        raise ValueError(stage)
    if not gpu_indices or len(gpu_indices) > GPU_COUNT:
        raise ValueError(f"select 1..{GPU_COUNT} assigned G1 GPUs")
    if stage == "calibration" and not MODEL_LOCK_PATH.is_file():
        raise RuntimeError("all terminal models must be locked before calibration")
    if stage == "calibration":
        _cal_rows(run_root)
    if stage in {"reload-check", "evaluation"} and not CAL_THRESHOLD_LOCK.is_file():
        raise RuntimeError("CAL thresholds must be locked before model reload checks")
    stage_receipts = {
        "calibration": ANALYSIS_ROOT / "CALIBRATION_SCHEDULER.json",
        "reload-check": ANALYSIS_ROOT / "RELOAD_CHECK_SCHEDULER.json",
        "evaluation": ANALYSIS_ROOT / "EVALUATION_SCHEDULER.json",
    }
    result_path = stage_receipts[stage]
    if result_path.exists():
        raise FileExistsError(result_path)
    queue: list[dict[str, Any]] = []
    completed_existing: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            receipt_path = _stage_receipt_path(stage, arm, seed)
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("status") in {"CAL_PREDICTIONS_COMPLETE", "CAL_RELOAD_CHECK_PASS", "LOCKED_EVALUATION_COMPLETE"}:
                    completed_existing.append({"arm": arm, "seed": seed, "receipt_path": str(receipt_path), "allocated_gpu_wall_seconds": receipt.get("allocated_gpu_wall_seconds", 0.0)})
                    continue
            queue.append({"arm": arm, "seed": seed})
    if completed_existing and not resume:
        raise RuntimeError(f"partial {stage} analysis artifacts exist; use --resume to finish the same locked stage")
    gpu_info = p4._live_gpu_records()
    selected = [next((g for g in gpu_info if g["index"] == int(i)), None) for i in gpu_indices]
    if any(g is None or g["utilization_percent"] > 5 or g["memory_used_mib"] > 512 for g in selected):
        raise RuntimeError("one or more requested G1 GPUs are not currently idle")
    task_dir = ANALYSIS_ROOT / stage.replace("-", "_")
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = ANALYSIS_ROOT / f"{stage.replace('-', '_')}_scheduler.jsonl"
    processes: dict[int, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_job = 0
    started = time.perf_counter()
    cpu_started = p4._process_cpu_seconds("self")

    def launch(gpu: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--stage", stage, "--arm", job["arm"], "--seed", str(job["seed"]), "--run-root", str(run_root)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        env["P4_GPU_UUID"] = str(gpu["uuid"])
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        stdout_path = task_dir / f"{job['arm'].replace('-', '_')}_seed{job['seed']}.stdout.log"
        stdout = stdout_path.open("ab", buffering=0)
        proc = subprocess.Popen(command, cwd=p4.REPO, env=env, stdout=stdout, stderr=subprocess.STDOUT, close_fds=True)
        p4.append_jsonl(log_path, {"event": "launch", "stage": stage, "arm": job["arm"], "seed": job["seed"], "gpu_uuid": gpu["uuid"], "pid": proc.pid, "utc": p4.utc_now()})
        return {"process": proc, "stdout": stdout, "job": job, "gpu": gpu, "started": time.perf_counter()}

    for gpu in selected:
        if next_job < len(queue):
            job = queue[next_job]
            next_job += 1
            processes[int(gpu["index"])] = launch(gpu, job)
    while processes:
        ended: list[int] = []
        for index, entry in list(processes.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["stdout"].close()
            elapsed = time.perf_counter() - entry["started"]
            job = entry["job"]
            receipt_path = _stage_receipt_path(stage, job["arm"], job["seed"])
            if code != 0 or not receipt_path.is_file():
                failure = {
                    "arm": job["arm"], "seed": job["seed"], "return_code": code,
                    "gpu_uuid": entry["gpu"]["uuid"], "elapsed_gpu_seconds": elapsed,
                    "failed_worker_cpu_core_seconds_upper_bound": elapsed * (p4.WORKER_COUNT + 1),
                }
                failures.append(failure)
                p4.append_jsonl(log_path, {"event": "worker_failure", **failure, "utc": p4.utc_now()})
                raise RuntimeError(f"P4 {stage} worker failed: {failure}; see stdout log {task_dir}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            item = {"arm": job["arm"], "seed": job["seed"], "gpu_uuid": entry["gpu"]["uuid"], "allocated_gpu_wall_seconds": elapsed, "receipt_path": str(receipt_path), "status": receipt["status"]}
            ledger.append(item)
            p4.append_jsonl(log_path, {"event": "complete", **item, "utc": p4.utc_now()})
            ended.append(index)
        for index in ended:
            processes.pop(index, None)
            if next_job < len(queue):
                job = queue[next_job]
                next_job += 1
                gpu = next(g for g in selected if int(g["index"]) == index)
                processes[index] = launch(gpu, job)
        if processes:
            time.sleep(0.25)
    complete_keys = {(x["arm"], int(x["seed"])) for x in ledger} | {(x["arm"], int(x["seed"])) for x in completed_existing}
    result = {
        "status": stage.upper().replace("-", "_") + "_QUEUE_COMPLETE",
        "jobs_planned": len(ARMS) * len(SEEDS), "jobs_completed": len(complete_keys),
        "completed_this_schedule": len(ledger), "completed_before_resume": completed_existing,
        "failures": failures,
        "gpu_assignments": [{"index": g["index"], "uuid": g["uuid"], "name": g["name"]} for g in selected],
        "max_simultaneous_workers": len(selected),
        "allocated_gpu_seconds_sum": sum(float(x["allocated_gpu_wall_seconds"]) for x in ledger) + sum(float(x.get("allocated_gpu_wall_seconds", 0.0)) for x in completed_existing),
        "scheduler_wall_seconds": time.perf_counter() - started,
        "scheduler_cpu_core_seconds": max(0.0, p4._process_cpu_seconds("self") - cpu_started),
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "updated_utc": p4.utc_now(),
    }
    p4.write_json(result_path, result)
    return result


def _weighted_qwk_many(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum(axis=1)
    grade_axis = np.arange(5, dtype=np.float64)
    onehot_y = (y[:, None] == np.arange(5)[None, :]).astype(np.float64)
    onehot_p = (pred[:, None] == np.arange(5)[None, :]).astype(np.float64)
    row = w @ onehot_y
    col = w @ onehot_p
    observed = w @ ((y - pred).astype(np.float64) ** 2)
    row_moment1 = row @ grade_axis
    col_moment1 = col @ grade_axis
    row_moment2 = row @ (grade_axis ** 2)
    col_moment2 = col @ (grade_axis ** 2)
    expected = row_moment2 + col_moment2 - 2.0 * row_moment1 * col_moment1 / np.maximum(total, 1.0)
    return np.divide(1.0 - observed / np.maximum(expected, 1e-300), 1.0, out=np.full_like(observed, np.nan), where=expected > 0)


def _weighted_auc_many(y_binary: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y_binary, dtype=np.int8)
    scores = np.asarray(score, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_score = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    sorted_w = w[:, order]
    pos_group = np.add.reduceat(sorted_w * y[order][None, :], starts, axis=1)
    neg_group = np.add.reduceat(sorted_w * (1 - y[order])[None, :], starts, axis=1)
    neg_before = np.cumsum(neg_group, axis=1) - neg_group
    numerator = np.sum(pos_group * (neg_before + 0.5 * neg_group), axis=1)
    positive_total = np.sum(pos_group, axis=1)
    negative_total = np.sum(neg_group, axis=1)
    denominator = positive_total * negative_total
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _clinical_rates_many(y: np.ndarray, referrals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return [draw, model, metric] rates using one shared cluster draw."""
    y = np.asarray(y, dtype=np.int8)
    ref = np.asarray(referrals, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    n_draws, n_images = w.shape
    if ref.shape[0] != n_images:
        raise ValueError("fixed clinical referrals and bootstrap weights differ in image count")
    n_models = ref.shape[1]
    onehot = (y[:, None] == np.arange(5)[None, :]).astype(np.float64)
    denominators = w @ onehot
    by_model_grade = np.concatenate([
        (ref[:, model][:, None] & (y[:, None] == np.arange(5)[None, :])).astype(np.float64)
        for model in range(n_models)
    ], axis=1)
    numerators = (w @ by_model_grade).reshape(n_draws, n_models, 5)
    den = denominators[:, None, :]
    out = np.full((n_draws, n_models, len(CLINICAL_METRICS)), np.nan, dtype=np.float64)
    out[:, :, 0] = np.divide(numerators[:, :, 0], den[:, :, 0], out=np.full((n_draws, n_models), np.nan), where=den[:, :, 0] > 0)
    out[:, :, 1] = np.divide(numerators[:, :, 1], den[:, :, 1], out=np.full((n_draws, n_models), np.nan), where=den[:, :, 1] > 0)
    denom01 = den[:, :, 0] + den[:, :, 1]
    out[:, :, 2] = np.divide(numerators[:, :, 0] + numerators[:, :, 1], denom01, out=np.full((n_draws, n_models), np.nan), where=denom01 > 0)
    for metric_index, grade in ((3, 2), (4, 3), (5, 4)):
        out[:, :, metric_index] = np.divide(den[:, :, grade] - numerators[:, :, grade], den[:, :, grade], out=np.full((n_draws, n_models), np.nan), where=den[:, :, grade] > 0)
    denom234 = den[:, :, 2] + den[:, :, 3] + den[:, :, 4]
    misses234 = denom234 - numerators[:, :, 2] - numerators[:, :, 3] - numerators[:, :, 4]
    out[:, :, 6] = np.divide(misses234, denom234, out=np.full((n_draws, n_models), np.nan), where=denom234 > 0)
    out[:, :, 7] = np.divide(numerators[:, :, 2] + numerators[:, :, 3] + numerators[:, :, 4], denom234, out=np.full((n_draws, n_models), np.nan), where=denom234 > 0)
    out[:, :, 8] = np.divide(denom01 - numerators[:, :, 0] - numerators[:, :, 1], denom01, out=np.full((n_draws, n_models), np.nan), where=denom01 > 0)
    return out


def _cluster_weight_batches(
    n_clusters: int,
    cluster_index: np.ndarray,
    rng: np.random.Generator,
    *,
    attempt_cap: int = 20000,
    batch_size: int = 256,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Reproduce the original shared cluster-draw stream, returning weights and sampled indices."""
    attempts = 0
    while attempts < attempt_cap:
        batch = min(batch_size, attempt_cap - attempts)
        draws = rng.integers(0, n_clusters, size=(batch, n_clusters), endpoint=False, dtype=np.int32)
        cluster_weights = np.zeros((batch, n_clusters), dtype=np.uint16)
        for draw_index in range(batch):
            np.add.at(cluster_weights[draw_index], draws[draw_index], 1)
        yield cluster_weights[:, cluster_index], draws
        attempts += batch


def _benchmark_clinical_bootstrap_cpu(attempts: int = 512) -> dict[str, Any]:
    """Bound the planned two-cohort bootstrap cost on synthetic, dimension-matched inputs."""
    estimates: dict[str, Any] = {}
    total_estimated = 0.0
    for cohort, n_images, n_clusters, entropy_stream in (
        ("Retino", 1392, 1392, 11),
        ("SetB", 400, 100, 29),
    ):
        grade = (np.arange(n_images, dtype=np.int8) % 5).astype(np.int8)
        if n_clusters == n_images:
            cluster_index = np.arange(n_images, dtype=np.int32)
        else:
            cluster_index = np.repeat(np.arange(n_clusters, dtype=np.int32), n_images // n_clusters)
        referral_rng = np.random.Generator(np.random.PCG64(991 + entropy_stream))
        referrals = referral_rng.integers(0, 2, size=(n_images, len(ARMS) * len(SEEDS))).astype(bool)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261001, entropy_stream])))
        started = time.process_time()
        completed = 0
        for weights, _draws in _cluster_weight_batches(n_clusters, cluster_index, rng, attempt_cap=attempts):
            _clinical_rates_many(grade, referrals, weights)
            completed += len(weights)
        cpu_seconds = max(0.0, time.process_time() - started)
        expanded = cpu_seconds * (20000.0 / max(completed, 1))
        estimates[cohort] = {
            "synthetic_images": n_images,
            "synthetic_clusters": n_clusters,
            "models": len(ARMS) * len(SEEDS),
            "pilot_attempts": completed,
            "pilot_cpu_core_seconds": cpu_seconds,
            "estimated_20000_attempt_cpu_core_seconds": expanded,
        }
        total_estimated += expanded
    io_and_validation_allowance = 300.0
    estimate_with_allowance = total_estimated + io_and_validation_allowance
    reserved_with_10pct_slack = estimate_with_allowance * 1.10
    return {
        "method": "dimension-matched synthetic clinical-rate bootstrap; no CAL/evaluation outcomes read",
        "attempt_cap": 20000,
        "pilot_attempts_per_cohort": attempts,
        "cohorts": estimates,
        "combined_estimated_bootstrap_cpu_core_seconds": total_estimated,
        "io_validation_and_finalization_allowance_cpu_core_seconds": io_and_validation_allowance,
        "estimated_total_cpu_core_seconds": estimate_with_allowance,
        "completion_reserve_fraction": 0.10,
        "reserved_estimate_cpu_core_seconds": reserved_with_10pct_slack,
    }


def _validate_saved_prediction_archive(
    pred: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    *,
    context: str,
) -> None:
    expected_grade = np.asarray([int(row["grade"]) for row in rows], dtype=np.int8)
    expected_label = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    expected_row_index = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int32)
    expected_components = np.asarray([str(row["component_hash"]) for row in rows], dtype="U64")
    if not np.array_equal(expected_label, (expected_grade >= 2).astype(np.int8)):
        raise RuntimeError(f"frozen label is not grade>=2 in {context}")
    required = {"grade_probs", "q2", "grade", "row_index", "component_hash"}
    if not required.issubset(pred):
        raise RuntimeError(f"saved prediction fields missing in {context}: {sorted(required - set(pred))}")
    probs = np.asarray(pred["grade_probs"], dtype=np.float64)
    q2 = np.asarray(pred["q2"], dtype=np.float64)
    grade = np.asarray(pred["grade"], dtype=np.int8)
    row_index = np.asarray(pred["row_index"], dtype=np.int32)
    components = np.asarray(pred["component_hash"]).astype(str)
    if probs.shape != (len(rows), 5) or q2.shape != (len(rows),):
        raise RuntimeError(f"saved prediction dimensions differ from frozen rows in {context}")
    if not np.array_equal(grade, expected_grade):
        raise RuntimeError(f"saved grade labels differ from frozen manifest in {context}")
    if not np.array_equal(row_index, expected_row_index):
        raise RuntimeError(f"saved prediction row order differs from frozen manifest in {context}")
    if not np.array_equal(components, expected_components):
        raise RuntimeError(f"saved component identity differs from frozen manifest in {context}")
    if (
        not np.all(np.isfinite(probs))
        or not np.all(np.isfinite(q2))
        or np.any(probs < -1e-7)
        or np.any(probs > 1.0 + 1e-6)
        or np.any(q2 < -1e-7)
        or np.any(q2 > 1.0 + 1e-6)
        or not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
        or not np.allclose(q2, probs[:, 2:].sum(axis=1), rtol=1e-6, atol=1e-7)
    ):
        raise RuntimeError(f"non-finite or inconsistent saved probabilities/q2 in {context}")


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _check_correction_budget(cpu_start: float, wall_start: float, max_cpu_seconds: float, max_wall_seconds: float) -> None:
    if p4._process_cpu_seconds("self") - cpu_start > max_cpu_seconds:
        raise TimeoutError("clinical correction exceeded its bounded CPU allowance")
    if time.perf_counter() - wall_start > max_wall_seconds:
        raise TimeoutError("clinical correction exceeded its bounded wall allowance")


def _selected_curve_point(old_row: dict[str, Any]) -> dict[str, Any] | None:
    tau = old_row.get("tau_q2")
    if tau is None:
        return None
    return next((point for point in old_row.get("curve", []) if float(point["tau"]) == float(tau)), None)


def _clinical_bootstrap_correction(
    cohort: str,
    predictions: dict[tuple[str, int], dict[str, np.ndarray]],
    thresholds: dict[tuple[str, int], float | None],
    *,
    cpu_start: float,
    wall_start: float,
    max_cpu_seconds: float,
    max_wall_seconds: float,
) -> dict[str, Any]:
    reference = predictions[(ARMS[0], SEEDS[0])]
    y = np.asarray(reference["grade"], dtype=np.int8)
    components = np.asarray(reference["component_hash"], dtype="U64")
    for key, pred in predictions.items():
        if not np.array_equal(pred["grade"], y) or not np.array_equal(pred["component_hash"], components):
            raise RuntimeError(f"corrected clinical prediction alignment differs within {cohort}: {key}")
    unique_components, cluster_index = np.unique(components, return_inverse=True)
    n_clusters = len(unique_components)
    if n_clusters < 2:
        raise RuntimeError(f"cluster bootstrap requires at least two registered components for {cohort}")
    model_order = [(arm, seed) for arm in ARMS for seed in SEEDS]
    model_index = {key: i for i, key in enumerate(model_order)}
    referrals = np.column_stack([
        predictions[key]["q2"] >= float(thresholds[key]) if thresholds[key] is not None
        else np.zeros(len(y), dtype=bool)
        for key in model_order
    ])
    estimable_model = np.asarray([thresholds[key] is not None for key in model_order], dtype=bool)
    entropy_stream = 11 if cohort == "Retino" else 29
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261001, entropy_stream])))
    draw_digest = hashlib.sha256()
    clinical_chunks: list[np.ndarray] = []
    attempts = 0
    for weights, draw_indices in _cluster_weight_batches(n_clusters, cluster_index, rng):
        _check_correction_budget(cpu_start, wall_start, max_cpu_seconds, max_wall_seconds)
        draw_digest.update(draw_indices.tobytes(order="C"))
        rate = _clinical_rates_many(y, referrals, weights)
        rate[:, ~estimable_model, :] = np.nan
        clinical_chunks.append(rate.reshape(len(draw_indices), len(ARMS), len(SEEDS), len(CLINICAL_METRICS)))
        attempts += len(draw_indices)
    clinical_draws = np.concatenate(clinical_chunks, axis=0)
    arm_indices = {arm: ARMS.index(arm) for arm in ARMS}
    point_values: dict[tuple[str, str, int], float | None] = {}
    for model_i, key in enumerate(model_order):
        point = _referral_metrics(predictions[key]["grade"], predictions[key]["q2"], thresholds[key])
        for metric_name in CLINICAL_METRICS:
            point_values[(key[0], metric_name, key[1])] = point[metric_name]["rate"]

    summaries: dict[str, Any] = {"per_arm_equal_seed_mean": {}, "paired_M3_minus_C2_equal_seed_mean": {}}
    valid_draws_by_estimand: dict[str, int] = {}
    for metric_i, metric_name in enumerate(CLINICAL_METRICS):
        for arm in ARMS:
            rates = clinical_draws[:, arm_indices[arm], :, metric_i]
            valid = np.isfinite(rates).all(axis=1)
            values = rates[valid].mean(axis=1)
            summary = _percentile_summary(values)
            point_rates = [
                point_values.get((arm, metric_name, seed))
                for seed in SEEDS
            ]
            summary["point_estimate_equal_seed_mean"] = (
                float(np.mean(point_rates)) if all(value is not None for value in point_rates) else None
            )
            key = f"{cohort}:{arm}:{metric_name}"
            summaries["per_arm_equal_seed_mean"][key] = summary
            valid_draws_by_estimand[key] = int(summary["valid_draws"])

        m3_rates = clinical_draws[:, arm_indices["M3"], :, metric_i]
        c2_rates = clinical_draws[:, arm_indices["C2"], :, metric_i]
        valid = np.isfinite(m3_rates).all(axis=1) & np.isfinite(c2_rates).all(axis=1)
        values = (m3_rates[valid] - c2_rates[valid]).mean(axis=1)
        summary = _percentile_summary(values)
        per_seed_point: list[float | None] = []
        for seed in SEEDS:
            m3 = point_values.get(("M3", metric_name, seed))
            c2 = point_values.get(("C2", metric_name, seed))
            per_seed_point.append(float(m3 - c2) if m3 is not None and c2 is not None else None)
        summary["per_seed_point_deltas_M3_minus_C2"] = per_seed_point
        summary["point_estimate_equal_seed_mean"] = (
            float(np.mean(per_seed_point)) if all(value is not None for value in per_seed_point) else None
        )
        key = f"{cohort}:M3-C2:{metric_name}"
        summaries["paired_M3_minus_C2_equal_seed_mean"][key] = summary
        valid_draws_by_estimand[key] = int(summary["valid_draws"])
    return {
        "cohort": cohort,
        "attempts": attempts,
        "required_valid_draws": 10000,
        "attempt_cap": 20000,
        "n_images": int(len(y)),
        "n_clusters": int(n_clusters),
        "cluster_unit": "registered image component" if cohort == "Retino" else "registered patient",
        "rng_stream": f"PCG64(SeedSequence([20261001,{entropy_stream}]))",
        "batch_size": 256,
        "seed_resampling": False,
        "draw_index_sha256": draw_digest.hexdigest(),
        "draw_indices_persisted_in_original_run": False,
        "valid_draws_by_estimand": valid_draws_by_estimand,
        "summaries": summaries,
    }


def run_calibration_rule_correction(
    run_root: Path = RUN_ROOT,
    *,
    max_cpu_seconds: float = 3600.0,
    max_wall_seconds: float = 7200.0,
    prior_attempt_cpu_seconds: float = 0.0,
    prior_attempt_wall_seconds: float = 0.0,
    prior_other_cpu_seconds: float = 0.0,
    prior_other_wall_seconds: float = 0.0,
) -> dict[str, Any]:
    """Rebuild the adopted CAL-only threshold rule and clinical statistics from saved predictions only."""
    run_root = Path(run_root)
    analysis_root = run_root / "analysis"
    correction_dir = analysis_root / CAL_RULE_CORRECTION_ID
    if correction_dir.exists():
        raise FileExistsError(f"correction output already exists and will not be overwritten: {correction_dir}")
    started_utc = p4.utc_now()
    wall_start = time.perf_counter()
    cpu_start = p4._process_cpu_seconds("self")
    prior_cpu_seconds = prior_attempt_cpu_seconds + prior_other_cpu_seconds
    prior_wall_seconds = prior_attempt_wall_seconds + prior_other_wall_seconds
    current_cpu_limit = max(0.0, max_cpu_seconds - prior_cpu_seconds)
    current_wall_limit = max(0.0, max_wall_seconds - prior_wall_seconds)
    run_contract_path = run_root / "EXECUTION_CONTRACT.json"
    model_lock_path = run_root / "FINAL_MODELS_LOCKED.json"
    cal_input = run_root / "inputs/cal_eye_manifest.csv"
    old_lock_path = run_root / "CAL_THRESHOLDS_LOCKED.json"
    eval_input_receipt_path = run_root / "evaluation_inputs/EVALUATION_INPUT_RECEIPT.json"
    completion_path = analysis_root / "P4_EXECUTION_COMPLETE.json"
    incidents_path = analysis_root / "P4_ANALYSIS_INCIDENTS.json"
    old_stats_path = analysis_root / "P4_STATISTICAL_SUMMARY.json"
    old_metrics_path = analysis_root / "METRICS_BY_RUN.json"
    required_paths = (
        run_contract_path, model_lock_path, cal_input, old_lock_path,
        eval_input_receipt_path, completion_path, incidents_path, old_stats_path, old_metrics_path,
        analysis_root / "CALIBRATION_SCHEDULER.json",
        analysis_root / "RELOAD_CHECK_SCHEDULER.json",
        analysis_root / "EVALUATION_SCHEDULER.json",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P4 correction inputs missing: {missing}")
    preserved_paths = {
        "CAL_THRESHOLDS_LOCKED.json": old_lock_path,
        "P4_STATISTICAL_SUMMARY.json": old_stats_path,
        "METRICS_BY_RUN.json": old_metrics_path,
        "P4_EXECUTION_COMPLETE.json": completion_path,
        "P4_ANALYSIS_INCIDENTS.json": incidents_path,
        "FINAL_MODELS_LOCKED.json": model_lock_path,
    }
    protected_hashes_before = {name: p4.sha256_file(path) for name, path in preserved_paths.items()}

    cost_estimate = _benchmark_clinical_bootstrap_cpu()
    if cost_estimate["reserved_estimate_cpu_core_seconds"] > current_cpu_limit:
        raise RuntimeError(
            "bounded CPU budget is below the pilot-based estimate plus 10% completion reserve: "
            f"{cost_estimate['reserved_estimate_cpu_core_seconds']:.3f}>{current_cpu_limit:.3f}"
        )
    _check_correction_budget(cpu_start, wall_start, current_cpu_limit, current_wall_limit)

    contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    contract_sha = str(contract["contract_sha256"])
    if contract_sha != p4._contract_hash_from_run(run_root):
        raise RuntimeError("execution contract identity mismatch")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    incidents = json.loads(incidents_path.read_text(encoding="utf-8"))
    old_statistics = json.loads(old_stats_path.read_text(encoding="utf-8"))
    cal_scheduler = json.loads((analysis_root / "CALIBRATION_SCHEDULER.json").read_text(encoding="utf-8"))
    reload_scheduler = json.loads((analysis_root / "RELOAD_CHECK_SCHEDULER.json").read_text(encoding="utf-8"))
    eval_scheduler = json.loads((analysis_root / "EVALUATION_SCHEDULER.json").read_text(encoding="utf-8"))
    if any(scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS) for scheduler in (cal_scheduler, reload_scheduler, eval_scheduler)):
        raise RuntimeError("P4 CAL/reload/evaluation scheduler receipts must all be complete")
    if completion.get("status") != "P4_TRAINING_CAL_DEV_EXTERNAL_AND_STATISTICS_COMPLETE":
        raise RuntimeError("original P4 completion receipt is not complete")
    incident_audit = incidents.get("analysis_code_fingerprint_audit", {})
    actual_old_analysis_sha = str(completion.get("analysis_code_sha256", ""))
    if (
        actual_old_analysis_sha != CORRECTION_BASELINE_ANALYSIS_SHA256
        or incidents.get("analysis_code_sha256") != CORRECTION_BASELINE_ANALYSIS_SHA256
        or incident_audit.get("final_analysis_source_sha256") != CORRECTION_BASELINE_ANALYSIS_SHA256
    ):
        raise RuntimeError("actual P4 analysis source does not match the K1-verified threshold implementation")
    old_lock_sha = p4.sha256_file(old_lock_path)
    if completion.get("cal_threshold_lock_sha256") != old_lock_sha:
        raise RuntimeError("old threshold lock hash does not match P4 completion receipt")
    if old_statistics.get("cal_threshold_lock_sha256") != old_lock_sha:
        raise RuntimeError("old statistical summary is not bound to the preserved threshold lock")
    if old_statistics.get("contract_sha256") != contract_sha:
        raise RuntimeError("old statistical summary contract hash mismatch")
    if old_lock.get("contract_sha256") != contract_sha:
        raise RuntimeError("preserved CAL threshold lock contract hash mismatch")
    if old_lock.get("cal_manifest_sha256") != p4.sha256_file(cal_input):
        raise RuntimeError("preserved CAL threshold lock manifest hash mismatch")

    old_lock_by_model = {(row["arm"], int(row["seed"])): row for row in old_lock["models"]}
    model_by_run = {(row["arm"], int(row["seed"])): row for row in model_lock["models"]}
    expected_model_keys = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(old_lock_by_model) != expected_model_keys or set(model_by_run) != expected_model_keys:
        raise RuntimeError("old threshold/model lock does not enumerate exactly the 35 registered models")

    input_hashes: list[dict[str, Any]] = []
    cal_rows = _read_frozen_csv(cal_input)
    cal_manifest_sha = p4.sha256_file(cal_input)
    cal_positive = sum(int(row["label"]) == 1 for row in cal_rows)
    cal_negative = sum(int(row["label"]) == 0 for row in cal_rows)
    if cal_positive != 489 or cal_negative != 2035 or len(cal_rows) != 2524:
        raise RuntimeError(f"frozen CAL support differs from the locked P4 source cohort: {len(cal_rows)}/{cal_positive}/{cal_negative}")
    corrected_models: list[dict[str, Any]] = []
    corrected_thresholds: dict[tuple[str, int], float | None] = {}
    threshold_comparison_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            _check_correction_budget(cpu_start, wall_start, current_cpu_limit, current_wall_limit)
            key = (arm, seed)
            model = model_by_run[key]
            cal_dir = analysis_root / "calibration" / arm.replace("-", "_") / f"seed_{seed}"
            cal_receipt_path = cal_dir / "run_receipt.json"
            pred_path = cal_dir / "cal_predictions.npz"
            cal_receipt = json.loads(cal_receipt_path.read_text(encoding="utf-8"))
            if (
                cal_receipt.get("status") != "CAL_PREDICTIONS_COMPLETE"
                or cal_receipt.get("contract_sha256") != contract_sha
                or cal_receipt.get("cal_manifest_sha256") != cal_manifest_sha
                or int(cal_receipt.get("cal_image_count", -1)) != len(cal_rows)
                or cal_receipt.get("model_sha256") != model.get("model_sha256")
                or Path(cal_receipt.get("prediction_path", "")) != pred_path
            ):
                raise RuntimeError(f"CAL prediction receipt identity mismatch for {arm}/{seed}")
            prediction_sha = p4.sha256_file(pred_path)
            if prediction_sha != cal_receipt.get("prediction_sha256"):
                raise RuntimeError(f"CAL prediction hash mismatch for {arm}/{seed}")
            pred = _load_npz_arrays(pred_path)
            _validate_saved_prediction_archive(pred, cal_rows, context=f"CAL/{arm}/{seed}")
            labels = np.asarray([int(row["label"]) for row in cal_rows], dtype=np.int8)
            selection = select_calibration_threshold(pred["q2"], labels)
            corrected_thresholds[key] = selection["tau_q2"]
            selected = selection.get("selected")
            corrected_model = {
                "arm": arm,
                "seed": seed,
                "status": selection["status"],
                "tau_q2": selection["tau_q2"],
                "cal_positive_images": selection["cal_positive_images"],
                "cal_negative_images": selection["cal_negative_images"],
                "candidate_count": selection["candidate_count"],
                "selected_true_positive": selected["true_positive"] if selected else None,
                "selected_sensitivity": selected["sensitivity"] if selected else None,
                "selected_wilson_lower_90": selected["wilson_lower_90"] if selected else None,
                "cal_predicted_positive": selected["predicted_positive"] if selected else None,
                "cal_predicted_negative": selected["predicted_negative"] if selected else None,
                "all_positive_on_cal": bool(selected and selected["predicted_positive"] == len(cal_rows)),
                "criterion": selection.get("criterion", CAL_THRESHOLD_CRITERION),
                "wilson_z": selection.get("wilson_z", 1.2815515655),
                "prediction_sha256": prediction_sha,
                "model_sha256": model["model_sha256"],
                "reason": selection.get("reason"),
            }
            corrected_models.append(corrected_model)
            old_model = old_lock_by_model[key]
            old_selected = _selected_curve_point(old_model)
            threshold_comparison_rows.append({
                "arm": arm,
                "seed": seed,
                "old_status": old_model["status"],
                "old_tau_q2": old_model["tau_q2"],
                "old_cal_predicted_positive": old_selected.get("predicted_positive") if old_selected else None,
                "old_cal_predicted_negative": old_selected.get("predicted_negative") if old_selected else None,
                "new_status": corrected_model["status"],
                "new_tau_q2": corrected_model["tau_q2"],
                "new_cal_predicted_positive": corrected_model["cal_predicted_positive"],
                "new_cal_predicted_negative": corrected_model["cal_predicted_negative"],
                "new_cal_all_positive": corrected_model["all_positive_on_cal"],
                "new_cal_sensitivity": corrected_model["selected_sensitivity"],
                "new_cal_wilson_lower_90": corrected_model["selected_wilson_lower_90"],
                "candidate_count": corrected_model["candidate_count"],
            })
            input_hashes.append({"kind": "CAL_prediction", "arm": arm, "seed": seed, "sha256": prediction_sha, "bytes": pred_path.stat().st_size})

    corrected_lock = {
        "status": "CORRECTED_CAL_THRESHOLDS_COMPLETE_POST_EXPOSURE_RECONSTRUCTION",
        "correction_id": CAL_RULE_CORRECTION_ID,
        "contract_sha256": contract_sha,
        "model_lock_sha256": p4.sha256_file(model_lock_path),
        "old_cal_threshold_lock_sha256": old_lock_sha,
        "cal_manifest_sha256": cal_manifest_sha,
        "cal_image_count": len(cal_rows),
        "cal_positive_images": cal_positive,
        "cal_negative_images": cal_negative,
        "score": "q2=p2+p3+p4",
        "positive_rule": "q2 >= tau_q2",
        "criterion": CAL_THRESHOLD_CRITERION,
        "wilson_z": 1.2815515655,
        "thresholds_chosen_using_only_eyePACS_CAL": True,
        "external_results_were_already_opened_before_this_correction": True,
        "prospective_or_blind_confirmation": False,
        "models": corrected_models,
        "recomputed_utc": p4.utc_now(),
    }

    eval_input_receipt = json.loads(eval_input_receipt_path.read_text(encoding="utf-8"))
    eval_input_receipt_sha = p4.sha256_file(eval_input_receipt_path)
    eval_rows_by_cohort: dict[str, list[dict[str, str]]] = {}
    cohort_predictions: dict[str, dict[tuple[str, int], dict[str, np.ndarray]]] = {"Retino": {}, "SetB": {}}
    point_estimates: list[dict[str, Any]] = []
    for cohort in ("DEV", "Retino", "SetB"):
        input_path = run_root / "evaluation_inputs" / f"{cohort}.csv"
        input_sha = p4.sha256_file(input_path)
        if eval_input_receipt.get(f"frozen_{cohort}_csv_sha256") != input_sha:
            raise RuntimeError(f"frozen evaluation manifest hash mismatch for {cohort}")
        rows = _read_frozen_csv(input_path)
        eval_rows_by_cohort[cohort] = rows
        expected_images = {"DEV": 3004, "Retino": 1392, "SetB": 400}[cohort]
        if len(rows) != expected_images:
            raise RuntimeError(f"unexpected frozen evaluation row count for {cohort}: {len(rows)}")
    for arm in ARMS:
        for seed in SEEDS:
            _check_correction_budget(cpu_start, wall_start, current_cpu_limit, current_wall_limit)
            key = (arm, seed)
            eval_receipt_path = analysis_root / "evaluation" / arm.replace("-", "_") / f"seed_{seed}" / "run_receipt.json"
            eval_receipt = json.loads(eval_receipt_path.read_text(encoding="utf-8"))
            old_model = old_lock_by_model[key]
            if (
                eval_receipt.get("status") != "LOCKED_EVALUATION_COMPLETE"
                or eval_receipt.get("contract_sha256") != contract_sha
                or eval_receipt.get("model_sha256") != model_by_run[key].get("model_sha256")
                or eval_receipt.get("cal_threshold_lock_sha256") != old_lock_sha
                or eval_receipt.get("threshold_status") != old_model.get("status")
                or eval_receipt.get("tau_q2") != old_model.get("tau_q2")
            ):
                raise RuntimeError(f"evaluation receipt is not bound to the preserved old threshold for {arm}/{seed}")
            for cohort in ("DEV", "Retino", "SetB"):
                rows = eval_rows_by_cohort[cohort]
                path = analysis_root / "evaluation" / arm.replace("-", "_") / f"seed_{seed}" / f"{cohort}_predictions.npz"
                receipt_entry = eval_receipt.get("prediction_files", {}).get(cohort, {})
                prediction_sha = p4.sha256_file(path)
                if (
                    receipt_entry.get("sha256") != prediction_sha
                    or Path(receipt_entry.get("path", "")) != path
                    or int(receipt_entry.get("n_images", -1)) != len(rows)
                ):
                    raise RuntimeError(f"saved evaluation prediction receipt mismatch for {cohort}/{arm}/{seed}")
                pred = _load_npz_arrays(path)
                _validate_saved_prediction_archive(pred, rows, context=f"{cohort}/{arm}/{seed}")
                tau = corrected_thresholds[key]
                metrics = _referral_metrics(pred["grade"], pred["q2"], tau)
                all_referred = bool(np.all(pred["q2"] >= float(tau))) if tau is not None else None
                point_estimates.append({
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                    "n_images": len(rows),
                    "n_registered_components": int(len(np.unique(pred["component_hash"]))),
                    "threshold_status": next(row["status"] for row in corrected_models if row["arm"] == arm and row["seed"] == seed),
                    "tau_q2": tau,
                    "all_positive_at_threshold": all_referred,
                    "clinical_counts": metrics,
                })
                if cohort != "DEV":
                    cohort_predictions[cohort][key] = {
                        "grade": np.asarray(pred["grade"], dtype=np.int8),
                        "q2": np.asarray(pred["q2"], dtype=np.float64),
                        "component_hash": np.asarray(pred["component_hash"]).astype("U64"),
                    }
                input_hashes.append({"kind": f"{cohort}_prediction", "arm": arm, "seed": seed, "sha256": prediction_sha, "bytes": path.stat().st_size})

    threshold_by_model = corrected_thresholds
    clinical_bootstrap = {
        cohort: _clinical_bootstrap_correction(
            cohort,
            cohort_predictions[cohort],
            threshold_by_model,
            cpu_start=cpu_start,
            wall_start=wall_start,
            max_cpu_seconds=current_cpu_limit,
            max_wall_seconds=current_wall_limit,
        )
        for cohort in ("Retino", "SetB")
    }
    clinical_statistics = {
        "status": "CORRECTED_CLINICAL_STATISTICS_COMPLETE",
        "correction_id": CAL_RULE_CORRECTION_ID,
        "contract_sha256": contract_sha,
        "model_lock_sha256": p4.sha256_file(model_lock_path),
        "old_cal_threshold_lock_sha256": old_lock_sha,
        "corrected_cal_threshold_lock_sha256": p4.sha256_bytes(p4.canonical_json_bytes(corrected_lock)),
        "evaluation_input_receipt_sha256": eval_input_receipt_sha,
        "clinical_metrics": list(CLINICAL_METRICS),
        "point_estimates": point_estimates,
        "bootstrap": {
            "method": "paired cluster percentile intervals; linear quantile interpolation",
            "repetitions_required": 10000,
            "attempt_cap": 20000,
            "seed_resampling": False,
            "point_estimator_not_bootstrap_mean": True,
            "original_draw_index_files": "NOT_SAVED; reconstructed from the original fixed PCG64 streams and exact registered cluster counts",
            "clinical": clinical_bootstrap,
        },
        "clinical_intervals_are_estimation_only": True,
        "clinical_noninferiority_or_safety_pass": False,
        "not_a_new_blind_confirmation": True,
        "recomputed_utc": p4.utc_now(),
    }

    for item in input_hashes:
        if item["kind"] == "CAL_prediction":
            path = analysis_root / "calibration" / item["arm"].replace("-", "_") / f"seed_{item['seed']}" / "cal_predictions.npz"
        else:
            cohort = item["kind"].removesuffix("_prediction")
            path = analysis_root / "evaluation" / item["arm"].replace("-", "_") / f"seed_{item['seed']}" / f"{cohort}_predictions.npz"
        if p4.sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"saved prediction changed during clinical correction: {path}")
    protected_hashes_after = {name: p4.sha256_file(path) for name, path in preserved_paths.items()}
    if protected_hashes_before != protected_hashes_after:
        raise RuntimeError("a preserved P4 lock/statistic/model artifact changed during correction")
    _check_correction_budget(cpu_start, wall_start, current_cpu_limit, current_wall_limit)

    status_counts: dict[str, int] = {}
    old_status_counts: dict[str, int] = {}
    for row in corrected_models:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    for row in old_lock["models"]:
        old_status_counts[row["status"]] = old_status_counts.get(row["status"], 0) + 1
    tsv_columns = [
        "arm", "seed", "old_status", "old_tau_q2", "old_cal_predicted_positive", "old_cal_predicted_negative",
        "new_status", "new_tau_q2", "new_cal_predicted_positive", "new_cal_predicted_negative", "new_cal_all_positive",
        "new_cal_sensitivity", "new_cal_wilson_lower_90", "candidate_count",
    ]
    tsv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(tsv_buffer, fieldnames=tsv_columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(threshold_comparison_rows)

    correction_dir.mkdir(parents=True, exist_ok=False)
    corrected_lock_path = correction_dir / "CORRECTED_CAL_THRESHOLDS.json"
    clinical_path = correction_dir / "CORRECTED_CLINICAL_STATISTICS.json"
    comparison_path = correction_dir / "THRESHOLD_COMPARISON.tsv"
    receipt_path = correction_dir / "CORRECTION_RECEIPT.json"
    p4.write_json(corrected_lock_path, corrected_lock)
    p4.write_json(clinical_path, clinical_statistics)
    comparison_path.write_text(tsv_buffer.getvalue(), encoding="utf-8")
    now_cpu = p4._process_cpu_seconds("self")
    now_wall = time.perf_counter()
    known_resource = completion.get("resource_accounting", {})
    cpu_cap_seconds = float(known_resource.get("hard_caps", {}).get("cpu_core_hours", 300.0)) * 3600.0
    known_cpu_seconds = float(known_resource.get("known_cpu_core_seconds_including_preparation_gate_input_freeze_and_prior_failed_upper_bound", 0.0))
    receipt = {
        "status": "CAL_RULE_CORRECTION_AND_CLINICAL_RECOMPUTATION_COMPLETE_WITH_HISTORICAL_CPU_GAP_RECORDED",
        "correction_id": CAL_RULE_CORRECTION_ID,
        "plan_commit": "4e0696dfd719f8f7973f25d5f43ab8800145f533",
        "plan_blob_oid": "8c435662699cc7add99d55006ea209da66a44ae2",
        "plan_source_sha256": "96602ffd9457795a300d2705f0bdd5b57146ae747776bfced8ed274a3115a0b6",
        "evidence_branch": "codex/reference_data",
        "evidence_parent_commit": "ddcc92617e51a348beff2fdea1a9b16abc6b4f69",
        "p4_run_root": str(run_root),
        "started_utc": started_utc,
        "completed_utc": p4.utc_now(),
        "contract_sha256": contract_sha,
        "model_lock_sha256": protected_hashes_before["FINAL_MODELS_LOCKED.json"],
        "old_cal_threshold_lock_sha256": old_lock_sha,
        "old_cal_status_counts": old_status_counts,
        "corrected_cal_status_counts": status_counts,
        "old_actual_analysis_source_sha256": actual_old_analysis_sha,
        "preflight_analysis_source_sha256": incident_audit.get("contract_preflight_analysis_source_sha256"),
        "corrected_analysis_source_sha256": p4.sha256_file(Path(__file__).resolve()),
        "error_origin_commit": "84910d806274e3c6b63bc996ecfb4ed8d3245e7b",
        "error_origin_fields": ["scripts/dr/sparrow/p4_analysis.py:318", "scripts/dr/sparrow/p4_analysis.py:331"],
        "correction": "Removed the non-contractual predicted_negative>=1 feasibility condition; retained the source-CAL-only one-sided 90% Wilson lower sensitivity rule, candidate set, max-tau selection, tie handling and support states.",
        "external_results_open_before_correction": True,
        "new_blind_or_prospective_confirmation": False,
        "saved_prediction_files": {"CAL": 35, "DEV_Retino_SetB": 105, "total": len(input_hashes)},
        "saved_prediction_hashes": input_hashes,
        "cal_manifest_sha256": cal_manifest_sha,
        "evaluation_input_receipt_sha256": eval_input_receipt_sha,
        "original_bootstrap_draw_arrays": "NOT_SAVED; reconstructed using frozen PCG64 streams, cohort cluster counts, 256-row batches, int32 index draws, uint16 cluster multiplicities, 10000-valid/20000-attempt rule.",
        "protected_original_hashes_before_and_after": {"before": protected_hashes_before, "after": protected_hashes_after},
        "correction_cpu_preflight": cost_estimate,
        "actual_process_cpu_core_seconds_before_receipt_write": max(0.0, now_cpu - cpu_start),
        "actual_wall_seconds_before_receipt_write": max(0.0, now_wall - wall_start),
        "bounded_task_limits": {"max_cpu_core_seconds": max_cpu_seconds, "max_wall_seconds": max_wall_seconds},
        "prior_failed_attempts": [
            {
                "cause": "clinical-only bootstrap array was indexed as 4D before reshaping from the 3D [draw,model,metric] rate result",
                "result": "failed before writing correction artifacts; original lock, predictions and statistical outputs were untouched",
                "wall_seconds": prior_attempt_wall_seconds,
                "cpu_core_seconds": prior_attempt_cpu_seconds,
                "training_updates": 0,
                "model_loads": 0,
                "gpu_seconds": 0,
                "artifact_files_written": 0,
            }
        ] if prior_attempt_cpu_seconds or prior_attempt_wall_seconds else [],
        "prior_nonfailure_work": {
            "cpu_core_seconds": prior_other_cpu_seconds,
            "wall_seconds": prior_other_wall_seconds,
            "description": "separate 512-draw-per-cohort synthetic sizing pilot performed before the failed end-to-end attempt",
        } if prior_other_cpu_seconds or prior_other_wall_seconds else None,
        "current_attempt_cpu_limit_seconds": current_cpu_limit,
        "current_attempt_wall_limit_seconds": current_wall_limit,
        "estimated_cumulative_cpu_with_10pct_reserve_seconds": prior_cpu_seconds + cost_estimate["reserved_estimate_cpu_core_seconds"],
        "historical_resource_accounting": {
            "cpu_core_hour_cap": cpu_cap_seconds / 3600.0,
            "previously_counted_cpu_core_hours": known_cpu_seconds / 3600.0,
            "known_headroom_before_correction_cpu_core_hours": max(0.0, cpu_cap_seconds - known_cpu_seconds) / 3600.0,
            "unmeasured_previous_coordinator_bootstrap_cpu": "UNBOUNDED_FROM_RECORDED_LOGS; excluded from prior known sum and not treated as zero",
            "cumulative_budget_note": "The corrected run plus recorded failed implementation attempt is explicitly bounded by the per-task CPU/wall limits and includes a 10% preflight completion reserve; the historical uninstrumented coordinator segment prevents claiming an exact cumulative CPU total.",
        },
        "prohibited_actions": {"training_updates": 0, "optimizer_calls": 0, "model_loads": 0, "gpu_seconds": 0, "image_decodes": 0, "model_forwards": 0, "qwk_auroc_or_factorial_recomputation": 0},
        "nonclinical_result_artifacts_unchanged": True,
        "outputs": {},
    }
    output_paths = {
        "corrected_thresholds": corrected_lock_path,
        "clinical_statistics": clinical_path,
        "threshold_comparison": comparison_path,
    }
    receipt["outputs"] = {name: {"path": str(path), "sha256": p4.sha256_file(path), "bytes": path.stat().st_size} for name, path in output_paths.items()}
    receipt["logical_bytes_written_before_receipt"] = sum(item["bytes"] for item in receipt["outputs"].values())
    p4.write_json(receipt_path, receipt)
    result = {
        "status": receipt["status"],
        "correction_dir": str(correction_dir),
        "old_cal_status_counts": old_status_counts,
        "corrected_cal_status_counts": status_counts,
        "threshold_comparison_sha256": p4.sha256_file(comparison_path),
        "corrected_thresholds_sha256": p4.sha256_file(corrected_lock_path),
        "clinical_statistics_sha256": p4.sha256_file(clinical_path),
        "correction_receipt_sha256": p4.sha256_file(receipt_path),
        "actual_process_cpu_core_seconds": receipt["actual_process_cpu_core_seconds_before_receipt_write"],
        "actual_wall_seconds": receipt["actual_wall_seconds_before_receipt_write"],
    }
    return result


def _bootstrap_cohort(cohort: str, predictions: dict[tuple[str, int], dict[str, np.ndarray]], threshold_by_model: dict[tuple[str, int], float | None]) -> dict[str, Any]:
    reference = predictions[(ARMS[0], SEEDS[0])]
    y = np.asarray(reference["grade"], dtype=np.int8)
    components = np.asarray(reference["component_hash"], dtype="U64")
    for key, pred in predictions.items():
        if not np.array_equal(pred["grade"], y) or not np.array_equal(pred["component_hash"], components):
            raise RuntimeError(f"evaluation row alignment differs within {cohort}: {key}")
    unique_components, cluster_index = np.unique(components, return_inverse=True)
    n_clusters = len(unique_components)
    if n_clusters < 2:
        raise RuntimeError(f"cluster bootstrap requires at least two registered components for {cohort}")
    model_order = [(arm, seed) for arm in ARMS for seed in SEEDS]
    model_idx = {key: i for i, key in enumerate(model_order)}
    grade_predictions = np.column_stack([predictions[key]["grade_pred"] for key in model_order]).astype(np.int8)
    q2_scores = np.column_stack([predictions[key]["q2"] for key in model_order]).astype(np.float64)
    referrals = np.column_stack([
        (predictions[key]["q2"] >= float(threshold_by_model[key])) if threshold_by_model[key] is not None else np.zeros(len(y), dtype=bool)
        for key in model_order
    ])
    q2_binary = (y >= 2).astype(np.int8)
    entropy_stream = 11 if cohort == "Retino" else 29
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261001, entropy_stream])))
    qwk_chunks: list[np.ndarray] = []
    auc_chunks: list[np.ndarray] = []
    clinical_chunks: list[np.ndarray] = []
    attempts = 0
    while attempts < 20000:
        batch = min(256, 20000 - attempts)
        draws = rng.integers(0, n_clusters, size=(batch, n_clusters), endpoint=False, dtype=np.int32)
        cluster_weights = np.zeros((batch, n_clusters), dtype=np.uint16)
        for b in range(batch):
            np.add.at(cluster_weights[b], draws[b], 1)
        weights = cluster_weights[:, cluster_index]
        qwk = np.full((batch, len(ARMS), len(SEEDS)), np.nan, dtype=np.float64)
        auc = np.full_like(qwk, np.nan)
        for model_i, (arm, seed) in enumerate(model_order):
            a_i = ARMS.index(arm)
            s_i = SEEDS.index(seed)
            qwk[:, a_i, s_i] = _weighted_qwk_many(y, grade_predictions[:, model_i], weights)
            auc[:, a_i, s_i] = _weighted_auc_many(q2_binary, q2_scores[:, model_i], weights)
        rate = _clinical_rates_many(y, referrals, weights).reshape(batch, len(ARMS), len(SEEDS), len(CLINICAL_METRICS))
        for model_i, key in enumerate(model_order):
            if threshold_by_model[key] is None:
                rate[:, model_i // len(SEEDS), model_i % len(SEEDS), :] = np.nan
        qwk_chunks.append(qwk)
        auc_chunks.append(auc)
        clinical_chunks.append(rate)
        attempts += batch
    qwk_all = np.concatenate(qwk_chunks, axis=0)
    auc_all = np.concatenate(auc_chunks, axis=0)
    clinical_all = np.concatenate(clinical_chunks, axis=0)
    return {
        "cohort": cohort, "attempts": attempts, "n_images": int(len(y)), "n_clusters": int(n_clusters),
        "model_order": model_order, "model_index": model_idx,
        "qwk_draws": qwk_all, "auroc_draws": auc_all, "clinical_rate_draws": clinical_all,
        "cluster_unit": "registered image component" if cohort == "Retino" else "registered patient",
        "rng_stream": f"PCG64(SeedSequence([20261001,{entropy_stream}]))",
    }


def _percentile_summary(values: np.ndarray, *, required: int = 10000) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < required:
        return {"status": "INSUFFICIENT_VALID_DRAWS", "valid_draws": int(len(finite)), "attempt_cap": 20000, "ci95": None}
    return {
        "status": "ESTIMABLE", "valid_draws": int(len(finite)),
        "ci95": [float(np.quantile(finite[:required], 0.025, method="linear")), float(np.quantile(finite[:required], 0.975, method="linear"))],
    }


def run_statistics(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
    eval_receipt = json.loads((run_root / "evaluation_inputs/EVALUATION_INPUT_RECEIPT.json").read_text(encoding="utf-8"))
    thresholds = json.loads((run_root / "CAL_THRESHOLDS_LOCKED.json").read_text(encoding="utf-8"))
    eval_scheduler = json.loads((ANALYSIS_ROOT / "EVALUATION_SCHEDULER.json").read_text(encoding="utf-8"))
    if eval_scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("statistics require all 35 locked evaluation jobs")
    predictions_by_cohort: dict[str, dict[tuple[str, int], dict[str, np.ndarray]]] = {"DEV": {}, "Retino": {}, "SetB": {}}
    threshold_by_model: dict[tuple[str, int], float | None] = {}
    for item in thresholds["models"]:
        threshold_by_model[(item["arm"], int(item["seed"]))] = item["tau_q2"] if item["status"] == "THRESHOLD_LOCKED" else None
    run_metrics: list[dict[str, Any]] = []
    point_values: dict[str, dict[str, dict[tuple[str, int], float | None]]] = {cohort: {metric: {} for metric in PRIMARY_METRICS} for cohort in predictions_by_cohort}
    for arm in ARMS:
        for seed in SEEDS:
            receipt_path = _stage_receipt_path("evaluation", arm, seed)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "LOCKED_EVALUATION_COMPLETE":
                raise RuntimeError(f"locked evaluation receipt invalid for {arm}/{seed}")
            for cohort in predictions_by_cohort:
                path = ANALYSIS_ROOT / "evaluation" / arm.replace("-", "_") / f"seed_{seed}" / f"{cohort}_predictions.npz"
                if p4.sha256_file(path) != receipt["prediction_files"][cohort]["sha256"]:
                    raise RuntimeError(f"prediction archive hash mismatch for {arm}/{seed}/{cohort}")
                loaded_npz = np.load(path, allow_pickle=False)
                pred = {key: loaded_npz[key] for key in loaded_npz.files}
                predictions_by_cohort[cohort][(arm, seed)] = pred
                metrics = receipt["metrics"][cohort]
                point_values[cohort]["qwk"][(arm, seed)] = metrics["qwk"]
                point_values[cohort]["auroc_q2"][(arm, seed)] = metrics["auroc_q2"]
                run_metrics.append({"cohort": cohort, "arm": arm, "seed": seed, **metrics})

    # Recompute registered point estimands from the locked arrays as an independent arithmetic check.
    for cohort, runs in predictions_by_cohort.items():
        reference = runs[(ARMS[0], SEEDS[0])]
        for key, pred in runs.items():
            if not np.array_equal(pred["grade"], reference["grade"]) or not np.array_equal(pred["component_hash"], reference["component_hash"]):
                raise RuntimeError(f"locked prediction alignment mismatch: {cohort}/{key}")
            y = pred["grade"].astype(np.int64)
            pred_grade = pred["grade_pred"].astype(np.int64)
            q2 = pred["q2"].astype(np.float64)
            qwk_val = cohen_kappa_score(y, pred_grade, labels=[0, 1, 2, 3, 4], weights="quadratic")
            auc_val = roc_auc_score((y >= 2).astype(np.int8), q2) if len(np.unique(y >= 2)) == 2 else np.nan
            if not np.isclose(float(qwk_val), float(point_values[cohort]["qwk"][key]), atol=1e-12, rtol=1e-12, equal_nan=True):
                raise RuntimeError(f"QWK arithmetic check failed: {cohort}/{key}")
            if not np.isclose(float(auc_val), float(point_values[cohort]["auroc_q2"][key]), atol=1e-12, rtol=1e-12, equal_nan=True):
                raise RuntimeError(f"q2 AUROC arithmetic check failed: {cohort}/{key}")

    bootstrap = {cohort: _bootstrap_cohort(cohort, predictions, threshold_by_model) for cohort, predictions in predictions_by_cohort.items() if cohort != "DEV"}
    pair_summaries: list[dict[str, Any]] = []
    primary_seed_rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        for cohort in ("Retino", "SetB"):
            deltas = np.asarray([
                (point_values[cohort][metric][("M3", seed)] - point_values[cohort][metric][("C2", seed)])
                if point_values[cohort][metric][("M3", seed)] is not None and point_values[cohort][metric][("C2", seed)] is not None else np.nan
                for seed in SEEDS
            ], dtype=np.float64)
            primary_seed_rows.extend({"metric": metric, "cohort": cohort, "seed": seed, "delta_M3_minus_C2": float(delta) if np.isfinite(delta) else None} for seed, delta in zip(SEEDS, deltas))
            pair_summaries.append({
                "metric": metric, "cohort": cohort, "contrast": "M3-C2",
                "mean_over_five_fixed_seeds": float(np.nanmean(deltas)),
                "seed_sd_ddof1": float(np.nanstd(deltas, ddof=1)),
                "seed_min": float(np.nanmin(deltas)), "seed_max": float(np.nanmax(deltas)),
                "positive_seed_count": int(np.sum(deltas > 0)),
                "per_seed_deltas": [float(x) if np.isfinite(x) else None for x in deltas],
            })

    bootstrap_summary: dict[str, Any] = {"primary": {}, "planned_secondary_contrasts": {}, "factorial_mechanisms": {}, "clinical": {}}
    primary_boot_values: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in PRIMARY_METRICS}
    for cohort in ("Retino", "SetB"):
        boot = bootstrap[cohort]
        qwk_draws = boot["qwk_draws"]
        auc_draws = boot["auroc_draws"]
        arm_idx = {arm: ARMS.index(arm) for arm in ARMS}
        c2_i, m3_i = arm_idx["C2"], arm_idx["M3"]
        primary_valid = (
            np.isfinite(qwk_draws[:, arm_idx["M3"], :]).all(axis=1)
            & np.isfinite(qwk_draws[:, arm_idx["C2"], :]).all(axis=1)
            & np.isfinite(auc_draws[:, arm_idx["M3"], :]).all(axis=1)
            & np.isfinite(auc_draws[:, arm_idx["C2"], :]).all(axis=1)
        )
        for metric, draws in (("qwk", qwk_draws), ("auroc_q2", auc_draws)):
            m3 = draws[:, m3_i, :]
            c2 = draws[:, c2_i, :]
            cohort_values = (m3[primary_valid] - c2[primary_valid]).mean(axis=1)
            primary_boot_values[metric][cohort] = cohort_values[:10000]
            bootstrap_summary["primary"][f"{cohort}:{metric}"] = _percentile_summary(cohort_values)

        for left, right in PLANNED_CONTRASTS:
            contrast_name = f"{left}-{right}"
            for metric, draws in (("qwk", qwk_draws), ("auroc_q2", auc_draws)):
                left_values = draws[:, arm_idx[left], :]
                right_values = draws[:, arm_idx[right], :]
                valid = np.isfinite(left_values).all(axis=1) & np.isfinite(right_values).all(axis=1)
                values = (left_values[valid] - right_values[valid]).mean(axis=1)
                bootstrap_summary["planned_secondary_contrasts"][f"{cohort}:{contrast_name}:{metric}"] = _percentile_summary(values)

        for metric, draws in (("qwk", qwk_draws), ("auroc_q2", auc_draws)):
            c2 = draws[:, arm_idx["C2"], :]
            sx = draws[:, arm_idx["S-X"], :]
            fr = draws[:, arm_idx["F-R"], :]
            m3 = draws[:, arm_idx["M3"], :]
            valid = np.isfinite(c2).all(axis=1) & np.isfinite(sx).all(axis=1) & np.isfinite(fr).all(axis=1) & np.isfinite(m3).all(axis=1)
            scope_seed = 0.5 * ((sx[valid] - c2[valid]) + (m3[valid] - fr[valid]))
            contrast_seed = 0.5 * ((fr[valid] - c2[valid]) + (m3[valid] - sx[valid]))
            interaction_seed = (m3[valid] - fr[valid]) - (sx[valid] - c2[valid])
            bootstrap_summary["factorial_mechanisms"][f"{cohort}:{metric}"] = {
                "scope_main_effect": _percentile_summary(scope_seed.mean(axis=1)),
                "contrast_input_main_effect": _percentile_summary(contrast_seed.mean(axis=1)),
                "interaction": _percentile_summary(interaction_seed.mean(axis=1)),
            }

        clinical_draws = boot["clinical_rate_draws"]
        for metric_index, metric_name in enumerate(CLINICAL_METRICS):
            for arm in ARMS:
                arm_rates = clinical_draws[:, arm_idx[arm], :, metric_index]
                valid = np.isfinite(arm_rates).all(axis=1)
                values = arm_rates[valid].mean(axis=1)
                bootstrap_summary["clinical"][f"{cohort}:{arm}:{metric_name}"] = _percentile_summary(values)
            m3_rates = clinical_draws[:, arm_idx["M3"], :, metric_index]
            c2_rates = clinical_draws[:, arm_idx["C2"], :, metric_index]
            valid = np.isfinite(m3_rates).all(axis=1) & np.isfinite(c2_rates).all(axis=1)
            diff_values = (m3_rates[valid] - c2_rates[valid]).mean(axis=1)
            bootstrap_summary["clinical"][f"{cohort}:M3-C2:{metric_name}"] = _percentile_summary(diff_values)

    for metric in PRIMARY_METRICS:
        r = primary_boot_values[metric].get("Retino", np.asarray([]))
        s = primary_boot_values[metric].get("SetB", np.asarray([]))
        if len(r) >= 10000 and len(s) >= 10000:
            combined = 0.5 * (r[:10000] + s[:10000])
            summary = _percentile_summary(combined)
            delta_mean = float(np.mean([
                np.mean([point_values[cohort][metric][("M3", seed)] - point_values[cohort][metric][("C2", seed)] for cohort in ("Retino", "SetB")])
                for seed in SEEDS
            ]))
            summary["point_estimate_equal_cohort_equal_seed"] = delta_mean
            summary["positive_lower_bound"] = bool(summary.get("ci95") and summary["ci95"][0] > 0)
            bootstrap_summary["primary"][f"combined_equal_cohort_equal_seed:{metric}"] = summary
        else:
            bootstrap_summary["primary"][f"combined_equal_cohort_equal_seed:{metric}"] = {"status": "INSUFFICIENT_VALID_DRAWS", "ci95": None}

    joint_pass = all(
        bootstrap_summary["primary"].get(f"combined_equal_cohort_equal_seed:{metric}", {}).get("ci95") is not None
        and bootstrap_summary["primary"][f"combined_equal_cohort_equal_seed:{metric}"]["ci95"][0] > 0
        for metric in PRIMARY_METRICS
    )
    by_run_path = ANALYSIS_ROOT / "METRICS_BY_RUN.json"
    p4.write_json(by_run_path, {"status": "METRICS_BY_RUN_COMPLETE", "contract_sha256": p4._contract_hash_from_run(run_root), "rows": run_metrics})
    summary = {
        "status": "P4_SHARED_STATISTICS_COMPLETE",
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "model_lock_sha256": p4.sha256_file(MODEL_LOCK_PATH),
        "cal_threshold_lock_sha256": p4.sha256_file(CAL_THRESHOLD_LOCK),
        "evaluation_input_receipt_sha256": p4.sha256_file(run_root / "evaluation_inputs/EVALUATION_INPUT_RECEIPT.json"),
        "metrics_by_run_sha256": p4.sha256_file(by_run_path),
        "primary_estimand": "mean over the five fixed seeds of the equal-weight mean of Retino and SetB M3-C2 deltas",
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_joint_rule": "both two-sided 95% paired cluster-bootstrap lower bounds > 0",
        "primary_joint_result_on_five_fixed_pairs_and_two_open_cohorts": "PASS_STATISTICAL_JOINT_IMPROVEMENT" if joint_pass else "DOES_NOT_MEET_JOINT_RULE",
        "interpretation_limit": "conditional on the five fixed model pairs and already-open Retino/SetB evaluation cohorts; not seed-population stability, per-cohort significance, clinical importance, or new external confirmation",
        "seed_uncertainty_summary": pair_summaries,
        "per_seed_primary_deltas": primary_seed_rows,
        "bootstrap": {
            "repetitions_required": 10000, "attempt_cap": 20000, "seed_resampling": False,
            "point_estimator_not_bootstrap_mean": True, "method": "paired cluster percentile intervals; linear quantile interpolation",
            "independent_streams": {cohort: bootstrap[cohort]["rng_stream"] for cohort in ("Retino", "SetB")},
            "cluster_units": {cohort: bootstrap[cohort]["cluster_unit"] for cohort in ("Retino", "SetB")},
            "cluster_counts": {cohort: bootstrap[cohort]["n_clusters"] for cohort in ("Retino", "SetB")},
            "attempts": {cohort: bootstrap[cohort]["attempts"] for cohort in ("Retino", "SetB")},
        },
        "bootstrap_summary": bootstrap_summary,
        "planned_secondary_contrasts_unadjusted": True,
        "factorial_estimands": {
            "scope_effect": "0.5*((S-X-C2)+(M3-F-R))",
            "contrast_input_effect": "0.5*((F-R-C2)+(M3-S-X))",
            "interaction": "(M3-F-R)-(S-X-C2)",
            "interpretation": "algorithm-factor contrasts; not disease/source causal decomposition",
        },
        "clinical_intervals_are_estimation_only": True,
        "clinical_noninferiority_or_safety_pass": False,
        "created_utc": p4.utc_now(),
    }
    summary_path = ANALYSIS_ROOT / "P4_STATISTICAL_SUMMARY.json"
    p4.write_json(summary_path, summary)
    return summary


def execute_analysis_sequence(gpu_indices: Sequence[int], run_root: Path = RUN_ROOT) -> dict[str, Any]:
    model_lock = lock_final_models(run_root)
    def stage_or_reuse(stage: str) -> dict[str, Any]:
        path = {
            "calibration": ANALYSIS_ROOT / "CALIBRATION_SCHEDULER.json",
            "reload-check": ANALYSIS_ROOT / "RELOAD_CHECK_SCHEDULER.json",
            "evaluation": ANALYSIS_ROOT / "EVALUATION_SCHEDULER.json",
        }[stage]
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("jobs_completed") != len(ARMS) * len(SEEDS):
                raise RuntimeError(f"existing {stage} scheduler receipt is incomplete")
            return result
        return schedule_analysis_stage(stage, gpu_indices, run_root, resume=True)

    calibration_scheduler = stage_or_reuse("calibration")
    if calibration_scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("the complete CAL prediction stage did not finish")
    thresholds = lock_calibration_thresholds(run_root)
    eval_input = freeze_evaluation_inputs(run_root)
    reload_scheduler = stage_or_reuse("reload-check")
    if reload_scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("not all terminal models passed CAL reload verification")
    evaluation_scheduler = stage_or_reuse("evaluation")
    if evaluation_scheduler.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("the complete locked evaluation stage did not finish")
    statistics_path = ANALYSIS_ROOT / "P4_STATISTICAL_SUMMARY.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.is_file() else run_statistics(run_root)
    final = {
        "status": "P4_TRAINING_CAL_DEV_EXTERNAL_AND_STATISTICS_COMPLETE",
        "contract_sha256": p4._contract_hash_from_run(run_root),
        "model_lock_sha256": p4.sha256_file(run_root / "FINAL_MODELS_LOCKED.json"),
        "cal_threshold_lock_sha256": p4.sha256_file(run_root / "CAL_THRESHOLDS_LOCKED.json"),
        "evaluation_input_receipt_sha256": p4.sha256_file(run_root / "evaluation_inputs/EVALUATION_INPUT_RECEIPT.json"),
        "calibration_scheduler_sha256": p4.sha256_file(ANALYSIS_ROOT / "CALIBRATION_SCHEDULER.json"),
        "reload_check_scheduler_sha256": p4.sha256_file(ANALYSIS_ROOT / "RELOAD_CHECK_SCHEDULER.json"),
        "evaluation_scheduler_sha256": p4.sha256_file(ANALYSIS_ROOT / "EVALUATION_SCHEDULER.json"),
        "statistics_sha256": p4.sha256_file(statistics_path),
        "formal_model_count": len(model_lock["models"]),
        "cal_threshold_statuses": {status: sum(row["status"] == status for row in thresholds["models"]) for status in sorted({row["status"] for row in thresholds["models"]})},
        "evaluation_cohorts": list(eval_input["manifests"]),
        "primary_joint_result": statistics["primary_joint_result_on_five_fixed_pairs_and_two_open_cohorts"],
        "completed_utc": p4.utc_now(),
    }
    p4.write_json(ANALYSIS_ROOT / "P4_EXECUTION_COMPLETE.json", final)
    return final


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--stage", choices=("calibration", "reload-check", "evaluation"), required=True)
    worker.add_argument("--arm", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--run-root", type=Path, default=RUN_ROOT)
    lock = sub.add_parser("lock-models")
    lock.add_argument("--run-root", type=Path, default=RUN_ROOT)
    freeze = sub.add_parser("freeze-evaluation-inputs")
    freeze.add_argument("--run-root", type=Path, default=RUN_ROOT)
    cal_lock = sub.add_parser("lock-cal-thresholds")
    cal_lock.add_argument("--run-root", type=Path, default=RUN_ROOT)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--stage", choices=("calibration", "reload-check", "evaluation"), required=True)
    schedule.add_argument("--gpus", default="0,1,2,3")
    schedule.add_argument("--resume", action="store_true")
    schedule.add_argument("--run-root", type=Path, default=RUN_ROOT)
    stats = sub.add_parser("statistics")
    stats.add_argument("--run-root", type=Path, default=RUN_ROOT)
    all_steps = sub.add_parser("execute-all")
    all_steps.add_argument("--gpus", default="0,1,2,3")
    all_steps.add_argument("--run-root", type=Path, default=RUN_ROOT)
    correction = sub.add_parser("correct-threshold-clinical")
    correction.add_argument("--run-root", type=Path, default=RUN_ROOT)
    correction.add_argument("--max-cpu-seconds", type=float, default=3600.0)
    correction.add_argument("--max-wall-seconds", type=float, default=7200.0)
    correction.add_argument("--prior-attempt-cpu-seconds", type=float, default=0.0)
    correction.add_argument("--prior-attempt-wall-seconds", type=float, default=0.0)
    correction.add_argument("--prior-other-cpu-seconds", type=float, default=0.0)
    correction.add_argument("--prior-other-wall-seconds", type=float, default=0.0)
    args = parser.parse_args()
    configure_run_root(args.run_root)
    if args.command == "worker":
        result = _worker_for_eval(args.arm, args.seed, args.stage, args.run_root)
    elif args.command == "lock-models":
        result = lock_final_models(args.run_root)
    elif args.command == "freeze-evaluation-inputs":
        result = freeze_evaluation_inputs(args.run_root)
    elif args.command == "lock-cal-thresholds":
        result = lock_calibration_thresholds(args.run_root)
    elif args.command == "schedule":
        result = schedule_analysis_stage(args.stage, [int(x) for x in args.gpus.split(",") if x.strip()], args.run_root, resume=args.resume)
    elif args.command == "statistics":
        result = run_statistics(args.run_root)
    elif args.command == "execute-all":
        result = execute_analysis_sequence([int(x) for x in args.gpus.split(",") if x.strip()], args.run_root)
    elif args.command == "correct-threshold-clinical":
        result = run_calibration_rule_correction(
            args.run_root,
            max_cpu_seconds=args.max_cpu_seconds,
            max_wall_seconds=args.max_wall_seconds,
            prior_attempt_cpu_seconds=args.prior_attempt_cpu_seconds,
            prior_attempt_wall_seconds=args.prior_attempt_wall_seconds,
            prior_other_cpu_seconds=args.prior_other_cpu_seconds,
            prior_other_wall_seconds=args.prior_other_wall_seconds,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
