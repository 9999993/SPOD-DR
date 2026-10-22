#!/usr/bin/env python3
"""G1 SPARROW V1 P4 execution: fixed inputs, seven arms, five paired seeds.

Large inputs and run products live in the registered SPARROW_ASSETS namespace.
This source file is the runner of record; it never reads evaluation partitions
during training or preflight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
ASSET_ROOT = Path(__file__).resolve().parents[7]
ARCHIVE = ASSET_ROOT / "archive"
DATA_ROOT = ARCHIVE / "reference_assets/datasets"
RUN_ROOT = ARCHIVE / "reference_assets/GROUP_1/training_runs"
HISTORICAL_G1 = ARCHIVE / "model_sources/model_core"
HISTORICAL_SRC = HISTORICAL_G1 / "src"
CRRL_SOURCE = ARCHIVE / "model_sources/clinical_sources"
MANIFEST_DIR = CRRL_SOURCE / "analysis/public_medcv_3direction_program_v1/preprocessing/diabetic_retinopathy_backbone_compatibility/res_512/manifests"
SOURCE_MANIFESTS = {
    "EyePACS": MANIFEST_DIR / "eyepacs_fold0_train.csv",
    "APTOS": MANIFEST_DIR / "aptos_external.csv",
    "DeepDRiD_Set_A": ARCHIVE / "model_sources/data_preparation/analysis/public_medcv_3direction_program_v1/reports/diabetic_retinopathy/dr_boma_dr_v1/DR_BOMA_DEEPDRID_SET_A_TRAIN_INDEX.csv",
}
EXTERNAL_ROOT = HISTORICAL_G1 / "analysis/public_medcv_3direction_program_v1/reports/diabetic_retinopathy/sparrow_dr_v1/external"
EXTERNAL_MANIFESTS = {
    "Retino": EXTERNAL_ROOT / "retino_manifest.csv",
    "SetB": EXTERNAL_ROOT / "deepdrid_set_b_manifest.csv",
}
PUBLIC_INIT_DIR = ASSET_ROOT.parents[4] / "checkpoints/initialization/B3"
PUBLIC_WEIGHT = PUBLIC_INIT_DIR / "model.safetensors"
PUBLIC_INIT_RECEIPT = ARCHIVE / "reference_assets/initialization_sources/reports/group2/update-projection/public-init-screen/public_initialization/INITIALIZATION_RECEIPT.json"
G2_INIT_COMMIT = "49f9eeba"
PUBLIC_WEIGHT_SHA256 = "279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469"
PUBLIC_MODEL_ID = "timm/efficientnet_b3.ra2_in1k"
PUBLIC_MODEL_REVISION = "0366a75518620e0f2077789202073759f2d61393"
SNR_SOURCE_COMMIT = "f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b"
SNR_SOURCE_ROOT = Path("/tmp/g1-p4-snr-source")
SNR_SOURCE_FILES = {
    "classification/PACS/resnet_SNR.py": "1c4a545f007ad51b36a28f3989d56bb243276bae3cd1e8f6fa79ecfada132f25",
    "classification/PACS/model_resnet.py": "c36d29264f736d41aa49537bc28266f2d126a2c14eb21c6b7792b0ecbf93934d",
    "classification/PACS/run_main_agg_snr_causality.sh": "a3ad3442a81da5ad7d0fe42bef4fdca91e199d5d39210c3fa33ad0fde7e542d3",
    "LICENSE": "dea9265341829002e2c23a7372393eb2ed6e26085fb623f38a4ba0af833f30a6",
}
PLAN_COMMIT = "c8e95da02b0932ffdf5d59c944e82ffe82575a68"
PLAN_BLOB_OID = "2e7c3432702cac395a4263e713be3876800f465b"
PLAN_SHA256 = "a1b168970dfcc0e7e22377300ed9a1383d1414660c94bc217617b9b52ddec9f8"
EVIDENCE_BASE = "d9e1f358e848f135428a39789bdc4747c36145e0"
EVIDENCE_BRANCH = "codex/reference_data"
SEEDS = (2101, 2102, 2103, 2104, 2105)
ARMS = ("C0", "C1", "C2", "M3", "S-X", "F-R", "SNR-B3-adapted")
C2_LOO_ARMS = ("C2-no-ordinal", "C2-no-style", "C2-no-orbit", "C2-no-severity")
P4_LOSS_WEIGHTS = {"ord": 0.20, "style": 0.10, "orbit": 0.20, "severity": 0.10}
C2_LOO_LOSS_WEIGHTS = {
    "C2-no-ordinal": {**P4_LOSS_WEIGHTS, "ord": 0.0},
    "C2-no-style": {**P4_LOSS_WEIGHTS, "style": 0.0},
    "C2-no-orbit": {**P4_LOSS_WEIGHTS, "orbit": 0.0},
    "C2-no-severity": {**P4_LOSS_WEIGHTS, "severity": 0.0},
}
TWO_VIEW_ARMS = frozenset(("C1", "C2", "M3", "S-X", "F-R", *C2_LOO_ARMS))
FORMAL_UPDATES = 15001
PREFLIGHT_UPDATES_PER_RUN = 100
PREFLIGHT_MAX_PER_RUN = 200
PREFLIGHT_MAX_TOTAL = 7000
PHYSICAL_UPDATE_CAP = 532035
EXTRA_REPLAY_CAP = 7000
BATCH_SIZE = 16
INPUT_SIZE = 512
SPLIT_SALT = "training_reference"
SPLIT_SEED = "PCG64-SeedSequence(seed,epoch,0x5034)"
CHECKPOINT_EVERY = 500
WORKER_COUNT = 2
MAX_GPUS = 4
GPU_HOUR_CAP = 100.0
CPU_CORE_HOUR_CAP = 300.0
WALL_HOUR_CAP = 14.0 * 24.0
DISK_GIB_CAP = 200.0
RESOURCE_MARGIN = 1.20
RESOURCE_RESERVE_FRACTION = 0.10

OLD_PROJECT_PREFIX = "/PROJECT_INPUTS/clinical_sources"
NEW_PROJECT_PREFIX = str(CRRL_SOURCE)
OLD_DATA_PREFIX = "/PROJECT_INPUTS/datasets"
NEW_DATA_PREFIX = str(DATA_ROOT)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(canonical_json_bytes(value))


def _mapped_path(raw: str) -> Path:
    value = str(raw).strip()
    if value.startswith(OLD_PROJECT_PREFIX + "/"):
        value = NEW_PROJECT_PREFIX + value[len(OLD_PROJECT_PREFIX):]
    elif value == OLD_PROJECT_PREFIX:
        value = NEW_PROJECT_PREFIX
    elif value.startswith(OLD_DATA_PREFIX + "/"):
        value = NEW_DATA_PREFIX + value[len(OLD_DATA_PREFIX):]
    elif value == OLD_DATA_PREFIX:
        value = NEW_DATA_PREFIX
    path = Path(value)
    if OLD_PROJECT_PREFIX in str(path) or str(path).startswith(OLD_DATA_PREFIX):
        raise ValueError(f"unmapped pre-migration input path: {raw}")
    return path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV header missing: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _source_identity_hash(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        # Raw identifiers stay in memory and are never emitted in the package.
        digest.update(str(row["source"]).encode("utf-8") + b"\0")
        digest.update(str(row["image_id"]).encode("utf-8") + b"\0")
        digest.update(str(row["component_raw"]).encode("utf-8") + b"\0")
        digest.update(str(row["grade"]).encode("ascii") + b"\0")
        digest.update(str(row["label"]).encode("ascii") + b"\n")
    return digest.hexdigest()


def _npy_metadata(path: Path) -> tuple[tuple[int, ...], str]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return tuple(int(v) for v in array.shape), str(array.dtype)
    finally:
        del array


def prepare_inputs(*, run_root: Path = RUN_ROOT, hash_cache_files: bool = True) -> dict[str, Any]:
    """Freeze registered source rows, relocated input-cache identities and roles.

    This reads only the three permitted training-source manifests and NPY
    headers/cache bytes. It does not open CAL/DEV/external manifests or images.
    """
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"P4 artifact root is not empty; refusing overwrite: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    session_path = run_root / "SESSION.json"
    session_started = utc_now()
    write_json(session_path, {"session_started_utc": session_started, "run_root": str(run_root), "evidence_branch": EVIDENCE_BRANCH})
    input_dir = run_root / "inputs"
    input_dir.mkdir()
    started = time.perf_counter()
    cpu_started = _process_cpu_seconds("self")
    rows: list[dict[str, Any]] = []
    row_index_by_source = {source: 0 for source in SOURCE_MANIFESTS}
    source_receipts: dict[str, Any] = {}
    cache_paths: dict[str, Path] = {}
    cache_metadata: dict[str, dict[str, Any]] = {}

    for source, manifest_path in SOURCE_MANIFESTS.items():
        fields, source_rows = _read_csv(manifest_path)
        required = {"patient_id", "image_id", "grade", "label", "cache_path"}
        if not required.issubset(fields):
            raise ValueError(f"missing required columns for {source}: {sorted(required - set(fields))}")
        row_identity_seen: set[str] = set()
        for local_index, raw in enumerate(source_rows):
            image_id = str(raw.get("image_id", "")).strip()
            component_raw = str(raw.get("patient_id", "")).strip()
            if not image_id or not component_raw:
                raise ValueError(f"missing registered image/component key in {source}, row={local_index}")
            if image_id in row_identity_seen:
                raise ValueError(f"duplicate image_id within {source}; refusing silent deduplication")
            row_identity_seen.add(image_id)
            grade = int(raw.get("grade", raw.get("label", -1)))
            label = int(raw.get("label", int(grade >= 2)))
            if grade not in range(5) or label != int(grade >= 2):
                raise ValueError(f"grade/binary mapping invalid in {source}; refusing training")
            cache_path = _mapped_path(raw["cache_path"])
            if not cache_path.is_file():
                raise FileNotFoundError(f"registered cache missing after explicit migration mapping: {cache_path}")
            cache_key = str(cache_path)
            if cache_key not in cache_metadata:
                shape, dtype = _npy_metadata(cache_path)
                if shape != (3, 512, 512) or dtype != "float32":
                    raise ValueError(f"cache metadata mismatch: {cache_path} shape={shape} dtype={dtype}")
                cache_metadata[cache_key] = {"shape": list(shape), "dtype": dtype, "bytes": cache_path.stat().st_size}
                cache_paths[cache_key] = cache_path
            row_index = row_index_by_source[source]
            row_index_by_source[source] += 1
            component_hash = sha256_bytes(("sparrow-v1-p4-component|" + source + "|" + component_raw).encode("utf-8"))
            rows.append({
                "source": source,
                "source_row": local_index,
                "row_index": row_index,
                "image_id": image_id,
                "component_raw": component_raw,
                "component_hash": component_hash,
                "grade": grade,
                "label": label,
                "cache_path": cache_path,
            })
        source_receipts[source] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_bytes": manifest_path.stat().st_size,
            "row_count": len(source_rows),
            "source_identity_sha256": _source_identity_hash([r for r in rows if r["source"] == source]),
            "columns_used": ["patient_id", "image_id", "grade", "label", "cache_path"],
            "columns_read_for_identity_only": ["patient_id", "image_id"],
        }

    def hash_one(item: tuple[str, Path]) -> tuple[str, str]:
        key, path = item
        return key, sha256_file(path) if hash_cache_files else "NOT_HASHED"

    hash_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        cache_hashes = dict(pool.map(hash_one, cache_paths.items()))
    cache_hash_seconds = time.perf_counter() - hash_started

    # Known cross-source identity links are absent in the adopted P2 input.
    # Source-specific keys therefore remain distinct and cross-source identity
    # remains explicitly unknown; matching literal IDs are not treated as links.
    component_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    literal_id_sources: dict[str, set[str]] = {}
    for row in rows:
        component_rows.setdefault((row["source"], row["component_raw"]), []).append(row)
        literal_id_sources.setdefault(row["component_raw"], set()).add(row["source"])
    cross_source_literal_ids = sum(len(sources) > 1 for sources in literal_id_sources.values())
    strata: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for key, member_rows in component_rows.items():
        max_grade = max(int(r["grade"]) for r in member_rows)
        strata.setdefault((key[0], max_grade), []).append(key)
    role_by_component: dict[tuple[str, str], str] = {}
    stratum_counts: dict[str, Any] = {}
    for (source, max_grade), components in sorted(strata.items()):
        ordered = sorted(
            components,
            key=lambda key: hashlib.sha256(
                f"{SPLIT_SALT}|{source}|{key[1]}".encode("utf-8")
            ).hexdigest(),
        )
        n_cal = math.floor(0.1 * len(ordered))
        n_dev = math.floor(0.1 * len(ordered))
        for i, key in enumerate(ordered):
            role_by_component[key] = "CAL" if i < n_cal else "DEV" if i < n_cal + n_dev else "TRAIN"
        stratum_counts[f"{source}|max_grade={max_grade}"] = {
            "components": len(ordered), "CAL": n_cal, "DEV": n_dev,
            "TRAIN": len(ordered) - n_cal - n_dev,
        }
    if len(role_by_component) != len(component_rows):
        raise AssertionError("component assignment incomplete")

    assignment_digest = hashlib.sha256()
    for row in rows:
        row["role"] = role_by_component[(row["source"], row["component_raw"])]
        assignment_digest.update(
            f"{row['source']}\0{row['source_row']}\0{row['component_hash']}\0{row['grade']}\0{row['role']}\n".encode("utf-8")
        )

    # A single row-ordered file is private to the registered artifact tree;
    # per-role derivatives ensure preflight workers read TRAIN only.
    columns = ["source", "row_index", "source_row", "component_hash", "grade", "label", "role", "cache_path"]
    by_role: dict[str, list[dict[str, Any]]] = {"TRAIN": [], "CAL": [], "DEV": []}
    for row in rows:
        by_role[row["role"]].append({k: row[k] for k in columns})
    for role, role_rows in by_role.items():
        path = input_dir / f"{role.lower()}_manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(role_rows)

    # Per-cache hashes are aggregate input provenance, not data rows.
    cache_index_path = input_dir / "cache_hash_index.tsv"
    with cache_index_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source\trow_index\tcache_sha256\tbytes\tshape\tdtype\n")
        for row in rows:
            key = str(row["cache_path"])
            meta = cache_metadata[key]
            handle.write(f"{row['source']}\t{row['row_index']}\t{cache_hashes[key]}\t{meta['bytes']}\t3,512,512\tfloat32\n")

    source_summary: dict[str, Any] = {}
    for source in SOURCE_MANIFESTS:
        source_rows2 = [r for r in rows if r["source"] == source]
        source_summary[source] = {
            "images_by_role": {role: sum(r["role"] == role for r in source_rows2) for role in ("TRAIN", "CAL", "DEV")},
            "registered_components_by_role": {
                role: len({r["component_hash"] for r in source_rows2 if r["role"] == role})
                for role in ("TRAIN", "CAL", "DEV")
            },
            "grades_by_role": {
                role: {str(grade): sum(r["role"] == role and r["grade"] == grade for r in source_rows2) for grade in range(5)}
                for role in ("TRAIN", "CAL", "DEV")
            },
        }
    role_component_sets = {
        role: {r["component_hash"] for r in rows if r["role"] == role}
        for role in ("TRAIN", "CAL", "DEV")
    }
    if role_component_sets["TRAIN"] & role_component_sets["CAL"] or role_component_sets["TRAIN"] & role_component_sets["DEV"] or role_component_sets["CAL"] & role_component_sets["DEV"]:
        raise AssertionError("registered components cross TRAIN/CAL/DEV")
    cal_eye = [r for r in rows if r["source"] == "EyePACS" and r["role"] == "CAL"]
    cal_eye_positive = sum(int(r["label"]) == 1 for r in cal_eye)
    cal_eye_negative = len(cal_eye) - cal_eye_positive

    cache_digest = hashlib.sha256()
    for row in rows:
        key = str(row["cache_path"])
        cache_digest.update(f"{row['source']}\0{row['row_index']}\0{cache_hashes[key]}\0{cache_metadata[key]['bytes']}\n".encode("utf-8"))
    receipt = {
        "status": "P4_INPUTS_FROZEN",
        "created_utc": utc_now(),
        "plan_commit": PLAN_COMMIT,
        "plan_blob_oid": PLAN_BLOB_OID,
        "plan_sha256": PLAN_SHA256,
        "evidence_base_commit": EVIDENCE_BASE,
        "source_manifest_receipts": source_receipts,
        "source_rows_total": len(rows),
        "source_summary": source_summary,
        "split_salt": SPLIT_SALT,
        "split_rule": "source-local registered component; maximum member grade; per source/max-grade stratum sorted by SHA256(salt|source|component); first floor(0.1K) CAL, next floor(0.1K) DEV, remainder TRAIN",
        "component_field": "registered patient_id value; source-prefixed for within-source grouping, not asserted to be a verified biological patient identifier",
        "known_cross_source_links": "NONE_FOUND_IN_ADOPTED_P2; unknown relations remain unknown and are not merged by matching literal IDs",
        "literal_component_key_collisions_across_sources": cross_source_literal_ids,
        "split_strata": stratum_counts,
        "component_assignment_sha256": assignment_digest.hexdigest(),
        "cache_hash_index_sha256": sha256_file(cache_index_path),
        "cache_content_merkle_style_sha256": cache_digest.hexdigest(),
        "cache_files_unique": len(cache_paths),
        "cache_files_sha256_hashed": len(cache_paths) if hash_cache_files else 0,
        "cache_hash_wall_seconds": cache_hash_seconds,
        "cal_eye_image_count": len(cal_eye),
        "cal_eye_grade_ge2_count": cal_eye_positive,
        "cal_eye_grade0_1_count": cal_eye_negative,
        "cal_eye_threshold_necessary_condition_possible": cal_eye_positive >= 15 and cal_eye_negative >= 1,
        "external_manifests_opened": False,
        "cal_dev_label_metrics_computed": False,
        "arrays_decoded_during_prepare": 0,
        "p4_prepare_elapsed_seconds": time.perf_counter() - started,
        "p4_prepare_cpu_core_seconds": max(0.0, _process_cpu_seconds("self") - cpu_started),
        "session_started_utc": session_started,
        "cache_metadata": {
            "shape": [3, 512, 512], "dtype": "float32", "cache_hashes_complete": bool(hash_cache_files),
        },
    }
    for role in by_role:
        receipt[f"{role.lower()}_manifest_sha256"] = sha256_file(input_dir / f"{role.lower()}_manifest.csv")
    write_json(input_dir / "INPUT_SPLIT_RECEIPT.json", receipt)
    return receipt


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bootstrap_old_sources() -> tuple[Any, Any, Any, Any]:
    if str(HISTORICAL_SRC) not in sys.path:
        sys.path.insert(0, str(HISTORICAL_SRC))
    from tasks.diabetic_retinopathy.models.clinical_risk.crrl_model import ClinicalRiskRepresentationLearningModel
    from tasks.diabetic_retinopathy.sparrow_dr.model import SparrowDRModel
    from tasks.diabetic_retinopathy.sparrow_dr.orbits import apply_orbit, calibrate_source_quantiles
    from tasks.diabetic_retinopathy.sparrow_dr.thresholds import select_lower_wilson_sensitivity_threshold
    return ClinicalRiskRepresentationLearningModel, SparrowDRModel, apply_orbit, calibrate_source_quantiles


def _tensor_sha256(value: Any) -> str:
    import torch
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_digest(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if hasattr(value, "detach"):
            h.update(key.encode("utf-8") + b"\0" + _tensor_sha256(value).encode("ascii") + b"\n")
    return h.hexdigest()


def load_pinned_backbone(target_backbone: Any, weight_path: Path = PUBLIC_WEIGHT) -> dict[str, Any]:
    """Strictly map the already verified G2 public file; never download/fallback."""
    import timm
    from safetensors.torch import load_file
    if not weight_path.is_file() or sha256_file(weight_path) != PUBLIC_WEIGHT_SHA256:
        raise RuntimeError("pinned public initialization file is missing or has a different SHA256")
    if not PUBLIC_INIT_RECEIPT.is_file():
        raise FileNotFoundError(PUBLIC_INIT_RECEIPT)
    g2_receipt = json.loads(PUBLIC_INIT_RECEIPT.read_text(encoding="utf-8"))
    if g2_receipt.get("consumed_backbone_tensor_count") != 566:
        raise RuntimeError("G2 initialization receipt tensor count does not match the adopted identity")
    source_model = timm.create_model(PUBLIC_MODEL_ID.split("/", 1)[1], pretrained=False, num_classes=1000, global_pool="")
    source_state = load_file(str(weight_path), device="cpu")
    source_model.load_state_dict(source_state, strict=True)
    source_model.reset_classifier(0, global_pool="")
    source_state = source_model.state_dict()
    target_state = target_backbone.state_dict()
    target_keys = set(target_state)
    source_keys = set(source_state)
    excluded = source_keys - target_keys
    allowed = {"conv_head.weight", "bn2.bias", "bn2.num_batches_tracked", "bn2.running_mean", "bn2.running_var", "bn2.weight"}
    if target_keys - source_keys or excluded != allowed:
        raise RuntimeError(f"public feature mapping mismatch: missing={sorted(target_keys-source_keys)[:5]} excluded={sorted(excluded)}")
    mapped: dict[str, Any] = {}
    for key, expected in target_state.items():
        value = source_state[key]
        if tuple(value.shape) != tuple(expected.shape) or value.dtype != expected.dtype:
            raise RuntimeError(f"public backbone tensor metadata mismatch: {key}")
        mapped[key] = value
    target_backbone.load_state_dict(mapped, strict=True)
    loaded = target_backbone.state_dict()
    per_key = {key: _tensor_sha256(loaded[key]) for key in sorted(target_keys)}
    if per_key != g2_receipt.get("per_key_consumed_value_sha256"):
        raise RuntimeError("per-key consumed backbone tensor hashes differ from the G2 receipt")
    digest = hashlib.sha256()
    for key, value_hash in per_key.items():
        digest.update(key.encode("utf-8") + b"\0" + value_hash.encode("ascii") + b"\n")
    if digest.hexdigest() != g2_receipt.get("consumed_state_values_sha256"):
        raise RuntimeError("aggregate consumed backbone tensor hash differs from the G2 receipt")
    return {
        "model_id": PUBLIC_MODEL_ID,
        "revision": PUBLIC_MODEL_REVISION,
        "weight_path": str(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": PUBLIC_WEIGHT_SHA256,
        "timm_version": timm.__version__,
        "g2_receipt_timm_version": "1.0.19",
        "g2_receipt_path": str(PUBLIC_INIT_RECEIPT),
        "g2_consumed_state_values_sha256": digest.hexdigest(),
        "consumed_tensor_count": len(per_key),
        "strict_source_load": True,
        "strict_target_load": True,
        "excluded_source_keys": sorted(excluded),
        "source_only_identity_from_g2_receipt": True,
    }


def build_p4_model(arm: str, seed: int, *, weight_path: Path = PUBLIC_WEIGHT) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn
    arm = str(arm)
    if arm not in (*ARMS, *C2_LOO_ARMS):
        raise ValueError(f"unknown P4 arm: {arm}")
    _set_seed(int(seed))
    ClinicalModel, SparrowDRModel, _, _ = _bootstrap_old_sources()
    base = ClinicalModel(
        backbone_name="efficientnet_b3",
        pretrained=False,
        backbone_img_size=INPUT_SIZE,
        pooling="gem",
        local_query_count=6,
        feature_dim=256,
        attention_heads=4,
        mode="local_global",
        beta=1.0,
        crrl_mode="consistency",
        risk_hidden_dim=256,
        risk_embedding_dim=256,
    )
    load_receipt = load_pinned_backbone(base.feature_pyramid.backbone, weight_path)
    common_head_state = {
        key: value for key, value in base.state_dict().items()
        if not key.startswith("feature_pyramid.")
    }
    common_head_sha = _state_digest(common_head_state)
    if arm == "SNR-B3-adapted":
        from torch import nn
        class SNRUnit(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.channels = int(channels)
                self.in_norm = nn.InstanceNorm2d(channels, affine=True, eps=1e-5, track_running_stats=False)
                hidden = int(channels) // 16
                if hidden < 1:
                    raise ValueError("SNR gate hidden width must be positive")
                self.gate1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
                self.gate_relu = nn.ReLU(inplace=True)
                self.gate2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
                self.classifier = nn.Linear(channels, 5)
                for module in (self.gate1, self.gate2):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
                nn.init.constant_(self.classifier.bias, 0.0)

            def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                normalized = self.in_norm(feature)
                residual = feature - normalized
                gate = torch.sigmoid(self.gate2(self.gate_relu(self.gate1(torch.nn.functional.adaptive_avg_pool2d(residual, 1)))))
                plus = normalized + gate * residual
                minus = normalized + (1.0 - gate) * residual
                logits0 = self.classifier(normalized.mean(dim=(-2, -1)))
                logits_plus = self.classifier(plus.mean(dim=(-2, -1)))
                logits_minus = self.classifier(minus.mean(dim=(-2, -1)))
                return plus, {"logits0": logits0, "logits_plus": logits_plus, "logits_minus": logits_minus}

        class SNRFeaturePyramid(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                info = list(backbone.feature_info.info)
                info_by_module = {str(item["module"]): int(item["num_chs"]) for item in info}
                insertion_channels = {name: info_by_module.get(name) for name in ("blocks.2", "blocks.4")}
                if insertion_channels != {"blocks.2": 48, "blocks.4": 136} or backbone.feature_info.channels() != [48, 136]:
                    raise RuntimeError(f"SNR insertion points differ from the frozen B3 contract: {insertion_channels}; outputs={backbone.feature_info.channels()}")
                self.snr48 = SNRUnit(48)
                self.snr136 = SNRUnit(136)
                self._last_aux: dict[str, torch.Tensor] | None = None

            def pop_aux(self) -> dict[str, torch.Tensor]:
                if self._last_aux is None:
                    raise RuntimeError("SNR auxiliary output missing for this forward")
                value = self._last_aux
                self._last_aux = None
                return value

            def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                backbone = self.backbone
                value = backbone.conv_stem(image)
                value = backbone.bn1(value)
                stage48: torch.Tensor | None = None
                stage136: torch.Tensor | None = None
                aux48: dict[str, torch.Tensor] | None = None
                aux136: dict[str, torch.Tensor] | None = None
                for block_index, block in enumerate(backbone.blocks):
                    value = block(value)
                    if block_index == 2:
                        if value.shape[1] != 48:
                            raise RuntimeError(f"SNR stage3 expected 48 channels, got {value.shape[1]}")
                        value, aux48 = self.snr48(value)
                        stage48 = value
                    elif block_index == 4:
                        if value.shape[1] != 136:
                            raise RuntimeError(f"SNR stage4 expected 136 channels, got {value.shape[1]}")
                        value, aux136 = self.snr136(value)
                        stage136 = value
                if stage48 is None or stage136 is None or aux48 is None or aux136 is None:
                    raise RuntimeError("SNR B3 forward failed to reach both registered feature boundaries")
                self._last_aux = {
                    **{f"48_{key}": value for key, value in aux48.items()},
                    **{f"136_{key}": value for key, value in aux136.items()},
                }
                return stage48, stage136

        base.feature_pyramid = SNRFeaturePyramid(base.feature_pyramid.backbone)
        model = base
    elif arm == "C0":
        model = base
    else:
        implementation_arm = "C2" if arm in C2_LOO_ARMS else "M3" if arm == "S-X" else "C2" if arm == "F-R" else arm
        model_behavior_arm = "C2" if arm in C2_LOO_ARMS else arm
        model = P4V1Arm(base, SparrowDRModel, model_behavior_arm, implementation_arm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    receipt = {
        "arm": arm,
        "model_behavior_arm": "C2" if arm in C2_LOO_ARMS else arm,
        "seed": int(seed),
        "backbone_load": load_receipt,
        "common_task_head_sha256": common_head_sha,
        "model_parameter_count": sum(int(p.numel()) for p in model.parameters()),
        "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "model_state_sha256": _state_digest(model.state_dict()),
        "feature_channels": [48, 136],
        "feature_blocks": ["blocks.2", "blocks.4"],
        "training_bn_policy": "update each arm's own BN statistics; eval freezes them; no post-training BN calibration",
        "device": str(device),
    }
    return model, receipt


class P4V1Arm:
    """Small arm adapter around the exact frozen V1 feature transforms."""
    def __new__(cls, base: Any, SparrowDRModel: Any, p4_arm: str, implementation_arm: str):
        import torch
        from torch import nn
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p4_arm = p4_arm
                self.inner = SparrowDRModel(base, arm_id=implementation_arm, method_enabled=implementation_arm != "C0")

            def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
                if self.p4_arm == "C0":
                    return self.inner.base(image)
                stage3, stage4 = self.inner.base.feature_pyramid(image)
                output = self.inner._build_from_features(stage3, stage4)
                if self.p4_arm == "S-X":
                    contrast3, contrast4 = stage3, stage4
                elif self.p4_arm == "F-R":
                    _, contrast3 = self.inner.axis_stage3.decompose(stage3)
                    _, contrast4 = self.inner.axis_stage4.decompose(stage4)
                elif self.p4_arm == "C2":
                    contrast3, contrast4 = stage3, stage4
                else:
                    contrast3, contrast4 = output["residual_stage3"], output["residual_stage4"]
                output["contrastive_stage3"] = contrast3
                output["contrastive_stage4"] = contrast4
                return output
        return _Model()


def _entropy_from_prob(prob: Any) -> Any:
    import torch
    mask = prob >= 1e-6
    selected = torch.where(mask, prob * torch.log(prob.clamp_min(1e-6)), torch.zeros_like(prob))
    return -selected.sum() / float(prob.shape[0])


def _gradient_connectivity_probe(model: Any) -> dict[str, Any]:
    """Record trainable parameter gradient presence after the first backward."""
    import torch
    none_names: list[str] = []
    zero_names: list[str] = []
    nonzero_count = 0
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_count += 1
        if parameter.grad is None:
            none_names.append(name)
        elif bool(torch.count_nonzero(parameter.grad).item() == 0):
            zero_names.append(name)
        else:
            nonzero_count += 1
    return {
        "sampled_before_optimizer_step": True,
        "trainable_parameter_tensors": trainable_count,
        "grad_none_parameter_tensors": len(none_names),
        "grad_zero_parameter_tensors": len(zero_names),
        "grad_nonzero_parameter_tensors": nonzero_count,
        "grad_none_parameter_names": none_names,
        "grad_zero_parameter_names": zero_names,
    }


def p4_loss(
    arm: str,
    output: dict[str, Any],
    grade: Any,
    source: Sequence[str],
    view_output: dict[str, Any] | None,
    *,
    loss_weights_override: dict[str, float] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch
    from torch.nn import functional as F
    for field in (
        "grade_probs", "severity_score", "aux_cumulative_stage3", "aux_cumulative_stage4",
    ):
        value = output.get(field)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite P4 output {field} for {arm}")
    if view_output is not None:
        for field in ("grade_probs", "severity_score"):
            value = view_output.get(field)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite P4 second-view output {field} for {arm}")
    prob = output["grade_probs"].float().clamp_min(1e-7)
    target = grade.long()
    main = -prob[torch.arange(prob.shape[0], device=prob.device), target].log().mean()
    components: dict[str, Any] = {"grade": main}
    counts: dict[str, Any] = {"style_anchors": 0, "style_anchor_by_grade": [0] * 5, "anchors_by_grade": [0] * 5}
    if arm == "C0":
        return main, components, counts
    if arm == "SNR-B3-adapted":
        aux = output["snr_aux"]
        dual = main.new_zeros(())
        for position in ("48", "136"):
            p0 = torch.softmax(aux[f"{position}_logits0"].float(), dim=1)
            pplus = torch.softmax(aux[f"{position}_logits_plus"].float(), dim=1)
            pminus = torch.softmax(aux[f"{position}_logits_minus"].float(), dim=1)
            e0, ep, em = _entropy_from_prob(p0), _entropy_from_prob(pplus), _entropy_from_prob(pminus)
            part = F.softplus(ep - e0) + F.softplus(e0 - em)
            components[f"snr_dual_{position}"] = part
            dual = dual + part
        snr_ce = F.cross_entropy(aux["136_logits_plus"].float(), target)
        components["snr_ce_136_plus"] = snr_ce
        total = main + 0.01 * dual + 0.01 * snr_ce
        return total, components, counts

    ordinal_targets = torch.stack([(target >= k).float() for k in range(1, 5)], dim=1)
    ord_loss = 0.5 * (
        F.binary_cross_entropy(output["aux_cumulative_stage3"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
        + F.binary_cross_entropy(output["aux_cumulative_stage4"].float().clamp(1e-6, 1.0 - 1e-6), ordinal_targets)
    )
    parts = []
    for key in ("contrastive_stage3", "contrastive_stage4"):
        pooled = output[key].mean(dim=(-2, -1))
        parts.append(F.normalize(pooled.float(), dim=1))
    embedding = F.normalize(torch.cat(parts, dim=1), dim=1)
    batch = int(grade.shape[0])
    same = target.view(-1, 1).eq(target.view(1, -1))
    other_source = torch.tensor([[source[i] != source[j] for j in range(batch)] for i in range(batch)], device=target.device, dtype=torch.bool)
    diagonal = torch.eye(batch, device=target.device, dtype=torch.bool)
    positive_mask = same & other_source & ~diagonal
    similarity = embedding @ embedding.T
    valid = positive_mask.any(dim=1)
    style = main.new_zeros(())
    if bool(valid.any()):
        positives = similarity[positive_mask]
        negatives = similarity[~diagonal]
        style = (1.0 - positives).mean() + 0.05 * F.relu(negatives.mean() - positives.mean() + 0.1)
    valid_cpu = valid.detach().cpu().numpy()
    grades_cpu = target.detach().cpu().numpy()
    counts["style_anchors"] = int(valid_cpu.sum())
    for k in range(5):
        counts["anchors_by_grade"][k] = int((grades_cpu == k).sum())
        counts["style_anchor_by_grade"][k] = int(((grades_cpu == k) & valid_cpu).sum())
    orbit = main.new_zeros(())
    severity = main.new_zeros(())
    if view_output is not None:
        p = output["grade_probs"].float().clamp_min(1e-7)
        q = view_output["grade_probs"].float().clamp_min(1e-7)
        orbit = 0.5 * (F.kl_div(p.log(), q, reduction="batchmean") + F.kl_div(q.log(), p, reduction="batchmean"))
        severity = (output["severity_score"].float() - view_output["severity_score"].float()).abs().mean()
    components.update({"ord": ord_loss, "style": style, "orbit": orbit, "severity": severity})
    weights = loss_weights_override or C2_LOO_LOSS_WEIGHTS.get(arm, P4_LOSS_WEIGHTS)
    for name, value in components.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite P4 loss component {name} for {arm}")
    total = main + weights["ord"] * ord_loss + weights["style"] * style + weights["orbit"] * orbit + weights["severity"] * severity
    return total, components, counts


class P4ManifestDataset:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        row = self.rows[index]
        array = np.load(row["cache_path"], allow_pickle=False)
        if array.shape != (3, 512, 512) or array.dtype != np.float32:
            raise ValueError(f"runtime cache metadata mismatch at local row={index}")
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        return {
            "image": tensor,
            "grade": int(row["grade"]),
            "label": int(row["label"]),
            "source": row["source"],
            "component_hash": row["component_hash"],
            "row_index": int(row["row_index"]),
        }


class P4BatchSampler:
    def __init__(self, size: int, seed: int, start_update: int, stop_update: int, batch_size: int = BATCH_SIZE) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.start_update = int(start_update)
        self.stop_update = int(stop_update)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = self.size // self.batch_size
        if self.steps_per_epoch <= 0:
            raise ValueError("TRAIN has fewer than one complete batch")

    def __len__(self) -> int:
        return max(0, self.stop_update - self.start_update)

    def __iter__(self) -> Iterator[list[int]]:
        permutations: dict[int, np.ndarray] = {}
        for global_zero_index in range(self.start_update, self.stop_update):
            epoch = global_zero_index // self.steps_per_epoch
            offset = (global_zero_index % self.steps_per_epoch) * self.batch_size
            if epoch not in permutations:
                seed_seq = np.random.SeedSequence([self.seed, epoch, 0x5034])
                permutations[epoch] = np.random.Generator(np.random.PCG64(seed_seq)).permutation(self.size)
            yield permutations[epoch][offset:offset + self.batch_size].astype(np.int64).tolist()


def _normalize_from_01(image: Any) -> Any:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def _denormalize(image: Any) -> Any:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _rng_state() -> dict[str, Any]:
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), 0)


def _atomic_torch_save(value: Any, path: Path) -> int:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    size = temporary.stat().st_size
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return size


def _atomic_csv_row(path: Path, fields: Sequence[str], row: dict[str, Any], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _optimizer_state_bytes(model: Any, optimizer: Any, seed: int, arm: str) -> int:
    tmp = RUN_ROOT / "preflight" / "state-size" / f"{arm.replace('-', '_')}_seed{seed}_optimizer_probe.pt"
    size = _atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(), "next_update": 0}, tmp)
    tmp.unlink()
    return size


def _load_training_rows(run_root: Path) -> list[dict[str, str]]:
    _, rows = _read_csv(run_root / "inputs/train_manifest.csv")
    if not rows:
        raise ValueError("frozen TRAIN manifest is empty")
    if any(r["role"] != "TRAIN" for r in rows):
        raise RuntimeError("preflight/training manifest contains non-TRAIN records")
    return rows


def worker_run(
    arm: str,
    seed: int,
    *,
    mode: str,
    device_index: int,
    gpu_uuid: str,
    updates: int,
    run_root: Path = RUN_ROOT,
    resume: bool = False,
    retry_index: int = 0,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.utils import clip_grad_norm_
    arm_dir = run_root / ("preflight" if mode == "preflight" else "formal") / arm.replace("-", "_") / f"seed_{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / (f"attempt_{retry_index}.progress.jsonl" if retry_index else "progress.jsonl")
    receipt_path = arm_dir / (f"attempt_{retry_index}.receipt.json" if retry_index else "run_receipt.json")
    progress_path = arm_dir / (f"attempt_{retry_index}.progress.json" if retry_index else "progress.json")
    if mode == "preflight" and retry_index == 0 and receipt_path.exists():
        raise FileExistsError(receipt_path)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("P4 worker requires CUDA; CPU training is not permitted")
    torch.cuda.set_device(int(device_index))
    device = torch.device("cuda", int(device_index))

    started = time.perf_counter()
    ru0_self = _process_cpu_seconds("self")
    ru0_children = _process_cpu_seconds("children")
    model, init_receipt = build_p4_model(arm, seed)
    model.to(device)
    rows = _load_training_rows(run_root)
    dataset = P4ManifestDataset(rows)
    from torch.optim import AdamW
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    orbit_fn = orbit_cfg = None
    if arm != "SNR-B3-adapted":
        _, _, orbit_fn, calibrate = _bootstrap_old_sources()
        orbit_cfg = calibrate(seed=seed)
    state_path = arm_dir / "recovery.pt"
    start_update = 0
    physical_updates_prior = 0
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume requested without a verified recovery state: {state_path}")
        checkpoint = torch.load(state_path, map_location=device, weights_only=False)
        if checkpoint.get("arm") != arm or checkpoint.get("seed") != int(seed):
            raise RuntimeError("recovery state identity mismatch")
        if checkpoint.get("contract_sha256") != _contract_hash_from_run(run_root):
            raise RuntimeError("recovery state belongs to a different frozen contract")
        if checkpoint.get("training_manifest_sha256") != sha256_file(run_root / "inputs/train_manifest.csv"):
            raise RuntimeError("recovery state belongs to a different frozen TRAIN partition")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["next_update"])
        physical_updates_prior = int(checkpoint.get("physical_updates", start_update))
        saved_recovery_bytes = int(state_path.stat().st_size)
        _restore_rng_state(checkpoint["rng"])
    else:
        _set_seed(int(seed))
    sampler = P4BatchSampler(len(dataset), seed, start_update, int(updates))
    data_loader_generator = torch.Generator(device="cpu")
    data_loader_generator.manual_seed(int(seed) ^ 0x50443431)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=WORKER_COUNT,
        pin_memory=True,
        persistent_workers=WORKER_COUNT > 0,
        generator=data_loader_generator,
    )
    run_loop_start = time.perf_counter()
    model_initial_digest = init_receipt["model_state_sha256"]
    grad_norm_sum = 0.0
    gradient_probe: dict[str, Any] | None = None
    loss_records: list[dict[str, Any]] = []
    style_totals = {"anchors": 0, "by_grade": [0] * 5, "grade_support": [0] * 5}
    update_times: list[float] = []
    saved_recovery_bytes = 0
    preflight_recovery_checked = False
    physical_updates = physical_updates_prior
    try:
        for local_offset, batch in enumerate(loader):
            zero_based = start_update + local_offset
            update_number = zero_based + 1
            tick = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True)
            grade = batch["grade"].to(device, non_blocking=True)
            source = list(batch["source"])
            if not bool(torch.isfinite(image).all()):
                raise FloatingPointError("non-finite pixels in registered TRAIN cache")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            second_output = None
            if arm in TWO_VIEW_ARMS:
                view_01 = _denormalize(image)
                fov = torch.ones((view_01.shape[0], view_01.shape[-2], view_01.shape[-1]), dtype=torch.bool, device=device)
                view = orbit_fn(view_01, fov, "gamma", int(seed) * 100000 + update_number, orbit_cfg).image
                second_output = model(_normalize_from_01(view))
            elif arm == "SNR-B3-adapted":
                feature_pyramid = model.feature_pyramid
                output["snr_aux"] = feature_pyramid.pop_aux()
            loss, components, support = p4_loss(arm, output, grade, source, second_output)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at logical update {update_number}")
            loss.backward()
            if update_number == 1 and arm in C2_LOO_ARMS:
                gradient_probe = _gradient_connectivity_probe(model)
            grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu())
            optimizer.step()
            physical_updates += 1
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - tick
            update_times.append(update_seconds)
            grad_norm_sum += grad_norm
            style_totals["anchors"] += int(support["style_anchors"])
            for grade_i in range(5):
                style_totals["by_grade"][grade_i] += int(support["style_anchor_by_grade"][grade_i])
                style_totals["grade_support"][grade_i] += int(support["anchors_by_grade"][grade_i])
            if update_number == start_update + 1 or update_number % 25 == 0 or update_number == int(updates):
                record = {
                    "utc": utc_now(), "logical_update": update_number,
                    "physical_updates": physical_updates,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm_preclip": grad_norm,
                    "loss_components": {k: float(v.detach().cpu()) for k, v in components.items()},
                    "style_anchor_count": int(support["style_anchors"]),
                    "update_wall_seconds": update_seconds,
                }
                if update_number == 1 and gradient_probe is not None:
                    record["gradient_connectivity_probe"] = gradient_probe
                append_jsonl(log_path, record)
            if mode == "formal" and (update_number % CHECKPOINT_EVERY == 0 or update_number == int(updates)):
                checkpoint = {
                    "schema": "sparrow_v1_p4_training_state_v1",
                    "arm": arm, "seed": int(seed), "next_update": update_number,
                    "physical_updates": physical_updates,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": _rng_state(), "contract_sha256": _contract_hash_from_run(run_root),
                    "training_manifest_sha256": sha256_file(run_root / "inputs/train_manifest.csv"),
                }
                saved_recovery_bytes = _atomic_torch_save(checkpoint, state_path)
                write_json(arm_dir / "recovery_meta.json", {
                    "arm": arm, "seed": int(seed), "next_update": update_number,
                    "physical_updates": physical_updates, "checkpoint_sha256": sha256_file(state_path),
                    "checkpoint_bytes": saved_recovery_bytes, "contract_sha256": _contract_hash_from_run(run_root),
                    "training_manifest_sha256": sha256_file(run_root / "inputs/train_manifest.csv"),
                    "updated_utc": utc_now(),
                })
            if mode == "preflight" and update_number in {50} and not preflight_recovery_checked:
                checkpoint = {
                    "schema": "sparrow_v1_p4_training_state_v1",
                    "arm": arm, "seed": int(seed), "next_update": update_number,
                    "physical_updates": physical_updates,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(),
                    "contract_sha256": _contract_hash_from_run(run_root),
                }
                tmp = arm_dir / "preflight_recovery_probe.pt"
                saved_recovery_bytes = _atomic_torch_save(checkpoint, tmp)
                reload_state = torch.load(tmp, map_location=device, weights_only=False)
                model.load_state_dict(reload_state["model"], strict=True)
                optimizer.load_state_dict(reload_state["optimizer"])
                _restore_rng_state(reload_state["rng"])
                if _state_digest(model.state_dict()) != _state_digest(reload_state["model"]):
                    raise RuntimeError("preflight recovery state failed exact model reload")
                preflight_recovery_checked = True
                tmp.unlink()
            write_json(progress_path, {
                "arm": arm, "seed": int(seed), "logical_update": update_number,
                "physical_updates": physical_updates, "mode": mode,
                "gpu_uuid": gpu_uuid, "updated_utc": utc_now(),
            })
    finally:
        if getattr(loader, "_iterator", None) is not None:
            loader._iterator._shutdown_workers()
    run_loop_seconds = time.perf_counter() - run_loop_start
    elapsed_seconds = time.perf_counter() - started
    cpu_seconds = max(0.0, (_process_cpu_seconds("self") - ru0_self) + (_process_cpu_seconds("children") - ru0_children))
    if mode == "formal" and int(updates) == FORMAL_UPDATES:
        final_model = arm_dir / "final_model.pt"
        final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        final_model_bytes = _atomic_torch_save(final_state, final_model)
        final_sha = sha256_file(final_model)
        if not state_path.is_file() or saved_recovery_bytes <= 0:
            raise RuntimeError("formal terminal checkpoint missing its full recovery state")
        receipt_status = "FORMAL_TRAINING_COMPLETE"
    else:
        final_model_bytes = 0
        final_sha = None
        receipt_status = "PREFLIGHT_COMPLETE" if mode == "preflight" else "FORMAL_RESUMABLE_PARTIAL"
    steps_done = int(updates) - start_update
    receipt = {
        "status": receipt_status,
        "contract_sha256": _contract_hash_from_run(run_root),
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "logical_updates_requested": int(updates),
        "logical_updates_done_this_process": steps_done,
        "logical_next_update": int(updates),
        "physical_optimizer_updates_this_process": steps_done,
        "physical_updates_total_for_run": physical_updates,
        "resume": bool(resume),
        "retry_index": int(retry_index),
        "gpu_uuid": gpu_uuid,
        "device_index_within_visible_set": int(device_index),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "timm_version": init_receipt["backbone_load"]["timm_version"] if "backbone_load" in init_receipt else None,
        "initialization": init_receipt,
        "formal_training_config": {
            "input": "pre-normalized float32 NPY [3,512,512], original V1 ImageNet mean/std normalization",
            "optimizer": "AdamW", "learning_rate": 3e-5, "weight_decay": 1e-4,
            "betas": [0.9, 0.999], "eps": 1e-8,
            "batch_size": BATCH_SIZE, "world_size": 1, "gradient_accumulation": 1,
            "grad_clip_norm": 1.0, "precision": "FP32", "tf32": False,
            "scheduler": None, "ema": False, "swa": False,
            "checkpoint_selection": "fixed final step 15001; no early stop or DEV selection",
            "bn_train": "per-arm running-stat updates", "bn_eval": "frozen running stats",
            "preflight_update_cap_per_run": PREFLIGHT_MAX_PER_RUN,
            "two_view_arms": sorted(TWO_VIEW_ARMS),
            "one_view_arms": ["C0", "SNR-B3-adapted"],
        },
        "loss_weights": ({"grade": 1.0, **C2_LOO_LOSS_WEIGHTS.get(arm, P4_LOSS_WEIGHTS)} if arm not in {"C0", "SNR-B3-adapted"} else None),
        "zero_weight_loss_components": [
            {"component": name, "weight": 0.0, "computed_and_finite": True}
            for name, weight in C2_LOO_LOSS_WEIGHTS.get(arm, {}).items()
            if weight == 0.0
        ],
        "gradient_connectivity_probe": gradient_probe,
        "train_rows": len(rows),
        "train_grade_support": {str(k): sum(int(r["grade"]) == k for r in rows) for k in range(5)},
        "training_manifest_sha256": sha256_file(run_root / "inputs/train_manifest.csv"),
        "recovery_checkpoint_bytes": saved_recovery_bytes,
        "recovery_checkpoint_path": str(state_path) if state_path.is_file() else None,
        "final_model_path": str(arm_dir / "final_model.pt") if final_sha else None,
        "final_model_bytes": final_model_bytes,
        "final_model_sha256": final_sha,
        "style_cross_source_anchors": style_totals,
        "mean_grad_norm_preclip": grad_norm_sum / max(steps_done, 1),
        "mean_step_wall_seconds": float(np.mean(update_times)) if update_times else None,
        "p50_step_wall_seconds": float(np.median(update_times)) if update_times else None,
        "p95_step_wall_seconds": float(np.percentile(update_times, 95)) if update_times else None,
        "training_loop_wall_seconds": run_loop_seconds,
        "allocated_gpu_wall_seconds": elapsed_seconds,
        "cpu_core_seconds": cpu_seconds,
        "preflight_recovery_reload_verified": preflight_recovery_checked,
        "training_image_forward_inputs_this_process": steps_done * BATCH_SIZE * (2 if arm in TWO_VIEW_ARMS else 1),
        "prohibited_reads": {"CAL": 0, "DEV": 0, "Retino": 0, "SetB": 0, "SetC": 0},
        "completed_utc": utc_now(),
    }
    write_json(receipt_path, receipt)
    if retry_index > 0:
        write_json(arm_dir / "run_receipt.json", receipt)
    return receipt


def _process_cpu_seconds(which: str) -> float:
    import resource
    who = resource.RUSAGE_SELF if which == "self" else resource.RUSAGE_CHILDREN
    usage = resource.getrusage(who)
    return float(usage.ru_utime + usage.ru_stime)


def _contract_hash_from_run(run_root: Path) -> str:
    path = run_root / "EXECUTION_CONTRACT.json"
    if not path.is_file():
        return "CONTRACT_NOT_YET_FROZEN"
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(value["contract_sha256"])


def _attempt_progress_updates(arm_dir: Path, attempt: int) -> int:
    stem = "progress" if int(attempt) == 0 else f"attempt_{int(attempt)}.progress"
    log_path = arm_dir / f"{stem}.jsonl"
    values: list[int] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                values.append(int(json.loads(line).get("logical_update", 0)))
    state_path = arm_dir / f"{stem}.json"
    if state_path.is_file():
        values.append(int(json.loads(state_path.read_text(encoding="utf-8")).get("logical_update", 0)))
    return max(values, default=0)


def _attempt_ids_with_progress(arm_dir: Path) -> list[int]:
    attempts: set[int] = set()
    if any((arm_dir / name).is_file() for name in ("progress.json", "progress.jsonl")):
        attempts.add(0)
    for pattern in ("attempt_*.progress.json", "attempt_*.progress.jsonl"):
        for path in arm_dir.glob(pattern):
            attempts.add(int(path.name.split("_")[1].split(".")[0]))
    return sorted(attempts)


def _contract_payload(input_receipt: dict[str, Any], implementation_receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "SPARROW_V1_P4_EXECUTION_CONTRACT_V1",
        "plan": {"commit": PLAN_COMMIT, "blob_oid": PLAN_BLOB_OID, "sha256": PLAN_SHA256},
        "evidence_worktree": {"base_commit": EVIDENCE_BASE, "branch": EVIDENCE_BRANCH},
        "arms": list(ARMS), "seeds": list(SEEDS), "formal_updates_per_arm_seed": FORMAL_UPDATES,
        "split": {
            "salt": SPLIT_SALT,
            "component": "source-local registered patient_id component; maximum component grade",
            "ordering": "SHA256(salt|source|component)",
            "allocation": "per source/max-grade stratum; first floor(0.1K) components CAL; next floor(0.1K) DEV; remainder TRAIN",
            "known_cross_source_links": "none found in adopted P2; cross-source patient identity remains unknown",
            "cal_threshold_source": "only rows with role=CAL and source=EyePACS",
        },
        "initialization": {
            "model_id": PUBLIC_MODEL_ID, "revision": PUBLIC_MODEL_REVISION,
            "sha256": PUBLIC_WEIGHT_SHA256, "bytes": 49335454,
            "consumed_tensor_count": 566,
            "consumed_state_values_sha256": "66575982be4dc1754b703a366c970673ba44817b6c3aab5e0a86e565b1d6e756",
            "file_path": str(PUBLIC_WEIGHT),
            "g2_receipt": str(PUBLIC_INIT_RECEIPT),
            "g2_receipt_source_commit": "0be7227bf9e8006582a256c905c707742eea8897",
            "timm_in_execution_environment": implementation_receipt["timm_version"],
        },
        "common_training": {
            "input_size": 512, "batch_size": 16, "world_size": 1, "accumulation": 1,
            "optimizer": "AdamW", "lr": 3e-5, "weight_decay": 1e-4,
            "betas": [0.9, 0.999], "eps": 1e-8, "clip_norm": 1.0,
            "precision": "FP32", "tf32": False, "scheduler": None, "ema": False, "swa": False,
            "loss_weights": {"grade_nll": 1.0, "ordinal": 0.20, "style": 0.10, "orbit_kl": 0.20, "severity_consistency": 0.10},
            "style_loss": "mean(1-cosine_similarity) over same-grade cross-source positive pairs + 0.05*relu(mean(all_off_diagonal_similarity)-mean(positive_similarity)+0.1)",
            "sampler": "per-epoch PCG64 permutation seeded by SeedSequence([model_seed, epoch, 0x5034]); complete batches of 16; drop epoch remainder",
            "checkpoint": "fixed last step; no early stop or DEV checkpoint selection",
            "recovery": "atomic full-state checkpoint every 500 formal updates and at final update; resume only the same contract and frozen TRAIN manifest; preflight verifies model/optimizer/RNG save-reload at update 50",
            "bn": "update per arm during training; freeze at evaluation; no recalibration",
            "sampler_rng": SPLIT_SEED,
            "augmentation": "existing V1 gamma orbit config (exact parameters and registry hash in implementation fingerprints); per update sample_seed=model_seed*100000+logical_update; config seed=model_seed; independent from model RNG",
            "two_view": {"C1": True, "C2": True, "M3": True, "S-X": True, "F-R": True, "C0": False, "SNR-B3-adapted": False},
        },
        "arm_definitions": {
            "C0": {"grade_path": "base", "contrastive": "none", "loss": "grade NLL"},
            "C1": {"grade_path": "x", "contrastive": "axis residual r", "loss": "V1 ordinal/style/orbit/severity auxiliaries"},
            "C2": {"grade_path": "D(x)", "contrastive": "full x", "loss": "V1 ordinal/style/orbit/severity auxiliaries"},
            "M3": {"grade_path": "P x + D((I-P)x)", "contrastive": "residual r", "loss": "V1 ordinal/style/orbit/severity auxiliaries"},
            "S-X": {"grade_path": "P x + D((I-P)x)", "contrastive": "full x", "loss": "V1 ordinal/style/orbit/severity auxiliaries"},
            "F-R": {"grade_path": "D(x)", "contrastive": "axis residual r", "loss": "V1 ordinal/style/orbit/severity auxiliaries"},
            "SNR-B3-adapted": {
                "source_commit": SNR_SOURCE_COMMIT,
                "positions": ["blocks.2:48", "blocks.4:136"],
                "operator": "IN; R=x-IN(x); gate=GAP->Conv1x1->ReLU->Conv1x1->sigmoid; Fplus=IN+gate*R; Fminus=IN+(1-gate)*R; Fplus is inserted into the actual main path",
                "gate_reduction": 16, "hidden_widths": [3, 8], "shared_classifier_within_position": True,
                "loss": "grade NLL + 0.01*sum(batch-mean dual entropy losses at both positions) + 0.01*CE(136-channel Fplus,grade)",
                "uses_v1_axis_or_lambda": False,
            },
        },
        "threshold": {
            "score": "q2=p2+p3+p4", "positive_rule": "q2>=tau",
            "candidates": "unique q2 values from EyePACS CAL plus 0; ties kept together",
            "criterion": "maximum tau with one-sided 90% Wilson sensitivity lower bound >=0.90; z=1.2815515655",
            "minimum_positive_images": 15, "minimum_negative_images": 1,
            "if_unavailable": "CAL_INSUFFICIENT or NOT_ESTIMABLE; do not borrow a threshold; threshold-free metrics continue",
            "clinical_estimands": "estimation_only; no noninferiority or clinical acceptance margin",
            "clinical_metrics": ["false_referral_grade0", "false_referral_grade1", "false_referral_grade01", "miss_grade2", "miss_grade3", "miss_grade4", "miss_grade234", "sensitivity", "specificity"],
        },
        "primary_estimands": {
            "contrasts": ["M3-C2"], "metrics": ["QWK", "AUROC(q2)"],
            "per_seed_cohort_delta": "Y[M3,seed,cohort]-Y[C2,seed,cohort]",
            "summary": "mean over 5 seeds of the equal-weight mean of Retino and Set-B deltas",
            "bootstrap_summary": "for each replicate, average the five within-cohort paired deltas; then take the equal-weight mean of the independently resampled Retino and Set-B cohort deltas",
            "joint_rule": "both two-sided 95% paired cluster-bootstrap lower bounds >0",
            "interpretation_scope": "five fixed model pairs and two already exposed evaluation cohorts; not seed-population stability, per-cohort significance, clinical importance, or external confirmation",
        },
        "evaluation_protocol": {
            "models": "all 35 terminal models locked before any CAL prediction",
            "calibration": "all EyePACS rows in the frozen CAL manifest, in manifest order; one q2 threshold per model using only EyePACS CAL labels",
            "cal_reload_check": "for each model, the first min(32, EyePACS_CAL_count) EyePACS CAL rows in frozen manifest order; saved-vs-reloaded probability and q2 tolerance rtol=1e-6, atol=1e-7",
            "evaluation_order": "CAL predictions -> lock all 35 threshold decisions -> freeze DEV and existing Retino/SetB inputs -> all-model CAL reload check -> DEV/Retino/SetB scoring -> shared statistics",
            "image_forward_inputs": 35 * (sum(int(source["images_by_role"]["DEV"]) for source in input_receipt["source_summary"].values()) + int(input_receipt["cal_eye_image_count"]) + 1392 + 400 + min(32, int(input_receipt["cal_eye_image_count"]))),
            "engineering_full_forward_inputs_cap": 2048,
        },
        "bootstrap": {
            "valid_replicates": 10000, "attempt_cap": 20000,
            "cluster_unit": {"Retino": "registered image component", "SetB": "registered patient"},
            "shared_draws": "all arms and seeds within cohort",
            "rng": "PCG64(20261001), independent cohort streams",
            "interval": "paired percentile 2.5%/97.5%, linear interpolation",
            "seed_resampling": False,
        },
        "resources": {
            "gpu_hours": GPU_HOUR_CAP, "cpu_core_hours": CPU_CORE_HOUR_CAP,
            "wall_hours": WALL_HOUR_CAP, "additional_disk_gib": DISK_GIB_CAP,
            "max_gpus": MAX_GPUS, "updates_total_cap": PHYSICAL_UPDATE_CAP,
            "formal_updates": FORMAL_UPDATES * len(ARMS) * len(SEEDS),
            "preflight_updates_cap": PREFLIGHT_MAX_TOTAL,
            "preflight_updates_reserved": int(implementation_receipt.get("prior_preflight_failure_context", {}).get("physical_updates", 0)) + PREFLIGHT_UPDATES_PER_RUN * len(ARMS) * len(SEEDS),
            "prior_failed_preflight_updates": int(implementation_receipt.get("prior_preflight_failure_context", {}).get("physical_updates", 0)),
            "replay_updates_reserved_after_prior_wave_and_base_preflight": EXTRA_REPLAY_CAP - int(implementation_receipt.get("prior_preflight_failure_context", {}).get("physical_updates", 0)) - PREFLIGHT_UPDATES_PER_RUN * len(ARMS) * len(SEEDS),
            "preflight_retry_policy": "restart the same fixed arm/seed after a failed attempt; count all completed updates; no more than 200 total optimizer updates per arm/seed; every retry reduces the shared 7000-update pool available to preflight and formal replay",
            "resource_margin": RESOURCE_MARGIN,
            "remaining_cost_gate": "consumed + 1.2*estimated_remaining <= 0.9*hard_limit for GPU, CPU, disk, and wall",
            "dynamic_fill": "up to four registered G1 GPUs, one independent world_size=1 job per GPU; refill immediately on job completion",
        },
        "input_fingerprints": input_receipt,
        "implementation_fingerprints": implementation_receipt,
        "created_utc": utc_now(),
    }


def freeze_contract(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    input_receipt_path = run_root / "inputs/INPUT_SPLIT_RECEIPT.json"
    if not input_receipt_path.is_file():
        raise FileNotFoundError("P4 inputs must be frozen before the execution contract")
    input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
    import torch, timm
    snr_head = subprocess.run(["git", "-C", str(SNR_SOURCE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if snr_head != SNR_SOURCE_COMMIT:
        raise RuntimeError(f"SNR source checkout differs from the frozen commit: {snr_head}")
    snr_file_hashes = {name: sha256_file(SNR_SOURCE_ROOT / name) for name in SNR_SOURCE_FILES}
    if snr_file_hashes != SNR_SOURCE_FILES:
        raise RuntimeError("pinned SNR source file hashes changed")
    if sha256_file(REPO / "scripts/dr/sparrow/SNR_LICENSE_MIT.txt") != SNR_SOURCE_FILES["LICENSE"]:
        raise RuntimeError("copied MIT license does not match the pinned upstream license")
    historical_head = subprocess.run(["git", "-C", str(HISTORICAL_G1), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if str(HISTORICAL_SRC) not in sys.path:
        sys.path.insert(0, str(HISTORICAL_SRC))
    from tasks.diabetic_retinopathy.sparrow_dr.orbits import calibrate_source_quantiles, orbit_registry_hash
    orbit_config = calibrate_source_quantiles(seed=SEEDS[0])
    prior_failure_path = run_root / "PREFLIGHT_PRIOR_FAILED_WAVE.json"
    prior_failure_context = json.loads(prior_failure_path.read_text(encoding="utf-8")) if prior_failure_path.is_file() else {"status": "NO_PRIOR_FAILED_PREFLIGHT_WAVE"}
    implementation_receipt = {
        "execution_source": str(Path(__file__).resolve()),
        "execution_source_sha256": sha256_file(Path(__file__).resolve()),
        "historical_method_source_commit": historical_head,
        "historical_model_source_sha256": sha256_file(HISTORICAL_SRC / "tasks/diabetic_retinopathy/sparrow_dr/model.py"),
        "historical_loss_source_sha256": sha256_file(HISTORICAL_SRC / "tasks/diabetic_retinopathy/sparrow_dr/losses.py"),
        "historical_orbit_source_sha256": sha256_file(HISTORICAL_SRC / "tasks/diabetic_retinopathy/sparrow_dr/orbits.py"),
        "v1_appearance_orbit_config_seed_2101": orbit_config.to_dict(),
        "v1_appearance_orbit_registry_hash_seed_2101": orbit_registry_hash(orbit_config),
        "v1_appearance_orbit_seed_rule": "config seed=model_seed; apply_orbit sample_seed=model_seed*100000+logical_update",
        "prior_preflight_failure_context": prior_failure_context,
        "analysis_source": str(REPO / "scripts/dr/sparrow/p4_analysis.py"),
        "analysis_source_sha256": sha256_file(REPO / "scripts/dr/sparrow/p4_analysis.py"),
        "snr_source_checkout": str(SNR_SOURCE_ROOT),
        "snr_source_commit": snr_head,
        "snr_source_files": snr_file_hashes,
        "snr_license_copied_sha256": sha256_file(REPO / "scripts/dr/sparrow/SNR_LICENSE_MIT.txt"),
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "timm_version": timm.__version__, "numpy": np.__version__,
        "g1_host": subprocess.run(["hostname"], capture_output=True, text=True, check=True).stdout.strip(),
    }
    payload = _contract_payload(input_receipt, implementation_receipt)
    digest = sha256_bytes(canonical_json_bytes(payload))
    contract = dict(payload)
    contract["contract_sha256"] = digest
    path = run_root / "EXECUTION_CONTRACT.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != digest:
            raise FileExistsError("frozen contract differs from existing contract; do not overwrite")
        return previous
    write_json(path, contract)
    return contract


def _process_cpu_seconds(which: str) -> float:
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF if which == "self" else resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _contract_hash_from_run(run_root: Path) -> str:
    path = run_root / "EXECUTION_CONTRACT.json"
    if not path.is_file():
        return "CONTRACT_NOT_YET_FROZEN"
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(value["contract_sha256"])


def dynamic_schedule(
    *, mode: str,
    gpu_indices: Sequence[int],
    run_root: Path = RUN_ROOT,
    updates: int | None = None,
    resume: bool = False,
    required_preflight_updates: int | None = None,
) -> dict[str, Any]:
    if len(gpu_indices) > MAX_GPUS or not gpu_indices:
        raise ValueError(f"select 1..{MAX_GPUS} available G1 GPUs")
    planned = PREFLIGHT_UPDATES_PER_RUN if mode == "preflight" else FORMAL_UPDATES
    if updates is not None:
        planned = int(updates)
    if mode == "preflight" and planned > PREFLIGHT_MAX_PER_RUN:
        raise ValueError("per-arm/seed preflight exceeds the fixed 200-update cap")
    if mode not in {"preflight", "formal"}:
        raise ValueError(mode)
    queue_all = [{"arm": arm, "seed": seed} for arm in ARMS for seed in SEEDS]
    status_path = run_root / ("PREFLIGHT_SCHEDULER.json" if mode == "preflight" else "FORMAL_SCHEDULER.json")
    if status_path.exists():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if not resume or prior_status.get("status") in {"PREFLIGHT_QUEUE_COMPLETE", "FORMAL_QUEUE_COMPLETE"}:
            raise FileExistsError(f"scheduler receipt already exists: {status_path}")
        archived_status = status_path.with_name(status_path.stem + f"_FAILED_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
        os.replace(status_path, archived_status)
    if not (run_root / "inputs/INPUT_SPLIT_RECEIPT.json").is_file() or not (run_root / "EXECUTION_CONTRACT.json").is_file():
        raise RuntimeError("input split and frozen contract must exist before any P4 worker starts")
    queue: list[dict[str, Any]] = []
    existing_completed: list[dict[str, Any]] = []
    queue_keys = {(str(job["arm"]), int(job["seed"])) for job in queue_all}
    attempt_counts = {key: 0 for key in queue_keys}
    replay_counter = 0
    failed_gpu_seconds = 0.0
    failed_worker_cpu_upper_bound = 0.0
    preflight_retry_updates = 0
    failures: list[dict[str, Any]] = []
    prior_failure_path = run_root / "PREFLIGHT_PRIOR_FAILED_WAVE.json"
    prior_failure = json.loads(prior_failure_path.read_text(encoding="utf-8")) if prior_failure_path.is_file() else {}
    prior_preflight_updates = int(prior_failure.get("physical_updates", 0))
    prior_preflight_gpu_seconds = float(prior_failure.get("gpu_seconds_upper_bound", 0.0))
    prior_preflight_cpu_seconds = float(prior_failure.get("cpu_core_seconds_upper_bound", 0.0))
    if mode != "preflight":
        prior_preflight_gpu_seconds = 0.0
        prior_preflight_cpu_seconds = 0.0
    log_path = run_root / ("preflight_scheduler.jsonl" if mode == "preflight" else "formal_scheduler.jsonl")
    if mode == "formal" and (run_root / "BUDGET_RESERVATION.json").is_file():
        budget = json.loads((run_root / "BUDGET_RESERVATION.json").read_text(encoding="utf-8"))
        if budget.get("status") != "PREFLIGHT_RESOURCE_GATE_PASS" or budget.get("contract_sha256") != _contract_hash_from_run(run_root):
            raise RuntimeError("formal schedule lacks a passing reservation for this frozen contract")
    elif mode == "formal":
        raise RuntimeError("formal training is blocked until the preflight resource gate passes")
    if resume and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "worker_failure":
                failures.append(event)
                if mode == "formal":
                    replay_counter += int(event.get("replay_updates_charged", 0))
                failed_gpu_seconds += float(event.get("elapsed_seconds", 0.0))
                failed_worker_cpu_upper_bound += float(
                    event.get("failed_worker_cpu_core_seconds_upper_bound", float(event.get("elapsed_seconds", 0.0)) * (WORKER_COUNT + 1))
                )
            if event.get("event") == "launch":
                key = (event["arm"], int(event["seed"]))
                attempt_counts[key] = max(attempt_counts[key], int(event["attempt"]) + 1)
    for job in queue_all:
        arm, seed = job["arm"], int(job["seed"])
        arm_dir = run_root / mode / arm.replace("-", "_") / f"seed_{seed}"
        if mode == "formal":
            canonical_receipt = arm_dir / "run_receipt.json"
            final_model = arm_dir / "final_model.pt"
            if canonical_receipt.is_file() and final_model.is_file():
                receipt = json.loads(canonical_receipt.read_text(encoding="utf-8"))
                if receipt.get("status") == "FORMAL_TRAINING_COMPLETE" and sha256_file(final_model) == receipt.get("final_model_sha256"):
                    existing_completed.append({"arm": arm, "seed": seed, "receipt_path": str(canonical_receipt), "allocated_gpu_seconds": float(receipt.get("allocated_gpu_wall_seconds", 0.0)), "status": "FORMAL_TRAINING_COMPLETE"})
                    continue
            progress_files = list(arm_dir.glob("progress.json")) + list(arm_dir.glob("attempt_*.progress.json"))
            recovery_exists = (arm_dir / "recovery.pt").is_file()
            if not resume and (recovery_exists or progress_files):
                raise RuntimeError(f"partial formal state exists for {arm}/{seed}; rerun schedule with --resume")
            job["resume"] = bool(resume and recovery_exists)
            if resume and progress_files:
                if not any(x.get("arm") == arm and int(x.get("seed", -1)) == seed for x in failures):
                    progress_file = sorted(progress_files, key=lambda p: p.stat().st_mtime)[-1]
                    progress = json.loads(progress_file.read_text(encoding="utf-8"))
                    recovery_meta = arm_dir / "recovery_meta.json"
                    checkpoint_next = int(json.loads(recovery_meta.read_text(encoding="utf-8")).get("next_update", 0)) if recovery_meta.is_file() else 0
                    replay_counter += max(1, int(progress.get("logical_update", 0)) - checkpoint_next + 1)
        else:
            canonical_receipt = arm_dir / "run_receipt.json"
            if canonical_receipt.is_file():
                receipt = json.loads(canonical_receipt.read_text(encoding="utf-8"))
                if receipt.get("status") == "PREFLIGHT_COMPLETE" and receipt.get("logical_updates_done_this_process") == planned and receipt.get("contract_sha256") == _contract_hash_from_run(run_root):
                    physical_done = sum(_attempt_progress_updates(arm_dir, attempt) for attempt in _attempt_ids_with_progress(arm_dir))
                    existing_completed.append({
                        "arm": arm, "seed": seed, "receipt_path": str(canonical_receipt),
                        "allocated_gpu_seconds": float(receipt.get("allocated_gpu_wall_seconds", 0.0)),
                        "physical_updates": physical_done or int(receipt.get("physical_optimizer_updates_this_process", planned)),
                        "status": "PREFLIGHT_COMPLETE",
                    })
                    continue
            attempt_ids = _attempt_ids_with_progress(arm_dir) if arm_dir.exists() else []
            has_artifacts = arm_dir.exists() and any(arm_dir.iterdir())
            if has_artifacts and not resume:
                raise FileExistsError(f"preflight artifacts already exist for {arm}/{seed}; resume required to preserve and finish")
            prior_updates = sum(_attempt_progress_updates(arm_dir, attempt) for attempt in attempt_ids)
            if prior_updates + planned > PREFLIGHT_MAX_PER_RUN:
                raise RuntimeError(f"preflight retry would exceed the 200-update per-run cap for {arm}/{seed}")
            attempt_counts[(arm, seed)] = max(attempt_ids, default=-1) + 1
            job["prior_updates"] = prior_updates
            job["resume"] = False
        if mode == "formal" and resume:
            attempt_counts[(arm, seed)] = max(attempt_counts[(arm, seed)], 1 if (arm_dir / "progress.json").exists() else 0)
            attempt_counts[(arm, seed)] = max(attempt_counts[(arm, seed)], 1 + max([int(p.name.split("_")[1].split(".")[0]) for p in arm_dir.glob("attempt_*.progress.json")], default=-1))
        queue.append(job)
    local_preflight_prior_updates = 0
    if mode == "preflight":
        existing_preflight_updates = sum(int(item.get("physical_updates", 0)) for item in existing_completed)
        retry_preflight_updates = sum(int(job.get("prior_updates", 0)) for job in queue)
        local_preflight_prior_updates = existing_preflight_updates + retry_preflight_updates
        prospective_preflight_updates = prior_preflight_updates + existing_preflight_updates + retry_preflight_updates + planned * len(queue)
        if prospective_preflight_updates > PREFLIGHT_MAX_TOTAL:
            raise RuntimeError(f"preflight attempts would exceed the {PREFLIGHT_MAX_TOTAL}-update global cap: {prospective_preflight_updates}")
    preflight_scheduler_path = run_root / "PREFLIGHT_SCHEDULER.json"
    preflight_updates_used = 0
    if mode == "formal":
        if not preflight_scheduler_path.is_file():
            raise RuntimeError("formal training requires the complete preflight scheduler receipt")
        preflight_updates_used = int(json.loads(preflight_scheduler_path.read_text(encoding="utf-8")).get("updates_executed_total", -1))
        minimum_preflight_updates = (
            int(required_preflight_updates)
            if required_preflight_updates is not None
            else PREFLIGHT_UPDATES_PER_RUN * len(ARMS) * len(SEEDS)
        )
        if preflight_updates_used < minimum_preflight_updates:
            raise RuntimeError("formal training preflight update accounting is incomplete")
    if mode == "formal" and preflight_updates_used + replay_counter > EXTRA_REPLAY_CAP:
        raise RuntimeError("preflight plus replay update pool exhausted before resuming")
    task_dir = run_root / mode
    task_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cpu_start = _process_cpu_seconds("self")
    gpu_records = _live_gpu_records()
    chosen = [next((g for g in gpu_records if g["index"] == int(i)), None) for i in gpu_indices]
    if any(g is None for g in chosen):
        raise RuntimeError("requested GPU index missing from current nvidia-smi inventory")
    if any(g["utilization_percent"] > 5.0 or g["memory_used_mib"] > 512 for g in chosen):
        raise RuntimeError("requested G1 GPU is occupied; no G2 device may be borrowed")
    procs: dict[int, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    next_job = 0
    physical_counter = 0
    hard_failure: dict[str, Any] | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def launch(gpu: dict[str, Any], job: dict[str, Any], *, is_resume: bool) -> dict[str, Any]:
        nonlocal physical_counter
        arm, seed = job["arm"], int(job["seed"])
        attempt = attempt_counts[(arm, seed)]
        attempt_counts[(arm, seed)] += 1
        command = [
            sys.executable, str(Path(__file__).resolve()), "worker",
            "--arm", arm, "--seed", str(seed), "--mode", mode,
            "--device-index", "0", "--gpu-uuid", gpu["uuid"],
            "--updates", str(planned), "--run-root", str(run_root),
            "--retry-index", str(attempt),
        ]
        if is_resume:
            command.append("--resume")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        stdout_path = task_dir / f"{arm.replace('-', '_')}_seed{seed}_attempt{attempt}.stdout.log"
        stdout = stdout_path.open("ab", buffering=0)
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=subprocess.STDOUT, close_fds=True)
        event = {
            "event": "launch", "utc": utc_now(), "pid": proc.pid,
            "arm": arm, "seed": seed, "attempt": attempt,
            "gpu_index": int(gpu["index"]), "gpu_uuid": gpu["uuid"],
            "mode": mode, "resume": is_resume, "reserved_logical_updates": planned,
            "contract_sha256": _contract_hash_from_run(run_root),
            "stdout_path": str(stdout_path),
        }
        append_jsonl(log_path, event)
        return {"process": proc, "stdout": stdout, "job": job, "gpu": gpu, "attempt": attempt, "start": time.perf_counter(), "event": event}

    for gpu in chosen:
        if next_job < len(queue):
            job = queue[next_job]
            next_job += 1
            procs[int(gpu["index"])] = launch(gpu, job, is_resume=bool(job.get("resume", False)))
    while procs:
        ended: list[int] = []
        for idx, rec in list(procs.items()):
            ret = rec["process"].poll()
            if ret is None:
                continue
            rec["stdout"].close()
            elapsed = time.perf_counter() - rec["start"]
            job, gpu = rec["job"], rec["gpu"]
            arm, seed = job["arm"], int(job["seed"])
            receipt_path = task_dir / arm.replace("-", "_") / f"seed_{seed}" / (f"attempt_{rec['attempt']}.receipt.json" if rec["attempt"] else "run_receipt.json")
            if ret == 0 and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                physical_counter += int(receipt.get("physical_optimizer_updates_this_process", 0))
                ledger_item = {
                    "event": "complete", "utc": utc_now(), "arm": arm, "seed": seed,
                    "attempt": rec["attempt"], "gpu_uuid": gpu["uuid"],
                    "allocated_gpu_seconds": elapsed,
                    "physical_updates": receipt.get("physical_optimizer_updates_this_process", 0),
                    "receipt_path": str(receipt_path), "status": receipt.get("status"),
                }
                ledger.append(ledger_item)
                append_jsonl(log_path, ledger_item)
                ended.append(idx)
            else:
                failed_gpu_seconds += elapsed
                failed_worker_cpu_upper_bound += elapsed * (WORKER_COUNT + 1)
                arm_dir = task_dir / arm.replace("-", "_") / f"seed_{seed}"
                last_update = _attempt_progress_updates(arm_dir, int(rec["attempt"]))
                recovery_meta = arm_dir / "recovery_meta.json"
                recovery_next = int(json.loads(recovery_meta.read_text(encoding="utf-8")).get("next_update", 0)) if recovery_meta.is_file() else 0
                replay_charge = max(1, last_update - recovery_next + 1) if mode == "formal" else int(last_update)
                if mode == "formal":
                    replay_counter += replay_charge
                else:
                    preflight_retry_updates += replay_charge
                    job["prior_updates"] = int(job.get("prior_updates", 0)) + replay_charge
                failure = {
                    "event": "worker_failure", "arm": arm, "seed": seed, "attempt": rec["attempt"],
                    "return_code": ret, "gpu_uuid": gpu["uuid"], "elapsed_seconds": elapsed,
                    "failed_worker_cpu_core_seconds_upper_bound": elapsed * (WORKER_COUNT + 1),
                    "replay_updates_charged": replay_charge,
                    "preflight_prior_updates_for_run": int(job.get("prior_updates", 0)),
                }
                failures.append(failure)
                append_jsonl(log_path, {**failure, "utc": utc_now()})
                can_retry_preflight = hard_failure is None and mode == "preflight" and int(job.get("prior_updates", 0)) + planned <= PREFLIGHT_MAX_PER_RUN
                can_retry_formal = hard_failure is None and mode == "formal" and rec["attempt"] < 2 and preflight_updates_used + replay_counter <= EXTRA_REPLAY_CAP
                if can_retry_preflight or can_retry_formal:
                    procs.pop(idx, None)
                    procs[idx] = launch(gpu, job, is_resume=can_retry_formal and (arm_dir / "recovery.pt").is_file())
                    continue
                if mode == "formal" and preflight_updates_used + replay_counter > EXTRA_REPLAY_CAP:
                    failure["failure_reason"] = "combined preflight plus formal replay update cap exhausted"
                hard_failure = failure
                ended.append(idx)
        for idx in ended:
            if idx not in procs:
                continue
            procs.pop(idx, None)
            if hard_failure is None and next_job < len(queue):
                job = queue[next_job]
                next_job += 1
                gpu = next(g for g in chosen if int(g["index"]) == idx)
                procs[idx] = launch(gpu, job, is_resume=bool(job.get("resume", False)))
        if procs:
            time.sleep(0.25)

    elapsed_total = time.perf_counter() - start
    if mode == "preflight":
        preflight_total_updates = prior_preflight_updates + local_preflight_prior_updates + preflight_retry_updates + physical_counter
    else:
        preflight_total_updates = preflight_updates_used
    completed_keys = {(x["arm"], int(x["seed"])) for x in ledger} | {(x["arm"], int(x["seed"])) for x in existing_completed}
    all_jobs_complete = len(completed_keys) == len(queue_all) and hard_failure is None
    result = {
        "status": ("PREFLIGHT_QUEUE_COMPLETE" if all_jobs_complete else "PREFLIGHT_QUEUE_FAILED") if mode == "preflight" else ("FORMAL_QUEUE_COMPLETE" if all_jobs_complete else "FORMAL_QUEUE_FAILED"),
        "mode": mode, "arms": list(ARMS), "seeds": list(SEEDS),
        "jobs_planned": len(queue_all), "jobs_completed": len(completed_keys),
        "completed_attempts": len(ledger),
        "completed_before_resume": existing_completed,
        "failed_attempts": failures, "per_run_updates": planned,
        "updates_executed_this_queue": physical_counter,
        "updates_executed_total": preflight_total_updates,
        "replay_updates_reserved_or_charged": replay_counter,
        "gpu_assignments": [{"index": g["index"], "uuid": g["uuid"], "name": g["name"]} for g in chosen],
        "max_simultaneous_workers": len(chosen),
        "scheduler_wall_seconds": elapsed_total,
        "scheduler_cpu_core_seconds": max(0.0, _process_cpu_seconds("self") - cpu_start),
        "failed_attempt_gpu_seconds": failed_gpu_seconds,
        "failed_worker_cpu_core_seconds_upper_bound": failed_worker_cpu_upper_bound,
        "allocated_gpu_seconds_sum": prior_preflight_gpu_seconds + sum(float(x["allocated_gpu_seconds"]) for x in ledger) + failed_gpu_seconds + sum(float(x.get("allocated_gpu_seconds", 0.0)) for x in existing_completed),
        "prior_failed_preflight_updates": prior_preflight_updates,
        "prior_failed_preflight_gpu_seconds_upper_bound": prior_preflight_gpu_seconds,
        "prior_failed_preflight_cpu_core_seconds_upper_bound": prior_preflight_cpu_seconds,
        "completion_order": [{"arm": x["arm"], "seed": x["seed"], "gpu_uuid": x["gpu_uuid"]} for x in ledger],
        "ledger_path": str(log_path),
        "updated_utc": utc_now(),
    }
    write_json(status_path, result)
    if not all_jobs_complete:
        raise RuntimeError(f"P4 {mode} queue stopped after safely draining in-flight workers; see {status_path} and {log_path}; last failure={hard_failure}")
    return result


def _live_gpu_records() -> list[dict[str, Any]]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    records = []
    for line in output.splitlines():
        fields = [x.strip() for x in line.split(",")]
        records.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(float(fields[3])), "memory_used_mib": int(float(fields[4])),
            "utilization_percent": float(fields[5]),
        })
    return records


def _tree_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def benchmark_evaluation_forward(*, gpu_index: int, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    """Measure all seven full B3 evaluation paths on synthetic inputs only."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation qualification requires CUDA")
    records = _live_gpu_records()
    gpu = next((item for item in records if item["index"] == int(gpu_index)), None)
    if gpu is None or gpu["utilization_percent"] > 5.0 or gpu["memory_used_mib"] > 512:
        raise RuntimeError("evaluation qualification GPU is unavailable or occupied")
    torch.cuda.set_device(int(gpu_index))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device="cpu").manual_seed(20261001)
    sample = torch.randn((BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE), generator=generator, dtype=torch.float32).cuda(int(gpu_index))
    rates: dict[str, Any] = {}
    started_cpu = _process_cpu_seconds("self")
    started_wall = time.perf_counter()
    total_image_forwards = 0
    for arm in ARMS:
        model, init_receipt = build_p4_model(arm, SEEDS[0])
        model.to(torch.device("cuda", int(gpu_index)))
        model.eval()
        timings: list[float] = []
        with torch.inference_mode():
            for repeat in range(5):
                torch.cuda.synchronize(int(gpu_index))
                tick = time.perf_counter()
                output = model(sample)
                if arm == "SNR-B3-adapted":
                    model.feature_pyramid.pop_aux()
                torch.cuda.synchronize(int(gpu_index))
                elapsed = time.perf_counter() - tick
                total_image_forwards += BATCH_SIZE
                if repeat >= 2:
                    timings.append(elapsed)
        rates[arm] = {
            "seconds_per_batch16": timings,
            "median_seconds_per_image": float(np.median(timings) / BATCH_SIZE),
            "model_parameter_count": init_receipt["model_parameter_count"],
            "trainable_parameter_count": init_receipt["trainable_parameter_count"],
            "initial_state_sha256": init_receipt["model_state_sha256"],
        }
        del model, output
        torch.cuda.empty_cache()
    del sample
    elapsed_wall = time.perf_counter() - started_wall
    receipt = {
        "status": "EVALUATION_FORWARD_QUALIFICATION_COMPLETE",
        "gpu_index": int(gpu_index), "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
        "rates": rates,
        "engineering_image_forward_inputs": total_image_forwards,
        "engineering_forward_limit": 2048,
        "elapsed_wall_seconds": elapsed_wall,
        "cpu_core_seconds": max(0.0, _process_cpu_seconds("self") - started_cpu),
        "completed_utc": utc_now(),
    }
    if total_image_forwards > 2048:
        raise RuntimeError("engineering qualification exceeded its 2,048 image-forward limit")
    write_json(run_root / "EVALUATION_FORWARD_QUALIFICATION.json", receipt)
    return receipt


def run_preflight_resource_gate(*, gpu_index: int, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    """One fixed pre-result budget gate, with 20% forecast margin and 10% reserve."""
    gate_cpu_started = _process_cpu_seconds("self")
    preflight_path = run_root / "PREFLIGHT_SCHEDULER.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("all preflight runs must finish before the resource gate")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PREFLIGHT_QUEUE_COMPLETE" or preflight.get("jobs_completed") != len(ARMS) * len(SEEDS):
        raise RuntimeError("the complete 35-run preflight queue is not successful")
    receipts: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = run_root / "preflight" / arm.replace("-", "_") / f"seed_{seed}" / "run_receipt.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("status") != "PREFLIGHT_COMPLETE" or not receipt.get("preflight_recovery_reload_verified"):
                raise RuntimeError(f"preflight run failed qualification: {arm}/{seed}")
            if receipt.get("logical_updates_done_this_process") != PREFLIGHT_UPDATES_PER_RUN:
                raise RuntimeError(f"preflight update count differs: {arm}/{seed}")
            receipts[(arm, seed)] = receipt

    eval_benchmark_path = run_root / "EVALUATION_FORWARD_QUALIFICATION.json"
    if eval_benchmark_path.is_file():
        raise FileExistsError(eval_benchmark_path)
    eval_rates = benchmark_evaluation_forward(gpu_index=gpu_index, run_root=run_root)
    split_receipt = json.loads((run_root / "inputs/INPUT_SPLIT_RECEIPT.json").read_text(encoding="utf-8"))
    n_dev = sum(int(v["images_by_role"]["DEV"]) for v in split_receipt["source_summary"].values())
    n_cal_eye = int(split_receipt["cal_eye_image_count"])
    n_external = 1392 + 400
    n_reload = min(32, n_cal_eye)
    eval_input_forwards = len(ARMS) * len(SEEDS) * (n_dev + n_cal_eye + n_external) + len(ARMS) * len(SEEDS) * n_reload

    by_arm_rate: dict[str, float] = {}
    by_arm_cpu_rate: dict[str, float] = {}
    by_arm_startup: dict[str, float] = {}
    max_state_bytes = 0
    for arm in ARMS:
        arm_receipts = [receipts[(arm, seed)] for seed in SEEDS]
        rates = [float(r["p95_step_wall_seconds"]) for r in arm_receipts if r.get("p95_step_wall_seconds") is not None]
        if len(rates) != len(SEEDS) or any(not math.isfinite(x) or x <= 0 for x in rates):
            raise RuntimeError(f"invalid measured p95 update rate for {arm}")
        by_arm_rate[arm] = max(rates)
        by_arm_cpu_rate[arm] = max(float(r["cpu_core_seconds"]) / PREFLIGHT_UPDATES_PER_RUN for r in arm_receipts)
        by_arm_startup[arm] = max(max(0.0, float(r["allocated_gpu_wall_seconds"]) - float(r["training_loop_wall_seconds"])) for r in arm_receipts)
        max_state_bytes = max(max_state_bytes, *(int(r["recovery_checkpoint_bytes"]) for r in arm_receipts))

    training_remaining_gpu_seconds = sum(
        len(SEEDS) * FORMAL_UPDATES * by_arm_rate[arm] for arm in ARMS
    )
    preflight_updates_total = int(preflight.get("updates_executed_total", PREFLIGHT_UPDATES_PER_RUN * len(ARMS) * len(SEEDS)))
    if preflight_updates_total > EXTRA_REPLAY_CAP:
        raise RuntimeError("completed preflight attempts exceed the shared preflight/replay update pool")
    replay_remaining_updates = EXTRA_REPLAY_CAP - preflight_updates_total
    replay_remaining_gpu_seconds = replay_remaining_updates
    replay_rate = max(by_arm_rate.values())
    replay_remaining_gpu_seconds *= replay_rate
    eval_remaining_gpu_seconds = sum(
        len(SEEDS) * (n_dev + n_cal_eye + n_external + n_reload) * float(eval_rates["rates"][arm]["median_seconds_per_image"])
        for arm in ARMS
    )
    formal_startup_gpu_seconds = sum(len(SEEDS) * by_arm_startup[arm] for arm in ARMS)
    evaluation_startup_gpu_seconds = formal_startup_gpu_seconds
    gpu_consumed_seconds = float(preflight.get("allocated_gpu_seconds_sum", 0.0)) + float(eval_rates["elapsed_wall_seconds"])
    gpu_remaining_seconds = training_remaining_gpu_seconds + replay_remaining_gpu_seconds + eval_remaining_gpu_seconds + formal_startup_gpu_seconds + evaluation_startup_gpu_seconds
    gpu_projected_hours = (gpu_consumed_seconds + RESOURCE_MARGIN * gpu_remaining_seconds) / 3600.0

    cpu_consumed_seconds = float(split_receipt.get("p4_prepare_cpu_core_seconds", 0.0)) + float(preflight.get("scheduler_cpu_core_seconds", 0.0)) + sum(float(r["cpu_core_seconds"]) for r in receipts.values()) + float(eval_rates["cpu_core_seconds"]) + float(preflight.get("failed_worker_cpu_core_seconds_upper_bound", 0.0)) + float(preflight.get("prior_failed_preflight_cpu_core_seconds_upper_bound", 0.0))
    cpu_training_remaining = sum(len(SEEDS) * FORMAL_UPDATES * by_arm_cpu_rate[arm] for arm in ARMS)
    cpu_replay_remaining = replay_remaining_gpu_seconds * max(by_arm_cpu_rate[arm] / by_arm_rate[arm] for arm in ARMS)
    cpu_per_image = max(by_arm_cpu_rate[arm] / BATCH_SIZE for arm in ARMS)
    cpu_eval_remaining = eval_input_forwards * cpu_per_image
    cpu_bootstrap_reserve = 24.0 * 3600.0
    cpu_remaining_seconds = cpu_training_remaining + cpu_replay_remaining + cpu_eval_remaining + cpu_bootstrap_reserve
    cpu_projected_hours = (cpu_consumed_seconds + RESOURCE_MARGIN * cpu_remaining_seconds) / 3600.0

    train_task_walls = [FORMAL_UPDATES * by_arm_rate[arm] + by_arm_startup[arm] for arm in ARMS for _ in SEEDS]
    train_wall_seconds = sum(train_task_walls) / MAX_GPUS + max(train_task_walls)
    eval_task_walls = [
        (n_dev + n_cal_eye + n_external + n_reload) * float(eval_rates["rates"][arm]["median_seconds_per_image"])
        for arm in ARMS for _ in SEEDS
    ]
    eval_wall_seconds = sum(eval_task_walls) / MAX_GPUS + max(eval_task_walls) + max(by_arm_startup.values())
    session = json.loads((run_root / "SESSION.json").read_text(encoding="utf-8"))
    session_start = datetime.fromisoformat(session["session_started_utc"].replace("Z", "+00:00"))
    wall_consumed_seconds = max(0.0, (datetime.now(timezone.utc) - session_start).total_seconds())
    wall_remaining_seconds = train_wall_seconds + eval_wall_seconds + cpu_bootstrap_reserve
    wall_projected_hours = (wall_consumed_seconds + RESOURCE_MARGIN * wall_remaining_seconds) / 3600.0

    current_bytes = _tree_bytes(run_root)
    eval_artifacts_reserved = eval_input_forwards * 512 + len(ARMS) * len(SEEDS) * 2_000_000
    model_artifacts_reserved = len(ARMS) * len(SEEDS) * 2 * max_state_bytes
    training_log_reserved = len(ARMS) * len(SEEDS) * 32_000_000
    disk_remaining_bytes = model_artifacts_reserved + eval_artifacts_reserved + training_log_reserved
    disk_projected_gib = (current_bytes + RESOURCE_MARGIN * disk_remaining_bytes) / (1024.0 ** 3)
    free_bytes = shutil.disk_usage(run_root.parent).free
    disk_free_after_reserved_gib = (free_bytes - RESOURCE_MARGIN * disk_remaining_bytes) / (1024.0 ** 3)

    resource_rows = {
        "gpu_hours": {"consumed": gpu_consumed_seconds / 3600.0, "remaining_estimate": gpu_remaining_seconds / 3600.0, "projected_with_20pct": gpu_projected_hours, "limit_with_10pct_reserve": GPU_HOUR_CAP * 0.9, "pass": gpu_projected_hours <= GPU_HOUR_CAP * 0.9},
        "cpu_core_hours": {"consumed": cpu_consumed_seconds / 3600.0, "remaining_estimate": cpu_remaining_seconds / 3600.0, "projected_with_20pct": cpu_projected_hours, "limit_with_10pct_reserve": CPU_CORE_HOUR_CAP * 0.9, "pass": cpu_projected_hours <= CPU_CORE_HOUR_CAP * 0.9},
        "wall_hours": {"consumed": wall_consumed_seconds / 3600.0, "remaining_estimate": wall_remaining_seconds / 3600.0, "projected_with_20pct": wall_projected_hours, "limit_with_10pct_reserve": WALL_HOUR_CAP * 0.9, "pass": wall_projected_hours <= WALL_HOUR_CAP * 0.9},
        "incremental_disk_gib": {"consumed": current_bytes / (1024.0 ** 3), "remaining_estimate": disk_remaining_bytes / (1024.0 ** 3), "projected_with_20pct": disk_projected_gib, "limit_with_10pct_reserve": DISK_GIB_CAP * 0.9, "free_after_reserved_gib": disk_free_after_reserved_gib, "pass": disk_projected_gib <= DISK_GIB_CAP * 0.9 and disk_free_after_reserved_gib > 0},
    }
    passed = all(row["pass"] for row in resource_rows.values())
    gate = {
        "status": "PREFLIGHT_RESOURCE_GATE_PASS" if passed else "BUDGET_REVIEW",
        "plan_commit": PLAN_COMMIT,
        "execution_contract_sha256": _contract_hash_from_run(run_root),
        "preflight_runs": len(receipts),
        "preflight_updates": preflight_updates_total,
        "measured_p95_step_seconds_by_arm": by_arm_rate,
        "measured_cpu_core_seconds_per_update_by_arm": by_arm_cpu_rate,
        "measured_evaluation_seconds_per_image_by_arm": {k: v["median_seconds_per_image"] for k, v in eval_rates["rates"].items()},
        "evaluation_image_forward_count_from_contract": eval_input_forwards,
        "engineering_image_forward_count": eval_rates["engineering_image_forward_inputs"],
        "engineering_image_forward_cap": 2048,
        "cost_table": resource_rows,
        "resource_gate_cpu_core_seconds": max(0.0, _process_cpu_seconds("self") - gate_cpu_started),
        "estimate_components": {
            "formal_optimizer_updates": FORMAL_UPDATES * len(ARMS) * len(SEEDS),
            "preflight_optimizer_updates": preflight_updates_total,
            "max_replay_updates_reserved": replay_remaining_gpu_seconds / max(replay_rate, 1e-12),
            "formal_gpu_training_hours_before_margin": training_remaining_gpu_seconds / 3600.0,
            "evaluation_gpu_hours_before_margin": eval_remaining_gpu_seconds / 3600.0,
            "cpu_bootstrap_reserve_hours": cpu_bootstrap_reserve / 3600.0,
            "actual_max_preflight_full_state_bytes": max_state_bytes,
            "disk_reserved_model_and_recovery_bytes": model_artifacts_reserved,
            "disk_reserved_eval_artifacts_bytes": eval_artifacts_reserved,
            "disk_reserved_training_logs_bytes": training_log_reserved,
            "available_cards": _live_gpu_records(),
        },
        "created_utc": utc_now(),
    }
    write_json(run_root / "PREFLIGHT_RESOURCE_GATE.json", gate)
    if passed:
        reservation_path = run_root / "BUDGET_RESERVATION.json"
        reservation = {
            "status": gate["status"],
            "contract_sha256": gate["execution_contract_sha256"],
            "plan_commit": PLAN_COMMIT,
            "hard_caps": {"gpu_hours": GPU_HOUR_CAP, "cpu_core_hours": CPU_CORE_HOUR_CAP, "wall_hours": WALL_HOUR_CAP, "incremental_disk_gib": DISK_GIB_CAP, "extra_updates": EXTRA_REPLAY_CAP},
            "cost_table": resource_rows,
            "reserved_for": "all fixed 35 formal chains, bounded recovery pool, locked CAL/DEV/Retino/SetB inference, shared statistics",
            "created_utc": utc_now(),
        }
        if reservation_path.exists():
            existing = json.loads(reservation_path.read_text(encoding="utf-8"))
            if existing.get("contract_sha256") != reservation["contract_sha256"]:
                raise FileExistsError("a different contract already owns the P4 resource reservation")
        else:
            write_json(reservation_path, reservation)
    return gate


def _read_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, default=RUN_ROOT)
    prepare.add_argument("--skip-cache-content-hash", action="store_true")
    freeze = sub.add_parser("freeze-contract")
    freeze.add_argument("--run-root", type=Path, default=RUN_ROOT)
    gate = sub.add_parser("resource-gate")
    gate.add_argument("--run-root", type=Path, default=RUN_ROOT)
    gate.add_argument("--gpu-index", required=True, type=int)
    worker = sub.add_parser("worker")
    worker.add_argument("--arm", required=True)
    worker.add_argument("--seed", required=True, type=int)
    worker.add_argument("--mode", choices=("preflight", "formal"), required=True)
    worker.add_argument("--device-index", type=int, required=True)
    worker.add_argument("--gpu-uuid", required=True)
    worker.add_argument("--updates", type=int, required=True)
    worker.add_argument("--run-root", type=Path, default=RUN_ROOT)
    worker.add_argument("--resume", action="store_true")
    worker.add_argument("--retry-index", type=int, default=0)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--mode", choices=("preflight", "formal"), required=True)
    schedule.add_argument("--gpus", default="0,1,2,3")
    schedule.add_argument("--updates", type=int)
    schedule.add_argument("--run-root", type=Path, default=RUN_ROOT)
    schedule.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    args = _read_cli()
    if args.command == "prepare":
        result = prepare_inputs(run_root=args.run_root, hash_cache_files=not args.skip_cache_content_hash)
    elif args.command == "freeze-contract":
        result = freeze_contract(args.run_root)
    elif args.command == "resource-gate":
        result = run_preflight_resource_gate(gpu_index=args.gpu_index, run_root=args.run_root)
    elif args.command == "worker":
        result = worker_run(args.arm, args.seed, mode=args.mode, device_index=args.device_index, gpu_uuid=args.gpu_uuid, updates=args.updates, run_root=args.run_root, resume=args.resume, retry_index=args.retry_index)
    elif args.command == "schedule":
        result = dynamic_schedule(mode=args.mode, gpu_indices=[int(x) for x in args.gpus.split(",") if x.strip()], run_root=args.run_root, updates=args.updates, resume=args.resume)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
