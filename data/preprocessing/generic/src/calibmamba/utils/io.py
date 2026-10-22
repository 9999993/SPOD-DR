from __future__ import annotations

import csv
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv_rows(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, object]],
) -> None:
    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    json_path = Path(path)
    ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@lru_cache(maxsize=32)
def load_volume(path: str | Path) -> np.ndarray:
    volume_path = Path(path)
    suffixes = "".join(volume_path.suffixes[-2:]) if len(volume_path.suffixes) >= 2 else volume_path.suffix

    if suffixes == ".nii.gz" or volume_path.suffix == ".nii":
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "nibabel is required to read NIfTI files. Install it before "
                "loading medical image volumes."
            ) from exc

        data = np.asarray(nib.load(str(volume_path)).get_fdata(), dtype=np.float32)
        return canonicalize_medical_volume(data)

    if volume_path.suffix == ".npy":
        return np.asarray(np.load(volume_path), dtype=np.float32)

    if volume_path.suffix == ".npz":
        with np.load(volume_path) as archive:
            first_key = next(iter(archive.keys()))
            return np.asarray(archive[first_key], dtype=np.float32)

    raise ValueError(f"Unsupported volume format: {volume_path}")


def canonicalize_medical_volume(volume: np.ndarray) -> np.ndarray:
    """Return a numeric medical image volume in [z, y, x] layout.

    PLC_CECT stores most files as [x, y, z], but a subset of CT files is stored
    as vector-valued [x, y, z, 1, 3]. The downstream ROI code expects a 3D
    volume indexed as volume[z, y, x], so all readers pass through this helper.
    """

    data = np.asarray(volume, dtype=np.float32)

    while data.ndim > 3 and 1 in data.shape[3:]:
        squeeze_axes = [axis for axis in range(3, data.ndim) if data.shape[axis] == 1]
        data = np.squeeze(data, axis=tuple(squeeze_axes))

    if data.ndim == 4 and data.shape[-1] in {3, 4}:
        data = data[..., :3].mean(axis=-1)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D medical volume after canonicalization, got shape={data.shape}")

    # nibabel exposes NIfTI arrays in x, y, z order. Use z, y, x for slice-wise
    # lesion indexing and 2D ROI extraction.
    return np.transpose(data, (2, 1, 0)).astype(np.float32, copy=False)
