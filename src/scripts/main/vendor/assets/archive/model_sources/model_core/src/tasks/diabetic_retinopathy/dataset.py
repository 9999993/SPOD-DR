from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from calibmamba.utils.io import read_csv_rows


@dataclass
class DRRow:
    patient_id: str
    image_id: str
    label: int
    grade: int
    image_path: str
    cache_path: str
    dataset: str
    split_role: str
    cargo_group_probs: tuple[float, ...] = ()
    cargo_hard_group: int = -1
    cargo_sample_weight: float = 1.0

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DRRow":
        raw_group_probs = str(row.get("cargo_group_probs", "")).strip()
        cargo_group_probs: tuple[float, ...] = ()
        if raw_group_probs:
            parsed = json.loads(raw_group_probs)
            cargo_group_probs = tuple(float(value) for value in parsed)
        return cls(
            patient_id=str(row["patient_id"]),
            image_id=str(row["image_id"]),
            label=int(row["label"]),
            grade=int(row.get("grade", row["label"])),
            image_path=str(row["image_path"]),
            cache_path=str(row.get("cache_path", "")),
            dataset=str(row.get("dataset", "")),
            split_role=str(row.get("split_role", "")),
            cargo_group_probs=cargo_group_probs,
            cargo_hard_group=int(row.get("cargo_hard_group", -1)),
            cargo_sample_weight=float(row.get("cargo_sample_weight", 1.0)),
        )


class DiabeticRetinopathyDataset(Dataset[dict[str, Any]]):
    """Fundus DR dataset that prefers prebuilt numpy caches."""

    def __init__(
        self,
        index_file: str | Path,
        *,
        require_cache: bool = True,
        cache_mode: str = "numpy",
        preload_count: int = 0,
        memory_cache_size: int = 0,
        local_cache_root: str | Path | None = None,
    ) -> None:
        self.index_file = Path(index_file)
        self.require_cache = bool(require_cache)
        self.cache_mode = str(cache_mode).strip().lower()
        if self.cache_mode not in {"numpy", "mmap"}:
            raise ValueError(f"Unsupported DR cache_mode: {cache_mode}")
        self.preload_count = max(int(preload_count), 0)
        self.memory_cache_size = max(int(memory_cache_size), self.preload_count, 0)
        self.local_cache_root = Path(local_cache_root) if local_cache_root else None
        self.rows = self._load_rows(self.index_file)
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        if self.preload_count > 0:
            self._preload_cache_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = self._load_image(row)
        sample = {
            "image": torch.tensor(image, dtype=torch.float32),
            "label": torch.tensor(int(row.label), dtype=torch.float32),
            "grade": torch.tensor(int(row.grade), dtype=torch.long),
            "patient_id": row.patient_id,
            "image_id": row.image_id,
            "dataset": row.dataset,
            "split_role": row.split_role,
            "sample_id": f"{row.dataset}:{row.patient_id}:{row.image_id}",
        }
        if row.cargo_group_probs:
            sample["cargo_group_probs"] = torch.tensor(row.cargo_group_probs, dtype=torch.float32)
            sample["cargo_hard_group"] = int(row.cargo_hard_group)
        if abs(float(row.cargo_sample_weight) - 1.0) > 1e-8:
            sample["cargo_sample_weight"] = torch.tensor(float(row.cargo_sample_weight), dtype=torch.float32)
        return sample

    def _load_rows(self, index_file: Path) -> list[DRRow]:
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        rows = read_csv_rows(index_file)
        if not rows:
            raise ValueError(f"Index file is empty: {index_file}")
        return [DRRow.from_row(row) for row in rows]

    def _remember_array(self, cache_key: str, array: np.ndarray) -> np.ndarray:
        if self.memory_cache_size <= 0:
            return array
        self._memory_cache[cache_key] = array
        self._memory_cache.move_to_end(cache_key)
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
        return array

    def _load_cached_array(self, cache_path: Path) -> np.ndarray:
        mmap_mode = "r" if self.cache_mode == "mmap" else None
        loaded = np.load(cache_path, mmap_mode=mmap_mode, allow_pickle=False)
        if self.cache_mode == "mmap":
            return np.asarray(loaded, dtype=np.float32)
        return np.asarray(loaded, dtype=np.float32)

    def _resolve_cache_path(self, raw_cache_path: str) -> Path:
        cache_path = Path(raw_cache_path)
        if self.local_cache_root is None:
            return cache_path
        if cache_path.exists():
            return cache_path
        parts = list(cache_path.parts)
        for index, part in enumerate(parts):
            if part == "cache" and index + 1 < len(parts):
                return self.local_cache_root / Path(*parts[index + 1 :])
        return self.local_cache_root / cache_path.name

    def _preload_cache_rows(self) -> None:
        for row in self.rows[: min(self.preload_count, len(self.rows))]:
            if not row.cache_path:
                continue
            cache_path = self._resolve_cache_path(row.cache_path)
            if not cache_path.exists():
                continue
            array = np.array(self._load_cached_array(cache_path), dtype=np.float32, copy=True)
            self._remember_array(cache_path.as_posix(), array)

    def _load_image(self, row: DRRow) -> np.ndarray:
        if row.cache_path:
            cache_path = self._resolve_cache_path(row.cache_path)
            if cache_path.exists():
                cache_key = cache_path.as_posix()
                cached = self._memory_cache.get(cache_key)
                if cached is not None:
                    self._memory_cache.move_to_end(cache_key)
                    return cached
                cached = self._load_cached_array(cache_path)
                if cached.ndim != 3 or cached.shape[0] != 3:
                    raise ValueError(f"Expected cache shaped [3,H,W], got {tuple(cached.shape)} at {cache_path}")
                if self.memory_cache_size > 0:
                    cached = self._remember_array(cache_key, np.array(cached, dtype=np.float32, copy=True))
                return cached
        if self.require_cache:
            raise FileNotFoundError(
                f"Cache missing for image_id={row.image_id}. Expected prebuilt cache at {row.cache_path}."
            )
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Pillow is required when require_cache=False.") from exc
        image_path = Path(row.image_path)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
