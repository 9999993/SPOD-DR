from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibmamba.utils.io import ensure_dir, read_csv_rows

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-size fundus RGB caches for diabetic-retinopathy phase-1 indices.")
    parser.add_argument("--index-files", nargs="+", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--normalization", choices=["imagenet", "zero_one"], default="imagenet")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--report-path", required=True)
    return parser.parse_args()


def _center_crop_square(image):
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def _preprocess_image(path: Path, input_size: int, normalization: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required to build the diabetic-retinopathy fundus cache."
        ) from exc
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        cropped = _center_crop_square(rgb)
        resized = cropped.resize((input_size, input_size), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    if normalization == "imagenet":
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
    return array


def cache_file_is_valid(cache_file: Path, *, input_size: int) -> bool:
    try:
        cached = np.load(cache_file, allow_pickle=False)
    except Exception:
        return False
    try:
        return (
            isinstance(cached, np.ndarray)
            and cached.dtype == np.float32
            and cached.shape == (3, int(input_size), int(input_size))
            and np.isfinite(cached).all()
        )
    finally:
        del cached


def _atomic_save(cache_file: Path, array: np.ndarray) -> None:
    ensure_dir(cache_file.parent)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{cache_file.stem}.",
        suffix=".tmp.npy",
        dir=cache_file.parent.as_posix(),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, cache_file)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _process_single_job(
    *,
    image_path: str,
    cache_path: str,
    input_size: int,
    normalization: str,
    overwrite: bool,
) -> tuple[str, str]:
    image_file = Path(image_path)
    cache_file = Path(cache_path)
    if not image_file.exists():
        return "missing", image_file.as_posix()
    if cache_file.exists() and not overwrite:
        if cache_file_is_valid(cache_file, input_size=input_size):
            return "skipped", cache_file.as_posix()
    array = _preprocess_image(image_file, input_size=input_size, normalization=normalization)
    _atomic_save(cache_file, array)
    return "written", cache_file.as_posix()


def build_outputs(
    *,
    index_files: list[Path],
    input_size: int,
    normalization: str,
    overwrite: bool,
    num_workers: int,
    progress_every: int,
    report_path: Path,
) -> dict[str, object]:
    rows_total = 0
    written = 0
    skipped = 0
    missing_sources: list[str] = []
    cache_targets: list[str] = []
    seen_jobs: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []

    for index_file in index_files:
        for row in read_csv_rows(index_file):
            rows_total += 1
            image_path = Path(str(row["image_path"]))
            cache_path = Path(str(row["cache_path"]))
            cache_path_str = cache_path.as_posix()
            image_path_str = image_path.as_posix()
            previous = seen_jobs.get(cache_path_str)
            if previous is not None:
                if previous != image_path_str:
                    raise ValueError(
                        "Conflicting cache targets detected for the same cache_path: "
                        f"{cache_path_str}\nprevious_image={previous}\ncurrent_image={image_path_str}"
                    )
                continue
            seen_jobs[cache_path_str] = image_path_str
            cache_targets.append(cache_path_str)
            jobs.append((image_path_str, cache_path_str))

    total_jobs = len(jobs)
    progress_every = max(int(progress_every), 1)
    num_workers = max(int(num_workers), 1)

    if num_workers == 1:
        for index, (image_path, cache_path) in enumerate(jobs, start=1):
            status, payload = _process_single_job(
                image_path=image_path,
                cache_path=cache_path,
                input_size=input_size,
                normalization=normalization,
                overwrite=overwrite,
            )
            if status == "missing":
                missing_sources.append(payload)
            elif status == "skipped":
                skipped += 1
            else:
                written += 1
            if index % progress_every == 0 or index == total_jobs:
                print(
                    f"progress {index}/{total_jobs} "
                    f"written={written} skipped={skipped} missing={len(missing_sources)}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_single_job,
                    image_path=image_path,
                    cache_path=cache_path,
                    input_size=input_size,
                    normalization=normalization,
                    overwrite=overwrite,
                )
                for image_path, cache_path in jobs
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                status, payload = future.result()
                if status == "missing":
                    missing_sources.append(payload)
                elif status == "skipped":
                    skipped += 1
                else:
                    written += 1
                if index % progress_every == 0 or index == total_jobs:
                    print(
                        f"progress {index}/{total_jobs} "
                        f"written={written} skipped={skipped} missing={len(missing_sources)}",
                        flush=True,
                    )

    report = {
        "decision": "DIABETIC_RETINOPATHY_FUNDUS_CACHE_BUILD_READY",
        "input_size": int(input_size),
        "normalization": normalization,
        "rows_total": rows_total,
        "unique_cache_targets": len(cache_targets),
        "num_workers": num_workers,
        "cache_written": written,
        "cache_skipped_existing": skipped,
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources[:50],
        "report_path": report_path.as_posix(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_report = report_path.with_suffix(".csv")
    with csv_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cache_path"])
        writer.writeheader()
        for cache_path in cache_targets:
            writer.writerow({"cache_path": cache_path})

    if missing_sources:
        raise FileNotFoundError(
            f"Missing {len(missing_sources)} source images during cache build. "
            f"First missing path: {missing_sources[0]}"
        )
    return report


def main() -> None:
    args = parse_args()
    payload = build_outputs(
        index_files=[Path(path) for path in args.index_files],
        input_size=int(args.input_size),
        normalization=str(args.normalization),
        overwrite=bool(args.overwrite),
        num_workers=int(args.num_workers),
        progress_every=int(args.progress_every),
        report_path=Path(args.report_path),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
