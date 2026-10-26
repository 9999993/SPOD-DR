#!/usr/bin/env python3
"""Build the registered Stable-B0-compatible 512 RGB cache for DeepDRiD."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def job(item: tuple[str, str]) -> tuple[str, str, str, int, int]:
    source, target = map(Path, item)
    from PIL import Image
    with Image.open(source) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        cropped = rgb.crop((left, top, left + side, top + side))
        resized = cropped.resize((512, 512), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose((arr - MEAN) / STD, (2, 0, 1)).astype(np.float32, copy=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.stem + ".", suffix=".npy", dir=target.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        np.save(tmp, arr)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return source.as_posix(), target.as_posix(), sha256(target), w, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-manifest", required=True, type=Path)
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    args = ap.parse_args()
    rows = list(csv.DictReader(args.image_manifest.open(encoding="utf-8", newline="")))
    jobs: list[tuple[str, str]] = []
    for r in rows:
        source = Path(r["image_path"])
        target = args.cache_root / r["split"] / r["patient_id"] / (r["image_id"] + "_rgb512.npy")
        jobs.append((source.as_posix(), target.as_posix()))
        r["cache_path"] = target.as_posix()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(job, x) for x in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 200 == 0 or i == len(futures):
                print(f"cached {i}/{len(futures)}", flush=True)
    by_source = {x[0]: x for x in results}
    for r in rows:
        source, target, digest, w, h = by_source[r["image_path"]]
        r.update({"cache_sha256": digest, "source_width": w, "source_height": h,
                  "preprocess": "RGB;center-crop-square;BICUBIC;512;ImageNet-normalized;CHW-float32"})
    manifest = args.report.with_name("DR_DEEPDRID_512_CACHE_MANIFEST.csv")
    fields = list(rows[0].keys())
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    result = {
        "decision": "DEEPDRID_STAGE_B_512_CACHE_READY",
        "rows": len(rows), "cache_files": len(results), "missing": 0,
        "input_size": 512, "normalization": "ImageNet mean/std",
        "preprocess": "center crop to square, bicubic resize, CHW float32",
        "source_manifest_sha256": sha256(args.image_manifest),
        "cache_manifest": manifest.as_posix(),
        "cache_manifest_sha256": sha256(manifest),
        "cache_root": args.cache_root.as_posix(),
    }
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
