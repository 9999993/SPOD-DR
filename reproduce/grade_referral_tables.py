"""Reproduce Tables S15a and S15b from the released, saved predictions.

Uses the Python standard library. No inference or threshold fitting is performed.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


BACKBONES = ("B0", "B3", "ConvNeXtTiny", "SwinTiny", "ResNet50")
COHORTS = ("DDR", "IDRiD", "MESSIDOR-2")
LABELS = {"B0": "EfficientNet-B0", "B3": "EfficientNet-B3",
          "ConvNeXtTiny": "ConvNeXt-Tiny", "SwinTiny": "Swin-Tiny",
          "ResNet50": "ResNet-50"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tabulate(inputs_root, registry_path, policies_root):
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    models = {m["artifact_id"]: m for m in registry["models"]}
    table_a, sources = [], []
    table_b = [[0, 0] for _ in range(5)]
    for cohort in COHORTS:
        for backbone in BACKBONES:
            model = models[f"main/{backbone}"]
            policy = (policies_root / model["policy_file"]).read_bytes()
            if sha256(policy) != model["policy_sha256"]:
                raise ValueError(f"POLICY hash mismatch: {model['artifact_id']}")
            entry = model["predictions"][cohort]
            path = inputs_root / entry["path"]
            data = path.read_bytes()
            if sha256(data) != entry["sha256"]:
                raise ValueError(f"Prediction hash mismatch: {path}")
            counts = [[0, 0] for _ in range(5)]
            seen = set()
            threshold = float(model["threshold"])
            for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
                image_id = row["image_id"]
                if image_id in seen:
                    raise ValueError(f"Duplicate image ID in {path}: {image_id}")
                seen.add(image_id)
                grade, reference = int(row["pred_grade"]), int(row["y_true"])
                if grade not in range(5) or reference not in range(5):
                    raise ValueError(f"Invalid grade in {path}: {image_id}")
                score = float(row["referral_log_odds"])
                if not math.isfinite(score) or float(row["cal_log_odds_threshold"]) != threshold:
                    raise ValueError(f"Invalid score or threshold in {path}: {image_id}")
                referred = int(score >= threshold)
                flag = row["pred_referral"].strip().lower()
                if flag not in {"0", "1", "false", "true"}:
                    raise ValueError(f"Invalid saved referral flag in {path}: {image_id}")
                if referred != int(flag in {"1", "true"}):
                    raise ValueError(f"Saved referral flag mismatch in {path}: {image_id}")
                counts[grade][referred] += 1
                if cohort == "MESSIDOR-2" and backbone == "B3" and referred and grade < 2:
                    table_b[reference][grade] += 1
            if not seen:
                raise ValueError(f"Empty prediction file: {path}")
            for grade, (negative, positive) in enumerate(counts):
                table_a.append([cohort, LABELS[backbone], grade, negative, positive, negative + positive])
            sources.append({"model": model["artifact_id"], "dataset": cohort,
                            "path": entry["path"], "sha256": entry["sha256"],
                            "images": len(seen), "threshold": threshold,
                            "policy_sha256": model["policy_sha256"]})
    rows_b = [[grade, *counts, sum(counts)] for grade, counts in enumerate(table_b)]
    return table_a, rows_b, {"registry_sha256": sha256(registry_bytes), "inputs": sources}


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, default=root / "local-data/ci")
    parser.add_argument("--registry", type=Path, default=root / "reproduce/ci/INPUT_REGISTRY.json")
    parser.add_argument("--policies-root", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b, provenance = tabulate(args.inputs_root, args.registry, args.policies_root)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "S15a.csv", ["Dataset", "Backbone", "Predicted grade",
              "Referral-negative images", "Referral-positive images", "Total images"], a)
    write_csv(args.output / "S15b.csv", ["Reference grade", "Predicted grade 0",
              "Predicted grade 1", "Total images"], b)
    provenance["outputs"] = {name: sha256((args.output / name).read_bytes())
                             for name in ("S15a.csv", "S15b.csv")}
    (args.output / "S15_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote S15a (75 rows) and S15b (5 reference-grade rows) to {args.output}")


if __name__ == "__main__":
    main()
