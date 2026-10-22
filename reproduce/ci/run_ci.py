"""Evaluate saved predictions with fixed thresholds and paired confidence intervals."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_referral_flag(value):
    """Accept the two explicit encodings present in archived prediction CSVs."""
    if value not in ("0", "1", "False", "True"):
        raise ValueError(f"Unsupported saved pred_referral value: {value!r}")
    return int(value in ("1", "True"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", type=Path, required=True,
                        help="Root containing model-family CSVs and order2101/... NPZs")
    parser.add_argument("--policies-root", type=Path, required=True,
                        help="Current release root containing checkpoints/.../POLICY.json")
    parser.add_argument("--profile", choices=["main", "ablation", "all"], default="main")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, help="New, nonexistent output directory; required for --run")
    args = parser.parse_args()
    if args.run and (args.output is None or args.output.exists()):
        parser.error("--run requires a new nonexistent --output directory")

    import numpy as np
    from threadpoolctl import threadpool_limits
    from ci_methods import absolute_interval, paired_interval
    from order_inputs import load_order_prediction, policy_selection
    original = HERE / "metric_core.py"
    if sha(original) != "c52561599127aa6a6d447766d32d02cd836f411f02f924d00bd9b5bd04d80d35":
        raise ValueError("Metric implementation hash mismatch")
    spec = importlib.util.spec_from_file_location("metric_core", original)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    registry = json.loads((HERE / "INPUT_REGISTRY.json").read_text(encoding="utf-8"))
    frozen = json.loads((HERE / "groups.json").read_text(encoding="utf-8"))["groups"]
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["base_seed"] == 20261016 and protocol["replicates"] == 10000
    assert protocol["min_valid_replicates"] == 9900
    assert registry["cohorts"] == ["DDR", "MESSIDOR-2", "IDRiD"]
    comparisons = [c for c in registry["comparisons"]
                   if args.profile == "all" or c["profile"] == args.profile]
    selected = {c[k] for c in comparisons for k in ["candidate", "reference"]}
    models = [m for m in registry["models"] if m["artifact_id"] in selected]
    ids = [m["artifact_id"] for m in models]
    loaded, input_hashes = {}, []
    for model in models:
        policy_root = HERE if model.get("policy_base") == "supplement" else args.policies_root
        path = policy_root / model["policy_file"]
        actual = sha(path)
        if actual not in {model["policy_sha256"], model["original_policy_sha256"]}:
            raise ValueError(f"Unregistered POLICY: {path}")
        policy, selection = policy_selection(model, json.loads(path.read_text(encoding="utf-8")))
        threshold = selection["threshold"]
        if float(threshold) != model["threshold"]:
            raise ValueError(f"Fixed threshold differs: {path}")
        if "global_step" in policy and policy["global_step"] != model["selected_global_step"]:
            raise ValueError(f"Selected step differs: {path}")
        # Publication metadata cleaning can change a policy's byte hash; only the
        # two registered original/current hashes are allowed, never arbitrary JSON.
        for key in ["model_sha256", "selected_sha256", "weight_sha256"]:
            if key in policy and policy[key] != model["weight_sha256"]:
                raise ValueError(f"Policy weight identity differs: {path}")
        input_hashes.append({"path": str(path), "sha256": actual})
        for cohort, entry in model["predictions"].items():
            group = np.asarray(frozen[cohort], dtype=np.int64)
            binding = registry["cohort_bindings"][cohort]
            if entry["format"] == "order_npz":
                image_ids, grade, pred, score = load_order_prediction(
                    entry, args.inputs_root, binding, engine.array_sha, input_hashes)
            else:
                path = args.inputs_root / entry["path"]
                if sha(path) != entry["sha256"]:
                    raise ValueError(f"Prediction SHA256 differs: {path}")
                rows = read_csv(path)
                image_ids = [r["image_id"] for r in rows]
                grade = np.asarray([int(r["y_true"]) for r in rows])
                pred = np.asarray([int(r["pred_grade"]) for r in rows])
                skey = "referral_log_odds"
                tkey = "cal_log_odds_threshold"
                score = np.asarray([float(r[skey]) for r in rows], dtype=np.float64)
                assert all(float(r[tkey]) == model["threshold"] for r in rows)
                if "pred_referral" in rows[0]:
                    assert np.array_equal(score >= model["threshold"],
                                      [parse_referral_flag(r["pred_referral"]) for r in rows])
                input_hashes.append({"path": str(path), "sha256": entry["sha256"]})
            assert len(image_ids) == int(binding["images"]) == len(group)
            assert len(set(image_ids)) == len(image_ids)
            assert engine.array_sha(image_ids) == binding["ordered_original_id_sha256"]
            assert engine.array_sha(grade) == binding["grade_sha256"]
            assert engine.array_sha(group) == binding["group_vector_sha256"]
            assert set(np.unique(group)) == set(range(int(group.max()) + 1))
            assert np.isfinite(score).all() and np.isin(pred, range(5)).all()
            if cohort == "MESSIDOR-2":
                assert np.unique(group).size == 874
                sizes = np.bincount(group)
                assert (sizes == 2).sum() == 870 and (sizes == 1).sum() == 4
            loaded[model["artifact_id"], cohort] = (grade, pred, score)
    if args.check_inputs:
        print(f"INPUTS_OK: {len(models)} models; no metrics, draws or output written")
        return

    args.output.mkdir(parents=True, exist_ok=False)
    absolute, paired, sampling = [], [], []
    metrics = engine.METRICS
    with threadpool_limits(limits=1):
        for ci, cohort in enumerate(registry["cohorts"]):
            groups = np.asarray(frozen[cohort], dtype=np.int64)
            calculators = [engine.WeightedMetrics(*loaded[m["artifact_id"], cohort], m["threshold"])
                           for m in models]
            points = np.stack([calc(np.ones(len(groups)))[0] for calc in calculators])
            for mi, model in enumerate(models):
                if "expected_point_estimates" in model:
                    expected = [model["expected_point_estimates"][cohort][metric] for metric in metrics]
                    np.testing.assert_allclose(points[mi], expected, rtol=0, atol=1e-12)
            for comparison in comparisons:
                if "expected_point_estimates" in comparison:
                    ai, bi = ids.index(comparison["candidate"]), ids.index(comparison["reference"])
                    expected = [comparison["expected_point_estimates"][cohort][metric] for metric in metrics]
                    np.testing.assert_allclose(points[ai] - points[bi], expected, rtol=0, atol=1e-12)
            samples = np.empty((10000, len(models), 4), dtype=np.float64)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261016, ci])))
            digest = hashlib.sha256()
            for first in range(0, 10000, 64):
                weights = engine.bootstrap_weights(rng, groups, min(64, 10000 - first))
                digest.update(weights.astype("<u4").tobytes())
                for mi, calc in enumerate(calculators):
                    samples[first:first + len(weights), mi] = calc(weights)
            assert digest.hexdigest() == registry["cohort_bindings"][cohort]["sampling_sha256"]
            np.savez_compressed(args.output / (cohort + ".npz"), values=samples, points=points,
                                model_ids=np.asarray(ids), metric_names=np.asarray(metrics))
            for mi, ident in enumerate(ids):
                grade, _, score = loaded[ident, cohort]
                referred = score >= models[mi]["threshold"]
                for ki, metric in enumerate(metrics):
                    events, trials = None, None
                    if metric == "miss_grade234":
                        events = int(((grade >= 2) & ~referred).sum())
                        trials = int((grade >= 2).sum())
                    elif metric == "false_referral_grade01":
                        events = int(((grade < 2) & referred).sum())
                        trials = int((grade < 2).sum())
                    absolute.append({"artifact_id": ident, "cohort": cohort, "metric": metric,
                                     **absolute_interval(points[mi, ki], samples[:, mi, ki], protocol,
                                         cohort=cohort, metric=metric, events=events, trials=trials,
                                         interval_function=engine.interval)})
            for comparison in comparisons:
                ai, bi = ids.index(comparison["candidate"]), ids.index(comparison["reference"])
                for ki, metric in enumerate(metrics):
                    result = paired_interval(points[ai, ki] - points[bi, ki],
                                             samples[:, ai, ki] - samples[:, bi, ki], protocol,
                                             interval_function=engine.interval)
                    identity = {k: v for k, v in comparison.items() if not k.startswith("expected_point")}
                    paired.append({**identity, "cohort": cohort, "metric": metric, **result})
            sampling.append({"cohort": cohort, "images": len(groups), "units": int(groups.max()) + 1,
                             "sampling_sha256": digest.hexdigest(), "seed_sequence": [20261016, ci]})
    for item in input_hashes:
        assert sha(Path(item["path"])) == item["sha256"]
    public_inputs = []
    for item in input_hashes:
        resolved = Path(item["path"]).resolve()
        for label, root in [("predictions", args.inputs_root), ("repository", args.policies_root), ("ci", HERE)]:
            try:
                relative = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            public_inputs.append({"file": label + "/" + relative, "sha256": item["sha256"]})
            break
        else:
            raise ValueError("Input is outside the configured input directories")
    write_csv(args.output / "ABSOLUTE_CI.csv", absolute)
    write_csv(args.output / "PAIRED_CI.csv", paired)
    (args.output / "RUN.json").write_text(json.dumps({"inputs": public_inputs, "sampling": sampling,
        "replicates": 10000, "base_seed": 20261016, "min_valid_replicates": 9900,
        "absolute_interval_rule": registry["absolute_interval_rule"],
        "numpy": np.__version__, "adapter_sha256": sha(Path(__file__)),
        "interval_dispatch_sha256": sha(HERE / "ci_methods.py"),
        "order_loader_sha256": sha(HERE / "order_inputs.py"),
        "metric_implementation_sha256": sha(original),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "group_vector_file_sha256": sha(HERE / "groups.json"),
        "registry_sha256": sha(HERE / "INPUT_REGISTRY.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
