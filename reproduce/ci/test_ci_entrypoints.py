"""Small artificial examples only: no real-cohort metrics or formal bootstrap."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import beta

import ci_methods
from ci_methods import absolute_interval, paired_interval, zero_event_clopper_pearson
from order_inputs import policy_selection, project_order, validate_npz_identity

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("archived_test_ci", HERE / "metric_core.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
TOY_PROTOCOL = {"min_valid_replicates": 4}  # Four artificial values, never a production run.


class IntervalExamples(unittest.TestCase):
    def test_zero_of_64_two_sided_exact_and_paper_rounding(self):
        low, high = zero_event_clopper_pearson(64)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, beta.ppf(.975, 1, 64), places=14)
        self.assertAlmostEqual((1 - high) ** 64, .025, places=14)
        self.assertEqual((f"{100*low:.2f}", f"{100*high:.2f}"), ("0.00", "5.60"))

    def test_both_named_zero_event_rows_select_exact_without_bootstrap(self):
        for artifact in ["main/B0", "native/ConvNeXtTiny"]:
            with self.subTest(artifact=artifact):
                result = absolute_interval(0., np.zeros(4), TOY_PROTOCOL, cohort="IDRiD",
                    metric="miss_grade234", events=0, trials=64,
                    interval_function=lambda *a: self.fail("Exact absolute row must not call bootstrap interval"))
                self.assertEqual(result["ci_method"], ci_methods.EXACT_METHOD)
                self.assertEqual(result["event_numerator"], 0)
                self.assertEqual(result["event_denominator"], 64)
                self.assertEqual(result["interval_bootstrap_replicates"], 0)
                self.assertIsNone(result["bootstrap_valid"])

    def test_other_absolute_cases_keep_original_interval(self):
        for cohort, metric, events in [("DDR", "miss_grade234", 0),
                                      ("IDRiD", "miss_grade234", 1),
                                      ("IDRiD", "false_referral_grade01", 0),
                                      ("IDRiD", "auroc", None), ("IDRiD", "qwk", None)]:
            samples = np.asarray([0., .01, .02, .03])
            old = engine.interval(.01, samples, TOY_PROTOCOL)
            with self.subTest(cohort=cohort, metric=metric, events=events):
                result = absolute_interval(.01, samples, TOY_PROTOCOL, cohort=cohort,
                    metric=metric, events=events, trials=64, interval_function=engine.interval)
                self.assertEqual(result["ci_method"], ci_methods.BOOTSTRAP_METHOD)
                self.assertEqual({k: result[k] for k in old}, old)

    def test_paired_zero_difference_never_uses_exact_binomial(self):
        with patch.object(ci_methods, "zero_event_clopper_pearson", side_effect=AssertionError("not paired")):
            result = paired_interval(0., np.zeros(4), TOY_PROTOCOL, interval_function=engine.interval)
        self.assertEqual((result["ci95_low"], result["ci95_high"]), (0., 0.))
        self.assertEqual(result["ci_scope"], "paired_difference")
        self.assertEqual(result["ci_method"], ci_methods.BOOTSTRAP_METHOD)

    def test_invalid_trial_count_and_conflicting_point_rejected(self):
        for n in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                zero_event_clopper_pearson(n)
        with self.assertRaises(ValueError):
            absolute_interval(.1, np.zeros(4), TOY_PROTOCOL, cohort="IDRiD",
                metric="miss_grade234", events=0, trials=64, interval_function=engine.interval)


class OrderExamples(unittest.TestCase):
    def test_equal_probability_fusion_argmax_and_log_odds(self):
        a = np.asarray([[.6, .1, .2, .05, .05], [.1, .1, .1, .6, .1]])
        b = np.asarray([[.2, .1, .4, .2, .1], [.1, .1, .1, .5, .2]])
        pred, score = project_order({"projection": "ORDER_components", "weights": [.5, .5]},
                                    [{"probabilities": a}, {"probabilities": b}])
        np.testing.assert_array_equal(pred, [0, 3])
        np.testing.assert_allclose(score, [0., np.log(4)], rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            project_order({"projection": "ORDER_components", "weights": [.4, .6]},
                          [{"probabilities": a}, {"probabilities": b}])

    def test_saved_standard_score_and_grade_are_retained(self):
        z = {"probabilities": np.asarray([[.6, .1, .2, .05, .05]]),
             "pred": np.asarray([1]), "score": np.asarray([42.5])}
        pred, score = project_order({"projection": "standard"}, [z])
        np.testing.assert_array_equal(pred, [1])
        np.testing.assert_array_equal(score, [42.5])

    def test_order_uses_unified_cal_not_author_dev(self):
        model = {"policy_entry": "ORDER_CODE__2101", "policy_selection": "UNIFIED_CAL",
                 "member_ids": ["CB__2101", "LORS_CODE__2101"]}
        policy = {"ORDER_CODE__2101": {"fusion": [.5, .5], "members": model["member_ids"],
                  "AUTHOR_DEV": {"threshold": 999}, "UNIFIED_CAL": {"threshold": -3.92291527808383}}}
        _, selected = policy_selection(model, policy)
        self.assertEqual(selected["threshold"], -3.92291527808383)

    def test_order_identity_and_order_are_frozen(self):
        z = {"image_key": np.asarray(["source:a", "source:b"]), "grade": np.asarray([0, 3]),
             "row_index": np.asarray([0, 1]), "component_hash": np.asarray(["u", "v"])}
        binding = {"images": 2, "ordered_original_id_sha256": engine.array_sha(["a", "b"]),
                   "grade_sha256": engine.array_sha(z["grade"]),
                   "row_index_sha256": engine.array_sha(z["row_index"]),
                   "component_sha256": engine.array_sha(z["component_hash"])}
        ids, _ = validate_npz_identity(z, binding, engine.array_sha)
        self.assertEqual(ids, ["a", "b"])
        z["image_key"] = z["image_key"][::-1]
        with self.assertRaises(ValueError):
            validate_npz_identity(z, binding, engine.array_sha)

    def test_registered_paper_pair_is_b3_seed2101_not_old_b0_seed2102(self):
        reg = json.loads((HERE / "INPUT_REGISTRY.json").read_text(encoding="utf-8"))
        order = next(m for m in reg["models"] if m["id"].startswith("ORDER"))
        self.assertEqual((order["id"], order["seed"]), ("ORDER_CODE__2101", 2101))
        pairs = [c for c in reg["comparisons"] if c["reference"] == order["artifact_id"]]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["candidate"], "main/B3")
        ids = {c[k] for c in reg["comparisons"] if c["profile"] == "main"
               for k in ["candidate", "reference"]}
        self.assertEqual(len(ids), 11)
        self.assertEqual(sum(c["profile"] == "main" for c in reg["comparisons"]), 6)



class ReferralCsvCompatibility(unittest.TestCase):
    def test_original_boolean_and_numeric_csv_encodings(self):
        import run_ci
        for token, expected in [("False", 0), ("True", 1), ("0", 0), ("1", 1)]:
            with self.subTest(token=token):
                self.assertEqual(run_ci.parse_referral_flag(token), expected)

    def test_unknown_values_are_not_silently_coerced(self):
        import run_ci
        for token in ["", "2", "-1", "0.0", "yes", "nan", None]:
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    run_ci.parse_referral_flag(token)


if __name__ == "__main__":
    unittest.main()
