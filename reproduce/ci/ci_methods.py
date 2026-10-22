"""Confidence intervals for absolute metrics and paired differences."""
import math


BOOTSTRAP_METHOD = "bootstrap_percentile_two_sided_95"
EXACT_METHOD = "clopper_pearson_exact_binomial_two_sided_95"


def zero_event_clopper_pearson(trials, confidence=0.95):
    """Two-sided CP for x=0: (1-p_upper)**n = alpha/2; no RNG."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("A positive integer number of binomial trials is required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    upper = -math.expm1(math.log((1 - confidence) / 2) / trials)
    return 0.0, upper


def absolute_interval(point, samples, protocol, *, cohort, metric, events=None,
                      trials=None, interval_function):
    """Only zero-event IDRiD absolute FNR gets the exact binomial interval."""
    exact = cohort == "IDRiD" and metric == "miss_grade234" and events == 0
    if exact:
        low, high = zero_event_clopper_pearson(trials)
        if point != 0:
            raise ValueError("Zero-event count conflicts with the original point estimate")
        result = {"estimate": 0.0, "ci95_low": low, "ci95_high": high,
                  "status": "PASS", "NE_reason": "", "bootstrap_valid": None,
                  "bootstrap_invalid": None}
    else:
        result = interval_function(point, samples, protocol)
    return {**result, "ci_method": EXACT_METHOD if exact else BOOTSTRAP_METHOD,
            "confidence_level": 0.95, "ci_scope": "absolute",
            "event_numerator": events, "event_denominator": trials,
            "interval_bootstrap_replicates": 0 if exact else len(samples)}


def paired_interval(point, samples, protocol, *, interval_function):
    """No exact-binomial exception is defined for paired differences."""
    return {**interval_function(point, samples, protocol), "ci_method": BOOTSTRAP_METHOD,
            "confidence_level": 0.95, "ci_scope": "paired_difference",
            "interval_bootstrap_replicates": len(samples)}
