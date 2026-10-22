"""Weighted evaluation metrics and paired bootstrap sampling."""
import hashlib
import json
import numpy as np

METRICS = ('qwk', 'auroc', 'false_referral_grade01', 'miss_grade234')

def array_sha(array):
    # Portable digest for ordered scalar identity fields, not native-endian bytes.
    return hashlib.sha256(json.dumps(np.asarray(array).tolist(), ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()

class WeightedMetrics:
    """Exactly the metrics on a repeated sample, evaluated by image multiplicity."""
    def __init__(self, grade, predicted, score, threshold):
        y = np.asarray(grade, dtype=np.float64)
        p = np.asarray(predicted, dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        threshold = -np.inf if threshold == '-inf' else float(threshold)
        positive = y >= 2
        referred = score >= threshold
        self.features = np.column_stack((np.ones(len(y)), y, y*y, p, p*p,
                                          (y-p)**2, ~positive,
                                          (~positive) & referred, positive & ~referred)).astype(np.float64)
        self.order = np.argsort(score, kind='stable')
        ordered = score[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.negative_sorted = (~positive)[self.order]

    def __call__(self, weights):
        weights = np.atleast_2d(weights).astype(np.float64, copy=False)
        s = weights @ self.features
        n, sy, sy2, sp, sp2, observed, negative, fp, fn = s.T
        positive = n - negative
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = sy2 + sp2 - 2*sy*sp/n
            qwk = 1 - observed / expected
            qwk[(n == 0) | (expected <= 0)] = np.nan
            ordered = weights[:, self.order]
            totals = np.add.reduceat(ordered, self.starts, axis=1)
            neg = np.add.reduceat(ordered * self.negative_sorted, self.starts, axis=1)
            pos = totals - neg
            auc = np.sum(pos * (np.cumsum(neg, axis=1) - 0.5*neg), axis=1) / (positive*negative)
            auc[(positive == 0) | (negative == 0)] = np.nan
            fr = fp / negative
            miss = fn / positive
        return np.column_stack((qwk, auc, fr, miss))

def bootstrap_weights(rng, groups, count):
    units = int(np.max(groups)) + 1
    draws = rng.integers(0, units, size=(count, units), dtype=np.int64)
    frequencies = np.array([np.bincount(row, minlength=units) for row in draws])
    return frequencies[:, groups].astype(np.float64)

def interval(point, samples, protocol):
    finite = np.isfinite(samples)
    valid = int(finite.sum())
    status, reason, low, high = 'PASS', '', None, None
    if not np.isfinite(point):
        status, reason = 'NE', 'original metric denominator undefined'
    elif valid < protocol['min_valid_replicates']:
        status, reason = 'NE', 'fewer than 9900 finite bootstrap replicates'
    else:
        low, high = np.quantile(samples[finite], [.025, .975], method='linear').tolist()
    return {'estimate': float(point) if np.isfinite(point) else None,
            'ci95_low': low, 'ci95_high': high, 'status': status, 'NE_reason': reason,
            'bootstrap_valid': valid, 'bootstrap_invalid': len(samples) - valid}
