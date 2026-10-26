"""C9 score-only Gaussian diagnostic; all ranking keys stay float64.

S1 retains the original per-bin 1e-7 floor and renormalization. S2 removes
only that floor. Both use log odds as a strictly increasing risk scale;
referral is score >= threshold, with -inf the all-referral candidate.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp, expit


def logdiff(a, b):
    """log(exp(a)-exp(b)), using expm1 near equality."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return a + np.log(-np.expm1(b-a))


def stable_scores(mu, sigma):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError('nonfinite sufficient statistics')
    z = (np.array([.5, 1.5, 2.5, 3.5])[None, :] - mu[:, None])/sigma[:, None]
    lc, ls = log_ndtr(z), log_ndtr(-z)
    middle = np.empty((len(mu), 3), np.float64)
    for j in range(3):
        # Use survival differences on the positive tail, CDF otherwise.
        positive = z[:, j] >= 0
        middle[positive, j] = logdiff(ls[positive, j], ls[positive, j+1])
        middle[~positive, j] = logdiff(lc[~positive, j+1], lc[~positive, j])
    logmass = np.column_stack([lc[:, 0], middle, ls[:, 3]])
    floored = np.maximum(logmass, np.log(1e-7))
    # Renormalization cancels in the odds, but remains in S1 probabilities.
    s1 = logsumexp(floored[:, 2:], axis=1)-logsumexp(floored[:, :2], axis=1)
    # Telescoping bins with unbounded endpoints gives exact Gaussian tail.
    s2 = ls[:, 1]-lc[:, 1]
    probs1 = np.exp(floored-logsumexp(floored, axis=1)[:, None])
    if not np.isfinite(s1).all() or not np.isfinite(s2).all():
        raise ValueError('nonfinite stable score')
    return s1, s2, probs1


def wilson_lower(k, n):
    z = 1.2815515655
    p = np.asarray(k, dtype=np.float64)/n
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def select_threshold(score, grade):
    s = np.asarray(score, np.float64)
    y = np.asarray(grade) >= 2
    if not np.isfinite(s).all():
        raise ValueError('score must be finite (only candidate may be -inf)')
    if y.sum() < 15 or (~y).sum() < 1:
        return dict(status='NOT_ESTIMABLE', threshold=None)
    candidates = np.r_[-np.inf, np.unique(s)]
    positives = np.sort(s[y])
    tp = len(positives)-np.searchsorted(positives, candidates, side='left')
    feasible = wilson_lower(tp, len(positives)) >= .90
    if not feasible.any():
        return dict(status='NOT_ESTIMABLE', threshold=None)
    tau = float(candidates[feasible][-1])
    return dict(status='PASS', threshold=tau, scale='float64 log odds', comparison='>=',
                risk_probability=float(expit(tau)), candidate_count=len(candidates),
                CAL_positive=int(y.sum()), CAL_negative=int((~y).sum()),
                CAL_TP=int(((s >= tau)&y).sum()),
                wilson_lower_90=float(wilson_lower(((s >= tau)&y).sum(), y.sum())))
