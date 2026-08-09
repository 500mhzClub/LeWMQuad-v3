#!/usr/bin/env python3
"""Capped variance-only seed re-estimation for the factorial interaction.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The interim uses the **variance and nothing else**.  ``interim`` accepts the five
interaction values and returns the sample sd, its upper bound and the required
sample size; it deliberately does not compute, return or accept a mean, a sign, an
interval or any family breakdown, so an interim decision cannot be contaminated by
the observed effect.  ``final`` is a separate function and refuses to run until
the seed count is fixed.

Definitions, fixed in advance:

    s_I      sample standard deviation of the five per-seed interaction values
    sigma_U  = s_I * sqrt(4 / chi2_{0.10, 4})     90 % one-sided upper bound
    power    exact noncentral-t power for a one-sample t-test on I_s
    n*       the smallest n in [5, 10] reaching the target power at sigma_U
"""
from __future__ import annotations

import json
import math
from pathlib import Path

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

MINIMALLY_RELEVANT = 0.005      # delta, cosine units
ALPHA = 0.05                    # two-sided
TARGET_POWER = 0.80
INTERIM_N = 5
MIN_N = 5
MAX_N = 10
UPPER_BOUND_CONFIDENCE = 0.90   # one-sided


def _stats():
    try:
        from scipy import stats
    except ImportError as error:                       # pragma: no cover
        raise RuntimeError(
            "scipy is required: the sample size must come from the exact noncentral-t "
            "distribution, not an approximation") from error
    return stats


def sd_upper_bound(sample_sd: float, n: int = INTERIM_N,
                   confidence: float = UPPER_BOUND_CONFIDENCE) -> float:
    """One-sided upper confidence bound on sigma from n observations.

    sigma_U = s * sqrt((n - 1) / chi2_{1 - confidence, n - 1}); with n = 5 and
    confidence 0.90 this is exactly the prescribed s_I * sqrt(4 / chi2_{0.10,4}).
    """
    stats = _stats()
    df = n - 1
    quantile = stats.chi2.ppf(1.0 - confidence, df)
    return float(sample_sd * math.sqrt(df / quantile))


def power_at(n: int, sigma: float, delta: float = MINIMALLY_RELEVANT,
             alpha: float = ALPHA) -> float:
    """Exact two-sided noncentral-t power for a one-sample test on I_s."""
    stats = _stats()
    df = n - 1
    if df < 1 or sigma <= 0:
        return float("nan")
    ncp = math.sqrt(n) * delta / sigma
    critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    upper = stats.nct.sf(critical, df, ncp)
    lower = stats.nct.cdf(-critical, df, ncp)
    return float(upper + lower)


def required_n(sigma_upper: float, delta: float = MINIMALLY_RELEVANT,
               alpha: float = ALPHA, target: float = TARGET_POWER,
               minimum: int = MIN_N, maximum: int = MAX_N) -> dict:
    curve = {n: power_at(n, sigma_upper, delta, alpha) for n in range(minimum, maximum + 1)}
    meeting = [n for n in sorted(curve) if curve[n] >= target]
    if meeting:
        return {"n_final": meeting[0], "precision_limited": False, "power_curve": curve,
                "power_at_final": curve[meeting[0]]}
    return {"n_final": maximum, "precision_limited": True, "power_curve": curve,
            "power_at_final": curve[maximum]}


def interim(interaction_values, out: Path | None = None) -> dict:
    """Variance-only interim.  Returns NO mean, sign, interval or family result."""
    values = list(interaction_values)
    if len(values) != INTERIM_N:
        raise ValueError(f"the interim is defined on exactly {INTERIM_N} quadruplets; "
                         f"got {len(values)}")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite interaction value at the interim")
    mean = sum(values) / len(values)
    sample_sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    sigma_upper = sd_upper_bound(sample_sd, len(values))
    decision = required_n(sigma_upper)
    record = {
        "status": STATUS, "claim_bearing": False,
        "stage": "interim", "quadruplets_completed": len(values),
        "sample_sd_of_interaction": sample_sd,
        "sd_upper_bound_90pc_one_sided": sigma_upper,
        "minimally_relevant_interaction": MINIMALLY_RELEVANT,
        "alpha": ALPHA, "target_power": TARGET_POWER,
        **decision,
        "suppressed_at_interim": ["cell means", "interaction mean", "interaction sign",
                                  "confidence intervals", "family comparisons"],
        "decision_depends_only_on": "the variance of the interaction",
    }
    # The mean was computed only as an intermediate of the sd and is not reported.
    assert "interaction_mean" not in record
    if out:
        Path(out).write_text(json.dumps(record, indent=2))
    return record


def final(interaction_values, n_final: int, reestimation: dict) -> dict:
    """Final analysis: quadruplets are the replication units."""
    stats = _stats()
    values = list(interaction_values)
    n = len(values)
    if n != n_final:
        raise ValueError(f"final analysis expects the decided {n_final} quadruplets, got {n}")
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = stats.t.ppf(1.0 - ALPHA / 2.0, n - 1) * sd / math.sqrt(n)
    return {
        "status": STATUS, "claim_bearing": False, "stage": "final",
        "replication_unit": "training seed quadruplet",
        "individual_interactions": values,
        "mean_interaction": mean,
        "sample_sd": sd,
        "t_interval_95": [mean - half, mean + half],
        "n_final": n,
        "variance_reestimation_decision": reestimation,
        "episode_bootstrap_role": (
            "within-seed evaluation uncertainty only; it does not replace "
            "between-training-seed inference"),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(interim([float(v) for v in sys.argv[1:6]]), indent=2))
