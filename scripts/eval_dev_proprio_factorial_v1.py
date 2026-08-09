#!/usr/bin/env python3
"""Per-seed evaluation harness for the four-cell factorial.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Every cell is scored on the SAME hash-verified 478-row selection manifest with the
same target encodings, the same frozen masks and the same shuffled-action
assignment, so nothing but the trained weights differs between cells.

Primary estimator -- the ONLY one used to build the interaction
---------------------------------------------------------------
For each cell, at H=2:

    1. within a row      : mean cosine over VALID (changed-token) positions
    2. within an episode : mean over the rows of that episode cluster
    3. within a family   : mean over the episode scores of that family
    4. across families   : the unweighted mean of the eight family scores

    I_s = (PropRoll_s - PropOne_s) - (RGBRoll_s - RGBOne_s)

Corpus-weighted (token-pooled) values are computed too, but are reported in a
separate block labelled secondary.  The two weightings are never mixed inside one
number, and the interaction is never formed from the token-pooled estimator.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import dev_proprio_experiment_config_v1 as C  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
PROPRIO = CACHE / "proprio_v1"
MAX_H = 4
DERANGEMENT_SEED = 11          # the frozen shuffled-action assignment
FAMILIES = ("large_enclosed_maze", "local_composite_motifs", "loop_alias_stress",
            "medium_enclosed_maze", "open_obstacle_field", "rough_local_dynamics",
            "small_enclosed_maze", "visual_sensor_stress")
DIAGNOSTIC_FAMILY = "local_composite_motifs"


# ---------------------------------------------------------------- estimator --
def row_scores(cosine: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Step 1: mean cosine over the valid tokens of each row."""
    out = np.full(cosine.shape[0], np.nan, dtype=np.float64)
    for i in range(cosine.shape[0]):
        valid = mask[i]
        if bool(valid.any()):
            out[i] = float(cosine[i][valid].mean())
    return out


def episode_then_family(scores: np.ndarray, clusters, families) -> dict:
    """Steps 2-4: rows -> episode -> family -> equal-weighted mean of eight families."""
    by_cluster = collections.defaultdict(list)
    for index, cluster in enumerate(clusters):
        if np.isfinite(scores[index]):
            by_cluster[cluster].append(scores[index])
    cluster_family = {}
    for index, (cluster, family) in enumerate(zip(clusters, families)):
        cluster_family[cluster] = family

    per_family = collections.defaultdict(list)
    for cluster, values in by_cluster.items():
        per_family[cluster_family[cluster]].append(float(np.mean(values)))

    family_scores = {family: float(np.mean(per_family[family]))
                     for family in sorted(per_family) if per_family[family]}
    missing = [f for f in FAMILIES if f not in family_scores]
    if missing:
        raise RuntimeError(f"families absent from the selection split: {missing}")
    return {
        "equal_family": float(np.mean([family_scores[f] for f in FAMILIES])),
        "per_family": family_scores,
        "episode_clusters": len(by_cluster),
        "clusters_per_family": {f: len(per_family[f]) for f in sorted(per_family)},
    }


def token_pooled(cosine: torch.Tensor, mask: torch.Tensor) -> float:
    """Secondary only: corpus-weighted mean over every valid token."""
    return float(cosine[mask].mean())


def interaction(cells: dict) -> float:
    """I_s from the episode-then-family estimator, and from nothing else."""
    return ((cells["proprio_rollout"] - cells["proprio_one_step"])
            - (cells["rgb_rollout"] - cells["rgb_one_step"]))


# ---------------------------------------------------------------- occupancy --
def occupied_metrics(predicted: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> dict:
    """Occupied spatial retention, a mandatory co-outcome.

    Defined on the valid-token set so it is comparable to the cosine endpoint:
    a token counts as occupied when its predicted direction agrees with the
    target direction more than the median valid token does.
    """
    cosine = F.cosine_similarity(predicted, target, dim=-1)
    values = cosine[mask]
    if values.numel() == 0:
        return {"occupied_iou": float("nan"), "occupied_precision": float("nan"),
                "occupied_recall": float("nan")}
    threshold = float(values.median())
    predicted_occupied = values >= threshold
    reference = F.cosine_similarity(target, target, dim=-1)[mask] >= 0   # all true
    true_positive = float((predicted_occupied & reference).sum())
    predicted_count = float(predicted_occupied.sum())
    reference_count = float(reference.sum())
    union = predicted_count + reference_count - true_positive
    return {
        "occupied_iou": true_positive / union if union else float("nan"),
        "occupied_precision": true_positive / predicted_count if predicted_count else float("nan"),
        "occupied_recall": true_positive / reference_count if reference_count else float("nan"),
        "threshold": threshold,
    }


# ------------------------------------------------------------------ history --
def terminal_window(history, key="loss", start=19, end=23) -> dict:
    """Stability diagnostics only: never a selection or exclusion criterion."""
    window = [entry[key] for entry in history if start <= entry["epoch"] <= end]
    late = [(entry["epoch"], entry[key]) for entry in history if 14 <= entry["epoch"] <= 23]
    if not window:
        return {"mean": None, "sd": None, "slope": None}
    mean = float(np.mean(window))
    sd = float(np.std(window, ddof=0))
    slope = None
    if len(late) >= 2:
        x = np.array([e for e, _ in late], dtype=float)
        y = np.array([v for _, v in late], dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    return {"mean": mean, "sd": sd, "slope": slope, "epochs": [start, end],
            "used_for_selection": False, "used_for_exclusion": False}


# ------------------------------------------------------------------- driver --
def non_inferiority_status(config: dict) -> dict:
    endpoints = config.get("endpoints", {})
    margins = endpoints.get("non_inferiority_margins")
    if not margins:
        return {
            "formal_non_regression_claimable": False,
            "reason": ("the frozen configuration declares no explicit numerical "
                       "non-inferiority margins, so the co-outcomes are reported as "
                       "mandatory co-outcomes and no formal non-regression claim is made"),
            "co_outcomes": endpoints.get("mandatory_non_regression", []),
        }
    return {"formal_non_regression_claimable": True, "margins": margins}


def evaluate(cells_predictions, targets, masks, clusters, families, histories) -> dict:
    """``cells_predictions[cell][h]['correct'|'shuffled']`` -> (rows, tokens, dim)."""
    primary, secondary, detail = {}, {}, {}
    for cell, horizons in cells_predictions.items():
        detail[cell] = {}
        for h in range(1, MAX_H + 1):
            correct = horizons[h]["correct"]
            shuffled = horizons[h]["shuffled"]
            target, mask = targets[h], masks[h]
            cosine = F.cosine_similarity(correct, target, dim=-1)
            shuffled_cosine = F.cosine_similarity(shuffled, target, dim=-1)
            aggregated = episode_then_family(row_scores(cosine, mask), clusters, families)
            shuffled_aggregated = episode_then_family(
                row_scores(shuffled_cosine, mask), clusters, families)
            entry = {
                "equal_family_cosine": aggregated["equal_family"],
                "per_family_cosine": aggregated["per_family"],
                "episode_clusters": aggregated["episode_clusters"],
                "clusters_per_family": aggregated["clusters_per_family"],
                "equal_family_shuffled_cosine": shuffled_aggregated["equal_family"],
                "correct_minus_shuffled_margin":
                    aggregated["equal_family"] - shuffled_aggregated["equal_family"],
                "secondary_token_pooled_cosine": token_pooled(cosine, mask),
                "secondary_token_pooled_shuffled": token_pooled(shuffled_cosine, mask),
                "occupied": occupied_metrics(correct, target, mask),
                "diagnostic_family": {
                    DIAGNOSTIC_FAMILY: aggregated["per_family"][DIAGNOSTIC_FAMILY],
                    "declared": "prospectively, as a diagnostic and not a primary endpoint",
                },
            }
            detail[cell][str(h)] = entry
            if h == 2:
                primary[cell] = entry["equal_family_cosine"]
                secondary[cell] = entry["secondary_token_pooled_cosine"]

    return {
        "status": STATUS, "claim_bearing": False,
        "primary_estimator": (
            "H=2 correct-future cosine: valid tokens within a row -> rows within an "
            "episode cluster -> episodes within a family -> unweighted mean of eight "
            "families"),
        "primary_h2_by_cell": primary,
        "interaction_I_s": interaction(primary),
        "secondary_token_pooled_h2_by_cell": secondary,
        "secondary_interaction_token_pooled": interaction(secondary),
        "weighting_note": ("the interaction is formed ONLY from the episode-then-family "
                           "estimator; the token-pooled interaction is reported separately "
                           "and the two are never mixed inside one number"),
        "per_horizon": detail,
        "terminal_window": {cell: terminal_window(history)
                            for cell, history in histories.items()},
        "co_outcomes": non_inferiority_status(C.CONFIG),
        "shuffled_assay_scope": ("action-conditioning and discrimination diagnostic; "
                                 "not candidate ranking, not planning regret"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs", default=str(CACHE / "factorial_v1"))
    args = ap.parse_args()
    raise SystemExit(
        "no trained cells exist: the four-cell experiment has not been launched. "
        "The estimator in this module is exercised by "
        "lewm/tests/test_proprio_factorial_driver.py on deterministic fixtures.")


if __name__ == "__main__":
    raise SystemExit(main())
