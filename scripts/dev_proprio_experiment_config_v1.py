#!/usr/bin/env python3
"""Frozen configuration for the proprioception x rollout factorial.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Declarative only -- importing this module
trains nothing.  It exists so that the experiment's every prospective choice is
recorded, hashed and unchangeable before the first run.

Nothing here may be edited once the first scientific run has started.  The
engineering-validation run is explicitly not a scientific run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402
from scripts import build_dev_v03_proprio_action_manifest_v1 as M  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CONFIG = {
    "status": STATUS,
    "claim_bearing": False,
    "name": "proprioception x rollout factorial, corrected action contract",
    "description": (
        "A reference-informed quadruped adaptation. NOT an official upstream "
        "configuration: no upstream config combines a V-JEPA 2.1 encoder with an "
        "action-conditioned predictor, the context is a fixed sliding three-frame "
        "window rather than the official growing context, positional treatment is "
        "learned-absolute rather than RoPE, and the target is a single endpoint "
        "frame rather than all-frame supervision."
    ),

    "scope_exclusions": [
        "growing context", "RoPE", "all-frame target supervision",
        "proprioceptive prediction target", "auxiliary proprioceptive loss",
        "encoder movement", "counterfactual branch corpus",
    ],

    "cells": {
        "rgb_one_step": {"use_proprio": False, "objective": "e1"},
        "rgb_rollout": {"use_proprio": False, "objective": "1.5*e1 + 0.5*e2"},
        "proprio_one_step": {"use_proprio": True, "objective": "e1"},
        "proprio_rollout": {"use_proprio": True, "objective": "1.5*e1 + 0.5*e2"},
    },

    "encoder": {
        "checkpoint": "~/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
        "frozen": True, "movement": "none", "ema": "none",
        "preprocessing": "unchanged v03 centre-crop contract",
    },

    "predictor": {
        "module": "scripts/dev_proprio_predictor_v1.py::ProprioActionPredictor",
        "width": 384, "depth": 6, "heads": 6,
        "conditioning": "AdaLN-Zero, unchanged",
        "context_slots": 3, "positional": "learned absolute spatial + temporal",
        "target": "single endpoint frame, visual only",
    },

    "action": {
        "representation": "five-tick post-slew command trajectory",
        "dim": SLEW.ACTION_DIM,
        "reconstruction": "applied[k] = prev + clip(requested[k] - prev, +-rate)",
        "rates": {"vx": SLEW.VX_RATE, "vy": SLEW.VY_RATE, "yaw": SLEW.YAW_RATE},
        "reset_behaviour": "previous applied command returns to (0,0,0) at a respawn",
        "identical_in_all_cells": True,
        "planning_time_function_is_identical": True,
        "future_measured_body_motion_used": False,
    },

    "proprioception": {
        "dim": M.PROPRIO_DIM, "samples_per_slot": M.SAMPLES_PER_SLOT,
        "channels": [list(c) for c in M.CHANNELS],
        "window": "trailing, ending at the slot's own observation",
        "entry": "one token per context slot, predictor only",
        "absence": "learned absent token + validity mask for predicted slots",
        "observed_slots_by_horizon": {"1": 3, "2": 2, "3": 1, "4": 0},
        "target": "none -- the prediction target stays visual",
        "excluded": ["body linear velocity", "absolute yaw", "world pose",
                     "camera extrinsics", "foot contacts", "joint effort",
                     "IMU linear acceleration"],
    },

    "pairing": {
        "shared_initialisation": "torch.manual_seed(seed) before shared parameters",
        "proprio_initialisation": f"separate generator, seed + {7919}",
        "data_order": "one generator seeded from the seed, identical row order in all cells",
        "augmentation_order": "no train-time augmentation in this experiment",
        "optimisation_randomness": "no dropout, no stochastic depth; AdamW is deterministic",
        "verified_by": "lewm/tests/test_proprio_action_contract.py::"
                       "test_shared_weights_are_identical_across_cells_for_one_seed",
    },

    "budget": {
        "epochs": 24, "fixed": True, "extension_permitted": False,
        "batch": 4, "optimiser": "AdamW", "lr": 3.0e-4, "weight_decay": 0.01,
        "grad_clip": 1.0, "precision": "bf16",
    },

    "checkpoint_rule": {
        "rule": "fixed epoch 21 for every technically valid run",
        "selection_permitted": False,
        "diagnostics_reported": ["terminal-window (19-23) mean", "terminal-window sd",
                                 "OLS slope over epochs 14-23"],
        "exclusion_on_trend": "none -- an improving or deteriorating run is reported, not dropped",
    },

    "endpoints": {
        "primary": "equal-family H=2 correct-future changed-token cosine",
        "principal_estimand": "I_s = (PropRoll_s - PropOne_s) - (RGBRoll_s - RGBOne_s)",
        "secondary": "corpus-weighted results",
        "h3": "beyond-trained-horizon transfer",
        "h4": "longer-horizon diagnostic",
        "combined_h2_h3_endpoint": "prohibited",
        "mandatory_non_regression": [
            "occupied spatial information (occupied IoU / precision / recall)",
            "correct-versus-shuffled action-sequence margin",
        ],
        "non_regression_note": (
            "a cosine gain accompanied by a material loss of spatial retention or "
            "action-sequence discrimination is not an unqualified success"
        ),
    },

    "seed_design": {
        "type": "capped internal pilot",
        "initial_stage": {"quadruplets": 5, "runs": 20},
        "interim_analysis": (
            "compute ONLY the seed-level standard deviation of the H=2 interaction; "
            "no comparative mean, sign, interval or family plot may be produced first"
        ),
        "power_inputs": {"minimally_relevant_interaction": 0.005,
                         "alpha": 0.05, "power": 0.80,
                         "sd_estimate": "conservative upper confidence bound"},
        "bounds": {"min_quadruplets": 5, "max_quadruplets": 10},
        "escalation_forbidden_on": ["observed interaction mean", "direction", "significance"],
        "if_requirement_exceeds_cap": "complete 10 and label the study precision-limited",
        "replication_unit": "the training seed quadruplet",
        "episode_bootstrap_role": (
            "quantifies within-seed evaluation uncertainty only; it may not "
            "substitute for between-training-seed replication"
        ),
    },

    "family_diagnostic": {
        "family": "local_composite_motifs",
        "status": "prospectively declared diagnostic, not a primary endpoint",
        "reason": "post hoc horizon-dependent control advantage in the completed RGB-only study",
        "tuning_to_this_family": "prohibited",
        "reporting": "equal-family reporting preserved so corpus weighting cannot hide it",
    },

    "data": {
        "manifest": "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_rows.jsonl",
        "normalisation": "proprio_norm_stats.json, train split only, frozen and hashed",
        "identical_rows_in_all_cells": True,
    },
}


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()


def required_quadruplets(sd_upper: float, delta: float = 0.005,
                         z_alpha: float = 1.959964, z_power: float = 0.841621) -> int:
    """Paired-difference sample size for the interaction, capped at 10.

    ``I_s`` is a single per-seed number, so this is a one-sample paired design:
    n >= ((z_alpha + z_power) * sd / delta)^2, with ``sd_upper`` the conservative
    UPPER confidence bound on the seed-level sd of ``I_s``, not the point estimate.
    """
    import math
    need = math.ceil(((z_alpha + z_power) * sd_upper / delta) ** 2)
    return max(CONFIG["seed_design"]["bounds"]["min_quadruplets"],
               min(need, CONFIG["seed_design"]["bounds"]["max_quadruplets"])), need


def sd_upper_bound(sd_hat: float, n: int, confidence: float = 0.80) -> float:
    """One-sided upper confidence bound on a normal sd from n observations.

    Uses the chi-square lower quantile: sd_upper = sd_hat * sqrt((n-1)/chi2_q).
    Falls back to a fixed inflation factor when scipy is unavailable, so the
    pilot never silently uses the point estimate.
    """
    try:
        from scipy.stats import chi2
        quantile = chi2.ppf(1.0 - confidence, n - 1)
        return sd_hat * ((n - 1) / quantile) ** 0.5
    except Exception:
        return sd_hat * 1.35     # documented conservative fallback for n = 5


if __name__ == "__main__":
    print(json.dumps({"config_sha256": config_sha256(), **CONFIG}, indent=2))
