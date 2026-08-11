"""Utility-scorer contract v1.2 — the v1 contract retargeted onto oracle v1.2.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Baseline: the frozen prospective scorer contract
``d32118552b6fd373aefab143917bb04e63ffbe196129266a1546affc08f763ff``.

**Only the target definitions and the bound digests change.** Everything the
frozen contract fixes — separate progress/safety/completion heads, the full
H=1..4 latent-trajectory input, the explicit goal binding, the 10-D five-tick
post-slew candidate action input, the trunk architecture, the fixed optimiser
and epoch budget, final-epoch weights with no best-epoch selection, the
scene-disjoint fit/calibration split, every component and composite
qualification criterion, the no-latent baseline and the >= 0.05 pairwise
dominance requirement — is carried over verbatim.

What changes, and nothing else:

* ``progress`` target becomes the frozen continuous metric-geodesic progress of
  oracle v1.2 (``840328d9…``) instead of the integer BFS-cell difference;
* ``safety`` target becomes the frozen graded path-level safety cost of oracle
  v1.2 (``5cf4572b…``) instead of the binary max-over-ticks hazard;
* ``completion`` is unchanged — the bound landmark reached at or before the
  branch horizon;
* the composite is unchanged: ``U_hat = 1.0*P_hat - 2.0*S_hat + 0.5*C_hat``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from lewm.oracle.go2_branch_oracle_v1_2 import (
    UTILITY_WEIGHTS,
    oracle_digest as oracle_v1_2_digest,
    progress_digest,
    safety_digest,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
BASELINE_CONTRACT_DIGEST = (
    "d32118552b6fd373aefab143917bb04e63ffbe196129266a1546affc08f763ff"
)
CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)

# ---- the target encoder and preprocessing, bound exactly as the factorial ran --
TARGET_ENCODER = {
    "arm": "vjepa2_1_vitl_dist_vitG_384",
    "checkpoint": "~/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "frozen": True, "movement": "none", "ema": "none",
    "token_grid": [24, 32], "tokens": 768, "token_dim": 1024,
    "preprocess": "scripts/dev_frozen_dense_representation_encoders_v1.py::"
                  "preprocess_vjepa — PIL RGB, resize to (512, 384) BICUBIC, "
                  "ImageNet normalisation",
    "target_normalisation": "F.layer_norm over the token dimension "
                            "(run_dev_v03_temporal_action_jepa_v1.normalise)",
}

RENDER_CONTRACT = {
    "name": "textured_v03",
    "genesis_camera": "native 640x480, pinhole, platform-manifest intrinsics",
    "camera_pose": "lewm_genesis safe_camera_pose_from_base with the pack's "
                   "effective mount xyz/rpy and the pack camera-safety config",
    "apply_textures": True,
    "render_robot": False,
    "store_resolution_wh": [224, 224],
    "resample": "PIL LANCZOS (render_replay_genesis._resize_rgb)",
    "format": "png",
    "cadence": "one frame per command tick boundary; context slots and H=1..4 "
               "targets are one action block (5 ticks, 0.5 s) apart",
}

PREDICTOR_INPUT_CONTRACT = {
    "context_slots": 3,
    "context_frame_steps": "the three block boundaries s-10, s-5, s, where s is "
                           "the branch capture step",
    "horizons": [1, 2, 3, 4],
    "horizon_frame_steps": "s+5h — the end of branch block h",
    "action": "scripts/dev_action_slew_reconstruction_v1.flatten — 10-D, "
              "five ticks x (vx, yaw_rate), post-slew",
    "proprio": "30-D sensed physical state (projected gravity offset by "
               "(0,0,-1), body angular velocity, 12 joint positions, 12 joint "
               "velocities in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER), five "
               "trailing 10 Hz samples per slot",
    "control": "2-D previous applied command (vx, yaw_rate), five samples per slot",
    "normalisation": "the frozen train-split z-score statistics "
                     "(proprio_norm_stats.json); projected gravity is offset only",
}

SCORER = {
    "name": "go2_utility_scorer_v1_2",
    "baseline_contract": BASELINE_CONTRACT_DIGEST,
    "composite": "U_hat = 1.0*progress_hat - 2.0*safety_hat + 0.5*completion_hat",
    "weights": UTILITY_WEIGHTS,
    "consumes": {
        "latent_trajectory": "the FULL H=1..4 latent trajectory, not H=4 alone",
        "temporal_aggregation": "per-horizon MLP 1024->512 shared across h, then "
                                "a learned attention pool over h=1..4; the pooled "
                                "vector feeds all three heads",
        "spatial_aggregation": "mean over the 768 tokens of each horizon grid",
        "candidate_action_sequence_supplied": True,
        "action_representation": "the 10-D five-tick post-slew trajectory per "
                                 "block, 4 blocks (40-D)",
        "goal_context_supplied": True,
    },
    "goal_binding": {
        "representation": "landmark identifier plus its planning-time observable "
                          "(bearing_body_rad, range_m) at the branch state",
        "assigned_at": "snapshot time, before any branch is executed",
        "prohibited": "inferring the goal after collection from whichever "
                      "landmark yields the most favourable progress",
        "geodesic_distance_is_a_label_only": True,
    },
    "heads": {
        "progress": {"kind": "regression",
                     "target": "oracle v1.2 continuous metric-geodesic progress",
                     "target_digest": progress_digest()},
        "safety": {"kind": "bounded regression in [0,1] with a logistic output",
                   "target": "oracle v1.2 graded path-level safety cost",
                   "target_digest": safety_digest()},
        "completion": {"kind": "binary",
                       "target": "bound landmark reached at or before the horizon"},
        "rationale": "separate heads are separately qualifiable, so one failing "
                     "component cannot silently corrupt the composite",
    },
    "training": {
        "budget": "fixed epoch budget, FINAL-epoch weights, no best-epoch selection",
        "epochs": 60, "batch": 64, "lr": 3e-4, "weight_decay": 0.01,
        "grad_clip": 1.0, "optimiser": "AdamW", "seed": 20260811,
        "fit_calibration_split": "BY SCENE, never by branch or row",
    },
    "no_latent_baseline": {
        "inputs": "candidate action identity and goal context ONLY, no latent",
        "same_heads_budget_and_split": True,
    },
    "qualification_on_true_latent_trajectories": {
        "performed_on": "the scene-disjoint 24-state calibration set, using TRUE "
                        "latent trajectories, before any predicted latent is scored",
        "criteria": {
            "progress": "Spearman rank correlation >= 0.50",
            "safety": "ROC AUC >= 0.75 and calibration error <= 0.10",
            "completion": "ROC AUC >= 0.75 and calibration error <= 0.10",
        },
        "criterion_definitions": {
            "safety_auc_label": "the v1.2 safety cost is graded, so ROC AUC is "
                                "computed against the indicator (safety > 0), i.e. "
                                "whether the branch incurred any path hazard at all",
            "calibration_error": "expected calibration error: |mean predicted - "
                                 "mean actual| within each of 10 equal-width bins "
                                 "of the predicted value, weighted by bin count; "
                                 "computed against the GRADED target for safety "
                                 "and the binary target for completion",
            "pairwise_ordering_accuracy": "over all within-state candidate pairs "
                                          "whose true utilities differ by more than "
                                          "the frozen 0.02 tie tolerance",
            "composite": "pairwise candidate ordering accuracy >= 0.65 within a state",
            "baseline_dominance": "the composite must exceed the no-latent "
                                  "baseline by >= 0.05 pairwise accuracy",
        },
        "degenerate_completion": "if completion labels are degenerate in fit or "
                                 "calibration the scorer FAILS qualification; the "
                                 "head is not removed and the utility is not changed",
        "on_failure": "STOP; do not open any predictor checkpoint",
    },
    "prohibited_inputs": [
        "the realised branch outcome",
        "a true future latent at application time",
        "future proprioception",
        "simulator ground-truth pose or velocity unavailable in deployment",
        "model-specific calibration on the final evaluation branches",
    ],
    "applied_identically_to": "all 32 frozen predictors, all four cells, all eight seeds",
}


def contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "name": "go2_utility_scorer_contract_v1_2",
        "baseline_contract_digest": BASELINE_CONTRACT_DIGEST,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "oracle_v1_2_digest": oracle_v1_2_digest(),
        "progress_target_digest": progress_digest(),
        "safety_target_digest": safety_digest(),
        "target_encoder": TARGET_ENCODER,
        "render_contract": RENDER_CONTRACT,
        "predictor_input_contract": PREDICTOR_INPUT_CONTRACT,
        "scorer": SCORER,
    }


def contract_digest() -> str:
    return hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    print(json.dumps({"scorer_contract_v1_2_digest": contract_digest(),
                      "contract": contract()}, indent=2))
