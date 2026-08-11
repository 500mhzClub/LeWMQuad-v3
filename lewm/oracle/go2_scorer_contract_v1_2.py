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
import os
from pathlib import Path
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
SCORER_FIT_ALLOCATION_DESIGN_DIGEST = (
    "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"
)
ROOT = Path(__file__).resolve().parents[2]


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file_binding(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    raw = path.read_bytes()
    return {
        "path": relative_path,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

# ---- the target encoder and preprocessing, bound exactly as the factorial ran --
TARGET_ENCODER = {
    "arm": "scripts.dev_frozen_dense_representation_encoders_v1."
           "VJepa21CroppedV03Arm",
    "constructor": "vjepa2_1_vit_large_384",
    "checkpoint": "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "checkpoint_sha256": (
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
    ),
    "checkpoint_byte_count": 5_151_198_524,
    "source_repository": "facebookresearch/vjepa2",
    "source_repository_commit": "204698b45b3712590f06245fbfba32d3be539812",
    "frozen": True, "movement": "none", "ema": "none",
    "token_grid": [24, 32], "tokens": 768, "token_dim": 1024,
    "token_order": "row-major 24x32 patch grid from norms_block[-1]",
    "preprocess": "scripts/dev_frozen_dense_representation_encoders_v1.py::"
                  "preprocess_vjepa_v03_crop — require 224x224 PIL RGB, crop "
                  "rows 28:196 to 224x168, resize to (512,384) BICUBIC, "
                  "ImageNet normalisation, no padding",
    "preprocessing_identity_sha256": (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
    ),
    "target_normalisation": "F.layer_norm over the token dimension "
                            "(run_dev_v03_temporal_action_jepa_v1.normalise)",
}

RENDER_CONTRACT = {
    "name": "textured_v03",
    "native_resolution_wh": [224, 224],
    "store_resolution_wh": [224, 224],
    "resample": "none",
    "format": "png",
    "genesis_yfov_deg": 78.323,
    "near_m": 0.05,
    "far_m": 200.0,
    "scene_geometry": "manifest walls, obstacles and landmarks; no robot and "
                      "no distractors",
    "visual_mode": "textured_v03",
    "apply_textures": True,
    "camera_pose": "captured physical base pose plus the nominal platform "
                   "mount; no per-scene extrinsic jitter and no safety retraction",
    "render_execution": "separate single non-batched static render scene; the "
                        "physical CPU branch scene is never rendered",
    "historical_renderer": {
        "path": "scripts/render_replay_v03.py",
        "sha256": "99453ee5fe5c068a0d9c63d663e651a2a871971dd6122fda10fc72b909fb659d",
        "byte_count": 9_590,
    },
    "runtime_wrapper_contract_digest": (
        "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"
    ),
    "cadence": "one frame per command tick boundary; context slots and H=1..4 "
               "targets are one action block (5 ticks, 0.5 s) apart",
}

PREPROCESS_CONTRACT = {
    "implementation": "preprocess_vjepa_v03_crop",
    "source_frame_hw": [224, 224],
    "crop_xyxy": [0, 28, 224, 196],
    "cropped_hw": [168, 224],
    "input_hw": [384, 512],
    "resample": "PIL BICUBIC",
    "normalisation": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "padding": "none",
    "token_grid_hw": [24, 32],
    "token_dim": 1024,
    "output_layer": "encoder final block, norms_block[-1]",
    "preprocessing_identity_sha256": (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
    ),
}

CORPUS_SELECTION_CONTRACT = {
    "name": "go2_planning_corpus_selection_v1_2",
    "allocation_design_digest": SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
    "candidate_allocator_contract_digest": (
        "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"
    ),
    "scene_order": "all eligible corpus scenes sorted by (family, scene_id)",
    "scorer_fit": "first snapshot-time eligible 15 distinct scenes per family "
                  "after all frozen exclusions; five per frozen stratum",
    "final_eval": "first snapshot-time eligible 25 remaining distinct scenes per "
                  "family after all frozen exclusions and scorer-fit scenes",
    "one_state_per_scene": True,
    "warmup_blocks": [40, 120],
    "drive_seed_rule": "20260811 XOR crc32(scene_id)",
    "backend": "cpu",
    "fit_calibration": "within each family/stratum, lexicographically first "
                       "selected scene is calibration and the remaining four fit",
    "strata": {
        "general": "reachable bound landmark with graph_edges >= 2",
        "safety_enriched": "general plus snapshot-time body-probe clearance <= 0.10m",
        "completion_enriched": "snapshot-time metric geodesic <= 0.75m and "
                               "absolute body bearing <= 75 degrees",
    },
    "goal_type": "snapshot-bound landmark material_id; allocator-only balance key",
    "candidate_allocation": "canonical exact allocation manifest under the bound "
                            "allocator for scorer_fit; all 12 for final_eval",
    "exclusions": [
        "all 80 scenes represented in the frozen factorial manifest",
        "all v1.1 replay-qualification and failed-pilot scenes",
        "all successful v1.2 pilot scenes",
        "for final_eval, every scorer-fit selected scene",
    ],
    "candidate_outcomes_used_for_selection": False,
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
        "model_numeric_vector": "[sin(bearing_body_rad), "
                                "cos(bearing_body_rad), range_m]; landmark_id is "
                                "the immutable semantic/label binding and is not "
                                "embedded as a scene-specific token",
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


def source_bindings() -> dict[str, Any]:
    """Implementations that must be frozen before any branch outcome exists."""

    return {
        "branch_corpus_and_goal_binding": _file_binding(
            "scripts/build_go2_branch_corpus_v1_2.py"),
        "candidate_allocator": _file_binding(
            "lewm/oracle/go2_candidate_allocation_v1_2.py"),
        "latent_encoder_driver": _file_binding(
            "scripts/encode_go2_branch_corpus_v1_2.py"),
        "target_encoder_and_preprocessing": _file_binding(
            "scripts/dev_frozen_dense_representation_encoders_v1.py"),
        "candidate_action_representation": _file_binding(
            "scripts/dev_action_slew_reconstruction_v1.py"),
        "fit_calibration_estimators_and_trainer": _file_binding(
            "scripts/train_go2_utility_scorer_v1_2.py"),
        "historical_renderer": _file_binding("scripts/render_replay_v03.py"),
        "historical_renderer_wrapper": _file_binding(
            "lewm/oracle/go2_textured_v03_renderer.py"),
    }


def contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "name": "go2_utility_scorer_contract_v1_2",
        "baseline_contract_digest": BASELINE_CONTRACT_DIGEST,
        "scorer_fit_allocation_design_digest": SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "oracle_v1_2_digest": oracle_v1_2_digest(),
        "progress_target_digest": progress_digest(),
        "safety_target_digest": safety_digest(),
        "target_encoder": TARGET_ENCODER,
        "render_contract": RENDER_CONTRACT,
        "preprocess_contract": PREPROCESS_CONTRACT,
        "corpus_selection_contract": CORPUS_SELECTION_CONTRACT,
        "corpus_selection_digest": _digest(CORPUS_SELECTION_CONTRACT),
        "predictor_input_contract": PREDICTOR_INPUT_CONTRACT,
        "scorer": SCORER,
        "bound_implementations": source_bindings(),
    }


def contract_digest() -> str:
    return _digest(contract())


def target_encoder_digest() -> str:
    return _digest(TARGET_ENCODER)


def render_contract_digest() -> str:
    return _digest(RENDER_CONTRACT)


def preprocess_contract_digest() -> str:
    return _digest(PREPROCESS_CONTRACT)


def _stream_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def issue_contract(path: Path) -> dict[str, Any]:
    """Validate external bindings and atomically issue the pre-outcome contract."""

    checkpoint = Path(TARGET_ENCODER["checkpoint"])
    if (not checkpoint.is_file()
            or checkpoint.stat().st_size != TARGET_ENCODER["checkpoint_byte_count"]
            or _stream_file_sha256(checkpoint) != TARGET_ENCODER["checkpoint_sha256"]):
        raise RuntimeError("frozen target-encoder checkpoint binding failed")
    renderer = ROOT / RENDER_CONTRACT["historical_renderer"]["path"]
    if (renderer.stat().st_size != RENDER_CONTRACT["historical_renderer"]["byte_count"]
            or _stream_file_sha256(renderer)
            != RENDER_CONTRACT["historical_renderer"]["sha256"]):
        raise RuntimeError("historical textured-v03 renderer binding failed")
    from lewm.oracle.go2_candidate_allocation_v1_2 import (
        allocation_contract_digest,
    )
    from lewm.oracle.go2_textured_v03_renderer import renderer_contract_digest
    from scripts import dev_frozen_dense_representation_encoders_v1 as encoders

    arm = encoders.VJepa21CroppedV03Arm(
        checkpoint=checkpoint, constructor=TARGET_ENCODER["constructor"])
    if (arm.preprocess is not encoders.preprocess_vjepa_v03_crop
            or list(arm.token_grid) != TARGET_ENCODER["token_grid"]
            or int(arm.token_dim) != TARGET_ENCODER["token_dim"]
            or encoders.preprocessing_hash(arm)
            != TARGET_ENCODER["preprocessing_identity_sha256"]):
        raise RuntimeError("frozen target-encoder preprocessing binding failed")
    if renderer_contract_digest() != RENDER_CONTRACT[
            "runtime_wrapper_contract_digest"]:
        raise RuntimeError("historical renderer-wrapper contract binding failed")
    if allocation_contract_digest() != CORPUS_SELECTION_CONTRACT[
            "candidate_allocator_contract_digest"]:
        raise RuntimeError("candidate-allocation contract binding failed")
    payload = {
        "schema": "go2_utility_scorer_contract_v1_2_artifact",
        "status": STATUS,
        "complete": True,
        "scorer_contract_v1_2_digest": contract_digest(),
        "target_encoder_checkpoint_verified": True,
        "target_encoder_preprocessing_verified": True,
        "historical_renderer_verified": True,
        "historical_renderer_wrapper_verified": True,
        "candidate_allocator_verified": True,
        "contract": contract(),
    }
    payload["contract_artifact_digest"] = _digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.output is None:
        print(json.dumps({"scorer_contract_v1_2_digest": contract_digest(),
                          "contract": contract()}, indent=2))
    else:
        issued = issue_contract(arguments.output)
        print(json.dumps({
            "output": str(arguments.output),
            "scorer_contract_v1_2_digest":
                issued["scorer_contract_v1_2_digest"],
            "contract_artifact_digest": issued["contract_artifact_digest"],
        }, indent=2, sort_keys=True))
