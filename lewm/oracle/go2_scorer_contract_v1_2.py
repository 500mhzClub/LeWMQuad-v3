"""Utility-scorer contract v1.2 — the v1 contract retargeted onto oracle v1.2.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Baseline: the frozen prospective scorer contract
``d32118552b6fd373aefab143917bb04e63ffbe196129266a1546affc08f763ff``.

The oracle-v1.2 target retargeting remains unchanged.  The first prospective
selector amendment made graph hops diagnostic for completion enrichment after
the original shared hop conjunction proved infeasible on two graph families.
The exhaustive 1,284-scene outcome-free census then proved that its start-state
``d0 <= 0.75m`` ceiling left the required small-maze completion cell empty.
This successor binds the final prospective horizon-reachability amendment made
before any identity manifest or scientific outcome.  Everything else the
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
* completion-enriched state selection retains graph-hop-zero as diagnostic and
  replaces the superseded start-distance ceiling with the outcome-free
  horizon-reachability condition ``max(d0 - 0.75m, 0) <= L_max``.  The 75-degree
  bearing condition, oracle graph-cell completion label, and production
  collector claim predicate remain unchanged and distinct.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from lewm.oracle.go2_branch_oracle_v1_2 import (
    UTILITY_WEIGHTS,
    oracle_digest as oracle_v1_2_digest,
    progress_digest,
    safety_digest,
)
from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC
from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as INVALID_IDS
from lewm.oracle import go2_scorer_projection_fix_interruption_v1 as INTERRUPTION
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as STATE_SELECTOR

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
SCORER_PACKAGE_ROOT_RELATIVE = Path(
    ".generated/go2_utility_scorer_v1_2"
)
SCORER_CONTRACT_ARTIFACT_NAME = "scorer_contract_v1_2.json"

SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT = {
    "scorer_contract_v1_2_digest": (
        "0fc7a3db0ca86ae206050ee6da2894208fa11707e840b112a8a6810e18ac3e21"
    ),
    "contract_artifact_digest": (
        "375d372b2196c89c5e9856128bcf15386ed2c6b79bca01ad070f4a146d6c9d24"
    ),
    "raw_sha256": (
        "c20967ade214b4815f288e811a5e53108171f8e3ed470b60cd4c71d75e12f43f"
    ),
    "byte_count": 13_839,
    "outcomes_generated": False,
    "disposition": "superseded_pre_run_preserve_do_not_reuse",
}

# Immediate predecessor issued cleanly at 38e7fc84 before state selection.  It
# generated no branch identity or outcome, but its shared hops>=1/completion
# conjunction is proven infeasible and it must be archived rather than reused.
SUPERSEDED_GRAPH_INFEASIBLE_CONTRACT_ARTIFACT = (
    STATE_SELECTOR.PREDECESSOR_CONTRACT_ARTIFACT
)


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
    "name": "go2_planning_corpus_selection_v1_2_selector_amendment_v2",
    "predecessor_selection_digest":
        STATE_SELECTOR.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST,
    "state_selector_amendment":
        STATE_SELECTOR.state_selector_amendment_contract(),
    "state_selector_amendment_digest":
        STATE_SELECTOR.state_selector_amendment_digest(),
    "allocation_design_digest": SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
    "candidate_allocator_contract_digest": (
        "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"
    ),
    "candidate_allocator_amendment_digest": ALLOC.allocation_amendment_digest(),
    "scene_order": "all eligible corpus scenes sorted by (family, scene_id)",
    "scorer_fit": (
        "the unchanged one-pass family ordering retains 37 exact predecessor "
        "identities and fills eight completion vacancies with the first "
        "eligible snapshots in their retained-anchor lexical intervals; the "
        "other four non-small successor families remain unchanged; "
        "small-enclosed general/safety retain their frozen ordering and its "
        "completion cell is the first lexicographically feasible "
        "five-distinct-scene combination under the bound all-120 identity "
        "projection, unchanged canonical allocator, and exact-mask "
        "horizon-reachability search; exactly 15 distinct scenes per family "
        "and five per frozen stratum"
    ),
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
        "completion_enriched": (
            "separately enumerated finite snapshot-bound landmark, including "
            "graph_hops == 0, for which max(continuous metric-geodesic d0 - "
            "the unchanged selector parameter r_complete=0.75m, 0) is no "
            "greater than L_max from the exact allocated six-candidate subset; "
            "L_max is computed from the actual previous applied command, exact "
            "requested plans, frozen slew limiter and exact 20 ticks without "
            "branch execution; absolute body bearing remains <=75 degrees and "
            "snapshot task-completed, goal-claimed, terminated and truncated "
            "flags remain false; graph hops are diagnostic only"
        ),
    },
    "state_selection_priority":
        list(STATE_SELECTOR.SCORER_FIT_SELECTION_PRIORITY),
    "completion_semantic_separation":
        STATE_SELECTOR.state_selector_amendment_contract()["preserved"][
            "completion_semantic_separation"
        ],
    "completion_horizon_reachability":
        STATE_SELECTOR.state_selector_amendment_contract()["single_replacement"],
    "candidate_allocation_circularity_resolution":
        STATE_SELECTOR.state_selector_amendment_contract()[
            "allocation_circularity_resolution"
        ],
    "state_selector_feasibility_receipt": {
        "required_before_successor_contract_issue": True,
        "schema": STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_SCHEMA,
        "path": STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH,
        "all_eight_families_all_three_strata": True,
        "minimum_distinct_eligible_scenes_per_family_stratum": 5,
        "identity_and_outcome_free": True,
    },
    "preserved_state_mixed_precontract_disposition_receipt": {
        "required_before_successor_contract_issue": True,
        "schema":
            STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_SCHEMA,
        "path":
            STATE_SELECTOR.PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH,
        "frozen_failed_predecessor_state_count": 45,
        "retained_predecessor_state_count": 37,
        "rejected_predecessor_state_count": 8,
        "prospective_replacement_slot_count": 8,
        "retained_identity_unchanged_and_outcome_free": True,
        "replacements_selected_pre_outcome_only": True,
        "replacement_scene_policy": (
            "a rejected predecessor scene may be reused only for a structurally "
            "different physical snapshot; re-signing the rejected snapshot or "
            "reusing any retained scene is forbidden"
        ),
        "frozen_failure_receipt_preserved": True,
        "candidate_allocation_verified": False,
        "scientific_reason": (
            "the frozen outcome-free phase-1 revalidation established 37 exact "
            "passes and eight completion-only amended-classification failures; "
            "only those eight slots are prospectively replaced before allocation"
        ),
        "not_a_response_to_branch_or_scorer_outcomes": True,
    },
    "preserved_state_revalidation_receipt": {
        "required_after_all_120_identities_and_candidate_allocation": True,
        "required_before_active_identity_manifest_or_branch_execution": True,
        "schema": STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA,
        "path": STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
        "retained_predecessor_state_count": 37,
        "replacement_state_count": 8,
        "retained_exact_allocated_candidate_masks_verified": True,
        "rejected_predecessor_identities_absent": True,
        "replacement_slots_filled_exactly": True,
        "active_disjointness": (
            "120 unique scenes, episode clusters, state identities and snapshot "
            "observation-boundary identities"
        ),
        "expected_completion_enriched_state_count": 40,
        "all_completion_exact_allocated_mask_reachability_verified": True,
        "retained_identity_unchanged_and_outcome_free": True,
        "replacement_identity_selected_outcome_free": True,
    },
    "goal_type": "snapshot-bound landmark material_id; allocator-only balance key",
    "candidate_allocation": "canonical exact allocation manifest under the bound "
                            "allocator for scorer_fit; all 12 for final_eval; the "
                            "sole reversing candidate occurs in exactly 60 distinct "
                            "scorer-fit state subsets under the prospective amendment",
    "exclusions": [
        "all 80 scenes represented in the frozen factorial manifest",
        "all v1.1 replay-qualification and failed-pilot scenes",
        "all successful v1.2 pilot scenes",
        "all 45 scenes in the exactly bound abandoned pre-outcome scorer-fit attempt",
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
                       "target": "oracle v1.2 bound landmark cell reached at any "
                                 "candidate branch tick at or before the horizon; "
                                 "not the production collector claim or task-reset "
                                 "predicate"},
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
        "scorer_contract_implementation": _file_binding(
            "lewm/oracle/go2_scorer_contract_v1_2.py"),
        "branch_corpus_and_goal_binding": _file_binding(
            "scripts/build_go2_branch_corpus_v1_2.py"),
        "candidate_allocator": _file_binding(
            "lewm/oracle/go2_candidate_allocation_v1_2.py"),
        "candidate_allocation_amendment_authority": _file_binding(
            ALLOC.AMENDMENT_ARTIFACT_PATH),
        "candidate_allocation_preoutcome_failure_receipt": _file_binding(
            ALLOC.FAILURE_RECEIPT_PATH),
        "state_selector_amendment_implementation": _file_binding(
            "lewm/oracle/go2_scorer_state_selector_amendment_v2.py"),
        "state_selector_predecessor_amendment_implementation": _file_binding(
            "lewm/oracle/go2_scorer_state_selector_amendment_v1.py"),
        "state_selector_amendment_authority": _file_binding(
            STATE_SELECTOR.AMENDMENT_ARTIFACT_PATH),
        "projection_fix_interruption_lineage": _file_binding(
            "lewm/oracle/go2_scorer_projection_fix_interruption_v1.py"),
        "state_selector_preoutcome_failure_receipt": _file_binding(
            STATE_SELECTOR.FAILURE_REPORT_PATH),
        "state_selector_predecessor_amendment_authority": _file_binding(
            STATE_SELECTOR.PREDECESSOR_AMENDMENT_ARTIFACT["path"]),
        "state_selector_initial_graph_failure_receipt": _file_binding(
            STATE_SELECTOR.PREDECESSOR.FAILURE_RECEIPT_PATH),
        "oracle_v1_2_completion_target_implementation": _file_binding(
            STATE_SELECTOR.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                "oracle_v1_2_completion_target"
            ]["path"]),
        "production_designated_goal_claim_implementation": _file_binding(
            STATE_SELECTOR.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                "snapshot_production_designated_goal_claim"
            ]["path"]),
        "production_task_completion_reset_implementation": _file_binding(
            STATE_SELECTOR.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                "production_task_completion_and_reset"
            ]["path"]),
        "invalid_scorer_identity_exclusion": _file_binding(
            "lewm/oracle/go2_invalid_scorer_identity_exclusion_v1_2.py"),
        "latent_encoder_driver": _file_binding(
            "scripts/encode_go2_branch_corpus_v1_2.py"),
        "target_encoder_and_preprocessing": _file_binding(
            "scripts/dev_frozen_dense_representation_encoders_v1.py"),
        "candidate_action_representation": _file_binding(
            "scripts/dev_action_slew_reconstruction_v1.py"),
        "fit_calibration_estimators_and_trainer": _file_binding(
            "scripts/train_go2_utility_scorer_v1_2.py"),
        "qualified_development_transfer_consumer": _file_binding(
            "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py"),
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
        "candidate_allocator_contract_digest": ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment": ALLOC.allocation_amendment_contract(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "state_selector_amendment":
            STATE_SELECTOR.state_selector_amendment_contract(),
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "preoutcome_projection_fix_interruption_lineage":
            INTERRUPTION.lineage_contract(),
        "superseded_graph_infeasible_contract_artifact":
            SUPERSEDED_GRAPH_INFEASIBLE_CONTRACT_ARTIFACT,
        "invalid_scorer_identity_exclusion":
            INVALID_IDS.INVALID_SCORER_IDENTITY_EXCLUSION,
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        "superseded_pre_run_contract_artifact":
            SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
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


def _compact_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _validated_repository_state(*, head: str, status: str,
                                top_level: str,
                                bindings: dict[str, Any]) -> dict[str, Any]:
    """Validate injected git facts and construct the clean-source binding."""

    try:
        resolved_top = Path(top_level).resolve()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("cannot resolve scorer source repository root") from exc
    if resolved_top != ROOT.resolve():
        raise RuntimeError("scorer source is not issued from the custody repository root")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("source repository HEAD is not a full SHA-1 commit")
    if status:
        raise RuntimeError(
            "source repository is not clean; commit every source/untracked change "
            "before scorer launch"
        )
    return {
        "schema": "go2_utility_scorer_v1_2_clean_source_binding",
        "source_repository_root": str(ROOT.resolve()),
        "source_repository_commit": head,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "git_status_untracked_files": "all",
        "git_ignored_generated_artifacts_permitted": True,
        "nonignored_tracked_or_untracked_changes_permitted": False,
        "bound_implementations": bindings,
        "bound_implementations_digest": _digest(bindings),
    }


def clean_source_binding() -> dict[str, Any]:
    """Require a clean exact HEAD while leaving ignored generated data alone."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return completed.stdout.strip("\n")

    try:
        head = git("rev-parse", "HEAD")
        top_level = git("rev-parse", "--show-toplevel")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify clean scorer source repository") from exc
    return _validated_repository_state(
        head=head, status=status, top_level=top_level, bindings=source_bindings())


def _has_inaccessible_custody_component(path: Path) -> bool:
    return any(
        part == ".."
        or part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        for part in path.parts
    )


def _assert_no_scorer_package_symlink(path: Path) -> None:
    if _has_inaccessible_custody_component(path):
        raise RuntimeError(
            "scorer-package path crosses an inaccessible custody component"
        )
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("symlinked scorer-package path is inaccessible")


def _managed_scorer_contract_output_path(
    path: Path, *, root: Path = ROOT,
) -> Path:
    """Pin the exact production contract output under its sole managed alias."""

    repository_root = Path(root)
    if not repository_root.is_absolute():
        repository_root = Path.cwd() / repository_root
    managed_root = repository_root / SCORER_PACKAGE_ROOT_RELATIVE
    requested = Path(path)
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    expected = managed_root / SCORER_CONTRACT_ARTIFACT_NAME
    if requested != expected:
        raise RuntimeError(
            "scorer contract must target the exact managed package artifact"
        )
    if (
        _has_inaccessible_custody_component(managed_root)
        or _has_inaccessible_custody_component(requested)
    ):
        raise RuntimeError(
            "scorer-contract output crosses an inaccessible custody component"
        )

    _assert_no_scorer_package_symlink(managed_root.parent)
    if managed_root.is_symlink():
        raw_target = managed_root.readlink()
        target = (
            raw_target
            if raw_target.is_absolute()
            else managed_root.parent / raw_target
        )
        if (
            target.name != managed_root.name
            or _has_inaccessible_custody_component(target)
        ):
            raise RuntimeError(
                "managed scorer-package alias target identity is inaccessible"
            )
        _assert_no_scorer_package_symlink(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                "managed scorer-package alias target is missing"
            ) from exc
    else:
        if not managed_root.is_dir():
            raise RuntimeError("managed scorer-package root is missing")
        canonical_root = managed_root.resolve(strict=True)
    if (
        not canonical_root.is_dir()
        or canonical_root.name != managed_root.name
        or _has_inaccessible_custody_component(canonical_root)
    ):
        raise RuntimeError("managed scorer-package root identity changed")
    _assert_no_scorer_package_symlink(canonical_root)
    canonical_output = canonical_root / SCORER_CONTRACT_ARTIFACT_NAME
    _assert_no_scorer_package_symlink(canonical_output)
    return canonical_output


def _prepare_contract_output(
    path: Path, payload: dict[str, Any], *, managed_root: Path | None = None,
) -> str:
    """Preserve a known pre-run predecessor and refuse unknown overwrites.

    Returns ``new``, ``current`` or ``superseded_archived``.  This helper is
    intentionally separate from external checkpoint validation so its recovery
    semantics can be tested without opening the 5.1 GB encoder checkpoint.
    """

    if managed_root is not None:
        if path.parent != managed_root:
            raise RuntimeError("scorer-contract output escaped its managed root")
        _assert_no_scorer_package_symlink(path)
        # Check the only possible archive directory before even reading an
        # active predecessor; a pre-existing descendant alias cannot become a
        # write target after the predecessor bytes are inspected.
        _assert_no_scorer_package_symlink(
            managed_root / "superseded_pre_run" / ".custody-probe"
        )

    if not path.exists():
        return "new"
    if not path.is_file():
        raise RuntimeError(f"scorer-contract output is not a file: {path}")
    raw = path.read_bytes()
    try:
        existing = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("refusing to overwrite malformed scorer-contract artifact") \
            from exc
    existing_self = existing.get("contract_artifact_digest")
    if existing_self != _digest({key: value for key, value in existing.items()
                                 if key != "contract_artifact_digest"}):
        raise RuntimeError("refusing to overwrite scorer-contract artifact with bad self digest")
    if existing == payload:
        return "current"

    predecessor = next((candidate for candidate in (
        SUPERSEDED_GRAPH_INFEASIBLE_CONTRACT_ARTIFACT,
        SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
    ) if (len(raw) == candidate["byte_count"]
          and hashlib.sha256(raw).hexdigest() == candidate["raw_sha256"]
          and existing.get("scorer_contract_v1_2_digest")
          == candidate["scorer_contract_v1_2_digest"]
          and existing_self == candidate["contract_artifact_digest"])), None)
    if predecessor is None:
        raise RuntimeError("refusing to overwrite an unknown scorer-contract artifact")
    archive = path.parent / "superseded_pre_run" / (
        "scorer_contract_v1_2."
        f"{predecessor['scorer_contract_v1_2_digest']}.json"
    )
    if managed_root is not None:
        if archive.parent.parent != managed_root:
            raise RuntimeError("scorer-contract archive escaped its managed root")
        _assert_no_scorer_package_symlink(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        if (not archive.is_file() or archive.read_bytes() != raw):
            raise RuntimeError("superseded scorer-contract archive collision")
        raise RuntimeError("predecessor exists at both active and archive paths")
    os.replace(path, archive)
    return "superseded_archived"


def _atomic_write_contract_output(
    path: Path, payload: dict[str, Any], *, managed_root: Path,
) -> None:
    """Create one no-follow exclusive temporary and atomically install it."""

    if path.parent != managed_root:
        raise RuntimeError("scorer-contract output escaped its managed root")
    _assert_no_scorer_package_symlink(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _assert_no_scorer_package_symlink(temporary)
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("scorer-contract temporary output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(temporary, flags, 0o644)
        created = True
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = None
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        created = False
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _contract_artifact_payload(
        source_launch_binding: dict[str, Any],
        state_selector_feasibility_receipt: dict[str, Any],
        mixed_precontract_disposition_receipt: dict[str, Any],
        interruption_receipt_binding: dict[str, Any],
) -> dict[str, Any]:
    """Build the post-verification artifact; pure for focused contract tests."""

    if (source_launch_binding.get("source_repository_clean") is not True
            or not source_launch_binding.get("source_repository_commit")):
        raise RuntimeError("contract artifact requires a clean-source launch binding")
    selection_digest = _digest(CORPUS_SELECTION_CONTRACT)
    frozen_feasibility = STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(
        root=ROOT
    )
    if state_selector_feasibility_receipt != frozen_feasibility:
        raise RuntimeError("contract feasibility input differs from frozen pass")
    feasibility_digest = state_selector_feasibility_receipt[
        "state_selector_feasibility_receipt_digest"
    ]
    STATE_SELECTOR.validate_preserved_state_mixed_precontract_disposition_receipt(
        mixed_precontract_disposition_receipt,
        expected_source_commit=source_launch_binding["source_repository_commit"],
        expected_successor_selection_digest=selection_digest,
        expected_clean_source_binding_digest=_digest(source_launch_binding),
        expected_bound_implementations_digest=
            source_launch_binding["bound_implementations_digest"],
        root=ROOT,
    )
    if (
        set(interruption_receipt_binding) != {
            "path", "receipt_digest", "raw_sha256", "byte_count", "status"
        }
        or interruption_receipt_binding.get("path")
        != str(INTERRUPTION.RECEIPT_RELATIVE_PATH)
        or interruption_receipt_binding.get("status") != INTERRUPTION.STATUS
        or any(
            not isinstance(interruption_receipt_binding.get(key), str)
            or len(interruption_receipt_binding[key]) != 64
            for key in ("receipt_digest", "raw_sha256")
        )
        or not isinstance(interruption_receipt_binding.get("byte_count"), int)
        or interruption_receipt_binding["byte_count"] <= 0
    ):
        raise RuntimeError("projection-fix interruption binding is invalid")
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
        "candidate_allocation_amendment_verified": True,
        "state_selector_amendment_verified": True,
        "state_selector_feasibility_verified": True,
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "preserved_state_mixed_precontract_disposition_verified": True,
        "mixed_precontract_disposition_receipt_digest":
            mixed_precontract_disposition_receipt[
                "mixed_precontract_disposition_receipt_digest"
            ],
        "retained_predecessor_state_count": 37,
        "rejected_predecessor_state_count": 8,
        "prospective_replacement_slot_count": 8,
        "preoutcome_projection_fix_interruption_verified": True,
        "preoutcome_projection_fix_interruption":
            dict(interruption_receipt_binding),
        "mixed_state_post_allocation_revalidation": {
            "status": "PENDING_POST_IDENTITY_PRE_OUTCOME",
            "required_before_active_identity_manifest": True,
            "schema": STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA,
            "path": STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
            "realized_receipt_digest_bound_at_contract_issue": False,
        },
        "preoutcome_allocation_failure_receipt_verified": True,
        "preoutcome_state_selection_failure_receipt_verified": True,
        "invalid_scorer_identity_exclusion_verified": True,
        "source_repository_commit":
            source_launch_binding["source_repository_commit"],
        "source_repository_clean": True,
        "clean_source_binding": source_launch_binding,
        "clean_source_binding_digest": _digest(source_launch_binding),
        "superseded_pre_run_contract_preservation":
            SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
        "superseded_graph_infeasible_contract_preservation":
            SUPERSEDED_GRAPH_INFEASIBLE_CONTRACT_ARTIFACT,
        "contract": contract(),
    }
    payload["contract_artifact_digest"] = _digest(payload)
    return payload


def issue_contract(path: Path) -> dict[str, Any]:
    """Validate external bindings and atomically issue the pre-outcome contract."""

    # Pin the sole permitted production destination before any active-output
    # probe.  Every later read/archive/write uses this canonical path, so an
    # alias swap cannot redirect issuance.
    path = _managed_scorer_contract_output_path(path)
    source_launch_binding = clean_source_binding()
    STATE_SELECTOR.validate_authority_artifacts()
    selection_digest = _digest(CORPUS_SELECTION_CONTRACT)
    # Both generated receipts are opened only by central custody guards.  In
    # particular, do not probe or parse the managed output alias before those
    # helpers have rejected any descendant or leaf symlink and pinned its
    # canonical target.
    feasibility_receipt = (
        STATE_SELECTOR.validate_frozen_reachability_feasibility_pass(root=ROOT)
    )
    disposition_receipt = (
        STATE_SELECTOR
        .load_and_validate_preserved_state_mixed_precontract_disposition_receipt(
        expected_source_commit=source_launch_binding["source_repository_commit"],
        expected_successor_selection_digest=selection_digest,
        expected_clean_source_binding_digest=_digest(source_launch_binding),
        expected_bound_implementations_digest=
            source_launch_binding["bound_implementations_digest"],
        root=ROOT,
        )
    )
    interruption_receipt = INTERRUPTION.load_and_validate_interruption_receipt(
        expected_source_repository_commit=source_launch_binding[
            "source_repository_commit"],
        expected_clean_source_binding_digest=_digest(source_launch_binding),
        expected_bound_implementations_digest=source_launch_binding[
            "bound_implementations_digest"],
        root=ROOT,
    )
    interruption_binding = INTERRUPTION.receipt_binding(
        interruption_receipt, root=ROOT)
    invalid_index = INVALID_IDS.load_invalid_identity_index()
    if (invalid_index.binding()["invalid_scorer_identity_exclusion_digest"]
            != INVALID_IDS.invalid_identity_exclusion_digest()):
        raise RuntimeError("invalid scorer-identity exclusion binding failed")

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

    amendment_path = ROOT / ALLOC.AMENDMENT_ARTIFACT_PATH
    amendment_artifact = json.loads(amendment_path.read_text())
    ALLOC.validate_allocation_amendment_artifact(amendment_artifact)
    failure_path = ROOT / ALLOC.FAILURE_RECEIPT_PATH
    if _stream_file_sha256(failure_path) != ALLOC.FAILURE_RECEIPT_RAW_SHA256:
        raise RuntimeError("pre-outcome allocation-failure receipt raw binding failed")
    failure_receipt = json.loads(failure_path.read_text())
    failure_digest = failure_receipt.pop("failure_receipt_digest", None)
    if (failure_digest != ALLOC.FAILURE_RECEIPT_DIGEST
            or _compact_digest(failure_receipt) != ALLOC.FAILURE_RECEIPT_DIGEST):
        raise RuntimeError("pre-outcome allocation-failure receipt self binding failed")

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
    if ALLOC.allocation_amendment_digest() != CORPUS_SELECTION_CONTRACT[
            "candidate_allocator_amendment_digest"]:
        raise RuntimeError("candidate-allocation amendment binding failed")
    payload = _contract_artifact_payload(
        source_launch_binding, feasibility_receipt, disposition_receipt,
        interruption_binding,
    )
    disposition = _prepare_contract_output(
        path, payload, managed_root=path.parent
    )
    if disposition == "current":
        return payload
    _atomic_write_contract_output(path, payload, managed_root=path.parent)
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
