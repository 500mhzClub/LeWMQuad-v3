#!/usr/bin/env python3
"""Predeclared claim analysis for the bounded WM-A branch experiment.

The evaluator uses requested action IDs as candidate inputs.  Future executed
command tapes remain physical target/audit fields and are never passed to the
model.  Direct branch fidelity is measured in train-standardized target-encoder
space before the separate train-only physical utility readout is evaluated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    ACTION_COUNT,
    load_bound_pilot_v1,
    read_bound_rgb_bytes_v1,
)
from lewm.benchmarks.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    FAMILIES,
)
from scripts import evaluate_go2_world_model_counterfactual_action_regret_v1 as base  # noqa: E402
from scripts import analyze_go2_world_model_progression_v1 as progression_analyzer  # noqa: E402


REPORT_SCHEMA = "lewm_go2_world_model_bounded_branch_experiment_result_v1"
PANEL_REPORT_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_experiment_panel_result_v1"
)
EVALUATION_CONTRACT_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_evaluation_contract_v1"
)
DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260802
RIDGE_LAMBDA = 1.0e-3
EXPECTED_TERMINAL_UPDATE = 700
CHANCE_RETRIEVAL = 1.0 / ACTION_COUNT
MINIMUM_DISCRIMINATION_COVERAGE = 0.25
PROGRESSION_SNAPSHOT_SCHEMA = "lewm_go2_world_model_progression_v1_snapshot_v1"
PROGRESSION_ANALYSIS_SCHEMA = progression_analyzer.SCHEMA
MODEL_ARMS = ("masked_plain", "masked_delta", "full_plain", "full_delta")
TRAINING_SEEDS = (2026080201, 2026080202, 2026080203)
PRIMARY_PLAIN_ARMS = ("masked_plain", "full_plain")
MECHANISM_CONTROL_ARMS = ("masked_delta", "full_delta")
PLAIN_FAMILY_TWO_SIDED_ALPHA = 0.025
CONTROL_TWO_SIDED_ALPHA = 0.05
LATENT_STANDARDIZER_REFERENCE_ARM = "masked_plain"
LATENT_STANDARDIZER_REFERENCE_SEED = TRAINING_SEEDS[0]
PROGRESSION_OUTPUT_PARENT = REPO_ROOT / ".generated/dev/world_model_progression_v1"
PROGRESSION_PACK_ROOT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3"
    / "attempt_v1/pack"
)
PROGRESSION_PREDECESSOR = (
    REPO_ROOT
    / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    / "attempt_v1/snapshots/update_1000.pt"
)
PREDECESSOR_TERMINAL_ACCESS = (
    REPO_ROOT
    / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    / "attempt_v1/receipts/terminal_access.json"
)
PREDECESSOR_TERMINAL_ACCESS_BYTE_COUNT = 5_593
PREDECESSOR_TERMINAL_ACCESS_SHA256 = (
    "7d828a6e16c0ac50c85f5b94760cb3dee5058af7a9bcc6361e98e3388416631c"
)
PREDECESSOR_PLACE_INDEX = (
    REPO_ROOT
    / ".generated/go2_memory_role_place_triplet_index_v1/"
    "checkpoint_selection.jsonl"
)
PREDECESSOR_PLACE_INDEX_BYTE_COUNT = 473_508
PREDECESSOR_PLACE_INDEX_SHA256 = (
    "a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0"
)
PREDECESSOR_PLACE_MANIFEST = (
    REPO_ROOT / ".generated/go2_memory_role_place_triplet_index_v1/manifest.json"
)
PREDECESSOR_PLACE_MANIFEST_BYTE_COUNT = 42_308
PREDECESSOR_PLACE_MANIFEST_SHA256 = (
    "a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca"
)
PILOT_TERMINAL_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_experiment_supervision_terminal_v1"
)
PILOT_TERMINAL_REVIEW_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_experiment_terminal_review_v1"
)
USEFUL_EFFECT_THRESHOLDS = {
    "ceiling_rank_regret_reduction_vs_current": 0.05,
    "direct_error_reduction_vs_shuffled": 0.02,
    "direct_margin_gain_vs_shuffled": 0.02,
    "retrieval_advantage_over_chance": 0.05,
    "rank_regret_reduction_vs_current": 0.05,
    "maximum_unsafe_rate_increase_vs_current": 0.02,
    "target_progress_gain_vs_current_m": 0.01,
    "maximum_absolute_unsafe_rate": 0.05,
    "minimum_absolute_target_progress_m": 0.01,
}
MECHANISM_EFFECT_THRESHOLDS = {
    "minimum_direct_error_treatment_reduction": 0.02,
    "full_grid_direct_error_noninferiority_margin": 0.02,
    "minimum_supportive_physical_regret_treatment_reduction": 0.02,
}


class BoundedBranchEvaluationError(RuntimeError):
    """Raised before a malformed comparison can produce a scientific verdict."""


def checkpoint_interval_alpha_v1(arm: str) -> float:
    if arm in PRIMARY_PLAIN_ARMS:
        return PLAIN_FAMILY_TWO_SIDED_ALPHA
    if arm in MECHANISM_CONTROL_ARMS:
        return CONTROL_TWO_SIDED_ALPHA
    raise BoundedBranchEvaluationError("checkpoint arm is outside the fixed panel")


def _reject_protected_path(path: Path, *, label: str) -> None:
    for part in Path(path).parts:
        lowered = part.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
        ):
            raise BoundedBranchEvaluationError(f"{label} names protected material")


def _require_nofollow_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    """Reject protected names and symlinks in every existing path component."""

    selected = Path(os.path.abspath(os.fspath(path)))
    _reject_protected_path(selected, label=label)
    parts = selected.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        if current.is_symlink():
            raise BoundedBranchEvaluationError(
                f"{label} traverses a symlink component"
            )
        if not current.exists():
            if must_exist:
                raise BoundedBranchEvaluationError(f"{label} is absent")
            break
    if must_exist and not selected.exists():
        raise BoundedBranchEvaluationError(f"{label} is absent")
    return selected


def evaluation_contract_v1() -> dict[str, Any]:
    """Return the immutable primary/falsification metric and gate contract."""

    return {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "status": "PREREGISTERED_SOURCE_ONLY",
        "candidate_model_input": "requested_action_id",
        "future_executed_command_tape_usage": "target_and_audit_only",
        "roles": {
            "train": "readout_and_latent_standardization_only",
            "eval": "all_reported_generalization_claims",
        },
        "direct_latent": {
            "representation": "four_fixed_masks_target_encoder_mean_std_descriptor",
            "normalization": (
                "per_dimension_train_true_future_mean_std_from_one_fixed_"
                "masked_plain_seed_2026080201_reference_for_all_12_members"
            ),
            "distance": "root_mean_squared_standardized_euclidean",
            "always_primary": ["matched_branch_error"],
            "separable_outcome_primary": [
                "nearest_nonequivalent_minus_equivalent_margin",
                "equivalence_aware_action_retrieval_advantage",
            ],
            "outcome_equivalence": {
                "basis": (
                    "equal_frozen_calibration_tolerance_aware_physical_oracle_dense_rank"
                ),
                "model_dependent": False,
                "latent_proximity_used_for_equivalence": False,
                "minimum_discrimination_coverage": (
                    MINIMUM_DISCRIMINATION_COVERAGE
                ),
                "query_eligibility": (
                    "at_least_one_physically_nonequivalent_alternative_and_"
                    "every_physically_nonequivalent_alternative_has_different_"
                    "executed_tape_and_stored_rgb_raw_pixel_identity"
                ),
                "family_requirement": (
                    "eligible_query_coverage_at_least_0.25_in_each_fixed_family"
                ),
                "scene_requirement": (
                    "both_fixed_evaluation_scenes_in_each_family_have_at_"
                    "least_one_eligible_query"
                ),
                "below_coverage": "INCONCLUSIVE_DATA",
            },
            "collapse_diagnostic": (
                "jointly_eligible_action_state_and_scene_coverage"
            ),
        },
        "physical_utility": {
            "primary": "normalized_dense_physical_rank_regret",
            "safety": "fell_or_tipped",
            "progress": "physical_target_progress_m",
            "train_only_readout": "nine_action_specific_ridge_heads",
        },
        "uncertainty": {
            "unit": "scene_within_fixed_family_stratum",
            "paired": True,
            "family_weights": "fixed_equal_eight_family_weights",
            "bootstrap_resamples": DEFAULT_RESAMPLES,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "readout": {
            "ridge_lambda": RIDGE_LAMBDA,
            "action_specific_heads": ACTION_COUNT,
            "fit_role": "train",
        },
        "model_panel": {
            "arms": list(MODEL_ARMS),
            "training_seeds": list(TRAINING_SEEDS),
            "checkpoints": len(MODEL_ARMS) * len(TRAINING_SEEDS),
            "primary_arm_family": list(PRIMARY_PLAIN_ARMS),
            "checkpoint_selection_from_branch_eval": False,
            "fixed_terminal_within_seed_factorial_comparisons": True,
            "common_latent_standardizer_reference": {
                "arm": LATENT_STANDARDIZER_REFERENCE_ARM,
                "seed": LATENT_STANDARDIZER_REFERENCE_SEED,
            },
            "all_panel_members_reported": True,
            "confirmatory_primary_decision": (
                "any_plain_arm_clears_all_gates_all_three_seeds"
            ),
            "plain_arm_family_two_sided_alpha_each": (
                PLAIN_FAMILY_TWO_SIDED_ALPHA
            ),
            "plain_arm_family_multiplicity": (
                "bonferroni_two_arm_family_one_sided_family_alpha_0.025"
            ),
            "delta_arms": list(MECHANISM_CONTROL_ARMS),
            "delta_arm_role_when_proxy_not_meaningful": (
                "negative_and_mechanism_controls_only"
            ),
        },
        "mechanism_adjudication": {
            "paired_cross_model_unit": "scene_within_fixed_family_stratum",
            "all_three_seed_rule": True,
            "thresholds": dict(MECHANISM_EFFECT_THRESHOLDS),
            "delta_proxy_routing_is_binding": True,
            "full_grid_rollout_requires_masked_noninferiority": True,
            "mechanism_and_usefulness_decisions_are_separate": True,
        },
        "visual_domain_prerequisite": {
            "required_before_generation": True,
            "independent_review_required": True,
            "reference_domain": "textured_world_model_training_rgb",
            "candidate_domain": "counterfactual_branch_renderer_rgb",
            "reference_renderer_source": "scripts/render_replay_v03.py",
            "reference_native_resolution": [224, 224],
            "reference_genesis_yfov_deg": 78.323,
            "requires_same_build_scene_texture_helpers_and_default_scene_options": True,
            "native_640x480_then_downsample_or_horizontal_to_vertical_fov_conversion_is_not_parity": True,
            "solid_material_or_uniform_gray_substitute_is_ineligible": True,
            "missing_or_failed_parity_evidence": "STOP_NO_GENERATION_AUTHORITY",
        },
        "thresholds": dict(USEFUL_EFFECT_THRESHOLDS),
        "per_checkpoint_output": "measurement_only_no_scientific_verdict",
        "verdict_rule": (
            "arm_agnostic_plain_family_usefulness_only_after_all_12_fixed_members_"
            "reported_and_at_least_one_bonferroni_controlled_plain_arm_passes_"
            "all_gates_in_all_three_seeds;_delta_cells_are_controls;_below_"
            "physical_action_separability_coverage_routes_INCONCLUSIVE_DATA"
        ),
        "claim_scope": "scene_disjoint_development_planning_evidence_only",
        "does_not_authorize": [
            "held_out_or_sealed_access",
            "deployment",
            "promotion",
            "checkpoint_promotion_or_planning_eligibility",
            "automatic_rollout_from_development_evidence",
            "retry",
            "resume",
            "hyperparameter_adaptation",
        ],
    }


def _latent_prefix(feature: object) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float64).reshape(-1)
    if vector.size <= 3 or (vector.size - 3) % 3:
        raise BoundedBranchEvaluationError("task-conditioned latent shape changed")
    latent_size = (vector.size - 3) // 3
    latent = vector[:latent_size]
    if latent_size <= 0 or not np.isfinite(latent).all():
        raise BoundedBranchEvaluationError("latent descriptor is invalid")
    return latent


def validate_progression_snapshot_metadata_v1(
    payload: object,
    *,
    expected_arm: str,
    expected_seed: int,
    expected_update: int,
) -> dict[str, Any]:
    if expected_arm not in MODEL_ARMS or expected_seed not in TRAINING_SEEDS:
        raise BoundedBranchEvaluationError("checkpoint is outside the preregistered model panel")
    if not isinstance(payload, Mapping):
        raise BoundedBranchEvaluationError("progression checkpoint payload is malformed")
    required = {
        "schema",
        "status",
        "development_only",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "arm",
        "seed",
        "update",
        "full_grid_training",
        "action_auxiliary_weight",
        "metrics",
        "arm_state_dict",
        "decoder_state_dict",
    }
    if set(payload) != required:
        raise BoundedBranchEvaluationError("progression checkpoint field set changed")
    is_delta = expected_arm.endswith("_delta")
    weight = payload.get("action_auxiliary_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise BoundedBranchEvaluationError("progression checkpoint auxiliary weight changed")
    if (
        payload.get("schema") != PROGRESSION_SNAPSHOT_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("development_only") is not True
        or payload.get("citable_as_scientific_evidence") is not False
        or payload.get("authorizes_retry_or_resume") is not False
        or payload.get("arm") != expected_arm
        or payload.get("seed") != expected_seed
        or payload.get("update") != expected_update
        or payload.get("full_grid_training") is not expected_arm.startswith("full_")
        or float(weight) != (0.1 if is_delta else 0.0)
        or not isinstance(payload.get("metrics"), Mapping)
        or not isinstance(payload.get("arm_state_dict"), Mapping)
        or (is_delta and not isinstance(payload.get("decoder_state_dict"), Mapping))
        or (not is_delta and payload.get("decoder_state_dict") is not None)
    ):
        raise BoundedBranchEvaluationError("progression checkpoint identity changed")
    return {
        "schema": PROGRESSION_SNAPSHOT_SCHEMA,
        "arm": expected_arm,
        "seed": expected_seed,
        "update": expected_update,
        "full_grid_training": expected_arm.startswith("full_"),
        "action_auxiliary_weight": 0.1 if is_delta else 0.0,
    }


def _sha_binding(path: Path, *, label: str) -> dict[str, Any]:
    selected = _require_nofollow_path(path, label=label)
    if not selected.is_file():
        raise BoundedBranchEvaluationError(f"{label} is not a regular file")
    raw = selected.read_bytes()
    return {
        "path": str(selected.resolve(strict=True)),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_predecessor_observational_scenes_v1(
    binding: object,
) -> tuple[set[str], dict[str, Any]]:
    """Reopen the frozen encoder's terminal provenance and bound scene indices."""

    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "byte_count",
        "sha256",
    }:
        raise BoundedBranchEvaluationError(
            "predecessor terminal-access binding changed"
        )
    path = _require_nofollow_path(
        Path(str(binding["path"])), label="predecessor terminal access"
    )
    actual = _sha_binding(path, label="predecessor terminal access")
    if actual != dict(binding):
        raise BoundedBranchEvaluationError(
            "predecessor terminal-access binding changed"
        )
    try:
        receipt = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedBranchEvaluationError(
            "predecessor terminal access is not strict JSON"
        ) from exc
    required = {
        "action_tensor_count",
        "checkpoint_deserialize_count_after_initialization",
        "content_sha256",
        "future_rgb_tensor_count",
        "h6_training_current_rgb_presentations",
        "held_out_or_sealed_opened",
        "n320",
        "navigation_executed",
        "observations",
        "place_rgb_presentations",
        "runtime_inputs",
        "schema",
    }
    runtime_inputs = receipt.get("runtime_inputs") if isinstance(receipt, Mapping) else None
    audit = runtime_inputs.get("audit") if isinstance(runtime_inputs, Mapping) else None
    checks = runtime_inputs.get("checks") if isinstance(runtime_inputs, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != required
        or receipt.get("schema")
        != "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_terminal_access_receipt_v1"
        or receipt.get("held_out_or_sealed_opened") is not False
        or receipt.get("navigation_executed") is not False
        or receipt.get("action_tensor_count") != 0
        or receipt.get("future_rgb_tensor_count") != 0
        or not isinstance(runtime_inputs, Mapping)
        or set(runtime_inputs) != {"audit", "checks", "passed"}
        or runtime_inputs.get("passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks)
        != {
            "no_actions",
            "no_future_rgb",
            "place_index",
            "place_manifest",
            "train_index",
            "validation_index",
        }
        or any(value is not True for value in checks.values())
        or not isinstance(audit, Mapping)
        or set(audit)
        != {
            "action_tensor_count",
            "future_rgb_tensor_count",
            "place",
            "train",
            "validation",
        }
    ):
        raise BoundedBranchEvaluationError(
            "predecessor terminal-access evidence changed"
        )
    scenes: set[str] = set()
    index_bindings: dict[str, Any] = {}
    place = audit["place"]
    expected_place_audit = {
        "index_file_sha256": PREDECESSOR_PLACE_INDEX_SHA256,
        "manifest_file_sha256": PREDECESSOR_PLACE_MANIFEST_SHA256,
        "privileged_label_fields_emitted_to_model": 0,
        "rgb_open_count": 0,
        "role": "checkpoint_selection",
        "row_count": 320,
        "schema": "lewm_go2_memory_role_place_triplet_index_v1",
    }
    if place != expected_place_audit:
        raise BoundedBranchEvaluationError(
            "predecessor place checkpoint-selection audit changed"
        )
    place_manifest_binding = _sha_binding(
        PREDECESSOR_PLACE_MANIFEST, label="predecessor place manifest"
    )
    place_index_binding = _sha_binding(
        PREDECESSOR_PLACE_INDEX, label="predecessor place index"
    )
    if place_manifest_binding != {
        "path": str(PREDECESSOR_PLACE_MANIFEST.resolve(strict=True)),
        "byte_count": PREDECESSOR_PLACE_MANIFEST_BYTE_COUNT,
        "sha256": PREDECESSOR_PLACE_MANIFEST_SHA256,
    } or place_index_binding != {
        "path": str(PREDECESSOR_PLACE_INDEX.resolve(strict=True)),
        "byte_count": PREDECESSOR_PLACE_INDEX_BYTE_COUNT,
        "sha256": PREDECESSOR_PLACE_INDEX_SHA256,
    }:
        raise BoundedBranchEvaluationError("predecessor place binding changed")
    try:
        place_manifest = json.loads(PREDECESSOR_PLACE_MANIFEST.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedBranchEvaluationError(
            "predecessor place manifest is not strict JSON"
        ) from exc
    artifact = (
        place_manifest.get("artifacts", {}).get("checkpoint_selection.jsonl")
        if isinstance(place_manifest, Mapping) else None
    )
    if (
        not isinstance(place_manifest, Mapping)
        or place_manifest.get("schema")
        != "lewm_go2_memory_role_place_triplet_index_manifest_v1"
        or place_manifest.get("status") != "PASS"
        or artifact != {
            "byte_count": PREDECESSOR_PLACE_INDEX_BYTE_COUNT,
            "path": "checkpoint_selection.jsonl",
            "row_count": 320,
            "sha256": PREDECESSOR_PLACE_INDEX_SHA256,
        }
    ):
        raise BoundedBranchEvaluationError(
            "predecessor place manifest contract changed"
        )
    place_lines = PREDECESSOR_PLACE_INDEX.read_bytes().splitlines()
    if len(place_lines) != 320:
        raise BoundedBranchEvaluationError("predecessor place row count changed")
    for line in place_lines:
        try:
            item = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BoundedBranchEvaluationError(
                "predecessor place index contains malformed JSON"
            ) from exc
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "anchor", "content_sha256", "family", "negative",
                "positive", "role", "scene_id", "schema", "selection_proof",
            }
            or item.get("schema")
            != "lewm_go2_memory_role_place_triplet_index_v1"
            or item.get("role") != "checkpoint_selection"
            or item.get("family") not in FAMILIES
            or not isinstance(item.get("scene_id"), str)
            or not item["scene_id"]
        ):
            raise BoundedBranchEvaluationError(
                "predecessor place scene identity changed"
            )
        scenes.add(str(item["scene_id"]))
    index_bindings["place"] = place_index_binding
    for receipt_role, expected_role in (("train", "train"), ("validation", "val")):
        row = audit[receipt_role]
        if not isinstance(row, Mapping):
            raise BoundedBranchEvaluationError("predecessor index audit is malformed")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or row.get("role") != expected_role
            or type(row.get("byte_count")) is not int
            or not isinstance(row.get("file_sha256"), str)
        ):
            raise BoundedBranchEvaluationError("predecessor index binding changed")
        index_path = _require_nofollow_path(
            REPO_ROOT / relative,
            label=f"predecessor {receipt_role} index",
        )
        raw = index_path.read_bytes()
        if (
            len(raw) != int(row["byte_count"])
            or hashlib.sha256(raw).hexdigest() != row["file_sha256"]
        ):
            raise BoundedBranchEvaluationError("predecessor index bytes changed")
        lines = raw.splitlines()
        if len(lines) != int(row.get("row_count", -1)):
            raise BoundedBranchEvaluationError("predecessor index row count changed")
        for line in lines:
            try:
                item = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BoundedBranchEvaluationError(
                    "predecessor index contains malformed JSON"
                ) from exc
            if (
                not isinstance(item, Mapping)
                or item.get("role") != expected_role
                or not isinstance(item.get("scene_id"), str)
                or not item["scene_id"]
            ):
                raise BoundedBranchEvaluationError(
                    "predecessor index scene identity changed"
                )
            scenes.add(str(item["scene_id"]))
        index_bindings[receipt_role] = {
            "path": str(index_path),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return scenes, {
        "predecessor_terminal_access_binding": actual,
        "predecessor_index_bindings": index_bindings,
        "predecessor_place_manifest_binding": place_manifest_binding,
        "predecessor_observational_scene_count": len(scenes),
    }


def load_and_validate_progression_analysis_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    selected_checkpoint: Path,
    expected_arm: str,
    expected_seed: int,
    pilot_scene_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the payload-free analysis and freeze its exact 12 snapshots.

    The committed progression runner intentionally does not embed snapshot
    receipts.  The separately preregistered offline analyzer hashes the fixed
    2x2x3 terminal panel without deserializing tensors.  This consumer reruns
    that analyzer, rehashes every upstream source/input/provenance object, and
    rejects any disagreement before a branch scene can be selected.
    """

    if expected_arm not in MODEL_ARMS or expected_seed not in TRAINING_SEEDS:
        raise BoundedBranchEvaluationError(
            "checkpoint is outside the preregistered progression panel"
        )
    path = _require_nofollow_path(path, label="progression analysis")
    selected_checkpoint = _require_nofollow_path(
        selected_checkpoint, label="selected checkpoint"
    )
    from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as contract
    from scripts import dev_train_temporal_jepa_scaled as scaled

    analysis, analysis_binding = contract.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="world-model progression analysis",
    )
    required_analysis = {
        "schema",
        "status",
        "development_only",
        "citable_as_world_model_usefulness_evidence",
        "input_result",
        "configuration",
        "decoder_anchor_by_seed",
        "contrasts",
        "proxy_routing",
        "terminal_snapshot_bindings",
        "uncertainty_limit",
    }
    input_result = analysis.get("input_result") if isinstance(analysis, Mapping) else None
    snapshot_receipts = (
        analysis.get("terminal_snapshot_bindings")
        if isinstance(analysis, Mapping)
        else None
    )
    if (
        not isinstance(analysis, Mapping)
        or set(analysis) != required_analysis
        or analysis.get("schema") != PROGRESSION_ANALYSIS_SCHEMA
        or analysis.get("status") != "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS"
        or analysis.get("development_only") is not True
        or analysis.get("citable_as_world_model_usefulness_evidence") is not False
        or analysis.get("configuration") != progression_analyzer.EXPECTED_CONFIGURATION
        or not isinstance(input_result, Mapping)
        or set(input_result) != {"path", "byte_count", "sha256"}
        or not isinstance(snapshot_receipts, Mapping)
        or set(snapshot_receipts) != {str(seed) for seed in TRAINING_SEEDS}
    ):
        raise BoundedBranchEvaluationError("progression analysis identity changed")
    result_path = _require_nofollow_path(
        Path(str(input_result["path"])), label="progression result"
    )
    output_root = result_path.parent
    if (
        path.name != "analysis.json"
        or result_path.name != "result.json"
        or path.parent.resolve() != output_root.resolve()
        or output_root.is_symlink()
        or output_root.resolve().parent != PROGRESSION_OUTPUT_PARENT.resolve()
    ):
        raise BoundedBranchEvaluationError(
            "progression analysis/result is outside the fixed output contract"
        )
    if (
        type(input_result.get("byte_count")) is not int
        or int(input_result["byte_count"]) <= 0
        or not isinstance(input_result.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(input_result["sha256"])) is None
    ):
        raise BoundedBranchEvaluationError("progression result binding is malformed")
    document, result_binding = contract.read_bound_json(
        result_path,
        expected_sha256=str(input_result["sha256"]),
        expected_byte_count=int(input_result["byte_count"]),
        label="world-model progression result",
    )
    normalized_result_binding = {
        "path": result_binding["path"],
        "byte_count": result_binding["byte_count"],
        "sha256": result_binding["file_sha256"],
    }
    if normalized_result_binding != dict(input_result):
        raise BoundedBranchEvaluationError("analysis input-result binding changed")
    try:
        regenerated_analysis = progression_analyzer.analyze(
            document, result_path=result_path
        )
    except progression_analyzer.AnalysisError as exc:
        raise BoundedBranchEvaluationError(
            f"progression offline analysis no longer passes: {exc}"
        ) from exc
    if regenerated_analysis != dict(analysis):
        raise BoundedBranchEvaluationError(
            "progression analysis disagrees with an independent recomputation"
        )

    required_document = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "protected_material_opened",
        "configuration",
        "runtime",
        "inputs",
        "source_bindings",
        "seed_results",
    }
    inputs = document.get("inputs") if isinstance(document, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or set(document) != required_document
        or document.get("schema") != progression_analyzer.RUNNER_SCHEMA
        or document.get("status") != progression_analyzer.RUNNER_STATUS
        or document.get("citable_as_scientific_evidence") is not False
        or document.get("protected_material_opened") is not False
        or document.get("configuration") != progression_analyzer.EXPECTED_CONFIGURATION
        or not isinstance(inputs, Mapping)
        or set(inputs) != {"predecessor", "pack_root", "train", "val"}
    ):
        raise BoundedBranchEvaluationError("progression result identity changed")
    expected_checkpoint = (
        output_root
        / f"seed_{expected_seed}"
        / f"{expected_arm}_update_{EXPECTED_TERMINAL_UPDATE:06d}.pt"
    ).resolve()
    if selected_checkpoint.resolve() != expected_checkpoint:
        raise BoundedBranchEvaluationError(
            "checkpoint is not the bound fixed-terminal panel member"
        )

    expected_predecessor = {
        "path": str(PROGRESSION_PREDECESSOR.resolve()),
        "byte_count": progression_analyzer.EXPECTED_PREDECESSOR["byte_count"],
        "sha256": progression_analyzer.EXPECTED_PREDECESSOR["sha256"],
    }
    if inputs["predecessor"] != expected_predecessor:
        raise BoundedBranchEvaluationError("progression predecessor binding changed")
    if _sha_binding(
        PROGRESSION_PREDECESSOR, label="progression predecessor"
    ) != expected_predecessor:
        raise BoundedBranchEvaluationError("progression predecessor bytes changed")
    terminal_access_binding = {
        "path": str(PREDECESSOR_TERMINAL_ACCESS.resolve()),
        "byte_count": PREDECESSOR_TERMINAL_ACCESS_BYTE_COUNT,
        "sha256": PREDECESSOR_TERMINAL_ACCESS_SHA256,
    }
    ancestor_scenes, ancestor_receipt = _read_predecessor_observational_scenes_v1(
        terminal_access_binding
    )

    pack_root_value = inputs["pack_root"]
    if not isinstance(pack_root_value, str):
        raise BoundedBranchEvaluationError("progression pack root is absent")
    pack_root = _require_nofollow_path(
        Path(pack_root_value), label="progression pack root"
    )
    if pack_root.resolve() != PROGRESSION_PACK_ROOT.resolve():
        raise BoundedBranchEvaluationError("progression pack root changed")

    def canonical_pack_manifest_path(value: object, *, role: str) -> str:
        if not isinstance(value, (str, os.PathLike)):
            raise BoundedBranchEvaluationError(
                f"progression {role} pack manifest path is malformed"
            )
        selected = Path(value)
        if not selected.is_absolute():
            selected = REPO_ROOT / selected
        return str(
            _require_nofollow_path(
                selected, label=f"progression {role} pack manifest"
            ).resolve(strict=True)
        )

    def current_pack_role(role: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
        validated = scaled.validate_pack_role(pack_root, role)
        current_role = validated["role"]
        return {
            "manifest_path": canonical_pack_manifest_path(
                validated["manifest_path"], role=role
            ),
            "manifest_sha256": validated["manifest_sha256"],
            "role": role,
            "row_identity_sha256": current_role["row_identity_sha256"],
            "source_rgb": current_role["source_rgb"],
            "index_binding": current_role["index_binding"],
            "frames": current_role["frames"],
            "actions": current_role["actions"],
            "metadata": current_role["metadata"],
        }, validated

    observational_scenes: set[str] = set(ancestor_scenes)
    metadata_bindings: dict[str, Any] = {}
    pack_role_bindings: dict[str, Any] = {}
    for role in ("train", "val"):
        current_binding, validated = current_pack_role(role)
        declared_binding = inputs[role]
        if not isinstance(declared_binding, Mapping):
            raise BoundedBranchEvaluationError(
                f"progression {role} pack binding changed"
            )
        normalized_declared_binding = dict(declared_binding)
        normalized_declared_binding["manifest_path"] = (
            canonical_pack_manifest_path(
                normalized_declared_binding.get("manifest_path"), role=role
            )
        )
        if normalized_declared_binding != current_binding:
            raise BoundedBranchEvaluationError(
                f"progression {role} pack binding changed"
            )
        pack_role_bindings[role] = current_binding
        metadata = validated["role"]["metadata"]
        metadata_path = _require_nofollow_path(
            validated["paths"]["metadata"],
            label=f"progression {role} metadata",
        )
        raw = metadata_path.read_bytes()
        if (
            len(raw) != int(metadata["byte_count"])
            or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
        ):
            raise BoundedBranchEvaluationError("progression metadata binding changed")
        try:
            metadata_document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BoundedBranchEvaluationError("progression metadata is not JSON") from exc
        if (
            not isinstance(metadata_document, Mapping)
            or set(metadata_document) != {"scene_ids", "families"}
            or not isinstance(metadata_document["scene_ids"], list)
            or any(
                not isinstance(item, str) or not item
                for item in metadata_document["scene_ids"]
            )
        ):
            raise BoundedBranchEvaluationError("progression scene metadata changed")
        observational_scenes.update(metadata_document["scene_ids"])
        metadata_bindings[role] = {
            "path": str(metadata_path.resolve()),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    expected_sources = list(progression_analyzer.EXPECTED_SOURCE_BINDINGS)
    if document["source_bindings"] != expected_sources:
        raise BoundedBranchEvaluationError(
            "progression training source receipt changed"
        )

    def rehash_progression_sources() -> None:
        for expected in expected_sources:
            source_path = _require_nofollow_path(
                REPO_ROOT / str(expected["path"]),
                label="progression training source",
            )
            actual = _sha_binding(source_path, label="progression training source")
            normalized = {
                "path": source_path.relative_to(REPO_ROOT).as_posix(),
                "byte_count": actual["byte_count"],
                "sha256": actual["sha256"],
            }
            if normalized != expected:
                raise BoundedBranchEvaluationError(
                    "progression training source bytes changed"
                )

    rehash_progression_sources()
    seed_results = document["seed_results"]
    if not isinstance(seed_results, Mapping) or set(seed_results) != {
        str(seed) for seed in TRAINING_SEEDS
    }:
        raise BoundedBranchEvaluationError("progression seed panel changed")
    required_seed_result = {
        "build",
        "decoder_pretraining_trace",
        "decoder_anchor_balanced_accuracy",
        "update_zero",
        "terminal",
        "terminal_losses",
        "training_trace",
        "terminal_core_sha256",
        "terminal_decoder_sha256",
        "wall_seconds",
    }
    checkpoint_panel_bindings: dict[str, dict[str, Any]] = {}
    for seed in TRAINING_SEEDS:
        seed_result = seed_results[str(seed)]
        seed_snapshots = snapshot_receipts[str(seed)]
        if (
            not isinstance(seed_result, Mapping)
            or set(seed_result) != required_seed_result
            or not isinstance(seed_snapshots, Mapping)
            or set(seed_snapshots) != set(MODEL_ARMS)
        ):
            raise BoundedBranchEvaluationError(
                "progression fixed-terminal snapshot panel changed"
            )
        for arm in MODEL_ARMS:
            expected_path = (
                output_root
                / f"seed_{seed}"
                / f"{arm}_update_{EXPECTED_TERMINAL_UPDATE:06d}.pt"
            )
            actual_snapshot = _sha_binding(
                expected_path,
                label=f"progression snapshot {arm}/{seed}",
            )
            if seed_snapshots[arm] != actual_snapshot:
                raise BoundedBranchEvaluationError(
                    "progression analysis snapshot binding changed"
                )
            checkpoint_panel_bindings[f"{arm}/seed_{seed}"] = actual_snapshot
    selected_key = f"{expected_arm}/seed_{expected_seed}"
    if checkpoint_panel_bindings[selected_key]["path"] != str(
        selected_checkpoint.resolve()
    ):
        raise BoundedBranchEvaluationError(
            "selected checkpoint is not its analysis-bound panel member"
        )
    overlap = observational_scenes & pilot_scene_ids
    if overlap:
        raise BoundedBranchEvaluationError(
            "checkpoint training/validation scenes overlap frozen branch scenes"
        )

    # Terminal rehash: reject any source/input/snapshot race during validation.
    if contract.file_binding(result_path) != result_binding:
        raise BoundedBranchEvaluationError("progression result changed during validation")
    if contract.file_binding(path) != analysis_binding:
        raise BoundedBranchEvaluationError("progression analysis changed during validation")
    if _sha_binding(
        PROGRESSION_PREDECESSOR, label="progression predecessor"
    ) != expected_predecessor:
        raise BoundedBranchEvaluationError("progression predecessor changed during validation")
    if _sha_binding(
        PREDECESSOR_TERMINAL_ACCESS, label="predecessor terminal access"
    ) != terminal_access_binding:
        raise BoundedBranchEvaluationError(
            "predecessor provenance changed during validation"
        )
    for role, binding in ancestor_receipt["predecessor_index_bindings"].items():
        if _sha_binding(
            Path(str(binding["path"])), label=f"predecessor {role} index"
        ) != binding:
            raise BoundedBranchEvaluationError(
                "predecessor observational index changed during validation"
            )
    if _sha_binding(
        Path(str(ancestor_receipt["predecessor_place_manifest_binding"]["path"])),
        label="predecessor place manifest",
    ) != ancestor_receipt["predecessor_place_manifest_binding"]:
        raise BoundedBranchEvaluationError(
            "predecessor place manifest changed during validation"
        )
    for role in ("train", "val"):
        current_binding, _validated = current_pack_role(role)
        if current_binding != pack_role_bindings[role]:
            raise BoundedBranchEvaluationError(
                f"progression {role} pack changed during validation"
            )
    for key, binding in checkpoint_panel_bindings.items():
        if _sha_binding(
            Path(str(binding["path"])), label=f"progression snapshot {key}"
        ) != binding:
            raise BoundedBranchEvaluationError(
                "progression snapshot changed during validation"
            )
    rehash_progression_sources()
    return dict(analysis), {
        "progression_analysis_binding": analysis_binding,
        "training_result_binding": result_binding,
        "progression_proxy_routing": dict(analysis["proxy_routing"]),
        "training_pack_role_bindings": pack_role_bindings,
        "training_pack_metadata_bindings": metadata_bindings,
        **ancestor_receipt,
        "checkpoint_panel_bindings": checkpoint_panel_bindings,
        "observational_scene_count": len(observational_scenes),
        "observational_scene_ids": sorted(observational_scenes),
        "pilot_scene_count": len(pilot_scene_ids),
        "scene_overlap": [],
    }


def _validate_reservation_document_v1(
    value: object,
    *,
    attempt: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    supervisor_nonce: object,
    supervisor_pid: object,
) -> dict[str, Any]:
    required = {
        "schema", "status", "attempt", "plan_binding", "authority_binding",
        "supervisor_nonce", "supervisor_pid", "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt", "retry_authorized",
        "resume_authorized", "overwrite_authorized", "refill_authorized",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema")
        != "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
        or value.get("status") != "RESERVED_ATTEMPT_CONSUMED"
        or value.get("attempt") != dict(attempt)
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("authority_binding") != dict(authority_binding)
        or not isinstance(supervisor_nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", supervisor_nonce) is None
        or value.get("supervisor_nonce") != supervisor_nonce
        or type(supervisor_pid) is not int
        or supervisor_pid <= 1
        or value.get("supervisor_pid") != supervisor_pid
        or value.get("root_creation_consumes_attempt") is not True
        or value.get("reservation_records_consumed_attempt") is not True
        or any(
            value.get(field) is not False
            for field in (
                "retry_authorized", "resume_authorized",
                "overwrite_authorized", "refill_authorized",
            )
        )
    ):
        raise BoundedBranchEvaluationError(
            "bounded pilot reservation ownership/content changed"
        )
    return dict(value)


def _require_model_panel_lineage_match_v1(
    freeze: object,
    separation: object,
) -> None:
    if not isinstance(freeze, Mapping) or not isinstance(separation, Mapping):
        raise BoundedBranchEvaluationError(
            "generation and evaluation model-panel lineage differ"
        )
    for key in (
        "progression_analysis_binding", "training_result_binding",
        "checkpoint_panel_bindings", "progression_proxy_routing",
        "predecessor_place_manifest_binding",
    ):
        if freeze.get(key) != separation.get(key):
            raise BoundedBranchEvaluationError(
                "generation and evaluation model-panel lineage differ"
            )


def _validate_checker_report_values_v1(
    report: object,
    *,
    report_schema: str,
    phase: str,
    manifest_binding: Mapping[str, Any],
    attempt_id: str,
    expected_counts: Mapping[str, Any],
) -> dict[str, Any]:
    common = {
        "schema", "status", "phase", "authority_granted",
        "scientific_claim_granted", "runtime_payloads_opened",
        "rgb_bytes_opened", "checkpoints_opened", "manifest_binding",
        "attempt_id", "purpose", "counts", "roles",
        "can_freeze_pilot_contract",
    }
    expected_fields = common if phase == "physics_collection" else common | {
        "rgb_artifacts"
    }
    expected_roles: object = (
        sorted(expected_counts["roles"])
        if phase == "physics_collection" else expected_counts["roles"]
    )
    if (
        phase not in {"physics_collection", "joined_pilot"}
        or not isinstance(report, Mapping)
        or set(report) != expected_fields
        or report.get("schema") != report_schema
        or report.get("status") != "PASS"
        or report.get("phase") != phase
        or report.get("purpose") != "bounded_wm_a_pilot"
        or report.get("attempt_id") != attempt_id
        or report.get("manifest_binding") != dict(manifest_binding)
        or report.get("authority_granted") is not False
        or report.get("scientific_claim_granted") is not False
        or report.get("runtime_payloads_opened") is not False
        or report.get("rgb_bytes_opened") is not False
        or report.get("checkpoints_opened") is not False
        or report.get("can_freeze_pilot_contract")
        is not (phase == "joined_pilot")
        or report.get("counts") != dict(expected_counts)
        or report.get("roles") != expected_roles
        or (
            phase == "joined_pilot"
            and report.get("rgb_artifacts")
            != int(expected_counts["context_frames"])
            + int(expected_counts["target_frames"])
        )
    ):
        raise BoundedBranchEvaluationError(
            f"bounded pilot {phase} check measurements changed"
        )
    return dict(report)


def load_and_validate_pilot_terminal_gate_v1(
    terminal_path: Path,
    *,
    expected_terminal_sha256: str,
    expected_terminal_byte_count: int,
    review_path: Path,
    expected_review_sha256: str,
    expected_review_byte_count: int,
    pilot_manifest_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an independently reviewed frozen pilot before opening RGB."""

    terminal_path = _require_nofollow_path(
        terminal_path, label="pilot terminal"
    )
    review_path = _require_nofollow_path(
        review_path, label="pilot terminal review"
    )
    from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as contract
    from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker
    from scripts import build_go2_world_model_bounded_branch_experiment_authority_v1 as authority_contract
    from scripts import run_go2_world_model_bounded_branch_experiment_authorized_v1 as pilot_supervisor

    terminal, terminal_binding = contract.read_bound_json(
        terminal_path,
        expected_sha256=expected_terminal_sha256,
        expected_byte_count=expected_terminal_byte_count,
        label="bounded pilot terminal",
    )
    review, review_binding = contract.read_bound_json(
        review_path,
        expected_sha256=expected_review_sha256,
        expected_byte_count=expected_review_byte_count,
        label="bounded pilot terminal review",
    )
    sensor = terminal.get("same_sensor_visual_domain_contract") if isinstance(terminal, Mapping) else None
    required_terminal = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "authorizes_refill_or_screening",
        "scientific_verdict_emitted",
        "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt",
        "authority_binding",
        "plan_binding",
        "reservation_binding",
        "calibration_gate_binding",
        "source_commit",
        "attempt_root",
        "wall_elapsed_seconds",
        "wall_ceiling_seconds",
        "phase_receipts",
        "physics_result_binding",
        "physics_receipt_check_binding",
        "joined_manifest_binding",
        "joined_receipt_check_binding",
        "gpu_memory_measurement",
        "active_vram_ceiling_enforcement",
        "same_sensor_visual_domain_contract",
        "evaluation_contract",
        "failure",
        "terminal_reviewer",
        "supervisor_nonce",
        "supervisor_pid",
    }
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != required_terminal
        or terminal.get("schema") != PILOT_TERMINAL_SCHEMA
        or terminal.get("status") != "COMPLETE_PENDING_INDEPENDENT_TERMINAL_REVIEW"
        or terminal.get("citable_as_scientific_evidence") is not False
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("authorizes_refill_or_screening") is not False
        or terminal.get("scientific_verdict_emitted") is not False
        or terminal.get("root_creation_consumes_attempt") is not True
        or terminal.get("reservation_records_consumed_attempt") is not True
        or terminal.get("joined_manifest_binding") != dict(pilot_manifest_binding)
        or terminal.get("failure") is not None
        or not isinstance(terminal.get("physics_result_binding"), Mapping)
        or not isinstance(terminal.get("physics_receipt_check_binding"), Mapping)
        or not isinstance(terminal.get("joined_receipt_check_binding"), Mapping)
        or not isinstance(sensor, Mapping)
        or sensor.get("enforced") is not True
        or sensor.get("task_valid_near_wall_or_low_texture_observations_screened") is not False
        or sensor.get("camera_or_render_failure_causes_terminal_failure") is not True
        or terminal.get("evaluation_contract") != evaluation_contract_v1()
    ):
        raise BoundedBranchEvaluationError("bounded pilot terminal did not pass exactly")
    authority_binding = contract.require_binding(
        terminal["authority_binding"], label="bounded pilot authority"
    )
    authority_path = _require_nofollow_path(
        Path(str(authority_binding["path"])), label="bounded pilot authority"
    )
    authority, actual_authority = contract.read_bound_json(
        authority_path,
        expected_sha256=str(authority_binding["file_sha256"]),
        expected_byte_count=int(authority_binding["byte_count"]),
        label="bounded pilot authority",
    )
    plan_binding = contract.require_binding(
        terminal["plan_binding"], label="bounded pilot plan"
    )
    plan_path = _require_nofollow_path(
        Path(str(plan_binding["path"])), label="bounded pilot plan"
    )
    plan, actual_plan = contract.read_bound_json(
        plan_path,
        expected_sha256=str(plan_binding["file_sha256"]),
        expected_byte_count=int(plan_binding["byte_count"]),
        label="bounded pilot plan",
    )
    gate_binding = contract.require_binding(
        terminal["calibration_gate_binding"], label="bounded pilot calibration gate"
    )
    gate_path = _require_nofollow_path(
        Path(str(gate_binding["path"])), label="bounded pilot calibration gate"
    )
    gate, actual_gate = contract.read_bound_json(
        gate_path,
        expected_sha256=str(gate_binding["file_sha256"]),
        expected_byte_count=int(gate_binding["byte_count"]),
        label="bounded pilot calibration gate",
    )
    if actual_plan != plan_binding or actual_gate != gate_binding:
        raise BoundedBranchEvaluationError("bounded pilot plan/gate binding changed")
    try:
        normalized_plan = contract.validate_plan(plan)
        normalized_authority = authority_contract.validate_authority_v1(
            authority,
            plan=normalized_plan,
            plan_binding=plan_binding,
            gate=gate,
            gate_binding=gate_binding,
        )
    except (
        contract.PilotContractError,
        authority_contract.BoundedBranchAuthorityError,
    ) as exc:
        raise BoundedBranchEvaluationError(
            f"bounded pilot full authority validation failed: {exc}"
        ) from exc
    expected_pilot_counts = {
        "scenes": 32,
        "states": 256,
        "roles": {"eval": 128, "train": 128},
        "actions": ACTION_COUNT,
        "candidate_branches": 2_304,
        "sentinel_branches": 0,
        "total_branches": 2_304,
        "context_frames": 768,
        "target_frames": 2_304,
    }
    if normalized_plan.get("expected_counts") != expected_pilot_counts:
        raise BoundedBranchEvaluationError(
            "bounded pilot plan does not have exact 32/256/2304 counts"
        )
    source_review_binding = contract.require_binding(
        normalized_authority["review_binding"], label="bounded pilot source review"
    )
    source_review_path = _require_nofollow_path(
        Path(str(source_review_binding["path"])), label="bounded pilot source review"
    )
    source_review, actual_source_review = contract.read_bound_json(
        source_review_path,
        expected_sha256=str(source_review_binding["file_sha256"]),
        expected_byte_count=int(source_review_binding["byte_count"]),
        label="bounded pilot source review",
    )
    try:
        contract.validate_source_review(
            source_review, authority=normalized_authority
        )
        current_committed_sources = (
            authority_contract.committed_source_bindings_v1(
                str(normalized_authority["source_commit"])
            )
        )
    except (
        contract.PilotContractError,
        authority_contract.BoundedBranchAuthorityError,
    ) as exc:
        raise BoundedBranchEvaluationError(
            f"bounded pilot reviewed source closure failed: {exc}"
        ) from exc
    if (
        actual_source_review != source_review_binding
        or current_committed_sources != normalized_authority["source_bindings"]
    ):
        raise BoundedBranchEvaluationError(
            "bounded pilot reviewed source closure changed"
        )
    if (
        actual_authority != authority_binding
        or not isinstance(authority, Mapping)
        or authority.get("schema") != authority_contract.AUTHORITY_SCHEMA
        or authority.get("status") != authority_contract.AUTHORITY_STATUS
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("plan_binding") != plan_binding
        or authority.get("calibration_gate_binding") != gate_binding
        or authority.get("source_commit") != terminal["source_commit"]
        or authority.get("evaluation_contract") != evaluation_contract_v1()
        or authority.get("visual_domain_parity_freeze")
        != gate.get("visual_domain_parity_freeze")
    ):
        raise BoundedBranchEvaluationError("bounded pilot authority link changed")
    attempt_root = _require_nofollow_path(
        Path(str(normalized_authority["attempt"]["root"])),
        label="bounded pilot attempt root",
    )
    if (
        terminal.get("attempt_root") != str(attempt_root)
        or normalized_plan.get("output_root") != str(attempt_root)
        or terminal_path.resolve(strict=True)
        != (attempt_root / "terminal_supervision.json").resolve(strict=True)
    ):
        raise BoundedBranchEvaluationError("bounded pilot attempt path changed")
    physics_binding = contract.require_binding(
        terminal["physics_result_binding"], label="bounded pilot physics result"
    )
    physics_path = _require_nofollow_path(
        Path(str(physics_binding["path"])), label="bounded pilot physics result"
    )
    if physics_path.resolve(strict=True) != (
        attempt_root / "physics_result.json"
    ).resolve(strict=True):
        raise BoundedBranchEvaluationError("bounded pilot physics path changed")
    physics, actual_physics = contract.read_bound_json(
        physics_path,
        expected_sha256=str(physics_binding["file_sha256"]),
        expected_byte_count=int(physics_binding["byte_count"]),
        label="bounded pilot physics result",
    )
    if (
        actual_physics != physics_binding
        or not isinstance(physics, Mapping)
        or physics.get("schema") != contract.PHYSICS_RESULT_SCHEMA
        or physics.get("status") != "PHYSICS_COMPLETE"
        or physics.get("purpose") != "bounded_wm_a_pilot"
        or physics.get("physics_validated") is not False
        or physics.get("citable_as_scientific_evidence") is not False
        or physics.get("authorizes_retry_or_resume") is not False
        or physics.get("plan_binding") != terminal["plan_binding"]
        or physics.get("authority_binding") != authority_binding
        or physics.get("reservation_binding") != terminal["reservation_binding"]
        or physics.get("caps") != authority.get("caps")
        or physics.get("failure") is not None
    ):
        raise BoundedBranchEvaluationError("bounded pilot physics link changed")

    def checked_report(
        binding_value: object,
        *,
        phase: str,
        manifest_binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        report_binding = contract.require_binding(
            binding_value, label=f"bounded pilot {phase} check"
        )
        report_path = _require_nofollow_path(
            Path(str(report_binding["path"])),
            label=f"bounded pilot {phase} check",
        )
        expected_name = (
            "physics_receipt_check.json"
            if phase == "physics_collection"
            else "joined_receipt_check.json"
        )
        if report_path.resolve(strict=True) != (
            attempt_root / expected_name
        ).resolve(strict=True):
            raise BoundedBranchEvaluationError(
                f"bounded pilot {phase} check path changed"
            )
        report, actual_report = contract.read_bound_json(
            report_path,
            expected_sha256=str(report_binding["file_sha256"]),
            expected_byte_count=int(report_binding["byte_count"]),
            label=f"bounded pilot {phase} check",
        )
        if (
            actual_report != report_binding
        ):
            raise BoundedBranchEvaluationError(
                f"bounded pilot {phase} check link changed"
            )
        normalized_report = _validate_checker_report_values_v1(
            report,
            report_schema=checker.REPORT_SCHEMA,
            phase=phase,
            manifest_binding=manifest_binding,
            attempt_id=str(normalized_plan["attempt_id"]),
            expected_counts=normalized_plan["expected_counts"],
        )
        return report_binding, normalized_report

    physics_check_binding, physics_check = checked_report(
        terminal["physics_receipt_check_binding"],
        phase="physics_collection",
        manifest_binding=physics_binding,
    )
    joined_check_binding, joined_check = checked_report(
        terminal["joined_receipt_check_binding"],
        phase="joined_pilot",
        manifest_binding=pilot_manifest_binding,
    )
    caps = authority.get("caps")
    gpu = terminal.get("gpu_memory_measurement")
    active_vram = terminal.get("active_vram_ceiling_enforcement")
    elapsed = terminal.get("wall_elapsed_seconds")
    ceiling = terminal.get("wall_ceiling_seconds")
    required_gpu = {
        "scope",
        "attribution_limitation",
        "vendor_id",
        "device_id",
        "used_counter_path",
        "total_counter_path",
        "sample_interval_seconds",
        "sample_count",
        "read_errors",
        "baseline_used_bytes",
        "peak_used_bytes",
        "peak_delta_above_baseline_bytes",
        "device_total_bytes",
    }
    required_active_vram = {
        "enabled",
        "scope",
        "ceiling_bytes",
        "sample_interval_seconds",
        "collector_started",
        "collector_pid",
        "collector_exit_code",
        "collector_terminated",
        "termination_reason",
        "peak_observed_during_collector_bytes",
    }
    reservation = terminal.get("reservation_binding")
    if (
        not isinstance(reservation, Mapping)
        or set(reservation) != {"path", "file_sha256", "byte_count"}
        or reservation.get("path") != "reservation.json"
    ):
        raise BoundedBranchEvaluationError("bounded pilot reservation link changed")
    reservation_path = _require_nofollow_path(
        attempt_root / "reservation.json", label="bounded pilot reservation"
    )
    reservation_actual = contract.file_binding(reservation_path)
    if {
        "path": "reservation.json",
        "file_sha256": reservation_actual["file_sha256"],
        "byte_count": reservation_actual["byte_count"],
    } != dict(reservation):
        raise BoundedBranchEvaluationError("bounded pilot reservation bytes changed")
    try:
        reservation_document = json.loads(reservation_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundedBranchEvaluationError(
            "bounded pilot reservation is not strict JSON"
        ) from exc
    terminal_nonce = terminal.get("supervisor_nonce")
    terminal_pid = terminal.get("supervisor_pid")
    _validate_reservation_document_v1(
        reservation_document,
        attempt=normalized_authority["attempt"],
        plan_binding=plan_binding,
        authority_binding=authority_binding,
        supervisor_nonce=terminal_nonce,
        supervisor_pid=terminal_pid,
    )
    strict_physics, strict_physics_binding = (
        pilot_supervisor._load_physics_result_if_present(  # noqa: SLF001
            attempt_root,
            plan=normalized_plan,
            plan_binding=plan_binding,
            authority=normalized_authority,
            authority_binding=authority_binding,
            reservation_binding=reservation,
        )
    )
    if strict_physics != physics or strict_physics_binding != physics_binding:
        raise BoundedBranchEvaluationError(
            "bounded pilot physics result failed independent reconstruction"
        )
    joined_manifest_path = _require_nofollow_path(
        Path(str(pilot_manifest_binding["path"])),
        label="bounded pilot joined manifest",
    )
    if joined_manifest_path.resolve(strict=True) != (
        attempt_root / "manifest.json"
    ).resolve(strict=True):
        raise BoundedBranchEvaluationError(
            "bounded pilot joined manifest path changed"
        )
    invocation = str(normalized_plan["execution_contract"]["python_invocation_path"])
    expected_command_argv = [
        [
            invocation,
            str((REPO_ROOT / pilot_supervisor.COLLECTOR_RELATIVE).resolve()),
            "--plan", str(plan_path.resolve()),
            "--expected-plan-byte-count", str(plan_binding["byte_count"]),
            "--expected-plan-sha256", str(plan_binding["file_sha256"]),
            "--authority", str(authority_path.resolve()),
            "--expected-authority-byte-count", str(authority_binding["byte_count"]),
            "--expected-authority-sha256", str(authority_binding["file_sha256"]),
            "--supervisor-nonce", str(terminal_nonce),
            "--supervisor-pid", str(terminal_pid),
        ],
        [
            invocation,
            str((REPO_ROOT / pilot_supervisor.CHECKER_RELATIVE).resolve()),
            "--manifest", str(physics_path),
            "--expected-file-sha256", str(physics_binding["file_sha256"]),
            "--expected-byte-count", str(physics_binding["byte_count"]),
            "--output", str((attempt_root / "physics_receipt_check.json").resolve()),
        ],
        [
            invocation,
            str((REPO_ROOT / pilot_supervisor.JOINER_RELATIVE).resolve()),
            "--collection", str(physics_path),
            "--expected-collection-sha256", str(physics_binding["file_sha256"]),
            "--expected-collection-byte-count", str(physics_binding["byte_count"]),
            "--calibration-receipt", str(gate["calibration_receipt_binding"]["path"]),
            "--expected-calibration-sha256",
            str(gate["calibration_receipt_binding"]["file_sha256"]),
            "--expected-calibration-byte-count",
            str(gate["calibration_receipt_binding"]["byte_count"]),
        ],
        [
            invocation,
            str((REPO_ROOT / pilot_supervisor.CHECKER_RELATIVE).resolve()),
            "--manifest", str(joined_manifest_path),
            "--expected-file-sha256", str(pilot_manifest_binding["file_sha256"]),
            "--expected-byte-count", str(pilot_manifest_binding["byte_count"]),
            "--output", str((attempt_root / "joined_receipt_check.json").resolve()),
        ],
    ]
    phases = terminal.get("phase_receipts")
    if not isinstance(phases, list) or len(phases) != 5:
        raise BoundedBranchEvaluationError("bounded pilot phase receipt count changed")
    graphics = phases[0]
    if (
        not isinstance(graphics, Mapping)
        or set(graphics) != {
            "phase", "status", "environment", "expectation",
            "vulkan_stdout_sha256", "egl_stdout_sha256", "egl_stderr_sha256",
            "egl_exit_code",
        }
        or graphics.get("phase") != "graphics_preflight"
        or graphics.get("status") != "PASS"
        or graphics.get("environment")
        != normalized_plan["execution_contract"]["environment"]
        or graphics.get("expectation")
        != normalized_plan["execution_contract"]["graphics_preflight"]
        or graphics.get("egl_exit_code")
        != graphics["expectation"]["eglinfo_expected_exit_code"]
        or any(
            not isinstance(graphics.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", graphics[field]) is None
            for field in (
                "vulkan_stdout_sha256", "egl_stdout_sha256", "egl_stderr_sha256"
            )
        )
    ):
        raise BoundedBranchEvaluationError(
            "bounded pilot graphics preflight receipt changed"
        )
    for offset, expected_argv in enumerate(expected_command_argv, start=1):
        phase = phases[offset]
        expected_fields = {"argv", "elapsed_seconds", "exit_code"}
        if offset == 1:
            expected_fields.add("active_vram_ceiling_enforced")
        elapsed_phase = phase.get("elapsed_seconds") if isinstance(phase, Mapping) else None
        if (
            not isinstance(phase, Mapping)
            or set(phase) != expected_fields
            or phase.get("argv") != expected_argv
            or phase.get("exit_code") != 0
            or isinstance(elapsed_phase, bool)
            or not isinstance(elapsed_phase, (int, float))
            or not math.isfinite(float(elapsed_phase))
            or float(elapsed_phase) < 0.0
            or (
                offset == 1
                and phase.get("active_vram_ceiling_enforced") is not True
            )
        ):
            raise BoundedBranchEvaluationError(
                "bounded pilot supervised phase receipt changed"
            )
    if (
        not isinstance(caps, Mapping)
        or not isinstance(gpu, Mapping)
        or set(gpu) != required_gpu
        or gpu.get("scope") != "selected_device_global_vram_not_process_attributed"
        or gpu.get("read_errors") != 0
        or type(gpu.get("sample_interval_seconds")) not in (int, float)
        or float(gpu["sample_interval_seconds"]) != 0.02
        or type(gpu.get("sample_count")) is not int
        or gpu["sample_count"] <= 0
        or any(
            type(gpu.get(field)) is not int or gpu[field] < 0
            for field in (
                "baseline_used_bytes", "peak_used_bytes",
                "peak_delta_above_baseline_bytes", "device_total_bytes",
            )
        )
        or gpu["peak_used_bytes"] < gpu["baseline_used_bytes"]
        or gpu["peak_delta_above_baseline_bytes"]
        != gpu["peak_used_bytes"] - gpu["baseline_used_bytes"]
        or gpu["peak_used_bytes"] > gpu["device_total_bytes"]
        or gpu["device_total_bytes"] != gate["selected_device_total_vram_bytes"]
        or type(gpu.get("peak_used_bytes")) is not int
        or type(caps.get("selected_device_vram_byte_ceiling")) is not int
        or gpu["peak_used_bytes"] > caps["selected_device_vram_byte_ceiling"]
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or isinstance(ceiling, bool)
        or not isinstance(ceiling, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < sum(
            float(phase["elapsed_seconds"]) for phase in phases[1:]
        )
        or not math.isfinite(float(ceiling))
        or float(elapsed) > float(ceiling)
        or float(ceiling) != float(caps.get("wall_seconds", -1))
        or not isinstance(active_vram, Mapping)
        or set(active_vram) != required_active_vram
        or active_vram.get("enabled") is not True
        or active_vram.get("scope")
        != "selected_device_global_vram_not_process_attributed"
        or type(active_vram.get("sample_interval_seconds")) not in (int, float)
        or float(active_vram["sample_interval_seconds"]) != 0.02
        or active_vram.get("ceiling_bytes")
        != caps.get("selected_device_vram_byte_ceiling")
        or active_vram.get("collector_started") is not True
        or type(active_vram.get("collector_pid")) is not int
        or active_vram["collector_pid"] <= 1
        or active_vram["collector_pid"] == terminal_pid
        or active_vram.get("collector_exit_code") != 0
        or active_vram.get("collector_terminated") is not False
        or active_vram.get("termination_reason") is not None
        or type(active_vram.get("peak_observed_during_collector_bytes")) is not int
        or active_vram["peak_observed_during_collector_bytes"]
        > active_vram["ceiling_bytes"]
        or active_vram["peak_observed_during_collector_bytes"]
        > gpu["peak_used_bytes"]
        or terminal.get("terminal_reviewer")
        != normalized_authority["external_supervisor"]["terminal_reviewer"]
        or terminal.get("source_commit") != normalized_authority["source_commit"]
    ):
        raise BoundedBranchEvaluationError("bounded pilot resource cap link changed")
    required_review = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_granted",
        "terminal_binding",
        "joined_manifest_binding",
        "reviewer",
        "reviewed_at",
        "checks",
        "remaining_findings",
    }
    checks = review.get("checks") if isinstance(review, Mapping) else None
    if (
        not isinstance(review, Mapping)
        or set(review) != required_review
        or review.get("schema") != PILOT_TERMINAL_REVIEW_SCHEMA
        or review.get("status") != "PASS_FROZEN_BOUNDED_PILOT"
        or review.get("authority_granted_by_this_document") is not False
        or review.get("scientific_claim_granted") is not False
        or review.get("terminal_binding") != terminal_binding
        or review.get("joined_manifest_binding") != dict(pilot_manifest_binding)
        or review.get("remaining_findings") != []
        or not isinstance(checks, Mapping)
        or set(checks) != {
            "terminal_complete",
            "physics_receipt_passed",
            "joined_receipt_passed",
            "same_sensor_contract_passed",
            "resource_caps_passed",
            "reservation_owned",
            "active_vram_enforced",
            "no_retry_resume_refill",
        }
        or any(value is not True for value in checks.values())
    ):
        raise BoundedBranchEvaluationError("bounded pilot terminal review did not pass")
    reviewer = review.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or any(not isinstance(reviewer[key], str) or not reviewer[key].strip() for key in reviewer)
        or not isinstance(review.get("reviewed_at"), str)
        or not review["reviewed_at"].strip()
    ):
        raise BoundedBranchEvaluationError("bounded pilot terminal reviewer is invalid")
    terminal_rehashes = (
        (terminal_path, terminal_binding, "terminal"),
        (review_path, review_binding, "terminal review"),
        (authority_path, authority_binding, "authority"),
        (plan_path, plan_binding, "plan"),
        (gate_path, gate_binding, "calibration gate"),
        (source_review_path, source_review_binding, "source review"),
        (physics_path, physics_binding, "physics result"),
        (
            Path(str(physics_check_binding["path"])),
            physics_check_binding,
            "physics check",
        ),
        (joined_manifest_path, dict(pilot_manifest_binding), "joined manifest"),
        (
            Path(str(joined_check_binding["path"])),
            joined_check_binding,
            "joined check",
        ),
    )
    for path_value, expected_binding, label in terminal_rehashes:
        if contract.file_binding(path_value) != expected_binding:
            raise BoundedBranchEvaluationError(
                f"bounded pilot {label} changed during terminal validation"
            )
    reservation_rehash = contract.file_binding(reservation_path)
    if {
        "path": "reservation.json",
        "file_sha256": reservation_rehash["file_sha256"],
        "byte_count": reservation_rehash["byte_count"],
    } != dict(reservation):
        raise BoundedBranchEvaluationError(
            "bounded pilot reservation changed during terminal validation"
        )
    for source_row in normalized_authority["source_bindings"]:
        if contract.file_binding(Path(source_row["binding"]["path"])) != source_row[
            "binding"
        ]:
            raise BoundedBranchEvaluationError(
                "bounded pilot source changed during terminal validation"
            )
    return {
        "pilot_terminal_binding": terminal_binding,
        "pilot_terminal_review_binding": review_binding,
        "joined_manifest_binding": dict(pilot_manifest_binding),
        "authority_binding": authority_binding,
        "physics_result_binding": physics_binding,
        "physics_receipt_check_binding": physics_check_binding,
        "joined_receipt_check_binding": joined_check_binding,
        "plan_binding": plan_binding,
        "calibration_gate_binding": gate_binding,
        "source_review_binding": source_review_binding,
        "model_panel_freeze": dict(normalized_authority["model_panel_freeze"]),
        "scene_panel_freeze": dict(normalized_authority["scene_panel_freeze"]),
        "visual_domain_parity_freeze": dict(
            normalized_authority["visual_domain_parity_freeze"]
        ),
        "status": "PASS_FROZEN_BOUNDED_PILOT",
    }


def fit_train_latent_standardizer_v1(
    train_true_features_by_state: Mapping[str, Sequence[object]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        _latent_prefix(feature)
        for state_id in sorted(train_true_features_by_state)
        for feature in train_true_features_by_state[state_id]
    ]
    if not rows:
        raise BoundedBranchEvaluationError("train true-future latents are absent")
    matrix = np.stack(rows)
    if matrix.shape[0] < ACTION_COUNT:
        raise BoundedBranchEvaluationError("latent standardizer has too few train rows")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return mean, scale


def _standardized(feature: object, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    latent = _latent_prefix(feature)
    if latent.shape != mean.shape or scale.shape != mean.shape:
        raise BoundedBranchEvaluationError("latent descriptor dimension changed")
    return (latent - mean) / scale


def _eligible_action_ids_from_joint_signatures_v1(
    signatures: Sequence[Mapping[str, object]],
) -> list[int]:
    if len(signatures) != ACTION_COUNT:
        raise BoundedBranchEvaluationError(
            "joint discrimination signature grid changed"
        )
    eligible: list[int] = []
    for action_id, signature in enumerate(signatures):
        if (
            set(signature)
            != {
                "action_id",
                "executed_tape_class_sha256",
                "physical_outcome_dense_rank",
                "stored_rgb_pixel_class_sha256",
            }
            or signature["action_id"] != action_id
            or not isinstance(signature["executed_tape_class_sha256"], str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(signature["executed_tape_class_sha256"]),
            )
            is None
            or type(signature["physical_outcome_dense_rank"]) is not int
            or int(signature["physical_outcome_dense_rank"]) < 0
            or not isinstance(signature["stored_rgb_pixel_class_sha256"], str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(signature["stored_rgb_pixel_class_sha256"]),
            )
            is None
        ):
            raise BoundedBranchEvaluationError(
                "joint discrimination signature changed"
            )
        nonequivalent = [
            candidate
            for candidate in signatures
            if candidate["physical_outcome_dense_rank"]
            != signature["physical_outcome_dense_rank"]
        ]
        if nonequivalent and all(
            candidate["executed_tape_class_sha256"]
            != signature["executed_tape_class_sha256"]
            and candidate["stored_rgb_pixel_class_sha256"]
            != signature["stored_rgb_pixel_class_sha256"]
            for candidate in nonequivalent
        ):
            eligible.append(action_id)
    return eligible


def _jointly_eligible_actions_v1(
    branches: Sequence[object], physical_ranks: np.ndarray
) -> tuple[list[dict[str, object]], list[int]]:
    """Return queries with no visually/tape-aliased physical alternative."""

    if len(branches) != ACTION_COUNT or physical_ranks.shape != (ACTION_COUNT,):
        raise BoundedBranchEvaluationError(
            "joint discrimination branch grid changed"
        )
    signatures: list[dict[str, object]] = []
    for action_id, branch in enumerate(branches):
        signatures.append({
            "action_id": action_id,
            "executed_tape_class_sha256": getattr(
                branch, "executed_command_tape_sha256", None
            ),
            "physical_outcome_dense_rank": int(physical_ranks[action_id]),
            "stored_rgb_pixel_class_sha256": getattr(
                branch, "target_rgb_pixel_sha256", None
            ),
        })
    eligible = _eligible_action_ids_from_joint_signatures_v1(signatures)
    return signatures, eligible


def direct_matched_branch_metrics_v1(
    *,
    groups: Sequence[object],
    predicted_features_by_state: Mapping[str, Sequence[object]],
    true_features_by_state: Mapping[str, Sequence[object]],
    train_mean: np.ndarray,
    train_scale: np.ndarray,
) -> dict[str, Any]:
    """Score exact fidelity and fixed physical-outcome-equivalent retrieval."""

    rows: list[dict[str, Any]] = []
    separable_rows: list[dict[str, Any]] = []
    separable_actions_total = 0
    for group in groups:
        state_id = str(group.state_id)
        predicted_raw = predicted_features_by_state.get(state_id)
        true_raw = true_features_by_state.get(state_id)
        branches = getattr(group, "branches", None)
        if (
            predicted_raw is None
            or true_raw is None
            or len(predicted_raw) != ACTION_COUNT
            or len(true_raw) != ACTION_COUNT
            or branches is None
            or len(branches) != ACTION_COUNT
        ):
            raise BoundedBranchEvaluationError(
                f"state {state_id} lacks nine bound physical/latent branches"
            )
        physical_ranks = np.asarray(
            [getattr(branch, "oracle_dense_rank", None) for branch in branches]
        )
        if (
            physical_ranks.shape != (ACTION_COUNT,)
            or physical_ranks.dtype.kind not in {"i", "u"}
            or bool((physical_ranks < 0).any())
            or not np.array_equal(
                np.unique(physical_ranks),
                np.arange(np.unique(physical_ranks).size),
            )
        ):
            raise BoundedBranchEvaluationError(
                f"state {state_id} physical oracle classes changed"
            )
        joint_signatures, eligible_action_ids = _jointly_eligible_actions_v1(
            branches, physical_ranks
        )
        eligible_actions = set(eligible_action_ids)
        predicted = np.stack([
            _standardized(value, train_mean, train_scale) for value in predicted_raw
        ])
        truth = np.stack([
            _standardized(value, train_mean, train_scale) for value in true_raw
        ])
        distances = np.sqrt(np.mean(
            (predicted[:, None, :] - truth[None, :, :]) ** 2,
            axis=2,
        ))
        matched = np.diag(distances)
        retrieved = np.argmin(distances, axis=1)
        margins: list[float] = []
        retrieval_correct: list[float] = []
        retrieval_chance: list[float] = []
        minimum_nonequivalent: list[float] = []
        equivalence_class_sizes: list[int] = []
        for action in range(ACTION_COUNT):
            equivalent = physical_ranks == physical_ranks[action]
            nonequivalent = ~equivalent
            if action not in eligible_actions:
                continue
            nearest_equivalent = float(distances[action, equivalent].min())
            nearest_nonequivalent = float(distances[action, nonequivalent].min())
            margins.append(nearest_nonequivalent - nearest_equivalent)
            minimum_nonequivalent.append(nearest_nonequivalent)
            retrieval_correct.append(float(bool(equivalent[int(retrieved[action])])))
            class_size = int(equivalent.sum())
            equivalence_class_sizes.append(class_size)
            retrieval_chance.append(class_size / ACTION_COUNT)
        separable_actions = len(margins)
        separable_actions_total += separable_actions
        row = {
            "state_id": state_id,
            "scene_id": str(group.scene_id),
            "family": str(group.family),
            "matched_branch_error": float(matched.mean()),
            "physical_oracle_dense_ranks": [
                int(value) for value in physical_ranks.tolist()
            ],
            "physical_oracle_class_count": int(np.unique(physical_ranks).size),
            "joint_contrast_signatures_by_action": joint_signatures,
            "eligible_action_ids": eligible_action_ids,
            "separable_actions": separable_actions,
            "separable_action_fraction": separable_actions / ACTION_COUNT,
        }
        rows.append(row)
        if separable_actions:
            accuracy = float(np.mean(retrieval_correct))
            chance = float(np.mean(retrieval_chance))
            separable_rows.append(
                {
                    **row,
                    "nearest_nonequivalent_branch_error": float(
                        np.mean(minimum_nonequivalent)
                    ),
                    "branch_margin": float(np.mean(margins)),
                    "equivalence_aware_retrieval_accuracy": accuracy,
                    "equivalence_adjusted_chance": chance,
                    "retrieval_advantage": accuracy - chance,
                    "mean_equivalence_class_size": float(
                        np.mean(equivalence_class_sizes)
                    ),
                }
            )
    if not rows:
        raise BoundedBranchEvaluationError("direct branch evaluation has no groups")
    total_actions = len(rows) * ACTION_COUNT
    separable_scene_keys = {
        (str(row["family"]), str(row["scene_id"])) for row in separable_rows
    }
    all_scene_keys = {(str(row["family"]), str(row["scene_id"])) for row in rows}
    return {
        "summary": {
            "matched_branch_error": float(
                np.mean([float(row["matched_branch_error"]) for row in rows])
            ),
            "separable_action_coverage": separable_actions_total / total_actions,
            "separable_state_coverage": len(separable_rows) / len(rows),
            "separable_scene_coverage": len(separable_scene_keys) / len(all_scene_keys),
            "branch_margin": (
                float(np.mean([float(row["branch_margin"]) for row in separable_rows]))
                if separable_rows
                else None
            ),
            "equivalence_aware_retrieval_accuracy": (
                float(
                    np.mean(
                        [
                            float(row["equivalence_aware_retrieval_accuracy"])
                            for row in separable_rows
                        ]
                    )
                )
                if separable_rows
                else None
            ),
            "retrieval_advantage": (
                float(
                    np.mean([float(row["retrieval_advantage"]) for row in separable_rows])
                )
                if separable_rows
                else None
            ),
        },
        "outcome_equivalence": {
            "basis": (
                "equal_frozen_calibration_tolerance_aware_physical_oracle_dense_rank"
            ),
            "model_dependent": False,
            "latent_proximity_used": False,
            "query_eligibility": (
                "at_least_one_physically_nonequivalent_alternative_and_every_"
                "physically_nonequivalent_alternative_has_different_executed_"
                "tape_and_stored_rgb_raw_pixel_identity"
            ),
        },
        "group_results": rows,
        "separable_group_results": separable_rows,
    }


def scene_cluster_interval_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    two_sided_alpha: float = 0.05,
) -> dict[str, Any]:
    if resamples <= 0:
        raise BoundedBranchEvaluationError("bootstrap resamples must be positive")
    if not 0.0 < float(two_sided_alpha) < 1.0:
        raise BoundedBranchEvaluationError("bootstrap alpha must be between zero and one")
    by_scene: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            raise BoundedBranchEvaluationError(f"{field} must be finite")
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise BoundedBranchEvaluationError(
                "family-stratified interval requires family identity"
            )
        by_scene[(family, str(row["scene_id"]))].append(value)
    if not by_scene:
        raise BoundedBranchEvaluationError("scene-cluster interval has no rows")
    by_family: dict[str, list[float]] = defaultdict(list)
    for (family, scene), values in sorted(by_scene.items()):
        del scene
        by_family[family].append(float(np.mean(values)))
    if set(by_family) != set(FAMILIES):
        raise BoundedBranchEvaluationError(
            "family-stratified interval lost a fixed pilot family"
        )
    rng = np.random.default_rng(seed)
    family_draws = []
    for family in sorted(by_family):
        values = np.asarray(by_family[family], dtype=np.float64)
        family_draws.append(
            values[
                rng.integers(0, len(values), size=(resamples, len(values)))
            ].mean(axis=1)
        )
    samples = np.stack(family_draws, axis=1).mean(axis=1)
    lower, upper = np.quantile(
        samples, [two_sided_alpha / 2.0, 1.0 - two_sided_alpha / 2.0]
    )
    point = float(
        np.mean([np.mean(values) for values in by_family.values()])
    )
    return {
        "field": field,
        "scene_clusters": len(by_scene),
        "family_strata": len(by_family),
        "scenes_per_family": {
            family: len(values) for family, values in sorted(by_family.items())
        },
        "groups": len(rows),
        "resamples": resamples,
        "seed": seed,
        "two_sided_alpha": float(two_sided_alpha),
        "mean": point,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _physical_rows_with_unsafe(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in report["group_results"]:
        row = dict(source)
        row["physical_unsafe"] = float(
            bool(row["physical_fell"]) or bool(row["physical_tipped"])
        )
        rows.append(row)
    return rows


def _paired(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    field: str,
    resamples: int,
    seed: int,
    two_sided_alpha: float = 0.05,
) -> dict[str, Any]:
    candidate_by_state = {str(row["state_id"]): row for row in candidate}
    baseline_by_state = {str(row["state_id"]): row for row in baseline}
    if set(candidate_by_state) != set(baseline_by_state):
        raise BoundedBranchEvaluationError("paired comparison state identities changed")
    rows = []
    for state_id in sorted(candidate_by_state):
        candidate_row = candidate_by_state[state_id]
        baseline_row = baseline_by_state[state_id]
        if (
            candidate_row.get("scene_id") != baseline_row.get("scene_id")
            or candidate_row.get("family") != baseline_row.get("family")
        ):
            raise BoundedBranchEvaluationError(
                "paired comparison scene/family identity changed"
            )
        rows.append(
            {
                "state_id": state_id,
                "scene_id": candidate_row["scene_id"],
                "family": candidate_row["family"],
                "difference": float(candidate_row[field])
                - float(baseline_row[field]),
            }
        )
    result = scene_cluster_interval_v1(
        rows,
        field="difference",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    result["field"] = field
    result["direction"] = "candidate_minus_baseline_lower_is_better"
    if field in {"branch_margin", "physical_target_progress_m"}:
        result["direction"] = "candidate_minus_baseline_higher_is_better"
    return result


def preregistered_verdict_v1(
    *,
    direct_forecast: Mapping[str, Any],
    direct_shuffled: Mapping[str, Any],
    physical_arms: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    two_sided_alpha: float = 0.05,
) -> tuple[str, dict[str, Any]]:
    thresholds = USEFUL_EFFECT_THRESHOLDS
    direct_error = _paired(
        direct_forecast["group_results"],
        direct_shuffled["group_results"],
        field="matched_branch_error",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    discrimination_coverage = float(
        direct_forecast["summary"]["separable_action_coverage"]
    )
    family_actions = {family: 0 for family in FAMILIES}
    family_total = {family: 0 for family in FAMILIES}
    family_scenes = {family: set() for family in FAMILIES}
    family_eligible_scenes = {family: set() for family in FAMILIES}
    for row in direct_forecast["group_results"]:
        family = str(row["family"])
        if family not in family_actions:
            raise BoundedBranchEvaluationError(
                "direct discrimination row names an unexpected family"
            )
        family_actions[family] += int(row["separable_actions"])
        family_total[family] += ACTION_COUNT
        scene_id = str(row["scene_id"])
        family_scenes[family].add(scene_id)
        if int(row["separable_actions"]) > 0:
            family_eligible_scenes[family].add(scene_id)
    family_coverage = {
        family: family_actions[family] / family_total[family]
        if family_total[family] else 0.0
        for family in FAMILIES
    }
    every_family_meets_coverage = all(
        family_coverage[family] >= MINIMUM_DISCRIMINATION_COVERAGE
        for family in FAMILIES
    )
    both_fixed_scenes_per_family_have_support = all(
        len(family_scenes[family]) == 2
        and family_eligible_scenes[family] == family_scenes[family]
        for family in FAMILIES
    )
    discrimination_applicable = (
        discrimination_coverage >= MINIMUM_DISCRIMINATION_COVERAGE
        and every_family_meets_coverage
        and both_fixed_scenes_per_family_have_support
    )
    direct_margin = None
    retrieval = None
    if discrimination_applicable:
        direct_margin = _paired(
            direct_forecast["separable_group_results"],
            direct_shuffled["separable_group_results"],
            field="branch_margin",
            resamples=resamples,
            seed=seed,
            two_sided_alpha=two_sided_alpha,
        )
        retrieval = scene_cluster_interval_v1(
            direct_forecast["separable_group_results"],
            field="retrieval_advantage",
            resamples=resamples,
            seed=seed,
            two_sided_alpha=two_sided_alpha,
        )
    forecast_rows = _physical_rows_with_unsafe(physical_arms["forecast"])
    current_rows = _physical_rows_with_unsafe(physical_arms["current_state_action"])
    safety = _paired(
        forecast_rows,
        current_rows,
        field="physical_unsafe",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    progress = _paired(
        forecast_rows,
        current_rows,
        field="physical_target_progress_m",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    forecast_safety = scene_cluster_interval_v1(
        forecast_rows,
        field="physical_unsafe",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    forecast_progress = scene_cluster_interval_v1(
        forecast_rows,
        field="physical_target_progress_m",
        resamples=resamples,
        seed=seed,
        two_sided_alpha=two_sided_alpha,
    )
    gates = {
        "evaluator_sensitivity": {
            "passed": float(comparisons["ceiling_vs_current"]["upper_95"])
            <= -thresholds["ceiling_rank_regret_reduction_vs_current"],
            "measurement": dict(comparisons["ceiling_vs_current"]),
        },
        "direct_matched_error": {
            "passed": float(direct_error["upper_95"])
            <= -thresholds["direct_error_reduction_vs_shuffled"],
            "measurement": direct_error,
        },
        "direct_branch_margin": {
            "applicable": discrimination_applicable,
            "passed": (
                float(direct_margin["lower_95"])
                >= thresholds["direct_margin_gain_vs_shuffled"]
                if direct_margin is not None
                else None
            ),
            "measurement": direct_margin,
        },
        "direct_action_retrieval": {
            "applicable": discrimination_applicable,
            "passed": (
                float(retrieval["lower_95"])
                >= thresholds["retrieval_advantage_over_chance"]
                if retrieval is not None
                else None
            ),
            "measurement": retrieval,
        },
        "direct_discrimination_coverage": {
            "passed": discrimination_applicable,
            "measurement": {
                "minimum_for_applicability": MINIMUM_DISCRIMINATION_COVERAGE,
                "query_eligibility": (
                    "at_least_one_physically_nonequivalent_alternative_and_"
                    "every_physically_nonequivalent_alternative_has_different_"
                    "executed_tape_and_stored_rgb_raw_pixel_identity"
                ),
                "every_fixed_family_requires_minimum_eligible_coverage": True,
                "minimum_within_each_family": (
                    MINIMUM_DISCRIMINATION_COVERAGE
                ),
                "both_fixed_eval_scenes_per_family_require_support": True,
                "separable_actions_by_family": family_actions,
                "separable_action_coverage_by_family": family_coverage,
                "eligible_scene_ids_by_family": {
                    family: sorted(family_eligible_scenes[family])
                    for family in FAMILIES
                },
                **dict(direct_forecast["summary"]),
            },
            "interpretation": (
                "discrimination_gates_applicable"
                if discrimination_applicable
                else "insufficient_physical_action_separability_inconclusive_data"
            ),
        },
        "physical_rank_regret": {
            "passed": float(comparisons["forecast_vs_current"]["upper_95"])
            <= -thresholds["rank_regret_reduction_vs_current"],
            "measurement": dict(comparisons["forecast_vs_current"]),
        },
        "safety_noninferiority": {
            "passed": float(safety["upper_95"])
            <= thresholds["maximum_unsafe_rate_increase_vs_current"],
            "measurement": safety,
        },
        "absolute_safety": {
            "passed": float(forecast_safety["upper_95"])
            <= thresholds["maximum_absolute_unsafe_rate"],
            "measurement": forecast_safety,
        },
        "target_progress": {
            "passed": float(progress["lower_95"])
            >= thresholds["target_progress_gain_vs_current_m"],
            "measurement": progress,
        },
        "absolute_target_progress": {
            "passed": float(forecast_progress["lower_95"])
            >= thresholds["minimum_absolute_target_progress_m"],
            "measurement": forecast_progress,
        },
        "falsification_controls": {
            "passed": all(
                float(comparisons[name]["upper_95"]) < 0.0
                for name in (
                    "forecast_vs_task_action",
                    "forecast_vs_hold_blind",
                    "forecast_vs_shuffled",
                    "forecast_vs_random",
                )
            ),
            "measurement": {
                name: dict(comparisons[name])
                for name in (
                    "forecast_vs_task_action",
                    "forecast_vs_hold_blind",
                    "forecast_vs_shuffled",
                    "forecast_vs_random",
                )
            },
        },
    }
    if not discrimination_applicable:
        return "CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA", gates
    required_passes = [
        bool(row["passed"])
        for row in gates.values()
        if row.get("applicable", True)
    ]
    verdict = (
        "CHECKPOINT_MEASUREMENT_PASSES_PREREGISTERED_GATES"
        if all(required_passes)
        else "CHECKPOINT_MEASUREMENT_FAILS_PREREGISTERED_GATES"
    )
    return verdict, gates


def _recomputed_direct_summary_v1(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    rows = report.get("group_results")
    separable = report.get("separable_group_results")
    if not isinstance(rows, list) or not rows or not isinstance(separable, list):
        raise BoundedBranchEvaluationError("direct raw group results changed")
    row_by_state: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise BoundedBranchEvaluationError("direct raw row is malformed")
        state_id = str(row.get("state_id"))
        if state_id in row_by_state:
            raise BoundedBranchEvaluationError("direct raw state repeats")
        required = {
            "state_id",
            "scene_id",
            "family",
            "matched_branch_error",
            "physical_oracle_dense_ranks",
            "physical_oracle_class_count",
            "joint_contrast_signatures_by_action",
            "eligible_action_ids",
            "separable_actions",
            "separable_action_fraction",
        }
        if set(row) != required:
            raise BoundedBranchEvaluationError("direct raw row schema changed")
        error = float(row["matched_branch_error"])
        separable_actions = row["separable_actions"]
        physical_ranks = row["physical_oracle_dense_ranks"]
        signatures = row["joint_contrast_signatures_by_action"]
        eligible_action_ids = row["eligible_action_ids"]
        if (
            not isinstance(signatures, list)
            or not isinstance(eligible_action_ids, list)
        ):
            raise BoundedBranchEvaluationError(
                "direct joint-discrimination evidence changed"
            )
        recomputed_eligible = _eligible_action_ids_from_joint_signatures_v1(
            signatures
        )
        if (
            not math.isfinite(error)
            or error < 0.0
            or not isinstance(physical_ranks, list)
            or len(physical_ranks) != ACTION_COUNT
            or any(type(value) is not int or value < 0 for value in physical_ranks)
            or set(physical_ranks) != set(range(len(set(physical_ranks))))
            or [
                int(signature["physical_outcome_dense_rank"])
                for signature in signatures
            ]
            != physical_ranks
            or type(separable_actions) is not int
            or not 0 <= separable_actions <= ACTION_COUNT
            or eligible_action_ids != recomputed_eligible
            or separable_actions != len(recomputed_eligible)
            or float(row["separable_action_fraction"])
            != separable_actions / ACTION_COUNT
            or type(row["physical_oracle_class_count"]) is not int
            or not 1 <= int(row["physical_oracle_class_count"]) <= ACTION_COUNT
            or int(row["physical_oracle_class_count"])
            != len(set(physical_ranks))
            or row.get("family") not in FAMILIES
            or not isinstance(row.get("scene_id"), str)
            or not row["scene_id"]
        ):
            raise BoundedBranchEvaluationError("direct raw row value changed")
        row_by_state[state_id] = row
    separable_by_state: dict[str, Mapping[str, Any]] = {}
    for row in separable:
        if not isinstance(row, Mapping):
            raise BoundedBranchEvaluationError("direct separable row is malformed")
        state_id = str(row.get("state_id"))
        source = row_by_state.get(state_id)
        extra = {
            "nearest_nonequivalent_branch_error",
            "branch_margin",
            "equivalence_aware_retrieval_accuracy",
            "equivalence_adjusted_chance",
            "retrieval_advantage",
            "mean_equivalence_class_size",
        }
        if (
            source is None
            or state_id in separable_by_state
            or set(row) != set(source) | extra
            or any(row[key] != source[key] for key in source)
            or int(source["separable_actions"]) <= 0
        ):
            raise BoundedBranchEvaluationError("direct separable row changed")
        for field in extra:
            if not math.isfinite(float(row[field])):
                raise BoundedBranchEvaluationError(
                    "direct separable measurement is nonfinite"
                )
        accuracy = float(row["equivalence_aware_retrieval_accuracy"])
        chance = float(row["equivalence_adjusted_chance"])
        if (
            not 0.0 <= accuracy <= 1.0
            or not 0.0 < chance <= 1.0
            or float(row["retrieval_advantage"]) != accuracy - chance
        ):
            raise BoundedBranchEvaluationError("direct retrieval row changed")
        separable_by_state[state_id] = row
    expected_separable = {
        state_id for state_id, row in row_by_state.items()
        if int(row["separable_actions"]) > 0
    }
    if set(separable_by_state) != expected_separable:
        raise BoundedBranchEvaluationError("direct separable state set changed")
    all_scene_keys = {
        (str(row["family"]), str(row["scene_id"])) for row in rows
    }
    separable_scene_keys = {
        (str(row["family"]), str(row["scene_id"])) for row in separable
    }
    return {
        "matched_branch_error": float(
            np.mean([float(row["matched_branch_error"]) for row in rows])
        ),
        "separable_action_coverage": sum(
            int(row["separable_actions"]) for row in rows
        ) / (len(rows) * ACTION_COUNT),
        "separable_state_coverage": len(separable) / len(rows),
        "separable_scene_coverage": len(separable_scene_keys) / len(all_scene_keys),
        "branch_margin": (
            float(np.mean([float(row["branch_margin"]) for row in separable]))
            if separable else None
        ),
        "equivalence_aware_retrieval_accuracy": (
            float(np.mean([
                float(row["equivalence_aware_retrieval_accuracy"])
                for row in separable
            ])) if separable else None
        ),
        "retrieval_advantage": (
            float(np.mean([float(row["retrieval_advantage"]) for row in separable]))
            if separable else None
        ),
    }


def _recompute_checkpoint_report_v1(
    report: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Recompute every checkpoint gate from raw rows and reject labels by fiat."""

    direct = report.get("direct_matched_branch_fidelity")
    physical = report.get("physical_arms")
    identity = report.get("checkpoint_panel_identity")
    if not isinstance(identity, Mapping):
        raise BoundedBranchEvaluationError("checkpoint panel identity changed")
    interval_alpha = checkpoint_interval_alpha_v1(str(identity.get("arm")))
    if not isinstance(direct, Mapping) or set(direct) != {
        "forecast", "shuffled", "hold_blind"
    } or not isinstance(physical, Mapping):
        raise BoundedBranchEvaluationError("checkpoint raw measurements changed")
    oracle_contract = {
        "basis": "equal_frozen_calibration_tolerance_aware_physical_oracle_dense_rank",
        "model_dependent": False,
        "latent_proximity_used": False,
        "query_eligibility": (
            "at_least_one_physically_nonequivalent_alternative_and_every_"
            "physically_nonequivalent_alternative_has_different_executed_"
            "tape_and_stored_rgb_raw_pixel_identity"
        ),
    }
    coverage_identities: dict[str, list[tuple[Any, ...]]] = {}
    for arm, direct_report in direct.items():
        if not isinstance(direct_report, Mapping) or set(direct_report) != {
            "summary", "outcome_equivalence", "group_results",
            "separable_group_results",
        }:
            raise BoundedBranchEvaluationError("direct report schema changed")
        if direct_report["outcome_equivalence"] != oracle_contract:
            raise BoundedBranchEvaluationError("physical outcome equivalence changed")
        recomputed_summary = _recomputed_direct_summary_v1(direct_report)
        if recomputed_summary != direct_report["summary"]:
            raise BoundedBranchEvaluationError("direct summary was not derived from rows")
        coverage_identities[arm] = [
            (
                row["state_id"], row["scene_id"], row["family"],
                tuple(row["physical_oracle_dense_ranks"]),
                tuple(row["eligible_action_ids"]),
                json.dumps(
                    row["joint_contrast_signatures_by_action"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                row["physical_oracle_class_count"], row["separable_actions"],
                row["separable_action_fraction"],
            )
            for row in direct_report["group_results"]
        ]
    if any(
        coverage_identities[arm] != coverage_identities["forecast"]
        for arm in ("shuffled", "hold_blind")
    ):
        raise BoundedBranchEvaluationError(
            "model-independent physical separability changed across controls"
        )
    required_physical = {
        "true_future_ceiling", "current_state_action", "task_action_only",
        "forecast", "shuffled", "hold_blind",
    }
    if not required_physical.issubset(physical):
        raise BoundedBranchEvaluationError("physical raw arms are incomplete")
    pairs = {
        "ceiling_vs_current": ("true_future_ceiling", "current_state_action"),
        "forecast_vs_current": ("forecast", "current_state_action"),
        "forecast_vs_task_action": ("forecast", "task_action_only"),
        "forecast_vs_hold_blind": ("forecast", "hold_blind"),
        "forecast_vs_shuffled": ("forecast", "shuffled"),
    }
    comparisons = {
        name: _paired(
            physical[candidate]["group_results"],
            physical[baseline]["group_results"],
            field="normalized_rank_regret",
            resamples=DEFAULT_RESAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED,
            two_sided_alpha=interval_alpha,
        )
        for name, (candidate, baseline) in pairs.items()
    }
    forecast_rows = physical["forecast"].get("group_results")
    if not isinstance(forecast_rows, list) or not forecast_rows:
        raise BoundedBranchEvaluationError("forecast physical rows changed")
    random_rows = [
        {
            **row,
            "normalized_rank_regret": row["random_expected_normalized_rank_regret"],
        }
        for row in forecast_rows
    ]
    comparisons["forecast_vs_random"] = _paired(
        forecast_rows,
        random_rows,
        field="normalized_rank_regret",
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
        two_sided_alpha=interval_alpha,
    )
    if comparisons != report.get("paired_scene_cluster_comparisons"):
        raise BoundedBranchEvaluationError(
            "checkpoint comparisons were not derived from raw rows"
        )
    status, gates = preregistered_verdict_v1(
        direct_forecast=direct["forecast"],
        direct_shuffled=direct["shuffled"],
        physical_arms=physical,
        comparisons=comparisons,
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
        two_sided_alpha=interval_alpha,
    )
    if gates != report.get("preregistered_gates") or status != report.get(
        "checkpoint_gate_status"
    ):
        raise BoundedBranchEvaluationError(
            "checkpoint gate label was not derived from raw measurements"
        )
    return status, gates, comparisons


def _factorial_interval_v1(
    reports_by_arm: Mapping[str, Mapping[str, Any]],
    *,
    source: str,
    field: str,
    contrast: str,
) -> dict[str, Any]:
    if source == "direct":
        arm_rows = {
            arm: report["direct_matched_branch_fidelity"]["forecast"][
                "group_results"
            ]
            for arm, report in reports_by_arm.items()
        }
    elif source == "physical":
        arm_rows = {
            arm: report["physical_arms"]["forecast"]["group_results"]
            for arm, report in reports_by_arm.items()
        }
    else:
        raise AssertionError(source)
    indexed = {
        arm: {str(row["state_id"]): row for row in rows}
        for arm, rows in arm_rows.items()
    }
    state_ids = set(indexed["masked_plain"])
    if any(set(rows) != state_ids for rows in indexed.values()):
        raise BoundedBranchEvaluationError("cross-model paired states changed")
    coefficients = {
        "delta_main": {
            "masked_plain": -0.5, "masked_delta": 0.5,
            "full_plain": -0.5, "full_delta": 0.5,
        },
        "spatial_main": {
            "masked_plain": -0.5, "masked_delta": -0.5,
            "full_plain": 0.5, "full_delta": 0.5,
        },
        "delta_within_masked": {"masked_plain": -1.0, "masked_delta": 1.0},
        "delta_within_full": {"full_plain": -1.0, "full_delta": 1.0},
        "full_within_plain": {"masked_plain": -1.0, "full_plain": 1.0},
        "full_within_delta": {"masked_delta": -1.0, "full_delta": 1.0},
        "interaction": {
            "masked_plain": 1.0, "masked_delta": -1.0,
            "full_plain": -1.0, "full_delta": 1.0,
        },
    }
    if contrast not in coefficients:
        raise AssertionError(contrast)
    rows = []
    for state_id in sorted(state_ids):
        reference = indexed["masked_plain"][state_id]
        for arm_rows_by_state in indexed.values():
            candidate = arm_rows_by_state[state_id]
            if (
                candidate.get("scene_id") != reference.get("scene_id")
                or candidate.get("family") != reference.get("family")
            ):
                raise BoundedBranchEvaluationError(
                    "cross-model scene/family identity changed"
                )
        effect = sum(
            coefficient * float(indexed[arm][state_id][field])
            for arm, coefficient in coefficients[contrast].items()
        )
        rows.append({
            "state_id": state_id,
            "scene_id": reference["scene_id"],
            "family": reference["family"],
            "effect": effect,
        })
    result = scene_cluster_interval_v1(
        rows,
        field="effect",
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    result.update({
        "source": source,
        "source_field": field,
        "contrast": contrast,
        "direction": "negative_favors_named_mechanism",
    })
    return result


def aggregate_model_panel_v1(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute all 12 members, then separate usefulness from mechanism."""

    if len(reports) != len(MODEL_ARMS) * len(TRAINING_SEEDS):
        raise BoundedBranchEvaluationError("model panel must contain exactly 12 reports")
    required_report = {
        "schema", "status", "citable_as_scientific_evidence",
        "authorizes_retry_or_resume", "scientific_verdict_emitted",
        "pilot_manifest_binding", "pilot_terminal_gate", "checkpoint",
        "checkpoint_binding", "checkpoint_panel_identity",
        "training_scene_separation", "model_label", "model_identity",
        "source_bindings", "evaluation_contract", "latent_standardizer",
        "physical_outcome_equivalence", "direct_matched_branch_fidelity",
        "physical_arms", "paired_scene_cluster_comparisons",
        "preregistered_gates", "checkpoint_gate_status",
    }
    expected_oracle = {
        "basis": "equal_frozen_calibration_tolerance_aware_physical_oracle_dense_rank",
        "model_dependent": False,
        "latent_proximity_used": False,
        "source": "bound_branch_oracle_dense_rank",
    }
    by_member: dict[tuple[str, int], Mapping[str, Any]] = {}
    common: dict[str, Any] | None = None
    common_coverage: list[tuple[Any, ...]] | None = None
    for report in reports:
        identity = report.get("checkpoint_panel_identity") if isinstance(report, Mapping) else None
        if (
            not isinstance(report, Mapping)
            or set(report) != required_report
            or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or report.get("citable_as_scientific_evidence") is not False
            or report.get("authorizes_retry_or_resume") is not False
            or report.get("scientific_verdict_emitted") is not False
            or report.get("evaluation_contract") != evaluation_contract_v1()
            or report.get("physical_outcome_equivalence") != expected_oracle
            or not isinstance(identity, Mapping)
        ):
            raise BoundedBranchEvaluationError("model-panel member report changed")
        arm = identity.get("arm")
        seed = identity.get("seed")
        if arm not in MODEL_ARMS or seed not in TRAINING_SEEDS:
            raise BoundedBranchEvaluationError("model-panel member identity changed")
        key = (str(arm), int(seed))
        if key in by_member:
            raise BoundedBranchEvaluationError("model-panel member repeats")
        _recompute_checkpoint_report_v1(report)
        training = report.get("training_scene_separation")
        panel_bindings = (
            training.get("checkpoint_panel_bindings")
            if isinstance(training, Mapping) else None
        )
        binding_key = f"{arm}/seed_{seed}"
        if (
            not isinstance(panel_bindings, Mapping)
            or panel_bindings.get(binding_key) != report.get("checkpoint_binding")
        ):
            raise BoundedBranchEvaluationError(
                "model-panel member is not bound by the progression analysis"
            )
        terminal_gate = report.get("pilot_terminal_gate")
        if not isinstance(terminal_gate, Mapping):
            raise BoundedBranchEvaluationError("pilot terminal gate changed")
        frozen_panel = terminal_gate.get("model_panel_freeze")
        _require_model_panel_lineage_match_v1(frozen_panel, training)
        selected_common = {
            "pilot_manifest_binding": report["pilot_manifest_binding"],
            "pilot_terminal_gate": terminal_gate,
            "progression_analysis_binding": training.get("progression_analysis_binding"),
            "training_result_binding": training.get("training_result_binding"),
            "progression_proxy_routing": training.get("progression_proxy_routing"),
            "checkpoint_panel_bindings": panel_bindings,
            "source_bindings": report["source_bindings"],
            "evaluation_contract": report["evaluation_contract"],
            "latent_standardizer": report["latent_standardizer"],
            "physical_outcome_equivalence": report["physical_outcome_equivalence"],
        }
        if common is None:
            common = selected_common
        elif selected_common != common:
            raise BoundedBranchEvaluationError(
                "model-panel members do not share one frozen experiment identity"
            )
        coverage = [
            (
                row["state_id"], row["scene_id"], row["family"],
                tuple(row["physical_oracle_dense_ranks"]),
                tuple(row["eligible_action_ids"]),
                json.dumps(
                    row["joint_contrast_signatures_by_action"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                row["physical_oracle_class_count"], row["separable_actions"],
            )
            for row in report["direct_matched_branch_fidelity"]["forecast"][
                "group_results"
            ]
        ]
        if common_coverage is None:
            common_coverage = coverage
        elif coverage != common_coverage:
            raise BoundedBranchEvaluationError(
                "physical separability changed across model-panel members"
            )
        by_member[key] = report
    expected = {(arm, seed) for arm in MODEL_ARMS for seed in TRAINING_SEEDS}
    if set(by_member) != expected or common is None:
        raise BoundedBranchEvaluationError("model panel is incomplete")
    proxy = common.get("progression_proxy_routing")
    proxy_decision = proxy.get("decision") if isinstance(proxy, Mapping) else None
    if proxy_decision not in {
        "DELTA_PROXY_MEANINGFUL", "DELTA_PROXY_NOT_MEANINGFUL"
    }:
        raise BoundedBranchEvaluationError("progression proxy routing changed")

    checkpoint_statuses = {
        arm: {
            str(seed): str(by_member[(arm, seed)]["checkpoint_gate_status"])
            for seed in TRAINING_SEEDS
        }
        for arm in MODEL_ARMS
    }
    arm_usefulness = {}
    for arm in MODEL_ARMS:
        statuses = checkpoint_statuses[arm]
        if any(value == "CHECKPOINT_MEASUREMENT_INCONCLUSIVE_DATA" for value in statuses.values()):
            decision = "INCONCLUSIVE_DATA"
        elif all(value == "CHECKPOINT_MEASUREMENT_PASSES_PREREGISTERED_GATES" for value in statuses.values()):
            decision = "USEFUL_ALL_THREE_SEEDS"
        else:
            decision = "USEFULNESS_NOT_ESTABLISHED_ALL_THREE_SEEDS"
        arm_usefulness[arm] = {
            "seed_statuses": statuses,
            "all_three_seed_decision": decision,
        }

    contrasts = (
        "delta_main", "spatial_main", "delta_within_masked",
        "delta_within_full", "full_within_plain", "full_within_delta",
        "interaction",
    )
    factorial: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        seed_reports = {arm: by_member[(arm, seed)] for arm in MODEL_ARMS}
        factorial[str(seed)] = {
            "direct_matched_branch_error": {
                contrast: _factorial_interval_v1(
                    seed_reports,
                    source="direct",
                    field="matched_branch_error",
                    contrast=contrast,
                )
                for contrast in contrasts
            },
            "physical_normalized_rank_regret_supportive": {
                contrast: _factorial_interval_v1(
                    seed_reports,
                    source="physical",
                    field="normalized_rank_regret",
                    contrast=contrast,
                )
                for contrast in contrasts
            },
        }

    treatment = MECHANISM_EFFECT_THRESHOLDS[
        "minimum_direct_error_treatment_reduction"
    ]
    noninferiority = MECHANISM_EFFECT_THRESHOLDS[
        "full_grid_direct_error_noninferiority_margin"
    ]

    def treatment_seed_pass(seed: int, contrast: str) -> bool:
        row = factorial[str(seed)]["direct_matched_branch_error"][contrast]
        return float(row["mean"]) < 0.0 and float(row["upper_95"]) <= -treatment

    def noninferiority_seed_pass(seed: int, contrast: str) -> bool:
        row = factorial[str(seed)]["direct_matched_branch_error"][contrast]
        return float(row["upper_95"]) <= noninferiority

    delta_seed_passes = {
        str(seed): treatment_seed_pass(seed, "delta_main")
        for seed in TRAINING_SEEDS
    }
    spatial_seed_passes = {
        str(seed): treatment_seed_pass(seed, "spatial_main")
        for seed in TRAINING_SEEDS
    }
    full_plain_noninferiority = {
        str(seed): noninferiority_seed_pass(seed, "full_within_plain")
        for seed in TRAINING_SEEDS
    }
    full_delta_noninferiority = {
        str(seed): noninferiority_seed_pass(seed, "full_within_delta")
        for seed in TRAINING_SEEDS
    }
    delta_pass = all(delta_seed_passes.values())
    spatial_pass = all(spatial_seed_passes.values())
    if delta_pass and spatial_pass:
        mechanism = "BOTH_DELTA_AND_FULL_GRID_PRACTICAL_ALL_THREE_SEEDS"
    elif delta_pass:
        mechanism = "DELTA_ONLY_PRACTICAL_ALL_THREE_SEEDS"
    elif spatial_pass:
        mechanism = "FULL_GRID_ONLY_PRACTICAL_ALL_THREE_SEEDS"
    else:
        mechanism = "NEITHER_MECHANISM_PRACTICAL_ALL_THREE_SEEDS"
    if proxy_decision == "DELTA_PROXY_NOT_MEANINGFUL":
        delta_scale_route = "STOP_DELTA_OBSERVATIONAL_SCALING_PROXY_NOT_MEANINGFUL"
    elif delta_pass:
        delta_scale_route = "ADVANCE_DELTA_CAUSAL_MECHANISM_ONLY"
    else:
        delta_scale_route = "STOP_DELTA_OBSERVATIONAL_SCALING_NO_PRACTICAL_CAUSAL_EFFECT"
    if not delta_pass and not spatial_pass:
        next_data_route = (
            "STOP_OBSERVATIONAL_MECHANISM_TUNING_AND_COLLECT_MATCHED_BRANCH_"
            "TRAINING_DATA_THEN_COMPARE_CONVENTIONAL_AND_DREAMER_BASELINES"
        )
    else:
        next_data_route = "PURSUE_ONLY_MECHANISMS_WITH_ALL_THREE_SEED_PRACTICAL_EFFECT"

    full_grid_rollout = {}
    for cell, full_arm, noninferiority_rows in (
        ("plain", "full_plain", full_plain_noninferiority),
        ("delta", "full_delta", full_delta_noninferiority),
    ):
        useful = arm_usefulness[full_arm]["all_three_seed_decision"] == (
            "USEFUL_ALL_THREE_SEEDS"
        )
        delta_control_only = (
            cell == "delta"
            and proxy_decision == "DELTA_PROXY_NOT_MEANINGFUL"
        )
        eligible = (
            useful
            and all(noninferiority_rows.values())
            and not delta_control_only
        )
        full_grid_rollout[cell] = {
            "full_arm": full_arm,
            "useful_all_three_seeds": useful,
            "noninferior_all_three_seeds": all(noninferiority_rows.values()),
            "seed_noninferiority_passes": noninferiority_rows,
            "delta_proxy_control_only": delta_control_only,
            "decision": (
                "ELIGIBLE_ONLY_FOR_SEPARATELY_PREREGISTERED_BLIND_ROLLOUT"
                if eligible else "STOP_NOT_ELIGIBLE_FOR_BLIND_ROLLOUT"
            ),
        }

    plain_decisions = {
        arm: arm_usefulness[arm]["all_three_seed_decision"]
        for arm in PRIMARY_PLAIN_ARMS
    }
    passing_plain_arms = [
        arm for arm, decision in plain_decisions.items()
        if decision == "USEFUL_ALL_THREE_SEEDS"
    ]
    if passing_plain_arms:
        global_verdict = "USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_DEVELOPMENT_ONLY"
    elif any(
        decision == "INCONCLUSIVE_DATA" for decision in plain_decisions.values()
    ):
        global_verdict = "INCONCLUSIVE_DATA_PHYSICAL_ACTION_SEPARABILITY"
    else:
        global_verdict = "USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_NOT_ESTABLISHED"

    def plain_arm_metric(arm: str, name: str) -> dict[str, Any]:
        values = []
        for seed in TRAINING_SEEDS:
            report = by_member[(arm, seed)]
            if name == "physical_regret_effect":
                value = report["paired_scene_cluster_comparisons"][
                    "forecast_vs_current"
                ]["mean"]
            elif name == "matched_branch_error":
                value = report["direct_matched_branch_fidelity"]["forecast"][
                    "summary"
                ]["matched_branch_error"]
            else:
                raise AssertionError(name)
            values.append(float(value))
        vector = np.asarray(values, dtype=np.float64)
        return {
            "values_by_seed": dict(zip(map(str, TRAINING_SEEDS), values, strict=True)),
            "mean": float(vector.mean()),
            "sample_standard_deviation": float(vector.std(ddof=1)),
            "minimum": float(vector.min()),
            "maximum": float(vector.max()),
            "interval_not_claimed": "three_fixed_seeds_are_reported_raw",
        }

    return {
        "schema": PANEL_REPORT_SCHEMA,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "all_fixed_panel_members_reported": True,
        "common_experiment_identity": common,
        "member_identities": [
            {
                "arm": arm,
                "seed": seed,
                "checkpoint_binding": by_member[(arm, seed)]["checkpoint_binding"],
                "checkpoint_gate_status": checkpoint_statuses[arm][str(seed)],
            }
            for arm in MODEL_ARMS for seed in TRAINING_SEEDS
        ],
        "primary_plain_arm_family": {
            "arms": list(PRIMARY_PLAIN_ARMS),
            "multiplicity_policy": (
                "bonferroni_two_arm_family_one_sided_family_alpha_0.025"
            ),
            "all_three_seed_arm_decisions": plain_decisions,
            "passing_plain_arms": passing_plain_arms,
            "decision": global_verdict,
        },
        "delta_mechanism_controls": {
            "arms": list(MECHANISM_CONTROL_ARMS),
            "planning_usefulness_claim_eligible": False,
            "reason": (
                "delta_proxy_not_meaningful_negative_and_mechanism_controls_only"
                if proxy_decision == "DELTA_PROXY_NOT_MEANINGFUL"
                else "outside_arm_agnostic_plain_primary_family"
            ),
            "measurement_results": {
                arm: arm_usefulness[arm] for arm in MECHANISM_CONTROL_ARMS
            },
        },
        "all_arm_measurements": arm_usefulness,
        "primary_plain_seed_gate_passes": {
            arm: {
                str(seed): checkpoint_statuses[arm][str(seed)]
                == "CHECKPOINT_MEASUREMENT_PASSES_PREREGISTERED_GATES"
                for seed in TRAINING_SEEDS
            }
            for arm in PRIMARY_PLAIN_ARMS
        },
        "primary_plain_seed_uncertainty": {
            arm: {
                name: plain_arm_metric(arm, name)
                for name in ("physical_regret_effect", "matched_branch_error")
            }
            for arm in PRIMARY_PLAIN_ARMS
        },
        "cross_model_paired_effects": factorial,
        "factorial_analysis": {
            "interaction_policy": "descriptive_only",
            "mechanism_surface": "direct_matched_branch_error",
            "physical_regret_role": "supportive_planning_utility_not_mechanism_gate",
        },
        "mechanism_adjudication": {
            "thresholds": dict(MECHANISM_EFFECT_THRESHOLDS),
            "delta_seed_passes": delta_seed_passes,
            "spatial_seed_passes": spatial_seed_passes,
            "all_three_seed_decision": mechanism,
            "progression_proxy_routing": proxy,
            "delta_observational_scale_route": delta_scale_route,
            "full_grid_rollout": full_grid_rollout,
            "next_data_and_baseline_route": next_data_route,
        },
        "global_verdict": global_verdict,
    }


def evaluate_bound_model_v1(
    *,
    pilot_root: Path,
    manifest_byte_count: int,
    manifest_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    progression_analysis: Path,
    progression_analysis_sha256: str,
    progression_analysis_byte_count: int,
    pilot_terminal: Path,
    pilot_terminal_sha256: str,
    pilot_terminal_byte_count: int,
    pilot_terminal_review: Path,
    pilot_terminal_review_sha256: str,
    pilot_terminal_review_byte_count: int,
    expected_arm: str,
    expected_training_seed: int,
    device_name: str,
) -> dict[str, Any]:
    _reject_protected_path(pilot_root, label="pilot root")
    _reject_protected_path(checkpoint, label="checkpoint")
    bundle = load_bound_pilot_v1(
        pilot_root,
        expected_manifest_byte_count=manifest_byte_count,
        expected_manifest_sha256=manifest_sha256,
    )
    if len(bundle.groups_by_role["train"]) != 128 or len(bundle.groups_by_role["eval"]) != 128:
        raise BoundedBranchEvaluationError("claim experiment requires exact 128/128 role split")
    pilot_terminal_gate = load_and_validate_pilot_terminal_gate_v1(
        pilot_terminal,
        expected_terminal_sha256=pilot_terminal_sha256,
        expected_terminal_byte_count=pilot_terminal_byte_count,
        review_path=pilot_terminal_review,
        expected_review_sha256=pilot_terminal_review_sha256,
        expected_review_byte_count=pilot_terminal_review_byte_count,
        pilot_manifest_binding=bundle.manifest_binding,
    )
    pilot_scene_ids = {
        group.scene_id
        for role in ("train", "eval")
        for group in bundle.groups_by_role[role]
    }
    import torch
    from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as pilot_consumer
    from scripts import dev_probe_counterfactual_action_fidelity as probe

    selected_checkpoint = probe.require_development_checkpoint(checkpoint)
    _analysis_document, training_separation = load_and_validate_progression_analysis_v1(
        progression_analysis,
        expected_sha256=progression_analysis_sha256,
        expected_byte_count=progression_analysis_byte_count,
        selected_checkpoint=selected_checkpoint,
        expected_arm=expected_arm,
        expected_seed=expected_training_seed,
        pilot_scene_ids=pilot_scene_ids,
    )
    frozen_model_panel = pilot_terminal_gate.get("model_panel_freeze")
    _require_model_panel_lineage_match_v1(
        frozen_model_panel, training_separation
    )
    checkpoint_binding = probe.file_binding(selected_checkpoint)
    panel_key = f"{expected_arm}/seed_{expected_training_seed}"
    if (
        checkpoint_binding["sha256"] != checkpoint_sha256
        or checkpoint_binding
        != training_separation["checkpoint_panel_bindings"][panel_key]
    ):
        raise BoundedBranchEvaluationError("checkpoint binding changed before model load")
    checkpoint_payload = torch.load(
        selected_checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_panel_identity = validate_progression_snapshot_metadata_v1(
        checkpoint_payload,
        expected_arm=expected_arm,
        expected_seed=expected_training_seed,
        expected_update=EXPECTED_TERMINAL_UPDATE,
    )
    if probe.file_binding(selected_checkpoint) != checkpoint_binding:
        raise BoundedBranchEvaluationError("checkpoint changed while metadata was loaded")
    device = torch.device(device_name)
    source_bindings = [
        probe.file_binding(Path(path))
        for path in (
            __file__,
            base.__file__,
            progression_analyzer.__file__,
            pilot_consumer.__file__,
            probe.__file__,
            probe.model_module.__file__,
            probe.evaluation.__file__,
            probe.metrics.__file__,
            probe.h6.__file__,
            probe.trainer.__file__,
        )
    ]
    reference_key = (
        f"{LATENT_STANDARDIZER_REFERENCE_ARM}/"
        f"seed_{LATENT_STANDARDIZER_REFERENCE_SEED}"
    )
    reference_binding = training_separation["checkpoint_panel_bindings"].get(
        reference_key
    )
    if not isinstance(reference_binding, Mapping):
        raise BoundedBranchEvaluationError(
            "common latent-standardizer reference checkpoint is absent"
        )
    reference_checkpoint = probe.require_development_checkpoint(
        Path(str(reference_binding["path"]))
    )
    if probe.file_binding(reference_checkpoint) != reference_binding:
        raise BoundedBranchEvaluationError(
            "common latent-standardizer checkpoint binding changed"
        )
    reference_payload = torch.load(
        reference_checkpoint, map_location="cpu", weights_only=True
    )
    validate_progression_snapshot_metadata_v1(
        reference_payload,
        expected_arm=LATENT_STANDARDIZER_REFERENCE_ARM,
        expected_seed=LATENT_STANDARDIZER_REFERENCE_SEED,
        expected_update=EXPECTED_TERMINAL_UPDATE,
    )
    reference_model, reference_label, reference_model_identity = probe.build_model(
        reference_checkpoint,
        device,
        expected_checkpoint_sha256=str(reference_binding["sha256"]),
        expected_update=EXPECTED_TERMINAL_UPDATE,
    )
    reference_features = base._extract_features(  # noqa: SLF001
        bundle, reference_model, device
    )
    mean, scale = fit_train_latent_standardizer_v1(
        reference_features["train"]["true_future_ceiling"]
    )
    if checkpoint_binding == reference_binding:
        model = reference_model
        label = reference_label
        model_identity = reference_model_identity
        features = reference_features
    else:
        del reference_model, reference_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, label, model_identity = probe.build_model(
            selected_checkpoint,
            device,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_update=EXPECTED_TERMINAL_UPDATE,
        )
        features = base._extract_features(bundle, model, device)  # noqa: SLF001
    if (
        probe.file_binding(reference_checkpoint) != reference_binding
        or probe.file_binding(selected_checkpoint) != checkpoint_binding
    ):
        raise BoundedBranchEvaluationError(
            "checkpoint changed after common-standardizer/model load"
        )
    # Ordering is claim-bearing: ``_extract_features`` calls ``probe.decode``
    # for every context/target, which calls ``read_bound_rgb_bytes_v1`` and
    # verifies the exact decoded raw-pixel SHA before this joint-eligibility
    # gate can inspect any branch pixel-class identity.
    direct = {
        arm: direct_matched_branch_metrics_v1(
            groups=bundle.groups_by_role["eval"],
            predicted_features_by_state=features["eval"][arm],
            true_features_by_state=features["eval"]["true_future_ceiling"],
            train_mean=mean,
            train_scale=scale,
        )
        for arm in ("forecast", "shuffled", "hold_blind")
    }
    arm_reports: dict[str, dict[str, Any]] = {}
    for arm in base.ARM_NAMES:
        train_x = [[] for _ in range(ACTION_COUNT)]
        train_y = [[] for _ in range(ACTION_COUNT)]
        for group in bundle.groups_by_role["train"]:
            ranks = [branch.oracle_dense_rank for branch in group.branches]
            denominator = max(1, max(ranks))
            for action_id in range(ACTION_COUNT):
                train_x[action_id].append(features["train"][arm][group.state_id][action_id])
                train_y[action_id].append(ranks[action_id] / denominator)
        readout = base.fit_action_specific_ridge_readouts_v1(
            [np.stack(rows) for rows in train_x],
            train_y,
            ridge_lambda=RIDGE_LAMBDA,
        )
        scores = {
            group.state_id: base.predict_action_specific_scores_v1(
                readout, features["eval"][arm][group.state_id]
            ).tolist()
            for group in bundle.groups_by_role["eval"]
        }
        arm_reports[arm] = base.selection_metrics_v1(
            bundle.groups_by_role["eval"], scores
        )
        arm_reports[arm]["readout_identity_sha256"] = readout.identity_sha256
    forecast_rows = arm_reports["forecast"]["group_results"]
    random_rows = [
        {**row, "normalized_rank_regret": row["random_expected_normalized_rank_regret"]}
        for row in forecast_rows
    ]
    pairs = {
        "ceiling_vs_current": ("true_future_ceiling", "current_state_action"),
        "forecast_vs_current": ("forecast", "current_state_action"),
        "forecast_vs_task_action": ("forecast", "task_action_only"),
        "forecast_vs_hold_blind": ("forecast", "hold_blind"),
        "forecast_vs_shuffled": ("forecast", "shuffled"),
    }
    interval_alpha = checkpoint_interval_alpha_v1(expected_arm)
    comparisons = {
        name: _paired(
            arm_reports[candidate]["group_results"],
            arm_reports[baseline_name]["group_results"],
            field="normalized_rank_regret",
            resamples=DEFAULT_RESAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED,
            two_sided_alpha=interval_alpha,
        )
        for name, (candidate, baseline_name) in pairs.items()
    }
    comparisons["forecast_vs_random"] = _paired(
        forecast_rows,
        random_rows,
        field="normalized_rank_regret",
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
        two_sided_alpha=interval_alpha,
    )
    checkpoint_gate_status, gates = preregistered_verdict_v1(
        direct_forecast=direct["forecast"],
        direct_shuffled=direct["shuffled"],
        physical_arms=arm_reports,
        comparisons=comparisons,
        resamples=DEFAULT_RESAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
        two_sided_alpha=interval_alpha,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "scientific_verdict_emitted": False,
        "pilot_manifest_binding": dict(bundle.manifest_binding),
        "pilot_terminal_gate": pilot_terminal_gate,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_panel_identity": checkpoint_panel_identity,
        "training_scene_separation": training_separation,
        "model_label": label,
        "model_identity": model_identity,
        "source_bindings": source_bindings,
        "evaluation_contract": evaluation_contract_v1(),
        "latent_standardizer": {
            "fit_role": "train",
            "reference_arm": LATENT_STANDARDIZER_REFERENCE_ARM,
            "reference_seed": LATENT_STANDARDIZER_REFERENCE_SEED,
            "reference_checkpoint_binding": dict(reference_binding),
            "dimensions": int(mean.size),
            "mean_sha256": __import__("hashlib").sha256(np.ascontiguousarray(mean.astype("<f8")).tobytes()).hexdigest(),
            "scale_sha256": __import__("hashlib").sha256(np.ascontiguousarray(scale.astype("<f8")).tobytes()).hexdigest(),
        },
        "physical_outcome_equivalence": {
            "basis": (
                "equal_frozen_calibration_tolerance_aware_physical_oracle_dense_rank"
            ),
            "model_dependent": False,
            "latent_proximity_used": False,
            "source": "bound_branch_oracle_dense_rank",
        },
        "direct_matched_branch_fidelity": direct,
        "physical_arms": arm_reports,
        "paired_scene_cluster_comparisons": comparisons,
        "preregistered_gates": gates,
        "checkpoint_gate_status": checkpoint_gate_status,
    }
    reloaded = load_bound_pilot_v1(
        pilot_root,
        expected_manifest_byte_count=manifest_byte_count,
        expected_manifest_sha256=manifest_sha256,
    )
    if (
        dict(reloaded.manifest_binding) != dict(bundle.manifest_binding)
        or dict(reloaded.rgb_manifest_binding) != dict(bundle.rgb_manifest_binding)
        or {
            role: dict(reloaded.role_bindings[role]) for role in ("train", "eval")
        }
        != {role: dict(bundle.role_bindings[role]) for role in ("train", "eval")}
    ):
        raise BoundedBranchEvaluationError("pilot receipts changed during evaluation")
    for artifact_id in sorted(reloaded.artifacts):
        read_bound_rgb_bytes_v1(reloaded, artifact_id)
    if probe.file_binding(selected_checkpoint) != checkpoint_binding:
        raise BoundedBranchEvaluationError("checkpoint changed during evaluation")
    if probe.file_binding(reference_checkpoint) != reference_binding:
        raise BoundedBranchEvaluationError(
            "common latent-standardizer checkpoint changed during evaluation"
        )
    probe.assert_file_bindings_unchanged(
        source_bindings, kind="bounded branch evaluator source"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--expected-pilot-manifest-byte-count", required=True, type=int)
    parser.add_argument("--expected-pilot-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--progression-analysis", required=True, type=Path)
    parser.add_argument("--expected-progression-analysis-sha256", required=True)
    parser.add_argument(
        "--expected-progression-analysis-byte-count", required=True, type=int
    )
    parser.add_argument("--pilot-terminal", required=True, type=Path)
    parser.add_argument("--expected-pilot-terminal-sha256", required=True)
    parser.add_argument("--expected-pilot-terminal-byte-count", required=True, type=int)
    parser.add_argument("--pilot-terminal-review", required=True, type=Path)
    parser.add_argument("--expected-pilot-terminal-review-sha256", required=True)
    parser.add_argument("--expected-pilot-terminal-review-byte-count", required=True, type=int)
    parser.add_argument("--expected-arm", required=True, choices=MODEL_ARMS)
    parser.add_argument("--expected-training-seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    for path, label in (
        (args.pilot_root, "pilot root"),
        (args.checkpoint, "checkpoint"),
        (args.progression_analysis, "progression analysis"),
        (args.pilot_terminal, "pilot terminal"),
        (args.pilot_terminal_review, "pilot terminal review"),
        (args.out, "output"),
    ):
        _reject_protected_path(path, label=label)
    for digest in (
        args.expected_pilot_manifest_sha256,
        args.expected_checkpoint_sha256,
        args.expected_progression_analysis_sha256,
        args.expected_pilot_terminal_sha256,
        args.expected_pilot_terminal_review_sha256,
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BoundedBranchEvaluationError("caller SHA-256 is malformed")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.out}")
    expected_output_name = (
        f"{args.expected_arm}_seed_{args.expected_training_seed}_"
        f"update_{EXPECTED_TERMINAL_UPDATE:06d}.json"
    )
    if args.out.name != expected_output_name:
        raise BoundedBranchEvaluationError("evaluation output name changed from model identity")
    if min(
        args.expected_progression_analysis_byte_count,
        args.expected_pilot_terminal_byte_count,
        args.expected_pilot_terminal_review_byte_count,
    ) <= 0:
        raise BoundedBranchEvaluationError("caller byte counts must be positive")
    report = evaluate_bound_model_v1(
        pilot_root=args.pilot_root,
        manifest_byte_count=args.expected_pilot_manifest_byte_count,
        manifest_sha256=args.expected_pilot_manifest_sha256,
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.expected_checkpoint_sha256,
        progression_analysis=args.progression_analysis,
        progression_analysis_sha256=args.expected_progression_analysis_sha256,
        progression_analysis_byte_count=args.expected_progression_analysis_byte_count,
        pilot_terminal=args.pilot_terminal,
        pilot_terminal_sha256=args.expected_pilot_terminal_sha256,
        pilot_terminal_byte_count=args.expected_pilot_terminal_byte_count,
        pilot_terminal_review=args.pilot_terminal_review,
        pilot_terminal_review_sha256=args.expected_pilot_terminal_review_sha256,
        pilot_terminal_review_byte_count=args.expected_pilot_terminal_review_byte_count,
        expected_arm=args.expected_arm,
        expected_training_seed=args.expected_training_seed,
        device_name="cuda",
    )
    from scripts import dev_probe_counterfactual_action_fidelity as probe

    output = probe.require_development_output(args.out)
    probe.write_json_atomic(output, report)
    print(json.dumps({
        "checkpoint_gate_status": report["checkpoint_gate_status"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedBranchEvaluationError",
    "CHANCE_RETRIEVAL",
    "EVALUATION_CONTRACT_SCHEMA",
    "MODEL_ARMS",
    "PANEL_REPORT_SCHEMA",
    "PRIMARY_PLAIN_ARMS",
    "MECHANISM_CONTROL_ARMS",
    "PILOT_TERMINAL_SCHEMA",
    "PILOT_TERMINAL_REVIEW_SCHEMA",
    "PROGRESSION_ANALYSIS_SCHEMA",
    "PROGRESSION_SNAPSHOT_SCHEMA",
    "RIDGE_LAMBDA",
    "EXPECTED_TERMINAL_UPDATE",
    "REPORT_SCHEMA",
    "USEFUL_EFFECT_THRESHOLDS",
    "TRAINING_SEEDS",
    "aggregate_model_panel_v1",
    "checkpoint_interval_alpha_v1",
    "direct_matched_branch_metrics_v1",
    "evaluation_contract_v1",
    "fit_train_latent_standardizer_v1",
    "load_and_validate_progression_analysis_v1",
    "load_and_validate_pilot_terminal_gate_v1",
    "preregistered_verdict_v1",
    "scene_cluster_interval_v1",
    "validate_progression_snapshot_metadata_v1",
]
