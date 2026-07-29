#!/usr/bin/env python3
"""One-shot V12 physical-evidence calibration using the frozen V4 protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence
    as _v4,
)


PREREGISTRATION_COMMIT = "c63e98162a1b03a33225e6e0a04b67a357c7ed89"
REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT = (
    "e983e0abd9349426f69262563e12d90a4488180e"
)
TERMINAL_RESULT_COMMIT = "c25b27cea61baf8ec2625f5995b59ce6d15e1dcb"
CANDIDATE_PREREGISTRATION_COMMIT = (
    "ae1568e8f434d715d379eefc3eaf644369154f76"
)
ADAPTER_MODULE = (
    "lewm.benchmarks.go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_physical_evidence_adapter"
)
ADAPTER_SOURCE_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_physical_evidence_adapter.py"
)
ADAPTER_SOURCE_SHA256 = (
    "96060ad821050e9958a9cef8383b0cc3f44206b4d53874f93989ace4ce057171"
)

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_physical_evidence_calibration/"
    "attempt_v1"
)
CANDIDATE_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition/attempt_v1"
)
CANDIDATE_RESULT_BYTE_COUNT = 74_226
CANDIDATE_RESULT_FILE_SHA256 = (
    "8268cabd23b57c66597c8ffd0f0b18b3eb296e9887acbc81363a666b70ff6ab6"
)
CANDIDATE_RESULT_CONTENT_SHA256 = (
    "6a6a4ef0d8545b1510f9830cb35ebf67ea3e8cdff25006b889b2ef6d0511feff"
)
CANDIDATE_CHECKPOINT_BYTE_COUNT = 29_676_571
CANDIDATE_CHECKPOINT_SHA256 = (
    "8212925759c0f496b0b6b1690168391d497c13688ba3cbb47b57640d173fe33f"
)
CANDIDATE_RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_result_v1"
)
CANDIDATE_CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_checkpoint_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_"
    "ternary_competition_physical_evidence_calibration_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_"
    "ternary_competition_physical_evidence_calibration_failure_v1"
)

ROLE_COUNTS = _v4.ROLE_COUNTS
ROLE_CELL_COUNTS = _v4.ROLE_CELL_COUNTS
BATCH_SIZE = _v4.BATCH_SIZE
FREE_CANDIDATES = _v4.FREE_CANDIDATES
OCCUPIED_CANDIDATES = _v4.OCCUPIED_CANDIDATES
UNKNOWN_CANDIDATES = _v4.UNKNOWN_CANDIDATES
OCCUPIED_DETECTION_CANDIDATES = _v4.OCCUPIED_DETECTION_CANDIDATES

# V12 changes only admission/loading and receipt names. All science remains the
# exact reviewed V4 implementation.
_canonical_bytes = _v4._canonical_bytes
_content_sha256 = _v4._content_sha256
_hashed = _v4._hashed
_parse_canonical = _v4._parse_canonical
_read_regular = _v4._read_regular
_atomic_write = _v4._atomic_write
_write_json = _v4._write_json
_build_data_boundary = _v4._build_data_boundary
_collect_role = _v4._collect_role
_fit_select_score = _v4._fit_select_score
_raw_access_snapshot = _v4._raw_access_snapshot

SOURCE_SHA256 = {
    "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py": (
        "6bcdb2b2551f0950d2abe120a9081eb6aeed19dd39207fe648bcc1d18e1c3426"
    ),
    "scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence.py": (
        "cee7c9c70e6bb9d2bacc6528ef77d009c80e2f484400de9f6445ebfd0c010313"
    ),
    "lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter.py": (
        "1ddbfd743d89614932823ae2247534ac6a76e2eaaf031911617a9311562b4b58"
    ),
    "lewm/hierarchical_probability_calibration.py": (
        "2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b"
    ),
    "lewm/benchmarks/traversability_metrics.py": (
        "97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396"
    ),
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py": (
        "79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6"
    ),
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py": (
        "33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000"
    ),
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py": (
        "ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c"
    ),
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py": (
        "8c35f0cbafe78185ac74d4412914c177de20f899b0f009a9b9dc7aafdf7695a5"
    ),
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py": (
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578"
    ),
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py": (
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a"
    ),
}

EXPECTED_ACCOUNTING_V12 = {
    "backward_calls": 4_000,
    "ema_steps": 1_000,
    "microbatch_graphs": 4_000,
    "optimizer_steps": 1_000,
    "predictor_forwards": 4_000,
    "predictor_objectives": 4_000,
    "presentations": 16_000,
    "updates": 1_000,
}
EXPECTED_GATE_THRESHOLDS_V12 = {
    "family_informative_utility_min": 0.70,
    "family_pair_concordance_min": 0.60,
    "family_selected_zero_prefix_rate_max": 0.20,
    "informative_utility_min": 0.85,
    "pair_concordance_min": 0.75,
    "positive_control_family_count_min": 6,
    "selected_zero_prefix_rate_max": 0.05,
    "semantic_balanced_accuracy_min": 0.80,
    "semantic_free_recall_min": 0.85,
    "semantic_occupied_recall_min": 0.70,
    "semantic_rough_occupied_recall_min": 0.65,
    "semantic_unknown_recall_min": 0.90,
}
_CONTROL_NAMES_V12 = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "train_action_mean_prior",
    "wrong_rgb",
)
EXPECTED_GATE_CHECKS_V12 = frozenset(
    {
        "all_family_pair_concordance",
        "all_family_utility",
        "all_family_zero_prefix_rate",
        "selection_informative_utility",
        "selection_pair_concordance",
        "selection_registered_families",
        "selection_zero_prefix_rate",
        "semantic_balanced_accuracy",
        "semantic_free_recall",
        "semantic_occupied_recall",
        "semantic_rough_occupied_recall",
        "semantic_unknown_recall",
    }
    | {
        f"{control}:{suffix}"
        for control in _CONTROL_NAMES_V12
        for suffix in (
            "positive_bootstrap_lower_95",
            "positive_equal_scene_delta",
            "positive_family_count",
        )
    }
)
EXPECTED_RESULT_TOP_LEVEL_KEYS_V12 = frozenset(
    {
        "access",
        "action_prior_mean_progress_m",
        "authority",
        "caps",
        "content_sha256",
        "determinism",
        "full_arm_gate",
        "gate",
        "hardware",
        "label_manifest",
        "masks",
        "n320",
        "physical_evidence_calibration",
        "preregistration_commit",
        "roles",
        "schedule_prefix_sha256",
        "schema",
        "scientific_change_from_v11",
        "seeds",
        "selection_control_comparisons",
        "selection_semantic",
        "status",
        "training",
        "wrong_rgb_mapping_sha256",
    }
)
ATTENTION_PARAMETER_SUFFIXES_V12 = frozenset(
    {
        f"{role}_{projection}{suffix}"
        for role in ("floor", "elevated")
        for projection, suffixes in (
            ("query_projection", (".weight", ".bias")),
            ("key_projection", (".weight",)),
            ("value_projection", (".weight", ".bias")),
            ("output_projection", (".weight", ".bias")),
        )
        for suffix in suffixes
    }
)
SEMANTIC_PARAMETER_SUFFIXES_V12 = frozenset(
    {
        f"{axis}.{module}.{suffix}"
        for axis in ("free_axis", "occupied_axis")
        for module in ("base", "local", "residual_output")
        for suffix in ("weight", "bias")
    }
)
ATTENTION_PARAMETER_INVENTORY_SHA256_V12 = (
    "c05baa32dc25a0b2ae62b77fbae8187c5aeaa1d87b4395dd7e84cfd952a1bc0f"
)
SEMANTIC_PARAMETER_INVENTORY_SHA256_V12 = (
    "3c67ea306d03e2ac68ba6d2cbffc709b445e6d918cb911b9ac62659c20aaff99"
)
EXPECTED_SEEDS_V12 = {
    "inherited_fresh_component_constructor": 20_260_712,
    "height_role_private_cpu_generators": 20_260_730,
    "experiment_and_stochastic_execution": 20_260_728,
    "bootstrap": 20_260_728,
}
EXPECTED_AUXILIARY_OBJECTIVE_V12 = {
    "name": "occupied_vs_rest_safety",
    "coefficient": 0.5,
    "logit_definition": (
        "occupied_semantic_logit_minus_logsumexp_free_and_unknown_semantic_logits"
    ),
    "row_balancing": (
        "per_raster_row_equal_average_of_present_occupied_and_rest_target_classes"
    ),
    "current_next_aggregation": "equal_average",
    "normalization": "binary_cross_entropy_with_logits_divided_by_log_2",
    "new_trainable_parameters": False,
}
EXPECTED_ARCHITECTURE_V12 = {
    "schema": "lewm_v12_neutral_disjoint_ternary_competition_architecture_v1",
    "predecessor": "fresh_v11_source_architecture_with_no_v11_runtime_reuse",
    "sole_change": "neutral_unknown_free_occupied_evidence_competition",
    "v11_parameter_or_buffer_change": False,
    "added_parameter_count": 0,
    "axis_inputs": {
        "free": {"latent_channels": [0, 32], "invalid_evidence": -20.0},
        "occupied": {"latent_channels": [32, 64], "invalid_evidence": -20.0},
    },
    "supported_cell_logits": {
        "unknown": "0",
        "free": "f",
        "occupied": "o",
        "normalization": "log_softmax",
    },
    "all_invalid_logits": [0.0, -20.0, -20.0],
    "objective": "S+P+U+R+O",
    "occupied_auxiliary_coefficient": 0.5,
    "new_loss_or_loss_weight": False,
    "predictor_consumes_shared_role_ordered_64_channel_state": True,
}


def _fresh_output_v12(repository_root: Path) -> Path:
    output = repository_root / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh V12 physical-evidence attempt_v1 exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _expected_sources_v12() -> dict[str, str]:
    if (
        type(ADAPTER_SOURCE_SHA256) is not str
        or len(ADAPTER_SOURCE_SHA256) != 64
        or any(character not in "0123456789abcdef" for character in ADAPTER_SOURCE_SHA256)
    ):
        raise PermissionError("V12 physical-evidence adapter source is not frozen")
    return {**SOURCE_SHA256, ADAPTER_SOURCE_RELATIVE_PATH: ADAPTER_SOURCE_SHA256}


def _validate_sources_v12(repository_root: Path) -> Mapping[str, str]:
    expected = _expected_sources_v12()
    observed = {
        relative: hashlib.sha256(_read_regular(repository_root / relative)).hexdigest()
        for relative in expected
    }
    if observed != expected:
        raise PermissionError("frozen V12 calibration dependency source changed")
    return observed


def _validate_activity_v12(
    activity: Any,
    *,
    schema: str,
    suffixes: frozenset[str],
    inventory_sha256: str,
    parameter_count: int,
    target_tensor_count: int,
    minimum_active: int,
) -> None:
    first_active = activity.get("first_active_update") if type(activity) is dict else None
    gradient_minimum = activity.get("minimum_gradient_l2") if type(activity) is dict else None
    gradient_maximum = activity.get("maximum_gradient_l2") if type(activity) is dict else None
    if (
        type(activity) is not dict
        or activity.get("schema") != schema
        or activity.get("update_count") != 1_000
        or activity.get("online_parameter_count") != parameter_count
        or activity.get("online_parameter_tensor_count") != len(suffixes)
        or activity.get("parameter_suffix_inventory_sha256") != inventory_sha256
        or activity.get("all_online_parameter_tensors_active_by_update_2") is not True
        or type(first_active) is not dict
        or set(first_active) != suffixes
        or any(type(update) is not int or update not in (1, 2) for update in first_active.values())
        or activity.get("latest_first_active_update") != max(first_active.values())
        or activity.get("active_update_count") != 1_000
        or activity.get("minimum_active_parameter_tensor_count") != minimum_active
        or activity.get("maximum_active_parameter_tensor_count") != len(suffixes)
        or type(gradient_minimum) not in (int, float)
        or type(gradient_maximum) not in (int, float)
        or not math.isfinite(float(gradient_minimum))
        or not math.isfinite(float(gradient_maximum))
        or not 0.0 < float(gradient_minimum) <= float(gradient_maximum)
        or activity.get("target_parameter_tensor_count") != target_tensor_count
        or activity.get("target_gradient_tensor_count") != 0
    ):
        raise PermissionError("V12 training activity receipt changed")


def _validate_initial_model_v12(initial: Any) -> None:
    if type(initial) is not dict:
        raise PermissionError("V12 initial model receipt changed")
    identity = initial.get("fresh_v11_state_identity")
    migration = identity.get("v11_source_migration_witness") if type(identity) is dict else None
    if (
        initial.get("schema")
        != "lewm_v12_neutral_disjoint_ternary_initial_model_v1"
        or initial.get("architecture") != EXPECTED_ARCHITECTURE_V12
        or initial.get("online_branch_attention_parameter_count") != 14_528
        or initial.get("online_branch_attention_parameter_tensor_count") != 14
        or initial.get("target_branch_attention_parameter_count") != 14_528
        or initial.get("target_branch_attention_parameter_tensor_count") != 14
        or initial.get("factorized_semantic_parameter_count") != 18_628
        or initial.get("factorized_semantic_parameter_tensor_count") != 12
        or initial.get("all_v11_parameters_partitioned_exactly_once") is not True
        or initial.get("optimizer_parameter_membership_changed_from_v11") is not False
        or initial.get("target_initial_gradient_tensor_count") != 0
        or initial.get("initial_hard_sync_count") != 1
        or initial.get("initial_ema_update_count") != 0
        or type(identity) is not dict
        or identity.get("schema")
        != "lewm_v12_fresh_v11_zero_parameter_state_identity_v1"
        or identity.get("predecessor_experiment_checkpoint_read") is not False
        or identity.get("v12_parameter_tensor_count") != 233
        or identity.get("v11_parameter_tensor_count") != 233
        or identity.get("v12_parameter_count") != 6_122_053
        or identity.get("v11_parameter_count") != 6_122_053
        or identity.get("added_parameter_tensor_count") != 0
        or identity.get("added_parameter_count") != 0
        or any(
            identity.get(name) is not True
            for name in (
                "all_parameter_values_bit_exact",
                "all_buffer_values_bit_exact",
                "semantic_axis_modules_reused_without_aliasing",
                "neutral_algebra_exact",
                "supported_probabilities_finite_and_normalized",
                "branch_invalid_evidence_fixed_to_minus_20",
                "all_invalid_logits_exact",
                "shared_predictor_state_unchanged",
                "ema_target_state_unchanged_and_frozen",
            )
        )
        or type(migration) is not dict
        or migration.get("predecessor_experiment_checkpoint_read") is not False
        or migration.get("all_common_v10_parameter_values_bit_exact") is not True
        or migration.get("all_common_v10_buffer_values_bit_exact") is not True
        or migration.get("online_branch_attention_parameter_count") != 14_528
        or migration.get("target_branch_attention_parameter_count") != 14_528
        or migration.get("factorized_semantic_parameter_count") != 18_628
        or migration.get("online_target_branch_attention_initial_copy_exact") is not True
        or migration.get("target_branch_attention_initial_gradient_tensor_count") != 0
    ):
        raise PermissionError("V12 fresh-state identity receipt changed")


def _validate_candidate_result_v12(receipt: Mapping[str, Any]) -> None:
    if type(receipt) is not dict or set(receipt) != EXPECTED_RESULT_TOP_LEVEL_KEYS_V12:
        raise PermissionError("V12 result top-level inventory changed")
    gate = receipt.get("gate")
    training = receipt.get("training")
    authority = receipt.get("authority")
    physical = receipt.get("physical_evidence_calibration")
    access = receipt.get("access")
    science = receipt.get("scientific_change_from_v11")
    if (
        receipt.get("schema") != CANDIDATE_RESULT_SCHEMA
        or receipt.get("status") != "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION"
        or receipt.get("content_sha256") != CANDIDATE_RESULT_CONTENT_SHA256
        or receipt.get("preregistration_commit") != CANDIDATE_PREREGISTRATION_COMMIT
        or receipt.get("seeds") != EXPECTED_SEEDS_V12
        or receipt.get("schedule_prefix_sha256")
        != "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528"
        or type(gate) is not dict
        or receipt.get("full_arm_gate") != gate
        or gate.get("status") != "PASS_FULL_ARM"
        or gate.get("passed") is not True
        or gate.get("failed_checks") != []
        or gate.get("thresholds") != EXPECTED_GATE_THRESHOLDS_V12
        or type(gate.get("checks")) is not dict
        or set(gate["checks"]) != EXPECTED_GATE_CHECKS_V12
        or len(gate["checks"]) != 24
        or any(value is not True for value in gate["checks"].values())
        or receipt.get("caps")
        != {"updates": 1_000, "microbatch_graphs": 4_000, "presentations": 16_000}
        or type(training) is not dict
        or set(training)
        != {
            "accounting",
            "checkpoint",
            "checkpoint_access_status",
            "core",
            "diagnostics",
            "factorized_semantic_axes_activity",
            "height_role_branch_attention_activity",
            "joint_from_update_one",
            "separate_head_or_predictor_training",
            "trace",
        }
        or training.get("accounting") != EXPECTED_ACCOUNTING_V12
        or training.get("checkpoint")
        != {
            "path": "checkpoint_update_1000.pt",
            "byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
            "file_sha256": CANDIDATE_CHECKPOINT_SHA256,
        }
        or training.get("checkpoint_access_status")
        != "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION"
        or training.get("core")
        != "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
        "height_role_factorized_evidence_lift"
        or training.get("joint_from_update_one") is not True
        or training.get("separate_head_or_predictor_training") is not False
        or type(training.get("diagnostics")) is not dict
    ):
        raise PermissionError("V12 terminal result contract changed")

    branch = training["height_role_branch_attention_activity"]
    semantic = training["factorized_semantic_axes_activity"]
    _validate_activity_v12(
        branch,
        schema="lewm_v11_height_role_branch_attention_training_activity_v1",
        suffixes=ATTENTION_PARAMETER_SUFFIXES_V12,
        inventory_sha256=ATTENTION_PARAMETER_INVENTORY_SHA256_V12,
        parameter_count=14_528,
        target_tensor_count=14,
        minimum_active=14,
    )
    _validate_activity_v12(
        semantic,
        schema="lewm_v11_factorized_semantic_axes_training_activity_v1",
        suffixes=SEMANTIC_PARAMETER_SUFFIXES_V12,
        inventory_sha256=SEMANTIC_PARAMETER_INVENTORY_SHA256_V12,
        parameter_count=18_628,
        target_tensor_count=0,
        minimum_active=8,
    )
    diagnostics = training["diagnostics"]
    contract = diagnostics.get("v12_contract")
    if (
        diagnostics.get("height_role_branch_attention") != branch
        or diagnostics.get("factorized_semantic_axes") != semantic
        or contract
        != {
            "schema": "lewm_v12_unchanged_joint_training_contract_v1",
            "training_helper": (
                "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
                "height_role_factorized_evidence_lift"
            ),
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_weight": False,
            "height_role_branch_attention": branch,
            "factorized_semantic_axes": semantic,
        }
    ):
        raise PermissionError("V12 joint-training diagnostics changed")
    if authority != {
        "development_only": True,
        "g2_navigation_final_evaluation_opened": False,
        "heldout_or_sealed_opened": False,
        "physical_calibration_run": False,
        "physical_evidence_gate_passed": False,
        "checkpoint_qualified": False,
        "promotion_performed": False,
        "retry_or_resume_authorized": False,
        "checkpoint_access_authorized_for_physical_calibration": False,
        "separate_physical_preregistration_required": True,
    }:
        raise PermissionError("V12 terminal authority changed")
    if physical != {
        "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT",
        "physical_calibration_run_in_this_attempt": False,
        "requires_full_arm_pass": True,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "threshold_tuple_count": 2_016,
        "physical_gate_passed": False,
        "schema": "lewm_v12_unchanged_physical_calibration_stage_v1",
        "source": "numerically_unchanged_v10_v4_2016_tuple_protocol",
        "v10_directional_baselines_are_interpretation_only": True,
        "physical_calibration_authorized_in_this_attempt": False,
    }:
        raise PermissionError("V12 physical-calibration staging changed")

    initial = science.get("initial_v12_model") if type(science) is dict else None
    _validate_initial_model_v12(initial)
    if (
        type(science) is not dict
        or science.get("only_change") != "neutral_disjoint_ternary_semantic_algebra"
        or science.get("architecture") != EXPECTED_ARCHITECTURE_V12
        or science.get("objective") != "S+P+U+R+O"
        or science.get("inherited_occupied_auxiliary")
        != EXPECTED_AUXILIARY_OBJECTIVE_V12
        or science.get("model_code_changed") is not True
        or science.get("parameter_or_buffer_state_changed") is not False
        or science.get("added_parameter_count") != 0
        or science.get("loss_gradient_surface_changed_by_registered_semantic_algebra")
        is not True
        or any(
            science.get(name) is not False
            for name in (
                "data_changed",
                "dataset_identity_changed",
                "input_tensorization_changed",
                "optimizer_rules_changed",
                "optimizer_parameter_tensor_membership_changed",
                "loss_source_or_coefficient_changed",
                "new_loss_or_loss_weight",
                "schedule_changed",
                "evaluation_changed",
            )
        )
    ):
        raise PermissionError("V12 science receipt changed")
    narrow = access.get("narrow_loader") if type(access) is dict else None
    if (
        type(access) is not dict
        or access.get("forbidden_input_count") != 0
        or access.get("g2_navigation_final_evaluation_open_count") != 0
        or type(narrow) is not dict
        or int(narrow.get("rgb_request_count", {}).get("fixed_negative", -1)) != 0
        or any(
            int(value) != 0
            for value in narrow.get("forbidden_semantic_counters", {}).values()
        )
        or receipt.get("n320", {}).get("encoder_only_initialization") is not True
        or receipt.get("n320", {}).get("predecessor_experiment_checkpoint_read")
        is not False
    ):
        raise PermissionError("V12 result custody receipt changed")


def _new_access_v12() -> dict[str, int]:
    access = dict(_v4._new_access())
    for name in (
        "candidate_receipt_reads",
        "candidate_checkpoint_reads",
        "candidate_checkpoint_loads",
    ):
        access.pop(name)
    access.update(
        {
            "candidate_result_read_attempts": 0,
            "candidate_result_read_successes": 0,
            "candidate_result_validations": 0,
            "candidate_checkpoint_read_attempts": 0,
            "candidate_checkpoint_read_successes": 0,
            "candidate_checkpoint_load_attempts": 0,
            "candidate_checkpoint_load_successes": 0,
        }
    )
    return access


def _load_candidate_v12(repository_root: Path, access: dict[str, int]) -> Any:
    root = repository_root / CANDIDATE_ROOT_RELATIVE_PATH
    access["candidate_result_read_attempts"] += 1
    result_raw = _read_regular(root / "result.json")
    access["candidate_result_read_successes"] += 1
    if (
        len(result_raw) != CANDIDATE_RESULT_BYTE_COUNT
        or hashlib.sha256(result_raw).hexdigest() != CANDIDATE_RESULT_FILE_SHA256
    ):
        raise PermissionError("V12 result file identity changed")
    receipt = _parse_canonical(result_raw, name="V12 terminal result")
    _validate_candidate_result_v12(receipt)
    access["candidate_result_validations"] += 1

    access["candidate_checkpoint_read_attempts"] += 1
    checkpoint_raw = _read_regular(root / "checkpoint_update_1000.pt")
    access["candidate_checkpoint_read_successes"] += 1
    if (
        len(checkpoint_raw) != CANDIDATE_CHECKPOINT_BYTE_COUNT
        or hashlib.sha256(checkpoint_raw).hexdigest() != CANDIDATE_CHECKPOINT_SHA256
    ):
        raise PermissionError("V12 checkpoint identity changed")
    adapter = importlib.import_module(ADAPTER_MODULE)
    if (
        getattr(adapter, "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT", None)
        != PREREGISTRATION_COMMIT
    ):
        raise PermissionError("V12 physical-evidence adapter authority changed")
    load_checkpoint = getattr(adapter, "load_checkpoint", None)
    if not callable(load_checkpoint):
        raise PermissionError("V12 physical-evidence adapter API changed")
    access["candidate_checkpoint_load_attempts"] += 1
    model = load_checkpoint(checkpoint_raw)
    access["candidate_checkpoint_load_successes"] += 1
    return model


def _validate_development_access_v12(
    inputs: Any, loader: Any, access: dict[str, int]
) -> None:
    loader_counts = loader.model_facing_access_counts()
    loader_receipt = loader.receipt()
    if (
        loader_counts["endpoint_rgb_row_request_count"] != sum(ROLE_COUNTS.values())
        or loader_counts["raster_label_row_request_count"]
        != sum(ROLE_COUNTS.values())
        or any(
            loader_counts[name] != 0
            for name in (
                "current_rgb_row_request_count",
                "next_rgb_row_request_count",
                "fixed_negative_rgb_row_request_count",
            )
        )
    ):
        raise PermissionError("model-facing development access changed")
    if (
        loader_receipt.get("raw_inputs_frame_attribute_invocation_count") != 0
        or any(
            int(value) != 0
            for value in loader_receipt.get("forbidden_semantic_counters", {}).values()
        )
    ):
        raise PermissionError("forbidden development access was recorded")
    payload_records = [
        record
        for record in inputs.consumed.values()
        if record.get("kind") in {"development_rgb", "raw_supervision"}
    ]
    if any("train" in record.get("roles", []) for record in payload_records):
        access["train_role_payload_requests"] += 1
        raise PermissionError("train-role payload entered calibration")


def _authority_v12(*, physical_passed: bool) -> Mapping[str, Any]:
    return {
        "development_only": True,
        "development_physical_evidence_passed": physical_passed,
        "g2_binding_preparation_authorized": physical_passed,
        "g2_opened": False,
        "g2_qualified": False,
        "navigation_qualified": False,
        "promotion_performed": False,
        "deployment_authorized": False,
        "training_or_resume_authorized": False,
        "scientific_retry_authorized": False,
        "heldout_or_sealed_opened": False,
    }


def _candidate_receipt_v12(*, validated: bool) -> Mapping[str, Any]:
    return {
        "terminal_result_commit": TERMINAL_RESULT_COMMIT,
        "result": {
            "path": "result.json",
            "byte_count": CANDIDATE_RESULT_BYTE_COUNT,
            "file_sha256": CANDIDATE_RESULT_FILE_SHA256,
            "content_sha256": CANDIDATE_RESULT_CONTENT_SHA256,
        },
        "checkpoint": {
            "path": "checkpoint_update_1000.pt",
            "schema": CANDIDATE_CHECKPOINT_SCHEMA,
            "byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
            "file_sha256": CANDIDATE_CHECKPOINT_SHA256,
        },
        "result_validated_before_checkpoint_read": validated,
        "adapter_module": ADAPTER_MODULE,
    }


def execute_v12(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    output = _fresh_output_v12(repository_root)
    access = _new_access_v12()
    stage = "reserved_output"
    source_hashes: Mapping[str, str] | None = None
    runtime = None
    inputs = None
    loader = None
    progress: Mapping[str, Any] | None = None
    try:
        stage = "validated_sources"
        source_hashes = _validate_sources_v12(repository_root)
        stage = "loaded_candidate"
        model = _load_candidate_v12(repository_root, access)
        stage = "constructed_development_boundary"
        runtime, inputs, loader, progress = _build_data_boundary(repository_root)
        role_arrays: dict[str, tuple[Any, Any]] = {}
        role_receipts: dict[str, Mapping[str, Any]] = {}
        for role in ("probability_calibration", "checkpoint_selection"):
            stage = f"collected_{role}"
            pairs = inputs.role_pairs(role)
            logits, labels, receipt = _collect_role(
                model,
                loader,
                pairs,
                role=role,
                torch=runtime.torch,
            )
            role_arrays[role] = (logits, labels)
            role_receipts[role] = receipt
        stage = "fit_select_score"
        science = _fit_select_score(
            *role_arrays["probability_calibration"],
            *role_arrays["checkpoint_selection"],
            provenance={
                "role": "probability_calibration",
                "candidate_result_content_sha256": CANDIDATE_RESULT_CONTENT_SHA256,
                "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
                "pair_count": ROLE_COUNTS["probability_calibration"],
                "cell_count": ROLE_CELL_COUNTS["probability_calibration"],
                "next_endpoint_order_sha256": role_receipts[
                    "probability_calibration"
                ]["next_endpoint_order_sha256"],
                "all_cells_used": True,
            },
            operation_counts=access,
        )
        stage = "validated_access"
        _validate_development_access_v12(inputs, loader, access)
        if access["calibration_fit_calls"] != 1 or access["threshold_selection_calls"] != 1:
            raise RuntimeError("V12 calibration or threshold selection count changed")
        stage = "write_outputs"
        calibration_raw = _canonical_bytes(science["calibration"]) + b"\n"
        calibration_binding = _atomic_write(output / "calibration.json", calibration_raw)
        physical_passed = bool(science["gate"]["passed"])
        result, _ = _write_json(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": science["gate"]["status"],
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "reference_calibration_preregistration_commit": (
                    REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT
                ),
                "candidate": _candidate_receipt_v12(validated=True),
                "source_sha256": source_hashes,
                "protocol": {
                    "scientific_change_from_v4": False,
                    "fit_select_score_implementation": (
                        "scripts.calibrate_go2_rgb_swept_progress_survival_joint_"
                        "jepa_v4_physical_evidence._fit_select_score"
                    ),
                    "routing": "NOT_APPLICABLE",
                },
                "roles": role_receipts,
                "calibration_artifact": {
                    **calibration_binding,
                    "content_sha256": science["calibration"]["content_sha256"],
                    "id": science["calibration"]["id"],
                },
                "threshold_selection": science["threshold_selection"],
                "selection": science["selection"],
                "gate": science["gate"],
                "routing": {
                    "status": "NOT_APPLICABLE",
                    "included_in_gate": False,
                    "reason": "physical_evidence_is_not_configuration_space",
                    "deferred_to": "G3_post_memory_multi_view_fusion",
                },
                "raw_access": _raw_access_snapshot(inputs, loader, progress),
                "access": access,
                "authority": _authority_v12(physical_passed=physical_passed),
            },
        )
        return result
    except Exception as error:
        failure, _ = _write_json(
            output / "failure.json",
            {
                "schema": FAILURE_SCHEMA,
                "status": "FAILED_OPERATIONALLY",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "reference_calibration_preregistration_commit": (
                    REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT
                ),
                "stage": stage,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "candidate": _candidate_receipt_v12(
                    validated=access["candidate_result_validations"] == 1
                ),
                "source_sha256": source_hashes,
                "raw_access": _raw_access_snapshot(inputs, loader, progress),
                "access": access,
                "authority": _authority_v12(physical_passed=False),
            },
        )
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = execute_v12()
    print(
        json.dumps(
            {"status": result["status"], "output": OUTPUT_RELATIVE_PATH},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    if result["status"] == "FAILED_OPERATIONALLY":
        return 1
    return 0 if result.get("gate", {}).get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
