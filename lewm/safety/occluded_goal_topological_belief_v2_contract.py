"""Pure contract for OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2.

V2 is the canonical-cache replacement for the outcome-observed, technically
invalid V1 execution.  It changes encoding custody only.  Every scientific
condition, formula, threshold, gate, classification, and next-decision rule is
inherited byte-for-value from V1.

Importing this module performs no file I/O, rendering, encoding, inference,
filtering, simulation, training, or execution.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lewm.safety import occluded_goal_topological_belief_v1_contract as V1


class OccludedGoalV2ContractError(ValueError):
    """Raised when V2 contract or custody evidence drifts."""


EXPERIMENT_ID = "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
SOURCE_PARENT_COMMIT = "dafff4b5cfdb7a18009f8719506bcdf337c962b6"
SCIENTIFIC_BASELINE_RESULT_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
CONTRACT_FREEZE_COMMIT_SUBJECT = (
    "Freeze canonical occluded-goal topological belief replacement"
)
RESULT_COMMIT_SUBJECT = (
    "Evaluate canonical occluded-goal topological belief replacement"
)

TECHNICAL_FAILURE_DISPOSITION = "BATCH_SLOT_DEPENDENT_DUPLICATE_IMAGE_ENCODING"
CANONICAL_SINGLETON_ENCODER_NONDETERMINISM = (
    "CANONICAL_SINGLETON_ENCODER_NONDETERMINISM"
)
V1_EXPERIMENT_ID = V1.EXPERIMENT_ID
V1_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v1"
)
OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v2"
)

# Frozen scientific authority: aliases deliberately preserve V1 values.  The
# tests assert exact equality so an encoding correction cannot drift science.
EPISODE_COUNT = V1.EPISODE_COUNT
QUERIES_PER_EPISODE = V1.QUERIES_PER_EPISODE
FAMILY_IDS = tuple(V1.FAMILY_IDS)
EPISODES_PER_FAMILY = V1.EPISODES_PER_FAMILY
SPLIT_ROLE_IDS = tuple(V1.SPLIT_ROLE_IDS)
SPLIT_EPISODE_COUNTS = copy.deepcopy(V1.SPLIT_EPISODE_COUNTS)
SPLIT_FAMILY_EPISODE_COUNTS = copy.deepcopy(V1.SPLIT_FAMILY_EPISODE_COUNTS)
HELDOUT_QUERY_COUNT = V1.HELDOUT_QUERY_COUNT
CALIBRATION_QUERY_COUNT = V1.CALIBRATION_QUERY_COUNT
TOTAL_QUERY_COUNT = V1.TOTAL_QUERY_COUNT
QUERY_INDEX_VALUES = tuple(V1.QUERY_INDEX_VALUES)
DEPTH_SINCE_LAST_UNAMBIGUOUS_VALUES = tuple(
    V1.DEPTH_SINCE_LAST_UNAMBIGUOUS_VALUES
)
CONSTRUCTED_SET_CAVEAT = V1.CONSTRUCTED_SET_CAVEAT
IDENTITY_DOMAIN = V1.IDENTITY_DOMAIN
SCENE_ID_PREFIX = V1.SCENE_ID_PREFIX
EPISODE_ID_PREFIX = V1.EPISODE_ID_PREFIX
EPISODE_PATH_ID_PREFIX = V1.EPISODE_PATH_ID_PREFIX
PROCEDURAL_SEED_BASE = V1.PROCEDURAL_SEED_BASE
PROCEDURAL_SEED_VALUES = tuple(V1.PROCEDURAL_SEED_VALUES)
PRIOR_PANEL_EXCLUSION_AUTHORITY = copy.deepcopy(V1.PRIOR_PANEL_EXCLUSION_AUTHORITY)
PORT_LABEL_ORDER = tuple(V1.PORT_LABEL_ORDER)
STAGE_B_LOCAL_CANDIDATE_IDS = tuple(V1.STAGE_B_LOCAL_CANDIDATE_IDS)
CONDITION_IDS = tuple(V1.CONDITION_IDS)
PRIMARY_CONDITION_IDS = tuple(V1.PRIMARY_CONDITION_IDS)
ABLATION_CONDITION_IDS = tuple(V1.ABLATION_CONDITION_IDS)
TOP_K = V1.TOP_K
FIXED_WINDOW_OBSERVATIONS = V1.FIXED_WINDOW_OBSERVATIONS
FIXED_WINDOW_TRANSITIONS = V1.FIXED_WINDOW_TRANSITIONS
PROBABILITY_FLOOR = V1.PROBABILITY_FLOOR
ECE_BIN_COUNT = V1.ECE_BIN_COUNT
CALIBRATION_GRID = copy.deepcopy(V1.CALIBRATION_GRID)
CALIBRATION_GRID_ORDER = tuple(V1.CALIBRATION_GRID_ORDER)
CALIBRATION_SELECTION = copy.deepcopy(V1.CALIBRATION_SELECTION)
SCIENTIFIC_OBSERVATION_LIKELIHOOD = copy.deepcopy(V1.OBSERVATION_LIKELIHOOD)
OBSERVATION_LIKELIHOOD = copy.deepcopy(V1.OBSERVATION_LIKELIHOOD)
OBSERVATION_LIKELIHOOD["persisted_arrays"] = {
    "observations.npz": {"dtype": "uint8", "layout": "N,H,W,RGB"},
    "canonical_tokens.npz/raw_tokens": {
        "dtype": "float16", "shape": [157, 768, 1024],
        "spatial_grid": [24, 32],
    },
    "canonical_descriptors.npz/spatial_descriptors": {
        "dtype": "float32", "shape": [157, 768, 1024],
        "spatial_grid": [24, 32],
    },
    "template_resolution": (
        "template_to_pixel_index.json -> pixel_index.json -> one canonical row"
    ),
    "occurrence_resolution": (
        "occurrence_index.json -> template_to_pixel_index.json -> one canonical row"
    ),
    "stage_a_beliefs.jsonl/observation_similarities": {
        "dtype": "JSON finite number", "shape": "one value per full graph node",
    },
}
FILTER_AUTHORITY = copy.deepcopy(V1.FILTER_AUTHORITY)
ALIASING_AUTHORITY = copy.deepcopy(V1.ALIASING_AUTHORITY)
METRIC_IDS = tuple(V1.METRIC_IDS)
METRIC_FORMULAS = copy.deepcopy(V1.METRIC_FORMULAS)
ABSOLUTE_GATE = copy.deepcopy(V1.ABSOLUTE_GATE)
INCREMENTAL_OVER_CURRENT_GATE = copy.deepcopy(V1.INCREMENTAL_OVER_CURRENT_GATE)
INCREMENTAL_OVER_MAP_GATE = copy.deepcopy(V1.INCREMENTAL_OVER_MAP_GATE)
SHORT_HISTORY_MATCH_GATE = copy.deepcopy(V1.SHORT_HISTORY_MATCH_GATE)
STAGE_A_CLASSIFICATIONS = tuple(V1.STAGE_A_CLASSIFICATIONS)
STAGE_A_PRECEDENCE = tuple(V1.STAGE_A_PRECEDENCE)
STAGE_B_CONDITION_IDS = tuple(V1.STAGE_B_CONDITION_IDS)
STAGE_B_EXECUTION_POLICY = copy.deepcopy(V1.STAGE_B_EXECUTION_POLICY)
STAGE_B_GATE = copy.deepcopy(V1.STAGE_B_GATE)
STAGE_B_METRIC_FORMULAS = copy.deepcopy(V1.STAGE_B_METRIC_FORMULAS)
STAGE_B_CLASSIFICATIONS = tuple(V1.STAGE_B_CLASSIFICATIONS)
NEXT_DECISION_BY_CLASSIFICATION = copy.deepcopy(V1.NEXT_DECISION_BY_CLASSIFICATION)
STAGE_B_AUTHORITY = copy.deepcopy(V1.STAGE_B_AUTHORITY)
SCIENTIFIC_PROHIBITIONS = copy.deepcopy(V1.PROHIBITIONS)
PROHIBITIONS = {
    **copy.deepcopy(V1.PROHIBITIONS),
    "action_conditioned_predictor_open": 0,
    "action_conditioned_predictor_training": 0,
    "new_local_ranker_training": 0,
    "new_panel_collection": 0,
    "invalid_v1_latent_reuse": 0,
    "invalid_v1_calibration_reuse": 0,
    "invalid_v1_belief_reuse": 0,
    "invalid_v1_metric_reuse": 0,
    "online_graph_construction": 0,
    "novelty_execution": 0,
    "beacon_discovery": 0,
}
SAFETY_WORKSTREAM_STATUS = "REQUIREMENTS_ACQUISITION_REQUIRED"
VJEPA_ENCODER_BINDING = copy.deepcopy(V1.VJEPA_ENCODER_BINDING)
STAGE_B_CURRENT_VISUAL_BINDING = copy.deepcopy(V1.STAGE_B_CURRENT_VISUAL_BINDING)
FROZEN_SOURCE_BINDINGS = copy.deepcopy(V1.FROZEN_SOURCE_BINDINGS)

V1_CONTRACT_SOURCE_BINDING = {
    "path": "lewm/safety/occluded_goal_topological_belief_v1_contract.py",
    "sha256": "55e95c74ff1793ff0c51d7cad99b619a0acca23f9e65ecb1188b64f35f23ffe3",
}
V1_METRICS_SOURCE_BINDING = {
    "path": "lewm/safety/occluded_goal_topological_belief_metrics_v1.py",
    "sha256": "fe17ee749859fdf7ec2b31843735f0664694e6bad31e627560b019ddf8bce533",
}

V1_RETAINED_LEAVES = (
    {"path": "calibration.json", "bytes": 3227066, "sha256": "4659e8d5d01df22a18875cb535eb73d42f0ad5414ccf8c93cc2884c886386f9d"},
    {"path": "contract.json", "bytes": 30192, "sha256": "061a1cfa6b239193f0f0ed095d8dafd4d5d8082d5ab087bf8651d07d07e6370e"},
    {"path": "graph_manifest.json", "bytes": 31169157, "sha256": "34902c7b25cf1a68076931da85811b9f2a89ff89c3622c06a28f8bd9b13aee09"},
    {"path": "keyframe_index.json", "bytes": 9974080, "sha256": "7aabae32f2cff756ffb90fe133ff53bd3cc265eb8854cbd481d7be433dbe3fe0"},
    {"path": "latent_index.json", "bytes": 10406766, "sha256": "4cd4414bec8d8801b5d0816090b91d94b587f7f2cd3bed29d3d8d40cd1a33579"},
    {"path": "latents.npz", "bytes": 1643337122, "sha256": "546b6c46f56a53d9c032dd63220c65091f6cec3be7cfbb8ebe0aff7ac5fd27d4"},
    {"path": "observations.npz", "bytes": 368315, "sha256": "040d995918bd4dbbd60a74fce33e8547abf27930b0a75b44975c6cfe44f88fa4"},
    {"path": "panel_manifest.json", "bytes": 38219, "sha256": "e348a83ec605256d93f6b6aeb09837919e927b61f95c46a60edf0bb958afe844"},
    {"path": "query_ledger.jsonl", "bytes": 15307920, "sha256": "b96576986416edcbadb053aedbf0bbd07c7e16226a1ce1c09cebd5ffe9feb3df"},
    {"path": "split_manifest.json", "bytes": 13162, "sha256": "c6029ec7743fcd2a971e8194752070a51fdacf18003a3e9308a450f15abd6904"},
    {"path": "stage_a_beliefs.jsonl", "bytes": 30043346, "sha256": "403db7b6e5d37cd0f720071ab6a456cd928141627a7b2deb2cee970bf4899dc4"},
    {"path": "stage_a_metrics.json", "bytes": 149707, "sha256": "ac5cf6a404739436be222d7cee4469fe5a147fbe54c746cb3962016acb85b49e"},
)
V1_REUSABLE_LEAVES = (
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "query_ledger.jsonl",
    "keyframe_index.json",
    "observations.npz",
)
V1_INVALID_NONREUSABLE_LEAVES = (
    "latent_index.json",
    "latents.npz",
    "calibration.json",
    "stage_a_beliefs.jsonl",
    "stage_a_metrics.json",
)
V1_CUSTODY_ONLY_LEAVES = ("contract.json",)
V1_RETAINED_ROOT_BINDING = {
    "root": str(V1_OUTPUT_ROOT),
    "source_freeze_commit": SOURCE_PARENT_COMMIT,
    "disposition": TECHNICAL_FAILURE_DISPOSITION,
    "leaves": [copy.deepcopy(value) for value in V1_RETAINED_LEAVES],
    "inventory": [copy.deepcopy(value) for value in V1_RETAINED_LEAVES],
    "leaf_count": 12,
    "total_bytes": 1744065052,
    "reusable_leaves": list(V1_REUSABLE_LEAVES),
    "invalid_nonreusable_leaves": list(V1_INVALID_NONREUSABLE_LEAVES),
    "custody_only_leaves": list(V1_CUSTODY_ONLY_LEAVES),
    "nonreusable_leaves": [
        *V1_INVALID_NONREUSABLE_LEAVES,
        *V1_CUSTODY_ONLY_LEAVES,
    ],
    "technical_failure": (
        "the batch-size-eight encoder evaluated exact-byte duplicate images in "
        "different batch slots and emitted slot-dependent token rows"
    ),
    "scientific_disposition": {
        "valid_scientific_classification": None,
        "observed_invalid_stage_a_classification": "TOPOLOGICAL_MAP_SUFFICIENT",
        "observed_invalid_strongest_condition": "MAP_FILTER",
        "stage_b_ranker_initialized": True,
        "stage_b_ranker_score_evaluated": False,
        "stage_b_ranker_call_count": 0,
        "stage_b_rollouts_executed": False,
        "stage_b_trace_persisted": False,
        "stage_b_result": None,
        "v1_result_commit": None,
        "v1_result_json": None,
        "v1_report_markdown": None,
        "v1_file_hashes": None,
    },
    "defect_evidence": {
        "template_rows": 375,
        "unique_exact_pixel_groups": 157,
        "duplicate_exact_pixel_groups": 76,
        "conflicting_duplicate_exact_pixel_groups": 75,
        "affected_template_rows": 292,
        "batch_size": 8,
        "slot_identity": "row_index % 8",
        "same_pixel_same_slot_digest_drift_count": 0,
    },
}

OUTCOME_OBSERVED_CLAIMS_BOUNDARY = {
    "v1_stage_a_outcome_observed": True,
    "v1_stage_a_outcome_scientifically_usable": False,
    "v1_stage_a_outcome_may_select_v2_parameters_or_gates": False,
    "v1_invalid_latents_calibration_beliefs_metrics_reusable": False,
    "v2_status": "outcome-observed corrected development replacement",
    "v2_is_fresh_untouched_or_confirmatory": False,
    "permitted_claim": "JEPA place belief under oracle topology on a corrected development cache",
    "must_disclose": (
        "V1 Stage A was observed before the batch-slot-dependent duplicate-image "
        "encoding defect was detected; V1 latent-derived evidence is invalid and unused."
    ),
    "positive_v2_confirmation_requirement": (
        "a fresh scene-disjoint panel, one exploratory memory configuration, and "
        "the unchanged descriptor and belief method"
    ),
    "does_not_establish": list(V1.CLAIMS["does_not_establish"]),
}
CLAIMS = {
    "development_only": True,
    "outcome_observed": True,
    "positive_wording": OUTCOME_OBSERVED_CLAIMS_BOUNDARY["permitted_claim"],
    "does_not_establish": tuple(V1.CLAIMS["does_not_establish"]),
}

TEMPLATE_ROW_COUNT = 375
UNIQUE_PIXEL_COUNT = 157
REUSED_TEMPLATE_ROW_COUNT = TEMPLATE_ROW_COUNT - UNIQUE_PIXEL_COUNT
MULTI_TEMPLATE_PIXEL_GROUP_COUNT = 76
OCCURRENCE_COUNT = 33384
ENCODING_PASS_COUNT = 2
SINGLETON_ENCODER_BATCH_SIZE = 1
PREPROCESSING_DIGEST = "a1e27a421d635deb7ac12d05f0099623492772ea35ce493b8997bfd6e6226732"

CANONICAL_HASH_DOMAINS = {
    "rgb_pixel_sha256": (
        "SHA-256(canonical compact JSON {shape:[168,224,3],dtype:'|u1',"
        "layout:'C'} || NUL || C-contiguous uint8 bytes)"
    ),
    "preprocessed_tensor_sha256": (
        "SHA-256(canonical compact JSON {shape:[3,384,512],dtype:'<f4',"
        "layout:'C'} || NUL || C-contiguous float32 bytes)"
    ),
    "raw_token_sha256": (
        "SHA-256(canonical compact JSON {shape:[768,1024],dtype:'<f2',"
        "layout:'C'} || NUL || C-contiguous float16 bytes)"
    ),
    "spatial_descriptor_sha256": (
        "SHA-256(canonical compact JSON {shape:[768,1024],dtype:'<f4',"
        "layout:'C'} || NUL || C-contiguous float32 bytes)"
    ),
    "canonical_json": "sort keys, compact separators, UTF-8, allow_nan false, no trailing LF",
}
PRE_OUTCOME_FORBIDDEN_LEAVES = (
    "calibration.json",
    "stage_a_beliefs.jsonl",
    "stage_a_metrics.json",
    "stage_b_trace.jsonl",
    "stage_b_metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
)
PRE_OUTCOME_BOUNDARY_AUTHORITY = {
    "boundary": "IMMEDIATELY_BEFORE_FIRST_CANONICAL_ENCODER_INITIALIZATION",
    "present_v2_leaf_names": sorted(("contract.json", *V1_REUSABLE_LEAVES)),
    "forbidden_outcome_leaf_names": list(PRE_OUTCOME_FORBIDDEN_LEAVES),
    "observed_forbidden_outcome_leaf_names": [],
    "calibration_outcome_documents_opened": 0,
    "heldout_outcome_documents_opened": 0,
    "external_regeneration_receipt_present": False,
}
CANONICAL_CACHE_GATE_IDS = (
    "v1_root_binding_exact",
    "reusable_inputs_byte_identical",
    "invalid_v1_science_not_reused",
    "no_calibration_or_heldout_outcome_opened",
    "canonical_pixel_hashes_valid",
    "canonical_template_identity_valid",
    "preprocessing_hashes_exact_between_passes",
    "singleton_encoder_batch_size",
    "fresh_encoder_instances",
    "two_pass_pixel_order_exact",
    "two_pass_raw_tokens_exact",
    "two_pass_descriptors_exact",
    "template_mapping_complete",
    "occurrence_mapping_complete",
    "no_occurrence_mapped_by_gpu_slot",
    "token_digest_singleton_per_pixel",
    "descriptor_digest_singleton_per_pixel",
    "no_template_specific_token_copy_differs",
    "canonical_regeneration_byte_identical",
    "cache_file_bindings_valid",
    "pass",
)
CANONICAL_ENCODING_AUTHORITY = {
    "pixel_identity": "verified rgb_pixel_sha256; never recipe or template identity",
    "pixel_order": "157 unique rgb_pixel_sha256 values in lexicographic order",
    "canonical_template": "lexicographically smallest member template ID for each pixel SHA",
    "preprocessing": {
        **copy.deepcopy(VJEPA_ENCODER_BINDING),
        "preprocessing_digest": PREPROCESSING_DIGEST,
        "prepared_tensor_shape": [3, 384, 512],
        "prepared_tensor_dtype": "float32",
    },
    "hash_domains": copy.deepcopy(CANONICAL_HASH_DOMAINS),
    "pre_outcome_boundary": copy.deepcopy(PRE_OUTCOME_BOUNDARY_AUTHORITY),
    "required_cache_gates": list(CANONICAL_CACHE_GATE_IDS),
    "passes": 2,
    "fresh_encoder_load_per_pass": True,
    "batch_size": 1,
    "encoder_invocations_per_pass": UNIQUE_PIXEL_COUNT,
    "exact_pass_comparison": (
        "pixel order, preprocessed tensor digest, raw-token digest, spatial-descriptor "
        "digest, and canonical cache-content digest are bitwise identical"
    ),
    "cache_layout": (
        "157 canonical rows only; resolve 375 templates and 33384 occurrences through "
        "their explicit indices; no duplicate per-template token or descriptor rows"
    ),
    "gate_times": ["before calibration", "immediately before conditional Stage B"],
    "tolerance": 0,
    "terminal_failure": {
        "classification": CANONICAL_SINGLETON_ENCODER_NONDETERMINISM,
        "trigger": "any pass-1/pass-2 preprocessed, raw-token, descriptor, order, or canonical-cache digest mismatch",
        "boundary": "stop before calibration",
        "scientific_result": None,
        "tolerance": 0,
    },
}

RUNTIME_OUTPUT_PATHS = {
    "contract": "contract.json",
    "panel_manifest": "panel_manifest.json",
    "split_manifest": "split_manifest.json",
    "graph_manifest": "graph_manifest.json",
    "query_ledger": "query_ledger.jsonl",
    "keyframe_index": "keyframe_index.json",
    "observations": "observations.npz",
    "pixel_index": "pixel_index.json",
    "template_to_pixel_index": "template_to_pixel_index.json",
    "occurrence_index": "occurrence_index.json",
    "canonical_tokens": "canonical_tokens.npz",
    "canonical_descriptors": "canonical_descriptors.npz",
    "canonical_latent_index": "canonical_latent_index.json",
    "canonical_descriptor_index": "canonical_descriptor_index.json",
    "encoding_determinism_receipt": "encoding_determinism_receipt.json",
    "cache_integrity_receipt": "cache_integrity_receipt.json",
    "calibration": "calibration.json",
    "stage_a_beliefs": "stage_a_beliefs.jsonl",
    "stage_a_metrics": "stage_a_metrics.json",
    "conditional_stage_b_trace": "stage_b_trace.jsonl",
    "conditional_stage_b_metrics": "stage_b_metrics.json",
    "result": "result.json",
    "report": "result.md",
    "file_hashes": "file_hashes.json",
}
UNCONDITIONAL_OUTPUT_LEAVES = tuple(
    value
    for key, value in RUNTIME_OUTPUT_PATHS.items()
    if key not in {"conditional_stage_b_trace", "conditional_stage_b_metrics"}
)

TRACKED_SOURCE_PATHS = (
    "docs/lewm_go2_occluded_goal_topological_belief_v2_contract_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v2_fixture_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v2_output_schema_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v2_preregistration_2026-09-01.md",
    "docs/lewm_go2_occluded_goal_topological_belief_v2_source_closure_2026-09-01.json",
    "lewm/safety/occluded_goal_topological_belief_metrics_v2.py",
    "lewm/safety/occluded_goal_topological_belief_v2_contract.py",
    "lewm/tests/test_evaluate_occluded_goal_topological_belief_v2.py",
    "lewm/tests/test_occluded_goal_topological_belief_metrics_v2.py",
    "lewm/tests/test_occluded_goal_topological_belief_v2_contract.py",
    "lewm/tests/test_run_occluded_goal_topological_belief_v2.py",
    "scripts/evaluate_occluded_goal_topological_belief_v2.py",
    "scripts/run_occluded_goal_topological_belief_v2.py",
)
SOURCE_DEPENDENCY_PATHS = tuple(V1.SOURCE_DEPENDENCY_PATHS) + (
    V1_CONTRACT_SOURCE_BINDING["path"],
    V1_METRICS_SOURCE_BINDING["path"],
    "scripts/run_occluded_goal_topological_belief_v1.py",
    "scripts/evaluate_occluded_goal_topological_belief_v1.py",
)

DIRECT_RUNTIME_POLICY = {
    "direct_foreground_processes_only": True,
    "custom_audit_hooks": False,
    "custom_startup_or_forensic_framework": False,
    "v1_root_immutable": True,
    "v2_root_absent_before_start": True,
    "reusable_v1_leaves_copied_byte_identically": list(V1_REUSABLE_LEAVES),
    "invalid_v1_science_opened_or_reused": False,
    "cache_gate_before_calibration": True,
    "cache_gate_before_stage_b": True,
}


def canonical_json_bytes(value: Any) -> bytes:
    return V1.canonical_json_bytes(value)


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return V1.attach_content_digest(value)


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return V1.validate_content_digest(value)
    except V1.OccludedGoalContractError as exc:
        raise OccludedGoalV2ContractError(str(exc)) from exc


def v1_retained_root_authority() -> dict[str, Any]:
    return copy.deepcopy(V1_RETAINED_ROOT_BINDING)


def validate_v1_retained_root_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != V1_RETAINED_ROOT_BINDING:
        raise OccludedGoalV2ContractError("V1 retained-root binding drift")
    return v1_retained_root_authority()


def build_contract() -> dict[str, Any]:
    return attach_content_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "scientific_baseline_result_commit": SCIENTIFIC_BASELINE_RESULT_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "v1_retained_root_binding": v1_retained_root_authority(),
            "outcome_observed_claims_boundary": copy.deepcopy(
                OUTCOME_OBSERVED_CLAIMS_BOUNDARY
            ),
            "scientific_design_source": {
                "experiment_id": V1_EXPERIMENT_ID,
                "contract_source": copy.deepcopy(V1_CONTRACT_SOURCE_BINDING),
                "metrics_source": copy.deepcopy(V1_METRICS_SOURCE_BINDING),
                "design_unchanged": True,
            },
            "prior_panel_exclusion_authority": copy.deepcopy(
                PRIOR_PANEL_EXCLUSION_AUTHORITY
            ),
            "frozen_source_bindings": copy.deepcopy(FROZEN_SOURCE_BINDINGS),
            "panel": {
                "episode_count": EPISODE_COUNT,
                "queries_per_episode": QUERIES_PER_EPISODE,
                "families": list(FAMILY_IDS),
                "split_episode_counts": copy.deepcopy(SPLIT_EPISODE_COUNTS),
                "aliasing": copy.deepcopy(ALIASING_AUTHORITY),
            },
            "conditions": list(CONDITION_IDS),
            "observation_likelihood": copy.deepcopy(OBSERVATION_LIKELIHOOD),
            "filter": copy.deepcopy(FILTER_AUTHORITY),
            "calibration": {
                "grid": {key: list(value) for key, value in CALIBRATION_GRID.items()},
                "grid_order": list(CALIBRATION_GRID_ORDER),
                "selection": copy.deepcopy(CALIBRATION_SELECTION),
            },
            "metrics": {
                "ids": list(METRIC_IDS),
                "formulas": copy.deepcopy(METRIC_FORMULAS),
            },
            "stage_a": {
                "absolute_gate": copy.deepcopy(ABSOLUTE_GATE),
                "incremental_over_current_gate": copy.deepcopy(
                    INCREMENTAL_OVER_CURRENT_GATE
                ),
                "incremental_over_map_gate": copy.deepcopy(INCREMENTAL_OVER_MAP_GATE),
                "short_history_match_gate": copy.deepcopy(SHORT_HISTORY_MATCH_GATE),
                "classifications": list(STAGE_A_CLASSIFICATIONS),
                "precedence": list(STAGE_A_PRECEDENCE),
            },
            "stage_b": copy.deepcopy(STAGE_B_AUTHORITY),
            "stage_b_current_visual_binding": copy.deepcopy(
                STAGE_B_CURRENT_VISUAL_BINDING
            ),
            "stage_b_gate": copy.deepcopy(STAGE_B_GATE),
            "stage_b_metric_formulas": copy.deepcopy(STAGE_B_METRIC_FORMULAS),
            "stage_b_classifications": list(STAGE_B_CLASSIFICATIONS),
            "next_decisions": copy.deepcopy(NEXT_DECISION_BY_CLASSIFICATION),
            "canonical_encoding": copy.deepcopy(CANONICAL_ENCODING_AUTHORITY),
            "claims": copy.deepcopy(CLAIMS),
            "safety_workstream": SAFETY_WORKSTREAM_STATUS,
            "prohibitions": copy.deepcopy(PROHIBITIONS),
            "runtime_policy": copy.deepcopy(DIRECT_RUNTIME_POLICY),
            "output": {
                "root": str(OUTPUT_ROOT),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "unconditional_leaf_count": len(UNCONDITIONAL_OUTPUT_LEAVES),
                "conditional_leaf_count": 2,
                "receipt_self_digests": False,
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "source_dependency_paths": list(SOURCE_DEPENDENCY_PATHS),
        }
    )


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    if dict(value) != expected:
        raise OccludedGoalV2ContractError("V2 contract value drift")
    return copy.deepcopy(expected)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "OccludedGoalV2ContractError",
    "attach_content_digest",
    "build_contract",
    "canonical_json_bytes",
    "v1_retained_root_authority",
    "validate_content_digest",
    "validate_contract",
    "validate_v1_retained_root_binding",
]
