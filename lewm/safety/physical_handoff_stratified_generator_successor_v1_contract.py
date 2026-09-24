"""Pure prospective authority for the stratified handoff generator successor.

The successor is a development-only continuation of the completed physical
graph-edge handoff V4 qualification.  It changes only candidate allocation:
each of the frozen four-family by sixteen-stratum streams receives as many as
64 fresh deterministic attempts and stops only after its fourth qualified
state or attempt 63.  Physical construction, state disposition, snapshot and
probe evidence, teacher qualification, panel selection, role assignment, and
all downstream scientific authorities remain those frozen by V4.

This module is pure.  It imports no simulator, model, encoder, or ranker and
does not read any runtime output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
from pathlib import Path
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as _V4


class PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(ValueError):
    """Raised when the prospective successor authority drifts."""


V4 = _V4


def _frozen_v4(name: str) -> Any:
    """Copy one explicitly named, unchanged V4 authority value."""

    return copy.deepcopy(getattr(V4, name))


# Explicit frozen physical authorities used by the native successor schema.
# This list is intentionally finite: there is no bulk inherited namespace.
BOUNDARY_EVIDENCE_FIELDS = _frozen_v4("BOUNDARY_EVIDENCE_FIELDS")
CONTACT_AUTHORITY = _frozen_v4("CONTACT_AUTHORITY")
HARD_STOP_DISPOSITIONS = _frozen_v4("HARD_STOP_DISPOSITIONS")
INITIAL_BOUNDARY_PAYLOAD_AUTHORITY = _frozen_v4(
    "INITIAL_BOUNDARY_PAYLOAD_AUTHORITY"
)
INITIAL_TIPPED_METADATA_FIELDS = _frozen_v4("INITIAL_TIPPED_METADATA_FIELDS")
MATERIAL_FILE_BINDING_FIELDS = _frozen_v4("MATERIAL_FILE_BINDING_FIELDS")
NUMERICAL_TOLERANCES = _frozen_v4("NUMERICAL_TOLERANCES")
PHYSICAL_OUTCOME_AUTHORITY = _frozen_v4("PHYSICAL_OUTCOME_AUTHORITY")
PHYSICAL_RUNTIME_CORE_FIELDS = _frozen_v4("PHYSICAL_RUNTIME_CORE_FIELDS")
PORT_DWELL_PHYSICS_SAMPLES = _frozen_v4("PORT_DWELL_PHYSICS_SAMPLES")
PROBE_TIPPED_METADATA_FIELDS = _frozen_v4("PROBE_TIPPED_METADATA_FIELDS")
QUALIFICATION_BACKEND_RUNTIME_AUTHORITY = _frozen_v4(
    "QUALIFICATION_BACKEND_RUNTIME_AUTHORITY"
)
QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS = _frozen_v4(
    "QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS"
)
QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS = _frozen_v4(
    "QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS"
)
RUNTIME_ENVIRONMENT_AUTHORITY = _frozen_v4("RUNTIME_ENVIRONMENT_AUTHORITY")
SNAPSHOT_IDENTITY_FIELDS = _frozen_v4("SNAPSHOT_IDENTITY_FIELDS")
STATE_DISPOSITION_FIELDS = _frozen_v4("STATE_DISPOSITION_FIELDS")
STATE_DISPOSITION_STAGES = _frozen_v4("STATE_DISPOSITION_STAGES")
STATE_DISPOSITIONS = _frozen_v4("STATE_DISPOSITIONS")
TEACHER_CONTROLLER_AUTHORITY = _frozen_v4("TEACHER_CONTROLLER_AUTHORITY")
TEACHER_CRITERIA_FIELDS = _frozen_v4("TEACHER_CRITERIA_FIELDS")
TEACHER_QUALIFICATION_COMPONENT_IDS = _frozen_v4(
    "TEACHER_QUALIFICATION_COMPONENT_IDS"
)
TEACHER_TERMINAL_METADATA_FIELDS = _frozen_v4(
    "TEACHER_TERMINAL_METADATA_FIELDS"
)
TEACHER_TRACE_DT_S = _frozen_v4("TEACHER_TRACE_DT_S")
TEACHER_TRACE_MEMBER_ORDER = _frozen_v4("TEACHER_TRACE_MEMBER_ORDER")
TERMINAL_NONFINITE_INITIAL_MEMBER_IDS = _frozen_v4(
    "TERMINAL_NONFINITE_INITIAL_MEMBER_IDS"
)
TERMINAL_NONFINITE_TRACE_MEMBER_IDS = _frozen_v4(
    "TERMINAL_NONFINITE_TRACE_MEMBER_IDS"
)
TERMINATION_FLAGS_FIELDS = _frozen_v4("TERMINATION_FLAGS_FIELDS")
TERMINATION_FLAG_ORDER = _frozen_v4("TERMINATION_FLAG_ORDER")
V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS = _frozen_v4(
    "V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS"
)
V4_RAW_TEACHER_REDUCTION_FIELDS = _frozen_v4(
    "V4_RAW_TEACHER_REDUCTION_FIELDS"
)
V4_TEACHER_MATERIAL_SUMMARY_FIELDS = _frozen_v4(
    "V4_TEACHER_MATERIAL_SUMMARY_FIELDS"
)
V4_TEACHER_RECORD_FIELDS = _frozen_v4("V4_TEACHER_RECORD_FIELDS")
PERSISTED_ARRAY_EVIDENCE_FIELDS = _frozen_v4(
    "PERSISTED_ARRAY_EVIDENCE_FIELDS"
)
PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS = _frozen_v4(
    "PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS"
)
PERSISTED_ARRAY_ROW_FIELDS = _frozen_v4("PERSISTED_ARRAY_ROW_FIELDS")
ROLE_IDS = _frozen_v4("ROLE_IDS")
ROLE_COUNTS = _frozen_v4("ROLE_COUNTS")
FAMILY_ROLE_COUNTS = _frozen_v4("FAMILY_ROLE_COUNTS")
TARGET_IDS = _frozen_v4("TARGET_IDS")
CANDIDATE_IDS = _frozen_v4("CANDIDATE_IDS")
HORIZON_TICKS = _frozen_v4("HORIZON_TICKS")
RESET_TRACE_PAIR_COMPARISON_AUTHORITY = _frozen_v4(
    "RESET_TRACE_PAIR_COMPARISON_AUTHORITY"
)
PRIMARY_HORIZON = _frozen_v4("PRIMARY_HORIZON")
ROUTE_LOOKAHEAD_DISTANCE_M = _frozen_v4("ROUTE_LOOKAHEAD_DISTANCE_M")
TARGET_FEATURE_ORDER = _frozen_v4("TARGET_FEATURE_ORDER")
TARGET_DEFINITIONS = _frozen_v4("TARGET_DEFINITIONS")
DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC = _frozen_v4(
    "DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC"
)
HELDOUT_CONDITION_IDS = _frozen_v4("HELDOUT_CONDITION_IDS")
REPEAT_BRANCH_IDS = _frozen_v4("REPEAT_BRANCH_IDS")
REPEATS_PER_BRANCH = _frozen_v4("REPEATS_PER_BRANCH")
HANDOFF_GATE = _frozen_v4("HANDOFF_GATE")
TARGET_MATERIALITY = _frozen_v4("TARGET_MATERIALITY")
PRIMARY_CLASSIFICATIONS = _frozen_v4("PRIMARY_CLASSIFICATIONS")
SECONDARY_CLASSIFICATIONS = _frozen_v4("SECONDARY_CLASSIFICATIONS")
NEXT_DECISION_BY_CLASSIFICATION = _frozen_v4(
    "NEXT_DECISION_BY_CLASSIFICATION"
)
VJEPA_ENCODER_BINDING = _frozen_v4("VJEPA_ENCODER_BINDING")
CURRENT_VISUAL_RANKER_BINDING = _frozen_v4(
    "CURRENT_VISUAL_RANKER_BINDING"
)
ORIGINAL_RANKER_TRAINING_SUPPORT = _frozen_v4(
    "ORIGINAL_RANKER_TRAINING_SUPPORT"
)
RANKER_CONTRACT_DIFFERENCE_INVENTORY = _frozen_v4(
    "RANKER_CONTRACT_DIFFERENCE_INVENTORY"
)
CANONICAL_ENCODING_AUTHORITY = _frozen_v4("CANONICAL_ENCODING_AUTHORITY")
CANDIDATE_PRIMITIVES = _frozen_v4("CANDIDATE_PRIMITIVES")
CANDIDATE_BANK = _frozen_v4("CANDIDATE_BANK")
COMMAND_TICKS_PER_BLOCK = _frozen_v4("COMMAND_TICKS_PER_BLOCK")
EXECUTED_BLOCK_COUNT = _frozen_v4("EXECUTED_BLOCK_COUNT")
PHYSICS_STEPS_PER_BRANCH = _frozen_v4("PHYSICS_STEPS_PER_BRANCH")
COMMAND_TRACKING_AUTHORITY = _frozen_v4("COMMAND_TRACKING_AUTHORITY")
RESET_FIXTURE_COMMAND = _frozen_v4("RESET_FIXTURE_COMMAND")
RESET_FIXTURE_COMMAND_TICKS = _frozen_v4("RESET_FIXTURE_COMMAND_TICKS")
RESET_FIXTURE_PHYSICS_SAMPLES = _frozen_v4("RESET_FIXTURE_PHYSICS_SAMPLES")
PHYSICS_STEPS_PER_COMMAND_TICK = _frozen_v4(
    "PHYSICS_STEPS_PER_COMMAND_TICK"
)
POLICY_STEPS_PER_COMMAND_TICK = _frozen_v4(
    "POLICY_STEPS_PER_COMMAND_TICK"
)
PHYSICS_STEPS_PER_POLICY_STEP = _frozen_v4(
    "PHYSICS_STEPS_PER_POLICY_STEP"
)


EXPERIMENT_ID = "PHYSICAL_HANDOFF_STRATIFIED_GENERATOR_SUCCESSOR_V1"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
FINAL_EVALUATION_ELIGIBLE = False
OUTPUT_BASENAME = "physical_handoff_stratified_generator_successor_v1"
IDENTITY_NAMESPACE = "phsgs-v1"
FORBIDDEN_PREDECESSOR_IDENTITY_PREFIXES = ("pgehq-v1", "ogtb-v1")
SOURCE_PARENT_COMMIT = "4e52b30bde4cd521b6bfc1d37d4d4744f7eede56"
SOURCE_BASELINE_COMMIT = "2c45f1835ef810041d356b60a16a67ab163597f3"
V4_RESULT_COMMIT = SOURCE_PARENT_COMMIT
V4_SOURCE_FREEZE_COMMIT = SOURCE_BASELINE_COMMIT
V4_SOURCE_PARENT_COMMIT = "5b08d433f2e69f6e4fe85c9b14696726e32d2ab1"
CONTRACT_FREEZE_COMMIT_SUBJECT = "Freeze stratified physical handoff generator successor"
RESULT_COMMIT_SUBJECT = "Evaluate stratified physical handoff generator successor"

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_handoff_stratified_generator_successor_v1"
)
MATERIAL_ROOT = OUTPUT_ROOT.parent / f"{OUTPUT_BASENAME}_material"
EXTERNAL_REGENERATION_RECEIPT = None
ENGINEERING_LOG_PATH = OUTPUT_ROOT.parent / f"{OUTPUT_BASENAME}_engineering_log.jsonl"
PROHIBITED_EXTERNAL_PUBLICATION_PATHS = (
    str(OUTPUT_ROOT.parent / f"{OUTPUT_BASENAME}_regeneration_receipt.json"),
    str(OUTPUT_ROOT.parent / f"{OUTPUT_BASENAME}_custody_receipt.json"),
    str(OUTPUT_ROOT.parent / f"{OUTPUT_BASENAME}_terminal_custody_bundle.json"),
)

FAMILY_IDS = tuple(V4.FAMILY_IDS)
STRATUM_FACTOR_IDS = (
    "family",
    "edge_direction",
    "opening_width_id",
    "opening_width_m",
    "port_offset_id",
    "port_offset_m",
    "starting_bearing_rad",
    "base_spawn_lateral_offset_m",
)
STRATUM_FACTOR_ALIAS_AUTHORITY = {
    "edge_direction": "candidate_spec.route_direction",
    "opening_width_id": "candidate_spec.passage_width_id",
    "opening_width_m": (
        "candidate_spec.passage_width_m and selected_directed_edge.opening_width_m"
    ),
    "port_offset_id": "candidate_spec.port_distance_id",
    "port_offset_m": "candidate_spec.port_distance_m",
    "starting_bearing_rad": "candidate_spec.spawn_yaw_offset_rad",
    "base_spawn_lateral_offset_m": "candidate_spec.spawn_lateral_offset_m",
}
HASH_PERTURBATION_PARAMETER_IDS = (
    "geometry_jitter_x_m",
    "geometry_jitter_y_m",
    "port_distance_delta_m",
    "spawn_lateral_delta_m",
    "spawn_yaw_delta_rad",
    "corridor_length_delta_m",
    "opening_lateral_delta_m",
)
COMMON_FAILURE_PARAMETER_IDS = (
    "family",
    "stratum_index",
    "edge_direction",
    "opening_width_id",
    "opening_width_m",
    "port_offset_id",
    "port_offset_m",
    "base_spawn_lateral_offset_m",
    "starting_bearing_rad",
    *HASH_PERTURBATION_PARAMETER_IDS,
)
COMMON_PHYSICAL_PARAMETER_FIELDS = frozenset(
    {
        "family",
        "stratum_index",
        "failed_attempt_count",
        "parameter_order",
        "canonical_value_counts",
        "constant_across_all_failed_attempts",
        "predeclared_parameters_only",
        "post_hoc_factor_added",
    }
)
STRATA_PER_FAMILY = 16
STREAM_COUNT = len(FAMILY_IDS) * STRATA_PER_FAMILY
MAX_ATTEMPTS_PER_STREAM = 64
TARGET_QUALIFIED_PER_STREAM = 4
MAX_CANDIDATE_COUNT = STREAM_COUNT * MAX_ATTEMPTS_PER_STREAM
PANEL_STATE_COUNT = V4.V3.STATE_COUNT

GENERATOR_PANEL_AVAILABLE = "STRATIFIED_PHYSICAL_HANDOFF_PANEL_AVAILABLE"
GENERATOR_LOW_YIELD = "STRATIFIED_GENERATOR_LOW_YIELD"
GENERATOR_FEASIBILITY_NO_GO = "PHYSICAL_HANDOFF_STRATUM_FEASIBILITY_NO_GO"
GENERATOR_TERMINAL_STATUSES = (
    GENERATOR_PANEL_AVAILABLE,
    GENERATOR_LOW_YIELD,
    GENERATOR_FEASIBILITY_NO_GO,
)
TURNING_JUNCTION_GENERATOR_NEXT_DECISION = (
    "TURNING_JUNCTION_GENERATOR_SUCCESSOR_V1"
)
OFFSET_OPENING_GENERATOR_NEXT_DECISION = (
    "OFFSET_OPENING_GENERATOR_SUCCESSOR_V1"
)
PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION = (
    "PHYSICAL_EDGE_STRATUM_CONTRACT_REVISION_V1"
)
GENERATOR_FAILURE_NEXT_DECISIONS = (
    TURNING_JUNCTION_GENERATOR_NEXT_DECISION,
    OFFSET_OPENING_GENERATOR_NEXT_DECISION,
    PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION,
)

OUTPUT_LEAVES = (
    "contract.json",
    "V4_context.json",
    "generator_stream_manifest.json",
    "generator_terminal_records.jsonl",
    "generator_metrics.json",
    "panel_manifest.json",
    "split_manifest.json",
    "state_snapshot_index.json",
    "teacher_trace_index.json",
    "edge_port_index.json",
    "target_contracts.json",
    "candidate_fanout.jsonl",
    "development_target_selection.json",
    "heldout_scores.jsonl",
    "repeatability.jsonl",
    "metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
)
SUCCESS_OUTPUT_LEAVES = OUTPUT_LEAVES
SUCCESS_OUTPUT_LEAF_COUNT = 19
GENERATOR_TERMINAL_OUTPUT_LEAVES = OUTPUT_LEAVES[:5] + OUTPUT_LEAVES[-4:]
GENERATOR_TERMINAL_OUTPUT_LEAF_COUNT = 9
DOWNSTREAM_OUTPUT_LEAVES = OUTPUT_LEAVES[5:15]

SCIENTIFIC_QUESTIONS = (
    "Can all 64 registered physical edge strata produce qualified states under the unchanged teacher and route semantics?",
    "Were the V4 missing strata caused primarily by shallow sampling rather than an intrinsically infeasible family or stratum contract?",
    "Once a balanced physical panel exists, can a directed graph edge be converted into an executable local target?",
    "Does the existing twelve-action bank contain actions that enter the requested physical edge?",
    "Can the frozen current-visual V-JEPA ranker select those actions?",
    "Does the selected command execute repeatably through the intended port?",
)
CLAIMS_BOUNDARY = {
    "concerns": [
        "simulated physical graph-edge handoff",
        "stratified benchmark construction",
        "directed local target representation",
        "candidate-bank coverage",
        "frozen ranker selection",
        "simulated low-level execution",
    ],
    "does_not_establish": [
        "deployment safety",
        "physical Go2 performance",
        "material-contact safety",
        "memory confirmation",
        "online graph construction",
        "novelty or beacon discovery",
        "complete maze navigation",
    ],
    "development_only": True,
    "final_evaluation_eligible": False,
}
PROHIBITED_INFRASTRUCTURE = (
    "custom Python audit hooks",
    "launcher or child role frameworks",
    "PREEXECUTION attempt accounting",
    "terminal custody bundles",
    "custom finalisers",
    "cross-version opaque serialization gates",
    "another forensic task",
)
UNCHANGED_V4_INVARIANT_IDS = (
    "four registered physical handoff families",
    "sixteen frozen stratum meanings per family",
    "teacher route construction",
    "teacher controller",
    "physical duration and cadence limits",
    "teacher qualification criteria and disposition precedence",
    "physics-contact instrumentation and ontology",
    "tipped-state evidence and continuation semantics",
    "source-region leave predicate",
    "selected-port crossing predicate and deterministic binary64 projection",
    "directed-port and competing-port semantics",
    "twelve-action candidate bank",
    "H1 through H3 horizons",
    "two-trial reset semantics and tolerances",
    "directed local target representations",
    "development and heldout role assignment",
    "frozen current-visual V-JEPA encoder",
    "frozen ranker",
    "downstream metrics, gates, and classifications",
)


def canonical_json_bytes(value: Any) -> bytes:
    return V4.canonical_json_bytes(value)


def _canonical_no_lf_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return V4.attach_content_digest(value)


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return V4.validate_content_digest(value)
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            str(exc)
        ) from exc


PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "prohibited_external_publication_authority.v1"
        ),
        "paths": list(PROHIBITED_EXTERNAL_PUBLICATION_PATHS),
        "required_absent_before_scientific_execution": True,
        "required_absent_before_publication": True,
        "required_absent_after_publication": True,
        "external_regeneration_receipt_authorized": False,
        "terminal_custody_bundle_authorized": False,
    }
)


V4_RESULT_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "result.json"),
    "bytes": 102122,
    "sha256": "b354b5a40d6124f475a65f867d8eb7ffe27f326d624a1e22e885d68bbd8706c8",
    "content_digest": "e7e555ba68e6193514ed6f28ab710511f1d684a63efa1c81d89fedf0f41ec138",
}
V4_PANEL_ADEQUACY_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "panel_adequacy.json"),
    "bytes": 25161,
    "sha256": "da3dc15f4256035a63b52a2975e7abc59ffba5bac68db66e1a685419818d2172",
    "content_digest": "45d1cd7d2c156365a4a599b72a70d34cf581c8f2286b37b8341703455b0224ab",
}
V4_METRICS_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "metrics.json"),
    "bytes": 100972,
    "sha256": "ead6cd3518d8048c5c5195a5c4a4d75526bc1df2e18df0fe355b3ca2e80a3089",
    "content_digest": "87dc07bf93c2aa43122b9332e54184d5a9e5545926371924dd905040d8c748fd",
}
V4_FILE_HASHES_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "file_hashes.json"),
    "bytes": 1359,
    "sha256": "c6ccde279be8f7eb371c751e1f0e79283874d0e08a3d8a6d6b2958b905e9e1b3",
    "content_digest": "825d8fce1f029e5726e89c1d23c12906fc582f64690872b77a98cfa07c0a8ee6",
}
V4_SCIENTIFIC_INVARIANCE_RECEIPT_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "scientific_invariance_receipt.json"),
    "bytes": 4434,
    "sha256": "85152872724416fa8029fb8968db16f5ba7a795e64f23dd9ba955e6e6eb4ef43",
}
V4_HISTORICAL_CUSTODY_RECEIPT_BINDING = {
    "path": str(V4.OUTPUT_ROOT / "v1_v2_v3_custody_and_nonreuse.json"),
    "bytes": 1468,
    "sha256": "164bb04af3ee7b4cff51141a7ce4cbb935eb9d3df351451895dd3bbd2ae75cb5",
}
V4_INDEPENDENT_REDUCER_RECEIPT_BINDING = {
    "path": str(V4.EXTERNAL_REGENERATION_RECEIPT),
    "bytes": 49170,
    "sha256": "b00f64831a0e2e347b02f3d0081d3cb278251deafa1d35ac07be1095d9eae1f4",
}

V4_CANONICAL_OFFSET_SHORTFALL_STRATA = (0, 1, 2, 4, 5, 6, 10)
V4_CANONICAL_TURNING_SHORTFALL_STRATA = (0, 1, 2, 4, 6)
V4_CANONICAL_SHORTFALL_STREAMS = tuple(
    [("TURNING_JUNCTION", index) for index in V4_CANONICAL_TURNING_SHORTFALL_STRATA]
    + [("OFFSET_OPENING", index) for index in V4_CANONICAL_OFFSET_SHORTFALL_STRATA]
)
V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING = (
    "SUPPORTS_SHALLOW_SAMPLING_AS_PRIMARY_V4_SHORTFALL_CAUSE"
)
V4_SHORTFALL_INSUFFICIENT = (
    "INSUFFICIENT_TO_ESTABLISH_SHALLOW_SAMPLING_AS_PRIMARY_V4_SHORTFALL_CAUSE"
)
V4_CONTEXT_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "v4_context_authority.v1"
        ),
        "v4_experiment_id": V4.EXPERIMENT_ID,
        "v4_source_parent_commit": V4_SOURCE_PARENT_COMMIT,
        "v4_source_freeze_commit": V4_SOURCE_FREEZE_COMMIT,
        "v4_result_commit": V4_RESULT_COMMIT,
        "v4_result_commit_parent": V4_SOURCE_FREEZE_COMMIT,
        "v4_result_and_freeze_tree_oid": (
            "e7114e1b1bae2b746c19f80c854b6f60a3b7da28"
        ),
        "bindings": {
            "result": copy.deepcopy(V4_RESULT_BINDING),
            "panel_adequacy": copy.deepcopy(V4_PANEL_ADEQUACY_BINDING),
            "metrics": copy.deepcopy(V4_METRICS_BINDING),
            "file_hashes": copy.deepcopy(V4_FILE_HASHES_BINDING),
            "scientific_invariance_receipt": copy.deepcopy(
                V4_SCIENTIFIC_INVARIANCE_RECEIPT_BINDING
            ),
            "v1_v2_v3_custody_and_nonreuse": copy.deepcopy(
                V4_HISTORICAL_CUSTODY_RECEIPT_BINDING
            ),
            "independent_reducer_receipt": copy.deepcopy(
                V4_INDEPENDENT_REDUCER_RECEIPT_BINDING
            ),
        },
        "result_primary_classification": "PHYSICAL_HANDOFF_PANEL_INADEQUATE",
        "result_content_digest": V4_RESULT_BINDING["content_digest"],
        "panel_adequacy_content_digest": V4_PANEL_ADEQUACY_BINDING[
            "content_digest"
        ],
        "v4_qualified_count": 180,
        "v4_nonqualified_count": 76,
        "v4_terminal_identity_count": 256,
        "v4_expected_terminal_identity_count": 256,
        "v4_teacher_execution_count": 248,
        "v4_hard_technical_stop_count": 0,
        "v4_missing_terminal_identity_count": 0,
        "v4_duplicated_terminal_identity_count": 0,
        "v4_tipped_and_teacher_ineligible_states_recorded_as_dispositions": True,
        "v4_collection_continued_after_defined_nonqualification": True,
        "v4_production_fake_runtime": False,
        "v4_adequate_stratum_count": 52,
        "v4_total_stratum_count": 64,
        "v4_shortfall_stratum_count": 12,
        "v4_all_filled_families": [
            "STRAIGHT_PASSAGE",
            "ROOM_OR_LOOP_EXIT",
        ],
        "turning_junction_shortfall_strata": list(
            V4_CANONICAL_TURNING_SHORTFALL_STRATA
        ),
        "offset_opening_shortfall_strata": list(
            V4_CANONICAL_OFFSET_SHORTFALL_STRATA
        ),
        "offset_opening_shortfall_stratum_count": 7,
        "requested_offset_shortfall_stratum_index": 8,
        "requested_offset_stratum_index_8_is_canonical_shortfall": False,
        "requested_offset_stratum_index_8_canonical_evidence": {
            "pool_indices": [160, 161, 162, 163],
            "qualified_count": 4,
            "selected_pool_index": 160,
            "adequate": True,
        },
        "corrected_offset_shortfall_stratum_index": 5,
        "corrected_offset_stratum_index_5_canonical_evidence": {
            "pool_indices": [148, 149, 150, 151],
            "qualified_count": 0,
            "primary_disposition_counts": {"TEACHER_PHYSICS_CONTACT": 4},
            "adequate": False,
        },
        "discrepancy_resolution": (
            "THE_PERSISTED_CANONICAL_V4_EVIDENCE_GOVERNS_OFFSET_HAS_SEVEN_"
            "SHORTFALL_STRATA"
        ),
        "offset_opening_qualified_count": 20,
        "offset_opening_selected_count": 9,
        "offset_opening_supplies_any_qualified_state": True,
        "three_identity_contracts_retained": {
            "artifact_file_sha256": "descriptive transport bytes",
            "snapshot_semantic_digest_v1": "canonical semantic snapshot identity",
            "snapshot_behavioural_digest_v1": (
                "designated trial-0 restoration/probe identity"
            ),
        },
        "v4_representation_authority_canonical_json_lf_sha256": {
            "targets": (
                "108d0372fce5e349d13befa2c8e65eb8e6d9f91444399f32a0192a570d0e52e7"
            ),
            "canonical_encoding": (
                "23c305ebe9f9b58e35033ce2e1caa2a1519d6b9c5df7f2ad1a035add4bd04455"
            ),
            "frozen_models": (
                "a180df4af253b342c8d7f92981d6d369be2faa53f20d636bdf2b1078b4c0e312"
            ),
            "ranker_input_authority": (
                "cdbc8b518b9d48de6934430e25c3d0cacdb541e9520eb1428c71bc7e854eadac"
            ),
        },
        "v4_runtime_representation_outputs": {
            "panel_manifest_present": False,
            "pixel_index_present": False,
            "latent_index_present": False,
            "encoding_receipt_present": False,
            "ranker_output_present": False,
            "candidate_fanout_present": False,
            "heldout_scores_present": False,
            "repeatability_present": False,
            "absence_reason": "V4 panel adequacy gate failed before downstream execution",
            "absence_is_not_a_representation_receipt": True,
        },
        "v4_interpretation": [
            (
                "V4 established that a fixed four-candidate allocation per "
                "stratum does not reliably produce a complete physical handoff "
                "panel."
            ),
            (
                "It did not test whether graph edges can be translated into "
                "physical action, because the panel gate blocked every downstream "
                "model and candidate evaluation."
            ),
        ],
        "v4_development_only": True,
        "v4_final_evaluation_eligible": False,
        "v4_models_trained": 0,
        "v4_downstream_outcomes_opened": 0,
        "v4_context_is_read_only": True,
        "v4_runtime_artifact_or_candidate_outcome_reuse_authorized": False,
    }
)
V4_INTERPRETATION_SENTENCES = tuple(V4_CONTEXT_AUTHORITY["v4_interpretation"])

V4_SHORTFALL_RESOLUTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "v4_shortfall_resolution_authority.v1"
        ),
        "v4_context_authority_content_digest": V4_CONTEXT_AUTHORITY[
            "content_digest"
        ],
        "canonical_stream_order": [
            {"family": family, "stratum_index": stratum}
            for family, stratum in V4_CANONICAL_SHORTFALL_STREAMS
        ],
        "canonical_stream_count": 12,
        "offset_shortfall_correction_preserved": {
            "requested_but_adequate_stratum_index": 8,
            "corrected_shortfall_stratum_index": 5,
            "canonical_offset_shortfall_strata": list(
                V4_CANONICAL_OFFSET_SHORTFALL_STRATA
            ),
        },
        "conclusion_rule": {
            "all_twelve_successor_streams_at_least_four_qualified": (
                V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
            ),
            "any_twelve_successor_stream_between_one_and_three_qualified": (
                V4_SHORTFALL_INSUFFICIENT
            ),
            "any_twelve_successor_stream_zero_qualified": (
                GENERATOR_FEASIBILITY_NO_GO
            ),
        },
        "stream_resolution_statuses": [
            "RESOLVED_AT_FOUR_QUALIFIED",
            "PARTIAL_YIELD_BELOW_FOUR",
            "ZERO_YIELD_AT_ATTEMPT_LIMIT",
        ],
        "supports_is_evidence_not_causal_proof": True,
        "intrinsic_infeasibility_is_not_established_by_this_projection": True,
    }
)


SEED_DERIVATION_DOMAIN = "SEED_STREAM_V1"
SEED_NAMESPACE_PREIMAGE = f"{EXPERIMENT_ID}\0{SEED_DERIVATION_DOMAIN}"
SEED_NAMESPACE_SHA256 = (
    "121983a0a24260cb0d55f12a02c3240dcfecfece0e3c00318d2083cc1c47cecb"
)
PROCEDURAL_SEED_BASE = 8221747320681816064
PROCEDURAL_SEED_TERMINAL = 8221747320681820159
PROCEDURAL_SEED_SORTED_PROJECTION_SHA256 = (
    "d6e9bfad1d89f2e4cc1baf91dc7a6636e818137751631c1ba20e770af304a9a1"
)
SEED_DOMAIN_MINIMUM = PROCEDURAL_SEED_BASE
SEED_DOMAIN_MAXIMUM = PROCEDURAL_SEED_TERMINAL
SEED_DERIVATION_INPUT_FIELDS = (
    "domain",
    "experiment_id",
    "family",
    "stratum_index",
    "attempt_index",
)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            f"{label} drift"
        )
    return value


def _family(value: Any) -> str:
    if value not in FAMILY_IDS:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "family drift"
        )
    return str(value)


def stream_index(family: str, stratum_index: int) -> int:
    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    return FAMILY_IDS.index(family) * STRATA_PER_FAMILY + stratum


def candidate_index(family: str, stratum_index: int, attempt_index: int) -> int:
    attempt = _integer(
        attempt_index, "attempt_index", 0, MAX_ATTEMPTS_PER_STREAM - 1
    )
    return attempt * STREAM_COUNT + stream_index(family, stratum_index)


def _seed_derivation_input(
    family: str, stratum_index: int, attempt_index: int
) -> dict[str, Any]:
    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    attempt = _integer(
        attempt_index, "attempt_index", 0, MAX_ATTEMPTS_PER_STREAM - 1
    )
    return {
        "domain": SEED_DERIVATION_DOMAIN,
        "experiment_id": EXPERIMENT_ID,
        "family": family,
        "stratum_index": stratum,
        "attempt_index": attempt,
    }


def seed_derivation_sha256(
    family: str, stratum_index: int, attempt_index: int
) -> str:
    return _canonical_no_lf_sha256(
        _seed_derivation_input(family, stratum_index, attempt_index)
    )


def derive_procedural_seed(
    family: str, stratum_index: int, attempt_index: int
) -> int:
    stream = stream_index(family, stratum_index)
    attempt = _integer(
        attempt_index, "attempt_index", 0, MAX_ATTEMPTS_PER_STREAM - 1
    )
    return PROCEDURAL_SEED_BASE + stream * MAX_ATTEMPTS_PER_STREAM + attempt


FAMILY_SLUGS = {
    "STRAIGHT_PASSAGE": "straight-passage",
    "TURNING_JUNCTION": "turning-junction",
    "OFFSET_OPENING": "offset-opening",
    "ROOM_OR_LOOP_EXIT": "room-or-loop-exit",
}


def _identity_stem(family: str, stratum_index: int, attempt_index: int) -> str:
    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    attempt = _integer(
        attempt_index, "attempt_index", 0, MAX_ATTEMPTS_PER_STREAM - 1
    )
    return (
        f"{IDENTITY_NAMESPACE}-{FAMILY_SLUGS[family]}-"
        f"stratum-{stratum:02d}-attempt-{attempt:02d}"
    )


def _stream_id(family: str, stratum_index: int) -> str:
    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    return (
        f"{IDENTITY_NAMESPACE}-{FAMILY_SLUGS[family]}-stratum-{stratum:02d}"
    )


def _stratum_axes(family: str, stratum_index: int) -> dict[str, Any]:
    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    width_id = ("NARROW", "WIDE")[(stratum // 8) % 2]
    distance_id = ("NEAR", "FAR")[(stratum // 4) % 2]
    lateral = tuple(V4.GEOMETRY_AUTHORITY["spawn_lateral_offset_m"])[
        (stratum // 2) % 2
    ]
    yaw = tuple(V4.GEOMETRY_AUTHORITY["spawn_yaw_offset_rad"])[stratum % 2]
    if family == "STRAIGHT_PASSAGE":
        direction = "STRAIGHT"
    elif family in {"TURNING_JUNCTION", "OFFSET_OPENING"}:
        direction = "LEFT" if stratum < 8 else "RIGHT"
    elif stratum < 8:
        direction = "STRAIGHT"
    elif stratum < 12:
        direction = "LEFT"
    else:
        direction = "RIGHT"
    return {
        "passage_width_id": width_id,
        "passage_width_m": V4.GEOMETRY_AUTHORITY["passage_width_m"][width_id],
        "port_distance_id": distance_id,
        "port_distance_m": V4.GEOMETRY_AUTHORITY["port_distance_m"][distance_id],
        "spawn_lateral_offset_m": lateral,
        "spawn_yaw_offset_rad": yaw,
        "route_direction": direction,
    }


def build_candidate_spec(
    family: str, stratum_index: int, attempt_index: int
) -> dict[str, Any]:
    """Build one fresh candidate with the frozen V4 geometry formulas."""

    family = _family(family)
    stratum = _integer(
        stratum_index, "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    attempt = _integer(
        attempt_index, "attempt_index", 0, MAX_ATTEMPTS_PER_STREAM - 1
    )
    seed = derive_procedural_seed(family, stratum, attempt)
    adjustment_identity = f"{family}:{stratum}:{attempt}:{seed}"
    digest = hashlib.sha256(adjustment_identity.encode("utf-8")).digest()
    unit_x = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    unit_y = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    unit_port = int.from_bytes(digest[16:20], "big") / float(2**32 - 1)
    unit_lateral = int.from_bytes(digest[20:24], "big") / float(2**32 - 1)
    unit_yaw = int.from_bytes(digest[24:28], "big") / float(2**32 - 1)
    unit_length = int.from_bytes(digest[28:30], "big") / float(2**16 - 1)
    unit_opening = int.from_bytes(digest[30:32], "big") / float(2**16 - 1)
    jitter_limit = V4.GEOMETRY_AUTHORITY["hash_jitter_limit_m"]
    axes = _stratum_axes(family, stratum)
    adjustments = {
        "port_distance_delta_m": (2.0 * unit_port - 1.0) * 0.01,
        "spawn_lateral_delta_m": (2.0 * unit_lateral - 1.0) * 0.01,
        "spawn_yaw_delta_rad": (2.0 * unit_yaw - 1.0) * 0.02,
        "corridor_length_delta_m": (2.0 * unit_length - 1.0) * 0.08,
        "opening_lateral_delta_m": (2.0 * unit_opening - 1.0) * 0.03,
    }
    stem = _identity_stem(family, stratum, attempt)
    spec = {
        "candidate_index": candidate_index(family, stratum, attempt),
        "stream_index": stream_index(family, stratum),
        "stream_id": _stream_id(family, stratum),
        "candidate_spec_id": f"{stem}-spec",
        "scene_id": f"{stem}-scene",
        "state_id": f"{stem}-state",
        "episode_id": f"{stem}-episode",
        "graph_id": f"{stem}-graph",
        "family": family,
        "stratum_index": stratum,
        "attempt_index": attempt,
        # V4's geometry builder called this within-stratum axis variant_index.
        # In the deeper successor stream the deterministic attempt is the
        # corresponding within-stratum variant.
        "variant_index": attempt,
        **axes,
        "geometry_jitter_xy_m": [
            (2.0 * unit_x - 1.0) * jitter_limit,
            (2.0 * unit_y - 1.0) * jitter_limit,
        ],
        "variant_adjustments": adjustments,
        "procedural_seed": seed,
        "procedural_seed_derivation_sha256": seed_derivation_sha256(
            family, stratum, attempt
        ),
        "role": None,
    }
    route_geometry = V4.V3.V2.V1._route_geometry
    spec["geometry"] = route_geometry(
        family=family,
        direction=axes["route_direction"],
        width_m=axes["passage_width_m"],
        port_distance_m=axes["port_distance_m"],
        translation_xy=tuple(spec["geometry_jitter_xy_m"]),
        spawn_lateral_m=axes["spawn_lateral_offset_m"],
        spawn_yaw_rad=axes["spawn_yaw_offset_rad"],
        variant_port_distance_delta_m=adjustments["port_distance_delta_m"],
        variant_spawn_lateral_delta_m=adjustments["spawn_lateral_delta_m"],
        variant_spawn_yaw_delta_rad=adjustments["spawn_yaw_delta_rad"],
        variant_corridor_length_delta_m=adjustments["corridor_length_delta_m"],
        variant_opening_lateral_delta_m=adjustments[
            "opening_lateral_delta_m"
        ],
    )
    spec["canonical_spec_sha256"] = _canonical_no_lf_sha256(spec)
    return spec


def validate_candidate_spec(
    value: Mapping[str, Any], *, expected_candidate_index: int | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "candidate spec is not a mapping"
        )
    row = copy.deepcopy(dict(value))
    family = _family(row.get("family"))
    stratum = _integer(
        row.get("stratum_index"), "stratum_index", 0, STRATA_PER_FAMILY - 1
    )
    attempt = _integer(
        row.get("attempt_index"),
        "attempt_index",
        0,
        MAX_ATTEMPTS_PER_STREAM - 1,
    )
    expected = build_candidate_spec(family, stratum, attempt)
    identity_values = [
        row.get(field)
        for field in (
            "stream_id",
            "candidate_spec_id",
            "scene_id",
            "state_id",
            "episode_id",
            "graph_id",
        )
    ]
    if any(
        not isinstance(item, str)
        or not item.startswith(f"{IDENTITY_NAMESPACE}-")
        or item.startswith(FORBIDDEN_PREDECESSOR_IDENTITY_PREFIXES)
        for item in identity_values
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "candidate identity namespace drift"
        )
    if expected_candidate_index is not None and expected["candidate_index"] != (
        expected_candidate_index
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "candidate index drift"
        )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "candidate spec drift"
        )
    return row


def build_candidate_specs() -> list[dict[str, Any]]:
    """Return all 4,096 specs in deterministic global registry order."""

    return [
        build_candidate_spec(family, stratum, attempt)
        for attempt in range(MAX_ATTEMPTS_PER_STREAM)
        for family in FAMILY_IDS
        for stratum in range(STRATA_PER_FAMILY)
    ]


def build_prospective_pool_specs() -> list[dict[str, Any]]:
    return build_candidate_specs()


def _candidate_identity_row(
    family: str, stratum_index: int, attempt_index: int
) -> dict[str, Any]:
    stem = _identity_stem(family, stratum_index, attempt_index)
    return {
        "candidate_index": candidate_index(family, stratum_index, attempt_index),
        "stream_index": stream_index(family, stratum_index),
        "stream_id": _stream_id(family, stratum_index),
        "family": family,
        "stratum_index": stratum_index,
        "attempt_index": attempt_index,
        "candidate_spec_id": f"{stem}-spec",
        "scene_id": f"{stem}-scene",
        "state_id": f"{stem}-state",
        "episode_id": f"{stem}-episode",
        "graph_id": f"{stem}-graph",
        "procedural_seed": derive_procedural_seed(
            family, stratum_index, attempt_index
        ),
        "procedural_seed_derivation_sha256": seed_derivation_sha256(
            family, stratum_index, attempt_index
        ),
    }


def build_candidate_identity_manifest() -> list[dict[str, Any]]:
    return [
        _candidate_identity_row(family, stratum, attempt)
        for attempt in range(MAX_ATTEMPTS_PER_STREAM)
        for family in FAMILY_IDS
        for stratum in range(STRATA_PER_FAMILY)
    ]


PROCEDURAL_SEED_VALUES = tuple(
    row["procedural_seed"] for row in build_candidate_identity_manifest()
)
if len(set(PROCEDURAL_SEED_VALUES)) != MAX_CANDIDATE_COUNT:
    raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
        "successor procedural seed collision"
    )
if set(PROCEDURAL_SEED_VALUES) & set(V4.PROCEDURAL_SEED_VALUES):
    raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
        "successor/V4 procedural seed overlap"
    )
if _canonical_no_lf_sha256(sorted(PROCEDURAL_SEED_VALUES)) != (
    PROCEDURAL_SEED_SORTED_PROJECTION_SHA256
):
    raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
        "successor procedural seed registry digest drift"
    )

V4_REGISTERED_SEED_REGISTRY_SHA256 = _canonical_no_lf_sha256(
    sorted(V4.PROCEDURAL_SEED_VALUES)
)
REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY = {
    "v4_inherited_broad_numeric_seed_registry": {
        "count": V4.PRIOR_SCENE_EXCLUSION_AUTHORITY["inherited_authority"][
            "prior_projection"
        ]["numeric_seed_count"],
        "canonical_sorted_unique_no_lf_sha256": V4.PRIOR_SCENE_EXCLUSION_AUTHORITY[
            "inherited_authority"
        ]["prior_projection"]["numeric_seed_canonical_json_sha256"],
    },
    "v4_registered_candidate_seeds": {
        "count": len(V4.PROCEDURAL_SEED_VALUES),
        "canonical_sorted_unique_no_lf_sha256": (
            V4_REGISTERED_SEED_REGISTRY_SHA256
        ),
    },
    "v4_broad_predecessor_numeric_seed_registry": {
        "count": V4.PRIOR_SCENE_EXCLUSION_AUTHORITY[
            "broad_prior_plus_v2_projection"
        ]["numeric_seed_count"],
        "canonical_sorted_unique_no_lf_sha256": V4.PRIOR_SCENE_EXCLUSION_AUTHORITY[
            "broad_prior_plus_v2_projection"
        ]["numeric_seed_canonical_json_sha256"],
    },
    "v4_nonregistered_family_fixture_seeds": {
        "count": 4,
        "canonical_sorted_unique_no_lf_sha256": (
            "fe1264bc943059ad84df5d53dab32171068c539f1cbb6086b35dceed2e77164f"
        ),
    },
}

SEED_DERIVATION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "seed_derivation_authority.v1"
        ),
        "algorithm": (
            "domain-separated SHA-256 namespace-block selection followed by "
            "exact linear family-stratum-attempt allocation"
        ),
        "input": (
            "canonical compact JSON without LF over exactly "
            "{domain,experiment_id,family,stratum_index,attempt_index}"
        ),
        "input_fields": list(SEED_DERIVATION_INPUT_FIELDS),
        "domain": SEED_DERIVATION_DOMAIN,
        "namespace_preimage": (
            "UTF-8(EXPERIMENT_ID) || NUL || UTF-8(SEED_STREAM_V1)"
        ),
        "namespace_sha256": SEED_NAMESPACE_SHA256,
        "namespace_block_base": PROCEDURAL_SEED_BASE,
        "namespace_block_terminal": PROCEDURAL_SEED_TERMINAL,
        "namespace_block_alignment": 4096,
        "namespace_block_derivation": (
            "take namespace SHA-256 first 64 bits big-endian, retain its low "
            "61 bits, set the two high signed-safe namespace bits to binary11, "
            "then clear the low 12 bits to obtain a 4096-aligned block"
        ),
        "mapping": (
            "namespace_block_base + (family_index*16+stratum_index)*64 "
            "+ attempt_index"
        ),
        "safe_integer_domain_inclusive": [
            SEED_DOMAIN_MINIMUM,
            SEED_DOMAIN_MAXIMUM,
        ],
        "generated_seed_count": MAX_CANDIDATE_COUNT,
        "generated_seed_unique_count": len(set(PROCEDURAL_SEED_VALUES)),
        "generated_sorted_unique_no_lf_sha256": (
            PROCEDURAL_SEED_SORTED_PROJECTION_SHA256
        ),
        "collision_policy": "FAIL_CLOSED_NO_REHASH_OR_SEED_SUBSTITUTION",
        "v4_registered_seed_overlap_count": 0,
        "required_predecessor_seed_registries": copy.deepcopy(
            REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY
        ),
        "every_enumerated_predecessor_registry_must_be_reopened_and_compared": True,
        "all_predecessor_seed_overlap_counts_required_zero": True,
    }
)


def validate_seed_nonoverlap(
    predecessor_seed_registries: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Validate exact predecessor registries and prove zero seed overlap."""

    if (
        not isinstance(predecessor_seed_registries, Mapping)
        or set(predecessor_seed_registries)
        != set(REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "predecessor seed registry inventory drift"
        )
    generated = set(PROCEDURAL_SEED_VALUES)
    projections: list[dict[str, Any]] = []
    for name in sorted(REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY):
        raw = predecessor_seed_registries[name]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(
            raw, Sequence
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{name} is not a seed sequence"
            )
        values = list(raw)
        if any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or item >= 1 << 63
            for item in values
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{name} contains an unsafe seed"
            )
        if values != sorted(set(values)):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{name} is not sorted unique"
            )
        expected = REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY[name]
        observed_sha = _canonical_no_lf_sha256(values)
        if len(values) != expected["count"] or observed_sha != expected[
            "canonical_sorted_unique_no_lf_sha256"
        ]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{name} binding drift"
            )
        overlap = sorted(generated.intersection(values))
        if overlap:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{name} overlaps successor seeds"
            )
        projections.append(
            {
                "registry": name,
                "count": len(values),
                "canonical_sorted_unique_no_lf_sha256": observed_sha,
                "successor_overlap_count": 0,
            }
        )
    return {
        "generated_seed_count": len(PROCEDURAL_SEED_VALUES),
        "generated_seed_unique_count": len(generated),
        "registries": projections,
        "all_overlap_counts_zero": True,
    }


PREDECESSOR_IDENTITY_PROJECTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "predecessor_identity_projection_authority.v1"
        ),
        "broad_predecessor_identity_projection": {
            "scene_identity": {
                "count": 1391,
                "canonical_sorted_unique_no_lf_sha256": (
                    "32716a7c25791b67885e0660ed948ed4b3c5fa61495dda712fcb4af23c0e3b35"
                ),
            },
            "scene_identity_sha256": {
                "count": 1583,
                "canonical_sorted_unique_no_lf_sha256": (
                    "ba6b05288b523baeb46a90e3fa9eaa4104957d3a511c0ccb1da615c1a55b2b33"
                ),
            },
            "episode_or_state_identity": {
                "count": 492,
                "canonical_sorted_unique_no_lf_sha256": (
                    "1862748709d3aa9b7df15a23eff3d070cf1ae158b1c84dded475e7286189afb8"
                ),
            },
            "numeric_seed": {
                "count": 284,
                "canonical_sorted_unique_no_lf_sha256": (
                    "644f6e66db0343aacf58418337d23beafd52181c633078602e62696b09ff86e5"
                ),
            },
            "textual_path_geometry_or_source_identity": {
                "count": 131860,
                "canonical_sorted_unique_no_lf_sha256": (
                    "1c486aa5039d8f2f725680db517e1b95b0d3814137a2b511ae6dbaca3c0cf177"
                ),
            },
            "structured_waypoint_or_sequence_path": {
                "count": 40,
                "canonical_sorted_unique_no_lf_sha256": (
                    "e4fb713698fe98d2e47f2334cb4aa5a482e5a2eeb71b7e2e641d7e577e148605"
                ),
                "intersection_not_applied": (
                    "structured family and stratum semantics are scientifically invariant"
                ),
            },
        },
        "v4_registered_and_fixture_identity_count": 1300,
        "v4_registered_and_fixture_identity_projection_sha256": (
            "8896c935edba36e039379fe6d19d0f1641335777acbd04dd6dd4c0c8587e080e"
        ),
        "successor_identity_count": 20480,
        "successor_identity_projection_sha256": (
            "fff38e051c806d7b9589a3bf8f5482eadd47aa1eb8e2db82eb2646a22c81055e"
        ),
        "identity_namespace": IDENTITY_NAMESPACE,
        "forbidden_predecessor_identity_prefixes": list(
            FORBIDDEN_PREDECESSOR_IDENTITY_PREFIXES
        ),
        "structured_semantic_overlap_is_not_an_identity_gate": True,
        "all_identity_and_seed_overlap_counts_required_zero": True,
    }
)

IDENTITY_AND_SEED_NONOVERLAP_FIELDS = frozenset(
    {
        "seed_nonoverlap",
        "broad_predecessor_identity_projection",
        "v4_registered_and_fixture_identity_count",
        "v4_registered_and_fixture_identity_projection_sha256",
        "successor_identity_count",
        "successor_identity_projection_sha256",
        "successor_namespace",
        "successor_namespace_required_prefix",
        "broad_predecessor_identity_overlap_count",
        "broad_predecessor_scene_hash_overlap_count",
        "v4_registered_and_fixture_identity_overlap_count",
        "v4_fixture_seed_overlap_count",
        "structured_semantic_overlap_is_not_an_identity_gate",
        "all_identity_and_seed_overlap_counts_zero",
    }
)


def validate_identity_and_seed_nonoverlap(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        IDENTITY_AND_SEED_NONOVERLAP_FIELDS
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "identity/seed nonoverlap field drift"
        )
    row = copy.deepcopy(dict(value))
    authority = PREDECESSOR_IDENTITY_PROJECTION_AUTHORITY
    if (
        row["broad_predecessor_identity_projection"]
        != authority["broad_predecessor_identity_projection"]
        or row["v4_registered_and_fixture_identity_count"]
        != authority["v4_registered_and_fixture_identity_count"]
        or row["v4_registered_and_fixture_identity_projection_sha256"]
        != authority["v4_registered_and_fixture_identity_projection_sha256"]
        or row["successor_identity_count"]
        != authority["successor_identity_count"]
        or row["successor_identity_projection_sha256"]
        != authority["successor_identity_projection_sha256"]
        or row["successor_namespace"] != IDENTITY_NAMESPACE
        or row["successor_namespace_required_prefix"]
        != f"{IDENTITY_NAMESPACE}-"
        or row["structured_semantic_overlap_is_not_an_identity_gate"] is not True
        or row["all_identity_and_seed_overlap_counts_zero"] is not True
        or any(
            row[field] != 0
            for field in (
                "broad_predecessor_identity_overlap_count",
                "broad_predecessor_scene_hash_overlap_count",
                "v4_registered_and_fixture_identity_overlap_count",
                "v4_fixture_seed_overlap_count",
            )
        )
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "identity/seed nonoverlap projection drift"
        )
    seed = row["seed_nonoverlap"]
    if (
        not isinstance(seed, Mapping)
        or seed.get("generated_seed_count") != MAX_CANDIDATE_COUNT
        or seed.get("generated_seed_unique_count") != MAX_CANDIDATE_COUNT
        or seed.get("all_overlap_counts_zero") is not True
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "seed nonoverlap projection drift"
        )
    expected_registries = REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY
    observed = seed.get("registries")
    if not isinstance(observed, list) or len(observed) != len(expected_registries):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "seed registry projection cardinality drift"
        )
    for item, name in zip(observed, sorted(expected_registries)):
        expected = expected_registries[name]
        if item != {
            "registry": name,
            "count": expected["count"],
            "canonical_sorted_unique_no_lf_sha256": expected[
                "canonical_sorted_unique_no_lf_sha256"
            ],
            "successor_overlap_count": 0,
        }:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                "seed registry projection drift"
            )
    return row


def build_generator_stream_manifest() -> dict[str, Any]:
    candidates = build_candidate_identity_manifest()
    streams = []
    for family in FAMILY_IDS:
        for stratum in range(STRATA_PER_FAMILY):
            index = stream_index(family, stratum)
            streams.append(
                {
                    "stream_index": index,
                    "stream_id": _stream_id(family, stratum),
                    "family": family,
                    "stratum_index": stratum,
                    "maximum_attempt_count": MAX_ATTEMPTS_PER_STREAM,
                    "target_qualified_count": TARGET_QUALIFIED_PER_STREAM,
                    "candidate_indices": [
                        attempt * STREAM_COUNT + index
                        for attempt in range(MAX_ATTEMPTS_PER_STREAM)
                    ],
                    "stop_rule": (
                        "stop only immediately after the fourth QUALIFIED record "
                        "or after attempt_index 63"
                    ),
                }
            )
    return attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_stream_manifest.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "family_order": list(FAMILY_IDS),
            "strata_per_family": STRATA_PER_FAMILY,
            "stream_count": STREAM_COUNT,
            "maximum_attempts_per_stream": MAX_ATTEMPTS_PER_STREAM,
            "target_qualified_per_stream": TARGET_QUALIFIED_PER_STREAM,
            "maximum_candidate_count": MAX_CANDIDATE_COUNT,
            "candidate_registry_and_ledger_order": (
                "attempt_index major, then frozen family order, then stratum_index"
            ),
            "within_stream_execution_order": "attempt_index 0 through the stop boundary",
            "inter_stream_scheduling": "unconstrained and scientifically irrelevant",
            "seed_derivation_authority_content_digest": SEED_DERIVATION_AUTHORITY[
                "content_digest"
            ],
            "identity_namespace": IDENTITY_NAMESPACE,
            "forbidden_predecessor_identity_prefixes": list(
                FORBIDDEN_PREDECESSOR_IDENTITY_PREFIXES
            ),
            "v4_context_authority_content_digest": V4_CONTEXT_AUTHORITY[
                "content_digest"
            ],
            "v4_shortfall_resolution_authority_content_digest": (
                V4_SHORTFALL_RESOLUTION_AUTHORITY["content_digest"]
            ),
            "streams": streams,
            "candidate_identities": candidates,
        }
    )


def validate_generator_stream_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_digest(value)
    expected = build_generator_stream_manifest()
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "generator stream manifest drift"
        )
    return copy.deepcopy(row)


STREAM_MANIFEST_CONTENT_DIGEST = build_generator_stream_manifest()[
    "content_digest"
]

GENERATOR_ALLOCATION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "allocation_authority.v1"
        ),
        "family_order": list(FAMILY_IDS),
        "strata_per_family": STRATA_PER_FAMILY,
        "independent_stream_count": STREAM_COUNT,
        "maximum_attempts_per_stream": MAX_ATTEMPTS_PER_STREAM,
        "target_qualified_per_stream": TARGET_QUALIFIED_PER_STREAM,
        "maximum_total_attempts": MAX_CANDIDATE_COUNT,
        "identity_namespace": IDENTITY_NAMESPACE,
        "forbidden_predecessor_identity_prefixes": list(
            FORBIDDEN_PREDECESSOR_IDENTITY_PREFIXES
        ),
        "stop_rule": (
            "each stream stops only after four qualified states or attempt 63"
        ),
        "candidate_registry_and_ledger_order": (
            "attempt-major over frozen family and stratum order"
        ),
        "within_stream_execution_order": (
            "strict increasing attempt_index 0 through the stop boundary"
        ),
        "inter_stream_scheduling": "unconstrained and scientifically irrelevant",
        "panel_selection": (
            "for each stream choose the qualified candidate with the smallest "
            "(canonical_spec_sha256,candidate_spec_id) after all streams have "
            "reached four qualified states"
        ),
        "panel_selection_is_deterministic_sha_order": True,
        "v4_role_assignment_after_selection_is_unchanged": True,
        "v4_shortfall_resolution_authority_content_digest": (
            V4_SHORTFALL_RESOLUTION_AUTHORITY["content_digest"]
        ),
        "role_assignment_hash_experiment_salt": V4.V3.V2.V1.EXPERIMENT_ID,
        "role_assignment_hash_operation": (
            "exact frozen V1 SHA-256 over canonical compact projection without LF"
        ),
        "stratum_factor_aggregation_order": list(STRATUM_FACTOR_IDS),
        "stratum_factor_aliases": copy.deepcopy(
            STRATUM_FACTOR_ALIAS_AUTHORITY
        ),
        "common_physical_parameter_fields": sorted(
            COMMON_PHYSICAL_PARAMETER_FIELDS
        ),
        "hash_perturbation_parameter_order": list(
            HASH_PERTURBATION_PARAMETER_IDS
        ),
        "attempts_to_qualification_semantics": (
            "one-based number of terminal attempts through the first or fourth "
            "QUALIFIED disposition; null when that qualification ordinal is absent"
        ),
        "correct_port_ever_semantics": (
            "teacher_criteria.teacher_crossed_directed_port is true"
        ),
        "contact_dominant_semantics": (
            "TEACHER_PHYSICS_CONTACT is the unique modal primary disposition "
            "among nonqualified attempts; false for zero rejections or any tie"
        ),
        "common_physical_parameters_semantics": (
            "predeclared exact summary over nonqualified attempts only: canonical "
            "value counts are reported for every frozen stratum axis and every "
            "frozen per-attempt adjustment, with values constant across all "
            "failures reported separately; family and stratum identity remain "
            "explicit and no post-hoc factor is introduced"
        ),
        "common_failure_parameter_order": list(COMMON_FAILURE_PARAMETER_IDS),
        "factor_aggregation_semantics": (
            "aggregate attempts and dispositions over every level of every "
            "frozen stratum factor"
        ),
        "failure_next_decision_policy": {
            "exclusively_turning_junction": TURNING_JUNCTION_GENERATOR_NEXT_DECISION,
            "exclusively_offset_opening": OFFSET_OPENING_GENERATOR_NEXT_DECISION,
            "mixed_or_straight_or_room_loop": PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION,
            "available_panel_has_failure_next_decision": False,
            "teacher_or_policy_change_inference": False,
        },
        "all_streams_at_least_four_status": GENERATOR_PANEL_AVAILABLE,
        "all_streams_at_least_one_but_any_below_four_status": GENERATOR_LOW_YIELD,
        "any_stream_zero_status": GENERATOR_FEASIBILITY_NO_GO,
        "downstream_execution_requires_status": GENERATOR_PANEL_AVAILABLE,
        "stream_manifest_content_digest": STREAM_MANIFEST_CONTENT_DIGEST,
    }
)

GENERATOR_TERMINAL_RECORD_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "candidate_index",
        "stream_index",
        "stream_id",
        "family",
        "stratum_index",
        "attempt_index",
        "candidate_spec_id",
        "canonical_spec_sha256",
        "procedural_seed",
        "disposition",
        "qualified",
        "hard_stop",
        "continuation_authorized",
        "stage_reached",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "state_disposition",
        "material_metadata_binding",
        "material_payload_binding",
        "persisted_array_evidence_sha256",
        "content_digest",
    }
)
GENERATOR_STREAM_METRIC_FIELDS = frozenset(
    {
        "stream_index",
        "stream_id",
        "family",
        "stratum_index",
        "attempt_count",
        "attempt_indices",
        "candidate_indices",
        "qualified_count",
        "nonqualified_count",
        "qualified_candidate_indices",
        "disposition_counts",
        "disposition_rates",
        "attempts_to_first_qualified",
        "attempts_to_fourth_qualified",
        "valid_initial_state_count",
        "valid_initial_state_fraction",
        "teacher_executed_count",
        "teacher_left_source_count",
        "teacher_left_source_fraction_of_teacher_executed",
        "correct_port_ever_count",
        "correct_port_ever_fraction_of_teacher_executed",
        "contact_rejection_count",
        "contact_rejection_fraction",
        "contact_dominant",
        "common_physical_parameters",
        "observed_hash_perturbation_ranges",
        "termination_reason",
        "target_reached",
        "yield_fraction",
        "qualification_rate",
        "rejection_rate",
        "selected_candidate_index",
        "selected_candidate_spec_id",
    }
)
GENERATOR_FAMILY_METRIC_FIELDS = frozenset(
    {
        "family",
        "stream_count",
        "attempt_count",
        "qualified_count",
        "nonqualified_count",
        "qualification_rate",
        "rejection_rate",
        "zero_yield_stream_count",
        "low_yield_stream_count",
        "target_reached_stream_count",
        "minimum_stream_qualified_count",
        "disposition_counts",
        "disposition_rates",
        "valid_initial_state_count",
        "teacher_executed_count",
        "teacher_left_source_count",
        "correct_port_ever_count",
        "contact_rejection_count",
        "contact_dominant_stream_count",
    }
)
GENERATOR_FACTOR_METRIC_FIELDS = frozenset(
    {
        "factor",
        "level",
        "stream_count",
        "attempt_count",
        "qualified_count",
        "nonqualified_count",
        "qualification_rate",
        "rejection_rate",
        "disposition_counts",
        "disposition_rates",
        "zero_yield_stream_count",
        "low_yield_stream_count",
        "target_reached_stream_count",
        "valid_initial_state_count",
        "teacher_executed_count",
        "teacher_left_source_count",
        "correct_port_ever_count",
        "contact_rejection_count",
        "contact_dominant_stream_count",
    }
)
V4_SHORTFALL_RESOLUTION_ROW_FIELDS = frozenset(
    {
        "family",
        "stratum_index",
        "v4_fixed_four_shortfall",
        "successor_attempt_count",
        "successor_qualified_count",
        "attempts_to_first_qualified",
        "attempts_to_fourth_qualified",
        "successor_target_reached",
        "resolution_status",
    }
)
V4_SHORTFALL_RESOLUTION_FIELDS = frozenset(
    {
        "authority_content_digest",
        "canonical_v4_shortfall_stream_count",
        "stream_rows",
        "resolved_stream_count",
        "partial_stream_count",
        "zero_yield_stream_count",
        "all_canonical_v4_shortfalls_resolved",
        "conclusion",
        "supports_shallow_sampling_as_primary_cause",
        "intrinsic_infeasibility_established",
    }
)
GENERATOR_METRICS_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "v4_context_authority_content_digest",
        "stream_manifest_content_digest",
        "allocation_authority_content_digest",
        "terminal_record_count",
        "terminal_unique_identity_count",
        "maximum_candidate_count",
        "expected_stream_count",
        "completed_stream_count",
        "missing_stream_count",
        "duplicated_candidate_identity_count",
        "stream_count",
        "maximum_attempts_per_stream",
        "target_qualified_per_stream",
        "qualified_count",
        "nonqualified_count",
        "hard_stop_count",
        "teacher_execution_count",
        "disposition_counts",
        "disposition_rates",
        "stream_rows",
        "family_rows",
        "stratum_factor_rows",
        "v4_shortfall_resolution",
        "qualification_rate",
        "rejection_rate",
        "zero_yield_streams",
        "low_yield_streams",
        "target_reached_streams",
        "selected_candidate_indices",
        "selected_candidate_spec_ids",
        "status",
        "primary_classification",
        "next_decision",
        "next_decision_evidence",
        "downstream_scientific_execution_authorized",
        "downstream_outcomes_opened",
        "models_trained",
        "development_only",
        "final_evaluation_eligible",
        "content_digest",
    }
)
PANEL_HANDOFF_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "generator_metrics_content_digest",
        "v4_context_authority_content_digest",
        "selected_state_count",
        "selected_candidate_indices",
        "selected_candidate_spec_ids",
        "selected_candidate_specs",
        "selected_terminal_records",
        "family_selected_counts",
        "stratum_selected_counts",
        "selection_rule",
        "v4_downstream_science_authorized",
        "content_digest",
    }
)

FROZEN_ENCODER_CHECKPOINT_SHA256 = (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
FROZEN_RANKER_CHECKPOINT_SHA256 = (
    "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127"
)
if VJEPA_ENCODER_BINDING["checkpoint_sha256"] != FROZEN_ENCODER_CHECKPOINT_SHA256:
    raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
        "frozen V-JEPA encoder checkpoint drift"
    )
if CURRENT_VISUAL_RANKER_BINDING["checkpoint_sha256"] != FROZEN_RANKER_CHECKPOINT_SHA256:
    raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
        "frozen current-visual ranker checkpoint drift"
    )

RESET_COMMAND_RECONSTRUCTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "reset_command_reconstruction_authority.v1"
        ),
        "nominal_requested_command": list(RESET_FIXTURE_COMMAND),
        "raw_requested_command_conversion": "float64(float32(nominal))",
        "command_tick_count": RESET_FIXTURE_COMMAND_TICKS,
        "physics_samples_per_command_tick": PHYSICS_STEPS_PER_COMMAND_TICK,
        "policy_steps_per_command_tick": POLICY_STEPS_PER_COMMAND_TICK,
        "physics_steps_per_policy_step": PHYSICS_STEPS_PER_POLICY_STEP,
        "absolute_lower_bounds_float32": [-0.3, 0.0, -0.5],
        "absolute_upper_bounds_float32": [0.3, 0.0, 0.5],
        "maximum_delta_per_tick_float32": [0.25, 0.0, 0.35],
        "initial_previous_applied_command": (
            "selected snapshot snapshot__previous_applied_command cast to float32"
        ),
        "operation_order": (
            "for each five-tick block, float32 absolute clip then float32 "
            "previous-plus-delta clip in tick order; carry the final float32 "
            "command across blocks; promote each recorded sample to float64"
        ),
        "timestamp_operation_order": (
            "restore snapshot integer simulator time; for each policy step use "
            "float(sim_time_ns)/1e9 + (physics_step+1)*0.002"
        ),
    }
)

HELDOUT_COMPARATOR_ALIAS_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "heldout_comparator_alias_authority.v1"
        ),
        "ordinal_base": 1,
        "internal_condition_order": list(HELDOUT_CONDITION_IDS),
        "user_facing_comparator_order": [
            "DETERMINISTIC_KINEMATICS",
            "FROZEN_CURRENT_VISUAL_RANKER",
            "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
            "PHYSICAL_TEACHER",
        ],
        "user_facing_to_internal_condition_id": {
            "DETERMINISTIC_KINEMATICS": "DETERMINISTIC_KINEMATICS",
            "FROZEN_CURRENT_VISUAL_RANKER": "FROZEN_CURRENT_VISUAL_RANKER",
            "ORACLE_BEST_ADMISSIBLE_CANDIDATE": (
                "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
            ),
            "PHYSICAL_TEACHER": "TEACHER_TRACE",
        },
        "physical_teacher_comparator_ordinal": 4,
        "physical_teacher_semantics": (
            "user-facing name for the frozen TEACHER_TRACE held-out comparator, "
            "which summarizes each selected state's already persisted "
            "qualification teacher trace"
        ),
        "internal_condition_ids_renamed": False,
        "new_teacher_execution": False,
        "model_policy_formula_threshold_or_gate_change": False,
    }
)

FROZEN_DOWNSTREAM_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "frozen_downstream_authority.v1"
        ),
        "target_ids": list(TARGET_IDS),
        "target_feature_order": list(TARGET_FEATURE_ORDER),
        "route_lookahead_distance_m": ROUTE_LOOKAHEAD_DISTANCE_M,
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_count": len(CANDIDATE_IDS),
        "horizon_ticks": copy.deepcopy(HORIZON_TICKS),
        "primary_horizon": PRIMARY_HORIZON,
        "primary_horizon_tick_count": HORIZON_TICKS[PRIMARY_HORIZON],
        "encoder_checkpoint_sha256": FROZEN_ENCODER_CHECKPOINT_SHA256,
        "ranker_checkpoint_sha256": FROZEN_RANKER_CHECKPOINT_SHA256,
        "candidate_bank_coverage_rate_minimum": HANDOFF_GATE[
            "coverage_rate_minimum"
        ],
        "repeatability_rate_minimum": HANDOFF_GATE[
            "repeatability_rate_minimum"
        ],
        "handoff_gate": copy.deepcopy(HANDOFF_GATE),
        "target_materiality": copy.deepcopy(TARGET_MATERIALITY),
        "development_target_selection_lexicographic": copy.deepcopy(
            DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC
        ),
        "heldout_condition_ids": list(HELDOUT_CONDITION_IDS),
        "heldout_comparator_alias_authority": copy.deepcopy(
            HELDOUT_COMPARATOR_ALIAS_AUTHORITY
        ),
        "repeat_branch_ids": list(REPEAT_BRANCH_IDS),
        "repeats_per_branch": REPEATS_PER_BRANCH,
        "reset_command_reconstruction_authority": copy.deepcopy(
            RESET_COMMAND_RECONSTRUCTION_AUTHORITY
        ),
        "primary_classifications": list(PRIMARY_CLASSIFICATIONS),
        "secondary_classifications": list(SECONDARY_CLASSIFICATIONS),
        "next_decision_by_classification": copy.deepcopy(
            NEXT_DECISION_BY_CLASSIFICATION
        ),
        "all_formulas_thresholds_and_classifications_equal_v4": True,
        "forbidden_model_or_policy_changes": [
            "MAP retraining or replacement",
            "place model training",
            "local-route model training",
            "action-predictor model training",
            "safety model training",
            "novelty or beacon discovery",
            "deployment-safety threshold or interpretation change",
        ],
    }
)

IMPLEMENTATION_PERSISTENCE_RAMIFICATIONS_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "implementation_persistence_ramifications.v1"
        ),
        "scientific_change": (
            "sampling depth and seed allocation only; physical and downstream "
            "formulas, thresholds, roles, models, and classifications are unchanged"
        ),
        "v4_success_official_layout_reused": False,
        "successor_exact_official_leaf_count": SUCCESS_OUTPUT_LEAF_COUNT,
        "successor_exact_official_leaves": list(SUCCESS_OUTPUT_LEAVES),
        "evidence_layout_only_changes": [
            "aggregate NPZ official leaves are replaced by transitive bindings to ordinary material shards",
            "graph, pixel, and latent official index leaves are omitted and their exact rows are carried by panel, target, and encoding material evidence",
            "waypoint contracts are named target_contracts.json",
            "repeat evidence is named repeatability.jsonl",
        ],
        "transitive_material_locations": {
            "all_generator_attempts": (
                "qualification/<stream-id>/attempt-NN/{metadata.json,payload.npz}"
            ),
            "selected_reset_and_snapshot": (
                "selected/<state-id>/{metadata.json,payload.npz}"
            ),
            "canonical_pixels_and_latents": (
                "downstream_workspace/encoding/{metadata.json,payload.npz}"
            ),
            "candidate_fanout": (
                "fanout/<state-id>/{metadata.json,payload.npz}"
            ),
            "repeat_execution": (
                "repeatability/<heldout-state-id>/{metadata.json,payload.npz}"
            ),
        },
        "every_omitted_official_aggregate_is_reopened_from_bound_material": True,
        "encoding_runtime_bound_ancillary_evidence": {
            "pixel_order": (
                "exact sorted set of the 64 selected raw-RGB SHA-256 identities"
            ),
            "raw_tokens": "persisted float16 checkpoint output and byte-digest bound",
            "spatial_descriptors": (
                "persisted float32 output formed from the pre-quantization float32 "
                "checkpoint tokens; not falsely rederived from stored float16 tokens"
            ),
            "preprocessed_tensor_sha256": (
                "bound to the frozen external preprocessing implementation and "
                "runtime receipt; preprocessing is not reimplemented by the pure reducer"
            ),
            "checkpoint_inference_is_runtime_bound_not_reexecuted_by_reducer": True,
        },
        "unchanged_downstream_formula_authority_content_digest": (
            FROZEN_DOWNSTREAM_AUTHORITY["content_digest"]
        ),
        "formula_or_threshold_change": False,
        "scientific_interpretation_change": False,
    }
)

V4_SCIENTIFIC_INVARIANCE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "v4_scientific_invariance_authority.v1"
        ),
        "v4_context_authority_content_digest": V4_CONTEXT_AUTHORITY[
            "content_digest"
        ],
        "v4_shortfall_resolution_authority_content_digest": (
            V4_SHORTFALL_RESOLUTION_AUTHORITY["content_digest"]
        ),
        "v4_state_disposition_authority_content_digest": V4.STATE_DISPOSITION_AUTHORITY[
            "content_digest"
        ],
        "v4_binary64_projection_authority_content_digest": V4.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY[
            "content_digest"
        ],
        "v4_snapshot_semantic_serializer_authority_content_digest": V4.V3.SEMANTIC_SERIALIZER_AUTHORITY[
            "content_digest"
        ],
        "v4_snapshot_behavioural_probe_authority_content_digest": V4.V3.BEHAVIOURAL_PROBE_AUTHORITY[
            "content_digest"
        ],
        "v4_teacher_criteria_unchanged": True,
        "v4_state_disposition_taxonomy_and_precedence_unchanged": True,
        "v4_physical_geometry_axes_and_perturbation_formulas_unchanged": True,
        "v4_snapshot_probe_teacher_and_runtime_authorities_unchanged": True,
        "v4_panel_hash_selection_and_role_assignment_unchanged": True,
        "v4_downstream_metrics_gates_thresholds_models_and_next_decisions_unchanged": True,
        "frozen_downstream_authority_content_digest": (
            FROZEN_DOWNSTREAM_AUTHORITY["content_digest"]
        ),
        "implementation_persistence_ramifications": copy.deepcopy(
            IMPLEMENTATION_PERSISTENCE_RAMIFICATIONS_AUTHORITY
        ),
        "unchanged_v4_invariants": [
            {"invariant": name, "preserved": True}
            for name in UNCHANGED_V4_INVARIANT_IDS
        ],
        "missing_stratum_redefinition_forbidden": True,
        "deterministic_sha_panel_selection_preserved": True,
        "v4_role_assignment_after_panel_selection_preserved": True,
        "only_scientific_procedure_change": (
            "sampling depth and seed allocation: replace four fixed candidates "
            "per family-stratum with an independent deterministic stream of at "
            "most 64 fresh attempts, stopping only after four qualified states "
            "or attempt 63"
        ),
        "v4_candidate_outcome_or_runtime_artifact_reuse_authorized": False,
        "v4_is_read_only_context": True,
        "historical_snapshot_or_probe_rerun": False,
        "models_trained": 0,
        "development_only": True,
        "final_evaluation_eligible": False,
    }
)

QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "qualification_candidate_lifecycle_authority.v1"
        ),
        "scope": "successor qualification candidate calls only",
        "candidate_boundary_sequence": [
            "_qualify constructs a complete raw packet or raises",
            (
                "destroy every owned qualification Scene exactly once in "
                "reverse creation order"
            ),
            "clear all owned qualification session references",
            "call lewm_genesis.scene_builder.shutdown_genesis() exactly once",
            "run Python garbage collection",
            "return the raw packet or re-raise the primary exception",
        ],
        "owned_scene_release": {
            "scene_destroy_call": "session.ctx.build.scene.destroy()",
            "destroy_exactly_once": True,
            "destroy_order": "reverse creation order",
            "retain_main_and_probe_sessions_until_candidate_boundary": True,
            "clear_owned_session_references_before_process_global_reset": True,
        },
        "process_global_reset": {
            "public_helper": "lewm_genesis.scene_builder.shutdown_genesis",
            "public_helper_call": "shutdown_genesis()",
            "helper_invocations_per_qualification_candidate_call": 1,
            "helper_calls_genesis_destroy_when_initialized": True,
            "genesis_destroy_call": "gs.destroy()",
            "helper_clears_process_global_initialized_flag_even_on_failure": True,
            "applies_after_final_or_initial_tipped_candidate": True,
            "applies_on_python_exception_path": True,
            "precedes_outer_candidate_validation_and_persistence": True,
            "conditional_on_a_next_attempt": False,
        },
        "next_candidate_reinitialization": {
            "entrypoint": "lewm_genesis.scene_builder.initialize_genesis",
            "trigger": "the next existing build_scene_from_pack call",
            "backend": "unchanged successor physical backend value",
            "seed": "next candidate specification procedural_seed",
            "genesis_seed_mapping": "int(procedural_seed) & 0x7FFFFFFF",
            "eager_or_speculative_reinitialization": False,
        },
        "failure_policy": {
            "cleanup_only_failure_hard_stops_before_return_or_persistence": True,
            "process_global_shutdown_failure_hard_stops_before_return_or_persistence": True,
            "next_reinitialization_failure_hard_stops": True,
            "primary_base_exception_is_preserved": True,
            "simultaneous_cleanup_failure_is_a_deterministic_exception_note": True,
            "fabricated_terminal_disposition_on_lifecycle_failure": False,
        },
        "prohibited_mechanisms": {
            "runner_spawned_child_process": False,
            "launcher_or_child_role_framework": False,
            "partial_stream_resume_after_fault": False,
            "custom_finalizer": False,
            "result_audit_or_custody_logic_in_lifecycle_cleanup": False,
            "lifecycle_cleanup_in_qualify_stream_stage": False,
        },
        "scientific_effect": {
            "lifecycle_only": True,
            "physical_geometry_or_initial_state_change": False,
            "render_or_capture_change": False,
            "physics_step_or_command_change": False,
            "snapshot_probe_teacher_or_evidence_change": False,
            "formula_threshold_gate_or_classification_change": False,
            "candidate_identity_seed_order_or_stop_rule_change": False,
        },
    }
)

EXECUTION_AND_FAULT_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "execution_and_fault_authority.v1"
        ),
        "ordinary_execution": {
            "foreground_python_processes": True,
            "stdout_and_stderr_logs": True,
            "python_faulthandler": True,
            "evidence_formats": ["JSON", "JSONL", "NPZ"],
            "digest_algorithm": "SHA-256",
            "independent_reducer_count": 1,
        },
        "qualification_candidate_lifecycle": copy.deepcopy(
            QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY
        ),
        "prohibited_infrastructure": {
            name: False for name in PROHIBITED_INFRASTRUCTURE
        },
        "pre_panel_ordinary_fault": {
            "discard_incomplete_official_root": True,
            "discard_incomplete_material_root": True,
            "engineering_log_path": str(ENGINEERING_LOG_PATH),
            "engineering_log_is_concise_and_external_to_result_tree": True,
            "engineering_log_is_operator_owned": True,
            "engineering_log_absent_during_ordinary_success": True,
            "engineering_log_creation_trigger": (
                "only a real adjudicated pre-panel implementation or persistence "
                "fault that triggers complete successor namespace discard and restart"
            ),
            "runner_automatic_exception_finaliser_or_log_writer": False,
            "engineering_log_is_non_scientific": True,
            "restart_every_stream_from_deterministic_attempt_zero": True,
            "partial_attempt_or_stream_reuse": False,
        },
        "post_panel_immutability": {
            "source_replacement": False,
            "selected_state_replacement": False,
            "threshold_change": False,
            "result_triggered_rerun": False,
        },
        "external_regeneration_or_custody_receipt": False,
    }
)

EXPECTED_COMMIT_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "expected_commit_authority.v1"
        ),
        "freeze": {
            "parent_commit": V4_RESULT_COMMIT,
            "subject": CONTRACT_FREEZE_COMMIT_SUBJECT,
            "direct_child_required": True,
        },
        "evaluation": {
            "parent_is_successor_freeze_commit": True,
            "subject": RESULT_COMMIT_SUBJECT,
            "empty_same_tree_commit_required": True,
            "scientific_outputs_are_external": True,
        },
        "result_commit_participates_in_result_json_content_identity": False,
        "result_markdown_participates_in_result_json_content_identity": False,
    }
)

# Successor material uses the exact V4 array-byte rules with a new provenance
# marker.  This changes no array bytes, dtype, shape, or semantic digest domain.
NPZ_ARCHIVE_COMMENT = f"{EXPERIMENT_ID}:FRESH"
MATERIAL_FILE_BINDING_FIELDS = frozenset(
    {
        "path",
        "bytes",
        "sha256",
        "nlink",
        "ordinary_regular_file",
        "resolved_path_ancestor_symlink_count",
    }
)
PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS = MATERIAL_FILE_BINDING_FIELDS
_PERSISTED_ARRAY_HASH_AUTHORITY_BODY = copy.deepcopy(
    V4.PERSISTED_ARRAY_HASH_AUTHORITY
)
_PERSISTED_ARRAY_HASH_AUTHORITY_BODY.pop("content_digest", None)
_PERSISTED_ARRAY_HASH_AUTHORITY_BODY.update(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "persisted_array_hash_authority.v1"
        ),
        "npz_archive_comment_utf8": NPZ_ARCHIVE_COMMENT,
        "npz_archive_comment_scope": "every successor material NPZ payload",
        "npz_archive_comment_semantics": (
            "successor-only deterministic container provenance"
        ),
    }
)
_PERSISTED_ARRAY_HASH_AUTHORITY_BODY["digest_field_scope"] = [
    text.replace("V4", "successor")
    for text in V4.PERSISTED_ARRAY_HASH_AUTHORITY["digest_field_scope"]
]
PERSISTED_ARRAY_HASH_AUTHORITY = attach_content_digest(
    _PERSISTED_ARRAY_HASH_AUTHORITY_BODY
)

QUALIFICATION_DISPOSITIONS_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "qualification_state_disposition.v1"
)
QUALIFICATION_MATERIAL_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "qualification_terminal_material.v1"
)
PERSISTED_ARRAY_EVIDENCE_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "persisted_array_evidence.v1"
)
SELECTED_RESET_MATERIAL_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "selected_state_material.v1"
)
SELECTED_RESET_MATERIAL_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "candidate_spec",
        "source_terminal_metadata_binding",
        "source_terminal_payload_binding",
        "snapshot",
        "snapshot_identity",
        "snapshot_payload_sha256",
        "reset_trials",
        "reset_pair_comparison",
        "reset_fixture_passed",
        "rgb_sha256",
        "stage_runtime",
        "backend_runtime",
        "fanout_or_ranker_outcome_opened",
        "models_trained",
        "persisted_array_evidence",
        "payload",
        "content_digest",
    }
)
RESET_TRIAL_MATERIAL_FIELDS = frozenset(
    {
        "trial_index",
        "serialized_restore_used",
        "clone_equivalence_used",
        "restored_snapshot_sha256",
        "post_restore_state_sha256",
        "current_rgb_sha256",
        "base_pose_world",
        "joint_position_sha256",
        "joint_velocity_sha256",
        "controller_state_sha256",
        "rng_state_sha256",
        "requested_command_sequence_sha256",
        "post_slew_applied_command_sequence_sha256",
        "contact_sequence_sha256",
        "termination_reason",
        "stuck",
        "trace_digests",
    }
)
RESET_PAIR_COMPARISON_FIELDS = frozenset(
    {
        "state_id",
        "trial_indices",
        "physics_sample_count",
        "maximum_base_position_error_m",
        "maximum_base_quaternion_component_error",
        "maximum_base_twist_error",
        "maximum_joint_position_error_rad",
        "maximum_joint_velocity_error_rad_s",
        "exact_member_equal",
        "endpoint_position_error_m",
        "endpoint_heading_error_rad",
        "termination_reason_equal",
        "stuck_equal",
        "passed",
    }
)

PANEL_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "physical_runtime_environment",
        "physical_runtime_environment_sha256",
        "selected_reset_runtime_sha256s",
        "generator_metrics_content_digest",
        "panel_handoff_content_digest",
        "state_count",
        "canonical_state_order",
        "role_counts",
        "family_role_counts",
        "selection_rule",
        "states",
        "content_digest",
    }
)
PANEL_STATE_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "stream_index",
        "stream_id",
        "attempt_index",
        "state_id",
        "scene_id",
        "episode_id",
        "graph_id",
        "family",
        "stratum_index",
        "route_direction",
        "passage_width_id",
        "port_distance_id",
        "candidate_spec_id",
        "canonical_spec_sha256",
        "procedural_seed",
        "geometry_sha256",
        "source_node_id",
        "target_node_id",
        "directed_edge_id",
        "role",
        "assignment_sha256",
        "rank_within_family",
        "qualification_terminal_record_content_digest",
        "qualification_material_metadata_binding",
        "qualification_material_payload_binding",
        "selected_material_metadata_binding",
        "selected_material_payload_binding",
        "snapshot_identity",
        "teacher_trace_index",
        "goal_reachable",
        "eligible",
        "eligibility_evidence",
    }
)
PANEL_ELIGIBILITY_FIELDS = frozenset(
    {
        "snapshot_complete",
        "exact_reset_fixture_passed",
        "teacher_trace_contact_free",
        "teacher_valid",
        "directed_port_defined",
        "current_rgb_valid",
        "graph_edge_physically_executable",
    }
)
SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "panel_manifest_content_digest",
        "assignment_algorithm",
        "complete_eligible_population_sha256",
        "heldout_outcomes_opened_before_assignment",
        "role_counts",
        "family_role_counts",
        "assignments",
        "content_digest",
    }
)
SPLIT_ASSIGNMENT_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "role",
        "assignment_sha256",
        "rank_within_family",
    }
)
STATE_SNAPSHOT_INDEX_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "panel_manifest_content_digest",
        "record_count",
        "records",
        "content_digest",
    }
)
STATE_SNAPSHOT_RECORD_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "snapshot_identity",
        "snapshot_payload_sha256",
        "selected_material_metadata_binding",
        "selected_material_payload_binding",
        "previous_applied_command_sha256",
        "rgb_sha256",
        "reset_fixture_passed",
        "reset_pair_comparison",
    }
)
TEACHER_TRACE_INDEX_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "generator_terminal_record_count",
        "teacher_executed_count",
        "selected_teacher_trace_count",
        "records",
        "content_digest",
    }
)
TEACHER_TRACE_RECORD_FIELDS = frozenset(
    {
        "trace_index",
        "qualification_candidate_index",
        "qualification_material_path",
        "qualification_material_metadata_binding",
        "qualification_material_payload_binding",
        "candidate_spec_id",
        "state_id",
        "family",
        "stratum_index",
        "attempt_index",
        "canonical_spec_sha256",
        "teacher_trace_id",
        "selected",
        "panel_index",
        "sample_count",
        "trace_digests",
        "teacher_reduction_sha256",
        "raw_projection",
    }
)
EDGE_PORT_INDEX_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "panel_manifest_content_digest",
        "teacher_trace_index_content_digest",
        "record_count",
        "records",
        "content_digest",
    }
)
EDGE_PORT_RECORD_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "directed_edge_id",
        "source_node_id",
        "target_node_id",
        "teacher_trace_index",
        "teacher_trace_id",
        "source_boundary_polygon_world",
        "opening_segment_world",
        "port_normal_world",
        "lateral_bounds_m",
        "route_polyline_world",
        "crossing_sample_before",
        "crossing_sample_after",
        "crossing_fraction",
        "teacher_crossing_time_s",
        "teacher_crossing_velocity_world_xy",
        "teacher_crossing_velocity_heading_world_rad",
        "directed_port_world",
        "route_lookahead_world",
        "route_lookahead_clipped",
        "remaining_route_length_m",
        "port_definition",
    }
)
TARGET_CONTRACTS_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "panel_manifest_content_digest",
        "edge_port_index_content_digest",
        "target_ids",
        "feature_order",
        "training_contract_binding",
        "original_ranker_support",
        "contract_difference_inventory",
        "target_support_counts",
        "rows",
        "content_digest",
    }
)
TARGET_CONTRACT_ROW_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "target_id",
        "source_body_pose_world",
        "target_world_pose",
        "target_body_pose",
        "dx_m",
        "dy_m",
        "distance_m",
        "relative_heading_rad",
        "relative_heading_sin",
        "relative_heading_cos",
        "target_tangent_heading_rad",
        "target_tangent_heading_sin",
        "target_tangent_heading_cos",
        "route_intent_dx",
        "route_intent_dy",
        "roundtrip_world_pose",
        "position_transform_error_m",
        "heading_transform_error_rad",
        "transform_valid",
    }
)

CANONICAL_ENCODING_MATERIAL_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "canonical_encoding_material.v1"
)
CANONICAL_ENCODING_MATERIAL_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "panel_manifest_binding",
        "split_manifest_binding",
        "state_rows",
        "unique_pixel_count",
        "pixel_order",
        "encoding_rows",
        "encoder_binding",
        "preprocessing_authority",
        "external_encoder_source",
        "encoder_runtime_environment",
        "batch_size",
        "fanout_outcomes_opened",
        "models_trained",
        "persisted_array_evidence",
        "payload",
        "content_digest",
    }
)
ENCODING_STATE_ROW_FIELDS = frozenset(
    {
        "panel_index",
        "state_id",
        "qualification_candidate_index",
        "pixel_sha256",
        "canonical_pixel_index",
    }
)
ENCODING_ROW_FIELDS = frozenset(
    {
        "canonical_pixel_index",
        "pixel_sha256",
        "preprocessed_tensor_sha256",
        "raw_token_sha256",
        "spatial_descriptor_sha256",
    }
)
CANDIDATE_FANOUT_MATERIAL_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "candidate_fanout_material.v1"
)
CANDIDATE_FANOUT_MATERIAL_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "role",
        "candidate_spec_id",
        "canonical_spec_sha256",
        "source_selected_metadata_binding",
        "source_selected_payload_binding",
        "development_target_selection_binding",
        "development_target_selection_opened",
        "snapshot_payload_sha256",
        "directed_port_world",
        "branch_count",
        "outcome_rows",
        "stage_runtime",
        "backend_runtime",
        "models_trained",
        "persisted_array_evidence",
        "payload",
        "content_digest",
    }
)
REPEATABILITY_MATERIAL_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1."
    "repeatability_material.v1"
)
REPEATABILITY_MATERIAL_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "role",
        "source_fanout_metadata_binding",
        "source_fanout_payload_binding",
        "heldout_scores_binding",
        "repeat_count",
        "outcome_rows",
        "stage_runtime",
        "backend_runtime",
        "models_trained",
        "persisted_array_evidence",
        "payload",
        "content_digest",
    }
)

CANDIDATE_OUTCOME_FIELDS = frozenset(
    {
        "branch_candidate_id",
        "branch_candidate_index",
        "requested_commands",
        "post_slew_applied_commands",
        "physics_sample_count",
        "port_crossing_sample_before",
        "port_crossing_sample_after",
        "port_crossing_fraction",
        "port_crossing_directed_normal_dot",
        "port_crossing_lateral_fraction",
        "port_crossing_displacement_world_xy",
        "port_crossing_direction_heading_world_rad",
        "beyond_port_consecutive_physics_samples",
        "target_entered_before_dwell_complete",
        "competing_port_crossing_first",
        "first_wrong_edge_id",
        "wrong_port_crossing_sample_before",
        "wrong_port_crossing_sample_after",
        "wrong_port_crossing_fraction",
        "wrong_port_crossing_directed_normal_dot",
        "wrong_port_crossing_lateral_fraction",
        "wrong_port_crossing_displacement_world_xy",
        "wrong_port_crossing_direction_heading_world_rad",
        "h1_endpoint_body",
        "h2_endpoint_body",
        "h3_endpoint_body",
        "h3_endpoint_world",
        "h3_base_height_m",
        "h3_roll_rad",
        "h3_pitch_rad",
        "h3_solver_finite",
        "h3_disallowed_contact",
        "physics_contact",
        "stuck",
        "successor_viable",
        "entered_correct_edge",
        "entered_wrong_edge",
        "no_edge",
        "port_progress_m",
        "lateral_error_m",
        "angular_error_rad",
        "oracle_admissible",
        "positive_port_progress",
        "endpoint_node_id",
        "endpoint_edge_id",
        "left_source_region",
        "first_source_exit_sample_index",
        "reached_target_node",
        "first_target_entry_sample_index",
        "command_tracking_rows",
    }
)
REPEAT_OUTCOME_FIELDS = frozenset(
    CANDIDATE_OUTCOME_FIELDS - {"branch_candidate_id", "branch_candidate_index"}
)
CANDIDATE_FANOUT_ROW_FIELDS = frozenset(
    {
        "branch_id",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "role",
        "candidate_spec_id",
        "branch_candidate_index",
        "branch_candidate_id",
        "snapshot_payload_sha256",
        "material_metadata_binding",
        "material_payload_binding",
        "outcome_row_index",
        "trace_digests",
        *CANDIDATE_OUTCOME_FIELDS,
    }
)
DEVELOPMENT_TARGET_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "panel_manifest_content_digest",
        "target_contracts_content_digest",
        "candidate_fanout_projection_sha256",
        "encoding_material_metadata_binding",
        "encoding_material_payload_binding",
        "role",
        "target_ids",
        "selection_lexicographic",
        "heldout_outcome_documents_opened",
        "state_target_rows",
        "target_summaries",
        "selected_target_id",
        "selected_target_index",
        "selection_frozen",
        "ranker_runtime_environment",
        "models_trained",
        "content_digest",
    }
)
STATE_TARGET_ROW_FIELDS = frozenset(
    {
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "target_id",
        "candidate_ids",
        "scores",
        "ranking",
        "eligible_correct_edge_candidate_indices",
        "selected_candidate_index",
        "correct_edge_top1",
        "correct_edge_top3",
        "correct_edge_mrr",
        "selected_correct_edge_execution",
        "selected_port_progress_m",
        "oracle_best_port_progress_m",
        "minimum_admissible_port_progress_m",
        "normalized_port_regret",
        "target_transform_error_m",
        "pairwise_correct_edge_ordering",
        "selected_wrong_edge",
        "selected_no_edge",
        "selected_lateral_error_m",
        "selected_angular_error_rad",
        "selected_contact",
        "selected_stuck",
        "selected_successor_viable",
    }
)
TARGET_SUMMARY_FIELDS = frozenset(
    {
        "target_id",
        "state_count",
        "selected_correct_edge_execution_rate",
        "correct_edge_top3_rate",
        "correct_edge_top1_rate",
        "correct_edge_mrr",
        "normalized_port_regret",
        "mean_selected_port_progress_m",
        "mean_target_transform_error_m",
        "selection_key",
        "pairwise_correct_edge_ordering",
        "selected_wrong_edge_rate",
        "selected_no_edge_rate",
        "mean_selected_lateral_error_m",
        "mean_selected_angular_error_rad",
        "selected_contact_rate",
        "selected_stuck_rate",
        "selected_successor_viable_rate",
    }
)
HELDOUT_SCORE_ROW_FIELDS = frozenset(
    {
        "score_row_id",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "condition_id",
        "target_id",
        "candidate_ids",
        "scores",
        "ranking",
        "eligible_correct_edge_candidate_indices",
        "selected_candidate_index",
        "correct_edge_top1",
        "correct_edge_top3",
        "correct_edge_mrr",
        "selected_correct_edge_execution",
        "selected_port_progress_m",
        "oracle_best_port_progress_m",
        "minimum_admissible_port_progress_m",
        "normalized_port_regret",
        "teacher_trace_index",
        "teacher_trace_id",
        "teacher_correct_execution",
        "ranker_checkpoint_sha256",
        "pairwise_correct_edge_ordering",
        "selected_wrong_edge",
        "selected_no_edge",
        "selected_lateral_error_m",
        "selected_angular_error_rad",
        "selected_contact",
        "selected_stuck",
        "selected_successor_viable",
        "ranker_runtime_environment_sha256",
    }
)
REPEATABILITY_ROW_FIELDS = frozenset(
    {
        "repeat_id",
        "panel_index",
        "qualification_candidate_index",
        "state_id",
        "family",
        "branch_selector_id",
        "repeat_index",
        "source_candidate_index",
        "source_branch_id",
        "snapshot_payload_sha256",
        "material_metadata_binding",
        "material_payload_binding",
        "outcome_row_index",
        "trace_digests",
        "source_correct_edge_execution",
        "repeat_correct_edge_execution",
        "source_endpoint_body",
        "repeat_endpoint_body",
        "endpoint_position_error_m",
        "endpoint_heading_error_rad",
        "physics_contact",
        "source_applied_command_sequence_sha256",
        "repeat_applied_command_sequence_sha256",
        "source_endpoint_edge_id",
        "repeat_endpoint_edge_id",
        "source_physics_contact",
        "source_stuck",
        "stuck",
        "repeat_success",
        "physical_runtime_core_sha256",
    }
)

SUCCESSOR_METRICS_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "generator_status",
        "primary_classification",
        "secondary_classifications",
        "next_decision",
        "generator_metrics",
        "generator_runtime_environment",
        "source_freeze_observation",
        "material_inventory_projection",
        "panel",
        "downstream",
        "runtime_environments",
        "scientific_counters",
        "models_trained",
        "development_only",
        "final_evaluation_eligible",
        "content_digest",
    }
)
SCIENTIFIC_COUNTER_FIELDS = frozenset(
    {
        "terminal_attempt_count",
        "terminal_unique_identity_count",
        "qualified_count",
        "nonqualified_count",
        "teacher_execution_count",
        "hard_stop_count",
        "selected_state_count",
        "candidate_branch_count",
        "development_target_row_count",
        "heldout_score_row_count",
        "repeatability_row_count",
        "models_trained",
    }
)
RESULT_DOCUMENT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "status",
        "primary_classification",
        "secondary_classifications",
        "next_decision",
        "metrics_content_digest",
        "generator_status",
        "generator_summary",
        "panel_summary",
        "downstream_summary",
        "runtime_environments",
        "source_freeze_observation",
        "material_inventory_projection",
        "scientific_bindings",
        "models_trained",
        "development_only",
        "final_evaluation_eligible",
        "content_digest",
    }
)
RESULT_PUBLICATION_PROJECTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "branch",
        "metrics",
        "scientific_bindings",
        "result_document",
    }
)
SOURCE_FREEZE_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "observed_head_commit_at_scientific_reduction",
        "source_freeze_commit",
        "source_freeze_parent_commit",
        "source_freeze_subject",
        "source_freeze_tree_oid",
        "source_closure_content_digest",
        "source_closure_file_sha256",
        "source_paths_clean",
        "content_digest",
    }
)

MATERIAL_INVENTORY_PROJECTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "root",
        "branch",
        "attempted_candidate_count",
        "file_count",
        "directory_count",
        "files",
        "directories",
        "unexpected_file_count",
        "unexpected_directory_count",
        "v4_physical_shard_file_count",
        "v4_physical_shard_sha256_overlap_count",
        "v4_physical_shard_copy_reuse_detected",
        "content_digest",
    }
)
SNAPSHOT_NUMERIC_MEMBER_SHAPES = {
    "base_pose_world": [7],
    "base_twist_world": [6],
    "joint_position": [12],
    "joint_velocity": [12],
    "controller_observation": [45],
    "policy_last_action": [12],
    "previous_policy_action": [12],
    "previous_applied_command": [3],
    "command_history": [15, 3],
    "control_history": [15, 2],
    "low_level_policy_state": [12],
    "camera_world_transform": [4, 4],
}
SNAPSHOT_NUMERIC_FIELDS = tuple(SNAPSHOT_NUMERIC_MEMBER_SHAPES)
CANDIDATE_TRACE_MEMBER_ORDER = (
    "timestamp_s",
    "base_pose_world",
    "base_twist_world",
    "joint_position",
    "joint_velocity",
    "requested_command",
    "post_slew_applied_command",
    "physics_contact",
    "source_region_member",
    "correct_edge_region_member",
    "wrong_edge_region_member",
    "target_region_member",
)
CANDIDATE_TRACE_MEMBER_SHAPES = {
    "timestamp_s": [],
    "base_pose_world": [7],
    "base_twist_world": [6],
    "joint_position": [12],
    "joint_velocity": [12],
    "requested_command": [3],
    "post_slew_applied_command": [3],
    "physics_contact": [],
    "source_region_member": [],
    "correct_edge_region_member": [],
    "wrong_edge_region_member": [],
    "target_region_member": [],
}
MATERIAL_LAYOUT_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "material_layout_authority.v1"
        ),
        "terminal_file_count_formula": "66 + 2*attempted_candidate_count",
        "terminal_directory_count_formula": "69 + attempted_candidate_count",
        "success_file_count_formula": "356 + 2*attempted_candidate_count",
        "success_directory_count_formula": "214 + attempted_candidate_count",
        "common_files": [
            "material_contract.json",
            "generator_runtime_environment.json",
            "qualification/<stream-id>/stream_completion.json x64",
            "qualification/<stream-id>/attempt-NN/{metadata.json,payload.npz} xA",
        ],
        "success_additional_files": [
            "selected/<state-id>/{metadata.json,payload.npz} x64",
            "fanout/<state-id>/{metadata.json,payload.npz} x64",
            "repeatability/<heldout-state-id>/{metadata.json,payload.npz} x16",
            "downstream_workspace/encoding/{metadata.json,payload.npz} x1",
        ],
        "ordinary_regular_nlink_one_required": True,
        "resolved_path_ancestor_symlink_count_required": 0,
        "hardlink_or_copy_reuse_from_v4_authorized": False,
        "v4_physical_shard_files_compared_by_sha256": 512,
        "v4_physical_shard_sha256_overlap_required": 0,
        "copy_nonreuse_scope": (
            "the 256 bound V4 qualification metadata/payload pairs versus every "
            "successor material metadata/payload shard"
        ),
        "unlisted_material_file_or_directory_authorized": False,
    }
)


def expected_material_inventory_counts(
    attempted_candidate_count: int, *, panel_available: bool
) -> dict[str, int]:
    attempts = _integer(
        attempted_candidate_count,
        "attempted_candidate_count",
        STREAM_COUNT,
        MAX_CANDIDATE_COUNT,
    )
    if type(panel_available) is not bool:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "panel_available drift"
        )
    return {
        "file_count": (356 if panel_available else 66) + 2 * attempts,
        "directory_count": (214 if panel_available else 69) + attempts,
    }

# Explicit historical source-authority alias.  Successor validation never
# substitutes this experiment's population into an inherited module.
V3 = V4.V3


DOC_PREFIX = "docs/lewm_go2_physical_handoff_stratified_generator_successor_v1"
TRACKED_SOURCE_PATHS = (
    f"{DOC_PREFIX}_contract_2026-09-04.json",
    f"{DOC_PREFIX}_fixture_2026-09-04.json",
    f"{DOC_PREFIX}_output_schema_2026-09-04.json",
    f"{DOC_PREFIX}_preregistration_2026-09-04.md",
    f"{DOC_PREFIX}_source_closure_2026-09-04.json",
    f"{DOC_PREFIX}_scientific_invariance_2026-09-04.json",
    f"{DOC_PREFIX}_v4_context_binding_2026-09-04.json",
    "lewm_genesis/lewm_genesis/scene_builder.py",
    "lewm/safety/physical_handoff_stratified_generator_successor_v1_contract.py",
    "lewm/safety/physical_handoff_stratified_generator_successor_v1_metrics.py",
    "lewm/tests/test_physical_handoff_stratified_generator_successor_v1_contract.py",
    "lewm/tests/test_physical_handoff_stratified_generator_successor_v1_metrics.py",
    "lewm/tests/test_run_physical_handoff_stratified_generator_successor_v1.py",
    "lewm/tests/test_evaluate_physical_handoff_stratified_generator_successor_v1.py",
    "scripts/run_physical_handoff_stratified_generator_successor_v1.py",
    "scripts/evaluate_physical_handoff_stratified_generator_successor_v1.py",
)
SOURCE_CLOSURE_PATHS = tuple(
    dict.fromkeys(TRACKED_SOURCE_PATHS[7:] + tuple(V4.SOURCE_CLOSURE_PATHS))
)

RUNTIME_POLICY = {
    "development_only": True,
    "final_evaluation_eligible": False,
    "sealed_path_accesses": 0,
    "ignore_bypasses": 0,
    "models_trained": 0,
    "v4_context_reads": "exact named bindings only",
    "v4_runtime_artifact_or_candidate_outcome_reuse": False,
    "physical_execution": (
        "fresh successor candidate only; exact V4 state disposition, snapshot, "
        "probe, teacher, runtime, and hard-stop authorities"
    ),
    "allocation": copy.deepcopy(GENERATOR_ALLOCATION_AUTHORITY),
    "qualification_candidate_lifecycle": copy.deepcopy(
        QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY
    ),
    "before_downstream_execution": (
        "persist and validate all terminal records and generator metrics; proceed "
        "only for STRATIFIED_PHYSICAL_HANDOFF_PANEL_AVAILABLE"
    ),
    "external_regeneration_or_custody_receipt": False,
    "prohibited_external_publication_authority": copy.deepcopy(
        PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY
    ),
}


def build_v4_context() -> dict[str, Any]:
    return copy.deepcopy(V4_CONTEXT_AUTHORITY)


def validate_v4_context(value: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_digest(value)
    if canonical_json_bytes(row) != canonical_json_bytes(V4_CONTEXT_AUTHORITY):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "V4 context drift"
        )
    return copy.deepcopy(row)


def build_contract() -> dict[str, Any]:
    return attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1.contract.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "development_only": DEVELOPMENT_ONLY,
            "final_evaluation_eligible": FINAL_EVALUATION_ELIGIBLE,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "scientific_questions": list(SCIENTIFIC_QUESTIONS),
            "claims_boundary": copy.deepcopy(CLAIMS_BOUNDARY),
            "output": {
                "root": str(OUTPUT_ROOT),
                "material_root": str(MATERIAL_ROOT),
                "external_regeneration_receipt": None,
                "prohibited_external_publication_authority": copy.deepcopy(
                    PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY
                ),
                "success_leaf_count": SUCCESS_OUTPUT_LEAF_COUNT,
                "success_inventory": list(SUCCESS_OUTPUT_LEAVES),
                "generator_terminal_leaf_count": GENERATOR_TERMINAL_OUTPUT_LEAF_COUNT,
                "generator_terminal_inventory": list(
                    GENERATOR_TERMINAL_OUTPUT_LEAVES
                ),
                "downstream_inventory": list(DOWNSTREAM_OUTPUT_LEAVES),
            },
            "v4_context_authority": copy.deepcopy(V4_CONTEXT_AUTHORITY),
            "v4_shortfall_resolution_authority": copy.deepcopy(
                V4_SHORTFALL_RESOLUTION_AUTHORITY
            ),
            "seed_derivation_authority": copy.deepcopy(
                SEED_DERIVATION_AUTHORITY
            ),
            "generator_allocation_authority": copy.deepcopy(
                GENERATOR_ALLOCATION_AUTHORITY
            ),
            "predecessor_identity_projection_authority": copy.deepcopy(
                PREDECESSOR_IDENTITY_PROJECTION_AUTHORITY
            ),
            "v4_scientific_invariance_authority": copy.deepcopy(
                V4_SCIENTIFIC_INVARIANCE_AUTHORITY
            ),
            "frozen_downstream_authority": copy.deepcopy(
                FROZEN_DOWNSTREAM_AUTHORITY
            ),
            "implementation_persistence_ramifications_authority": copy.deepcopy(
                IMPLEMENTATION_PERSISTENCE_RAMIFICATIONS_AUTHORITY
            ),
            "execution_and_fault_authority": copy.deepcopy(
                EXECUTION_AND_FAULT_AUTHORITY
            ),
            "expected_commit_authority": copy.deepcopy(
                EXPECTED_COMMIT_AUTHORITY
            ),
            "state_disposition_authority": copy.deepcopy(
                V4.STATE_DISPOSITION_AUTHORITY
            ),
            "persisted_array_hash_authority": copy.deepcopy(
                PERSISTED_ARRAY_HASH_AUTHORITY
            ),
            "material_layout_authority": copy.deepcopy(
                MATERIAL_LAYOUT_AUTHORITY
            ),
            "runtime_policy": copy.deepcopy(RUNTIME_POLICY),
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "source_closure_paths": list(SOURCE_CLOSURE_PATHS),
        }
    )


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_digest(value)
    if canonical_json_bytes(row) != canonical_json_bytes(build_contract()):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "contract value drift"
        )
    return copy.deepcopy(row)


RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "status",
        "source_freeze_commit",
        "source_parent_commit",
        "source_baseline_commit",
        "scientific_contract",
        "v4_context",
        "external_artifact_bindings",
        "physical_runtime_policy",
        "successor_runtime_policy",
        "content_digest",
    }
)


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            f"invalid {label}"
        )
    return value


def build_runtime_contract(source_freeze_commit: str) -> dict[str, Any]:
    freeze = _commit(source_freeze_commit, "source_freeze_commit")
    if freeze in {SOURCE_PARENT_COMMIT, V4_SOURCE_FREEZE_COMMIT}:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "successor runtime requires its own source-freeze commit"
        )
    return attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "runtime_contract.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "status": "FROZEN_BEFORE_SUCCESSOR_PHYSICAL_COLLECTION",
            "source_freeze_commit": freeze,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "scientific_contract": build_contract(),
            "v4_context": build_v4_context(),
            "external_artifact_bindings": copy.deepcopy(
                V4.EXTERNAL_ARTIFACT_BINDINGS
            ),
            "physical_runtime_policy": copy.deepcopy(
                V4.V3.V2.V1.DIRECT_RUNTIME_POLICY
            ),
            "successor_runtime_policy": copy.deepcopy(RUNTIME_POLICY),
        }
    )


def validate_runtime_contract(
    value: Mapping[str, Any], *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != set(RUNTIME_CONTRACT_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "runtime contract field drift"
        )
    observed = _commit(row["source_freeze_commit"], "source_freeze_commit")
    if source_freeze_commit is not None and observed != source_freeze_commit:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "source freeze commit drift"
        )
    expected = build_runtime_contract(observed)
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "runtime contract value drift"
        )
    return copy.deepcopy(row)


def build_source_freeze_observation(
    *,
    source_freeze_commit: str,
    source_freeze_tree_oid: str,
    source_closure_content_digest: str,
    source_closure_file_sha256: str,
    observed_head_commit_at_scientific_reduction: str,
) -> dict[str, Any]:
    freeze = _commit(source_freeze_commit, "source_freeze_commit")
    head = _commit(
        observed_head_commit_at_scientific_reduction,
        "observed_head_commit_at_scientific_reduction",
    )
    tree = _commit(source_freeze_tree_oid, "source_freeze_tree_oid")
    for value, label in (
        (source_closure_content_digest, "source_closure_content_digest"),
        (source_closure_file_sha256, "source_closure_file_sha256"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
                f"{label} drift"
            )
    if head != freeze:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "scientific reduction was not observed at source freeze"
        )
    return attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "source_freeze_observation.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "observed_head_commit_at_scientific_reduction": head,
            "source_freeze_commit": freeze,
            "source_freeze_parent_commit": V4_RESULT_COMMIT,
            "source_freeze_subject": CONTRACT_FREEZE_COMMIT_SUBJECT,
            "source_freeze_tree_oid": tree,
            "source_closure_content_digest": source_closure_content_digest,
            "source_closure_file_sha256": source_closure_file_sha256,
            "source_paths_clean": True,
        }
    )


def validate_source_freeze_observation(
    value: Mapping[str, Any], *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != set(SOURCE_FREEZE_OBSERVATION_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "source-freeze observation field drift"
        )
    expected = build_source_freeze_observation(
        source_freeze_commit=row["source_freeze_commit"],
        source_freeze_tree_oid=row["source_freeze_tree_oid"],
        source_closure_content_digest=row["source_closure_content_digest"],
        source_closure_file_sha256=row["source_closure_file_sha256"],
        observed_head_commit_at_scientific_reduction=row[
            "observed_head_commit_at_scientific_reduction"
        ],
    )
    if source_freeze_commit is not None and row["source_freeze_commit"] != (
        source_freeze_commit
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "source-freeze observation runtime cross-link drift"
        )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "source-freeze observation value drift"
        )
    return copy.deepcopy(row)


MATERIAL_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "stream_manifest_content_digest",
        "predecessor_identity_and_seed_nonoverlap",
        "models_trained",
        "content_digest",
    }
)


def build_material_contract(
    source_freeze_commit: str,
    runtime_contract_content_digest: str,
    predecessor_identity_and_seed_nonoverlap: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = _commit(source_freeze_commit, "source_freeze_commit")
    if freeze in {SOURCE_PARENT_COMMIT, V4_SOURCE_FREEZE_COMMIT}:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "material contract requires successor source freeze"
        )
    digest = runtime_contract_content_digest
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "material runtime contract digest drift"
        )
    nonoverlap = validate_identity_and_seed_nonoverlap(
        predecessor_identity_and_seed_nonoverlap
    )
    return attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "material_contract.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "source_freeze_commit": freeze,
            "runtime_contract_content_digest": digest,
            "stream_manifest_content_digest": STREAM_MANIFEST_CONTENT_DIGEST,
            "predecessor_identity_and_seed_nonoverlap": nonoverlap,
            "models_trained": 0,
        }
    )


def validate_material_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != set(MATERIAL_CONTRACT_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "material contract field drift"
        )
    expected = build_material_contract(
        row["source_freeze_commit"],
        row["runtime_contract_content_digest"],
        row["predecessor_identity_and_seed_nonoverlap"],
    )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError(
            "material contract value drift"
        )
    return copy.deepcopy(row)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError",
    "attach_content_digest",
    "build_candidate_identity_manifest",
    "build_candidate_spec",
    "build_candidate_specs",
    "build_contract",
    "build_generator_stream_manifest",
    "build_prospective_pool_specs",
    "build_runtime_contract",
    "build_source_freeze_observation",
    "build_v4_context",
    "candidate_index",
    "canonical_json_bytes",
    "derive_procedural_seed",
    "expected_material_inventory_counts",
    "seed_derivation_sha256",
    "stream_index",
    "validate_candidate_spec",
    "validate_content_digest",
    "validate_contract",
    "validate_generator_stream_manifest",
    "validate_identity_and_seed_nonoverlap",
    "validate_material_contract",
    "validate_runtime_contract",
    "validate_source_freeze_observation",
    "validate_seed_nonoverlap",
    "validate_v4_context",
    "build_material_contract",
]
