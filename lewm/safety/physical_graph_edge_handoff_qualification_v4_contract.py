"""Pure prospective authority for tipped-state handoff qualification V4.

V4 is a development-only continuation of the frozen V3 experiment.  It keeps
the complete V3/V2/V1 scientific design, candidate pool, controller, physical
formulas, thresholds, selection rule, ranker, and classification tree.  Its
only scientific-procedure change is: Invalid tipped boundaries are explicitly
recorded as nonqualified panel candidates instead of aborting the complete
collection.  The terminal-evidence rules also define how already reported
failure predicates are persisted without changing the frozen physical
formulas.  A complete but insufficient 256-state pool therefore produces the
legitimate ``PHYSICAL_HANDOFF_PANEL_INADEQUATE`` scientific terminal instead
of aborting collection.

The first V4 qualification attempt was stopped after pool 127 and before panel
construction when a two-element NumPy accumulation proved byte-dependent on
the producer versus reducer NumPy version.  That partial collection is
invalidated and must not be reused.  V4 now states the frozen NumPy-2 operation
order as deterministic binary64 arithmetic; this is an evidence/reducer
implementation correction and changes no physical formula or decision.

This module is pure: it imports no simulator, model, encoder, or ranker.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
from pathlib import Path
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as _V3


class PhysicalGraphEdgeHandoffV4ContractError(ValueError):
    """Raised when the prospective V4 authority drifts."""


for _name in tuple(_V3.__all__):
    if _name.isupper():
        _value = getattr(_V3, _name)
        try:
            _value = copy.deepcopy(_value)
        except (TypeError, ValueError):
            pass
        globals()[_name] = _value

# Restore the actual immediate-predecessor module after inherited re-exports.
V3 = _V3


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
FINAL_EVALUATION_ELIGIBLE = False
V3_EXPERIMENT_ID = V3.EXPERIMENT_ID
V2_EXPERIMENT_ID = V3.V2_EXPERIMENT_ID
V1_EXPERIMENT_ID = V3.V1_EXPERIMENT_ID
V3_SOURCE_FREEZE_COMMIT = "5b08d433f2e69f6e4fe85c9b14696726e32d2ab1"
V2_SOURCE_FREEZE_COMMIT = V3.V2_SOURCE_FREEZE_COMMIT
V1_SOURCE_FREEZE_COMMIT = V3.V1_SOURCE_FREEZE_COMMIT
SOURCE_PARENT_COMMIT = V3_SOURCE_FREEZE_COMMIT
SOURCE_BASELINE_COMMIT = V3.SOURCE_BASELINE_COMMIT
CONTRACT_FREEZE_COMMIT_SUBJECT = (
    "Freeze tipped-state physical graph edge handoff qualification V4"
)
RESULT_COMMIT_SUBJECT = (
    "Evaluate tipped-state physical graph edge handoff qualification V4"
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v4"
)
MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v4_material"
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v4_regeneration_receipt.json"
)
HISTORICAL_CUSTODY_RECEIPT_PATH = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v1_v2_v3_custody_receipt.json"
)
V1_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1"
V1_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1_material"
V2_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v2"
V2_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v2_material"
V3_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v3"
V3_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v3_material"

V3_TERMINAL_DIAGNOSIS = "TIPPED_BOUNDARY_STATE_DISPOSITION_UNSPECIFIED"
V3_TERMINAL_INTERPRETATION_FIELDS = frozenset(
    {
        "diagnosis", "snapshot_semantic_equivalence_passed",
        "snapshot_behavioural_equivalence_passed",
        "first_eight_reproduction_exact",
        "raw_torch_snapshot_transport_bytes_are_scientific_evidence",
        "prospective_panel_collection_stopped_because_tipped_state_disposition_unspecified",
        "panel_fanout_ranker_or_heldout_outcomes_opened",
        "scientific_handoff_result_produced",
    }
)
V3_TERMINAL_INTERPRETATION = {
    "diagnosis": V3_TERMINAL_DIAGNOSIS,
    "snapshot_semantic_equivalence_passed": True,
    "snapshot_behavioural_equivalence_passed": True,
    "first_eight_reproduction_exact": True,
    "raw_torch_snapshot_transport_bytes_are_scientific_evidence": False,
    "prospective_panel_collection_stopped_because_tipped_state_disposition_unspecified": True,
    "panel_fanout_ranker_or_heldout_outcomes_opened": False,
    "scientific_handoff_result_produced": False,
}
SOLE_SCIENTIFIC_PROCEDURE_CHANGE = (
    "Invalid tipped boundaries are explicitly recorded as nonqualified panel "
    "candidates instead of aborting the complete collection."
)

PRE_PANEL_ENGINEERING_CORRECTION_STATUS = (
    "PARTIAL_QUALIFICATION_INVALIDATED_RESTART_REQUIRED"
)
INVALIDATED_V4_SOURCE_FREEZE_COMMIT = (
    "cb8c1a225550dc0af16626aa5446ad3ac2aaa80a"
)
INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST = (
    "9b844e34ab4693e68fe229831edc82ac72d3fec89fcd2bae75d27ef4efc3af56"
)
INVALIDATED_V4_SCIENTIFIC_CONTRACT_CONTENT_DIGEST = (
    "391d355e47aec470b4e44c7e8ab79c5950a615cf5b033dd8d83470480db21b26"
)
V4_BINARY64_PLANAR_SEGMENT_PROJECTION_FIELDS = (
    "segment_width_m",
    "midpoint_world_xy",
    "unit_tangent_world_xy",
    "lateral_coordinate_m",
)


def canonical_json_bytes(value: Any) -> bytes:
    return V3.canonical_json_bytes(value)


def _canonical_no_lf_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return V3.attach_content_digest(value)


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return V3.validate_content_digest(value)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4ContractError(str(exc)) from exc


INVALIDATED_V4_PARTIAL_ROOT_MANIFEST_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "invalidated_partial_root_manifest_authority.v1"
        ),
        "portable_complete_root_projection_fields": [
            "path", "file_count", "regular_file_apparent_bytes", "files",
        ],
        "portable_file_row_fields": ["path", "bytes", "sha256"],
        "portable_files_sort": "relative POSIX path lexicographic",
        "complete_root_projection_sha256_domain": (
            "SHA-256 of canonical_json_bytes over the complete portable root "
            "projection, including the canonical terminal LF"
        ),
        "files_array_sha256_domain": (
            "SHA-256 of canonical_json_bytes over only the complete sorted "
            "files array, including the canonical terminal LF"
        ),
        "roots": {
            "official": {
                "path": str(OUTPUT_ROOT),
                "file_count": 3,
                "regular_file_apparent_bytes": 150396,
                "complete_root_projection_canonical_byte_count": 552,
                "complete_root_projection_sha256": (
                    "c996f0f1242bc03756198d7eddfab69b6aeacf513150ffda13fdd567589a2766"
                ),
                "files_array_sha256": (
                    "e9c8c9519a744a85b1cccc1746e7d7b473bcb67d999db820f5c469d486c49f20"
                ),
            },
            "material": {
                "path": str(MATERIAL_ROOT),
                "file_count": 258,
                "regular_file_apparent_bytes": 124565766,
                "complete_root_projection_canonical_byte_count": 35783,
                "complete_root_projection_sha256": (
                    "2bd85555bc1e7f7e48039364273982925a0110921e130eee761f332896a02d76"
                ),
                "files_array_sha256": (
                    "00346fd51dc42052143568e0b9bf26462182800eb1071cd8ec305f9772fc6cff"
                ),
            },
        },
        "qualification_material_pair_count": 128,
        "all_observed_leaves_ordinary_single_link_regular_files": True,
        "symlinks_observed": 0,
        "observed_before_invalidated_roots_are_replaced": True,
        "manifest_is_transparency_evidence_not_reusable_scientific_input": True,
        "fresh_recomputation_may_legitimately_reproduce_identical_file_bytes": True,
        "file_hash_inequality_is_nonreuse_proof": False,
    }
)


PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "pre_panel_engineering_correction_authority.v1"
        ),
        "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "invalidated_source_freeze_commit": INVALIDATED_V4_SOURCE_FREEZE_COMMIT,
        "invalidated_runtime_contract_content_digest": (
            INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        ),
        "invalidated_scientific_contract_content_digest": (
            INVALIDATED_V4_SCIENTIFIC_CONTRACT_CONTENT_DIGEST
        ),
        "invalidated_partial_root_manifest_authority": copy.deepcopy(
            INVALIDATED_V4_PARTIAL_ROOT_MANIFEST_AUTHORITY
        ),
        "completed_pool_indices_at_detection": list(range(128)),
        "qualification_material_pair_count_at_detection": 128,
        "official_leaves_at_detection": [
            "contract.json",
            "scientific_invariance_receipt.json",
            "v1_v2_v3_custody_and_nonreuse.json",
        ],
        "detected_before_qualification_ledger_or_panel_construction": True,
        "qualification_ledger_or_panel_adequacy_persisted": False,
        "downstream_outcomes_opened": False,
        "producer_runtime_numpy_version": "2.4.6",
        "independent_reducer_numpy_version": "1.26.4",
        "defect": (
            "two-element NumPy norm/matmul accumulation was not byte-stable "
            "across the frozen producer and independent reducer runtimes"
        ),
        "persisted_crossing_mismatch_pool_indices": [
            66, 76, 78, 85, 89, 95, 102, 103, 104, 105, 106, 110,
            111, 112, 114, 116, 117, 118, 124,
        ],
        "unmaterialized_compact_projection_mismatch_pool_indices": [
            64, 66, 67, 68, 69, 71, 72, 74, 75, 76, 78, 81, 82, 84,
            85, 87, 88, 89, 94, 95, 102, 103, 104, 105, 106, 107, 108,
            110, 111, 112, 114, 116, 117, 118, 120, 122, 124, 126,
        ],
        "affected_fields": [
            "teacher.crossing.lateral_coordinate_m",
            "teacher.competing_crossing.lateral_coordinate_m",
            "teacher_trace_index.records[].crossing_lateral_fraction",
            "teacher_trace_index.records[].endpoint_lateral_error_m",
        ],
        "dependent_fields_recomputed_from_canonical_projection": [
            "state_disposition.teacher_criteria.teacher_within_lateral_bounds",
            "panel_manifest.pool_qualification.teacher_within_lateral_bounds",
        ],
        "pure_projection_fields": list(
            V4_BINARY64_PLANAR_SEGMENT_PROJECTION_FIELDS
        ),
        "binary64_operation_order": {
            "segment_delta": (
                "dx=x1-x0 then dy=y1-y0, each rounded to binary64"
            ),
            "segment_norm_squared": (
                "FMA(dy,dy,round_binary64(dx*dx)) rounded once to binary64"
            ),
            "segment_width": (
                "correctly rounded binary64 square root of segment_norm_squared"
            ),
            "unit_tangent": (
                "tx=dx/segment_width then ty=dy/segment_width, each rounded "
                "to binary64"
            ),
            "midpoint": (
                "mx=round_binary64((x0+x1)/2) then "
                "my=round_binary64((y0+y1)/2)"
            ),
            "relative_point": (
                "rx=point_x-mx then ry=point_y-my, each rounded to binary64"
            ),
            "lateral_coordinate": (
                "FMA(ry,ty,round_binary64(rx*tx)) rounded once to binary64"
            ),
            "software_fma": (
                "convert each finite binary64 operand to its exact rational "
                "value, evaluate a*b+c exactly, then round once to nearest-even "
                "binary64; exact zero is canonical positive zero"
            ),
        },
        "matched_frozen_producer_values_for_available_pool_count": 128,
        "corrected_full_raw_reduction_cross_runtime_exact_for_available_pool_count": 128,
        "corrected_full_raw_reduction_aggregate_sha256_domain": (
            "SHA-256 of canonical_json_bytes over the ordered pool-000..pool-127 "
            "array of complete reduce_qualification_teacher_trace projections, "
            "including the canonical terminal LF"
        ),
        "corrected_full_raw_reduction_aggregate_sha256": (
            "a52b828c76286eb4e9d7ec6a0437db252fe93210c1057706a61f7cb3de97f746"
        ),
        "cross_runtime_exact_match_required": True,
        "existing_partial_material_reuse_authorized": False,
        "file_hash_inequality_is_nonreuse_proof": False,
        "nonreuse_proof": (
            "every replacement terminal binds the corrected runtime contract "
            "content digest and corrected source freeze commit; all 256 bindings "
            "must equal the initialized runtime contract and must differ from "
            "the invalidated source/runtime identities"
        ),
        "restart_from_fresh_v4_roots_required": True,
        "qualification_disposition_or_criterion_changed": False,
        "mathematical_formula_or_tolerance_changed": False,
        "metric_gate_threshold_model_tuning_or_role_rule_changed": False,
        "v1_v2_v3_source_or_evidence_changed": False,
    }
)


DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "development_source_audit_disclosure.v1"
        ),
        "disposition": (
            "ABORTED_DEVELOPMENT_TIME_IGNORE_RULE_BYPASS_ATTEMPT"
        ),
        "operation": "Path('.').rglob('*.py') metadata traversal",
        "matched_file_opens_or_reads": 0,
        "printed_paths": 0,
        "usable_output_items": 0,
        "evidence_derived": False,
        "sealed_content_accessed": False,
        "scientific_outcomes_contaminated": False,
        "replacement_audit_scope": (
            "FROZEN_SOURCE_CLOSURE_PATHS_ONLY_USING_IGNORE_HONORING_TOOLS"
        ),
    }
)


# ---------------------------------------------------------------------------
# Official inventories and terminal scopes
# ---------------------------------------------------------------------------

NEW_RUNTIME_OUTPUT_PATHS = {
    "v1_v2_v3_custody_and_nonreuse": "v1_v2_v3_custody_and_nonreuse.json",
    "scientific_invariance_receipt": "scientific_invariance_receipt.json",
    "qualification_state_dispositions": "qualification_state_dispositions.jsonl",
    "panel_adequacy": "panel_adequacy.json",
}
RUNTIME_OUTPUT_PATHS = {
    **copy.deepcopy(V3.V2.V1.RUNTIME_OUTPUT_PATHS),
    **NEW_RUNTIME_OUTPUT_PATHS,
}
OUTPUT_LEAF_COUNT = 27
SUCCESS_OUTPUT_LEAVES = tuple(RUNTIME_OUTPUT_PATHS.values())
PANEL_INADEQUATE_DISPOSITION = "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
PANEL_INADEQUATE_OUTPUT_LEAVES = (
    "contract.json",
    "v1_v2_v3_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "qualification_state_dispositions.jsonl",
    "panel_adequacy.json",
    "metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
)
PANEL_INADEQUATE_OUTPUT_LEAF_COUNT = 9
TECHNICAL_HARD_STOP_OUTPUT_LEAVES = (
    "contract.json",
    "v1_v2_v3_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
)
NEXT_DECISION_PANEL_INADEQUATE = (
    "REVISE_GENERATOR_FOR_SHORTFALL_FAMILIES_KEEP_TEACHER_CONTRACT_FROZEN"
)


# ---------------------------------------------------------------------------
# Exact disposition taxonomy and material evidence
# ---------------------------------------------------------------------------

STATE_DISPOSITIONS = (
    "QUALIFIED",
    "TEACHER_PHYSICS_CONTACT",
    "TEACHER_CROSSING_INVALID",
    "TEACHER_DID_NOT_LEAVE_SOURCE",
    "TEACHER_NO_POSITIVE_PROGRESS",
    "INITIAL_BOUNDARY_TIPPED",
    "RESTORATION_PROBE_TIPPED",
    "TEACHER_TERMINATED_UNSAFELY",
    "STATE_MATERIALISATION_CORRUPT",
    "STATE_NONDETERMINISTIC",
    "UNRESOLVED_STATE_FAILURE",
)
STATE_DISPOSITION_STAGES = (
    "INITIAL_BOUNDARY",
    "RESTORATION_PROBE",
    "TEACHER_EXECUTION",
    "COMPLETE",
)
HARD_STOP_DISPOSITIONS = (
    "STATE_MATERIALISATION_CORRUPT",
    "STATE_NONDETERMINISTIC",
)
GLOBAL_HARD_STOP_REASONS = (
    "UNSUPPORTED_SNAPSHOT_SERIALIZATION",
    "SNAPSHOT_IDENTITY_REPRODUCTION_FAILURE",
)
DEFINED_NONQUALIFIED_DISPOSITIONS = tuple(
    item for item in STATE_DISPOSITIONS
    if item != "QUALIFIED" and item not in HARD_STOP_DISPOSITIONS
)
DISPOSITION_PRECEDENCE = STATE_DISPOSITIONS[8:10] + (
    "INITIAL_BOUNDARY_TIPPED",
    "RESTORATION_PROBE_TIPPED",
    "TEACHER_TERMINATED_UNSAFELY",
    "TEACHER_PHYSICS_CONTACT",
    "TEACHER_CROSSING_INVALID",
    "TEACHER_DID_NOT_LEAVE_SOURCE",
    "TEACHER_NO_POSITIVE_PROGRESS",
    "UNRESOLVED_STATE_FAILURE",
    "QUALIFIED",
)
TEACHER_CROSSING_VALID_COMPONENT_IDS = (
    "teacher_crossed_directed_port",
    "teacher_no_competing_port",
    "teacher_normal_positive",
    "teacher_within_lateral_bounds",
    "teacher_dwell_satisfied",
    "directed_port_defined",
)
TEACHER_QUALIFICATION_COMPONENT_IDS = tuple(V3.TEACHER_QUALIFICATION_COMPONENT_IDS)
TEACHER_CRITERIA_FIELDS = frozenset(
    set(TEACHER_QUALIFICATION_COMPONENT_IDS) | {"teacher_valid"}
)
TEACHER_TRACE_MEMBER_ORDER = (
    "timestamp_s", "base_pose_world", "base_twist_world", "joint_position",
    "joint_velocity", "applied_command", "requested_command", "physics_contact",
    "source_region_member", "edge_region_member", "target_region_member",
)
TERMINATION_FLAG_ORDER = ("fall", "out_of_bounds", "tipped", "nan")
TERMINATION_FLAGS_FIELDS = frozenset(TERMINATION_FLAG_ORDER)
PARTIAL_PROBE_TERMINATION_REASON_BY_FLAG = {
    "fall": "FALL",
    "out_of_bounds": "OUT_OF_BOUNDS",
    "tipped": "TIPPED",
    "nan": "NAN",
}
PARTIAL_PROBE_NO_FLAG_TERMINATION_REASON = "INCOMPLETE_WITHOUT_TERMINATION_FLAG"

INITIAL_BOUNDARY_PAYLOAD_AUTHORITY = {
    "intended_base_pose_world": {
        "dtype_str": "<f8", "shape": [7],
        "source": "contract-derived spawn xyz plus quaternion xyzw",
    },
    "base_pose_world": {
        "dtype_str": "<f8", "shape": [7],
        "source": "direct no-step simulator observation; quaternion xyzw",
    },
    "base_twist_world": {"dtype_str": "<f8", "shape": [6], "source": "direct no-step observation"},
    "joint_position": {"dtype_str": "<f8", "shape": [12], "source": "direct no-step observation"},
    "joint_velocity": {"dtype_str": "<f8", "shape": [12], "source": "direct no-step observation"},
    "previous_applied_command": {"dtype_str": "<f8", "shape": [3], "source": "direct controller state"},
    "physics_contact": {"dtype_str": "|u1", "shape": [1], "source": "direct no-step contact observation"},
    "sim_time_ns": {"dtype_str": "<i8", "shape": [1], "source": "direct simulator counter"},
    "episode_step": {"dtype_str": "<i8", "shape": [1], "source": "direct episode counter"},
    "command_ticks": {"dtype_str": "<i8", "shape": [1], "source": "direct controller counter"},
    "policy_steps": {"dtype_str": "<i8", "shape": [1], "source": "direct policy counter"},
    "termination_flags": {
        "dtype_str": "|u1", "shape": [4],
        "order": list(TERMINATION_FLAG_ORDER),
        "source": "direct frozen termination predicate",
    },
}
INITIAL_BOUNDARY_DIAGNOSTIC_MEMBER_ORDER = tuple(INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)
INTENDED_BASE_POSE_AUTHORITY = {
    "layout": "[x,y,z,qx,qy,qz,qw]",
    "formula": (
        "[spawn_x,spawn_y,0.375,0,0,sin(spawn_yaw/2),cos(spawn_yaw/2)]"
    ),
    "no_observed_value_substitution": True,
}

V4_PROBE_TRACE_MEMBER_AUTHORITY = copy.deepcopy(V3.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY)
TERMINATED_BEHAVIOURAL_TRACE_FIELDS = frozenset(
    set(V4_PROBE_TRACE_MEMBER_AUTHORITY)
    | {"termination_reason", "stuck", "termination_flags", "tip_sample_index"}
)
TIPPED_BEHAVIOURAL_TRACE_FIELDS = TERMINATED_BEHAVIOURAL_TRACE_FIELDS
TERMINATED_BEHAVIOURAL_PROBE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.terminated_behavioural_probe_authority.v1",
        "trace_fields": sorted(TERMINATED_BEHAVIOURAL_TRACE_FIELDS),
        "trace_member_authority": copy.deepcopy(V4_PROBE_TRACE_MEMBER_AUTHORITY),
        "sample_count_range_inclusive": [1, V3.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES],
        "both_trials_must_have_equal_sample_count_and_terminal_sample_index": True,
        "tip_sample_index_field_means_inclusive_terminal_sample_for_any_partial_probe": True,
        "termination_flags_order": list(TERMINATION_FLAG_ORDER),
        "termination_reason_by_first_true_flag": copy.deepcopy(
            PARTIAL_PROBE_TERMINATION_REASON_BY_FLAG
        ),
        "no_true_flag_termination_reason": PARTIAL_PROBE_NO_FLAG_TERMINATION_REASON,
        "comparison": copy.deepcopy(V3.RESET_TRACE_PAIR_COMPARISON_AUTHORITY),
        "controller_policy_tolerance": V3.BEHAVIOURAL_PROBE_AUTHORITY[
            "controller_policy_samplewise_tolerance"
        ],
        "controller_policy_partial_final_act_rule": (
            "each complete ten-sample policy-act block and any final partial block "
            "repeat the exact observation/action captured for that act"
        ),
        "matching_tipped_trials_disposition": "RESTORATION_PROBE_TIPPED",
        "matching_non_tipped_partial_trials_disposition": "UNRESOLVED_STATE_FAILURE",
        "mismatched_trial_outcomes_or_traces_disposition": "STATE_NONDETERMINISTIC",
        "final_snapshot_semantic_digest_absent": True,
    }
)
TIPPED_BEHAVIOURAL_PROBE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.tipped_behavioural_probe_authority.v1",
        "terminated_probe_authority_content_digest": (
            TERMINATED_BEHAVIOURAL_PROBE_AUTHORITY["content_digest"]
        ),
        "additional_requirement": "both trials have termination_flags.tipped=true",
        "tip_sample_is_included": True,
    }
)
TERMINAL_NONFINITE_INITIAL_MEMBER_IDS = (
    "base_pose_world", "base_twist_world", "joint_position",
    "joint_velocity",
)
TERMINAL_NONFINITE_TRACE_MEMBER_IDS = (
    "base_pose_world", "base_twist_world", "joint_position",
    "joint_velocity",
)
V4_TERMINAL_NONFINITE_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "terminal_nonfinite_authority.v1"
        ),
        "scope": "V4 terminal rejection evidence only",
        "v3_snapshot_semantic_serializer_changed": False,
        "initial_allowed_members": list(TERMINAL_NONFINITE_INITIAL_MEMBER_IDS),
        "trace_allowed_members": list(TERMINAL_NONFINITE_TRACE_MEMBER_IDS),
        "initial_rule": (
            "nan=true requires a nonfinite observed base_pose_world component; "
            "nonfinite values may otherwise occur only in the listed observed "
            "state diagnostics; the intended pose, previous command, counters, "
            "and every unlisted diagnostic remain finite"
        ),
        "trace_rule": (
            "nan=true requires a nonfinite base_pose_world component in the "
            "inclusive final sample; nonfinite values may otherwise occur only "
            "in the listed observed state members of that sample; every earlier "
            "sample and every unlisted floating member remain finite"
        ),
        "persistence": (
            "exact dtype, shape, and C-order IEEE bytes including NaN payload and "
            "signed infinity are bound by persisted_array_evidence"
        ),
        "probe_pair_rule": (
            "nonfinite masks and dtype-sized IEEE bit patterns are exact-equal; "
            "finite cells retain the frozen samplewise tolerances"
        ),
        "teacher_geometry_rule": (
            "crossing, competing-port, progress, membership, endpoint, and stuck "
            "bookkeeping use the maximal finite trace prefix; when that prefix is "
            "empty, the valid captured initial snapshot pose is the sole geometric "
            "anchor; full terminal contact and termination evidence remain bound"
        ),
        "terminal_region_membership_rule": (
            "raw region-membership bytes are rederived from the registered polygon "
            "when terminal x/y are finite, including quaternion-only or z-only "
            "nonfinite termination; if terminal x or y is nonfinite, geometric "
            "membership is undefined and every derived region-membership byte is zero"
        ),
        "teacher_successor_viable_when_nan": False,
        "false_nan_flag_with_any_nonfinite_is_materialisation_corrupt": True,
        "nonfinite_before_final_sample_is_materialisation_corrupt": True,
    }
)
NORMAL_QUALIFICATION_AUGMENTATION_FIELDS = frozenset(
    {"snapshot_semantic_evidence", "snapshot_identity", "behavioural_probe"}
)
SNAPSHOT_IDENTITY_FIELDS = copy.deepcopy(V3.SNAPSHOT_IDENTITY_FIELDS)
PROBE_TIPPED_SNAPSHOT_IDENTITY_FIELDS = frozenset(
    {"artifact_file_sha256", "snapshot_semantic_digest_v1", "snapshot_behavioural_digest_v1"}
)
PROBE_TIPPED_TRIAL_FIELDS = frozenset(
    {
        "trial_index", "trace_member_manifest", "termination_flags",
        "tipped", "tip_sample_index", "contact", "stuck",
        "termination_reason", "final_executable_snapshot_exists",
        "final_snapshot_semantic_digest_v1", "snapshot_behavioural_digest_v1",
    }
)

FAILED_CRITERION_IDS = (
    "state_materialisation_corrupt",
    "state_nondeterministic",
    "initial_boundary_tipped",
    "restoration_probe_tipped",
    "teacher_terminated_unsafely",
    *TEACHER_QUALIFICATION_COMPONENT_IDS,
    "unresolved_state_failure",
)
STATE_DISPOSITION_FIELDS = frozenset(
    {
        "schema", "experiment_id", "pool_index", "candidate_spec_id",
        "state_id", "scene_id", "episode_id", "graph_id", "family",
        "stratum_index", "variant_index", "procedural_seed",
        "canonical_spec_sha256", "disposition", "qualified",
        "failed_criteria", "stage_reached", "hard_stop",
        "continuation_authorized", "executable_snapshot_exists",
        "teacher_executed", "reset_or_candidate_outcome_opened",
        "initial_termination_flags", "probe_trial_termination_flags",
        "teacher_termination_flags", "probe_tip_sample_indices",
        "teacher_criteria", "snapshot_identity", "diagnostics_inventory",
        "payload_member_inventory", "content_digest",
    }
)
MATERIAL_FILE_BINDING_FIELDS = frozenset({"path", "bytes", "sha256"})
QUALIFICATION_DISPOSITION_ROW_FIELDS = frozenset(
    set(STATE_DISPOSITION_FIELDS)
    | {
        "material_metadata_binding", "material_payload_binding",
        "persisted_array_evidence_sha256",
    }
)
BOUNDARY_EVIDENCE_FIELDS = frozenset(
    {
        "intended_pose_representation", "termination_flag_order",
        "available_simulator_diagnostics", "previous_applied_command_sha256",
        "previous_applied_command_dtype", "previous_applied_command_shape",
    }
)
MATERIAL_METADATA_PERSISTENCE_FIELDS = frozenset(
    {"persisted_array_evidence", "payload", "content_digest"}
)
INITIAL_TIPPED_METADATA_FIELDS = frozenset(
    {
        "schema", "experiment_id", "pool_index", "candidate_spec",
        "candidate_spec_sha256", "disposition", "qualified", "rejection_reason",
        "rejection_components", "stage_reached", "executable_snapshot_exists",
        "teacher_executed", "boundary_evidence", "snapshot", "graph", "teacher",
        "state_disposition", "stage_runtime", "backend_runtime",
        "source_freeze_commit", "runtime_contract_content_digest",
        "reset_or_candidate_outcome_opened",
    }
    | MATERIAL_METADATA_PERSISTENCE_FIELDS
)
PROBE_TIPPED_METADATA_FIELDS = frozenset(
    {
        "schema", "experiment_id", "pool_index", "candidate_spec",
        "candidate_spec_sha256", "initial_decision_state_sha256", "snapshot",
        "snapshot_semantic_evidence", "snapshot_identity", "behavioural_probe",
        "current_rgb_sha256", "disposition", "qualified", "rejection_reason",
        "rejection_components", "stage_reached", "executable_snapshot_exists",
        "teacher_executed", "graph", "teacher", "state_disposition",
        "stage_runtime", "backend_runtime", "reset_or_candidate_outcome_opened",
        "source_freeze_commit", "runtime_contract_content_digest",
    }
    | MATERIAL_METADATA_PERSISTENCE_FIELDS
)
TEACHER_TERMINAL_METADATA_FIELDS = frozenset(
    {
        "schema", "experiment_id", "pool_index", "candidate_spec",
        "candidate_spec_sha256", "initial_decision_state_sha256", "snapshot",
        "snapshot_semantic_evidence", "snapshot_identity", "behavioural_probe",
        "graph", "teacher", "current_rgb_sha256", "goal_reachable",
        "graph_edge_physically_executable", "disposition", "qualified",
        "rejection_reason", "rejection_components", "stage_reached",
        "executable_snapshot_exists", "teacher_executed", "contact_instrumentation",
        "state_disposition", "stage_runtime", "backend_runtime",
        "source_freeze_commit", "runtime_contract_content_digest",
        "reset_or_candidate_outcome_opened",
    }
    | MATERIAL_METADATA_PERSISTENCE_FIELDS
)
BEHAVIOURAL_PROBE_METADATA_FIELDS = frozenset(
    {"completed", "trials", "trial_pair_comparison"}
)
MATERIAL_PAYLOAD_BINDING_FIELDS = frozenset(
    {"role", "path", "bytes", "sha256", "kind"}
)
QUALIFICATION_DISPOSITIONS_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v4.qualification_state_disposition.v1"
)
QUALIFICATION_DISPOSITIONS_JSONL_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v4.qualification_state_dispositions_jsonl.v1"
)
STATE_DISPOSITION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.state_disposition_authority.v1",
        "dispositions_in_order": list(STATE_DISPOSITIONS),
        "stages_in_order": list(STATE_DISPOSITION_STAGES),
        "precedence_in_order": list(DISPOSITION_PRECEDENCE),
        "hard_stop_dispositions": list(HARD_STOP_DISPOSITIONS),
        "global_hard_stop_reasons": list(GLOBAL_HARD_STOP_REASONS),
        "defined_nonqualification_continues": True,
        "unresolved_state_failure_continues": True,
        "full_failed_criteria_always_persisted": True,
        "teacher_crossing_valid_component_ids": list(TEACHER_CROSSING_VALID_COMPONENT_IDS),
        "teacher_qualification_component_ids": list(TEACHER_QUALIFICATION_COMPONENT_IDS),
        "state_disposition_fields": sorted(STATE_DISPOSITION_FIELDS),
        "official_jsonl_row_fields": sorted(QUALIFICATION_DISPOSITION_ROW_FIELDS),
        "initial_tipped_metadata_fields": sorted(INITIAL_TIPPED_METADATA_FIELDS),
        "probe_tipped_metadata_fields": sorted(PROBE_TIPPED_METADATA_FIELDS),
        "teacher_terminal_metadata_fields": sorted(TEACHER_TERMINAL_METADATA_FIELDS),
        "boundary_evidence_fields": sorted(BOUNDARY_EVIDENCE_FIELDS),
        "every_pool_has_atomic_metadata_and_payload": True,
        "zero_fill_or_fabricated_evidence_forbidden": True,
        "initial_boundary_payload_authority": copy.deepcopy(INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        "intended_base_pose_authority": copy.deepcopy(INTENDED_BASE_POSE_AUTHORITY),
        "probe_tipped_rule": (
            "both independent fresh-session trials tipped with matching termination and "
            "trace evidence under frozen tolerances; otherwise STATE_NONDETERMINISTIC"
        ),
        "partial_probe_rule": (
            "matching non-tipped partial fresh-session trials persist as "
            "UNRESOLVED_STATE_FAILURE; any trial outcome or trace mismatch under the "
            "frozen authority is STATE_NONDETERMINISTIC"
        ),
        "terminated_behavioural_probe_authority_content_digest": (
            TERMINATED_BEHAVIOURAL_PROBE_AUTHORITY["content_digest"]
        ),
        "tipped_behavioural_probe_authority_content_digest": (
            TIPPED_BEHAVIOURAL_PROBE_AUTHORITY["content_digest"]
        ),
        "terminal_nonfinite_authority_content_digest": (
            V4_TERMINAL_NONFINITE_AUTHORITY["content_digest"]
        ),
        "probe_tipped_final_snapshot_or_digest_must_be_absent": True,
        "valid_probe_rule": "retain exact V3 two-fresh-session restoration probe",
        "normal_teacher_science": "exact inherited V3/V2/V1 teacher formulas",
        "terminal_source_runtime_binding": {
            "metadata_fields": [
                "source_freeze_commit", "runtime_contract_content_digest",
            ],
            "source_freeze_commit": (
                "exact initialized runtime_contract.source_freeze_commit"
            ),
            "runtime_contract_content_digest": (
                "exact initialized runtime_contract.content_digest"
            ),
            "invalidated_source_freeze_commit_rejected": (
                INVALIDATED_V4_SOURCE_FREEZE_COMMIT
            ),
            "invalidated_runtime_contract_content_digest_rejected": (
                INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
            ),
            "all_256_terminal_bindings_must_be_equal": True,
            "file_hash_inequality_is_nonreuse_proof": False,
        },
    }
)

NPZ_ARCHIVE_COMMENT = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4:FRESH"
PERSISTED_ARRAY_HASH_AUTHORITY = copy.deepcopy(V3.PERSISTED_ARRAY_HASH_AUTHORITY)
PERSISTED_ARRAY_HASH_AUTHORITY.update(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.persisted_array_hash_authority.v1",
        "digest_field_scope": [
            "persisted_array_evidence.arrays[].array_bytes_sha256 on every V4 "
            "material shard payload written through the shared corrected writer",
            "snapshot.previous_applied_command_sha256 in every V4 material snapshot",
            "state_snapshot_index.records[].previous_applied_command_sha256 in the "
            "assembled V4 official evidence",
        ],
        "npz_archive_comment_utf8": NPZ_ARCHIVE_COMMENT,
        "npz_archive_comment_scope": (
            "every V4 material shard payload.npz governed by "
            "persisted_array_evidence"
        ),
        "npz_archive_comment_semantics": (
            "V4-only deterministic container provenance; it changes no NPZ member, "
            "dtype, shape, logical value, candidate identity, or scientific outcome"
        ),
    }
)
PERSISTED_ARRAY_ROW_FIELDS = copy.deepcopy(V3.PERSISTED_ARRAY_ROW_FIELDS)
PERSISTED_ARRAY_EVIDENCE_FIELDS = copy.deepcopy(V3.PERSISTED_ARRAY_EVIDENCE_FIELDS)
PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS = copy.deepcopy(V3.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS)


# ---------------------------------------------------------------------------
# Complete-pool adequacy
# ---------------------------------------------------------------------------

PANEL_ADEQUACY_STRATUM_FIELDS = frozenset(
    {
        "family", "stratum_index", "candidate_count", "candidate_pool_indices",
        "qualified_count", "qualified_pool_indices", "disposition_counts",
        "adequate", "selected_pool_index", "selected_candidate_spec_id",
        "shortfall_causes",
    }
)
PANEL_ADEQUACY_FAMILY_FIELDS = frozenset(
    {
        "family", "candidate_count", "qualified_count", "selected_count",
        "shortfall_stratum_count", "shortfall_strata", "disposition_counts",
        "adequate",
    }
)
PANEL_ADEQUACY_FIELDS = frozenset(
    {
        "schema", "experiment_id", "qualification_record_count",
        "all_pool_indices_present", "hard_stop_count", "disposition_counts",
        "qualified_count", "nonqualified_count", "family_rows", "stratum_rows",
        "adequate_stratum_count", "shortfall_stratum_count", "shortfall_strata",
        "shortfall_causes", "offset_opening_supplies_any_qualified_state",
        "selected_pool_indices", "selected_candidate_spec_ids", "panel_state_count",
        "family_role_counts_if_adequate", "adequate", "status",
        "downstream_scientific_execution_authorized", "downstream_outcomes_opened",
        "next_decision", "content_digest",
    }
)
PANEL_ADEQUACY_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.panel_adequacy_authority.v1",
        "candidate_count": V3.PROSPECTIVE_POOL_COUNT,
        "families": list(V3.FAMILY_IDS),
        "strata_per_family": 16,
        "candidates_per_stratum": 4,
        "required_selected_per_stratum": 1,
        "required_selected_per_family": 16,
        "required_panel_state_count": V3.STATE_COUNT,
        "selection_order": "ascending (canonical_spec_sha256,candidate_spec_id)",
        "role_assignment_after_adequacy": copy.deepcopy(V3.FAMILY_ROLE_COUNTS),
        "inadequate_if_any_family_stratum_has_zero_qualified_states": True,
        "offset_opening_supplies_any_qualified_state_must_be_reported": True,
        "inadequate_disposition": PANEL_INADEQUATE_DISPOSITION,
        "inadequate_next_decision": NEXT_DECISION_PANEL_INADEQUATE,
        "inadequate_output_leaves": list(PANEL_INADEQUATE_OUTPUT_LEAVES),
        "success_output_leaves": list(SUCCESS_OUTPUT_LEAVES),
        "candidate_ranker_development_heldout_execution_forbidden_when_inadequate": True,
    }
)

V4_PANEL_TEACHER_SUBSET_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.panel_teacher_subset_authority.v1",
        "qualification_row_count": V3.PROSPECTIVE_POOL_COUNT,
        "qualification_order": "exact registered pool order",
        "pre_teacher_rows": (
            "teacher identity, trace, graph-criterion, and teacher-outcome fields are null; "
            "no placeholder, padding, or fabricated physical value is permitted"
        ),
        "pre_teacher_dispositions": [
            "INITIAL_BOUNDARY_TIPPED",
            "RESTORATION_PROBE_TIPPED",
            "UNRESOLVED_STATE_FAILURE",
        ],
        "teacher_trace_records": "only states for which teacher_executed=true",
        "teacher_trace_record_order": "ascending original registered pool index",
        "teacher_trace_index_semantics": "compact teacher-record and NPZ-slice index",
        "qualification_pool_index_field": "teacher_records[].qualification_pool_index",
        "qualification_pool_index_semantics": "original registered pool index",
        "teacher_npz_slice_order": "compact teacher-record order",
        "qualification_disposition_semantics": (
            "disposition remains QUALIFIED for every scientifically qualified state"
        ),
        "qualification_selection_reason_semantics": (
            "selected qualified row has null rejection_reason; unselected qualified "
            "row has HASH_ORDER_NOT_SELECTED; a nonqualified row has its singular "
            "state disposition"
        ),
        "rejection_reason_counts_semantics": (
            "counts only nonqualified singular state dispositions; excludes QUALIFIED "
            "and HASH_ORDER_NOT_SELECTED"
        ),
        "teacher_trace_minimum_count_for_adequate_panel": V3.STATE_COUNT,
        "teacher_trace_maximum_count": V3.PROSPECTIVE_POOL_COUNT,
        "selected_state_count": V3.STATE_COUNT,
        "selected_states_must_be_qualified_teacher_executed": True,
        "complete_disposition_ledger_cross_binding_required": True,
        "metric_formula_scope": (
            "exact inherited V1 aggregation over the selected 64 and all real downstream "
            "rows after V4-native panel/teacher validation"
        ),
        "metric_evidence_count_reports_actual_teacher_trace_count": True,
        "compatibility_placeholder_evidence_forbidden": True,
    }
)
V4_TEACHER_RECORD_ADDITIONAL_FIELDS = frozenset({"qualification_pool_index"})
V4_TEACHER_RECORD_FIELDS = frozenset(
    {
        "candidate_spec_id", "state_id", "family", "stratum_index",
        "variant_index", "canonical_spec_sha256", "teacher_trace_id",
        "trace_index", "qualification_pool_index", "selected",
        "initial_decision_state_sha256", "current_rgb_sha256",
        "current_rgb_valid", "goal_reachable", "directed_port_defined",
        "graph_edge_physically_executable", "source_node_id",
        "target_node_id", "directed_edge_id", "trace_slice",
        "trace_array_slice_sha256s", "sample_count", "contact_free",
        "left_source_region", "first_source_exit_sample_index",
        "first_crossing_sample_index", "crossing_segment_fraction",
        "crossed_directed_port", "positive_route_progress", "route_progress_m",
        "crossing_directed_normal_dot", "crossing_lateral_fraction",
        "crossing_velocity_world_xy", "crossing_velocity_heading_world_rad",
        "beyond_port_consecutive_physics_samples",
        "target_entered_before_dwell_complete", "competing_port_entered",
        "reached_target_node", "first_target_entry_sample_index",
        "where_reached", "endpoint_lateral_error_m",
        "endpoint_angular_error_rad", "successor_viable", "stuck",
        "teacher_valid",
    }
)
V4_POOL_QUALIFICATION_ADDITIONAL_FIELDS = frozenset({"disposition"})
V4_POOL_QUALIFICATION_FIELDS = frozenset(
    {
        "candidate_spec_id", "state_id", "family", "stratum_index",
        "variant_index", "canonical_spec_sha256", "teacher_trace_id",
        "teacher_trace_index", "teacher_trace_slice_sha256",
        "initial_decision_state_sha256", "goal_reachable",
        "teacher_trace_contact_free", "teacher_left_source_region",
        "teacher_crossed_directed_port", "teacher_positive_route_progress",
        "teacher_competing_port_entered", "teacher_normal_positive",
        "teacher_within_lateral_bounds", "teacher_dwell_satisfied",
        "directed_port_defined", "current_rgb_valid",
        "graph_edge_physically_executable", "teacher_valid", "qualified",
        "rejection_reason", "selection_key_sha256", "rank_within_stratum",
        "selected", "disposition",
    }
)
V4_TEACHER_MATERIAL_SUMMARY_FIELDS = frozenset(
    {
        "sample_count", "trace_digests", "contact_free",
        "left_source_region", "positive_route_progress", "route_progress_m",
        "competing_port_entered", "competing_crossing", "reached_target_node",
        "crossing", "crossing_error", "termination_flags",
        "terminated_unsafe", "teacher_valid",
    }
)
V4_RAW_TEACHER_REDUCTION_FIELDS = frozenset(
    {
        "pool_index", "candidate_spec_id", "sample_count",
        "goal_reachable", "graph_edge_physically_executable",
        "termination_flags", "contact_free", "left_source_region",
        "first_source_exit_sample_index", "crossing", "crossing_error",
        "competing_crossing", "competing_port_entered", "route_progress_m",
        "positive_route_progress", "reached_target_node", "teacher_valid",
        "stuck", "teacher_criteria", "disposition", "failed_criteria",
        "teacher_record_raw_projection",
    }
)
V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS = frozenset(
    {
        "sample_count", "contact_free", "left_source_region",
        "first_source_exit_sample_index", "first_crossing_sample_index",
        "crossing_segment_fraction", "crossed_directed_port",
        "positive_route_progress", "route_progress_m",
        "crossing_directed_normal_dot", "crossing_lateral_fraction",
        "crossing_velocity_world_xy",
        "crossing_velocity_heading_world_rad",
        "beyond_port_consecutive_physics_samples",
        "target_entered_before_dwell_complete", "competing_port_entered",
        "reached_target_node", "first_target_entry_sample_index",
        "where_reached", "endpoint_lateral_error_m",
        "endpoint_angular_error_rad", "successor_viable", "stuck",
        "teacher_valid", "goal_reachable", "directed_port_defined",
        "graph_edge_physically_executable",
    }
)
V4_TEACHER_TERMINATION_DERIVATION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.teacher_termination_derivation_authority.v1",
        "source": "frozen production boundary predicates evaluated on the inclusive final teacher sample",
        "base_pose_quaternion_order": "xyzw",
        "fall_z_threshold_m": 0.15,
        "world_bounds_xy_m": [[-4.0, -4.0], [4.0, 4.0]],
        "out_of_bounds_pad_m": 0.5,
        "tip_threshold_rad": 1.0471975511965976,
        "nan_rule": "any nonfinite base position or quaternion component",
        "all_four_flags_are_rederived": True,
        "partial_trace_includes_the_first_terminal_sample": True,
    }
)
TEACHER_TRACE_DT_S = 0.002
V4_RAW_TEACHER_REDUCTION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.raw_teacher_reduction_authority.v1",
        "fields": sorted(V4_RAW_TEACHER_REDUCTION_FIELDS),
        "teacher_record_raw_projection_fields": sorted(
            V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS
        ),
        "terminal_nonfinite_authority_content_digest": (
            V4_TERMINAL_NONFINITE_AUTHORITY["content_digest"]
        ),
        "trace_members": list(TEACHER_TRACE_MEMBER_ORDER),
        "physics_sample_period_s": TEACHER_TRACE_DT_S,
        "geometry_source": "exact registered candidate specification at qualification pool index",
        "membership_rederived_from_pose_and_registered_polygons": True,
        "first_port_crossing": copy.deepcopy(V3.V2.V1.PHYSICAL_OUTCOME_AUTHORITY["entered_correct_edge"]),
        "competing_port_order": "sample-after index, exact crossing fraction, lexical edge_id",
        "route_progress": "initial distance to registered selected-opening midpoint minus minimum trace distance",
        "dwell_physics_samples": V3.PORT_DWELL_PHYSICS_SAMPLES,
        "stuck_authority": copy.deepcopy(V3.PHYSICAL_OUTCOME_AUTHORITY["stuck"]),
        "termination_authority_content_digest": V4_TEACHER_TERMINATION_DERIVATION_AUTHORITY["content_digest"],
        "deterministic_binary64_projection_authority_content_digest": (
            PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY["content_digest"]
        ),
        "graph_rebuilt_from_registered_geometry": True,
        "metadata_summary_is_non_authoritative_until_exact_raw_cross_link": True,
    }
)

TEACHER_SELECTION_FIELDS = frozenset(
    {
        "schema", "experiment_id", "candidate_specs_sha256",
        "teacher_traces_file", "teacher_records", "qualification_rows",
        "qualification_projection_sha256", "selected_specs",
        "selected_pool_indices", "rejection_reason_counts",
        "panel_adequacy_sha256", "fanout_or_ranker_outcomes_opened",
        "content_digest",
    }
)
TEACHER_SELECTION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.teacher_selection_authority.v1",
        "fields": sorted(TEACHER_SELECTION_FIELDS),
        "location": "material/teacher_selection.json",
        "ordinary_canonical_content_digested_json": True,
        "qualification_row_fields": sorted(V4_POOL_QUALIFICATION_FIELDS),
        "teacher_record_fields": sorted(V4_TEACHER_RECORD_FIELDS),
        "teacher_trace_mapping": copy.deepcopy(V4_PANEL_TEACHER_SUBSET_AUTHORITY),
        "selected_state_count": V3.STATE_COUNT,
        "cross_bindings": [
            "qualification_state_dispositions.jsonl",
            "panel_adequacy.json",
            "panel_manifest.json",
            "teacher_trace_index.json",
            "teacher_traces.npz",
        ],
        "placeholder_or_padded_teacher_records_forbidden": True,
    }
)

PANEL_CONTEXT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "selected_state_ids", "role_by_state",
        "panel_sha256", "split_sha256", "heldout_outcomes_opened",
        "content_digest",
    }
)
ENCODING_RECEIPT_RECORD_FIELDS = frozenset(
    {
        "canonical_pixel_index", "pixel_sha256",
        "preprocessed_tensor_sha256", "raw_token_sha256",
        "spatial_descriptor_sha256",
    }
)
ENCODING_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "singleton_count", "pixel_order",
        "records", "preprocessing_authority", "external_encoder_source",
        "encoder_runtime_environment", "fanout_outcomes_opened",
        "content_digest",
    }
)
MATERIAL_INVENTORY_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.material_inventory_authority.v1",
        "atomic_shard_files": ["metadata.json", "payload.npz"],
        "qualification_shard_count": V3.PROSPECTIVE_POOL_COUNT,
        "selected_shard_count_on_success": V3.STATE_COUNT,
        "fanout_shard_count_on_success": V3.STATE_COUNT,
        "repeat_shard_count_on_success": V3.V2.V1.ROLE_COUNTS[
            "DEVELOPMENT_HELDOUT"
        ],
        "panel_inadequate_root_files": [
            "material_contract.json", "prospective_pool.json",
        ],
        "success_root_files": [
            "material_contract.json", "prospective_pool.json",
            "teacher_selection.json", "panel_context.json",
            "encoding_receipt.json",
        ],
        "panel_inadequate_exact_file_count": 514,
        "success_exact_file_count": 805,
        "panel_context_fields": sorted(PANEL_CONTEXT_FIELDS),
        "encoding_receipt_fields": sorted(ENCODING_RECEIPT_FIELDS),
        "encoding_receipt_record_fields": sorted(
            ENCODING_RECEIPT_RECORD_FIELDS
        ),
        "no_unlisted_material_files": True,
    }
)


def teacher_trace_npz_authority(teacher_trace_count: int) -> dict[str, Any]:
    if (
        not isinstance(teacher_trace_count, int)
        or isinstance(teacher_trace_count, bool)
        or not V3.STATE_COUNT <= teacher_trace_count <= V3.PROSPECTIVE_POOL_COUNT
    ):
        raise PhysicalGraphEdgeHandoffV4ContractError(
            "adequate-panel teacher trace count drift"
        )
    authority = copy.deepcopy(V3.NPZ_PAYLOAD_AUTHORITY["teacher_traces.npz"])
    authority["trace_offsets"]["shape"] = [teacher_trace_count + 1]
    return authority


# ---------------------------------------------------------------------------
# Qualification runtime custody
# ---------------------------------------------------------------------------

QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS = (
    "backend", "n_envs", "physics_dt_s", "policy_dt_s", "command_dt_s",
    "policy_evaluation_mode", "policy_device", "simulate_action_latency",
    "contact_api", "forbidden_net_force_api_used", "snapshot_restore_source",
    "solver_field_collector", "ppo_observation_contact_inputs",
    "foot_contact_source_zero_scope",
)
QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS = (
    "snapshot_captured_before_teacher",
    "teacher_restored_from_serialized_snapshot",
    "teacher_snapshot_sha256",
)
QUALIFICATION_RUNTIME_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "source_freeze_commit",
        "runtime_contract_content_digest", "external_artifact_bindings_sha256",
        "historical_custody_receipt_binding", "physical_runtime_core",
        "physical_runtime_core_sha256", "backend_runtime_core",
        "backend_runtime_core_sha256", "qualification_shard_count",
        "qualification_stage_runtime_sha256s",
        "qualification_backend_runtime_sha256s",
        "all_qualification_stage_runtime_cores_equal",
        "all_qualification_backend_runtime_cores_equal", "fake_runtime",
        "models_trained", "prohibited_components_trained_or_implemented",
        "content_digest",
    }
)
QUALIFICATION_RUNTIME_MATERIAL_HASH_FIELDS = (
    "stage_runtime_sha256", "backend_runtime_sha256",
    "backend_runtime_core_sha256",
)
QUALIFICATION_BACKEND_RUNTIME_AUTHORITY = {
    "backend": RUNTIME_ENVIRONMENT_AUTHORITY["physical"]["backend"],
    "n_envs": 1,
    "physics_dt_s": 0.002,
    "policy_dt_s": 0.02,
    "command_dt_s": 0.1,
    "policy_evaluation_mode": True,
    "policy_device": RUNTIME_ENVIRONMENT_AUTHORITY["physical"]["device"],
    "simulate_action_latency": True,
    "contact_api": "robot.get_contacts(exclude_self_contact=False)",
    "forbidden_net_force_api_used": False,
    "snapshot_restore_source": "scripts/run_go2_oracle_branch_pilot_v1.py",
    "solver_field_collector": (
        "runner-owned exact reviewed walk port for quadrants 0.6.2"
    ),
    "ppo_observation_contact_inputs": "none_in_frozen_45_element_contract",
    "foot_contact_source_zero_scope": (
        "rollout telemetry placeholder only; physical contact labels use "
        "robot.get_contacts at every 2 ms physics step"
    ),
}
QUALIFICATION_RUNTIME_AUTHORITY = attach_content_digest(
    {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "qualification_runtime_authority.v1"
        ),
        "terminal_metadata_identity": {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4."
                "teacher_pool_terminal.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
        },
        "stage_runtime_fields": list(PHYSICAL_RUNTIME_CORE_FIELDS),
        "projection_fields": sorted(QUALIFICATION_RUNTIME_ENVIRONMENT_FIELDS),
        "material_validation_hash_fields": list(
            QUALIFICATION_RUNTIME_MATERIAL_HASH_FIELDS
        ),
        "mapping_digest_domain": (
            "SHA-256 of canonical compact JSON projection without trailing LF"
        ),
        "external_artifact_bindings_sha256_domain": (
            "exact runtime_contract.external_artifact_bindings array in the "
            "mapping digest domain"
        ),
        "backend_runtime_core_fields": list(
            QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS
        ),
        "teacher_only_backend_runtime_fields": list(
            QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS
        ),
        "backend_runtime_mode_rule": (
            "teacher_executed=false carries exactly the common backend core; "
            "teacher_executed=true adds exactly snapshot_captured_before_teacher=true, "
            "teacher_restored_from_serialized_snapshot=true, and a teacher snapshot "
            "SHA equal to the terminal's initial artifact identity"
        ),
        "backend_runtime_core": copy.deepcopy(
            QUALIFICATION_BACKEND_RUNTIME_AUTHORITY
        ),
        "qualification_shard_count": V3.PROSPECTIVE_POOL_COUNT,
        "terminal_source_runtime_binding_fields": [
            "source_freeze_commit", "runtime_contract_content_digest",
        ],
        "terminal_source_freeze_commit_must_equal_initialized_runtime": True,
        "terminal_runtime_contract_digest_must_equal_initialized_runtime": True,
        "invalidated_source_freeze_commit_rejected": (
            INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        ),
        "invalidated_runtime_contract_content_digest_rejected": (
            INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        ),
        "all_256_terminal_source_runtime_bindings_must_be_identical": True,
        "file_hash_inequality_is_nonreuse_proof": False,
        "stage_runtime_must_be_identical_across_all_shards": True,
        "backend_runtime_core_must_be_identical_across_all_shards": True,
        "full_backend_runtime_sha256_persisted_per_shard": True,
        "runtime_contract_content_digest_bound": True,
        "external_artifact_bindings_projection_bound": True,
        "historical_custody_receipt_binding_bound": True,
        "production_fake_runtime": False,
        "models_trained": 0,
        "prohibited_components_trained_or_implemented": [],
        "synthetic_fixture_rule": (
            "fake runtime evidence may be validated only through an explicit "
            "test-only allowance and must be rejected by the production independent "
            "reducer before publication"
        ),
        "metrics_only_projection_no_additional_leaf": True,
    }
)

V4_STATE_DISPOSITION_METRIC_FIELDS = frozenset(
    {
        "record_count", "qualified_count", "nonqualified_count",
        "disposition_counts", "initial_boundary_tipped_count",
        "restoration_probe_tipped_count", "teacher_terminated_unsafely_count",
        "hard_stop_count", "all_material_shards_reopened_and_valid",
    }
)
V4_HISTORICAL_CUSTODY_METRIC_FIELDS = frozenset(
    {
        "external_receipt_binding", "external_receipt_projection_sha256",
        "v3_source_freeze_commit", "v3_completed_pool_count",
        "v3_failed_pool_indices", "historical_snapshot_or_probe_rerun",
        "historical_roots_unchanged", "pass",
    }
)
V4_SCIENTIFIC_INVARIANCE_METRIC_FIELDS = frozenset(
    {
        "v3_contract_content_digest", "v4_contract_content_digest",
        "scientific_projection_equal", "candidate_specs_equal",
        "all_regressions_passed", "pass",
    }
)
V4_PANEL_INADEQUATE_METRICS_FIELDS = frozenset(
    {
        "schema", "experiment_id", "development_only", "final_evaluation_eligible",
        "scientific_result_produced", "terminal_disposition",
        "primary_classification", "secondary_classifications", "next_experiment",
        "v4_state_dispositions", "v4_panel_adequacy", "v4_historical_custody",
        "v4_scientific_invariance", "downstream_scientific_metrics",
        "qualification_runtime_environment", "runtime_environments",
        "models_trained", "prohibited_components_trained_or_implemented",
        "content_digest",
    }
)
V4_INHERITED_DOWNSTREAM_METRIC_FIELDS = frozenset(
    {
        "evidence_counts", "panel", "development", "heldout",
        "repeatability", "command_tracking", "runtime_environments",
        "stratified", "classification_input", "gate", "component_failures",
        "active_components_in_precedence_order", "earliest_failing_component",
        "primary_classification", "secondary_classifications",
        "next_experiment", "predecessor_context", "claims",
        "safety_workstream", "prohibited_action_counters",
    }
)
V4_RESULT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "source_commit", "source_parent_commit",
        "source_baseline_commit", "development_only", "final_evaluation_eligible",
        "scientific_result_produced", "terminal_disposition", "primary_classification",
        "secondary_classifications", "next_experiment", "v3_terminal_diagnosis",
        "three_identity_contracts_retained", "state_dispositions", "panel_adequacy",
        "historical_custody", "scientific_invariance", "downstream_scientific_metrics",
        "runtime_environments", "qualification_runtime_environment",
        "metrics_sha256", "independent_reducer_receipt_sha256",
        "runtime_seconds", "scientific_storage_bytes", "models_trained",
        "prohibited_components_trained_or_implemented", "content_digest",
    }
)
V4_PUBLICATION_PROJECTION_FIELDS = frozenset(
    {"schema", "experiment_id", "result_document", "recomputed_metrics", "content_digest"}
)
V4_RESULT_REPORT_SECTION_ORDER = (
    "Disposition", "V1/V2/V3 custody and V3 terminal diagnosis",
    "Three snapshot identity contracts", "State dispositions", "Panel adequacy",
    "Inherited downstream science", "Runtime, storage, and training",
)
RESULT_PUBLICATION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.result_publication_authority.v1",
        "result_fields": sorted(V4_RESULT_FIELDS),
        "publication_projection_fields": sorted(V4_PUBLICATION_PROJECTION_FIELDS),
        "report_section_order": list(V4_RESULT_REPORT_SECTION_ORDER),
        "success_prepublication_binding_count": OUTPUT_LEAF_COUNT - 3,
        "panel_inadequate_prepublication_binding_count": PANEL_INADEQUATE_OUTPUT_LEAF_COUNT - 3,
        "scientific_input_binding_fields": ["path", "bytes", "sha256"],
        "contract_binding": "exact canonical LF-terminated runtime contract bytes",
        "metrics_binding": "exact canonical LF-terminated metrics bytes",
        "scientific_storage_bytes": "sum exact bytes over every prepublication official leaf",
        "report_bytes": "exact UTF-8 LF-terminated deterministic builder output",
        "success_downstream_scientific_metric_fields": sorted(
            V4_INHERITED_DOWNSTREAM_METRIC_FIELDS
        ),
        "panel_inadequate_downstream_scientific_metrics": None,
        "qualification_runtime_environment_fields": sorted(
            QUALIFICATION_RUNTIME_ENVIRONMENT_FIELDS
        ),
        "qualification_runtime_environment_required_for_both_terminals": True,
        "successful_qualification_runtime_cross_bound_to_assembled_physical_runtime": True,
        "panel_inadequate_is_scientific_result": True,
        "external_reducer_receipt_required_for_success_or_panel_inadequate": True,
        "development_only": True,
        "final_evaluation_eligible": False,
    }
)


# ---------------------------------------------------------------------------
# Historical custody and exact V3 boundary
# ---------------------------------------------------------------------------

HISTORICAL_CUSTODY_RECEIPT_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v1_v2_v3.custody_receipt.v1"
)
HISTORICAL_CUSTODY_RECEIPT_BINDING = {
    "path": str(HISTORICAL_CUSTODY_RECEIPT_PATH),
    "bytes": 95476,
    "sha256": (
        "0dfc1d39f8f9810d6a99cc69312a77935abf6f4bf71e10184ded3a49850c07cd"
    ),
}
HISTORICAL_CUSTODY_BINDING_FIELDS = frozenset({"path", "bytes", "sha256"})
HISTORICAL_CUSTODY_FILE_FIELDS = copy.deepcopy(V3.HISTORICAL_CUSTODY_FILE_FIELDS)
HISTORICAL_CUSTODY_DIRECTORY_FIELDS = copy.deepcopy(V3.HISTORICAL_CUSTODY_DIRECTORY_FIELDS)
HISTORICAL_CUSTODY_ROOT_FIELDS = copy.deepcopy(V3.HISTORICAL_CUSTODY_ROOT_FIELDS)
EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "generated_before_v4_simulator_creation", "repository",
        "prior_receipt_bindings", "roots", "v3_partial_boundary",
        "v3_failure_log_bindings", "v3_terminal_interpretation",
        "immutability", "nonreuse",
    }
)
HISTORICAL_CUSTODY_REPOSITORY_FIELDS = frozenset(
    {
        "v1_source_freeze_commit", "v2_source_freeze_commit",
        "v3_source_freeze_commit", "v1_freeze_subject", "v2_freeze_subject",
        "v3_freeze_subject", "v4_source_parent_commit", "current_head_commit",
        "repository_worktree_clean_at_receipt_emission",
        "historical_tracked_sources_match_frozen_commits", "v4_development_only",
        "v4_permanently_ineligible_for_final_evaluation",
        "sealed_paths_accessed", "tracked_ignore_bypass_used",
    }
)
HISTORICAL_PRIOR_RECEIPT_BINDING_FIELDS = frozenset(
    {
        "v1_custody_receipt_binding", "v1_v2_custody_receipt_binding",
        "v2_regeneration_receipt_present", "v3_regeneration_receipt_present",
    }
)
HISTORICAL_ROOT_KEYS = (
    "v1_official_root", "v1_material_root", "v2_official_root",
    "v2_material_root", "v3_official_root", "v3_material_root",
)
HISTORICAL_V3_PARTIAL_BOUNDARY_FIELDS = frozenset(
    {
        "official_leaf_count", "official_leaves", "first_eight_pass",
        "full_collection_authorized", "material_file_count",
        "material_directory_count",
        "completed_pool_count", "completed_pool_indices", "missing_pool_indices",
        "failed_pool_indices", "unattempted_pool_indices", "qualified_count",
        "rejected_count", "panel_constructed", "role_assignment_performed",
        "selected_snapshot_executions", "reset_fixture_executions",
        "candidate_fanout_executions",
        "encoder_initializations", "ranker_inference_calls",
        "development_or_heldout_outcomes_opened", "metrics_persisted",
        "result_persisted", "external_regeneration_receipt_present",
    }
)
HISTORICAL_V3_FAILURE_ROW_FIELDS = frozenset(
    {
        "pool_index", "exit_code", "exception_type", "exception_message",
        "qualification_shard_present", "scientific_outcome_opened",
        "temporary_log_binding_is_descriptive_only",
    }
)
HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS = frozenset(
    {
        "audit_mode", "v1_official_root_unchanged_during_audit",
        "v1_material_root_unchanged_during_audit",
        "v2_official_root_unchanged_during_audit",
        "v2_material_root_unchanged_during_audit",
        "v3_official_root_unchanged_during_audit",
        "v3_material_root_unchanged_during_audit",
        "all_files_regular_single_link", "receipt_outside_all_experiment_roots",
    }
)
HISTORICAL_CUSTODY_NONREUSE_FIELDS = frozenset(
    {
        "historical_roots_shared_inode_count", "historical_payload_copy_into_v4_count",
        "historical_hardlink_into_v4_count", "historical_runtime_artifact_or_shard_reused",
        "historical_snapshot_deserializations", "model_initializations", "training_runs",
        "simulator_initializations", "runner_calls", "encoder_calls", "ranker_calls",
        "sealed_path_accesses", "ignore_bypasses",
    }
)
V1_RECEIPT_BINDING = copy.deepcopy(V3.V1_CUSTODY_RECEIPT_BINDING)
V1_V2_RECEIPT_BINDING = copy.deepcopy(V3.HISTORICAL_CUSTODY_RECEIPT_BINDING)
HISTORICAL_ROOT_EXPECTATIONS = {
    "v1_official_root": {"path": str(V1_OFFICIAL_ROOT), "file_count": 1, "regular_file_apparent_bytes": 66418, "manifest_sha256": "e52b8823a92d1e14b9c5e3cb718224e9179d5a4af9faf9555ff53330288c565a"},
    "v1_material_root": {"path": str(V1_MATERIAL_ROOT), "file_count": 34, "regular_file_apparent_bytes": 4299695, "manifest_sha256": "f55cd6db93ee22cb59da93e4064262246feedcfda5e2b688cee4e98ef73c5247"},
    "v2_official_root": {"path": str(V2_OFFICIAL_ROOT), "file_count": 4, "regular_file_apparent_bytes": 102527, "manifest_sha256": "5c6bfbc48b62d1aff77cd5f57ca1dcbe8b1aac08d3b1dec5513480f2bab68c64"},
    "v2_material_root": {"path": str(V2_MATERIAL_ROOT), "file_count": 34, "regular_file_apparent_bytes": 4382459, "manifest_sha256": "7d7e0c32cf8d5fdb6993fc98b08a41054c59723dfeafb759cf2eb68b47ba3189"},
    "v3_official_root": {"path": str(V3_OFFICIAL_ROOT), "file_count": 4, "regular_file_apparent_bytes": 148298, "manifest_sha256": "29babfa702144fa373d6f5f7562710f86a19f746f5d20bba6453c3632108c3aa"},
    "v3_material_root": {"path": str(V3_MATERIAL_ROOT), "file_count": 269, "regular_file_apparent_bytes": 142341244, "manifest_sha256": "b58a5be3eed13557d060e369d7956a585f46fc2f6c17029189cf1ee2b5376340"},
}
V3_COMPLETED_POOL_INDICES = tuple((*range(128), *range(131, 136)))
V3_FAILED_POOL_INDICES = (128, 129, 130)
V3_UNATTEMPTED_POOL_INDICES = tuple(range(136, 256))
V3_MISSING_POOL_INDICES = V3_FAILED_POOL_INDICES + V3_UNATTEMPTED_POOL_INDICES
V3_PARTIAL_BOUNDARY_EXPECTATION = {
    "official_leaf_count": 4,
    "official_leaves": [
        "contract.json", "scientific_invariance_receipt.json",
        "v1_v2_custody_and_nonreuse.json",
        "v1_v2_v3_first_eight_reproduction.json",
    ],
    "first_eight_pass": True,
    "full_collection_authorized": True,
    "material_file_count": 269,
    "material_directory_count": 140,
    "completed_pool_count": 133,
    "completed_pool_indices": list(V3_COMPLETED_POOL_INDICES),
    "missing_pool_indices": list(V3_MISSING_POOL_INDICES),
    "failed_pool_indices": list(V3_FAILED_POOL_INDICES),
    "unattempted_pool_indices": list(V3_UNATTEMPTED_POOL_INDICES),
    "qualified_count": 100,
    "rejected_count": 33,
    "panel_constructed": False,
    "role_assignment_performed": False,
    "selected_snapshot_executions": 0,
    "reset_fixture_executions": 0,
    "candidate_fanout_executions": 0,
    "encoder_initializations": 0,
    "ranker_inference_calls": 0,
    "development_or_heldout_outcomes_opened": 0,
    "metrics_persisted": False,
    "result_persisted": False,
    "external_regeneration_receipt_present": False,
}
V3_FAILURE_EXPECTATION = [
    {
        "pool_index": index, "exit_code": 1, "exception_type": "BoundaryRefused",
        "exception_message": "termination flag tipped is True",
        "qualification_shard_present": False, "scientific_outcome_opened": False,
        "temporary_log_binding_is_descriptive_only": True,
    }
    for index in V3_FAILED_POOL_INDICES
]
EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.external_historical_custody_authority.v1",
        "receipt_path": str(HISTORICAL_CUSTODY_RECEIPT_PATH),
        "receipt_schema": HISTORICAL_CUSTODY_RECEIPT_SCHEMA,
        "ordinary_canonical_json_without_self_digest": True,
        "exact_binding_required_before_source_freeze": True,
        "fields": sorted(EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS),
        "root_keys": list(HISTORICAL_ROOT_KEYS),
        "root_fields": sorted(HISTORICAL_CUSTODY_ROOT_FIELDS),
        "file_fields": sorted(HISTORICAL_CUSTODY_FILE_FIELDS),
        "directory_fields": sorted(HISTORICAL_CUSTODY_DIRECTORY_FIELDS),
        "repository_fields": sorted(HISTORICAL_CUSTODY_REPOSITORY_FIELDS),
        "prior_receipt_binding_fields": sorted(HISTORICAL_PRIOR_RECEIPT_BINDING_FIELDS),
        "v3_partial_boundary_fields": sorted(HISTORICAL_V3_PARTIAL_BOUNDARY_FIELDS),
        "v3_failure_row_fields": sorted(HISTORICAL_V3_FAILURE_ROW_FIELDS),
        "v3_terminal_interpretation": copy.deepcopy(V3_TERMINAL_INTERPRETATION),
        "immutability_fields": sorted(HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS),
        "nonreuse_fields": sorted(HISTORICAL_CUSTODY_NONREUSE_FIELDS),
        "root_expectations": copy.deepcopy(HISTORICAL_ROOT_EXPECTATIONS),
        "v3_partial_boundary_expectation": copy.deepcopy(V3_PARTIAL_BOUNDARY_EXPECTATION),
        "v3_failure_expectation": copy.deepcopy(V3_FAILURE_EXPECTATION),
        "zero_counter_scope": (
            "EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_BUILD_AND_EMISSION_PROCESS_ONLY"
        ),
        "development_source_audit_disclosure": copy.deepcopy(
            DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
        ),
        "historical_snapshot_or_probe_reexecution_forbidden": True,
        "historical_roots_may_not_be_scientific_inputs_to_v4": True,
    }
)

V1_V2_V3_CUSTODY_AND_NONREUSE_FIELDS = frozenset(
    {
        "schema", "experiment_id", "v4_source_freeze_commit",
        "external_historical_custody_receipt_binding",
        "external_historical_custody_projection_sha256",
        "v1_v2_v3_roots_unchanged", "historical_payloads_copied_into_v4",
        "historical_hardlinks_into_v4", "historical_shared_inodes_with_v4",
        "historical_runtime_artifact_or_shard_reused", "historical_snapshot_or_probe_rerun",
        "sealed_path_accessed", "ignore_bypassed", "allowed_read_scope",
        "v3_terminal_interpretation", "pass",
    }
)
V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.v1_v2_v3_custody_and_nonreuse_authority.v1",
        "fields": sorted(V1_V2_V3_CUSTODY_AND_NONREUSE_FIELDS),
        "external_authority_content_digest": EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY["content_digest"],
        "allowed_read_scope": "CUSTODY_BINDING_ONLY_NO_HISTORICAL_SNAPSHOT_OR_PROBE_REEXECUTION",
        "v3_terminal_interpretation": copy.deepcopy(
            V3_TERMINAL_INTERPRETATION
        ),
        "zero_counter_scope": "V4_SCIENTIFIC_EXECUTION_PROCESS_ONLY",
        "development_source_audit_disclosure": copy.deepcopy(
            DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
        ),
        "development_only": True,
        "final_evaluation_eligible": False,
        "sealed_access_forbidden": True,
        "ignore_bypass_forbidden": True,
    }
)


# ---------------------------------------------------------------------------
# Prospective regression and invariance authorities
# ---------------------------------------------------------------------------

STATE_DISPOSITION_REGRESSION_IDS = (
    "VALID_INITIAL_STATE_CLASSIFIED",
    "INITIAL_TIPPED_CLASSIFIED",
    "RESTORATION_PROBE_TIPPED_CLASSIFIED",
    "TEACHER_PHYSICS_CONTACT_CLASSIFIED",
    "TEACHER_CROSSING_INVALID_CLASSIFIED",
    "TEACHER_DID_NOT_LEAVE_SOURCE_CLASSIFIED",
    "TEACHER_NO_POSITIVE_PROGRESS_CLASSIFIED",
    "FULLY_QUALIFIED_TEACHER_CLASSIFIED",
    "MATERIALISATION_CORRUPTION_HARD_STOP",
    "DETERMINISTIC_REJECTION_RECORD_IDENTICAL",
)
FAMILY_FIXTURE_REGRESSION_IDS = tuple(
    f"{family}_NORMAL_STATE_CLASSIFICATION_PATH" for family in V3.FAMILY_IDS
)
ALL_V4_REGRESSION_IDS = STATE_DISPOSITION_REGRESSION_IDS + FAMILY_FIXTURE_REGRESSION_IDS
REGRESSION_RESULT_FIELDS = frozenset({"requirement_id", "passed", "evidence"})
SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "v3_contract_content_digest",
        "v4_contract_content_digest", "v3_scientific_projection_sha256",
        "v4_scientific_projection_sha256", "scientific_projection_equal",
        "candidate_specs_equal", "source_dependency_paths_equal",
        "v3_source_freeze_commit", "state_disposition_authority_content_digest",
        "panel_adequacy_authority_content_digest",
        "persisted_array_hash_authority_retained",
        "snapshot_semantic_serializer_authority_retained",
        "snapshot_behavioural_probe_authority_retained",
        "authorized_change_scope", "terminal_evidence_implementation_ramifications",
        "pre_panel_engineering_correction",
        "v3_terminal_interpretation",
        "metric_formula_gate_threshold_model_tuning_changed",
        "historical_snapshot_or_probe_rerun", "development_only",
        "final_evaluation_eligible", "regression_results",
        "all_regressions_passed", "pass",
    }
)
V3_SCIENTIFIC_CONTRACT = V3.build_contract()
V3_CONTRACT_CONTENT_DIGEST = V3_SCIENTIFIC_CONTRACT["content_digest"]
V3_SCIENTIFIC_PROJECTION = V3.scientific_invariance_projection(V3_SCIENTIFIC_CONTRACT)
V3_SCIENTIFIC_PROJECTION_SHA256 = _canonical_no_lf_sha256(V3_SCIENTIFIC_PROJECTION)
V3_CANDIDATE_SPECS = V3.build_candidate_specs()
V3_CANDIDATE_SPECS_SHA256 = _canonical_no_lf_sha256(V3_CANDIDATE_SPECS)


def build_candidate_specs() -> list[dict[str, Any]]:
    return V3.build_candidate_specs()


def build_prospective_pool_specs() -> list[dict[str, Any]]:
    return V3.build_prospective_pool_specs()


def scientific_invariance_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return V3.scientific_invariance_projection(value)


def scientific_constant_projection() -> dict[str, Any]:
    return V3.scientific_constant_projection()


SCIENTIFIC_INVARIANCE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v4.scientific_invariance_authority.v1",
        "v3_experiment_id": V3_EXPERIMENT_ID,
        "v4_experiment_id": EXPERIMENT_ID,
        "v3_source_freeze_commit": V3_SOURCE_FREEZE_COMMIT,
        "v3_contract_content_digest": V3_CONTRACT_CONTENT_DIGEST,
        "v3_scientific_projection_sha256": V3_SCIENTIFIC_PROJECTION_SHA256,
        "v4_scientific_projection_sha256": V3_SCIENTIFIC_PROJECTION_SHA256,
        "v3_candidate_specs_sha256": V3_CANDIDATE_SPECS_SHA256,
        "v4_candidate_specs_sha256": V3_CANDIDATE_SPECS_SHA256,
        "source_dependency_paths_equal": True,
        "logical_ids_seeds_order_and_geometry_unchanged": True,
        "controller_model_ranker_candidate_and_physical_authorities_unchanged": True,
        "metric_formulas_gates_thresholds_classes_precedence_and_next_decisions_unchanged_after_adequate_panel": True,
        "raw_previous_command_hash_authority_retained": True,
        "snapshot_semantic_serializer_authority_content_digest": V3.SEMANTIC_SERIALIZER_AUTHORITY["content_digest"],
        "snapshot_behavioural_probe_authority_content_digest": V3.BEHAVIOURAL_PROBE_AUTHORITY["content_digest"],
        "authorized_change_scope": SOLE_SCIENTIFIC_PROCEDURE_CHANGE,
        "terminal_evidence_implementation_ramifications": (
            "persist exact defined initial, restoration-probe, and teacher terminal "
            "evidence, including the V4-only inclusive-final-sample IEEE rule when "
            "the simulator reports nan=True; no physical formula, criterion, gate, "
            "threshold, model, tuning, or role-assignment rule changes"
        ),
        "pre_panel_engineering_correction": {
            "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
            "authority_content_digest": (
                PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY["content_digest"]
            ),
            "existing_partial_material_reuse_authorized": False,
            "restart_from_fresh_v4_roots_required": True,
            "scientific_formula_or_decision_changed": False,
        },
        "v3_terminal_interpretation": copy.deepcopy(V3_TERMINAL_INTERPRETATION),
        "historical_snapshot_or_probe_rerun": False,
        "development_only": True,
        "final_evaluation_eligible": False,
    }
)


# ---------------------------------------------------------------------------
# Source closure and contract/runtime documents
# ---------------------------------------------------------------------------

DOC_PREFIX = "docs/lewm_go2_physical_graph_edge_handoff_qualification_v4"
TRACKED_SOURCE_PATHS = (
    f"{DOC_PREFIX}_contract_2026-09-03.json",
    f"{DOC_PREFIX}_fixture_2026-09-03.json",
    f"{DOC_PREFIX}_output_schema_2026-09-03.json",
    f"{DOC_PREFIX}_preregistration_2026-09-03.md",
    f"{DOC_PREFIX}_source_closure_2026-09-03.json",
    f"{DOC_PREFIX}_scientific_invariance_2026-09-03.json",
    f"{DOC_PREFIX}_v1_v2_v3_custody_binding_2026-09-03.json",
    "lewm/safety/physical_graph_edge_handoff_qualification_v4_contract.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v4_metrics.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v4_contract.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v4_metrics.py",
    "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v4.py",
    "lewm/tests/test_evaluate_physical_graph_edge_handoff_qualification_v4.py",
    "scripts/run_physical_graph_edge_handoff_qualification_v4.py",
    "scripts/evaluate_physical_graph_edge_handoff_qualification_v4.py",
)
SOURCE_DEPENDENCY_PATHS = tuple(V3.SOURCE_DEPENDENCY_PATHS)
V4_WRAPPER_DEPENDENCY_PATHS = tuple(V3.SOURCE_CLOSURE_PATHS)
SOURCE_CLOSURE_PATHS = tuple(dict.fromkeys(TRACKED_SOURCE_PATHS[7:] + V4_WRAPPER_DEPENDENCY_PATHS))

V4_RUNTIME_POLICY = {
    "development_only": True,
    "permanently_ineligible_for_final_evaluation": True,
    "sealed_path_accesses": 0,
    "ignore_bypasses": 0,
    "zero_counter_scope": "V4_SCIENTIFIC_EXECUTION_PROCESS_ONLY",
    "development_source_audit_disclosure": copy.deepcopy(
        DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
    ),
    "before_simulator_creation": [
        "validate V3-to-V4 scientific invariance",
        "validate exact external V1/V2/V3 custody receipt and unchanged historical roots",
        "prove V4 official/material roots and external reducer receipt are absent",
        "pass all ten state-disposition and four non-scientific family fixtures",
    ],
    "qualification": (
        "write exactly one atomic metadata.json plus payload.npz per pool; defined "
        "nonqualification continues; hard-stop dispositions and global hard stops stop"
    ),
    "after_256_records": (
        "persist canonical qualification_state_dispositions.jsonl and panel_adequacy.json; "
        "if inadequate publish the exact nine-leaf terminal and no downstream outcomes"
    ),
    "adequate_panel": "resume exact inherited V3/V2/V1 physical-science stages",
    "historical_snapshot_or_probe_rerun": False,
    "historical_runtime_artifact_or_shard_reuse": False,
    "models_trained": 0,
    "pre_panel_engineering_correction": {
        "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "authority_content_digest": (
            PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY["content_digest"]
        ),
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
    },
}


def build_contract() -> dict[str, Any]:
    base = copy.deepcopy(V3.build_contract())
    base.pop("content_digest", None)
    # These V3 keys governed historical deserialization, the first-eight
    # reproduction gate, and V3's 28-leaf execution.  V4 binds that completed
    # or terminal history only through the external custody receipt and never
    # re-runs or adopts those runtime gates.
    for obsolete_v3_runtime_key in (
        "first_eight_reproduction_authority",
        "qualification_shard_augmentation_authority",
        "historical_snapshot_deserializer_authority",
        "v1_v2_custody_and_nonreuse_authority",
        "v3_runtime_policy",
        "v3_wrapper_dependency_paths",
    ):
        base.pop(obsolete_v3_runtime_key, None)
    base.update(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "development_only": True,
            "final_evaluation_eligible": False,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "commit_subjects": {"freeze": CONTRACT_FREEZE_COMMIT_SUBJECT, "result": RESULT_COMMIT_SUBJECT},
            "output": {
                "root": str(OUTPUT_ROOT),
                "material_root": str(MATERIAL_ROOT),
                "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "successful_leaf_count": OUTPUT_LEAF_COUNT,
                "successful_complete_inventory": list(SUCCESS_OUTPUT_LEAVES),
                "panel_inadequate_leaf_count": PANEL_INADEQUATE_OUTPUT_LEAF_COUNT,
                "panel_inadequate_inventory": list(PANEL_INADEQUATE_OUTPUT_LEAVES),
                "technical_hard_stop_inventory": list(TECHNICAL_HARD_STOP_OUTPUT_LEAVES),
                "receipt_self_digests": False,
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "source_closure_paths": list(SOURCE_CLOSURE_PATHS),
            "state_disposition_authority": copy.deepcopy(STATE_DISPOSITION_AUTHORITY),
            "pre_panel_engineering_correction_authority": copy.deepcopy(
                PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
            ),
            "v4_terminal_nonfinite_authority": copy.deepcopy(
                V4_TERMINAL_NONFINITE_AUTHORITY
            ),
            "panel_adequacy_authority": copy.deepcopy(PANEL_ADEQUACY_AUTHORITY),
            "v4_panel_teacher_subset_authority": copy.deepcopy(
                V4_PANEL_TEACHER_SUBSET_AUTHORITY
            ),
            "qualification_runtime_authority": copy.deepcopy(
                QUALIFICATION_RUNTIME_AUTHORITY
            ),
            "result_publication_authority": copy.deepcopy(RESULT_PUBLICATION_AUTHORITY),
            "material_inventory_authority": copy.deepcopy(
                MATERIAL_INVENTORY_AUTHORITY
            ),
            "external_historical_custody_authority": copy.deepcopy(EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY),
            "v1_v2_v3_custody_and_nonreuse_authority": copy.deepcopy(V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY),
            "scientific_invariance_authority": copy.deepcopy(SCIENTIFIC_INVARIANCE_AUTHORITY),
            "persisted_array_hash_authority": copy.deepcopy(PERSISTED_ARRAY_HASH_AUTHORITY),
            "snapshot_semantic_serializer_authority": copy.deepcopy(V3.SEMANTIC_SERIALIZER_AUTHORITY),
            "snapshot_behavioural_probe_authority": copy.deepcopy(V3.BEHAVIOURAL_PROBE_AUTHORITY),
            "v4_runtime_policy": copy.deepcopy(V4_RUNTIME_POLICY),
            "development_source_audit_disclosure": copy.deepcopy(
                DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
            ),
            "v4_wrapper_dependency_paths": list(V4_WRAPPER_DEPENDENCY_PATHS),
        }
    )
    result = attach_content_digest(base)
    if scientific_invariance_projection(result) != V3_SCIENTIFIC_PROJECTION:
        raise PhysicalGraphEdgeHandoffV4ContractError("V4 scientific projection differs from V3")
    if build_candidate_specs() != V3_CANDIDATE_SPECS:
        raise PhysicalGraphEdgeHandoffV4ContractError("V4 candidate specs differ from V3")
    if len(SUCCESS_OUTPUT_LEAVES) != 27 or len(set(SUCCESS_OUTPUT_LEAVES)) != 27:
        raise PhysicalGraphEdgeHandoffV4ContractError("V4 success inventory drift")
    if len(PANEL_INADEQUATE_OUTPUT_LEAVES) != 9 or len(set(PANEL_INADEQUATE_OUTPUT_LEAVES)) != 9:
        raise PhysicalGraphEdgeHandoffV4ContractError("V4 inadequate inventory drift")
    return result


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_content_digest(value)
    if canonical_json_bytes(row) != canonical_json_bytes(build_contract()):
        raise PhysicalGraphEdgeHandoffV4ContractError("contract value drift")
    return copy.deepcopy(row)


RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema", "experiment_id", "status", "source_freeze_commit",
        "source_parent_commit", "source_baseline_commit", "v1_source_freeze_commit",
        "v2_source_freeze_commit", "v3_source_freeze_commit", "scientific_contract",
        "predecessor_result_binding", "external_artifact_bindings", "runtime_policy",
        "historical_custody_receipt_binding", "v4_runtime_policy", "content_digest",
    }
)


def _commit(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise PhysicalGraphEdgeHandoffV4ContractError(f"invalid {label}")
    return value


def _historical_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != HISTORICAL_CUSTODY_BINDING_FIELDS:
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody binding field drift")
    row = copy.deepcopy(dict(value))
    if row["path"] != str(HISTORICAL_CUSTODY_RECEIPT_PATH):
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody path drift")
    if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody byte count drift")
    digest = row["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody SHA-256 drift")
    if HISTORICAL_CUSTODY_RECEIPT_BINDING is not None and row != HISTORICAL_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody exact binding drift")
    return row


def build_runtime_contract(
    source_freeze_commit: str,
    historical_custody_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    source_freeze_commit = _commit(source_freeze_commit, "source_freeze_commit")
    if source_freeze_commit == INVALIDATED_V4_SOURCE_FREEZE_COMMIT:
        raise PhysicalGraphEdgeHandoffV4ContractError(
            "invalidated V4 source freeze cannot initialize qualification"
        )
    binding = _historical_binding(historical_custody_receipt_binding)
    return attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.runtime_contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "source_freeze_commit": source_freeze_commit,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "v1_source_freeze_commit": V1_SOURCE_FREEZE_COMMIT,
            "v2_source_freeze_commit": V2_SOURCE_FREEZE_COMMIT,
            "v3_source_freeze_commit": V3_SOURCE_FREEZE_COMMIT,
            "scientific_contract": build_contract(),
            "predecessor_result_binding": copy.deepcopy(V3.V2.V1.PREDECESSOR_RESULT_BINDING),
            "external_artifact_bindings": copy.deepcopy(V3.EXTERNAL_ARTIFACT_BINDINGS),
            "runtime_policy": copy.deepcopy(V3.V2.V1.DIRECT_RUNTIME_POLICY),
            "historical_custody_receipt_binding": binding,
            "v4_runtime_policy": copy.deepcopy(V4_RUNTIME_POLICY),
        }
    )


def validate_runtime_contract(
    value: Mapping[str, Any], *, source_freeze_commit: str | None = None,
    historical_custody_receipt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != RUNTIME_CONTRACT_FIELDS:
        raise PhysicalGraphEdgeHandoffV4ContractError("runtime contract field drift")
    observed = _commit(row["source_freeze_commit"], "source_freeze_commit")
    if source_freeze_commit is not None and observed != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffV4ContractError("source freeze commit drift")
    binding = _historical_binding(row["historical_custody_receipt_binding"])
    if historical_custody_receipt_binding is not None and binding != _historical_binding(historical_custody_receipt_binding):
        raise PhysicalGraphEdgeHandoffV4ContractError("historical custody binding cross-link drift")
    expected = build_runtime_contract(observed, binding)
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4ContractError("runtime contract value drift")
    return copy.deepcopy(row)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "PhysicalGraphEdgeHandoffV4ContractError",
    "attach_content_digest",
    "build_candidate_specs",
    "build_contract",
    "build_prospective_pool_specs",
    "build_runtime_contract",
    "canonical_json_bytes",
    "scientific_constant_projection",
    "scientific_invariance_projection",
    "teacher_trace_npz_authority",
    "validate_content_digest",
    "validate_contract",
    "validate_runtime_contract",
]
