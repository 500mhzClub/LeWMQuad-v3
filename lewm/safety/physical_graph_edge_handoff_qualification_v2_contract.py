"""Pure prospective authority for the corrected physical handoff qualification.

V2 is deliberately not a new scientific experiment.  It keeps every V1
scientific constant and formula and reruns the same role-free physical states.
Its V2-specific changes are experiment/source identity, output custody, the
persisted array-byte hash contract that failed during the first V1 teacher
shards, and two explicitly authorized implementation alignments:
``INHERITED_PORT_HEADING_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY`` changes
only ``directed_port_world[2]``; and
``INHERITED_CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY``
changes only ``port_progress_m``, ``lateral_error_m``, and
``positive_port_progress``.  Both make the inherited writer satisfy already-
frozen V1 authorities.  Neither changes a physical crossing, formula, gate,
threshold, class, next decision, model, or tuning parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
from pathlib import Path
import types
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as V1


class PhysicalGraphEdgeHandoffV2ContractError(ValueError):
    """Raised when the corrected prospective authority drifts."""


# Re-export the complete frozen V1 scientific vocabulary.  Identity and
# custody constants are overwritten below; every other uppercase constant is
# intentionally the same Python value as V1.
for _name in tuple(V1.__all__):
    if _name.isupper():
        _value = getattr(V1, _name)
        try:
            _value = copy.deepcopy(_value)
        except (TypeError, ValueError):
            # V1 exports its imported V2 contract module under the uppercase
            # name ``V2``.  Modules are immutable authorities here and are not
            # pickle/deepcopy values.
            pass
        globals()[_name] = _value


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
V1_EXPERIMENT_ID = V1.EXPERIMENT_ID
V1_SOURCE_FREEZE_COMMIT = "3dfff6caec1c3162d8123c737d04bcdd42799653"
SOURCE_PARENT_COMMIT = V1_SOURCE_FREEZE_COMMIT
SOURCE_BASELINE_COMMIT = V1.SOURCE_BASELINE_COMMIT
CONTRACT_FREEZE_COMMIT_SUBJECT = (
    "Freeze corrected physical graph edge handoff qualification V2"
)
RESULT_COMMIT_SUBJECT = (
    "Evaluate corrected physical graph edge handoff qualification V2"
)

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2"
)
MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v2_material"
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
)
V1_CUSTODY_RECEIPT_PATH = OUTPUT_ROOT.parent / (
    "physical_graph_edge_handoff_qualification_v1_custody_receipt.json"
)
V1_OFFICIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1"
V1_MATERIAL_ROOT = OUTPUT_ROOT.parent / "physical_graph_edge_handoff_qualification_v1_material"

NEW_RUNTIME_OUTPUT_PATHS = {
    "v1_custody_and_nonreuse": "v1_custody_and_nonreuse.json",
    "scientific_invariance_receipt": "scientific_invariance_receipt.json",
    "v1_v2_first_eight_reproduction": "v1_v2_first_eight_reproduction.json",
}
RUNTIME_OUTPUT_PATHS = {
    **copy.deepcopy(V1.RUNTIME_OUTPUT_PATHS),
    **NEW_RUNTIME_OUTPUT_PATHS,
}
OUTPUT_LEAF_COUNT = 26
SUCCESS_OUTPUT_LEAVES = tuple(RUNTIME_OUTPUT_PATHS.values())
REPRODUCTION_MISMATCH_LEAVES = (
    "contract.json",
    "v1_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "v1_v2_first_eight_reproduction.json",
)
REPRODUCTION_MISMATCH_DISPOSITION = "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"


REGRESSION_REQUIREMENT_IDS = (
    "FLOAT64_EXACT_PERSISTED_BYTES_VALID",
    "FLOAT32_CAST_DIGEST_REJECTED",
    "PERSISTED_DTYPE_MISMATCH_REJECTED",
    "PERSISTED_SHAPE_MISMATCH_REJECTED",
    "PERSISTED_VALUE_MUTATION_REJECTED",
    "NONCONTIGUOUS_VIEW_C_CONTIGUOUS_NO_DTYPE_CAST",
    "SAVE_RELOAD_DIGEST_IDENTICAL",
    "OTHER_METADATA_BINDINGS_UNCHANGED",
    "V1_SCIENTIFIC_CONSTANTS_AND_SOURCE_PATHS_UNCHANGED",
    "FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS",
)
REGRESSION_FIXTURE = {
    "dtype_str": "<f8",
    "shape": [2, 3],
    "values": [[0.125, -0.25, 0.375], [0.5, -0.625, 0.75]],
    "noncontiguous_base_shape": [3, 4],
    "noncontiguous_slice": "[:,::2]",
    "pre_simulator_required": True,
}

NPZ_ARCHIVE_COMMENT = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2:FRESH"
PERSISTED_ARRAY_HASH_AUTHORITY = {
    "schema": "physical_graph_edge_handoff_qualification_v2.persisted_array_hash_authority.v1",
    "digest_algorithm": "sha256",
    "digest_field_scope": [
        (
            "persisted_array_evidence.arrays[].array_bytes_sha256 on every V2 "
            "material shard payload written through the shared corrected writer"
        ),
        "snapshot.previous_applied_command_sha256 in every V2 material snapshot",
        (
            "state_snapshot_index.records[].previous_applied_command_sha256 in the "
            "assembled V2 official evidence"
        ),
    ],
    "digest_domain": "exact C-contiguous array.tobytes(order='C') only",
    "dtype_binding": "persist exact numpy dtype.str separately; no implicit or hash-time cast",
    "shape_binding": "persist exact integer shape separately",
    "normalization": (
        "the exact array passed to numpy.savez is made C-contiguous without changing dtype; "
        "metadata and post-save validation use that same array"
    ),
    "save_reopen_rule": (
        "immediately reopen the written NPZ, require member inventory, dtype.str, shape, "
        "C-order logical bytes, and raw-byte SHA-256 to equal the pre-save manifest"
    ),
    "npz_archive_comment_utf8": NPZ_ARCHIVE_COMMENT,
    "npz_archive_comment_required": True,
    "npz_archive_comment_scope": (
        "every V2 material shard payload.npz governed by persisted_array_evidence"
    ),
    "npz_archive_comment_semantics": (
        "V2-only deterministic container provenance; it changes no NPZ member, dtype, "
        "shape, logical value, candidate identity, or scientific outcome"
    ),
    "forbidden_domains": [
        "V1 canonical dtype/shape header plus NUL plus bytes in array_bytes_sha256",
        "digest of a float32 source followed by float64 persistence",
        "digest after any dtype-changing coercion",
    ],
    "assembled_official_npz_policy": (
        "the five inherited official evidence NPZ schemas and indices remain exact V1 "
        "science and are freshly assembled only from corrected, reopened material arrays; "
        "only the identified previous_applied_command_sha256 field uses the corrected "
        "raw-byte domain, while every other legacy index hash keeps its frozen V1 domain"
    ),
    "identified_defect_field": {
        "material_path": "snapshot.previous_applied_command_sha256",
        "official_path": (
            "state_snapshot_index.records[].previous_applied_command_sha256"
        ),
        "member": "snapshot__previous_applied_command",
        "digest_domain": "exact C-contiguous persisted bytes only",
        "dtype_str": "<f8",
        "per_snapshot_shape": [3],
        "first_eight_expected_raw_bytes_sha256": (
            "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
        ),
        "v1_invalid_float32_metadata_sha256": (
            "cd9cdaf845ca5fd82a869bd1f17362d780cfd1bd3a46fe76cd93575fb5ece5c6"
        ),
        "v1_persisted_float64_canonical_header_sha256": (
            "b237b7cbc24dd1d0bc36a02fa9a0d58e12f511b0fc76d2bbd0aaa48cab0265e3"
        ),
        "dtype_and_shape_bound_separately_by": [
            "persisted_array_evidence arrays row in each material shard",
            "state_snapshots.npz authority and independent official-row inspection",
        ],
        "other_snapshot_metadata_hash_fields_keep_frozen_v1_domains": True,
    },
    "tolerance": 0,
}
PERSISTED_ARRAY_ROW_FIELDS = frozenset(
    {"member", "dtype_str", "shape", "c_contiguous", "array_bytes_sha256"}
)
PERSISTED_ARRAY_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "shard_kind",
        "shard_id",
        "payload_file",
        "array_count",
        "array_inventory_sha256",
        "arrays",
        "save_reopen_validation_passed",
    }
)
PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS = frozenset({"path", "bytes", "sha256"})


V1_SCIENTIFIC_CONTRACT = V1.build_contract()
V1_CONTRACT_CONTENT_DIGEST = V1_SCIENTIFIC_CONTRACT["content_digest"]
# These are the only exported V1 constants whose values may legitimately
# differ in V2.  Everything else -- including all geometry, controller,
# selection, metric, gate, classification, safety and source-dependency
# constants -- is projected and byte-digested below.
V1_SCIENTIFIC_CONSTANT_EXCLUSIONS = {
    "V2": "imported module, not a JSON scientific value",
    "EXPERIMENT_ID": "V2 experiment identity",
    "SOURCE_PARENT_COMMIT": "V2 source lineage",
    "CONTRACT_FREEZE_COMMIT_SUBJECT": "V2 commit identity",
    "RESULT_COMMIT_SUBJECT": "V2 commit identity",
    "OUTPUT_ROOT": "fresh V2 custody root",
    "RUNTIME_OUTPUT_PATHS": "three additional V2 receipt leaves",
    "OUTPUT_LEAF_COUNT": "three additional V2 receipt leaves",
    "RUNTIME_CONTRACT_FIELDS": "V2 custody and correction fields",
    "TRACKED_SOURCE_PATHS": "V2 changed-path allow-list",
    "SOURCE_CLOSURE_PATHS": "V2 wrapper source closure",
}


def _json_native_constant(value: Any) -> Any:
    """Normalize a frozen constant without changing its semantic value."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_native_constant(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_native_constant(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_native_constant(item) for item in value]
        return sorted(normalized, key=lambda item: V1.canonical_json_bytes(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, types.ModuleType):
        raise PhysicalGraphEdgeHandoffV2ContractError(
            "module is not a scientific constant value"
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return copy.deepcopy(value)
    raise PhysicalGraphEdgeHandoffV2ContractError(
        f"unsupported scientific constant type: {type(value).__name__}"
    )


V1_SCIENTIFIC_CONSTANT_NAMES = tuple(
    name
    for name in V1.__all__
    if name.isupper() and name not in V1_SCIENTIFIC_CONSTANT_EXCLUSIONS
)


def scientific_constant_projection(
    namespace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project every non-identity V1 exported constant in canonical form."""

    source: Mapping[str, Any] = globals() if namespace is None else namespace
    missing = set(V1_SCIENTIFIC_CONSTANT_NAMES) - set(source)
    if missing:
        raise PhysicalGraphEdgeHandoffV2ContractError(
            f"scientific constants missing: {sorted(missing)}"
        )
    return {
        name: _json_native_constant(source[name])
        for name in V1_SCIENTIFIC_CONSTANT_NAMES
    }


V1_SCIENTIFIC_CONSTANT_PROJECTION = scientific_constant_projection(vars(V1))
V2_SCIENTIFIC_CONSTANT_PROJECTION = scientific_constant_projection()
V1_SCIENTIFIC_CONSTANTS_SHA256 = hashlib.sha256(
    V1.canonical_json_bytes(V1_SCIENTIFIC_CONSTANT_PROJECTION)[:-1]
).hexdigest()
V2_SCIENTIFIC_CONSTANTS_SHA256 = hashlib.sha256(
    V1.canonical_json_bytes(V2_SCIENTIFIC_CONSTANT_PROJECTION)[:-1]
).hexdigest()
ALLOWED_V1_V2_CONTRACT_DIFFERENCE_KEYS = (
    "schema",
    "experiment_id",
    "source_parent_commit",
    "commit_subjects",
    "output",
    "tracked_source_paths",
    "content_digest",
)
V2_ONLY_CONTRACT_KEYS = (
    "persisted_array_hash_authority",
    "regression_gate_authority",
    "scientific_invariance_authority",
    "v1_compatibility_identity_authority",
    "port_heading_implementation_alignment_authority",
    "candidate_port_metric_implementation_alignment_authority",
    "v1_custody_and_nonreuse_authority",
    "first_eight_reproduction_authority",
    "v2_correction_runtime_policy",
    "v2_wrapper_dependency_paths",
)
SCIENTIFIC_INVARIANT_KEYS = tuple(
    key
    for key in V1_SCIENTIFIC_CONTRACT
    if key not in ALLOWED_V1_V2_CONTRACT_DIFFERENCE_KEYS
)


def canonical_json_bytes(value: Any) -> bytes:
    """Use the frozen V1 canonical JSON domain."""

    return V1.canonical_json_bytes(value)


def _canonical_no_lf_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)[:-1]).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return V1.attach_content_digest(value)


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return V1.validate_content_digest(value)
    except Exception as exc:  # pragma: no cover - exact error translated for API clarity
        raise PhysicalGraphEdgeHandoffV2ContractError(str(exc)) from exc


def scientific_invariance_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return every field that V2 is forbidden to change from V1."""

    missing = set(SCIENTIFIC_INVARIANT_KEYS) - set(value)
    if missing:
        raise PhysicalGraphEdgeHandoffV2ContractError(
            f"scientific invariance fields missing: {sorted(missing)}"
        )
    return {
        key: copy.deepcopy(value[key])
        for key in SCIENTIFIC_INVARIANT_KEYS
    }


V1_SCIENTIFIC_PROJECTION = scientific_invariance_projection(V1_SCIENTIFIC_CONTRACT)
V1_SCIENTIFIC_PROJECTION_SHA256 = _canonical_no_lf_sha256(V1_SCIENTIFIC_PROJECTION)
V1_CANDIDATE_SPECS_SHA256 = _canonical_no_lf_sha256(V1.build_candidate_specs())
V1_SOURCE_DEPENDENCY_PATHS_SHA256 = _canonical_no_lf_sha256(
    list(V1.SOURCE_DEPENDENCY_PATHS)
)

V1_COMPATIBILITY_IDENTITY_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.v1_compatibility_identity_authority.v1",
        "official_experiment_id": EXPERIMENT_ID,
        "official_schema_token": "physical_graph_edge_handoff_qualification_v2",
        "frozen_inherited_scientific_identity_salt": V1_EXPERIMENT_ID,
        "v1_identity_scopes": [
            "in-memory inherited V1 scientific builders before the storage adapter",
            "split-assignment key projection, preserving the exact V1 development/heldout roles",
            "opaque serialized BranchSnapshot identity bytes, preserving exact first-eight reproduction",
        ],
        "official_v2_identity_scopes": [
            "every persisted JSON and JSONL experiment_id",
            "every persisted V2 schema",
            "runtime contract, material metadata, official evidence, metrics, result and report",
            "V2 runner CLI, output roots, source freeze and result commit",
        ],
        "logical_scene_state_spec_episode_ids_and_seeds": "unchanged pgehq-v1-* authority",
        "opaque_snapshot_is_freshly_generated_by_v2": True,
        "opaque_snapshot_v1_identity_affects_physics_target_selection_or_metric_formulas": False,
        "v1_runtime_artifact_or_snapshot_reused": False,
        "v1_scientific_completion_claimed": False,
        "v2_execution_and_result_identity_claimed": True,
        "storage_adapter_may_change_only_schema_experiment_and_content_digest_identity": True,
    }
)

PORT_HEADING_ALIGNMENT_DISPOSITION = (
    "INHERITED_PORT_HEADING_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
)
PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.port_heading_implementation_alignment_authority.v1",
        "disposition": PORT_HEADING_ALIGNMENT_DISPOSITION,
        "inherited_defect": (
            "V1 _edge_port_record copied interpolated robot yaw into directed_port_world[2] "
            "although the prospectively frozen V1 validator requires the selected opening-normal heading"
        ),
        "wrapper_hook": (
            "scripts.run_physical_graph_edge_handoff_qualification_v2."
            "_edge_port_record_authority_alignment"
        ),
        "exact_formula": (
            "directed_port_world[2] = atan2(opening_normal_world[1], "
            "opening_normal_world[0])"
        ),
        "changed_output_fields": ["directed_port_world[2]"],
        "required_exact_preservations": [
            "directed_port_world[0:2]",
            "teacher_crossing_velocity_world_xy",
            "teacher_crossing_velocity_heading_world_rad",
            "crossing_sample_before",
            "crossing_sample_after",
            "crossing_fraction",
            "route_lookahead_world",
            "route_lookahead_clipped",
            "remaining_route_length_m",
            "every other edge-port record field",
        ],
        "uses_outcome_values_to_choose_formula_or_parameter": False,
        "continuous_tuning_or_new_target_definition": False,
        "changes_frozen_metric_gate_class_precedence_or_next_decision": False,
        "changes_frozen_scientific_design": False,
        "aligns_implementation_to_frozen_v1_authority": True,
    }
)

CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION = (
    "INHERITED_CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNED_TO_FROZEN_AUTHORITY"
)
CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.candidate_port_metric_implementation_alignment_authority.v1",
        "disposition": CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION,
        "inherited_defect": (
            "V1 derive_candidate_outcome measured candidate port progress and lateral "
            "error from the prospective opening midpoint although the prospectively "
            "frozen V1 PHYSICAL_OUTCOME_AUTHORITY defines the reference as the canonical "
            "directed port at the actual first teacher crossing"
        ),
        "wrapper_hook": (
            "scripts.run_physical_graph_edge_handoff_qualification_v2."
            "_derive_candidate_outcome_authority_alignment"
        ),
        "input_authority": {
            "reset_pose_xy": "exact selected serialized-snapshot reset base pose",
            "physics_pose_xy": "all 750 retained candidate/repeat base_pose_world samples",
            "canonical_port": (
                "edge_port_index directed_port_world: actual first teacher crossing xy "
                "and frozen selected-opening-normal heading"
            ),
        },
        "exact_formulas": {
            "port_progress_m": (
                "hypot(reset_x-port_x,reset_y-port_y) - "
                "min_i hypot(pose_i_x-port_x,pose_i_y-port_y) over exactly 750 samples"
            ),
            "lateral_error_m": (
                "abs(-(endpoint_x-port_x)*sin(port_heading) + "
                "(endpoint_y-port_y)*cos(port_heading)) at physics sample 749"
            ),
            "positive_port_progress": "port_progress_m > 0 with no tolerance",
        },
        "changed_output_fields": [
            "port_progress_m",
            "lateral_error_m",
            "positive_port_progress",
        ],
        "required_exact_preservations": [
            "angular_error_rad",
            "oracle_admissible",
            "entered_correct_edge",
            "entered_wrong_edge",
            "no_edge",
            "every crossing and dwell evidence field",
            "every endpoint and source/target-region evidence field",
            "physics_contact",
            "stuck",
            "successor_viable",
            "requested_commands",
            "post_slew_applied_commands",
            "command_tracking_rows",
            "every other candidate outcome field",
        ],
        "applies_to": [
            "all 768 candidate-fanout branches",
            "all 64 repeated-execution branches",
        ],
        "required_after_edge_port_index_freeze": True,
        "uses_outcome_values_to_choose_formula_or_parameter": False,
        "continuous_tuning_or_new_target_definition": False,
        "changes_frozen_metric_gate_class_precedence_or_next_decision": False,
        "changes_frozen_scientific_design": False,
        "aligns_implementation_to_frozen_v1_authority": True,
    }
)

SCIENTIFIC_INVARIANCE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.scientific_invariance_authority.v1",
        "v1_experiment_id": V1_EXPERIMENT_ID,
        "v2_experiment_id": EXPERIMENT_ID,
        "v1_source_freeze_commit": V1_SOURCE_FREEZE_COMMIT,
        "v1_contract_content_digest": V1_CONTRACT_CONTENT_DIGEST,
        "v1_scientific_constants_sha256": V1_SCIENTIFIC_CONSTANTS_SHA256,
        "v2_scientific_constants_sha256": V2_SCIENTIFIC_CONSTANTS_SHA256,
        "scientific_constant_names": list(V1_SCIENTIFIC_CONSTANT_NAMES),
        "scientific_constant_exclusions": copy.deepcopy(
            V1_SCIENTIFIC_CONSTANT_EXCLUSIONS
        ),
        "scientific_constants_equal": (
            V2_SCIENTIFIC_CONSTANT_PROJECTION
            == V1_SCIENTIFIC_CONSTANT_PROJECTION
        ),
        "v1_scientific_projection_sha256": V1_SCIENTIFIC_PROJECTION_SHA256,
        "v1_candidate_specs_sha256": V1_CANDIDATE_SPECS_SHA256,
        "v1_source_dependency_paths_sha256": V1_SOURCE_DEPENDENCY_PATHS_SHA256,
        "invariant_top_level_keys": list(SCIENTIFIC_INVARIANT_KEYS),
        "allowed_difference_top_level_keys": list(
            ALLOWED_V1_V2_CONTRACT_DIFFERENCE_KEYS
        ),
        "v2_only_top_level_keys": list(V2_ONLY_CONTRACT_KEYS),
        "logical_scene_state_spec_episode_ids_and_seeds_unchanged": True,
        "all_metric_formulas_gates_classes_precedence_and_next_decisions_unchanged": True,
        "all_model_controller_geometry_candidate_and_runtime_science_unchanged": True,
        "v1_compatibility_identity_authority_content_digest": (
            V1_COMPATIBILITY_IDENTITY_AUTHORITY["content_digest"]
        ),
        "port_heading_implementation_alignment_authority_content_digest": (
            PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
        ),
        "candidate_port_metric_implementation_alignment_authority_content_digest": (
            CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "content_digest"
            ]
        ),
    }
)

REGRESSION_GATE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.regression_gate_authority.v1",
        "requirements_in_order": list(REGRESSION_REQUIREMENT_IDS),
        "all_required": True,
        "required_before_simulator_creation": True,
        "fixture": copy.deepcopy(REGRESSION_FIXTURE),
        "production_writer_requirement": (
            "exercise the actual V2 first-pool NPZ writer and validator on a deterministic "
            "synthetic payload without constructing a simulator"
        ),
    }
)


V1_CUSTODY_RECEIPT_BINDING = {
    "path": str(V1_CUSTODY_RECEIPT_PATH),
    "bytes": 18403,
    "sha256": "bb4950d2bde0bf1971e643c746b15a28a1949b1ef3d70736bdaae9da45282d32",
}
V1_CUSTODY_EXPECTED_PROJECTION = {
    "source_freeze_commit": V1_SOURCE_FREEZE_COMMIT,
    "disposition": "TECHNICALLY_INVALID_PERSISTED_ARRAY_HASH_METADATA",
    "qualification_pair_count": 8,
    "qualified_count": 7,
    "rejected_count": 1,
    "rejected_pool_indices": [2],
    "pool_002_rejection_reason": "TEACHER_PHYSICS_CONTACT",
    "official_root": {
        "file_count": 1,
        "directory_count": 1,
        "regular_file_apparent_bytes": 66418,
        "allocated_bytes": 73728,
    },
    "material_root": {
        "file_count": 34,
        "directory_count": 15,
        "regular_file_apparent_bytes": 4299695,
        "allocated_bytes": 4435968,
    },
    "defect_evidence": {
        "affected_pair_count": 8,
        "field": "snapshot.previous_applied_command_sha256",
        "metadata_sha256": "cd9cdaf845ca5fd82a869bd1f17362d780cfd1bd3a46fe76cd93575fb5ece5c6",
        "metadata_source_dtype": "<f4",
        "persisted_dtype": "<f8",
        "persisted_v1_canonical_array_sha256": "b237b7cbc24dd1d0bc36a02fa9a0d58e12f511b0fc76d2bbd0aaa48cab0265e3",
        "shape": [3],
        "values": [0.0, 0.0, 0.0],
        "other_embedded_numeric_hash_mismatch_count": 0,
    },
    "scientific_boundary": {
        "teacher_qualification_rows_opened": 8,
        "teacher_qualified": 7,
        "teacher_rejected": 1,
        "rejected_pool_index": 2,
        "rejection_reason": "TEACHER_PHYSICS_CONTACT",
        "prospective_pool_rows": 256,
        "selection_performed": False,
        "selected_state_count": 0,
        "reset_fixture_executions": 0,
        "candidate_fanout_executions": 0,
        "encoder_initializations": 0,
        "ranker_inference_calls": 0,
        "heldout_outcomes_opened": 0,
        "metrics_persisted": False,
        "result_persisted": False,
    },
}
V1_CUSTODY_AND_NONREUSE_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.v1_custody_and_nonreuse_authority.v1",
        "external_receipt": copy.deepcopy(V1_CUSTODY_RECEIPT_BINDING),
        "expected_projection": copy.deepcopy(V1_CUSTODY_EXPECTED_PROJECTION),
        "external_receipt_has_no_content_or_self_digest": True,
        "v1_roots_must_remain_byte_and_inode_stable": True,
        "v1_runtime_artifact_or_shard_reuse_authorized": False,
        "v1_arrays_as_v2_scientific_inputs_authorized": False,
        "v1_copy_or_hardlink_into_v2_authorized": False,
        "permitted_v1_reads": (
            "read-only custody verification and exact first-eight comparison after each "
            "corresponding V2 shard has been freshly produced"
        ),
        "v2_first_eight_must_be_fresh_physical_executions": True,
    }
)

FIRST_EIGHT_REPRODUCTION_ROW_FIELDS = frozenset(
    {
        "pool_index",
        "candidate_spec_id",
        "state_id",
        "scene_id",
        "episode_id",
        "graph_id",
        "identity_equal",
        "snapshot_payload_sha256_equal",
        "teacher_trace_member_inventory_equal",
        "teacher_trace_dtypes_equal",
        "teacher_trace_shapes_equal",
        "teacher_trace_logical_arrays_equal",
        "contact_sequence_equal",
        "stuck_equal",
        "qualified_equal",
        "rejection_reason_equal",
        "all_payload_member_dtypes_equal",
        "all_payload_member_shapes_equal",
        "shared_logical_array_members_equal",
        "noncomparable_v1_known_bad_hash_fields",
        "pass",
    }
)
FIRST_EIGHT_REPRODUCTION_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "status",
        "v1_custody_receipt_binding",
        "comparison_rule",
        "row_count",
        "rows",
        "pass",
        "technical_disposition",
        "full_collection_authorized",
        "compared_before_pool_index",
        "candidate_ranker_development_heldout_outcomes_opened",
    }
)
FIRST_EIGHT_REPRODUCTION_AUTHORITY = attach_content_digest(
    {
        "schema": "physical_graph_edge_handoff_qualification_v2.first_eight_reproduction_authority.v1",
        "pool_indices": list(range(8)),
        "row_fields": sorted(FIRST_EIGHT_REPRODUCTION_ROW_FIELDS),
        "root_fields": sorted(FIRST_EIGHT_REPRODUCTION_FIELDS),
        "comparison_rule": (
            "fresh V2 logical reopened arrays, dtype.str, shapes, identities, contacts, stuck, "
            "qualification and rejection reason must equal immutable V1; compare no V1 "
            "scientific payload until its corresponding fresh V2 shard exists"
        ),
        "known_noncomparable_v1_metadata_fields": [
            "snapshot.previous_applied_command_sha256"
        ],
        "corrected_v2_previous_command_binding": copy.deepcopy(
            PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
        ),
        "stuck_equal_derivation": {
            "source": "reopened teacher trace arrays; V1 shard metadata has no teacher.stuck field",
            "command_activity": "maximum absolute requested_command component over the complete retained teacher trace",
            "translation": "Euclidean XY displacement from first to final base_pose_world sample, expressed in the first-sample body frame",
            "heading": "absolute wrapped yaw change from first to final base_pose_world quaternion",
            "stuck": copy.deepcopy(V1.PHYSICAL_OUTCOME_AUTHORITY["stuck"]),
            "comparison": "derive independently for V1 and fresh V2 and require exact boolean equality",
        },
        "required_before_pool_index": 8,
        "success_status": "PASS",
        "mismatch_status": REPRODUCTION_MISMATCH_DISPOSITION,
        "mismatch_is_technical_terminal_outside_scientific_classes": True,
        "mismatch_output_leaves": list(REPRODUCTION_MISMATCH_LEAVES),
        "mismatch_forbids_panel_roles_candidates_metrics_result_and_commit": True,
    }
)

V2_CORRECTION_RUNTIME_POLICY = {
    "before_simulator_creation": [
        "re-hash and validate the immutable external V1 custody receipt and both V1 roots",
        "validate exact V1/V2 scientific invariance",
        "pass all ten persisted-array regression requirements including the actual production writer fixture",
        "prove the V2 official and material roots are fresh and share no inode with V1",
    ],
    "first_eight": (
        "freshly execute pool indices 0..7 under V2; V1 arrays may be opened read-only "
        "only after the corresponding V2 shard has been persisted and reopened"
    ),
    "before_pool_index_8": (
        "persist and validate v1_v2_first_eight_reproduction.json; continue iff every "
        "logical array, dtype, shape, physical label, qualification and rejection reason matches"
    ),
    "mismatch": (
        "terminal V1_V2_PHYSICAL_REPRODUCTION_MISMATCH outside the six scientific classes; "
        "write no panel, roles, candidates, metrics, result, file manifest, or result commit"
    ),
    "success": "resume the exact inherited V1 stage_order beginning at pool index 8",
    "v1_runtime_artifact_reuse": False,
    "v1_copy_or_hardlink": False,
    "changed_scientific_gate_or_threshold": False,
    "authorized_inherited_implementation_alignment": (
        PORT_HEADING_ALIGNMENT_DISPOSITION
    ),
    "authorized_inherited_candidate_port_metric_alignment": (
        CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION
    ),
}

SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "v1_contract_content_digest",
        "v2_contract_content_digest",
        "v1_scientific_constants_sha256",
        "v2_scientific_constants_sha256",
        "scientific_constants_equal",
        "official_documents_and_result_use_v2_identity_only",
        "v1_compatibility_identity_scopes_exact",
        "inherited_implementation_alignment_disposition",
        "port_heading_alignment_authority_content_digest",
        "candidate_port_metric_alignment_disposition",
        "candidate_port_metric_alignment_authority_content_digest",
        "v1_scientific_projection_sha256",
        "v2_scientific_projection_sha256",
        "candidate_specs_equal",
        "source_dependency_paths_equal",
        "scientific_projection_equal",
        "regression_results",
        "all_ten_regressions_passed_before_simulator_creation",
        "pass",
    }
)
REGRESSION_RESULT_FIELDS = frozenset({"requirement_id", "passed", "evidence"})
V1_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "v1_source_freeze_commit",
        "v2_source_freeze_commit",
        "external_custody_receipt_binding",
        "external_custody_projection_sha256",
        "v1_official_root_unchanged",
        "v1_material_root_unchanged",
        "v1_payloads_copied_into_v2",
        "v1_hardlinks_into_v2",
        "v1_shared_inodes_with_v2",
        "v1_runtime_artifact_or_shard_reused",
        "allowed_read_scope",
        "pass",
    }
)


DOC_PREFIX = "docs/lewm_go2_physical_graph_edge_handoff_qualification_v2"
TRACKED_SOURCE_PATHS = (
    f"{DOC_PREFIX}_contract_2026-09-01.json",
    f"{DOC_PREFIX}_fixture_2026-09-01.json",
    f"{DOC_PREFIX}_output_schema_2026-09-01.json",
    f"{DOC_PREFIX}_preregistration_2026-09-01.md",
    f"{DOC_PREFIX}_source_closure_2026-09-01.json",
    f"{DOC_PREFIX}_scientific_invariance_2026-09-01.json",
    f"{DOC_PREFIX}_v1_custody_binding_2026-09-01.json",
    "lewm/safety/physical_graph_edge_handoff_qualification_v2_contract.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v2_metrics.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v2_contract.py",
    "lewm/tests/test_physical_graph_edge_handoff_qualification_v2_metrics.py",
    "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v2.py",
    "lewm/tests/test_evaluate_physical_graph_edge_handoff_qualification_v2.py",
    "scripts/run_physical_graph_edge_handoff_qualification_v2.py",
    "scripts/evaluate_physical_graph_edge_handoff_qualification_v2.py",
)
SOURCE_DEPENDENCY_PATHS = tuple(V1.SOURCE_DEPENDENCY_PATHS)
V2_WRAPPER_DEPENDENCY_PATHS = (
    "lewm/safety/physical_graph_edge_handoff_qualification_v1_contract.py",
    "lewm/safety/physical_graph_edge_handoff_qualification_v1_metrics.py",
    "lewm/tests/test_run_physical_graph_edge_handoff_qualification_v1.py",
    "scripts/run_physical_graph_edge_handoff_qualification_v1.py",
    "scripts/evaluate_physical_graph_edge_handoff_qualification_v1.py",
)
SOURCE_CLOSURE_PATHS = (
    TRACKED_SOURCE_PATHS[7:]
    + SOURCE_DEPENDENCY_PATHS
    + V2_WRAPPER_DEPENDENCY_PATHS
)


def build_candidate_specs() -> list[dict[str, Any]]:
    return V1.build_candidate_specs()


def build_prospective_pool_specs() -> list[dict[str, Any]]:
    return V1.build_prospective_pool_specs()


def build_contract() -> dict[str, Any]:
    base = copy.deepcopy(V1_SCIENTIFIC_CONTRACT)
    base.pop("content_digest", None)
    base.update(
        {
            "schema": "physical_graph_edge_handoff_qualification_v2.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "output": {
                "root": str(OUTPUT_ROOT),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "leaf_count": OUTPUT_LEAF_COUNT,
                "successful_complete_inventory": list(SUCCESS_OUTPUT_LEAVES),
                "reproduction_mismatch_inventory": list(
                    REPRODUCTION_MISMATCH_LEAVES
                ),
                "receipt_self_digests": False,
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "persisted_array_hash_authority": copy.deepcopy(
                PERSISTED_ARRAY_HASH_AUTHORITY
            ),
            "regression_gate_authority": copy.deepcopy(REGRESSION_GATE_AUTHORITY),
            "scientific_invariance_authority": copy.deepcopy(
                SCIENTIFIC_INVARIANCE_AUTHORITY
            ),
            "v1_compatibility_identity_authority": copy.deepcopy(
                V1_COMPATIBILITY_IDENTITY_AUTHORITY
            ),
            "port_heading_implementation_alignment_authority": copy.deepcopy(
                PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "candidate_port_metric_implementation_alignment_authority": copy.deepcopy(
                CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "v1_custody_and_nonreuse_authority": copy.deepcopy(
                V1_CUSTODY_AND_NONREUSE_AUTHORITY
            ),
            "first_eight_reproduction_authority": copy.deepcopy(
                FIRST_EIGHT_REPRODUCTION_AUTHORITY
            ),
            "v2_correction_runtime_policy": copy.deepcopy(
                V2_CORRECTION_RUNTIME_POLICY
            ),
            "v2_wrapper_dependency_paths": list(V2_WRAPPER_DEPENDENCY_PATHS),
        }
    )
    result = attach_content_digest(base)
    if scientific_invariance_projection(result) != V1_SCIENTIFIC_PROJECTION:
        raise PhysicalGraphEdgeHandoffV2ContractError(
            "V2 scientific projection differs from frozen V1"
        )
    if list(result["source_dependency_paths"]) != list(V1.SOURCE_DEPENDENCY_PATHS):
        raise PhysicalGraphEdgeHandoffV2ContractError(
            "V1 source dependency authority changed"
        )
    if build_candidate_specs() != V1.build_candidate_specs():
        raise PhysicalGraphEdgeHandoffV2ContractError("V1 candidate specs changed")
    if scientific_constant_projection() != V1_SCIENTIFIC_CONSTANT_PROJECTION:
        raise PhysicalGraphEdgeHandoffV2ContractError(
            "V1 scientific constants changed"
        )
    return result


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV2ContractError("contract value drift")
    return copy.deepcopy(dict(value))


RUNTIME_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "status",
        "source_freeze_commit",
        "source_parent_commit",
        "source_baseline_commit",
        "v1_source_freeze_commit",
        "scientific_contract",
        "predecessor_result_binding",
        "external_artifact_bindings",
        "runtime_policy",
        "v1_custody_receipt_binding",
        "v2_correction_runtime_policy",
        "content_digest",
    }
)


def _commit(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalGraphEdgeHandoffV2ContractError(f"invalid {label}")
    return value


def build_runtime_contract(source_freeze_commit: str) -> dict[str, Any]:
    source_freeze_commit = _commit(source_freeze_commit, "source_freeze_commit")
    return attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v2.runtime_contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "FROZEN_BEFORE_PHYSICAL_COLLECTION",
            "source_freeze_commit": source_freeze_commit,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "source_baseline_commit": SOURCE_BASELINE_COMMIT,
            "v1_source_freeze_commit": V1_SOURCE_FREEZE_COMMIT,
            "scientific_contract": build_contract(),
            "predecessor_result_binding": copy.deepcopy(V1.PREDECESSOR_RESULT_BINDING),
            "external_artifact_bindings": [
                copy.deepcopy(row) for row in V1.EXTERNAL_ARTIFACT_BINDINGS
            ],
            "runtime_policy": copy.deepcopy(V1.DIRECT_RUNTIME_POLICY),
            "v1_custody_receipt_binding": copy.deepcopy(
                V1_CUSTODY_RECEIPT_BINDING
            ),
            "v2_correction_runtime_policy": copy.deepcopy(
                V2_CORRECTION_RUNTIME_POLICY
            ),
        }
    )


def validate_runtime_contract(
    value: Mapping[str, Any], *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = validate_content_digest(value)
    if set(row) != RUNTIME_CONTRACT_FIELDS:
        raise PhysicalGraphEdgeHandoffV2ContractError("runtime contract field drift")
    observed = _commit(str(row["source_freeze_commit"]), "source_freeze_commit")
    if source_freeze_commit is not None and observed != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffV2ContractError("source freeze commit drift")
    expected = build_runtime_contract(observed)
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV2ContractError("runtime contract value drift")
    return copy.deepcopy(row)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "PhysicalGraphEdgeHandoffV2ContractError",
    "attach_content_digest",
    "build_candidate_specs",
    "build_contract",
    "build_prospective_pool_specs",
    "build_runtime_contract",
    "canonical_json_bytes",
    "scientific_constant_projection",
    "scientific_invariance_projection",
    "validate_content_digest",
    "validate_contract",
    "validate_runtime_contract",
]
