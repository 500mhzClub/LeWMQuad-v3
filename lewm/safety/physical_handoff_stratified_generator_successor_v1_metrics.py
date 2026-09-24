"""Pure reducer authority for the stratified handoff generator successor.

Successor envelopes and candidate cardinality are validated natively.  The
module calls frozen V4 numerical and semantic helpers directly only where
their inputs are explicit raw values and independent of an experiment
registry.  It never rewrites an inherited module namespace or translates a
successor document into a predecessor schema.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v4_metrics as _V4M
from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as _V1M
from lewm.safety import physical_handoff_stratified_generator_successor_v1_contract as C


class PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(ValueError):
    """Raised when successor evidence is incomplete or inconsistent."""


class PhysicalHandoffStratifiedGeneratorSuccessorV1HardStop(
    PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError
):
    """Raised for the unchanged exhaustive V4 technical hard-stop set."""


def _mapping(value: Any, fields: Sequence[str] | set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} field drift"
        )
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not a sequence"
        )
    return [copy.deepcopy(item) for item in value]


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not SHA-256"
        )
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not a lowercase Git commit"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not a nonnegative integer"
        )
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not bool"
        )
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} is not finite"
        )
    return result


def _vector(value: Any, size: int, label: str) -> list[float]:
    row = _sequence(value, label)
    if len(row) != size:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} extent drift"
        )
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(row)]


def _close(left: Any, right: Any, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance)


def _canonical_no_lf_sha256(value: Any) -> str:
    return hashlib.sha256(C.canonical_json_bytes(value)[:-1]).hexdigest()


def _candidate_coordinates(index: int) -> tuple[str, int, int]:
    index = _nonnegative_int(index, "candidate_index")
    if index >= C.MAX_CANDIDATE_COUNT:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate_index drift"
        )
    attempt, stream = divmod(index, C.STREAM_COUNT)
    family_index, stratum = divmod(stream, C.STRATA_PER_FAMILY)
    return C.FAMILY_IDS[family_index], stratum, attempt


# Registry-independent V4 formulas are ordinary direct calls.  Functions that
# bind a candidate population are implemented natively below.
frozen_binary64_fma = _V4M.frozen_binary64_fma
canonical_v1_planar_segment_projection = (
    _V4M.canonical_v1_planar_segment_projection
)
persisted_array_bytes_sha256 = _V4M.persisted_array_bytes_sha256
persisted_array_manifest_row = _V4M.persisted_array_manifest_row
derive_candidate_port_metrics = _V4M.derive_candidate_port_metrics
canonical_semantic_snapshot = _V4M.canonical_semantic_snapshot
semantic_snapshot_sha256 = _V4M.semantic_snapshot_sha256
snapshot_semantic_evidence = _V4M.snapshot_semantic_evidence
snapshot_behavioural_digest = _V4M.snapshot_behavioural_digest
compare_behavioural_probe_traces = _V4M.compare_behavioural_probe_traces
compare_terminated_behavioural_probe_traces = (
    _V4M.compare_terminated_behavioural_probe_traces
)
compare_tipped_behavioural_probe_traces = (
    _V4M.compare_tipped_behavioural_probe_traces
)
classify_state_disposition = _V4M.classify_state_disposition
runtime_environment_sha256 = _V4M.runtime_environment_sha256
validate_physical_runtime_environment = _V4M.validate_physical_runtime_environment
point_in_polygon_inclusive = _V4M.point_in_polygon_inclusive
first_registered_port_crossing = _V4M.first_registered_port_crossing
_terminal_nonfinite_masks = _V4M._terminal_nonfinite_masks
_teacher_termination_flags_from_pose = _V4M._teacher_termination_flags_from_pose
_pose_roll_pitch_yaw_xyzw = _V4M._pose_roll_pitch_yaw_xyzw
_wrap_angle_v1 = _V4M._wrap_angle_v1
_teacher_crossing_reduction = _V4M._teacher_crossing_reduction
_expected_registered_graph = _V4M._expected_registered_graph
_canonical_array_sha256 = _V4M._canonical_array_sha256
_trace_member_manifest = _V4M._trace_member_manifest
classify_physical_handoff_aggregates = _V1M.classify_physical_handoff_aggregates
_condition_summary = _V1M._condition_summary
_materially_outperforms = _V1M._materially_outperforms
_command_tracking_summary = _V1M._command_tracking_summary


def _termination_flags(
    value: Any, label: str, *, nullable: bool = False,
) -> dict[str, bool] | None:
    if value is None and nullable:
        return None
    row = _mapping(value, C.TERMINATION_FLAGS_FIELDS, label)
    for field in C.TERMINATION_FLAG_ORDER:
        _bool(row[field], f"{label}.{field}")
    return row


def _teacher_criteria(value: Any) -> dict[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) not in (
        set(C.TEACHER_QUALIFICATION_COMPONENT_IDS),
        set(C.TEACHER_CRITERIA_FIELDS),
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher criteria field drift"
        )
    result = copy.deepcopy(dict(value))
    for field in C.TEACHER_QUALIFICATION_COMPONENT_IDS:
        _bool(result[field], f"teacher_criteria.{field}")
    expected_valid = all(
        result[field]
        for field in C.TEACHER_QUALIFICATION_COMPONENT_IDS
        if field not in {"goal_reachable", "current_rgb_valid"}
    )
    if "teacher_valid" in result and result["teacher_valid"] is not expected_valid:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher_valid formula drift"
        )
    result["teacher_valid"] = expected_valid
    return result


def _snapshot_identity(
    value: Any, label: str, *, probe_tipped: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    row = _mapping(value, C.SNAPSHOT_IDENTITY_FIELDS, label)
    _sha(row["artifact_file_sha256"], f"{label}.artifact")
    _sha(row["snapshot_semantic_digest_v1"], f"{label}.semantic")
    if probe_tipped:
        if row["snapshot_behavioural_digest_v1"] is not None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "probe-tipped behavioural digest must be null"
            )
    else:
        _sha(row["snapshot_behavioural_digest_v1"], f"{label}.behavioural")
    return row


def build_state_disposition_record(
    candidate_spec: Mapping[str, Any], *, stage_reached: str,
    initial_termination_flags: Mapping[str, Any],
    probe_trial_termination_flags: Sequence[Mapping[str, Any]] | None = None,
    teacher_termination_flags: Mapping[str, Any] | None = None,
    probe_tip_sample_indices: Sequence[int | None] | None = None,
    teacher_criteria: Mapping[str, Any] | None = None,
    executable_snapshot_exists: bool, teacher_executed: bool,
    snapshot_identity: Mapping[str, Any] | None = None,
    diagnostics_inventory: Sequence[str] = (),
    payload_member_inventory: Sequence[str] = (),
    materialisation_corrupt: bool = False,
    state_nondeterministic: bool = False,
    unresolved_state_failure: bool = False,
    reset_or_candidate_outcome_opened: bool = False,
) -> dict[str, Any]:
    spec = C.validate_candidate_spec(candidate_spec)
    index = spec["candidate_index"]
    if stage_reached not in C.STATE_DISPOSITION_STAGES:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "stage_reached drift"
        )
    initial = _termination_flags(initial_termination_flags, "initial flags")
    assert initial is not None
    probe = None
    if probe_trial_termination_flags is not None:
        probe = [
            _termination_flags(item, f"probe flags[{offset}]")
            for offset, item in enumerate(
                _sequence(probe_trial_termination_flags, "probe flags")
            )
        ]
        if len(probe) != 2 or any(item is None for item in probe):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "probe trial cardinality drift"
            )
    tips = None
    if probe_tip_sample_indices is not None:
        tips = _sequence(probe_tip_sample_indices, "probe tip sample indices")
        if len(tips) != 2 or any(
            item is not None
            and (
                not isinstance(item, int)
                or isinstance(item, bool)
                or not 0 <= item < C.V4.V3.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES
            )
            for item in tips
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "probe tip sample index drift"
            )
    teacher_flags = _termination_flags(
        teacher_termination_flags, "teacher flags", nullable=True
    )
    criteria = _teacher_criteria(teacher_criteria)
    disposition, failed = classify_state_disposition(
        initial_termination_flags=initial,
        probe_trial_termination_flags=probe,
        teacher_termination_flags=teacher_flags,
        teacher_criteria=criteria,
        materialisation_corrupt=_bool(
            materialisation_corrupt, "materialisation_corrupt"
        ),
        state_nondeterministic=_bool(
            state_nondeterministic, "state_nondeterministic"
        ),
        unresolved_state_failure=_bool(
            unresolved_state_failure, "unresolved_state_failure"
        ),
    )
    executable = _bool(executable_snapshot_exists, "executable_snapshot_exists")
    teacher_done = _bool(teacher_executed, "teacher_executed")
    if _bool(reset_or_candidate_outcome_opened, "reset_or_candidate_outcome_opened"):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification opened reset/candidate outcome"
        )
    if stage_reached == "INITIAL_BOUNDARY":
        if (
            executable
            or teacher_done
            or probe is not None
            or tips is not None
            or teacher_flags is not None
            or criteria is not None
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial-boundary stage/evidence drift"
            )
    elif stage_reached == "RESTORATION_PROBE":
        if (
            any(initial.values())
            or not executable
            or teacher_done
            or probe is None
            or tips is None
            or teacher_flags is not None
            or criteria is not None
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "restoration-probe stage/evidence drift"
            )
    else:
        if (
            any(initial.values())
            or not executable
            or not teacher_done
            or probe is None
            or any(any(item.values()) for item in probe if item is not None)
            or tips != [None, None]
            or teacher_flags is None
            or criteria is None
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "teacher stage/evidence drift"
            )
    qualified = disposition == "QUALIFIED"
    if qualified and stage_reached != "COMPLETE":
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualified completion drift"
        )
    if teacher_done and not qualified and stage_reached != "TEACHER_EXECUTION":
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "nonqualified teacher stage drift"
        )
    identity = _snapshot_identity(
        snapshot_identity,
        "snapshot_identity",
        probe_tipped=stage_reached == "RESTORATION_PROBE",
    )
    if executable is not (identity is not None):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "snapshot identity/existence drift"
        )
    diagnostics = _sequence(diagnostics_inventory, "diagnostics inventory")
    members = _sequence(payload_member_inventory, "payload member inventory")
    for label, values in (("diagnostics", diagnostics), ("payload members", members)):
        if (
            any(not isinstance(item, str) or not item for item in values)
            or values != sorted(set(values))
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} inventory drift"
            )
    if stage_reached == "INITIAL_BOUNDARY" and disposition not in C.HARD_STOP_DISPOSITIONS:
        if set(members) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial-boundary payload inventory drift"
            )
        if set(diagnostics) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial-boundary diagnostics inventory drift"
            )
    result = C.attach_content_digest(
        {
            "schema": C.QUALIFICATION_DISPOSITIONS_SCHEMA,
            "experiment_id": C.EXPERIMENT_ID,
            "pool_index": index,
            **{
                key: spec[key]
                for key in (
                    "candidate_spec_id",
                    "state_id",
                    "scene_id",
                    "episode_id",
                    "graph_id",
                    "family",
                    "stratum_index",
                    "variant_index",
                    "procedural_seed",
                )
            },
            "canonical_spec_sha256": spec["canonical_spec_sha256"],
            "disposition": disposition,
            "qualified": qualified,
            "failed_criteria": failed,
            "stage_reached": stage_reached,
            "hard_stop": disposition in C.HARD_STOP_DISPOSITIONS,
            "continuation_authorized": disposition not in C.HARD_STOP_DISPOSITIONS,
            "executable_snapshot_exists": executable,
            "teacher_executed": teacher_done,
            "reset_or_candidate_outcome_opened": False,
            "initial_termination_flags": initial,
            "probe_trial_termination_flags": probe,
            "teacher_termination_flags": teacher_flags,
            "probe_tip_sample_indices": tips,
            "teacher_criteria": criteria,
            "snapshot_identity": identity,
            "diagnostics_inventory": diagnostics,
            "payload_member_inventory": members,
        }
    )
    if set(result) != set(C.STATE_DISPOSITION_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "state disposition output field drift"
        )
    return result


def validate_state_disposition_record(
    value: Any, *, expected_pool_index: int | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.STATE_DISPOSITION_FIELDS, "state disposition")
    C.validate_content_digest(row)
    index = _nonnegative_int(row["pool_index"], "candidate_index")
    if index >= C.MAX_CANDIDATE_COUNT or (
        expected_pool_index is not None and index != expected_pool_index
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "state disposition candidate index drift"
        )
    family, stratum, attempt = _candidate_coordinates(index)
    spec = C.build_candidate_spec(family, stratum, attempt)
    expected = build_state_disposition_record(
        spec,
        stage_reached=row["stage_reached"],
        initial_termination_flags=row["initial_termination_flags"],
        probe_trial_termination_flags=row["probe_trial_termination_flags"],
        teacher_termination_flags=row["teacher_termination_flags"],
        probe_tip_sample_indices=row["probe_tip_sample_indices"],
        teacher_criteria=row["teacher_criteria"],
        executable_snapshot_exists=row["executable_snapshot_exists"],
        teacher_executed=row["teacher_executed"],
        snapshot_identity=row["snapshot_identity"],
        diagnostics_inventory=row["diagnostics_inventory"],
        payload_member_inventory=row["payload_member_inventory"],
        materialisation_corrupt=(
            row["disposition"] == "STATE_MATERIALISATION_CORRUPT"
        ),
        state_nondeterministic=(
            row["disposition"] == "STATE_NONDETERMINISTIC"
        ),
        unresolved_state_failure=(
            "unresolved_state_failure" in row["failed_criteria"]
        ),
        reset_or_candidate_outcome_opened=row[
            "reset_or_candidate_outcome_opened"
        ],
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "state disposition value drift"
        )
    return row


def validate_npz_archive_comment(value: bytes | str) -> str:
    if isinstance(value, bytes):
        try:
            observed = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "NPZ archive comment is not UTF-8"
            ) from exc
    elif isinstance(value, str):
        observed = value
    else:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "NPZ archive comment type drift"
        )
    if observed != C.NPZ_ARCHIVE_COMMENT:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "NPZ archive comment drift"
        )
    return observed


def build_persisted_array_evidence(
    *, shard_kind: str, shard_id: str, payload_file: Mapping[str, Any],
    arrays: Mapping[str, Any], reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _file_binding(payload_file, "payload_file")
    if not isinstance(arrays, Mapping) or not arrays:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "persisted arrays are empty"
        )
    if set(arrays) != set(reopened_arrays):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "save/reopen member inventory drift"
        )
    rows: list[dict[str, Any]] = []
    for member in sorted(arrays):
        before = persisted_array_manifest_row(str(member), arrays[member])
        after = persisted_array_manifest_row(
            str(member), reopened_arrays[member]
        )
        if before != after:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"save/reopen persisted array drift: {member}"
            )
        rows.append(before)
    result = {
        "schema": C.PERSISTED_ARRAY_EVIDENCE_SCHEMA,
        "experiment_id": C.EXPERIMENT_ID,
        "shard_kind": str(shard_kind),
        "shard_id": str(shard_id),
        "payload_file": payload,
        "array_count": len(rows),
        "array_inventory_sha256": hashlib.sha256(
            C.canonical_json_bytes(rows)[:-1]
        ).hexdigest(),
        "arrays": rows,
        "save_reopen_validation_passed": True,
    }
    return validate_persisted_array_evidence(
        result, reopened_arrays=reopened_arrays
    )


def validate_persisted_array_evidence(
    value: Any, *, reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    row = _mapping(
        value, C.PERSISTED_ARRAY_EVIDENCE_FIELDS, "persisted array evidence"
    )
    if (
        row["schema"] != C.PERSISTED_ARRAY_EVIDENCE_SCHEMA
        or row["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "persisted array identity drift"
        )
    if not isinstance(row["shard_kind"], str) or not row["shard_kind"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "shard_kind is empty"
        )
    if not isinstance(row["shard_id"], str) or not row["shard_id"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "shard_id is empty"
        )
    payload = _file_binding(row["payload_file"], "payload_file")
    arrays = _sequence(row["arrays"], "persisted arrays")
    if (
        not isinstance(row["array_count"], int)
        or isinstance(row["array_count"], bool)
        or row["array_count"] != len(arrays)
        or not arrays
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "persisted array count drift"
        )
    members: list[str] = []
    validated_rows: list[dict[str, Any]] = []
    for index, item in enumerate(arrays):
        part = _mapping(
            item, C.PERSISTED_ARRAY_ROW_FIELDS, f"array[{index}]"
        )
        member = part["member"]
        if not isinstance(member, str) or not member:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array member is empty"
            )
        if not isinstance(part["dtype_str"], str) or not part["dtype_str"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array dtype is empty"
            )
        try:
            dtype = np.dtype(part["dtype_str"])
        except TypeError as exc:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array dtype is invalid"
            ) from exc
        if dtype.hasobject or dtype.str != part["dtype_str"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array dtype is not exact numpy dtype.str"
            )
        if (
            not isinstance(part["shape"], list)
            or any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                for size in part["shape"]
            )
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array shape is invalid"
            )
        if part["c_contiguous"] is not True:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "array is not C-contiguous"
            )
        _sha(part["array_bytes_sha256"], f"array[{index}] bytes sha256")
        members.append(member)
        validated_rows.append(part)
    if members != sorted(members) or len(set(members)) != len(members):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "array member order/uniqueness drift"
        )
    inventory_sha = hashlib.sha256(
        C.canonical_json_bytes(validated_rows)[:-1]
    ).hexdigest()
    if row["array_inventory_sha256"] != inventory_sha:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "array inventory digest drift"
        )
    if row["save_reopen_validation_passed"] is not True:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "save/reopen validation did not pass"
        )
    if reopened_arrays is not None:
        if set(reopened_arrays) != set(members):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "reopened member inventory drift"
            )
        for part in validated_rows:
            expected = persisted_array_manifest_row(
                part["member"], reopened_arrays[part["member"]]
            )
            if part != expected:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"reopened array evidence drift: {part['member']}"
                )
    return row


def validate_snapshot_previous_applied_command_binding(
    snapshot_metadata: Mapping[str, Any],
    persisted_array_evidence: Mapping[str, Any], *,
    reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(snapshot_metadata, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "snapshot metadata is not a mapping"
        )
    evidence = validate_persisted_array_evidence(
        persisted_array_evidence, reopened_arrays=reopened_arrays
    )
    expected = C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
    matches = [
        item
        for item in evidence["arrays"]
        if item["member"] == expected["member"]
    ]
    if len(matches) != 1:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "previous-command persisted-array member coverage drift"
        )
    member = matches[0]
    if (
        member["dtype_str"] != expected["dtype_str"]
        or member["shape"] != expected["per_snapshot_shape"]
        or snapshot_metadata.get("previous_applied_command_sha256")
        != member["array_bytes_sha256"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "snapshot previous-command raw-byte binding drift"
        )
    return {
        "member": member["member"],
        "dtype_str": member["dtype_str"],
        "shape": copy.deepcopy(member["shape"]),
        "array_bytes_sha256": member["array_bytes_sha256"],
    }


def validate_state_material_payload(
    state_disposition: Mapping[str, Any],
    persisted_array_evidence: Mapping[str, Any], *,
    reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    state = validate_state_disposition_record(state_disposition)
    persisted = validate_persisted_array_evidence(
        persisted_array_evidence, reopened_arrays=reopened_arrays
    )
    names = sorted(str(name) for name in reopened_arrays)
    if names != state["payload_member_inventory"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "state/payload member inventory drift"
        )
    if (
        state["stage_reached"] == "INITIAL_BOUNDARY"
        and state["disposition"] not in C.HARD_STOP_DISPOSITIONS
    ):
        import numpy as np

        rows = {item["member"]: item for item in persisted["arrays"]}
        if set(rows) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial payload member drift"
            )
        for member, authority in C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY.items():
            if (
                rows[member]["dtype_str"] != authority["dtype_str"]
                or rows[member]["shape"] != authority["shape"]
            ):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"initial payload extent drift: {member}"
                )
        family, stratum, attempt = _candidate_coordinates(state["pool_index"])
        spec = C.build_candidate_spec(family, stratum, attempt)
        spawn_x, spawn_y, spawn_yaw = spec["geometry"]["spawn_se2_world"]
        expected_pose = np.asarray(
            [
                spawn_x,
                spawn_y,
                0.375,
                0.0,
                0.0,
                math.sin(spawn_yaw / 2.0),
                math.cos(spawn_yaw / 2.0),
            ],
            dtype=np.float64,
        )
        if not np.array_equal(
            np.asarray(reopened_arrays["intended_base_pose_world"]),
            expected_pose,
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "intended initial pose formula drift"
            )
        expected_flags = np.asarray(
            [
                int(state["initial_termination_flags"][key])
                for key in C.TERMINATION_FLAG_ORDER
            ],
            dtype=np.uint8,
        )
        if not np.array_equal(
            np.asarray(reopened_arrays["termination_flags"]), expected_flags
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial termination flags binding drift"
            )
        observed = {
            member: np.asarray(reopened_arrays[member])
            for member in C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY
        }
        try:
            _terminal_nonfinite_masks(
                observed,
                state["initial_termination_flags"],
                label="initial boundary",
                allowed_members=C.TERMINAL_NONFINITE_INITIAL_MEMBER_IDS,
                require_trace_axis=False,
            )
        except Exception as exc:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                str(exc)
            ) from exc
        for member in ("intended_base_pose_world", "previous_applied_command"):
            if not bool(np.isfinite(observed[member]).all()):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"initial boundary nonfinite diagnostic: {member}"
                )
        if not bool(np.isin(observed["physics_contact"], [0, 1]).all()):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial boundary physics contact is not binary"
            )
        if _teacher_termination_flags_from_pose(
            reopened_arrays["base_pose_world"]
        ) != state["initial_termination_flags"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial boundary flags differ from observed pose"
            )
        for member in (
            "sim_time_ns",
            "episode_step",
            "command_ticks",
            "policy_steps",
        ):
            if int(np.asarray(reopened_arrays[member])[0]) < 0:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"initial boundary negative counter: {member}"
                )
    return {
        "state_disposition": state,
        "persisted_array_evidence": persisted,
    }


def reduce_qualification_teacher_trace(
    metadata: Mapping[str, Any], reopened_arrays: Mapping[str, Any], *,
    expected_pool_index: int | None = None,
) -> dict[str, Any]:
    """Independently reduce all teacher predicates from successor raw arrays."""

    import numpy as np

    if not isinstance(metadata, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher material metadata is not mapping"
        )
    index = _nonnegative_int(metadata.get("pool_index"), "candidate_index")
    if index >= C.MAX_CANDIDATE_COUNT or (
        expected_pool_index is not None and index != expected_pool_index
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher candidate identity drift"
        )
    family, stratum, attempt = _candidate_coordinates(index)
    spec = C.build_candidate_spec(family, stratum, attempt)
    if C.canonical_json_bytes(metadata.get("candidate_spec")) != (
        C.canonical_json_bytes(spec)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher candidate spec drift"
        )
    traces: dict[str, Any] = {}
    byte_members = {
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "target_region_member",
    }
    tails = {
        "timestamp_s": (),
        "base_pose_world": (7,),
        "base_twist_world": (6,),
        "joint_position": (12,),
        "joint_velocity": (12,),
        "applied_command": (3,),
        "requested_command": (3,),
        "physics_contact": (),
        "source_region_member": (),
        "edge_region_member": (),
        "target_region_member": (),
    }
    sample_count: int | None = None
    for member in C.TEACHER_TRACE_MEMBER_ORDER:
        name = f"teacher__{member}"
        if name not in reopened_arrays:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"teacher raw trace member absent: {member}"
            )
        array = np.asarray(reopened_arrays[name])
        expected_dtype = "|u1" if member in byte_members else "<f8"
        if (
            array.dtype.str != expected_dtype
            or array.ndim != 1 + len(tails[member])
            or tuple(array.shape[1:]) != tails[member]
            or not array.flags.c_contiguous
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"teacher raw trace extent drift: {member}"
            )
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif int(array.shape[0]) != sample_count:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "teacher raw trace sample alignment drift"
            )
        if member in byte_members and not bool(np.isin(array, [0, 1]).all()):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"teacher raw trace nonbinary member: {member}"
            )
        traces[member] = array
    maximum = int(C.TEACHER_CONTROLLER_AUTHORITY["maximum_command_ticks"]) * int(
        round(
            float(C.TEACHER_CONTROLLER_AUTHORITY["command_tick_s"])
            / float(C.TEACHER_TRACE_DT_S)
        )
    )
    if sample_count is None or not 1 <= sample_count <= maximum:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher raw trace sample count drift"
        )
    termination_flags = _teacher_termination_flags_from_pose(
        traces["base_pose_world"][-1]
    )
    try:
        _terminal_nonfinite_masks(
            traces,
            termination_flags,
            label="teacher trace",
            allowed_members=C.TERMINAL_NONFINITE_TRACE_MEMBER_IDS,
            require_trace_axis=True,
        )
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            str(exc)
        ) from exc
    if sample_count > 1 and not bool(
        np.allclose(
            np.diff(traces["timestamp_s"]),
            C.TEACHER_TRACE_DT_S,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher raw trace timestamp cadence drift"
        )
    geometry = spec["geometry"]
    poses = traces["base_pose_world"]
    nan_terminal = termination_flags["nan"]
    finite_stop = sample_count - 1 if nan_terminal else sample_count
    if finite_stop:
        science_poses = poses[:finite_stop]
    else:
        anchor = np.asarray(reopened_arrays.get("snapshot__base_pose_world"))
        if (
            anchor.dtype.str != "<f8"
            or list(anchor.shape) != [7]
            or not bool(np.isfinite(anchor).all())
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "first-sample nan teacher lacks finite captured pose anchor"
            )
        science_poses = np.ascontiguousarray(anchor.reshape(1, 7))
    memberships = {
        "source_region_member": geometry["source_node"]["boundary_polygon_world"],
        "edge_region_member": geometry["selected_directed_edge"][
            "edge_region_polygon_world"
        ],
        "target_region_member": geometry["target_node"]["boundary_polygon_world"],
    }
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    science_memberships: dict[str, Any] = {}
    for member, polygon in memberships.items():
        expected_all = np.asarray(
            [
                int(
                    point_in_polygon_inclusive(
                        pose[:2].tolist(), polygon, tolerance_m=tolerance
                    )
                )
                if bool(np.isfinite(pose[:2]).all())
                else 0
                for pose in poses
            ],
            dtype=np.uint8,
        )
        if not np.array_equal(traces[member], expected_all):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"teacher {member} differs from registered geometry"
            )
        if finite_stop:
            science_memberships[member] = expected_all[:finite_stop]
        else:
            science_memberships[member] = np.asarray(
                [
                    int(
                        point_in_polygon_inclusive(
                            science_poses[0, :2].tolist(),
                            polygon,
                            tolerance_m=tolerance,
                        )
                    )
                ],
                dtype=np.uint8,
            )
    try:
        crossing, crossing_error, competing_crossing, competing = (
            _teacher_crossing_reduction(
                science_poses,
                science_memberships["target_region_member"],
                spec,
            )
        )
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            str(exc)
        ) from exc
    selected_opening = np.asarray(
        geometry["selected_directed_edge"]["opening_segment_world"],
        dtype=np.float64,
    )
    opening_projection = canonical_v1_planar_segment_projection(
        selected_opening[0], selected_opening
    )
    opening_width = opening_projection["segment_width_m"]
    midpoint = np.asarray(
        opening_projection["midpoint_world_xy"], dtype=np.float64
    )
    distances = np.linalg.norm(science_poses[:, :2] - midpoint[None, :], axis=1)
    route_progress = float(distances[0] - distances.min())
    positive = route_progress > 0.0
    contact_free = not bool(traces["physics_contact"].any())
    source = science_memberships["source_region_member"]
    left_source = bool((source == 0).any())
    exits = np.flatnonzero((source[:-1] != 0) & (source[1:] == 0)) + 1
    first_exit = int(exits[0]) if len(exits) else None
    reached_target = bool(science_memberships["target_region_member"].any())
    try:
        graph = _expected_registered_graph(
            spec,
            positive_route_progress=positive,
            competing_port_entered=competing_crossing is not None,
        )
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            str(exc)
        ) from exc
    if C.canonical_json_bytes(metadata.get("graph")) != C.canonical_json_bytes(graph):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher graph differs from registered geometry"
        )
    criteria = {
        "goal_reachable": True,
        "teacher_trace_contact_free": contact_free,
        "teacher_left_source_region": left_source,
        "teacher_crossed_directed_port": crossing is not None,
        "teacher_positive_route_progress": positive,
        "teacher_no_competing_port": not competing,
        "teacher_normal_positive": bool(
            crossing and crossing["normal_dot_displacement_m"] > 0.0
        ),
        "teacher_within_lateral_bounds": bool(
            crossing
            and abs(crossing["lateral_coordinate_m"])
            <= float(spec["passage_width_m"]) / 2.0 + 1.0e-9
        ),
        "teacher_dwell_satisfied": bool(
            crossing and crossing["sustained_or_target_reached"]
        ),
        "directed_port_defined": crossing is not None,
        "current_rgb_valid": True,
        "graph_edge_physically_executable": True,
    }
    teacher_valid = bool(
        crossing is not None
        and contact_free
        and left_source
        and positive
        and not competing
    )
    criteria["teacher_valid"] = teacher_valid
    state = validate_state_disposition_record(
        metadata.get("state_disposition"), expected_pool_index=index
    )
    if (
        state["teacher_criteria"] != criteria
        or state["teacher_termination_flags"] != termination_flags
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "teacher state criteria differ from raw trace"
        )
    start_pose = science_poses[0]
    end_pose = science_poses[-1]
    _start_roll, _start_pitch, start_yaw = _pose_roll_pitch_yaw_xyzw(start_pose)
    _end_roll, _end_pitch, end_yaw = _pose_roll_pitch_yaw_xyzw(end_pose)
    dx = float(end_pose[0] - start_pose[0])
    dy = float(end_pose[1] - start_pose[1])
    body_dx = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
    body_dy = -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
    heading = _wrap_angle_v1(end_yaw - start_yaw)
    activity = float(np.max(np.abs(traces["requested_command"])))
    stuck_authority = C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]
    stuck = bool(
        activity > float(stuck_authority["command_activity_threshold"])
        and math.hypot(body_dx, body_dy)
        < float(stuck_authority["h3_translation_threshold_m"])
        and abs(heading) < float(stuck_authority["h3_heading_threshold_rad"])
    )
    target_indices = np.flatnonzero(
        science_memberships["target_region_member"] != 0
    )
    first_target_index = int(target_indices[0]) if len(target_indices) else None
    normal = np.asarray(
        geometry["selected_directed_edge"]["opening_normal_world"],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    endpoint_lateral = abs(
        canonical_v1_planar_segment_projection(
            end_pose[:2], selected_opening
        )["lateral_coordinate_m"]
    )
    endpoint_angular = abs(
        _wrap_angle_v1(
            end_yaw - math.atan2(float(normal[1]), float(normal[0]))
        )
    )
    successor_authority = C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]
    successor = bool(
        not nan_terminal
        and float(end_pose[2])
        >= float(successor_authority["minimum_base_height_m"])
        and abs(_end_roll)
        <= float(successor_authority["maximum_absolute_roll_rad"])
        and abs(_end_pitch)
        <= float(successor_authority["maximum_absolute_pitch_rad"])
        and not bool(traces["physics_contact"][-1])
    )
    if crossing is None:
        crossing_velocity = None
        crossing_velocity_heading = None
        target_early = False
        dwell_count = 0
    else:
        before = int(crossing["sample_before"])
        after = int(crossing["sample_after"])
        alpha = float(crossing["fraction"])
        velocity = (
            (1.0 - alpha) * traces["base_twist_world"][before, :2]
            + alpha * traces["base_twist_world"][after, :2]
        )
        crossing_velocity = [float(item) for item in velocity]
        crossing_velocity_heading = (
            math.atan2(crossing_velocity[1], crossing_velocity[0])
            if math.hypot(*crossing_velocity) > 0.0
            else None
        )
        dwell_count = int(crossing["sustained_beyond_samples"])
        target_early = bool(
            first_target_index is not None
            and first_target_index - after + 1 < C.PORT_DWELL_PHYSICS_SAMPLES
        )
    record_projection = {
        "sample_count": sample_count,
        "contact_free": contact_free,
        "left_source_region": left_source,
        "first_source_exit_sample_index": first_exit,
        "first_crossing_sample_index": None if crossing is None else crossing["sample_after"],
        "crossing_segment_fraction": None if crossing is None else crossing["fraction"],
        "crossed_directed_port": crossing is not None,
        "positive_route_progress": positive,
        "route_progress_m": route_progress,
        "crossing_directed_normal_dot": None if crossing is None else crossing["normal_dot_displacement_m"],
        "crossing_lateral_fraction": None if crossing is None else float(crossing["lateral_coordinate_m"]) / opening_width + 0.5,
        "crossing_velocity_world_xy": crossing_velocity,
        "crossing_velocity_heading_world_rad": crossing_velocity_heading,
        "beyond_port_consecutive_physics_samples": dwell_count,
        "target_entered_before_dwell_complete": target_early,
        "competing_port_entered": competing,
        "reached_target_node": reached_target,
        "first_target_entry_sample_index": first_target_index,
        "where_reached": "TARGET_NODE" if reached_target else "BEYOND_DIRECTED_PORT",
        "endpoint_lateral_error_m": endpoint_lateral,
        "endpoint_angular_error_rad": endpoint_angular,
        "successor_viable": successor,
        "stuck": stuck,
        "teacher_valid": teacher_valid,
        "goal_reachable": True,
        "directed_port_defined": crossing is not None,
        "graph_edge_physically_executable": True,
    }
    if set(record_projection) != set(C.V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "raw teacher-record projection field drift"
        )
    result = {
        "pool_index": index,
        "candidate_spec_id": spec["candidate_spec_id"],
        "sample_count": sample_count,
        "goal_reachable": True,
        "graph_edge_physically_executable": True,
        "termination_flags": termination_flags,
        "contact_free": contact_free,
        "left_source_region": left_source,
        "first_source_exit_sample_index": first_exit,
        "crossing": crossing,
        "crossing_error": crossing_error,
        "competing_crossing": competing_crossing,
        "competing_port_entered": competing,
        "route_progress_m": route_progress,
        "positive_route_progress": positive,
        "reached_target_node": reached_target,
        "teacher_valid": teacher_valid,
        "stuck": stuck,
        "teacher_criteria": criteria,
        "disposition": state["disposition"],
        "failed_criteria": copy.deepcopy(state["failed_criteria"]),
        "teacher_record_raw_projection": record_projection,
    }
    if set(result) != set(C.V4_RAW_TEACHER_REDUCTION_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "raw teacher reduction field drift"
        )
    return result


def validate_qualification_teacher_raw_evidence(
    metadata: Mapping[str, Any], reopened_arrays: Mapping[str, Any], *,
    expected_pool_index: int | None = None,
    teacher_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reduction = reduce_qualification_teacher_trace(
        metadata,
        reopened_arrays,
        expected_pool_index=expected_pool_index,
    )
    if teacher_record is not None:
        record = _mapping(
            teacher_record, C.V4_TEACHER_RECORD_FIELDS, "teacher record"
        )
        if (
            record["qualification_pool_index"] != reduction["pool_index"]
            or record["candidate_spec_id"] != reduction["candidate_spec_id"]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "teacher record/raw shard identity drift"
            )
        projection = reduction["teacher_record_raw_projection"]
        if any(
            C.canonical_json_bytes(record[field])
            != C.canonical_json_bytes(projection[field])
            for field in C.V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "teacher record differs from raw shard reduction"
            )
    return reduction


def validate_qualification_material_shard(
    metadata: Any, *, reopened_arrays: Mapping[str, Any],
    expected_pool_index: int | None = None,
) -> dict[str, Any]:
    """Validate one complete successor terminal metadata/NPZ pair."""

    import numpy as np

    if not isinstance(metadata, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification metadata is not mapping"
        )
    disposition = metadata.get("disposition")
    if disposition == "INITIAL_BOUNDARY_TIPPED" or (
        metadata.get("stage_reached") == "INITIAL_BOUNDARY"
        and disposition == "UNRESOLVED_STATE_FAILURE"
    ):
        fields = C.INITIAL_TIPPED_METADATA_FIELDS
    elif disposition == "RESTORATION_PROBE_TIPPED" or (
        metadata.get("stage_reached") == "RESTORATION_PROBE"
        and disposition == "UNRESOLVED_STATE_FAILURE"
    ):
        fields = C.PROBE_TIPPED_METADATA_FIELDS
    else:
        fields = C.TEACHER_TERMINAL_METADATA_FIELDS
    row = _mapping(metadata, fields, "qualification material metadata")
    C.validate_content_digest(row)
    if (
        row["schema"] != C.QUALIFICATION_MATERIAL_SCHEMA
        or row["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material identity drift"
        )
    _commit(row["source_freeze_commit"], "qualification source freeze commit")
    _sha(
        row["runtime_contract_content_digest"],
        "qualification runtime contract digest",
    )
    index = _nonnegative_int(row["pool_index"], "qualification candidate index")
    if index >= C.MAX_CANDIDATE_COUNT or (
        expected_pool_index is not None and index != expected_pool_index
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material candidate index drift"
        )
    family, stratum, attempt = _candidate_coordinates(index)
    spec = C.build_candidate_spec(family, stratum, attempt)
    if (
        row["candidate_spec"] != spec
        or row["candidate_spec_sha256"] != spec["canonical_spec_sha256"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material candidate identity drift"
        )
    state = validate_state_disposition_record(
        row["state_disposition"], expected_pool_index=index
    )
    if any(
        row[field] != state[field]
        for field in (
            "disposition",
            "qualified",
            "stage_reached",
            "executable_snapshot_exists",
            "teacher_executed",
            "reset_or_candidate_outcome_opened",
        )
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "outer/state disposition cross-link drift"
        )
    if row["stage_reached"] in {"INITIAL_BOUNDARY", "RESTORATION_PROBE"} and (
        row["qualified"] is not False
        or row["rejection_reason"] != row["disposition"]
        or row["rejection_components"] != [row["disposition"]]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "pre-teacher rejection projection drift"
        )
    persisted = validate_persisted_array_evidence(
        row["persisted_array_evidence"], reopened_arrays=reopened_arrays
    )
    expected_shard_id = (
        f"qualification/{spec['stream_id']}/attempt-{attempt:02d}"
    )
    if (
        persisted["shard_kind"] != "qualification_terminal_material"
        or persisted["shard_id"] != expected_shard_id
        or persisted["payload_file"]["path"]
        != f"{expected_shard_id}/payload.npz"
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification persisted shard identity/path drift"
        )
    if row["payload"] != {
        "role": "material_shard_payload",
        **persisted["payload_file"],
        "kind": "npz",
    }:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material payload binding drift"
        )
    validate_state_material_payload(
        state, persisted, reopened_arrays=reopened_arrays
    )
    if disposition == "INITIAL_BOUNDARY_TIPPED" or row["stage_reached"] == "INITIAL_BOUNDARY":
        boundary = _mapping(
            row["boundary_evidence"], C.BOUNDARY_EVIDENCE_FIELDS,
            "boundary evidence",
        )
        expected_boundary = {
            "intended_pose_representation": "xyz_plus_quaternion_xyzw",
            "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
            "available_simulator_diagnostics": sorted(
                C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY
            ),
            "previous_applied_command_sha256": hashlib.sha256(
                np.asarray(reopened_arrays["previous_applied_command"]).tobytes(
                    order="C"
                )
            ).hexdigest(),
            "previous_applied_command_dtype": "<f8",
            "previous_applied_command_shape": [3],
        }
        if boundary != expected_boundary:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "boundary evidence/raw array drift"
            )
        if row["snapshot"] is not None or row["graph"] is not None or row["teacher"] is not None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial rejection contains fabricated executable evidence"
            )
    else:
        completed = row["stage_reached"] != "RESTORATION_PROBE"
        try:
            probe = _V4M._validate_probe_material(
                row, reopened_arrays, completed=completed
            )
        except Exception as exc:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                str(exc)
            ) from exc
        if probe["identity"] != state["snapshot_identity"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "probe/state snapshot identity drift"
            )
        if (
            [trial["termination_flags"] for trial in probe["trials"]]
            != state["probe_trial_termination_flags"]
            or [trial["tip_sample_index"] for trial in probe["trials"]]
            != state["probe_tip_sample_indices"]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "probe/state terminal evidence drift"
            )
        snapshot = row["snapshot"]
        artifact_sha = probe["identity"]["artifact_file_sha256"]
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("snapshot_payload_sha256") != artifact_sha
            or row["initial_decision_state_sha256"] != artifact_sha
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "initial snapshot transport binding drift"
            )
        validate_snapshot_previous_applied_command_binding(
            snapshot, persisted, reopened_arrays=reopened_arrays
        )
        rgb = np.asarray(reopened_arrays.get("rgb"))
        if (
            rgb.dtype.str != "|u1"
            or list(rgb.shape) != [168, 224, 3]
            or row["current_rgb_sha256"] != _canonical_array_sha256(rgb)
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "qualification RGB binding drift"
            )
        if not completed:
            if row["graph"] is not None or row["teacher"] is not None:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "probe rejection contains teacher evidence"
                )
        else:
            teacher = row["teacher"]
            if not isinstance(teacher, Mapping) or not isinstance(row["graph"], Mapping):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher material evidence absent"
                )
            teacher = _mapping(
                teacher,
                C.V4_TEACHER_MATERIAL_SUMMARY_FIELDS,
                "teacher material summary",
            )
            trace_names = [
                f"teacher__{name}" for name in C.TEACHER_TRACE_MEMBER_ORDER
            ]
            if any(name not in reopened_arrays for name in trace_names):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher raw trace member absent"
                )
            digests = teacher.get("trace_digests")
            if (
                not isinstance(digests, Mapping)
                or set(digests) != set(C.TEACHER_TRACE_MEMBER_ORDER)
                or any(
                    digests[name]
                    != _canonical_array_sha256(reopened_arrays[f"teacher__{name}"])
                    for name in C.TEACHER_TRACE_MEMBER_ORDER
                )
            ):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher trace digest drift"
                )
            reduced = reduce_qualification_teacher_trace(
                row, reopened_arrays, expected_pool_index=index
            )
            expected_teacher = {
                "sample_count": reduced["sample_count"],
                "trace_digests": copy.deepcopy(dict(digests)),
                "contact_free": reduced["contact_free"],
                "left_source_region": reduced["left_source_region"],
                "positive_route_progress": reduced["positive_route_progress"],
                "route_progress_m": reduced["route_progress_m"],
                "competing_port_entered": reduced["competing_port_entered"],
                "competing_crossing": reduced["competing_crossing"],
                "reached_target_node": reduced["reached_target_node"],
                "crossing": reduced["crossing"],
                "crossing_error": reduced["crossing_error"],
                "termination_flags": reduced["termination_flags"],
                "terminated_unsafe": any(reduced["termination_flags"].values()),
                "teacher_valid": reduced["teacher_valid"],
            }
            if C.canonical_json_bytes(teacher) != C.canonical_json_bytes(expected_teacher):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher summary differs from independent raw reduction"
                )
            components: list[str] = []
            if any(reduced["termination_flags"].values()):
                components.append("TEACHER_TERMINATED_UNSAFELY")
            if not reduced["contact_free"]:
                components.append("TEACHER_PHYSICS_CONTACT")
            if reduced["crossing"] is None or reduced["competing_port_entered"]:
                components.append("TEACHER_CROSSING_INVALID")
            if not reduced["left_source_region"]:
                components.append("TEACHER_DID_NOT_LEAVE_SOURCE")
            if not reduced["positive_route_progress"]:
                components.append("TEACHER_NO_POSITIVE_PROGRESS")
            if not reduced["goal_reachable"] or not reduced["graph_edge_physically_executable"]:
                components.append("UNRESOLVED_STATE_FAILURE")
            expected_disposition = "QUALIFIED" if not components else components[0]
            if (
                row["goal_reachable"] is not reduced["goal_reachable"]
                or row["graph_edge_physically_executable"]
                is not reduced["graph_edge_physically_executable"]
                or row["disposition"] != expected_disposition
                or row["qualified"] is not (expected_disposition == "QUALIFIED")
                or row["rejection_reason"]
                != (None if expected_disposition == "QUALIFIED" else expected_disposition)
                or row["rejection_components"] != components
                or row["stage_reached"]
                != ("COMPLETE" if expected_disposition == "QUALIFIED" else "TEACHER_EXECUTION")
            ):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher outer disposition differs from raw reduction"
                )
            expected_contact = {
                "api": "robot.get_contacts",
                "sample_period_s": C.TEACHER_TRACE_DT_S,
                "forbidden_net_force_api_used": False,
                "ontology_sha256": C.CONTACT_AUTHORITY["ontology_sha256"],
            }
            if row["contact_instrumentation"] != expected_contact:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "teacher contact instrumentation drift"
                )
    return {
        "metadata": row,
        "arrays": {name: reopened_arrays[name] for name in reopened_arrays},
    }


def _file_binding(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.MATERIAL_FILE_BINDING_FIELDS, label)
    path = row["path"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or ".." in path.split("/")
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} path drift"
        )
    if (
        not isinstance(row["bytes"], int)
        or isinstance(row["bytes"], bool)
        or row["bytes"] <= 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} byte count drift"
        )
    _sha(row["sha256"], f"{label}.sha256")
    if (
        row["nlink"] != 1
        or row["ordinary_regular_file"] is not True
        or row["resolved_path_ancestor_symlink_count"] != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} ordinary-file/nonreuse drift"
        )
    return row


def build_generator_terminal_record(
    state_disposition: Mapping[str, Any], *, source_freeze_commit: str,
    runtime_contract_content_digest: str,
    material_metadata_binding: Mapping[str, Any],
    material_payload_binding: Mapping[str, Any],
    persisted_array_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind one independently validated physical terminal to its stream."""

    state = validate_state_disposition_record(state_disposition)
    index = _nonnegative_int(state["pool_index"], "candidate_index")
    family, stratum, attempt = _candidate_coordinates(index)
    spec = C.build_candidate_spec(family, stratum, attempt)
    if any(
        state[field] != spec[field]
        for field in (
            "candidate_spec_id",
            "family",
            "stratum_index",
            "variant_index",
            "procedural_seed",
            "canonical_spec_sha256",
        )
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal state/candidate cross-link drift"
        )
    metadata_binding = _file_binding(
        material_metadata_binding, "material metadata binding"
    )
    payload_binding = _file_binding(
        material_payload_binding, "material payload binding"
    )
    shard_prefix = f"qualification/{spec['stream_id']}/attempt-{attempt:02d}"
    if (
        metadata_binding["path"] != f"{shard_prefix}/metadata.json"
        or payload_binding["path"] != f"{shard_prefix}/payload.npz"
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal material binding path drift"
        )
    return C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_terminal_record.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "candidate_index": index,
            "stream_index": spec["stream_index"],
            "stream_id": spec["stream_id"],
            "family": family,
            "stratum_index": stratum,
            "attempt_index": attempt,
            "candidate_spec_id": spec["candidate_spec_id"],
            "canonical_spec_sha256": spec["canonical_spec_sha256"],
            "procedural_seed": spec["procedural_seed"],
            "disposition": state["disposition"],
            "qualified": state["qualified"],
            "hard_stop": state["hard_stop"],
            "continuation_authorized": state["continuation_authorized"],
            "stage_reached": state["stage_reached"],
            "source_freeze_commit": _commit(
                source_freeze_commit, "source_freeze_commit"
            ),
            "runtime_contract_content_digest": _sha(
                runtime_contract_content_digest,
                "runtime_contract_content_digest",
            ),
            "state_disposition": state,
            "material_metadata_binding": metadata_binding,
            "material_payload_binding": payload_binding,
            "persisted_array_evidence_sha256": _sha(
                persisted_array_evidence_sha256,
                "persisted_array_evidence_sha256",
            ),
        }
    )


def validate_generator_terminal_record(
    value: Any, *, expected_candidate_index: int | None = None,
    source_freeze_commit: str | None = None,
    runtime_contract_content_digest: str | None = None,
) -> dict[str, Any]:
    row = _mapping(
        value, C.GENERATOR_TERMINAL_RECORD_FIELDS, "generator terminal record"
    )
    C.validate_content_digest(row)
    index = _nonnegative_int(row["candidate_index"], "candidate_index")
    if expected_candidate_index is not None and index != expected_candidate_index:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "expected candidate index drift"
        )
    state = validate_state_disposition_record(
        row["state_disposition"], expected_pool_index=index
    )
    expected = build_generator_terminal_record(
        state,
        source_freeze_commit=row["source_freeze_commit"],
        runtime_contract_content_digest=row[
            "runtime_contract_content_digest"
        ],
        material_metadata_binding=row["material_metadata_binding"],
        material_payload_binding=row["material_payload_binding"],
        persisted_array_evidence_sha256=row[
            "persisted_array_evidence_sha256"
        ],
    )
    if source_freeze_commit is not None and row["source_freeze_commit"] != (
        source_freeze_commit
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal source freeze cross-link drift"
        )
    if (
        runtime_contract_content_digest is not None
        and row["runtime_contract_content_digest"]
        != runtime_contract_content_digest
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal runtime contract cross-link drift"
        )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator terminal record drift"
        )
    return row


def _validate_complete_terminal_population(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    supplied = _sequence(records, "generator terminal records")
    rows = [validate_generator_terminal_record(item) for item in supplied]
    if not rows or len(rows) > C.MAX_CANDIDATE_COUNT:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal record cardinality drift"
        )
    candidate_indices = [row["candidate_index"] for row in rows]
    if candidate_indices != sorted(set(candidate_indices)):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal records are not unique candidate-index order"
        )
    sources = {row["source_freeze_commit"] for row in rows}
    runtimes = {row["runtime_contract_content_digest"] for row in rows}
    if len(sources) != 1 or len(runtimes) != 1:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal source/runtime changed across candidates"
        )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["stream_index"]].append(row)
        if row["hard_stop"] or not row["continuation_authorized"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1HardStop(
                f"{row['disposition']}: generator terminal population hard stop"
            )
    if set(grouped) != set(range(C.STREAM_COUNT)):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "not every generator stream has a terminal record"
        )
    for stream, stream_rows in grouped.items():
        attempts = [row["attempt_index"] for row in stream_rows]
        if attempts != list(range(len(stream_rows))):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"stream {stream} attempt sequence drift"
            )
        qualified = [row for row in stream_rows if row["qualified"]]
        if len(qualified) > C.TARGET_QUALIFIED_PER_STREAM:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"stream {stream} continued after qualification target"
            )
        if len(qualified) == C.TARGET_QUALIFIED_PER_STREAM:
            if stream_rows[-1]["qualified"] is not True:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"stream {stream} continued beyond fourth qualified state"
                )
        elif len(stream_rows) != C.MAX_ATTEMPTS_PER_STREAM:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"stream {stream} stopped before attempt 63"
            )
    return rows, dict(grouped)


def build_generator_terminal_records_jsonl(
    records: Sequence[Mapping[str, Any]],
) -> bytes:
    rows, _ = _validate_complete_terminal_population(records)
    return b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def validate_generator_terminal_records_jsonl(
    value: bytes | str,
) -> list[dict[str, Any]]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    if (
        not isinstance(raw, bytes)
        or not raw.endswith(b"\n")
        or b"\r" in raw
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator terminal JSONL byte format drift"
        )
    try:
        parsed = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator terminal JSONL parse failed"
        ) from exc
    if raw != build_generator_terminal_records_jsonl(parsed):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator terminal JSONL is not canonical"
        )
    return [copy.deepcopy(row) for row in parsed]


def _full_disposition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    observed = Counter(str(item["disposition"]) for item in rows)
    if set(observed) - set(C.STATE_DISPOSITIONS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "unknown disposition in generator population"
        )
    return {name: int(observed.get(name, 0)) for name in C.STATE_DISPOSITIONS}


def _candidate_spec_for_terminal(row: Mapping[str, Any]) -> dict[str, Any]:
    return C.build_candidate_spec(
        str(row["family"]), int(row["stratum_index"]), int(row["attempt_index"])
    )


def _physical_parameter_projection(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": spec["family"],
        "stratum_index": spec["stratum_index"],
        "edge_direction": spec["route_direction"],
        "opening_width_id": spec["passage_width_id"],
        "opening_width_m": spec["passage_width_m"],
        "port_offset_id": spec["port_distance_id"],
        "port_offset_m": spec["port_distance_m"],
        "base_spawn_lateral_offset_m": spec["spawn_lateral_offset_m"],
        "starting_bearing_rad": spec["spawn_yaw_offset_rad"],
        "geometry_jitter_x_m": spec["geometry_jitter_xy_m"][0],
        "geometry_jitter_y_m": spec["geometry_jitter_xy_m"][1],
        **{
            name: spec["variant_adjustments"][name]
            for name in C.HASH_PERTURBATION_PARAMETER_IDS[2:]
        },
    }


def _common_physical_parameters(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "common physical parameter stream is empty"
        )
    first = _candidate_spec_for_terminal(rows[0])
    failed = [item for item in rows if not bool(item["qualified"])]
    projections = [
        _physical_parameter_projection(_candidate_spec_for_terminal(item))
        for item in failed
    ]
    counts: dict[str, list[dict[str, Any]]] = {}
    constants: dict[str, Any] = {}
    for name in C.COMMON_FAILURE_PARAMETER_IDS:
        grouped: dict[bytes, dict[str, Any]] = {}
        for projection in projections:
            value = copy.deepcopy(projection[name])
            key = C.canonical_json_bytes(value)
            if key not in grouped:
                grouped[key] = {"value": value, "count": 0}
            grouped[key]["count"] += 1
        counts[name] = [grouped[key] for key in sorted(grouped)]
        constants[name] = (
            copy.deepcopy(counts[name][0]["value"])
            if len(counts[name]) == 1
            else None
        )
    result = {
        "family": first["family"],
        "stratum_index": first["stratum_index"],
        "failed_attempt_count": len(failed),
        "parameter_order": list(C.COMMON_FAILURE_PARAMETER_IDS),
        "canonical_value_counts": counts,
        "constant_across_all_failed_attempts": constants,
        "predeclared_parameters_only": True,
        "post_hoc_factor_added": False,
    }
    if set(result) != set(C.COMMON_PHYSICAL_PARAMETER_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "common physical parameter field drift"
        )
    for item in rows:
        spec = _candidate_spec_for_terminal(item)
        if (
            spec["family"] != first["family"]
            or spec["stratum_index"] != first["stratum_index"]
            or spec["geometry"]["selected_directed_edge"]["opening_width_m"]
            != spec["passage_width_m"]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "stream does not share its frozen physical parameters"
            )
    return result


def _hash_perturbation_ranges(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {
        name: [] for name in C.HASH_PERTURBATION_PARAMETER_IDS
    }
    for row in rows:
        spec = _candidate_spec_for_terminal(row)
        values["geometry_jitter_x_m"].append(
            float(spec["geometry_jitter_xy_m"][0])
        )
        values["geometry_jitter_y_m"].append(
            float(spec["geometry_jitter_xy_m"][1])
        )
        for name in C.HASH_PERTURBATION_PARAMETER_IDS[2:]:
            values[name].append(float(spec["variant_adjustments"][name]))
    return {
        name: {"minimum": min(values[name]), "maximum": max(values[name])}
        for name in C.HASH_PERTURBATION_PARAMETER_IDS
    }


def _stream_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    qualified = [item for item in rows if item["qualified"]]
    target = len(qualified) == C.TARGET_QUALIFIED_PER_STREAM
    selected = min(
        qualified,
        key=lambda item: (item["canonical_spec_sha256"], item["candidate_spec_id"]),
    ) if target else None
    dispositions = _full_disposition_counts(rows)
    disposition_rates = {
        name: count / len(rows) for name, count in dispositions.items()
    }
    nonqualified_count = len(rows) - len(qualified)
    initial_valid = sum(
        not any(item["state_disposition"]["initial_termination_flags"].values())
        for item in rows
    )
    teacher_rows = [
        item for item in rows if item["state_disposition"]["teacher_executed"]
    ]
    teacher_left_source = sum(
        bool(item["state_disposition"]["teacher_criteria"]["teacher_left_source_region"])
        for item in teacher_rows
    )
    correct_port_ever = sum(
        bool(item["state_disposition"]["teacher_criteria"]["teacher_crossed_directed_port"])
        for item in teacher_rows
    )
    contact_count = dispositions["TEACHER_PHYSICS_CONTACT"]
    other_rejection_counts = [
        count
        for disposition, count in dispositions.items()
        if disposition not in {"QUALIFIED", "TEACHER_PHYSICS_CONTACT"}
    ]
    contact_dominant = (
        contact_count > 0
        and all(contact_count > count for count in other_rejection_counts)
    )
    qualification_ordinals = [
        offset + 1 for offset, item in enumerate(rows) if item["qualified"]
    ]
    return {
        "stream_index": first["stream_index"],
        "stream_id": first["stream_id"],
        "family": first["family"],
        "stratum_index": first["stratum_index"],
        "attempt_count": len(rows),
        "attempt_indices": [item["attempt_index"] for item in rows],
        "candidate_indices": [item["candidate_index"] for item in rows],
        "qualified_count": len(qualified),
        "nonqualified_count": nonqualified_count,
        "qualified_candidate_indices": [
            item["candidate_index"] for item in qualified
        ],
        "disposition_counts": dispositions,
        "disposition_rates": disposition_rates,
        "attempts_to_first_qualified": (
            qualification_ordinals[0] if qualification_ordinals else None
        ),
        "attempts_to_fourth_qualified": (
            qualification_ordinals[3]
            if len(qualification_ordinals) >= C.TARGET_QUALIFIED_PER_STREAM
            else None
        ),
        "valid_initial_state_count": initial_valid,
        "valid_initial_state_fraction": initial_valid / len(rows),
        "teacher_executed_count": len(teacher_rows),
        "teacher_left_source_count": teacher_left_source,
        "teacher_left_source_fraction_of_teacher_executed": (
            teacher_left_source / len(teacher_rows) if teacher_rows else 0.0
        ),
        "correct_port_ever_count": correct_port_ever,
        "correct_port_ever_fraction_of_teacher_executed": (
            correct_port_ever / len(teacher_rows) if teacher_rows else 0.0
        ),
        "contact_rejection_count": contact_count,
        "contact_rejection_fraction": (
            contact_count / nonqualified_count if nonqualified_count else 0.0
        ),
        "contact_dominant": contact_dominant,
        "common_physical_parameters": _common_physical_parameters(rows),
        "observed_hash_perturbation_ranges": _hash_perturbation_ranges(rows),
        "termination_reason": (
            "TARGET_QUALIFIED_REACHED" if target else "ATTEMPT_LIMIT_REACHED"
        ),
        "target_reached": target,
        "yield_fraction": len(qualified) / len(rows),
        "qualification_rate": len(qualified) / len(rows),
        "rejection_rate": nonqualified_count / len(rows),
        "selected_candidate_index": (
            None if selected is None else selected["candidate_index"]
        ),
        "selected_candidate_spec_id": (
            None if selected is None else selected["candidate_spec_id"]
        ),
    }


def _aggregate_counts(
    metric_rows: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = len(terminal_rows)
    qualified = sum(bool(item["qualified"]) for item in terminal_rows)
    dispositions = _full_disposition_counts(terminal_rows)
    return {
        "stream_count": len(metric_rows),
        "attempt_count": attempts,
        "qualified_count": qualified,
        "nonqualified_count": attempts - qualified,
        "qualification_rate": qualified / attempts,
        "rejection_rate": (attempts - qualified) / attempts,
        "disposition_counts": dispositions,
        "disposition_rates": {
            name: count / attempts for name, count in dispositions.items()
        },
        "zero_yield_stream_count": sum(
            item["qualified_count"] == 0 for item in metric_rows
        ),
        "low_yield_stream_count": sum(
            0 < item["qualified_count"] < C.TARGET_QUALIFIED_PER_STREAM
            for item in metric_rows
        ),
        "target_reached_stream_count": sum(
            bool(item["target_reached"]) for item in metric_rows
        ),
        "valid_initial_state_count": sum(
            int(item["valid_initial_state_count"]) for item in metric_rows
        ),
        "teacher_executed_count": sum(
            int(item["teacher_executed_count"]) for item in metric_rows
        ),
        "teacher_left_source_count": sum(
            int(item["teacher_left_source_count"]) for item in metric_rows
        ),
        "correct_port_ever_count": sum(
            int(item["correct_port_ever_count"]) for item in metric_rows
        ),
        "contact_rejection_count": sum(
            int(item["contact_rejection_count"]) for item in metric_rows
        ),
        "contact_dominant_stream_count": sum(
            bool(item["contact_dominant"]) for item in metric_rows
        ),
    }


def _v4_shortfall_resolution(
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve the twelve persisted V4 shortfalls from successor stream yields."""

    by_key = {
        (str(row["family"]), int(row["stratum_index"])): row
        for row in metric_rows
    }
    expected_keys = set(C.V4_CANONICAL_SHORTFALL_STREAMS)
    if not expected_keys <= set(by_key):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "canonical V4 shortfall stream is absent"
        )
    rows: list[dict[str, Any]] = []
    for family, stratum in C.V4_CANONICAL_SHORTFALL_STREAMS:
        stream = by_key[(family, stratum)]
        qualified = int(stream["qualified_count"])
        if qualified >= C.TARGET_QUALIFIED_PER_STREAM:
            status = "RESOLVED_AT_FOUR_QUALIFIED"
        elif qualified > 0:
            status = "PARTIAL_YIELD_BELOW_FOUR"
        else:
            status = "ZERO_YIELD_AT_ATTEMPT_LIMIT"
        item = {
            "family": family,
            "stratum_index": stratum,
            "v4_fixed_four_shortfall": True,
            "successor_attempt_count": int(stream["attempt_count"]),
            "successor_qualified_count": qualified,
            "attempts_to_first_qualified": stream["attempts_to_first_qualified"],
            "attempts_to_fourth_qualified": stream[
                "attempts_to_fourth_qualified"
            ],
            "successor_target_reached": bool(stream["target_reached"]),
            "resolution_status": status,
        }
        _mapping(
            item,
            C.V4_SHORTFALL_RESOLUTION_ROW_FIELDS,
            "V4 shortfall resolution row",
        )
        rows.append(item)
    resolved = sum(
        row["resolution_status"] == "RESOLVED_AT_FOUR_QUALIFIED"
        for row in rows
    )
    partial = sum(
        row["resolution_status"] == "PARTIAL_YIELD_BELOW_FOUR"
        for row in rows
    )
    zero = sum(
        row["resolution_status"] == "ZERO_YIELD_AT_ATTEMPT_LIMIT"
        for row in rows
    )
    if zero:
        conclusion = C.GENERATOR_FEASIBILITY_NO_GO
    elif partial:
        conclusion = C.V4_SHORTFALL_INSUFFICIENT
    else:
        conclusion = C.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
    result = {
        "authority_content_digest": C.V4_SHORTFALL_RESOLUTION_AUTHORITY[
            "content_digest"
        ],
        "canonical_v4_shortfall_stream_count": len(rows),
        "stream_rows": rows,
        "resolved_stream_count": resolved,
        "partial_stream_count": partial,
        "zero_yield_stream_count": zero,
        "all_canonical_v4_shortfalls_resolved": resolved == len(rows),
        "conclusion": conclusion,
        "supports_shallow_sampling_as_primary_cause": (
            conclusion == C.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
        ),
        "intrinsic_infeasibility_established": False,
    }
    _mapping(
        result,
        C.V4_SHORTFALL_RESOLUTION_FIELDS,
        "V4 shortfall resolution",
    )
    return result


def _factor_level(stream: Mapping[str, Any], factor: str) -> Any:
    spec = C.build_candidate_spec(
        str(stream["family"]), int(stream["stratum_index"]), 0
    )
    keys = {
        "family": "family",
        "edge_direction": "route_direction",
        "opening_width_id": "passage_width_id",
        "opening_width_m": "passage_width_m",
        "port_offset_id": "port_distance_id",
        "port_offset_m": "port_distance_m",
        "starting_bearing_rad": "spawn_yaw_offset_rad",
        "base_spawn_lateral_offset_m": "spawn_lateral_offset_m",
    }
    return copy.deepcopy(spec[keys[factor]])


def select_failure_next_decision(failure_families: Sequence[str]) -> str:
    """Select one exact generator-revision literal from observed shortfalls."""

    supplied = _sequence(failure_families, "failure families")
    ordered = [
        family for family in C.FAMILY_IDS if family in set(supplied)
    ]
    if supplied != ordered or not ordered:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "failure family projection drift"
        )
    if ordered == ["TURNING_JUNCTION"]:
        return C.TURNING_JUNCTION_GENERATOR_NEXT_DECISION
    if ordered == ["OFFSET_OPENING"]:
        return C.OFFSET_OPENING_GENERATOR_NEXT_DECISION
    return C.PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION


def build_generator_metrics(
    records: Sequence[Mapping[str, Any]], *, downstream_outcomes_opened: int = 0
) -> dict[str, Any]:
    rows, grouped = _validate_complete_terminal_population(records)
    if (
        not isinstance(downstream_outcomes_opened, int)
        or isinstance(downstream_outcomes_opened, bool)
        or downstream_outcomes_opened != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator gate was built after downstream outcomes opened"
        )
    streams = [_stream_metric(grouped[index]) for index in range(C.STREAM_COUNT)]
    for stream in streams:
        _mapping(stream, C.GENERATOR_STREAM_METRIC_FIELDS, "stream metric")
    zero = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if row["qualified_count"] == 0
    ]
    low = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if 0 < row["qualified_count"] < C.TARGET_QUALIFIED_PER_STREAM
    ]
    reached = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if row["target_reached"]
    ]
    if zero:
        status = C.GENERATOR_FEASIBILITY_NO_GO
    elif low:
        status = C.GENERATOR_LOW_YIELD
    else:
        status = C.GENERATOR_PANEL_AVAILABLE
    selected_indices = (
        [int(row["selected_candidate_index"]) for row in streams]
        if status == C.GENERATOR_PANEL_AVAILABLE
        else []
    )
    selected_ids = (
        [str(row["selected_candidate_spec_id"]) for row in streams]
        if status == C.GENERATOR_PANEL_AVAILABLE
        else []
    )
    families = []
    for family in C.FAMILY_IDS:
        family_streams = [row for row in streams if row["family"] == family]
        family_records = [row for row in rows if row["family"] == family]
        aggregate = _aggregate_counts(family_streams, family_records)
        result = {
            "family": family,
            **aggregate,
            "minimum_stream_qualified_count": min(
                row["qualified_count"] for row in family_streams
            ),
        }
        _mapping(result, C.GENERATOR_FAMILY_METRIC_FIELDS, "family metric")
        families.append(result)
    factors: list[dict[str, Any]] = []
    for factor in C.STRATUM_FACTOR_IDS:
        levels: list[Any] = []
        for stream in streams:
            level = _factor_level(stream, factor)
            if level not in levels:
                levels.append(level)
        for level in levels:
            factor_streams = [
                stream
                for stream in streams
                if _factor_level(stream, factor) == level
            ]
            stream_indices = {item["stream_index"] for item in factor_streams}
            factor_records = [
                item for item in rows if item["stream_index"] in stream_indices
            ]
            factor_row = {
                "factor": factor,
                "level": copy.deepcopy(level),
                **_aggregate_counts(factor_streams, factor_records),
            }
            _mapping(
                factor_row,
                C.GENERATOR_FACTOR_METRIC_FIELDS,
                "stratum factor metric",
            )
            factors.append(factor_row)
    below_target = zero + low
    failure_families = [
        family
        for family in C.FAMILY_IDS
        if any(item["family"] == family for item in below_target)
    ]
    if status == C.GENERATOR_PANEL_AVAILABLE:
        next_decision = None
    else:
        next_decision = select_failure_next_decision(failure_families)
    failure_stream_rows = [
        stream
        for stream in streams
        if stream["qualified_count"] < C.TARGET_QUALIFIED_PER_STREAM
    ]
    next_decision_evidence = {
        "below_target_stream_count": len(failure_stream_rows),
        "below_target_families": failure_families,
        "failure_streams": [
            {
                "family": stream["family"],
                "stratum_index": stream["stratum_index"],
                "qualified_count": stream["qualified_count"],
                "nonqualified_count": stream["nonqualified_count"],
                "disposition_counts": stream["disposition_counts"],
                "disposition_rates": stream["disposition_rates"],
                "valid_initial_state_count": stream[
                    "valid_initial_state_count"
                ],
                "valid_initial_state_ever": bool(
                    stream["valid_initial_state_count"]
                ),
                "teacher_left_source_count": stream[
                    "teacher_left_source_count"
                ],
                "teacher_left_source_ever": bool(
                    stream["teacher_left_source_count"]
                ),
                "correct_port_ever_count": stream[
                    "correct_port_ever_count"
                ],
                "correct_port_ever": bool(stream["correct_port_ever_count"]),
                "contact_rejection_count": stream[
                    "contact_rejection_count"
                ],
                "contact_rejection_fraction": stream[
                    "contact_rejection_fraction"
                ],
                "contact_dominant": stream["contact_dominant"],
                "common_physical_parameters": stream[
                    "common_physical_parameters"
                ],
            }
            for stream in failure_stream_rows
        ],
        "teacher_or_policy_change_inferred": False,
    }
    qualified_count = sum(bool(item["qualified"]) for item in rows)
    nonqualified_count = len(rows) - qualified_count
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_metrics.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "v4_context_authority_content_digest": C.V4_CONTEXT_AUTHORITY[
                "content_digest"
            ],
            "stream_manifest_content_digest": C.STREAM_MANIFEST_CONTENT_DIGEST,
            "allocation_authority_content_digest": C.GENERATOR_ALLOCATION_AUTHORITY[
                "content_digest"
            ],
            "terminal_record_count": len(rows),
            "terminal_unique_identity_count": len(
                {row["candidate_spec_id"] for row in rows}
            ),
            "maximum_candidate_count": C.MAX_CANDIDATE_COUNT,
            "expected_stream_count": C.STREAM_COUNT,
            "completed_stream_count": len(grouped),
            "missing_stream_count": C.STREAM_COUNT - len(grouped),
            "duplicated_candidate_identity_count": (
                len(rows) - len({row["candidate_spec_id"] for row in rows})
            ),
            "stream_count": C.STREAM_COUNT,
            "maximum_attempts_per_stream": C.MAX_ATTEMPTS_PER_STREAM,
            "target_qualified_per_stream": C.TARGET_QUALIFIED_PER_STREAM,
            "qualified_count": qualified_count,
            "nonqualified_count": nonqualified_count,
            "hard_stop_count": 0,
            "teacher_execution_count": sum(
                bool(row["state_disposition"]["teacher_executed"])
                for row in rows
            ),
            "disposition_counts": _full_disposition_counts(rows),
            "disposition_rates": {
                name: count / len(rows)
                for name, count in _full_disposition_counts(rows).items()
            },
            "stream_rows": streams,
            "family_rows": families,
            "stratum_factor_rows": factors,
            "v4_shortfall_resolution": _v4_shortfall_resolution(streams),
            "qualification_rate": qualified_count / len(rows),
            "rejection_rate": nonqualified_count / len(rows),
            "zero_yield_streams": zero,
            "low_yield_streams": low,
            "target_reached_streams": reached,
            "selected_candidate_indices": selected_indices,
            "selected_candidate_spec_ids": selected_ids,
            "status": status,
            "primary_classification": status,
            "next_decision": next_decision,
            "next_decision_evidence": next_decision_evidence,
            "downstream_scientific_execution_authorized": (
                status == C.GENERATOR_PANEL_AVAILABLE
            ),
            "downstream_outcomes_opened": 0,
            "models_trained": 0,
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )
    if set(result) != set(C.GENERATOR_METRICS_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator metrics output field drift"
        )
    return result


def validate_generator_metrics(
    value: Any,
    records: Sequence[Mapping[str, Any]],
    stream_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.GENERATOR_METRICS_FIELDS, "generator metrics")
    C.validate_content_digest(row)
    if stream_manifest is not None:
        manifest = C.validate_generator_stream_manifest(stream_manifest)
        if manifest["content_digest"] != row["stream_manifest_content_digest"]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "generator metric/stream manifest binding drift"
            )
    expected = build_generator_metrics(records)
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator metrics drift"
        )
    return row


def build_panel_handoff(
    records: Sequence[Mapping[str, Any]],
    generator_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows, _ = _validate_complete_terminal_population(records)
    metrics = (
        build_generator_metrics(rows)
        if generator_metrics is None
        else validate_generator_metrics(generator_metrics, rows)
    )
    if metrics["status"] != C.GENERATOR_PANEL_AVAILABLE:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "V4 downstream panel handoff requires generator availability"
        )
    by_index = {row["candidate_index"]: row for row in rows}
    selected_rows = [by_index[index] for index in metrics["selected_candidate_indices"]]
    if any(
        not row["qualified"]
        or not row["state_disposition"]["teacher_executed"]
        or not row["state_disposition"]["executable_snapshot_exists"]
        for row in selected_rows
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected handoff contains no executable qualified teacher evidence"
        )
    specs = []
    for row in selected_rows:
        family, stratum, attempt = _candidate_coordinates(row["candidate_index"])
        specs.append(C.build_candidate_spec(family, stratum, attempt))
    family_counts = dict(
        sorted(Counter(row["family"] for row in selected_rows).items())
    )
    stratum_counts = [
        {
            "family": family,
            "stratum_index": stratum,
            "selected_count": sum(
                row["family"] == family and row["stratum_index"] == stratum
                for row in selected_rows
            ),
        }
        for family in C.FAMILY_IDS
        for stratum in range(C.STRATA_PER_FAMILY)
    ]
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "panel_handoff.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "generator_metrics_content_digest": metrics["content_digest"],
            "v4_context_authority_content_digest": C.V4_CONTEXT_AUTHORITY[
                "content_digest"
            ],
            "selected_state_count": len(selected_rows),
            "selected_candidate_indices": [
                row["candidate_index"] for row in selected_rows
            ],
            "selected_candidate_spec_ids": [
                row["candidate_spec_id"] for row in selected_rows
            ],
            "selected_candidate_specs": specs,
            "selected_terminal_records": selected_rows,
            "family_selected_counts": family_counts,
            "stratum_selected_counts": stratum_counts,
            "selection_rule": C.GENERATOR_ALLOCATION_AUTHORITY[
                "panel_selection"
            ],
            "v4_downstream_science_authorized": True,
        }
    )
    if set(result) != set(C.PANEL_HANDOFF_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel handoff output field drift"
        )
    if (
        len(selected_rows) != C.PANEL_STATE_COUNT
        or family_counts != {family: 16 for family in C.FAMILY_IDS}
        or any(row["selected_count"] != 1 for row in stratum_counts)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel handoff balance drift"
        )
    return result


def validate_panel_handoff(
    value: Any, records: Sequence[Mapping[str, Any]],
    generator_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, C.PANEL_HANDOFF_FIELDS, "panel handoff")
    C.validate_content_digest(row)
    expected = build_panel_handoff(records, generator_metrics)
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel handoff drift"
        )
    return row


def _canonical_array_sha256_local(value: Any) -> str:
    return _canonical_array_sha256(value)


def _validate_successor_payload_binding(
    metadata: Mapping[str, Any],
    persisted: Mapping[str, Any],
    *,
    shard_kind: str,
    shard_id: str,
) -> None:
    if (
        persisted["shard_kind"] != shard_kind
        or persisted["shard_id"] != shard_id
        or persisted["payload_file"]["path"] != f"{shard_id}/payload.npz"
        or metadata["payload"]
        != {
            "role": "material_shard_payload",
            **persisted["payload_file"],
            "kind": "npz",
        }
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material persisted shard identity/path drift"
        )


def _build_reset_pair_comparison(
    state_id: str,
    traces: Sequence[Mapping[str, Any]],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild the frozen V4 reset comparison directly from both raw traces."""

    import numpy as np

    first, second = traces
    authority = C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY
    exact = {
        member: bool(np.array_equal(first[member], second[member]))
        for member in authority["exact_members"]
    }
    delta_pose = np.abs(
        np.asarray(first["base_pose_world"])
        - np.asarray(second["base_pose_world"])
    )
    endpoint_delta = (
        np.asarray(first["base_pose_world"])[-1, :3]
        - np.asarray(second["base_pose_world"])[-1, :3]
    )
    first_yaw = _pose_roll_pitch_yaw_xyzw(
        np.asarray(first["base_pose_world"])[-1]
    )[2]
    second_yaw = _pose_roll_pitch_yaw_xyzw(
        np.asarray(second["base_pose_world"])[-1]
    )[2]
    row: dict[str, Any] = {
        "state_id": state_id,
        "trial_indices": [0, 1],
        "physics_sample_count": int(len(first["timestamp_s"])),
        "maximum_base_position_error_m": float(
            delta_pose[:, :3].max(initial=0.0)
        ),
        "maximum_base_quaternion_component_error": float(
            delta_pose[:, 3:7].max(initial=0.0)
        ),
        "maximum_base_twist_error": float(
            np.abs(
                np.asarray(first["base_twist_world"])
                - np.asarray(second["base_twist_world"])
            ).max(initial=0.0)
        ),
        "maximum_joint_position_error_rad": float(
            np.abs(
                np.asarray(first["joint_position"])
                - np.asarray(second["joint_position"])
            ).max(initial=0.0)
        ),
        "maximum_joint_velocity_error_rad_s": float(
            np.abs(
                np.asarray(first["joint_velocity"])
                - np.asarray(second["joint_velocity"])
            ).max(initial=0.0)
        ),
        "exact_member_equal": exact,
        "endpoint_position_error_m": math.sqrt(
            math.fsum(float(value) ** 2 for value in endpoint_delta)
        ),
        "endpoint_heading_error_rad": abs(
            _wrap_angle_v1(float(first_yaw) - float(second_yaw))
        ),
        "termination_reason_equal": (
            trials[0]["termination_reason"] == trials[1]["termination_reason"]
        ),
        "stuck_equal": bool(trials[0]["stuck"]) is bool(trials[1]["stuck"]),
    }
    tolerance_pairs = (
        ("maximum_base_position_error_m", "reset_base_position_m"),
        (
            "maximum_base_quaternion_component_error",
            "reset_base_quaternion_component",
        ),
        ("maximum_base_twist_error", "reset_base_twist"),
        ("maximum_joint_position_error_rad", "reset_joint_position_rad"),
        ("maximum_joint_velocity_error_rad_s", "reset_joint_velocity_rad_s"),
        ("endpoint_position_error_m", "repeat_endpoint_position_m"),
        ("endpoint_heading_error_rad", "repeat_endpoint_heading_rad"),
    )
    row["passed"] = bool(
        all(
            row[field] <= C.NUMERICAL_TOLERANCES[tolerance]
            for field, tolerance in tolerance_pairs
        )
        and all(exact.values())
        and row["termination_reason_equal"]
        and row["stuck_equal"]
    )
    return row


def _expected_reset_timestamp_tape(capture_timestamp_s: Any) -> Any:
    """Rebuild the exact V1 simulator timestamp operation order."""

    import numpy as np

    capture = _finite(capture_timestamp_s, "snapshot capture timestamp")
    initial_ns = int(round(capture * 1.0e9))
    if float(initial_ns) / 1.0e9 != capture:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "snapshot capture timestamp is not integer-nanosecond aligned"
        )
    values = []
    for sample_index in range(C.RESET_FIXTURE_PHYSICS_SAMPLES):
        policy_step = sample_index // C.PHYSICS_STEPS_PER_POLICY_STEP
        physics_step = sample_index % C.PHYSICS_STEPS_PER_POLICY_STEP
        base_ns = initial_ns + policy_step * int(round(0.02 * 1.0e9))
        values.append(
            float(base_ns) / 1.0e9
            + (physics_step + 1) * C.TEACHER_TRACE_DT_S
        )
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _expected_reset_command_tapes(previous_applied_command: Any) -> tuple[Any, Any]:
    """Rebuild V1 float32 absolute clipping and per-tick slew limiting."""

    import numpy as np

    previous = np.asarray(previous_applied_command, dtype=np.float32).reshape(3)
    requested_tick = np.asarray(C.RESET_FIXTURE_COMMAND, dtype=np.float32)
    lower = np.asarray([-0.3, 0.0, -0.5], dtype=np.float32)
    upper = np.asarray([0.3, 0.0, 0.5], dtype=np.float32)
    delta = np.asarray([0.25, 0.0, 0.35], dtype=np.float32)
    requested_ticks: list[Any] = []
    applied_ticks: list[Any] = []
    for _ in range(C.RESET_FIXTURE_COMMAND_TICKS):
        requested = requested_tick.copy()
        bounded = np.clip(requested, lower, upper)
        bounded = np.clip(bounded, previous - delta, previous + delta)
        requested_ticks.append(requested.copy())
        applied_ticks.append(np.asarray(bounded, dtype=np.float32).copy())
        previous = np.asarray(bounded, dtype=np.float32).copy()
    requested_samples = np.repeat(
        np.asarray(requested_ticks, dtype=np.float32),
        C.PHYSICS_STEPS_PER_COMMAND_TICK,
        axis=0,
    ).astype(np.float64)
    applied_samples = np.repeat(
        np.asarray(applied_ticks, dtype=np.float32),
        C.PHYSICS_STEPS_PER_COMMAND_TICK,
        axis=0,
    ).astype(np.float64)
    return (
        np.ascontiguousarray(requested_samples),
        np.ascontiguousarray(applied_samples),
    )


def _expected_reset_region_memberships(
    base_pose_world: Any, geometry: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild V1 boundary-inclusive indicator tapes from reopened poses."""

    import numpy as np

    poses = np.asarray(base_pose_world)
    points = [[float(pose[0]), float(pose[1])] for pose in poses]
    return {
        "source_region_member": np.asarray(
            [
                point_in_polygon_inclusive(
                    point, geometry["source_node"]["boundary_polygon_world"]
                )
                for point in points
            ],
            dtype=np.uint8,
        ),
        "correct_edge_region_member": np.asarray(
            [
                point_in_polygon_inclusive(
                    point,
                    geometry["selected_directed_edge"][
                        "edge_region_polygon_world"
                    ],
                )
                for point in points
            ],
            dtype=np.uint8,
        ),
        "wrong_edge_region_member": np.asarray(
            [
                any(
                    point_in_polygon_inclusive(
                        point, edge["edge_region_polygon_world"]
                    )
                    for edge in geometry["competing_directed_edges"]
                )
                for point in points
            ],
            dtype=np.uint8,
        ),
        "target_region_member": np.asarray(
            [
                point_in_polygon_inclusive(
                    point, geometry["target_node"]["boundary_polygon_world"]
                )
                for point in points
            ],
            dtype=np.uint8,
        ),
    }


def _validate_reset_region_memberships(
    trace: Mapping[str, Any], geometry: Mapping[str, Any]
) -> None:
    """Require every persisted reset indicator to equal reopened geometry."""

    import numpy as np

    expected = _expected_reset_region_memberships(
        trace["base_pose_world"], geometry
    )
    if any(
        not np.array_equal(trace[name], tape)
        for name, tape in expected.items()
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset region-membership tape drift"
        )


def validate_selected_reset_material_shard(
    metadata: Any,
    *,
    reopened_arrays: Mapping[str, Any],
    selected_candidate_spec: Mapping[str, Any],
    expected_panel_index: int,
    material_metadata_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one state-id keyed selected snapshot and two reset traces."""

    import numpy as np

    row = _mapping(
        metadata, C.SELECTED_RESET_MATERIAL_FIELDS, "selected reset metadata"
    )
    C.validate_content_digest(row)
    if (
        row["schema"] != C.SELECTED_RESET_MATERIAL_SCHEMA
        or row["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset identity drift"
        )
    panel_index = _nonnegative_int(row["panel_index"], "panel_index")
    if panel_index >= C.PANEL_STATE_COUNT or panel_index != expected_panel_index:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset panel index drift"
        )
    spec = C.validate_candidate_spec(selected_candidate_spec)
    if (
        row["candidate_spec"] != spec
        or row["qualification_candidate_index"] != spec["candidate_index"]
        or row["state_id"] != spec["state_id"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset candidate identity drift"
        )
    _commit(row["source_freeze_commit"], "selected reset source freeze")
    _sha(row["runtime_contract_content_digest"], "selected reset runtime digest")
    source_prefix = (
        f"qualification/{spec['stream_id']}/attempt-{spec['attempt_index']:02d}"
    )
    source_metadata = _file_binding(
        row["source_terminal_metadata_binding"], "source terminal metadata"
    )
    source_payload = _file_binding(
        row["source_terminal_payload_binding"], "source terminal payload"
    )
    if (
        source_metadata["path"] != f"{source_prefix}/metadata.json"
        or source_payload["path"] != f"{source_prefix}/payload.npz"
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset source material path drift"
        )
    persisted = validate_persisted_array_evidence(
        row["persisted_array_evidence"], reopened_arrays=reopened_arrays
    )
    shard_id = f"selected/{spec['state_id']}"
    _validate_successor_payload_binding(
        row,
        persisted,
        shard_kind="selected_state_material",
        shard_id=shard_id,
    )
    expected_members = {
        "snapshot_payload_bytes",
        "rgb",
        *(f"snapshot__{name}" for name in C.SNAPSHOT_NUMERIC_FIELDS),
        *(
            f"fixture_{trial}__{name}"
            for trial in range(2)
            for name in C.CANDIDATE_TRACE_MEMBER_ORDER
        ),
    }
    if set(reopened_arrays) != expected_members:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset array inventory drift"
        )
    snapshot_payload = np.asarray(reopened_arrays["snapshot_payload_bytes"])
    rgb = np.asarray(reopened_arrays["rgb"])
    if (
        snapshot_payload.dtype.str != "|u1"
        or snapshot_payload.ndim != 1
        or not snapshot_payload.flags.c_contiguous
        or not snapshot_payload.size
        or rgb.dtype.str != "|u1"
        or list(rgb.shape) != [168, 224, 3]
        or not rgb.flags.c_contiguous
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected snapshot payload/RGB extent drift"
        )
    snapshot_sha = hashlib.sha256(snapshot_payload.tobytes(order="C")).hexdigest()
    if (
        row["snapshot_payload_sha256"] != snapshot_sha
        or row["rgb_sha256"] != _canonical_array_sha256_local(rgb)
        or row["snapshot_identity"].get("artifact_file_sha256") != snapshot_sha
        or row["snapshot"].get("snapshot_payload_sha256") != snapshot_sha
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected snapshot digest cross-link drift"
        )
    _snapshot_identity(row["snapshot_identity"], "selected snapshot identity", probe_tipped=False)
    for member, shape in C.SNAPSHOT_NUMERIC_MEMBER_SHAPES.items():
        value = np.asarray(reopened_arrays[f"snapshot__{member}"])
        if (
            value.dtype.str != "<f8"
            or list(value.shape) != shape
            or not value.flags.c_contiguous
            or not bool(np.isfinite(value).all())
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"selected snapshot numeric extent drift: {member}"
            )
    trials = _sequence(row["reset_trials"], "selected reset trials")
    if len(trials) != 2:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset trial cardinality drift"
        )
    byte_members = {
        "physics_contact",
        "source_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    tails = C.CANDIDATE_TRACE_MEMBER_SHAPES
    trace_rows: list[dict[str, Any]] = []
    validated_trials: list[dict[str, Any]] = []
    expected_timestamps = _expected_reset_timestamp_tape(
        row["snapshot"].get("capture_timestamp_s")
    )
    expected_requested, expected_applied = _expected_reset_command_tapes(
        reopened_arrays["snapshot__previous_applied_command"]
    )
    for trial_index, raw_trial in enumerate(trials):
        trial = _mapping(
            raw_trial,
            C.RESET_TRIAL_MATERIAL_FIELDS,
            f"selected reset trial[{trial_index}]",
        )
        if (
            trial["trial_index"] != trial_index
            or trial["serialized_restore_used"] is not True
            or trial["clone_equivalence_used"] is not False
            or trial["restored_snapshot_sha256"] != snapshot_sha
            or trial["termination_reason"] != "H3_COMPLETE"
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset trial authority drift"
            )
        _bool(trial["stuck"], "selected reset stuck")
        for field in (
            "post_restore_state_sha256",
            "current_rgb_sha256",
            "joint_position_sha256",
            "joint_velocity_sha256",
            "controller_state_sha256",
            "rng_state_sha256",
            "requested_command_sequence_sha256",
            "post_slew_applied_command_sequence_sha256",
            "contact_sequence_sha256",
        ):
            _sha(trial[field], f"selected reset {field}")
        digests = _mapping(
            trial["trace_digests"],
            C.CANDIDATE_TRACE_MEMBER_ORDER,
            "selected reset trace digests",
        )
        trace: dict[str, Any] = {}
        for member in C.CANDIDATE_TRACE_MEMBER_ORDER:
            value = np.asarray(reopened_arrays[f"fixture_{trial_index}__{member}"])
            expected_dtype = "|u1" if member in byte_members else "<f8"
            expected_tail = list(tails[member])
            if (
                value.dtype.str != expected_dtype
                or list(value.shape) != [C.V4.RESET_FIXTURE_PHYSICS_SAMPLES, *expected_tail]
                or not value.flags.c_contiguous
                or digests[member] != _canonical_array_sha256_local(value)
            ):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"selected reset raw trace drift: {trial_index}/{member}"
                )
            trace[member] = value
        if not np.array_equal(trace["timestamp_s"], expected_timestamps):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset timestamp tape drift"
            )
        for member in byte_members:
            if not bool(np.isin(trace[member], [0, 1]).all()):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"selected reset nonbinary trace member: {member}"
                )
        geometry = spec["geometry"]
        _validate_reset_region_memberships(trace, geometry)
        if not np.array_equal(trace["requested_command"], expected_requested):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset requested-command tape drift"
            )
        applied = trace["post_slew_applied_command"]
        if (
            not np.array_equal(applied, expected_applied)
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset deterministic post-slew tape drift"
            )
        floating_members = set(C.CANDIDATE_TRACE_MEMBER_ORDER) - byte_members
        if any(
            not bool(np.isfinite(trace[member]).all())
            for member in floating_members
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset contains nonfinite trace evidence"
            )
        if any(
            any(_teacher_termination_flags_from_pose(pose).values())
            for pose in trace["base_pose_world"]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset terminates before declared H3_COMPLETE"
            )
        start_pose = trace["base_pose_world"][0]
        end_pose = trace["base_pose_world"][-1]
        start_yaw = _pose_roll_pitch_yaw_xyzw(start_pose)[2]
        end_yaw = _pose_roll_pitch_yaw_xyzw(end_pose)[2]
        activity = float(np.max(np.abs(trace["requested_command"])))
        expected_stuck = bool(
            activity
            > C.PHYSICAL_OUTCOME_AUTHORITY["stuck"][
                "command_activity_threshold"
            ]
            and math.hypot(
                float(end_pose[0]) - float(start_pose[0]),
                float(end_pose[1]) - float(start_pose[1]),
            )
            < C.PHYSICAL_OUTCOME_AUTHORITY["stuck"][
                "h3_translation_threshold_m"
            ]
            and abs(_wrap_angle_v1(float(end_yaw) - float(start_yaw)))
            < C.PHYSICAL_OUTCOME_AUTHORITY["stuck"][
                "h3_heading_threshold_rad"
            ]
        )
        if trial["stuck"] is not expected_stuck:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset stuck derivation drift"
            )
        if (
            trial["requested_command_sequence_sha256"]
            != _canonical_array_sha256_local(trace["requested_command"])
            or trial["post_slew_applied_command_sequence_sha256"]
            != _canonical_array_sha256_local(
                trace["post_slew_applied_command"]
            )
            or trial["contact_sequence_sha256"]
            != _canonical_array_sha256_local(trace["physics_contact"])
            or trial["joint_position_sha256"]
            != _canonical_array_sha256_local(trace["joint_position"][-1])
            or trial["joint_velocity_sha256"]
            != _canonical_array_sha256_local(trace["joint_velocity"][-1])
            or trial["current_rgb_sha256"] != row["rgb_sha256"]
            or trial["base_pose_world"]
            != [float(value) for value in trace["base_pose_world"][-1]]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset trial/raw summary drift"
            )
        trace_rows.append(trace)
        validated_trials.append(trial)
    for field in (
        "post_restore_state_sha256",
        "controller_state_sha256",
        "rng_state_sha256",
    ):
        if validated_trials[0][field] != validated_trials[1][field]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"selected reset cross-trial {field} drift"
            )
    pair = _mapping(
        row["reset_pair_comparison"],
        C.RESET_PAIR_COMPARISON_FIELDS,
        "selected reset pair comparison",
    )
    rebuilt_pair = _build_reset_pair_comparison(
        spec["state_id"], trace_rows, validated_trials
    )
    if (
        pair != rebuilt_pair
        or pair["passed"] is not True
        or row["reset_fixture_passed"] is not True
        or row["fanout_or_ranker_outcome_opened"] is not False
        or row["models_trained"] != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset pair/stage boundary drift"
        )
    previous = next(
        item
        for item in persisted["arrays"]
        if item["member"] == "snapshot__previous_applied_command"
    )
    if row["snapshot"].get("previous_applied_command_sha256") != previous[
        "array_bytes_sha256"
    ]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected reset previous-command raw-byte binding drift"
        )
    metadata_binding = _validate_optional_metadata_binding(
        material_metadata_binding,
        row,
        expected_path=f"{shard_id}/metadata.json",
        label="selected material metadata binding",
    )
    return {
        "metadata": row,
        "arrays": {name: reopened_arrays[name] for name in reopened_arrays},
        "candidate_spec": spec,
        "material_metadata_binding": metadata_binding,
        "material_payload_binding": persisted["payload_file"],
    }


def _split_assignments(
    selected_specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    assignments: dict[str, dict[str, Any]] = {}
    population: list[dict[str, Any]] = []
    for family in C.FAMILY_IDS:
        ranked: list[tuple[str, str, Mapping[str, Any]]] = []
        for spec in selected_specs:
            if spec["family"] != family:
                continue
            projection = {
                "experiment_id": C.V4.V3.V2.V1.EXPERIMENT_ID,
                "state_id": spec["state_id"],
                "family": family,
                "candidate_spec_id": spec["candidate_spec_id"],
                "canonical_spec_sha256": spec["canonical_spec_sha256"],
            }
            digest = _canonical_no_lf_sha256(projection)
            ranked.append((digest, str(spec["state_id"]), spec))
        ranked.sort(key=lambda item: (item[0], item[1]))
        if len(ranked) != 16:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "panel split family population drift"
            )
        for rank, (digest, state_id, spec) in enumerate(ranked):
            role = "DEVELOPMENT" if rank < 12 else "DEVELOPMENT_HELDOUT"
            assignments[state_id] = {
                "state_id": state_id,
                "family": family,
                "role": role,
                "assignment_sha256": digest,
                "rank_within_family": rank,
            }
            population.append(
                {
                    "state_id": state_id,
                    "family": family,
                    "candidate_spec_id": spec["candidate_spec_id"],
                    "assignment_sha256": digest,
                }
            )
    population.sort(key=lambda item: item["state_id"])
    return assignments, _canonical_no_lf_sha256(population)


def _interpolated_pose_se2(
    before: Sequence[float], after: Sequence[float], fraction: float
) -> list[float]:
    alpha = _finite(fraction, "crossing fraction")
    yaw_before = _pose_roll_pitch_yaw_xyzw(before)[2]
    yaw_after = _pose_roll_pitch_yaw_xyzw(after)[2]
    return [
        float(before[0]) + alpha * (float(after[0]) - float(before[0])),
        float(before[1]) + alpha * (float(after[1]) - float(before[1])),
        _wrap_angle_v1(
            float(yaw_before)
            + alpha * _wrap_angle_v1(float(yaw_after) - float(yaw_before))
        ),
    ]


def _lookahead_from_port(
    port_pose: Sequence[float],
    route: Sequence[Sequence[float]],
    distance_m: float,
) -> tuple[list[float], bool, float]:
    points = [_vector(point, 2, "route point") for point in route]
    if len(points) < 2:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "directed route is degenerate"
        )
    px, py = float(port_pose[0]), float(port_pose[1])
    candidates: list[tuple[float, int, float, list[float]]] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        dx, dy = right[0] - left[0], right[1] - left[1]
        denominator = dx * dx + dy * dy
        if denominator <= 0.0:
            continue
        fraction = max(
            0.0,
            min(
                1.0,
                ((px - left[0]) * dx + (py - left[1]) * dy) / denominator,
            ),
        )
        projected = [left[0] + fraction * dx, left[1] + fraction * dy]
        candidates.append(
            (math.hypot(px - projected[0], py - projected[1]), index, fraction, projected)
        )
    if not candidates:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "directed route has no positive-length segment"
        )
    _, segment_index, _, projected = min(candidates)
    segments = [(projected, points[segment_index + 1])] + list(
        zip(points[segment_index + 1 :], points[segment_index + 2 :])
    )
    remaining = float(sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in segments
    ))
    desired = min(float(distance_m), remaining)
    travelled = 0.0
    target = projected
    heading = float(port_pose[2])
    for left, right in segments:
        length = math.hypot(right[0] - left[0], right[1] - left[1])
        if length <= 0.0:
            continue
        heading = math.atan2(right[1] - left[1], right[0] - left[0])
        if travelled + length >= desired:
            alpha = (desired - travelled) / length
            target = [
                left[0] + alpha * (right[0] - left[0]),
                left[1] + alpha * (right[1] - left[1]),
            ]
            break
        travelled += length
        target = right
    return (
        [float(target[0]), float(target[1]), _wrap_angle_v1(heading)],
        remaining < float(distance_m),
        float(remaining),
    )


def _body_from_world(
    world_pose: Sequence[float], body_pose: Sequence[float]
) -> list[float]:
    wx, wy, wyaw = _vector(world_pose, 3, "world pose")
    bx, by, byaw = _vector(body_pose, 3, "body pose")
    cosine, sine = math.cos(byaw), math.sin(byaw)
    dx, dy = wx - bx, wy - by
    return [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        _wrap_angle_v1(wyaw - byaw),
    ]


def _world_from_body(
    target: Sequence[float], body_pose: Sequence[float]
) -> list[float]:
    dx, dy, dyaw = _vector(target, 3, "body target")
    bx, by, byaw = _vector(body_pose, 3, "body pose")
    cosine, sine = math.cos(byaw), math.sin(byaw)
    return [
        bx + cosine * dx - sine * dy,
        by + sine * dx + cosine * dy,
        _wrap_angle_v1(byaw + dyaw),
    ]


def _target_contract_row(
    *,
    panel_index: int,
    qualification_candidate_index: int,
    state_id: str,
    target_id: str,
    source_pose: Sequence[float],
    target_pose: Sequence[float],
    route_intent_world: Sequence[float],
) -> dict[str, Any]:
    body = _body_from_world(target_pose, source_pose)
    roundtrip = _world_from_body(body, source_pose)
    position_error = math.hypot(
        roundtrip[0] - float(target_pose[0]),
        roundtrip[1] - float(target_pose[1]),
    )
    heading_error = abs(_wrap_angle_v1(roundtrip[2] - float(target_pose[2])))
    cosine, sine = math.cos(float(source_pose[2])), math.sin(float(source_pose[2]))
    route_dx = cosine * float(route_intent_world[0]) + sine * float(route_intent_world[1])
    route_dy = -sine * float(route_intent_world[0]) + cosine * float(route_intent_world[1])
    bearing = math.atan2(body[1], body[0])
    return {
        "panel_index": panel_index,
        "qualification_candidate_index": qualification_candidate_index,
        "state_id": state_id,
        "target_id": target_id,
        "source_body_pose_world": [float(value) for value in source_pose],
        "target_world_pose": [float(value) for value in target_pose],
        "target_body_pose": body,
        "dx_m": body[0],
        "dy_m": body[1],
        "distance_m": math.hypot(body[0], body[1]),
        "relative_heading_rad": bearing,
        "relative_heading_sin": math.sin(bearing),
        "relative_heading_cos": math.cos(bearing),
        "target_tangent_heading_rad": body[2],
        "target_tangent_heading_sin": math.sin(body[2]),
        "target_tangent_heading_cos": math.cos(body[2]),
        "route_intent_dx": route_dx,
        "route_intent_dy": route_dy,
        "roundtrip_world_pose": roundtrip,
        "position_transform_error_m": position_error,
        "heading_transform_error_rad": heading_error,
        "transform_valid": bool(
            position_error
            <= C.NUMERICAL_TOLERANCES["inverse_transform_position_m"]
            and heading_error
            <= C.NUMERICAL_TOLERANCES["inverse_transform_heading_rad"]
        ),
    }


def build_frozen_panel_documents(
    terminal_records: Sequence[Mapping[str, Any]],
    generator_metrics: Mapping[str, Any],
    panel_handoff: Mapping[str, Any],
    qualification_material: Sequence[Mapping[str, Any]],
    selected_material: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    """Build six native panel documents from reopened physical evidence."""

    import numpy as np

    runtime = C.validate_runtime_contract(runtime_contract)
    records = validate_generator_terminal_records_jsonl(
        build_generator_terminal_records_jsonl(terminal_records)
    )
    generator = validate_generator_metrics(generator_metrics, records)
    handoff = validate_panel_handoff(panel_handoff, records, generator)
    if generator["status"] != C.GENERATOR_PANEL_AVAILABLE:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel documents require generator availability"
        )
    record_by_index = {row["candidate_index"]: row for row in records}
    qualification_by_index: dict[int, dict[str, Any]] = {}
    qualification_rows = _sequence(
        qualification_material, "qualification material"
    )
    if len(qualification_rows) != len(records):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material population cardinality drift"
        )
    for raw in qualification_rows:
        if not isinstance(raw, Mapping) or set(raw) < {"metadata", "arrays"}:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "qualification validation wrapper drift"
            )
        index = _nonnegative_int(raw["metadata"].get("pool_index"), "qualification candidate index")
        if index in qualification_by_index:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "duplicate qualification material candidate"
            )
        validated = validate_qualification_material_shard(
            raw["metadata"], reopened_arrays=raw["arrays"], expected_pool_index=index
        )
        qualification_by_index[index] = validated
    if (
        set(qualification_by_index) != set(record_by_index)
        or len(qualification_by_index) != len(records)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material/terminal population drift"
        )
    selected_specs = sorted(
        (C.validate_candidate_spec(spec) for spec in handoff["selected_candidate_specs"]),
        key=lambda spec: spec["state_id"],
    )
    if len(selected_specs) != C.PANEL_STATE_COUNT:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected panel cardinality drift"
        )
    selected_rows = _sequence(selected_material, "selected material")
    expected_selected_states = {spec["state_id"] for spec in selected_specs}
    observed_selected_states = [
        raw.get("metadata", {}).get("state_id")
        if isinstance(raw, Mapping)
        and isinstance(raw.get("metadata"), Mapping)
        else None
        for raw in selected_rows
    ]
    if (
        len(selected_rows) != C.PANEL_STATE_COUNT
        or set(observed_selected_states) != expected_selected_states
        or len(set(observed_selected_states)) != C.PANEL_STATE_COUNT
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected material exact state population drift"
        )
    generator_runtime = build_generator_runtime_environment(
        [qualification_by_index[index]["metadata"] for index in sorted(qualification_by_index)],
        runtime,
        allow_fake_runtime=allow_fake_runtime,
    )
    physical_runtime = generator_runtime["physical_runtime_core"]
    backend_runtime_core = generator_runtime["backend_runtime_core"]
    selected_runtime_sha256s: list[str] = []
    selected_by_state: dict[str, dict[str, Any]] = {}
    for panel_index, spec in enumerate(selected_specs):
        matches = [
            raw
            for raw in selected_rows
            if isinstance(raw, Mapping)
            and isinstance(raw.get("metadata"), Mapping)
            and raw["metadata"].get("state_id") == spec["state_id"]
        ]
        if len(matches) != 1:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected material state coverage drift"
            )
        raw = matches[0]
        validated = validate_selected_reset_material_shard(
            raw["metadata"],
            reopened_arrays=raw["arrays"],
            selected_candidate_spec=spec,
            expected_panel_index=panel_index,
            material_metadata_binding=raw.get("material_metadata_binding"),
        )
        if validated["material_metadata_binding"] is None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected metadata file binding absent"
            )
        if (
            validated["metadata"]["source_freeze_commit"]
            != runtime["source_freeze_commit"]
            or validated["metadata"]["runtime_contract_content_digest"]
            != runtime["content_digest"]
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected material/runtime binding drift"
            )
        selected_stage = _mapping(
            validated["metadata"]["stage_runtime"],
            set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
            "selected reset stage runtime",
        )
        selected_backend = _mapping(
            validated["metadata"]["backend_runtime"],
            set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS),
            "selected reset backend runtime",
        )
        if selected_stage != physical_runtime or selected_backend != backend_runtime_core:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset runtime differs from qualification runtime core"
            )
        selected_runtime_sha256s.append(runtime_environment_sha256(selected_stage))
        selected_by_state[spec["state_id"]] = validated
    selected_indices = {spec["candidate_index"] for spec in selected_specs}
    panel_index_by_candidate = {
        spec["candidate_index"]: index for index, spec in enumerate(selected_specs)
    }
    teacher_records: list[dict[str, Any]] = []
    teacher_index_by_candidate: dict[int, int] = {}
    for index in sorted(qualification_by_index):
        validation = qualification_by_index[index]
        metadata = validation["metadata"]
        if metadata["state_disposition"]["teacher_executed"] is not True:
            continue
        reduction = reduce_qualification_teacher_trace(
            metadata, validation["arrays"], expected_pool_index=index
        )
        terminal = record_by_index[index]
        trace_index = len(teacher_records)
        teacher_index_by_candidate[index] = trace_index
        projection = copy.deepcopy(reduction["teacher_record_raw_projection"])
        teacher_records.append(
            {
                "trace_index": trace_index,
                "qualification_candidate_index": index,
                "qualification_material_path": str(
                    terminal["material_metadata_binding"]["path"]
                ).rsplit("/", 1)[0],
                "qualification_material_metadata_binding": copy.deepcopy(
                    terminal["material_metadata_binding"]
                ),
                "qualification_material_payload_binding": copy.deepcopy(
                    terminal["material_payload_binding"]
                ),
                "candidate_spec_id": terminal["candidate_spec_id"],
                "state_id": metadata["candidate_spec"]["state_id"],
                "family": terminal["family"],
                "stratum_index": terminal["stratum_index"],
                "attempt_index": terminal["attempt_index"],
                "canonical_spec_sha256": terminal["canonical_spec_sha256"],
                "teacher_trace_id": f"{metadata['candidate_spec']['state_id']}::teacher",
                "selected": index in selected_indices,
                "panel_index": panel_index_by_candidate.get(index),
                "sample_count": reduction["sample_count"],
                "trace_digests": copy.deepcopy(metadata["teacher"]["trace_digests"]),
                "teacher_reduction_sha256": _canonical_no_lf_sha256(reduction),
                "raw_projection": projection,
            }
        )
    if [row["trace_index"] for row in teacher_records] != list(range(len(teacher_records))):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "compact teacher trace index drift"
        )
    if any(index not in teacher_index_by_candidate for index in selected_indices):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected state lacks teacher trace"
        )
    assignments, population_sha = _split_assignments(selected_specs)
    panel_states: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    port_records: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for panel_index, spec in enumerate(selected_specs):
        index = spec["candidate_index"]
        terminal = record_by_index[index]
        qualification = qualification_by_index[index]
        qmeta = qualification["metadata"]
        selected = selected_by_state[spec["state_id"]]
        smeta = selected["metadata"]
        if (
            terminal["qualified"] is not True
            or terminal["disposition"] != "QUALIFIED"
            or qmeta["qualified"] is not True
            or qmeta["disposition"] != "QUALIFIED"
            or terminal["state_disposition"] != qmeta["state_disposition"]
            or smeta["source_terminal_metadata_binding"]
            != terminal["material_metadata_binding"]
            or smeta["source_terminal_payload_binding"]
            != terminal["material_payload_binding"]
            or terminal["persisted_array_evidence_sha256"]
            != _canonical_no_lf_sha256(qmeta["persisted_array_evidence"])
            or terminal["material_metadata_binding"]
            != _canonical_file_binding(
                terminal["material_metadata_binding"]["path"],
                C.canonical_json_bytes(qmeta),
            )
            or qmeta["payload"]
            != {
                "role": "material_shard_payload",
                **terminal["material_payload_binding"],
                "kind": "npz",
            }
            or smeta["snapshot"] != qmeta["snapshot"]
            or smeta["snapshot_identity"] != qmeta["snapshot_identity"]
            or smeta["snapshot_payload_sha256"]
            != qmeta["snapshot"]["snapshot_payload_sha256"]
            or smeta["rgb_sha256"] != qmeta["current_rgb_sha256"]
            or not all(
                selected["arrays"][name] is not None
                and qualification["arrays"][name] is not None
                and np.array_equal(
                    selected["arrays"][name], qualification["arrays"][name]
                )
                for name in (
                    "snapshot_payload_bytes",
                    "rgb",
                    *(f"snapshot__{field}" for field in C.SNAPSHOT_NUMERIC_FIELDS),
                )
            )
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected reset does not exactly bind qualified source evidence"
            )
        assignment = assignments[spec["state_id"]]
        graph = qmeta["graph"]
        if not isinstance(graph, Mapping):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected graph evidence absent"
            )
        selected_edge = spec["geometry"]["selected_directed_edge"]
        source_node = spec["geometry"]["source_node"]
        target_node = spec["geometry"]["target_node"]
        teacher_index = teacher_index_by_candidate[index]
        eligibility = {
            "snapshot_complete": True,
            "exact_reset_fixture_passed": True,
            "teacher_trace_contact_free": bool(qmeta["teacher"]["contact_free"]),
            "teacher_valid": bool(qmeta["teacher"]["teacher_valid"]),
            "directed_port_defined": qmeta["teacher"]["crossing"] is not None,
            "current_rgb_valid": bool(
                qmeta["state_disposition"]["teacher_criteria"]["current_rgb_valid"]
            ),
            "graph_edge_physically_executable": bool(
                qmeta["graph_edge_physically_executable"]
            ),
        }
        if not all(eligibility.values()):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "selected state is not fully eligible"
            )
        panel_states.append(
            {
                "panel_index": panel_index,
                "qualification_candidate_index": index,
                "stream_index": spec["stream_index"],
                "stream_id": spec["stream_id"],
                "attempt_index": spec["attempt_index"],
                "state_id": spec["state_id"],
                "scene_id": spec["scene_id"],
                "episode_id": spec["episode_id"],
                "graph_id": spec["graph_id"],
                "family": spec["family"],
                "stratum_index": spec["stratum_index"],
                "route_direction": spec["route_direction"],
                "passage_width_id": spec["passage_width_id"],
                "port_distance_id": spec["port_distance_id"],
                "candidate_spec_id": spec["candidate_spec_id"],
                "canonical_spec_sha256": spec["canonical_spec_sha256"],
                "procedural_seed": spec["procedural_seed"],
                "geometry_sha256": _canonical_no_lf_sha256(spec["geometry"]),
                "source_node_id": selected_edge["source_node_id"],
                "target_node_id": selected_edge["target_node_id"],
                "directed_edge_id": selected_edge["edge_id"],
                "role": assignment["role"],
                "assignment_sha256": assignment["assignment_sha256"],
                "rank_within_family": assignment["rank_within_family"],
                "qualification_terminal_record_content_digest": terminal[
                    "content_digest"
                ],
                "qualification_material_metadata_binding": copy.deepcopy(
                    terminal["material_metadata_binding"]
                ),
                "qualification_material_payload_binding": copy.deepcopy(
                    terminal["material_payload_binding"]
                ),
                "selected_material_metadata_binding": copy.deepcopy(
                    selected["material_metadata_binding"]
                ),
                "selected_material_payload_binding": copy.deepcopy(
                    selected["material_payload_binding"]
                ),
                "snapshot_identity": copy.deepcopy(smeta["snapshot_identity"]),
                "teacher_trace_index": teacher_index,
                "goal_reachable": True,
                "eligible": True,
                "eligibility_evidence": eligibility,
            }
        )
        snapshot_records.append(
            {
                "panel_index": panel_index,
                "qualification_candidate_index": index,
                "state_id": spec["state_id"],
                "snapshot_identity": copy.deepcopy(smeta["snapshot_identity"]),
                "snapshot_payload_sha256": smeta["snapshot_payload_sha256"],
                "selected_material_metadata_binding": copy.deepcopy(
                    selected["material_metadata_binding"]
                ),
                "selected_material_payload_binding": copy.deepcopy(
                    selected["material_payload_binding"]
                ),
                "previous_applied_command_sha256": smeta["snapshot"][
                    "previous_applied_command_sha256"
                ],
                "rgb_sha256": smeta["rgb_sha256"],
                "reset_fixture_passed": True,
                "reset_pair_comparison": copy.deepcopy(
                    smeta["reset_pair_comparison"]
                ),
            }
        )
        teacher = teacher_records[teacher_index]
        projection = teacher["raw_projection"]
        after = int(projection["first_crossing_sample_index"])
        before = after - 1
        trace_pose = qualification["arrays"]["teacher__base_pose_world"]
        port = _interpolated_pose_se2(
            trace_pose[before],
            trace_pose[after],
            projection["crossing_segment_fraction"],
        )
        route = copy.deepcopy(spec["geometry"]["teacher_route_polyline_world"])
        lookahead, clipped, remaining = _lookahead_from_port(
            port, route, C.ROUTE_LOOKAHEAD_DISTANCE_M
        )
        port_record = {
            "panel_index": panel_index,
            "qualification_candidate_index": index,
            "state_id": spec["state_id"],
            "directed_edge_id": selected_edge["edge_id"],
            "source_node_id": source_node["node_id"],
            "target_node_id": target_node["node_id"],
            "teacher_trace_index": teacher_index,
            "teacher_trace_id": teacher["teacher_trace_id"],
            "source_boundary_polygon_world": copy.deepcopy(
                source_node["boundary_polygon_world"]
            ),
            "opening_segment_world": copy.deepcopy(
                selected_edge["opening_segment_world"]
            ),
            "port_normal_world": copy.deepcopy(
                selected_edge["opening_normal_world"]
            ),
            "lateral_bounds_m": [
                -float(selected_edge["opening_width_m"]) / 2.0,
                float(selected_edge["opening_width_m"]) / 2.0,
            ],
            "route_polyline_world": route,
            "crossing_sample_before": before,
            "crossing_sample_after": after,
            "crossing_fraction": projection["crossing_segment_fraction"],
            "teacher_crossing_time_s": float(
                qualification["arrays"]["teacher__timestamp_s"][before]
            )
            + float(projection["crossing_segment_fraction"])
            * (
                float(qualification["arrays"]["teacher__timestamp_s"][after])
                - float(qualification["arrays"]["teacher__timestamp_s"][before])
            ),
            "teacher_crossing_velocity_world_xy": copy.deepcopy(
                projection["crossing_velocity_world_xy"]
            ),
            "teacher_crossing_velocity_heading_world_rad": projection[
                "crossing_velocity_heading_world_rad"
            ],
            "directed_port_world": port,
            "route_lookahead_world": lookahead,
            "route_lookahead_clipped": clipped,
            "remaining_route_length_m": remaining,
            "port_definition": (
                "first actual teacher source-boundary crossing through selected transverse opening"
            ),
        }
        port_records.append(port_record)
        snapshot_pose = selected["arrays"]["snapshot__base_pose_world"]
        source_pose = [
            float(snapshot_pose[0]),
            float(snapshot_pose[1]),
            float(_pose_roll_pitch_yaw_xyzw(snapshot_pose)[2]),
        ]
        route_last_a, route_last_b = route[-2], route[-1]
        target_heading = math.atan2(
            float(route_last_b[1]) - float(route_last_a[1]),
            float(route_last_b[0]) - float(route_last_a[0]),
        )
        targets = {
            "TARGET_NODE_CENTRE": [
                *target_node["centre_world"],
                target_heading,
            ],
            "DIRECTED_EDGE_PORT": port,
            "ROUTE_LOOKAHEAD": lookahead,
        }
        normal = selected_edge["opening_normal_world"]
        for target_id in C.TARGET_IDS:
            target_rows.append(
                _target_contract_row(
                    panel_index=panel_index,
                    qualification_candidate_index=index,
                    state_id=spec["state_id"],
                    target_id=target_id,
                    source_pose=source_pose,
                    target_pose=targets[target_id],
                    route_intent_world=normal,
                )
            )
    if [row["state_id"] for row in panel_states] != sorted(
        row["state_id"] for row in panel_states
    ) or [row["panel_index"] for row in panel_states] != list(range(C.PANEL_STATE_COUNT)):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel canonical state order drift"
        )
    role_counts = dict(sorted(Counter(row["role"] for row in panel_states).items()))
    family_role_counts = {
        family: {
            role: sum(
                row["family"] == family and row["role"] == role
                for row in panel_states
            )
            for role in C.ROLE_IDS
        }
        for family in C.FAMILY_IDS
    }
    panel = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1.panel_manifest.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "physical_runtime_environment": copy.deepcopy(physical_runtime),
            "physical_runtime_environment_sha256": runtime_environment_sha256(
                physical_runtime
            ),
            "selected_reset_runtime_sha256s": selected_runtime_sha256s,
            "generator_metrics_content_digest": generator["content_digest"],
            "panel_handoff_content_digest": handoff["content_digest"],
            "state_count": len(panel_states),
            "canonical_state_order": "ascending state_id",
            "role_counts": role_counts,
            "family_role_counts": family_role_counts,
            "selection_rule": C.GENERATOR_ALLOCATION_AUTHORITY["panel_selection"],
            "states": panel_states,
        }
    )
    split = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1.split_manifest.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "panel_manifest_content_digest": panel["content_digest"],
            "assignment_algorithm": (
                "within each frozen family sort by (assignment_sha256,state_id); "
                "ranks 0..11 DEVELOPMENT and 12..15 DEVELOPMENT_HELDOUT"
            ),
            "complete_eligible_population_sha256": population_sha,
            "heldout_outcomes_opened_before_assignment": 0,
            "role_counts": role_counts,
            "family_role_counts": family_role_counts,
            "assignments": [
                {
                    "panel_index": row["panel_index"],
                    "qualification_candidate_index": row[
                        "qualification_candidate_index"
                    ],
                    **assignments[row["state_id"]],
                }
                for row in panel_states
            ],
        }
    )
    snapshots = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "state_snapshot_index.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "panel_manifest_content_digest": panel["content_digest"],
            "record_count": len(snapshot_records),
            "records": snapshot_records,
        }
    )
    teachers = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "teacher_trace_index.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "generator_terminal_record_count": len(records),
            "teacher_executed_count": len(teacher_records),
            "selected_teacher_trace_count": sum(
                row["selected"] for row in teacher_records
            ),
            "records": teacher_records,
        }
    )
    ports = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "edge_port_index.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "panel_manifest_content_digest": panel["content_digest"],
            "teacher_trace_index_content_digest": teachers["content_digest"],
            "record_count": len(port_records),
            "records": port_records,
        }
    )
    support = C.ORIGINAL_RANKER_TRAINING_SUPPORT
    support_counts = {
        target_id: {"inside": 0, "outside": 0} for target_id in C.TARGET_IDS
    }
    for row in target_rows:
        tolerance = support["support_tolerance"]
        inside = all(
            interval[0] - tolerance <= value <= interval[1] + tolerance
            for value, interval in (
                (row["dx_m"], support["body_dx_m"]),
                (row["dy_m"], support["body_dy_m"]),
                (row["distance_m"], support["distance_m"]),
                (row["relative_heading_rad"], support["relative_heading_rad"]),
                (row["relative_heading_sin"], support["relative_heading_sin"]),
                (row["relative_heading_cos"], support["relative_heading_cos"]),
            )
        )
        support_counts[row["target_id"]]["inside" if inside else "outside"] += 1
    targets = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "target_contracts.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "panel_manifest_content_digest": panel["content_digest"],
            "edge_port_index_content_digest": ports["content_digest"],
            "target_ids": list(C.TARGET_IDS),
            "feature_order": list(C.TARGET_FEATURE_ORDER),
            "training_contract_binding": copy.deepcopy(C.CURRENT_VISUAL_RANKER_BINDING),
            "original_ranker_support": copy.deepcopy(C.ORIGINAL_RANKER_TRAINING_SUPPORT),
            "contract_difference_inventory": list(
                C.RANKER_CONTRACT_DIFFERENCE_INVENTORY
            ),
            "target_support_counts": support_counts,
            "rows": target_rows,
        }
    )
    documents = {
        "panel_manifest.json": panel,
        "split_manifest.json": split,
        "state_snapshot_index.json": snapshots,
        "teacher_trace_index.json": teachers,
        "edge_port_index.json": ports,
        "target_contracts.json": targets,
    }
    validate_frozen_panel_documents(
        documents,
        records,
        generator,
        handoff,
        qualification_material,
        selected_material,
        runtime,
        allow_fake_runtime=allow_fake_runtime,
        _rebuilt=documents,
    )
    return documents


def validate_frozen_panel_documents(
    documents: Mapping[str, Any],
    terminal_records: Sequence[Mapping[str, Any]],
    generator_metrics: Mapping[str, Any],
    panel_handoff: Mapping[str, Any],
    qualification_material: Sequence[Mapping[str, Any]],
    selected_material: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool = False,
    _rebuilt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_names = {
        "panel_manifest.json",
        "split_manifest.json",
        "state_snapshot_index.json",
        "teacher_trace_index.json",
        "edge_port_index.json",
        "target_contracts.json",
    }
    if not isinstance(documents, Mapping) or set(documents) != expected_names:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "frozen panel document inventory drift"
        )
    for leaf, fields in (
        ("panel_manifest.json", C.PANEL_MANIFEST_FIELDS),
        ("split_manifest.json", C.SPLIT_MANIFEST_FIELDS),
        ("state_snapshot_index.json", C.STATE_SNAPSHOT_INDEX_FIELDS),
        ("teacher_trace_index.json", C.TEACHER_TRACE_INDEX_FIELDS),
        ("edge_port_index.json", C.EDGE_PORT_INDEX_FIELDS),
        ("target_contracts.json", C.TARGET_CONTRACTS_FIELDS),
    ):
        row = _mapping(documents[leaf], fields, leaf)
        C.validate_content_digest(row)
    panel = documents["panel_manifest.json"]
    split = documents["split_manifest.json"]
    snapshots = documents["state_snapshot_index.json"]
    teachers = documents["teacher_trace_index.json"]
    ports = documents["edge_port_index.json"]
    targets = documents["target_contracts.json"]
    panel_runtime = _mapping(
        panel["physical_runtime_environment"],
        set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
        "panel physical runtime",
    )
    panel_runtime_sha = runtime_environment_sha256(panel_runtime)
    if (
        panel["physical_runtime_environment_sha256"] != panel_runtime_sha
        or panel["selected_reset_runtime_sha256s"]
        != [panel_runtime_sha] * C.PANEL_STATE_COUNT
        or (panel_runtime["fake_runtime"] and not allow_fake_runtime)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel physical runtime projection drift"
        )
    for index, value in enumerate(_sequence(panel["states"], "panel states")):
        state = _mapping(value, C.PANEL_STATE_FIELDS, f"panel state[{index}]")
        _mapping(
            state["eligibility_evidence"],
            C.PANEL_ELIGIBILITY_FIELDS,
            f"panel eligibility[{index}]",
        )
    for index, value in enumerate(
        _sequence(split["assignments"], "split assignments")
    ):
        _mapping(value, C.SPLIT_ASSIGNMENT_FIELDS, f"split assignment[{index}]")
    for index, value in enumerate(
        _sequence(snapshots["records"], "snapshot records")
    ):
        _mapping(value, C.STATE_SNAPSHOT_RECORD_FIELDS, f"snapshot[{index}]")
    for index, value in enumerate(
        _sequence(teachers["records"], "teacher records")
    ):
        teacher = _mapping(
            value, C.TEACHER_TRACE_RECORD_FIELDS, f"teacher[{index}]"
        )
        _mapping(
            teacher["raw_projection"],
            C.V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS,
            f"teacher projection[{index}]",
        )
    for index, value in enumerate(_sequence(ports["records"], "edge ports")):
        _mapping(value, C.EDGE_PORT_RECORD_FIELDS, f"edge port[{index}]")
    for index, value in enumerate(_sequence(targets["rows"], "target rows")):
        _mapping(value, C.TARGET_CONTRACT_ROW_FIELDS, f"target row[{index}]")
    if _rebuilt is None:
        rebuilt = build_frozen_panel_documents(
            terminal_records,
            generator_metrics,
            panel_handoff,
            qualification_material,
            selected_material,
            runtime_contract,
            allow_fake_runtime=allow_fake_runtime,
        )
    else:
        rebuilt = _rebuilt
    if C.canonical_json_bytes(documents) != C.canonical_json_bytes(rebuilt):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "frozen panel documents differ from raw-evidence rebuild"
        )
    return {key: copy.deepcopy(value) for key, value in documents.items()}


def _validate_visual_runtime(
    value: Any, *, role: str, allow_fake_runtime: bool
) -> dict[str, Any]:
    try:
        row = _V1M.validate_visual_runtime_environment(value, runtime_role=role)
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{role} runtime validation failed: {exc}"
        ) from exc
    if row["fake_runtime"] and not allow_fake_runtime:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"production {role} evidence contains fake runtime"
        )
    return copy.deepcopy(row)


def _validate_optional_metadata_binding(
    value: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    binding = _file_binding(value, label)
    raw = C.canonical_json_bytes(metadata)
    if (
        binding["path"] != expected_path
        or binding["bytes"] != len(raw)
        or binding["sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} byte/path binding drift"
        )
    return binding


def validate_encoding_material_shard(
    metadata: Any,
    *,
    reopened_arrays: Mapping[str, Any],
    panel_documents: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
    material_metadata_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate canonical singleton encoding material from raw arrays."""

    import numpy as np

    row = _mapping(
        metadata, C.CANONICAL_ENCODING_MATERIAL_FIELDS, "encoding metadata"
    )
    C.validate_content_digest(row)
    runtime = C.validate_runtime_contract(runtime_contract)
    if (
        row["schema"] != C.CANONICAL_ENCODING_MATERIAL_SCHEMA
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["source_freeze_commit"] != runtime["source_freeze_commit"]
        or row["runtime_contract_content_digest"] != runtime["content_digest"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding identity/runtime drift"
        )
    if not isinstance(panel_documents, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding panel documents are absent"
        )
    panel = _mapping(
        panel_documents.get("panel_manifest.json"),
        C.PANEL_MANIFEST_FIELDS,
        "encoding panel manifest",
    )
    split = _mapping(
        panel_documents.get("split_manifest.json"),
        C.SPLIT_MANIFEST_FIELDS,
        "encoding split manifest",
    )
    snapshots = _mapping(
        panel_documents.get("state_snapshot_index.json"),
        C.STATE_SNAPSHOT_INDEX_FIELDS,
        "encoding snapshot index",
    )
    for document in (panel, split, snapshots):
        C.validate_content_digest(document)
    panel_bytes = C.canonical_json_bytes(panel)
    split_bytes = C.canonical_json_bytes(split)
    _validate_exact_canonical_file_binding(
        row["panel_manifest_binding"],
        path="panel_manifest.json",
        raw=panel_bytes,
        label="encoding panel binding",
    )
    _validate_exact_canonical_file_binding(
        row["split_manifest_binding"],
        path="split_manifest.json",
        raw=split_bytes,
        label="encoding split binding",
    )
    persisted = validate_persisted_array_evidence(
        row["persisted_array_evidence"], reopened_arrays=reopened_arrays
    )
    shard_id = "downstream_workspace/encoding"
    _validate_successor_payload_binding(
        row,
        persisted,
        shard_kind="canonical_encoding_material",
        shard_id=shard_id,
    )
    if set(reopened_arrays) != {
        "raw_tokens",
        "spatial_descriptors",
        "pixel_sha256_bytes",
    }:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding array inventory drift"
        )
    state_rows = [
        _mapping(value, C.ENCODING_STATE_ROW_FIELDS, f"encoding state[{index}]")
        for index, value in enumerate(_sequence(row["state_rows"], "encoding states"))
    ]
    encoding_rows = [
        _mapping(value, C.ENCODING_ROW_FIELDS, f"encoding row[{index}]")
        for index, value in enumerate(_sequence(row["encoding_rows"], "encoding rows"))
    ]
    panel_states = _sequence(panel["states"], "encoding panel states")
    snapshot_by_state = {
        value["state_id"]: value
        for value in _sequence(snapshots["records"], "encoding snapshots")
    }
    expected_state_rows = [
        {
            "panel_index": state["panel_index"],
            "state_id": state["state_id"],
            "qualification_candidate_index": state[
                "qualification_candidate_index"
            ],
            "pixel_sha256": snapshot_by_state[state["state_id"]]["rgb_sha256"],
            "canonical_pixel_index": None,
        }
        for state in panel_states
    ]
    pixel_order = _sequence(row["pixel_order"], "canonical pixel order")
    expected_pixel_order = sorted(
        {snapshot_by_state[state["state_id"]]["rgb_sha256"] for state in panel_states}
    )
    if pixel_order != expected_pixel_order or pixel_order != sorted(set(pixel_order)) or any(
        _sha(value, "pixel SHA-256") != value for value in pixel_order
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "canonical pixel order drift"
        )
    pixel_index = {value: index for index, value in enumerate(pixel_order)}
    for expected in expected_state_rows:
        expected["canonical_pixel_index"] = pixel_index[expected["pixel_sha256"]]
    if state_rows != expected_state_rows:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding state/pixel projection drift"
        )
    unique_count = _nonnegative_int(row["unique_pixel_count"], "unique pixel count")
    raw_tokens = np.asarray(reopened_arrays["raw_tokens"])
    descriptors = np.asarray(reopened_arrays["spatial_descriptors"])
    pixel_bytes = np.asarray(reopened_arrays["pixel_sha256_bytes"])
    if (
        unique_count != len(pixel_order)
        or len(encoding_rows) != unique_count
        or raw_tokens.dtype.str != "<f2"
        or list(raw_tokens.shape) != [unique_count, 768, 1024]
        or descriptors.dtype.str != "<f4"
        or list(descriptors.shape) != [unique_count, 768, 1024]
        or pixel_bytes.dtype.str != "|u1"
        or list(pixel_bytes.shape) != [unique_count, 32]
        or not all(
            value.flags.c_contiguous for value in (raw_tokens, descriptors, pixel_bytes)
        )
        or not bool(np.isfinite(raw_tokens).all())
        or not bool(np.isfinite(descriptors).all())
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding array extent/value drift"
        )
    for index, item in enumerate(encoding_rows):
        if (
            item["canonical_pixel_index"] != index
            or item["pixel_sha256"] != pixel_order[index]
            or bytes(pixel_bytes[index]).hex() != pixel_order[index]
            or item["raw_token_sha256"]
            != _canonical_array_sha256_local(raw_tokens[index])
            or item["spatial_descriptor_sha256"]
            != _canonical_array_sha256_local(descriptors[index])
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "encoding raw row binding drift"
            )
        _sha(item["preprocessed_tensor_sha256"], "preprocessed tensor SHA-256")
    visual_runtime = _validate_visual_runtime(
        row["encoder_runtime_environment"],
        role="encoder",
        allow_fake_runtime=allow_fake_runtime,
    )
    source = C.CANONICAL_ENCODING_AUTHORITY["external_encoder_source"]
    if (
        row["encoder_binding"] != C.VJEPA_ENCODER_BINDING
        or row["preprocessing_authority"]
        != C.CANONICAL_ENCODING_AUTHORITY["preprocessing"]
        or row["external_encoder_source"]
        != {
            "repository_path": source["repository_path"],
            "commit": source["commit"],
            "worktree_clean": True,
        }
        or visual_runtime["checkpoint_sha256"]
        != C.FROZEN_ENCODER_CHECKPOINT_SHA256
        or row["batch_size"] != 1
        or row["fanout_outcomes_opened"] != 0
        or row["models_trained"] != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding frozen model/stage boundary drift"
        )
    metadata_binding = _validate_optional_metadata_binding(
        material_metadata_binding,
        row,
        expected_path=f"{shard_id}/metadata.json",
        label="encoding material metadata binding",
    )
    return {
        "metadata": row,
        "arrays": {name: reopened_arrays[name] for name in reopened_arrays},
        "material_metadata_binding": metadata_binding,
        "material_payload_binding": persisted["payload_file"],
        "encoder_runtime_environment": visual_runtime,
    }


def _candidate_requested_commands(candidate_index: int) -> list[list[float]]:
    """Expand one frozen bank member to the exact fifteen requested ticks."""

    index = _nonnegative_int(candidate_index, "branch candidate index")
    if index >= len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "branch candidate index outside frozen bank"
        )
    candidate_id, primitives = C.CANDIDATE_BANK[index]
    if candidate_id != C.CANDIDATE_IDS[index]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate bank/order authority drift"
        )
    result: list[list[float]] = []
    for primitive in primitives[:3]:
        command = [float(value) for value in C.CANDIDATE_PRIMITIVES[primitive]]
        result.extend([command] * int(C.COMMAND_TICKS_PER_BLOCK))
    if len(result) != 15:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate requested-command extent drift"
        )
    return result


def _expected_candidate_command_tapes(
    previous_applied_command: Any, candidate_index: int,
) -> tuple[Any, Any, list[list[float]], list[list[float]]]:
    """Rebuild the frozen float32 request/clip/slew tape for one bank branch."""

    import numpy as np

    nominal = _candidate_requested_commands(candidate_index)
    previous = np.asarray(previous_applied_command, dtype=np.float32).reshape(3)
    lower = np.asarray([-0.3, 0.0, -0.5], dtype=np.float32)
    upper = np.asarray([0.3, 0.0, 0.5], dtype=np.float32)
    delta = np.asarray([0.25, 0.0, 0.35], dtype=np.float32)
    requested_ticks: list[Any] = []
    applied_ticks: list[Any] = []
    for raw in nominal:
        requested = np.asarray(raw, dtype=np.float32)
        clipped = np.clip(requested, lower, upper).astype(np.float32)
        applied = np.clip(clipped, previous - delta, previous + delta).astype(
            np.float32
        )
        requested_ticks.append(requested.copy())
        applied_ticks.append(applied.copy())
        previous = applied
    requested_rows = [[float(value) for value in row] for row in requested_ticks]
    applied_rows = [[float(value) for value in row] for row in applied_ticks]
    requested = np.ascontiguousarray(
        np.repeat(np.asarray(requested_ticks, dtype=np.float32), 50, axis=0),
        dtype=np.float64,
    )
    applied = np.ascontiguousarray(
        np.repeat(np.asarray(applied_ticks, dtype=np.float32), 50, axis=0),
        dtype=np.float64,
    )
    return requested, applied, nominal, applied_rows


def _normalise_candidate_trace(
    arrays: Mapping[str, Any], *, prefix: str, selected_metadata: Mapping[str, Any],
    selected_arrays: Mapping[str, Any], candidate_index: int,
    candidate_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every raw member and its exact frozen time/command/region tape."""

    import numpy as np

    trace: dict[str, Any] = {}
    binary = {
        "physics_contact",
        "source_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    for member in C.CANDIDATE_TRACE_MEMBER_ORDER:
        name = f"{prefix}__{member}"
        if name not in arrays:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"candidate trace member absent: {name}"
            )
        value = np.asarray(arrays[name])
        expected_shape = [C.RESET_FIXTURE_PHYSICS_SAMPLES, *C.CANDIDATE_TRACE_MEMBER_SHAPES[member]]
        expected_dtype = "|u1" if member in binary else "<f8"
        if (
            list(value.shape) != expected_shape
            or value.dtype.str != expected_dtype
            or not value.flags.c_contiguous
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"candidate trace member shape/dtype/layout drift: {name}"
            )
        if member in binary:
            if not bool(np.isin(value, [0, 1]).all()):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"candidate indicator is not binary: {name}"
                )
        elif not bool(np.isfinite(value).all()):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"candidate floating trace member is not finite: {name}"
            )
        trace[member] = value
    capture = selected_metadata["snapshot"]["capture_timestamp_s"]
    expected_time = _expected_reset_timestamp_tape(capture)
    if not np.array_equal(trace["timestamp_s"], expected_time):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate timestamp tape drift"
        )
    previous = selected_arrays["snapshot__previous_applied_command"]
    requested, applied, _, _ = _expected_candidate_command_tapes(
        previous, candidate_index
    )
    if not np.array_equal(trace["requested_command"], requested):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate requested-command tape drift"
        )
    if not np.array_equal(trace["post_slew_applied_command"], applied):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate post-slew command tape drift"
        )
    geometry = candidate_spec["geometry"]
    selected_edge = geometry["selected_directed_edge"]
    source_polygon = geometry["source_node"]["boundary_polygon_world"]
    target_polygon = geometry["target_node"]["boundary_polygon_world"]
    correct_polygon = selected_edge["edge_region_polygon_world"]
    wrong_polygons = [
        edge["edge_region_polygon_world"]
        for edge in geometry["competing_directed_edges"]
    ]
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    expected_regions = {
        "source_region_member": [],
        "correct_edge_region_member": [],
        "wrong_edge_region_member": [],
        "target_region_member": [],
    }
    for pose in trace["base_pose_world"]:
        point = [float(pose[0]), float(pose[1])]
        expected_regions["source_region_member"].append(
            int(point_in_polygon_inclusive(point, source_polygon, tolerance_m=tolerance))
        )
        expected_regions["correct_edge_region_member"].append(
            int(point_in_polygon_inclusive(point, correct_polygon, tolerance_m=tolerance))
        )
        expected_regions["wrong_edge_region_member"].append(
            int(any(point_in_polygon_inclusive(point, polygon, tolerance_m=tolerance) for polygon in wrong_polygons))
        )
        expected_regions["target_region_member"].append(
            int(point_in_polygon_inclusive(point, target_polygon, tolerance_m=tolerance))
        )
    for member, values in expected_regions.items():
        expected = np.asarray(values, dtype=np.uint8)
        if not np.array_equal(trace[member], expected):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"candidate raw geometric indicator drift: {member}"
            )
    return trace


def _registered_crossing_for_edge(
    poses: Any, edge: Mapping[str, Any]
) -> dict[str, Any] | None:
    raw = first_registered_port_crossing(
        [[float(value[0]), float(value[1])] for value in poses],
        edge,
        [],
        tolerance_m=float(C.NUMERICAL_TOLERANCES["se2_position_m"]),
    )
    if raw is None:
        return None
    return {
        "edge_id": str(raw["edge_id"]),
        "is_selected_edge": True,
        "sample_before": int(raw["crossing_sample_before"]),
        "sample_after": int(raw["crossing_sample_after"]),
        "fraction": float(raw["crossing_fraction"]),
        "normal_dot_displacement_m": float(raw["directed_normal_displacement_m"]),
        "lateral_fraction": float(raw["lateral_fraction"]),
    }


def _first_competing_crossing_native(
    poses: Any, edges: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for edge in edges:
        item = _registered_crossing_for_edge(poses, edge)
        if item is not None:
            item["edge_id"] = str(edge["edge_id"])
            item["is_selected_edge"] = False
            candidates.append(item)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["sample_after"], item["fraction"], item["edge_id"]
        ),
    )


def _body_from_world_candidate(
    world_pose: Sequence[float], body_pose: Sequence[float]
) -> list[float]:
    wx, wy, wyaw = (float(value) for value in world_pose)
    bx, by, byaw = (float(value) for value in body_pose)
    cosine, sine = math.cos(byaw), math.sin(byaw)
    dx, dy = wx - bx, wy - by
    return [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        _wrap_angle_v1(wyaw - byaw),
    ]


def _command_tracking_rows_native(
    trace: Mapping[str, Any], requested: Sequence[Sequence[float]],
    applied: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    import numpy as np

    result: list[dict[str, Any]] = []
    threshold = float(C.COMMAND_TRACKING_AUTHORITY["active_command_threshold"])
    for tick in range(15):
        values = []
        for index in range(tick * 50 + 20, tick * 50 + 50):
            yaw = _pose_roll_pitch_yaw_xyzw(trace["base_pose_world"][index])[2]
            vxw, vyw = (
                float(value) for value in trace["base_twist_world"][index, :2]
            )
            values.append(
                [
                    math.cos(yaw) * vxw + math.sin(yaw) * vyw,
                    -math.sin(yaw) * vxw + math.cos(yaw) * vyw,
                    float(trace["base_twist_world"][index, 5]),
                ]
            )
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        post_slew = [float(value) for value in applied[tick]]
        result.append(
            {
                "command_tick_index": tick,
                "requested_command": [float(value) for value in requested[tick]],
                "post_slew_command": post_slew,
                "mean_achieved_body_velocity": [float(value) for value in mean],
                "active_vx": abs(post_slew[0]) > threshold,
                "active_yaw": abs(post_slew[2]) > threshold,
            }
        )
    return result


def _consecutive_beyond_samples_native(
    poses: Any,
    crossing_after: int,
    opening: Sequence[Sequence[float]],
    normal: Sequence[float],
    target_member: Any,
) -> tuple[int, bool]:
    """Exact frozen-V1 NumPy binary64 beyond-port reduction."""

    import numpy as np

    midpoint = np.asarray(opening, dtype=np.float64).mean(axis=0)
    direction = np.asarray(normal, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    beyond = (
        np.asarray(poses, dtype=np.float64)[crossing_after:, :2] - midpoint
    ) @ direction
    epsilon = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    count = 0
    for value in beyond:
        if float(value) <= epsilon:
            break
        count += 1
    target = np.asarray(target_member, dtype=np.uint8)[crossing_after:]
    hits = np.flatnonzero(target != 0)
    target_early = bool(len(hits) and int(hits[0]) + 1 < 100)
    return int(count), target_early


def _derive_candidate_outcome_native(
    candidate_spec: Mapping[str, Any], trace: Mapping[str, Any],
    reset_pose_world: Sequence[float], candidate_index: int,
    directed_port_world: Sequence[float],
) -> dict[str, Any]:
    """Derive every frozen V4 outcome from the 750 reopened samples."""

    import numpy as np

    geometry = candidate_spec["geometry"]
    selected_edge = geometry["selected_directed_edge"]
    competitors = geometry["competing_directed_edges"]
    poses = trace["base_pose_world"]
    correct_crossing = _registered_crossing_for_edge(poses, selected_edge)
    wrong_crossing = _first_competing_crossing_native(poses, competitors)
    registered = first_registered_port_crossing(
        [[float(value[0]), float(value[1])] for value in poses],
        selected_edge,
        competitors,
        tolerance_m=float(C.NUMERICAL_TOLERANCES["se2_position_m"]),
    )
    correct_first = bool(registered is not None and registered["is_selected_edge"])
    dwell = 0
    target_early = False
    if correct_first and correct_crossing is not None:
        after = int(correct_crossing["sample_after"])
        dwell, target_early = _consecutive_beyond_samples_native(
            poses,
            after,
            selected_edge["opening_segment_world"],
            selected_edge["opening_normal_world"],
            trace["target_region_member"],
        )
    entered_correct = bool(correct_first and (dwell >= 100 or target_early))
    entered_wrong = bool(
        not entered_correct and registered is not None and not registered["is_selected_edge"]
    )
    start_se2 = [
        float(reset_pose_world[0]),
        float(reset_pose_world[1]),
        _pose_roll_pitch_yaw_xyzw(reset_pose_world)[2],
    ]
    endpoints_world = [
        [float(poses[index, 0]), float(poses[index, 1]), _pose_roll_pitch_yaw_xyzw(poses[index])[2]]
        for index in (249, 499, 749)
    ]
    endpoints_body = [
        _body_from_world_candidate(value, start_se2) for value in endpoints_world
    ]
    aligned = derive_candidate_port_metrics(
        reset_pose_world, poses, directed_port_world
    )
    # Frozen V1 angular error is relative to the normalized selected-opening
    # normal, not the teacher's interpolated crossing yaw carried by the
    # directed-port record.
    target_normal = np.asarray(
        selected_edge["opening_normal_world"], dtype=np.float64
    )
    target_normal /= np.linalg.norm(target_normal)
    target_heading = math.atan2(
        float(target_normal[1]), float(target_normal[0])
    )
    angular_error = abs(_wrap_angle_v1(endpoints_world[-1][2] - target_heading))
    roll, pitch, _ = _pose_roll_pitch_yaw_xyzw(poses[-1])
    finite = bool(all(np.isfinite(value).all() for value in trace.values()))
    physics_contact = bool(trace["physics_contact"].any())
    final_contact = bool(trace["physics_contact"][-1])
    successor = bool(
        finite
        and float(poses[-1, 2])
        >= float(C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["minimum_base_height_m"])
        and abs(roll)
        <= float(C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_roll_rad"])
        and abs(pitch)
        <= float(C.PHYSICAL_OUTCOME_AUTHORITY["successor_viable"]["maximum_absolute_pitch_rad"])
        and not final_contact
    )
    requested_rows = _candidate_requested_commands(candidate_index)
    applied_rows = [
        [float(value) for value in trace["post_slew_applied_command"][tick * 50]]
        for tick in range(15)
    ]
    # Preserve the frozen V1 outcome formula: command activity is taken from
    # the nominal Python candidate-bank rows.  The independently validated raw
    # requested tape is a persistence cross-check, not a replacement salt for
    # this predicate.
    activity = max(
        max(abs(value) for value in command) for command in requested_rows
    )
    stuck = bool(
        activity > float(C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"])
        and math.hypot(endpoints_body[-1][0], endpoints_body[-1][1])
        < float(C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_translation_threshold_m"])
        and abs(endpoints_body[-1][2])
        < float(C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["h3_heading_threshold_rad"])
    )
    oracle_admissible = bool(not physics_contact and finite and successor and not stuck)
    wrong_for_endpoint = wrong_crossing if entered_wrong else None
    if trace["target_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = "target", "selected-edge"
    elif entered_correct or trace["correct_edge_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = None, "selected-edge"
    elif entered_wrong and wrong_for_endpoint is not None:
        endpoint_node_id, endpoint_edge_id = None, str(wrong_for_endpoint["edge_id"])
    elif trace["source_region_member"][-1]:
        endpoint_node_id, endpoint_edge_id = "source", None
    else:
        endpoint_node_id = endpoint_edge_id = None
    source_exits = np.flatnonzero(trace["source_region_member"] == 0)
    target_entries = np.flatnonzero(trace["target_region_member"] != 0)

    def crossing_projection(item: Mapping[str, Any] | None) -> tuple[Any, ...]:
        if item is None:
            return (None,) * 7
        before, after = int(item["sample_before"]), int(item["sample_after"])
        displacement = [
            float(poses[after, axis] - poses[before, axis]) for axis in range(2)
        ]
        return (
            before,
            after,
            float(item["fraction"]),
            float(item["normal_dot_displacement_m"]),
            float(item["lateral_fraction"]),
            displacement,
            math.atan2(displacement[1], displacement[0]),
        )

    correct_evidence = correct_crossing if entered_correct else None
    wrong_evidence = wrong_crossing if entered_wrong else None
    cb, ca, cf, cn, cl, cd, ch = crossing_projection(correct_evidence)
    wb, wa, wf, wn, wl, wd, wh = crossing_projection(wrong_evidence)
    result = {
        "branch_candidate_id": C.CANDIDATE_IDS[candidate_index],
        "branch_candidate_index": candidate_index,
        "requested_commands": requested_rows,
        "post_slew_applied_commands": applied_rows,
        "physics_sample_count": C.RESET_FIXTURE_PHYSICS_SAMPLES,
        "port_crossing_sample_before": cb,
        "port_crossing_sample_after": ca,
        "port_crossing_fraction": cf,
        "port_crossing_directed_normal_dot": cn,
        "port_crossing_lateral_fraction": cl,
        "port_crossing_displacement_world_xy": cd,
        "port_crossing_direction_heading_world_rad": ch,
        "beyond_port_consecutive_physics_samples": int(dwell),
        "target_entered_before_dwell_complete": bool(target_early),
        "competing_port_crossing_first": bool(entered_wrong),
        "first_wrong_edge_id": None if wrong_evidence is None else str(wrong_evidence["edge_id"]),
        "wrong_port_crossing_sample_before": wb,
        "wrong_port_crossing_sample_after": wa,
        "wrong_port_crossing_fraction": wf,
        "wrong_port_crossing_directed_normal_dot": wn,
        "wrong_port_crossing_lateral_fraction": wl,
        "wrong_port_crossing_displacement_world_xy": wd,
        "wrong_port_crossing_direction_heading_world_rad": wh,
        "h1_endpoint_body": endpoints_body[0],
        "h2_endpoint_body": endpoints_body[1],
        "h3_endpoint_body": endpoints_body[2],
        "h3_endpoint_world": endpoints_world[2],
        "h3_base_height_m": float(poses[-1, 2]),
        "h3_roll_rad": float(roll),
        "h3_pitch_rad": float(pitch),
        "h3_solver_finite": finite,
        "h3_disallowed_contact": final_contact,
        "physics_contact": physics_contact,
        "stuck": stuck,
        "successor_viable": successor,
        "entered_correct_edge": entered_correct,
        "entered_wrong_edge": entered_wrong,
        "no_edge": not entered_correct and not entered_wrong,
        "port_progress_m": float(aligned["port_progress_m"]),
        "lateral_error_m": float(aligned["lateral_error_m"]),
        "angular_error_rad": angular_error,
        "oracle_admissible": oracle_admissible,
        "positive_port_progress": bool(aligned["positive_port_progress"]),
        "endpoint_node_id": endpoint_node_id,
        "endpoint_edge_id": endpoint_edge_id,
        "left_source_region": bool(len(source_exits)),
        "first_source_exit_sample_index": int(source_exits[0]) if len(source_exits) else None,
        "reached_target_node": bool(len(target_entries)),
        "first_target_entry_sample_index": int(target_entries[0]) if len(target_entries) else None,
        "command_tracking_rows": _command_tracking_rows_native(trace, requested_rows, applied_rows),
    }
    if set(result) != set(C.CANDIDATE_OUTCOME_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate outcome native field drift"
        )
    return result


def _require_semantic_equal(
    supplied: Any, expected: Any, label: str, *, tolerance: float | None = None,
) -> None:
    """Compare an evidence tree exactly except for frozen summary float tolerance."""

    if isinstance(expected, Mapping):
        if not isinstance(supplied, Mapping) or set(supplied) != set(expected):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} mapping drift"
            )
        for key in expected:
            _require_semantic_equal(
                supplied[key], expected[key], f"{label}.{key}", tolerance=tolerance
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        if isinstance(supplied, (str, bytes, bytearray)) or not isinstance(
            supplied, Sequence
        ) or len(supplied) != len(expected):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} sequence drift"
            )
        for index, (left, right) in enumerate(zip(supplied, expected)):
            _require_semantic_equal(
                left, right, f"{label}[{index}]", tolerance=tolerance
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if supplied != expected or type(supplied) is not type(expected):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} drift"
            )
        return
    if isinstance(expected, int):
        if not isinstance(supplied, int) or isinstance(supplied, bool) or supplied != expected:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} integer drift"
            )
        return
    if isinstance(expected, float):
        observed = _finite(supplied, label)
        limit = (
            float(C.NUMERICAL_TOLERANCES["trace_summary_float"])
            if tolerance is None
            else float(tolerance)
        )
        if abs(observed - expected) > limit:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{label} numeric drift"
            )
        return
    if supplied != expected:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} drift"
        )


def _validated_selected_wrapper(
    selected_material: Any, candidate_spec: Mapping[str, Any], panel_index: int,
) -> dict[str, Any]:
    if not isinstance(selected_material, Mapping) or not {
        "metadata", "arrays", "material_metadata_binding"
    }.issubset(selected_material):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "selected material validation wrapper drift"
        )
    return validate_selected_reset_material_shard(
        selected_material["metadata"],
        reopened_arrays=selected_material["arrays"],
        selected_candidate_spec=candidate_spec,
        expected_panel_index=panel_index,
        material_metadata_binding=selected_material["material_metadata_binding"],
    )


def validate_fanout_material_shard(
    metadata: Any,
    *,
    reopened_arrays: Mapping[str, Any],
    panel_state: Mapping[str, Any],
    candidate_spec: Mapping[str, Any],
    selected_material: Mapping[str, Any],
    directed_port_record: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
    material_metadata_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one twelve-branch shard from its complete reopened raw arrays."""

    panel = _mapping(panel_state, C.PANEL_STATE_FIELDS, "fanout panel state")
    spec = C.validate_candidate_spec(candidate_spec)
    panel_index = _nonnegative_int(panel["panel_index"], "fanout panel index")
    if (
        panel_index >= C.PANEL_STATE_COUNT
        or panel["state_id"] != spec["state_id"]
        or panel["qualification_candidate_index"] != spec["candidate_index"]
        or panel["candidate_spec_id"] != spec["candidate_spec_id"]
        or panel["canonical_spec_sha256"] != spec["canonical_spec_sha256"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout panel/spec identity drift"
        )
    selected = _validated_selected_wrapper(selected_material, spec, panel_index)
    smeta, sarrays = selected["metadata"], selected["arrays"]
    port = _mapping(
        directed_port_record, C.EDGE_PORT_RECORD_FIELDS, "fanout directed port"
    )
    if (
        port["panel_index"] != panel_index
        or port["qualification_candidate_index"] != spec["candidate_index"]
        or port["state_id"] != spec["state_id"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout directed-port identity drift"
        )
    runtime = C.validate_runtime_contract(runtime_contract)
    row = _mapping(metadata, C.CANDIDATE_FANOUT_MATERIAL_FIELDS, "fanout material")
    C.validate_content_digest(row)
    expected_scalar = {
        "schema": C.CANDIDATE_FANOUT_MATERIAL_SCHEMA,
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": runtime["source_freeze_commit"],
        "runtime_contract_content_digest": runtime["content_digest"],
        "panel_index": panel_index,
        "qualification_candidate_index": spec["candidate_index"],
        "state_id": spec["state_id"],
        "family": spec["family"],
        "role": panel["role"],
        "candidate_spec_id": spec["candidate_spec_id"],
        "canonical_spec_sha256": spec["canonical_spec_sha256"],
        "source_selected_metadata_binding": selected["material_metadata_binding"],
        "source_selected_payload_binding": selected["material_payload_binding"],
        "snapshot_payload_sha256": smeta["snapshot_payload_sha256"],
        "directed_port_world": port["directed_port_world"],
        "branch_count": len(C.CANDIDATE_IDS),
        "models_trained": 0,
    }
    for field, expected in expected_scalar.items():
        if row[field] != expected:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"fanout {field} drift"
            )
    if panel["role"] == "DEVELOPMENT":
        if (
            row["development_target_selection_binding"] is not None
            or row["development_target_selection_opened"] is not False
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "development fanout opened target selection"
            )
    else:
        selection_binding = _file_binding(
            row["development_target_selection_binding"],
            "heldout fanout development-target binding",
        )
        if (
            selection_binding["path"] != "development_target_selection.json"
            or row["development_target_selection_opened"] is not True
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "heldout fanout target-selection timing drift"
            )
    persisted = validate_persisted_array_evidence(
        row["persisted_array_evidence"], reopened_arrays=reopened_arrays
    )
    shard_id = f"fanout/{spec['state_id']}"
    _validate_successor_payload_binding(
        row, persisted, shard_kind="candidate_fanout_material", shard_id=shard_id
    )
    expected_names = {
        f"candidate_{index:02d}__{member}"
        for index in range(len(C.CANDIDATE_IDS))
        for member in C.CANDIDATE_TRACE_MEMBER_ORDER
    }
    if set(reopened_arrays) != expected_names:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout raw array inventory drift"
        )
    outcomes = _sequence(row["outcome_rows"], "fanout outcome rows")
    if len(outcomes) != len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout outcome cardinality drift"
        )
    traces: list[dict[str, Any]] = []
    rebuilt: list[dict[str, Any]] = []
    for index in range(len(C.CANDIDATE_IDS)):
        trace = _normalise_candidate_trace(
            reopened_arrays,
            prefix=f"candidate_{index:02d}",
            selected_metadata=smeta,
            selected_arrays=sarrays,
            candidate_index=index,
            candidate_spec=spec,
        )
        expected = _derive_candidate_outcome_native(
            spec,
            trace,
            sarrays["snapshot__base_pose_world"],
            index,
            port["directed_port_world"],
        )
        supplied = _mapping(
            outcomes[index], C.CANDIDATE_OUTCOME_FIELDS, f"fanout outcome[{index}]"
        )
        _require_semantic_equal(supplied, expected, f"fanout outcome[{index}]")
        traces.append(trace)
        rebuilt.append(expected)
    stage = _mapping(
        row["stage_runtime"], set(C.PHYSICAL_RUNTIME_CORE_FIELDS), "fanout runtime"
    )
    backend = _mapping(
        row["backend_runtime"],
        set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS),
        "fanout backend runtime",
    )
    if stage != smeta["stage_runtime"] or backend != smeta["backend_runtime"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout physical runtime drift"
        )
    if stage["fake_runtime"] and not allow_fake_runtime:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "production fanout contains fake runtime"
        )
    metadata_binding = _validate_optional_metadata_binding(
        material_metadata_binding,
        row,
        expected_path=f"{shard_id}/metadata.json",
        label="fanout material metadata binding",
    )
    return {
        "metadata": row,
        "arrays": {name: reopened_arrays[name] for name in reopened_arrays},
        "traces": traces,
        "rebuilt_outcome_rows": rebuilt,
        "material_metadata_binding": metadata_binding,
        "material_payload_binding": persisted["payload_file"],
        "physical_runtime_core_sha256": runtime_environment_sha256(stage),
    }


def _correct_candidate(row: Mapping[str, Any]) -> bool:
    return bool(
        row["oracle_admissible"]
        and row["entered_correct_edge"]
        and row["successor_viable"]
        and row["positive_port_progress"]
        and not row["physics_contact"]
        and not row["stuck"]
    )


def _rank_scores(scores: Sequence[Any]) -> list[int]:
    values = [_finite(value, "candidate score") for value in _sequence(scores, "scores")]
    if len(values) != len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate score cardinality drift"
        )
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def _oracle_ranking(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(
        range(len(C.CANDIDATE_IDS)),
        key=lambda index: (
            0 if _correct_candidate(rows[index]) else 1,
            -float(rows[index]["port_progress_m"]),
            float(rows[index]["lateral_error_m"]),
            float(rows[index]["angular_error_rad"]),
            index,
        ),
    )


def _kinematic_scores(
    rows: Sequence[Mapping[str, Any]], target_body: Sequence[Any]
) -> list[float]:
    target = _vector(target_body, 3, "kinematic target body pose")
    start_distance = math.hypot(target[0], target[1])
    result: list[float] = []
    for row in rows:
        x = y = yaw = 0.0
        for vx, vy, yaw_rate in row["post_slew_applied_commands"]:
            x += 0.1 * (float(vx) * math.cos(yaw) - float(vy) * math.sin(yaw))
            y += 0.1 * (float(vx) * math.sin(yaw) + float(vy) * math.cos(yaw))
            yaw = _wrap_angle_v1(yaw + 0.1 * float(yaw_rate))
        result.append(start_distance - math.hypot(target[0] - x, target[1] - y))
    return result


def _selection_metrics(
    rows: Sequence[Mapping[str, Any]], ranking: Sequence[Any],
    scores: Sequence[Any] | None,
) -> dict[str, Any]:
    order = [_nonnegative_int(value, "ranking index") for value in ranking]
    if sorted(order) != list(range(len(C.CANDIDATE_IDS))):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate ranking is not a permutation"
        )
    values = None if scores is None else [
        _finite(value, "candidate score") for value in scores
    ]
    if values is not None and len(values) != len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate score cardinality drift"
        )
    selected = order[0]
    correct = [index for index, row in enumerate(rows) if _correct_candidate(row)]
    if not any(row["oracle_admissible"] for row in rows):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout state has no oracle-admissible candidate"
        )
    first_rank = next(
        (rank for rank, index in enumerate(order, 1) if index in correct), None
    )
    incorrect = [index for index in range(len(C.CANDIDATE_IDS)) if index not in correct]
    positions = {index: rank for rank, index in enumerate(order)}
    credits = []
    for left in correct:
        for right in incorrect:
            if values is None:
                credits.append(float(positions[left] < positions[right]))
            else:
                credits.append(
                    1.0
                    if values[left] > values[right]
                    else 0.5
                    if values[left] == values[right]
                    else 0.0
                )
    admissible_progress = [
        float(row["port_progress_m"]) for row in rows if row["oracle_admissible"]
    ]
    best, minimum = max(admissible_progress), min(admissible_progress)
    selected_row = rows[selected]
    span = best - minimum
    regret = (
        0.0
        if span <= float(C.NUMERICAL_TOLERANCES["division_floor"])
        else min(
            1.0,
            max(0.0, (best - float(selected_row["port_progress_m"])) / span),
        )
    )
    return {
        "candidate_ids": list(C.CANDIDATE_IDS),
        "scores": values,
        "ranking": order,
        "eligible_correct_edge_candidate_indices": correct,
        "selected_candidate_index": selected,
        "correct_edge_top1": selected in correct,
        "correct_edge_top3": any(value in correct for value in order[:3]),
        "correct_edge_mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "selected_correct_edge_execution": bool(selected_row["entered_correct_edge"]),
        "selected_port_progress_m": float(selected_row["port_progress_m"]),
        "oracle_best_port_progress_m": best,
        "minimum_admissible_port_progress_m": minimum,
        "normalized_port_regret": regret,
        "pairwise_correct_edge_ordering": (
            math.fsum(credits) / len(credits) if credits else 0.0
        ),
        "selected_wrong_edge": bool(selected_row["entered_wrong_edge"]),
        "selected_no_edge": bool(selected_row["no_edge"]),
        "selected_lateral_error_m": float(selected_row["lateral_error_m"]),
        "selected_angular_error_rad": float(selected_row["angular_error_rad"]),
        "selected_contact": bool(selected_row["physics_contact"]),
        "selected_stuck": bool(selected_row["stuck"]),
        "selected_successor_viable": bool(selected_row["successor_viable"]),
    }


def _target_summary(
    rows: Sequence[Mapping[str, Any]], target_id: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["target_id"] == target_id]
    if len(selected) != 48:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development target summary state cardinality drift"
        )

    def mean(field: str) -> float:
        return math.fsum(float(row[field]) for row in selected) / len(selected)

    result = {
        "target_id": target_id,
        "state_count": len(selected),
        "selected_correct_edge_execution_rate": mean("selected_correct_edge_execution"),
        "correct_edge_top3_rate": mean("correct_edge_top3"),
        "correct_edge_top1_rate": mean("correct_edge_top1"),
        "correct_edge_mrr": mean("correct_edge_mrr"),
        "normalized_port_regret": mean("normalized_port_regret"),
        "mean_selected_port_progress_m": mean("selected_port_progress_m"),
        "mean_target_transform_error_m": mean("target_transform_error_m"),
        "pairwise_correct_edge_ordering": mean("pairwise_correct_edge_ordering"),
        "selected_wrong_edge_rate": mean("selected_wrong_edge"),
        "selected_no_edge_rate": mean("selected_no_edge"),
        "mean_selected_lateral_error_m": mean("selected_lateral_error_m"),
        "mean_selected_angular_error_rad": mean("selected_angular_error_rad"),
        "selected_contact_rate": mean("selected_contact"),
        "selected_stuck_rate": mean("selected_stuck"),
        "selected_successor_viable_rate": mean("selected_successor_viable"),
    }
    result["selection_key"] = [
        -result["selected_correct_edge_execution_rate"],
        -result["correct_edge_top3_rate"],
        result["normalized_port_regret"],
        -result["mean_selected_port_progress_m"],
        result["mean_target_transform_error_m"],
        C.TARGET_IDS.index(target_id),
    ]
    return result


def _fanout_projection(
    fanout_material: Sequence[Mapping[str, Any]], *, role: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for wrapper in sorted(
        fanout_material, key=lambda item: int(item["metadata"]["panel_index"])
    ):
        metadata = wrapper["metadata"]
        if role is not None and metadata["role"] != role:
            continue
        if wrapper.get("material_metadata_binding") is None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "fanout projection lacks metadata binding"
            )
        rows.append(
            {
                "panel_index": metadata["panel_index"],
                "qualification_candidate_index": metadata[
                    "qualification_candidate_index"
                ],
                "state_id": metadata["state_id"],
                "role": metadata["role"],
                "candidate_spec_id": metadata["candidate_spec_id"],
                "material_metadata_binding": wrapper["material_metadata_binding"],
                "material_payload_binding": wrapper["material_payload_binding"],
                "outcome_rows": metadata["outcome_rows"],
            }
        )
    return rows


def _validate_panel_document_envelopes(
    documents: Mapping[str, Any], *, allow_fake_runtime: bool
) -> dict[str, Any]:
    expected = {
        "panel_manifest.json": C.PANEL_MANIFEST_FIELDS,
        "split_manifest.json": C.SPLIT_MANIFEST_FIELDS,
        "state_snapshot_index.json": C.STATE_SNAPSHOT_INDEX_FIELDS,
        "teacher_trace_index.json": C.TEACHER_TRACE_INDEX_FIELDS,
        "edge_port_index.json": C.EDGE_PORT_INDEX_FIELDS,
        "target_contracts.json": C.TARGET_CONTRACTS_FIELDS,
    }
    if not isinstance(documents, Mapping) or set(documents) != set(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel document envelope inventory drift"
        )
    result = {}
    for name, fields in expected.items():
        row = _mapping(documents[name], fields, name)
        C.validate_content_digest(row)
        if row["experiment_id"] != C.EXPERIMENT_ID:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{name} experiment identity drift"
            )
        result[name] = row
    panel = result["panel_manifest.json"]
    states = [_mapping(row, C.PANEL_STATE_FIELDS, "panel state") for row in panel["states"]]
    if (
        len(states) != C.PANEL_STATE_COUNT
        or [row["panel_index"] for row in states] != list(range(C.PANEL_STATE_COUNT))
        or [row["state_id"] for row in states] != sorted(row["state_id"] for row in states)
        or panel["state_count"] != C.PANEL_STATE_COUNT
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel canonical state population drift"
        )
    runtime = _mapping(
        panel["physical_runtime_environment"],
        set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
        "panel physical runtime",
    )
    if (
        panel["physical_runtime_environment_sha256"]
        != runtime_environment_sha256(runtime)
        or (runtime["fake_runtime"] and not allow_fake_runtime)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "panel runtime envelope drift"
        )
    return result


def _fanout_population(
    panel_documents: Mapping[str, Any], fanout_material: Sequence[Mapping[str, Any]],
    *, expected_role: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    states = [
        dict(row)
        for row in panel_documents["panel_manifest.json"]["states"]
        if expected_role is None or row["role"] == expected_role
    ]
    supplied = _sequence(fanout_material, "fanout material population")
    by_state: dict[str, Mapping[str, Any]] = {}
    for wrapper in supplied:
        if not isinstance(wrapper, Mapping) or not {
            "metadata", "arrays", "material_metadata_binding", "material_payload_binding"
        }.issubset(wrapper):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "fanout validation wrapper drift"
            )
        state_id = wrapper["metadata"].get("state_id")
        if not isinstance(state_id, str) or state_id in by_state:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "fanout wrapper identity duplicated"
            )
        by_state[state_id] = wrapper
    if set(by_state) != {row["state_id"] for row in states}:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "fanout wrapper state population drift"
        )
    for state in states:
        metadata = by_state[state["state_id"]]["metadata"]
        if (
            metadata["panel_index"] != state["panel_index"]
            or metadata["qualification_candidate_index"]
            != state["qualification_candidate_index"]
            or metadata["role"] != state["role"]
            or metadata["family"] != state["family"]
            or len(metadata["outcome_rows"]) != len(C.CANDIDATE_IDS)
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "fanout wrapper/panel projection drift"
            )
    return states, by_state


def build_development_target_selection(
    panel_documents: Mapping[str, Any],
    encoding_material: Mapping[str, Any],
    fanout_material: Sequence[Mapping[str, Any]],
    state_target_rows: Sequence[Mapping[str, Any]],
    target_summaries: Sequence[Mapping[str, Any]],
    ranker_runtime: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    """Build the irrevocable three-target choice from development evidence only."""

    documents = _validate_panel_document_envelopes(
        panel_documents, allow_fake_runtime=allow_fake_runtime
    )
    C.validate_runtime_contract(runtime_contract)
    if not isinstance(encoding_material, Mapping) or not {
        "metadata", "material_metadata_binding", "material_payload_binding",
        "encoder_runtime_environment"
    }.issubset(encoding_material):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development encoding validation wrapper drift"
        )
    encoding = encoding_material["metadata"]
    if (
        encoding_material["material_metadata_binding"] is None
        or encoding["fanout_outcomes_opened"] != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development selection encoding/timing binding drift"
        )
    states, fanout_by_state = _fanout_population(
        documents, fanout_material, expected_role="DEVELOPMENT"
    )
    if len(states) != 48:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development state cardinality drift"
        )
    targets = documents["target_contracts.json"]
    target_by_identity = {
        (row["state_id"], row["target_id"]): row for row in targets["rows"]
    }
    expected_identity_order = [
        (state["state_id"], target_id)
        for state in states
        for target_id in C.TARGET_IDS
    ]
    supplied_rows = [
        _mapping(row, C.STATE_TARGET_ROW_FIELDS, f"development target row[{index}]")
        for index, row in enumerate(_sequence(state_target_rows, "development target rows"))
    ]
    if [
        (row["state_id"], row["target_id"]) for row in supplied_rows
    ] != expected_identity_order:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development target row order/population drift"
        )
    rebuilt_rows: list[dict[str, Any]] = []
    for supplied, (state_id, target_id) in zip(supplied_rows, expected_identity_order):
        state = next(row for row in states if row["state_id"] == state_id)
        outcomes = fanout_by_state[state_id]["metadata"]["outcome_rows"]
        scores = [_finite(value, "development ranker score") for value in supplied["scores"]]
        ranking = _rank_scores(scores)
        target = target_by_identity[(state_id, target_id)]
        rebuilt = {
            "panel_index": state["panel_index"],
            "qualification_candidate_index": state["qualification_candidate_index"],
            "state_id": state_id,
            "target_id": target_id,
            **_selection_metrics(outcomes, ranking, scores),
            "target_transform_error_m": max(
                float(target["position_transform_error_m"]),
                float(target["heading_transform_error_rad"]),
            ),
        }
        _require_semantic_equal(supplied, rebuilt, "development target row")
        rebuilt_rows.append(rebuilt)
    rebuilt_summaries = [_target_summary(rebuilt_rows, target_id) for target_id in C.TARGET_IDS]
    supplied_summaries = [
        _mapping(row, C.TARGET_SUMMARY_FIELDS, f"target summary[{index}]")
        for index, row in enumerate(_sequence(target_summaries, "target summaries"))
    ]
    _require_semantic_equal(
        supplied_summaries, rebuilt_summaries, "development target summaries"
    )
    selected_index = min(
        range(len(rebuilt_summaries)),
        key=lambda index: tuple(rebuilt_summaries[index]["selection_key"]),
    )
    visual = _validate_visual_runtime(
        ranker_runtime, role="ranker", allow_fake_runtime=allow_fake_runtime
    )
    fanout_projection = _fanout_projection(fanout_material, role="DEVELOPMENT")
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "development_target_selection.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "panel_manifest_content_digest": documents["panel_manifest.json"]["content_digest"],
            "target_contracts_content_digest": targets["content_digest"],
            "candidate_fanout_projection_sha256": _canonical_no_lf_sha256(
                fanout_projection
            ),
            "encoding_material_metadata_binding": copy.deepcopy(
                encoding_material["material_metadata_binding"]
            ),
            "encoding_material_payload_binding": copy.deepcopy(
                encoding_material["material_payload_binding"]
            ),
            "role": "DEVELOPMENT",
            "target_ids": list(C.TARGET_IDS),
            "selection_lexicographic": copy.deepcopy(
                C.DEVELOPMENT_TARGET_SELECTION_LEXICOGRAPHIC
            ),
            "heldout_outcome_documents_opened": 0,
            "state_target_rows": rebuilt_rows,
            "target_summaries": rebuilt_summaries,
            "selected_target_id": C.TARGET_IDS[selected_index],
            "selected_target_index": selected_index,
            "selection_frozen": True,
            "ranker_runtime_environment": visual,
            "models_trained": 0,
        }
    )
    if set(result) != set(C.DEVELOPMENT_TARGET_SELECTION_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development target selection output field drift"
        )
    return result


def validate_development_target_selection(
    value: Any,
    *,
    panel_documents: Mapping[str, Any],
    encoding_material: Mapping[str, Any],
    fanout_material: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    row = _mapping(
        value, C.DEVELOPMENT_TARGET_SELECTION_FIELDS, "development target selection"
    )
    C.validate_content_digest(row)
    expected = build_development_target_selection(
        panel_documents,
        encoding_material,
        fanout_material,
        row["state_target_rows"],
        row["target_summaries"],
        row["ranker_runtime_environment"],
        runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "development target selection differs from raw-evidence rebuild"
        )
    return row


def _teacher_heldout_row(
    state: Mapping[str, Any], teacher: Mapping[str, Any], target_id: str,
    ranker_runtime_sha256: str,
) -> dict[str, Any]:
    projection = teacher["raw_projection"]
    return {
        "score_row_id": f"{state['state_id']}::TEACHER_TRACE",
        "panel_index": state["panel_index"],
        "qualification_candidate_index": state["qualification_candidate_index"],
        "state_id": state["state_id"],
        "family": state["family"],
        "condition_id": "TEACHER_TRACE",
        "target_id": target_id,
        "candidate_ids": [],
        "scores": None,
        "ranking": None,
        "eligible_correct_edge_candidate_indices": [],
        "selected_candidate_index": None,
        "correct_edge_top1": None,
        "correct_edge_top3": None,
        "correct_edge_mrr": None,
        "selected_correct_edge_execution": None,
        "selected_port_progress_m": float(projection["route_progress_m"]),
        "oracle_best_port_progress_m": None,
        "minimum_admissible_port_progress_m": None,
        "normalized_port_regret": None,
        "teacher_trace_index": int(teacher["trace_index"]),
        "teacher_trace_id": teacher["teacher_trace_id"],
        "teacher_correct_execution": bool(projection["teacher_valid"]),
        "ranker_checkpoint_sha256": None,
        "pairwise_correct_edge_ordering": None,
        "selected_wrong_edge": bool(projection["competing_port_entered"]),
        "selected_no_edge": bool(
            not projection["crossed_directed_port"]
            and not projection["competing_port_entered"]
        ),
        "selected_lateral_error_m": float(projection["endpoint_lateral_error_m"]),
        "selected_angular_error_rad": float(projection["endpoint_angular_error_rad"]),
        "selected_contact": not bool(projection["contact_free"]),
        "selected_stuck": bool(projection["stuck"]),
        "selected_successor_viable": bool(projection["successor_viable"]),
        "ranker_runtime_environment_sha256": ranker_runtime_sha256,
    }


def build_heldout_scores(
    panel_documents: Mapping[str, Any],
    development_target_selection: Mapping[str, Any],
    encoding_material: Mapping[str, Any],
    fanout_material: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    ranker_runtime: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    *,
    allow_fake_runtime: bool = False,
) -> list[dict[str, Any]]:
    """Validate and rebuild all four held-out comparator rows per state."""

    documents = _validate_panel_document_envelopes(
        panel_documents, allow_fake_runtime=allow_fake_runtime
    )
    C.validate_runtime_contract(runtime_contract)
    all_states, all_fanout = _fanout_population(documents, fanout_material)
    development_fanout = [
        all_fanout[state["state_id"]]
        for state in all_states
        if state["role"] == "DEVELOPMENT"
    ]
    selection = validate_development_target_selection(
        development_target_selection,
        panel_documents=documents,
        encoding_material=encoding_material,
        fanout_material=development_fanout,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    heldout = [state for state in all_states if state["role"] == "DEVELOPMENT_HELDOUT"]
    if len(heldout) != 16:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "heldout state cardinality drift"
        )
    selection_bytes = C.canonical_json_bytes(selection)
    for state in all_states:
        metadata = all_fanout[state["state_id"]]["metadata"]
        if state["role"] == "DEVELOPMENT":
            if metadata["development_target_selection_binding"] is not None:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "development fanout contains target selection binding"
                )
        else:
            binding = _validate_exact_canonical_file_binding(
                metadata["development_target_selection_binding"],
                path="development_target_selection.json",
                raw=selection_bytes,
                label="heldout target selection binding",
            )
            if metadata["development_target_selection_opened"] is not True:
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "heldout fanout target selection content binding drift"
                )
    visual = _validate_visual_runtime(
        ranker_runtime, role="ranker", allow_fake_runtime=allow_fake_runtime
    )
    if visual != selection["ranker_runtime_environment"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "ranker runtime changed after development target freeze"
        )
    runtime_sha = runtime_environment_sha256(visual)
    supplied = [
        _mapping(row, C.HELDOUT_SCORE_ROW_FIELDS, f"heldout score[{index}]")
        for index, row in enumerate(_sequence(score_rows, "heldout score rows"))
    ]
    identities = [
        (state["state_id"], condition)
        for state in heldout
        for condition in C.HELDOUT_CONDITION_IDS
    ]
    if [
        (row["state_id"], row["condition_id"]) for row in supplied
    ] != identities:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "heldout score order/population drift"
        )
    target_id = selection["selected_target_id"]
    targets = {
        row["state_id"]: row
        for row in documents["target_contracts.json"]["rows"]
        if row["target_id"] == target_id
    }
    teachers = {
        row["state_id"]: row
        for row in documents["teacher_trace_index.json"]["records"]
        if row["selected"]
    }
    rebuilt_rows: list[dict[str, Any]] = []
    for row, (state_id, condition) in zip(supplied, identities):
        state = next(item for item in heldout if item["state_id"] == state_id)
        outcomes = all_fanout[state_id]["metadata"]["outcome_rows"]
        if condition == "TEACHER_TRACE":
            rebuilt = _teacher_heldout_row(
                state, teachers[state_id], target_id, runtime_sha
            )
        else:
            if condition == "DETERMINISTIC_KINEMATICS":
                scores = _kinematic_scores(outcomes, targets[state_id]["target_body_pose"])
                ranking = _rank_scores(scores)
            elif condition == "FROZEN_CURRENT_VISUAL_RANKER":
                scores = [
                    _finite(value, "heldout frozen-ranker score")
                    for value in _sequence(row["scores"], "heldout scores")
                ]
                ranking = _rank_scores(scores)
            elif condition == "ORACLE_BEST_ADMISSIBLE_CANDIDATE":
                scores = None
                ranking = _oracle_ranking(outcomes)
            else:  # pragma: no cover - exact authority checked by identities
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    "unknown heldout condition"
                )
            rebuilt = {
                "score_row_id": f"{state_id}::{condition}",
                "panel_index": state["panel_index"],
                "qualification_candidate_index": state["qualification_candidate_index"],
                "state_id": state_id,
                "family": state["family"],
                "condition_id": condition,
                "target_id": target_id,
                **_selection_metrics(outcomes, ranking, scores),
                "teacher_trace_index": None,
                "teacher_trace_id": None,
                "teacher_correct_execution": None,
                "ranker_checkpoint_sha256": (
                    C.FROZEN_RANKER_CHECKPOINT_SHA256
                    if condition == "FROZEN_CURRENT_VISUAL_RANKER"
                    else None
                ),
                "ranker_runtime_environment_sha256": runtime_sha,
            }
        _require_semantic_equal(row, rebuilt, f"heldout score {state_id}/{condition}")
        rebuilt_rows.append(rebuilt)
    return rebuilt_rows


def validate_heldout_scores(
    value: Any,
    *,
    panel_documents: Mapping[str, Any],
    development_target_selection: Mapping[str, Any],
    encoding_material: Mapping[str, Any],
    fanout_material: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
) -> list[dict[str, Any]]:
    selection = _mapping(
        development_target_selection,
        C.DEVELOPMENT_TARGET_SELECTION_FIELDS,
        "development target selection",
    )
    return build_heldout_scores(
        panel_documents,
        selection,
        encoding_material,
        fanout_material,
        _sequence(value, "heldout scores"),
        selection["ranker_runtime_environment"],
        runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_jsonl_bytes(rows)).hexdigest()


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(C.canonical_json_bytes(row) for row in rows)


def validate_repeatability_material_shard(
    metadata: Any,
    *,
    reopened_arrays: Mapping[str, Any],
    panel_state: Mapping[str, Any],
    candidate_spec: Mapping[str, Any],
    selected_material: Mapping[str, Any],
    source_fanout_material: Mapping[str, Any],
    heldout_score_rows: Sequence[Mapping[str, Any]],
    directed_port_record: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
    material_metadata_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild four selected-branch repeats from raw material evidence."""

    panel = _mapping(panel_state, C.PANEL_STATE_FIELDS, "repeat panel state")
    if panel["role"] != "DEVELOPMENT_HELDOUT":
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat material state is not heldout"
        )
    spec = C.validate_candidate_spec(candidate_spec)
    panel_index = _nonnegative_int(panel["panel_index"], "repeat panel index")
    if (
        spec["state_id"] != panel["state_id"]
        or spec["candidate_index"] != panel["qualification_candidate_index"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat panel/spec identity drift"
        )
    selected = _validated_selected_wrapper(selected_material, spec, panel_index)
    if not isinstance(source_fanout_material, Mapping) or not {
        "metadata", "arrays", "material_metadata_binding"
    }.issubset(source_fanout_material):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat source fanout wrapper drift"
        )
    source = validate_fanout_material_shard(
        source_fanout_material["metadata"],
        reopened_arrays=source_fanout_material["arrays"],
        panel_state=panel,
        candidate_spec=spec,
        selected_material=selected,
        directed_port_record=directed_port_record,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
        material_metadata_binding=source_fanout_material["material_metadata_binding"],
    )
    scores = [
        _mapping(item, C.HELDOUT_SCORE_ROW_FIELDS, "repeat heldout score")
        for item in _sequence(heldout_score_rows, "repeat heldout scores")
    ]
    score_by_condition = {
        row["condition_id"]: row for row in scores if row["state_id"] == panel["state_id"]
    }
    if not {
        "FROZEN_CURRENT_VISUAL_RANKER",
        "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
    }.issubset(score_by_condition):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat selector rows absent"
        )
    selected_indices = [
        score_by_condition["FROZEN_CURRENT_VISUAL_RANKER"]["selected_candidate_index"],
        score_by_condition["FROZEN_CURRENT_VISUAL_RANKER"]["selected_candidate_index"],
        score_by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]["selected_candidate_index"],
        score_by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]["selected_candidate_index"],
    ]
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(C.CANDIDATE_IDS)
        for index in selected_indices
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat selector index drift"
        )
    runtime = C.validate_runtime_contract(runtime_contract)
    row = _mapping(
        metadata, C.REPEATABILITY_MATERIAL_FIELDS, "repeatability material"
    )
    C.validate_content_digest(row)
    heldout_bytes = _canonical_jsonl_bytes(scores)
    _validate_exact_canonical_file_binding(
        row["heldout_scores_binding"],
        path="heldout_scores.jsonl",
        raw=heldout_bytes,
        label="heldout scores binding",
    )
    if (
        row["schema"] != C.REPEATABILITY_MATERIAL_SCHEMA
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["source_freeze_commit"] != runtime["source_freeze_commit"]
        or row["runtime_contract_content_digest"] != runtime["content_digest"]
        or row["panel_index"] != panel_index
        or row["qualification_candidate_index"] != spec["candidate_index"]
        or row["state_id"] != spec["state_id"]
        or row["family"] != spec["family"]
        or row["role"] != "DEVELOPMENT_HELDOUT"
        or row["source_fanout_metadata_binding"] != source["material_metadata_binding"]
        or row["source_fanout_payload_binding"] != source["material_payload_binding"]
        or row["repeat_count"] != 4
        or row["models_trained"] != 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability provenance/identity drift"
        )
    persisted = validate_persisted_array_evidence(
        row["persisted_array_evidence"], reopened_arrays=reopened_arrays
    )
    shard_id = f"repeatability/{spec['state_id']}"
    _validate_successor_payload_binding(
        row, persisted, shard_kind="repeatability_material", shard_id=shard_id
    )
    expected_names = {
        f"repeat_{index:02d}__{member}"
        for index in range(4)
        for member in C.CANDIDATE_TRACE_MEMBER_ORDER
    }
    if set(reopened_arrays) != expected_names:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability raw array inventory drift"
        )
    outcomes = _sequence(row["outcome_rows"], "repeatability outcome rows")
    if len(outcomes) != 4:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability outcome cardinality drift"
        )
    traces: list[dict[str, Any]] = []
    rebuilt_outcomes: list[dict[str, Any]] = []
    for index, candidate_index in enumerate(selected_indices):
        trace = _normalise_candidate_trace(
            reopened_arrays,
            prefix=f"repeat_{index:02d}",
            selected_metadata=selected["metadata"],
            selected_arrays=selected["arrays"],
            candidate_index=candidate_index,
            candidate_spec=spec,
        )
        rebuilt_full = _derive_candidate_outcome_native(
            spec,
            trace,
            selected["arrays"]["snapshot__base_pose_world"],
            candidate_index,
            directed_port_record["directed_port_world"],
        )
        rebuilt = {
            field: value
            for field, value in rebuilt_full.items()
            if field in C.REPEAT_OUTCOME_FIELDS
        }
        supplied = _mapping(
            outcomes[index], C.REPEAT_OUTCOME_FIELDS, f"repeat outcome[{index}]"
        )
        _require_semantic_equal(supplied, rebuilt, f"repeat outcome[{index}]")
        traces.append(trace)
        rebuilt_outcomes.append(rebuilt)
    stage = _mapping(
        row["stage_runtime"], set(C.PHYSICAL_RUNTIME_CORE_FIELDS), "repeat runtime"
    )
    backend = _mapping(
        row["backend_runtime"],
        set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS),
        "repeat backend runtime",
    )
    if (
        stage != selected["metadata"]["stage_runtime"]
        or backend != selected["metadata"]["backend_runtime"]
        or stage != source["metadata"]["stage_runtime"]
        or backend != source["metadata"]["backend_runtime"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability physical runtime drift"
        )
    if stage["fake_runtime"] and not allow_fake_runtime:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "production repeatability contains fake runtime"
        )
    metadata_binding = _validate_optional_metadata_binding(
        material_metadata_binding,
        row,
        expected_path=f"{shard_id}/metadata.json",
        label="repeatability material metadata binding",
    )
    return {
        "metadata": row,
        "arrays": {name: reopened_arrays[name] for name in reopened_arrays},
        "traces": traces,
        "rebuilt_outcome_rows": rebuilt_outcomes,
        "selected_candidate_indices": selected_indices,
        "material_metadata_binding": metadata_binding,
        "material_payload_binding": persisted["payload_file"],
        "physical_runtime_core_sha256": runtime_environment_sha256(stage),
    }


def validate_candidate_fanout_rows(
    value: Any,
    *,
    panel_documents: Mapping[str, Any],
    fanout_material: Sequence[Mapping[str, Any]],
    development_target_selection: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
) -> list[dict[str, Any]]:
    """Validate the 768-row official fanout ledger against material shards."""

    documents = _validate_panel_document_envelopes(
        panel_documents, allow_fake_runtime=allow_fake_runtime
    )
    C.validate_runtime_contract(runtime_contract)
    states, by_state = _fanout_population(documents, fanout_material)
    selection = _mapping(
        development_target_selection,
        C.DEVELOPMENT_TARGET_SELECTION_FIELDS,
        "fanout ledger development selection",
    )
    C.validate_content_digest(selection)
    selection_bytes = C.canonical_json_bytes(selection)
    supplied = [
        _mapping(row, C.CANDIDATE_FANOUT_ROW_FIELDS, f"fanout ledger[{index}]")
        for index, row in enumerate(_sequence(value, "candidate fanout rows"))
    ]
    rebuilt_rows: list[dict[str, Any]] = []
    for state in states:
        wrapper = by_state[state["state_id"]]
        metadata = wrapper["metadata"]
        if state["role"] == "DEVELOPMENT_HELDOUT":
            _validate_exact_canonical_file_binding(
                metadata["development_target_selection_binding"],
                path="development_target_selection.json",
                raw=selection_bytes,
                label="heldout fanout ledger selection binding",
            )
        for branch_index, outcome in enumerate(metadata["outcome_rows"]):
            trace_digests = {
                member: _canonical_array_sha256_local(
                    wrapper["arrays"][f"candidate_{branch_index:02d}__{member}"]
                )
                for member in sorted(C.CANDIDATE_TRACE_MEMBER_ORDER)
            }
            rebuilt_rows.append(
                {
                    "branch_id": f"{state['state_id']}::{C.CANDIDATE_IDS[branch_index]}",
                    "panel_index": state["panel_index"],
                    "qualification_candidate_index": state[
                        "qualification_candidate_index"
                    ],
                    "state_id": state["state_id"],
                    "family": state["family"],
                    "role": state["role"],
                    "candidate_spec_id": state["candidate_spec_id"],
                    "branch_candidate_index": branch_index,
                    "branch_candidate_id": C.CANDIDATE_IDS[branch_index],
                    "snapshot_payload_sha256": metadata["snapshot_payload_sha256"],
                    "material_metadata_binding": wrapper["material_metadata_binding"],
                    "material_payload_binding": wrapper["material_payload_binding"],
                    "outcome_row_index": branch_index,
                    "trace_digests": trace_digests,
                    **copy.deepcopy(outcome),
                }
            )
    if len(supplied) != C.PANEL_STATE_COUNT * len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "candidate fanout ledger cardinality drift"
        )
    _require_semantic_equal(supplied, rebuilt_rows, "candidate fanout ledger")
    return supplied


def validate_repeatability_rows(
    value: Any,
    *,
    panel_documents: Mapping[str, Any],
    candidate_fanout_rows: Sequence[Mapping[str, Any]],
    heldout_score_rows: Sequence[Mapping[str, Any]],
    repeatability_material: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    allow_fake_runtime: bool = False,
) -> list[dict[str, Any]]:
    """Validate the 64-row repeat ledger against raw repeat material."""

    documents = _validate_panel_document_envelopes(
        panel_documents, allow_fake_runtime=allow_fake_runtime
    )
    C.validate_runtime_contract(runtime_contract)
    panel_states = documents["panel_manifest.json"]["states"]
    heldout_states = [
        row for row in panel_states if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    fanout = [
        _mapping(row, C.CANDIDATE_FANOUT_ROW_FIELDS, "repeat source fanout row")
        for row in _sequence(candidate_fanout_rows, "repeat source fanout rows")
    ]
    if len(fanout) != C.PANEL_STATE_COUNT * len(C.CANDIDATE_IDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat source fanout ledger cardinality drift"
        )
    fanout_by_identity = {
        (row["state_id"], row["branch_candidate_index"]): row for row in fanout
    }
    if len(fanout_by_identity) != len(fanout):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeat source fanout identity duplicated"
        )
    scores = [
        _mapping(row, C.HELDOUT_SCORE_ROW_FIELDS, "repeat score row")
        for row in _sequence(heldout_score_rows, "repeat score rows")
    ]
    score_by_identity = {
        (row["state_id"], row["condition_id"]): row for row in scores
    }
    wrappers = _sequence(repeatability_material, "repeatability material population")
    by_state: dict[str, Mapping[str, Any]] = {}
    for wrapper in wrappers:
        if not isinstance(wrapper, Mapping) or not {
            "metadata", "arrays", "material_metadata_binding", "material_payload_binding",
            "selected_candidate_indices", "physical_runtime_core_sha256"
        }.issubset(wrapper):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "repeatability validation wrapper drift"
            )
        state_id = wrapper["metadata"].get("state_id")
        if not isinstance(state_id, str) or state_id in by_state:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "repeatability material identity duplicated"
            )
        by_state[state_id] = wrapper
    if set(by_state) != {row["state_id"] for row in heldout_states}:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability material state population drift"
        )
    rebuilt_rows: list[dict[str, Any]] = []
    for state in heldout_states:
        state_id = state["state_id"]
        wrapper = by_state[state_id]
        metadata = wrapper["metadata"]
        expected_indices = [
            score_by_identity[(state_id, "FROZEN_CURRENT_VISUAL_RANKER")][
                "selected_candidate_index"
            ],
            score_by_identity[(state_id, "ORACLE_BEST_ADMISSIBLE_CANDIDATE")][
                "selected_candidate_index"
            ],
        ]
        if wrapper["selected_candidate_indices"] != [
            expected_indices[0], expected_indices[0], expected_indices[1], expected_indices[1]
        ]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "repeatability wrapper selector projection drift"
            )
        for selector_index, selector_id in enumerate(C.REPEAT_BRANCH_IDS):
            source_index = expected_indices[selector_index]
            source = fanout_by_identity[(state_id, source_index)]
            for repeat_index in range(C.REPEATS_PER_BRANCH):
                outcome_index = selector_index * C.REPEATS_PER_BRANCH + repeat_index
                outcome = metadata["outcome_rows"][outcome_index]
                trace_digests = {
                    member: _canonical_array_sha256_local(
                        wrapper["arrays"][f"repeat_{outcome_index:02d}__{member}"]
                    )
                    for member in sorted(C.CANDIDATE_TRACE_MEMBER_ORDER)
                }
                source_endpoint = [float(value) for value in source["h3_endpoint_body"]]
                repeat_endpoint = [float(value) for value in outcome["h3_endpoint_body"]]
                position_error = math.hypot(
                    source_endpoint[0] - repeat_endpoint[0],
                    source_endpoint[1] - repeat_endpoint[1],
                )
                heading_error = abs(
                    _wrap_angle_v1(source_endpoint[2] - repeat_endpoint[2])
                )
                source_applied = source["trace_digests"][
                    "post_slew_applied_command"
                ]
                repeat_applied = trace_digests["post_slew_applied_command"]
                repeat_success = bool(
                    source_applied == repeat_applied
                    and bool(source["entered_correct_edge"])
                    is bool(outcome["entered_correct_edge"])
                    and source["endpoint_edge_id"] == outcome["endpoint_edge_id"]
                    and bool(source["physics_contact"])
                    is bool(outcome["physics_contact"])
                    and bool(source["stuck"]) is bool(outcome["stuck"])
                    and position_error
                    <= float(C.NUMERICAL_TOLERANCES["repeat_endpoint_position_m"])
                    and heading_error
                    <= float(C.NUMERICAL_TOLERANCES["repeat_endpoint_heading_rad"])
                )
                rebuilt_rows.append(
                    {
                        "repeat_id": f"{state_id}::{selector_id}::{repeat_index}",
                        "panel_index": state["panel_index"],
                        "qualification_candidate_index": state[
                            "qualification_candidate_index"
                        ],
                        "state_id": state_id,
                        "family": state["family"],
                        "branch_selector_id": selector_id,
                        "repeat_index": repeat_index,
                        "source_candidate_index": source_index,
                        "source_branch_id": source["branch_id"],
                        "snapshot_payload_sha256": metadata.get(
                            "snapshot_payload_sha256",
                            source["snapshot_payload_sha256"],
                        ),
                        "material_metadata_binding": wrapper[
                            "material_metadata_binding"
                        ],
                        "material_payload_binding": wrapper["material_payload_binding"],
                        "outcome_row_index": outcome_index,
                        "trace_digests": trace_digests,
                        "source_correct_edge_execution": bool(source["entered_correct_edge"]),
                        "repeat_correct_edge_execution": bool(outcome["entered_correct_edge"]),
                        "source_endpoint_body": source_endpoint,
                        "repeat_endpoint_body": repeat_endpoint,
                        "endpoint_position_error_m": position_error,
                        "endpoint_heading_error_rad": heading_error,
                        "physics_contact": bool(outcome["physics_contact"]),
                        "source_applied_command_sequence_sha256": source_applied,
                        "repeat_applied_command_sequence_sha256": repeat_applied,
                        "source_endpoint_edge_id": source["endpoint_edge_id"],
                        "repeat_endpoint_edge_id": outcome["endpoint_edge_id"],
                        "source_physics_contact": bool(source["physics_contact"]),
                        "source_stuck": bool(source["stuck"]),
                        "stuck": bool(outcome["stuck"]),
                        "repeat_success": repeat_success,
                        "physical_runtime_core_sha256": wrapper[
                            "physical_runtime_core_sha256"
                        ],
                    }
                )
    supplied = [
        _mapping(row, C.REPEATABILITY_ROW_FIELDS, f"repeatability row[{index}]")
        for index, row in enumerate(_sequence(value, "repeatability rows"))
    ]
    if len(supplied) != len(heldout_states) * len(C.REPEAT_BRANCH_IDS) * C.REPEATS_PER_BRANCH:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "repeatability ledger cardinality drift"
        )
    _require_semantic_equal(supplied, rebuilt_rows, "repeatability ledger")
    return supplied


def validate_material_inventory_projection(
    value: Any, *, attempted_candidate_count: int, panel_available: bool,
    expected_root: str | None = None,
) -> dict[str, Any]:
    row = _mapping(
        value, C.MATERIAL_INVENTORY_PROJECTION_FIELDS, "material inventory projection"
    )
    C.validate_content_digest(row)
    counts = C.expected_material_inventory_counts(
        attempted_candidate_count, panel_available=panel_available
    )
    expected_branch = "SUCCESS" if panel_available else "GENERATOR_TERMINAL"
    files = [
        _file_binding(item, f"material inventory file[{index}]")
        for index, item in enumerate(_sequence(row["files"], "material inventory files"))
    ]
    directories = _sequence(row["directories"], "material inventory directories")
    if (
        row["schema"]
        != "physical_handoff_stratified_generator_successor_v1.material_inventory_projection.v1"
        or row["experiment_id"] != C.EXPERIMENT_ID
        or not isinstance(row["root"], str)
        or not row["root"].startswith("/")
        or (expected_root is not None and row["root"] != expected_root)
        or row["branch"] != expected_branch
        or row["attempted_candidate_count"] != attempted_candidate_count
        or row["file_count"] != counts["file_count"]
        or row["directory_count"] != counts["directory_count"]
        or len(files) != counts["file_count"]
        or len(directories) != counts["directory_count"]
        or [item["path"] for item in files]
        != sorted(item["path"] for item in files)
        or len({item["path"] for item in files}) != len(files)
        or directories != sorted(set(directories))
        or any(not isinstance(item, str) or not item for item in directories)
        or row["unexpected_file_count"] != 0
        or row["unexpected_directory_count"] != 0
        or row["v4_physical_shard_file_count"] != 512
        or row["v4_physical_shard_sha256_overlap_count"] != 0
        or row["v4_physical_shard_copy_reuse_detected"] is not False
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material inventory projection drift"
        )
    return row


def _mean(values: Sequence[Any]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "mean population is empty"
        )
    return math.fsum(rows) / len(rows)


def _success_downstream_reduction(
    documents: Mapping[str, Any],
    selection: Mapping[str, Any],
    fanout: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
    encoding: Mapping[str, Any],
) -> dict[str, Any]:
    panel = documents["panel_manifest.json"]
    targets = documents["target_contracts.json"]
    target_summaries = {
        item["target_id"]: item for item in selection["target_summaries"]
    }
    selected_target = selection["selected_target_id"]
    selected_summary = target_summaries[selected_target]
    node_summary = target_summaries["TARGET_NODE_CENTRE"]
    development_selected_pass = bool(
        selected_summary["correct_edge_top1_rate"]
        >= C.HANDOFF_GATE["ranker_correct_edge_top1_rate_minimum"]
        and selected_summary["correct_edge_top3_rate"]
        >= C.HANDOFF_GATE["ranker_correct_edge_top3_rate_minimum"]
        and selected_summary["selected_correct_edge_execution_rate"]
        >= C.HANDOFF_GATE["ranker_selected_correct_edge_execution_rate_minimum"]
        and selected_summary["normalized_port_regret"]
        <= C.HANDOFF_GATE["ranker_normalized_port_regret_maximum"]
    )
    selected_material = _materially_outperforms(selected_summary, node_summary)
    condition_summaries = {
        condition: _condition_summary(heldout, condition)
        for condition in C.HELDOUT_CONDITION_IDS
    }
    fanout_by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fanout:
        fanout_by_state[row["state_id"]].append(row)
    for state_id, rows in fanout_by_state.items():
        rows.sort(key=lambda item: item["branch_candidate_index"])
        if [item["branch_candidate_index"] for item in rows] != list(
            range(len(C.CANDIDATE_IDS))
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"fanout branch population drift for {state_id}"
            )
    heldout_states = [
        item for item in panel["states"] if item["role"] == "DEVELOPMENT_HELDOUT"
    ]
    all_panel_states = list(panel["states"])
    all_correct_counts = {
        item["state_id"]: sum(
            _correct_candidate(branch)
            for branch in fanout_by_state[item["state_id"]]
        )
        for item in all_panel_states
    }
    correct_counts = {
        item["state_id"]: sum(
            _correct_candidate(branch)
            for branch in fanout_by_state[item["state_id"]]
        )
        for item in heldout_states
    }
    coverage_by_state = {state_id: count >= 1 for state_id, count in correct_counts.items()}
    coverage_rate = _mean([float(value) for value in coverage_by_state.values()])
    two_rate = _mean([float(value >= 2) for value in correct_counts.values()])
    ranker_rows = [
        item for item in heldout if item["condition_id"] == "FROZEN_CURRENT_VISUAL_RANKER"
    ]
    oracle_rows = [
        item for item in heldout if item["condition_id"] == "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
    ]
    teacher_summary = condition_summaries["TEACHER_TRACE"]
    ranker_summary = condition_summaries["FROZEN_CURRENT_VISUAL_RANKER"]
    oracle_summary = condition_summaries["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]
    covered_oracle = [
        item for item in oracle_rows if coverage_by_state[item["state_id"]]
    ]
    oracle_covered_rate = (
        _mean([float(item["selected_correct_edge_execution"]) for item in covered_oracle])
        if covered_oracle
        else 0.0
    )
    family_execution = {
        family: sum(
            bool(item["selected_correct_edge_execution"])
            for item in ranker_rows
            if item["family"] == family
        )
        for family in C.FAMILY_IDS
    }
    repeatability_rate = _mean([float(item["repeat_success"]) for item in repeats])
    command_tracking = _command_tracking_summary(heldout, fanout_by_state)
    full_gate = bool(
        teacher_summary["teacher_correct_execution_count"]
        == C.HANDOFF_GATE["teacher_correct_execution_count"]
        and coverage_rate >= C.HANDOFF_GATE["coverage_rate_minimum"]
        and oracle_covered_rate >= 1.0
        and ranker_summary["correct_edge_top1_rate"]
        >= C.HANDOFF_GATE["ranker_correct_edge_top1_rate_minimum"]
        and ranker_summary["correct_edge_top3_rate"]
        >= C.HANDOFF_GATE["ranker_correct_edge_top3_rate_minimum"]
        and ranker_summary["selected_correct_edge_execution_rate"]
        >= C.HANDOFF_GATE["ranker_selected_correct_edge_execution_rate_minimum"]
        and ranker_summary["normalized_port_regret"]
        <= C.HANDOFF_GATE["ranker_normalized_port_regret_maximum"]
        and repeatability_rate >= C.HANDOFF_GATE["repeatability_rate_minimum"]
        and command_tracking["passed"]
        and min(family_execution.values())
        >= C.HANDOFF_GATE["minimum_correct_execution_per_family"]
    )
    classification_input = {
        "teacher_correct_execution_count": teacher_summary[
            "teacher_correct_execution_count"
        ],
        "coverage_rate": coverage_rate,
        "ranker_correct_edge_top1_rate": ranker_summary["correct_edge_top1_rate"],
        "ranker_correct_edge_top3_rate": ranker_summary["correct_edge_top3_rate"],
        "ranker_selected_correct_edge_execution_rate": ranker_summary[
            "selected_correct_edge_execution_rate"
        ],
        "ranker_normalized_port_regret": ranker_summary["normalized_port_regret"],
        "oracle_selected_correct_edge_execution_rate": oracle_summary[
            "selected_correct_edge_execution_rate"
        ],
        "oracle_covered_state_correct_execution_rate": oracle_covered_rate,
        "repeatability_rate": repeatability_rate,
        "command_tracking_pass": command_tracking["passed"],
        "minimum_family_correct_execution_count": min(family_execution.values()),
        "selected_target_id": selected_target,
        "selected_target_passes_handoff_gate": full_gate,
        "selected_target_materially_outperforms_node_centre": selected_material,
    }
    try:
        disposition = classify_physical_handoff_aggregates(classification_input)
    except Exception as exc:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"downstream classification failed: {exc}"
        ) from exc

    def grouped_rate(attribute: str, groups: Sequence[str]) -> dict[str, Any]:
        result = {}
        ranker_by_state = {item["state_id"]: item for item in ranker_rows}
        for group in groups:
            states = [item for item in heldout_states if item[attribute] == group]
            covered = [coverage_by_state[item["state_id"]] for item in states]
            two = [correct_counts[item["state_id"]] >= 2 for item in states]
            executed = [
                bool(ranker_by_state[item["state_id"]]["selected_correct_edge_execution"])
                for item in states
            ]
            result[group] = {
                "state_count": len(states),
                "coverage_rate": _mean([float(item) for item in covered]) if covered else 0.0,
                "two_or_more_candidate_count": sum(two),
                "two_or_more_candidate_rate": _mean([float(item) for item in two]) if two else 0.0,
                "ranker_execution_rate": _mean([float(item) for item in executed]) if executed else 0.0,
            }
        return result

    direction = grouped_rate("route_direction", ("STRAIGHT", "LEFT", "RIGHT"))
    distance = grouped_rate("port_distance_id", ("NEAR", "FAR"))
    family_coverage = grouped_rate("family", C.FAMILY_IDS)

    def bank_coverage_summary(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counts = [all_correct_counts[item["state_id"]] for item in states]
        return {
            "state_count": len(states),
            "covered_state_count": sum(value >= 1 for value in counts),
            "coverage_rate": _mean([float(value >= 1) for value in counts])
            if counts
            else 0.0,
            "two_or_more_candidate_state_count": sum(value >= 2 for value in counts),
            "two_or_more_candidate_rate": _mean(
                [float(value >= 2) for value in counts]
            )
            if counts
            else 0.0,
            "correct_candidate_count": sum(counts),
        }

    bank_by_family = {
        family: bank_coverage_summary(
            [item for item in all_panel_states if item["family"] == family]
        )
        for family in C.FAMILY_IDS
    }
    bank_by_stratum = [
        {
            "family": family,
            "stratum_index": stratum,
            **bank_coverage_summary(
                [
                    item
                    for item in all_panel_states
                    if item["family"] == family
                    and item["stratum_index"] == stratum
                ]
            ),
        }
        for family in C.FAMILY_IDS
        for stratum in range(C.STRATA_PER_FAMILY)
    ]
    bank_by_direction = {
        direction_id: bank_coverage_summary(
            [
                item
                for item in all_panel_states
                if item["route_direction"] == direction_id
            ]
        )
        for direction_id in ("STRAIGHT", "LEFT", "RIGHT")
    }
    bank_by_distance = {
        distance_id: bank_coverage_summary(
            [
                item
                for item in all_panel_states
                if item["port_distance_id"] == distance_id
            ]
        )
        for distance_id in ("NEAR", "FAR")
    }
    bank_aggregate = bank_coverage_summary(all_panel_states)
    bank_by_target = {
        target_id: {
            **copy.deepcopy(bank_aggregate),
            "physical_trajectory_population_identical_across_target_representations": True,
        }
        for target_id in C.TARGET_IDS
    }

    distribution_features = (
        "dx_m",
        "dy_m",
        "distance_m",
        "relative_heading_rad",
        "relative_heading_sin",
        "relative_heading_cos",
        "route_intent_dx",
        "route_intent_dy",
    )
    target_distribution_by_id: dict[str, Any] = {}
    for target_id in C.TARGET_IDS:
        rows = [row for row in targets["rows"] if row["target_id"] == target_id]
        target_distribution_by_id[target_id] = {
            "state_count": len(rows),
            "feature_summaries": {
                field: {
                    "minimum": min(float(row[field]) for row in rows),
                    "maximum": max(float(row[field]) for row in rows),
                    "mean": math.fsum(float(row[field]) for row in rows)
                    / len(rows),
                }
                for field in distribution_features
            },
            "frozen_ranker_support": copy.deepcopy(
                targets["target_support_counts"][target_id]
            ),
        }
    contributions = {
        candidate_id: {
            "correct_edge_candidate_count": sum(
                _correct_candidate(fanout_by_state[state["state_id"]][candidate_index])
                for state in heldout_states
            ),
            "heldout_state_rate": _mean(
                [
                    float(_correct_candidate(fanout_by_state[state["state_id"]][candidate_index]))
                    for state in heldout_states
                ]
            ),
        }
        for candidate_index, candidate_id in enumerate(C.CANDIDATE_IDS)
    }
    per_family_conditions = {
        family: {
            condition: _condition_summary(
                [item for item in heldout if item["family"] == family], condition
            )
            for condition in C.HELDOUT_CONDITION_IDS
        }
        for family in C.FAMILY_IDS
    }
    repeat_by_selector = {
        selector: {
            "row_count": sum(item["branch_selector_id"] == selector for item in repeats),
            "agreement_rate": _mean(
                [float(item["repeat_success"]) for item in repeats if item["branch_selector_id"] == selector]
            ),
        }
        for selector in C.REPEAT_BRANCH_IDS
    }
    repeat_by_family = {
        family: {
            "row_count": sum(item["family"] == family for item in repeats),
            "agreement_rate": _mean(
                [float(item["repeat_success"]) for item in repeats if item["family"] == family]
            ),
        }
        for family in C.FAMILY_IDS
    }
    left_right = max(
        abs(direction["LEFT"]["coverage_rate"] - direction["RIGHT"]["coverage_rate"]),
        abs(direction["LEFT"]["ranker_execution_rate"] - direction["RIGHT"]["ranker_execution_rate"]),
    )
    far_limit = max(
        distance["NEAR"]["coverage_rate"] - distance["FAR"]["coverage_rate"],
        distance["NEAR"]["ranker_execution_rate"] - distance["FAR"]["ranker_execution_rate"],
    )
    support = targets["target_support_counts"]
    secondary = []
    if support["TARGET_NODE_CENTRE"]["outside"] > 0:
        secondary.append("NODE_CENTRE_TARGET_OUT_OF_DISTRIBUTION")
    if _materially_outperforms(target_summaries["DIRECTED_EDGE_PORT"], node_summary):
        secondary.append("DIRECTED_PORT_TARGET_SIGNAL")
    if _materially_outperforms(target_summaries["ROUTE_LOOKAHEAD"], node_summary):
        secondary.append("ROUTE_LOOKAHEAD_TARGET_SIGNAL")
    if any(value["coverage_rate"] < 0.90 for value in direction.values()):
        secondary.append("CANDIDATE_BANK_DIRECTIONAL_GAP")
    if left_right >= 0.20:
        secondary.append("LEFT_RIGHT_ASYMMETRY")
    if far_limit >= 0.20:
        secondary.append("PORT_DISTANCE_LIMITATION")
    if (
        support[selected_target]["outside"] > 0
        and ranker_summary["selected_correct_edge_execution_rate"]
        < oracle_summary["selected_correct_edge_execution_rate"]
    ):
        secondary.append("RANKER_TARGET_DISTRIBUTION_SHIFT")
    if repeatability_rate < C.HANDOFF_GATE["repeatability_rate_minimum"] or not command_tracking["passed"]:
        secondary.append("CONTROLLER_TRACKING_LIMITATION")
    secondary = [name for name in C.SECONDARY_CLASSIFICATIONS if name in secondary]
    return {
        "panel": {
            "role_counts": copy.deepcopy(panel["role_counts"]),
            "family_counts": {family: 16 for family in C.FAMILY_IDS},
            "coverage_rate": coverage_rate,
            "covered_heldout_state_count": sum(coverage_by_state.values()),
            "two_or_more_correct_candidate_state_count": sum(value >= 2 for value in correct_counts.values()),
            "two_or_more_correct_candidate_rate": two_rate,
            "per_family_coverage": family_coverage,
            "candidate_identity_contributions": contributions,
        },
        "downstream": {
            "development": {
                "target_summaries": [target_summaries[target] for target in C.TARGET_IDS],
                "selected_target_id": selected_target,
                "selected_target_index": selection["selected_target_index"],
                "development_selected_target_passes_ranker_thresholds": development_selected_pass,
                "selected_target_passes_full_heldout_handoff_gate": full_gate,
                "selected_target_materially_outperforms_node_centre": selected_material,
            },
            "heldout": {
                "condition_summaries": [condition_summaries[condition] for condition in C.HELDOUT_CONDITION_IDS],
                "comparator_alias_authority": copy.deepcopy(
                    C.HELDOUT_COMPARATOR_ALIAS_AUTHORITY
                ),
                "oracle_covered_state_correct_execution_rate": oracle_covered_rate,
                "family_ranker_correct_execution_counts": family_execution,
                "per_family_condition_summaries": per_family_conditions,
            },
            "repeatability": {
                "rate": repeatability_rate,
                "successful_rows": sum(bool(item["repeat_success"]) for item in repeats),
                "row_count": len(repeats),
                "by_selector": repeat_by_selector,
                "by_family": repeat_by_family,
            },
            "command_tracking": command_tracking,
            "candidate_bank_coverage": {
                "aggregate": bank_aggregate,
                "by_family": bank_by_family,
                "by_family_and_stratum": bank_by_stratum,
                "by_edge_direction": bank_by_direction,
                "by_port_distance": bank_by_distance,
                "by_target_representation": bank_by_target,
                "heldout_gate_population_is_separately_reported": True,
            },
            "target_distribution_audit": {
                "feature_order": list(distribution_features),
                "route_intent_role": (
                    "selected directed-opening normal transformed into the "
                    "snapshot body frame; reported as contract context and not "
                    "added to the frozen ranker feature input"
                ),
                "frozen_ranker_support_authority": copy.deepcopy(
                    targets["original_ranker_support"]
                ),
                "by_target_representation": target_distribution_by_id,
            },
            "stratified": {
                "route_direction": direction,
                "port_distance": distance,
                "left_right_maximum_gap": left_right,
                "far_port_maximum_deficit": far_limit,
            },
            "classification_input": classification_input,
            "gate": {"authority": copy.deepcopy(C.HANDOFF_GATE), "passed": disposition["handoff_gate_passed"]},
            "component_failures": disposition["component_failures"],
            "active_components_in_precedence_order": disposition["active_components_in_precedence_order"],
            "earliest_failing_component": disposition["earliest_failing_component"],
        },
        "runtime_environments": {
            "physical": copy.deepcopy(panel["physical_runtime_environment"]),
            "encoder": copy.deepcopy(encoding["encoder_runtime_environment"]),
            "ranker": copy.deepcopy(selection["ranker_runtime_environment"]),
            "any_fake_runtime": bool(
                panel["physical_runtime_environment"]["fake_runtime"]
                or encoding["encoder_runtime_environment"]["fake_runtime"]
                or selection["ranker_runtime_environment"]["fake_runtime"]
            ),
        },
        "primary_classification": disposition["primary_classification"],
        "secondary_classifications": secondary,
        "next_decision": disposition["next_experiment"],
    }


GENERATOR_RUNTIME_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "runtime_contract_content_digest",
        "terminal_record_count",
        "candidate_indices",
        "physical_runtime_core",
        "physical_runtime_core_sha256",
        "backend_runtime_core",
        "backend_runtime_core_sha256",
        "stage_runtime_sha256s",
        "backend_runtime_sha256s",
        "all_stage_runtime_cores_equal",
        "all_backend_runtime_cores_equal",
        "fake_runtime",
        "models_trained",
        "content_digest",
    }
)


def _validate_qualification_terminal_runtime(
    metadata: Mapping[str, Any], runtime_contract: Mapping[str, Any], *,
    expected_pool_index: int, allow_fake_runtime: bool,
) -> dict[str, Any]:
    """Validate one successor terminal's explicit physical runtime evidence."""

    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != C.QUALIFICATION_MATERIAL_SCHEMA
        or metadata.get("experiment_id") != C.EXPERIMENT_ID
        or metadata.get("pool_index") != expected_pool_index
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification runtime terminal identity drift"
        )
    source = _commit(
        metadata.get("source_freeze_commit"),
        "qualification terminal source freeze commit",
    )
    runtime_digest = _sha(
        metadata.get("runtime_contract_content_digest"),
        "qualification terminal runtime contract digest",
    )
    if (
        source != runtime_contract["source_freeze_commit"]
        or runtime_digest != runtime_contract["content_digest"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification terminal source/runtime contract binding drift"
        )
    stage = _mapping(
        metadata.get("stage_runtime"),
        set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
        f"qualification[{expected_pool_index}] stage runtime",
    )
    physical_authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    if stage["stage_id"] != physical_authority["stage_id"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification physical runtime stage drift"
        )
    for field in (
        "python_executable",
        "python_version",
        "torch_version",
        "torch_hip_version",
        "genesis_version",
        "quadrants_version",
        "device",
        "backend",
    ):
        if not isinstance(stage[field], str) or not stage[field]:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"qualification physical runtime {field} drift"
            )
    if (
        not isinstance(stage["visible_device_count"], int)
        or isinstance(stage["visible_device_count"], bool)
        or stage["visible_device_count"] < 0
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification physical visible-device count drift"
        )
    fake = _bool(stage["fake_runtime"], "qualification fake runtime")
    if fake and not allow_fake_runtime:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "production qualification contains fake runtime"
        )
    expected_environment = runtime_contract["physical_runtime_policy"][
        "required_environment_before_simulator_creation"
    ]
    environment = stage["deterministic_environment"]
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(expected_environment)
        or any(
            item is not None and not isinstance(item, str)
            for item in environment.values()
        )
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification deterministic environment drift"
        )
    if not fake:
        expected_real = {
            "python_executable": physical_authority["real_python_executable"],
            "python_version": physical_authority["python_version"],
            "torch_version": physical_authority["torch_version"],
            "torch_hip_version": physical_authority["torch_hip_version"],
            "genesis_version": physical_authority["genesis_version"],
            "quadrants_version": physical_authority["quadrants_version"],
            "visible_device_count": physical_authority["visible_device_count"],
            "device": physical_authority["device"],
            "backend": physical_authority["backend"],
        }
        if (
            any(stage[field] != value for field, value in expected_real.items())
            or dict(environment) != expected_environment
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "real qualification physical runtime drift"
            )
    backend = metadata.get("backend_runtime")
    if not isinstance(backend, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification backend runtime is not mapping"
        )
    backend = copy.deepcopy(dict(backend))
    teacher_executed = _bool(
        metadata.get("teacher_executed"), "qualification teacher_executed"
    )
    expected_fields = set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS)
    if teacher_executed:
        expected_fields.update(C.QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS)
    if set(backend) != expected_fields:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification backend runtime field drift"
        )
    core = {
        field: copy.deepcopy(backend[field])
        for field in C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS
    }
    expected_core = copy.deepcopy(C.QUALIFICATION_BACKEND_RUNTIME_AUTHORITY)
    if fake:
        expected_core["backend"] = stage["backend"]
        expected_core["policy_device"] = stage["device"]
    if core != expected_core or backend["backend"] != stage["backend"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification backend/runtime component drift"
        )
    if teacher_executed:
        snapshot_sha = _sha(
            backend["teacher_snapshot_sha256"],
            "qualification teacher snapshot SHA-256",
        )
        identity = metadata.get("snapshot_identity")
        artifact_sha = (
            identity.get("artifact_file_sha256")
            if isinstance(identity, Mapping)
            else None
        )
        if (
            backend["snapshot_captured_before_teacher"] is not True
            or backend["teacher_restored_from_serialized_snapshot"] is not True
            or snapshot_sha != metadata.get("initial_decision_state_sha256")
            or snapshot_sha != artifact_sha
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "qualification teacher backend/snapshot runtime drift"
            )
    return {
        "stage_runtime": stage,
        "stage_runtime_sha256": runtime_environment_sha256(stage),
        "backend_runtime": backend,
        "backend_runtime_sha256": hashlib.sha256(
            C.canonical_json_bytes(backend)[:-1]
        ).hexdigest(),
        "backend_runtime_core": core,
        "backend_runtime_core_sha256": hashlib.sha256(
            C.canonical_json_bytes(core)[:-1]
        ).hexdigest(),
    }


def build_generator_runtime_environment(
    metadata_rows: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any], *, allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    """Reduce the variable attempted population to one exact runtime core."""

    runtime = C.validate_runtime_contract(runtime_contract)
    supplied = _sequence(metadata_rows, "generator runtime metadata rows")
    if not supplied or len(supplied) > C.MAX_CANDIDATE_COUNT:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator runtime row cardinality drift"
        )
    indices = [_nonnegative_int(row.get("pool_index"), "candidate_index") for row in supplied]
    if indices != sorted(set(indices)):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator runtime candidate order drift"
        )
    validated = [
        _validate_qualification_terminal_runtime(
            metadata,
            runtime,
            expected_pool_index=index,
            allow_fake_runtime=allow_fake_runtime,
        )
        for index, metadata in zip(indices, supplied)
    ]
    stage = validated[0]["stage_runtime"]
    backend = validated[0]["backend_runtime_core"]
    if any(item["stage_runtime"] != stage for item in validated[1:]):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "physical runtime changed across successor attempts"
        )
    if any(item["backend_runtime_core"] != backend for item in validated[1:]):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "backend runtime changed across successor attempts"
        )
    return C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_runtime_environment.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "terminal_record_count": len(supplied),
            "candidate_indices": indices,
            "physical_runtime_core": stage,
            "physical_runtime_core_sha256": validated[0][
                "stage_runtime_sha256"
            ],
            "backend_runtime_core": backend,
            "backend_runtime_core_sha256": validated[0][
                "backend_runtime_core_sha256"
            ],
            "stage_runtime_sha256s": [
                item["stage_runtime_sha256"] for item in validated
            ],
            "backend_runtime_sha256s": [
                item["backend_runtime_sha256"] for item in validated
            ],
            "all_stage_runtime_cores_equal": True,
            "all_backend_runtime_cores_equal": True,
            "fake_runtime": bool(stage["fake_runtime"]),
            "models_trained": 0,
        }
    )


def validate_generator_runtime_environment(
    value: Any, runtime_contract: Mapping[str, Any], *,
    metadata_rows: Sequence[Mapping[str, Any]], allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    row = _mapping(
        value, GENERATOR_RUNTIME_ENVIRONMENT_FIELDS,
        "generator runtime environment",
    )
    C.validate_content_digest(row)
    expected = build_generator_runtime_environment(
        metadata_rows, runtime_contract, allow_fake_runtime=allow_fake_runtime
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "generator runtime environment drift"
        )
    return row


_MATERIAL_VALIDATION_FIELDS = frozenset(
    {
        "terminal_record_count",
        "metadata_rows",
        "material_validations",
        "generator_runtime_environment",
        "common_expected_material_files",
        "common_expected_material_directories",
        "observed_material_files",
        "observed_material_directories",
        "material_inventory_projection",
        "downstream_validation",
        "models_trained",
    }
)
_DOWNSTREAM_VALIDATION_FIELDS = frozenset(
    {
        "panel_documents",
        "selected_material",
        "encoding_material",
        "fanout_material",
        "candidate_fanout_rows",
        "development_target_selection",
        "heldout_scores",
        "repeatability_material",
        "repeatability_rows",
    }
)
COMMON_RECOMPUTE_EVIDENCE_KEYS = frozenset(
    {
        "runtime_contract",
        "v4_context",
        "generator_stream_manifest",
        "generator_terminal_records_jsonl",
        "generator_terminal_records",
        "generator_metrics",
        "common_bindings",
        "source_freeze_observation",
        "material_root",
        "material_validation",
    }
)
SUCCESS_RECOMPUTE_EVIDENCE_KEYS = frozenset(
    set(COMMON_RECOMPUTE_EVIDENCE_KEYS)
    | set(C.DOWNSTREAM_OUTPUT_LEAVES)
    | {f"{leaf}:raw" for leaf in C.DOWNSTREAM_OUTPUT_LEAVES}
)


def _canonical_file_binding(path: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nlink": 1,
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }


def _validate_exact_canonical_file_binding(
    value: Any, *, path: str, raw: bytes, label: str
) -> dict[str, Any]:
    binding = _file_binding(value, label)
    if binding != _canonical_file_binding(path, raw):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            f"{label} content/byte binding drift"
        )
    return binding


def _validate_common_scientific_bindings(
    value: Any,
    *,
    runtime_contract: Mapping[str, Any],
    v4_context: Mapping[str, Any],
    stream_manifest: Mapping[str, Any],
    terminal_raw: bytes,
    generator_metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_raw = {
        "contract.json": C.canonical_json_bytes(runtime_contract),
        "V4_context.json": C.canonical_json_bytes(v4_context),
        "generator_stream_manifest.json": C.canonical_json_bytes(stream_manifest),
        "generator_terminal_records.jsonl": terminal_raw,
        "generator_metrics.json": C.canonical_json_bytes(generator_metrics),
    }
    if not isinstance(value, Mapping) or set(value) != set(expected_raw):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "common scientific binding inventory drift"
        )
    result: dict[str, dict[str, Any]] = {}
    for path, raw in expected_raw.items():
        binding = _file_binding(value[path], f"common scientific binding {path}")
        if binding != _canonical_file_binding(path, raw):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"common scientific binding {path} content drift"
            )
        result[path] = binding
    return result


def _validate_material_reduction(
    value: Any,
    *,
    records: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    panel_available: bool,
    material_root: str,
    allow_fake_runtime: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    material = _mapping(value, _MATERIAL_VALIDATION_FIELDS, "material reduction")
    if material["models_trained"] != 0:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material reduction reports model training"
        )
    validations = _sequence(
        material["material_validations"], "qualification material validations"
    )
    metadata_rows = _sequence(material["metadata_rows"], "qualification metadata rows")
    if (
        material["terminal_record_count"] != len(records)
        or len(validations) != len(records)
        or len(metadata_rows) != len(records)
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "qualification material population drift"
        )
    rebuilt: list[dict[str, Any]] = []
    for offset, (record, wrapper, supplied_metadata) in enumerate(
        zip(records, validations, metadata_rows)
    ):
        if not isinstance(wrapper, Mapping) or set(wrapper) != {"metadata", "arrays"}:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"qualification validation wrapper[{offset}] drift"
            )
        candidate_index = int(record["candidate_index"])
        validated = validate_qualification_material_shard(
            wrapper["metadata"],
            reopened_arrays=wrapper["arrays"],
            expected_pool_index=candidate_index,
        )
        metadata = validated["metadata"]
        if metadata != supplied_metadata:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "qualification metadata projection drift"
            )
        metadata_raw = C.canonical_json_bytes(metadata)
        expected_metadata = _canonical_file_binding(
            record["material_metadata_binding"]["path"], metadata_raw
        )
        if (
            record["material_metadata_binding"] != expected_metadata
            or metadata["payload"]
            != {
                "role": "material_shard_payload",
                **record["material_payload_binding"],
                "kind": "npz",
            }
            or record["persisted_array_evidence_sha256"]
            != _canonical_no_lf_sha256(metadata["persisted_array_evidence"])
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "qualification terminal/material byte binding drift"
            )
        rebuilt.append(validated)
    runtime = validate_generator_runtime_environment(
        material["generator_runtime_environment"],
        runtime_contract,
        metadata_rows=metadata_rows,
        allow_fake_runtime=allow_fake_runtime,
    )
    inventory = validate_material_inventory_projection(
        material["material_inventory_projection"],
        attempted_candidate_count=len(records),
        panel_available=panel_available,
        expected_root=material_root,
    )
    expected_files = _sequence(
        material["common_expected_material_files"], "expected material files"
    )
    observed_files = _sequence(
        material["observed_material_files"], "observed material files"
    )
    expected_directories = _sequence(
        material["common_expected_material_directories"],
        "expected material directories",
    )
    observed_directories = _sequence(
        material["observed_material_directories"],
        "observed material directories",
    )
    if (
        expected_files != sorted(set(expected_files))
        or observed_files != expected_files
        or expected_directories != sorted(set(expected_directories))
        or observed_directories != expected_directories
        or len(observed_files) != inventory["file_count"]
        or len(observed_directories) != inventory["directory_count"]
        or [row["path"] for row in inventory["files"]] != observed_files
        or inventory["directories"] != observed_directories
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material inventory/file projection cross-link drift"
        )
    return material, rebuilt, runtime


def _validate_official_downstream_bytes(
    evidence: Mapping[str, Any], documents: Mapping[str, Any]
) -> None:
    for leaf in C.DOWNSTREAM_OUTPUT_LEAVES:
        value = evidence.get(leaf)
        expected = documents.get(leaf, value)
        if leaf.endswith(".jsonl"):
            raw = _canonical_jsonl_bytes(
                _sequence(expected, f"{leaf} official rows")
            )
        else:
            if not isinstance(expected, Mapping):
                raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                    f"{leaf} official document drift"
                )
            raw = C.canonical_json_bytes(expected)
        supplied_raw = evidence.get(f"{leaf}:raw")
        if isinstance(supplied_raw, str):
            supplied_raw = supplied_raw.encode("utf-8")
        if value != expected or supplied_raw != raw:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                f"{leaf} official byte projection drift"
            )


def _validate_success_downstream(
    evidence: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    generator: Mapping[str, Any],
    qualifications: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    downstream_value: Any,
    allow_fake_runtime: bool,
) -> dict[str, Any]:
    downstream = _mapping(
        downstream_value, _DOWNSTREAM_VALIDATION_FIELDS, "downstream validation"
    )
    handoff = build_panel_handoff(records, generator)
    documents = validate_frozen_panel_documents(
        downstream["panel_documents"],
        records,
        generator,
        handoff,
        qualifications,
        downstream["selected_material"],
        runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    _validate_official_downstream_bytes(evidence, {
        **documents,
        "candidate_fanout.jsonl": downstream["candidate_fanout_rows"],
        "development_target_selection.json": downstream[
            "development_target_selection"
        ],
        "heldout_scores.jsonl": downstream["heldout_scores"],
        "repeatability.jsonl": downstream["repeatability_rows"],
    })
    encoding_wrapper = downstream["encoding_material"]
    if not isinstance(encoding_wrapper, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "encoding validation wrapper drift"
        )
    encoding = validate_encoding_material_shard(
        encoding_wrapper["metadata"],
        reopened_arrays=encoding_wrapper["arrays"],
        panel_documents=documents,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
        material_metadata_binding=encoding_wrapper["material_metadata_binding"],
    )
    panel_states = documents["panel_manifest.json"]["states"]
    specs = {
        spec["state_id"]: spec
        for spec in handoff["selected_candidate_specs"]
    }
    selected = {
        wrapper["metadata"]["state_id"]: wrapper
        for wrapper in downstream["selected_material"]
    }
    ports = {
        row["state_id"]: row
        for row in documents["edge_port_index.json"]["records"]
    }
    fanout_wrappers = _sequence(downstream["fanout_material"], "fanout material")
    fanout: list[dict[str, Any]] = []
    fanout_by_state: dict[str, dict[str, Any]] = {}
    for state in panel_states:
        state_id = state["state_id"]
        matches = [
            wrapper
            for wrapper in fanout_wrappers
            if wrapper["metadata"].get("state_id") == state_id
        ]
        if len(matches) != 1 or state_id not in selected:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "fanout/selected state population drift"
            )
        wrapper = matches[0]
        validated = validate_fanout_material_shard(
            wrapper["metadata"],
            reopened_arrays=wrapper["arrays"],
            panel_state=state,
            candidate_spec=specs[state_id],
            selected_material=selected[state_id],
            directed_port_record=ports[state_id],
            runtime_contract=runtime_contract,
            allow_fake_runtime=allow_fake_runtime,
            material_metadata_binding=wrapper["material_metadata_binding"],
        )
        fanout.append(validated)
        fanout_by_state[state_id] = validated
    development_fanout = [
        wrapper for wrapper in fanout if wrapper["metadata"]["role"] == "DEVELOPMENT"
    ]
    selection = validate_development_target_selection(
        downstream["development_target_selection"],
        panel_documents=documents,
        encoding_material=encoding,
        fanout_material=development_fanout,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    heldout = validate_heldout_scores(
        downstream["heldout_scores"],
        panel_documents=documents,
        development_target_selection=selection,
        encoding_material=encoding,
        fanout_material=fanout,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    fanout_rows = validate_candidate_fanout_rows(
        downstream["candidate_fanout_rows"],
        panel_documents=documents,
        fanout_material=fanout,
        development_target_selection=selection,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    repeat_wrappers = _sequence(
        downstream["repeatability_material"], "repeatability material"
    )
    repeat: list[dict[str, Any]] = []
    for state in panel_states:
        if state["role"] != "DEVELOPMENT_HELDOUT":
            continue
        state_id = state["state_id"]
        matches = [
            wrapper
            for wrapper in repeat_wrappers
            if wrapper["metadata"].get("state_id") == state_id
        ]
        if len(matches) != 1:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "repeatability state population drift"
            )
        wrapper = matches[0]
        repeat.append(
            validate_repeatability_material_shard(
                wrapper["metadata"],
                reopened_arrays=wrapper["arrays"],
                panel_state=state,
                candidate_spec=specs[state_id],
                selected_material=selected[state_id],
                source_fanout_material=fanout_by_state[state_id],
                heldout_score_rows=heldout,
                directed_port_record=ports[state_id],
                runtime_contract=runtime_contract,
                allow_fake_runtime=allow_fake_runtime,
                material_metadata_binding=wrapper["material_metadata_binding"],
            )
        )
    repeat_rows = validate_repeatability_rows(
        downstream["repeatability_rows"],
        panel_documents=documents,
        candidate_fanout_rows=fanout_rows,
        heldout_score_rows=heldout,
        repeatability_material=repeat,
        runtime_contract=runtime_contract,
        allow_fake_runtime=allow_fake_runtime,
    )
    return {
        "documents": documents,
        "encoding": encoding,
        "fanout": fanout,
        "fanout_rows": fanout_rows,
        "selection": selection,
        "heldout": heldout,
        "repeatability": repeat,
        "repeatability_rows": repeat_rows,
    }


def _scientific_counters(
    generator: Mapping[str, Any], *, panel_available: bool
) -> dict[str, int]:
    counters = {
        "terminal_attempt_count": int(generator["terminal_record_count"]),
        "terminal_unique_identity_count": int(
            generator["terminal_unique_identity_count"]
        ),
        "qualified_count": int(generator["qualified_count"]),
        "nonqualified_count": int(generator["nonqualified_count"]),
        "teacher_execution_count": int(generator["teacher_execution_count"]),
        "hard_stop_count": int(generator["hard_stop_count"]),
        "selected_state_count": C.PANEL_STATE_COUNT if panel_available else 0,
        "candidate_branch_count": (
            C.PANEL_STATE_COUNT * len(C.CANDIDATE_IDS)
            if panel_available
            else 0
        ),
        "development_target_row_count": (
            48 * len(C.TARGET_IDS) if panel_available else 0
        ),
        "heldout_score_row_count": (
            16 * len(C.HELDOUT_CONDITION_IDS) if panel_available else 0
        ),
        "repeatability_row_count": (
            16 * len(C.REPEAT_BRANCH_IDS) * C.REPEATS_PER_BRANCH
            if panel_available
            else 0
        ),
        "models_trained": 0,
    }
    _mapping(counters, C.SCIENTIFIC_COUNTER_FIELDS, "scientific counters")
    return counters


def recompute_metrics(
    evidence: Mapping[str, Any], *, allow_fake_runtime: bool = False
) -> dict[str, Any]:
    """Independently reduce the complete successor evidence population."""

    if not isinstance(evidence, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "recompute evidence is not a mapping"
        )
    runtime = C.validate_runtime_contract(evidence.get("runtime_contract"))
    context = C.validate_v4_context(evidence.get("v4_context"))
    manifest = C.validate_generator_stream_manifest(
        evidence.get("generator_stream_manifest")
    )
    terminal_raw = evidence.get("generator_terminal_records_jsonl")
    if isinstance(terminal_raw, str):
        terminal_raw = terminal_raw.encode("utf-8")
    records = validate_generator_terminal_records_jsonl(terminal_raw)
    if evidence.get("generator_terminal_records") != records:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "terminal parsed/raw projection drift"
        )
    generator = validate_generator_metrics(
        evidence.get("generator_metrics"), records, manifest
    )
    _validate_common_scientific_bindings(
        evidence.get("common_bindings"),
        runtime_contract=runtime,
        v4_context=context,
        stream_manifest=manifest,
        terminal_raw=terminal_raw,
        generator_metrics=generator,
    )
    observation = C.validate_source_freeze_observation(
        evidence.get("source_freeze_observation"),
        source_freeze_commit=runtime["source_freeze_commit"],
    )
    material_root = evidence.get("material_root")
    if not isinstance(material_root, str) or not material_root.startswith("/"):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material root projection drift"
        )
    available = generator["status"] == C.GENERATOR_PANEL_AVAILABLE
    expected_evidence_keys = (
        SUCCESS_RECOMPUTE_EVIDENCE_KEYS
        if available
        else COMMON_RECOMPUTE_EVIDENCE_KEYS
    )
    if set(evidence) != set(expected_evidence_keys):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "recompute evidence inventory drift"
        )
    material, qualifications, generator_runtime = _validate_material_reduction(
        evidence.get("material_validation"),
        records=records,
        runtime_contract=runtime,
        panel_available=available,
        material_root=material_root,
        allow_fake_runtime=allow_fake_runtime,
    )
    if available:
        if material["downstream_validation"] is None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "available generator lacks downstream evidence"
            )
        downstream_validation = _validate_success_downstream(
            evidence,
            records=records,
            generator=generator,
            qualifications=qualifications,
            runtime_contract=runtime,
            downstream_value=material["downstream_validation"],
            allow_fake_runtime=allow_fake_runtime,
        )
        reduction = _success_downstream_reduction(
            downstream_validation["documents"],
            downstream_validation["selection"],
            downstream_validation["fanout_rows"],
            downstream_validation["heldout"],
            downstream_validation["repeatability_rows"],
            downstream_validation["encoding"]["metadata"],
        )
        panel = reduction["panel"]
        downstream = reduction["downstream"]
        runtime_environments = reduction["runtime_environments"]
        primary = reduction["primary_classification"]
        secondary = reduction["secondary_classifications"]
        next_decision = reduction["next_decision"]
    else:
        if material["downstream_validation"] is not None:
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "generator-terminal branch contains downstream evidence"
            )
        panel = None
        downstream = None
        runtime_environments = {
            "physical": copy.deepcopy(generator_runtime["physical_runtime_core"]),
            "encoder": None,
            "ranker": None,
            "any_fake_runtime": bool(generator_runtime["fake_runtime"]),
        }
        primary = generator["primary_classification"]
        secondary = []
        next_decision = generator["next_decision"]
    counters = _scientific_counters(generator, panel_available=available)
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1.metrics.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "generator_status": generator["status"],
            "primary_classification": primary,
            "secondary_classifications": secondary,
            "next_decision": next_decision,
            "generator_metrics": copy.deepcopy(generator),
            "generator_runtime_environment": copy.deepcopy(generator_runtime),
            "source_freeze_observation": copy.deepcopy(observation),
            "material_inventory_projection": copy.deepcopy(
                material["material_inventory_projection"]
            ),
            "panel": panel,
            "downstream": downstream,
            "runtime_environments": runtime_environments,
            "scientific_counters": counters,
            "models_trained": 0,
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )
    if set(result) != set(C.SUCCESSOR_METRICS_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "successor metrics field drift"
        )
    return result


def validate_recomputed_metrics(value: Any) -> dict[str, Any]:
    row = _mapping(value, C.SUCCESSOR_METRICS_FIELDS, "successor metrics")
    C.validate_content_digest(row)
    generator = _mapping(
        row["generator_metrics"], C.GENERATOR_METRICS_FIELDS, "generator metrics"
    )
    C.validate_content_digest(generator)
    generator_runtime = _mapping(
        row["generator_runtime_environment"],
        GENERATOR_RUNTIME_ENVIRONMENT_FIELDS,
        "generator runtime environment",
    )
    C.validate_content_digest(generator_runtime)
    observation = C.validate_source_freeze_observation(
        row["source_freeze_observation"],
        source_freeze_commit=generator_runtime["source_freeze_commit"],
    )
    available = generator["status"] == C.GENERATOR_PANEL_AVAILABLE
    inventory_value = row["material_inventory_projection"]
    if not isinstance(inventory_value, Mapping):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "material inventory projection is not mapping"
        )
    inventory_root = inventory_value.get("root")
    inventory = validate_material_inventory_projection(
        inventory_value,
        attempted_candidate_count=generator["terminal_record_count"],
        panel_available=available,
        expected_root=inventory_root,
    )
    counters = _mapping(
        row["scientific_counters"], C.SCIENTIFIC_COUNTER_FIELDS, "scientific counters"
    )
    expected_counters = _scientific_counters(generator, panel_available=available)
    runtime_environments = row["runtime_environments"]
    if not isinstance(runtime_environments, Mapping) or set(runtime_environments) != {
        "physical",
        "encoder",
        "ranker",
        "any_fake_runtime",
    }:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "successor runtime environment field drift"
        )
    if (
        row["schema"]
        != "physical_handoff_stratified_generator_successor_v1.metrics.v1"
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["generator_status"] != generator["status"]
        or generator_runtime["terminal_record_count"]
        != generator["terminal_record_count"]
        or generator_runtime["models_trained"] != 0
        or observation != row["source_freeze_observation"]
        or inventory != row["material_inventory_projection"]
        or row["models_trained"] != 0
        or row["development_only"] is not True
        or row["final_evaluation_eligible"] is not False
        or counters != expected_counters
        or (row["panel"] is None) is available
        or (row["downstream"] is None) is available
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "successor metrics semantic drift"
        )
    if available:
        if (
            row["primary_classification"] not in C.PRIMARY_CLASSIFICATIONS
            or row["next_decision"]
            != C.NEXT_DECISION_BY_CLASSIFICATION[
                row["primary_classification"]
            ]
            or any(
                value not in C.SECONDARY_CLASSIFICATIONS
                for value in row["secondary_classifications"]
            )
            or runtime_environments["physical"]
            != generator_runtime["physical_runtime_core"]
            or runtime_environments["encoder"] is None
            or runtime_environments["ranker"] is None
            or runtime_environments["any_fake_runtime"]
            is not bool(
                generator_runtime["fake_runtime"]
                or runtime_environments["encoder"]["fake_runtime"]
                or runtime_environments["ranker"]["fake_runtime"]
            )
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "successor downstream classification drift"
            )
    else:
        if (
            row["primary_classification"] != generator["primary_classification"]
            or row["secondary_classifications"] != []
            or row["next_decision"] != generator["next_decision"]
            or row["panel"] is not None
            or row["downstream"] is not None
            or runtime_environments["physical"]
            != generator_runtime["physical_runtime_core"]
            or runtime_environments["encoder"] is not None
            or runtime_environments["ranker"] is not None
            or runtime_environments["any_fake_runtime"]
            is not bool(generator_runtime["fake_runtime"])
        ):
            raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
                "generator-terminal metrics drift"
            )
    return row


def _scientific_bindings(
    value: Any, *, metrics: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    available = metrics["generator_status"] == C.GENERATOR_PANEL_AVAILABLE
    leaves = C.SUCCESS_OUTPUT_LEAVES if available else C.GENERATOR_TERMINAL_OUTPUT_LEAVES
    expected = set(leaves) - {"result.json", "result.md", "file_hashes.json"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "scientific publication binding inventory drift"
        )
    rows = {
        path: _file_binding(binding, f"scientific publication binding {path}")
        for path, binding in value.items()
    }
    if any(rows[path]["path"] != path for path in rows):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "scientific publication binding path drift"
        )
    metrics_raw = C.canonical_json_bytes(metrics)
    if rows["metrics.json"] != _canonical_file_binding("metrics.json", metrics_raw):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "metrics publication binding drift"
        )
    return rows


def _generator_result_summary(generator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": generator["status"],
        "terminal_record_count": generator["terminal_record_count"],
        "terminal_unique_identity_count": generator[
            "terminal_unique_identity_count"
        ],
        "expected_stream_count": generator["expected_stream_count"],
        "completed_stream_count": generator["completed_stream_count"],
        "missing_stream_count": generator["missing_stream_count"],
        "duplicated_candidate_identity_count": generator[
            "duplicated_candidate_identity_count"
        ],
        "qualified_count": generator["qualified_count"],
        "nonqualified_count": generator["nonqualified_count"],
        "qualification_rate": generator["qualification_rate"],
        "rejection_rate": generator["rejection_rate"],
        "teacher_execution_count": generator["teacher_execution_count"],
        "hard_stop_count": generator["hard_stop_count"],
        "disposition_counts": copy.deepcopy(generator["disposition_counts"]),
        "disposition_rates": copy.deepcopy(generator["disposition_rates"]),
        "zero_yield_streams": copy.deepcopy(generator["zero_yield_streams"]),
        "low_yield_streams": copy.deepcopy(generator["low_yield_streams"]),
        "target_reached_streams": copy.deepcopy(
            generator["target_reached_streams"]
        ),
        "family_rows": copy.deepcopy(generator["family_rows"]),
        "stratum_factor_rows": copy.deepcopy(generator["stratum_factor_rows"]),
        "v4_shortfall_resolution": copy.deepcopy(
            generator["v4_shortfall_resolution"]
        ),
        "next_decision_evidence": copy.deepcopy(
            generator["next_decision_evidence"]
        ),
    }


def build_result_document(
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = validate_recomputed_metrics(recomputed_metrics)
    bindings = _scientific_bindings(scientific_bindings, metrics=metrics)
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1.result.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "status": "COMPLETE",
            "primary_classification": metrics["primary_classification"],
            "secondary_classifications": copy.deepcopy(
                metrics["secondary_classifications"]
            ),
            "next_decision": metrics["next_decision"],
            "metrics_content_digest": metrics["content_digest"],
            "generator_status": metrics["generator_status"],
            "generator_summary": _generator_result_summary(
                metrics["generator_metrics"]
            ),
            "panel_summary": copy.deepcopy(metrics["panel"]),
            "downstream_summary": copy.deepcopy(metrics["downstream"]),
            "runtime_environments": copy.deepcopy(metrics["runtime_environments"]),
            "source_freeze_observation": copy.deepcopy(
                metrics["source_freeze_observation"]
            ),
            "material_inventory_projection": copy.deepcopy(
                metrics["material_inventory_projection"]
            ),
            "scientific_bindings": bindings,
            "models_trained": 0,
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )
    if set(result) != set(C.RESULT_DOCUMENT_FIELDS):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "result document field drift"
        )
    return result


def validate_result_document(
    value: Any,
    *,
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = _mapping(value, C.RESULT_DOCUMENT_FIELDS, "successor result")
    C.validate_content_digest(row)
    expected = build_result_document(recomputed_metrics, scientific_bindings)
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "result document drift"
        )
    return row


def build_result_publication_projection(
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = validate_recomputed_metrics(recomputed_metrics)
    bindings = _scientific_bindings(scientific_bindings, metrics=metrics)
    result = build_result_document(metrics, bindings)
    projection = {
        "schema": (
            "physical_handoff_stratified_generator_successor_v1."
            "result_publication_projection.v1"
        ),
        "experiment_id": C.EXPERIMENT_ID,
        "branch": (
            "SUCCESS"
            if metrics["generator_status"] == C.GENERATOR_PANEL_AVAILABLE
            else "GENERATOR_TERMINAL"
        ),
        "metrics": metrics,
        "scientific_bindings": bindings,
        "result_document": result,
    }
    _mapping(
        projection,
        C.RESULT_PUBLICATION_PROJECTION_FIELDS,
        "result publication projection",
    )
    return projection


def validate_result_publication_projection(
    value: Any,
    *,
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supplied = _mapping(value, C.RESULT_DOCUMENT_FIELDS, "publication result")
    C.validate_content_digest(supplied)
    expected = build_result_publication_projection(
        recomputed_metrics, scientific_bindings
    )
    if C.canonical_json_bytes(supplied) != C.canonical_json_bytes(
        expected["result_document"]
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "publication result projection drift"
        )
    return expected


def build_result_report(
    result_document: Mapping[str, Any], recomputed_metrics: Mapping[str, Any]
) -> str:
    result = _mapping(result_document, C.RESULT_DOCUMENT_FIELDS, "report result")
    C.validate_content_digest(result)
    metrics = validate_recomputed_metrics(recomputed_metrics)
    if result["metrics_content_digest"] != metrics["content_digest"]:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "report result/metrics binding drift"
        )
    generator = metrics["generator_metrics"]
    stream_lines = "\n".join(
        (
            f"- {row['family']} stratum {row['stratum_index']}: "
            f"attempts={row['attempt_count']}, qualified={row['qualified_count']}, "
            f"qualification_rate={row['qualification_rate']:.17g}, "
            f"rejection_rate={row['rejection_rate']:.17g}, "
            f"termination={row['termination_reason']}"
        )
        for row in generator["stream_rows"]
    )
    failure_lines = "\n".join(
        f"- {json.dumps(row, sort_keys=True, separators=(',', ':'))}"
        for row in generator["next_decision_evidence"]["failure_streams"]
    ) or "- none"
    panel_text = (
        "Panel unavailable; every downstream model and candidate evaluation was blocked."
        if metrics["panel"] is None
        else json.dumps(metrics["panel"], sort_keys=True, separators=(",", ":"))
    )
    downstream_text = (
        "Not executed."
        if metrics["downstream"] is None
        else json.dumps(metrics["downstream"], sort_keys=True, separators=(",", ":"))
    )
    text = f"""# Stratified physical handoff generator successor V1

## Result

Generator status: {result['generator_status']}

Primary classification: {result['primary_classification']}

Secondary classifications: {json.dumps(result['secondary_classifications'], separators=(',', ':'))}

Next decision: {result['next_decision'] or 'none'}

Development-only: true; final-evaluation eligible: false.

## Generator accounting

Attempts: {generator['terminal_record_count']} unique identities: {generator['terminal_unique_identity_count']} qualified: {generator['qualified_count']} nonqualified: {generator['nonqualified_count']} teacher executions: {generator['teacher_execution_count']} hard technical stops: {generator['hard_stop_count']}.

Qualification rate: {generator['qualification_rate']:.17g}; rejection rate: {generator['rejection_rate']:.17g}.

Disposition counts: {json.dumps(generator['disposition_counts'], sort_keys=True, separators=(',', ':'))}

Disposition rates: {json.dumps(generator['disposition_rates'], sort_keys=True, separators=(',', ':'))}

Family metrics: {json.dumps(generator['family_rows'], sort_keys=True, separators=(',', ':'))}

Frozen-factor metrics: {json.dumps(generator['stratum_factor_rows'], sort_keys=True, separators=(',', ':'))}

V4 canonical-shortfall resolution: {json.dumps(generator['v4_shortfall_resolution'], sort_keys=True, separators=(',', ':'))}

### Per-stream yields

{stream_lines}

### Below-target stream evidence

{failure_lines}

## V4 interpretation and preserved boundary

{C.V4_INTERPRETATION_SENTENCES[0]}

{C.V4_INTERPRETATION_SENTENCES[1]}

## Panel

{panel_text}

## Downstream physical handoff

Held-out comparator alias: PHYSICAL_TEACHER -> TEACHER_TRACE (user-facing comparator 4 -> frozen internal condition_id; no new teacher execution).

{downstream_text}

## Runtime, source, material, and training

Runtime environments: {json.dumps(result['runtime_environments'], sort_keys=True, separators=(',', ':'))}

Source-freeze observation: {json.dumps(result['source_freeze_observation'], sort_keys=True, separators=(',', ':'))}

Material inventory: {json.dumps(result['material_inventory_projection'], sort_keys=True, separators=(',', ':'))}

Models trained: 0.

## Claims boundary

Concerns: {json.dumps(C.CLAIMS_BOUNDARY['concerns'], separators=(',', ':'))}

Does not establish: {json.dumps(C.CLAIMS_BOUNDARY['does_not_establish'], separators=(',', ':'))}
"""
    return text if text.endswith("\n") else text + "\n"


def build_result_report_bytes(value: Any) -> bytes:
    projection = _mapping(
        value,
        C.RESULT_PUBLICATION_PROJECTION_FIELDS,
        "result publication projection",
    )
    if (
        projection["schema"]
        != "physical_handoff_stratified_generator_successor_v1.result_publication_projection.v1"
        or projection["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "result publication projection identity drift"
        )
    metrics = validate_recomputed_metrics(projection["metrics"])
    bindings = _scientific_bindings(
        projection["scientific_bindings"], metrics=metrics
    )
    result = validate_result_document(
        projection["result_document"],
        recomputed_metrics=metrics,
        scientific_bindings=bindings,
    )
    expected_branch = (
        "SUCCESS"
        if metrics["generator_status"] == C.GENERATOR_PANEL_AVAILABLE
        else "GENERATOR_TERMINAL"
    )
    if projection["branch"] != expected_branch:
        raise PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError(
            "result publication branch drift"
        )
    return build_result_report(result, metrics).encode("utf-8")


def reducer_authority() -> dict[str, Any]:
    return C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "reducer_authority.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "v4_context_authority_content_digest": C.V4_CONTEXT_AUTHORITY[
                "content_digest"
            ],
            "seed_derivation_authority_content_digest": C.SEED_DERIVATION_AUTHORITY[
                "content_digest"
            ],
            "allocation_authority_content_digest": C.GENERATOR_ALLOCATION_AUTHORITY[
                "content_digest"
            ],
            "v4_scientific_invariance_authority_content_digest": C.V4_SCIENTIFIC_INVARIANCE_AUTHORITY[
                "content_digest"
            ],
            "terminal_record_fields": sorted(C.GENERATOR_TERMINAL_RECORD_FIELDS),
            "generator_metrics_fields": sorted(C.GENERATOR_METRICS_FIELDS),
            "successor_metrics_fields": sorted(C.SUCCESSOR_METRICS_FIELDS),
            "scientific_counter_fields": sorted(C.SCIENTIFIC_COUNTER_FIELDS),
            "panel_handoff_fields": sorted(C.PANEL_HANDOFF_FIELDS),
            "result_fields": sorted(C.RESULT_DOCUMENT_FIELDS),
            "result_publication_projection_fields": sorted(
                C.RESULT_PUBLICATION_PROJECTION_FIELDS
            ),
            "success_output_inventory": list(C.SUCCESS_OUTPUT_LEAVES),
            "generator_terminal_output_inventory": list(
                C.GENERATOR_TERMINAL_OUTPUT_LEAVES
            ),
            "physical_evidence_validator": (
                "native successor outer-schema validators independently reduce "
                "reopened arrays; registry-independent frozen V4 numerical and "
                "semantic helpers are called directly without namespace cloning, "
                "schema translation, or cross-version serialization gates"
            ),
            "material_layout_authority": copy.deepcopy(
                C.MATERIAL_LAYOUT_AUTHORITY
            ),
            "prohibited_external_publication_authority": copy.deepcopy(
                C.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY
            ),
            "external_regeneration_or_custody_receipt": False,
            "models_trained": 0,
        }
    )


__all__ = [
    "COMMON_RECOMPUTE_EVIDENCE_KEYS",
    "GENERATOR_RUNTIME_ENVIRONMENT_FIELDS",
    "PhysicalHandoffStratifiedGeneratorSuccessorV1HardStop",
    "PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError",
    "SUCCESS_RECOMPUTE_EVIDENCE_KEYS",
    "build_development_target_selection",
    "build_frozen_panel_documents",
    "build_generator_metrics",
    "build_generator_runtime_environment",
    "build_generator_terminal_record",
    "build_generator_terminal_records_jsonl",
    "build_heldout_scores",
    "build_panel_handoff",
    "build_persisted_array_evidence",
    "build_result_document",
    "build_result_publication_projection",
    "build_result_report",
    "build_result_report_bytes",
    "build_state_disposition_record",
    "canonical_semantic_snapshot",
    "canonical_v1_planar_segment_projection",
    "classify_state_disposition",
    "compare_behavioural_probe_traces",
    "compare_terminated_behavioural_probe_traces",
    "compare_tipped_behavioural_probe_traces",
    "derive_candidate_port_metrics",
    "first_registered_port_crossing",
    "frozen_binary64_fma",
    "persisted_array_bytes_sha256",
    "persisted_array_manifest_row",
    "point_in_polygon_inclusive",
    "recompute_metrics",
    "reduce_qualification_teacher_trace",
    "reducer_authority",
    "runtime_environment_sha256",
    "select_failure_next_decision",
    "semantic_snapshot_sha256",
    "snapshot_behavioural_digest",
    "snapshot_semantic_evidence",
    "validate_candidate_fanout_rows",
    "validate_development_target_selection",
    "validate_encoding_material_shard",
    "validate_fanout_material_shard",
    "validate_frozen_panel_documents",
    "validate_generator_metrics",
    "validate_generator_runtime_environment",
    "validate_generator_terminal_record",
    "validate_generator_terminal_records_jsonl",
    "validate_heldout_scores",
    "validate_material_inventory_projection",
    "validate_npz_archive_comment",
    "validate_panel_handoff",
    "validate_persisted_array_evidence",
    "validate_physical_runtime_environment",
    "validate_qualification_material_shard",
    "validate_qualification_teacher_raw_evidence",
    "validate_recomputed_metrics",
    "validate_repeatability_material_shard",
    "validate_repeatability_rows",
    "validate_result_document",
    "validate_result_publication_projection",
    "validate_selected_reset_material_shard",
    "validate_snapshot_previous_applied_command_binding",
    "validate_state_disposition_record",
    "validate_state_material_payload",
]
