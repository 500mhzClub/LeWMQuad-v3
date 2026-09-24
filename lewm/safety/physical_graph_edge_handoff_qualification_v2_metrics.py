"""Pure V2 validation and exact V1-science metric wrappers.

This module contains no torch, simulator, encoder, or ranker dependency.  The
V1 scientific validators and reducer remain the formula authority; V2 maps
only experiment/schema identity and the corrected persisted-byte digest
evidence into that frozen authority.  The separately bound canonical-port
heading and candidate-port-field alignments merely make the inherited writer
satisfy that same frozen authority and do not relax any validator or metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import io
import math
from typing import Any, Callable

from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as V1M
from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as C


class PhysicalGraphEdgeHandoffV2MetricsError(ValueError):
    """Raised when corrected evidence is incomplete or drifts."""


_V1_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v1"
_V2_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v2"


def _mapping(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} field drift")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} is not a sequence")
    return [copy.deepcopy(item) for item in value]


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} is not SHA-256")
    return value


def _identity_string(value: str, *, to_v1: bool) -> str:
    if to_v1:
        if value == C.EXPERIMENT_ID:
            return C.V1_EXPERIMENT_ID
        return value.replace(_V2_SCHEMA_TOKEN, _V1_SCHEMA_TOKEN)
    if value == C.V1_EXPERIMENT_ID:
        return C.EXPERIMENT_ID
    return value.replace(_V1_SCHEMA_TOKEN, _V2_SCHEMA_TOKEN)


def _map_identity(value: Any, *, to_v1: bool) -> Any:
    """Map only V1/V2 experiment and schema identity, preserving all science."""

    if isinstance(value, Mapping):
        has_digest = "content_digest" in value
        if has_digest:
            # Reject an invalid V2 input before rebuilding the digest required
            # by the V1 validator after identity projection.
            if to_v1:
                C.validate_content_digest(value)
        mapped = {
            str(key): _map_identity(item, to_v1=to_v1)
            for key, item in value.items()
            if key != "content_digest"
        }
        if has_digest:
            attach = C.V1.attach_content_digest if to_v1 else C.attach_content_digest
            mapped = attach(mapped)
        return mapped
    if isinstance(value, (tuple, list)):
        return [_map_identity(item, to_v1=to_v1) for item in value]
    if isinstance(value, str):
        return _identity_string(value, to_v1=to_v1)
    return copy.deepcopy(value)


def _require_v2_document(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} is not a mapping")
    row = copy.deepcopy(dict(value))
    if row.get("experiment_id") != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} experiment identity drift")
    schema = row.get("schema")
    if not isinstance(schema, str) or _V2_SCHEMA_TOKEN not in schema:
        raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} schema identity drift")
    if "content_digest" in row:
        C.validate_content_digest(row)
    return row


def project_v1_evidence_to_v2(value: Any) -> Any:
    """Public identity-only adapter; no scientific value is changed."""

    return _map_identity(value, to_v1=False)


def project_v2_evidence_to_v1(value: Any) -> Any:
    """Public identity-only adapter used before frozen V1 validation."""

    return _map_identity(value, to_v1=True)


def _call_v1_document(
    validator: Callable[..., Any], value: Any, *dependencies: Any, label: str
) -> Any:
    row = _require_v2_document(value, label)
    try:
        validator(
            _map_identity(row, to_v1=True),
            *[_map_identity(item, to_v1=True) for item in dependencies],
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            f"{label} violates frozen V1 scientific authority: {exc}"
        ) from exc
    return row


def persisted_array_bytes_sha256(value: Any) -> str:
    """Hash exact persisted C-order bytes, with dtype/shape bound separately."""

    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise PhysicalGraphEdgeHandoffV2MetricsError("object array is forbidden")
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def derive_candidate_port_metrics(
    reset_pose_world: Sequence[Any],
    base_pose_world: Sequence[Sequence[Any]],
    directed_port_world: Sequence[Any],
) -> dict[str, Any]:
    """Apply the frozen directed-port progress/lateral formulas exactly."""

    def values(value: Any, label: str) -> list[Any]:
        if isinstance(value, (str, bytes, bytearray, Mapping)):
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                f"{label} is not an array sequence"
            )
        try:
            return list(value)
        except TypeError as exc:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                f"{label} is not an array sequence"
            ) from exc

    reset = values(reset_pose_world, "reset pose")
    poses = values(base_pose_world, "candidate base poses")
    port = values(directed_port_world, "directed port")
    if len(reset) < 2 or len(port) != 3 or len(poses) != C.PHYSICS_STEPS_PER_BRANCH:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "candidate port metric input dimensions drift"
        )

    def finite(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} is not numeric")
        result = float(value)
        if not math.isfinite(result):
            raise PhysicalGraphEdgeHandoffV2MetricsError(f"{label} is non-finite")
        return result

    reset_x = finite(reset[0], "reset x")
    reset_y = finite(reset[1], "reset y")
    port_x = finite(port[0], "port x")
    port_y = finite(port[1], "port y")
    port_heading = finite(port[2], "port heading")
    xy: list[tuple[float, float]] = []
    for index, pose in enumerate(poses):
        row = values(pose, f"base pose[{index}]")
        if len(row) < 2:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "candidate base-pose dimension drift"
            )
        xy.append(
            (
                finite(row[0], f"base pose[{index}].x"),
                finite(row[1], f"base pose[{index}].y"),
            )
        )
    progress = math.hypot(reset_x - port_x, reset_y - port_y) - min(
        math.hypot(x - port_x, y - port_y) for x, y in xy
    )
    endpoint_x, endpoint_y = xy[-1]
    lateral = abs(
        -(endpoint_x - port_x) * math.sin(port_heading)
        + (endpoint_y - port_y) * math.cos(port_heading)
    )
    return {
        "port_progress_m": progress,
        "lateral_error_m": lateral,
        "positive_port_progress": progress > 0.0,
    }


def validate_npz_archive_comment(value: bytes | str) -> str:
    """Validate the V2-only ZIP container provenance marker."""

    if isinstance(value, bytes):
        try:
            observed = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "NPZ archive comment is not UTF-8"
            ) from exc
    elif isinstance(value, str):
        observed = value
    else:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "NPZ archive comment has invalid type"
        )
    if observed != C.NPZ_ARCHIVE_COMMENT:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "NPZ archive provenance comment drift"
        )
    return observed


def persisted_array_manifest_row(member: str, value: Any) -> dict[str, Any]:
    import numpy as np

    if not isinstance(member, str) or not member:
        raise PhysicalGraphEdgeHandoffV2MetricsError("array member is empty")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise PhysicalGraphEdgeHandoffV2MetricsError("object array is forbidden")
    canonical = np.ascontiguousarray(array)
    if canonical.dtype != array.dtype:
        raise PhysicalGraphEdgeHandoffV2MetricsError("array normalization changed dtype")
    return {
        "member": member,
        "dtype_str": canonical.dtype.str,
        "shape": [int(size) for size in canonical.shape],
        "c_contiguous": True,
        "array_bytes_sha256": persisted_array_bytes_sha256(canonical),
    }


def _inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(C.canonical_json_bytes(list(rows))[:-1]).hexdigest()


def build_persisted_array_evidence(
    *,
    shard_kind: str,
    shard_id: str,
    payload_file: Mapping[str, Any],
    arrays: Mapping[str, Any],
    reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact per-shard pre-save/post-reopen array manifest."""

    payload = _mapping(payload_file, C.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS, "payload_file")
    if not isinstance(payload["path"], str) or not payload["path"]:
        raise PhysicalGraphEdgeHandoffV2MetricsError("payload path is empty")
    if (
        not isinstance(payload["bytes"], int)
        or isinstance(payload["bytes"], bool)
        or payload["bytes"] <= 0
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("payload byte count is invalid")
    _sha(payload["sha256"], "payload sha256")
    if not isinstance(arrays, Mapping) or not arrays:
        raise PhysicalGraphEdgeHandoffV2MetricsError("persisted arrays are empty")
    if set(arrays) != set(reopened_arrays):
        raise PhysicalGraphEdgeHandoffV2MetricsError("save/reopen member inventory drift")
    rows: list[dict[str, Any]] = []
    for member in sorted(arrays):
        before = persisted_array_manifest_row(str(member), arrays[member])
        after = persisted_array_manifest_row(str(member), reopened_arrays[member])
        if before != after:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                f"save/reopen persisted array drift: {member}"
            )
        rows.append(before)
    return {
        "schema": "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "shard_kind": str(shard_kind),
        "shard_id": str(shard_id),
        "payload_file": payload,
        "array_count": len(rows),
        "array_inventory_sha256": _inventory_sha256(rows),
        "arrays": rows,
        "save_reopen_validation_passed": True,
    }


def validate_persisted_array_evidence(
    value: Any, *, reopened_arrays: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    import numpy as np

    row = _mapping(value, C.PERSISTED_ARRAY_EVIDENCE_FIELDS, "persisted_array_evidence")
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV2MetricsError("persisted array evidence identity drift")
    if not isinstance(row["shard_kind"], str) or not row["shard_kind"]:
        raise PhysicalGraphEdgeHandoffV2MetricsError("shard_kind is empty")
    if not isinstance(row["shard_id"], str) or not row["shard_id"]:
        raise PhysicalGraphEdgeHandoffV2MetricsError("shard_id is empty")
    payload = _mapping(
        row["payload_file"], C.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS, "payload_file"
    )
    if not isinstance(payload["path"], str) or not payload["path"]:
        raise PhysicalGraphEdgeHandoffV2MetricsError("payload path is empty")
    if (
        not isinstance(payload["bytes"], int)
        or isinstance(payload["bytes"], bool)
        or payload["bytes"] <= 0
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("payload byte count is invalid")
    _sha(payload["sha256"], "payload sha256")
    arrays = _sequence(row["arrays"], "persisted arrays")
    if (
        not isinstance(row["array_count"], int)
        or isinstance(row["array_count"], bool)
        or row["array_count"] != len(arrays)
        or not arrays
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("persisted array count drift")
    validated_rows: list[dict[str, Any]] = []
    members: list[str] = []
    for index, item in enumerate(arrays):
        part = _mapping(item, C.PERSISTED_ARRAY_ROW_FIELDS, f"array[{index}]")
        member = part["member"]
        if not isinstance(member, str) or not member:
            raise PhysicalGraphEdgeHandoffV2MetricsError("array member is empty")
        if not isinstance(part["dtype_str"], str) or not part["dtype_str"]:
            raise PhysicalGraphEdgeHandoffV2MetricsError("array dtype is empty")
        try:
            dtype = np.dtype(part["dtype_str"])
        except TypeError as exc:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "array dtype is invalid"
            ) from exc
        if dtype.hasobject or dtype.str != part["dtype_str"]:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "array dtype is not exact numpy dtype.str"
            )
        if (
            not isinstance(part["shape"], list)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in part["shape"]
            )
        ):
            raise PhysicalGraphEdgeHandoffV2MetricsError("array shape is invalid")
        if part["c_contiguous"] is not True:
            raise PhysicalGraphEdgeHandoffV2MetricsError("array is not C-contiguous")
        _sha(part["array_bytes_sha256"], f"array[{index}] bytes sha256")
        members.append(member)
        validated_rows.append(part)
    if members != sorted(members) or len(set(members)) != len(members):
        raise PhysicalGraphEdgeHandoffV2MetricsError("array member order/uniqueness drift")
    if row["array_inventory_sha256"] != _inventory_sha256(validated_rows):
        raise PhysicalGraphEdgeHandoffV2MetricsError("array inventory digest drift")
    if row["save_reopen_validation_passed"] is not True:
        raise PhysicalGraphEdgeHandoffV2MetricsError("save/reopen validation did not pass")
    if reopened_arrays is not None:
        if set(reopened_arrays) != set(members):
            raise PhysicalGraphEdgeHandoffV2MetricsError("reopened member inventory drift")
        for part in validated_rows:
            expected = persisted_array_manifest_row(
                part["member"], reopened_arrays[part["member"]]
            )
            if part != expected:
                raise PhysicalGraphEdgeHandoffV2MetricsError(
                    f"reopened array evidence drift: {part['member']}"
                )
    return row


def validate_snapshot_previous_applied_command_binding(
    snapshot_metadata: Mapping[str, Any],
    persisted_array_evidence: Mapping[str, Any],
    *,
    reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the identified snapshot field to exact persisted raw bytes."""

    if not isinstance(snapshot_metadata, Mapping):
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "snapshot metadata is not a mapping"
        )
    evidence = validate_persisted_array_evidence(
        persisted_array_evidence, reopened_arrays=reopened_arrays
    )
    expected = C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
    matches = [
        item for item in evidence["arrays"] if item["member"] == expected["member"]
    ]
    if len(matches) != 1:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "previous-command persisted-array member coverage drift"
        )
    member = matches[0]
    if (
        member["dtype_str"] != expected["dtype_str"]
        or member["shape"] != expected["per_snapshot_shape"]
        or snapshot_metadata.get("previous_applied_command_sha256")
        != member["array_bytes_sha256"]
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "snapshot previous-command raw-byte binding drift"
        )
    return {
        "member": member["member"],
        "dtype_str": member["dtype_str"],
        "shape": copy.deepcopy(member["shape"]),
        "array_bytes_sha256": member["array_bytes_sha256"],
    }


def validate_state_snapshot_previous_applied_command_hashes(
    state_snapshot_index: Mapping[str, Any],
    raw_row_sha256s: Sequence[str],
) -> list[str]:
    """Cross-bind all 64 official snapshot rows to independently read raw bytes."""

    row = _require_v2_document(state_snapshot_index, "state_snapshot_index")
    records = _sequence(row.get("records"), "state snapshot records")
    raw = _sequence(raw_row_sha256s, "previous-command raw row hashes")
    if len(records) != C.STATE_COUNT or len(raw) != C.STATE_COUNT:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "previous-command official row cardinality drift"
        )
    validated: list[str] = []
    for index, (record, digest) in enumerate(zip(records, raw)):
        observed = _sha(digest, f"previous-command raw row sha256[{index}]")
        if not isinstance(record, Mapping) or record.get(
            "previous_applied_command_sha256"
        ) != observed:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "official previous-command raw-byte binding drift"
            )
        validated.append(observed)
    return validated


def build_regression_results(
    *,
    first_pool_production_writer_validator_passed: bool,
    other_metadata_bindings_unchanged: bool = True,
) -> list[dict[str, Any]]:
    """Regenerate the ten pre-simulator correction regressions."""

    import numpy as np

    fixture = np.asarray(C.REGRESSION_FIXTURE["values"], dtype=np.float64)
    expected_row = persisted_array_manifest_row("previous_applied_command", fixture)
    float32 = fixture.astype(np.float32)
    float32_row = persisted_array_manifest_row("previous_applied_command", float32)

    def rejected(candidate: Any) -> bool:
        evidence = {
            "schema": "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "shard_kind": "REGRESSION",
            "shard_id": "fixture",
            "payload_file": {"path": "fixture.npz", "bytes": 1, "sha256": "1" * 64},
            "array_count": 1,
            "array_inventory_sha256": _inventory_sha256([candidate]),
            "arrays": [candidate],
            "save_reopen_validation_passed": True,
        }
        try:
            validate_persisted_array_evidence(
                evidence, reopened_arrays={"previous_applied_command": fixture}
            )
        except PhysicalGraphEdgeHandoffV2MetricsError:
            return True
        return False

    dtype_bad = copy.deepcopy(expected_row)
    dtype_bad["dtype_str"] = float32.dtype.str
    shape_bad = copy.deepcopy(expected_row)
    shape_bad["shape"] = [3, 2]
    value_bad = copy.deepcopy(expected_row)
    mutated = fixture.copy()
    mutated[0, 0] += 1.0
    value_bad["array_bytes_sha256"] = persisted_array_bytes_sha256(mutated)

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]
    contiguous = np.ascontiguousarray(view)
    noncontiguous_pass = bool(
        not view.flags.c_contiguous
        and contiguous.flags.c_contiguous
        and contiguous.dtype == view.dtype
        and np.array_equal(contiguous, view)
    )
    buffer = io.BytesIO()
    np.savez(buffer, previous_applied_command=fixture)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        reloaded = np.ascontiguousarray(archive["previous_applied_command"])
    save_reload_pass = persisted_array_manifest_row(
        "previous_applied_command", reloaded
    ) == expected_row
    invariance_pass = bool(
        C.scientific_invariance_projection(C.build_contract())
        == C.V1_SCIENTIFIC_PROJECTION
        and C.scientific_constant_projection()
        == C.V1_SCIENTIFIC_CONSTANT_PROJECTION
        and C.build_candidate_specs() == C.V1.build_candidate_specs()
        and tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V1.SOURCE_DEPENDENCY_PATHS)
    )
    results = {
        "FLOAT64_EXACT_PERSISTED_BYTES_VALID": (
            expected_row["dtype_str"] == "<f8"
            and expected_row["shape"] == [2, 3]
            and expected_row["array_bytes_sha256"]
            == hashlib.sha256(fixture.tobytes(order="C")).hexdigest()
        ),
        "FLOAT32_CAST_DIGEST_REJECTED": (
            float32_row["array_bytes_sha256"]
            != expected_row["array_bytes_sha256"]
            and rejected(float32_row)
        ),
        "PERSISTED_DTYPE_MISMATCH_REJECTED": rejected(dtype_bad),
        "PERSISTED_SHAPE_MISMATCH_REJECTED": rejected(shape_bad),
        "PERSISTED_VALUE_MUTATION_REJECTED": rejected(value_bad),
        "NONCONTIGUOUS_VIEW_C_CONTIGUOUS_NO_DTYPE_CAST": noncontiguous_pass,
        "SAVE_RELOAD_DIGEST_IDENTICAL": save_reload_pass,
        "OTHER_METADATA_BINDINGS_UNCHANGED": bool(other_metadata_bindings_unchanged),
        "V1_SCIENTIFIC_CONSTANTS_AND_SOURCE_PATHS_UNCHANGED": invariance_pass,
        "FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS": bool(
            first_pool_production_writer_validator_passed
        ),
    }
    return [
        {
            "requirement_id": requirement,
            "passed": bool(results[requirement]),
            "evidence": (
                "deterministically regenerated by the pure V2 regression authority"
                if requirement != "FIRST_POOL_PRODUCTION_WRITER_VALIDATOR_PASS"
                else "actual production writer/validator synthetic pre-simulator fixture"
            ),
        }
        for requirement in C.REGRESSION_REQUIREMENT_IDS
    ]


def build_scientific_invariance_receipt(
    v2_contract: Mapping[str, Any], regression_results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    contract = C.validate_contract(v2_contract)
    results = [copy.deepcopy(dict(item)) for item in regression_results]
    projection = C.scientific_invariance_projection(contract)
    projection_sha = hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()
    v2_constants = C.scientific_constant_projection()
    v2_constants_sha = hashlib.sha256(
        C.canonical_json_bytes(v2_constants)[:-1]
    ).hexdigest()
    constants_equal = v2_constants == C.V1_SCIENTIFIC_CONSTANT_PROJECTION
    all_ten = bool(
        len(results) == len(C.REGRESSION_REQUIREMENT_IDS)
        and [item.get("requirement_id") for item in results]
        == list(C.REGRESSION_REQUIREMENT_IDS)
        and all(item.get("passed") is True for item in results)
    )
    return {
        "schema": "physical_graph_edge_handoff_qualification_v2.scientific_invariance_receipt.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "v1_contract_content_digest": C.V1_CONTRACT_CONTENT_DIGEST,
        "v2_contract_content_digest": contract["content_digest"],
        "v1_scientific_constants_sha256": C.V1_SCIENTIFIC_CONSTANTS_SHA256,
        "v2_scientific_constants_sha256": v2_constants_sha,
        "scientific_constants_equal": constants_equal,
        "official_documents_and_result_use_v2_identity_only": bool(
            C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "v2_execution_and_result_identity_claimed"
            ]
            and not C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "v1_scientific_completion_claimed"
            ]
        ),
        "v1_compatibility_identity_scopes_exact": bool(
            C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "frozen_inherited_scientific_identity_salt"
            ]
            == C.V1_EXPERIMENT_ID
            and C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "official_experiment_id"
            ]
            == C.EXPERIMENT_ID
        ),
        "inherited_implementation_alignment_disposition": (
            C.PORT_HEADING_ALIGNMENT_DISPOSITION
        ),
        "port_heading_alignment_authority_content_digest": (
            C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
        ),
        "candidate_port_metric_alignment_disposition": (
            C.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION
        ),
        "candidate_port_metric_alignment_authority_content_digest": (
            C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "content_digest"
            ]
        ),
        "v1_scientific_projection_sha256": C.V1_SCIENTIFIC_PROJECTION_SHA256,
        "v2_scientific_projection_sha256": projection_sha,
        "candidate_specs_equal": C.build_candidate_specs() == C.V1.build_candidate_specs(),
        "source_dependency_paths_equal": tuple(C.SOURCE_DEPENDENCY_PATHS)
        == tuple(C.V1.SOURCE_DEPENDENCY_PATHS),
        "scientific_projection_equal": projection == C.V1_SCIENTIFIC_PROJECTION,
        "regression_results": results,
        "all_ten_regressions_passed_before_simulator_creation": all_ten,
        "pass": bool(
            all_ten
            and constants_equal
            and C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "v2_execution_and_result_identity_claimed"
            ]
            and not C.V1_COMPATIBILITY_IDENTITY_AUTHORITY[
                "v1_scientific_completion_claimed"
            ]
            and C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "aligns_implementation_to_frozen_v1_authority"
            ]
            and not C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "changes_frozen_scientific_design"
            ]
            and C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "aligns_implementation_to_frozen_v1_authority"
            ]
            and not C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY[
                "changes_frozen_scientific_design"
            ]
            and projection == C.V1_SCIENTIFIC_PROJECTION
            and C.build_candidate_specs() == C.V1.build_candidate_specs()
            and tuple(C.SOURCE_DEPENDENCY_PATHS)
            == tuple(C.V1.SOURCE_DEPENDENCY_PATHS)
        ),
    }


def validate_scientific_invariance_receipt(value: Any) -> dict[str, Any]:
    row = _mapping(
        value, C.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS, "scientific_invariance_receipt"
    )
    if "content_digest" in row:
        raise PhysicalGraphEdgeHandoffV2MetricsError("receipt self-digest is forbidden")
    expected_results = build_regression_results(
        first_pool_production_writer_validator_passed=True
    )
    expected = build_scientific_invariance_receipt(C.build_contract(), expected_results)
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "scientific invariance receipt value drift"
        )
    return row


def v1_custody_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    boundary = value.get("scientific_boundary")
    pairs = value.get("qualification_pairs")
    if not isinstance(boundary, Mapping) or not isinstance(pairs, Sequence):
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody projection is incomplete")
    rejected = [dict(item) for item in pairs if not bool(item.get("qualified"))]
    def root_projection(name: str) -> dict[str, Any]:
        root = value.get(name)
        if not isinstance(root, Mapping):
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                f"V1 custody {name} projection is incomplete"
            )
        return {
            key: root.get(key)
            for key in (
                "file_count", "directory_count", "regular_file_apparent_bytes",
                "allocated_bytes",
            )
        }
    defect = value.get("defect_evidence")
    if not isinstance(defect, Mapping):
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 defect projection is incomplete")
    return {
        "source_freeze_commit": value.get("source_freeze_commit"),
        "disposition": value.get("disposition"),
        "qualification_pair_count": len(pairs),
        "qualified_count": sum(bool(item.get("qualified")) for item in pairs),
        "rejected_count": len(rejected),
        "rejected_pool_indices": [int(item["pool_index"]) for item in rejected],
        "pool_002_rejection_reason": (
            rejected[0].get("rejection_reason") if len(rejected) == 1 else None
        ),
        "official_root": root_projection("official_root"),
        "material_root": root_projection("material_root"),
        "defect_evidence": {
            key: copy.deepcopy(defect.get(key))
            for key in (
                "affected_pair_count", "field", "metadata_sha256",
                "metadata_source_dtype", "persisted_dtype",
                "persisted_v1_canonical_array_sha256", "shape", "values",
                "other_embedded_numeric_hash_mismatch_count",
            )
        },
        "scientific_boundary": copy.deepcopy(dict(boundary)),
    }


def validate_external_v1_custody_receipt(value: Any) -> dict[str, Any]:
    fields = {
        "schema", "experiment_id", "source_freeze_commit", "disposition",
        "official_root", "material_root", "qualification_pairs", "defect_evidence",
        "scientific_boundary", "repository", "immutability",
    }
    row = _mapping(value, fields, "external V1 custody receipt")
    if "content_digest" in row or row["schema"] != (
        "physical_graph_edge_handoff_qualification_v1.custody_receipt.v1"
    ) or row["experiment_id"] != C.V1_EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody receipt identity drift")
    canonical = C.canonical_json_bytes(row)
    if (
        len(canonical) != C.V1_CUSTODY_RECEIPT_BINDING["bytes"]
        or hashlib.sha256(canonical).hexdigest()
        != C.V1_CUSTODY_RECEIPT_BINDING["sha256"]
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "V1 custody receipt byte binding drift"
        )
    if v1_custody_projection(row) != C.V1_CUSTODY_EXPECTED_PROJECTION:
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody projection drift")
    for root_name in ("official_root", "material_root"):
        root = row[root_name]
        root_fields = {
            "path", "file_count", "directory_count", "regular_file_apparent_bytes",
            "regular_file_allocated_bytes", "directory_allocated_bytes", "allocated_bytes",
            "files", "directories",
        }
        part = _mapping(root, root_fields, f"V1 {root_name}")
        files = _sequence(part["files"], f"V1 {root_name}.files")
        directories = _sequence(part["directories"], f"V1 {root_name}.directories")
        if part["file_count"] != len(files) or part["directory_count"] != len(directories):
            raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody root count drift")
        file_fields = {"path", "bytes", "allocated_bytes", "sha256", "device", "inode", "nlink"}
        directory_fields = {"path", "allocated_bytes", "device", "inode", "nlink"}
        paths: list[str] = []
        for index, item in enumerate(files):
            file_row = _mapping(item, file_fields, f"V1 {root_name}.file[{index}]")
            _sha(file_row["sha256"], "V1 custody file sha256")
            if file_row["nlink"] != 1:
                raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody file is linked")
            paths.append(file_row["path"])
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody file order drift")
        directory_paths: list[str] = []
        for index, item in enumerate(directories):
            directory = _mapping(
                item, directory_fields, f"V1 {root_name}.directory[{index}]"
            )
            if directory["nlink"] < 1:
                raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody directory link drift")
            directory_paths.append(directory["path"])
        if directory_paths != sorted(directory_paths) or len(set(directory_paths)) != len(directory_paths):
            raise PhysicalGraphEdgeHandoffV2MetricsError("V1 custody directory order drift")
    pairs = _sequence(row["qualification_pairs"], "V1 qualification pairs")
    if [item.get("pool_index") for item in pairs] != list(range(8)):
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 qualification pair order drift")
    return row


def validate_v1_custody_and_nonreuse(
    value: Any, *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = _mapping(
        value,
        C.V1_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS,
        "v1_custody_and_nonreuse",
    )
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v2.v1_custody_and_nonreuse.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 nonreuse receipt identity drift")
    if row["v1_source_freeze_commit"] != C.V1_SOURCE_FREEZE_COMMIT:
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 source freeze drift")
    freeze = row["v2_source_freeze_commit"]
    if (
        not isinstance(freeze, str)
        or len(freeze) != 40
        or any(char not in "0123456789abcdef" for char in freeze)
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("V2 source freeze is invalid")
    if source_freeze_commit is not None and freeze != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            "V2 nonreuse source freeze cross-binding drift"
        )
    binding = _mapping(
        row["external_custody_receipt_binding"],
        {"path", "bytes", "sha256"},
        "external custody binding",
    )
    if binding != C.V1_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV2MetricsError("external custody binding drift")
    expected_projection_sha = hashlib.sha256(
        C.canonical_json_bytes(C.V1_CUSTODY_EXPECTED_PROJECTION)[:-1]
    ).hexdigest()
    if row["external_custody_projection_sha256"] != expected_projection_sha:
        raise PhysicalGraphEdgeHandoffV2MetricsError("custody projection digest drift")
    if (
        row["v1_official_root_unchanged"] is not True
        or row["v1_material_root_unchanged"] is not True
        or not isinstance(row["v1_payloads_copied_into_v2"], int)
        or isinstance(row["v1_payloads_copied_into_v2"], bool)
        or row["v1_payloads_copied_into_v2"] != 0
        or not isinstance(row["v1_hardlinks_into_v2"], int)
        or isinstance(row["v1_hardlinks_into_v2"], bool)
        or row["v1_hardlinks_into_v2"] != 0
        or not isinstance(row["v1_shared_inodes_with_v2"], int)
        or isinstance(row["v1_shared_inodes_with_v2"], bool)
        or row["v1_shared_inodes_with_v2"] != 0
        or row["v1_runtime_artifact_or_shard_reused"] is not False
        or row["allowed_read_scope"] != "CUSTODY_AND_FIRST_EIGHT_COMPARISON_ONLY"
        or row["pass"] is not True
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("V1 nonreuse gate failed")
    return row


def build_v1_custody_and_nonreuse(
    *, source_freeze_commit: str
) -> dict[str, Any]:
    """Build the exact pre-simulator V1 custody/nonreuse evidence document."""

    projection_sha = hashlib.sha256(
        C.canonical_json_bytes(C.V1_CUSTODY_EXPECTED_PROJECTION)[:-1]
    ).hexdigest()
    result = {
        "schema": "physical_graph_edge_handoff_qualification_v2.v1_custody_and_nonreuse.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": source_freeze_commit,
        "external_custody_receipt_binding": copy.deepcopy(
            C.V1_CUSTODY_RECEIPT_BINDING
        ),
        "external_custody_projection_sha256": projection_sha,
        "v1_official_root_unchanged": True,
        "v1_material_root_unchanged": True,
        "v1_payloads_copied_into_v2": 0,
        "v1_hardlinks_into_v2": 0,
        "v1_shared_inodes_with_v2": 0,
        "v1_runtime_artifact_or_shard_reused": False,
        "allowed_read_scope": "CUSTODY_AND_FIRST_EIGHT_COMPARISON_ONLY",
        "pass": True,
    }
    return validate_v1_custody_and_nonreuse(result)


def validate_first_eight_reproduction(value: Any) -> dict[str, Any]:
    row = _mapping(
        value, C.FIRST_EIGHT_REPRODUCTION_FIELDS, "first-eight reproduction"
    )
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v2.first_eight_reproduction.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction identity drift")
    if row["v1_custody_receipt_binding"] != C.V1_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction custody binding drift")
    if row["comparison_rule"] != C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["comparison_rule"]:
        raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction comparison rule drift")
    rows = _sequence(row["rows"], "first-eight rows")
    if row["row_count"] != 8 or len(rows) != 8:
        raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction row count drift")
    specs = C.build_prospective_pool_specs()[:8]
    validated: list[dict[str, Any]] = []
    equality_fields = sorted(
        C.FIRST_EIGHT_REPRODUCTION_ROW_FIELDS
        - {
            "pool_index", "candidate_spec_id", "state_id", "scene_id", "episode_id",
            "graph_id", "noncomparable_v1_known_bad_hash_fields", "pass",
        }
    )
    for index, (item, spec) in enumerate(zip(rows, specs)):
        part = _mapping(item, C.FIRST_EIGHT_REPRODUCTION_ROW_FIELDS, f"reproduction[{index}]")
        expected_identity = {
            "pool_index": index,
            "candidate_spec_id": spec["candidate_spec_id"],
            "state_id": spec["state_id"],
            "scene_id": spec["scene_id"],
            "episode_id": spec["episode_id"],
            "graph_id": spec["graph_id"],
        }
        if (
            not isinstance(part["pool_index"], int)
            or isinstance(part["pool_index"], bool)
            or any(part[key] != expected for key, expected in expected_identity.items())
        ):
            raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction identity/order drift")
        if part["noncomparable_v1_known_bad_hash_fields"] != [
            "snapshot.previous_applied_command_sha256"
        ]:
            raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction exception drift")
        if any(type(part[field]) is not bool for field in equality_fields):
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "reproduction equality evidence is not boolean"
            )
        expected_pass = all(part[field] is True for field in equality_fields)
        if type(part["pass"]) is not bool or part["pass"] is not expected_pass:
            raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction row pass drift")
        validated.append(part)
    all_pass = all(part["pass"] for part in validated)
    expected_status = "PASS" if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION
    if (
        not isinstance(row["row_count"], int)
        or isinstance(row["row_count"], bool)
        or not isinstance(row["compared_before_pool_index"], int)
        or isinstance(row["compared_before_pool_index"], bool)
        or not isinstance(
            row["candidate_ranker_development_heldout_outcomes_opened"], int
        )
        or isinstance(
            row["candidate_ranker_development_heldout_outcomes_opened"], bool
        )
        or row["status"] != expected_status
        or row["pass"] is not all_pass
        or row["technical_disposition"]
        != (None if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION)
        or row["full_collection_authorized"] is not all_pass
        or row["compared_before_pool_index"] != 8
        or row["candidate_ranker_development_heldout_outcomes_opened"] != 0
    ):
        raise PhysicalGraphEdgeHandoffV2MetricsError("reproduction disposition drift")
    return row


def build_first_eight_reproduction(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact success or technical-terminal first-eight document."""

    copied = [copy.deepcopy(dict(item)) for item in rows]
    all_pass = bool(len(copied) == 8 and all(item.get("pass") is True for item in copied))
    result = {
        "schema": "physical_graph_edge_handoff_qualification_v2.first_eight_reproduction.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "status": "PASS" if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "v1_custody_receipt_binding": copy.deepcopy(C.V1_CUSTODY_RECEIPT_BINDING),
        "comparison_rule": C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["comparison_rule"],
        "row_count": len(copied),
        "rows": copied,
        "pass": all_pass,
        "technical_disposition": None if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "full_collection_authorized": all_pass,
        "compared_before_pool_index": 8,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
    }
    return validate_first_eight_reproduction(result)


def authorizes_full_v2_collection(value: Any) -> bool:
    return bool(validate_first_eight_reproduction(value)["full_collection_authorized"])


# The geometry/physics/runtime functions are exact V1 science and need no
# identity adapter.
classify_physical_handoff_aggregates = V1M.classify_physical_handoff_aggregates
point_in_polygon_inclusive = V1M.point_in_polygon_inclusive
transverse_port_crossing = V1M.transverse_port_crossing
first_registered_port_crossing = V1M.first_registered_port_crossing
runtime_environment_sha256 = V1M.runtime_environment_sha256
validate_physical_runtime_environment = V1M.validate_physical_runtime_environment
validate_visual_runtime_environment = V1M.validate_visual_runtime_environment


def physical_trace_reduction_authority() -> dict[str, Any]:
    return _map_identity(V1M.physical_trace_reduction_authority(), to_v1=False)


def panel_manifest_authority() -> dict[str, Any]:
    return _map_identity(V1M.panel_manifest_authority(), to_v1=False)


def external_artifact_bindings(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    if contract.get("schema") == (
        "physical_graph_edge_handoff_qualification_v2.runtime_contract.v1"
    ):
        row = C.validate_runtime_contract(contract)
        return copy.deepcopy(row["external_artifact_bindings"])
    C.validate_contract(contract)
    return [copy.deepcopy(item) for item in C.V1.EXTERNAL_ARTIFACT_BINDINGS]


def predecessor_context_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") == (
        "physical_graph_edge_handoff_qualification_v2.runtime_contract.v1"
    ):
        row = C.validate_runtime_contract(contract)["scientific_contract"]
    else:
        row = C.validate_contract(contract)
    if row["v2_context_binding"] != C.V1.V2_CONTEXT_BINDING:
        raise PhysicalGraphEdgeHandoffV2MetricsError("predecessor context drift")
    return copy.deepcopy(row["v2_context_binding"])


def validate_npz_inspections(value: Any) -> dict[str, dict[str, Any]]:
    # Dtype/shape and row/slice binding are unchanged; only the opaque SHA
    # domain is now raw persisted bytes, which V1's structural validator does
    # not reinterpret.
    try:
        return V1M.validate_npz_inspections(value)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(str(exc)) from exc


def validate_panel_manifest(value: Any) -> dict[str, Any]:
    return _call_v1_document(V1M.validate_panel_manifest, value, label="panel_manifest")


def validate_split_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_split_manifest, value, panel_manifest, label="split_manifest"
    )


def validate_graph_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_graph_manifest, value, panel_manifest, label="graph_manifest"
    )


def validate_state_snapshot_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_state_snapshot_index,
        value,
        panel_manifest,
        label="state_snapshot_index",
    )


def validate_teacher_trace_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_teacher_trace_index,
        value,
        panel_manifest,
        label="teacher_trace_index",
    )


def validate_edge_port_index(
    value: Any,
    panel_manifest: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_edge_port_index,
        value,
        panel_manifest,
        graph_manifest,
        teacher_trace_index,
        label="edge_port_index",
    )


def validate_waypoint_contracts(
    value: Any,
    panel_manifest: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    edge_port_index: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_waypoint_contracts,
        value,
        panel_manifest,
        graph_manifest,
        edge_port_index,
        label="waypoint_contracts",
    )


def validate_pixel_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_pixel_index, value, panel_manifest, label="pixel_index"
    )


def validate_latent_index(value: Any, pixel_index: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_latent_index, value, pixel_index, label="latent_index"
    )


def validate_candidate_fanout_rows(
    value: Any,
    panel_manifest: Mapping[str, Any],
    state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        projected = _map_identity(value, to_v1=True)
        result = V1M.validate_candidate_fanout_rows(
            projected,
            _map_identity(panel_manifest, to_v1=True),
            _map_identity(state_snapshot_index, to_v1=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(str(exc)) from exc
    if len(result) != len(value):
        raise PhysicalGraphEdgeHandoffV2MetricsError("candidate fanout cardinality drift")
    return copy.deepcopy(list(value))


def validate_development_target_selection(
    value: Any,
    panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    waypoint_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v1_document(
        V1M.validate_development_target_selection,
        value,
        panel_manifest,
        candidate_fanout,
        waypoint_contracts,
        label="development_target_selection",
    )


def validate_heldout_ranker_score_rows(
    value: Any,
    panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    development_target_selection: Mapping[str, Any],
    waypoint_contracts: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        result = V1M.validate_heldout_ranker_score_rows(
            _map_identity(value, to_v1=True),
            _map_identity(panel_manifest, to_v1=True),
            _map_identity(candidate_fanout, to_v1=True),
            _map_identity(development_target_selection, to_v1=True),
            _map_identity(waypoint_contracts, to_v1=True),
            _map_identity(teacher_trace_index, to_v1=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(str(exc)) from exc
    if len(result) != len(value):
        raise PhysicalGraphEdgeHandoffV2MetricsError("heldout row cardinality drift")
    return copy.deepcopy(list(value))


def validate_repeated_execution_rows(
    value: Any,
    panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    heldout_ranker_scores: Sequence[Mapping[str, Any]],
    state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        result = V1M.validate_repeated_execution_rows(
            _map_identity(value, to_v1=True),
            _map_identity(panel_manifest, to_v1=True),
            _map_identity(candidate_fanout, to_v1=True),
            _map_identity(heldout_ranker_scores, to_v1=True),
            _map_identity(state_snapshot_index, to_v1=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(str(exc)) from exc
    if len(result) != len(value):
        raise PhysicalGraphEdgeHandoffV2MetricsError("repeat row cardinality drift")
    return copy.deepcopy(list(value))


def recompute_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute with the exact V1 formulas, returning V2 identity only."""

    v1_evidence = _map_identity(evidence, to_v1=True)
    # V2's corrected central metadata field binds the exact persisted raw bytes.
    # V1's otherwise unchanged reducer expects its legacy header+NUL digest in
    # this one validation slot.  Bridge only the private V1 validation copy from
    # the already-inspected NPZ canonical rows; the V2 evidence and returned
    # metrics retain the raw-byte field.  Independent custody validates the V2
    # raw row digests before this pure formula call.
    try:
        inspections = V1M.validate_npz_inspections(v1_evidence["npz_inspections"])
        canonical_rows = inspections["state_snapshots.npz"]["members"][
            "previous_applied_command"
        ]["row_or_slice_sha256s"]
        snapshot_rows = v1_evidence["state_snapshot_index"]["records"]
        if len(canonical_rows) != len(snapshot_rows) or not snapshot_rows:
            raise PhysicalGraphEdgeHandoffV2MetricsError(
                "previous-command validation bridge cardinality drift"
            )
        for index, snapshot in enumerate(snapshot_rows):
            _sha(
                snapshot["previous_applied_command_sha256"],
                f"V2 previous_applied_command raw sha256[{index}]",
            )
            snapshot["previous_applied_command_sha256"] = canonical_rows[index]
        # The substitutions above deliberately occur only in this private V1
        # compatibility document.  They change a content-digested nested
        # document, so rebuild that V1 digest before invoking the inherited
        # validators.  The caller's V2 evidence remains untouched and retains
        # the exact raw-C-byte hashes.
        legacy_state_snapshot_index = dict(v1_evidence["state_snapshot_index"])
        legacy_state_snapshot_index.pop("content_digest", None)
        v1_evidence["state_snapshot_index"] = C.V1.attach_content_digest(
            legacy_state_snapshot_index
        )
    except PhysicalGraphEdgeHandoffV2MetricsError:
        raise
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            f"previous-command validation bridge failed: {exc}"
        ) from exc
    try:
        v1_result = V1M.recompute_metrics(v1_evidence)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV2MetricsError(
            f"frozen V1 metric recomputation failed: {exc}"
        ) from exc
    result = _map_identity(v1_result, to_v1=False)
    result.pop("content_digest", None)
    result["schema"] = "physical_graph_edge_handoff_qualification_v2.metrics.v1"
    result["experiment_id"] = C.EXPERIMENT_ID
    return C.attach_content_digest(result)


def reducer_authority() -> dict[str, Any]:
    base = _map_identity(V1M.reducer_authority(), to_v1=False)
    base.pop("content_digest", None)
    base["schema"] = "physical_graph_edge_handoff_qualification_v2.reducer_authority.v1"
    base["experiment_id"] = C.EXPERIMENT_ID
    base["runtime_paths"] = copy.deepcopy(C.RUNTIME_OUTPUT_PATHS)
    base["successful_output_leaf_count"] = C.OUTPUT_LEAF_COUNT
    base["successful_output_leaves"] = list(C.SUCCESS_OUTPUT_LEAVES)
    base["reproduction_mismatch_output_leaves"] = list(
        C.REPRODUCTION_MISMATCH_LEAVES
    )
    base["persisted_array_hash_authority"] = copy.deepcopy(
        C.PERSISTED_ARRAY_HASH_AUTHORITY
    )
    base["persisted_array_evidence"] = {
        "root_fields": sorted(C.PERSISTED_ARRAY_EVIDENCE_FIELDS),
        "array_fields": sorted(C.PERSISTED_ARRAY_ROW_FIELDS),
        "payload_file_fields": sorted(C.PERSISTED_ARRAY_PAYLOAD_FILE_FIELDS),
    }
    base["scientific_invariance_authority"] = copy.deepcopy(
        C.SCIENTIFIC_INVARIANCE_AUTHORITY
    )
    base["v1_compatibility_identity_authority"] = copy.deepcopy(
        C.V1_COMPATIBILITY_IDENTITY_AUTHORITY
    )
    base["port_heading_implementation_alignment_authority"] = copy.deepcopy(
        C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    )
    base["candidate_port_metric_implementation_alignment_authority"] = copy.deepcopy(
        C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    )
    base["regression_gate_authority"] = copy.deepcopy(C.REGRESSION_GATE_AUTHORITY)
    base["v1_custody_and_nonreuse_authority"] = copy.deepcopy(
        C.V1_CUSTODY_AND_NONREUSE_AUTHORITY
    )
    base["first_eight_reproduction_authority"] = copy.deepcopy(
        C.FIRST_EIGHT_REPRODUCTION_AUTHORITY
    )
    base["new_documents"] = {
        "v1_custody_and_nonreuse": {
            "path": "v1_custody_and_nonreuse.json",
            "fields": sorted(C.V1_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS),
            "content_digest_forbidden": True,
        },
        "scientific_invariance_receipt": {
            "path": "scientific_invariance_receipt.json",
            "fields": sorted(C.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS),
            "content_digest_forbidden": True,
        },
        "v1_v2_first_eight_reproduction": {
            "path": "v1_v2_first_eight_reproduction.json",
            "fields": sorted(C.FIRST_EIGHT_REPRODUCTION_FIELDS),
            "row_fields": sorted(C.FIRST_EIGHT_REPRODUCTION_ROW_FIELDS),
            "row_count": 8,
            "content_digest_forbidden": True,
        },
    }
    return C.attach_content_digest(base)


def scientific_invariance_authority() -> dict[str, Any]:
    return copy.deepcopy(C.SCIENTIFIC_INVARIANCE_AUTHORITY)


def v1_compatibility_identity_authority() -> dict[str, Any]:
    return copy.deepcopy(C.V1_COMPATIBILITY_IDENTITY_AUTHORITY)


def port_heading_implementation_alignment_authority() -> dict[str, Any]:
    return copy.deepcopy(C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY)


def candidate_port_metric_implementation_alignment_authority() -> dict[str, Any]:
    return copy.deepcopy(C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY)


def regression_gate_authority() -> dict[str, Any]:
    return copy.deepcopy(C.REGRESSION_GATE_AUTHORITY)


def v1_custody_and_nonreuse_authority() -> dict[str, Any]:
    return copy.deepcopy(C.V1_CUSTODY_AND_NONREUSE_AUTHORITY)


def first_eight_reproduction_authority() -> dict[str, Any]:
    return copy.deepcopy(C.FIRST_EIGHT_REPRODUCTION_AUTHORITY)


__all__ = [
    "PhysicalGraphEdgeHandoffV2MetricsError",
    "authorizes_full_v2_collection",
    "build_first_eight_reproduction",
    "build_persisted_array_evidence",
    "build_regression_results",
    "build_scientific_invariance_receipt",
    "build_v1_custody_and_nonreuse",
    "candidate_port_metric_implementation_alignment_authority",
    "classify_physical_handoff_aggregates",
    "derive_candidate_port_metrics",
    "external_artifact_bindings",
    "first_registered_port_crossing",
    "first_eight_reproduction_authority",
    "panel_manifest_authority",
    "persisted_array_bytes_sha256",
    "persisted_array_manifest_row",
    "physical_trace_reduction_authority",
    "point_in_polygon_inclusive",
    "port_heading_implementation_alignment_authority",
    "predecessor_context_binding",
    "project_v1_evidence_to_v2",
    "project_v2_evidence_to_v1",
    "recompute_metrics",
    "regression_gate_authority",
    "reducer_authority",
    "runtime_environment_sha256",
    "scientific_invariance_authority",
    "transverse_port_crossing",
    "v1_custody_projection",
    "v1_custody_and_nonreuse_authority",
    "v1_compatibility_identity_authority",
    "validate_candidate_fanout_rows",
    "validate_development_target_selection",
    "validate_edge_port_index",
    "validate_external_v1_custody_receipt",
    "validate_first_eight_reproduction",
    "validate_graph_manifest",
    "validate_heldout_ranker_score_rows",
    "validate_latent_index",
    "validate_npz_inspections",
    "validate_npz_archive_comment",
    "validate_panel_manifest",
    "validate_persisted_array_evidence",
    "validate_physical_runtime_environment",
    "validate_pixel_index",
    "validate_repeated_execution_rows",
    "validate_scientific_invariance_receipt",
    "validate_snapshot_previous_applied_command_binding",
    "validate_split_manifest",
    "validate_state_snapshot_previous_applied_command_hashes",
    "validate_state_snapshot_index",
    "validate_teacher_trace_index",
    "validate_v1_custody_and_nonreuse",
    "validate_visual_runtime_environment",
    "validate_waypoint_contracts",
]
