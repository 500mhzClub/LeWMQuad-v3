"""Pure V3 snapshot qualification and inherited physical-science reduction.

The module imports no simulator, model, encoder, or ranker.  V3 evidence is
validated in its own identity, then inherited physical evidence is projected
into the exact V2/V1 validators.  Snapshot transport bytes are descriptive;
semantic bytes and the frozen two-restore behavioural probe are the state
equivalence authorities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from typing import Any, Callable

from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as V2M
from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as C
from lewm.safety import physical_graph_edge_handoff_snapshot_semantics_v1 as S


class PhysicalGraphEdgeHandoffV3MetricsError(ValueError):
    """Raised when V3 evidence is incomplete, noncanonical, or inconsistent."""


_V2_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v2"
_V3_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v3"


def _mapping(
    value: Any, fields: set[str] | frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} field drift")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} is not a sequence")
    return [copy.deepcopy(item) for item in value]


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} is not SHA-256")
    return value


def _identity_string(value: str, *, to_v2: bool) -> str:
    if to_v2:
        if value == C.EXPERIMENT_ID:
            return C.V2_EXPERIMENT_ID
        return value.replace(_V3_SCHEMA_TOKEN, _V2_SCHEMA_TOKEN)
    if value == C.V2_EXPERIMENT_ID:
        return C.EXPERIMENT_ID
    return value.replace(_V2_SCHEMA_TOKEN, _V3_SCHEMA_TOKEN)


def _map_identity(value: Any, *, to_v2: bool) -> Any:
    """Map only V2/V3 official experiment/schema identity."""

    if isinstance(value, Mapping):
        has_digest = "content_digest" in value
        if has_digest and to_v2:
            C.validate_content_digest(value)
        mapped = {
            str(key): _map_identity(item, to_v2=to_v2)
            for key, item in value.items()
            if key != "content_digest"
        }
        if has_digest:
            attach = C.V2.attach_content_digest if to_v2 else C.attach_content_digest
            mapped = attach(mapped)
        return mapped
    if isinstance(value, (tuple, list)):
        return [_map_identity(item, to_v2=to_v2) for item in value]
    if isinstance(value, str):
        return _identity_string(value, to_v2=to_v2)
    return copy.deepcopy(value)


def _require_v3_document(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} is not a mapping")
    row = copy.deepcopy(dict(value))
    if row.get("experiment_id") != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"{label} experiment identity drift"
        )
    schema = row.get("schema")
    if not isinstance(schema, str) or _V3_SCHEMA_TOKEN not in schema:
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} schema identity drift")
    if "content_digest" in row:
        C.validate_content_digest(row)
    return row


def project_v2_evidence_to_v3(value: Any) -> Any:
    return _map_identity(value, to_v2=False)


def project_v3_evidence_to_v2(value: Any) -> Any:
    return _map_identity(value, to_v2=True)


def _call_v2_document(
    validator: Callable[..., Any], value: Any, *dependencies: Any, label: str
) -> Any:
    row = _require_v3_document(value, label)
    try:
        validator(
            _map_identity(row, to_v2=True),
            *[_map_identity(item, to_v2=True) for item in dependencies],
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"{label} violates frozen V2 scientific authority: {exc}"
        ) from exc
    return row


# ---------------------------------------------------------------------------
# Exact persisted-byte correction retained from V2
# ---------------------------------------------------------------------------

persisted_array_bytes_sha256 = V2M.persisted_array_bytes_sha256
persisted_array_manifest_row = V2M.persisted_array_manifest_row
derive_candidate_port_metrics = V2M.derive_candidate_port_metrics


def _inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(C.canonical_json_bytes(list(rows))[:-1]).hexdigest()


def validate_npz_archive_comment(value: bytes | str) -> str:
    if isinstance(value, bytes):
        try:
            observed = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "NPZ archive comment is not UTF-8"
            ) from exc
    elif isinstance(value, str):
        observed = value
    else:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "NPZ archive comment has invalid type"
        )
    if observed != C.NPZ_ARCHIVE_COMMENT:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "NPZ archive provenance comment drift"
        )
    return observed


def build_persisted_array_evidence(
    *,
    shard_kind: str,
    shard_id: str,
    payload_file: Mapping[str, Any],
    arrays: Mapping[str, Any],
    reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    v2 = V2M.build_persisted_array_evidence(
        shard_kind=shard_kind,
        shard_id=shard_id,
        payload_file=payload_file,
        arrays=arrays,
        reopened_arrays=reopened_arrays,
    )
    result = _map_identity(v2, to_v2=False)
    result["schema"] = (
        "physical_graph_edge_handoff_qualification_v3.persisted_array_evidence.v1"
    )
    result["experiment_id"] = C.EXPERIMENT_ID
    return validate_persisted_array_evidence(result, reopened_arrays=reopened_arrays)


def validate_persisted_array_evidence(
    value: Any, *, reopened_arrays: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    row = _mapping(
        value, C.PERSISTED_ARRAY_EVIDENCE_FIELDS, "persisted_array_evidence"
    )
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v3.persisted_array_evidence.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "persisted array evidence identity drift"
        )
    projected = _map_identity(row, to_v2=True)
    projected["schema"] = (
        "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1"
    )
    projected["experiment_id"] = C.V2_EXPERIMENT_ID
    try:
        V2M.validate_persisted_array_evidence(
            projected, reopened_arrays=reopened_arrays
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    return row


def validate_snapshot_previous_applied_command_binding(
    snapshot_metadata: Mapping[str, Any],
    persisted_array_evidence: Mapping[str, Any],
    *,
    reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    projected = _map_identity(persisted_array_evidence, to_v2=True)
    projected["schema"] = (
        "physical_graph_edge_handoff_qualification_v2.persisted_array_evidence.v1"
    )
    projected["experiment_id"] = C.V2_EXPERIMENT_ID
    try:
        return V2M.validate_snapshot_previous_applied_command_binding(
            snapshot_metadata, projected, reopened_arrays=reopened_arrays
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc


def validate_state_snapshot_previous_applied_command_hashes(
    state_snapshot_index: Mapping[str, Any], raw_row_sha256s: Sequence[str]
) -> list[str]:
    try:
        return V2M.validate_state_snapshot_previous_applied_command_hashes(
            _map_identity(state_snapshot_index, to_v2=True), raw_row_sha256s
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Canonical semantic snapshot evidence
# ---------------------------------------------------------------------------

canonical_semantic_snapshot = S.canonical_semantic_snapshot
semantic_snapshot_sha256 = S.semantic_snapshot_sha256


def semantic_snapshot_serializer_authority() -> dict[str, Any]:
    if (
        S.SEMANTIC_SERIALIZER_SCHEMA
        != C.SEMANTIC_SERIALIZER_AUTHORITY["serializer_schema"]
        or S.SEMANTIC_BINARY_MAGIC.hex()
        != C.SEMANTIC_SERIALIZER_AUTHORITY["binary_magic_hex"]
        or S.REFERENCE_POLICY
        != C.SEMANTIC_SERIALIZER_AUTHORITY["reference_policy"]
        or {
            key: list(fields) for key, fields in S.STRUCTURED_FIELD_AUTHORITY.items()
        }
        != C.STRUCTURED_TYPE_FIELD_AUTHORITY
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "serializer module/contract authority drift"
        )
    return copy.deepcopy(C.SEMANTIC_SERIALIZER_AUTHORITY)


def historical_snapshot_deserializer_authority() -> dict[str, Any]:
    return copy.deepcopy(C.HISTORICAL_DESERIALIZER_AUTHORITY)


def snapshot_semantic_evidence(value: Any) -> dict[str, Any]:
    """Build and validate the complete semantic/type/alias evidence."""

    try:
        evidence = S.semantic_snapshot_evidence(value)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    return validate_snapshot_semantic_evidence(evidence)


def validate_snapshot_semantic_evidence(value: Any) -> dict[str, Any]:
    # Exact fields are intentionally projected from the serializer API rather
    # than accepted heuristically.  This set is cross-checked by focused tests.
    expected_fields = C.SNAPSHOT_SEMANTIC_EVIDENCE_FIELDS
    row = _mapping(value, expected_fields, "snapshot semantic evidence")
    if row["serializer_schema"] != C.SEMANTIC_SERIALIZER_AUTHORITY["serializer_schema"]:
        raise PhysicalGraphEdgeHandoffV3MetricsError("serializer schema drift")
    _sha(row["snapshot_semantic_digest_v1"], "snapshot semantic digest")
    for name in (
        "canonical_semantic_byte_count", "referenceable_object_count",
        "reference_alias_edge_count", "reference_cycle_edge_count",
    ):
        if (
            not isinstance(row[name], int)
            or isinstance(row[name], bool)
            or row[name] < 0
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"semantic evidence {name} drift"
            )
    if row["canonical_semantic_byte_count"] <= len(S.SEMANTIC_BINARY_MAGIC):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "semantic byte count is implausible"
        )
    if row["reference_policy"] != S.REFERENCE_POLICY:
        raise PhysicalGraphEdgeHandoffV3MetricsError("reference policy drift")
    reference_manifest = _sequence(row["reference_manifest"], "reference manifest")
    if len(reference_manifest) != row["referenceable_object_count"]:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "reference manifest cardinality drift"
        )
    for index, item in enumerate(reference_manifest):
        part = _mapping(item, {"object_id", "path", "type"}, f"reference[{index}]")
        if part["object_id"] != index or not isinstance(part["path"], str) or not isinstance(part["type"], str):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "reference manifest order drift"
            )
    edges = _sequence(row["reference_edge_manifest"], "reference edges")
    if len(edges) != row["reference_alias_edge_count"]:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "reference edge cardinality drift"
        )
    cycle_count = 0
    for index, item in enumerate(edges):
        part = _mapping(item, {"path", "reference_id", "cycle"}, f"edge[{index}]")
        if (
            not isinstance(part["path"], str)
            or not isinstance(part["reference_id"], int)
            or isinstance(part["reference_id"], bool)
            or not 0 <= part["reference_id"] < row["referenceable_object_count"]
            or type(part["cycle"]) is not bool
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError("reference edge drift")
        cycle_count += int(part["cycle"])
    if cycle_count != row["reference_cycle_edge_count"]:
        raise PhysicalGraphEdgeHandoffV3MetricsError("reference cycle count drift")
    reference_by_id = {item["object_id"]: item for item in reference_manifest}
    storage_rows = _sequence(row["storage_manifest"], "storage manifest")
    covered_storage_objects: list[int] = []
    for index, item in enumerate(storage_rows):
        part = _mapping(
            item,
            {"storage_id", "kind", "owner_object_id", "member_object_ids", "member_paths"},
            f"storage[{index}]",
        )
        members = _sequence(part["member_object_ids"], f"storage[{index}] members")
        paths = _sequence(part["member_paths"], f"storage[{index}] paths")
        if (
            part["storage_id"] != index
            or part["kind"] not in {"numpy", "torch"}
            or not members or members != sorted(set(members))
            or len(paths) != len(members)
            or part["owner_object_id"] != members[0]
            or any(member not in reference_by_id for member in members)
            or paths != [reference_by_id[member]["path"] for member in members]
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError("storage manifest drift")
        covered_storage_objects.extend(members)
    array_object_ids = sorted(
        item["object_id"] for item in reference_manifest
        if item["type"] == "numpy.ndarray" or item["type"].startswith("torch.")
    )
    if sorted(covered_storage_objects) != array_object_ids:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "storage/reference inventory cross-link drift"
        )
    structured_rows = _sequence(
        row["structured_type_inventory"], "structured type inventory"
    )
    structured = []
    for index, item in enumerate(structured_rows):
        part = _mapping(item, {"type", "declared_fields"}, f"structured[{index}]")
        if (
            part["type"] not in C.STRUCTURED_TYPE_FIELD_AUTHORITY
            or part["declared_fields"]
            != C.STRUCTURED_TYPE_FIELD_AUTHORITY[part["type"]]
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "structured type inventory drift"
            )
        structured.append(part["type"])
    if structured != sorted(set(structured)):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "structured type inventory order drift"
        )
    tensor_rows = _sequence(row["tensor_device_manifest"], "tensor device manifest")
    tensor_objects = {
        item["object_id"] for item in reference_manifest
        if item["type"].startswith("torch.")
    }
    observed_tensor_objects: set[int] = set()
    for index, item in enumerate(tensor_rows):
        part = _mapping(
            item,
            {"path", "object_id", "semantic_device_class", "source_device_type", "source_device_index"},
            f"tensor device[{index}]",
        )
        source_index = part["source_device_index"]
        if (
            part["object_id"] not in tensor_objects
            or part["object_id"] in observed_tensor_objects
            or part["path"] != reference_by_id[part["object_id"]]["path"]
            or part["semantic_device_class"] not in {"cpu", "accelerator"}
            or not isinstance(part["source_device_type"], str)
            or not part["source_device_type"]
            or source_index is not None and (
                not isinstance(source_index, int) or isinstance(source_index, bool)
                or source_index < 0
            )
            or (part["source_device_type"] == "cpu")
            != (part["semantic_device_class"] == "cpu")
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "tensor device manifest drift"
            )
        observed_tensor_objects.add(part["object_id"])
    if observed_tensor_objects != tensor_objects:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "tensor device inventory cardinality drift"
        )
    type_rows = _sequence(row["type_inventory"], "type inventory")
    type_names = []
    for index, item in enumerate(type_rows):
        part = _mapping(item, {"type", "count"}, f"type inventory[{index}]")
        if (
            not isinstance(part["type"], str) or not part["type"]
            or not isinstance(part["count"], int) or isinstance(part["count"], bool)
            or part["count"] <= 0
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError("type inventory drift")
        type_names.append(part["type"])
    if type_names != sorted(set(type_names)):
        raise PhysicalGraphEdgeHandoffV3MetricsError("type inventory order drift")
    allowed_sentinels = C.NONFINITE_SENTINEL_AUTHORITY["allowed_numpy_arrays"]
    sentinel_rows = _sequence(
        row["nonfinite_sentinel_inventory"], "nonfinite sentinel inventory"
    )
    if (
        sentinel_rows != sorted(sentinel_rows, key=lambda item: item.get("path", "") if isinstance(item, Mapping) else "")
        or any(item not in allowed_sentinels for item in sentinel_rows)
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "nonfinite sentinel inventory drift"
        )
    return row


def validate_snapshot_semantic_bytes(
    canonical_bytes: bytes, evidence: Mapping[str, Any]
) -> bytes:
    if not isinstance(canonical_bytes, bytes) or not canonical_bytes.startswith(
        S.SEMANTIC_BINARY_MAGIC
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "canonical semantic bytes domain drift"
        )
    row = validate_snapshot_semantic_evidence(evidence)
    if (
        len(canonical_bytes) != row["canonical_semantic_byte_count"]
        or hashlib.sha256(canonical_bytes).hexdigest()
        != row["snapshot_semantic_digest_v1"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "canonical semantic bytes/evidence mismatch"
        )
    return canonical_bytes


# ---------------------------------------------------------------------------
# Qualification-shard semantic/behavioural augmentation
# ---------------------------------------------------------------------------

def _qualification_versions(pool_index: int) -> tuple[str, ...]:
    if (
        not isinstance(pool_index, int)
        or isinstance(pool_index, bool)
        or not 0 <= pool_index < 256
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification pool index drift"
        )
    return ("V1", "V2", "V3") if pool_index < 8 else ("V3",)


def _probe_member_name(version: str, trial_index: int, member: str) -> str:
    return f"probe__{version}__{trial_index}__{member}"


def _probe_addition_member_names(pool_index: int) -> set[str]:
    names = {"snapshot_semantic_bytes"}
    for version in _qualification_versions(pool_index):
        for trial_index in (0, 1):
            names.update(
                _probe_member_name(version, trial_index, member)
                for member in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY
            )
            names.add(
                _probe_member_name(
                    version, trial_index,
                    "final_snapshot_semantic_digest_bytes",
                )
            )
    return names


def build_qualification_shard_augmentation(
    *,
    pool_index: int,
    snapshot_payload_bytes: Any,
    canonical_semantic_bytes: bytes,
    snapshot_semantic_evidence: Mapping[str, Any],
    version_snapshot_identities: Mapping[str, Mapping[str, Any]],
    behavioural_probe_traces: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build exact metadata and NPZ additions for one qualification shard."""

    import numpy as np

    versions = _qualification_versions(pool_index)
    if (
        not isinstance(version_snapshot_identities, Mapping)
        or set(version_snapshot_identities) != set(versions)
        or not isinstance(behavioural_probe_traces, Mapping)
        or set(behavioural_probe_traces) != set(versions)
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification version inventory drift"
        )
    semantic = validate_snapshot_semantic_evidence(snapshot_semantic_evidence)
    validate_snapshot_semantic_bytes(canonical_semantic_bytes, semantic)
    payload = np.asarray(snapshot_payload_bytes)
    if payload.dtype.str != "|u1" or payload.ndim != 1 or payload.size == 0:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot transport payload dtype/shape drift"
        )
    artifact_sha = hashlib.sha256(
        np.ascontiguousarray(payload).tobytes(order="C")
    ).hexdigest()
    additions: dict[str, Any] = {
        "snapshot_semantic_bytes": np.frombuffer(
            canonical_semantic_bytes, dtype=np.uint8
        ).copy()
    }
    probes: dict[str, Any] = {}
    for version in versions:
        traces = _sequence(
            behavioural_probe_traces[version],
            f"qualification {version} behavioural traces",
        )
        if len(traces) != 2:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification behavioural trial cardinality drift"
            )
        validated_traces = [
            _trace_arrays(trace, f"qualification {version} trial {trial}")
            for trial, trace in enumerate(traces)
        ]
        trial_digests = [
            snapshot_behavioural_digest(trace) for trace in validated_traces
        ]
        identity = _version_identity(
            version_snapshot_identities[version],
            f"qualification {version} snapshot identity",
        )
        if identity["snapshot_behavioural_digest_v1"] != trial_digests[0]:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification designated behavioural identity drift"
            )
        if version == "V3" and (
            identity["artifact_file_sha256"] != artifact_sha
            or identity["snapshot_semantic_digest_v1"]
            != semantic["snapshot_semantic_digest_v1"]
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification V3 artifact/semantic identity drift"
            )
        trace_manifests = []
        for trial_index, trace in enumerate(validated_traces):
            manifest = _trace_manifest(trace)
            trace_manifests.append(manifest["arrays"])
            for member in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
                additions[_probe_member_name(version, trial_index, member)] = (
                    trace[member].copy()
                )
            additions[
                _probe_member_name(
                    version, trial_index,
                    "final_snapshot_semantic_digest_bytes",
                )
            ] = np.frombuffer(
                bytes.fromhex(trace["final_snapshot_semantic_digest_v1"]),
                dtype=np.uint8,
            ).copy()
        probes[version] = {
            "snapshot_identity": identity,
            "trial_1_behavioural_digest_v1": trial_digests[1],
            "final_snapshot_semantic_digests": [
                trace["final_snapshot_semantic_digest_v1"]
                for trace in validated_traces
            ],
            "trial_stuck": [trace["stuck"] for trace in validated_traces],
            "trial_termination_reasons": [
                trace["termination_reason"] for trace in validated_traces
            ],
            "trial_pair_comparison": compare_behavioural_probe_traces(
                validated_traces[0], validated_traces[1]
            ),
            "trace_member_manifests": trace_manifests,
        }
    metadata = {
        "snapshot_semantic_evidence": semantic,
        "snapshot_identity": copy.deepcopy(probes["V3"]["snapshot_identity"]),
        "behavioural_probes": probes,
    }
    validate_qualification_shard_augmentation(
        metadata,
        pool_index=pool_index,
        reopened_arrays={"snapshot_payload_bytes": payload, **additions},
    )
    return {"metadata": metadata, "arrays": additions}


def validate_qualification_shard_augmentation(
    value: Any,
    *,
    pool_index: int,
    reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one reopened qualification payload and its V3 augmentation."""

    import numpy as np

    versions = _qualification_versions(pool_index)
    row = _mapping(
        value, C.QUALIFICATION_SHARD_AUGMENTATION_FIELDS,
        "qualification shard augmentation",
    )
    semantic = validate_snapshot_semantic_evidence(
        row["snapshot_semantic_evidence"]
    )
    root_identity = _version_identity(
        row["snapshot_identity"], "qualification snapshot identity"
    )
    probes = row["behavioural_probes"]
    if not isinstance(probes, Mapping) or set(probes) != set(versions):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification behavioural version inventory drift"
        )
    if not isinstance(reopened_arrays, Mapping):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification reopened arrays are not a mapping"
        )
    expected_additions = _probe_addition_member_names(pool_index)
    observed_additions = {
        str(name) for name in reopened_arrays
        if name == "snapshot_semantic_bytes" or str(name).startswith("probe__")
    }
    if observed_additions != expected_additions:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification probe NPZ member inventory drift"
        )
    if "snapshot_payload_bytes" not in reopened_arrays:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification snapshot transport member is absent"
        )
    payload = np.asarray(reopened_arrays["snapshot_payload_bytes"])
    semantic_bytes_array = np.asarray(reopened_arrays["snapshot_semantic_bytes"])
    if (
        payload.dtype.str != "|u1" or payload.ndim != 1 or payload.size == 0
        or semantic_bytes_array.dtype.str != "|u1"
        or semantic_bytes_array.ndim != 1 or semantic_bytes_array.size == 0
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification snapshot payload dtype/shape drift"
        )
    canonical = np.ascontiguousarray(semantic_bytes_array).tobytes(order="C")
    validate_snapshot_semantic_bytes(canonical, semantic)
    artifact_sha = hashlib.sha256(
        np.ascontiguousarray(payload).tobytes(order="C")
    ).hexdigest()
    if (
        root_identity["artifact_file_sha256"] != artifact_sha
        or root_identity["snapshot_semantic_digest_v1"]
        != semantic["snapshot_semantic_digest_v1"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "qualification root snapshot identity cross-link drift"
        )
    for version in versions:
        version_row = _mapping(
            probes[version], C.BEHAVIOURAL_PROBE_VERSION_EVIDENCE_FIELDS,
            f"qualification {version} probe evidence",
        )
        identity = _version_identity(
            version_row["snapshot_identity"],
            f"qualification {version} snapshot identity",
        )
        if version == "V3" and identity != root_identity:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification V3 identity root cross-link drift"
            )
        trace_rows: list[dict[str, Any]] = []
        observed_manifests = _sequence(
            version_row["trace_member_manifests"],
            f"qualification {version} trace manifests",
        )
        stuck_rows = _sequence(
            version_row["trial_stuck"], f"qualification {version} stuck rows"
        )
        terminations = _sequence(
            version_row["trial_termination_reasons"],
            f"qualification {version} termination rows",
        )
        finals = _sequence(
            version_row["final_snapshot_semantic_digests"],
            f"qualification {version} final semantic digests",
        )
        if not all(len(items) == 2 for items in (
            observed_manifests, stuck_rows, terminations, finals
        )):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification behavioural trial metadata cardinality drift"
            )
        for trial_index in (0, 1):
            trace: dict[str, Any] = {}
            for member, authority in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
                name = _probe_member_name(version, trial_index, member)
                array = np.asarray(reopened_arrays[name])
                expected_shape = [750, *authority["shape"][1:]]
                if array.dtype.str != authority["descr"] or list(array.shape) != expected_shape:
                    raise PhysicalGraphEdgeHandoffV3MetricsError(
                        f"qualification {name} dtype/shape drift"
                    )
                trace[member] = np.ascontiguousarray(array)
            digest_name = _probe_member_name(
                version, trial_index,
                "final_snapshot_semantic_digest_bytes",
            )
            digest_array = np.asarray(reopened_arrays[digest_name])
            if digest_array.dtype.str != "|u1" or list(digest_array.shape) != [32]:
                raise PhysicalGraphEdgeHandoffV3MetricsError(
                    f"qualification {digest_name} dtype/shape drift"
                )
            final_digest = np.ascontiguousarray(digest_array).tobytes(order="C").hex()
            _sha(final_digest, f"qualification {version} final semantic digest")
            trace["final_snapshot_semantic_digest_v1"] = final_digest
            trace["termination_reason"] = terminations[trial_index]
            trace["stuck"] = stuck_rows[trial_index]
            trace = _trace_arrays(
                trace, f"qualification {version} trial {trial_index}"
            )
            if (
                trace["termination_reason"] != "H3_COMPLETE"
                or trace["stuck"] is not _derive_stuck(trace)
                or finals[trial_index] != final_digest
                or observed_manifests[trial_index]
                != _trace_manifest(trace)["arrays"]
            ):
                raise PhysicalGraphEdgeHandoffV3MetricsError(
                    "qualification behavioural trial projection drift"
                )
            trace_rows.append(trace)
        if (
            identity["snapshot_behavioural_digest_v1"]
            != snapshot_behavioural_digest(trace_rows[0])
            or version_row["trial_1_behavioural_digest_v1"]
            != snapshot_behavioural_digest(trace_rows[1])
            or version_row["trial_pair_comparison"]
            != compare_behavioural_probe_traces(trace_rows[0], trace_rows[1])
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "qualification behavioural identity/comparison drift"
            )
    return row


# ---------------------------------------------------------------------------
# Frozen behavioural probe
# ---------------------------------------------------------------------------

_BEHAVIOURAL_TRACE_FIELDS = frozenset(
    set(C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY)
    | {"termination_reason", "stuck", "final_snapshot_semantic_digest_v1"}
)


def _trace_arrays(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    import numpy as np

    row = _mapping(value, _BEHAVIOURAL_TRACE_FIELDS, label)
    if row["termination_reason"] != C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY[
        "completion_termination_reason"
    ] or type(row["stuck"]) is not bool:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"{label} termination/stuck drift"
        )
    _sha(
        row["final_snapshot_semantic_digest_v1"],
        f"{label}.final_snapshot_semantic_digest_v1",
    )
    for member, authority in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        array = np.asarray(row[member])
        expected_shape = [
            C.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES,
            *authority["shape"][1:],
        ]
        if array.dtype.str != authority["descr"] or list(array.shape) != expected_shape:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"{label}.{member} dtype/shape drift"
            )
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"{label}.{member} is nonfinite"
            )
        row[member] = np.ascontiguousarray(array)
    samples_per_act = C.BEHAVIOURAL_PROBE_AUTHORITY[
        "controller_policy_sampling"
    ]["physics_samples_per_policy_act"]
    for member in ("controller_observation", "policy_output"):
        array = row[member]
        blocks = array.reshape(-1, samples_per_act, array.shape[1])
        if not bool(np.array_equal(blocks, np.repeat(blocks[:, :1], samples_per_act, axis=1))):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"{label}.{member} policy-act repetition drift"
            )
    expected_requested = np.broadcast_to(
        np.asarray(
            C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND,
            dtype=np.float64,
        ),
        row["requested_command"].shape,
    )
    if not np.array_equal(row["requested_command"], expected_requested):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"{label}.requested_command fixture drift"
        )
    if not np.allclose(
        np.diff(row["timestamp_s"]),
        C.BEHAVIOURAL_PROBE_AUTHORITY["physics_dt_s"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"{label}.timestamp_s cadence drift"
        )
    return row


def _yaw_from_xyzw(quaternion: Sequence[float]) -> float:
    x, y, z, w = [float(value) for value in quaternion]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def compare_behavioural_probe_traces(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two 750-sample traces under the frozen reset authority."""

    import numpy as np

    a = _trace_arrays(left, "left trace")
    b = _trace_arrays(right, "right trace")
    authority = C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY
    exact_equal: dict[str, bool] = {}
    for member in authority["exact_members"]:
        exact_equal[member] = bool(np.array_equal(a[member], b[member]))
    tolerance = authority["samplewise_tolerances"]
    maxima = {
        "base_pose_world_position_xyz": float(
            np.max(np.abs(a["base_pose_world"][:, :3] - b["base_pose_world"][:, :3]))
        ),
        "base_pose_world_quaternion_xyzw": float(
            np.max(np.abs(a["base_pose_world"][:, 3:] - b["base_pose_world"][:, 3:]))
        ),
        "base_twist_world": float(
            np.max(np.abs(a["base_twist_world"] - b["base_twist_world"]))
        ),
        "joint_position": float(
            np.max(np.abs(a["joint_position"] - b["joint_position"]))
        ),
        "joint_velocity": float(
            np.max(np.abs(a["joint_velocity"] - b["joint_velocity"]))
        ),
        "controller_observation": float(
            np.max(
                np.abs(
                    a["controller_observation"] - b["controller_observation"]
                )
            )
        ),
        "policy_output": float(
            np.max(np.abs(a["policy_output"] - b["policy_output"]))
        ),
    }
    samplewise_pass = {
        name: value
        <= float(
            tolerance.get(
                name,
                C.BEHAVIOURAL_PROBE_AUTHORITY[
                    "controller_policy_samplewise_tolerance"
                ],
            )
        )
        for name, value in maxima.items()
    }
    endpoint_position_error = float(
        np.linalg.norm(a["base_pose_world"][-1, :3] - b["base_pose_world"][-1, :3])
    )
    endpoint_heading_error = abs(
        _wrapped(
            _yaw_from_xyzw(a["base_pose_world"][-1, 3:])
            - _yaw_from_xyzw(b["base_pose_world"][-1, 3:])
        )
    )
    result = {
        "exact_member_equal": exact_equal,
        "samplewise_max_abs_error": maxima,
        "samplewise_pass": samplewise_pass,
        "endpoint_position_error_m": endpoint_position_error,
        "endpoint_heading_error_rad": endpoint_heading_error,
        "termination_reason_equal": a["termination_reason"] == b["termination_reason"],
        "stuck_equal": a["stuck"] == b["stuck"],
        "final_snapshot_semantic_digest_equal": (
            a["final_snapshot_semantic_digest_v1"]
            == b["final_snapshot_semantic_digest_v1"]
        ),
    }
    result["pass"] = bool(
        all(exact_equal.values())
        and all(samplewise_pass.values())
        and endpoint_position_error <= authority["endpoint_position_tolerance_m"]
        and endpoint_heading_error <= authority["endpoint_heading_tolerance_rad"]
        and result["termination_reason_equal"]
        and result["stuck_equal"]
    )
    return result


def _trace_manifest(trace: Mapping[str, Any]) -> dict[str, Any]:
    row = _trace_arrays(trace, "behavioural trace")
    arrays = []
    for member in sorted(C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY):
        array = row[member]
        arrays.append(
            {
                "member": member,
                "dtype_str": array.dtype.str,
                "shape": [int(value) for value in array.shape],
                "array_bytes_sha256": hashlib.sha256(
                    array.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return {
        "arrays": arrays,
        "termination_reason": row["termination_reason"],
        "stuck": row["stuck"],
        "final_snapshot_semantic_digest_v1": row[
            "final_snapshot_semantic_digest_v1"
        ],
    }


def snapshot_behavioural_digest(trace: Mapping[str, Any]) -> str:
    """Digest one designated deterministic restoration/probe trial."""

    projection = {
        "schema": "physical_graph_edge_handoff_snapshot_behavioural_digest_v1",
        "trace": _trace_manifest(trace),
    }
    return hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()


def _validate_behavioural_probe_npz_arrays(
    value: Mapping[str, Any],
    authority: Mapping[str, Mapping[str, Any]],
    *,
    first_eight_only: bool,
) -> dict[str, Any]:
    import numpy as np

    if not isinstance(value, Mapping) or set(value) != set(authority):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe NPZ member drift"
        )
    arrays: dict[str, Any] = {}
    for member, member_authority in authority.items():
        array = np.asarray(value[member])
        if array.dtype.str != member_authority["descr"] or list(array.shape) != member_authority["shape"]:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"behavioural probe {member} dtype/shape drift"
            )
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"behavioural probe {member} is nonfinite"
            )
        arrays[member] = np.ascontiguousarray(array)
    trace_count = 48 if first_eight_only else C.BEHAVIOURAL_PROBE_TRACE_COUNT
    expected_offsets = np.arange(0, trace_count * 750 + 1, 750, dtype=np.int64)
    expected_versions: list[int] = []
    expected_pools: list[int] = []
    expected_trials: list[int] = []
    for pool_index in range(8 if first_eight_only else 256):
        version_codes = (1, 2, 3) if pool_index < 8 else (3,)
        for version_code in version_codes:
            for trial_index in (0, 1):
                expected_versions.append(version_code)
                expected_pools.append(pool_index)
                expected_trials.append(trial_index)
    if (
        not np.array_equal(arrays["trace_offsets"], expected_offsets)
        or not np.array_equal(arrays["version_code"], expected_versions)
        or not np.array_equal(arrays["pool_index"], expected_pools)
        or not np.array_equal(arrays["trial_index"], expected_trials)
        or not np.array_equal(arrays["termination_code"], np.ones(trace_count, dtype=np.int64))
        or any(value not in (0, 1) for value in arrays["stuck"].tolist())
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe trace order drift"
        )
    return arrays


def validate_behavioural_probe_npz_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete official 544-trace behavioural NPZ projection."""

    return _validate_behavioural_probe_npz_arrays(
        value, C.BEHAVIOURAL_PROBE_NPZ_AUTHORITY, first_eight_only=False
    )


def validate_first_eight_behavioural_probe_npz_arrays(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable pre-pool-8 48-trace material subset."""

    return _validate_behavioural_probe_npz_arrays(
        value, C.FIRST_EIGHT_BEHAVIOURAL_PROBE_NPZ_AUTHORITY,
        first_eight_only=True,
    )


def behavioural_probe_npz_projection_sha256(
    value: Mapping[str, Any], *, first_eight_only: bool
) -> str:
    arrays = (
        validate_first_eight_behavioural_probe_npz_arrays(value)
        if first_eight_only
        else validate_behavioural_probe_npz_arrays(value)
    )
    rows = [
        {
            "member": member,
            "dtype_str": arrays[member].dtype.str,
            "shape": [int(size) for size in arrays[member].shape],
            "array_bytes_sha256": hashlib.sha256(
                arrays[member].tobytes(order="C")
            ).hexdigest(),
        }
        for member in sorted(arrays)
    ]
    return hashlib.sha256(C.canonical_json_bytes(rows)[:-1]).hexdigest()


def validate_final_behavioural_probe_assembly(
    full_value: Mapping[str, Any], first_eight_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the immutable 48-trace gate subset is the final NPZ prefix."""

    import numpy as np

    full = validate_behavioural_probe_npz_arrays(full_value)
    early = validate_first_eight_behavioural_probe_npz_arrays(first_eight_value)
    trace_rows = {
        "trace_offsets": 49,
        "version_code": 48,
        "pool_index": 48,
        "trial_index": 48,
        "stuck": 48,
        "termination_code": 48,
        "final_snapshot_semantic_digest_bytes": 48,
    }
    for member, stop in trace_rows.items():
        if not np.array_equal(full[member][:stop], early[member]):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"final behavioural probe prefix drift: {member}"
            )
    for member in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
        if not np.array_equal(full[member][:36000], early[member]):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"final behavioural probe prefix drift: {member}"
            )
    return full


def _behavioural_trace_indices(pool_index: int, version: str) -> list[int]:
    if (
        not isinstance(pool_index, int)
        or isinstance(pool_index, bool)
        or not 0 <= pool_index < 256
        or version not in C.BEHAVIOURAL_PROBE_VERSION_ORDER
        or (pool_index >= 8 and version != "V3")
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe identity drift"
        )
    if pool_index < 8:
        first = pool_index * 6 + C.BEHAVIOURAL_PROBE_VERSION_ORDER.index(version) * 2
    else:
        first = 48 + (pool_index - 8) * 2
    return [first, first + 1]


def behavioural_probe_trials_from_arrays(
    value: Mapping[str, Any], pool_index: int, version: str,
    *, first_eight_only: bool = False,
) -> list[dict[str, Any]]:
    arrays = (
        validate_first_eight_behavioural_probe_npz_arrays(value)
        if first_eight_only
        else validate_behavioural_probe_npz_arrays(value)
    )
    return _behavioural_probe_trials_from_validated_arrays(
        arrays, pool_index, version, first_eight_only=first_eight_only
    )


def _behavioural_probe_trials_from_validated_arrays(
    arrays: Mapping[str, Any], pool_index: int, version: str,
    *, first_eight_only: bool,
) -> list[dict[str, Any]]:
    if (
        version not in C.BEHAVIOURAL_PROBE_VERSION_ORDER
        or not 0 <= pool_index < (8 if first_eight_only else 256)
        or (pool_index >= 8 and version != "V3")
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe identity drift"
        )
    trace_indices = _behavioural_trace_indices(pool_index, version)
    result = []
    for trace_index in trace_indices:
        start = int(arrays["trace_offsets"][trace_index])
        stop = int(arrays["trace_offsets"][trace_index + 1])
        row = {
            member: arrays[member][start:stop]
            for member in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY
        }
        # Completion/stuck are regenerated from raw arrays by the independent
        # evaluator.  The frozen fixture is active and completes H3.
        row["termination_reason"] = "H3_COMPLETE"
        row["stuck"] = _derive_stuck(row)
        if bool(arrays["stuck"][trace_index]) is not row["stuck"]:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "behavioural probe stuck projection drift"
            )
        row["final_snapshot_semantic_digest_v1"] = bytes(
            arrays["final_snapshot_semantic_digest_bytes"][trace_index].tolist()
        ).hex()
        result.append(row)
    return result


def _derive_stuck(trace: Mapping[str, Any]) -> bool:
    import numpy as np

    requested = np.asarray(trace["requested_command"])
    poses = np.asarray(trace["base_pose_world"])
    active = float(np.max(np.abs(requested))) > float(
        C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"]
    )
    translation = float(np.linalg.norm(poses[-1, :2] - poses[0, :2]))
    heading = abs(
        _wrapped(_yaw_from_xyzw(poses[-1, 3:]) - _yaw_from_xyzw(poses[0, 3:]))
    )
    authority = C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]
    return bool(
        active
        and translation < float(authority["h3_translation_threshold_m"])
        and heading < float(authority["h3_heading_threshold_rad"])
    )


# ---------------------------------------------------------------------------
# Equivalence, custody, and technical gate
# ---------------------------------------------------------------------------

def _version_identity(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.SNAPSHOT_IDENTITY_FIELDS, label)
    for key in C.SNAPSHOT_IDENTITY_FIELDS:
        _sha(row[key], f"{label}.{key}")
    return row


def build_snapshot_equivalence_record(
    *,
    pool_index: int,
    versions: Mapping[str, Mapping[str, Any]],
    behavioural_probe_traces: Mapping[str, Sequence[Mapping[str, Any]]],
    semantic_manifests_equal: bool,
) -> dict[str, Any]:
    """Build one trace-derived historical or V3-only equivalence row."""

    expected_versions = set(_qualification_versions(pool_index))
    if (
        not isinstance(versions, Mapping) or set(versions) != expected_versions
        or not isinstance(behavioural_probe_traces, Mapping)
        or set(behavioural_probe_traces) != expected_versions
        or type(semantic_manifests_equal) is not bool
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence builder inventory drift"
        )
    identities = {
        version: _version_identity(
            versions[version], f"equivalence {version} identity"
        )
        for version in expected_versions
    }
    traces: dict[str, list[dict[str, Any]]] = {}
    within: dict[str, bool] = {}
    trial_one: dict[str, str] = {}
    finals: dict[str, list[str]] = {}
    trace_indices: dict[str, list[int]] = {}
    for version in expected_versions:
        rows = _sequence(
            behavioural_probe_traces[version],
            f"equivalence {version} traces",
        )
        if len(rows) != 2:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "snapshot equivalence trial cardinality drift"
            )
        traces[version] = [
            _trace_arrays(row, f"equivalence {version} trial {trial}")
            for trial, row in enumerate(rows)
        ]
        trial_zero_digest = snapshot_behavioural_digest(traces[version][0])
        if (
            identities[version]["snapshot_behavioural_digest_v1"]
            != trial_zero_digest
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "snapshot equivalence designated behavioural identity drift"
            )
        trial_one[version] = snapshot_behavioural_digest(traces[version][1])
        finals[version] = [
            trace["final_snapshot_semantic_digest_v1"]
            for trace in traces[version]
        ]
        trace_indices[version] = _behavioural_trace_indices(pool_index, version)
        within[version] = compare_behavioural_probe_traces(
            traces[version][0], traces[version][1]
        )["pass"]
    historical = pool_index < 8
    pair_behaviour: dict[tuple[str, str], bool] = {}
    if historical:
        for left, right in (("V1", "V2"), ("V1", "V3"), ("V2", "V3")):
            pair_behaviour[(left, right)] = all(
                compare_behavioural_probe_traces(
                    traces[left][trial], traces[right][trial]
                )["pass"]
                for trial in (0, 1)
            )
    spec = C.build_prospective_pool_specs()[pool_index]
    row: dict[str, Any] = {
        "pool_index": pool_index,
        **{
            key: spec[key]
            for key in (
                "candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id"
            )
        },
        "versions": identities,
        "v1_v2_artifact_file_sha256_equal": (
            identities["V1"]["artifact_file_sha256"]
            == identities["V2"]["artifact_file_sha256"]
        ) if historical else None,
        "v1_v3_artifact_file_sha256_equal": (
            identities["V1"]["artifact_file_sha256"]
            == identities["V3"]["artifact_file_sha256"]
        ) if historical else None,
        "v2_v3_artifact_file_sha256_equal": (
            identities["V2"]["artifact_file_sha256"]
            == identities["V3"]["artifact_file_sha256"]
        ) if historical else None,
        "v1_v2_semantic_equal": (
            identities["V1"]["snapshot_semantic_digest_v1"]
            == identities["V2"]["snapshot_semantic_digest_v1"]
        ) if historical else None,
        "v1_v3_semantic_equal": (
            identities["V1"]["snapshot_semantic_digest_v1"]
            == identities["V3"]["snapshot_semantic_digest_v1"]
        ) if historical else None,
        "v2_v3_semantic_equal": (
            identities["V2"]["snapshot_semantic_digest_v1"]
            == identities["V3"]["snapshot_semantic_digest_v1"]
        ) if historical else None,
        "v1_v2_behavioural_equal": pair_behaviour.get(("V1", "V2")),
        "v1_v3_behavioural_equal": pair_behaviour.get(("V1", "V3")),
        "v2_v3_behavioural_equal": pair_behaviour.get(("V2", "V3")),
        "v1_restore_trials_equal": within.get("V1"),
        "v2_restore_trials_equal": within.get("V2"),
        "v3_restore_trials_equal": within["V3"],
        "historical_behavioural_probe_evidence_present": historical,
        "semantic_manifests_equal": semantic_manifests_equal,
        "historical_comparison_applicable": historical,
        "trial_1_behavioural_digests": trial_one,
        "final_snapshot_semantic_digests": finals,
        "behavioural_probe_trace_indices": trace_indices,
        "pass": False,
    }
    if historical:
        row["pass"] = bool(
            semantic_manifests_equal
            and all(within.values())
            and all(pair_behaviour.values())
            and all(
                row[field]
                for field in (
                    "v1_v2_semantic_equal", "v1_v3_semantic_equal",
                    "v2_v3_semantic_equal",
                )
            )
        )
    else:
        row["pass"] = bool(semantic_manifests_equal and within["V3"])
    return _validate_equivalence_record(row, pool_index)


def _validate_equivalence_record(
    value: Any, index: int, *, require_pass: bool | None = None
) -> dict[str, Any]:
    row = _mapping(
        value, C.SNAPSHOT_EQUIVALENCE_RECORD_FIELDS, f"equivalence[{index}]"
    )
    spec = C.build_prospective_pool_specs()[index]
    identity = {
        "pool_index": index,
        "candidate_spec_id": spec["candidate_spec_id"],
        "state_id": spec["state_id"],
        "scene_id": spec["scene_id"],
        "episode_id": spec["episode_id"],
        "graph_id": spec["graph_id"],
    }
    if any(row[key] != expected for key, expected in identity.items()):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence identity/order drift"
        )
    if not isinstance(row["versions"], Mapping):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence version inventory drift"
        )
    historical = index < 8
    if row["historical_comparison_applicable"] is not historical:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical comparison applicability drift"
        )
    expected_versions = {"V1", "V2", "V3"} if historical else {"V3"}
    if set(row["versions"]) != expected_versions:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence version inventory drift"
        )
    versions: dict[str, dict[str, Any] | None] = {"V1": None, "V2": None}
    for version in expected_versions:
        versions[version] = _version_identity(
            row["versions"][version], f"{version} identity"
        )
    assert versions["V3"] is not None

    pair_fields = (
        "v1_v2_artifact_file_sha256_equal",
        "v1_v3_artifact_file_sha256_equal",
        "v2_v3_artifact_file_sha256_equal",
        "v1_v2_semantic_equal", "v1_v3_semantic_equal", "v2_v3_semantic_equal",
        "v1_v2_behavioural_equal", "v1_v3_behavioural_equal",
        "v2_v3_behavioural_equal",
    )
    if historical:
        assert versions["V1"] is not None and versions["V2"] is not None
        expected = {
            "v1_v2_artifact_file_sha256_equal": versions["V1"]["artifact_file_sha256"] == versions["V2"]["artifact_file_sha256"],
            "v1_v3_artifact_file_sha256_equal": versions["V1"]["artifact_file_sha256"] == versions["V3"]["artifact_file_sha256"],
            "v2_v3_artifact_file_sha256_equal": versions["V2"]["artifact_file_sha256"] == versions["V3"]["artifact_file_sha256"],
            "v1_v2_semantic_equal": versions["V1"]["snapshot_semantic_digest_v1"] == versions["V2"]["snapshot_semantic_digest_v1"],
            "v1_v3_semantic_equal": versions["V1"]["snapshot_semantic_digest_v1"] == versions["V3"]["snapshot_semantic_digest_v1"],
            "v2_v3_semantic_equal": versions["V2"]["snapshot_semantic_digest_v1"] == versions["V3"]["snapshot_semantic_digest_v1"],
        }
        if any(row[key] is not value for key, value in expected.items()):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "snapshot equivalence digest projection drift"
            )
        if any(type(row[field]) is not bool for field in pair_fields):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "historical pair evidence is not boolean"
            )
        if (
            type(row["v1_restore_trials_equal"]) is not bool
            or type(row["v2_restore_trials_equal"]) is not bool
            or row["historical_behavioural_probe_evidence_present"] is not True
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "historical restore evidence drift"
            )
    else:
        if any(row[field] is not None for field in pair_fields) or any(
            row[field] is not None
            for field in ("v1_restore_trials_equal", "v2_restore_trials_equal")
        ) or row["historical_behavioural_probe_evidence_present"] is not False:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "V3-only not-applicable evidence drift"
            )

    if (
        type(row["v3_restore_trials_equal"]) is not bool
        or type(row["semantic_manifests_equal"]) is not bool
        or type(row["pass"]) is not bool
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence gate evidence is not boolean"
        )
    trial_one = row["trial_1_behavioural_digests"]
    finals = row["final_snapshot_semantic_digests"]
    if not isinstance(trial_one, Mapping) or set(trial_one) != expected_versions:
        raise PhysicalGraphEdgeHandoffV3MetricsError("trial-one digest inventory drift")
    if not isinstance(finals, Mapping) or set(finals) != expected_versions:
        raise PhysicalGraphEdgeHandoffV3MetricsError("final semantic digest inventory drift")
    for version in expected_versions:
        _sha(trial_one[version], f"{version} trial-one behavioural digest")
        pair = _sequence(finals[version], f"{version} final semantic digests")
        if len(pair) != 2:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "final semantic trial cardinality drift"
            )
        _sha(pair[0], f"{version} final semantic digest[0]")
        _sha(pair[1], f"{version} final semantic digest[1]")
        # Final semantic identities are persisted for audit.  Exact digest
        # equality is descriptive: the frozen pass is the raw-trace comparison
        # under named tolerances and must not be silently strengthened to byte-
        # exact semantic equality after the probe.

    trace_indices = row["behavioural_probe_trace_indices"]
    if not isinstance(trace_indices, Mapping) or set(trace_indices) != expected_versions:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe trace-index inventory drift"
        )
    for version in expected_versions:
        expected_pair = _behavioural_trace_indices(index, version)
        observed_pair = _sequence(
            trace_indices[version], f"{version} behavioural trace indices"
        )
        if observed_pair != expected_pair:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "behavioural probe trace-index projection drift"
            )

    expected_pass = bool(
        row["v3_restore_trials_equal"]
        and row["semantic_manifests_equal"]
        and (
            not historical
            or all(
                row[field]
                for field in (
                    "v1_v2_semantic_equal", "v1_v3_semantic_equal",
                    "v2_v3_semantic_equal", "v1_v2_behavioural_equal",
                    "v1_v3_behavioural_equal", "v2_v3_behavioural_equal",
                    "v1_restore_trials_equal", "v2_restore_trials_equal",
                    "historical_behavioural_probe_evidence_present",
                )
            )
        )
    )
    if row["pass"] is not expected_pass:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence pass drift"
        )
    if require_pass is not None and row["pass"] is not require_pass:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence required disposition drift"
        )
    return row


def build_snapshot_equivalence_index(
    records: Sequence[Mapping[str, Any]],
    *,
    behavioural_probe_npz_binding: Mapping[str, Any],
    behavioural_probe_npz_projection_sha256: str,
    first_eight_material_probe_binding: Mapping[str, Any],
    first_eight_material_probe_projection_sha256: str,
    first_eight_prefix_exact: bool,
) -> dict[str, Any]:
    copied = [copy.deepcopy(dict(item)) for item in records]
    result = C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3."
                "snapshot_equivalence_index.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "serializer_authority_content_digest": C.SEMANTIC_SERIALIZER_AUTHORITY[
                "content_digest"
            ],
            "behavioural_probe_authority_content_digest": C.BEHAVIOURAL_PROBE_AUTHORITY[
                "content_digest"
            ],
            "row_count": len(copied),
            "historical_row_count": 8,
            "v3_only_row_count": 248,
            "records": copied,
            "pass": bool(len(copied) == 256 and all(item.get("pass") is True for item in copied)),
            "behavioural_probe_npz_binding": copy.deepcopy(
                dict(behavioural_probe_npz_binding)
            ),
            "behavioural_probe_npz_projection_sha256": (
                behavioural_probe_npz_projection_sha256
            ),
            "first_eight_material_probe_binding": copy.deepcopy(
                dict(first_eight_material_probe_binding)
            ),
            "first_eight_material_probe_projection_sha256": (
                first_eight_material_probe_projection_sha256
            ),
            "first_eight_prefix_exact": first_eight_prefix_exact,
        }
    )
    return validate_snapshot_equivalence_index(result)


def validate_snapshot_equivalence_index(value: Any) -> dict[str, Any]:
    row = _mapping(
        value, C.SNAPSHOT_EQUIVALENCE_INDEX_FIELDS, "snapshot_equivalence_index"
    )
    C.validate_content_digest(row)
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v3.snapshot_equivalence_index.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence index identity drift"
        )
    if (
        row["serializer_authority_content_digest"]
        != C.SEMANTIC_SERIALIZER_AUTHORITY["content_digest"]
        or row["behavioural_probe_authority_content_digest"]
        != C.BEHAVIOURAL_PROBE_AUTHORITY["content_digest"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence authority binding drift"
        )
    records = _sequence(row["records"], "snapshot equivalence records")
    if (
        row["row_count"] != 256
        or row["historical_row_count"] != 8
        or row["v3_only_row_count"] != 248
        or len(records) != 256
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence row count drift"
        )
    for index, item in enumerate(records):
        _validate_equivalence_record(item, index, require_pass=True)
    final_binding = _file_binding(
        row["behavioural_probe_npz_binding"], "official behavioural probe binding"
    )
    early_binding = _file_binding(
        row["first_eight_material_probe_binding"],
        "first-eight material probe binding",
    )
    assert final_binding is not None and early_binding is not None
    if (
        final_binding["path"] != "snapshot_behavioural_probes.npz"
        or early_binding["path"]
        != "reproduction/first_eight_behavioural_probes.npz"
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "behavioural probe binding path drift"
        )
    _sha(
        row["behavioural_probe_npz_projection_sha256"],
        "official behavioural probe projection sha256",
    )
    _sha(
        row["first_eight_material_probe_projection_sha256"],
        "first-eight material probe projection sha256",
    )
    if row["pass"] is not True or row["first_eight_prefix_exact"] is not True:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence index did not pass"
        )
    return row


def validate_snapshot_equivalence_evidence(
    index_value: Any,
    behavioural_probe_arrays: Mapping[str, Any],
    first_eight_behavioural_probe_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind all 256 index rows to the reopened 544-trace NPZ."""

    arrays = validate_final_behavioural_probe_assembly(
        behavioural_probe_arrays, first_eight_behavioural_probe_arrays
    )
    row = validate_snapshot_equivalence_index(index_value)
    if (
        row["behavioural_probe_npz_projection_sha256"]
        != behavioural_probe_npz_projection_sha256(
            arrays, first_eight_only=False
        )
        or row["first_eight_material_probe_projection_sha256"]
        != behavioural_probe_npz_projection_sha256(
            first_eight_behavioural_probe_arrays, first_eight_only=True
        )
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot equivalence NPZ projection cross-link drift"
        )
    records = row["records"]
    for pool_index, record in enumerate(records):
        versions = _qualification_versions(pool_index)
        traces = {
            version: _behavioural_probe_trials_from_validated_arrays(
                arrays, pool_index, version, first_eight_only=False
            )
            for version in versions
        }
        for version in versions:
            if (
                record["versions"][version]["snapshot_behavioural_digest_v1"]
                != snapshot_behavioural_digest(traces[version][0])
                or record["trial_1_behavioural_digests"][version]
                != snapshot_behavioural_digest(traces[version][1])
                or record["final_snapshot_semantic_digests"][version]
                != [
                    trace["final_snapshot_semantic_digest_v1"]
                    for trace in traces[version]
                ]
                or record[f"{version.lower()}_restore_trials_equal"]
                is not compare_behavioural_probe_traces(
                    traces[version][0], traces[version][1]
                )["pass"]
            ):
                raise PhysicalGraphEdgeHandoffV3MetricsError(
                    "snapshot equivalence trace projection drift"
                )
        if pool_index < 8:
            for left, right in (("V1", "V2"), ("V1", "V3"), ("V2", "V3")):
                expected = all(
                    compare_behavioural_probe_traces(
                        traces[left][trial], traces[right][trial]
                    )["pass"]
                    for trial in (0, 1)
                )
                if record[
                    f"{left.lower()}_{right.lower()}_behavioural_equal"
                ] is not expected:
                    raise PhysicalGraphEdgeHandoffV3MetricsError(
                        "snapshot equivalence cross-version trace projection drift"
                    )
    return row


def _file_binding(value: Any, label: str, *, nullable: bool = False) -> dict[str, Any] | None:
    if value is None and nullable:
        return None
    row = _mapping(value, {"path", "bytes", "sha256"}, label)
    if (
        not isinstance(row["path"], str)
        or not row["path"]
        or not isinstance(row["bytes"], int)
        or isinstance(row["bytes"], bool)
        or row["bytes"] <= 0
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} value drift")
    _sha(row["sha256"], f"{label}.sha256")
    return row


def validate_first_eight_reproduction(
    value: Any,
    *,
    historical_custody_receipt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _mapping(
        value, C.FIRST_EIGHT_REPRODUCTION_FIELDS, "first-eight reproduction"
    )
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v3."
        "v1_v2_v3_first_eight_reproduction.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight reproduction identity drift"
        )
    observed_historical_binding = _file_binding(
        row["historical_custody_receipt_binding"], "historical custody"
    )
    assert observed_historical_binding is not None
    if (
        observed_historical_binding != C.HISTORICAL_CUSTODY_RECEIPT_BINDING
        or historical_custody_receipt_binding is not None
        and observed_historical_binding
        != _file_binding(
            historical_custody_receipt_binding,
            "expected historical custody",
        )
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight historical custody binding drift"
        )
    if row["comparison_rule"] != C.FIRST_EIGHT_REPRODUCTION_AUTHORITY[
        "comparison_rule"
    ]:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight comparison rule drift"
        )
    records = _sequence(row["rows"], "first-eight rows")
    if row["row_count"] != 8 or len(records) != 8:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight row count drift"
        )
    validated = [
        _validate_equivalence_record(item, index)
        for index, item in enumerate(records)
    ]
    semantic_all = all(
        item[field]
        for item in validated
        for field in (
            "v1_v2_semantic_equal", "v1_v3_semantic_equal",
            "v2_v3_semantic_equal", "semantic_manifests_equal",
        )
    )
    behavioural_all = all(
        item[field]
        for item in validated
        for field in (
            "v1_v2_behavioural_equal", "v1_v3_behavioural_equal",
            "v2_v3_behavioural_equal", "v1_restore_trials_equal",
            "v2_restore_trials_equal", "v3_restore_trials_equal",
            "historical_behavioural_probe_evidence_present",
        )
    )
    all_pass = bool(semantic_all and behavioural_all and all(item["pass"] for item in validated))
    expected_status = "PASS" if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION
    if (
        row["semantic_all_pass"] is not semantic_all
        or row["behavioural_all_pass"] is not behavioural_all
        or row["pass"] is not all_pass
        or row["status"] != expected_status
        or row["technical_disposition"]
        != (None if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION)
        or row["full_collection_authorized"] is not all_pass
        or row["compared_before_pool_index"] != 8
        or row["candidate_ranker_development_heldout_outcomes_opened"] != 0
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight technical disposition drift"
        )
    probe_binding = _file_binding(
        row["first_eight_behavioural_probe_material_binding"],
        "first-eight behavioural probe material binding",
    )
    assert probe_binding is not None
    if probe_binding["path"] != (
        "reproduction/first_eight_behavioural_probes.npz"
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight material probe path drift"
        )
    _sha(
        row["first_eight_behavioural_probe_projection_sha256"],
        "first-eight behavioural probe projection sha256",
    )
    return row


def build_first_eight_reproduction(
    rows: Sequence[Mapping[str, Any]],
    *,
    historical_custody_receipt_binding: Mapping[str, Any],
    first_eight_behavioural_probe_material_binding: Mapping[str, Any],
    first_eight_behavioural_probe_projection_sha256: str,
) -> dict[str, Any]:
    copied = [copy.deepcopy(dict(item)) for item in rows]
    semantic_all = bool(
        len(copied) == 8
        and all(
            item.get(field) is True
            for item in copied
            for field in (
                "v1_v2_semantic_equal", "v1_v3_semantic_equal",
                "v2_v3_semantic_equal", "semantic_manifests_equal",
            )
        )
    )
    behavioural_all = bool(
        len(copied) == 8
        and all(
            item.get(field) is True
            for item in copied
            for field in (
                "v1_v2_behavioural_equal", "v1_v3_behavioural_equal",
                "v2_v3_behavioural_equal", "v1_restore_trials_equal",
                "v2_restore_trials_equal", "v3_restore_trials_equal",
                "historical_behavioural_probe_evidence_present",
            )
        )
    )
    all_pass = bool(
        semantic_all and behavioural_all and all(item.get("pass") is True for item in copied)
    )
    result = {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "v1_v2_v3_first_eight_reproduction.v1"
        ),
        "experiment_id": C.EXPERIMENT_ID,
        "status": "PASS" if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "historical_custody_receipt_binding": copy.deepcopy(
            dict(historical_custody_receipt_binding)
        ),
        "comparison_rule": C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["comparison_rule"],
        "row_count": len(copied),
        "rows": copied,
        "semantic_all_pass": semantic_all,
        "behavioural_all_pass": behavioural_all,
        "pass": all_pass,
        "technical_disposition": None if all_pass else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "full_collection_authorized": all_pass,
        "compared_before_pool_index": 8,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
        "first_eight_behavioural_probe_material_binding": copy.deepcopy(
            dict(first_eight_behavioural_probe_material_binding)
        ),
        "first_eight_behavioural_probe_projection_sha256": (
            first_eight_behavioural_probe_projection_sha256
        ),
    }
    return validate_first_eight_reproduction(result)


def validate_first_eight_reproduction_evidence(
    value: Any, first_eight_behavioural_probe_arrays: Mapping[str, Any]
) -> dict[str, Any]:
    """Cross-bind the technical gate receipt to its immutable 48 traces."""

    arrays = validate_first_eight_behavioural_probe_npz_arrays(
        first_eight_behavioural_probe_arrays
    )
    row = validate_first_eight_reproduction(value)
    if row["first_eight_behavioural_probe_projection_sha256"] != (
        behavioural_probe_npz_projection_sha256(arrays, first_eight_only=True)
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight behavioural material projection drift"
        )
    for pool_index, record in enumerate(row["rows"]):
        traces = {
            version: _behavioural_probe_trials_from_validated_arrays(
                arrays, pool_index, version, first_eight_only=True
            )
            for version in C.BEHAVIOURAL_PROBE_VERSION_ORDER
        }
        rebuilt = build_snapshot_equivalence_record(
            pool_index=pool_index,
            versions=record["versions"],
            behavioural_probe_traces=traces,
            semantic_manifests_equal=record["semantic_manifests_equal"],
        )
        if C.canonical_json_bytes(rebuilt) != C.canonical_json_bytes(record):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "first-eight raw trace/row cross-link drift"
            )
    return row


def authorizes_full_v3_collection(value: Any) -> bool:
    return bool(validate_first_eight_reproduction(value)["full_collection_authorized"])


# External combined receipt fields are frozen in the contract so the
# independent custody producer and the portable reducer share one exact schema.

def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} is not nonnegative int")
    return value


def _historical_file_row(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_FILE_FIELDS, label)
    if (
        not isinstance(row["path"], str) or not row["path"]
        or row["path"].startswith("/") or ".." in row["path"].split("/")
        or _nonnegative_int(row["bytes"], f"{label}.bytes") < 0
        or _nonnegative_int(row["allocated_bytes"], f"{label}.allocated_bytes") < 0
        or _nonnegative_int(row["device"], f"{label}.device") < 0
        or _nonnegative_int(row["inode"], f"{label}.inode") <= 0
        or not isinstance(row["nlink"], int) or isinstance(row["nlink"], bool)
        or row["nlink"] != 1
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} value drift")
    _sha(row["sha256"], f"{label}.sha256")
    return row


def _historical_directory_row(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_DIRECTORY_FIELDS, label)
    if (
        not isinstance(row["path"], str) or not row["path"]
        or _nonnegative_int(row["allocated_bytes"], f"{label}.allocated_bytes") < 0
        or _nonnegative_int(row["device"], f"{label}.device") < 0
        or _nonnegative_int(row["inode"], f"{label}.inode") <= 0
        or not isinstance(row["nlink"], int) or isinstance(row["nlink"], bool)
        or row["nlink"] < 1
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} value drift")
    return row


def _historical_root(value: Any, label: str, expected_path: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_ROOT_FIELDS, label)
    if row["path"] != expected_path:
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} path drift")
    files = [
        _historical_file_row(item, f"{label}.files[{index}]")
        for index, item in enumerate(_sequence(row["files"], f"{label}.files"))
    ]
    directories = [
        _historical_directory_row(item, f"{label}.directories[{index}]")
        for index, item in enumerate(
            _sequence(row["directories"], f"{label}.directories")
        )
    ]
    for field in (
        "file_count", "directory_count", "regular_file_apparent_bytes",
        "regular_file_allocated_bytes", "directory_allocated_bytes",
        "allocated_bytes",
    ):
        _nonnegative_int(row[field], f"{label}.{field}")
    if (
        [item["path"] for item in files] != sorted(item["path"] for item in files)
        or [item["path"] for item in directories]
        != sorted(item["path"] for item in directories)
        or row["file_count"] != len(files)
        or row["directory_count"] != len(directories)
        or row["regular_file_apparent_bytes"] != sum(item["bytes"] for item in files)
        or row["regular_file_allocated_bytes"]
        != sum(item["allocated_bytes"] for item in files)
        or row["directory_allocated_bytes"]
        != sum(item["allocated_bytes"] for item in directories)
        or row["allocated_bytes"]
        != row["regular_file_allocated_bytes"] + row["directory_allocated_bytes"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} aggregate drift")
    identities = [(item["device"], item["inode"]) for item in files]
    if len(identities) != len(set(identities)):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} duplicate file inode")
    projection = copy.deepcopy(row)
    projection.pop("manifest_sha256")
    if row["manifest_sha256"] != hashlib.sha256(
        C.canonical_json_bytes(projection)
    ).hexdigest():
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} manifest digest drift")
    return row


def _historical_version(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_VERSION_FIELDS, label)
    _historical_file_row(row["metadata_binding"], f"{label}.metadata_binding")
    _historical_file_row(row["payload_binding"], f"{label}.payload_binding")
    _sha(row["artifact_file_sha256"], f"{label}.artifact_file_sha256")
    _sha(row["snapshot_semantic_digest_v1"], f"{label}.semantic_digest")
    _sha(row["semantic_payload_sha256"], f"{label}.semantic_payload_sha256")
    if (
        not isinstance(row["artifact_bytes"], int)
        or isinstance(row["artifact_bytes"], bool) or row["artifact_bytes"] <= 0
        or not isinstance(row["semantic_payload_bytes"], int)
        or isinstance(row["semantic_payload_bytes"], bool)
        or row["semantic_payload_bytes"] <= len(S.SEMANTIC_BINARY_MAGIC)
        or row["semantic_payload_sha256"] != row["snapshot_semantic_digest_v1"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} identity drift")
    evidence = validate_snapshot_semantic_evidence(row["semantic_evidence"])
    if (
        evidence["snapshot_semantic_digest_v1"]
        != row["snapshot_semantic_digest_v1"]
        or evidence["canonical_semantic_byte_count"] != row["semantic_payload_bytes"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(f"{label} evidence drift")
    return row


def _historical_pair(value: Any, index: int) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_PAIR_FIELDS, f"historical pair[{index}]")
    spec = C.build_prospective_pool_specs()[index]
    expected_identity = {
        "pool_index": index,
        **{
            key: spec[key]
            for key in (
                "candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id"
            )
        },
    }
    if any(row[key] != expected for key, expected in expected_identity.items()):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical pair identity/order drift"
        )
    v1 = _historical_version(row["v1"], f"historical pair[{index}].v1")
    v2 = _historical_version(row["v2"], f"historical pair[{index}].v2")
    projections = {
        "artifact_file_sha256_equal": (
            v1["artifact_file_sha256"] == v2["artifact_file_sha256"]
        ),
        "snapshot_semantic_digest_v1_equal": (
            v1["snapshot_semantic_digest_v1"]
            == v2["snapshot_semantic_digest_v1"]
        ),
        "semantic_evidence_equal": (
            C.canonical_json_bytes(v1["semantic_evidence"])
            == C.canonical_json_bytes(v2["semantic_evidence"])
        ),
    }
    if any(row[key] is not expected for key, expected in projections.items()):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical pair digest projection drift"
        )
    required_true = C.HISTORICAL_CUSTODY_PAIR_FIELDS - {
        "pool_index", "candidate_spec_id", "state_id", "scene_id", "episode_id",
        "graph_id", "v1", "v2", "artifact_file_sha256_equal",
    }
    if row["artifact_file_sha256_equal"] is not False or any(
        row[field] is not True for field in required_true
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical pair substantive gate failed"
        )
    return row


def historical_custody_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete validated receipt as the immutable bound projection."""

    return validate_external_v1_v2_custody_receipt(value)


def historical_custody_projection_sha256(value: Mapping[str, Any]) -> str:
    projection = historical_custody_projection(value)
    return hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()


def validate_external_v1_v2_custody_receipt(
    value: Any, *, expected_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    row = _mapping(
        value, C.EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS,
        "external historical custody receipt",
    )
    if (
        row["schema"]
        != "physical_graph_edge_handoff_qualification_v1_v2.custody_receipt.v1"
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["generated_before_v3_simulator_creation"] is not True
        or row["v1_external_custody_receipt_binding"] != C.V1_CUSTODY_RECEIPT_BINDING
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical custody receipt identity/binding drift"
        )
    canonical = C.canonical_json_bytes(row)
    actual_binding = {
        "path": str(C.HISTORICAL_CUSTODY_RECEIPT_PATH),
        "bytes": len(canonical),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }
    if actual_binding != C.HISTORICAL_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical custody receipt differs from the frozen exact bytes"
        )
    if expected_binding is not None:
        binding = _file_binding(expected_binding, "historical custody receipt binding")
        assert binding is not None
        if binding != actual_binding:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "historical custody receipt byte binding drift"
            )
    roots = {
        "v1_official_root": _historical_root(
            row["v1_official_root"], "v1_official_root", str(C.V1_OFFICIAL_ROOT)
        ),
        "v1_material_root": _historical_root(
            row["v1_material_root"], "v1_material_root", str(C.V1_MATERIAL_ROOT)
        ),
        "v2_official_root": _historical_root(
            row["v2_official_root"], "v2_official_root", str(C.V2_OFFICIAL_ROOT)
        ),
        "v2_material_root": _historical_root(
            row["v2_material_root"], "v2_material_root", str(C.V2_MATERIAL_ROOT)
        ),
    }
    expected_root_counts = {
        "v1_official_root": (1, 66418),
        "v1_material_root": (34, 4299695),
        "v2_official_root": (4, 102527),
        "v2_material_root": (34, 4382459),
    }
    if any(
        (root["file_count"], root["regular_file_apparent_bytes"])
        != expected_root_counts[name]
        for name, root in roots.items()
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical custody root projection drift"
        )
    all_inodes = [
        (item["device"], item["inode"])
        for root in roots.values() for item in root["files"]
    ]
    if len(all_inodes) != len(set(all_inodes)):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical roots share a file inode"
        )
    terminal = _mapping(
        row["v2_terminal_evidence"], C.HISTORICAL_CUSTODY_V2_TERMINAL_FIELDS,
        "V2 terminal evidence",
    )
    first_eight_binding = _file_binding(
        terminal["first_eight_receipt_binding"], "V2 terminal first-eight binding"
    )
    assert first_eight_binding is not None
    if terminal != {
        "disposition": C.V2.REPRODUCTION_MISMATCH_DISPOSITION,
        "official_leaf_count": 4,
        "material_pair_count": 8,
        "full_collection_authorized": False,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
        "external_regeneration_receipt_present": False,
        "first_eight_receipt_binding": first_eight_binding,
    } or first_eight_binding != {
        "path": str(C.V2_OFFICIAL_ROOT / "v1_v2_first_eight_reproduction.json"),
        "bytes": 6797,
        "sha256": "d3437d2560af6005b871b94bed6ef3f7f0f185497984725eae41de80224069c6",
    }:
        raise PhysicalGraphEdgeHandoffV3MetricsError("V2 terminal evidence drift")
    pairs = [
        _historical_pair(item, index)
        for index, item in enumerate(
            _sequence(row["first_eight_pairs"], "historical first-eight pairs")
        )
    ]
    if len(pairs) != 8:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "historical first-eight pair count drift"
        )
    material_files = {
        "V1": {item["path"]: item for item in roots["v1_material_root"]["files"]},
        "V2": {item["path"]: item for item in roots["v2_material_root"]["files"]},
    }
    for index, pair in enumerate(pairs):
        for version, key in (("V1", "v1"), ("V2", "v2")):
            for binding_name in ("metadata_binding", "payload_binding"):
                binding = pair[key][binding_name]
                if material_files[version].get(binding["path"]) != binding:
                    raise PhysicalGraphEdgeHandoffV3MetricsError(
                        f"historical pair[{index}] {version} {binding_name} root cross-link drift"
                    )
    expected_scientific = {
        "v1_teacher_qualification_rows_opened": 8,
        "v1_teacher_qualified": 7,
        "v1_teacher_rejected": 1,
        "v2_fresh_teacher_qualification_rows_opened": 8,
        "v2_teacher_qualified": 7,
        "v2_teacher_rejected": 1,
        "v2_pool_002_physics_contact_reproduced": True,
        "selection_performed": False,
        "reset_fixture_executions": 0,
        "candidate_fanout_executions": 0,
        "encoder_initializations": 0,
        "ranker_inference_calls": 0,
        "heldout_outcomes_opened": 0,
        "metrics_persisted": False,
        "result_persisted": False,
    }
    expected_repository = {
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": C.V2_SOURCE_FREEZE_COMMIT,
        "v1_freeze_subject": "Freeze physical graph edge handoff qualification",
        "v2_freeze_subject": "Freeze corrected physical graph edge handoff qualification V2",
    }
    expected_immutability = {
        "audit_mode": "read_only_same_inode_before_after",
        "v1_official_root_unchanged_during_audit": True,
        "v1_material_root_unchanged_during_audit": True,
        "v2_official_root_unchanged_during_audit": True,
        "v2_material_root_unchanged_during_audit": True,
        "all_files_regular_single_link": True,
        "receipt_outside_all_experiment_roots": True,
    }
    expected_nonreuse = {
        "historical_roots_shared_inode_count": 0,
        "v1_v2_artifact_file_sha256_equal_count": 0,
        "v1_v2_snapshot_semantic_digest_v1_equal_count": 8,
        "raw_artifact_inequality_is_descriptive": True,
        "historical_payload_copy_into_v3_count": 0,
        "historical_hardlink_into_v3_count": 0,
        "historical_runtime_artifact_or_shard_reused": False,
        "historical_deserializer_invocations": 16,
        "model_initializations": 0,
        "training_runs": 0,
        "simulator_initializations": 0,
        "runner_calls": 0,
        "encoder_calls": 0,
        "ranker_calls": 0,
    }
    nested = (
        ("scientific_boundary", C.HISTORICAL_CUSTODY_SCIENTIFIC_BOUNDARY_FIELDS, expected_scientific),
        ("repository", C.HISTORICAL_CUSTODY_REPOSITORY_FIELDS, expected_repository),
        ("immutability", C.HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS, expected_immutability),
        ("nonreuse", C.HISTORICAL_CUSTODY_NONREUSE_FIELDS, expected_nonreuse),
    )
    for name, fields, expected in nested:
        observed = _mapping(row[name], fields, f"historical custody {name}")
        if observed != expected:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"historical custody {name} projection drift"
            )
    return row


def build_v1_v2_custody_and_nonreuse(
    *,
    source_freeze_commit: str,
    external_binding: Mapping[str, Any],
    external_projection_sha256: str,
) -> dict[str, Any]:
    result = {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "v1_v2_custody_and_nonreuse.v1"
        ),
        "experiment_id": C.EXPERIMENT_ID,
        "v3_source_freeze_commit": source_freeze_commit,
        "external_historical_custody_receipt_binding": copy.deepcopy(
            dict(external_binding)
        ),
        "external_historical_custody_projection_sha256": external_projection_sha256,
        "v1_official_root_unchanged": True,
        "v1_material_root_unchanged": True,
        "v2_official_root_unchanged": True,
        "v2_material_root_unchanged": True,
        "historical_payloads_copied_into_v3": 0,
        "historical_hardlinks_into_v3": 0,
        "historical_shared_inodes_with_v3": 0,
        "historical_runtime_artifact_or_shard_reused": False,
        "allowed_read_scope": "CUSTODY_SEMANTIC_AND_BEHAVIOURAL_FIRST_EIGHT_ONLY",
        "pass": True,
    }
    return validate_v1_v2_custody_and_nonreuse(result)


def validate_v1_v2_custody_and_nonreuse(
    value: Any, *, source_freeze_commit: str | None = None
) -> dict[str, Any]:
    row = _mapping(
        value, C.V1_V2_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS,
        "v1_v2_custody_and_nonreuse",
    )
    if row["schema"] != (
        "physical_graph_edge_handoff_qualification_v3."
        "v1_v2_custody_and_nonreuse.v1"
    ) or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV3MetricsError("nonreuse identity drift")
    freeze = C._commit(row["v3_source_freeze_commit"], "v3_source_freeze_commit")
    if source_freeze_commit is not None and freeze != source_freeze_commit:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "nonreuse source freeze cross-binding drift"
        )
    external_binding = _file_binding(
        row["external_historical_custody_receipt_binding"],
        "external historical custody binding",
    )
    assert external_binding is not None
    if external_binding != C.HISTORICAL_CUSTODY_RECEIPT_BINDING:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "external historical custody exact binding drift"
        )
    _sha(
        row["external_historical_custody_projection_sha256"],
        "external historical custody projection sha256",
    )
    if (
        any(
            row[field] is not True
            for field in (
                "v1_official_root_unchanged", "v1_material_root_unchanged",
                "v2_official_root_unchanged", "v2_material_root_unchanged",
            )
        )
        or any(
            not isinstance(row[field], int)
            or isinstance(row[field], bool)
            or row[field] != 0
            for field in (
                "historical_payloads_copied_into_v3",
                "historical_hardlinks_into_v3",
                "historical_shared_inodes_with_v3",
            )
        )
        or row["historical_runtime_artifact_or_shard_reused"] is not False
        or row["allowed_read_scope"]
        != "CUSTODY_SEMANTIC_AND_BEHAVIOURAL_FIRST_EIGHT_ONLY"
        or row["pass"] is not True
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError("historical nonreuse gate failed")
    return row


# ---------------------------------------------------------------------------
# Scientific invariance receipt
# ---------------------------------------------------------------------------

def build_semantic_regression_results(
    evidence: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    supplied = {} if evidence is None else dict(evidence)
    return [
        {
            "requirement_id": requirement,
            "passed": bool(supplied.get(requirement, False)),
            "evidence": "FOCUSED_PURE_OR_PRODUCTION_FIXTURE_PASS" if supplied.get(requirement, False) else "NOT_PROVEN",
        }
        for requirement in C.SEMANTIC_SERIALIZER_REGRESSION_IDS
    ]


def build_scientific_invariance_receipt(
    contract: Mapping[str, Any],
    semantic_regression_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated_contract = C.validate_contract(contract)
    results = _sequence(semantic_regression_results, "semantic regression results")
    if len(results) != len(C.SEMANTIC_SERIALIZER_REGRESSION_IDS):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "semantic regression result cardinality drift"
        )
    normalized = []
    for index, (item, requirement) in enumerate(
        zip(results, C.SEMANTIC_SERIALIZER_REGRESSION_IDS)
    ):
        part = _mapping(
            item, C.SEMANTIC_REGRESSION_RESULT_FIELDS,
            f"semantic regression[{index}]",
        )
        if (
            part["requirement_id"] != requirement
            or type(part["passed"]) is not bool
            or not isinstance(part["evidence"], str)
            or not part["evidence"]
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                "semantic regression result drift"
            )
        normalized.append(part)
    all_pass = all(item["passed"] for item in normalized)
    projection = C.scientific_invariance_projection(validated_contract)
    projection_sha = hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()
    receipt = {
        "schema": (
            "physical_graph_edge_handoff_qualification_v3."
            "scientific_invariance_receipt.v1"
        ),
        "experiment_id": C.EXPERIMENT_ID,
        "v2_contract_content_digest": C.V2_CONTRACT_CONTENT_DIGEST,
        "v3_contract_content_digest": validated_contract["content_digest"],
        "v2_scientific_projection_sha256": C.V2_SCIENTIFIC_PROJECTION_SHA256,
        "v3_scientific_projection_sha256": projection_sha,
        "scientific_projection_equal": projection == C.V2_SCIENTIFIC_PROJECTION,
        "candidate_specs_equal": C.build_candidate_specs() == C.V2.build_candidate_specs(),
        "source_dependency_paths_equal": tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V2.SOURCE_DEPENDENCY_PATHS),
        "raw_previous_command_hash_authority_retained": (
            C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
            == C.V2.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"]
        ),
        "port_heading_alignment_authority_content_digest": C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"],
        "candidate_port_metric_alignment_authority_content_digest": C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"],
        "serializer_authority_content_digest": C.SEMANTIC_SERIALIZER_AUTHORITY["content_digest"],
        "historical_deserializer_authority_content_digest": C.HISTORICAL_DESERIALIZER_AUTHORITY["content_digest"],
        "behavioural_probe_authority_content_digest": C.BEHAVIOURAL_PROBE_AUTHORITY["content_digest"],
        "qualification_shard_authority_content_digest": C.QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY["content_digest"],
        "result_publication_authority_content_digest": C.RESULT_PUBLICATION_AUTHORITY["content_digest"],
        "external_artifact_runtime_path_correction_authority_content_digest": C.EXTERNAL_ARTIFACT_RUNTIME_PATH_CORRECTION_AUTHORITY["content_digest"],
        "semantic_regression_results": normalized,
        "all_semantic_regressions_passed": all_pass,
        "official_documents_and_result_use_v3_identity_only": True,
        "pass": bool(
            all_pass
            and projection == C.V2_SCIENTIFIC_PROJECTION
            and C.build_candidate_specs() == C.V2.build_candidate_specs()
            and tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V2.SOURCE_DEPENDENCY_PATHS)
        ),
    }
    return receipt


def validate_scientific_invariance_receipt(value: Any) -> dict[str, Any]:
    row = _mapping(
        value, C.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS,
        "scientific_invariance_receipt",
    )
    if "content_digest" in row:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "receipt self-digest is forbidden"
        )
    results = _sequence(row["semantic_regression_results"], "semantic regressions")
    expected = build_scientific_invariance_receipt(C.build_contract(), results)
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected) or row["pass"] is not True:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "scientific invariance receipt value drift"
        )
    return row


# ---------------------------------------------------------------------------
# Exact inherited V2/V1 scientific validators and reducer
# ---------------------------------------------------------------------------

classify_physical_handoff_aggregates = V2M.classify_physical_handoff_aggregates
point_in_polygon_inclusive = V2M.point_in_polygon_inclusive
transverse_port_crossing = V2M.transverse_port_crossing
first_registered_port_crossing = V2M.first_registered_port_crossing
runtime_environment_sha256 = V2M.runtime_environment_sha256
validate_physical_runtime_environment = V2M.validate_physical_runtime_environment
validate_visual_runtime_environment = V2M.validate_visual_runtime_environment


def physical_trace_reduction_authority() -> dict[str, Any]:
    return _map_identity(V2M.physical_trace_reduction_authority(), to_v2=False)


def panel_manifest_authority() -> dict[str, Any]:
    return _map_identity(V2M.panel_manifest_authority(), to_v2=False)


def external_artifact_bindings(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    if contract.get("schema") == (
        "physical_graph_edge_handoff_qualification_v3.runtime_contract.v1"
    ):
        row = C.validate_runtime_contract(contract)
        return copy.deepcopy(row["external_artifact_bindings"])
    C.validate_contract(contract)
    return [copy.deepcopy(item) for item in C.EXTERNAL_ARTIFACT_BINDINGS]


def predecessor_context_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") == (
        "physical_graph_edge_handoff_qualification_v3.runtime_contract.v1"
    ):
        row = C.validate_runtime_contract(contract)["scientific_contract"]
    else:
        row = C.validate_contract(contract)
    if row["v2_context_binding"] != C.V2.V1.V2_CONTEXT_BINDING:
        raise PhysicalGraphEdgeHandoffV3MetricsError("predecessor context drift")
    return copy.deepcopy(row["v2_context_binding"])


def validate_npz_inspections(value: Any) -> dict[str, dict[str, Any]]:
    rows = _sequence(value, "npz_inspections")
    base_paths = set(C.V2.V1.NPZ_PAYLOAD_AUTHORITY)
    if len(rows) != len(base_paths) + 1:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 NPZ inspection count drift"
        )
    new_rows = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("path") == "snapshot_behavioural_probes.npz"
    ]
    base_rows = [
        row for row in rows
        if not isinstance(row, Mapping)
        or row.get("path") != "snapshot_behavioural_probes.npz"
    ]
    if len(new_rows) != 1:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 behavioural NPZ inspection is absent or duplicated"
        )
    try:
        by_path = V2M.validate_npz_inspections(base_rows)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    row = _mapping(
        new_rows[0], {"path", "bytes", "sha256", "members"},
        "snapshot_behavioural_probes.npz inspection",
    )
    if (
        not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool)
        or row["bytes"] <= 0
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot behavioural NPZ byte count drift"
        )
    _sha(row["sha256"], "snapshot behavioural NPZ sha256")
    members = row["members"]
    authority = C.BEHAVIOURAL_PROBE_NPZ_AUTHORITY
    if not isinstance(members, Mapping) or set(members) != set(authority):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot behavioural NPZ inspected member drift"
        )
    offsets: list[int] | None = None
    for member, spec in authority.items():
        observed = _mapping(
            members[member],
            {
                "descr", "digest_dtype", "shape", "c_contiguous",
                "object_dtype", "member_sha256", "row_or_slice_sha256s",
                "offset_values",
            },
            f"snapshot behavioural NPZ {member}",
        )
        shape = _sequence(observed["shape"], f"snapshot behavioural {member} shape")
        if (
            observed["descr"] != spec["descr"]
            or observed["digest_dtype"] != spec["digest_dtype"]
            or shape != spec["shape"]
            or observed["c_contiguous"] is not True
            or observed["object_dtype"] is not False
        ):
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"snapshot behavioural NPZ {member} layout drift"
            )
        _sha(observed["member_sha256"], f"snapshot behavioural {member} member SHA")
        digests = _sequence(
            observed["row_or_slice_sha256s"],
            f"snapshot behavioural {member} row/slice hashes",
        )
        for index, digest in enumerate(digests):
            _sha(digest, f"snapshot behavioural {member} digest[{index}]")
        expected_digest_count = (
            1 if spec["hash_mode"] == "whole"
            else shape[0] if spec["hash_mode"] == "rows_axis0"
            else C.BEHAVIOURAL_PROBE_TRACE_COUNT
        )
        if len(digests) != expected_digest_count:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"snapshot behavioural NPZ {member} digest coverage drift"
            )
        if member == "trace_offsets":
            offsets = _sequence(
                observed["offset_values"], "snapshot behavioural trace offsets"
            )
            if offsets != C.BEHAVIOURAL_PROBE_AUTHORITY["trace_offsets"]:
                raise PhysicalGraphEdgeHandoffV3MetricsError(
                    "snapshot behavioural trace offsets drift"
                )
        elif observed["offset_values"] is not None:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"snapshot behavioural NPZ {member} unexpected offsets"
            )
    if offsets is None:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "snapshot behavioural trace offsets absent"
        )
    by_path["snapshot_behavioural_probes.npz"] = row
    return by_path


def validate_panel_manifest(value: Any) -> dict[str, Any]:
    return _call_v2_document(V2M.validate_panel_manifest, value, label="panel_manifest")


def validate_split_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_split_manifest, value, panel_manifest, label="split_manifest"
    )


def validate_graph_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_graph_manifest, value, panel_manifest, label="graph_manifest"
    )


def validate_state_snapshot_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_state_snapshot_index, value, panel_manifest,
        label="state_snapshot_index",
    )


def validate_teacher_trace_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_teacher_trace_index, value, panel_manifest,
        label="teacher_trace_index",
    )


def validate_edge_port_index(
    value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_edge_port_index, value, panel_manifest, graph_manifest,
        teacher_trace_index, label="edge_port_index",
    )


def validate_waypoint_contracts(
    value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any],
    edge_port_index: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_waypoint_contracts, value, panel_manifest, graph_manifest,
        edge_port_index, label="waypoint_contracts",
    )


def validate_pixel_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(V2M.validate_pixel_index, value, panel_manifest, label="pixel_index")


def validate_latent_index(value: Any, pixel_index: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v2_document(V2M.validate_latent_index, value, pixel_index, label="latent_index")


def validate_candidate_fanout_rows(
    value: Any, panel_manifest: Mapping[str, Any], state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        V2M.validate_candidate_fanout_rows(
            _map_identity(value, to_v2=True),
            _map_identity(panel_manifest, to_v2=True),
            _map_identity(state_snapshot_index, to_v2=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    return copy.deepcopy(list(value))


def validate_development_target_selection(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]], waypoint_contracts: Mapping[str, Any],
) -> dict[str, Any]:
    return _call_v2_document(
        V2M.validate_development_target_selection, value, panel_manifest,
        candidate_fanout, waypoint_contracts, label="development_target_selection",
    )


def validate_heldout_ranker_score_rows(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    development_target_selection: Mapping[str, Any],
    waypoint_contracts: Mapping[str, Any], teacher_trace_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        V2M.validate_heldout_ranker_score_rows(
            _map_identity(value, to_v2=True), _map_identity(panel_manifest, to_v2=True),
            _map_identity(candidate_fanout, to_v2=True),
            _map_identity(development_target_selection, to_v2=True),
            _map_identity(waypoint_contracts, to_v2=True),
            _map_identity(teacher_trace_index, to_v2=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    return copy.deepcopy(list(value))


def validate_repeated_execution_rows(
    value: Any, panel_manifest: Mapping[str, Any],
    candidate_fanout: Sequence[Mapping[str, Any]],
    heldout_ranker_scores: Sequence[Mapping[str, Any]],
    state_snapshot_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        V2M.validate_repeated_execution_rows(
            _map_identity(value, to_v2=True), _map_identity(panel_manifest, to_v2=True),
            _map_identity(candidate_fanout, to_v2=True),
            _map_identity(heldout_ranker_scores, to_v2=True),
            _map_identity(state_snapshot_index, to_v2=True),
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(str(exc)) from exc
    return copy.deepcopy(list(value))


def recompute_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute with exact inherited V2/V1 physical formulas."""

    inherited_keys = set(V2M.V1M.EVIDENCE_KEYS)
    expected_keys = inherited_keys | set(C.V3_ADDITIONAL_RECOMPUTE_EVIDENCE_KEYS)
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 recompute evidence field drift"
        )
    external = validate_external_v1_v2_custody_receipt(
        evidence["external_historical_custody_receipt"]
    )
    external_bytes = C.canonical_json_bytes(external)
    external_binding = {
        "path": str(C.HISTORICAL_CUSTODY_RECEIPT_PATH),
        "bytes": len(external_bytes),
        "sha256": hashlib.sha256(external_bytes).hexdigest(),
    }
    external_projection_sha = historical_custody_projection_sha256(external)
    custody = validate_v1_v2_custody_and_nonreuse(
        evidence["v1_v2_custody_and_nonreuse"]
    )
    if (
        custody["external_historical_custody_receipt_binding"] != external_binding
        or custody["external_historical_custody_projection_sha256"]
        != external_projection_sha
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "internal/external historical custody cross-link drift"
        )
    invariance = validate_scientific_invariance_receipt(
        evidence["scientific_invariance_receipt"]
    )
    first_eight = validate_first_eight_reproduction_evidence(
        evidence["v1_v2_v3_first_eight_reproduction"],
        evidence["first_eight_behavioural_probe_arrays"],
    )
    if first_eight["historical_custody_receipt_binding"] != external_binding:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight/external historical custody binding drift"
        )
    equivalence = validate_snapshot_equivalence_evidence(
        evidence["snapshot_equivalence_index"],
        evidence["snapshot_behavioural_probe_arrays"],
        evidence["first_eight_behavioural_probe_arrays"],
    )
    if (
        C.canonical_json_bytes(equivalence["records"][:8])
        != C.canonical_json_bytes(first_eight["rows"])
        or equivalence["first_eight_material_probe_binding"]
        != first_eight["first_eight_behavioural_probe_material_binding"]
        or equivalence["first_eight_material_probe_projection_sha256"]
        != first_eight["first_eight_behavioural_probe_projection_sha256"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "first-eight/final equivalence cross-link drift"
        )
    inspections = validate_npz_inspections(evidence["npz_inspections"])
    behavioural_inspection = inspections["snapshot_behavioural_probes.npz"]
    if equivalence["behavioural_probe_npz_binding"] != {
        "path": "snapshot_behavioural_probes.npz",
        "bytes": behavioural_inspection["bytes"],
        "sha256": behavioural_inspection["sha256"],
    }:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "equivalence/NPZ inspection file binding drift"
        )
    inherited_evidence = {
        key: copy.deepcopy(evidence[key]) for key in inherited_keys
    }
    inherited_evidence["npz_inspections"] = [
        copy.deepcopy(item) for item in evidence["npz_inspections"]
        if item.get("path") != "snapshot_behavioural_probes.npz"
    ]
    try:
        result = V2M.recompute_metrics(
            _map_identity(inherited_evidence, to_v2=True)
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            f"frozen V2 metric recomputation failed: {exc}"
        ) from exc
    mapped = _map_identity(result, to_v2=False)
    mapped.pop("content_digest", None)
    mapped["schema"] = "physical_graph_edge_handoff_qualification_v3.metrics.v1"
    mapped["experiment_id"] = C.EXPERIMENT_ID
    records = equivalence["records"]
    historical_records = records[:8]
    mapped["v3_snapshot_qualification"] = {
        "serializer_authority_content_digest": C.SEMANTIC_SERIALIZER_AUTHORITY[
            "content_digest"
        ],
        "behavioural_probe_authority_content_digest": C.BEHAVIOURAL_PROBE_AUTHORITY[
            "content_digest"
        ],
        "qualification_shard_authority_content_digest": C.QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY[
            "content_digest"
        ],
        "snapshot_identity_fields": sorted(C.SNAPSHOT_IDENTITY_FIELDS),
        "snapshot_identity_contract": {
            "artifact_file_sha256": (
                "descriptive SHA-256 of exact protocol-4 transport artifact bytes; "
                "equality is neither required nor sufficient"
            ),
            "snapshot_semantic_digest_v1": (
                "SHA-256 of frozen canonical semantic snapshot bytes"
            ),
            "snapshot_behavioural_digest_v1": (
                "SHA-256 of the designated trial-0 deterministic restoration/probe "
                "manifest; trace tolerance comparison remains the gate"
            ),
        },
        "semantic_regression_results": copy.deepcopy(
            invariance["semantic_regression_results"]
        ),
        "qualification_state_count": 256,
        "equivalence_row_count": 256,
        "historical_row_count": 8,
        "v3_only_row_count": 248,
        "historical_semantic_pair_pass_count": sum(
            int(
                item["v1_v2_semantic_equal"]
                and item["v1_v3_semantic_equal"]
                and item["v2_v3_semantic_equal"]
                and item["semantic_manifests_equal"]
            )
            for item in historical_records
        ),
        "historical_behavioural_pair_pass_count": sum(
            int(
                item["v1_v2_behavioural_equal"]
                and item["v1_v3_behavioural_equal"]
                and item["v2_v3_behavioural_equal"]
            )
            for item in historical_records
        ),
        "v3_restore_pair_pass_count": sum(
            int(item["v3_restore_trials_equal"]) for item in records
        ),
        "v1_v2_raw_artifact_equal_count": sum(
            int(item["v1_v2_artifact_file_sha256_equal"])
            for item in historical_records
        ),
        "v1_v3_raw_artifact_equal_count": sum(
            int(item["v1_v3_artifact_file_sha256_equal"])
            for item in historical_records
        ),
        "v2_v3_raw_artifact_equal_count": sum(
            int(item["v2_v3_artifact_file_sha256_equal"])
            for item in historical_records
        ),
        "raw_artifact_equality_is_descriptive": True,
        "behavioural_probe_trace_count": C.BEHAVIOURAL_PROBE_TRACE_COUNT,
        "behavioural_probe_total_samples": C.BEHAVIOURAL_PROBE_TOTAL_SAMPLES,
        "controller_observation_persisted": True,
        "raw_policy_output_persisted": True,
        "first_eight_prefix_exact": equivalence["first_eight_prefix_exact"],
        "first_eight_gate_pass": first_eight["pass"],
        "all_pass": bool(
            invariance["pass"] and first_eight["pass"] and equivalence["pass"]
        ),
    }
    nonreuse = external["nonreuse"]
    terminal = external["v2_terminal_evidence"]
    mapped["v3_historical_custody"] = {
        "external_receipt_binding": external_binding,
        "external_receipt_projection_sha256": external_projection_sha,
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": C.V2_SOURCE_FREEZE_COMMIT,
        "v2_terminal_disposition": terminal["disposition"],
        "v2_terminal_official_leaf_count": terminal["official_leaf_count"],
        "v2_terminal_material_pair_count": terminal["material_pair_count"],
        "v1_v2_semantic_equal_count": nonreuse[
            "v1_v2_snapshot_semantic_digest_v1_equal_count"
        ],
        "v1_v2_raw_artifact_equal_count": nonreuse[
            "v1_v2_artifact_file_sha256_equal_count"
        ],
        "raw_artifact_inequality_is_descriptive": nonreuse[
            "raw_artifact_inequality_is_descriptive"
        ],
        "historical_deserializer_invocations": nonreuse[
            "historical_deserializer_invocations"
        ],
        "model_initializations": nonreuse["model_initializations"],
        "training_runs": nonreuse["training_runs"],
        "simulator_initializations": nonreuse["simulator_initializations"],
        "runner_calls": nonreuse["runner_calls"],
        "encoder_calls": nonreuse["encoder_calls"],
        "ranker_calls": nonreuse["ranker_calls"],
        "historical_roots_shared_inode_count": nonreuse[
            "historical_roots_shared_inode_count"
        ],
        "historical_runtime_artifact_or_shard_reused": nonreuse[
            "historical_runtime_artifact_or_shard_reused"
        ],
        "all_roots_unchanged": all(
            external["immutability"][field]
            for field in (
                "v1_official_root_unchanged_during_audit",
                "v1_material_root_unchanged_during_audit",
                "v2_official_root_unchanged_during_audit",
                "v2_material_root_unchanged_during_audit",
            )
        ),
        "pass": custody["pass"],
    }
    if set(mapped["v3_snapshot_qualification"]) != set(
        C.V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS
    ) or set(mapped["v3_historical_custody"]) != set(
        C.V3_HISTORICAL_CUSTODY_METRIC_FIELDS
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 metric projection field drift"
        )
    return C.attach_content_digest(mapped)


def build_result_document(
    recomputed_metrics: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    *,
    metrics_sha256: str,
    independent_reducer_receipt_sha256: str,
    runtime_seconds: float,
    scientific_storage_bytes: int,
) -> dict[str, Any]:
    """Build the exact V3 success-only result.json document."""

    metrics = C.validate_content_digest(recomputed_metrics)
    runtime = C.validate_runtime_contract(runtime_contract)
    _sha(metrics_sha256, "metrics SHA-256")
    _sha(independent_reducer_receipt_sha256, "reducer receipt SHA-256")
    if (
        metrics.get("schema")
        != "physical_graph_edge_handoff_qualification_v3.metrics.v1"
        or metrics.get("experiment_id") != C.EXPERIMENT_ID
        or set(metrics.get("v3_snapshot_qualification", {}))
        != set(C.V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS)
        or set(metrics.get("v3_historical_custody", {}))
        != set(C.V3_HISTORICAL_CUSTODY_METRIC_FIELDS)
        or isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) < 0.0
        or not isinstance(scientific_storage_bytes, int)
        or isinstance(scientific_storage_bytes, bool)
        or scientific_storage_bytes <= 0
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result input projection drift"
        )
    context = runtime["scientific_contract"].get("v2_context_binding")
    if not isinstance(context, Mapping) or not isinstance(
        context.get("result_commit"), str
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result predecessor context drift"
        )
    return C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v3.result.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "source_commit": runtime["source_freeze_commit"],
            "source_baseline_commit": C.SOURCE_BASELINE_COMMIT,
            "predecessor_result_commit": context["result_commit"],
            "development_only": True,
            "primary_classification": metrics["primary_classification"],
            "secondary_classifications": copy.deepcopy(
                metrics["secondary_classifications"]
            ),
            "next_experiment": metrics["next_experiment"],
            "selected_target_id": metrics["development"]["selected_target_id"],
            "evidence_counts": copy.deepcopy(metrics["evidence_counts"]),
            "panel_metrics": copy.deepcopy(metrics["panel"]),
            "development_metrics": copy.deepcopy(metrics["development"]),
            "heldout_metrics": copy.deepcopy(metrics["heldout"]),
            "repeatability": copy.deepcopy(metrics["repeatability"]),
            "command_tracking": copy.deepcopy(metrics["command_tracking"]),
            "runtime_environments": copy.deepcopy(metrics["runtime_environments"]),
            "stratified_metrics": copy.deepcopy(metrics["stratified"]),
            "gate": copy.deepcopy(metrics["gate"]),
            "component_failures": copy.deepcopy(metrics["component_failures"]),
            "v3_snapshot_qualification": copy.deepcopy(
                metrics["v3_snapshot_qualification"]
            ),
            "v3_historical_custody": copy.deepcopy(
                metrics["v3_historical_custody"]
            ),
            "metrics_sha256": metrics_sha256,
            "independent_reducer_receipt_sha256": (
                independent_reducer_receipt_sha256
            ),
            "runtime_seconds": float(runtime_seconds),
            "scientific_storage_bytes": scientific_storage_bytes,
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
    )


def validate_result_document(
    value: Any,
    recomputed_metrics: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    *,
    metrics_sha256: str,
    independent_reducer_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.V3_RESULT_FIELDS, "V3 result.json")
    C.validate_content_digest(row)
    observed_receipt_sha = row["independent_reducer_receipt_sha256"]
    if (
        independent_reducer_receipt_sha256 is not None
        and observed_receipt_sha != independent_reducer_receipt_sha256
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result reducer receipt SHA drift"
        )
    expected = build_result_document(
        recomputed_metrics,
        runtime_contract,
        metrics_sha256=metrics_sha256,
        independent_reducer_receipt_sha256=observed_receipt_sha,
        runtime_seconds=row["runtime_seconds"],
        scientific_storage_bytes=row["scientific_storage_bytes"],
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result.json differs from reduced evidence"
        )
    return row


def build_result_report(
    result_document: Mapping[str, Any], recomputed_metrics: Mapping[str, Any]
) -> str:
    """Rebuild exact result.md bytes from the validated result and metrics."""

    result = _mapping(result_document, C.V3_RESULT_FIELDS, "V3 result report input")
    C.validate_content_digest(result)
    metrics = C.validate_content_digest(recomputed_metrics)
    if (
        result["experiment_id"] != C.EXPERIMENT_ID
        or metrics.get("experiment_id") != C.EXPERIMENT_ID
        or result["primary_classification"] != metrics["primary_classification"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result report identity drift"
        )
    target_lines = "\n".join(
        f"- {item['target_id']}: {json.dumps(item, sort_keys=True)}"
        for item in metrics["development"]["target_summaries"]
    )
    condition_lines = "\n".join(
        f"- {item['condition_id']}: {json.dumps(item, sort_keys=True)}"
        for item in metrics["heldout"]["condition_summaries"]
    )
    sections = [
        (
            "Disposition",
            f"Primary classification: {metrics['primary_classification']}\n\n"
            "Secondary classifications: "
            f"{', '.join(metrics['secondary_classifications']) or 'none'}\n\n"
            f"Selected target: {metrics['development']['selected_target_id']}\n\n"
            f"Handoff gate passed: {str(bool(metrics['gate']['passed'])).lower()}\n\n"
            f"Next experiment: {metrics['next_experiment']}",
        ),
        (
            "V1/V2 historical custody and V2 terminal diagnosis",
            json.dumps(metrics["v3_historical_custody"], sort_keys=True),
        ),
        (
            "Snapshot identity contract and serializer fixtures",
            json.dumps(
                {
                    "snapshot_identity_contract": metrics[
                        "v3_snapshot_qualification"
                    ]["snapshot_identity_contract"],
                    "semantic_regression_results": metrics[
                        "v3_snapshot_qualification"
                    ]["semantic_regression_results"],
                },
                sort_keys=True,
            ),
        ),
        (
            "V3 semantic and behavioural qualification",
            json.dumps(metrics["v3_snapshot_qualification"], sort_keys=True),
        ),
        (
            "Physical evidence and panel",
            f"Evidence counts: {json.dumps(metrics['evidence_counts'], sort_keys=True)}\n\n"
            f"Panel and coverage: {json.dumps(metrics['panel'], sort_keys=True)}",
        ),
        ("Development target selection", target_lines),
        ("Held-out conditions", condition_lines),
        (
            "Reset, repeat, and controller qualification",
            f"Repeatability: {json.dumps(metrics['repeatability'], sort_keys=True)}\n\n"
            f"Command tracking: {json.dumps(metrics['command_tracking'], sort_keys=True)}",
        ),
        (
            "Runtime, storage, and training",
            f"Runtime environments: {json.dumps(metrics['runtime_environments'], sort_keys=True)}\n\n"
            f"Stratified outcomes: {json.dumps(metrics['stratified'], sort_keys=True)}\n\n"
            f"Runtime seconds: {result['runtime_seconds']:.6f}; scientific storage bytes: "
            f"{result['scientific_storage_bytes']}; models trained: 0; prohibited components: [].",
        ),
        (
            "Claims boundary",
            "This development-only physical handoff qualification preserves every frozen "
            "V2/V1 scientific formula, gate, threshold, class, precedence rule, model, "
            "controller, geometry, seed, and logical identity. V3 adds independently "
            "reproducible semantic and behavioural snapshot qualification. Historical V1/V2 "
            "artifacts were used read-only for custody and the authorized first-eight "
            "semantic/behavioural comparison only; none was copied, hard-linked, adopted as "
            "V3 runtime evidence, or used for training. Contact remains a descriptive "
            "simulated proxy and does not establish material safety.",
        ),
    ]
    if tuple(title for title, _body in sections) != C.V3_RESULT_REPORT_SECTION_ORDER:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 report section order drift"
        )
    return (
        f"# {C.EXPERIMENT_ID}\n\n"
        + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
        + "\n"
    )


def validate_result_report(
    value: Any, result_document: Mapping[str, Any], recomputed_metrics: Mapping[str, Any]
) -> str:
    if not isinstance(value, str):
        raise PhysicalGraphEdgeHandoffV3MetricsError("V3 result report is not text")
    expected = build_result_report(result_document, recomputed_metrics)
    if value != expected:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result.md differs from exact reconstruction"
        )
    return value


def _validate_scientific_publication_bindings(
    value: Any,
    *,
    recomputed_metrics: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int, str]:
    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 scientific publication bindings are not a mapping"
        )
    expected_leaves = set(C.SUCCESS_OUTPUT_LEAVES) - {
        "result.json",
        "result.md",
        "file_hashes.json",
    }
    if set(value) != expected_leaves:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 scientific publication binding inventory drift"
        )
    bindings: dict[str, dict[str, Any]] = {}
    for leaf in sorted(expected_leaves):
        binding = _file_binding(value[leaf], f"V3 scientific binding {leaf}")
        assert binding is not None
        if binding["path"] != leaf:
            raise PhysicalGraphEdgeHandoffV3MetricsError(
                f"V3 scientific binding path drift: {leaf}"
            )
        bindings[leaf] = binding
    metrics_bytes = C.canonical_json_bytes(recomputed_metrics)
    metrics_sha256 = hashlib.sha256(metrics_bytes).hexdigest()
    if bindings["metrics.json"] != {
        "path": "metrics.json",
        "bytes": len(metrics_bytes),
        "sha256": metrics_sha256,
    }:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 metrics scientific binding differs from canonical recomputation"
        )
    contract_bytes = C.canonical_json_bytes(runtime_contract)
    if bindings["contract.json"] != {
        "path": "contract.json",
        "bytes": len(contract_bytes),
        "sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }:
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 runtime contract scientific binding differs from canonical bytes"
        )
    return (
        bindings,
        sum(binding["bytes"] for binding in bindings.values()),
        metrics_sha256,
    )


def _validate_snapshot_equivalence_publication(
    value: Any,
    *,
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    snapshot_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(
        value,
        C.V3_SNAPSHOT_EQUIVALENCE_PUBLICATION_FIELDS,
        "V3 snapshot equivalence publication projection",
    )
    for field in (
        "snapshot_equivalence_index_binding",
        "snapshot_behavioural_probe_binding",
        "first_eight_material_binding",
    ):
        binding = _file_binding(row[field], f"V3 snapshot projection {field}")
        assert binding is not None
        row[field] = binding
    for field in (
        "snapshot_behavioural_probe_projection_sha256",
        "first_eight_projection_sha256",
    ):
        _sha(row[field], f"V3 snapshot projection {field}")
    exact_values = {
        "terminal": False,
        "first_eight_row_count": 8,
        "complete_row_count": C.BEHAVIOURAL_PROBE_POOL_COUNT,
        "historical_row_count": 8,
        "v3_only_row_count": C.BEHAVIOURAL_PROBE_POOL_COUNT - 8,
        "first_eight_receipt_exact_rebuild": True,
        "all_semantic_and_behavioural_rows_pass": True,
        "scientific_result_authorized": True,
    }
    if any(row[field] != expected for field, expected in exact_values.items()):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 snapshot equivalence publication outcome drift"
        )
    if (
        row["snapshot_equivalence_index_binding"]
        != scientific_bindings["snapshot_equivalence_index.json"]
        or row["snapshot_behavioural_probe_binding"]
        != scientific_bindings["snapshot_behavioural_probes.npz"]
        or row["first_eight_material_binding"]["path"]
        != C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["first_eight_material_probe"][
            "path"
        ]
        or snapshot_metrics.get("qualification_state_count")
        != C.BEHAVIOURAL_PROBE_POOL_COUNT
        or snapshot_metrics.get("equivalence_row_count")
        != C.BEHAVIOURAL_PROBE_POOL_COUNT
        or snapshot_metrics.get("historical_row_count") != 8
        or snapshot_metrics.get("v3_only_row_count")
        != C.BEHAVIOURAL_PROBE_POOL_COUNT - 8
        or snapshot_metrics.get("first_eight_prefix_exact") is not True
        or snapshot_metrics.get("first_eight_gate_pass") is not True
        or snapshot_metrics.get("all_pass") is not True
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 snapshot equivalence publication cross-link drift"
        )
    return row


def build_result_publication_projection(
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    *,
    independent_reducer_receipt_sha256: str,
    snapshot_equivalence: Mapping[str, Any],
    historical_custody_receipt_binding: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the exact persisted result plus its report-rebuild projection."""

    metrics = C.validate_content_digest(recomputed_metrics)
    runtime = C.validate_runtime_contract(runtime_contract)
    reducer_sha = _sha(
        independent_reducer_receipt_sha256,
        "V3 independent reducer receipt SHA-256",
    )
    bindings, storage_bytes, metrics_sha = _validate_scientific_publication_bindings(
        scientific_bindings,
        recomputed_metrics=metrics,
        runtime_contract=runtime,
    )
    historical = _file_binding(
        historical_custody_receipt_binding,
        "V3 publication historical custody binding",
    )
    assert historical is not None
    if (
        historical["path"] != str(C.HISTORICAL_CUSTODY_RECEIPT_PATH)
        or runtime.get("historical_custody_receipt_binding") != historical
        or metrics.get("v3_historical_custody", {}).get(
            "external_receipt_binding"
        )
        != historical
        or metrics.get("v3_historical_custody", {}).get("pass") is not True
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 publication historical custody cross-link drift"
        )
    _validate_snapshot_equivalence_publication(
        snapshot_equivalence,
        scientific_bindings=bindings,
        snapshot_metrics=metrics.get("v3_snapshot_qualification", {}),
    )
    result = build_result_document(
        metrics,
        runtime,
        metrics_sha256=metrics_sha,
        independent_reducer_receipt_sha256=reducer_sha,
        runtime_seconds=runtime_seconds,
        scientific_storage_bytes=storage_bytes,
    )
    return C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3."
                "result_publication_projection.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "result_document": result,
            "recomputed_metrics": copy.deepcopy(metrics),
        }
    )


def validate_result_publication_projection(
    value: Any,
    *,
    recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    independent_reducer_receipt_sha256: str,
    snapshot_equivalence: Mapping[str, Any],
    historical_custody_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate result.json and return the exact report-rebuild projection."""

    result = _mapping(value, C.V3_RESULT_FIELDS, "V3 result publication")
    C.validate_content_digest(result)
    expected = build_result_publication_projection(
        recomputed_metrics,
        scientific_bindings,
        runtime_contract,
        independent_reducer_receipt_sha256=(
            independent_reducer_receipt_sha256
        ),
        snapshot_equivalence=snapshot_equivalence,
        historical_custody_receipt_binding=historical_custody_receipt_binding,
        runtime_seconds=result["runtime_seconds"],
    )
    if C.canonical_json_bytes(result) != C.canonical_json_bytes(
        expected["result_document"]
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result publication differs from exact reduced projection"
        )
    return expected


def build_result_report_bytes(value: Any) -> bytes:
    """Return the exact UTF-8, LF-terminated result.md bytes."""

    projection = _mapping(
        value,
        C.V3_RESULT_PUBLICATION_PROJECTION_FIELDS,
        "V3 result publication projection",
    )
    C.validate_content_digest(projection)
    if (
        projection["schema"]
        != (
            "physical_graph_edge_handoff_qualification_v3."
            "result_publication_projection.v1"
        )
        or projection["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalGraphEdgeHandoffV3MetricsError(
            "V3 result publication projection identity drift"
        )
    result = _mapping(
        projection["result_document"],
        C.V3_RESULT_FIELDS,
        "V3 result publication result",
    )
    C.validate_content_digest(result)
    metrics = C.validate_content_digest(projection["recomputed_metrics"])
    return build_result_report(result, metrics).encode("utf-8")


def reducer_authority() -> dict[str, Any]:
    base = _map_identity(V2M.reducer_authority(), to_v2=False)
    base.pop("content_digest", None)
    base["schema"] = "physical_graph_edge_handoff_qualification_v3.reducer_authority.v1"
    base["experiment_id"] = C.EXPERIMENT_ID
    base["runtime_paths"] = copy.deepcopy(C.RUNTIME_OUTPUT_PATHS)
    base["successful_output_leaf_count"] = C.OUTPUT_LEAF_COUNT
    base["successful_output_leaves"] = list(C.SUCCESS_OUTPUT_LEAVES)
    base["reproduction_mismatch_output_leaves"] = list(C.REPRODUCTION_MISMATCH_LEAVES)
    base["persisted_array_hash_authority"] = copy.deepcopy(C.PERSISTED_ARRAY_HASH_AUTHORITY)
    base["snapshot_semantic_serializer_authority"] = copy.deepcopy(C.SEMANTIC_SERIALIZER_AUTHORITY)
    base["historical_snapshot_deserializer_authority"] = copy.deepcopy(C.HISTORICAL_DESERIALIZER_AUTHORITY)
    base["snapshot_behavioural_probe_authority"] = copy.deepcopy(C.BEHAVIOURAL_PROBE_AUTHORITY)
    base["qualification_shard_augmentation_authority"] = copy.deepcopy(
        C.QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY
    )
    base["v1_v2_custody_and_nonreuse_authority"] = copy.deepcopy(C.V1_V2_CUSTODY_AND_NONREUSE_AUTHORITY)
    base["first_eight_reproduction_authority"] = copy.deepcopy(C.FIRST_EIGHT_REPRODUCTION_AUTHORITY)
    base["result_publication_authority"] = copy.deepcopy(
        C.RESULT_PUBLICATION_AUTHORITY
    )
    base["v3_additional_recompute_evidence_keys"] = sorted(
        C.V3_ADDITIONAL_RECOMPUTE_EVIDENCE_KEYS
    )
    base["v3_recompute_evidence_keys"] = sorted(
        set(V2M.V1M.EVIDENCE_KEYS)
        | set(C.V3_ADDITIONAL_RECOMPUTE_EVIDENCE_KEYS)
    )
    base["external_historical_custody_receipt_authority"] = {
        "fields": sorted(C.EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS),
        "root_fields": sorted(C.HISTORICAL_CUSTODY_ROOT_FIELDS),
        "file_fields": sorted(C.HISTORICAL_CUSTODY_FILE_FIELDS),
        "directory_fields": sorted(C.HISTORICAL_CUSTODY_DIRECTORY_FIELDS),
        "version_fields": sorted(C.HISTORICAL_CUSTODY_VERSION_FIELDS),
        "pair_fields": sorted(C.HISTORICAL_CUSTODY_PAIR_FIELDS),
        "v2_terminal_fields": sorted(C.HISTORICAL_CUSTODY_V2_TERMINAL_FIELDS),
        "scientific_boundary_fields": sorted(
            C.HISTORICAL_CUSTODY_SCIENTIFIC_BOUNDARY_FIELDS
        ),
        "repository_fields": sorted(C.HISTORICAL_CUSTODY_REPOSITORY_FIELDS),
        "immutability_fields": sorted(C.HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS),
        "nonreuse_fields": sorted(C.HISTORICAL_CUSTODY_NONREUSE_FIELDS),
    }
    base["new_documents"] = {
        "v1_v2_custody_and_nonreuse": {
            "path": "v1_v2_custody_and_nonreuse.json",
            "fields": sorted(C.V1_V2_CUSTODY_AND_NONREUSE_RECEIPT_FIELDS),
            "content_digest_forbidden": True,
        },
        "scientific_invariance_receipt": {
            "path": "scientific_invariance_receipt.json",
            "fields": sorted(C.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS),
            "content_digest_forbidden": True,
        },
        "v1_v2_v3_first_eight_reproduction": {
            "path": "v1_v2_v3_first_eight_reproduction.json",
            "fields": sorted(C.FIRST_EIGHT_REPRODUCTION_FIELDS),
            "row_fields": sorted(C.SNAPSHOT_EQUIVALENCE_RECORD_FIELDS),
            "row_count": 8,
            "content_digest_forbidden": True,
        },
        "snapshot_equivalence_index": {
            "path": "snapshot_equivalence_index.json",
            "fields": sorted(C.SNAPSHOT_EQUIVALENCE_INDEX_FIELDS),
            "row_fields": sorted(C.SNAPSHOT_EQUIVALENCE_RECORD_FIELDS),
            "row_count": 256,
            "historical_row_count": 8,
            "v3_only_row_count": 248,
            "content_digest_required": True,
        },
        "snapshot_behavioural_probes": {
            "path": "snapshot_behavioural_probes.npz",
            "members": copy.deepcopy(C.BEHAVIOURAL_PROBE_NPZ_AUTHORITY),
            "trace_count": C.BEHAVIOURAL_PROBE_TRACE_COUNT,
            "physics_samples_per_trace": C.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES,
            "total_physics_samples": C.BEHAVIOURAL_PROBE_TOTAL_SAMPLES,
            "first_eight_material_members": copy.deepcopy(
                C.FIRST_EIGHT_BEHAVIOURAL_PROBE_NPZ_AUTHORITY
            ),
            "first_eight_trace_count": 48,
            "first_eight_physics_samples": 36000,
        },
    }
    return C.attach_content_digest(base)


def scientific_invariance_authority() -> dict[str, Any]:
    return copy.deepcopy(C.SCIENTIFIC_INVARIANCE_AUTHORITY)


def behavioural_probe_authority() -> dict[str, Any]:
    return copy.deepcopy(C.BEHAVIOURAL_PROBE_AUTHORITY)


def qualification_shard_augmentation_authority() -> dict[str, Any]:
    return copy.deepcopy(C.QUALIFICATION_SHARD_AUGMENTATION_AUTHORITY)


def result_publication_authority() -> dict[str, Any]:
    return copy.deepcopy(C.RESULT_PUBLICATION_AUTHORITY)


def v1_v2_custody_and_nonreuse_authority() -> dict[str, Any]:
    return copy.deepcopy(C.V1_V2_CUSTODY_AND_NONREUSE_AUTHORITY)


def first_eight_reproduction_authority() -> dict[str, Any]:
    return copy.deepcopy(C.FIRST_EIGHT_REPRODUCTION_AUTHORITY)


__all__ = [
    "PhysicalGraphEdgeHandoffV3MetricsError",
    "authorizes_full_v3_collection",
    "behavioural_probe_authority",
    "behavioural_probe_npz_projection_sha256",
    "behavioural_probe_trials_from_arrays",
    "build_first_eight_reproduction",
    "build_persisted_array_evidence",
    "build_qualification_shard_augmentation",
    "build_result_document",
    "build_result_publication_projection",
    "build_result_report",
    "build_result_report_bytes",
    "build_scientific_invariance_receipt",
    "build_semantic_regression_results",
    "build_snapshot_equivalence_record",
    "build_snapshot_equivalence_index",
    "build_v1_v2_custody_and_nonreuse",
    "canonical_semantic_snapshot",
    "classify_physical_handoff_aggregates",
    "compare_behavioural_probe_traces",
    "derive_candidate_port_metrics",
    "external_artifact_bindings",
    "first_eight_reproduction_authority",
    "first_registered_port_crossing",
    "historical_custody_projection",
    "historical_custody_projection_sha256",
    "historical_snapshot_deserializer_authority",
    "panel_manifest_authority",
    "persisted_array_bytes_sha256",
    "persisted_array_manifest_row",
    "physical_trace_reduction_authority",
    "point_in_polygon_inclusive",
    "predecessor_context_binding",
    "project_v2_evidence_to_v3",
    "project_v3_evidence_to_v2",
    "qualification_shard_augmentation_authority",
    "recompute_metrics",
    "reducer_authority",
    "result_publication_authority",
    "runtime_environment_sha256",
    "scientific_invariance_authority",
    "semantic_snapshot_serializer_authority",
    "semantic_snapshot_sha256",
    "snapshot_behavioural_digest",
    "snapshot_semantic_evidence",
    "transverse_port_crossing",
    "v1_v2_custody_and_nonreuse_authority",
    "validate_behavioural_probe_npz_arrays",
    "validate_candidate_fanout_rows",
    "validate_development_target_selection",
    "validate_edge_port_index",
    "validate_external_v1_v2_custody_receipt",
    "validate_final_behavioural_probe_assembly",
    "validate_first_eight_behavioural_probe_npz_arrays",
    "validate_first_eight_reproduction",
    "validate_first_eight_reproduction_evidence",
    "validate_graph_manifest",
    "validate_heldout_ranker_score_rows",
    "validate_latent_index",
    "validate_npz_archive_comment",
    "validate_npz_inspections",
    "validate_panel_manifest",
    "validate_persisted_array_evidence",
    "validate_physical_runtime_environment",
    "validate_pixel_index",
    "validate_qualification_shard_augmentation",
    "validate_repeated_execution_rows",
    "validate_result_document",
    "validate_result_publication_projection",
    "validate_result_report",
    "validate_scientific_invariance_receipt",
    "validate_snapshot_equivalence_index",
    "validate_snapshot_equivalence_evidence",
    "validate_snapshot_previous_applied_command_binding",
    "validate_snapshot_semantic_bytes",
    "validate_snapshot_semantic_evidence",
    "validate_split_manifest",
    "validate_state_snapshot_index",
    "validate_state_snapshot_previous_applied_command_hashes",
    "validate_teacher_trace_index",
    "validate_v1_v2_custody_and_nonreuse",
    "validate_visual_runtime_environment",
    "validate_waypoint_contracts",
]
