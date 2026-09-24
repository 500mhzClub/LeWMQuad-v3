"""Pure V4 state-disposition, panel, metric, and publication authority.

No simulator, model, encoder, ranker, or historical snapshot deserializer is
imported here.  An adequate panel is reduced with the exact inherited V2/V1
scientific formulas.  An inadequate complete panel is itself a scientific
terminal and never opens reset, fanout, encoder, ranker, or held-out outcomes.
The V4-only planar projection helper spells out the frozen NumPy-2 binary64
accumulation so producer and independent reducer obtain identical bytes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from fractions import Fraction
import hashlib
import json
import math
import types
from typing import Any

from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as V2M
from lewm.safety import physical_graph_edge_handoff_qualification_v3_metrics as V3M
from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as C

V1M = V2M.V1M


class PhysicalGraphEdgeHandoffV4MetricsError(ValueError):
    """Raised when V4 evidence is incomplete, inconsistent, or noncanonical."""


class PhysicalGraphEdgeHandoffV4HardStop(PhysicalGraphEdgeHandoffV4MetricsError):
    """Raised for the exhaustive technical hard-stop set."""


_V3_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v3"
_V4_SCHEMA_TOKEN = "physical_graph_edge_handoff_qualification_v4"


def _mapping(value: Any, fields: set[str] | frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} field drift")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} is not a sequence")
    return [copy.deepcopy(item) for item in value]


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} is not SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} is not a lowercase Git commit"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} is not a nonnegative integer")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} is not bool")
    return value


def _finite_binary64(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} is not finite binary64 numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} is not finite binary64 numeric"
        )
    return result


def frozen_binary64_fma(
    multiplicand: Any, multiplier: Any, addend: Any,
) -> float:
    """Return deterministic correctly-rounded binary64 ``a*b+c``.

    Exact rational evaluation avoids depending on whether a NumPy build uses
    a fused vector kernel for a two-element dot product.  Conversion back to
    ``float`` performs the one required round-to-nearest-even operation.
    """

    left = _finite_binary64(multiplicand, "FMA multiplicand")
    right = _finite_binary64(multiplier, "FMA multiplier")
    tail = _finite_binary64(addend, "FMA addend")
    exact = (
        Fraction.from_float(left) * Fraction.from_float(right)
        + Fraction.from_float(tail)
    )
    if exact == 0:
        return 0.0
    try:
        result = float(exact)
    except OverflowError as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "FMA result is outside finite binary64"
        ) from exc
    if not math.isfinite(result):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "FMA result is outside finite binary64"
        )
    return result


def _binary64_vector(value: Any, length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} is not a numeric vector"
        )
    try:
        values = list(value)
    except TypeError as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} is not a numeric vector"
        ) from exc
    if len(values) != length:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} length drift")
    return [
        _finite_binary64(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]


def canonical_v1_planar_segment_projection(
    point_world_xy: Any, opening_segment_world: Any,
) -> dict[str, Any]:
    """Reproduce the frozen NumPy-2 two-vector projection byte-exactly."""

    point = _binary64_vector(point_world_xy, 2, "projection point")
    if isinstance(opening_segment_world, (str, bytes, bytearray, Mapping)):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "projection opening segment is not a two-row sequence"
        )
    try:
        rows = list(opening_segment_world)
    except TypeError as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "projection opening segment is not a two-row sequence"
        ) from exc
    if len(rows) != 2:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "projection opening segment row count drift"
        )
    start = _binary64_vector(rows[0], 2, "projection opening start")
    stop = _binary64_vector(rows[1], 2, "projection opening stop")
    delta_x = stop[0] - start[0]
    delta_y = stop[1] - start[1]
    norm_squared = frozen_binary64_fma(
        delta_y, delta_y, delta_x * delta_x,
    )
    if norm_squared <= 1.0e-24:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "projection opening segment is degenerate"
        )
    width = math.sqrt(norm_squared)
    tangent_x = delta_x / width
    tangent_y = delta_y / width
    midpoint_x = (start[0] + stop[0]) / 2.0
    midpoint_y = (start[1] + stop[1]) / 2.0
    relative_x = point[0] - midpoint_x
    relative_y = point[1] - midpoint_y
    lateral = frozen_binary64_fma(
        relative_y, tangent_y, relative_x * tangent_x,
    )
    result = {
        "segment_width_m": width,
        "midpoint_world_xy": [midpoint_x, midpoint_y],
        "unit_tangent_world_xy": [tangent_x, tangent_y],
        "lateral_coordinate_m": lateral,
    }
    if tuple(result) != C.V4_BINARY64_PLANAR_SEGMENT_PROJECTION_FIELDS:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "binary64 planar projection field drift"
        )
    return result


def _terminal_nonfinite_masks(
    arrays: Mapping[str, Any], flags: Mapping[str, bool], *, label: str,
    allowed_members: Sequence[str], require_trace_axis: bool,
) -> dict[str, Any]:
    """Validate the exact V4-only inclusive terminal nonfinite boundary."""

    import numpy as np

    allowed = set(allowed_members)
    masks: dict[str, Any] = {}
    for member, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fc":
            continue
        mask = ~np.isfinite(array)
        if not bool(mask.any()):
            continue
        if member not in allowed:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"{label}.{member} nonfinite outside terminal authority"
            )
        if require_trace_axis:
            if array.ndim < 1 or bool(mask[:-1].any()):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    f"{label}.{member} nonfinite before inclusive terminal sample"
                )
        masks[member] = np.ascontiguousarray(mask)
    base_mask = masks.get("base_pose_world")
    base_nan = bool(base_mask is not None and base_mask.any())
    if flags["nan"] is not base_nan:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} nan flag/nonfinite base-pose drift"
        )
    if masks and flags["nan"] is not True:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{label} nonfinite evidence without nan termination"
        )
    return masks


def _nonfinite_bit_patterns_equal(left: Any, right: Any) -> bool:
    """Compare exact dtype-sized IEEE bits at pairwise nonfinite cells."""

    import numpy as np

    a = np.ascontiguousarray(np.asarray(left))
    b = np.ascontiguousarray(np.asarray(right))
    if a.dtype != b.dtype or a.shape != b.shape or a.dtype.kind not in "fc":
        return False
    mask_a = ~np.isfinite(a)
    mask_b = ~np.isfinite(b)
    if not np.array_equal(mask_a, mask_b):
        return False
    if not bool(mask_a.any()):
        return True
    bits_a = a.view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
    bits_b = b.view(np.uint8).reshape(b.shape + (b.dtype.itemsize,))
    return bool(np.array_equal(bits_a[mask_a], bits_b[mask_b]))


def _finite_max_abs_error(left: Any, right: Any) -> float:
    """Maximum absolute error over cells finite in both operands."""

    import numpy as np

    a = np.asarray(left)
    b = np.asarray(right)
    finite = np.isfinite(a) & np.isfinite(b)
    if not bool(finite.any()):
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def _identity_string(value: str, *, to_v3: bool) -> str:
    if to_v3:
        if value == C.EXPERIMENT_ID:
            return C.V3_EXPERIMENT_ID
        return value.replace(_V4_SCHEMA_TOKEN, _V3_SCHEMA_TOKEN)
    if value == C.V3_EXPERIMENT_ID:
        return C.EXPERIMENT_ID
    return value.replace(_V3_SCHEMA_TOKEN, _V4_SCHEMA_TOKEN)


def _map_identity(value: Any, *, to_v3: bool) -> Any:
    """Map only official V3/V4 experiment and schema identity.

    Frozen lowercase ``pgehq-v1-*`` logical IDs, seeds, and all scientific
    values are intentionally untouched.
    """

    if isinstance(value, Mapping):
        has_digest = "content_digest" in value
        if has_digest:
            if to_v3:
                C.validate_content_digest(value)
            else:
                try:
                    C.V3.validate_content_digest(value)
                except Exception as exc:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "V3 evidence content digest drift"
                    ) from exc
        mapped = {
            str(key): _map_identity(item, to_v3=to_v3)
            for key, item in value.items() if key != "content_digest"
        }
        if has_digest:
            mapped = (C.V3.attach_content_digest if to_v3 else C.attach_content_digest)(mapped)
        return mapped
    if isinstance(value, (tuple, list)):
        return [_map_identity(item, to_v3=to_v3) for item in value]
    if isinstance(value, str):
        return _identity_string(value, to_v3=to_v3)
    return copy.deepcopy(value)


def project_v3_evidence_to_v4(value: Any) -> Any:
    """Identity-only V3→V4 publication/storage adapter."""

    return _map_identity(value, to_v3=False)


def project_v4_evidence_to_v3(value: Any) -> Any:
    """Identity-only V4→V3 adapter used by frozen predecessor validators."""

    return _map_identity(value, to_v3=True)


def _project_v4_evidence_to_v1(value: Any) -> Any:
    return V2M.project_v2_evidence_to_v1(
        V3M.project_v3_evidence_to_v2(project_v4_evidence_to_v3(value))
    )


def _project_v1_evidence_to_v4(value: Any) -> Any:
    return project_v3_evidence_to_v4(
        V3M.project_v2_evidence_to_v3(V2M.project_v1_evidence_to_v2(value))
    )


def _local_v1_call(
    function: Any, *args: Any, replacements: Mapping[str, Any] | None = None,
) -> Any:
    """Call frozen V1 bytecode with local dependency bindings only.

    This never mutates an imported module.  It lets unchanged V1 downstream
    validators and aggregation formulas consume a V4-native, already-validated
    panel/teacher context without inventing the teacher rows that V1 assumed
    every registered pool state would possess.
    """

    namespace = dict(function.__globals__)
    if replacements:
        namespace.update(replacements)
    local = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__,
        function.__closure__,
    )
    local.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    return local(*args)


def _exact_v1_dependency(expected: Any, label: str) -> Any:
    def validate(value: Any, *_args: Any, **_kwargs: Any) -> Any:
        if C.canonical_json_bytes(value) != C.canonical_json_bytes(expected):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"{label} changed inside inherited validator"
            )
        return copy.deepcopy(expected)

    return validate


def _call_v1_with_validated_dependencies(
    validator: Any, value: Any, *dependencies: Any,
    dependency_replacements: Mapping[str, Any],
) -> Any:
    v1_value = _project_v4_evidence_to_v1(value)
    v1_dependencies = [_project_v4_evidence_to_v1(item) for item in dependencies]
    replacements = {
        name: _exact_v1_dependency(_project_v4_evidence_to_v1(expected), name)
        for name, expected in dependency_replacements.items()
    }
    try:
        result = _local_v1_call(
            validator, v1_value, *v1_dependencies, replacements=replacements,
        )
        mapped = _project_v1_evidence_to_v4(result)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc
    if C.canonical_json_bytes(mapped) != C.canonical_json_bytes(value):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"{validator.__name__} altered validated V4 evidence"
        )
    return copy.deepcopy(value)


# Exact inherited helpers that are unchanged scientifically.
persisted_array_bytes_sha256 = V3M.persisted_array_bytes_sha256
persisted_array_manifest_row = V3M.persisted_array_manifest_row
derive_candidate_port_metrics = V3M.derive_candidate_port_metrics
canonical_semantic_snapshot = V3M.canonical_semantic_snapshot
semantic_snapshot_sha256 = V3M.semantic_snapshot_sha256
snapshot_semantic_evidence = V3M.snapshot_semantic_evidence
snapshot_behavioural_digest = V3M.snapshot_behavioural_digest
compare_behavioural_probe_traces = V3M.compare_behavioural_probe_traces
point_in_polygon_inclusive = V3M.point_in_polygon_inclusive
transverse_port_crossing = V3M.transverse_port_crossing
first_registered_port_crossing = V3M.first_registered_port_crossing
runtime_environment_sha256 = V3M.runtime_environment_sha256
validate_physical_runtime_environment = V3M.validate_physical_runtime_environment
_CANDIDATE_SPECS = C.build_candidate_specs()
_SPEC_INDEX_BY_ID = {
    item["candidate_spec_id"]: index for index, item in enumerate(_CANDIDATE_SPECS)
}


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(C.canonical_json_bytes(value)[:-1]).hexdigest()


def _validate_qualification_terminal_runtime(
    metadata: Mapping[str, Any], runtime_contract: Mapping[str, Any], *,
    expected_pool_index: int, allow_fake_runtime: bool,
) -> dict[str, Any]:
    """Validate one terminal's native physical and backend runtime evidence."""

    runtime = runtime_contract
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1"
        or metadata.get("experiment_id") != C.EXPERIMENT_ID
        or metadata.get("pool_index") != expected_pool_index
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification runtime terminal identity drift"
        )
    source_freeze_commit = _commit(
        metadata.get("source_freeze_commit"),
        "qualification terminal source freeze commit",
    )
    runtime_contract_content_digest = _sha(
        metadata.get("runtime_contract_content_digest"),
        "qualification terminal runtime contract digest",
    )
    if (
        source_freeze_commit == C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or runtime_contract_content_digest
        == C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification terminal binds invalidated pre-correction runtime"
        )
    if (
        source_freeze_commit != runtime["source_freeze_commit"]
        or runtime_contract_content_digest != runtime["content_digest"]
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification terminal source/runtime contract binding drift"
        )
    stage = _mapping(
        metadata.get("stage_runtime"), set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
        f"qualification[{expected_pool_index}] stage runtime",
    )
    fake = _bool(stage["fake_runtime"], "qualification fake runtime")
    if fake and not allow_fake_runtime:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "production qualification contains fake runtime"
        )
    digest = runtime_environment_sha256(stage)
    environment = {
        **copy.deepcopy(stage),
        "runtime_core_sha256": digest,
        "qualification_runtime_sha256s": [digest] * C.V3.PROSPECTIVE_POOL_COUNT,
        "selected_snapshot_runtime_sha256s": [digest] * C.V3.STATE_COUNT,
    }
    try:
        V1M.validate_physical_runtime_environment(environment)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"qualification physical runtime drift: {exc}"
        ) from exc
    backend_raw = metadata.get("backend_runtime")
    if not isinstance(backend_raw, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification backend runtime is not mapping"
        )
    backend = copy.deepcopy(dict(backend_raw))
    teacher_executed = _bool(
        metadata.get("teacher_executed"), "qualification teacher_executed"
    )
    expected_fields = set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS)
    if teacher_executed:
        expected_fields.update(C.QUALIFICATION_BACKEND_RUNTIME_TEACHER_FIELDS)
    if set(backend) != expected_fields:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification backend runtime field drift"
        )
    core = {
        field: copy.deepcopy(backend[field])
        for field in C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS
    }
    expected_core = copy.deepcopy(C.QUALIFICATION_BACKEND_RUNTIME_AUTHORITY)
    if fake:
        # Synthetic integration fixtures may name their backend, but every
        # other physical component remains exact and the name must agree with
        # the observed stage core.  Production never reaches this branch.
        expected_core["backend"] = stage["backend"]
        expected_core["policy_device"] = stage["device"]
    if core != expected_core or backend["backend"] != stage["backend"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
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
            if isinstance(identity, Mapping) else None
        )
        if (
            backend["snapshot_captured_before_teacher"] is not True
            or backend["teacher_restored_from_serialized_snapshot"] is not True
            or snapshot_sha != metadata.get("initial_decision_state_sha256")
            or snapshot_sha != artifact_sha
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "qualification teacher backend/snapshot runtime drift"
            )
    return {
        "stage_runtime": stage,
        "stage_runtime_sha256": digest,
        "backend_runtime": backend,
        "backend_runtime_sha256": _canonical_value_sha256(backend),
        "backend_runtime_core": core,
        "backend_runtime_core_sha256": _canonical_value_sha256(core),
    }


def build_qualification_runtime_environment(
    metadata_rows: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any], *,
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    """Build the exact metrics-only runtime projection from all 256 shards."""

    runtime = C.validate_runtime_contract(runtime_contract)
    supplied = _sequence(metadata_rows, "qualification runtime metadata rows")
    if len(supplied) != C.V3.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification runtime shard cardinality drift"
        )
    rows = [
        _validate_qualification_terminal_runtime(
            metadata, runtime, expected_pool_index=index,
            allow_fake_runtime=allow_fake_runtime,
        )
        for index, metadata in enumerate(supplied)
    ]
    stage = rows[0]["stage_runtime"]
    backend = rows[0]["backend_runtime_core"]
    if any(row["stage_runtime"] != stage for row in rows[1:]):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification stage runtime changed across 256 shards"
        )
    if any(row["backend_runtime_core"] != backend for row in rows[1:]):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification backend runtime core changed across 256 shards"
        )
    return C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4."
                "qualification_runtime_environment.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "external_artifact_bindings_sha256": _canonical_value_sha256(
                runtime["external_artifact_bindings"]
            ),
            "historical_custody_receipt_binding": copy.deepcopy(
                runtime["historical_custody_receipt_binding"]
            ),
            "physical_runtime_core": copy.deepcopy(stage),
            "physical_runtime_core_sha256": rows[0]["stage_runtime_sha256"],
            "backend_runtime_core": copy.deepcopy(backend),
            "backend_runtime_core_sha256": rows[0][
                "backend_runtime_core_sha256"
            ],
            "qualification_shard_count": C.V3.PROSPECTIVE_POOL_COUNT,
            "qualification_stage_runtime_sha256s": [
                row["stage_runtime_sha256"] for row in rows
            ],
            "qualification_backend_runtime_sha256s": [
                row["backend_runtime_sha256"] for row in rows
            ],
            "all_qualification_stage_runtime_cores_equal": True,
            "all_qualification_backend_runtime_cores_equal": True,
            "fake_runtime": bool(stage["fake_runtime"]),
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
    )


def validate_qualification_runtime_environment(
    value: Any, runtime_contract: Mapping[str, Any], *,
    metadata_rows: Sequence[Mapping[str, Any]] | None = None,
    material_shard_validations: Sequence[Mapping[str, Any]] | None = None,
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    """Validate the 256-shard runtime projection and optional raw cross-links."""

    row = _mapping(
        value, C.QUALIFICATION_RUNTIME_ENVIRONMENT_FIELDS,
        "qualification runtime environment",
    )
    C.validate_content_digest(row)
    runtime = C.validate_runtime_contract(runtime_contract)
    if metadata_rows is not None:
        expected = build_qualification_runtime_environment(
            metadata_rows, runtime, allow_fake_runtime=allow_fake_runtime,
        )
        if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "qualification runtime/raw terminal projection drift"
            )
        return row
    if (
        row["schema"]
        != "physical_graph_edge_handoff_qualification_v4.qualification_runtime_environment.v1"
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["source_freeze_commit"] != runtime["source_freeze_commit"]
        or row["runtime_contract_content_digest"] != runtime["content_digest"]
        or row["external_artifact_bindings_sha256"]
        != _canonical_value_sha256(runtime["external_artifact_bindings"])
        or row["historical_custody_receipt_binding"]
        != runtime["historical_custody_receipt_binding"]
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification runtime contract binding drift"
        )
    core = _mapping(
        row["physical_runtime_core"], set(C.PHYSICAL_RUNTIME_CORE_FIELDS),
        "qualification physical runtime core",
    )
    core_sha = runtime_environment_sha256(core)
    if (
        row["physical_runtime_core_sha256"] != core_sha
        or row["fake_runtime"] is not core["fake_runtime"]
        or (row["fake_runtime"] and not allow_fake_runtime)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification physical runtime digest/fake drift"
        )
    # Reuse the exact inherited real/fake physical-core validator.
    _validate_qualification_terminal_runtime(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4."
                "teacher_pool_terminal.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "pool_index": 0,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "stage_runtime": core,
            "backend_runtime": row["backend_runtime_core"],
            "teacher_executed": False,
        },
        runtime,
        expected_pool_index=0,
        allow_fake_runtime=allow_fake_runtime,
    )
    backend = _mapping(
        row["backend_runtime_core"],
        set(C.QUALIFICATION_BACKEND_RUNTIME_CORE_FIELDS),
        "qualification backend runtime core",
    )
    if row["backend_runtime_core_sha256"] != _canonical_value_sha256(backend):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification backend runtime core digest drift"
        )
    count = _nonnegative_int(
        row["qualification_shard_count"], "qualification runtime shard count"
    )
    stage_hashes = _sequence(
        row["qualification_stage_runtime_sha256s"],
        "qualification stage runtime hashes",
    )
    backend_hashes = _sequence(
        row["qualification_backend_runtime_sha256s"],
        "qualification backend runtime hashes",
    )
    if (
        count != C.V3.PROSPECTIVE_POOL_COUNT
        or len(stage_hashes) != count
        or len(backend_hashes) != count
        or any(_sha(item, "qualification stage runtime SHA-256") != core_sha for item in stage_hashes)
        or any(_sha(item, "qualification backend runtime SHA-256") != item for item in backend_hashes)
        or row["all_qualification_stage_runtime_cores_equal"] is not True
        or row["all_qualification_backend_runtime_cores_equal"] is not True
        or not isinstance(row["models_trained"], int)
        or isinstance(row["models_trained"], bool)
        or row["models_trained"] != 0
        or row["prohibited_components_trained_or_implemented"] != []
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification runtime population/no-training drift"
        )
    if material_shard_validations is not None:
        validations = _sequence(
            material_shard_validations, "qualification material validations"
        )
        if (
            len(validations) != count
            or [item.get("stage_runtime_sha256") for item in validations]
            != stage_hashes
            or [item.get("backend_runtime_sha256") for item in validations]
            != backend_hashes
            or any(
                item.get("backend_runtime_core_sha256")
                != row["backend_runtime_core_sha256"]
                for item in validations
            )
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "qualification runtime/material validation cross-link drift"
            )
    return row


def _call_v3(validator: Any, value: Any, *dependencies: Any, **kwargs: Any) -> Any:
    try:
        validator(
            project_v4_evidence_to_v3(value),
            *[project_v4_evidence_to_v3(item) for item in dependencies],
            **kwargs,
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc
    return copy.deepcopy(value)


def physical_trace_reduction_authority() -> dict[str, Any]:
    return project_v3_evidence_to_v4(V3M.physical_trace_reduction_authority())


def panel_manifest_authority() -> dict[str, Any]:
    return project_v3_evidence_to_v4(V3M.panel_manifest_authority())


def external_artifact_bindings(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        if contract.get("schema") == (
            "physical_graph_edge_handoff_qualification_v4.runtime_contract.v1"
        ):
            row = C.validate_runtime_contract(contract)
            return copy.deepcopy(row["external_artifact_bindings"])
        C.validate_contract(contract)
        return [copy.deepcopy(item) for item in C.EXTERNAL_ARTIFACT_BINDINGS]
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc


def predecessor_context_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if contract.get("schema") == (
            "physical_graph_edge_handoff_qualification_v4.runtime_contract.v1"
        ):
            row = C.validate_runtime_contract(contract)["scientific_contract"]
        else:
            row = C.validate_contract(contract)
        expected = C.V3.V2.V1.V2_CONTEXT_BINDING
        if row["v2_context_binding"] != expected:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "predecessor context drift"
            )
        return copy.deepcopy(row["v2_context_binding"])
    except PhysicalGraphEdgeHandoffV4MetricsError:
        raise
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc


def teacher_trace_npz_authority(teacher_trace_count: int) -> dict[str, Any]:
    return C.teacher_trace_npz_authority(teacher_trace_count)


def validate_npz_inspections(
    value: Any, *, teacher_trace_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate five success NPZs with a real sparse-teacher trace extent."""

    rows = _sequence(value, "npz inspections")
    teacher_rows = [row for row in rows if isinstance(row, Mapping) and row.get("path") == "teacher_traces.npz"]
    if len(teacher_rows) != 1:
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher NPZ inspection coverage drift")
    try:
        offsets = teacher_rows[0]["members"]["trace_offsets"]
        observed = _nonnegative_int(offsets["shape"][0], "teacher offset extent") - 1
    except (KeyError, IndexError, TypeError) as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher NPZ offset inspection drift"
        ) from exc
    if teacher_trace_count is not None and observed != teacher_trace_count:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher NPZ/record cardinality drift"
        )
    dynamic_teacher = C.teacher_trace_npz_authority(observed)
    payload_authority = copy.deepcopy(V1M.C.NPZ_PAYLOAD_AUTHORITY)
    payload_authority["teacher_traces.npz"] = dynamic_teacher

    class _SparseNpzContract:
        def __init__(self) -> None:
            self.NPZ_PAYLOAD_AUTHORITY = payload_authority

        def __getattr__(self, name: str) -> Any:
            return getattr(V1M.C, name)

    try:
        result = _local_v1_call(
            V1M.validate_npz_inspections,
            _project_v4_evidence_to_v1(rows),
            replacements={"C": _SparseNpzContract()},
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc
    return copy.deepcopy(result)


def validate_visual_runtime_environment(value: Any, *, runtime_role: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(
            V3M.validate_visual_runtime_environment(value, runtime_role=runtime_role)
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc


def validate_panel_manifest(value: Any) -> dict[str, Any]:
    """Validate the V4 panel without fabricating pre-teacher evidence."""

    try:
        if not isinstance(value, Mapping):
            raise PhysicalGraphEdgeHandoffV4MetricsError("panel manifest is not mapping")
        supplied = copy.deepcopy(dict(value))
        C.validate_content_digest(supplied)
        root = V1M._validate_document(
            _project_v4_evidence_to_v1(supplied), V1M.PANEL_FIELDS,
            "physical_graph_edge_handoff_qualification_v1.panel_manifest.v1",
            "panel_manifest",
        )
        root["physical_runtime_environment"] = V1M.validate_physical_runtime_environment(
            root["physical_runtime_environment"]
        )
        B = V1M.C
        if (
            root["constructed_set_caveat"] != B.CONSTRUCTED_SET_CAVEAT
            or root["identity_domain"] != B.IDENTITY_DOMAIN
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError("panel claims/identity drift")
        exclusion = V1M._mapping(
            root["prior_exclusion_evidence"], V1M.EXCLUSION_EVIDENCE_FIELDS,
            "prior exclusion",
        )
        if exclusion["authority_digest"] != hashlib.sha256(
            B.canonical_json_bytes(B.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
        ).hexdigest():
            raise PhysicalGraphEdgeHandoffV4MetricsError("prior exclusion authority drift")
        for key in (
            "scene_overlap_count", "scene_hash_overlap_count",
            "state_or_episode_overlap_count", "seed_overlap_count",
            "path_or_geometry_overlap_count", "structured_path_overlap_count",
        ):
            if V1M._integer(exclusion[key], f"prior exclusion {key}") != 0:
                raise PhysicalGraphEdgeHandoffV4MetricsError("prior-panel identity overlap")
        if (
            exclusion["checked_before_simulator_creation"] is not True
            or exclusion["all_zero"] is not True
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError("prior exclusion gate failed")

        pool = V1M._mapping(
            root["prospective_pool_selection"], V1M.POOL_SELECTION_FIELDS,
            "prospective pool selection",
        )
        specifications = B.build_candidate_specs()
        if (
            pool["candidate_spec_count"] != B.PROSPECTIVE_POOL_COUNT
            or pool["candidate_specs_sha256"]
            != hashlib.sha256(B.canonical_json_bytes(specifications)[:-1]).hexdigest()
            or pool["selection_rule"]
            != B.GEOMETRY_AUTHORITY["teacher_only_scan"]["selection"]
            or pool["teacher_scan_completed_before_role_assignment"] is not True
            or pool["ranker_or_fanout_outcomes_opened"] != 0
            or pool["complete"] is not True
            or pool["per_family_scanned_counts"]
            != {family: B.PROSPECTIVE_POOL_PER_FAMILY for family in B.FAMILY_IDS}
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError("prospective pool authority drift")
        qualification_rows = V1M._sequence(
            pool["qualification_rows"], "qualification rows"
        )
        if len(qualification_rows) != B.PROSPECTIVE_POOL_COUNT:
            raise PhysicalGraphEdgeHandoffV4MetricsError("qualification row count drift")
        teacher_nullable_fields = (
            "goal_reachable", "teacher_trace_contact_free",
            "teacher_left_source_region", "teacher_crossed_directed_port",
            "teacher_positive_route_progress", "teacher_competing_port_entered",
            "teacher_normal_positive", "teacher_within_lateral_bounds",
            "teacher_dwell_satisfied", "directed_port_defined",
            "graph_edge_physically_executable",
        )
        qualification: list[dict[str, Any]] = []
        eligible_counts: Counter[str] = Counter()
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        teacher_trace_count = 0
        for index, (raw, spec) in enumerate(zip(qualification_rows, specifications)):
            row = V1M._mapping(
                raw, C.V4_POOL_QUALIFICATION_FIELDS,
                f"qualification[{index}]",
            )
            for field in (
                "candidate_spec_id", "state_id", "family", "stratum_index",
                "variant_index", "canonical_spec_sha256",
            ):
                if row[field] != spec[field]:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        f"qualification {field} drift"
                    )
            if row["selection_key_sha256"] != spec["canonical_spec_sha256"]:
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "qualification selection key drift"
                )
            teacher_executed = row["teacher_trace_id"] is not None
            trace_triplet = (
                row["teacher_trace_id"], row["teacher_trace_index"],
                row["teacher_trace_slice_sha256"],
            )
            if teacher_executed:
                if (
                    row["teacher_trace_index"] != teacher_trace_count
                    or any(item is None for item in trace_triplet)
                ):
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "teacher qualification identity/order drift"
                    )
                V1M._string(row["teacher_trace_id"], "teacher trace id")
                V1M._sha(row["teacher_trace_slice_sha256"], "teacher trace slice")
                V1M._sha(row["initial_decision_state_sha256"], "initial decision state")
                components = {
                    "goal_reachable": V1M._boolean(row["goal_reachable"], "goal reachable"),
                    "teacher_trace_contact_free": V1M._boolean(row["teacher_trace_contact_free"], "teacher contact"),
                    "teacher_left_source_region": V1M._boolean(row["teacher_left_source_region"], "teacher source leave"),
                    "teacher_crossed_directed_port": V1M._boolean(row["teacher_crossed_directed_port"], "teacher crossing"),
                    "teacher_positive_route_progress": V1M._boolean(row["teacher_positive_route_progress"], "teacher progress"),
                    "teacher_no_competing_port": not V1M._boolean(row["teacher_competing_port_entered"], "teacher competing"),
                    "teacher_normal_positive": V1M._boolean(row["teacher_normal_positive"], "teacher normal"),
                    "teacher_within_lateral_bounds": V1M._boolean(row["teacher_within_lateral_bounds"], "teacher lateral"),
                    "teacher_dwell_satisfied": V1M._boolean(row["teacher_dwell_satisfied"], "teacher dwell"),
                    "directed_port_defined": V1M._boolean(row["directed_port_defined"], "directed port"),
                    "current_rgb_valid": V1M._boolean(row["current_rgb_valid"], "current RGB"),
                    "graph_edge_physically_executable": V1M._boolean(row["graph_edge_physically_executable"], "graph executable"),
                }
                teacher_valid = all(
                    components[name]
                    for name in B.TEACHER_QUALIFICATION_COMPONENT_IDS
                    if name not in {"goal_reachable", "current_rgb_valid"}
                )
                qualified = row["disposition"] == "QUALIFIED"
                if (
                    row["teacher_valid"] is not teacher_valid
                    or row["qualified"] is not qualified
                    or qualified and not all(components.values())
                ):
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "qualification predicate drift"
                    )
                if row["disposition"] not in C.STATE_DISPOSITIONS:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "teacher qualification disposition drift"
                    )
                expected_selection_reason = (
                    None if row["selected"] else "HASH_ORDER_NOT_SELECTED"
                ) if qualified else row["disposition"]
                if row["rejection_reason"] != expected_selection_reason:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "teacher qualification rejection reason drift"
                    )
                teacher_trace_count += 1
            else:
                if any(item is not None for item in trace_triplet):
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "partial teacher identity was fabricated"
                    )
                if row["initial_decision_state_sha256"] is not None:
                    V1M._sha(row["initial_decision_state_sha256"], "initial snapshot")
                if any(row[field] is not None for field in teacher_nullable_fields):
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "pre-teacher qualification contains fabricated values"
                    )
                V1M._boolean(row["current_rgb_valid"], "current RGB")
                if (
                    row["current_rgb_valid"]
                    is not (row["initial_decision_state_sha256"] is not None)
                    or
                    row["teacher_valid"] is not False
                    or row["qualified"] is not False
                    or row["disposition"] not in {
                        "INITIAL_BOUNDARY_TIPPED", "RESTORATION_PROBE_TIPPED",
                        "UNRESOLVED_STATE_FAILURE",
                    }
                    or row["rejection_reason"] != row["disposition"]
                    or row["rank_within_stratum"] is not None
                    or row["selected"] is not False
                ):
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "pre-teacher qualification projection drift"
                    )
            if row["qualified"]:
                eligible_counts[spec["family"]] += 1
                grouped[(spec["family"], spec["stratum_index"])].append(row)
            elif row["rank_within_stratum"] is not None or row["selected"] is not False:
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "nonqualified row has selection evidence"
                )
            qualification.append(row)

        selected_by_stratum: dict[tuple[str, int], str] = {}
        selected_ids: list[str] = []
        for family in B.FAMILY_IDS:
            for stratum in range(16):
                rows = sorted(
                    grouped[(family, stratum)],
                    key=lambda item: (
                        item["selection_key_sha256"], item["candidate_spec_id"]
                    ),
                )
                if not rows:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(
                        "teacher-qualified stratum empty in adequate panel"
                    )
                for rank, row in enumerate(rows):
                    if row["rank_within_stratum"] != rank or row["selected"] is not (rank == 0):
                        raise PhysicalGraphEdgeHandoffV4MetricsError(
                            "qualification rank/selection drift"
                        )
                selected_by_stratum[(family, stratum)] = rows[0]["candidate_spec_id"]
        selected_ids = [row["state_id"] for row in qualification if row["selected"]]
        if not C.V3.STATE_COUNT <= teacher_trace_count <= C.V3.PROSPECTIVE_POOL_COUNT:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "adequate-panel teacher trace cardinality drift"
            )
        expected_eligible = {family: eligible_counts[family] for family in B.FAMILY_IDS}
        if pool["per_family_teacher_eligible_counts"] != expected_eligible:
            raise PhysicalGraphEdgeHandoffV4MetricsError("eligible count drift")
        if pool["rejection_reason_counts"] != dict(sorted(Counter(
            row["disposition"] for row in qualification if not row["qualified"]
        ).items())):
            raise PhysicalGraphEdgeHandoffV4MetricsError("disposition count drift")
        if pool["qualification_projection_sha256"] != hashlib.sha256(
            B.canonical_json_bytes(qualification)[:-1]
        ).hexdigest():
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "qualification projection digest drift"
            )
        selected_projection = pool["selected_candidate_spec_ids_by_family_and_stratum"]
        if not isinstance(selected_projection, Mapping) or set(selected_projection) != set(B.FAMILY_IDS):
            raise PhysicalGraphEdgeHandoffV4MetricsError("selected projection fields drift")
        for family in B.FAMILY_IDS:
            values = V1M._sequence(selected_projection[family], f"selected {family}")
            if len(values) != 16 or any(
                value != selected_by_stratum[(family, stratum)]
                for stratum, value in enumerate(values)
            ):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "selected stratum projection drift"
                )
        if pool["selected_state_ids"] != selected_ids or len(selected_ids) != B.STATE_COUNT:
            raise PhysicalGraphEdgeHandoffV4MetricsError("selected state projection drift")

        states = V1M._sequence(root["states"], "panel states")
        if len(states) != B.STATE_COUNT:
            raise PhysicalGraphEdgeHandoffV4MetricsError("panel state count drift")
        selected_specs = {row["candidate_spec_id"] for row in qualification if row["selected"]}
        spec_by_id = {row["candidate_spec_id"]: row for row in specifications}
        ids: set[str] = set(); scenes: set[str] = set(); episodes: set[str] = set()
        graphs: set[str] = set(); seeds: set[int] = set()
        family_counts: Counter[str] = Counter(); role_counts: Counter[str] = Counter()
        family_role: Counter[tuple[str, str]] = Counter(); turn_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter(); family_route: Counter[tuple[str, str]] = Counter()
        widths: Counter[str] = Counter(); distances: Counter[str] = Counter()
        strata: Counter[tuple[str, int]] = Counter(); validated_states: list[dict[str, Any]] = []
        for index, raw in enumerate(states):
            state = V1M._mapping(raw, V1M.PANEL_STATE_FIELDS, f"panel state[{index}]")
            for field, seen in (
                ("state_id", ids), ("scene_id", scenes), ("episode_id", episodes),
                ("graph_id", graphs),
            ):
                item = V1M._string(state[field], field)
                if item in seen:
                    raise PhysicalGraphEdgeHandoffV4MetricsError("panel identity duplicate")
                seen.add(item)
            seed = V1M._integer(state["procedural_seed"], "procedural seed")
            if seed not in B.PROCEDURAL_SEED_VALUES or seed in seeds:
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel seed drift")
            seeds.add(seed)
            family = state["family"]; role = state["role"]; route = state["route_direction"]
            if family not in B.FAMILY_IDS or role not in B.ROLE_IDS or route not in {"STRAIGHT", "LEFT", "RIGHT"}:
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel family/role/route drift")
            family_counts[family] += 1; role_counts[role] += 1
            family_role[(family, role)] += 1; route_counts[route] += 1
            family_route[(family, route)] += 1
            width = state["passage_width_id"]; distance = state["port_distance_id"]
            if width not in {"NARROW", "WIDE"} or distance not in {"NEAR", "FAR"}:
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel stratum label drift")
            widths[width] += 1; distances[distance] += 1
            stratum = V1M._integer(state["stratum_index"], "stratum index")
            if stratum >= 16:
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel stratum index drift")
            strata[(family, stratum)] += 1
            turn = state["turn_direction"]
            if family == "TURNING_JUNCTION":
                if turn not in {"LEFT", "RIGHT"}:
                    raise PhysicalGraphEdgeHandoffV4MetricsError("turn direction drift")
                turn_counts[turn] += 1
            elif turn is not None:
                raise PhysicalGraphEdgeHandoffV4MetricsError("non-turn family has turn")
            if state["candidate_spec_id"] not in selected_specs:
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel state not selected")
            spec = spec_by_id[state["candidate_spec_id"]]
            for field in (
                "state_id", "scene_id", "episode_id", "graph_id", "family",
                "route_direction", "passage_width_id", "port_distance_id",
                "stratum_index", "procedural_seed",
            ):
                if state[field] != spec[field]:
                    raise PhysicalGraphEdgeHandoffV4MetricsError(f"panel/spec {field} drift")
            if state["geometry_sha256"] != hashlib.sha256(
                B.canonical_json_bytes(spec["geometry"])[:-1]
            ).hexdigest():
                raise PhysicalGraphEdgeHandoffV4MetricsError("panel geometry digest drift")
            for field in ("source_node_id", "target_node_id", "directed_edge_id"):
                V1M._string(state[field], field)
            eligibility = V1M._mapping(
                state["eligibility_evidence"], V1M.ELIGIBILITY_FIELDS,
                "eligibility evidence",
            )
            if state["goal_reachable"] is not True or state["eligible"] is not True or not all(item is True for item in eligibility.values()):
                raise PhysicalGraphEdgeHandoffV4MetricsError("ineligible selected state")
            validated_states.append(state)
        if (
            family_counts != Counter({family: 16 for family in B.FAMILY_IDS})
            or role_counts != Counter(B.ROLE_COUNTS)
            or family_role != Counter({(family, role): count for family, values in B.FAMILY_ROLE_COUNTS.items() for role, count in values.items()})
            or turn_counts != Counter(B.TURNING_JUNCTION_SIDE_COUNTS)
            or route_counts != Counter(B.GEOMETRY_AUTHORITY["direction_counts"])
            or family_route != Counter({(family, direction): count for family, counts in B.GEOMETRY_AUTHORITY["family_direction_counts"].items() for direction, count in counts.items()})
            or widths != Counter({"NARROW": 32, "WIDE": 32})
            or distances != Counter({"NEAR": 32, "FAR": 32})
            or strata != Counter({(family, index): 1 for family in B.FAMILY_IDS for index in range(16)})
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError("panel quota/balance drift")
        order = [state["state_id"] for state in validated_states]
        if order != sorted(order) or order != selected_ids:
            raise PhysicalGraphEdgeHandoffV4MetricsError("panel canonical state order drift")
        return supplied
    except PhysicalGraphEdgeHandoffV4MetricsError:
        raise
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc


def validate_split_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_split_manifest, value, panel,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_graph_manifest(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_graph_manifest, value, panel,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_state_snapshot_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_state_snapshot_index, value, panel,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_teacher_trace_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only actually executed teachers, with compact trace indices."""

    panel = validate_panel_manifest(panel_manifest)
    if not isinstance(value, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher trace index is not mapping")
    supplied = copy.deepcopy(dict(value))
    C.validate_content_digest(supplied)
    if (
        supplied.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.teacher_trace_index.v1"
        or supplied.get("experiment_id") != C.EXPERIMENT_ID
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher trace identity drift")
    raw_records = _sequence(supplied.get("records"), "teacher records")
    teacher_qualifications = [
        row for row in panel["prospective_pool_selection"]["qualification_rows"]
        if row["teacher_trace_id"] is not None
    ]
    if (
        len(raw_records) != len(teacher_qualifications)
        or not C.V3.STATE_COUNT <= len(raw_records) <= C.V3.PROSPECTIVE_POOL_COUNT
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "sparse teacher trace cardinality drift"
        )
    field_set = set(V1M.TEACHER_RECORD_FIELDS) | set(
        C.V4_TEACHER_RECORD_ADDITIONAL_FIELDS
    )
    specifications = _CANDIDATE_SPECS
    stripped_records: list[dict[str, Any]] = []
    previous_pool = -1
    for compact_index, (raw, qualification) in enumerate(
        zip(raw_records, teacher_qualifications)
    ):
        record = _mapping(raw, field_set, f"teacher[{compact_index}]")
        pool_index = _nonnegative_int(
            record["qualification_pool_index"],
            f"teacher[{compact_index}].qualification_pool_index",
        )
        if (
            pool_index >= C.V3.PROSPECTIVE_POOL_COUNT
            or pool_index <= previous_pool
            or record["trace_index"] != compact_index
            or qualification["teacher_trace_index"] != compact_index
            or qualification["candidate_spec_id"]
            != specifications[pool_index]["candidate_spec_id"]
            or record["candidate_spec_id"] != qualification["candidate_spec_id"]
            or record["teacher_trace_id"] != qualification["teacher_trace_id"]
            or record["trace_array_slice_sha256s"]["base_pose_world"]
            != qualification["teacher_trace_slice_sha256"]
            or record["initial_decision_state_sha256"]
            != qualification["initial_decision_state_sha256"]
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "sparse teacher pool/trace mapping drift"
            )
        previous_pool = pool_index
        stripped = copy.deepcopy(record)
        stripped.pop("qualification_pool_index")
        stripped_records.append(stripped)

    class _TeacherSubsetContract:
        def __init__(self) -> None:
            self.TEACHER_TRACE_COUNT = len(stripped_records)

        def build_candidate_specs(self) -> list[dict[str, Any]]:
            return [
                copy.deepcopy(specifications[row["qualification_pool_index"]])
                for row in raw_records
            ]

        def __getattr__(self, name: str) -> Any:
            return getattr(V1M.C, name)

    def teacher_integer(
        item: Any, label: str, minimum: int = 0,
    ) -> int:
        return V1M._integer(
            item, label,
            minimum=1 if label == "teacher sample_count" else minimum,
        )

    v1_document = _project_v4_evidence_to_v1(supplied)
    v1_document.pop("content_digest", None)
    v1_document["records"] = stripped_records
    v1_document = V1M.C.attach_content_digest(v1_document)
    v1_panel = _project_v4_evidence_to_v1(panel)
    try:
        _local_v1_call(
            V1M.validate_teacher_trace_index, v1_document, v1_panel,
            replacements={
                "C": _TeacherSubsetContract(),
                "_integer": teacher_integer,
                "validate_panel_manifest": _exact_v1_dependency(
                    v1_panel, "validate_panel_manifest"
                ),
            },
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc
    return supplied


def validate_teacher_selection(
    value: Any, panel_manifest: Mapping[str, Any],
    teacher_trace_index: Mapping[str, Any],
    qualification_dispositions: Sequence[Mapping[str, Any]] | bytes | str,
    panel_adequacy: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind the material compact-teacher construction authority."""

    selection = _mapping(value, C.TEACHER_SELECTION_FIELDS, "teacher selection")
    C.validate_content_digest(selection)
    if (
        selection["schema"]
        != "physical_graph_edge_handoff_qualification_v4.teacher_selection.v1"
        or selection["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection identity drift"
        )
    panel = validate_panel_manifest(panel_manifest)
    teachers = validate_teacher_trace_index(teacher_trace_index, panel)
    if isinstance(qualification_dispositions, (bytes, str)):
        dispositions = validate_qualification_state_dispositions_jsonl(
            qualification_dispositions
        )
    else:
        dispositions = [
            validate_qualification_disposition_row(item, expected_pool_index=index)
            for index, item in enumerate(
                _sequence(qualification_dispositions, "teacher selection dispositions")
            )
        ]
        if len(dispositions) != C.V3.PROSPECTIVE_POOL_COUNT:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher selection disposition cardinality drift"
            )
    adequacy = validate_panel_adequacy(panel_adequacy, dispositions)
    if adequacy["adequate"] is not True:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection exists for inadequate panel"
        )
    expected_specs_sha = hashlib.sha256(
        C.canonical_json_bytes(_CANDIDATE_SPECS)[:-1]
    ).hexdigest()
    if selection["candidate_specs_sha256"] != expected_specs_sha:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection candidate-spec digest drift"
        )
    trace_binding = V1M._validate_file_binding(
        selection["teacher_traces_file"], "teacher selection trace file",
        "teacher_traces.npz",
    )
    if trace_binding != teachers["traces_file"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection/index trace-file binding drift"
        )
    raw_records = _sequence(selection["teacher_records"], "teacher selection records")
    if C.canonical_json_bytes(raw_records) != C.canonical_json_bytes(teachers["records"]):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection/index record drift"
        )
    raw_qualification = _sequence(
        selection["qualification_rows"], "teacher selection qualification rows"
    )
    panel_qualification = panel["prospective_pool_selection"]["qualification_rows"]
    if C.canonical_json_bytes(raw_qualification) != C.canonical_json_bytes(panel_qualification):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection/panel qualification drift"
        )
    if selection["qualification_projection_sha256"] != hashlib.sha256(
        C.canonical_json_bytes(raw_qualification)[:-1]
    ).hexdigest():
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection qualification digest drift"
        )
    selected_indices = _sequence(
        selection["selected_pool_indices"], "teacher selected pool indices"
    )
    if selected_indices != adequacy["selected_pool_indices"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection/adequacy selected pools drift"
        )
    expected_selected_specs = [
        copy.deepcopy(_CANDIDATE_SPECS[index]) for index in selected_indices
    ]
    if C.canonical_json_bytes(selection["selected_specs"]) != C.canonical_json_bytes(expected_selected_specs):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection selected-spec projection drift"
        )
    if (
        selection["rejection_reason_counts"]
        != panel["prospective_pool_selection"]["rejection_reason_counts"]
        or selection["panel_adequacy_sha256"]
        != hashlib.sha256(C.canonical_json_bytes(adequacy)).hexdigest()
        or selection["fanout_or_ranker_outcomes_opened"] != 0
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher selection boundary projection drift"
        )
    record_by_pool = {
        record["qualification_pool_index"]: record
        for record in raw_records
    }
    selected_set = set(selected_indices)
    for pool_index, (qualification, disposition) in enumerate(
        zip(raw_qualification, dispositions)
    ):
        identity = disposition["snapshot_identity"]
        expected_snapshot_sha = (
            identity["artifact_file_sha256"]
            if disposition["executable_snapshot_exists"] else None
        )
        if (
            qualification["candidate_spec_id"] != disposition["candidate_spec_id"]
            or qualification["state_id"] != disposition["state_id"]
            or qualification["disposition"] != disposition["disposition"]
            or qualification["qualified"] is not disposition["qualified"]
            or qualification["initial_decision_state_sha256"]
            != expected_snapshot_sha
            or qualification["current_rgb_valid"]
            is not disposition["executable_snapshot_exists"]
            or qualification["selected"] is not (pool_index in selected_set)
            or (pool_index in record_by_pool) is not disposition["teacher_executed"]
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher selection/disposition cross-link drift"
            )
        if disposition["teacher_executed"]:
            record = record_by_pool[pool_index]
            criteria = disposition["teacher_criteria"]
            assert criteria is not None
            expected_teacher_projection = {
                "goal_reachable": criteria["goal_reachable"],
                "teacher_trace_contact_free": criteria[
                    "teacher_trace_contact_free"
                ],
                "teacher_left_source_region": criteria[
                    "teacher_left_source_region"
                ],
                "teacher_crossed_directed_port": criteria[
                    "teacher_crossed_directed_port"
                ],
                "teacher_positive_route_progress": criteria[
                    "teacher_positive_route_progress"
                ],
                "teacher_competing_port_entered": not criteria[
                    "teacher_no_competing_port"
                ],
                "teacher_normal_positive": criteria["teacher_normal_positive"],
                "teacher_within_lateral_bounds": criteria[
                    "teacher_within_lateral_bounds"
                ],
                "teacher_dwell_satisfied": criteria["teacher_dwell_satisfied"],
                "directed_port_defined": criteria["directed_port_defined"],
                "current_rgb_valid": criteria["current_rgb_valid"],
                "graph_edge_physically_executable": criteria[
                    "graph_edge_physically_executable"
                ],
                "teacher_valid": criteria["teacher_valid"],
            }
            if (
                record["trace_index"] != qualification["teacher_trace_index"]
                or record["teacher_trace_id"] != qualification["teacher_trace_id"]
                or record["selected"] is not qualification["selected"]
                or any(
                    qualification[field] is not expected
                    for field, expected in expected_teacher_projection.items()
                )
            ):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "teacher selection compact mapping drift"
                )
    return selection


def validate_panel_context(
    value: Any, panel_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the material panel/split construction cross-link."""

    context = _mapping(value, C.PANEL_CONTEXT_FIELDS, "panel context")
    C.validate_content_digest(context)
    if (
        context["schema"]
        != "physical_graph_edge_handoff_qualification_v4.panel_context.v1"
        or context["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "panel context identity drift"
        )
    panel = validate_panel_manifest(panel_manifest)
    split = validate_split_manifest(split_manifest, panel)
    selected_state_ids = [row["state_id"] for row in panel["states"]]
    role_by_state = {
        row["state_id"]: row["role"] for row in panel["states"]
    }
    assignments = {row["state_id"]: row["role"] for row in split["assignments"]}
    if role_by_state != assignments:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "panel context split role drift"
        )
    expected = C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4.panel_context.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "selected_state_ids": selected_state_ids,
            "role_by_state": role_by_state,
            "panel_sha256": hashlib.sha256(
                C.canonical_json_bytes(panel)
            ).hexdigest(),
            "split_sha256": hashlib.sha256(
                C.canonical_json_bytes(split)
            ).hexdigest(),
            "heldout_outcomes_opened": 0,
        }
    )
    if C.canonical_json_bytes(context) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "panel context projection drift"
        )
    return context


def validate_encoding_receipt(
    value: Any, panel_manifest: Mapping[str, Any],
    pixel_index: Mapping[str, Any], latent_index: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every material singleton-encoding receipt cross-link."""

    receipt = _mapping(value, C.ENCODING_RECEIPT_FIELDS, "encoding receipt")
    C.validate_content_digest(receipt)
    if (
        receipt["schema"]
        != "physical_graph_edge_handoff_qualification_v4.encoding_receipt.v1"
        or receipt["experiment_id"] != C.EXPERIMENT_ID
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "encoding receipt identity drift"
        )
    panel = validate_panel_manifest(panel_manifest)
    pixels = validate_pixel_index(pixel_index, panel)
    latents = validate_latent_index(latent_index, pixels)
    pixel_order_by_index: dict[int, str] = {}
    for row in pixels["records"]:
        index = _nonnegative_int(
            row["canonical_pixel_index"], "canonical pixel index"
        )
        digest = _sha(row["pixel_sha256"], "canonical pixel SHA-256")
        old = pixel_order_by_index.setdefault(index, digest)
        if old != digest:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "canonical pixel index collision"
            )
    pixel_order = [
        pixel_order_by_index[index] for index in range(len(pixel_order_by_index))
    ]
    if pixel_order != sorted(pixel_order):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "canonical pixel lexical order drift"
        )
    expected_records: list[dict[str, Any]] = []
    for index, raw in enumerate(latents["records"]):
        record = _mapping(
            raw, V1M.LATENT_RECORD_FIELDS, f"latent receipt record[{index}]",
        )
        if record["canonical_pixel_index"] != index:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "latent receipt row order drift"
            )
        expected_records.append(
            {
                "canonical_pixel_index": index,
                "pixel_sha256": record["pixel_sha256"],
                "preprocessed_tensor_sha256": record[
                    "preprocessed_tensor_sha256"
                ],
                "raw_token_sha256": record["raw_token_sha256"],
                "spatial_descriptor_sha256": record[
                    "spatial_descriptor_sha256"
                ],
            }
        )
    expected = C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4.encoding_receipt.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "singleton_count": len(pixel_order),
            "pixel_order": pixel_order,
            "records": expected_records,
            "preprocessing_authority": copy.deepcopy(
                latents["preprocessing_authority"]
            ),
            "external_encoder_source": copy.deepcopy(
                latents["external_encoder_source"]
            ),
            "encoder_runtime_environment": copy.deepcopy(
                latents["encoder_runtime_environment"]
            ),
            "fanout_outcomes_opened": 0,
        }
    )
    if C.canonical_json_bytes(receipt) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "encoding receipt projection drift"
        )
    return receipt


def validate_edge_port_index(value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any], teacher_trace_index: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    graph = validate_graph_manifest(graph_manifest, panel)
    teachers = validate_teacher_trace_index(teacher_trace_index, panel)
    return _call_v1_with_validated_dependencies(
        V1M.validate_edge_port_index, value, panel, graph, teachers,
        dependency_replacements={
            "validate_panel_manifest": panel,
            "validate_graph_manifest": graph,
            "validate_teacher_trace_index": teachers,
        },
    )


def validate_waypoint_contracts(value: Any, panel_manifest: Mapping[str, Any], graph_manifest: Mapping[str, Any], edge_port_index: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    graph = validate_graph_manifest(graph_manifest, panel)
    return _call_v1_with_validated_dependencies(
        V1M.validate_waypoint_contracts, value, panel, graph, edge_port_index,
        dependency_replacements={
            "validate_panel_manifest": panel,
            "validate_graph_manifest": graph,
        },
    )


def validate_pixel_index(value: Any, panel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_pixel_index, value, panel,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_latent_index(value: Any, pixel_index: Mapping[str, Any]) -> dict[str, Any]:
    return _call_v1_with_validated_dependencies(
        V1M.validate_latent_index, value, pixel_index,
        dependency_replacements={},
    )


def validate_candidate_fanout_rows(value: Any, panel_manifest: Mapping[str, Any], state_snapshot_index: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_candidate_fanout_rows, value, panel, state_snapshot_index,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_development_target_selection(value: Any, panel_manifest: Mapping[str, Any], candidate_fanout: Sequence[Mapping[str, Any]], waypoint_contracts: Mapping[str, Any]) -> dict[str, Any]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_development_target_selection, value, panel,
        candidate_fanout, waypoint_contracts,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def validate_heldout_ranker_score_rows(value: Any, panel_manifest: Mapping[str, Any], candidate_fanout: Sequence[Mapping[str, Any]], development_target_selection: Mapping[str, Any], waypoint_contracts: Mapping[str, Any], teacher_trace_index: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    teachers = validate_teacher_trace_index(teacher_trace_index, panel)
    return _call_v1_with_validated_dependencies(
        V1M.validate_heldout_ranker_score_rows, value, panel, candidate_fanout,
        development_target_selection, waypoint_contracts, teachers,
        dependency_replacements={
            "validate_panel_manifest": panel,
            "validate_teacher_trace_index": teachers,
        },
    )


def validate_repeated_execution_rows(value: Any, panel_manifest: Mapping[str, Any], candidate_fanout: Sequence[Mapping[str, Any]], heldout_ranker_scores: Sequence[Mapping[str, Any]], state_snapshot_index: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel = validate_panel_manifest(panel_manifest)
    return _call_v1_with_validated_dependencies(
        V1M.validate_repeated_execution_rows, value, panel, candidate_fanout,
        heldout_ranker_scores, state_snapshot_index,
        dependency_replacements={"validate_panel_manifest": panel},
    )


def _partial_probe_termination_reason(flags: Mapping[str, bool]) -> str:
    return next(
        (
            C.PARTIAL_PROBE_TERMINATION_REASON_BY_FLAG[name]
            for name in C.TERMINATION_FLAG_ORDER
            if flags[name]
        ),
        C.PARTIAL_PROBE_NO_FLAG_TERMINATION_REASON,
    )


def _terminated_trace(
    value: Any, label: str, *, require_tipped: bool = False,
) -> dict[str, Any]:
    import numpy as np

    row = _mapping(value, C.TIPPED_BEHAVIOURAL_TRACE_FIELDS, label)
    sample_count: int | None = None
    for member, authority in C.V4_PROBE_TRACE_MEMBER_AUTHORITY.items():
        array = np.asarray(row[member])
        if array.ndim != len(authority["shape"]) or array.dtype.str != authority["descr"]:
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label}.{member} dtype/rank drift")
        if list(array.shape[1:]) != list(authority["shape"][1:]):
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label}.{member} trailing shape drift")
        if sample_count is None:
            sample_count = int(array.shape[0])
        if int(array.shape[0]) != sample_count:
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} sample cardinality drift")
        if array.dtype.kind not in "fc" and not bool(
            np.isin(array, [0, 1]).all()
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"{label}.{member} nonbinary"
            )
        row[member] = np.ascontiguousarray(array)
    if sample_count is None or not 1 <= sample_count <= C.V3.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} sample count drift")
    tip = row["tip_sample_index"]
    if not isinstance(tip, int) or isinstance(tip, bool) or tip != sample_count - 1:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} inclusive tip sample drift")
    flags = _termination_flags(row["termination_flags"], f"{label}.termination_flags")
    assert flags is not None
    _terminal_nonfinite_masks(
        {member: row[member] for member in C.V4_PROBE_TRACE_MEMBER_AUTHORITY},
        flags,
        label=label,
        allowed_members=C.TERMINAL_NONFINITE_TRACE_MEMBER_IDS,
        require_trace_axis=True,
    )
    if (
        (require_tipped and flags["tipped"] is not True)
        or type(row["stuck"]) is not bool
        or row["termination_reason"] != _partial_probe_termination_reason(flags)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} terminal metadata drift")
    expected_request = np.broadcast_to(
        np.asarray(C.V3.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64),
        row["requested_command"].shape,
    )
    if not np.array_equal(row["requested_command"], expected_request):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} requested command drift")
    if sample_count > 1 and not np.allclose(
        np.diff(row["timestamp_s"]), C.V3.BEHAVIOURAL_PROBE_AUTHORITY["physics_dt_s"],
        rtol=0.0, atol=1.0e-12,
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} timestamp cadence drift")
    per_act = C.V3.BEHAVIOURAL_PROBE_AUTHORITY["controller_policy_sampling"]["physics_samples_per_policy_act"]
    for member in ("controller_observation", "policy_output"):
        array = row[member]
        for start in range(0, sample_count, per_act):
            block = array[start:min(start + per_act, sample_count)]
            if not np.array_equal(block, np.repeat(block[:1], len(block), axis=0)):
                raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label}.{member} policy-act repetition drift")
    return row


def compare_terminated_behavioural_probe_traces(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two inclusive, variable-length partial probe traces."""

    import numpy as np

    a = _terminated_trace(left, "left terminated trace")
    b = _terminated_trace(right, "right terminated trace")
    count_equal = len(a["timestamp_s"]) == len(b["timestamp_s"])
    tip_equal = a["tip_sample_index"] == b["tip_sample_index"]
    if not count_equal:
        return {
            "sample_count_equal": False, "tip_sample_index_equal": tip_equal,
            "exact_member_equal": {}, "samplewise_max_abs_error": {},
            "samplewise_pass": {}, "termination_reason_equal": a["termination_reason"] == b["termination_reason"],
            "termination_flags_equal": a["termination_flags"] == b["termination_flags"],
            "stuck_equal": a["stuck"] == b["stuck"],
            "nonfinite_masks_equal": False,
            "nonfinite_bit_patterns_equal": False,
            "pass": False,
        }
    authority = C.V3.RESET_TRACE_PAIR_COMPARISON_AUTHORITY
    exact = {member: bool(np.array_equal(a[member], b[member])) for member in authority["exact_members"]}
    tolerance = authority["samplewise_tolerances"]
    maxima = {
        "base_pose_world_position_xyz": _finite_max_abs_error(
            a["base_pose_world"][:, :3], b["base_pose_world"][:, :3]
        ),
        "base_pose_world_quaternion_xyzw": _finite_max_abs_error(
            a["base_pose_world"][:, 3:], b["base_pose_world"][:, 3:]
        ),
        "base_twist_world": _finite_max_abs_error(
            a["base_twist_world"], b["base_twist_world"]
        ),
        "joint_position": _finite_max_abs_error(
            a["joint_position"], b["joint_position"]
        ),
        "joint_velocity": _finite_max_abs_error(
            a["joint_velocity"], b["joint_velocity"]
        ),
        "controller_observation": _finite_max_abs_error(
            a["controller_observation"], b["controller_observation"]
        ),
        "policy_output": _finite_max_abs_error(
            a["policy_output"], b["policy_output"]
        ),
    }
    sample_pass = {
        name: value <= float(tolerance.get(name, C.V3.BEHAVIOURAL_PROBE_AUTHORITY["controller_policy_samplewise_tolerance"]))
        for name, value in maxima.items()
    }
    terminal_equal = a["termination_reason"] == b["termination_reason"]
    flags_equal = a["termination_flags"] == b["termination_flags"]
    stuck_equal = a["stuck"] == b["stuck"]
    float_members = [
        member for member in C.V4_PROBE_TRACE_MEMBER_AUTHORITY
        if np.asarray(a[member]).dtype.kind in "fc"
    ]
    masks_equal = all(
        np.array_equal(np.isfinite(a[member]), np.isfinite(b[member]))
        for member in float_members
    )
    bit_patterns_equal = masks_equal and all(
        _nonfinite_bit_patterns_equal(a[member], b[member])
        for member in float_members
    )
    return {
        "sample_count_equal": count_equal,
        "tip_sample_index_equal": tip_equal,
        "exact_member_equal": exact,
        "samplewise_max_abs_error": maxima,
        "samplewise_pass": sample_pass,
        "termination_reason_equal": terminal_equal,
        "termination_flags_equal": flags_equal,
        "stuck_equal": stuck_equal,
        "nonfinite_masks_equal": masks_equal,
        "nonfinite_bit_patterns_equal": bit_patterns_equal,
        "pass": bool(
            tip_equal and all(exact.values()) and all(sample_pass.values())
            and terminal_equal and flags_equal and stuck_equal
            and masks_equal and bit_patterns_equal
        ),
    }


def compare_tipped_behavioural_probe_traces(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two inclusive tipped traces under the general partial authority."""

    _terminated_trace(left, "left tipped trace", require_tipped=True)
    _terminated_trace(right, "right tipped trace", require_tipped=True)
    return compare_terminated_behavioural_probe_traces(left, right)


def _probe_stuck_from_trace(trace: Mapping[str, Any]) -> bool:
    """Frozen V3 stuck formula over the maximal finite pose prefix."""

    import numpy as np

    requested = np.asarray(trace["requested_command"])
    poses = np.asarray(trace["base_pose_world"])
    finite_rows = np.isfinite(poses).all(axis=1)
    usable = poses[finite_rows]
    if not len(usable):
        return False

    def yaw(quaternion: Any) -> float:
        x, y, z, w = [float(value) for value in quaternion]
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    active = float(np.max(np.abs(requested))) > float(
        C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]["command_activity_threshold"]
    )
    translation = float(np.linalg.norm(usable[-1, :2] - usable[0, :2]))
    delta = yaw(usable[-1, 3:]) - yaw(usable[0, 3:])
    heading = abs(math.atan2(math.sin(delta), math.cos(delta)))
    authority = C.PHYSICAL_OUTCOME_AUTHORITY["stuck"]
    return bool(
        active
        and translation < float(authority["h3_translation_threshold_m"])
        and heading < float(authority["h3_heading_threshold_rad"])
    )


def validate_npz_archive_comment(value: bytes | str) -> str:
    if isinstance(value, bytes):
        try:
            observed = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicalGraphEdgeHandoffV4MetricsError("NPZ archive comment is not UTF-8") from exc
    elif isinstance(value, str):
        observed = value
    else:
        raise PhysicalGraphEdgeHandoffV4MetricsError("NPZ archive comment type drift")
    if observed != C.NPZ_ARCHIVE_COMMENT:
        raise PhysicalGraphEdgeHandoffV4MetricsError("NPZ archive comment drift")
    return observed


def build_persisted_array_evidence(
    *, shard_kind: str, shard_id: str, payload_file: Mapping[str, Any],
    arrays: Mapping[str, Any], reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    built = V3M.build_persisted_array_evidence(
        shard_kind=shard_kind, shard_id=shard_id, payload_file=payload_file,
        arrays=arrays, reopened_arrays=reopened_arrays,
    )
    result = project_v3_evidence_to_v4(built)
    result["schema"] = "physical_graph_edge_handoff_qualification_v4.persisted_array_evidence.v1"
    result["experiment_id"] = C.EXPERIMENT_ID
    return validate_persisted_array_evidence(result, reopened_arrays=reopened_arrays)


def validate_persisted_array_evidence(
    value: Any, *, reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.PERSISTED_ARRAY_EVIDENCE_FIELDS, "persisted_array_evidence")
    if row["schema"] != "physical_graph_edge_handoff_qualification_v4.persisted_array_evidence.v1" or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV4MetricsError("persisted array identity drift")
    projected = project_v4_evidence_to_v3(row)
    projected["schema"] = "physical_graph_edge_handoff_qualification_v3.persisted_array_evidence.v1"
    projected["experiment_id"] = C.V3_EXPERIMENT_ID
    try:
        V3M.validate_persisted_array_evidence(projected, reopened_arrays=reopened_arrays)
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc
    return row


def validate_snapshot_previous_applied_command_binding(
    snapshot_metadata: Mapping[str, Any], persisted_array_evidence: Mapping[str, Any],
    *, reopened_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    projected = project_v4_evidence_to_v3(persisted_array_evidence)
    projected["schema"] = "physical_graph_edge_handoff_qualification_v3.persisted_array_evidence.v1"
    projected["experiment_id"] = C.V3_EXPERIMENT_ID
    return V3M.validate_snapshot_previous_applied_command_binding(
        snapshot_metadata, projected, reopened_arrays=reopened_arrays
    )


def validate_state_snapshot_previous_applied_command_hashes(
    state_snapshot_index: Mapping[str, Any], raw_row_sha256s: Sequence[str],
) -> list[str]:
    """Cross-bind all 64 V4 snapshot rows to independently read raw bytes."""

    try:
        return V3M.validate_state_snapshot_previous_applied_command_hashes(
            project_v4_evidence_to_v3(state_snapshot_index), raw_row_sha256s
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Per-state disposition and material binding
# ---------------------------------------------------------------------------

def _termination_flags(value: Any, label: str, *, nullable: bool = False) -> dict[str, bool] | None:
    if value is None and nullable:
        return None
    row = _mapping(value, C.TERMINATION_FLAGS_FIELDS, label)
    for field in C.TERMINATION_FLAG_ORDER:
        _bool(row[field], f"{label}.{field}")
    return row


def _snapshot_identity(value: Any, label: str, *, probe_tipped: bool) -> dict[str, Any] | None:
    if value is None:
        return None
    row = _mapping(value, C.SNAPSHOT_IDENTITY_FIELDS, label)
    _sha(row["artifact_file_sha256"], f"{label}.artifact")
    _sha(row["snapshot_semantic_digest_v1"], f"{label}.semantic")
    if probe_tipped:
        if row["snapshot_behavioural_digest_v1"] is not None:
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe-tipped behavioural digest must be null")
    else:
        _sha(row["snapshot_behavioural_digest_v1"], f"{label}.behavioural")
    return row


def _teacher_criteria(value: Any) -> dict[str, bool] | None:
    if value is None:
        return None
    supplied = copy.deepcopy(dict(value)) if isinstance(value, Mapping) else None
    if supplied is None or set(supplied) not in (
        set(C.TEACHER_QUALIFICATION_COMPONENT_IDS), set(C.TEACHER_CRITERIA_FIELDS)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher criteria field drift")
    for key in C.TEACHER_QUALIFICATION_COMPONENT_IDS:
        _bool(supplied[key], f"teacher_criteria.{key}")
    teacher_valid = all(
        supplied[key] for key in C.TEACHER_QUALIFICATION_COMPONENT_IDS
        if key not in {"goal_reachable", "current_rgb_valid"}
    )
    if "teacher_valid" in supplied and supplied["teacher_valid"] is not teacher_valid:
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher_valid formula drift")
    supplied["teacher_valid"] = teacher_valid
    return supplied


def _ordered_failed_criteria(
    *, materialisation_corrupt: bool, state_nondeterministic: bool,
    initial_flags: Mapping[str, bool], probe_flags: Sequence[Mapping[str, bool]] | None,
    teacher_flags: Mapping[str, bool] | None, teacher_criteria: Mapping[str, bool] | None,
    unresolved_state_failure: bool,
) -> list[str]:
    observed: set[str] = set()
    if materialisation_corrupt:
        observed.add("state_materialisation_corrupt")
    if state_nondeterministic:
        observed.add("state_nondeterministic")
    if initial_flags["tipped"]:
        observed.add("initial_boundary_tipped")
    if any(initial_flags[key] for key in ("fall", "out_of_bounds", "nan")):
        observed.add("unresolved_state_failure")
    if probe_flags and any(flags["tipped"] for flags in probe_flags):
        observed.add("restoration_probe_tipped")
    if probe_flags and any(
        flags[key]
        for flags in probe_flags
        for key in ("fall", "out_of_bounds", "nan")
    ):
        observed.add("unresolved_state_failure")
    if teacher_flags and any(teacher_flags.values()):
        observed.add("teacher_terminated_unsafely")
    if teacher_criteria is not None:
        observed.update(key for key in C.TEACHER_QUALIFICATION_COMPONENT_IDS if not teacher_criteria[key])
    if unresolved_state_failure:
        observed.add("unresolved_state_failure")
    return [item for item in C.FAILED_CRITERION_IDS if item in observed]


def classify_state_disposition(
    *, initial_termination_flags: Mapping[str, Any],
    probe_trial_termination_flags: Sequence[Mapping[str, Any]] | None = None,
    teacher_termination_flags: Mapping[str, Any] | None = None,
    teacher_criteria: Mapping[str, Any] | None = None,
    materialisation_corrupt: bool = False, state_nondeterministic: bool = False,
    unresolved_state_failure: bool = False,
) -> tuple[str, list[str]]:
    """Return the exact singular disposition plus every failed criterion."""

    initial = _termination_flags(initial_termination_flags, "initial flags")
    assert initial is not None
    probe = None
    if probe_trial_termination_flags is not None:
        probe = [
            _termination_flags(item, f"probe flags[{index}]")
            for index, item in enumerate(_sequence(probe_trial_termination_flags, "probe flags"))
        ]
        if len(probe) != 2 or any(item is None for item in probe):
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe trial cardinality drift")
    teacher_flags = _termination_flags(teacher_termination_flags, "teacher flags", nullable=True)
    criteria = _teacher_criteria(teacher_criteria)
    materialisation_corrupt = _bool(materialisation_corrupt, "materialisation_corrupt")
    state_nondeterministic = _bool(state_nondeterministic, "state_nondeterministic")
    unresolved_state_failure = _bool(unresolved_state_failure, "unresolved_state_failure")
    if probe is not None and probe[0]["tipped"] != probe[1]["tipped"]:
        state_nondeterministic = True
    failed = _ordered_failed_criteria(
        materialisation_corrupt=materialisation_corrupt,
        state_nondeterministic=state_nondeterministic,
        initial_flags=initial, probe_flags=probe,
        teacher_flags=teacher_flags, teacher_criteria=criteria,
        unresolved_state_failure=unresolved_state_failure,
    )
    if materialisation_corrupt:
        return "STATE_MATERIALISATION_CORRUPT", failed
    if state_nondeterministic:
        return "STATE_NONDETERMINISTIC", failed
    if initial["tipped"]:
        return "INITIAL_BOUNDARY_TIPPED", failed
    if any(initial[key] for key in ("fall", "out_of_bounds", "nan")):
        return "UNRESOLVED_STATE_FAILURE", failed or ["unresolved_state_failure"]
    if probe is not None and all(item["tipped"] for item in probe):
        return "RESTORATION_PROBE_TIPPED", failed
    if probe is not None and any(
        item[key] for item in probe for key in ("fall", "out_of_bounds", "nan")
    ):
        return "UNRESOLVED_STATE_FAILURE", failed or ["unresolved_state_failure"]
    if teacher_flags is not None and any(teacher_flags.values()):
        return "TEACHER_TERMINATED_UNSAFELY", failed
    if criteria is None:
        return "UNRESOLVED_STATE_FAILURE", failed or ["unresolved_state_failure"]
    if not criteria["teacher_trace_contact_free"]:
        return "TEACHER_PHYSICS_CONTACT", failed
    if not all(criteria[key] for key in C.TEACHER_CROSSING_VALID_COMPONENT_IDS):
        return "TEACHER_CROSSING_INVALID", failed
    if not criteria["teacher_left_source_region"]:
        return "TEACHER_DID_NOT_LEAVE_SOURCE", failed
    if not criteria["teacher_positive_route_progress"]:
        return "TEACHER_NO_POSITIVE_PROGRESS", failed
    if not all(criteria.values()):
        return "UNRESOLVED_STATE_FAILURE", failed or ["unresolved_state_failure"]
    return "QUALIFIED", failed


def build_state_disposition_record(
    candidate_spec: Mapping[str, Any], *, stage_reached: str,
    initial_termination_flags: Mapping[str, Any],
    probe_trial_termination_flags: Sequence[Mapping[str, Any]] | None = None,
    teacher_termination_flags: Mapping[str, Any] | None = None,
    probe_tip_sample_indices: Sequence[int | None] | None = None,
    teacher_criteria: Mapping[str, Any] | None = None,
    executable_snapshot_exists: bool, teacher_executed: bool,
    snapshot_identity: Mapping[str, Any] | None = None,
    diagnostics_inventory: Sequence[str] = (), payload_member_inventory: Sequence[str] = (),
    materialisation_corrupt: bool = False, state_nondeterministic: bool = False,
    unresolved_state_failure: bool = False,
    reset_or_candidate_outcome_opened: bool = False,
) -> dict[str, Any]:
    specs = _CANDIDATE_SPECS
    if not isinstance(candidate_spec, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError("candidate spec is not mapping")
    spec = copy.deepcopy(dict(candidate_spec))
    try:
        pool_index = _SPEC_INDEX_BY_ID[spec.get("candidate_spec_id")]
    except (KeyError, TypeError) as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError("unknown candidate spec") from exc
    if C.canonical_json_bytes(spec) != C.canonical_json_bytes(specs[pool_index]):
        raise PhysicalGraphEdgeHandoffV4MetricsError("candidate spec drift")
    if stage_reached not in C.STATE_DISPOSITION_STAGES:
        raise PhysicalGraphEdgeHandoffV4MetricsError("stage_reached drift")
    initial = _termination_flags(initial_termination_flags, "initial flags")
    assert initial is not None
    probe = None if probe_trial_termination_flags is None else [
        _termination_flags(item, f"probe flags[{i}]")
        for i, item in enumerate(_sequence(probe_trial_termination_flags, "probe flags"))
    ]
    if probe is not None and (len(probe) != 2 or any(item is None for item in probe)):
        raise PhysicalGraphEdgeHandoffV4MetricsError("probe trial cardinality drift")
    tips = None
    if probe_tip_sample_indices is not None:
        tips = _sequence(probe_tip_sample_indices, "probe tip sample indices")
        if len(tips) != 2 or any(item is not None and (not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < C.V3.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES) for item in tips):
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe tip sample index drift")
    teacher_flags = _termination_flags(teacher_termination_flags, "teacher flags", nullable=True)
    criteria = _teacher_criteria(teacher_criteria)
    disposition, failed = classify_state_disposition(
        initial_termination_flags=initial,
        probe_trial_termination_flags=probe,
        teacher_termination_flags=teacher_flags,
        teacher_criteria=criteria,
        materialisation_corrupt=materialisation_corrupt,
        state_nondeterministic=state_nondeterministic,
        unresolved_state_failure=unresolved_state_failure,
    )
    executable = _bool(executable_snapshot_exists, "executable_snapshot_exists")
    teacher_done = _bool(teacher_executed, "teacher_executed")
    reset_opened = _bool(reset_or_candidate_outcome_opened, "reset_or_candidate_outcome_opened")
    if reset_opened:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification opened reset/candidate outcome")
    if stage_reached == "INITIAL_BOUNDARY":
        if (
            executable or teacher_done or probe is not None or tips is not None
            or teacher_flags is not None or criteria is not None
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "initial-boundary stage/evidence drift"
            )
    elif stage_reached == "RESTORATION_PROBE":
        if (
            any(initial.values()) or not executable or teacher_done
            or probe is None or tips is None or teacher_flags is not None
            or criteria is not None
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "restoration-probe stage/evidence drift"
            )
    else:
        if (
            any(initial.values()) or not executable or not teacher_done
            or probe is None or any(any(item.values()) for item in probe)
            or tips != [None, None] or teacher_flags is None
            or criteria is None
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher stage/evidence drift"
            )
    if teacher_done is not (criteria is not None):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher evidence/execution drift")
    if disposition in C.HARD_STOP_DISPOSITIONS:
        continuation = False
    else:
        continuation = True
    qualified = disposition == "QUALIFIED"
    if qualified and stage_reached != "COMPLETE":
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualified completion drift")
    if teacher_done and not qualified and stage_reached != "TEACHER_EXECUTION":
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "nonqualified teacher stage drift"
        )
    identity = _snapshot_identity(
        snapshot_identity, "snapshot_identity",
        probe_tipped=stage_reached == "RESTORATION_PROBE",
    )
    if executable is not (identity is not None):
        raise PhysicalGraphEdgeHandoffV4MetricsError("snapshot identity/existence drift")
    diagnostics = _sequence(diagnostics_inventory, "diagnostics_inventory")
    members = _sequence(payload_member_inventory, "payload_member_inventory")
    for label, values in (("diagnostics", diagnostics), ("payload members", members)):
        if any(not isinstance(item, str) or not item for item in values) or values != sorted(set(values)):
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} inventory drift")
    if (
        stage_reached == "INITIAL_BOUNDARY"
        and disposition not in C.HARD_STOP_DISPOSITIONS
        and set(members) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "initial-boundary payload inventory drift"
        )
    if (
        stage_reached == "INITIAL_BOUNDARY"
        and disposition not in C.HARD_STOP_DISPOSITIONS
        and set(diagnostics) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "initial-boundary diagnostics inventory drift"
        )
    result = C.attach_content_digest(
        {
            "schema": C.QUALIFICATION_DISPOSITIONS_SCHEMA,
            "experiment_id": C.EXPERIMENT_ID,
            "pool_index": pool_index,
            **{key: spec[key] for key in (
                "candidate_spec_id", "state_id", "scene_id", "episode_id", "graph_id",
                "family", "stratum_index", "variant_index", "procedural_seed",
            )},
            "canonical_spec_sha256": spec["canonical_spec_sha256"],
            "disposition": disposition,
            "qualified": qualified,
            "failed_criteria": failed,
            "stage_reached": stage_reached,
            "hard_stop": disposition in C.HARD_STOP_DISPOSITIONS,
            "continuation_authorized": continuation,
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
    if set(result) != C.STATE_DISPOSITION_FIELDS:
        raise PhysicalGraphEdgeHandoffV4MetricsError("state disposition output fields drift")
    return result


def validate_state_disposition_record(value: Any, *, expected_pool_index: int | None = None) -> dict[str, Any]:
    row = _mapping(value, C.STATE_DISPOSITION_FIELDS, "state disposition")
    C.validate_content_digest(row)
    pool = _nonnegative_int(row["pool_index"], "pool_index")
    if pool >= C.V3.PROSPECTIVE_POOL_COUNT or (expected_pool_index is not None and pool != expected_pool_index):
        raise PhysicalGraphEdgeHandoffV4MetricsError("state disposition pool index drift")
    spec = _CANDIDATE_SPECS[pool]
    materialisation = row["disposition"] == "STATE_MATERIALISATION_CORRUPT"
    nondeterministic = row["disposition"] == "STATE_NONDETERMINISTIC"
    unresolved = "unresolved_state_failure" in row["failed_criteria"]
    expected = build_state_disposition_record(
        spec, stage_reached=row["stage_reached"],
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
        materialisation_corrupt=materialisation,
        state_nondeterministic=nondeterministic,
        unresolved_state_failure=unresolved,
        reset_or_candidate_outcome_opened=row["reset_or_candidate_outcome_opened"],
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError("state disposition value drift")
    return row


def _file_binding(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.MATERIAL_FILE_BINDING_FIELDS, label)
    if not isinstance(row["path"], str) or not row["path"] or row["path"].startswith("/") or ".." in row["path"].split("/"):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} path drift")
    if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} bytes drift")
    _sha(row["sha256"], f"{label}.sha256")
    return row


def build_qualification_disposition_row(
    state_disposition: Mapping[str, Any], *, material_metadata_binding: Mapping[str, Any],
    material_payload_binding: Mapping[str, Any], persisted_array_evidence_sha256: str,
) -> dict[str, Any]:
    state = validate_state_disposition_record(state_disposition)
    result = copy.deepcopy(state)
    result.pop("content_digest")
    result.update(
        {
            "material_metadata_binding": _file_binding(material_metadata_binding, "material metadata"),
            "material_payload_binding": _file_binding(material_payload_binding, "material payload"),
            "persisted_array_evidence_sha256": _sha(persisted_array_evidence_sha256, "persisted evidence"),
        }
    )
    return C.attach_content_digest(result)


def validate_qualification_disposition_row(value: Any, *, expected_pool_index: int | None = None) -> dict[str, Any]:
    row = _mapping(value, C.QUALIFICATION_DISPOSITION_ROW_FIELDS, "qualification disposition row")
    C.validate_content_digest(row)
    core = {key: copy.deepcopy(row[key]) for key in C.STATE_DISPOSITION_FIELDS}
    core_without = dict(core)
    core_without.pop("content_digest")
    core["content_digest"] = hashlib.sha256(C.canonical_json_bytes(core_without)[:-1]).hexdigest()
    validate_state_disposition_record(core, expected_pool_index=expected_pool_index)
    _file_binding(row["material_metadata_binding"], "material metadata")
    _file_binding(row["material_payload_binding"], "material payload")
    _sha(row["persisted_array_evidence_sha256"], "persisted evidence")
    return row


def build_qualification_state_dispositions_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    rows = [
        validate_qualification_disposition_row(item, expected_pool_index=index)
        for index, item in enumerate(_sequence(records, "qualification disposition records"))
    ]
    if len(rows) != C.V3.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification disposition cardinality drift")
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )


def validate_qualification_state_dispositions_jsonl(value: bytes | str) -> list[dict[str, Any]]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    if not isinstance(raw, bytes) or not raw.endswith(b"\n") or b"\r" in raw:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification JSONL byte format drift")
    lines = raw.splitlines()
    if len(lines) != C.V3.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification JSONL row count drift")
    try:
        rows = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification JSONL parse failed") from exc
    expected = build_qualification_state_dispositions_jsonl(rows)
    if raw != expected:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification JSONL is not canonical")
    return [copy.deepcopy(row) for row in rows]


def validate_state_material_payload(
    state_disposition: Mapping[str, Any], persisted_array_evidence: Mapping[str, Any],
    *, reopened_arrays: Mapping[str, Any],
) -> dict[str, Any]:
    state = validate_state_disposition_record(state_disposition)
    persisted = validate_persisted_array_evidence(persisted_array_evidence, reopened_arrays=reopened_arrays)
    names = sorted(str(name) for name in reopened_arrays)
    if names != state["payload_member_inventory"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError("state/payload member inventory drift")
    if (
        state["stage_reached"] == "INITIAL_BOUNDARY"
        and state["disposition"] not in C.HARD_STOP_DISPOSITIONS
    ):
        import numpy as np

        rows = {item["member"]: item for item in persisted["arrays"]}
        if set(rows) != set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY):
            raise PhysicalGraphEdgeHandoffV4MetricsError("initial payload member drift")
        for member, authority in C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY.items():
            if rows[member]["dtype_str"] != authority["dtype_str"] or rows[member]["shape"] != authority["shape"]:
                raise PhysicalGraphEdgeHandoffV4MetricsError(f"initial payload extent drift: {member}")
        spec = _CANDIDATE_SPECS[state["pool_index"]]
        spawn_x, spawn_y, spawn_yaw = spec["geometry"]["spawn_se2_world"]
        expected_pose = np.asarray(
            [spawn_x, spawn_y, 0.375, 0.0, 0.0, math.sin(spawn_yaw / 2.0), math.cos(spawn_yaw / 2.0)],
            dtype=np.float64,
        )
        if not np.array_equal(np.asarray(reopened_arrays["intended_base_pose_world"]), expected_pose):
            raise PhysicalGraphEdgeHandoffV4MetricsError("intended initial pose formula drift")
        expected_flags = np.asarray(
            [int(state["initial_termination_flags"][key]) for key in C.TERMINATION_FLAG_ORDER],
            dtype=np.uint8,
        )
        if not np.array_equal(np.asarray(reopened_arrays["termination_flags"]), expected_flags):
            raise PhysicalGraphEdgeHandoffV4MetricsError("initial termination flags binding drift")
        observed = {
            member: np.asarray(reopened_arrays[member])
            for member in C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY
        }
        _terminal_nonfinite_masks(
            observed,
            state["initial_termination_flags"],
            label="initial boundary",
            allowed_members=C.TERMINAL_NONFINITE_INITIAL_MEMBER_IDS,
            require_trace_axis=False,
        )
        for member in (
            "intended_base_pose_world", "previous_applied_command",
        ):
            if not bool(np.isfinite(observed[member]).all()):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    f"initial boundary nonfinite diagnostic: {member}"
                )
        if not bool(np.isin(observed["physics_contact"], [0, 1]).all()):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "initial boundary physics contact is not binary"
            )
        derived_flags = _teacher_termination_flags_from_pose(
            reopened_arrays["base_pose_world"]
        )
        if derived_flags != state["initial_termination_flags"]:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "initial boundary flags differ from observed pose"
            )
        for member in ("sim_time_ns", "episode_step", "command_ticks", "policy_steps"):
            scalar = int(np.asarray(reopened_arrays[member])[0])
            if scalar < 0:
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    f"initial boundary negative counter: {member}"
                )
    return {"state_disposition": state, "persisted_array_evidence": persisted}


def _canonical_array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = C.canonical_json_bytes(
        {"shape": list(array.shape), "dtype": array.dtype.str, "layout": "C"}
    )[:-1]
    return hashlib.sha256(header + b"\x00" + array.tobytes(order="C")).hexdigest()


def _wrap_angle_v1(value: float) -> float:
    """Exact frozen V1 arithmetic order, including modulo rounding."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _pose_roll_pitch_yaw_xyzw(pose: Any) -> tuple[float, float, float]:
    values = [float(item) for item in pose]
    if len(values) != 7:
        raise PhysicalGraphEdgeHandoffV4MetricsError("base pose extent drift")
    qx, qy, qz, qw = values[3:7]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return float(roll), float(pitch), _wrap_angle_v1(yaw)


def _termination_roll_pitch_xyzw(pose: Any) -> tuple[float, float]:
    """Exact production boundary roll/pitch arithmetic."""

    values = [float(item) for item in pose]
    qx, qy, qz, qw = values[3:7]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = (
        math.copysign(math.pi / 2.0, sinp)
        if abs(sinp) >= 1.0 else math.asin(sinp)
    )
    return float(roll), float(pitch)


def _teacher_termination_flags_from_pose(pose: Any) -> dict[str, bool]:
    import numpy as np

    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (7,):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher terminal base pose extent drift"
        )
    authority = C.V4_TEACHER_TERMINATION_DERIVATION_AUTHORITY
    finite = bool(np.isfinite(values).all())
    # Preserve the production predicate order and independence even when one
    # coordinate is nonfinite: another finite coordinate may simultaneously
    # satisfy fall/out-of-bounds, and a finite quaternion may also be tipped.
    # The `nan` flag therefore does not erase any co-occurring physical flag.
    roll, pitch = _termination_roll_pitch_xyzw(values)
    (xmin, ymin), (xmax, ymax) = authority["world_bounds_xy_m"]
    pad = float(authority["out_of_bounds_pad_m"])
    fall = bool(values[2] < float(authority["fall_z_threshold_m"]))
    out_of_bounds = bool(
        values[0] < xmin - pad or values[0] > xmax + pad
        or values[1] < ymin - pad or values[1] > ymax + pad
    )
    tipped = bool(
        max(abs(roll), abs(pitch)) > float(authority["tip_threshold_rad"])
    )
    return {
        "fall": fall,
        "out_of_bounds": out_of_bounds,
        "tipped": tipped,
        "nan": not finite,
    }


def _polyline_length(polyline: Sequence[Sequence[Any]]) -> float:
    rows = _sequence(polyline, "graph route polyline")
    return float(sum(
        math.hypot(float(right[0]) - float(left[0]),
                   float(right[1]) - float(left[1]))
        for left, right in zip(rows, rows[1:])
    ))


def _expected_registered_graph(
    spec: Mapping[str, Any], *, positive_route_progress: bool,
    competing_port_entered: bool,
) -> dict[str, Any]:
    geometry = spec["geometry"]
    source = geometry["source_node"]
    target = geometry["target_node"]
    nodes: list[dict[str, Any]] = [
        {
            "node_id": "source", "node_kind": "SOURCE",
            "centre_world": copy.deepcopy(source["centre_world"]),
            "boundary_polygon_world": copy.deepcopy(
                source["boundary_polygon_world"]
            ),
        },
        {
            "node_id": "target", "node_kind": "TARGET",
            "centre_world": copy.deepcopy(target["centre_world"]),
            "boundary_polygon_world": copy.deepcopy(
                target["boundary_polygon_world"]
            ),
        },
    ]
    selected = geometry["selected_directed_edge"]
    route = geometry["teacher_route_polyline_world"]
    edges: list[dict[str, Any]] = [
        {
            "edge_id": "selected-edge", "source_node_id": "source",
            "target_node_id": "target",
            "port_label": str(selected["route_direction"]),
            "route_polyline_world": copy.deepcopy(route),
            "edge_length_m": _polyline_length(route),
            "oracle_reachable": True, "physically_executable": True,
            "opening_segment_world": copy.deepcopy(
                selected["opening_segment_world"]
            ),
            "opening_normal_world": copy.deepcopy(
                selected["opening_normal_world"]
            ),
            "edge_kind": "SELECTED",
        }
    ]
    for index, edge in enumerate(geometry["competing_directed_edges"]):
        target_id = f"competing-node-{index}"
        polygon = edge["edge_region_polygon_world"]
        centre = [
            sum(float(point[axis]) for point in polygon) / len(polygon)
            for axis in (0, 1)
        ]
        nodes.append(
            {
                "node_id": target_id, "node_kind": "COMPETING",
                "centre_world": centre,
                "boundary_polygon_world": copy.deepcopy(polygon),
            }
        )
        midpoint = [
            sum(float(point[axis]) for point in edge["opening_segment_world"])
            / 2.0
            for axis in (0, 1)
        ]
        competitor_route = [
            copy.deepcopy(source["centre_world"]), midpoint, centre,
        ]
        edges.append(
            {
                "edge_id": str(edge["edge_id"]),
                "source_node_id": "source", "target_node_id": target_id,
                "port_label": str(edge.get("boundary_side", edge["edge_id"])),
                "route_polyline_world": competitor_route,
                "edge_length_m": _polyline_length(competitor_route),
                "oracle_reachable": True, "physically_executable": True,
                "opening_segment_world": copy.deepcopy(
                    edge["opening_segment_world"]
                ),
                "opening_normal_world": copy.deepcopy(
                    edge["opening_normal_world"]
                ),
                "edge_kind": "COMPETING",
            }
        )
    return {
        "graph_id": str(spec["graph_id"]),
        "source_node_id": "source", "target_node_id": "target",
        "directed_edge_id": "selected-edge", "nodes": nodes, "edges": edges,
        "goal_reachable": True, "graph_edge_physically_executable": True,
        "teacher_positive_route_progress": positive_route_progress,
        "teacher_competing_port_entered": competing_port_entered,
    }


def _registered_crossing_projection(
    raw: Mapping[str, Any], edge: Mapping[str, Any], *, include_selected: bool,
) -> dict[str, Any]:
    point = _binary64_vector(
        raw["crossing_point_world_xy"], 2, "registered crossing point",
    )
    planar = canonical_v1_planar_segment_projection(
        point, edge["opening_segment_world"],
    )
    result: dict[str, Any] = {
        "sample_before": int(raw["crossing_sample_before"]),
        "sample_after": int(raw["crossing_sample_after"]),
        "fraction": float(raw["crossing_fraction"]),
        "normal_dot_displacement_m": float(
            raw["directed_normal_displacement_m"]
        ),
        "lateral_coordinate_m": planar["lateral_coordinate_m"],
        "lateral_fraction": float(raw["lateral_fraction"]),
        "point_world": point,
    }
    if include_selected:
        return {
            "edge_id": str(raw["edge_id"]),
            "is_selected_edge": bool(raw["is_selected_edge"]),
            **result,
        }
    return {"edge_id": str(edge["edge_id"]), **result}


def _teacher_crossing_reduction(
    poses: Any, target: Any, spec: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, bool]:
    import numpy as np

    geometry = spec["geometry"]
    selected = geometry["selected_directed_edge"]
    competing = geometry["competing_directed_edges"]
    positions = np.asarray(poses, dtype=np.float64)[:, :2].tolist()
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    first = first_registered_port_crossing(
        positions, selected, competing, tolerance_m=tolerance,
    ) if len(positions) >= 2 else None

    competing_rows: list[tuple[int, float, str, dict[str, Any]]] = []
    for edge in competing:
        candidate = first_registered_port_crossing(
            positions, edge, [], tolerance_m=tolerance,
        ) if len(positions) >= 2 else None
        if candidate is not None:
            projection = _registered_crossing_projection(
                candidate, edge, include_selected=False,
            )
            competing_rows.append(
                (
                    projection["sample_after"], projection["fraction"],
                    projection["edge_id"], projection,
                )
            )
    competing_crossing = min(competing_rows)[3] if competing_rows else None
    if first is None:
        return (
            None,
            "teacher trace never crosses the canonical directed port",
            competing_crossing,
            competing_crossing is not None,
        )
    if not bool(first["is_selected_edge"]):
        return (
            None,
            "teacher enters a competing physical port first",
            competing_crossing,
            True,
        )
    crossing = _registered_crossing_projection(
        first, selected, include_selected=True,
    )
    opening = np.asarray(selected["opening_segment_world"], dtype=np.float64)
    normal = np.asarray(selected["opening_normal_world"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    midpoint = opening.mean(axis=0)
    after = crossing["sample_after"]
    beyond = (np.asarray(poses, dtype=np.float64)[after:, :2] - midpoint) @ normal
    count = 0
    for value in beyond:
        if float(value) <= tolerance:
            break
        count += 1
    target_hits = np.flatnonzero(np.asarray(target, dtype=np.uint8)[after:] != 0)
    target_early = bool(
        len(target_hits)
        and int(target_hits[0]) + 1 < C.PORT_DWELL_PHYSICS_SAMPLES
    )
    if count < C.PORT_DWELL_PHYSICS_SAMPLES and not target_early:
        return (
            None,
            "teacher did not remain beyond the port for 100 physics samples",
            competing_crossing,
            competing_crossing is not None,
        )
    heading = math.atan2(float(normal[1]), float(normal[0]))
    crossing.update(
        {
            "port_world": [*crossing["point_world"], heading],
            "competing_crossing_before": False,
            "sustained_beyond_samples": int(count),
            "sustained_or_target_reached": True,
        }
    )
    selected_order = (
        crossing["sample_after"], crossing["fraction"],
        str(selected["edge_id"]),
    )
    competitor_first = bool(
        competing_crossing is not None
        and (
            competing_crossing["sample_after"],
            competing_crossing["fraction"],
            competing_crossing["edge_id"],
        ) <= selected_order
    )
    return crossing, None, competing_crossing, competitor_first


def reduce_qualification_teacher_trace(
    metadata: Mapping[str, Any], reopened_arrays: Mapping[str, Any], *,
    expected_pool_index: int | None = None,
) -> dict[str, Any]:
    """Independently reduce every qualification-affecting teacher predicate.

    The only inputs are the registered pool specification and the exact
    reopened material arrays.  Persisted teacher and graph summaries are then
    required to match this projection byte-for-byte.
    """

    import numpy as np

    if not isinstance(metadata, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher material metadata is not mapping"
        )
    pool = _nonnegative_int(metadata.get("pool_index"), "teacher pool index")
    if (
        pool >= len(_CANDIDATE_SPECS)
        or expected_pool_index is not None and pool != expected_pool_index
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher pool identity drift")
    spec = _CANDIDATE_SPECS[pool]
    if C.canonical_json_bytes(metadata.get("candidate_spec")) != C.canonical_json_bytes(spec):
        raise PhysicalGraphEdgeHandoffV4MetricsError("teacher candidate spec drift")
    traces: dict[str, Any] = {}
    byte_members = {
        "physics_contact", "source_region_member", "edge_region_member",
        "target_region_member",
    }
    tails = {
        "timestamp_s": (), "base_pose_world": (7,), "base_twist_world": (6,),
        "joint_position": (12,), "joint_velocity": (12,),
        "applied_command": (3,), "requested_command": (3,),
        "physics_contact": (), "source_region_member": (),
        "edge_region_member": (), "target_region_member": (),
    }
    sample_count: int | None = None
    for member in C.TEACHER_TRACE_MEMBER_ORDER:
        name = f"teacher__{member}"
        if name not in reopened_arrays:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
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
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"teacher raw trace extent drift: {member}"
            )
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif int(array.shape[0]) != sample_count:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher raw trace sample alignment drift"
            )
        if member in byte_members:
            if not bool(np.isin(array, [0, 1]).all()):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
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
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher raw trace sample count drift"
        )
    termination_flags = _teacher_termination_flags_from_pose(
        traces["base_pose_world"][-1]
    )
    _terminal_nonfinite_masks(
        traces,
        termination_flags,
        label="teacher trace",
        allowed_members=C.TERMINAL_NONFINITE_TRACE_MEMBER_IDS,
        require_trace_axis=True,
    )
    if sample_count > 1 and not bool(np.allclose(
        np.diff(traces["timestamp_s"]), C.TEACHER_TRACE_DT_S,
        rtol=0.0, atol=1.0e-12,
    )):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
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
            anchor.dtype.str != "<f8" or list(anchor.shape) != [7]
            or not bool(np.isfinite(anchor).all())
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "first-sample nan teacher lacks finite captured pose anchor"
            )
        science_poses = np.ascontiguousarray(anchor.reshape(1, 7))
    memberships = {
        "source_region_member": geometry["source_node"]["boundary_polygon_world"],
        "edge_region_member": geometry["selected_directed_edge"]["edge_region_polygon_world"],
        "target_region_member": geometry["target_node"]["boundary_polygon_world"],
    }
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    science_memberships: dict[str, Any] = {}
    for member, polygon in memberships.items():
        expected_all = np.asarray(
            [
                int(
                    point_in_polygon_inclusive(
                        pose[:2].tolist(), polygon, tolerance_m=tolerance,
                    )
                )
                if bool(np.isfinite(pose[:2]).all()) else 0
                for pose in poses
            ],
            dtype=np.uint8,
        )
        if not np.array_equal(traces[member], expected_all):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"teacher {member} differs from registered geometry"
            )
        if finite_stop:
            science_memberships[member] = expected_all[:finite_stop]
        else:
            science_memberships[member] = np.asarray(
                [
                    int(point_in_polygon_inclusive(
                        science_poses[0, :2].tolist(), polygon,
                        tolerance_m=tolerance,
                    ))
                ],
                dtype=np.uint8,
            )
    crossing, crossing_error, competing_crossing, competing = (
        _teacher_crossing_reduction(
            science_poses, science_memberships["target_region_member"], spec,
        )
    )
    selected_opening = np.asarray(
        geometry["selected_directed_edge"]["opening_segment_world"],
        dtype=np.float64,
    )
    opening_projection = canonical_v1_planar_segment_projection(
        selected_opening[0], selected_opening,
    )
    opening_width = opening_projection["segment_width_m"]
    midpoint = np.asarray(
        opening_projection["midpoint_world_xy"], dtype=np.float64,
    )
    distances = np.linalg.norm(
        science_poses[:, :2] - midpoint[None, :], axis=1
    )
    route_progress = float(distances[0] - distances.min())
    positive = route_progress > 0.0
    contact_free = not bool(traces["physics_contact"].any())
    source = science_memberships["source_region_member"]
    left_source = bool((source == 0).any())
    exits = np.flatnonzero((source[:-1] != 0) & (source[1:] == 0)) + 1
    first_exit = int(exits[0]) if len(exits) else None
    reached_target = bool(science_memberships["target_region_member"].any())
    goal_reachable = True
    physically_executable = True
    graph = _expected_registered_graph(
        spec, positive_route_progress=positive,
        competing_port_entered=competing_crossing is not None,
    )
    if C.canonical_json_bytes(metadata.get("graph")) != C.canonical_json_bytes(graph):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "teacher graph differs from registered geometry"
        )
    criteria = {
        "goal_reachable": goal_reachable,
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
        "graph_edge_physically_executable": physically_executable,
    }
    teacher_valid = bool(
        crossing is not None and contact_free and left_source and positive
        and not competing and physically_executable
    )
    criteria["teacher_valid"] = teacher_valid
    state = validate_state_disposition_record(
        metadata.get("state_disposition"), expected_pool_index=pool,
    )
    if (
        state["teacher_criteria"] != criteria
        or state["teacher_termination_flags"] != termination_flags
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
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
            end_pose[:2], selected_opening,
        )["lateral_coordinate_m"]
    )
    endpoint_angular = abs(_wrap_angle_v1(
        end_yaw - math.atan2(float(normal[1]), float(normal[0]))
    ))
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
        crossing_velocity_heading = math.atan2(
            crossing_velocity[1], crossing_velocity[0]
        ) if math.hypot(*crossing_velocity) > 0.0 else None
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
        "first_crossing_sample_index": (
            None if crossing is None else crossing["sample_after"]
        ),
        "crossing_segment_fraction": (
            None if crossing is None else crossing["fraction"]
        ),
        "crossed_directed_port": crossing is not None,
        "positive_route_progress": positive,
        "route_progress_m": route_progress,
        "crossing_directed_normal_dot": (
            None if crossing is None else crossing["normal_dot_displacement_m"]
        ),
        "crossing_lateral_fraction": (
            None if crossing is None else (
                float(crossing["lateral_coordinate_m"]) / opening_width + 0.5
            )
        ),
        "crossing_velocity_world_xy": crossing_velocity,
        "crossing_velocity_heading_world_rad": crossing_velocity_heading,
        "beyond_port_consecutive_physics_samples": dwell_count,
        "target_entered_before_dwell_complete": target_early,
        "competing_port_entered": competing,
        "reached_target_node": reached_target,
        "first_target_entry_sample_index": first_target_index,
        "where_reached": (
            "TARGET_NODE" if reached_target else "BEYOND_DIRECTED_PORT"
        ),
        "endpoint_lateral_error_m": endpoint_lateral,
        "endpoint_angular_error_rad": endpoint_angular,
        "successor_viable": successor,
        "stuck": stuck,
        "teacher_valid": teacher_valid,
        "goal_reachable": goal_reachable,
        "directed_port_defined": crossing is not None,
        "graph_edge_physically_executable": physically_executable,
    }
    if set(record_projection) != set(C.V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "raw teacher-record projection field drift"
        )
    result = {
        "pool_index": pool,
        "candidate_spec_id": spec["candidate_spec_id"],
        "sample_count": sample_count,
        "goal_reachable": goal_reachable,
        "graph_edge_physically_executable": physically_executable,
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
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "raw teacher reduction field drift"
        )
    return result


def validate_qualification_teacher_raw_evidence(
    metadata: Mapping[str, Any], reopened_arrays: Mapping[str, Any], *,
    expected_pool_index: int | None = None,
    teacher_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one raw teacher shard and optionally its compact index row."""

    reduction = reduce_qualification_teacher_trace(
        metadata, reopened_arrays, expected_pool_index=expected_pool_index,
    )
    if teacher_record is not None:
        record = _mapping(
            teacher_record, C.V4_TEACHER_RECORD_FIELDS,
            "V4 sparse teacher record",
        )
        if (
            record["qualification_pool_index"] != reduction["pool_index"]
            or record["candidate_spec_id"] != reduction["candidate_spec_id"]
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher record/raw shard identity drift"
            )
        projection = reduction["teacher_record_raw_projection"]
        if any(
            C.canonical_json_bytes(record[field])
            != C.canonical_json_bytes(projection[field])
            for field in C.V4_RAW_TEACHER_RECORD_PROJECTION_FIELDS
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "teacher record differs from raw shard reduction"
            )
    return reduction


def _trace_member_manifest(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    return [
        {
            "member": member,
            "dtype_str": np.asarray(trace[member]).dtype.str,
            "shape": list(np.asarray(trace[member]).shape),
            "array_bytes_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(trace[member])).tobytes(order="C")
            ).hexdigest(),
        }
        for member in sorted(C.V4_PROBE_TRACE_MEMBER_AUTHORITY)
    ]


def _validate_probe_material(
    metadata: Mapping[str, Any], arrays: Mapping[str, Any], *, completed: bool,
) -> dict[str, Any]:
    import numpy as np

    semantic = V3M.validate_snapshot_semantic_evidence(metadata["snapshot_semantic_evidence"])
    semantic_array = np.asarray(arrays.get("snapshot_semantic_bytes"))
    if semantic_array.dtype.str != "|u1" or semantic_array.ndim != 1:
        raise PhysicalGraphEdgeHandoffV4MetricsError("snapshot semantic member extent drift")
    V3M.validate_snapshot_semantic_bytes(semantic_array.tobytes(order="C"), semantic)
    identity = _snapshot_identity(metadata["snapshot_identity"], "material snapshot identity", probe_tipped=not completed)
    assert identity is not None
    payload = np.asarray(arrays.get("snapshot_payload_bytes"))
    if payload.dtype.str != "|u1" or payload.ndim != 1 or hashlib.sha256(payload.tobytes(order="C")).hexdigest() != identity["artifact_file_sha256"] or semantic["snapshot_semantic_digest_v1"] != identity["snapshot_semantic_digest_v1"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError("snapshot identity/raw material drift")
    probe = _mapping(metadata["behavioural_probe"], C.BEHAVIOURAL_PROBE_METADATA_FIELDS, "behavioural probe")
    if probe["completed"] is not completed:
        raise PhysicalGraphEdgeHandoffV4MetricsError("behavioural probe completion drift")
    trials = _sequence(probe["trials"], "behavioural probe trials")
    if len(trials) != 2:
        raise PhysicalGraphEdgeHandoffV4MetricsError("behavioural probe trial count drift")
    traces = []
    for trial_index, raw_trial in enumerate(trials):
        trial = _mapping(raw_trial, C.PROBE_TIPPED_TRIAL_FIELDS, f"probe trial[{trial_index}]")
        if trial["trial_index"] != trial_index or type(trial["tipped"]) is not bool or type(trial["contact"]) is not bool or type(trial["stuck"]) is not bool or type(trial["final_executable_snapshot_exists"]) is not bool:
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe trial scalar drift")
        flags = _termination_flags(trial["termination_flags"], f"probe trial[{trial_index}].flags")
        assert flags is not None
        trace = {
            member: np.ascontiguousarray(np.asarray(arrays[f"probe__{trial_index}__{member}"]))
            for member in C.V4_PROBE_TRACE_MEMBER_AUTHORITY
        }
        if _teacher_termination_flags_from_pose(trace["base_pose_world"][-1]) != flags:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "probe termination flags differ from inclusive terminal pose"
            )
        if trial["trace_member_manifest"] != _trace_member_manifest(trace):
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe trace manifest drift")
        if trial["contact"] is not bool(np.asarray(trace["physics_contact"]).any()):
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe contact projection drift")
        digest_member = f"probe__{trial_index}__final_snapshot_semantic_digest_bytes"
        if completed:
            digest_array = np.asarray(arrays.get(digest_member))
            if digest_array.dtype.str != "|u1" or list(digest_array.shape) != [32]:
                raise PhysicalGraphEdgeHandoffV4MetricsError("probe final semantic digest extent drift")
            digest = digest_array.tobytes(order="C").hex()
            if trial["final_snapshot_semantic_digest_v1"] != digest or trial["final_executable_snapshot_exists"] is not True or trial["tip_sample_index"] is not None or trial["tipped"] is not False or any(flags.values()) or trial["termination_reason"] != "H3_COMPLETE":
                raise PhysicalGraphEdgeHandoffV4MetricsError("completed probe terminal projection drift")
            trace.update(
                {
                    "termination_reason": trial["termination_reason"],
                    "stuck": trial["stuck"],
                    "final_snapshot_semantic_digest_v1": digest,
                }
            )
            observed_digest = V3M.snapshot_behavioural_digest(trace)
            if trial["snapshot_behavioural_digest_v1"] != observed_digest:
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "per-trial behavioural digest drift"
                )
            if trial_index == 0 and identity["snapshot_behavioural_digest_v1"] != observed_digest:
                raise PhysicalGraphEdgeHandoffV4MetricsError("trial-0 behavioural identity drift")
        else:
            if (
                digest_member in arrays
                or trial["final_snapshot_semantic_digest_v1"] is not None
                or trial["snapshot_behavioural_digest_v1"] is not None
                or trial["final_executable_snapshot_exists"] is not False
                or trial["tip_sample_index"] is None
                or trial["tipped"] is not flags["tipped"]
            ):
                raise PhysicalGraphEdgeHandoffV4MetricsError("tipped probe fabricated final evidence")
            trace.update(
                {
                    "termination_reason": trial["termination_reason"],
                    "stuck": trial["stuck"],
                    "termination_flags": flags,
                    "tip_sample_index": trial["tip_sample_index"],
                }
            )
        if trial["stuck"] is not _probe_stuck_from_trace(trace):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "probe stuck projection differs from raw trace"
            )
        traces.append(trace)
    expected_comparison = (
        V3M.compare_behavioural_probe_traces(traces[0], traces[1])
        if completed else compare_terminated_behavioural_probe_traces(traces[0], traces[1])
    )
    if C.canonical_json_bytes(probe["trial_pair_comparison"]) != C.canonical_json_bytes(expected_comparison) or expected_comparison["pass"] is not True:
        raise PhysicalGraphEdgeHandoffV4MetricsError("behavioural probe pair comparison drift")
    return {"identity": identity, "trials": trials, "traces": traces}


def validate_qualification_material_shard(
    metadata: Any, *, reopened_arrays: Mapping[str, Any], expected_pool_index: int | None = None,
) -> dict[str, Any]:
    """Validate one complete atomic V4 qualification metadata/NPZ pair."""

    import numpy as np

    if not isinstance(metadata, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification metadata is not mapping")
    disposition = metadata.get("disposition")
    if disposition == "INITIAL_BOUNDARY_TIPPED" or (
        metadata.get("stage_reached") == "INITIAL_BOUNDARY" and disposition == "UNRESOLVED_STATE_FAILURE"
    ):
        fields = C.INITIAL_TIPPED_METADATA_FIELDS
    elif disposition == "RESTORATION_PROBE_TIPPED" or (
        metadata.get("stage_reached") == "RESTORATION_PROBE" and disposition == "UNRESOLVED_STATE_FAILURE"
    ):
        fields = C.PROBE_TIPPED_METADATA_FIELDS
    else:
        fields = C.TEACHER_TERMINAL_METADATA_FIELDS
    row = _mapping(metadata, fields, "qualification material metadata")
    C.validate_content_digest(row)
    if row["schema"] != "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1" or row["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification material identity drift")
    source_freeze_commit = _commit(
        row["source_freeze_commit"],
        "qualification material source freeze commit",
    )
    runtime_contract_content_digest = _sha(
        row["runtime_contract_content_digest"],
        "qualification material runtime contract digest",
    )
    if (
        source_freeze_commit == C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or runtime_contract_content_digest
        == C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification material binds invalidated pre-correction runtime"
        )
    pool = _nonnegative_int(row["pool_index"], "qualification material pool")
    if pool >= len(_CANDIDATE_SPECS) or (expected_pool_index is not None and pool != expected_pool_index) or row["candidate_spec"] != _CANDIDATE_SPECS[pool] or row["candidate_spec_sha256"] != _CANDIDATE_SPECS[pool]["canonical_spec_sha256"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError("qualification material candidate identity drift")
    state = validate_state_disposition_record(row["state_disposition"], expected_pool_index=pool)
    if any(row[field] != state[field] for field in ("disposition", "qualified", "stage_reached", "executable_snapshot_exists", "teacher_executed", "reset_or_candidate_outcome_opened")):
        raise PhysicalGraphEdgeHandoffV4MetricsError("outer/state disposition cross-link drift")
    if row["stage_reached"] in {"INITIAL_BOUNDARY", "RESTORATION_PROBE"} and (
        row["qualified"] is not False
        or row["rejection_reason"] != row["disposition"]
        or row["rejection_components"] != [row["disposition"]]
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "pre-teacher rejection projection drift"
        )
    persisted = validate_persisted_array_evidence(row["persisted_array_evidence"], reopened_arrays=reopened_arrays)
    if row["payload"] != {"role": "material_shard_payload", **persisted["payload_file"], "kind": "npz"}:
        raise PhysicalGraphEdgeHandoffV4MetricsError("material payload binding drift")
    validate_state_material_payload(state, persisted, reopened_arrays=reopened_arrays)
    if disposition == "INITIAL_BOUNDARY_TIPPED" or row["stage_reached"] == "INITIAL_BOUNDARY":
        boundary = _mapping(row["boundary_evidence"], C.BOUNDARY_EVIDENCE_FIELDS, "boundary evidence")
        if boundary != {
            "intended_pose_representation": "xyz_plus_quaternion_xyzw",
            "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
            "available_simulator_diagnostics": sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
            "previous_applied_command_sha256": hashlib.sha256(np.asarray(reopened_arrays["previous_applied_command"]).tobytes(order="C")).hexdigest(),
            "previous_applied_command_dtype": "<f8",
            "previous_applied_command_shape": [3],
        }:
            raise PhysicalGraphEdgeHandoffV4MetricsError("boundary evidence/raw array drift")
        if row["snapshot"] is not None or row["graph"] is not None or row["teacher"] is not None:
            raise PhysicalGraphEdgeHandoffV4MetricsError("initial rejection contains fabricated executable evidence")
    else:
        completed = row["stage_reached"] != "RESTORATION_PROBE"
        probe = _validate_probe_material(row, reopened_arrays, completed=completed)
        if probe["identity"] != state["snapshot_identity"]:
            raise PhysicalGraphEdgeHandoffV4MetricsError("probe/state snapshot identity drift")
        if (
            [trial["termination_flags"] for trial in probe["trials"]]
            != state["probe_trial_termination_flags"]
            or [trial["tip_sample_index"] for trial in probe["trials"]]
            != state["probe_tip_sample_indices"]
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "probe/state terminal evidence drift"
            )
        snapshot = row["snapshot"]
        artifact_sha = probe["identity"]["artifact_file_sha256"]
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("snapshot_payload_sha256") != artifact_sha
            or row["initial_decision_state_sha256"] != artifact_sha
        ):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "initial snapshot transport binding drift"
            )
        validate_snapshot_previous_applied_command_binding(
            snapshot, persisted, reopened_arrays=reopened_arrays,
        )
        rgb = np.asarray(reopened_arrays.get("rgb"))
        if rgb.dtype.str != "|u1" or list(rgb.shape) != [168, 224, 3] or row["current_rgb_sha256"] != _canonical_array_sha256(rgb):
            raise PhysicalGraphEdgeHandoffV4MetricsError("qualification RGB binding drift")
        if not completed:
            if row["graph"] is not None or row["teacher"] is not None:
                raise PhysicalGraphEdgeHandoffV4MetricsError("probe rejection contains teacher evidence")
        else:
            teacher = row["teacher"]
            if not isinstance(teacher, Mapping) or not isinstance(row["graph"], Mapping):
                raise PhysicalGraphEdgeHandoffV4MetricsError("teacher material evidence absent")
            teacher = _mapping(
                teacher, C.V4_TEACHER_MATERIAL_SUMMARY_FIELDS,
                "teacher material summary",
            )
            trace_names = [f"teacher__{name}" for name in C.TEACHER_TRACE_MEMBER_ORDER]
            if any(name not in reopened_arrays for name in trace_names):
                raise PhysicalGraphEdgeHandoffV4MetricsError("teacher raw trace member absent")
            digests = teacher.get("trace_digests")
            if not isinstance(digests, Mapping) or set(digests) != set(C.TEACHER_TRACE_MEMBER_ORDER) or any(
                digests[name] != _canonical_array_sha256(reopened_arrays[f"teacher__{name}"])
                for name in C.TEACHER_TRACE_MEMBER_ORDER
            ):
                raise PhysicalGraphEdgeHandoffV4MetricsError("teacher trace digest drift")
            reduced = reduce_qualification_teacher_trace(
                row, reopened_arrays, expected_pool_index=pool,
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
                raise PhysicalGraphEdgeHandoffV4MetricsError(
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
            if (
                not reduced["goal_reachable"]
                or not reduced["graph_edge_physically_executable"]
            ):
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
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "teacher outer disposition differs from raw reduction"
                )
            contact = row["contact_instrumentation"]
            expected_contact = {
                "api": "robot.get_contacts",
                "sample_period_s": C.TEACHER_TRACE_DT_S,
                "forbidden_net_force_api_used": False,
                "ontology_sha256": C.CONTACT_AUTHORITY["ontology_sha256"],
            }
            if contact != expected_contact:
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    "teacher contact instrumentation drift"
                )
    return {"metadata": row, "arrays": {name: reopened_arrays[name] for name in reopened_arrays}}


# ---------------------------------------------------------------------------
# Complete 256-state panel adequacy
# ---------------------------------------------------------------------------

def build_panel_adequacy(records: Sequence[Mapping[str, Any]], *, downstream_outcomes_opened: int = 0) -> dict[str, Any]:
    rows = [
        validate_qualification_disposition_row(item, expected_pool_index=index)
        for index, item in enumerate(_sequence(records, "panel adequacy records"))
    ]
    if len(rows) != C.V3.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffV4MetricsError("panel adequacy record count drift")
    downstream = _nonnegative_int(downstream_outcomes_opened, "downstream outcomes opened")
    hard = [row for row in rows if row["hard_stop"]]
    if hard:
        raise PhysicalGraphEdgeHandoffV4HardStop(
            f"hard-stop state disposition at pools {[row['pool_index'] for row in hard]}"
        )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["stratum_index"])].append(row)
    stratum_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for family in C.V3.FAMILY_IDS:
        for stratum_index in range(16):
            group = grouped[(family, stratum_index)]
            if len(group) != 4:
                raise PhysicalGraphEdgeHandoffV4MetricsError("family/stratum candidate count drift")
            qualified = sorted(
                (row for row in group if row["qualified"]),
                key=lambda row: (row["canonical_spec_sha256"], row["candidate_spec_id"]),
            )
            selected = qualified[0] if qualified else None
            if selected is not None:
                selected_rows.append(selected)
            causes = sorted({row["disposition"] for row in group if not row["qualified"]}) if not qualified else []
            stratum_rows.append(
                {
                    "family": family,
                    "stratum_index": stratum_index,
                    "candidate_count": 4,
                    "candidate_pool_indices": [row["pool_index"] for row in group],
                    "qualified_count": len(qualified),
                    "qualified_pool_indices": [row["pool_index"] for row in qualified],
                    "disposition_counts": dict(sorted(Counter(row["disposition"] for row in group).items())),
                    "adequate": bool(qualified),
                    "selected_pool_index": None if selected is None else selected["pool_index"],
                    "selected_candidate_spec_id": None if selected is None else selected["candidate_spec_id"],
                    "shortfall_causes": causes,
                }
            )
    family_rows = []
    for family in C.V3.FAMILY_IDS:
        family_candidates = [row for row in rows if row["family"] == family]
        family_strata = [row for row in stratum_rows if row["family"] == family]
        shortfall = [row["stratum_index"] for row in family_strata if not row["adequate"]]
        family_rows.append(
            {
                "family": family,
                "candidate_count": len(family_candidates),
                "qualified_count": sum(row["qualified"] for row in family_candidates),
                "selected_count": sum(row["adequate"] for row in family_strata),
                "shortfall_stratum_count": len(shortfall),
                "shortfall_strata": shortfall,
                "disposition_counts": dict(sorted(Counter(row["disposition"] for row in family_candidates).items())),
                "adequate": not shortfall,
            }
        )
    shortfall_rows = [row for row in stratum_rows if not row["adequate"]]
    adequate = not shortfall_rows
    if not adequate and downstream:
        raise PhysicalGraphEdgeHandoffV4MetricsError("downstream outcomes opened for inadequate panel")
    selected_rows.sort(key=lambda row: (C.V3.FAMILY_IDS.index(row["family"]), row["stratum_index"]))
    result = C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.panel_adequacy.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "qualification_record_count": len(rows),
            "all_pool_indices_present": True,
            "hard_stop_count": 0,
            "disposition_counts": dict(sorted(Counter(row["disposition"] for row in rows).items())),
            "qualified_count": sum(row["qualified"] for row in rows),
            "nonqualified_count": sum(not row["qualified"] for row in rows),
            "family_rows": family_rows,
            "stratum_rows": stratum_rows,
            "adequate_stratum_count": sum(row["adequate"] for row in stratum_rows),
            "shortfall_stratum_count": len(shortfall_rows),
            "shortfall_strata": [
                {"family": row["family"], "stratum_index": row["stratum_index"]}
                for row in shortfall_rows
            ],
            "shortfall_causes": [
                {"family": row["family"], "stratum_index": row["stratum_index"], "causes": row["shortfall_causes"]}
                for row in shortfall_rows
            ],
            "offset_opening_supplies_any_qualified_state": any(
                row["family"] == "OFFSET_OPENING" and row["qualified"] for row in rows
            ),
            "selected_pool_indices": [row["pool_index"] for row in selected_rows] if adequate else [],
            "selected_candidate_spec_ids": [row["candidate_spec_id"] for row in selected_rows] if adequate else [],
            "panel_state_count": C.V3.STATE_COUNT if adequate else 0,
            "family_role_counts_if_adequate": copy.deepcopy(C.V3.FAMILY_ROLE_COUNTS) if adequate else None,
            "adequate": adequate,
            "status": "ADEQUATE" if adequate else C.PANEL_INADEQUATE_DISPOSITION,
            "downstream_scientific_execution_authorized": adequate,
            "downstream_outcomes_opened": downstream,
            "next_decision": None if adequate else C.NEXT_DECISION_PANEL_INADEQUATE,
        }
    )
    if set(result) != C.PANEL_ADEQUACY_FIELDS:
        raise PhysicalGraphEdgeHandoffV4MetricsError("panel adequacy output fields drift")
    return result


def validate_panel_adequacy(value: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row = _mapping(value, C.PANEL_ADEQUACY_FIELDS, "panel adequacy")
    C.validate_content_digest(row)
    expected = build_panel_adequacy(records, downstream_outcomes_opened=row["downstream_outcomes_opened"])
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError("panel adequacy value drift")
    return row


# ---------------------------------------------------------------------------
# V1/V2/V3 immutable custody
# ---------------------------------------------------------------------------

def _historical_file(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_FILE_FIELDS, label)
    if not isinstance(row["path"], str) or not row["path"] or row["path"].startswith("/") or ".." in row["path"].split("/"):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} path drift")
    for field in ("bytes", "allocated_bytes", "device", "inode"):
        _nonnegative_int(row[field], f"{label}.{field}")
    if row["inode"] <= 0 or row["nlink"] != 1:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} link/inode drift")
    _sha(row["sha256"], f"{label}.sha256")
    return row


def _historical_directory(value: Any, label: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_DIRECTORY_FIELDS, label)
    if not isinstance(row["path"], str) or not row["path"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} path drift")
    for field in ("allocated_bytes", "device", "inode", "nlink"):
        _nonnegative_int(row[field], f"{label}.{field}")
    if row["inode"] <= 0 or row["nlink"] < 1:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"{label} inode drift")
    return row


def _historical_root(value: Any, key: str) -> dict[str, Any]:
    row = _mapping(value, C.HISTORICAL_CUSTODY_ROOT_FIELDS, f"historical root {key}")
    expected = C.HISTORICAL_ROOT_EXPECTATIONS[key]
    if row["path"] != expected["path"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"historical root {key} path drift")
    files = [_historical_file(item, f"{key}.files[{i}]") for i, item in enumerate(_sequence(row["files"], f"{key}.files"))]
    dirs = [_historical_directory(item, f"{key}.directories[{i}]") for i, item in enumerate(_sequence(row["directories"], f"{key}.directories"))]
    for field in ("file_count", "directory_count", "regular_file_apparent_bytes", "regular_file_allocated_bytes", "directory_allocated_bytes", "allocated_bytes"):
        _nonnegative_int(row[field], f"{key}.{field}")
    if (
        [item["path"] for item in files] != sorted(item["path"] for item in files)
        or [item["path"] for item in dirs] != sorted(item["path"] for item in dirs)
        or row["file_count"] != len(files)
        or row["directory_count"] != len(dirs)
        or row["regular_file_apparent_bytes"] != sum(item["bytes"] for item in files)
        or row["regular_file_allocated_bytes"] != sum(item["allocated_bytes"] for item in files)
        or row["directory_allocated_bytes"] != sum(item["allocated_bytes"] for item in dirs)
        or row["allocated_bytes"] != row["regular_file_allocated_bytes"] + row["directory_allocated_bytes"]
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"historical root {key} aggregate drift")
    projection = copy.deepcopy(row)
    projection.pop("manifest_sha256")
    if row["manifest_sha256"] != hashlib.sha256(C.canonical_json_bytes(projection)).hexdigest():
        raise PhysicalGraphEdgeHandoffV4MetricsError(f"historical root {key} manifest drift")
    for field in ("file_count", "regular_file_apparent_bytes", "manifest_sha256"):
        if row[field] != expected[field]:
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"historical root {key} frozen projection drift")
    return row


def validate_external_v1_v2_v3_custody_receipt(
    value: Any, *, expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_FIELDS, "external historical custody receipt")
    if "content_digest" in row or row["schema"] != C.HISTORICAL_CUSTODY_RECEIPT_SCHEMA or row["experiment_id"] != C.EXPERIMENT_ID or row["generated_before_v4_simulator_creation"] is not True:
        raise PhysicalGraphEdgeHandoffV4MetricsError("historical custody identity/self-digest drift")
    repository = _mapping(row["repository"], C.HISTORICAL_CUSTODY_REPOSITORY_FIELDS, "historical repository")
    expected_repository = {
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": C.V2_SOURCE_FREEZE_COMMIT,
        "v3_source_freeze_commit": C.V3_SOURCE_FREEZE_COMMIT,
        "v1_freeze_subject": C.V3.V2.V1.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v2_freeze_subject": C.V3.V2.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v3_freeze_subject": C.V3.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v4_source_parent_commit": C.SOURCE_PARENT_COMMIT,
        "current_head_commit": C.SOURCE_PARENT_COMMIT,
        "repository_worktree_clean_at_receipt_emission": False,
        "historical_tracked_sources_match_frozen_commits": True,
        "v4_development_only": True,
        "v4_permanently_ineligible_for_final_evaluation": True,
        "sealed_paths_accessed": 0,
        "tracked_ignore_bypass_used": False,
    }
    if repository != expected_repository:
        raise PhysicalGraphEdgeHandoffV4MetricsError("historical repository projection drift")
    prior = _mapping(row["prior_receipt_bindings"], C.HISTORICAL_PRIOR_RECEIPT_BINDING_FIELDS, "prior receipts")
    if prior != {
        "v1_custody_receipt_binding": C.V1_RECEIPT_BINDING,
        "v1_v2_custody_receipt_binding": C.V1_V2_RECEIPT_BINDING,
        "v2_regeneration_receipt_present": False,
        "v3_regeneration_receipt_present": False,
    }:
        raise PhysicalGraphEdgeHandoffV4MetricsError("prior receipt binding drift")
    roots_raw = row["roots"]
    if (
        not isinstance(roots_raw, Mapping)
        or set(roots_raw) != set(C.HISTORICAL_ROOT_KEYS)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "historical root inventory drift"
        )
    roots = {key: _historical_root(roots_raw[key], key) for key in C.HISTORICAL_ROOT_KEYS}
    boundary = _mapping(row["v3_partial_boundary"], C.HISTORICAL_V3_PARTIAL_BOUNDARY_FIELDS, "V3 partial boundary")
    if boundary != C.V3_PARTIAL_BOUNDARY_EXPECTATION:
        raise PhysicalGraphEdgeHandoffV4MetricsError("V3 partial boundary drift")
    failures = [
        _mapping(item, C.HISTORICAL_V3_FAILURE_ROW_FIELDS, f"V3 failure[{i}]")
        for i, item in enumerate(_sequence(row["v3_failure_log_bindings"], "V3 failure bindings"))
    ]
    if failures != C.V3_FAILURE_EXPECTATION:
        raise PhysicalGraphEdgeHandoffV4MetricsError("V3 failure projection drift")
    interpretation = _mapping(
        row["v3_terminal_interpretation"],
        C.V3_TERMINAL_INTERPRETATION_FIELDS,
        "V3 terminal interpretation",
    )
    if interpretation != C.V3_TERMINAL_INTERPRETATION:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "V3 terminal interpretation drift"
        )
    immutability = _mapping(row["immutability"], C.HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS, "historical immutability")
    if immutability["audit_mode"] != "READ_ONLY" or any(immutability[field] is not True for field in C.HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS if field != "audit_mode"):
        raise PhysicalGraphEdgeHandoffV4MetricsError("historical immutability gate failed")
    nonreuse = _mapping(row["nonreuse"], C.HISTORICAL_CUSTODY_NONREUSE_FIELDS, "historical nonreuse")
    for field, item in nonreuse.items():
        if field == "historical_runtime_artifact_or_shard_reused":
            if item is not False:
                raise PhysicalGraphEdgeHandoffV4MetricsError("historical runtime reuse")
        elif not isinstance(item, int) or isinstance(item, bool) or item != 0:
            raise PhysicalGraphEdgeHandoffV4MetricsError(f"historical nonreuse drift: {field}")
    all_inodes = [(item["device"], item["inode"]) for root in roots.values() for item in root["files"]]
    if len(all_inodes) != len(set(all_inodes)):
        raise PhysicalGraphEdgeHandoffV4MetricsError("historical cross-root shared inode")
    encoded = C.canonical_json_bytes(row)
    binding = {"path": str(C.HISTORICAL_CUSTODY_RECEIPT_PATH), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if expected_binding is not None and binding != C._historical_binding(expected_binding):
        raise PhysicalGraphEdgeHandoffV4MetricsError("external historical binding mismatch")
    return row


def historical_custody_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return validate_external_v1_v2_v3_custody_receipt(value)


def historical_custody_projection_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(C.canonical_json_bytes(historical_custody_projection(value))[:-1]).hexdigest()


def build_v1_v2_v3_custody_and_nonreuse(
    *, v4_source_freeze_commit: str, external_receipt_binding: Mapping[str, Any],
    external_receipt_projection_sha256: str,
) -> dict[str, Any]:
    source = C._commit(v4_source_freeze_commit, "v4_source_freeze_commit")
    binding = C._historical_binding(external_receipt_binding)
    projection = _sha(external_receipt_projection_sha256, "historical projection")
    return {
        "schema": "physical_graph_edge_handoff_qualification_v4.v1_v2_v3_custody_and_nonreuse.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "v4_source_freeze_commit": source,
        "external_historical_custody_receipt_binding": binding,
        "external_historical_custody_projection_sha256": projection,
        "v1_v2_v3_roots_unchanged": True,
        "historical_payloads_copied_into_v4": 0,
        "historical_hardlinks_into_v4": 0,
        "historical_shared_inodes_with_v4": 0,
        "historical_runtime_artifact_or_shard_reused": False,
        "historical_snapshot_or_probe_rerun": False,
        "sealed_path_accessed": False,
        "ignore_bypassed": False,
        "allowed_read_scope": "CUSTODY_BINDING_ONLY_NO_HISTORICAL_SNAPSHOT_OR_PROBE_REEXECUTION",
        "v3_terminal_interpretation": copy.deepcopy(
            C.V3_TERMINAL_INTERPRETATION
        ),
        "pass": True,
    }


def validate_v1_v2_v3_custody_and_nonreuse(
    value: Any, *, v4_source_freeze_commit: str | None = None,
    external_receipt_binding: Mapping[str, Any] | None = None,
    external_receipt_projection_sha256: str | None = None,
) -> dict[str, Any]:
    row = _mapping(value, C.V1_V2_V3_CUSTODY_AND_NONREUSE_FIELDS, "internal historical custody")
    if "content_digest" in row:
        raise PhysicalGraphEdgeHandoffV4MetricsError("internal custody self digest forbidden")
    expected = build_v1_v2_v3_custody_and_nonreuse(
        v4_source_freeze_commit=v4_source_freeze_commit or row["v4_source_freeze_commit"],
        external_receipt_binding=external_receipt_binding or row["external_historical_custody_receipt_binding"],
        external_receipt_projection_sha256=external_receipt_projection_sha256 or row["external_historical_custody_projection_sha256"],
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError("internal historical custody drift")
    return row


# ---------------------------------------------------------------------------
# Scientific invariance receipt
# ---------------------------------------------------------------------------

def build_regression_results(evidence: Mapping[str, bool] | None = None) -> list[dict[str, Any]]:
    supplied = {} if evidence is None else dict(evidence)
    return [
        {
            "requirement_id": requirement,
            "passed": bool(supplied.get(requirement, False)),
            "evidence": "FOCUSED_PURE_OR_NONSCIENTIFIC_FIXTURE_PASS" if supplied.get(requirement, False) else "NOT_PROVEN",
        }
        for requirement in C.ALL_V4_REGRESSION_IDS
    ]


def build_scientific_invariance_receipt(
    contract: Mapping[str, Any], regression_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated = C.validate_contract(contract)
    results = _sequence(regression_results, "V4 regression results")
    if len(results) != len(C.ALL_V4_REGRESSION_IDS):
        raise PhysicalGraphEdgeHandoffV4MetricsError("V4 regression result cardinality drift")
    normalized = []
    for index, (item, requirement) in enumerate(zip(results, C.ALL_V4_REGRESSION_IDS)):
        part = _mapping(item, C.REGRESSION_RESULT_FIELDS, f"regression[{index}]")
        if part["requirement_id"] != requirement or type(part["passed"]) is not bool or not isinstance(part["evidence"], str) or not part["evidence"]:
            raise PhysicalGraphEdgeHandoffV4MetricsError("V4 regression row drift")
        normalized.append(part)
    projection = C.scientific_invariance_projection(validated)
    projection_sha = hashlib.sha256(C.canonical_json_bytes(projection)[:-1]).hexdigest()
    all_pass = all(item["passed"] for item in normalized)
    return {
        "schema": "physical_graph_edge_handoff_qualification_v4.scientific_invariance_receipt.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "v3_contract_content_digest": C.V3_CONTRACT_CONTENT_DIGEST,
        "v4_contract_content_digest": validated["content_digest"],
        "v3_scientific_projection_sha256": C.V3_SCIENTIFIC_PROJECTION_SHA256,
        "v4_scientific_projection_sha256": projection_sha,
        "scientific_projection_equal": projection == C.V3_SCIENTIFIC_PROJECTION,
        "candidate_specs_equal": C.build_candidate_specs() == C.V3.build_candidate_specs(),
        "source_dependency_paths_equal": tuple(C.SOURCE_DEPENDENCY_PATHS) == tuple(C.V3.SOURCE_DEPENDENCY_PATHS),
        "v3_source_freeze_commit": C.V3_SOURCE_FREEZE_COMMIT,
        "state_disposition_authority_content_digest": C.STATE_DISPOSITION_AUTHORITY["content_digest"],
        "panel_adequacy_authority_content_digest": C.PANEL_ADEQUACY_AUTHORITY["content_digest"],
        "persisted_array_hash_authority_retained": C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"] == C.V3.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"],
        "snapshot_semantic_serializer_authority_retained": C.V3.SEMANTIC_SERIALIZER_AUTHORITY["content_digest"] == C.SEMANTIC_SERIALIZER_AUTHORITY["content_digest"],
        "snapshot_behavioural_probe_authority_retained": C.V3.BEHAVIOURAL_PROBE_AUTHORITY["content_digest"] == C.BEHAVIOURAL_PROBE_AUTHORITY["content_digest"],
        "authorized_change_scope": C.SCIENTIFIC_INVARIANCE_AUTHORITY["authorized_change_scope"],
        "terminal_evidence_implementation_ramifications": (
            C.SCIENTIFIC_INVARIANCE_AUTHORITY[
                "terminal_evidence_implementation_ramifications"
            ]
        ),
        "pre_panel_engineering_correction": copy.deepcopy(
            C.SCIENTIFIC_INVARIANCE_AUTHORITY[
                "pre_panel_engineering_correction"
            ]
        ),
        "v3_terminal_interpretation": copy.deepcopy(
            C.V3_TERMINAL_INTERPRETATION
        ),
        "metric_formula_gate_threshold_model_tuning_changed": False,
        "historical_snapshot_or_probe_rerun": False,
        "development_only": True,
        "final_evaluation_eligible": False,
        "regression_results": normalized,
        "all_regressions_passed": all_pass,
        "pass": bool(all_pass and projection == C.V3_SCIENTIFIC_PROJECTION and C.build_candidate_specs() == C.V3.build_candidate_specs()),
    }


def validate_scientific_invariance_receipt(value: Any) -> dict[str, Any]:
    row = _mapping(value, C.SCIENTIFIC_INVARIANCE_RECEIPT_FIELDS, "scientific invariance receipt")
    if "content_digest" in row:
        raise PhysicalGraphEdgeHandoffV4MetricsError("invariance receipt self digest forbidden")
    expected = build_scientific_invariance_receipt(C.build_contract(), row["regression_results"])
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected) or row["pass"] is not True:
        raise PhysicalGraphEdgeHandoffV4MetricsError("scientific invariance receipt drift")
    return row


# ---------------------------------------------------------------------------
# Metrics and publication
# ---------------------------------------------------------------------------

V1_EVIDENCE_KEYS = frozenset(V2M.V1M.EVIDENCE_KEYS)
V4_COMMON_EVIDENCE_KEYS = frozenset(
    {
        "runtime_contract", "external_historical_custody_receipt",
        "v1_v2_v3_custody_and_nonreuse", "scientific_invariance_receipt",
        "qualification_state_dispositions_jsonl", "panel_adequacy",
        "material_shard_validations", "qualification_runtime_environment",
    }
)
V4_PANEL_INADEQUATE_EVIDENCE_KEYS = V4_COMMON_EVIDENCE_KEYS
V4_SUCCESS_EVIDENCE_KEYS = (
    V4_COMMON_EVIDENCE_KEYS | V1_EVIDENCE_KEYS
    | {"teacher_selection", "panel_context", "encoding_receipt"}
)
MATERIAL_SHARD_VALIDATION_FIELDS = frozenset(
    {
        "pool_index", "metadata_binding", "payload_binding",
        "persisted_array_evidence_sha256", "state_disposition_sha256",
        "stage_runtime_sha256", "backend_runtime_sha256",
        "backend_runtime_core_sha256",
        "metadata_payload_reopened", "persisted_arrays_valid", "pass",
    }
)


def _validate_material_shard_validations(value: Any, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    supplied = _sequence(value, "material shard validations")
    if len(supplied) != C.V3.PROSPECTIVE_POOL_COUNT:
        raise PhysicalGraphEdgeHandoffV4MetricsError("material validation cardinality drift")
    result = []
    for index, (item, state) in enumerate(zip(supplied, rows)):
        row = _mapping(item, MATERIAL_SHARD_VALIDATION_FIELDS, f"material validation[{index}]")
        if row["pool_index"] != index or row["metadata_binding"] != state["material_metadata_binding"] or row["payload_binding"] != state["material_payload_binding"] or row["persisted_array_evidence_sha256"] != state["persisted_array_evidence_sha256"]:
            raise PhysicalGraphEdgeHandoffV4MetricsError("material validation cross-link drift")
        for digest_field in (
            "stage_runtime_sha256", "backend_runtime_sha256",
            "backend_runtime_core_sha256",
        ):
            _sha(row[digest_field], f"material validation {digest_field}")
        expected_state_sha = hashlib.sha256(C.canonical_json_bytes({key: state[key] for key in C.STATE_DISPOSITION_FIELDS})[:-1]).hexdigest()
        if row["state_disposition_sha256"] != expected_state_sha or any(row[field] is not True for field in ("metadata_payload_reopened", "persisted_arrays_valid", "pass")):
            raise PhysicalGraphEdgeHandoffV4MetricsError("material validation gate failed")
        result.append(row)
    return result


def _common_reduction(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = C.validate_runtime_contract(evidence["runtime_contract"])
    external = validate_external_v1_v2_v3_custody_receipt(
        evidence["external_historical_custody_receipt"],
        expected_binding=runtime["historical_custody_receipt_binding"],
    )
    external_sha = historical_custody_projection_sha256(external)
    internal = validate_v1_v2_v3_custody_and_nonreuse(
        evidence["v1_v2_v3_custody_and_nonreuse"],
        v4_source_freeze_commit=runtime["source_freeze_commit"],
        external_receipt_binding=runtime["historical_custody_receipt_binding"],
        external_receipt_projection_sha256=external_sha,
    )
    invariance = validate_scientific_invariance_receipt(evidence["scientific_invariance_receipt"])
    rows = validate_qualification_state_dispositions_jsonl(evidence["qualification_state_dispositions_jsonl"])
    panel = validate_panel_adequacy(evidence["panel_adequacy"], rows)
    material = _validate_material_shard_validations(
        evidence["material_shard_validations"], rows
    )
    qualification_runtime = validate_qualification_runtime_environment(
        evidence["qualification_runtime_environment"], runtime,
        material_shard_validations=material,
        # Pure synthetic integration retains its truthful fake flag.  The
        # production independent reducer calls the raw-metadata path with the
        # default false allowance before this reducer can publish.
        allow_fake_runtime=True,
    )
    return runtime, rows, panel, internal, invariance, qualification_runtime


def _v1_panel_compatibility_document(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the V4 disposition column for private V1 aggregation."""

    result = _project_v4_evidence_to_v1(panel)
    result.pop("content_digest", None)
    pool = copy.deepcopy(result["prospective_pool_selection"])
    qualification = []
    for raw in pool["qualification_rows"]:
        row = copy.deepcopy(dict(raw))
        row.pop("disposition", None)
        qualification.append(row)
    pool["qualification_rows"] = qualification
    pool["qualification_projection_sha256"] = hashlib.sha256(
        V1M.C.canonical_json_bytes(qualification)[:-1]
    ).hexdigest()
    result["prospective_pool_selection"] = pool
    return V1M.C.attach_content_digest(result)


def _v1_teacher_compatibility_document(
    teachers: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only the V4 pool-index column from actual compact teachers."""

    result = _project_v4_evidence_to_v1(teachers)
    result.pop("content_digest", None)
    records = []
    for raw in result["records"]:
        row = copy.deepcopy(dict(raw))
        row.pop("qualification_pool_index", None)
        records.append(row)
    result["records"] = records
    return V1M.C.attach_content_digest(result)


def _v1_snapshot_compatibility_document(
    snapshots: Mapping[str, Any], inspections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bridge the one corrected raw-byte hash in a private V1 validation copy."""

    result = _project_v4_evidence_to_v1(snapshots)
    result.pop("content_digest", None)
    canonical_rows = inspections["state_snapshots.npz"]["members"][
        "previous_applied_command"
    ]["row_or_slice_sha256s"]
    if len(canonical_rows) != len(result["records"]) or not canonical_rows:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "previous-command private bridge cardinality drift"
        )
    for record, digest in zip(result["records"], canonical_rows):
        _sha(record["previous_applied_command_sha256"], "V4 raw previous-command digest")
        record["previous_applied_command_sha256"] = _sha(
            digest, "V1 canonical previous-command digest"
        )
    return V1M.C.attach_content_digest(result)


def _exact_v1_input_result(
    expected_input: Any, result: Any, label: str,
) -> Any:
    def validate(value: Any, *_args: Any, **_kwargs: Any) -> Any:
        if V1M.C.canonical_json_bytes(value) != V1M.C.canonical_json_bytes(expected_input):
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                f"{label} changed inside inherited recomputation"
            )
        return copy.deepcopy(result)

    return validate


def _validate_sparse_npz_crosslinks(
    inspections: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Any], teachers: Mapping[str, Any],
    pixels: Mapping[str, Any], latents: Mapping[str, Any],
    fanout: Sequence[Mapping[str, Any]], repeats: Sequence[Mapping[str, Any]],
) -> None:
    """Run exact V1 raw cross-links over real compact teachers, never padding."""

    dynamic_payload = copy.deepcopy(V1M.C.NPZ_PAYLOAD_AUTHORITY)
    dynamic_payload["teacher_traces.npz"] = C.teacher_trace_npz_authority(
        len(teachers["records"])
    )

    class _SparseCrossContract:
        def __init__(self) -> None:
            self.NPZ_PAYLOAD_AUTHORITY = dynamic_payload

        def __getattr__(self, name: str) -> Any:
            return getattr(V1M.C, name)

    try:
        _local_v1_call(
            V1M._cross_validate_npz_bindings,
            inspections, snapshots, teachers, pixels, latents, fanout, repeats,
            replacements={"C": _SparseCrossContract()},
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"sparse NPZ cross-link failed: {exc}"
        ) from exc


def validate_v4_success_evidence(
    evidence: Mapping[str, Any], qualification_rows: Sequence[Mapping[str, Any]],
    panel_adequacy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every successful document and sparse physical cross-link."""

    inherited = {key: copy.deepcopy(evidence[key]) for key in V1_EVIDENCE_KEYS}
    teacher_count_hint = len(
        inherited["teacher_trace_index"].get("records", [])
    ) if isinstance(inherited["teacher_trace_index"], Mapping) else None
    inspections = validate_npz_inspections(
        inherited["npz_inspections"], teacher_trace_count=teacher_count_hint,
    )
    panel = validate_panel_manifest(inherited["panel_manifest"])
    split = validate_split_manifest(inherited["split_manifest"], panel)
    panel_context = validate_panel_context(
        evidence["panel_context"], panel, split,
    )
    graph = validate_graph_manifest(inherited["graph_manifest"], panel)
    snapshots = validate_state_snapshot_index(
        inherited["state_snapshot_index"], panel,
    )
    teachers = validate_teacher_trace_index(
        inherited["teacher_trace_index"], panel,
    )
    selection = validate_teacher_selection(
        evidence["teacher_selection"], panel, teachers,
        qualification_rows, panel_adequacy,
    )
    ports = validate_edge_port_index(
        inherited["edge_port_index"], panel, graph, teachers,
    )
    waypoints = validate_waypoint_contracts(
        inherited["waypoint_contracts"], panel, graph, ports,
    )
    pixels = validate_pixel_index(inherited["pixel_index"], panel)
    latents = validate_latent_index(inherited["latent_index"], pixels)
    encoding_receipt = validate_encoding_receipt(
        evidence["encoding_receipt"], panel, pixels, latents,
    )
    fanout = validate_candidate_fanout_rows(
        inherited["candidate_fanout"], panel, snapshots,
    )
    development = validate_development_target_selection(
        inherited["development_target_selection"], panel, fanout, waypoints,
    )
    heldout = validate_heldout_ranker_score_rows(
        inherited["heldout_ranker_scores"], panel, fanout, development,
        waypoints, teachers,
    )
    repeats = validate_repeated_execution_rows(
        inherited["repeated_execution"], panel, fanout, heldout, snapshots,
    )
    v1_panel = _v1_panel_compatibility_document(panel)
    v1_teachers = _v1_teacher_compatibility_document(teachers)
    v1_snapshots = _v1_snapshot_compatibility_document(snapshots, inspections)
    v1_inspections = {
        path: _project_v4_evidence_to_v1(row)
        for path, row in inspections.items()
    }
    v1_pixels = _project_v4_evidence_to_v1(pixels)
    v1_latents = _project_v4_evidence_to_v1(latents)
    v1_fanout = _project_v4_evidence_to_v1(fanout)
    v1_repeats = _project_v4_evidence_to_v1(repeats)
    _validate_sparse_npz_crosslinks(
        v1_inspections, v1_snapshots, v1_teachers, v1_pixels, v1_latents,
        v1_fanout, v1_repeats,
    )
    return {
        "npz_inspections": inspections,
        "panel_manifest": panel,
        "split_manifest": split,
        "graph_manifest": graph,
        "state_snapshot_index": snapshots,
        "teacher_trace_index": teachers,
        "teacher_selection": selection,
        "panel_context": panel_context,
        "edge_port_index": ports,
        "waypoint_contracts": waypoints,
        "pixel_index": pixels,
        "latent_index": latents,
        "encoding_receipt": encoding_receipt,
        "candidate_fanout": fanout,
        "development_target_selection": development,
        "heldout_ranker_scores": heldout,
        "repeated_execution": repeats,
        "v1_panel_manifest": v1_panel,
        "v1_state_snapshot_index": v1_snapshots,
        "v1_teacher_trace_index": v1_teachers,
        "teacher_trace_count": len(teachers["records"]),
    }


def _recompute_selected_science(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute frozen V1 aggregation after complete V4-native validation."""

    v1_evidence = {
        key: _project_v4_evidence_to_v1(validated[key])
        for key in V1_EVIDENCE_KEYS
    }
    v1_evidence["panel_manifest"] = copy.deepcopy(
        validated["v1_panel_manifest"]
    )
    v1_evidence["state_snapshot_index"] = copy.deepcopy(
        validated["v1_state_snapshot_index"]
    )
    v1_evidence["teacher_trace_index"] = copy.deepcopy(
        validated["v1_teacher_trace_index"]
    )
    inspection_result = copy.deepcopy(v1_evidence["npz_inspections"])
    if (
        not isinstance(inspection_result, Mapping)
        or set(inspection_result) != set(V1M.C.NPZ_PAYLOAD_AUTHORITY)
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "validated NPZ inspection mapping drift"
        )
    # V1's public reducer receives a sequence and its validator returns a
    # path-keyed mapping.  Preserve that exact interface even though the V4
    # native validation stage has already returned the latter representation.
    v1_evidence["npz_inspections"] = [
        copy.deepcopy(inspection_result[path])
        for path in V1M.C.NPZ_PAYLOAD_AUTHORITY
    ]
    replacements: dict[str, Any] = {
        "validate_npz_inspections": _exact_v1_input_result(
            v1_evidence["npz_inspections"], inspection_result,
            "validate_npz_inspections",
        ),
        "_cross_validate_npz_bindings": lambda *_args, **_kwargs: None,
    }
    for function_name, key in (
        ("validate_panel_manifest", "panel_manifest"),
        ("validate_split_manifest", "split_manifest"),
        ("validate_graph_manifest", "graph_manifest"),
        ("validate_state_snapshot_index", "state_snapshot_index"),
        ("validate_teacher_trace_index", "teacher_trace_index"),
        ("validate_edge_port_index", "edge_port_index"),
        ("validate_waypoint_contracts", "waypoint_contracts"),
        ("validate_pixel_index", "pixel_index"),
        ("validate_latent_index", "latent_index"),
        ("validate_candidate_fanout_rows", "candidate_fanout"),
        ("validate_development_target_selection", "development_target_selection"),
        ("validate_heldout_ranker_score_rows", "heldout_ranker_scores"),
        ("validate_repeated_execution_rows", "repeated_execution"),
    ):
        replacements[function_name] = _exact_v1_input_result(
            v1_evidence[key], v1_evidence[key], function_name,
        )
    try:
        result = _local_v1_call(
            V1M.recompute_metrics, v1_evidence, replacements=replacements,
        )
    except Exception as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            f"frozen selected-64 aggregation failed: {exc}"
        ) from exc
    mapped = _project_v1_evidence_to_v4(result)
    mapped.pop("content_digest", None)
    mapped["evidence_counts"]["prospective_teacher_traces"] = validated[
        "teacher_trace_count"
    ]
    return mapped


def recompute_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError("evidence is not mapping")
    panel_raw = evidence.get("panel_adequacy")
    if not isinstance(panel_raw, Mapping) or type(panel_raw.get("adequate")) is not bool:
        raise PhysicalGraphEdgeHandoffV4MetricsError("panel adequacy selector absent")
    expected_keys = V4_SUCCESS_EVIDENCE_KEYS if panel_raw["adequate"] else V4_PANEL_INADEQUATE_EVIDENCE_KEYS
    if set(evidence) != set(expected_keys):
        raise PhysicalGraphEdgeHandoffV4MetricsError("V4 recompute evidence field drift")
    (
        runtime, rows, panel, internal, invariance, qualification_runtime,
    ) = _common_reduction(evidence)
    disposition_metrics = {
        "record_count": len(rows),
        "qualified_count": sum(row["qualified"] for row in rows),
        "nonqualified_count": sum(not row["qualified"] for row in rows),
        "disposition_counts": dict(sorted(Counter(row["disposition"] for row in rows).items())),
        "initial_boundary_tipped_count": sum(row["disposition"] == "INITIAL_BOUNDARY_TIPPED" for row in rows),
        "restoration_probe_tipped_count": sum(row["disposition"] == "RESTORATION_PROBE_TIPPED" for row in rows),
        "teacher_terminated_unsafely_count": sum(row["disposition"] == "TEACHER_TERMINATED_UNSAFELY" for row in rows),
        "hard_stop_count": sum(row["hard_stop"] for row in rows),
        "all_material_shards_reopened_and_valid": True,
    }
    custody_metrics = {
        "external_receipt_binding": copy.deepcopy(runtime["historical_custody_receipt_binding"]),
        "external_receipt_projection_sha256": historical_custody_projection_sha256(evidence["external_historical_custody_receipt"]),
        "v3_source_freeze_commit": C.V3_SOURCE_FREEZE_COMMIT,
        "v3_completed_pool_count": C.V3_PARTIAL_BOUNDARY_EXPECTATION["completed_pool_count"],
        "v3_failed_pool_indices": list(C.V3_FAILED_POOL_INDICES),
        "historical_snapshot_or_probe_rerun": False,
        "historical_roots_unchanged": internal["v1_v2_v3_roots_unchanged"],
        "pass": internal["pass"],
    }
    if not panel["adequate"]:
        metrics = {
            "schema": "physical_graph_edge_handoff_qualification_v4.metrics.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "development_only": True,
            "final_evaluation_eligible": False,
            "scientific_result_produced": True,
            "terminal_disposition": C.PANEL_INADEQUATE_DISPOSITION,
            "primary_classification": C.PANEL_INADEQUATE_DISPOSITION,
            "secondary_classifications": [],
            "next_experiment": C.NEXT_DECISION_PANEL_INADEQUATE,
            "v4_state_dispositions": disposition_metrics,
            "v4_panel_adequacy": copy.deepcopy(panel),
            "v4_historical_custody": custody_metrics,
            "v4_scientific_invariance": {
                "v3_contract_content_digest": C.V3_CONTRACT_CONTENT_DIGEST,
                "v4_contract_content_digest": C.build_contract()["content_digest"],
                "scientific_projection_equal": invariance["scientific_projection_equal"],
                "candidate_specs_equal": invariance["candidate_specs_equal"],
                "all_regressions_passed": invariance["all_regressions_passed"],
                "pass": invariance["pass"],
            },
            "downstream_scientific_metrics": None,
            "qualification_runtime_environment": copy.deepcopy(
                qualification_runtime
            ),
            "runtime_environments": {
                "qualification": copy.deepcopy(qualification_runtime),
                "physical": None,
                "encoder": None,
                "ranker": None,
                "any_fake_runtime": bool(
                    qualification_runtime["fake_runtime"]
                ),
            },
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
        return C.attach_content_digest(metrics)
    validated_success = validate_v4_success_evidence(evidence, rows, panel)
    mapped = _recompute_selected_science(validated_success)
    assembled_physical = mapped.get("runtime_environments", {}).get("physical")
    if not isinstance(assembled_physical, Mapping):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "assembled physical runtime environment absent"
        )
    expected_core = {
        field: assembled_physical[field]
        for field in C.PHYSICAL_RUNTIME_CORE_FIELDS
    }
    if (
        expected_core != qualification_runtime["physical_runtime_core"]
        or assembled_physical.get("runtime_core_sha256")
        != qualification_runtime["physical_runtime_core_sha256"]
        or assembled_physical.get("qualification_runtime_sha256s")
        != qualification_runtime["qualification_stage_runtime_sha256s"]
    ):
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "qualification/assembled physical runtime cross-link drift"
        )
    downstream_scientific_metrics = {
        field: copy.deepcopy(mapped[field])
        for field in C.V4_INHERITED_DOWNSTREAM_METRIC_FIELDS
    }
    mapped["schema"] = "physical_graph_edge_handoff_qualification_v4.metrics.v1"
    mapped["experiment_id"] = C.EXPERIMENT_ID
    mapped.update(
        {
            "development_only": True,
            "final_evaluation_eligible": False,
            "scientific_result_produced": True,
            "terminal_disposition": None,
            "v4_state_dispositions": disposition_metrics,
            "v4_panel_adequacy": copy.deepcopy(panel),
            "v4_historical_custody": custody_metrics,
            "v4_scientific_invariance": {
                "v3_contract_content_digest": C.V3_CONTRACT_CONTENT_DIGEST,
                "v4_contract_content_digest": C.build_contract()["content_digest"],
                "scientific_projection_equal": invariance["scientific_projection_equal"],
                "candidate_specs_equal": invariance["candidate_specs_equal"],
                "all_regressions_passed": invariance["all_regressions_passed"],
                "pass": invariance["pass"],
            },
            "downstream_scientific_metrics": downstream_scientific_metrics,
            "qualification_runtime_environment": copy.deepcopy(
                qualification_runtime
            ),
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
    )
    return C.attach_content_digest(mapped)


V4_RESULT_FIELDS = C.V4_RESULT_FIELDS
V4_PUBLICATION_PROJECTION_FIELDS = C.V4_PUBLICATION_PROJECTION_FIELDS


def build_result_document(
    recomputed_metrics: Mapping[str, Any], runtime_contract: Mapping[str, Any], *,
    metrics_sha256: str, independent_reducer_receipt_sha256: str,
    runtime_seconds: float, scientific_storage_bytes: int,
) -> dict[str, Any]:
    metrics = C.validate_content_digest(recomputed_metrics)
    runtime = C.validate_runtime_contract(runtime_contract)
    qualification_runtime = validate_qualification_runtime_environment(
        metrics.get("qualification_runtime_environment"), runtime,
        allow_fake_runtime=True,
    )
    _sha(metrics_sha256, "metrics SHA-256")
    _sha(independent_reducer_receipt_sha256, "reducer receipt SHA-256")
    if not isinstance(runtime_seconds, (int, float)) or isinstance(runtime_seconds, bool) or not math.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise PhysicalGraphEdgeHandoffV4MetricsError("runtime_seconds drift")
    if not isinstance(scientific_storage_bytes, int) or isinstance(scientific_storage_bytes, bool) or scientific_storage_bytes <= 0:
        raise PhysicalGraphEdgeHandoffV4MetricsError("scientific_storage_bytes drift")
    terminal = metrics.get("terminal_disposition")
    if terminal not in (None, C.PANEL_INADEQUATE_DISPOSITION):
        raise PhysicalGraphEdgeHandoffV4MetricsError("result terminal disposition drift")
    downstream = metrics.get("downstream_scientific_metrics")
    if terminal is None:
        downstream = _mapping(
            downstream, C.V4_INHERITED_DOWNSTREAM_METRIC_FIELDS,
            "inherited downstream scientific metrics",
        )
        for field in (
            "primary_classification", "secondary_classifications",
            "next_experiment", "runtime_environments",
        ):
            if C.canonical_json_bytes(downstream[field]) != C.canonical_json_bytes(
                metrics[field]
            ):
                raise PhysicalGraphEdgeHandoffV4MetricsError(
                    f"downstream/top-level {field} drift"
                )
    elif downstream is not None:
        raise PhysicalGraphEdgeHandoffV4MetricsError(
            "panel-inadequate result contains downstream science"
        )
    if terminal == C.PANEL_INADEQUATE_DISPOSITION:
        expected_runtime_environments = {
            "qualification": copy.deepcopy(qualification_runtime),
            "physical": None,
            "encoder": None,
            "ranker": None,
            "any_fake_runtime": bool(qualification_runtime["fake_runtime"]),
        }
        if metrics.get("runtime_environments") != expected_runtime_environments:
            raise PhysicalGraphEdgeHandoffV4MetricsError(
                "panel-inadequate qualification runtime reporting drift"
            )
    return C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.result.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "source_commit": runtime["source_freeze_commit"],
            "source_parent_commit": C.SOURCE_PARENT_COMMIT,
            "source_baseline_commit": C.SOURCE_BASELINE_COMMIT,
            "development_only": True,
            "final_evaluation_eligible": False,
            "scientific_result_produced": True,
            "terminal_disposition": terminal,
            "primary_classification": metrics["primary_classification"],
            "secondary_classifications": copy.deepcopy(metrics["secondary_classifications"]),
            "next_experiment": metrics["next_experiment"],
            "v3_terminal_diagnosis": copy.deepcopy(
                C.V3_TERMINAL_INTERPRETATION
            ),
            "three_identity_contracts_retained": {
                "artifact_file_sha256": "descriptive transport bytes",
                "snapshot_semantic_digest_v1": "canonical semantic snapshot identity",
                "snapshot_behavioural_digest_v1": "designated trial-0 restoration/probe identity",
            },
            "state_dispositions": copy.deepcopy(metrics["v4_state_dispositions"]),
            "panel_adequacy": copy.deepcopy(metrics["v4_panel_adequacy"]),
            "historical_custody": copy.deepcopy(metrics["v4_historical_custody"]),
            "scientific_invariance": copy.deepcopy(metrics["v4_scientific_invariance"]),
            "downstream_scientific_metrics": copy.deepcopy(metrics["downstream_scientific_metrics"]),
            "runtime_environments": copy.deepcopy(metrics.get("runtime_environments")),
            "qualification_runtime_environment": copy.deepcopy(
                qualification_runtime
            ),
            "metrics_sha256": metrics_sha256,
            "independent_reducer_receipt_sha256": independent_reducer_receipt_sha256,
            "runtime_seconds": float(runtime_seconds),
            "scientific_storage_bytes": scientific_storage_bytes,
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
    )


def validate_result_document(
    value: Any, recomputed_metrics: Mapping[str, Any], runtime_contract: Mapping[str, Any], *,
    metrics_sha256: str, independent_reducer_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    row = _mapping(value, V4_RESULT_FIELDS, "V4 result")
    C.validate_content_digest(row)
    receipt_sha = row["independent_reducer_receipt_sha256"]
    if independent_reducer_receipt_sha256 is not None and receipt_sha != independent_reducer_receipt_sha256:
        raise PhysicalGraphEdgeHandoffV4MetricsError("result reducer receipt SHA drift")
    expected = build_result_document(
        recomputed_metrics, runtime_contract, metrics_sha256=metrics_sha256,
        independent_reducer_receipt_sha256=receipt_sha,
        runtime_seconds=row["runtime_seconds"], scientific_storage_bytes=row["scientific_storage_bytes"],
    )
    if C.canonical_json_bytes(row) != C.canonical_json_bytes(expected):
        raise PhysicalGraphEdgeHandoffV4MetricsError("result document drift")
    return row


def _scientific_bindings(
    value: Mapping[str, Mapping[str, Any]], *, adequate: bool, metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], int, str]:
    expected_leaves = C.SUCCESS_OUTPUT_LEAVES if adequate else C.PANEL_INADEQUATE_OUTPUT_LEAVES
    prepublication = set(expected_leaves) - {"result.json", "result.md", "file_hashes.json"}
    if not isinstance(value, Mapping) or set(value) != prepublication:
        raise PhysicalGraphEdgeHandoffV4MetricsError("scientific binding inventory drift")
    rows = {key: _file_binding(item, f"scientific binding {key}") for key, item in value.items()}
    if any(rows[key]["path"] != key for key in rows):
        raise PhysicalGraphEdgeHandoffV4MetricsError("scientific binding path drift")
    if rows["contract.json"]["sha256"] != hashlib.sha256(C.canonical_json_bytes(runtime)).hexdigest():
        raise PhysicalGraphEdgeHandoffV4MetricsError("contract publication binding drift")
    metrics_bytes = C.canonical_json_bytes(metrics)
    if rows["metrics.json"] != {"path": "metrics.json", "bytes": len(metrics_bytes), "sha256": hashlib.sha256(metrics_bytes).hexdigest()}:
        raise PhysicalGraphEdgeHandoffV4MetricsError("metrics publication binding drift")
    return rows, sum(item["bytes"] for item in rows.values()), rows["metrics.json"]["sha256"]


def build_result_publication_projection(
    recomputed_metrics: Mapping[str, Any], scientific_bindings: Mapping[str, Mapping[str, Any]],
    runtime_contract: Mapping[str, Any], *, independent_reducer_receipt_sha256: str,
    historical_custody_receipt_binding: Mapping[str, Any], runtime_seconds: float,
) -> dict[str, Any]:
    metrics = C.validate_content_digest(recomputed_metrics)
    runtime = C.validate_runtime_contract(runtime_contract)
    adequate = metrics["v4_panel_adequacy"]["adequate"]
    _rows, storage, metrics_sha = _scientific_bindings(scientific_bindings, adequate=adequate, metrics=metrics, runtime=runtime)
    if C._historical_binding(historical_custody_receipt_binding) != runtime["historical_custody_receipt_binding"]:
        raise PhysicalGraphEdgeHandoffV4MetricsError("publication historical custody binding drift")
    result = build_result_document(
        metrics, runtime, metrics_sha256=metrics_sha,
        independent_reducer_receipt_sha256=independent_reducer_receipt_sha256,
        runtime_seconds=runtime_seconds, scientific_storage_bytes=storage,
    )
    return C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.result_publication_projection.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "result_document": result,
            "recomputed_metrics": copy.deepcopy(metrics),
        }
    )


def validate_result_publication_projection(
    value: Any, *, recomputed_metrics: Mapping[str, Any],
    scientific_bindings: Mapping[str, Mapping[str, Any]], runtime_contract: Mapping[str, Any],
    independent_reducer_receipt_sha256: str,
    historical_custody_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = _mapping(value, V4_RESULT_FIELDS, "V4 publication result")
    C.validate_content_digest(result)
    expected = build_result_publication_projection(
        recomputed_metrics, scientific_bindings, runtime_contract,
        independent_reducer_receipt_sha256=independent_reducer_receipt_sha256,
        historical_custody_receipt_binding=historical_custody_receipt_binding,
        runtime_seconds=result["runtime_seconds"],
    )
    if C.canonical_json_bytes(result) != C.canonical_json_bytes(expected["result_document"]):
        raise PhysicalGraphEdgeHandoffV4MetricsError("publication result projection drift")
    return expected


def build_result_report(result_document: Mapping[str, Any], recomputed_metrics: Mapping[str, Any]) -> str:
    result = _mapping(result_document, V4_RESULT_FIELDS, "V4 report result")
    C.validate_content_digest(result)
    metrics = C.validate_content_digest(recomputed_metrics)
    panel = metrics["v4_panel_adequacy"]
    shortfall = "\n".join(
        f"- {item['family']} stratum {item['stratum_index']}: {', '.join(item['causes'])}"
        for item in panel["shortfall_causes"]
    ) or "- none"
    disposition_lines = "\n".join(
        f"- {name}: {count}" for name, count in metrics["v4_state_dispositions"]["disposition_counts"].items()
    )
    text = f"""# Tipped-state physical graph edge handoff qualification V4

## Disposition

Primary classification: {result['primary_classification']}

Terminal disposition: {result['terminal_disposition'] or 'none'}

Next experiment: {result['next_experiment']}

## V1/V2/V3 custody and V3 terminal diagnosis

{json.dumps(result['historical_custody'], sort_keys=True)}

{json.dumps(result['v3_terminal_diagnosis'], sort_keys=True)}

## Three snapshot identity contracts

{json.dumps(result['three_identity_contracts_retained'], sort_keys=True)}

## State dispositions

{disposition_lines}

## Panel adequacy

Adequate: {str(panel['adequate']).lower()}

Qualified states: {panel['qualified_count']} / {panel['qualification_record_count']}

Adequate strata: {panel['adequate_stratum_count']} / 64

OFFSET_OPENING supplies any qualified state: {str(panel['offset_opening_supplies_any_qualified_state']).lower()}

### Shortfall strata and causes

{shortfall}

## Inherited downstream science

{json.dumps(result['downstream_scientific_metrics'], sort_keys=True)}

## Runtime, storage, and training

Runtime environments: {json.dumps(result['runtime_environments'], sort_keys=True)}

Qualification runtime environment: {json.dumps(result['qualification_runtime_environment'], sort_keys=True)}

Scientific storage bytes: {result['scientific_storage_bytes']}

Models trained: 0

Development-only: true; permanently ineligible for final evaluation.
"""
    if not text.endswith("\n"):
        text += "\n"
    return text


def validate_result_report(value: str | bytes, result_document: Mapping[str, Any], recomputed_metrics: Mapping[str, Any]) -> str:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
    except UnicodeDecodeError as exc:
        raise PhysicalGraphEdgeHandoffV4MetricsError("result report is not UTF-8") from exc
    expected = build_result_report(result_document, recomputed_metrics)
    if text != expected:
        raise PhysicalGraphEdgeHandoffV4MetricsError("result report drift")
    return text


def build_result_report_bytes(value: Any) -> bytes:
    projection = _mapping(value, V4_PUBLICATION_PROJECTION_FIELDS, "V4 publication projection")
    C.validate_content_digest(projection)
    if projection["schema"] != "physical_graph_edge_handoff_qualification_v4.result_publication_projection.v1" or projection["experiment_id"] != C.EXPERIMENT_ID:
        raise PhysicalGraphEdgeHandoffV4MetricsError("publication projection identity drift")
    return build_result_report(projection["result_document"], projection["recomputed_metrics"]).encode("utf-8")


def reducer_authority() -> dict[str, Any]:
    # V4 inherits the frozen downstream V1 scientific authorities, but it does
    # not inherit either of the V2/V3 execution terminals or publication
    # surfaces.  Starting from V1 keeps the selected-panel formulas exact while
    # making it impossible for the V3 first-eight/28-leaf contract to leak into
    # the prospective V4 reducer authority.
    base = _project_v1_evidence_to_v4(V1M.reducer_authority())
    base.pop("content_digest", None)
    documents = copy.deepcopy(base["documents"])
    documents["panel_manifest"].update(
        {
            "qualification_count": C.V3.PROSPECTIVE_POOL_COUNT,
            "qualification_row": sorted(C.V4_POOL_QUALIFICATION_FIELDS),
            "qualification_disposition_cross_binding": (
                "exact qualification_state_dispositions.jsonl pool row"
            ),
        }
    )
    documents["teacher_trace_index"] = {
        "root": sorted(V1M.TEACHER_INDEX_FIELDS),
        "container": "records",
        "count": "actual teacher_executed state count in [64,256]",
        "minimum_count": C.V3.STATE_COUNT,
        "maximum_count": C.V3.PROSPECTIVE_POOL_COUNT,
        "identity_order": ["qualification_pool_index"],
        "row": sorted(C.V4_TEACHER_RECORD_FIELDS),
        "trace_index_semantics": "compact 0..N-1 NPZ slice index",
        "qualification_pool_index_semantics": "original registered pool index",
    }
    npz_authorities = copy.deepcopy(base["npz_authorities"])
    npz_authorities["teacher_traces.npz"] = {
        "dynamic_authority_builder": "teacher_trace_npz_authority",
        "teacher_trace_count_source": "validated teacher_trace_index.records length",
        "minimum_teacher_trace_count": C.V3.STATE_COUNT,
        "maximum_teacher_trace_count": C.V3.PROSPECTIVE_POOL_COUNT,
        "minimum_count_authority": C.teacher_trace_npz_authority(C.V3.STATE_COUNT),
        "maximum_count_authority": C.teacher_trace_npz_authority(
            C.V3.PROSPECTIVE_POOL_COUNT
        ),
        "placeholder_or_padded_rows_forbidden": True,
    }
    base.update(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.reducer_authority.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "runtime_paths": copy.deepcopy(C.RUNTIME_OUTPUT_PATHS),
            "documents": documents,
            "npz_authorities": npz_authorities,
            "evidence_keys": sorted(V4_SUCCESS_EVIDENCE_KEYS),
            "success_output_leaf_count": C.OUTPUT_LEAF_COUNT,
            "success_output_leaves": list(C.SUCCESS_OUTPUT_LEAVES),
            "panel_inadequate_output_leaf_count": C.PANEL_INADEQUATE_OUTPUT_LEAF_COUNT,
            "panel_inadequate_output_leaves": list(C.PANEL_INADEQUATE_OUTPUT_LEAVES),
            "technical_hard_stop_output_leaves": list(C.TECHNICAL_HARD_STOP_OUTPUT_LEAVES),
            "state_disposition_authority": copy.deepcopy(C.STATE_DISPOSITION_AUTHORITY),
            "terminal_nonfinite_authority": copy.deepcopy(
                C.V4_TERMINAL_NONFINITE_AUTHORITY
            ),
            "panel_adequacy_authority": copy.deepcopy(C.PANEL_ADEQUACY_AUTHORITY),
            "panel_teacher_subset_authority": copy.deepcopy(
                C.V4_PANEL_TEACHER_SUBSET_AUTHORITY
            ),
            "qualification_runtime_authority": copy.deepcopy(
                C.QUALIFICATION_RUNTIME_AUTHORITY
            ),
            "raw_teacher_reduction_authority": copy.deepcopy(
                C.V4_RAW_TEACHER_REDUCTION_AUTHORITY
            ),
            "pre_panel_engineering_correction_authority": copy.deepcopy(
                C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
            ),
            "teacher_termination_derivation_authority": copy.deepcopy(
                C.V4_TEACHER_TERMINATION_DERIVATION_AUTHORITY
            ),
            "teacher_selection_authority": copy.deepcopy(
                C.TEACHER_SELECTION_AUTHORITY
            ),
            "material_inventory_authority": copy.deepcopy(
                C.MATERIAL_INVENTORY_AUTHORITY
            ),
            "persisted_array_hash_authority": copy.deepcopy(
                C.PERSISTED_ARRAY_HASH_AUTHORITY
            ),
            "snapshot_semantic_serializer_authority": copy.deepcopy(
                C.V3.SEMANTIC_SERIALIZER_AUTHORITY
            ),
            "snapshot_behavioural_probe_authority": copy.deepcopy(
                C.V3.BEHAVIOURAL_PROBE_AUTHORITY
            ),
            "port_heading_implementation_alignment_authority": copy.deepcopy(
                C.V3.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "candidate_port_metric_implementation_alignment_authority": copy.deepcopy(
                C.V3.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
            ),
            "external_historical_custody_authority": copy.deepcopy(C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY),
            "v1_v2_v3_custody_and_nonreuse_authority": copy.deepcopy(C.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY),
            "scientific_invariance_authority": copy.deepcopy(C.SCIENTIFIC_INVARIANCE_AUTHORITY),
            "material_shard_validation_fields": sorted(MATERIAL_SHARD_VALIDATION_FIELDS),
            "panel_inadequate_evidence_keys": sorted(V4_PANEL_INADEQUATE_EVIDENCE_KEYS),
            "success_evidence_keys": sorted(V4_SUCCESS_EVIDENCE_KEYS),
            "result_fields": sorted(V4_RESULT_FIELDS),
            "publication_projection_fields": sorted(V4_PUBLICATION_PROJECTION_FIELDS),
            "result_publication_authority": copy.deepcopy(
                C.RESULT_PUBLICATION_AUTHORITY
            ),
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )
    return C.attach_content_digest(base)


def state_disposition_authority() -> dict[str, Any]:
    return copy.deepcopy(C.STATE_DISPOSITION_AUTHORITY)


def panel_adequacy_authority() -> dict[str, Any]:
    return copy.deepcopy(C.PANEL_ADEQUACY_AUTHORITY)


def scientific_invariance_authority() -> dict[str, Any]:
    return copy.deepcopy(C.SCIENTIFIC_INVARIANCE_AUTHORITY)


def historical_custody_authority() -> dict[str, Any]:
    return copy.deepcopy(C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY)


__all__ = [
    "PhysicalGraphEdgeHandoffV4HardStop",
    "PhysicalGraphEdgeHandoffV4MetricsError",
    "MATERIAL_SHARD_VALIDATION_FIELDS",
    "V1_EVIDENCE_KEYS",
    "V4_COMMON_EVIDENCE_KEYS",
    "V4_PANEL_INADEQUATE_EVIDENCE_KEYS",
    "V4_SUCCESS_EVIDENCE_KEYS",
    "V4_RESULT_FIELDS",
    "V4_PUBLICATION_PROJECTION_FIELDS",
    "build_persisted_array_evidence",
    "build_qualification_disposition_row",
    "build_qualification_state_dispositions_jsonl",
    "build_regression_results",
    "build_result_document",
    "build_result_publication_projection",
    "build_result_report",
    "build_result_report_bytes",
    "build_scientific_invariance_receipt",
    "build_state_disposition_record",
    "build_v1_v2_v3_custody_and_nonreuse",
    "build_panel_adequacy",
    "build_qualification_runtime_environment",
    "canonical_v1_planar_segment_projection",
    "canonical_semantic_snapshot",
    "classify_state_disposition",
    "compare_behavioural_probe_traces",
    "compare_terminated_behavioural_probe_traces",
    "compare_tipped_behavioural_probe_traces",
    "derive_candidate_port_metrics",
    "external_artifact_bindings",
    "first_registered_port_crossing",
    "frozen_binary64_fma",
    "historical_custody_authority",
    "historical_custody_projection",
    "historical_custody_projection_sha256",
    "panel_adequacy_authority",
    "panel_manifest_authority",
    "persisted_array_bytes_sha256",
    "persisted_array_manifest_row",
    "point_in_polygon_inclusive",
    "predecessor_context_binding",
    "physical_trace_reduction_authority",
    "project_v3_evidence_to_v4",
    "project_v4_evidence_to_v3",
    "recompute_metrics",
    "reduce_qualification_teacher_trace",
    "reducer_authority",
    "runtime_environment_sha256",
    "scientific_invariance_authority",
    "semantic_snapshot_sha256",
    "snapshot_behavioural_digest",
    "snapshot_semantic_evidence",
    "state_disposition_authority",
    "transverse_port_crossing",
    "validate_external_v1_v2_v3_custody_receipt",
    "validate_candidate_fanout_rows",
    "validate_development_target_selection",
    "validate_edge_port_index",
    "validate_encoding_receipt",
    "validate_graph_manifest",
    "validate_heldout_ranker_score_rows",
    "validate_latent_index",
    "validate_npz_archive_comment",
    "validate_npz_inspections",
    "validate_panel_adequacy",
    "validate_panel_context",
    "validate_panel_manifest",
    "validate_persisted_array_evidence",
    "validate_physical_runtime_environment",
    "validate_pixel_index",
    "validate_qualification_disposition_row",
    "validate_qualification_material_shard",
    "validate_qualification_runtime_environment",
    "validate_qualification_teacher_raw_evidence",
    "validate_qualification_state_dispositions_jsonl",
    "validate_result_document",
    "validate_result_publication_projection",
    "validate_result_report",
    "validate_repeated_execution_rows",
    "validate_scientific_invariance_receipt",
    "validate_snapshot_previous_applied_command_binding",
    "validate_state_snapshot_previous_applied_command_hashes",
    "validate_state_disposition_record",
    "validate_state_material_payload",
    "validate_state_snapshot_index",
    "validate_split_manifest",
    "validate_teacher_trace_index",
    "validate_teacher_selection",
    "teacher_trace_npz_authority",
    "validate_visual_runtime_environment",
    "validate_v4_success_evidence",
    "validate_waypoint_contracts",
    "validate_v1_v2_v3_custody_and_nonreuse",
]
