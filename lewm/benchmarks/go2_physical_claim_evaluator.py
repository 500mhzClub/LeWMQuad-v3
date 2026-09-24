"""Canonical physical verification for Go2 claim-attempt traces.

This module is deliberately independent of controllers, datasets, renderers,
learned models, and repository runtime imports. It validates the immutable
physical scene-manifest object structurally and uses it as ground truth.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence

__all__ = ["evaluate_physical_claim_trace"]


EVALUATOR_CONTRACT_SHA256 = (
    "2de4ff20cff2901ab07b681f042c231f1a1e06f95a77d8c4ae2c20c9e2bb8112"
)
RAW_TRACE_SCHEMA = "lewm_go2_claim_trace_v1"
EVALUATED_TRACE_SCHEMA = "lewm_go2_evaluated_claim_trace_v1"
EVENT_SCHEMA = "lewm_go2_physical_claim_evaluation_v1"
SUMMARY_SCHEMA = "lewm_go2_physical_claim_summary_v1"
TASK_SET_SCHEMA = "lewm_go2_claim_task_set_v1"

CLAIM_DISTANCE_M = 1.20
CLAIM_ABSOLUTE_BEARING_RAD = 0.25
LINE_OF_SIGHT_INFLATION_M = 0.0

_SCENE_MANIFEST_FIELDS = frozenset(
    {
        "scene_id",
        "family",
        "difficulty_tier",
        "topology_seed",
        "visual_seed",
        "physics_seed",
        "world_bounds_xy_m",
        "spawn",
        "graph_nodes",
        "graph_edges",
        "obstacles",
        "landmarks",
        "camera_constraints",
        "split",
        "walls",
        "visual_randomization",
        "physics_randomization",
        "camera_extrinsic_jitter",
    }
)
_BOX_OBJECT_FIELDS = frozenset(
    {
        "object_id",
        "kind",
        "center_xyz_m",
        "size_xyz_m",
        "yaw_rad",
        "material_id",
        "roll_rad",
        "pitch_rad",
    }
)
_VISUAL_RANDOMIZATION_FIELDS = frozenset(
    {"material_overrides", "lighting", "distractor_objects"}
)

_TRACE_KEYS = frozenset(
    {
        "schema",
        "trace_id",
        "episode_id",
        "scene_id",
        "physical_manifest_sha256",
        "task_object_ids",
        "task_object_set_sha256",
        "controller_claim_attempts",
        "evaluator_feedback_to_controller",
    }
)
_EVENT_KEYS = frozenset(
    {
        "trace_id",
        "episode_id",
        "scene_id",
        "event_id",
        "tick",
        "event_index",
        "requested_target",
        "claimed_target",
        "robot_pose_world_xy_yaw",
        "pose_binary64_le_sha256",
        "pose_hex",
        "pose_provenance",
        "physical_manifest_sha256",
    }
)
_MODERN_PROVENANCE = frozenset(
    {
        "runtime_full_precision",
        "oracle_full_precision",
        "eligibility_candidate_full_precision",
    }
)
_TASK_COLORS = frozenset({"red", "green", "blue", "yellow"})

UNVERIFIABLE_REASONS = (
    "trace_schema_or_key_set_invalid",
    "trace_id_missing_or_invalid",
    "episode_id_missing_or_invalid",
    "scene_manifest_identity_mismatch",
    "physical_manifest_commitment_mismatch",
    "task_object_ids_not_exact_sorted_unique",
    "task_object_set_mismatch",
    "task_object_commitment_mismatch",
    "evaluator_feedback_to_controller_nonempty",
    "trace_event_order_invalid",
    "manifest_duplicate_object_id",
    "manifest_invalid_physical_geometry",
    "event_key_set_or_type_invalid",
    "event_trace_identity_mismatch",
    "event_id_missing_or_duplicate",
    "claim_tick_or_index_invalid",
    "requested_reference_malformed",
    "requested_namespace_forbidden_for_provenance",
    "requested_identity_unresolved",
    "requested_identity_ambiguous",
    "claimed_reference_malformed",
    "claimed_namespace_forbidden_for_provenance",
    "claimed_identity_unresolved",
    "claimed_identity_ambiguous",
    "pose_provenance_invalid",
    "claim_pose_missing_or_nonfinite",
    "claim_pose_precision_commitment_mismatch",
    "physical_computation_nonfinite",
    "legacy_provenance_noncanonical",
    "legacy_pose_missing_yaw",
    "legacy_pose_rounded_or_inferred",
)
REJECTION_REASONS = (
    "requested_identity_not_in_task_set",
    "claimed_identity_not_in_task_set",
    "requested_claimed_identity_mismatch",
    "outside_inclusive_claim_distance",
    "zero_inflation_physical_los_blocked",
    "outside_inclusive_claim_bearing",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_plain_json(value: object) -> object:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _content_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _utf8_bytes(value: object) -> bytes | None:
    if type(value) is not str:
        return None
    try:
        return str.encode(value, "utf-8")
    except UnicodeEncodeError:
        return None


def _is_nonempty_utf8_string(value: object) -> bool:
    return bool(type(value) is str and value and _utf8_bytes(value) is not None)


def _utf8_sorted(values: Sequence[str]) -> list[str]:
    return sorted(values, key=lambda value: _utf8_bytes(value) or b"")


def _is_nonbool_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= (2**63 - 1)
    )


def _finite_float(value: object) -> float | None:
    # Manifest values have not passed through the trace's canonical JSON
    # snapshot. Exact builtins prevent numeric subclasses from committing one
    # value in JSON while coercing to another during physical evaluation.
    if type(value) not in {int, float}:
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _finite_triple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    converted = tuple(_finite_float(item) for item in value)
    if any(item is None for item in converted):
        return None
    return (converted[0], converted[1], converted[2])  # type: ignore[return-value]


def _task_set_commitment(
    *, scene_id: str, physical_manifest_sha256: str, task_object_ids: Sequence[str]
) -> str:
    return _content_sha256(
        {
            "schema": TASK_SET_SCHEMA,
            "scene_id": scene_id,
            "physical_manifest_sha256": physical_manifest_sha256,
            "task_object_ids": list(task_object_ids),
        }
    )


def _is_box_like(box: object) -> bool:
    return bool(
        is_dataclass(box)
        and not isinstance(box, type)
        and {field.name for field in fields(box)} == _BOX_OBJECT_FIELDS
    )


def _box_geometry_is_valid(box: object) -> bool:
    if not _is_box_like(box):
        return False
    if (
        not _is_nonempty_utf8_string(box.object_id)  # type: ignore[attr-defined]
        or type(box.kind) is not str  # type: ignore[attr-defined]
        or type(box.material_id) is not str  # type: ignore[attr-defined]
    ):
        return False
    center = box.center_xyz_m  # type: ignore[attr-defined]
    size = box.size_xyz_m  # type: ignore[attr-defined]
    if (
        type(center) not in {list, tuple}
        or type(size) not in {list, tuple}
        or len(center) != 3
        or len(size) != 3
    ):
        return False
    numeric = (
        *center,
        *size,
        box.yaw_rad,  # type: ignore[attr-defined]
        getattr(box, "roll_rad", 0.0),
        getattr(box, "pitch_rad", 0.0),
    )
    converted = tuple(_finite_float(value) for value in numeric)
    if any(value is None for value in converted):
        return False
    sx = converted[3]
    sy = converted[4]
    return bool(sx is not None and sy is not None and sx > 0.0 and sy > 0.0)


def _manifest_inventory(manifest: object) -> dict[str, Any]:
    def collection(value: object) -> tuple[object, ...]:
        return tuple(value) if type(value) in {list, tuple} else ()

    manifest_like = bool(
        is_dataclass(manifest)
        and not isinstance(manifest, type)
        and {field.name for field in fields(manifest)} == _SCENE_MANIFEST_FIELDS
        and callable(getattr(manifest, "to_dict", None))
    )
    landmarks = collection(getattr(manifest, "landmarks", None))
    walls = collection(getattr(manifest, "walls", None))
    obstacles = collection(getattr(manifest, "obstacles", None))
    distractors: tuple[object, ...] = ()
    visual_invalid = False
    visual = getattr(manifest, "visual_randomization", None)
    if (
        visual is not None
        and is_dataclass(visual)
        and not isinstance(visual, type)
        and {field.name for field in fields(visual)}
        == _VISUAL_RANDOMIZATION_FIELDS
    ):
        raw_distractors = visual.distractor_objects
        distractors = collection(raw_distractors)
        visual_invalid = type(raw_distractors) not in {list, tuple}
    elif visual is not None:
        visual_invalid = True

    collections = (
        ("walls", walls),
        ("obstacles", obstacles),
        ("visual_randomization.distractor_objects", distractors),
        ("landmarks", landmarks),
    )
    all_objects = tuple(box for _name, boxes in collections for box in boxes)
    object_ids = [box.object_id for box in all_objects if _is_box_like(box)]
    string_object_id_keys = [
        str.encode(object_id, "utf-8", "surrogatepass")
        for object_id in object_ids
        if isinstance(object_id, str)
    ]
    duplicate_object_id = len(string_object_id_keys) != len(
        set(string_object_id_keys)
    )
    geometry_invalid = (
        not manifest_like
        or not _is_nonempty_utf8_string(getattr(manifest, "scene_id", None))
        or type(getattr(manifest, "landmarks", None)) not in {list, tuple}
        or type(getattr(manifest, "walls", None)) not in {list, tuple}
        or type(getattr(manifest, "obstacles", None)) not in {list, tuple}
        or visual_invalid
        or any(not _box_geometry_is_valid(box) for box in all_objects)
        or any(
            type(box.material_id) is not str
            for box in landmarks
            if _is_box_like(box)
        )
    )

    explicit_hash: str | None = None
    if manifest_like:
        try:
            manifest_dict = _canonical_plain_json(
                manifest.to_dict()  # type: ignore[attr-defined]
            )
            dataclass_dict = _canonical_plain_json(asdict(manifest))
            if manifest_dict != dataclass_dict:
                raise ValueError("manifest to_dict does not match dataclass fields")
            explicit_hash = hashlib.sha256(
                _canonical_bytes(manifest_dict)
            ).hexdigest()
        except (OverflowError, TypeError, ValueError):
            geometry_invalid = True
    # lewm_worlds.manifest.manifest_sha256 uses this exact serialization for a
    # finite manifest. Keeping the evaluator stdlib-only avoids importing the
    # lewm_worlds package, whose __init__ eagerly imports renderer exporters.
    hash_compatible = explicit_hash is not None

    landmark_by_id: dict[str, list[object]] = {}
    for landmark in landmarks:
        if _is_box_like(landmark) and _is_nonempty_utf8_string(landmark.object_id):
            landmark_by_id.setdefault(landmark.object_id, []).append(landmark)

    ordered_occluders = tuple(
        (collection, box)
        for collection, boxes in collections
        for box in sorted(
            boxes,
            key=lambda item: (
                _utf8_bytes(item.object_id) or b""
                if _is_box_like(item)
                else b""
            ),
        )
    )
    return {
        "landmarks": landmarks,
        "landmark_by_id": landmark_by_id,
        "all_landmark_ids": _utf8_sorted(
            [
                landmark.object_id
                for landmark in landmarks
                if _is_box_like(landmark)
                and _is_nonempty_utf8_string(landmark.object_id)
            ]
        ),
        "ordered_occluders": ordered_occluders,
        "duplicate_object_id": duplicate_object_id,
        "geometry_invalid": geometry_invalid,
        "manifest_sha256": explicit_hash,
        "manifest_hash_compatible": hash_compatible,
    }


def _reference_resolution(
    reference: object,
    *,
    provenance: object,
    landmarks: Sequence[object],
) -> dict[str, Any]:
    malformed = (
        not isinstance(reference, Mapping)
        or set(reference) != {"namespace", "value"}
        or not isinstance(reference.get("namespace"), str)
        or not _is_nonempty_utf8_string(reference.get("value"))
        or reference.get("namespace")
        not in {"object_id", "task_color", "legacy_alias"}
    )
    if malformed:
        return {"status": "malformed", "resolved_object_id": None}

    namespace = reference["namespace"]
    value = reference["value"]
    legacy_provenance = isinstance(provenance, str) and provenance.startswith("legacy_")
    if namespace == "legacy_alias" and not legacy_provenance:
        return {"status": "forbidden_for_provenance", "resolved_object_id": None}

    candidates: list[str] = []
    if namespace == "object_id":
        candidates = [
            landmark.object_id
            for landmark in landmarks
            if _is_box_like(landmark)
            and _is_nonempty_utf8_string(landmark.object_id)
            and landmark.object_id == value
        ]
    elif namespace == "task_color":
        if value in _TASK_COLORS:
            material = f"landmark_{value}"
            candidates = [
                landmark.object_id
                for landmark in landmarks
                if _is_box_like(landmark)
                and _is_nonempty_utf8_string(landmark.object_id)
                and isinstance(landmark.material_id, str)
                and landmark.material_id.casefold() == material
            ]
    else:
        query = value.strip(" \t\n\r\f\v").casefold()
        for landmark in landmarks:
            if (
                not _is_box_like(landmark)
                or not _is_nonempty_utf8_string(landmark.object_id)
                or not isinstance(landmark.material_id, str)
            ):
                continue
            object_id = landmark.object_id.casefold()
            material = landmark.material_id.casefold()
            aliases = {object_id, material}
            if material.startswith("landmark_"):
                aliases.add(material.removeprefix("landmark_"))
                color = material.removeprefix("landmark_")
                if color in _TASK_COLORS:
                    aliases.add(color)
            if query in aliases:
                candidates.append(landmark.object_id)

    if not candidates:
        return {"status": "unresolved", "resolved_object_id": None}
    if len(candidates) != 1:
        return {"status": "ambiguous", "resolved_object_id": None}
    return {"status": "resolved", "resolved_object_id": candidates[0]}


def _pose_analysis(event: Mapping[str, Any]) -> dict[str, Any]:
    decimal = _finite_triple(event.get("robot_pose_world_xy_yaw"))
    raw_hex = event.get("pose_hex")
    parsed_hex: tuple[float, float, float] | None = None
    canonical_hex = False
    if (
        isinstance(raw_hex, list)
        and len(raw_hex) == 3
        and all(isinstance(item, str) for item in raw_hex)
    ):
        try:
            values = tuple(float.fromhex(item) for item in raw_hex)
        except (OverflowError, ValueError):
            values = ()
        if len(values) == 3 and all(math.isfinite(value) for value in values):
            parsed_hex = (values[0], values[1], values[2])
            canonical_hex = [value.hex() for value in parsed_hex] == raw_hex

    missing_or_nonfinite = decimal is None or parsed_hex is None
    supplied_hash = event.get("pose_binary64_le_sha256")
    precision_mismatch = False
    packed_hash: str | None = None
    if decimal is not None and parsed_hex is not None:
        decimal_bytes = struct.pack("<3d", *decimal)
        hex_bytes = struct.pack("<3d", *parsed_hex)
        packed_hash = hashlib.sha256(hex_bytes).hexdigest()
        precision_mismatch = (
            decimal_bytes != hex_bytes
            or not canonical_hex
            or not _is_sha256(supplied_hash)
            or supplied_hash != packed_hash
        )

    provenance = event.get("pose_provenance")
    legacy = isinstance(provenance, str) and provenance.startswith("legacy_")
    decimal_value = event.get("robot_pose_world_xy_yaw")
    decimal_has_yaw = (
        isinstance(decimal_value, list)
        and len(decimal_value) == 3
        and _finite_float(decimal_value[2]) is not None
    )
    hex_has_yaw = parsed_hex is not None
    legacy_missing_yaw = legacy and not (decimal_has_yaw and hex_has_yaw)
    legacy_rounded_or_inferred = legacy and (
        missing_or_nonfinite
        or precision_mismatch
        or "rounded" in provenance
        or "inferred" in provenance
    )
    return {
        "decimal": decimal,
        "authoritative": parsed_hex,
        "canonical_hex": (
            [value.hex() for value in parsed_hex] if parsed_hex is not None else None
        ),
        "packed_hash": packed_hash,
        "missing_or_nonfinite": missing_or_nonfinite,
        "precision_mismatch": precision_mismatch,
        "legacy_missing_yaw": legacy_missing_yaw,
        "legacy_rounded_or_inferred": legacy_rounded_or_inferred,
    }


def _segment_intersects_box(
    robot_xy: tuple[float, float],
    target_xy: tuple[float, float],
    box: object,
) -> tuple[bool | None, bool]:
    rx, ry = robot_xy
    tx, ty = target_xy
    cx = float(box.center_xyz_m[0])  # type: ignore[attr-defined]
    cy = float(box.center_xyz_m[1])  # type: ignore[attr-defined]
    sx = float(box.size_xyz_m[0])  # type: ignore[attr-defined]
    sy = float(box.size_xyz_m[1])  # type: ignore[attr-defined]
    q = float(box.yaw_rad)  # type: ignore[attr-defined]
    try:
        c = math.cos(-q)
        s = math.sin(-q)
        rx_delta = rx - cx
        ry_delta = ry - cy
        tx_delta = tx - cx
        ty_delta = ty - cy
        x0_left = c * rx_delta
        x0_right = s * ry_delta
        x0 = x0_left - x0_right
        y0_left = s * rx_delta
        y0_right = c * ry_delta
        y0 = y0_left + y0_right
        x1_left = c * tx_delta
        x1_right = s * ty_delta
        x1 = x1_left - x1_right
        y1_left = s * tx_delta
        y1_right = c * ty_delta
        y1 = y1_left + y1_right
        dx_local = x1 - x0
        dy_local = y1 - y0
        hx = sx / 2.0
        hy = sy / 2.0
    except (OverflowError, ValueError):
        return None, False
    intermediates = (
        c,
        s,
        rx_delta,
        ry_delta,
        tx_delta,
        ty_delta,
        x0_left,
        x0_right,
        x0,
        y0_left,
        y0_right,
        y0,
        x1_left,
        x1_right,
        x1,
        y1_left,
        y1_right,
        y1,
        dx_local,
        dy_local,
        hx,
        hy,
    )
    if not all(math.isfinite(value) for value in intermediates):
        return None, False

    t_enter = 0.0
    t_exit = 1.0
    for p, direction, half_extent in (
        (x0, dx_local, hx),
        (y0, dy_local, hy),
    ):
        if direction == 0.0:
            if p < -half_extent or p > half_extent:
                return False, True
            continue
        a = (-half_extent - p) / direction
        b = (half_extent - p) / direction
        if not math.isfinite(a) or not math.isfinite(b):
            return None, False
        if a > b:
            a, b = b, a
        t_enter = max(t_enter, a)
        t_exit = min(t_exit, b)
        if not math.isfinite(t_enter) or not math.isfinite(t_exit):
            return None, False
        if t_enter > t_exit:
            return False, True
    return True, True


def _physical_metrics(
    pose: tuple[float, float, float] | None,
    claimed_object_id: str | None,
    *,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target": None,
        "distance_m": None,
        "target_world_bearing_rad": None,
        "signed_bearing_error_rad": None,
        "absolute_bearing_error_rad": None,
        "distance_passes": None,
        "line_of_sight_passes": None,
        "bearing_passes": None,
        "blockers": [],
        "computation_nonfinite": False,
    }
    if claimed_object_id is None or inventory["geometry_invalid"]:
        return result
    matches = inventory["landmark_by_id"].get(claimed_object_id, ())
    if len(matches) != 1:
        return result
    target = matches[0]
    result["target"] = target
    if pose is None or inventory["duplicate_object_id"]:
        return result
    rx, ry, yaw = pose
    tx = float(target.center_xyz_m[0])
    ty = float(target.center_xyz_m[1])
    try:
        dx = tx - rx
        dy = ty - ry
        distance = math.hypot(dx, dy)
        target_bearing = math.atan2(dy, dx)
        angle_delta = target_bearing - yaw
        sine = math.sin(angle_delta)
        cosine = math.cos(angle_delta)
        signed_error = math.atan2(sine, cosine)
        absolute_error = abs(signed_error)
    except (OverflowError, ValueError):
        result["computation_nonfinite"] = True
        return result
    if not all(
        math.isfinite(value)
        for value in (
            dx,
            dy,
            distance,
            target_bearing,
            angle_delta,
            sine,
            cosine,
            signed_error,
            absolute_error,
        )
    ):
        result["computation_nonfinite"] = True
        return result

    result.update(
        {
            "distance_m": distance,
            "target_world_bearing_rad": target_bearing,
            "signed_bearing_error_rad": signed_error,
            "absolute_bearing_error_rad": absolute_error,
            "distance_passes": distance <= CLAIM_DISTANCE_M,
            "bearing_passes": absolute_error <= CLAIM_ABSOLUTE_BEARING_RAD,
        }
    )

    blockers: list[list[str]] = []
    los_finite = True
    for collection, box in inventory["ordered_occluders"]:
        if collection == "landmarks" and box.object_id == claimed_object_id:
            continue
        intersects, finite = _segment_intersects_box((rx, ry), (tx, ty), box)
        if not finite:
            los_finite = False
            continue
        if intersects:
            blockers.append([collection, box.object_id])
    result["blockers"] = blockers
    if los_finite:
        result["line_of_sight_passes"] = not blockers
    else:
        result["computation_nonfinite"] = True
    return result


def _event_top_level_types_are_valid(event: object) -> bool:
    if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
        return False
    return bool(
        all(isinstance(event.get(name), str) for name in ("trace_id", "episode_id", "scene_id"))
        and isinstance(event.get("event_id"), str)
        and isinstance(event.get("tick"), int)
        and not isinstance(event.get("tick"), bool)
        and isinstance(event.get("event_index"), int)
        and not isinstance(event.get("event_index"), bool)
        and isinstance(event.get("requested_target"), Mapping)
        and isinstance(event.get("claimed_target"), Mapping)
        and isinstance(event.get("robot_pose_world_xy_yaw"), list)
        and isinstance(event.get("pose_binary64_le_sha256"), str)
        and isinstance(event.get("pose_hex"), list)
        and isinstance(event.get("pose_provenance"), str)
        and isinstance(event.get("physical_manifest_sha256"), str)
    )


def _ordered_reasons(predicates: Mapping[str, bool], order: Sequence[str]) -> list[str]:
    return [reason for reason in order if bool(predicates.get(reason, False))]


def _safe_event_mapping(event: object) -> Mapping[str, Any]:
    return event if isinstance(event, Mapping) else {}


def evaluate_physical_claim_trace(
    trace: Mapping[str, Any],
    physical_manifest: object,
    expected_task_object_ids: Sequence[str],
    expected_task_object_set_sha256: str,
) -> dict[str, Any]:
    """Evaluate one complete claim trace using the frozen two-pass contract."""

    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a JSON object")
    # The returned trace binds the raw attempts. Non-JSON or nonfinite raw input
    # cannot have the required canonical hash and is rejected before evaluation.
    try:
        frozen_trace = _canonical_plain_json(trace)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("trace must be finite canonical-JSON-compatible input") from exc
    if not isinstance(frozen_trace, dict):
        raise TypeError("trace must serialize as a JSON object")

    try:
        frozen_expected_task_object_ids = _canonical_plain_json(
            expected_task_object_ids
        )
    except (OverflowError, TypeError, ValueError):
        frozen_expected_task_object_ids = None
    try:
        frozen_expected_task_object_set_sha256 = _canonical_plain_json(
            expected_task_object_set_sha256
        )
    except (OverflowError, TypeError, ValueError):
        frozen_expected_task_object_set_sha256 = None

    inventory = _manifest_inventory(physical_manifest)
    manifest_hash = inventory["manifest_sha256"]
    manifest_scene_id = getattr(physical_manifest, "scene_id", None)
    envelope = frozen_trace
    raw_attempts_value = envelope.get("controller_claim_attempts")
    raw_attempts = raw_attempts_value if isinstance(raw_attempts_value, list) else []

    expected_ids_valid = (
        isinstance(frozen_expected_task_object_ids, list)
        and all(
            _is_nonempty_utf8_string(value)
            for value in frozen_expected_task_object_ids
        )
    )
    expected_ids = (
        list(frozen_expected_task_object_ids) if expected_ids_valid else []
    )
    expected_ids_are_sorted_unique = (
        expected_ids_valid
        and expected_ids == _utf8_sorted(expected_ids)
        and len(expected_ids) == len(set(expected_ids))
        and set(expected_ids).issubset(set(inventory["all_landmark_ids"]))
    )

    trace_task_value = envelope.get("task_object_ids")
    trace_task_ids = (
        list(trace_task_value)
        if isinstance(trace_task_value, list)
        and all(isinstance(value, str) for value in trace_task_value)
        else []
    )
    trace_task_sorted_unique = (
        isinstance(trace_task_value, list)
        and all(_is_nonempty_utf8_string(value) for value in trace_task_value)
        and trace_task_ids == _utf8_sorted(trace_task_ids)
        and len(trace_task_ids) == len(set(trace_task_ids))
    )
    canonical_task_ids = expected_ids if expected_ids_are_sorted_unique else []

    computed_task_commitment: str | None = None
    if (
        isinstance(manifest_scene_id, str)
        and manifest_scene_id
        and isinstance(manifest_hash, str)
        and expected_ids_are_sorted_unique
    ):
        computed_task_commitment = _task_set_commitment(
            scene_id=manifest_scene_id,
            physical_manifest_sha256=manifest_hash,
            task_object_ids=expected_ids,
        )

    event_maps = [_safe_event_mapping(event) for event in raw_attempts]
    valid_event_ids = [
        event.get("event_id")
        for event in event_maps
        if _is_nonempty_utf8_string(event.get("event_id"))
    ]
    event_id_counts = Counter(valid_event_ids)

    per_event_tick_index_invalid: list[bool] = []
    order_valid = isinstance(raw_attempts_value, list)
    previous_order: tuple[int, int, bytes] | None = None
    previous_event_index: int | None = None
    for event in event_maps:
        tick = event.get("tick")
        event_index = event.get("event_index")
        event_id = event.get("event_id")
        tick_and_index_valid = _is_nonbool_int(tick) and _is_nonbool_int(event_index)
        event_id_valid = _is_nonempty_utf8_string(event_id)
        index_increases = bool(
            _is_nonbool_int(event_index)
            and (previous_event_index is None or event_index > previous_event_index)
        )
        per_event_tick_index_invalid.append(
            not tick_and_index_valid or not index_increases
        )
        if tick_and_index_valid and event_id_valid:
            current_order = (tick, event_index, _utf8_bytes(event_id) or b"")
            if previous_order is not None and current_order <= previous_order:
                order_valid = False
            previous_order = current_order
        else:
            order_valid = False
        if _is_nonbool_int(event_index):
            if previous_event_index is not None and event_index <= previous_event_index:
                order_valid = False
            previous_event_index = event_index

    trace_physical_hash = envelope.get("physical_manifest_sha256")
    event_manifest_commitments_match = all(
        event.get("physical_manifest_sha256") == trace_physical_hash
        for event in event_maps
    )
    trace_identity = envelope.get("trace_id")
    episode_identity = envelope.get("episode_id")
    scene_identity = envelope.get("scene_id")

    global_predicates = {
        "trace_schema_or_key_set_invalid": (
            set(envelope) != _TRACE_KEYS or envelope.get("schema") != RAW_TRACE_SCHEMA
        ),
        "trace_id_missing_or_invalid": not _is_nonempty_utf8_string(trace_identity),
        "episode_id_missing_or_invalid": not _is_nonempty_utf8_string(
            episode_identity
        ),
        "scene_manifest_identity_mismatch": (
            not _is_nonempty_utf8_string(scene_identity)
            or scene_identity != manifest_scene_id
        ),
        "physical_manifest_commitment_mismatch": (
            not _is_sha256(trace_physical_hash)
            or not inventory["manifest_hash_compatible"]
            or trace_physical_hash != manifest_hash
            or not event_manifest_commitments_match
        ),
        "task_object_ids_not_exact_sorted_unique": not trace_task_sorted_unique,
        "task_object_set_mismatch": (
            not expected_ids_are_sorted_unique or trace_task_ids != expected_ids
        ),
        "task_object_commitment_mismatch": (
            computed_task_commitment is None
            or not _is_sha256(frozen_expected_task_object_set_sha256)
            or frozen_expected_task_object_set_sha256 != computed_task_commitment
            or envelope.get("task_object_set_sha256") != computed_task_commitment
        ),
        "evaluator_feedback_to_controller_nonempty": envelope.get(
            "evaluator_feedback_to_controller"
        )
        != [],
        "trace_event_order_invalid": not order_valid,
        "manifest_duplicate_object_id": bool(inventory["duplicate_object_id"]),
        "manifest_invalid_physical_geometry": bool(inventory["geometry_invalid"]),
    }
    global_reasons = _ordered_reasons(global_predicates, UNVERIFIABLE_REASONS[:12])

    provisional: list[dict[str, Any]] = []
    task_set = set(canonical_task_ids)
    for event_position, (raw_event, event) in enumerate(zip(raw_attempts, event_maps)):
        provenance = event.get("pose_provenance")
        legacy_provenance = isinstance(provenance, str) and provenance.startswith("legacy_")
        provenance_valid = isinstance(provenance, str) and (
            provenance in _MODERN_PROVENANCE or legacy_provenance
        )
        requested = _reference_resolution(
            event.get("requested_target"),
            provenance=provenance,
            landmarks=inventory["landmarks"],
        )
        claimed = _reference_resolution(
            event.get("claimed_target"),
            provenance=provenance,
            landmarks=inventory["landmarks"],
        )
        pose = _pose_analysis(event)
        claimed_id = claimed["resolved_object_id"]
        metrics = _physical_metrics(
            pose["authoritative"],
            claimed_id,
            inventory=inventory,
        )

        requested_id = requested["resolved_object_id"]
        requested_in_task = requested_id in task_set if requested_id is not None else None
        claimed_in_task = claimed_id in task_set if claimed_id is not None else None
        identity_passes = (
            bool(
                requested_id == claimed_id
                and requested_in_task
                and claimed_in_task
            )
            if requested_id is not None and claimed_id is not None
            else None
        )

        event_id = event.get("event_id")
        event_predicates = dict(global_predicates)
        event_predicates.update(
            {
                "event_key_set_or_type_invalid": not _event_top_level_types_are_valid(
                    raw_event
                ),
                "event_trace_identity_mismatch": (
                    event.get("trace_id") != trace_identity
                    or event.get("episode_id") != episode_identity
                    or event.get("scene_id") != scene_identity
                ),
                "event_id_missing_or_duplicate": (
                    not _is_nonempty_utf8_string(event_id)
                    or event_id_counts.get(event_id, 0) != 1
                ),
                "claim_tick_or_index_invalid": per_event_tick_index_invalid[
                    event_position
                ],
                "requested_reference_malformed": requested["status"] == "malformed",
                "requested_namespace_forbidden_for_provenance": requested["status"]
                == "forbidden_for_provenance",
                "requested_identity_unresolved": requested["status"] == "unresolved",
                "requested_identity_ambiguous": requested["status"] == "ambiguous",
                "claimed_reference_malformed": claimed["status"] == "malformed",
                "claimed_namespace_forbidden_for_provenance": claimed["status"]
                == "forbidden_for_provenance",
                "claimed_identity_unresolved": claimed["status"] == "unresolved",
                "claimed_identity_ambiguous": claimed["status"] == "ambiguous",
                "pose_provenance_invalid": not provenance_valid,
                "claim_pose_missing_or_nonfinite": pose["missing_or_nonfinite"],
                "claim_pose_precision_commitment_mismatch": pose[
                    "precision_mismatch"
                ],
                "physical_computation_nonfinite": metrics["computation_nonfinite"],
                "legacy_provenance_noncanonical": legacy_provenance,
                "legacy_pose_missing_yaw": pose["legacy_missing_yaw"],
                "legacy_pose_rounded_or_inferred": pose[
                    "legacy_rounded_or_inferred"
                ],
            }
        )
        unverifiable_reasons = _ordered_reasons(
            event_predicates, UNVERIFIABLE_REASONS
        )
        rejection_predicates = {
            "requested_identity_not_in_task_set": requested_in_task is False,
            "claimed_identity_not_in_task_set": claimed_in_task is False,
            "requested_claimed_identity_mismatch": (
                requested_id is not None
                and claimed_id is not None
                and requested_id != claimed_id
            ),
            "outside_inclusive_claim_distance": metrics["distance_passes"] is False,
            "zero_inflation_physical_los_blocked": metrics[
                "line_of_sight_passes"
            ]
            is False,
            "outside_inclusive_claim_bearing": metrics["bearing_passes"] is False,
        }
        available_rejections = _ordered_reasons(
            rejection_predicates, REJECTION_REASONS
        )
        if unverifiable_reasons:
            decision = "unverifiable"
            rejection_reasons: list[str] = []
        elif available_rejections:
            decision = "rejected"
            rejection_reasons = available_rejections
        else:
            decision = "accepted"
            rejection_reasons = []

        target = metrics["target"]
        target_center = (
            [float(value) for value in target.center_xyz_m]
            if _is_box_like(target)
            else None
        )
        target_center_hex = (
            [value.hex() for value in target_center]
            if target_center is not None
            else None
        )

        def value_and_hex(name: str) -> tuple[float | None, str | None]:
            value = metrics[name]
            return (value, value.hex() if isinstance(value, float) else None)

        distance, distance_hex = value_and_hex("distance_m")
        target_bearing, target_bearing_hex = value_and_hex(
            "target_world_bearing_rad"
        )
        signed_bearing, signed_bearing_hex = value_and_hex(
            "signed_bearing_error_rad"
        )
        absolute_bearing, absolute_bearing_hex = value_and_hex(
            "absolute_bearing_error_rad"
        )
        event_core = {
            "schema": EVENT_SCHEMA,
            "evaluator_contract_sha256": EVALUATOR_CONTRACT_SHA256,
            "trace_id": trace_identity if isinstance(trace_identity, str) else None,
            "episode_id": (
                episode_identity if isinstance(episode_identity, str) else None
            ),
            "scene_id": scene_identity if isinstance(scene_identity, str) else None,
            "physical_manifest_sha256": manifest_hash,
            "task_object_ids": list(canonical_task_ids),
            "task_object_set_sha256": computed_task_commitment,
            "event_id": event_id if isinstance(event_id, str) else None,
            "tick": event.get("tick") if _is_nonbool_int(event.get("tick")) else None,
            "event_index": (
                event.get("event_index")
                if _is_nonbool_int(event.get("event_index"))
                else None
            ),
            "pose_provenance": provenance if isinstance(provenance, str) else None,
            "requested_target": deepcopy(event.get("requested_target")),
            "claimed_target": deepcopy(event.get("claimed_target")),
            "requested_resolution": requested,
            "claimed_resolution": claimed,
            "requested_in_task_set": requested_in_task,
            "claimed_in_task_set": claimed_in_task,
            "robot_pose_world_xy_yaw": (
                list(pose["authoritative"])
                if pose["authoritative"] is not None
                else None
            ),
            "pose_hex": pose["canonical_hex"],
            "pose_binary64_le_sha256": pose["packed_hash"],
            "claimed_target_object_id": claimed_id,
            "claimed_target_center_xyz_m": target_center,
            "claimed_target_center_hex": target_center_hex,
            "physical_contract": {
                "claim_distance_m": CLAIM_DISTANCE_M,
                "claim_absolute_bearing_rad": CLAIM_ABSOLUTE_BEARING_RAD,
                "line_of_sight_inflation_m": LINE_OF_SIGHT_INFLATION_M,
                "line_of_sight_geometry": (
                    "closed_segment_oriented_rectangles_scalar_binary64_x_then_y"
                ),
            },
            "distance_m": distance,
            "distance_hex": distance_hex,
            "target_world_bearing_rad": target_bearing,
            "target_world_bearing_hex": target_bearing_hex,
            "signed_bearing_error_rad": signed_bearing,
            "signed_bearing_error_hex": signed_bearing_hex,
            "absolute_bearing_error_rad": absolute_bearing,
            "absolute_bearing_error_hex": absolute_bearing_hex,
            "physical_blockers": metrics["blockers"],
            "factors": {
                "identity_passes": identity_passes,
                "distance_passes": metrics["distance_passes"],
                "line_of_sight_passes": metrics["line_of_sight_passes"],
                "bearing_passes": metrics["bearing_passes"],
            },
            "decision": decision,
            "accepted": decision == "accepted",
            "physically_verified": decision == "accepted",
            "unverifiable_reasons": unverifiable_reasons,
            "rejection_reasons": rejection_reasons,
            "credited": False,
            "duplicate_physical_claim_not_credited": False,
        }
        provisional.append(event_core)

    credited_ids: set[str] = set()
    evaluations: list[dict[str, Any]] = []
    first_credited: dict[str, dict[str, Any]] = {}
    for event in provisional:
        claimed_id = event["claimed_target_object_id"]
        if event["accepted"] and claimed_id in task_set:
            if claimed_id not in credited_ids:
                event["credited"] = True
                credited_ids.add(claimed_id)
                first_credited[claimed_id] = {
                    "object_id": claimed_id,
                    "tick": event["tick"],
                    "event_id": event["event_id"],
                }
            else:
                event["duplicate_physical_claim_not_credited"] = True
        event["content_sha256"] = _content_sha256(event)
        evaluations.append(event)

    credited_object_ids = _utf8_sorted(list(credited_ids))
    unverifiable_counts = {
        reason: sum(reason in event["unverifiable_reasons"] for event in evaluations)
        for reason in UNVERIFIABLE_REASONS
    }
    rejection_counts = {
        reason: sum(reason in event["rejection_reasons"] for event in evaluations)
        for reason in REJECTION_REASONS
    }
    duplicate_count = sum(
        bool(event["duplicate_physical_claim_not_credited"])
        for event in evaluations
    )
    summary_core = {
        "schema": SUMMARY_SCHEMA,
        "evaluator_contract_sha256": EVALUATOR_CONTRACT_SHA256,
        "trace_id": trace_identity if isinstance(trace_identity, str) else None,
        "episode_id": episode_identity if isinstance(episode_identity, str) else None,
        "scene_id": scene_identity if isinstance(scene_identity, str) else None,
        "physical_manifest_sha256": manifest_hash,
        "task_object_ids": list(canonical_task_ids),
        "task_object_set_sha256": computed_task_commitment,
        "attempted_count": len(evaluations),
        "accepted_count": sum(event["decision"] == "accepted" for event in evaluations),
        "rejected_count": sum(event["decision"] == "rejected" for event in evaluations),
        "unverifiable_count": sum(
            event["decision"] == "unverifiable" for event in evaluations
        ),
        "credited_count": sum(bool(event["credited"]) for event in evaluations),
        "duplicate_physical_claim_not_credited_count": duplicate_count,
        "unverifiable_reason_counts": unverifiable_counts,
        "rejection_reason_counts": rejection_counts,
        "aggregate_reason_counts": {
            "duplicate_physical_claim_not_credited": duplicate_count
        },
        "trace_unverifiable_reasons": global_reasons,
        "credited_object_ids": credited_object_ids,
        "first_credited_by_object": [
            first_credited[object_id] for object_id in credited_object_ids
        ],
        "event_content_sha256s": [
            event["content_sha256"] for event in evaluations
        ],
        "all_targets_claimed": bool(
            computed_task_commitment is not None
            and not global_reasons
            and credited_object_ids == list(canonical_task_ids)
        ),
    }
    summary = {**summary_core, "content_sha256": _content_sha256(summary_core)}

    trace_core = {
        "schema": EVALUATED_TRACE_SCHEMA,
        "trace_id": deepcopy(envelope.get("trace_id")),
        "episode_id": deepcopy(envelope.get("episode_id")),
        "scene_id": deepcopy(envelope.get("scene_id")),
        "physical_manifest_sha256": deepcopy(
            envelope.get("physical_manifest_sha256")
        ),
        "task_object_ids": deepcopy(envelope.get("task_object_ids")),
        "task_object_set_sha256": deepcopy(envelope.get("task_object_set_sha256")),
        "controller_claim_attempts": deepcopy(raw_attempts_value),
        "evaluator_feedback_to_controller": deepcopy(
            envelope.get("evaluator_feedback_to_controller")
        ),
        "physical_claim_evaluations": evaluations,
        "physical_claim_summary": summary,
    }
    return {**trace_core, "trace_content_sha256": _content_sha256(trace_core)}
