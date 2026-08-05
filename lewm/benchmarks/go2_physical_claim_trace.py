"""Canonical raw-trace construction for Go2 physical claim observers.

This module constructs evaluator inputs only. It deliberately does not import
the evaluator or expose any physical acceptance result to a controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence

from lewm_worlds.manifest import SceneManifest, manifest_sha256


RAW_TRACE_SCHEMA = "lewm_go2_claim_trace_v1"
TASK_SET_SCHEMA = "lewm_go2_claim_task_set_v1"
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
_PROVENANCE = frozenset(
    {
        "runtime_full_precision",
        "oracle_full_precision",
        "eligibility_candidate_full_precision",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: object) -> object:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def canonical_task_object_ids(
    manifest: SceneManifest,
    task_object_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return an exact UTF-8-sorted, unique manifest-landmark task set."""

    manifest_ids = tuple(
        sorted(
            (_nonempty_string(item.object_id, name="landmark object_id") for item in manifest.landmarks),
            key=lambda value: value.encode("utf-8"),
        )
    )
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("manifest landmark object IDs must be unique")
    selected = manifest_ids if task_object_ids is None else tuple(task_object_ids)
    selected = tuple(
        _nonempty_string(value, name="task object ID") for value in selected
    )
    selected = tuple(sorted(selected, key=lambda value: value.encode("utf-8")))
    if not selected:
        raise ValueError("task object set must be nonempty")
    if len(selected) != len(set(selected)):
        raise ValueError("task object IDs must be unique")
    if not set(selected).issubset(manifest_ids):
        raise ValueError("task object IDs must be exact manifest landmark IDs")
    return selected


def task_object_set_sha256(
    manifest: SceneManifest,
    task_object_ids: Sequence[str] | None = None,
) -> str:
    task_ids = canonical_task_object_ids(manifest, task_object_ids)
    payload = {
        "schema": TASK_SET_SCHEMA,
        "scene_id": _nonempty_string(manifest.scene_id, name="scene_id"),
        "physical_manifest_sha256": manifest_sha256(manifest),
        "task_object_ids": list(task_ids),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def object_id_reference(object_id: str) -> dict[str, str]:
    return {
        "namespace": "object_id",
        "value": _nonempty_string(object_id, name="object_id"),
    }


def build_claim_attempt(
    *,
    manifest: SceneManifest,
    trace_id: str,
    episode_id: str,
    event_id: str,
    tick: int,
    event_index: int,
    requested_target: Mapping[str, str],
    claimed_target: Mapping[str, str],
    robot_pose_world_xy_yaw: Sequence[float],
    pose_provenance: str,
) -> dict[str, Any]:
    """Construct one exact 13-key raw attempt with bit-exact pose binding."""

    trace_id = _nonempty_string(trace_id, name="trace_id")
    episode_id = _nonempty_string(episode_id, name="episode_id")
    event_id = _nonempty_string(event_id, name="event_id")
    if type(tick) is not int or not 0 <= tick <= 2**63 - 1:
        raise ValueError("tick must be a nonnegative signed-64-bit integer")
    if type(event_index) is not int or not 0 <= event_index <= 2**63 - 1:
        raise ValueError("event_index must be a nonnegative signed-64-bit integer")
    if type(pose_provenance) is not str or pose_provenance not in _PROVENANCE:
        raise ValueError("pose_provenance is not canonical")
    if len(robot_pose_world_xy_yaw) != 3:
        raise ValueError("robot pose must contain x, y, and yaw")
    if any(type(value) not in {int, float} for value in robot_pose_world_xy_yaw):
        raise ValueError("robot pose values must be exact JSON numbers")
    pose = tuple(float(value) for value in robot_pose_world_xy_yaw)
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("robot pose must be finite")
    requested = _plain_json(requested_target)
    claimed = _plain_json(claimed_target)
    for name, reference in (("requested_target", requested), ("claimed_target", claimed)):
        if (
            type(reference) is not dict
            or set(reference) != {"namespace", "value"}
            or type(reference["namespace"]) is not str
            or type(reference["value"]) is not str
            or not reference["value"]
            or reference["namespace"] not in {"object_id", "task_color"}
        ):
            raise ValueError(f"{name} must be an exact typed target reference")
    pose_bytes = struct.pack("<3d", *pose)
    event = {
        "trace_id": trace_id,
        "episode_id": episode_id,
        "scene_id": _nonempty_string(manifest.scene_id, name="scene_id"),
        "event_id": event_id,
        "tick": tick,
        "event_index": event_index,
        "requested_target": requested,
        "claimed_target": claimed,
        "robot_pose_world_xy_yaw": list(pose),
        "pose_binary64_le_sha256": hashlib.sha256(pose_bytes).hexdigest(),
        "pose_hex": [value.hex() for value in pose],
        "pose_provenance": pose_provenance,
        "physical_manifest_sha256": manifest_sha256(manifest),
    }
    assert set(event) == _EVENT_KEYS
    return event


def build_claim_trace(
    *,
    manifest: SceneManifest,
    trace_id: str,
    episode_id: str,
    controller_claim_attempts: Sequence[Mapping[str, Any]],
    task_object_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    """Construct a raw trace and return its independent task-set binding."""

    trace_id = _nonempty_string(trace_id, name="trace_id")
    episode_id = _nonempty_string(episode_id, name="episode_id")
    task_ids = canonical_task_object_ids(manifest, task_object_ids)
    task_hash = task_object_set_sha256(manifest, task_ids)
    attempts = _plain_json(controller_claim_attempts)
    if type(attempts) is not list or any(type(item) is not dict for item in attempts):
        raise ValueError("controller_claim_attempts must be a JSON list of objects")
    trace = {
        "schema": RAW_TRACE_SCHEMA,
        "trace_id": trace_id,
        "episode_id": episode_id,
        "scene_id": _nonempty_string(manifest.scene_id, name="scene_id"),
        "physical_manifest_sha256": manifest_sha256(manifest),
        "task_object_ids": list(task_ids),
        "task_object_set_sha256": task_hash,
        "controller_claim_attempts": attempts,
        "evaluator_feedback_to_controller": [],
    }
    return trace, task_ids, task_hash


__all__ = [
    "build_claim_attempt",
    "build_claim_trace",
    "canonical_task_object_ids",
    "object_id_reference",
    "task_object_set_sha256",
]
