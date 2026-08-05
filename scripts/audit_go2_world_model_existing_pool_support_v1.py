#!/usr/bin/env python3
"""Read-only support audit for the bound development H6 train/validation pool.

The audit opens only the two SHA-256-bound corrected-H6 V2 indices, the frozen
main-pool census receipt, and the exact allowlisted train/validation
``frames.jsonl`` leaves selected by the corpus inventory.  It never opens RGB,
raw messages, labels, checkpoints, test, held-out, sealed, or other roles.

The result separates four questions that row counts alone cannot answer:

* whether nearby pre-action physical histories support multiple candidate
  actions across scenes;
* what proprioceptive and ego-motion range the selected rows cover;
* how often half-second candidate transitions have measurable translation or
  yaw; and
* what can and cannot be learned about requested versus executed commands from
  ``frames.jsonl``.

``twist_body`` is an observed robot motion variable.  It is deliberately
reported as requested-versus-realized tracking, never as the executed command:
the selected metadata does not contain the controller's executed/clipped
command tape.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (  # noqa: E402
    BLOCK_SIZE,
    COMMAND_DT_S,
    ENVS_PER_SOURCE,
    PRIMITIVES,
    ROWS_PER_SOURCE,
    SourceRef,
    discover_sources,
)
from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)


SCHEMA = "lewm_go2_world_model_existing_pool_support_audit_v1"
AUDIT_NAMESPACE = "lewm_go2_world_model_existing_pool_support_audit_v1_20260802"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".generated/dev/world_model_existing_pool_support_audit_v1/result.json"
)
CENSUS_RECEIPT = (
    REPO_ROOT / ".generated/go2_recurrent_jepa_main_pool_census_v2/receipt.json"
)
CENSUS_RECEIPT_SHA256 = (
    "aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408"
)
CENSUS_RECEIPT_BYTES = 54_695
EXPECTED_SOURCE_BINDING_SHA256 = (
    "0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696"
)
FRAME_RE = re.compile(r"^frame_([0-9]{6})_env_([0-9]{2})\.png$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KNN_GROUPS = ("all", "family", "history", "family_history")
MOTION_THRESHOLDS = (
    ("translation_ge_0p01m", "translation", 0.01),
    ("translation_ge_0p025m", "translation", 0.025),
    ("translation_ge_0p05m", "translation", 0.05),
    ("translation_ge_0p10m", "translation", 0.10),
    ("abs_yaw_ge_0p025rad", "yaw", 0.025),
    ("abs_yaw_ge_0p05rad", "yaw", 0.05),
    ("abs_yaw_ge_0p10rad", "yaw", 0.10),
    ("abs_yaw_ge_0p20rad", "yaw", 0.20),
)
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
QUANTILE_NAMES = ("min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max")
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class PoolSupportAuditError(RuntimeError):
    """A custody, binding, schema, or metric invariant failed closed."""


def _reject_protected_path(path: Path, *, label: str) -> None:
    for component in Path(path).parts:
        lowered = component.lower()
        if (
            lowered == "sealed_test.json"
            or lowered == "sealed"
            or lowered.startswith("sealed_")
            or lowered in {"heldout", "held_out", "held-out"}
            or lowered.startswith("heldout_")
            or lowered.startswith("held_out_")
            or lowered.startswith("held-out-")
            or lowered in {"raw", "labels", "test"}
        ):
            raise PoolSupportAuditError(f"{label} names out-of-scope material")


def _sha256_file(path: Path) -> tuple[str, int]:
    selected = Path(path)
    _reject_protected_path(selected, label="bound file")
    if selected.is_symlink() or not selected.is_file():
        raise PoolSupportAuditError(f"not a regular non-symlink file: {selected}")
    before = selected.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    descriptor = os.open(selected, _FILE_FLAGS)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PoolSupportAuditError(f"not a regular file: {selected}")
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = selected.stat(follow_symlinks=False)
    fingerprints = {
        (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)
        for value in (before, opened, after_open, after)
    }
    if len(fingerprints) != 1:
        raise PoolSupportAuditError(f"file changed while hashing: {selected}")
    return digest.hexdigest(), int(opened.st_size)


def _strict_json(raw: bytes, *, label: str) -> Any:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PoolSupportAuditError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise PoolSupportAuditError(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except PoolSupportAuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PoolSupportAuditError(f"invalid JSON in {label}") from error


def _finite(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PoolSupportAuditError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PoolSupportAuditError(f"{label} is not finite")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PoolSupportAuditError(f"{label} is not an object")
    return value


def _sequence_scene(source: SourceRef) -> str:
    parts = source.sequence.split("_", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise PoolSupportAuditError("source sequence identity changed")
    return parts[1]


def _source_path(repo_root: Path, source: SourceRef) -> Path:
    selected = (
        repo_root
        / ".generated/datagen_full/rollout"
        / source.role
        / source.family
        / source.chunk
        / "plan"
        / source.sequence
        / "frames.jsonl"
    )
    _reject_protected_path(selected, label="source path")
    return selected


def _frame_index(leaf: str, *, scene_id: str) -> tuple[int, int]:
    path = PurePosixPath(leaf)
    if path.parts[:2] != (scene_id, "rgb") or len(path.parts) != 3:
        raise PoolSupportAuditError("H6 RGB identity left the scene/rgb contract")
    match = FRAME_RE.fullmatch(path.name)
    if match is None:
        raise PoolSupportAuditError("H6 RGB filename changed")
    frame, environment = map(int, match.groups())
    if frame % ENVS_PER_SOURCE != environment:
        raise PoolSupportAuditError("H6 frame/environment identity changed")
    return frame, environment


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _pose(frame: Mapping[str, Any], *, label: str) -> tuple[float, float, float, float]:
    base = _mapping(frame.get("base_pose_world"), label=f"{label}.base_pose_world")
    position = _mapping(base.get("position"), label=f"{label}.position")
    rpy = _mapping(frame.get("base_rpy_rad"), label=f"{label}.base_rpy_rad")
    return (
        _finite(position.get("x"), label=f"{label}.x"),
        _finite(position.get("y"), label=f"{label}.y"),
        _finite(position.get("z"), label=f"{label}.z"),
        _finite(rpy.get("yaw"), label=f"{label}.yaw"),
    )


def body_delta(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float, float]:
    """Return body-forward, body-left, yaw, planar, and vertical displacement."""

    if len(left) != 4 or len(right) != 4:
        raise PoolSupportAuditError("pose delta requires x/y/z/yaw")
    dx = float(right[0]) - float(left[0])
    dy = float(right[1]) - float(left[1])
    yaw = float(left[3])
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return (
        forward,
        lateral,
        _wrap_angle(float(right[3]) - yaw),
        math.hypot(dx, dy),
        float(right[2]) - float(left[2]),
    )


def _twist(frame: Mapping[str, Any], *, label: str) -> tuple[float, ...]:
    twist = _mapping(frame.get("twist_body"), label=f"{label}.twist_body")
    linear = _mapping(twist.get("linear"), label=f"{label}.linear")
    angular = _mapping(twist.get("angular"), label=f"{label}.angular")
    return tuple(
        _finite(group.get(axis), label=f"{label}.{kind}.{axis}")
        for kind, group in (("linear", linear), ("angular", angular))
        for axis in ("x", "y", "z")
    )


def _joints(frame: Mapping[str, Any], *, label: str) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    joints = _mapping(frame.get("joint_state"), label=f"{label}.joint_state")
    names = joints.get("names")
    positions = joints.get("position")
    velocities = joints.get("velocity")
    efforts = joints.get("effort")
    if (
        not isinstance(names, list)
        or len(names) != 12
        or len(set(names)) != 12
        or not isinstance(positions, list)
        or len(positions) != 12
        or not isinstance(velocities, list)
        or len(velocities) != 12
        or not isinstance(efforts, list)
    ):
        raise PoolSupportAuditError(f"{label} joint schema changed")
    return (
        tuple(_finite(value, label=f"{label}.joint_position") for value in positions),
        tuple(_finite(value, label=f"{label}.joint_velocity") for value in velocities),
        len(efforts) == 12,
    )


def _recursive_execution_keys(value: Any, *, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if "executed" in lowered or "clipp" in lowered:
                result.add(path)
            result.update(_recursive_execution_keys(child, prefix=path))
    elif isinstance(value, list):
        for child in value:
            result.update(_recursive_execution_keys(child, prefix=prefix))
    return result


def _command_tape(
    frames: Sequence[Mapping[str, Any]],
    *,
    expected_primitive: str,
    preaction_timestamp_ns: int,
    label: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], set[str]]:
    observed: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None
    execution_keys: set[str] = set()
    for tick, frame in enumerate(frames):
        context = _mapping(frame.get("command_context"), label=f"{label}.command_context")
        if (
            context.get("primitive_name") != expected_primitive
            or context.get("block_size") != BLOCK_SIZE
            or abs(_finite(context.get("command_dt_s"), label=f"{label}.command_dt_s") - COMMAND_DT_S) > 2e-4
            or context.get("timestamp_ns") != preaction_timestamp_ns
        ):
            raise PoolSupportAuditError(f"{label} request-time action context changed")
        tapes: list[tuple[float, ...]] = []
        for field in ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"):
            values = context.get(field)
            if not isinstance(values, list) or len(values) != BLOCK_SIZE:
                raise PoolSupportAuditError(f"{label}.{field} command tape changed")
            tapes.append(tuple(_finite(value, label=f"{label}.{field}") for value in values))
        current = (tapes[0], tapes[1], tapes[2])
        if observed is None:
            observed = current
        elif current != observed:
            raise PoolSupportAuditError(f"{label} command tape drifted within block")
        execution_keys.update(_recursive_execution_keys(frame))
        expected_frame_timestamp = preaction_timestamp_ns + int(round((tick + 1) * COMMAND_DT_S * 1e9))
        if abs(int(frame.get("timestamp_ns", -1)) - expected_frame_timestamp) > 200_000:
            raise PoolSupportAuditError(f"{label} candidate tick timing changed")
    if observed is None:
        raise PoolSupportAuditError(f"{label} candidate block is empty")
    return observed[0], observed[1], observed[2], execution_keys


def _extract_row(
    row: Mapping[str, Any], frames: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    frame_indices = tuple(int(value) for value in row["context_frame_indices"])
    candidate_indices = tuple(int(value) for value in row["candidate_tick_indices"])
    context = [frames[index] for index in frame_indices]
    candidate = [frames[index] for index in candidate_indices]
    actions = tuple(int(value) for value in row["actions"])
    scene_id = str(row["scene_id"])
    label = f"{row['role']}:{row['index']}"
    for expected_index, frame in zip((*frame_indices, *candidate_indices), (*context, *candidate), strict=True):
        if frame.get("frame_index") != expected_index:
            raise PoolSupportAuditError(f"{label} selected frame identity changed")
        episode = _mapping(frame.get("episode"), label=f"{label}.episode")
        if episode.get("split") != row["role"]:
            raise PoolSupportAuditError(f"{label} selected frame role changed")

    for context_position, action_position in ((1, 0), (2, 1)):
        context_value = _mapping(
            context[context_position].get("command_context"),
            label=f"{label}.history_context",
        )
        if context_value.get("primitive_name") != PRIMITIVES[actions[action_position]]:
            raise PoolSupportAuditError(f"{label} historical action provenance changed")

    poses = [_pose(frame, label=f"{label}.context") for frame in context]
    candidate_pose = _pose(candidate[-1], label=f"{label}.candidate_endpoint")
    delta01 = body_delta(poses[0], poses[1])
    delta12 = body_delta(poses[1], poses[2])
    outcome = body_delta(poses[2], candidate_pose)
    current_twist = _twist(context[2], label=f"{label}.current")
    joints, joint_velocities, effort_available = _joints(context[2], label=f"{label}.current")
    current_rpy = _mapping(context[2].get("base_rpy_rad"), label=f"{label}.current_rpy")
    roll = _finite(current_rpy.get("roll"), label=f"{label}.roll")
    pitch = _finite(current_rpy.get("pitch"), label=f"{label}.pitch")
    state_vector = (
        *delta01[:3],
        *delta12[:3],
        poses[2][2],
        roll,
        pitch,
        *current_twist,
        *joints,
        *joint_velocities,
    )
    requested_vx, requested_vy, requested_yaw, execution_keys = _command_tape(
        candidate,
        expected_primitive=PRIMITIVES[actions[2]],
        preaction_timestamp_ns=int(context[2]["timestamp_ns"]),
        label=label,
    )
    realized_twists = np.asarray(
        [_twist(frame, label=f"{label}.candidate_twist") for frame in candidate],
        dtype=np.float64,
    )
    realized_mean = realized_twists.mean(axis=0)
    return {
        "role": row["role"],
        "index": int(row["index"]),
        "family": row["family"],
        "scene_id": scene_id,
        "history_actions": list(actions[:2]),
        "candidate_action": actions[2],
        "state_vector": list(state_vector),
        "current": {
            "base_z_m": poses[2][2],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "twist_linear_x_mps": current_twist[0],
            "twist_linear_y_mps": current_twist[1],
            "twist_linear_z_mps": current_twist[2],
            "twist_angular_x_radps": current_twist[3],
            "twist_angular_y_radps": current_twist[4],
            "twist_angular_z_radps": current_twist[5],
            "joint_position_l2": float(np.linalg.norm(joints)),
            "joint_velocity_l2": float(np.linalg.norm(joint_velocities)),
        },
        "history_egomotion": {
            "d01_forward_m": delta01[0],
            "d01_lateral_m": delta01[1],
            "d01_yaw_rad": delta01[2],
            "d12_forward_m": delta12[0],
            "d12_lateral_m": delta12[1],
            "d12_yaw_rad": delta12[2],
        },
        "candidate_outcome": {
            "forward_m": outcome[0],
            "lateral_m": outcome[1],
            "yaw_rad": outcome[2],
            "planar_m": outcome[3],
            "vertical_m": outcome[4],
        },
        "requested": {
            "vx_tape": list(requested_vx),
            "vy_tape": list(requested_vy),
            "yaw_rate_tape": list(requested_yaw),
            "mean_vx_mps": float(np.mean(requested_vx)),
            "mean_vy_mps": float(np.mean(requested_vy)),
            "mean_yaw_rate_radps": float(np.mean(requested_yaw)),
        },
        "realized": {
            "mean_twist_linear_x_mps": float(realized_mean[0]),
            "mean_twist_linear_y_mps": float(realized_mean[1]),
            "mean_twist_angular_z_radps": float(realized_mean[5]),
        },
        "execution_or_clipping_metadata_keys": sorted(execution_keys),
        "joint_effort_available": effort_available,
    }


def _scan_source(task: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(str(task["repo_root"]))
    source = SourceRef(**task["source"])
    path = _source_path(repo_root, source)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source.byte_count:
        raise PoolSupportAuditError(f"source file binding changed: {path}")
    wanted = {int(value) for row in task["rows"] for value in (*row["context_frame_indices"], *row["candidate_tick_indices"])}
    captured: dict[int, Mapping[str, Any]] = {}
    digest = hashlib.sha256()
    byte_count = 0
    descriptor = os.open(path, _FILE_FLAGS)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != source.byte_count:
            raise PoolSupportAuditError("source descriptor binding changed")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            for frame_index, raw in enumerate(stream):
                digest.update(raw)
                byte_count += len(raw)
                if frame_index in wanted:
                    payload = _strict_json(raw, label=f"{source.sequence}:{frame_index}")
                    if not isinstance(payload, Mapping) or payload.get("frame_index") != frame_index:
                        raise PoolSupportAuditError("selected source row identity changed")
                    captured[frame_index] = payload
            row_count = frame_index + 1 if byte_count else 0
        after = path.stat(follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        row_count != ROWS_PER_SOURCE
        or byte_count != source.byte_count
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or set(captured) != wanted
    ):
        raise PoolSupportAuditError(f"source scan contract failed: {source.sequence}")
    rows = [_extract_row(row, captured) for row in task["rows"]]
    return {
        "ordinal": source.ordinal,
        "binding_row": [
            source.role,
            source.family,
            source.chunk,
            source.sequence,
            source.byte_count,
            digest.hexdigest(),
        ],
        "rows": rows,
        "opened_frames_jsonl_count": 1,
        "parsed_selected_metadata_row_count": len(captured),
        "rgb_open_count": 0,
        "raw_open_count": 0,
        "label_open_count": 0,
        "protected_open_count": 0,
    }


def _entropy(values: Iterable[int]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -math.fsum(
        count / total * math.log2(count / total) for count in counts.values()
    )


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise PoolSupportAuditError("quantile input is empty or non-finite")
    observed = np.quantile(array, QUANTILES, method="linear")
    return {name: float(value) for name, value in zip(QUANTILE_NAMES, observed, strict=True)}


def summarize_index(rows_by_role: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    role_scenes: dict[str, set[str]] = {}
    for role, rows in rows_by_role.items():
        scenes = {row.scene_id for row in rows}
        role_scenes[role] = scenes
        scene_rows: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            scene_rows[row.scene_id].append(row)
        exact_histories: dict[tuple[Any, ...], set[int]] = defaultdict(set)
        for row in rows:
            key = (row.scene_id, *row.rgb[:3], *row.actions[:2])
            exact_histories[key].add(row.actions[2])
        full_action_scenes = sum(
            len({row.actions[2] for row in scene_values}) == len(PRIMITIVES)
            for scene_values in scene_rows.values()
        )
        result[role] = {
            "row_count": len(rows),
            "scene_count": len(scenes),
            "family_row_counts": dict(sorted(Counter(row.family for row in rows).items())),
            "family_scene_counts": {
                family: len({row.scene_id for row in rows if row.family == family})
                for family in sorted({row.family for row in rows})
            },
            "rows_per_scene": _quantiles(len(value) for value in scene_rows.values()),
            "candidate_action_counts": [
                sum(row.actions[2] == action for row in rows)
                for action in range(len(PRIMITIVES))
            ],
            "candidate_action_entropy_bits": _entropy(row.actions[2] for row in rows),
            "candidate_action_normalized_entropy": (
                _entropy(row.actions[2] for row in rows) / math.log2(len(PRIMITIVES))
            ),
            "scene_macro_candidate_action_entropy_bits": float(
                np.mean([
                    _entropy(row.actions[2] for row in scene_values)
                    for scene_values in scene_rows.values()
                ])
            ),
            "minimum_scene_candidate_action_entropy_bits": min(
                _entropy(row.actions[2] for row in scene_values)
                for scene_values in scene_rows.values()
            ),
            "full_nine_action_support_scene_count": full_action_scenes,
            "full_nine_action_support_scene_fraction": full_action_scenes / len(scene_rows),
            "exact_preaction_history_count": len(exact_histories),
            "exact_preaction_history_duplicate_count": len(rows) - len(exact_histories),
            "exact_preaction_histories_with_multiple_candidate_actions": sum(
                len(actions) > 1 for actions in exact_histories.values()
            ),
            "history_pair_count": len({tuple(row.actions[:2]) for row in rows}),
            "candidate_action_given_history_pair_entropy_bits": {
                f"{left},{right}": _entropy(
                    row.actions[2]
                    for row in rows
                    if tuple(row.actions[:2]) == (left, right)
                )
                for left, right in sorted({tuple(row.actions[:2]) for row in rows})
            },
        }
    overlap = role_scenes.get("train", set()) & role_scenes.get("val", set())
    result["cross_role_scene_overlap_count"] = len(overlap)
    return result


def summarize_census(census: Mapping[str, Any]) -> dict[str, Any]:
    """Compact the frozen full-pool census without reopening source payloads."""

    totals = _mapping(census.get("totals"), label="census.totals")
    by_role_family = _mapping(
        census.get("by_role_family"), label="census.by_role_family"
    )
    action_counts: dict[str, dict[str, int]] = {}
    for role in ("train", "val"):
        families = _mapping(by_role_family.get(role), label=f"census.{role}")
        counts: Counter[str] = Counter()
        for family, value in families.items():
            payload = _mapping(value, label=f"census.{role}.{family}")
            positions = _mapping(
                payload.get("action_position_counts"),
                label=f"census.{role}.{family}.action_position_counts",
            )
            for primitive in PRIMITIVES:
                count = positions.get(f"p2:{primitive}")
                if type(count) is not int or count < 0:
                    raise PoolSupportAuditError("census p2 action count changed")
                counts[primitive] += count
        action_counts[role] = {
            primitive: counts[primitive] for primitive in PRIMITIVES
        }
    required_totals = (
        "byte_count",
        "packed_h6",
        "primitive_transitions",
        "row_count",
        "sliding_h6",
        "source_count",
    )
    if any(type(totals.get(name)) is not int or totals[name] < 1 for name in required_totals):
        raise PoolSupportAuditError("census totals changed")
    return {
        "totals": {name: int(totals[name]) for name in required_totals},
        "sliding_h6_candidate_position_p2_action_counts": action_counts,
        "interpretation": (
            "full-corpus sliding-H6 requested-action counts from the frozen census; "
            "local physical-neighborhood metrics below use the bound 16,000/2,048 pack"
        ),
    }


def _group_key(row: Mapping[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "all":
        return ()
    if mode == "family":
        return (row["family"],)
    if mode == "history":
        return tuple(row["history_actions"])
    if mode == "family_history":
        return (row["family"], *row["history_actions"])
    raise PoolSupportAuditError(f"unknown KNN grouping: {mode}")


def _majority(values: Sequence[int]) -> int:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def neighborhood_support(
    train_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    *,
    k: int,
    exclude_same_scene: bool,
) -> dict[str, Any]:
    if k < 1 or not train_rows or not query_rows:
        raise PoolSupportAuditError("KNN audit inputs are invalid")
    train_features = np.asarray([row["state_vector"] for row in train_rows], dtype=np.float64)
    query_features = np.asarray([row["state_vector"] for row in query_rows], dtype=np.float64)
    if train_features.ndim != 2 or query_features.shape[1] != train_features.shape[1]:
        raise PoolSupportAuditError("KNN feature shapes changed")
    center = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    inactive = scale < 1e-8
    scale[inactive] = 1.0
    train_standard = (train_features - center) / scale
    query_standard = (query_features - center) / scale
    output: dict[str, Any] = {
        "k": k,
        "feature_dimension": int(train_features.shape[1]),
        "inactive_train_feature_count": int(inactive.sum()),
        "distance": "euclidean_after_train_feature_standardization",
    }
    for mode in KNN_GROUPS:
        references_by_group: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        queries_by_group: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index, row in enumerate(train_rows):
            references_by_group[_group_key(row, mode)].append(index)
        for index, row in enumerate(query_rows):
            queries_by_group[_group_key(row, mode)].append(index)
        records: list[dict[str, Any]] = []
        missing_reference_group_count = 0
        for group, query_indices in queries_by_group.items():
            reference_indices = references_by_group.get(group, [])
            if not reference_indices:
                missing_reference_group_count += len(query_indices)
                continue
            group_features = train_standard[reference_indices]
            tree = cKDTree(group_features)
            max_same_scene = max(Counter(train_rows[index]["scene_id"] for index in reference_indices).values())
            search_k = min(len(reference_indices), k + max_same_scene + 1)
            distances, positions = tree.query(query_standard[query_indices], k=search_k)
            distances = np.atleast_2d(distances)
            positions = np.atleast_2d(positions)
            if len(query_indices) == 1 and distances.shape[0] != 1:
                distances = distances.reshape(1, -1)
                positions = positions.reshape(1, -1)
            group_action_entropy = _entropy(
                train_rows[index]["candidate_action"] for index in reference_indices
            )
            for row_slot, query_index in enumerate(query_indices):
                query = query_rows[query_index]
                neighbors: list[int] = []
                neighbor_distances: list[float] = []
                for distance, local_position in zip(distances[row_slot], positions[row_slot], strict=True):
                    reference_index = reference_indices[int(local_position)]
                    reference = train_rows[reference_index]
                    if exclude_same_scene and reference["scene_id"] == query["scene_id"]:
                        continue
                    if query is reference:
                        continue
                    neighbors.append(int(reference["candidate_action"]))
                    neighbor_distances.append(float(distance))
                    if len(neighbors) == k:
                        break
                if len(neighbors) < k:
                    continue
                counts = Counter(neighbors)
                factual = int(query["candidate_action"])
                entropy = _entropy(neighbors)
                records.append(
                    {
                        "factual": factual,
                        "predicted": _majority(neighbors),
                        "entropy": entropy,
                        "effective_actions": 2.0**entropy,
                        "unique_actions": len(counts),
                        "top_share": max(counts.values()) / k,
                        "factual_support": counts[factual] / k,
                        "factual_zero": counts[factual] == 0,
                        "kth_distance": neighbor_distances[-1],
                        "group_entropy": group_action_entropy,
                    }
                )
        eligible = len(records)
        if not records:
            output[mode] = {
                "eligible_query_count": 0,
                "eligible_query_fraction": 0.0,
                "missing_reference_group_count": missing_reference_group_count,
            }
            continue
        factual = np.asarray([record["factual"] for record in records], dtype=np.int64)
        predicted = np.asarray([record["predicted"] for record in records], dtype=np.int64)
        recalls = []
        per_action_support: dict[str, Any] = {}
        for action in range(len(PRIMITIVES)):
            mask = factual == action
            if mask.any():
                recalls.append(float(np.mean(predicted[mask] == action)))
                values = [record["factual_support"] for record, keep in zip(records, mask, strict=True) if keep]
                total_query_count = sum(
                    int(row["candidate_action"]) == action for row in query_rows
                )
                supported_count = sum(value > 0.0 for value in values)
                per_action_support[PRIMITIVES[action]] = {
                    "total_query_count": total_query_count,
                    "eligible_query_count": len(values),
                    "eligible_query_fraction": len(values) / total_query_count,
                    "mean_local_factual_action_share": float(np.mean(values)),
                    "zero_local_factual_action_support_fraction": float(np.mean(np.asarray(values) == 0.0)),
                    "k_neighbor_and_factual_action_support_query_fraction": (
                        supported_count / total_query_count
                    ),
                }
        local_entropy = np.asarray([record["entropy"] for record in records])
        group_entropy = np.asarray([record["group_entropy"] for record in records])
        output[mode] = {
            "eligible_query_count": eligible,
            "eligible_query_fraction": eligible / len(query_rows),
            "missing_reference_group_count": missing_reference_group_count,
            "mean_local_action_entropy_bits": float(local_entropy.mean()),
            "local_action_entropy_bits": _quantiles(local_entropy),
            "mean_effective_local_action_count": float(np.mean([record["effective_actions"] for record in records])),
            "mean_unique_local_action_count": float(np.mean([record["unique_actions"] for record in records])),
            "fraction_with_at_least_three_local_actions": float(np.mean([record["unique_actions"] >= 3 for record in records])),
            "fraction_with_at_least_five_local_actions": float(np.mean([record["unique_actions"] >= 5 for record in records])),
            "fraction_with_all_nine_local_actions": float(np.mean([record["unique_actions"] == len(PRIMITIVES) for record in records])),
            "mean_top_local_action_share": float(np.mean([record["top_share"] for record in records])),
            "mean_local_factual_action_share": float(np.mean([record["factual_support"] for record in records])),
            "zero_local_factual_action_support_fraction": float(np.mean([record["factual_zero"] for record in records])),
            "k_neighbor_and_factual_action_support_query_fraction": float(
                sum(not record["factual_zero"] for record in records)
                / len(query_rows)
            ),
            "knn_action_accuracy": float(np.mean(factual == predicted)),
            "knn_action_balanced_accuracy": float(np.mean(recalls)),
            "mean_reference_group_action_entropy_bits": float(group_entropy.mean()),
            "mean_local_to_group_entropy_ratio": float(np.mean(np.divide(local_entropy, group_entropy, out=np.zeros_like(local_entropy), where=group_entropy > 0))),
            "kth_neighbor_standardized_distance": _quantiles(record["kth_distance"] for record in records),
            "per_action_factual_support": per_action_support,
        }
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size != y.size or x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_physical(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise PoolSupportAuditError("physical audit row population is empty")
    current_fields = sorted(rows[0]["current"])
    history_fields = sorted(rows[0]["history_egomotion"])
    outcome_fields = sorted(rows[0]["candidate_outcome"])
    by_action: dict[str, Any] = {}
    for action, primitive in enumerate(PRIMITIVES):
        selected = [row for row in rows if row["candidate_action"] == action]
        if not selected:
            raise PoolSupportAuditError(f"physical audit lacks action {primitive}")
        motion = {}
        for name, kind, threshold in MOTION_THRESHOLDS:
            if kind == "translation":
                values = [row["candidate_outcome"]["planar_m"] for row in selected]
            else:
                values = [abs(row["candidate_outcome"]["yaw_rad"]) for row in selected]
            motion[name] = float(np.mean(np.asarray(values) >= threshold))
        meaningful = [
            row["candidate_outcome"]["planar_m"] >= 0.025
            or abs(row["candidate_outcome"]["yaw_rad"]) >= 0.05
            for row in selected
        ]
        near_static = [
            row["candidate_outcome"]["planar_m"] < 0.01
            and abs(row["candidate_outcome"]["yaw_rad"]) < 0.025
            for row in selected
        ]
        by_action[primitive] = {
            "row_count": len(selected),
            "outcome_forward_m": _quantiles(row["candidate_outcome"]["forward_m"] for row in selected),
            "outcome_planar_m": _quantiles(row["candidate_outcome"]["planar_m"] for row in selected),
            "outcome_yaw_rad": _quantiles(row["candidate_outcome"]["yaw_rad"] for row in selected),
            "meaningful_motion_fraction_translation_0p025m_or_yaw_0p05rad": float(np.mean(meaningful)),
            "near_static_fraction_translation_lt_0p01m_and_yaw_lt_0p025rad": float(np.mean(near_static)),
            **motion,
        }
    request_tapes: dict[int, set[tuple[float, ...]]] = defaultdict(set)
    execution_keys: Counter[str] = Counter()
    for row in rows:
        request = row["requested"]
        request_tapes[int(row["candidate_action"])].add(
            tuple(request["vx_tape"] + request["vy_tape"] + request["yaw_rate_tape"])
        )
        execution_keys.update(row["execution_or_clipping_metadata_keys"])
    requested_vx = [row["requested"]["mean_vx_mps"] for row in rows]
    requested_vy = [row["requested"]["mean_vy_mps"] for row in rows]
    requested_yaw = [row["requested"]["mean_yaw_rate_radps"] for row in rows]
    realized_vx = [row["realized"]["mean_twist_linear_x_mps"] for row in rows]
    realized_vy = [row["realized"]["mean_twist_linear_y_mps"] for row in rows]
    realized_yaw = [row["realized"]["mean_twist_angular_z_radps"] for row in rows]
    return {
        "row_count": len(rows),
        "current_proprio_quantiles": {
            field: _quantiles(row["current"][field] for row in rows)
            for field in current_fields
        },
        "history_egomotion_quantiles": {
            field: _quantiles(row["history_egomotion"][field] for row in rows)
            for field in history_fields
        },
        "candidate_outcome_quantiles": {
            field: _quantiles(row["candidate_outcome"][field] for row in rows)
            for field in outcome_fields
        },
        "joint_effort_available_row_count": sum(row["joint_effort_available"] for row in rows),
        "motion_density_by_action": by_action,
        "requested_command": {
            "unique_tape_count_by_action": {
                PRIMITIVES[action]: len(request_tapes[action])
                for action in range(len(PRIMITIVES))
            },
            "representative_tape_by_action": {
                PRIMITIVES[action]: list(sorted(request_tapes[action])[0])
                for action in range(len(PRIMITIVES))
            },
        },
        "executed_or_clipping_metadata": {
            "matching_key_row_count": sum(bool(row["execution_or_clipping_metadata_keys"]) for row in rows),
            "matching_key_counts": dict(sorted(execution_keys.items())),
            "exact_requested_vs_executed_command_comparison_available": bool(execution_keys),
            "interpretation": (
                "twist_body is observed realized robot motion, not the controller's "
                "executed/clipped command tape"
            ),
        },
        "requested_vs_realized_tracking": {
            "requested_vx_to_realized_twist_x_pearson": _pearson(requested_vx, realized_vx),
            "requested_vy_to_realized_twist_y_pearson": _pearson(requested_vy, realized_vy),
            "requested_yaw_rate_to_realized_twist_yaw_pearson": _pearson(requested_yaw, realized_yaw),
            "realized_minus_requested_vx_mps": _quantiles(np.asarray(realized_vx) - np.asarray(requested_vx)),
            "realized_minus_requested_vy_mps": _quantiles(np.asarray(realized_vy) - np.asarray(requested_vy)),
            "realized_minus_requested_yaw_rate_radps": _quantiles(np.asarray(realized_yaw) - np.asarray(requested_yaw)),
        },
    }


def _row_spec(row: Any) -> dict[str, Any]:
    context = [_frame_index(row.rgb[position], scene_id=row.scene_id)[0] for position in range(3)]
    next_frame = _frame_index(row.rgb[3], scene_id=row.scene_id)[0]
    if next_frame - context[2] != BLOCK_SIZE * ENVS_PER_SOURCE:
        raise PoolSupportAuditError("candidate endpoint cadence changed")
    candidate = [context[2] + tick * ENVS_PER_SOURCE for tick in range(1, BLOCK_SIZE + 1)]
    if candidate[-1] != next_frame:
        raise PoolSupportAuditError("candidate tick derivation changed")
    return {
        "role": row.role,
        "index": row.index,
        "family": row.family,
        "scene_id": row.scene_id,
        "actions": list(row.actions),
        "context_frame_indices": context,
        "candidate_tick_indices": candidate,
    }


def _binding_digest(binding_rows: Sequence[Sequence[Any]]) -> str:
    raw = json.dumps(
        list(binding_rows),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    selected = Path(path)
    _reject_protected_path(selected, label="output path")
    selected.parent.mkdir(parents=True, exist_ok=True)
    if selected.exists() or selected.is_symlink():
        raise PoolSupportAuditError(f"refusing to overwrite output: {selected}")
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with selected.open("x", encoding="utf-8") as handle:
        handle.write(raw)


def run_audit(repo_root: Path, *, workers: int, knn_k: int) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(repo_root).resolve()
    if root != REPO_ROOT.resolve():
        raise PoolSupportAuditError("repository root must be this exact checkout")
    _reject_protected_path(root, label="repository root")
    census_sha, census_bytes = _sha256_file(CENSUS_RECEIPT)
    if census_sha != CENSUS_RECEIPT_SHA256 or census_bytes != CENSUS_RECEIPT_BYTES:
        raise PoolSupportAuditError("frozen main-pool census receipt binding changed")
    census = _strict_json(CENSUS_RECEIPT.read_bytes(), label="census receipt")
    if (
        not isinstance(census, Mapping)
        or census.get("scope", {}).get("roles") != ["train", "val"]
        or census.get("identity", {}).get("ordered_source_content_binding_sha256")
        != EXPECTED_SOURCE_BINDING_SHA256
    ):
        raise PoolSupportAuditError("frozen census scope or identity changed")

    rows_by_role: dict[str, Sequence[Any]] = {}
    index_audits: dict[str, Any] = {}
    for role in ("train", "val"):
        rows, audit = h6.load_bound_index(root, role=role)
        rows_by_role[role] = rows
        index_audits[role] = audit
    all_rows = [*rows_by_role["train"], *rows_by_role["val"]]
    specs_by_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        specs_by_scene[(row.role, row.scene_id)].append(_row_spec(row))

    sources, discovery_access = discover_sources(root)
    source_by_scene: dict[tuple[str, str], SourceRef] = {}
    for source in sources:
        key = (source.role, _sequence_scene(source))
        if key in source_by_scene:
            raise PoolSupportAuditError("duplicate allowlisted source scene")
        source_by_scene[key] = source
    if set(source_by_scene) != set(specs_by_scene):
        raise PoolSupportAuditError("bound H6 scene inventory and source inventory differ")

    tasks = [
        {
            "repo_root": str(root),
            "source": {
                "role": source.role,
                "family": source.family,
                "chunk": source.chunk,
                "sequence": source.sequence,
                "byte_count": source.byte_count,
                "ordinal": source.ordinal,
            },
            "rows": specs_by_scene[key],
        }
        for key, source in sorted(source_by_scene.items(), key=lambda item: item[1].ordinal)
    ]
    scan_results: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_scan_source, task) for task in tasks]
        for future in as_completed(futures):
            scan_results.append(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"scanned {completed}/{len(tasks)} bound train/val metadata sources", flush=True)
    scan_results.sort(key=lambda value: int(value["ordinal"]))
    binding_rows = [value["binding_row"] for value in scan_results]
    observed_binding = _binding_digest(binding_rows)
    if observed_binding != EXPECTED_SOURCE_BINDING_SHA256:
        raise PoolSupportAuditError("live ordered source-content binding changed")
    physical_rows = [row for result in scan_results for row in result["rows"]]
    physical_rows.sort(key=lambda row: (row["role"], row["index"]))
    if len(physical_rows) != len(all_rows):
        raise PoolSupportAuditError("physical extraction row count changed")
    train_physical = [row for row in physical_rows if row["role"] == "train"]
    val_physical = [row for row in physical_rows if row["role"] == "val"]

    train_support = neighborhood_support(
        train_physical,
        train_physical,
        k=knn_k,
        exclude_same_scene=True,
    )
    validation_support = neighborhood_support(
        train_physical,
        val_physical,
        k=knn_k,
        exclude_same_scene=False,
    )
    elapsed = time.monotonic() - started
    return {
        "schema": SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_AUDIT",
        "date": "2026-08-02",
        "claim_scope": (
            "bound corrected-H6 V2 development train/validation metadata only; "
            "not held-out, qualification, promotion, or deployed-planning evidence"
        ),
        "result_interpretation": {
            "local_overlap": (
                "descriptive positivity diagnostic in standardized physical-history space; "
                "it does not include RGB appearance and cannot prove causal identifiability"
            ),
            "requested_vs_executed": (
                "exact comparison is unavailable because frames.jsonl contains requested "
                "command context and observed twist, not executed/clipped command arrays"
            ),
            "motion_density": (
                "threshold sweep is descriptive; no threshold is a preregistered pass gate"
            ),
        },
        "inputs": {
            "index_audits": index_audits,
            "census_receipt": {
                "path": CENSUS_RECEIPT.relative_to(root).as_posix(),
                "file_sha256": census_sha,
                "byte_count": census_bytes,
            },
            "source_inventory_count": len(sources),
            "live_ordered_source_content_binding_sha256": observed_binding,
            "expected_ordered_source_content_binding_sha256": EXPECTED_SOURCE_BINDING_SHA256,
            "full_pool_census_summary": summarize_census(census),
        },
        "access_accounting": {
            "discovery": discovery_access,
            "frames_jsonl_open_count": sum(value["opened_frames_jsonl_count"] for value in scan_results),
            "frames_jsonl_byte_count": sum(int(value["binding_row"][4]) for value in scan_results),
            "selected_metadata_row_parse_count": sum(value["parsed_selected_metadata_row_count"] for value in scan_results),
            "rgb_open_count": 0,
            "raw_open_count": 0,
            "label_open_count": 0,
            "checkpoint_open_count": 0,
            "protected_open_count": 0,
            "network_access_count": 0,
            "gpu_use_count": 0,
        },
        "scene_and_discrete_action_diversity": summarize_index(rows_by_role),
        "local_preaction_physical_history_action_support": {
            "feature_definition": {
                "history": "two body-frame endpoint deltas (forward,lateral,yaw)",
                "current": (
                    "base z/roll/pitch, six-axis body twist, twelve joint positions, "
                    "and twelve joint velocities"
                ),
                "feature_count": len(train_physical[0]["state_vector"]),
                "candidate_action": "requested action a2 beginning after current rgb[2]",
                "neighbor_exclusion": "same-scene references excluded for train leave-one-out",
            },
            "train_leave_one_scene_out": train_support,
            "validation_queries_against_train": validation_support,
        },
        "physical_coverage_and_motion": {
            "train": summarize_physical(train_physical),
            "val": summarize_physical(val_physical),
        },
        "prior_evidence_reused": {
            "path": "docs/lewm_go2_main_pool_action_frame_alignment_audit_2026-07-28.md",
            "relevant_facts": [
                "all selected action IDs matched source primitive context",
                "corrected boundary validation action separability was 0.452270 balanced accuracy",
                "frames.jsonl lacks executed/clipped arrays and clipping flags",
            ],
        },
        "limitations": [
            "Development train/validation roles are not blind held-out evidence.",
            "Neighborhoods use physical/proprioceptive histories but not RGB appearance.",
            "Nearest-neighbor entropy is scale- and feature-definition-dependent.",
            "Observed twist includes locomotion dynamics, contacts, inertia, and tracking error; it is not an executed command receipt.",
            "The metadata has no direct collision/contact flag and joint effort is usually absent, so dynamic-event density is limited to motion and attitude proxies.",
            "This audit does not train, evaluate, or authorize a world model or planner.",
        ],
        "runtime": {
            "workers": workers,
            "knn_k": knn_k,
            "wall_elapsed_seconds": elapsed,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--knn-k", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.knn_k < 2:
        raise PoolSupportAuditError("workers must be positive and knn-k must be at least two")
    result = run_audit(args.repo_root, workers=args.workers, knn_k=args.knn_k)
    _write_exclusive(args.output, result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
