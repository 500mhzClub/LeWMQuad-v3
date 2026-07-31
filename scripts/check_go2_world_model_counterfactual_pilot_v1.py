#!/usr/bin/env python3
"""Receipt-only checker for the development counterfactual world-model pilot.

This checker is intentionally incapable of loading an image, checkpoint, or
simulator payload.  Its read boundary is the caller-bound top-level manifest
and the JSON/JSONL receipt files explicitly bound by that manifest.  RGB leaf
paths and hashes are treated as inert receipt values.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_world_model_counterfactual_pilot_v1 as producer_contract,
)
from lewm.datasets import (  # noqa: E402
    go2_world_model_counterfactual_pilot_v1 as shared_consumer,
)


MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_manifest_v1"
COLLECTION_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_physics_result_v1"
PLAN_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_plan_v1"
STATE_RECEIPT_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_state_receipt_v1"
RGB_MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_rgb_manifest_v1"
GROUP_SCHEMA = "lewm_go2_world_model_counterfactual_group_v1"
REPORT_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_receipt_check_v1"

PRIMITIVE_NAMES = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
CANONICAL_ACTION_BLOCKS = (
    ((0.20, 0.0, 0.45),) * 5,
    ((0.20, 0.0, -0.45),) * 5,
    ((-0.20, 0.0, 0.0),) * 5,
    ((0.30, 0.0, 0.0),) * 5,
    ((0.25, 0.0, 0.0),) * 5,
    ((0.20, 0.0, 0.0),) * 5,
    ((0.0, 0.0, 0.0),) * 5,
    ((0.0, 0.0, 0.45),) * 5,
    ((0.0, 0.0, -0.45),) * 5,
)
ROLE_NAMES = ("train", "eval")
FAMILY_NAMES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

COMMAND_TICKS_PER_BLOCK = 5
COMMAND_WIDTH = 3
CANDIDATE_BRANCHES_PER_STATE = 9
TOTAL_BRANCHES_PER_STATE = 10
CONTEXT_FRAMES_PER_STATE = 3
HISTORY_BLOCKS_PER_STATE = 2
TRAJECTORY_POLICY_STEP_SAMPLES = 25
POLICY_STEP_DURATION_NS = 20_000_000
PREBRANCH_SIM_TIME_NS = 1_000_000_000
BRANCH_ENDPOINT_SIM_TIME_NS = 1_500_000_000
TRAJECTORY_VECTOR_LENGTHS = {
    "base_pos_world": 3,
    "base_quat_wxyz": 4,
    "base_lin_vel_world": 3,
    "base_ang_vel_world": 3,
    "leg_joint_pos": 12,
    "leg_joint_vel": 12,
}
ENDPOINT_VECTOR_LENGTHS = {
    "qpos": 19,
    "dofs_velocity": 18,
    **TRAJECTORY_VECTOR_LENGTHS,
    "runner_last_executed": 3,
    "policy_last_actions": 12,
}
MAX_SCENES = 32
MAX_STATES = 384
MAX_CANDIDATE_BRANCHES = 3_456
MAX_SENTINEL_BRANCHES = 384
MAX_TOTAL_BRANCHES = 3_456
MAX_RECEIPT_BYTES = 64 * 1024 * 1024

_SHA256_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_RECEIPT_COMPONENTS = frozenset(
    {
        "heldout",
        "held_out",
        "held-out",
        "checkpoint",
        "checkpoints",
        "rgb",
        "depth",
    }
)


class PilotReceiptError(RuntimeError):
    """Raised when bound metadata cannot establish the pilot contract."""


def _fail(message: str) -> None:
    raise PilotReceiptError(message)


def _branch_count_for_role(role: str) -> int:
    return 10 if role == "calibration" else 9


def _has_sentinel(state: Mapping[str, Any]) -> bool:
    return str(state["role"]) == "calibration"


def _plain_dict(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{name} must be a plain JSON object")
    return value


def _plain_list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{name} must be a plain JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, name: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        _fail(
            f"{name} keys changed: missing={sorted(expected_set - observed)}, "
            f"unexpected={sorted(observed - expected_set)}"
        )


def _require_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{name} must be a JSON boolean")
    return value


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{name} must be a finite JSON number")
    return float(value)


def _finite_vector(value: object, *, length: int, name: str) -> list[float]:
    vector = _plain_list(value, name=name)
    if len(vector) != length:
        _fail(f"{name} must contain exactly {length} finite numbers")
    return [
        _require_number(item, name=f"{name}[{index}]")
        for index, item in enumerate(vector)
    ]


def _require_recomputed_number(
    declared: object,
    recomputed: float,
    *,
    name: str,
) -> float:
    observed = _require_number(declared, name=name)
    if not math.isclose(observed, recomputed, rel_tol=1.0e-6, abs_tol=1.0e-7):
        _fail(
            f"{name} disagrees with receipt-only recomputation: "
            f"declared={observed!r} recomputed={recomputed!r}"
        )
    return observed


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{name} must be a non-empty string")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
    )


def _require_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        _fail(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PilotReceiptError("value is not finite canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> None:
    raise PilotReceiptError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotReceiptError(f"{name} is not strict UTF-8 JSON") from error
    return _plain_dict(value, name=name)


def _parse_jsonl(raw: bytes, *, name: str) -> list[dict[str, Any]]:
    if raw and not raw.endswith(b"\n"):
        _fail(f"{name} JSONL must end with a newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            _fail(f"{name} contains a blank JSONL row at line {line_number}")
        rows.append(_parse_json(line, name=f"{name} line {line_number}"))
    return rows


def _fingerprint(item: os.stat_result) -> tuple[int, int, int, int]:
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)


def _read_regular_file(path: Path, *, expected_bytes: int, name: str) -> bytes:
    if expected_bytes < 1 or expected_bytes > MAX_RECEIPT_BYTES:
        _fail(f"{name} byte count is outside the receipt-only bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.stat(follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PilotReceiptError(f"cannot open bound {name}: {path}") from error
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            _fail(f"bound {name} is not a regular file")
        if before_fd.st_size != expected_bytes:
            _fail(f"bound {name} byte count changed")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"bound {name} ended before its declared byte count")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"bound {name} exceeds its declared byte count")
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    if not (
        _fingerprint(before_path)
        == _fingerprint(before_fd)
        == _fingerprint(after_fd)
        == _fingerprint(after_path)
    ):
        _fail(f"bound {name} changed while read")
    return b"".join(chunks)


def _binding(value: object, *, name: str) -> dict[str, Any]:
    binding = _plain_dict(value, name=name)
    _exact_keys(binding, ("path", "file_sha256", "byte_count"), name=name)
    _require_string(binding["path"], name=f"{name}.path")
    _require_sha256(binding["file_sha256"], name=f"{name}.file_sha256")
    _require_int(binding["byte_count"], name=f"{name}.byte_count", minimum=1)
    return binding


def _validate_inert_binding(value: object, *, name: str) -> dict[str, Any]:
    """Validate an identity binding without opening the bound payload."""

    return _binding(value, name=name)


def _receipt_path(receipt_root: Path, relative: str, *, name: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        _fail(f"{name} must be a normalized relative receipt path")
    lowered = tuple(part.lower() for part in pure.parts)
    if any(
        part == "sealed"
        or part.startswith("sealed_")
        or part == "sealed_test.json"
        or part in _FORBIDDEN_RECEIPT_COMPONENTS
        for part in lowered
    ):
        _fail(f"{name} names a forbidden receipt path component")
    if pure.suffix not in (".json", ".jsonl"):
        _fail(f"{name} is not a JSON/JSONL receipt")
    candidate = receipt_root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PilotReceiptError(f"{name} does not resolve to a bound receipt") from error
    try:
        resolved.relative_to(receipt_root)
    except ValueError as error:
        raise PilotReceiptError(f"{name} escapes the receipt root") from error
    return resolved


def _load_bound_receipt(
    receipt_root: Path,
    value: object,
    *,
    name: str,
) -> tuple[dict[str, Any] | list[dict[str, Any]], bytes]:
    binding = _binding(value, name=f"{name} binding")
    path = _receipt_path(receipt_root, binding["path"], name=name)
    raw = _read_regular_file(path, expected_bytes=binding["byte_count"], name=name)
    if _sha256_bytes(raw) != binding["file_sha256"]:
        _fail(f"{name} file SHA-256 changed")
    if path.suffix == ".json":
        return _parse_json(raw, name=name), raw
    return _parse_jsonl(raw, name=name), raw


def _command_tape(value: object, *, name: str) -> list[list[float]]:
    rows = _plain_list(value, name=name)
    if len(rows) != COMMAND_TICKS_PER_BLOCK:
        _fail(f"{name} must contain exactly five command ticks")
    result: list[list[float]] = []
    for row_index, row_value in enumerate(rows):
        row = _plain_list(row_value, name=f"{name}[{row_index}]")
        if len(row) != COMMAND_WIDTH:
            _fail(f"{name}[{row_index}] must contain exactly three command values")
        result.append(
            [
                _require_number(item, name=f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    return result


def _validate_tape_digest(tape: object, digest: object, *, name: str) -> list[list[float]]:
    normalized = _command_tape(tape, name=name)
    expected = _require_sha256(digest, name=f"{name}_sha256")
    if _sha256_bytes(_canonical_json_bytes(tape)) != expected:
        _fail(f"{name} canonical SHA-256 changed")
    return normalized


def _validate_frame_identity(
    value: object,
    *,
    state_id: str,
    name: str,
    context_index: int | None = None,
    lane_index: int | None = None,
    lane_offset: int | None = None,
    action_id: int | None = None,
) -> str:
    del lane_index, action_id
    identity = _require_string(value, name=name)
    if context_index is not None:
        expected = f"{state_id}:context:{context_index}"
    else:
        if lane_offset is None:
            _fail(f"{name} lacks a branch identity index")
        kind = "candidate" if lane_offset < CANDIDATE_BRANCHES_PER_STATE else "sentinel"
        expected = f"{state_id}:{kind}:{lane_offset}"
    if identity != expected:
        _fail(f"{name} changed from the canonical frame identity")
    return identity


def _count_contract(states: Sequence[Mapping[str, Any]], source_count: int) -> dict[str, Any]:
    del source_count
    roles = Counter(str(state["role"]) for state in states)
    scenes = {
        (str(state["role"]), str(state["family"]), str(state["scene_id"]))
        for state in states
    }
    sentinel_count = int(roles.get("calibration", 0))
    return {
        "scenes": len(scenes),
        "states": len(states),
        "roles": dict(sorted(roles.items())),
        "actions": len(PRIMITIVE_NAMES),
        "candidate_branches": CANDIDATE_BRANCHES_PER_STATE * len(states),
        "sentinel_branches": sentinel_count,
        "total_branches": CANDIDATE_BRANCHES_PER_STATE * len(states) + sentinel_count,
        "context_frames": CONTEXT_FRAMES_PER_STATE * len(states),
        "target_frames": CANDIDATE_BRANCHES_PER_STATE * len(states) + sentinel_count,
    }


def _validate_caps(value: object, counts: Mapping[str, Any]) -> dict[str, Any]:
    caps = _plain_dict(value, name="caps")
    if {"scenes", "states", "candidate_branches", "sentinel_branches", "total_branches"}.issubset(caps):
        for key in ("scenes", "states", "candidate_branches", "sentinel_branches", "total_branches"):
            if caps[key] != counts[key]:
                _fail(f"caps.{key} changed from the exact plan count")
        for key, item in caps.items():
            if key in {"scenes", "states", "candidate_branches", "sentinel_branches", "total_branches"}:
                continue
            if type(item) not in (int, float) or not math.isfinite(float(item)) or float(item) <= 0.0:
                _fail(f"caps.{key} must be a positive finite ceiling")
        return {key: int(value) if type(value) is int else value for key, value in caps.items()}
    expected_keys = (
        "max_scenes",
        "max_states",
        "max_candidate_branches",
        "max_sentinel_branches",
        "max_total_branches",
        "max_context_frames",
        "max_target_frames",
    )
    _exact_keys(caps, expected_keys, name="caps")
    normalized = {
        key: _require_int(
            caps[key],
            name=f"caps.{key}",
            minimum=0 if key == "max_sentinel_branches" else 1,
        )
        for key in expected_keys
    }
    hard_limits = {
        "max_scenes": MAX_SCENES,
        "max_states": MAX_STATES,
        "max_candidate_branches": MAX_CANDIDATE_BRANCHES,
        "max_sentinel_branches": MAX_SENTINEL_BRANCHES,
        "max_total_branches": MAX_TOTAL_BRANCHES,
        "max_context_frames": CONTEXT_FRAMES_PER_STATE * MAX_STATES,
        "max_target_frames": MAX_TOTAL_BRANCHES,
    }
    for key, hard_limit in hard_limits.items():
        if normalized[key] > hard_limit:
            _fail(f"caps.{key} exceeds the source-only hard ceiling")
    relationships = {
        "max_scenes": int(counts["scenes"]),
        "max_states": int(counts["states"]),
        "max_candidate_branches": int(counts["candidate_branches"]),
        "max_sentinel_branches": int(counts["sentinel_branches"]),
        "max_total_branches": int(counts["total_branches"]),
        "max_context_frames": int(counts["context_frames"]),
        "max_target_frames": int(counts["target_frames"]),
    }
    for cap_name, observed in relationships.items():
        if observed > normalized[cap_name]:
            _fail(f"observed {cap_name.removeprefix('max_')} exceeds its manifest cap")
    return normalized


def _validate_no_retry(value: object, *, planned_states: int) -> dict[str, int]:
    accounting = _plain_dict(value, name="no_retry_accounting")
    expected_keys = (
        "planned_states",
        "attempted_states",
        "completed_states",
        "failed_states",
        "refilled_states",
        "retried_states",
    )
    _exact_keys(accounting, expected_keys, name="no_retry_accounting")
    normalized = {
        key: _require_int(accounting[key], name=f"no_retry_accounting.{key}")
        for key in expected_keys
    }
    if normalized["planned_states"] != planned_states:
        _fail("no-retry planned-state count does not match the plan")
    if normalized["attempted_states"] != planned_states:
        _fail("a COMPLETE pilot must attempt every planned state exactly once")
    if normalized["completed_states"] != planned_states:
        _fail("a COMPLETE pilot must complete every planned state")
    if normalized["failed_states"] != 0:
        _fail("a COMPLETE pilot cannot contain failed states")
    if normalized["refilled_states"] != 0 or normalized["retried_states"] != 0:
        _fail("refill and retry are forbidden")
    return normalized


def _require_keys(value: Mapping[str, Any], required: Iterable[str], *, name: str) -> None:
    missing = set(required) - set(value)
    if missing:
        _fail(f"{name} is missing required keys: {sorted(missing)}")


def _validate_named_bindings(value: object, *, name: str) -> list[dict[str, Any]]:
    if type(value) is dict:
        values = list(value.values())
    else:
        values = _plain_list(value, name=name)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(values):
        entry_name = f"{name}[{index}]"
        binding = _validate_inert_binding(item, name=entry_name)
        if binding["path"] in seen:
            _fail(f"{name} contains a duplicate path binding")
        seen.add(binding["path"])
        result.append(binding)
    if not result:
        _fail(f"{name} cannot be empty")
    return result


def _validate_action_catalog(value: object) -> tuple[dict[str, Any], ...]:
    entries = _plain_list(value, name="action_catalog")
    if len(entries) != len(PRIMITIVE_NAMES):
        _fail("action_catalog must contain exactly nine actions")
    result: list[dict[str, Any]] = []
    requested_digests: set[str] = set()
    for action_id, item in enumerate(entries):
        entry = _plain_dict(item, name=f"action_catalog[{action_id}]")
        _exact_keys(
            entry,
            ("action_id", "name", "requested_block"),
            name=f"action_catalog[{action_id}]",
        )
        if entry["action_id"] != action_id:
            _fail("action_catalog IDs must be the ordered range 0..8")
        if entry["name"] != PRIMITIVE_NAMES[action_id]:
            _fail("action_catalog names changed from the canonical vocabulary")
        normalized_block = _command_tape(
            entry["requested_block"],
            name=f"action_catalog[{action_id}].requested_block",
        )
        if normalized_block != [list(row) for row in CANONICAL_ACTION_BLOCKS[action_id]]:
            _fail("action_catalog requested block changed from the primitive registry")
        digest = _sha256_bytes(_canonical_json_bytes(entry["requested_block"]))
        if digest in requested_digests:
            _fail("action_catalog contains duplicate requested command blocks")
        requested_digests.add(digest)
        result.append(entry)
    return tuple(result)


def _scene_binding_identity(value: object, *, name: str) -> tuple[str, str, int]:
    binding = _validate_inert_binding(value, name=name)
    return (binding["path"], binding["file_sha256"], binding["byte_count"])


def _validate_plan_state(
    value: object,
    *,
    position: int,
    action_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state = _plain_dict(value, name=f"plan.states[{position}]")
    required = (
        "state_id",
        "role",
        "family",
        "scene_id",
        "scene_manifest_binding",
        "scene_genesis_binding",
        "scene_generation",
        "group_index",
        "state_index_in_scene",
        "target_xy_m",
        "history_action_ids",
        "candidate_action_ids",
    )
    _require_keys(state, required, name=f"plan.states[{position}]")
    _require_string(state["state_id"], name=f"plan.states[{position}].state_id")
    role = _require_string(state["role"], name=f"plan.states[{position}].role")
    if role not in (*ROLE_NAMES, "calibration"):
        _fail(f"plan.states[{position}].role is unsupported")
    family = _require_string(state["family"], name=f"plan.states[{position}].family")
    if family not in FAMILY_NAMES:
        _fail(f"plan.states[{position}].family is outside the eight-family contract")
    _require_string(state["scene_id"], name=f"plan.states[{position}].scene_id")
    if state["scene_generation"] is None:
        _scene_binding_identity(
            state["scene_manifest_binding"],
            name=f"plan.states[{position}].scene_manifest_binding",
        )
        _scene_binding_identity(
            state["scene_genesis_binding"],
            name=f"plan.states[{position}].scene_genesis_binding",
        )
    else:
        if state["scene_manifest_binding"] is not None or state["scene_genesis_binding"] is not None:
            _fail("generated smoke scene cannot also borrow existing scene bindings")
        generation = _plain_dict(
            state["scene_generation"], name=f"plan.states[{position}].scene_generation"
        )
        _validate_inert_binding(
            generation.get("scene_generator_binding"),
            name=f"plan.states[{position}].scene_generator_binding",
        )
    _require_int(state["group_index"], name=f"plan.states[{position}].group_index")
    _require_int(
        state["state_index_in_scene"],
        name=f"plan.states[{position}].state_index_in_scene",
    )
    target_xy = _plain_list(state["target_xy_m"], name=f"plan.states[{position}].target_xy_m")
    if len(target_xy) != 2:
        _fail("plan state target_xy_m must have length two")
    for coordinate, item in enumerate(target_xy):
        _require_number(item, name=f"plan.states[{position}].target_xy_m[{coordinate}]")
    history = _plain_list(
        state["history_action_ids"],
        name=f"plan.states[{position}].history_action_ids",
    )
    if len(history) != HISTORY_BLOCKS_PER_STATE or any(
        type(action_id) is not int or not 0 <= action_id < len(action_catalog)
        for action_id in history
    ):
        _fail("every plan state must bind exactly two valid historical action IDs")
    candidates = _plain_list(
        state["candidate_action_ids"],
        name=f"plan.states[{position}].candidate_action_ids",
    )
    if candidates != list(range(CANDIDATE_BRANCHES_PER_STATE)):
        _fail("every plan state must contain the ordered complete action IDs 0..8")
    sentinel_action = state.get("sentinel_duplicate_action_id")
    if role == "calibration":
        if type(sentinel_action) is not int or sentinel_action not in candidates:
            _fail("calibration state must select one sentinel duplicate action")
        expected_sentinel = 6 if state["state_index_in_scene"] == 0 else 4
        if sentinel_action != expected_sentinel:
            _fail("calibration sentinel allocation must be HOLD first, forward_medium second")
    elif sentinel_action is not None:
        _fail("train/eval states cannot add an unregistered repeat branch")
    return state


def _validate_plan(value: object, *, attempt_id: str, output_root: str) -> dict[str, Any]:
    plan = _plain_dict(value, name="pilot plan")
    _require_keys(
        plan,
        (
            "schema",
            "purpose",
            "citable_as_scientific_evidence",
            "authorizes_retry_or_resume",
            "allows_refill",
            "allows_overwrite",
            "branch_mechanism",
            "states_per_scene",
            "history_blocks",
            "attempt_id",
            "output_root",
            "runtime_bindings",
            "render_contract",
            "execution_contract",
            "action_catalog",
            "expected_counts",
            "states",
        ),
        name="pilot plan",
    )
    if plan["schema"] != PLAN_SCHEMA:
        _fail("pilot plan schema changed")
    if _require_bool(
        plan["citable_as_scientific_evidence"],
        name="pilot plan citable_as_scientific_evidence",
    ):
        _fail("pilot plan is not citable scientific evidence")
    if plan["attempt_id"] != attempt_id or plan["output_root"] != output_root:
        _fail("pilot plan attempt/output root does not match the joined manifest")
    if _require_bool(
        plan["authorizes_retry_or_resume"], name="pilot plan authorizes_retry_or_resume"
    ):
        _fail("pilot plan must not authorize retry or resume")
    if _require_bool(plan["allows_refill"], name="pilot plan allows_refill"):
        _fail("pilot plan must not authorize refill")
    if _require_bool(plan["allows_overwrite"], name="pilot plan allows_overwrite"):
        _fail("pilot plan must not authorize overwrite")
    if plan["branch_mechanism"] != "parallel_lockstep_envs_no_restore":
        _fail("pilot plan branch mechanism changed")
    if plan["history_blocks"] != HISTORY_BLOCKS_PER_STATE:
        _fail("pilot plan must use exactly two historical action blocks")
    _require_int(plan["states_per_scene"], name="pilot plan states_per_scene", minimum=1)
    runtime_bindings = _plain_dict(plan["runtime_bindings"], name="pilot plan runtime_bindings")
    required_runtime = {
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
        "python_interpreter",
    }
    if not required_runtime.issubset(runtime_bindings):
        _fail("pilot plan runtime binding set changed")
    for binding_name, binding in runtime_bindings.items():
        _validate_inert_binding(binding, name=f"runtime_bindings.{binding_name}")
    render_contract = _plain_dict(plan["render_contract"], name="pilot plan render_contract")
    if render_contract != {
        "native_resolution": [640, 480],
        "stored_resolution": [224, 224],
        "rgb_format": "png",
        "depth_validation": "transient_not_persisted",
        "replay_env_mode": "single_non_batched_sequential",
        "replay_pose_source": "captured_physical_base_pose",
        "physical_scene_rendering": False,
    }:
        _fail("pilot plan render contract changed")
    execution = _plain_dict(plan["execution_contract"], name="pilot plan execution_contract")
    _require_keys(
        execution,
        (
            "backend",
            "policy_device",
            "seed",
            "fall_z_threshold_m",
            "tip_threshold_rad",
            "policy_steps_per_command_tick",
        ),
        name="pilot plan execution_contract",
    )
    if execution["policy_steps_per_command_tick"] != 5:
        _fail("pilot plan must use five policy steps per command tick")
    action_catalog = _validate_action_catalog(plan["action_catalog"])
    raw_states = _plain_list(plan["states"], name="pilot plan states")
    states = [
        _validate_plan_state(item, position=index, action_catalog=action_catalog)
        for index, item in enumerate(raw_states)
    ]
    if not states:
        _fail("pilot plan cannot contain zero states")
    identities = [str(state["state_id"]) for state in states]
    if len(set(identities)) != len(identities):
        _fail("pilot plan contains duplicate state IDs")
    group_indices = [int(state["group_index"]) for state in states]
    if group_indices != list(range(len(states))):
        _fail("pilot plan group indices must be the ordered range 0..N-1")
    cumulative_lane = 0
    for state in states:
        state["_lane_start"] = cumulative_lane
        state["_lane_count"] = _branch_count_for_role(str(state["role"]))
        cumulative_lane += int(state["_lane_count"])
    scene_bindings: dict[tuple[str, str], str] = {}
    scene_roles: dict[str, set[str]] = defaultdict(set)
    scene_state_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, state in enumerate(states):
        scene_key = (str(state["role"]), str(state["scene_id"]))
        identity = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "scene_manifest_binding": state["scene_manifest_binding"],
                    "scene_genesis_binding": state["scene_genesis_binding"],
                    "scene_generation": state["scene_generation"],
                }
            )
        )
        previous = scene_bindings.setdefault(scene_key, identity)
        if previous != identity:
            _fail("one role/scene has conflicting scene-manifest bindings")
        scene_roles[str(state["scene_id"])].add(str(state["role"]))
        scene_state_indices[scene_key].append(int(state["state_index_in_scene"]))
    if any(len(roles) != 1 for roles in scene_roles.values()):
        _fail("scene IDs must be disjoint across all roles")
    for key, indices in scene_state_indices.items():
        if sorted(indices) != list(range(len(indices))):
            _fail(f"state indices are not contiguous within role/scene {key!r}")
    return {
        "document": plan,
        "action_catalog": action_catalog,
        "states": tuple(states),
        "state_by_id": {str(state["state_id"]): state for state in states},
        "caps": plan.get("caps"),
        "expected_counts": plan["expected_counts"],
    }


def _validate_canonical_producer_plan(
    value: object,
    *,
    attempt_id: str,
    output_root: str,
) -> dict[str, Any]:
    plan = _plain_dict(value, name="pilot plan")
    try:
        normalized = producer_contract.validate_plan(plan)
    except producer_contract.PilotContractError as error:
        raise PilotReceiptError(f"producer plan contract failed: {error}") from error
    if normalized["attempt_id"] != attempt_id or normalized["output_root"] != output_root:
        _fail("producer plan attempt/output root changed")
    states = [dict(state) for state in normalized["states"]]
    cumulative_lane = 0
    for state in states:
        state["_lane_start"] = cumulative_lane
        state["_lane_count"] = _branch_count_for_role(str(state["role"]))
        cumulative_lane += int(state["_lane_count"])
    return {
        "document": normalized,
        "action_catalog": tuple(normalized["action_catalog"]),
        "states": tuple(states),
        "state_by_id": {str(state["state_id"]): state for state in states},
        "caps": None,
        "expected_counts": normalized["expected_counts"],
    }


def _validate_sync_audit(value: object, *, state: Mapping[str, Any]) -> dict[str, Any]:
    audit = _plain_dict(value, name=f"state {state['state_id']} synchronization_audit")
    _exact_keys(
        audit,
        (
            "state_id",
            "group_index",
            "lane_start",
            "lane_count",
            "exact_equality_required",
            "passed",
            "prebranch_state_sha256",
            "lane_state_sha256s",
            "components",
        ),
        name=f"state {state['state_id']} synchronization_audit",
    )
    lane_count = _branch_count_for_role(str(state["role"]))
    lane_start = int(state["_lane_start"])
    if (
        audit["state_id"] != state["state_id"]
        or audit["group_index"] != state["group_index"]
        or audit["lane_start"] != lane_start
        or audit["lane_count"] != lane_count
        or audit["exact_equality_required"] is not True
        or audit["passed"] is not True
    ):
        _fail("pre-branch synchronization audit did not pass its exact contract")
    digest = _require_sha256(
        audit["prebranch_state_sha256"],
        name=f"state {state['state_id']} prebranch_state_sha256",
    )
    lane_hashes = _plain_list(
        audit["lane_state_sha256s"],
        name=f"state {state['state_id']} lane_state_sha256s",
    )
    if len(lane_hashes) != lane_count or any(item != digest for item in lane_hashes):
        _fail("pre-branch lane state hashes are not exactly equal across all lanes")
    components = _plain_dict(
        audit["components"], name=f"state {state['state_id']} sync components"
    )
    if set(components) != set(producer_contract.SYNC_COMPONENTS):
        _fail("synchronization audit component set changed")
    for component_name, raw_component in components.items():
        component = _plain_dict(raw_component, name=f"sync component {component_name}")
        if (
            component.get("exact_equal") is not True
            or _require_number(
                component.get("max_abs_difference"),
                name=f"sync component {component_name}.max_abs_difference",
            )
            != 0.0
            or type(component.get("shape_per_lane")) is not list
        ):
            _fail("synchronization component exact-equality receipt failed")
    return audit


def _validate_sentinel_audit(
    value: object,
    *,
    state: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = _plain_dict(value, name=f"state {state['state_id']} sentinel")
    _exact_keys(
        audit,
        (
            "state_id",
            "group_index",
            "action_id",
            "candidate_lane",
            "sentinel_lane",
            "policy_step_count",
            "exact_equality_required",
            "physics_equal",
            "candidate_trajectory_sha256",
            "sentinel_trajectory_sha256",
            "components",
        ),
        name=f"state {state['state_id']} sentinel",
    )
    action_id = int(state["sentinel_duplicate_action_id"])
    lane_start = int(state["_lane_start"])
    if (
        audit["state_id"] != state["state_id"]
        or audit["group_index"] != state["group_index"]
        or audit["action_id"] != action_id
        or audit["candidate_lane"] != lane_start + action_id
        or audit["sentinel_lane"] != lane_start + CANDIDATE_BRANCHES_PER_STATE
        or audit["policy_step_count"] != 25
        or audit["exact_equality_required"] is not True
        or audit["physics_equal"] is not True
    ):
        _fail("sentinel physical-repeat audit did not pass its exact contract")
    candidate_digest = _require_sha256(
        audit["candidate_trajectory_sha256"], name="candidate trajectory digest"
    )
    sentinel_digest = _require_sha256(
        audit["sentinel_trajectory_sha256"], name="sentinel trajectory digest"
    )
    if (
        candidate_digest != sentinel_digest
        or branches[action_id]["trajectory_policy_step_samples"]
        != branches[-1]["trajectory_policy_step_samples"]
    ):
        _fail("sentinel and candidate trajectory receipts are not exactly equal")
    components = _plain_dict(
        audit["components"], name=f"state {state['state_id']} sentinel components"
    )
    if set(components) != set(producer_contract.SYNC_COMPONENTS):
        _fail("sentinel physical component audit set changed")
    for component_name, raw_component in components.items():
        component = _plain_dict(raw_component, name=f"sentinel component {component_name}")
        _exact_keys(
            component,
            ("exact_equal", "max_abs_difference"),
            name=f"sentinel component {component_name}",
        )
        if (
            component.get("exact_equal") is not True
            or _require_number(
                component.get("max_abs_difference"),
                name=f"sentinel component {component_name}.max_abs_difference",
            )
            != 0.0
        ):
            _fail("every sentinel physical component equality must pass")
    return audit


def _validate_render_sentinel_audit(
    value: object,
    *,
    state: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del branches
    audit = _plain_dict(value, name=f"state {state['state_id']} render sentinel")
    _exact_keys(
        audit,
        (
            "state_id",
            "group_index",
            "action_id",
            "candidate_lane",
            "sentinel_lane",
            "exact_equality_required",
            "stored_rgb_equal",
            "candidate_stored_rgb_sha256",
            "sentinel_stored_rgb_sha256",
            "passed",
        ),
        name=f"state {state['state_id']} render sentinel",
    )
    action_id = int(state["sentinel_duplicate_action_id"])
    lane_start = int(state["_lane_start"])
    candidate_sha = _require_sha256(
        audit["candidate_stored_rgb_sha256"], name="render sentinel candidate SHA-256"
    )
    sentinel_sha = _require_sha256(
        audit["sentinel_stored_rgb_sha256"], name="render sentinel SHA-256"
    )
    if (
        audit["state_id"] != state["state_id"]
        or audit["group_index"] != state["group_index"]
        or audit["action_id"] != action_id
        or audit["candidate_lane"] != lane_start + action_id
        or audit["sentinel_lane"] != lane_start + CANDIDATE_BRANCHES_PER_STATE
        or audit["exact_equality_required"] is not True
        or audit["stored_rgb_equal"] is not True
        or audit["passed"] is not True
        or candidate_sha != sentinel_sha
    ):
        _fail("render sentinel exact-repeat audit failed")
    return audit


def _validate_context(
    value: object,
    *,
    state: Mapping[str, Any],
    action_catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    context = _plain_dict(value, name=f"state {state['state_id']} context")
    _exact_keys(
        context,
        (
            "frame_identities",
            "rgb_artifact_ids",
            "history_action_ids",
            "history_executed_blocks",
            "executed_block_sha256s",
            "endpoint_command_ticks",
            "prebranch_state_sha256",
            "prebranch_base_pose_world",
            "context_base_pose_world_sequence",
            "target_relative_body_xy_m",
        ),
        name=f"state {state['state_id']} context",
    )
    frames = _plain_list(context["frame_identities"], name=f"state {state['state_id']} context.frame_identities")
    if len(frames) != CONTEXT_FRAMES_PER_STATE:
        _fail("every state must bind exactly three context frame identities")
    frame_indices: set[str] = set()
    for context_index, frame in enumerate(frames):
        identity = _validate_frame_identity(
            frame,
            state_id=str(state["state_id"]),
            context_index=context_index,
            name=f"state {state['state_id']} context frame {context_index}",
        )
        if identity in frame_indices:
            _fail("context frame identities must be distinct")
        frame_indices.add(identity)
    if context["rgb_artifact_ids"] != frames:
        _fail("context RGB artifact IDs must exactly equal frame identities")
    history_ids = _plain_list(
        context["history_action_ids"],
        name=f"state {state['state_id']} context history_action_ids",
    )
    if history_ids != state["history_action_ids"]:
        _fail("state receipt historical action IDs changed from the plan")
    executed = _plain_list(
        context["history_executed_blocks"],
        name=f"state {state['state_id']} context history_executed_blocks",
    )
    digests = _plain_list(
        context["executed_block_sha256s"],
        name=f"state {state['state_id']} context executed_block_sha256s",
    )
    if not (len(executed) == len(digests) == HISTORY_BLOCKS_PER_STATE):
        _fail("context must contain exactly two executed historical blocks")
    if context["endpoint_command_ticks"] != [0, 5, 10]:
        _fail("context endpoint command ticks changed")
    prebranch_digest = _require_sha256(
        context["prebranch_state_sha256"], name="context prebranch_state_sha256"
    )
    if "prebranch_state_sha256" in state and state["prebranch_state_sha256"] != prebranch_digest:
        _fail("context prebranch state hash changed")
    for index, action_id in enumerate(history_ids):
        _validate_tape_digest(
            executed[index],
            digests[index],
            name=f"state {state['state_id']} executed history block {index}",
        )
    pose = _plain_dict(
        context["prebranch_base_pose_world"],
        name=f"state {state['state_id']} prebranch_base_pose_world",
    )
    _exact_keys(
        pose,
        ("position_xyz_m", "quaternion_wxyz"),
        name=f"state {state['state_id']} prebranch_base_pose_world",
    )
    position = _plain_list(
        pose["position_xyz_m"],
        name=f"state {state['state_id']} prebranch position",
    )
    quaternion = _plain_list(
        pose["quaternion_wxyz"],
        name=f"state {state['state_id']} prebranch quaternion",
    )
    if len(position) != 3 or len(quaternion) != 4:
        _fail("prebranch base pose shape changed")
    position_values = [
        _require_number(value, name=f"prebranch position[{index}]")
        for index, value in enumerate(position)
    ]
    quaternion_values = [
        _require_number(value, name=f"prebranch quaternion[{index}]")
        for index, value in enumerate(quaternion)
    ]
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion_values))
    if quaternion_norm == 0.0 or abs(quaternion_norm - 1.0) > 1.0e-5:
        _fail("prebranch wxyz quaternion is not finite near-unit orientation")
    pose_sequence = _plain_list(
        context["context_base_pose_world_sequence"],
        name=f"state {state['state_id']} context_base_pose_world_sequence",
    )
    if len(pose_sequence) != CONTEXT_FRAMES_PER_STATE:
        _fail("context base-pose sequence must contain exactly three poses")
    normalized_pose_sequence: list[dict[str, Any]] = []
    for index, raw_context_pose in enumerate(pose_sequence):
        context_pose = _plain_dict(
            raw_context_pose,
            name=f"state {state['state_id']} context pose {index}",
        )
        _exact_keys(
            context_pose,
            ("position_xyz_m", "quaternion_wxyz"),
            name=f"state {state['state_id']} context pose {index}",
        )
        context_position = _finite_vector(
            context_pose["position_xyz_m"],
            length=3,
            name=f"state {state['state_id']} context pose {index} position",
        )
        context_quaternion = _finite_vector(
            context_pose["quaternion_wxyz"],
            length=4,
            name=f"state {state['state_id']} context pose {index} quaternion",
        )
        context_quaternion_norm = math.sqrt(
            sum(value * value for value in context_quaternion)
        )
        if (
            context_quaternion_norm == 0.0
            or abs(context_quaternion_norm - 1.0) > 1.0e-5
        ):
            _fail("context base-pose quaternion is not finite near-unit")
        normalized_pose_sequence.append(
            {
                "position_xyz_m": context_position,
                "quaternion_wxyz": context_quaternion,
            }
        )
    if _canonical_json_bytes(normalized_pose_sequence[-1]) != _canonical_json_bytes(
        {
            "position_xyz_m": position_values,
            "quaternion_wxyz": quaternion_values,
        }
    ):
        _fail("final context replay pose is not the prebranch base pose")
    qw, qx, qy, qz = (value / quaternion_norm for value in quaternion_values)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    target_world = state["target_xy_m"]
    delta_x = float(target_world[0]) - position_values[0]
    delta_y = float(target_world[1]) - position_values[1]
    expected_relative = [
        math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
        -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
    ]
    declared_relative = _plain_list(
        context["target_relative_body_xy_m"],
        name=f"state {state['state_id']} target_relative_body_xy_m",
    )
    if len(declared_relative) != 2:
        _fail("body-frame target must contain exactly two coordinates")
    declared_values = [
        _require_number(value, name=f"body-frame target[{index}]")
        for index, value in enumerate(declared_relative)
    ]
    if any(
        not math.isclose(declared, expected, rel_tol=1.0e-6, abs_tol=1.0e-6)
        for declared, expected in zip(declared_values, expected_relative, strict=True)
    ):
        _fail("declared body-frame target disagrees with world target and prebranch pose")
    return {
        "document": context,
        "relative_target_xy_body_m": expected_relative,
    }


def _validate_physics_endpoint(value: object, *, name: str) -> dict[str, Any]:
    endpoint = _plain_dict(value, name=name)
    required = (
        "physical_fell",
        "physical_tipped",
        "physical_path_length_m",
        "physical_target_progress_m",
    )
    _require_keys(endpoint, required, name=name)
    _require_bool(endpoint["physical_fell"], name=f"{name}.physical_fell")
    _require_bool(endpoint["physical_tipped"], name=f"{name}.physical_tipped")
    path_length = _require_number(endpoint["physical_path_length_m"], name=f"{name}.physical_path_length_m")
    _require_number(endpoint["physical_target_progress_m"], name=f"{name}.physical_target_progress_m")
    if path_length < 0.0:
        _fail(f"{name}.physical_path_length_m cannot be negative")
    for optional_bool in ("camera_valid", "physical_recoverable_proxy"):
        if optional_bool in endpoint:
            _require_bool(endpoint[optional_bool], name=f"{name}.{optional_bool}")
    for optional_number in ("physical_clearance_proxy_m", "physical_heading_error_rad"):
        if optional_number in endpoint:
            _require_number(endpoint[optional_number], name=f"{name}.{optional_number}")
    return endpoint


def _quat_tip_rad(quaternion_wxyz: Sequence[float]) -> float:
    qw, qx, qy, qz = (float(value) for value in quaternion_wxyz)
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch_argument = max(
        -1.0,
        min(1.0, 2.0 * (qw * qy - qz * qx)),
    )
    pitch = math.asin(pitch_argument)
    return max(abs(roll), abs(pitch))


def _validate_branch_trajectory_and_endpoint(
    branch: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    context: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    lane_offset: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    name = f"state {state['state_id']} branch {lane_offset}"
    raw_trajectory = _plain_list(
        branch["trajectory_policy_step_samples"],
        name=f"{name}.trajectory_policy_step_samples",
    )
    if len(raw_trajectory) != TRAJECTORY_POLICY_STEP_SAMPLES:
        _fail(
            f"{name} must contain exactly "
            f"{TRAJECTORY_POLICY_STEP_SAMPLES} trajectory policy-step samples"
        )
    trajectory: list[dict[str, Any]] = []
    expected_sample_keys = {
        "policy_step_index",
        "timestamp_ns",
        *TRAJECTORY_VECTOR_LENGTHS,
    }
    for sample_index, raw_sample in enumerate(raw_trajectory):
        sample = _plain_dict(raw_sample, name=f"{name} trajectory[{sample_index}]")
        _exact_keys(
            sample,
            expected_sample_keys,
            name=f"{name} trajectory[{sample_index}]",
        )
        expected_timestamp = (
            PREBRANCH_SIM_TIME_NS
            + (sample_index + 1) * POLICY_STEP_DURATION_NS
        )
        if (
            sample["policy_step_index"] != sample_index
            or type(sample["timestamp_ns"]) is not int
            or sample["timestamp_ns"] != expected_timestamp
        ):
            _fail(
                f"{name} trajectory must use ordered 20 ms samples from "
                "1.02 s through 1.50 s"
            )
        for field, length in TRAJECTORY_VECTOR_LENGTHS.items():
            _finite_vector(
                sample[field],
                length=length,
                name=f"{name} trajectory[{sample_index}].{field}",
            )
        trajectory.append(sample)
    if trajectory[-1]["timestamp_ns"] != BRANCH_ENDPOINT_SIM_TIME_NS:
        _fail(f"{name} trajectory does not end at 1.50 simulated seconds")

    endpoint = _plain_dict(branch["endpoint_state"], name=f"{name}.endpoint_state")
    _exact_keys(endpoint, ENDPOINT_VECTOR_LENGTHS, name=f"{name}.endpoint_state")
    for field, length in ENDPOINT_VECTOR_LENGTHS.items():
        _finite_vector(
            endpoint[field],
            length=length,
            name=f"{name}.endpoint_state.{field}",
        )
    final_sample = trajectory[-1]
    for field in TRAJECTORY_VECTOR_LENGTHS:
        if _canonical_json_bytes(endpoint[field]) != _canonical_json_bytes(
            final_sample[field]
        ):
            _fail(
                f"{name}.endpoint_state.{field} does not exactly equal "
                "the final trajectory sample"
            )

    label_receipt = _validate_physics_endpoint(
        {
            key: branch[key]
            for key in (
                "physical_fell",
                "physical_tipped",
                "physical_path_length_m",
                "physical_target_progress_m",
            )
        },
        name=f"{name} physical labels",
    )
    fall_threshold = _require_number(
        execution_contract.get("fall_z_threshold_m"),
        name="execution_contract.fall_z_threshold_m",
    )
    tip_threshold = _require_number(
        execution_contract.get("tip_threshold_rad"),
        name="execution_contract.tip_threshold_rad",
    )
    positions = [
        _finite_vector(
            context["prebranch_base_pose_world"]["position_xyz_m"],
            length=3,
            name=f"{name} prebranch position",
        )
    ]
    positions.extend(
        _finite_vector(
            sample["base_pos_world"],
            length=3,
            name=f"{name} trajectory position",
        )
        for sample in trajectory
    )
    recomputed_fell = any(position[2] < fall_threshold for position in positions[1:])
    recomputed_tipped = any(
        _quat_tip_rad(
            _finite_vector(
                sample["base_quat_wxyz"],
                length=4,
                name=f"{name} trajectory quaternion",
            )
        )
        > tip_threshold
        for sample in trajectory
    )
    if label_receipt["physical_fell"] is not recomputed_fell:
        _fail(f"{name}.physical_fell disagrees with trajectory heights")
    if label_receipt["physical_tipped"] is not recomputed_tipped:
        _fail(f"{name}.physical_tipped disagrees with trajectory attitudes")
    recomputed_path_length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(positions, positions[1:])
    )
    _require_recomputed_number(
        label_receipt["physical_path_length_m"],
        recomputed_path_length,
        name=f"{name}.physical_path_length_m",
    )
    target = _finite_vector(
        state["target_xy_m"],
        length=2,
        name=f"{name} target_xy_m",
    )
    start = positions[0]
    endpoint_position = _finite_vector(
        endpoint["base_pos_world"],
        length=3,
        name=f"{name} endpoint position",
    )
    recomputed_progress = math.hypot(
        start[0] - target[0], start[1] - target[1]
    ) - math.hypot(
        endpoint_position[0] - target[0], endpoint_position[1] - target[1]
    )
    _require_recomputed_number(
        label_receipt["physical_target_progress_m"],
        recomputed_progress,
        name=f"{name}.physical_target_progress_m",
    )
    return tuple(trajectory), endpoint


def _validate_frame_receipt(
    value: object,
    *,
    expected_identity: str | None = None,
) -> dict[str, Any]:
    receipt = _plain_dict(value, name="frame receipt")
    _exact_keys(
        receipt,
        (
            "artifact_id",
            "frame_identity",
            "path",
            "file_sha256",
            "byte_count",
            "width",
            "height",
            "mode",
            "format",
            "camera_valid",
        ),
        name="frame receipt",
    )
    identity = _require_string(receipt["frame_identity"], name="frame receipt identity")
    if receipt["artifact_id"] != identity or (
        expected_identity is not None and identity != expected_identity
    ):
        _fail("frame receipt artifact/frame identity changed")
    _artifact_path(receipt["path"], name="frame receipt path")
    _require_sha256(receipt["file_sha256"], name="frame receipt SHA-256")
    _require_int(receipt["byte_count"], name="frame receipt byte_count", minimum=1)
    if (
        receipt["width"] != 224
        or receipt["height"] != 224
        or receipt["mode"] != "RGB"
        or receipt["format"] != "PNG"
        or receipt["camera_valid"] is not True
    ):
        _fail("frame receipt shape/format/camera validity changed")
    return receipt


def _validate_state_branches(
    value: object,
    *,
    state: Mapping[str, Any],
    action_catalog: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_branches = _plain_list(value, name=f"state {state['state_id']} branches")
    lane_count = _branch_count_for_role(str(state["role"]))
    if len(raw_branches) != lane_count:
        _fail("state branch count changed from its role contract")
    branches: list[dict[str, Any]] = []
    candidate_tape_digests: list[str] = []
    candidate_trajectory_digests: list[str] = []
    candidate_endpoint_pose_digests: list[str] = []
    candidate_png_digests: list[str] = []
    frame_identities: set[str] = set()
    for lane_offset, raw_branch in enumerate(raw_branches):
        branch = _plain_dict(raw_branch, name=f"state {state['state_id']} branch {lane_offset}")
        _exact_keys(
            branch,
            (
                "lane_index",
                "lane_offset",
                "kind",
                "action_id",
                "action_name",
                "duplicates_candidate_action_id",
                "requested_block",
                "executed_block",
                "executed_block_sha256",
                "clipped",
                "trajectory_policy_step_samples",
                "endpoint_state",
                "physical_fell",
                "physical_tipped",
                "physical_path_length_m",
                "physical_target_progress_m",
                "render_frame_identity",
                "frame_receipt",
            ),
            name=f"state {state['state_id']} branch {lane_offset}",
        )
        lane_start = int(state["_lane_start"])
        expected_lane = lane_start + lane_offset
        if branch["lane_index"] != expected_lane or branch["lane_offset"] != lane_offset:
            _fail("branch global/local lane indices changed")
        if lane_offset < CANDIDATE_BRANCHES_PER_STATE:
            action_id = lane_offset
            if branch["kind"] != "candidate" or branch["duplicates_candidate_action_id"] is not None:
                _fail("candidate branch kind/duplicate marker changed")
        else:
            action_id = int(state["sentinel_duplicate_action_id"])
            if (
                branch["kind"] != "sentinel"
                or branch["duplicates_candidate_action_id"] != action_id
            ):
                _fail("sentinel branch does not identify its duplicated candidate")
        if branch["action_id"] != action_id or branch["action_name"] != PRIMITIVE_NAMES[action_id]:
            _fail("branch action identity changed")
        if branch["requested_block"] != action_catalog[action_id]["requested_block"]:
            _fail("branch requested block does not match the action catalog")
        _command_tape(branch["requested_block"], name=f"state {state['state_id']} branch {lane_offset} requested_block")
        _validate_tape_digest(
            branch["executed_block"],
            branch["executed_block_sha256"],
            name=f"state {state['state_id']} branch {lane_offset} executed_block",
        )
        _require_bool(branch["clipped"], name=f"state {state['state_id']} branch {lane_offset}.clipped")
        trajectory, endpoint = _validate_branch_trajectory_and_endpoint(
            branch,
            state=state,
            context=context,
            execution_contract=execution_contract,
            lane_offset=lane_offset,
        )
        identity = _validate_frame_identity(
            branch["render_frame_identity"],
            state_id=str(state["state_id"]),
            lane_index=expected_lane,
            lane_offset=lane_offset,
            action_id=action_id,
            name=f"state {state['state_id']} branch {lane_offset}.render_frame_identity",
        )
        identity_digest = _sha256_bytes(_canonical_json_bytes(identity))
        if identity_digest in frame_identities:
            _fail("branch render frame identities must be distinct, including the sentinel")
        frame_identities.add(identity_digest)
        frame_receipt = _plain_dict(
            branch["frame_receipt"],
            name=f"state {state['state_id']} branch {lane_offset}.frame_receipt",
        )
        _validate_frame_receipt(frame_receipt, expected_identity=identity)
        if lane_offset < CANDIDATE_BRANCHES_PER_STATE:
            candidate_tape_digests.append(str(branch["executed_block_sha256"]))
            candidate_trajectory_digests.append(
                _sha256_bytes(
                    _canonical_json_bytes(
                        [
                            {
                                "base_pos_world": row["base_pos_world"],
                                "base_quat_wxyz": row["base_quat_wxyz"],
                            }
                            for row in trajectory
                        ]
                    )
                )
            )
            candidate_endpoint_pose_digests.append(
                _sha256_bytes(
                    _canonical_json_bytes(
                        {
                            "base_pos_world": endpoint["base_pos_world"],
                            "base_quat_wxyz": endpoint["base_quat_wxyz"],
                        }
                    )
                )
            )
            candidate_png_digests.append(str(frame_receipt["file_sha256"]))
        branches.append(branch)
    if len(set(candidate_tape_digests)) != CANDIDATE_BRANCHES_PER_STATE:
        _fail("the nine candidate actions collapse to duplicate executed command tapes")
    if (
        len(set(candidate_trajectory_digests)) < 2
        or len(set(candidate_endpoint_pose_digests)) < 2
    ):
        _fail("the nine candidate actions collapse to one physical response")
    if len(set(candidate_png_digests)) < 2:
        _fail("the nine candidate actions collapse to one sequential RGB response")
    if _has_sentinel(state):
        sentinel_action = int(state["sentinel_duplicate_action_id"])
        sentinel = branches[-1]
        candidate = branches[sentinel_action]
        if (
            sentinel["requested_block"] != candidate["requested_block"]
            or sentinel["executed_block"] != candidate["executed_block"]
            or sentinel["executed_block_sha256"] != candidate["executed_block_sha256"]
            or sentinel["endpoint_state"] != candidate["endpoint_state"]
            or sentinel["frame_receipt"]["file_sha256"]
            != candidate["frame_receipt"]["file_sha256"]
        ):
            _fail(
                "sentinel command/RGB receipts or endpoint state do not exactly duplicate the selected candidate"
            )
    return tuple(branches)


def _validate_state_receipt(
    value: object,
    *,
    plan_state: Mapping[str, Any],
    action_catalog: Sequence[Mapping[str, Any]],
    execution_contract: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    receipt = _plain_dict(value, name=f"state receipt {plan_state['state_id']}")
    _exact_keys(
        receipt,
        (
            "schema",
            "attempt_id",
            "status",
            "physics_validated",
            "citable_as_scientific_evidence",
            "authorizes_retry_or_resume",
            "state",
            "synchronization_audit",
            "context",
            "branches",
            "sentinel_audit",
            "render_sentinel_audit",
            "render_receipt_binding",
        ),
        name=f"state receipt {plan_state['state_id']}",
    )
    if receipt["schema"] != STATE_RECEIPT_SCHEMA or receipt["status"] != "PHYSICS_COMPLETE":
        _fail("state receipt schema/status changed")
    if receipt["attempt_id"] != attempt_id:
        _fail("state receipt attempt identity changed")
    if _require_bool(receipt["physics_validated"], name="state receipt physics_validated"):
        _fail("physics-only state receipts cannot self-promote to joined evidence")
    if receipt["citable_as_scientific_evidence"] is not False:
        _fail("physics-only state receipt is not citable")
    if receipt["authorizes_retry_or_resume"] is not False:
        _fail("state receipt cannot authorize retry/resume")
    state_value = _plain_dict(receipt["state"], name=f"state receipt {plan_state['state_id']}.state")
    _exact_keys(
        state_value,
        (
            "state_id",
            "role",
            "family",
            "scene_id",
            "group_index",
            "state_index_in_scene",
            "lane_start",
            "lane_count",
            "scene_manifest_binding",
            "scene_genesis_binding",
            "target_xy_m",
        ),
        name=f"state receipt {plan_state['state_id']}.state",
    )
    for key in ("state_id", "role", "family", "scene_id", "group_index", "state_index_in_scene"):
        if state_value.get(key) != plan_state[key]:
            _fail(f"state receipt {key} changed from the plan")
    for key in ("scene_manifest_binding", "scene_genesis_binding"):
        if plan_state[key] is None:
            _validate_inert_binding(state_value.get(key), name=f"state receipt {key}")
        elif state_value.get(key) != plan_state[key]:
            _fail(f"state receipt {key} changed from the plan")
    if state_value.get("target_xy_m") != plan_state["target_xy_m"]:
        _fail("state receipt target_xy_m changed from the plan")
    if state_value.get("lane_start") != plan_state["_lane_start"]:
        _fail("state receipt lane_start changed from the plan")
    if state_value.get("lane_count") != _branch_count_for_role(str(plan_state["role"])):
        _fail("state receipt lane count changed from its role contract")
    _validate_sync_audit(receipt["synchronization_audit"], state=plan_state)
    checked_context = _validate_context(
        receipt["context"], state=plan_state, action_catalog=action_catalog
    )
    context = checked_context["document"]
    if context["prebranch_state_sha256"] != receipt["synchronization_audit"]["prebranch_state_sha256"]:
        _fail("context and synchronization prebranch hashes differ")
    branches = _validate_state_branches(
        receipt["branches"],
        state=plan_state,
        action_catalog=action_catalog,
        context=context,
        execution_contract=execution_contract,
    )
    if _has_sentinel(plan_state):
        _validate_sentinel_audit(receipt["sentinel_audit"], state=plan_state, branches=branches)
        _validate_render_sentinel_audit(
            receipt["render_sentinel_audit"], state=plan_state, branches=branches
        )
    elif receipt["sentinel_audit"] is not None:
        _fail("train/eval state receipts cannot contain sentinel evidence")
    elif receipt["render_sentinel_audit"] is not None:
        _fail("train/eval state receipts cannot contain render sentinel evidence")
    _validate_inert_binding(receipt["render_receipt_binding"], name="state render_receipt_binding")
    return {
        "document": receipt,
        "state": state_value,
        "context": context,
        "relative_target_xy_body_m": checked_context[
            "relative_target_xy_body_m"
        ],
        "branches": branches,
    }


def _validate_expected_counts(value: object, expected: Mapping[str, Any], *, name: str) -> None:
    observed = _plain_dict(value, name=name)
    if observed != expected:
        _fail(f"{name} does not equal the counts recomputed from plan states")


def _validate_failure_counts(value: object) -> dict[str, int]:
    failures = _plain_dict(value, name="failure_counts")
    if not failures:
        _fail("failure_counts cannot be empty")
    result = {
        key: _require_int(count, name=f"failure_counts.{key}")
        for key, count in failures.items()
    }
    if any(result.values()):
        _fail("a PHYSICS_COMPLETE collection cannot declare failures")
    return result


def _validate_purpose_counts(purpose: object, counts: Mapping[str, Any]) -> str:
    purpose_text = _require_string(purpose, name="plan purpose")
    compact = {
        key: counts[key]
        for key in (
            "scenes",
            "states",
            "candidate_branches",
            "sentinel_branches",
            "total_branches",
            "context_frames",
            "target_frames",
        )
    }
    if purpose_text == "source_integration_smoke":
        expected = {
            "scenes": 1,
            "states": 1,
            "candidate_branches": 9,
            "sentinel_branches": 1,
            "total_branches": 10,
            "context_frames": 3,
            "target_frames": 10,
        }
        if compact != expected or counts["roles"] != {"calibration": 1}:
            _fail("source-integration smoke must be exactly 1 calibration scene/state and 10 branches")
    elif purpose_text == "sizing_calibration_only":
        expected = {
            "scenes": 8,
            "states": 16,
            "candidate_branches": 144,
            "sentinel_branches": 16,
            "total_branches": 160,
            "context_frames": 48,
            "target_frames": 160,
        }
        if compact != expected or counts["roles"] != {"calibration": 16}:
            _fail("full calibration must be exactly 8 scenes, 16 states, and 160 branches")
    elif purpose_text == "bounded_wm_a_pilot":
        if set(counts["roles"]) != set(ROLE_NAMES):
            _fail("pilot purpose requires exact train/eval roles")
        if counts["sentinel_branches"] != 0 or counts["total_branches"] != counts["candidate_branches"]:
            _fail("pilot train/eval groups cannot silently add calibration repeats")
        if counts["states"] not in (128, 256, 384):
            _fail("pilot state count is outside the frozen low/recommended/hard-cap ladder")
    else:
        _fail(
            "plan purpose is not source_integration_smoke, "
            "sizing_calibration_only, or bounded_wm_a_pilot"
        )
    return purpose_text


def _validate_render_plan_rows(
    rows: object,
    *,
    states: Sequence[Mapping[str, Any]],
) -> None:
    if type(rows) is not list:
        _fail("render plan binding must name a JSONL receipt")
    expected: set[str] = set()
    for state in states:
        for branch in state["branches"]:
            expected.add(_sha256_bytes(_canonical_json_bytes(branch["render_frame_identity"])))
    observed: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _plain_dict(raw_row, name=f"render plan row {index}")
        _require_keys(row, ("frame_identity",), name=f"render plan row {index}")
        identity = _plain_dict(row["frame_identity"], name=f"render plan row {index}.frame_identity")
        digest = _sha256_bytes(_canonical_json_bytes(identity))
        if digest in observed:
            _fail("render plan contains a duplicate frame identity")
        observed.add(digest)
    if observed != expected:
        _fail("render plan frame identities do not exactly cover physical branch targets")


def _validate_live_render_receipt(
    value: object,
    *,
    attempt_id: str,
) -> dict[str, Any]:
    receipt = _plain_dict(value, name="live render receipt")
    _exact_keys(
        receipt,
        (
            "schema",
            "attempt_id",
            "status",
            "physics_validated",
            "citable_as_scientific_evidence",
            "scene",
            "render_contract",
            "native_render_calls",
            "stored_rgb_frames",
            "depth_rendered",
            "depth_persisted",
            "visual_mode",
            "visual_domain_fidelity_claimed",
            "frame_receipts",
            "quality_audits",
            "render_sentinel_audits",
        ),
        name="live render receipt",
    )
    if (
        receipt["schema"] != "lewm_go2_world_model_counterfactual_live_render_receipt_v1"
        or receipt["attempt_id"] != attempt_id
        or receipt["status"] != "RENDER_COMPLETE"
        or receipt["physics_validated"] is not False
        or receipt["citable_as_scientific_evidence"] is not False
    ):
        _fail("live render receipt identity/status boundary changed")
    scene = _plain_dict(receipt["scene"], name="live render scene")
    _exact_keys(
        scene,
        ("role", "scene_id", "family", "scene_manifest_binding", "scene_genesis_binding"),
        name="live render scene",
    )
    if (
        scene["role"] not in ("calibration", *ROLE_NAMES)
        or scene["family"] not in FAMILY_NAMES
        or type(scene["scene_id"]) is not str
        or not scene["scene_id"]
    ):
        _fail("live render scene identity changed")
    _validate_inert_binding(scene["scene_manifest_binding"], name="render scene manifest")
    _validate_inert_binding(scene["scene_genesis_binding"], name="render Genesis scene")
    if receipt["render_contract"] != producer_contract.RENDER_CONTRACT:
        _fail("live render contract changed")
    native_render_calls = _require_int(
        receipt["native_render_calls"], name="native_render_calls", minimum=1
    )
    if (
        receipt["depth_rendered"] is not True
        or receipt["depth_persisted"] is not False
        or receipt["visual_mode"] != "solid_materials_box_physics_preserved"
        or receipt["visual_domain_fidelity_claimed"] is not False
    ):
        _fail("live RGB/depth/visual-domain receipt changed")
    frames = _plain_list(receipt["frame_receipts"], name="live frame receipts")
    if (
        receipt["stored_rgb_frames"] != len(frames)
        or native_render_calls != len(frames)
        or not frames
    ):
        _fail("stored RGB frame count changed")
    checked_frames = [_validate_frame_receipt(frame) for frame in frames]
    identities = [str(frame["frame_identity"]) for frame in checked_frames]
    if len(identities) != len(set(identities)):
        _fail("live render receipt repeats frame identities")
    quality = _plain_list(receipt["quality_audits"], name="quality audits")
    if len(quality) != len(frames):
        _fail("quality audit count changed from frame count")
    quality_ids: list[str] = []
    checked_quality: list[dict[str, Any]] = []
    for index, raw_audit in enumerate(quality):
        audit = _plain_dict(raw_audit, name=f"quality audit {index}")
        _exact_keys(
            audit,
            (
                "frame_identity",
                "native_resolution",
                "camera_valid",
                "quality",
                "replay_pose",
            ),
            name=f"quality audit {index}",
        )
        if (
            audit.get("native_resolution") != [640, 480]
            or audit.get("camera_valid") is not True
            or type(audit.get("frame_identity")) is not str
            or _plain_dict(audit.get("quality"), name=f"quality audit {index}.quality").get("valid") is not True
        ):
            _fail("native RGB/depth quality audit failed")
        replay_pose = _plain_dict(
            audit["replay_pose"], name=f"quality audit {index}.replay_pose"
        )
        _exact_keys(
            replay_pose,
            ("source_base_pose_world", "camera_pose_world"),
            name=f"quality audit {index}.replay_pose",
        )
        source_pose = _plain_dict(
            replay_pose["source_base_pose_world"],
            name=f"quality audit {index}.source_base_pose_world",
        )
        _exact_keys(
            source_pose,
            ("position_xyz_m", "quaternion_wxyz"),
            name=f"quality audit {index}.source_base_pose_world",
        )
        _finite_vector(
            source_pose["position_xyz_m"],
            length=3,
            name=f"quality audit {index}.source position",
        )
        source_quaternion = _finite_vector(
            source_pose["quaternion_wxyz"],
            length=4,
            name=f"quality audit {index}.source quaternion",
        )
        quaternion_norm = math.sqrt(sum(value * value for value in source_quaternion))
        if quaternion_norm == 0.0 or abs(quaternion_norm - 1.0) > 1.0e-5:
            _fail("quality replay source quaternion is not finite near-unit")
        camera_pose = _plain_dict(
            replay_pose["camera_pose_world"],
            name=f"quality audit {index}.camera_pose_world",
        )
        _exact_keys(
            camera_pose,
            ("position_xyz_m", "lookat_xyz_m", "up_xyz"),
            name=f"quality audit {index}.camera_pose_world",
        )
        for field in ("position_xyz_m", "lookat_xyz_m", "up_xyz"):
            _finite_vector(
                camera_pose[field],
                length=3,
                name=f"quality audit {index}.camera_pose_world.{field}",
            )
        quality_ids.append(str(audit["frame_identity"]))
        checked_quality.append(audit)
    if quality_ids != identities:
        _fail("quality audits do not preserve frame receipt order")
    sentinel_audits = _plain_list(
        receipt["render_sentinel_audits"], name="render sentinel audits"
    )
    checked_sentinel_audits: list[dict[str, Any]] = []
    sentinel_state_ids: set[str] = set()
    for index, raw_audit in enumerate(sentinel_audits):
        audit = _plain_dict(raw_audit, name=f"render sentinel audit {index}")
        state_id = _require_string(
            audit.get("state_id"), name=f"render sentinel audit {index}.state_id"
        )
        if state_id in sentinel_state_ids:
            _fail("live render receipt repeats a sentinel state identity")
        sentinel_state_ids.add(state_id)
        checked_sentinel_audits.append(audit)
    return {
        "document": receipt,
        "scene": scene,
        "frames": checked_frames,
        "quality_audits": tuple(checked_quality),
        "sentinel_audits": tuple(checked_sentinel_audits),
    }


def _validate_scene_metrics(
    value: object,
    *,
    plan_states: Sequence[Mapping[str, Any]],
    collection_wall_seconds: float,
) -> tuple[dict[str, Any], ...]:
    rows = _plain_list(value, name="scene_metrics")
    planned: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for state in plan_states:
        planned[(str(state["role"]), str(state["scene_id"]))].append(state)
    if len(rows) != len(planned):
        _fail("scene metric count changed")
    expected_fields = (
        "scene_id",
        "family",
        "role",
        "states",
        "envs",
        "physics_build_wall_seconds",
        "physics_simulation_wall_seconds",
        "render_scene_build_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "lockstep_execution_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "scene_pipeline_wall_seconds",
        "scene_total_wall_seconds",
        "native_render_calls",
        "stored_rgb_frames",
        "depth_rendered",
        "depth_persisted",
        "visual_mode",
    )
    timing_fields = (
        "physics_build_wall_seconds",
        "physics_simulation_wall_seconds",
        "render_scene_build_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "lockstep_execution_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "scene_pipeline_wall_seconds",
        "scene_total_wall_seconds",
    )
    seen: set[tuple[str, str]] = set()
    checked: list[dict[str, Any]] = []
    total_scene_wall = 0.0
    for index, raw_row in enumerate(rows):
        row = _plain_dict(raw_row, name=f"scene_metrics[{index}]")
        _exact_keys(row, expected_fields, name=f"scene_metrics[{index}]")
        key = (str(row["role"]), str(row["scene_id"]))
        scene_states = planned.get(key)
        if scene_states is None or key in seen:
            _fail("scene metric role/scene identity is absent or duplicated")
        seen.add(key)
        role = str(scene_states[0]["role"])
        if (
            row["family"] != scene_states[0]["family"]
            or row["states"] != len(scene_states)
            or row["envs"]
            != sum(_branch_count_for_role(str(state["role"])) for state in scene_states)
            or row["native_render_calls"]
            != len(scene_states)
            * (CONTEXT_FRAMES_PER_STATE + _branch_count_for_role(role))
            or row["stored_rgb_frames"]
            != len(scene_states) * (CONTEXT_FRAMES_PER_STATE + _branch_count_for_role(role))
            or row["depth_rendered"] is not True
            or row["depth_persisted"] is not False
            or row["visual_mode"] != "solid_materials_box_physics_preserved"
        ):
            _fail("scene metric counts or render contract changed")
        times = {
            name: _require_number(row[name], name=f"scene_metrics[{index}].{name}")
            for name in timing_fields
        }
        if any(value < 0.0 for value in times.values()):
            _fail("scene stage wall timings cannot be negative")
        if times["physics_simulation_wall_seconds"] != times[
            "lockstep_execution_wall_seconds"
        ]:
            _fail("physics simulation and lockstep wall timings differ")
        if (
            times["native_render_wall_seconds"]
            + times["camera_quality_resize_wall_seconds"]
            + times["lockstep_execution_wall_seconds"]
            + times["render_scene_build_wall_seconds"]
            + times["post_lockstep_receipt_wall_seconds"]
            > times["scene_pipeline_wall_seconds"]
            or times["png_encode_write_hash_wall_seconds"]
            > times["post_lockstep_receipt_wall_seconds"]
            or times["physics_build_wall_seconds"]
            + times["scene_pipeline_wall_seconds"]
            > times["scene_total_wall_seconds"]
        ):
            _fail("scene stage wall timings are internally inconsistent")
        total_scene_wall += times["scene_total_wall_seconds"]
        checked.append(row)
    if seen != set(planned):
        _fail("scene metrics do not exactly cover planned role/scenes")
    if total_scene_wall > collection_wall_seconds:
        _fail("summed scene wall time exceeds collection wall time")
    return tuple(checked)


def _validate_collection(
    value: object,
    *,
    receipt_root: Path,
    attempt_id: str,
    output_root: str,
) -> dict[str, Any]:
    collection = _plain_dict(value, name="collection receipt")
    _exact_keys(
        collection,
        (
            "schema",
            "attempt_id",
            "purpose",
            "status",
            "physics_validated",
            "citable_as_scientific_evidence",
            "authorizes_retry_or_resume",
            "allows_refill",
            "allows_overwrite",
            "branch_mechanism",
            "plan_binding",
            "plan_receipt_binding",
            "authority_binding",
            "review_binding",
            "reservation_binding",
            "execution_contract",
            "runtime_bindings",
            "runtime_versions",
            "source_bindings",
            "caps",
            "expected_counts",
            "observed_counts",
            "scene_materialization",
            "state_receipt_bindings",
            "render_receipt_bindings",
            "scene_metrics",
            "visual_domain_limitation",
            "collection_wall_seconds",
            "failure",
        ),
        name="collection receipt",
    )
    if collection["schema"] != COLLECTION_SCHEMA or collection["status"] != "PHYSICS_COMPLETE":
        _fail("collection schema/status changed")
    if collection["physics_validated"] is not False:
        _fail("physics-only collection cannot self-promote before render join")
    if collection["citable_as_scientific_evidence"] is not False:
        _fail("collection receipt is not independently citable evidence")
    if collection["authorizes_retry_or_resume"] is not False:
        _fail("collection receipt must not authorize retry/resume")
    if collection["allows_refill"] is not False or collection["allows_overwrite"] is not False:
        _fail("collection receipt cannot authorize refill/overwrite")
    if collection["branch_mechanism"] != "parallel_lockstep_envs_no_restore":
        _fail("collection branch mechanism changed")
    if collection["attempt_id"] != attempt_id:
        _fail("collection attempt changed")
    _validate_inert_binding(collection["authority_binding"], name="collection authority_binding")
    _validate_inert_binding(collection["review_binding"], name="collection review_binding")
    _validate_inert_binding(collection["reservation_binding"], name="collection reservation_binding")
    raw_sources = _plain_list(collection["source_bindings"], name="source_bindings")
    source_bindings: list[dict[str, Any]] = []
    source_names: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        source = _plain_dict(raw_source, name=f"source_bindings[{index}]")
        _exact_keys(source, ("name", "binding"), name=f"source_bindings[{index}]")
        source_name = _require_string(source["name"], name=f"source_bindings[{index}].name")
        if source_name in source_names:
            _fail("source binding names must be unique")
        source_names.add(source_name)
        _validate_inert_binding(source["binding"], name=f"source binding {source_name}")
        source_bindings.append(source)
    runtime_bindings = _plain_dict(collection["runtime_bindings"], name="runtime_bindings")
    for name, binding in runtime_bindings.items():
        _validate_inert_binding(binding, name=f"runtime_bindings.{name}")
    runtime_versions = _plain_dict(
        collection["runtime_versions"], name="runtime_versions"
    )
    _exact_keys(
        runtime_versions,
        ("python", "genesis", "torch", "numpy", "pillow"),
        name="runtime_versions",
    )
    for name, version in runtime_versions.items():
        if (
            type(version) is not str
            or not version
            or version != version.strip()
            or not version.isprintable()
        ):
            _fail(f"runtime_versions.{name} must be a trimmed printable version string")
    external_plan_binding = _validate_inert_binding(
        collection["plan_binding"], name="external authorized pilot plan"
    )
    if not Path(str(external_plan_binding["path"])).is_absolute():
        _fail("external authorized pilot plan binding must be absolute")
    local_plan_binding = _binding(
        collection["plan_receipt_binding"],
        name="local authorized pilot plan receipt binding",
    )
    plan_value, plan_raw = _load_bound_receipt(
        receipt_root,
        local_plan_binding,
        name="local authorized pilot plan receipt",
    )
    if (
        local_plan_binding["file_sha256"]
        != external_plan_binding["file_sha256"]
        or local_plan_binding["byte_count"]
        != external_plan_binding["byte_count"]
        or len(plan_raw) != external_plan_binding["byte_count"]
        or _sha256_bytes(plan_raw) != external_plan_binding["file_sha256"]
    ):
        _fail("local plan receipt differs from external authorized plan binding")
    if type(plan_value) is not dict:
        _fail("plan binding must name a JSON receipt")
    plan = _validate_canonical_producer_plan(
        plan_value, attempt_id=attempt_id, output_root=output_root
    )
    if collection["purpose"] != plan["document"]["purpose"]:
        _fail("collection purpose changed from plan")
    if collection["execution_contract"] != plan["document"]["execution_contract"]:
        _fail("collection execution contract changed from plan")
    if collection["runtime_bindings"] != plan["document"]["runtime_bindings"]:
        _fail("collection runtime bindings changed from plan")
    counts = _count_contract(plan["states"], len(source_bindings))
    purpose = _validate_purpose_counts(plan["document"]["purpose"], counts)
    _validate_expected_counts(plan["expected_counts"], counts, name="plan expected_counts")
    _validate_expected_counts(collection["expected_counts"], counts, name="collection expected_counts")
    _validate_expected_counts(collection["observed_counts"], counts, name="collection observed_counts")
    _validate_caps(collection["caps"], counts)
    if collection["failure"] not in (None, {}):
        _fail("PHYSICS_COMPLETE collection cannot carry a failure")
    collection_wall_seconds = _require_number(
        collection["collection_wall_seconds"], name="collection_wall_seconds"
    )
    if collection_wall_seconds < 0.0:
        _fail("collection wall time cannot be negative")
    _validate_scene_metrics(
        collection["scene_metrics"],
        plan_states=plan["states"],
        collection_wall_seconds=collection_wall_seconds,
    )
    receipt_bindings = _plain_list(collection["state_receipt_bindings"], name="state_receipt_bindings")
    if len(receipt_bindings) != len(plan["states"]):
        _fail("state receipt binding count changed from the plan")
    states: list[dict[str, Any]] = []
    receipt_paths: set[str] = set()
    for position, (binding_value, plan_state) in enumerate(zip(receipt_bindings, plan["states"])):
        binding = _binding(binding_value, name=f"state_receipt_bindings[{position}]")
        if binding["path"] in receipt_paths:
            _fail("duplicate state receipt path")
        receipt_paths.add(binding["path"])
        state_value, _ = _load_bound_receipt(
            receipt_root, binding, name=f"state receipt {position}"
        )
        if type(state_value) is not dict:
            _fail("state receipt binding must name JSON")
        states.append(
            _validate_state_receipt(
                state_value,
                plan_state=plan_state,
                action_catalog=plan["action_catalog"],
                execution_contract=plan["document"]["execution_contract"],
                attempt_id=attempt_id,
            )
        )
    render_bindings = _plain_list(
        collection["render_receipt_bindings"], name="render_receipt_bindings"
    )
    if len(render_bindings) != counts["scenes"]:
        _fail("render receipt count changed from scene count")
    render_by_path: dict[str, dict[str, Any]] = {}
    render_scene_keys: set[tuple[str, str]] = set()
    all_frame_receipts: dict[str, dict[str, Any]] = {}
    for index, binding_value in enumerate(render_bindings):
        binding = _binding(binding_value, name=f"render_receipt_bindings[{index}]")
        if binding["path"] in render_by_path:
            _fail("duplicate live render receipt path")
        render_value, _ = _load_bound_receipt(
            receipt_root, binding, name=f"live render receipt {index}"
        )
        if type(render_value) is not dict:
            _fail("live render receipt binding must name JSON")
        checked_render = _validate_live_render_receipt(
            render_value, attempt_id=attempt_id
        )
        scene = checked_render["scene"]
        scene_key = (str(scene["role"]), str(scene["scene_id"]))
        if scene_key in render_scene_keys:
            _fail("duplicate live render receipt role/scene identity")
        render_scene_keys.add(scene_key)
        render_by_path[binding["path"]] = {
            "binding": binding,
            "scene": scene,
            "sentinel_audits": checked_render["sentinel_audits"],
            "quality_by_identity": {
                str(audit["frame_identity"]): audit
                for audit in checked_render["quality_audits"]
            },
        }
        for frame in checked_render["frames"]:
            identity = str(frame["frame_identity"])
            if identity in all_frame_receipts:
                _fail("frame identity repeats across render receipts")
            all_frame_receipts[identity] = frame
    referenced_render_paths: set[str] = set()
    referenced_render_sentinel_audits: set[tuple[str, str]] = set()
    declared_render_sentinel_audits = {
        (path, str(audit["state_id"]))
        for path, render in render_by_path.items()
        for audit in render["sentinel_audits"]
    }
    for state in states:
        render_binding = state["document"]["render_receipt_binding"]
        render_entry = render_by_path.get(render_binding["path"])
        if render_entry is None or render_entry["binding"] != render_binding:
            _fail("state receipt references an undeclared live render receipt")
        referenced_render_paths.add(str(render_binding["path"]))
        render_scene = render_entry["scene"]
        state_scene = state["state"]
        if any(
            render_scene[key] != state_scene[key]
            for key in (
                "role",
                "scene_id",
                "family",
                "scene_manifest_binding",
                "scene_genesis_binding",
            )
        ):
            _fail("state receipt and live render receipt scene identities differ")
        for context_index, identity in enumerate(
            state["context"]["frame_identities"]
        ):
            if identity not in all_frame_receipts:
                _fail("context frame is absent from live render receipts")
            quality_audit = render_entry["quality_by_identity"].get(identity)
            if quality_audit is None:
                _fail("context frame is absent from replay-pose audits")
            source_pose = quality_audit["replay_pose"]["source_base_pose_world"]
            expected_pose = state["context"]["context_base_pose_world_sequence"][
                context_index
            ]
            if _canonical_json_bytes(source_pose) != _canonical_json_bytes(
                expected_pose
            ):
                _fail("context replay pose changed from captured physical context")
        for branch in state["branches"]:
            identity = branch["render_frame_identity"]
            if all_frame_receipts.get(identity) != branch["frame_receipt"]:
                _fail("branch frame receipt changed from live render receipt")
            quality_audit = render_entry["quality_by_identity"].get(identity)
            if quality_audit is None:
                _fail("branch frame is absent from replay-pose audits")
            source_pose = quality_audit["replay_pose"]["source_base_pose_world"]
            expected_pose = {
                "position_xyz_m": branch["endpoint_state"]["base_pos_world"],
                "quaternion_wxyz": branch["endpoint_state"]["base_quat_wxyz"],
            }
            if _canonical_json_bytes(source_pose) != _canonical_json_bytes(
                expected_pose
            ):
                _fail("branch replay pose changed from physical endpoint")
        state_id = str(state_scene["state_id"])
        live_sentinel_matches = [
            audit
            for audit in render_entry["sentinel_audits"]
            if audit.get("state_id") == state_id
        ]
        state_sentinel = state["document"]["render_sentinel_audit"]
        if state_scene["role"] == "calibration":
            if len(live_sentinel_matches) != 1:
                _fail(
                    "calibration state does not have exactly one matching live-render "
                    "sentinel audit"
                )
            if live_sentinel_matches[0] != state_sentinel:
                _fail(
                    "state and live-render sentinel identities or hashes disagree"
                )
            referenced_render_sentinel_audits.add(
                (str(render_binding["path"]), state_id)
            )
        elif live_sentinel_matches or state_sentinel is not None:
            _fail("non-calibration state carries a render sentinel audit")
    if len(all_frame_receipts) != counts["context_frames"] + counts["target_frames"]:
        _fail("live render frame receipts do not exactly cover context and targets")
    if referenced_render_paths != set(render_by_path):
        _fail("one or more declared live render receipts are unreferenced by states")
    if referenced_render_sentinel_audits != declared_render_sentinel_audits:
        _fail("live-render sentinel audits do not exactly cover calibration states")
    if "checker" not in source_names:
        _fail("collection source closure does not bind this checker")
    materialization = collection["scene_materialization"]
    if purpose == "source_integration_smoke":
        materialized = _plain_dict(materialization, name="smoke scene materialization")
        _exact_keys(
            materialized,
            (
                "declaration",
                "scene_manifest_binding",
                "scene_genesis_binding",
                "scene_seed",
                "scene_seed_salt",
                "target_landmark_id",
            ),
            name="smoke scene materialization",
        )
        if len(states) != 1:
            _fail("smoke materialization requires one state")
        if (
            materialized["declaration"]
            != plan["document"]["states"][0]["scene_generation"]
            or
            states[0]["state"]["scene_manifest_binding"]
            != materialized["scene_manifest_binding"]
            or states[0]["state"]["scene_genesis_binding"]
            != materialized["scene_genesis_binding"]
        ):
            _fail("state receipt changed the materialized smoke scene bindings")
        for seed_name in ("scene_seed", "scene_seed_salt"):
            _require_int(materialized[seed_name], name=f"materialization.{seed_name}")
        _require_string(
            materialized["target_landmark_id"],
            name="materialization.target_landmark_id",
        )
    elif materialization is not None:
        _fail("pre-existing calibration/pilot scenes cannot claim smoke materialization")
    _require_string(collection["visual_domain_limitation"], name="visual_domain_limitation")
    return {
        "document": collection,
        "plan": plan,
        "states": tuple(states),
        "frame_receipts": dict(all_frame_receipts),
        "counts": counts,
        "source_bindings": source_bindings,
        "purpose": purpose,
    }


def _artifact_path(value: object, *, name: str) -> str:
    path = _require_string(value, name=name)
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        _fail(f"{name} must be a normalized relative artifact path")
    if pure.suffix.lower() != ".png":
        _fail(f"{name} must identify a PNG receipt leaf")
    return path


def _validate_rgb_manifest(value: object) -> dict[str, Any]:
    manifest = _plain_dict(value, name="RGB artifact manifest")
    _require_keys(manifest, ("schema", "artifacts"), name="RGB artifact manifest")
    if manifest["schema"] != RGB_MANIFEST_SCHEMA:
        _fail("RGB artifact manifest schema changed")
    artifacts = _plain_list(manifest["artifacts"], name="RGB artifact manifest artifacts")
    by_id: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    identities: set[str] = set()
    for index, raw_artifact in enumerate(artifacts):
        artifact = _plain_dict(raw_artifact, name=f"RGB artifact {index}")
        _require_keys(
            artifact,
            (
                "artifact_id",
                "frame_identity",
                "path",
                "file_sha256",
                "byte_count",
                "width",
                "height",
                "mode",
                "format",
                "camera_valid",
            ),
            name=f"RGB artifact {index}",
        )
        artifact_id = _require_string(artifact["artifact_id"], name=f"RGB artifact {index}.artifact_id")
        if artifact_id in by_id:
            _fail("RGB artifact IDs must be unique")
        frame_identity = _require_string(
            artifact["frame_identity"], name=f"RGB artifact {index}.frame_identity"
        )
        identity_digest = _sha256_bytes(_canonical_json_bytes(frame_identity))
        if identity_digest in identities:
            _fail("RGB artifact frame identities must be unique")
        identities.add(identity_digest)
        path = _artifact_path(artifact["path"], name=f"RGB artifact {index}.path")
        if path in paths:
            _fail("RGB artifact paths must be unique")
        paths.add(path)
        _require_sha256(artifact["file_sha256"], name=f"RGB artifact {index}.file_sha256")
        _require_int(artifact["byte_count"], name=f"RGB artifact {index}.byte_count", minimum=1)
        if (
            artifact["width"] != 224
            or artifact["height"] != 224
            or artifact["mode"] != "RGB"
            or artifact["format"] != "PNG"
            or artifact["camera_valid"] is not True
        ):
            _fail("RGB artifact shape/format/camera-valid receipt changed")
        by_id[artifact_id] = artifact
    return {"document": manifest, "by_id": by_id}


def _quantize(value: float, tolerance: float) -> int:
    scaled = abs(value) / tolerance
    return (1 if value >= 0.0 else -1) * math.floor(scaled + 0.5)


def _dense_oracle_ranks(
    branches: Sequence[Mapping[str, Any]],
    *,
    progress_tolerance_m: float,
    path_length_tolerance_m: float,
) -> list[int]:
    keys: list[tuple[int, int, int, int]] = []
    for branch in branches:
        labels = _plain_dict(branch["labels"], name="group branch labels")
        key = (
            int(_require_bool(labels["physical_fell"], name="labels.physical_fell")),
            int(_require_bool(labels["physical_tipped"], name="labels.physical_tipped")),
            -_quantize(
                _require_number(labels["physical_target_progress_m"], name="labels.physical_target_progress_m"),
                progress_tolerance_m,
            ),
            _quantize(
                _require_number(labels["physical_path_length_m"], name="labels.physical_path_length_m"),
                path_length_tolerance_m,
            ),
        )
        keys.append(key)
    ordered = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return [ordered[key] for key in keys]


def _validate_group(
    value: object,
    *,
    state: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    progress_tolerance_m: float,
    path_length_tolerance_m: float,
) -> set[str]:
    row = _plain_dict(value, name=f"joined group {state['document']['state_id']}")
    _require_keys(
        row,
        (
            "schema",
            "role",
            "group_id",
            "family",
            "scene_id",
            "state_index",
            "task",
            "context",
            "common_prefix_receipt",
            "branches",
        ),
        name=f"joined group {state['document']['state_id']}",
    )
    source = state["document"]
    if row["schema"] != GROUP_SCHEMA:
        _fail("joined group schema changed")
    expected_identity = {
        "role": source["role"],
        "group_id": source["state_id"],
        "family": source["family"],
        "scene_id": source["scene_id"],
        "state_index": source["state_index_in_scene"],
    }
    if any(row[key] != expected for key, expected in expected_identity.items()):
        _fail("joined group identity changed from its physics receipt")
    task = _plain_dict(row["task"], name="joined group task")
    if task.get("target_present") is not True:
        _fail("joined pilot groups require an explicit target")
    target_xy = _plain_list(task.get("relative_target_xy_body_m"), name="relative target")
    if len(target_xy) != 2:
        _fail("relative target must be a two-vector")
    for index, item in enumerate(target_xy):
        _require_number(item, name=f"relative target [{index}]")
    if row["common_prefix_receipt"] != source["synchronization_audit"]:
        _fail("joined group common-prefix receipt changed from physics collection")
    context = _plain_dict(row["context"], name="joined group context")
    _require_keys(
        context,
        ("rgb_artifact_ids", "historical_action_ids", "historical_executed_tapes"),
        name="joined group context",
    )
    if context["historical_action_ids"] != state["context"]["historical_action_ids"]:
        _fail("joined context historical action IDs changed")
    if context["historical_executed_tapes"] != state["context"]["executed_blocks"]:
        _fail("joined context executed tapes changed")
    artifact_ids = _plain_list(context["rgb_artifact_ids"], name="context RGB artifact IDs")
    if len(artifact_ids) != CONTEXT_FRAMES_PER_STATE:
        _fail("joined group must reference three context RGB artifacts")
    referenced: set[str] = set()
    for index, artifact_id in enumerate(artifact_ids):
        if type(artifact_id) is not str or artifact_id not in artifacts:
            _fail("joined context references an unknown RGB artifact")
        if artifacts[artifact_id]["frame_identity"] != state["context"]["frames"][index]:
            _fail("context artifact frame identity changed")
        if artifact_id in referenced:
            _fail("joined group repeats a context artifact")
        referenced.add(artifact_id)
    branches = _plain_list(row["branches"], name="joined group branches")
    if len(branches) != CANDIDATE_BRANCHES_PER_STATE:
        _fail("train/eval joined group must contain exactly nine candidates")
    candidate_tapes: set[str] = set()
    for action_id, raw_branch in enumerate(branches):
        branch = _plain_dict(raw_branch, name=f"joined branch {action_id}")
        _require_keys(
            branch,
            (
                "action_id",
                "requested_primitive",
                "executed_command_tape",
                "executed_command_tape_sha256",
                "target_rgb_artifact_id",
                "physics_validated",
                "camera_valid",
                "labels",
                "declared_oracle_dense_rank",
            ),
            name=f"joined branch {action_id}",
        )
        physics = state["branches"][action_id]
        if branch["action_id"] != action_id or branch["requested_primitive"] != PRIMITIVE_NAMES[action_id]:
            _fail("joined branch action identity changed")
        if (
            branch["executed_command_tape"] != physics["executed_block"]
            or branch["executed_command_tape_sha256"] != physics["executed_block_sha256"]
        ):
            _fail("joined branch executed tape changed from physics collection")
        candidate_tapes.add(str(branch["executed_command_tape_sha256"]))
        if branch["physics_validated"] is not True or branch["camera_valid"] is not True:
            _fail("joined branch is not physics- and camera-valid")
        labels = _plain_dict(branch["labels"], name=f"joined branch {action_id} labels")
        endpoint = _plain_dict(physics["endpoint"], name="physics endpoint")
        for key in (
            "physical_fell",
            "physical_tipped",
            "physical_path_length_m",
            "physical_target_progress_m",
        ):
            if labels.get(key) != endpoint.get(key):
                _fail("joined physical labels changed from the physics receipt")
        artifact_id = branch["target_rgb_artifact_id"]
        if type(artifact_id) is not str or artifact_id not in artifacts:
            _fail("joined branch references an unknown target RGB artifact")
        if artifacts[artifact_id]["frame_identity"] != physics["render_frame_identity"]:
            _fail("target artifact frame identity changed")
        if artifact_id in referenced:
            _fail("joined group reuses an RGB artifact")
        referenced.add(artifact_id)
    if len(candidate_tapes) != CANDIDATE_BRANCHES_PER_STATE:
        _fail("joined candidates collapse to duplicate executed tapes")
    ranks = _dense_oracle_ranks(
        branches,
        progress_tolerance_m=progress_tolerance_m,
        path_length_tolerance_m=path_length_tolerance_m,
    )
    if [branch["declared_oracle_dense_rank"] for branch in branches] != ranks:
        _fail("declared oracle dense ranks do not match physical labels/tolerances")
    return referenced


def _validate_action_contract(value: object) -> None:
    contract = _plain_dict(value, name="action_contract")
    _exact_keys(
        contract,
        ("primitive_names", "command_ticks_per_block", "executed_tape_shape"),
        name="action_contract",
    )
    if (
        contract["primitive_names"] != list(PRIMITIVE_NAMES)
        or contract["command_ticks_per_block"] != COMMAND_TICKS_PER_BLOCK
        or contract["executed_tape_shape"] != [COMMAND_TICKS_PER_BLOCK, COMMAND_WIDTH]
    ):
        _fail("joined action contract changed")


def _validate_calibration_contract(value: object) -> dict[str, Any]:
    contract = _plain_dict(value, name="calibration_contract")
    _require_keys(
        contract,
        (
            "excluded_scene_ids",
            "progress_tolerance_m",
            "path_length_tolerance_m",
        ),
        name="calibration_contract",
    )
    excluded = _plain_list(contract["excluded_scene_ids"], name="excluded calibration scenes")
    if len(excluded) != len(set(excluded)) or any(type(item) is not str or not item for item in excluded):
        _fail("excluded calibration scene IDs must be unique non-empty strings")
    progress = _require_number(contract["progress_tolerance_m"], name="progress_tolerance_m")
    path_length = _require_number(contract["path_length_tolerance_m"], name="path_length_tolerance_m")
    if progress <= 0.0 or path_length <= 0.0:
        _fail("ranking tolerances must be positive")
    for optional in ("clearance_tolerance_m", "heading_tolerance_rad"):
        if optional in contract and _require_number(contract[optional], name=optional) <= 0.0:
            _fail(f"{optional} must be positive")
    return {
        "excluded": set(excluded),
        "progress_tolerance_m": progress,
        "path_length_tolerance_m": path_length,
    }


def _validate_joined_manifest(value: object, *, receipt_root: Path) -> dict[str, Any]:
    manifest = _plain_dict(value, name="pilot manifest")
    _require_keys(
        manifest,
        (
            "schema",
            "status",
            "physics_validated",
            "citable_as_scientific_evidence",
            "authorizes_retry_or_resume",
            "evidence_scope",
            "attempt_id",
            "receipt_root",
            "output_root",
            "action_contract",
            "calibration_contract",
            "roles",
            "rgb_artifact_manifest",
            "source_bindings",
            "collection_receipt",
        ),
        name="pilot manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["status"] != "COMPLETE":
        _fail("joined pilot manifest schema/status changed")
    if manifest["physics_validated"] is not True:
        _fail("joined COMPLETE pilot must explicitly be physics-valid")
    if manifest["citable_as_scientific_evidence"] is not False:
        _fail("pilot artifact remains non-citable until a separate evaluator")
    if manifest["authorizes_retry_or_resume"] is not False:
        _fail("joined pilot cannot authorize retry/resume")
    if manifest["evidence_scope"] != "physics_executed":
        _fail("joined pilot evidence scope changed")
    attempt_id = _require_string(manifest["attempt_id"], name="attempt_id")
    expected_root = str(receipt_root)
    if manifest["receipt_root"] != expected_root or manifest["output_root"] != expected_root:
        _fail("joined manifest receipt/output root does not match its actual receipt root")
    _validate_action_contract(manifest["action_contract"])
    calibration = _validate_calibration_contract(manifest["calibration_contract"])
    collection_value, _ = _load_bound_receipt(
        receipt_root, manifest["collection_receipt"], name="collection receipt"
    )
    if type(collection_value) is not dict:
        _fail("collection receipt binding must name JSON")
    collection = _validate_collection(
        collection_value,
        receipt_root=receipt_root,
        attempt_id=attempt_id,
        output_root=expected_root,
    )
    if any(state["document"]["role"] not in ROLE_NAMES for state in collection["states"]):
        _fail("joined pilot collection may contain only train/eval roles")
    top_sources = _validate_named_bindings(manifest["source_bindings"], name="joined source_bindings")
    if top_sources != collection["source_bindings"]:
        _fail("joined source bindings changed from physics collection")
    rgb_value, _ = _load_bound_receipt(
        receipt_root, manifest["rgb_artifact_manifest"], name="RGB artifact manifest"
    )
    if type(rgb_value) is not dict:
        _fail("RGB artifact manifest binding must name JSON")
    rgb = _validate_rgb_manifest(rgb_value)
    roles = _plain_dict(manifest["roles"], name="roles")
    if set(roles) != set(ROLE_NAMES):
        _fail("joined pilot must have exact train and eval roles")
    state_by_role = {
        role: [state for state in collection["states"] if state["document"]["role"] == role]
        for role in ROLE_NAMES
    }
    all_scene_sets: dict[str, set[str]] = {}
    referenced_artifacts: set[str] = set()
    for role in ROLE_NAMES:
        role_contract = _plain_dict(roles[role], name=f"roles.{role}")
        _require_keys(
            role_contract,
            ("index", "group_count", "branch_count", "scene_ids"),
            name=f"roles.{role}",
        )
        rows, _ = _load_bound_receipt(receipt_root, role_contract["index"], name=f"{role} group index")
        if type(rows) is not list:
            _fail(f"{role} index must be JSONL")
        expected_states = state_by_role[role]
        if role_contract["group_count"] != len(expected_states) or len(rows) != len(expected_states):
            _fail(f"{role} group count changed")
        if role_contract["branch_count"] != CANDIDATE_BRANCHES_PER_STATE * len(expected_states):
            _fail(f"{role} branch count changed")
        scene_ids = _plain_list(role_contract["scene_ids"], name=f"{role} scene_ids")
        observed_scene_ids = {str(state["document"]["scene_id"]) for state in expected_states}
        if set(scene_ids) != observed_scene_ids or len(scene_ids) != len(observed_scene_ids):
            _fail(f"{role} scene list changed from collection")
        if observed_scene_ids & calibration["excluded"]:
            _fail(f"{role} includes a calibration-excluded scene")
        all_scene_sets[role] = observed_scene_ids
        families = {str(state["document"]["family"]) for state in expected_states}
        if families != set(FAMILY_NAMES):
            _fail(f"{role} does not cover all eight scene families")
        for row, state in zip(rows, expected_states):
            references = _validate_group(
                row,
                state=state,
                artifacts=rgb["by_id"],
                progress_tolerance_m=calibration["progress_tolerance_m"],
                path_length_tolerance_m=calibration["path_length_tolerance_m"],
            )
            if referenced_artifacts & references:
                _fail("RGB artifacts are reused across joined groups")
            referenced_artifacts.update(references)
    if all_scene_sets["train"] & all_scene_sets["eval"]:
        _fail("train and eval scene roles are not disjoint")
    if referenced_artifacts != set(rgb["by_id"]):
        _fail("RGB artifact manifest has missing or undeclared joined references")
    return {
        "attempt_id": attempt_id,
        "counts": collection["counts"],
        "rgb_artifacts": len(rgb["by_id"]),
        "roles": {role: len(state_by_role[role]) for role in ROLE_NAMES},
    }


def _validate_canonical_joined_manifest(
    value: object,
    *,
    receipt_root: Path,
    expected_file_sha256: str,
    expected_byte_count: int,
) -> dict[str, Any]:
    manifest = _plain_dict(value, name="joined manifest")
    required = {
        "schema",
        "attempt_id",
        "purpose",
        "status",
        "physics_validated",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "evidence_scope",
        "receipt_root",
        "output_root",
        "action_catalog",
        "action_contract",
        "calibration_contract",
        "calibration_receipt",
        "roles",
        "rgb_artifact_manifest",
        "source_bindings",
        "collection_receipt",
    }
    _exact_keys(manifest, required, name="joined manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["status"] != "COMPLETE":
        _fail("joined manifest schema/status changed")
    purpose = manifest["purpose"]
    if purpose not in {
        "source_integration_smoke",
        "sizing_calibration_only",
        "bounded_wm_a_pilot",
    }:
        _fail("joined manifest purpose changed")
    if (
        manifest["physics_validated"] is not True
        or manifest["citable_as_scientific_evidence"] is not False
        or manifest["authorizes_retry_or_resume"] is not False
    ):
        _fail("joined manifest validity/noncitation/no-retry boundary changed")
    if manifest["receipt_root"] != str(receipt_root) or manifest["output_root"] != str(receipt_root):
        _fail("joined receipt/output root changed")
    _validate_action_catalog(manifest["action_catalog"])
    _validate_action_contract(manifest["action_contract"])
    collection_value, _ = _load_bound_receipt(
        receipt_root, manifest["collection_receipt"], name="collection receipt"
    )
    if type(collection_value) is not dict:
        _fail("collection receipt binding must name JSON")
    collection = _validate_collection(
        collection_value,
        receipt_root=receipt_root,
        attempt_id=str(manifest["attempt_id"]),
        output_root=str(receipt_root),
    )
    if collection["purpose"] != purpose:
        _fail("joined purpose changed from collection")
    if purpose == "bounded_wm_a_pilot":
        if manifest["evidence_scope"] != "physics_executed":
            _fail("pilot evidence scope changed")
        try:
            bundle = shared_consumer.load_bound_pilot_v1(
                receipt_root,
                expected_manifest_byte_count=expected_byte_count,
                expected_manifest_sha256=expected_file_sha256,
                allowed_parent=receipt_root.parent,
                synthetic_test_mode=False,
            )
        except shared_consumer.CounterfactualPilotContractError as error:
            raise PilotReceiptError(f"shared consumer rejected joined pilot: {error}") from error
        if bundle.access_audit.get("rgb_leaf_open_count") != 0:
            _fail("shared consumer crossed the RGB receipt boundary")
        return {
            "attempt_id": manifest["attempt_id"],
            "purpose": purpose,
            "counts": collection["counts"],
            "roles": {
                role: len(bundle.groups_by_role[role]) for role in ROLE_NAMES
            },
            "rgb_artifacts": len(bundle.artifacts),
            "can_freeze_pilot_contract": True,
        }
    roles = _plain_dict(manifest["roles"], name="joined calibration roles")
    if set(roles) != {"calibration"}:
        _fail("smoke/calibration join may contain only the calibration role")
    if manifest["evidence_scope"] != "physics_executed_calibration_only":
        _fail("smoke/calibration evidence scope must remain calibration-only")
    return {
        "attempt_id": manifest["attempt_id"],
        "purpose": purpose,
        "counts": collection["counts"],
        "roles": {"calibration": collection["counts"]["roles"]["calibration"]},
        "can_freeze_pilot_contract": purpose == "sizing_calibration_only",
        "consumer_eligible": False,
    }


def load_bound_collection_receipts(
    manifest_path: Path,
    *,
    expected_file_sha256: str,
    expected_byte_count: int,
) -> dict[str, Any]:
    """Load one caller-bound producer collection through the receipt-only boundary.

    The returned mapping contains validated JSON receipt values and inert RGB
    frame receipts.  This function never opens an RGB leaf, runtime input, or
    checkpoint and is the only supported input seam for the final receipt
    joiner.
    """

    _require_sha256(expected_file_sha256, name="caller manifest SHA-256")
    _require_int(expected_byte_count, name="caller manifest byte count", minimum=1)
    path = manifest_path.resolve(strict=True)
    raw = _read_regular_file(
        path,
        expected_bytes=expected_byte_count,
        name="top-level collection manifest",
    )
    if _sha256_bytes(raw) != expected_file_sha256:
        _fail("caller-bound top-level manifest SHA-256 changed")
    manifest = _parse_json(raw, name="top-level collection manifest")
    if manifest.get("schema") != COLLECTION_SCHEMA:
        _fail("receipt join input must be a producer physics collection")
    receipt_root = path.parent
    if not receipt_root.is_dir():
        _fail("collection receipt root is not a directory")
    attempt_id = _require_string(manifest.get("attempt_id"), name="attempt_id")
    return _validate_collection(
        manifest,
        receipt_root=receipt_root,
        attempt_id=attempt_id,
        output_root=str(receipt_root),
    )


def check_manifest(
    manifest_path: Path,
    *,
    expected_file_sha256: str,
    expected_byte_count: int,
) -> dict[str, Any]:
    """Validate a caller-bound manifest without reading any payload leaf."""

    _require_sha256(expected_file_sha256, name="caller manifest SHA-256")
    _require_int(expected_byte_count, name="caller manifest byte count", minimum=1)
    path = manifest_path.resolve(strict=True)
    raw = _read_regular_file(path, expected_bytes=expected_byte_count, name="top-level manifest")
    if _sha256_bytes(raw) != expected_file_sha256:
        _fail("caller-bound top-level manifest SHA-256 changed")
    manifest = _parse_json(raw, name="top-level manifest")
    schema = manifest.get("schema")
    if schema == COLLECTION_SCHEMA:
        receipt_root = path.parent
    else:
        root_value = manifest.get("receipt_root", manifest.get("output_root"))
        root_text = _require_string(root_value, name="top-level receipt/output root")
        root_candidate = Path(root_text)
        if not root_candidate.is_absolute():
            _fail("top-level receipt/output root must be absolute")
        receipt_root = root_candidate.resolve(strict=True)
    if not receipt_root.is_dir() or path.parent != receipt_root:
        _fail("top-level manifest must be a direct child of its bound receipt root")
    if schema == MANIFEST_SCHEMA:
        checked = _validate_canonical_joined_manifest(
            manifest,
            receipt_root=receipt_root,
            expected_file_sha256=expected_file_sha256,
            expected_byte_count=expected_byte_count,
        )
        phase = "joined_pilot"
    elif schema == COLLECTION_SCHEMA:
        attempt_id = _require_string(manifest.get("attempt_id"), name="attempt_id")
        checked = _validate_collection(
            manifest,
            receipt_root=receipt_root,
            attempt_id=attempt_id,
            output_root=str(receipt_root),
        )
        phase = "physics_collection"
        checked = {
            "attempt_id": attempt_id,
            "purpose": checked["purpose"],
            "counts": checked["counts"],
            "roles": sorted(checked["counts"]["roles"]),
            "can_freeze_pilot_contract": False,
        }
    else:
        _fail("top-level manifest schema is not a supported pilot receipt")
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "phase": phase,
        "authority_granted": False,
        "scientific_claim_granted": False,
        "runtime_payloads_opened": False,
        "rgb_bytes_opened": False,
        "checkpoints_opened": False,
        "manifest_binding": {
            "path": str(path),
            "file_sha256": expected_file_sha256,
            "byte_count": expected_byte_count,
        },
        **checked,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(report) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-byte-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = check_manifest(
            args.manifest,
            expected_file_sha256=args.expected_file_sha256,
            expected_byte_count=args.expected_byte_count,
        )
        if args.output is None:
            sys.stdout.buffer.write(_canonical_json_bytes(report) + b"\n")
        else:
            _write_report(args.output, report)
    except (OSError, PilotReceiptError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
