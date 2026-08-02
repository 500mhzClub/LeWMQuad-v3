"""Fail-closed contracts for the Go2 synchronized counterfactual pilot V1.

This module is deliberately independent of Genesis, Torch, RGB decoding, and
generated data.  It defines the immutable plan, lane layout, provenance
bindings, and synchronization/sentinel audits used by the separately
authorized physical collector.

V1 never restores a simulator snapshot.  Ten cloned Genesis environments are
advanced through an identical history within each state group; lanes 0..8 then
execute the nine canonical actions and lane 9 repeats one deterministic action.
The physical stage may report ``PHYSICS_COMPLETE`` but must keep
``physics_validated`` false.  Only a later renderer-receipt join may emit a
``COMPLETE`` manifest with physics-validated RGB targets.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Mapping, Sequence

import numpy as np


PLAN_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_plan_v1"
STATE_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_pilot_state_receipt_v1"
)
PHYSICS_RESULT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_pilot_physics_result_v1"
)
FINAL_MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_manifest_v1"
SMOKE_AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_smoke_execution_authority_v1"
)
SOURCE_REVIEW_SCHEMA = "lewm_go2_world_model_follow_on_independent_source_review_v1"
AUTHORITY_SOURCE_NAMES = (
    "lewm_package_init",
    "benchmarks_package_init",
    "counterfactual_benchmark_support",
    "collector",
    "contract",
    "datasets_package_init",
    "pilot_consumer",
    "genesis_package_init",
    "genesis_batch_renderer",
    "genesis_parity_checks",
    "genesis_render_replay",
    "rollout",
    "scene_builder",
    "scene_loader",
    "go2_adapter",
    "genesis_contract",
    "camera_safety",
    "vision_quality",
    "textures",
    "collectors_package_init",
    "collector_base",
    "collector_frontier",
    "collector_ou_noise",
    "collector_primitive_curriculum",
    "collector_recovery",
    "collector_route_teacher",
    "worlds_package_init",
    "world_corpus",
    "world_exporters_package_init",
    "world_gazebo_exporter",
    "world_splits",
    "world_families",
    "world_randomization",
    "world_manifest",
    "world_scene_validation",
    "world_planning_grid",
    "world_scene_graph",
    "world_genesis_exporter",
    "world_labels_package_init",
    "world_labels_derived",
    "world_labels_topology",
    "scene_generator_materializer",
    "smoke_rgb_writer",
    "checker",
    "external_supervisor",
)
AUTHORITY_SOURCE_PATHS = (
    ("lewm_package_init", "lewm/__init__.py"),
    ("benchmarks_package_init", "lewm/benchmarks/__init__.py"),
    ("counterfactual_benchmark_support", "lewm/benchmarks/counterfactual.py"),
    ("collector", "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"),
    ("contract", "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py"),
    ("datasets_package_init", "lewm/datasets/__init__.py"),
    ("pilot_consumer", "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py"),
    ("genesis_package_init", "lewm_genesis/lewm_genesis/__init__.py"),
    ("genesis_batch_renderer", "lewm_genesis/lewm_genesis/batch_renderer.py"),
    ("genesis_parity_checks", "lewm_genesis/lewm_genesis/parity_checks.py"),
    ("genesis_render_replay", "lewm_genesis/lewm_genesis/render_replay.py"),
    ("rollout", "lewm_genesis/lewm_genesis/rollout.py"),
    ("scene_builder", "lewm_genesis/lewm_genesis/scene_builder.py"),
    ("scene_loader", "lewm_genesis/lewm_genesis/scene_loader.py"),
    ("go2_adapter", "lewm_genesis/lewm_genesis/go2_adapter.py"),
    ("genesis_contract", "lewm_genesis/lewm_genesis/lewm_contract.py"),
    ("camera_safety", "lewm_genesis/lewm_genesis/camera_safety.py"),
    ("vision_quality", "lewm_genesis/lewm_genesis/vision_quality.py"),
    ("textures", "lewm_genesis/lewm_genesis/textures.py"),
    ("collectors_package_init", "lewm_genesis/lewm_genesis/collectors/__init__.py"),
    ("collector_base", "lewm_genesis/lewm_genesis/collectors/base.py"),
    ("collector_frontier", "lewm_genesis/lewm_genesis/collectors/frontier.py"),
    ("collector_ou_noise", "lewm_genesis/lewm_genesis/collectors/ou_noise.py"),
    (
        "collector_primitive_curriculum",
        "lewm_genesis/lewm_genesis/collectors/primitive_curriculum.py",
    ),
    ("collector_recovery", "lewm_genesis/lewm_genesis/collectors/recovery.py"),
    (
        "collector_route_teacher",
        "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
    ),
    ("worlds_package_init", "lewm_worlds/lewm_worlds/__init__.py"),
    ("world_corpus", "lewm_worlds/lewm_worlds/corpus.py"),
    ("world_exporters_package_init", "lewm_worlds/lewm_worlds/exporters/__init__.py"),
    ("world_gazebo_exporter", "lewm_worlds/lewm_worlds/exporters/to_gazebo_sdf.py"),
    ("world_splits", "lewm_worlds/lewm_worlds/splits.py"),
    ("world_families", "lewm_worlds/lewm_worlds/families.py"),
    ("world_randomization", "lewm_worlds/lewm_worlds/randomization.py"),
    ("world_manifest", "lewm_worlds/lewm_worlds/manifest.py"),
    ("world_scene_validation", "lewm_worlds/lewm_worlds/scene_validation.py"),
    ("world_planning_grid", "lewm_worlds/lewm_worlds/planning_grid.py"),
    ("world_scene_graph", "lewm_worlds/lewm_worlds/scene_graph.py"),
    ("world_genesis_exporter", "lewm_worlds/lewm_worlds/exporters/to_genesis.py"),
    ("world_labels_package_init", "lewm_worlds/lewm_worlds/labels/__init__.py"),
    ("world_labels_derived", "lewm_worlds/lewm_worlds/labels/derived.py"),
    ("world_labels_topology", "lewm_worlds/lewm_worlds/labels/topology.py"),
    ("scene_generator_materializer", "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"),
    ("smoke_rgb_writer", "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"),
    ("checker", "scripts/check_go2_world_model_counterfactual_pilot_v1.py"),
    ("external_supervisor", "scripts/run_go2_world_model_counterfactual_smoke_authorized_v1.py"),
)
if tuple(name for name, _path in AUTHORITY_SOURCE_PATHS) != AUTHORITY_SOURCE_NAMES:
    raise RuntimeError("authority source names and exact paths disagree")
RENDER_PLAN_INDEX_SCHEMA = (
    "lewm_go2_world_model_counterfactual_pilot_render_plan_index_v1"
)

CANONICAL_ACTIONS = (
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
ACTION_COUNT = len(CANONICAL_ACTIONS)
CALIBRATION_LANES_PER_STATE = ACTION_COUNT + 1
BLOCK_SIZE = 5
COMMAND_DIM = 3
CONTEXT_FRAME_COUNT = 3
HISTORY_BLOCK_COUNT = 2

# These are part of the scientific action identity, not merely convenient
# registry defaults.  A plan that preserves an action name while changing its
# commanded velocity is a different experiment and must fail validation.
CANONICAL_ACTION_COMMANDS = (
    (0.20, 0.0, 0.45),
    (0.20, 0.0, -0.45),
    (-0.20, 0.0, 0.0),
    (0.30, 0.0, 0.0),
    (0.25, 0.0, 0.0),
    (0.20, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, -0.45),
)
CANONICAL_ACTION_BLOCKS = tuple(
    tuple(command for _ in range(BLOCK_SIZE))
    for command in CANONICAL_ACTION_COMMANDS
)

FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ROLES = ("calibration", "train", "eval")

BRANCH_MECHANISM = "parallel_lockstep_envs_no_restore"
RENDER_CONTRACT = {
    "native_resolution": [640, 480],
    "stored_resolution": [224, 224],
    "rgb_format": "png",
    "depth_validation": "transient_not_persisted",
    "replay_env_mode": "single_non_batched_sequential",
    "replay_pose_source": "captured_physical_base_pose",
    "physical_scene_rendering": False,
}

# The smoke supervisor removes every known accelerator/render selector from its
# inherited environment and installs exactly this mapping in both the hardware
# preflight and collector children.  The values were source-safely re-enumerated
# on 2026-07-31; they are experiment identity, not user-tunable CLI knobs.
EXECUTION_ENVIRONMENT = {
    "EGL_DEVICE_ID": "1",
    "GS_BACKEND": "vulkan",
    "GS_PARA_LEVEL": "0",
    "MESA_VK_DEVICE_SELECT": "1002:7551!",
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
}
GRAPHICS_PREFLIGHT_EXPECTATION = {
    "egl_device_index": 1,
    "eglinfo_expected_exit_code": 2,
    "egl_renderer_name_contains": "AMD Radeon AI PRO R9700",
    "vulkan_device_index": 0,
    "vulkan_vendor_id": "0x1002",
    "vulkan_device_id": "0x7551",
    "vulkan_device_name": "AMD Radeon AI PRO R9700",
}
PLATFORM_GATE_DISPOSITION = {
    "platform_hard_gates_resolved": False,
    "scope": "one_non_citable_source_integration_smoke_only",
    "outputs_eligible_for_training": False,
    "outputs_eligible_for_scientific_claim": False,
    "authorizes_full_data_generation": False,
    "authorizes_promotion": False,
    "basis": "explicit_user_directive_2026-07-31_do_it_narrow_smoke",
}

SYNC_COMPONENTS = (
    "qpos",
    "dofs_velocity",
    "base_pos_world",
    "base_quat_wxyz",
    "base_lin_vel_world",
    "base_ang_vel_world",
    "leg_joint_pos",
    "leg_joint_vel",
    "runner_last_executed",
    "policy_last_actions",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_REPO_ROOT = Path(__file__).resolve().parents[2]


class PilotContractError(RuntimeError):
    """Raised when a pilot plan, state, or receipt violates the V1 contract."""


class PilotDiagnosticError(PilotContractError):
    """Contract failure carrying JSON-safe diagnostics for a consumed attempt."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only JSON encoding used for hash-bound V1 values."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PilotContractError("value is not canonical finite JSON") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_block_sha256(block: Sequence[Sequence[float]]) -> str:
    normalized = _validate_block(block, label="executed block")
    return canonical_json_sha256(normalized)


def target_world_to_body_xy(
    *,
    target_xy_m: Sequence[float],
    base_position_xyz_m: Sequence[float],
    base_quaternion_wxyz: Sequence[float],
) -> list[float]:
    """Convert a world-frame target to planar body coordinates at one pose."""

    def finite_vector(
        value: Sequence[float], *, length: int, label: str
    ) -> list[float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != length
        ):
            raise PilotContractError(f"{label} must have length {length}")
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise PilotContractError(f"{label} must be finite numeric values")
            number = float(item)
            if not math.isfinite(number):
                raise PilotContractError(f"{label} must be finite numeric values")
            result.append(number)
        return result

    target = finite_vector(target_xy_m, length=2, label="target_xy_m")
    position = finite_vector(
        base_position_xyz_m, length=3, label="base_position_xyz_m"
    )
    quaternion = finite_vector(
        base_quaternion_wxyz, length=4, label="base_quaternion_wxyz"
    )
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise PilotContractError("base_quaternion_wxyz has zero norm")
    qw, qx, qy, qz = (component / norm for component in quaternion)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    result = [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
    ]
    if not all(math.isfinite(value) for value in result):
        raise PilotContractError("world-to-body target conversion is nonfinite")
    return result


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _validate_binding_shape(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise PilotContractError(
            f"{label} must contain path, file_sha256, and byte_count"
        )
    path = value["path"]
    byte_count = value["byte_count"]
    if not isinstance(path, str) or not path:
        raise PilotContractError(f"{label} path is invalid")
    lowered_parts = [part.lower() for part in Path(path).parts]
    if any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out"}
        or part.startswith("heldout_")
        or part.startswith("held_out_")
        or part.startswith("held-out-")
        for part in lowered_parts
    ):
        raise PilotContractError(f"{label} path is custody-protected")
    _validate_sha256(value["file_sha256"], label=f"{label} file_sha256")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise PilotContractError(f"{label} byte_count is invalid")
    return dict(value)


def file_binding(path: Path) -> dict[str, Any]:
    """Hash one exact regular non-symlink file without following its leaf."""

    selected = Path(path)
    if selected.is_symlink():
        raise PilotContractError(f"bound file must not be a symlink: {selected}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise PilotContractError(f"cannot open bound file: {selected}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PilotContractError(f"bound input is not regular: {selected}")
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size)
    identity_after = (after.st_dev, after.st_ino, after.st_size)
    if identity_before != identity_after or byte_count != before.st_size:
        raise PilotContractError(f"bound file changed while read: {selected}")
    return {
        "path": str(selected.resolve(strict=True)),
        "file_sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def require_binding(value: object, *, label: str) -> dict[str, Any]:
    expected = _validate_binding_shape(value, label=label)
    actual = file_binding(Path(expected["path"]))
    if actual != expected:
        raise PilotContractError(f"{label} disagrees with the exact file binding")
    return actual


def read_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_sha256(expected_sha256, label=f"expected {label} SHA-256")
    binding = file_binding(path)
    if (
        isinstance(expected_byte_count, bool)
        or not isinstance(expected_byte_count, int)
        or expected_byte_count < 0
        or binding["file_sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise PilotContractError(
            f"{label} SHA-256/byte-count disagrees with expectation"
        )
    def reject_constant(token: str) -> None:
        raise PilotContractError(f"{label} contains nonfinite JSON token {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PilotContractError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            Path(binding["path"]).read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotContractError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise PilotContractError(f"{label} must be a JSON object")
    if file_binding(Path(binding["path"])) != binding:
        raise PilotContractError(f"{label} changed after JSON decoding")
    return payload, binding


def deterministic_sentinel_action_id(
    *,
    state_index_in_scene: int,
    group_index: int | None = None,
    purpose: str = "source_integration_smoke",
) -> int:
    if purpose == "sizing_calibration_only":
        if type(group_index) is not int or group_index < 0:
            raise PilotContractError(
                "rotating calibration sentinel requires a non-negative group_index"
            )
        return group_index % ACTION_COUNT
    if state_index_in_scene == 0:
        return CANONICAL_ACTIONS.index("hold")
    if state_index_in_scene == 1:
        return CANONICAL_ACTIONS.index("forward_medium")
    raise PilotContractError(
        "calibration sentinel allocation is defined only for scene states 0 and 1"
    )


def lane_count_for_role(role: str) -> int:
    if role == "calibration":
        return CALIBRATION_LANES_PER_STATE
    if role in {"train", "eval"}:
        return ACTION_COUNT
    raise PilotContractError(f"unsupported state role: {role!r}")


def lane_layout(
    state_id: str,
    *,
    role: str,
    state_index_in_scene: int,
    sentinel_duplicate_action_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(state_id, str) or _STATE_ID_RE.fullmatch(state_id) is None:
        raise PilotContractError("state_id is invalid")
    candidates = tuple(
        {
            "lane_offset": action_id,
            "kind": "candidate",
            "action_id": action_id,
            "action_name": CANONICAL_ACTIONS[action_id],
        }
        for action_id in range(ACTION_COUNT)
    )
    if role != "calibration":
        lane_count_for_role(role)
        return candidates
    duplicate = sentinel_duplicate_action_id
    if duplicate is None:
        duplicate = deterministic_sentinel_action_id(
            state_index_in_scene=state_index_in_scene
        )
    if type(duplicate) is not int or not 0 <= duplicate < ACTION_COUNT:
        raise PilotContractError("calibration sentinel action ID is invalid")
    sentinel = {
        "lane_offset": ACTION_COUNT,
        "kind": "sentinel",
        "action_id": duplicate,
        "action_name": CANONICAL_ACTIONS[duplicate],
        "duplicates_candidate_action_id": duplicate,
    }
    return (*candidates, sentinel)


def _validate_block(value: object, *, label: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PilotContractError(f"{label} must be a {BLOCK_SIZE}x{COMMAND_DIM} block")
    if len(value) != BLOCK_SIZE:
        raise PilotContractError(f"{label} must contain {BLOCK_SIZE} command ticks")
    result: list[list[float]] = []
    for tick in value:
        if not isinstance(tick, Sequence) or isinstance(tick, (str, bytes)):
            raise PilotContractError(f"{label} contains a malformed command tick")
        if len(tick) != COMMAND_DIM:
            raise PilotContractError(f"{label} command dimension changed")
        row: list[float] = []
        for item in tick:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise PilotContractError(f"{label} contains a nonnumeric command")
            number = float(item)
            if not math.isfinite(number):
                raise PilotContractError(f"{label} contains a nonfinite command")
            row.append(number)
        result.append(row)
    return result


def expected_counts_from_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scene_keys = {
        (str(state["role"]), str(state["family"]), str(state["scene_id"]))
        for state in states
    }
    roles = Counter(str(state["role"]) for state in states)
    state_count = len(states)
    calibration_count = int(roles.get("calibration", 0))
    return {
        "scenes": len(scene_keys),
        "states": state_count,
        "roles": dict(sorted(roles.items())),
        "actions": ACTION_COUNT,
        "candidate_branches": ACTION_COUNT * state_count,
        "sentinel_branches": calibration_count,
        "total_branches": ACTION_COUNT * state_count + calibration_count,
        "context_frames": CONTEXT_FRAME_COUNT * state_count,
        "target_frames": ACTION_COUNT * state_count + calibration_count,
    }


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the complete metadata-only V1 plan."""

    if not isinstance(plan, Mapping):
        raise PilotContractError("pilot plan must be an object")
    required = {
        "schema",
        "attempt_id",
        "purpose",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "allows_refill",
        "allows_overwrite",
        "branch_mechanism",
        "states_per_scene",
        "history_blocks",
        "output_root",
        "runtime_bindings",
        "execution_contract",
        "render_contract",
        "action_catalog",
        "states",
        "expected_counts",
    }
    if set(plan) != required:
        raise PilotContractError(
            "pilot plan field set changed; "
            f"missing={sorted(required - set(plan))} extra={sorted(set(plan) - required)}"
        )
    if plan["schema"] != PLAN_SCHEMA:
        raise PilotContractError("pilot plan schema changed")
    if (
        not isinstance(plan["attempt_id"], str)
        or _STATE_ID_RE.fullmatch(plan["attempt_id"]) is None
    ):
        raise PilotContractError("attempt_id is invalid")
    if plan["purpose"] not in {
        "source_integration_smoke",
        "sizing_calibration_only",
        "bounded_wm_a_pilot",
    }:
        raise PilotContractError("pilot purpose is unsupported")
    if plan["citable_as_scientific_evidence"] is not False:
        raise PilotContractError("pilot plan must remain non-citable development work")
    for field in (
        "authorizes_retry_or_resume",
        "allows_refill",
        "allows_overwrite",
    ):
        if plan[field] is not False:
            raise PilotContractError(f"pilot plan must set {field}=false")
    if plan["branch_mechanism"] != BRANCH_MECHANISM:
        raise PilotContractError("pilot plan may not claim restore or fallback branching")
    states_per_scene = plan["states_per_scene"]
    history_blocks = plan["history_blocks"]
    if (
        isinstance(states_per_scene, bool)
        or not isinstance(states_per_scene, int)
        or states_per_scene < 1
    ):
        raise PilotContractError("states_per_scene must be positive")
    if history_blocks != HISTORY_BLOCK_COUNT:
        raise PilotContractError(
            f"history_blocks must equal {HISTORY_BLOCK_COUNT} in pilot V1"
        )
    if (
        not isinstance(plan["output_root"], str)
        or not plan["output_root"]
        or not Path(plan["output_root"]).is_absolute()
    ):
        raise PilotContractError("output_root is invalid")
    runtime_bindings = plan["runtime_bindings"]
    required_runtime_bindings = {
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
        "python_executable_target",
        "python_environment_config",
        "eglinfo_executable",
        "vulkaninfo_executable",
    }
    if not isinstance(runtime_bindings, Mapping) or set(
        runtime_bindings
    ) != required_runtime_bindings:
        raise PilotContractError("runtime bindings are incomplete")
    for name, binding in runtime_bindings.items():
        _validate_binding_shape(binding, label=f"runtime binding {name}")
    execution_contract = plan["execution_contract"]
    if not isinstance(execution_contract, Mapping) or set(execution_contract) != {
        "backend",
        "policy_device",
        "seed",
        "fall_z_threshold_m",
        "tip_threshold_rad",
        "policy_steps_per_command_tick",
        "python_invocation_path",
        "environment",
        "graphics_preflight",
    }:
        raise PilotContractError("execution contract shape changed")
    if execution_contract["backend"] != "vulkan":
        raise PilotContractError("execution backend must be exact Vulkan")
    if execution_contract["policy_device"] != "cpu":
        raise PilotContractError("policy device must be exact CPU")
    invocation_path = execution_contract["python_invocation_path"]
    if (
        not isinstance(invocation_path, str)
        or not invocation_path
        or not Path(invocation_path).is_absolute()
    ):
        raise PilotContractError("Python invocation path is invalid")
    if (
        isinstance(execution_contract["seed"], bool)
        or not isinstance(execution_contract["seed"], int)
    ):
        raise PilotContractError("execution seed is invalid")
    for threshold_name in ("fall_z_threshold_m", "tip_threshold_rad"):
        threshold = execution_contract[threshold_name]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) <= 0.0
        ):
            raise PilotContractError(f"execution {threshold_name} is invalid")
    if execution_contract["policy_steps_per_command_tick"] != 5:
        raise PilotContractError(
            "pilot V1 requires five policy steps per command tick"
        )
    if canonical_json_bytes(execution_contract["environment"]) != canonical_json_bytes(
        EXECUTION_ENVIRONMENT
    ):
        raise PilotContractError("execution environment selector contract changed")
    if canonical_json_bytes(
        execution_contract["graphics_preflight"]
    ) != canonical_json_bytes(GRAPHICS_PREFLIGHT_EXPECTATION):
        raise PilotContractError("graphics preflight expectation changed")
    if canonical_json_bytes(plan["render_contract"]) != canonical_json_bytes(
        RENDER_CONTRACT
    ):
        raise PilotContractError("render contract changed")

    catalog = plan["action_catalog"]
    if not isinstance(catalog, list) or len(catalog) != ACTION_COUNT:
        raise PilotContractError("action catalog must contain nine actions")
    normalized_catalog: list[dict[str, Any]] = []
    for action_id, entry in enumerate(catalog):
        if not isinstance(entry, Mapping) or set(entry) != {
            "action_id",
            "name",
            "requested_block",
        }:
            raise PilotContractError("action catalog entry shape changed")
        if (
            type(entry["action_id"]) is not int
            or entry["action_id"] != action_id
            or entry["name"] != CANONICAL_ACTIONS[action_id]
        ):
            raise PilotContractError("action catalog order or identity changed")
        requested_block = _validate_block(
            entry["requested_block"], label=f"action {action_id} requested block"
        )
        expected_block = [
            list(command) for command in CANONICAL_ACTION_BLOCKS[action_id]
        ]
        if canonical_json_bytes(requested_block) != canonical_json_bytes(
            expected_block
        ):
            raise PilotContractError(
                f"action {action_id} requested block changed from the canonical primitive"
            )
        normalized_catalog.append(
            {
                "action_id": action_id,
                "name": CANONICAL_ACTIONS[action_id],
                "requested_block": requested_block,
            }
        )

    raw_states = plan["states"]
    if not isinstance(raw_states, list) or not raw_states:
        raise PilotContractError("pilot plan states must be a nonempty ordered list")
    normalized_states: list[dict[str, Any]] = []
    seen_state_ids: set[str] = set()
    scene_metadata: dict[
        tuple[str, str],
        tuple[
            str,
            dict[str, Any] | None,
            dict[str, Any] | None,
        ],
    ] = {}
    scene_roles: dict[str, str] = {}
    groups_by_scene: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, Mapping):
            raise PilotContractError(f"state {position} is malformed")
        state_required = {
            "state_id",
            "role",
            "family",
            "scene_id",
            "scene_manifest_binding",
            "scene_genesis_binding",
            "scene_generation",
            "group_index",
            "state_index_in_scene",
            "history_action_ids",
            "candidate_action_ids",
            "sentinel_duplicate_action_id",
            "target_xy_m",
        }
        if set(raw_state) != state_required:
            raise PilotContractError(
                f"state {position} field set changed; "
                f"missing={sorted(state_required - set(raw_state))} "
                f"extra={sorted(set(raw_state) - state_required)}"
            )
        state_id = raw_state["state_id"]
        if not isinstance(state_id, str) or _STATE_ID_RE.fullmatch(state_id) is None:
            raise PilotContractError(f"state {position} has an invalid state_id")
        if state_id in seen_state_ids:
            raise PilotContractError(f"duplicate state_id: {state_id}")
        seen_state_ids.add(state_id)
        role = raw_state["role"]
        family = raw_state["family"]
        scene_id = raw_state["scene_id"]
        if role not in ROLES or family not in FAMILIES:
            raise PilotContractError(f"state {state_id} role or family is invalid")
        if not isinstance(scene_id, str) or not scene_id:
            raise PilotContractError(f"state {state_id} scene_id is invalid")
        previous_role = scene_roles.setdefault(scene_id, role)
        if previous_role != role:
            raise PilotContractError("one scene appears in more than one role")
        scene_generation = raw_state["scene_generation"]
        if plan["purpose"] == "source_integration_smoke":
            if (
                raw_state["scene_manifest_binding"] is not None
                or raw_state["scene_genesis_binding"] is not None
            ):
                raise PilotContractError(
                    "source integration smoke may not borrow pre-existing scene files"
                )
            if not isinstance(scene_generation, Mapping) or set(scene_generation) != {
                "family",
                "split",
                "plan_seed",
                "scene_index",
                "scene_generator_binding",
            }:
                raise PilotContractError("smoke scene generation declaration changed")
            if (
                scene_generation["family"] != family
                or scene_generation["split"] != "calibration_smoke"
                or isinstance(scene_generation["plan_seed"], bool)
                or not isinstance(scene_generation["plan_seed"], int)
                or scene_generation["scene_index"] != 0
            ):
                raise PilotContractError("smoke scene generation identity is invalid")
            generator_binding = _validate_binding_shape(
                scene_generation["scene_generator_binding"],
                label="smoke scene generator binding",
            )
            normalized_generation: dict[str, Any] | None = dict(scene_generation)
            normalized_generation["scene_generator_binding"] = generator_binding
            manifest_binding = None
            genesis_binding = None
        else:
            if scene_generation is not None:
                raise PilotContractError(
                    "calibration/pilot scenes must be exact pre-existing bindings"
                )
            manifest_binding = _validate_binding_shape(
                raw_state["scene_manifest_binding"],
                label=f"state {state_id} scene manifest binding",
            )
            genesis_binding = _validate_binding_shape(
                raw_state["scene_genesis_binding"],
                label=f"state {state_id} Genesis scene binding",
            )
            if not Path(manifest_binding["path"]).is_absolute() or not Path(
                genesis_binding["path"]
            ).is_absolute():
                raise PilotContractError(
                    "calibration/pilot scene bindings must be absolute inputs"
                )
            if Path(manifest_binding["path"]).name != "manifest.json":
                raise PilotContractError("scene manifest binding must name manifest.json")
            if Path(genesis_binding["path"]).name != "genesis_scene.json":
                raise PilotContractError(
                    "Genesis scene binding must name genesis_scene.json"
                )
            if Path(manifest_binding["path"]).parent != Path(
                genesis_binding["path"]
            ).parent:
                raise PilotContractError(
                    "scene manifest and Genesis scene bindings must share one directory"
                )
            normalized_generation = None
        scene_key = (role, scene_id)
        previous_metadata = scene_metadata.setdefault(
            scene_key, (family, manifest_binding, genesis_binding)
        )
        if previous_metadata != (family, manifest_binding, genesis_binding):
            raise PilotContractError("one scene has inconsistent family or manifest identity")
        group_index = raw_state["group_index"]
        state_index = raw_state["state_index_in_scene"]
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or group_index != position
            or isinstance(state_index, bool)
            or not isinstance(state_index, int)
            or state_index < 0
        ):
            raise PilotContractError(
                f"state {state_id} group/state index contract changed"
            )
        groups_by_scene[scene_key].append(state_index)
        history = raw_state["history_action_ids"]
        if (
            not isinstance(history, list)
            or len(history) != history_blocks
            or any(
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < ACTION_COUNT
                for action in history
            )
        ):
            raise PilotContractError(f"state {state_id} history action tape is invalid")
        if (
            raw_state["candidate_action_ids"] != list(range(ACTION_COUNT))
            or any(type(value) is not int for value in raw_state["candidate_action_ids"])
        ):
            raise PilotContractError(f"state {state_id} candidate action grid changed")
        sentinel = (
            deterministic_sentinel_action_id(
                state_index_in_scene=state_index,
                group_index=group_index,
                purpose=str(plan["purpose"]),
            )
            if role == "calibration"
            else None
        )
        if raw_state["sentinel_duplicate_action_id"] != sentinel:
            raise PilotContractError(
                f"state {state_id} sentinel action is not deterministic"
            )
        target_xy = raw_state["target_xy_m"]
        if (
            not isinstance(target_xy, list)
            or len(target_xy) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in target_xy
            )
        ):
            raise PilotContractError(f"state {state_id} target_xy_m is invalid")
        normalized_state = dict(raw_state)
        normalized_state["scene_manifest_binding"] = manifest_binding
        normalized_state["scene_genesis_binding"] = genesis_binding
        normalized_state["scene_generation"] = normalized_generation
        normalized_state["history_action_ids"] = list(history)
        normalized_state["candidate_action_ids"] = list(range(ACTION_COUNT))
        normalized_state["target_xy_m"] = [float(value) for value in target_xy]
        normalized_states.append(normalized_state)

    expected_group_indices = list(range(states_per_scene))
    for scene_key, values in groups_by_scene.items():
        if sorted(values) != expected_group_indices:
            raise PilotContractError(
                f"scene {scene_key} does not contain the exact state-group grid"
            )
    computed_counts = expected_counts_from_states(normalized_states)
    if canonical_json_bytes(plan["expected_counts"]) != canonical_json_bytes(
        computed_counts
    ):
        raise PilotContractError("expected_counts disagree with the ordered state plan")
    if plan["purpose"] == "sizing_calibration_only" and set(
        computed_counts["roles"]
    ) != {"calibration"}:
        raise PilotContractError("sizing calibration states must all use calibration role")
    if plan["purpose"] == "source_integration_smoke":
        required_smoke_counts = {
            "scenes": 1,
            "states": 1,
            "roles": {"calibration": 1},
            "actions": ACTION_COUNT,
            "candidate_branches": ACTION_COUNT,
            "sentinel_branches": 1,
            "total_branches": CALIBRATION_LANES_PER_STATE,
            "context_frames": CONTEXT_FRAME_COUNT,
            "target_frames": CALIBRATION_LANES_PER_STATE,
        }
        if states_per_scene != 1 or computed_counts != required_smoke_counts:
            raise PilotContractError(
                "source integration smoke must be exactly one calibration scene, "
                "one state, and ten branches"
            )
    if plan["purpose"] == "sizing_calibration_only":
        if (
            states_per_scene != 2
            or computed_counts["scenes"] != 8
            or computed_counts["states"] != 16
            or computed_counts["sentinel_branches"] != 16
            or computed_counts["total_branches"] != 160
            or {state["family"] for state in normalized_states} != set(FAMILIES)
        ):
            raise PilotContractError(
                "sizing calibration must be eight families/scenes, two states "
                "per scene, and 160 physical branches"
            )
    if plan["purpose"] == "bounded_wm_a_pilot" and "calibration" in computed_counts[
        "roles"
    ]:
        raise PilotContractError("train/eval pilot may not consume calibration rows")
    normalized = dict(plan)
    normalized["action_catalog"] = normalized_catalog
    normalized["states"] = normalized_states
    normalized["expected_counts"] = computed_counts
    return normalized


def validate_authority(
    authority: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the semantic one-shot grant before any runtime import."""

    if not isinstance(authority, Mapping):
        raise PilotContractError("execution authority must be an object")
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_authorized",
        "authorizer",
        "issued_at",
        "source_commit",
        "review_binding",
        "plan_binding",
        "source_bindings",
        "attempt",
        "caps",
        "runtime_bindings",
        "execution",
        "network_access",
        "external_supervisor",
        "platform_gate_disposition",
    }
    if set(authority) != required:
        raise PilotContractError(
            "execution authority field set changed; "
            f"missing={sorted(required - set(authority))} "
            f"extra={sorted(set(authority) - required)}"
        )
    if (
        plan["purpose"] != "source_integration_smoke"
        or authority["schema"] != SMOKE_AUTHORITY_SCHEMA
        or authority["status"] != "AUTHORIZED_ONE_EXACT_SOURCE_INTEGRATION_SMOKE"
        or authority["authority_granted_by_this_document"] is not True
        or authority["scientific_claim_authorized"] is not False
    ):
        raise PilotContractError("authority does not grant the one exact source smoke")
    authorizer = authority["authorizer"]
    if (
        not isinstance(authorizer, Mapping)
        or set(authorizer) != {"identity", "basis"}
        or not isinstance(authorizer["identity"], str)
        or not authorizer["identity"].strip()
        or not isinstance(authorizer["basis"], str)
        or not authorizer["basis"].strip()
    ):
        raise PilotContractError("authority authorizer identity/basis is invalid")
    issued_at = authority["issued_at"]
    if not isinstance(issued_at, str) or not issued_at:
        raise PilotContractError("authority issued_at is invalid")
    try:
        datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PilotContractError("authority issued_at is not ISO-8601") from exc
    source_commit = authority["source_commit"]
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise PilotContractError("authority source_commit is invalid")
    normalized_binding = _validate_binding_shape(
        authority["plan_binding"], label="authority plan binding"
    )
    if normalized_binding != dict(plan_binding):
        raise PilotContractError("authority does not bind the selected plan")
    review_binding = _validate_binding_shape(
        authority["review_binding"], label="authority review binding"
    )
    source_bindings = authority["source_bindings"]
    if not isinstance(source_bindings, list) or [
        entry.get("name") if isinstance(entry, Mapping) else None
        for entry in source_bindings
    ] != list(AUTHORITY_SOURCE_NAMES):
        raise PilotContractError("authority source binding order changed")
    normalized_sources: list[dict[str, Any]] = []
    for entry in source_bindings:
        if set(entry) != {"name", "binding"}:
            raise PilotContractError("authority source binding shape changed")
        normalized_sources.append(
            {
                "name": str(entry["name"]),
                "binding": _validate_binding_shape(
                    entry["binding"], label=f"source {entry['name']} binding"
                ),
            }
        )
    expected_source_paths = dict(AUTHORITY_SOURCE_PATHS)
    for source in normalized_sources:
        expected_path = str(
            (_REPO_ROOT / expected_source_paths[source["name"]]).absolute()
        )
        actual_path = str(source["binding"]["path"])
        if (
            not Path(actual_path).is_absolute()
            or os.path.normpath(actual_path) != expected_path
            or actual_path != expected_path
        ):
            raise PilotContractError(
                f"authority source {source['name']} path is not the exact reviewed path"
            )
    attempt = authority["attempt"]
    exact_attempt = {
        "id": plan["attempt_id"],
        "root": plan["output_root"],
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    if canonical_json_bytes(attempt) != canonical_json_bytes(exact_attempt):
        raise PilotContractError("authority attempt boundary disagrees with plan")
    caps = authority["caps"]
    required_caps = {
        "scenes": 1,
        "states": 1,
        "candidate_branches": 9,
        "sentinel_branches": 1,
        "total_branches": 10,
        "candidate_branch_simulated_seconds": 5.0,
        "total_lane_simulated_seconds_including_common_prefix": 15.0,
        "policy_steps_per_lane": 75,
        "total_lane_policy_steps": 750,
        "total_lane_physics_steps": 7500,
        "native_render_calls": 13,
        "stored_rgb_frames": 13,
    }
    if not isinstance(caps, Mapping) or set(caps) != {
        *required_caps,
        "wall_seconds",
    }:
        raise PilotContractError("authority smoke cap shape changed")
    if any(
        type(caps[name]) is not type(value) or caps[name] != value
        for name, value in required_caps.items()
    ):
        raise PilotContractError("authority smoke caps changed")
    wall_seconds = caps["wall_seconds"]
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
    ):
        raise PilotContractError("authority wall_seconds cap is invalid")
    if canonical_json_bytes(authority["runtime_bindings"]) != canonical_json_bytes(
        plan["runtime_bindings"]
    ):
        raise PilotContractError("authority runtime bindings disagree with plan")
    if canonical_json_bytes(authority["execution"]) != canonical_json_bytes(
        plan["execution_contract"]
    ):
        raise PilotContractError("authority execution contract disagrees with plan")
    if authority["network_access"] is not False:
        raise PilotContractError("smoke authority must forbid network access")
    if canonical_json_bytes(
        authority["platform_gate_disposition"]
    ) != canonical_json_bytes(PLATFORM_GATE_DISPOSITION):
        raise PilotContractError(
            "authority lacks the exact non-citable platform-gate disposition"
        )
    supervisor = authority["external_supervisor"]
    if (
        not isinstance(supervisor, Mapping)
        or set(supervisor) != {"source_binding", "terminal_reviewer"}
        or not isinstance(supervisor["terminal_reviewer"], str)
        or not supervisor["terminal_reviewer"].strip()
    ):
        raise PilotContractError("external supervisor contract is invalid")
    supervisor_binding = _validate_binding_shape(
        supervisor["source_binding"], label="external supervisor source binding"
    )
    reviewed_supervisor = next(
        row["binding"]
        for row in normalized_sources
        if row["name"] == "external_supervisor"
    )
    if supervisor_binding != reviewed_supervisor:
        raise PilotContractError(
            "external supervisor binding is not the reviewed supervisor source"
        )
    normalized = dict(authority)
    normalized["plan_binding"] = normalized_binding
    normalized["review_binding"] = review_binding
    normalized["source_bindings"] = normalized_sources
    normalized["external_supervisor"] = {
        "source_binding": supervisor_binding,
        "terminal_reviewer": supervisor["terminal_reviewer"],
    }
    return normalized


def validate_source_review(
    review: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "reviewed_source_commit",
        "reviewed_source_bindings",
        "remaining_findings",
        "reviewer",
        "reviewed_at",
        "review_method",
        "test_evidence",
        "accepted_limitations",
    }
    if not isinstance(review, Mapping) or set(review) != required:
        raise PilotContractError("independent source review field set changed")
    if (
        review["schema"] != SOURCE_REVIEW_SCHEMA
        or review["status"] != "PASS_SOURCE_ONLY_NOT_AUTHORITY"
        or review["authority_granted_by_this_document"] is not False
        or review["reviewed_source_commit"] != authority["source_commit"]
        or canonical_json_bytes(review["reviewed_source_bindings"])
        != canonical_json_bytes(authority["source_bindings"])
        or review["remaining_findings"] != []
    ):
        raise PilotContractError("independent source review does not exactly pass")
    reviewer = review["reviewer"]
    if (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"identity", "independence_basis"}
        or not isinstance(reviewer["identity"], str)
        or not reviewer["identity"].strip()
        or not isinstance(reviewer["independence_basis"], str)
        or not reviewer["independence_basis"].strip()
    ):
        raise PilotContractError("independent source review provenance is invalid")
    reviewed_at = review["reviewed_at"]
    if not isinstance(reviewed_at, str) or not reviewed_at:
        raise PilotContractError("independent source review time is invalid")
    try:
        datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PilotContractError(
            "independent source review time is not ISO-8601"
        ) from exc
    for field in ("review_method", "test_evidence", "accepted_limitations"):
        rows = review[field]
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, str) or not row.strip() for row in rows)
        ):
            raise PilotContractError(
                f"independent source review {field} must be a nonempty string list"
            )
    return dict(review)


def require_plan_bindings(plan: Mapping[str, Any]) -> None:
    for name, binding in plan["runtime_bindings"].items():
        require_binding(binding, label=f"runtime binding {name}")
    seen: set[tuple[str, str, int]] = set()
    for state in plan["states"]:
        if state["scene_generation"] is not None:
            require_binding(
                state["scene_generation"]["scene_generator_binding"],
                label="smoke scene generator binding",
            )
            continue
        for binding_name in ("scene_manifest_binding", "scene_genesis_binding"):
            binding = state[binding_name]
            identity = (
                str(binding["path"]),
                str(binding["file_sha256"]),
                int(binding["byte_count"]),
            )
            if identity in seen:
                continue
            require_binding(
                binding,
                label=f"{binding_name} {state['scene_id']}",
            )
            seen.add(identity)


def fresh_development_output_root(path: Path, *, development_root: Path) -> Path:
    dev = development_root.resolve()
    dev.mkdir(parents=True, exist_ok=True)
    selected = Path(path)
    if selected.is_symlink() or selected.exists():
        raise PilotContractError(f"pilot output root must be fresh: {selected}")
    resolved = selected.resolve(strict=False)
    try:
        resolved.relative_to(dev)
    except ValueError as exc:
        raise PilotContractError(
            f"pilot output root must remain below development root {dev}"
        ) from exc
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_json_exclusive(path: Path, value: Any) -> dict[str, Any]:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    with selected.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return file_binding(selected)


def write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    with selected.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row).decode("utf-8"))
            stream.write("\n")
    return file_binding(selected)


def _canonical_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in "fc":
        if not np.isfinite(array).all():
            raise PilotContractError("state component contains nonfinite values")
        return np.ascontiguousarray(array.astype("<f4", copy=False))
    if array.dtype.kind in "biu":
        return np.ascontiguousarray(array.astype("<i8", copy=False))
    raise PilotContractError(f"unsupported state component dtype: {array.dtype}")


def _lane_state_sha256(
    components: Mapping[str, np.ndarray], lane_index: int
) -> str:
    digest = hashlib.sha256()
    for name in SYNC_COMPONENTS:
        lane = _canonical_array(np.asarray(components[name])[lane_index])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(lane.shape).encode("ascii") + b"\0")
        digest.update(lane.dtype.str.encode("ascii") + b"\0")
        digest.update(lane.tobytes(order="C"))
    return digest.hexdigest()


def audit_prebranch_synchronization(
    components: Mapping[str, np.ndarray],
    *,
    state_ids: Sequence[str],
    roles: Sequence[str],
) -> list[dict[str, Any]]:
    """Require exact equality within each role-aware state lane group."""

    if set(components) != set(SYNC_COMPONENTS):
        missing = sorted(set(SYNC_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(SYNC_COMPONENTS))
        raise PilotContractError(
            f"synchronization component set changed; missing={missing} extra={extra}"
        )
    if len(roles) != len(state_ids):
        raise PilotContractError("state ids and roles disagree in length")
    lane_counts = [lane_count_for_role(role) for role in roles]
    expected_lanes = sum(lane_counts)
    canonical: dict[str, np.ndarray] = {}
    for name in SYNC_COMPONENTS:
        array = _canonical_array(np.asarray(components[name]))
        if array.ndim < 1 or array.shape[0] != expected_lanes:
            raise PilotContractError(
                f"state component {name} must begin with {expected_lanes} lanes"
            )
        canonical[name] = array

    audits: list[dict[str, Any]] = []
    lane_start = 0
    for group_index, (state_id, lane_count) in enumerate(
        zip(state_ids, lane_counts, strict=True)
    ):
        start = lane_start
        stop = start + lane_count
        component_audits: dict[str, Any] = {}
        group_passed = True
        for name in SYNC_COMPONENTS:
            group = canonical[name][start:stop]
            reference = group[0]
            exact = all(np.array_equal(reference, candidate) for candidate in group[1:])
            delta = group.astype(np.float64) - reference.astype(np.float64)
            absolute = np.abs(delta)
            max_abs = float(np.max(absolute))
            rms = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            per_lane_max_abs = np.max(
                absolute.reshape((lane_count, -1)), axis=1
            ).tolist()
            component_audits[name] = {
                "exact_equal": exact,
                "max_abs_difference": max_abs,
                "rms_difference": rms,
                "per_lane_max_abs_difference": per_lane_max_abs,
                "shape_per_lane": list(reference.shape),
            }
            group_passed = group_passed and exact
        lane_hashes = [
            _lane_state_sha256(canonical, lane)
            for lane in range(start, stop)
        ]
        if len(set(lane_hashes)) != 1:
            group_passed = False
        audits.append(
            {
                "state_id": state_id,
                "group_index": group_index,
                "lane_start": start,
                "lane_count": lane_count,
                "exact_equality_required": True,
                "passed": group_passed,
                "prebranch_state_sha256": lane_hashes[0],
                "lane_state_sha256s": lane_hashes,
                "components": component_audits,
            }
        )
        lane_start = stop
    return audits


def audit_sentinel_trajectories(
    trajectory_components: Mapping[str, np.ndarray],
    *,
    state_ids: Sequence[str],
    roles: Sequence[str],
    state_indices_in_scene: Sequence[int],
    sentinel_action_ids: Sequence[int | None] | None = None,
) -> list[dict[str, Any]]:
    """Compare each lane-9 repeat with its deterministic candidate lane."""

    if not trajectory_components:
        raise PilotContractError("sentinel trajectory components are absent")
    if len(roles) != len(state_ids) or len(state_indices_in_scene) != len(state_ids):
        raise PilotContractError("state ids, roles, and indices disagree in length")
    if sentinel_action_ids is None:
        sentinel_action_ids = tuple(
            deterministic_sentinel_action_id(state_index_in_scene=int(index))
            if role == "calibration"
            else None
            for role, index in zip(roles, state_indices_in_scene, strict=True)
        )
    if len(sentinel_action_ids) != len(state_ids):
        raise PilotContractError("sentinel action IDs disagree in length")
    lane_counts = [lane_count_for_role(role) for role in roles]
    expected_lanes = sum(lane_counts)
    canonical: dict[str, np.ndarray] = {}
    step_count: int | None = None
    for name, value in trajectory_components.items():
        array = _canonical_array(np.asarray(value))
        if array.ndim < 2 or array.shape[1] != expected_lanes:
            raise PilotContractError(
                f"trajectory component {name} must have shape (steps,{expected_lanes},...)"
            )
        if step_count is None:
            step_count = int(array.shape[0])
        elif int(array.shape[0]) != step_count:
            raise PilotContractError("trajectory components disagree on step count")
        canonical[name] = array
    assert step_count is not None
    audits: list[dict[str, Any]] = []
    lane_start = 0
    for group_index, (state_id, role, state_index, lane_count) in enumerate(
        zip(state_ids, roles, state_indices_in_scene, lane_counts, strict=True)
    ):
        if role != "calibration":
            lane_start += lane_count
            continue
        duplicate = sentinel_action_ids[group_index]
        if type(duplicate) is not int or not 0 <= duplicate < ACTION_COUNT:
            raise PilotContractError("calibration sentinel action ID is invalid")
        candidate_lane = lane_start + duplicate
        sentinel_lane = lane_start + ACTION_COUNT
        component_audits: dict[str, Any] = {}
        passed = True
        digest_candidate = hashlib.sha256()
        digest_sentinel = hashlib.sha256()
        for name in sorted(canonical):
            candidate = canonical[name][:, candidate_lane]
            sentinel = canonical[name][:, sentinel_lane]
            exact = np.array_equal(candidate, sentinel)
            max_abs = float(np.max(np.abs(candidate - sentinel)))
            component_audits[name] = {
                "exact_equal": exact,
                "max_abs_difference": max_abs,
            }
            passed = passed and exact
            digest_candidate.update(name.encode() + b"\0" + candidate.tobytes())
            digest_sentinel.update(name.encode() + b"\0" + sentinel.tobytes())
        candidate_sha = digest_candidate.hexdigest()
        sentinel_sha = digest_sentinel.hexdigest()
        passed = passed and candidate_sha == sentinel_sha
        audits.append(
            {
                "state_id": state_id,
                "group_index": group_index,
                "action_id": duplicate,
                "candidate_lane": candidate_lane,
                "sentinel_lane": sentinel_lane,
                "policy_step_count": step_count,
                "exact_equality_required": True,
                "physics_equal": passed,
                "candidate_trajectory_sha256": candidate_sha,
                "sentinel_trajectory_sha256": sentinel_sha,
                "components": component_audits,
            }
        )
        lane_start += lane_count
    return audits


def _copy_state_components(
    components: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(components) != set(SYNC_COMPONENTS):
        missing = sorted(set(SYNC_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(SYNC_COMPONENTS))
        raise PilotContractError(
            f"captured state component set changed; missing={missing} extra={extra}"
        )
    return {
        name: _canonical_array(np.asarray(components[name])).copy()
        for name in SYNC_COMPONENTS
    }


def execute_lockstep_trial(
    *,
    runner: Any,
    states: Sequence[Mapping[str, Any]],
    action_blocks: Sequence[Sequence[Sequence[float]]],
    capture_components: Any,
    capture_sim_time_ns: Any,
    capture_render_batch: Any | None = None,
) -> dict[str, Any]:
    """Execute one scene's cloned histories and one synchronized branch.

    This function intentionally depends only on the narrow runner method
    ``execute_requested_block``.  Genesis setup, reset, rendering, and receipt
    persistence stay outside it, which makes the causal experiment testable
    with a synthetic runner and prevents an accidental snapshot/restore path.
    """

    if not states:
        raise PilotContractError("one lockstep trial needs at least one state")
    state_ids = [str(state["state_id"]) for state in states]
    roles = [str(state["role"]) for state in states]
    state_indices = [int(state["state_index_in_scene"]) for state in states]
    lane_counts = [lane_count_for_role(role) for role in roles]
    lane_starts = np.cumsum([0, *lane_counts[:-1]]).tolist()
    expected_envs = sum(lane_counts)
    if int(getattr(runner, "n_envs", -1)) != expected_envs:
        raise PilotContractError(
            f"runner has {getattr(runner, 'n_envs', None)} envs; expected {expected_envs}"
        )
    normalized_blocks = np.asarray(
        [
            _validate_block(block, label=f"action {action_id} block")
            for action_id, block in enumerate(action_blocks)
        ],
        dtype=np.float32,
    )
    if normalized_blocks.shape != (ACTION_COUNT, BLOCK_SIZE, COMMAND_DIM):
        raise PilotContractError("lockstep action block grid changed")
    for state in states:
        history = state.get("history_action_ids")
        if not isinstance(history, list) or len(history) != HISTORY_BLOCK_COUNT:
            raise PilotContractError("lockstep state history must contain two actions")

    history_snapshots = [_copy_state_components(capture_components())]
    history_times_ns = [int(capture_sim_time_ns())]
    history_synchronization_audits = [
        audit_prebranch_synchronization(
            history_snapshots[0], state_ids=state_ids, roles=roles
        )
    ]
    if any(not audit["passed"] for audit in history_synchronization_audits[0]):
        raise PilotDiagnosticError(
            "initial cloned environments are not exactly equal",
            diagnostics={
                "phase": "initial_clone",
                "sim_time_ns": history_times_ns[0],
                "synchronization_audits": history_synchronization_audits[0],
            },
        )
    render_batches: list[Any] = []

    def capture_render_without_physics_mutation() -> Any:
        before = _copy_state_components(capture_components())
        batch = capture_render_batch()
        after = _copy_state_components(capture_components())
        changed = [
            name
            for name in SYNC_COMPONENTS
            if not np.array_equal(before[name], after[name])
        ]
        if changed:
            raise PilotContractError(
                f"live render mutated physical/controller state: {changed}"
            )
        return batch

    if capture_render_batch is not None:
        render_batches.append(capture_render_without_physics_mutation())
    history_blocks: list[dict[str, Any]] = []
    common_prefix_step_wall_seconds = 0.0
    for history_index in range(HISTORY_BLOCK_COUNT):
        requested = np.empty(
            (expected_envs, BLOCK_SIZE, COMMAND_DIM), dtype=np.float32
        )
        for group_index, state in enumerate(states):
            action_id = int(state["history_action_ids"][history_index])
            start = int(lane_starts[group_index])
            lane_count = lane_counts[group_index]
            requested[start : start + lane_count] = normalized_blocks[action_id]
        def audit_history_policy_step(
            command_tick_index: int, policy_step_index: int
        ) -> None:
            checkpoint = audit_prebranch_synchronization(
                _copy_state_components(capture_components()),
                state_ids=state_ids,
                roles=roles,
            )
            failed = [row["state_id"] for row in checkpoint if not row["passed"]]
            if failed:
                raise PilotDiagnosticError(
                    f"history block {history_index} first diverged at command tick "
                    f"{command_tick_index}, policy step {policy_step_index} for states: "
                    + ", ".join(failed),
                    diagnostics={
                        "phase": "common_history_policy_step",
                        "history_index": history_index,
                        "command_tick_index": int(command_tick_index),
                        "policy_step_index": int(policy_step_index),
                        "block_policy_step_index": (
                            int(command_tick_index)
                            * int(getattr(runner, "policy_steps_per_command_tick", 5))
                            + int(policy_step_index)
                        ),
                        "sim_time_ns": int(capture_sim_time_ns()),
                        "synchronization_audits": checkpoint,
                    },
                )

        history_step_started = time.perf_counter()
        trajectory = runner.execute_requested_block(
            requested, after_policy_step=audit_history_policy_step
        )
        common_prefix_step_wall_seconds += time.perf_counter() - history_step_started
        executed = np.asarray(trajectory.executed, dtype=np.float32)
        clipped = np.asarray(trajectory.clipped, dtype=np.bool_)
        if executed.shape != requested.shape or clipped.shape != (expected_envs,):
            raise PilotContractError("runner returned a malformed history trajectory")
        per_state: list[dict[str, Any]] = []
        for group_index, state in enumerate(states):
            start = int(lane_starts[group_index])
            lane_count = lane_counts[group_index]
            group = executed[start : start + lane_count]
            if not all(np.array_equal(group[0], row) for row in group[1:]):
                raise PilotContractError(
                    f"history execution diverged within state {state['state_id']}"
                )
            if not all(
                bool(clipped[start]) == bool(value)
                for value in clipped[start + 1 : start + lane_count]
            ):
                raise PilotContractError(
                    f"history clipping diverged within state {state['state_id']}"
                )
            per_state.append(
                {
                    "state_id": str(state["state_id"]),
                    "action_id": int(state["history_action_ids"][history_index]),
                    "executed": group[0].copy(),
                    "clipped": bool(clipped[start]),
                }
            )
        history_blocks.append(
            {"history_index": history_index, "states": per_state}
        )
        history_snapshots.append(_copy_state_components(capture_components()))
        history_times_ns.append(int(capture_sim_time_ns()))
        checkpoint_audits = audit_prebranch_synchronization(
            history_snapshots[-1], state_ids=state_ids, roles=roles
        )
        history_synchronization_audits.append(checkpoint_audits)
        failed_checkpoints = [
            audit["state_id"] for audit in checkpoint_audits if not audit["passed"]
        ]
        if failed_checkpoints:
            raise PilotDiagnosticError(
                f"history checkpoint {history_index} diverged for states: "
                + ", ".join(failed_checkpoints),
                diagnostics={
                    "phase": "common_history_checkpoint",
                    "history_index": history_index,
                    "sim_time_ns": history_times_ns[-1],
                    "synchronization_audits": checkpoint_audits,
                },
            )
        if capture_render_batch is not None:
            render_batches.append(capture_render_without_physics_mutation())

    synchronization_audits = history_synchronization_audits[-1]
    failed_sync = [audit["state_id"] for audit in synchronization_audits if not audit["passed"]]
    if failed_sync:
        raise PilotDiagnosticError(
            "prebranch exact-equality audit failed for states: "
            + ", ".join(failed_sync),
            diagnostics={
                "phase": "prebranch_checkpoint",
                "sim_time_ns": history_times_ns[-1],
                "synchronization_audits": synchronization_audits,
            },
        )

    branch_requested = np.empty(
        (expected_envs, BLOCK_SIZE, COMMAND_DIM), dtype=np.float32
    )
    for group_index, state in enumerate(states):
        start = int(lane_starts[group_index])
        for lane in lane_layout(
            str(state["state_id"]),
            role=roles[group_index],
            state_index_in_scene=state_indices[group_index],
            sentinel_duplicate_action_id=state.get(
                "sentinel_duplicate_action_id"
            ),
        ):
            branch_requested[start + int(lane["lane_offset"])] = normalized_blocks[
                int(lane["action_id"])
            ]

    trajectory_samples: list[dict[str, np.ndarray]] = []
    trajectory_times_ns: list[int] = []

    def after_policy_step(_tick_index: int, _policy_step_index: int) -> None:
        trajectory_samples.append(_copy_state_components(capture_components()))
        trajectory_times_ns.append(int(capture_sim_time_ns()))

    branch_step_started = time.perf_counter()
    branch_trajectory = runner.execute_requested_block(
        branch_requested,
        after_policy_step=after_policy_step,
    )
    branch_step_wall_seconds = time.perf_counter() - branch_step_started
    branch_executed = np.asarray(branch_trajectory.executed, dtype=np.float32)
    branch_clipped = np.asarray(branch_trajectory.clipped, dtype=np.bool_)
    if branch_executed.shape != branch_requested.shape or branch_clipped.shape != (
        expected_envs,
    ):
        raise PilotContractError("runner returned a malformed branch trajectory")
    branch_endpoint = _copy_state_components(capture_components())
    expected_policy_steps = BLOCK_SIZE * int(
        getattr(runner, "policy_steps_per_command_tick", 5)
    )
    if len(trajectory_samples) != expected_policy_steps:
        raise PilotContractError(
            f"branch captured {len(trajectory_samples)} policy steps; expected {expected_policy_steps}"
        )
    stacked = {
        name: np.stack([sample[name] for sample in trajectory_samples], axis=0)
        for name in SYNC_COMPONENTS
    }
    sentinel_audits = audit_sentinel_trajectories(
        stacked,
        state_ids=state_ids,
        roles=roles,
        state_indices_in_scene=state_indices,
        sentinel_action_ids=[
            (
                state.get("sentinel_duplicate_action_id")
                if state.get("sentinel_duplicate_action_id") is not None
                else deterministic_sentinel_action_id(
                    state_index_in_scene=int(state["state_index_in_scene"])
                )
            )
            if state["role"] == "calibration"
            else None
            for state in states
        ],
    )
    failed_sentinels = [
        audit["state_id"] for audit in sentinel_audits if not audit["physics_equal"]
    ]
    if failed_sentinels:
        raise PilotDiagnosticError(
            "duplicate-sentinel exact-equality audit failed for states: "
            + ", ".join(failed_sentinels),
            diagnostics={
                "phase": "duplicate_sentinel_trajectory",
                "sim_time_ns": int(capture_sim_time_ns()),
                "sentinel_audits": sentinel_audits,
            },
        )
    if capture_render_batch is not None:
        render_batches.append(capture_render_without_physics_mutation())
        if len(render_batches) != CONTEXT_FRAME_COUNT + 1:
            raise PilotContractError("live render call count changed")
    return {
        "state_ids": state_ids,
        "history_snapshots": history_snapshots,
        "history_times_ns": history_times_ns,
        "history_blocks": history_blocks,
        "history_synchronization_audits": history_synchronization_audits,
        "synchronization_audits": synchronization_audits,
        "common_prefix_step_wall_seconds": common_prefix_step_wall_seconds,
        "branch_step_wall_seconds": branch_step_wall_seconds,
        "branch_requested": branch_requested,
        "branch_executed": branch_executed,
        "branch_clipped": branch_clipped,
        "trajectory_samples": trajectory_samples,
        "trajectory_times_ns": trajectory_times_ns,
        "branch_endpoint": branch_endpoint,
        "sentinel_audits": sentinel_audits,
        "render_batches": render_batches,
    }


def render_frame_identity(
    *,
    state_id: str,
    frame_kind: str,
    index: int,
) -> str:
    if _STATE_ID_RE.fullmatch(state_id) is None:
        raise PilotContractError("render frame state identity is invalid")
    if frame_kind not in {"context", "candidate", "sentinel"}:
        raise PilotContractError("render frame kind is invalid")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PilotContractError("render frame index is invalid")
    return f"{state_id}:{frame_kind}:{index}"


__all__ = [
    "ACTION_COUNT",
    "AUTHORITY_SOURCE_NAMES",
    "AUTHORITY_SOURCE_PATHS",
    "BLOCK_SIZE",
    "BRANCH_MECHANISM",
    "CANONICAL_ACTION_BLOCKS",
    "CANONICAL_ACTION_COMMANDS",
    "CANONICAL_ACTIONS",
    "CONTEXT_FRAME_COUNT",
    "FAMILIES",
    "FINAL_MANIFEST_SCHEMA",
    "HISTORY_BLOCK_COUNT",
    "CALIBRATION_LANES_PER_STATE",
    "PHYSICS_RESULT_SCHEMA",
    "PLAN_SCHEMA",
    "PilotContractError",
    "PilotDiagnosticError",
    "RENDER_CONTRACT",
    "EXECUTION_ENVIRONMENT",
    "GRAPHICS_PREFLIGHT_EXPECTATION",
    "PLATFORM_GATE_DISPOSITION",
    "RENDER_PLAN_INDEX_SCHEMA",
    "ROLES",
    "STATE_RECEIPT_SCHEMA",
    "SMOKE_AUTHORITY_SCHEMA",
    "SOURCE_REVIEW_SCHEMA",
    "SYNC_COMPONENTS",
    "audit_prebranch_synchronization",
    "audit_sentinel_trajectories",
    "canonical_block_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "deterministic_sentinel_action_id",
    "expected_counts_from_states",
    "execute_lockstep_trial",
    "file_binding",
    "fresh_development_output_root",
    "lane_layout",
    "lane_count_for_role",
    "read_bound_json",
    "render_frame_identity",
    "require_binding",
    "require_plan_bindings",
    "target_world_to_body_xy",
    "validate_plan",
    "validate_authority",
    "validate_source_review",
    "write_json_exclusive",
    "write_jsonl_exclusive",
]
