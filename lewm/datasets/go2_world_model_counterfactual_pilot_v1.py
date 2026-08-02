"""Strict development-pilot contract for physical counterfactual world models.

The consumer deliberately starts from one caller-pinned manifest.  It does not
fall back to historical counterfactual files, infer an RGB root, or accept
kinematic pose renders as physical outcomes. Receipt-only validation opens the
manifest and its bound collection, plan, state, calibration, role-index, and
RGB-manifest receipts, but never an RGB leaf.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = (REPO_ROOT / ".generated/dev").resolve()
PILOT_MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_manifest_v1"
RGB_MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_rgb_manifest_v1"
GROUP_SCHEMA = "lewm_go2_world_model_counterfactual_group_v1"
COLLECTION_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_physics_result_v1"
PLAN_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_plan_v1"
STATE_RECEIPT_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_state_receipt_v1"
CALIBRATION_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_receipt_v1"
)
PHYSICAL_RANK_CONTRACT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_physical_rank_contract_v1"
)
TOLERANCE_DERIVATION_SCHEMA = (
    "lewm_go2_world_model_counterfactual_tolerance_derivation_v1"
)
EVIDENCE_SCOPE = "physics_executed"
ROLE_NAMES = ("train", "eval")
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
ACTION_COUNT = len(PRIMITIVE_NAMES)
TOTAL_BRANCH_COUNT = ACTION_COUNT
COMMAND_TICKS_PER_BLOCK = 5
EXECUTED_TAPE_SHAPE = (5, 3)
CONTEXT_ENDPOINT_COMMAND_TICKS = (0, 5, 10)
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
    tuple(command for _ in range(COMMAND_TICKS_PER_BLOCK))
    for command in CANONICAL_ACTION_COMMANDS
)
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

_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_SHA_CHARS = frozenset("0123456789abcdef")


class CounterfactualPilotContractError(RuntimeError):
    """Raised before a malformed pilot can become model input."""


@dataclass(frozen=True)
class RGBArtifactV1:
    artifact_id: str
    frame_identity: str
    relative_path: str
    byte_count: int
    file_sha256: str


@dataclass(frozen=True)
class PhysicalLabelsV1:
    fell: bool
    tipped: bool
    target_progress_m: float
    path_length_m: float
    planar_clearance_proxy_min_m: float | None
    grid_recoverability_proxy: float | bool | None


@dataclass(frozen=True)
class CandidateActionInputV1:
    """Information available when a planner proposes a candidate action.

    The future executed tape is intentionally absent: it is an outcome/audit
    field and is not known before the safety/controller stack executes the
    requested primitive.
    """

    requested_action_id: int
    requested_primitive: str
    requested_block: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class CounterfactualBranchV1:
    lane_index: int
    lane_offset: int
    action_id: int
    action_name: str
    requested_block: tuple[tuple[float, float, float], ...]
    executed_command_tape: tuple[tuple[float, float, float], ...]
    executed_command_tape_sha256: str
    target_rgb_artifact_id: str
    labels: PhysicalLabelsV1
    oracle_dense_rank: int

    @property
    def requested_primitive(self) -> str:
        return self.action_name

    @property
    def model_input_action_id(self) -> int:
        return self.action_id


@dataclass(frozen=True)
class CounterfactualGroupV1:
    role: str
    state_id: str
    family: str
    scene_id: str
    group_index: int
    state_index_in_scene: int
    relative_target_xy_body_m: tuple[float, float]
    context_rgb_artifact_ids: tuple[str, str, str]
    history_action_ids: tuple[int, int]
    history_executed_blocks: tuple[
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
    ]
    branches: tuple[CounterfactualBranchV1, ...]

    @property
    def group_id(self) -> str:
        return self.state_id

    @property
    def state_index(self) -> int:
        return self.state_index_in_scene

    @property
    def historical_action_ids(self) -> tuple[int, int]:
        return self.history_action_ids

    @property
    def historical_executed_tapes(self) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
    ]:
        return self.history_executed_blocks


@dataclass(frozen=True)
class CounterfactualPilotBundleV1:
    root: Path
    manifest_binding: Mapping[str, object]
    manifest: Mapping[str, object]
    rgb_manifest_binding: Mapping[str, object]
    artifacts: Mapping[str, RGBArtifactV1]
    groups_by_role: Mapping[str, tuple[CounterfactualGroupV1, ...]]
    role_bindings: Mapping[str, Mapping[str, object]]
    calibration_receipt: Mapping[str, object]
    calibration_tolerances: Mapping[str, float]
    access_audit: Mapping[str, object]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA_CHARS for character in value)
    )


def _reject_constant(value: str) -> Any:
    raise CounterfactualPilotContractError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _no_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CounterfactualPilotContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CounterfactualPilotContractError(f"invalid {name} JSON") from exc


def canonical_json_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CounterfactualPilotContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CounterfactualPilotContractError(
            f"{name} must be a non-negative integer"
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CounterfactualPilotContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CounterfactualPilotContractError(f"{name} must be finite")
    return result


def _positive_finite(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise CounterfactualPilotContractError(f"{name} must be positive")
    return result


def _canonical_relative(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise CounterfactualPilotContractError(f"{name} must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise CounterfactualPilotContractError(
            f"{name} must be one canonical relative POSIX path"
        )
    return path


def _safe_path(root: Path, relative: PurePosixPath, *, final_file: bool) -> Path:
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        if current.is_symlink():
            raise CounterfactualPilotContractError(
                f"pilot path crosses a symlink: {relative.as_posix()}"
            )
        if index < len(relative.parts) - 1 and not current.is_dir():
            raise CounterfactualPilotContractError(
                f"pilot path has a non-directory component: {relative.as_posix()}"
            )
    resolved = current.resolve()
    if not resolved.is_relative_to(root):
        raise CounterfactualPilotContractError("pilot path escapes its bound root")
    if final_file and not resolved.is_file():
        raise CounterfactualPilotContractError(
            f"pilot artifact is not a regular file: {relative.as_posix()}"
        )
    return resolved


def _read_bound_file(root: Path, binding: Mapping[str, object], *, name: str) -> tuple[bytes, dict]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise CounterfactualPilotContractError(f"{name} binding is malformed")
    relative = _canonical_relative(binding["path"], name=f"{name} path")
    expected_bytes = _positive_int(binding["byte_count"], name=f"{name} byte_count")
    expected_sha = binding["file_sha256"]
    if not _is_sha256(expected_sha):
        raise CounterfactualPilotContractError(f"{name} SHA-256 is malformed")
    descriptor = os.open(root, _DIR_FLAGS)
    file_descriptor = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            relative.parts[-1], _READ_FLAGS, dir_fd=descriptor
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CounterfactualPilotContractError(f"{name} is not a regular file")
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(file_descriptor)
    except OSError as exc:
        raise CounterfactualPilotContractError(
            f"cannot safely open {name}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(raw) != expected_bytes
        or before.st_size != expected_bytes
        or digest.hexdigest() != expected_sha
    ):
        raise CounterfactualPilotContractError(f"{name} bytes or identity changed")
    return raw, {
        "path": relative.as_posix(),
        "file_sha256": expected_sha,
        "byte_count": expected_bytes,
    }


def _validate_tape(value: object, *, name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list) or len(value) != EXECUTED_TAPE_SHAPE[0]:
        raise CounterfactualPilotContractError(f"{name} must have shape (5,3)")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != EXECUTED_TAPE_SHAPE[1]:
            raise CounterfactualPilotContractError(f"{name} must have shape (5,3)")
        rows.append(tuple(
            _finite(item, name=f"{name}[{row_index}]") for item in row
        ))
    return tuple(rows)  # type: ignore[return-value]


def _quantized(value: float, tolerance: float) -> int:
    scaled = abs(float(value)) / float(tolerance)
    magnitude = math.floor(scaled + 0.5)
    return magnitude if value >= 0.0 else -magnitude


def _target_world_to_body_xy(
    *,
    target_xy_m: Sequence[float],
    position_xyz_m: Sequence[float],
    quaternion_wxyz: Sequence[float],
) -> tuple[float, float]:
    qw, qx, qy, qz = quaternion_wxyz
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise CounterfactualPilotContractError(
            "prebranch base quaternion is not normalized"
        )
    qw, qx, qy, qz = (component / norm for component in (qw, qx, qy, qz))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    dx = target_xy_m[0] - position_xyz_m[0]
    dy = target_xy_m[1] - position_xyz_m[1]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
    )


def physical_oracle_key_v1(
    labels: PhysicalLabelsV1,
    tolerances: Mapping[str, float],
) -> tuple[int, ...]:
    """Return the calibration-tolerance-aware physical action ordering."""
    return (
        int(labels.fell),
        int(labels.tipped),
        -_quantized(labels.target_progress_m, tolerances["progress_tolerance_m"]),
        _quantized(labels.path_length_m, tolerances["path_length_tolerance_m"]),
    )


def physical_dense_ranks_v1(
    labels: Sequence[PhysicalLabelsV1],
    tolerances: Mapping[str, float],
) -> tuple[int, ...]:
    keys = [physical_oracle_key_v1(item, tolerances) for item in labels]
    unique = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return tuple(unique[key] for key in keys)


def candidate_model_inputs_v1(
    group: CounterfactualGroupV1,
) -> tuple[CandidateActionInputV1, ...]:
    """Return the nine pre-execution candidate inputs with no future leakage."""

    if len(group.branches) != ACTION_COUNT:
        raise CounterfactualPilotContractError(
            "candidate model input requires exactly nine branches"
        )
    result = tuple(
        CandidateActionInputV1(
            requested_action_id=branch.action_id,
            requested_primitive=branch.requested_primitive,
            requested_block=branch.requested_block,
        )
        for branch in group.branches
    )
    if tuple(item.requested_action_id for item in result) != tuple(range(ACTION_COUNT)):
        raise CounterfactualPilotContractError(
            "candidate model-input action order changed"
        )
    return result


def _physical_labels(value: object, *, name: str) -> PhysicalLabelsV1:
    required = {
        "physical_fell",
        "physical_tipped",
        "physical_target_progress_m",
        "physical_path_length_m",
    }
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CounterfactualPilotContractError(f"{name} physical labels changed")
    optional = set(value) - required
    if (
        not required.issubset(value)
        or any(
            not (
                key.startswith("planar_clearance_proxy_")
                or key.startswith("grid_recoverability_proxy")
            )
            for key in optional
        )
    ):
        raise CounterfactualPilotContractError(f"{name} physical labels changed")
    for field in ("physical_fell", "physical_tipped"):
        if type(value[field]) is not bool:
            raise CounterfactualPilotContractError(f"{name}.{field} must be boolean")
    for field in optional:
        if not isinstance(value[field], bool):
            _finite(value[field], name=f"{name}.{field}")
    path_length = _finite(
        value["physical_path_length_m"], name=f"{name}.path_length"
    )
    if path_length < 0.0:
        raise CounterfactualPilotContractError(
            f"{name}.physical_path_length_m must be non-negative"
        )
    return PhysicalLabelsV1(
        fell=value["physical_fell"],
        tipped=value["physical_tipped"],
        target_progress_m=_finite(
            value["physical_target_progress_m"], name=f"{name}.progress"
        ),
        path_length_m=path_length,
        planar_clearance_proxy_min_m=(
            _finite(value["planar_clearance_proxy_min_m"], name=f"{name}.clearance_proxy")
            if "planar_clearance_proxy_min_m" in value
            else None
        ),
        grid_recoverability_proxy=(
            (
                value["grid_recoverability_proxy"]
                if isinstance(value["grid_recoverability_proxy"], bool)
                else _finite(
                    value["grid_recoverability_proxy"],
                    name=f"{name}.recoverability_proxy",
                )
            )
            if "grid_recoverability_proxy" in value
            else None
        ),
    )


def _parse_group(
    value: object,
    *,
    role: str,
    artifacts: Mapping[str, RGBArtifactV1],
    tolerances: Mapping[str, float],
    requested_blocks: Sequence[tuple[tuple[float, float, float], ...]],
    collection_state: Mapping[str, object],
) -> CounterfactualGroupV1:
    expected = {
        "schema",
        "role",
        "state_id",
        "family",
        "scene_id",
        "group_index",
        "state_index_in_scene",
        "task",
        "context",
        "synchronization_audit",
        "branches",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CounterfactualPilotContractError("pilot group schema fields changed")
    if value["schema"] != GROUP_SCHEMA or value["role"] != role:
        raise CounterfactualPilotContractError("pilot group schema or role changed")
    for field in ("state_id", "family", "scene_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise CounterfactualPilotContractError(f"pilot group {field} is invalid")
    group_index = _nonnegative_int(value["group_index"], name="group_index")
    state_index = _nonnegative_int(
        value["state_index_in_scene"], name="state_index_in_scene"
    )
    receipt_state = collection_state.get("state")
    if not isinstance(receipt_state, Mapping) or any(
        value[key] != receipt_state.get(key)
        for key in (
            "role",
            "state_id",
            "family",
            "scene_id",
            "group_index",
            "state_index_in_scene",
        )
    ):
        raise CounterfactualPilotContractError(
            "joined group identity changed from the physics collection"
        )

    task = value["task"]
    if (
        not isinstance(task, Mapping)
        or set(task) != {"target_present", "relative_target_xy_body_m"}
        or task["target_present"] is not True
        or not isinstance(task["relative_target_xy_body_m"], list)
        or len(task["relative_target_xy_body_m"]) != 2
    ):
        raise CounterfactualPilotContractError("WM-A.4 task contract is invalid")
    target_xy = tuple(
        _finite(item, name="relative target")
        for item in task["relative_target_xy_body_m"]
    )

    context = value["context"]
    if not isinstance(context, Mapping) or set(context) != {
        "rgb_artifact_ids",
        "frame_identities",
        "history_action_ids",
        "history_executed_blocks",
        "executed_block_sha256s",
        "endpoint_command_ticks",
        "prebranch_state_sha256",
    }:
        raise CounterfactualPilotContractError("pilot context contract changed")
    rgb_ids = context["rgb_artifact_ids"]
    actions = context["history_action_ids"]
    tapes = context["history_executed_blocks"]
    frame_identities = context["frame_identities"]
    tape_hashes_declared = context["executed_block_sha256s"]
    if (
        not isinstance(rgb_ids, list)
        or len(rgb_ids) != 3
        or any(not isinstance(item, str) or item not in artifacts for item in rgb_ids)
        or len(set(rgb_ids)) != 3
        or not isinstance(frame_identities, list)
        or len(frame_identities) != 3
        or [artifacts[item].frame_identity for item in rgb_ids] != frame_identities
        or not isinstance(actions, list)
        or len(actions) != 2
        or any(type(item) is not int or not 0 <= item < ACTION_COUNT for item in actions)
        or not isinstance(tapes, list)
        or len(tapes) != 2
        or not isinstance(tape_hashes_declared, list)
        or len(tape_hashes_declared) != 2
        or context["endpoint_command_ticks"] != list(CONTEXT_ENDPOINT_COMMAND_TICKS)
        or not _is_sha256(context["prebranch_state_sha256"])
    ):
        raise CounterfactualPilotContractError("pilot context identities are invalid")
    history_blocks = tuple(
        _validate_tape(tape, name=f"history block {index}")
        for index, tape in enumerate(tapes)
    )
    if tape_hashes_declared != [
        canonical_json_sha256([list(row) for row in block])
        for block in history_blocks
    ]:
        raise CounterfactualPilotContractError("history executed-block hashes changed")
    collection_context = collection_state.get("context")
    collection_context_fields = set(context) | {
        "prebranch_base_pose_world",
        "context_base_pose_world_sequence",
        "target_relative_body_xy_m",
    }
    if (
        not isinstance(collection_context, Mapping)
        or set(collection_context) != collection_context_fields
        or any(collection_context.get(key) != item for key, item in context.items())
    ):
        raise CounterfactualPilotContractError(
            "joined history/context changed from the physics collection"
        )
    prebranch_pose = collection_context["prebranch_base_pose_world"]
    context_pose_sequence = collection_context["context_base_pose_world_sequence"]
    receipt_relative_target = collection_context["target_relative_body_xy_m"]
    receipt_state_target = receipt_state.get("target_xy_m")
    if (
        not isinstance(prebranch_pose, Mapping)
        or set(prebranch_pose) != {"position_xyz_m", "quaternion_wxyz"}
        or not isinstance(prebranch_pose["position_xyz_m"], list)
        or len(prebranch_pose["position_xyz_m"]) != 3
        or not isinstance(prebranch_pose["quaternion_wxyz"], list)
        or len(prebranch_pose["quaternion_wxyz"]) != 4
        or not isinstance(receipt_relative_target, list)
        or len(receipt_relative_target) != 2
        or not isinstance(receipt_state_target, list)
        or len(receipt_state_target) != 2
        or not isinstance(context_pose_sequence, list)
        or len(context_pose_sequence) != 3
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"position_xyz_m", "quaternion_wxyz"}
            for item in context_pose_sequence
        )
        or context_pose_sequence[-1] != prebranch_pose
    ):
        raise CounterfactualPilotContractError(
            "prebranch pose or body-frame target receipt changed"
        )
    position = tuple(
        _finite(item, name="prebranch position")
        for item in prebranch_pose["position_xyz_m"]
    )
    quaternion = tuple(
        _finite(item, name="prebranch quaternion")
        for item in prebranch_pose["quaternion_wxyz"]
    )
    world_target = tuple(
        _finite(item, name="world target") for item in receipt_state_target
    )
    declared_relative = tuple(
        _finite(item, name="receipt body-frame target")
        for item in receipt_relative_target
    )
    recomputed_relative = _target_world_to_body_xy(
        target_xy_m=world_target,
        position_xyz_m=position,
        quaternion_wxyz=quaternion,
    )
    if any(
        not math.isclose(declared, recomputed, rel_tol=0.0, abs_tol=1e-12)
        for declared, recomputed in zip(
            declared_relative, recomputed_relative, strict=True
        )
    ) or any(
        not math.isclose(joined, declared, rel_tol=0.0, abs_tol=1e-12)
        for joined, declared in zip(target_xy, declared_relative, strict=True)
    ):
        raise CounterfactualPilotContractError(
            "joined task target is not the verified prebranch body-frame target"
        )

    prefix = value["synchronization_audit"]
    if prefix != collection_state.get("synchronization_audit"):
        raise CounterfactualPilotContractError(
            "joined synchronization audit changed from the physics collection"
        )
    expected_sync_fields = {
        "state_id",
        "group_index",
        "lane_start",
        "lane_count",
        "exact_equality_required",
        "passed",
        "prebranch_state_sha256",
        "lane_state_sha256s",
        "components",
    }
    if (
        not isinstance(prefix, Mapping)
        or set(prefix) != expected_sync_fields
        or prefix["state_id"] != value["state_id"]
        or prefix["group_index"] != group_index
        or type(prefix["lane_start"]) is not int
        or prefix["lane_start"] < 0
        or prefix["lane_count"] != TOTAL_BRANCH_COUNT
        or prefix["exact_equality_required"] is not True
        or prefix["passed"] is not True
        or prefix["prebranch_state_sha256"] != context["prebranch_state_sha256"]
        or not isinstance(prefix["lane_state_sha256s"], list)
        or len(prefix["lane_state_sha256s"]) != TOTAL_BRANCH_COUNT
        or prefix["lane_state_sha256s"]
        != [prefix["prebranch_state_sha256"]] * TOTAL_BRANCH_COUNT
        or not isinstance(prefix["components"], Mapping)
        or set(prefix["components"]) != set(SYNC_COMPONENTS)
    ):
        raise CounterfactualPilotContractError(
            "common-prefix physical-state equality is absent"
        )
    for component_name, component in prefix["components"].items():
        if (
            not isinstance(component, Mapping)
            or set(component) != {
                "exact_equal",
                "max_abs_difference",
                "shape_per_lane",
            }
            or component["exact_equal"] is not True
            or _finite(
                component["max_abs_difference"],
                name=f"sync {component_name} maximum difference",
            )
            != 0.0
            or not isinstance(component["shape_per_lane"], list)
        ):
            raise CounterfactualPilotContractError(
                "common-prefix component equality is absent"
            )

    raw_branches = value["branches"]
    if not isinstance(raw_branches, list) or len(raw_branches) != ACTION_COUNT:
        raise CounterfactualPilotContractError("pilot group must contain nine candidates")
    branches = []
    receipt_branches = collection_state.get("branches")
    if not isinstance(receipt_branches, list) or len(receipt_branches) != ACTION_COUNT:
        raise CounterfactualPilotContractError(
            "physics collection candidate grid changed"
        )
    tape_hashes = set()
    target_ids = set()
    for slot, branch in enumerate(raw_branches):
        branch_required = {
            "lane_index",
            "lane_offset",
            "kind",
            "action_id",
            "action_name",
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
        }
        branch_optional = {
            "declared_oracle_dense_rank",
            "duplicates_candidate_action_id",
        }
        if (
            not isinstance(branch, Mapping)
            or not branch_required.issubset(branch)
            or any(
                key not in branch_required | branch_optional
                and not key.startswith("planar_clearance_proxy_")
                and not key.startswith("grid_recoverability_proxy")
                for key in branch
            )
        ):
            raise CounterfactualPilotContractError("pilot branch fields changed")
        receipt_branch = receipt_branches[slot]
        if not isinstance(receipt_branch, Mapping):
            raise CounterfactualPilotContractError("physics collection branch is malformed")
        normalized_receipt_branch = {
            key: item
            for key, item in receipt_branch.items()
            if not (key == "duplicates_candidate_action_id" and item is None)
        }
        if (
            set(branch) != set(normalized_receipt_branch) | {"declared_oracle_dense_rank"}
            or any(branch.get(key) != item for key, item in normalized_receipt_branch.items())
        ):
            raise CounterfactualPilotContractError(
                "joined branch changed from the physics collection"
            )
        expected_action = slot
        if (
            branch["lane_index"] != prefix["lane_start"] + slot
            or branch["lane_offset"] != slot
            or branch["kind"] != "candidate"
            or type(expected_action) is not int
            or not 0 <= expected_action < ACTION_COUNT
            or branch["action_id"] != expected_action
            or branch["action_name"] != PRIMITIVE_NAMES[expected_action]
            or type(branch["clipped"]) is not bool
            or not isinstance(branch["trajectory_policy_step_samples"], list)
            or not branch["trajectory_policy_step_samples"]
            or not isinstance(branch["endpoint_state"], Mapping)
            or not branch["endpoint_state"]
        ):
            raise CounterfactualPilotContractError(
                "pilot branch action or physical-validity contract changed"
            )
        if (
            "duplicates_candidate_action_id" in branch
            or "declared_oracle_dense_rank" not in branch
        ):
            raise CounterfactualPilotContractError("candidate branch fields changed")
        requested = _validate_tape(branch["requested_block"], name="requested block")
        if requested != requested_blocks[expected_action]:
            raise CounterfactualPilotContractError("requested action block changed")
        tape = _validate_tape(branch["executed_block"], name="executed block")
        tape_digest = canonical_json_sha256([list(row) for row in tape])
        if branch["executed_block_sha256"] != tape_digest:
            raise CounterfactualPilotContractError("executed command tape hash changed")
        receipt = branch["frame_receipt"]
        receipt_fields = {
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
        }
        if not isinstance(receipt, Mapping) or set(receipt) != receipt_fields:
            raise CounterfactualPilotContractError("branch frame receipt changed")
        target_id = receipt["artifact_id"]
        artifact = artifacts.get(target_id) if isinstance(target_id, str) else None
        if (
            artifact is None
            or receipt["frame_identity"] != branch["render_frame_identity"]
            or receipt["frame_identity"] != artifact.frame_identity
            or receipt["path"] != artifact.relative_path
            or receipt["file_sha256"] != artifact.file_sha256
            or receipt["byte_count"] != artifact.byte_count
            or receipt["width"] != 224
            or receipt["height"] != 224
            or receipt["mode"] != "RGB"
            or receipt["format"] != "PNG"
            or receipt["camera_valid"] is not True
        ):
            raise CounterfactualPilotContractError("branch target RGB is unbound")
        label_fields = {
            key: branch[key]
            for key in branch
            if key in {
                "physical_fell",
                "physical_tipped",
                "physical_path_length_m",
                "physical_target_progress_m",
            }
            or key.startswith("planar_clearance_proxy_")
            or key.startswith("grid_recoverability_proxy")
        }
        parsed_labels = _physical_labels(label_fields, name=f"branch {slot}")
        tape_hashes.add(tape_digest)
        target_ids.add(target_id)
        branches.append(CounterfactualBranchV1(
            lane_index=branch["lane_index"],
            lane_offset=slot,
            action_id=slot,
            action_name=PRIMITIVE_NAMES[slot],
            requested_block=requested,
            executed_command_tape=tape,
            executed_command_tape_sha256=tape_digest,
            target_rgb_artifact_id=target_id,
            labels=parsed_labels,
            oracle_dense_rank=_nonnegative_int(
                branch["declared_oracle_dense_rank"], name="oracle dense rank"
            ),
        ))
    if len(tape_hashes) != ACTION_COUNT:
        raise CounterfactualPilotContractError(
            "requested actions collapse to duplicate executed command tapes"
        )
    if len(target_ids) != ACTION_COUNT:
        raise CounterfactualPilotContractError("physical target RGB identities repeat")
    recomputed = physical_dense_ranks_v1(
        [branch.labels for branch in branches], tolerances
    )
    if recomputed != tuple(branch.oracle_dense_rank for branch in branches):
        raise CounterfactualPilotContractError(
            "declared physical oracle ranks disagree with the bound labels"
        )
    return CounterfactualGroupV1(
        role=role,
        state_id=value["state_id"],
        family=value["family"],
        scene_id=value["scene_id"],
        group_index=group_index,
        state_index_in_scene=state_index,
        relative_target_xy_body_m=(target_xy[0], target_xy[1]),
        context_rgb_artifact_ids=(rgb_ids[0], rgb_ids[1], rgb_ids[2]),
        history_action_ids=(actions[0], actions[1]),
        history_executed_blocks=(history_blocks[0], history_blocks[1]),
        branches=tuple(branches),
    )


def _manifest_binding_from_caller(
    root: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[bytes, dict]:
    if type(expected_byte_count) is not int or expected_byte_count <= 0:
        raise CounterfactualPilotContractError(
            "expected pilot manifest byte count must be positive"
        )
    if not _is_sha256(expected_sha256):
        raise CounterfactualPilotContractError(
            "expected pilot manifest SHA-256 must be lowercase hex"
        )
    return _read_bound_file(
        root,
        {
            "path": "manifest.json",
            "file_sha256": expected_sha256,
            "byte_count": expected_byte_count,
        },
        name="pilot manifest",
    )


def _validate_inert_binding(
    value: object,
    *,
    name: str,
    synthetic_test_mode: bool,
    require_absolute: bool = False,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise CounterfactualPilotContractError(f"{name} binding is malformed")
    path = value["path"]
    if not isinstance(path, str) or not path:
        raise CounterfactualPilotContractError(f"{name} path is malformed")
    selected = Path(path)
    if require_absolute:
        if not selected.is_absolute() or str(selected) != path:
            raise CounterfactualPilotContractError(
                f"{name} path must be canonical absolute"
            )
    else:
        if selected.is_absolute():
            if str(selected) != path:
                raise CounterfactualPilotContractError(
                    f"{name} absolute path is not canonical"
                )
        else:
            _canonical_relative(path, name=f"{name} path")
    if not synthetic_test_mode and selected.is_absolute() and selected.parts[:2] == (
        "/",
        "synthetic",
    ):
        raise CounterfactualPilotContractError(
            f"{name} uses synthetic provenance outside explicit test mode"
        )
    byte_count = _positive_int(value["byte_count"], name=f"{name} byte_count")
    if not _is_sha256(value["file_sha256"]):
        raise CounterfactualPilotContractError(f"{name} SHA-256 is malformed")
    return {
        "path": path,
        "file_sha256": value["file_sha256"],
        "byte_count": byte_count,
    }


def _validate_source_bindings(
    value: object, *, synthetic_test_mode: bool
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise CounterfactualPilotContractError("collector source bindings are absent")
    result = []
    seen_names = set()
    seen_paths = set()
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "binding"}:
            raise CounterfactualPilotContractError("collector source binding is malformed")
        source_name = entry["name"]
        if not isinstance(source_name, str) or not source_name or source_name in seen_names:
            raise CounterfactualPilotContractError(
                "collector source binding name repeats or is invalid"
            )
        binding = _validate_inert_binding(
            entry["binding"],
            name=f"collector source {source_name}",
            synthetic_test_mode=synthetic_test_mode,
            require_absolute=True,
        )
        source_path = binding["path"]
        if source_path in seen_paths:
            raise CounterfactualPilotContractError("collector source binding path repeats")
        seen_names.add(source_name)
        seen_paths.add(source_path)
        result.append({"name": source_name, "binding": binding})
    if not {"collector", "contract", "checker"}.issubset(seen_names):
        raise CounterfactualPilotContractError(
            "collector source closure omits collector, contract, or checker"
        )
    return tuple(result)


def _validate_calibration_contract(value: object) -> tuple[tuple[str, ...], Mapping[str, float]]:
    expected = {
        "schema",
        "excluded_scene_ids",
        "progress_tolerance_m",
        "path_length_tolerance_m",
        "quantization_rule",
        "lexicographic_key",
        "proxy_fields_excluded",
        "tolerance_derivation",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CounterfactualPilotContractError("calibration contract fields changed")
    if (
        value["schema"] != PHYSICAL_RANK_CONTRACT_SCHEMA
        or value["quantization_rule"] != "sign(x)*floor(abs(x)/t+0.5)"
        or value["lexicographic_key"] != [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ]
        or value["proxy_fields_excluded"] is not True
    ):
        raise CounterfactualPilotContractError("physical rank calibration changed")
    excluded = value["excluded_scene_ids"]
    if (
        not isinstance(excluded, list)
        or not excluded
        or excluded != sorted(excluded)
        or len(set(excluded)) != len(excluded)
        or any(not isinstance(scene, str) or not scene for scene in excluded)
    ):
        raise CounterfactualPilotContractError(
            "calibration scene exclusion contract is invalid"
        )
    derivation = value["tolerance_derivation"]
    if not isinstance(derivation, Mapping) or set(derivation) != {
        "schema",
        "method",
        "minimum_numerical_resolution_m",
        "repeat_controls",
        "repeated_action_ids",
        "all_requested_primitives_covered",
        "deterministic_repeat_gate_passed",
        "empirical_noise_scale_estimated",
    }:
        raise CounterfactualPilotContractError(
            "calibration tolerance derivation is absent"
        )
    numerical_floor = _positive_finite(
        derivation["minimum_numerical_resolution_m"],
        name="minimum_numerical_resolution_m",
    )
    if (
        derivation["schema"] != TOLERANCE_DERIVATION_SCHEMA
        or derivation["method"]
        != "fixed_numerical_floor_after_exact_deterministic_repeat_gate"
        or numerical_floor != 1.0e-6
        or derivation["repeat_controls"] != 16
        or derivation["repeated_action_ids"] != [index % ACTION_COUNT for index in range(16)]
        or derivation["all_requested_primitives_covered"] is not True
        or derivation["deterministic_repeat_gate_passed"] is not True
        or derivation["empirical_noise_scale_estimated"] is not False
        or value["progress_tolerance_m"] != numerical_floor
        or value["path_length_tolerance_m"] != numerical_floor
    ):
        raise CounterfactualPilotContractError(
            "physical-rank numerical floor or deterministic repeat gate changed"
        )
    tolerances = MappingProxyType({
        "progress_tolerance_m": _positive_finite(
            value["progress_tolerance_m"], name="progress_tolerance_m"
        ),
        "path_length_tolerance_m": _positive_finite(
            value["path_length_tolerance_m"], name="path_length_tolerance_m"
        ),
    })
    return tuple(excluded), tolerances


def _load_calibration_receipt(
    root: Path,
    binding: object,
    *,
    attempt_id: str,
    top_contract: object,
    synthetic_test_mode: bool,
) -> Mapping[str, object]:
    raw, normalized_binding = _read_bound_file(
        root, binding, name="calibration receipt"
    )
    receipt = strict_json_loads(raw, name="calibration receipt")
    expected = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "calibration_id",
        "role",
        "train_eval_scenes_accessed",
        "decision",
        "calibration_collection_receipt",
        "calibration_contract",
        "repeatability_analysis",
        "physics_validation",
        "visual_validation",
        "resource_measurements",
        "analyzer_binding",
        "checker_binding",
        "source_bindings",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise CounterfactualPilotContractError("calibration receipt fields changed")
    expected_status = "SYNTHETIC_TEST_ONLY" if synthetic_test_mode else "COMPLETE"
    if (
        receipt["schema"] != CALIBRATION_RECEIPT_SCHEMA
        or receipt["status"] != expected_status
        or receipt["citable_as_scientific_evidence"] is not False
        or receipt["authorizes_retry_or_resume"] is not False
        or not isinstance(receipt["calibration_id"], str)
        or not receipt["calibration_id"]
        or receipt["role"] != "calibration"
        or receipt["train_eval_scenes_accessed"] is not False
        or receipt["decision"] != "FREEZE_PILOT_CONTRACT"
        or receipt["calibration_contract"] != top_contract
    ):
        raise CounterfactualPilotContractError(
            "calibration receipt status, identity, or contract changed"
        )
    _validate_inert_binding(
        receipt["calibration_collection_receipt"],
        name="calibration collection receipt",
        synthetic_test_mode=synthetic_test_mode,
        require_absolute=True,
    )
    for field, required_values in (
        (
            "repeatability_analysis",
            {
                "repeat_controls": 16,
                "repeated_action_ids": [index % ACTION_COUNT for index in range(16)],
                "all_requested_primitives_covered": True,
                "interpretation": (
                    "deterministic_replay_gate_not_empirical_noise_estimate"
                ),
                "empirical_noise_scale_estimated": False,
                "executed_command_tapes_exact": True,
                "physical_trajectories_exact": True,
                "stored_rgb_exact": True,
            },
        ),
        (
            "physics_validation",
            {
                "receipt_checker_passed": True,
                "common_prefix_exact": True,
                "nine_unique_executed_tapes_per_state": True,
                "physics_validated_for_branch_outcomes": True,
            },
        ),
        (
            "visual_validation",
            {
                "camera_quality_receipts_passed": True,
                "endpoint_pose_replay_bound": True,
                "visual_domain_fidelity_claimed": False,
                "eligible_for_physical_branch_evaluation": True,
                "eligible_for_visual_domain_parity_claim": False,
            },
        ),
    ):
        value = receipt[field]
        if not isinstance(value, Mapping) or any(
            value.get(key) != expected_value
            for key, expected_value in required_values.items()
        ):
            raise CounterfactualPilotContractError(
                f"calibration {field} evidence changed"
            )
    resources = receipt["resource_measurements"]
    if (
        not isinstance(resources, Mapping)
        or set(resources) != {
            "schema",
            "stored_rgb_png",
            "stage_wall_seconds",
            "outcome_counts",
            "gpu_peak_memory_measurement_scope",
        }
        or resources["schema"]
        != "lewm_go2_world_model_counterfactual_calibration_resource_measurements_v1"
        or resources["gpu_peak_memory_measurement_scope"]
        != "external_terminal_required_not_observed_by_analyzer"
    ):
        raise CounterfactualPilotContractError(
            "calibration resource measurements changed"
        )
    stored_rgb = resources["stored_rgb_png"]
    stages = resources["stage_wall_seconds"]
    outcomes = resources["outcome_counts"]
    if (
        not isinstance(stored_rgb, Mapping)
        or stored_rgb.get("context_frames") != 48
        or stored_rgb.get("target_frames") != 160
        or stored_rgb.get("total_frames") != 208
        or stored_rgb.get("raw_uncompressed_rgb_ceiling_bytes")
        != 208 * 224 * 224 * 3
        or type(stored_rgb.get("context_bytes")) is not int
        or type(stored_rgb.get("target_bytes")) is not int
        or type(stored_rgb.get("total_bytes")) is not int
        or stored_rgb["context_bytes"] < 0
        or stored_rgb["target_bytes"] < 0
        or stored_rgb["total_bytes"]
        != stored_rgb["context_bytes"] + stored_rgb["target_bytes"]
        or not isinstance(stages, Mapping)
        or any(_finite(value, name=f"calibration stage {name}") < 0.0 for name, value in stages.items())
        or not isinstance(outcomes, Mapping)
        or outcomes.get("complete_all_nine_action_groups") != 16
        or outcomes.get("executed_tape_distinct_groups") != 16
        or outcomes.get("prebranch_exact_groups") != 16
        or outcomes.get("camera_invalid_frames") != 0
        or outcomes.get("incomplete_states") != 0
    ):
        raise CounterfactualPilotContractError(
            "calibration resource measurement evidence is invalid"
        )
    analyzer_binding = _validate_inert_binding(
        receipt["analyzer_binding"],
        name="calibration analyzer",
        synthetic_test_mode=synthetic_test_mode,
        require_absolute=True,
    )
    checker_binding = _validate_inert_binding(
        receipt["checker_binding"],
        name="calibration checker",
        synthetic_test_mode=synthetic_test_mode,
        require_absolute=True,
    )
    sources = receipt["source_bindings"]
    expected_names = ["checker", "calibration_analyzer", "pilot_joiner"]
    if (
        not isinstance(sources, list)
        or [
            entry.get("name") if isinstance(entry, Mapping) else None
            for entry in sources
        ]
        != expected_names
    ):
        raise CounterfactualPilotContractError(
            "deterministic calibration analyzer source closure changed"
        )
    normalized_sources = []
    for entry in sources:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "binding"}:
            raise CounterfactualPilotContractError(
                "calibration analyzer source binding is malformed"
            )
        normalized_sources.append(_validate_inert_binding(
            entry["binding"],
            name=f"calibration source {entry['name']}",
            synthetic_test_mode=synthetic_test_mode,
            require_absolute=True,
        ))
    if normalized_sources[0] != checker_binding or normalized_sources[1] != analyzer_binding:
        raise CounterfactualPilotContractError(
            "calibration analyzer/checker aliases changed"
        )
    return MappingProxyType({
        "binding": MappingProxyType(normalized_binding),
        "document": MappingProxyType(dict(receipt)),
    })


def _load_collection_receipt(
    root: Path,
    binding: object,
    *,
    attempt_id: str,
    top_action_catalog: object,
    top_source_bindings: Sequence[Mapping[str, object]],
    synthetic_test_mode: bool,
) -> tuple[Mapping[str, Mapping[str, object]], int]:
    raw, _ = _read_bound_file(root, binding, name="collection receipt")
    collection = strict_json_loads(raw, name="collection receipt")
    expected = {
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
        "runtime_versions",
        "runtime_bindings",
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
    }
    if not isinstance(collection, Mapping) or set(collection) != expected:
        raise CounterfactualPilotContractError("collection receipt fields changed")
    if (
        collection["schema"] != COLLECTION_SCHEMA
        or collection["attempt_id"] != attempt_id
        or collection["purpose"] != "bounded_wm_a_pilot"
        or collection["status"] != "PHYSICS_COMPLETE"
        or collection["physics_validated"] is not False
        or collection["citable_as_scientific_evidence"] is not False
        or collection["authorizes_retry_or_resume"] is not False
        or collection["allows_refill"] is not False
        or collection["allows_overwrite"] is not False
        or collection["branch_mechanism"] != "parallel_lockstep_envs_no_restore"
        or collection["failure"] not in (None, {})
        or collection["expected_counts"] != collection["observed_counts"]
        or not isinstance(collection["visual_domain_limitation"], str)
        or not collection["visual_domain_limitation"]
        or _finite(
            collection["collection_wall_seconds"], name="collection wall seconds"
        )
        < 0.0
    ):
        raise CounterfactualPilotContractError(
            "collection status, purpose, or one-shot boundary changed"
        )
    for name in ("authority", "review", "reservation"):
        _validate_inert_binding(
            collection[f"{name}_binding"],
            name=f"collection {name}",
            synthetic_test_mode=synthetic_test_mode,
        )
    collection_sources = _validate_source_bindings(
        collection["source_bindings"], synthetic_test_mode=synthetic_test_mode
    )
    if tuple(top_source_bindings) != collection_sources:
        raise CounterfactualPilotContractError(
            "top-level source bindings changed from the collection"
        )
    execution = collection["execution_contract"]
    if (
        not isinstance(execution, Mapping)
        or execution.get("policy_steps_per_command_tick") != 5
        or type(execution.get("seed")) is not int
        or not isinstance(execution.get("backend"), str)
        or not execution.get("backend")
        or not isinstance(execution.get("policy_device"), str)
        or not execution.get("policy_device")
        or _positive_finite(
            execution.get("fall_z_threshold_m"), name="fall_z_threshold_m"
        )
        <= 0.0
        or _positive_finite(
            execution.get("tip_threshold_rad"), name="tip_threshold_rad"
        )
        <= 0.0
    ):
        raise CounterfactualPilotContractError("collection execution contract changed")
    runtime_versions = collection["runtime_versions"]
    if (
        not isinstance(runtime_versions, Mapping)
        or set(runtime_versions) != {"python", "genesis", "torch", "numpy", "pillow"}
        or any(
            not isinstance(version, str)
            or not version
            or version != version.strip()
            or any(ord(character) < 32 or ord(character) > 126 for character in version)
            for version in runtime_versions.values()
        )
    ):
        raise CounterfactualPilotContractError(
            "collection runtime version receipt changed"
        )
    runtime = collection["runtime_bindings"]
    required_runtime = {
        "platform_manifest",
        "primitive_registry",
        "policy_checkpoint",
        "policy_config",
        "go2_urdf",
    }
    if not isinstance(runtime, Mapping) or not required_runtime.issubset(runtime):
        raise CounterfactualPilotContractError("collection runtime bindings are incomplete")
    for name, runtime_binding in runtime.items():
        _validate_inert_binding(
            runtime_binding,
            name=f"runtime {name}",
            synthetic_test_mode=synthetic_test_mode,
            require_absolute=True,
        )

    external_plan_binding = _validate_inert_binding(
        collection["plan_binding"],
        name="external authorized pilot plan",
        synthetic_test_mode=synthetic_test_mode,
        require_absolute=True,
    )
    plan_raw, local_plan_binding = _read_bound_file(
        root, collection["plan_receipt_binding"], name="local authorized pilot plan receipt"
    )
    if (
        local_plan_binding["file_sha256"] != external_plan_binding["file_sha256"]
        or local_plan_binding["byte_count"] != external_plan_binding["byte_count"]
    ):
        raise CounterfactualPilotContractError(
            "local plan receipt differs from the external authorized plan binding"
        )
    plan = strict_json_loads(plan_raw, name="pilot plan")
    plan_fields = {
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
    if (
        not isinstance(plan, Mapping)
        or set(plan) != plan_fields
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("attempt_id") != attempt_id
        or plan.get("purpose") != "bounded_wm_a_pilot"
        or plan.get("citable_as_scientific_evidence") is not False
        or plan.get("authorizes_retry_or_resume") is not False
        or plan.get("allows_refill") is not False
        or plan.get("allows_overwrite") is not False
        or plan.get("branch_mechanism") != "parallel_lockstep_envs_no_restore"
        or plan.get("output_root") != str(root)
        or type(plan.get("states_per_scene")) is not int
        or plan["states_per_scene"] <= 0
        or plan.get("history_blocks") != 2
        or plan.get("action_catalog") != top_action_catalog
        or plan.get("execution_contract") != execution
        or plan.get("runtime_bindings") != runtime
        or not isinstance(plan.get("states"), list)
        or not plan["states"]
    ):
        raise CounterfactualPilotContractError(
            "bound pilot plan disagrees with the final pilot"
        )
    plan_states = plan["states"]
    if [state.get("group_index") for state in plan_states] != list(
        range(len(plan_states))
    ):
        raise CounterfactualPilotContractError(
            "bound pilot plan global group indices changed"
        )
    scene_indices: dict[tuple[str, str], list[int]] = {}
    plan_by_id: dict[str, Mapping[str, object]] = {}
    plan_state_fields = {
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
    for state in plan_states:
        if not isinstance(state, Mapping) or set(state) != plan_state_fields:
            raise CounterfactualPilotContractError("bound pilot plan state is malformed")
        state_id = state.get("state_id")
        if not isinstance(state_id, str) or not state_id or state_id in plan_by_id:
            raise CounterfactualPilotContractError("bound pilot state identity repeats")
        if state.get("role") not in ROLE_NAMES or state.get("family") not in FAMILIES:
            raise CounterfactualPilotContractError("bound pilot state role/family changed")
        if (
            state.get("scene_generation") is not None
            or state.get("candidate_action_ids") != list(range(ACTION_COUNT))
            or state.get("sentinel_duplicate_action_id") is not None
            or not isinstance(state.get("history_action_ids"), list)
            or len(state["history_action_ids"]) != 2
            or any(type(item) is not int or not 0 <= item < ACTION_COUNT for item in state["history_action_ids"])
            or not isinstance(state.get("target_xy_m"), list)
            or len(state["target_xy_m"]) != 2
        ):
            raise CounterfactualPilotContractError(
                "bound pilot train/eval action or target contract changed"
            )
        for target_coordinate in state["target_xy_m"]:
            _finite(target_coordinate, name="plan world target coordinate")
        for binding_name in ("scene_manifest_binding", "scene_genesis_binding"):
            _validate_inert_binding(
                state.get(binding_name),
                name=f"plan {binding_name}",
                synthetic_test_mode=synthetic_test_mode,
                require_absolute=True,
            )
        state_index = _nonnegative_int(
            state.get("state_index_in_scene"), name="plan state_index_in_scene"
        )
        key = (str(state["role"]), str(state.get("scene_id")))
        scene_indices.setdefault(key, []).append(state_index)
        plan_by_id[state_id] = state
    if any(
        sorted(indices) != list(range(len(indices)))
        for indices in scene_indices.values()
    ):
        raise CounterfactualPilotContractError(
            "bound pilot per-scene state indices are not contiguous"
        )
    if any(
        len(indices) != plan["states_per_scene"] for indices in scene_indices.values()
    ):
        raise CounterfactualPilotContractError(
            "bound pilot states_per_scene disagrees with the state grid"
        )

    receipt_bindings = collection["state_receipt_bindings"]
    if not isinstance(receipt_bindings, list) or len(receipt_bindings) != len(plan_states):
        raise CounterfactualPilotContractError("collection state receipt count changed")
    state_receipts: dict[str, Mapping[str, object]] = {}
    for position, (receipt_binding, plan_state) in enumerate(
        zip(receipt_bindings, plan_states, strict=True)
    ):
        receipt_raw, _ = _read_bound_file(
            root, receipt_binding, name=f"state receipt {position}"
        )
        receipt = strict_json_loads(receipt_raw, name=f"state receipt {position}")
        required = {
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
        }
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != required
            or receipt["schema"] != STATE_RECEIPT_SCHEMA
            or receipt["attempt_id"] != attempt_id
            or receipt["status"] != "PHYSICS_COMPLETE"
            or receipt["physics_validated"] is not False
            or receipt["citable_as_scientific_evidence"] is not False
            or receipt["authorizes_retry_or_resume"] is not False
            or receipt["sentinel_audit"] is not None
            or receipt["render_sentinel_audit"] is not None
            or not isinstance(receipt["state"], Mapping)
        ):
            raise CounterfactualPilotContractError(
                "train/eval state receipt status or role boundary changed"
            )
        state = receipt["state"]
        for key in (
            "state_id",
            "role",
            "family",
            "scene_id",
            "group_index",
            "state_index_in_scene",
        ):
            if state.get(key) != plan_state.get(key):
                raise CounterfactualPilotContractError(
                    f"state receipt {key} changed from the plan"
                )
        if state.get("lane_count") != ACTION_COUNT:
            raise CounterfactualPilotContractError(
                "train/eval state receipt lane count changed"
            )
        _validate_inert_binding(
            receipt["render_receipt_binding"],
            name="state live-render receipt",
            synthetic_test_mode=synthetic_test_mode,
        )
        if (
            not isinstance(receipt["context"], Mapping)
            or not isinstance(receipt["branches"], list)
            or len(receipt["branches"]) != ACTION_COUNT
            or receipt["context"].get("history_action_ids")
            != plan_state.get("history_action_ids")
        ):
            raise CounterfactualPilotContractError(
                "state receipt context or candidate grid changed"
            )
        state_id = str(state["state_id"])
        state_receipts[state_id] = receipt
    if set(state_receipts) != set(plan_by_id):
        raise CounterfactualPilotContractError(
            "collection state receipts do not exactly cover the plan"
        )
    return MappingProxyType(state_receipts), len(state_receipts)


def load_bound_pilot_v1(
    pilot_root: Path,
    *,
    expected_manifest_byte_count: int,
    expected_manifest_sha256: str,
    allowed_parent: Path | None = None,
    synthetic_test_mode: bool = False,
) -> CounterfactualPilotBundleV1:
    """Open manifest/index receipts without opening an RGB leaf."""
    if type(synthetic_test_mode) is not bool:
        raise CounterfactualPilotContractError(
            "synthetic_test_mode must be an explicit boolean"
        )
    if synthetic_test_mode and allowed_parent is None:
        raise CounterfactualPilotContractError(
            "synthetic test mode requires an explicit confined allowed_parent"
        )
    selected = Path(pilot_root)
    if selected.is_symlink() or not selected.is_dir():
        raise CounterfactualPilotContractError(
            "pilot root must be a regular non-symlink directory"
        )
    root = selected.resolve()
    parent = (allowed_parent or DEV_ROOT).resolve()
    if not root.is_relative_to(parent):
        raise CounterfactualPilotContractError(
            f"pilot root must remain under {parent}"
        )
    manifest_raw, manifest_binding = _manifest_binding_from_caller(
        root,
        expected_byte_count=expected_manifest_byte_count,
        expected_sha256=expected_manifest_sha256,
    )
    manifest = strict_json_loads(manifest_raw, name="pilot manifest")
    expected_manifest_fields = {
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
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise CounterfactualPilotContractError("pilot manifest fields are incomplete")
    if (
        manifest["schema"] != PILOT_MANIFEST_SCHEMA
        or manifest["purpose"] != "bounded_wm_a_pilot"
        or manifest["status"] != "COMPLETE"
        or manifest["physics_validated"] is not True
        or manifest["citable_as_scientific_evidence"] is not False
        or manifest["authorizes_retry_or_resume"] is not False
        or manifest["evidence_scope"] != EVIDENCE_SCOPE
    ):
        raise CounterfactualPilotContractError("pilot manifest status or scope changed")
    if not isinstance(manifest["attempt_id"], str) or not manifest["attempt_id"]:
        raise CounterfactualPilotContractError("pilot attempt identity is absent")
    if manifest["output_root"] != str(root):
        raise CounterfactualPilotContractError("pilot output_root does not bind selected root")
    receipt_root_value = manifest["receipt_root"]
    if not isinstance(receipt_root_value, str):
        raise CounterfactualPilotContractError("pilot receipt_root must be absolute")
    receipt_root_selected = Path(receipt_root_value)
    if (
        not receipt_root_selected.is_absolute()
        or receipt_root_selected.is_symlink()
        or not receipt_root_selected.is_dir()
    ):
        raise CounterfactualPilotContractError("pilot receipt_root is invalid")
    receipt_root = receipt_root_selected.resolve()
    if receipt_root != root:
        raise CounterfactualPilotContractError(
            "pilot receipt_root must equal the selected output_root"
        )
    action_catalog = manifest["action_catalog"]
    if not isinstance(action_catalog, list) or len(action_catalog) != ACTION_COUNT:
        raise CounterfactualPilotContractError("pilot action catalog changed")
    for action_id, entry in enumerate(action_catalog):
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"action_id", "name", "requested_block"}
            or entry["action_id"] != action_id
            or entry["name"] != PRIMITIVE_NAMES[action_id]
        ):
            raise CounterfactualPilotContractError("pilot action catalog changed")
    requested_blocks = tuple(
        _validate_tape(entry["requested_block"], name=f"action {action_id} requested block")
        for action_id, entry in enumerate(action_catalog)
    )
    if requested_blocks != CANONICAL_ACTION_BLOCKS:
        raise CounterfactualPilotContractError(
            "pilot action catalog command values changed from the canonical registry"
        )
    action_contract = manifest["action_contract"]
    if (
        not isinstance(action_contract, Mapping)
        or set(action_contract) != {
            "primitive_names",
            "command_ticks_per_block",
            "executed_tape_shape",
            "candidate_model_input",
            "future_executed_tape_usage",
        }
        or action_contract.get("primitive_names") != list(PRIMITIVE_NAMES)
        or action_contract.get("command_ticks_per_block") != COMMAND_TICKS_PER_BLOCK
        or action_contract.get("executed_tape_shape") != list(EXECUTED_TAPE_SHAPE)
        or action_contract.get("candidate_model_input") != "requested_action_id"
        or action_contract.get("future_executed_tape_usage")
        != "target_and_audit_only"
    ):
        raise CounterfactualPilotContractError("pilot action contract changed")
    source_bindings = _validate_source_bindings(
        manifest["source_bindings"], synthetic_test_mode=synthetic_test_mode
    )

    collection_binding = manifest["collection_receipt"]
    if not isinstance(collection_binding, Mapping) or set(collection_binding) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise CounterfactualPilotContractError("collection receipt binding changed")
    collection_relative = _canonical_relative(
        collection_binding["path"], name="collection receipt path"
    )
    collection_path = _safe_path(root, collection_relative, final_file=True)
    if not collection_path.is_relative_to(receipt_root):
        raise CounterfactualPilotContractError("collection receipt escapes receipt_root")
    collection_states, state_receipt_open_count = _load_collection_receipt(
        root,
        collection_binding,
        attempt_id=manifest["attempt_id"],
        top_action_catalog=action_catalog,
        top_source_bindings=source_bindings,
        synthetic_test_mode=synthetic_test_mode,
    )

    excluded, tolerances = _validate_calibration_contract(
        manifest["calibration_contract"]
    )
    calibration_receipt = _load_calibration_receipt(
        root,
        manifest["calibration_receipt"],
        attempt_id=manifest["attempt_id"],
        top_contract=manifest["calibration_contract"],
        synthetic_test_mode=synthetic_test_mode,
    )

    rgb_raw, rgb_binding = _read_bound_file(
        root, manifest["rgb_artifact_manifest"], name="RGB artifact manifest"
    )
    rgb_manifest = strict_json_loads(rgb_raw, name="RGB artifact manifest")
    if (
        not isinstance(rgb_manifest, Mapping)
        or set(rgb_manifest) != {"schema", "artifacts"}
        or rgb_manifest["schema"] != RGB_MANIFEST_SCHEMA
        or not isinstance(rgb_manifest["artifacts"], list)
        or not rgb_manifest["artifacts"]
    ):
        raise CounterfactualPilotContractError("RGB artifact manifest schema changed")
    artifacts: dict[str, RGBArtifactV1] = {}
    artifact_paths = set()
    frame_identities = set()
    for item in rgb_manifest["artifacts"]:
        if not isinstance(item, Mapping) or set(item) != {
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
        }:
            raise CounterfactualPilotContractError("RGB artifact receipt changed")
        artifact_id = item["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in artifacts:
            raise CounterfactualPilotContractError("RGB artifact ID repeats")
        relative = _canonical_relative(item["path"], name="RGB artifact path")
        if (
            relative.as_posix() in artifact_paths
            or item.get("frame_identity") in frame_identities
        ):
            raise CounterfactualPilotContractError("RGB artifact identity repeats")
        if (
            item["width"] != 224
            or item["height"] != 224
            or item["mode"] != "RGB"
            or item["format"] != "PNG"
            or item["camera_valid"] is not True
            or not isinstance(item["frame_identity"], str)
            or not item["frame_identity"]
            or not _is_sha256(item["file_sha256"])
        ):
            raise CounterfactualPilotContractError("RGB artifact media contract changed")
        artifact_path = _safe_path(root, relative, final_file=True)
        artifact_byte_count = _positive_int(
            item["byte_count"], name="RGB byte_count"
        )
        if artifact_path.stat().st_size != artifact_byte_count:
            raise CounterfactualPilotContractError(
                "RGB artifact byte-count receipt changed"
            )
        artifacts[artifact_id] = RGBArtifactV1(
            artifact_id=artifact_id,
            frame_identity=item["frame_identity"],
            relative_path=relative.as_posix(),
            byte_count=artifact_byte_count,
            file_sha256=item["file_sha256"],
        )
        artifact_paths.add(relative.as_posix())
        frame_identities.add(item["frame_identity"])

    roles = manifest["roles"]
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_NAMES):
        raise CounterfactualPilotContractError("pilot must bind exactly train and eval")
    groups_by_role: dict[str, tuple[CounterfactualGroupV1, ...]] = {}
    role_bindings: dict[str, Mapping[str, object]] = {}
    role_artifacts: dict[str, set[str]] = {}
    all_group_ids = set()
    all_group_indices = set()
    all_state_ids = set()
    claimed_artifacts = set()
    role_scenes: dict[str, set[str]] = {}
    for role in ROLE_NAMES:
        role_manifest = roles[role]
        if not isinstance(role_manifest, Mapping) or set(role_manifest) != {
            "index",
            "group_count",
            "branch_count",
            "scene_ids",
        }:
            raise CounterfactualPilotContractError(f"pilot {role} role changed")
        raw, binding = _read_bound_file(
            root, role_manifest["index"], name=f"pilot {role} index"
        )
        rows = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line:
                raise CounterfactualPilotContractError(
                    f"pilot {role} index contains an empty row"
                )
            parsed_row = strict_json_loads(
                line, name=f"pilot {role} row {line_number}"
            )
            state_id = (
                parsed_row.get("state_id")
                if isinstance(parsed_row, Mapping)
                else None
            )
            if not isinstance(state_id, str) or state_id not in collection_states:
                raise CounterfactualPilotContractError(
                    f"pilot {role} row is not bound to a collection state"
                )
            rows.append(_parse_group(
                parsed_row,
                role=role,
                artifacts=artifacts,
                tolerances=tolerances,
                requested_blocks=requested_blocks,
                collection_state=collection_states[state_id],
            ))
        declared_count = _positive_int(
            role_manifest["group_count"], name=f"{role} group_count"
        )
        if (
            len(rows) != declared_count
            or role_manifest["branch_count"] != len(rows) * ACTION_COUNT
        ):
            raise CounterfactualPilotContractError(f"pilot {role} row counts changed")
        scenes = {row.scene_id for row in rows}
        if {row.family for row in rows} != set(FAMILIES):
            raise CounterfactualPilotContractError(
                f"pilot {role} does not cover the exact eight-family panel"
            )
        for scene_id in scenes:
            indices = sorted(
                row.state_index_in_scene
                for row in rows
                if row.scene_id == scene_id
            )
            if indices != list(range(len(indices))):
                raise CounterfactualPilotContractError(
                    f"pilot {role} scene state grid is not contiguous"
                )
        declared_scenes = role_manifest["scene_ids"]
        if (
            not isinstance(declared_scenes, list)
            or declared_scenes != sorted(scenes)
            or not scenes
        ):
            raise CounterfactualPilotContractError(f"pilot {role} scene receipt changed")
        for row in rows:
            state_identity = (row.scene_id, row.state_index)
            if (
                row.group_id in all_group_ids
                or row.group_index in all_group_indices
                or state_identity in all_state_ids
            ):
                raise CounterfactualPilotContractError("pilot state or group identity repeats")
            all_group_ids.add(row.group_id)
            all_group_indices.add(row.group_index)
            all_state_ids.add(state_identity)
        used = set()
        for row in rows:
            references = (
                *row.context_rgb_artifact_ids,
                *(branch.target_rgb_artifact_id for branch in row.branches),
            )
            if len(set(references)) != len(references):
                raise CounterfactualPilotContractError(
                    "one pilot group reuses a context/target RGB artifact"
                )
            if claimed_artifacts & set(references):
                raise CounterfactualPilotContractError(
                    "RGB artifacts are reused across pilot groups"
                )
            used.update(references)
            claimed_artifacts.update(references)
        role_artifacts[role] = used
        role_scenes[role] = scenes
        groups_by_role[role] = tuple(rows)
        role_bindings[role] = MappingProxyType(binding)
    overlap = role_scenes["train"] & role_scenes["eval"]
    calibration_overlap = set(excluded) & (
        role_scenes["train"] | role_scenes["eval"]
    )
    if overlap or calibration_overlap:
        raise CounterfactualPilotContractError(
            "calibration/train/eval scene roles are not disjoint"
        )
    if role_artifacts["train"] & role_artifacts["eval"]:
        raise CounterfactualPilotContractError("train/eval RGB identities overlap")
    if role_artifacts["train"] | role_artifacts["eval"] != set(artifacts):
        raise CounterfactualPilotContractError("RGB manifest contains unreferenced artifacts")
    if sorted(all_group_indices) != list(range(len(all_group_indices))):
        raise CounterfactualPilotContractError(
            "pilot global group_index values are not the contiguous range"
        )
    if all_group_ids != set(collection_states):
        raise CounterfactualPilotContractError(
            "joined train/eval groups do not exactly cover collection states"
        )

    return CounterfactualPilotBundleV1(
        root=root,
        manifest_binding=MappingProxyType(manifest_binding),
        manifest=MappingProxyType(dict(manifest)),
        rgb_manifest_binding=MappingProxyType(rgb_binding),
        artifacts=MappingProxyType(artifacts),
        groups_by_role=MappingProxyType(groups_by_role),
        role_bindings=MappingProxyType(role_bindings),
        calibration_receipt=calibration_receipt,
        calibration_tolerances=tolerances,
        access_audit=MappingProxyType({
            "manifest_open_count": 1,
            "role_index_open_count": 2,
            "rgb_manifest_open_count": 1,
            "collection_receipt_open_count": 1,
            "plan_receipt_open_count": 1,
            "state_receipt_open_count": state_receipt_open_count,
            "calibration_receipt_open_count": 1,
            "rgb_leaf_open_count": 0,
            "checkpoint_open_count": 0,
        }),
    )


def read_bound_rgb_bytes_v1(
    bundle: CounterfactualPilotBundleV1,
    artifact_id: str,
) -> bytes:
    """Read and hash one manifest-bound RGB leaf without following links."""
    artifact = bundle.artifacts.get(artifact_id)
    if artifact is None:
        raise CounterfactualPilotContractError(f"unknown RGB artifact: {artifact_id}")
    relative = PurePosixPath(artifact.relative_path)
    descriptor = os.open(bundle.root, _DIR_FLAGS)
    file_descriptor = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CounterfactualPilotContractError("RGB artifact is not a regular file")
        chunks = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(file_descriptor)
    except OSError as exc:
        raise CounterfactualPilotContractError(
            f"cannot safely open RGB artifact {artifact_id}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(raw) != artifact.byte_count
        or digest.hexdigest() != artifact.file_sha256
    ):
        raise CounterfactualPilotContractError(
            f"RGB artifact bytes changed: {artifact_id}"
        )
    return raw


__all__ = [
    "ACTION_COUNT",
    "CandidateActionInputV1",
    "CounterfactualBranchV1",
    "CounterfactualGroupV1",
    "CounterfactualPilotBundleV1",
    "CounterfactualPilotContractError",
    "FAMILIES",
    "PhysicalLabelsV1",
    "GROUP_SCHEMA",
    "PILOT_MANIFEST_SCHEMA",
    "PRIMITIVE_NAMES",
    "RGB_MANIFEST_SCHEMA",
    "TOTAL_BRANCH_COUNT",
    "canonical_json_sha256",
    "candidate_model_inputs_v1",
    "load_bound_pilot_v1",
    "physical_dense_ranks_v1",
    "physical_oracle_key_v1",
    "read_bound_rgb_bytes_v1",
    "strict_json_loads",
]
