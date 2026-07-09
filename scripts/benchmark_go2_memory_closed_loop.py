#!/usr/bin/env python3
"""Closed-loop Go2 hidden-target memory navigation in Genesis (kinematic mode).

The robot explores a held-out maze, sees a colored target landmark (binds it in
the vector-memory controller), the landmark leaves view, and the controller's
metric-direction memory steers the robot back to claim it -- using only ego RGB,
proprioceptive egomotion (pose deltas), and the goal color. This is the
closed-loop analog of the 2D see->hide->claim demo and the live counterpart to
the offline leave-one-scene-out steering gate.

Driving is kinematic (named velocity primitives integrated over command_dt_s with
grid feasibility) -- it tests navigation/exploration/memory LOGIC without gait
stability; RL-gait via RolloutRunner is a later deployability upgrade. Runtime
inputs are ego RGB + proprioceptive egomotion + goal color; landmark world
positions are used only to choose the target and score success.

Run in the vulkan venv:
  .generated/venvs/genesis_render_vulkan/bin/python scripts/benchmark_go2_memory_closed_loop.py \
    --controller .../exact_cv/exact_000c67a65968_s20260820.pt \
    --frozen-jepa-checkpoint .../contrast02.pt \
    --scene-id medium_enclosed_maze_000c67a65968 --target-color green \
    --policy memory --max-ticks 120 --demo-video out.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_lewm_closed_loop_mpc import (  # noqa: E402
    _current_pose,
    _execute_kinematic_primitive,
    _execute_physical_primitive,
    _render_tensor_from_base,
    _yaw_from_quat_wxyz,
    _set_pose,
)
from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    PRIMITIVE_NAMES,
    load_controller,
)
from lewm.models.go2_jepa import (  # noqa: E402
    Go2FrontBlockedHead,
    Go2PrimitiveOutcomeHead,
    load_go2_jepa_encoder,
)
from lewm_genesis.lewm_contract import (  # noqa: E402
    PrimitiveRegistry,
    SafetyLimits,
    expand_primitive_to_block,
)
from lewm_genesis.rollout import (  # noqa: E402
    GenesisGo2PPOPolicy,
    RolloutConfig,
    RolloutRunner,
)
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_genesis.collectors.base import wrap_angle_pi  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid, safe_standoff_xys  # noqa: E402

# Primitive velocity table (config/go2_primitive_registry.yaml) for aux command.
_PRIM_CMD = {
    "forward_medium": (0.25, 0.0, 0.0), "forward_slow": (0.2, 0.0, 0.0),
    "arc_left": (0.2, 0.0, 0.45), "arc_right": (0.2, 0.0, -0.45),
    "yaw_left": (0.0, 0.0, 0.45), "yaw_right": (0.0, 0.0, -0.45),
    "backward": (-0.2, 0.0, 0.0), "hold": (0.0, 0.0, 0.0),
}
_STRAIGHT_FORWARD_PRIMITIVES = frozenset(("forward_slow", "forward_medium", "forward_fast"))
_FORWARD_PRIMITIVES = frozenset(("forward_slow", "forward_medium", "forward_fast", "arc_left", "arc_right"))
_TURN_PRIMITIVES = frozenset(("yaw_left", "yaw_right"))
_TRANSLATING_PRIMITIVES = frozenset((*_FORWARD_PRIMITIVES, "backward"))
_LEARNED_LOCAL_POLICY_PRIMITIVES = (
    "forward_fast",
    "forward_medium",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
)
_LEARNED_LOCAL_STATE_FEATURES = ("EXPLORE", "SEEK", "SERVO", "CLAIM")
_LEARNED_LOCAL_ONLINE_MAP_CHANNELS = 8
_CLAIM_SUCCESS_FEATURE_SCHEMA = "lewm_go2_claim_success_head_features_v0"
_CLAIM_SUCCESS_CHECKPOINT_SCHEMA = "lewm_go2_claim_success_head_v0"
_TARGET_SCHEDULER_FEATURE_SCHEMA = "lewm_go2_target_scheduler_features_v1"
_TARGET_SCHEDULER_CHECKPOINT_SCHEMA = "lewm_go2_target_scheduler_head_v0"


def _learned_local_policy_label_primitive(primitive: str | None) -> str | None:
    """Map executable primitives to the compact learned-local policy action vocab."""
    if primitive is None:
        return None
    name = str(primitive)
    if name in _LEARNED_LOCAL_POLICY_PRIMITIVES:
        return name
    if name == "forward_slow":
        return "forward_medium"
    return None


def _load_debug_force_primitive_script(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload_text = path.read_text(encoding="utf-8").strip()
    if not payload_text:
        return {}
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = []
        for line_no, raw_line in enumerate(payload_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise SystemExit(
                    f"{path}:{line_no}: expected 'tick primitive' or 'tick,primitive'"
                )
            payload.append({"tick": parts[0], "primitive": parts[1]})

    rows: list[tuple[Any, Any]] = []
    if isinstance(payload, dict):
        if "overrides" in payload and isinstance(payload["overrides"], dict):
            rows = list(payload["overrides"].items())
        else:
            rows = list(payload.items())
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise SystemExit(f"{path}: list item {idx} is not an object")
            rows.append((item.get("tick"), item.get("primitive")))
    else:
        raise SystemExit(f"{path}: expected JSON object or list")

    script: dict[int, str] = {}
    allowed = set(_LEARNED_LOCAL_POLICY_PRIMITIVES) | {"forward_slow"}
    for raw_tick, raw_primitive in rows:
        if raw_tick is None or raw_primitive is None:
            raise SystemExit(f"{path}: forced primitive row missing tick or primitive")
        tick = int(raw_tick)
        primitive = str(raw_primitive)
        if primitive not in allowed:
            raise SystemExit(
                f"{path}: unsupported forced primitive {primitive!r} at tick {tick}"
            )
        script[tick] = primitive
    return dict(sorted(script.items()))


def _scene_spawn(scene_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads((scene_dir / "genesis_scene.json").read_text())
    sp = d["spawn"]
    return np.asarray(sp["xyz_m"], dtype=np.float32), np.asarray(sp["quat_wxyz"], dtype=np.float32)


def _scene_landmarks(scene_dir: Path) -> dict[str, np.ndarray]:
    d = json.loads((scene_dir / "genesis_scene.json").read_text())
    out = {}
    for o in d.get("objects", ()):
        if str(o.get("kind")) != "landmark":
            continue
        mat = str(o.get("material_id", ""))
        color = mat.replace("landmark_", "") if mat.startswith("landmark_") else mat
        out[color] = np.asarray(o["center_xyz_m"][:2], dtype=np.float32)
    return out


def _quat_wxyz_from_yaw_local(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _load_slice_start(
    path: Path,
    *,
    start_tick: int,
    preclaimed_colors: set[str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    log = payload.get("log", [])
    if not isinstance(log, list):
        raise SystemExit(f"{path} has no list log for slice start")
    rows = [
        row
        for row in log
        if isinstance(row, dict)
        and row.get("tick") is not None
        and int(row.get("tick")) >= int(start_tick)
        and row.get("post_xy") is not None
        and row.get("post_yaw") is not None
    ]
    if not rows:
        raise SystemExit(f"{path} has no pose row at or after slice start tick {start_tick}")
    start_row = rows[0]
    preclaims: list[dict[str, Any]] = []
    for claim in result.get("beacon_claims", []):
        if not isinstance(claim, dict):
            continue
        color = str(claim.get("target_color", "")).lower()
        if color in preclaimed_colors:
            preclaims.append(dict(claim))
    missing = sorted(preclaimed_colors - {str(item.get("target_color", "")).lower() for item in preclaims})
    if missing:
        raise SystemExit(f"{path} is missing requested preclaimed colors: {','.join(missing)}")
    return {
        "path": str(path),
        "source_ticks_used": int(result.get("ticks_used", 0) or 0),
        "start_tick": int(start_row.get("tick")),
        "start_xy": [float(start_row["post_xy"][0]), float(start_row["post_xy"][1])],
        "start_yaw": float(start_row["post_yaw"]),
        "preclaims": preclaims,
        "preload_log": log,
        "source_target_xy": result.get("target_xy"),
        "source_claimed_colors": result.get("claimed_colors", []),
    }


def _look_ahead_free(grid, pos_xy, yaw, dist_m: float = 0.5) -> bool:
    return grid.is_free((float(pos_xy[0]) + dist_m * math.cos(yaw),
                         float(pos_xy[1]) + dist_m * math.sin(yaw)))


def _round_float(value: float | None, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _claim_gate_entry(
    *,
    enabled: bool,
    seen: bool,
    in_cone: bool,
    seen_age_ticks: int,
    min_seen_ticks: int,
    area: float,
    area_threshold: float | None,
    bearing: float,
    bearing_threshold: float | None,
    extra_rejections: list[str] | None = None,
) -> dict[str, Any]:
    rejections: list[str] = []
    if not bool(enabled):
        rejections.append("disabled")
    if not bool(seen):
        rejections.append("not_seen")
    if not bool(in_cone):
        rejections.append("not_in_cone")
    if int(seen_age_ticks) < int(min_seen_ticks):
        rejections.append("seen_age_low")
    if area_threshold is None:
        rejections.append("area_threshold_disabled")
    elif float(area) <= float(area_threshold):
        rejections.append("area_low")
    if bearing_threshold is None:
        rejections.append("bearing_threshold_disabled")
    elif abs(float(bearing)) >= float(bearing_threshold):
        rejections.append("bearing_high")
    if extra_rejections:
        rejections.extend(str(item) for item in extra_rejections if str(item))
    return {
        "enabled": bool(enabled),
        "passed": not rejections,
        "reject_reasons": rejections,
        "area_threshold": None if area_threshold is None else float(area_threshold),
        "bearing_threshold": None if bearing_threshold is None else float(bearing_threshold),
        "min_seen_ticks": int(min_seen_ticks),
    }


def _claim_success_proxy_gate_entry(
    *,
    area: float,
    area_threshold: float | None,
    bearing: float,
    bearing_threshold: float | None,
    model_score: float | None = None,
    model_threshold: float | None = None,
) -> dict[str, Any]:
    rejections: list[str] = []
    enabled = bool(
        area_threshold is not None
        or bearing_threshold is not None
        or model_threshold is not None
    )
    if area_threshold is not None and float(area) <= float(area_threshold):
        rejections.append("area_low")
    if bearing_threshold is not None and abs(float(bearing)) >= float(bearing_threshold):
        rejections.append("bearing_high")
    if (
        model_threshold is not None
        and (model_score is None or float(model_score) < float(model_threshold))
    ):
        rejections.append("model_low")
    return {
        "enabled": bool(enabled),
        "passed": bool(not enabled or not rejections),
        "reject_reasons": rejections,
        "area_threshold": None if area_threshold is None else float(area_threshold),
        "bearing_threshold": None if bearing_threshold is None else float(bearing_threshold),
        "model_score": None if model_score is None else _round_float(float(model_score), 4),
        "model_threshold": None if model_threshold is None else float(model_threshold),
        "source": "learned_rgb_area_bearing",
    }


class ClaimSuccessHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TargetSchedulerHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, color_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(color_count)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _target_scheduler_feature(
    *,
    colors: list[str],
    current_color: str,
    current_target_age_ticks: int,
    claimed_colors: set[str],
    color_readouts: dict[str, dict[str, Any]],
    tick: int,
    max_ticks: int,
    device: torch.device,
) -> torch.Tensor:
    current_key = str(current_color).lower()
    claimed = {str(item).lower() for item in claimed_colors}
    max_tick = max(1.0, float(max_ticks))
    feature: list[float] = [
        min(1.0, max(0.0, float(tick) / max_tick)),
        min(1.0, max(0.0, float(len(claimed)) / max(1.0, float(len(colors))))),
        min(1.0, max(0.0, float(current_target_age_ticks) / 256.0)),
    ]
    for color in colors:
        key = str(color).lower()
        readout = color_readouts.get(key, {})
        first_seen_tick = readout.get("first_seen_tick")
        last_seen_tick = readout.get("last_seen_tick")
        first_age = (
            9999.0
            if first_seen_tick is None
            else max(0.0, float(tick) - float(first_seen_tick))
        )
        last_age = (
            9999.0
            if last_seen_tick is None
            else max(0.0, float(tick) - float(last_seen_tick))
        )
        feature.extend(
            [
                1.0 if key == current_key else 0.0,
                1.0 if key in claimed or bool(readout.get("claimed", False)) else 0.0,
                _safe_float(readout.get("mem_conf"), 0.0),
                _safe_float(readout.get("area"), -99.0),
                _safe_float(readout.get("read_score"), 0.0),
                1.0 if bool(readout.get("read_gate_pass", False)) else 0.0,
                min(1.0, first_age / 128.0),
                min(1.0, last_age / 128.0),
            ]
        )
    return torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)


def _ctrl_state_to_cpu(ctrl_state: tuple | None) -> tuple | None:
    if ctrl_state is None:
        return None
    return tuple(
        item.detach().cpu() if isinstance(item, torch.Tensor) else item
        for item in ctrl_state
    )


def _ctrl_state_to_device(ctrl_state: tuple | None, *, device: torch.device) -> tuple | None:
    if ctrl_state is None:
        return None
    return tuple(
        item.to(device) if isinstance(item, torch.Tensor) else item
        for item in ctrl_state
    )


def _write_slice_snapshot(
    path: Path,
    *,
    build: Any,
    runner: RolloutRunner | None,
    scene_id: str,
    tick: int,
    next_tick: int,
    pos: np.ndarray,
    quat: np.ndarray,
    yaw: float,
    ctrl_state: tuple | None,
    target_sequence: list[str],
    target_index: int,
    target_active_since_tick: int,
    beacon_claims: list[dict[str, Any]],
    first_seen_ticks: dict[str, int],
    last_seen_ticks: dict[str, int],
    last_primitive: str,
    last_cmd: tuple[float, float, float],
    online_map: OnlineEgomotionMap | None,
    feature_max_ticks: int,
    source_result: str | None,
) -> None:
    physics_state = _slice_physics_state(build=build, runner=runner)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "lewm_go2_yellow_slice_resume_snapshot_v0",
            "scene_id": str(scene_id),
            "tick": int(tick),
            "next_tick": int(next_tick),
            "pos_xyz": np.asarray(pos, dtype=np.float32).tolist(),
            "quat_wxyz": np.asarray(quat, dtype=np.float32).tolist(),
            "yaw_rad": float(yaw),
            "ctrl_state": _ctrl_state_to_cpu(ctrl_state),
            "target_sequence": list(target_sequence),
            "target_index": int(target_index),
            "target_active_since_tick": int(target_active_since_tick),
            "beacon_claims": [dict(item) for item in beacon_claims],
            "first_seen_ticks": {str(k): int(v) for k, v in first_seen_ticks.items()},
            "last_seen_ticks": {str(k): int(v) for k, v in last_seen_ticks.items()},
            "last_primitive": str(last_primitive),
            "last_cmd": tuple(float(v) for v in last_cmd),
            "online_map": None if online_map is None else online_map.state_dict(),
            "feature_max_ticks": int(feature_max_ticks),
            "source_result": source_result,
            "physics_state": physics_state,
        },
        path,
    )


def _first_env_np(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    arr = RolloutRunner._as_np(value)
    out = np.asarray(arr, dtype=dtype)
    if out.ndim >= 2:
        return np.asarray(out[0], dtype=dtype)
    return out


def _slice_physics_state(*, build: Any, runner: RolloutRunner | None) -> dict[str, Any] | None:
    if build is None or runner is None or not hasattr(build, "robot"):
        return None
    robot = build.robot
    state: dict[str, Any] = {
        "qpos": _first_env_np(robot.get_qpos()).tolist(),
        "dofs_velocity": _first_env_np(robot.get_dofs_velocity()).tolist(),
        "n_qs": int(getattr(robot, "n_qs", -1)),
        "n_dofs": int(getattr(robot, "n_dofs", -1)),
        "runner_last_executed": _first_env_np(getattr(runner, "_last_executed", np.zeros((1, 3), dtype=np.float32))).tolist(),
        "runner_sim_time_ns": int(getattr(runner, "_sim_time_ns", 0)),
        "runner_sequence_id_counter": int(getattr(runner, "_sequence_id_counter", 0)),
    }
    policy_last_actions = getattr(getattr(runner, "policy", None), "_last_actions", None)
    if policy_last_actions is not None:
        state["policy_last_actions"] = _first_env_np(policy_last_actions).tolist()
    return state


def _restore_slice_snapshot_physics(
    *,
    build: Any,
    runner: RolloutRunner | None,
    snapshot: dict[str, Any],
    fallback_pos: np.ndarray,
    fallback_quat: np.ndarray,
) -> bool:
    physics_state = snapshot.get("physics_state")
    if runner is None or not isinstance(physics_state, dict):
        _set_pose(build=build, runner=runner, pos_xyz=fallback_pos, quat_wxyz=fallback_quat)
        return False
    robot = build.robot
    envs = [0]
    qpos = np.asarray(physics_state.get("qpos"), dtype=np.float32)
    dofs_velocity = np.asarray(physics_state.get("dofs_velocity"), dtype=np.float32)
    if qpos.ndim != 1 or dofs_velocity.ndim != 1:
        _set_pose(build=build, runner=runner, pos_xyz=fallback_pos, quat_wxyz=fallback_quat)
        return False
    if int(physics_state.get("n_qs", qpos.shape[0])) != int(getattr(robot, "n_qs", qpos.shape[0])):
        _set_pose(build=build, runner=runner, pos_xyz=fallback_pos, quat_wxyz=fallback_quat)
        return False
    if int(physics_state.get("n_dofs", dofs_velocity.shape[0])) != int(
        getattr(robot, "n_dofs", dofs_velocity.shape[0])
    ):
        _set_pose(build=build, runner=runner, pos_xyz=fallback_pos, quat_wxyz=fallback_quat)
        return False

    robot.set_qpos(qpos[None, :], envs_idx=envs, zero_velocity=False)
    robot.set_dofs_velocity(dofs_velocity[None, :], envs_idx=envs)
    last_executed = np.asarray(physics_state.get("runner_last_executed", [0.0, 0.0, 0.0]), dtype=np.float32)
    if last_executed.shape == (3,):
        runner._last_executed[0] = last_executed
    runner._sim_time_ns = int(physics_state.get("runner_sim_time_ns", getattr(runner, "_sim_time_ns", 0)))
    runner._sequence_id_counter = int(
        physics_state.get("runner_sequence_id_counter", getattr(runner, "_sequence_id_counter", 0))
    )
    policy_last_actions = physics_state.get("policy_last_actions")
    if policy_last_actions is not None and hasattr(runner.policy, "_last_actions"):
        arr = np.asarray(policy_last_actions, dtype=np.float32)
        expected = getattr(runner.policy, "_last_actions", None)
        if expected is not None and np.asarray(expected).ndim == 2 and arr.shape == np.asarray(expected)[0].shape:
            runner.policy._last_actions[0] = arr
        elif expected is None:
            runner.policy._last_actions = arr[None, :]
    return True


def _claim_success_model_feature(
    *,
    color: str,
    color_vocab: list[str],
    area: float,
    bearing: float,
    mem_conf: float,
    read_score: float | None,
    seen_age_ticks: int,
    seen: bool,
    in_cone: bool,
    claimed_count: int,
    tick: int,
    max_ticks: int,
    device: torch.device,
) -> torch.Tensor:
    color_key = str(color).lower()
    one_hot = [1.0 if color_key == str(item).lower() else 0.0 for item in color_vocab]
    feature = np.asarray(
        [
            *one_hot,
            float(area),
            float(bearing),
            abs(float(bearing)),
            float(mem_conf),
            0.0 if read_score is None else float(read_score),
            min(1.0, max(0.0, float(seen_age_ticks) / 64.0)),
            1.0 if bool(seen) else 0.0,
            1.0 if bool(in_cone) else 0.0,
            min(1.0, max(0.0, float(claimed_count) / max(1.0, float(len(color_vocab))))),
            min(1.0, max(0.0, float(tick) / max(1.0, float(max_ticks)))),
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(feature).to(device=device).unsqueeze(0)


def _post_claim_acquisition_diagnostics(
    log: list[dict[str, Any]],
    *,
    target_sequence: list[str],
    min_claims: int,
) -> dict[str, Any]:
    min_claims = max(0, int(min_claims))
    claimed: set[str] = set()
    active = False
    start_tick: int | None = None
    start_xy: list[float] | None = None
    last_xy: list[float] | None = None
    xs: list[float] = []
    ys: list[float] = []
    primitive_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    remaining_seen: dict[str, int] = {}
    remaining_area_max: dict[str, float] = {}
    remaining_mem_conf_max: dict[str, float] = {}
    remaining_read_score_max: dict[str, float] = {}
    active_claim_rejects: dict[str, int] = {}
    wall_requested_counts: dict[str, int] = {}
    wall_selected_counts: dict[str, int] = {}
    wall_vetoes = 0
    stalled_ticks = 0
    hard_stalled_ticks = 0

    def _xy_from_row(row: dict[str, Any]) -> list[float] | None:
        xy = row.get("post_xy")
        if not isinstance(xy, list) or len(xy) < 2:
            xy = row.get("xy")
        if not isinstance(xy, list) or len(xy) < 2:
            return None
        try:
            return [float(xy[0]), float(xy[1])]
        except (TypeError, ValueError):
            return None

    for row in log:
        if not isinstance(row, dict):
            continue
        if str(row.get("state", "")).upper() == "CLAIM":
            claim_color = str(row.get("target_color", "")).lower()
            if claim_color:
                claimed.add(claim_color)
        if len(claimed) < min_claims:
            continue
        tick = int(row.get("tick", 0))
        if not active:
            active = True
            start_tick = tick
            start_xy = _xy_from_row(row)
        xy = _xy_from_row(row)
        if xy is not None:
            if start_xy is None:
                start_xy = xy
            last_xy = xy
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
        state_name = str(row.get("state", "")).upper()
        if state_name:
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
        primitive = str(row.get("primitive", ""))
        if primitive:
            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
        if bool(row.get("stalled", False)):
            stalled_ticks += 1
        if bool(row.get("hard_stalled", False)):
            hard_stalled_ticks += 1
        wall_guard = row.get("wall_guard")
        if isinstance(wall_guard, dict):
            requested = str(wall_guard.get("requested", ""))
            selected = str(wall_guard.get("selected", ""))
            if requested:
                wall_requested_counts[requested] = wall_requested_counts.get(requested, 0) + 1
            if selected:
                wall_selected_counts[selected] = wall_selected_counts.get(selected, 0) + 1
            if bool(wall_guard.get("vetoed", False)):
                wall_vetoes += 1
        readouts = row.get("color_readouts")
        if isinstance(readouts, dict):
            for color in target_sequence:
                color_key = str(color).lower()
                if color_key in claimed:
                    continue
                item = readouts.get(color_key)
                if not isinstance(item, dict):
                    continue
                if bool(item.get("claimed", False)):
                    continue
                if bool(item.get("first_seen_tick") is not None) or float(item.get("mem_conf", 0.0)) > 0.0:
                    remaining_seen.setdefault(color_key, tick)
                for out_key, dest in (
                    ("area", remaining_area_max),
                    ("mem_conf", remaining_mem_conf_max),
                    ("read_score", remaining_read_score_max),
                ):
                    value = item.get(out_key)
                    if value is None:
                        continue
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        continue
                    dest[color_key] = max(dest.get(color_key, float("-inf")), value_f)
        gate = row.get("claim_gate")
        if isinstance(gate, dict) and not bool(gate.get("accepted", False)):
            target_color = str(row.get("target_color", "")).lower()
            if target_color and target_color not in claimed:
                for section in (
                    "standard",
                    "near",
                    "contact",
                    "stalled_visual",
                    "success_proxy",
                ):
                    detail = gate.get(section)
                    if not isinstance(detail, dict):
                        continue
                    for reason in detail.get("reject_reasons", []) or []:
                        key = f"{section}:{reason}"
                        active_claim_rejects[key] = active_claim_rejects.get(key, 0) + 1

    span_x = (max(xs) - min(xs)) if xs else 0.0
    span_y = (max(ys) - min(ys)) if ys else 0.0
    return {
        "min_claims": int(min_claims),
        "active": bool(active),
        "start_tick": start_tick,
        "ticks": 0 if start_tick is None else max(0, len(log) - int(start_tick)),
        "start_xy": None if start_xy is None else [_round_float(v, 3) for v in start_xy],
        "final_xy": None if last_xy is None else [_round_float(v, 3) for v in last_xy],
        "xy_span_m": _round_float(float(math.hypot(span_x, span_y)), 3),
        "claimed_at_start_or_later": sorted(claimed),
        "remaining_first_seen_tick": remaining_seen,
        "remaining_area_max": {
            k: _round_float(v, 4) for k, v in sorted(remaining_area_max.items())
        },
        "remaining_mem_conf_max": {
            k: _round_float(v, 4) for k, v in sorted(remaining_mem_conf_max.items())
        },
        "remaining_read_score_max": {
            k: _round_float(v, 4) for k, v in sorted(remaining_read_score_max.items())
        },
        "state_counts": dict(sorted(state_counts.items())),
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "wall_requested_counts": dict(sorted(wall_requested_counts.items())),
        "wall_selected_counts": dict(sorted(wall_selected_counts.items())),
        "wall_vetoes": int(wall_vetoes),
        "stalled_ticks": int(stalled_ticks),
        "hard_stalled_ticks": int(hard_stalled_ticks),
        "active_claim_rejects": dict(sorted(active_claim_rejects.items())),
    }


def _roll_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[-4], q[-3], q[-2], q[-1]
    return float(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def _pitch_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[-4], q[-3], q[-2], q[-1]
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        return float(math.copysign(math.pi / 2.0, sinp))
    return float(math.asin(sinp))


def _stall_penalty_family(primitive: str) -> tuple[str, ...]:
    if primitive in ("arc_left", "arc_right"):
        return (primitive, "forward_slow", "forward_medium", "forward_fast")
    if primitive in ("forward_slow", "forward_medium", "forward_fast"):
        return ("forward_slow", "forward_medium", "forward_fast")
    if primitive == "backward":
        return ("backward",)
    return (primitive,)


def _make_escape_plan(last_primitive: str, blocks: int) -> list[str]:
    count = max(1, int(blocks))
    turn = "yaw_right" if last_primitive in ("yaw_left", "arc_left") else "yaw_left"
    if count == 1:
        return ["backward"]
    return (["backward"] * (count - 1) + [turn])[:count]


def _proprio_contact_tick_features(
    *,
    displacement_m: float,
    post_yaw: float,
    post_roll: float,
    post_pitch: float,
    post_z: float,
    post_tip_rad: float,
    primitive: str,
    requested_primitive: str | None,
    prev_yaw: float | None,
    prev_z: float | None,
) -> np.ndarray:
    """Nonprivileged proprioceptive features, parity with the offline builder.

    Values are rounded to the same precision the result log stores so runtime
    inference matches the training distribution built from replayed logs.
    """
    from build_go2_proprio_contact_dataset import (
        FEATURE_DIM,
        NOMINAL_ABS_DYAW,
        NOMINAL_DISP_M,
        PRIMITIVE_INDEX,
        PRIMITIVES,
    )

    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    disp = float(np.clip(round(float(displacement_m), 3), 0.0, 0.3))
    features[0] = disp
    dyaw = 0.0
    if prev_yaw is not None:
        dyaw = float(
            np.clip(
                math.atan2(
                    math.sin(round(float(post_yaw), 3) - float(prev_yaw)),
                    math.cos(round(float(post_yaw), 3) - float(prev_yaw)),
                ),
                -0.6,
                0.6,
            )
        )
        features[1] = dyaw
        if prev_z is not None:
            features[4] = float(np.clip(round(float(post_z), 4) - float(prev_z), -0.1, 0.1))
    features[2] = float(np.clip(round(float(post_roll), 4), -0.8, 0.8))
    features[3] = float(np.clip(round(float(post_pitch), 4), -0.8, 0.8))
    features[5] = (
        1.0
        if requested_primitive is not None and str(requested_primitive) != str(primitive)
        else 0.0
    )
    features[6] = float(np.clip(round(float(post_tip_rad), 4), 0.0, 1.0))
    features[7] = float(NOMINAL_DISP_M.get(str(primitive), 0.05)) - disp
    features[8] = float(NOMINAL_ABS_DYAW.get(str(primitive), 0.05)) - abs(dyaw)
    features[9 + PRIMITIVE_INDEX.get(str(primitive), len(PRIMITIVES))] = 1.0
    return features


def _primitive_turns_toward_bearing(primitive: str, bearing: float | None) -> bool:
    if bearing is None:
        return False
    if primitive in ("arc_left", "yaw_left"):
        return float(bearing) > 0.0
    if primitive in ("arc_right", "yaw_right"):
        return float(bearing) < 0.0
    return False


def _body_probe_clearance(
    grid: InflatedOccupancyGrid,
    xy: np.ndarray | tuple[float, float],
    yaw: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float = 0.03,
) -> float:
    """Minimum explicit footprint-point clearance over a compact Go2 probe.

    The grid itself is inflated for centerline planning. Once we sample the
    robot's front/shoulder points explicitly, subtracting that same inflation
    again double-counts body size and makes narrow-but-traversable corridors
    look blocked. Use obstacle clearance here and keep a small probe margin.
    """

    x = float(xy[0])
    y = float(xy[1])
    fx, fy = math.cos(yaw), math.sin(yaw)
    lx, ly = -fy, fx
    probes = (
        (0.0, 0.0),
        (body_forward_m, 0.0),
        (body_forward_m, body_half_width_m),
        (body_forward_m, -body_half_width_m),
        (0.0, body_half_width_m),
        (0.0, -body_half_width_m),
    )
    min_clearance = float("inf")
    for forward, lateral in probes:
        px = x + forward * fx + lateral * lx
        py = y + forward * fy + lateral * ly
        min_clearance = min(
            min_clearance,
            grid.obstacle_clearance_m((px, py)) - float(body_probe_margin_m),
        )
    return float(min_clearance)


def _swept_forward_body_clearance(
    grid: InflatedOccupancyGrid,
    xy: np.ndarray | tuple[float, float],
    yaw: float,
    *,
    distance_m: float,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float = 0.03,
    step_m: float = 0.05,
) -> float:
    x = float(xy[0])
    y = float(xy[1])
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    distance = float(distance_m)
    steps = max(1, int(math.ceil(abs(distance) / max(0.01, float(step_m)))))
    min_clearance = float("inf")
    for i in range(1, steps + 1):
        t = distance * (float(i) / float(steps))
        min_clearance = min(
            min_clearance,
            _body_probe_clearance(
                grid,
                (x + fx * t, y + fy * t),
                yaw,
                body_forward_m=body_forward_m,
                body_half_width_m=body_half_width_m,
                body_probe_margin_m=body_probe_margin_m,
            ),
        )
    return float(min_clearance)


def _primitive_clearance_report(
    registry: PrimitiveRegistry,
    primitive: str,
    pos_xy: np.ndarray | tuple[float, float],
    yaw: float,
    grid: InflatedOccupancyGrid,
    command_dt_s: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
    min_clearance_m: float,
) -> dict[str, Any]:
    block = expand_primitive_to_block(registry, primitive)
    x = float(pos_xy[0])
    y = float(pos_xy[1])
    heading = float(yaw)
    min_clearance = float("inf")
    feasible = 0
    samples = 0
    predicted_path_m = 0.0
    for vx, vy, yaw_rate in block:
        vx = float(vx)
        vy = float(vy)
        yaw_rate = float(yaw_rate)
        dx = (math.cos(heading) * vx - math.sin(heading) * vy) * command_dt_s
        dy = (math.sin(heading) * vx + math.cos(heading) * vy) * command_dt_s
        x += dx
        y += dy
        heading = wrap_angle_pi(heading + yaw_rate * command_dt_s)
        predicted_path_m += math.hypot(dx, dy)
        clearance = _body_probe_clearance(
            grid, (x, y), heading,
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
        )
        min_clearance = min(min_clearance, clearance)
        samples += 1
        if clearance >= min_clearance_m and grid.is_free((x, y)):
            feasible += 1
    feasible_fraction = float(feasible / max(1, samples))
    if not math.isfinite(min_clearance):
        min_clearance = -1.0
    blocked = feasible_fraction < 1.0 or min_clearance < min_clearance_m
    return {
        "primitive": primitive,
        "min_clearance_m": float(min_clearance),
        "feasible_fraction": feasible_fraction,
        "blocked": bool(blocked),
        "predicted_path_m": float(predicted_path_m),
    }


def _turn_preference(primitive: str, bearing: float | None) -> float:
    if bearing is None or abs(float(bearing)) < 0.05:
        return 0.0
    left = primitive.endswith("left")
    right = primitive.endswith("right")
    if not (left or right):
        return 0.0
    turns_toward = (bearing > 0.0 and left) or (bearing < 0.0 and right)
    return 0.12 if turns_toward else -0.06


def _score_clearance_candidate(report: dict[str, Any], bearing: float | None) -> float:
    primitive = str(report["primitive"])
    clearance = max(-0.4, min(0.6, float(report["min_clearance_m"])))
    score = 2.0 * float(report["feasible_fraction"]) + 0.8 * clearance
    score += _turn_preference(primitive, bearing)
    if primitive == "backward":
        score -= 0.02
    elif primitive in _TURN_PRIMITIVES:
        score -= 0.16
    elif primitive == "hold":
        score -= 0.50
    return score


def _unique_primitives(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _parse_xy_waypoints(spec: str | None) -> list[tuple[float, float]]:
    if not spec:
        return []
    out: list[tuple[float, float]] = []
    for raw_item in str(spec).split(";"):
        item = raw_item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"bad waypoint '{item}', expected x,y")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _parse_color_float_map(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad color:value item '{item}'")
        color, value = item.split(":", 1)
        color = color.strip()
        if not color:
            raise ValueError(f"bad empty color in '{item}'")
        out[color] = float(value.strip())
    return out


def _parse_primitive_float_map(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    allowed = set(_LEARNED_LOCAL_POLICY_PRIMITIVES) | {"forward_slow", "hold"}
    out: dict[str, float] = {}
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad primitive:value item '{item}'")
        primitive, value = item.split(":", 1)
        primitive = primitive.strip()
        if not primitive:
            raise ValueError(f"bad empty primitive in '{item}'")
        if primitive not in allowed:
            raise ValueError(f"unsupported primitive in '{item}'")
        out[primitive] = float(value.strip())
    return out


def _parse_color_int_map(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}
    out: dict[str, int] = {}
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad color:value item '{item}'")
        color, value = item.split(":", 1)
        color = color.strip()
        if not color:
            raise ValueError(f"bad empty color in '{item}'")
        out[color] = int(value.strip())
    return out


def _wall_guard_select(
    *,
    requested: str,
    pos_xy: np.ndarray | tuple[float, float],
    yaw: float,
    bearing: float | None,
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid,
    command_dt_s: float,
    enabled: bool,
    min_clearance_m: float,
    feasible_threshold: float,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
    force_escape: bool = False,
) -> tuple[str, dict[str, Any]]:
    requested_report = _primitive_clearance_report(
        registry, requested, pos_xy, yaw, grid, command_dt_s,
        body_forward_m=body_forward_m, body_half_width_m=body_half_width_m,
        body_probe_margin_m=body_probe_margin_m,
        min_clearance_m=min_clearance_m,
    )
    should_veto = (
        enabled
        and (force_escape or requested in _FORWARD_PRIMITIVES)
        and (
            force_escape
            or bool(requested_report["blocked"])
            or float(requested_report["feasible_fraction"]) < feasible_threshold
        )
    )
    candidate_names = [requested]
    if should_veto:
        if bearing is not None and bearing > 0.0:
            turns = ["arc_left", "yaw_left", "arc_right", "yaw_right"]
        elif bearing is not None and bearing < 0.0:
            turns = ["arc_right", "yaw_right", "arc_left", "yaw_left"]
        else:
            turns = ["arc_left", "arc_right", "yaw_left", "yaw_right"]
        candidate_names = _unique_primitives([requested, "forward_slow", *turns, "backward", "hold"])
    reports = [
        _primitive_clearance_report(
            registry, name, pos_xy, yaw, grid, command_dt_s,
            body_forward_m=body_forward_m, body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
            min_clearance_m=min_clearance_m,
        )
        if name != requested else requested_report
        for name in candidate_names
    ]
    selected_report = requested_report
    selected = requested
    if should_veto:
        def _score(report: dict[str, Any]) -> float:
            score = _score_clearance_candidate(report, bearing)
            if force_escape and str(report["primitive"]) in _FORWARD_PRIMITIVES:
                score -= 0.7
            return score

        selected_report = max(reports, key=_score)
        selected = str(selected_report["primitive"])
    compact_reports = [
        {
            "primitive": str(r["primitive"]),
            "min_clearance_m": _round_float(float(r["min_clearance_m"])),
            "feasible_fraction": _round_float(float(r["feasible_fraction"])),
            "blocked": bool(r["blocked"]),
        }
        for r in reports
    ]
    guard = {
        "enabled": bool(enabled),
        "requested": requested,
        "selected": selected,
        "vetoed": bool(selected != requested),
        "force_escape": bool(force_escape),
        "requested_min_clearance_m": _round_float(float(requested_report["min_clearance_m"])),
        "requested_feasible_fraction": _round_float(float(requested_report["feasible_fraction"])),
        "requested_blocked": bool(requested_report["blocked"]),
        "selected_min_clearance_m": _round_float(float(selected_report["min_clearance_m"])),
        "selected_feasible_fraction": _round_float(float(selected_report["feasible_fraction"])),
        "selected_blocked": bool(selected_report["blocked"]),
        "candidates": compact_reports,
    }
    return selected, guard


def _geometry_clearance_veto_select(
    *,
    selected: str,
    requested: str,
    pos_xy: np.ndarray | tuple[float, float],
    yaw: float,
    bearing: float | None,
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid,
    command_dt_s: float,
    enabled: bool,
    min_clearance_m: float,
    feasible_threshold: float,
    body_forward_m: float,
    body_half_width_m: float,
    body_probe_margin_m: float,
    selected_primitives: set[str] | None,
    replacement_primitives: set[str] | None,
    blocked_fallback_primitives: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    selected_report = _primitive_clearance_report(
        registry,
        selected,
        pos_xy,
        yaw,
        grid,
        command_dt_s,
        body_forward_m=body_forward_m,
        body_half_width_m=body_half_width_m,
        body_probe_margin_m=body_probe_margin_m,
        min_clearance_m=min_clearance_m,
    )
    active_selected_primitives = set(
        selected_primitives or _FORWARD_PRIMITIVES.union(_TURN_PRIMITIVES)
    )
    should_veto = bool(
        enabled
        and selected in active_selected_primitives
        and (
            bool(selected_report["blocked"])
            or float(selected_report["feasible_fraction"]) < float(feasible_threshold)
        )
    )
    if bearing is not None and bearing > 0.0:
        turns = ["arc_left", "yaw_left", "arc_right", "yaw_right"]
    elif bearing is not None and bearing < 0.0:
        turns = ["arc_right", "yaw_right", "arc_left", "yaw_left"]
    else:
        turns = ["arc_left", "arc_right", "yaw_left", "yaw_right"]
    default_replacements = ["forward_slow", *turns, "backward", "hold"]
    allowed_replacements = set(
        replacement_primitives or default_replacements
    )
    candidate_names = [selected]
    if should_veto:
        candidate_names = _unique_primitives(
            [selected, *[name for name in default_replacements if name in allowed_replacements]]
        )
    reports = [
        selected_report
        if name == selected
        else _primitive_clearance_report(
            registry,
            name,
            pos_xy,
            yaw,
            grid,
            command_dt_s,
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            body_probe_margin_m=body_probe_margin_m,
            min_clearance_m=min_clearance_m,
        )
        for name in candidate_names
    ]
    replacement = selected
    replacement_report = selected_report
    blocked_fallback_active = False
    if should_veto:
        replacement_report = max(reports, key=lambda report: _score_clearance_candidate(report, bearing))
        replacement = str(replacement_report["primitive"])
        if blocked_fallback_primitives and all(bool(report["blocked"]) for report in reports):
            reports_by_name = {str(report["primitive"]): report for report in reports}
            for name in blocked_fallback_primitives:
                if name in reports_by_name:
                    replacement = name
                    replacement_report = reports_by_name[name]
                    blocked_fallback_active = True
                    break
    compact_reports = [
        {
            "primitive": str(report["primitive"]),
            "min_clearance_m": _round_float(float(report["min_clearance_m"])),
            "feasible_fraction": _round_float(float(report["feasible_fraction"])),
            "blocked": bool(report["blocked"]),
        }
        for report in reports
    ]
    return replacement, {
        "active": bool(should_veto and replacement != selected),
        "enabled": bool(enabled),
        "from": selected,
        "to": replacement,
        "requested": requested,
        "min_clearance_m": _round_float(float(min_clearance_m), 4),
        "feasible_threshold": _round_float(float(feasible_threshold), 4),
        "selected_min_clearance_m": _round_float(float(selected_report["min_clearance_m"])),
        "selected_feasible_fraction": _round_float(float(selected_report["feasible_fraction"])),
        "selected_blocked": bool(selected_report["blocked"]),
        "replacement_min_clearance_m": _round_float(float(replacement_report["min_clearance_m"])),
        "replacement_feasible_fraction": _round_float(float(replacement_report["feasible_fraction"])),
        "replacement_blocked": bool(replacement_report["blocked"]),
        "blocked_fallback_active": bool(blocked_fallback_active),
        "blocked_fallback_primitives": list(blocked_fallback_primitives or []),
        "candidates": compact_reports,
    }


def _learned_front_guard_select(
    *,
    requested: str,
    bearing: float | None,
    enabled: bool,
    front_blocked_prob: float | None,
    threshold: float,
    force_escape: bool = False,
) -> tuple[str, dict[str, Any]]:
    prob = None if front_blocked_prob is None else float(front_blocked_prob)
    requested_blocked = bool(prob is not None and prob >= float(threshold))
    should_veto = (
        enabled
        and (force_escape or requested in _FORWARD_PRIMITIVES)
        and (force_escape or requested_blocked)
    )
    selected = requested
    candidates = [{"primitive": requested, "front_blocked_prob": _round_float(prob), "blocked": requested_blocked}]
    if should_veto:
        if force_escape:
            selected = "backward"
        elif requested == "arc_right":
            selected = "yaw_right"
        elif requested == "arc_left":
            selected = "yaw_left"
        elif bearing is not None and bearing < 0.0:
            selected = "yaw_right"
        else:
            selected = "yaw_left"
        candidates.append({"primitive": selected, "front_blocked_prob": None, "blocked": False})
    guard = {
        "enabled": bool(enabled),
        "source": "learned_front_blocked",
        "requested": requested,
        "selected": selected,
        "vetoed": bool(selected != requested),
        "force_escape": bool(force_escape),
        "front_blocked_prob": _round_float(prob, 4),
        "threshold": _round_float(float(threshold), 4),
        "requested_min_clearance_m": None,
        "requested_feasible_fraction": None,
        "requested_blocked": requested_blocked,
        "selected_min_clearance_m": None,
        "selected_feasible_fraction": None,
        "selected_blocked": False if selected != requested else requested_blocked,
        "candidates": candidates,
    }
    return selected, guard


def _predict_primitive_outcomes(
    head: Go2PrimitiveOutcomeHead,
    latent: torch.Tensor,
    *,
    primitive_vocab: list[str],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    primitive_count = len(primitive_vocab)
    idx = torch.arange(primitive_count, device=device)
    one_hot = F.one_hot(idx, num_classes=primitive_count).float()
    latents = latent.to(device).expand(primitive_count, -1)
    with torch.no_grad():
        blocked_logits, progress_m = head(latents, one_hot)
    probs = torch.sigmoid(blocked_logits).detach().cpu().tolist()
    progress = progress_m.detach().cpu().clamp_min(0.0).tolist()
    return {
        name: {
            "blocked_prob": float(probs[i]),
            "progress_m": float(progress[i]),
        }
        for i, name in enumerate(primitive_vocab)
    }


def _fuse_clearance_outcomes(
    outcomes: dict[str, dict[str, float]] | None,
    clearance: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    if not outcomes:
        return outcomes
    if not clearance:
        return outcomes
    fused: dict[str, dict[str, float]] = {}
    for name, prediction in outcomes.items():
        item = dict(prediction)
        clearance_prediction = clearance.get(name)
        if clearance_prediction is not None:
            outcome_prob = float(item.get("blocked_prob", 1.0))
            clearance_prob = float(clearance_prediction.get("blocked_prob", 1.0))
            item["outcome_blocked_prob"] = outcome_prob
            item["clearance_blocked_prob"] = clearance_prob
        fused[name] = item
    return fused


def _score_learned_action_candidate(
    primitive: str,
    prediction: dict[str, float],
    *,
    bearing: float | None,
    target_area: float | None,
    requested: str,
    force_escape: bool,
    blocked_threshold: float,
    blocked_weight: float,
    progress_weight: float,
    requested_bonus: float,
    turn_progress_scale: float,
    body_clearance_enabled: bool = False,
    body_clearance_target_area: float = 2.0,
    body_clearance_arc_penalty: float = 0.0,
    body_clearance_yaw_penalty_weight: float = 0.0,
    body_clearance_prob_floor: float = 0.35,
    body_clearance_prob_weight: float = 0.0,
    body_clearance_near_forward_prob_floor: float | None = None,
    body_clearance_near_forward_prob_weight: float | None = None,
    body_clearance_near_yaw_prob_floor: float | None = None,
    body_clearance_yaw_always: bool = False,
    body_clearance_min_area: float | None = None,
    forward_progress_floor: float | None = None,
    forward_progress_floor_min_blocked_prob: float | None = None,
    forward_progress_floor_force_below: float | None = None,
    forward_progress_floor_penalty: float = 0.0,
    runtime_penalties: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    blocked_prob = float(prediction.get("blocked_prob", 1.0))
    progress_m = float(prediction.get("progress_m", 0.0))
    if primitive in _TURN_PRIMITIVES or primitive == "hold":
        progress_m *= max(0.0, float(turn_progress_scale))
    score = float(progress_weight) * progress_m - float(blocked_weight) * blocked_prob
    score += _turn_preference(primitive, bearing)
    if primitive == requested:
        score += float(requested_bonus)
    if runtime_penalties and primitive in runtime_penalties:
        score -= float(runtime_penalties[primitive])
    if primitive == "backward":
        score += 0.25 if force_escape else -0.55
    elif primitive in _TURN_PRIMITIVES:
        score -= 0.32 if requested in _FORWARD_PRIMITIVES and not force_escape else 0.10
    elif primitive == "hold":
        score -= 0.75
    if bearing is not None:
        abs_bearing = abs(float(bearing))
        if primitive == "forward_medium" and abs_bearing > 0.55:
            score -= 0.35
        if primitive in _TURN_PRIMITIVES and abs_bearing < 0.18:
            score -= 0.30
        if primitive == "arc_left":
            score += 0.18 if bearing > 0.18 else (-0.45 if bearing < -0.18 else 0.0)
        if primitive == "arc_right":
            score += 0.18 if bearing < -0.18 else (-0.45 if bearing > 0.18 else 0.0)
    if primitive in _FORWARD_PRIMITIVES and blocked_prob >= float(blocked_threshold):
        score -= 0.45
    if force_escape and primitive in _FORWARD_PRIMITIVES:
        score -= 0.70
    progress_floor_penalty = 0.0
    if (
        forward_progress_floor is not None
        and primitive in _FORWARD_PRIMITIVES
        and progress_m < float(forward_progress_floor)
        and (
            (
                forward_progress_floor_force_below is not None
                and progress_m < float(forward_progress_floor_force_below)
            )
            or
            forward_progress_floor_min_blocked_prob is None
            or blocked_prob >= float(forward_progress_floor_min_blocked_prob)
        )
    ):
        progress_floor_penalty = float(forward_progress_floor_penalty)
        score -= progress_floor_penalty
    body_clearance_penalty = 0.0
    body_clearance_area_active = (
        body_clearance_min_area is None
        or (
            target_area is not None
            and float(target_area) >= float(body_clearance_min_area)
        )
    )
    near_target_for_body = bool(
        target_area is not None
        and float(target_area) >= float(body_clearance_target_area)
    )
    if (
        body_clearance_enabled
        and body_clearance_area_active
        and (
            primitive in _FORWARD_PRIMITIVES
            or primitive == "backward"
            or primitive == "hold"
            or primitive in ("arc_left", "arc_right")
        )
    ):
        body_blocked_prob = float(prediction.get("clearance_blocked_prob", blocked_prob))
        excess = max(0.0, body_blocked_prob - float(body_clearance_prob_floor))
        body_clearance_penalty += float(body_clearance_prob_weight) * excess
        if near_target_for_body and body_clearance_near_forward_prob_floor is not None:
            near_forward_weight = (
                float(body_clearance_prob_weight)
                if body_clearance_near_forward_prob_weight is None
                else float(body_clearance_near_forward_prob_weight)
            )
            near_excess = max(
                0.0,
                body_blocked_prob - float(body_clearance_near_forward_prob_floor),
            )
            body_clearance_penalty += near_forward_weight * near_excess
        if (
            primitive in ("arc_left", "arc_right")
            and near_target_for_body
        ):
            # Near a visible target, a same-direction arc can sweep the Go2's flank
            # across an inside corner even when the camera ray itself is clear.
            body_clearance_penalty += float(body_clearance_arc_penalty)
    if (
        body_clearance_enabled
        and body_clearance_area_active
        and primitive in _TURN_PRIMITIVES
        and (near_target_for_body or bool(body_clearance_yaw_always))
    ):
        body_blocked_prob = float(prediction.get("clearance_blocked_prob", blocked_prob))
        yaw_floor = (
            float(body_clearance_prob_floor)
            if body_clearance_near_yaw_prob_floor is None
            else float(body_clearance_near_yaw_prob_floor)
        )
        excess = max(0.0, body_blocked_prob - yaw_floor)
        body_clearance_penalty += float(body_clearance_yaw_penalty_weight) * excess
    score -= body_clearance_penalty
    return float(score), float(body_clearance_penalty), float(progress_floor_penalty)


def _prediction_alias_for_primitive(
    primitive: str,
    predictions: dict[str, dict[str, float]] | None,
) -> str | None:
    """Map execution-only speed variants onto the learned action vocabulary."""

    if not predictions:
        return None
    if primitive in predictions:
        return primitive
    if primitive in ("forward_slow", "forward_fast") and "forward_medium" in predictions:
        return "forward_medium"
    return None


def _select_aux_clearance_veto(
    *,
    selected: str,
    primary_predictions: dict[str, dict[str, float]] | None,
    aux_clearance_predictions: dict[str, dict[str, float]] | None,
    candidate_primitives: list[str],
    enabled: bool,
    aux_veto_prob: float,
    primary_max_prob: float,
    aux_veto_margin: float,
    aux_replacement_cap: float,
    selected_primitives: set[str] | None,
    replacement_primitives: set[str] | None,
) -> tuple[str, dict[str, Any]]:
    if (
        not enabled
        or not primary_predictions
        or not aux_clearance_predictions
        or float(aux_veto_prob) > 1.0
        or selected not in set(selected_primitives or _FORWARD_PRIMITIVES.union(_TURN_PRIMITIVES))
    ):
        return selected, {"active": False}
    selected_primary_alias = _prediction_alias_for_primitive(selected, primary_predictions)
    selected_aux_alias = _prediction_alias_for_primitive(selected, aux_clearance_predictions)
    selected_primary = (
        primary_predictions.get(selected_primary_alias)
        if selected_primary_alias is not None
        else None
    )
    selected_aux = (
        aux_clearance_predictions.get(selected_aux_alias)
        if selected_aux_alias is not None
        else None
    )
    if selected_primary is None or selected_aux is None:
        return selected, {"active": False}
    selected_primary_prob = selected_primary.get("clearance_blocked_prob")
    selected_aux_prob = selected_aux.get("blocked_prob")
    if selected_primary_prob is None or selected_aux_prob is None:
        return selected, {"active": False}
    if float(selected_primary_prob) > float(primary_max_prob):
        return selected, {
            "active": False,
            "suppressed": "primary_above_max",
            "selected": selected,
            "selected_primary_clearance_prob": _round_float(selected_primary_prob, 4),
            "selected_aux_clearance_prob": _round_float(selected_aux_prob, 4),
        }
    if float(selected_aux_prob) < float(aux_veto_prob):
        return selected, {
            "active": False,
            "suppressed": "aux_below_threshold",
            "selected": selected,
            "selected_primary_clearance_prob": _round_float(selected_primary_prob, 4),
            "selected_aux_clearance_prob": _round_float(selected_aux_prob, 4),
        }
    allowed_replacements = set(
        replacement_primitives
        or {"backward", "yaw_left", "yaw_right", "arc_left", "arc_right", "hold"}
    )
    candidates = _unique_primitives([*candidate_primitives, *sorted(allowed_replacements)])
    scored: list[tuple[float, str, float, float | None, float | None]] = []
    for name in candidates:
        if name == selected or name not in allowed_replacements:
            continue
        aux_alias = _prediction_alias_for_primitive(name, aux_clearance_predictions)
        if aux_alias is None:
            continue
        aux_pred = aux_clearance_predictions.get(aux_alias)
        if aux_pred is None or aux_pred.get("blocked_prob") is None:
            continue
        aux_prob = float(aux_pred["blocked_prob"])
        if aux_prob > float(selected_aux_prob) - float(aux_veto_margin):
            continue
        if aux_prob > float(aux_replacement_cap):
            continue
        primary_alias = _prediction_alias_for_primitive(name, primary_predictions)
        primary_pred = (
            primary_predictions.get(primary_alias)
            if primary_alias is not None
            else None
        )
        primary_blocked = (
            None
            if primary_pred is None or primary_pred.get("blocked_prob") is None
            else float(primary_pred["blocked_prob"])
        )
        progress_m = (
            None
            if primary_pred is None or primary_pred.get("progress_m") is None
            else float(primary_pred["progress_m"])
        )
        if name in _TURN_PRIMITIVES:
            primitive_bias = 0.0
        elif name == "backward":
            primitive_bias = 0.08
        elif name == "hold":
            primitive_bias = 0.18
        else:
            primitive_bias = 0.12
        score = aux_prob + 0.12 * float(primary_blocked or 0.0) + primitive_bias
        if progress_m is not None:
            score -= 0.02 * progress_m
        scored.append((score, name, aux_prob, primary_blocked, progress_m))
    if not scored:
        return selected, {
            "active": False,
            "suppressed": "no_replacement",
            "selected": selected,
            "selected_primary_clearance_prob": _round_float(selected_primary_prob, 4),
            "selected_aux_clearance_prob": _round_float(selected_aux_prob, 4),
            "aux_veto_prob": _round_float(float(aux_veto_prob), 4),
            "primary_max_prob": _round_float(float(primary_max_prob), 4),
            "aux_replacement_cap": _round_float(float(aux_replacement_cap), 4),
        }
    scored.sort(key=lambda item: item[0])
    score, replacement, replacement_aux_prob, replacement_primary_blocked, replacement_progress = scored[0]
    return replacement, {
        "active": True,
        "from": selected,
        "to": replacement,
        "selected_primary_clearance_prob": _round_float(selected_primary_prob, 4),
        "selected_aux_clearance_prob": _round_float(selected_aux_prob, 4),
        "replacement_aux_clearance_prob": _round_float(replacement_aux_prob, 4),
        "replacement_primary_blocked_prob": _round_float(replacement_primary_blocked, 4),
        "replacement_progress_m": _round_float(replacement_progress, 4),
        "score": _round_float(score, 4),
        "aux_veto_prob": _round_float(float(aux_veto_prob), 4),
        "primary_max_prob": _round_float(float(primary_max_prob), 4),
        "aux_veto_margin": _round_float(float(aux_veto_margin), 4),
        "aux_replacement_cap": _round_float(float(aux_replacement_cap), 4),
    }


def _select_learned_local_explore_primitive(
    *,
    tick: int,
    requested: str,
    predictions: dict[str, dict[str, float]] | None,
    scan_interval: int,
    scan_len: int,
    scan_primitive: str,
    blocked_threshold: float,
    blocked_weight: float,
    progress_weight: float,
    requested_bonus: float,
    turn_progress_scale: float,
    forward_progress_floor: float | None,
    forward_progress_floor_min_blocked_prob: float | None,
    forward_progress_floor_force_below: float | None,
    forward_progress_floor_penalty: float,
    turn_balance: int = 0,
    turn_run: int = 0,
    translation_pressure_after: int = 4,
    translation_pressure_bonus: float = 0.9,
    translation_pressure_yaw_penalty: float = 0.45,
    translation_pressure_max_risk: float = 0.72,
    translation_pressure_min_progress: float = 0.02,
) -> tuple[str, dict[str, Any]]:
    """Choose a route-free local explore primitive from learned outcome scores."""

    if not predictions:
        return requested, {"enabled": False, "reason": "missing_predictions"}

    interval = max(1, int(scan_interval))
    scan_ticks = max(0, int(scan_len))
    scan_active = bool(scan_ticks > 0 and int(tick) % interval < scan_ticks)
    sweep_index = int(tick) // interval
    turn_left = (sweep_index % 2) == 0
    preferred_arc = "arc_left" if turn_left else "arc_right"
    preferred_yaw = "yaw_left" if turn_left else "yaw_right"
    opposite_arc = "arc_right" if turn_left else "arc_left"
    opposite_yaw = "yaw_right" if turn_left else "yaw_left"
    balance_mag = min(1.6, 0.08 * abs(int(turn_balance)))
    translation_pressure_active = int(turn_run) >= max(0, int(translation_pressure_after))

    candidate_names = (
        "forward_fast",
        "forward_medium",
        "arc_left",
        "arc_right",
        "yaw_left",
        "yaw_right",
        "backward",
        "hold",
    )
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for primitive in candidate_names:
        alias = _prediction_alias_for_primitive(primitive, predictions)
        if alias is None:
            continue
        pred = predictions.get(alias)
        if pred is None:
            continue
        blocked_prob = float(pred.get("blocked_prob", 1.0))
        clearance_prob = pred.get("clearance_blocked_prob")
        risk = max(
            blocked_prob,
            blocked_prob if clearance_prob is None else float(clearance_prob),
        )
        progress_m = float(pred.get("progress_m", 0.0))
        effective_progress = progress_m
        if primitive in _TURN_PRIMITIVES or primitive == "hold":
            effective_progress *= max(0.0, float(turn_progress_scale))

        score = float(progress_weight) * effective_progress - float(blocked_weight) * risk
        if primitive == requested:
            score += float(requested_bonus)
        if primitive == "backward":
            score -= 0.65
        elif primitive == "hold":
            score -= 0.95
        elif primitive in _TURN_PRIMITIVES:
            score -= 0.18

        if scan_active:
            if str(scan_primitive) == "arc":
                if primitive == preferred_arc:
                    score += 1.25
                elif primitive == preferred_yaw:
                    score += 0.55
                elif primitive == opposite_arc:
                    score += 0.20
            else:
                if primitive == preferred_yaw:
                    score += 1.25
                elif primitive == preferred_arc:
                    score += 0.50
                elif primitive == opposite_yaw:
                    score += 0.20
        else:
            if primitive in ("forward_fast", "forward_medium"):
                score += 0.25
            elif primitive in ("arc_left", "arc_right"):
                score += 0.08

        if turn_balance > 0:
            if primitive in ("yaw_left", "arc_left"):
                score -= balance_mag
            elif primitive in ("yaw_right", "arc_right"):
                score += balance_mag
        elif turn_balance < 0:
            if primitive in ("yaw_right", "arc_right"):
                score -= balance_mag
            elif primitive in ("yaw_left", "arc_left"):
                score += balance_mag

        if translation_pressure_active:
            translation_candidate = bool(
                primitive in _FORWARD_PRIMITIVES
                and risk <= float(translation_pressure_max_risk)
                and progress_m >= float(translation_pressure_min_progress)
            )
            if translation_candidate:
                pressure_bonus = float(translation_pressure_bonus)
                if primitive in ("arc_left", "arc_right"):
                    pressure_bonus *= 1.15
                score += pressure_bonus
            elif primitive in _TURN_PRIMITIVES:
                score -= float(translation_pressure_yaw_penalty)

        low_progress = False
        if (
            forward_progress_floor is not None
            and primitive in _FORWARD_PRIMITIVES
            and progress_m < float(forward_progress_floor)
            and (
                (
                    forward_progress_floor_force_below is not None
                    and progress_m < float(forward_progress_floor_force_below)
                )
                or forward_progress_floor_min_blocked_prob is None
                or blocked_prob >= float(forward_progress_floor_min_blocked_prob)
            )
        ):
            low_progress = True
            score -= float(forward_progress_floor_penalty)
        if primitive in _FORWARD_PRIMITIVES and blocked_prob >= float(blocked_threshold):
            score -= 0.55

        scored.append(
            (
                float(score),
                primitive,
                {
                    "primitive": primitive,
                    "prediction_alias": alias,
                    "blocked_prob": _round_float(blocked_prob, 4),
                    "clearance_blocked_prob": _round_float(clearance_prob, 4),
                    "risk": _round_float(risk, 4),
                    "progress_m": _round_float(progress_m, 4),
                    "score": _round_float(score, 4),
                    "low_progress": bool(low_progress),
                },
            )
        )

    if not scored:
        return requested, {"enabled": False, "reason": "no_scored_candidates"}
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[0][1]
    return selected, {
        "enabled": True,
        "selected": selected,
        "requested_before": requested,
        "scan_active": bool(scan_active),
        "scan_primitive": str(scan_primitive),
        "preferred_turn": "left" if turn_left else "right",
        "turn_balance_before": int(turn_balance),
        "turn_run_before": int(turn_run),
        "translation_pressure_active": bool(translation_pressure_active),
        "translation_pressure_after": int(translation_pressure_after),
        "translation_pressure_max_risk": _round_float(float(translation_pressure_max_risk), 4),
        "translation_pressure_min_progress": _round_float(float(translation_pressure_min_progress), 4),
        "top_candidates": [item[2] for item in scored[:5]],
    }


def _outcome_stats_for_primitive(
    predictions: dict[str, dict[str, float]] | None,
    primitive: str,
) -> tuple[float, float, float | None, str | None]:
    alias = _prediction_alias_for_primitive(primitive, predictions)
    if alias is None or not predictions or alias not in predictions:
        return 1.0, 0.0, None, alias
    pred = predictions[alias]
    blocked_prob = float(pred.get("blocked_prob", 1.0))
    clearance_prob = pred.get("clearance_blocked_prob")
    risk = max(
        blocked_prob,
        blocked_prob if clearance_prob is None else float(clearance_prob),
    )
    return float(risk), float(pred.get("progress_m", 0.0)), clearance_prob, alias


def _select_learned_wall_follow_explore_primitive(
    *,
    tick: int,
    requested: str,
    predictions: dict[str, dict[str, float]] | None,
    scan_interval: int,
    scan_len: int,
    scan_primitive: str,
    side: str,
    turn_run: int,
    safe_risk: float,
    progress_floor: float,
    turn_pressure_after: int,
) -> tuple[str, dict[str, Any]]:
    """Route-free local coverage from learned primitive outcomes.

    This is intentionally not a map or target-route follower. It keeps one wall
    side and asks the learned action-outcome head which local primitive is safe
    enough to execute from the current RGB/JEPA view.
    """

    if not predictions:
        return requested, {"enabled": False, "reason": "missing_predictions"}

    side = "left" if str(side).lower() == "left" else "right"
    side_arc = "arc_left" if side == "left" else "arc_right"
    away_arc = "arc_right" if side == "left" else "arc_left"
    side_yaw = "yaw_left" if side == "left" else "yaw_right"
    away_yaw = "yaw_right" if side == "left" else "yaw_left"
    interval = max(1, int(scan_interval))
    scan_ticks = max(0, int(scan_len))
    scan_active = bool(scan_ticks > 0 and int(tick) % interval < scan_ticks)
    scan_left = ((int(tick) // interval) % 2) == 0
    scan_arc = "arc_left" if scan_left else "arc_right"
    scan_yaw = "yaw_left" if scan_left else "yaw_right"

    def stats(name: str) -> dict[str, Any]:
        risk, progress_m, clearance_prob, alias = _outcome_stats_for_primitive(predictions, name)
        translating = name in _FORWARD_PRIMITIVES
        safe = risk <= float(safe_risk) and (not translating or progress_m >= float(progress_floor))
        return {
            "primitive": name,
            "prediction_alias": alias,
            "risk": float(risk),
            "progress_m": float(progress_m),
            "clearance_blocked_prob": clearance_prob,
            "safe": bool(safe),
        }

    names = (
        "forward_fast",
        "forward_medium",
        "arc_left",
        "arc_right",
        "yaw_left",
        "yaw_right",
        "backward",
    )
    cand = {name: stats(name) for name in names}

    def is_safe(name: str) -> bool:
        return bool(cand[name]["safe"])

    def choose_from(sequence: tuple[str, ...]) -> str:
        for name in sequence:
            if is_safe(name):
                return name
        return sequence[-1]

    reason = "wall_follow"
    selected: str
    if scan_active:
        reason = "scan"
        if str(scan_primitive) == "arc" and is_safe(scan_arc):
            selected = scan_arc
        else:
            selected = scan_yaw
    elif int(turn_run) >= max(1, int(turn_pressure_after)):
        reason = "turn_pressure"
        selected = choose_from(("forward_fast", "forward_medium", side_arc, away_arc, "backward"))
    else:
        side_safe = is_safe(side_arc)
        forward_safe = is_safe("forward_fast") or is_safe("forward_medium")
        away_safe = is_safe(away_arc)
        if side_safe:
            selected = side_arc
        elif forward_safe:
            selected = "forward_fast" if is_safe("forward_fast") else "forward_medium"
        elif away_safe:
            selected = away_arc
        elif int(turn_run) >= 3:
            reason = "blocked_escape"
            selected = "backward"
        else:
            reason = "blocked_yaw"
            selected = away_yaw

    ranked = sorted(
        cand.values(),
        key=lambda item: (
            bool(item["safe"]),
            float(item["progress_m"]) - 0.25 * float(item["risk"]),
        ),
        reverse=True,
    )
    return selected, {
        "enabled": True,
        "selected": selected,
        "requested_before": requested,
        "side": side,
        "reason": reason,
        "scan_active": bool(scan_active),
        "turn_run_before": int(turn_run),
        "safe_risk": _round_float(float(safe_risk), 4),
        "progress_floor": _round_float(float(progress_floor), 4),
        "top_candidates": [
            {
                "primitive": item["primitive"],
                "prediction_alias": item["prediction_alias"],
                "risk": _round_float(float(item["risk"]), 4),
                "progress_m": _round_float(float(item["progress_m"]), 4),
                "clearance_blocked_prob": _round_float(item["clearance_blocked_prob"], 4),
                "safe": bool(item["safe"]),
            }
            for item in ranked[:5]
        ],
    }


def _select_learned_policy_translation_pressure_primitive(
    *,
    predictions: dict[str, dict[str, float]] | None,
    candidate_primitives: list[str],
    max_blocked_prob: float,
    min_progress_m: float,
    bearing: float | None,
) -> tuple[str | None, dict[str, Any]]:
    if not predictions:
        return None, {"enabled": False, "reason": "missing_predictions"}
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for name in candidate_primitives:
        if name not in _LEARNED_LOCAL_POLICY_PRIMITIVES:
            continue
        if name in _TURN_PRIMITIVES or name == "hold":
            continue
        alias = _prediction_alias_for_primitive(name, predictions)
        pred = predictions.get(alias) if alias is not None else None
        if pred is None:
            continue
        blocked_prob = float(pred.get("blocked_prob", 1.0))
        progress_m = float(pred.get("progress_m", 0.0))
        clearance_prob = float(pred.get("clearance_blocked_prob", blocked_prob))
        accepted = bool(
            blocked_prob <= float(max_blocked_prob)
            and clearance_prob <= float(max_blocked_prob)
            and progress_m >= float(min_progress_m)
        )
        score = progress_m - 0.45 * max(blocked_prob, clearance_prob)
        if name == "backward":
            score -= 0.18
        if bearing is not None:
            if name == "arc_left" and float(bearing) > 0.0:
                score += 0.08
            elif name == "arc_right" and float(bearing) < 0.0:
                score += 0.08
        item = {
            "primitive": name,
            "prediction_alias": alias if alias != name else None,
            "blocked_prob": _round_float(blocked_prob, 4),
            "clearance_blocked_prob": _round_float(clearance_prob, 4),
            "progress_m": _round_float(progress_m, 4),
            "accepted": bool(accepted),
            "score": _round_float(score, 4),
        }
        if accepted:
            scored.append((float(score), name, item))
    if not scored:
        return None, {
            "enabled": True,
            "selected": None,
            "reason": "no_accepted_translation",
            "max_blocked_prob": _round_float(float(max_blocked_prob), 4),
            "min_progress_m": _round_float(float(min_progress_m), 4),
        }
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], {
        "enabled": True,
        "selected": scored[0][1],
        "max_blocked_prob": _round_float(float(max_blocked_prob), 4),
        "min_progress_m": _round_float(float(min_progress_m), 4),
        "top_candidates": [item for _, _, item in scored[:5]],
    }


def _guard_candidate_for_primitive(
    wall_guard: dict[str, Any],
    primitive: str,
) -> dict[str, Any] | None:
    for candidate in wall_guard.get("candidates", ()):
        if str(candidate.get("primitive", "")) == primitive:
            return candidate
    return None


def _guard_candidate_is_blocked(
    wall_guard: dict[str, Any],
    primitive: str,
) -> bool:
    candidate = _guard_candidate_for_primitive(wall_guard, primitive)
    if candidate is None:
        return True
    if bool(candidate.get("blocked")):
        return True
    threshold = wall_guard.get("threshold")
    blocked_prob = candidate.get("blocked_prob")
    if (
        primitive in _FORWARD_PRIMITIVES
        and threshold is not None
        and blocked_prob is not None
        and float(blocked_prob) >= float(threshold)
    ):
        return True
    for key in ("body_clearance_penalty", "progress_floor_penalty", "runtime_penalty"):
        value = candidate.get(key)
        if value is not None and float(value) > 0.0:
            return True
    return False


def _update_guard_selected_from_candidate(
    wall_guard: dict[str, Any],
    primitive: str,
    requested_primitive: str,
) -> None:
    candidate = _guard_candidate_for_primitive(wall_guard, primitive)
    wall_guard["selected"] = primitive
    wall_guard["vetoed"] = bool(primitive != requested_primitive)
    if candidate is None:
        wall_guard["selected_blocked"] = False
        return
    wall_guard["selected_prediction_alias"] = candidate.get("prediction_alias")
    wall_guard["selected_blocked"] = bool(candidate.get("blocked"))
    for candidate_key, guard_key in (
        ("blocked_prob", "front_blocked_prob"),
        ("outcome_blocked_prob", "selected_outcome_blocked_prob"),
        ("clearance_blocked_prob", "selected_clearance_blocked_prob"),
        ("progress_m", "selected_progress_m"),
        ("score", "selected_score"),
        ("body_clearance_penalty", "selected_body_clearance_penalty"),
        ("progress_floor_penalty", "selected_progress_floor_penalty"),
    ):
        if candidate_key in candidate:
            wall_guard[guard_key] = candidate.get(candidate_key)


def _current_contact_escape_score(
    primitive: str,
    *,
    clearance_blocked_prob: float | None,
    blocked_prob: float | None,
    candidate_score: float | None = None,
) -> float:
    if primitive == "backward":
        primitive_bias = 0.02
    elif primitive in _TURN_PRIMITIVES:
        primitive_bias = 0.04
    elif primitive == "hold":
        primitive_bias = 0.08
    else:
        primitive_bias = 0.12
    clearance = 1.0 if clearance_blocked_prob is None else float(clearance_blocked_prob)
    blocked = 0.0 if blocked_prob is None else float(blocked_prob)
    policy_score = 0.0 if candidate_score is None else float(candidate_score)
    return clearance + 0.15 * blocked + primitive_bias - 0.02 * policy_score


def _current_contact_projected_clearance_ok(
    primitive: str,
    *,
    projected_clearances: dict[str, float] | None,
    current_body_clearance_m: float | None,
    min_projected_clearance_m: float | None,
    min_projected_improvement_m: float,
) -> tuple[bool, float | None, str | None]:
    if (
        projected_clearances is None
        or (
            min_projected_clearance_m is None
            and float(min_projected_improvement_m) <= 0.0
        )
    ):
        return True, None, None
    projected = projected_clearances.get(str(primitive))
    if projected is None:
        return False, None, "missing_projected_clearance"
    projected = float(projected)
    if (
        min_projected_clearance_m is not None
        and projected < float(min_projected_clearance_m)
    ):
        return False, projected, "projected_clearance"
    if (
        str(primitive) != "hold"
        and current_body_clearance_m is not None
        and float(min_projected_improvement_m) > 0.0
    ):
        required = float(current_body_clearance_m) + float(min_projected_improvement_m)
        if projected < required:
            return False, projected, "projected_improvement"
    return True, projected, None


class LearnedLocalPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(float(dropout))
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(primitive_count)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_dropout(features))


class LearnedLocalRecurrentPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.input_dropout = nn.Dropout(float(dropout))
        self.output_dropout = nn.Dropout(float(dropout))
        self.embed = nn.Sequential(
            nn.Linear(int(input_dim), int(embed_dim)),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=int(embed_dim),
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.out = nn.Linear(int(hidden_dim), int(primitive_count))

    def forward_step(
        self,
        feature: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        embedded = self.embed(self.norm(self.input_dropout(feature)))
        if hidden is None:
            hidden = torch.zeros(
                (1, embedded.shape[0], self.gru.hidden_size),
                dtype=embedded.dtype,
                device=embedded.device,
            )
        elif hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        sequence, next_hidden = self.gru(embedded.unsqueeze(1), hidden)
        return self.out(self.output_dropout(sequence[:, -1])), next_hidden


class LearnedLocalMapCnnPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        primitive_count: int,
        *,
        map_size: int,
        map_channels: int = _LEARNED_LOCAL_ONLINE_MAP_CHANNELS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.map_size = max(3, int(map_size))
        if self.map_size % 2 == 0:
            self.map_size += 1
        self.map_channels = int(map_channels)
        self.map_feature_dim = self.map_channels * self.map_size * self.map_size
        self.base_dim = self.input_dim - self.map_feature_dim
        if self.base_dim <= 0:
            raise ValueError(
                "map_cnn input_dim must include a non-map prefix before the online-map suffix"
            )
        self.input_dropout = nn.Dropout(float(dropout))
        self.base_norm = nn.LayerNorm(self.base_dim)
        self.map_norm = nn.LayerNorm(self.map_feature_dim)
        self.base_embed = nn.Sequential(
            nn.Linear(self.base_dim, int(hidden_dim) // 2),
            nn.GELU(),
        )
        self.map_conv = nn.Sequential(
            nn.Conv2d(self.map_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, self.map_channels, self.map_size, self.map_size)
            map_embed_dim = int(self.map_conv(probe).shape[1])
        fused_dim = int(hidden_dim) // 2 + map_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(primitive_count)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        dropped = self.input_dropout(features)
        base = dropped[:, : self.base_dim]
        map_flat = dropped[:, self.base_dim : self.base_dim + self.map_feature_dim]
        map_tensor = self.map_norm(map_flat).reshape(
            -1,
            self.map_channels,
            self.map_size,
            self.map_size,
        )
        return self.head(
            torch.cat(
                [
                    self.base_embed(self.base_norm(base)),
                    self.map_conv(map_tensor),
                ],
                dim=1,
            )
        )


class LearnedLocalKnnPolicy(nn.Module):
    def __init__(
        self,
        *,
        prototype_features: torch.Tensor,
        prototype_labels: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        feature_weights: torch.Tensor | None = None,
        k: int = 9,
        primitive_count: int,
    ) -> None:
        super().__init__()
        if prototype_features.ndim != 2:
            raise ValueError("KNN prototype_features must be a 2D tensor")
        self.register_buffer("prototype_features", prototype_features.float())
        self.register_buffer("prototype_labels", prototype_labels.long())
        self.register_buffer("feature_mean", feature_mean.float().flatten())
        self.register_buffer("feature_scale", feature_scale.float().flatten().clamp_min(1e-6))
        if feature_weights is None:
            feature_weights = torch.ones_like(self.feature_mean)
        self.register_buffer("feature_weights", feature_weights.float().flatten())
        self.k = max(1, int(k))
        self.primitive_count = max(1, int(primitive_count))

    def select(self, feature: torch.Tensor) -> tuple[int, torch.Tensor, list[dict[str, Any]]]:
        x = feature.detach().flatten().float().to(self.prototype_features.device)
        x = (x - self.feature_mean) / self.feature_scale
        weighted_delta = (self.prototype_features - x.unsqueeze(0)) * self.feature_weights.unsqueeze(0)
        distances = torch.sum(weighted_delta * weighted_delta, dim=1)
        k = min(int(self.k), int(distances.numel()))
        nearest_dist, nearest_idx = torch.topk(distances, k=k, largest=False)
        nearest_labels = self.prototype_labels[nearest_idx]
        vote = torch.zeros(self.primitive_count, dtype=torch.float32, device=distances.device)
        weights = 1.0 / nearest_dist.clamp_min(1e-6)
        vote.scatter_add_(0, nearest_labels.clamp(0, self.primitive_count - 1), weights)
        probs = vote / vote.sum().clamp_min(1e-6)
        selected_idx = int(torch.argmax(probs).detach().cpu())
        neighbors = [
            {
                "label_index": int(nearest_labels[i].detach().cpu()),
                "distance": _round_float(float(nearest_dist[i].detach().cpu()), 4),
                "weight": _round_float(float(weights[i].detach().cpu()), 4),
            }
            for i in range(k)
        ]
        return selected_idx, probs, neighbors


def _load_learned_local_policy(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema") != "lewm_go2_closed_loop_learned_local_policy_v0":
        raise SystemExit(f"unsupported learned-local policy schema: {checkpoint.get('schema')}")
    model_type = str(checkpoint.get("model_type", "mlp"))
    if model_type == "knn":
        model = LearnedLocalKnnPolicy(
            prototype_features=torch.as_tensor(checkpoint["prototype_features"], dtype=torch.float32, device=device),
            prototype_labels=torch.as_tensor(checkpoint["prototype_labels"], dtype=torch.long, device=device),
            feature_mean=torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device),
            feature_scale=torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=device),
            feature_weights=(
                None
                if "feature_weights" not in checkpoint
                else torch.as_tensor(checkpoint["feature_weights"], dtype=torch.float32, device=device)
            ),
            k=int(checkpoint.get("k", 9)),
            primitive_count=len(checkpoint["primitive_vocab"]),
        ).to(device)
    elif model_type == "gru":
        model = LearnedLocalRecurrentPolicyHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint.get("hidden_dim", 192)),
            primitive_count=len(checkpoint["primitive_vocab"]),
            embed_dim=int(checkpoint.get("embed_dim", 128)),
            dropout=float(checkpoint.get("dropout", 0.0)),
        ).to(device)
    elif model_type == "map_cnn":
        feature_variant = str(checkpoint.get("feature_variant", "base"))
        fallback_channels = _learned_local_online_map_channel_count(feature_variant)
        if fallback_channels <= 0:
            fallback_channels = _LEARNED_LOCAL_ONLINE_MAP_CHANNELS
        model = LearnedLocalMapCnnPolicyHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint.get("hidden_dim", 192)),
            primitive_count=len(checkpoint["primitive_vocab"]),
            map_size=int(checkpoint.get("online_map_size", 11)),
            map_channels=int(checkpoint.get("online_map_channels", fallback_channels)),
            dropout=float(checkpoint.get("dropout", 0.0)),
        ).to(device)
    else:
        model = LearnedLocalPolicyHead(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint.get("hidden_dim", 192)),
            primitive_count=len(checkpoint["primitive_vocab"]),
            dropout=float(checkpoint.get("dropout", 0.0)),
        ).to(device)
    if model_type != "knn":
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if checkpoint.get("input_mask") is not None:
        input_mask = torch.as_tensor(
            checkpoint["input_mask"],
            dtype=torch.float32,
            device=device,
        ).flatten()
        if int(input_mask.numel()) != int(checkpoint["input_dim"]):
            raise SystemExit(
                f"input_mask dim {int(input_mask.numel())} does not match "
                f"checkpoint input_dim {int(checkpoint['input_dim'])}"
            )
        checkpoint["input_mask"] = input_mask
    checkpoint["model_type"] = model_type
    return model, dict(checkpoint)


def _load_target_scheduler(
    path: Path,
    *,
    device: torch.device,
) -> tuple[TargetSchedulerHead, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema") != _TARGET_SCHEDULER_CHECKPOINT_SCHEMA:
        raise SystemExit(
            "--learned-target-scheduler-checkpoint has unsupported schema: "
            f"{checkpoint.get('schema')}"
        )
    if checkpoint.get("feature_schema") != _TARGET_SCHEDULER_FEATURE_SCHEMA:
        raise SystemExit(
            "--learned-target-scheduler-checkpoint has unsupported feature schema: "
            f"{checkpoint.get('feature_schema')}"
        )
    model = TargetSchedulerHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 64)),
        color_count=len(checkpoint["color_vocab"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dict(checkpoint)


def _one_hot(index: int, count: int, *, device: torch.device) -> torch.Tensor:
    out = torch.zeros(max(0, int(count)), dtype=torch.float32, device=device)
    if 0 <= int(index) < int(count):
        out[int(index)] = 1.0
    return out


class OnlineEgomotionMap:
    """Runtime-safe egomotion map built only from executed motion and stalls."""

    def __init__(
        self,
        *,
        size: int = 11,
        cell_m: float = 0.45,
        hard_guard_blocks: bool = False,
    ) -> None:
        size = int(size)
        if size < 3:
            size = 3
        if size % 2 == 0:
            size += 1
        self.size = size
        self.radius = size // 2
        self.cell_m = max(1e-3, float(cell_m))
        self.hard_guard_blocks = bool(hard_guard_blocks)
        self.visited: dict[tuple[int, int], int] = {}
        self.blocked: set[tuple[int, int]] = set()
        self.claimed: set[tuple[int, int]] = set()
        self.attempted_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.guard_blocked: set[tuple[int, int]] = set()
        self.guard_blocked_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def observe_pose(self, pose_xy: np.ndarray | tuple[float, float] | list[float], *, tick: int) -> None:
        cell = self._cell(pose_xy)
        self.visited[cell] = int(tick)
        self.blocked.discard(cell)
        self.guard_blocked.discard(cell)

    def mark_claim(self, pose_xy: np.ndarray | tuple[float, float] | list[float]) -> None:
        self.claimed.add(self._cell(pose_xy))

    def state_dict(self) -> dict[str, Any]:
        def _cells(cells: Any) -> list[list[int]]:
            return [[int(a), int(b)] for a, b in sorted(cells)]

        def _edges(edges: Any) -> list[list[list[int]]]:
            return [
                [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]
                for a, b in sorted(edges)
            ]

        return {
            "size": int(self.size),
            "cell_m": float(self.cell_m),
            "hard_guard_blocks": bool(self.hard_guard_blocks),
            "visited": [
                [[int(cell[0]), int(cell[1])], int(tick)]
                for cell, tick in sorted(self.visited.items())
            ],
            "blocked": _cells(self.blocked),
            "claimed": _cells(self.claimed),
            "attempted_edges": _edges(self.attempted_edges),
            "blocked_edges": _edges(self.blocked_edges),
            "guard_blocked": _cells(self.guard_blocked),
            "guard_blocked_edges": _edges(self.guard_blocked_edges),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.size = int(state.get("size", self.size))
        self.radius = self.size // 2
        self.cell_m = max(1e-3, float(state.get("cell_m", self.cell_m)))
        self.hard_guard_blocks = bool(state.get("hard_guard_blocks", self.hard_guard_blocks))
        self.visited = {
            (int(cell[0]), int(cell[1])): int(tick)
            for cell, tick in state.get("visited", [])
        }
        self.blocked = {
            (int(cell[0]), int(cell[1])) for cell in state.get("blocked", [])
        }
        self.claimed = {
            (int(cell[0]), int(cell[1])) for cell in state.get("claimed", [])
        }
        self.attempted_edges = {
            ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
            for a, b in state.get("attempted_edges", [])
        }
        self.blocked_edges = {
            ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
            for a, b in state.get("blocked_edges", [])
        }
        self.guard_blocked = {
            (int(cell[0]), int(cell[1])) for cell in state.get("guard_blocked", [])
        }
        self.guard_blocked_edges = {
            ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
            for a, b in state.get("guard_blocked_edges", [])
        }

    def mark_rotation_blocked(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        target_cell: tuple[int, int],
    ) -> None:
        """Mark the edge toward a route target unreachable by rotation.

        Physical-mode wall contact can mechanically block in-place yaw, so a
        route requiring alignment the body cannot perform must be rerouted.
        """
        start_cell = self._cell(pose_xy)
        cell = (int(target_cell[0]), int(target_cell[1]))
        if cell == start_cell:
            return
        step = self._cardinal_step_toward(start_cell, cell)
        if step == start_cell:
            return
        self.blocked_edges.add((start_cell, step))
        self.blocked_edges.add((step, start_cell))
        self.blocked.add(step)

    def reset_after_claim(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        *,
        tick: int,
    ) -> None:
        claimed = set(self.claimed)
        self.visited.clear()
        self.blocked.clear()
        self.guard_blocked.clear()
        self.attempted_edges.clear()
        self.blocked_edges.clear()
        self.guard_blocked_edges.clear()
        self.claimed = claimed
        self.observe_pose(pose_xy, tick=int(tick))

    def mark_blocked_primitive(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        primitive: str,
    ) -> bool:
        if str(primitive) not in _TRANSLATING_PRIMITIVES:
            return False
        start_cell = self._cell(pose_xy)
        target_cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        edge_target = self._cardinal_step_toward(start_cell, target_cell)
        if edge_target == start_cell:
            return False
        before = len(self.guard_blocked_edges)
        self.attempted_edges.add((start_cell, edge_target))
        self.guard_blocked_edges.add((start_cell, edge_target))
        self.guard_blocked_edges.add((edge_target, start_cell))
        self.guard_blocked.add(edge_target)
        return len(self.guard_blocked_edges) > before

    def update_after_action(
        self,
        *,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        post_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        primitive: str,
        stalled: bool,
        tick: int,
    ) -> None:
        start_cell = self._cell(pose_xy)
        post_cell = self._cell(post_xy)
        target_cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        edge_target = (
            post_cell
            if post_cell != start_cell and not bool(stalled)
            else self._cardinal_step_toward(start_cell, target_cell)
        )
        if edge_target != start_cell and str(primitive) in _TRANSLATING_PRIMITIVES:
            # Only conclusive outcomes (stall or an actual cell transition)
            # consume the edge's attempt budget; a partial in-cell step is
            # inconclusive and must not disqualify future probes.
            if bool(stalled) or post_cell != start_cell:
                self.attempted_edges.add((start_cell, edge_target))
            if bool(stalled):
                self.blocked_edges.add((start_cell, edge_target))
                self.blocked_edges.add((edge_target, start_cell))
                self.blocked.add(edge_target)
                self.guard_blocked_edges.discard((start_cell, edge_target))
                self.guard_blocked_edges.discard((edge_target, start_cell))
                self.guard_blocked.discard(edge_target)
            elif post_cell != start_cell:
                # Only a real cell transition may clear a blocked edge. A
                # partial in-cell translation (e.g. forward re-entering a wall
                # pocket after a backward escape) previously counted as
                # reaching the ahead cell and erased the stall mark every
                # backward/forward oscillation, so the same wall was pushed
                # hundreds of times.
                self.attempted_edges.add((edge_target, start_cell))
                self.blocked_edges.discard((start_cell, edge_target))
                self.blocked_edges.discard((edge_target, start_cell))
                self.guard_blocked_edges.discard((start_cell, edge_target))
                self.guard_blocked_edges.discard((edge_target, start_cell))
                self.blocked.discard(edge_target)
                self.blocked.discard(post_cell)
                self.guard_blocked.discard(edge_target)
                self.guard_blocked.discard(post_cell)
                self.visited[edge_target] = int(tick) + 1
        self.observe_pose(post_xy, tick=int(tick) + 1)
        if bool(stalled) and str(primitive) in _FORWARD_PRIMITIVES:
            self.blocked.add(self._ahead_cell(pose_xy, yaw_rad))

    def primitive_novelty(
        self,
        *,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        primitive: str,
        tick: int,
        blocked_penalty: float = 1.5,
        turn_scale: float = 0.2,
        claim_repulsion_weight: float = 0.0,
        frontier_route_weight: float = 0.0,
    ) -> dict[str, Any]:
        cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        current_cell = self._cell(pose_xy)
        visited_tick = self.visited.get(cell)
        visited = visited_tick is not None
        age_ticks = None if visited_tick is None else max(0, int(tick) - int(visited_tick))
        stale_score = (
            1.0
            if not visited
            else 0.45 * min(1.0, float(age_ticks if age_ticks is not None else 0) / 240.0)
        )
        unknown_neighbors = self._unknown_cardinal_neighbors(cell)
        frontier_score = 0.25 * float(unknown_neighbors) / 4.0
        blocked = cell in self.blocked
        edge_target = self._cardinal_step_toward(current_cell, cell)
        edge_blocked = (current_cell, edge_target) in self.blocked_edges
        guard_blocked = cell in self.guard_blocked
        guard_edge_blocked = (current_cell, edge_target) in self.guard_blocked_edges
        edge_attempted = (current_cell, edge_target) in self.attempted_edges
        claimed = cell in self.claimed
        score = stale_score + frontier_score
        if str(primitive) == "backward":
            score = 0.35 * stale_score + 0.05 * float(unknown_neighbors) / 4.0 - 0.65
        if claimed:
            score -= 0.35
        if blocked or edge_blocked:
            score -= float(blocked_penalty)
        elif guard_blocked or guard_edge_blocked:
            score -= min(0.45, 0.35 * float(blocked_penalty))
        elif edge_attempted and not visited:
            score -= max(0.35, 0.5 * float(blocked_penalty))
        claim_repulsion = 0.0
        if self.claimed and float(claim_repulsion_weight) != 0.0:
            current_dist = min(_cell_distance(current_cell, item) for item in self.claimed)
            target_dist = min(_cell_distance(cell, item) for item in self.claimed)
            claim_repulsion = max(-1.5, min(1.5, float(target_dist - current_dist)))
            score += float(claim_repulsion_weight) * claim_repulsion
        frontier_route = 0.0
        if float(frontier_route_weight) != 0.0:
            frontier_route = self._frontier_route_improvement(current_cell, cell)
            score += float(frontier_route_weight) * frontier_route
        if str(primitive) in _TURN_PRIMITIVES:
            score *= max(0.0, float(turn_scale))
        if str(primitive) == "hold":
            score -= 0.4
        row_col = self._egocentric_row_col(cell, pose_xy, yaw_rad)
        return {
            "enabled": True,
            "primitive": str(primitive),
            "score": float(score),
            "cell": [int(cell[0]), int(cell[1])],
            "egocentric_row_col": (
                None if row_col is None else [int(row_col[0]), int(row_col[1])]
            ),
            "visited": bool(visited),
            "blocked": bool(blocked),
            "edge_blocked": bool(edge_blocked),
            "guard_blocked": bool(guard_blocked),
            "guard_edge_blocked": bool(guard_edge_blocked),
            "edge_attempted": bool(edge_attempted),
            "claimed": bool(claimed),
            "age_ticks": None if age_ticks is None else int(age_ticks),
            "unknown_neighbors": int(unknown_neighbors),
            "stale_score": float(stale_score),
            "frontier_score": float(frontier_score),
            "claim_repulsion": float(claim_repulsion),
            "claim_repulsion_weight": float(claim_repulsion_weight),
            "frontier_route": float(frontier_route),
            "frontier_route_weight": float(frontier_route_weight),
            "turn_scale": float(turn_scale) if str(primitive) in _TURN_PRIMITIVES else None,
        }

    def frontier_pressure_primitive(
        self,
        *,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        predictions: dict[str, dict[str, float]] | None,
        candidate_primitives: list[str],
        max_blocked_prob: float,
        min_progress_m: float,
        min_route_cells: int = 1,
        route_weight: float = 1.0,
        claim_weight: float = 0.35,
        novelty_weight: float = 0.25,
        guard_blocked_penalty: float = 0.18,
        allow_nonroute_backward_claim_escape: bool = False,
        prefer_unguarded_candidates: bool = False,
        allow_map_blocked_backward_claim_escape: bool = False,
        allow_guarded_retry: bool = False,
        allow_combined_blocked_retry: bool = False,
        probe_route_steps: bool = False,
    ) -> tuple[str | None, dict[str, Any]]:
        if not predictions:
            return None, {"enabled": False, "reason": "missing_predictions"}
        current_cell = self._cell(pose_xy)
        min_route = max(0, int(min_route_cells))
        route_path = self._path_to_claim_repelled_frontier(
            current_cell,
            min_distance=min_route,
            pose_xy=pose_xy,
            yaw_rad=float(yaw_rad),
        )
        route_next = route_path[1] if len(route_path) > 1 else None
        route_goal = route_path[-1] if route_path else None
        route_bearing: float | None = None
        if route_next is not None:
            arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
            px = float(arr[0]) if arr.size > 0 else 0.0
            py = float(arr[1]) if arr.size > 1 else 0.0
            nx, ny = self._cell_center(route_next)
            route_bearing = float(wrap_angle_pi(math.atan2(ny - py, nx - px) - float(yaw_rad)))
        current_frontier_dist = self._distance_to_self_built_frontier(
            current_cell,
            min_distance=min_route,
        )
        current_claim_dist = self._distance_to_nearest_claim(current_cell)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        for name in candidate_primitives:
            if name not in _LEARNED_LOCAL_POLICY_PRIMITIVES:
                continue
            if name in _TURN_PRIMITIVES or name == "hold":
                continue
            alias = _prediction_alias_for_primitive(name, predictions)
            pred = predictions.get(alias) if alias is not None else None
            if pred is None:
                continue
            blocked_prob = float(pred.get("blocked_prob", 1.0))
            progress_m = float(pred.get("progress_m", 0.0))
            clearance_prob = float(pred.get("clearance_blocked_prob", blocked_prob))
            target_cell = self._primitive_target_cell(pose_xy, yaw_rad, name)
            edge_target = self._cardinal_step_toward(current_cell, target_cell)
            map_blocked = bool(
                target_cell in self.blocked
                or edge_target in self.blocked
                or target_cell in self.claimed
                or edge_target in self.claimed
                or (current_cell, edge_target) in self.blocked_edges
            )
            guard_map_blocked = bool(
                target_cell in self.guard_blocked
                or edge_target in self.guard_blocked
                or (current_cell, edge_target) in self.guard_blocked_edges
            )
            route_score = 0.0
            route_step = bool(route_next is not None and edge_target == route_next)
            if route_step:
                route_score += 1.25
            if current_frontier_dist is not None:
                dist_cell = target_cell if target_cell in self.visited else edge_target
                target_frontier_dist = self._distance_to_self_built_frontier(
                    dist_cell,
                    min_distance=min_route,
                )
                if target_frontier_dist is not None:
                    route_score += max(
                        -1.0,
                        min(1.0, float(current_frontier_dist - target_frontier_dist)),
                    )
            novelty_score = 0.0
            if (
                target_cell not in self.visited
                and target_cell not in self.blocked
                and (
                    not self.hard_guard_blocks
                    or target_cell not in self.guard_blocked
                )
                and (current_cell, edge_target) not in self.attempted_edges
                and (
                    not self.hard_guard_blocks
                    or (current_cell, edge_target) not in self.guard_blocked_edges
                )
            ):
                novelty_score += 1.0
            elif edge_target in self.visited:
                visited_tick = self.visited.get(edge_target, 0)
                novelty_score += 0.2 * min(
                    1.0,
                    max(0.0, float(max(self.visited.values(), default=0) - int(visited_tick))) / 160.0,
                )
            claim_score = 0.0
            if current_claim_dist is not None:
                target_claim_dist = self._distance_to_nearest_claim(target_cell)
                if target_claim_dist is not None:
                    claim_score = max(
                        -1.0,
                        min(1.0, float(target_claim_dist - current_claim_dist)),
                    )
            backward_claim_escape = bool(
                name == "backward"
                and bool(self.claimed)
                and claim_score > 0.0
                and (
                    route_step
                    or bool(allow_nonroute_backward_claim_escape)
                )
            )
            map_blocked_allowed = bool(
                map_blocked
                and backward_claim_escape
                and bool(allow_map_blocked_backward_claim_escape)
                and not guard_map_blocked
            )
            prediction_ok = bool(
                blocked_prob <= float(max_blocked_prob)
                and clearance_prob <= float(max_blocked_prob)
                and progress_m >= float(min_progress_m)
            )
            if (
                not prediction_ok
                and bool(probe_route_steps)
                and route_step
                and (current_cell, edge_target) not in self.attempted_edges
            ):
                # Contact-tolerant probing: a never-attempted route step may be
                # tried once even when the heads call it impassable; physics
                # resolves it and the map records the outcome permanently.
                prediction_ok = True
            accepted = bool(
                (not map_blocked or map_blocked_allowed)
                and (not guard_map_blocked or not self.hard_guard_blocks)
                and (
                    name != "backward"
                    or not bool(self.claimed)
                    or backward_claim_escape
                )
                and prediction_ok
            )
            score = (
                float(route_weight) * route_score
                + float(claim_weight) * claim_score
                + float(novelty_weight) * novelty_score
                + progress_m
                - 0.45 * max(blocked_prob, clearance_prob)
            )
            if guard_map_blocked:
                score -= float(guard_blocked_penalty)
            if name == "backward" and route_next is not None and edge_target != route_next:
                score -= 0.15
            if bool(self.claimed) and name == "backward":
                score -= 0.15 if backward_claim_escape else 1.0
            item = {
                "primitive": name,
                "prediction_alias": alias if alias != name else None,
                "blocked_prob": _round_float(blocked_prob, 4),
                "clearance_blocked_prob": _round_float(clearance_prob, 4),
                "progress_m": _round_float(progress_m, 4),
                "target_cell": [int(target_cell[0]), int(target_cell[1])],
                "edge_target": [int(edge_target[0]), int(edge_target[1])],
                "map_blocked": bool(map_blocked),
                "map_blocked_allowed": bool(map_blocked_allowed),
                "guard_map_blocked": bool(guard_map_blocked),
                "accepted": bool(accepted),
                "route_step": bool(route_step),
                "backward_claim_escape": bool(backward_claim_escape),
                "nonroute_backward_claim_escape": bool(
                    name == "backward"
                    and bool(self.claimed)
                    and bool(allow_nonroute_backward_claim_escape)
                    and not route_step
                    and claim_score > 0.0
                ),
                "route_score": _round_float(route_score, 4),
                "claim_score": _round_float(claim_score, 4),
                "novelty_score": _round_float(novelty_score, 4),
                "score": _round_float(score, 4),
            }
            if accepted:
                scored.append((float(score), name, item))
            else:
                rejected.append(item)
        base_log = {
            "enabled": True,
            "selected": None,
            "max_blocked_prob": _round_float(float(max_blocked_prob), 4),
            "min_progress_m": _round_float(float(min_progress_m), 4),
            "min_route_cells": int(min_route),
            "route_path_len": int(len(route_path)),
            "route_next": None if route_next is None else [int(route_next[0]), int(route_next[1])],
            "route_goal": None if route_goal is None else [int(route_goal[0]), int(route_goal[1])],
            "route_bearing": _round_float(route_bearing, 4),
            "guard_blocked_penalty": _round_float(float(guard_blocked_penalty), 4),
            "allow_nonroute_backward_claim_escape": bool(
                allow_nonroute_backward_claim_escape
            ),
            "prefer_unguarded_candidates": bool(prefer_unguarded_candidates),
            "allow_map_blocked_backward_claim_escape": bool(
                allow_map_blocked_backward_claim_escape
            ),
            "allow_guarded_retry": bool(allow_guarded_retry),
            "allow_combined_blocked_retry": bool(allow_combined_blocked_retry),
        }
        defer_route_align_for_retry = bool(
            bool(allow_guarded_retry)
            and bool(self.claimed)
            and route_bearing is not None
            and abs(float(route_bearing)) > 2.35
        )
        if (
            bool(self.claimed)
            and route_bearing is not None
            and abs(float(route_bearing)) > 2.35
            and not defer_route_align_for_retry
        ):
            selected_yaw = "yaw_left" if float(route_bearing) > 0.0 else "yaw_right"
            return selected_yaw, {
                **base_log,
                "selected": selected_yaw,
                "reason": "post_claim_route_align_yaw",
                "route_align_threshold": 2.35,
                "rejected_candidates": rejected[:5],
            }
        if not scored:
            if bool(allow_guarded_retry):
                retry_candidates = []
                for item in rejected:
                    if (
                        str(item.get("primitive")) == "backward"
                        or float(item.get("blocked_prob", 1.0)) > float(max_blocked_prob)
                        or float(item.get("clearance_blocked_prob", 1.0)) > float(max_blocked_prob)
                        or float(item.get("progress_m", 0.0)) < float(min_progress_m)
                    ):
                        continue
                    guard_blocked = bool(item.get("guard_map_blocked"))
                    map_blocked = bool(item.get("map_blocked"))
                    if guard_blocked and not map_blocked:
                        retry_item = dict(item)
                        retry_item["retry_kind"] = "guard_blocked_no_commit"
                        retry_candidates.append(retry_item)
                    elif map_blocked and not guard_blocked:
                        retry_item = dict(item)
                        retry_item["retry_kind"] = "map_blocked_learned_guard_safe"
                        retry_candidates.append(retry_item)
                    elif (
                        map_blocked
                        and guard_blocked
                        and bool(allow_combined_blocked_retry)
                    ):
                        retry_item = dict(item)
                        retry_item["retry_kind"] = "combined_map_guard_blocked_no_commit"
                        retry_candidates.append(retry_item)
                if retry_candidates:
                    retry_candidates.sort(
                        key=lambda item: float(item.get("score", -1e9)),
                        reverse=True,
                    )
                    selected_item = retry_candidates[0]
                    selected_name = str(selected_item.get("primitive", ""))
                    return selected_name, {
                        **base_log,
                        "selected": selected_name,
                        "reason": "frontier_retry_after_noops",
                        "guarded_retry_no_commit": True,
                        "combined_blocked_retry": (
                            str(selected_item.get("retry_kind"))
                            == "combined_map_guard_blocked_no_commit"
                        ),
                        "guarded_retry_candidates": retry_candidates[:5],
                        "rejected_candidates": rejected[:5],
                    }
            route_align_threshold = 0.28
            if route_bearing is not None and abs(float(route_bearing)) >= route_align_threshold:
                selected_yaw = "yaw_left" if float(route_bearing) > 0.0 else "yaw_right"
                return selected_yaw, {
                    **base_log,
                    "selected": selected_yaw,
                    "reason": (
                        "post_claim_route_align_yaw"
                        if bool(defer_route_align_for_retry)
                        else "route_align_yaw"
                    ),
                    "route_align_threshold": _round_float(route_align_threshold, 4),
                    "deferred_for_guarded_retry": bool(defer_route_align_for_retry),
                    "rejected_candidates": rejected[:5],
                }
            base_log["reason"] = "no_accepted_frontier_pressure"
            base_log["rejected_candidates"] = rejected[:5]
            return None, base_log
        if (
            route_bearing is not None
            and abs(float(route_bearing)) >= 0.28
            and not any(bool(item[2].get("route_step")) for item in scored)
        ):
            # No accepted candidate advances the frontier route and the route
            # is off-bearing: align first. Without this, an unblocked arc that
            # points away from the route wins on novelty/progress every tick
            # and the robot orbits in place next to its own frontier.
            selected_yaw = "yaw_left" if float(route_bearing) > 0.0 else "yaw_right"
            return selected_yaw, {
                **base_log,
                "selected": selected_yaw,
                "reason": "route_align_yaw_over_nonroute",
                "route_align_threshold": 0.28,
                "rejected_candidates": rejected[:5],
            }
        selected_scored = scored
        if bool(prefer_unguarded_candidates):
            unguarded_scored = [
                item for item in scored if not bool(item[2].get("guard_map_blocked"))
            ]
            if unguarded_scored:
                selected_scored = unguarded_scored
                base_log["prefer_unguarded_candidates_active"] = True
                base_log["prefer_unguarded_candidates_suppressed"] = int(
                    len(scored) - len(unguarded_scored)
                )
            else:
                base_log["prefer_unguarded_candidates_active"] = False
                base_log["prefer_unguarded_candidates_suppressed"] = 0
        selected_scored.sort(key=lambda item: item[0], reverse=True)
        scored.sort(key=lambda item: item[0], reverse=True)
        return selected_scored[0][1], {
            **base_log,
            "selected": selected_scored[0][1],
            "top_candidates": [item for _, _, item in scored[:5]],
            "rejected_candidates": rejected[:5],
        }

    def feature(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        *,
        tick: int,
        device: torch.device,
        channel_count: int = _LEARNED_LOCAL_ONLINE_MAP_CHANNELS,
    ) -> torch.Tensor:
        channel_count = int(channel_count)
        if channel_count <= 0:
            channel_count = _LEARNED_LOCAL_ONLINE_MAP_CHANNELS
        channels = np.zeros(
            (channel_count, self.size, self.size),
            dtype=np.float32,
        )
        self._scatter_cells(channels[0], self.visited.keys(), pose_xy, yaw_rad, value=1.0)
        self._scatter_cells(channels[1], self.blocked, pose_xy, yaw_rad, value=1.0)
        self._scatter_cells(channels[3], self.claimed, pose_xy, yaw_rad, value=1.0)
        if channel_count >= 8:
            current_cell = self._cell(pose_xy)
            frontier_cells = self._self_built_frontier_cells()
            frontier_path = self._path_to_self_built_frontier(current_cell)
            frontier_targets = self._self_built_frontier_targets(frontier_cells)
            attempted_targets = [dst for _, dst in self.attempted_edges]
            self._scatter_cells(channels[4], frontier_cells, pose_xy, yaw_rad, value=1.0)
            self._scatter_cells(channels[5], frontier_path, pose_xy, yaw_rad, value=1.0)
            self._scatter_cells(channels[6], frontier_targets, pose_xy, yaw_rad, value=1.0)
            self._scatter_cells(channels[7], attempted_targets, pose_xy, yaw_rad, value=1.0)
        for cell, seen_tick in self.visited.items():
            row_col = self._egocentric_row_col(cell, pose_xy, yaw_rad)
            if row_col is None:
                continue
            age = max(0, int(tick) - int(seen_tick))
            channels[2, row_col[0], row_col[1]] = max(
                channels[2, row_col[0], row_col[1]],
                max(0.0, 1.0 - float(age) / 160.0),
            )
        return torch.from_numpy(channels.reshape(-1)).to(device=device)

    def summary(self) -> dict[str, int | float]:
        return {
            "size": int(self.size),
            "cell_m": float(self.cell_m),
            "hard_guard_blocks": bool(self.hard_guard_blocks),
            "visited_cells": int(len(self.visited)),
            "blocked_cells": int(len(self.blocked)),
            "claimed_cells": int(len(self.claimed)),
            "attempted_edges": int(len(self.attempted_edges)),
            "blocked_edges": int(len(self.blocked_edges)),
            "guard_blocked_cells": int(len(self.guard_blocked)),
            "guard_blocked_edges": int(len(self.guard_blocked_edges)),
        }

    def _cell(self, pose_xy: np.ndarray | tuple[float, float] | list[float]) -> tuple[int, int]:
        arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
        x = float(arr[0]) if arr.size > 0 else 0.0
        y = float(arr[1]) if arr.size > 1 else 0.0
        return (int(round(x / self.cell_m)), int(round(y / self.cell_m)))

    def _ahead_cell(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
    ) -> tuple[int, int]:
        arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
        x = float(arr[0]) if arr.size > 0 else 0.0
        y = float(arr[1]) if arr.size > 1 else 0.0
        return self._cell(
            (
                x + self.cell_m * math.cos(float(yaw_rad)),
                y + self.cell_m * math.sin(float(yaw_rad)),
            )
        )

    def _primitive_target_cell(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        primitive: str,
    ) -> tuple[int, int]:
        arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
        x = float(arr[0]) if arr.size > 0 else 0.0
        y = float(arr[1]) if arr.size > 1 else 0.0
        name = str(primitive)
        yaw_delta = 0.0
        distance_m = 0.0
        if name == "forward_fast":
            distance_m = 1.6 * self.cell_m
        elif name == "forward_medium":
            distance_m = 1.25 * self.cell_m
        elif name == "forward_slow":
            distance_m = 1.0 * self.cell_m
        elif name == "arc_left":
            distance_m = 1.1 * self.cell_m
            yaw_delta = 0.45
        elif name == "arc_right":
            distance_m = 1.1 * self.cell_m
            yaw_delta = -0.45
        elif name == "backward":
            distance_m = -1.0 * self.cell_m
        elif name == "yaw_left":
            yaw_delta = math.pi / 3.0
        elif name == "yaw_right":
            yaw_delta = -math.pi / 3.0
        yaw = float(yaw_rad) + yaw_delta
        return self._cell((x + distance_m * math.cos(yaw), y + distance_m * math.sin(yaw)))

    def _unknown_cardinal_neighbors(self, cell: tuple[int, int]) -> int:
        cx, cy = int(cell[0]), int(cell[1])
        neighbors = ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1))
        return sum(
            1
            for item in neighbors
            if item not in self.visited
            and item not in self.blocked
            and (not self.hard_guard_blocks or item not in self.guard_blocked)
            and (cell, item) not in self.blocked_edges
            and (
                not self.hard_guard_blocks
                or (cell, item) not in self.guard_blocked_edges
            )
        )

    def _cardinal_step_toward(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int]:
        sx, sy = int(source[0]), int(source[1])
        dx = int(target[0]) - sx
        dy = int(target[1]) - sy
        if dx == 0 and dy == 0:
            return (sx, sy)
        if abs(dx) >= abs(dy) and dx != 0:
            return (sx + (1 if dx > 0 else -1), sy)
        if dy != 0:
            return (sx, sy + (1 if dy > 0 else -1))
        return (sx, sy)

    def _frontier_route_improvement(
        self,
        current_cell: tuple[int, int],
        target_cell: tuple[int, int],
    ) -> float:
        if not self.visited:
            return 0.0
        current_dist = self._distance_to_self_built_frontier(current_cell)
        if current_dist is None:
            return 0.0
        if target_cell not in self.visited:
            if target_cell in self.blocked or (
                self.hard_guard_blocks and target_cell in self.guard_blocked
            ):
                return -1.0
            edge_target = self._cardinal_step_toward(current_cell, target_cell)
            if (
                (current_cell, edge_target) in self.blocked_edges
                or (
                    self.hard_guard_blocks
                    and (current_cell, edge_target) in self.guard_blocked_edges
                )
            ):
                return -1.0
            if (current_cell, edge_target) not in self.attempted_edges:
                return 0.9
            return -0.5
        target_dist = self._distance_to_self_built_frontier(target_cell)
        if target_dist is None:
            return -0.5
        return max(-1.0, min(1.0, float(current_dist - target_dist)))

    def route_replay_guard_evidence(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        primitive: str,
    ) -> dict[str, Any]:
        current_cell = self._cell(pose_xy)
        target_cell = self._primitive_target_cell(pose_xy, yaw_rad, primitive)
        edge_target = self._cardinal_step_toward(current_cell, target_cell)
        route_path = self._path_to_claim_repelled_frontier(
            current_cell,
            min_distance=0,
            pose_xy=pose_xy,
            yaw_rad=float(yaw_rad),
        )
        route_next = route_path[1] if len(route_path) > 1 else None
        route_step = bool(route_next is not None and edge_target == route_next)
        dist_cell = target_cell if target_cell in self.visited else edge_target
        current_frontier_dist = self._distance_to_self_built_frontier(current_cell)
        target_frontier_dist = (
            None
            if dist_cell not in self.visited
            else self._distance_to_self_built_frontier(dist_cell)
        )
        route_improvement = 0.0
        if current_frontier_dist is not None and target_frontier_dist is not None:
            route_improvement = float(current_frontier_dist - target_frontier_dist)
        map_blocked = bool(
            target_cell in self.blocked
            or edge_target in self.blocked
            or (current_cell, edge_target) in self.blocked_edges
        )
        guard_blocked = bool(
            target_cell in self.guard_blocked
            or edge_target in self.guard_blocked
            or (current_cell, edge_target) in self.guard_blocked_edges
        )
        nonstraight_replay = bool(str(primitive) in ("arc_left", "arc_right", "backward"))
        guard_blocked_recovery = bool(guard_blocked and nonstraight_replay)
        claimed = bool(target_cell in self.claimed or edge_target in self.claimed)
        visited_replay = bool(target_cell in self.visited)
        allow = bool(
            str(primitive) in _TRANSLATING_PRIMITIVES
            and str(primitive) not in _STRAIGHT_FORWARD_PRIMITIVES
            and current_cell in self.visited
            and edge_target != current_cell
            and visited_replay
            and not map_blocked
            and (not guard_blocked or guard_blocked_recovery)
            and not claimed
            and (route_step or route_improvement > 0.0)
        )
        return {
            "allow": bool(allow),
            "primitive": str(primitive),
            "current_cell": [int(current_cell[0]), int(current_cell[1])],
            "target_cell": [int(target_cell[0]), int(target_cell[1])],
            "edge_target": [int(edge_target[0]), int(edge_target[1])],
            "visited_target": bool(target_cell in self.visited),
            "visited_edge_target": bool(edge_target in self.visited),
            "map_blocked": bool(map_blocked),
            "guard_blocked": bool(guard_blocked),
            "guard_blocked_recovery": bool(guard_blocked_recovery),
            "nonstraight_replay": bool(nonstraight_replay),
            "claimed": bool(claimed),
            "route_step": bool(route_step),
            "route_path_len": int(len(route_path)),
            "route_next": None if route_next is None else [int(route_next[0]), int(route_next[1])],
            "current_frontier_dist": (
                None if current_frontier_dist is None else int(current_frontier_dist)
            ),
            "target_frontier_dist": (
                None if target_frontier_dist is None else int(target_frontier_dist)
            ),
            "route_improvement": _round_float(float(route_improvement), 4),
        }

    def _distance_to_self_built_frontier(
        self,
        start: tuple[int, int],
        *,
        min_distance: int = 0,
    ) -> int | None:
        path = self._path_to_self_built_frontier(start, min_distance=int(min_distance))
        if not path:
            return None
        return max(0, int(len(path) - 1))

    def _path_to_self_built_frontier(
        self,
        start: tuple[int, int],
        *,
        min_distance: int = 0,
    ) -> list[tuple[int, int]]:
        if start not in self.visited:
            return []
        queue: list[tuple[int, int]] = [start]
        seen = {start}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        for cell in queue:
            if self._is_self_built_frontier(cell) and _cell_distance(start, cell) >= int(min_distance):
                out: list[tuple[int, int]] = []
                cursor: tuple[int, int] | None = cell
                while cursor is not None:
                    out.append(cursor)
                    cursor = parent.get(cursor)
                out.reverse()
                return out
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    neighbor in seen
                    or neighbor not in self.visited
                    or (neighbor in self.blocked and neighbor not in self.visited)
                    or (cell, neighbor) in self.blocked_edges
                    or (
                        self.hard_guard_blocks
                        and (cell, neighbor) in self.guard_blocked_edges
                    )
                ):
                    continue
                seen.add(neighbor)
                parent[neighbor] = cell
                queue.append(neighbor)
        return []

    def path_to_goal_biased_frontier(
        self,
        start_xy: np.ndarray | tuple[float, float] | list[float],
        goal_xy: tuple[float, float],
        *,
        goal_weight: float = 1.0,
        optimistic: bool = False,
        max_cells: int = 4000,
    ) -> list[tuple[int, int]]:
        """BFS over visited cells to the frontier that best approaches a goal.

        The goal is an (approximate) world position estimated from a seen
        target's bearing and area; it is usually OUTSIDE the visited region,
        so the route targets the reachable frontier cell minimising
        path_length + goal_weight * remaining_cell_distance. If the goal cell
        itself is inside the visited region, route straight to it.
        """
        start = self._cell(start_xy)
        goal = self._cell(goal_xy)
        if start not in self.visited:
            return []
        queue: list[tuple[int, int]] = [start]
        seen = {start}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        depth: dict[tuple[int, int], int] = {start: 0}
        best_cell: tuple[int, int] | None = None
        best_score: float | None = None
        approach_cell: tuple[int, int] | None = None
        approach_key: tuple[int, int] | None = None
        start_goal_distance = _cell_distance(start, goal)
        goal_reached = False
        for cell in queue:
            if cell == goal:
                best_cell, best_score = cell, -1.0
                goal_reached = True
                break
            remaining = _cell_distance(cell, goal)
            if cell != start and remaining < start_goal_distance:
                # Track the best goal-approaching visited cell as a fallback
                # when no qualifying frontier exists (edges may all have been
                # attempted): repositioning closer still helps.
                key = (int(remaining), int(depth[cell]))
                if approach_key is None or key < approach_key:
                    approach_key, approach_cell = key, cell
                if self._is_self_built_frontier(cell):
                    score = float(depth[cell]) + float(goal_weight) * float(remaining)
                    if best_score is None or score < best_score:
                        best_cell, best_score = cell, score
            if len(seen) > int(max_cells):
                break
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                vision_free = getattr(self, "vision_free", None)
                if (
                    neighbor in seen
                    or (
                        not optimistic
                        and neighbor not in self.visited
                        and not (vision_free is not None and neighbor in vision_free)
                    )
                    or neighbor in self.blocked
                    or neighbor in getattr(self, "vision_blocked", ())
                    or (cell, neighbor) in self.blocked_edges
                    or (
                        self.hard_guard_blocks
                        and (cell, neighbor) in self.guard_blocked_edges
                    )
                ):
                    continue
                if optimistic and (abs(neighbor[0]) > 24 or abs(neighbor[1]) > 24):
                    continue
                seen.add(neighbor)
                parent[neighbor] = cell
                depth[neighbor] = depth[cell] + 1
                queue.append(neighbor)
        if best_cell is None and not goal_reached:
            best_cell = approach_cell
        if best_cell is None:
            return []
        out: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = best_cell
        while cursor is not None:
            out.append(cursor)
            cursor = parent.get(cursor)
        out.reverse()
        return out

    def cell_center_xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (float(cell[0]) * self.cell_m, float(cell[1]) * self.cell_m)

    def integrate_rays(
        self,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        angles_rad: np.ndarray,
        depths_m: np.ndarray,
        *,
        depth_cap_m: float,
        hit_confirmations: int = 3,
    ) -> int:
        """Fuse predicted free-depths into the map as vision evidence.

        Cells along each ray (short of the predicted obstacle) become
        vision_free — traversable for routing even though never driven.
        Ray terminals below the cap accumulate hit counts and become
        vision_blocked once confirmed, so single bad predictions cannot
        wall off a corridor.
        """
        if not hasattr(self, "vision_free"):
            self.vision_free: set[tuple[int, int]] = set()
            self.vision_hit_counts: dict[tuple[int, int], int] = {}
            self.vision_blocked: set[tuple[int, int]] = set()
        x0, y0 = float(pose_xy[0]), float(pose_xy[1])
        added = 0
        for angle, depth in zip(angles_rad, depths_m):
            ang = float(yaw_rad) + float(angle)
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            free_len = max(0.0, min(float(depth), float(depth_cap_m)) - 0.3)
            step = 0.15
            n = int(free_len / step)
            for k in range(1, n + 1):
                cell = self._cell((x0 + cos_a * step * k, y0 + sin_a * step * k))
                if cell in self.blocked or cell in self.vision_blocked:
                    break
                if cell not in self.vision_free:
                    self.vision_free.add(cell)
                    added += 1
            if float(depth) < float(depth_cap_m) - 0.1:
                hit = self._cell(
                    (x0 + cos_a * float(depth), y0 + sin_a * float(depth))
                )
                if hit not in self.visited:
                    count = self.vision_hit_counts.get(hit, 0) + 1
                    self.vision_hit_counts[hit] = count
                    if count >= int(hit_confirmations):
                        self.vision_blocked.add(hit)
                        self.vision_free.discard(hit)
        return added

    def _path_to_claim_repelled_frontier(
        self,
        start: tuple[int, int],
        *,
        min_distance: int = 0,
        pose_xy: np.ndarray | tuple[float, float] | list[float] | None = None,
        yaw_rad: float | None = None,
    ) -> list[tuple[int, int]]:
        if start not in self.visited:
            return []
        if not self.claimed:
            return self._path_to_self_built_frontier(start, min_distance=int(min_distance))
        queue: list[tuple[int, int]] = [start]
        seen = {start}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        dist: dict[tuple[int, int], int] = {start: 0}
        candidates: list[tuple[float, tuple[int, int]]] = []
        current_claim_dist = self._distance_to_nearest_claim(start)
        for cell in queue:
            path_dist = int(dist[cell])
            if self._is_self_built_frontier(cell) and _cell_distance(start, cell) >= int(min_distance):
                claim_dist = self._distance_to_nearest_claim(cell)
                claim_gain = 0.0
                if claim_dist is not None and current_claim_dist is not None:
                    claim_gain = float(claim_dist - current_claim_dist)
                heading_score = 0.0
                if pose_xy is not None and yaw_rad is not None:
                    arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
                    px = float(arr[0]) if arr.size > 0 else 0.0
                    py = float(arr[1]) if arr.size > 1 else 0.0
                    cxw, cyw = self._cell_center(cell)
                    bearing = float(
                        wrap_angle_pi(math.atan2(cyw - py, cxw - px) - float(yaw_rad))
                    )
                    heading_score = math.cos(bearing)
                    if abs(bearing) > 2.35:
                        heading_score -= 0.75
                score = (
                    1.15 * claim_gain
                    + 0.70 * heading_score
                    + 0.22 * float(self._unknown_cardinal_neighbors(cell))
                    - 0.10 * float(path_dist)
                )
                candidates.append((score, cell))
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    neighbor in seen
                    or neighbor not in self.visited
                    or (neighbor in self.blocked and neighbor not in self.visited)
                    or (cell, neighbor) in self.blocked_edges
                    or (
                        self.hard_guard_blocks
                        and (cell, neighbor) in self.guard_blocked_edges
                    )
                ):
                    continue
                seen.add(neighbor)
                parent[neighbor] = cell
                dist[neighbor] = path_dist + 1
                queue.append(neighbor)
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        out: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = candidates[0][1]
        while cursor is not None:
            out.append(cursor)
            cursor = parent.get(cursor)
        out.reverse()
        return out

    def _distance_to_nearest_claim(self, cell: tuple[int, int]) -> int | None:
        if not self.claimed:
            return None
        return min(_cell_distance(cell, item) for item in self.claimed)

    def _is_self_built_frontier(self, cell: tuple[int, int]) -> bool:
        if cell not in self.visited:
            return False
        cx, cy = int(cell[0]), int(cell[1])
        for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (
                neighbor not in self.visited
                and neighbor not in self.blocked
                and (
                    not self.hard_guard_blocks
                    or neighbor not in self.guard_blocked
                )
                and neighbor not in self.claimed
                and (cell, neighbor) not in self.attempted_edges
                and (cell, neighbor) not in self.blocked_edges
                and (
                    not self.hard_guard_blocks
                    or (cell, neighbor) not in self.guard_blocked_edges
                )
            ):
                return True
        return False

    def _self_built_frontier_cells(self) -> list[tuple[int, int]]:
        return [cell for cell in self.visited if self._is_self_built_frontier(cell)]

    def _self_built_frontier_targets(
        self,
        frontier_cells: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        targets: list[tuple[int, int]] = []
        for cell in frontier_cells:
            cx, cy = int(cell[0]), int(cell[1])
            for neighbor in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (
                    neighbor not in self.visited
                    and neighbor not in self.blocked
                    and (
                        not self.hard_guard_blocks
                        or neighbor not in self.guard_blocked
                    )
                    and neighbor not in self.claimed
                    and (cell, neighbor) not in self.attempted_edges
                    and (cell, neighbor) not in self.blocked_edges
                    and (
                        not self.hard_guard_blocks
                        or (cell, neighbor) not in self.guard_blocked_edges
                    )
                ):
                    targets.append(neighbor)
        return targets

    def _cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (float(cell[0]) * self.cell_m, float(cell[1]) * self.cell_m)

    def _egocentric_row_col(
        self,
        cell: tuple[int, int],
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
    ) -> tuple[int, int] | None:
        arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
        px = float(arr[0]) if arr.size > 0 else 0.0
        py = float(arr[1]) if arr.size > 1 else 0.0
        cx, cy = self._cell_center(cell)
        dxw = cx - px
        dyw = cy - py
        yaw = float(yaw_rad)
        ahead = math.cos(yaw) * dxw + math.sin(yaw) * dyw
        lateral = -math.sin(yaw) * dxw + math.cos(yaw) * dyw
        row = self.radius - int(round(ahead / self.cell_m))
        col = self.radius + int(round(lateral / self.cell_m))
        if 0 <= row < self.size and 0 <= col < self.size:
            return row, col
        return None

    def _scatter_cells(
        self,
        channel: np.ndarray,
        cells: Any,
        pose_xy: np.ndarray | tuple[float, float] | list[float],
        yaw_rad: float,
        *,
        value: float,
    ) -> None:
        for cell in cells:
            row_col = self._egocentric_row_col(tuple(cell), pose_xy, yaw_rad)
            if row_col is not None:
                channel[row_col[0], row_col[1]] = max(float(channel[row_col[0], row_col[1]]), value)


def _cell_distance(left: tuple[int, int], right: tuple[int, int]) -> float:
    return float(math.hypot(float(left[0] - right[0]), float(left[1] - right[1])))


def _learned_local_feature_variant_has_clock(variant: str) -> bool:
    text = str(variant)
    return "clock" in text or "pose_topology" in text


def _learned_local_feature_variant_has_state(variant: str) -> bool:
    return "state" in str(variant)


def _learned_local_feature_variant_has_visual_readout(variant: str) -> bool:
    return "visual_readout" in str(variant)


def _learned_local_feature_variant_has_pose_topology(variant: str) -> bool:
    return "pose_topology" in str(variant)


def _learned_local_feature_variant_has_online_map(variant: str) -> bool:
    return "online_map" in str(variant)


def _learned_local_online_map_channel_count(variant: str) -> int:
    text = str(variant)
    if "online_map_edge" in text:
        return _LEARNED_LOCAL_ONLINE_MAP_CHANNELS
    if "online_map" in text:
        return 4
    return 0


def _controller_read_score(outputs: dict[str, torch.Tensor], color_idx: int) -> float | None:
    read_logits = outputs.get("read_logits")
    if read_logits is None:
        return None
    try:
        return float(torch.sigmoid(read_logits[int(color_idx)]).detach().cpu())
    except Exception:
        return None


def _learned_local_policy_feature(
    *,
    ctrl_state: tuple | None,
    outputs: dict[str, torch.Tensor],
    color_vocab: list[str],
    active_target_tc: int,
    beacon_claims: list[dict[str, Any]],
    primitive_outcomes: dict[str, dict[str, float]] | None,
    last_primitive: str,
    tick: int,
    max_ticks: int,
    append_clock_features: bool,
    append_state_features: bool,
    append_visual_readout_features: bool,
    append_online_map_features: bool,
    online_map_feature: torch.Tensor | None,
    online_map_feature_dim: int,
    controller_state_name: str,
    append_pose_topology_features: bool,
    pose_xy: np.ndarray | tuple[float, float] | list[float],
    yaw_rad: float,
    pose_scale_m: float,
    device: torch.device,
) -> torch.Tensor | None:
    if ctrl_state is None:
        return None
    memory_vec, memory_conf, memory_latent, recurrent_state = ctrl_state
    color_count = len(color_vocab)
    claimed = {str(item.get("target_color")) for item in beacon_claims}
    claimed_mask = torch.tensor(
        [1.0 if color in claimed else 0.0 for color in color_vocab],
        dtype=torch.float32,
        device=device,
    )
    area_logits = outputs.get("rgb_area_logits")
    if area_logits is None:
        area_logits = torch.zeros(color_count, dtype=torch.float32, device=device)
    evidence_vec = outputs.get("evidence_vec")
    if evidence_vec is None:
        evidence_vec = torch.zeros(color_count, 2, dtype=torch.float32, device=device)

    outcome_parts: list[float] = []
    for primitive in _LEARNED_LOCAL_POLICY_PRIMITIVES:
        alias = _prediction_alias_for_primitive(primitive, primitive_outcomes)
        pred = primitive_outcomes.get(alias) if alias is not None and primitive_outcomes else None
        if pred is None:
            outcome_parts.extend([1.0, 0.0, 1.0])
            continue
        outcome_parts.extend(
            [
                float(pred.get("blocked_prob", 1.0)),
                float(pred.get("progress_m", 0.0)),
                float(pred.get("clearance_blocked_prob", pred.get("blocked_prob", 1.0))),
            ]
        )
    outcome_t = torch.tensor(outcome_parts, dtype=torch.float32, device=device)
    last_index = (
        _LEARNED_LOCAL_POLICY_PRIMITIVES.index(last_primitive)
        if last_primitive in _LEARNED_LOCAL_POLICY_PRIMITIVES
        else -1
    )
    active_idx = int(max(0, min(int(active_target_tc), color_count - 1)))
    parts = [
        recurrent_state.detach().flatten().float(),
        memory_vec.detach().flatten().float(),
        memory_conf.detach().flatten().float(),
        memory_latent[active_idx].detach().flatten().float(),
        memory_latent.detach().mean(dim=0).flatten().float(),
        area_logits.detach().flatten().float(),
        evidence_vec.detach().flatten().float(),
        _one_hot(active_idx, color_count, device=device),
        claimed_mask,
        _one_hot(last_index, len(_LEARNED_LOCAL_POLICY_PRIMITIVES), device=device),
        outcome_t,
    ]
    if append_clock_features:
        last_claim_tick = 0
        for claim in beacon_claims:
            last_claim_tick = max(last_claim_tick, int(claim.get("tick", 0)))
        denom_ticks = max(1.0, float(max_ticks))
        denom_target = max(1.0, float(color_count - 1))
        parts.append(
            torch.tensor(
                [
                    float(tick) / denom_ticks,
                    float(max(0, int(tick) - int(last_claim_tick))) / denom_ticks,
                    float(active_idx) / denom_target,
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    if append_visual_readout_features:
        active_area = float(area_logits[active_idx].detach().cpu())
        active_mem_conf = float(memory_conf[active_idx].detach().cpu())
        active_read_score = _controller_read_score(outputs, active_idx)
        active_evid = evidence_vec[active_idx].detach().float()
        active_mem_vec = memory_vec[active_idx].detach().float()
        if active_area > 0.0:
            bearing = math.atan2(float(active_evid[1].detach().cpu()), float(active_evid[0].detach().cpu()))
            in_cone = 1.0
        else:
            bearing = math.atan2(float(active_mem_vec[1].detach().cpu()), float(active_mem_vec[0].detach().cpu()))
            in_cone = 0.0
        parts.append(
            torch.tensor(
                [
                    active_area / 4.0,
                    math.sin(float(bearing)),
                    math.cos(float(bearing)),
                    float(bearing) / math.pi,
                    active_mem_conf,
                    -1.0 if active_read_score is None else float(active_read_score),
                    in_cone,
                    float(len(beacon_claims)) / max(1.0, float(color_count)),
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    if append_pose_topology_features:
        pose_arr = np.asarray(pose_xy, dtype=np.float32).reshape(-1)
        px = float(pose_arr[0]) if pose_arr.size > 0 else 0.0
        py = float(pose_arr[1]) if pose_arr.size > 1 else 0.0
        pose_scale = max(1e-6, float(pose_scale_m))
        claim_denom = max(1.0, float(color_count))
        parts.append(
            torch.tensor(
                [
                    px / pose_scale,
                    py / pose_scale,
                    math.sin(float(yaw_rad)),
                    math.cos(float(yaw_rad)),
                    float(len(beacon_claims)) / claim_denom,
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    if append_state_features:
        state_index = (
            _LEARNED_LOCAL_STATE_FEATURES.index(str(controller_state_name).upper())
            if str(controller_state_name).upper() in _LEARNED_LOCAL_STATE_FEATURES
            else -1
        )
        parts.append(_one_hot(state_index, len(_LEARNED_LOCAL_STATE_FEATURES), device=device))
    if append_online_map_features:
        if online_map_feature is None:
            parts.append(
                torch.zeros(
                    int(online_map_feature_dim),
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            parts.append(online_map_feature.detach().flatten().float())
    return torch.cat(parts, dim=0)


def _select_learned_local_policy_primitive(
    *,
    model: nn.Module,
    checkpoint: dict[str, Any],
    feature: torch.Tensor,
    requested: str,
    recurrent_hidden: torch.Tensor | None = None,
    primitive_outcomes: dict[str, dict[str, float]] | None = None,
    outcome_rerank: bool = False,
    outcome_threshold: float = 0.5,
    forward_progress_floor: float | None = None,
    rerank_top_k: int = 5,
    rerank_policy_weight: float = 0.2,
    rerank_blocked_weight: float = 3.0,
    rerank_clearance_weight: float = 0.5,
    rerank_progress_weight: float = 1.0,
    rerank_hard_blocked_penalty: float = 2.0,
    rerank_backward_penalty: float = 0.0,
    rerank_switch_margin: float = 0.0,
    rerank_protect_top_prob: float = 0.0,
    rerank_override_min_prob: float = 0.0,
    bearing: float | None = None,
    rerank_bearing_turn_threshold: float = 0.4,
    rerank_bearing_turn_bonus: float = 0.4,
    online_map: OnlineEgomotionMap | None = None,
    online_map_pose_xy: np.ndarray | tuple[float, float] | list[float] | None = None,
    online_map_yaw_rad: float | None = None,
    online_map_tick: int | None = None,
    online_map_novelty_weight: float = 0.0,
    online_map_blocked_penalty: float = 1.5,
    online_map_turn_scale: float = 0.2,
    online_map_claim_repulsion_weight: float = 0.0,
    online_map_frontier_route_weight: float = 0.0,
    online_map_hard_veto: bool = False,
    controller_state_name: str | None = None,
    online_map_novelty_states: set[str] | None = None,
) -> tuple[str, dict[str, Any], torch.Tensor | None]:
    primitive_vocab = [str(item) for item in checkpoint["primitive_vocab"]]
    expected_dim = int(checkpoint["input_dim"])
    if int(feature.numel()) != expected_dim:
        return requested, {
            "enabled": False,
            "reason": "feature_dim_mismatch",
            "feature_dim": int(feature.numel()),
            "expected_dim": int(expected_dim),
        }, recurrent_hidden
    input_mask = checkpoint.get("input_mask")
    input_mask_active = input_mask is not None
    if input_mask_active:
        feature = feature * torch.as_tensor(
            input_mask,
            dtype=feature.dtype,
            device=feature.device,
        ).flatten()
    with torch.no_grad():
        if str(checkpoint.get("model_type", "mlp")) == "knn":
            if not isinstance(model, LearnedLocalKnnPolicy):
                return requested, {
                    "enabled": False,
                    "reason": "model_type_mismatch",
                    "model_type": str(type(model).__name__),
                }, recurrent_hidden
            selected_idx, probs, neighbors = model.select(feature)
            next_hidden = recurrent_hidden
            logits = torch.log(probs.clamp_min(1e-6))
        elif str(checkpoint.get("model_type", "mlp")) == "gru":
            if not isinstance(model, LearnedLocalRecurrentPolicyHead):
                return requested, {
                    "enabled": False,
                    "reason": "model_type_mismatch",
                    "model_type": str(type(model).__name__),
                }, recurrent_hidden
            logits_b, next_hidden = model.forward_step(feature, recurrent_hidden)
            logits = logits_b.squeeze(0)
            probs = torch.softmax(logits, dim=0)
            neighbors = []
        else:
            logits = model(feature.unsqueeze(0)).squeeze(0)
            next_hidden = recurrent_hidden
            probs = torch.softmax(logits, dim=0)
            neighbors = []
    forbidden_outputs = {
        str(item)
        for item in checkpoint.get("forbid_output_primitives", [])
        if str(item) in primitive_vocab
    }
    if forbidden_outputs:
        masked_probs = probs.detach().clone()
        for name in forbidden_outputs:
            masked_probs[primitive_vocab.index(name)] = 0.0
        prob_sum = masked_probs.sum()
        if float(prob_sum.detach().cpu()) > 0.0:
            probs = masked_probs / prob_sum
    ranked = torch.argsort(probs, descending=True).detach().cpu().tolist()
    if str(checkpoint.get("model_type", "mlp")) != "knn":
        selected_idx = int(ranked[0]) if ranked else 0
    selected = primitive_vocab[max(0, min(selected_idx, len(primitive_vocab) - 1))]
    top_selected = selected
    top_prob = float(probs[selected_idx].detach().cpu()) if primitive_vocab else 0.0
    rerank_log: dict[str, Any] | None = None
    if bool(outcome_rerank) and primitive_outcomes:
        threshold = float(outcome_threshold)
        novelty_active = bool(
            online_map is not None
            and float(online_map_novelty_weight) > 0.0
            and online_map_pose_xy is not None
            and online_map_yaw_rad is not None
            and online_map_tick is not None
            and (
                not online_map_novelty_states
                or str(controller_state_name or "").upper() in online_map_novelty_states
            )
        )
        candidates_idx = list(ranked[: max(1, int(rerank_top_k))])
        if novelty_active:
            for idx in range(len(primitive_vocab)):
                if (
                    idx not in candidates_idx
                    and primitive_vocab[idx] != "hold"
                ):
                    candidates_idx.append(idx)
        if bearing is not None and abs(float(bearing)) >= float(rerank_bearing_turn_threshold):
            for turn_name in ("yaw_left", "yaw_right"):
                if turn_name in primitive_vocab:
                    turn_idx = primitive_vocab.index(turn_name)
                    if turn_idx not in candidates_idx:
                        candidates_idx.append(turn_idx)
        scored: list[tuple[float, int, str, dict[str, Any]]] = []
        top_score: float | None = None
        top_unsafe_forward = False
        top_unsafe_translation = False
        top_online_map_hard_veto = False
        for idx in candidates_idx:
            if int(idx) < 0 or int(idx) >= len(primitive_vocab):
                continue
            name = primitive_vocab[int(idx)]
            alias = _prediction_alias_for_primitive(name, primitive_outcomes)
            pred = primitive_outcomes.get(alias) if alias is not None else None
            prob = float(probs[int(idx)].detach().cpu())
            blocked_prob = float(pred.get("blocked_prob", 0.5)) if pred is not None else 0.5
            progress_m = float(pred.get("progress_m", 0.0)) if pred is not None else 0.0
            clearance_prob = (
                float(pred.get("clearance_blocked_prob", blocked_prob))
                if pred is not None
                else blocked_prob
            )
            low_progress = bool(
                name in _FORWARD_PRIMITIVES
                and forward_progress_floor is not None
                and progress_m < float(forward_progress_floor)
            )
            unsafe_forward = bool(
                name in _FORWARD_PRIMITIVES
                and (blocked_prob >= threshold or low_progress)
            )
            unsafe_translation = bool(
                name in _TRANSLATING_PRIMITIVES
                and blocked_prob >= threshold
            )
            novelty_log = None
            online_map_hard_blocked = False
            if novelty_active and online_map is not None:
                novelty_log = online_map.primitive_novelty(
                    pose_xy=online_map_pose_xy,
                    yaw_rad=float(online_map_yaw_rad),
                    primitive=name,
                    tick=int(online_map_tick),
                    blocked_penalty=float(online_map_blocked_penalty),
                    turn_scale=float(online_map_turn_scale),
                    claim_repulsion_weight=float(online_map_claim_repulsion_weight),
                    frontier_route_weight=float(online_map_frontier_route_weight),
                )
                online_map_hard_blocked = bool(
                    online_map_hard_veto
                    and name in _TRANSLATING_PRIMITIVES
                    and (
                        bool(novelty_log.get("blocked"))
                        or bool(novelty_log.get("edge_blocked"))
                        or bool(novelty_log.get("guard_blocked"))
                        or bool(novelty_log.get("guard_edge_blocked"))
                    )
                )
                if online_map_hard_blocked:
                    if name in _FORWARD_PRIMITIVES:
                        unsafe_forward = True
                    else:
                        unsafe_translation = True
            hard_blocked = bool(unsafe_forward or unsafe_translation)
            score = (
                float(rerank_policy_weight) * math.log(max(prob, 1e-6))
                + float(rerank_progress_weight) * progress_m
                - float(rerank_blocked_weight) * blocked_prob
                - float(rerank_clearance_weight) * clearance_prob
            )
            if hard_blocked:
                score -= float(rerank_hard_blocked_penalty)
            if name == "backward":
                score -= float(rerank_backward_penalty)
            if bearing is not None and abs(float(bearing)) >= float(rerank_bearing_turn_threshold):
                preferred_turn = "yaw_left" if float(bearing) > 0.0 else "yaw_right"
                if name == preferred_turn:
                    score += float(rerank_bearing_turn_bonus)
            if novelty_log is not None:
                score += float(online_map_novelty_weight) * float(
                    novelty_log.get("score", 0.0)
                )
            item_log = {
                "primitive": name,
                "prob": _round_float(prob, 4),
                "prediction_alias": alias if alias != name else None,
                "blocked_prob": _round_float(blocked_prob, 4),
                "clearance_blocked_prob": _round_float(clearance_prob, 4),
                "progress_m": _round_float(progress_m, 4),
                "low_progress": bool(low_progress),
                "unsafe_forward": bool(unsafe_forward),
                "unsafe_translation": bool(unsafe_translation),
                "online_map_hard_veto": bool(online_map_hard_blocked),
                "score": _round_float(score, 4),
            }
            if novelty_log is not None:
                item_log["online_map_novelty"] = {
                    **novelty_log,
                    "score": _round_float(novelty_log.get("score"), 4),
                    "stale_score": _round_float(novelty_log.get("stale_score"), 4),
                    "frontier_score": _round_float(novelty_log.get("frontier_score"), 4),
                }
            scored.append((float(score), int(idx), name, item_log))
            if name == top_selected and top_score is None:
                top_score = float(score)
                top_unsafe_forward = bool(unsafe_forward)
                top_unsafe_translation = bool(unsafe_translation)
                top_online_map_hard_veto = bool(online_map_hard_blocked)
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_idx, best_name, _ = scored[0]
            selected_score = top_score if top_score is not None else best_score
            best_prob = float(probs[int(best_idx)].detach().cpu())
            preserve_top_reason: str | None = None
            if (
                best_name != top_selected
                and not bool(top_unsafe_forward or top_unsafe_translation)
            ):
                if (
                    float(rerank_protect_top_prob) > 0.0
                    and top_prob >= float(rerank_protect_top_prob)
                    and not bool(top_online_map_hard_veto)
                ):
                    preserve_top_reason = "top_policy_confidence"
                elif (
                    float(rerank_override_min_prob) > 0.0
                    and best_prob < float(rerank_override_min_prob)
                ):
                    preserve_top_reason = "candidate_policy_floor"
                elif float(best_score) < float(selected_score) + float(rerank_switch_margin):
                    preserve_top_reason = "switch_margin"
                if preserve_top_reason is not None:
                    best_idx = selected_idx
                    best_name = top_selected
            selected_idx = int(best_idx)
            selected = best_name
            rerank_log = {
                "enabled": True,
                "selected_before": top_selected,
                "selected_after": selected,
                "top_prob": _round_float(top_prob, 4),
                "selected_after_prob": _round_float(
                    float(probs[int(selected_idx)].detach().cpu()),
                    4,
                ),
                "preserve_top_reason": preserve_top_reason,
                "top_unsafe_forward": bool(top_unsafe_forward),
                "top_unsafe_translation": bool(top_unsafe_translation),
                "top_online_map_hard_veto": bool(top_online_map_hard_veto),
                "bearing": _round_float(bearing, 4),
                "threshold": _round_float(threshold, 4),
                "forward_progress_floor": _round_float(forward_progress_floor, 4),
                "top_k": int(rerank_top_k),
                "protect_top_prob": _round_float(rerank_protect_top_prob, 4),
                "override_min_prob": _round_float(rerank_override_min_prob, 4),
                "backward_penalty": _round_float(rerank_backward_penalty, 4),
                "online_map_novelty_enabled": bool(novelty_active),
                "online_map_hard_veto_enabled": bool(online_map_hard_veto),
                "online_map_novelty_weight": _round_float(online_map_novelty_weight, 4),
                "online_map_blocked_penalty": _round_float(online_map_blocked_penalty, 4),
                "online_map_turn_scale": _round_float(online_map_turn_scale, 4),
                "online_map_claim_repulsion_weight": _round_float(
                    online_map_claim_repulsion_weight,
                    4,
                ),
                "online_map_frontier_route_weight": _round_float(
                    online_map_frontier_route_weight,
                    4,
                ),
                "online_map_novelty_state": str(controller_state_name or ""),
                "online_map_novelty_states": sorted(online_map_novelty_states or []),
                "forbid_output_primitives": sorted(forbidden_outputs),
                "candidates": [item_log for _, _, _, item_log in scored],
            }
    return selected, {
        "enabled": True,
        "model_type": str(checkpoint.get("model_type", "mlp")),
        "selected": selected,
        "selected_before_outcome_rerank": top_selected,
        "requested_before": requested,
        "confidence": _round_float(float(probs[selected_idx].detach().cpu()), 4),
        "top_candidates": [
            {
                "primitive": primitive_vocab[int(idx)],
                "prob": _round_float(float(probs[int(idx)].detach().cpu()), 4),
            }
            for idx in ranked[:5]
        ],
        "nearest_neighbors": neighbors[:5],
        "outcome_rerank": rerank_log or {"enabled": False},
        "checkpoint": str(checkpoint.get("source", "")),
        "input_mask_active": bool(input_mask_active),
        "input_mask_nonzero": (
            None
            if not input_mask_active
            else int(torch.count_nonzero(torch.as_tensor(input_mask)).detach().cpu())
        ),
        "forbid_output_primitives": sorted(forbidden_outputs),
    }, next_hidden


def _load_learned_topology_route_table(path: Path) -> dict[str, Any]:
    table = json.loads(path.read_text())
    if table.get("schema") != "lewm_go2_learned_topology_route_table_v1":
        raise SystemExit(f"unsupported learned-topology route table schema: {table.get('schema')}")
    routes = table.get("routes", {})
    if not isinstance(routes, dict) or not routes:
        raise SystemExit(f"learned-topology route table has no routes: {path}")
    normalised_routes: dict[str, dict[str, Any]] = {}
    for key, route in routes.items():
        if not isinstance(route, dict):
            continue
        waypoints = []
        for item in route.get("waypoints", []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                waypoints.append((float(item[0]), float(item[1])))
        if waypoints:
            raw_primitives = route.get("primitives", [])
            primitives = []
            if isinstance(raw_primitives, list):
                primitives = [
                    str(item)
                    for item in raw_primitives[: len(waypoints)]
                    if str(item)
                ]
            if primitives and len(primitives) < len(waypoints):
                primitives.extend([primitives[-1]] * (len(waypoints) - len(primitives)))
            normalised_routes[str(key)] = {
                "target_color": str(route.get("target_color", key)),
                "target_index": int(route.get("target_index", key if str(key).isdigit() else 0)),
                "waypoints": tuple(waypoints),
                "primitives": tuple(primitives),
            }
    if not normalised_routes:
        raise SystemExit(f"learned-topology route table has no usable waypoints: {path}")
    table["routes"] = normalised_routes
    return table


def _select_learned_topology_route_primitive(
    *,
    route_table: dict[str, Any],
    route_state: dict[str, Any],
    target_color: str,
    target_index: int,
    pos_xy: np.ndarray | tuple[float, float],
    yaw: float,
    advance_m: float,
    lookahead_m: float,
    reproject_window: int,
    reproject_trigger_m: float,
    yaw_bearing_threshold: float,
    forward_bearing_threshold: float,
    arc_max_bearing: float,
    forward_primitive: str,
    use_stored_primitives: bool,
) -> tuple[str, dict[str, Any]]:
    route = route_table.get("routes", {}).get(str(target_color))
    if route is None:
        route = route_table.get("routes", {}).get(str(target_index))
    if not isinstance(route, dict):
        return forward_primitive, {"enabled": False, "reason": "missing_route"}
    waypoints = tuple(route.get("waypoints", ()))
    if not waypoints:
        return forward_primitive, {"enabled": False, "reason": "empty_route"}

    px, py = float(pos_xy[0]), float(pos_xy[1])
    active_key = f"{target_color}:{int(target_index)}"
    if route_state.get("active_key") != active_key:
        nearest = min(
            range(len(waypoints)),
            key=lambda i: (waypoints[i][0] - px) ** 2 + (waypoints[i][1] - py) ** 2,
        )
        route_state["active_key"] = active_key
        route_state["idx"] = int(nearest)
        route_state["resets"] = int(route_state.get("resets", 0)) + 1

    idx = int(route_state.get("idx", 0))
    advance2 = max(0.05, float(advance_m)) ** 2
    while idx < len(waypoints) - 1:
        wx, wy = waypoints[idx]
        if (wx - px) ** 2 + (wy - py) ** 2 <= advance2:
            idx += 1
        else:
            break

    reprojected = False
    reproject_from_idx = int(idx)
    reproject_to_idx = int(idx)
    reproject_current_dist_m = None
    reproject_best_dist_m = None
    reproject_window_i = max(0, int(reproject_window))
    reproject_trigger_m_f = max(0.0, float(reproject_trigger_m))
    if reproject_window_i > 0 and idx < len(waypoints) - 1:
        wx, wy = waypoints[idx]
        current_d2 = (wx - px) ** 2 + (wy - py) ** 2
        should_reproject = (
            reproject_trigger_m_f <= 0.0
            or current_d2 > reproject_trigger_m_f * reproject_trigger_m_f
        )
        if should_reproject:
            end_idx = min(len(waypoints) - 1, idx + reproject_window_i)
            best_idx = min(
                range(idx + 1, end_idx + 1),
                key=lambda i: (waypoints[i][0] - px) ** 2
                + (waypoints[i][1] - py) ** 2,
            )
            bx, by = waypoints[best_idx]
            best_d2 = (bx - px) ** 2 + (by - py) ** 2
            if best_d2 + 1e-9 < current_d2:
                idx = int(best_idx)
                reprojected = True
                reproject_to_idx = int(best_idx)
                reproject_current_dist_m = math.sqrt(float(current_d2))
                reproject_best_dist_m = math.sqrt(float(best_d2))

    while idx < len(waypoints) - 1:
        wx, wy = waypoints[idx]
        if (wx - px) ** 2 + (wy - py) ** 2 <= advance2:
            idx += 1
        else:
            break
    route_state["idx"] = int(idx)

    goal_idx = int(idx)
    remaining_lookahead = max(0.0, float(lookahead_m))
    if remaining_lookahead > 0.0:
        while goal_idx < len(waypoints) - 1 and remaining_lookahead > 0.0:
            x0, y0 = waypoints[goal_idx]
            x1, y1 = waypoints[goal_idx + 1]
            segment_m = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
            goal_idx += 1
            remaining_lookahead -= max(0.0, float(segment_m))

    wx, wy = waypoints[min(goal_idx, len(waypoints) - 1)]
    bearing = wrap_angle_pi(math.atan2(wy - py, wx - px) - float(yaw))
    abs_bearing = abs(float(bearing))
    stored_primitives = tuple(route.get("primitives", ()))
    stored_primitive = (
        str(stored_primitives[min(idx, len(stored_primitives) - 1)])
        if bool(use_stored_primitives) and stored_primitives
        else None
    )
    if stored_primitive:
        primitive = stored_primitive
    elif abs_bearing >= float(yaw_bearing_threshold) or abs_bearing > float(arc_max_bearing):
        primitive = "yaw_left" if bearing > 0 else "yaw_right"
    elif abs_bearing >= float(forward_bearing_threshold):
        primitive = "arc_left" if bearing > 0 else "arc_right"
    else:
        primitive = str(forward_primitive)
    return primitive, {
        "enabled": True,
        "target_color": str(target_color),
        "target_index": int(target_index),
        "idx": int(idx),
        "goal_idx": int(goal_idx),
        "route_len": int(len(waypoints)),
        "goal_xy": [_round_float(wx, 4), _round_float(wy, 4)],
        "lookahead_m": _round_float(float(lookahead_m), 4),
        "yaw_bearing_threshold": _round_float(float(yaw_bearing_threshold), 4),
        "forward_bearing_threshold": _round_float(float(forward_bearing_threshold), 4),
        "arc_max_bearing": _round_float(float(arc_max_bearing), 4),
        "reproject_window": int(reproject_window_i),
        "reproject_trigger_m": _round_float(float(reproject_trigger_m_f), 4),
        "reprojected": bool(reprojected),
        "reproject_from_idx": int(reproject_from_idx),
        "reproject_to_idx": int(reproject_to_idx),
        "reproject_current_dist_m": (
            None
            if reproject_current_dist_m is None
            else _round_float(float(reproject_current_dist_m), 4)
        ),
        "reproject_best_dist_m": (
            None
            if reproject_best_dist_m is None
            else _round_float(float(reproject_best_dist_m), 4)
        ),
        "bearing": _round_float(float(bearing), 4),
        "stored_primitive": stored_primitive,
        "use_stored_primitives": bool(use_stored_primitives),
        "primitive": primitive,
        "resets": int(route_state.get("resets", 0)),
    }


def _learned_action_guard_select(
    *,
    requested: str,
    bearing: float | None,
    enabled: bool,
    predictions: dict[str, dict[str, float]] | None,
    primitive_vocab: list[str],
    blocked_threshold: float,
    blocked_weight: float,
    progress_weight: float,
    requested_bonus: float,
    turn_progress_scale: float,
    switch_margin: float,
    target_area: float | None = None,
    body_clearance_enabled: bool = False,
    body_clearance_target_area: float = 2.0,
    body_clearance_arc_penalty: float = 0.0,
    body_clearance_yaw_penalty_weight: float = 0.0,
    body_clearance_prob_floor: float = 0.35,
    body_clearance_prob_weight: float = 0.0,
    body_clearance_near_forward_prob_floor: float | None = None,
    body_clearance_near_forward_prob_weight: float | None = None,
    body_clearance_near_yaw_prob_floor: float | None = None,
    body_clearance_yaw_always: bool = False,
    body_clearance_min_area: float | None = None,
    body_clearance_hard_veto_prob: float = 1.01,
    body_clearance_hard_veto_margin: float = 0.05,
    body_clearance_hard_veto_replacement_cap: float = 1.01,
    body_clearance_hard_veto_primitives: set[str] | None = None,
    body_clearance_hard_veto_selected_primitives: set[str] | None = None,
    body_clearance_arc_sweep_veto_prob: float = 1.01,
    body_clearance_arc_sweep_veto_selected_primitives: set[str] | None = None,
    body_clearance_saturated_veto_prob: float = 1.01,
    body_clearance_saturated_veto_spread: float = 0.01,
    body_clearance_saturated_veto_primitives: set[str] | None = None,
    body_clearance_saturated_veto_selected_primitives: set[str] | None = None,
    body_clearance_yaw_direction_veto_prob: float = 1.01,
    body_clearance_yaw_contact_veto_prob: float = 1.01,
    body_clearance_yaw_direction_veto_margin: float = 0.05,
    current_body_clearance_m: float | None = None,
    body_clearance_current_contact_escape_m: float | None = None,
    body_clearance_current_contact_escape_m_by_primitive: dict[str, float] | None = None,
    body_clearance_current_contact_escape_primitives: set[str] | None = None,
    body_clearance_current_contact_escape_replacements: set[str] | None = None,
    body_clearance_current_contact_escape_replacement_cap: float = 1.01,
    body_clearance_current_contact_escape_require_replacement_under_cap: bool = False,
    body_clearance_current_contact_escape_projected_clearances: dict[str, float] | None = None,
    body_clearance_current_contact_escape_min_projected_clearance_m: float | None = None,
    body_clearance_current_contact_escape_min_projected_improvement_m: float = 0.0,
    forward_progress_floor: float | None = None,
    forward_progress_floor_min_blocked_prob: float | None = None,
    forward_progress_floor_force_below: float | None = None,
    forward_progress_floor_penalty: float = 0.0,
    low_progress_hard_veto: bool = False,
    low_progress_hard_veto_primitives: set[str] | None = None,
    blocked_hard_veto: bool = False,
    blocked_hard_veto_primitives: set[str] | None = None,
    blocked_hard_veto_selected_primitives: set[str] | None = None,
    blocked_hard_veto_max_abs_bearing: float | None = None,
    blocked_hard_veto_bearing: float | None = None,
    progress_floor_prefer_yaw: bool = False,
    runtime_penalties: dict[str, float] | None = None,
    preserve_turn_requests: bool = False,
    preserve_arc_requests: bool = False,
    turn_body_rerank_primitives: set[str] | None = None,
    preserve_straight_requests: bool = False,
    preserve_backward_requests: bool = False,
    preserve_backward_clearance_margin: float | None = None,
    force_escape: bool = False,
    force_single_candidate: bool | None = None,
    candidate_names_override: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    force_single_candidate_active = bool(
        force_escape if force_single_candidate is None else force_single_candidate
    )
    requested_alias = _prediction_alias_for_primitive(requested, predictions)
    requested_pred = (
        (predictions or {}).get(requested_alias)
        if requested_alias is not None
        else {"blocked_prob": None, "progress_m": None}
    )
    requested_prob = requested_pred.get("blocked_prob")
    requested_progress = requested_pred.get("progress_m")
    requested_low_progress = bool(
        requested in _FORWARD_PRIMITIVES
        and forward_progress_floor is not None
        and requested_progress is not None
        and float(requested_progress) < float(forward_progress_floor)
        and (
            (
                forward_progress_floor_force_below is not None
                and float(requested_progress) < float(forward_progress_floor_force_below)
            )
            or
            forward_progress_floor_min_blocked_prob is None
            or (
                requested_prob is not None
                and float(requested_prob) >= float(forward_progress_floor_min_blocked_prob)
            )
        )
    )
    requested_blocked = bool(
        requested in _FORWARD_PRIMITIVES
        and requested_prob is not None
        and (
            float(requested_prob) >= float(blocked_threshold)
            or requested_low_progress
        )
    )
    selected = requested
    candidates: list[dict[str, Any]] = []
    selected_score: float | None = None
    requested_score: float | None = None
    selected_body_clearance_penalty: float | None = None
    requested_body_clearance_penalty: float | None = None
    selected_progress_floor_penalty: float | None = None
    requested_progress_floor_penalty: float | None = None
    body_clearance_hard_vetoed = False
    body_clearance_hard_veto_from: str | None = None
    body_clearance_hard_veto_from_prob: float | None = None
    body_clearance_area_gate_active = True
    body_clearance_hard_veto_relaxed_motion_replacement = False
    body_clearance_arc_sweep_vetoed = False
    body_clearance_arc_sweep_veto_from: str | None = None
    body_clearance_arc_sweep_veto_from_prob: float | None = None
    body_clearance_saturated_vetoed = False
    body_clearance_saturated_veto_from: str | None = None
    body_clearance_saturated_veto_from_prob: float | None = None
    body_clearance_yaw_direction_vetoed = False
    body_clearance_yaw_direction_veto_from: str | None = None
    body_clearance_yaw_direction_veto_from_prob: float | None = None
    body_clearance_yaw_contact_vetoed = False
    body_clearance_yaw_contact_veto_from: str | None = None
    body_clearance_yaw_contact_veto_from_prob: float | None = None
    body_clearance_current_contact_escape = False
    body_clearance_current_contact_escape_from: str | None = None
    body_clearance_current_contact_escape_from_prob: float | None = None
    body_clearance_current_contact_escape_to_prob: float | None = None
    body_clearance_current_contact_escape_candidate: dict[str, Any] | None = None
    body_clearance_current_contact_escape_projected_rejections: list[dict[str, Any]] = []
    body_clearance_current_contact_escape_suppressed_reason: str | None = None
    blocked_hard_vetoed = False
    blocked_hard_veto_from: str | None = None
    blocked_hard_veto_from_prob: float | None = None
    blocked_veto_reference_bearing = (
        bearing if blocked_hard_veto_bearing is None else blocked_hard_veto_bearing
    )
    blocked_veto_bearing_ok = bool(
        blocked_hard_veto_max_abs_bearing is None
        or (
            blocked_veto_reference_bearing is not None
            and abs(float(blocked_veto_reference_bearing)) <= float(blocked_hard_veto_max_abs_bearing)
        )
    )
    low_progress_hard_vetoed = False
    low_progress_hard_veto_from: str | None = None
    low_progress_hard_veto_from_progress: float | None = None
    if enabled and predictions:
        if force_single_candidate_active:
            candidate_names = [requested]
        elif candidate_names_override is not None:
            candidate_names = _unique_primitives([requested, *candidate_names_override])
        else:
            candidate_names = _unique_primitives([requested, *primitive_vocab])
        if (
            bool(preserve_straight_requests)
            and requested in _STRAIGHT_FORWARD_PRIMITIVES
            and not force_single_candidate_active
        ):
            candidate_names = [
                name for name in candidate_names
                if name not in ("arc_left", "arc_right")
            ]
        if (
            bool(progress_floor_prefer_yaw)
            and requested_low_progress
            and not force_single_candidate_active
        ):
            candidate_names = [
                name for name in candidate_names
                if name != "backward"
            ]
        scored: list[tuple[float, str, dict[str, float], float, float]] = []
        for name in candidate_names:
            pred_alias = _prediction_alias_for_primitive(name, predictions)
            if pred_alias is None:
                continue
            pred = predictions[pred_alias]
            score, body_clearance_penalty, progress_floor_penalty = _score_learned_action_candidate(
                name,
                pred,
                bearing=bearing,
                target_area=target_area,
                requested=requested,
                force_escape=bool(force_single_candidate_active),
                blocked_threshold=float(blocked_threshold),
                blocked_weight=float(blocked_weight),
                progress_weight=float(progress_weight),
                requested_bonus=float(requested_bonus),
                turn_progress_scale=float(turn_progress_scale),
                body_clearance_enabled=bool(body_clearance_enabled),
                body_clearance_target_area=float(body_clearance_target_area),
                body_clearance_arc_penalty=float(body_clearance_arc_penalty),
                body_clearance_yaw_penalty_weight=float(body_clearance_yaw_penalty_weight),
                body_clearance_prob_floor=float(body_clearance_prob_floor),
                body_clearance_prob_weight=float(body_clearance_prob_weight),
                body_clearance_near_forward_prob_floor=body_clearance_near_forward_prob_floor,
                body_clearance_near_forward_prob_weight=body_clearance_near_forward_prob_weight,
                body_clearance_near_yaw_prob_floor=body_clearance_near_yaw_prob_floor,
                body_clearance_yaw_always=bool(body_clearance_yaw_always),
                body_clearance_min_area=body_clearance_min_area,
                forward_progress_floor=forward_progress_floor,
                forward_progress_floor_min_blocked_prob=forward_progress_floor_min_blocked_prob,
                forward_progress_floor_force_below=forward_progress_floor_force_below,
                forward_progress_floor_penalty=float(forward_progress_floor_penalty),
                runtime_penalties=runtime_penalties,
            )
            scored.append((score, name, pred, body_clearance_penalty, progress_floor_penalty))
            if name == requested:
                requested_score = float(score)
                requested_body_clearance_penalty = float(body_clearance_penalty)
                requested_progress_floor_penalty = float(progress_floor_penalty)
            candidates.append({
                "primitive": name,
                "prediction_alias": pred_alias if pred_alias != name else None,
                "blocked_prob": _round_float(float(pred["blocked_prob"]), 4),
                "outcome_blocked_prob": _round_float(pred.get("outcome_blocked_prob"), 4),
                "clearance_blocked_prob": _round_float(pred.get("clearance_blocked_prob"), 4),
                "progress_m": _round_float(float(pred["progress_m"]), 4),
                "score": _round_float(float(score), 4),
                "body_clearance_penalty": _round_float(float(body_clearance_penalty), 4),
                "progress_floor_penalty": _round_float(float(progress_floor_penalty), 4),
                "runtime_penalty": _round_float((runtime_penalties or {}).get(name), 4),
                "blocked": bool(
                    name in _FORWARD_PRIMITIVES
                    and (
                        float(pred["blocked_prob"]) >= float(blocked_threshold)
                        or (
                            forward_progress_floor is not None
                            and float(pred["progress_m"]) < float(forward_progress_floor)
                            and (
                                (
                                    forward_progress_floor_force_below is not None
                                    and float(pred["progress_m"]) < float(forward_progress_floor_force_below)
                                )
                                or
                                forward_progress_floor_min_blocked_prob is None
                                or float(pred["blocked_prob"]) >= float(forward_progress_floor_min_blocked_prob)
                            )
                        )
                    )
                    or float(body_clearance_penalty) > 0.0
                ),
            })
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_score, selected, selected_pred, selected_body_clearance_penalty, selected_progress_floor_penalty = scored[0]
            requested_body_blocked = bool(
                requested_body_clearance_penalty is not None
                and float(requested_body_clearance_penalty) > 0.0
            )
            if requested_body_blocked:
                requested_blocked = True
            if requested_score is None and not force_single_candidate_active:
                selected = requested
                selected_pred = requested_pred
                selected_score = None
                selected_body_clearance_penalty = None
                selected_progress_floor_penalty = None
            if selected != requested and not force_single_candidate_active and requested_score is not None:
                required_margin = float(switch_margin)
                if requested_blocked:
                    required_margin = 0.0
                if (
                    bool(preserve_turn_requests)
                    and requested in _TURN_PRIMITIVES
                    and selected != requested
                ):
                    requested_clearance_prob = requested_pred.get("clearance_blocked_prob")
                    selected_clearance_prob_for_turn = selected_pred.get("clearance_blocked_prob")
                    selected_allowed_turn_rerank = selected
                    selected_allowed_turn_pred = selected_pred
                    selected_allowed_turn_score = selected_score
                    selected_allowed_turn_body_penalty = selected_body_clearance_penalty
                    selected_allowed_turn_progress_penalty = selected_progress_floor_penalty
                    allow_backward_recovery = bool(
                        selected == "backward"
                        and requested_clearance_prob is not None
                        and selected_clearance_prob_for_turn is not None
                        and float(selected_clearance_prob_for_turn)
                        <= float(requested_clearance_prob) - 0.08
                    )
                    allowed_body_rerank = set(turn_body_rerank_primitives or ())
                    if requested_body_blocked and allowed_body_rerank and selected not in allowed_body_rerank:
                        allowed_candidates = [
                            item for item in scored
                            if item[1] != requested and item[1] in allowed_body_rerank
                        ]
                        if allowed_candidates:
                            (
                                selected_allowed_turn_score,
                                selected_allowed_turn_rerank,
                                selected_allowed_turn_pred,
                                selected_allowed_turn_body_penalty,
                                selected_allowed_turn_progress_penalty,
                            ) = allowed_candidates[0]
                            selected_clearance_prob_for_turn = selected_allowed_turn_pred.get(
                                "clearance_blocked_prob"
                            )
                            allow_backward_recovery = bool(
                                selected_allowed_turn_rerank == "backward"
                                and requested_clearance_prob is not None
                                and selected_clearance_prob_for_turn is not None
                                and float(selected_clearance_prob_for_turn)
                                <= float(requested_clearance_prob) - 0.08
                            )
                    allow_learned_body_rerank = bool(
                        requested_body_blocked
                        and (not allowed_body_rerank or (
                            selected_allowed_turn_rerank in allowed_body_rerank
                            and (
                                selected_allowed_turn_rerank != "backward"
                                or allow_backward_recovery
                            )
                        )
                        )
                    )
                    if not allow_backward_recovery and not allow_learned_body_rerank:
                        selected = requested
                        selected_pred = requested_pred
                        selected_score = requested_score
                        selected_body_clearance_penalty = requested_body_clearance_penalty
                        selected_progress_floor_penalty = requested_progress_floor_penalty
                        required_margin = float("inf")
                    elif selected_allowed_turn_rerank != selected:
                        selected = selected_allowed_turn_rerank
                        selected_pred = selected_allowed_turn_pred
                        selected_score = selected_allowed_turn_score
                        selected_body_clearance_penalty = selected_allowed_turn_body_penalty
                        selected_progress_floor_penalty = selected_allowed_turn_progress_penalty
                if (
                    bool(preserve_backward_requests)
                    and requested == "backward"
                    and selected != requested
                ):
                    preserve_backward = bool(
                        requested_body_clearance_penalty is None
                        or float(requested_body_clearance_penalty) <= 0.0
                    )
                    if not preserve_backward and preserve_backward_clearance_margin is not None:
                        requested_clearance_prob = requested_pred.get("clearance_blocked_prob")
                        selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
                        selected_is_clearly_safer = bool(
                            requested_clearance_prob is not None
                            and selected_clearance_prob is not None
                            and float(selected_clearance_prob)
                            <= float(requested_clearance_prob) - float(preserve_backward_clearance_margin)
                        )
                        preserve_backward = not selected_is_clearly_safer
                    if preserve_backward:
                        selected = requested
                        selected_pred = requested_pred
                        selected_score = requested_score
                        selected_body_clearance_penalty = requested_body_clearance_penalty
                        selected_progress_floor_penalty = requested_progress_floor_penalty
                        required_margin = float("inf")
                if (
                    requested in _TURN_PRIMITIVES
                    and selected in _TRANSLATING_PRIMITIVES
                    and bearing is not None
                    and abs(float(bearing)) > 0.45
                    and not _primitive_turns_toward_bearing(selected, bearing)
                ):
                    required_margin += 0.25
                if requested in _FORWARD_PRIMITIVES and selected in _TURN_PRIMITIVES and not requested_blocked:
                    translating = [item for item in scored if item[1] in _TRANSLATING_PRIMITIVES]
                    if translating and float(translating[0][0]) >= requested_score + float(switch_margin):
                        selected_score, selected, selected_pred, selected_body_clearance_penalty, selected_progress_floor_penalty = translating[0]
                    else:
                        required_margin += 0.25
                if (
                    bool(preserve_arc_requests)
                    and requested in ("arc_left", "arc_right")
                    and selected in _STRAIGHT_FORWARD_PRIMITIVES
                    and selected != requested
                ):
                    selected = requested
                    selected_pred = requested_pred
                    selected_score = requested_score
                    selected_body_clearance_penalty = requested_body_clearance_penalty
                    selected_progress_floor_penalty = requested_progress_floor_penalty
                    required_margin = float("inf")
                if selected != requested and float(selected_score) < requested_score + required_margin:
                    selected = requested
                    selected_pred = requested_pred
                    selected_score = requested_score
                    selected_body_clearance_penalty = requested_body_clearance_penalty
                    selected_progress_floor_penalty = requested_progress_floor_penalty
            selected_clearance_prob = selected_pred.get(
                "clearance_blocked_prob",
                selected_pred.get("blocked_prob"),
            )
            body_clearance_area_gate_active = bool(
                body_clearance_min_area is None
                or (
                    target_area is not None
                    and float(target_area) >= float(body_clearance_min_area)
                )
            )
            hard_veto_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and float(body_clearance_hard_veto_prob) <= 1.0
                and selected in set(body_clearance_hard_veto_selected_primitives or _TRANSLATING_PRIMITIVES)
                and selected_clearance_prob is not None
                and float(selected_clearance_prob) >= float(body_clearance_hard_veto_prob)
            )
            if hard_veto_allowed:
                allowed_primitives = set(body_clearance_hard_veto_primitives or ())
                target_turn_motion_veto = bool(
                    requested in _TURN_PRIMITIVES
                    and selected in _TRANSLATING_PRIMITIVES
                )
                hard_veto_candidates: list[tuple[float, float, str, dict[str, float], float, float, bool]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name == selected:
                        continue
                    if allowed_primitives and name not in allowed_primitives:
                        continue
                    clearance_prob = pred.get("clearance_blocked_prob")
                    if clearance_prob is None:
                        continue
                    strict_margin_ok = bool(
                        float(clearance_prob)
                        <= float(selected_clearance_prob) - float(body_clearance_hard_veto_margin)
                    )
                    relaxed_motion_ok = bool(
                        target_turn_motion_veto
                        and name != "hold"
                        and float(clearance_prob) <= float(selected_clearance_prob)
                        and float(pred.get("blocked_prob", 1.0)) < float(blocked_threshold)
                    )
                    if not strict_margin_ok and not relaxed_motion_ok:
                        continue
                    if float(clearance_prob) > float(body_clearance_hard_veto_replacement_cap):
                        continue
                    blocked_prob = float(pred.get("blocked_prob", 0.0))
                    if target_turn_motion_veto and name == requested:
                        primitive_bias = -0.12
                    elif name in _TURN_PRIMITIVES:
                        primitive_bias = 0.0
                    elif name == "backward":
                        primitive_bias = 0.08
                    elif name == "hold":
                        primitive_bias = 0.65 if target_turn_motion_veto else 0.25
                    else:
                        primitive_bias = 0.12
                    hard_veto_candidates.append((
                        float(clearance_prob) + 0.15 * blocked_prob + primitive_bias,
                        -float(score),
                        name,
                        pred,
                        float(body_clearance_penalty),
                        float(progress_floor_penalty),
                        bool(relaxed_motion_ok and not strict_margin_ok),
                    ))
                if hard_veto_candidates:
                    (
                        _,
                        _,
                        veto_selected,
                        veto_pred,
                        veto_body_penalty,
                        veto_progress_penalty,
                        veto_relaxed_motion,
                    ) = min(
                        hard_veto_candidates,
                        key=lambda item: (item[0], item[1]),
                    )
                    body_clearance_hard_vetoed = True
                    body_clearance_hard_veto_from = selected
                    body_clearance_hard_veto_from_prob = float(selected_clearance_prob)
                    body_clearance_hard_veto_relaxed_motion_replacement = bool(
                        veto_relaxed_motion
                    )
                    selected = veto_selected
                    selected_pred = veto_pred
                    selected_score = next(
                        (float(score) for score, name, _, _, _ in scored if name == selected),
                        selected_score,
                    )
                    selected_body_clearance_penalty = veto_body_penalty
                    selected_progress_floor_penalty = veto_progress_penalty
            selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
            arc_sweep_veto_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and not body_clearance_hard_vetoed
                and float(body_clearance_arc_sweep_veto_prob) <= 1.0
                and selected in set(
                    body_clearance_arc_sweep_veto_selected_primitives
                    or {"arc_left", "arc_right"}
                )
                and selected_clearance_prob is not None
                and float(selected_clearance_prob) >= float(body_clearance_arc_sweep_veto_prob)
            )
            if arc_sweep_veto_allowed:
                allowed_primitives = set(body_clearance_hard_veto_primitives or ())
                arc_veto_candidates: list[tuple[float, float, str, dict[str, float], float, float]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name == selected:
                        continue
                    if allowed_primitives and name not in allowed_primitives:
                        continue
                    clearance_prob = pred.get("clearance_blocked_prob")
                    if clearance_prob is None:
                        continue
                    if float(clearance_prob) > float(selected_clearance_prob) - float(
                        body_clearance_hard_veto_margin
                    ):
                        continue
                    if float(clearance_prob) > float(body_clearance_hard_veto_replacement_cap):
                        continue
                    blocked_prob = float(pred.get("blocked_prob", 0.0))
                    if name in _TURN_PRIMITIVES:
                        primitive_bias = 0.0
                    elif name == "backward":
                        primitive_bias = 0.08
                    elif name == "hold":
                        primitive_bias = 0.25
                    else:
                        primitive_bias = 0.12
                    arc_veto_candidates.append((
                        float(clearance_prob) + 0.15 * blocked_prob + primitive_bias,
                        -float(score),
                        name,
                        pred,
                        float(body_clearance_penalty),
                        float(progress_floor_penalty),
                    ))
                if arc_veto_candidates:
                    _, _, veto_selected, veto_pred, veto_body_penalty, veto_progress_penalty = min(
                        arc_veto_candidates,
                        key=lambda item: (item[0], item[1]),
                    )
                    body_clearance_arc_sweep_vetoed = True
                    body_clearance_arc_sweep_veto_from = selected
                    body_clearance_arc_sweep_veto_from_prob = float(selected_clearance_prob)
                    selected = veto_selected
                    selected_pred = veto_pred
                    selected_score = next(
                        (float(score) for score, name, _, _, _ in scored if name == selected),
                        selected_score,
                    )
                    selected_body_clearance_penalty = veto_body_penalty
                    selected_progress_floor_penalty = veto_progress_penalty
            selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
            yaw_direction_veto_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and float(body_clearance_yaw_direction_veto_prob) <= 1.0
                and selected in ("yaw_left", "yaw_right")
                and selected_clearance_prob is not None
                and float(selected_clearance_prob) >= float(body_clearance_yaw_direction_veto_prob)
            )
            if yaw_direction_veto_allowed:
                opposite_yaw = "yaw_right" if selected == "yaw_left" else "yaw_left"
                opposite_candidates = [
                    (score, name, pred, body_clearance_penalty, progress_floor_penalty)
                    for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored
                    if name == opposite_yaw and pred.get("clearance_blocked_prob") is not None
                ]
                if opposite_candidates:
                    score, name, pred, body_clearance_penalty, progress_floor_penalty = opposite_candidates[0]
                    opposite_prob = float(pred["clearance_blocked_prob"])
                    if opposite_prob <= float(selected_clearance_prob) - float(
                        body_clearance_yaw_direction_veto_margin
                    ):
                        body_clearance_yaw_direction_vetoed = True
                        body_clearance_yaw_direction_veto_from = selected
                        body_clearance_yaw_direction_veto_from_prob = float(selected_clearance_prob)
                        selected = name
                        selected_pred = pred
                        selected_score = float(score)
                        selected_body_clearance_penalty = float(body_clearance_penalty)
                        selected_progress_floor_penalty = float(progress_floor_penalty)
            selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
            selected_current_contact_escape_m = (
                body_clearance_current_contact_escape_m_by_primitive.get(selected)
                if body_clearance_current_contact_escape_m_by_primitive
                and selected in body_clearance_current_contact_escape_m_by_primitive
                else body_clearance_current_contact_escape_m
            )
            current_contact_escape_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and selected_current_contact_escape_m is not None
                and current_body_clearance_m is not None
                and float(current_body_clearance_m) <= float(selected_current_contact_escape_m)
                and selected
                in set(
                    body_clearance_current_contact_escape_primitives
                    or {"forward_fast", "forward_medium", "arc_left", "arc_right", "yaw_left", "yaw_right"}
                )
            )
            if (
                body_clearance_enabled
                and not force_single_candidate_active
                and body_clearance_current_contact_escape_m is not None
                and selected_current_contact_escape_m is not None
                and current_body_clearance_m is not None
                and float(current_body_clearance_m) > float(selected_current_contact_escape_m)
                and selected
                in set(
                    body_clearance_current_contact_escape_primitives
                    or {"forward_fast", "forward_medium", "arc_left", "arc_right", "yaw_left", "yaw_right"}
                )
            ):
                body_clearance_current_contact_escape_suppressed_reason = (
                    "primitive_clearance"
                )
            if current_contact_escape_allowed:
                selected_projected_ok, selected_projected_clearance, selected_projected_reason = (
                    _current_contact_projected_clearance_ok(
                        selected,
                        projected_clearances=body_clearance_current_contact_escape_projected_clearances,
                        current_body_clearance_m=current_body_clearance_m,
                        min_projected_clearance_m=(
                            body_clearance_current_contact_escape_min_projected_clearance_m
                        ),
                        min_projected_improvement_m=(
                            body_clearance_current_contact_escape_min_projected_improvement_m
                        ),
                    )
                )
                if (
                    body_clearance_current_contact_escape_projected_clearances is not None
                    and selected_projected_ok
                ):
                    body_clearance_current_contact_escape_suppressed_reason = (
                        "selected_projected_safe"
                    )
                    body_clearance_current_contact_escape_candidate = {
                        "primitive": selected,
                        "projected_clearance_m": _round_float(
                            selected_projected_clearance, 4
                        ),
                        "suppressed": True,
                    }
                    current_contact_escape_allowed = False
                elif selected_projected_reason is not None:
                    body_clearance_current_contact_escape_projected_rejections.append({
                        "primitive": selected,
                        "projected_clearance_m": _round_float(
                            selected_projected_clearance, 4
                        ),
                        "reason": selected_projected_reason,
                        "selected": True,
                    })
            if current_contact_escape_allowed:
                replacement_primitives = set(
                    body_clearance_current_contact_escape_replacements
                    or {"backward", "yaw_left", "yaw_right", "hold"}
                )
                escape_candidates = []
                escape_candidates_under_cap = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name not in replacement_primitives or name == selected:
                        continue
                    replacement_contact_prob = pred.get(
                        "clearance_blocked_prob",
                        pred.get("blocked_prob"),
                    )
                    replacement_under_cap = bool(
                        float(body_clearance_current_contact_escape_replacement_cap) > 1.0
                        or (
                            replacement_contact_prob is not None
                            and float(replacement_contact_prob)
                            <= float(body_clearance_current_contact_escape_replacement_cap)
                        )
                    )
                    if not replacement_under_cap:
                        body_clearance_current_contact_escape_projected_rejections.append({
                            "primitive": name,
                            "clearance_blocked_prob": _round_float(
                                replacement_contact_prob, 4
                            ),
                            "reason": "replacement_cap",
                            "selected": False,
                        })
                    projected_ok, projected_clearance, projected_reason = (
                        _current_contact_projected_clearance_ok(
                            name,
                            projected_clearances=body_clearance_current_contact_escape_projected_clearances,
                            current_body_clearance_m=current_body_clearance_m,
                            min_projected_clearance_m=(
                                body_clearance_current_contact_escape_min_projected_clearance_m
                            ),
                            min_projected_improvement_m=(
                                body_clearance_current_contact_escape_min_projected_improvement_m
                            ),
                        )
                    )
                    if not projected_ok:
                        body_clearance_current_contact_escape_projected_rejections.append({
                            "primitive": name,
                            "projected_clearance_m": _round_float(
                                projected_clearance, 4
                            ),
                            "reason": projected_reason,
                            "selected": False,
                        })
                        continue
                    escape_candidate = (
                        score,
                        name,
                        pred,
                        body_clearance_penalty,
                        progress_floor_penalty,
                    )
                    escape_candidates.append(escape_candidate)
                    if replacement_under_cap:
                        escape_candidates_under_cap.append(escape_candidate)
                if escape_candidates:
                    if (
                        body_clearance_current_contact_escape_require_replacement_under_cap
                        and float(body_clearance_current_contact_escape_replacement_cap) <= 1.0
                        and not escape_candidates_under_cap
                    ):
                        body_clearance_current_contact_escape_suppressed_reason = (
                            "replacement_cap"
                        )
                    else:
                        ranked_escape_candidates = (
                            escape_candidates_under_cap or escape_candidates
                        )
                        score, name, pred, body_clearance_penalty, progress_floor_penalty = min(
                            ranked_escape_candidates,
                            key=lambda item: (
                                _current_contact_escape_score(
                                    item[1],
                                    clearance_blocked_prob=item[2].get(
                                        "clearance_blocked_prob",
                                        item[2].get("blocked_prob"),
                                    ),
                                    blocked_prob=item[2].get("blocked_prob"),
                                    candidate_score=item[0],
                                ),
                                float(
                                    item[2].get(
                                        "clearance_blocked_prob",
                                        item[2].get("blocked_prob", 1.0),
                                    )
                                ),
                                -float(item[0]),
                            ),
                        )
                        replacement_contact_prob = pred.get(
                            "clearance_blocked_prob",
                            pred.get("blocked_prob"),
                        )
                        body_clearance_current_contact_escape = True
                        body_clearance_current_contact_escape_from = selected
                        body_clearance_current_contact_escape_from_prob = (
                            None if selected_clearance_prob is None else float(selected_clearance_prob)
                        )
                        body_clearance_current_contact_escape_to_prob = (
                            None
                            if replacement_contact_prob is None
                            else float(replacement_contact_prob)
                        )
                        body_clearance_current_contact_escape_candidate = {
                            "primitive": name,
                            "clearance_blocked_prob": _round_float(
                                replacement_contact_prob, 4
                            ),
                            "blocked_prob": _round_float(pred.get("blocked_prob"), 4),
                            "score": _round_float(float(score), 4),
                            "projected_clearance_m": _round_float(
                                (
                                    body_clearance_current_contact_escape_projected_clearances
                                    or {}
                                ).get(name),
                                4,
                            ),
                            "threshold_m": _round_float(
                                selected_current_contact_escape_m, 4
                            ),
                            "replacement_cap": _round_float(
                                body_clearance_current_contact_escape_replacement_cap,
                                4,
                            ),
                            "replacement_cap_relaxed": bool(
                                float(body_clearance_current_contact_escape_replacement_cap) <= 1.0
                                and not escape_candidates_under_cap
                            ),
                        }
                        selected = name
                        selected_pred = pred
                        selected_score = float(score)
                        selected_body_clearance_penalty = float(body_clearance_penalty)
                        selected_progress_floor_penalty = float(progress_floor_penalty)
                elif body_clearance_current_contact_escape_suppressed_reason is None:
                    body_clearance_current_contact_escape_suppressed_reason = (
                        "no_scored_candidate"
                    )
            selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
            yaw_contact_veto_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and float(body_clearance_yaw_contact_veto_prob) <= 1.0
                and selected in ("yaw_left", "yaw_right")
                and selected_clearance_prob is not None
                and float(selected_clearance_prob) >= float(body_clearance_yaw_contact_veto_prob)
            )
            if yaw_contact_veto_allowed:
                # Yaw-in-place with the swept body already in contact can lever
                # the base over a wall lip in a single tick (unrecoverable
                # capsize). When the learned clearance head flags the yaw
                # itself, back out if that is safer; otherwise hold for a tick
                # when the learned head says every movement primitive is risky.
                contact_fallback_candidates = [
                    (score, name, pred, body_clearance_penalty, progress_floor_penalty)
                    for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored
                    if name in ("backward", "hold") and pred.get("clearance_blocked_prob") is not None
                ]
                contact_fallbacks: list[tuple[float, float, str, dict[str, float], float, float]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in contact_fallback_candidates:
                    fallback_prob = float(pred["clearance_blocked_prob"])
                    if name == "backward":
                        allowed = fallback_prob < float(selected_clearance_prob)
                        priority = 0.0
                    else:
                        allowed = fallback_prob <= float(selected_clearance_prob) - float(
                            body_clearance_yaw_direction_veto_margin
                        )
                        priority = 1.0
                    if allowed:
                        contact_fallbacks.append((
                            priority,
                            fallback_prob,
                            name,
                            pred,
                            float(body_clearance_penalty),
                            float(progress_floor_penalty),
                        ))
                if contact_fallbacks:
                    _, _, name, pred, body_clearance_penalty, progress_floor_penalty = min(
                        contact_fallbacks,
                        key=lambda item: (item[0], item[1]),
                    )
                    score = next(
                        (float(score) for score, scored_name, _, _, _ in scored if scored_name == name),
                        selected_score,
                    )
                    if name != selected:
                        body_clearance_yaw_contact_vetoed = True
                        body_clearance_yaw_contact_veto_from = selected
                        body_clearance_yaw_contact_veto_from_prob = float(selected_clearance_prob)
                        selected = name
                        selected_pred = pred
                        selected_score = float(score)
                        selected_body_clearance_penalty = float(body_clearance_penalty)
                        selected_progress_floor_penalty = float(progress_floor_penalty)
            selected_prob_for_blocked_veto = selected_pred.get("blocked_prob")
            selected_blocked_for_veto = bool(
                bool(blocked_hard_veto)
                and not force_single_candidate_active
                and selected in _FORWARD_PRIMITIVES
                and blocked_veto_bearing_ok
                and (
                    not blocked_hard_veto_selected_primitives
                    or selected in blocked_hard_veto_selected_primitives
                )
                and selected_prob_for_blocked_veto is not None
                and float(selected_prob_for_blocked_veto) >= float(blocked_threshold)
            )
            if selected_blocked_for_veto:
                allowed_primitives = set(
                    blocked_hard_veto_primitives or {"yaw_left", "yaw_right", "backward"}
                )
                preferred_turn = None
                if bearing is not None and abs(float(bearing)) >= 0.08:
                    preferred_turn = "yaw_left" if float(bearing) > 0.0 else "yaw_right"
                elif requested in _TURN_PRIMITIVES:
                    preferred_turn = requested
                blocked_veto_candidates: list[tuple[float, str, dict[str, float], float, float]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name == selected:
                        continue
                    if allowed_primitives and name not in allowed_primitives:
                        continue
                    if (
                        name in _FORWARD_PRIMITIVES
                        and pred.get("blocked_prob") is not None
                        and float(pred["blocked_prob"]) >= float(blocked_threshold)
                    ):
                        continue
                    if float(body_clearance_penalty) > 0.0:
                        continue
                    clearance_prob = pred.get("clearance_blocked_prob")
                    blocked_prob = float(pred.get("blocked_prob", 0.0))
                    if name == preferred_turn:
                        primitive_bias = -0.06
                    elif name in _TURN_PRIMITIVES:
                        primitive_bias = 0.02
                    elif name == "backward":
                        primitive_bias = 0.10
                    elif name == "hold":
                        primitive_bias = 0.18
                    else:
                        primitive_bias = 0.14
                    blocked_veto_candidates.append((
                        (1.0 if clearance_prob is None else float(clearance_prob))
                        + 0.20 * blocked_prob
                        + primitive_bias
                        - 0.02 * float(score),
                        name,
                        pred,
                        float(body_clearance_penalty),
                        float(progress_floor_penalty),
                    ))
                if blocked_veto_candidates:
                    _, veto_selected, veto_pred, veto_body_penalty, veto_progress_penalty = min(
                        blocked_veto_candidates,
                        key=lambda item: item[0],
                    )
                    blocked_hard_vetoed = True
                    blocked_hard_veto_from = selected
                    blocked_hard_veto_from_prob = float(selected_prob_for_blocked_veto)
                    selected = veto_selected
                    selected_pred = veto_pred
                    selected_score = next(
                        (float(score) for score, name, _, _, _ in scored if name == selected),
                        selected_score,
                    )
                    selected_body_clearance_penalty = veto_body_penalty
                    selected_progress_floor_penalty = veto_progress_penalty
            selected_prob_for_low_progress = selected_pred.get("blocked_prob")
            selected_progress_for_low_progress = selected_pred.get("progress_m")
            selected_low_progress_for_veto = bool(
                bool(low_progress_hard_veto)
                and not force_single_candidate_active
                and selected in _FORWARD_PRIMITIVES
                and forward_progress_floor is not None
                and selected_progress_for_low_progress is not None
                and float(selected_progress_for_low_progress) < float(forward_progress_floor)
                and (
                    (
                        forward_progress_floor_force_below is not None
                        and float(selected_progress_for_low_progress) < float(forward_progress_floor_force_below)
                    )
                    or forward_progress_floor_min_blocked_prob is None
                    or (
                        selected_prob_for_low_progress is not None
                        and float(selected_prob_for_low_progress) >= float(forward_progress_floor_min_blocked_prob)
                    )
                )
            )
            if selected_low_progress_for_veto:
                allowed_primitives = set(low_progress_hard_veto_primitives or ())
                preferred_turn = None
                if bearing is not None and abs(float(bearing)) >= 0.08:
                    preferred_turn = "yaw_left" if float(bearing) > 0.0 else "yaw_right"
                low_progress_candidates: list[tuple[float, str, dict[str, float], float, float]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name == selected:
                        continue
                    if allowed_primitives and name not in allowed_primitives:
                        continue
                    clearance_prob = pred.get("clearance_blocked_prob")
                    blocked_prob = float(pred.get("blocked_prob", 0.0))
                    if name == preferred_turn:
                        primitive_bias = -0.05
                    elif name in _TURN_PRIMITIVES:
                        primitive_bias = 0.02
                    elif name == "backward":
                        primitive_bias = 0.10
                    elif name == "hold":
                        primitive_bias = 0.18
                    else:
                        primitive_bias = 0.12
                    low_progress_candidates.append((
                        (1.0 if clearance_prob is None else float(clearance_prob))
                        + 0.20 * blocked_prob
                        + primitive_bias,
                        name,
                        pred,
                        float(body_clearance_penalty),
                        float(progress_floor_penalty),
                    ))
                if low_progress_candidates:
                    _, veto_selected, veto_pred, veto_body_penalty, veto_progress_penalty = min(
                        low_progress_candidates,
                        key=lambda item: item[0],
                    )
                    low_progress_hard_vetoed = True
                    low_progress_hard_veto_from = selected
                    low_progress_hard_veto_from_progress = float(selected_progress_for_low_progress)
                    selected = veto_selected
                    selected_pred = veto_pred
                    selected_score = next(
                        (float(score) for score, name, _, _, _ in scored if name == selected),
                        selected_score,
                    )
                    selected_body_clearance_penalty = veto_body_penalty
                    selected_progress_floor_penalty = veto_progress_penalty
            selected_clearance_prob = selected_pred.get("clearance_blocked_prob")
            saturated_veto_allowed = (
                body_clearance_enabled
                and not force_single_candidate_active
                and not body_clearance_hard_vetoed
                and float(body_clearance_saturated_veto_prob) <= 1.0
                and selected in set(body_clearance_saturated_veto_selected_primitives or ())
                and selected_clearance_prob is not None
                and float(selected_clearance_prob) >= float(body_clearance_saturated_veto_prob)
            )
            if saturated_veto_allowed:
                allowed_primitives = set(body_clearance_saturated_veto_primitives or ())
                preferred_turn = None
                if bearing is not None and abs(float(bearing)) >= 0.08:
                    preferred_turn = "yaw_left" if float(bearing) > 0.0 else "yaw_right"
                elif requested in _TURN_PRIMITIVES:
                    preferred_turn = requested
                elif selected in ("arc_left", "yaw_left"):
                    preferred_turn = "yaw_left"
                elif selected in ("arc_right", "yaw_right"):
                    preferred_turn = "yaw_right"
                saturated_candidates: list[tuple[float, str, dict[str, float], float, float]] = []
                for score, name, pred, body_clearance_penalty, progress_floor_penalty in scored:
                    if name == selected:
                        continue
                    if allowed_primitives and name not in allowed_primitives:
                        continue
                    clearance_prob = pred.get("clearance_blocked_prob")
                    if clearance_prob is None:
                        continue
                    if float(clearance_prob) > float(selected_clearance_prob) + float(body_clearance_saturated_veto_spread):
                        continue
                    blocked_prob = float(pred.get("blocked_prob", 0.0))
                    if name == preferred_turn:
                        primitive_bias = -0.05
                    elif name in _TURN_PRIMITIVES:
                        primitive_bias = 0.02
                    elif name == "backward":
                        primitive_bias = 0.08
                    elif name == "hold":
                        primitive_bias = 0.16
                    else:
                        primitive_bias = 0.12
                    saturated_candidates.append((
                        float(clearance_prob) + 0.15 * blocked_prob + primitive_bias,
                        name,
                        pred,
                        float(body_clearance_penalty),
                        float(progress_floor_penalty),
                    ))
                if saturated_candidates:
                    _, veto_selected, veto_pred, veto_body_penalty, veto_progress_penalty = min(
                        saturated_candidates,
                        key=lambda item: item[0],
                    )
                    body_clearance_saturated_vetoed = True
                    body_clearance_saturated_veto_from = selected
                    body_clearance_saturated_veto_from_prob = float(selected_clearance_prob)
                    selected = veto_selected
                    selected_pred = veto_pred
                    selected_score = next(
                        (float(score) for score, name, _, _, _ in scored if name == selected),
                        selected_score,
                    )
                    selected_body_clearance_penalty = veto_body_penalty
                    selected_progress_floor_penalty = veto_progress_penalty
        else:
            selected_pred = requested_pred
    else:
        selected_pred = requested_pred
        candidates.append({
            "primitive": requested,
            "blocked_prob": _round_float(requested_prob, 4),
            "progress_m": _round_float(requested_pred.get("progress_m"), 4),
            "score": None,
            "body_clearance_penalty": None,
            "blocked": requested_blocked,
        })
    selected_prob = selected_pred.get("blocked_prob")
    selected_alias = _prediction_alias_for_primitive(selected, predictions)
    selected_progress = selected_pred.get("progress_m")
    selected_low_progress = bool(
        selected in _FORWARD_PRIMITIVES
        and forward_progress_floor is not None
        and selected_progress is not None
        and float(selected_progress) < float(forward_progress_floor)
        and (
            (
                forward_progress_floor_force_below is not None
                and float(selected_progress) < float(forward_progress_floor_force_below)
            )
            or
            forward_progress_floor_min_blocked_prob is None
            or (
                selected_prob is not None
                and float(selected_prob) >= float(forward_progress_floor_min_blocked_prob)
            )
        )
    )
    selected_blocked = bool(
        selected in _FORWARD_PRIMITIVES
        and selected_prob is not None
        and (
            float(selected_prob) >= float(blocked_threshold)
            or selected_low_progress
        )
        or (
            selected_body_clearance_penalty is not None
            and float(selected_body_clearance_penalty) > 0.0
        )
    )
    guard = {
        "enabled": bool(enabled),
        "source": "learned_action_outcome",
        "requested": requested,
        "requested_prediction_alias": requested_alias if requested_alias != requested else None,
        "selected": selected,
        "selected_prediction_alias": selected_alias if selected_alias != selected else None,
        "vetoed": bool(selected != requested),
        "force_escape": bool(force_single_candidate_active),
        "force_context": bool(force_escape),
        "force_single_candidate": bool(force_single_candidate_active),
        "candidate_names_override": list(candidate_names_override or []),
        "front_blocked_prob": _round_float(requested_prob, 4),
        "outcome_blocked_prob": _round_float(requested_pred.get("outcome_blocked_prob"), 4),
        "clearance_blocked_prob": _round_float(requested_pred.get("clearance_blocked_prob"), 4),
        "threshold": _round_float(float(blocked_threshold), 4),
        "requested_min_clearance_m": None,
        "requested_feasible_fraction": None,
        "requested_blocked": requested_blocked,
        "requested_progress_m": _round_float(requested_pred.get("progress_m"), 4),
        "requested_score": _round_float(requested_score, 4),
        "requested_body_clearance_penalty": _round_float(requested_body_clearance_penalty, 4),
        "requested_progress_floor_penalty": _round_float(requested_progress_floor_penalty, 4),
        "requested_low_progress": bool(requested_low_progress),
        "selected_min_clearance_m": None,
        "selected_feasible_fraction": None,
        "selected_blocked": selected_blocked,
        "selected_outcome_blocked_prob": _round_float(selected_pred.get("outcome_blocked_prob"), 4),
        "selected_clearance_blocked_prob": _round_float(selected_pred.get("clearance_blocked_prob"), 4),
        "selected_progress_m": _round_float(selected_pred.get("progress_m"), 4),
        "selected_score": _round_float(selected_score, 4),
        "selected_body_clearance_penalty": _round_float(selected_body_clearance_penalty, 4),
        "selected_progress_floor_penalty": _round_float(selected_progress_floor_penalty, 4),
        "selected_low_progress": bool(selected_low_progress),
        "body_clearance_enabled": bool(body_clearance_enabled),
        "body_clearance_target_area": _round_float(float(body_clearance_target_area), 4),
        "body_clearance_min_area": _round_float(body_clearance_min_area, 4),
        "body_clearance_area_gate_active": bool(body_clearance_area_gate_active),
        "body_clearance_near_forward_prob_floor": _round_float(body_clearance_near_forward_prob_floor, 4),
        "body_clearance_near_forward_prob_weight": _round_float(body_clearance_near_forward_prob_weight, 4),
        "body_clearance_near_yaw_prob_floor": _round_float(body_clearance_near_yaw_prob_floor, 4),
        "body_clearance_yaw_always": bool(body_clearance_yaw_always),
        "body_clearance_yaw_penalty_weight": _round_float(float(body_clearance_yaw_penalty_weight), 4),
        "body_clearance_hard_veto": bool(body_clearance_hard_vetoed),
        "body_clearance_hard_veto_relaxed_motion_replacement": bool(
            body_clearance_hard_veto_relaxed_motion_replacement
        ),
        "body_clearance_hard_veto_prob": _round_float(float(body_clearance_hard_veto_prob), 4),
        "body_clearance_hard_veto_margin": _round_float(float(body_clearance_hard_veto_margin), 4),
        "body_clearance_hard_veto_replacement_cap": _round_float(
            float(body_clearance_hard_veto_replacement_cap), 4
        ),
        "body_clearance_hard_veto_primitives": sorted(body_clearance_hard_veto_primitives or ()),
        "body_clearance_hard_veto_selected_primitives": sorted(
            body_clearance_hard_veto_selected_primitives or _TRANSLATING_PRIMITIVES
        ),
        "selected_before_body_clearance_hard_veto": body_clearance_hard_veto_from,
        "selected_before_body_clearance_hard_veto_prob": _round_float(
            body_clearance_hard_veto_from_prob, 4
        ),
        "body_clearance_arc_sweep_veto": bool(body_clearance_arc_sweep_vetoed),
        "body_clearance_arc_sweep_veto_prob": _round_float(
            float(body_clearance_arc_sweep_veto_prob), 4
        ),
        "body_clearance_arc_sweep_veto_selected_primitives": sorted(
            body_clearance_arc_sweep_veto_selected_primitives
            or {"arc_left", "arc_right"}
        ),
        "selected_before_body_clearance_arc_sweep_veto": body_clearance_arc_sweep_veto_from,
        "selected_before_body_clearance_arc_sweep_veto_prob": _round_float(
            body_clearance_arc_sweep_veto_from_prob, 4
        ),
        "body_clearance_saturated_veto": bool(body_clearance_saturated_vetoed),
        "body_clearance_saturated_veto_prob": _round_float(float(body_clearance_saturated_veto_prob), 4),
        "body_clearance_saturated_veto_spread": _round_float(float(body_clearance_saturated_veto_spread), 4),
        "body_clearance_saturated_veto_primitives": sorted(body_clearance_saturated_veto_primitives or ()),
        "body_clearance_saturated_veto_selected_primitives": sorted(
            body_clearance_saturated_veto_selected_primitives or ()
        ),
        "selected_before_body_clearance_saturated_veto": body_clearance_saturated_veto_from,
        "selected_before_body_clearance_saturated_veto_prob": _round_float(
            body_clearance_saturated_veto_from_prob, 4
        ),
        "body_clearance_yaw_direction_veto": bool(body_clearance_yaw_direction_vetoed),
        "body_clearance_yaw_direction_veto_prob": _round_float(
            float(body_clearance_yaw_direction_veto_prob), 4
        ),
        "body_clearance_yaw_direction_veto_margin": _round_float(
            float(body_clearance_yaw_direction_veto_margin), 4
        ),
        "body_clearance_yaw_contact_veto": bool(body_clearance_yaw_contact_vetoed),
        "body_clearance_yaw_contact_veto_prob": _round_float(
            float(body_clearance_yaw_contact_veto_prob), 4
        ),
        "current_body_clearance_m": _round_float(current_body_clearance_m, 4),
        "body_clearance_current_contact_escape": bool(body_clearance_current_contact_escape),
        "body_clearance_current_contact_escape_m": (
            None
            if body_clearance_current_contact_escape_m is None
            else _round_float(float(body_clearance_current_contact_escape_m), 4)
        ),
        "body_clearance_current_contact_escape_min_projected_clearance_m": (
            None
            if body_clearance_current_contact_escape_min_projected_clearance_m is None
            else _round_float(
                float(body_clearance_current_contact_escape_min_projected_clearance_m),
                4,
            )
        ),
        "body_clearance_current_contact_escape_min_projected_improvement_m": _round_float(
            float(body_clearance_current_contact_escape_min_projected_improvement_m),
            4,
        ),
        "body_clearance_current_contact_escape_m_by_primitive": dict(
            sorted((body_clearance_current_contact_escape_m_by_primitive or {}).items())
        ),
        "body_clearance_current_contact_escape_primitives": sorted(
            body_clearance_current_contact_escape_primitives
            or {"forward_fast", "forward_medium", "arc_left", "arc_right", "yaw_left", "yaw_right"}
        ),
        "body_clearance_current_contact_escape_replacements": sorted(
            body_clearance_current_contact_escape_replacements
            or {"backward", "yaw_left", "yaw_right", "hold"}
        ),
        "body_clearance_current_contact_escape_replacement_cap": _round_float(
            body_clearance_current_contact_escape_replacement_cap,
            4,
        ),
        "body_clearance_current_contact_escape_require_replacement_under_cap": bool(
            body_clearance_current_contact_escape_require_replacement_under_cap
        ),
        "selected_before_body_clearance_current_contact_escape": (
            body_clearance_current_contact_escape_from
        ),
        "selected_before_body_clearance_current_contact_escape_prob": _round_float(
            body_clearance_current_contact_escape_from_prob, 4
        ),
        "selected_after_body_clearance_current_contact_escape_prob": _round_float(
            body_clearance_current_contact_escape_to_prob, 4
        ),
        "body_clearance_current_contact_escape_candidate": (
            body_clearance_current_contact_escape_candidate
        ),
        "body_clearance_current_contact_escape_projected_rejections": (
            body_clearance_current_contact_escape_projected_rejections[:8]
        ),
        "body_clearance_current_contact_escape_suppressed_reason": (
            body_clearance_current_contact_escape_suppressed_reason
        ),
        "selected_before_body_clearance_yaw_contact_veto": body_clearance_yaw_contact_veto_from,
        "selected_before_body_clearance_yaw_contact_veto_prob": _round_float(
            body_clearance_yaw_contact_veto_from_prob, 4
        ),
        "selected_before_body_clearance_yaw_direction_veto": body_clearance_yaw_direction_veto_from,
        "selected_before_body_clearance_yaw_direction_veto_prob": _round_float(
            body_clearance_yaw_direction_veto_from_prob, 4
        ),
        "blocked_hard_veto": bool(blocked_hard_vetoed),
        "blocked_hard_veto_enabled": bool(blocked_hard_veto),
        "blocked_hard_veto_primitives": sorted(
            blocked_hard_veto_primitives or {"yaw_left", "yaw_right", "backward"}
        ),
        "blocked_hard_veto_selected_primitives": sorted(
            blocked_hard_veto_selected_primitives or _FORWARD_PRIMITIVES
        ),
        "blocked_hard_veto_max_abs_bearing": _round_float(
            blocked_hard_veto_max_abs_bearing, 4
        ),
        "blocked_hard_veto_reference_bearing": _round_float(
            blocked_veto_reference_bearing, 4
        ),
        "blocked_hard_veto_bearing_ok": bool(blocked_veto_bearing_ok),
        "selected_before_blocked_hard_veto": blocked_hard_veto_from,
        "selected_before_blocked_hard_veto_prob": _round_float(
            blocked_hard_veto_from_prob, 4
        ),
        "forward_progress_floor": _round_float(forward_progress_floor, 4),
        "forward_progress_floor_min_blocked_prob": _round_float(forward_progress_floor_min_blocked_prob, 4),
        "forward_progress_floor_force_below": _round_float(forward_progress_floor_force_below, 4),
        "forward_progress_floor_penalty": _round_float(float(forward_progress_floor_penalty), 4),
        "low_progress_hard_veto": bool(low_progress_hard_vetoed),
        "low_progress_hard_veto_enabled": bool(low_progress_hard_veto),
        "low_progress_hard_veto_primitives": sorted(low_progress_hard_veto_primitives or ()),
        "selected_before_low_progress_hard_veto": low_progress_hard_veto_from,
        "selected_before_low_progress_hard_veto_progress_m": _round_float(
            low_progress_hard_veto_from_progress, 4
        ),
        "progress_floor_prefer_yaw": bool(progress_floor_prefer_yaw),
        "preserve_turn_requests": bool(preserve_turn_requests),
        "preserve_arc_requests": bool(preserve_arc_requests),
        "turn_body_rerank_primitives": sorted(turn_body_rerank_primitives or ()),
        "preserve_straight_requests": bool(preserve_straight_requests),
        "preserve_backward_requests": bool(preserve_backward_requests),
        "preserve_backward_clearance_margin": _round_float(
            preserve_backward_clearance_margin, 4
        ),
        "switch_margin": _round_float(float(switch_margin), 4),
        "candidates": sorted(candidates, key=lambda item: item.get("score") if item.get("score") is not None else -999.0, reverse=True),
    }
    return selected, guard


def _los_clear(grid, a, b, stop_short_m: float = 0.35, step: float = 0.05) -> bool:
    dx, dy = b[0] - a[0], b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return False
    ux, uy = dx / dist, dy / dist
    end = max(0.0, dist - stop_short_m)
    n = max(1, int(end / step))
    for i in range(n + 1):
        t = i * step
        if not grid.is_free((a[0] + ux * t, a[1] + uy * t)):
            return False
    return True


def _los_placement(grid, green_xy, free_cells, dmin: float = 0.65, dmax: float = 1.15):
    """A free standoff cell with clear line-of-sight to the target (privileged
    exploration scaffold: ensures the robot sees the target once to bind it)."""
    cands = []
    for (cx, cy) in free_cells.values():
        d = math.hypot(cx - green_xy[0], cy - green_xy[1])
        if dmin <= d <= dmax and _los_clear(grid, (cx, cy), green_xy):
            cands.append((d, (cx, cy)))
    if not cands:
        return None
    cands.sort()
    px, py = cands[len(cands) // 2][1]
    return np.array([px, py, 0.375], dtype=np.float32), math.atan2(green_xy[1] - py, green_xy[0] - px)


class FrontierExplorer:
    """Grid coverage scaffold over a coarse free-cell graph.

    The target remains hidden from the scaffold. It only chooses unvisited free
    nav-cells, while memory/perception decide when a landmark has been found.
    """

    def __init__(
        self,
        grid,
        bounds,
        step_m: float = 0.3,
        goal_policy: str = "nearest",
        yaw_bearing_threshold: float = 0.5,
        forward_bearing_threshold: float = 0.18,
        lookahead_m: float = 0.35,
        forward_primitive: str = "forward_medium",
        coverage_lookahead_cells: int = 8,
        dfs_neighbor_order: str = "nesw",
        scan_interval: int = 24,
        scan_len: int = 7,
        scan_primitive: str = "yaw",
        route_waypoints: list[tuple[float, float]] | None = None,
        route_start_after_claims: int = 0,
        route_advance_m: float = 0.55,
        standoff_route: bool = False,
        standoff_targets: dict[str, np.ndarray] | None = None,
        standoff_grid: InflatedOccupancyGrid | None = None,
        standoff_scene_graph: Any | None = None,
        standoff_m: float = 1.05,
        standoff_lookahead_m: float = 0.55,
        standoff_replan_interval: int = 12,
        standoff_candidates: int = 16,
        standoff_arrival_m: float = 0.45,
        standoff_path_spacing_m: float = 0.30,
        standoff_clearance_weight: float = 0.0,
        standoff_clearance_target_m: float = 0.0,
        standoff_body_route_clearance_weight: float = 0.0,
        standoff_body_route_clearance_target_m: float = 0.0,
        standoff_body_route_ignore_start_m: float = 0.0,
        standoff_cardinal_route: bool = False,
        standoff_corner_guard: bool = False,
        standoff_corner_commit_m: float = 0.12,
        standoff_corner_standoff_m: float = 0.0,
        standoff_allow_arcs: bool = False,
        standoff_arc_min_bearing: float | None = None,
        standoff_arc_max_bearing: float | None = None,
        standoff_arc_min_target_dist_m: float = 0.0,
        standoff_heading_mode: str = "target",
        standoff_heading_lookahead_m: float = 0.35,
        standoff_prefix_snap_start: bool = False,
        standoff_snap_start_min_dist_m: float = 0.05,
        standoff_body_check: bool = False,
        standoff_body_lookahead_m: float = 0.30,
        standoff_body_min_clearance_m: float = -0.02,
        standoff_body_recovery_clearance_m: float | None = None,
        standoff_intent_smoothing: bool = False,
        standoff_sticky_target_ticks: int = 0,
        standoff_sticky_target_release_m: float = 0.18,
        standoff_yaw_enter_threshold: float | None = None,
        standoff_yaw_exit_threshold: float | None = None,
        standoff_yaw_flip_threshold: float | None = None,
        body_forward_m: float = 0.35,
        body_half_width_m: float = 0.18,
        body_probe_margin_m: float = 0.03,
    ):
        self.step = step_m
        self.goal_policy = goal_policy
        self.yaw_bearing_threshold = float(yaw_bearing_threshold)
        self.forward_bearing_threshold = float(forward_bearing_threshold)
        self.lookahead_m = float(lookahead_m)
        self.forward_primitive = forward_primitive
        self.coverage_lookahead_cells = max(1, int(coverage_lookahead_cells))
        self.dfs_neighbor_order = str(dfs_neighbor_order)
        self.coverage_neighbor_dirs: tuple[tuple[int, int], ...] | None = None
        self.scan_interval = max(0, int(scan_interval))
        self.scan_len = max(0, int(scan_len))
        self.scan_primitive = str(scan_primitive)
        self.route_waypoints = list(route_waypoints or [])
        self.route_start_after_claims = max(0, int(route_start_after_claims))
        self.route_advance_m = max(0.05, float(route_advance_m))
        self.route_idx = 0
        self.route_claim_count = 0
        self.standoff_route = bool(standoff_route)
        self.standoff_targets = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in (standoff_targets or {}).items()
        }
        self.standoff_grid = standoff_grid
        self.standoff_scene_graph = standoff_scene_graph
        self.standoff_m = max(0.05, float(standoff_m))
        self.standoff_lookahead_m = max(0.05, float(standoff_lookahead_m))
        self.standoff_replan_interval = max(1, int(standoff_replan_interval))
        self.standoff_candidates = max(4, int(standoff_candidates))
        self.standoff_arrival_m = max(0.05, float(standoff_arrival_m))
        self.standoff_path_spacing_m = max(0.05, float(standoff_path_spacing_m))
        self.standoff_clearance_weight = max(0.0, float(standoff_clearance_weight))
        self.standoff_clearance_target_m = float(standoff_clearance_target_m)
        self.standoff_body_route_clearance_weight = max(
            0.0, float(standoff_body_route_clearance_weight)
        )
        self.standoff_body_route_clearance_target_m = float(standoff_body_route_clearance_target_m)
        self.standoff_body_route_ignore_start_m = max(0.0, float(standoff_body_route_ignore_start_m))
        self.standoff_cardinal_route = bool(standoff_cardinal_route)
        self.standoff_corner_guard = bool(standoff_corner_guard)
        self.standoff_corner_commit_m = max(0.0, float(standoff_corner_commit_m))
        self.standoff_corner_standoff_m = max(0.0, float(standoff_corner_standoff_m))
        self.standoff_allow_arcs = bool(standoff_allow_arcs)
        self.standoff_arc_min_bearing = (
            None if standoff_arc_min_bearing is None else max(0.0, float(standoff_arc_min_bearing))
        )
        self.standoff_arc_max_bearing = (
            None if standoff_arc_max_bearing is None else max(0.0, float(standoff_arc_max_bearing))
        )
        self.standoff_arc_min_target_dist_m = max(0.0, float(standoff_arc_min_target_dist_m))
        self.standoff_heading_mode = str(standoff_heading_mode).strip().lower()
        self.standoff_heading_lookahead_m = max(0.05, float(standoff_heading_lookahead_m))
        self.standoff_prefix_snap_start = bool(standoff_prefix_snap_start)
        self.standoff_snap_start_min_dist_m = max(0.0, float(standoff_snap_start_min_dist_m))
        self.standoff_body_check = bool(standoff_body_check)
        self.standoff_body_lookahead_m = max(0.05, float(standoff_body_lookahead_m))
        self.standoff_body_min_clearance_m = float(standoff_body_min_clearance_m)
        self.standoff_body_recovery_clearance_m = (
            None
            if standoff_body_recovery_clearance_m is None
            else float(standoff_body_recovery_clearance_m)
        )
        self.standoff_intent_smoothing = bool(standoff_intent_smoothing)
        self.standoff_sticky_target_ticks = max(0, int(standoff_sticky_target_ticks))
        self.standoff_sticky_target_release_m = max(0.0, float(standoff_sticky_target_release_m))
        self.standoff_yaw_enter_threshold = (
            None if standoff_yaw_enter_threshold is None else max(0.0, float(standoff_yaw_enter_threshold))
        )
        self.standoff_yaw_exit_threshold = (
            None if standoff_yaw_exit_threshold is None else max(0.0, float(standoff_yaw_exit_threshold))
        )
        self.standoff_yaw_flip_threshold = (
            None if standoff_yaw_flip_threshold is None else max(0.0, float(standoff_yaw_flip_threshold))
        )
        self.body_forward_m = max(0.0, float(body_forward_m))
        self.body_half_width_m = max(0.0, float(body_half_width_m))
        self.body_probe_margin_m = max(0.0, float(body_probe_margin_m))
        self.standoff_target_color: str | None = None
        self.standoff_goal_xy: tuple[float, float] | None = None
        self.standoff_beacon_xy: tuple[float, float] | None = None
        self.standoff_path: tuple[tuple[float, float], ...] = ()
        self.standoff_path_idx = 0
        self.standoff_heading_xy: tuple[float, float] | None = None
        self.standoff_plan_age = 0
        self.standoff_replans = 0
        self.standoff_plan_failures = 0
        self.standoff_selected_clearance_m: float | None = None
        self.standoff_selected_body_route_clearance_m: float | None = None
        self.standoff_blocked_cells: set[tuple[int, int]] = set()
        self.standoff_blocked_waypoints = 0
        self.standoff_body_vetoes = 0
        self.standoff_body_forward_clearance_m: float | None = None
        self.standoff_body_current_clearance_m: float | None = None
        self.standoff_body_left_clearance_m: float | None = None
        self.standoff_body_right_clearance_m: float | None = None
        self.standoff_body_backward_clearance_m: float | None = None
        self.standoff_corner_guard_caps = 0
        self.standoff_corner_standoff_caps = 0
        self.standoff_tangent_heading_ticks = 0
        self.standoff_snap_start_prefixes = 0
        self.standoff_sticky_target_xy: tuple[float, float] | None = None
        self.standoff_sticky_heading_xy: tuple[float, float] | None = None
        self.standoff_sticky_ticks_left = 0
        self.standoff_intent_target_holds = 0
        self.standoff_intent_target_releases = 0
        self.standoff_yaw_mode: str | None = None
        self.standoff_yaw_holds = 0
        self.standoff_yaw_flip_suppressions = 0
        self.standoff_yaw_exits = 0
        self.standoff_arc_bearing_suppressions = 0
        self.standoff_arc_target_dist_suppressions = 0
        x0, y0, x1, y1 = bounds
        self.free: dict[tuple[int, int], tuple[float, float]] = {}
        # online_frontier is the runtime-contract-clean coverage mode: the free
        # graph starts optimistic (every cell presumed traversable, no manifest
        # reads) and is pruned only by executed-contact evidence via the
        # existing waypoint-block machinery.
        self.optimistic_free_graph = str(goal_policy).lower() == "online_frontier"
        if self.optimistic_free_graph:
            # "mixed" commits to a far frontier every third replan: pure
            # nearest-cell hopping pays several yaw-alignment ticks per
            # 1-cell hop and stalls coverage (~15 cells / 2400 ticks).
            goal_policy = "mixed"
            self.goal_policy = "mixed"
        if str(goal_policy).lower() not in (
            "learned_sweep",
            "learned_local",
            "learned_wall_follow",
            "learned_policy",
        ):
            nx = int((x1 - x0) / step_m) + 1
            ny = int((y1 - y0) / step_m) + 1
            for i in range(nx):
                for j in range(ny):
                    x, y = x0 + i * step_m, y0 + j * step_m
                    if self.optimistic_free_graph or grid.is_free((x, y)):
                        self.free[(i, j)] = (x, y)
        self.visited: set[tuple[int, int]] = set()
        self.blocked: set[tuple[int, int]] = set()
        self.target_cell = None
        self.path = None
        self.wp_idx = 0
        self.replans = 0
        self.goal_cell: tuple[int, int] | None = None
        # Periodic look-around so the narrow forward camera catches landmarks.
        self.tick = 0
        self.scan_remaining = 0
        self.scan_dir = "yaw_left"
        self.scan_active = False
        self.last_bearing: float | None = None
        self.last_waypoint_cell: tuple[int, int] | None = None
        self.coverage_order: list[tuple[int, int]] = []
        self.coverage_cursor = 0
        self.coverage_root: tuple[int, int] | None = None
        self.blocked_generation = 0
        self.coverage_blocked_generation = -1

    def notify_claim(self, pos_xy: np.ndarray | tuple[float, float] | None = None) -> None:
        self.route_claim_count += 1
        if (
            pos_xy is None
            or not self.route_waypoints
            or self.route_claim_count < self.route_start_after_claims
            or self.route_idx >= len(self.route_waypoints)
        ):
            return
        px, py = float(pos_xy[0]), float(pos_xy[1])
        remaining = range(int(self.route_idx), len(self.route_waypoints))
        nearest = min(
            remaining,
            key=lambda i: (self.route_waypoints[i][0] - px) ** 2 + (self.route_waypoints[i][1] - py) ** 2,
        )
        self.route_idx = int(nearest)
        self._advance_route_index((px, py))

    def reset_route_state(self, *, clear_visited: bool = False) -> None:
        """Discard stale route-following state while preserving blocked evidence."""
        self.target_cell = None
        self.path = None
        self.wp_idx = 0
        self.goal_cell = None
        self.last_waypoint_cell = None
        self.last_bearing = None
        self.coverage_order = []
        self.coverage_cursor = 0
        self.coverage_root = None
        self.coverage_neighbor_dirs = None
        self.coverage_blocked_generation = -1
        self.scan_remaining = 0
        self.scan_active = False
        self.standoff_target_color = None
        self.standoff_goal_xy = None
        self.standoff_beacon_xy = None
        self.standoff_path = ()
        self.standoff_path_idx = 0
        self.standoff_heading_xy = None
        self._reset_standoff_intent()
        self.standoff_plan_age = 0
        self.standoff_selected_clearance_m = None
        self.standoff_body_forward_clearance_m = None
        self.standoff_body_current_clearance_m = None
        if clear_visited:
            self.visited.clear()

    def _reset_standoff_intent(self) -> None:
        self.standoff_sticky_target_xy = None
        self.standoff_sticky_heading_xy = None
        self.standoff_sticky_ticks_left = 0
        self.standoff_yaw_mode = None

    def trace(self) -> dict[str, Any]:
        return {
            "goal_policy": self.goal_policy,
            "forward_primitive": self.forward_primitive,
            "coverage_lookahead_cells": int(self.coverage_lookahead_cells),
            "dfs_neighbor_order": self.dfs_neighbor_order,
            "scan_interval": int(self.scan_interval),
            "scan_len": int(self.scan_len),
            "scan_primitive": self.scan_primitive,
            "scan_active": bool(self.scan_active),
            "scan_remaining": int(self.scan_remaining),
            "goal_cell": list(self.goal_cell) if self.goal_cell is not None else None,
            "waypoint_cell": list(self.last_waypoint_cell) if self.last_waypoint_cell is not None else None,
            "wp_idx": int(self.wp_idx),
            "replans": int(self.replans),
            "bearing": _round_float(self.last_bearing, 4),
            "coverage_cursor": int(self.coverage_cursor),
            "coverage_order_len": int(len(self.coverage_order)),
            "route_idx": int(self.route_idx),
            "route_len": int(len(self.route_waypoints)),
            "route_claim_count": int(self.route_claim_count),
            "route_active": bool(self._route_active()),
            "route_goal_xy": (
                [
                    _round_float(self.route_waypoints[self.route_idx][0]),
                    _round_float(self.route_waypoints[self.route_idx][1]),
                ]
                if self._route_active() and self.route_idx < len(self.route_waypoints)
                else None
            ),
            "standoff_route": bool(self.standoff_route),
            "standoff_target_color": self.standoff_target_color,
            "standoff_goal_xy": (
                [
                    _round_float(self.standoff_goal_xy[0]),
                    _round_float(self.standoff_goal_xy[1]),
                ]
                if self.standoff_goal_xy is not None
                else None
            ),
            "standoff_path_len": int(len(self.standoff_path)),
            "standoff_path_idx": int(self.standoff_path_idx),
            "standoff_plan_age": int(self.standoff_plan_age),
            "standoff_replans": int(self.standoff_replans),
            "standoff_plan_failures": int(self.standoff_plan_failures),
            "standoff_selected_clearance_m": _round_float(self.standoff_selected_clearance_m, 4),
            "standoff_path_spacing_m": _round_float(self.standoff_path_spacing_m, 4),
            "standoff_clearance_weight": _round_float(self.standoff_clearance_weight, 4),
            "standoff_clearance_target_m": _round_float(self.standoff_clearance_target_m, 4),
            "standoff_body_route_clearance_weight": _round_float(
                self.standoff_body_route_clearance_weight, 4
            ),
            "standoff_body_route_clearance_target_m": _round_float(
                self.standoff_body_route_clearance_target_m, 4
            ),
            "standoff_body_route_ignore_start_m": _round_float(
                self.standoff_body_route_ignore_start_m, 4
            ),
            "standoff_cardinal_route": bool(self.standoff_cardinal_route),
            "standoff_corner_guard": bool(self.standoff_corner_guard),
            "standoff_corner_commit_m": _round_float(self.standoff_corner_commit_m, 4),
            "standoff_corner_standoff_m": _round_float(self.standoff_corner_standoff_m, 4),
            "standoff_allow_arcs": bool(self.standoff_allow_arcs),
            "standoff_arc_min_bearing": _round_float(self.standoff_arc_min_bearing, 4),
            "standoff_arc_max_bearing": _round_float(self.standoff_arc_max_bearing, 4),
            "standoff_arc_min_target_dist_m": _round_float(
                self.standoff_arc_min_target_dist_m, 4
            ),
            "standoff_arc_bearing_suppressions": int(self.standoff_arc_bearing_suppressions),
            "standoff_arc_target_dist_suppressions": int(
                self.standoff_arc_target_dist_suppressions
            ),
            "standoff_heading_mode": self.standoff_heading_mode,
            "standoff_heading_xy": (
                [
                    _round_float(self.standoff_heading_xy[0]),
                    _round_float(self.standoff_heading_xy[1]),
                ]
                if self.standoff_heading_xy is not None
                else None
            ),
            "standoff_heading_lookahead_m": _round_float(self.standoff_heading_lookahead_m, 4),
            "standoff_prefix_snap_start": bool(self.standoff_prefix_snap_start),
            "standoff_snap_start_min_dist_m": _round_float(self.standoff_snap_start_min_dist_m, 4),
            "standoff_tangent_heading_ticks": int(self.standoff_tangent_heading_ticks),
            "standoff_snap_start_prefixes": int(self.standoff_snap_start_prefixes),
            "standoff_corner_guard_caps": int(self.standoff_corner_guard_caps),
            "standoff_corner_standoff_caps": int(self.standoff_corner_standoff_caps),
            "standoff_selected_body_route_clearance_m": _round_float(
                self.standoff_selected_body_route_clearance_m, 4
            ),
            "standoff_blocked_cells": int(len(self.standoff_blocked_cells)),
            "standoff_blocked_waypoints": int(self.standoff_blocked_waypoints),
            "standoff_body_check": bool(self.standoff_body_check),
            "standoff_body_vetoes": int(self.standoff_body_vetoes),
            "standoff_body_current_clearance_m": _round_float(self.standoff_body_current_clearance_m, 4),
            "standoff_body_forward_clearance_m": _round_float(self.standoff_body_forward_clearance_m, 4),
            "standoff_body_left_clearance_m": _round_float(self.standoff_body_left_clearance_m, 4),
            "standoff_body_right_clearance_m": _round_float(self.standoff_body_right_clearance_m, 4),
            "standoff_body_backward_clearance_m": _round_float(self.standoff_body_backward_clearance_m, 4),
            "standoff_body_min_clearance_m": _round_float(self.standoff_body_min_clearance_m, 4),
            "standoff_body_recovery_clearance_m": _round_float(
                self.standoff_body_recovery_clearance_m, 4
            ),
            "standoff_intent_smoothing": bool(self.standoff_intent_smoothing),
            "standoff_sticky_target_ticks": int(self.standoff_sticky_target_ticks),
            "standoff_sticky_target_release_m": _round_float(
                self.standoff_sticky_target_release_m, 4
            ),
            "standoff_sticky_ticks_left": int(self.standoff_sticky_ticks_left),
            "standoff_intent_target_holds": int(self.standoff_intent_target_holds),
            "standoff_intent_target_releases": int(self.standoff_intent_target_releases),
            "standoff_yaw_mode": self.standoff_yaw_mode,
            "standoff_yaw_enter_threshold": _round_float(self.standoff_yaw_enter_threshold, 4),
            "standoff_yaw_exit_threshold": _round_float(self.standoff_yaw_exit_threshold, 4),
            "standoff_yaw_flip_threshold": _round_float(self.standoff_yaw_flip_threshold, 4),
            "standoff_yaw_holds": int(self.standoff_yaw_holds),
            "standoff_yaw_flip_suppressions": int(self.standoff_yaw_flip_suppressions),
            "standoff_yaw_exits": int(self.standoff_yaw_exits),
            "body_probe_margin_m": _round_float(self.body_probe_margin_m, 4),
        }

    def _route_active(self) -> bool:
        return bool(
            self.route_waypoints
            and self.route_claim_count >= self.route_start_after_claims
            and self.route_idx < len(self.route_waypoints)
        )

    def _advance_route_index(self, pos_xy: np.ndarray | tuple[float, float]) -> None:
        px, py = float(pos_xy[0]), float(pos_xy[1])
        while self.route_idx < len(self.route_waypoints):
            wx, wy = self.route_waypoints[self.route_idx]
            if (wx - px) ** 2 + (wy - py) ** 2 <= self.route_advance_m ** 2:
                self.route_idx += 1
            else:
                break

    def _standoff_active(self, target_color: str | None) -> bool:
        return bool(
            self.standoff_route
            and target_color is not None
            and str(target_color) in self.standoff_targets
            and self.standoff_grid is not None
        )

    def _standoffs_with_los(self, beacon_xy: tuple[float, float]) -> tuple[tuple[float, float], ...]:
        grid = self.standoff_grid
        if grid is None:
            return ()
        candidates = safe_standoff_xys(
            grid,
            beacon_xy,
            standoff_m=self.standoff_m,
            n_candidates=self.standoff_candidates,
        )
        los = getattr(self.standoff_scene_graph, "has_line_of_sight", None)
        if los is None:
            return candidates
        bx, by = float(beacon_xy[0]), float(beacon_xy[1])
        out: list[tuple[float, float]] = []
        for sx, sy in candidates:
            if los(
                (float(sx), float(sy)),
                (bx, by),
                margin_m=0.02,
                exclude_landmark_xy=(bx, by),
            ):
                out.append((float(sx), float(sy)))
        return tuple(out) or candidates

    def _standoff_path_cost(
        self,
        path,
        start_xy: tuple[float, float],
        standoff_xy: tuple[float, float],
        beacon_xy: tuple[float, float],
    ) -> float:
        waypoints = tuple(path.waypoints_xy)
        base = float(path.cost_cells)
        if waypoints:
            prev = start_xy if len(waypoints) == 1 else waypoints[-2]
            last = waypoints[-1]
        else:
            prev = start_xy
            last = standoff_xy
        approach_yaw = math.atan2(float(last[1]) - float(prev[1]), float(last[0]) - float(prev[0]))
        beacon_yaw = math.atan2(float(beacon_xy[1]) - float(last[1]), float(beacon_xy[0]) - float(last[0]))
        heading_penalty = 60.0 * abs(wrap_angle_pi(beacon_yaw - approach_yaw)) / math.pi
        clearance = None
        grid = self.standoff_grid
        if grid is not None:
            clearance = grid.configuration_clearance_m(standoff_xy)
        clearance_penalty = 0.0 if clearance is None else max(0.0, 0.08 - float(clearance)) * 80.0
        body_clearance = self._standoff_path_body_clearance(
            path,
            start_xy,
            standoff_xy,
            beacon_xy,
            ignore_start_m=self.standoff_body_route_ignore_start_m,
        )
        body_penalty = 0.0
        if (
            body_clearance is not None
            and self.standoff_body_route_clearance_weight > 0.0
        ):
            body_penalty = (
                max(0.0, self.standoff_body_route_clearance_target_m - float(body_clearance))
                * self.standoff_body_route_clearance_weight
            )
        return base + heading_penalty + clearance_penalty + body_penalty

    def _standoff_path_body_clearance(
        self,
        path,
        start_xy: tuple[float, float],
        standoff_xy: tuple[float, float],
        beacon_xy: tuple[float, float],
        *,
        ignore_start_m: float = 0.0,
    ) -> float | None:
        grid = self.standoff_grid
        if grid is None:
            return None
        raw_points = [(float(start_xy[0]), float(start_xy[1]))]
        for x, y in tuple(path.waypoints_xy):
            point = (float(x), float(y))
            if math.dist(raw_points[-1], point) > 1e-4:
                raw_points.append(point)
        standoff_point = (float(standoff_xy[0]), float(standoff_xy[1]))
        if math.dist(raw_points[-1], standoff_point) > 1e-4:
            raw_points.append(standoff_point)
        if len(raw_points) == 1:
            raw_points.append((float(beacon_xy[0]), float(beacon_xy[1])))
        cumulative_m = [0.0]
        for idx in range(1, len(raw_points)):
            cumulative_m.append(
                cumulative_m[-1]
                + math.hypot(
                    float(raw_points[idx][0]) - float(raw_points[idx - 1][0]),
                    float(raw_points[idx][1]) - float(raw_points[idx - 1][1]),
                )
            )
        ignore_start_m = max(0.0, float(ignore_start_m))
        min_clearance = float("inf")
        sample_step_m = 0.08
        yaw_step_rad = 0.25

        def update_clearance(point: tuple[float, float], yaw: float) -> None:
            nonlocal min_clearance
            min_clearance = min(
                min_clearance,
                _body_probe_clearance(
                    grid,
                    point,
                    yaw,
                    body_forward_m=self.body_forward_m,
                    body_half_width_m=self.body_half_width_m,
                    body_probe_margin_m=self.body_probe_margin_m,
                ),
            )

        for idx, point in enumerate(raw_points[:-1]):
            next_point = raw_points[idx + 1]
            yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            segment_len = math.hypot(
                float(next_point[0]) - float(point[0]),
                float(next_point[1]) - float(point[1]),
            )
            steps = max(1, int(math.ceil(segment_len / sample_step_m)))
            for step_idx in range(steps + 1):
                t = float(step_idx) / float(steps)
                sample_progress_m = float(cumulative_m[idx]) + segment_len * t
                if sample_progress_m < ignore_start_m:
                    continue
                sample = (
                    float(point[0]) + (float(next_point[0]) - float(point[0])) * t,
                    float(point[1]) + (float(next_point[1]) - float(point[1])) * t,
                )
                update_clearance(sample, yaw)
        for idx in range(1, len(raw_points)):
            if float(cumulative_m[idx]) < ignore_start_m:
                continue
            point = raw_points[idx]
            prev_point = raw_points[idx - 1]
            prev_yaw = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
            if idx + 1 < len(raw_points):
                next_point = raw_points[idx + 1]
                next_yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            else:
                next_yaw = math.atan2(
                    float(beacon_xy[1]) - float(point[1]),
                    float(beacon_xy[0]) - float(point[0]),
                )
            dyaw = wrap_angle_pi(next_yaw - prev_yaw)
            steps = max(1, int(math.ceil(abs(float(dyaw)) / yaw_step_rad)))
            for step_idx in range(steps + 1):
                yaw = wrap_angle_pi(prev_yaw + dyaw * (float(step_idx) / float(steps)))
                update_clearance(point, yaw)
        return None if not math.isfinite(min_clearance) else float(min_clearance)

    def _sparsify_standoff_path(
        self,
        waypoints: tuple[tuple[float, float], ...],
        spacing_m: float = 0.30,
    ) -> tuple[tuple[float, float], ...]:
        if len(waypoints) <= 2:
            return waypoints
        spacing = max(0.05, float(spacing_m))
        out: list[tuple[float, float]] = [waypoints[0]]
        last = waypoints[0]
        for wp in waypoints[1:-1]:
            if math.hypot(float(wp[0]) - float(last[0]), float(wp[1]) - float(last[1])) >= spacing:
                out.append(wp)
                last = wp
        if out[-1] != waypoints[-1]:
            out.append(waypoints[-1])
        return tuple(out)

    def _plan_standoff_route(self, pos_xy: tuple[float, float], target_color: str) -> bool:
        grid = self.standoff_grid
        if grid is None or target_color not in self.standoff_targets:
            return False
        beacon_xy = self.standoff_targets[target_color]
        start = grid.nearest_free(
            (float(pos_xy[0]), float(pos_xy[1])),
            max_radius_m=max(0.25, 4.0 * float(grid.inflation_m)),
        )
        if start is None:
            self.standoff_plan_failures += 1
            return False
        best: tuple[float, tuple[float, float], Any] | None = None
        for standoff_xy in self._standoffs_with_los(beacon_xy):
            path = grid.astar(
                start,
                standoff_xy,
                clearance_weight=self.standoff_clearance_weight,
                clearance_target_m=self.standoff_clearance_target_m,
                allow_diagonal=not self.standoff_cardinal_route,
                blocked_cells=self.standoff_blocked_cells,
            )
            if path is None:
                continue
            cost = self._standoff_path_cost(path, start, standoff_xy, beacon_xy)
            if best is None or cost < best[0]:
                best = (cost, standoff_xy, path)
        if best is None:
            self.standoff_target_color = str(target_color)
            self.standoff_goal_xy = None
            self.standoff_beacon_xy = beacon_xy
            self.standoff_path = ()
            self.standoff_path_idx = 0
            self.standoff_plan_age = 0
            self.standoff_selected_body_route_clearance_m = None
            self.standoff_plan_failures += 1
            return False
        _, standoff_xy, path = best
        self.standoff_target_color = str(target_color)
        self.standoff_goal_xy = (float(standoff_xy[0]), float(standoff_xy[1]))
        self.standoff_beacon_xy = beacon_xy
        route_waypoints = tuple(path.waypoints_xy)
        if (
            self.standoff_prefix_snap_start
            and math.hypot(float(start[0]) - float(pos_xy[0]), float(start[1]) - float(pos_xy[1]))
            >= self.standoff_snap_start_min_dist_m
            and (not route_waypoints or math.hypot(
                float(route_waypoints[0][0]) - float(start[0]),
                float(route_waypoints[0][1]) - float(start[1]),
            ) > 1e-4)
        ):
            route_waypoints = ((float(start[0]), float(start[1])),) + route_waypoints
            self.standoff_snap_start_prefixes += 1
        self.standoff_path = self._sparsify_standoff_path(
            route_waypoints,
            spacing_m=self.standoff_path_spacing_m,
        )
        self.standoff_path_idx = 0
        self.standoff_heading_xy = None
        self._reset_standoff_intent()
        self.standoff_plan_age = 0
        self.standoff_replans += 1
        self.standoff_selected_clearance_m = (
            None if grid is None else float(grid.configuration_clearance_m(self.standoff_goal_xy))
        )
        self.standoff_selected_body_route_clearance_m = self._standoff_path_body_clearance(
            path,
            start,
            self.standoff_goal_xy,
            beacon_xy,
            ignore_start_m=self.standoff_body_route_ignore_start_m,
        )
        return True

    def _smooth_standoff_target(
        self,
        pos_xy: tuple[float, float],
        target_xy: tuple[float, float],
        heading_xy: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if (
            not self.standoff_intent_smoothing
            or self.standoff_sticky_target_ticks <= 0
            or self.standoff_grid is None
        ):
            return target_xy, heading_xy
        px, py = float(pos_xy[0]), float(pos_xy[1])
        sticky = self.standoff_sticky_target_xy
        if sticky is not None and self.standoff_sticky_ticks_left > 0:
            sx, sy = float(sticky[0]), float(sticky[1])
            sticky_dist = math.hypot(sx - px, sy - py)
            if (
                sticky_dist > self.standoff_sticky_target_release_m
                and self.standoff_grid.has_free_line((px, py), sticky)
            ):
                self.standoff_sticky_ticks_left -= 1
                self.standoff_intent_target_holds += 1
                return (
                    (sx, sy),
                    self.standoff_sticky_heading_xy or (sx, sy),
                )
            self.standoff_intent_target_releases += 1
        self.standoff_sticky_target_xy = (float(target_xy[0]), float(target_xy[1]))
        self.standoff_sticky_heading_xy = (float(heading_xy[0]), float(heading_xy[1]))
        self.standoff_sticky_ticks_left = int(self.standoff_sticky_target_ticks)
        return target_xy, heading_xy

    def _standoff_arc_allowed(
        self,
        abs_bearing: float,
        target_dist_m: float | None,
    ) -> bool:
        min_bearing = (
            self.forward_bearing_threshold
            if self.standoff_arc_min_bearing is None
            else float(self.standoff_arc_min_bearing)
        )
        max_bearing = (
            self.yaw_bearing_threshold
            if self.standoff_arc_max_bearing is None
            else float(self.standoff_arc_max_bearing)
        )
        if abs_bearing <= min_bearing or abs_bearing > max_bearing:
            self.standoff_arc_bearing_suppressions += 1
            return False
        if (
            self.standoff_arc_min_target_dist_m > 0.0
            and target_dist_m is not None
            and float(target_dist_m) < self.standoff_arc_min_target_dist_m
        ):
            self.standoff_arc_target_dist_suppressions += 1
            return False
        return True

    def _standoff_turn_request(
        self,
        bearing: float,
        target_dist_m: float | None = None,
    ) -> str | None:
        abs_bearing = abs(float(bearing))
        turn = "yaw_left" if bearing > 0.0 else "yaw_right"
        if not self.standoff_intent_smoothing:
            if abs_bearing > self.yaw_bearing_threshold:
                return turn
            if (
                bool(self.standoff_allow_arcs)
                and abs_bearing > self.forward_bearing_threshold
                and self._standoff_arc_allowed(abs_bearing, target_dist_m)
            ):
                return "arc_left" if bearing > 0.0 else "arc_right"
            if abs_bearing > self.forward_bearing_threshold:
                return turn
            return None

        enter = (
            max(self.forward_bearing_threshold, 0.16)
            if self.standoff_yaw_enter_threshold is None
            else float(self.standoff_yaw_enter_threshold)
        )
        exit_threshold = (
            min(self.forward_bearing_threshold, max(0.0, enter * 0.55))
            if self.standoff_yaw_exit_threshold is None
            else float(self.standoff_yaw_exit_threshold)
        )
        flip = (
            max(enter, self.yaw_bearing_threshold)
            if self.standoff_yaw_flip_threshold is None
            else float(self.standoff_yaw_flip_threshold)
        )
        if self.standoff_yaw_mode in ("yaw_left", "yaw_right"):
            if abs_bearing <= exit_threshold:
                self.standoff_yaw_mode = None
                self.standoff_yaw_exits += 1
                return None
            if turn != self.standoff_yaw_mode and abs_bearing < flip:
                self.standoff_yaw_flip_suppressions += 1
                self.standoff_yaw_holds += 1
                return self.standoff_yaw_mode
            if turn == self.standoff_yaw_mode:
                self.standoff_yaw_holds += 1
            self.standoff_yaw_mode = turn
            return turn
        if abs_bearing > enter:
            self.standoff_yaw_mode = turn
            return turn
        if (
            bool(self.standoff_allow_arcs)
            and abs_bearing > self.forward_bearing_threshold
            and self._standoff_arc_allowed(abs_bearing, target_dist_m)
        ):
            return "arc_left" if bearing > 0.0 else "arc_right"
        if abs_bearing > self.forward_bearing_threshold:
            return turn
        return None

    def _standoff_tangent_heading_target(
        self,
        pos_xy: tuple[float, float],
        path: tuple[tuple[float, float], ...],
        closest_idx: int,
    ) -> tuple[float, float] | None:
        if len(path) < 2:
            return None
        px, py = float(pos_xy[0]), float(pos_xy[1])
        seg_idx = max(0, min(int(closest_idx), len(path) - 2))
        # If the base is already at the corner, allow the heading target to
        # advance to the next segment. Otherwise keep the body squared with
        # the current corridor instead of aiming through the corner vertex.
        while seg_idx < len(path) - 2:
            bx, by = float(path[seg_idx + 1][0]), float(path[seg_idx + 1][1])
            if math.hypot(bx - px, by - py) > max(0.02, self.standoff_corner_commit_m):
                break
            seg_idx += 1
        ax, ay = float(path[seg_idx][0]), float(path[seg_idx][1])
        bx, by = float(path[seg_idx + 1][0]), float(path[seg_idx + 1][1])
        vx, vy = bx - ax, by - ay
        seg_len = math.hypot(vx, vy)
        if seg_len <= 1e-6:
            return None
        ux, uy = vx / seg_len, vy / seg_len
        progress = (px - ax) * ux + (py - ay) * uy
        progress = max(0.0, min(seg_len, progress))
        lookahead = min(float(self.standoff_heading_lookahead_m), max(0.05, seg_len - progress))
        hx = ax + ux * min(seg_len, progress + lookahead)
        hy = ay + uy * min(seg_len, progress + lookahead)
        self.standoff_tangent_heading_ticks += 1
        return (float(hx), float(hy))

    def _standoff_progress_index(
        self,
        pos_xy: tuple[float, float],
        path: tuple[tuple[float, float], ...],
        idx: int,
    ) -> int:
        if len(path) < 2:
            return max(0, min(int(idx), len(path) - 1))
        px, py = float(pos_xy[0]), float(pos_xy[1])
        start_j = max(0, min(int(idx) - 1, len(path) - 2))
        best: tuple[float, int, float] | None = None
        for j in range(start_j, len(path) - 1):
            ax, ay = float(path[j][0]), float(path[j][1])
            bx, by = float(path[j + 1][0]), float(path[j + 1][1])
            vx, vy = bx - ax, by - ay
            seg_len2 = vx * vx + vy * vy
            if seg_len2 <= 1e-9:
                continue
            t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
            clamped = max(0.0, min(1.0, t))
            cx = ax + vx * clamped
            cy = ay + vy * clamped
            dist = math.hypot(px - cx, py - cy)
            # Small monotonicity penalty prevents noisy lateral drift from
            # jumping several route segments ahead.
            score = dist + 0.02 * float(max(0, j - start_j))
            if best is None or score < best[0]:
                best = (score, j, clamped)
        if best is None:
            return max(0, min(int(idx), len(path) - 1))
        _, seg_idx, _ = best
        bx, by = float(path[seg_idx + 1][0]), float(path[seg_idx + 1][1])
        if math.hypot(bx - px, by - py) <= max(0.02, self.standoff_corner_commit_m):
            return min(seg_idx + 1, len(path) - 1)
        return seg_idx

    def _standoff_lookahead_target(self, pos_xy: tuple[float, float]) -> tuple[float, float] | None:
        grid = self.standoff_grid
        path = self.standoff_path
        if grid is None or not path:
            self.standoff_heading_xy = None
            return None
        px, py = float(pos_xy[0]), float(pos_xy[1])
        idx = max(0, min(int(self.standoff_path_idx), len(path) - 1))
        best_idx = self._standoff_progress_index((px, py), path, idx)
        self.standoff_path_idx = best_idx
        chosen = path[best_idx]
        for j, wp in enumerate(path[best_idx:], start=best_idx):
            if math.hypot(float(wp[0]) - px, float(wp[1]) - py) > self.standoff_lookahead_m:
                break
            if not grid.has_free_line((px, py), wp):
                break
            chosen = wp
            self.standoff_path_idx = max(self.standoff_path_idx, j)
        heading_xy = chosen
        if bool(self.standoff_corner_guard) and bool(self.standoff_cardinal_route):
            chosen_idx = int(self.standoff_path_idx)
            current_dir: tuple[int, int] | None = None
            corner_idx: int | None = None
            for j in range(best_idx, len(path) - 1):
                dx = float(path[j + 1][0]) - float(path[j][0])
                dy = float(path[j + 1][1]) - float(path[j][1])
                if abs(dx) >= abs(dy):
                    step_dir = (1 if dx > 1e-6 else -1 if dx < -1e-6 else 0, 0)
                else:
                    step_dir = (0, 1 if dy > 1e-6 else -1 if dy < -1e-6 else 0)
                if step_dir == (0, 0):
                    continue
                if current_dir is None:
                    current_dir = step_dir
                    continue
                if step_dir != current_dir:
                    corner_idx = j
                    break
            if corner_idx is not None and chosen_idx > corner_idx:
                corner_wp = path[corner_idx]
                corner_dist = math.hypot(float(corner_wp[0]) - px, float(corner_wp[1]) - py)
                if corner_dist > self.standoff_corner_commit_m:
                    chosen = corner_wp
                    if (
                        current_dir is not None
                        and self.standoff_corner_standoff_m > 0.0
                    ):
                        ux, uy = float(current_dir[0]), float(current_dir[1])
                        segment_start = path[best_idx]
                        corner_progress = (
                            (float(corner_wp[0]) - float(segment_start[0])) * ux
                            + (float(corner_wp[1]) - float(segment_start[1])) * uy
                        )
                        pos_progress = (
                            (px - float(segment_start[0])) * ux
                            + (py - float(segment_start[1])) * uy
                        )
                        target_progress = corner_progress - float(self.standoff_corner_standoff_m)
                        if target_progress > max(0.0, pos_progress + 0.04):
                            chosen = (
                                float(segment_start[0]) + ux * target_progress,
                                float(segment_start[1]) + uy * target_progress,
                            )
                            self.standoff_corner_standoff_caps += 1
                    self.standoff_path_idx = min(int(self.standoff_path_idx), int(corner_idx))
                    self.standoff_corner_guard_caps += 1
        if self.standoff_heading_mode in {"tangent", "path_tangent", "route_tangent"}:
            tangent_heading_xy = self._standoff_tangent_heading_target((px, py), path, best_idx)
            if tangent_heading_xy is not None:
                heading_xy = tangent_heading_xy
        chosen, heading_xy = self._smooth_standoff_target((px, py), chosen, heading_xy)
        self.last_waypoint_cell = grid.to_grid(chosen)
        self.standoff_heading_xy = heading_xy
        return chosen

    def _standoff_primitive(self, pos, yaw: float, target_color: str | None) -> str | None:
        if not self._standoff_active(target_color):
            return None
        color = str(target_color)
        pos_xy = (float(pos[0]), float(pos[1]))
        need_plan = (
            self.standoff_target_color != color
            or not self.standoff_path
            or self.standoff_plan_age >= self.standoff_replan_interval
        )
        if need_plan and not self._plan_standoff_route(pos_xy, color):
            return None
        self.standoff_plan_age += 1
        target_xy = self._standoff_lookahead_target(pos_xy)
        if target_xy is None:
            target_xy = self.standoff_goal_xy
            self.standoff_heading_xy = target_xy
        if target_xy is None:
            return None
        heading_xy = self.standoff_heading_xy or target_xy
        near_standoff = False
        if self.standoff_goal_xy is not None:
            near_standoff = (
                math.hypot(
                    float(self.standoff_goal_xy[0]) - pos_xy[0],
                    float(self.standoff_goal_xy[1]) - pos_xy[1],
                )
                <= self.standoff_arrival_m
            )
        if near_standoff and self.standoff_beacon_xy is not None:
            target_xy = self.standoff_beacon_xy
            heading_xy = self.standoff_beacon_xy
            self.standoff_heading_xy = heading_xy
        bearing = wrap_angle_pi(
            math.atan2(float(heading_xy[1]) - pos_xy[1], float(heading_xy[0]) - pos_xy[0]) - yaw
        )
        beacon_dist_m = (
            None
            if self.standoff_beacon_xy is None
            else math.hypot(
                float(self.standoff_beacon_xy[0]) - pos_xy[0],
                float(self.standoff_beacon_xy[1]) - pos_xy[1],
            )
        )
        self.last_bearing = float(bearing)
        if self.standoff_body_check and self.standoff_grid is not None:
            current_clearance = _body_probe_clearance(
                self.standoff_grid,
                pos_xy,
                yaw,
                body_forward_m=self.body_forward_m,
                body_half_width_m=self.body_half_width_m,
                body_probe_margin_m=self.body_probe_margin_m,
            )
            forward_clearance = _swept_forward_body_clearance(
                self.standoff_grid,
                pos_xy,
                yaw,
                distance_m=min(self.standoff_body_lookahead_m, self.lookahead_m),
                body_forward_m=self.body_forward_m,
                body_half_width_m=self.body_half_width_m,
                body_probe_margin_m=self.body_probe_margin_m,
            )
            left_clearance = _body_probe_clearance(
                self.standoff_grid,
                pos_xy,
                wrap_angle_pi(yaw + 0.30),
                body_forward_m=self.body_forward_m,
                body_half_width_m=self.body_half_width_m,
                body_probe_margin_m=self.body_probe_margin_m,
            )
            right_clearance = _body_probe_clearance(
                self.standoff_grid,
                pos_xy,
                wrap_angle_pi(yaw - 0.30),
                body_forward_m=self.body_forward_m,
                body_half_width_m=self.body_half_width_m,
                body_probe_margin_m=self.body_probe_margin_m,
            )
            backward_clearance = _swept_forward_body_clearance(
                self.standoff_grid,
                pos_xy,
                yaw,
                distance_m=-min(self.standoff_body_lookahead_m, self.lookahead_m),
                body_forward_m=self.body_forward_m,
                body_half_width_m=self.body_half_width_m,
                body_probe_margin_m=self.body_probe_margin_m,
            )
            self.standoff_body_current_clearance_m = float(current_clearance)
            self.standoff_body_forward_clearance_m = float(forward_clearance)
            self.standoff_body_left_clearance_m = float(left_clearance)
            self.standoff_body_right_clearance_m = float(right_clearance)
            self.standoff_body_backward_clearance_m = float(backward_clearance)

            def body_safe_choice(
                requested: str,
                requested_clearance: float,
                fallback_order: tuple[str, ...],
            ) -> str:
                clearances = {
                    "yaw_left": left_clearance,
                    "yaw_right": right_clearance,
                    "backward": backward_clearance,
                    self.forward_primitive: forward_clearance,
                }
                if (
                    self.standoff_body_recovery_clearance_m is not None
                    and current_clearance < self.standoff_body_recovery_clearance_m
                ):
                    translating = [
                        name
                        for name in (self.forward_primitive, "backward")
                        if name in clearances and clearances[name] >= current_clearance - 0.005
                    ]
                    if translating:
                        best_translation = max(
                            translating,
                            key=lambda name: (
                                float(clearances[name]),
                                1.0 if name == requested else 0.0,
                            ),
                        )
                        if best_translation != requested:
                            self.standoff_body_vetoes += 1
                        return best_translation
                if (
                    requested_clearance >= self.standoff_body_min_clearance_m
                    and requested_clearance >= current_clearance - 0.005
                ):
                    return requested
                candidates = [(requested, requested_clearance)]
                candidates.extend((name, clearances[name]) for name in fallback_order)
                candidates = _unique_primitives([name for name, _ in candidates])
                best = max(
                    candidates,
                    key=lambda name: (
                        float(clearances.get(name, requested_clearance)),
                        1.0 if name == requested else 0.0,
                    ),
                )
                if best != requested:
                    self.standoff_body_vetoes += 1
                return best

            turn_request = self._standoff_turn_request(float(bearing), beacon_dist_m)
            if turn_request in ("arc_left", "arc_right"):
                return turn_request
            if turn_request in ("yaw_left", "yaw_right"):
                requested_turn = turn_request
                opposite_turn = "yaw_right" if requested_turn == "yaw_left" else "yaw_left"
                requested_clearance = left_clearance if requested_turn == "yaw_left" else right_clearance
                return body_safe_choice(
                    requested_turn,
                    requested_clearance,
                    (opposite_turn, "backward"),
                )
            if not _look_ahead_free(grid_global[0], pos[:2], yaw, self.lookahead_m):
                requested_turn = "yaw_left" if bearing >= 0.0 else "yaw_right"
                opposite_turn = "yaw_right" if requested_turn == "yaw_left" else "yaw_left"
                requested_clearance = left_clearance if requested_turn == "yaw_left" else right_clearance
                return body_safe_choice(
                    requested_turn,
                    requested_clearance,
                    (opposite_turn, "backward"),
                )
            if forward_clearance < self.standoff_body_min_clearance_m:
                return body_safe_choice(
                    self.forward_primitive,
                    forward_clearance,
                    ("yaw_left", "yaw_right", "backward"),
                )
        turn_request = self._standoff_turn_request(float(bearing), beacon_dist_m)
        if turn_request is not None:
            return turn_request
        if not _look_ahead_free(grid_global[0], pos[:2], yaw, self.lookahead_m):
            return "yaw_left" if bearing >= 0.0 else "yaw_right"
        return self.forward_primitive

    def _cell(self, xy):
        # nearest free cell to xy
        best, bd = None, 1e9
        for c, (cx, cy) in self.free.items():
            d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
            if d < bd:
                bd, best = d, c
        return best

    def _bfs_path(self, start, goal):
        """Cell path start->goal over the free-cell 4-graph, or None."""
        from collections import deque
        parent = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                path = [c]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in self.free and n not in self.blocked and n not in parent:
                    parent[n] = c
                    q.append(n)
        return None

    def _order_dirs_for_yaw(self, yaw: float) -> tuple[tuple[int, int], ...]:
        dirs = ((1, 0), (0, 1), (0, -1), (-1, 0))

        def score(d: tuple[int, int]) -> float:
            return abs(wrap_angle_pi(math.atan2(float(d[1]), float(d[0])) - float(yaw)))

        return tuple(sorted(dirs, key=score))

    def _ordered_coverage_dirs(self, yaw: float | None = None) -> tuple[tuple[int, int], ...]:
        order = str(self.dfs_neighbor_order).lower()
        if order == "heading":
            if self.coverage_neighbor_dirs is None:
                self.coverage_neighbor_dirs = self._order_dirs_for_yaw(0.0 if yaw is None else float(yaw))
            return self.coverage_neighbor_dirs
        presets = {
            "nesw": ((0, 1), (1, 0), (0, -1), (-1, 0)),
            "eswn": ((1, 0), (0, -1), (-1, 0), (0, 1)),
            "senw": ((0, -1), (1, 0), (0, 1), (-1, 0)),
            "wsen": ((-1, 0), (0, -1), (1, 0), (0, 1)),
        }
        return presets.get(order, presets["nesw"])

    def _coverage_neighbors(self, c: tuple[int, int], yaw: float | None = None):
        for d in self._ordered_coverage_dirs(yaw):
            n = (c[0] + d[0], c[1] + d[1])
            if n in self.free and n not in self.blocked:
                yield n

    def _ensure_coverage_order(self, start: tuple[int, int], yaw: float | None = None) -> None:
        if (
            self.coverage_order
            and self.coverage_root in self.free
            and int(self.coverage_blocked_generation) == int(self.blocked_generation)
        ):
            return
        self.coverage_order = []
        self.coverage_cursor = 0
        self.coverage_root = start
        self.coverage_blocked_generation = int(self.blocked_generation)
        seen: set[tuple[int, int]] = set()

        def visit(c: tuple[int, int]) -> None:
            seen.add(c)
            self.coverage_order.append(c)
            for n in self._coverage_neighbors(c, yaw):
                if n not in seen:
                    visit(n)

        if start in self.free and start not in self.blocked:
            visit(start)

    def _unvisited_candidates(self, start) -> list[tuple[tuple[int, int], int]]:
        from collections import deque
        seen = {start}
        q = deque([(start, 0)])
        candidates: list[tuple[tuple[int, int], int]] = []
        while q:
            c, dist = q.popleft()
            if c not in self.visited and c not in self.blocked and c != start:
                candidates.append((c, dist))
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in self.free and n not in self.blocked and n not in seen:
                    seen.add(n)
                    q.append((n, dist + 1))
        return candidates

    def _nearest_unvisited(self, start):
        candidates = self._unvisited_candidates(start)
        return candidates[0][0] if candidates else None

    def _select_unvisited(self, start, yaw: float | None = None):
        candidates = self._unvisited_candidates(start)
        if not candidates:
            return None
        policy = str(self.goal_policy).lower()
        if policy == "farthest":
            return max(candidates, key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]
        if policy == "mixed":
            # Periodically choose a far frontier so local wall-adjacent cells do
            # not dominate physical exploration, but keep short plans in between.
            if self.replans > 0 and self.replans % 3 == 0:
                return max(candidates, key=lambda item: (item[1], -item[0][0], -item[0][1]))[0]
            return candidates[0][0]
        if policy == "dfs":
            self._ensure_coverage_order(start, yaw)
            if self.coverage_order:
                best_cell = None
                best_cursor = int(self.coverage_cursor)
                valid_seen = 0
                scan_limit = max(1, int(self.coverage_lookahead_cells))
                for _ in range(len(self.coverage_order)):
                    idx = int(self.coverage_cursor) % len(self.coverage_order)
                    cell = self.coverage_order[idx]
                    self.coverage_cursor += 1
                    if cell == start or cell in self.visited or cell in self.blocked:
                        continue
                    if self._bfs_path(start, cell):
                        valid_seen += 1
                        best_cell = cell
                        best_cursor = int(self.coverage_cursor)
                        if valid_seen >= scan_limit:
                            break
                if best_cell is not None:
                    self.coverage_cursor = best_cursor
                    return best_cell
            return candidates[0][0]
        return candidates[0][0]

    def _scan_step_primitive(self) -> str:
        if self.scan_primitive == "arc":
            return "arc_left" if self.scan_dir == "yaw_left" else "arc_right"
        return self.scan_dir

    def mark_current_waypoint_blocked(self) -> bool:
        if self.standoff_path and self.standoff_grid is not None:
            idx = max(0, min(int(self.standoff_path_idx), len(self.standoff_path) - 1))
            blocked_cell = self.standoff_grid.to_grid(self.standoff_path[idx])
            self.standoff_blocked_cells.add(blocked_cell)
            self.standoff_blocked_waypoints += 1
            self.standoff_path = ()
            self.standoff_path_idx = 0
            self.standoff_plan_age = self.standoff_replan_interval
            self.last_waypoint_cell = None
            return True
        if not self.path or self.wp_idx >= len(self.path):
            return False
        cell = self.path[self.wp_idx]
        self.blocked.add(cell)
        self.blocked_generation += 1
        self.visited.add(cell)
        self.path = None
        self.wp_idx = 0
        return True

    def primitive(self, pos, yaw, target_color: str | None = None):
        self.tick += 1
        self.last_bearing = None
        self.scan_active = False
        # Mark cells within a radius of the robot visited.
        r2 = (self.step * 1.4) ** 2
        for c, (cx, cy) in self.free.items():
            if (cx - pos[0]) ** 2 + (cy - pos[1]) ** 2 < r2:
                self.visited.add(c)
        cur = self._cell(pos[:2])
        standoff_primitive = self._standoff_primitive(pos, yaw, target_color)
        if standoff_primitive is not None:
            self.path = None
            self.goal_cell = None
            self.last_waypoint_cell = None
            return standoff_primitive
        # Periodic look-around so the narrow forward camera catches landmarks.
        # This happens after visited-cell accounting so scan ticks do not freeze
        # the coverage state. Clean demos can make it sparse or disable it.
        if self.scan_len > 0 and self.scan_remaining > 0:
            self.scan_active = True
            self.scan_remaining -= 1
            return self._scan_step_primitive()
        if (
            self.scan_len > 0
            and self.scan_interval > 0
            and self.tick % self.scan_interval == 0
        ):
            self.scan_active = True
            self.scan_remaining = max(0, self.scan_len - 1)
            self.scan_dir = "yaw_left" if (self.tick // self.scan_interval) % 2 == 0 else "yaw_right"
            return self._scan_step_primitive()
        if str(self.goal_policy).lower() in (
            "learned_sweep",
            "learned_local",
            "learned_wall_follow",
            "learned_policy",
        ):
            self.path = None
            self.goal_cell = None
            self.last_waypoint_cell = None
            return self.forward_primitive
        if self._route_active():
            self._advance_route_index(pos[:2])
            if self._route_active():
                goal = self._cell(self.route_waypoints[self.route_idx])
                if goal is not None and (
                    goal != self.goal_cell or not self.path or self.wp_idx >= len(self.path)
                ):
                    self.path = self._bfs_path(cur, goal)
                    self.wp_idx = 1
                    self.goal_cell = goal
                    self.replans += 1
                if goal is None or not self.path:
                    self.route_idx += 1
                    self.path = None
                    self.goal_cell = None
        # (Re)plan a path to the nearest unvisited frontier when we have none / finished.
        if not self.path or self.wp_idx >= len(self.path):
            goal = self._select_unvisited(cur, yaw)
            if goal is None:
                self.visited.clear()
                goal = self._select_unvisited(cur, yaw)
            self.path = self._bfs_path(cur, goal) if goal else None
            self.wp_idx = 1
            self.goal_cell = goal
            self.replans += 1
        if not self.path:
            return "arc_left"
        # Advance through reached waypoints (skip ones we're already near).
        while self.wp_idx < len(self.path):
            wx, wy = self.free[self.path[self.wp_idx]]
            if (wx - pos[0]) ** 2 + (wy - pos[1]) ** 2 < (self.step * 0.8) ** 2:
                self.wp_idx += 1
            else:
                break
        if self.wp_idx >= len(self.path):
            self.path = None
            self.last_waypoint_cell = None
            return self.forward_primitive
        self.last_waypoint_cell = self.path[self.wp_idx]
        wx, wy = self.free[self.last_waypoint_cell]
        bearing = wrap_angle_pi(math.atan2(wy - pos[1], wx - pos[0]) - yaw)
        self.last_bearing = float(bearing)
        if abs(bearing) > self.yaw_bearing_threshold:  # face the waypoint
            return "yaw_left" if bearing > 0 else "yaw_right"
        if not self.optimistic_free_graph and not _look_ahead_free(
            grid_global[0], pos[:2], yaw, self.lookahead_m
        ):
            # Manifest-grid lookahead is privileged; in online_frontier mode
            # the learned wall guard downstream owns pre-contact safety.
            return "yaw_left" if bearing >= 0 else "yaw_right"
        if abs(bearing) < self.forward_bearing_threshold:
            return self.forward_primitive
        return "arc_left" if bearing > 0 else "arc_right"


grid_global = [None]  # set in main so FrontierExplorer.primitive can look ahead


def _body_delta(prev, cur):
    """[dx_m, dy_m, dyaw] in prev body frame s.t. cur = R(-dyaw)(prev - [dx,dy])."""
    x0, y0, yaw0 = prev
    x1, y1, yaw1 = cur
    dxw, dyw = x1 - x0, y1 - y0
    dx = math.cos(yaw0) * dxw + math.sin(yaw0) * dyw
    dy = -math.sin(yaw0) * dxw + math.cos(yaw0) * dyw
    dyaw = wrap_angle_pi(yaw1 - yaw0)
    return float(dx), float(dy), float(dyaw)


def _build_aux(motion_m, command, primitive) -> np.ndarray:
    block = list(motion_m)
    window = list(motion_m)
    cmd = list(command)
    one_hot = [1.0 if primitive == n else 0.0 for n in PRIMITIVE_NAMES]
    return np.asarray(block + window + cmd + one_hot, dtype=np.float32)


def _learned_topology_route_contract_summary(path: Path) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "path": str(path),
        "schema": None,
        "source_dataset": None,
        "source_scene": None,
        "source_success": None,
        "route_count": 0,
        "waypoint_count": 0,
        "routes": {},
    }
    failures: list[str] = []
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return summary, [f"--learned-topology-route-table could not be read: {exc}"]

    summary["schema"] = table.get("schema")
    summary["source_dataset"] = table.get("source_dataset")
    summary["source_scene"] = table.get("source_scene")
    summary["source_success"] = table.get("source_success")
    if table.get("schema") != "lewm_go2_learned_topology_route_table_v1":
        failures.append(
            "--learned-topology-route-table must use schema "
            "lewm_go2_learned_topology_route_table_v1"
        )
    if table.get("source_success") is not True:
        failures.append("--learned-topology-route-table must come from a successful teacher trace")
    if not table.get("source_dataset"):
        failures.append("--learned-topology-route-table must record source_dataset")

    routes = table.get("routes", {})
    if not isinstance(routes, dict) or not routes:
        failures.append("--learned-topology-route-table must contain routes")
        return summary, failures

    route_summaries: dict[str, dict[str, Any]] = {}
    waypoint_count = 0
    for key, route in routes.items():
        if not isinstance(route, dict):
            continue
        waypoints = route.get("waypoints", [])
        count = len(waypoints) if isinstance(waypoints, list) else 0
        waypoint_count += count
        route_summaries[str(key)] = {
            "target_color": str(route.get("target_color", key)),
            "target_index": route.get("target_index"),
            "waypoint_count": int(count),
        }
    summary["route_count"] = int(len(route_summaries))
    summary["waypoint_count"] = int(waypoint_count)
    summary["routes"] = route_summaries
    if not route_summaries or waypoint_count <= 0:
        failures.append("--learned-topology-route-table must contain usable waypoints")
    return summary, failures


def _learned_local_policy_contract_summary(
    path: Path | None,
    *,
    flag_name: str = "--learned-local-policy-checkpoint",
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, []
    summary: dict[str, Any] = {
        "path": str(path),
        "schema": None,
        "model_type": None,
        "feature_variant": None,
        "input_dim": None,
        "primitive_count": 0,
        "forbid_output_primitives": [],
    }
    failures: list[str] = []
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return summary, [f"{flag_name} could not be read: {exc}"]
    summary["schema"] = checkpoint.get("schema")
    summary["model_type"] = checkpoint.get("model_type", "mlp")
    summary["feature_variant"] = checkpoint.get("feature_variant", "base")
    summary["input_dim"] = checkpoint.get("input_dim")
    primitive_vocab = checkpoint.get("primitive_vocab", [])
    summary["primitive_count"] = len(primitive_vocab) if isinstance(primitive_vocab, list) else 0
    summary["forbid_output_primitives"] = [
        str(item) for item in checkpoint.get("forbid_output_primitives", [])
    ]
    if checkpoint.get("schema") != "lewm_go2_closed_loop_learned_local_policy_v0":
        failures.append(
            f"{flag_name} must use schema "
            "lewm_go2_closed_loop_learned_local_policy_v0"
        )
    if not primitive_vocab:
        failures.append(f"{flag_name} must contain primitive_vocab")
    return summary, failures


def _parse_target_policy_checkpoint_specs(spec: str | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in str(spec or "").split(","):
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
        elif ":" in text:
            key, value = text.split(":", 1)
        else:
            raise ValueError(
                "--learned-local-target-policy-checkpoints entries must be color=path"
            )
        color = key.strip().lower()
        if not color:
            raise ValueError(
                "--learned-local-target-policy-checkpoints contains an empty color"
            )
        path_text = value.strip()
        if not path_text:
            raise ValueError(
                f"--learned-local-target-policy-checkpoints has no path for {color!r}"
            )
        result[color] = Path(path_text)
    return result


def _parse_target_state_policy_checkpoint_specs(spec: str | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in str(spec or "").split(","):
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
        else:
            raise ValueError(
                "--learned-local-target-policy-state-checkpoints entries must be color:STATE=path"
            )
        key = key.strip().lower()
        if ":" not in key:
            raise ValueError(
                "--learned-local-target-policy-state-checkpoints entries must be color:STATE=path"
            )
        color, state = key.split(":", 1)
        color = color.strip().lower()
        state = state.strip().upper()
        if not color or not state:
            raise ValueError(
                "--learned-local-target-policy-state-checkpoints contains an empty color or state"
            )
        path_text = value.strip()
        if not path_text:
            raise ValueError(
                f"--learned-local-target-policy-state-checkpoints has no path for {color}:{state}"
            )
        result[f"{color}:{state}"] = Path(path_text)
    return result


def _fully_learned_runtime_contract_report(
    args: argparse.Namespace,
    *,
    explore_route_waypoints: list[tuple[float, float]],
) -> dict[str, Any]:
    """Validate that enabled action sources are claim-clean at runtime."""

    failures: list[str] = []
    explore_policy = str(args.explore_goal_policy).lower()
    wall_source = str(args.wall_decision_source).lower()
    generalized = bool(getattr(args, "generalized_runtime_contract", False))
    learned_topology_route_summary: dict[str, Any] | None = None
    if args.learned_topology_route_table is not None:
        learned_topology_route_summary, route_failures = _learned_topology_route_contract_summary(
            args.learned_topology_route_table
        )
        failures.extend(route_failures)
    learned_local_policy_summary, policy_failures = _learned_local_policy_contract_summary(
        args.learned_local_policy_checkpoint
    )
    failures.extend(policy_failures)
    (
        learned_local_post_claim_policy_summary,
        post_claim_policy_failures,
    ) = _learned_local_policy_contract_summary(
        args.learned_local_post_claim_policy_checkpoint,
        flag_name="--learned-local-post-claim-policy-checkpoint",
    )
    failures.extend(post_claim_policy_failures)
    learned_local_target_policy_summaries: dict[str, Any] = {}
    try:
        target_policy_specs = _parse_target_policy_checkpoint_specs(
            getattr(args, "learned_local_target_policy_checkpoints", "")
        )
    except ValueError as exc:
        target_policy_specs = {}
        failures.append(str(exc))
    for color, checkpoint_path in sorted(target_policy_specs.items()):
        summary, target_failures = _learned_local_policy_contract_summary(
            checkpoint_path,
            flag_name=f"--learned-local-target-policy-checkpoints[{color}]",
        )
        learned_local_target_policy_summaries[color] = summary
        failures.extend(target_failures)
    try:
        target_state_policy_specs = _parse_target_state_policy_checkpoint_specs(
            getattr(args, "learned_local_target_policy_state_checkpoints", "")
        )
    except ValueError as exc:
        target_state_policy_specs = {}
        failures.append(str(exc))
    for key, checkpoint_path in sorted(target_state_policy_specs.items()):
        summary, target_failures = _learned_local_policy_contract_summary(
            checkpoint_path,
            flag_name=f"--learned-local-target-policy-state-checkpoints[{key}]",
        )
        learned_local_target_policy_summaries[key] = summary
        failures.extend(target_failures)

    uses_learned_local_policy = bool(
        explore_policy == "learned_policy"
        and args.learned_local_policy_checkpoint is not None
    )
    uses_online_frontier = bool(explore_policy == "online_frontier")
    uses_learned_topology_route_memory = bool(args.learned_topology_route_table is not None)

    if str(args.policy).lower() != "memory":
        failures.append("--policy must be memory")
    if str(args.demo_mode).lower() != "explore":
        failures.append("--demo-mode must be explore")
    if bool(args.face_target):
        failures.append("--face-target is a privileged diagnostic")
    if not (uses_learned_local_policy or uses_learned_topology_route_memory or uses_online_frontier):
        failures.append(
            "runtime EXPLORE action source must be --explore-goal-policy learned_policy "
            "with --learned-local-policy-checkpoint, online_frontier (optimistic coverage "
            "over the runtime-built contact map, no manifest reads), or "
            "--learned-topology-route-table"
        )
    if args.learned_topology_route_table is None:
        if explore_policy not in ("learned_policy", "online_frontier"):
            failures.append("--explore-goal-policy must be learned_policy or online_frontier without a route-memory table")
        if explore_policy == "learned_policy" and args.learned_local_policy_checkpoint is None:
            failures.append("--learned-local-policy-checkpoint is required without a route-memory table")
    if uses_learned_topology_route_memory and args.learned_topology_route_until_area_logit is None:
        failures.append("--learned-topology-route-until-area-logit is required for route-memory runtime")
    if bool(args.explore_standoff_route):
        failures.append("--explore-standoff-route is forbidden")
    if explore_route_waypoints:
        failures.append("--explore-route-waypoints is forbidden")
    if bool(args.learned_local_oracle_standoff_labels):
        failures.append("--learned-local-oracle-standoff-labels is for data collection only")
    if args.learned_local_dataset_output is not None:
        failures.append("--learned-local-dataset-output is for data collection only")
    if args.debug_force_primitive_script is not None:
        failures.append("--debug-force-primitive-script is for offline data collection only")
    if bool(args.wall_aware_planner) and wall_source == "privileged_grid":
        failures.append("--wall-decision-source privileged_grid is forbidden")
    if bool(args.wall_aware_planner) and wall_source == "learned_action" and args.primitive_outcome_checkpoint is None:
        failures.append("--primitive-outcome-checkpoint is required for learned_action wall decisions")
    if generalized:
        if not (uses_learned_local_policy or uses_online_frontier):
            failures.append(
                "--generalized-runtime-contract requires --explore-goal-policy "
                "learned_policy with --learned-local-policy-checkpoint or online_frontier"
            )
        if uses_learned_topology_route_memory:
            failures.append(
                "--learned-topology-route-table is same-scene route memory and "
                "is forbidden by --generalized-runtime-contract"
            )
        feature_variant = (
            ""
            if learned_local_policy_summary is None
            else str(learned_local_policy_summary.get("feature_variant", "base"))
        )
        if _learned_local_feature_variant_has_pose_topology(feature_variant):
            failures.append(
                "pose_topology learned-local checkpoints encode same-scene "
                "pose/topology features and are forbidden by --generalized-runtime-contract"
            )
        post_claim_feature_variant = (
            ""
            if learned_local_post_claim_policy_summary is None
            else str(learned_local_post_claim_policy_summary.get("feature_variant", "base"))
        )
        if _learned_local_feature_variant_has_pose_topology(post_claim_feature_variant):
            failures.append(
                "pose_topology post-claim learned-local checkpoints encode same-scene "
                "pose/topology features and are forbidden by --generalized-runtime-contract"
            )
        for color, target_policy_summary in sorted(learned_local_target_policy_summaries.items()):
            target_feature_variant = (
                ""
                if target_policy_summary is None
                else str(target_policy_summary.get("feature_variant", "base"))
            )
            if _learned_local_feature_variant_has_pose_topology(target_feature_variant):
                failures.append(
                    "pose_topology target learned-local checkpoints encode same-scene "
                    f"pose/topology features and are forbidden by --generalized-runtime-contract: {color}"
                )

    return {
        "enabled": bool(args.fully_learned_runtime_contract),
        "generalized": generalized,
        "passed": not failures,
        "failures": failures,
        "runtime_path": (
            "learned_topology_route_memory"
            if uses_learned_topology_route_memory
            else (
                "learned_local_policy"
                if uses_learned_local_policy
                else ("online_frontier" if uses_online_frontier else "invalid")
            )
        ),
        "learned_local_policy_checkpoint": learned_local_policy_summary,
        "learned_local_post_claim_policy_checkpoint": learned_local_post_claim_policy_summary,
        "learned_local_target_policy_checkpoints": learned_local_target_policy_summaries,
        "learned_topology_route_table": learned_topology_route_summary,
        "allowed_runtime_inputs": [
            "egocentric_rgb",
            "frozen_jepa_features",
            "learned_memory_state",
            "previous_primitive_and_proprioceptive_egomotion",
            "online_egomotion_visit_stall_and_claim_memory",
            *(
                []
                if generalized
                else [
                    "proprioceptive_pose_estimate_for_learned_topology_memory",
                    "learned_topology_route_memory_from_teacher_trace",
                ]
            ),
            "learned_primitive_outcome_predictions",
            "fixed_low_level_go2_primitive_executor",
        ],
        "forbidden_runtime_components_checked": [
            "standoff_route",
            "route_waypoints",
            "arbitrary_or_privileged_route_table_schema",
            "privileged_grid_wall_decision",
            "oracle_standoff_labels",
            "dataset_collection_labels",
            "nonlearned_explore_goal_policy",
            *(
                [
                    "same_scene_learned_topology_route_table",
                    "pose_topology_policy_features",
                ]
                if generalized
                else []
            ),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-corpus", type=Path,
                        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--platform-manifest", type=Path,
                        default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path,
                        default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument(
        "--slice-start-result",
        type=Path,
        default=None,
        help=(
            "Start a short physical benchmark slice from a pose in a prior result log. "
            "This is for fast local-policy iteration; full all-beacon runs remain the "
            "acceptance test."
        ),
    )
    parser.add_argument(
        "--slice-start-tick",
        type=int,
        default=0,
        help="First result-log tick at or after which to take post_xy/post_yaw for a slice start.",
    )
    parser.add_argument(
        "--slice-active-target-color",
        default="",
        help="If set with --slice-start-result, force this target color active at slice start.",
    )
    parser.add_argument(
        "--slice-preclaimed-colors",
        default="",
        help="Comma-separated colors to seed as already claimed for a slice benchmark.",
    )
    parser.add_argument(
        "--slice-feature-max-ticks",
        type=int,
        default=0,
        help=(
            "Clock denominator for learned features in slice mode. Use the source full-run "
            "budget, e.g. 1000, when --max-ticks is shortened."
        ),
    )
    parser.add_argument(
        "--slice-preload-online-map",
        action="store_true",
        help=(
            "Reconstruct the online egomotion map up to --slice-start-tick from the "
            "source result log. This restores runtime-safe map memory but not the "
            "controller RNN hidden state."
        ),
    )
    parser.add_argument(
        "--slice-snapshot-output",
        type=Path,
        default=None,
        help="Save a faithful resume snapshot containing pose, controller state, claims, and online map.",
    )
    parser.add_argument(
        "--slice-snapshot-after-claims",
        type=int,
        default=0,
        help="When >0 with --slice-snapshot-output, write the snapshot after this many claims.",
    )
    parser.add_argument(
        "--slice-snapshot-at-tick",
        type=int,
        default=-1,
        help=(
            "When >=0 with --slice-snapshot-output, write the snapshot after executing "
            "the first tick whose index is at least this value."
        ),
    )
    parser.add_argument(
        "--slice-snapshot-exit",
        action="store_true",
        help="Exit the rollout immediately after writing --slice-snapshot-output.",
    )
    parser.add_argument(
        "--slice-snapshot-input",
        type=Path,
        default=None,
        help="Resume from a faithful slice snapshot saved by --slice-snapshot-output.",
    )
    parser.add_argument("--mode", choices=("kinematic", "physical"), default="kinematic",
                        help="physical: drive the robot with the trained Go2 PPO walking "
                             "policy stepped in Genesis physics (real gait + rigid-body "
                             "collisions); kinematic: integrate named velocity primitives "
                             "against a 2D grid (teleport, no contact).")
    parser.add_argument("--policy-device", default="cpu",
                        help="Torch device for the PPO locomotion policy (physical mode).")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the JEPA/controller and learned wall/action heads.")
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15,
                        help="Base-height fall threshold for the physical rollout.")
    parser.add_argument("--tip-threshold-rad", type=float, default=math.radians(60.0),
                        help="Maximum allowed absolute body roll or pitch in physical mode.")
    parser.add_argument("--allow-unstable-base-success", action="store_true",
                        help="Do not fail success on physical base fall/tip events.")
    parser.add_argument("--success-min-body-clearance-m", type=float, default=0.0,
                        help="Minimum post-primitive body footprint clearance required for success.")
    parser.add_argument("--allow-body-clearance-violation-success", action="store_true",
                        help="Do not fail success on body footprint clearance violations.")
    parser.add_argument("--policy", choices=("wander", "memory"), default="memory")
    parser.add_argument("--demo-mode", choices=("explore", "recall"), default="explore",
                        help="recall: place at a line-of-sight standoff (privileged scaffold), "
                             "observe to bind, turn away to hide, then memory recalls + claims "
                             "(the 2D see->hide->recall analog). explore: autonomous coverage.")
    parser.add_argument("--fully-learned-runtime-contract", action="store_true",
                        help="Fail fast unless EXPLORE action selection uses the learned "
                             "local-policy hook or learned topology route memory with no "
                             "standoff route, no route waypoints, no oracle labels, and no "
                             "privileged-grid wall decisions. Post-run grid safety metrics are "
                             "still recorded.")
    parser.add_argument("--generalized-runtime-contract", action="store_true",
                        help="Stricter fully learned contract for scene-disjoint claims: "
                             "requires learned-local policy inference, forbids learned topology "
                             "route tables, and rejects pose/topology policy features.")
    parser.add_argument("--explore-goal-policy", choices=("nearest", "farthest", "mixed", "dfs", "online_frontier", "learned_sweep", "learned_local", "learned_wall_follow", "learned_policy"), default="nearest",
                        help="Unvisited free-cell selection policy for autonomous EXPLORE coverage. "
                             "learned_sweep skips manifest-route waypointing and lets the learned "
                             "action-outcome guard select safe traversal primitives from RGB/JEPA "
                             "latents. learned_local also skips waypointing but chooses the EXPLORE "
                             "request directly from learned RGB/JEPA primitive progress/risk scores "
                             "plus a target-independent scan prior. learned_wall_follow skips "
                             "waypointing and follows a persistent side using only learned "
                             "RGB/JEPA primitive outcome predictions. learned_policy skips waypointing "
                             "and delegates EXPLORE primitive choice to --learned-local-policy-checkpoint. "
                             "Graph policies use the occupancy/free-cell graph, never "
                             "target position.")
    parser.add_argument("--explore-yaw-bearing-threshold", type=float, default=0.5,
                        help="Waypoint bearing magnitude above which EXPLORE requests in-place yaw "
                             "instead of an arc primitive.")
    parser.add_argument("--explore-forward-bearing-threshold", type=float, default=0.18,
                        help="Waypoint bearing magnitude below which EXPLORE requests straight forward.")
    parser.add_argument("--explore-lookahead-m", type=float, default=0.35,
                        help="Forward grid probe distance before EXPLORE requests an avoidance turn.")
    parser.add_argument("--explore-forward-primitive", choices=("forward_slow", "forward_medium", "forward_fast"),
                        default="forward_medium",
                        help="Straight primitive requested by the EXPLORE path follower.")
    parser.add_argument("--explore-coverage-lookahead-cells", type=int, default=8,
                        help="For dfs EXPLORE policy, target this many usable route cells ahead "
                             "instead of replanning one adjacent cell at a time.")
    parser.add_argument("--explore-dfs-neighbor-order",
                        choices=("nesw", "eswn", "senw", "wsen", "heading"),
                        default="nesw",
                        help="Neighbor expansion order for dfs EXPLORE coverage. heading uses "
                             "the robot's initial facing direction as a target-independent prior.")
    parser.add_argument("--explore-scan-interval", type=int, default=24,
                        help="EXPLORE look-around interval in ticks. Set <=0 to disable periodic scans.")
    parser.add_argument("--explore-scan-len", type=int, default=7,
                        help="Number of EXPLORE look-around ticks per scan. Set 0 to disable scans.")
    parser.add_argument("--explore-scan-primitive", choices=("yaw", "arc"), default="yaw",
                        help="Primitive family used for periodic look-around. arc keeps the robot "
                             "moving while widening visual coverage; yaw preserves old behavior.")
    parser.add_argument("--explore-route-waypoints", default="",
                        help="Optional semicolon-separated coarse route waypoints as x,y pairs. "
                             "When active, EXPLORE follows this corridor bias before falling back "
                             "to target-independent coverage; local wall decisions still come "
                             "from --wall-decision-source.")
    parser.add_argument("--explore-route-start-after-claims", type=int, default=0,
                        help="Number of beacon claims before --explore-route-waypoints activates.")
    parser.add_argument("--explore-route-advance-m", type=float, default=0.55,
                        help="Distance to a route waypoint at which EXPLORE advances to the next "
                             "coarse waypoint.")
    parser.add_argument("--explore-standoff-route", action="store_true",
                        help="Target-aware benchmark scaffold: in EXPLORE, plan to a line-of-sight "
                             "standoff for the active beacon on the inflated body grid, then let "
                             "the learned wall/action guard select safe primitive execution.")
    parser.add_argument("--explore-standoff-m", type=float, default=1.05,
                        help="Base-center standoff distance from the active beacon for "
                             "--explore-standoff-route.")
    parser.add_argument("--explore-standoff-lookahead-m", type=float, default=0.55,
                        help="Pure-pursuit lookahead distance along the standoff A* path.")
    parser.add_argument("--explore-standoff-replan-interval", type=int, default=12,
                        help="Closed-loop ticks between periodic standoff-route replans.")
    parser.add_argument("--explore-standoff-candidates", type=int, default=16,
                        help="Number of angular line-of-sight standoff candidates per beacon.")
    parser.add_argument("--explore-standoff-arrival-m", type=float, default=0.45,
                        help="Distance from the selected standoff at which the route follower "
                             "turns to face the beacon center.")
    parser.add_argument("--explore-standoff-path-spacing-m", type=float, default=0.30,
                        help="Maximum spacing when sparsifying the standoff A* route. Smaller "
                             "values keep denser corner waypoints so pure-pursuit does not cut "
                             "inside the body-inflated corridor.")
    parser.add_argument("--explore-standoff-clearance-weight", type=float, default=0.0,
                        help="Optional A* step-cost weight for low configuration-space "
                             "clearance in the privileged standoff-route scaffold. Zero keeps "
                             "the shortest-path route.")
    parser.add_argument("--explore-standoff-clearance-target-m", type=float, default=0.0,
                        help="Configuration-space clearance target for "
                             "--explore-standoff-clearance-weight.")
    parser.add_argument("--explore-standoff-body-route-clearance-weight", type=float, default=0.0,
                        help="Candidate standoff-route cost weight for low explicit Go2 "
                             "body-probe clearance along route waypoints.")
    parser.add_argument("--explore-standoff-body-route-clearance-target-m", type=float, default=0.0,
                        help="Route waypoint body-probe clearance target for "
                             "--explore-standoff-body-route-clearance-weight.")
    parser.add_argument("--explore-standoff-body-route-ignore-start-m", type=float, default=0.0,
                        help="Ignore this much path distance from the current pose when scoring "
                             "candidate standoff routes by explicit Go2 body clearance.")
    parser.add_argument("--explore-standoff-cardinal-route", action="store_true",
                        help="Use 4-connected A* for the privileged standoff-route scaffold. "
                             "This avoids diagonal corner cuts that a centerline grid may allow "
                             "but a physical Go2 shoulder envelope cannot follow cleanly.")
    parser.add_argument("--explore-standoff-corner-guard", action="store_true",
                        help="For cardinal standoff routes, cap pure-pursuit lookahead at the "
                             "next route corner until the base is close enough to commit to the "
                             "turn.")
    parser.add_argument("--explore-standoff-corner-commit-m", type=float, default=0.12,
                        help="Distance to a cardinal route corner at which "
                             "--explore-standoff-corner-guard may look beyond the turn.")
    parser.add_argument("--explore-standoff-corner-standoff-m", type=float, default=0.0,
                        help="When --explore-standoff-corner-guard is active, first target a "
                             "point this far before the next cardinal corner instead of aiming "
                             "the base center at the corner vertex. This gives the Go2 shoulder "
                             "envelope room to square up before committing to the turn.")
    parser.add_argument("--explore-standoff-allow-arcs", action="store_true",
                        help="For standoff routes, request arc primitives for moderate bearing "
                             "errors and reserve yaw-in-place for larger heading errors. The "
                             "learned wall/action guard still scores and may veto the request.")
    parser.add_argument("--explore-standoff-arc-min-bearing", type=float, default=None,
                        help="Minimum absolute standoff-route bearing error required before "
                             "an arc primitive can be requested. Defaults to the forward "
                             "bearing threshold, preserving the legacy behavior.")
    parser.add_argument("--explore-standoff-arc-max-bearing", type=float, default=None,
                        help="Maximum absolute standoff-route bearing error that may request "
                             "an arc primitive. Defaults to the yaw bearing threshold, "
                             "preserving the legacy behavior.")
    parser.add_argument("--explore-standoff-arc-min-target-dist-m", type=float, default=0.0,
                        help="If positive, suppress standoff-route arc primitives when the "
                             "active beacon is closer than this distance; this hands near-beacon "
                             "alignment back to yaw/forward primitives to reduce orbiting.")
    parser.add_argument("--explore-standoff-heading-mode", default="target",
                        choices=("target", "tangent", "path_tangent", "route_tangent"),
                        help="Heading target used by the standoff route follower. 'target' "
                             "points the body at the pure-pursuit point; tangent modes aim "
                             "along the current route segment so corners are approached "
                             "squarely before the learned wall/action guard scores primitives.")
    parser.add_argument("--explore-standoff-heading-lookahead-m", type=float, default=0.35,
                        help="Lookahead distance along the current route segment for "
                             "tangent standoff heading modes.")
    parser.add_argument("--explore-standoff-prefix-snap-start", action="store_true",
                        help="If the current physical pose is outside the inflated route "
                             "grid and must be snapped to a nearby safe cell, prefix that "
                             "safe snapped start as the first standoff waypoint. This makes "
                             "the route follower recover to the centerline before advancing.")
    parser.add_argument("--explore-standoff-snap-start-min-dist-m", type=float, default=0.05,
                        help="Minimum current-pose to snapped-start distance before "
                             "--explore-standoff-prefix-snap-start inserts a recovery waypoint.")
    parser.add_argument("--explore-standoff-body-check", action="store_true",
                        help="For the privileged standoff-route scaffold, veto a route forward "
                             "request when a swept Go2 footprint probe would lose shoulder "
                             "clearance around an inside corner. Runtime wall decisions can "
                             "still use --wall-decision-source learned_action.")
    parser.add_argument("--explore-standoff-body-lookahead-m", type=float, default=0.30,
                        help="Forward distance used by --explore-standoff-body-check.")
    parser.add_argument("--explore-standoff-body-min-clearance-m", type=float, default=-0.02,
                        help="Minimum swept body configuration clearance for "
                             "--explore-standoff-body-check.")
    parser.add_argument("--explore-standoff-body-recovery-clearance-m", type=float, default=None,
                        help="If set, and the current body probe clearance drops below this "
                             "buffer during --explore-standoff-body-check, prefer a "
                             "clearance-improving translating primitive over route yaw. This "
                             "prevents marginal-shoulder wall walking before the route resumes.")
    parser.add_argument("--explore-standoff-intent-smoothing", action="store_true",
                        help="Opt-in standoff route-intent smoothing: sticky route target plus yaw hysteresis.")
    parser.add_argument("--explore-standoff-sticky-target-ticks", type=int, default=0,
                        help="When intent smoothing is enabled, hold the current standoff lookahead target "
                             "for up to this many ticks unless reached or line-of-sight is lost.")
    parser.add_argument("--explore-standoff-sticky-target-release-m", type=float, default=0.18,
                        help="Distance at which a sticky standoff lookahead target is considered reached.")
    parser.add_argument("--explore-standoff-yaw-enter-threshold", type=float, default=None,
                        help="Bearing magnitude that enters standoff yaw mode. Defaults to max(forward threshold, 0.16).")
    parser.add_argument("--explore-standoff-yaw-exit-threshold", type=float, default=None,
                        help="Bearing magnitude that exits standoff yaw mode. Defaults below the enter threshold.")
    parser.add_argument("--explore-standoff-yaw-flip-threshold", type=float, default=None,
                        help="Bearing magnitude needed to flip directly between yaw_left and yaw_right.")
    parser.add_argument("--explore-standoff-route-until-area-logit", type=float, default=None,
                        help="When --explore-standoff-route is active, keep using the standoff "
                             "route while the active target is unseen, outside the cone, or "
                             "visible with an RGB area logit below this value. This prevents "
                             "early SERVO from chasing tiny far-away blobs through walls.")
    parser.add_argument("--explore-standoff-release-on-seen", action="store_true",
                        help="When the active target has been remembered and the robot is near "
                             "the chosen standoff point, release the standoff route even if the "
                             "target is not yet centered. This lets visual SEEK/SERVO handle "
                             "final alignment instead of orbiting the standoff route.")
    parser.add_argument("--explore-standoff-release-m", type=float, default=0.55,
                        help="Distance to the chosen standoff point at which "
                             "--explore-standoff-release-on-seen may hand off to visual "
                             "SEEK/SERVO.")
    parser.add_argument("--explore-standoff-release-min-seen-ticks", type=int, default=4,
                        help="Minimum remembered-target age before "
                             "--explore-standoff-release-on-seen may hand off to visual "
                             "SEEK/SERVO.")
    parser.add_argument("--explore-reset-on-claim", action="store_true",
                        help="For multi-beacon episodes, discard the current EXPLORE path and DFS "
                             "coverage cursor after each claim so the next target starts coverage "
                             "from the robot's new pose. Learned blocked-cell evidence is kept.")
    parser.add_argument("--explore-clear-visited-on-claim", action="store_true",
                        help="With --explore-reset-on-claim, also clear visited coverage cells. "
                             "Blocked cells are still kept; this is mainly a diagnostic escape "
                             "hatch if previous coverage over-prunes the next beacon search.")
    parser.add_argument("--observe-ticks", type=int, default=6)
    parser.add_argument("--hide-ticks", type=int, default=10)
    parser.add_argument("--controller", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, default=None)
    parser.add_argument("--target-color", default="green")
    parser.add_argument("--target-colors", default=None,
                        help="Comma-separated target colors for one continuous all-beacon "
                             "episode. The controller state and robot pose are not reset "
                             "between claims; the queried color advances after each claim.")
    parser.add_argument("--post-claim-explore-primitives", default="",
                        help="Comma-separated primitives to request immediately after a "
                             "successful claim before returning to the learned policy. "
                             "The learned action-outcome guard still scores the request; "
                             "this uses no map geometry or target coordinates.")
    parser.add_argument("--post-claim-explore-min-claimed-count", type=int, default=0,
                        help="Minimum number of total claimed beacons required before "
                             "--post-claim-explore-primitives schedules a post-claim plan.")
    parser.add_argument("--multi-target-switch-policy",
                        choices=("fixed", "seen_when_active_unseen", "visible_priority", "memory_priority"),
                        default="fixed",
                        help="For --target-colors episodes, optionally switch the active "
                             "target using only controller perception/memory outputs. "
                             "fixed preserves the requested order. seen_when_active_unseen "
                             "switches to an unclaimed remembered color only while the "
                             "current target is not remembered. visible_priority switches "
                             "to an unclaimed currently visible color if the current target "
                             "is not currently visible. memory_priority always considers "
                             "unclaimed remembered colors and prefers visible ones.")
    parser.add_argument("--multi-target-switch-conf", type=float, default=None,
                        help="Memory confidence required for non-fixed multi-target "
                             "switching. Defaults to --seen-conf.")
    parser.add_argument("--learned-target-scheduler-checkpoint", type=Path, default=None,
                        help="Learned per-tick target-color scheduler using only "
                             "controller color readouts and the claimed-color set. "
                             "When set, it runs before rule-based multi-target "
                             "switching and masks already-claimed colors.")
    parser.add_argument("--learned-target-scheduler-log-scores", action="store_true",
                        help="Log per-color learned scheduler probabilities every tick.")
    parser.add_argument("--multi-target-switch-area-logit", type=float, default=0.0,
                        help="RGB area logit required for visible_priority target "
                             "switching.")
    parser.add_argument("--multi-target-seen-switch-min-area-logit", type=float, default=None,
                        help="Optional RGB area-logit floor for candidate targets under "
                             "seen_when_active_unseen. By default that policy preserves "
                             "its historical memory-confidence-only candidate gate.")
    parser.add_argument("--multi-target-stale-seen-switch-after-frontier-noops",
                        type=int, default=0,
                        help="If >0, after this many consecutive frontier-pressure "
                             "noops, switch from an unreliable active target to the "
                             "most recently seen unclaimed color using only controller "
                             "memory/readout history.")
    parser.add_argument("--multi-target-stale-seen-switch-max-age-ticks",
                        type=int, default=160,
                        help="Maximum age of a last-seen unclaimed color for "
                             "--multi-target-stale-seen-switch-after-frontier-noops. "
                             "Set <=0 to allow any age.")
    parser.add_argument("--multi-target-opportunistic-claims", action="store_true",
                        help="Allow any unclaimed color with a large centered RGB/evidence "
                             "readout to be claimed, even when it is not the active query. "
                             "This is route-free and uses only controller perception outputs.")
    parser.add_argument("--multi-target-opportunistic-claim-area-logit", type=float, default=None,
                        help="Area-logit threshold for --multi-target-opportunistic-claims. "
                             "Defaults to the per-color near-claim threshold.")
    parser.add_argument("--multi-target-opportunistic-claim-bearing", type=float, default=None,
                        help="Bearing threshold for --multi-target-opportunistic-claims. "
                             "Defaults to the per-color near-claim bearing.")
    parser.add_argument("--multi-target-opportunistic-claim-min-visible-ticks", type=int, default=2,
                        help="Consecutive centered visual ticks required before an "
                             "opportunistic non-active target claim.")
    parser.add_argument("--weak-memory-seek-conf", type=float, default=0.0,
                        help="For route-free multi-target episodes, treat non-visible "
                             "SEEK memories below this confidence as unreliable after "
                             "repeated stalls and temporarily return to EXPLORE. Set <=0 "
                             "to disable.")
    parser.add_argument("--weak-memory-seek-area-logit", type=float, default=None,
                        help="For route-free multi-target episodes, also treat SEEK as "
                             "recoverable when the active target is remembered but the "
                             "current RGB area logit is below this threshold. This uses "
                             "only controller perception and prevents high-confidence "
                             "but non-visible memories from trapping exploration.")
    parser.add_argument("--weak-memory-seek-colors", default="",
                        help="Optional comma-separated active target colors where weak-memory "
                             "SEEK recovery may fire. Empty allows all colors.")
    parser.add_argument("--weak-memory-seek-stall-streak", type=int, default=2,
                        help="Consecutive stalled SEEK ticks required to trigger "
                             "--weak-memory-seek-conf recovery.")
    parser.add_argument("--weak-memory-seek-yaw-loop-streak", type=int, default=0,
                        help="If >0, consecutive weak SEEK yaw ticks with near-zero "
                             "translation required to arm EXPLORE recovery. This "
                             "catches centered-target yaw oscillations that are not "
                             "classified as stalls.")
    parser.add_argument("--weak-memory-seek-yaw-loop-max-displacement-m",
                        type=float, default=0.01,
                        help="Maximum per-tick XY displacement counted as a weak SEEK "
                             "yaw-loop tick.")
    parser.add_argument("--weak-memory-seek-explore-cooldown-ticks", type=int, default=36,
                        help="EXPLORE ticks to force after weak-memory SEEK stalls.")
    parser.add_argument("--weak-memory-seek-force-explore-on-recovery", action="store_true",
                        help="When weak-memory recovery is active, keep route-free multi-target "
                             "control in EXPLORE immediately instead of waiting for SEEK stalls. "
                             "This uses only controller memory/RGB outputs and avoids chasing "
                             "high-confidence but visually weak memories through walls.")
    parser.add_argument("--target-pursuit-stale-ticks", type=int, default=0,
                        help="If >0, consecutive SEEK/SERVO target-pursuit ticks with no "
                             "passing claim gate before forcing a temporary EXPLORE cooldown. "
                             "This is runtime-safe and prevents stale visual/memory target "
                             "latches from occupying the rest of an all-beacon episode.")
    parser.add_argument("--target-pursuit-stale-states", default="SEEK,SERVO",
                        help="Comma-separated controller states counted by "
                             "--target-pursuit-stale-ticks.")
    parser.add_argument("--target-pursuit-stale-explore-cooldown-ticks", type=int, default=36,
                        help="EXPLORE ticks to force after --target-pursuit-stale-ticks arms.")
    parser.add_argument("--target-pursuit-stale-window-ticks", type=int, default=0,
                        help="If >0, count stale target-pursuit candidate ticks over this "
                             "rolling per-color window instead of requiring a perfectly "
                             "consecutive SEEK/SERVO run. The threshold remains "
                             "--target-pursuit-stale-ticks.")
    parser.add_argument("--target-pursuit-stale-suppress-color-ticks", type=int, default=0,
                        help="If >0, temporarily suppress the stale active color from "
                             "multi-target switching after a stale-pursuit recovery. "
                             "This lets the learned explorer break far-readout loops "
                             "without using target positions or a route table.")
    parser.add_argument("--seen-read-threshold", type=float, default=None,
                        help="Optional learned read-score threshold for treating a target "
                             "as remembered. The score comes from the RGB/JEPA vector-memory "
                             "controller's read head; unset preserves legacy memory_conf gating.")
    parser.add_argument("--log-color-readouts", action="store_true",
                        help="Include compact per-target memory/read/area telemetry and "
                             "multi-target switch gate diagnostics in the JSON log.")
    parser.add_argument("--learned-wall-follow-side-period", type=int, default=160,
                        help="EXPLORE ticks before the learned wall-follow local planner "
                             "flips its side preference. Set <=0 to keep a fixed side.")
    parser.add_argument("--learned-wall-follow-initial-side", choices=("left", "right"),
                        default="right",
                        help="Initial side preference for learned_wall_follow.")
    parser.add_argument("--learned-wall-follow-flip-on-claim", action="store_true",
                        help="Flip learned_wall_follow side after each beacon claim.")
    parser.add_argument("--learned-wall-follow-safe-risk", type=float, default=0.58,
                        help="Maximum learned blocked/clearance risk accepted by "
                             "learned_wall_follow translating primitives.")
    parser.add_argument("--learned-wall-follow-progress-floor", type=float, default=0.02,
                        help="Minimum learned progress accepted by learned_wall_follow "
                             "translating primitives.")
    parser.add_argument("--learned-wall-follow-turn-pressure-after", type=int, default=5,
                        help="Consecutive yaw ticks before learned_wall_follow forces a "
                             "safe translating primitive or backward escape.")
    parser.add_argument("--learned-local-policy-checkpoint", type=Path, default=None,
                        help="Optional learned primitive policy trained from offline "
                             "teacher labels. Runtime inference consumes only the online learned "
                             "memory/controller state and learned RGB/JEPA action-outcome features, "
                             "not standoff routes or map geometry.")
    parser.add_argument("--learned-local-post-claim-policy-checkpoint", type=Path, default=None,
                        help="Optional second learned primitive policy used only for "
                             "--learned-local-policy-post-claim-states after at least one beacon "
                             "has been claimed. This allows post-claim pursuit to be learned "
                             "without changing the pre-claim EXPLORE checkpoint.")
    parser.add_argument("--learned-local-post-claim-policy-min-claims", type=int, default=1,
                        help="Minimum claimed beacon count before "
                             "--learned-local-post-claim-policy-checkpoint may override actions. "
                             "This lets a learned recovery policy specialize to late-episode "
                             "acquisition without perturbing early visible-target claims.")
    parser.add_argument("--learned-local-target-policy-checkpoints", default="",
                        help="Optional comma-separated color=checkpoint map for learned "
                             "primitive policies keyed by active target color. The router "
                             "uses only task target identity and runtime claim memory; each "
                             "checkpoint is included in the fully learned contract report.")
    parser.add_argument("--learned-local-target-policy-state-checkpoints", default="",
                        help="Optional comma-separated color:STATE=checkpoint map for learned "
                             "primitive policies keyed by active target color and controller "
                             "state. Uses only task color, runtime claim memory, and controller "
                             "state; each checkpoint is included in the contract report.")
    parser.add_argument("--learned-local-target-policy-priority-over-post-claim",
                        action="store_true",
                        help="When a target-specific learned-local checkpoint exists for the "
                             "active color, route through it before the generic post-claim "
                             "checkpoint. This uses only active task color and claim memory.")
    parser.add_argument("--learned-local-target-policy-priority-on-aux-clearance-switch",
                        action="store_true",
                        help="When the learned auxiliary clearance switch is active, route "
                             "through the active target-specific learned-local checkpoint "
                             "before the generic post-claim checkpoint. This gates policy "
                             "routing on learned RGB/JEPA risk, not pose or geometry.")
    parser.add_argument("--learned-local-target-policy-outcome-rerank",
                        choices=("inherit", "on", "off"), default="inherit",
                        help="Override learned-local outcome reranking for target-specific "
                             "policy checkpoints. Defaults to inheriting "
                             "--learned-local-policy-outcome-rerank.")
    parser.add_argument("--learned-local-target-policy-rerank-policy-weight",
                        type=float, default=None,
                        help="Optional policy-logit weight override for reranking target-specific "
                             "learned-local policies. Defaults to inheriting "
                             "--learned-local-policy-rerank-policy-weight.")
    parser.add_argument("--learned-local-policy-states", default="EXPLORE",
                        help="Comma-separated controller states where the learned primitive "
                             "policy may collect labels or override actions. Use EXPLORE,SEEK,SERVO "
                             "for full closed-loop scaffold replacement.")
    parser.add_argument("--learned-local-policy-post-claim-states", default="",
                        help="Additional controller states where the learned primitive policy "
                             "may run after at least one target has been claimed. This keeps "
                             "pre-claim behavior unchanged while allowing learned post-claim "
                             "target pursuit.")
    parser.add_argument("--learned-local-dataset-states", default="",
                        help="Optional comma-separated controller states to capture for "
                             "--learned-local-dataset-output. Empty preserves "
                             "--learned-local-policy-states plus post-claim states.")
    parser.add_argument("--learned-local-dataset-output", type=Path, default=None,
                        help="If set, save feature/teacher-label pairs to this NPZ for states "
                             "enabled by --learned-local-dataset-states or "
                             "--learned-local-policy-states. "
                             "Use with the privileged standoff route enabled to generate offline "
                             "labels, then rerun with --explore-standoff-route off.")
    parser.add_argument("--learned-local-dataset-label-source",
                        choices=("teacher", "executed"), default="teacher",
                        help="For --learned-local-dataset-output, save labels from the pre-policy "
                             "teacher/request path (default) or from the final primitive selected "
                             "for execution after learned policy and runtime guards.")
    parser.add_argument("--learned-local-dataset-min-claimed-count", type=int, default=0,
                        help="For learned-local dataset collection, skip rows before this many "
                             "target claims have been recorded.")
    parser.add_argument("--debug-force-primitive-script", type=Path, default=None,
                        help="Offline data-collection/debug only: force final executable "
                             "primitives at specific ticks from a JSON/JSONL/text script. "
                             "This is forbidden by the fully learned runtime contract.")
    parser.add_argument("--learned-local-oracle-standoff-labels", action="store_true",
                        help="For dataset collection only, label EXPLORE examples with a separate "
                             "standoff-route oracle while executing the normal controller action. "
                             "This is intended for DAgger-style recovery data; runtime evaluation "
                             "must leave this off.")
    parser.add_argument("--learned-local-oracle-standoff-label-states", default="EXPLORE",
                        help="Comma-separated controller states where "
                             "--learned-local-oracle-standoff-labels may replace the executed "
                             "primitive label. This is collection-only and must be disabled for "
                             "strict runtime evaluation.")
    parser.add_argument("--learned-local-clock-features", action="store_true",
                        help="Append runtime-safe normalized tick/target-progress scalars to "
                             "learned-local policy features. Checkpoints tagged clock_v1 enable "
                             "this automatically at inference.")
    parser.add_argument("--learned-local-state-features", action="store_true",
                        help="Append a runtime-safe one-hot controller-state feature to "
                             "learned-local policy features. Checkpoints tagged state_v1 or "
                             "clock_state_v1 enable this automatically at inference.")
    parser.add_argument("--learned-local-visual-readout-features", action="store_true",
                        help="Append runtime-safe active-target RGB/controller readout scalars "
                             "to learned-local policy features. Checkpoints tagged "
                             "visual_readout enable this automatically at inference.")
    parser.add_argument("--learned-local-pose-topology-features", action="store_true",
                        help="Append same-scene odometry-like pose/topology scalars to learned-local "
                             "features. Checkpoints tagged pose_topology_v1 enable this automatically.")
    parser.add_argument("--learned-local-pose-scale-m", type=float, default=4.0,
                        help="Position scale used by --learned-local-pose-topology-features.")
    parser.add_argument("--learned-local-online-map-features", action="store_true",
                        help="Append a runtime-safe egomotion visit/stall map to learned-local "
                             "features. Checkpoints tagged online_map enable this automatically.")
    parser.add_argument("--learned-local-online-map-size", type=int, default=11,
                        help="Odd egocentric side length for --learned-local-online-map-features.")
    parser.add_argument("--learned-local-online-map-cell-m", type=float, default=0.45,
                        help="Cell size in meters for --learned-local-online-map-features.")
    parser.add_argument("--learned-local-online-map-hard-guard-blocks", action="store_true",
                        help="Treat learned wall-guard blocked edges as hard online-map frontier blocks.")
    parser.add_argument("--learned-local-online-map-wall-guard-block-source",
                        choices=("all", "requested", "selected", "requested_selected", "none"),
                        default="all",
                        help="Which learned wall/action guard blocked candidates may update "
                             "the runtime online-map guard-block memory. The default preserves "
                             "legacy behavior. requested_selected avoids hard-blocking routes "
                             "from guard candidates the robot never attempted.")
    parser.add_argument("--learned-local-online-map-current-contact-projection-blocks",
                        action="store_true",
                        help="When current-contact projected-clearance gating rejects the "
                             "selected translating primitive, mark that primitive's online-map "
                             "edge blocked so frontier pressure can reroute instead of "
                             "requesting the same unsafe edge forever.")
    parser.add_argument("--learned-local-online-map-hard-veto-hold-escape-projection-blocks",
                        action="store_true",
                        help="When projected-clearance filtering leaves hard-veto hold escape "
                             "without a safe translating recovery, mark the primitive that was "
                             "hard-vetoed to hold as an online-map guard block.")
    parser.add_argument("--learned-local-online-map-geometry-veto-hold-blocks",
                        action="store_true",
                        help="When the generic body-clearance geometry veto replaces a "
                             "translating primitive with hold, mark the blocked primitive's "
                             "online-map edge as a guard block so frontier pressure can reroute.")
    parser.add_argument("--learned-local-online-map-route-replay-guard-override",
                        action="store_true",
                        help="Let a learned-policy translating request pass through the "
                             "learned action guard when the runtime online egomotion map "
                             "shows that the primitive follows a previously visited, "
                             "unblocked route edge toward a self-built frontier. This uses "
                             "no scene map, route table, or privileged planner.")
    parser.add_argument("--learned-local-online-map-low-progress-block-m",
                        type=float, default=0.0,
                        help="If >0, mark a translating online-map edge blocked when the "
                             "executed egomotion displacement is below this threshold, "
                             "without counting it as a contact-like stall. This is a "
                             "runtime-safe proprioceptive progress guard for repeated "
                             "false-clear learned action predictions.")
    parser.add_argument("--stability-guard-roll-pitch-threshold", type=float, default=0.0,
                        help="If >0 (physical mode), when the previous tick's |roll| or "
                             "|pitch| meets this threshold, hold in place for "
                             "--stability-guard-hold-ticks so the gait settles instead of "
                             "levering the body over against a wall. Proprioceptive only.")
    parser.add_argument("--stability-guard-hold-ticks", type=int, default=4)
    parser.add_argument("--stability-guard-primitive", default="hold",
                        help="Recovery primitive when the stability guard fires. hold "
                             "freezes in place; backward steps out of a wall-lean, which "
                             "is statically stable and cannot be fixed by holding.")
    parser.add_argument("--learned-local-online-map-rotation-stall-block-ticks",
                        type=int, default=0,
                        help="If >0, after this many consecutive turn ticks with almost no "
                             "executed rotation (physical wall contact can mechanically "
                             "block in-place yaw), mark the online-map edge toward the "
                             "current frontier route target blocked so routing moves on. "
                             "Proprioceptive only; no scene geometry.")
    parser.add_argument("--learned-local-policy-outcome-rerank", action="store_true",
                        help="Rerank the learned policy's top primitive candidates with learned "
                             "primitive-outcome predictions before the wall guard. This uses only "
                             "runtime learned predictions, not map or standoff geometry.")
    parser.add_argument("--learned-local-post-claim-policy-outcome-rerank",
                        choices=("inherit", "on", "off"), default="inherit",
                        help="Override learned-local outcome reranking for the optional "
                             "post-claim policy checkpoint. Defaults to inheriting "
                             "--learned-local-policy-outcome-rerank.")
    parser.add_argument("--learned-local-policy-rerank-top-k", type=int, default=5,
                        help="Number of learned policy candidates considered by "
                             "--learned-local-policy-outcome-rerank.")
    parser.add_argument("--learned-local-policy-rerank-policy-weight", type=float, default=0.2)
    parser.add_argument("--learned-local-post-claim-policy-rerank-policy-weight",
                        type=float, default=None,
                        help="Optional policy-logit weight override for reranking the "
                             "post-claim learned-local policy. Defaults to inheriting "
                             "--learned-local-policy-rerank-policy-weight.")
    parser.add_argument("--learned-local-policy-rerank-blocked-weight", type=float, default=3.0)
    parser.add_argument("--learned-local-policy-rerank-clearance-weight", type=float, default=0.5)
    parser.add_argument("--learned-local-policy-rerank-progress-weight", type=float, default=1.0)
    parser.add_argument("--learned-local-policy-rerank-hard-blocked-penalty", type=float, default=2.0)
    parser.add_argument("--learned-local-policy-rerank-backward-penalty", type=float, default=0.0)
    parser.add_argument("--learned-local-policy-rerank-switch-margin", type=float, default=0.0)
    parser.add_argument("--learned-local-policy-rerank-protect-top-prob", type=float, default=0.0,
                        help="If >0, preserve a learned policy top action at or above this "
                             "probability unless learned outcome/online-map evidence marks it "
                             "unsafe. This keeps JEPA/memory in veto mode instead of letting "
                             "novelty become a planner.")
    parser.add_argument("--learned-local-policy-rerank-override-min-prob", type=float, default=0.0,
                        help="If >0, do not rerank onto a candidate below this learned-policy "
                             "probability unless the top action is unsafe.")
    parser.add_argument("--learned-local-policy-rerank-bearing-turn-threshold", type=float, default=0.4)
    parser.add_argument("--learned-local-policy-rerank-bearing-turn-bonus", type=float, default=0.4)
    parser.add_argument("--learned-local-policy-online-map-novelty-weight", type=float, default=0.0,
                        help="Optional runtime-safe novelty bonus in the learned-local reranker. "
                             "It uses only the online egomotion visit/stall/claim map.")
    parser.add_argument("--learned-local-policy-online-map-blocked-penalty", type=float, default=1.5,
                        help="Penalty applied by --learned-local-policy-online-map-novelty-weight "
                             "when a candidate projects into a self-marked blocked map cell.")
    parser.add_argument("--learned-local-policy-online-map-turn-scale", type=float, default=0.2,
                        help="Scale applied to online-map novelty for in-place yaw candidates. "
                             "Translating primitives keep full novelty credit.")
    parser.add_argument("--learned-local-policy-online-map-claim-repulsion-weight",
                        type=float, default=0.0,
                        help="Runtime-safe novelty bonus for moving farther from cells where "
                             "targets were already claimed.")
    parser.add_argument("--learned-local-policy-online-map-frontier-route-weight",
                        type=float, default=0.0,
                        help="Runtime-safe novelty bonus for reducing BFS distance through "
                             "self-visited cells to an unknown-neighbor frontier.")
    parser.add_argument("--learned-local-policy-online-map-hard-veto", action="store_true",
                        help="Treat self-marked online-map blocked cells/edges as hard unsafe "
                             "evidence inside learned-local reranking.")
    parser.add_argument("--learned-local-policy-online-map-novelty-states", default="EXPLORE",
                        help="Comma-separated controller states where the online-map novelty "
                             "bonus may affect learned-local reranking. Empty means all states.")
    parser.add_argument("--learned-local-policy-translation-pressure-after", type=int, default=0,
                        help="If >0, break learned-policy turn-only runs after this many "
                             "consecutive yaw ticks by selecting a translating primitive from "
                             "learned primitive-outcome predictions.")
    parser.add_argument("--learned-local-policy-translation-pressure-max-blocked-prob",
                        type=float, default=1.01,
                        help="Maximum learned blocked/clearance probability accepted by "
                             "--learned-local-policy-translation-pressure-after.")
    parser.add_argument("--learned-local-policy-translation-pressure-min-progress-m",
                        type=float, default=0.02,
                        help="Minimum learned progress for translation-pressure candidates.")
    parser.add_argument("--learned-local-policy-translation-pressure-primitives",
                        default="forward_medium,arc_left,arc_right,forward_fast,backward",
                        help="Comma-separated candidate primitives used by learned-policy "
                             "translation pressure.")
    parser.add_argument("--learned-local-policy-translation-pressure-states", default="",
                        help="Comma-separated controller states where translation pressure "
                             "may override learned-local turn-only runs. Empty means all "
                             "states for backward compatibility.")
    parser.add_argument("--learned-local-policy-frontier-pressure-after", type=int, default=0,
                        help="If >0, after this many consecutive learned-policy yaw ticks, "
                             "allow the runtime-safe online egomotion map to pick a "
                             "translating primitive that routes through self-visited cells "
                             "toward a non-current frontier. This uses no scene geometry.")
    parser.add_argument("--learned-local-policy-frontier-pressure-probe-route-steps",
                        action="store_true",
                        help="Accept a never-attempted frontier route step even when the "
                             "learned heads predict it blocked/no-progress, so contact "
                             "resolves the uncertainty once and the online map remembers. "
                             "Contact-tolerant demos only.")
    parser.add_argument("--learned-local-policy-frontier-pressure-always",
                        action="store_true",
                        help="Run frontier pressure on every eligible EXPLORE tick instead "
                             "of only after yaw/nonprogress runs. Without this, any "
                             "translation proposal resets frontier authority for a tick and "
                             "a guard conversion to the opposite yaw can lock the robot in "
                             "a two-tick align/convert cycle.")
    parser.add_argument("--learned-local-policy-frontier-pressure-pre-claim",
                        action="store_true",
                        help="Arm frontier pressure from tick 0 instead of only after the "
                             "first claim. Without this, scenes where the first target is "
                             "not found by local wandering get no exploration drive at all. "
                             "Uses only the online egomotion map and learned predictions.")
    parser.add_argument("--learned-local-policy-frontier-pressure-states", default="",
                        help="Comma-separated controller states where frontier-pressure "
                             "overrides may run. Empty preserves the legacy all-state "
                             "behavior.")
    parser.add_argument("--learned-local-policy-frontier-pressure-max-blocked-prob",
                        type=float, default=1.01,
                        help="Maximum learned blocked/clearance probability accepted by "
                             "--learned-local-policy-frontier-pressure-after.")
    parser.add_argument("--learned-local-policy-frontier-pressure-min-progress-m",
                        type=float, default=0.0,
                        help="Minimum learned progress for online-map frontier-pressure "
                             "candidates.")
    parser.add_argument("--learned-local-policy-frontier-pressure-min-route-cells",
                        type=int, default=1,
                        help="Minimum self-visited graph distance for the target frontier. "
                             "This prevents the fallback from repeatedly treating the "
                             "current cell as the frontier during turn loops.")
    parser.add_argument("--learned-local-policy-frontier-pressure-guard-blocked-penalty",
                        type=float, default=0.18,
                        help="Score penalty for frontier-pressure candidates whose target "
                             "cell or edge is marked blocked by the learned wall/action "
                             "guard. This uses only runtime learned-outcome memory.")
    parser.add_argument("--learned-local-policy-frontier-pressure-nonroute-backward-claim-escape",
                        action="store_true",
                        help="Allow backward frontier-pressure candidates after a claim "
                             "when they increase distance from the claimed cell even if "
                             "they are not the current route step. Uses only online "
                             "egomotion claim memory and learned outcome scores.")
    parser.add_argument("--learned-local-policy-frontier-pressure-prefer-unguarded",
                        action="store_true",
                        help="When any accepted frontier-pressure candidate is not marked "
                             "blocked by the learned wall/action guard, prefer the best "
                             "unguarded candidate over guard-blocked candidates. Uses only "
                             "runtime learned-outcome memory.")
    parser.add_argument("--learned-local-policy-frontier-pressure-map-blocked-backward-claim-escape",
                        action="store_true",
                        help="Allow backward claim-escape frontier candidates through "
                             "self-map blocked cells/edges when the learned wall/action "
                             "guard does not mark the edge blocked. This is a diagnostic "
                             "for stale online egomotion blocking after a claim.")
    parser.add_argument("--learned-local-policy-frontier-pressure-guarded-retry-after-noops",
                        type=int, default=0,
                        help="After this many consecutive frontier-pressure noops, allow a "
                             "candidate blocked only by the runtime learned/egomotion guard "
                             "to be retried without guard-commit force. The learned action "
                             "and body-clearance guard still screens the primitive.")
    parser.add_argument("--learned-local-policy-frontier-pressure-combined-blocked-retry-after-noops",
                        type=int, default=0,
                        help="After this many consecutive frontier-pressure noops, allow a "
                             "translation candidate blocked by both the runtime egomotion map "
                             "and learned guard memory to be retried without guard-commit "
                             "force. The learned action and body-clearance guard still screens "
                             "the primitive.")
    parser.add_argument("--learned-local-policy-frontier-pressure-commit",
                        action="store_true",
                        help="When frontier pressure selects a translating primitive, commit "
                             "that primitive through the learned action guard for this tick. "
                             "The frontier selection still uses only online egomotion memory "
                             "and learned primitive-outcome predictions.")
    parser.add_argument("--learned-local-policy-frontier-pressure-guard-rerank-on-commit",
                        action="store_true",
                        help="When --learned-local-policy-frontier-pressure-commit is active, "
                             "treat the frontier primitive as the requested command while "
                             "still letting the learned action/body-clearance guard rank all "
                             "runtime-available primitive candidates. This avoids turning "
                             "frontier pressure into a single-candidate force.")
    parser.add_argument("--learned-local-policy-frontier-pressure-guard-recovery-rerank-on-commit",
                        action="store_true",
                        help="When --learned-local-policy-frontier-pressure-commit is active, "
                             "treat the frontier primitive as the requested command and let "
                             "the learned guard compare it only against a small recovery set. "
                             "This preserves frontier drive while allowing learned safety "
                             "fallbacks.")
    parser.add_argument("--learned-local-policy-frontier-pressure-guard-recovery-primitives",
                        default="yaw_left,yaw_right,backward",
                        help="Comma-separated recovery primitives exposed by "
                             "--learned-local-policy-frontier-pressure-guard-recovery-rerank-on-commit.")
    parser.add_argument("--learned-topology-route-table", type=Path, default=None,
                        help="JSON route/topology table learned offline from teacher traces. "
                             "Runtime uses only stored waypoints plus odometry; it does not "
                             "run standoff planning or free-cell graph search.")
    parser.add_argument("--learned-topology-route-advance-m", type=float, default=0.38)
    parser.add_argument("--learned-topology-route-lookahead-m", type=float, default=0.0,
                        help="Additional route distance to steer toward after the committed "
                             "--learned-topology-route-advance-m index. This smooths dense "
                             "teacher traces without changing route progress accounting.")
    parser.add_argument("--learned-topology-route-reproject-window", type=int, default=0,
                        help="If positive, allow a route-memory follower to jump only forward "
                             "to the nearest waypoint within this many future route indices "
                             "when physical drift leaves the committed waypoint behind.")
    parser.add_argument("--learned-topology-route-reproject-trigger-m", type=float, default=0.0,
                        help="Minimum distance from the committed waypoint before bounded "
                             "forward re-projection is allowed. Values <= 0 allow projection "
                             "whenever it improves distance.")
    parser.add_argument("--learned-topology-route-until-area-logit", type=float, default=None,
                        help="If set, keep following the learned topology route while the "
                             "active target is remembered but not visually large enough "
                             "for a stable SERVO handoff.")
    parser.add_argument("--learned-topology-route-release-on-seen-area-logit",
                        type=float, default=None,
                        help="If set, release route-memory control once the active target "
                             "is visually seen at or above this area logit, even when it "
                             "is not yet in the tight in-cone gate.")
    parser.add_argument("--learned-topology-route-yaw-threshold", type=float, default=0.50)
    parser.add_argument("--learned-topology-route-yaw-threshold-by-color", default="",
                        help="Optional comma-separated color:threshold overrides for "
                             "--learned-topology-route-yaw-threshold.")
    parser.add_argument("--learned-topology-route-forward-threshold", type=float, default=0.12)
    parser.add_argument("--learned-topology-route-arc-max-bearing", type=float, default=0.35)
    parser.add_argument("--learned-topology-route-arc-max-bearing-by-color", default="",
                        help="Optional comma-separated color:bearing overrides for "
                             "--learned-topology-route-arc-max-bearing.")
    parser.add_argument("--learned-topology-route-use-stored-primitives", action="store_true",
                        help="Use primitive labels stored in the route table instead of deriving "
                             "route actions from waypoint bearing.")
    parser.add_argument("--learned-topology-route-geometry-veto-min-clearance-m",
                        type=float, default=None,
                        help="If set, route-memory ticks run an explicit body-footprint "
                             "primitive clearance veto after learned-action selection.")
    parser.add_argument("--learned-topology-route-geometry-veto-feasible-threshold",
                        type=float, default=1.0)
    parser.add_argument("--learned-topology-route-geometry-veto-selected-primitives",
                        default="forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right",
                        help="Comma-separated primitives eligible for the route geometry veto.")
    parser.add_argument("--learned-topology-route-geometry-veto-replacements",
                        default="forward_slow,arc_left,arc_right,yaw_left,yaw_right,backward,hold",
                        help="Comma-separated replacement primitives considered by the route geometry veto.")
    parser.add_argument("--body-clearance-geometry-veto-min-clearance-m",
                        type=float, default=None,
                        help="If set, all learned-action guard ticks run an explicit "
                             "body-footprint primitive clearance veto after learned-action "
                             "selection. This is route-free and uses the same runtime "
                             "occupancy/body projection as the wall guard.")
    parser.add_argument("--body-clearance-geometry-veto-feasible-threshold",
                        type=float, default=1.0)
    parser.add_argument("--body-clearance-geometry-veto-states", default="",
                        help="Comma-separated controller states where the generic geometry "
                             "veto may run. Empty means all states.")
    parser.add_argument("--body-clearance-geometry-veto-min-claimed-count",
                        type=int, default=0,
                        help="Minimum number of already claimed beacons before the "
                             "generic geometry veto may run.")
    parser.add_argument("--body-clearance-geometry-veto-target-colors",
                        default="",
                        help="Optional comma-separated active target colors where the "
                             "generic geometry veto may run. Empty means all target colors.")
    parser.add_argument("--body-clearance-geometry-veto-allow-force-single-candidate",
                        action="store_true",
                        help="Allow the generic body-clearance geometry veto to override "
                             "force-single-candidate learned guard decisions. This is intended "
                             "for body-safety vetoes on route-replay or escape context.")
    parser.add_argument("--body-clearance-geometry-veto-allow-guard-disabled",
                        action="store_true",
                        help="Allow the generic body-clearance geometry veto to run even "
                             "when the learned guard is disabled by state/body-clearance "
                             "locks, while the wall-aware planner itself is enabled.")
    parser.add_argument("--body-clearance-geometry-veto-selected-primitives",
                        default="forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right,backward",
                        help="Comma-separated primitives eligible for the generic geometry veto.")
    parser.add_argument("--body-clearance-geometry-veto-replacements",
                        default="forward_slow,arc_left,arc_right,yaw_left,yaw_right,backward,hold",
                        help="Comma-separated replacement primitives considered by the generic geometry veto.")
    parser.add_argument("--body-clearance-geometry-veto-blocked-fallback-primitives",
                        default="",
                        help="Optional ordered primitives to use when every generic "
                             "geometry-veto candidate is projected blocked. Empty keeps "
                             "the normal best-scored blocked candidate.")
    parser.add_argument("--body-clearance-geometry-veto-override-replacements",
                        default="",
                        help="Optional comma-separated replacement primitives that replace "
                             "--body-clearance-geometry-veto-replacements once "
                             "--body-clearance-geometry-veto-override-min-claimed-count "
                             "is satisfied.")
    parser.add_argument("--body-clearance-geometry-veto-override-min-claimed-count",
                        type=int, default=0,
                        help="Minimum number of already claimed beacons before the optional "
                             "generic geometry-veto replacement override activates.")
    parser.add_argument("--max-ticks", type=int, default=120)
    parser.add_argument("--command-dt-s", type=float, default=0.10)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--seen-conf", type=float, default=0.3)
    parser.add_argument("--mask-sigma", type=float, default=None,
                        help="Override the color-mask sigma at inference to tolerate the "
                             "closed-loop render's desaturated colors (training used 0.20).")
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--mask-area-threshold", type=float, default=None)
    parser.add_argument("--claim-area-logit", type=float, default=3.0)
    parser.add_argument("--claim-bearing", type=float, default=0.25)
    parser.add_argument("--claim-min-seen-ticks", type=int, default=0,
                        help="Minimum closed-loop ticks since first target evidence before "
                             "a CLAIM may terminate the episode. This prevents one-frame, "
                             "distant detections from becoming premature claims.")
    parser.add_argument("--claim-near-area-logit", type=float, default=None,
                        help="Optional perception-only near-target CLAIM shortcut. When set, "
                             "a large centered target blob may claim after "
                             "--claim-near-min-seen-ticks even if --claim-min-seen-ticks has "
                             "not matured yet. This avoids pushing the physical robot into "
                             "the target/wall once visual proximity is already unambiguous.")
    parser.add_argument("--claim-near-area-logit-by-color", default="",
                        help="Optional comma-separated color:area-logit overrides for "
                             "--claim-near-area-logit. This lets corner beacons stop earlier "
                             "without making every target use the same loose threshold.")
    parser.add_argument("--claim-near-bearing", type=float, default=0.25,
                        help="Bearing threshold for --claim-near-area-logit.")
    parser.add_argument("--claim-near-bearing-by-color", default="",
                        help="Optional comma-separated color:bearing overrides for "
                             "--claim-near-bearing. This lets corner-specific targets use "
                             "a wider visual standoff claim without loosening every beacon.")
    parser.add_argument("--claim-near-min-seen-ticks", type=int, default=8,
                        help="Minimum seen age for --claim-near-area-logit.")
    parser.add_argument("--claim-near-min-seen-ticks-by-color", default="",
                        help="Optional comma-separated color:ticks overrides for "
                             "--claim-near-min-seen-ticks.")
    parser.add_argument("--claim-success-proxy-area-logit", type=float, default=None,
                        help="Optional learned RGB area-logit proxy required before any "
                             "visual CLAIM is accepted. This is a runtime-safe learned "
                             "distance/validity gate for keeping all-color claims inside "
                             "--success-dist-m without using privileged target distance.")
    parser.add_argument("--claim-success-proxy-area-logit-by-color", default="",
                        help="Optional comma-separated color:area-logit overrides for "
                             "--claim-success-proxy-area-logit.")
    parser.add_argument("--claim-success-proxy-bearing", type=float, default=None,
                        help="Optional learned RGB bearing proxy required before any "
                             "visual CLAIM is accepted.")
    parser.add_argument("--claim-success-proxy-bearing-by-color", default="",
                        help="Optional comma-separated color:bearing overrides for "
                             "--claim-success-proxy-bearing.")
    parser.add_argument("--claim-success-model-checkpoint", type=Path, default=None,
                        help="Optional learned claim-valid/distance classifier checkpoint. "
                             "The model consumes only runtime RGB/readout, active color, "
                             "claim memory, and controller-state timing scalars.")
    parser.add_argument("--claim-success-model-threshold", type=float, default=None,
                        help="Probability threshold for --claim-success-model-checkpoint. "
                             "Defaults to the threshold saved in the checkpoint.")
    parser.add_argument("--claim-success-model-threshold-by-color", default="",
                        help="Optional comma-separated color:threshold overrides for "
                             "--claim-success-model-threshold.")
    parser.add_argument("--claim-success-model-positive-trigger", action="store_true",
                        help="Allow the learned claim-valid/distance classifier to trigger "
                             "CLAIM directly once runtime perception says the active target "
                             "is visible, instead of using only hand-set RGB area gates.")
    parser.add_argument("--claim-success-model-trigger-min-seen-ticks", type=int, default=None,
                        help="Minimum visible age for --claim-success-model-positive-trigger. "
                             "Defaults to the active color's near-claim min-seen setting.")
    parser.add_argument("--claim-contact-area-logit", type=float, default=None,
                        help="Optional high-confidence visual standoff claim. If the target "
                             "blob is very large, allow a wider bearing than --claim-bearing "
                             "so physical beacon/corner contact does not become indefinite "
                             "wall-walking while trying to over-center the camera view.")
    parser.add_argument("--claim-contact-bearing", type=float, default=0.5,
                        help="Bearing threshold for --claim-contact-area-logit.")
    parser.add_argument("--claim-contact-min-seen-ticks", type=int, default=8,
                        help="Minimum seen age for --claim-contact-area-logit.")
    parser.add_argument("--multi-target-success-requires-claim-distance", action="store_true",
                        help="For multi-target runs, require every accepted claim's "
                             "diagnostic privileged distance to be within "
                             "--success-dist-m before setting result.success. By "
                             "default multi-target success follows accepted learned "
                             "claim gates plus physical safety; claim distances remain "
                             "reported as diagnostics.")
    parser.add_argument("--claim-stalled-area-logit", type=float, default=None,
                        help="Optional visual-stall claim. If a previous near-target "
                             "forward step stalled with high learned clearance risk, allow "
                             "a centered visible target above this area logit to claim on "
                             "the next tick. This stops wall-walking at target corners "
                             "without using privileged distance.")
    parser.add_argument("--claim-stalled-bearing", type=float, default=0.35,
                        help="Bearing threshold for --claim-stalled-area-logit and for "
                             "arming its previous-tick visual stall latch.")
    parser.add_argument("--claim-stalled-min-seen-ticks", type=int, default=8,
                        help="Minimum seen age for --claim-stalled-area-logit.")
    parser.add_argument("--claim-stalled-clearance-prob", type=float, default=0.9,
                        help="Selected learned clearance-blocked probability required to "
                             "arm the visual-stall claim latch after a forward stall. Set "
                             "below zero to arm from centered visual stall alone when a "
                             "clearance head is not active.")
    parser.add_argument("--claim-stalled-latch-ticks", type=int, default=2,
                        help="Number of ticks the visual-stall claim latch remains armed.")
    parser.add_argument("--success-dist-m", type=float, default=0.8)
    parser.add_argument("--wall-aware-planner", action="store_true",
                        help="Opt-in privileged-grid local guard: predicts each primitive's "
                             "near-field clearance and vetoes blocked forward/arc commands. "
                             "This is a planning scaffold, not a deployment-valid latent model.")
    parser.add_argument("--wall-decision-source", choices=("privileged_grid", "learned_front", "learned_action"),
                        default="privileged_grid",
                        help="Source for wall-aware veto decisions. learned_front uses a frozen "
                             "JEPA/RGB front-blocked head; learned_action ranks candidate "
                             "primitives with a JEPA/RGB action-conditioned outcome head. Neither "
                             "learned mode uses the manifest grid for runtime wall decisions.")
    parser.add_argument("--front-blocked-checkpoint", type=Path, default=None,
                        help="Checkpoint from train_go2_jepa_front_blocked_predictor.py "
                             "(required for --wall-decision-source learned_front).")
    parser.add_argument("--front-blocked-threshold", type=float, default=None,
                        help="Override the learned front-blocked probability threshold.")
    parser.add_argument("--primitive-outcome-checkpoint", type=Path, default=None,
                        help="Checkpoint from train_go2_jepa_primitive_outcome_predictor.py "
                             "(required for --wall-decision-source learned_action).")
    parser.add_argument("--primitive-outcome-frozen-jepa-checkpoint", type=Path, default=None,
                        help="Optional encoder override for the primitive-outcome head only. "
                             "Lets the guard heads run on a geometry-retrained frozen JEPA "
                             "encoder while the memory controller keeps its own encoder. "
                             "Falls back to --frozen-jepa-checkpoint, then to the head "
                             "checkpoint's recorded encoder.")
    parser.add_argument("--primitive-clearance-frozen-jepa-checkpoint", type=Path, default=None,
                        help="Optional encoder override for the body-clearance head only; "
                             "same precedence as --primitive-outcome-frozen-jepa-checkpoint.")
    parser.add_argument("--primitive-aux-clearance-frozen-jepa-checkpoint", type=Path, default=None,
                        help="Optional encoder override for the auxiliary body-clearance head. "
                             "Falls back to --primitive-clearance-frozen-jepa-checkpoint, "
                             "--frozen-jepa-checkpoint, then the auxiliary checkpoint's "
                             "recorded encoder.")
    parser.add_argument("--primitive-post-claim-outcome-checkpoint", type=Path, default=None,
                        help="Optional learned primitive-outcome checkpoint used after at "
                             "least one target claim when the controller state is enabled "
                             "for the post-claim learned-local policy. Defaults to the "
                             "primary --primitive-outcome-checkpoint.")
    parser.add_argument("--primitive-outcome-threshold", type=float, default=None,
                        help="Override the learned-action blocked probability threshold.")
    parser.add_argument("--primitive-clearance-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from train_go2_jepa_primitive_outcome_predictor.py "
                             "trained with --label-mode counterfactual_body_clearance. Its "
                             "learned swept-body blocked probability is fused with the normal "
                             "action-outcome probability, keeping runtime wall decisions "
                             "nonprivileged.")
    parser.add_argument("--primitive-aux-clearance-checkpoint", type=Path, default=None,
                        help="Optional second learned clearance checkpoint used only for "
                             "primary-low/aux-high disagreement vetoes. It is not fused into "
                             "normal scoring, so the primary clearance head remains the global "
                             "calibration source.")
    parser.add_argument("--primitive-clearance-threshold", type=float, default=None,
                        help="Override threshold for the learned swept-body clearance checkpoint. "
                             "When provided, the learned-action blocked threshold becomes the "
                             "minimum of the outcome and clearance thresholds.")
    parser.add_argument("--primitive-outcome-blocked-weight", type=float, default=1.2)
    parser.add_argument("--primitive-outcome-progress-weight", type=float, default=3.0)
    parser.add_argument("--primitive-outcome-requested-bonus", type=float, default=0.18,
                        help="Score bonus for preserving the upstream planner's requested primitive.")
    parser.add_argument("--primitive-outcome-turn-progress-scale", type=float, default=0.0,
                        help="Scale applied to learned progress for yaw/hold candidates. The action "
                             "head predicts executed displacement; pure yaw should not receive full "
                             "translational planning credit.")
    parser.add_argument("--primitive-outcome-switch-margin", type=float, default=0.12,
                        help="Minimum score improvement required before learned-action ranking "
                             "may replace the upstream planner's primitive.")
    parser.add_argument("--primitive-outcome-forward-progress-floor", type=float, default=None,
                        help="Optional nonprivileged learned-progress gate. Forward/arc "
                             "candidates whose predicted progress is below this value receive "
                             "--primitive-outcome-forward-progress-penalty and are logged as "
                             "low-progress blocked candidates.")
    parser.add_argument("--primitive-outcome-forward-progress-floor-states", default="",
                        help="Optional comma-separated guard states where "
                             "--primitive-outcome-forward-progress-floor is active. Empty "
                             "keeps historical behavior and applies the floor in every state.")
    parser.add_argument("--primitive-outcome-forward-progress-penalty", type=float, default=0.0,
                        help="Score penalty applied when --primitive-outcome-forward-progress-floor "
                             "is active and a translating forward candidate predicts too little "
                             "progress.")
    parser.add_argument("--primitive-outcome-low-progress-hard-veto", action="store_true",
                        help="When the learned action-outcome head predicts that the selected "
                             "translating primitive will make below-floor progress, force a "
                             "fallback primitive instead of allowing the requested-command bonus "
                             "to keep driving toward a wall. This uses only learned outcomes.")
    parser.add_argument("--primitive-outcome-low-progress-hard-veto-primitives",
                        default="yaw_left,yaw_right,backward,hold",
                        help="Comma-separated fallback primitives eligible for "
                             "--primitive-outcome-low-progress-hard-veto.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto", action="store_true",
                        help="When the learned action-outcome head predicts that the selected "
                             "forward primitive is blocked, force a non-forward fallback "
                             "primitive instead of letting the score/requested bonus keep it.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto-after-first-claim",
                        action="store_true",
                        help="Enable --primitive-outcome-blocked-hard-veto only after the first "
                             "beacon claim. This keeps target acquisition behavior unchanged "
                             "while allowing stronger learned safety during post-claim exploration.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto-primitives",
                        default="yaw_left,yaw_right,backward",
                        help="Comma-separated fallback primitives eligible for "
                             "--primitive-outcome-blocked-hard-veto.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto-selected-primitives",
                        default="",
                        help="Optional comma-separated selected primitives that "
                             "--primitive-outcome-blocked-hard-veto may replace. "
                             "Empty preserves the historical behavior for all "
                             "forward/arc primitives.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto-max-abs-bearing",
                        type=float, default=None,
                        help="Optional visual-bearing gate for "
                             "--primitive-outcome-blocked-hard-veto. When set, the "
                             "blocked fallback only fires if abs(active target "
                             "bearing) is below this value. This is useful for "
                             "suppressing shallow blocked arcs without turning the "
                             "learned action head into a global brake.")
    parser.add_argument("--primitive-outcome-blocked-hard-veto-use-guard-bearing",
                        action="store_true",
                        help="Use the current guard/route bearing, rather than the visual "
                             "target bearing, for --primitive-outcome-blocked-hard-veto "
                             "bearing gates.")
    parser.add_argument("--primitive-outcome-progress-floor-min-blocked-prob", type=float, default=None,
                        help="Optional learned blocked-probability floor for the forward-progress "
                             "gate. When set, low learned progress only vetoes a forward candidate "
                             "if the same learned action-outcome head also predicts at least this "
                             "blocked probability. This avoids treating ordinary low-displacement "
                             "gait/noise predictions as wall contact.")
    parser.add_argument("--primitive-outcome-progress-floor-force-below", type=float, default=None,
                        help="Optional learned progress threshold below which the forward-progress "
                             "gate ignores --primitive-outcome-progress-floor-min-blocked-prob. This "
                             "catches near-zero-progress body/corner stalls even when the learned "
                             "blocked-probability channel is falsely low.")
    parser.add_argument("--primitive-outcome-progress-floor-prefer-yaw", action="store_true",
                        help="When a requested forward primitive is vetoed only by learned low "
                             "progress, remove backward from the normal candidate set so the "
                             "selector reorients instead of backing along a wall. Explicit "
                             "escape/recovery blocks can still command backward.")
    parser.add_argument("--primitive-outcome-preserve-turn-requests", action="store_true",
                        help="For learned-action wall decisions, do not upgrade an upstream "
                             "pure-yaw request into a translating arc. The learned outcome "
                             "head still scores/logs candidates and still vetoes unsafe "
                             "forward requests, but this prevents shoulder-sweeping arcs "
                             "around narrow corners.")
    parser.add_argument("--primitive-outcome-preserve-turn-states", default="",
                        help="Comma-separated controller states where learned-action wall "
                             "decisions preserve upstream pure-yaw requests. This gives "
                             "EXPLORE waypoint turns body clearance without forcing the same "
                             "behavior in target SEEK/SERVO.")
    parser.add_argument("--primitive-outcome-preserve-turn-until-first-claim",
                        action="store_true",
                        help="Apply turn-request preservation only until the first beacon claim. "
                             "After a claim, the learned action guard may replace yaw requests "
                             "with translating primitives based on learned outcome scores.")
    parser.add_argument("--primitive-outcome-preserve-arc-requests", action="store_true",
                        help="For learned-action wall decisions, do not upgrade an upstream "
                             "arc request into a straight forward primitive. Blocked-hard "
                             "vetoes may still replace the arc with yaw/backward recovery.")
    parser.add_argument("--primitive-outcome-turn-body-rerank-primitives", default="",
                        help="When --primitive-outcome-preserve-turn-requests is active "
                             "and the requested pure-yaw primitive has learned body-clearance "
                             "risk, only these comma-separated primitives may replace the yaw "
                             "request. Empty preserves the previous behavior of allowing any "
                             "learned body-clearance rerank.")
    parser.add_argument("--primitive-outcome-preserve-straight-states", default="",
                        help="Comma-separated controller states where learned-action wall "
                             "decisions may veto straight-forward requests to yaw/backward "
                             "but may not upgrade them into arc primitives. This preserves "
                             "body-clearance route followers that already turn in place.")
    parser.add_argument("--primitive-outcome-preserve-backward-requests", action="store_true",
                        help="For learned-action wall decisions, preserve an upstream "
                             "backward request instead of upgrading it into a progress-scoring "
                             "arc or forward primitive. This keeps body-clearance escape "
                             "requests authoritative while retaining learned wall scoring.")
    parser.add_argument("--primitive-outcome-preserve-backward-clearance-margin",
                        type=float, default=None,
                        help="When preserving backward requests, allow a learned rerank "
                             "only if its swept-body blocked probability is lower than "
                             "backward by at least this margin. Unset keeps historical "
                             "behavior: backward is preserved only when it has no learned "
                             "body-clearance penalty.")
    parser.add_argument("--body-clearance-target-servo", action="store_true",
                        help="When a target is visually near, turn in place to reduce the "
                             "bearing error and use a slower forward primitive. This keeps "
                             "the Go2's body from sweeping into inside corners during final "
                             "beacon pursuit without using privileged geometry.")
    parser.add_argument("--body-clearance-target-area-logit", type=float, default=2.0,
                        help="RGB target-area logit at which target-pursuit body-clearance "
                             "behavior activates.")
    parser.add_argument("--body-clearance-target-bearing", type=float, default=0.12,
                        help="Near-target bearing error above which target-pursuit body "
                             "clearance turns in place instead of arcing.")
    parser.add_argument("--body-clearance-target-forward-primitive", default="forward_slow",
                        help="Forward primitive used by --body-clearance-target-servo once "
                             "the near target is centered.")
    parser.add_argument("--body-clearance-latch-ticks", type=int, default=4,
                        help="Keep near-target body-clearance behavior active for this many "
                             "ticks after it first triggers. This prevents transient target "
                             "dropouts near a corner from falling back to faster SEEK/SERVO "
                             "forward commands.")
    parser.add_argument("--body-clearance-learned-prob-floor", type=float, default=0.35,
                        help="Learned blocked probability above which translating primitives "
                             "receive an extra body-clearance score penalty.")
    parser.add_argument("--body-clearance-learned-prob-weight", type=float, default=1.0,
                        help="Score penalty weight for learned body-clearance blocked "
                             "probability excess.")
    parser.add_argument("--body-clearance-near-forward-prob-floor", type=float, default=None,
                        help="Optional lower learned body-clearance probability floor for "
                             "translating primitives once the target is visually near. "
                             "This can protect final corner approach without making open "
                             "exploration over-conservative.")
    parser.add_argument("--body-clearance-near-forward-prob-weight", type=float, default=None,
                        help="Optional score penalty weight for "
                             "--body-clearance-near-forward-prob-floor. Defaults to "
                             "--body-clearance-learned-prob-weight when omitted.")
    parser.add_argument("--body-clearance-learned-min-area-logit", type=float, default=None,
                        help="Optional target-area logit required before learned "
                             "body-clearance penalties are applied. This keeps the "
                             "normal learned action-outcome planner in charge of open "
                             "exploration while still enabling body-aware corner "
                             "approach near a visible beacon.")
    parser.add_argument("--body-clearance-near-arc-penalty", type=float, default=0.65,
                        help="Extra learned-action score penalty for arc primitives while "
                             "a target is visually near. This favors yaw-then-forward over "
                             "flank-sweeping arcs at tight beacon corners.")
    parser.add_argument("--body-clearance-near-yaw-prob-floor", type=float, default=None,
                        help="Optional learned body-clearance probability floor for pure "
                             "yaw primitives while a target is visually near. Defaults to "
                             "--body-clearance-learned-prob-floor when omitted.")
    parser.add_argument("--body-clearance-near-yaw-prob-weight", type=float, default=0.0,
                        help="Extra learned-action score penalty weight for pure-yaw "
                             "primitives while a target is visually near, using the "
                             "optional learned swept-body clearance probability.")
    parser.add_argument("--body-clearance-yaw-always", action="store_true",
                        help="Apply the learned swept-body yaw probability penalty outside "
                             "near-target pursuit too. This lets the nonprivileged clearance "
                             "head discourage in-place turns that would sweep a shoulder "
                             "into a narrow corner during exploration.")
    parser.add_argument("--body-clearance-hard-veto-prob", type=float, default=1.01,
                        help="If the selected translating primitive has at least this "
                             "learned swept-body blocked probability, replace it with a "
                             "materially safer fallback primitive. Values above 1 disable "
                             "the hard veto. This uses only the learned clearance head.")
    parser.add_argument("--body-clearance-hard-veto-margin", type=float, default=0.05,
                        help="Minimum learned clearance probability improvement required "
                             "before --body-clearance-hard-veto-prob may replace a selected "
                             "translating primitive.")
    parser.add_argument("--body-clearance-hard-veto-replacement-cap", type=float, default=1.01,
                        help="Maximum learned clearance blocked probability allowed for a "
                             "replacement primitive under --body-clearance-hard-veto-prob. "
                             "If no fallback is at or below this cap, the originally "
                             "selected translating primitive executes instead of being "
                             "swapped for a marginally-less-blocked non-translating one. "
                             "Values above 1 disable the cap, preserving historical "
                             "behavior. This uses only the learned clearance head.")
    parser.add_argument("--body-clearance-target-area-hard-veto-prob", type=float, default=1.01,
                        help="Optional lower learned swept-body hard-veto threshold used "
                             "only when the active target is visually large. Values above "
                             "1 disable. This keeps near-beacon shoulder/yaw cleanup local "
                             "to target approach while still using only learned clearance "
                             "predictions and RGB target area.")
    parser.add_argument("--body-clearance-target-area-hard-veto-min-area-logit", type=float, default=None,
                        help="Minimum target area logit for "
                             "--body-clearance-target-area-hard-veto-prob. Defaults to "
                             "--body-clearance-target-area-logit.")
    parser.add_argument("--body-clearance-hard-veto-primitives", default="yaw_left,yaw_right,backward,hold",
                        help="Comma-separated fallback primitives eligible for "
                             "--body-clearance-hard-veto-prob.")
    parser.add_argument("--body-clearance-hard-veto-selected-primitives", default="",
                        help="Comma-separated selected primitives eligible for "
                             "--body-clearance-hard-veto-prob. Empty means all "
                             "translating primitives, preserving historical behavior.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-after",
                        type=int, default=0,
                        help="If >0, after this many consecutive learned body-clearance "
                             "hard-veto replacements to hold, allow a learned-clearance "
                             "ranked non-hold recovery primitive. This is route-free and "
                             "uses only the same learned primitive-outcome candidates.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-max-clearance-prob",
                        type=float, default=0.70,
                        help="Maximum learned swept-body blocked probability allowed for "
                             "--body-clearance-hard-veto-hold-escape-after recovery "
                             "candidates.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-primitives",
                        default="backward,yaw_left,yaw_right",
                        help="Comma-separated non-hold primitives eligible for bounded "
                             "hard-veto hold escape.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-states",
                        default="EXPLORE",
                        help="Comma-separated controller states where bounded hard-veto "
                             "hold escape may run. Empty means all states.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-override-primitives",
                        default="",
                        help="Optional comma-separated non-hold primitives that replace "
                             "--body-clearance-hard-veto-hold-escape-primitives once "
                             "--body-clearance-hard-veto-hold-escape-override-min-claimed-count "
                             "and state gates are satisfied.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-override-states",
                        default="",
                        help="Comma-separated states for the hold-escape override. Empty "
                             "means all states once the claimed-count gate passes.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-override-min-claimed-count",
                        type=int, default=0,
                        help="Minimum number of already claimed beacons before the optional "
                             "hold-escape override primitive set may replace the base set.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-override-max-clearance-prob",
                        type=float, default=None,
                        help="Optional max learned swept-body blocked probability for the "
                             "hold-escape override. Defaults to the base hold-escape cap.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-override-min-current-clearance-m",
                        type=float, default=None,
                        help="Optional minimum current swept-body clearance required before "
                             "the hold-escape override may replace the base primitive set.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-min-projected-clearance-m",
                        type=float, default=None,
                        help="If set, hard-veto hold-escape candidates must have at least "
                             "this projected swept-body clearance under the geometric "
                             "primitive rollout before they may replace hold.")
    parser.add_argument("--body-clearance-hard-veto-hold-escape-min-projected-improvement-m",
                        type=float, default=0.0,
                        help="Minimum projected swept-body clearance improvement required "
                             "for non-hold hard-veto hold-escape candidates. Requires a "
                             "current body-clearance probe.")
    parser.add_argument("--body-clearance-aux-switch-hard-veto-primitives", default="",
                        help="Optional comma-separated fallback primitives used for "
                             "--body-clearance-hard-veto-prob only while the learned "
                             "auxiliary clearance switch is active. Empty keeps "
                             "--body-clearance-hard-veto-primitives for both primary "
                             "and auxiliary-switch guard decisions.")
    parser.add_argument("--body-clearance-veto-min-claimed-count", type=int, default=0,
                        help="Minimum number of already claimed beacons before learned "
                             "body-clearance veto hooks may replace the selected action. "
                             "This gates only veto-style interventions; learned clearance "
                             "scoring remains available.")
    parser.add_argument("--body-clearance-aux-veto-prob", type=float, default=1.01,
                        help="If the auxiliary learned clearance head predicts the selected "
                             "primitive at least this risky while the primary learned clearance "
                             "head is below --body-clearance-aux-veto-primary-max-prob, replace "
                             "with an auxiliary-safer fallback. Values above 1 disable.")
    parser.add_argument("--body-clearance-aux-veto-primary-max-prob", type=float, default=0.65,
                        help="Maximum primary learned clearance probability for the auxiliary "
                             "disagreement veto. This keeps the aux head focused on primary "
                             "blind spots instead of duplicating ordinary hard vetoes.")
    parser.add_argument("--body-clearance-aux-veto-margin", type=float, default=0.10,
                        help="Minimum auxiliary learned clearance probability improvement "
                             "required before the auxiliary disagreement veto may replace "
                             "the selected primitive.")
    parser.add_argument("--body-clearance-aux-veto-replacement-cap", type=float, default=0.90,
                        help="Maximum auxiliary learned clearance probability allowed for an "
                             "auxiliary-veto replacement primitive.")
    parser.add_argument("--body-clearance-aux-veto-primitives",
                        default="backward,yaw_left,yaw_right,arc_left,arc_right,hold",
                        help="Comma-separated fallback primitives eligible for the auxiliary "
                             "learned clearance disagreement veto.")
    parser.add_argument("--body-clearance-aux-veto-selected-primitives",
                        default="forward_medium,arc_left,arc_right,yaw_left,yaw_right",
                        help="Comma-separated selected primitives eligible for the auxiliary "
                             "learned clearance disagreement veto.")
    parser.add_argument("--primitive-aux-clearance-switch-current-body-risk",
                        action="store_true",
                        help="Use the current-body-risk RGB/JEPA classifier as a learned "
                             "switch that fuses --primitive-aux-clearance-checkpoint into "
                             "primitive scoring instead of the primary clearance head. This "
                             "keeps the switch nonprivileged; the classifier must be trained "
                             "offline for the desired visual phase.")
    parser.add_argument("--primitive-aux-clearance-switch-threshold", type=float, default=None,
                        help="Threshold for --primitive-aux-clearance-switch-current-body-risk. "
                             "Defaults to the loaded current-body-risk checkpoint threshold.")
    parser.add_argument("--primitive-aux-clearance-switch-min-claimed-count", type=int, default=0,
                        help="Minimum claimed beacon count before the learned auxiliary "
                             "clearance-head switch may activate.")
    parser.add_argument("--primitive-aux-clearance-switch-latch-ticks", type=int, default=0,
                        help="After the learned auxiliary clearance-head switch triggers, "
                             "keep the auxiliary clearance head active for this many ticks. "
                             "This smooths intermittent RGB/JEPA switch detections without "
                             "using pose or geometry.")
    parser.add_argument("--primitive-aux-clearance-switch-policy-features",
                        action="store_true",
                        help="When the learned auxiliary clearance switch is active, also "
                             "fuse the auxiliary clearance head into learned-local policy "
                             "features. By default the switch is guard-only, preserving the "
                             "navigation policy inputs while still letting the learned "
                             "clearance head veto unsafe actions.")
    parser.add_argument("--body-clearance-aux-switch-enable",
                        action="store_true",
                        help="While the learned auxiliary clearance switch is active, enable "
                             "learned swept-body clearance scoring/vetoes even outside the "
                             "near-target body-clearance servo gate. This uses the learned "
                             "current-body-risk switch to protect pre-beacon corner approach "
                             "without privileged geometry.")
    parser.add_argument("--body-clearance-aux-switch-ignore-min-area",
                        action="store_true",
                        help="While --body-clearance-aux-switch-enable is active, ignore "
                             "--body-clearance-learned-min-area-logit so learned clearance "
                             "penalties can act before the target occupies a large image area.")
    parser.add_argument("--body-clearance-aux-switch-arc-sweep-veto-prob",
                        type=float, default=1.01,
                        help="Auxiliary-switch-only learned clearance threshold for selected "
                             "arc primitives. When active, an arc whose learned swept-body "
                             "blocked probability exceeds this threshold may be replaced by "
                             "a safer learned-clearance fallback using the normal hard-veto "
                             "margin, cap, and fallback primitive list. Values above 1 disable.")
    parser.add_argument("--body-clearance-aux-switch-arc-sweep-veto-selected-primitives",
                        default="arc_left,arc_right",
                        help="Comma-separated selected primitives eligible for "
                             "--body-clearance-aux-switch-arc-sweep-veto-prob. This can "
                             "target a learned one-sided shoulder sweep without changing "
                             "opposite-direction arc decisions.")
    parser.add_argument("--body-clearance-saturated-veto-prob", type=float, default=1.01,
                        help="If the selected primitive's learned swept-body blocked "
                             "probability is at least this high, allow a low-sweep "
                             "fallback even when all candidates are similarly risky. "
                             "This targets saturated clearance-head corner cases where "
                             "a progress-scored arc would shoulder-sweep into a wall. "
                             "Values above 1 disable the hook.")
    parser.add_argument("--body-clearance-saturated-veto-spread", type=float, default=0.01,
                        help="Maximum learned clearance-probability increase allowed "
                             "for --body-clearance-saturated-veto-prob fallbacks.")
    parser.add_argument("--body-clearance-saturated-veto-primitives",
                        default="yaw_left,yaw_right,backward,hold",
                        help="Comma-separated fallback primitives eligible for "
                             "--body-clearance-saturated-veto-prob.")
    parser.add_argument("--body-clearance-saturated-veto-selected-primitives",
                        default="arc_left,arc_right",
                        help="Comma-separated selected primitives eligible for "
                             "--body-clearance-saturated-veto-prob.")
    parser.add_argument("--body-clearance-yaw-contact-veto-prob", type=float, default=1.01,
                        help="If a selected yaw-in-place primitive has at least this learned "
                             "swept-body clearance blocked probability, execute backward "
                             "instead: rotating with the body in wall contact can lever the "
                             "base over a lip in one tick (unrecoverable capsize). Values "
                             "above 1 disable. Learned clearance head only.")
    parser.add_argument("--body-clearance-yaw-direction-veto-prob", type=float, default=1.01,
                        help="If a selected pure-yaw primitive has at least this learned "
                             "swept-body blocked probability, allow the learned clearance "
                             "head to switch to the opposite pure-yaw direction when it is "
                             "materially safer. This preserves yaw-in-place behavior while "
                             "avoiding shoulder sweeps into nearby walls.")
    parser.add_argument("--body-clearance-yaw-direction-veto-margin", type=float, default=0.05,
                        help="Minimum learned clearance-probability improvement required "
                             "before --body-clearance-yaw-direction-veto-prob flips yaw "
                             "direction.")
    parser.add_argument("--body-clearance-current-contact-escape-m", type=float, default=None,
                        help="When the current explicit shoulder/body probe clearance is at "
                             "or below this margin-adjusted value, replace eligible yaw/arc/"
                             "forward selections with the lowest learned-clearance-risk "
                             "escape primitive. Omit to disable. This is a local contact "
                             "escape gate, not a route planner.")
    parser.add_argument("--body-clearance-current-contact-escape-m-by-primitive", default="",
                        help="Optional comma-separated primitive:meters overrides for "
                             "--body-clearance-current-contact-escape-m. Use this when "
                             "forward/arc shoulder moves need an earlier guard than pure "
                             "yaw turns; omitted primitives keep the base threshold.")
    parser.add_argument("--body-clearance-current-contact-escape-min-streak",
                        type=int, default=1,
                        help="Minimum consecutive low-current-clearance ticks required before "
                             "--body-clearance-current-contact-escape-m may fire. Values <=1 "
                             "preserve threshold-only behavior.")
    parser.add_argument("--body-clearance-current-contact-escape-cooldown-ticks",
                        type=int, default=0,
                        help="Minimum closed-loop ticks between current-contact escape "
                             "overrides. The clearance streak still updates during cooldown.")
    parser.add_argument("--body-clearance-current-contact-escape-min-claimed-count",
                        type=int, default=0,
                        help="Minimum number of claimed targets before current-contact escape "
                             "may fire. Use this to avoid disturbing the early route.")
    parser.add_argument("--body-clearance-current-contact-escape-states", default="",
                        help="Comma-separated controller states where current-contact escape "
                             "may fire. Empty means all states.")
    parser.add_argument("--body-clearance-current-contact-escape-target-colors", default="",
                        help="Comma-separated active target colors where current-contact "
                             "escape may fire. Empty means all targets.")
    parser.add_argument("--body-clearance-current-contact-escape-primitives",
                        default="forward_fast,forward_medium,arc_left,arc_right,yaw_left,yaw_right",
                        help="Comma-separated selected primitives eligible for "
                             "--body-clearance-current-contact-escape-m.")
    parser.add_argument("--body-clearance-current-contact-escape-replacements",
                        default="backward,yaw_left,yaw_right,hold",
                        help="Comma-separated learned-clearance-ranked replacement "
                             "primitives for --body-clearance-current-contact-escape-m.")
    parser.add_argument("--body-clearance-current-contact-escape-replacement-cap",
                        type=float, default=1.01,
                        help="Maximum learned swept-body clearance blocked probability "
                             "allowed for a current-contact escape replacement. Values "
                             "above 1 disable the cap. This prevents a low-clearance "
                             "escape from swapping into a replacement that the learned "
                             "clearance head also marks risky.")
    parser.add_argument("--body-clearance-current-contact-escape-require-replacement-under-cap",
                        action="store_true",
                        help="When --body-clearance-current-contact-escape-replacement-cap "
                             "is active, suppress the escape if every scored replacement "
                             "is over cap instead of relaxing to the least-bad over-cap "
                             "candidate.")
    parser.add_argument("--body-clearance-current-contact-escape-min-area-logit",
                        type=float, default=None,
                        help="Optional active-target RGB area-logit floor before "
                             "current-contact escape may run. This lets final approach "
                             "close distance before the low-clearance escape starts "
                             "intervening.")
    parser.add_argument("--body-clearance-current-contact-escape-min-area-states",
                        default="",
                        help="Comma-separated controller states where "
                             "--body-clearance-current-contact-escape-min-area-logit "
                             "applies. Empty applies the area gate in every state.")
    parser.add_argument("--body-clearance-current-contact-escape-min-projected-clearance-m",
                        type=float, default=None,
                        help="Optional swept-body geometry floor for current-contact escape "
                             "replacements. When set, replacements below this projected "
                             "clearance are rejected, and an already-selected primitive that "
                             "passes the floor is not replaced.")
    parser.add_argument("--body-clearance-current-contact-escape-min-projected-improvement-m",
                        type=float, default=0.0,
                        help="Optional required projected clearance improvement over the "
                             "current body-clearance probe before current-contact escape may "
                             "use a replacement. Values <=0 disable the improvement gate.")
    parser.add_argument("--body-clearance-risk-escape-threshold", type=float, default=1.01,
                        help="If the optional learned swept-body clearance head assigns at "
                             "least this blocked probability to the selected forward/yaw "
                             "primitive, execute a short backward/opposite-turn escape before "
                             "contact. Values above 1 disable the hook.")
    parser.add_argument("--body-clearance-risk-escape-blocks", type=int, default=0,
                        help="Number of learned-risk escape blocks to schedule when "
                             "--body-clearance-risk-escape-threshold fires. The trigger is "
                             "the nonprivileged RGB/JEPA clearance head, not map geometry.")
    parser.add_argument("--body-clearance-risk-escape-cooldown-ticks", type=int, default=0,
                        help="Minimum closed-loop ticks between learned body-risk escape "
                             "triggers. The already scheduled escape blocks still run.")
    parser.add_argument("--body-clearance-risk-escape-states", default="EXPLORE,SEEK,SERVO",
                        help="Comma-separated controller states where learned body-risk "
                             "escape may preempt a selected forward/yaw primitive.")
    parser.add_argument("--proprio-contact-detector-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from train_go2_proprio_contact_detector.py. "
                             "Runs a nonprivileged proprioceptive contact_now classifier over "
                             "a window of executed-primitive egomotion features and schedules "
                             "a committed escape when sustained contact is detected.")
    parser.add_argument("--proprio-contact-escape-threshold", type=float, default=0.7,
                        help="Rolling-mean (3 ticks) contact probability at or above which "
                             "the proprio contact streak advances.")
    parser.add_argument("--proprio-contact-escape-streak", type=int, default=2,
                        help="Consecutive over-threshold ticks required before a proprio "
                             "contact escape is scheduled.")
    parser.add_argument("--proprio-contact-escape-blocks", type=int, default=0,
                        help="Escape blocks scheduled on a proprio contact trigger. "
                             "Zero disables the escape (detector still logs probabilities).")
    parser.add_argument("--proprio-contact-escape-cooldown-ticks", type=int, default=12,
                        help="Minimum ticks between proprio contact escape triggers.")
    parser.add_argument("--proprio-contact-escape-states", default="EXPLORE,SEEK,SERVO",
                        help="Comma-separated controller states where the proprio contact "
                             "escape may schedule blocks.")
    parser.add_argument("--proprio-contact-map-blocks", action="store_true",
                        help="On a proprio contact trigger during a translating primitive, "
                             "mark the online-map edge for that primitive blocked. Rotation "
                             "and hold grinds never mark the map (lateral wall direction is "
                             "unobserved).")
    parser.add_argument("--history-risk-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from train_go2_history_risk_head.py. "
                             "History+action-conditioned per-primitive contact risk over "
                             "frozen-JEPA latents and proprioceptive egomotion; sees flank "
                             "walls through history that the single-frame head misses.")
    parser.add_argument("--history-risk-veto-threshold", type=float, default=1.01,
                        help="Veto the selected primitive when its history-risk blocked "
                             "probability is at or above this value. Values above 1 "
                             "disable the veto (probabilities still logged).")
    parser.add_argument("--history-risk-veto-primitives",
                        default="forward_slow,forward_medium,forward_fast,arc_left,arc_right,yaw_left,yaw_right",
                        help="Primitives eligible for the history-risk veto.")
    parser.add_argument("--history-risk-replacements",
                        default="backward,yaw_left,yaw_right,hold",
                        help="Replacement candidates ordered by predicted risk; the lowest "
                             "risk under --history-risk-replacement-cap is selected, else hold.")
    parser.add_argument("--history-risk-replacement-cap", type=float, default=0.9,
                        help="Maximum predicted risk allowed for a veto replacement.")
    parser.add_argument("--history-risk-states", default="EXPLORE,SEEK,SERVO",
                        help="Comma-separated controller states where the history-risk veto "
                             "may fire.")
    parser.add_argument("--history-risk-wedge-escape-blocks", type=int, default=2,
                        help="When every replacement is above the cap (wedged), commit this "
                             "many extra ticks of the least-risky escape direction instead "
                             "of per-tick reselection. Zero disables the committed escape.")
    parser.add_argument("--history-risk-wedge-escape-cooldown-ticks", type=int, default=6,
                        help="Minimum ticks between history-risk wedge escapes.")
    parser.add_argument("--history-risk-fuse-outcomes", action="store_true",
                        help="Fuse history-risk probabilities into primitive_outcomes "
                             "blocked_prob (max-combine) so the learned-policy rerank and "
                             "outcome guards avoid risky primitives at selection time "
                             "instead of relying on post-hoc vetoes.")
    parser.add_argument("--history-risk-fuse-weight", type=float, default=1.0,
                        help="Multiplier applied to history-risk probability before "
                             "max-combining into blocked_prob.")
    parser.add_argument("--history-risk-corridor-commit", action="store_true",
                        help="When the learned risk signature says corridor (both yaws "
                             "risky, forward safe) and a scan yaw/hold was selected, "
                             "commit to forward instead. Scanning inside a tight lane is "
                             "the dominant shoulder-contact source.")
    parser.add_argument("--history-risk-corridor-yaw-min", type=float, default=0.7,
                        help="Minimum predicted risk for BOTH yaws before corridor commit.")
    parser.add_argument("--history-risk-corridor-forward-max", type=float, default=0.3,
                        help="Maximum predicted forward risk for corridor commit.")
    parser.add_argument("--history-risk-corridor-max-run", type=int, default=6,
                        help="Maximum consecutive corridor commits before yielding a tick "
                             "back to normal selection.")
    parser.add_argument("--history-risk-corridor-states", default="EXPLORE,SEEK",
                        help="Controller states where corridor commit may fire.")
    parser.add_argument("--history-risk-relax-min-claims", type=int, default=-1,
                        help="When at least this many beacons are claimed, switch the "
                             "history-risk veto threshold and fuse weight to the relaxed "
                             "values so the endgame may thread tight approaches. Negative "
                             "disables relaxation.")
    parser.add_argument("--history-risk-relaxed-veto-threshold", type=float, default=0.97,
                        help="Veto threshold used once the claim relaxation is active.")
    parser.add_argument("--history-risk-relaxed-fuse-weight", type=float, default=0.4,
                        help="Fuse weight used once the claim relaxation is active.")
    parser.add_argument("--seen-target-route", action="store_true",
                        help="Route toward a seen-but-distant active target over the "
                             "runtime-built online map: estimate the target position from "
                             "bearing plus a calibrated area->distance model, then follow a "
                             "goal-biased frontier path. Bridges the gap between local "
                             "visual servo and blind exploration.")
    parser.add_argument("--seen-target-route-max-age-ticks", type=int, default=240,
                        help="Maximum age of the last sighting before the route estimate "
                             "expires.")
    parser.add_argument("--seen-target-route-goal-weight", type=float, default=3.0,
                        help="Weight of remaining goal distance vs path length when "
                             "selecting the frontier to route through.")
    parser.add_argument("--seen-target-route-handoff-area-logit", type=float, default=1.2,
                        help="When the target is currently seen at or above this area "
                             "logit, the router yields to the visual servo.")
    parser.add_argument("--seen-target-route-dist-calib", default="0.9531,-0.1972",
                        help="Comma-separated a,b of the learned distance calibration "
                             "dist_m = exp(a + b * area_logit), fitted offline on "
                             "train-scene logs.")
    parser.add_argument("--seen-target-route-states", default="EXPLORE,SEEK",
                        help="Controller states where the seen-target router may override "
                             "the requested primitive.")
    parser.add_argument("--broad-explorer-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from train_go2_broad_explorer_bc.py. "
                             "History-conditioned BC head over frozen-JEPA latents plus "
                             "proprioception, trained on corpus teacher/frontier slices "
                             "across many scenes; drives EXPLORE primitive selection.")
    parser.add_argument("--broad-explorer-states", default="EXPLORE",
                        help="Controller states where the broad explorer proposes the "
                             "requested primitive.")
    parser.add_argument("--novelty-route", action="store_true",
                        help="When no live seen-target estimate exists, draw the route "
                             "goal from the online memory instead: nearest unexplored "
                             "cell, held for a commitment window. Memory-directed "
                             "exploration through the same validated route-following "
                             "machinery as --seen-target-route (which must be enabled).")
    parser.add_argument("--novelty-route-commit-ticks", type=int, default=40,
                        help="Ticks a novelty goal is held before re-selection.")
    parser.add_argument("--novelty-route-scan-ticks", type=int, default=8,
                        help="Look-around yaw ticks injected on novelty-goal arrival "
                             "(sightings come from scanning at new places, and route "
                             "following otherwise suppresses the policy's scan yaws).")
    parser.add_argument("--visual-ray-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from train_go2_ray_depth_head.py. "
                             "Learned 1D-lidar: predicts free depth along K FOV rays from "
                             "the frozen-JEPA latent each tick and fuses the result into "
                             "the online map (vision_free/vision_blocked), so routing "
                             "plans over seen-but-undriven space.")
    parser.add_argument("--current-body-risk-checkpoint", type=Path, default=None,
                        help="Optional checkpoint from "
                             "train_go2_jepa_current_body_risk_predictor.py. This "
                             "nonprivileged RGB/JEPA head predicts whether the current "
                             "body envelope is already too close to a wall/corner.")
    parser.add_argument("--current-body-risk-threshold", type=float, default=None,
                        help="Override threshold for --current-body-risk-checkpoint.")
    parser.add_argument("--current-body-risk-min-claimed-count", type=int, default=0,
                        help="Minimum claimed beacon count before current-body-risk "
                             "recovery, preserve-yaw, or clearance-rerank hooks may "
                             "alter the selected primitive.")
    parser.add_argument("--current-body-risk-recovery-blocks", type=int, default=0,
                        help="Number of recovery blocks to schedule when current body "
                             "risk fires. The first block is backward.")
    parser.add_argument("--current-body-risk-recovery-selected-prob-floor",
                        type=float, default=None,
                        help="Optional learned swept-body clearance probability floor "
                             "for the already selected primitive before current-body "
                             "risk recovery may fire. This requires the current-body "
                             "risk head and primitive-clearance head to agree.")
    parser.add_argument("--current-body-risk-recovery-selected-primitives",
                        default="",
                        help="Optional comma-separated primitive names for which the "
                             "already selected primitive may trigger current-body "
                             "risk recovery. Empty means any non-backward primitive.")
    parser.add_argument("--current-body-risk-preserve-yaw", action="store_true",
                        help="When current body risk is high, preserve a requested pure "
                             "yaw correction instead of allowing the learned action "
                             "selector to upgrade it to a translating primitive. This "
                             "uses only the nonprivileged RGB/JEPA current-body-risk "
                             "head and the controller's requested primitive.")
    parser.add_argument("--current-body-risk-preserve-yaw-threshold", type=float, default=None,
                        help="Optional threshold for --current-body-risk-preserve-yaw. "
                             "Defaults to --current-body-risk-threshold.")
    parser.add_argument("--current-body-risk-preserve-yaw-min-area-logit", type=float, default=None,
                        help="Optional active-target area logit required before "
                             "--current-body-risk-preserve-yaw may fire. Defaults to "
                             "--current-body-risk-min-area-logit.")
    parser.add_argument("--current-body-risk-preserve-yaw-max-clearance-prob",
                        type=float, default=None,
                        help="Optional learned swept-body clearance probability ceiling "
                             "for --current-body-risk-preserve-yaw. When set, a requested "
                             "yaw is preserved only if the primitive-clearance head says "
                             "that yaw is body-clear; otherwise a safer learned rerank or "
                             "backward primitive may stand.")
    parser.add_argument("--current-body-risk-clearance-rerank", action="store_true",
                        help="When current body risk is high, choose the lowest learned "
                             "body-clearance-risk primitive from a small recovery set. "
                             "This is a nonprivileged RGB/JEPA rerank of the already "
                             "scored primitive candidates.")
    parser.add_argument("--current-body-risk-clearance-rerank-threshold", type=float, default=None,
                        help="Optional threshold for --current-body-risk-clearance-rerank. "
                             "Defaults to --current-body-risk-threshold.")
    parser.add_argument("--current-body-risk-clearance-rerank-min-area-logit", type=float, default=None,
                        help="Optional active-target area logit required before "
                             "--current-body-risk-clearance-rerank may fire. Defaults to "
                             "--current-body-risk-min-area-logit.")
    parser.add_argument("--current-body-risk-clearance-rerank-selected-prob-floor",
                        type=float, default=None,
                        help="Optional learned swept-body clearance probability floor "
                             "for the already selected primitive before current-body "
                             "clearance reranking may fire. This keeps the current-body "
                             "head from becoming a global brake on frames where the "
                             "selected primitive is already learned-clear.")
    parser.add_argument("--current-body-risk-clearance-rerank-selected-primitives",
                        default="",
                        help="Optional comma-separated primitive names for which the "
                             "already selected primitive may trigger current-body "
                             "clearance reranking. Empty means any selected primitive.")
    parser.add_argument("--current-body-risk-clearance-rerank-primitives",
                        default="yaw_left,yaw_right,backward",
                        help="Comma-separated primitive names eligible for current-body "
                             "clearance reranking.")
    parser.add_argument("--current-body-risk-min-area-logit", type=float, default=None,
                        help="Optional active-target area logit required before "
                             "current-body-risk recovery may fire. This keeps a "
                             "current-body head trained on near-corner labels from "
                             "becoming a global exploration brake.")
    parser.add_argument("--current-body-risk-cooldown-ticks", type=int, default=0,
                        help="Minimum closed-loop ticks between current-body-risk "
                             "recovery triggers.")
    parser.add_argument("--current-body-risk-states", default="EXPLORE,SEEK,SERVO",
                        help="Comma-separated controller states where current-body-risk "
                             "recovery may preempt a non-backward primitive.")
    parser.add_argument("--wall-guard-states", default="EXPLORE",
                        help="Comma-separated controller states where --wall-aware-planner may veto "
                             "forward/arc commands. Default keeps target SEEK/SERVO unmodified.")
    parser.add_argument("--wall-guard-post-claim-states", default="",
                        help="Additional comma-separated controller states where "
                             "--wall-aware-planner may veto after at least one beacon "
                             "has been claimed. This lets learned outcome guards protect "
                             "post-claim target seeking without changing first-acquisition "
                             "behavior.")
    parser.add_argument("--wall-guard-post-claim-min-claims", type=int, default=1,
                        help="Minimum claimed beacon count before "
                             "--wall-guard-post-claim-states are enabled.")
    parser.add_argument("--wall-min-clearance-m", type=float, default=0.02,
                        help="Minimum predicted C-space clearance for the wall-aware guard.")
    parser.add_argument("--wall-feasible-threshold", type=float, default=0.8,
                        help="Minimum feasible sample fraction before a forward primitive is vetoed.")
    parser.add_argument("--wall-body-forward-m", type=float, default=0.35,
                        help="Forward body probe length used by the wall-aware guard.")
    parser.add_argument("--wall-body-half-width-m", type=float, default=0.18,
                        help="Half-width body probe used by the wall-aware guard.")
    parser.add_argument("--wall-body-probe-margin-m", type=float, default=0.03,
                        help="Safety margin subtracted from explicit front/shoulder "
                             "probe-point obstacle clearance.")
    parser.add_argument("--wall-stall-displacement-m", type=float, default=0.03,
                        help="Physical blocks with less XY displacement than this count as contact-like stalls.")
    parser.add_argument("--wall-hard-stall-displacement-m", type=float, default=0.012,
                        help="Stricter displacement threshold for gate metrics that should count "
                             "hard wall-push stalls, separate from the softer recovery trigger.")
    parser.add_argument("--wall-stall-streak", type=int, default=2,
                        help="Consecutive forward stalls before scheduling escape blocks.")
    parser.add_argument("--wall-stall-block-waypoint", action="store_true",
                        help="In EXPLORE, treat repeated contact-like forward stalls as evidence "
                             "that the current coarse waypoint is physically blocked, then replan.")
    parser.add_argument("--wall-escape-blocks", type=int, default=2,
                        help="Backward/yaw escape blocks scheduled after a stall streak.")
    parser.add_argument("--wall-stall-penalty-score", type=float, default=0.45,
                        help="Temporary score penalty applied to a primitive family after a "
                             "physical contact-like stall.")
    parser.add_argument("--wall-stall-penalty-ticks", type=int, default=8,
                        help="Number of closed-loop ticks to keep a physical stall penalty active.")
    parser.add_argument("--wall-turn-loop-streak", type=int, default=12,
                        help="Consecutive turn commands before scheduling a deterministic "
                             "backward/opposite-turn escape.")
    parser.add_argument("--wall-turn-escape-blocks", type=int, default=3,
                        help="Escape-plan length scheduled after a long in-place turn loop.")
    parser.add_argument("--wall-turn-loop-block-waypoint", action="store_true",
                        help="Allow turn-loop recovery to permanently mark the current waypoint "
                             "blocked. Default only schedules an escape and replans.")
    parser.add_argument("--wall-predicted-blocked-waypoint-replan", action="store_true",
                        help="Allow learned predicted-blocked forward requests to mark the current "
                             "EXPLORE waypoint blocked before a physical stall proves it.")
    parser.add_argument("--wall-predicted-blocked-waypoint-streak", type=int, default=1,
                        help="Consecutive learned predicted-blocked forward requests for the same "
                             "EXPLORE waypoint before marking it blocked.")
    parser.add_argument("--command-smoothing-min-ticks", type=int, default=0,
                        help="Opt-in anti-twitch hold. If >1, keep the previously executed "
                             "eligible primitive until it has run for this many command ticks, "
                             "provided the learned guard still scores that primitive as clear.")
    parser.add_argument("--command-smoothing-states", default="EXPLORE",
                        help="Comma-separated controller states where --command-smoothing-min-ticks "
                             "may apply.")
    parser.add_argument("--command-smoothing-primitives",
                        default="forward_medium,yaw_left,yaw_right",
                        help="Comma-separated primitive names eligible for command smoothing.")
    parser.add_argument("--learned-safe-stride-primitive", default="",
                        help="Optional longer forward primitive to request only when learned action predictions "
                             "score it as clear enough, e.g. forward_fast.")
    parser.add_argument("--learned-safe-stride-from", default="forward_medium",
                        help="Only upgrade this requested primitive when --learned-safe-stride-primitive is set.")
    parser.add_argument("--learned-safe-stride-states", default="EXPLORE",
                        help="Comma-separated controller states where learned-safe stride upgrades may apply.")
    parser.add_argument("--learned-safe-stride-max-blocked-prob", type=float, default=0.08,
                        help="Maximum learned blocked probability allowed for a stride upgrade.")
    parser.add_argument("--learned-safe-stride-max-clearance-blocked-prob", type=float, default=None,
                        help="Optional maximum learned clearance blocked probability for a stride upgrade.")
    parser.add_argument("--learned-safe-stride-min-progress-m", type=float, default=0.055,
                        help="Minimum learned progress prediction required for a stride upgrade.")
    parser.add_argument("--learned-safe-stride-max-bearing", type=float, default=0.10,
                        help="Maximum route/target bearing magnitude allowed for a stride upgrade.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--demo-capture-rate", choices=("command", "policy"), default="command",
                        help="Video capture cadence. command captures once per 0.10 s command tick; "
                             "policy captures physical-mode videos once per 0.02 s locomotion "
                             "policy step for true 50 fps real-time playback.")
    parser.add_argument("--demo-fps", type=float, default=None,
                        help="Override output video FPS. Defaults to the selected capture cadence.")
    parser.add_argument(
        "--render-robot",
        action="store_true",
        help=(
            "Legacy/debug mode: render Go2 visual meshes in the main camera "
            "scene, including egocentric RGB. By default ego/policy RGB hides "
            "the robot body; demo videos use a separate robot-visible scene "
            "for the third-person panel."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--face-target", action="store_true",
                        help="Diagnostic: place robot at several poses facing the target color and "
                             "report the controller's color-mask area/bearing (tests in-sim detection).")
    args = parser.parse_args()
    if bool(args.generalized_runtime_contract):
        args.fully_learned_runtime_contract = True
    explore_route_waypoints = _parse_xy_waypoints(args.explore_route_waypoints)
    fully_learned_contract_report = _fully_learned_runtime_contract_report(
        args,
        explore_route_waypoints=explore_route_waypoints,
    )
    if bool(args.fully_learned_runtime_contract) and not bool(
        fully_learned_contract_report["passed"]
    ):
        raise SystemExit(
            "fully learned runtime contract violation:\n"
            + "\n".join(f"  - {item}" for item in fully_learned_contract_report["failures"])
        )
    claim_near_area_by_color = _parse_color_float_map(args.claim_near_area_logit_by_color)
    claim_near_bearing_by_color = _parse_color_float_map(args.claim_near_bearing_by_color)
    claim_near_min_seen_by_color = _parse_color_int_map(args.claim_near_min_seen_ticks_by_color)
    claim_success_proxy_area_by_color = _parse_color_float_map(
        args.claim_success_proxy_area_logit_by_color
    )
    claim_success_proxy_bearing_by_color = _parse_color_float_map(
        args.claim_success_proxy_bearing_by_color
    )
    body_clearance_current_contact_escape_m_by_primitive = _parse_primitive_float_map(
        args.body_clearance_current_contact_escape_m_by_primitive
    )
    learned_topology_route_yaw_threshold_by_color = _parse_color_float_map(
        args.learned_topology_route_yaw_threshold_by_color
    )
    learned_topology_route_arc_max_bearing_by_color = _parse_color_float_map(
        args.learned_topology_route_arc_max_bearing_by_color
    )
    wall_guard_states = {s.strip().upper() for s in str(args.wall_guard_states).split(",") if s.strip()}
    wall_guard_post_claim_states = {
        s.strip().upper()
        for s in str(args.wall_guard_post_claim_states).split(",")
        if s.strip()
    }
    primitive_outcome_preserve_turn_states = {
        s.strip().upper()
        for s in str(args.primitive_outcome_preserve_turn_states).split(",")
        if s.strip()
    }
    primitive_outcome_turn_body_rerank_primitives = {
        s.strip()
        for s in str(args.primitive_outcome_turn_body_rerank_primitives).split(",")
        if s.strip()
    }
    primitive_outcome_preserve_straight_states = {
        s.strip().upper()
        for s in str(args.primitive_outcome_preserve_straight_states).split(",")
        if s.strip()
    }
    primitive_outcome_forward_progress_floor_states = {
        s.strip().upper()
        for s in str(args.primitive_outcome_forward_progress_floor_states).split(",")
        if s.strip()
    }
    target_pursuit_stale_states = {
        s.strip().upper()
        for s in str(args.target_pursuit_stale_states).split(",")
        if s.strip()
    }
    primitive_outcome_low_progress_hard_veto_primitives = {
        s.strip()
        for s in str(args.primitive_outcome_low_progress_hard_veto_primitives).split(",")
        if s.strip()
    }
    primitive_outcome_blocked_hard_veto_primitives = {
        s.strip()
        for s in str(args.primitive_outcome_blocked_hard_veto_primitives).split(",")
        if s.strip()
    }
    primitive_outcome_blocked_hard_veto_selected_primitives = {
        s.strip()
        for s in str(args.primitive_outcome_blocked_hard_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_risk_escape_states = {
        s.strip().upper()
        for s in str(args.body_clearance_risk_escape_states).split(",")
        if s.strip()
    }
    body_clearance_hard_veto_primitives = {
        s.strip()
        for s in str(args.body_clearance_hard_veto_primitives).split(",")
        if s.strip()
    }
    body_clearance_aux_switch_hard_veto_primitives = {
        s.strip()
        for s in str(args.body_clearance_aux_switch_hard_veto_primitives).split(",")
        if s.strip()
    }
    body_clearance_hard_veto_selected_primitives = {
        s.strip()
        for s in str(args.body_clearance_hard_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_hard_veto_hold_escape_primitives = {
        s.strip()
        for s in str(args.body_clearance_hard_veto_hold_escape_primitives).split(",")
        if s.strip() and s.strip() != "hold"
    }
    body_clearance_hard_veto_hold_escape_states = {
        s.strip().upper()
        for s in str(args.body_clearance_hard_veto_hold_escape_states).split(",")
        if s.strip()
    }
    body_clearance_hard_veto_hold_escape_override_primitives = {
        s.strip()
        for s in str(args.body_clearance_hard_veto_hold_escape_override_primitives).split(",")
        if s.strip() and s.strip() != "hold"
    }
    body_clearance_hard_veto_hold_escape_override_states = {
        s.strip().upper()
        for s in str(args.body_clearance_hard_veto_hold_escape_override_states).split(",")
        if s.strip()
    }
    body_clearance_aux_switch_arc_sweep_veto_selected_primitives = {
        s.strip()
        for s in str(args.body_clearance_aux_switch_arc_sweep_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_aux_veto_primitives = {
        s.strip()
        for s in str(args.body_clearance_aux_veto_primitives).split(",")
        if s.strip()
    }
    body_clearance_aux_veto_selected_primitives = {
        s.strip()
        for s in str(args.body_clearance_aux_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_saturated_veto_primitives = {
        s.strip()
        for s in str(args.body_clearance_saturated_veto_primitives).split(",")
        if s.strip()
    }
    body_clearance_saturated_veto_selected_primitives = {
        s.strip()
        for s in str(args.body_clearance_saturated_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_current_contact_escape_primitives = {
        s.strip()
        for s in str(args.body_clearance_current_contact_escape_primitives).split(",")
        if s.strip()
    }
    body_clearance_current_contact_escape_replacements = {
        s.strip()
        for s in str(args.body_clearance_current_contact_escape_replacements).split(",")
        if s.strip()
    }
    body_clearance_current_contact_escape_states = {
        s.strip().upper()
        for s in str(args.body_clearance_current_contact_escape_states).split(",")
        if s.strip()
    }
    body_clearance_current_contact_escape_min_area_states = {
        s.strip().upper()
        for s in str(args.body_clearance_current_contact_escape_min_area_states).split(",")
        if s.strip()
    }
    body_clearance_current_contact_escape_target_colors = {
        s.strip().lower()
        for s in str(args.body_clearance_current_contact_escape_target_colors).split(",")
        if s.strip()
    }
    learned_topology_route_geometry_veto_selected_primitives = {
        s.strip()
        for s in str(args.learned_topology_route_geometry_veto_selected_primitives).split(",")
        if s.strip()
    }
    learned_topology_route_geometry_veto_replacements = {
        s.strip()
        for s in str(args.learned_topology_route_geometry_veto_replacements).split(",")
        if s.strip()
    }
    body_clearance_geometry_veto_states = {
        s.strip().upper()
        for s in str(args.body_clearance_geometry_veto_states).split(",")
        if s.strip()
    }
    body_clearance_geometry_veto_target_colors = {
        s.strip().lower()
        for s in str(args.body_clearance_geometry_veto_target_colors).split(",")
        if s.strip()
    }
    body_clearance_geometry_veto_selected_primitives = {
        s.strip()
        for s in str(args.body_clearance_geometry_veto_selected_primitives).split(",")
        if s.strip()
    }
    body_clearance_geometry_veto_replacements = {
        s.strip()
        for s in str(args.body_clearance_geometry_veto_replacements).split(",")
        if s.strip()
    }
    body_clearance_geometry_veto_blocked_fallback_primitives = [
        s.strip()
        for s in str(args.body_clearance_geometry_veto_blocked_fallback_primitives).split(",")
        if s.strip()
    ]
    body_clearance_geometry_veto_override_replacements = {
        s.strip()
        for s in str(args.body_clearance_geometry_veto_override_replacements).split(",")
        if s.strip()
    }
    current_body_risk_states = {
        s.strip().upper()
        for s in str(args.current_body_risk_states).split(",")
        if s.strip()
    }
    current_body_risk_recovery_selected_primitives = {
        s.strip()
        for s in str(args.current_body_risk_recovery_selected_primitives).split(",")
        if s.strip()
    }
    current_body_risk_clearance_rerank_primitives = {
        s.strip()
        for s in str(args.current_body_risk_clearance_rerank_primitives).split(",")
        if s.strip()
    }
    current_body_risk_clearance_rerank_selected_primitives = {
        s.strip()
        for s in str(args.current_body_risk_clearance_rerank_selected_primitives).split(",")
        if s.strip()
    }
    learned_local_policy_frontier_pressure_guard_recovery_primitives = [
        s.strip()
        for s in str(args.learned_local_policy_frontier_pressure_guard_recovery_primitives).split(",")
        if s.strip()
    ]
    command_smoothing_states = {
        s.strip().upper()
        for s in str(args.command_smoothing_states).split(",")
        if s.strip()
    }
    command_smoothing_primitives = {
        s.strip()
        for s in str(args.command_smoothing_primitives).split(",")
        if s.strip()
    }
    learned_safe_stride_states = {
        s.strip().upper()
        for s in str(args.learned_safe_stride_states).split(",")
        if s.strip()
    }
    learned_local_policy_states = {
        s.strip().upper()
        for s in str(args.learned_local_policy_states).split(",")
        if s.strip()
    }
    learned_local_policy_post_claim_states = {
        s.strip().upper()
        for s in str(args.learned_local_policy_post_claim_states).split(",")
        if s.strip()
    }
    learned_local_dataset_states = {
        s.strip().upper()
        for s in str(args.learned_local_dataset_states).split(",")
        if s.strip()
    }
    if not learned_local_dataset_states:
        learned_local_dataset_states = set(learned_local_policy_states)
    debug_force_primitive_script = _load_debug_force_primitive_script(
        args.debug_force_primitive_script
    )
    learned_local_policy_online_map_novelty_states = {
        s.strip().upper()
        for s in str(args.learned_local_policy_online_map_novelty_states).split(",")
        if s.strip()
    }
    learned_local_policy_frontier_pressure_states = {
        s.strip().upper()
        for s in str(args.learned_local_policy_frontier_pressure_states).split(",")
        if s.strip()
    }
    learned_local_oracle_standoff_label_states = {
        s.strip().upper()
        for s in str(args.learned_local_oracle_standoff_label_states).split(",")
        if s.strip()
    }
    learned_local_policy_translation_pressure_primitives = [
        s.strip()
        for s in str(args.learned_local_policy_translation_pressure_primitives).split(",")
        if s.strip()
    ]
    learned_local_policy_translation_pressure_states = {
        s.strip().upper()
        for s in str(args.learned_local_policy_translation_pressure_states).split(",")
        if s.strip()
    }
    post_claim_explore_primitives = [
        s.strip()
        for s in str(args.post_claim_explore_primitives).split(",")
        if s.strip()
    ]
    use_learned_wall_source = bool(
        args.wall_aware_planner and args.wall_decision_source == "learned_front"
    )
    use_learned_action_source = bool(
        args.wall_aware_planner and args.wall_decision_source == "learned_action"
    )

    if str(args.device).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))
    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family)
    if args.scene_id:
        scene_dirs = [d for d in scene_dirs if d.name == args.scene_id]
        if not scene_dirs:
            raise SystemExit(
                f"scene-id not found in split={args.split} family={args.family}: {args.scene_id}"
            )
    if args.slice_start_result is not None and args.slice_snapshot_input is not None:
        raise SystemExit("--slice-start-result and --slice-snapshot-input are mutually exclusive")
    scene_dir = scene_dirs[0]
    print(f"scene={scene_dir.name} target={args.target_color}", flush=True)

    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    build = build_scene_from_pack(pack, n_envs=1, backend=str(args.backend),
                                  show_viewer=False, render_robot=bool(args.render_robot),
                                  apply_textures=bool(args.apply_textures))
    third_person_build = None
    if args.demo_video is not None and not bool(args.render_robot):
        third_person_build = build_scene_from_pack(
            pack,
            n_envs=1,
            backend=str(args.backend),
            show_viewer=False,
            render_robot=True,
            apply_textures=bool(args.apply_textures),
        )
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    grid = InflatedOccupancyGrid(pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=float(args.inflation_m))
    grid_global[0] = grid
    spawn_pos, spawn_quat = _scene_spawn(scene_dir)
    landmarks = _scene_landmarks(scene_dir)
    wb = json.loads((scene_dir / "genesis_scene.json").read_text())["world_bounds_xy_m"]
    bounds = (wb[0][0], wb[0][1], wb[1][0], wb[1][1]) if isinstance(wb[0], (list, tuple)) else tuple(wb)
    explorer = FrontierExplorer(
        grid,
        bounds,
        goal_policy=str(args.explore_goal_policy),
        yaw_bearing_threshold=float(args.explore_yaw_bearing_threshold),
        forward_bearing_threshold=float(args.explore_forward_bearing_threshold),
        lookahead_m=float(args.explore_lookahead_m),
        forward_primitive=str(args.explore_forward_primitive),
        coverage_lookahead_cells=int(args.explore_coverage_lookahead_cells),
        dfs_neighbor_order=str(args.explore_dfs_neighbor_order),
        scan_interval=int(args.explore_scan_interval),
        scan_len=int(args.explore_scan_len),
        scan_primitive=str(args.explore_scan_primitive),
        route_waypoints=explore_route_waypoints,
        route_start_after_claims=int(args.explore_route_start_after_claims),
        route_advance_m=float(args.explore_route_advance_m),
        standoff_route=bool(args.explore_standoff_route),
        standoff_targets=landmarks,
        standoff_grid=grid,
        standoff_scene_graph=pack.scene_graph,
        standoff_m=float(args.explore_standoff_m),
        standoff_lookahead_m=float(args.explore_standoff_lookahead_m),
        standoff_replan_interval=int(args.explore_standoff_replan_interval),
        standoff_candidates=int(args.explore_standoff_candidates),
        standoff_arrival_m=float(args.explore_standoff_arrival_m),
        standoff_path_spacing_m=float(args.explore_standoff_path_spacing_m),
        standoff_clearance_weight=float(args.explore_standoff_clearance_weight),
        standoff_clearance_target_m=float(args.explore_standoff_clearance_target_m),
        standoff_body_route_clearance_weight=float(args.explore_standoff_body_route_clearance_weight),
        standoff_body_route_clearance_target_m=float(args.explore_standoff_body_route_clearance_target_m),
        standoff_body_route_ignore_start_m=float(args.explore_standoff_body_route_ignore_start_m),
        standoff_cardinal_route=bool(args.explore_standoff_cardinal_route),
        standoff_corner_guard=bool(args.explore_standoff_corner_guard),
        standoff_corner_commit_m=float(args.explore_standoff_corner_commit_m),
        standoff_corner_standoff_m=float(args.explore_standoff_corner_standoff_m),
        standoff_allow_arcs=bool(args.explore_standoff_allow_arcs),
        standoff_arc_min_bearing=(
            None
            if args.explore_standoff_arc_min_bearing is None
            else float(args.explore_standoff_arc_min_bearing)
        ),
        standoff_arc_max_bearing=(
            None
            if args.explore_standoff_arc_max_bearing is None
            else float(args.explore_standoff_arc_max_bearing)
        ),
        standoff_arc_min_target_dist_m=float(args.explore_standoff_arc_min_target_dist_m),
        standoff_heading_mode=str(args.explore_standoff_heading_mode),
        standoff_heading_lookahead_m=float(args.explore_standoff_heading_lookahead_m),
        standoff_prefix_snap_start=bool(args.explore_standoff_prefix_snap_start),
        standoff_snap_start_min_dist_m=float(args.explore_standoff_snap_start_min_dist_m),
        standoff_body_check=bool(args.explore_standoff_body_check),
        standoff_body_lookahead_m=float(args.explore_standoff_body_lookahead_m),
        standoff_body_min_clearance_m=float(args.explore_standoff_body_min_clearance_m),
        standoff_body_recovery_clearance_m=(
            None
            if args.explore_standoff_body_recovery_clearance_m is None
            else float(args.explore_standoff_body_recovery_clearance_m)
        ),
        standoff_intent_smoothing=bool(args.explore_standoff_intent_smoothing),
        standoff_sticky_target_ticks=int(args.explore_standoff_sticky_target_ticks),
        standoff_sticky_target_release_m=float(args.explore_standoff_sticky_target_release_m),
        standoff_yaw_enter_threshold=(
            None
            if args.explore_standoff_yaw_enter_threshold is None
            else float(args.explore_standoff_yaw_enter_threshold)
        ),
        standoff_yaw_exit_threshold=(
            None
            if args.explore_standoff_yaw_exit_threshold is None
            else float(args.explore_standoff_yaw_exit_threshold)
        ),
        standoff_yaw_flip_threshold=(
            None
            if args.explore_standoff_yaw_flip_threshold is None
            else float(args.explore_standoff_yaw_flip_threshold)
        ),
        body_forward_m=float(args.wall_body_forward_m),
        body_half_width_m=float(args.wall_body_half_width_m),
        body_probe_margin_m=float(args.wall_body_probe_margin_m),
    )
    print(f"explorer free nav-cells: {len(explorer.free)}", flush=True)
    oracle_standoff_explorer = None
    if bool(args.learned_local_oracle_standoff_labels):
        oracle_standoff_explorer = FrontierExplorer(
            grid,
            bounds,
            goal_policy="learned_policy",
            yaw_bearing_threshold=float(args.explore_yaw_bearing_threshold),
            forward_bearing_threshold=float(args.explore_forward_bearing_threshold),
            lookahead_m=float(args.explore_lookahead_m),
            forward_primitive=str(args.explore_forward_primitive),
            scan_interval=int(args.explore_scan_interval),
            scan_len=int(args.explore_scan_len),
            scan_primitive=str(args.explore_scan_primitive),
            standoff_route=True,
            standoff_targets=landmarks,
            standoff_grid=grid,
            standoff_scene_graph=pack.scene_graph,
            standoff_m=float(args.explore_standoff_m),
            standoff_lookahead_m=float(args.explore_standoff_lookahead_m),
            standoff_replan_interval=int(args.explore_standoff_replan_interval),
            standoff_candidates=int(args.explore_standoff_candidates),
            standoff_arrival_m=float(args.explore_standoff_arrival_m),
            standoff_path_spacing_m=float(args.explore_standoff_path_spacing_m),
            standoff_clearance_weight=float(args.explore_standoff_clearance_weight),
            standoff_clearance_target_m=float(args.explore_standoff_clearance_target_m),
            standoff_body_route_clearance_weight=float(args.explore_standoff_body_route_clearance_weight),
            standoff_body_route_clearance_target_m=float(args.explore_standoff_body_route_clearance_target_m),
            standoff_body_route_ignore_start_m=float(args.explore_standoff_body_route_ignore_start_m),
            standoff_cardinal_route=bool(args.explore_standoff_cardinal_route),
            standoff_corner_guard=bool(args.explore_standoff_corner_guard),
            standoff_corner_commit_m=float(args.explore_standoff_corner_commit_m),
            standoff_corner_standoff_m=float(args.explore_standoff_corner_standoff_m),
            standoff_allow_arcs=bool(args.explore_standoff_allow_arcs),
            standoff_arc_min_bearing=(
                None
                if args.explore_standoff_arc_min_bearing is None
                else float(args.explore_standoff_arc_min_bearing)
            ),
            standoff_arc_max_bearing=(
                None
                if args.explore_standoff_arc_max_bearing is None
                else float(args.explore_standoff_arc_max_bearing)
            ),
            standoff_arc_min_target_dist_m=float(args.explore_standoff_arc_min_target_dist_m),
            standoff_heading_mode=str(args.explore_standoff_heading_mode),
            standoff_heading_lookahead_m=float(args.explore_standoff_heading_lookahead_m),
            standoff_prefix_snap_start=bool(args.explore_standoff_prefix_snap_start),
            standoff_snap_start_min_dist_m=float(args.explore_standoff_snap_start_min_dist_m),
            standoff_body_check=bool(args.explore_standoff_body_check),
            standoff_body_lookahead_m=float(args.explore_standoff_body_lookahead_m),
            standoff_body_min_clearance_m=float(args.explore_standoff_body_min_clearance_m),
            standoff_body_recovery_clearance_m=(
                None
                if args.explore_standoff_body_recovery_clearance_m is None
                else float(args.explore_standoff_body_recovery_clearance_m)
            ),
            standoff_intent_smoothing=bool(args.explore_standoff_intent_smoothing),
            standoff_sticky_target_ticks=int(args.explore_standoff_sticky_target_ticks),
            standoff_sticky_target_release_m=float(args.explore_standoff_sticky_target_release_m),
            standoff_yaw_enter_threshold=(
                None
                if args.explore_standoff_yaw_enter_threshold is None
                else float(args.explore_standoff_yaw_enter_threshold)
            ),
            standoff_yaw_exit_threshold=(
                None
                if args.explore_standoff_yaw_exit_threshold is None
                else float(args.explore_standoff_yaw_exit_threshold)
            ),
            standoff_yaw_flip_threshold=(
                None
                if args.explore_standoff_yaw_flip_threshold is None
                else float(args.explore_standoff_yaw_flip_threshold)
            ),
            body_forward_m=float(args.wall_body_forward_m),
            body_half_width_m=float(args.wall_body_half_width_m),
            body_probe_margin_m=float(args.wall_body_probe_margin_m),
        )

    # Physical mode: drive the robot with the trained Go2 PPO walking policy stepped
    # in Genesis physics (real gait + rigid-body collisions). The datagen that trained
    # the controller used this same RolloutRunner path, so the walking camera matches
    # the training perception distribution. Kinematic mode keeps the teleport fallback.
    runner = None
    if args.mode == "physical":
        safety = SafetyLimits.from_manifest(platform)
        policy = GenesisGo2PPOPolicy.from_platform_manifest(
            platform, REPO_ROOT, device=str(args.policy_device))
        config = RolloutConfig(
            n_blocks=int(args.max_ticks), fall_z_threshold_m=float(args.fall_z_threshold_m),
            rgb_capture_per_block=False, seed=int(args.seed),
            log_progress_every_blocks=0, foot_contact_source="zero",
            randomize_spawn_pose=False,
        )
        runner = RolloutRunner(build, policy, registry, safety, config=config)
    _set_pose(build=build, runner=runner, pos_xyz=spawn_pos, quat_wxyz=spawn_quat)

    model = color_vocab = aux_mean = aux_std = None
    tc = range_scale = None
    target_sequence = [str(args.target_color)]
    if args.target_colors is not None:
        target_sequence = [c.strip() for c in str(args.target_colors).split(",") if c.strip()]
    elif str(args.target_color).lower() == "all":
        target_sequence = [c for c in ("red", "green", "blue", "yellow") if c in landmarks]
    if not target_sequence:
        raise SystemExit("no target colors requested")
    slice_start: dict[str, Any] | None = None
    slice_preclaimed_colors = {
        item.strip().lower()
        for item in str(args.slice_preclaimed_colors).split(",")
        if item.strip()
    }
    ctrl_state = None
    if args.policy == "memory":
        if args.controller is None:
            raise SystemExit("--controller required for memory policy")
        model, color_vocab, aux_mean, aux_std, ck = load_controller(
            args.controller, device=device, frozen_jepa_checkpoint=args.frozen_jepa_checkpoint)
        if args.mask_sigma is not None:
            model.rgb_evidence_sigma = max(1e-4, float(args.mask_sigma))
        if args.mask_threshold is not None:
            model.rgb_evidence_threshold = float(args.mask_threshold)
        if args.mask_area_threshold is not None:
            model.rgb_evidence_area_threshold = max(1e-6, float(args.mask_area_threshold))
        print(f"mask: sigma={model.rgb_evidence_sigma} threshold={model.rgb_evidence_threshold} "
              f"area_threshold={model.rgb_evidence_area_threshold}", flush=True)
        missing_colors = [color for color in target_sequence if color not in color_vocab]
        if missing_colors:
            raise SystemExit(f"target colors {missing_colors} not in {color_vocab}")
        missing_landmarks = [color for color in target_sequence if color not in landmarks]
        if missing_landmarks:
            raise SystemExit(f"target colors {missing_landmarks} not in scene landmarks {sorted(landmarks)}")
        if len(target_sequence) > 1 and args.demo_mode != "explore":
            raise SystemExit("--target-colors is currently supported for --demo-mode explore only")
        tc = color_vocab.index(target_sequence[0])
        range_scale = float(ck["range_scale_m"])
    if args.slice_start_result is not None:
        slice_active_target = str(args.slice_active_target_color or "").strip().lower()
        if not slice_active_target:
            raise SystemExit("--slice-active-target-color is required with --slice-start-result")
        target_lookup = {str(color).lower(): idx for idx, color in enumerate(target_sequence)}
        if slice_active_target not in target_lookup:
            raise SystemExit(
                f"--slice-active-target-color={slice_active_target!r} is not in "
                f"--target-colors/--target-color sequence {target_sequence}"
            )
        slice_start = _load_slice_start(
            args.slice_start_result,
            start_tick=int(args.slice_start_tick),
            preclaimed_colors=slice_preclaimed_colors,
        )
        sx, sy = slice_start["start_xy"]
        spawn_pos = np.asarray([float(sx), float(sy), float(spawn_pos[2])], dtype=np.float32)
        spawn_quat = _quat_wxyz_from_yaw_local(float(slice_start["start_yaw"]))
        _set_pose(build=build, runner=runner, pos_xyz=spawn_pos, quat_wxyz=spawn_quat)
        print(
            "slice-start: "
            f"source={args.slice_start_result} tick={slice_start['start_tick']} "
            f"xy={[round(float(v), 3) for v in slice_start['start_xy']]} "
            f"yaw={math.degrees(float(slice_start['start_yaw'])):.1f}deg "
            f"active={slice_active_target} preclaimed={sorted(slice_preclaimed_colors)}",
            flush=True,
        )
    slice_snapshot: dict[str, Any] | None = None
    if args.slice_snapshot_input is not None:
        try:
            slice_snapshot = torch.load(args.slice_snapshot_input, map_location="cpu", weights_only=False)
        except TypeError:
            slice_snapshot = torch.load(args.slice_snapshot_input, map_location="cpu")
        if slice_snapshot.get("schema") != "lewm_go2_yellow_slice_resume_snapshot_v0":
            raise SystemExit(
                f"unsupported --slice-snapshot-input schema: {slice_snapshot.get('schema')}"
            )
        if str(slice_snapshot.get("scene_id")) != str(scene_dir.name):
            raise SystemExit(
                f"snapshot scene {slice_snapshot.get('scene_id')} does not match {scene_dir.name}"
            )
        snap_sequence = [str(item) for item in slice_snapshot.get("target_sequence", [])]
        if snap_sequence and snap_sequence != list(target_sequence):
            raise SystemExit(
                f"snapshot target sequence {snap_sequence} does not match requested {target_sequence}"
            )
        spawn_pos = np.asarray(slice_snapshot["pos_xyz"], dtype=np.float32)
        spawn_quat = np.asarray(slice_snapshot["quat_wxyz"], dtype=np.float32)
        restored_physics = _restore_slice_snapshot_physics(
            build=build,
            runner=runner,
            snapshot=slice_snapshot,
            fallback_pos=spawn_pos,
            fallback_quat=spawn_quat,
        )
        print(
            "slice-snapshot: "
            f"source={args.slice_snapshot_input} next_tick={slice_snapshot.get('next_tick')} "
            f"xy={[round(float(v), 3) for v in spawn_pos[:2]]} "
            f"yaw={math.degrees(float(slice_snapshot.get('yaw_rad', 0.0))):.1f}deg "
            f"active={target_sequence[int(slice_snapshot.get('target_index', 0))]} "
            f"physics_state={'restored' if restored_physics else 'pose_only'}",
            flush=True,
        )

    front_encoder = front_head = None
    front_image_size = 64
    front_threshold = float(args.front_blocked_threshold) if args.front_blocked_threshold is not None else None
    if use_learned_wall_source:
        if args.front_blocked_checkpoint is None:
            raise SystemExit("--front-blocked-checkpoint required for learned_front wall decisions")
        try:
            front_ck = torch.load(args.front_blocked_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            front_ck = torch.load(args.front_blocked_checkpoint, map_location=device)
        encoder_checkpoint = args.frozen_jepa_checkpoint or Path(str(front_ck["frozen_jepa_checkpoint"]))
        front_encoder, _front_encoder_ck = load_go2_jepa_encoder(encoder_checkpoint, device=device, freeze=True)
        front_head = Go2FrontBlockedHead(
            latent_dim=int(front_ck.get("latent_dim", 96)),
            hidden_dim=int(front_ck.get("hidden_dim", 128)),
        ).to(device)
        front_head.load_state_dict(front_ck["model_state_dict"])
        front_head.eval()
        front_image_size = int(front_ck.get("image_size", 64))
        if front_threshold is None:
            front_threshold = float(front_ck.get("threshold", 0.5))
        print(
            f"front-blocked: checkpoint={args.front_blocked_checkpoint.name} "
            f"threshold={front_threshold:.3f} image_size={front_image_size}",
            flush=True,
        )
    outcome_encoder = outcome_head = None
    post_claim_outcome_encoder = post_claim_outcome_head = None
    clearance_encoder = clearance_head = None
    aux_clearance_encoder = aux_clearance_head = None
    current_body_encoder = current_body_head = None
    outcome_image_size = 64
    post_claim_outcome_image_size = 64
    clearance_image_size = 64
    aux_clearance_image_size = 64
    current_body_image_size = 64
    outcome_primitive_vocab = list(PRIMITIVE_NAMES)
    post_claim_outcome_primitive_vocab = list(PRIMITIVE_NAMES)
    clearance_primitive_vocab = list(PRIMITIVE_NAMES)
    aux_clearance_primitive_vocab = list(PRIMITIVE_NAMES)
    outcome_threshold = (
        float(args.primitive_outcome_threshold)
        if args.primitive_outcome_threshold is not None
        else None
    )
    clearance_threshold = (
        float(args.primitive_clearance_threshold)
        if args.primitive_clearance_threshold is not None
        else None
    )
    if use_learned_action_source:
        if args.primitive_outcome_checkpoint is None:
            raise SystemExit("--primitive-outcome-checkpoint required for learned_action wall decisions")
        try:
            outcome_ck = torch.load(args.primitive_outcome_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            outcome_ck = torch.load(args.primitive_outcome_checkpoint, map_location=device)
        encoder_checkpoint = (
            args.primitive_outcome_frozen_jepa_checkpoint
            or args.frozen_jepa_checkpoint
            or Path(str(outcome_ck["frozen_jepa_checkpoint"]))
        )
        outcome_encoder, _outcome_encoder_ck = load_go2_jepa_encoder(encoder_checkpoint, device=device, freeze=True)
        outcome_primitive_vocab = [str(item) for item in outcome_ck.get("primitive_vocab", PRIMITIVE_NAMES)]
        outcome_head = Go2PrimitiveOutcomeHead(
            latent_dim=int(outcome_ck.get("latent_dim", 96)),
            primitive_count=len(outcome_primitive_vocab),
            hidden_dim=int(outcome_ck.get("hidden_dim", 160)),
        ).to(device)
        outcome_head.load_state_dict(outcome_ck["model_state_dict"])
        outcome_head.eval()
        outcome_image_size = int(outcome_ck.get("image_size", 64))
        if outcome_threshold is None:
            outcome_threshold = float(outcome_ck.get("threshold", 0.5))
        print(
            f"primitive-outcome: checkpoint={args.primitive_outcome_checkpoint.name} "
            f"threshold={outcome_threshold:.3f} image_size={outcome_image_size}",
            flush=True,
        )
        if args.primitive_post_claim_outcome_checkpoint is not None:
            try:
                post_claim_outcome_ck = torch.load(
                    args.primitive_post_claim_outcome_checkpoint,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:
                post_claim_outcome_ck = torch.load(
                    args.primitive_post_claim_outcome_checkpoint,
                    map_location=device,
                )
            post_claim_encoder_checkpoint = args.frozen_jepa_checkpoint or Path(
                str(post_claim_outcome_ck["frozen_jepa_checkpoint"])
            )
            post_claim_outcome_encoder, _post_claim_outcome_encoder_ck = load_go2_jepa_encoder(
                post_claim_encoder_checkpoint,
                device=device,
                freeze=True,
            )
            post_claim_outcome_primitive_vocab = [
                str(item) for item in post_claim_outcome_ck.get("primitive_vocab", PRIMITIVE_NAMES)
            ]
            post_claim_outcome_head = Go2PrimitiveOutcomeHead(
                latent_dim=int(post_claim_outcome_ck.get("latent_dim", 96)),
                primitive_count=len(post_claim_outcome_primitive_vocab),
                hidden_dim=int(post_claim_outcome_ck.get("hidden_dim", 160)),
            ).to(device)
            post_claim_outcome_head.load_state_dict(post_claim_outcome_ck["model_state_dict"])
            post_claim_outcome_head.eval()
            post_claim_outcome_image_size = int(post_claim_outcome_ck.get("image_size", 64))
            print(
                "primitive-post-claim-outcome: "
                f"checkpoint={args.primitive_post_claim_outcome_checkpoint.name} "
                f"threshold={outcome_threshold:.3f} image_size={post_claim_outcome_image_size}",
                flush=True,
            )
        if args.primitive_clearance_checkpoint is not None:
            try:
                clearance_ck = torch.load(args.primitive_clearance_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                clearance_ck = torch.load(args.primitive_clearance_checkpoint, map_location=device)
            clearance_encoder_checkpoint = (
                args.primitive_clearance_frozen_jepa_checkpoint
                or args.frozen_jepa_checkpoint
                or Path(str(clearance_ck["frozen_jepa_checkpoint"]))
            )
            clearance_encoder, _clearance_encoder_ck = load_go2_jepa_encoder(
                clearance_encoder_checkpoint,
                device=device,
                freeze=True,
            )
            clearance_primitive_vocab = [
                str(item) for item in clearance_ck.get("primitive_vocab", PRIMITIVE_NAMES)
            ]
            clearance_head = Go2PrimitiveOutcomeHead(
                latent_dim=int(clearance_ck.get("latent_dim", 96)),
                primitive_count=len(clearance_primitive_vocab),
                hidden_dim=int(clearance_ck.get("hidden_dim", 160)),
            ).to(device)
            clearance_head.load_state_dict(clearance_ck["model_state_dict"])
            clearance_head.eval()
            clearance_image_size = int(clearance_ck.get("image_size", 64))
            if clearance_threshold is None:
                clearance_threshold = float(clearance_ck.get("threshold", 0.5))
            print(
                f"primitive-clearance: checkpoint={args.primitive_clearance_checkpoint.name} "
                f"threshold={clearance_threshold:.3f} image_size={clearance_image_size}",
                flush=True,
            )
        if args.primitive_aux_clearance_checkpoint is not None:
            try:
                aux_clearance_ck = torch.load(
                    args.primitive_aux_clearance_checkpoint,
                    map_location=device,
                    weights_only=False,
                )
            except TypeError:
                aux_clearance_ck = torch.load(args.primitive_aux_clearance_checkpoint, map_location=device)
            aux_clearance_encoder_checkpoint = (
                args.primitive_aux_clearance_frozen_jepa_checkpoint
                or args.primitive_clearance_frozen_jepa_checkpoint
                or args.frozen_jepa_checkpoint
                or Path(str(aux_clearance_ck["frozen_jepa_checkpoint"]))
            )
            aux_clearance_encoder, _aux_clearance_encoder_ck = load_go2_jepa_encoder(
                aux_clearance_encoder_checkpoint,
                device=device,
                freeze=True,
            )
            aux_clearance_primitive_vocab = [
                str(item) for item in aux_clearance_ck.get("primitive_vocab", PRIMITIVE_NAMES)
            ]
            aux_clearance_head = Go2PrimitiveOutcomeHead(
                latent_dim=int(aux_clearance_ck.get("latent_dim", 96)),
                primitive_count=len(aux_clearance_primitive_vocab),
                hidden_dim=int(aux_clearance_ck.get("hidden_dim", 160)),
            ).to(device)
            aux_clearance_head.load_state_dict(aux_clearance_ck["model_state_dict"])
            aux_clearance_head.eval()
            aux_clearance_image_size = int(aux_clearance_ck.get("image_size", 64))
            print(
                f"primitive-aux-clearance: checkpoint={args.primitive_aux_clearance_checkpoint.name} "
                f"image_size={aux_clearance_image_size}",
                flush=True,
            )
    current_body_threshold = (
        float(args.current_body_risk_threshold)
        if args.current_body_risk_threshold is not None
        else None
    )
    if args.current_body_risk_checkpoint is not None:
        try:
            current_body_ck = torch.load(
                args.current_body_risk_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            current_body_ck = torch.load(args.current_body_risk_checkpoint, map_location=device)
        current_body_encoder_checkpoint = (
            Path(str(current_body_ck["frozen_jepa_checkpoint"]))
            if current_body_ck.get("frozen_jepa_checkpoint")
            else args.frozen_jepa_checkpoint
        )
        current_body_encoder, _current_body_encoder_ck = load_go2_jepa_encoder(
            current_body_encoder_checkpoint,
            device=device,
            freeze=True,
        )
        current_body_head = Go2FrontBlockedHead(
            latent_dim=int(current_body_ck.get("latent_dim", 96)),
            hidden_dim=int(current_body_ck.get("hidden_dim", 128)),
        ).to(device)
        current_body_head.load_state_dict(current_body_ck["model_state_dict"])
        current_body_head.eval()
        current_body_image_size = int(current_body_ck.get("image_size", 64))
        if current_body_threshold is None:
            current_body_threshold = float(current_body_ck.get("threshold", 0.5))
        print(
            f"current-body-risk: checkpoint={args.current_body_risk_checkpoint.name} "
            f"threshold={current_body_threshold:.3f} image_size={current_body_image_size}",
            flush=True,
        )

    proprio_contact_detector = None
    proprio_contact_window = 0
    proprio_contact_feature_mean: np.ndarray | None = None
    proprio_contact_feature_std: np.ndarray | None = None
    proprio_contact_escape_states = {
        state.strip().upper()
        for state in str(args.proprio_contact_escape_states).split(",")
        if state.strip()
    }
    if args.proprio_contact_detector_checkpoint is not None:
        from train_go2_proprio_contact_detector import ProprioContactDetector

        try:
            proprio_contact_ck = torch.load(
                args.proprio_contact_detector_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            proprio_contact_ck = torch.load(
                args.proprio_contact_detector_checkpoint, map_location=device
            )
        proprio_contact_window = int(proprio_contact_ck["window"])
        proprio_contact_detector = ProprioContactDetector(
            proprio_contact_window,
            int(proprio_contact_ck["feature_dim"]),
            hidden_dim=int(proprio_contact_ck.get("hidden_dim", 128)),
            arch=str(proprio_contact_ck.get("arch", "mlp")),
        ).to(device)
        proprio_contact_detector.load_state_dict(proprio_contact_ck["model_state_dict"])
        proprio_contact_detector.eval()
        proprio_contact_feature_mean = np.asarray(
            proprio_contact_ck["feature_mean"], dtype=np.float32
        )
        proprio_contact_feature_std = np.asarray(
            proprio_contact_ck["feature_std"], dtype=np.float32
        )
        print(
            f"proprio-contact-detector: checkpoint={args.proprio_contact_detector_checkpoint.name} "
            f"arch={proprio_contact_ck.get('arch', 'mlp')} window={proprio_contact_window} "
            f"threshold={float(args.proprio_contact_escape_threshold):.3f} "
            f"blocks={int(args.proprio_contact_escape_blocks)}",
            flush=True,
        )

    history_risk_model = None
    history_risk_encoder = None
    history_risk_vocab: list[str] = []
    history_risk_window = 0
    history_risk_image_size = 128
    history_risk_latent_mean: np.ndarray | None = None
    history_risk_latent_std: np.ndarray | None = None
    history_risk_proprio_mean: np.ndarray | None = None
    history_risk_proprio_std: np.ndarray | None = None
    history_risk_states = {
        state.strip().upper()
        for state in str(args.history_risk_states).split(",")
        if state.strip()
    }
    history_risk_corridor_states = {
        state.strip().upper()
        for state in str(args.history_risk_corridor_states).split(",")
        if state.strip()
    }
    history_risk_veto_primitives = {
        name.strip()
        for name in str(args.history_risk_veto_primitives).split(",")
        if name.strip()
    }
    history_risk_replacements = [
        name.strip()
        for name in str(args.history_risk_replacements).split(",")
        if name.strip()
    ]
    if args.history_risk_checkpoint is not None:
        from train_go2_history_risk_head import HistoryRiskHead

        try:
            history_risk_ck = torch.load(
                args.history_risk_checkpoint, map_location=device, weights_only=False
            )
        except TypeError:
            history_risk_ck = torch.load(args.history_risk_checkpoint, map_location=device)
        history_risk_vocab = [str(name) for name in history_risk_ck["primitive_vocab"]]
        history_risk_window = int(history_risk_ck["window"])
        history_risk_image_size = int(history_risk_ck.get("image_size", 128))
        history_risk_model = HistoryRiskHead(
            int(history_risk_ck["latent_dim"]),
            int(history_risk_ck["proprio_dim"]),
            len(history_risk_vocab),
            hidden_dim=int(history_risk_ck.get("hidden_dim", 192)),
            latent_proj_dim=int(history_risk_ck.get("latent_proj_dim", 96)),
        ).to(device)
        history_risk_model.load_state_dict(history_risk_ck["model_state_dict"])
        history_risk_model.eval()
        history_risk_encoder, _history_risk_encoder_ck = load_go2_jepa_encoder(
            Path(str(history_risk_ck["frozen_jepa_checkpoint"])),
            device=device,
            freeze=True,
        )
        history_risk_latent_mean = np.asarray(history_risk_ck["latent_mean"], dtype=np.float32)
        history_risk_latent_std = np.asarray(history_risk_ck["latent_std"], dtype=np.float32)
        history_risk_proprio_mean = np.asarray(history_risk_ck["proprio_mean"], dtype=np.float32)
        history_risk_proprio_std = np.asarray(history_risk_ck["proprio_std"], dtype=np.float32)
        print(
            f"history-risk: checkpoint={args.history_risk_checkpoint.name} "
            f"window={history_risk_window} image_size={history_risk_image_size} "
            f"veto_threshold={float(args.history_risk_veto_threshold):.3f}",
            flush=True,
        )

    broad_explorer_model = None
    broad_explorer_encoder = None
    broad_explorer_vocab: list[str] = []
    broad_explorer_window = 0
    broad_explorer_image_size = 128
    broad_explorer_latent_mean: np.ndarray | None = None
    broad_explorer_latent_std: np.ndarray | None = None
    broad_explorer_proprio_mean: np.ndarray | None = None
    broad_explorer_proprio_std: np.ndarray | None = None
    broad_explorer_states = {
        state.strip().upper()
        for state in str(args.broad_explorer_states).split(",")
        if state.strip()
    }
    if args.broad_explorer_checkpoint is not None:
        from train_go2_broad_explorer_bc import BroadExplorerBC

        try:
            broad_explorer_ck = torch.load(
                args.broad_explorer_checkpoint, map_location=device, weights_only=False
            )
        except TypeError:
            broad_explorer_ck = torch.load(args.broad_explorer_checkpoint, map_location=device)
        broad_explorer_vocab = [str(p) for p in broad_explorer_ck["primitive_vocab"]]
        broad_explorer_window = int(broad_explorer_ck["window"])
        broad_explorer_image_size = int(broad_explorer_ck.get("image_size", 128))
        broad_explorer_model = BroadExplorerBC(
            int(broad_explorer_ck["latent_dim"]),
            int(broad_explorer_ck["proprio_dim"]),
            len(broad_explorer_vocab),
            hidden_dim=int(broad_explorer_ck.get("hidden_dim", 192)),
            latent_proj_dim=int(broad_explorer_ck.get("latent_proj_dim", 96)),
        ).to(device)
        broad_explorer_model.load_state_dict(broad_explorer_ck["model_state_dict"])
        broad_explorer_model.eval()
        if history_risk_encoder is not None and str(
            broad_explorer_ck["frozen_jepa_checkpoint"]
        ) != str(history_risk_ck["frozen_jepa_checkpoint"]):
            raise SystemExit(
                "broad-explorer and history-risk checkpoints must share the same "
                "frozen encoder (shared sequence latent buffer)"
            )
        if history_risk_encoder is None:
            broad_explorer_encoder, _bx_ck = load_go2_jepa_encoder(
                Path(str(broad_explorer_ck["frozen_jepa_checkpoint"])),
                device=device,
                freeze=True,
            )
        broad_explorer_latent_mean = np.asarray(broad_explorer_ck["latent_mean"], dtype=np.float32)
        broad_explorer_latent_std = np.asarray(broad_explorer_ck["latent_std"], dtype=np.float32)
        broad_explorer_proprio_mean = np.asarray(broad_explorer_ck["proprio_mean"], dtype=np.float32)
        broad_explorer_proprio_std = np.asarray(broad_explorer_ck["proprio_std"], dtype=np.float32)
        print(
            f"broad-explorer: checkpoint={args.broad_explorer_checkpoint.name} "
            f"window={broad_explorer_window} states={sorted(broad_explorer_states)}",
            flush=True,
        )

    visual_ray_model = None
    visual_ray_encoder = None
    visual_ray_image_size = 128
    visual_ray_depth_cap = 4.0
    visual_ray_angles: np.ndarray | None = None
    visual_ray_latent_mean: np.ndarray | None = None
    visual_ray_latent_std: np.ndarray | None = None
    if args.visual_ray_checkpoint is not None:
        from train_go2_ray_depth_head import RayDepthHead

        try:
            visual_ray_ck = torch.load(
                args.visual_ray_checkpoint, map_location=device, weights_only=False
            )
        except TypeError:
            visual_ray_ck = torch.load(args.visual_ray_checkpoint, map_location=device)
        visual_ray_model = RayDepthHead(
            int(visual_ray_ck["latent_dim"]),
            int(visual_ray_ck["k_rays"]),
            hidden_dim=int(visual_ray_ck.get("hidden_dim", 256)),
        ).to(device)
        visual_ray_model.load_state_dict(visual_ray_ck["model_state_dict"])
        visual_ray_model.eval()
        visual_ray_image_size = int(visual_ray_ck.get("image_size", 128))
        visual_ray_depth_cap = float(visual_ray_ck.get("depth_cap_m", 4.0))
        half_fov = math.radians(float(visual_ray_ck.get("fov_deg", 78.323))) / 2.0
        visual_ray_angles = np.linspace(-half_fov, half_fov, int(visual_ray_ck["k_rays"]))
        visual_ray_latent_mean = np.asarray(visual_ray_ck["latent_mean"], dtype=np.float32)
        visual_ray_latent_std = np.asarray(visual_ray_ck["latent_std"], dtype=np.float32)
        if history_risk_encoder is None and broad_explorer_encoder is None:
            visual_ray_encoder, _vr_ck = load_go2_jepa_encoder(
                Path(str(visual_ray_ck["frozen_jepa_checkpoint"])),
                device=device,
                freeze=True,
            )
        print(
            f"visual-ray: checkpoint={args.visual_ray_checkpoint.name} "
            f"k={int(visual_ray_ck['k_rays'])} cap={visual_ray_depth_cap}m "
            f"err={visual_ray_ck.get('best_val_median_err_m')}",
            flush=True,
        )

    claim_success_model: ClaimSuccessHead | None = None
    claim_success_checkpoint: dict[str, Any] | None = None
    claim_success_model_threshold: float | None = None
    claim_success_model_threshold_by_color: dict[str, float] = {}
    claim_success_model_color_vocab: list[str] = []
    if args.claim_success_model_checkpoint is not None:
        try:
            claim_success_checkpoint = torch.load(
                args.claim_success_model_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            claim_success_checkpoint = torch.load(
                args.claim_success_model_checkpoint,
                map_location=device,
            )
        if claim_success_checkpoint.get("schema") != _CLAIM_SUCCESS_CHECKPOINT_SCHEMA:
            raise SystemExit(
                "--claim-success-model-checkpoint has unsupported schema: "
                f"{claim_success_checkpoint.get('schema')}"
            )
        if claim_success_checkpoint.get("feature_schema") != _CLAIM_SUCCESS_FEATURE_SCHEMA:
            raise SystemExit(
                "--claim-success-model-checkpoint has unsupported feature schema: "
                f"{claim_success_checkpoint.get('feature_schema')}"
            )
        claim_success_model = ClaimSuccessHead(
            input_dim=int(claim_success_checkpoint["input_dim"]),
            hidden_dim=int(claim_success_checkpoint.get("hidden_dim", 64)),
        ).to(device)
        claim_success_model.load_state_dict(claim_success_checkpoint["model_state_dict"])
        claim_success_model.eval()
        claim_success_model_color_vocab = [
            str(item).lower()
            for item in claim_success_checkpoint.get("color_vocab", color_vocab or [])
        ]
        claim_success_model_threshold = (
            float(args.claim_success_model_threshold)
            if args.claim_success_model_threshold is not None
            else float(claim_success_checkpoint.get("threshold", 0.5))
        )
        raw_threshold_by_color = claim_success_checkpoint.get("threshold_by_color", {})
        if isinstance(raw_threshold_by_color, dict):
            claim_success_model_threshold_by_color = {
                str(color).lower(): float(value)
                for color, value in raw_threshold_by_color.items()
            }
        claim_success_model_threshold_by_color.update(
            _parse_color_float_map(args.claim_success_model_threshold_by_color)
        )
        print(
            f"claim-success-model: checkpoint={args.claim_success_model_checkpoint.name} "
            f"threshold={claim_success_model_threshold:.3f} "
            f"threshold_by_color={claim_success_model_threshold_by_color} "
            f"input_dim={claim_success_checkpoint['input_dim']}",
            flush=True,
        )

    learned_target_scheduler_model: TargetSchedulerHead | None = None
    learned_target_scheduler_checkpoint: dict[str, Any] | None = None
    learned_target_scheduler_color_vocab: list[str] = []
    if args.learned_target_scheduler_checkpoint is not None:
        (
            learned_target_scheduler_model,
            learned_target_scheduler_checkpoint,
        ) = _load_target_scheduler(args.learned_target_scheduler_checkpoint, device=device)
        learned_target_scheduler_color_vocab = [
            str(item).lower()
            for item in learned_target_scheduler_checkpoint.get("color_vocab", color_vocab or [])
        ]
        missing_scheduler_colors = [
            color for color in target_sequence if color not in learned_target_scheduler_color_vocab
        ]
        if missing_scheduler_colors:
            raise SystemExit(
                "--learned-target-scheduler-checkpoint is missing target colors: "
                + ",".join(missing_scheduler_colors)
            )
        print(
            f"learned-target-scheduler: checkpoint={args.learned_target_scheduler_checkpoint.name} "
            f"input_dim={learned_target_scheduler_checkpoint['input_dim']} "
            f"colors={','.join(learned_target_scheduler_color_vocab)}",
            flush=True,
        )

    learned_local_policy_model = None
    learned_local_policy_checkpoint: dict[str, Any] | None = None
    learned_local_post_claim_policy_model = None
    learned_local_post_claim_policy_checkpoint: dict[str, Any] | None = None
    learned_local_target_policy_models: dict[str, nn.Module] = {}
    learned_local_target_policy_checkpoints: dict[str, dict[str, Any]] = {}
    if args.learned_local_policy_checkpoint is not None:
        learned_local_policy_model, learned_local_policy_checkpoint = _load_learned_local_policy(
            args.learned_local_policy_checkpoint,
            device=device,
        )
        learned_local_policy_checkpoint["source"] = str(args.learned_local_policy_checkpoint)
        print(
            f"learned-local-policy: checkpoint={args.learned_local_policy_checkpoint.name} "
            f"model={learned_local_policy_checkpoint.get('model_type', 'mlp')} "
            f"input_dim={learned_local_policy_checkpoint['input_dim']} "
            f"primitives={','.join(str(p) for p in learned_local_policy_checkpoint['primitive_vocab'])}",
            flush=True,
        )
    if args.learned_local_post_claim_policy_checkpoint is not None:
        (
            learned_local_post_claim_policy_model,
            learned_local_post_claim_policy_checkpoint,
        ) = _load_learned_local_policy(
            args.learned_local_post_claim_policy_checkpoint,
            device=device,
        )
        learned_local_post_claim_policy_checkpoint["source"] = str(
            args.learned_local_post_claim_policy_checkpoint
        )
        print(
            f"learned-local-post-claim-policy: checkpoint="
            f"{args.learned_local_post_claim_policy_checkpoint.name} "
            f"model={learned_local_post_claim_policy_checkpoint.get('model_type', 'mlp')} "
            f"input_dim={learned_local_post_claim_policy_checkpoint['input_dim']} "
            f"primitives={','.join(str(p) for p in learned_local_post_claim_policy_checkpoint['primitive_vocab'])}",
            flush=True,
        )
    learned_local_target_policy_specs = _parse_target_policy_checkpoint_specs(
        args.learned_local_target_policy_checkpoints
    )
    learned_local_target_policy_specs.update(
        _parse_target_state_policy_checkpoint_specs(
            args.learned_local_target_policy_state_checkpoints
        )
    )
    for target_key, checkpoint_path in sorted(learned_local_target_policy_specs.items()):
        target_model, target_checkpoint = _load_learned_local_policy(
            checkpoint_path,
            device=device,
        )
        target_checkpoint["source"] = str(checkpoint_path)
        learned_local_target_policy_models[target_key] = target_model
        learned_local_target_policy_checkpoints[target_key] = target_checkpoint
        print(
            f"learned-local-target-policy[{target_key}]: checkpoint={checkpoint_path.name} "
            f"model={target_checkpoint.get('model_type', 'mlp')} "
            f"input_dim={target_checkpoint['input_dim']} "
            f"primitives={','.join(str(p) for p in target_checkpoint['primitive_vocab'])}",
            flush=True,
        )
    learned_local_policy_feature_variant = "base"
    checkpoint_feature_variant = (
        "base"
        if learned_local_policy_checkpoint is None
        else str(learned_local_policy_checkpoint.get("feature_variant", "base"))
    )
    post_claim_checkpoint_feature_variant = (
        "base"
        if learned_local_post_claim_policy_checkpoint is None
        else str(learned_local_post_claim_policy_checkpoint.get("feature_variant", "base"))
    )
    checkpoint_feature_variants = [checkpoint_feature_variant]
    if learned_local_post_claim_policy_checkpoint is not None:
        checkpoint_feature_variants.append(post_claim_checkpoint_feature_variant)
    for target_checkpoint in learned_local_target_policy_checkpoints.values():
        checkpoint_feature_variants.append(str(target_checkpoint.get("feature_variant", "base")))

    def _runtime_feature_flags(variant: str) -> dict[str, bool]:
        return {
            "clock": bool(args.learned_local_clock_features)
            or _learned_local_feature_variant_has_clock(variant),
            "state": bool(args.learned_local_state_features)
            or _learned_local_feature_variant_has_state(variant),
            "visual_readout": bool(args.learned_local_visual_readout_features)
            or _learned_local_feature_variant_has_visual_readout(variant),
            "pose_topology": bool(args.learned_local_pose_topology_features)
            or _learned_local_feature_variant_has_pose_topology(variant),
            "online_map": bool(args.learned_local_online_map_features)
            or _learned_local_feature_variant_has_online_map(variant),
        }

    learned_local_primary_feature_flags = _runtime_feature_flags(checkpoint_feature_variant)
    learned_local_post_claim_feature_flags = (
        _runtime_feature_flags(post_claim_checkpoint_feature_variant)
        if learned_local_post_claim_policy_checkpoint is not None
        else dict(learned_local_primary_feature_flags)
    )
    learned_local_target_policy_feature_flags = {
        color: _runtime_feature_flags(str(checkpoint.get("feature_variant", "base")))
        for color, checkpoint in learned_local_target_policy_checkpoints.items()
    }
    learned_local_policy_clock_features = bool(
        learned_local_primary_feature_flags["clock"]
        or learned_local_post_claim_feature_flags["clock"]
        or any(flags["clock"] for flags in learned_local_target_policy_feature_flags.values())
    )
    learned_local_policy_state_features = bool(
        learned_local_primary_feature_flags["state"]
        or learned_local_post_claim_feature_flags["state"]
        or any(flags["state"] for flags in learned_local_target_policy_feature_flags.values())
    )
    learned_local_policy_visual_readout_features = bool(
        learned_local_primary_feature_flags["visual_readout"]
        or learned_local_post_claim_feature_flags["visual_readout"]
        or any(
            flags["visual_readout"]
            for flags in learned_local_target_policy_feature_flags.values()
        )
    )
    learned_local_policy_pose_topology_features = bool(
        learned_local_primary_feature_flags["pose_topology"]
        or learned_local_post_claim_feature_flags["pose_topology"]
        or any(
            flags["pose_topology"]
            for flags in learned_local_target_policy_feature_flags.values()
        )
    )
    learned_local_policy_online_map_features = bool(
        learned_local_primary_feature_flags["online_map"]
        or learned_local_post_claim_feature_flags["online_map"]
        or any(flags["online_map"] for flags in learned_local_target_policy_feature_flags.values())
    )
    variant_parts: list[str] = []
    if bool(learned_local_policy_pose_topology_features):
        variant_parts.append("pose_topology")
    elif bool(learned_local_policy_clock_features):
        variant_parts.append("clock")
    if bool(learned_local_policy_state_features):
        variant_parts.append("state")
    if bool(learned_local_policy_visual_readout_features):
        variant_parts.append("visual_readout")
    if bool(learned_local_policy_online_map_features):
        if any("online_map_edge" in variant for variant in checkpoint_feature_variants):
            variant_parts.append("online_map_edge")
        else:
            variant_parts.append("online_map")
    learned_local_policy_feature_variant = (
        "_".join(variant_parts) + "_v1" if variant_parts else "base"
    )
    learned_local_online_map_channel_count = (
        max(_learned_local_online_map_channel_count(variant) for variant in checkpoint_feature_variants)
        if bool(learned_local_policy_online_map_features)
        else 0
    )
    if bool(learned_local_policy_online_map_features) and learned_local_online_map_channel_count <= 0:
        learned_local_online_map_channel_count = _LEARNED_LOCAL_ONLINE_MAP_CHANNELS
    learned_local_online_map_feature_dim = int(
        learned_local_online_map_channel_count
        * int(args.learned_local_online_map_size)
        * int(args.learned_local_online_map_size)
    )

    learned_topology_route_table = None
    if args.learned_topology_route_table is not None:
        learned_topology_route_table = _load_learned_topology_route_table(
            args.learned_topology_route_table
        )
        route_count = len(learned_topology_route_table.get("routes", {}))
        waypoint_count = sum(
            len(route.get("waypoints", ()))
            for route in learned_topology_route_table.get("routes", {}).values()
        )
        print(
            f"learned-topology-route-table: checkpoint={args.learned_topology_route_table.name} "
            f"routes={route_count} waypoints={waypoint_count}",
            flush=True,
        )

    if args.demo_mode == "recall" and args.policy == "memory" and not args.face_target:
        from benchmark_lewm_closed_loop_mpc import _quat_wxyz_from_yaw
        place = _los_placement(grid, landmarks[args.target_color], explorer.free)
        if place is None:
            raise SystemExit(f"no line-of-sight standoff found for {args.target_color}")
        rpos, ryaw = place
        _set_pose(build=build, runner=runner, pos_xyz=rpos, quat_wxyz=_quat_wxyz_from_yaw(ryaw))
        print(f"recall: placed at {rpos[:2].tolist()} facing {math.degrees(ryaw):.0f}deg, "
              f"target at {landmarks[args.target_color].tolist()}", flush=True)

    if args.face_target and args.policy == "memory":
        from benchmark_lewm_closed_loop_mpc import _quat_wxyz_from_yaw
        target_xy = landmarks[args.target_color]
        print(f"target {args.target_color} at {target_xy.tolist()}")
        for dist_m in (1.0, 1.5, 2.0):
            for jitter in (-0.4, 0.0, 0.4):
                # place robot dist_m from target, heading toward it + jitter
                to_t = target_xy - spawn_pos[:2]
                base_heading = math.atan2(target_xy[1] - 0.0, target_xy[0] - 0.0)
                # approach from spawn side: stand between spawn and target
                dirv = target_xy - spawn_pos[:2]
                dirv = dirv / (np.linalg.norm(dirv) + 1e-6)
                rpos = np.array([target_xy[0] - dirv[0] * dist_m, target_xy[1] - dirv[1] * dist_m,
                                 float(spawn_pos[2])], dtype=np.float32)
                heading = math.atan2(target_xy[1] - rpos[1], target_xy[0] - rpos[0]) + jitter
                quat = _quat_wxyz_from_yaw(heading)
                _set_pose(build=build, runner=None, pos_xyz=rpos, quat_wxyz=quat)
                ego = _render_tensor_from_base(build, pack, base_xyz_m=rpos, base_quat_wxyz=quat, device=device)
                ego64 = F.interpolate(ego.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0]
                aux = _build_aux((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "hold")
                aux_t = (torch.from_numpy(aux).to(device) - aux_mean) / aux_std
                outputs, _ = model.step_online(ego64, aux_t, None, None)
                area = float(outputs["rgb_area_logits"][tc])
                evid = outputs["evidence_vec"][tc]
                bearing = math.atan2(float(evid[1]), float(evid[0]))
                tb = wrap_angle_pi(math.atan2(target_xy[1] - rpos[1], target_xy[0] - rpos[0]) - heading)
                g, r, b = ego64[1], ego64[0], ego64[2]
                gm = (g > 0.45) & (r < 0.5) & (b < 0.5)
                ng = int(gm.sum())
                gmean = [round(float(ego64[c][gm].mean()), 2) for c in range(3)] if ng > 5 else None
                print(f"  dist={dist_m} jit={jitter:+.1f} | area={area:+.2f} fires={area>0} "
                      f"est_bearing={math.degrees(bearing):+.0f} true_bearing={math.degrees(tb):+.0f} "
                      f"green_px={ng} green_mean={gmean}")
                if abs(jitter) < 0.01 and dist_m == 1.0:
                    import imageio
                    p = REPO_ROOT / ".generated/go2_memory_closed_loop" / f"facetarget_{args.target_color}_tex{int(args.apply_textures)}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(str(p), ego64.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
                    print(f"  saved {p}")
        return 0

    rng = np.random.default_rng(args.seed)
    state = {"rng": rng}
    frames: list = []
    capture = args.demo_video is not None
    prev_pose = (float(spawn_pos[0]), float(spawn_pos[1]), _yaw_from_quat_wxyz(spawn_quat))
    last_primitive, last_cmd = "hold", (0.0, 0.0, 0.0)
    if slice_snapshot is not None:
        ctrl_state = _ctrl_state_to_device(slice_snapshot.get("ctrl_state"), device=device)
        last_primitive = str(slice_snapshot.get("last_primitive", "hold"))
        last_cmd = tuple(float(v) for v in slice_snapshot.get("last_cmd", (0.0, 0.0, 0.0)))
    last_primitive_run_ticks = 0
    log = []
    claimed = False
    target_index = 0
    if slice_start is not None:
        slice_active_target = str(args.slice_active_target_color).strip().lower()
        target_index = {
            str(color).lower(): idx for idx, color in enumerate(target_sequence)
        }[slice_active_target]
    if slice_snapshot is not None:
        target_index = int(slice_snapshot.get("target_index", target_index))
    slice_tick_offset = int(slice_start["start_tick"]) if slice_start is not None else 0
    if slice_snapshot is not None:
        slice_tick_offset = int(slice_snapshot.get("next_tick", slice_snapshot.get("tick", 0)))
    feature_max_ticks = int(args.slice_feature_max_ticks or args.max_ticks)
    if slice_start is not None:
        feature_max_ticks = int(
            args.slice_feature_max_ticks
            or slice_start.get("source_ticks_used")
            or max(int(args.max_ticks), int(slice_tick_offset) + int(args.max_ticks))
        )
    if slice_snapshot is not None:
        feature_max_ticks = int(
            args.slice_feature_max_ticks
            or slice_snapshot.get("feature_max_ticks")
            or max(int(args.max_ticks), int(slice_tick_offset) + int(args.max_ticks))
        )
    target_active_since_tick = int(slice_tick_offset)
    if slice_snapshot is not None:
        target_active_since_tick = int(
            slice_snapshot.get("target_active_since_tick", target_active_since_tick)
        )
    active_target_color = target_sequence[target_index]
    active_target_tc = color_vocab.index(active_target_color) if color_vocab is not None else None
    full_from_spawn_contract = bool(
        slice_start is None
        and slice_snapshot is None
        and (args.fully_learned_runtime_contract or args.generalized_runtime_contract)
    )
    configured_body_clearance_veto_min_claimed_count = max(
        0,
        int(args.body_clearance_veto_min_claimed_count),
    )
    effective_body_clearance_veto_min_claimed_count = (
        configured_body_clearance_veto_min_claimed_count
    )
    configured_aux_clearance_switch_min_claimed_count = max(
        0,
        int(args.primitive_aux_clearance_switch_min_claimed_count),
    )
    effective_aux_clearance_switch_min_claimed_count = (
        configured_aux_clearance_switch_min_claimed_count
    )
    first_seen_ticks: dict[str, int] = {}
    last_seen_ticks: dict[str, int] = {}
    beacon_claims: list[dict[str, Any]] = []
    if slice_start is not None:
        beacon_claims.extend(dict(item) for item in slice_start["preclaims"])
        for claim in beacon_claims:
            color = str(claim.get("target_color", "")).lower()
            if color:
                claim_tick = int(claim.get("tick", 0))
                first_seen_ticks.setdefault(color, claim_tick)
                last_seen_ticks[color] = max(int(last_seen_ticks.get(color, claim_tick)), claim_tick)
    if slice_snapshot is not None:
        beacon_claims = [dict(item) for item in slice_snapshot.get("beacon_claims", [])]
        first_seen_ticks = {
            str(k): int(v) for k, v in slice_snapshot.get("first_seen_ticks", {}).items()
        }
        last_seen_ticks = {
            str(k): int(v) for k, v in slice_snapshot.get("last_seen_ticks", {}).items()
        }
        slice_preclaimed_colors = {
            str(item.get("target_color", "")).lower()
            for item in beacon_claims
            if str(item.get("target_color", "")).strip()
        }
    first_seen_tick = None
    wall_source = (
        "learned_front_blocked"
        if use_learned_wall_source
        else (
            "learned_action_outcome"
            if use_learned_action_source
            else ("privileged_manifest_grid" if args.wall_aware_planner else "diagnostic_only")
        )
    )
    wall_metrics: dict[str, Any] = {
        "enabled": bool(args.wall_aware_planner),
        "source": wall_source,
        "fully_learned_runtime_contract": bool(args.fully_learned_runtime_contract),
        "generalized_runtime_contract": bool(args.generalized_runtime_contract),
        "fully_learned_runtime_contract_report": fully_learned_contract_report,
        "slice_benchmark": bool(slice_start is not None or slice_snapshot is not None),
        "slice_start": (
            (
                {
                    "path": str(args.slice_snapshot_input),
                    "start_tick": int(slice_snapshot.get("next_tick", 0)),
                    "start_xy": [
                        float(v) for v in np.asarray(slice_snapshot.get("pos_xyz", [0.0, 0.0]))[:2]
                    ],
                    "start_yaw": float(slice_snapshot.get("yaw_rad", 0.0)),
                    "snapshot": True,
                    "source_result": slice_snapshot.get("source_result"),
                }
                if slice_snapshot is not None
                else None
            )
            if slice_start is None
            else {
                key: value
                for key, value in slice_start.items()
                if key not in {"preload_log"}
            }
        ),
        "slice_preclaimed_colors": sorted(slice_preclaimed_colors),
        "slice_active_target_color": (
            (
                None
                if slice_start is None and slice_snapshot is None
                else str(target_sequence[target_index]).lower()
            )
        ),
        "slice_tick_offset": int(slice_tick_offset),
        "slice_feature_max_ticks": int(feature_max_ticks),
        "explore_goal_policy": str(args.explore_goal_policy),
        "explore_yaw_bearing_threshold": float(args.explore_yaw_bearing_threshold),
        "explore_forward_bearing_threshold": float(args.explore_forward_bearing_threshold),
        "explore_lookahead_m": float(args.explore_lookahead_m),
        "explore_forward_primitive": str(args.explore_forward_primitive),
        "explore_coverage_lookahead_cells": int(args.explore_coverage_lookahead_cells),
        "explore_dfs_neighbor_order": str(args.explore_dfs_neighbor_order),
        "explore_scan_interval": int(args.explore_scan_interval),
        "explore_scan_len": int(args.explore_scan_len),
        "explore_scan_primitive": str(args.explore_scan_primitive),
        "explore_route_waypoints": [
            [_round_float(x), _round_float(y)] for x, y in explore_route_waypoints
        ],
        "explore_route_start_after_claims": int(args.explore_route_start_after_claims),
        "explore_route_advance_m": float(args.explore_route_advance_m),
        "explore_standoff_route": bool(args.explore_standoff_route),
        "explore_standoff_m": float(args.explore_standoff_m),
        "explore_standoff_lookahead_m": float(args.explore_standoff_lookahead_m),
        "explore_standoff_replan_interval": int(args.explore_standoff_replan_interval),
        "explore_standoff_candidates": int(args.explore_standoff_candidates),
        "explore_standoff_arrival_m": float(args.explore_standoff_arrival_m),
        "explore_standoff_path_spacing_m": float(args.explore_standoff_path_spacing_m),
        "explore_standoff_clearance_weight": float(args.explore_standoff_clearance_weight),
        "explore_standoff_clearance_target_m": float(args.explore_standoff_clearance_target_m),
        "explore_standoff_body_route_clearance_weight": float(
            args.explore_standoff_body_route_clearance_weight
        ),
        "explore_standoff_body_route_clearance_target_m": float(
            args.explore_standoff_body_route_clearance_target_m
        ),
        "explore_standoff_body_route_ignore_start_m": float(
            args.explore_standoff_body_route_ignore_start_m
        ),
        "explore_standoff_cardinal_route": bool(args.explore_standoff_cardinal_route),
        "explore_standoff_corner_guard": bool(args.explore_standoff_corner_guard),
        "explore_standoff_corner_commit_m": float(args.explore_standoff_corner_commit_m),
        "explore_standoff_corner_standoff_m": float(args.explore_standoff_corner_standoff_m),
        "explore_standoff_allow_arcs": bool(args.explore_standoff_allow_arcs),
        "explore_standoff_arc_min_bearing": (
            None
            if args.explore_standoff_arc_min_bearing is None
            else float(args.explore_standoff_arc_min_bearing)
        ),
        "explore_standoff_arc_max_bearing": (
            None
            if args.explore_standoff_arc_max_bearing is None
            else float(args.explore_standoff_arc_max_bearing)
        ),
        "explore_standoff_arc_min_target_dist_m": float(
            args.explore_standoff_arc_min_target_dist_m
        ),
        "explore_standoff_arc_bearing_suppressions": 0,
        "explore_standoff_arc_target_dist_suppressions": 0,
        "explore_standoff_heading_mode": str(args.explore_standoff_heading_mode),
        "explore_standoff_heading_lookahead_m": float(args.explore_standoff_heading_lookahead_m),
        "explore_standoff_prefix_snap_start": bool(args.explore_standoff_prefix_snap_start),
        "explore_standoff_snap_start_min_dist_m": float(args.explore_standoff_snap_start_min_dist_m),
        "explore_standoff_tangent_heading_ticks": 0,
        "explore_standoff_snap_start_prefixes": 0,
        "explore_standoff_corner_guard_caps": 0,
        "explore_standoff_corner_standoff_caps": 0,
        "explore_standoff_body_check": bool(args.explore_standoff_body_check),
        "explore_standoff_body_lookahead_m": float(args.explore_standoff_body_lookahead_m),
        "explore_standoff_body_min_clearance_m": float(args.explore_standoff_body_min_clearance_m),
        "explore_standoff_body_recovery_clearance_m": (
            None
            if args.explore_standoff_body_recovery_clearance_m is None
            else float(args.explore_standoff_body_recovery_clearance_m)
        ),
        "explore_standoff_intent_smoothing": bool(args.explore_standoff_intent_smoothing),
        "explore_standoff_sticky_target_ticks": int(args.explore_standoff_sticky_target_ticks),
        "explore_standoff_sticky_target_release_m": float(
            args.explore_standoff_sticky_target_release_m
        ),
        "explore_standoff_yaw_enter_threshold": (
            None
            if args.explore_standoff_yaw_enter_threshold is None
            else float(args.explore_standoff_yaw_enter_threshold)
        ),
        "explore_standoff_yaw_exit_threshold": (
            None
            if args.explore_standoff_yaw_exit_threshold is None
            else float(args.explore_standoff_yaw_exit_threshold)
        ),
        "explore_standoff_yaw_flip_threshold": (
            None
            if args.explore_standoff_yaw_flip_threshold is None
            else float(args.explore_standoff_yaw_flip_threshold)
        ),
        "explore_standoff_intent_target_holds": 0,
        "explore_standoff_intent_target_releases": 0,
        "explore_standoff_yaw_holds": 0,
        "explore_standoff_yaw_flip_suppressions": 0,
        "explore_standoff_yaw_exits": 0,
        "wall_body_forward_m": float(args.wall_body_forward_m),
        "wall_body_half_width_m": float(args.wall_body_half_width_m),
        "wall_body_probe_margin_m": float(args.wall_body_probe_margin_m),
        "explore_standoff_route_until_area_logit": (
            None
            if args.explore_standoff_route_until_area_logit is None
            else float(args.explore_standoff_route_until_area_logit)
        ),
        "explore_standoff_release_on_seen": bool(args.explore_standoff_release_on_seen),
        "explore_standoff_release_m": float(args.explore_standoff_release_m),
        "explore_standoff_release_min_seen_ticks": int(args.explore_standoff_release_min_seen_ticks),
        "explore_standoff_releases_on_seen": 0,
        "explore_reset_on_claim": bool(args.explore_reset_on_claim),
        "explore_clear_visited_on_claim": bool(args.explore_clear_visited_on_claim),
        "explore_claim_route_resets": 0,
        "claim_min_seen_ticks": int(args.claim_min_seen_ticks),
        "claim_near_area_logit": (
            None if args.claim_near_area_logit is None else float(args.claim_near_area_logit)
        ),
        "claim_near_area_logit_by_color": dict(sorted(claim_near_area_by_color.items())),
        "claim_near_bearing": float(args.claim_near_bearing),
        "claim_near_bearing_by_color": dict(sorted(claim_near_bearing_by_color.items())),
        "claim_near_min_seen_ticks": int(args.claim_near_min_seen_ticks),
        "claim_near_min_seen_ticks_by_color": dict(sorted(claim_near_min_seen_by_color.items())),
        "claim_success_proxy_area_logit": (
            None
            if args.claim_success_proxy_area_logit is None
            else float(args.claim_success_proxy_area_logit)
        ),
        "claim_success_proxy_area_logit_by_color": dict(
            sorted(claim_success_proxy_area_by_color.items())
        ),
        "claim_success_proxy_bearing": (
            None
            if args.claim_success_proxy_bearing is None
            else float(args.claim_success_proxy_bearing)
        ),
        "claim_success_proxy_bearing_by_color": dict(
            sorted(claim_success_proxy_bearing_by_color.items())
        ),
        "claim_success_model_checkpoint": (
            None if args.claim_success_model_checkpoint is None else str(args.claim_success_model_checkpoint)
        ),
        "claim_success_model_threshold": (
            None if claim_success_model_threshold is None else float(claim_success_model_threshold)
        ),
        "claim_success_model_threshold_by_color": dict(
            sorted(claim_success_model_threshold_by_color.items())
        ),
        "claim_success_model_evaluations": 0,
        "claim_success_model_rejections": 0,
        "claim_success_model_positive_trigger": bool(args.claim_success_model_positive_trigger),
        "claim_success_model_trigger_claims": 0,
        "claim_success_proxy_rejections": 0,
        "claim_success_proxy_opportunistic_rejections": 0,
        "claim_contact_area_logit": (
            None if args.claim_contact_area_logit is None else float(args.claim_contact_area_logit)
        ),
        "claim_contact_bearing": float(args.claim_contact_bearing),
        "claim_contact_min_seen_ticks": int(args.claim_contact_min_seen_ticks),
        "multi_target_success_requires_claim_distance": bool(
            args.multi_target_success_requires_claim_distance
        ),
        "claim_stalled_area_logit": (
            None if args.claim_stalled_area_logit is None else float(args.claim_stalled_area_logit)
        ),
        "claim_stalled_bearing": float(args.claim_stalled_bearing),
        "claim_stalled_min_seen_ticks": int(args.claim_stalled_min_seen_ticks),
        "claim_stalled_clearance_prob": float(args.claim_stalled_clearance_prob),
        "claim_stalled_latch_ticks": int(args.claim_stalled_latch_ticks),
        "claim_stalled_visual_armed_ticks": 0,
        "multi_target_switch_policy": str(args.multi_target_switch_policy),
        "multi_target_switch_conf": (
            float(args.seen_conf)
            if args.multi_target_switch_conf is None
            else float(args.multi_target_switch_conf)
        ),
        "multi_target_switch_area_logit": float(args.multi_target_switch_area_logit),
        "learned_target_scheduler_checkpoint": (
            None
            if args.learned_target_scheduler_checkpoint is None
            else str(args.learned_target_scheduler_checkpoint)
        ),
        "learned_target_scheduler_ticks": 0,
        "learned_target_scheduler_switches": 0,
        "learned_target_scheduler_masked_claimed_colors": True,
        "learned_target_scheduler_color_vocab": list(learned_target_scheduler_color_vocab),
        "multi_target_seen_switch_min_area_logit": (
            None
            if args.multi_target_seen_switch_min_area_logit is None
            else float(args.multi_target_seen_switch_min_area_logit)
        ),
        "multi_target_stale_seen_switch_after_frontier_noops": int(
            args.multi_target_stale_seen_switch_after_frontier_noops
        ),
        "multi_target_stale_seen_switch_max_age_ticks": int(
            args.multi_target_stale_seen_switch_max_age_ticks
        ),
        "multi_target_opportunistic_claims": bool(
            args.multi_target_opportunistic_claims
        ),
        "multi_target_opportunistic_claim_area_logit": (
            None
            if args.multi_target_opportunistic_claim_area_logit is None
            else float(args.multi_target_opportunistic_claim_area_logit)
        ),
        "multi_target_opportunistic_claim_bearing": (
            None
            if args.multi_target_opportunistic_claim_bearing is None
            else float(args.multi_target_opportunistic_claim_bearing)
        ),
        "multi_target_opportunistic_claim_min_visible_ticks": int(
            args.multi_target_opportunistic_claim_min_visible_ticks
        ),
        "target_opportunistic_claim_candidate_ticks": 0,
        "target_opportunistic_claims": 0,
        "log_color_readouts": bool(args.log_color_readouts),
        "target_switch_candidate_ticks": 0,
        "target_switches": 0,
        "target_switch_rejections_conf": 0,
        "target_switch_rejections_read": 0,
        "target_switch_rejections_area": 0,
        "target_stale_seen_switch_candidate_ticks": 0,
        "target_stale_seen_switches": 0,
        "target_stale_seen_switch_rejections_age": 0,
        "weak_memory_seek_conf": float(args.weak_memory_seek_conf),
        "weak_memory_seek_area_logit": (
            None if args.weak_memory_seek_area_logit is None else float(args.weak_memory_seek_area_logit)
        ),
        "weak_memory_seek_colors": [
            item.strip().lower()
            for item in str(args.weak_memory_seek_colors).split(",")
            if item.strip()
        ],
        "weak_memory_seek_force_explore_on_recovery": bool(
            args.weak_memory_seek_force_explore_on_recovery
        ),
        "weak_memory_seek_stall_streak": int(args.weak_memory_seek_stall_streak),
        "weak_memory_seek_yaw_loop_streak": int(args.weak_memory_seek_yaw_loop_streak),
        "weak_memory_seek_yaw_loop_max_displacement_m": float(
            args.weak_memory_seek_yaw_loop_max_displacement_m
        ),
        "weak_memory_seek_explore_cooldown_ticks": int(args.weak_memory_seek_explore_cooldown_ticks),
        "weak_memory_seek_recoveries": 0,
        "weak_memory_seek_yaw_loop_events": 0,
        "weak_memory_seek_yaw_loop_recoveries": 0,
        "weak_memory_seek_immediate_explore_recoveries": 0,
        "weak_memory_seek_forced_explore_ticks": 0,
        "weak_memory_seek_stall_events": 0,
        "target_pursuit_stale_ticks": int(args.target_pursuit_stale_ticks),
        "target_pursuit_stale_states": sorted(target_pursuit_stale_states),
        "target_pursuit_stale_explore_cooldown_ticks": int(
            args.target_pursuit_stale_explore_cooldown_ticks
        ),
        "target_pursuit_stale_window_ticks": int(args.target_pursuit_stale_window_ticks),
        "target_pursuit_stale_suppress_color_ticks": int(
            args.target_pursuit_stale_suppress_color_ticks
        ),
        "target_pursuit_stale_candidate_ticks": 0,
        "target_pursuit_stale_claim_proxy_pending_ticks": 0,
        "target_pursuit_stale_window_candidate_ticks_max": 0,
        "target_pursuit_stale_recoveries": 0,
        "target_pursuit_stale_window_recoveries": 0,
        "target_pursuit_stale_forced_explore_ticks": 0,
        "target_pursuit_stale_color_suppressions": 0,
        "target_pursuit_stale_suppressed_active_ticks": 0,
        "target_pursuit_stale_switches_from_suppressed": 0,
        "target_pursuit_stale_switch_rejections_suppressed": 0,
        "seen_read_threshold": (
            None if args.seen_read_threshold is None else float(args.seen_read_threshold)
        ),
        "seen_read_gate_low_ticks": 0,
        "primitive_outcome_checkpoint": (
            None if args.primitive_outcome_checkpoint is None else str(args.primitive_outcome_checkpoint)
        ),
        "primitive_post_claim_outcome_checkpoint": (
            None
            if args.primitive_post_claim_outcome_checkpoint is None
            else str(args.primitive_post_claim_outcome_checkpoint)
        ),
        "primitive_post_claim_outcome_ticks": 0,
        "wall_guard_states": sorted(wall_guard_states),
        "wall_guard_post_claim_states": sorted(wall_guard_post_claim_states),
        "wall_guard_post_claim_min_claims": int(args.wall_guard_post_claim_min_claims),
        "primitive_outcome_preserve_turn_requests": bool(args.primitive_outcome_preserve_turn_requests),
        "primitive_outcome_preserve_turn_until_first_claim": bool(
            args.primitive_outcome_preserve_turn_until_first_claim
        ),
        "primitive_outcome_preserve_turn_post_claim_suppressed_ticks": 0,
        "primitive_outcome_preserve_arc_requests": bool(args.primitive_outcome_preserve_arc_requests),
        "primitive_outcome_blocked_hard_veto_use_guard_bearing": bool(
            args.primitive_outcome_blocked_hard_veto_use_guard_bearing
        ),
        "primitive_outcome_blocked_hard_veto_after_first_claim": bool(
            args.primitive_outcome_blocked_hard_veto_after_first_claim
        ),
        "primitive_outcome_blocked_hard_veto_pre_claim_suppressed_ticks": 0,
        "primitive_outcome_preserve_backward_requests": bool(args.primitive_outcome_preserve_backward_requests),
        "primitive_outcome_preserve_backward_clearance_margin": (
            None
            if args.primitive_outcome_preserve_backward_clearance_margin is None
            else float(args.primitive_outcome_preserve_backward_clearance_margin)
        ),
        "primitive_outcome_preserve_turn_states": sorted(primitive_outcome_preserve_turn_states),
        "primitive_outcome_turn_body_rerank_primitives": sorted(
            primitive_outcome_turn_body_rerank_primitives
        ),
        "primitive_outcome_preserve_straight_states": sorted(primitive_outcome_preserve_straight_states),
        "primitive_outcome_forward_progress_floor": (
            None
            if args.primitive_outcome_forward_progress_floor is None
            else float(args.primitive_outcome_forward_progress_floor)
        ),
        "primitive_outcome_forward_progress_floor_states": sorted(
            primitive_outcome_forward_progress_floor_states
        ),
        "primitive_outcome_progress_floor_min_blocked_prob": (
            None
            if args.primitive_outcome_progress_floor_min_blocked_prob is None
            else float(args.primitive_outcome_progress_floor_min_blocked_prob)
        ),
        "primitive_outcome_progress_floor_force_below": (
            None
            if args.primitive_outcome_progress_floor_force_below is None
            else float(args.primitive_outcome_progress_floor_force_below)
        ),
        "primitive_outcome_forward_progress_penalty": float(args.primitive_outcome_forward_progress_penalty),
        "primitive_outcome_low_progress_hard_veto": bool(args.primitive_outcome_low_progress_hard_veto),
        "primitive_outcome_low_progress_hard_veto_primitives": sorted(
            primitive_outcome_low_progress_hard_veto_primitives
        ),
        "primitive_outcome_blocked_hard_veto": bool(args.primitive_outcome_blocked_hard_veto),
        "primitive_outcome_blocked_hard_veto_primitives": sorted(
            primitive_outcome_blocked_hard_veto_primitives
        ),
        "primitive_outcome_blocked_hard_veto_selected_primitives": sorted(
            primitive_outcome_blocked_hard_veto_selected_primitives
        ),
        "primitive_outcome_blocked_hard_veto_max_abs_bearing": (
            None
            if args.primitive_outcome_blocked_hard_veto_max_abs_bearing is None
            else float(args.primitive_outcome_blocked_hard_veto_max_abs_bearing)
        ),
        "primitive_outcome_progress_floor_prefer_yaw": bool(args.primitive_outcome_progress_floor_prefer_yaw),
        "primitive_clearance_checkpoint": (
            None if args.primitive_clearance_checkpoint is None else str(args.primitive_clearance_checkpoint)
        ),
        "primitive_aux_clearance_checkpoint": (
            None
            if args.primitive_aux_clearance_checkpoint is None
            else str(args.primitive_aux_clearance_checkpoint)
        ),
        "primitive_clearance_threshold": clearance_threshold,
        "body_clearance_target_servo": bool(args.body_clearance_target_servo),
        "body_clearance_target_area_logit": float(args.body_clearance_target_area_logit),
        "body_clearance_target_bearing": float(args.body_clearance_target_bearing),
        "body_clearance_target_forward_primitive": str(args.body_clearance_target_forward_primitive),
        "body_clearance_latch_ticks": int(args.body_clearance_latch_ticks),
        "body_clearance_learned_prob_floor": float(args.body_clearance_learned_prob_floor),
        "body_clearance_learned_prob_weight": float(args.body_clearance_learned_prob_weight),
        "body_clearance_near_forward_prob_floor": (
            None
            if args.body_clearance_near_forward_prob_floor is None
            else float(args.body_clearance_near_forward_prob_floor)
        ),
        "body_clearance_near_forward_prob_weight": (
            None
            if args.body_clearance_near_forward_prob_weight is None
            else float(args.body_clearance_near_forward_prob_weight)
        ),
        "body_clearance_learned_min_area_logit": (
            None
            if args.body_clearance_learned_min_area_logit is None
            else float(args.body_clearance_learned_min_area_logit)
        ),
        "body_clearance_near_yaw_prob_floor": (
            None
            if args.body_clearance_near_yaw_prob_floor is None
            else float(args.body_clearance_near_yaw_prob_floor)
        ),
        "body_clearance_near_yaw_prob_weight": float(args.body_clearance_near_yaw_prob_weight),
        "body_clearance_yaw_always": bool(args.body_clearance_yaw_always),
        "body_clearance_hard_veto_prob": float(args.body_clearance_hard_veto_prob),
        "body_clearance_hard_veto_margin": float(args.body_clearance_hard_veto_margin),
        "body_clearance_hard_veto_replacement_cap": float(
            args.body_clearance_hard_veto_replacement_cap
        ),
        "body_clearance_target_area_hard_veto_prob": float(
            args.body_clearance_target_area_hard_veto_prob
        ),
        "body_clearance_target_area_hard_veto_min_area_logit": (
            float(args.body_clearance_target_area_logit)
            if args.body_clearance_target_area_hard_veto_min_area_logit is None
            else float(args.body_clearance_target_area_hard_veto_min_area_logit)
        ),
        "body_clearance_target_area_hard_veto_ticks": 0,
        "body_clearance_hard_veto_primitives": sorted(body_clearance_hard_veto_primitives),
        "body_clearance_aux_switch_hard_veto_primitives": sorted(
            body_clearance_aux_switch_hard_veto_primitives
        ),
        "body_clearance_hard_veto_selected_primitives": sorted(
            body_clearance_hard_veto_selected_primitives or _TRANSLATING_PRIMITIVES
        ),
        "body_clearance_hard_veto_hold_escape_after": int(
            args.body_clearance_hard_veto_hold_escape_after
        ),
        "body_clearance_hard_veto_hold_escape_max_clearance_prob": float(
            args.body_clearance_hard_veto_hold_escape_max_clearance_prob
        ),
        "body_clearance_hard_veto_hold_escape_primitives": sorted(
            body_clearance_hard_veto_hold_escape_primitives
        ),
        "body_clearance_hard_veto_hold_escape_states": sorted(
            body_clearance_hard_veto_hold_escape_states
        ),
        "body_clearance_hard_veto_hold_escape_override_primitives": sorted(
            body_clearance_hard_veto_hold_escape_override_primitives
        ),
        "body_clearance_hard_veto_hold_escape_override_states": sorted(
            body_clearance_hard_veto_hold_escape_override_states
        ),
        "body_clearance_hard_veto_hold_escape_override_min_claimed_count": max(
            0,
            int(args.body_clearance_hard_veto_hold_escape_override_min_claimed_count),
        ),
        "body_clearance_hard_veto_hold_escape_override_max_clearance_prob": (
            None
            if args.body_clearance_hard_veto_hold_escape_override_max_clearance_prob is None
            else float(args.body_clearance_hard_veto_hold_escape_override_max_clearance_prob)
        ),
        "body_clearance_hard_veto_hold_escape_override_min_current_clearance_m": (
            None
            if args.body_clearance_hard_veto_hold_escape_override_min_current_clearance_m is None
            else float(args.body_clearance_hard_veto_hold_escape_override_min_current_clearance_m)
        ),
        "body_clearance_hard_veto_hold_escape_min_projected_clearance_m": (
            None
            if args.body_clearance_hard_veto_hold_escape_min_projected_clearance_m is None
            else float(args.body_clearance_hard_veto_hold_escape_min_projected_clearance_m)
        ),
        "body_clearance_hard_veto_hold_escape_min_projected_improvement_m": float(
            args.body_clearance_hard_veto_hold_escape_min_projected_improvement_m
        ),
        "body_clearance_veto_configured_min_claimed_count": int(
            configured_body_clearance_veto_min_claimed_count
        ),
        "body_clearance_veto_min_claimed_count": int(
            effective_body_clearance_veto_min_claimed_count
        ),
        "body_clearance_veto_min_claimed_count_clamped_for_full_run": bool(
            full_from_spawn_contract
            and configured_body_clearance_veto_min_claimed_count
            != effective_body_clearance_veto_min_claimed_count
        ),
        "body_clearance_aux_veto_prob": float(args.body_clearance_aux_veto_prob),
        "body_clearance_aux_veto_primary_max_prob": float(
            args.body_clearance_aux_veto_primary_max_prob
        ),
        "body_clearance_aux_veto_margin": float(args.body_clearance_aux_veto_margin),
        "body_clearance_aux_veto_replacement_cap": float(
            args.body_clearance_aux_veto_replacement_cap
        ),
        "body_clearance_aux_veto_primitives": sorted(body_clearance_aux_veto_primitives),
        "body_clearance_aux_veto_selected_primitives": sorted(
            body_clearance_aux_veto_selected_primitives
        ),
        "primitive_aux_clearance_switch_current_body_risk": bool(
            args.primitive_aux_clearance_switch_current_body_risk
        ),
        "primitive_aux_clearance_switch_threshold": (
            current_body_threshold
            if args.primitive_aux_clearance_switch_threshold is None
            else float(args.primitive_aux_clearance_switch_threshold)
        ),
        "primitive_aux_clearance_switch_configured_min_claimed_count": int(
            configured_aux_clearance_switch_min_claimed_count
        ),
        "primitive_aux_clearance_switch_min_claimed_count": int(
            effective_aux_clearance_switch_min_claimed_count
        ),
        "primitive_aux_clearance_switch_min_claimed_count_clamped_for_full_run": bool(
            full_from_spawn_contract
            and configured_aux_clearance_switch_min_claimed_count
            != effective_aux_clearance_switch_min_claimed_count
        ),
        "primitive_aux_clearance_switch_latch_ticks": int(
            args.primitive_aux_clearance_switch_latch_ticks
        ),
        "primitive_aux_clearance_switch_policy_features": bool(
            args.primitive_aux_clearance_switch_policy_features
        ),
        "body_clearance_aux_switch_enable": bool(args.body_clearance_aux_switch_enable),
        "body_clearance_aux_switch_ignore_min_area": bool(
            args.body_clearance_aux_switch_ignore_min_area
        ),
        "body_clearance_aux_switch_arc_sweep_veto_prob": float(
            args.body_clearance_aux_switch_arc_sweep_veto_prob
        ),
        "body_clearance_aux_switch_arc_sweep_veto_selected_primitives": sorted(
            body_clearance_aux_switch_arc_sweep_veto_selected_primitives
        ),
        "body_clearance_aux_switch_enabled_ticks": 0,
        "body_clearance_aux_switch_min_area_ignored_ticks": 0,
        "body_clearance_arc_sweep_vetoes": 0,
        "primitive_aux_clearance_switch_ticks": 0,
        "primitive_aux_clearance_switch_suppressed_ticks": 0,
        "body_clearance_saturated_veto_prob": float(args.body_clearance_saturated_veto_prob),
        "body_clearance_saturated_veto_spread": float(args.body_clearance_saturated_veto_spread),
        "body_clearance_saturated_veto_primitives": sorted(body_clearance_saturated_veto_primitives),
        "body_clearance_saturated_veto_selected_primitives": sorted(
            body_clearance_saturated_veto_selected_primitives
        ),
        "body_clearance_yaw_contact_veto_prob": float(args.body_clearance_yaw_contact_veto_prob),
        "body_clearance_yaw_direction_veto_prob": float(args.body_clearance_yaw_direction_veto_prob),
        "body_clearance_yaw_direction_veto_margin": float(args.body_clearance_yaw_direction_veto_margin),
        "body_clearance_current_contact_escape_m": (
            None
            if args.body_clearance_current_contact_escape_m is None
            else float(args.body_clearance_current_contact_escape_m)
        ),
        "body_clearance_current_contact_escape_m_by_primitive": dict(
            sorted(body_clearance_current_contact_escape_m_by_primitive.items())
        ),
        "body_clearance_current_contact_escape_min_projected_clearance_m": (
            None
            if args.body_clearance_current_contact_escape_min_projected_clearance_m is None
            else float(args.body_clearance_current_contact_escape_min_projected_clearance_m)
        ),
        "body_clearance_current_contact_escape_min_projected_improvement_m": float(
            args.body_clearance_current_contact_escape_min_projected_improvement_m
        ),
        "body_clearance_current_contact_escape_min_streak": max(
            1,
            int(args.body_clearance_current_contact_escape_min_streak),
        ),
        "body_clearance_current_contact_escape_cooldown_ticks": max(
            0,
            int(args.body_clearance_current_contact_escape_cooldown_ticks),
        ),
        "body_clearance_current_contact_escape_min_claimed_count": max(
            0,
            int(args.body_clearance_current_contact_escape_min_claimed_count),
        ),
        "body_clearance_current_contact_escape_states": sorted(
            body_clearance_current_contact_escape_states
        ),
        "body_clearance_current_contact_escape_target_colors": sorted(
            body_clearance_current_contact_escape_target_colors
        ),
        "body_clearance_current_contact_escape_primitives": sorted(
            body_clearance_current_contact_escape_primitives
        ),
        "body_clearance_current_contact_escape_replacements": sorted(
            body_clearance_current_contact_escape_replacements
        ),
        "body_clearance_current_contact_escape_replacement_cap": float(
            args.body_clearance_current_contact_escape_replacement_cap
        ),
        "body_clearance_current_contact_escape_require_replacement_under_cap": bool(
            args.body_clearance_current_contact_escape_require_replacement_under_cap
        ),
        "body_clearance_current_contact_escape_min_area_logit": (
            None
            if args.body_clearance_current_contact_escape_min_area_logit is None
            else float(args.body_clearance_current_contact_escape_min_area_logit)
        ),
        "body_clearance_current_contact_escape_min_area_states": sorted(
            body_clearance_current_contact_escape_min_area_states
        ),
        "body_clearance_near_arc_penalty": float(args.body_clearance_near_arc_penalty),
        "body_clearance_risk_escape_threshold": float(args.body_clearance_risk_escape_threshold),
        "body_clearance_risk_escape_blocks": int(args.body_clearance_risk_escape_blocks),
        "body_clearance_risk_escape_cooldown_ticks": int(args.body_clearance_risk_escape_cooldown_ticks),
        "body_clearance_risk_escape_states": sorted(body_clearance_risk_escape_states),
        "current_body_risk_checkpoint": (
            None
            if args.current_body_risk_checkpoint is None
            else str(args.current_body_risk_checkpoint)
        ),
        "current_body_risk_threshold": current_body_threshold,
        "current_body_risk_min_claimed_count": max(
            0,
            int(args.current_body_risk_min_claimed_count),
        ),
        "current_body_risk_recovery_blocks": int(args.current_body_risk_recovery_blocks),
        "proprio_contact_detector_checkpoint": (
            None
            if args.proprio_contact_detector_checkpoint is None
            else str(args.proprio_contact_detector_checkpoint)
        ),
        "proprio_contact_escape_threshold": float(args.proprio_contact_escape_threshold),
        "proprio_contact_escape_streak": int(args.proprio_contact_escape_streak),
        "proprio_contact_escape_blocks": int(args.proprio_contact_escape_blocks),
        "proprio_contact_escape_cooldown_ticks": int(args.proprio_contact_escape_cooldown_ticks),
        "proprio_contact_escape_states": str(args.proprio_contact_escape_states),
        "proprio_contact_map_blocks": bool(args.proprio_contact_map_blocks),
        "history_risk_checkpoint": (
            None
            if args.history_risk_checkpoint is None
            else str(args.history_risk_checkpoint)
        ),
        "history_risk_veto_threshold": float(args.history_risk_veto_threshold),
        "history_risk_veto_primitives": str(args.history_risk_veto_primitives),
        "history_risk_replacements": str(args.history_risk_replacements),
        "history_risk_replacement_cap": float(args.history_risk_replacement_cap),
        "history_risk_states": str(args.history_risk_states),
        "current_body_risk_recovery_selected_prob_floor": (
            None
            if args.current_body_risk_recovery_selected_prob_floor is None
            else float(args.current_body_risk_recovery_selected_prob_floor)
        ),
        "current_body_risk_recovery_selected_primitives": sorted(
            current_body_risk_recovery_selected_primitives
        ),
        "current_body_risk_preserve_yaw": bool(args.current_body_risk_preserve_yaw),
        "current_body_risk_preserve_yaw_threshold": (
            current_body_threshold
            if args.current_body_risk_preserve_yaw_threshold is None
            else float(args.current_body_risk_preserve_yaw_threshold)
        ),
        "current_body_risk_preserve_yaw_min_area_logit": (
            (
                None
                if args.current_body_risk_min_area_logit is None
                else float(args.current_body_risk_min_area_logit)
            )
            if args.current_body_risk_preserve_yaw_min_area_logit is None
            else float(args.current_body_risk_preserve_yaw_min_area_logit)
        ),
        "current_body_risk_preserve_yaw_max_clearance_prob": (
            None
            if args.current_body_risk_preserve_yaw_max_clearance_prob is None
            else float(args.current_body_risk_preserve_yaw_max_clearance_prob)
        ),
        "current_body_risk_clearance_rerank": bool(args.current_body_risk_clearance_rerank),
        "current_body_risk_clearance_rerank_threshold": (
            current_body_threshold
            if args.current_body_risk_clearance_rerank_threshold is None
            else float(args.current_body_risk_clearance_rerank_threshold)
        ),
        "current_body_risk_clearance_rerank_min_area_logit": (
            (
                None
                if args.current_body_risk_min_area_logit is None
                else float(args.current_body_risk_min_area_logit)
            )
            if args.current_body_risk_clearance_rerank_min_area_logit is None
            else float(args.current_body_risk_clearance_rerank_min_area_logit)
        ),
        "current_body_risk_clearance_rerank_selected_prob_floor": (
            None
            if args.current_body_risk_clearance_rerank_selected_prob_floor is None
            else float(args.current_body_risk_clearance_rerank_selected_prob_floor)
        ),
        "current_body_risk_clearance_rerank_selected_primitives": sorted(
            current_body_risk_clearance_rerank_selected_primitives
        ),
        "current_body_risk_clearance_rerank_primitives": sorted(
            current_body_risk_clearance_rerank_primitives
        ),
        "current_body_risk_min_area_logit": (
            None
            if args.current_body_risk_min_area_logit is None
            else float(args.current_body_risk_min_area_logit)
        ),
        "current_body_risk_cooldown_ticks": int(args.current_body_risk_cooldown_ticks),
        "current_body_risk_states": sorted(current_body_risk_states),
        "body_clearance_target_interventions": 0,
        "body_clearance_latched_interventions": 0,
        "body_clearance_learned_penalty_ticks": 0,
        "body_clearance_learned_vetoes": 0,
        "body_clearance_hard_vetoes": 0,
        "body_clearance_hard_veto_hold_ticks": 0,
        "body_clearance_hard_veto_hold_streak_max": 0,
        "body_clearance_hard_veto_hold_escapes": 0,
        "body_clearance_hard_veto_hold_escape_overrides": 0,
        "body_clearance_hard_veto_hold_escape_no_candidate_ticks": 0,
        "body_clearance_hard_veto_hold_escape_capped_candidates": 0,
        "body_clearance_hard_veto_hold_escape_projection_rejections": 0,
        "body_clearance_veto_claim_gate_suppressed_ticks": 0,
        "body_clearance_veto_claim_gate_suppressed_high_risk_ticks": 0,
        "body_clearance_aux_vetoes": 0,
        "body_clearance_aux_veto_suppressed_ticks": 0,
        "body_clearance_saturated_vetoes": 0,
        "body_clearance_yaw_direction_vetoes": 0,
        "body_clearance_yaw_contact_vetoes": 0,
        "body_clearance_current_contact_escape_low_clearance_ticks": 0,
        "body_clearance_current_contact_escape_gate_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_streak_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_cooldown_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_claimed_count_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_state_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_target_suppressed_ticks": 0,
        "body_clearance_current_contact_escape_area_suppressed_ticks": 0,
        "body_clearance_current_contact_escapes": 0,
        "blocked_hard_vetoes": 0,
        "low_progress_hard_vetoes": 0,
        "body_clearance_risk_escapes": 0,
        "current_body_risk_ticks": 0,
        "current_body_risk_recoveries": 0,
        "current_body_risk_recovery_blocks_executed": 0,
        "current_body_risk_recovery_selected_floor_blocks": 0,
        "current_body_risk_recovery_selected_primitive_blocks": 0,
        "current_body_risk_preserve_yaw_overrides": 0,
        "current_body_risk_preserve_yaw_suppressed": 0,
        "current_body_risk_clearance_reranks": 0,
        "current_body_risk_clearance_rerank_selected_floor_blocks": 0,
        "current_body_risk_clearance_rerank_selected_primitive_blocks": 0,
        "current_body_risk_claim_gate_suppressed_ticks": 0,
        "current_body_risk_claim_gate_suppressed_high_risk_ticks": 0,
        "current_body_risk_prob_max": None,
        "primitive_aux_clearance_switch_claim_gate_suppressed_ticks": 0,
        "primitive_aux_clearance_switch_claim_gate_suppressed_high_risk_ticks": 0,
        "primitive_aux_clearance_switch_area_suppressed_ticks": 0,
        "commands_total": 0,
        "forward_requests": 0,
        "blocked_forward_requests": 0,
        "forward_executions": 0,
        "blocked_forward_executions": 0,
        "wall_vetoes": 0,
        "escape_blocks_executed": 0,
        "contact_like_stalls": 0,
        "hard_contact_like_stalls": 0,
        "stuck_recoveries": 0,
        "stall_waypoint_blocks": 0,
        "stall_block_waypoint_enabled": bool(args.wall_stall_block_waypoint),
        "turn_loop_recoveries": 0,
        "turn_loop_waypoint_blocks": 0,
        "turn_loop_block_waypoint_enabled": bool(args.wall_turn_loop_block_waypoint),
        "learned_blocked_waypoint_replans": 0,
        "predicted_blocked_waypoint_replan_enabled": bool(args.wall_predicted_blocked_waypoint_replan),
        "predicted_blocked_waypoint_streak": int(args.wall_predicted_blocked_waypoint_streak),
        "command_smoothing_min_ticks": int(args.command_smoothing_min_ticks),
        "command_smoothing_states": sorted(command_smoothing_states),
        "command_smoothing_primitives": sorted(command_smoothing_primitives),
        "command_smoothing_overrides": 0,
        "command_smoothing_blocked_holds": 0,
        "command_smoothing_opposite_yaw_blocks": 0,
        "learned_safe_stride_primitive": str(args.learned_safe_stride_primitive),
        "learned_safe_stride_from": str(args.learned_safe_stride_from),
        "learned_safe_stride_states": sorted(learned_safe_stride_states),
        "learned_safe_stride_max_blocked_prob": float(args.learned_safe_stride_max_blocked_prob),
        "learned_safe_stride_max_clearance_blocked_prob": (
            None
            if args.learned_safe_stride_max_clearance_blocked_prob is None
            else float(args.learned_safe_stride_max_clearance_blocked_prob)
        ),
        "learned_safe_stride_min_progress_m": float(args.learned_safe_stride_min_progress_m),
        "learned_safe_stride_max_bearing": float(args.learned_safe_stride_max_bearing),
        "learned_safe_stride_upgrades": 0,
        "learned_safe_stride_skips": 0,
        "learned_local_explore_ticks": 0,
        "learned_local_explore_overrides": 0,
        "learned_local_explore_scan_ticks": 0,
        "learned_local_translation_pressure_ticks": 0,
        "learned_local_max_turn_run": 0,
        "learned_wall_follow_side_period": int(args.learned_wall_follow_side_period),
        "learned_wall_follow_initial_side": str(args.learned_wall_follow_initial_side),
        "learned_wall_follow_flip_on_claim": bool(args.learned_wall_follow_flip_on_claim),
        "learned_wall_follow_safe_risk": float(args.learned_wall_follow_safe_risk),
        "learned_wall_follow_progress_floor": float(args.learned_wall_follow_progress_floor),
        "learned_wall_follow_turn_pressure_after": int(args.learned_wall_follow_turn_pressure_after),
        "learned_wall_follow_ticks": 0,
        "learned_wall_follow_overrides": 0,
        "learned_wall_follow_scan_ticks": 0,
        "learned_wall_follow_side_switches": 0,
        "learned_wall_follow_max_turn_run": 0,
        "learned_local_policy_checkpoint": (
            None
            if args.learned_local_policy_checkpoint is None
            else str(args.learned_local_policy_checkpoint)
        ),
        "learned_local_post_claim_policy_checkpoint": (
            None
            if args.learned_local_post_claim_policy_checkpoint is None
            else str(args.learned_local_post_claim_policy_checkpoint)
        ),
        "learned_local_post_claim_policy_min_claims": int(
            args.learned_local_post_claim_policy_min_claims
        ),
        "learned_local_target_policy_checkpoints": {
            color: str(checkpoint.get("source", ""))
            for color, checkpoint in sorted(learned_local_target_policy_checkpoints.items())
        },
        "learned_local_target_policy_priority_over_post_claim": bool(
            args.learned_local_target_policy_priority_over_post_claim
        ),
        "learned_local_target_policy_priority_on_aux_clearance_switch": bool(
            args.learned_local_target_policy_priority_on_aux_clearance_switch
        ),
        "learned_local_policy_states": sorted(learned_local_policy_states),
        "learned_local_policy_post_claim_states": sorted(
            learned_local_policy_post_claim_states
        ),
        "learned_local_dataset_states": sorted(learned_local_dataset_states),
        "learned_local_dataset_output": (
            None
            if args.learned_local_dataset_output is None
            else str(args.learned_local_dataset_output)
        ),
        "learned_local_dataset_label_source": str(args.learned_local_dataset_label_source),
        "learned_local_dataset_min_claimed_count": int(args.learned_local_dataset_min_claimed_count),
        "debug_force_primitive_script": (
            None if args.debug_force_primitive_script is None else str(args.debug_force_primitive_script)
        ),
        "debug_force_primitive_script_ticks": [
            int(tick) for tick in sorted(debug_force_primitive_script)
        ],
        "debug_force_primitive_overrides": 0,
        "learned_local_oracle_standoff_labels": bool(args.learned_local_oracle_standoff_labels),
        "learned_local_oracle_standoff_label_states": sorted(
            learned_local_oracle_standoff_label_states
        ),
        "learned_local_oracle_standoff_label_ticks": 0,
        "learned_local_oracle_standoff_label_overrides": 0,
        "learned_local_oracle_standoff_replans": 0,
        "learned_local_oracle_standoff_plan_failures": 0,
        "learned_local_policy_ticks": 0,
        "learned_local_primary_policy_ticks": 0,
        "learned_local_post_claim_policy_ticks": 0,
        "learned_local_target_policy_ticks": 0,
        "learned_local_target_policy_ticks_by_color": {
            color: 0 for color in sorted(learned_local_target_policy_checkpoints)
        },
        "learned_local_policy_disabled_ticks": 0,
        "learned_local_post_claim_policy_disabled_ticks": 0,
        "learned_local_target_policy_disabled_ticks": 0,
        "learned_local_policy_feature_mismatch_ticks": 0,
        "learned_local_post_claim_policy_feature_mismatch_ticks": 0,
        "learned_local_target_policy_feature_mismatch_ticks": 0,
        "learned_local_policy_overrides": 0,
        "learned_local_post_claim_policy_overrides": 0,
        "learned_local_target_policy_overrides": 0,
        "learned_local_policy_outcome_rerank": bool(args.learned_local_policy_outcome_rerank),
        "learned_local_post_claim_policy_outcome_rerank": str(
            args.learned_local_post_claim_policy_outcome_rerank
        ),
        "learned_local_target_policy_outcome_rerank": str(
            args.learned_local_target_policy_outcome_rerank
        ),
        "learned_local_policy_rerank_top_k": int(args.learned_local_policy_rerank_top_k),
        "learned_local_policy_rerank_policy_weight": float(args.learned_local_policy_rerank_policy_weight),
        "learned_local_post_claim_policy_rerank_policy_weight": (
            None
            if args.learned_local_post_claim_policy_rerank_policy_weight is None
            else float(args.learned_local_post_claim_policy_rerank_policy_weight)
        ),
        "learned_local_target_policy_rerank_policy_weight": (
            None
            if args.learned_local_target_policy_rerank_policy_weight is None
            else float(args.learned_local_target_policy_rerank_policy_weight)
        ),
        "learned_local_policy_rerank_blocked_weight": float(args.learned_local_policy_rerank_blocked_weight),
        "learned_local_policy_rerank_clearance_weight": float(args.learned_local_policy_rerank_clearance_weight),
        "learned_local_policy_rerank_progress_weight": float(args.learned_local_policy_rerank_progress_weight),
        "learned_local_policy_rerank_hard_blocked_penalty": float(
            args.learned_local_policy_rerank_hard_blocked_penalty
        ),
        "learned_local_policy_rerank_backward_penalty": float(
            args.learned_local_policy_rerank_backward_penalty
        ),
        "learned_local_policy_rerank_switch_margin": float(args.learned_local_policy_rerank_switch_margin),
        "learned_local_policy_rerank_protect_top_prob": float(
            args.learned_local_policy_rerank_protect_top_prob
        ),
        "learned_local_policy_rerank_override_min_prob": float(
            args.learned_local_policy_rerank_override_min_prob
        ),
        "learned_local_policy_rerank_bearing_turn_threshold": float(
            args.learned_local_policy_rerank_bearing_turn_threshold
        ),
        "learned_local_policy_rerank_bearing_turn_bonus": float(
            args.learned_local_policy_rerank_bearing_turn_bonus
        ),
        "learned_local_policy_online_map_novelty_weight": float(
            args.learned_local_policy_online_map_novelty_weight
        ),
        "learned_local_policy_online_map_hard_veto": bool(
            args.learned_local_policy_online_map_hard_veto
        ),
        "learned_local_policy_online_map_blocked_penalty": float(
            args.learned_local_policy_online_map_blocked_penalty
        ),
        "learned_local_policy_online_map_turn_scale": float(
            args.learned_local_policy_online_map_turn_scale
        ),
        "learned_local_policy_online_map_claim_repulsion_weight": float(
            args.learned_local_policy_online_map_claim_repulsion_weight
        ),
        "learned_local_policy_online_map_frontier_route_weight": float(
            args.learned_local_policy_online_map_frontier_route_weight
        ),
        "learned_local_policy_online_map_novelty_states": sorted(
            learned_local_policy_online_map_novelty_states
        ),
        "learned_local_policy_online_map_novelty_ticks": 0,
        "learned_local_policy_online_map_novelty_overrides": 0,
        "learned_local_policy_outcome_rerank_overrides": 0,
        "learned_local_policy_outcome_rerank_unsafe_top_ticks": 0,
        "learned_local_policy_explore_state_ticks": 0,
        "learned_local_policy_translation_pressure_after": int(
            args.learned_local_policy_translation_pressure_after
        ),
        "learned_local_policy_translation_pressure_max_blocked_prob": float(
            args.learned_local_policy_translation_pressure_max_blocked_prob
        ),
        "learned_local_policy_translation_pressure_min_progress_m": float(
            args.learned_local_policy_translation_pressure_min_progress_m
        ),
        "learned_local_policy_translation_pressure_primitives": list(
            learned_local_policy_translation_pressure_primitives
        ),
        "learned_local_policy_translation_pressure_states": sorted(
            learned_local_policy_translation_pressure_states
        ),
        "learned_local_policy_translation_pressure_ticks": 0,
        "learned_local_policy_translation_pressure_overrides": 0,
        "learned_local_policy_translation_pressure_noops": 0,
        "learned_local_policy_frontier_pressure_after": int(
            args.learned_local_policy_frontier_pressure_after
        ),
        "learned_local_policy_frontier_pressure_states": sorted(
            learned_local_policy_frontier_pressure_states
        ),
        "learned_local_policy_frontier_pressure_max_blocked_prob": float(
            args.learned_local_policy_frontier_pressure_max_blocked_prob
        ),
        "learned_local_policy_frontier_pressure_min_progress_m": float(
            args.learned_local_policy_frontier_pressure_min_progress_m
        ),
        "learned_local_policy_frontier_pressure_min_route_cells": int(
            args.learned_local_policy_frontier_pressure_min_route_cells
        ),
        "learned_local_policy_frontier_pressure_guard_blocked_penalty": float(
            args.learned_local_policy_frontier_pressure_guard_blocked_penalty
        ),
        "learned_local_policy_frontier_pressure_nonroute_backward_claim_escape": bool(
            args.learned_local_policy_frontier_pressure_nonroute_backward_claim_escape
        ),
        "learned_local_policy_frontier_pressure_prefer_unguarded": bool(
            args.learned_local_policy_frontier_pressure_prefer_unguarded
        ),
        "learned_local_policy_frontier_pressure_map_blocked_backward_claim_escape": bool(
            args.learned_local_policy_frontier_pressure_map_blocked_backward_claim_escape
        ),
        "learned_local_policy_frontier_pressure_guarded_retry_after_noops": int(
            args.learned_local_policy_frontier_pressure_guarded_retry_after_noops
        ),
        "learned_local_policy_frontier_pressure_combined_blocked_retry_after_noops": int(
            args.learned_local_policy_frontier_pressure_combined_blocked_retry_after_noops
        ),
        "learned_local_policy_frontier_pressure_commit": bool(
            args.learned_local_policy_frontier_pressure_commit
        ),
        "learned_local_policy_frontier_pressure_guard_rerank_on_commit": bool(
            args.learned_local_policy_frontier_pressure_guard_rerank_on_commit
        ),
        "learned_local_policy_frontier_pressure_guard_recovery_rerank_on_commit": bool(
            args.learned_local_policy_frontier_pressure_guard_recovery_rerank_on_commit
        ),
        "learned_local_policy_frontier_pressure_guard_recovery_primitives": list(
            learned_local_policy_frontier_pressure_guard_recovery_primitives
        ),
        "learned_local_policy_frontier_pressure_ticks": 0,
        "learned_local_policy_frontier_pressure_overrides": 0,
        "learned_local_policy_frontier_pressure_noops": 0,
        "learned_local_policy_frontier_pressure_guard_commits": 0,
        "learned_local_policy_frontier_pressure_guarded_retries": 0,
        "learned_local_policy_frontier_pressure_combined_blocked_retries": 0,
        "learned_local_online_map_route_replay_guard_override": bool(
            args.learned_local_online_map_route_replay_guard_override
        ),
        "learned_local_online_map_wall_guard_block_source": str(
            args.learned_local_online_map_wall_guard_block_source
        ),
        "learned_local_online_map_current_contact_projection_blocks": bool(
            args.learned_local_online_map_current_contact_projection_blocks
        ),
        "learned_local_online_map_current_contact_projection_blocked_edges": 0,
        "learned_local_online_map_hard_veto_hold_escape_projection_blocks": bool(
            args.learned_local_online_map_hard_veto_hold_escape_projection_blocks
        ),
        "learned_local_online_map_hard_veto_hold_escape_projection_blocked_edges": 0,
        "learned_local_online_map_geometry_veto_hold_blocks": bool(
            args.learned_local_online_map_geometry_veto_hold_blocks
        ),
        "learned_local_online_map_geometry_veto_hold_blocked_edges": 0,
        "learned_local_online_map_route_replay_guard_override_ticks": 0,
        "learned_local_online_map_low_progress_block_m": float(
            args.learned_local_online_map_low_progress_block_m
        ),
        "learned_local_online_map_low_progress_block_ticks": 0,
        "learned_local_online_map_rotation_stall_blocks": 0,
        "stability_guard_events": 0,
        "stability_guard_hold_ticks": 0,
        "learned_local_policy_max_turn_run": 0,
        "learned_local_policy_nonprogress_ticks": 0,
        "learned_local_policy_max_nonprogress_run": 0,
        "learned_local_policy_collected_examples": 0,
        "learned_local_policy_label_mapped_examples": 0,
        "learned_local_policy_skipped_unmapped_examples": 0,
        "learned_local_policy_skipped_feature_dim_examples": 0,
        "learned_local_policy_privileged_explorer_skipped_ticks": 0,
        "learned_local_policy_feature_dim": (
            None
            if learned_local_policy_checkpoint is None
            else int(learned_local_policy_checkpoint["input_dim"])
        ),
        "learned_local_post_claim_policy_feature_dim": (
            None
            if learned_local_post_claim_policy_checkpoint is None
            else int(learned_local_post_claim_policy_checkpoint["input_dim"])
        ),
        "learned_local_policy_model_type": (
            None
            if learned_local_policy_checkpoint is None
            else str(learned_local_policy_checkpoint.get("model_type", "mlp"))
        ),
        "learned_local_post_claim_policy_model_type": (
            None
            if learned_local_post_claim_policy_checkpoint is None
            else str(learned_local_post_claim_policy_checkpoint.get("model_type", "mlp"))
        ),
        "learned_local_policy_feature_variant": (
            str(learned_local_policy_feature_variant)
        ),
        "learned_local_post_claim_policy_feature_variant": (
            None
            if learned_local_post_claim_policy_checkpoint is None
            else str(learned_local_post_claim_policy_checkpoint.get("feature_variant", "base"))
        ),
        "learned_local_target_policy_feature_variants": {
            color: str(checkpoint.get("feature_variant", "base"))
            for color, checkpoint in sorted(learned_local_target_policy_checkpoints.items())
        },
        "learned_local_online_map_features": bool(learned_local_policy_online_map_features),
        "learned_local_online_map_channel_count": int(learned_local_online_map_channel_count),
        "learned_local_online_map_size": int(args.learned_local_online_map_size),
        "learned_local_online_map_cell_m": float(args.learned_local_online_map_cell_m),
        "learned_local_online_map_visited_cells": 0,
        "learned_local_online_map_blocked_cells": 0,
        "learned_local_online_map_claimed_cells": 0,
        "learned_local_online_map_attempted_edges": 0,
        "learned_local_online_map_blocked_edges": 0,
        "learned_local_online_map_guard_blocked_cells": 0,
        "learned_local_online_map_guard_blocked_edges": 0,
        "learned_local_online_map_wall_guard_blocked_edges": 0,
        "post_claim_explore_primitives": list(post_claim_explore_primitives),
        "post_claim_explore_min_claimed_count": max(
            0,
            int(args.post_claim_explore_min_claimed_count),
        ),
        "post_claim_explore_plans": 0,
        "post_claim_explore_blocks_scheduled": 0,
        "post_claim_explore_blocks_executed": 0,
        "learned_topology_route_table": (
            None if args.learned_topology_route_table is None else str(args.learned_topology_route_table)
        ),
        "learned_topology_route_until_area_logit": (
            None
            if args.learned_topology_route_until_area_logit is None
            else float(args.learned_topology_route_until_area_logit)
        ),
        "learned_topology_route_release_on_seen_area_logit": (
            None
            if args.learned_topology_route_release_on_seen_area_logit is None
            else float(args.learned_topology_route_release_on_seen_area_logit)
        ),
        "learned_topology_route_advance_m": float(args.learned_topology_route_advance_m),
        "learned_topology_route_lookahead_m": float(args.learned_topology_route_lookahead_m),
        "learned_topology_route_reproject_window": int(
            args.learned_topology_route_reproject_window
        ),
        "learned_topology_route_reproject_trigger_m": float(
            args.learned_topology_route_reproject_trigger_m
        ),
        "learned_topology_route_yaw_threshold_by_color": dict(
            sorted(learned_topology_route_yaw_threshold_by_color.items())
        ),
        "learned_topology_route_arc_max_bearing_by_color": dict(
            sorted(learned_topology_route_arc_max_bearing_by_color.items())
        ),
        "learned_topology_route_use_stored_primitives": bool(
            args.learned_topology_route_use_stored_primitives
        ),
        "learned_topology_route_geometry_veto_min_clearance_m": (
            None
            if args.learned_topology_route_geometry_veto_min_clearance_m is None
            else float(args.learned_topology_route_geometry_veto_min_clearance_m)
        ),
        "learned_topology_route_geometry_veto_feasible_threshold": float(
            args.learned_topology_route_geometry_veto_feasible_threshold
        ),
        "learned_topology_route_geometry_veto_selected_primitives": sorted(
            learned_topology_route_geometry_veto_selected_primitives
        ),
        "learned_topology_route_geometry_veto_replacements": sorted(
            learned_topology_route_geometry_veto_replacements
        ),
        "body_clearance_geometry_veto_min_clearance_m": (
            None
            if args.body_clearance_geometry_veto_min_clearance_m is None
            else float(args.body_clearance_geometry_veto_min_clearance_m)
        ),
        "body_clearance_geometry_veto_feasible_threshold": float(
            args.body_clearance_geometry_veto_feasible_threshold
        ),
        "body_clearance_geometry_veto_states": sorted(
            body_clearance_geometry_veto_states
        ),
        "body_clearance_geometry_veto_min_claimed_count": max(
            0,
            int(args.body_clearance_geometry_veto_min_claimed_count),
        ),
        "body_clearance_geometry_veto_target_colors": sorted(
            body_clearance_geometry_veto_target_colors
        ),
        "body_clearance_geometry_veto_allow_force_single_candidate": bool(
            args.body_clearance_geometry_veto_allow_force_single_candidate
        ),
        "body_clearance_geometry_veto_allow_guard_disabled": bool(
            args.body_clearance_geometry_veto_allow_guard_disabled
        ),
        "body_clearance_geometry_veto_selected_primitives": sorted(
            body_clearance_geometry_veto_selected_primitives
        ),
        "body_clearance_geometry_veto_replacements": sorted(
            body_clearance_geometry_veto_replacements
        ),
        "body_clearance_geometry_veto_blocked_fallback_primitives": list(
            body_clearance_geometry_veto_blocked_fallback_primitives
        ),
        "body_clearance_geometry_veto_override_replacements": sorted(
            body_clearance_geometry_veto_override_replacements
        ),
        "body_clearance_geometry_veto_override_min_claimed_count": max(
            0,
            int(args.body_clearance_geometry_veto_override_min_claimed_count),
        ),
        "body_clearance_geometry_veto_ticks": 0,
        "body_clearance_geometry_vetoes": 0,
        "body_clearance_geometry_veto_claimed_count_suppressed_ticks": 0,
        "body_clearance_geometry_veto_target_suppressed_ticks": 0,
        "body_clearance_geometry_veto_selected_min_clearance_m": None,
        "learned_topology_route_ticks": 0,
        "learned_topology_route_overrides": 0,
        "learned_topology_route_reprojects": 0,
        "learned_topology_route_release_on_seen_ticks": 0,
        "learned_topology_route_geometry_veto_ticks": 0,
        "learned_topology_route_geometry_vetoes": 0,
        "learned_topology_route_geometry_veto_selected_min_clearance_m": None,
        "learned_topology_route_seen_gate_ticks": 0,
        "learned_topology_route_privileged_explorer_skipped_ticks": 0,
        "learned_topology_route_final_idx": 0,
        "learned_topology_route_resets": 0,
        "stable_base_required": bool(args.mode == "physical" and not args.allow_unstable_base_success),
        "fall_z_threshold_m": float(args.fall_z_threshold_m),
        "tip_threshold_rad": float(args.tip_threshold_rad),
        "base_z_min_m": None,
        "max_abs_roll_pitch_rad": 0.0,
        "fall_events": 0,
        "tip_events": 0,
        "unstable_base_events": 0,
        "first_unstable_tick": None,
        "first_unstable_reason": None,
        "body_clearance_success_required": bool(not args.allow_body_clearance_violation_success),
        "success_min_body_clearance_m": float(args.success_min_body_clearance_m),
        "body_clearance_min_m": None,
        "body_clearance_contact_threshold_m": 1e-4,
        "body_clearance_contact_events": 0,
        "first_body_clearance_contact_tick": None,
        "body_clearance_violation_events": 0,
        "first_body_clearance_violation_tick": None,
        "explorer_replans": 0,
        "explorer_visited_cells": 0,
        "explorer_blocked_cells": 0,
        "requested_min_clearance_min_m": None,
        "selected_min_clearance_min_m": None,
        "forward_execution_displacement_sum_m": 0.0,
        "proprio_contact_detector_ticks": 0,
        "proprio_contact_prob_max": None,
        "proprio_contact_escapes": 0,
        "proprio_contact_map_blocked_edges": 0,
        "history_risk_vetoes": 0,
        "history_risk_wedge_escapes": 0,
        "history_risk_corridor_commits": 0,
        "seen_target_route_ticks": 0,
        "seen_target_route_overrides": 0,
        "broad_explorer_ticks": 0,
        "broad_explorer_overrides": 0,
        "novelty_route_goals": 0,
        "novelty_route_scans": 0,
        "visual_ray_cells_added": 0,
    }
    stuck_streak = 0
    turn_streak = 0
    escape_plan: list[str] = []
    proprio_contact_feature_buffer: list[np.ndarray] = []
    proprio_contact_prob_history: list[float] = []
    proprio_contact_streak = 0
    proprio_contact_cooldown = 0
    proprio_contact_prev_yaw: float | None = None
    proprio_contact_prev_z: float | None = None
    history_risk_rows: list[tuple[np.ndarray, np.ndarray]] = []
    history_risk_pending_proprio: np.ndarray | None = None
    history_risk_wedge_cooldown = 0
    sequence_rows_window = max(int(history_risk_window), int(broad_explorer_window), 1)
    history_risk_corridor_run = 0
    stalled_prev_tick = False
    seen_target_estimates: dict[str, tuple[float, float, int]] = {}
    novelty_route_goal: tuple[float, float] | None = None
    novelty_route_goal_expiry = 0
    novelty_route_direction = 0.0
    novelty_route_visited_goals: set[tuple[int, int]] = set()
    seen_target_dist_calib = tuple(
        float(v) for v in str(args.seen_target_route_dist_calib).split(",")
    )
    seen_target_route_states = {
        state.strip().upper()
        for state in str(args.seen_target_route_states).split(",")
        if state.strip()
    }
    post_claim_explore_plan: list[str] = []
    stall_penalties: dict[str, int] = {}
    predicted_blocked_streak = 0
    predicted_blocked_cell: tuple[int, int] | None = None
    body_clearance_latch = 0
    claim_stalled_visual_latch = 0
    body_clearance_risk_escape_cooldown = 0
    body_clearance_current_contact_escape_streak = 0
    body_clearance_current_contact_escape_last_tick: int | None = None
    body_clearance_hard_veto_hold_streak = 0
    current_body_risk_cooldown = 0
    primitive_aux_clearance_switch_latch = 0
    opportunistic_claim_visible_ticks = {str(color): 0 for color in target_sequence}
    learned_local_turn_balance = 0
    learned_local_turn_run = 0
    learned_wall_follow_side = str(args.learned_wall_follow_initial_side)
    learned_wall_follow_side_ticks = 0
    learned_wall_follow_turn_run = 0
    learned_local_primary_policy_recurrent_hidden: torch.Tensor | None = None
    learned_local_post_claim_policy_recurrent_hidden: torch.Tensor | None = None
    learned_local_policy_turn_run = 0
    learned_local_policy_nonprogress_run = 0
    learned_local_policy_frontier_noop_run = 0
    learned_local_rotation_stall_streak = 0
    learned_local_last_route_next: tuple[int, int] | None = None
    stability_hold_remaining = 0
    prev_post_roll = 0.0
    prev_post_pitch = 0.0
    learned_topology_route_state: dict[str, Any] = {}
    learned_local_online_map = (
        OnlineEgomotionMap(
            size=int(args.learned_local_online_map_size),
            cell_m=float(args.learned_local_online_map_cell_m),
            hard_guard_blocks=bool(args.learned_local_online_map_hard_guard_blocks),
        )
        if bool(learned_local_policy_online_map_features)
        else None
    )
    weak_memory_seek_stall_streak = 0
    weak_memory_seek_yaw_loop_streak = 0
    weak_memory_seek_explore_cooldown = 0
    target_pursuit_stale_streak = 0
    target_pursuit_escape_cooldown = 0
    target_pursuit_stale_last_color: str | None = None
    target_pursuit_stale_window_ticks_by_color: dict[str, list[int]] = {}
    target_pursuit_suppressed_until: dict[str, int] = {}
    weak_memory_seek_colors = {
        item.strip().lower()
        for item in str(args.weak_memory_seek_colors).split(",")
        if item.strip()
    }
    if (
        slice_start is not None
        and bool(args.slice_preload_online_map)
        and learned_local_online_map is not None
    ):
        preload_rows = 0
        for row in slice_start.get("preload_log", []):
            if not isinstance(row, dict):
                continue
            row_tick = row.get("tick")
            if row_tick is None or int(row_tick) >= int(slice_start["start_tick"]):
                continue
            post_xy = row.get("post_xy")
            if isinstance(post_xy, (list, tuple)) and len(post_xy) >= 2:
                learned_local_online_map.observe_pose(post_xy[:2], tick=int(row_tick))
                preload_rows += 1
        for claim in beacon_claims:
            color = str(claim.get("target_color", "")).lower()
            if color not in slice_preclaimed_colors:
                continue
            xy = None
            claim_tick = int(claim.get("tick", 0))
            for row in slice_start.get("preload_log", []):
                if not isinstance(row, dict):
                    continue
                if row.get("tick") is None or int(row.get("tick")) < claim_tick:
                    continue
                post_xy = row.get("post_xy")
                if isinstance(post_xy, (list, tuple)) and len(post_xy) >= 2:
                    xy = post_xy[:2]
                    break
            if xy is None:
                continue
            learned_local_online_map.mark_claim(xy)
        wall_metrics["slice_online_map_preloaded"] = True
        wall_metrics["slice_online_map_preload_rows"] = int(preload_rows)
    else:
        wall_metrics["slice_online_map_preloaded"] = False
        wall_metrics["slice_online_map_preload_rows"] = 0
    if (
        slice_snapshot is not None
        and learned_local_online_map is not None
        and slice_snapshot.get("online_map") is not None
    ):
        learned_local_online_map.load_state_dict(slice_snapshot["online_map"])
        wall_metrics["slice_online_map_preloaded"] = True
        wall_metrics["slice_online_map_preload_rows"] = int(
            len(slice_snapshot["online_map"].get("visited", []))
        )
    learned_local_dataset_features: list[np.ndarray] = []
    learned_local_dataset_labels: list[int] = []
    learned_local_dataset_meta: list[dict[str, Any]] = []
    learned_local_dataset_feature_dim: int | None = None
    slice_snapshot_saved = False

    for tick in range(int(slice_tick_offset), int(slice_tick_offset) + int(args.max_ticks)):
        pos, quat = _current_pose(build)
        yaw = _yaw_from_quat_wxyz(quat)
        cur_pose = (float(pos[0]), float(pos[1]), float(yaw))
        if learned_local_online_map is not None:
            learned_local_online_map.observe_pose(pos[:2], tick=int(tick))
        log_entry: dict[str, Any] | None = None
        bearing_for_guard: float | None = None
        front_blocked_prob: float | None = None
        current_body_risk_prob: float | None = None
        history_risk_probs: dict[str, float] | None = None
        broad_explorer_primitive: str | None = None
        primitive_outcomes: dict[str, dict[str, float]] | None = None
        primitive_guard_outcomes: dict[str, dict[str, float]] | None = None
        primitive_clearance_outcomes: dict[str, dict[str, float]] | None = None
        primitive_aux_clearance_outcomes: dict[str, dict[str, float]] | None = None
        ego64: torch.Tensor | None = None
        primitive_outcome_slot = "primary"
        primitive_clearance_slot = "primary"
        primitive_policy_clearance_slot = "primary"
        primitive_aux_clearance_switch_active = False
        primitive_aux_clearance_switch_prob: float | None = None
        primitive_aux_clearance_switch_requested = False
        primitive_aux_clearance_switch_threshold: float | None = None
        current_body_risk_area_active_for_switch = True
        body_clearance_request = False
        frontier_pressure_committed = False
        weak_memory_recovery_active = False
        weak_memory_force_explore = False
        standoff_route_gate = False
        standoff_route_released_on_seen = False
        standoff_route_release_dist_m: float | None = None

        if args.policy == "wander":
            primitive = explorer.primitive(pos, yaw)
            log_entry = {"tick": tick, "state": "WANDER", "primitive": primitive, "explorer": explorer.trace()}
            bearing_for_guard = explorer.last_bearing
        else:
            ego = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
            ego64 = F.interpolate(ego.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0]
            if front_encoder is not None and front_head is not None:
                front_input = ego64
                if int(front_image_size) != 64:
                    front_input = F.interpolate(
                        ego64.unsqueeze(0),
                        size=(int(front_image_size), int(front_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    latent = front_encoder(front_input.unsqueeze(0).to(device))
                    front_blocked_prob = float(torch.sigmoid(front_head(latent))[0].cpu().item())
            if outcome_encoder is not None and outcome_head is not None:
                outcome_input = ego64
                if int(outcome_image_size) != 64:
                    outcome_input = F.interpolate(
                        ego.unsqueeze(0),
                        size=(int(outcome_image_size), int(outcome_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    outcome_latent = outcome_encoder(outcome_input.unsqueeze(0).to(device))
                primitive_outcomes = _predict_primitive_outcomes(
                    outcome_head,
                    outcome_latent,
                    primitive_vocab=outcome_primitive_vocab,
                    device=device,
                )
            if clearance_encoder is not None and clearance_head is not None:
                clearance_input = ego64
                if int(clearance_image_size) != 64:
                    clearance_input = F.interpolate(
                        ego.unsqueeze(0),
                        size=(int(clearance_image_size), int(clearance_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    clearance_latent = clearance_encoder(clearance_input.unsqueeze(0).to(device))
                primitive_clearance_outcomes = _predict_primitive_outcomes(
                    clearance_head,
                    clearance_latent,
                    primitive_vocab=clearance_primitive_vocab,
                    device=device,
                )
                primitive_outcomes = _fuse_clearance_outcomes(
                    primitive_outcomes,
                    primitive_clearance_outcomes,
                )
            if aux_clearance_encoder is not None and aux_clearance_head is not None:
                aux_clearance_input = ego64
                if int(aux_clearance_image_size) != 64:
                    aux_clearance_input = F.interpolate(
                        ego.unsqueeze(0),
                        size=(int(aux_clearance_image_size), int(aux_clearance_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    aux_clearance_latent = aux_clearance_encoder(
                        aux_clearance_input.unsqueeze(0).to(device)
                    )
                primitive_aux_clearance_outcomes = _predict_primitive_outcomes(
                    aux_clearance_head,
                    aux_clearance_latent,
                    primitive_vocab=aux_clearance_primitive_vocab,
                    device=device,
                )
            if (
                history_risk_model is not None
                or broad_explorer_model is not None
                or visual_ray_model is not None
            ):
                sequence_encoder = (
                    history_risk_encoder
                    if history_risk_encoder is not None
                    else (
                        broad_explorer_encoder
                        if broad_explorer_encoder is not None
                        else visual_ray_encoder
                    )
                )
                sequence_image_size = (
                    int(history_risk_image_size)
                    if history_risk_model is not None
                    else (
                        int(broad_explorer_image_size)
                        if broad_explorer_model is not None
                        else int(visual_ray_image_size)
                    )
                )
                history_input = ego64
                if int(sequence_image_size) != 64:
                    history_input = F.interpolate(
                        ego.unsqueeze(0),
                        size=(int(sequence_image_size), int(sequence_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    history_latent_now = (
                        sequence_encoder(history_input.unsqueeze(0).to(device))[0]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                if (
                    visual_ray_model is not None
                    and learned_local_online_map is not None
                ):
                    ray_lat = torch.from_numpy(
                        (history_latent_now - visual_ray_latent_mean)
                        / visual_ray_latent_std
                    ).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        ray_depths = (
                            visual_ray_model(ray_lat)[0].cpu().numpy().astype(np.float64)
                        )
                    np.clip(ray_depths, 0.0, float(visual_ray_depth_cap), out=ray_depths)
                    wall_metrics["visual_ray_cells_added"] += int(
                        learned_local_online_map.integrate_rays(
                            pos[:2],
                            float(yaw),
                            visual_ray_angles,
                            ray_depths,
                            depth_cap_m=float(visual_ray_depth_cap),
                        )
                    )
                if history_risk_pending_proprio is not None:
                    history_risk_rows.append(
                        (history_latent_now, history_risk_pending_proprio)
                    )
                    history_risk_pending_proprio = None
                    if len(history_risk_rows) > int(sequence_rows_window):
                        history_risk_rows.pop(0)
                if (
                    history_risk_model is not None
                    and len(history_risk_rows) >= int(history_risk_window)
                ):
                    history_rows = history_risk_rows[-int(history_risk_window):]
                    history_lat_w = torch.from_numpy(
                        (
                            np.stack([row[0] for row in history_rows])
                            - history_risk_latent_mean
                        )
                        / history_risk_latent_std
                    ).float().unsqueeze(0).to(device)
                    history_pro_w = torch.from_numpy(
                        (
                            np.stack([row[1] for row in history_rows])
                            - history_risk_proprio_mean
                        )
                        / history_risk_proprio_std
                    ).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        history_logits = history_risk_model(history_lat_w, history_pro_w)[0]
                    history_risk_probs = {
                        name: float(torch.sigmoid(history_logits[idx]).item())
                        for idx, name in enumerate(history_risk_vocab)
                    }
                if (
                    broad_explorer_model is not None
                    and len(history_risk_rows) >= int(broad_explorer_window)
                ):
                    explorer_rows = history_risk_rows[-int(broad_explorer_window):]
                    explorer_lat_w = torch.from_numpy(
                        (
                            np.stack([row[0] for row in explorer_rows])
                            - broad_explorer_latent_mean
                        )
                        / broad_explorer_latent_std
                    ).float().unsqueeze(0).to(device)
                    explorer_pro_w = torch.from_numpy(
                        (
                            np.stack([row[1] for row in explorer_rows])
                            - broad_explorer_proprio_mean
                        )
                        / broad_explorer_proprio_std
                    ).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        explorer_logits = broad_explorer_model(
                            explorer_lat_w, explorer_pro_w
                        )[0]
                    broad_explorer_primitive = str(
                        broad_explorer_vocab[int(explorer_logits.argmax().item())]
                    )
                history_risk_relaxed = bool(
                    int(args.history_risk_relax_min_claims) >= 0
                    and len(beacon_claims) >= int(args.history_risk_relax_min_claims)
                )
                if (
                    history_risk_probs is not None
                    and bool(args.history_risk_fuse_outcomes)
                    and primitive_outcomes is not None
                ):
                    active_fuse_weight = (
                        float(args.history_risk_relaxed_fuse_weight)
                        if history_risk_relaxed
                        else float(args.history_risk_fuse_weight)
                    )
                    for fuse_name, fuse_prob in history_risk_probs.items():
                        fuse_prediction = primitive_outcomes.get(fuse_name)
                        if fuse_prediction is None:
                            continue
                        fused_prob = max(
                            float(fuse_prediction.get("blocked_prob", 0.0)),
                            float(fuse_prob) * active_fuse_weight,
                        )
                        fuse_prediction["history_risk_prob"] = float(fuse_prob)
                        fuse_prediction["blocked_prob"] = float(min(1.0, fused_prob))
            if current_body_encoder is not None and current_body_head is not None:
                current_body_input = ego64
                if int(current_body_image_size) != 64:
                    current_body_input = F.interpolate(
                        ego64.unsqueeze(0),
                        size=(int(current_body_image_size), int(current_body_image_size)),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                with torch.no_grad():
                    current_body_latent = current_body_encoder(current_body_input.unsqueeze(0).to(device))
                    current_body_risk_prob = float(
                        torch.sigmoid(current_body_head(current_body_latent))[0].cpu().item()
                    )
            primitive_aux_clearance_switch_threshold = (
                current_body_threshold
                if args.primitive_aux_clearance_switch_threshold is None
                else float(args.primitive_aux_clearance_switch_threshold)
            )
            primitive_aux_clearance_switch_prob = current_body_risk_prob
            primitive_aux_clearance_switch_requested = bool(
                args.primitive_aux_clearance_switch_current_body_risk
                and primitive_outcomes is not None
                and primitive_aux_clearance_outcomes is not None
                and current_body_risk_prob is not None
                and primitive_aux_clearance_switch_threshold is not None
            )
            if (
                primitive_aux_clearance_switch_requested
                and args.current_body_risk_min_area_logit is None
            ):
                switch_claim_gate_pass = bool(
                    len(beacon_claims)
                    >= int(effective_aux_clearance_switch_min_claimed_count)
                )
                switch_triggered = bool(
                    switch_claim_gate_pass
                    and float(current_body_risk_prob)
                    >= float(primitive_aux_clearance_switch_threshold)
                )
                if not switch_claim_gate_pass:
                    wall_metrics[
                        "primitive_aux_clearance_switch_claim_gate_suppressed_ticks"
                    ] += 1
                    if float(current_body_risk_prob) >= float(
                        primitive_aux_clearance_switch_threshold
                    ):
                        wall_metrics[
                            "primitive_aux_clearance_switch_claim_gate_suppressed_high_risk_ticks"
                        ] += 1
                if switch_triggered:
                    primitive_aux_clearance_switch_latch = max(
                        int(primitive_aux_clearance_switch_latch),
                        int(args.primitive_aux_clearance_switch_latch_ticks),
                    )
                switch_latched = int(primitive_aux_clearance_switch_latch) > 0
                if switch_triggered or switch_latched:
                    primitive_aux_clearance_switch_active = True
                    wall_metrics["primitive_aux_clearance_switch_ticks"] += 1
                    if switch_latched:
                        primitive_aux_clearance_switch_latch = max(
                            0,
                            int(primitive_aux_clearance_switch_latch) - 1,
                        )
                else:
                    wall_metrics["primitive_aux_clearance_switch_suppressed_ticks"] += 1
            dx, dy, dyaw = _body_delta(prev_pose, cur_pose)
            aux = _build_aux((dx, dy, dyaw), last_cmd, last_primitive)
            aux_t = (torch.from_numpy(aux).to(device) - aux_mean) / aux_std
            motion_delta = torch.tensor([dx / range_scale, dy / range_scale, dyaw], dtype=torch.float32)
            outputs, ctrl_state = model.step_online(ego64, aux_t, motion_delta, ctrl_state)
            target_switch_info: dict[str, Any] | None = None
            target_switch_candidates: list[dict[str, Any]] = []
            seen_read_threshold = args.seen_read_threshold
            target_color_readouts: dict[str, dict[str, Any]] = {}
            for color in target_sequence:
                color_idx = color_vocab.index(color)
                color_conf = float(outputs["memory_conf"][color_idx])
                color_read_score = _controller_read_score(outputs, color_idx)
                color_read_gate_pass = bool(
                    seen_read_threshold is None
                    or color_read_score is None
                    or color_read_score >= float(seen_read_threshold)
                )
                color_area_logit = (
                    float(outputs["rgb_area_logits"][color_idx])
                    if "rgb_area_logits" in outputs
                    else -99.0
                )
                target_color_readouts[color] = {
                    "mem_conf": _round_float(color_conf, 4),
                    "area": _round_float(color_area_logit, 4),
                    "read_score": (
                        None
                        if color_read_score is None
                        else _round_float(color_read_score, 4)
                    ),
                    "read_gate_pass": bool(color_read_gate_pass),
                    "claimed": bool(
                        any(str(item.get("target_color")) == color for item in beacon_claims)
                    ),
                }
                if (
                    color not in first_seen_ticks
                    and color_conf > float(args.seen_conf)
                    and color_read_gate_pass
                ):
                    first_seen_ticks[color] = int(tick)
                if color_conf > float(args.seen_conf) and color_read_gate_pass:
                    last_seen_ticks[color] = int(tick)
                if color in first_seen_ticks:
                    target_color_readouts[color]["first_seen_tick"] = int(first_seen_ticks[color])
                if color in last_seen_ticks:
                    target_color_readouts[color]["last_seen_tick"] = int(last_seen_ticks[color])
            if learned_target_scheduler_model is not None and len(target_sequence) > 1:
                claimed_color_set = {str(item.get("target_color")) for item in beacon_claims}
                current_color = target_sequence[target_index]
                scheduler_feature = _target_scheduler_feature(
                    colors=learned_target_scheduler_color_vocab,
                    current_color=current_color,
                    current_target_age_ticks=max(
                        0,
                        int(tick) - int(target_active_since_tick),
                    ),
                    claimed_colors=claimed_color_set,
                    color_readouts=target_color_readouts,
                    tick=int(tick),
                    max_ticks=int(feature_max_ticks),
                    device=device,
                )
                with torch.no_grad():
                    scheduler_logits = learned_target_scheduler_model(scheduler_feature)[0]
                    scheduler_probs = torch.softmax(scheduler_logits, dim=0)
                masked_logits = scheduler_logits.detach().clone()
                for idx, color in enumerate(learned_target_scheduler_color_vocab):
                    if color in claimed_color_set or color not in target_sequence:
                        masked_logits[idx] = -1.0e9
                if torch.isfinite(masked_logits).any() and float(masked_logits.max().detach().cpu()) > -1.0e8:
                    selected_scheduler_idx = int(torch.argmax(masked_logits).detach().cpu())
                    selected_scheduler_color = learned_target_scheduler_color_vocab[
                        selected_scheduler_idx
                    ]
                    selected_scheduler_prob = float(
                        scheduler_probs[selected_scheduler_idx].detach().cpu()
                    )
                    sorted_logits = torch.sort(masked_logits, descending=True).values
                    scheduler_margin = (
                        float((sorted_logits[0] - sorted_logits[1]).detach().cpu())
                        if int(sorted_logits.numel()) > 1 and float(sorted_logits[1].detach().cpu()) > -1.0e8
                        else None
                    )
                    wall_metrics["learned_target_scheduler_ticks"] += 1
                    if bool(args.learned_target_scheduler_log_scores):
                        for idx, color in enumerate(learned_target_scheduler_color_vocab):
                            target_switch_candidates.append(
                                {
                                    "color": color,
                                    "target_index": (
                                        target_sequence.index(color)
                                        if color in target_sequence
                                        else None
                                    ),
                                    "claimed": bool(color in claimed_color_set),
                                    "policy": "learned_target_scheduler",
                                    "current_target_age_ticks": max(
                                        0,
                                        int(tick) - int(target_active_since_tick),
                                    ),
                                    "logit": _round_float(
                                        float(scheduler_logits[idx].detach().cpu()),
                                        4,
                                    ),
                                    "prob": _round_float(
                                        float(scheduler_probs[idx].detach().cpu()),
                                        4,
                                    ),
                                    "accepted": bool(color == selected_scheduler_color),
                                }
                            )
                    if selected_scheduler_color in target_sequence:
                        new_index = target_sequence.index(selected_scheduler_color)
                        if int(new_index) != int(target_index):
                            target_switch_info = {
                                "from": current_color,
                                "to": selected_scheduler_color,
                                "policy": "learned_target_scheduler",
                                "prob": _round_float(selected_scheduler_prob, 4),
                                "margin": _round_float(scheduler_margin, 4),
                            }
                            wall_metrics["learned_target_scheduler_switches"] += 1
                            wall_metrics["target_switches"] += 1
                            target_index = int(new_index)
                            target_active_since_tick = int(tick)
            if (
                len(target_sequence) > 1
                and str(args.multi_target_switch_policy) != "fixed"
                and learned_target_scheduler_model is None
            ):
                claimed_color_set = {str(item.get("target_color")) for item in beacon_claims}
                current_color = target_sequence[target_index]
                current_idx = color_vocab.index(current_color)
                current_conf = float(outputs["memory_conf"][current_idx])
                current_read_score = _controller_read_score(outputs, current_idx)
                current_read_gate_pass = bool(
                    seen_read_threshold is None
                    or current_read_score is None
                    or current_read_score >= float(seen_read_threshold)
                )
                current_area = (
                    float(outputs["rgb_area_logits"][current_idx])
                    if "rgb_area_logits" in outputs
                    else -99.0
                )
                current_suppressed_until = int(
                    target_pursuit_suppressed_until.get(str(current_color), 0)
                )
                current_stale_suppressed = bool(current_suppressed_until > int(tick))
                switch_conf = (
                    float(args.seen_conf)
                    if args.multi_target_switch_conf is None
                    else float(args.multi_target_switch_conf)
                )
                switch_area = float(args.multi_target_switch_area_logit)
                seen_switch_min_area = args.multi_target_seen_switch_min_area_logit
                policy_name = str(args.multi_target_switch_policy)
                should_consider_switch = False
                if policy_name == "seen_when_active_unseen":
                    should_consider_switch = current_conf < switch_conf or not current_read_gate_pass
                elif policy_name == "visible_priority":
                    should_consider_switch = current_area < switch_area
                elif policy_name == "memory_priority":
                    should_consider_switch = True
                if current_stale_suppressed:
                    should_consider_switch = True
                if should_consider_switch:
                    candidates: list[tuple[float, int, str, float, float]] = []
                    for cand_index, color in enumerate(target_sequence):
                        color_suppressed_until = int(
                            target_pursuit_suppressed_until.get(str(color), 0)
                        )
                        color_stale_suppressed = bool(
                            color_suppressed_until > int(tick)
                        )
                        candidate_log: dict[str, Any] | None = (
                            {
                                "color": color,
                                "target_index": int(cand_index),
                                "claimed": bool(color in claimed_color_set),
                                "stale_suppressed": bool(color_stale_suppressed),
                                "stale_suppressed_until": (
                                    int(color_suppressed_until)
                                    if color_stale_suppressed
                                    else None
                                ),
                            }
                            if bool(args.log_color_readouts)
                            else None
                        )
                        if color in claimed_color_set:
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "claimed"
                                target_switch_candidates.append(candidate_log)
                            continue
                        if color_stale_suppressed:
                            wall_metrics[
                                "target_pursuit_stale_switch_rejections_suppressed"
                            ] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "stale_suppressed"
                                target_switch_candidates.append(candidate_log)
                            continue
                        color_idx = color_vocab.index(color)
                        conf = float(outputs["memory_conf"][color_idx])
                        read_score = _controller_read_score(outputs, color_idx)
                        read_gate_pass = bool(
                            seen_read_threshold is None
                            or read_score is None
                            or read_score >= float(seen_read_threshold)
                        )
                        area_logit = (
                            float(outputs["rgb_area_logits"][color_idx])
                            if "rgb_area_logits" in outputs
                            else -99.0
                        )
                        if candidate_log is not None:
                            candidate_log.update({
                                "mem_conf": _round_float(conf, 4),
                                "read_score": (
                                    None
                                    if read_score is None
                                    else _round_float(read_score, 4)
                                ),
                                "read_gate_pass": bool(read_gate_pass),
                                "area": _round_float(area_logit, 4),
                            })
                        relaxed_for_suppressed_current = bool(current_stale_suppressed)
                        if conf < switch_conf and not relaxed_for_suppressed_current:
                            wall_metrics["target_switch_rejections_conf"] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "conf"
                                target_switch_candidates.append(candidate_log)
                            continue
                        if not read_gate_pass and not relaxed_for_suppressed_current:
                            wall_metrics["target_switch_rejections_read"] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "read"
                                target_switch_candidates.append(candidate_log)
                            continue
                        if (
                            policy_name == "visible_priority"
                            and area_logit < switch_area
                            and not relaxed_for_suppressed_current
                        ):
                            wall_metrics["target_switch_rejections_area"] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "area"
                                target_switch_candidates.append(candidate_log)
                            continue
                        if (
                            policy_name == "seen_when_active_unseen"
                            and seen_switch_min_area is not None
                            and area_logit < float(seen_switch_min_area)
                            and not relaxed_for_suppressed_current
                        ):
                            wall_metrics["target_switch_rejections_area"] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "seen_area"
                                target_switch_candidates.append(candidate_log)
                            continue
                        visible_bonus = 10.0 if area_logit >= switch_area else 0.0
                        if relaxed_for_suppressed_current:
                            read_bonus = 0.0 if read_score is None else 0.05 * float(read_score)
                            score = (
                                visible_bonus
                                + area_logit
                                + 0.1 * conf
                                + read_bonus
                                - 1e-3 * cand_index
                            )
                        else:
                            score = visible_bonus + area_logit + 0.1 * conf - 1e-3 * cand_index
                        if candidate_log is not None:
                            candidate_log["accepted"] = True
                            candidate_log["score"] = _round_float(score, 4)
                            if relaxed_for_suppressed_current:
                                candidate_log["relaxed_for_stale_suppressed_current"] = True
                            target_switch_candidates.append(candidate_log)
                        candidates.append((score, cand_index, color, conf, area_logit))
                    if candidates:
                        wall_metrics["target_switch_candidate_ticks"] += 1
                        candidates.sort(reverse=True)
                        _, new_index, new_color, new_conf, new_area = candidates[0]
                        if int(new_index) != int(target_index):
                            target_switch_info = {
                                "from": current_color,
                                "to": new_color,
                                "policy": policy_name,
                                "conf": _round_float(new_conf, 4),
                                "area": _round_float(new_area, 4),
                            }
                            wall_metrics["target_switches"] += 1
                            if current_stale_suppressed:
                                wall_metrics[
                                    "target_pursuit_stale_switches_from_suppressed"
                                ] += 1
                                target_switch_info["from_stale_suppressed"] = True
                                target_switch_info["suppressed_until"] = int(
                                    current_suppressed_until
                                )
                            target_index = int(new_index)
                            target_active_since_tick = int(tick)
            stale_switch_after_noops = int(
                args.multi_target_stale_seen_switch_after_frontier_noops
            )
            if (
                stale_switch_after_noops > 0
                and len(target_sequence) > 1
                and target_switch_info is None
                and learned_target_scheduler_model is None
                and int(learned_local_policy_frontier_noop_run) >= stale_switch_after_noops
            ):
                claimed_color_set = {str(item.get("target_color")) for item in beacon_claims}
                current_color = target_sequence[target_index]
                current_idx = color_vocab.index(current_color)
                current_conf = float(outputs["memory_conf"][current_idx])
                current_read_score = _controller_read_score(outputs, current_idx)
                current_read_gate_pass = bool(
                    seen_read_threshold is None
                    or current_read_score is None
                    or current_read_score >= float(seen_read_threshold)
                )
                current_area = (
                    float(outputs["rgb_area_logits"][current_idx])
                    if "rgb_area_logits" in outputs
                    else -99.0
                )
                switch_conf = (
                    float(args.seen_conf)
                    if args.multi_target_switch_conf is None
                    else float(args.multi_target_switch_conf)
                )
                switch_area = float(args.multi_target_switch_area_logit)
                active_unreliable = (
                    current_conf < switch_conf
                    or not current_read_gate_pass
                    or current_area < switch_area
                )
                if active_unreliable:
                    max_age = int(args.multi_target_stale_seen_switch_max_age_ticks)
                    stale_candidates: list[tuple[float, int, str, int]] = []
                    for cand_index, color in enumerate(target_sequence):
                        candidate_log: dict[str, Any] | None = (
                            {
                                "color": color,
                                "target_index": int(cand_index),
                                "claimed": bool(color in claimed_color_set),
                                "policy": "stale_seen_after_frontier_noops",
                            }
                            if bool(args.log_color_readouts)
                            else None
                        )
                        if color in claimed_color_set or int(cand_index) == int(target_index):
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = (
                                    "claimed" if color in claimed_color_set else "current"
                                )
                                target_switch_candidates.append(candidate_log)
                            continue
                        last_seen = last_seen_ticks.get(color)
                        if last_seen is None:
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "never_seen"
                                target_switch_candidates.append(candidate_log)
                            continue
                        age = int(tick) - int(last_seen)
                        if max_age > 0 and age > max_age:
                            wall_metrics["target_stale_seen_switch_rejections_age"] += 1
                            if candidate_log is not None:
                                candidate_log["accepted"] = False
                                candidate_log["reject_reason"] = "age"
                                candidate_log["age_ticks"] = int(age)
                                target_switch_candidates.append(candidate_log)
                            continue
                        score = -float(age) - 1e-3 * float(cand_index)
                        if candidate_log is not None:
                            candidate_log["accepted"] = True
                            candidate_log["last_seen_tick"] = int(last_seen)
                            candidate_log["age_ticks"] = int(age)
                            candidate_log["score"] = _round_float(score, 4)
                            target_switch_candidates.append(candidate_log)
                        stale_candidates.append((score, int(cand_index), color, int(last_seen)))
                    if stale_candidates:
                        wall_metrics["target_stale_seen_switch_candidate_ticks"] += 1
                        stale_candidates.sort(reverse=True)
                        _, new_index, new_color, last_seen = stale_candidates[0]
                        if int(new_index) != int(target_index):
                            target_switch_info = {
                                "from": current_color,
                                "to": new_color,
                                "policy": "stale_seen_after_frontier_noops",
                                "last_seen_tick": int(last_seen),
                                "age_ticks": int(tick) - int(last_seen),
                                "frontier_noop_run": int(learned_local_policy_frontier_noop_run),
                            }
                            wall_metrics["target_stale_seen_switches"] += 1
                            wall_metrics["target_switches"] += 1
                            target_index = int(new_index)
                            target_active_since_tick = int(tick)
            active_target_color = target_sequence[target_index]
            active_target_tc = color_vocab.index(active_target_color)
            mem_vec = outputs["memory_vec"][active_target_tc]
            mem_conf = float(outputs["memory_conf"][active_target_tc])
            read_score = _controller_read_score(outputs, active_target_tc)
            read_gate_pass = bool(
                seen_read_threshold is None
                or read_score is None
                or read_score >= float(seen_read_threshold)
            )
            area = float(outputs["rgb_area_logits"][active_target_tc]) if "rgb_area_logits" in outputs else -9.0
            evid = outputs["evidence_vec"][active_target_tc]
            in_cone = area > 0.0
            seen = mem_conf > float(args.seen_conf) and read_gate_pass
            if (
                seen_read_threshold is not None
                and mem_conf > float(args.seen_conf)
                and not read_gate_pass
            ):
                wall_metrics["seen_read_gate_low_ticks"] += 1
            if seen and active_target_color not in first_seen_ticks:
                first_seen_ticks[active_target_color] = int(tick)
            first_seen_tick = first_seen_ticks.get(active_target_color)
            if in_cone:
                bearing = math.atan2(float(evid[1]), float(evid[0]))
            else:
                bearing = math.atan2(float(mem_vec[1]), float(mem_vec[0]))
            bearing_for_guard = bearing
            weak_memory_recovery_active = False
            weak_memory_force_explore = False
            target_pursuit_force_explore = False
            active_target_stale_suppressed_until = int(
                target_pursuit_suppressed_until.get(str(active_target_color), 0)
            )
            active_target_stale_suppressed = bool(
                active_target_stale_suppressed_until > int(tick)
            )
            claim_gate_log: dict[str, Any] = {}

            # Recall preamble (scaffold): OBSERVE to bind, then HIDE (turn away).
            recall_preamble = False
            if args.demo_mode == "recall":
                if tick < int(args.observe_ticks):
                    primitive, st, recall_preamble = "hold", "OBSERVE", True
                    if tick == 0:
                        import imageio
                        op = REPO_ROOT / ".generated/go2_memory_closed_loop" / f"observe_{args.target_color}.png"
                        imageio.imwrite(str(op), ego64.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
                        print(f"OBSERVE frame: area={area:.3f} fires={in_cone} saved {op.name}", flush=True)
                elif tick < int(args.observe_ticks) + int(args.hide_ticks):
                    primitive, st, recall_preamble = "yaw_right", "HIDE", True

            if not recall_preamble:
                # CLAIM: target centered, in cone, close (large blob).
                seen_age_ticks = (tick - int(first_seen_tick) + 1) if first_seen_tick is not None else 0
                weak_memory_conf_recovery_active = bool(
                    len(target_sequence) > 1
                    and (
                        not weak_memory_seek_colors
                        or str(active_target_color).lower() in weak_memory_seek_colors
                    )
                    and float(args.weak_memory_seek_conf) > 0.0
                    and seen
                    and not in_cone
                    and mem_conf < float(args.weak_memory_seek_conf)
                )
                weak_memory_visual_recovery_active = bool(
                    len(target_sequence) > 1
                    and (
                        not weak_memory_seek_colors
                        or str(active_target_color).lower() in weak_memory_seek_colors
                    )
                    and args.weak_memory_seek_area_logit is not None
                    and seen
                    and area < float(args.weak_memory_seek_area_logit)
                )
                weak_memory_recovery_active = bool(
                    weak_memory_conf_recovery_active or weak_memory_visual_recovery_active
                )
                weak_memory_force_explore = bool(
                    weak_memory_recovery_active
                    and (
                        int(weak_memory_seek_explore_cooldown) > 0
                        or bool(args.weak_memory_seek_force_explore_on_recovery)
                    )
                )
                target_pursuit_force_explore = bool(
                    int(target_pursuit_escape_cooldown) > 0
                )
                state_seen = bool(
                    seen
                    and not weak_memory_force_explore
                    and not target_pursuit_force_explore
                    and not active_target_stale_suppressed
                )
                if weak_memory_force_explore:
                    wall_metrics["weak_memory_seek_forced_explore_ticks"] += 1
                    if bool(args.weak_memory_seek_force_explore_on_recovery):
                        wall_metrics["weak_memory_seek_immediate_explore_recoveries"] += 1
                if target_pursuit_force_explore:
                    wall_metrics["target_pursuit_stale_forced_explore_ticks"] += 1
                if active_target_stale_suppressed:
                    wall_metrics["target_pursuit_stale_suppressed_active_ticks"] += 1
                claim_mature = seen_age_ticks >= max(0, int(args.claim_min_seen_ticks))
                standard_claim = (
                    seen and in_cone
                    and claim_mature
                    and area > float(args.claim_area_logit)
                    and abs(bearing) < float(args.claim_bearing)
                )
                near_area_logit = claim_near_area_by_color.get(
                    active_target_color,
                    args.claim_near_area_logit,
                )
                near_bearing = claim_near_bearing_by_color.get(
                    active_target_color,
                    args.claim_near_bearing,
                )
                near_min_seen_ticks = claim_near_min_seen_by_color.get(
                    active_target_color,
                    int(args.claim_near_min_seen_ticks),
                )
                claim_success_proxy_area_logit = claim_success_proxy_area_by_color.get(
                    active_target_color,
                    args.claim_success_proxy_area_logit,
                )
                claim_success_proxy_bearing = claim_success_proxy_bearing_by_color.get(
                    active_target_color,
                    args.claim_success_proxy_bearing,
                )
                claim_success_model_score: float | None = None
                active_claim_success_model_threshold = (
                    claim_success_model_threshold_by_color.get(
                        str(active_target_color).lower(),
                        claim_success_model_threshold,
                    )
                )
                if claim_success_model is not None and claim_success_model_threshold is not None:
                    with torch.no_grad():
                        claim_success_model_score = float(
                            torch.sigmoid(
                                claim_success_model(
                                    _claim_success_model_feature(
                                        color=active_target_color,
                                        color_vocab=claim_success_model_color_vocab,
                                        area=area,
                                        bearing=bearing,
                                        mem_conf=mem_conf,
                                        read_score=read_score,
                                        seen_age_ticks=int(seen_age_ticks),
                                        seen=bool(seen),
                                        in_cone=bool(in_cone),
                                        claimed_count=int(len(beacon_claims)),
                                        tick=int(tick),
                                        max_ticks=int(feature_max_ticks),
                                        device=device,
                                    )
                                )
                            ).detach().cpu().item()
                        )
                    wall_metrics["claim_success_model_evaluations"] += 1
                claim_success_proxy_gate = _claim_success_proxy_gate_entry(
                    area=area,
                    area_threshold=claim_success_proxy_area_logit,
                    bearing=bearing,
                    bearing_threshold=claim_success_proxy_bearing,
                    model_score=claim_success_model_score,
                    model_threshold=active_claim_success_model_threshold,
                )
                claim_success_proxy_pass = bool(claim_success_proxy_gate["passed"])
                if not claim_success_proxy_pass:
                    wall_metrics["claim_success_proxy_rejections"] += 1
                    if "model_low" in claim_success_proxy_gate.get("reject_reasons", []):
                        wall_metrics["claim_success_model_rejections"] += 1
                claim_success_model_trigger_min_seen_ticks = (
                    int(near_min_seen_ticks)
                    if args.claim_success_model_trigger_min_seen_ticks is None
                    else int(args.claim_success_model_trigger_min_seen_ticks)
                )
                learned_model_claim = bool(
                    args.claim_success_model_positive_trigger
                    and claim_success_model is not None
                    and active_claim_success_model_threshold is not None
                    and seen
                    and in_cone
                    and read_gate_pass
                    and seen_age_ticks >= max(0, claim_success_model_trigger_min_seen_ticks)
                    and claim_success_proxy_pass
                )
                standard_claim = bool(standard_claim and claim_success_proxy_pass)
                near_claim = (
                    near_area_logit is not None
                    and seen and in_cone
                    and seen_age_ticks >= max(0, int(near_min_seen_ticks))
                    and area > float(near_area_logit)
                    and abs(bearing) < float(near_bearing)
                    and claim_success_proxy_pass
                )
                contact_claim = (
                    args.claim_contact_area_logit is not None
                    and seen and in_cone
                    and seen_age_ticks >= max(0, int(args.claim_contact_min_seen_ticks))
                    and area > float(args.claim_contact_area_logit)
                    and abs(bearing) < float(args.claim_contact_bearing)
                    and claim_success_proxy_pass
                )
                stalled_visual_claim = (
                    args.claim_stalled_area_logit is not None
                    and int(claim_stalled_visual_latch) > 0
                    and seen and in_cone
                    and seen_age_ticks >= max(0, int(args.claim_stalled_min_seen_ticks))
                    and area > float(args.claim_stalled_area_logit)
                    and abs(bearing) < float(args.claim_stalled_bearing)
                    and claim_success_proxy_pass
                )
                learned_model_gate_rejections: list[str] = []
                learned_model_gate_enabled = bool(
                    args.claim_success_model_positive_trigger
                    and claim_success_model is not None
                    and active_claim_success_model_threshold is not None
                )
                if not bool(args.claim_success_model_positive_trigger):
                    learned_model_gate_rejections.append("disabled")
                if claim_success_model is None or active_claim_success_model_threshold is None:
                    learned_model_gate_rejections.append("model_unavailable")
                if not bool(seen):
                    learned_model_gate_rejections.append("not_seen")
                if not bool(in_cone):
                    learned_model_gate_rejections.append("not_in_cone")
                if not bool(read_gate_pass):
                    learned_model_gate_rejections.append("read_gate_low")
                if seen_age_ticks < max(0, claim_success_model_trigger_min_seen_ticks):
                    learned_model_gate_rejections.append("seen_age_low")
                if not claim_success_proxy_pass:
                    learned_model_gate_rejections.extend(
                        f"success_proxy:{reason}"
                        for reason in claim_success_proxy_gate.get("reject_reasons", [])
                    )
                learned_model_gate = {
                    "enabled": bool(learned_model_gate_enabled),
                    "passed": bool(learned_model_claim),
                    "reject_reasons": [] if learned_model_claim else learned_model_gate_rejections,
                    "min_seen_ticks": int(claim_success_model_trigger_min_seen_ticks),
                    "model_score": (
                        None
                        if claim_success_model_score is None
                        else _round_float(claim_success_model_score, 4)
                    ),
                    "model_threshold": (
                        None
                        if active_claim_success_model_threshold is None
                        else float(active_claim_success_model_threshold)
                    ),
                }
                opportunistic_claim_info: dict[str, Any] | None = None
                if bool(args.multi_target_opportunistic_claims) and len(target_sequence) > 1:
                    claimed_color_set = {
                        str(item.get("target_color")) for item in beacon_claims
                    }
                    rgb_area_logits = outputs.get("rgb_area_logits")
                    evidence_vec = outputs.get("evidence_vec")
                    ready_candidates: list[dict[str, Any]] = []
                    visible_candidate_tick = False
                    if rgb_area_logits is not None and evidence_vec is not None:
                        for cand_index, color in enumerate(target_sequence):
                            color_name = str(color)
                            if color_name in claimed_color_set or color_name not in color_vocab:
                                opportunistic_claim_visible_ticks[color_name] = 0
                                continue
                            color_idx = color_vocab.index(color_name)
                            color_area = float(rgb_area_logits[color_idx])
                            color_evid = evidence_vec[color_idx]
                            color_bearing = math.atan2(
                                float(color_evid[1]),
                                float(color_evid[0]),
                            )
                            area_threshold = (
                                args.multi_target_opportunistic_claim_area_logit
                                if args.multi_target_opportunistic_claim_area_logit is not None
                                else claim_near_area_by_color.get(
                                    color_name,
                                    args.claim_near_area_logit,
                                )
                            )
                            bearing_threshold = (
                                args.multi_target_opportunistic_claim_bearing
                                if args.multi_target_opportunistic_claim_bearing is not None
                                else claim_near_bearing_by_color.get(
                                    color_name,
                                    args.claim_near_bearing,
                                )
                            )
                            if area_threshold is None or bearing_threshold is None:
                                opportunistic_claim_visible_ticks[color_name] = 0
                                continue
                            visual_ok = bool(
                                color_area > float(area_threshold)
                                and abs(color_bearing) < float(bearing_threshold)
                            )
                            candidate_success_proxy_area = (
                                claim_success_proxy_area_by_color.get(
                                    color_name,
                                    args.claim_success_proxy_area_logit,
                                )
                            )
                            candidate_success_proxy_bearing = (
                                claim_success_proxy_bearing_by_color.get(
                                    color_name,
                                    args.claim_success_proxy_bearing,
                                )
                            )
                            cand_read_score = _controller_read_score(outputs, color_idx)
                            candidate_success_model_score: float | None = None
                            candidate_claim_success_model_threshold = (
                                claim_success_model_threshold_by_color.get(
                                    str(color_name).lower(),
                                    claim_success_model_threshold,
                                )
                            )
                            if (
                                claim_success_model is not None
                                and claim_success_model_threshold is not None
                            ):
                                with torch.no_grad():
                                    candidate_success_model_score = float(
                                        torch.sigmoid(
                                            claim_success_model(
                                                _claim_success_model_feature(
                                                    color=color_name,
                                                    color_vocab=claim_success_model_color_vocab,
                                                    area=color_area,
                                                    bearing=color_bearing,
                                                    mem_conf=float(outputs["memory_conf"][color_idx]),
                                                    read_score=cand_read_score,
                                                    seen_age_ticks=int(
                                                        opportunistic_claim_visible_ticks.get(
                                                            color_name, 0
                                                        )
                                                    ),
                                                    seen=bool(color_area > float(area_threshold)),
                                                    in_cone=bool(
                                                        abs(color_bearing)
                                                        < float(bearing_threshold)
                                                    ),
                                                    claimed_count=int(len(beacon_claims)),
                                                    tick=int(tick),
                                                    max_ticks=int(feature_max_ticks),
                                                    device=device,
                                                )
                                            )
                                        ).detach().cpu().item()
                                    )
                                wall_metrics["claim_success_model_evaluations"] += 1
                            candidate_success_proxy_gate = _claim_success_proxy_gate_entry(
                                area=color_area,
                                area_threshold=candidate_success_proxy_area,
                                bearing=color_bearing,
                                bearing_threshold=candidate_success_proxy_bearing,
                                model_score=candidate_success_model_score,
                                model_threshold=candidate_claim_success_model_threshold,
                            )
                            if visual_ok and not bool(candidate_success_proxy_gate["passed"]):
                                wall_metrics[
                                    "claim_success_proxy_opportunistic_rejections"
                                ] += 1
                                if "model_low" in candidate_success_proxy_gate.get(
                                    "reject_reasons", []
                                ):
                                    wall_metrics["claim_success_model_rejections"] += 1
                            visual_ok = bool(
                                visual_ok and candidate_success_proxy_gate["passed"]
                            )
                            if visual_ok:
                                visible_candidate_tick = True
                                opportunistic_claim_visible_ticks[color_name] = min(
                                    1000000,
                                    int(opportunistic_claim_visible_ticks.get(color_name, 0)) + 1,
                                )
                            else:
                                opportunistic_claim_visible_ticks[color_name] = 0
                            visible_ticks = int(opportunistic_claim_visible_ticks[color_name])
                            if visible_ticks >= max(
                                1,
                                int(args.multi_target_opportunistic_claim_min_visible_ticks),
                            ):
                                ready_candidates.append({
                                    "target_color": color_name,
                                    "target_index": int(cand_index),
                                    "mem_conf": float(outputs["memory_conf"][color_idx]),
                                    "area": float(color_area),
                                    "bearing": float(color_bearing),
                                    "visible_ticks": int(visible_ticks),
                                    "read_score": cand_read_score,
                                    "area_threshold": float(area_threshold),
                                    "bearing_threshold": float(bearing_threshold),
                                    "success_proxy": dict(candidate_success_proxy_gate),
                                })
                    if visible_candidate_tick:
                        wall_metrics["target_opportunistic_claim_candidate_ticks"] += 1
                    if ready_candidates:
                        ready_candidates.sort(
                            key=lambda item: (
                                float(item["area"]),
                                -abs(float(item["bearing"])),
                            ),
                            reverse=True,
                        )
                        opportunistic_claim_info = ready_candidates[0]
                standard_gate = _claim_gate_entry(
                    enabled=True,
                    seen=seen,
                    in_cone=in_cone,
                    seen_age_ticks=seen_age_ticks,
                    min_seen_ticks=int(args.claim_min_seen_ticks),
                    area=area,
                    area_threshold=float(args.claim_area_logit),
                    bearing=bearing,
                    bearing_threshold=float(args.claim_bearing),
                )
                near_gate = _claim_gate_entry(
                    enabled=near_area_logit is not None,
                    seen=seen,
                    in_cone=in_cone,
                    seen_age_ticks=seen_age_ticks,
                    min_seen_ticks=int(near_min_seen_ticks),
                    area=area,
                    area_threshold=None if near_area_logit is None else float(near_area_logit),
                    bearing=bearing,
                    bearing_threshold=None if near_bearing is None else float(near_bearing),
                )
                contact_gate = _claim_gate_entry(
                    enabled=args.claim_contact_area_logit is not None,
                    seen=seen,
                    in_cone=in_cone,
                    seen_age_ticks=seen_age_ticks,
                    min_seen_ticks=int(args.claim_contact_min_seen_ticks),
                    area=area,
                    area_threshold=(
                        None
                        if args.claim_contact_area_logit is None
                        else float(args.claim_contact_area_logit)
                    ),
                    bearing=bearing,
                    bearing_threshold=float(args.claim_contact_bearing),
                )
                stalled_gate = _claim_gate_entry(
                    enabled=args.claim_stalled_area_logit is not None,
                    seen=seen,
                    in_cone=in_cone,
                    seen_age_ticks=seen_age_ticks,
                    min_seen_ticks=int(args.claim_stalled_min_seen_ticks),
                    area=area,
                    area_threshold=(
                        None
                        if args.claim_stalled_area_logit is None
                        else float(args.claim_stalled_area_logit)
                    ),
                    bearing=bearing,
                    bearing_threshold=float(args.claim_stalled_bearing),
                    extra_rejections=(
                        []
                        if int(claim_stalled_visual_latch) > 0
                        else ["stalled_latch_inactive"]
                    ),
                )
                opportunistic_gate = {
                    "enabled": bool(args.multi_target_opportunistic_claims)
                    and len(target_sequence) > 1,
                    "passed": opportunistic_claim_info is not None,
                    "reject_reasons": (
                        []
                        if opportunistic_claim_info is not None
                        else ["no_ready_candidate"]
                    ),
                }
                if opportunistic_claim_info is not None:
                    opportunistic_gate["candidate"] = dict(opportunistic_claim_info)
                claim_gate_log = {
                    "accepted": bool(
                        standard_claim
                        or near_claim
                        or contact_claim
                        or stalled_visual_claim
                        or learned_model_claim
                        or opportunistic_claim_info is not None
                    ),
                    "seen": bool(seen),
                    "in_cone": bool(in_cone),
                    "seen_age_ticks": int(seen_age_ticks),
                    "area": _round_float(area, 4),
                    "bearing": _round_float(bearing, 4),
                    "mem_conf": _round_float(mem_conf, 4),
                    "read_score": None if read_score is None else _round_float(read_score, 4),
                    "read_gate_pass": bool(read_gate_pass),
                    "standard": standard_gate,
                    "near": near_gate,
                    "contact": contact_gate,
                    "stalled_visual": stalled_gate,
                    "success_proxy": claim_success_proxy_gate,
                    "learned_model": learned_model_gate,
                    "opportunistic_visible": opportunistic_gate,
                }
                if (
                    standard_claim
                    or near_claim
                    or contact_claim
                    or stalled_visual_claim
                    or learned_model_claim
                    or opportunistic_claim_info is not None
                ):
                    claim_reason = "standard"
                    claim_target_color = active_target_color
                    claim_target_index = int(target_index)
                    claim_mem_conf = float(mem_conf)
                    claim_area = float(area)
                    claim_bearing = float(bearing)
                    claim_seen_age_ticks = int(seen_age_ticks)
                    claim_read_score = read_score
                    claim_near_area_logit = near_area_logit
                    if learned_model_claim:
                        claim_reason = "learned_claim_success_model"
                        wall_metrics["claim_success_model_trigger_claims"] += 1
                    elif contact_claim and not standard_claim and not near_claim:
                        claim_reason = "contact_visual"
                    elif near_claim and not standard_claim:
                        claim_reason = "near_visual"
                    elif stalled_visual_claim and not standard_claim and not near_claim and not contact_claim:
                        claim_reason = "stalled_visual"
                    elif opportunistic_claim_info is not None and not (
                        standard_claim or near_claim or contact_claim or stalled_visual_claim
                    ):
                        claim_reason = "opportunistic_visible"
                        claim_target_color = str(opportunistic_claim_info["target_color"])
                        claim_target_index = int(opportunistic_claim_info["target_index"])
                        claim_mem_conf = float(opportunistic_claim_info["mem_conf"])
                        claim_area = float(opportunistic_claim_info["area"])
                        claim_bearing = float(opportunistic_claim_info["bearing"])
                        claim_seen_age_ticks = int(opportunistic_claim_info["visible_ticks"])
                        claim_read_score = opportunistic_claim_info.get("read_score")
                        claim_near_area_logit = float(
                            opportunistic_claim_info["area_threshold"]
                        )
                        target_index = int(claim_target_index)
                        target_active_since_tick = int(tick)
                        wall_metrics["target_opportunistic_claims"] += 1
                        if claim_target_color not in first_seen_ticks:
                            first_seen_ticks[claim_target_color] = int(tick)
                    claim_dist = (
                        float(np.linalg.norm(np.asarray(pos[:2], dtype=np.float32) - landmarks[claim_target_color]))
                        if claim_target_color in landmarks
                        else None
                    )
                    claim_entry = {
                        "tick": tick, "state": "CLAIM", "target_color": claim_target_color,
                        "mem_conf": claim_mem_conf, "area": claim_area, "bearing": claim_bearing,
                        "seen_age_ticks": claim_seen_age_ticks,
                        "claim_reason": claim_reason,
                        "near_area_logit_threshold": (
                            None if claim_near_area_logit is None else float(claim_near_area_logit)
                        ),
                        "stalled_visual_latch": int(claim_stalled_visual_latch),
                        "dist_to_target_m": claim_dist,
                        "target_index": int(claim_target_index),
                        "claim_gate": claim_gate_log,
                    }
                    if opportunistic_claim_info is not None:
                        claim_entry["opportunistic_claim"] = dict(opportunistic_claim_info)
                    if claim_read_score is not None:
                        claim_entry["read_score"] = _round_float(claim_read_score, 4)
                    if bool(args.log_color_readouts):
                        claim_entry["color_readouts"] = target_color_readouts
                        if target_switch_candidates:
                            claim_entry["target_switch_candidates"] = target_switch_candidates
                    if target_switch_info is not None:
                        claim_entry["target_switch"] = target_switch_info
                    log.append(claim_entry)
                    beacon_claims.append(claim_entry)
                    if learned_local_online_map is not None:
                        learned_local_online_map.mark_claim(pos[:2])
                        if bool(args.explore_clear_visited_on_claim):
                            learned_local_online_map.reset_after_claim(
                                pos[:2],
                                tick=int(tick),
                            )
                    explorer.notify_claim(pos[:2])
                    if oracle_standoff_explorer is not None:
                        oracle_standoff_explorer.notify_claim(pos[:2])
                    if (
                        str(args.explore_goal_policy).lower() == "learned_wall_follow"
                        and bool(args.learned_wall_follow_flip_on_claim)
                    ):
                        learned_wall_follow_side = (
                            "left" if learned_wall_follow_side == "right" else "right"
                        )
                        learned_wall_follow_side_ticks = 0
                        learned_wall_follow_turn_run = 0
                        wall_metrics["learned_wall_follow_side_switches"] += 1
                    claim_entry["explore_route_progress"] = {
                        "route_idx": int(getattr(explorer, "route_idx", 0)),
                        "route_len": int(len(getattr(explorer, "route_waypoints", []))),
                        "route_claim_count": int(getattr(explorer, "route_claim_count", 0)),
                        "route_active": bool(
                            getattr(explorer, "_route_active", lambda: False)()
                        ),
                        "standoff_route": bool(getattr(explorer, "standoff_route", False)),
                        "standoff_target_color": getattr(explorer, "standoff_target_color", None),
                        "standoff_goal_xy": (
                            list(getattr(explorer, "standoff_goal_xy"))
                            if getattr(explorer, "standoff_goal_xy", None) is not None
                            else None
                        ),
                        "standoff_replans": int(getattr(explorer, "standoff_replans", 0)),
                    }
                    claimed_color_set = {str(item.get("target_color")) for item in beacon_claims}
                    remaining_indices = [
                        i for i, color in enumerate(target_sequence)
                        if color not in claimed_color_set
                    ]
                    if not remaining_indices:
                        claimed = True
                        break
                    next_indices = [i for i in remaining_indices if i > int(target_index)]
                    target_index = int((next_indices or remaining_indices)[0])
                    target_active_since_tick = int(tick) + 1
                    active_target_color = target_sequence[target_index]
                    active_target_tc = color_vocab.index(active_target_color)
                    body_clearance_latch = 0
                    claim_stalled_visual_latch = 0
                    stuck_streak = 0
                    turn_streak = 0
                    escape_plan = []
                    post_claim_explore_plan = []
                    if len(beacon_claims) >= max(
                        0,
                        int(args.post_claim_explore_min_claimed_count),
                    ):
                        post_claim_explore_plan = list(post_claim_explore_primitives)
                    if post_claim_explore_plan:
                        wall_metrics["post_claim_explore_plans"] += 1
                        wall_metrics["post_claim_explore_blocks_scheduled"] += len(
                            post_claim_explore_plan
                        )
                        claim_entry["post_claim_explore_plan"] = list(post_claim_explore_plan)
                    if bool(args.explore_reset_on_claim):
                        explorer.reset_route_state(
                            clear_visited=bool(args.explore_clear_visited_on_claim)
                        )
                        wall_metrics["explore_claim_route_resets"] += 1
                        claim_entry["explore_route_reset"] = {
                            "clear_visited": bool(args.explore_clear_visited_on_claim),
                            "blocked_cells_kept": int(len(getattr(explorer, "blocked", []))),
                            "visited_cells_after": int(len(getattr(explorer, "visited", []))),
                        }
                    last_primitive = "hold"
                    last_cmd = _PRIM_CMD["hold"]
                    last_primitive_run_ticks = 0
                    if (
                        args.slice_snapshot_output is not None
                        and int(args.slice_snapshot_after_claims) > 0
                        and len(beacon_claims) >= int(args.slice_snapshot_after_claims)
                        and not slice_snapshot_saved
                    ):
                        _write_slice_snapshot(
                            args.slice_snapshot_output,
                            build=build,
                            runner=runner,
                            scene_id=scene_dir.name,
                            tick=int(tick),
                            next_tick=int(tick) + 1,
                            pos=np.asarray(pos, dtype=np.float32),
                            quat=np.asarray(quat, dtype=np.float32),
                            yaw=float(yaw),
                            ctrl_state=ctrl_state,
                            target_sequence=target_sequence,
                            target_index=int(target_index),
                            target_active_since_tick=int(target_active_since_tick),
                            beacon_claims=beacon_claims,
                            first_seen_ticks=first_seen_ticks,
                            last_seen_ticks=last_seen_ticks,
                            last_primitive=last_primitive,
                            last_cmd=last_cmd,
                            online_map=learned_local_online_map,
                            feature_max_ticks=int(feature_max_ticks),
                            source_result=str(args.output),
                        )
                        slice_snapshot_saved = True
                        wall_metrics["slice_snapshot_output"] = str(args.slice_snapshot_output)
                        wall_metrics["slice_snapshot_tick"] = int(tick)
                        if bool(args.slice_snapshot_exit):
                            break
                    learned_local_turn_balance = 0
                    learned_local_turn_run = 0
                    learned_local_policy_turn_run = 0
                    weak_memory_seek_stall_streak = 0
                    weak_memory_seek_explore_cooldown = 0
                    target_pursuit_stale_streak = 0
                    target_pursuit_escape_cooldown = 0
                    target_pursuit_stale_last_color = None
                    target_pursuit_stale_window_ticks_by_color.clear()
                    prev_pose = cur_pose
                    continue
                if bool(args.explore_standoff_release_on_seen) and seen:
                    standoff_goal_xy = getattr(explorer, "standoff_goal_xy", None)
                    if standoff_goal_xy is not None:
                        standoff_route_release_dist_m = math.hypot(
                            float(standoff_goal_xy[0]) - float(pos[0]),
                            float(standoff_goal_xy[1]) - float(pos[1]),
                        )
                        standoff_route_released_on_seen = bool(
                            seen_age_ticks >= max(0, int(args.explore_standoff_release_min_seen_ticks))
                            and standoff_route_release_dist_m <= float(args.explore_standoff_release_m)
                        )
                standoff_route_wants_gate = (
                    bool(args.explore_standoff_route)
                    and args.explore_standoff_route_until_area_logit is not None
                    and (
                        (not state_seen)
                        or (not in_cone)
                        or area < float(args.explore_standoff_route_until_area_logit)
                    )
                )
                standoff_route_gate = bool(
                    standoff_route_wants_gate and not standoff_route_released_on_seen
                )
                learned_topology_route_released_on_seen = bool(
                    learned_topology_route_table is not None
                    and args.learned_topology_route_release_on_seen_area_logit is not None
                    and bool(seen)
                    and float(area)
                    >= float(args.learned_topology_route_release_on_seen_area_logit)
                )
                learned_topology_route_wants_gate = (
                    learned_topology_route_table is not None
                    and args.learned_topology_route_until_area_logit is not None
                    and not learned_topology_route_released_on_seen
                    and (
                        (not state_seen)
                        or (not in_cone)
                        or area < float(args.learned_topology_route_until_area_logit)
                    )
                )
                learned_topology_route_selected_this_tick = False
                if standoff_route_wants_gate and standoff_route_released_on_seen:
                    wall_metrics["explore_standoff_releases_on_seen"] += 1
                if learned_topology_route_released_on_seen:
                    wall_metrics["learned_topology_route_release_on_seen_ticks"] += 1
                if standoff_route_gate:
                    primitive = explorer.primitive(pos, yaw, target_color=active_target_color)
                    st = "EXPLORE"
                    bearing_for_guard = explorer.last_bearing
                    explorer_trace = explorer.trace()
                elif learned_topology_route_wants_gate or not state_seen:
                    st = "EXPLORE"
                    if learned_topology_route_table is not None:
                        route_yaw_threshold = learned_topology_route_yaw_threshold_by_color.get(
                            str(active_target_color),
                            float(args.learned_topology_route_yaw_threshold),
                        )
                        route_arc_max_bearing = learned_topology_route_arc_max_bearing_by_color.get(
                            str(active_target_color),
                            float(args.learned_topology_route_arc_max_bearing),
                        )
                        route_primitive, route_log = _select_learned_topology_route_primitive(
                            route_table=learned_topology_route_table,
                            route_state=learned_topology_route_state,
                            target_color=active_target_color,
                            target_index=int(target_index),
                            pos_xy=pos[:2],
                            yaw=float(yaw),
                            advance_m=float(args.learned_topology_route_advance_m),
                            lookahead_m=float(args.learned_topology_route_lookahead_m),
                            reproject_window=int(
                                args.learned_topology_route_reproject_window
                            ),
                            reproject_trigger_m=float(
                                args.learned_topology_route_reproject_trigger_m
                            ),
                            yaw_bearing_threshold=float(route_yaw_threshold),
                            forward_bearing_threshold=float(args.learned_topology_route_forward_threshold),
                            arc_max_bearing=float(route_arc_max_bearing),
                            forward_primitive=str(args.explore_forward_primitive),
                            use_stored_primitives=bool(
                                args.learned_topology_route_use_stored_primitives
                            ),
                        )
                        wall_metrics["learned_topology_route_ticks"] += 1
                        if learned_topology_route_wants_gate and state_seen:
                            wall_metrics["learned_topology_route_seen_gate_ticks"] += 1
                        if route_primitive != str(args.explore_forward_primitive):
                            wall_metrics["learned_topology_route_overrides"] += 1
                        if bool(route_log.get("reprojected")):
                            wall_metrics["learned_topology_route_reprojects"] += 1
                        primitive = route_primitive
                        learned_topology_route_selected_this_tick = True
                        bearing_for_guard = (
                            None
                            if route_log.get("bearing") is None
                            else float(route_log.get("bearing"))
                        )
                        explorer_trace = {
                            "policy": "learned_topology_route_memory",
                            "privileged_grid_explorer_skipped": True,
                            "learned_topology_route": route_log,
                        }
                        wall_metrics[
                            "learned_topology_route_privileged_explorer_skipped_ticks"
                        ] += 1
                    elif (
                        learned_local_policy_model is not None
                        and learned_local_policy_checkpoint is not None
                        and str(args.explore_goal_policy).lower() == "learned_policy"
                    ):
                        primitive = str(args.explore_forward_primitive)
                        bearing_for_guard = bearing
                        explorer_trace = {
                            "policy": "learned_local_policy",
                            "privileged_grid_explorer_skipped": True,
                            "runtime_safe_request": primitive,
                        }
                        wall_metrics["learned_local_policy_privileged_explorer_skipped_ticks"] += 1
                    else:
                        primitive = explorer.primitive(pos, yaw, target_color=active_target_color)
                        bearing_for_guard = explorer.last_bearing
                        explorer_trace = explorer.trace()
                    if learned_topology_route_table is not None:
                        explorer_trace["learned_topology_route"] = route_log
                elif in_cone:
                    latched_body_clearance = (
                        bool(args.body_clearance_target_servo)
                        and int(args.body_clearance_latch_ticks) > 0
                        and int(body_clearance_latch) > 0
                    )
                    near_body_clearance = (
                        bool(args.body_clearance_target_servo)
                        and (
                            area >= float(args.body_clearance_target_area_logit)
                            or latched_body_clearance
                        )
                    )
                    if near_body_clearance and abs(bearing) >= float(args.body_clearance_target_bearing):
                        primitive = "yaw_left" if bearing > 0 else "yaw_right"
                        body_clearance_request = True
                    elif near_body_clearance:
                        primitive = str(args.body_clearance_target_forward_primitive)
                        body_clearance_request = primitive != "forward_medium"
                    elif abs(bearing) < 0.15:
                        primitive = "forward_medium"
                    else:
                        primitive = "arc_left" if bearing > 0 else "arc_right"
                    st = "SERVO"
                else:
                    latched_body_clearance = (
                        bool(args.body_clearance_target_servo)
                        and int(args.body_clearance_latch_ticks) > 0
                        and int(body_clearance_latch) > 0
                    )
                    if latched_body_clearance and abs(bearing) >= float(args.body_clearance_target_bearing):
                        primitive = "yaw_left" if bearing > 0 else "yaw_right"
                        body_clearance_request = True
                    elif latched_body_clearance:
                        primitive = str(args.body_clearance_target_forward_primitive)
                        body_clearance_request = primitive != "forward_medium"
                    elif bearing > 0.1:
                        primitive = "yaw_left"
                    elif bearing < -0.1:
                        primitive = "yaw_right"
                    else:
                        primitive = "forward_medium"
                    st = "SEEK"
                log_entry = {"tick": tick, "state": st, "target_color": active_target_color,
                             "target_index": int(target_index),
                             "primitive": primitive, "mem_conf": round(mem_conf, 3),
                             "area": round(area, 2), "bearing": round(bearing, 2), "in_cone": in_cone,
                             "seen": bool(seen), "state_seen": bool(state_seen),
                             "seen_age_ticks": seen_age_ticks,
                             "claim_gate": claim_gate_log}
                if read_score is not None:
                    log_entry["read_score"] = _round_float(read_score, 4)
                    if args.seen_read_threshold is not None:
                        log_entry["read_gate_pass"] = bool(read_gate_pass)
                if bool(args.log_color_readouts):
                    log_entry["color_readouts"] = target_color_readouts
                    if target_switch_candidates:
                        log_entry["target_switch_candidates"] = target_switch_candidates
                if target_switch_info is not None:
                    log_entry["target_switch"] = target_switch_info
                if weak_memory_recovery_active:
                    log_entry["weak_memory_seek"] = {
                        "force_explore": bool(weak_memory_force_explore),
                        "conf_recovery_active": bool(weak_memory_conf_recovery_active),
                        "visual_recovery_active": bool(weak_memory_visual_recovery_active),
                        "cooldown": int(weak_memory_seek_explore_cooldown),
                        "stall_streak": int(weak_memory_seek_stall_streak),
                        "force_explore_on_recovery": bool(
                            args.weak_memory_seek_force_explore_on_recovery
                        ),
                        "conf_threshold": float(args.weak_memory_seek_conf),
                        "area_logit_threshold": (
                            None
                            if args.weak_memory_seek_area_logit is None
                            else float(args.weak_memory_seek_area_logit)
                        ),
                    }
                if target_pursuit_force_explore:
                    log_entry["target_pursuit_stale_escape"] = {
                        "force_explore": True,
                        "cooldown": int(target_pursuit_escape_cooldown),
                        "streak": int(target_pursuit_stale_streak),
                    }
                if active_target_stale_suppressed:
                    log_entry["target_pursuit_stale_suppressed_active"] = {
                        "color": str(active_target_color),
                        "until_tick": int(active_target_stale_suppressed_until),
                        "remaining_ticks": max(
                            0,
                            int(active_target_stale_suppressed_until) - int(tick),
                        ),
                    }
                if st == "EXPLORE":
                    log_entry["standoff_route_gate"] = bool(standoff_route_gate)
                if standoff_route_released_on_seen:
                    log_entry["standoff_route_released_on_seen"] = True
                    log_entry["standoff_route_release_dist_m"] = _round_float(
                        standoff_route_release_dist_m, 4
                    )
                if learned_topology_route_released_on_seen:
                    log_entry["learned_topology_route_released_on_seen"] = True
                    log_entry["learned_topology_route_release_area_logit"] = _round_float(
                        float(args.learned_topology_route_release_on_seen_area_logit),
                        4,
                    )
                if st == "SERVO" and bool(args.body_clearance_target_servo):
                    log_entry["body_clearance_target_servo"] = bool(
                        area >= float(args.body_clearance_target_area_logit)
                        or (
                            int(args.body_clearance_latch_ticks) > 0
                            and int(body_clearance_latch) > 0
                        )
                    )
                if body_clearance_request:
                    log_entry["body_clearance_request"] = True
                    log_entry["body_clearance_latched"] = bool(
                        area < float(args.body_clearance_target_area_logit)
                        and int(body_clearance_latch) > 0
                    )
                if st == "EXPLORE":
                    log_entry["explorer"] = explorer_trace

        if (
            primitive_aux_clearance_switch_requested
            and args.current_body_risk_min_area_logit is not None
        ):
            current_body_risk_area_active_for_switch = bool(
                log_entry is not None
                and "area" in log_entry
                and float(log_entry["area"]) >= float(args.current_body_risk_min_area_logit)
            )
            switch_claim_gate_pass = bool(
                len(beacon_claims)
                >= int(effective_aux_clearance_switch_min_claimed_count)
            )
            switch_triggered = bool(
                switch_claim_gate_pass
                and current_body_risk_area_active_for_switch
                and current_body_risk_prob is not None
                and primitive_aux_clearance_switch_threshold is not None
                and float(current_body_risk_prob)
                >= float(primitive_aux_clearance_switch_threshold)
            )
            if not switch_claim_gate_pass:
                wall_metrics[
                    "primitive_aux_clearance_switch_claim_gate_suppressed_ticks"
                ] += 1
                if (
                    current_body_risk_prob is not None
                    and primitive_aux_clearance_switch_threshold is not None
                    and float(current_body_risk_prob)
                    >= float(primitive_aux_clearance_switch_threshold)
                ):
                    wall_metrics[
                        "primitive_aux_clearance_switch_claim_gate_suppressed_high_risk_ticks"
                    ] += 1
            elif (
                not current_body_risk_area_active_for_switch
                and current_body_risk_prob is not None
                and primitive_aux_clearance_switch_threshold is not None
                and float(current_body_risk_prob)
                >= float(primitive_aux_clearance_switch_threshold)
            ):
                wall_metrics[
                    "primitive_aux_clearance_switch_area_suppressed_ticks"
                ] += 1
            if switch_triggered:
                primitive_aux_clearance_switch_latch = max(
                    int(primitive_aux_clearance_switch_latch),
                    int(args.primitive_aux_clearance_switch_latch_ticks),
                )
            switch_latched = int(primitive_aux_clearance_switch_latch) > 0
            if switch_triggered or switch_latched:
                primitive_aux_clearance_switch_active = True
                wall_metrics["primitive_aux_clearance_switch_ticks"] += 1
                if switch_latched:
                    primitive_aux_clearance_switch_latch = max(
                        0,
                        int(primitive_aux_clearance_switch_latch) - 1,
                    )
            else:
                wall_metrics["primitive_aux_clearance_switch_suppressed_ticks"] += 1

        learned_local_state_name = (
            "" if log_entry is None else str(log_entry.get("state", "")).upper()
        )
        learned_local_target_policy_state_active = bool(
            log_entry is not None
            and learned_local_state_name
            and (
                f"{str(active_target_color).lower()}:{learned_local_state_name}"
                in learned_local_target_policy_models
                or str(active_target_color).lower() in learned_local_target_policy_models
            )
        )
        learned_local_policy_state_active = bool(
            log_entry is not None
            and (
                not learned_local_policy_states
                or learned_local_state_name in learned_local_policy_states
                or learned_local_target_policy_state_active
                or (
                    bool(beacon_claims)
                    and learned_local_state_name in learned_local_policy_post_claim_states
                )
            )
        )
        learned_local_dataset_state_active = bool(
            log_entry is not None
            and args.learned_local_dataset_output is not None
            and len(beacon_claims) >= max(0, int(args.learned_local_dataset_min_claimed_count))
            and (
                not learned_local_dataset_states
                or learned_local_state_name in learned_local_dataset_states
                or learned_local_target_policy_state_active
                or (
                    bool(beacon_claims)
                    and learned_local_state_name in learned_local_policy_post_claim_states
                )
            )
        )
        use_post_claim_outcome = bool(
            bool(beacon_claims)
            and learned_local_state_name in learned_local_policy_post_claim_states
            and post_claim_outcome_encoder is not None
            and post_claim_outcome_head is not None
            and ego64 is not None
        )
        if use_post_claim_outcome:
            post_claim_outcome_input = ego64
            if int(post_claim_outcome_image_size) != 64:
                post_claim_outcome_input = F.interpolate(
                    post_claim_outcome_input.unsqueeze(0),
                    size=(int(post_claim_outcome_image_size), int(post_claim_outcome_image_size)),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            with torch.no_grad():
                post_claim_outcome_latent = post_claim_outcome_encoder(
                    post_claim_outcome_input.unsqueeze(0).to(device)
                )
            primitive_outcomes = _predict_primitive_outcomes(
                post_claim_outcome_head,
                post_claim_outcome_latent,
                primitive_vocab=post_claim_outcome_primitive_vocab,
                device=device,
            )
            primitive_outcomes = _fuse_clearance_outcomes(
                primitive_outcomes,
                primitive_clearance_outcomes,
            )
            primitive_outcome_slot = "post_claim"
            wall_metrics["primitive_post_claim_outcome_ticks"] += 1
        primitive_guard_outcomes = primitive_outcomes
        if primitive_aux_clearance_switch_active:
            primitive_guard_outcomes = _fuse_clearance_outcomes(
                primitive_outcomes,
                primitive_aux_clearance_outcomes,
            )
            primitive_clearance_slot = "aux_switch"
            if bool(args.primitive_aux_clearance_switch_policy_features):
                primitive_outcomes = primitive_guard_outcomes
                primitive_policy_clearance_slot = "aux_switch"
        learned_policy_feature_for_dataset: torch.Tensor | None = None
        learned_policy_feature_slot_for_dataset = "primary"
        if (
            log_entry is not None
            and (learned_local_policy_state_active or learned_local_dataset_state_active)
            and color_vocab is not None
            and active_target_tc is not None
        ):
            learned_online_map_feature = (
                None
                if learned_local_online_map is None
                else learned_local_online_map.feature(
                    pos[:2],
                    float(yaw),
                    tick=int(tick),
                    device=device,
                    channel_count=int(learned_local_online_map_channel_count),
                )
            )
            learned_primary_policy_feature = _learned_local_policy_feature(
                ctrl_state=ctrl_state,
                outputs=outputs,
                color_vocab=color_vocab,
                active_target_tc=int(active_target_tc),
                beacon_claims=beacon_claims,
                primitive_outcomes=primitive_outcomes,
                last_primitive=last_primitive,
                tick=int(tick),
                max_ticks=int(feature_max_ticks),
                append_clock_features=bool(learned_local_primary_feature_flags["clock"]),
                append_state_features=bool(learned_local_primary_feature_flags["state"]),
                append_visual_readout_features=bool(
                    learned_local_primary_feature_flags["visual_readout"]
                ),
                append_online_map_features=bool(learned_local_primary_feature_flags["online_map"]),
                online_map_feature=learned_online_map_feature,
                online_map_feature_dim=int(learned_local_online_map_feature_dim),
                controller_state_name=str(log_entry.get("state", "")),
                append_pose_topology_features=bool(
                    learned_local_primary_feature_flags["pose_topology"]
                ),
                pose_xy=pos[:2],
                yaw_rad=float(yaw),
                pose_scale_m=float(args.learned_local_pose_scale_m),
                device=device,
            )
            learned_post_claim_policy_feature = learned_primary_policy_feature
            if learned_local_post_claim_policy_checkpoint is not None:
                learned_post_claim_policy_feature = _learned_local_policy_feature(
                    ctrl_state=ctrl_state,
                    outputs=outputs,
                    color_vocab=color_vocab,
                    active_target_tc=int(active_target_tc),
                    beacon_claims=beacon_claims,
                    primitive_outcomes=primitive_outcomes,
                    last_primitive=last_primitive,
                    tick=int(tick),
                    max_ticks=int(feature_max_ticks),
                    append_clock_features=bool(learned_local_post_claim_feature_flags["clock"]),
                    append_state_features=bool(learned_local_post_claim_feature_flags["state"]),
                    append_visual_readout_features=bool(
                        learned_local_post_claim_feature_flags["visual_readout"]
                    ),
                    append_online_map_features=bool(
                        learned_local_post_claim_feature_flags["online_map"]
                    ),
                    online_map_feature=learned_online_map_feature,
                    online_map_feature_dim=int(learned_local_online_map_feature_dim),
                    controller_state_name=str(log_entry.get("state", "")),
                    append_pose_topology_features=bool(
                        learned_local_post_claim_feature_flags["pose_topology"]
                    ),
                    pose_xy=pos[:2],
                    yaw_rad=float(yaw),
                    pose_scale_m=float(args.learned_local_pose_scale_m),
                    device=device,
                )
            learned_target_policy_features: dict[str, torch.Tensor | None] = {}
            for target_policy_color, target_flags in learned_local_target_policy_feature_flags.items():
                learned_target_policy_features[target_policy_color] = _learned_local_policy_feature(
                    ctrl_state=ctrl_state,
                    outputs=outputs,
                    color_vocab=color_vocab,
                    active_target_tc=int(active_target_tc),
                    beacon_claims=beacon_claims,
                    primitive_outcomes=primitive_outcomes,
                    last_primitive=last_primitive,
                    tick=int(tick),
                    max_ticks=int(feature_max_ticks),
                    append_clock_features=bool(target_flags["clock"]),
                    append_state_features=bool(target_flags["state"]),
                    append_visual_readout_features=bool(target_flags["visual_readout"]),
                    append_online_map_features=bool(target_flags["online_map"]),
                    online_map_feature=learned_online_map_feature,
                    online_map_feature_dim=int(learned_local_online_map_feature_dim),
                    controller_state_name=str(log_entry.get("state", "")),
                    append_pose_topology_features=bool(target_flags["pose_topology"]),
                    pose_xy=pos[:2],
                    yaw_rad=float(yaw),
                    pose_scale_m=float(args.learned_local_pose_scale_m),
                    device=device,
                )
            learned_policy_feature = learned_primary_policy_feature
            learned_policy_feature_slot = "primary"
            if (
                learned_local_post_claim_policy_checkpoint is not None
                and bool(beacon_claims)
                and len(beacon_claims) >= max(
                    1,
                    int(args.learned_local_post_claim_policy_min_claims),
                )
                and learned_local_state_name in learned_local_policy_post_claim_states
            ):
                learned_policy_feature = learned_post_claim_policy_feature
                learned_policy_feature_slot = "post_claim"
            dataset_target_policy_color = str(active_target_color).lower()
            dataset_target_policy_state_key = (
                f"{dataset_target_policy_color}:{learned_local_state_name}"
            )
            dataset_target_policy_key = (
                dataset_target_policy_state_key
                if dataset_target_policy_state_key in learned_local_target_policy_models
                else dataset_target_policy_color
            )
            dataset_target_policy_available = bool(
                dataset_target_policy_key in learned_local_target_policy_models
                and dataset_target_policy_key in learned_local_target_policy_checkpoints
            )
            dataset_post_claim_policy_available = bool(
                bool(beacon_claims)
                and len(beacon_claims) >= max(
                    1,
                    int(args.learned_local_post_claim_policy_min_claims),
                )
                and learned_local_state_name in learned_local_policy_post_claim_states
                and learned_local_post_claim_policy_model is not None
                and learned_local_post_claim_policy_checkpoint is not None
            )
            dataset_use_target_policy = bool(
                dataset_target_policy_available
                and (
                    bool(args.learned_local_target_policy_priority_over_post_claim)
                    or (
                        bool(
                            args.learned_local_target_policy_priority_on_aux_clearance_switch
                        )
                        and bool(primitive_aux_clearance_switch_active)
                    )
                    or not dataset_post_claim_policy_available
                )
            )
            if dataset_use_target_policy:
                target_feature = (
                    learned_target_policy_features.get(dataset_target_policy_color)
                    if dataset_target_policy_key == dataset_target_policy_color
                    else learned_target_policy_features.get(dataset_target_policy_key)
                )
                if target_feature is not None:
                    learned_policy_feature = target_feature
                    learned_policy_feature_slot = f"target:{dataset_target_policy_key}"
            learned_policy_feature_for_dataset = learned_policy_feature
            learned_policy_feature_slot_for_dataset = learned_policy_feature_slot
            if (
                args.learned_local_dataset_output is not None
                and learned_local_dataset_state_active
                and learned_policy_feature is not None
                and str(args.learned_local_dataset_label_source) == "teacher"
            ):
                teacher_label_primitive = _learned_local_policy_label_primitive(primitive)
                if teacher_label_primitive is None:
                    wall_metrics["learned_local_policy_skipped_unmapped_examples"] += 1
                else:
                    if teacher_label_primitive != primitive:
                        wall_metrics["learned_local_policy_label_mapped_examples"] += 1
                    oracle_label_used = False
                    if (
                        oracle_standoff_explorer is not None
                        and str(log_entry.get("state", "")).upper()
                        in learned_local_oracle_standoff_label_states
                    ):
                        oracle_primitive = oracle_standoff_explorer.primitive(
                            pos,
                            yaw,
                            target_color=active_target_color,
                        )
                        oracle_label_used = True
                        wall_metrics["learned_local_oracle_standoff_label_ticks"] += 1
                        oracle_label_primitive = _learned_local_policy_label_primitive(oracle_primitive)
                        if oracle_label_primitive is None:
                            wall_metrics["learned_local_policy_skipped_unmapped_examples"] += 1
                        else:
                            if oracle_label_primitive != oracle_primitive:
                                wall_metrics["learned_local_policy_label_mapped_examples"] += 1
                            if oracle_label_primitive != teacher_label_primitive:
                                wall_metrics["learned_local_oracle_standoff_label_overrides"] += 1
                            teacher_label_primitive = oracle_label_primitive
                    if teacher_label_primitive is not None:
                        feature_np = (
                            learned_policy_feature.detach().cpu().numpy().astype(np.float32)
                        )
                        feature_dim = int(feature_np.reshape(-1).shape[0])
                        if learned_local_dataset_feature_dim is None:
                            learned_local_dataset_feature_dim = feature_dim
                        if feature_dim != int(learned_local_dataset_feature_dim):
                            wall_metrics["learned_local_policy_skipped_feature_dim_examples"] += 1
                        else:
                            learned_local_dataset_features.append(feature_np)
                            learned_local_dataset_labels.append(
                                _LEARNED_LOCAL_POLICY_PRIMITIVES.index(teacher_label_primitive)
                            )
                            learned_local_dataset_meta.append(
                                {
                                "tick": int(tick),
                                "state": str(log_entry.get("state", "")),
                                "label": teacher_label_primitive,
                                "dataset_label_source": "teacher",
                                "policy_feature_slot": str(learned_policy_feature_slot),
                                "executed_label_source_primitive": primitive,
                                "oracle_standoff_label": bool(oracle_label_used),
                                "target_color": str(log_entry.get("target_color", "")),
                                "target_index": int(log_entry.get("target_index", -1)),
                                "mem_conf": float(log_entry.get("mem_conf", 0.0)),
                                "area": float(log_entry.get("area", -99.0)),
                                "bearing": float(log_entry.get("bearing", 0.0)),
                                "in_cone": bool(log_entry.get("in_cone", False)),
                                "seen": bool(log_entry.get("seen", False)),
                                "state_seen": bool(log_entry.get("state_seen", False)),
                                "read_score": (
                                    None
                                    if log_entry.get("read_score") is None
                                    else float(log_entry.get("read_score", 0.0))
                                ),
                                "read_gate_pass": (
                                    None
                                    if log_entry.get("read_gate_pass") is None
                                    else bool(log_entry.get("read_gate_pass", False))
                                ),
                                "seen_age_ticks": int(log_entry.get("seen_age_ticks", 0)),
                                "pose_xy": [float(pos[0]), float(pos[1])],
                                "yaw_rad": float(yaw),
                                "claimed_count": int(len(beacon_claims)),
                                "standoff_route_gate": bool(log_entry.get("standoff_route_gate", False)),
                                }
                            )
                            wall_metrics["learned_local_policy_collected_examples"] += 1
            post_claim_policy_available = bool(
                bool(beacon_claims)
                and len(beacon_claims) >= max(
                    1,
                    int(args.learned_local_post_claim_policy_min_claims),
                )
                and learned_local_state_name in learned_local_policy_post_claim_states
                and learned_local_post_claim_policy_model is not None
                and learned_local_post_claim_policy_checkpoint is not None
            )
            active_target_policy_color = str(active_target_color).lower()
            active_target_policy_state_key = (
                f"{active_target_policy_color}:{learned_local_state_name}"
            )
            active_target_policy_key = (
                active_target_policy_state_key
                if active_target_policy_state_key in learned_local_target_policy_models
                else active_target_policy_color
            )
            target_policy_available = bool(
                active_target_policy_key in learned_local_target_policy_models
                and active_target_policy_key in learned_local_target_policy_checkpoints
            )
            use_target_policy = bool(
                target_policy_available
                and (
                    bool(args.learned_local_target_policy_priority_over_post_claim)
                    or (
                        bool(
                            args.learned_local_target_policy_priority_on_aux_clearance_switch
                        )
                        and bool(primitive_aux_clearance_switch_active)
                    )
                    or not post_claim_policy_available
                )
            )
            use_post_claim_policy = bool(
                post_claim_policy_available
                and not use_target_policy
            )
            if use_post_claim_policy:
                active_policy_model = learned_local_post_claim_policy_model
                active_policy_checkpoint = learned_local_post_claim_policy_checkpoint
            elif use_target_policy:
                active_policy_model = learned_local_target_policy_models[active_target_policy_key]
                active_policy_checkpoint = learned_local_target_policy_checkpoints[
                    active_target_policy_key
                ]
            else:
                active_policy_model = learned_local_policy_model
                active_policy_checkpoint = learned_local_policy_checkpoint
            active_policy_hidden = (
                learned_local_post_claim_policy_recurrent_hidden
                if use_post_claim_policy
                else learned_local_primary_policy_recurrent_hidden
            )
            active_policy_feature = (
                learned_post_claim_policy_feature
                if use_post_claim_policy
                else (
                    learned_target_policy_features.get(active_target_policy_color)
                    if (
                        use_target_policy
                        and active_target_policy_key == active_target_policy_color
                    )
                    else learned_target_policy_features.get(active_target_policy_key)
                    if use_target_policy
                    else learned_primary_policy_feature
                )
            )
            if (
                learned_local_policy_state_active
                and active_policy_model is not None
                and active_policy_checkpoint is not None
                and str(args.explore_goal_policy).lower() == "learned_policy"
                and active_policy_feature is not None
            ):
                active_outcome_rerank = bool(args.learned_local_policy_outcome_rerank)
                if use_post_claim_policy:
                    post_claim_rerank_mode = str(
                        args.learned_local_post_claim_policy_outcome_rerank
                    ).lower()
                    if post_claim_rerank_mode == "on":
                        active_outcome_rerank = True
                    elif post_claim_rerank_mode == "off":
                        active_outcome_rerank = False
                elif use_target_policy:
                    target_rerank_mode = str(
                        args.learned_local_target_policy_outcome_rerank
                    ).lower()
                    if target_rerank_mode == "on":
                        active_outcome_rerank = True
                    elif target_rerank_mode == "off":
                        active_outcome_rerank = False
                active_rerank_policy_weight = float(
                    args.learned_local_policy_rerank_policy_weight
                )
                if (
                    use_post_claim_policy
                    and args.learned_local_post_claim_policy_rerank_policy_weight is not None
                ):
                    active_rerank_policy_weight = float(
                        args.learned_local_post_claim_policy_rerank_policy_weight
                    )
                elif (
                    use_target_policy
                    and args.learned_local_target_policy_rerank_policy_weight is not None
                ):
                    active_rerank_policy_weight = float(
                        args.learned_local_target_policy_rerank_policy_weight
                    )
                (
                    selected_policy_primitive,
                    learned_policy_log,
                    next_policy_hidden,
                ) = _select_learned_local_policy_primitive(
                    model=active_policy_model,
                    checkpoint=active_policy_checkpoint,
                    feature=active_policy_feature,
                    requested=primitive,
                    recurrent_hidden=active_policy_hidden,
                    primitive_outcomes=primitive_outcomes,
                    outcome_rerank=bool(active_outcome_rerank),
                    outcome_threshold=float(outcome_threshold if outcome_threshold is not None else 0.5),
                    forward_progress_floor=(
                        None
                        if (
                            args.primitive_outcome_forward_progress_floor is None
                            or (
                                primitive_outcome_forward_progress_floor_states
                                and log_entry is not None
                                and str(log_entry.get("state", "")).upper()
                                not in primitive_outcome_forward_progress_floor_states
                            )
                        )
                        else float(args.primitive_outcome_forward_progress_floor)
                    ),
                    rerank_top_k=int(args.learned_local_policy_rerank_top_k),
                    rerank_policy_weight=active_rerank_policy_weight,
                    rerank_blocked_weight=float(args.learned_local_policy_rerank_blocked_weight),
                    rerank_clearance_weight=float(args.learned_local_policy_rerank_clearance_weight),
                    rerank_progress_weight=float(args.learned_local_policy_rerank_progress_weight),
                    rerank_hard_blocked_penalty=float(
                        args.learned_local_policy_rerank_hard_blocked_penalty
                    ),
                    rerank_backward_penalty=float(
                        args.learned_local_policy_rerank_backward_penalty
                    ),
                    rerank_switch_margin=float(args.learned_local_policy_rerank_switch_margin),
                    rerank_protect_top_prob=float(
                        args.learned_local_policy_rerank_protect_top_prob
                    ),
                    rerank_override_min_prob=float(
                        args.learned_local_policy_rerank_override_min_prob
                    ),
                    bearing=(
                        None
                        if log_entry is None or "bearing" not in log_entry
                        else float(log_entry["bearing"])
                    ),
                    rerank_bearing_turn_threshold=float(
                        args.learned_local_policy_rerank_bearing_turn_threshold
                    ),
                    rerank_bearing_turn_bonus=float(
                        args.learned_local_policy_rerank_bearing_turn_bonus
                    ),
                    online_map=learned_local_online_map,
                    online_map_pose_xy=pos[:2],
                    online_map_yaw_rad=float(yaw),
                    online_map_tick=int(tick),
                    online_map_novelty_weight=float(
                        args.learned_local_policy_online_map_novelty_weight
                    ),
                    online_map_blocked_penalty=float(
                        args.learned_local_policy_online_map_blocked_penalty
                    ),
                    online_map_turn_scale=float(
                        args.learned_local_policy_online_map_turn_scale
                    ),
                    online_map_claim_repulsion_weight=float(
                        args.learned_local_policy_online_map_claim_repulsion_weight
                    ),
                    online_map_frontier_route_weight=float(
                        args.learned_local_policy_online_map_frontier_route_weight
                    ),
                    online_map_hard_veto=bool(
                        args.learned_local_policy_online_map_hard_veto
                    ),
                    controller_state_name=str(log_entry.get("state", "")),
                    online_map_novelty_states=learned_local_policy_online_map_novelty_states,
                )
                if use_post_claim_policy:
                    learned_local_post_claim_policy_recurrent_hidden = next_policy_hidden
                else:
                    learned_local_primary_policy_recurrent_hidden = next_policy_hidden
                if use_post_claim_policy:
                    active_policy_slot = "post_claim"
                elif use_target_policy:
                    active_policy_slot = f"target:{active_target_policy_key}"
                else:
                    active_policy_slot = "primary"
                learned_policy_log["policy_slot"] = active_policy_slot
                learned_policy_log["effective_outcome_rerank"] = bool(active_outcome_rerank)
                learned_policy_log["effective_rerank_policy_weight"] = float(
                    active_rerank_policy_weight
                )
                learned_policy_log["checkpoint_source"] = str(
                    active_policy_checkpoint.get("source", "")
                )
                wall_metrics["learned_local_policy_ticks"] += 1
                if use_post_claim_policy:
                    wall_metrics["learned_local_post_claim_policy_ticks"] += 1
                elif use_target_policy:
                    wall_metrics["learned_local_target_policy_ticks"] += 1
                    ticks_by_color = wall_metrics.get("learned_local_target_policy_ticks_by_color")
                    if isinstance(ticks_by_color, dict):
                        ticks_by_color[active_target_policy_key] = int(
                            ticks_by_color.get(active_target_policy_key, 0)
                        ) + 1
                else:
                    wall_metrics["learned_local_primary_policy_ticks"] += 1
                if not bool(learned_policy_log.get("enabled", False)):
                    wall_metrics["learned_local_policy_disabled_ticks"] += 1
                    if use_post_claim_policy:
                        wall_metrics["learned_local_post_claim_policy_disabled_ticks"] += 1
                    elif use_target_policy:
                        wall_metrics["learned_local_target_policy_disabled_ticks"] += 1
                    if str(learned_policy_log.get("reason", "")) == "feature_dim_mismatch":
                        wall_metrics["learned_local_policy_feature_mismatch_ticks"] += 1
                        if use_post_claim_policy:
                            wall_metrics[
                                "learned_local_post_claim_policy_feature_mismatch_ticks"
                            ] += 1
                        elif use_target_policy:
                            wall_metrics[
                                "learned_local_target_policy_feature_mismatch_ticks"
                            ] += 1
                if str(log_entry.get("state", "")).upper() == "EXPLORE":
                    wall_metrics["learned_local_policy_explore_state_ticks"] += 1
                if selected_policy_primitive != primitive:
                    wall_metrics["learned_local_policy_overrides"] += 1
                    if use_post_claim_policy:
                        wall_metrics["learned_local_post_claim_policy_overrides"] += 1
                    elif use_target_policy:
                        wall_metrics["learned_local_target_policy_overrides"] += 1
                outcome_rerank_log = learned_policy_log.get("outcome_rerank") or {}
                if bool(outcome_rerank_log.get("enabled")):
                    if outcome_rerank_log.get("selected_after") != outcome_rerank_log.get("selected_before"):
                        wall_metrics["learned_local_policy_outcome_rerank_overrides"] += 1
                    if bool(outcome_rerank_log.get("top_unsafe_forward")):
                        wall_metrics["learned_local_policy_outcome_rerank_unsafe_top_ticks"] += 1
                    if bool(outcome_rerank_log.get("online_map_novelty_enabled")):
                        wall_metrics["learned_local_policy_online_map_novelty_ticks"] += 1
                        if (
                            outcome_rerank_log.get("selected_after")
                            != outcome_rerank_log.get("selected_before")
                        ):
                            wall_metrics["learned_local_policy_online_map_novelty_overrides"] += 1
                primitive = selected_policy_primitive
                pressure_after = int(args.learned_local_policy_translation_pressure_after)
                if selected_policy_primitive in _TURN_PRIMITIVES:
                    learned_local_policy_turn_run = min(
                        64,
                        int(learned_local_policy_turn_run) + 1,
                    )
                else:
                    learned_local_policy_turn_run = 0
                wall_metrics["learned_local_policy_max_turn_run"] = max(
                    int(wall_metrics["learned_local_policy_max_turn_run"]),
                    int(learned_local_policy_turn_run),
                )
                learned_local_policy_pressure_run = max(
                    int(learned_local_policy_turn_run),
                    int(learned_local_policy_nonprogress_run),
                )
                frontier_selected = False
                frontier_after = int(args.learned_local_policy_frontier_pressure_after)
                frontier_pressure_state_allowed = bool(
                    not learned_local_policy_frontier_pressure_states
                    or str(learned_local_state_name).upper()
                    in learned_local_policy_frontier_pressure_states
                )
                if (
                    frontier_after > 0
                    and frontier_pressure_state_allowed
                    and (
                        learned_local_policy_pressure_run >= frontier_after
                        or bool(args.learned_local_policy_frontier_pressure_always)
                    )
                    and primitive_outcomes is not None
                    and learned_local_online_map is not None
                    and (
                        bool(learned_local_online_map.claimed)
                        or bool(args.learned_local_policy_frontier_pressure_pre_claim)
                    )
                ):
                    frontier_primitive, frontier_log = learned_local_online_map.frontier_pressure_primitive(
                        pose_xy=pos[:2],
                        yaw_rad=float(yaw),
                        predictions=primitive_outcomes,
                        candidate_primitives=learned_local_policy_translation_pressure_primitives,
                        max_blocked_prob=float(
                            args.learned_local_policy_frontier_pressure_max_blocked_prob
                        ),
                        min_progress_m=float(
                            args.learned_local_policy_frontier_pressure_min_progress_m
                        ),
                        min_route_cells=int(
                            args.learned_local_policy_frontier_pressure_min_route_cells
                        ),
                        guard_blocked_penalty=float(
                            args.learned_local_policy_frontier_pressure_guard_blocked_penalty
                        ),
                        allow_nonroute_backward_claim_escape=bool(
                            args.learned_local_policy_frontier_pressure_nonroute_backward_claim_escape
                        ),
                        prefer_unguarded_candidates=bool(
                            args.learned_local_policy_frontier_pressure_prefer_unguarded
                        ),
                        allow_map_blocked_backward_claim_escape=bool(
                            args.learned_local_policy_frontier_pressure_map_blocked_backward_claim_escape
                        ),
                        allow_guarded_retry=bool(
                            int(
                                args.learned_local_policy_frontier_pressure_guarded_retry_after_noops
                            )
                            > 0
                            and int(learned_local_policy_frontier_noop_run)
                            >= int(
                                args.learned_local_policy_frontier_pressure_guarded_retry_after_noops
                            )
                        ),
                        allow_combined_blocked_retry=bool(
                            int(
                                args.learned_local_policy_frontier_pressure_combined_blocked_retry_after_noops
                            )
                            > 0
                            and int(learned_local_policy_frontier_noop_run)
                            >= int(
                                args.learned_local_policy_frontier_pressure_combined_blocked_retry_after_noops
                            )
                        ),
                        probe_route_steps=bool(
                            args.learned_local_policy_frontier_pressure_probe_route_steps
                        ),
                    )
                    frontier_log["turn_run_before"] = int(learned_local_policy_turn_run)
                    frontier_log["nonprogress_run_before"] = int(
                        learned_local_policy_nonprogress_run
                    )
                    frontier_log["pressure_run_before"] = int(learned_local_policy_pressure_run)
                    frontier_log["frontier_noop_run_before"] = int(
                        learned_local_policy_frontier_noop_run
                    )
                    wall_metrics["learned_local_policy_frontier_pressure_ticks"] += 1
                    if frontier_primitive is None:
                        wall_metrics["learned_local_policy_frontier_pressure_noops"] += 1
                        learned_local_policy_frontier_noop_run = min(
                            1000000,
                            int(learned_local_policy_frontier_noop_run) + 1,
                        )
                    else:
                        primitive = frontier_primitive
                        learned_local_policy_turn_run = 0
                        learned_local_policy_nonprogress_run = 0
                        learned_local_policy_frontier_noop_run = 0
                        learned_local_policy_pressure_run = 0
                        frontier_selected = True
                        guarded_retry_no_commit = bool(
                            frontier_log.get("guarded_retry_no_commit")
                        )
                        frontier_pressure_committed = bool(
                            args.learned_local_policy_frontier_pressure_commit
                            and frontier_primitive in _TRANSLATING_PRIMITIVES
                            and not guarded_retry_no_commit
                        )
                        wall_metrics["learned_local_policy_frontier_pressure_overrides"] += 1
                        if guarded_retry_no_commit:
                            wall_metrics[
                                "learned_local_policy_frontier_pressure_guarded_retries"
                            ] += 1
                        if bool(frontier_log.get("combined_blocked_retry")):
                            wall_metrics[
                                "learned_local_policy_frontier_pressure_combined_blocked_retries"
                            ] += 1
                        if frontier_pressure_committed:
                            frontier_log["guard_commit_requested"] = True
                    route_next_cell = frontier_log.get("route_next")
                    learned_local_last_route_next = (
                        (int(route_next_cell[0]), int(route_next_cell[1]))
                        if route_next_cell
                        else None
                    )
                    learned_policy_log["frontier_pressure"] = frontier_log
                learned_local_policy_pressure_run = max(
                    int(learned_local_policy_turn_run),
                    int(learned_local_policy_nonprogress_run),
                )
                translation_pressure_state_allowed = bool(
                    not learned_local_policy_translation_pressure_states
                    or str(learned_local_state_name).upper()
                    in learned_local_policy_translation_pressure_states
                )
                if (
                    not frontier_selected
                    and pressure_after > 0
                    and translation_pressure_state_allowed
                    and learned_local_policy_pressure_run >= pressure_after
                    and primitive_outcomes is not None
                ):
                    pressure_primitive, pressure_log = (
                        _select_learned_policy_translation_pressure_primitive(
                            predictions=primitive_outcomes,
                            candidate_primitives=learned_local_policy_translation_pressure_primitives,
                            max_blocked_prob=float(
                                args.learned_local_policy_translation_pressure_max_blocked_prob
                            ),
                            min_progress_m=float(
                                args.learned_local_policy_translation_pressure_min_progress_m
                            ),
                            bearing=(
                                None
                                if log_entry is None or "bearing" not in log_entry
                                else float(log_entry["bearing"])
                            ),
                        )
                    )
                    pressure_log["turn_run_before"] = int(learned_local_policy_turn_run)
                    pressure_log["nonprogress_run_before"] = int(
                        learned_local_policy_nonprogress_run
                    )
                    pressure_log["pressure_run_before"] = int(learned_local_policy_pressure_run)
                    wall_metrics["learned_local_policy_translation_pressure_ticks"] += 1
                    if pressure_primitive is None:
                        wall_metrics["learned_local_policy_translation_pressure_noops"] += 1
                    else:
                        primitive = pressure_primitive
                        learned_local_policy_turn_run = 0
                        learned_local_policy_nonprogress_run = 0
                        wall_metrics["learned_local_policy_translation_pressure_overrides"] += 1
                    learned_policy_log["translation_pressure"] = pressure_log
                log_entry["primitive"] = primitive
                log_entry["learned_local_policy"] = learned_policy_log

        if (
            log_entry is not None
            and str(log_entry.get("state", "")).upper() == "EXPLORE"
            and str(args.explore_goal_policy).lower() == "learned_local"
        ):
            learned_local_primitive, learned_local_log = _select_learned_local_explore_primitive(
                tick=int(tick),
                requested=primitive,
                predictions=primitive_outcomes,
                scan_interval=int(args.explore_scan_interval),
                scan_len=int(args.explore_scan_len),
                scan_primitive=str(args.explore_scan_primitive),
                blocked_threshold=float(outcome_threshold if outcome_threshold is not None else 0.5),
                blocked_weight=float(args.primitive_outcome_blocked_weight),
                progress_weight=float(args.primitive_outcome_progress_weight),
                requested_bonus=float(args.primitive_outcome_requested_bonus),
                turn_progress_scale=float(args.primitive_outcome_turn_progress_scale),
                forward_progress_floor=(
                    None
                    if args.primitive_outcome_forward_progress_floor is None
                    else float(args.primitive_outcome_forward_progress_floor)
                ),
                forward_progress_floor_min_blocked_prob=(
                    None
                    if args.primitive_outcome_progress_floor_min_blocked_prob is None
                    else float(args.primitive_outcome_progress_floor_min_blocked_prob)
                ),
                forward_progress_floor_force_below=(
                    None
                    if args.primitive_outcome_progress_floor_force_below is None
                    else float(args.primitive_outcome_progress_floor_force_below)
                ),
                forward_progress_floor_penalty=float(args.primitive_outcome_forward_progress_penalty),
                turn_balance=int(learned_local_turn_balance),
                turn_run=int(learned_local_turn_run),
            )
            wall_metrics["learned_local_explore_ticks"] += 1
            if bool(learned_local_log.get("scan_active")):
                wall_metrics["learned_local_explore_scan_ticks"] += 1
            if bool(learned_local_log.get("translation_pressure_active")):
                wall_metrics["learned_local_translation_pressure_ticks"] += 1
            if learned_local_primitive != primitive:
                wall_metrics["learned_local_explore_overrides"] += 1
            if learned_local_primitive in ("yaw_left", "arc_left"):
                learned_local_turn_balance = min(24, int(learned_local_turn_balance) + 1)
            elif learned_local_primitive in ("yaw_right", "arc_right"):
                learned_local_turn_balance = max(-24, int(learned_local_turn_balance) - 1)
            elif learned_local_turn_balance > 0:
                learned_local_turn_balance -= 1
            elif learned_local_turn_balance < 0:
                learned_local_turn_balance += 1
            if learned_local_primitive in _TURN_PRIMITIVES:
                learned_local_turn_run = min(32, int(learned_local_turn_run) + 1)
            else:
                learned_local_turn_run = 0
            wall_metrics["learned_local_max_turn_run"] = max(
                int(wall_metrics["learned_local_max_turn_run"]),
                int(learned_local_turn_run),
            )
            learned_local_log["turn_balance_after"] = int(learned_local_turn_balance)
            learned_local_log["turn_run_after"] = int(learned_local_turn_run)
            primitive = learned_local_primitive
            log_entry["primitive"] = primitive
            log_entry["learned_local_explore"] = learned_local_log

        if (
            log_entry is not None
            and str(log_entry.get("state", "")).upper() == "EXPLORE"
            and str(args.explore_goal_policy).lower() == "learned_wall_follow"
        ):
            side_period = int(args.learned_wall_follow_side_period)
            learned_wall_follow_side_ticks += 1
            if side_period > 0 and learned_wall_follow_side_ticks > side_period:
                learned_wall_follow_side = (
                    "left" if learned_wall_follow_side == "right" else "right"
                )
                learned_wall_follow_side_ticks = 1
                learned_wall_follow_turn_run = 0
                wall_metrics["learned_wall_follow_side_switches"] += 1
            learned_wall_primitive, learned_wall_log = _select_learned_wall_follow_explore_primitive(
                tick=int(tick),
                requested=primitive,
                predictions=primitive_outcomes,
                scan_interval=int(args.explore_scan_interval),
                scan_len=int(args.explore_scan_len),
                scan_primitive=str(args.explore_scan_primitive),
                side=str(learned_wall_follow_side),
                turn_run=int(learned_wall_follow_turn_run),
                safe_risk=float(args.learned_wall_follow_safe_risk),
                progress_floor=float(args.learned_wall_follow_progress_floor),
                turn_pressure_after=int(args.learned_wall_follow_turn_pressure_after),
            )
            wall_metrics["learned_wall_follow_ticks"] += 1
            if bool(learned_wall_log.get("scan_active")):
                wall_metrics["learned_wall_follow_scan_ticks"] += 1
            if learned_wall_primitive != primitive:
                wall_metrics["learned_wall_follow_overrides"] += 1
            if learned_wall_primitive in _TURN_PRIMITIVES:
                learned_wall_follow_turn_run = min(32, int(learned_wall_follow_turn_run) + 1)
            else:
                learned_wall_follow_turn_run = 0
            wall_metrics["learned_wall_follow_max_turn_run"] = max(
                int(wall_metrics["learned_wall_follow_max_turn_run"]),
                int(learned_wall_follow_turn_run),
            )
            learned_wall_log["side_ticks"] = int(learned_wall_follow_side_ticks)
            learned_wall_log["turn_run_after"] = int(learned_wall_follow_turn_run)
            primitive = learned_wall_primitive
            log_entry["primitive"] = primitive
            log_entry["learned_wall_follow"] = learned_wall_log

        if log_entry is not None and post_claim_explore_plan:
            post_claim_primitive = post_claim_explore_plan.pop(0)
            primitive = post_claim_primitive
            log_entry["primitive"] = primitive
            log_entry["post_claim_explore_override"] = {
                "primitive": str(post_claim_primitive),
                "remaining_plan": list(post_claim_explore_plan),
            }
            wall_metrics["post_claim_explore_blocks_executed"] += 1

        requested_primitive = primitive
        guard_state = str(log_entry.get("state", "")) if log_entry is not None else ""
        if (
            broad_explorer_primitive is not None
            and guard_state.upper() in broad_explorer_states
            and not escape_plan
        ):
            if broad_explorer_primitive != primitive:
                wall_metrics["broad_explorer_overrides"] += 1
            primitive = broad_explorer_primitive
            requested_primitive = broad_explorer_primitive
            wall_metrics["broad_explorer_ticks"] += 1
            if log_entry is not None:
                log_entry["broad_explorer_primitive"] = broad_explorer_primitive
        if bool(args.seen_target_route) and log_entry is not None:
            seen_route_color = str(log_entry.get("target_color") or "")
            if (
                seen_route_color
                and log_entry.get("seen")
                and log_entry.get("area") is not None
                and log_entry.get("bearing") is not None
            ):
                seen_route_dist = float(
                    np.clip(
                        math.exp(
                            seen_target_dist_calib[0]
                            + seen_target_dist_calib[1] * float(log_entry["area"])
                        ),
                        0.5,
                        9.0,
                    )
                )
                seen_route_angle = float(yaw) + float(log_entry["bearing"])
                seen_route_new = (
                    float(pos[0]) + seen_route_dist * math.cos(seen_route_angle),
                    float(pos[1]) + seen_route_dist * math.sin(seen_route_angle),
                )
                seen_route_prev = seen_target_estimates.get(seen_route_color)
                if (
                    seen_route_prev is not None
                    and int(tick) - int(seen_route_prev[2])
                    <= int(args.seen_target_route_max_age_ticks)
                ):
                    seen_route_new = (
                        0.7 * float(seen_route_prev[0]) + 0.3 * seen_route_new[0],
                        0.7 * float(seen_route_prev[1]) + 0.3 * seen_route_new[1],
                    )
                seen_target_estimates[seen_route_color] = (
                    seen_route_new[0],
                    seen_route_new[1],
                    int(tick),
                )
            seen_route_estimate = seen_target_estimates.get(seen_route_color)
            if (
                seen_route_estimate is None
                or int(tick) - int(seen_route_estimate[2])
                > int(args.seen_target_route_max_age_ticks)
            ) and bool(args.novelty_route) and learned_local_online_map is not None:
                # No live target estimate: draw the goal from the online
                # memory instead — nearest unexplored cell, held for a
                # commitment window so route-following can make progress.
                novelty_goal_reached = bool(
                    novelty_route_goal is not None
                    and learned_local_online_map._cell(pos[:2])
                    == learned_local_online_map._cell(novelty_route_goal)
                )
                if (
                    novelty_goal_reached
                    and int(args.novelty_route_scan_ticks) > 0
                    and not escape_plan
                ):
                    # Arrived somewhere new: look around before routing on.
                    escape_plan = ["yaw_left"] * int(args.novelty_route_scan_ticks)
                    wall_metrics["novelty_route_scans"] += 1
                novelty_goal_stale = bool(
                    novelty_route_goal is None
                    or int(tick) >= int(novelty_route_goal_expiry)
                    or novelty_goal_reached
                )
                if novelty_goal_stale:
                    map_cell_m = float(learned_local_online_map.cell_m)
                    cur_cell = learned_local_online_map._cell(pos[:2])
                    found_goal = None
                    vision_free = getattr(learned_local_online_map, "vision_free", None)
                    if vision_free:
                        # Vision frontier: a seen-free cell bordering unknown
                        # space, preferring far ones (long committed traversals
                        # through known-free geometry).
                        known = vision_free | set(learned_local_online_map.visited)
                        blocked_all = set(learned_local_online_map.blocked) | set(
                            getattr(learned_local_online_map, "vision_blocked", set())
                        )
                        best_frontier = None
                        best_score = None
                        for cell in vision_free:
                            if cell in blocked_all or cell in novelty_route_visited_goals:
                                continue
                            cx_f, cy_f = cell
                            frontier = any(
                                (cx_f + dx, cy_f + dy) not in known
                                and (cx_f + dx, cy_f + dy) not in blocked_all
                                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                            )
                            if not frontier:
                                continue
                            dist = abs(cx_f - cur_cell[0]) + abs(cy_f - cur_cell[1])
                            if dist < 2:
                                continue
                            if best_score is None or dist > best_score:
                                best_score, best_frontier = dist, cell
                        found_goal = best_frontier
                        if found_goal is not None:
                            novelty_route_visited_goals.add(found_goal)
                    if found_goal is None:
                        for radius in range(2, 14):
                            ring = []
                            for k in range(8):
                                ang = (
                                    float(novelty_route_direction) + k * (2 * math.pi / 8)
                                )
                                gx = cur_cell[0] + int(round(radius * math.cos(ang)))
                                gy = cur_cell[1] + int(round(radius * math.sin(ang)))
                                cell = (gx, gy)
                                if (
                                    cell not in learned_local_online_map.visited
                                    and cell not in learned_local_online_map.blocked
                                    and cell not in learned_local_online_map.guard_blocked
                                ):
                                    ring.append(cell)
                            if ring:
                                found_goal = ring[0]
                                break
                    if found_goal is not None:
                        novelty_route_goal = learned_local_online_map.cell_center_xy(
                            found_goal
                        )
                        novelty_route_goal_expiry = int(tick) + int(
                            args.novelty_route_commit_ticks
                        )
                        novelty_route_direction = float(
                            novelty_route_direction
                        ) + 2 * math.pi / 8
                        wall_metrics["novelty_route_goals"] += 1
                if novelty_route_goal is not None:
                    seen_route_estimate = (
                        float(novelty_route_goal[0]),
                        float(novelty_route_goal[1]),
                        int(tick),
                    )
            seen_route_near_target = bool(
                log_entry.get("seen")
                and log_entry.get("area") is not None
                and float(log_entry["area"]) >= float(args.seen_target_route_handoff_area_logit)
            )
            if (
                seen_route_estimate is not None
                and int(tick) - int(seen_route_estimate[2])
                <= int(args.seen_target_route_max_age_ticks)
                and guard_state.upper() in seen_target_route_states
                and not seen_route_near_target
                and not escape_plan
                and learned_local_online_map is not None
            ):
                seen_route_path = learned_local_online_map.path_to_goal_biased_frontier(
                    pos[:2],
                    (float(seen_route_estimate[0]), float(seen_route_estimate[1])),
                    goal_weight=float(args.seen_target_route_goal_weight),
                )
                if len(seen_route_path) < 2:
                    # Strict routing is sealed inside the visited island (early
                    # blocked-edge evidence); plan optimistically through
                    # unknown cells and let execution evidence correct.
                    seen_route_path = learned_local_online_map.path_to_goal_biased_frontier(
                        pos[:2],
                        (float(seen_route_estimate[0]), float(seen_route_estimate[1])),
                        goal_weight=float(args.seen_target_route_goal_weight),
                        optimistic=True,
                    )
                if len(seen_route_path) >= 2:
                    seen_route_wp = None
                    for candidate_cell in seen_route_path[1:]:
                        wx, wy = learned_local_online_map.cell_center_xy(candidate_cell)
                        if (wx - float(pos[0])) ** 2 + (wy - float(pos[1])) ** 2 > (
                            0.6 * float(learned_local_online_map.cell_m)
                        ) ** 2:
                            seen_route_wp = (wx, wy)
                            break
                    if seen_route_wp is not None:
                        seen_route_bearing = wrap_angle_pi(
                            math.atan2(
                                seen_route_wp[1] - float(pos[1]),
                                seen_route_wp[0] - float(pos[0]),
                            )
                            - float(yaw)
                        )
                        if abs(seen_route_bearing) > 0.5:
                            seen_route_primitive = (
                                "yaw_left" if seen_route_bearing > 0 else "yaw_right"
                            )
                        elif abs(seen_route_bearing) > 0.25:
                            seen_route_primitive = (
                                "arc_left" if seen_route_bearing > 0 else "arc_right"
                            )
                        else:
                            seen_route_primitive = "forward_medium"
                        if seen_route_primitive != primitive:
                            wall_metrics["seen_target_route_overrides"] += 1
                        primitive = seen_route_primitive
                        requested_primitive = seen_route_primitive
                        wall_metrics["seen_target_route_ticks"] += 1
                        log_entry["seen_target_route"] = {
                            "goal_est": [
                                _round_float(seen_route_estimate[0]),
                                _round_float(seen_route_estimate[1]),
                            ],
                            "est_age": int(tick) - int(seen_route_estimate[2]),
                            "path_cells": len(seen_route_path),
                            "primitive": seen_route_primitive,
                        }
        guard_state_upper = guard_state.upper()
        guard_enabled = bool(
            args.wall_aware_planner
            and (
                not wall_guard_states
                or guard_state_upper in wall_guard_states
                or (
                    len(beacon_claims) >= int(args.wall_guard_post_claim_min_claims)
                    and guard_state_upper in wall_guard_post_claim_states
                )
            )
        )
        stride_primitive = str(args.learned_safe_stride_primitive).strip()
        stride_feature_active = bool(
            stride_primitive
            and use_learned_action_source
            and primitive_outcomes
            and guard_enabled
            and not escape_plan
            and requested_primitive == str(args.learned_safe_stride_from)
            and (not learned_safe_stride_states or guard_state.upper() in learned_safe_stride_states)
            and not bool(body_clearance_request)
        )
        if stride_feature_active:
            pred_alias = _prediction_alias_for_primitive(stride_primitive, primitive_outcomes)
            pred = primitive_outcomes.get(pred_alias) if pred_alias is not None else None
            blocked_prob = None if pred is None else pred.get("blocked_prob")
            progress_m = None if pred is None else pred.get("progress_m")
            clearance_prob = None if pred is None else pred.get("clearance_blocked_prob")
            clearance_limit = (
                float(args.learned_safe_stride_max_blocked_prob)
                if args.learned_safe_stride_max_clearance_blocked_prob is None
                else float(args.learned_safe_stride_max_clearance_blocked_prob)
            )
            bearing_ok = bool(
                bearing_for_guard is None
                or abs(float(bearing_for_guard)) <= float(args.learned_safe_stride_max_bearing)
            )
            stride_safe = bool(
                pred is not None
                and blocked_prob is not None
                and float(blocked_prob) <= float(args.learned_safe_stride_max_blocked_prob)
                and (
                    clearance_prob is None
                    or float(clearance_prob) <= float(clearance_limit)
                )
                and progress_m is not None
                and float(progress_m) >= float(args.learned_safe_stride_min_progress_m)
                and bearing_ok
            )
            stride_log = {
                "primitive": stride_primitive,
                "prediction_alias": pred_alias,
                "blocked_prob": _round_float(blocked_prob, 4),
                "clearance_blocked_prob": _round_float(clearance_prob, 4),
                "progress_m": _round_float(progress_m, 4),
                "bearing": _round_float(bearing_for_guard, 4),
                "safe": bool(stride_safe),
            }
            if stride_safe:
                primitive = stride_primitive
                requested_primitive = stride_primitive
                wall_metrics["learned_safe_stride_upgrades"] += 1
            else:
                wall_metrics["learned_safe_stride_skips"] += 1
            if log_entry is not None:
                log_entry["learned_safe_stride"] = stride_log
        if log_entry is not None and bool(log_entry.get("body_clearance_request")):
            allow_clearance_rerank = bool(
                use_learned_action_source
                and args.primitive_clearance_checkpoint is not None
                and float(args.body_clearance_near_yaw_prob_weight) > 0.0
            )
            allow_low_progress_veto = bool(
                requested_primitive in _FORWARD_PRIMITIVES
                and use_learned_action_source
                and args.primitive_outcome_forward_progress_floor is not None
                and args.primitive_outcome_progress_floor_prefer_yaw
            )
            if allow_clearance_rerank or allow_low_progress_veto:
                guard_enabled = bool(args.wall_aware_planner)
                log_entry["body_clearance_guard_low_progress_veto"] = bool(allow_low_progress_veto)
                log_entry["body_clearance_guard_clearance_rerank"] = bool(allow_clearance_rerank)
                log_entry["body_clearance_guard_enabled_override"] = True
            else:
                # The near-target body-clearance controller intentionally asks for
                # yaw-in-place or a slow forward step. Do not let learned ranking
                # upgrade that conservative request back into a flank-sweeping arc or
                # faster forward primitive.
                guard_enabled = False
                log_entry["body_clearance_guard_locked"] = True
        active_stall_penalties = {
            name: float(args.wall_stall_penalty_score)
            for name, ticks_left in stall_penalties.items()
            if int(ticks_left) > 0
        }
        escape_force = bool(guard_enabled and escape_plan)
        frontier_guard_force = bool(
            guard_enabled
            and not escape_force
            and frontier_pressure_committed
            and use_learned_action_source
            and primitive in _TRANSLATING_PRIMITIVES
        )
        route_replay_guard_log = None
        route_replay_guard_force = False
        if (
            guard_enabled
            and not escape_force
            and not frontier_guard_force
            and bool(args.learned_local_online_map_route_replay_guard_override)
            and use_learned_action_source
            and learned_local_online_map is not None
            and primitive in _TRANSLATING_PRIMITIVES
            and guard_state.upper() == "EXPLORE"
        ):
            route_replay_guard_log = learned_local_online_map.route_replay_guard_evidence(
                pos[:2],
                float(yaw),
                primitive,
            )
            route_replay_guard_force = bool(route_replay_guard_log.get("allow"))
            if route_replay_guard_force:
                wall_metrics[
                    "learned_local_online_map_route_replay_guard_override_ticks"
                ] += 1
                if log_entry is not None:
                    log_entry["online_map_route_replay_guard_override"] = route_replay_guard_log
        force_escape = bool(escape_force or frontier_guard_force or route_replay_guard_force)
        frontier_guard_rerank_on_commit = bool(
            frontier_guard_force
            and (
                args.learned_local_policy_frontier_pressure_guard_rerank_on_commit
                or args.learned_local_policy_frontier_pressure_guard_recovery_rerank_on_commit
            )
        )
        force_single_candidate = bool(
            escape_force
            or route_replay_guard_force
            or (
                frontier_guard_force
                and not frontier_guard_rerank_on_commit
            )
        )
        candidate_names_override = (
            learned_local_policy_frontier_pressure_guard_recovery_primitives
            if (
                frontier_guard_force
                and bool(args.learned_local_policy_frontier_pressure_guard_recovery_rerank_on_commit)
            )
            else None
        )
        if escape_force:
            primitive = escape_plan.pop(0)
            requested_primitive = primitive
            if log_entry is not None:
                log_entry["escape_override"] = True
            wall_metrics["escape_blocks_executed"] += 1
        elif frontier_guard_force and log_entry is not None:
            log_entry["frontier_pressure_guard_commit"] = True
            wall_metrics["learned_local_policy_frontier_pressure_guard_commits"] += 1
        current_body_clearance_for_guard = None
        current_contact_escape_low_clearance = False
        current_contact_escape_suppressed_reason: str | None = None
        active_current_contact_escape_m: float | None = None
        current_contact_escape_projected_clearances: dict[str, float] | None = None
        if use_learned_wall_source:
            primitive, wall_guard = _learned_front_guard_select(
                requested=requested_primitive,
                bearing=bearing_for_guard,
                enabled=guard_enabled,
                front_blocked_prob=front_blocked_prob,
                threshold=float(front_threshold if front_threshold is not None else 0.5),
                force_escape=force_single_candidate,
            )
        elif use_learned_action_source:
            if primitive_outcome_preserve_turn_states:
                preserve_turn_request = guard_state.upper() in primitive_outcome_preserve_turn_states
            else:
                preserve_turn_request = bool(args.primitive_outcome_preserve_turn_requests)
            active_forward_progress_floor = (
                None
                if (
                    args.primitive_outcome_forward_progress_floor is None
                    or (
                        primitive_outcome_forward_progress_floor_states
                        and guard_state.upper()
                        not in primitive_outcome_forward_progress_floor_states
                    )
                )
                else float(args.primitive_outcome_forward_progress_floor)
            )
            if (
                preserve_turn_request
                and bool(args.primitive_outcome_preserve_turn_until_first_claim)
                and bool(beacon_claims)
            ):
                preserve_turn_request = False
                wall_metrics["primitive_outcome_preserve_turn_post_claim_suppressed_ticks"] += 1
            blocked_hard_veto_active = bool(args.primitive_outcome_blocked_hard_veto)
            if (
                blocked_hard_veto_active
                and bool(args.primitive_outcome_blocked_hard_veto_after_first_claim)
                and not bool(beacon_claims)
            ):
                blocked_hard_veto_active = False
                wall_metrics[
                    "primitive_outcome_blocked_hard_veto_pre_claim_suppressed_ticks"
                ] += 1
            preserve_straight_request = bool(primitive_outcome_preserve_straight_states) and (
                guard_state.upper() in primitive_outcome_preserve_straight_states
            )
            current_body_clearance_for_guard = None
            current_contact_escape_suppressed_reason: str | None = None
            active_current_contact_escape_m: float | None = None
            current_contact_escape_area_gate_active = False
            current_contact_escape_target_area = (
                float(log_entry["area"])
                if log_entry is not None and "area" in log_entry
                else None
            )
            current_contact_escape_gate_m = (
                None
                if args.body_clearance_current_contact_escape_m is None
                else float(args.body_clearance_current_contact_escape_m)
            )
            if body_clearance_current_contact_escape_m_by_primitive:
                primitive_gate_m = max(
                    float(value)
                    for value in body_clearance_current_contact_escape_m_by_primitive.values()
                )
                current_contact_escape_gate_m = (
                    primitive_gate_m
                    if current_contact_escape_gate_m is None
                    else max(float(current_contact_escape_gate_m), primitive_gate_m)
                )
            if current_contact_escape_gate_m is not None:
                min_escape_streak = max(
                    1,
                    int(args.body_clearance_current_contact_escape_min_streak),
                )
                min_escape_claims = max(
                    0,
                    int(args.body_clearance_current_contact_escape_min_claimed_count),
                )
                escape_cooldown_ticks = max(
                    0,
                    int(args.body_clearance_current_contact_escape_cooldown_ticks),
                )
                if len(beacon_claims) < min_escape_claims:
                    current_contact_escape_suppressed_reason = "claimed_count"
                    wall_metrics[
                        "body_clearance_current_contact_escape_claimed_count_suppressed_ticks"
                    ] += 1
                elif (
                    body_clearance_current_contact_escape_states
                    and guard_state.upper()
                    not in body_clearance_current_contact_escape_states
                ):
                    current_contact_escape_suppressed_reason = "state"
                    wall_metrics[
                        "body_clearance_current_contact_escape_state_suppressed_ticks"
                    ] += 1
                elif (
                    body_clearance_current_contact_escape_target_colors
                    and str(active_target_color).lower()
                    not in body_clearance_current_contact_escape_target_colors
                ):
                    current_contact_escape_suppressed_reason = "target"
                    wall_metrics[
                        "body_clearance_current_contact_escape_target_suppressed_ticks"
                    ] += 1
                elif (
                    escape_cooldown_ticks > 0
                    and body_clearance_current_contact_escape_last_tick is not None
                    and int(tick) - int(body_clearance_current_contact_escape_last_tick)
                    < escape_cooldown_ticks
                ):
                    current_contact_escape_suppressed_reason = "cooldown"
                    wall_metrics[
                        "body_clearance_current_contact_escape_cooldown_suppressed_ticks"
                    ] += 1
                elif (
                    args.body_clearance_current_contact_escape_min_area_logit is not None
                    and (
                        not body_clearance_current_contact_escape_min_area_states
                        or guard_state.upper()
                        in body_clearance_current_contact_escape_min_area_states
                    )
                    and (
                        current_contact_escape_target_area is None
                        or float(current_contact_escape_target_area)
                        < float(args.body_clearance_current_contact_escape_min_area_logit)
                    )
                ):
                    current_contact_escape_area_gate_active = True
                    current_contact_escape_suppressed_reason = "area"
                    wall_metrics[
                        "body_clearance_current_contact_escape_area_suppressed_ticks"
                    ] += 1
                else:
                    current_contact_escape_area_gate_active = bool(
                        args.body_clearance_current_contact_escape_min_area_logit
                        is not None
                        and (
                            not body_clearance_current_contact_escape_min_area_states
                            or guard_state.upper()
                            in body_clearance_current_contact_escape_min_area_states
                        )
                    )
                    if bool(args.body_clearance_target_servo):
                        current_body_clearance_for_guard = _body_probe_clearance(
                            grid,
                            pos[:2],
                            yaw,
                            body_forward_m=float(args.wall_body_forward_m),
                            body_half_width_m=float(args.wall_body_half_width_m),
                            body_probe_margin_m=float(args.wall_body_probe_margin_m),
                        )
                    current_contact_escape_low_clearance = bool(
                        current_body_clearance_for_guard is not None
                        and float(current_body_clearance_for_guard)
                        <= float(current_contact_escape_gate_m)
                    )
                    if current_contact_escape_low_clearance:
                        body_clearance_current_contact_escape_streak += 1
                        wall_metrics[
                            "body_clearance_current_contact_escape_low_clearance_ticks"
                        ] += 1
                    else:
                        body_clearance_current_contact_escape_streak = 0
                    if not current_contact_escape_low_clearance:
                        current_contact_escape_suppressed_reason = "clearance"
                    elif body_clearance_current_contact_escape_streak < min_escape_streak:
                        current_contact_escape_suppressed_reason = "streak"
                        wall_metrics[
                            "body_clearance_current_contact_escape_streak_suppressed_ticks"
                        ] += 1
                    else:
                        active_current_contact_escape_m = float(
                            current_contact_escape_gate_m
                        )
                if (
                    current_contact_escape_low_clearance
                    and active_current_contact_escape_m is None
                ):
                    wall_metrics[
                        "body_clearance_current_contact_escape_gate_suppressed_ticks"
                    ] += 1
                if current_contact_escape_suppressed_reason in {
                    "claimed_count",
                    "state",
                    "target",
                    "cooldown",
                    "area",
                }:
                    body_clearance_current_contact_escape_streak = 0
            current_contact_escape_projected_clearances: dict[str, float] | None = None
            current_contact_escape_projection_active = bool(
                active_current_contact_escape_m is not None
                and (
                    args.body_clearance_current_contact_escape_min_projected_clearance_m
                    is not None
                    or float(
                        args.body_clearance_current_contact_escape_min_projected_improvement_m
                    )
                    > 0.0
                )
            )
            if current_contact_escape_projection_active:
                current_contact_escape_projected_clearances = {}
                for projection_primitive in outcome_primitive_vocab:
                    try:
                        projection_report = _primitive_clearance_report(
                            registry,
                            str(projection_primitive),
                            pos[:2],
                            yaw,
                            grid,
                            float(args.command_dt_s),
                            body_forward_m=float(args.wall_body_forward_m),
                            body_half_width_m=float(args.wall_body_half_width_m),
                            body_probe_margin_m=float(args.wall_body_probe_margin_m),
                            min_clearance_m=float(args.wall_min_clearance_m),
                        )
                    except Exception:
                        continue
                    current_contact_escape_projected_clearances[
                        str(projection_primitive)
                    ] = float(projection_report["min_clearance_m"])
            current_body_clearance_for_select = (
                current_body_clearance_for_guard
                if active_current_contact_escape_m is not None
                else None
            )
            body_clearance_veto_claim_gate_active = (
                len(beacon_claims) >= int(effective_body_clearance_veto_min_claimed_count)
            )
            effective_body_clearance_hard_veto_prob = (
                float(args.body_clearance_hard_veto_prob)
                if body_clearance_veto_claim_gate_active
                else 1.01
            )
            target_area_for_guard = (
                float(log_entry["area"])
                if log_entry is not None and "area" in log_entry
                else None
            )
            target_area_hard_veto_min_area = (
                float(args.body_clearance_target_area_logit)
                if args.body_clearance_target_area_hard_veto_min_area_logit is None
                else float(args.body_clearance_target_area_hard_veto_min_area_logit)
            )
            target_area_hard_veto_active = bool(
                body_clearance_veto_claim_gate_active
                and float(args.body_clearance_target_area_hard_veto_prob) <= 1.0
                and target_area_for_guard is not None
                and float(target_area_for_guard) >= float(target_area_hard_veto_min_area)
            )
            if target_area_hard_veto_active:
                effective_body_clearance_hard_veto_prob = min(
                    float(effective_body_clearance_hard_veto_prob),
                    float(args.body_clearance_target_area_hard_veto_prob),
                )
                wall_metrics["body_clearance_target_area_hard_veto_ticks"] += 1
            effective_body_clearance_saturated_veto_prob = (
                float(args.body_clearance_saturated_veto_prob)
                if body_clearance_veto_claim_gate_active
                else 1.01
            )
            effective_body_clearance_yaw_contact_veto_prob = (
                float(args.body_clearance_yaw_contact_veto_prob)
                if body_clearance_veto_claim_gate_active
                else 1.01
            )
            effective_body_clearance_yaw_direction_veto_prob = (
                float(args.body_clearance_yaw_direction_veto_prob)
                if body_clearance_veto_claim_gate_active
                else 1.01
            )
            active_guard_primitive_outcomes = (
                primitive_guard_outcomes
                if primitive_guard_outcomes is not None
                else primitive_outcomes
            )
            active_body_clearance_hard_veto_primitives = body_clearance_hard_veto_primitives
            if (
                primitive_aux_clearance_switch_active
                and body_clearance_aux_switch_hard_veto_primitives
            ):
                active_body_clearance_hard_veto_primitives = (
                    body_clearance_aux_switch_hard_veto_primitives
                )
            effective_body_clearance_enabled = bool(args.body_clearance_target_servo) or (
                bool(args.body_clearance_aux_switch_enable)
                and bool(primitive_aux_clearance_switch_active)
            )
            effective_body_clearance_min_area = (
                None
                if args.body_clearance_learned_min_area_logit is None
                else float(args.body_clearance_learned_min_area_logit)
            )
            if (
                bool(args.body_clearance_aux_switch_enable)
                and bool(primitive_aux_clearance_switch_active)
            ):
                wall_metrics["body_clearance_aux_switch_enabled_ticks"] += 1
                if (
                    bool(args.body_clearance_aux_switch_ignore_min_area)
                    and effective_body_clearance_min_area is not None
                ):
                    effective_body_clearance_min_area = None
                    wall_metrics["body_clearance_aux_switch_min_area_ignored_ticks"] += 1
            effective_body_clearance_arc_sweep_veto_prob = (
                float(args.body_clearance_aux_switch_arc_sweep_veto_prob)
                if (
                    bool(args.body_clearance_aux_switch_enable)
                    and bool(primitive_aux_clearance_switch_active)
                )
                else 1.01
            )
            primitive, wall_guard = _learned_action_guard_select(
                requested=requested_primitive,
                bearing=bearing_for_guard,
                enabled=guard_enabled,
                predictions=active_guard_primitive_outcomes,
                primitive_vocab=outcome_primitive_vocab,
                blocked_threshold=float(outcome_threshold if outcome_threshold is not None else 0.5),
                blocked_weight=float(args.primitive_outcome_blocked_weight),
                progress_weight=float(args.primitive_outcome_progress_weight),
                requested_bonus=float(args.primitive_outcome_requested_bonus),
                turn_progress_scale=float(args.primitive_outcome_turn_progress_scale),
                switch_margin=float(args.primitive_outcome_switch_margin),
                target_area=target_area_for_guard,
                body_clearance_enabled=bool(effective_body_clearance_enabled),
                body_clearance_target_area=float(args.body_clearance_target_area_logit),
                body_clearance_arc_penalty=float(args.body_clearance_near_arc_penalty),
                body_clearance_yaw_penalty_weight=float(args.body_clearance_near_yaw_prob_weight),
                body_clearance_prob_floor=float(args.body_clearance_learned_prob_floor),
                body_clearance_prob_weight=float(args.body_clearance_learned_prob_weight),
                body_clearance_near_forward_prob_floor=(
                    None
                    if args.body_clearance_near_forward_prob_floor is None
                    else float(args.body_clearance_near_forward_prob_floor)
                ),
                body_clearance_near_forward_prob_weight=(
                    None
                    if args.body_clearance_near_forward_prob_weight is None
                    else float(args.body_clearance_near_forward_prob_weight)
                ),
                body_clearance_near_yaw_prob_floor=(
                    None
                    if args.body_clearance_near_yaw_prob_floor is None
                    else float(args.body_clearance_near_yaw_prob_floor)
                ),
                body_clearance_yaw_always=bool(args.body_clearance_yaw_always),
                body_clearance_hard_veto_prob=effective_body_clearance_hard_veto_prob,
                body_clearance_hard_veto_margin=float(args.body_clearance_hard_veto_margin),
                body_clearance_hard_veto_replacement_cap=float(
                    args.body_clearance_hard_veto_replacement_cap
                ),
                body_clearance_hard_veto_primitives=(
                    active_body_clearance_hard_veto_primitives
                ),
                body_clearance_hard_veto_selected_primitives=(
                    body_clearance_hard_veto_selected_primitives or None
                ),
                body_clearance_arc_sweep_veto_prob=(
                    effective_body_clearance_arc_sweep_veto_prob
                ),
                body_clearance_arc_sweep_veto_selected_primitives=(
                    body_clearance_aux_switch_arc_sweep_veto_selected_primitives or None
                ),
                body_clearance_saturated_veto_prob=effective_body_clearance_saturated_veto_prob,
                body_clearance_saturated_veto_spread=float(args.body_clearance_saturated_veto_spread),
                body_clearance_saturated_veto_primitives=body_clearance_saturated_veto_primitives,
                body_clearance_saturated_veto_selected_primitives=(
                    body_clearance_saturated_veto_selected_primitives or None
                ),
                body_clearance_yaw_contact_veto_prob=effective_body_clearance_yaw_contact_veto_prob,
                body_clearance_yaw_direction_veto_prob=(
                    effective_body_clearance_yaw_direction_veto_prob
                ),
                body_clearance_yaw_direction_veto_margin=float(
                    args.body_clearance_yaw_direction_veto_margin
                ),
                current_body_clearance_m=current_body_clearance_for_select,
                body_clearance_current_contact_escape_m=active_current_contact_escape_m,
                body_clearance_current_contact_escape_m_by_primitive=(
                    body_clearance_current_contact_escape_m_by_primitive or None
                ),
                body_clearance_current_contact_escape_primitives=(
                    body_clearance_current_contact_escape_primitives or None
                ),
                body_clearance_current_contact_escape_replacements=(
                    body_clearance_current_contact_escape_replacements or None
                ),
                body_clearance_current_contact_escape_replacement_cap=float(
                    args.body_clearance_current_contact_escape_replacement_cap
                ),
                body_clearance_current_contact_escape_require_replacement_under_cap=bool(
                    args.body_clearance_current_contact_escape_require_replacement_under_cap
                ),
                body_clearance_current_contact_escape_projected_clearances=(
                    current_contact_escape_projected_clearances
                ),
                body_clearance_current_contact_escape_min_projected_clearance_m=(
                    None
                    if args.body_clearance_current_contact_escape_min_projected_clearance_m
                    is None
                    else float(
                        args.body_clearance_current_contact_escape_min_projected_clearance_m
                    )
                ),
                body_clearance_current_contact_escape_min_projected_improvement_m=float(
                    args.body_clearance_current_contact_escape_min_projected_improvement_m
                ),
                body_clearance_min_area=effective_body_clearance_min_area,
                forward_progress_floor=active_forward_progress_floor,
                forward_progress_floor_min_blocked_prob=(
                    None
                    if args.primitive_outcome_progress_floor_min_blocked_prob is None
                    else float(args.primitive_outcome_progress_floor_min_blocked_prob)
                ),
                forward_progress_floor_force_below=(
                    None
                    if args.primitive_outcome_progress_floor_force_below is None
                    else float(args.primitive_outcome_progress_floor_force_below)
                ),
                forward_progress_floor_penalty=float(args.primitive_outcome_forward_progress_penalty),
                low_progress_hard_veto=bool(args.primitive_outcome_low_progress_hard_veto),
                low_progress_hard_veto_primitives=primitive_outcome_low_progress_hard_veto_primitives,
                blocked_hard_veto=bool(blocked_hard_veto_active),
                blocked_hard_veto_primitives=primitive_outcome_blocked_hard_veto_primitives,
                blocked_hard_veto_selected_primitives=(
                    primitive_outcome_blocked_hard_veto_selected_primitives or None
                ),
                blocked_hard_veto_max_abs_bearing=(
                    None
                    if args.primitive_outcome_blocked_hard_veto_max_abs_bearing is None
                    else float(args.primitive_outcome_blocked_hard_veto_max_abs_bearing)
                ),
                blocked_hard_veto_bearing=(
                    None
                    if args.primitive_outcome_blocked_hard_veto_use_guard_bearing
                    else (
                        None
                        if log_entry is None or "bearing" not in log_entry
                        else float(log_entry["bearing"])
                    )
                ),
                progress_floor_prefer_yaw=bool(args.primitive_outcome_progress_floor_prefer_yaw),
                runtime_penalties=active_stall_penalties,
                preserve_turn_requests=preserve_turn_request,
                preserve_arc_requests=bool(args.primitive_outcome_preserve_arc_requests),
                turn_body_rerank_primitives=primitive_outcome_turn_body_rerank_primitives,
                preserve_straight_requests=preserve_straight_request,
                preserve_backward_requests=bool(args.primitive_outcome_preserve_backward_requests),
                preserve_backward_clearance_margin=(
                    None
                    if args.primitive_outcome_preserve_backward_clearance_margin is None
                    else float(args.primitive_outcome_preserve_backward_clearance_margin)
                ),
                force_escape=force_escape,
                force_single_candidate=force_single_candidate,
                candidate_names_override=candidate_names_override,
            )
            wall_guard["body_clearance_veto_claim_gate_active"] = bool(
                body_clearance_veto_claim_gate_active
            )
            wall_guard["body_clearance_veto_configured_min_claimed_count"] = int(
                configured_body_clearance_veto_min_claimed_count
            )
            wall_guard["body_clearance_veto_min_claimed_count"] = int(
                effective_body_clearance_veto_min_claimed_count
            )
            wall_guard["body_clearance_veto_claimed_count"] = int(len(beacon_claims))
            wall_guard["body_clearance_current_contact_escape_low_clearance"] = bool(
                current_contact_escape_low_clearance
            )
            wall_guard["body_clearance_current_contact_escape_streak"] = int(
                body_clearance_current_contact_escape_streak
            )
            wall_guard["body_clearance_current_contact_escape_gate_active"] = bool(
                active_current_contact_escape_m is not None
            )
            wall_guard["body_clearance_current_contact_escape_observed_clearance_m"] = (
                _round_float(current_body_clearance_for_guard, 4)
            )
            wall_guard["body_clearance_current_contact_escape_min_area_logit"] = (
                None
                if args.body_clearance_current_contact_escape_min_area_logit is None
                else _round_float(
                    float(args.body_clearance_current_contact_escape_min_area_logit),
                    4,
                )
            )
            wall_guard["body_clearance_current_contact_escape_area_gate_active"] = bool(
                current_contact_escape_area_gate_active
            )
            wall_guard["body_clearance_current_contact_escape_area_logit"] = _round_float(
                current_contact_escape_target_area,
                4,
            )
            wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                current_contact_escape_suppressed_reason
            )
            if not body_clearance_veto_claim_gate_active:
                wall_metrics["body_clearance_veto_claim_gate_suppressed_ticks"] += 1
                selected_clearance_for_gate = wall_guard.get("selected_clearance_blocked_prob")
                high_risk_thresholds = [
                    float(value)
                    for value in (
                        args.body_clearance_hard_veto_prob,
                        args.body_clearance_saturated_veto_prob,
                        args.body_clearance_yaw_contact_veto_prob,
                        args.body_clearance_yaw_direction_veto_prob,
                    )
                    if float(value) <= 1.0
                ]
                if (
                    selected_clearance_for_gate is not None
                    and high_risk_thresholds
                    and float(selected_clearance_for_gate) >= min(high_risk_thresholds)
                ):
                    wall_metrics[
                        "body_clearance_veto_claim_gate_suppressed_high_risk_ticks"
                    ] += 1
            wall_guard["primitive_clearance_slot"] = str(primitive_clearance_slot)
            wall_guard["primitive_policy_clearance_slot"] = str(
                primitive_policy_clearance_slot
            )
            wall_guard["primitive_aux_clearance_switch_active"] = bool(
                primitive_aux_clearance_switch_active
            )
            wall_guard["primitive_aux_clearance_switch_configured_min_claimed_count"] = int(
                configured_aux_clearance_switch_min_claimed_count
            )
            wall_guard["primitive_aux_clearance_switch_min_claimed_count"] = int(
                effective_aux_clearance_switch_min_claimed_count
            )
            wall_guard["body_clearance_target_area_hard_veto_active"] = bool(
                target_area_hard_veto_active
            )
            wall_guard["body_clearance_target_area_hard_veto_prob"] = _round_float(
                float(args.body_clearance_target_area_hard_veto_prob), 4
            )
            wall_guard["body_clearance_target_area_hard_veto_min_area_logit"] = _round_float(
                float(target_area_hard_veto_min_area), 4
            )
            wall_guard["primitive_aux_clearance_switch_policy_features"] = bool(
                args.primitive_aux_clearance_switch_policy_features
            )
            wall_guard["primitive_aux_clearance_switch_area_active"] = bool(
                current_body_risk_area_active_for_switch
            )
            wall_guard["body_clearance_hard_veto_active_primitives"] = sorted(
                active_body_clearance_hard_veto_primitives
            )
            wall_guard["primitive_aux_clearance_switch_prob"] = _round_float(
                primitive_aux_clearance_switch_prob,
                4,
            )
            wall_guard["primitive_aux_clearance_switch_latch_remaining"] = int(
                primitive_aux_clearance_switch_latch
            )
            aux_candidate_primitives = [
                str(item.get("primitive"))
                for item in wall_guard.get("candidates", [])
                if isinstance(item, dict) and item.get("primitive") is not None
            ]
            aux_selected, aux_log = _select_aux_clearance_veto(
                selected=primitive,
                primary_predictions=primitive_outcomes,
                aux_clearance_predictions=primitive_aux_clearance_outcomes,
                candidate_primitives=aux_candidate_primitives,
                enabled=bool(
                    body_clearance_veto_claim_gate_active
                    and args.body_clearance_target_servo
                    and args.primitive_aux_clearance_checkpoint is not None
                    and primitive_aux_clearance_outcomes is not None
                    and not force_single_candidate
                ),
                aux_veto_prob=float(args.body_clearance_aux_veto_prob),
                primary_max_prob=float(args.body_clearance_aux_veto_primary_max_prob),
                aux_veto_margin=float(args.body_clearance_aux_veto_margin),
                aux_replacement_cap=float(args.body_clearance_aux_veto_replacement_cap),
                selected_primitives=body_clearance_aux_veto_selected_primitives or None,
                replacement_primitives=body_clearance_aux_veto_primitives or None,
            )
            if bool(aux_log.get("active")):
                previous_primitive = primitive
                primitive = aux_selected
                wall_guard["body_clearance_aux_veto"] = True
                wall_guard["selected_before_body_clearance_aux_veto"] = previous_primitive
                wall_guard["selected"] = primitive
                wall_guard["vetoed"] = bool(primitive != requested_primitive)
                selected_alias = _prediction_alias_for_primitive(
                    primitive,
                    active_guard_primitive_outcomes,
                )
                selected_pred = (
                    active_guard_primitive_outcomes.get(selected_alias)
                    if selected_alias is not None and active_guard_primitive_outcomes is not None
                    else None
                )
                if selected_pred is not None:
                    wall_guard["selected_prediction_alias"] = (
                        selected_alias if selected_alias != primitive else None
                    )
                    wall_guard["selected_outcome_blocked_prob"] = _round_float(
                        selected_pred.get("outcome_blocked_prob"), 4
                    )
                    wall_guard["selected_clearance_blocked_prob"] = _round_float(
                        selected_pred.get("clearance_blocked_prob"), 4
                    )
                    wall_guard["selected_progress_m"] = _round_float(
                        selected_pred.get("progress_m"), 4
                    )
                wall_guard["body_clearance_aux_veto_log"] = aux_log
            elif aux_log.get("suppressed") is not None:
                wall_guard["body_clearance_aux_veto"] = False
                wall_guard["body_clearance_aux_veto_log"] = aux_log
        else:
            primitive, wall_guard = _wall_guard_select(
                requested=requested_primitive,
                pos_xy=pos[:2],
                yaw=yaw,
                bearing=bearing_for_guard,
                registry=registry,
                grid=grid,
                command_dt_s=float(args.command_dt_s),
                enabled=guard_enabled,
                min_clearance_m=float(args.wall_min_clearance_m),
                feasible_threshold=float(args.wall_feasible_threshold),
                body_forward_m=float(args.wall_body_forward_m),
                body_half_width_m=float(args.wall_body_half_width_m),
                body_probe_margin_m=float(args.wall_body_probe_margin_m),
                force_escape=force_escape,
            )
        if (
            bool(learned_topology_route_selected_this_tick)
            and args.learned_topology_route_geometry_veto_min_clearance_m is not None
            and bool(guard_enabled)
            and not force_single_candidate
        ):
            wall_metrics["learned_topology_route_geometry_veto_ticks"] += 1
            geometry_selected, geometry_log = _geometry_clearance_veto_select(
                selected=primitive,
                requested=requested_primitive,
                pos_xy=pos[:2],
                yaw=yaw,
                bearing=bearing_for_guard,
                registry=registry,
                grid=grid,
                command_dt_s=float(args.command_dt_s),
                enabled=True,
                min_clearance_m=float(
                    args.learned_topology_route_geometry_veto_min_clearance_m
                ),
                feasible_threshold=float(
                    args.learned_topology_route_geometry_veto_feasible_threshold
                ),
                body_forward_m=float(args.wall_body_forward_m),
                body_half_width_m=float(args.wall_body_half_width_m),
                body_probe_margin_m=float(args.wall_body_probe_margin_m),
                selected_primitives=(
                    learned_topology_route_geometry_veto_selected_primitives or None
                ),
                replacement_primitives=(
                    learned_topology_route_geometry_veto_replacements or None
                ),
            )
            selected_route_geometry_clearance = geometry_log.get(
                "selected_min_clearance_m"
            )
            if selected_route_geometry_clearance is not None:
                previous_min = wall_metrics[
                    "learned_topology_route_geometry_veto_selected_min_clearance_m"
                ]
                wall_metrics[
                    "learned_topology_route_geometry_veto_selected_min_clearance_m"
                ] = (
                    selected_route_geometry_clearance
                    if previous_min is None
                    else min(float(previous_min), float(selected_route_geometry_clearance))
                )
            wall_guard["learned_topology_route_geometry_veto"] = bool(
                geometry_log.get("active")
            )
            wall_guard["learned_topology_route_geometry_veto_log"] = geometry_log
            wall_guard["selected_min_clearance_m"] = geometry_log.get(
                "replacement_min_clearance_m"
            )
            wall_guard["selected_feasible_fraction"] = geometry_log.get(
                "replacement_feasible_fraction"
            )
            wall_guard["selected_blocked"] = bool(geometry_log.get("replacement_blocked"))
            if bool(geometry_log.get("active")):
                previous_primitive = primitive
                primitive = geometry_selected
                wall_metrics["learned_topology_route_geometry_vetoes"] += 1
                wall_guard["selected_before_learned_topology_route_geometry_veto"] = (
                    previous_primitive
                )
                wall_guard["selected"] = primitive
                wall_guard["vetoed"] = bool(primitive != requested_primitive)
        if frontier_guard_force:
            wall_guard["frontier_pressure_guard_commit"] = True
        if route_replay_guard_log is not None:
            wall_guard["online_map_route_replay_guard_evidence"] = route_replay_guard_log
            wall_guard["online_map_route_replay_guard_override"] = bool(
                route_replay_guard_force
            )
        body_clearance_geometry_veto_state_active = (
            not body_clearance_geometry_veto_states
            or guard_state.upper() in body_clearance_geometry_veto_states
        )
        body_clearance_geometry_veto_claim_active = len(beacon_claims) >= max(
            0,
            int(args.body_clearance_geometry_veto_min_claimed_count),
        )
        body_clearance_geometry_veto_target_active = (
            not body_clearance_geometry_veto_target_colors
            or str(active_target_color).lower()
            in body_clearance_geometry_veto_target_colors
        )
        body_clearance_geometry_veto_basic_active = (
            use_learned_action_source
            and args.body_clearance_geometry_veto_min_clearance_m is not None
            and (
                bool(guard_enabled)
                or (
                    bool(args.body_clearance_geometry_veto_allow_guard_disabled)
                    and bool(args.wall_aware_planner)
                )
            )
            and (
                not force_single_candidate
                or bool(args.body_clearance_geometry_veto_allow_force_single_candidate)
            )
            and body_clearance_geometry_veto_state_active
        )
        if args.body_clearance_geometry_veto_min_clearance_m is not None:
            wall_guard["body_clearance_geometry_veto_claim_gate_active"] = bool(
                body_clearance_geometry_veto_claim_active
            )
            wall_guard["body_clearance_geometry_veto_min_claimed_count"] = max(
                0,
                int(args.body_clearance_geometry_veto_min_claimed_count),
            )
            wall_guard["body_clearance_geometry_veto_target_gate_active"] = bool(
                body_clearance_geometry_veto_target_active
            )
            wall_guard["body_clearance_geometry_veto_target_colors"] = sorted(
                body_clearance_geometry_veto_target_colors
            )
            if (
                body_clearance_geometry_veto_basic_active
                and not body_clearance_geometry_veto_claim_active
            ):
                wall_metrics[
                    "body_clearance_geometry_veto_claimed_count_suppressed_ticks"
                ] += 1
            elif (
                body_clearance_geometry_veto_basic_active
                and not body_clearance_geometry_veto_target_active
            ):
                wall_metrics[
                    "body_clearance_geometry_veto_target_suppressed_ticks"
                ] += 1
        if (
            body_clearance_geometry_veto_basic_active
            and body_clearance_geometry_veto_claim_active
            and body_clearance_geometry_veto_target_active
        ):
            active_geometry_veto_replacements = body_clearance_geometry_veto_replacements
            if (
                body_clearance_geometry_veto_override_replacements
                and len(beacon_claims)
                >= max(
                    0,
                    int(args.body_clearance_geometry_veto_override_min_claimed_count),
                )
            ):
                active_geometry_veto_replacements = (
                    body_clearance_geometry_veto_override_replacements
                )
            wall_metrics["body_clearance_geometry_veto_ticks"] += 1
            geometry_selected, geometry_log = _geometry_clearance_veto_select(
                selected=primitive,
                requested=requested_primitive,
                pos_xy=pos[:2],
                yaw=yaw,
                bearing=bearing_for_guard,
                registry=registry,
                grid=grid,
                command_dt_s=float(args.command_dt_s),
                enabled=True,
                min_clearance_m=float(
                    args.body_clearance_geometry_veto_min_clearance_m
                ),
                feasible_threshold=float(
                    args.body_clearance_geometry_veto_feasible_threshold
                ),
                body_forward_m=float(args.wall_body_forward_m),
                body_half_width_m=float(args.wall_body_half_width_m),
                body_probe_margin_m=float(args.wall_body_probe_margin_m),
                selected_primitives=(
                    body_clearance_geometry_veto_selected_primitives or None
                ),
                replacement_primitives=(
                    active_geometry_veto_replacements or None
                ),
                blocked_fallback_primitives=(
                    body_clearance_geometry_veto_blocked_fallback_primitives or None
                ),
            )
            geometry_log["replacement_override_active"] = bool(
                active_geometry_veto_replacements
                is body_clearance_geometry_veto_override_replacements
            )
            selected_geometry_clearance = geometry_log.get("selected_min_clearance_m")
            if selected_geometry_clearance is not None:
                previous_min = wall_metrics[
                    "body_clearance_geometry_veto_selected_min_clearance_m"
                ]
                wall_metrics["body_clearance_geometry_veto_selected_min_clearance_m"] = (
                    selected_geometry_clearance
                    if previous_min is None
                    else min(float(previous_min), float(selected_geometry_clearance))
                )
            wall_guard["body_clearance_geometry_veto"] = bool(
                geometry_log.get("active")
            )
            wall_guard["body_clearance_geometry_veto_log"] = geometry_log
            wall_guard["selected_min_clearance_m"] = geometry_log.get(
                "replacement_min_clearance_m"
            )
            wall_guard["selected_feasible_fraction"] = geometry_log.get(
                "replacement_feasible_fraction"
            )
            wall_guard["selected_blocked"] = bool(geometry_log.get("replacement_blocked"))
            if bool(geometry_log.get("active")):
                previous_primitive = primitive
                primitive = geometry_selected
                wall_metrics["body_clearance_geometry_vetoes"] += 1
                wall_guard["selected_before_body_clearance_geometry_veto"] = (
                    previous_primitive
                )
                wall_guard["selected"] = primitive
                wall_guard["vetoed"] = bool(primitive != requested_primitive)
        selected_clearance_prob = wall_guard.get("selected_clearance_blocked_prob")
        body_risk_escape = bool(
            use_learned_action_source
            and args.primitive_clearance_checkpoint is not None
            and int(args.body_clearance_risk_escape_blocks) > 0
            and float(args.body_clearance_risk_escape_threshold) <= 1.0
            and int(body_clearance_risk_escape_cooldown) <= 0
            and not force_single_candidate
            and guard_state.upper() in body_clearance_risk_escape_states
            and primitive in _FORWARD_PRIMITIVES.union(_TURN_PRIMITIVES)
            and selected_clearance_prob is not None
            and float(selected_clearance_prob) >= float(args.body_clearance_risk_escape_threshold)
        )
        if body_risk_escape:
            risky_primitive = primitive
            body_escape_plan = _make_escape_plan(risky_primitive, int(args.body_clearance_risk_escape_blocks))
            if body_escape_plan:
                primitive = body_escape_plan.pop(0)
                escape_plan = body_escape_plan + escape_plan
                body_clearance_risk_escape_cooldown = max(
                    0,
                    int(args.body_clearance_risk_escape_cooldown_ticks),
                )
                wall_metrics["body_clearance_risk_escapes"] += 1
                wall_metrics["escape_blocks_executed"] += 1
                wall_guard["body_clearance_risk_escape"] = True
                wall_guard["body_clearance_risk_escape_threshold"] = _round_float(
                    float(args.body_clearance_risk_escape_threshold), 4
                )
                wall_guard["selected_before_body_clearance_escape"] = risky_primitive
                wall_guard["selected_before_body_clearance_escape_prob"] = _round_float(
                    selected_clearance_prob, 4
                )
                wall_guard["selected"] = primitive
                wall_guard["vetoed"] = bool(primitive != requested_primitive)
                wall_guard["force_escape"] = True
                if log_entry is not None:
                    log_entry["body_clearance_risk_escape"] = {
                        "from": risky_primitive,
                        "to": primitive,
                        "clearance_blocked_prob": _round_float(selected_clearance_prob, 4),
                        "remaining_plan": list(escape_plan),
                    }
        if learned_local_online_map is not None and bool(wall_guard.get("enabled", False)):
            wall_guard_map_updates = 0
            guard_block_source = str(args.learned_local_online_map_wall_guard_block_source)
            for candidate in wall_guard.get("candidates", ()):
                candidate_primitive = str(candidate.get("primitive", ""))
                if candidate_primitive not in _TRANSLATING_PRIMITIVES:
                    continue
                if not bool(candidate.get("blocked", False)):
                    continue
                if guard_block_source == "none":
                    continue
                if guard_block_source == "requested" and candidate_primitive != requested_primitive:
                    continue
                if guard_block_source == "selected" and candidate_primitive != primitive:
                    continue
                if (
                    guard_block_source == "requested_selected"
                    and candidate_primitive not in {requested_primitive, primitive}
                ):
                    continue
                if learned_local_online_map.mark_blocked_primitive(
                    pos[:2],
                    float(yaw),
                    candidate_primitive,
                ):
                    wall_guard_map_updates += 1
            if wall_guard_map_updates:
                wall_metrics["learned_local_online_map_wall_guard_blocked_edges"] += int(
                    wall_guard_map_updates
                )
                wall_guard["online_map_blocked_edge_updates"] = int(wall_guard_map_updates)
                wall_guard["online_map_blocked_edge_source"] = guard_block_source
            projection_block_updates = 0
            if bool(args.learned_local_online_map_current_contact_projection_blocks):
                for rejection in wall_guard.get(
                    "body_clearance_current_contact_escape_projected_rejections",
                    (),
                ):
                    if not isinstance(rejection, dict):
                        continue
                    primitive_name = str(rejection.get("primitive", ""))
                    if primitive_name not in _TRANSLATING_PRIMITIVES:
                        continue
                    if not bool(rejection.get("selected", False)):
                        continue
                    if str(rejection.get("reason")) != "projected_clearance":
                        continue
                    if learned_local_online_map.mark_blocked_primitive(
                        pos[:2],
                        float(yaw),
                        primitive_name,
                    ):
                        projection_block_updates += 1
            if projection_block_updates:
                wall_metrics[
                    "learned_local_online_map_current_contact_projection_blocked_edges"
                ] += int(projection_block_updates)
                wall_guard["online_map_projection_blocked_edge_updates"] = int(
                    projection_block_updates
                )
            hold_escape_projection_block_updates = 0
            if bool(
                args.learned_local_online_map_hard_veto_hold_escape_projection_blocks
            ):
                primitive_name = str(
                    wall_guard.get(
                        "body_clearance_hard_veto_hold_escape_projection_block_primitive",
                        "",
                    )
                )
                if primitive_name in _TRANSLATING_PRIMITIVES:
                    if learned_local_online_map.mark_blocked_primitive(
                        pos[:2],
                        float(yaw),
                        primitive_name,
                    ):
                        hold_escape_projection_block_updates += 1
            if hold_escape_projection_block_updates:
                wall_metrics[
                    "learned_local_online_map_hard_veto_hold_escape_projection_blocked_edges"
                ] += int(hold_escape_projection_block_updates)
                wall_guard[
                    "online_map_hold_escape_projection_blocked_edge_updates"
                ] = int(hold_escape_projection_block_updates)
            geometry_hold_block_updates = 0
            if (
                bool(args.learned_local_online_map_geometry_veto_hold_blocks)
                and bool(wall_guard.get("body_clearance_geometry_veto"))
                and str(wall_guard.get("selected", "")) == "hold"
            ):
                primitive_name = str(
                    wall_guard.get("selected_before_body_clearance_geometry_veto", "")
                )
                if primitive_name in _TRANSLATING_PRIMITIVES:
                    if learned_local_online_map.mark_blocked_primitive(
                        pos[:2],
                        float(yaw),
                        primitive_name,
                    ):
                        geometry_hold_block_updates += 1
            if geometry_hold_block_updates:
                wall_metrics[
                    "learned_local_online_map_geometry_veto_hold_blocked_edges"
                ] += int(geometry_hold_block_updates)
                wall_guard[
                    "online_map_geometry_veto_hold_blocked_edge_updates"
                ] = int(geometry_hold_block_updates)
        if current_body_risk_prob is not None:
            wall_metrics["current_body_risk_prob_max"] = (
                _round_float(current_body_risk_prob, 4)
                if wall_metrics["current_body_risk_prob_max"] is None
                else max(
                    float(wall_metrics["current_body_risk_prob_max"]),
                    _round_float(current_body_risk_prob, 4),
                )
            )
            wall_guard["current_body_risk_prob"] = _round_float(current_body_risk_prob, 4)
            wall_guard["current_body_risk_threshold"] = _round_float(current_body_threshold, 4)
            if (
                current_body_threshold is not None
                and float(current_body_risk_prob) >= float(current_body_threshold)
            ):
                wall_metrics["current_body_risk_ticks"] += 1
        current_body_area_active = bool(
            args.current_body_risk_min_area_logit is None
            or (
                log_entry is not None
                and "area" in log_entry
                and float(log_entry["area"]) >= float(args.current_body_risk_min_area_logit)
            )
        )
        current_body_min_claimed_count = max(
            0,
            int(args.current_body_risk_min_claimed_count),
        )
        current_body_claim_gate_active = bool(
            len(beacon_claims) >= int(current_body_min_claimed_count)
        )
        if (
            current_body_risk_prob is not None
            and current_body_threshold is not None
            and float(current_body_threshold) <= 1.0
            and not current_body_claim_gate_active
        ):
            wall_metrics["current_body_risk_claim_gate_suppressed_ticks"] += 1
            if float(current_body_risk_prob) >= float(current_body_threshold):
                wall_metrics[
                    "current_body_risk_claim_gate_suppressed_high_risk_ticks"
                ] += 1
        current_body_preserve_yaw_threshold = (
            current_body_threshold
            if args.current_body_risk_preserve_yaw_threshold is None
            else float(args.current_body_risk_preserve_yaw_threshold)
        )
        current_body_preserve_yaw_min_area = (
            args.current_body_risk_min_area_logit
            if args.current_body_risk_preserve_yaw_min_area_logit is None
            else args.current_body_risk_preserve_yaw_min_area_logit
        )
        current_body_preserve_yaw_area_active = bool(
            current_body_preserve_yaw_min_area is None
            or (
                log_entry is not None
                and "area" in log_entry
                and float(log_entry["area"]) >= float(current_body_preserve_yaw_min_area)
            )
        )
        current_body_clearance_rerank_threshold = (
            current_body_threshold
            if args.current_body_risk_clearance_rerank_threshold is None
            else float(args.current_body_risk_clearance_rerank_threshold)
        )
        current_body_clearance_rerank_min_area = (
            args.current_body_risk_min_area_logit
            if args.current_body_risk_clearance_rerank_min_area_logit is None
            else args.current_body_risk_clearance_rerank_min_area_logit
        )
        current_body_clearance_rerank_selected_floor = (
            None
            if args.current_body_risk_clearance_rerank_selected_prob_floor is None
            else float(args.current_body_risk_clearance_rerank_selected_prob_floor)
        )
        current_body_clearance_rerank_area_active = bool(
            current_body_clearance_rerank_min_area is None
            or (
                log_entry is not None
                and "area" in log_entry
                and float(log_entry["area"]) >= float(current_body_clearance_rerank_min_area)
            )
        )
        current_body_clearance_rerank_selected_prob = wall_guard.get("selected_clearance_blocked_prob")
        current_body_clearance_rerank_selected_active = bool(
            current_body_clearance_rerank_selected_floor is None
            or (
                current_body_clearance_rerank_selected_prob is not None
                and float(current_body_clearance_rerank_selected_prob) >= float(current_body_clearance_rerank_selected_floor)
            )
        )
        current_body_clearance_rerank_selected_primitive_active = bool(
            not current_body_risk_clearance_rerank_selected_primitives
            or primitive in current_body_risk_clearance_rerank_selected_primitives
        )
        current_body_clearance_rerank = bool(
            bool(args.current_body_risk_clearance_rerank)
            and current_body_risk_prob is not None
            and current_body_clearance_rerank_threshold is not None
            and float(current_body_clearance_rerank_threshold) <= 1.0
            and current_body_claim_gate_active
            and current_body_clearance_rerank_area_active
            and current_body_clearance_rerank_selected_active
            and current_body_clearance_rerank_selected_primitive_active
            and not bool(wall_guard.get("force_escape"))
            and guard_state.upper() in current_body_risk_states
            and float(current_body_risk_prob) >= float(current_body_clearance_rerank_threshold)
            and bool(wall_guard.get("candidates"))
        )
        if (
            bool(args.current_body_risk_clearance_rerank)
            and current_body_risk_prob is not None
            and current_body_clearance_rerank_threshold is not None
            and float(current_body_clearance_rerank_threshold) <= 1.0
            and current_body_claim_gate_active
            and current_body_clearance_rerank_area_active
            and not current_body_clearance_rerank_selected_active
            and not bool(wall_guard.get("force_escape"))
            and guard_state.upper() in current_body_risk_states
            and float(current_body_risk_prob) >= float(current_body_clearance_rerank_threshold)
        ):
            wall_metrics["current_body_risk_clearance_rerank_selected_floor_blocks"] += 1
        if (
            bool(args.current_body_risk_clearance_rerank)
            and current_body_risk_prob is not None
            and current_body_clearance_rerank_threshold is not None
            and float(current_body_clearance_rerank_threshold) <= 1.0
            and current_body_claim_gate_active
            and current_body_clearance_rerank_area_active
            and current_body_clearance_rerank_selected_active
            and not current_body_clearance_rerank_selected_primitive_active
            and not bool(wall_guard.get("force_escape"))
            and guard_state.upper() in current_body_risk_states
            and float(current_body_risk_prob) >= float(current_body_clearance_rerank_threshold)
        ):
            wall_metrics["current_body_risk_clearance_rerank_selected_primitive_blocks"] += 1
        wall_guard["current_body_risk_clearance_rerank_area_active"] = bool(
            current_body_clearance_rerank_area_active
        )
        wall_guard["current_body_risk_claim_gate_active"] = bool(
            current_body_claim_gate_active
        )
        wall_guard["current_body_risk_min_claimed_count"] = int(
            current_body_min_claimed_count
        )
        wall_guard["current_body_risk_claimed_count"] = int(len(beacon_claims))
        wall_guard["current_body_risk_clearance_rerank_selected_active"] = bool(
            current_body_clearance_rerank_selected_active
        )
        wall_guard["current_body_risk_clearance_rerank_selected_primitive_active"] = bool(
            current_body_clearance_rerank_selected_primitive_active
        )
        wall_guard["current_body_risk_clearance_rerank_threshold"] = (
            None
            if current_body_clearance_rerank_threshold is None
            else _round_float(float(current_body_clearance_rerank_threshold), 4)
        )
        wall_guard["current_body_risk_clearance_rerank_min_area_logit"] = (
            None
            if current_body_clearance_rerank_min_area is None
            else _round_float(float(current_body_clearance_rerank_min_area), 4)
        )
        wall_guard["current_body_risk_clearance_rerank_selected_prob_floor"] = (
            None
            if current_body_clearance_rerank_selected_floor is None
            else _round_float(float(current_body_clearance_rerank_selected_floor), 4)
        )
        wall_guard["current_body_risk_clearance_rerank_selected_primitives"] = sorted(
            current_body_risk_clearance_rerank_selected_primitives
        )
        if current_body_clearance_rerank:
            eligible_candidates: list[tuple[float, dict]] = []
            for cand in wall_guard.get("candidates", ()):
                name = str(cand.get("primitive", ""))
                if name not in current_body_risk_clearance_rerank_primitives:
                    continue
                clearance_prob = cand.get("clearance_blocked_prob")
                blocked_prob = cand.get("blocked_prob")
                body_penalty = cand.get("body_clearance_penalty")
                score = (
                    (1.0 if clearance_prob is None else float(clearance_prob))
                    + 0.15 * (0.0 if blocked_prob is None else float(blocked_prob))
                    + 0.05 * (0.0 if body_penalty is None else float(body_penalty))
                )
                eligible_candidates.append((float(score), cand))
            if eligible_candidates:
                _, rerank_candidate = min(eligible_candidates, key=lambda item: item[0])
                rerank_primitive = str(rerank_candidate.get("primitive", primitive))
                if rerank_primitive and rerank_primitive != primitive:
                    risky_primitive = primitive
                    primitive = rerank_primitive
                    selected_clearance_prob = rerank_candidate.get("clearance_blocked_prob")
                    wall_metrics["current_body_risk_clearance_reranks"] += 1
                    wall_guard["current_body_risk_clearance_rerank"] = True
                    wall_guard["selected_before_current_body_risk_clearance_rerank"] = risky_primitive
                    wall_guard["selected_before_current_body_risk_clearance_rerank_prob"] = _round_float(
                        current_body_risk_prob, 4
                    )
                    wall_guard["selected"] = primitive
                    wall_guard["selected_clearance_blocked_prob"] = _round_float(
                        rerank_candidate.get("clearance_blocked_prob"), 4
                    )
                    wall_guard["selected_outcome_blocked_prob"] = _round_float(
                        rerank_candidate.get("outcome_blocked_prob"), 4
                    )
                    wall_guard["selected_progress_m"] = _round_float(
                        rerank_candidate.get("progress_m"), 4
                    )
                    wall_guard["selected_score"] = _round_float(
                        rerank_candidate.get("score"), 4
                    )
                    wall_guard["selected_body_clearance_penalty"] = _round_float(
                        rerank_candidate.get("body_clearance_penalty"), 4
                    )
                    wall_guard["vetoed"] = bool(primitive != requested_primitive)
                    if log_entry is not None:
                        log_entry["current_body_risk_clearance_rerank"] = {
                            "from": risky_primitive,
                            "to": primitive,
                            "prob": _round_float(current_body_risk_prob, 4),
                            "clearance_blocked_prob": _round_float(
                                rerank_candidate.get("clearance_blocked_prob"), 4
                            ),
                        }
        current_body_preserve_yaw = bool(
            bool(args.current_body_risk_preserve_yaw)
            and current_body_risk_prob is not None
            and current_body_preserve_yaw_threshold is not None
            and float(current_body_preserve_yaw_threshold) <= 1.0
            and current_body_claim_gate_active
            and current_body_preserve_yaw_area_active
            and not bool(wall_guard.get("force_escape"))
            and not bool(wall_guard.get("current_body_risk_clearance_rerank"))
            and guard_state.upper() in current_body_risk_states
            and requested_primitive in _TURN_PRIMITIVES
            and primitive in _TRANSLATING_PRIMITIVES
            and float(current_body_risk_prob) >= float(current_body_preserve_yaw_threshold)
        )
        requested_yaw_clearance_prob = None
        if requested_primitive in _TURN_PRIMITIVES:
            for cand in wall_guard.get("candidates", ()):
                if str(cand.get("primitive", "")) == requested_primitive:
                    requested_yaw_clearance_prob = cand.get("clearance_blocked_prob")
                    break
        preserve_yaw_max_clearance_prob = (
            None
            if args.current_body_risk_preserve_yaw_max_clearance_prob is None
            else float(args.current_body_risk_preserve_yaw_max_clearance_prob)
        )
        current_body_preserve_yaw_clearance_active = bool(
            preserve_yaw_max_clearance_prob is None
            or (
                requested_yaw_clearance_prob is not None
                and float(requested_yaw_clearance_prob) <= float(preserve_yaw_max_clearance_prob)
            )
        )
        if current_body_preserve_yaw and not current_body_preserve_yaw_clearance_active:
            wall_metrics["current_body_risk_preserve_yaw_suppressed"] += 1
            current_body_preserve_yaw = False
        wall_guard["current_body_risk_preserve_yaw_area_active"] = bool(
            current_body_preserve_yaw_area_active
        )
        wall_guard["current_body_risk_preserve_yaw_clearance_active"] = bool(
            current_body_preserve_yaw_clearance_active
        )
        wall_guard["current_body_risk_preserve_yaw_requested_clearance_prob"] = _round_float(
            requested_yaw_clearance_prob, 4
        )
        wall_guard["current_body_risk_preserve_yaw_max_clearance_prob"] = (
            None
            if preserve_yaw_max_clearance_prob is None
            else _round_float(float(preserve_yaw_max_clearance_prob), 4)
        )
        wall_guard["current_body_risk_preserve_yaw_threshold"] = (
            None
            if current_body_preserve_yaw_threshold is None
            else _round_float(float(current_body_preserve_yaw_threshold), 4)
        )
        wall_guard["current_body_risk_preserve_yaw_min_area_logit"] = (
            None
            if current_body_preserve_yaw_min_area is None
            else _round_float(float(current_body_preserve_yaw_min_area), 4)
        )
        if current_body_preserve_yaw:
            risky_primitive = primitive
            primitive = requested_primitive
            wall_metrics["current_body_risk_preserve_yaw_overrides"] += 1
            wall_guard["current_body_risk_preserve_yaw"] = True
            wall_guard["selected_before_current_body_risk_preserve_yaw"] = risky_primitive
            wall_guard["selected_before_current_body_risk_preserve_yaw_prob"] = _round_float(
                current_body_risk_prob, 4
            )
            wall_guard["selected"] = primitive
            wall_guard["vetoed"] = bool(primitive != requested_primitive)
            if log_entry is not None:
                log_entry["current_body_risk_preserve_yaw"] = {
                    "from": risky_primitive,
                    "to": primitive,
                    "prob": _round_float(current_body_risk_prob, 4),
                }
        current_body_recovery_selected_floor = (
            None
            if args.current_body_risk_recovery_selected_prob_floor is None
            else float(args.current_body_risk_recovery_selected_prob_floor)
        )
        current_body_recovery_selected_prob = wall_guard.get("selected_clearance_blocked_prob")
        current_body_recovery_selected_active = bool(
            current_body_recovery_selected_floor is None
            or (
                current_body_recovery_selected_prob is not None
                and float(current_body_recovery_selected_prob)
                >= float(current_body_recovery_selected_floor)
            )
        )
        current_body_recovery_selected_primitive_active = bool(
            not current_body_risk_recovery_selected_primitives
            or primitive in current_body_risk_recovery_selected_primitives
        )
        current_body_recovery_base = bool(
            current_body_risk_prob is not None
            and current_body_threshold is not None
            and float(current_body_threshold) <= 1.0
            and int(args.current_body_risk_recovery_blocks) > 0
            and current_body_claim_gate_active
            and current_body_area_active
            and int(current_body_risk_cooldown) <= 0
            and not bool(wall_guard.get("force_escape"))
            and not escape_plan
            and guard_state.upper() in current_body_risk_states
            and primitive != "backward"
            and float(current_body_risk_prob) >= float(current_body_threshold)
        )
        current_body_recovery = bool(
            current_body_recovery_base
            and current_body_recovery_selected_active
            and current_body_recovery_selected_primitive_active
        )
        wall_guard["current_body_risk_area_active"] = bool(current_body_area_active)
        wall_guard["current_body_risk_min_area_logit"] = (
            None
            if args.current_body_risk_min_area_logit is None
            else _round_float(float(args.current_body_risk_min_area_logit), 4)
        )
        wall_guard["current_body_risk_recovery_selected_prob_floor"] = (
            None
            if current_body_recovery_selected_floor is None
            else _round_float(float(current_body_recovery_selected_floor), 4)
        )
        wall_guard["current_body_risk_recovery_selected_active"] = bool(
            current_body_recovery_selected_active
        )
        wall_guard["current_body_risk_recovery_selected_primitive_active"] = bool(
            current_body_recovery_selected_primitive_active
        )
        wall_guard["current_body_risk_recovery_selected_primitives"] = sorted(
            current_body_risk_recovery_selected_primitives
        )
        if current_body_recovery_base and not current_body_recovery_selected_active:
            wall_metrics["current_body_risk_recovery_selected_floor_blocks"] += 1
        if current_body_recovery_base and not current_body_recovery_selected_primitive_active:
            wall_metrics["current_body_risk_recovery_selected_primitive_blocks"] += 1
        if current_body_recovery:
            risky_primitive = primitive if primitive != "hold" else last_primitive
            current_body_plan = _make_escape_plan(
                risky_primitive,
                int(args.current_body_risk_recovery_blocks),
            )
            if current_body_plan:
                primitive = current_body_plan.pop(0)
                escape_plan = current_body_plan + escape_plan
                current_body_risk_cooldown = max(
                    0,
                    int(args.current_body_risk_cooldown_ticks),
                )
                wall_metrics["current_body_risk_recoveries"] += 1
                wall_metrics["current_body_risk_recovery_blocks_executed"] += 1
                wall_metrics["escape_blocks_executed"] += 1
                wall_guard["current_body_risk_recovery"] = True
                wall_guard["selected_before_current_body_risk_recovery"] = risky_primitive
                wall_guard["selected_before_current_body_risk_recovery_prob"] = _round_float(
                    current_body_risk_prob, 4
                )
                wall_guard["selected"] = primitive
                wall_guard["vetoed"] = bool(primitive != requested_primitive)
                wall_guard["force_escape"] = True
                if log_entry is not None:
                    log_entry["current_body_risk_recovery"] = {
                        "from": risky_primitive,
                        "to": primitive,
                        "prob": _round_float(current_body_risk_prob, 4),
                        "remaining_plan": list(escape_plan),
                    }
        post_guard_current_contact_escape_m = (
            body_clearance_current_contact_escape_m_by_primitive.get(primitive)
            if body_clearance_current_contact_escape_m_by_primitive
            and primitive in body_clearance_current_contact_escape_m_by_primitive
            else active_current_contact_escape_m
        )
        post_guard_current_contact_escape = bool(
            post_guard_current_contact_escape_m is not None
            and current_body_clearance_for_guard is not None
            and float(current_body_clearance_for_guard) <= float(post_guard_current_contact_escape_m)
            and not bool(wall_guard.get("body_clearance_current_contact_escape"))
            and primitive
            in set(
                body_clearance_current_contact_escape_primitives
                or {
                    "forward_fast",
                    "forward_medium",
                    "arc_left",
                    "arc_right",
                    "yaw_left",
                    "yaw_right",
                }
            )
            and primitive != "backward"
        )
        if (
            active_current_contact_escape_m is not None
            and post_guard_current_contact_escape_m is not None
            and current_body_clearance_for_guard is not None
            and float(current_body_clearance_for_guard) > float(post_guard_current_contact_escape_m)
            and primitive
            in set(
                body_clearance_current_contact_escape_primitives
                or {
                    "forward_fast",
                    "forward_medium",
                    "arc_left",
                    "arc_right",
                    "yaw_left",
                    "yaw_right",
                }
            )
            and primitive != "backward"
            and not bool(wall_guard.get("body_clearance_current_contact_escape"))
        ):
            wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                "primitive_clearance"
            )
        if post_guard_current_contact_escape:
            selected_projected_ok, selected_projected_clearance, selected_projected_reason = (
                _current_contact_projected_clearance_ok(
                    primitive,
                    projected_clearances=current_contact_escape_projected_clearances,
                    current_body_clearance_m=current_body_clearance_for_guard,
                    min_projected_clearance_m=(
                        None
                        if args.body_clearance_current_contact_escape_min_projected_clearance_m
                        is None
                        else float(
                            args.body_clearance_current_contact_escape_min_projected_clearance_m
                        )
                    ),
                    min_projected_improvement_m=float(
                        args.body_clearance_current_contact_escape_min_projected_improvement_m
                    ),
                )
            )
            if (
                current_contact_escape_projected_clearances is not None
                and selected_projected_ok
            ):
                post_guard_current_contact_escape = False
                wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                    "selected_projected_safe"
                )
                wall_guard["body_clearance_current_contact_escape_candidate"] = {
                    "primitive": primitive,
                    "projected_clearance_m": _round_float(
                        selected_projected_clearance, 4
                    ),
                    "suppressed": True,
                    "post_guard": True,
                }
            elif selected_projected_reason is not None:
                wall_guard.setdefault(
                    "body_clearance_current_contact_escape_projected_rejections",
                    [],
                ).append({
                    "primitive": primitive,
                    "projected_clearance_m": _round_float(
                        selected_projected_clearance, 4
                    ),
                    "reason": selected_projected_reason,
                    "selected": True,
                    "post_guard": True,
                })
        if post_guard_current_contact_escape:
            replacement_primitives = list(
                body_clearance_current_contact_escape_replacements
                or {"backward", "yaw_left", "yaw_right", "hold"}
            )
            ranked_replacements: list[tuple[float, str, dict[str, Any]]] = []
            ranked_replacements_under_cap: list[tuple[float, str, dict[str, Any]]] = []
            for candidate in wall_guard.get("candidates", ()):
                if not isinstance(candidate, dict):
                    continue
                candidate_primitive = str(candidate.get("primitive", ""))
                if (
                    candidate_primitive == primitive
                    or candidate_primitive not in replacement_primitives
                ):
                    continue
                clearance_prob = candidate.get(
                    "clearance_blocked_prob",
                    candidate.get("blocked_prob"),
                )
                replacement_under_cap = bool(
                    float(args.body_clearance_current_contact_escape_replacement_cap) > 1.0
                    or (
                        clearance_prob is not None
                        and float(clearance_prob)
                        <= float(args.body_clearance_current_contact_escape_replacement_cap)
                    )
                )
                if not replacement_under_cap:
                    wall_guard.setdefault(
                        "body_clearance_current_contact_escape_projected_rejections",
                        [],
                    ).append({
                        "primitive": candidate_primitive,
                        "clearance_blocked_prob": _round_float(clearance_prob, 4),
                        "reason": "replacement_cap",
                        "selected": False,
                        "post_guard": True,
                    })
                projected_ok, projected_clearance, projected_reason = (
                    _current_contact_projected_clearance_ok(
                        candidate_primitive,
                        projected_clearances=current_contact_escape_projected_clearances,
                        current_body_clearance_m=current_body_clearance_for_guard,
                        min_projected_clearance_m=(
                            None
                            if args.body_clearance_current_contact_escape_min_projected_clearance_m
                            is None
                            else float(
                                args.body_clearance_current_contact_escape_min_projected_clearance_m
                            )
                        ),
                        min_projected_improvement_m=float(
                            args.body_clearance_current_contact_escape_min_projected_improvement_m
                        ),
                    )
                )
                if not projected_ok:
                    wall_guard.setdefault(
                        "body_clearance_current_contact_escape_projected_rejections",
                        [],
                    ).append({
                        "primitive": candidate_primitive,
                        "projected_clearance_m": _round_float(
                            projected_clearance, 4
                        ),
                        "reason": projected_reason,
                        "selected": False,
                        "post_guard": True,
                    })
                    continue
                ranked_replacement = (
                    _current_contact_escape_score(
                        candidate_primitive,
                        clearance_blocked_prob=clearance_prob,
                        blocked_prob=candidate.get("blocked_prob"),
                        candidate_score=candidate.get("score"),
                    ),
                    candidate_primitive,
                    candidate,
                )
                ranked_replacements.append(ranked_replacement)
                if replacement_under_cap:
                    ranked_replacements_under_cap.append(ranked_replacement)
            replacement_candidate: dict[str, Any] | None = None
            replacement_primitive: str | None = None
            if ranked_replacements:
                if (
                    bool(
                        args.body_clearance_current_contact_escape_require_replacement_under_cap
                    )
                    and float(args.body_clearance_current_contact_escape_replacement_cap)
                    <= 1.0
                    and not ranked_replacements_under_cap
                ):
                    wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                        "replacement_cap"
                    )
                else:
                    ranked_replacement_pool = (
                        ranked_replacements_under_cap or ranked_replacements
                    )
                    _, replacement_primitive, replacement_candidate = min(
                        ranked_replacement_pool,
                        key=lambda item: (
                            item[0],
                            float(
                                item[2].get(
                                    "clearance_blocked_prob",
                                    item[2].get("blocked_prob", 1.0),
                                )
                            ),
                        ),
                    )
            elif current_contact_escape_projected_clearances is not None:
                fallback_replacements = [
                    name
                    for name in ("backward", "hold", "yaw_left", "yaw_right")
                    if name in replacement_primitives and name != primitive
                ]
                if not fallback_replacements:
                    fallback_replacements = [
                        name for name in replacement_primitives if name != primitive
                    ]
                replacement_primitive = (
                    next(
                        (
                            name
                            for name in fallback_replacements
                            if _current_contact_projected_clearance_ok(
                                name,
                                projected_clearances=current_contact_escape_projected_clearances,
                                current_body_clearance_m=current_body_clearance_for_guard,
                                min_projected_clearance_m=(
                                    None
                                    if args.body_clearance_current_contact_escape_min_projected_clearance_m
                                    is None
                                    else float(
                                        args.body_clearance_current_contact_escape_min_projected_clearance_m
                                    )
                                ),
                                min_projected_improvement_m=float(
                                    args.body_clearance_current_contact_escape_min_projected_improvement_m
                                ),
                            )[0]
                        ),
                        None,
                    )
                    if fallback_replacements
                    else None
                )
                if replacement_primitive is None:
                    wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                        "no_projected_fallback"
                    )
            else:
                wall_guard["body_clearance_current_contact_escape_suppressed_reason"] = (
                    "no_scored_candidate"
                )
            if replacement_primitive is not None:
                previous_primitive = primitive
                primitive = replacement_primitive
                wall_guard["body_clearance_current_contact_escape"] = True
                wall_guard["body_clearance_current_contact_escape_post_guard"] = True
                wall_guard["selected_before_body_clearance_current_contact_escape"] = (
                    previous_primitive
                )
                wall_guard["body_clearance_current_contact_escape_m"] = _round_float(
                    float(post_guard_current_contact_escape_m), 4
                )
                wall_guard["current_body_clearance_m"] = _round_float(
                    float(current_body_clearance_for_guard), 4
                )
                if replacement_candidate is not None:
                    wall_guard["body_clearance_current_contact_escape_candidate"] = {
                        "primitive": primitive,
                        "clearance_blocked_prob": _round_float(
                            replacement_candidate.get("clearance_blocked_prob"), 4
                        ),
                        "blocked_prob": _round_float(
                            replacement_candidate.get("blocked_prob"), 4
                        ),
                        "score": _round_float(replacement_candidate.get("score"), 4),
                        "projected_clearance_m": _round_float(
                            (
                                current_contact_escape_projected_clearances
                                or {}
                            ).get(primitive),
                            4,
                        ),
                        "replacement_cap": _round_float(
                            args.body_clearance_current_contact_escape_replacement_cap,
                            4,
                        ),
                        "replacement_cap_relaxed": bool(
                            float(args.body_clearance_current_contact_escape_replacement_cap) <= 1.0
                            and not ranked_replacements_under_cap
                        ),
                    }
                _update_guard_selected_from_candidate(
                    wall_guard,
                    primitive,
                    requested_primitive,
                )
                if log_entry is not None:
                    log_entry["body_clearance_current_contact_escape"] = {
                        "from": previous_primitive,
                        "to": primitive,
                        "current_body_clearance_m": _round_float(
                            float(current_body_clearance_for_guard), 4
                        ),
                        "post_guard": True,
                        "threshold_m": _round_float(
                            post_guard_current_contact_escape_m, 4
                        ),
                    }
        hard_veto_hold = bool(
            primitive == "hold" and wall_guard.get("body_clearance_hard_veto")
        )
        if hard_veto_hold:
            body_clearance_hard_veto_hold_streak += 1
            wall_metrics["body_clearance_hard_veto_hold_ticks"] += 1
            wall_metrics["body_clearance_hard_veto_hold_streak_max"] = max(
                int(wall_metrics["body_clearance_hard_veto_hold_streak_max"]),
                int(body_clearance_hard_veto_hold_streak),
            )
        else:
            body_clearance_hard_veto_hold_streak = 0
        hold_escape_after = max(
            0,
            int(args.body_clearance_hard_veto_hold_escape_after),
        )
        base_hold_escape_state_active = bool(
            not body_clearance_hard_veto_hold_escape_states
            or guard_state.upper() in body_clearance_hard_veto_hold_escape_states
        )
        hold_escape_override_min_claimed_count = max(
            0,
            int(args.body_clearance_hard_veto_hold_escape_override_min_claimed_count),
        )
        hold_escape_override_state_active = bool(
            not body_clearance_hard_veto_hold_escape_override_states
            or guard_state.upper() in body_clearance_hard_veto_hold_escape_override_states
        )
        hold_escape_override_min_current_clearance_m = (
            None
            if args.body_clearance_hard_veto_hold_escape_override_min_current_clearance_m
            is None
            else float(args.body_clearance_hard_veto_hold_escape_override_min_current_clearance_m)
        )
        hold_escape_override_clearance_active = bool(
            hold_escape_override_min_current_clearance_m is None
            or (
                current_body_clearance_for_guard is not None
                and float(current_body_clearance_for_guard)
                >= float(hold_escape_override_min_current_clearance_m)
            )
        )
        hold_escape_override_active = bool(
            body_clearance_hard_veto_hold_escape_override_primitives
            and len(beacon_claims) >= hold_escape_override_min_claimed_count
            and hold_escape_override_state_active
            and hold_escape_override_clearance_active
        )
        hold_escape_state_active = bool(
            base_hold_escape_state_active or hold_escape_override_active
        )
        hold_escape_allowed = bool(
            hard_veto_hold
            and hold_escape_after > 0
            and body_clearance_hard_veto_hold_streak >= hold_escape_after
            and hold_escape_state_active
            and not bool(wall_guard.get("force_escape"))
            and not escape_plan
        )
        if hold_escape_allowed:
            hold_escape_candidates: list[tuple[float, str, dict[str, Any], float | None]] = []
            hold_escape_projected_rejections: list[dict[str, Any]] = []
            capped_candidates = 0
            active_hold_escape_primitives = body_clearance_hard_veto_hold_escape_primitives
            max_clearance_prob = float(args.body_clearance_hard_veto_hold_escape_max_clearance_prob)
            if hold_escape_override_active:
                active_hold_escape_primitives = (
                    body_clearance_hard_veto_hold_escape_override_primitives
                )
                if (
                    args.body_clearance_hard_veto_hold_escape_override_max_clearance_prob
                    is not None
                ):
                    max_clearance_prob = float(
                        args.body_clearance_hard_veto_hold_escape_override_max_clearance_prob
                    )
            hold_escape_min_projected_clearance_m = (
                None
                if args.body_clearance_hard_veto_hold_escape_min_projected_clearance_m
                is None
                else float(args.body_clearance_hard_veto_hold_escape_min_projected_clearance_m)
            )
            hold_escape_min_projected_improvement_m = float(
                args.body_clearance_hard_veto_hold_escape_min_projected_improvement_m
            )
            hold_escape_projection_active = bool(
                hold_escape_min_projected_clearance_m is not None
                or hold_escape_min_projected_improvement_m > 0.0
            )
            hold_escape_projected_clearances: dict[str, float] | None = None
            if hold_escape_projection_active:
                if current_body_clearance_for_guard is None:
                    try:
                        current_body_clearance_for_guard = _body_probe_clearance(
                            grid,
                            pos[:2],
                            yaw,
                            body_forward_m=float(args.wall_body_forward_m),
                            body_half_width_m=float(args.wall_body_half_width_m),
                            body_probe_margin_m=float(args.wall_body_probe_margin_m),
                        )
                    except Exception:
                        current_body_clearance_for_guard = None
                if current_contact_escape_projected_clearances is not None:
                    hold_escape_projected_clearances = dict(
                        current_contact_escape_projected_clearances
                    )
                else:
                    hold_escape_projected_clearances = {}
                    for projection_primitive in outcome_primitive_vocab:
                        try:
                            projection_report = _primitive_clearance_report(
                                registry,
                                str(projection_primitive),
                                pos[:2],
                                yaw,
                                grid,
                                float(args.command_dt_s),
                                body_forward_m=float(args.wall_body_forward_m),
                                body_half_width_m=float(args.wall_body_half_width_m),
                                body_probe_margin_m=float(args.wall_body_probe_margin_m),
                                min_clearance_m=float(args.wall_min_clearance_m),
                            )
                        except Exception:
                            continue
                        hold_escape_projected_clearances[
                            str(projection_primitive)
                        ] = float(projection_report["min_clearance_m"])
            for cand in wall_guard.get("candidates", ()):
                cand_name = str(cand.get("primitive", ""))
                if (
                    cand_name == "hold"
                    or cand_name not in active_hold_escape_primitives
                ):
                    continue
                clearance_prob = cand.get("clearance_blocked_prob")
                if clearance_prob is None:
                    continue
                if float(clearance_prob) > max_clearance_prob:
                    capped_candidates += 1
                    continue
                projected_ok, projected_clearance, projected_reason = (
                    _current_contact_projected_clearance_ok(
                        cand_name,
                        projected_clearances=hold_escape_projected_clearances,
                        current_body_clearance_m=current_body_clearance_for_guard,
                        min_projected_clearance_m=hold_escape_min_projected_clearance_m,
                        min_projected_improvement_m=hold_escape_min_projected_improvement_m,
                    )
                )
                if not projected_ok:
                    hold_escape_projected_rejections.append({
                        "primitive": cand_name,
                        "reason": projected_reason,
                        "clearance_blocked_prob": _round_float(clearance_prob, 4),
                        "projected_clearance_m": _round_float(projected_clearance, 4),
                        "current_body_clearance_m": _round_float(
                            current_body_clearance_for_guard, 4
                        ),
                    })
                    continue
                blocked_prob = float(cand.get("blocked_prob", 0.0) or 0.0)
                candidate_score = float(cand.get("score", 0.0) or 0.0)
                if cand_name == "backward":
                    primitive_bias = 0.0
                elif cand_name in _TURN_PRIMITIVES:
                    primitive_bias = 0.04
                else:
                    primitive_bias = 0.18
                hold_escape_score = (
                    float(clearance_prob)
                    + 0.15 * blocked_prob
                    + primitive_bias
                    - 0.02 * candidate_score
                )
                hold_escape_candidates.append((
                    hold_escape_score,
                    cand_name,
                    cand,
                    projected_clearance,
                ))
            wall_metrics["body_clearance_hard_veto_hold_escape_capped_candidates"] += int(
                capped_candidates
            )
            if hold_escape_projected_rejections:
                wall_metrics[
                    "body_clearance_hard_veto_hold_escape_projection_rejections"
                ] += int(len(hold_escape_projected_rejections))
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_projected_rejections"
                ] = hold_escape_projected_rejections[:8]
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_min_projected_clearance_m"
                ] = (
                    None
                    if hold_escape_min_projected_clearance_m is None
                    else _round_float(hold_escape_min_projected_clearance_m, 4)
                )
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_min_projected_improvement_m"
                ] = _round_float(hold_escape_min_projected_improvement_m, 4)
            if hold_escape_candidates:
                hold_escape_streak_before = int(body_clearance_hard_veto_hold_streak)
                (
                    _,
                    hold_escape_primitive,
                    hold_escape_candidate,
                    hold_escape_projected_clearance,
                ) = min(
                    hold_escape_candidates,
                    key=lambda item: item[0],
                )
                previous_primitive = primitive
                primitive = hold_escape_primitive
                body_clearance_hard_veto_hold_streak = 0
                wall_metrics["body_clearance_hard_veto_hold_escapes"] += 1
                if hold_escape_override_active:
                    wall_metrics["body_clearance_hard_veto_hold_escape_overrides"] += 1
                wall_guard["body_clearance_hard_veto_hold_escape"] = True
                wall_guard["body_clearance_hard_veto_hold_escape_override"] = bool(
                    hold_escape_override_active
                )
                if hold_escape_override_min_current_clearance_m is not None:
                    wall_guard[
                        "body_clearance_hard_veto_hold_escape_override_min_current_clearance_m"
                    ] = _round_float(hold_escape_override_min_current_clearance_m, 4)
                wall_guard["body_clearance_hard_veto_hold_escape_after"] = int(
                    hold_escape_after
                )
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_streak_before"
                ] = int(hold_escape_streak_before)
                wall_guard[
                    "selected_before_body_clearance_hard_veto_hold_escape"
                ] = previous_primitive
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_max_clearance_prob"
                ] = _round_float(max_clearance_prob, 4)
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_primitives"
                ] = sorted(active_hold_escape_primitives)
                wall_guard["body_clearance_hard_veto_hold_escape_candidate"] = {
                    "primitive": str(hold_escape_primitive),
                    "clearance_blocked_prob": _round_float(
                        hold_escape_candidate.get("clearance_blocked_prob"), 4
                    ),
                    "blocked_prob": _round_float(
                        hold_escape_candidate.get("blocked_prob"), 4
                    ),
                    "progress_m": _round_float(
                        hold_escape_candidate.get("progress_m"), 4
                    ),
                }
                if hold_escape_projected_clearance is not None:
                    wall_guard[
                        "body_clearance_hard_veto_hold_escape_candidate"
                    ]["projected_clearance_m"] = _round_float(
                        hold_escape_projected_clearance, 4
                    )
                _update_guard_selected_from_candidate(
                    wall_guard,
                    primitive,
                    requested_primitive,
                )
                if log_entry is not None:
                    log_entry["body_clearance_hard_veto_hold_escape"] = {
                        "from": previous_primitive,
                        "to": primitive,
                        "after": int(hold_escape_after),
                        "max_clearance_prob": _round_float(max_clearance_prob, 4),
                        "override": bool(hold_escape_override_active),
                    }
            else:
                wall_metrics[
                    "body_clearance_hard_veto_hold_escape_no_candidate_ticks"
                ] += 1
                wall_guard["body_clearance_hard_veto_hold_escape_no_candidate"] = True
                wall_guard["body_clearance_hard_veto_hold_escape_override"] = bool(
                    hold_escape_override_active
                )
                if hold_escape_override_min_current_clearance_m is not None:
                    wall_guard[
                        "body_clearance_hard_veto_hold_escape_override_min_current_clearance_m"
                    ] = _round_float(hold_escape_override_min_current_clearance_m, 4)
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_max_clearance_prob"
                ] = _round_float(max_clearance_prob, 4)
                wall_guard[
                    "body_clearance_hard_veto_hold_escape_primitives"
                ] = sorted(active_hold_escape_primitives)
                if hold_escape_projected_rejections:
                    projection_block_primitive = str(
                        wall_guard.get("selected_before_body_clearance_hard_veto")
                        or ""
                    )
                    if projection_block_primitive in _TRANSLATING_PRIMITIVES:
                        wall_guard[
                            "body_clearance_hard_veto_hold_escape_projection_block_primitive"
                        ] = projection_block_primitive
        if (
            history_risk_probs is not None
            and bool(args.history_risk_corridor_commit)
            and not bool(wall_guard.get("force_escape"))
            and primitive in ("yaw_left", "yaw_right", "hold")
            and guard_state.upper() in history_risk_corridor_states
            and int(history_risk_corridor_run) < int(args.history_risk_corridor_max_run)
            and float(history_risk_probs.get("yaw_left", 0.0))
            >= float(args.history_risk_corridor_yaw_min)
            and float(history_risk_probs.get("yaw_right", 0.0))
            >= float(args.history_risk_corridor_yaw_min)
            and float(history_risk_probs.get("forward_medium", 1.0))
            <= float(args.history_risk_corridor_forward_max)
            and not bool(stalled_prev_tick)
        ):
            history_corridor_from = primitive
            primitive = "forward_medium"
            history_risk_corridor_run += 1
            wall_metrics["history_risk_corridor_commits"] += 1
            wall_guard["history_risk_corridor_commit"] = True
            wall_guard["selected"] = primitive
            wall_guard["vetoed"] = bool(primitive != requested_primitive)
            if log_entry is not None:
                log_entry["history_risk_corridor_commit"] = {
                    "from": history_corridor_from,
                    "yaw_left": _round_float(float(history_risk_probs.get("yaw_left", 0.0)), 4),
                    "yaw_right": _round_float(float(history_risk_probs.get("yaw_right", 0.0)), 4),
                    "forward": _round_float(float(history_risk_probs.get("forward_medium", 1.0)), 4),
                    "run": int(history_risk_corridor_run),
                }
        else:
            history_risk_corridor_run = 0
        effective_history_risk_veto_threshold = (
            float(args.history_risk_relaxed_veto_threshold)
            if (
                int(args.history_risk_relax_min_claims) >= 0
                and len(beacon_claims) >= int(args.history_risk_relax_min_claims)
            )
            else float(args.history_risk_veto_threshold)
        )
        if (
            history_risk_probs is not None
            and float(effective_history_risk_veto_threshold) <= 1.0
            and not bool(wall_guard.get("force_escape"))
            and primitive in history_risk_veto_primitives
            and (
                not history_risk_states
                or guard_state.upper() in history_risk_states
            )
            and float(history_risk_probs.get(primitive, 0.0))
            >= float(effective_history_risk_veto_threshold)
        ):
            if primitive in ("yaw_left", "yaw_right"):
                # A vetoed scanning yaw must stay a rotation where possible:
                # the opposite yaw sweeps away from the grinding wall and
                # preserves the explorer's scan; backward is the fallback.
                opposite_yaw = "yaw_right" if primitive == "yaw_left" else "yaw_left"
                history_replacement_pool = [opposite_yaw, "backward"]
            else:
                # A vetoed translation should scan for an opening rather than
                # retreat: prefer the lower-risk yaw, then backward.
                yaw_by_risk = sorted(
                    ("yaw_left", "yaw_right"),
                    key=lambda name: float(history_risk_probs.get(name, 1.0)),
                )
                history_replacement_pool = [*yaw_by_risk, "backward"] + [
                    name
                    for name in history_risk_replacements
                    if name not in ("yaw_left", "yaw_right", "backward")
                ]
            history_replacement_candidates = [
                (float(history_risk_probs.get(candidate, 1.0)), candidate)
                for candidate in history_replacement_pool
                if candidate != primitive
            ]
            history_vetoed_from = primitive
            history_vetoed_from_prob = float(history_risk_probs.get(primitive, 0.0))
            history_replacement: str | None = None
            for candidate_prob, candidate in history_replacement_candidates:
                if candidate_prob < float(args.history_risk_replacement_cap):
                    history_replacement = candidate
                    break
            if history_replacement is None and history_replacement_candidates:
                # Everything is risky (likely already wedged). Per-tick argmin
                # replacement ping-pongs between grinding yaws, so commit to a
                # short escape in the least-risky direction instead.
                wedge_safest = min(
                    (
                        (float(history_risk_probs.get(name, 1.0)), name)
                        for name in ("backward", "yaw_left", "yaw_right")
                        if name != primitive
                    ),
                )[1]
                history_replacement = wedge_safest
                if (
                    int(args.history_risk_wedge_escape_blocks) > 0
                    and not escape_plan
                    and int(history_risk_wedge_cooldown) <= 0
                ):
                    escape_plan = [wedge_safest] * int(
                        args.history_risk_wedge_escape_blocks
                    ) + escape_plan
                    history_risk_wedge_cooldown = int(
                        args.history_risk_wedge_escape_cooldown_ticks
                    )
                    wall_metrics["history_risk_wedge_escapes"] += 1
                    if (
                        bool(args.proprio_contact_map_blocks)
                        and learned_local_online_map is not None
                        and history_vetoed_from in _TRANSLATING_PRIMITIVES
                    ):
                        if learned_local_online_map.mark_blocked_primitive(
                            pos[:2],
                            float(yaw),
                            str(history_vetoed_from),
                        ):
                            wall_metrics["proprio_contact_map_blocked_edges"] += 1
                    if log_entry is not None:
                        log_entry["history_risk_wedge_escape"] = {
                            "primitive": wedge_safest,
                            "blocks": int(args.history_risk_wedge_escape_blocks),
                        }
            if history_replacement is None:
                history_replacement = "hold"
            primitive = history_replacement
            wall_metrics["history_risk_vetoes"] += 1
            wall_guard["history_risk_veto"] = True
            wall_guard["selected_before_history_risk_veto"] = history_vetoed_from
            wall_guard["selected_before_history_risk_veto_prob"] = _round_float(
                history_vetoed_from_prob, 4
            )
            wall_guard["selected"] = primitive
            wall_guard["vetoed"] = bool(primitive != requested_primitive)
            if log_entry is not None:
                log_entry["history_risk_veto"] = {
                    "from": history_vetoed_from,
                    "from_prob": _round_float(history_vetoed_from_prob, 4),
                    "to": primitive,
                    "to_prob": _round_float(
                        float(history_risk_probs.get(primitive, 1.0)), 4
                    ),
                }
        smoothing_enabled = bool(
            log_entry is not None
            and int(args.command_smoothing_min_ticks) > 1
            and last_primitive_run_ticks > 0
            and last_primitive_run_ticks < int(args.command_smoothing_min_ticks)
            and primitive != last_primitive
            and last_primitive in command_smoothing_primitives
            and primitive in command_smoothing_primitives
            and (not command_smoothing_states or guard_state.upper() in command_smoothing_states)
            and not bool(wall_guard.get("force_escape"))
            and not bool(log_entry.get("body_clearance_request"))
        )
        if smoothing_enabled:
            opposite_yaw = (
                (last_primitive == "yaw_left" and primitive == "yaw_right")
                or (last_primitive == "yaw_right" and primitive == "yaw_left")
            )
            if opposite_yaw:
                wall_metrics["command_smoothing_opposite_yaw_blocks"] += 1
            elif _guard_candidate_is_blocked(wall_guard, last_primitive):
                wall_metrics["command_smoothing_blocked_holds"] += 1
            else:
                unsmoothed_primitive = primitive
                primitive = last_primitive
                wall_metrics["command_smoothing_overrides"] += 1
                wall_guard["command_smoothing_override"] = True
                wall_guard["selected_before_command_smoothing"] = unsmoothed_primitive
                wall_guard["command_smoothing_min_ticks"] = int(args.command_smoothing_min_ticks)
                wall_guard["command_smoothing_run_ticks_before"] = int(last_primitive_run_ticks)
                _update_guard_selected_from_candidate(wall_guard, primitive, requested_primitive)
                if log_entry is not None:
                    log_entry["command_smoothing_override"] = {
                        "from": unsmoothed_primitive,
                        "to": primitive,
                        "run_ticks_before": int(last_primitive_run_ticks),
                    }
        forced_primitive = debug_force_primitive_script.get(int(tick))
        if forced_primitive is not None:
            previous_primitive = primitive
            primitive = str(forced_primitive)
            wall_metrics["debug_force_primitive_overrides"] += 1
            wall_guard["debug_force_primitive_override"] = True
            wall_guard["selected_before_debug_force_primitive"] = previous_primitive
            wall_guard["selected"] = primitive
            wall_guard["vetoed"] = bool(primitive != requested_primitive)
            if log_entry is not None:
                log_entry["debug_force_primitive_override"] = {
                    "from": previous_primitive,
                    "to": primitive,
                    "script": (
                        None
                        if args.debug_force_primitive_script is None
                        else str(args.debug_force_primitive_script)
                    ),
                }
        if (
            args.learned_local_dataset_output is not None
            and str(args.learned_local_dataset_label_source) == "executed"
            and learned_local_dataset_state_active
            and learned_policy_feature_for_dataset is not None
            and log_entry is not None
        ):
            executed_label_primitive = _learned_local_policy_label_primitive(primitive)
            if executed_label_primitive is None:
                wall_metrics["learned_local_policy_skipped_unmapped_examples"] += 1
            else:
                if executed_label_primitive != primitive:
                    wall_metrics["learned_local_policy_label_mapped_examples"] += 1
                feature_np = (
                    learned_policy_feature_for_dataset.detach().cpu().numpy().astype(np.float32)
                )
                feature_dim = int(feature_np.reshape(-1).shape[0])
                if learned_local_dataset_feature_dim is None:
                    learned_local_dataset_feature_dim = feature_dim
                if feature_dim != int(learned_local_dataset_feature_dim):
                    wall_metrics["learned_local_policy_skipped_feature_dim_examples"] += 1
                else:
                    learned_local_dataset_features.append(feature_np)
                    learned_local_dataset_labels.append(
                        _LEARNED_LOCAL_POLICY_PRIMITIVES.index(executed_label_primitive)
                    )
                    learned_local_dataset_meta.append(
                        {
                            "tick": int(tick),
                            "state": str(log_entry.get("state", "")),
                            "label": executed_label_primitive,
                            "dataset_label_source": "executed",
                            "policy_feature_slot": str(learned_policy_feature_slot_for_dataset),
                            "executed_label_source_primitive": primitive,
                            "oracle_standoff_label": False,
                            "target_color": str(log_entry.get("target_color", "")),
                            "target_index": int(log_entry.get("target_index", -1)),
                            "mem_conf": float(log_entry.get("mem_conf", 0.0)),
                            "area": float(log_entry.get("area", -99.0)),
                            "bearing": float(log_entry.get("bearing", 0.0)),
                            "in_cone": bool(log_entry.get("in_cone", False)),
                            "seen": bool(log_entry.get("seen", False)),
                            "state_seen": bool(log_entry.get("state_seen", False)),
                            "read_score": (
                                None
                                if log_entry.get("read_score") is None
                                else float(log_entry.get("read_score", 0.0))
                            ),
                            "read_gate_pass": (
                                None
                                if log_entry.get("read_gate_pass") is None
                                else bool(log_entry.get("read_gate_pass", False))
                            ),
                            "seen_age_ticks": int(log_entry.get("seen_age_ticks", 0)),
                            "pose_xy": [float(pos[0]), float(pos[1])],
                            "yaw_rad": float(yaw),
                            "claimed_count": int(len(beacon_claims)),
                            "standoff_route_gate": bool(log_entry.get("standoff_route_gate", False)),
                        }
                    )
                    wall_metrics["learned_local_policy_collected_examples"] += 1
        wall_metrics["commands_total"] += 1
        if requested_primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_requests"] += 1
            if bool(wall_guard["enabled"]) and bool(wall_guard["requested_blocked"]):
                wall_metrics["blocked_forward_requests"] += 1
        if primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_executions"] += 1
            if bool(wall_guard["enabled"]) and bool(wall_guard["selected_blocked"]):
                wall_metrics["blocked_forward_executions"] += 1
        if bool(wall_guard["vetoed"]):
            wall_metrics["wall_vetoes"] += 1
        requested_body_penalty = wall_guard.get("requested_body_clearance_penalty")
        if requested_body_penalty is not None and float(requested_body_penalty) > 0.0:
            wall_metrics["body_clearance_learned_penalty_ticks"] += 1
            if bool(wall_guard["vetoed"]):
                wall_metrics["body_clearance_learned_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_hard_veto")):
            wall_metrics["body_clearance_hard_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_arc_sweep_veto")):
            wall_metrics["body_clearance_arc_sweep_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_aux_veto")):
            wall_metrics["body_clearance_aux_vetoes"] += 1
        elif (
            isinstance(wall_guard.get("body_clearance_aux_veto_log"), dict)
            and wall_guard["body_clearance_aux_veto_log"].get("suppressed") is not None
        ):
            wall_metrics["body_clearance_aux_veto_suppressed_ticks"] += 1
        if bool(wall_guard.get("body_clearance_saturated_veto")):
            wall_metrics["body_clearance_saturated_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_yaw_direction_veto")):
            wall_metrics["body_clearance_yaw_direction_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_yaw_contact_veto")):
            wall_metrics["body_clearance_yaw_contact_vetoes"] += 1
        if bool(wall_guard.get("body_clearance_current_contact_escape")):
            wall_metrics["body_clearance_current_contact_escapes"] += 1
            body_clearance_current_contact_escape_last_tick = int(tick)
        if bool(wall_guard.get("blocked_hard_veto")):
            wall_metrics["blocked_hard_vetoes"] += 1
        if bool(wall_guard.get("low_progress_hard_veto")):
            wall_metrics["low_progress_hard_vetoes"] += 1
        if log_entry is not None and bool(log_entry.get("body_clearance_request")):
            wall_metrics["body_clearance_target_interventions"] += 1
            if bool(log_entry.get("body_clearance_latched")):
                wall_metrics["body_clearance_latched_interventions"] += 1
        predicted_blocked_waypoint = False
        if (
            guard_state.upper() == "EXPLORE"
            and bool(args.wall_predicted_blocked_waypoint_replan)
            and requested_primitive in _FORWARD_PRIMITIVES
            and bool(wall_guard.get("requested_blocked"))
            and hasattr(explorer, "mark_current_waypoint_blocked")
        ):
            cell = getattr(explorer, "last_waypoint_cell", None)
            cell_key = tuple(cell) if cell is not None else None
            if cell_key is not None and cell_key == predicted_blocked_cell:
                predicted_blocked_streak += 1
            else:
                predicted_blocked_cell = cell_key
                predicted_blocked_streak = 1
            predicted_blocked_waypoint = (
                predicted_blocked_streak >= max(1, int(args.wall_predicted_blocked_waypoint_streak))
            )
            if predicted_blocked_waypoint and explorer.mark_current_waypoint_blocked():
                wall_metrics["learned_blocked_waypoint_replans"] += 1
                predicted_blocked_cell = None
                predicted_blocked_streak = 0
                if log_entry is not None:
                    log_entry["learned_blocked_waypoint_replan"] = True
        else:
            predicted_blocked_cell = None
            predicted_blocked_streak = 0
        if log_entry is not None and bool(args.wall_predicted_blocked_waypoint_replan):
            log_entry["learned_blocked_waypoint_streak"] = int(predicted_blocked_streak)
            log_entry["learned_blocked_waypoint_replan_ready"] = bool(predicted_blocked_waypoint)
        for key, guard_key in (
            ("requested_min_clearance_min_m", "requested_min_clearance_m"),
            ("selected_min_clearance_min_m", "selected_min_clearance_m"),
        ):
            val = wall_guard.get(guard_key)
            if val is not None:
                wall_metrics[key] = val if wall_metrics[key] is None else min(float(wall_metrics[key]), float(val))
        if (
            float(args.stability_guard_roll_pitch_threshold) > 0.0
            and args.mode == "physical"
        ):
            if stability_hold_remaining > 0:
                primitive = str(args.stability_guard_primitive)
                stability_hold_remaining -= 1
                wall_metrics["stability_guard_hold_ticks"] += 1
                if log_entry is not None:
                    log_entry["stability_guard_hold"] = True
            elif max(abs(float(prev_post_roll)), abs(float(prev_post_pitch))) >= float(
                args.stability_guard_roll_pitch_threshold
            ):
                # Proprioceptive capsize prevention: yawing while pressed
                # against a wall can lever the body over in a few ticks.
                # A wall-lean is statically stable, so holding cannot right
                # the body; the recovery primitive must step away from it.
                primitive = str(args.stability_guard_primitive)
                stability_hold_remaining = max(0, int(args.stability_guard_hold_ticks) - 1)
                wall_metrics["stability_guard_events"] += 1
                wall_metrics["stability_guard_hold_ticks"] += 1
                if log_entry is not None:
                    log_entry["stability_guard_hold"] = True
        if log_entry is not None:
            log_entry["requested_primitive"] = requested_primitive
            log_entry["primitive"] = primitive
            log_entry["wall_guard"] = wall_guard
            log.append(log_entry)

        if args.mode == "physical":
            _execute_physical_primitive(
                runner, registry, primitive,
                frame_sink=frames if capture else None, build=build, pack=pack, device=device,
                capture_policy_steps=(str(args.demo_capture_rate) == "policy"),
                third_person_build=third_person_build)
        else:
            _execute_kinematic_primitive(
                build, registry, primitive, command_dt_s=float(args.command_dt_s),
                grid=grid, frame_sink=frames if capture else None, pack=pack, device=device,
                third_person_build=third_person_build)
        post_pos, post_quat = _current_pose(build)
        xy_displacement = float(math.hypot(float(post_pos[0]) - float(pos[0]), float(post_pos[1]) - float(pos[1])))
        post_roll = _roll_from_quat_wxyz(post_quat)
        post_pitch = _pitch_from_quat_wxyz(post_quat)
        prev_post_roll = float(post_roll)
        prev_post_pitch = float(post_pitch)
        post_yaw = float(_yaw_from_quat_wxyz(post_quat))
        post_tip = max(abs(post_roll), abs(post_pitch))
        post_base_z = float(post_pos[2])
        post_body_clearance = _body_probe_clearance(
            grid,
            (float(post_pos[0]), float(post_pos[1])),
            post_yaw,
            body_forward_m=float(args.wall_body_forward_m),
            body_half_width_m=float(args.wall_body_half_width_m),
            body_probe_margin_m=float(args.wall_body_probe_margin_m),
        )
        translating_nonprogress = bool(
            str(primitive) in _TRANSLATING_PRIMITIVES
            and xy_displacement < float(args.wall_stall_displacement_m)
        )
        online_map_low_progress_block = bool(
            learned_local_online_map is not None
            and float(args.learned_local_online_map_low_progress_block_m) > 0.0
            and str(primitive) in _TRANSLATING_PRIMITIVES
            and xy_displacement
            < float(args.learned_local_online_map_low_progress_block_m)
        )
        rotation_stall_ticks = int(args.learned_local_online_map_rotation_stall_block_ticks)
        if rotation_stall_ticks > 0:
            yaw_rotation_delta = abs(float(wrap_angle_pi(post_yaw - float(yaw))))
            route_align_hard_vetoed = bool(
                log_entry is not None
                and isinstance(log_entry.get("learned_local_policy"), dict)
                and isinstance(
                    log_entry["learned_local_policy"].get("frontier_pressure"),
                    dict,
                )
                and str(
                    log_entry["learned_local_policy"]["frontier_pressure"].get(
                        "reason", ""
                    )
                )
                in {
                    "route_align_yaw",
                    "route_align_yaw_over_nonroute",
                    "post_claim_route_align_yaw",
                }
                and str(requested_primitive) in _TURN_PRIMITIVES
                and bool(wall_guard.get("body_clearance_hard_veto"))
                and str(
                    wall_guard.get("selected_before_body_clearance_hard_veto", "")
                )
                in _TURN_PRIMITIVES
                and str(primitive) != str(requested_primitive)
            )
            if (
                str(primitive) in _TURN_PRIMITIVES
                and yaw_rotation_delta < 0.02
            ) or route_align_hard_vetoed:
                learned_local_rotation_stall_streak += 1
            else:
                learned_local_rotation_stall_streak = 0
            if (
                learned_local_rotation_stall_streak >= rotation_stall_ticks
                and learned_local_online_map is not None
                and learned_local_last_route_next is not None
            ):
                learned_local_online_map.mark_rotation_blocked(
                    pos[:2], learned_local_last_route_next
                )
                wall_metrics["learned_local_online_map_rotation_stall_blocks"] += 1
                learned_local_rotation_stall_streak = 0
                if log_entry is not None:
                    log_entry["rotation_stall_block"] = True
                    log_entry["rotation_stall_block_hard_veto_route_align"] = bool(
                        route_align_hard_vetoed
                    )
        wall_metrics["body_clearance_min_m"] = (
            post_body_clearance
            if wall_metrics["body_clearance_min_m"] is None
            else min(float(wall_metrics["body_clearance_min_m"]), post_body_clearance)
        )
        body_clearance_contact = post_body_clearance <= float(
            wall_metrics["body_clearance_contact_threshold_m"]
        )
        if body_clearance_contact:
            wall_metrics["body_clearance_contact_events"] += 1
            if wall_metrics["first_body_clearance_contact_tick"] is None:
                wall_metrics["first_body_clearance_contact_tick"] = int(len(log) - 1)
        success_min_body_clearance_m = float(args.success_min_body_clearance_m)
        body_clearance_violation = (
            post_body_clearance < success_min_body_clearance_m
            if success_min_body_clearance_m > 0.0
            else body_clearance_contact
        )
        if body_clearance_violation:
            wall_metrics["body_clearance_violation_events"] += 1
            if wall_metrics["first_body_clearance_violation_tick"] is None:
                wall_metrics["first_body_clearance_violation_tick"] = int(len(log) - 1)
        if args.mode == "physical":
            wall_metrics["base_z_min_m"] = (
                post_base_z
                if wall_metrics["base_z_min_m"] is None
                else min(float(wall_metrics["base_z_min_m"]), post_base_z)
            )
            wall_metrics["max_abs_roll_pitch_rad"] = max(
                float(wall_metrics["max_abs_roll_pitch_rad"]),
                float(post_tip),
            )
            fall_event = post_base_z < float(args.fall_z_threshold_m)
            tip_event = post_tip > float(args.tip_threshold_rad)
            if fall_event:
                wall_metrics["fall_events"] += 1
            if tip_event:
                wall_metrics["tip_events"] += 1
            if fall_event or tip_event:
                wall_metrics["unstable_base_events"] += 1
                if wall_metrics["first_unstable_tick"] is None:
                    wall_metrics["first_unstable_tick"] = int(len(log) - 1)
                    if fall_event and tip_event:
                        wall_metrics["first_unstable_reason"] = "fall_and_tip"
                    elif fall_event:
                        wall_metrics["first_unstable_reason"] = "fall"
                    else:
                        wall_metrics["first_unstable_reason"] = "tip"
        else:
            fall_event = False
            tip_event = False
        if primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_execution_displacement_sum_m"] += xy_displacement
            stalled = xy_displacement < float(args.wall_stall_displacement_m)
            hard_stalled = xy_displacement < float(args.wall_hard_stall_displacement_m)
            if hard_stalled:
                wall_metrics["hard_contact_like_stalls"] += 1
            if stalled:
                wall_metrics["contact_like_stalls"] += 1
                for penalized in _stall_penalty_family(primitive):
                    stall_penalties[penalized] = max(
                        int(stall_penalties.get(penalized, 0)),
                        int(args.wall_stall_penalty_ticks),
                    )
                stuck_streak += 1
                if stuck_streak >= int(args.wall_stall_streak):
                    if len(escape_plan) < int(args.wall_escape_blocks):
                        escape_plan = _make_escape_plan(primitive, int(args.wall_escape_blocks))
                    wall_metrics["stuck_recoveries"] += 1
                    blocked_stall_waypoint = bool(
                        guard_state.upper() == "EXPLORE"
                        and bool(args.wall_stall_block_waypoint)
                        and hasattr(explorer, "mark_current_waypoint_blocked")
                        and explorer.mark_current_waypoint_blocked()
                    )
                    if blocked_stall_waypoint:
                        wall_metrics["stall_waypoint_blocks"] += 1
                    if log_entry is not None:
                        log_entry["stall_recovery"] = True
                        log_entry["stall_waypoint_blocked"] = blocked_stall_waypoint
                    stuck_streak = 0
            else:
                stuck_streak = 0
            turn_streak = 0
        else:
            stalled = False
            hard_stalled = False
            if primitive in _TURN_PRIMITIVES:
                turn_streak += 1
                if (
                    guard_enabled
                    and not escape_plan
                    and int(args.wall_turn_loop_streak) > 0
                    and turn_streak >= int(args.wall_turn_loop_streak)
                ):
                    escape_plan = _make_escape_plan(primitive, int(args.wall_turn_escape_blocks))
                    wall_metrics["turn_loop_recoveries"] += 1
                    blocked_turn_waypoint = bool(
                        guard_state.upper() == "EXPLORE"
                        and bool(args.wall_turn_loop_block_waypoint)
                        and hasattr(explorer, "mark_current_waypoint_blocked")
                        and explorer.mark_current_waypoint_blocked()
                    )
                    if blocked_turn_waypoint:
                        wall_metrics["turn_loop_waypoint_blocks"] += 1
                    turn_streak = 0
                    if hasattr(explorer, "path") and not blocked_turn_waypoint:
                        explorer.path = None
                    if log_entry is not None:
                        log_entry["turn_loop_recovery"] = True
                        log_entry["turn_loop_waypoint_blocked"] = blocked_turn_waypoint
            elif primitive == "backward":
                stuck_streak = 0
                turn_streak = 0
            else:
                turn_streak = 0
        if (
            log_entry is not None
            and isinstance(log_entry.get("learned_local_policy"), dict)
            and str(log_entry.get("state", "")).upper() == "EXPLORE"
        ):
            guard_blocked_translation_nonprogress = bool(
                str(primitive) == "hold"
                and str(requested_primitive) in _TRANSLATING_PRIMITIVES
                and (
                    bool(wall_guard.get("body_clearance_current_contact_escape"))
                    or bool(wall_guard.get("body_clearance_hard_veto"))
                    or bool(wall_guard.get("body_clearance_arc_sweep_veto"))
                    or bool(wall_guard.get("body_clearance_saturated_veto"))
                    or bool(wall_guard.get("blocked_hard_veto"))
                    or bool(wall_guard.get("low_progress_hard_veto"))
                )
            )
            if (
                str(primitive) in _TRANSLATING_PRIMITIVES
                and bool(translating_nonprogress or online_map_low_progress_block)
            ) or guard_blocked_translation_nonprogress:
                learned_local_policy_nonprogress_run = min(
                    64,
                    int(learned_local_policy_nonprogress_run) + 1,
                )
                wall_metrics["learned_local_policy_nonprogress_ticks"] += 1
                penalized_primitive = (
                    str(requested_primitive)
                    if guard_blocked_translation_nonprogress
                    else str(primitive)
                )
                for penalized in _stall_penalty_family(penalized_primitive):
                    stall_penalties[penalized] = max(
                        int(stall_penalties.get(penalized, 0)),
                        int(args.wall_stall_penalty_ticks),
                    )
            elif str(primitive) in _TRANSLATING_PRIMITIVES or str(primitive) not in _TURN_PRIMITIVES:
                learned_local_policy_nonprogress_run = 0
            wall_metrics["learned_local_policy_max_nonprogress_run"] = max(
                int(wall_metrics["learned_local_policy_max_nonprogress_run"]),
                int(learned_local_policy_nonprogress_run),
            )
            log_entry["learned_local_policy_nonprogress_run_after"] = int(
                learned_local_policy_nonprogress_run
            )
            log_entry["learned_local_policy_translating_nonprogress"] = bool(
                translating_nonprogress
            )
            log_entry[
                "learned_local_policy_guard_blocked_translation_nonprogress"
            ] = bool(guard_blocked_translation_nonprogress)
            log_entry["learned_local_online_map_low_progress_block"] = bool(
                online_map_low_progress_block
            )
            if bool(online_map_low_progress_block):
                log_entry["learned_local_online_map_low_progress_block_m"] = _round_float(
                    args.learned_local_online_map_low_progress_block_m,
                    4,
                )
        if learned_local_online_map is not None:
            if bool(online_map_low_progress_block):
                wall_metrics["learned_local_online_map_low_progress_block_ticks"] += 1
            learned_local_online_map.update_after_action(
                pose_xy=pos[:2],
                post_xy=post_pos[:2],
                yaw_rad=float(yaw),
                primitive=str(primitive),
                stalled=bool(
                    stalled
                    or hard_stalled
                    or translating_nonprogress
                    or online_map_low_progress_block
                ),
                tick=int(tick),
            )
        for penalized in list(stall_penalties):
            stall_penalties[penalized] -= 1
            if stall_penalties[penalized] <= 0:
                del stall_penalties[penalized]
        if (
            log_entry is not None
            and bool(weak_memory_recovery_active)
            and str(log_entry.get("state", "")).upper() == "SEEK"
        ):
            yaw_loop_enabled = int(args.weak_memory_seek_yaw_loop_streak) > 0
            yaw_loop_tick = bool(
                yaw_loop_enabled
                and str(log_entry.get("primitive", "")) in _TURN_PRIMITIVES
                and float(xy_displacement) <= float(args.weak_memory_seek_yaw_loop_max_displacement_m)
            )
            if yaw_loop_tick:
                weak_memory_seek_yaw_loop_streak += 1
                wall_metrics["weak_memory_seek_yaw_loop_events"] += 1
                log_entry["weak_memory_seek_yaw_loop_streak_after"] = int(
                    weak_memory_seek_yaw_loop_streak
                )
                if (
                    int(weak_memory_seek_yaw_loop_streak)
                    >= max(1, int(args.weak_memory_seek_yaw_loop_streak))
                    and int(weak_memory_seek_explore_cooldown) <= 0
                ):
                    weak_memory_seek_explore_cooldown = max(
                        1,
                        int(args.weak_memory_seek_explore_cooldown_ticks),
                    )
                    weak_memory_seek_yaw_loop_streak = 0
                    wall_metrics["weak_memory_seek_yaw_loop_recoveries"] += 1
                    wall_metrics["weak_memory_seek_recoveries"] += 1
                    log_entry["weak_memory_seek_yaw_loop_recovery_armed"] = True
                    log_entry["weak_memory_seek_recovery_cooldown"] = int(
                        weak_memory_seek_explore_cooldown
                    )
            else:
                weak_memory_seek_yaw_loop_streak = 0
            if bool(stalled or hard_stalled):
                weak_memory_seek_stall_streak += 1
                wall_metrics["weak_memory_seek_stall_events"] += 1
                log_entry["weak_memory_seek_stall_streak_after"] = int(weak_memory_seek_stall_streak)
                if (
                    int(weak_memory_seek_stall_streak)
                    >= max(1, int(args.weak_memory_seek_stall_streak))
                    and int(weak_memory_seek_explore_cooldown) <= 0
                ):
                    weak_memory_seek_explore_cooldown = max(
                        1,
                        int(args.weak_memory_seek_explore_cooldown_ticks),
                    )
                    weak_memory_seek_stall_streak = 0
                    wall_metrics["weak_memory_seek_recoveries"] += 1
                    log_entry["weak_memory_seek_recovery_armed"] = True
                    log_entry["weak_memory_seek_recovery_cooldown"] = int(
                        weak_memory_seek_explore_cooldown
                    )
            else:
                weak_memory_seek_stall_streak = 0
        elif log_entry is not None and not bool(weak_memory_force_explore):
            weak_memory_seek_stall_streak = 0
            weak_memory_seek_yaw_loop_streak = 0
        if log_entry is not None and int(args.target_pursuit_stale_ticks) > 0:
            target_state = str(log_entry.get("state", "")).upper()
            target_color_for_stale = str(log_entry.get("target_color", ""))
            if target_color_for_stale != target_pursuit_stale_last_color:
                target_pursuit_stale_streak = 0
                target_pursuit_stale_last_color = target_color_for_stale
            target_gate = log_entry.get("claim_gate")
            claim_gate_passed = bool(
                isinstance(target_gate, dict) and target_gate.get("accepted")
            )
            claim_proxy_pending = False
            if isinstance(target_gate, dict):
                proxy_gate = target_gate.get("success_proxy")
                base_claim_ready = any(
                    bool(
                        isinstance(target_gate.get(section), dict)
                        and target_gate.get(section, {}).get("passed")
                    )
                    for section in ("standard", "near", "contact", "stalled_visual")
                )
                claim_proxy_pending = bool(
                    base_claim_ready
                    and isinstance(proxy_gate, dict)
                    and bool(proxy_gate.get("enabled"))
                    and not bool(proxy_gate.get("passed"))
                )
            if claim_proxy_pending:
                wall_metrics["target_pursuit_stale_claim_proxy_pending_ticks"] += 1
                log_entry["target_pursuit_stale_claim_proxy_pending"] = True
            stale_candidate = bool(
                target_state in target_pursuit_stale_states
                and bool(log_entry.get("seen", False))
                and not claim_gate_passed
                and not claim_proxy_pending
            )
            stale_window_ticks = max(0, int(args.target_pursuit_stale_window_ticks))
            stale_window_count = 0
            if stale_window_ticks > 0 and target_color_for_stale:
                cutoff_tick = int(tick) - stale_window_ticks + 1
                color_window = [
                    int(item)
                    for item in target_pursuit_stale_window_ticks_by_color.get(
                        target_color_for_stale,
                        [],
                    )
                    if int(item) >= cutoff_tick
                ]
                target_pursuit_stale_window_ticks_by_color[
                    target_color_for_stale
                ] = color_window
                stale_window_count = len(color_window)
            if stale_candidate:
                target_pursuit_stale_streak = min(
                    1000000,
                    int(target_pursuit_stale_streak) + 1,
                )
                wall_metrics["target_pursuit_stale_candidate_ticks"] += 1
                if stale_window_ticks > 0 and target_color_for_stale:
                    color_window = target_pursuit_stale_window_ticks_by_color.setdefault(
                        target_color_for_stale,
                        [],
                    )
                    color_window.append(int(tick))
                    cutoff_tick = int(tick) - stale_window_ticks + 1
                    color_window[:] = [
                        int(item) for item in color_window if int(item) >= cutoff_tick
                    ]
                    stale_window_count = len(color_window)
                    wall_metrics[
                        "target_pursuit_stale_window_candidate_ticks_max"
                    ] = max(
                        int(wall_metrics["target_pursuit_stale_window_candidate_ticks_max"]),
                        int(stale_window_count),
                    )
                    log_entry["target_pursuit_stale_window_candidate_ticks"] = int(
                        stale_window_count
                    )
                    log_entry["target_pursuit_stale_window_ticks"] = int(
                        stale_window_ticks
                    )
                log_entry["target_pursuit_stale_streak_after"] = int(
                    target_pursuit_stale_streak
                )
                stale_recovery_reason = None
                if int(target_pursuit_stale_streak) >= max(
                    1,
                    int(args.target_pursuit_stale_ticks),
                ):
                    stale_recovery_reason = "consecutive"
                elif (
                    stale_window_ticks > 0
                    and int(stale_window_count) >= max(
                        1,
                        int(args.target_pursuit_stale_ticks),
                    )
                ):
                    stale_recovery_reason = "window"
                if (
                    stale_recovery_reason is not None
                    and int(target_pursuit_escape_cooldown) <= 0
                ):
                    target_pursuit_escape_cooldown = max(
                        1,
                        int(args.target_pursuit_stale_explore_cooldown_ticks),
                    )
                    target_pursuit_stale_streak = 0
                    if stale_window_ticks > 0 and target_color_for_stale:
                        target_pursuit_stale_window_ticks_by_color[target_color_for_stale] = []
                    wall_metrics["target_pursuit_stale_recoveries"] += 1
                    if stale_recovery_reason == "window":
                        wall_metrics["target_pursuit_stale_window_recoveries"] += 1
                    suppress_ticks = max(
                        0,
                        int(args.target_pursuit_stale_suppress_color_ticks),
                    )
                    if suppress_ticks > 0 and target_color_for_stale:
                        suppress_until = int(tick) + suppress_ticks
                        target_pursuit_suppressed_until[target_color_for_stale] = max(
                            int(
                                target_pursuit_suppressed_until.get(
                                    target_color_for_stale,
                                    0,
                                )
                            ),
                            int(suppress_until),
                        )
                        wall_metrics["target_pursuit_stale_color_suppressions"] += 1
                        log_entry["target_pursuit_stale_color_suppression"] = {
                            "color": target_color_for_stale,
                            "until_tick": int(
                                target_pursuit_suppressed_until[
                                    target_color_for_stale
                                ]
                            ),
                            "ticks": int(suppress_ticks),
                        }
                    log_entry["target_pursuit_stale_recovery_armed"] = True
                    log_entry["target_pursuit_stale_recovery_reason"] = str(
                        stale_recovery_reason
                    )
                    log_entry["target_pursuit_stale_recovery_cooldown"] = int(
                        target_pursuit_escape_cooldown
                    )
            elif not bool(target_pursuit_force_explore):
                target_pursuit_stale_streak = 0
        elif log_entry is not None and not bool(target_pursuit_force_explore):
            target_pursuit_stale_streak = 0
        if body_clearance_risk_escape_cooldown > 0:
            body_clearance_risk_escape_cooldown -= 1
        if current_body_risk_cooldown > 0:
            current_body_risk_cooldown -= 1
        if history_risk_wedge_cooldown > 0:
            history_risk_wedge_cooldown -= 1
        stalled_prev_tick = bool(stalled)
        if weak_memory_seek_explore_cooldown > 0:
            weak_memory_seek_explore_cooldown -= 1
        if target_pursuit_escape_cooldown > 0:
            target_pursuit_escape_cooldown -= 1
        if log_entry is not None:
            log_entry["executed_displacement_m"] = _round_float(xy_displacement)
            log_entry["post_xy"] = [_round_float(float(post_pos[0])), _round_float(float(post_pos[1]))]
            log_entry["post_z"] = _round_float(post_base_z, 4)
            log_entry["post_yaw"] = _round_float(post_yaw)
            log_entry["post_roll"] = _round_float(post_roll, 4)
            log_entry["post_pitch"] = _round_float(post_pitch, 4)
            log_entry["post_tip_rad"] = _round_float(post_tip, 4)
            log_entry["post_body_clearance_m"] = _round_float(post_body_clearance, 4)
            log_entry["body_clearance_contact"] = bool(body_clearance_contact)
            log_entry["stalled"] = bool(stalled)
            log_entry["hard_stalled"] = bool(hard_stalled)
            log_entry["fall_event"] = bool(fall_event)
            log_entry["tip_event"] = bool(tip_event)
            log_entry["unstable_base"] = bool(fall_event or tip_event)
            log_entry["body_clearance_violation"] = bool(body_clearance_violation)
            if current_body_risk_prob is not None:
                log_entry["current_body_risk_prob"] = _round_float(current_body_risk_prob, 4)
            if history_risk_probs is not None:
                log_entry["history_risk_probs"] = {
                    name: _round_float(prob, 4)
                    for name, prob in history_risk_probs.items()
                }
            if active_stall_penalties:
                log_entry["stall_penalties"] = {
                    k: _round_float(v, 3) for k, v in sorted(active_stall_penalties.items())
                }
        if (
            proprio_contact_detector is not None
            or history_risk_model is not None
            or broad_explorer_model is not None
        ):
            proprio_tick_values = (xy_displacement, post_yaw, post_roll, post_pitch, post_base_z, post_tip)
            if any(value is None for value in proprio_tick_values):
                proprio_contact_feature_buffer.clear()
                proprio_contact_prob_history.clear()
                proprio_contact_prev_yaw = None
                proprio_contact_prev_z = None
                history_risk_pending_proprio = None
                history_risk_rows.clear()
            else:
                proprio_tick_feature_vec = _proprio_contact_tick_features(
                    displacement_m=float(xy_displacement),
                    post_yaw=float(post_yaw),
                    post_roll=float(post_roll),
                    post_pitch=float(post_pitch),
                    post_z=float(post_base_z),
                    post_tip_rad=float(post_tip),
                    primitive=str(primitive),
                    requested_primitive=(
                        None if requested_primitive is None else str(requested_primitive)
                    ),
                    prev_yaw=proprio_contact_prev_yaw,
                    prev_z=proprio_contact_prev_z,
                )
                proprio_contact_feature_buffer.append(proprio_tick_feature_vec)
                if (
                history_risk_model is not None
                or broad_explorer_model is not None
                or visual_ray_model is not None
            ):
                    history_risk_pending_proprio = proprio_tick_feature_vec
                proprio_contact_prev_yaw = round(float(post_yaw), 3)
                proprio_contact_prev_z = round(float(post_base_z), 4)
                if len(proprio_contact_feature_buffer) > max(1, int(proprio_contact_window)):
                    proprio_contact_feature_buffer.pop(0)
        if proprio_contact_detector is not None:
            proprio_contact_prob: float | None = None
            if len(proprio_contact_feature_buffer) == int(proprio_contact_window):
                proprio_window_arr = (
                    np.stack(proprio_contact_feature_buffer) - proprio_contact_feature_mean
                ) / proprio_contact_feature_std
                with torch.no_grad():
                    proprio_contact_prob = float(
                        torch.sigmoid(
                            proprio_contact_detector(
                                torch.from_numpy(proprio_window_arr).float().unsqueeze(0).to(device)
                            )
                        )[0].cpu().item()
                    )
                proprio_contact_prob_history.append(float(proprio_contact_prob))
                if len(proprio_contact_prob_history) > 3:
                    proprio_contact_prob_history.pop(0)
            if proprio_contact_cooldown > 0:
                proprio_contact_cooldown -= 1
            proprio_contact_rolling_prob = (
                float(np.mean(proprio_contact_prob_history))
                if proprio_contact_prob_history
                else None
            )
            if (
                proprio_contact_rolling_prob is not None
                and proprio_contact_rolling_prob >= float(args.proprio_contact_escape_threshold)
            ):
                proprio_contact_streak += 1
            else:
                proprio_contact_streak = 0
            if proprio_contact_prob is not None:
                wall_metrics["proprio_contact_detector_ticks"] += 1
                rolling_rounded = _round_float(proprio_contact_rolling_prob, 4)
                wall_metrics["proprio_contact_prob_max"] = (
                    rolling_rounded
                    if wall_metrics["proprio_contact_prob_max"] is None
                    else max(float(wall_metrics["proprio_contact_prob_max"]), float(rolling_rounded))
                )
                if log_entry is not None:
                    log_entry["proprio_contact_prob"] = _round_float(proprio_contact_prob, 4)
            proprio_tick_state = (
                "" if log_entry is None else str(log_entry.get("state", "")).upper()
            )
            if (
                int(args.proprio_contact_escape_blocks) > 0
                and proprio_contact_streak >= int(args.proprio_contact_escape_streak)
                and int(proprio_contact_cooldown) <= 0
                and not escape_plan
                and (
                    not proprio_contact_escape_states
                    or proprio_tick_state in proprio_contact_escape_states
                )
            ):
                proprio_risky_primitive = primitive if primitive != "hold" else last_primitive
                proprio_escape_plan = _make_escape_plan(
                    proprio_risky_primitive,
                    int(args.proprio_contact_escape_blocks),
                )
                if history_risk_probs is not None:
                    escape_candidates = sorted(
                        ("backward", "yaw_left", "yaw_right"),
                        key=lambda name: float(history_risk_probs.get(name, 1.0)),
                    )
                    safest = escape_candidates[0]
                    if float(history_risk_probs.get(safest, 1.0)) < float(
                        history_risk_probs.get(proprio_escape_plan[0], 1.0)
                    ):
                        proprio_escape_plan[0] = safest
                escape_plan = proprio_escape_plan + escape_plan
                proprio_contact_cooldown = int(args.proprio_contact_escape_cooldown_ticks)
                proprio_contact_streak = 0
                wall_metrics["proprio_contact_escapes"] += 1
                proprio_marked_edges = 0
                if (
                    bool(args.proprio_contact_map_blocks)
                    and learned_local_online_map is not None
                    and str(primitive) in _TRANSLATING_PRIMITIVES
                ):
                    if learned_local_online_map.mark_blocked_primitive(
                        pos[:2],
                        float(yaw),
                        str(primitive),
                    ):
                        proprio_marked_edges = 1
                        wall_metrics["proprio_contact_map_blocked_edges"] += 1
                if log_entry is not None:
                    log_entry["proprio_contact_escape"] = {
                        "prob": _round_float(proprio_contact_rolling_prob, 4),
                        "from": str(proprio_risky_primitive),
                        "plan": list(proprio_escape_plan),
                        "map_blocked_edges": int(proprio_marked_edges),
                    }
        if bool(args.body_clearance_target_servo):
            observed_near_target = bool(
                log_entry is not None
                and bool(log_entry.get("in_cone"))
                and float(log_entry.get("area", -99.0)) >= float(args.body_clearance_target_area_logit)
            )
            if observed_near_target:
                body_clearance_latch = max(0, int(args.body_clearance_latch_ticks))
            elif body_clearance_latch > 0:
                body_clearance_latch -= 1
            if log_entry is not None:
                log_entry["body_clearance_latch_remaining"] = int(body_clearance_latch)
        if args.claim_stalled_area_logit is not None:
            selected_clearance_prob_for_claim = wall_guard.get("selected_clearance_blocked_prob")
            stalled_visual_clearance_ok = bool(
                float(args.claim_stalled_clearance_prob) < 0.0
                or (
                    selected_clearance_prob_for_claim is not None
                    and float(selected_clearance_prob_for_claim) >= float(args.claim_stalled_clearance_prob)
                )
            )
            stalled_visual_arm = bool(
                log_entry is not None
                and bool(stalled)
                and str(log_entry.get("state", "")).upper() in ("SERVO", "SEEK")
                and bool(log_entry.get("in_cone"))
                and abs(float(log_entry.get("bearing", 999.0))) < float(args.claim_stalled_bearing)
                and stalled_visual_clearance_ok
            )
            if stalled_visual_arm:
                claim_stalled_visual_latch = max(0, int(args.claim_stalled_latch_ticks))
                wall_metrics["claim_stalled_visual_armed_ticks"] += 1
            elif claim_stalled_visual_latch > 0:
                claim_stalled_visual_latch -= 1
            if log_entry is not None:
                log_entry["claim_stalled_visual_arm"] = bool(stalled_visual_arm)
                log_entry["claim_stalled_visual_latch_remaining"] = int(claim_stalled_visual_latch)
        last_primitive_run_ticks = (
            int(last_primitive_run_ticks) + 1
            if primitive == last_primitive
            else 1
        )
        last_primitive = primitive
        last_cmd = _PRIM_CMD.get(primitive, (0.0, 0.0, 0.0))
        if (
            args.slice_snapshot_output is not None
            and int(args.slice_snapshot_at_tick) >= 0
            and int(tick) >= int(args.slice_snapshot_at_tick)
            and not slice_snapshot_saved
        ):
            _write_slice_snapshot(
                args.slice_snapshot_output,
                build=build,
                runner=runner,
                scene_id=scene_dir.name,
                tick=int(tick),
                next_tick=int(tick) + 1,
                pos=np.asarray(post_pos, dtype=np.float32),
                quat=np.asarray(post_quat, dtype=np.float32),
                yaw=float(post_yaw),
                ctrl_state=ctrl_state,
                target_sequence=target_sequence,
                target_index=int(target_index),
                target_active_since_tick=int(target_active_since_tick),
                beacon_claims=beacon_claims,
                first_seen_ticks=first_seen_ticks,
                last_seen_ticks=last_seen_ticks,
                last_primitive=last_primitive,
                last_cmd=last_cmd,
                online_map=learned_local_online_map,
                feature_max_ticks=int(feature_max_ticks),
                source_result=str(args.output),
            )
            slice_snapshot_saved = True
            wall_metrics["slice_snapshot_output"] = str(args.slice_snapshot_output)
            wall_metrics["slice_snapshot_tick"] = int(tick)
            if bool(args.slice_snapshot_exit):
                break
        prev_pose = cur_pose

    final_pos, _ = _current_pose(build)
    final_xy = np.asarray([float(final_pos[0]), float(final_pos[1])])
    final_target_color = target_sequence[min(target_index, len(target_sequence) - 1)]
    dist = (
        float(np.linalg.norm(final_xy - landmarks[final_target_color]))
        if final_target_color in landmarks
        else None
    )
    claimed_colors = [str(item.get("target_color")) for item in beacon_claims]
    all_target_colors_claimed = all(color in claimed_colors for color in target_sequence)
    claim_distances = {
        str(item.get("target_color")): item.get("dist_to_target_m")
        for item in beacon_claims
    }
    multi_target_claim_distance_success = bool(
        len(target_sequence) <= 1
        or all(
            claim_distances.get(color) is not None
            and float(claim_distances[color]) <= float(args.success_dist_m)
            for color in target_sequence
        )
    )
    wall_metrics["multi_target_claim_distance_success"] = bool(
        multi_target_claim_distance_success
    )
    stable_base_success = bool(
        args.mode != "physical"
        or args.allow_unstable_base_success
        or int(wall_metrics["unstable_base_events"]) == 0
    )
    body_clearance_success = bool(
        args.allow_body_clearance_violation_success
        or int(wall_metrics["body_clearance_violation_events"]) == 0
    )
    if len(target_sequence) > 1:
        success = bool(
            claimed
            and all_target_colors_claimed
            and stable_base_success
            and body_clearance_success
            and (
                not bool(args.multi_target_success_requires_claim_distance)
                or multi_target_claim_distance_success
            )
        )
    else:
        success = bool(
            claimed
            and stable_base_success
            and body_clearance_success
            and dist is not None
            and dist <= float(args.success_dist_m)
        )
    if wall_metrics["forward_executions"]:
        wall_metrics["mean_forward_execution_displacement_m"] = (
            wall_metrics["forward_execution_displacement_sum_m"] / wall_metrics["forward_executions"]
        )
    else:
        wall_metrics["mean_forward_execution_displacement_m"] = None
    wall_metrics["explorer_replans"] = int(getattr(explorer, "replans", 0))
    wall_metrics["explorer_visited_cells"] = int(len(getattr(explorer, "visited", [])))
    wall_metrics["explorer_blocked_cells"] = int(len(getattr(explorer, "blocked", [])))
    if learned_local_online_map is not None:
        online_map_summary = learned_local_online_map.summary()
        wall_metrics["learned_local_online_map_hard_guard_blocks"] = bool(
            online_map_summary["hard_guard_blocks"]
        )
        wall_metrics["learned_local_online_map_visited_cells"] = int(
            online_map_summary["visited_cells"]
        )
        wall_metrics["learned_local_online_map_blocked_cells"] = int(
            online_map_summary["blocked_cells"]
        )
        wall_metrics["learned_local_online_map_claimed_cells"] = int(
            online_map_summary["claimed_cells"]
        )
        wall_metrics["learned_local_online_map_attempted_edges"] = int(
            online_map_summary["attempted_edges"]
        )
        wall_metrics["learned_local_online_map_blocked_edges"] = int(
            online_map_summary["blocked_edges"]
        )
        wall_metrics["learned_local_online_map_guard_blocked_cells"] = int(
            online_map_summary["guard_blocked_cells"]
        )
        wall_metrics["learned_local_online_map_guard_blocked_edges"] = int(
            online_map_summary["guard_blocked_edges"]
        )
    wall_metrics["explore_route_final_idx"] = int(getattr(explorer, "route_idx", 0))
    wall_metrics["explore_route_claim_count"] = int(getattr(explorer, "route_claim_count", 0))
    wall_metrics["learned_topology_route_final_idx"] = int(
        learned_topology_route_state.get("idx", 0)
    )
    wall_metrics["learned_topology_route_resets"] = int(
        learned_topology_route_state.get("resets", 0)
    )
    wall_metrics["explore_standoff_replans"] = int(getattr(explorer, "standoff_replans", 0))
    wall_metrics["explore_standoff_plan_failures"] = int(getattr(explorer, "standoff_plan_failures", 0))
    if oracle_standoff_explorer is not None:
        wall_metrics["learned_local_oracle_standoff_replans"] = int(
            getattr(oracle_standoff_explorer, "standoff_replans", 0)
        )
        wall_metrics["learned_local_oracle_standoff_plan_failures"] = int(
            getattr(oracle_standoff_explorer, "standoff_plan_failures", 0)
        )
    wall_metrics["explore_standoff_final_target_color"] = getattr(explorer, "standoff_target_color", None)
    wall_metrics["explore_standoff_final_path_idx"] = int(getattr(explorer, "standoff_path_idx", 0))
    wall_metrics["explore_standoff_final_path_len"] = int(len(getattr(explorer, "standoff_path", ())))
    wall_metrics["explore_standoff_corner_guard_caps"] = int(getattr(explorer, "standoff_corner_guard_caps", 0))
    wall_metrics["explore_standoff_corner_standoff_caps"] = int(
        getattr(explorer, "standoff_corner_standoff_caps", 0)
    )
    wall_metrics["explore_standoff_tangent_heading_ticks"] = int(
        getattr(explorer, "standoff_tangent_heading_ticks", 0)
    )
    wall_metrics["explore_standoff_snap_start_prefixes"] = int(
        getattr(explorer, "standoff_snap_start_prefixes", 0)
    )
    wall_metrics["explore_standoff_blocked_cells"] = int(
        len(getattr(explorer, "standoff_blocked_cells", ()))
    )
    wall_metrics["explore_standoff_blocked_waypoints"] = int(
        getattr(explorer, "standoff_blocked_waypoints", 0)
    )
    wall_metrics["explore_standoff_body_vetoes"] = int(getattr(explorer, "standoff_body_vetoes", 0))
    wall_metrics["explore_standoff_intent_target_holds"] = int(
        getattr(explorer, "standoff_intent_target_holds", 0)
    )
    wall_metrics["explore_standoff_intent_target_releases"] = int(
        getattr(explorer, "standoff_intent_target_releases", 0)
    )
    wall_metrics["explore_standoff_yaw_holds"] = int(
        getattr(explorer, "standoff_yaw_holds", 0)
    )
    wall_metrics["explore_standoff_yaw_flip_suppressions"] = int(
        getattr(explorer, "standoff_yaw_flip_suppressions", 0)
    )
    wall_metrics["explore_standoff_yaw_exits"] = int(
        getattr(explorer, "standoff_yaw_exits", 0)
    )
    wall_metrics["explore_standoff_arc_bearing_suppressions"] = int(
        getattr(explorer, "standoff_arc_bearing_suppressions", 0)
    )
    wall_metrics["explore_standoff_arc_target_dist_suppressions"] = int(
        getattr(explorer, "standoff_arc_target_dist_suppressions", 0)
    )
    wall_metrics["explore_standoff_body_current_clearance_m"] = _round_float(
        getattr(explorer, "standoff_body_current_clearance_m", None), 4
    )
    wall_metrics["explore_standoff_body_forward_clearance_m"] = _round_float(
        getattr(explorer, "standoff_body_forward_clearance_m", None), 4
    )
    wall_metrics["post_claim_acquisition_diagnostics"] = (
        _post_claim_acquisition_diagnostics(
            log,
            target_sequence=target_sequence,
            min_claims=max(1, min(3, len(target_sequence))),
        )
        if len(target_sequence) > 1
        else {}
    )
    wall_metrics.pop("forward_execution_displacement_sum_m", None)
    result = {
        "scene": scene_dir.name, "policy": args.policy,
        "execution_mode": str(args.mode),
        "gait_executed": bool(args.mode == "physical"),
        "demo_capture_rate": str(args.demo_capture_rate),
        "target_color": target_sequence[0] if len(target_sequence) == 1 else "all",
        "target_colors": target_sequence,
        "ticks_used": len(log), "claimed": claimed,
        "claimed_colors": claimed_colors,
        "first_seen_tick": first_seen_ticks.get(target_sequence[0]),
        "first_seen_ticks": first_seen_ticks,
        "beacon_claims": beacon_claims,
        "final_xy": final_xy.tolist(),
        "target_xy": (
            landmarks.get(target_sequence[0], np.zeros(2)).tolist()
            if len(target_sequence) == 1
            else {color: landmarks[color].tolist() for color in target_sequence}
        ),
        "final_dist_to_target_m": dist,
        "claim_distances_m": claim_distances,
        "success": success, "wall_metrics": wall_metrics,
    }
    if args.learned_local_dataset_output is not None:
        args.learned_local_dataset_output.parent.mkdir(parents=True, exist_ok=True)
        if learned_local_dataset_features:
            feature_array = np.stack(learned_local_dataset_features).astype(np.float32)
        else:
            feature_array = np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(
            args.learned_local_dataset_output,
            schema=np.asarray(["lewm_go2_closed_loop_learned_local_policy_dataset_v0"]),
            features=feature_array,
            labels=np.asarray(learned_local_dataset_labels, dtype=np.int64),
            primitive_vocab=np.asarray(_LEARNED_LOCAL_POLICY_PRIMITIVES),
            meta_json=np.asarray([json.dumps(item, sort_keys=True) for item in learned_local_dataset_meta]),
            result_json=np.asarray([json.dumps(result, sort_keys=True)]),
        )
        print(
            f"wrote learned-local dataset {args.learned_local_dataset_output} "
            f"examples={feature_array.shape[0]} feature_dim={feature_array.shape[1] if feature_array.ndim == 2 else 0}",
            flush=True,
        )
    print(json.dumps(result, indent=2))
    if args.policy == "memory":
        print("STATES:", " ".join(f"{e['tick']}:{e['state'][0]}" for e in log[-40:]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "provenance": {
                        "argv": list(sys.argv),
                        "output": str(args.output),
                    },
                    "result": result,
                    "log": log,
                },
                indent=2,
            )
        )
    if capture and frames:
        import imageio
        args.demo_video.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for third_np, ego_np, *_ in frames:
            ego_up = np.asarray(F.interpolate(
                torch.from_numpy(ego_np).permute(2, 0, 1)[None].float(),
                size=(third_np.shape[0], third_np.shape[1]), mode="nearest")[0].permute(1, 2, 0).byte())
            out.append(np.concatenate([third_np, ego_up], axis=1))
        if args.demo_fps is not None:
            demo_fps = float(args.demo_fps)
        elif args.mode == "physical" and str(args.demo_capture_rate) == "policy":
            demo_fps = 1.0 / float(pack.timing.policy_dt_s)
        else:
            demo_fps = 1.0 / float(args.command_dt_s)
        imageio.mimwrite(str(args.demo_video), out, fps=demo_fps, macro_block_size=8)
        print(f"wrote {args.demo_video} ({len(out)} frames @ {demo_fps:.2f} fps)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
