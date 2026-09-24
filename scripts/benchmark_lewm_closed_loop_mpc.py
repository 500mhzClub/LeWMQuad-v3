#!/usr/bin/env python3
"""Closed-loop Genesis MPC benchmark for flat LeWM checkpoints.

This is the paper-style planning-readiness metric for this repo: use the
checkpoint as a latent cost model inside receding-horizon control, execute only
the first primitive block, reobserve, and replan. The output is JSON with
physical success/progress metrics, not latent MSE.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# The DINO ceiling needs the ROCm torch build and Genesis in one process.  The
# ROCm environment intentionally does not duplicate the renderer's dependencies;
# append those only after torch is loaded so its CPU-only torch cannot shadow the
# already imported ROCm package.
_EARLY_REPO_ROOT = Path(__file__).resolve().parents[1]
_RENDER_SITE_PACKAGES = (
    _EARLY_REPO_ROOT
    / ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages"
)
if (
    (importlib.util.find_spec("genesis") is None or importlib.util.find_spec("tqdm") is None)
    and _RENDER_SITE_PACKAGES.is_dir()
    and str(_RENDER_SITE_PACKAGES) not in sys.path
):
    sys.path.append(str(_RENDER_SITE_PACKAGES))

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_HOME = REPO_ROOT / ".generated" / "benchmark_home"
_CACHE_ROOT = REPO_ROOT / ".generated" / "cache"
_BENCH_HOME.mkdir(parents=True, exist_ok=True)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
if not os.access(Path.home() / ".cache", os.W_OK):
    os.environ["HOME"] = str(_BENCH_HOME)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TI_CACHE_HOME", str(_CACHE_ROOT / "taichi"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".generated" / "mplconfig"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# The local render venv was created without --system-site-packages, but Ubuntu's
# PyYAML is available here and is enough for manifest loading. Import only yaml
# from there: leaving the whole dist-packages path on sys.path exposes Ubuntu's
# coverage package to Numba and can break Genesis import.
try:
    import yaml as _yaml  # noqa: F401
except ModuleNotFoundError:
    _system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if _system_dist_packages.is_dir():
        sys.path.append(str(_system_dist_packages))
        import yaml as _yaml  # noqa: F401
        sys.path.remove(str(_system_dist_packages))

from lewm.actions import ACTIVE_BLOCK_DIM, active_block_to_matrix  # noqa: E402
from lewm_genesis.camera_safety import camera_safety_config_from_pack, safe_camera_pose_from_base  # noqa: E402
from lewm_genesis.collectors.base import primitive_toward_bearing, wrap_angle_pi  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits, expand_primitive_to_block  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    effective_camera_mount_xyz_rpy,
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from lewm.models.energy_head import GoalEnergyHead  # noqa: E402
from lewm.models.pose_head import RelPoseHead  # noqa: E402
# Stage 0 refactor (v3 §4.1): the planner now lives in lewm.planning / lewm.memory.
# The wrappers below delegate to it so this benchmark is behaviour-locked
# (see lewm/tests/test_planning_refactor.py).
from lewm.planning.primitive_bank import (  # noqa: E402
    active_blocks as _pb_active_blocks,
    candidate_action_tensor as _pb_candidate_action_tensor,
)
from lewm.planning.local_mpc import (  # noqa: E402
    choose_primitive as _lmpc_choose_primitive,
    primitive_costs as _lmpc_primitive_costs,
)


ORACLE_ASSAY_NAME = "privileged_kinematic_endpoint_distance"
ORACLE_ASSAY_VERSION = 1
ORACLE_COST_TIE_TOLERANCE_M = 1e-9
ORACLE_TIE_BREAK = "lowest_candidate_index_within_tolerance"
ORACLE_SHUFFLE_NAME = "deterministic_candidate_score_permutation"
ORACLE_SHUFFLE_VERSION = 1
ORACLE_SHUFFLE_MIX_CONSTANT = 0x9E3779B97F4A7C15
PLANNING_GRID_CELL_SIZE_M = 0.05
PLANNING_GRID_INFLATION_M = 0.20
DINO_ASSAY_NAME = "frozen_dinov2_true_successor_goal_cost"
DINO_ASSAY_VERSION = 1
DINO_ENCODER_NAME = "dinov2_vits14"
DINO_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_BYTES = 88_283_115
DINO_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
DINO_IMAGE_SIZE = 224
DINO_TOKEN_COUNT = 256
DINO_FEATURE_DIM = 384
DINO_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_COST_DEFINITION = (
    "mean_j(1-dot(l2_normalize(successor_patch_j),"
    "l2_normalize(single_goal_patch_j)))"
)
DINO_TRUE_SUCCESSOR_POLICIES = frozenset(
    ("dino_true_successor", "dino_true_successor_shuffled")
)
DINO_POLICIES = frozenset((*DINO_TRUE_SUCCESSOR_POLICIES, "dino_persistence"))


@dataclass(frozen=True)
class GoalSpec:
    object_id: str
    landmark_xy: tuple[float, float]
    target_xy: tuple[float, float]
    target_yaw_rad: float
    image: torch.Tensor
    approach_images: torch.Tensor | None = None  # (N, C, H, W) multi-view goal renders


@dataclass
class PolicyResult:
    policy: str
    scene_id: str
    goal_object_id: str
    initial_distance_m: float
    final_distance_m: float
    progress_m: float
    path_length_m: float
    path_efficiency: float
    success: bool
    fell: bool
    blocks_executed: int
    primitive_sequence: list[str]
    mean_plan_cost: float | None = None
    decision_log: list[dict[str, Any]] | None = None


def _quat_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_xyzw_from_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[-4], q[-3], q[-2], q[-1]
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _xy_distance(a: tuple[float, float] | np.ndarray, b: tuple[float, float] | np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa[:2] - bb[:2]))


def _first_env(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim >= 2:
        return out[0]
    return out


def _render_tensor_from_base(
    build: Any,
    pack: Any,
    *,
    base_xyz_m: np.ndarray,
    base_quat_wxyz: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    quat_xyzw = _quat_xyzw_from_wxyz(np.asarray(base_quat_wxyz, dtype=np.float32))
    mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(pack)
    camera_config = camera_safety_config_from_pack(pack)
    pose, _safety = safe_camera_pose_from_base(
        np.asarray(base_xyz_m, dtype=np.float32),
        quat_xyzw,
        mount_xyz_body=mount_xyz,
        mount_rpy_body=mount_rpy,
        objects=pack.static_objects,
        config=camera_config,
    )
    build.camera.set_pose(pos=pose.position, lookat=pose.lookat, up=pose.up)
    rgb = RolloutRunner._extract_rgb(build.camera.render())
    if rgb is None:
        raise RuntimeError("Genesis camera returned no RGB frame")
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = np.asarray(rgb, dtype=np.uint8)
    img = Image.fromarray(rgb).convert("RGB").resize((224, 224))
    chw = np.array(img, copy=True).transpose(2, 0, 1)
    return torch.from_numpy(chw).float().div_(255.0).to(device)


def _current_pose(build: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        _first_env(RolloutRunner._as_np(build.robot.get_pos())),
        _first_env(RolloutRunner._as_np(build.robot.get_quat())),
    )


def _set_pose(
    *,
    build: Any,
    runner: RolloutRunner | None,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    robot = build.robot
    envs = [0]
    pos_batch = np.asarray(pos_xyz, dtype=np.float32)[None, :]
    quat_batch = np.asarray(quat_wxyz, dtype=np.float32)[None, :]
    if runner is None:
        robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
        robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)
        return
    reset_stance = getattr(runner.policy, "reset_stance_rad", runner._stance)
    stance_batch = np.asarray(reset_stance, dtype=np.float32)[None, :]
    robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
    robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)
    robot.set_dofs_position(stance_batch, runner._leg_dof_idx.tolist(), envs_idx=envs)
    robot.set_dofs_velocity(
        np.zeros_like(stance_batch), runner._leg_dof_idx.tolist(), envs_idx=envs
    )
    reset = getattr(runner.policy, "reset", None)
    if callable(reset):
        reset(envs)
    runner._last_executed[:] = 0.0


def _sync_robot_for_third_person_render(
    *,
    source_build: Any,
    render_build: Any | None,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    leg_dof_idx: Any = None,
) -> Any:
    """Mirror the authoritative robot state into an optional visible render scene."""

    if render_build is None or render_build is source_build:
        return source_build

    envs = [0]
    pos_batch = np.asarray(pos_xyz, dtype=np.float32)[None, :]
    quat_batch = np.asarray(quat_wxyz, dtype=np.float32)[None, :]
    render_build.robot.set_pos(pos_batch, envs_idx=envs, zero_velocity=True)
    render_build.robot.set_quat(quat_batch, envs_idx=envs, zero_velocity=False)

    if leg_dof_idx is not None:
        dof_idx = [int(i) for i in np.asarray(leg_dof_idx).reshape(-1).tolist()]
        if dof_idx:
            try:
                joint_pos = RolloutRunner._as_np(source_build.robot.get_dofs_position(dof_idx))
                joint_pos = _first_env(joint_pos)[None, :].astype(np.float32, copy=False)
                render_build.robot.set_dofs_position(joint_pos, dof_idx, envs_idx=envs)
                render_build.robot.set_dofs_velocity(np.zeros_like(joint_pos), dof_idx, envs_idx=envs)
            except Exception:
                pass

    try:
        render_build.scene.step()
    except Exception:
        pass
    return render_build


def _render_synced_third_person(
    *,
    source_build: Any,
    render_build: Any | None,
    base_xyz: np.ndarray,
    base_quat_wxyz: np.ndarray,
    yaw: float,
    side: float = 0.0,
    leg_dof_idx: Any = None,
) -> np.ndarray:
    third_build = _sync_robot_for_third_person_render(
        source_build=source_build,
        render_build=render_build,
        pos_xyz=base_xyz,
        quat_wxyz=base_quat_wxyz,
        leg_dof_idx=leg_dof_idx,
    )
    return _render_third_person(third_build, base_xyz, yaw, side=side)


def _execute_physical_primitive(
    runner: RolloutRunner,
    registry: PrimitiveRegistry,
    primitive_name: str,
    *,
    frame_sink: list | None = None,
    build: Any = None,
    pack: Any = None,
    device: torch.device | None = None,
    cam_side: float = 0.0,
    capture_policy_steps: bool = False,
    third_person_build: Any | None = None,
) -> np.ndarray:
    requested = expand_primitive_to_block(registry, primitive_name)
    clipped = runner._clip_block(requested[None, :, :]).executed[0]
    for tick in clipped:
        capture = frame_sink is not None and build is not None and pack is not None and device is not None
        if capture_policy_steps and capture:
            for _ in range(runner._policy_steps_per_command_tick):
                obs = runner._build_observation(tick[None, :])
                joint_targets = runner.policy.act(obs)
                runner._apply_joint_targets(joint_targets)
                for _step in range(runner._physics_steps_per_policy):
                    runner.build.scene.step()
                runner._sim_time_ns += runner._policy_dt_ns
                pos, quat = _current_pose(build)
                yaw_t = _yaw_from_quat_wxyz(quat)
                ego = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
                ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                third_np = _render_synced_third_person(
                    source_build=build,
                    render_build=third_person_build,
                    base_xyz=pos,
                    base_quat_wxyz=quat,
                    yaw=yaw_t,
                    side=cam_side,
                    leg_dof_idx=runner._leg_dof_idx,
                )
                frame_sink.append((third_np, ego_np, float(pos[0]), float(pos[1]), float(yaw_t)))
        else:
            runner._step_command_tick(tick[None, :])
            if capture:
                # Demo capture: physics already steps the articulated robot, so render
                # the egocentric + third-person views directly each control tick.
                pos, quat = _current_pose(build)
                yaw_t = _yaw_from_quat_wxyz(quat)
                ego = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
                ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                third_np = _render_synced_third_person(
                    source_build=build,
                    render_build=third_person_build,
                    base_xyz=pos,
                    base_quat_wxyz=quat,
                    yaw=yaw_t,
                    side=cam_side,
                    leg_dof_idx=runner._leg_dof_idx,
                )
                frame_sink.append((third_np, ego_np, float(pos[0]), float(pos[1]), float(yaw_t)))
    runner._last_executed[0] = clipped[-1]
    return clipped


def _run_multi_beacon_demo(
    *,
    model: torch.nn.Module,
    build: Any,
    pack: Any,
    registry: PrimitiveRegistry,
    sequences: list,
    action_tensor: torch.Tensor,
    n_beacons: int,
    goal_standoff_m: float,
    goal_radius_m: float,
    approach_distance_m: float,
    max_blocks: int,
    command_dt_s: float,
    device: torch.device,
    grid: InflatedOccupancyGrid,
    frame_sink: list,
    runner: Any = None,
    goal_insets: list | None = None,
    commit_margin: float = 0.01,
    third_person_build: Any | None = None,
) -> int:
    """Chase scene landmarks one after another with the lewm image-goal planner.

    Returns the number of beacons claimed (reached). Each leg uses a goal-facing
    standoff render of the next beacon as the image goal.
    """
    graph = pack.scene_graph
    base_z = float(pack.robot.spawn_xyz_m[2])
    beacons = []
    for object_id, cell_id in graph.landmark_cells:
        xy = graph.landmark_xy_for_cell(cell_id) or graph.cell_center(cell_id)
        beacons.append((str(object_id), np.array([float(xy[0]), float(xy[1])], dtype=np.float64)))
    if len(beacons) < 2:
        return 0

    spawn_xy = np.asarray(pack.robot.spawn_xyz_m[:2], dtype=np.float64)
    remaining = beacons[:]
    pos = spawn_xy.copy()
    tour: list = []
    while remaining and len(tour) < int(n_beacons):
        remaining.sort(key=lambda b: float(np.linalg.norm(b[1] - pos)))
        nxt = remaining.pop(0)
        tour.append(nxt)
        pos = nxt[1]
    if len(tour) < 2:
        return 0

    def standoff_for(beacon_xy, from_xy):
        bx = np.asarray(beacon_xy, dtype=np.float64)
        fx = np.asarray(from_xy, dtype=np.float64)
        d = bx - fx
        base_ang = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0
        for dang in [0.0] + [s * math.radians(deg) for deg in (15, 30, 45, 60, 90, 120) for s in (1, -1)]:
            ang = base_ang + dang
            unit = np.array([math.cos(ang), math.sin(ang)])
            so = bx - unit * float(goal_standoff_m)
            if not grid.is_free((float(so[0]), float(so[1]))):
                continue
            # Line-of-sight to the beacon as a target (not has_free_line, whose
            # endpoint inside the beacon's own inflated occupancy always fails).
            if not _line_of_sight_to_beacon(pack, (float(so[0]), float(so[1])), (float(bx[0]), float(bx[1]))):
                continue
            return so, math.atan2(bx[1] - so[1], bx[0] - so[0])
        return None

    first = standoff_for(tour[0][1], spawn_xy)
    if first is None:
        return 0
    so0, face0 = first
    unit0 = (tour[0][1] - so0)
    unit0 = unit0 / (np.linalg.norm(unit0) + 1e-9)
    start_xy = so0 - unit0 * float(approach_distance_m)
    if not grid.is_free((float(start_xy[0]), float(start_xy[1]))):
        start_xy = so0
    _set_pose(
        build=build,
        runner=runner,
        pos_xyz=np.array([start_xy[0], start_xy[1], base_z], dtype=np.float32),
        quat_wxyz=_quat_wxyz_from_yaw(face0),
    )

    claimed = 0
    step_m = 2.0           # re-aim ~2 m toward the beacon each iteration
    blocks_per_sub = 8     # servo this many blocks per fresh sub-goal, then re-aim
    max_total_blocks = int(max_blocks)  # per-beacon cap (set via --max-blocks)
    prev_prim: str | None = None
    for leg_index, (object_id, beacon_xy) in enumerate(tour):
        cur_xy = np.asarray(_current_pose(build)[0][:2], dtype=np.float64)
        so = standoff_for(beacon_xy, cur_xy)
        if so is None:
            break
        target_xy = np.asarray(so[0], dtype=np.float64)
        beacon_face = so[1]
        cam_side = 1.0 if leg_index % 2 == 0 else -1.0  # alternate follow side per leg
        total = 0
        reached = False
        while total < max_total_blocks:
            cur_xy = np.asarray(_current_pose(build)[0][:2], dtype=np.float64)
            if _xy_distance(cur_xy, target_xy) <= float(goal_radius_m):
                reached = True
                break
            d = target_xy - cur_xy
            dist_t = float(np.linalg.norm(d))
            # fresh sub-goal ~step_m toward the standoff from the current pose,
            # rendered facing the beacon (a distinct, growing servoing target)
            sub = target_xy if dist_t <= step_m else cur_xy + d / dist_t * step_m
            gyaw = beacon_face if dist_t <= step_m else math.atan2(
                beacon_xy[1] - sub[1], beacon_xy[0] - sub[0]
            )
            goal_img = _render_tensor_from_base(
                build, pack,
                base_xyz_m=np.array([sub[0], sub[1], base_z], dtype=np.float32),
                base_quat_wxyz=_quat_wxyz_from_yaw(gyaw), device=device,
            )
            if goal_insets is not None:
                goal_img_np = goal_img.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                goal_insets.append((len(frame_sink), goal_img_np, str(object_id)))
            for _ in range(blocks_per_sub):
                pos, quat = _current_pose(build)
                if _xy_distance(pos[:2], sub) <= float(goal_radius_m):
                    break
                image = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
                costs, prims = _lewm_primitive_costs(model, image, goal_img, sequences, action_tensor)
                best_i = int(np.argmin(costs))
                primitive_name = prims[best_i]
                # Commitment hysteresis: near-tie costs flip the argmin between
                # blocks and the gait stutters; keep the previous primitive
                # while it stays within a small margin of the best. Margin is
                # delicate: 3% measurably locks onto stale headings and costs
                # whole legs; keep it well under typical decision margins.
                if (commit_margin > 0.0 and prev_prim is not None
                        and prev_prim != "hold" and prev_prim in prims):
                    prev_cost = min(c for c, p in zip(costs, prims) if p == prev_prim)
                    if float(prev_cost) <= float(costs[best_i]) * (1.0 + commit_margin):
                        primitive_name = prev_prim
                prev_prim = primitive_name
                if runner is None:
                    _execute_kinematic_primitive(
                        build, registry, primitive_name, command_dt_s=command_dt_s, grid=grid,
                        frame_sink=frame_sink, pack=pack, device=device,
                        third_person_build=third_person_build,
                    )
                else:
                    _execute_physical_primitive(
                        runner, registry, primitive_name,
                        frame_sink=frame_sink, build=build, pack=pack, device=device,
                        cam_side=cam_side,
                        third_person_build=third_person_build,
                    )
                total += 1
        if reached:
            claimed += 1
        else:
            break
    return claimed


def _run_perception_multibeacon_demo(
    *,
    model: torch.nn.Module,
    build: Any,
    pack: Any,
    registry: PrimitiveRegistry,
    sequences: list,
    action_tensor: torch.Tensor,
    n_beacons: int,
    goal_standoff_m: float,
    goal_radius_m: float,
    max_blocks: int,
    command_dt_s: float,
    device: torch.device,
    grid: InflatedOccupancyGrid,
    frame_sink: list,
    scan_thresh: float,
    runner: Any = None,
    third_person_build: Any | None = None,
) -> int:
    """Pure-perception multi-beacon: image-goal servo to each beacon, and SCAN
    (rotate) when no candidate action beats holding (i.e. the goal beacon is not
    in view) until it crosses the FOV. No privileged breadcrumb waypoints; the
    only privileged inputs are the per-beacon goal keyframe and the claim radius.
    """
    graph = pack.scene_graph
    base_z = float(pack.robot.spawn_xyz_m[2])
    beacons = []
    for object_id, cell_id in graph.landmark_cells:
        xy = graph.landmark_xy_for_cell(cell_id) or graph.cell_center(cell_id)
        beacons.append((str(object_id), np.array([float(xy[0]), float(xy[1])], dtype=np.float64)))
    if len(beacons) < 2:
        return 0
    spawn_xy = np.asarray(pack.robot.spawn_xyz_m[:2], dtype=np.float64)
    remaining = beacons[:]
    pos = spawn_xy.copy()
    tour: list = []
    while remaining and len(tour) < int(n_beacons):
        remaining.sort(key=lambda b: float(np.linalg.norm(b[1] - pos)))
        nxt = remaining.pop(0)
        tour.append(nxt)
        pos = nxt[1]
    if len(tour) < 2:
        return 0

    def standoff_for(beacon_xy, from_xy):
        bx = np.asarray(beacon_xy, dtype=np.float64)
        fx = np.asarray(from_xy, dtype=np.float64)
        d = bx - fx
        base_ang = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0
        for dang in [0.0] + [s * math.radians(deg) for deg in (15, 30, 45, 60, 90, 120) for s in (1, -1)]:
            ang = base_ang + dang
            unit = np.array([math.cos(ang), math.sin(ang)])
            so = bx - unit * float(goal_standoff_m)
            if not grid.is_free((float(so[0]), float(so[1]))):
                continue
            if not _line_of_sight_to_beacon(pack, (float(so[0]), float(so[1])), (float(bx[0]), float(bx[1]))):
                continue
            return so, math.atan2(bx[1] - so[1], bx[0] - so[0])
        return None

    first = standoff_for(tour[0][1], spawn_xy)
    if first is None:
        return 0
    so0, face0 = first
    unit0 = (tour[0][1] - so0)
    unit0 = unit0 / (np.linalg.norm(unit0) + 1e-9)
    start_xy = so0 - unit0 * float(goal_standoff_m + 0.7)
    if not grid.is_free((float(start_xy[0]), float(start_xy[1]))):
        start_xy = so0
    # face AWAY from the beacon so the robot must scan to find it
    _set_pose(
        build=build, runner=runner,
        pos_xyz=np.array([start_xy[0], start_xy[1], base_z], dtype=np.float32),
        quat_wxyz=_quat_wxyz_from_yaw(face0 + math.pi),
    )

    def execute(prim):
        if runner is None:
            _execute_kinematic_primitive(
                build, registry, prim, command_dt_s=command_dt_s, grid=grid,
                frame_sink=frame_sink, pack=pack, device=device,
                third_person_build=third_person_build,
            )
        else:
            _execute_physical_primitive(
                runner, registry, prim, frame_sink=frame_sink, build=build, pack=pack, device=device,
                third_person_build=third_person_build,
            )

    claimed = 0
    for _obj, beacon_xy in tour:
        so = standoff_for(beacon_xy, np.asarray(_current_pose(build)[0][:2]))
        if so is None:
            break
        target_xy = np.asarray(so[0], dtype=np.float64)
        beacon_face = so[1]
        # one goal keyframe per beacon (rendered once; the goal specification)
        goal_img = _render_tensor_from_base(
            build, pack,
            base_xyz_m=np.array([target_xy[0], target_xy[1], base_z], dtype=np.float32),
            base_quat_wxyz=_quat_wxyz_from_yaw(beacon_face), device=device,
        )
        reached = False
        for _ in range(int(max_blocks)):
            pos, quat = _current_pose(build)
            if _xy_distance(pos[:2], target_xy) <= float(goal_radius_m):
                reached = True
                break
            image = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
            costs, prims = _lewm_primitive_costs(model, image, goal_img, sequences, action_tensor)
            best_i = int(costs.argmin())
            best_prim = prims[best_i]
            best_cost = float(costs[best_i])
            hold_cost = float(costs[prims.index("hold")]) if "hold" in prims else float(costs.max())
            # servoing signal: how much the best action beats holding still
            signal = (hold_cost - best_cost) / (abs(hold_cost) + 1e-6)
            prim = best_prim if signal > float(scan_thresh) else "yaw_left"  # scan when no gradient
            execute(prim)
        if reached:
            claimed += 1
        else:
            break
    return claimed


def _strip_obstacles(pack: Any) -> Any:
    """Demo-only: remove free-standing obstacle boxes from the scene (visual
    meshes, physics, occupancy grid, minimap); walls and landmarks stay. Used
    to declutter the open-field beacon demo."""
    import dataclasses
    manifest = dataclasses.replace(pack.scene_graph.manifest, obstacles=())
    return dataclasses.replace(
        pack,
        static_objects=tuple(o for o in pack.static_objects if "obstacle" not in o.kind),
        manifest=manifest,
        scene_graph=type(pack.scene_graph)(manifest),
    )


def _inject_extra_beacons(pack: Any, n_extra: int) -> Any:
    """Demo-only: add extra coloured landmark pillars at the scene's unused
    corner/edge slots (mirroring the existing landmark geometry), registered in
    the manifest so the scene graph, beacon tour, physics, and HUD all see
    them. Colours come from the standard landmark palette."""
    import dataclasses
    manifest = pack.scene_graph.manifest
    existing = list(manifest.landmarks)
    if n_extra <= 0 or not existing:
        return pack
    ax = max(abs(b.center_xyz_m[0]) for b in existing)
    ay = max(abs(b.center_xyz_m[1]) for b in existing)
    used = {(round(b.center_xyz_m[0], 1), round(b.center_xyz_m[1], 1)) for b in existing}
    slots = [(sx * ax, sy * ay) for sx in (1, -1) for sy in (1, -1)]
    slots += [(0.0, ay), (0.0, -ay), (ax, 0.0), (-ax, 0.0)]
    slots = [s for s in slots if (round(s[0], 1), round(s[1], 1)) not in used]
    palette = ["landmark_green", "landmark_yellow", "landmark_red", "landmark_blue"]
    used_mats = {b.material_id for b in existing}
    mats = [m for m in palette if m not in used_mats] + [m for m in palette if m in used_mats]
    box_template = existing[0]
    static_template = next(o for o in pack.static_objects if o.kind == "landmark")
    new_boxes, new_statics = [], []
    for i in range(min(int(n_extra), len(slots))):
        x, y = slots[i]
        mat = mats[i % len(mats)]
        name = mat if mat not in used_mats else f"{mat}_extra{i}"
        center = (float(x), float(y), float(box_template.center_xyz_m[2]))
        new_boxes.append(dataclasses.replace(
            box_template, object_id=name, material_id=mat, center_xyz_m=center))
        new_statics.append(dataclasses.replace(
            static_template, object_id=name, material_id=mat, center_xyz_m=center))
        used_mats.add(mat)
    manifest = dataclasses.replace(manifest, landmarks=(*manifest.landmarks, *new_boxes))
    return dataclasses.replace(
        pack,
        static_objects=(*pack.static_objects, *new_statics),
        manifest=manifest,
        scene_graph=type(pack.scene_graph)(manifest),
    )


def _hud_world_to_px(x, y, bounds, mx0, my0, mw, mh):
    (xlo, ylo), (xhi, yhi) = bounds
    nx = (float(x) - xlo) / max(xhi - xlo, 1e-8)
    ny = (float(y) - ylo) / max(yhi - ylo, 1e-8)
    px = mx0 + int(np.clip(nx, 0.0, 1.0) * mw)
    py = my0 + mh - int(np.clip(ny, 0.0, 1.0) * mh)  # world +y is up
    return px, py


def _draw_minimap(draw, bounds, occ, beacons, trail, robot_xy, robot_yaw, target_idx, claimed, mx0, my0, mw, mh):
    draw.rectangle([mx0, my0, mx0 + mw, my0 + mh], fill=(22, 22, 28), outline=(95, 95, 105))
    rn = occ.shape[0]
    for j in range(rn):
        for i in range(rn):
            if occ[j, i]:
                x0 = mx0 + int(i / rn * mw)
                y0 = my0 + mh - int((j + 1) / rn * mh)
                x1 = mx0 + int((i + 1) / rn * mw)
                y1 = my0 + mh - int(j / rn * mh)
                draw.rectangle([x0, y0, x1, y1], fill=(120, 55, 55))
    if len(trail) > 1:
        pts = [_hud_world_to_px(x, y, bounds, mx0, my0, mw, mh) for (x, y) in trail[-600:]]
        draw.line(pts, fill=(255, 220, 80), width=2)
    for bi, (bxy, col, _name) in enumerate(beacons):
        bx, by = _hud_world_to_px(bxy[0], bxy[1], bounds, mx0, my0, mw, mh)
        draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=col, outline=(15, 15, 15))
        if bi in claimed:
            draw.ellipse([bx - 9, by - 9, bx + 9, by + 9], outline=(70, 230, 120), width=2)
        elif bi == target_idx:
            draw.ellipse([bx - 10, by - 10, bx + 10, by + 10], outline=(255, 255, 255), width=2)
    rx, ry = _hud_world_to_px(robot_xy[0], robot_xy[1], bounds, mx0, my0, mw, mh)
    s = 9.0
    tri = [
        (rx + int(math.cos(robot_yaw) * s), ry - int(math.sin(robot_yaw) * s)),
        (rx + int(math.cos(robot_yaw + 2.5) * s * 0.8), ry - int(math.sin(robot_yaw + 2.5) * s * 0.8)),
        (rx + int(math.cos(robot_yaw - 2.5) * s * 0.8), ry - int(math.sin(robot_yaw - 2.5) * s * 0.8)),
    ]
    draw.polygon(tri, fill=(255, 255, 255), outline=(10, 10, 10))


def _write_hud_video(path: Path, pack: Any, grid: Any, frames: list, fps: float, goal_radius: float, title: str,
                     goal_insets: list | None = None) -> None:
    """Compose a HUD video: title, follow + robot-eye panels, minimap, status text.

    frames is a list of (third_np, ego_np, robot_x, robot_y, robot_yaw).
    goal_insets: optional [(frame_idx, image_np, beacon_name), ...] — the live
    servo target image, shown as a bordered inset from frame_idx onwards.
    """
    import imageio
    from PIL import ImageDraw, ImageFont
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    manifest = pack.scene_graph.manifest
    (xlo, ylo), (xhi, yhi) = manifest.world_bounds_xy_m
    bounds = ((float(xlo), float(ylo)), (float(xhi), float(yhi)))
    raw = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.0)
    rn = 76
    occ = np.zeros((rn, rn), dtype=bool)
    for j in range(rn):
        for i in range(rn):
            x = xlo + (i + 0.5) / rn * (xhi - xlo)
            y = ylo + (j + 0.5) / rn * (yhi - ylo)
            occ[j, i] = not raw.is_free((float(x), float(y)))
    graph = pack.scene_graph
    beacons = []
    for obj_id, cell in graph.landmark_cells:
        xy = graph.landmark_xy_for_cell(cell) or graph.cell_center(cell)
        name = str(obj_id)
        col = ((225, 70, 70) if "red" in name else (70, 130, 255) if "blue" in name
               else (80, 205, 120) if "green" in name else (235, 200, 70))
        beacons.append((np.array([float(xy[0]), float(xy[1])]), col, name))

    def _font(sz, bold=False):
        try:
            p = "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf"
            return ImageFont.truetype(p, sz)
        except Exception:
            return ImageFont.load_default()

    f_title, f_lab, f_stat = _font(19, True), _font(14), _font(13)
    W, H = 896, 496
    out: list = []
    trail: list = []
    claimed: set = set()
    flash_left, flash_text = 0, ""
    insets = sorted(goal_insets or [], key=lambda t: t[0])
    inset_ptr, active_inset = 0, None
    name_col = {"red": (225, 70, 70), "blue": (70, 130, 255),
                "green": (80, 205, 120), "yellow": (235, 200, 70)}
    for fi, (third, ego, rx, ry, yaw) in enumerate(frames):
        while inset_ptr < len(insets) and insets[inset_ptr][0] <= fi:
            active_inset = insets[inset_ptr]
            inset_ptr += 1
        trail.append((rx, ry))
        for bi, (bxy, _c, _n) in enumerate(beacons):
            # The robot stops at a STANDOFF from the beacon (it never reaches
            # the beacon center), so the display claim radius must cover
            # standoff + claim radius or successful claims render as misses.
            if bi not in claimed and math.hypot(rx - bxy[0], ry - bxy[1]) <= goal_radius * 1.1:
                claimed.add(bi)
                flash_left = max(1, int(round(fps * 1.6)))
                flash_text = (f"{beacons[bi][2].replace('landmark_', '').upper()} BEACON CLAIMED"
                              f"  ({len(claimed)}/{len(beacons)})")
        unclaimed = [bi for bi in range(len(beacons)) if bi not in claimed]
        target = min(unclaimed, key=lambda bi: math.hypot(rx - beacons[bi][0][0], ry - beacons[bi][0][1])) if unclaimed else None
        canvas = Image.new("RGB", (W, H), (16, 16, 20))
        canvas.paste(Image.fromarray(third).resize((416, 416)), (12, 44))
        canvas.paste(Image.fromarray(ego).resize((300, 300)), (456, 44))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, W - 1, 36], fill=(10, 10, 13), outline=(70, 70, 80))
        draw.text((14, 9), title, fill=(0, 235, 120), font=f_title)
        draw.text((12, 462), "Third-person follow", fill=(195, 195, 200), font=f_lab)
        draw.text((456, 346), "Robot-eye (perception)", fill=(195, 195, 200), font=f_lab)
        _draw_minimap(draw, bounds, occ, beacons, trail, (rx, ry), yaw, target, claimed, 456, 372, 300, 106)
        draw.text((766, 372), "Map", fill=(195, 195, 200), font=f_lab)
        # Per-beacon checklist: coloured chip + name, ticked when claimed.
        for bi, (_bxy, col, name) in enumerate(beacons):
            yy = 392 + bi * 17
            draw.rectangle([766, yy + 3, 776, yy + 13], fill=col)
            label = name.replace("landmark_", "")
            if bi in claimed:
                draw.text((782, yy), f"{label} ✓", fill=(70, 230, 120), font=f_stat)
            else:
                draw.text((782, yy), label, fill=(160, 160, 168), font=f_stat)
        tgt_name = beacons[target][2].replace("landmark_", "") if target is not None else "done"
        tgt_d = math.hypot(rx - beacons[target][0][0], ry - beacons[target][0][1]) if target is not None else 0.0
        y0 = 392 + len(beacons) * 17 + 6
        draw.text((766, y0), f"target: {tgt_name}", fill=(220, 220, 110), font=f_stat)
        draw.text((766, min(y0 + 18, H - 18)), f"dist: {tgt_d:.1f} m", fill=(200, 200, 205), font=f_stat)
        if active_inset is not None:
            _idx, inset_np, inset_name = active_inset
            short = inset_name.replace("landmark_", "").split("_")[0]
            col = name_col.get(short, (235, 200, 70))
            canvas.paste(Image.fromarray(inset_np).resize((124, 124)), (766, 44))
            draw.rectangle([765, 43, 890, 168], outline=col, width=2)
            draw.text((766, 172), "Target view (input)", fill=col, font=f_stat)
        if flash_left > 0:
            draw.rectangle([12, 44, 428, 460], outline=(70, 230, 120), width=4)
            draw.rectangle([12, 200, 428, 252], fill=(10, 40, 16), outline=(70, 230, 120), width=2)
            draw.text((26, 214), flash_text, fill=(70, 230, 120), font=f_title)
            flash_left -= 1
        out.append(np.asarray(canvas))
    hold = [out[-1]] * max(1, int(round(fps)))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), out + hold, fps=fps, macro_block_size=8)


def _render_third_person(build: Any, base_xyz: np.ndarray, yaw: float, size: int = 224,
                         side: float = 0.0) -> np.ndarray:
    """Render a third-person follow view (behind + above the robot, looking
    ahead). ``side`` shifts the camera laterally (in robot widths) so demo legs
    can alternate the follow angle."""
    heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    perp = np.array([-heading[1], heading[0]], dtype=np.float32)
    cam_pos = np.array(
        [float(base_xyz[0]) - heading[0] * 1.25 + perp[0] * 0.8 * float(side),
         float(base_xyz[1]) - heading[1] * 1.25 + perp[1] * 0.8 * float(side),
         float(base_xyz[2]) + 1.55 + 0.2 * abs(float(side))],
        dtype=np.float32,
    )
    lookat = np.array(
        [float(base_xyz[0]) + heading[0] * 0.15,
         float(base_xyz[1]) + heading[1] * 0.15,
         float(base_xyz[2]) + 0.25],
        dtype=np.float32,
    )
    build.camera.set_pose(pos=cam_pos, lookat=lookat, up=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    rgb = RolloutRunner._extract_rgb(build.camera.render())
    if rgb is None:
        raise RuntimeError("Genesis camera returned no RGB frame (third-person)")
    if rgb.ndim == 4:
        rgb = rgb[0]
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((size, size))
    return np.array(img, copy=True)


def _execute_kinematic_primitive(
    build: Any,
    registry: PrimitiveRegistry,
    primitive_name: str,
    *,
    command_dt_s: float,
    grid: InflatedOccupancyGrid | None,
    frame_sink: list | None = None,
    pack: Any = None,
    device: torch.device | None = None,
    third_person_build: Any | None = None,
) -> np.ndarray:
    block = expand_primitive_to_block(registry, primitive_name)
    pos, quat = _current_pose(build)
    pos = np.asarray(pos, dtype=np.float32).copy()
    yaw = _yaw_from_quat_wxyz(quat)
    for vx_body, vy_body, yaw_rate in block:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        next_xy = (
            float(pos[0]) + (float(vx_body) * cos_yaw - float(vy_body) * sin_yaw) * command_dt_s,
            float(pos[1]) + (float(vx_body) * sin_yaw + float(vy_body) * cos_yaw) * command_dt_s,
        )
        if grid is not None and not grid.is_free(next_xy):
            break
        pos[0] = next_xy[0]
        pos[1] = next_xy[1]
        yaw = wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
        if frame_sink is not None and pack is not None and device is not None:
            # Demo capture: at each kinematic sub-step render the egocentric
            # (perception) view and a third-person follow view, side by side.
            build.robot.set_pos(pos[None, :], envs_idx=[0], zero_velocity=True)
            build.robot.set_quat(_quat_wxyz_from_yaw(yaw)[None, :], envs_idx=[0], zero_velocity=False)
            # Kinematic mode never steps, so a visible main scene needs one
            # refresh step for meshes to track set_pos. Split-render demos step
            # the separate third-person scene during sync instead.
            if third_person_build is None:
                try:
                    build.scene.step()
                except Exception:
                    pass
            ego = _render_tensor_from_base(
                build, pack, base_xyz_m=pos, base_quat_wxyz=_quat_wxyz_from_yaw(yaw), device=device,
            )
            ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            third_np = _render_synced_third_person(
                source_build=build,
                render_build=third_person_build,
                base_xyz=pos,
                base_quat_wxyz=_quat_wxyz_from_yaw(yaw),
                yaw=yaw,
            )
            frame_sink.append((third_np, ego_np, float(pos[0]), float(pos[1]), float(yaw)))
    build.robot.set_pos(pos[None, :], envs_idx=[0], zero_velocity=True)
    build.robot.set_quat(_quat_wxyz_from_yaw(yaw)[None, :], envs_idx=[0], zero_velocity=False)
    return block


def _kinematic_endpoint(
    sequence: tuple[str, ...],
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid | None,
    start_xy: tuple[float, float] | np.ndarray,
    start_yaw: float,
    command_dt_s: float,
) -> tuple[float, float]:
    """Integrate one candidate without mutating the scene.

    This deliberately mirrors ``_execute_kinematic_primitive`` tick for tick:
    a collision stops the current primitive, while a later primitive in the
    candidate sequence is still considered.  ``diagnose_nav_cost.py`` uses the
    same semantics for its endpoint oracle.
    """
    x, y = float(start_xy[0]), float(start_xy[1])
    yaw = float(start_yaw)
    for primitive_name in sequence:
        block = expand_primitive_to_block(registry, primitive_name)
        for vx_body, vy_body, yaw_rate in block:
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            next_xy = (
                x + (float(vx_body) * cos_yaw - float(vy_body) * sin_yaw) * command_dt_s,
                y + (float(vx_body) * sin_yaw + float(vy_body) * cos_yaw) * command_dt_s,
            )
            if grid is not None and not grid.is_free(next_xy):
                break
            x, y = next_xy
            yaw = wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
    return x, y


def _oracle_candidate_costs(
    sequences: list[tuple[str, ...]],
    registry: PrimitiveRegistry,
    grid: InflatedOccupancyGrid | None,
    start_xy: tuple[float, float] | np.ndarray,
    start_yaw: float,
    command_dt_s: float,
    target_xy: tuple[float, float],
) -> np.ndarray:
    """Privileged endpoint distance for every candidate, in candidate order."""
    return np.asarray(
        [
            _xy_distance(
                _kinematic_endpoint(
                    sequence,
                    registry,
                    grid,
                    start_xy,
                    start_yaw,
                    command_dt_s,
                ),
                target_xy,
            )
            for sequence in sequences
        ],
        dtype=np.float64,
    )


def _oracle_ranking(
    costs: np.ndarray,
    sequences: list[tuple[str, ...]],
    *,
    tie_tolerance_m: float = ORACLE_COST_TIE_TOLERANCE_M,
) -> dict[str, Any]:
    """Summarize a candidate-cost vector without hiding numerical ties.

    Every candidate within ``tie_tolerance_m`` of the exact minimum is treated
    as optimal.  The representative used by ``oracle_mpc`` is the first such
    row in the committed candidate order.  Regret always remains relative to
    the exact minimum, so the representative can have only a tolerance-sized
    positive regret.
    """
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("oracle costs must be a non-empty one-dimensional vector")
    if len(sequences) != int(values.size):
        raise ValueError(
            "oracle costs and candidate sequences disagree: "
            f"{values.size} costs for {len(sequences)} sequences"
        )
    if not np.isfinite(values).all():
        raise ValueError("oracle costs must all be finite")
    tolerance = float(tie_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("oracle tie tolerance must be finite and non-negative")

    exact_minimum = float(np.min(values))
    optimal_indices = np.flatnonzero(values <= exact_minimum + tolerance)
    representative_index = int(optimal_indices[0])
    optimal_first_primitives = list(
        dict.fromkeys(sequences[int(index)][0] for index in optimal_indices)
    )
    return {
        "best_candidate_index": representative_index,
        "best_cost_m": exact_minimum,
        "optimal_candidate_indices": [int(index) for index in optimal_indices],
        "optimal_candidate_count": int(optimal_indices.size),
        "optimal_first_primitives": optimal_first_primitives,
        "tie_tolerance_m": tolerance,
        "tie_break": ORACLE_TIE_BREAK,
    }


def _oracle_first_action_assessment(
    costs: np.ndarray,
    sequences: list[tuple[str, ...]],
    primitive_name: str,
    *,
    oracle_best_cost_m: float,
    tie_tolerance_m: float = ORACLE_COST_TIE_TOLERANCE_M,
) -> dict[str, Any]:
    """Return tie-aware regret for the executed first primitive."""
    values = np.asarray(costs, dtype=np.float64)
    matching = np.asarray(
        [index for index, sequence in enumerate(sequences) if sequence[0] == primitive_name],
        dtype=np.int64,
    )
    if not matching.size:
        return {
            "best_candidate_index": None,
            "best_cost_m": None,
            "regret_m": None,
            "disagreement": None,
        }
    local_costs = values[matching]
    best_cost = float(np.min(local_costs))
    local_optimal = matching[
        np.flatnonzero(local_costs <= best_cost + float(tie_tolerance_m))
    ]
    representative_index = int(local_optimal[0])
    regret = max(0.0, best_cost - float(oracle_best_cost_m))
    return {
        "best_candidate_index": representative_index,
        "best_cost_m": best_cost,
        "regret_m": regret,
        "disagreement": bool(regret > float(tie_tolerance_m)),
    }


def _deterministic_score_permutation(
    candidate_count: int,
    *,
    seed: int,
    block_index: int,
) -> np.ndarray:
    """Return a reproducible score permutation independent of policy order."""
    indices = list(range(int(candidate_count)))
    mixed_seed = (
        (int(seed) & ((1 << 64) - 1))
        ^ (((int(block_index) + 1) * ORACLE_SHUFFLE_MIX_CONSTANT) & ((1 << 64) - 1))
    )
    random.Random(mixed_seed).shuffle(indices)
    if len(indices) > 1 and indices == list(range(len(indices))):
        indices = indices[1:] + indices[:1]
    return np.asarray(indices, dtype=np.int64)


def _oracle_assay_provenance(
    *,
    seed: int,
    max_candidates: int | None,
    sequences: list[tuple[str, ...]],
    mode: str,
) -> dict[str, Any]:
    """Exact, JSON-ready definition of the oracle and shuffled intervention."""
    return {
        "name": ORACLE_ASSAY_NAME,
        "version": ORACLE_ASSAY_VERSION,
        "mode": str(mode),
        "validity": (
            "exact_execution_oracle_in_kinematic_mode"
            if str(mode) == "kinematic"
            else "privileged_nominal_geometric_controller_not_a_physical_outcome_oracle"
        ),
        "cost_definition": (
            "euclidean_distance_from_nominal_kinematic_sequence_endpoint_"
            "to_privileged_target_xy"
        ),
        "collision_semantics": (
            "stop_current_primitive_at_first_occupied_tick_then_continue_later_"
            "candidate_primitives"
        ),
        "tie_tolerance_m": ORACLE_COST_TIE_TOLERANCE_M,
        "tie_break": ORACLE_TIE_BREAK,
        "candidate_bank": {
            "count": len(sequences),
            "max_candidates": (
                None if max_candidates is None else int(max_candidates)
            ),
            "ordered_sequences": [list(sequence) for sequence in sequences],
        },
        "planning_grid": {
            "cell_size_m": PLANNING_GRID_CELL_SIZE_M,
            "inflation_m": PLANNING_GRID_INFLATION_M,
        },
        "shuffle": {
            "name": ORACLE_SHUFFLE_NAME,
            "version": ORACLE_SHUFFLE_VERSION,
            "seed": int(seed),
            "block_seed_definition": (
                "uint64(seed) XOR uint64((block_index + 1) * "
                "0x9E3779B97F4A7C15)"
            ),
            "permutation_definition": (
                "python_random_shuffle_candidate_indices; rotate_left_one_if_"
                "the_whole_permutation_is_identity"
            ),
            "score_assignment": (
                "policy_scores[candidate_index] = "
                "oracle_costs[permutation[candidate_index]]"
            ),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dino_repository_commit(repo_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot identify DINO repository commit: {repo_path}") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"invalid DINO repository commit: {commit!r}")
    return commit


def _dino_assay_provenance(
    *,
    repo_path: Path,
    repository_commit: str,
    checkpoint_path: Path,
    checkpoint_bytes: int,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    """Flat, analyzer-bound provenance for the frozen DINO ceiling."""

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    return {
        "name": DINO_ASSAY_NAME,
        "version": DINO_ASSAY_VERSION,
        "encoder_name": DINO_ENCODER_NAME,
        "repository_path": str(Path(repo_path).resolve()),
        "repository_commit": str(repository_commit),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_bytes": int(checkpoint_bytes),
        "checkpoint_sha256": str(checkpoint_sha256),
        "device": str(device),
        "device_name": device_name,
        "torch_version": str(torch.__version__),
        "hip_version": (
            None if torch.version.hip is None else str(torch.version.hip)
        ),
        "frozen": True,
        "eval_mode": True,
        "no_grad": True,
        "feature_cache_written": False,
        "input_rgb_shape": [3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
        "imagenet_mean": list(DINO_IMAGENET_MEAN),
        "imagenet_std": list(DINO_IMAGENET_STD),
        "patch_output_shape": [DINO_TOKEN_COUNT, DINO_FEATURE_DIM],
        "token_normalization": "per_patch_l2",
        "cost_definition": DINO_COST_DEFINITION,
        "goal_view_count": 1,
        "successor_evaluation": (
            "reset_observed_pose_execute_one_nominal_kinematic_candidate_render_"
            "actual_successor_restore_observed_pose"
        ),
    }


def _load_dinov2_encoder(
    repo_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load and bind the exact preregistered local frozen DINOv2 encoder."""

    repo = Path(repo_path).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if not repo.is_dir() or not checkpoint.is_file():
        raise RuntimeError("DINO repository/checkpoint types are invalid")
    repository_commit = _dino_repository_commit(repo)
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint_sha256 = _sha256_file(checkpoint)
    if repository_commit != DINO_REPOSITORY_COMMIT:
        raise RuntimeError(
            "DINO repository commit changed: "
            f"expected {DINO_REPOSITORY_COMMIT}, got {repository_commit}"
        )
    if (
        checkpoint_bytes != DINO_CHECKPOINT_BYTES
        or checkpoint_sha256 != DINO_CHECKPOINT_SHA256
    ):
        raise RuntimeError("DINO checkpoint binding changed")

    encoder = torch.hub.load(
        str(repo),
        DINO_ENCODER_NAME,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    del state
    encoder = encoder.to(device).eval().requires_grad_(False)
    provenance = _dino_assay_provenance(
        repo_path=repo,
        repository_commit=repository_commit,
        checkpoint_path=checkpoint,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )
    return encoder, provenance


def _preprocess_dinov2_images(images: torch.Tensor) -> torch.Tensor:
    """Apply exact 224-square RGB and ImageNet preprocessing for DINOv2."""

    if not isinstance(images, torch.Tensor):
        raise TypeError("DINO images must be a torch.Tensor")
    values = images.unsqueeze(0) if images.ndim == 3 else images
    if values.ndim != 4 or tuple(values.shape[1:]) != (
        3,
        DINO_IMAGE_SIZE,
        DINO_IMAGE_SIZE,
    ):
        raise ValueError("DINO images must have shape [B,3,224,224]")
    if values.shape[0] < 1:
        raise ValueError("DINO image batch must be nonempty")
    if values.dtype == torch.uint8:
        values = values.to(torch.float32).div(255.0)
    elif values.dtype.is_floating_point:
        values = values.to(torch.float32)
    else:
        raise TypeError("DINO images must be uint8 or floating point")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("DINO images contain a nonfinite value")
    if bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise ValueError("floating DINO images must be in [0,1]")
    mean = values.new_tensor(DINO_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = values.new_tensor(DINO_IMAGENET_STD).view(1, 3, 1, 1)
    return values.sub(mean).div(std).contiguous()


@torch.no_grad()
def _encode_dinov2_images(
    encoder: Any,
    images: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    inputs = _preprocess_dinov2_images(images).to(device)
    payload = encoder.forward_features(inputs)
    if not isinstance(payload, dict) or "x_norm_patchtokens" not in payload:
        raise RuntimeError("DINO encoder did not return x_norm_patchtokens")
    raw_tokens = payload["x_norm_patchtokens"]
    if not isinstance(raw_tokens, torch.Tensor) or raw_tokens.ndim != 3:
        raise RuntimeError("DINO patch tokens must be rank three")
    expected = (inputs.shape[0], DINO_TOKEN_COUNT, DINO_FEATURE_DIM)
    if tuple(raw_tokens.shape) != expected:
        raise RuntimeError(f"DINO patch-token shape changed: {tuple(raw_tokens.shape)}")
    tokens = raw_tokens.to(torch.float32)
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError("DINO patch tokens contain a nonfinite value")
    norms = torch.linalg.vector_norm(tokens, dim=-1)
    if bool((norms <= 0.0).any()):
        raise FloatingPointError("DINO patch tokens contain a zero vector")
    return F.normalize(tokens, p=2.0, dim=-1).contiguous()


def _dinov2_same_patch_costs(
    candidate_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
) -> torch.Tensor:
    """Mean same-position patch cosine distance to one encoded goal."""

    if not isinstance(candidate_tokens, torch.Tensor) or not isinstance(
        goal_tokens, torch.Tensor
    ):
        raise TypeError("candidate and goal tokens must be tensors")
    goal = goal_tokens.unsqueeze(0) if goal_tokens.ndim == 2 else goal_tokens
    if candidate_tokens.ndim != 3 or candidate_tokens.shape[0] < 1:
        raise ValueError("candidate tokens must have shape [N,256,D]")
    if tuple(candidate_tokens.shape[1:]) != (DINO_TOKEN_COUNT, DINO_FEATURE_DIM):
        raise ValueError("candidate tokens must have shape [N,256,384]")
    if tuple(goal.shape) != (1, DINO_TOKEN_COUNT, DINO_FEATURE_DIM):
        raise ValueError("goal tokens must have shape [1,256,384]")
    if candidate_tokens.device != goal.device:
        raise TypeError("candidate and goal tokens must share one device")
    if not bool(torch.isfinite(candidate_tokens).all()) or not bool(
        torch.isfinite(goal).all()
    ):
        raise FloatingPointError("DINO cost received nonfinite tokens")
    candidates = F.normalize(candidate_tokens.to(torch.float32), p=2.0, dim=-1)
    normalized_goal = F.normalize(goal.to(torch.float32), p=2.0, dim=-1)
    cosine = torch.sum(candidates * normalized_goal, dim=-1)
    costs = torch.mean(1.0 - cosine, dim=-1)
    if not bool(torch.isfinite(costs).all()):
        raise FloatingPointError("DINO candidate costs are nonfinite")
    return costs


def _rank_dino_policy_scores(
    unshuffled_costs: np.ndarray,
    *,
    policy_name: str,
    seed: int,
    block_index: int,
) -> dict[str, Any]:
    values = np.asarray(unshuffled_costs, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("DINO candidate costs must be a finite nonempty vector")
    if policy_name not in DINO_POLICIES:
        raise ValueError(f"not a DINO policy: {policy_name!r}")
    score_sources = np.arange(values.size, dtype=np.int64)
    if policy_name == "dino_true_successor_shuffled":
        score_sources = _deterministic_score_permutation(
            int(values.size), seed=int(seed), block_index=int(block_index)
        )
    policy_scores = values[score_sources]
    selected_index = int(np.argmin(policy_scores))
    margin = None
    if policy_scores.size > 1:
        two_smallest = np.partition(policy_scores, 1)[:2]
        margin = float(two_smallest[1] - two_smallest[0])
    return {
        "policy_candidate_scores": policy_scores,
        "score_source_candidate_indices": score_sources,
        "selected_candidate_index": selected_index,
        "selected_score_source_candidate_index": int(score_sources[selected_index]),
        "selected_policy_score": float(policy_scores[selected_index]),
        "policy_score_margin": margin,
    }


def _validate_dino_policy_scope(
    policies: list[str],
    *,
    mode: str,
    horizon: int,
    goal_views: int,
) -> None:
    requested = DINO_POLICIES.intersection(policies)
    if not requested:
        return
    if int(horizon) != 1:
        raise ValueError("DINO ceiling policies are preregistered for H1 only")
    if int(goal_views) != 0:
        raise ValueError("DINO ceiling policies support one goal image only")
    if DINO_TRUE_SUCCESSOR_POLICIES.intersection(requested) and mode != "kinematic":
        raise ValueError("DINO true-successor policies require kinematic mode")


def _render_kinematic_h1_successors(
    *,
    build: Any,
    pack: Any,
    registry: PrimitiveRegistry,
    sequences: list[tuple[str, ...]],
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    command_dt_s: float,
    grid: InflatedOccupancyGrid | None,
    render_device: torch.device,
) -> torch.Tensor:
    """Render candidates in order, resetting around every nominal H1 branch."""

    if not sequences or any(len(sequence) != 1 for sequence in sequences):
        raise ValueError("DINO true-successor evaluation requires nonempty H1 candidates")
    images: list[torch.Tensor] = []
    try:
        for sequence in sequences:
            _set_pose(
                build=build,
                runner=None,
                pos_xyz=start_pos,
                quat_wxyz=start_quat,
            )
            _execute_kinematic_primitive(
                build,
                registry,
                sequence[0],
                command_dt_s=command_dt_s,
                grid=grid,
            )
            successor_pos, successor_quat = _current_pose(build)
            images.append(
                _render_tensor_from_base(
                    build,
                    pack,
                    base_xyz_m=successor_pos,
                    base_quat_wxyz=successor_quat,
                    device=render_device,
                )
            )
    finally:
        _set_pose(
            build=build,
            runner=None,
            pos_xyz=start_pos,
            quat_wxyz=start_quat,
        )
    return torch.stack(images, dim=0)


def _dino_true_successor_candidate_costs(
    *,
    encoder: Any,
    goal_tokens: torch.Tensor,
    dino_device: torch.device,
    build: Any,
    pack: Any,
    registry: PrimitiveRegistry,
    sequences: list[tuple[str, ...]],
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    command_dt_s: float,
    grid: InflatedOccupancyGrid | None,
) -> np.ndarray:
    images = _render_kinematic_h1_successors(
        build=build,
        pack=pack,
        registry=registry,
        sequences=sequences,
        start_pos=start_pos,
        start_quat=start_quat,
        command_dt_s=command_dt_s,
        grid=grid,
        render_device=dino_device,
    )
    successor_tokens = _encode_dinov2_images(
        encoder, images, device=dino_device
    )
    return (
        _dinov2_same_patch_costs(successor_tokens, goal_tokens)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )


def _dino_persistence_candidate_costs(
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    *,
    candidate_count: int,
) -> np.ndarray:
    if int(candidate_count) < 1:
        raise ValueError("candidate_count must be positive")
    scalar = float(
        _dinov2_same_patch_costs(current_tokens, goal_tokens)[0].detach().cpu()
    )
    return np.full(int(candidate_count), scalar, dtype=np.float64)


def _primitive_active_blocks(
    registry: PrimitiveRegistry,
    primitive_names: list[str],
) -> dict[str, np.ndarray]:
    return _pb_active_blocks(registry, primitive_names)


def _candidate_action_tensor(
    primitive_blocks: dict[str, np.ndarray],
    primitive_names: list[str],
    horizon: int,
    *,
    max_candidates: int | None,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[tuple[str, ...]], torch.Tensor]:
    return _pb_candidate_action_tensor(
        primitive_blocks,
        primitive_names,
        horizon,
        max_candidates=max_candidates,
        rng=rng,
        device=device,
    )


@torch.no_grad()
def _encode_frame(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_raw, z_proj = model.encode(image[None, ...], None)
    return z_raw, z_proj


@torch.no_grad()
def _choose_lewm_primitive(
    model: torch.nn.Module,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
) -> tuple[str, float]:
    # Behaviour-locked delegation to lewm.planning.local_mpc (Stage 0 refactor):
    # energy-head cost if present, else plan_cost; min over goal views.
    return _lmpc_choose_primitive(model, image, goal_image, sequences, action_tensor)


@torch.no_grad()
def _lewm_primitive_costs(
    model: torch.nn.Module,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    """Full image-goal cost over every candidate first-primitive (lower = closer)."""
    # Behaviour-locked delegation to lewm.planning.local_mpc (Stage 0 refactor):
    # pose-head metric cost if present, elif energy head, else plan_cost.
    return _lmpc_primitive_costs(model, image, goal_image, sequences, action_tensor)


@torch.no_grad()
def _choose_lewm_pose_primitive(
    model: torch.nn.Module,
    pose_head: RelPoseHead,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
) -> tuple[str, float]:
    """No-privileged-runtime-geometry planner: cost = the model's own predicted
    distance-to-goal ||dxy(z_pred, z_goal)||. No privileged geometry at runtime."""
    z_start_raw, _ = _encode_frame(model, image)
    goal_views = goal_image if goal_image.dim() == 4 else goal_image[None]
    z_goal_proj = torch.cat([_encode_frame(model, gv)[1] for gv in goal_views], dim=0)  # (V, D) z_proj
    n_cand = action_tensor.shape[0]
    z_pred = model.plan_rollout(z_start_raw.repeat(n_cand, 1), action_tensor)
    z_pred_last = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred
    per_view = torch.stack(
        [pose_head(z_pred_last, z_goal_proj[v:v + 1].repeat(n_cand, 1))[:, :2].norm(dim=-1)
         for v in range(z_goal_proj.shape[0])],
        dim=0,
    )  # (V, n_cand) predicted distance-to-goal; min = "reach from whichever side"
    cost = per_view.min(dim=0).values
    best_idx = int(torch.argmin(cost).item())
    return sequences[best_idx][0], float(cost[best_idx].detach().cpu().item())


def _choose_bearing_primitive(
    build: Any,
    target_xy: tuple[float, float],
) -> str:
    pos, quat = _current_pose(build)
    yaw = _yaw_from_quat_wxyz(quat)
    bearing = math.atan2(float(target_xy[1]) - float(pos[1]), float(target_xy[0]) - float(pos[0]))
    return primitive_toward_bearing(heading_error_rad=wrap_angle_pi(bearing - yaw))


def _line_of_sight_to_beacon(
    pack: Any,
    src_xy: tuple[float, float],
    landmark_xy: tuple[float, float],
) -> bool:
    los = getattr(pack.scene_graph, "has_line_of_sight", None)
    if los is None:
        return True
    return bool(
        los(
            (float(src_xy[0]), float(src_xy[1])),
            (float(landmark_xy[0]), float(landmark_xy[1])),
            margin_m=0.02,
            exclude_landmark_xy=(float(landmark_xy[0]), float(landmark_xy[1])),
        )
    )


def _render_approach_views(
    build: Any,
    pack: Any,
    landmark_xy: tuple[float, float],
    grid: InflatedOccupancyGrid,
    standoff_m: float,
    n_views: int,
    base_z: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Render the beacon from N evenly-spaced approach directions (each facing it).

    Multi-view goal latents make image-goal servoing robust to the heading the
    robot actually arrives from — the LeJEPA latent is heading-dominated, so a
    single goal photo only matches one approach heading.
    """
    lx, ly = float(landmark_xy[0]), float(landmark_xy[1])
    imgs = []
    for k in range(int(n_views)):
        ang = 2.0 * math.pi * k / float(n_views)
        vx = lx + float(standoff_m) * math.cos(ang)
        vy = ly + float(standoff_m) * math.sin(ang)
        if not grid.is_free((vx, vy)):
            continue
        if not _line_of_sight_to_beacon(pack, (vx, vy), (lx, ly)):
            continue
        vyaw = math.atan2(ly - vy, lx - vx)
        imgs.append(_render_tensor_from_base(
            build, pack,
            base_xyz_m=np.asarray([vx, vy, base_z], dtype=np.float32),
            base_quat_wxyz=_quat_wxyz_from_yaw(vyaw),
            device=device,
        ))
    if not imgs:
        return None
    return torch.stack(imgs, dim=0)


def _select_visible_beacon_setup(
    pack: Any,
    rng: random.Random,
    *,
    device: torch.device,
    build: Any,
    grid: InflatedOccupancyGrid,
    approach_distance_m: float,
    goal_standoff_m: float,
    start_yaw_jitter_rad: float,
    n_goal_views: int = 0,
    goal_yaw_offset_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, GoalSpec]:
    graph = pack.scene_graph
    landmarks = list(graph.landmark_cells)
    if not landmarks:
        raise RuntimeError(f"scene {pack.scene_id} has no landmarks to target")
    rng.shuffle(landmarks)
    angles = [2.0 * math.pi * idx / 64.0 for idx in range(64)]
    rng.shuffle(angles)
    for object_id, cell_id in landmarks:
        landmark_xy = graph.landmark_xy_for_cell(cell_id) or graph.cell_center(cell_id)
        lx, ly = float(landmark_xy[0]), float(landmark_xy[1])
        for angle in angles:
            ux, uy = math.cos(angle), math.sin(angle)
            target_xy = (
                lx + float(goal_standoff_m) * ux,
                ly + float(goal_standoff_m) * uy,
            )
            start_xy = (
                lx + (float(goal_standoff_m) + float(approach_distance_m)) * ux,
                ly + (float(goal_standoff_m) + float(approach_distance_m)) * uy,
            )
            if not grid.is_free(target_xy) or not grid.is_free(start_xy):
                continue
            if not grid.has_free_line(start_xy, target_xy):
                continue
            if not _line_of_sight_to_beacon(pack, start_xy, (lx, ly)):
                continue
            if not _line_of_sight_to_beacon(pack, target_xy, (lx, ly)):
                continue

            target_yaw = math.atan2(ly - target_xy[1], lx - target_xy[0])
            start_yaw = math.atan2(ly - start_xy[1], lx - start_xy[0])
            if start_yaw_jitter_rad:
                start_yaw = wrap_angle_pi(
                    start_yaw + rng.uniform(-float(start_yaw_jitter_rad), float(start_yaw_jitter_rad))
                )
            base_z = float(pack.robot.spawn_xyz_m[2])
            # goal_yaw_offset_rad rotates the goal *image* away from facing the
            # beacon while keeping the goal *position* (success criterion) fixed.
            # Tests whether image-goal servoing depends on a goal-facing view.
            goal_render_yaw = wrap_angle_pi(target_yaw + float(goal_yaw_offset_rad))
            goal_img = _render_tensor_from_base(
                build,
                pack,
                base_xyz_m=np.asarray([target_xy[0], target_xy[1], base_z], dtype=np.float32),
                base_quat_wxyz=_quat_wxyz_from_yaw(goal_render_yaw),
                device=device,
            )
            approach_imgs = None
            if int(n_goal_views) > 0:
                approach_imgs = _render_approach_views(
                    build, pack, (lx, ly), grid, goal_standoff_m,
                    int(n_goal_views), base_z, device,
                )
                # Always include the primary target view first.
                approach_imgs = (
                    goal_img[None] if approach_imgs is None
                    else torch.cat([goal_img[None], approach_imgs], dim=0)
                )
            goal = GoalSpec(
                object_id=str(object_id),
                landmark_xy=(lx, ly),
                target_xy=(float(target_xy[0]), float(target_xy[1])),
                target_yaw_rad=float(target_yaw),
                image=goal_img,
                approach_images=approach_imgs,
            )
            start_pos = np.asarray([start_xy[0], start_xy[1], base_z], dtype=np.float32)
            start_quat = _quat_wxyz_from_yaw(start_yaw)
            return start_pos, start_quat, goal
    raise RuntimeError(f"scene {pack.scene_id} has no visible-beacon approach setup")


def _select_goal(
    pack: Any,
    rng: random.Random,
    *,
    start_xy: tuple[float, float],
    device: torch.device,
    build: Any,
    min_initial_distance_m: float,
    goal_standoff_m: float,
) -> GoalSpec:
    graph = pack.scene_graph
    grid = InflatedOccupancyGrid(
        pack.scene_graph.manifest,
        cell_size_m=PLANNING_GRID_CELL_SIZE_M,
        inflation_m=PLANNING_GRID_INFLATION_M,
    )
    landmarks = list(graph.landmark_cells)
    if not landmarks:
        raise RuntimeError(f"scene {pack.scene_id} has no landmarks to target")
    rng.shuffle(landmarks)
    for object_id, cell_id in landmarks:
        landmark_xy = graph.landmark_xy_for_cell(cell_id) or graph.cell_center(cell_id)
        if _xy_distance(start_xy, landmark_xy) < float(min_initial_distance_m):
            continue
        target = _standoff_target(grid, landmark_xy, start_xy, float(goal_standoff_m))
        if target is None:
            continue
        target_xy, yaw = target
        base_xyz = np.asarray([target_xy[0], target_xy[1], pack.robot.spawn_xyz_m[2]], dtype=np.float32)
        goal_img = _render_tensor_from_base(
            build,
            pack,
            base_xyz_m=base_xyz,
            base_quat_wxyz=_quat_wxyz_from_yaw(yaw),
            device=device,
        )
        return GoalSpec(
            object_id=str(object_id),
            landmark_xy=(float(landmark_xy[0]), float(landmark_xy[1])),
            target_xy=(float(target_xy[0]), float(target_xy[1])),
            target_yaw_rad=float(yaw),
            image=goal_img,
        )
    raise RuntimeError(f"scene {pack.scene_id} has no usable landmark goal")


def _standoff_target(
    grid: InflatedOccupancyGrid,
    landmark_xy: tuple[float, float],
    start_xy: tuple[float, float],
    standoff_m: float,
) -> tuple[tuple[float, float], float] | None:
    lx, ly = float(landmark_xy[0]), float(landmark_xy[1])
    sx, sy = float(start_xy[0]), float(start_xy[1])
    base_angle = math.atan2(sy - ly, sx - lx)
    candidates: list[tuple[float, tuple[float, float], float]] = []
    for idx in range(24):
        angle = base_angle + (idx // 2) * (math.pi / 12.0) * (-1.0 if idx % 2 else 1.0)
        xy = (lx + standoff_m * math.cos(angle), ly + standoff_m * math.sin(angle))
        snapped = grid.nearest_free(xy, max_radius_m=0.35)
        if snapped is None:
            continue
        yaw = math.atan2(ly - snapped[1], lx - snapped[0])
        candidates.append((_xy_distance(snapped, start_xy), snapped, yaw))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    _dist, xy, yaw = candidates[0]
    return xy, yaw


def _run_policy_trial(
    *,
    policy_name: str,
    model: torch.nn.Module,
    build: Any,
    pack: Any,
    runner: RolloutRunner | None,
    registry: PrimitiveRegistry,
    goal: GoalSpec,
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
    primitive_names: list[str],
    max_blocks: int,
    goal_radius_m: float,
    fall_z_threshold_m: float,
    rng: random.Random,
    device: torch.device,
    command_dt_s: float,
    grid: InflatedOccupancyGrid | None,
    oracle_shuffle_seed: int = 0,
    dino_encoder: Any | None = None,
    dino_device: torch.device | None = None,
    dino_provenance: dict[str, Any] | None = None,
    frame_sink: list | None = None,
    third_person_build: Any | None = None,
) -> PolicyResult:
    _set_pose(build=build, runner=runner, pos_xyz=start_pos, quat_wxyz=start_quat)
    initial_pos, _initial_quat = _current_pose(build)
    prev_xy = np.asarray(initial_pos[:2], dtype=np.float64)
    initial_distance = _xy_distance(prev_xy, goal.target_xy)
    path_length = 0.0
    primitives: list[str] = []
    plan_costs: list[float] = []
    decision_log: list[dict[str, Any]] = []
    fell = False
    dino_goal_tokens: torch.Tensor | None = None
    if policy_name in DINO_POLICIES:
        if dino_encoder is None or dino_device is None or dino_provenance is None:
            raise ValueError("DINO policy requires encoder, device, and provenance")
        if goal.approach_images is not None:
            raise ValueError("DINO ceiling policies support one goal image only")
        dino_goal_tokens = _encode_dinov2_images(
            dino_encoder,
            goal.image,
            device=dino_device,
        )

    for block_idx in range(int(max_blocks)):
        pos, quat = _current_pose(build)
        if float(pos[2]) < float(fall_z_threshold_m):
            fell = True
            break
        if _xy_distance(pos[:2], goal.target_xy) <= float(goal_radius_m):
            break

        yaw = _yaw_from_quat_wxyz(quat)
        oracle_costs = _oracle_candidate_costs(
            sequences,
            registry,
            grid,
            pos[:2],
            yaw,
            command_dt_s,
            goal.target_xy,
        )
        oracle_ranking = _oracle_ranking(oracle_costs, sequences)
        oracle_best_index = int(oracle_ranking["best_candidate_index"])
        oracle_best_cost = float(oracle_ranking["best_cost_m"])
        selected_candidate_index: int | None = None
        selected_score_source_index: int | None = None
        selected_policy_score: float | None = None
        policy_score_margin: float | None = None
        dino_unshuffled_costs: np.ndarray | None = None
        dino_policy_scores: np.ndarray | None = None
        dino_score_sources: np.ndarray | None = None

        if policy_name in ("lewm", "lewm_pose"):
            image = _render_tensor_from_base(
                build,
                pack,
                base_xyz_m=pos,
                base_quat_wxyz=quat,
                device=device,
            )
            goal_img = goal.approach_images if goal.approach_images is not None else goal.image
            if policy_name == "lewm":
                primitive_name, cost = _choose_lewm_primitive(
                    model, image, goal_img, sequences, action_tensor,
                )
            else:
                primitive_name, cost = _choose_lewm_pose_primitive(
                    model, model._pose_head, image, goal_img, sequences, action_tensor,
                )
            selected_policy_score = float(cost)
            plan_costs.append(cost)
        elif policy_name in ("oracle_mpc", "oracle_shuffled"):
            score_sources = np.arange(len(sequences), dtype=np.int64)
            policy_scores = oracle_costs
            if policy_name == "oracle_shuffled":
                score_sources = _deterministic_score_permutation(
                    len(sequences),
                    seed=oracle_shuffle_seed,
                    block_index=block_idx,
                )
                policy_scores = oracle_costs[score_sources]
            policy_ranking = _oracle_ranking(policy_scores, sequences)
            selected_candidate_index = int(policy_ranking["best_candidate_index"])
            selected_score_source_index = int(score_sources[selected_candidate_index])
            primitive_name = sequences[selected_candidate_index][0]
            selected_policy_score = float(policy_scores[selected_candidate_index])
            if len(policy_scores) > 1:
                two_smallest = np.partition(policy_scores, 1)[:2]
                policy_score_margin = float(two_smallest[1] - two_smallest[0])
            plan_costs.append(selected_policy_score)
        elif policy_name in DINO_POLICIES:
            assert dino_encoder is not None
            assert dino_device is not None
            assert dino_goal_tokens is not None
            if policy_name in DINO_TRUE_SUCCESSOR_POLICIES:
                if runner is not None:
                    raise ValueError("DINO true-successor policies require kinematic mode")
                dino_unshuffled_costs = _dino_true_successor_candidate_costs(
                    encoder=dino_encoder,
                    goal_tokens=dino_goal_tokens,
                    dino_device=dino_device,
                    build=build,
                    pack=pack,
                    registry=registry,
                    sequences=sequences,
                    start_pos=np.asarray(pos, dtype=np.float32).copy(),
                    start_quat=np.asarray(quat, dtype=np.float32).copy(),
                    command_dt_s=command_dt_s,
                    grid=grid,
                )
            else:
                current_image = _render_tensor_from_base(
                    build,
                    pack,
                    base_xyz_m=pos,
                    base_quat_wxyz=quat,
                    device=dino_device,
                )
                current_tokens = _encode_dinov2_images(
                    dino_encoder,
                    current_image,
                    device=dino_device,
                )
                dino_unshuffled_costs = _dino_persistence_candidate_costs(
                    current_tokens,
                    dino_goal_tokens,
                    candidate_count=len(sequences),
                )
            dino_ranking = _rank_dino_policy_scores(
                dino_unshuffled_costs,
                policy_name=policy_name,
                seed=oracle_shuffle_seed,
                block_index=block_idx,
            )
            dino_policy_scores = dino_ranking["policy_candidate_scores"]
            dino_score_sources = dino_ranking["score_source_candidate_indices"]
            selected_candidate_index = int(
                dino_ranking["selected_candidate_index"]
            )
            selected_score_source_index = int(
                dino_ranking["selected_score_source_candidate_index"]
            )
            selected_policy_score = float(dino_ranking["selected_policy_score"])
            policy_score_margin = dino_ranking["policy_score_margin"]
            primitive_name = sequences[selected_candidate_index][0]
            plan_costs.append(selected_policy_score)
        elif policy_name == "bearing":
            primitive_name = _choose_bearing_primitive(build, goal.target_xy)
            if primitive_name not in primitive_names:
                primitive_name = "forward_medium" if "forward_medium" in primitive_names else primitive_names[0]
        elif policy_name == "hold":
            primitive_name = "hold"
        elif policy_name == "random":
            primitive_name = rng.choice(primitive_names)
        else:
            raise ValueError(f"unknown policy_name={policy_name!r}")

        first_action_assessment = _oracle_first_action_assessment(
            oracle_costs,
            sequences,
            primitive_name,
            oracle_best_cost_m=oracle_best_cost,
        )
        decision = {
                "block_index": int(block_idx),
                "selected_primitive": primitive_name,
                "selected_candidate_index": selected_candidate_index,
                "selected_candidate_sequence": (
                    list(sequences[selected_candidate_index])
                    if selected_candidate_index is not None
                    else None
                ),
                "selected_policy_score": selected_policy_score,
                "policy_score_margin": policy_score_margin,
                "selected_score_source_candidate_index": selected_score_source_index,
                "selected_candidate_true_cost_m": (
                    float(oracle_costs[selected_candidate_index])
                    if selected_candidate_index is not None
                    else None
                ),
                "selected_candidate_oracle_regret_m": (
                    float(oracle_costs[selected_candidate_index] - oracle_best_cost)
                    if selected_candidate_index is not None
                    else None
                ),
                "oracle_best_candidate_index": oracle_best_index,
                "oracle_best_candidate_sequence": list(sequences[oracle_best_index]),
                "oracle_best_first_primitive": sequences[oracle_best_index][0],
                "oracle_best_cost_m": oracle_best_cost,
                "oracle_optimal_candidate_indices": oracle_ranking[
                    "optimal_candidate_indices"
                ],
                "oracle_optimal_candidate_count": oracle_ranking[
                    "optimal_candidate_count"
                ],
                "oracle_optimal_first_primitives": oracle_ranking[
                    "optimal_first_primitives"
                ],
                "oracle_cost_tie_tolerance_m": oracle_ranking[
                    "tie_tolerance_m"
                ],
                "oracle_candidate_tie_break": oracle_ranking["tie_break"],
                "selected_first_action_best_candidate_index": (
                    first_action_assessment["best_candidate_index"]
                ),
                "selected_first_action_best_cost_m": first_action_assessment[
                    "best_cost_m"
                ],
                "oracle_first_action_regret_m": first_action_assessment[
                    "regret_m"
                ],
                "oracle_first_action_disagreement": first_action_assessment[
                    "disagreement"
                ],
            }
        if policy_name in DINO_POLICIES:
            assert dino_unshuffled_costs is not None
            assert dino_policy_scores is not None
            assert dino_score_sources is not None
            assert dino_provenance is not None
            decision.update(
                {
                    "policy_candidate_scores": [
                        float(value) for value in dino_policy_scores
                    ],
                    "unshuffled_dino_candidate_costs": [
                        float(value) for value in dino_unshuffled_costs
                    ],
                    "score_source_candidate_indices": [
                        int(value) for value in dino_score_sources
                    ],
                    "dino_cost_definition": DINO_COST_DEFINITION,
                    "dino_checkpoint_sha256": dino_provenance[
                        "checkpoint_sha256"
                    ],
                }
            )
        decision_log.append(decision)

        if runner is None:
            _execute_kinematic_primitive(
                build,
                registry,
                primitive_name,
                command_dt_s=command_dt_s,
                grid=grid,
                frame_sink=frame_sink,
                pack=pack,
                device=device,
                third_person_build=third_person_build,
            )
        else:
            _execute_physical_primitive(
                runner, registry, primitive_name,
                frame_sink=frame_sink, build=build, pack=pack, device=device,
                third_person_build=third_person_build,
            )
        primitives.append(primitive_name)
        new_pos, _new_quat = _current_pose(build)
        new_xy = np.asarray(new_pos[:2], dtype=np.float64)
        path_length += float(np.linalg.norm(new_xy - prev_xy))
        prev_xy = new_xy

    final_pos, _final_quat = _current_pose(build)
    final_distance = _xy_distance(final_pos[:2], goal.target_xy)
    progress = initial_distance - final_distance
    efficiency = progress / path_length if path_length > 1e-6 else 0.0
    return PolicyResult(
        policy=policy_name,
        scene_id=str(pack.scene_id),
        goal_object_id=goal.object_id,
        initial_distance_m=initial_distance,
        final_distance_m=final_distance,
        progress_m=progress,
        path_length_m=path_length,
        path_efficiency=efficiency,
        success=final_distance <= float(goal_radius_m),
        fell=fell or float(final_pos[2]) < float(fall_z_threshold_m),
        blocks_executed=len(primitives),
        primitive_sequence=primitives,
        mean_plan_cost=(float(np.mean(plan_costs)) if plan_costs else None),
        decision_log=decision_log,
    )


def _aggregate(results: list[PolicyResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[PolicyResult]] = {}
    for result in results:
        grouped.setdefault(result.policy, []).append(result)
    out: dict[str, dict[str, float | int]] = {}
    for policy, rows in sorted(grouped.items()):
        out[policy] = {
            "trials": len(rows),
            "success_rate": float(np.mean([r.success for r in rows])) if rows else 0.0,
            "fall_rate": float(np.mean([r.fell for r in rows])) if rows else 0.0,
            "mean_initial_distance_m": float(np.mean([r.initial_distance_m for r in rows])),
            "mean_final_distance_m": float(np.mean([r.final_distance_m for r in rows])),
            "mean_progress_m": float(np.mean([r.progress_m for r in rows])),
            "mean_path_length_m": float(np.mean([r.path_length_m for r in rows])),
            "mean_path_efficiency": float(np.mean([r.path_efficiency for r in rows])),
        }
    return out


def _to_jsonable(result: PolicyResult) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "scene_id": result.scene_id,
        "goal_object_id": result.goal_object_id,
        "initial_distance_m": result.initial_distance_m,
        "final_distance_m": result.final_distance_m,
        "progress_m": result.progress_m,
        "path_length_m": result.path_length_m,
        "path_efficiency": result.path_efficiency,
        "success": result.success,
        "fell": result.fell,
        "blocks_executed": result.blocks_executed,
        "primitive_sequence": result.primitive_sequence,
        "mean_plan_cost": result.mean_plan_cost,
        "decision_log": result.decision_log,
    }


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, default=REPO_ROOT / ".generated" / "scene_corpus" / "minimum_20260520T080420Z")
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config" / "go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path, default=REPO_ROOT / "config" / "go2_primitive_registry.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--demo-video",
        type=Path,
        default=None,
        help="If set, run only the lewm policy, capture egocentric+third-person "
        "frames, and write an MP4 of the first successful episode to this path.",
    )
    parser.add_argument("--demo-fps", type=float, default=12.0)
    parser.add_argument(
        "--demo-beacons",
        type=int,
        default=1,
        help="If >1 (with --demo-video), the robot chases that many scene "
        "landmarks in sequence (claim one, then the next) in a single episode.",
    )
    parser.add_argument(
        "--demo-perception",
        action="store_true",
        help="Pure-perception multi-beacon: image-goal servo to each beacon and "
        "scan (rotate) when it is not in view, instead of privileged breadcrumb "
        "waypoints. Only the per-beacon goal keyframe and claim radius are privileged.",
    )
    parser.add_argument("--demo-scan-thresh", type=float, default=0.02)
    parser.add_argument(
        "--demo-commit-margin",
        type=float,
        default=0.01,
        help="Primitive commitment hysteresis for the beacon demo: keep the "
        "previous primitive while its cost is within this fraction of the "
        "best (kills near-tie flip-flop stutter; too high locks onto stale "
        "headings — 0.03 measurably loses legs). 0 disables.",
    )
    parser.add_argument(
        "--demo-extra-beacons",
        type=int,
        default=0,
        help="Demo-only: inject this many extra coloured landmark pillars at "
        "the scene's unused corner/edge slots (standard landmark palette).",
    )
    parser.add_argument(
        "--demo-clear-obstacles",
        action="store_true",
        help="Demo-only: strip free-standing obstacle boxes from the scene "
        "(walls and landmarks stay) to declutter the stage.",
    )
    parser.add_argument(
        "--render-robot",
        action="store_true",
        help=(
            "Legacy/debug mode: render Go2 visual meshes in the main camera "
            "scene, including egocentric RGB. By default ego RGB hides the "
            "robot body; demo videos use a separate robot-visible scene for "
            "the third-person panel."
        ),
    )
    parser.add_argument(
        "--demo-max-frames",
        type=int,
        default=900,
        help="Subsample the demo to at most this many frames (time-compresses "
        "playback; 12 fps shows ~1 control tick per frame at the default).",
    )
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--family", default=None)
    parser.add_argument("--scene-limit", type=int, default=1)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--trials-per-scene", type=int, default=1)
    parser.add_argument(
        "--task",
        choices=("landmark", "visible-beacon"),
        default="landmark",
        help=(
            "landmark: sample a landmark goal from the scene graph. "
            "visible-beacon: construct a local open-space approach with direct "
            "line of sight to the beacon."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("kinematic", "physical"),
        default="kinematic",
        help=(
            "kinematic: update base pose from primitive commands and render from it. "
            "physical: run the Go2 low-level policy in Genesis."
        ),
    )
    parser.add_argument("--backend", default="cpu")
    parser.add_argument(
        "--apply-textures",
        action="store_true",
        help="Render with per-scene CC0 textures (matches render_textured_v03 "
        "training distribution). Needs a texture-capable backend, e.g. "
        "--backend vulkan. Default off renders untextured box geometry, which "
        "is out-of-distribution for textured-trained checkpoints.",
    )
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument(
        "--dino-repo",
        type=Path,
        default=None,
        help="Exact local DINOv2 repository (required only by dino_* policies).",
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=Path,
        default=None,
        help="Exact frozen dinov2_vits14 checkpoint (required only by dino_* policies).",
    )
    parser.add_argument(
        "--dino-device",
        default=None,
        help="Torch device for frozen DINO encoding (required only by dino_* policies).",
    )
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--goal-radius-m", type=float, default=0.55)
    parser.add_argument("--goal-standoff-m", type=float, default=0.85)
    parser.add_argument("--min-initial-distance-m", type=float, default=1.5)
    parser.add_argument("--beacon-approach-distance-m", type=float, default=1.5)
    parser.add_argument("--beacon-start-yaw-jitter-rad", type=float, default=0.0)
    parser.add_argument(
        "--goal-yaw-offset-rad",
        type=float,
        default=0.0,
        help="Rotate the goal IMAGE away from facing the beacon (goal position "
        "and success criterion unchanged). 0 = goal-facing (default); pi = goal "
        "image looks away from the beacon. Tests whether image-goal servoing "
        "depends on a goal-facing view convention.",
    )
    parser.add_argument(
        "--goal-views",
        type=int,
        default=0,
        help="Multi-view image goals: render the beacon from N approach directions "
        "and take min energy over views (0 = single front view).",
    )
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--primitive-names",
        default="hold,forward_medium,arc_left,arc_right,yaw_left,yaw_right,backward",
    )
    parser.add_argument(
        "--policies",
        default="lewm,bearing,hold,random",
        help=(
            "Comma-separated policies: lewm, lewm_pose, oracle_mpc, "
            "oracle_shuffled, dino_true_successor, "
            "dino_true_successor_shuffled, dino_persistence, bearing, hold, random."
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument(
        "--head-ckpt",
        type=Path,
        default=None,
        help="Optional GoalEnergyHead checkpoint; replaces L2 plan_cost for the lewm policy.",
    )
    parser.add_argument(
        "--pose-head-ckpt",
        type=Path,
        default=None,
        help="Optional RelPoseHead checkpoint; enables the lewm_pose policy "
        "(cost = predicted distance-to-goal with no privileged runtime geometry).",
    )
    parser.add_argument(
        "--allow-pose-multiview-goal-set",
        action="store_true",
        help="Allow lewm_pose with --goal-views > 0. This changes the target from "
        "one pose to a set of approach poses and is not the primary benchmark.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    device = torch.device(args.model_device)
    torch.set_grad_enabled(False)
    model, model_config = load_model(
        SimpleNamespace(
            checkpoint=args.checkpoint.resolve(),
            max_seq_len=args.max_seq_len,
            sigreg_lambda=args.sigreg_lambda,
        ),
        device,
    )

    if args.head_ckpt is not None:
        head_ck = torch.load(args.head_ckpt.resolve(), map_location=device, weights_only=False)
        head = GoalEnergyHead(
            latent_dim=int(head_ck.get("latent_dim", model.latent_dim)),
            hidden=int(head_ck.get("hidden", 1024)),
            dropout=0.0,
        ).to(device)
        head.load_state_dict(head_ck["head_state_dict"])
        head.eval()
        model._energy_head = head
        print(f"[lewm] using learned GoalEnergyHead cost from {args.head_ckpt} "
              f"(train ranking acc {head_ck.get('best_eval_ranking_acc', '?')})", flush=True)

    if args.pose_head_ckpt is not None:
        pck = torch.load(args.pose_head_ckpt.resolve(), map_location=device, weights_only=False)
        phead = RelPoseHead(
            latent_dim=int(pck.get("latent_dim", model.latent_dim)),
            hidden=int(pck.get("hidden", 512)),
        ).to(device)
        phead.load_state_dict(pck["head_state_dict"])
        phead.eval()
        model._pose_head = phead
        print(f"[lewm_pose] using RelPoseHead metric cost from {args.pose_head_ckpt} "
              f"(epoch {pck.get('epoch', '?')})", flush=True)

    primitive_names = _parse_csv(args.primitive_names)
    policies = _parse_csv(args.policies)
    if args.demo_video is not None:
        policies = ["lewm"]
    if "hold" not in primitive_names:
        raise SystemExit("--primitive-names must include hold")
    supported_policies = {
        "lewm",
        "lewm_pose",
        "oracle_mpc",
        "oracle_shuffled",
        "dino_true_successor",
        "dino_true_successor_shuffled",
        "dino_persistence",
        "bearing",
        "hold",
        "random",
    }
    unsupported = sorted(set(policies) - supported_policies)
    if unsupported:
        raise SystemExit(f"unsupported policies: {unsupported}")
    if int(args.horizon) < 1:
        raise SystemExit("--horizon must be >= 1")
    try:
        _validate_dino_policy_scope(
            policies,
            mode=str(args.mode),
            horizon=int(args.horizon),
            goal_views=int(args.goal_views),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dino_requested = bool(DINO_POLICIES.intersection(policies))
    if dino_requested and (
        args.dino_repo is None
        or args.dino_checkpoint is None
        or args.dino_device is None
    ):
        raise SystemExit(
            "dino_* policies require --dino-repo, --dino-checkpoint, and --dino-device"
        )
    if (
        "lewm_pose" in policies
        and int(args.goal_views) > 0
        and not args.allow_pose_multiview_goal_set
    ):
        raise SystemExit(
            "lewm_pose with --goal-views > 0 changes the goal-set semantics; "
            "pass --allow-pose-multiview-goal-set only for that explicit ablation"
        )

    platform = load_platform_manifest(args.platform_manifest.resolve())
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
    command_dt_s = float(platform.get("timing", {}).get("command_dt_s", 0.10))
    for primitive_name in primitive_names:
        registry.get(primitive_name)
    primitive_blocks = _primitive_active_blocks(registry, primitive_names)
    for name, active in primitive_blocks.items():
        if active.shape != (ACTIVE_BLOCK_DIM,):
            raise RuntimeError(f"primitive {name!r} encoded to {active.shape}, expected {(ACTIVE_BLOCK_DIM,)}")
        active_block_to_matrix(active)

    rng = random.Random(int(args.seed))
    sequences, action_tensor = _candidate_action_tensor(
        primitive_blocks,
        primitive_names,
        int(args.horizon),
        max_candidates=args.max_candidates,
        rng=rng,
        device=device,
    )

    dino_encoder: torch.nn.Module | None = None
    dino_device: torch.device | None = None
    dino_provenance: dict[str, Any] | None = None
    if dino_requested:
        assert args.dino_repo is not None
        assert args.dino_checkpoint is not None
        assert args.dino_device is not None
        dino_device = torch.device(args.dino_device)
        try:
            dino_encoder, dino_provenance = _load_dinov2_encoder(
                args.dino_repo,
                args.dino_checkpoint,
                dino_device,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"cannot load frozen DINOv2 encoder: {exc}") from exc
        print(
            f"[dino] using {DINO_ENCODER_NAME} on {dino_device} "
            f"checkpoint={dino_provenance['checkpoint_sha256']}",
            flush=True,
        )

    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family)
    scene_dirs = sorted(scene_dirs, key=lambda p: p.name)
    if args.scene_offset:
        scene_dirs = scene_dirs[int(args.scene_offset) :]
    if args.scene_limit:
        scene_dirs = scene_dirs[: int(args.scene_limit)]
    if not scene_dirs:
        raise SystemExit(
            f"no scenes found under {args.scene_corpus} for split={args.split!r} family={args.family!r}"
        )

    policy = None
    if args.mode == "physical":
        try:
            policy = GenesisGo2PPOPolicy.from_platform_manifest(
                platform,
                repo_root,
                device=str(args.policy_device),
            )
        except RuntimeError as exc:
            raise SystemExit(
                f"cannot load Genesis locomotion policy: {exc}\n"
                "Install/use the Genesis training venv with torch, tensordict, and rsl-rl-lib."
            ) from exc

    started = time.time()
    results: list[PolicyResult] = []
    skipped: list[dict[str, str]] = []

    for scene_index, scene_dir in enumerate(scene_dirs):
        pack = load_scene_pack(
            scene_dir,
            platform_manifest=platform,
            workspace_root=repo_root,
        )
        if args.demo_clear_obstacles:
            pack = _strip_obstacles(pack)
        if int(args.demo_extra_beacons) > 0:
            pack = _inject_extra_beacons(pack, int(args.demo_extra_beacons))
        print(
            f"[{scene_index + 1}/{len(scene_dirs)}] scene={pack.scene_id} "
            f"family={pack.family} split={pack.split}",
            flush=True,
        )
        try:
            build = build_scene_from_pack(
                pack,
                n_envs=1,
                backend=str(args.backend),
                show_viewer=False,
                render_robot=bool(args.render_robot),
                apply_textures=bool(args.apply_textures),
            )
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
            runner: RolloutRunner | None = None
            if args.mode == "physical":
                if policy is None:
                    raise RuntimeError("physical mode requested but policy failed to load")
                config = RolloutConfig(
                    n_blocks=int(args.max_blocks),
                    fall_z_threshold_m=float(args.fall_z_threshold_m),
                    rgb_capture_per_block=False,
                    seed=int(args.seed) + scene_index,
                    log_progress_every_blocks=0,
                    foot_contact_source="zero",
                    randomize_spawn_pose=True,
                )
                runner = RolloutRunner(build, policy, registry, safety, config=config)
            grid = InflatedOccupancyGrid(
                pack.scene_graph.manifest,
                cell_size_m=PLANNING_GRID_CELL_SIZE_M,
                inflation_m=PLANNING_GRID_INFLATION_M,
            )
            if args.demo_video is not None and int(args.demo_beacons) > 1:
                demo_frames: list = []
                demo_goal_insets: list = []
                if args.demo_perception:
                    claimed = _run_perception_multibeacon_demo(
                        model=model,
                        build=build,
                        pack=pack,
                        registry=registry,
                        sequences=sequences,
                        action_tensor=action_tensor,
                        n_beacons=int(args.demo_beacons),
                        goal_standoff_m=float(args.goal_standoff_m),
                        goal_radius_m=float(args.goal_radius_m),
                        max_blocks=int(args.max_blocks),
                        command_dt_s=command_dt_s,
                        device=device,
                        grid=grid,
                        frame_sink=demo_frames,
                        scan_thresh=float(args.demo_scan_thresh),
                        runner=runner,
                        third_person_build=third_person_build,
                    )
                else:
                    claimed = _run_multi_beacon_demo(
                        model=model,
                        build=build,
                        pack=pack,
                        registry=registry,
                        sequences=sequences,
                        action_tensor=action_tensor,
                        n_beacons=int(args.demo_beacons),
                        goal_standoff_m=float(args.goal_standoff_m),
                        goal_radius_m=float(args.goal_radius_m),
                        approach_distance_m=float(args.beacon_approach_distance_m),
                        max_blocks=int(args.max_blocks),
                        command_dt_s=command_dt_s,
                        device=device,
                        grid=grid,
                        frame_sink=demo_frames,
                        runner=runner,
                        goal_insets=demo_goal_insets,
                        commit_margin=float(args.demo_commit_margin),
                        third_person_build=third_person_build,
                    )
                print(f"    [demo] claimed {claimed}/{int(args.demo_beacons)} beacons, frames={len(demo_frames)}", flush=True)
                playback = 1
                if len(demo_frames) > int(args.demo_max_frames):
                    playback = int(math.ceil(len(demo_frames) / int(args.demo_max_frames)))
                    demo_frames = demo_frames[::playback]
                    demo_goal_insets = [(idx // playback, img, name)
                                        for idx, img, name in demo_goal_insets]
                # Prefer a scene that claims all beacons; otherwise accept the
                # first substantial traversal (claims >=1 beacon and keeps going).
                full = claimed >= int(args.demo_beacons)
                if demo_frames and (full or (claimed >= 1 and len(demo_frames) >= 200)):
                    mb_title = (
                        "LeWM seq4 | perception servo + scan (no breadcrumbs)"
                        if args.demo_perception
                        else "LeWM seq4 | beacon tour: image-goal servo (route subgoals privileged)"
                    )
                    if playback > 1:
                        mb_title += f" | {playback}x"
                    _write_hud_video(
                        args.demo_video, pack, grid, demo_frames, float(args.demo_fps),
                        float(args.goal_radius_m) + float(args.goal_standoff_m), mb_title,
                        goal_insets=demo_goal_insets,
                    )
                    print(f"[demo] wrote {len(demo_frames)} frames ({claimed}/{int(args.demo_beacons)} claimed) -> {args.demo_video}", flush=True)
                    return 0
                continue
            for trial_idx in range(int(args.trials_per_scene)):
                if args.task == "visible-beacon":
                    start_pos, start_quat, goal = _select_visible_beacon_setup(
                        pack,
                        random.Random(int(args.seed) + scene_index * 1000 + trial_idx),
                        device=device,
                        build=build,
                        grid=grid,
                        approach_distance_m=float(args.beacon_approach_distance_m),
                        goal_standoff_m=float(args.goal_standoff_m),
                        start_yaw_jitter_rad=float(args.beacon_start_yaw_jitter_rad),
                        n_goal_views=int(args.goal_views),
                        goal_yaw_offset_rad=float(args.goal_yaw_offset_rad),
                    )
                    _set_pose(
                        build=build,
                        runner=runner,
                        pos_xyz=start_pos,
                        quat_wxyz=start_quat,
                    )
                elif runner is None:
                    restrict = pack.scene_graph.canonical_spawn_cells(
                        clearance_floor_m=0.20,
                        standoff_m=float(args.goal_standoff_m),
                    )
                    spawn_rng = random.Random(int(args.seed) + scene_index * 1000 + trial_idx)
                    start_xyz, start_wxyz, _cell_id = pack.scene_graph.sample_spawn_pose(
                        spawn_rng,
                        spawn_z_m=float(pack.robot.spawn_xyz_m[2]),
                        restrict_to_cells=restrict or None,
                    )
                    _set_pose(
                        build=build,
                        runner=None,
                        pos_xyz=np.asarray(start_xyz, dtype=np.float32),
                        quat_wxyz=np.asarray(start_wxyz, dtype=np.float32),
                    )
                else:
                    runner._reset_robot_to_spawn(None)
                if args.task != "visible-beacon":
                    start_pos, start_quat = _current_pose(build)
                    goal = _select_goal(
                        pack,
                        rng,
                        start_xy=(float(start_pos[0]), float(start_pos[1])),
                        device=device,
                        build=build,
                        min_initial_distance_m=float(args.min_initial_distance_m),
                        goal_standoff_m=float(args.goal_standoff_m),
                    )
                print(
                    f"  trial={trial_idx} goal={goal.object_id} "
                    f"target=({goal.target_xy[0]:.2f},{goal.target_xy[1]:.2f})",
                    flush=True,
                )
                for policy_name in policies:
                    demo_frames = [] if args.demo_video is not None else None
                    result = _run_policy_trial(
                        policy_name=policy_name,
                        model=model,
                        build=build,
                        pack=pack,
                        runner=runner,
                        registry=registry,
                        goal=goal,
                        start_pos=start_pos,
                        start_quat=start_quat,
                        sequences=sequences,
                        action_tensor=action_tensor,
                        primitive_names=primitive_names,
                        max_blocks=int(args.max_blocks),
                        goal_radius_m=float(args.goal_radius_m),
                        fall_z_threshold_m=float(args.fall_z_threshold_m),
                        rng=rng,
                        device=device,
                        command_dt_s=command_dt_s,
                        grid=grid,
                        oracle_shuffle_seed=int(args.seed),
                        dino_encoder=dino_encoder,
                        dino_device=dino_device,
                        dino_provenance=dino_provenance,
                        frame_sink=demo_frames,
                        third_person_build=third_person_build,
                    )
                    results.append(result)
                    print(
                        f"    {policy_name}: final={result.final_distance_m:.2f}m "
                        f"progress={result.progress_m:.2f}m success={int(result.success)} "
                        f"blocks={result.blocks_executed}",
                        flush=True,
                    )
                    if demo_frames and result.success:
                        _write_hud_video(
                            args.demo_video, pack, grid, demo_frames, float(args.demo_fps),
                            float(args.goal_radius_m), "LeWM seq4 | image-goal servoing to a visible beacon",
                        )
                        print(f"[demo] wrote {len(demo_frames)} frames -> {args.demo_video}", flush=True)
                        return 0
        except Exception as exc:
            skipped.append({"scene": str(scene_dir), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[SKIP] {scene_dir}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    summary = {
        "schema": "lewm_closed_loop_mpc_benchmark_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_config": model_config,
        "scene_corpus": str(args.scene_corpus.resolve()),
        "split": args.split,
        "family": args.family,
        "scene_count": len(scene_dirs),
        "trials_per_scene": int(args.trials_per_scene),
        "backend": args.backend,
        "mode": args.mode,
        "model_device": args.model_device,
        "policy_device": args.policy_device,
        "task": args.task,
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "candidate_count": len(sequences),
        "max_candidates": (
            None if args.max_candidates is None else int(args.max_candidates)
        ),
        "candidate_sequences": [list(sequence) for sequence in sequences],
        "planning_grid": {
            "cell_size_m": PLANNING_GRID_CELL_SIZE_M,
            "inflation_m": PLANNING_GRID_INFLATION_M,
        },
        "oracle_assay": _oracle_assay_provenance(
            seed=int(args.seed),
            max_candidates=args.max_candidates,
            sequences=sequences,
            mode=str(args.mode),
        ),
        "primitive_names": primitive_names,
        "policies": policies,
        "max_blocks": int(args.max_blocks),
        "goal_radius_m": float(args.goal_radius_m),
        "goal_standoff_m": float(args.goal_standoff_m),
        "min_initial_distance_m": float(args.min_initial_distance_m),
        "beacon_approach_distance_m": float(args.beacon_approach_distance_m),
        "beacon_start_yaw_jitter_rad": float(args.beacon_start_yaw_jitter_rad),
        "elapsed_s": time.time() - started,
        "aggregate": _aggregate(results),
        "results": [_to_jsonable(row) for row in results],
        "skipped": skipped,
    }
    if dino_provenance is not None:
        summary["dino_assay"] = dino_provenance
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
