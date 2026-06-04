#!/usr/bin/env python3
"""Closed-loop Genesis MPC benchmark for flat LeWM checkpoints.

This is the paper-style planning-readiness metric for this repo: use the
checkpoint as a latent cost model inside receding-horizon control, execute only
the first primitive block, reobserve, and replan. The output is JSON with
physical success/progress metrics, not latent MSE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
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

from lewm.actions import ACTIVE_BLOCK_DIM, active_block_to_matrix, encode_active_block  # noqa: E402
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


@dataclass(frozen=True)
class GoalSpec:
    object_id: str
    landmark_xy: tuple[float, float]
    target_xy: tuple[float, float]
    target_yaw_rad: float
    image: torch.Tensor


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


def _execute_physical_primitive(
    runner: RolloutRunner,
    registry: PrimitiveRegistry,
    primitive_name: str,
) -> np.ndarray:
    requested = expand_primitive_to_block(registry, primitive_name)
    clipped = runner._clip_block(requested[None, :, :]).executed[0]
    for tick in clipped:
        runner._step_command_tick(tick[None, :])
    runner._last_executed[0] = clipped[-1]
    return clipped


def _execute_kinematic_primitive(
    build: Any,
    registry: PrimitiveRegistry,
    primitive_name: str,
    *,
    command_dt_s: float,
    grid: InflatedOccupancyGrid | None,
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
    build.robot.set_pos(pos[None, :], envs_idx=[0], zero_velocity=True)
    build.robot.set_quat(_quat_wxyz_from_yaw(yaw)[None, :], envs_idx=[0], zero_velocity=False)
    return block


def _primitive_active_blocks(
    registry: PrimitiveRegistry,
    primitive_names: list[str],
) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for name in primitive_names:
        matrix = expand_primitive_to_block(registry, name)
        encoded[name] = encode_active_block(matrix[:, 0], matrix[:, 1], matrix[:, 2])
    return encoded


def _candidate_action_tensor(
    primitive_blocks: dict[str, np.ndarray],
    primitive_names: list[str],
    horizon: int,
    *,
    max_candidates: int | None,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[tuple[str, ...]], torch.Tensor]:
    all_sequences = list(itertools.product(primitive_names, repeat=int(horizon)))
    if max_candidates is not None and len(all_sequences) > int(max_candidates):
        all_sequences = rng.sample(all_sequences, int(max_candidates))
    actions = np.stack(
        [
            np.stack([primitive_blocks[name] for name in seq], axis=0)
            for seq in all_sequences
        ],
        axis=0,
    )
    return all_sequences, torch.from_numpy(actions).float().to(device)


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
    z_start_raw, _z_start_proj = _encode_frame(model, image)
    _z_goal_raw, z_goal_proj = _encode_frame(model, goal_image)
    repeated_start = z_start_raw.repeat(action_tensor.shape[0], 1)
    z_pred = model.plan_rollout(repeated_start, action_tensor)
    cost = model.plan_cost(z_pred, z_goal_proj.repeat(action_tensor.shape[0], 1))
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
            goal_img = _render_tensor_from_base(
                build,
                pack,
                base_xyz_m=np.asarray([target_xy[0], target_xy[1], base_z], dtype=np.float32),
                base_quat_wxyz=_quat_wxyz_from_yaw(target_yaw),
                device=device,
            )
            goal = GoalSpec(
                object_id=str(object_id),
                landmark_xy=(lx, ly),
                target_xy=(float(target_xy[0]), float(target_xy[1])),
                target_yaw_rad=float(target_yaw),
                image=goal_img,
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
    grid = InflatedOccupancyGrid(pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=0.20)
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
) -> PolicyResult:
    _set_pose(build=build, runner=runner, pos_xyz=start_pos, quat_wxyz=start_quat)
    initial_pos, _initial_quat = _current_pose(build)
    prev_xy = np.asarray(initial_pos[:2], dtype=np.float64)
    initial_distance = _xy_distance(prev_xy, goal.target_xy)
    path_length = 0.0
    primitives: list[str] = []
    plan_costs: list[float] = []
    fell = False

    for _block_idx in range(int(max_blocks)):
        pos, quat = _current_pose(build)
        if float(pos[2]) < float(fall_z_threshold_m):
            fell = True
            break
        if _xy_distance(pos[:2], goal.target_xy) <= float(goal_radius_m):
            break

        if policy_name == "lewm":
            image = _render_tensor_from_base(
                build,
                pack,
                base_xyz_m=pos,
                base_quat_wxyz=quat,
                device=device,
            )
            primitive_name, cost = _choose_lewm_primitive(
                model,
                image,
                goal.image,
                sequences,
                action_tensor,
            )
            plan_costs.append(cost)
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

        if runner is None:
            _execute_kinematic_primitive(
                build,
                registry,
                primitive_name,
                command_dt_s=command_dt_s,
                grid=grid,
            )
        else:
            _execute_physical_primitive(runner, registry, primitive_name)
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
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--goal-radius-m", type=float, default=0.55)
    parser.add_argument("--goal-standoff-m", type=float, default=0.85)
    parser.add_argument("--min-initial-distance-m", type=float, default=1.5)
    parser.add_argument("--beacon-approach-distance-m", type=float, default=1.5)
    parser.add_argument("--beacon-start-yaw-jitter-rad", type=float, default=0.0)
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--primitive-names",
        default="hold,forward_medium,arc_left,arc_right,yaw_left,yaw_right,backward",
    )
    parser.add_argument("--policies", default="lewm,bearing,hold,random")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
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

    primitive_names = _parse_csv(args.primitive_names)
    policies = _parse_csv(args.policies)
    if "hold" not in primitive_names:
        raise SystemExit("--primitive-names must include hold")
    unsupported = sorted(set(policies) - {"lewm", "bearing", "hold", "random"})
    if unsupported:
        raise SystemExit(f"unsupported policies: {unsupported}")
    if int(args.horizon) < 1:
        raise SystemExit("--horizon must be >= 1")

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
                render_robot=False,
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
            grid = InflatedOccupancyGrid(pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=0.20)
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
                    )
                    results.append(result)
                    print(
                        f"    {policy_name}: final={result.final_distance_m:.2f}m "
                        f"progress={result.progress_m:.2f}m success={int(result.success)} "
                        f"blocks={result.blocks_executed}",
                        flush=True,
                    )
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
        "horizon": int(args.horizon),
        "candidate_count": len(sequences),
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
