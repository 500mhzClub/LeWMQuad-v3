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
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402

# Primitive velocity table (config/go2_primitive_registry.yaml) for aux command.
_PRIM_CMD = {
    "forward_medium": (0.25, 0.0, 0.0), "forward_slow": (0.2, 0.0, 0.0),
    "arc_left": (0.2, 0.0, 0.45), "arc_right": (0.2, 0.0, -0.45),
    "yaw_left": (0.0, 0.0, 0.45), "yaw_right": (0.0, 0.0, -0.45),
    "backward": (-0.2, 0.0, 0.0), "hold": (0.0, 0.0, 0.0),
}
_FORWARD_PRIMITIVES = frozenset(("forward_slow", "forward_medium", "forward_fast", "arc_left", "arc_right"))
_TURN_PRIMITIVES = frozenset(("yaw_left", "yaw_right"))


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


def _look_ahead_free(grid, pos_xy, yaw, dist_m: float = 0.5) -> bool:
    return grid.is_free((float(pos_xy[0]) + dist_m * math.cos(yaw),
                         float(pos_xy[1]) + dist_m * math.sin(yaw)))


def _round_float(value: float | None, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _body_probe_clearance(
    grid: InflatedOccupancyGrid,
    xy: np.ndarray | tuple[float, float],
    yaw: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
) -> float:
    """Minimum C-space clearance over a compact Go2 footprint probe."""

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
        min_clearance = min(min_clearance, grid.configuration_clearance_m((px, py)))
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
    force_escape: bool = False,
) -> tuple[str, dict[str, Any]]:
    requested_report = _primitive_clearance_report(
        registry, requested, pos_xy, yaw, grid, command_dt_s,
        body_forward_m=body_forward_m, body_half_width_m=body_half_width_m,
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
    """Grid coverage scaffold: drive toward the nearest unvisited free nav-cell
    (BFS over a coarse free-cell graph), so the maze is systematically swept until
    the memory binds the target. Exploration is a scaffold; the memory recall +
    claim is the learned capability under test."""

    def __init__(self, grid, bounds, step_m: float = 0.3):
        self.step = step_m
        x0, y0, x1, y1 = bounds
        self.free: dict[tuple[int, int], tuple[float, float]] = {}
        nx = int((x1 - x0) / step_m) + 1
        ny = int((y1 - y0) / step_m) + 1
        for i in range(nx):
            for j in range(ny):
                x, y = x0 + i * step_m, y0 + j * step_m
                if grid.is_free((x, y)):
                    self.free[(i, j)] = (x, y)
        self.visited: set[tuple[int, int]] = set()
        self.target_cell = None
        self.path = None
        self.wp_idx = 0
        # Periodic look-around so the narrow forward camera catches landmarks.
        self.tick = 0
        self.scan_interval = 24
        self.scan_len = 7
        self.scan_remaining = 0
        self.scan_dir = "yaw_left"

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
                if n in self.free and n not in parent:
                    parent[n] = c
                    q.append(n)
        return None

    def _nearest_unvisited(self, start):
        from collections import deque
        seen = {start}
        q = deque([start])
        while q:
            c = q.popleft()
            if c not in self.visited and c != start:
                return c
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in self.free and n not in seen:
                    seen.add(n)
                    q.append(n)
        return None

    def primitive(self, pos, yaw):
        self.tick += 1
        # Periodic in-place scan to look around (catch off-axis landmarks).
        if self.scan_remaining > 0:
            self.scan_remaining -= 1
            return self.scan_dir
        if self.tick % self.scan_interval == 0:
            self.scan_remaining = self.scan_len - 1
            self.scan_dir = "yaw_left" if (self.tick // self.scan_interval) % 2 == 0 else "yaw_right"
            return self.scan_dir
        # Mark cells within a radius of the robot visited.
        r2 = (self.step * 1.4) ** 2
        for c, (cx, cy) in self.free.items():
            if (cx - pos[0]) ** 2 + (cy - pos[1]) ** 2 < r2:
                self.visited.add(c)
        cur = self._cell(pos[:2])
        # (Re)plan a path to the nearest unvisited frontier when we have none / finished.
        if not self.path or self.wp_idx >= len(self.path):
            goal = self._nearest_unvisited(cur)
            if goal is None:
                self.visited.clear()
                goal = self._nearest_unvisited(cur)
            self.path = self._bfs_path(cur, goal) if goal else None
            self.wp_idx = 1
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
            return "forward_medium"
        wx, wy = self.free[self.path[self.wp_idx]]
        bearing = wrap_angle_pi(math.atan2(wy - pos[1], wx - pos[0]) - yaw)
        if abs(bearing) > 0.5:  # face the waypoint
            return "yaw_left" if bearing > 0 else "yaw_right"
        if not _look_ahead_free(grid_global[0], pos[:2], yaw, 0.35):
            return "yaw_left" if bearing >= 0 else "yaw_right"
        if abs(bearing) < 0.18:
            return "forward_medium"
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
    parser.add_argument("--mode", choices=("kinematic", "physical"), default="kinematic",
                        help="physical: drive the robot with the trained Go2 PPO walking "
                             "policy stepped in Genesis physics (real gait + rigid-body "
                             "collisions); kinematic: integrate named velocity primitives "
                             "against a 2D grid (teleport, no contact).")
    parser.add_argument("--policy-device", default="cpu",
                        help="Torch device for the PPO locomotion policy (physical mode).")
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15,
                        help="Base-height fall threshold for the physical rollout.")
    parser.add_argument("--policy", choices=("wander", "memory"), default="memory")
    parser.add_argument("--demo-mode", choices=("explore", "recall"), default="explore",
                        help="recall: place at a line-of-sight standoff (privileged scaffold), "
                             "observe to bind, turn away to hide, then memory recalls + claims "
                             "(the 2D see->hide->recall analog). explore: autonomous coverage.")
    parser.add_argument("--observe-ticks", type=int, default=6)
    parser.add_argument("--hide-ticks", type=int, default=10)
    parser.add_argument("--controller", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, default=None)
    parser.add_argument("--target-color", default="green")
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
    parser.add_argument("--success-dist-m", type=float, default=0.8)
    parser.add_argument("--wall-aware-planner", action="store_true",
                        help="Opt-in privileged-grid local guard: predicts each primitive's "
                             "near-field clearance and vetoes blocked forward/arc commands. "
                             "This is a planning scaffold, not a deployment-valid latent model.")
    parser.add_argument("--wall-decision-source", choices=("privileged_grid", "learned_front"),
                        default="privileged_grid",
                        help="Source for wall-aware veto decisions. learned_front uses a frozen "
                             "JEPA/RGB front-blocked head and does not use the manifest grid for "
                             "runtime wall decisions.")
    parser.add_argument("--front-blocked-checkpoint", type=Path, default=None,
                        help="Checkpoint from train_go2_jepa_front_blocked_predictor.py "
                             "(required for --wall-decision-source learned_front).")
    parser.add_argument("--front-blocked-threshold", type=float, default=None,
                        help="Override the learned front-blocked probability threshold.")
    parser.add_argument("--wall-guard-states", default="EXPLORE",
                        help="Comma-separated controller states where --wall-aware-planner may veto "
                             "forward/arc commands. Default keeps target SEEK/SERVO unmodified.")
    parser.add_argument("--wall-min-clearance-m", type=float, default=0.02,
                        help="Minimum predicted C-space clearance for the wall-aware guard.")
    parser.add_argument("--wall-feasible-threshold", type=float, default=0.8,
                        help="Minimum feasible sample fraction before a forward primitive is vetoed.")
    parser.add_argument("--wall-body-forward-m", type=float, default=0.35,
                        help="Forward body probe length used by the wall-aware guard.")
    parser.add_argument("--wall-body-half-width-m", type=float, default=0.18,
                        help="Half-width body probe used by the wall-aware guard.")
    parser.add_argument("--wall-stall-displacement-m", type=float, default=0.03,
                        help="Physical blocks with less XY displacement than this count as contact-like stalls.")
    parser.add_argument("--wall-stall-streak", type=int, default=2,
                        help="Consecutive forward stalls before scheduling escape blocks.")
    parser.add_argument("--wall-escape-blocks", type=int, default=2,
                        help="Backward/yaw escape blocks scheduled after a stall streak.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--face-target", action="store_true",
                        help="Diagnostic: place robot at several poses facing the target color and "
                             "report the controller's color-mask area/bearing (tests in-sim detection).")
    args = parser.parse_args()
    wall_guard_states = {s.strip().upper() for s in str(args.wall_guard_states).split(",") if s.strip()}
    use_learned_wall_source = bool(
        args.wall_aware_planner and args.wall_decision_source == "learned_front"
    )

    device = torch.device("cpu")
    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family)
    if args.scene_id:
        scene_dirs = [d for d in scene_dirs if d.name == args.scene_id] or scene_dirs
    scene_dir = scene_dirs[0]
    print(f"scene={scene_dir.name} target={args.target_color}", flush=True)

    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    build = build_scene_from_pack(pack, n_envs=1, backend=str(args.backend),
                                  show_viewer=False, render_robot=args.demo_video is not None,
                                  apply_textures=bool(args.apply_textures))
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    grid = InflatedOccupancyGrid(pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=float(args.inflation_m))
    grid_global[0] = grid
    spawn_pos, spawn_quat = _scene_spawn(scene_dir)
    landmarks = _scene_landmarks(scene_dir)
    wb = json.loads((scene_dir / "genesis_scene.json").read_text())["world_bounds_xy_m"]
    bounds = (wb[0][0], wb[0][1], wb[1][0], wb[1][1]) if isinstance(wb[0], (list, tuple)) else tuple(wb)
    explorer = FrontierExplorer(grid, bounds)
    print(f"explorer free nav-cells: {len(explorer.free)}", flush=True)

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
        if args.target_color not in color_vocab:
            raise SystemExit(f"target {args.target_color} not in {color_vocab}")
        tc = color_vocab.index(args.target_color)
        range_scale = float(ck["range_scale_m"])

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
    log = []
    claimed = False
    first_seen_tick = None
    wall_source = (
        "learned_front_blocked"
        if use_learned_wall_source
        else ("privileged_manifest_grid" if args.wall_aware_planner else "diagnostic_only")
    )
    wall_metrics: dict[str, Any] = {
        "enabled": bool(args.wall_aware_planner),
        "source": wall_source,
        "commands_total": 0,
        "forward_requests": 0,
        "blocked_forward_requests": 0,
        "forward_executions": 0,
        "blocked_forward_executions": 0,
        "wall_vetoes": 0,
        "escape_blocks_executed": 0,
        "contact_like_stalls": 0,
        "stuck_recoveries": 0,
        "requested_min_clearance_min_m": None,
        "selected_min_clearance_min_m": None,
        "forward_execution_displacement_sum_m": 0.0,
    }
    stuck_streak = 0
    escape_blocks = 0

    for tick in range(int(args.max_ticks)):
        pos, quat = _current_pose(build)
        yaw = _yaw_from_quat_wxyz(quat)
        cur_pose = (float(pos[0]), float(pos[1]), float(yaw))
        log_entry: dict[str, Any] | None = None
        bearing_for_guard: float | None = None
        front_blocked_prob: float | None = None

        if args.policy == "wander":
            primitive = explorer.primitive(pos, yaw)
            log_entry = {"tick": tick, "state": "WANDER", "primitive": primitive}
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
            dx, dy, dyaw = _body_delta(prev_pose, cur_pose)
            aux = _build_aux((dx, dy, dyaw), last_cmd, last_primitive)
            aux_t = (torch.from_numpy(aux).to(device) - aux_mean) / aux_std
            motion_delta = torch.tensor([dx / range_scale, dy / range_scale, dyaw], dtype=torch.float32)
            outputs, ctrl_state = model.step_online(ego64, aux_t, motion_delta, ctrl_state)
            mem_vec = outputs["memory_vec"][tc]
            mem_conf = float(outputs["memory_conf"][tc])
            area = float(outputs["rgb_area_logits"][tc]) if "rgb_area_logits" in outputs else -9.0
            evid = outputs["evidence_vec"][tc]
            in_cone = area > 0.0
            seen = mem_conf > float(args.seen_conf)
            if seen and first_seen_tick is None:
                first_seen_tick = tick
            if in_cone:
                bearing = math.atan2(float(evid[1]), float(evid[0]))
            else:
                bearing = math.atan2(float(mem_vec[1]), float(mem_vec[0]))
            bearing_for_guard = bearing

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
                if seen and in_cone and area > float(args.claim_area_logit) and abs(bearing) < float(args.claim_bearing):
                    claimed = True
                    log.append({"tick": tick, "state": "CLAIM", "mem_conf": mem_conf, "area": area,
                                "bearing": bearing})
                    break
                if not seen:
                    primitive = explorer.primitive(pos, yaw)
                    st = "EXPLORE"
                elif in_cone:
                    if abs(bearing) < 0.15:
                        primitive = "forward_medium"
                    else:
                        primitive = "arc_left" if bearing > 0 else "arc_right"
                    st = "SERVO"
                else:
                    if bearing > 0.1:
                        primitive = "yaw_left"
                    elif bearing < -0.1:
                        primitive = "yaw_right"
                    else:
                        primitive = "forward_medium"
                    st = "SEEK"
            log_entry = {"tick": tick, "state": st, "primitive": primitive, "mem_conf": round(mem_conf, 3),
                         "area": round(area, 2), "bearing": round(bearing, 2), "in_cone": in_cone}

        requested_primitive = primitive
        guard_state = str(log_entry.get("state", "")) if log_entry is not None else ""
        guard_enabled = bool(args.wall_aware_planner and (not wall_guard_states or guard_state.upper() in wall_guard_states))
        force_escape = bool(guard_enabled and escape_blocks > 0 and requested_primitive in _FORWARD_PRIMITIVES)
        if force_escape:
            escape_blocks -= 1
            wall_metrics["escape_blocks_executed"] += 1
        if use_learned_wall_source:
            primitive, wall_guard = _learned_front_guard_select(
                requested=requested_primitive,
                bearing=bearing_for_guard,
                enabled=guard_enabled,
                front_blocked_prob=front_blocked_prob,
                threshold=float(front_threshold if front_threshold is not None else 0.5),
                force_escape=force_escape,
            )
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
                force_escape=force_escape,
            )
        wall_metrics["commands_total"] += 1
        if requested_primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_requests"] += 1
            if bool(wall_guard["requested_blocked"]):
                wall_metrics["blocked_forward_requests"] += 1
        if primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_executions"] += 1
            if bool(wall_guard["selected_blocked"]):
                wall_metrics["blocked_forward_executions"] += 1
        if bool(wall_guard["vetoed"]):
            wall_metrics["wall_vetoes"] += 1
        for key, guard_key in (
            ("requested_min_clearance_min_m", "requested_min_clearance_m"),
            ("selected_min_clearance_min_m", "selected_min_clearance_m"),
        ):
            val = wall_guard.get(guard_key)
            if val is not None:
                wall_metrics[key] = val if wall_metrics[key] is None else min(float(wall_metrics[key]), float(val))
        if log_entry is not None:
            log_entry["requested_primitive"] = requested_primitive
            log_entry["primitive"] = primitive
            log_entry["wall_guard"] = wall_guard
            log.append(log_entry)

        if args.mode == "physical":
            _execute_physical_primitive(
                runner, registry, primitive,
                frame_sink=frames if capture else None, build=build, pack=pack, device=device)
        else:
            _execute_kinematic_primitive(
                build, registry, primitive, command_dt_s=float(args.command_dt_s),
                grid=grid, frame_sink=frames if capture else None, pack=pack, device=device)
        post_pos, post_quat = _current_pose(build)
        xy_displacement = float(math.hypot(float(post_pos[0]) - float(pos[0]), float(post_pos[1]) - float(pos[1])))
        if primitive in _FORWARD_PRIMITIVES:
            wall_metrics["forward_execution_displacement_sum_m"] += xy_displacement
            stalled = xy_displacement < float(args.wall_stall_displacement_m)
            if stalled:
                wall_metrics["contact_like_stalls"] += 1
                stuck_streak += 1
                if stuck_streak >= int(args.wall_stall_streak):
                    escape_blocks = max(escape_blocks, int(args.wall_escape_blocks))
                    wall_metrics["stuck_recoveries"] += 1
                    stuck_streak = 0
            else:
                stuck_streak = 0
        else:
            stalled = False
            if primitive in _TURN_PRIMITIVES or primitive == "backward":
                stuck_streak = 0
        if log_entry is not None:
            log_entry["executed_displacement_m"] = _round_float(xy_displacement)
            log_entry["post_xy"] = [_round_float(float(post_pos[0])), _round_float(float(post_pos[1]))]
            log_entry["stalled"] = bool(stalled)
        last_primitive = primitive
        last_cmd = _PRIM_CMD.get(primitive, (0.0, 0.0, 0.0))
        prev_pose = cur_pose

    final_pos, _ = _current_pose(build)
    final_xy = np.asarray([float(final_pos[0]), float(final_pos[1])])
    dist = float(np.linalg.norm(final_xy - landmarks[args.target_color])) if args.target_color in landmarks else None
    success = bool(claimed and dist is not None and dist <= float(args.success_dist_m))
    if wall_metrics["forward_executions"]:
        wall_metrics["mean_forward_execution_displacement_m"] = (
            wall_metrics["forward_execution_displacement_sum_m"] / wall_metrics["forward_executions"]
        )
    else:
        wall_metrics["mean_forward_execution_displacement_m"] = None
    wall_metrics.pop("forward_execution_displacement_sum_m", None)
    result = {
        "scene": scene_dir.name, "policy": args.policy, "target_color": args.target_color,
        "ticks_used": len(log), "claimed": claimed, "first_seen_tick": first_seen_tick,
        "final_xy": final_xy.tolist(), "target_xy": landmarks.get(args.target_color, np.zeros(2)).tolist(),
        "final_dist_to_target_m": dist, "success": success, "wall_metrics": wall_metrics,
    }
    print(json.dumps(result, indent=2))
    if args.policy == "memory":
        print("STATES:", " ".join(f"{e['tick']}:{e['state'][0]}" for e in log[-40:]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"result": result, "log": log}, indent=2))
    if capture and frames:
        import imageio
        args.demo_video.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for third_np, ego_np, *_ in frames:
            ego_up = np.asarray(F.interpolate(
                torch.from_numpy(ego_np).permute(2, 0, 1)[None].float(),
                size=(third_np.shape[0], third_np.shape[1]), mode="nearest")[0].permute(1, 2, 0).byte())
            out.append(np.concatenate([third_np, ego_up], axis=1))
        imageio.mimwrite(str(args.demo_video), out, fps=10, macro_block_size=8)
        print(f"wrote {args.demo_video} ({len(out)} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
