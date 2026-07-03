#!/usr/bin/env python3
"""Micro-repro for physical-mode capsize windows.

Rebuilds the scene, teleports the robot to the pose recorded at
--start-tick in a closed-loop result log, then re-executes the logged
primitive sequence through real physics with a configurable stability
guard. Lets a guard variant be evaluated against a recorded capsize in
minutes instead of re-simulating the full run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

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

try:
    import yaml as _yaml  # noqa: F401
except ModuleNotFoundError:
    _system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if _system_dist_packages.is_dir():
        sys.path.append(str(_system_dist_packages))
        import yaml as _yaml  # noqa: F401
        sys.path.remove(str(_system_dist_packages))

from benchmark_lewm_closed_loop_mpc import (  # noqa: E402
    _execute_physical_primitive,
    _set_pose,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--start-tick", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--scene-corpus", type=Path,
                        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--platform-manifest", type=Path,
                        default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path,
                        default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--family", default="large_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--guard-threshold", type=float, default=0.0,
                        help="Stability guard roll/pitch threshold; 0 disables.")
    parser.add_argument("--guard-primitive", default="backward")
    parser.add_argument("--guard-hold-ticks", type=int, default=3)
    parser.add_argument("--settle-ticks", type=int, default=6,
                        help="Hold ticks after teleport so the gait settles at the pose.")
    args = parser.parse_args()

    payload = json.loads(args.result.read_text())
    result = payload.get("result", {})
    log = [e for e in payload.get("log", []) if isinstance(e, dict) and "post_xy" in e]
    scene_id = str(args.scene_id or result.get("scene") or "")
    by_tick = {int(e["tick"]): e for e in log}
    start = by_tick[int(args.start_tick)]
    primitives = []
    for t in range(int(args.start_tick) + 1, int(args.start_tick) + 1 + int(args.ticks)):
        e = by_tick.get(t)
        if e is None or not e.get("primitive"):
            break
        primitives.append(str(e["primitive"]))

    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=str(args.split),
                                 family=str(args.family))
    scene_dir = [p for p in scene_dirs if p.name == scene_id][0]
    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
    build = build_scene_from_pack(pack, n_envs=1, backend=str(args.backend),
                                  show_viewer=False, render_robot=True, apply_textures=False)
    policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, REPO_ROOT, device="cpu")
    runner = RolloutRunner(
        build, policy, registry, safety,
        config=RolloutConfig(
            n_blocks=len(primitives) + int(args.settle_ticks) + int(args.guard_hold_ticks) + 4,
            fall_z_threshold_m=0.15,
            rgb_capture_per_block=False,
            seed=0,
            log_progress_every_blocks=0,
            foot_contact_source="zero",
            randomize_spawn_pose=False,
        ),
    )
    pos = np.asarray([start["post_xy"][0], start["post_xy"][1], float(start.get("post_z", 0.35))],
                     dtype=np.float32)
    quat = _quat_wxyz_from_rpy(float(start.get("post_roll", 0.0)),
                               float(start.get("post_pitch", 0.0)),
                               float(start.get("post_yaw", 0.0)))
    _set_pose(build=build, runner=runner, pos_xyz=pos, quat_wxyz=quat)
    for _ in range(int(args.settle_ticks)):
        _execute_physical_primitive(runner, registry, "hold")

    def pose():
        p = build.robot.get_pos().cpu().numpy().reshape(-1)
        q = build.robot.get_quat().cpu().numpy().reshape(-1)
        w, x, y, z = [float(v) for v in q]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return p, roll, pitch

    hold_remaining = 0
    max_tip = 0.0
    guard_events = 0
    prev_roll = 0.0
    prev_pitch = 0.0
    for i, prim in enumerate(primitives):
        executed = prim
        if float(args.guard_threshold) > 0.0:
            if hold_remaining > 0:
                executed = str(args.guard_primitive)
                hold_remaining -= 1
            elif max(abs(prev_roll), abs(prev_pitch)) >= float(args.guard_threshold):
                executed = str(args.guard_primitive)
                hold_remaining = max(0, int(args.guard_hold_ticks) - 1)
                guard_events += 1
        _execute_physical_primitive(runner, registry, executed)
        p, roll, pitch = pose()
        prev_roll, prev_pitch = float(roll), float(pitch)
        tip = max(abs(roll), abs(pitch))
        max_tip = max(max_tip, tip)
        print(f"t+{i + 1:03d} cmd={prim:12s} exec={executed:12s} roll={roll:+.2f} "
              f"pitch={pitch:+.2f} z={float(p[2]):.2f}", flush=True)
    print(f"RESULT max_tip={max_tip:.2f} guard_events={guard_events} "
          f"capsized={'YES' if max_tip > 1.0 else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
