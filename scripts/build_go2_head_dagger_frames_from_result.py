#!/usr/bin/env python3
"""Render ego frames at closed-loop decision poses for head DAgger training.

Takes a closed-loop result JSON (benchmark_go2_memory_closed_loop.py output),
re-renders the egocentric RGB frame at every recorded decision pose, and writes
a JSONL of counterfactual-labelable rows for
train_go2_jepa_primitive_outcome_predictor.py (label modes
counterfactual_body_clearance / closed_loop_progress-compatible counterfactual
retraining). Labels themselves are computed by the trainer from the scene
manifest grid, so rows only need frame + pose + manifest.

One result (one scene) per invocation to keep the Genesis lifecycle simple.
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

import torch  # noqa: E402

from benchmark_go2_memory_closed_loop import _scene_spawn  # noqa: E402
from benchmark_lewm_closed_loop_mpc import (  # noqa: E402
    _render_tensor_from_base,
    _yaw_from_quat_wxyz,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _decision_poses(result_path: Path) -> tuple[str, list[dict[str, float]]]:
    payload = json.loads(result_path.read_text())
    result = payload.get("result", {})
    log = payload.get("log", [])
    scene_id = str(result.get("scene") or "")
    poses: list[dict[str, float]] = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        post_xy = entry.get("post_xy")
        if post_xy is None:
            continue
        try:
            poses.append(
                {
                    "x": float(post_xy[0]),
                    "y": float(post_xy[1]),
                    "z": float(entry.get("post_z", 0.34)),
                    "yaw": float(entry.get("post_yaw", 0.0)),
                    "roll": float(entry.get("post_roll", 0.0)),
                    "pitch": float(entry.get("post_pitch", 0.0)),
                }
            )
        except Exception:
            continue
    return scene_id, poses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="Closed-loop result JSONs; all must be from the same scene.")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z",
    )
    parser.add_argument("--platform-manifest", type=Path,
                        default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path,
                        default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument("--dedupe-xy-m", type=float, default=0.03)
    parser.add_argument("--dedupe-yaw-rad", type=float, default=0.05)
    parser.add_argument("--include-spawn", action="store_true", default=True)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    scene_id = str(args.scene_id or "")
    all_poses: list[dict[str, float]] = []
    for result_path in args.results:
        result_scene, poses = _decision_poses(result_path.resolve())
        if not scene_id:
            scene_id = result_scene
        elif result_scene and result_scene != scene_id:
            raise SystemExit(
                f"result {result_path} is for scene {result_scene!r}, expected {scene_id!r}"
            )
        all_poses.extend(poses)
    if not scene_id:
        raise SystemExit("scene id missing; pass --scene-id or use results with result.scene")

    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=str(args.split),
                                 family=str(args.family))
    matches = [path for path in scene_dirs if path.name == scene_id]
    if not matches:
        raise SystemExit(f"scene {scene_id!r} not found in {args.scene_corpus}")
    scene_dir = matches[0]
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest {manifest_path}")

    platform = load_platform_manifest(args.platform_manifest.resolve())
    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    build = build_scene_from_pack(
        pack,
        n_envs=1,
        backend=str(args.backend),
        show_viewer=False,
        render_robot=False,
        apply_textures=bool(args.apply_textures),
    )

    spawn_pos, spawn_quat = _scene_spawn(scene_dir)
    if args.include_spawn:
        all_poses.insert(
            0,
            {
                "x": float(spawn_pos[0]),
                "y": float(spawn_pos[1]),
                "z": float(spawn_pos[2]),
                "yaw": float(_yaw_from_quat_wxyz(spawn_quat)),
                "roll": 0.0,
                "pitch": 0.0,
            },
        )

    dedupe_xy = max(1e-6, float(args.dedupe_xy_m))
    dedupe_yaw = max(1e-6, float(args.dedupe_yaw_rad))
    seen: set[tuple[int, int, int]] = set()
    unique_poses: list[dict[str, float]] = []
    for pose in all_poses:
        key = (
            int(round(pose["x"] / dedupe_xy)),
            int(round(pose["y"] / dedupe_xy)),
            int(round(pose["yaw"] / dedupe_yaw)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_poses.append(pose)

    print(
        f"scene={scene_id} results={len(args.results)} poses={len(all_poses)} "
        f"unique={len(unique_poses)}",
        flush=True,
    )

    frames_dir = args.frames_dir.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    device = torch.device("cpu")
    rows = 0
    with args.output_jsonl.open("w") as sink:
        for index, pose in enumerate(unique_poses):
            pos_xyz = np.asarray([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
            quat = _quat_wxyz_from_rpy(pose["roll"], pose["pitch"], pose["yaw"])
            build.robot.set_pos(pos_xyz[None, :], envs_idx=[0], zero_velocity=True)
            build.robot.set_quat(quat[None, :], envs_idx=[0], zero_velocity=False)
            try:
                build.scene.step()
            except Exception:
                pass
            ego = _render_tensor_from_base(
                build, pack, base_xyz_m=pos_xyz, base_quat_wxyz=quat, device=device
            )
            ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            frame_path = frames_dir / f"{scene_id}_pose{index:05d}.png"
            imageio.imwrite(frame_path, ego_np)
            row = {
                "schema": "lewm_go2_head_dagger_pose_row_v0",
                "scene_id": scene_id,
                "scene_manifest": str(manifest_path),
                "start_frame": str(frame_path),
                "start_base_pose_world": {
                    "position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
                },
                "start_base_rpy_rad": {
                    "roll": pose["roll"],
                    "pitch": pose["pitch"],
                    "yaw": pose["yaw"],
                },
                "command_dt_s": float(registry.command_dt_s),
            }
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            rows += 1
            if args.progress_every > 0 and (index + 1) % int(args.progress_every) == 0:
                print(f"rendered {index + 1}/{len(unique_poses)}", flush=True)
    print(f"wrote {rows} rows -> {args.output_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
