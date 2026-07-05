#!/usr/bin/env python3
"""Replay a saved Go2 closed-loop primitive log and render it at sim cadence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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

from benchmark_go2_memory_closed_loop import _scene_spawn  # noqa: E402
from benchmark_lewm_closed_loop_mpc import (  # noqa: E402
    _current_pose,
    _execute_physical_primitive,
    _quat_wxyz_from_yaw,
    _render_synced_third_person,
    _render_tensor_from_base,
    _set_pose,
    _yaw_from_quat_wxyz,
)
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import find_scene_dirs, load_platform_manifest, load_scene_pack  # noqa: E402


class StreamingVideoSink:
    def __init__(self, path: Path, fps: float) -> None:
        import imageio

        self.path = path
        self.fps = float(fps)
        self.count = 0
        self.first_pose: tuple[float, float, float] | None = None
        self.last_pose: tuple[float, float, float] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=self.fps, macro_block_size=8)

    def append(self, frame: tuple[np.ndarray, np.ndarray, float, float, float]) -> None:
        third_np, ego_np, x, y, yaw, *_rest = frame
        ego_up = np.asarray(
            F.interpolate(
                torch.from_numpy(ego_np).permute(2, 0, 1)[None].float(),
                size=(third_np.shape[0], third_np.shape[1]),
                mode="nearest",
            )[0]
            .permute(1, 2, 0)
            .byte()
        )
        out = np.concatenate([third_np, ego_up], axis=1)
        self._writer.append_data(out)
        pose = (float(x), float(y), float(yaw))
        if self.first_pose is None:
            self.first_pose = pose
        self.last_pose = pose
        self.count += 1

    def close(self) -> None:
        self._writer.close()


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


def _load_replay_log(result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    payload = json.loads(result_path.read_text())
    result = payload.get("result", payload if isinstance(payload, dict) else {})
    log = payload.get("log", [])
    if not isinstance(log, list):
        raise ValueError(f"{result_path} does not contain a list-valued log")
    primitives: list[str] = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        primitive = entry.get("primitive")
        if primitive is None:
            continue
        primitive_name = str(primitive)
        if not primitive_name or primitive_name.lower() == "none":
            continue
        primitives.append(primitive_name)
    log_entries = [entry for entry in log if isinstance(entry, dict)]
    return result, log_entries, primitives


def _pose_entries_from_log(log: list[dict[str, Any]], max_entries: int | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in log:
        primitive = entry.get("primitive")
        post_xy = entry.get("post_xy")
        if primitive is None or post_xy is None:
            continue
        try:
            xy = [float(post_xy[0]), float(post_xy[1])]
            pose_entry = {
                "primitive": str(primitive),
                "post_xy": xy,
                "post_z": float(entry.get("post_z", 0.34)),
                "post_yaw": float(entry.get("post_yaw", 0.0)),
                "post_roll": float(entry.get("post_roll", 0.0)),
                "post_pitch": float(entry.get("post_pitch", 0.0)),
            }
        except Exception:
            continue
        entries.append(pose_entry)
        if max_entries is not None and len(entries) >= max(0, int(max_entries)):
            break
    return entries


def _wrap_pi(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(0.5 * float(roll))
    sr = math.sin(0.5 * float(roll))
    cp = math.cos(0.5 * float(pitch))
    sp = math.sin(0.5 * float(pitch))
    cy = math.cos(0.5 * float(yaw))
    sy = math.sin(0.5 * float(yaw))
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _set_robot_pose(build: Any, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
    build.robot.set_pos(np.asarray(pos_xyz, dtype=np.float32)[None, :], envs_idx=[0], zero_velocity=True)
    build.robot.set_quat(np.asarray(quat_wxyz, dtype=np.float32)[None, :], envs_idx=[0], zero_velocity=False)


def _render_recorded_pose_frame(
    *,
    build: Any,
    pack: Any,
    sink: StreamingVideoSink,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    yaw: float,
    device: torch.device,
    third_person_build: Any | None,
    leg_dof_idx: Any,
) -> None:
    _set_robot_pose(build, pos_xyz, quat_wxyz)
    if third_person_build is None:
        try:
            build.scene.step()
        except Exception:
            pass
    ego = _render_tensor_from_base(build, pack, base_xyz_m=pos_xyz, base_quat_wxyz=quat_wxyz, device=device)
    ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    third_np = _render_synced_third_person(
        source_build=build,
        render_build=third_person_build,
        base_xyz=pos_xyz,
        base_quat_wxyz=quat_wxyz,
        yaw=float(yaw),
        leg_dof_idx=leg_dof_idx,
    )
    sink.append((third_np, ego_np, float(pos_xyz[0]), float(pos_xyz[1]), float(yaw)))


def _select_scene(
    *,
    scene_corpus: Path,
    split: str,
    family: str,
    scene_id: str,
) -> Path:
    scene_dirs = find_scene_dirs(scene_corpus.resolve(), split=split, family=family)
    matches = [path for path in scene_dirs if path.name == scene_id]
    if not matches:
        raise SystemExit(f"scene {scene_id!r} not found in {scene_corpus}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z",
    )
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path, default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument(
        "--render-robot",
        action="store_true",
        help=(
            "Legacy/debug mode: render Go2 visual meshes in the main camera "
            "scene, including egocentric replay RGB. By default ego RGB hides "
            "the robot body; demo videos use a separate robot-visible scene "
            "for the third-person panel."
        ),
    )
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--fall-z-threshold-m", type=float, default=0.15)
    parser.add_argument("--tip-threshold-rad", type=float, default=math.radians(60.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture-rate", choices=("command", "policy"), default="policy")
    parser.add_argument("--demo-fps", type=float, default=None)
    parser.add_argument("--max-primitives", type=int, default=None)
    parser.add_argument(
        "--replay-mode",
        choices=("recorded", "physical"),
        default="recorded",
        help=(
            "recorded renders the saved post-pose trajectory exactly; physical "
            "re-executes primitive names through Genesis and can drift from the "
            "closed-loop result."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    result, log_entries, primitives = _load_replay_log(args.result.resolve())
    if args.max_primitives is not None:
        primitives = primitives[: max(0, int(args.max_primitives))]
    if not primitives:
        raise SystemExit("no executable primitives found in result log")
    pose_entries = _pose_entries_from_log(
        log_entries,
        max_entries=None if args.max_primitives is None else int(args.max_primitives),
    )
    if str(args.replay_mode) == "recorded":
        if not pose_entries:
            raise SystemExit("recorded replay requested but result log has no post_xy/post_yaw poses")
        primitives = [str(entry["primitive"]) for entry in pose_entries]

    scene_id = str(args.scene_id or result.get("scene") or "")
    if not scene_id:
        raise SystemExit("scene id missing; pass --scene-id or use a result with result.scene")

    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dir = _select_scene(
        scene_corpus=args.scene_corpus,
        split=str(args.split),
        family=str(args.family),
        scene_id=scene_id,
    )
    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
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
    policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, REPO_ROOT, device=str(args.policy_device))
    runner = RolloutRunner(
        build,
        policy,
        registry,
        safety,
        config=RolloutConfig(
            n_blocks=len(primitives),
            fall_z_threshold_m=float(args.fall_z_threshold_m),
            rgb_capture_per_block=False,
            seed=int(args.seed),
            log_progress_every_blocks=0,
            foot_contact_source="zero",
            randomize_spawn_pose=False,
        ),
    )
    spawn_pos, spawn_quat = _scene_spawn(scene_dir)
    _set_pose(build=build, runner=runner, pos_xyz=spawn_pos, quat_wxyz=spawn_quat)

    if args.demo_fps is not None:
        demo_fps = float(args.demo_fps)
    elif str(args.capture_rate) == "policy":
        demo_fps = 1.0 / float(pack.timing.policy_dt_s)
    else:
        demo_fps = 1.0 / float(pack.timing.command_dt_s)

    frames_per_primitive = int(registry.block_size)
    if str(args.capture_rate) == "policy":
        frames_per_primitive *= int(pack.timing.command_ticks_per_block)
    expected_frames = int(len(primitives) * frames_per_primitive)
    print(
        f"replay scene={scene_id} mode={args.replay_mode} primitives={len(primitives)} "
        f"capture={args.capture_rate} expected_frames={expected_frames} fps={demo_fps:.2f}",
        flush=True,
    )

    sink = StreamingVideoSink(args.demo_video, demo_fps) if args.demo_video is not None else None
    min_base_z = float("inf")
    max_abs_roll_pitch = 0.0
    fall_events = 0
    tip_events = 0
    first_unstable: dict[str, Any] | None = None
    try:
        if str(args.replay_mode) == "recorded":
            if sink is None:
                raise SystemExit("recorded replay requires --demo-video")
            prev_pos = np.asarray(spawn_pos, dtype=np.float32).copy()
            prev_yaw = _yaw_from_quat_wxyz(spawn_quat)
            prev_roll = 0.0
            prev_pitch = 0.0
            for index, entry in enumerate(pose_entries, start=1):
                end_pos = np.asarray(
                    [entry["post_xy"][0], entry["post_xy"][1], entry["post_z"]],
                    dtype=np.float32,
                )
                end_yaw = float(entry["post_yaw"])
                end_roll = float(entry["post_roll"])
                end_pitch = float(entry["post_pitch"])
                dyaw = _wrap_pi(end_yaw - prev_yaw)
                for step_idx in range(1, frames_per_primitive + 1):
                    alpha = float(step_idx) / float(frames_per_primitive)
                    pos = prev_pos * (1.0 - alpha) + end_pos * alpha
                    yaw = _wrap_pi(prev_yaw + dyaw * alpha)
                    roll = float(prev_roll * (1.0 - alpha) + end_roll * alpha)
                    pitch = float(prev_pitch * (1.0 - alpha) + end_pitch * alpha)
                    quat = _quat_wxyz_from_rpy(roll, pitch, yaw)
                    _render_recorded_pose_frame(
                        build=build,
                        pack=pack,
                        sink=sink,
                        pos_xyz=pos,
                        quat_wxyz=quat,
                        yaw=yaw,
                        device=torch.device("cpu"),
                        third_person_build=third_person_build,
                        leg_dof_idx=runner._leg_dof_idx,
                    )
                    tip = max(abs(roll), abs(pitch))
                    min_base_z = min(min_base_z, float(pos[2]))
                    max_abs_roll_pitch = max(max_abs_roll_pitch, tip)
                    fell = float(pos[2]) < float(args.fall_z_threshold_m)
                    tipped = tip > float(args.tip_threshold_rad)
                    if fell:
                        fall_events += 1
                    if tipped:
                        tip_events += 1
                    if first_unstable is None and (fell or tipped):
                        first_unstable = {
                            "primitive_index": index - 1,
                            "frame_index": int(sink.count - 1),
                            "time_s": float(sink.count) / float(demo_fps),
                            "primitive": str(entry["primitive"]),
                            "reason": "fall_and_tip" if fell and tipped else ("fall" if fell else "tip"),
                            "base_z_m": float(pos[2]),
                            "roll_rad": float(roll),
                            "pitch_rad": float(pitch),
                            "tip_rad": float(tip),
                        }
                prev_pos = end_pos
                prev_yaw = end_yaw
                prev_roll = end_roll
                prev_pitch = end_pitch
                if args.progress_every > 0 and (index == 1 or index % int(args.progress_every) == 0):
                    print(f"rendered {index}/{len(pose_entries)} recorded poses ({sink.count} frames)", flush=True)
        else:
            for index, primitive in enumerate(primitives, start=1):
                _execute_physical_primitive(
                    runner,
                    registry,
                    primitive,
                    frame_sink=sink,
                    build=build,
                    pack=pack,
                    device=torch.device("cpu"),
                    capture_policy_steps=(str(args.capture_rate) == "policy"),
                    third_person_build=third_person_build,
                )
                pos, quat = _current_pose(build)
                roll = _roll_from_quat_wxyz(quat)
                pitch = _pitch_from_quat_wxyz(quat)
                tip = max(abs(roll), abs(pitch))
                min_base_z = min(min_base_z, float(pos[2]))
                max_abs_roll_pitch = max(max_abs_roll_pitch, tip)
                fell = float(pos[2]) < float(args.fall_z_threshold_m)
                tipped = tip > float(args.tip_threshold_rad)
                if fell:
                    fall_events += 1
                if tipped:
                    tip_events += 1
                if first_unstable is None and (fell or tipped):
                    first_unstable = {
                        "primitive_index": index - 1,
                        "time_s": float(index) * float(registry.block_size) * float(pack.timing.command_dt_s),
                        "primitive": primitive,
                        "reason": "fall_and_tip" if fell and tipped else ("fall" if fell else "tip"),
                        "base_z_m": float(pos[2]),
                        "roll_rad": float(roll),
                        "pitch_rad": float(pitch),
                        "tip_rad": float(tip),
                    }
                if args.progress_every > 0 and (index == 1 or index % int(args.progress_every) == 0):
                    frame_count = 0 if sink is None else sink.count
                    print(f"rendered {index}/{len(primitives)} primitives ({frame_count} frames)", flush=True)
    finally:
        if sink is not None:
            sink.close()

    final_pos, final_quat = _current_pose(build)
    frame_count = 0 if sink is None else sink.count
    duration_s = float(frame_count) / float(demo_fps) if frame_count else (
        float(len(primitives)) * float(registry.block_size) * float(pack.timing.command_dt_s)
    )
    unstable_events = int(fall_events + tip_events)
    report = {
        "source_result": str(args.result),
        "video": None if args.demo_video is None else str(args.demo_video),
        "scene": scene_id,
        "replay_mode": str(args.replay_mode),
        "capture_rate": str(args.capture_rate),
        "fps": demo_fps,
        "frame_count": int(frame_count),
        "expected_frame_count": expected_frames,
        "duration_s": duration_s,
        "primitive_count": len(primitives),
        "registry_block_size": int(registry.block_size),
        "policy_dt_s": float(pack.timing.policy_dt_s),
        "command_dt_s": float(pack.timing.command_dt_s),
        "fall_z_threshold_m": float(args.fall_z_threshold_m),
        "tip_threshold_rad": float(args.tip_threshold_rad),
        "stable_base": bool(unstable_events == 0),
        "fall_events": int(fall_events),
        "tip_events": int(tip_events),
        "unstable_base_events": int(unstable_events),
        "min_base_z_m": None if min_base_z == float("inf") else float(min_base_z),
        "max_abs_roll_pitch_rad": float(max_abs_roll_pitch),
        "first_unstable": first_unstable,
        "final_xy": [float(final_pos[0]), float(final_pos[1])],
        "final_yaw": float(_yaw_from_quat_wxyz(final_quat)),
        "source_final_xy": result.get("final_xy"),
        "source_claimed_colors": result.get("claimed_colors"),
        "first_render_pose": None if sink is None else sink.first_pose,
        "last_render_pose": None if sink is None else sink.last_pose,
    }
    if result.get("final_xy") is not None:
        try:
            src_xy = np.asarray(result["final_xy"], dtype=np.float64)
            out_xy = np.asarray(report["final_xy"], dtype=np.float64)
            report["source_final_xy_error_m"] = float(np.linalg.norm(src_xy[:2] - out_xy[:2]))
        except Exception:
            report["source_final_xy_error_m"] = None
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
