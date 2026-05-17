#!/usr/bin/env python3
"""Render a planned raw_rollout replay with Genesis."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))

from lewm_genesis.rollout import (  # noqa: E402
    DEFAULT_GO2_LEG_DOF_INDICES,
    DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER,
)
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path, help="render_replay_plan.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene-corpus", type=Path, default=None)
    parser.add_argument("--platform-manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--replay-env-mode",
        choices=("single", "batched"),
        default="single",
        help=(
            "single replays source env frames through one Genesis env while "
            "preserving source env_index in metadata; batched builds all source envs."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    platform_path = (
        args.platform_manifest.resolve()
        if args.platform_manifest is not None
        else repo_root / "config" / "go2_platform_manifest.yaml"
    )
    scene_corpus = (
        args.scene_corpus.resolve()
        if args.scene_corpus is not None
        else repo_root / ".generated" / "scene_corpus" / "acceptance"
    )
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frames_path = Path(plan["frames_jsonl"])
    if not frames_path.is_absolute():
        frames_path = plan_path.parent / frames_path
    frames = _load_frames(frames_path, max_frames=args.max_frames)
    if not frames:
        raise SystemExit(f"no frames to render in {frames_path}")

    output_dir = (
        args.out.resolve()
        if args.out is not None
        else Path(plan.get("output_dir", plan_path.parent)).resolve() / "rendered_vision"
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"rendered output already exists: {summary_path}; pass --overwrite")
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    platform = load_platform_manifest(platform_path)
    backend = _resolve_render_backend(args.backend, plan)
    scene_dir = _find_scene_dir(
        scene_corpus,
        scene_id=str(plan.get("scene_id") or ""),
        split=plan.get("split"),
        family=plan.get("scene_family"),
    )
    pack = load_scene_pack(
        scene_dir,
        platform_manifest=platform,
        workspace_root=repo_root,
    )
    source_env_count = int(
        plan.get("source_env_count")
        or max(1, max((int(f.get("env_index") or 0) for f in frames), default=0) + 1)
    )
    render_env_count = source_env_count if args.replay_env_mode == "batched" else 1
    build = build_scene_from_pack(
        pack,
        n_envs=render_env_count,
        backend=backend,
        show_viewer=bool(args.show_viewer),
    )
    leg_dof_idx = _resolve_rollout_leg_dof_indices(build.robot)

    metadata_path = output_dir / "frames_rendered.jsonl"
    records: list[dict[str, Any]] = []
    wall_start = time.time()
    with metadata_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            record = _render_frame(
                frame,
                build,
                leg_dof_idx,
                rgb_dir=rgb_dir,
                depth_dir=None if args.no_depth else depth_dir,
                target_env_index=0 if args.replay_env_mode == "single" else None,
            )
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    invalid = [r for r in records if not r["camera_valid"]]
    summary = {
        "schema": "lewm_rendered_vision_v0",
        "render_status": "complete",
        "plan": str(plan_path),
        "source_raw_rollout_dir": plan.get("raw_rollout_dir"),
        "scene_id": pack.scene_id,
        "scene_family": pack.family,
        "split": pack.split,
        "backend": backend,
        "source_env_count": source_env_count,
        "render_env_count": render_env_count,
        "replay_env_mode": args.replay_env_mode,
        "frame_count": len(records),
        "invalid_frame_count": len(invalid),
        "invalid_frame_rate": 0.0 if not records else len(invalid) / len(records),
        "rgb_dir": str(rgb_dir),
        "depth_dir": None if args.no_depth else str(depth_dir),
        "frames_rendered_jsonl": str(metadata_path),
        "wall_seconds": time.time() - wall_start,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "frame_count": len(records),
                "invalid_frame_count": len(invalid),
                "backend": summary["backend"],
            },
            sort_keys=True,
        )
    )
    return 2 if invalid else 0


def _load_frames(path: Path, *, max_frames: int | None) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            frames.append(json.loads(line))
            if max_frames is not None and len(frames) >= max_frames:
                break
    return frames


def _resolve_render_backend(cli_backend: str | None, plan: dict[str, Any]) -> str:
    if cli_backend:
        return cli_backend
    backend = str(plan.get("backend") or "auto")
    if backend.lower() == "genesis":
        return "auto"
    return backend


def _find_scene_dir(
    scene_corpus: Path,
    *,
    scene_id: str,
    split: str | None,
    family: str | None,
) -> Path:
    if not scene_id:
        raise ValueError("render plan does not include scene_id; cannot locate scene manifest")
    for scene_dir in find_scene_dirs(scene_corpus, split=split, family=family):
        if scene_dir.name == scene_id:
            return scene_dir
    raise FileNotFoundError(
        f"scene {scene_id!r} not found under {scene_corpus} split={split!r} family={family!r}"
    )


def _resolve_rollout_leg_dof_indices(robot: Any) -> np.ndarray:
    joints = getattr(robot, "joints", None)
    if joints is None:
        return np.array(DEFAULT_GO2_LEG_DOF_INDICES, dtype=np.int64)
    joint_by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
    if not all(name in joint_by_name for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER):
        return np.array(DEFAULT_GO2_LEG_DOF_INDICES, dtype=np.int64)
    return np.array(
        [_single_dof_index(joint_by_name[name]) for name in DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER],
        dtype=np.int64,
    )


def _single_dof_index(joint: Any) -> int:
    dofs_idx = getattr(joint, "dofs_idx", None)
    if dofs_idx is None:
        dofs_idx = getattr(joint, "dof_idx", None)
    if isinstance(dofs_idx, (list, tuple)):
        if len(dofs_idx) != 1:
            raise ValueError(
                f"joint {getattr(joint, 'name', '<unnamed>')} has {len(dofs_idx)} DOFs; expected 1"
            )
        return int(dofs_idx[0])
    if dofs_idx is None:
        raise ValueError(f"joint {getattr(joint, 'name', '<unnamed>')} has no DOF index")
    return int(dofs_idx)


def _render_frame(
    frame: dict[str, Any],
    build: Any,
    leg_dof_idx: np.ndarray,
    *,
    rgb_dir: Path,
    depth_dir: Path | None,
    target_env_index: int | None,
) -> dict[str, Any]:
    env_index = int(frame.get("env_index") or 0)
    render_env_index = env_index if target_env_index is None else int(target_env_index)
    _apply_robot_state(frame, build.robot, leg_dof_idx, render_env_index)
    _apply_camera_pose(frame, build.camera, render_env_index)
    rendered = build.camera.render(rgb=True, depth=depth_dir is not None, force_render=True)
    rgb, depth = _extract_render_outputs(rendered)
    rgb = _select_env(rgb, render_env_index)
    depth = _select_env(depth, render_env_index) if depth is not None else None

    frame_index = int(frame["frame_index"])
    stem = f"frame_{frame_index:06d}_env_{env_index:02d}"
    rgb_path = rgb_dir / f"{stem}.png"
    depth_path = None if depth_dir is None else depth_dir / f"{stem}.npy"
    camera_valid, invalid_reasons, depth_stats = _validate_frame(
        rgb,
        depth,
        require_depth=depth_dir is not None,
    )
    if rgb is not None:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(rgb_path)
    if depth is not None and depth_path is not None:
        np.save(depth_path, np.asarray(depth, dtype=np.float32))

    return {
        "frame_index": frame_index,
        "env_index": env_index,
        "render_env_index": render_env_index,
        "timestamp_ns": int(frame["timestamp_ns"]),
        "timestamp_s": float(frame["timestamp_s"]),
        "rgb_path": None if rgb is None else str(rgb_path),
        "depth_path": None if depth_path is None or depth is None else str(depth_path),
        "camera_valid": camera_valid,
        "invalid_reasons": invalid_reasons,
        "depth_stats": depth_stats,
        "source_frame": {
            "source_line": frame.get("source_line"),
            "source_topic": frame.get("source_topic"),
            "canonical_topic": frame.get("canonical_topic"),
        },
    }


def _apply_robot_state(frame: dict[str, Any], robot: Any, leg_dof_idx: np.ndarray, env_index: int) -> None:
    pose = frame.get("base_pose_world", {})
    position = pose.get("position", {})
    quat_xyzw = frame.get("base_quat_world_xyzw")
    if not position or quat_xyzw is None:
        raise ValueError(f"frame {frame.get('frame_index')} missing base pose/quaternion")
    pos = np.array(
        [[float(position["x"]), float(position["y"]), float(position["z"])]],
        dtype=np.float32,
    )
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    quat_wxyz = np.array([[qw, qx, qy, qz]], dtype=np.float32)
    envs = [env_index]
    robot.set_pos(pos, envs_idx=envs, zero_velocity=True)
    robot.set_quat(quat_wxyz, envs_idx=envs, zero_velocity=False)
    joint_state = frame.get("joint_state") or {}
    positions = joint_state.get("position")
    if positions:
        qpos = np.asarray(positions, dtype=np.float32)[None, :]
        if qpos.shape[-1] != len(leg_dof_idx):
            raise ValueError(
                f"frame {frame.get('frame_index')} has {qpos.shape[-1]} joints; expected {len(leg_dof_idx)}"
            )
        robot.set_dofs_position(qpos, leg_dof_idx.tolist(), envs_idx=envs)


def _apply_camera_pose(frame: dict[str, Any], camera: Any, env_index: int) -> None:
    pose = frame.get("camera_pose_world")
    if not pose:
        raise ValueError(f"frame {frame.get('frame_index')} missing camera_pose_world")
    pos = np.asarray(pose["position"], dtype=np.float32)
    lookat = np.asarray(pose["lookat"], dtype=np.float32)
    up = np.asarray(pose["up"], dtype=np.float32)
    if bool(getattr(camera, "_is_batched", False)):
        camera.set_pose(pos=pos[None, :], lookat=lookat[None, :], up=up[None, :], envs_idx=[env_index])
    else:
        camera.set_pose(pos=pos, lookat=lookat, up=up)


def _extract_render_outputs(rendered: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(rendered, np.ndarray):
        return rendered, None
    if isinstance(rendered, tuple):
        rgb = rendered[0] if len(rendered) > 0 else None
        depth = rendered[1] if len(rendered) > 1 else None
        return _to_numpy(rgb), _to_numpy(depth)
    return _to_numpy(rendered), None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except ImportError:
        return np.asarray(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _select_env(arr: np.ndarray | None, env_index: int) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.ndim >= 4:
        return arr[min(env_index, arr.shape[0] - 1)]
    return arr


def _validate_frame(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    *,
    require_depth: bool,
) -> tuple[bool, list[str], dict[str, float | int | None]]:
    reasons: list[str] = []
    if rgb is None:
        reasons.append("missing_rgb")
    else:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            reasons.append(f"bad_rgb_shape:{tuple(rgb.shape)}")
        if not np.isfinite(rgb).all():
            reasons.append("rgb_nonfinite")
    depth_stats: dict[str, float | int | None] = {
        "finite_count": None,
        "nonfinite_count": None,
        "min_m": None,
        "max_m": None,
    }
    if depth is None:
        if require_depth:
            reasons.append("missing_depth")
    else:
        finite = np.isfinite(depth)
        finite_count = int(finite.sum())
        nonfinite_count = int(depth.size - finite_count)
        depth_stats["finite_count"] = finite_count
        depth_stats["nonfinite_count"] = nonfinite_count
        if finite_count:
            depth_stats["min_m"] = float(np.nanmin(depth[finite]))
            depth_stats["max_m"] = float(np.nanmax(depth[finite]))
        else:
            reasons.append("depth_all_nonfinite")
        if any(math.isnan(float(v)) for v in (depth_stats["min_m"] or 0.0, depth_stats["max_m"] or 0.0)):
            reasons.append("depth_nan_stats")
    return not reasons, reasons, depth_stats


if __name__ == "__main__":
    raise SystemExit(main())
