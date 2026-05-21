#!/usr/bin/env python3
"""Render a planned raw_rollout replay with Genesis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))

from lewm_genesis.camera_safety import (  # noqa: E402
    camera_safety_config_from_pack,
    safe_camera_pose_from_base,
)
from lewm_genesis.rollout import (  # noqa: E402
    DEFAULT_GO2_LEG_DOF_INDICES,
    DEFAULT_GO2_LEG_JOINT_NAMES_ROLLOUT_ORDER,
)
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    effective_camera_mount_xyz_rpy,
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_genesis.vision_quality import (  # noqa: E402
    LOW_INFO_REASON_NAMES,
    assess_rendered_frame,
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
    parser.add_argument(
        "--env-index",
        type=int,
        default=None,
        help="Render only one source env stream from the replay plan.",
    )
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument(
        "--depth-validate-only",
        action="store_true",
        help="Render depth for the camera-validity gate but do not persist the .npy "
        "(depth is not a training input; this avoids ~1.2MB/frame of storage).",
    )
    parser.add_argument(
        "--store-resolution",
        choices=("native", "training"),
        default="native",
        help="Resolution of the stored RGB. 'training' downsamples to the platform "
        "manifest's training_resolution (e.g. 224x224) — what the encoder consumes — "
        "while still rendering+validating at native resolution.",
    )
    parser.add_argument(
        "--rgb-format",
        choices=("png", "npy"),
        default="png",
        help="RGB frame storage. Use npy to avoid PNG compression on production render jobs.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("replay", "overview"),
        default="replay",
        help="Camera pose source. replay uses recorded egocentric camera poses; overview uses a static top-down QA camera.",
    )
    parser.add_argument(
        "--overlay-target-label",
        action="store_true",
        help=(
            "Draw the privileged route target label onto RGB frames. Use only for "
            "manual QA renders; training renders must leave this disabled."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-textures",
        dest="textures",
        action="store_false",
        help="Disable CC0 surface textures (floor/walls/obstacles render as solid colors).",
    )
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
    frames = _load_frames(frames_path, max_frames=args.max_frames, env_index=args.env_index)
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
    # Depth is rendered for the camera-validity gate whenever it is enabled, but
    # only persisted to disk when not in validate-only mode (it is not a
    # training input). render_depth drives rendering+validation; persist_depth
    # drives the .npy write.
    render_depth = not args.no_depth
    persist_depth = render_depth and not args.depth_validate_only
    if persist_depth:
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
    render_robot = args.camera_mode != "replay"
    build = build_scene_from_pack(
        pack,
        n_envs=render_env_count,
        backend=backend,
        show_viewer=bool(args.show_viewer),
        render_robot=render_robot,
        apply_textures=bool(args.textures),
        batched_camera=args.replay_env_mode == "batched",
    )
    leg_dof_idx = _resolve_rollout_leg_dof_indices(build.robot)
    overview_pose = _overview_camera_pose(pack) if args.camera_mode == "overview" else None
    camera_safety_config = camera_safety_config_from_pack(pack)
    mount_xyz_body, mount_rpy_body = effective_camera_mount_xyz_rpy(pack)

    store_resolution_wh = (
        tuple(int(v) for v in pack.camera.training_resolution)
        if args.store_resolution == "training"
        else None
    )

    metadata_path = output_dir / "frames_rendered.jsonl"
    records: list[dict[str, Any]] = []
    wall_start = time.time()
    with metadata_path.open("w", encoding="utf-8") as stream:
        def _emit(record: dict[str, Any]) -> None:
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

        if args.replay_env_mode == "batched":
            _render_frames_batched(
                frames,
                build,
                leg_dof_idx,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir if persist_depth else None,
                render_depth=render_depth,
                store_resolution_wh=store_resolution_wh,
                rgb_format=args.rgb_format,
                camera_mode=args.camera_mode,
                overview_pose=overview_pose,
                camera_safety_config=camera_safety_config,
                mount_xyz_body=mount_xyz_body,
                mount_rpy_body=mount_rpy_body,
                scene_graph=pack.scene_graph,
                overlay_target_label=bool(args.overlay_target_label),
                on_record=_emit,
            )
        else:
            for frame in frames:
                _emit(
                    _render_frame(
                        frame,
                        build,
                        leg_dof_idx,
                        rgb_dir=rgb_dir,
                        depth_dir=depth_dir if persist_depth else None,
                        render_depth=render_depth,
                        store_resolution_wh=store_resolution_wh,
                        target_env_index=0,
                        rgb_format=args.rgb_format,
                        camera_mode=args.camera_mode,
                        overview_pose=overview_pose,
                        camera_safety_config=camera_safety_config,
                        mount_xyz_body=mount_xyz_body,
                        mount_rpy_body=mount_rpy_body,
                        scene_graph=pack.scene_graph,
                        overlay_target_label=bool(args.overlay_target_label),
                    )
                )

    invalid = [r for r in records if not r["camera_valid"]]
    low_info = [r for r in records if r.get("low_info_reasons")]
    low_info_allowed = [r for r in low_info if bool(r.get("low_info_allowed"))]
    low_texture = [
        r for r in records
        if "low_rgb_texture" in r.get("low_info_reasons", ())
    ]
    near_wall = [
        r for r in records
        if any(
            reason in {"near_wall_depth", "near_forward_geometry"}
            for reason in r.get("low_info_reasons", ())
        )
    ]
    camera_safety_records = [r.get("camera_safety") or {} for r in records]
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
        "render_robot": render_robot,
        "frame_count": len(records),
        "invalid_frame_count": len(invalid),
        "invalid_frame_rate": 0.0 if not records else len(invalid) / len(records),
        "low_info_frame_count": len(low_info),
        "low_info_frame_rate": 0.0 if not records else len(low_info) / len(records),
        "low_info_allowed_frame_count": len(low_info_allowed),
        "low_info_invalid_frame_count": len(low_info) - len(low_info_allowed),
        "low_rgb_texture_frame_count": len(low_texture),
        "near_wall_frame_count": len(near_wall),
        "rgb_dir": str(rgb_dir),
        "rgb_format": args.rgb_format,
        "camera_mode": args.camera_mode,
        "overlay_target_label": bool(args.overlay_target_label),
        "camera_retracted_count": sum(
            1 for r in camera_safety_records if float(r.get("retracted_m") or 0.0) > 0.0
        ),
        "camera_safety_unresolved_count": sum(
            1 for r in camera_safety_records if bool(r.get("unsafe"))
        ),
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


def _load_frames(
    path: Path,
    *,
    max_frames: int | None,
    env_index: int | None,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            if env_index is not None and int(frame.get("env_index") or 0) != int(env_index):
                continue
            frames.append(frame)
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
    render_depth: bool = True,
    store_resolution_wh: tuple[int, int] | None = None,
    target_env_index: int | None,
    rgb_format: str,
    camera_mode: str,
    overview_pose: dict[str, list[float]] | None,
    camera_safety_config: Any,
    mount_xyz_body: tuple[float, float, float],
    mount_rpy_body: tuple[float, float, float],
    scene_graph: Any = None,
    overlay_target_label: bool = False,
) -> dict[str, Any]:
    env_index = int(frame.get("env_index") or 0)
    render_env_index = env_index if target_env_index is None else int(target_env_index)
    _apply_robot_state(frame, build.robot, leg_dof_idx, render_env_index)
    camera_pose, camera_safety = _apply_camera_pose(
        frame,
        build.camera,
        render_env_index,
        camera_mode=camera_mode,
        overview_pose=overview_pose,
        objects=build.pack.static_objects,
        camera_safety_config=camera_safety_config,
        mount_xyz_body=mount_xyz_body,
        mount_rpy_body=mount_rpy_body,
    )
    rendered = build.camera.render(rgb=True, depth=render_depth, force_render=True)
    rgb, depth = _extract_render_outputs(rendered)
    rgb = _select_env(rgb, render_env_index)
    depth = _select_env(depth, render_env_index) if depth is not None else None

    return _finalize_env_record(
        rgb,
        depth,
        frame,
        env_index=env_index,
        render_env_index=render_env_index,
        camera_pose=camera_pose,
        camera_safety=camera_safety,
        render_depth=render_depth,
        store_resolution_wh=store_resolution_wh,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        rgb_format=rgb_format,
        camera_mode=camera_mode,
        scene_graph=scene_graph,
        overlay_target_label=overlay_target_label,
    )


def _finalize_env_record(
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    frame: dict[str, Any],
    *,
    env_index: int,
    render_env_index: int,
    camera_pose: Any,
    camera_safety: Any,
    render_depth: bool,
    store_resolution_wh: tuple[int, int] | None,
    rgb_dir: Path,
    depth_dir: Path | None,
    rgb_format: str,
    camera_mode: str,
    scene_graph: Any,
    overlay_target_label: bool,
) -> dict[str, Any]:
    """Validate, overlay, resize, and write one env's rendered frame.

    Shared by the single-frame path and the batched-keep-all path so both
    produce identical per-frame records and on-disk layout.
    """

    rgb = _maybe_overlay_target_label(
        rgb, frame, scene_graph=scene_graph, enabled=overlay_target_label
    )
    frame_index = int(frame["frame_index"])
    stem = f"frame_{frame_index:06d}_env_{env_index:02d}"
    rgb_path = rgb_dir / f"{stem}.{rgb_format}"
    depth_path = None if depth_dir is None else depth_dir / f"{stem}.npy"
    camera_valid, invalid_reasons, rgb_stats, depth_stats = _validate_frame(
        rgb,
        depth,
        require_depth=render_depth,
        camera_safety=camera_safety,
        apply_low_info_gates=camera_mode == "replay",
    )
    low_info_reasons = [
        reason for reason in invalid_reasons if reason in LOW_INFO_REASON_NAMES
    ]
    low_info_allowed = bool(
        low_info_reasons
        and camera_mode == "replay"
        and _is_recovery_context(frame.get("command_context"))
    )
    if low_info_allowed:
        invalid_reasons = [
            reason for reason in invalid_reasons if reason not in LOW_INFO_REASON_NAMES
        ]
        camera_valid = not invalid_reasons
    if rgb is not None:
        rgb_store = (
            _resize_rgb(rgb, store_resolution_wh)
            if store_resolution_wh is not None
            else rgb
        )
        _write_rgb_frame(rgb_store, rgb_path, rgb_format=rgb_format)
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
        "low_info_reasons": low_info_reasons,
        "low_info_allowed": low_info_allowed,
        "rgb_stats": rgb_stats,
        "depth_stats": depth_stats,
        "camera_pose_world": camera_pose,
        "camera_safety": camera_safety,
        "command_context": frame.get("command_context"),
        "source_frame": {
            "source_line": frame.get("source_line"),
            "source_topic": frame.get("source_topic"),
            "canonical_topic": frame.get("canonical_topic"),
        },
    }


def _render_frames_batched(
    frames: list[dict[str, Any]],
    build: Any,
    leg_dof_idx: np.ndarray,
    *,
    rgb_dir: Path,
    depth_dir: Path | None,
    render_depth: bool,
    store_resolution_wh: tuple[int, int] | None,
    rgb_format: str,
    camera_mode: str,
    overview_pose: dict[str, list[float]] | None,
    camera_safety_config: Any,
    mount_xyz_body: tuple[float, float, float],
    mount_rpy_body: tuple[float, float, float],
    scene_graph: Any,
    overlay_target_label: bool,
    on_record: Any,
) -> None:
    """Render all envs per timestep in ONE camera.render() call, keep all.

    Groups source frames by ``frame_index`` (timestep), applies every env's
    base pose / joints / camera pose, renders the batched camera once (rgb
    shape ``(n_envs, H, W, 3)``), then finalizes every present env from that
    single render. This is the efficient path: one render dispatch produces one
    frame per parallel rollout stream instead of one render per stream-frame.
    """

    from collections import defaultdict

    # Group by sim timestamp (shared across envs) so each render() call covers
    # all envs at one timestep. frame_index is globally unique per (env, step),
    # so grouping on it would defeat batching (one render per frame).
    groups: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for frame in frames:
        groups[int(frame["timestamp_ns"])][int(frame.get("env_index") or 0)] = frame

    for timestamp_ns in sorted(groups):
        group = groups[timestamp_ns]
        pose_by_env: dict[int, tuple[Any, Any]] = {}
        for env_index, frame in group.items():
            _apply_robot_state(frame, build.robot, leg_dof_idx, env_index)
            camera_pose, camera_safety = _apply_camera_pose(
                frame,
                build.camera,
                env_index,
                camera_mode=camera_mode,
                overview_pose=overview_pose,
                objects=build.pack.static_objects,
                camera_safety_config=camera_safety_config,
                mount_xyz_body=mount_xyz_body,
                mount_rpy_body=mount_rpy_body,
            )
            pose_by_env[env_index] = (camera_pose, camera_safety)

        rendered = build.camera.render(rgb=True, depth=render_depth, force_render=True)
        rgb_all, depth_all = _extract_render_outputs(rendered)

        for env_index, frame in group.items():
            rgb_e = _select_env(rgb_all, env_index)
            depth_e = _select_env(depth_all, env_index) if depth_all is not None else None
            camera_pose, camera_safety = pose_by_env[env_index]
            record = _finalize_env_record(
                rgb_e,
                depth_e,
                frame,
                env_index=env_index,
                render_env_index=env_index,
                camera_pose=camera_pose,
                camera_safety=camera_safety,
                render_depth=render_depth,
                store_resolution_wh=store_resolution_wh,
                rgb_dir=rgb_dir,
                depth_dir=depth_dir,
                rgb_format=rgb_format,
                camera_mode=camera_mode,
                scene_graph=scene_graph,
                overlay_target_label=overlay_target_label,
            )
            on_record(record)


def _is_recovery_context(command_context: Any) -> bool:
    if not isinstance(command_context, dict):
        return False
    return str(command_context.get("command_source") or "") == "recovery"


def _maybe_overlay_target_label(
    rgb: np.ndarray | None,
    frame: dict[str, Any],
    *,
    scene_graph: Any,
    enabled: bool,
) -> np.ndarray | None:
    if rgb is None or not enabled or scene_graph is None:
        return rgb
    cmd_ctx = frame.get("command_context")
    if not isinstance(cmd_ctx, dict):
        return rgb
    target_id = cmd_ctx.get("route_target_id")
    if target_id is None or int(target_id) < 0:
        return rgb

    target_name = "unknown"
    for name, cell in scene_graph.landmark_cells:
        if cell == int(target_id):
            target_name = str(name)
            break

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    text = f"Target: {target_name} ({target_id})"
    draw.rectangle([5, 5, 220, 25], fill=(0, 0, 0, 128))
    draw.text((10, 10), text, fill=(255, 255, 255))
    return np.array(img)


def _resize_rgb(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Downsample the rendered RGB to the stored (training) resolution.

    Rendering happens at the camera's native resolution (for correct projection
    and the depth-clipping validity gate); the encoder consumes a smaller image,
    so the stored frame is resized here with Lanczos resampling.
    """

    w, h = int(size_wh[0]), int(size_wh[1])
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    return np.asarray(
        Image.fromarray(arr).resize((w, h), Image.LANCZOS), dtype=np.uint8
    )


def _write_rgb_frame(rgb: np.ndarray, path: Path, *, rgb_format: str) -> None:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_format == "png":
        Image.fromarray(rgb_u8).save(path)
        return
    if rgb_format == "npy":
        np.save(path, rgb_u8)
        return
    raise ValueError(f"unsupported rgb_format: {rgb_format!r}")


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


def _apply_camera_pose(
    frame: dict[str, Any],
    camera: Any,
    env_index: int,
    *,
    camera_mode: str,
    overview_pose: dict[str, list[float]] | None,
    objects: Any,
    camera_safety_config: Any,
    mount_xyz_body: tuple[float, float, float],
    mount_rpy_body: tuple[float, float, float],
) -> tuple[dict[str, list[float]], dict[str, float | bool]]:
    if camera_mode == "overview":
        if overview_pose is None:
            raise ValueError("overview camera mode requires an overview pose")
        pose = overview_pose
        safety: dict[str, float | bool] = {"unsafe": False, "retracted_m": 0.0}
    else:
        pose = frame.get("camera_pose_world")
    if camera_mode != "replay" and not pose:
        raise ValueError(f"frame {frame.get('frame_index')} missing camera_pose_world")
    if camera_mode == "replay":
        base_pose = frame.get("base_pose_world", {})
        position = base_pose.get("position", {})
        quat_xyzw = frame.get("base_quat_world_xyzw")
        if not position or quat_xyzw is None:
            raise ValueError(f"frame {frame.get('frame_index')} missing base pose/quaternion")
        adjusted, safety = safe_camera_pose_from_base(
            (
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            ),
            quat_xyzw,
            mount_xyz_body=mount_xyz_body,
            mount_rpy_body=mount_rpy_body,
            objects=objects,
            config=camera_safety_config,
        )
        pose = adjusted.to_dict()
    pos = np.asarray(pose["position"], dtype=np.float32)
    lookat = np.asarray(pose["lookat"], dtype=np.float32)
    up = np.asarray(pose["up"], dtype=np.float32)
    if bool(getattr(camera, "_is_batched", False)):
        camera.set_pose(pos=pos[None, :], lookat=lookat[None, :], up=up[None, :], envs_idx=[env_index])
    else:
        camera.set_pose(pos=pos, lookat=lookat, up=up)
    return pose, safety


def _overview_camera_pose(pack: Any) -> dict[str, list[float]]:
    (min_x, min_y), (max_x, max_y) = pack.world_bounds_xy_m
    center_x = (float(min_x) + float(max_x)) * 0.5
    center_y = (float(min_y) + float(max_y)) * 0.5
    span = max(float(max_x) - float(min_x), float(max_y) - float(min_y), 1.0)
    height = max(5.0, span * 1.1)
    return {
        "position": [center_x, center_y, height],
        "lookat": [center_x, center_y, 0.0],
        "up": [0.0, 1.0, 0.0],
    }


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
    camera_safety: dict[str, float | bool] | None,
    apply_low_info_gates: bool,
) -> tuple[
    bool,
    list[str],
    dict[str, float | int | bool | None],
    dict[str, float | int | bool | None],
]:
    quality = assess_rendered_frame(
        rgb,
        depth,
        require_depth=bool(require_depth),
        camera_safety=camera_safety,
    )
    reasons = list(quality["invalid_reasons"])
    if not apply_low_info_gates:
        reasons = [reason for reason in reasons if reason not in LOW_INFO_REASON_NAMES]
    return (
        not reasons,
        reasons,
        dict(quality["rgb_stats"]),
        dict(quality["depth_stats"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
