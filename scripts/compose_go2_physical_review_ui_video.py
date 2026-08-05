#!/usr/bin/env python3
"""Compose a review UI around an existing Go2 physical-policy split video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from render_go2_closed_loop_result_replay import (
    REPO_ROOT,
    ReplayReviewUi,
    _load_replay_log,
    _pose_entries_from_log,
    _select_scene,
)


def _source_video_meta(path: Path) -> tuple[float, int | None, float | None]:
    import imageio.v3 as iio

    meta = iio.immeta(path)
    fps = float(meta.get("fps") or 30.0)
    duration = meta.get("duration")
    duration_s = None if duration is None else float(duration)
    total_frames = None
    if duration_s is not None and duration_s > 0.0 and fps > 0.0:
        total_frames = max(1, int(round(duration_s * fps)))
    return fps, total_frames, duration_s


def _split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected RGB frame, got shape {frame.shape}")
    rgb = np.asarray(frame[:, :, :3], dtype=np.uint8)
    mid = rgb.shape[1] // 2
    if mid <= 0:
        raise ValueError(f"cannot split frame with width {rgb.shape[1]}")
    return rgb[:, :mid, :], rgb[:, mid:, :]


def _default_title(result: dict[str, Any]) -> str:
    scene = str(result.get("scene", "unknown scene"))
    mode = str(result.get("execution_mode") or "physical")
    gait = "gait" if bool(result.get("gait_executed", True)) else "no gait flag"
    return f"Go2 learned-nav physical review | {scene} | {mode} {gait}"


def _default_status_note(result: dict[str, Any]) -> str:
    success = "success" if bool(result.get("success")) else "not success"
    gait = "gait_executed=true" if bool(result.get("gait_executed", True)) else "gait flag unavailable"
    capture = str(result.get("demo_capture_rate") or "source video")
    return f"actual locomotion-policy frames | {gait} | {capture} capture | {success}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--status-note", default=None)
    parser.add_argument("--ui-width", type=int, default=1280)
    parser.add_argument("--ui-height", type=int, default=720)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    result, log_entries, _primitives = _load_replay_log(args.result.resolve())
    pose_entries = _pose_entries_from_log(log_entries, include_claim_only=True)
    if not pose_entries:
        raise SystemExit("result log has no post_xy/post_yaw entries for minimap/path UI")

    source_video = args.source_video.resolve()
    output_video = args.output_video.resolve()
    if not source_video.exists():
        raise SystemExit(f"source video does not exist: {source_video}")

    source_fps, estimated_total_frames, source_duration_s = _source_video_meta(source_video)
    fps = float(args.fps) if args.fps is not None else source_fps
    total_frames_for_mapping = (
        max(1, int(args.frame_limit))
        if args.frame_limit is not None
        else estimated_total_frames
    )

    scene_id = str(args.scene_id or result.get("scene") or "")
    if not scene_id:
        raise SystemExit("scene id missing; pass --scene-id or use a result with result.scene")
    scene_dir = _select_scene(
        scene_corpus=args.scene_corpus,
        split=str(args.split),
        family=str(args.family),
        scene_id=scene_id,
    )
    manifest_path = scene_dir / "manifest.json"
    scene_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    ui = ReplayReviewUi(
        result=result,
        pose_entries=pose_entries,
        scene_manifest=scene_manifest,
        frames_per_entry=1,
        total_frames=total_frames_for_mapping,
        title=args.title or _default_title(result),
        status_note=args.status_note or _default_status_note(result),
        width=int(args.ui_width),
        height=int(args.ui_height),
    )

    import imageio

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_video), fps=fps, macro_block_size=8)
    frame_count = 0
    try:
        for frame in imageio.get_reader(str(source_video)):
            if args.frame_limit is not None and frame_count >= int(args.frame_limit):
                break
            third_np, ego_np = _split_frame(np.asarray(frame))
            entry = ui.entry_for_frame(frame_count)
            xy = entry.get("post_xy", [0.0, 0.0])
            x = float(xy[0]) if isinstance(xy, list) and len(xy) >= 1 else 0.0
            y = float(xy[1]) if isinstance(xy, list) and len(xy) >= 2 else 0.0
            yaw = float(entry.get("post_yaw", 0.0))
            writer.append_data(
                ui.compose(
                    third_np=third_np,
                    ego_np=ego_np,
                    x=x,
                    y=y,
                    yaw=yaw,
                    frame_index=frame_count,
                    fps=fps,
                )
            )
            frame_count += 1
            if args.progress_every > 0 and frame_count % int(args.progress_every) == 0:
                print(f"composed {frame_count} frames", flush=True)
    finally:
        writer.close()

    duration_s = float(frame_count) / max(1.0, fps)
    report = {
        "source_result": str(args.result),
        "source_video": str(args.source_video),
        "output_video": str(args.output_video),
        "scene": scene_id,
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_s": duration_s,
        "source_duration_s": source_duration_s,
        "estimated_source_frames": estimated_total_frames,
        "pose_entries": len(pose_entries),
        "execution_mode": result.get("execution_mode"),
        "gait_executed": result.get("gait_executed"),
        "success": result.get("success"),
        "claimed": result.get("claimed"),
        "claimed_colors": result.get("claimed_colors"),
        "controller_beacon_claims": result.get(
            "controller_beacon_claims",
            result.get("beacon_claims"),
        ),
        "wall_metrics": {
            key: result.get("wall_metrics", {}).get(key)
            for key in (
                "fall_events",
                "tip_events",
                "unstable_base_events",
                "base_z_min_m",
                "max_abs_roll_pitch_rad",
                "contact_like_stalls",
                "hard_contact_like_stalls",
                "body_clearance_violation_events",
            )
        },
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
