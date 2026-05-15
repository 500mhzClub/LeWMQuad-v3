"""Replay scheduling helpers for deterministic GPU rendering.

The functions in this module deliberately avoid importing Genesis at module
import time. They define the replay contract consumed by a GPU renderer:
``raw_rollout`` state and command data in, deterministic per-frame camera/base
poses out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def replay_frame_schedule(
    command_timestamps_s: list[float],
    camera_hz: float,
    max_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic render-frame times aligned to command timestamps."""

    if camera_hz <= 0.0:
        raise ValueError("camera_hz must be positive")
    if not command_timestamps_s:
        return []

    start_s = min(command_timestamps_s)
    end_s = max(command_timestamps_s) if max_time_s is None else min(max(command_timestamps_s), max_time_s)
    period_s = 1.0 / camera_hz
    frames: list[dict[str, Any]] = []
    frame_index = 0
    t = start_s
    while t <= end_s + 1e-9:
        frames.append({"frame_index": frame_index, "timestamp_s": round(t, 9)})
        frame_index += 1
        t = start_s + frame_index * period_s
    return frames


def build_render_replay_plan(
    raw_rollout_dir: str | Path,
    output_dir: str | Path,
    *,
    backend: str = "genesis",
    camera_hz: float = 10.0,
    platform_manifest: str | Path | None = None,
    max_frames: int | None = None,
    require_quality_pass: bool = True,
) -> dict[str, Any]:
    """Write a GPU render-replay plan for a converted ``raw_rollout``.

    The output is not a rendered dataset. It is the immutable job contract that a
    backend such as Genesis must consume to produce ``rendered_vision``. Keeping
    this step separate lets Gazebo remain the dynamics oracle while rendering
    can scale on GPU hardware.
    """

    raw_dir = Path(raw_rollout_dir).resolve()
    out_dir = Path(output_dir).resolve()
    summary_path = raw_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"raw_rollout summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_quality = summary.get("data_quality_audit", {})
    contract = summary.get("contract_audit", {})
    if require_quality_pass and not contract.get("pass", False):
        raise ValueError(f"raw_rollout contract audit failed: {contract.get('issues', [])}")
    if require_quality_pass and not data_quality.get("pass", False):
        raise ValueError(f"raw_rollout data-quality audit failed: {data_quality.get('issues', [])}")

    messages_path = Path(summary["messages_jsonl"])
    if not messages_path.is_absolute():
        messages_path = raw_dir / messages_path
    platform = _load_platform_manifest(platform_manifest)
    camera = platform.get("camera", {})

    frames = _extract_base_state_frames(
        messages_path,
        camera_hz=camera_hz,
        max_frames=max_frames,
        camera_mount_body=_camera_mount(camera),
    )
    if not frames:
        raise ValueError(f"no /lewm/go2/base_state frames found in {messages_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    plan = {
        "schema": "lewm_render_replay_plan_v0",
        "backend": backend,
        "gpu_required": True,
        "render_status": "planned_not_rendered",
        "raw_rollout_dir": str(raw_dir),
        "raw_summary_json": str(summary_path),
        "source_messages_jsonl": str(messages_path),
        "output_dir": str(out_dir),
        "frames_jsonl": str(frames_path),
        "frame_count": len(frames),
        "camera_hz": camera_hz,
        "first_frame_timestamp_ns": frames[0]["timestamp_ns"],
        "last_frame_timestamp_ns": frames[-1]["timestamp_ns"],
        "raw_contract_audit_pass": contract.get("pass", False),
        "raw_data_quality_audit_pass": data_quality.get("pass", False),
        "raw_quality_profile": data_quality.get("profile"),
        "camera": {
            "native_resolution": camera.get("native_resolution"),
            "training_resolution": camera.get("training_resolution"),
            "fov_axis": camera.get("fov_axis"),
            "fov_deg": camera.get("fov_deg"),
            "near_m": camera.get("near_m"),
            "far_m": camera.get("far_m"),
            "encoding": camera.get("encoding"),
            "mount_body": _camera_mount(camera),
        },
        "backend_contract": {
            "must_preserve_frame_order": True,
            "must_write_camera_valid_flags": True,
            "must_not_modify_reset_or_command_arrays": True,
            "must_report_invalid_frame_rate": True,
            "depth_required_for_validity_audit": True,
        },
    }
    plan_path = out_dir / "render_replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def _extract_base_state_frames(
    messages_path: Path,
    *,
    camera_hz: float,
    max_frames: int | None,
    camera_mount_body: dict[str, Any],
) -> list[dict[str, Any]]:
    if camera_hz <= 0.0:
        raise ValueError("camera_hz must be positive")

    period_ns = int(round(1_000_000_000 / camera_hz))
    next_frame_ns: int | None = None
    frames: list[dict[str, Any]] = []
    latest_episode: dict[str, Any] = {}
    source_line = 0

    with messages_path.open(encoding="utf-8") as stream:
        for line in stream:
            source_line += 1
            record = json.loads(line)
            topic = record.get("topic")
            payload = record.get("payload", {})
            if topic == "/lewm/episode_info":
                latest_episode = {
                    "scene_id": payload.get("scene_id"),
                    "episode_id": payload.get("episode_id"),
                    "episode_step": payload.get("episode_step"),
                    "reset_count": payload.get("reset_count"),
                    "split": payload.get("split"),
                    "manifest_sha256": payload.get("manifest_sha256"),
                }
                continue
            if topic != "/lewm/go2/base_state":
                continue

            timestamp_ns = _payload_stamp_ns(payload)
            if timestamp_ns is None:
                timestamp_ns = int(record["timestamp_ns"])
            if next_frame_ns is None:
                next_frame_ns = timestamp_ns
            if timestamp_ns + 1_000 < next_frame_ns:
                continue

            frames.append(
                {
                    "frame_index": len(frames),
                    "source_line": source_line,
                    "source_topic": topic,
                    "timestamp_ns": timestamp_ns,
                    "timestamp_s": round(timestamp_ns / 1e9, 9),
                    "record_timestamp_ns": int(record["timestamp_ns"]),
                    "episode": dict(latest_episode),
                    "base_pose_world": payload.get("pose_world"),
                    "base_quat_world_xyzw": payload.get("quat_world_xyzw"),
                    "base_rpy_rad": {
                        "roll": payload.get("roll_rad"),
                        "pitch": payload.get("pitch_rad"),
                        "yaw": payload.get("yaw_rad"),
                    },
                    "twist_body": payload.get("twist_body"),
                    "camera_mount_body": camera_mount_body,
                }
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
            next_frame_ns = timestamp_ns + period_ns

    return frames


def _load_platform_manifest(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"platform manifest not found: {manifest_path}")
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def _camera_mount(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_link": camera.get("parent_link"),
        "xyz_body_m": camera.get("xyz_body_m"),
        "rpy_body_rad": camera.get("rpy_body_rad"),
    }


def _payload_stamp_ns(payload: dict[str, Any]) -> int | None:
    stamp = payload.get("header", {}).get("stamp")
    if not isinstance(stamp, dict):
        return None
    return int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nanosec", 0))
