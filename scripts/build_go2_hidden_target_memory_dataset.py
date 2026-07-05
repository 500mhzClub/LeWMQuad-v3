#!/usr/bin/env python3
"""Join rendered Go2 RGB frames with derived hidden-target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rendered_vision", type=Path, help="Rendered vision directory.")
    parser.add_argument("labels", type=Path, help="Derived labels.jsonl.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include frames that failed the render camera-validity gate.",
    )
    args = parser.parse_args()

    rendered = args.rendered_vision.resolve()
    frames_path = rendered / "frames_rendered.jsonl"
    if not frames_path.is_file():
        raise SystemExit(f"missing frames_rendered.jsonl: {frames_path}")
    labels_path = _resolve_labels(args.labels)
    label_index = _load_labels(labels_path)

    rows = []
    missing_labels = 0
    invalid_skipped = 0
    with frames_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            frame = json.loads(line)
            if not args.include_invalid and not bool(frame.get("camera_valid", False)):
                invalid_skipped += 1
                continue
            key = (
                int(frame.get("env_index", 0)),
                int(frame.get("timestamp_ns", 0)),
            )
            label = label_index.get(key)
            if label is None:
                missing_labels += 1
                continue
            rows.append(_dataset_row(frame, label, rendered_root=rendered))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    summary = {
        "schema": "lewm_go2_hidden_target_memory_dataset_v0",
        "rendered_vision": str(rendered),
        "labels": str(labels_path),
        "out": str(args.out),
        "row_count": len(rows),
        "missing_label_count": int(missing_labels),
        "invalid_frame_skipped_count": int(invalid_skipped),
        "include_invalid": bool(args.include_invalid),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "go2_hidden_target_dataset:"
        f" rows={summary['row_count']}"
        f" missing_labels={summary['missing_label_count']}"
        f" invalid_skipped={summary['invalid_frame_skipped_count']}"
        f" out={args.out}"
    )
    return 0


def _resolve_labels(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    candidate = path / "labels.jsonl"
    if candidate.is_file():
        return candidate
    candidate = path / "derived_labels" / "labels.jsonl"
    if candidate.is_file():
        return candidate
    raise SystemExit(f"missing labels.jsonl under: {path}")


def _load_labels(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    labels: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("env_idx", 0)), int(row.get("timestamp_ns", 0)))
            labels[key] = row
    return labels


def _dataset_row(
    frame: dict[str, Any],
    label: dict[str, Any],
    *,
    rendered_root: Path,
) -> dict[str, Any]:
    rgb_path = Path(str(frame.get("rgb_path", "")))
    if not rgb_path.is_absolute():
        rgb_path = rendered_root / rgb_path
    command = frame.get("command_context") or {}
    landmarks = list(label.get("landmarks") or ())
    visible_landmark_ids = [
        str(landmark.get("object_id", ""))
        for landmark in landmarks
        if bool(landmark.get("visible", False))
    ]
    hidden_landmark_ids = [
        str(landmark.get("object_id", ""))
        for landmark in landmarks
        if not bool(landmark.get("visible", False))
    ]
    return {
        "schema": "lewm_go2_hidden_target_memory_row_v0",
        "rgb_path": str(rgb_path),
        "camera_valid": bool(frame.get("camera_valid", False)),
        "invalid_reasons": list(frame.get("invalid_reasons") or ()),
        "go2_hidden_target_memory_selection": list(
            frame.get("go2_hidden_target_memory_selection") or ()
        ),
        "go2_causal_memory_pair_selection": list(
            frame.get("go2_causal_memory_pair_selection") or ()
        ),
        "timestamp_ns": int(frame.get("timestamp_ns", 0)),
        "timestamp_s": float(frame.get("timestamp_s", 0.0)),
        "env_idx": int(frame.get("env_index", 0)),
        "episode_id": int(label.get("episode_id", 0)),
        "episode_step": int(label.get("episode_step", 0)),
        "scene_id": str(label.get("scene_id", "")),
        "cell_id": int(label.get("cell_id", -1)),
        "yaw_bin": int(label.get("yaw_bin", -1)),
        "clearance_m": float(label.get("clearance_m", 0.0)),
        "traversability_forward_m": float(label.get("traversability_forward_m", 0.0)),
        "integrated_body_motion_block": list(label.get("integrated_body_motion_block", ())),
        "integrated_body_motion_window": list(label.get("integrated_body_motion_window", ())),
        "landmarks": landmarks,
        "visible_landmark_ids": visible_landmark_ids,
        "hidden_landmark_ids": hidden_landmark_ids,
        "command": {
            "sequence_id": _optional_int(command.get("sequence_id"), default=-1),
            "primitive_name": str(command.get("primitive_name", "")),
            "command_source": str(command.get("command_source", "")),
            "route_target_id": _optional_int(command.get("route_target_id"), default=-1),
            "next_waypoint_id": _optional_int(command.get("next_waypoint_id"), default=-1),
            "vx_body_mps": list(command.get("vx_body_mps") or ()),
            "vy_body_mps": list(command.get("vy_body_mps") or ()),
            "yaw_rate_radps": list(command.get("yaw_rate_radps") or ()),
        },
    }


def _optional_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
