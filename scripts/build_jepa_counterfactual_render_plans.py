#!/usr/bin/env python3
"""Build render plans for matched JEPA counterfactual future observations.

The counterfactual decision benchmark contains privileged kinematic
consequences but no future images for the branched actions. This script emits
one replay-compatible frame at every action-block endpoint so a spatial JEPA
can be trained against matched action-conditioned future observations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TextIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.actions import active_block_to_matrix  # noqa: E402
from lewm.benchmarks.counterfactual import Pose2D, integrate_action_blocks  # noqa: E402


def _quat_xyzw_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _uniform_candidate_indices(row: dict, max_candidates: int) -> list[int]:
    count = len(row["counterfactual_candidates"])
    if max_candidates <= 0 or max_candidates >= count:
        return list(range(count))
    oracle_index = int(row["counterfactual_oracle_index"])
    selected = {
        int(index)
        for index in np.linspace(0, count - 1, num=max(1, max_candidates), dtype=np.int64)
    }
    if oracle_index not in selected:
        selected.remove(max(selected, key=lambda index: abs(index - oracle_index)))
        selected.add(oracle_index)
    for index in range(count):
        if len(selected) >= max_candidates:
            break
        selected.add(index)
    return sorted(selected)


def _outcome_bucket(candidate: dict) -> str:
    if bool(candidate["enters_grid_unsafe"]) or bool(candidate["ends_grid_unsafe"]):
        return "kinematic_unsafe"
    progress = candidate.get("target_progress_m")
    if (
        progress is not None
        and float(progress) > 0.0
        and candidate.get("target_recoverable") is not False
    ):
        return "safe_positive_progress"
    return "safe_other"


def _evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return values
    positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return [values[int(position)] for position in positions]


def _selected_candidate_indices(
    row: dict,
    max_candidates: int,
    selection: str = "uniform",
) -> list[int]:
    if selection == "uniform":
        return _uniform_candidate_indices(row, max_candidates)
    if selection != "outcome_stratified":
        raise ValueError(f"unsupported candidate selection strategy: {selection}")

    candidates = row["counterfactual_candidates"]
    count = len(candidates)
    if max_candidates <= 0 or max_candidates >= count:
        return list(range(count))
    oracle_index = int(row["counterfactual_oracle_index"])
    bucket_names = ("safe_positive_progress", "kinematic_unsafe", "safe_other")
    buckets = {
        name: [
            index
            for index, candidate in enumerate(candidates)
            if _outcome_bucket(candidate) == name
        ]
        for name in bucket_names
    }
    selected = {oracle_index}
    base_quota, remainder = divmod(max_candidates, len(bucket_names))
    for bucket_index, name in enumerate(bucket_names):
        quota = base_quota + int(bucket_index < remainder)
        values = [index for index in buckets[name] if index != oracle_index]
        needed = max(0, quota - int(oracle_index in buckets[name]))
        selected.update(_evenly_spaced(values, needed))

    if len(selected) < max_candidates:
        remaining = [index for index in range(count) if index not in selected]
        selected.update(_evenly_spaced(remaining, max_candidates - len(selected)))
    if len(selected) > max_candidates:
        removable = [index for index in sorted(selected, reverse=True) if index != oracle_index]
        for index in removable[: len(selected) - max_candidates]:
            selected.remove(index)
    return sorted(selected)


def _frame(
    *,
    frame_index: int,
    row: dict,
    source_index: int,
    candidate_index: int,
    block_index: int,
    endpoint: Pose2D,
) -> dict:
    start_position = row["start_base_pose_world"]["position"]
    start_rpy = row["start_base_rpy_rad"]
    roll = float(start_rpy["roll"])
    pitch = float(start_rpy["pitch"])
    quat_xyzw = _quat_xyzw_from_rpy(roll, pitch, endpoint.yaw_rad)
    candidate = row["counterfactual_candidates"][candidate_index]
    primitive_sequence = candidate["primitive_sequence"]
    timestamp_ns = (frame_index + 1) * 100_000_000
    lineage = row.get("phase2d_source_state_lineage") or {}
    return {
        "frame_index": frame_index,
        "env_index": 0,
        "timestamp_ns": timestamp_ns,
        "timestamp_s": timestamp_ns / 1_000_000_000,
        "base_pose_world": {
            "position": {
                "x": endpoint.x_m,
                "y": endpoint.y_m,
                "z": float(start_position["z"]),
            },
            "orientation": {
                "x": quat_xyzw[0],
                "y": quat_xyzw[1],
                "z": quat_xyzw[2],
                "w": quat_xyzw[3],
            },
        },
        "base_quat_world_xyzw": quat_xyzw,
        "base_rpy_rad": {
            "roll": roll,
            "pitch": pitch,
            "yaw": endpoint.yaw_rad,
        },
        "command_context": {
            "command_source": "jepa_counterfactual_kinematic",
            "primitive_name": primitive_sequence[block_index],
            "block_size": int(row["block_size"]),
            "command_dt_s": float(row["command_dt_s"]),
        },
        "counterfactual_context": {
            "benchmark_schema": row["benchmark_schema"],
            "source_index": source_index,
            "source_start_frame": row["start_frame"],
            "scene_id": row.get("scene_id"),
            "topology_seed": row.get("topology_seed"),
            "visual_seed": row.get("visual_seed"),
            "phase2d_lineage_verified": bool(
                lineage.get("lineage_verified", False)
            ),
            "candidate_index": candidate_index,
            "primitive_sequence": primitive_sequence,
            "block_index": block_index,
            "is_oracle_candidate": candidate_index
            == int(row["counterfactual_oracle_index"]),
            "physics_validated": False,
        },
    }


def _write_plan(scene: dict, output_root: Path) -> Path:
    scene_dir = output_root / scene["split"] / scene["family"] / scene["scene_id"]
    frames_path = scene_dir / "frames.jsonl"
    plan = {
        "schema": "lewm_render_replay_plan_v0",
        "counterfactual_schema": "jepa_counterfactual_render_plan_v0",
        "backend": "genesis",
        "gpu_required": True,
        "render_status": "planned_not_rendered",
        "scene_id": scene["scene_id"],
        "scene_family": scene["family"],
        "split": scene["split"],
        "scene_manifest": scene["scene_manifest"],
        "topology_seed": scene.get("topology_seed"),
        "visual_seed": scene.get("visual_seed"),
        "output_dir": str(scene_dir.resolve()),
        "frames_jsonl": str(frames_path.resolve()),
        "frame_count": scene["frame_count"],
        "source_env_count": 1,
        "first_frame_timestamp_ns": 100_000_000,
        "last_frame_timestamp_ns": scene["frame_count"] * 100_000_000,
        "backend_contract": {
            "must_preserve_frame_order": True,
            "must_write_camera_valid_flags": True,
            "depth_required_for_validity_audit": True,
        },
    }
    plan_path = scene_dir / "render_replay_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--max-rows-per-scene",
        type=int,
        default=0,
        help="Maximum selected source states per scene; 0 keeps all.",
    )
    parser.add_argument(
        "--max-candidates-per-row",
        type=int,
        default=0,
        help="Deterministic bounded subset including the oracle; 0 emits all candidates.",
    )
    parser.add_argument(
        "--candidate-selection",
        choices=("uniform", "outcome_stratified"),
        default="outcome_stratified",
        help="Bounded candidate subset strategy; ignored when all candidates are emitted.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    scenes: dict[str, dict] = {}
    streams: dict[str, TextIO] = {}
    input_rows = 0
    output_rows = 0
    candidate_count = 0
    skipped_scene_cap_rows = 0
    rows_by_scene: dict[str, int] = {}
    try:
        with args.input.open() as source:
            for source_index, line in enumerate(source):
                if source_index < args.start_row:
                    continue
                if args.max_rows > 0 and output_rows >= args.max_rows:
                    break
                input_rows += 1
                row = json.loads(line)
                scene_id = str(row["scene_id"])
                if (
                    args.max_rows_per_scene > 0
                    and rows_by_scene.get(scene_id, 0) >= args.max_rows_per_scene
                ):
                    skipped_scene_cap_rows += 1
                    continue
                if scene_id not in scenes:
                    scene_dir = (
                        args.output_root
                        / str(row["split"])
                        / str(row["family"])
                        / scene_id
                    )
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    frames_path = scene_dir / "frames.jsonl"
                    if frames_path.exists() and not args.overwrite:
                        raise SystemExit(
                            f"counterfactual render frames already exist: {frames_path}; "
                            "pass --overwrite"
                        )
                    streams[scene_id] = frames_path.open("w")
                    scenes[scene_id] = {
                        "scene_id": scene_id,
                        "family": str(row["family"]),
                        "split": str(row["split"]),
                        "scene_manifest": str(row["scene_manifest"]),
                        "topology_seed": row.get("topology_seed"),
                        "visual_seed": row.get("visual_seed"),
                        "frame_count": 0,
                        "row_count": 0,
                        "candidate_count": 0,
                    }
                scene = scenes[scene_id]
                selected = _selected_candidate_indices(
                    row,
                    args.max_candidates_per_row,
                    args.candidate_selection,
                )
                start_position = row["start_base_pose_world"]["position"]
                start = Pose2D(
                    x_m=float(start_position["x"]),
                    y_m=float(start_position["y"]),
                    yaw_rad=float(row["start_base_rpy_rad"]["yaw"]),
                )
                for candidate_index in selected:
                    candidate = row["counterfactual_candidates"][candidate_index]
                    endpoints = integrate_action_blocks(
                        action_blocks=[
                            active_block_to_matrix(block)
                            for block in candidate["active_blocks"]
                        ],
                        start=start,
                        command_dt_s=float(row["command_dt_s"]),
                    )
                    for block_index, endpoint in enumerate(endpoints):
                        frame = _frame(
                            frame_index=scene["frame_count"],
                            row=row,
                            source_index=source_index,
                            candidate_index=candidate_index,
                            block_index=block_index,
                            endpoint=endpoint,
                        )
                        streams[scene_id].write(
                            json.dumps(frame, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        scene["frame_count"] += 1
                    candidate_count += 1
                    scene["candidate_count"] += 1
                output_rows += 1
                scene["row_count"] += 1
                rows_by_scene[scene_id] = rows_by_scene.get(scene_id, 0) + 1
    finally:
        for stream in streams.values():
            stream.close()

    if not scenes:
        raise SystemExit("no counterfactual render plans generated")
    plan_paths = [_write_plan(scene, args.output_root) for scene in scenes.values()]
    summary = {
        "schema": "jepa_counterfactual_render_plan_summary_v0",
        "input": str(args.input.resolve()),
        "output_root": str(args.output_root.resolve()),
        "start_row": args.start_row,
        "max_rows": args.max_rows,
        "max_rows_per_scene": args.max_rows_per_scene,
        "max_candidates_per_row": args.max_candidates_per_row,
        "candidate_selection": args.candidate_selection,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "skipped_scene_cap_rows": skipped_scene_cap_rows,
        "scene_count": len(scenes),
        "rows_by_scene": rows_by_scene,
        "candidate_count": candidate_count,
        "frame_count": sum(scene["frame_count"] for scene in scenes.values()),
        "plans": [str(path.resolve()) for path in plan_paths],
        "label_status": (
            "kinematic endpoint observations; physical replay calibration required"
        ),
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
