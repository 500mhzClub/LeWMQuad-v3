#!/usr/bin/env python3
"""Join JEPA counterfactual actions to matched rendered future observations."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CONSEQUENCE_FIELDS = (
    "starts_grid_unsafe",
    "enters_grid_unsafe",
    "ends_grid_unsafe",
    "unsafe_sample_fraction",
    "minimum_swept_configuration_clearance_m",
    "p05_swept_configuration_clearance_m",
    "clearance_gain_m",
    "target_progress_m",
    "target_heading_error_rad",
    "target_recoverable",
)


def _resolve_from(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _resolve_existing_render_path(parent: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    raw_path = Path(value)
    candidates = [_resolve_from(parent, value)]
    if raw_path.name:
        candidates.append(parent / raw_path.name)
    if raw_path.parent.name and raw_path.name:
        candidates.append(parent / raw_path.parent.name / raw_path.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _rendered_metadata_by_plan(render_root: Path) -> dict[Path, Path]:
    summaries = (
        [render_root / "summary.json"]
        if (render_root / "summary.json").is_file()
        else sorted(render_root.rglob("summary.json"))
    )
    result = {}
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        if summary.get("schema") != "lewm_rendered_vision_v0":
            continue
        plan = _resolve_from(summary_path.parent, str(summary["plan"])).resolve()
        metadata = _resolve_existing_render_path(
            summary_path.parent, str(summary["frames_rendered_jsonl"])
        )
        if metadata is not None and metadata.is_file():
            result[plan] = metadata
    return result


def _future_frame_index(
    plan_root: Path,
    render_root: Path,
) -> tuple[dict[tuple[int, int, int], dict], dict]:
    rendered_by_plan = _rendered_metadata_by_plan(render_root)
    future_frames: dict[tuple[int, int, int], dict] = {}
    stats = Counter()
    for plan_path in sorted(plan_root.rglob("render_replay_plan.json")):
        resolved_plan = plan_path.resolve()
        metadata_path = rendered_by_plan.get(resolved_plan)
        if metadata_path is None:
            stats["plans_without_rendered_metadata"] += 1
            continue
        plan = json.loads(plan_path.read_text())
        frames_path = _resolve_from(plan_path.parent, str(plan["frames_jsonl"]))
        planned = {
            (int(frame["frame_index"]), int(frame.get("env_index") or 0)): frame
            for frame in (json.loads(line) for line in frames_path.open())
        }
        for rendered in (json.loads(line) for line in metadata_path.open()):
            frame_key = (
                int(rendered["frame_index"]),
                int(rendered.get("env_index") or 0),
            )
            frame = planned.get(frame_key)
            if frame is None:
                stats["rendered_frames_without_plan_frame"] += 1
                continue
            context = frame.get("counterfactual_context")
            if context is None:
                stats["planned_frames_without_counterfactual_context"] += 1
                continue
            key = (
                int(context["source_index"]),
                int(context["candidate_index"]),
                int(context["block_index"]),
            )
            rgb_path = rendered.get("rgb_path")
            resolved_rgb = _resolve_existing_render_path(
                metadata_path.parent,
                None if rgb_path is None else str(rgb_path),
            )
            if resolved_rgb is not None and resolved_rgb.is_file():
                rgb_path = str(resolved_rgb)
            future_frames[key] = {
                "rgb_path": rgb_path,
                "camera_valid": bool(rendered["camera_valid"]),
                "invalid_reasons": rendered.get("invalid_reasons", []),
                "camera_safety": rendered.get("camera_safety"),
                "physics_validated": bool(context.get("physics_validated", False)),
                "topology_seed": context.get("topology_seed"),
                "visual_seed": context.get("visual_seed"),
                "phase2d_lineage_verified": bool(
                    context.get("phase2d_lineage_verified", False)
                ),
            }
            stats["rendered_frames_indexed"] += 1
    return future_frames, dict(stats)


def _candidate_bucket(candidate: dict) -> str:
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


def _first_present(values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _lineage_value(row: dict, matched: list[dict | None], field: str):
    return _first_present(
        [row.get(field)]
        + [frame.get(field) for frame in matched if frame is not None]
    )


def build_spatial_future_dataset(
    *,
    benchmark: Path,
    plan_root: Path,
    render_root: Path,
    output: Path,
    max_rows: int = 0,
    include_unplanned_candidates: bool = False,
) -> dict:
    """Write present/action/future tuples without hiding invalid future targets."""

    future_frames, render_stats = _future_frame_index(plan_root, render_root)
    stats = Counter()
    stats["candidate_sequences_written"] = 0
    stats["candidate_sequences_complete_valid"] = 0
    stats["candidate_sequences_incomplete_or_invalid"] = 0
    stats["candidate_sequences_unplanned_skipped"] = 0
    scenes = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with benchmark.open() as source, output.open("w") as destination:
        for source_index, line in enumerate(source):
            if max_rows > 0 and stats["benchmark_rows"] >= max_rows:
                break
            stats["benchmark_rows"] += 1
            row = json.loads(line)
            horizon = int(row["counterfactual_horizon_blocks"])
            for candidate_index, candidate in enumerate(row["counterfactual_candidates"]):
                planned = any(
                    (source_index, candidate_index, block_index) in future_frames
                    for block_index in range(horizon)
                )
                if not planned and not include_unplanned_candidates:
                    stats["candidate_sequences_unplanned_skipped"] += 1
                    continue
                scenes.add(str(row["scene_id"]))
                stats["candidate_sequences_total"] += 1
                bucket = _candidate_bucket(candidate)
                stats[f"candidate_sequences_{bucket}"] += 1
                stats[f"candidate_sequences_{bucket}_complete_valid"] += 0
                matched = [
                    future_frames.get((source_index, candidate_index, block_index))
                    for block_index in range(horizon)
                ]
                missing = sum(frame is None for frame in matched)
                invalid = sum(
                    frame is not None
                    and (
                        not frame["camera_valid"]
                        or frame["rgb_path"] is None
                        or not Path(str(frame["rgb_path"])).is_file()
                    )
                    for frame in matched
                )
                complete_valid = missing == 0 and invalid == 0
                topology_seed = _lineage_value(row, matched, "topology_seed")
                visual_seed = _lineage_value(row, matched, "visual_seed")
                if topology_seed is None:
                    stats["candidate_sequences_missing_topology_seed"] += 1
                if visual_seed is None:
                    stats["candidate_sequences_missing_visual_seed"] += 1
                if missing:
                    stats["candidate_sequences_missing_future_frames"] += 1
                    stats["future_frames_missing"] += missing
                if invalid:
                    stats["candidate_sequences_invalid_future_frames"] += 1
                    stats["future_frames_invalid"] += invalid
                if complete_valid:
                    stats["candidate_sequences_complete_valid"] += 1
                    stats[f"candidate_sequences_{bucket}_complete_valid"] += 1
                else:
                    stats["candidate_sequences_incomplete_or_invalid"] += 1
                payload = {
                    "schema": "jepa_spatial_future_sequence_v1",
                    "source_index": source_index,
                    "scene_id": row["scene_id"],
                    "family": row["family"],
                    "split": row["split"],
                    "scene_manifest": row.get("scene_manifest"),
                    "topology_seed": topology_seed,
                    "visual_seed": visual_seed,
                    "phase2d_source_state_lineage": row.get(
                        "phase2d_source_state_lineage"
                    ),
                    "counterfactual_sequence_grid": row.get(
                        "counterfactual_sequence_grid"
                    ),
                    "start_frame": row["start_frame"],
                    "goal_frame": row.get("local_target_frame"),
                    "goal_present": row.get("counterfactual_target_cell_id") is not None,
                    "candidate_index": candidate_index,
                    "primitive_sequence": candidate["primitive_sequence"],
                    "active_blocks": candidate["active_blocks"],
                    "future_frames": [
                        None if frame is None else frame["rgb_path"] for frame in matched
                    ],
                    "future_observations": [
                        {
                            "block_index": block_index,
                            "rgb_path": None if frame is None else frame["rgb_path"],
                            "observation_valid": bool(
                                frame is not None
                                and frame["camera_valid"]
                                and frame["rgb_path"] is not None
                                and Path(str(frame["rgb_path"])).is_file()
                            ),
                            "invalid_reasons": (
                                ["missing_rendered_frame"]
                                if frame is None
                                else frame["invalid_reasons"]
                            ),
                            "camera_safety": (
                                None if frame is None else frame["camera_safety"]
                            ),
                            "physics_validated": bool(
                                frame is not None and frame["physics_validated"]
                            ),
                        }
                        for block_index, frame in enumerate(matched)
                    ],
                    "complete_valid_future_sequence": complete_valid,
                    "future_observation_event": (
                        "complete_valid_observation_sequence"
                        if complete_valid
                        else "incomplete_or_renderer_invalid_observation_sequence"
                    ),
                    "future_frame_physics_validated": [
                        bool(frame is not None and frame["physics_validated"])
                        for frame in matched
                    ],
                    "is_oracle_candidate": candidate_index
                    == int(row["counterfactual_oracle_index"]),
                    "consequence_labels": {
                        field: candidate[field] for field in CONSEQUENCE_FIELDS
                    },
                }
                destination.write(json.dumps(payload, sort_keys=True) + "\n")
                stats["candidate_sequences_written"] += 1
                stats["future_observation_slots_written"] += horizon

    summary = {
        "schema": "jepa_spatial_future_dataset_summary_v1",
        "benchmark": str(benchmark.resolve()),
        "plan_root": str(plan_root.resolve()),
        "render_root": str(render_root.resolve()),
        "output": str(output.resolve()),
        "scene_count": len(scenes),
        **dict(stats),
        "render_index": render_stats,
        "contract": {
            "state_input": "single current RGB observation",
            "action_input": "ordered active command blocks",
            "prediction_target": (
                "ordered rendered RGB future observations when valid, plus explicit "
                "observation-validity event metadata"
            ),
            "privileged_consequences_are_evaluation_labels": True,
            "writes_every_candidate_sequence": include_unplanned_candidates,
            "candidate_scope": (
                "all benchmark candidates"
                if include_unplanned_candidates
                else "planned candidates only"
            ),
            "writes_every_planned_candidate_sequence": True,
            "token_prediction_requires_valid_future_observation": True,
            "renderer_invalidity_is_not_a_collision_label": True,
            "source_state_lineage_fields_propagated": True,
            "confirmatory_phase2d_requires_topology_and_visual_lineage": True,
        },
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--include-unplanned-candidates", action="store_true")
    args = parser.parse_args()

    summary = build_spatial_future_dataset(
        benchmark=args.benchmark,
        plan_root=args.plan_root,
        render_root=args.render_root,
        output=args.output,
        max_rows=args.max_rows,
        include_unplanned_candidates=args.include_unplanned_candidates,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
