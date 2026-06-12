#!/usr/bin/env python3
"""Audit whether existing rollout data can support task-aligned navigation mining."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _add_counter(target: Counter, values: dict | None) -> None:
    for key, value in (values or {}).items():
        target[str(key)] += int(value)


def _companion(label_summary: Path, kind: str, filename: str = "summary.json") -> Path:
    chunk = label_summary.parents[2]
    scene_id = label_summary.parent.name
    return chunk / kind / scene_id / filename


def _sample_label_files(
    summaries: list[Path],
    *,
    per_split_family: int,
) -> list[Path]:
    selected: list[Path] = []
    counts: Counter[tuple[str, str]] = Counter()
    for summary in summaries:
        payload = json.loads(summary.read_text())
        scene = payload.get("scene", {})
        key = (str(scene.get("split", "unknown")), str(scene.get("family", "unknown")))
        if counts[key] >= per_split_family:
            continue
        labels = summary.parent / "labels.jsonl"
        if labels.is_file():
            selected.append(labels)
            counts[key] += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path(".generated/datagen_full/rollout"),
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=Path(".generated/datagen_full/render_textured_v03"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".generated/audits/task_aligned_data_readiness.json"),
    )
    parser.add_argument("--sample-scenes-per-split-family", type=int, default=4)
    parser.add_argument("--sample-message-scenes-per-split-family", type=int, default=1)
    parser.add_argument("--near-collision-m", type=float, default=0.30)
    args = parser.parse_args()

    label_summaries = sorted(args.rollout_root.rglob("labels/*/summary.json"))
    if not label_summaries:
        raise SystemExit(f"no label summaries under {args.rollout_root}")

    exact = {
        "scenes": len(label_summaries),
        "label_rows": 0,
        "command_blocks": 0,
        "executed_command_blocks": 0,
        "missing_commands": 0,
        "missing_episode_info": 0,
        "rendered_scenes": 0,
        "rendered_frames": 0,
        "route_completions": 0,
        "goal_target_changes": 0,
        "recovery_interlock_handoffs": 0,
    }
    split_scenes = Counter()
    family_scenes = Counter()
    graph_type_rows = Counter()
    graph_type_nodes = Counter()
    command_sources = Counter()
    primitives = Counter()
    schema_gaps: list[str] = []
    source_scene_counts = Counter()

    for label_summary in label_summaries:
        label_payload = json.loads(label_summary.read_text())
        scene = label_payload.get("scene", {})
        split = str(scene.get("split", "unknown"))
        family = str(scene.get("family", "unknown"))
        scene_id = str(scene.get("scene_id", label_summary.parent.name))
        split_scenes[split] += 1
        family_scenes[family] += 1
        exact["label_rows"] += int(label_payload.get("label_count", 0))
        _add_counter(graph_type_rows, label_payload.get("local_graph_type_counts"))
        _add_counter(graph_type_nodes, scene.get("local_graph_type_histogram"))
        pose_join = label_payload.get("pose_join", {})
        exact["command_blocks"] += int(pose_join.get("command_block_count", 0))
        exact["executed_command_blocks"] += int(
            pose_join.get("executed_command_block_count", 0)
        )
        exact["missing_commands"] += int(pose_join.get("missing_command_count", 0))
        exact["missing_episode_info"] += int(
            pose_join.get("missing_episode_info_count", 0)
        )

        rollout_summary = _companion(label_summary, "rollout")
        raw_summary = _companion(label_summary, "raw")
        plan_matches = list(
            (label_summary.parents[2] / "plan").glob(
                f"*_{scene_id}/render_replay_plan.json"
            )
        )
        if not rollout_summary.is_file() or not raw_summary.is_file() or not plan_matches:
            schema_gaps.append(scene_id)
            continue
        rollout_payload = json.loads(rollout_summary.read_text())
        per_env = (
            rollout_payload.get("extra", {})
            .get("rollout_stats", {})
            .get("per_env_metrics", [])
        )
        scene_sources: set[str] = set()
        for metric in per_env:
            _add_counter(command_sources, metric.get("command_source_counts"))
            _add_counter(primitives, metric.get("primitive_counts"))
            scene_sources.update((metric.get("command_source_counts") or {}).keys())
            exact["route_completions"] += int(metric.get("route_completions", 0))
            exact["goal_target_changes"] += int(metric.get("goal_target_changes", 0))
            exact["recovery_interlock_handoffs"] += int(
                metric.get("recovery_interlock_handoffs", 0)
            )
        for source in scene_sources:
            source_scene_counts[str(source)] += 1

        render_summary = args.render_root / scene_id / "summary.json"
        if render_summary.is_file():
            render_payload = json.loads(render_summary.read_text())
            if render_payload.get("render_status") == "complete":
                exact["rendered_scenes"] += 1
                exact["rendered_frames"] += int(render_payload.get("frame_count", 0))

    sampled_label_files = _sample_label_files(
        label_summaries,
        per_split_family=args.sample_scenes_per_split_family,
    )
    sampled = {
        "scenes": 0,
        "label_rows": 0,
        "near_collision_rows": 0,
        "stuck_rows": 0,
        "branch_rows": 0,
        "dead_end_rows": 0,
        "visible_goal_rows": 0,
    }
    sampled_by_split_family: dict[str, Counter] = defaultdict(Counter)
    for labels_path in sampled_label_files:
        summary = json.loads((labels_path.parent / "summary.json").read_text())
        scene = summary.get("scene", {})
        key = f"{scene.get('split', 'unknown')}/{scene.get('family', 'unknown')}"
        sampled["scenes"] += 1
        with labels_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                sampled["label_rows"] += 1
                sampled_by_split_family[key]["label_rows"] += 1
                if float(row.get("clearance_m", 1e9)) < args.near_collision_m:
                    sampled["near_collision_rows"] += 1
                    sampled_by_split_family[key]["near_collision_rows"] += 1
                if bool(row.get("stuck_label", False)):
                    sampled["stuck_rows"] += 1
                    sampled_by_split_family[key]["stuck_rows"] += 1
                graph_type = str(row.get("local_graph_type", "unknown"))
                if graph_type in {"t_junction", "crossroad"}:
                    sampled["branch_rows"] += 1
                    sampled_by_split_family[key]["branch_rows"] += 1
                if graph_type == "dead_end":
                    sampled["dead_end_rows"] += 1
                    sampled_by_split_family[key]["dead_end_rows"] += 1
                if any(bool(item.get("visible")) for item in row.get("landmarks", ())):
                    sampled["visible_goal_rows"] += 1
                    sampled_by_split_family[key]["visible_goal_rows"] += 1

    message_sample = {
        "scenes": 0,
        "command_blocks": 0,
        "commands_with_route_target": 0,
        "commands_with_next_waypoint": 0,
        "executed_command_blocks": 0,
        "clipped_executions": 0,
        "safety_overridden_executions": 0,
    }
    message_command_sources = Counter()
    message_sources_with_route_target = Counter()
    message_sources_with_next_waypoint = Counter()
    message_label_files = _sample_label_files(
        label_summaries,
        per_split_family=args.sample_message_scenes_per_split_family,
    )
    for labels_path in message_label_files:
        messages_path = _companion(labels_path.parent / "summary.json", "raw", "messages.jsonl")
        if not messages_path.is_file():
            continue
        message_sample["scenes"] += 1
        with messages_path.open() as stream:
            for line in stream:
                record = json.loads(line)
                topic = record.get("canonical_topic")
                payload = record.get("payload", {})
                if topic == "/lewm/go2/command_block":
                    source = str(payload.get("command_source", "unknown"))
                    message_sample["command_blocks"] += 1
                    message_command_sources[source] += 1
                    if int(payload.get("route_target_id", -1)) >= 0:
                        message_sample["commands_with_route_target"] += 1
                        message_sources_with_route_target[source] += 1
                    if int(payload.get("next_waypoint_id", -1)) >= 0:
                        message_sample["commands_with_next_waypoint"] += 1
                        message_sources_with_next_waypoint[source] += 1
                elif topic == "/lewm/go2/executed_command_block":
                    message_sample["executed_command_blocks"] += 1
                    if bool(payload.get("clipped", False)):
                        message_sample["clipped_executions"] += 1
                    if bool(payload.get("safety_overridden", False)):
                        message_sample["safety_overridden_executions"] += 1

    sample_rows = max(1, sampled["label_rows"])
    projections = {
        "near_collision_rows": int(
            exact["label_rows"] * sampled["near_collision_rows"] / sample_rows
        ),
        "stuck_rows": int(exact["label_rows"] * sampled["stuck_rows"] / sample_rows),
    }
    branch_rows = graph_type_rows["t_junction"] + graph_type_rows["crossroad"]
    readiness = {
        "new_rollout_collection_required": False,
        "branch_choice_rows_exact": int(branch_rows),
        "branch_choice_macro_equivalent": int(branch_rows // 5),
        "recovery_command_blocks_exact": int(command_sources["recovery"]),
        "loop_revisit_command_blocks_exact": int(command_sources["loop_revisit"]),
        "route_teacher_command_blocks_exact": int(command_sources["route_teacher"]),
        "frontier_command_blocks_exact": int(command_sources["frontier"]),
        "near_collision_rows_projected_from_sample": projections["near_collision_rows"],
        "stuck_rows_projected_from_sample": projections["stuck_rows"],
        "recommended_action": (
            "Mine and index the existing synchronized corpus. Do not collect new rollouts "
            "before a task-aligned branch-choice/recovery baseline is trained and gated."
        ),
    }
    report = {
        "schema": "task_aligned_data_readiness_v0",
        "rollout_root": str(args.rollout_root),
        "render_root": str(args.render_root),
        "exact": exact,
        "split_scenes": dict(split_scenes),
        "family_scenes": dict(family_scenes),
        "graph_type_rows": dict(graph_type_rows),
        "graph_type_nodes": dict(graph_type_nodes),
        "command_sources": dict(command_sources),
        "source_scene_counts": dict(source_scene_counts),
        "primitives": dict(primitives),
        "sampled_label_audit": sampled,
        "sampled_message_audit": message_sample,
        "sampled_message_command_sources": dict(message_command_sources),
        "sampled_message_sources_with_route_target": dict(
            message_sources_with_route_target
        ),
        "sampled_message_sources_with_next_waypoint": dict(
            message_sources_with_next_waypoint
        ),
        "sampled_by_split_family": {
            key: dict(value) for key, value in sampled_by_split_family.items()
        },
        "sample_projections": projections,
        "schema_gap_scene_ids": schema_gaps,
        "readiness": readiness,
        "mining_contract": {
            "inputs": [
                "rendered RGB frame",
                "short RGB/action history",
                "requested and executed command blocks",
            ],
            "privileged_training_labels": [
                "cell_id and local_graph_type",
                "route_target_id and next_waypoint_id",
                "landmark visibility/bearing/range/BFS distance",
                "clearance, traversability_forward, and stuck_label",
                "integrated body motion",
                "command source, clipping, and safety override",
            ],
            "first_dataset": (
                "Mine branch-point and recovery decision rows from existing train/val "
                "splits; preserve test_id/test_hard as held-out evaluation."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"exact": exact, "readiness": readiness}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
