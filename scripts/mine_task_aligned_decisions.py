#!/usr/bin/env python3
"""Mine branch-choice and recovery decisions from existing synchronized rollouts."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from itertools import zip_longest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.actions import encode_active_block, encode_executed_command_block  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402
from probe_lewm_latent_aliasing import _find_manifest  # noqa: E402


def _balanced_scene_summaries(
    rollout_root: Path,
    *,
    split: str,
    family: str | None,
    limit: int,
    seed: int,
) -> list[Path]:
    by_family: dict[str, list[Path]] = defaultdict(list)
    for summary in rollout_root.rglob("labels/*/summary.json"):
        payload = json.loads(summary.read_text())
        scene = payload.get("scene", {})
        if str(scene.get("split")) != split:
            continue
        scene_family = str(scene.get("family"))
        if family is not None and scene_family != family:
            continue
        by_family[scene_family].append(summary)
    rng = random.Random(seed)
    for summaries in by_family.values():
        summaries.sort()
        rng.shuffle(summaries)
    ordered: list[Path] = []
    max_count = max((len(values) for values in by_family.values()), default=0)
    for index in range(max_count):
        for scene_family in sorted(by_family):
            if index < len(by_family[scene_family]):
                ordered.append(by_family[scene_family][index])
                if limit > 0 and len(ordered) >= limit:
                    return ordered
    return ordered


def _plan_for_summary(summary: Path) -> Path:
    scene_id = summary.parent.name
    matches = list(
        (summary.parents[2] / "plan").glob(f"*_{scene_id}/render_replay_plan.json")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one replay plan for {scene_id}, found {len(matches)}")
    return matches[0]


def _frame_path(render_root: Path, scene_id: str, frame: dict) -> Path:
    return (
        render_root
        / scene_id
        / "rgb"
        / f"frame_{int(frame['frame_index']):06d}_env_{int(frame['env_index']):02d}.png"
    )


def _messages_for_summary(summary: Path) -> Path:
    return summary.parents[2] / "raw" / summary.parent.name / "messages.jsonl"


def _load_executed_commands(messages_path: Path) -> dict[tuple[int, int], dict]:
    executed: dict[tuple[int, int], dict] = {}
    with messages_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("canonical_topic") != "/lewm/go2/executed_command_block":
                continue
            payload = record["payload"]
            key = (int(record["env_index"]), int(payload["sequence_id"]))
            if key in executed:
                raise RuntimeError(f"duplicate executed command {key} in {messages_path}")
            executed[key] = payload
    return executed


def _decision_types(start_label: dict, command: dict, near_collision_m: float) -> list[str]:
    result = []
    if (
        str(start_label.get("local_graph_type")) in {"t_junction", "crossroad"}
        and int(command.get("route_target_id", -1)) >= 0
    ):
        result.append("branch")
    if (
        str(command.get("command_source")) == "recovery"
        or bool(start_label.get("stuck_label", False))
        or float(start_label.get("clearance_m", 1e9)) < near_collision_m
    ):
        result.append("recovery")
    return result


def _make_row(
    *,
    graph: SceneGraph,
    scene_id: str,
    split: str,
    family: str,
    render_root: Path,
    start_frame: dict,
    start_label: dict,
    end_frame: dict,
    end_label: dict,
    command: dict,
    executed_command: dict,
    manifest_path: Path,
    history_frames: tuple[str, ...],
    near_collision_m: float,
) -> dict | None:
    decision_types = _decision_types(start_label, command, near_collision_m)
    if not decision_types:
        return None
    route_target = int(command.get("route_target_id", -1))
    start_cell = int(start_label["cell_id"])
    oracle_next = (
        graph.next_waypoint(
            start_cell,
            route_target,
            transit_blocked=graph.nav_blocked_cells,
        )
        if route_target >= 0 and route_target != start_cell
        else None
    )
    start_path = _frame_path(render_root, scene_id, start_frame)
    end_path = _frame_path(render_root, scene_id, end_frame)
    if not start_path.is_file() or not end_path.is_file():
        return None
    requested_active_block = encode_active_block(
        command["vx_body_mps"],
        command["vy_body_mps"],
        command["yaw_rate_radps"],
    )
    executed_active_block = encode_executed_command_block(executed_command)
    return {
        "schema": "task_aligned_decision_v0",
        "scene_id": scene_id,
        "split": split,
        "family": family,
        "scene_manifest": str(manifest_path),
        "env_index": int(start_frame["env_index"]),
        "episode_id": int(start_label["episode_id"]),
        "decision_types": decision_types,
        "start_frame": str(start_path),
        "end_frame": str(end_path),
        "history_frames": list(history_frames),
        "start_base_pose_world": start_frame["base_pose_world"],
        "start_base_rpy_rad": start_frame["base_rpy_rad"],
        "start_timestamp_ns": int(start_frame["timestamp_ns"]),
        "end_timestamp_ns": int(end_frame["timestamp_ns"]),
        "command_sequence_id": int(command["sequence_id"]),
        "command_source": str(command.get("command_source", "unknown")),
        "primitive_name": str(command.get("primitive_name", "unknown")),
        "command_dt_s": float(command["command_dt_s"]),
        "block_size": int(command["block_size"]),
        "requested_active_block": requested_active_block.tolist(),
        "executed_active_block": executed_active_block.tolist(),
        "execution_clipped": bool(executed_command.get("clipped", False)),
        "execution_safety_overridden": bool(
            executed_command.get("safety_overridden", False)
        ),
        "route_target_id": route_target,
        "logged_next_waypoint_id": int(command.get("next_waypoint_id", -1)),
        "oracle_next_cell_id": oracle_next,
        "start_cell_id": start_cell,
        "end_cell_id": int(end_label["cell_id"]),
        "start_local_graph_type": str(start_label.get("local_graph_type", "unknown")),
        "start_clearance_m": float(start_label.get("clearance_m", 0.0)),
        "end_clearance_m": float(end_label.get("clearance_m", 0.0)),
        "start_stuck": bool(start_label.get("stuck_label", False)),
        "end_stuck": bool(end_label.get("stuck_label", False)),
        "integrated_body_motion_block": end_label.get(
            "integrated_body_motion_block",
            [0.0, 0.0, 0.0],
        ),
    }


def _mine_scene(
    summary_path: Path,
    *,
    scene_corpus: Path,
    render_root: Path,
    near_collision_m: float,
    max_rows: int,
    history_frames: int,
) -> tuple[list[dict], dict]:
    summary = json.loads(summary_path.read_text())
    scene = summary["scene"]
    scene_id = str(scene["scene_id"])
    split = str(scene["split"])
    family = str(scene["family"])
    manifest_path = _find_manifest(scene_corpus, split, family, scene_id)
    if manifest_path is None:
        raise RuntimeError(f"manifest not found for {split}/{family}/{scene_id}")
    graph = SceneGraph(parse_scene_manifest_dict(json.loads(manifest_path.read_text())))
    plan = json.loads(_plan_for_summary(summary_path).read_text())
    frames_path = Path(plan["frames_jsonl"])
    labels_path = summary_path.parent / "labels.jsonl"
    messages_path = _messages_for_summary(summary_path)
    executed_commands = _load_executed_commands(messages_path)

    rows: list[dict] = []
    previous: dict[int, tuple[dict, dict]] = {}
    pending: dict[int, tuple[dict, dict, dict, tuple[str, ...]]] = {}
    recent_frames: dict[int, deque[str]] = defaultdict(
        lambda: deque(maxlen=max(1, history_frames))
    )
    previous_episode: dict[int, int] = {}
    target_frames: dict[int, str] = {}
    aligned = 0
    with frames_path.open() as frame_stream, labels_path.open() as label_stream:
        for frame_line, label_line in zip_longest(frame_stream, label_stream):
            if frame_line is None or label_line is None:
                raise RuntimeError(f"frame/label row-count mismatch in {scene_id}")
            frame = json.loads(frame_line)
            label = json.loads(label_line)
            env = int(frame["env_index"])
            if env != int(label["env_idx"]) or int(frame["timestamp_ns"]) != int(
                label["timestamp_ns"]
            ):
                raise RuntimeError(f"frame/label alignment mismatch in {scene_id}")
            aligned += 1
            episode = int(label["episode_id"])
            if env in previous_episode and previous_episode[env] != episode:
                previous.pop(env, None)
                pending.pop(env, None)
                recent_frames[env].clear()
            previous_episode[env] = episode
            target_frames.setdefault(
                int(label["cell_id"]),
                str(_frame_path(render_root, scene_id, frame)),
            )
            command = frame.get("command_context", {})
            prior = previous.get(env)
            if prior is not None:
                prior_frame, prior_label = prior
                prior_sequence = int(
                    prior_frame.get("command_context", {}).get("sequence_id", -1)
                )
                sequence = int(command.get("sequence_id", -1))
                if sequence != prior_sequence:
                    old_pending = pending.get(env)
                    if old_pending is not None:
                        start_frame, start_label, old_command, start_history = old_pending
                        command_key = (
                            int(start_frame["env_index"]),
                            int(old_command["sequence_id"]),
                        )
                        executed_command = executed_commands.get(command_key)
                        if executed_command is None:
                            raise RuntimeError(
                                f"missing executed command {command_key} in {scene_id}"
                            )
                        row = _make_row(
                            graph=graph,
                            scene_id=scene_id,
                            split=split,
                            family=family,
                            render_root=render_root,
                            start_frame=start_frame,
                            start_label=start_label,
                            end_frame=prior_frame,
                            end_label=prior_label,
                            command=old_command,
                            executed_command=executed_command,
                            manifest_path=manifest_path,
                            history_frames=start_history,
                            near_collision_m=near_collision_m,
                        )
                        if row is not None:
                            rows.append(row)
                    pending[env] = (
                        prior_frame,
                        prior_label,
                        command,
                        tuple(recent_frames[env]),
                    )
            previous[env] = (frame, label)
            recent_frames[env].append(str(_frame_path(render_root, scene_id, frame)))

    for row in rows:
        route_target = int(row["route_target_id"])
        oracle_next = row["oracle_next_cell_id"]
        route_frame = target_frames.get(route_target) if route_target >= 0 else None
        local_frame = (
            target_frames.get(int(oracle_next)) if oracle_next is not None else None
        )
        route_frame = (
            route_frame if route_frame is not None and Path(route_frame).is_file() else None
        )
        local_frame = (
            local_frame if local_frame is not None and Path(local_frame).is_file() else None
        )
        # v2 contract: separate the final-route-goal image from the local-subgoal
        # image, and target existence from image availability. score_task_aligned
        # targets the local next cell (oracle_next), so local_target_frame is the
        # matched goal image; route_target_frame is the harder routing-plus-control
        # input kept for an explicit control. target_present records that a scored
        # target exists regardless of whether its representative frame was rendered.
        row["route_target_frame"] = route_frame
        row["local_target_frame"] = local_frame
        row["target_frame"] = route_frame  # back-compat with v1 consumers
        row["target_present"] = bool(oracle_next is not None or route_target >= 0)
        row["route_target_image_present"] = route_frame is not None
        row["local_target_image_present"] = local_frame is not None
    if max_rows > 0 and len(rows) > max_rows:
        random.Random(scene_id).shuffle(rows)
        rows = rows[:max_rows]
    return rows, {
        "scene_id": scene_id,
        "aligned_frame_label_rows": aligned,
        "mined_rows": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=REPO_ROOT / ".generated/datagen_full/rollout",
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        default=REPO_ROOT / ".generated/datagen_full/render_textured_v03",
    )
    parser.add_argument(
        "--scene-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="all")
    parser.add_argument("--scene-limit", type=int, default=8)
    parser.add_argument("--max-rows-per-scene", type=int, default=512)
    parser.add_argument("--near-collision-m", type=float, default=0.30)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = _balanced_scene_summaries(
        args.rollout_root,
        split=args.split,
        family=None if args.family == "all" else args.family,
        limit=args.scene_limit,
        seed=args.seed,
    )
    if not summaries:
        raise SystemExit("no matching scene summaries")
    all_rows: list[dict] = []
    scenes = []
    for index, summary in enumerate(summaries, 1):
        rows, scene_summary = _mine_scene(
            summary,
            scene_corpus=args.scene_corpus,
            render_root=args.render_root,
            near_collision_m=args.near_collision_m,
            max_rows=args.max_rows_per_scene,
            history_frames=args.history_frames,
        )
        all_rows.extend(rows)
        scenes.append(scene_summary)
        print(
            f"[{index}/{len(summaries)}] {scene_summary['scene_id']} "
            f"rows={scene_summary['mined_rows']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    type_counts = Counter(
        decision_type
        for row in all_rows
        for decision_type in row["decision_types"]
    )
    source_counts = Counter(row["command_source"] for row in all_rows)
    primitive_counts = Counter(row["primitive_name"] for row in all_rows)
    state_keys = {
        (
            row["scene_id"],
            row["start_cell_id"],
            row["route_target_id"],
            tuple(row["decision_types"]),
        )
        for row in all_rows
    }
    summary = {
        "schema": "task_aligned_decision_index_summary_v0",
        "output": str(args.output),
        "split": args.split,
        "family": args.family,
        "history_frames": args.history_frames,
        "scene_count": len(scenes),
        "row_count": len(all_rows),
        "decision_type_counts": dict(type_counts),
        "command_source_counts": dict(source_counts),
        "primitive_counts": dict(primitive_counts),
        "unique_scene_cell_target_decision_states": len(state_keys),
        "execution_clipped_rows": sum(row["execution_clipped"] for row in all_rows),
        "execution_safety_overridden_rows": sum(
            row["execution_safety_overridden"] for row in all_rows
        ),
        "rows_with_target_frame": sum(row["target_frame"] is not None for row in all_rows),
        "rows_with_route_target_frame": sum(
            row["route_target_frame"] is not None for row in all_rows
        ),
        "rows_with_local_target_frame": sum(
            row["local_target_frame"] is not None for row in all_rows
        ),
        "rows_with_target_present": sum(row["target_present"] for row in all_rows),
        "rows_with_oracle_next_cell": sum(
            row["oracle_next_cell_id"] is not None for row in all_rows
        ),
        "scenes": scenes,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
