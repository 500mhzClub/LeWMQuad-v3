#!/usr/bin/env python3
"""Select render frames for Go2 current-view-matched causal memory pairs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_plan", type=Path)
    parser.add_argument("causal_pair_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument(
        "--object-id",
        action="append",
        default=[],
        help="Restrict selection to one or more landmark object ids.",
    )
    parser.add_argument("--max-groups", type=int, default=16)
    parser.add_argument("--max-examples-per-bucket", type=int, default=2)
    parser.add_argument("--context-steps", type=int, default=0)
    parser.add_argument(
        "--require-seen-future-claim",
        action="store_true",
        help="Only select groups/examples where seen-before rows have future claims.",
    )
    parser.add_argument(
        "--require-seen-hidden-future-claim",
        action="store_true",
        help="Only select groups/examples where seen-before rows have hidden future claims.",
    )
    args = parser.parse_args()

    plan_path = args.render_plan.resolve()
    report_path = args.causal_pair_report.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frames_path = Path(plan["frames_jsonl"])
    if not frames_path.is_absolute():
        frames_path = plan_path.parent / frames_path
    frames = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame_index = _index_frames(frames)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected_events = _selected_events(
        report,
        source_index=int(args.source_index),
        object_ids={str(item) for item in args.object_id},
        max_groups=int(args.max_groups),
        max_examples_per_bucket=int(args.max_examples_per_bucket),
        context_steps=int(args.context_steps),
        require_seen_future_claim=bool(args.require_seen_future_claim),
        require_seen_hidden_future_claim=bool(args.require_seen_hidden_future_claim),
    )

    selected = []
    missing = []
    for key, events in sorted(selected_events.items()):
        frame = frame_index.get(key)
        if frame is None:
            missing.append(
                {
                    "env_idx": key[0],
                    "episode_id": key[1],
                    "episode_step": key[2],
                    "events": events,
                }
            )
            continue
        frame = dict(frame)
        frame["go2_causal_memory_pair_selection"] = sorted(
            events,
            key=lambda item: (
                str(item["object_id"]),
                str(item["pair_role"]),
                int(item["context_delta_steps"]),
            ),
        )
        selected.append(frame)

    frames_out = out_dir / "frames.jsonl"
    with frames_out.open("w", encoding="utf-8") as stream:
        for frame in selected:
            stream.write(json.dumps(frame, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    selected_plan = dict(plan)
    selected_plan["frames_jsonl"] = str(frames_out)
    selected_plan["frame_count"] = len(selected)
    selected_plan["output_dir"] = str(out_dir)
    selected_plan["selection_source_plan"] = str(plan_path)
    selected_plan["selection_causal_pair_report"] = str(report_path)
    selected_plan["selection_source_index"] = int(args.source_index)
    selected_plan["selection_object_ids"] = [str(item) for item in args.object_id]
    selected_plan["selection_max_groups"] = int(args.max_groups)
    selected_plan["selection_max_examples_per_bucket"] = int(args.max_examples_per_bucket)
    selected_plan["selection_context_steps"] = int(args.context_steps)
    selected_plan_path = out_dir / "render_replay_plan.json"
    selected_plan_path.write_text(
        json.dumps(selected_plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "schema": "lewm_go2_causal_memory_pair_render_selection_v0",
        "source_plan": str(plan_path),
        "causal_pair_report": str(report_path),
        "out_plan": str(selected_plan_path),
        "source_index": int(args.source_index),
        "object_ids": [str(item) for item in args.object_id],
        "max_groups": int(args.max_groups),
        "max_examples_per_bucket": int(args.max_examples_per_bucket),
        "context_steps": int(args.context_steps),
        "require_seen_future_claim": bool(args.require_seen_future_claim),
        "require_seen_hidden_future_claim": bool(args.require_seen_hidden_future_claim),
        "source_frame_count": len(frames),
        "requested_key_count": len(selected_events),
        "selected_frame_count": len(selected),
        "missing_key_count": len(missing),
        "missing": missing[:50],
    }
    summary_path = out_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(report_path, out_dir / "causal_pair_report.json")

    print(
        "go2_causal_memory_pair_render_selection:"
        f" selected={len(selected)}"
        f" requested={len(selected_events)}"
        f" missing={len(missing)}"
        f" plan={selected_plan_path}"
    )
    return 0


def _index_frames(frames: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    indexed = {}
    for frame in frames:
        episode = frame.get("episode") or {}
        key = (
            int(frame.get("env_index", 0)),
            int(episode.get("episode_id", 0)),
            int(episode.get("episode_step", -1)),
        )
        indexed[key] = frame
    return indexed


def _selected_events(
    report: dict[str, Any],
    *,
    source_index: int,
    object_ids: set[str],
    max_groups: int,
    max_examples_per_bucket: int,
    context_steps: int,
    require_seen_future_claim: bool,
    require_seen_hidden_future_claim: bool,
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    group_count = 0
    radius = max(0, int(context_steps))
    groups = sorted(
        report.get("ambiguous_groups", ()),
        key=lambda group: (
            -int(group.get("seen_before_hidden_future_claim_count", 0)),
            -int(group.get("seen_before_future_claim_count", 0)),
            str(group.get("object_id", "")),
            int(group.get("cell_id", -1)),
            int(group.get("yaw_bin", -1)),
        ),
    )
    for group in groups:
        object_id = str(group.get("object_id", ""))
        if object_ids and object_id not in object_ids:
            continue
        if require_seen_hidden_future_claim and int(
            group.get("seen_before_hidden_future_claim_count", 0)
        ) <= 0:
            continue
        if require_seen_future_claim and int(
            group.get("seen_before_future_claim_count", 0)
        ) <= 0:
            continue
        group_examples = _examples_for_source(group, source_index=source_index)
        if not group_examples["seen_before"] or not group_examples["unseen_before"]:
            continue
        if require_seen_hidden_future_claim:
            group_examples["seen_before"] = [
                example
                for example in group_examples["seen_before"]
                if example.get("future_claim_step") is not None
                and not bool(example.get("future_claim_visible", False))
            ]
        elif require_seen_future_claim:
            group_examples["seen_before"] = [
                example
                for example in group_examples["seen_before"]
                if example.get("future_claim_step") is not None
            ]
        if not group_examples["seen_before"] or not group_examples["unseen_before"]:
            continue
        group_count += 1
        if group_count > max(0, int(max_groups)):
            break
        for bucket, examples in group_examples.items():
            for example in examples[: max(1, int(max_examples_per_bucket))]:
                _add_example_events(
                    events_by_key,
                    group=group,
                    example=example,
                    seen_before=(bucket == "seen_before"),
                    context_steps=radius,
                )
    return events_by_key


def _examples_for_source(
    group: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "seen_before": [
            example
            for example in group.get("seen_before_examples", ())
            if int(example.get("source_index", 0)) == int(source_index)
        ],
        "unseen_before": [
            example
            for example in group.get("unseen_before_examples", ())
            if int(example.get("source_index", 0)) == int(source_index)
        ],
    }


def _add_example_events(
    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]],
    *,
    group: dict[str, Any],
    example: dict[str, Any],
    seen_before: bool,
    context_steps: int,
) -> None:
    object_id = str(group.get("object_id", example.get("object_id", "")))
    current_step = int(example.get("episode_step", 0))
    _add_step_events(
        events_by_key,
        example=example,
        object_id=object_id,
        event_step=current_step,
        pair_role="current_seen_before" if seen_before else "current_unseen_before",
        seen_before=seen_before,
        context_steps=context_steps,
    )
    if seen_before and example.get("first_visible_step") is not None:
        _add_step_events(
            events_by_key,
            example=example,
            object_id=object_id,
            event_step=int(example["first_visible_step"]),
            pair_role="first_visible_evidence",
            seen_before=True,
            context_steps=context_steps,
        )


def _add_step_events(
    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]],
    *,
    example: dict[str, Any],
    object_id: str,
    event_step: int,
    pair_role: str,
    seen_before: bool,
    context_steps: int,
) -> None:
    for delta in range(-context_steps, context_steps + 1):
        step = int(event_step) + delta
        if step < 0:
            continue
        key = (
            int(example.get("env_idx", 0)),
            int(example.get("episode_id", 0)),
            step,
        )
        events_by_key.setdefault(key, []).append(
            {
                "object_id": object_id,
                "pair_role": pair_role,
                "seen_before": bool(seen_before),
                "event_step": int(event_step),
                "context_delta_steps": int(delta),
                "future_claim_step": example.get("future_claim_step"),
                "future_claim_visible": example.get("future_claim_visible"),
                "future_claim_bfs_distance": example.get("future_claim_bfs_distance"),
                "future_claim_range_m": example.get("future_claim_range_m"),
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
