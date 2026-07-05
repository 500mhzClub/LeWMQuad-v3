#!/usr/bin/env python3
"""Expand a selected Go2 render plan with continuous context frames.

The selected hidden-claim plans contain only event-adjacent frames. For pure
latent memory training we need the recurrent state to see more of the visual
history. This script reads an existing selected plan, reopens its full source
plan, and emits a new selected plan with +/- N episode steps around each
selected event frame. Event annotations are preserved only on the originally
selected frames; added context frames remain ordinary history frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVENT_KEYS = (
    "go2_hidden_target_memory_selection",
    "go2_causal_memory_pair_selection",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("selected_plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-steps", type=int, default=24)
    args = parser.parse_args()

    selected_plan_path = args.selected_plan.resolve()
    selected_plan = json.loads(selected_plan_path.read_text(encoding="utf-8"))
    source_plan_path = Path(
        selected_plan.get("selection_source_plan") or selected_plan_path
    )
    if not source_plan_path.is_absolute():
        source_plan_path = selected_plan_path.parent / source_plan_path
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))

    selected_frames = _load_plan_frames(selected_plan_path, selected_plan)
    source_frames = _load_plan_frames(source_plan_path, source_plan)
    source_index = _index_by_env_episode_step(source_frames)
    selected_index = _index_by_env_episode_step(selected_frames)

    wanted: dict[tuple[int, int, int], dict[str, Any]] = {}
    radius = max(0, int(args.context_steps))
    for key in sorted(selected_index):
        env_idx, episode_id, episode_step = key
        for delta in range(-radius, radius + 1):
            context_key = (env_idx, episode_id, episode_step + delta)
            frame = source_index.get(context_key)
            if frame is None:
                continue
            item = dict(frame)
            for event_key in EVENT_KEYS:
                item.pop(event_key, None)
            wanted[context_key] = item
    for key, frame in selected_index.items():
        item = dict(wanted.get(key, frame))
        for event_key in EVENT_KEYS:
            if event_key in frame:
                item[event_key] = frame[event_key]
        wanted[key] = item

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out = out_dir / "frames.jsonl"
    expanded = [wanted[key] for key in sorted(wanted)]
    with frames_out.open("w", encoding="utf-8") as stream:
        for frame in expanded:
            stream.write(json.dumps(frame, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    out_plan = dict(source_plan)
    out_plan["frames_jsonl"] = str(frames_out)
    out_plan["frame_count"] = len(expanded)
    out_plan["output_dir"] = str(out_dir)
    out_plan["selection_source_plan"] = str(source_plan_path)
    out_plan["selection_seed_plan"] = str(selected_plan_path)
    out_plan["selection_context_steps"] = int(args.context_steps)
    out_plan_path = out_dir / "render_replay_plan.json"
    out_plan_path.write_text(json.dumps(out_plan, indent=2, sort_keys=True), encoding="utf-8")

    event_frame_count = sum(
        1
        for frame in expanded
        if any(frame.get(event_key) for event_key in EVENT_KEYS)
    )
    summary = {
        "schema": "lewm_go2_expanded_render_context_v0",
        "selected_plan": str(selected_plan_path),
        "source_plan": str(source_plan_path),
        "out_plan": str(out_plan_path),
        "context_steps": int(args.context_steps),
        "seed_frame_count": len(selected_frames),
        "expanded_frame_count": len(expanded),
        "event_frame_count": int(event_frame_count),
    }
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "go2_expand_selected_context:"
        f" seed={len(selected_frames)}"
        f" expanded={len(expanded)}"
        f" event_frames={event_frame_count}"
        f" plan={out_plan_path}"
    )
    return 0


def _load_plan_frames(plan_path: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    frames_path = Path(plan["frames_jsonl"])
    if not frames_path.is_absolute():
        frames_path = plan_path.parent / frames_path
    return [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_by_env_episode_step(
    frames: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
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


if __name__ == "__main__":
    raise SystemExit(main())
