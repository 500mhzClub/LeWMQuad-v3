#!/usr/bin/env python3
"""Select render frames around Go2 hidden-target memory events."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


EVENT_STEP_FIELDS = (
    "first_visible_step",
    "memory_activation_step",
    "first_memory_on_claim_step",
    "first_visible_only_claim_step",
    "first_hidden_claim_step",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_plan", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=1,
        help="Also include +/- this many episode steps around each selected event.",
    )
    args = parser.parse_args()

    plan_path = args.render_plan.resolve()
    baseline_path = args.baseline_report.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    frames_path = Path(plan["frames_jsonl"])
    if not frames_path.is_absolute():
        frames_path = plan_path.parent / frames_path
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame_index = _index_frames(frames)
    selected_events = _selected_events(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        context_steps=int(args.context_steps),
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
                }
            )
            continue
        frame = dict(frame)
        frame["go2_hidden_target_memory_selection"] = sorted(
            events,
            key=lambda item: (
                str(item["object_id"]),
                str(item["event_field"]),
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
    selected_plan["selection_baseline_report"] = str(baseline_path)
    selected_plan["selection_context_steps"] = int(args.context_steps)
    selected_plan_path = out_dir / "render_replay_plan.json"
    selected_plan_path.write_text(
        json.dumps(selected_plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "schema": "lewm_go2_hidden_target_memory_render_selection_v0",
        "source_plan": str(plan_path),
        "baseline_report": str(baseline_path),
        "out_plan": str(selected_plan_path),
        "context_steps": int(args.context_steps),
        "source_frame_count": len(frames),
        "requested_key_count": len(selected_events),
        "selected_frame_count": len(selected),
        "missing_key_count": len(missing),
        "missing": missing[:50],
    }
    summary_path = out_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Keep a copy next to the selected plan so later provenance does not depend
    # on /tmp report cleanup.
    shutil.copy2(baseline_path, out_dir / "baseline_report.json")

    print(
        "go2_hidden_target_render_selection:"
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
    context_steps: int,
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    events_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    radius = max(0, int(context_steps))
    for episode in report.get("episodes", ()):
        env_idx = int(episode.get("env_idx", 0))
        episode_id = int(episode.get("episode_id", 0))
        for landmark in episode.get("landmarks", ()):
            if not bool(landmark.get("eligible", False)):
                continue
            object_id = str(landmark.get("object_id", ""))
            for field in EVENT_STEP_FIELDS:
                step = landmark.get(field)
                if step is None:
                    continue
                for delta in range(-radius, radius + 1):
                    selected_step = int(step) + delta
                    if selected_step >= 0:
                        key = (env_idx, episode_id, selected_step)
                        events_by_key.setdefault(key, []).append(
                            {
                                "object_id": object_id,
                                "event_field": field,
                                "event_step": int(step),
                                "context_delta_steps": int(delta),
                                "memory_on_success": bool(
                                    landmark.get("memory_on_success", False)
                                ),
                                "visible_only_success": bool(
                                    landmark.get("visible_only_success", False)
                                ),
                                "hidden_claim_success": bool(
                                    landmark.get("hidden_claim_success", False)
                                ),
                                "shuffled_memory_success": bool(
                                    landmark.get("shuffled_memory_success", False)
                                ),
                            }
                        )
    return events_by_key


if __name__ == "__main__":
    raise SystemExit(main())
