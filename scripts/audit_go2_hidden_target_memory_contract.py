#!/usr/bin/env python3
"""Audit derived Go2 labels for hidden-target memory episodes.

The Go2 memory milestone needs episodes where a target landmark is first
observed, then becomes hidden long enough that direct visual servoing is not
sufficient, and is later approached again. This script does not score a learned
controller; it audits whether a derived-label rollout contains the episode
structure required for that later evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EpisodeKey = tuple[str, int, int]


@dataclass
class LandmarkTrace:
    object_id: str
    steps: list[int]
    visible: list[bool]
    ranges_m: list[float]
    bfs_distances: list[int | None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels",
        nargs="+",
        type=Path,
        help=(
            "Derived labels.jsonl files, derived_labels directories, or rollout "
            "directories containing derived_labels/labels.jsonl."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--min-hidden-steps",
        type=int,
        default=5,
        help="Minimum contiguous post-sighting invisible steps required.",
    )
    parser.add_argument(
        "--claim-bfs-distance",
        type=int,
        default=1,
        help="BFS-hop distance treated as a return/claim opportunity.",
    )
    parser.add_argument(
        "--claim-range-m",
        type=float,
        default=1.0,
        help="Metric range treated as a return/claim opportunity.",
    )
    args = parser.parse_args()

    label_paths = list(_resolve_label_paths(args.labels))
    if not label_paths:
        raise SystemExit("no labels.jsonl inputs found")

    episodes = _load_traces(label_paths)
    report = _build_report(
        episodes,
        label_paths=label_paths,
        min_hidden_steps=int(args.min_hidden_steps),
        claim_bfs_distance=int(args.claim_bfs_distance),
        claim_range_m=float(args.claim_range_m),
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"report={args.out}")
    else:
        print(text)

    summary = report["summary"]
    print(
        "hidden_target_audit:"
        f" episodes={summary['episode_count']}"
        f" seen_pairs={summary['seen_landmark_pair_count']}"
        f" memory_candidate_pairs={summary['memory_candidate_pair_count']}"
        f" return_candidate_pairs={summary['return_candidate_pair_count']}"
        f" memory_candidate_episodes={summary['memory_candidate_episode_count']}"
        f" return_candidate_episodes={summary['return_candidate_episode_count']}"
    )
    return 0


def _resolve_label_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = path.resolve()
        if path.is_file():
            yield path
            continue
        candidates = (
            path / "labels.jsonl",
            path / "derived_labels" / "labels.jsonl",
        )
        for candidate in candidates:
            if candidate.is_file():
                yield candidate
                break


def _load_traces(paths: Iterable[Path]) -> dict[EpisodeKey, dict[str, LandmarkTrace]]:
    episodes: dict[EpisodeKey, dict[str, LandmarkTrace]] = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                scene_id = str(row.get("scene_id", ""))
                env_idx = int(row.get("env_idx", 0))
                episode_id = int(row.get("episode_id", 0))
                step = int(row.get("episode_step", line_no))
                key = (scene_id, env_idx, episode_id)
                for landmark in row.get("landmarks", ()):
                    object_id = str(landmark.get("object_id", ""))
                    if not object_id:
                        continue
                    trace = episodes[key].get(object_id)
                    if trace is None:
                        trace = LandmarkTrace(
                            object_id=object_id,
                            steps=[],
                            visible=[],
                            ranges_m=[],
                            bfs_distances=[],
                        )
                        episodes[key][object_id] = trace
                    trace.steps.append(step)
                    trace.visible.append(bool(landmark.get("visible", False)))
                    trace.ranges_m.append(float(landmark.get("range_m", float("inf"))))
                    bfs = landmark.get("bfs_distance_cells")
                    trace.bfs_distances.append(None if bfs is None else int(bfs))
    return episodes


def _build_report(
    episodes: dict[EpisodeKey, dict[str, LandmarkTrace]],
    *,
    label_paths: list[Path],
    min_hidden_steps: int,
    claim_bfs_distance: int,
    claim_range_m: float,
) -> dict[str, Any]:
    episode_reports: list[dict[str, Any]] = []
    seen_pairs = 0
    memory_pairs = 0
    return_pairs = 0
    memory_episode_keys: set[EpisodeKey] = set()
    return_episode_keys: set[EpisodeKey] = set()

    for key, traces in sorted(episodes.items()):
        scene_id, env_idx, episode_id = key
        landmark_reports = []
        for object_id, trace in sorted(traces.items()):
            result = _audit_trace(
                trace,
                min_hidden_steps=min_hidden_steps,
                claim_bfs_distance=claim_bfs_distance,
                claim_range_m=claim_range_m,
            )
            if result["ever_visible"]:
                seen_pairs += 1
            if result["memory_candidate"]:
                memory_pairs += 1
                memory_episode_keys.add(key)
            if result["return_candidate"]:
                return_pairs += 1
                return_episode_keys.add(key)
            landmark_reports.append({"object_id": object_id, **result})
        episode_reports.append(
            {
                "scene_id": scene_id,
                "env_idx": env_idx,
                "episode_id": episode_id,
                "landmarks": landmark_reports,
                "memory_candidate": any(item["memory_candidate"] for item in landmark_reports),
                "return_candidate": any(item["return_candidate"] for item in landmark_reports),
            }
        )

    return {
        "schema": "lewm_go2_hidden_target_memory_contract_audit_v0",
        "inputs": [str(path) for path in label_paths],
        "config": {
            "min_hidden_steps": int(min_hidden_steps),
            "claim_bfs_distance": int(claim_bfs_distance),
            "claim_range_m": float(claim_range_m),
        },
        "summary": {
            "episode_count": len(episodes),
            "episode_landmark_pair_count": sum(len(traces) for traces in episodes.values()),
            "seen_landmark_pair_count": seen_pairs,
            "memory_candidate_pair_count": memory_pairs,
            "return_candidate_pair_count": return_pairs,
            "memory_candidate_episode_count": len(memory_episode_keys),
            "return_candidate_episode_count": len(return_episode_keys),
        },
        "episodes": episode_reports,
    }


def _audit_trace(
    trace: LandmarkTrace,
    *,
    min_hidden_steps: int,
    claim_bfs_distance: int,
    claim_range_m: float,
) -> dict[str, Any]:
    order = sorted(range(len(trace.steps)), key=lambda idx: trace.steps[idx])
    steps = [trace.steps[idx] for idx in order]
    visible = [trace.visible[idx] for idx in order]
    ranges = [trace.ranges_m[idx] for idx in order]
    bfs = [trace.bfs_distances[idx] for idx in order]

    first_visible_idx = next((idx for idx, is_visible in enumerate(visible) if is_visible), None)
    if first_visible_idx is None:
        return {
            "step_count": len(steps),
            "ever_visible": False,
            "first_visible_step": None,
            "first_hidden_after_seen_step": None,
            "max_hidden_run_after_seen": 0,
            "min_bfs_after_hidden": None,
            "min_range_after_hidden_m": None,
            "memory_candidate": False,
            "return_candidate": False,
        }

    hidden_start_idx, max_hidden_run = _hidden_run_after_seen(
        visible,
        first_visible_idx=first_visible_idx,
        min_hidden_steps=min_hidden_steps,
    )
    memory_candidate = hidden_start_idx is not None
    min_bfs_after_hidden = None
    min_range_after_hidden = None
    return_candidate = False
    if hidden_start_idx is not None:
        post_bfs = [value for value in bfs[hidden_start_idx:] if value is not None]
        post_ranges = ranges[hidden_start_idx:]
        min_bfs_after_hidden = min(post_bfs) if post_bfs else None
        min_range_after_hidden = min(post_ranges) if post_ranges else None
        return_candidate = bool(
            (
                min_bfs_after_hidden is not None
                and min_bfs_after_hidden <= int(claim_bfs_distance)
            )
            or (
                min_range_after_hidden is not None
                and min_range_after_hidden <= float(claim_range_m)
            )
        )

    return {
        "step_count": len(steps),
        "ever_visible": True,
        "first_visible_step": steps[first_visible_idx],
        "first_hidden_after_seen_step": (
            None if hidden_start_idx is None else steps[hidden_start_idx]
        ),
        "max_hidden_run_after_seen": int(max_hidden_run),
        "min_bfs_after_hidden": min_bfs_after_hidden,
        "min_range_after_hidden_m": min_range_after_hidden,
        "memory_candidate": memory_candidate,
        "return_candidate": return_candidate,
    }


def _hidden_run_after_seen(
    visible: list[bool],
    *,
    first_visible_idx: int,
    min_hidden_steps: int,
) -> tuple[int | None, int]:
    run_start = None
    run_len = 0
    max_run = 0
    first_qualifying_start = None
    for idx in range(first_visible_idx + 1, len(visible)):
        if visible[idx]:
            run_start = None
            run_len = 0
            continue
        if run_start is None:
            run_start = idx
            run_len = 1
        else:
            run_len += 1
        max_run = max(max_run, run_len)
        if run_len >= min_hidden_steps and first_qualifying_start is None:
            first_qualifying_start = run_start
    return first_qualifying_start, max_run


if __name__ == "__main__":
    raise SystemExit(main())

