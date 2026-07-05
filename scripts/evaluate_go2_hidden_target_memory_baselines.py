#!/usr/bin/env python3
"""Evaluate simple Go2 hidden-target memory baselines from derived labels.

This is an offline evaluator for the first Go2 memory contract. It uses
privileged derived labels as an oracle observation stream so we can validate the
episode/evaluation shape before wiring a learned RGB memory controller.

The score is intentionally bounded: it asks whether a landmark was seen, became
hidden for long enough to require memory, and was later approached. It then
compares that memory-enabled oracle against direct-current-visibility and
shuffled-history controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_go2_hidden_target_memory_contract import (
    LandmarkTrace,
    _hidden_run_after_seen,
    _load_traces,
    _resolve_label_paths,
)


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
    parser.add_argument("--min-hidden-steps", type=int, default=5)
    parser.add_argument("--claim-bfs-distance", type=int, default=1)
    parser.add_argument("--claim-range-m", type=float, default=1.0)
    parser.add_argument(
        "--shuffle-offset",
        type=int,
        default=1,
        help="Deterministic within-episode landmark rotation for shuffled-memory control.",
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
        shuffle_offset=int(args.shuffle_offset),
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
        "hidden_target_baselines:"
        f" eligible_episodes={summary['eligible_episode_count']}"
        f" memory_on={summary['memory_on_success_episode_count']}"
        f" visible_only={summary['visible_only_success_episode_count']}"
        f" hidden_claim={summary['hidden_claim_success_episode_count']}"
        f" shuffled={summary['shuffled_memory_success_episode_count']}"
        f" no_memory={summary['no_memory_success_episode_count']}"
    )
    return 0


def _build_report(
    episodes: dict[tuple[str, int, int], dict[str, LandmarkTrace]],
    *,
    label_paths: list[Path],
    min_hidden_steps: int,
    claim_bfs_distance: int,
    claim_range_m: float,
    shuffle_offset: int,
) -> dict[str, Any]:
    episode_reports: list[dict[str, Any]] = []
    eligible_episode_keys: set[tuple[str, int, int]] = set()
    memory_on_episode_keys: set[tuple[str, int, int]] = set()
    visible_only_episode_keys: set[tuple[str, int, int]] = set()
    hidden_claim_episode_keys: set[tuple[str, int, int]] = set()
    shuffled_episode_keys: set[tuple[str, int, int]] = set()

    pair_counts = {
        "eligible": 0,
        "memory_on_success": 0,
        "visible_only_success": 0,
        "hidden_claim_success": 0,
        "shuffled_memory_success": 0,
        "no_memory_success": 0,
    }

    for key, traces in sorted(episodes.items()):
        scene_id, env_idx, episode_id = key
        object_ids = sorted(traces)
        landmark_reports: list[dict[str, Any]] = []
        for object_index, object_id in enumerate(object_ids):
            trace = traces[object_id]
            target = _evaluate_target(
                trace,
                min_hidden_steps=min_hidden_steps,
                claim_bfs_distance=claim_bfs_distance,
                claim_range_m=claim_range_m,
            )
            if not target["eligible"]:
                landmark_reports.append({"object_id": object_id, **target})
                continue

            shuffled_object_id = _shuffled_object_id(
                object_ids,
                object_index=object_index,
                offset=shuffle_offset,
            )
            shuffled = _evaluate_shuffled_target(
                traces[shuffled_object_id],
                activation_step=int(target["memory_activation_step"]),
                claim_bfs_distance=claim_bfs_distance,
                claim_range_m=claim_range_m,
            )
            merged = {
                "object_id": object_id,
                "shuffled_object_id": shuffled_object_id,
                **target,
                "shuffled_memory_success": shuffled["success"],
                "shuffled_first_claim_step": shuffled["first_claim_step"],
                "no_memory_success": False,
            }
            landmark_reports.append(merged)

            pair_counts["eligible"] += 1
            eligible_episode_keys.add(key)
            if merged["memory_on_success"]:
                pair_counts["memory_on_success"] += 1
                memory_on_episode_keys.add(key)
            if merged["visible_only_success"]:
                pair_counts["visible_only_success"] += 1
                visible_only_episode_keys.add(key)
            if merged["hidden_claim_success"]:
                pair_counts["hidden_claim_success"] += 1
                hidden_claim_episode_keys.add(key)
            if merged["shuffled_memory_success"]:
                pair_counts["shuffled_memory_success"] += 1
                shuffled_episode_keys.add(key)

        episode_reports.append(
            {
                "scene_id": scene_id,
                "env_idx": env_idx,
                "episode_id": episode_id,
                "eligible": any(item.get("eligible", False) for item in landmark_reports),
                "memory_on_success": any(
                    item.get("memory_on_success", False) for item in landmark_reports
                ),
                "visible_only_success": any(
                    item.get("visible_only_success", False) for item in landmark_reports
                ),
                "hidden_claim_success": any(
                    item.get("hidden_claim_success", False) for item in landmark_reports
                ),
                "shuffled_memory_success": any(
                    item.get("shuffled_memory_success", False) for item in landmark_reports
                ),
                "no_memory_success": False,
                "landmarks": landmark_reports,
            }
        )

    eligible_pairs = max(1, pair_counts["eligible"])
    eligible_episodes = max(1, len(eligible_episode_keys))
    summary = {
        "episode_count": len(episodes),
        "eligible_episode_count": len(eligible_episode_keys),
        "eligible_landmark_pair_count": pair_counts["eligible"],
        "memory_on_success_pair_count": pair_counts["memory_on_success"],
        "visible_only_success_pair_count": pair_counts["visible_only_success"],
        "hidden_claim_success_pair_count": pair_counts["hidden_claim_success"],
        "shuffled_memory_success_pair_count": pair_counts["shuffled_memory_success"],
        "no_memory_success_pair_count": pair_counts["no_memory_success"],
        "memory_on_success_episode_count": len(memory_on_episode_keys),
        "visible_only_success_episode_count": len(visible_only_episode_keys),
        "hidden_claim_success_episode_count": len(hidden_claim_episode_keys),
        "shuffled_memory_success_episode_count": len(shuffled_episode_keys),
        "no_memory_success_episode_count": 0,
        "memory_on_success_pair_rate": pair_counts["memory_on_success"] / eligible_pairs,
        "visible_only_success_pair_rate": pair_counts["visible_only_success"] / eligible_pairs,
        "hidden_claim_success_pair_rate": pair_counts["hidden_claim_success"] / eligible_pairs,
        "shuffled_memory_success_pair_rate": (
            pair_counts["shuffled_memory_success"] / eligible_pairs
        ),
        "no_memory_success_pair_rate": 0.0,
        "memory_on_success_episode_rate": len(memory_on_episode_keys) / eligible_episodes,
        "visible_only_success_episode_rate": len(visible_only_episode_keys) / eligible_episodes,
        "hidden_claim_success_episode_rate": len(hidden_claim_episode_keys) / eligible_episodes,
        "shuffled_memory_success_episode_rate": len(shuffled_episode_keys) / eligible_episodes,
        "no_memory_success_episode_rate": 0.0,
    }

    return {
        "schema": "lewm_go2_hidden_target_memory_baselines_v0",
        "inputs": [str(path) for path in label_paths],
        "config": {
            "min_hidden_steps": int(min_hidden_steps),
            "claim_bfs_distance": int(claim_bfs_distance),
            "claim_range_m": float(claim_range_m),
            "shuffle_offset": int(shuffle_offset),
        },
        "summary": summary,
        "episodes": episode_reports,
        "notes": [
            "This is an oracle-label evaluator, not a learned RGB controller.",
            "visible_only_success can reacquire a target after the hidden interval; "
            "it does not prove memory was unnecessary for choosing where to return.",
            "hidden_claim_success is the stricter subset where the close-approach "
            "step itself is not currently visible.",
        ],
    }


def _evaluate_target(
    trace: LandmarkTrace,
    *,
    min_hidden_steps: int,
    claim_bfs_distance: int,
    claim_range_m: float,
) -> dict[str, Any]:
    steps, visible, ranges, bfs = _ordered(trace)
    first_visible_idx = next((idx for idx, flag in enumerate(visible) if flag), None)
    if first_visible_idx is None:
        return _empty_target_result(steps)

    activation_idx, max_hidden_run = _hidden_run_after_seen(
        visible,
        first_visible_idx=first_visible_idx,
        min_hidden_steps=min_hidden_steps,
    )
    if activation_idx is None:
        return {
            **_empty_target_result(steps),
            "ever_visible": True,
            "first_visible_step": steps[first_visible_idx],
            "max_hidden_run_after_seen": int(max_hidden_run),
        }

    first_claim_idx = _first_claim_idx(
        bfs,
        ranges,
        start_idx=activation_idx,
        claim_bfs_distance=claim_bfs_distance,
        claim_range_m=claim_range_m,
        require_visible=None,
        visible=visible,
    )
    first_visible_claim_idx = _first_claim_idx(
        bfs,
        ranges,
        start_idx=activation_idx,
        claim_bfs_distance=claim_bfs_distance,
        claim_range_m=claim_range_m,
        require_visible=True,
        visible=visible,
    )
    first_hidden_claim_idx = _first_claim_idx(
        bfs,
        ranges,
        start_idx=activation_idx,
        claim_bfs_distance=claim_bfs_distance,
        claim_range_m=claim_range_m,
        require_visible=False,
        visible=visible,
    )
    return {
        "step_count": len(steps),
        "eligible": True,
        "ever_visible": True,
        "first_visible_step": steps[first_visible_idx],
        "memory_activation_step": steps[activation_idx],
        "max_hidden_run_after_seen": int(max_hidden_run),
        "memory_on_success": first_claim_idx is not None,
        "visible_only_success": first_visible_claim_idx is not None,
        "hidden_claim_success": first_hidden_claim_idx is not None,
        "first_memory_on_claim_step": None if first_claim_idx is None else steps[first_claim_idx],
        "first_visible_only_claim_step": (
            None if first_visible_claim_idx is None else steps[first_visible_claim_idx]
        ),
        "first_hidden_claim_step": (
            None if first_hidden_claim_idx is None else steps[first_hidden_claim_idx]
        ),
        "first_memory_on_claim_visible": (
            None if first_claim_idx is None else bool(visible[first_claim_idx])
        ),
        "min_bfs_after_activation": _min_optional_int(bfs[activation_idx:]),
        "min_range_after_activation_m": min(ranges[activation_idx:]),
    }


def _evaluate_shuffled_target(
    trace: LandmarkTrace,
    *,
    activation_step: int,
    claim_bfs_distance: int,
    claim_range_m: float,
) -> dict[str, Any]:
    steps, visible, ranges, bfs = _ordered(trace)
    start_idx = next(
        (idx for idx, step in enumerate(steps) if step >= activation_step),
        len(steps),
    )
    first_claim_idx = _first_claim_idx(
        bfs,
        ranges,
        start_idx=start_idx,
        claim_bfs_distance=claim_bfs_distance,
        claim_range_m=claim_range_m,
        require_visible=None,
        visible=visible,
    )
    return {
        "success": first_claim_idx is not None,
        "first_claim_step": None if first_claim_idx is None else steps[first_claim_idx],
    }


def _empty_target_result(steps: list[int]) -> dict[str, Any]:
    return {
        "step_count": len(steps),
        "eligible": False,
        "ever_visible": False,
        "first_visible_step": None,
        "memory_activation_step": None,
        "max_hidden_run_after_seen": 0,
        "memory_on_success": False,
        "visible_only_success": False,
        "hidden_claim_success": False,
        "first_memory_on_claim_step": None,
        "first_visible_only_claim_step": None,
        "first_hidden_claim_step": None,
        "first_memory_on_claim_visible": None,
        "min_bfs_after_activation": None,
        "min_range_after_activation_m": None,
    }


def _ordered(trace: LandmarkTrace) -> tuple[list[int], list[bool], list[float], list[int | None]]:
    order = sorted(range(len(trace.steps)), key=lambda idx: trace.steps[idx])
    return (
        [trace.steps[idx] for idx in order],
        [trace.visible[idx] for idx in order],
        [trace.ranges_m[idx] for idx in order],
        [trace.bfs_distances[idx] for idx in order],
    )


def _first_claim_idx(
    bfs: list[int | None],
    ranges: list[float],
    *,
    start_idx: int,
    claim_bfs_distance: int,
    claim_range_m: float,
    require_visible: bool | None,
    visible: list[bool],
) -> int | None:
    for idx in range(start_idx, len(ranges)):
        if require_visible is not None and bool(visible[idx]) is not require_visible:
            continue
        if _is_claim_step(
            bfs[idx],
            ranges[idx],
            claim_bfs_distance=claim_bfs_distance,
            claim_range_m=claim_range_m,
        ):
            return idx
    return None


def _is_claim_step(
    bfs_distance: int | None,
    range_m: float,
    *,
    claim_bfs_distance: int,
    claim_range_m: float,
) -> bool:
    return bool(
        (bfs_distance is not None and int(bfs_distance) <= int(claim_bfs_distance))
        or float(range_m) <= float(claim_range_m)
    )


def _min_optional_int(values: list[int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return min(present) if present else None


def _shuffled_object_id(
    object_ids: list[str],
    *,
    object_index: int,
    offset: int,
) -> str:
    if len(object_ids) <= 1:
        return object_ids[object_index]
    shift = int(offset) % len(object_ids)
    if shift == 0:
        shift = 1
    return object_ids[(object_index + shift) % len(object_ids)]


if __name__ == "__main__":
    raise SystemExit(main())

