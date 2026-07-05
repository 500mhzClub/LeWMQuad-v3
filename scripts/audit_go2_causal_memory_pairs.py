#!/usr/bin/env python3
"""Audit Go2 labels for current-view-matched causal memory pairs.

The supervised probe can overclaim if current frame/trajectory cues predict
which landmark was probably seen. This audit looks for stricter rows: same
scene, cell, yaw bin, and currently hidden landmark, but different prior
visibility histories across episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--max-examples-per-group", type=int, default=4)
    parser.add_argument("--claim-bfs-distance", type=int, default=1)
    parser.add_argument("--claim-range-m", type=float, default=1.0)
    parser.add_argument(
        "--require-hidden-claim",
        action="store_true",
        help="Count future claims only when the landmark is hidden at claim time.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.labels)
    rows.sort(key=lambda row: _seq_step_key(row))
    rows_by_sequence: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_sequence[_seq_key(row)].append(row)
    seen_by_sequence: dict[tuple[int, str, int, int], set[str]] = defaultdict(set)
    first_seen_step_by_sequence: dict[tuple[int, str, int, int], dict[str, int]] = defaultdict(
        dict
    )
    grouped: dict[tuple[str, int, int, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"seen_before": [], "unseen_before": []}
    )

    for row in rows:
        seq_key = _seq_key(row)
        step = int(row.get("episode_step", 0))
        visible_ids = {str(item) for item in row.get("visible_landmark_ids", ())}
        if not visible_ids:
            visible_ids = {
                str(landmark.get("object_id", ""))
                for landmark in row.get("landmarks", ())
                if bool(landmark.get("visible", False))
            }

        for landmark in row.get("landmarks", ()):
            object_id = str(landmark.get("object_id", ""))
            if not object_id or object_id in visible_ids:
                continue
            if step < int(args.min_step):
                continue
            group_key = (
                str(row.get("scene_id", "")),
                int(row.get("cell_id", -1)),
                int(row.get("yaw_bin", -1)),
                object_id,
            )
            bucket = (
                "seen_before"
                if object_id in seen_by_sequence[seq_key]
                else "unseen_before"
            )
            grouped[group_key][bucket].append(
                _compact_row(
                    row,
                    object_id=object_id,
                    first_visible_step=first_seen_step_by_sequence[seq_key].get(
                        object_id
                    ),
                    future_claim=_future_claim(
                        rows_by_sequence[seq_key],
                        start_step=step,
                        object_id=object_id,
                        claim_bfs_distance=int(args.claim_bfs_distance),
                        claim_range_m=float(args.claim_range_m),
                        require_hidden_claim=bool(args.require_hidden_claim),
                    ),
                )
            )

        seen_by_sequence[seq_key].update(visible_ids)
        for object_id in visible_ids:
            first_seen_step_by_sequence[seq_key].setdefault(object_id, step)

    ambiguous_groups = []
    seen_rows = 0
    unseen_rows = 0
    for key, buckets in sorted(grouped.items()):
        seen_count = len(buckets["seen_before"])
        unseen_count = len(buckets["unseen_before"])
        seen_rows += seen_count
        unseen_rows += unseen_count
        if seen_count <= 0 or unseen_count <= 0:
            continue
        examples_per_bucket = max(1, int(args.max_examples_per_group))
        scene_id, cell_id, yaw_bin, object_id = key
        seen_future_claim_count = sum(
            1 for item in buckets["seen_before"] if item["future_claim_step"] is not None
        )
        unseen_future_claim_count = sum(
            1 for item in buckets["unseen_before"] if item["future_claim_step"] is not None
        )
        seen_hidden_future_claim_count = sum(
            1
            for item in buckets["seen_before"]
            if item["future_claim_step"] is not None
            and not bool(item["future_claim_visible"])
        )
        unseen_hidden_future_claim_count = sum(
            1
            for item in buckets["unseen_before"]
            if item["future_claim_step"] is not None
            and not bool(item["future_claim_visible"])
        )
        ambiguous_groups.append(
            {
                "scene_id": scene_id,
                "cell_id": cell_id,
                "yaw_bin": yaw_bin,
                "object_id": object_id,
                "seen_before_count": seen_count,
                "unseen_before_count": unseen_count,
                "seen_before_future_claim_count": seen_future_claim_count,
                "unseen_before_future_claim_count": unseen_future_claim_count,
                "seen_before_hidden_future_claim_count": seen_hidden_future_claim_count,
                "unseen_before_hidden_future_claim_count": (
                    unseen_hidden_future_claim_count
                ),
                "seen_before_examples": buckets["seen_before"][:examples_per_bucket],
                "unseen_before_examples": buckets["unseen_before"][:examples_per_bucket],
            }
        )

    report = {
        "schema": "lewm_go2_causal_memory_pair_audit_v0",
        "inputs": [str(path) for path in args.labels],
        "config": {
            "min_step": int(args.min_step),
            "max_examples_per_group": int(args.max_examples_per_group),
            "claim_bfs_distance": int(args.claim_bfs_distance),
            "claim_range_m": float(args.claim_range_m),
            "require_hidden_claim": bool(args.require_hidden_claim),
        },
        "summary": {
            "row_count": len(rows),
            "current_hidden_seen_before_row_count": seen_rows,
            "current_hidden_unseen_before_row_count": unseen_rows,
            "current_view_group_count": len(grouped),
            "ambiguous_current_view_group_count": len(ambiguous_groups),
            "ambiguous_seen_before_row_count": sum(
                int(group["seen_before_count"]) for group in ambiguous_groups
            ),
            "ambiguous_unseen_before_row_count": sum(
                int(group["unseen_before_count"]) for group in ambiguous_groups
            ),
            "ambiguous_seen_before_future_claim_row_count": sum(
                int(group["seen_before_future_claim_count"]) for group in ambiguous_groups
            ),
            "ambiguous_unseen_before_future_claim_row_count": sum(
                int(group["unseen_before_future_claim_count"]) for group in ambiguous_groups
            ),
            "ambiguous_seen_before_hidden_future_claim_row_count": sum(
                int(group["seen_before_hidden_future_claim_count"])
                for group in ambiguous_groups
            ),
            "ambiguous_unseen_before_hidden_future_claim_row_count": sum(
                int(group["unseen_before_hidden_future_claim_count"])
                for group in ambiguous_groups
            ),
        },
        "ambiguous_groups": ambiguous_groups,
        "interpretation": (
            "Rows in an ambiguous group share scene_id, cell_id, yaw_bin, and a "
            "currently hidden object_id, but differ in whether that object was "
            "visible earlier in the same episode. These are the best existing "
            "label candidates for a causal memory probe because current view is "
            "matched and prior history must resolve the target."
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"report={args.out}")
    else:
        print(text)

    summary = report["summary"]
    print(
        "go2_causal_memory_pairs:"
        f" rows={summary['row_count']}"
        f" hidden_seen={summary['current_hidden_seen_before_row_count']}"
        f" hidden_unseen={summary['current_hidden_unseen_before_row_count']}"
        f" ambiguous_groups={summary['ambiguous_current_view_group_count']}"
        f" ambiguous_seen={summary['ambiguous_seen_before_row_count']}"
        f" ambiguous_unseen={summary['ambiguous_unseen_before_row_count']}"
    )
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        path = path.resolve()
        if path.is_dir():
            path = path / "labels.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    row["_source_index"] = int(source_index)
                    row["_source_label_path"] = str(path)
                    rows.append(row)
    return rows


def _seq_key(row: dict[str, Any]) -> tuple[int, str, int, int]:
    return (
        int(row.get("_source_index", 0)),
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _seq_step_key(row: dict[str, Any]) -> tuple[int, str, int, int, int]:
    source_index, scene_id, env_idx, episode_id = _seq_key(row)
    return source_index, scene_id, env_idx, episode_id, int(row.get("episode_step", 0))


def _compact_row(
    row: dict[str, Any],
    *,
    object_id: str,
    first_visible_step: int | None,
    future_claim: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source_index": int(row.get("_source_index", 0)),
        "source_label_path": str(row.get("_source_label_path", "")),
        "env_idx": int(row.get("env_idx", 0)),
        "episode_id": int(row.get("episode_id", 0)),
        "episode_step": int(row.get("episode_step", 0)),
        "timestamp_ns": int(row.get("timestamp_ns", 0)),
        "object_id": object_id,
        "first_visible_step": first_visible_step,
        "future_claim_step": None if future_claim is None else int(future_claim["step"]),
        "future_claim_visible": None
        if future_claim is None
        else bool(future_claim["visible"]),
        "future_claim_bfs_distance": None
        if future_claim is None
        else future_claim["bfs_distance_cells"],
        "future_claim_range_m": None
        if future_claim is None
        else float(future_claim["range_m"]),
        "local_graph_type": str(row.get("local_graph_type", "")),
    }


def _future_claim(
    rows: list[dict[str, Any]],
    *,
    start_step: int,
    object_id: str,
    claim_bfs_distance: int,
    claim_range_m: float,
    require_hidden_claim: bool,
) -> dict[str, Any] | None:
    for row in rows:
        step = int(row.get("episode_step", 0))
        if step < int(start_step):
            continue
        landmark = _landmark_by_id(row).get(object_id)
        if landmark is None:
            continue
        visible = bool(landmark.get("visible", False))
        if require_hidden_claim and visible:
            continue
        bfs = landmark.get("bfs_distance_cells")
        range_m = float(landmark.get("range_m", float("inf")))
        if (bfs is not None and int(bfs) <= int(claim_bfs_distance)) or range_m <= float(
            claim_range_m
        ):
            return {
                "step": step,
                "visible": visible,
                "bfs_distance_cells": None if bfs is None else int(bfs),
                "range_m": range_m,
            }
    return None


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): dict(landmark)
        for landmark in row.get("landmarks", ())
    }


if __name__ == "__main__":
    raise SystemExit(main())
