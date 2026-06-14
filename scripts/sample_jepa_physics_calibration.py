#!/usr/bin/env python3
"""Select a deterministic stratified sample for JEPA physics calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _bucket(candidate: dict) -> str:
    unsafe = bool(candidate["enters_grid_unsafe"] or candidate["ends_grid_unsafe"])
    progress = candidate["target_progress_m"]
    positive_progress = progress is not None and float(progress) > 0.0
    if unsafe and positive_progress:
        return "unsafe_progress"
    if unsafe:
        return "unsafe_no_progress"
    if positive_progress:
        return "safe_progress"
    return "safe_no_progress"


def _stable_rank(row: dict, candidate: dict, seed: int) -> str:
    key = "|".join(
        (
            str(seed),
            str(row["scene_id"]),
            str(row["start_timestamp_ns"]),
            ",".join(candidate["primitive_sequence"]),
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    candidates_by_group: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    with args.input.open() as stream:
        for line in stream:
            row = json.loads(line)
            for candidate_index, candidate in enumerate(row["counterfactual_candidates"]):
                bucket = _bucket(candidate)
                sample = {
                    "schema": "jepa_physics_calibration_sample_v0",
                    "source_benchmark": str(args.input.resolve()),
                    "scene_id": str(row["scene_id"]),
                    "split": str(row["split"]),
                    "family": str(row["family"]),
                    "scene_manifest": str(row["scene_manifest"]),
                    "start_timestamp_ns": int(row["start_timestamp_ns"]),
                    "start_base_pose_world": row["start_base_pose_world"],
                    "start_base_rpy_rad": row["start_base_rpy_rad"],
                    "target_xy": row.get("counterfactual_target_xy"),
                    "candidate_index": candidate_index,
                    "candidate_bucket": bucket,
                    "kinematic_candidate": candidate,
                }
                rank = _stable_rank(row, candidate, args.seed)
                candidates_by_group[(str(row["family"]), bucket)].append((rank, sample))

    selected = []
    for group in sorted(candidates_by_group):
        values = sorted(candidates_by_group[group], key=lambda item: item[0])
        selected.extend(sample for _rank, sample in values[: args.per_family_bucket])
    selected.sort(
        key=lambda row: (
            row["scene_id"],
            row["family"],
            row["candidate_bucket"],
            row["start_timestamp_ns"],
            row["candidate_index"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in selected:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema": "jepa_physics_calibration_sample_summary_v0",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "per_family_bucket": args.per_family_bucket,
        "row_count": len(selected),
        "family_counts": dict(Counter(row["family"] for row in selected)),
        "bucket_counts": dict(Counter(row["candidate_bucket"] for row in selected)),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
