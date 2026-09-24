#!/usr/bin/env python3
"""Relabel student learned-local rows with nearest-teacher-pose primitives.

Generic DAgger relabeler: for each student dataset row (optionally filtered to
one target color), find the nearest teacher log row of the same target color by
(x, y, yaw) distance and adopt the teacher's requested primitive as the label.
Rows with no teacher pose within the distance threshold are dropped.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def _teacher_rows(result_path: Path, target_color: str | None) -> list[tuple[float, float, float, str]]:
    payload = json.loads(result_path.read_text())
    rows = []
    for row in payload.get("log", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("state", "")).upper() == "CLAIM":
            continue
        if target_color and str(row.get("target_color", "")).lower() != target_color:
            continue
        post_xy = row.get("post_xy")
        yaw = row.get("post_yaw")
        primitive = row.get("requested_primitive") or row.get("primitive")
        if post_xy is None or yaw is None or primitive is None:
            continue
        rows.append((float(post_xy[0]), float(post_xy[1]), float(yaw), str(primitive)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--teacher-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-color", default=None, help="Filter both sides to this color.")
    parser.add_argument("--max-xy-m", type=float, default=0.6)
    parser.add_argument("--yaw-weight-m-per-rad", type=float, default=0.4)
    args = parser.parse_args()

    color = None if args.target_color is None else str(args.target_color).lower()
    teacher = _teacher_rows(args.teacher_result, color)
    if not teacher:
        raise SystemExit("no teacher rows")
    data = np.load(args.dataset, allow_pickle=True)
    vocab = [str(p) for p in data["primitive_vocab"]]
    vocab_index = {p: i for i, p in enumerate(vocab)}
    meta = [json.loads(str(m)) for m in data["meta_json"]]

    kept, labels_out, meta_out = [], [], []
    label_counts: Counter = Counter()
    dropped = Counter()
    for idx, m in enumerate(meta):
        if color and str(m.get("target_color", "")).lower() != color:
            dropped["color"] += 1
            continue
        pose_xy = m.get("pose_xy")
        yaw = m.get("yaw_rad")
        if pose_xy is None or yaw is None:
            dropped["pose_missing"] += 1
            continue
        x, y = float(pose_xy[0]), float(pose_xy[1])
        best = None
        for tx, ty, tyaw, tprim in teacher:
            d = math.hypot(x - tx, y - ty) + float(args.yaw_weight_m_per_rad) * abs(
                math.atan2(math.sin(float(yaw) - tyaw), math.cos(float(yaw) - tyaw))
            )
            if best is None or d < best[0]:
                best = (d, tprim)
        if best is None or best[0] > float(args.max_xy_m):
            dropped["too_far"] += 1
            continue
        prim = best[1]
        if prim not in vocab_index:
            dropped["label_vocab"] += 1
            continue
        kept.append(idx)
        labels_out.append(vocab_index[prim])
        label_counts[prim] += 1
        meta_out.append(json.dumps(dict(m, relabel_source="teacher_nn_pose"), sort_keys=True))

    if not kept:
        raise SystemExit("no rows kept")
    np.savez_compressed(
        args.output,
        schema=data["schema"],
        features=data["features"][kept],
        labels=np.asarray(labels_out, dtype=np.int64),
        primitive_vocab=data["primitive_vocab"],
        meta_json=np.asarray(meta_out),
        result_json=np.asarray([str(x) for x in data.get("result_json", [""])]),
    )
    print(json.dumps({
        "kept": len(kept),
        "dropped": dict(dropped),
        "labels": dict(label_counts),
        "teacher_rows": len(teacher),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
