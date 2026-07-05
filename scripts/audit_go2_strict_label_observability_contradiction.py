#!/usr/bin/env python3
"""Audit whether the Go2 strict hidden-target memory labels are a function of the
observable egocentric situation, or of a privileged ``seen_before`` flag.

The strict pure-RGB/JEPA gate labels each current-view query positive iff the
event's ``seen_before`` flag is set, and buckets it ``no_prior_current_hidden``
when the queried object has no prior rendered RGB footprint and is not currently
visible. If the *same* observable situation -- ``(scene_id, cell_id, yaw_bin,
color)`` -- carries both positive and negative labels, then a pure-RGB model
cannot separate them by construction, and no threshold tuning or additional
no-prior negative data can close the false-claim gap.

This script reconstructs the queries exactly as
``scripts/train_go2_jepa_relative_landmark_memory_probe.py`` does (per-row
``current_*`` events, per-row dedup by ``(color, seen_before)``,
``group_key=(scene, cell, yaw_bin, color)``) and reports the contradiction rate.
It is image-free: prior-RGB observability counts come from the separate
``audit_go2_rgb_memory_observability.py`` pass, cited in the doc.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# Color vocabulary mirrors train_go2_rgb_jepa_vector_memory_controller._COLOR_RGB
# (order matters only for substring matching, longest-distinct names are fine).
_COLORS = ("red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple")


def _object_color(object_id: str) -> str | None:
    lowered = str(object_id).lower()
    for color in _COLORS:
        if color in lowered:
            return color
    return None


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _extract_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct deduplicated current-view queries faithfully to the trainer."""
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)

    queries: list[dict[str, Any]] = []
    for sequence_rows in sequences.values():
        sequence_rows.sort(key=lambda r: int(r.get("episode_step", 0)))
        for row in sequence_rows:
            seen_keys: set[tuple[str, bool]] = set()
            for event in row.get("go2_causal_memory_pair_selection", ()):
                if not str(event.get("pair_role", "")).startswith("current_"):
                    continue
                color = _object_color(str(event.get("object_id", "")))
                if color is None:
                    continue
                seen_before = bool(event.get("seen_before", False))
                dedup = (color, seen_before)
                if dedup in seen_keys:
                    continue
                seen_keys.add(dedup)
                queries.append(
                    {
                        "scene_id": str(row.get("scene_id", "")),
                        "cell_id": int(row.get("cell_id", -1)),
                        "yaw_bin": int(row.get("yaw_bin", -1)),
                        "color": color,
                        "target": 1 if seen_before else 0,
                        "current_visible": bool(
                            event.get("current_visible", event.get("visible", False))
                        ),
                    }
                )
    return queries


def _contradiction_stats(
    queries: list[dict[str, Any]], key_fn
) -> dict[str, Any]:
    groups: dict[tuple, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    for q in queries:
        side = "pos" if q["target"] == 1 else "neg"
        groups[key_fn(q)][side] += 1
    contradictory = {k: v for k, v in groups.items() if v["pos"] and v["neg"]}
    total_pos = sum(v["pos"] for v in groups.values())
    total_neg = sum(v["neg"] for v in groups.values())
    pos_in_bad = sum(v["pos"] for v in contradictory.values())
    neg_in_bad = sum(v["neg"] for v in contradictory.values())
    return {
        "groups": len(groups),
        "contradictory_groups": len(contradictory),
        "total_positives": total_pos,
        "total_negatives": total_neg,
        "positives_in_contradictory_group": pos_in_bad,
        "negatives_in_contradictory_group": neg_in_bad,
        "frac_positives_contradictory": pos_in_bad / max(1, total_pos),
        "frac_negatives_contradictory": neg_in_bad / max(1, total_neg),
    }


def _per_scene_color(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"pos": 0, "neg": 0, "groups": set(), "bad_groups": set()}
    )
    pose_groups: dict[tuple, dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    for q in queries:
        sc = (q["scene_id"], q["color"])
        side = "pos" if q["target"] == 1 else "neg"
        groups[sc][side] += 1
        pk = (q["scene_id"], q["cell_id"], q["yaw_bin"], q["color"])
        pose_groups[pk][side] += 1
    # attribute pose-group contradiction back to (scene,color)
    bad_pose = {k for k, v in pose_groups.items() if v["pos"] and v["neg"]}
    out = []
    for (scene, color), v in sorted(groups.items()):
        bad = sum(1 for k in bad_pose if k[0] == scene and k[3] == color)
        out.append(
            {
                "scene_id": scene,
                "color": color,
                "positives": v["pos"],
                "negatives": v["neg"],
                "contradictory_pose_groups": bad,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+", help="dataset shard(s)")
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--label", type=str, default="split")
    args = parser.parse_args()

    rows = _load_rows(args.jsonl)
    queries = _extract_queries(rows)

    pose_key = lambda q: (q["scene_id"], q["cell_id"], q["yaw_bin"], q["color"])
    cell_key = lambda q: (q["scene_id"], q["cell_id"], q["color"])

    report = {
        "label": args.label,
        "shards": [str(p) for p in args.jsonl],
        "row_count": len(rows),
        "query_count": len(queries),
        "positive_queries": sum(1 for q in queries if q["target"] == 1),
        "negative_queries": sum(1 for q in queries if q["target"] == 0),
        "contradiction_scene_cell_yaw_color": _contradiction_stats(queries, pose_key),
        "contradiction_scene_cell_color": _contradiction_stats(queries, cell_key),
        "per_scene_color": _per_scene_color(queries),
    }

    out_path = args.report_output
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    s = report["contradiction_scene_cell_yaw_color"]
    print(
        f"[{args.label}] rows={report['row_count']} queries={report['query_count']} "
        f"(pos={report['positive_queries']} neg={report['negative_queries']})"
    )
    print(
        f"  (scene,cell,yaw,color): {s['contradictory_groups']}/{s['groups']} groups "
        f"contradictory; {s['frac_positives_contradictory']:.2f} of positives and "
        f"{s['frac_negatives_contradictory']:.2f} of negatives live in a both-labeled group"
    )
    s2 = report["contradiction_scene_cell_color"]
    print(
        f"  (scene,cell,color):     {s2['contradictory_groups']}/{s2['groups']} groups "
        f"contradictory"
    )
    if out_path is not None:
        print(f"  report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
