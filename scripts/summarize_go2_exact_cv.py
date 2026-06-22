#!/usr/bin/env python3
"""Verify the exact-odometry leave-one-scene-out CV gate.

Gate (per held-out scene): target_steering_pipeline_success >= 0.90,
false_claim_rate <= 0.12, corruption gap >= 0.30, with recall preserved.
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

STEER_BAR = 0.90
FCLAIM_BAR = 0.12
GAP_BAR = 0.30


def main() -> int:
    d = ".generated/go2_hidden_target_memory/observed_memory_gate_20260622/exact_cv"
    per_scene = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{d}/exact_*_report.json")):
        name = Path(f).name.replace("exact_", "").replace("_report.json", "")
        scene, seed = name.rsplit("_s", 1)
        r = json.load(open(f))
        m = r["best_validation_selected_metrics"]
        per_scene[scene].append(
            {
                "seed": seed,
                "recall": m["target_recall"],
                "steer": m["target_steering_pipeline_success"],
                "fclaim": m["false_claim_rate"],
                "gap": r["normal_minus_best_corrupted_target_steering_pipeline_success"],
            }
        )

    print(
        f'{"scene":14s} {"seeds":>5} | {"recall":>6} {"steer":>6} {"fclaim":>6} {"corrupt_gap":>11} | '
        f"steer_pass  gate_pass"
    )
    scenes_pass = 0
    for scene in sorted(per_scene):
        runs = per_scene[scene]
        n = len(runs)
        recall = sum(x["recall"] for x in runs) / n
        steer = sum(x["steer"] for x in runs) / n
        fclaim = sum(x["fclaim"] for x in runs) / n
        gap = sum(x["gap"] for x in runs) / n
        steer_pass = sum(1 for x in runs if x["steer"] >= STEER_BAR)
        gate_pass = sum(
            1 for x in runs
            if x["steer"] >= STEER_BAR and x["fclaim"] <= FCLAIM_BAR and x["gap"] >= GAP_BAR
        )
        scene_ok = steer >= STEER_BAR and fclaim <= FCLAIM_BAR and gap >= GAP_BAR
        scenes_pass += int(scene_ok)
        print(
            f"{scene:14s} {n:>5} | {recall:6.2f} {steer:6.2f} {fclaim:6.3f} {gap:11.2f} | "
            f"{steer_pass}/{n}        {gate_pass}/{n}  {'PASS' if scene_ok else 'fail'}"
        )
    print(f"\nScenes passing (mean steer>=0.90, fclaim<=0.12, gap>=0.30): {scenes_pass}/{len(per_scene)}")
    print(f"Gate target: >=4/5 scenes pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
