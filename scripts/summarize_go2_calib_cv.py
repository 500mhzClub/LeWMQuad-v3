#!/usr/bin/env python3
"""Aggregate the ranged-write CV: full-gate vector steering vs majority baseline
(pursue 3) and the short-horizon in-cone-gap breakdown (prove 1)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

MAJORITY = {
    "000c67a65968": 0.581,
    "01732aabc542": 1.0,
    "04f670cb21f8": 0.933,
    "48a6e58aedad": 0.641,
    "e06e3c25bf84": 0.828,
}


def main() -> int:
    d = ".generated/go2_hidden_target_memory/observed_memory_gate_20260622/calib_cv"
    print(
        f'{"scene":14s} {"recall":>6} {"fclaim":>6} {"gap":>5} | '
        f'{"maj":>5} {"head":>5} {"vector":>6} | vector by in-cone gap (le2/le4/le8/le16/gt16)'
    )
    for f in sorted(glob.glob(f"{d}/calib_*_report.json")):
        scene = Path(f).name.replace("calib_", "").replace("_report.json", "")
        r = json.load(open(f))
        sd = r["steering_diagnostics"]
        head = sd["head"]["selected_positive_accuracy"]
        vec = sd["vector"]["selected_positive_accuracy"]
        recall = r.get("target_recall", float("nan"))
        fc = r.get("false_claim_rate", float("nan"))
        gap = r.get("normal_minus_best_corrupted_target_steering_pipeline_success", float("nan"))
        maj = MAJORITY.get(scene, float("nan"))
        gb = sd.get("steering_by_incone_gap", {}).get("vector", {})

        def cell(lbl):
            c = gb.get(lbl, {})
            n = c.get("n", 0)
            a = c.get("acc", float("nan"))
            return f"{a:.2f}(n{n})" if n else "  -    "

        print(
            f"{scene:14s} {recall:6.2f} {fc:6.2f} {gap:5.2f} | "
            f"{maj:5.2f} {head:5.2f} {vec:6.2f} | "
            f"{cell('le2')} {cell('le4')} {cell('le8')} {cell('le16')} {cell('gt16')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
