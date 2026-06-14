#!/usr/bin/env python3
"""Compare the Phase 2C EMA target encoder against the Phase 2B spatial control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_jepa_phase2b_factorial import _cell_summary  # noqa: E402


def analyze(phase2b_root: Path, phase2c_report: Path) -> dict:
    online = _cell_summary(json.loads((phase2b_root / "spatial_var.json").read_text()))
    ema_report = json.loads(phase2c_report.read_text())
    ema = _cell_summary(ema_report)
    gates = {
        "scene_disjoint": not ema["scene_overlap"],
        "ema_target_configured": (
            ema_report["target_encoder"]["mode"] == "stop_gradient_ema"
        ),
        "not_collapsed": not ema["collapse_warning"],
        "step1_real_action_meaningfully_beats_zero": bool(
            ema["step1"]["meaningful_real_action_beats_zero"]
        ),
        "step1_real_action_meaningfully_beats_shuffled": bool(
            ema["step1"]["meaningful_real_action_beats_shuffled"]
        ),
        "step1_beats_persistence": bool(ema["step1"]["free_running_beats_persistence"]),
        "step2_improves_over_online_target": (
            float(ema["step2"]["free_running_vs_persistence_mse_ratio"])
            < float(online["step2"]["free_running_vs_persistence_mse_ratio"])
        ),
        "safe_positive_progress_not_worse": (
            float(ema["selection"]["safe_positive_progress_rate"])
            >= float(online["selection"]["safe_positive_progress_rate"])
        ),
        "newly_unsafe_not_worse": (
            float(ema["selection"]["selected_enters_grid_unsafe_rate"])
            <= float(online["selection"]["selected_enters_grid_unsafe_rate"])
        ),
    }
    return {
        "schema": "jepa_phase2c_ema_gate_analysis_v0",
        "online_target_control": online,
        "ema_target": ema,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
        "decision": (
            "retain_ema_target_and_replicate"
            if all(gates.values())
            else "redesign_action_conditioned_target_before_scaling"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2b-root", type=Path, required=True)
    parser.add_argument("--phase2c-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.phase2b_root, args.phase2c_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
