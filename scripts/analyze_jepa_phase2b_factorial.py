#!/usr/bin/env python3
"""Summarize and gate the bounded Phase 2B pooled/spatial factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _with_meaningful_action_gates(step: dict) -> dict:
    result = dict(step)
    target_change = float(result.get("target_step_delta_token_mse", 0.0))
    for action in ("zero", "shuffled"):
        advantage = float(result.get(f"{action}_minus_free_running_mse", 0.0))
        normalized = advantage / target_change if target_change > 0.0 else 0.0
        result.setdefault(
            f"{action}_action_advantage_over_target_change",
            normalized,
        )
        result.setdefault(
            f"meaningful_real_action_beats_{action}",
            normalized >= 0.1,
        )
    return result


def _cell_summary(report: dict) -> dict:
    evaluation = report["final"]["eval"]
    steps = evaluation["per_horizon_step"]
    selection = evaluation["selection"]
    representation = evaluation["representation"]
    return {
        "trainable_parameters": int(report["trainable_parameters"]),
        "scene_overlap": report["scene_overlap"],
        "usable_train_sequences": int(
            report["train_input_audit"]["usable_complete_valid_sequences"]
        ),
        "usable_eval_sequences": int(
            report["eval_input_audit"]["usable_complete_valid_sequences"]
        ),
        "collapse_warning": bool(representation["collapse_warning"]),
        "mean_feature_std": float(representation["mean_feature_std"]),
        "step1": _with_meaningful_action_gates(steps[0]),
        "step2": _with_meaningful_action_gates(steps[1]),
        "selection": selection,
    }


def analyze(root: Path) -> dict:
    cells = {
        name: _cell_summary(json.loads((root / f"{name}.json").read_text()))
        for name in ("pooled", "spatial_var", "spatial_no_var")
    }
    pooled = cells["pooled"]
    spatial = cells["spatial_var"]
    gates = {
        "scene_disjoint": not spatial["scene_overlap"],
        "spatial_not_collapsed": not spatial["collapse_warning"],
        "step1_real_action_meaningfully_beats_zero": bool(
            spatial["step1"]["meaningful_real_action_beats_zero"]
        ),
        "step1_real_action_meaningfully_beats_shuffled": bool(
            spatial["step1"]["meaningful_real_action_beats_shuffled"]
        ),
        "step1_beats_persistence": bool(
            spatial["step1"]["free_running_beats_persistence"]
        ),
        "step2_improves_over_pooled": (
            float(spatial["step2"]["free_running_vs_persistence_mse_ratio"])
            < float(pooled["step2"]["free_running_vs_persistence_mse_ratio"])
        ),
        "safe_positive_progress_not_worse_than_pooled": (
            float(spatial["selection"]["safe_positive_progress_rate"])
            >= float(pooled["selection"]["safe_positive_progress_rate"])
        ),
        "newly_unsafe_not_worse_than_pooled": (
            float(spatial["selection"]["selected_enters_grid_unsafe_rate"])
            <= float(pooled["selection"]["selected_enters_grid_unsafe_rate"])
        ),
    }
    return {
        "schema": "jepa_phase2b_bounded_factorial_analysis_v0",
        "root": str(root.resolve()),
        "cells": cells,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
        "decision": (
            "promote_to_full_capacity_replication"
            if all(gates.values())
            else "stop_and_redesign_before_scaling"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
