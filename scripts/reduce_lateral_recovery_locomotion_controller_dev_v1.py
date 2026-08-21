#!/usr/bin/env python3
"""Reduce controller training and qualification evidence without replay."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/lateral_recovery_locomotion_controller_dev_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/lateral_recovery_locomotion_controller_dev_v1"
SOURCE_CFG = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    training = json.loads((OUT / "training_result.json").read_text())
    smoke = json.loads((OUT / "smoke_result.json").read_text())
    qualification = json.loads((OUT / "controller_qualification.json").read_text())
    run_dir = OUT / "seed_2026082014"
    with SOURCE_CFG.open("rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    command_cfg = dict(command_cfg); command_cfg["lin_vel_y_range"] = [-0.2, 0.2]
    command_cfg["sampler"] = "fixed_50_route_25_lateral_25_transition_v1"
    command_cfg["transition_delay_policy_steps"] = 50
    inference_cfg = run_dir / "inference_cfgs.pkl"
    with inference_cfg.open("wb") as f:
        pickle.dump((env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg), f)

    event = next(run_dir.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event)); accumulator.Reload()
    metrics = {}
    for tag in accumulator.Tags()["scalars"]:
        values = accumulator.Scalars(tag)
        if values:
            metrics[tag] = {"first_step": values[0].step, "first": values[0].value,
                            "final_step": values[-1].step, "final": values[-1].value}

    ledger = CACHE / "controller_qualification_rows_v1.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with ledger.open("w") as stream:
        for panel in ("route_non_regression", "lateral_tracking", "mode_transition"):
            for row in qualification[panel]["rows"]:
                stream.write(json.dumps({"schema": "lateral_controller_qualification_row_v1",
                                         "panel": panel, **row}, sort_keys=True, separators=(",", ":")) + "\n")
                rows += 1

    result = {
        "schema": "lateral_recovery_locomotion_controller_dev_result_v1",
        "source_commit": "690bd1ffbf0a59ba806fb62d4d5fe521f296bd3f",
        "status": "STOPPED_AT_CONTROLLER_QUALIFICATION_GATE",
        "training": training,
        "training_metrics": metrics,
        "smoke": smoke,
        "qualification": {key: value for key, value in qualification.items()
                          if key not in ("route_non_regression", "lateral_tracking", "mode_transition")},
        "route_non_regression": {key: value for key, value in qualification["route_non_regression"].items() if key != "rows"},
        "lateral_tracking": {key: value for key, value in qualification["lateral_tracking"].items() if key != "rows"},
        "mode_transition": {key: value for key, value in qualification["mode_transition"].items() if key != "rows"},
        "controller_qualification_classification": "LATERAL_TRACKING_AUTHORITY_NO_GO",
        "primary_classification": "LATERAL_CONTROLLER_QUALIFICATION_NO_GO",
        "scientific_viability_stage_ran": False,
        "scientific_branch_counts": {"current_lateral": 0, "successor_augmentation": 0, "multi_cycle": 0},
        "full_panel_viability": {"before": "40/48", "after": "not evaluated; controller did not qualify",
                                 "classification": "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO"},
        "exact_blocker": "The one-seed controller did not meet mirrored lateral tracking or deterministic-repeat gates; no automatic second seed or tuning is authorized.",
        "inference_cfg": {"path": str(inference_cfg), "sha256": sha(inference_cfg), "bytes": inference_cfg.stat().st_size},
        "row_ledger": {"path": str(ledger), "sha256": sha(ledger), "bytes": ledger.stat().st_size, "rows": rows},
        "runtime_s": {"smoke": smoke["runtime_s"], "training": training["runtime_s"],
                      "qualification": qualification["runtime_s"],
                      "accepted_total": smoke["runtime_s"] + training["runtime_s"] + qualification["runtime_s"]},
        "preserved": ["SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO", "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO",
                      "LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO", "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION",
                      "CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO", "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
                      "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED", "REPLANNING_INTERFACE_UNRESOLVED",
                      "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING"],
        "prohibitions": {"complete_controller_seeds_trained": 1, "jepa_opened_or_executed": False,
                         "utility_or_safety_model": False, "historical_route_controller_modified": False,
                         "memory_or_navigation": False},
    }
    write(OUT / "development_result.json", result)
    print(json.dumps({"primary": result["primary_classification"], "checkpoint": training["checkpoint"],
                      "ledger": result["row_ledger"], "runtime": result["runtime_s"]}, indent=2))


if __name__ == "__main__": main()
