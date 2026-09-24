#!/usr/bin/env python3
"""Reproduce terminal metrics from persisted corpus rows and model probabilities."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from lewm.safety import lightweight_one_tick_viability_model_v1 as METRICS
from scripts import run_two_ply_successor_transition_corpus_repair_and_model_v1 as RUN

OUT = ROOT / ".generated/two_ply_successor_transition_corpus_repair_and_model_v1"
CORPUS = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1/corpus_index.json"


def close(left, right, tolerance=1e-12):
    return (left is None and right is None) or math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    result = json.loads((OUT / "result.json").read_text()); corpus = json.loads(CORPUS.read_text())
    records = {row["state_id"]: row for row in corpus["records"]}
    rows = [json.loads(line) for line in Path(result["row_ledger"]["path"]).read_text().splitlines()]
    held = [row for row in rows if row["role"] == "heldout"]
    organized = {}
    for state_id in sorted({row["state_id"] for row in held}):
        current = sorted([row for row in held if row["state_id"] == state_id and row["level"] == "current"], key=lambda row: row["action_index"])
        successor = [row for row in held if row["state_id"] == state_id and row["level"] == "successor"]
        successor_values = {}
        for action in sorted({row["current_action_index"] for row in successor}):
            selected = sorted([row for row in successor if row["current_action_index"] == action], key=lambda row: row["action_index"])
            successor_values[action] = {"probability": np.asarray([row["probability"] for row in selected]),
                                        "logits": np.asarray([row["raw_logit"] for row in selected])}
        organized[state_id] = {"current_probability": np.asarray([row["probability"] for row in current]),
                               "current_logits": np.asarray([row["raw_logit"] for row in current]), "successors": successor_values}
    held_records = [records[state_id] for state_id in json.loads((ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1/development_internal_calibration_repaired_v1.json").read_text())["development_heldout_state_ids"]]
    threshold = result["calibration"]["threshold"]
    current, successor, _families = RUN.contact_metrics(held_records, organized, threshold)
    decision, count = RUN.decisions(held_records, organized, threshold)
    checks = {
        "ledger_rows": len(rows) == result["row_ledger"]["rows"],
        "current_auc": close(current["auc"], result["heldout"]["current_contact"]["auc"]),
        "current_ap": close(current["ap"], result["heldout"]["current_contact"]["ap"]),
        "successor_auc": close(successor["auc"], result["heldout"]["successor_contact"]["auc"]),
        "successor_ap": close(successor["ap"], result["heldout"]["successor_contact"]["ap"]),
        "count_mae": close(count["mae"], result["heldout"]["safe_action_count"]["mae"]),
        "count_zero_nonzero": close(count["zero_vs_nonzero_accuracy"], result["heldout"]["safe_action_count"]["zero_vs_nonzero_accuracy"]),
        "decision_retention": decision["states_retaining_admitted_action"] == result["heldout"]["decision"]["states_retaining_admitted_action"],
        "decision_contacts": decision["selected_current_contacts"] == result["heldout"]["decision"]["selected_current_contacts"],
        "decision_nonviable": decision["selected_oracle_nonviable_successors"] == result["heldout"]["decision"]["selected_oracle_nonviable_successors"],
        "route_progress": close(decision["selected_h3_route_progress_m"], result["heldout"]["decision"]["selected_h3_route_progress_m"]),
        "route_regret": close(decision["normalized_regret"], result["heldout"]["decision"]["normalized_regret"]),
    }
    payload = {"schema": "two_ply_successor_transition_result_checker_v1", "checks": checks, "pass": all(checks.values()),
               "model_inference_executed": False, "rows_reduced": len(rows)}
    (OUT / "result_checker.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2)); return 0 if payload["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
