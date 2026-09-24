#!/usr/bin/env python3
"""Freeze the pre-training successor-evidence reconstruction blocker."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/two_ply_set_structured_micro_viability_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_set_structured_micro_viability_v1"
PREDECESSOR = OUT / "predecessor_failure_attribution.json"
BLOCKING_LOG = CACHE / "logs/state_092_viability-fit-2-04.log"
ROW_LEDGER = Path.home() / ".cache/lewm_go2_temporal_v03/lightweight_one_tick_viability_model_and_interface_v1/row_level_evidence_v1.jsonl"
MODEL_LEDGER = Path.home() / ".cache/lewm_go2_temporal_v03/development_micro_viability_model_screen_v1/row_level_model_evidence_v1.jsonl"
SPLIT = ROOT / ".generated/development_micro_viability_model_screen_v1/development_internal_calibration_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    if not path.exists(): return 0
    excluded = OUT / "result.json"
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and item != excluded)


def main() -> int:
    if sha(ROW_LEDGER) != "0a273a3f464f770ccf8d28a1c6c3d9ddad63efdb767c1a63175ddcb479a18eea":
        raise RuntimeError("oracle row ledger binding failed")
    if sha(MODEL_LEDGER) != "555ba6d2678e543cf78d6a53977eceeaa5bddf60a6c16c2510ee028db9f7cba2":
        raise RuntimeError("predecessor model ledger binding failed")
    records = []
    for path in sorted((OUT / "states").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("label_binding_version") == 3 and row.get("status") == "PASS": records.append(row)
    matches = re.findall(r"evaluation successor reconstruction mismatch: (\{.*?\})", BLOCKING_LOG.read_text())
    blocking = None
    for candidate in matches:
        try: blocking = ast.literal_eval(candidate)
        except (ValueError, SyntaxError): continue
    if blocking is None: raise RuntimeError("blocking mismatch not found")
    split = json.loads(SPLIT.read_text())
    if "viability-fit-2-04" not in split["internal_calibration_state_ids"]: raise RuntimeError("blocking role mismatch")
    transition_counts = {}
    for role in ("training", "calibration", "heldout"):
        selected = [row for row in records if row["role"] == role]
        compatible_successors = sum(len(row["successors"]) for row in selected)
        transition_counts[role] = {"completed_states": len(selected), "current_transitions": len(selected) * 14,
                                   "compatible_successor_state_groups": compatible_successors,
                                   "compatible_successor_transitions": compatible_successors * 14,
                                   "total_compatible_transitions": len(selected) * 14 + compatible_successors * 14}
    mtimes = [path.stat().st_mtime for path in (OUT / "states").glob("*.json")] + [BLOCKING_LOG.stat().st_mtime]
    active = subprocess.run(["pgrep", "-af", "materialize_two_ply_set_structured_micro_viability_v1|run_two_ply_set_structured_micro_viability|benchmark_two_ply_set_structured"], capture_output=True, text=True).stdout.splitlines()
    active = [line for line in active if "pgrep -af" not in line]
    predecessor = json.loads(PREDECESSOR.read_text())
    result = {"schema": "two_ply_successor_evidence_reconstruction_blocker_v1", "source_commit": "94693e5a1b102de52782cef642d87ea89965d67f",
        "requested_experiment": "TWO_PLY_SET_STRUCTURED_MICRO_VIABILITY_V1", "status": "STOPPED_BEFORE_TRAINING",
        "classification": "TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER",
        "not_a_model_classification": True, "true_successor_signal_gate_evaluated": False,
        "reason": "actual non-legacy successor snapshots and individual next-action rows were not persisted, and deterministic replay does not reproduce a frozen internal-calibration safe-action count",
        "blocking_internal_calibration_state": {"state_id": "viability-fit-2-04", "role": "calibration", **blocking},
        "training_only_incompatible_successors": [dict(item, state_id=row["state_id"], family=row["family"])
            for row in records for item in row.get("incompatible_successors_excluded_from_training", [])],
        "completed_materialisation": {"states": len(records), "transition_counts": transition_counts,
            "branches_replayed": sum(row["branches_replayed"] for row in records),
            "legacy_current_replay_discrepancies": sum(len(row["current_label_replay_discrepancies"]) for row in records),
            "legacy_or_bound_next_replay_discrepancies": sum(len(row["next_label_replay_discrepancies"]) for row in records)},
        "bindings": {"oracle_row_ledger": {"path": str(ROW_LEDGER), "sha256": sha(ROW_LEDGER)},
                     "prompt_oracle_row_ledger_binding_note": "prompt string has 65 hexadecimal characters and an extraneous trailing 5; actual SHA-256 is the 64-character value above",
                     "predecessor_model_ledger": {"path": str(MODEL_LEDGER), "sha256": sha(MODEL_LEDGER)},
                     "development_split": {"path": str(SPLIT), "sha256": sha(SPLIT)}},
        "predecessor_failure_attribution": {"classifications": predecessor["classifications"], "splits": predecessor["splits"],
                                             "fit_to_heldout_degradation": predecessor["fit_to_heldout_degradation"]},
        "training": {"seeds_trained": 0, "checkpoint_created": False, "calibration_run": False, "heldout_model_evaluation_run": False,
                     "compute_benchmark_run": False},
        "next_required_experiment": {"name": "TWO_PLY_SUCCESSOR_TRANSITION_CORPUS_REPAIR_V1",
            "requirements": ["persist actual successor planning-time observation and full controller state at branch creation",
                             "persist all fourteen individual next-action physics-rate contact labels",
                             "bind aggregate safe-action count to those rows", "verify byte-stable regeneration before model initialization"],
            "fresh_panel_v2_collected": False, "model_training_authorized_before_repair": False},
        "runtime": {"completed_state_runtime_sum_s": sum(float(row["runtime_s"]) for row in records),
                    "diagnostic_wall_span_s": max(mtimes) - min(mtimes)},
        "storage_bytes": directory_bytes(CACHE) + directory_bytes(OUT), "active_processes": active,
        "preserved": ["FRESH_MICRO_VIABILITY_PANEL_INADEQUATE", "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL", "MICRO_VIABILITY_COMPUTE_SIGNAL",
                      "REPLANNING_INTERFACE_UNRESOLVED", "ONE_TICK_VIABILITY_KERNEL_NO_GO", "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION",
                      "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING"],
        "prohibited_work_confirmed_absent": {"fresh_panel": True, "jepa_access": True, "successor_predictor": True,
                                             "memory": True, "navigation": True}}
    result["content_digest"] = digest(result); atomic_json(OUT / "result.json", result)
    print(json.dumps({key: result[key] for key in ("classification", "blocking_internal_calibration_state", "completed_materialisation", "training", "runtime", "storage_bytes", "active_processes")}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
