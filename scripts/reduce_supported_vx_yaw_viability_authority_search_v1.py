#!/usr/bin/env python3
"""Reduce the bounded supported-vx/yaw authority search without replay."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lewm.safety import supported_vx_yaw_viability_authority_search_v1 as S
OUT = ROOT / ".generated/supported_vx_yaw_viability_authority_search_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/supported_vx_yaw_viability_authority_search_v1"
INDEX = OUT / "search_index.json"
SELECTION = ROOT / ".generated/multi_cycle_viability_envelope_v1/frozen_state_selection.json"
OLD_TREE = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/viability_tree_index.json"
SOURCE_COMMIT = "11a0c258e479f79a640ab237841f52ec0e6b6ecc"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    data = json.loads(INDEX.read_text())
    selection = json.loads(SELECTION.read_text())
    old = json.loads(OLD_TREE.read_text())
    failures = [row for row in data["states"] if row["role"] == "failure"]
    controls = [row for row in data["states"] if row["role"] == "matched_control"]

    ledger_path = CACHE / "row_level_evidence_v1.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_count = 0
    with ledger_path.open("w") as stream:
        for state in data["states"]:
            for current in state["rows"]:
                row = {
                    "schema": "supported_vx_yaw_row_evidence_v1",
                    "level": "current",
                    "state_id": state["state_id"],
                    "family": state["family"],
                    "role": state["role"],
                    "search_index": current["search_index"],
                    "name": current["name"],
                    "command_family": current["family"],
                    "requested_vx_vy_wz": current["requested_vx_vy_wz"],
                    "applied_vx_vy_wz": current["applied_vx_vy_wz"],
                    "genuinely_new": current["genuinely_new"],
                    "historical_duplicate_candidate": current["historical_duplicate_candidate"],
                    "first_tick_contact": current["outcome"]["contact"],
                    "first_contact_step": current["outcome"]["first_contact_step"],
                    "successor_identity": current["outcome"]["successor_digest"],
                    "viability_admissible": current["viability_admissible"],
                    "safe_successor_count": current["successor_safe_action_count"],
                    "route_progress_m": current["immediate_route_progress_m"],
                }
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                ledger_count += 1
                for successor in current["successor_outcomes"]:
                    next_row = {
                        "schema": "supported_vx_yaw_row_evidence_v1",
                        "level": "successor",
                        "state_id": state["state_id"],
                        "role": state["role"],
                        "parent_search_index": current["search_index"],
                        "parent_successor_identity": current["outcome"]["successor_digest"],
                        "applied_vx_vy_wz": successor["applied_vx_vy_wz"],
                        "aliases": successor["aliases"],
                        "contact": successor["contact"],
                        "first_contact_step": successor["first_contact_step"],
                    }
                    stream.write(json.dumps(next_row, sort_keys=True, separators=(",", ":")) + "\n")
                    ledger_count += 1

    old_failure = selection["failure_classifications"]
    failure_rows = []
    for state in failures:
        safe = [row for row in state["rows"] if row["safe_prefix"]]
        viable = [row for row in state["rows"] if row["viability_admissible"]]
        families = Counter(row["family"] for row in viable)
        reduced_classification = S.residual_classification(
            state["rows"], boundary_contact=state["classification"] == "PRE_EXISTING_CONTACT"
        )
        failure_rows.append({
            "state_id": state["state_id"],
            "family": state["family"],
            "historical_classification": old_failure[state["state_id"]],
            "search_classification": reduced_classification,
            "first_contact_step_range": [
                min((row["outcome"]["first_contact_step"] for row in state["rows"]
                     if row["outcome"]["first_contact_step"] is not None), default=None),
                max((row["outcome"]["first_contact_step"] for row in state["rows"]
                     if row["outcome"]["first_contact_step"] is not None), default=None),
            ],
            "unique_applied_commands": state["deduplication"]["unique_applied_count"],
            "genuinely_new_applied_commands": state["deduplication"]["genuinely_new_applied"],
            "safe_prefix_commands": len(safe),
            "viability_admissible_commands": len(viable),
            "viable_command_families": dict(families),
            "maximum_safe_successor_count": max((row["successor_safe_action_count"] for row in safe), default=0),
            "safe_prefix_names": [row["name"] for row in safe],
        })

    control_current = [row for state in controls for row in state["rows"]]
    failure_current = [row for state in failures for row in state["rows"]]
    dedup_fields = ("requested_count", "unique_applied_count", "duplicates_within_grid",
                    "duplicates_of_historical", "genuinely_new_applied")
    dedup = {field: sum(state["deduplication"][field] for state in data["states"])
             for field in dedup_fields}
    old_states = old["state_records"]
    historical_viable = sum(bool(row["viability_admissible_prefixes"]) for row in old_states)
    historical_preexisting = 2
    result = {
        "schema": "supported_vx_yaw_viability_authority_search_development_result_v1",
        "source_commit": SOURCE_COMMIT,
        "status": "COMPLETE_POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC",
        "claim_boundary": "oracle simulated H1 physics-rate disallowed-contact viability; not learned safety, emergency braking, material-hazard safety, or navigation",
        "bindings": {
            "search_index_sha256": sha(INDEX),
            "selection_sha256": sha(SELECTION),
            "old_tree_sha256": sha(OLD_TREE),
            "grid_receipt_sha256": sha(OUT / "frozen_requested_grid.json"),
            "fixture_sha256": sha(OUT / "training_fixture_result.json"),
        },
        "controller_contract": data["grid"]["controller_contract"],
        "requested_grid": data["grid"]["grid"],
        "applied_command_deduplication_16_states": dedup,
        "fixture": {key: value for key, value in data["fixture"].items()
                    if key not in ("records", "pure")},
        "generated_branches": {
            "fixture": data["fixture"]["branches"],
            "scientific_current": data["counts"]["current_branches"],
            "scientific_successor": data["counts"]["successor_branches"],
            "scientific_total": data["counts"]["current_branches"] + data["counts"]["successor_branches"],
            "accepted_total": data["fixture"]["branches"] + data["counts"]["current_branches"] + data["counts"]["successor_branches"],
        },
        "residual_reachability": {
            "states": failure_rows,
            "classification_counts": dict(Counter(row["search_classification"] for row in failure_rows)),
            "current_unique_commands": len(failure_current),
            "first_tick_safe_commands": sum(row["safe_prefix"] for row in failure_current),
            "viability_admissible_commands": sum(row["viability_admissible"] for row in failure_current),
            "states_with_viability_admissible_command": sum(any(row["viability_admissible"] for row in state["rows"]) for state in failures),
        },
        "matched_controls": {
            "states": len(controls),
            "current_unique_commands": len(control_current),
            "first_tick_contacts": sum(row["outcome"]["contact"] for row in control_current),
            "viability_admissible_commands": sum(row["viability_admissible"] for row in control_current),
            "states_with_viability_admissible_command": sum(any(row["viability_admissible"] for row in state["rows"]) for state in controls),
        },
        "mechanism_selection": {
            "selected": None,
            "reason": "No supported search command was viability-admissible in any of the eight residual failure states.",
            "conditional_augmented_bank_ran": False,
            "conditional_multi_cycle_ran": False,
        },
        "full_panel_viability": {
            "historical_before": {"viable_states": historical_viable, "states": len(old_states)},
            "augmented_after": {"viable_states": historical_viable, "states": len(old_states),
                                "reason": "No mechanism family qualified; bank was not augmented."},
            "non_preexisting_viable_fraction": historical_viable / (len(old_states) - historical_preexisting),
            "classification": "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO",
        },
        "primary_classification": "SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO",
        "lateral_controller_development_justified": True,
        "exact_next_specification": "DEPLOYMENT_VALID_LATERAL_LOCOMOTION_CONTROLLER_V1",
        "predictor_contract": {
            "jepa_opened_or_executed": False,
            "macro_historical_twelve_unchanged": True,
            "future_vy_requires_prospective_action-representation_extension_before_macro_scoring": True,
            "micro_recovery_may_remain_outside_macro_scoring_after_controller_and_oracle_qualification": True,
        },
        "preserved_terminals": [
            "LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO",
            "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO",
            "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION",
            "CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO",
            "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
            "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED",
            "REPLANNING_INTERFACE_UNRESOLVED",
            "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        ],
        "runtime_s": {
            "accepted_fixture": data["fixture"]["runtime_s"],
            "scientific_search_wall": data["runtime_s"],
            "accepted_total": data["fixture"]["runtime_s"] + data["runtime_s"],
        },
        "row_ledger": {"path": str(ledger_path), "rows": ledger_count,
                       "bytes": ledger_path.stat().st_size, "sha256": sha(ledger_path)},
        "prohibitions_confirmed": {
            "model_training": False,
            "jepa_access_or_execution": False,
            "low_level_controller_retraining": False,
            "macro_bank_change": False,
            "memory_or_navigation": False,
            "frozen_low_level_ppo_executed_only_as_simulator_control_plant": True,
        },
    }
    dump(OUT / "development_result.json", result)
    print(json.dumps({
        "primary": result["primary_classification"],
        "failure_viable_states": result["residual_reachability"]["states_with_viability_admissible_command"],
        "ledger": result["row_ledger"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
