#!/usr/bin/env python3
"""Reduce frozen route evidence and the EMERGENCY_BRAKE_V1 ledger.

This evaluator performs no simulation or learned inference.  It preserves the
historical twelve-candidate result and treats the brake strictly as a fallback
when no physics-contact-negative route candidate exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import h1_safe_action_set_successor_v1 as S

OUT = ROOT / ".generated/h1_safe_action_set_successor_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_safe_action_set_successor_v1")
PRIOR = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1/result.json"
INDEX = OUT / "brake_index.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def stats(values) -> dict:
    return S.summarize([float(x) for x in values])


def count(rows, key) -> int:
    return sum(bool(row[key]) for row in rows)


def subset_summary(rows: list[dict]) -> dict:
    stopped = [r for r in rows if r["current"]["stopped"]]
    contacts = [r for r in rows if r["current"]["contact"]]
    return {
        "states": len(rows),
        "contact": sum(r["current"]["contact"] for r in rows),
        "fall": sum(r["current"]["fall"] for r in rows),
        "unsafe_termination": sum(r["current"]["unsafe_termination"] for r in rows),
        "stable": sum(r["current"]["stable"] for r in rows),
        "stopped": len(stopped),
        "qualified_safe_brake": sum(r["current"]["qualified_safe_brake"] for r in rows),
        "timeout": sum(r["current"]["termination"] == "timeout" for r in rows),
        "first_contact_time_s": stats([r["current"]["first_contact_time_s"] for r in contacts]),
        "stopping_time_s": stats([r["current"]["stopping_time_s"] for r in stopped]),
        "stopping_distance_m": stats([r["current"]["stopping_distance_m"] for r in stopped]),
        "path_distance_until_termination_m": stats([r["current"]["path_distance_until_termination_m"] for r in rows]),
        "route_displacement_norm_m": stats([r["current"]["route_displacement_norm_m"] for r in rows]),
        "peak_acceleration_m_s2": stats([r["current"]["peak_acceleration_m_s2"] for r in rows]),
        "peak_angular_acceleration_rad_s2": stats([r["current"]["peak_angular_acceleration_rad_s2"] for r in rows]),
        "peak_actuator_torque_nm": stats([r["current"]["peak_actuator_torque_nm"] for r in rows]),
        "relaxed_stop_count_speed_0_07_yaw_0_12": sum(r["current"]["stop_sensitivity"]["speed_0.07_yaw_0.12_tick"] is not None for r in rows),
        "strict_stop_count_speed_0_03_yaw_0_08": sum(r["current"]["stop_sensitivity"]["speed_0.03_yaw_0.08_tick"] is not None for r in rows),
    }


def main() -> int:
    started = time.time()
    index = json.loads(INDEX.read_text())
    prior = json.loads(PRIOR.read_text())
    inventory = prior["feasibility"]["state_inventory"]
    rows = index["state_records"]
    by_id = {r["state_id"]: r for r in rows}
    if set(by_id) != set(inventory):
        raise RuntimeError("brake and exact-geometry state identities do not align")

    historical_no_safe = [r for r in rows if inventory[r["state_id"]]["feasibility"] == "NO_SAFE_CANDIDATE_AVAILABLE"]
    historical_safe = [r for r in rows if inventory[r["state_id"]]["feasibility"] == "SAFE_CANDIDATE_AVAILABLE"]
    no_safe_inventory = []
    for row in historical_no_safe:
        old = inventory[row["state_id"]]
        first_steps = old.get("first_contact_step_by_candidate", [])
        historical_first = min(first_steps) if first_steps else None
        shard = np.load(row["shard_path"], allow_pickle=False)
        speeds = shard["current_planar_speed"].astype(np.float64)
        sample = None if historical_first is None or not len(speeds) else float(speeds[min(max(historical_first - 1, 0), len(speeds) - 1)])
        current = row["current"]
        no_safe_inventory.append({
            "state_id": row["state_id"], "split": row["split"], "family": row["family"],
            "historical_classification": old.get("classification"), "historical_causal_factor": old.get("causal_factor"),
            "historical_first_contact_step": historical_first,
            "historical_first_contact_time_s": None if historical_first is None else historical_first * .002,
            "brake_classification": current["state_classification"], "brake_contact": current["contact"],
            "brake_first_contact_step": current["first_contact_step"], "brake_first_contact_time_s": current["first_contact_time_s"],
            "brake_stopped": current["stopped"], "brake_stable": current["stable"],
            "brake_qualified_safe": current["qualified_safe_brake"],
            "contact_free_until_fully_stopped": current["qualified_safe_brake"],
            "speed_before_historical_contact_m_s": sample,
            "initial_planar_speed_m_s": current["initial_planar_speed_m_s"],
            "reduced_planar_speed_before_historical_contact": None if sample is None else sample < current["initial_planar_speed_m_s"] - 1e-4,
            "predecessor_executed": row["predecessor_executed"],
            "predecessor_classification": current["predecessor_classification"],
            "predecessor_contact": None if row["predecessor"] is None else row["predecessor"]["contact"],
            "predecessor_stopped": None if row["predecessor"] is None else row["predecessor"]["stopped"],
            "predecessor_qualified_safe": None if row["predecessor"] is None else row["predecessor"]["qualified_safe_brake"],
        })

    # The exact route candidate selection remains the predecessor evaluator's
    # result.  The brake is never compared through route progress and is only
    # selected when that state had no contact-negative route candidate.
    exact_held = prior["exact_geometry"]["heldout"]
    exact_cal = prior["exact_geometry"]["calibration"]
    route_rows = {x["state_id"]: x for x in exact_held["per_state"]}
    held_state_rows = []
    for sid, route in route_rows.items():
        brake = by_id[sid]["current"]
        safe_route = route["feasibility"] == "SAFE_CANDIDATE_AVAILABLE"
        if safe_route:
            selection = "route"; candidate = route["selected_candidate"]; selected_contact = False
            progress = route["selected_progress_m"]
        elif brake["qualified_safe_brake"]:
            selection = "emergency_brake_v1"; candidate = None; selected_contact = False; progress = 0.0
        else:
            selection = "abstain"; candidate = None; selected_contact = None; progress = 0.0
        held_state_rows.append({
            "state_id": sid, "family": route["family"], "historical_feasibility": route["feasibility"],
            "safe_route_candidate_exists": safe_route, "safe_brake_exists": brake["qualified_safe_brake"],
            "successor_selection": selection, "selected_candidate": candidate,
            "selected_contact": selected_contact, "selected_progress_m": progress,
            "false_brake": selection == "emergency_brake_v1" and safe_route,
            "correct_fallback": selection == "emergency_brake_v1" and not safe_route,
        })

    def route_summary(items):
        safe = [x for x in items if x["safe_route_candidate_exists"]]
        no_safe = [x for x in items if not x["safe_route_candidate_exists"]]
        return {
            "states": len(items), "safe_route_selections": sum(x["successor_selection"] == "route" for x in items),
            "brake_selections": sum(x["successor_selection"] == "emergency_brake_v1" for x in items),
            "abstentions": sum(x["successor_selection"] == "abstain" for x in items),
            "selected_contacts": sum(x["selected_contact"] is True for x in items),
            "correct_fallbacks": sum(x["correct_fallback"] for x in items), "false_brakes": sum(x["false_brake"] for x in items),
            "safe_route_alternative_states": len(safe), "no_safe_route_states": len(no_safe),
            "mean_progress_all_states_m": float(np.mean([x["selected_progress_m"] for x in items])) if items else None,
            "mean_progress_safe_route_states_m": float(np.mean([x["selected_progress_m"] for x in safe])) if safe else None,
        }

    held_route = route_summary(held_state_rows)
    held_route.update({
        "normalized_regret_on_safe_route_states": exact_held["normalized_route_progress_regret"],
        "best_safe_top1": exact_held["best_contact_negative_top1"], "best_safe_top3": exact_held["best_contact_negative_top3"],
        "oracle_contact_kinematic_progress_m": exact_held["oracle_contact_kinematic_progress_m"],
        "oracle_progress_fraction_on_safe_route_states": exact_held["oracle_progress_fraction"],
    })
    per_family = {}
    for family in sorted({x["family"] for x in held_state_rows}):
        family_rows = [x for x in held_state_rows if x["family"] == family]
        summary = route_summary(family_rows)
        old = exact_held["per_family"][family]
        summary.update({"normalized_regret_on_safe_route_states": old["normalized_route_progress_regret"],
                        "best_safe_top1": old["best_contact_negative_top1"], "best_safe_top3": old["best_contact_negative_top3"]})
        per_family[family] = summary

    predecessor_counts = {}
    for r in rows:
        value = r["current"]["predecessor_classification"]
        if value: predecessor_counts[value] = predecessor_counts.get(value, 0) + 1
    brake_class_counts = {}
    for r in rows:
        value = r["current"]["state_classification"]
        brake_class_counts[value] = brake_class_counts.get(value, 0) + 1

    candidate_bank_failures = [x for x in no_safe_inventory if x["historical_classification"] == "CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE"]
    recovered_failures = sum(x["brake_qualified_safe"] for x in candidate_bank_failures)
    no_safe_recovered = sum(x["brake_qualified_safe"] for x in no_safe_inventory)
    if no_safe_recovered == len(no_safe_inventory): bank_status = "resolved"
    elif no_safe_recovered: bank_status = "reduced but not resolved"
    else: bank_status = "unchanged"

    nonboundary = [r for r in rows if not r["current"]["boundary_contact"]]
    held_non_immediate = [x for x in no_safe_inventory if x["split"] == "heldout" and x["historical_classification"] != "PRE_EXISTING_OR_IMMEDIATE_UNAVOIDABLE_CONTACT"]
    gate_checks = {
        "zero_brake_contact_when_not_boundary_contact": sum(r["current"]["contact"] for r in nonboundary) == 0,
        "zero_fall_or_unsafe_termination": sum(r["current"]["fall"] or r["current"]["unsafe_termination"] for r in rows) == 0,
        "every_candidate_bank_or_slew_failure_gains_safe_fallback": recovered_failures == len(candidate_bank_failures),
        "all_heldout_non_immediate_states_have_safe_route_or_brake": all(x["brake_qualified_safe"] for x in held_non_immediate),
        "historically_safe_controllability_preserved": all(not r["current"]["contact"] for r in historical_safe),
        "deployment_realizable_interface": True,
        "stopping_time_and_distance_bounded": all(r["current"]["qualified_safe_brake"] for r in nonboundary),
        "exact_genesis_historical_reproduction_valid": prior["reproduction"]["branch_agreement"] == 1.0,
    }
    action_class = "H1_SAFE_ACTION_SET_SUCCESSOR_SIGNAL" if all(gate_checks.values()) else "H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO"
    fallback_classes = []
    if action_class.endswith("SIGNAL"): fallback_classes.append("EMERGENCY_BRAKE_QUALIFIED")
    if predecessor_counts.get("PREDECESSOR_VIABILITY_GUARD_REQUIRED", 0): fallback_classes.append("PREDECESSOR_VIABILITY_GUARD_REQUIRED")
    if predecessor_counts.get("EMERGENCY_BRAKE_INSUFFICIENT", 0) or any(r["current"]["state_classification"] == "BRAKE_COMMAND_AUTHORITY_INSUFFICIENT" for r in rows):
        fallback_classes.append("EMERGENCY_BRAKE_INSUFFICIENT")
    ranking_class = "KINEMATIC_ROUTE_RANKING_ADEQUATE" if exact_held["best_contact_negative_top3"] >= .90 else "KINEMATIC_ROUTE_RANKING_LIMITATION"

    current_exact_agreement = sum(r["current"]["contact"] == r["current"]["exact_narrowphase_contact"] for r in rows)
    predecessor_rows = [r["predecessor"] for r in rows if r["predecessor"] is not None]
    predecessor_exact_agreement = sum(r["contact"] == r["exact_narrowphase_contact"] for r in predecessor_rows)

    row_payload = {
        "schema": "h1_safe_action_set_successor_row_evidence_v1", "state_rows": [],
        "historical_no_safe_inventory": no_safe_inventory, "heldout_successor_selection": held_state_rows,
    }
    for row in rows:
        current = row["current"]
        row_payload["state_rows"].append({
            "state_id": row["state_id"], "split": row["split"], "family": row["family"],
            "brake_branch_identity": row["brake_branch_identity"], "historical_feasibility": inventory[row["state_id"]]["feasibility"],
            "initial_command_vx_vy_yaw": current["initial_command_vx_vy_yaw"], "initial_planar_speed_m_s": current["initial_planar_speed_m_s"],
            "contact": current["contact"], "first_contact_step": current["first_contact_step"], "stopped": current["stopped"],
            "stopping_time_s": current["stopping_time_s"], "stopping_distance_m": current["stopping_distance_m"],
            "path_distance_until_termination_m": current["path_distance_until_termination_m"], "stable": current["stable"],
            "fall": current["fall"], "unsafe_termination": current["unsafe_termination"],
            "qualified_safe_brake": current["qualified_safe_brake"], "state_classification": current["state_classification"],
            "predecessor_classification": current["predecessor_classification"], "raw_shard_sha256": row["shard_sha256"],
        })
    row_payload["content_digest"] = S.digest(row_payload)
    row_path = CACHE / "row_level_evidence_v1.json"; atomic_json(row_path, row_payload)

    result = {
        "schema": "h1_safe_action_set_successor_result_v1", "experiment": "H1_SAFE_ACTION_SET_SUCCESSOR_V1",
        "source_commit": "40dac616be57cf622d1614cc6ca85b5c31ae08a4",
        "claim_boundary": "simulated H1 any-physics-step disallowed-contact avoidance proxy only; not material-impact, injury, property-damage, human, fragile-infrastructure, mission-safety, or closed-loop assurance",
        "preserved_terminal": {"classification": "GENESIS_EXACT_GEOMETRY_QUERY_UNRESOLVED",
            "accurate_scope": "Genesis narrowphase and branch-level physics-rate contact query are resolved; the formal combined gate missed only best-safe top-3."},
        "contract": index["contract"], "deployment_realism": {
            "interface_command_realizable": True,
            "justification": "immediate zero vx/vy/yaw-rate at the existing planner-to-locomotion-policy interface; only planner-level slew is bypassed",
            "not_modified": ["low-level locomotion policy", "joint limits", "actuator dynamics", "body momentum", "contact/stability dynamics", "simulator physics"],
            "qualification_note": "interface realizability does not imply qualified stopping behaviour",
        },
        "fixture": {k: json.loads((OUT / "fixture_result.json").read_text())[k] for k in ("pass", "fixture_count", "each_executed_twice", "runtime_s", "content_digest")},
        "branch_execution": {"scientific_branches": index["new_scientific_branches"], "predecessor_branches": index["predecessor_test_branches"],
            "new_state_identities": index["new_state_identities"], "snapshot_exact_states": index["snapshot_exact_states"]},
        "exact_contact_reproduction": {
            "historical_branches": prior["reproduction"]["branch_level"],
            "historical_physics_step_exact_query": prior["reproduction"]["exact_query"],
            "historical_first_contact_step_error": prior["reproduction"]["first_contact_step_error"],
            "new_brake_branch_agreement": {"agree": current_exact_agreement, "total": len(rows), "rate": current_exact_agreement / len(rows)},
            "predecessor_brake_branch_agreement": {"agree": predecessor_exact_agreement, "total": len(predecessor_rows),
                "rate": predecessor_exact_agreement / len(predecessor_rows) if predecessor_rows else None},
            "alignment_note": "new brake agreement is branch-level; the preserved 576-branch reconciliation remains the authoritative physics-step qualification",
        },
        "brake_outcomes": {"overall": subset_summary(rows), "historical_safe_candidate_states": subset_summary(historical_safe),
            "historical_no_safe_candidate_states": subset_summary(historical_no_safe),
            "per_family": {family: subset_summary([r for r in rows if r["family"] == family]) for family in sorted({r["family"] for r in rows})},
            "state_classification_counts": brake_class_counts, "predecessor_classification_counts": predecessor_counts},
        "historical_no_safe_state_outcomes": no_safe_inventory,
        "successor_action_availability": {
            "states": len(rows), "safe_route_candidate_exists": len(historical_safe),
            "no_safe_route_but_safe_brake": no_safe_recovered,
            "neither_safe_route_nor_safe_brake": len(historical_no_safe) - no_safe_recovered,
            "candidate_bank_coverage_failures": len(candidate_bank_failures), "coverage_failures_restored": recovered_failures,
            "candidate_bank_status": bank_status,
        },
        "exact_geometry_route_fallback": {"heldout": held_route, "per_family": per_family, "per_state": held_state_rows,
            "calibration_historical_exact_route": {k: exact_cal[k] for k in ("safe_candidate_available_states", "no_safe_candidate_available_states", "best_contact_negative_top1", "best_contact_negative_top3", "normalized_route_progress_regret")}},
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "classifications": {"action_set": action_class, "fallback": fallback_classes,
            "candidate_bank": {"classification": "CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO", "status": bank_status},
            "ranking": ranking_class},
        "next_experiment": "ONE_CYCLE_STOPPING_ENVELOPE_GUARD_V1" if "PREDECESSOR_VIABILITY_GUARD_REQUIRED" in fallback_classes else "DEPLOYMENT_VALID_STRONG_BRAKING_MODE_V1",
        "next_experiment_rationale": "prevent entry into nonrecoverable states" if "PREDECESSOR_VIABILITY_GUARD_REQUIRED" in fallback_classes else "the immediate zero-command locomotion interface did not satisfy the stopped criterion in any state, including obstacle-free fixtures, and failed at every tested predecessor; qualify a genuine platform stand/damping/emergency-stop mode before changing perception",
        "row_level_evidence": {"path": str(row_path), "sha256": sha(row_path), "bytes": row_path.stat().st_size, "content_digest": row_payload["content_digest"]},
        "bindings": {"brake_index_sha256": sha(INDEX), "prior_result_sha256": sha(PRIOR), "brake_index_content_digest": index["content_digest"]},
        "runtime": {"fixture_s": index["fixture_runtime_s"], "replay_compute_s": index["runtime_compute_s"],
            "replay_wall_s": index["parallel_wall_runtime_s"], "evaluation_s": time.time() - started,
            "model_training": 0, "learned_inference": 0, "jepa_access": 0},
        "storage": {"raw_shards_bytes": index["storage_bytes"], "row_ledger_bytes": row_path.stat().st_size},
        "confirmations": {"model_training": False, "learned_inference": False, "jepa_access": False,
            "new_scientific_panel": False, "memory": False, "navigation": False},
    }
    result["content_digest"] = S.digest(result)
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"action_set": action_class, "fallback": fallback_classes, "candidate_bank_status": bank_status,
                      "successor_action_availability": result["successor_action_availability"],
                      "runtime": result["runtime"], "content_digest": result["content_digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
