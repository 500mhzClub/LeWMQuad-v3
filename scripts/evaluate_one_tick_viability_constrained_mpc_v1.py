#!/usr/bin/env python3
"""Reduce the frozen one-tick viability tree and compare oracle decisions."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import one_tick_viability_constrained_mpc_v1 as V
from scripts import evaluate_control_commitment_horizon_and_viability_v1 as PRIOR


SOURCE_COMMIT = "481253b5a504b0cd9fd05b14f5ad662b496fa0a8"
OUT = ROOT / ".generated/one_tick_viability_constrained_mpc_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/one_tick_viability_constrained_mpc_v1")
TREE = OUT / "viability_tree_index.json"
WIDE_STATES = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/states"
STAGE1 = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
PREVIOUS = ROOT / ".generated/control_commitment_horizon_and_viability_v1/result.json"
LATENCY = OUT / "latency_benchmark.json"
FAMILIES = (
    "large_enclosed_maze", "medium_enclosed_maze",
    "small_enclosed_maze", "loop_alias_stress",
)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _stage1_map() -> dict[tuple[str, int], dict]:
    with np.load(STAGE1, allow_pickle=False) as loaded:
        result = {}
        for index in range(len(loaded["state_id"])):
            state_id = str(loaded["state_id"][index])
            candidate = int(loaded["candidate_index"][index])
            if state_id.startswith("wide-"):
                result[(state_id, candidate)] = {
                    "p_d": float(loaded["p_d"][index]),
                    "p_theta": float(loaded["p_theta"][index]),
                    "kinematic_d": float(loaded["kinematic"][index, 4]),
                    "kinematic_theta": float(loaded["kinematic"][index, 5]),
                    "stuck": bool(loaded["stuck_labels"][index, -1, 1]),
                }
    return result


def load_states() -> tuple[list[dict], dict]:
    tree = json.loads(TREE.read_text())
    old_states, old_bindings = PRIOR.load_evidence()
    old_map = {row["state_id"]: row for row in old_states}
    h3 = _stage1_map()
    states = []
    for record in tree["state_records"]:
        state_id = record["state_id"]
        old = old_map[state_id]
        old_candidates = {row["candidate_index"]: row for row in old["branches"]}
        wide = json.loads((WIDE_STATES / f"{state_id}.json").read_text())
        wide_candidates = {int(row["candidate_index"]): row for row in wide["branches"]}
        rows = []
        for current in record["current"]:
            candidate = int(current["candidate_index"])
            prior = old_candidates[candidate]
            one = prior["horizons"][1]
            long = h3[(state_id, candidate)]
            frozen = wide_candidates[candidate]
            rows.append({
                "candidate_index": candidate,
                "candidate": current["candidate"],
                "safe_prefix": bool(current["safe_prefix"]),
                "viable": bool(current["viable"]),
                "admissible": bool(current["safe_prefix"] and current["viable"]),
                "successor_safe_candidate_count": int(current["successor_safe_candidate_count"]),
                "successor_safe_candidate_indices": current["successor_safe_candidate_indices"],
                "first_contact_step": current["first_contact_step"],
                "immediate_displacement_m": float(one["realised_displacement_m"]),
                "immediate_progress_m": float(one["realised_progress_m"]),
                "immediate_heading_improvement_rad": float(one["realised_heading_improvement_rad"]),
                "one_tick_nominal_progress_m": float(one["nominal_progress_m"]),
                "one_tick_nominal_heading_improvement_rad": float(one["nominal_heading_improvement_rad"]),
                "h3_nominal_progress_m": long["kinematic_d"],
                "h3_nominal_heading_improvement_rad": long["kinematic_theta"],
                "h3_realised_progress_m": long["p_d"],
                "h3_realised_heading_improvement_rad": long["p_theta"],
                "later_h1_contact": bool(one["later_physics_contact_before_h1_end"]),
                "later_h2_h3_contact": bool(one["later_sampled_contact_h2_h3"]),
                "stuck": long["stuck"],
                "completed": bool(frozen["completed"]),
            })
        if len(rows) != 12:
            raise RuntimeError(f"{state_id}: row count")
        prior_state = next(
            row for row in json.loads(PREVIOUS.read_text())["horizon_results"]["combined"]["1"]["per_state"]
            if row["state_id"] == state_id
        )
        classification = V.state_classification(
            rows,
            pre_existing=bool(old["boundary_contact"]),
            contact_before_authority=prior_state["availability_classification"] == "CONTACT_PRECEDES_CANDIDATE_DIVERGENCE",
        )
        states.append({
            "state_id": state_id, "split": record["split"], "family": record["family"],
            "pre_existing_contact": bool(old["boundary_contact"]),
            "classification": classification, "rows": rows,
        })
    return states, {"tree": tree, "prior": old_bindings}


def select(rows: list[dict], condition: str) -> tuple[int | None, list[int]]:
    safe = [index for index, row in enumerate(rows) if row["safe_prefix"]]
    viable = [index for index, row in enumerate(rows) if row["admissible"]]
    if condition == "A_ONE_TICK_SAFE_ONE_TICK_RANK":
        ids, dk, hk = safe, "one_tick_nominal_progress_m", "one_tick_nominal_heading_improvement_rad"
    elif condition == "B_ONE_TICK_SAFE_H3_RANK":
        ids, dk, hk = safe, "h3_nominal_progress_m", "h3_nominal_heading_improvement_rad"
    elif condition == "C_VIABILITY_ADMISSIBLE_H3_RANK":
        ids, dk, hk = viable, "h3_nominal_progress_m", "h3_nominal_heading_improvement_rad"
    elif condition == "D_VIABILITY_ADMISSIBLE_ORACLE_ROUTE":
        ids, dk, hk = viable, "h3_realised_progress_m", "h3_realised_heading_improvement_rad"
    else:
        raise ValueError(condition)
    subset = [rows[index] for index in ids]
    ordering = [ids[index] for index in V.route_order(subset, dk, hk)] if subset else []
    return (ordering[0] if ordering else None), ordering


def evaluate(states: list[dict], condition: str, include_rows: bool = True) -> dict:
    per_state = []
    regrets, top1, top3 = [], [], []
    for state in states:
        rows = state["rows"]
        pick, order = select(rows, condition)
        viable = [index for index, row in enumerate(rows) if row["admissible"]]
        best = None
        if viable:
            best, _ = select(rows, "D_VIABILITY_ADMISSIBLE_ORACLE_ROUTE")
            top1.append(pick == best)
            top3.append(best in order[:3])
            if pick is not None and len(viable) >= 2:
                values = [rows[index]["h3_realised_progress_m"] for index in viable]
                spread = max(values) - min(values)
                if spread > 1e-8:
                    regrets.append((rows[best]["h3_realised_progress_m"] - rows[pick]["h3_realised_progress_m"]) / spread)
        selected = None if pick is None else rows[pick]
        per_state.append({
            "state_id": state["state_id"], "split": state["split"], "family": state["family"],
            "state_classification": state["classification"],
            "safe_prefix_candidates": sum(row["safe_prefix"] for row in rows),
            "viability_admissible_candidates": len(viable),
            "selected_candidate": None if selected is None else selected["candidate_index"],
            "selected_candidate_name": None if selected is None else selected["candidate"],
            "selected_first_tick_contact": None if selected is None else not selected["safe_prefix"],
            "selected_successor_viable": None if selected is None else selected["viable"],
            "selected_successor_safe_actions": None if selected is None else selected["successor_safe_candidate_count"],
            "immediate_progress_m": None if selected is None else selected["immediate_progress_m"],
            "immediate_heading_improvement_rad": None if selected is None else selected["immediate_heading_improvement_rad"],
            "h3_nominal_progress_m": None if selected is None else selected["h3_nominal_progress_m"],
            "h3_nominal_heading_improvement_rad": None if selected is None else selected["h3_nominal_heading_improvement_rad"],
            "h3_realised_progress_m": None if selected is None else selected["h3_realised_progress_m"],
            "later_h1_contact": None if selected is None else selected["later_h1_contact"],
            "later_h2_h3_contact": None if selected is None else selected["later_h2_h3_contact"],
            "stuck": None if selected is None else selected["stuck"],
            "completed": None if selected is None else selected["completed"],
            "oracle_best_candidate": None if best is None else rows[best]["candidate_index"],
            "oracle_best_h3_realised_progress_m": None if best is None else rows[best]["h3_realised_progress_m"],
        })
    selected = [row for row in per_state if row["selected_candidate"] is not None]
    viable_states = [row for row in per_state if row["viability_admissible_candidates"]]
    oracle_values = [row["oracle_best_h3_realised_progress_m"] for row in per_state if row["oracle_best_h3_realised_progress_m"] is not None]
    result = {
        "condition": condition,
        "states": len(states),
        "selected_first_tick_contacts": sum(bool(row["selected_first_tick_contact"]) for row in selected),
        "first_tick_contact_negative_candidates": sum(sum(r["safe_prefix"] for r in state["rows"]) for state in states),
        "first_tick_contact_negative_retention": mean(sum(r["safe_prefix"] for r in state["rows"]) for state in states) / 12.0,
        "states_retaining_candidate": len(selected), "abstentions": len(states) - len(selected),
        "states_with_viability_admissible_candidate": len(viable_states),
        "selected_viable_successors": sum(bool(row["selected_successor_viable"]) for row in selected),
        "selected_nonviable_successors": sum(row["selected_successor_viable"] is False for row in selected),
        "mean_selected_successor_safe_actions": mean(row["selected_successor_safe_actions"] for row in selected),
        "mean_immediate_progress_m": mean(row["immediate_progress_m"] for row in selected),
        "mean_immediate_progress_rate_m_s": mean(row["immediate_progress_m"] for row in selected) / 0.1,
        "reverse_progress_rate": mean(row["immediate_progress_m"] < 0 for row in selected),
        "mean_immediate_heading_improvement_rad": mean(row["immediate_heading_improvement_rad"] for row in selected),
        "mean_h3_nominal_route_score_m": mean(row["h3_nominal_progress_m"] for row in selected),
        "mean_h3_realised_progress_m": mean(row["h3_realised_progress_m"] for row in selected),
        "oracle_viability_admissible_progress_m": mean(oracle_values),
        "oracle_progress_fraction": None if not oracle_values or abs(mean(oracle_values)) <= 1e-12 else mean(row["h3_realised_progress_m"] for row in selected) / mean(oracle_values),
        "normalized_viability_route_regret": None if not regrets else mean(regrets),
        "best_viability_admissible_top1": None if not top1 else mean(top1),
        "best_viability_admissible_top3": None if not top3 else mean(top3),
        "selected_later_h1_contact": sum(bool(row["later_h1_contact"]) for row in selected),
        "selected_later_h2_h3_contact": sum(bool(row["later_h2_h3_contact"]) for row in selected),
        "selected_stuck": sum(bool(row["stuck"]) for row in selected),
        "selected_completed": sum(bool(row["completed"]) for row in selected),
        "per_state": per_state if include_rows else None,
    }
    return result


def predecessor_result(bindings: dict) -> dict:
    prior = json.loads(PREVIOUS.read_text())
    prior_states = {
        row["state_id"]: row
        for row in prior["horizon_results"]["combined"]["1"]["per_state"]
        if row["contact_negative_candidates"] == 0
    }
    rows = []
    for record in bindings["tree"]["predecessor_records"]:
        current = record["current"]
        admissible = [row for row in current if row["safe_prefix"] and row["viable"]]
        safe = [row for row in current if row["safe_prefix"]]
        if admissible:
            classification = "ONE_TICK_EARLIER_VIABILITY_INTERVENTION_AVAILABLE"
        elif safe:
            classification = "MULTI_CYCLE_VIABILITY_ENVELOPE_REQUIRED"
        elif prior_states[record["original_state_id"]]["availability_classification"] == "CONTACT_PRECEDES_CANDIDATE_DIVERGENCE":
            classification = "CONTACT_ALREADY_UNAVOIDABLE_AT_PREDECESSOR"
        else:
            classification = "CANDIDATE_BANK_VIABILITY_COVERAGE_FAILURE"
        rows.append({
            "state_id": record["original_state_id"], "family": record["family"],
            "predecessor_safe_prefixes": len(safe),
            "predecessor_viability_admissible_prefixes": len(admissible),
            "classification": classification,
        })
    return {"states": len(rows), "classification_counts": dict(Counter(row["classification"] for row in rows)), "per_state": rows}


def build_ledger(states: list[dict], bindings: dict, conditions: dict) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "row_level_evidence_v1.jsonl"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    selected = {
        (name, row["state_id"]): row["selected_candidate"]
        for name, metrics in conditions.items() for row in metrics["per_state"]
    }
    row_count = 0
    with temporary.open("w") as stream:
        for state in states:
            for row in state["rows"]:
                payload = {"row_type": "current", "state_id": state["state_id"], "split": state["split"], "family": state["family"], **row,
                           "selected_by": [name for name in conditions if selected[(name, state["state_id"])] == row["candidate_index"]]}
                stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"); row_count += 1
        for record in bindings["tree"]["state_records"]:
            for row in record["successors"]:
                stream.write(json.dumps({"row_type": "successor", "state_id": record["state_id"], "split": record["split"], "family": record["family"], **row}, sort_keys=True, allow_nan=False) + "\n"); row_count += 1
        for record in bindings["tree"]["predecessor_records"]:
            for row in record["current"]:
                stream.write(json.dumps({"row_type": "predecessor_current", "state_id": record["original_state_id"], "split": record["split"], "family": record["family"], **row}, sort_keys=True, allow_nan=False) + "\n"); row_count += 1
            for row in record["successors"]:
                stream.write(json.dumps({"row_type": "predecessor_successor", "state_id": record["original_state_id"], "split": record["split"], "family": record["family"], **row}, sort_keys=True, allow_nan=False) + "\n"); row_count += 1
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "rows": row_count}


def main() -> int:
    started = time.time()
    states, bindings = load_states()
    conditions = {}
    for name in (
        "A_ONE_TICK_SAFE_ONE_TICK_RANK", "B_ONE_TICK_SAFE_H3_RANK",
        "C_VIABILITY_ADMISSIBLE_H3_RANK", "D_VIABILITY_ADMISSIBLE_ORACLE_ROUTE",
    ):
        metric = evaluate(states, name)
        metric["per_family"] = {family: evaluate([state for state in states if state["family"] == family], name, False) for family in FAMILIES}
        conditions[name] = metric
    current = conditions["C_VIABILITY_ADMISSIBLE_H3_RANK"]
    comparator = conditions["B_ONE_TICK_SAFE_H3_RANK"]
    non_preexisting = sum(not state["pre_existing_contact"] for state in states)
    family_ok = all(current["per_family"][family]["states_with_viability_admissible_candidate"] > 0 for family in FAMILIES)
    progress_not_regressed = current["mean_immediate_progress_rate_m_s"] >= 0.9 * comparator["mean_immediate_progress_rate_m_s"]
    gate = {
        "checks": {
            "zero_selected_first_tick_contacts": current["selected_first_tick_contacts"] == 0,
            "viability_availability_ge_0_95": current["states_with_viability_admissible_candidate"] / non_preexisting >= .95,
            "no_family_viability_collapse": family_ok,
            "all_selected_successors_viable": current["selected_nonviable_successors"] == 0,
            "oracle_progress_fraction_ge_0_80": current["oracle_progress_fraction"] is not None and current["oracle_progress_fraction"] >= .80,
            "normalized_regret_le_0_20": current["normalized_viability_route_regret"] is not None and current["normalized_viability_route_regret"] <= .20,
            "best_top3_ge_0_75": current["best_viability_admissible_top3"] is not None and current["best_viability_admissible_top3"] >= .75,
            "immediate_progress_not_materially_regressed": progress_not_regressed,
        }
    }
    gate["passed"] = all(gate["checks"].values())
    classifications = Counter(state["classification"] for state in states)
    candidate_bank = "CANDIDATE_BANK_ONE_TICK_VIABILITY_ADEQUATE" if current["states_with_viability_admissible_candidate"] / non_preexisting >= .95 else "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO"
    predecessor = predecessor_result(bindings)
    all_avoidable_recovered = predecessor["states"] > 0 and all(row["classification"] == "ONE_TICK_EARLIER_VIABILITY_INTERVENTION_AVAILABLE" for row in predecessor["per_state"])
    primary = "ONE_TICK_VIABILITY_SCAFFOLD_SIGNAL" if gate["passed"] else ("PREDECESSOR_VIABILITY_ENVELOPE_SIGNAL" if all_avoidable_recovered else "ONE_TICK_VIABILITY_KERNEL_NO_GO")
    ledger = build_ledger(states, bindings, conditions)
    latency = json.loads(LATENCY.read_text()) if LATENCY.is_file() else None
    if gate["passed"] and latency and latency["loop_rate_classification"] == "ONE_TICK_REPLANNING_COMPUTE_SIGNAL":
        experiment_primary = "ONE_TICK_VIABILITY_AND_REPLANNING_SIGNAL"
    elif gate["passed"]:
        experiment_primary = "VIABILITY_SIGNAL_REPLANNING_INTERFACE_BLOCKER"
    else:
        experiment_primary = primary
    result = {
        "schema": "one_tick_viability_constrained_mpc_result_v1",
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": "oracle simulated physics-rate contact avoidance and successor viability; no learned safety or closed-loop navigation claim",
        "replanning_interface": PRIOR.interface_contract(),
        "bindings": {
            "tree": {"path": str(TREE), "sha256": sha(TREE), "content_digest": bindings["tree"]["content_digest"]},
            "stage1_ledger": {"path": str(STAGE1), "sha256": sha(STAGE1)},
            "prior_exact": bindings["prior"]["exact_reproduction"],
        },
        "fixture": bindings["tree"]["fixture"],
        "generation": {key: bindings["tree"][key] for key in ("states", "historical_current_prefixes_replayed_as_one_tick", "new_successor_branches", "predecessor_states", "predecessor_prefix_branches", "predecessor_successor_branches", "runtime", "storage_bytes")},
        "current_state_classification_counts": dict(classifications),
        "current_state_viability": {
            "states": len(states), "non_pre_existing_states": non_preexisting,
            "contact_free_prefix_candidates": sum(sum(row["safe_prefix"] for row in state["rows"]) for state in states),
            "viability_admissible_candidates": sum(sum(row["admissible"] for row in state["rows"]) for state in states),
            "safe_prefix_nonviable_candidates": sum(sum(row["safe_prefix"] and not row["viable"] for row in state["rows"]) for state in states),
            "states_with_viability_admissible_action": current["states_with_viability_admissible_candidate"],
            "family": {family: current["per_family"][family]["states_with_viability_admissible_candidate"] for family in FAMILIES},
            "per_state": [{
                "state_id": state["state_id"], "split": state["split"], "family": state["family"],
                "classification": state["classification"],
                "contact_free_candidate_indices": [row["candidate_index"] for row in state["rows"] if row["safe_prefix"]],
                "viability_admissible_candidate_indices": [row["candidate_index"] for row in state["rows"] if row["admissible"]],
                "safe_now_nonviable_candidate_indices": [row["candidate_index"] for row in state["rows"] if row["safe_prefix"] and not row["viable"]],
                "hold_viable": any(row["candidate"] == "hold_all" and row["admissible"] for row in state["rows"]),
                "reverse_viable": any(row["candidate"] == "reverse_then_turn" and row["admissible"] for row in state["rows"]),
                "turn_viable": any(("turn" in row["candidate"] or "arc" in row["candidate"]) and row["admissible"] for row in state["rows"]),
            } for state in states],
        },
        "conditions": conditions,
        "viability_gate": gate,
        "predecessor_audit": predecessor,
        "candidate_bank_classification": candidate_bank,
        "viability_classification": primary,
        "latency_benchmark": latency,
        "primary_experiment_classification": experiment_primary,
        "next_experiment": {
            "name": "MULTI_CYCLE_VIABILITY_ENVELOPE_AND_STATE_ELIGIBILITY_V1",
            "reason": "prevent admission to states whose frozen action bank lacks a two-step viability-admissible response; do not resume learned planning yet",
            "implemented": False,
        },
        "row_level_ledger": ledger,
        "runtime_s": time.time() - started,
    }
    result["content_digest"] = V.digest(result)
    atomic_json(OUT / "viability_result.json", result)
    print(json.dumps({
        "generation": result["generation"], "current_state_viability": result["current_state_viability"],
        "conditions": {key: {k: v for k, v in value.items() if k != "per_state" and k != "per_family"} for key, value in conditions.items()},
        "gate": gate, "predecessor": predecessor, "candidate_bank": candidate_bank,
        "classification": primary, "ledger": ledger, "content_digest": result["content_digest"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
