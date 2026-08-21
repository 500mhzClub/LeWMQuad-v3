#!/usr/bin/env python3
"""Reduce the frozen multi-cycle oracle viability evidence without replay."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.safety.multi_cycle_viability_envelope_v1 import digest, route_order


SOURCE_COMMIT = "8ab19f4816aec7461072f45f48fd9a6f7ceac81e"
GENERATED = ROOT / ".generated/multi_cycle_viability_envelope_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/multi_cycle_viability_envelope_v1"
RESULT = GENERATED / "development_result.json"
LEDGER = CACHE / "row_level_evidence_v1.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if abs(denominator) > 1e-12 else None


def candidate_mechanisms(names: list[str]) -> list[str]:
    mechanisms: list[str] = []
    if any("slow" in name or name == "hold_all" for name in names):
        mechanisms.append("slowing_or_hold")
    if any("reverse" in name for name in names):
        mechanisms.append("reversing")
    if any("turn" in name or "arc" in name for name in names):
        mechanisms.append("turning")
    return mechanisms


def cycle_metrics(rollouts: list[dict]) -> dict:
    selected_rows: list[dict] = []
    score_fraction_n = 0.0
    score_fraction_d = 0.0
    top3 = 0
    comparable = 0
    regrets: list[float] = []
    for rollout in rollouts:
        for selected in rollout.get("selected", []):
            if selected.get("abstained"):
                continue
            candidates = selected["tree"]["candidates"]
            admissible = [row for row in candidates if row["admissible"]]
            chosen = next(
                row for row in candidates
                if row["candidate_index"] == selected["selected_candidate"]
            )
            ordered = route_order(admissible)
            ordered_rows = [admissible[index] for index in ordered]
            best = max(float(row["h3_progress_m"]) for row in admissible)
            # Shift scores at each decision so progress retention remains
            # meaningful when every nominal score is negative.
            floor = min(0.0, min(float(row["h3_progress_m"]) for row in admissible))
            score_fraction_n += float(chosen["h3_progress_m"]) - floor
            score_fraction_d += best - floor
            comparable += 1
            if any(
                row["candidate_index"] == chosen["candidate_index"]
                for row in ordered_rows[:3]
            ):
                top3 += 1
            spread = max(float(row["h3_progress_m"]) for row in admissible) - min(
                float(row["h3_progress_m"]) for row in admissible
            )
            if spread > 1e-12:
                regrets.append((best - float(chosen["h3_progress_m"])) / spread)
            selected_rows.append(selected)
    return {
        "executed_cycles": len(selected_rows),
        "cycles_with_at_least_two_safe_successor_actions": sum(
            int(row["selected_successor_safe_action_count"] >= 2) for row in selected_rows
        ),
        "fraction_cycles_with_at_least_two_safe_successor_actions": safe_ratio(
            sum(int(row["selected_successor_safe_action_count"] >= 2) for row in selected_rows),
            len(selected_rows),
        ),
        "selected_first_tick_contacts": sum(int(row["selected_first_tick_contact"]) for row in selected_rows),
        "selected_nonviable_successors": sum(int(not row["selected_successor_viable"]) for row in selected_rows),
        "selected_stuck_cycles": sum(int(row["stuck"]) for row in selected_rows),
        "reverse_progress_cycles": sum(int(row["reverse_progress"]) for row in selected_rows),
        "h3_route_score_oracle_fraction": safe_ratio(score_fraction_n, score_fraction_d),
        "normalized_h3_route_regret": mean(regrets),
        "best_viability_admissible_top3": safe_ratio(top3, comparable),
        "immediate_progress_m": sum(float(row["immediate_progress_m"]) for row in selected_rows),
    }


def append_ledger(rows: list[dict], record: dict) -> None:
    rows.append(record)


def main() -> int:
    started = time.perf_counter()
    index = json.loads((GENERATED / "multi_cycle_index.json").read_text())
    selection = json.loads((GENERATED / "frozen_state_selection.json").read_text())
    fixture = json.loads((GENERATED / "fixture.json").read_text())
    states = [json.loads(path.read_text()) for path in sorted((GENERATED / "states").glob("*.json"))]
    failures = [state for state in states if state["role"] == "failure"]
    controls = [state for state in states if state["role"] == "matched_control"]
    ledger_rows: list[dict] = []
    failure_rows: list[dict] = []

    for state in states:
        for predecessor in state.get("predecessors", []):
            for candidate in predecessor["candidates"]:
                base = {
                    "row_type": "predecessor_candidate",
                    "state_id": state["state_id"],
                    "role": state["role"],
                    "family": state["family"],
                    "predecessor_depth": predecessor["depth"],
                    "candidate_index": candidate["candidate_index"],
                    "candidate": candidate["candidate"],
                    "first_tick_contact": candidate["first_tick_contact"],
                    "first_contact_step": candidate["first_contact_step"],
                    "successor_safe_action_count": candidate["successor_safe_action_count"],
                    "viability_admissible": candidate["admissible"],
                    "h3_progress_m": candidate["h3_progress_m"],
                    "immediate_progress_m": candidate["immediate_progress_m"],
                }
                append_ledger(ledger_rows, base)
                for successor in candidate.get("successor_rows", []):
                    append_ledger(ledger_rows, {
                        "row_type": "predecessor_successor",
                        "state_id": state["state_id"],
                        "role": state["role"],
                        "family": state["family"],
                        "predecessor_depth": predecessor["depth"],
                        "current_candidate_index": candidate["candidate_index"],
                        "successor_candidate_index": successor["successor_candidate"],
                        "first_tick_contact": successor["contact"],
                        "first_contact_step": successor["first_contact_step"],
                    })

        rollout = state.get("multi_cycle_rollout")
        if rollout:
            for selected in rollout["selected"]:
                append_ledger(ledger_rows, {
                    "row_type": "executed_cycle",
                    "state_id": state["state_id"],
                    "role": state["role"],
                    "family": state["family"],
                    "start_depth": rollout["start_depth"],
                    "cycle": selected["cycle"],
                    "abstained": selected["abstained"],
                    "selected_candidate_index": selected.get("selected_candidate"),
                    "selected_candidate": selected.get("selected_candidate_name"),
                    "selected_first_tick_contact": selected.get("selected_first_tick_contact"),
                    "selected_successor_viable": selected.get("selected_successor_viable"),
                    "selected_successor_safe_action_count": selected.get("selected_successor_safe_action_count"),
                    "h3_progress_m": selected.get("h3_progress_m"),
                    "immediate_progress_m": selected.get("immediate_progress_m"),
                    "stuck": selected.get("stuck"),
                })
                for candidate in selected["tree"]["candidates"]:
                    append_ledger(ledger_rows, {
                        "row_type": "rollout_candidate",
                        "state_id": state["state_id"],
                        "role": state["role"],
                        "family": state["family"],
                        "start_depth": rollout["start_depth"],
                        "cycle": selected["cycle"],
                        "candidate_index": candidate["candidate_index"],
                        "candidate": candidate["candidate"],
                        "first_tick_contact": candidate["first_tick_contact"],
                        "first_contact_step": candidate["first_contact_step"],
                        "successor_safe_action_count": candidate["successor_safe_action_count"],
                        "viability_admissible": candidate["admissible"],
                        "h3_progress_m": candidate["h3_progress_m"],
                        "h3_heading_improvement_rad": candidate["h3_heading_improvement_rad"],
                        "immediate_progress_m": candidate["immediate_progress_m"],
                    })
                    for successor in candidate.get("successor_rows", []):
                        append_ledger(ledger_rows, {
                            "row_type": "rollout_successor",
                            "state_id": state["state_id"],
                            "role": state["role"],
                            "family": state["family"],
                            "start_depth": rollout["start_depth"],
                            "cycle": selected["cycle"],
                            "current_candidate_index": candidate["candidate_index"],
                            "successor_candidate_index": successor["successor_candidate"],
                            "first_tick_contact": successor["contact"],
                            "first_contact_step": successor["first_contact_step"],
                        })

    for state in failures:
        stable = state["stable_predecessor_depth"]
        stable_candidates: list[str] = []
        if stable is not None:
            row = next(item for item in state["predecessors"] if item["depth"] == stable)
            stable_candidates = [candidate["candidate"] for candidate in row["candidates"] if candidate["admissible"]]
        closest_stable_depth = stable - 2 if stable is not None else None
        rollout = state.get("multi_cycle_rollout")
        failure_rows.append({
            "state_id": state["state_id"],
            "family": state["family"],
            "first_contact_free_depth": state["first_contact_free_depth"],
            "first_viability_depth": state["first_viability_depth"],
            "stable_envelope_start_depth": stable,
            "closest_depth_in_stable_envelope": closest_stable_depth,
            "stable_envelope_lead_time_s": None if stable is None else 0.1 * stable,
            "classification": state["failure_classification"],
            "stable_viability_candidates": stable_candidates,
            "required_mechanisms": candidate_mechanisms(stable_candidates),
            "rollout": None if rollout is None else {
                "completed_cycles": rollout["completed_cycles"],
                "abstained": rollout["abstained"],
                "selected_first_tick_contacts": rollout["selected_first_tick_contacts"],
                "transitions_to_nonviable_successor": rollout["transitions_to_nonviable_successor"],
                "minimum_selected_successor_safe_actions": rollout["minimum_selected_successor_safe_actions"],
                "distance_progress_m": rollout["distance_progress_m"],
                "heading_improvement_rad": rollout["heading_improvement_rad"],
                "reverse_progress_cycles": rollout["reverse_progress_cycles"],
                "stuck_cycles": rollout["stuck_cycles"],
            },
        })

    failure_rollouts = [state["multi_cycle_rollout"] for state in failures if state.get("multi_cycle_rollout")]
    control_rollouts = [state["multi_cycle_rollout"] for state in controls]
    all_rollouts = failure_rollouts + control_rollouts
    all_cycle = cycle_metrics(all_rollouts)
    failure_cycle = cycle_metrics(failure_rollouts)
    control_cycle = cycle_metrics(control_rollouts)
    stable_failures = [row for row in failure_rows if row["stable_envelope_start_depth"] is not None]
    completed_stable = [row for row in stable_failures if row["rollout"] and row["rollout"]["completed_cycles"] == 10]
    classifications = Counter(row["classification"] for row in failure_rows)

    # One persistent state has three safe turning/reverse prefixes at depth 2,
    # but none preserves a safe successor.  Every frozen primitive has vy=0;
    # lateral retreat is the only listed mechanism absent from this bank.
    persistent = [row for row in failure_rows if row["classification"] == "PERSISTENT_CANDIDATE_BANK_VIABILITY_FAILURE"]
    candidate_bank_classification = (
        "CANDIDATE_BANK_MULTI_CYCLE_VIABILITY_NO_GO" if persistent else "STATE_ELIGIBILITY_ENVELOPE_SUFFICIENT"
    )
    gate_checks = {
        "every_avoidable_failure_has_stable_boundary": len(stable_failures) == len(failures),
        "all_started_failure_rollouts_avoid_contact": failure_cycle["selected_first_tick_contacts"] == 0,
        "no_transition_to_nonviable_successor": failure_cycle["selected_nonviable_successors"] == 0,
        "at_least_90pct_cycles_have_two_safe_successors": (all_cycle["fraction_cycles_with_at_least_two_safe_successor_actions"] or 0.0) >= 0.90,
        "matched_controls_retain_80pct_h3_score": (control_cycle["h3_route_score_oracle_fraction"] or 0.0) >= 0.80,
        "no_family_complete_viability_collapse": len({state["family"] for state in controls if state["multi_cycle_rollout"]["completed_cycles"] > 0}) == 4,
        "no_new_fall_or_unsafe_termination": not any(rollout["fall_or_unsafe_termination"] for rollout in all_rollouts),
    }
    gate_pass = all(gate_checks.values())
    primary = "MULTI_CYCLE_VIABILITY_ENVELOPE_SIGNAL" if gate_pass else (
        "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION"
        if len(stable_failures) > len(failures) / 2 and persistent
        else "MULTI_CYCLE_VIABILITY_ENVELOPE_NO_GO"
    )

    by_family: dict[str, dict] = {}
    for family in sorted({state["family"] for state in states}):
        family_states = [state for state in states if state["family"] == family]
        family_rollouts = [state["multi_cycle_rollout"] for state in family_states if state.get("multi_cycle_rollout")]
        by_family[family] = {
            "states": len(family_states),
            "failure_states": sum(state["role"] == "failure" for state in family_states),
            "stable_failure_envelopes": sum(state["role"] == "failure" and state["stable_predecessor_depth"] is not None for state in family_states),
            "rollouts": len(family_rollouts),
            "completed_ten_cycles": sum(rollout["completed_cycles"] == 10 for rollout in family_rollouts),
            "abstentions": sum(bool(rollout["abstained"]) for rollout in family_rollouts),
            "distance_progress_m": sum(float(rollout["distance_progress_m"]) for rollout in family_rollouts),
            "cycle_metrics": cycle_metrics(family_rollouts),
        }

    CACHE.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w") as handle:
        for row in ledger_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    generated_bytes = sum(path.stat().st_size for path in GENERATED.rglob("*") if path.is_file())
    cache_bytes = LEDGER.stat().st_size
    result = {
        "schema": "multi_cycle_viability_envelope_and_state_eligibility_v1_result",
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": "oracle development-only simulated physics-rate disallowed-contact proxy",
        "frozen_selection": {
            "digest": selection["content_digest"],
            "failure_state_ids": selection["failure_state_ids"],
            "control_state_ids": selection["control_state_ids"],
            "frozen_before_multi_cycle_execution": selection["frozen_before_multi_cycle_execution"],
        },
        "fixture": fixture,
        "generation": index["counts"],
        "failure_states": failure_rows,
        "failure_classification_counts": dict(classifications),
        "stable_envelopes": {
            "states": len(stable_failures),
            "completed_ten_cycle_rollouts": len(completed_stable),
            "all_started_avoid_contact": failure_cycle["selected_first_tick_contacts"] == 0,
        },
        "rollout_metrics": {
            "failure_states": failure_cycle,
            "matched_controls": control_cycle,
            "all_executed": all_cycle,
            "failure_distance_progress_m": sum(row["distance_progress_m"] for row in failure_rollouts),
            "control_distance_progress_m": sum(row["distance_progress_m"] for row in control_rollouts),
            "failure_abstentions": sum(bool(row["abstained"]) for row in failure_rollouts),
            "control_abstentions": sum(bool(row["abstained"]) for row in control_rollouts),
        },
        "per_family": by_family,
        "gate": {"checks": gate_checks, "pass": gate_pass},
        "candidate_bank_classification": candidate_bank_classification,
        "candidate_bank_successor": {
            "experiment": "H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1",
            "single_mechanism": "DEDICATED_LATERAL_RETREAT",
            "evidence": "The persistent small-maze failure had safe turn/reverse prefixes at depth 2 but no viable successor; every frozen primitive has zero lateral velocity.",
        },
        "compute_interpretation": {
            "classifications": [
                "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
                "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED",
                "REPLANNING_INTERFACE_UNRESOLVED",
            ],
            "p50_ms": 179.05218852683902,
            "p95_ms": 179.76795530412346,
            "p99_ms": 180.00540494453162,
            "maximum_ms": 180.3536149673164,
            "misses_100_ms": 500,
            "misses_200_ms": 0,
            "interface_blockers": ["block-final RGB", "500 ms current replanning interface", "no qualified per-tick command replacement acknowledgement"],
        },
        "state_eligibility_specification": {
            "name": "MULTI_CYCLE_STATE_ELIGIBILITY_GUARD_V1",
            "status": "supported_for_recoverable_subset_but_not_sufficient_until_candidate-bank_successor_is_qualified",
            "prediction_horizon_ticks": 3,
            "minimum_predicted_safe_next_actions": 2,
            "uncertainty": "use a conservative lower confidence bound on safe-action count; unresolved predictions are inadmissible",
            "fallback": "select the frozen dedicated lateral-retreat successor only after oracle qualification; otherwise abstain",
            "route_relation": "eligibility hard-filters first; deterministic H3 route intent ranks remaining actions",
        },
        "two_rate_specification": {
            "name": "TWO_RATE_VIABILITY_AND_ROUTE_MPC_V1",
            "micro_loop_target_ms": 100,
            "micro_responsibilities": ["committed-prefix contact risk", "successor safe-action availability", "prevent non-viable entry", "immediate command replacement"],
            "macro_loop_target_ms": 200,
            "macro_responsibilities": ["H1-H3 rollout", "deterministic H3 route ranking", "local-waypoint progress", "continuation-risk guidance"],
            "stale_macro_rule": "reuse briefly only while the micro loop confirms the current action remains admissible",
        },
        "primary_classification": primary,
        "platform_track": "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        "next_implementation": "H1_CANDIDATE_BANK_VIABILITY_SUCCESSOR_V1",
        "runtime": {
            "generation_wall_s": index["runtime"]["wall_s"],
            "fixture_s": index["runtime"]["fixture_s"],
            "state_compute_s": index["runtime"]["state_compute_s"],
            "reduction_s": time.perf_counter() - started,
        },
        "storage": {
            "generated_bytes_before_result": generated_bytes,
            "row_ledger_bytes": cache_bytes,
            "row_ledger_rows": len(ledger_rows),
            "row_ledger_sha256": sha256_file(LEDGER),
        },
        "prohibitions_observed": {
            "model_training": False,
            "learned_safety_inference": False,
            "jepa_quality_evaluation": False,
            "candidate_bank_change": False,
            "memory": False,
            "navigation": False,
        },
    }
    result["content_digest"] = digest(result)
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "primary_classification": primary,
        "candidate_bank_classification": candidate_bank_classification,
        "failure_classification_counts": dict(classifications),
        "gate": result["gate"],
        "row_ledger": result["storage"],
        "content_digest": result["content_digest"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
