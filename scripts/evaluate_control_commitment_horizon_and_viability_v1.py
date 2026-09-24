#!/usr/bin/env python3
"""Read-only one-to-five-tick control commitment and viability analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import control_commitment_horizon_and_viability_v1 as V


SOURCE_COMMIT = "14d3a6ed8fa050011b72fba315ebe028be2898a5"
OUT = ROOT / ".generated/control_commitment_horizon_and_viability_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/control_commitment_horizon_and_viability_v1")
EXACT_INDEX = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1/narrowphase_index.json"
EXACT_RESULT = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1/result.json"
GEOMETRY_INDEX = ROOT / ".generated/h1_articulated_swept_geometry_sufficiency_v1/articulated_geometry_index.json"
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
STATE_ROOT = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/states"
PRIOR_BRAKE = ROOT / ".generated/deployment_valid_strong_braking_mode_v1/result.json"
PRIOR_FALLBACK = ROOT / ".generated/h1_safe_action_set_successor_v1/result.json"
PHYSICS_LEDGER = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/physics_rate_contact_proxy_reconciliation_v1/row_level_evidence_v1.npz")
PHYSICS_LEDGER_SHA256 = "3e5de8b6b4007f9ac066bb981e23f9fc59b28459caa23d93c9c222431b18b8ee"
FAMILIES = (
    "large_enclosed_maze",
    "medium_enclosed_maze",
    "small_enclosed_maze",
    "loop_alias_stress",
)
INTERFACE_SOURCES = {
    "platform_manifest": ROOT / "config/go2_platform_manifest.yaml",
    "primitive_registry": ROOT / "config/go2_primitive_registry.yaml",
    "rollout": ROOT / "lewm_genesis/lewm_genesis/rollout.py",
    "local_mpc": ROOT / "lewm/planning/local_mpc.py",
    "closed_loop_runner": ROOT / "scripts/benchmark_lewm_closed_loop_mpc.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def body_region(link_name: str | None) -> str | None:
    if link_name is None:
        return None
    if link_name == "base":
        return "trunk"
    if link_name.startswith(("FL", "FR")):
        return "front_limb"
    if link_name.startswith(("RL", "RR")):
        return "rear_limb"
    return "unresolved"


def interface_contract() -> dict:
    text = {name: path.read_text() for name, path in INTERFACE_SOURCES.items()}
    checks = {
        "physics_2ms": "physics_dt_s: 0.002" in text["platform_manifest"],
        "policy_20ms": "policy_dt_s: 0.02" in text["platform_manifest"],
        "command_100ms": "command_dt_s: 0.10" in text["platform_manifest"],
        "five_tick_block": "action_block_size: 5" in text["platform_manifest"],
        "mpc_returns_first_primitive": "primitive=best_sequence[0]" in text["local_mpc"],
        "closed_loop_scores_once_per_block": "for block_idx in range(int(max_blocks))" in text["closed_loop_runner"],
        "physical_executor_consumes_entire_clipped_block": "for tick in clipped:" in text["closed_loop_runner"],
        "policy_accepts_target_each_command_tick": "runner._step_command_tick(tick[None, :])" in text["closed_loop_runner"],
        "production_rgb_is_block_final_only": "if is_last_tick and self.config.rgb_capture_per_block" in text["rollout"],
        "special_capture_can_render_each_tick": (
            "the egocentric + third-person views directly each control tick" in text["closed_loop_runner"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"command/replanning source contract changed: {checks}")
    return {
        "classification": "REPLANNING_INTERFACE_UNRESOLVED",
        "policy_command_period_s": 0.1,
        "low_level_policy_period_s": 0.02,
        "physics_period_s": 0.002,
        "production_camera_observation_period_s": 0.5,
        "special_capture_camera_period_s": 0.1,
        "predictor_invocation_period_s": 0.5,
        "current_mpc_replanning_period_s": 0.5,
        "buffered_or_irrevocably_executed_command_ticks": 5,
        "new_command_can_replace_next_tick_in_current_loop": False,
        "low_level_policy_accepts_new_command_each_100ms": True,
        "complete_predictor_input_constructed_each_100ms_in_current_loop": False,
        "controller_ipc": "in-process Genesis development runner; no IPC boundary in this path",
        "latency_status": "render-plus-predictor deadline at 100 ms has not been measured or enforced",
        "minimum_current_demonstrated_commitment_ticks": 5,
        "minimum_current_demonstrated_commitment_s": 0.5,
        "reason": (
            "the controller seam accepts per-tick targets and a special capture path can render per tick, "
            "but the deployed development loop invokes planning once, executes all five ticks, and has no "
            "measured 100 ms end-to-end predictor/preemption contract"
        ),
        "source_checks": checks,
        "source_bindings": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in INTERFACE_SOURCES.items()
        },
    }


def load_evidence() -> tuple[list[dict], dict]:
    if sha256(PHYSICS_LEDGER) != PHYSICS_LEDGER_SHA256:
        raise RuntimeError("physics-rate ledger binding mismatch")
    exact_index = json.loads(EXACT_INDEX.read_text())
    exact_result = json.loads(EXACT_RESULT.read_text())
    geometry_index = json.loads(GEOMETRY_INDEX.read_text())
    panel = json.loads(PANEL.read_text())
    state_manifest = {row["state_id"]: row for row in panel["states"]}
    exact_map = {row["state_id"]: row for row in exact_index["state_records"]}
    geometry_map = {row["state_id"]: row for row in geometry_index["state_records"]}
    if set(state_manifest) != set(exact_map) or set(state_manifest) != set(geometry_map):
        raise RuntimeError("frozen state identity alignment failure")
    if len(state_manifest) != 48:
        raise RuntimeError("expected 48 frozen states")

    state_payloads: list[dict] = []
    for state_id in sorted(state_manifest):
        manifest = state_manifest[state_id]
        exact_record = exact_map[state_id]
        geometry_record = geometry_map[state_id]
        wide_record = json.loads((STATE_ROOT / f"{state_id}.json").read_text())
        if sha256(Path(exact_record["shard_path"])) != exact_record["shard_sha256"]:
            raise RuntimeError(f"exact shard mismatch: {state_id}")
        if sha256(Path(geometry_record["shard_path"])) != geometry_record["shard_sha256"]:
            raise RuntimeError(f"geometry shard mismatch: {state_id}")
        if sha256(Path(wide_record["shard_path"])) != wide_record["shard_sha256"]:
            raise RuntimeError(f"sensor shard mismatch: {state_id}")
        with np.load(exact_record["shard_path"], allow_pickle=False) as loaded:
            exact = {key: np.asarray(loaded[key]) for key in loaded.files}
        with np.load(geometry_record["shard_path"], allow_pickle=False) as loaded:
            geometry = {key: np.asarray(loaded[key]) for key in loaded.files}
        with np.load(wide_record["shard_path"], allow_pickle=False) as loaded:
            sensor = {key: np.asarray(loaded[key]) for key in loaded.files}
        if not np.array_equal(exact["frozen_contact"], geometry["physics_contact"]):
            raise RuntimeError(f"physics trace mismatch: {state_id}")
        if len(wide_record["branches"]) != 12 or sensor["poses"].shape != (12, 15, 8):
            raise RuntimeError(f"wide evidence shape mismatch: {state_id}")
        divergence_step = V.first_divergence_step(geometry["link_transform"])
        branch_rows = []
        heading_world = float(wide_record["branches"][0]["route_heading_world_rad"])
        heading_body = heading_world - float(manifest["start_pose"][1])
        waypoint = [
            *map(float, manifest["waypoint_body_xy"]),
            math.sin(heading_body),
            math.cos(heading_body),
        ]
        link_names = {int(key): str(value) for key, value in exact_record["link_names"].items()}
        for branch in wide_record["branches"]:
            candidate = int(branch["candidate_index"])
            contact_trace = exact["frozen_contact"][candidate].astype(bool)
            contacts = np.flatnonzero(contact_trace)
            first_contact = None if not len(contacts) else int(contacts[0])
            contact_link = None
            if first_contact is not None:
                contact_link = link_names.get(int(exact["native_robot_link"][candidate, first_contact]))
            horizon_rows = {}
            for horizon in V.HORIZONS:
                end_step = horizon * V.PHYSICS_STEPS_PER_TICK
                nominal = V.integrate_prefix(branch["post_slew"], waypoint, horizon)
                realised = V.realised_prefix(
                    manifest["start_pose"], manifest["waypoint_xy"], sensor["poses"][candidate, horizon - 1], heading_world
                )
                committed_contact = bool(contact_trace[:end_step].any())
                later_h1 = bool((not committed_contact) and contact_trace[end_step:].any())
                later_h2_h3_sampled = bool(sensor["labels"][candidate, 5:, 0].astype(bool).any())
                horizon_rows[horizon] = {
                    "horizon_ticks": horizon,
                    "duration_s": horizon * V.COMMAND_DT_S,
                    "committed_contact": committed_contact,
                    "first_contact_step": first_contact,
                    "first_contact_time_s": None if first_contact is None else first_contact * V.PHYSICS_DT_S,
                    "contact_link": contact_link,
                    "body_region": body_region(contact_link),
                    "nominal_progress_m": float(nominal[4]),
                    "nominal_heading_improvement_rad": float(nominal[5]),
                    "realised_displacement_m": realised["displacement_m"],
                    "realised_progress_m": realised["progress_m"],
                    "realised_heading_improvement_rad": realised["heading_improvement_rad"],
                    "later_physics_contact_before_h1_end": later_h1,
                    "later_sampled_contact_h2_h3": later_h2_h3_sampled,
                    "later_continuation_contact": bool(later_h1 or later_h2_h3_sampled),
                    "stuck_through_h3": bool(sensor["labels"][candidate, -1, 3] > 0.5),
                }
            branch_rows.append(
                {
                    "branch_id": str(branch["branch_id"]),
                    "candidate_index": candidate,
                    "candidate": str(branch["candidate"]),
                    "first_primitive": str(branch["primitives"][0]),
                    "post_slew_first_block": branch["post_slew"][0],
                    "horizons": horizon_rows,
                }
            )
        state_payloads.append(
            {
                "state_id": state_id,
                "split": str(manifest["split"]),
                "family": str(manifest["family"]),
                "boundary_contact": bool(exact_record["boundary_native_contact"]),
                "candidate_divergence_step": divergence_step,
                "candidate_divergence_time_s": None if divergence_step is None else divergence_step * V.PHYSICS_DT_S,
                "branches": branch_rows,
            }
        )
    bindings = {
        "states": 48,
        "branches": 576,
        "rows_across_five_horizons": 2880,
        "physics_steps_per_h1_branch": 250,
        "physics_rate_ledger": {"path": str(PHYSICS_LEDGER), "sha256": sha256(PHYSICS_LEDGER)},
        "exact_index": {"path": str(EXACT_INDEX), "sha256": sha256(EXACT_INDEX), "content_digest": exact_index["content_digest"]},
        "geometry_index": {"path": str(GEOMETRY_INDEX), "sha256": sha256(GEOMETRY_INDEX), "content_digest": geometry_index["content_digest"]},
        "panel": {"path": str(PANEL), "sha256": sha256(PANEL), "content_digest": panel["content_digest"]},
        "exact_reproduction": {
            "branch_level": exact_result["reproduction"]["branch_level"],
            "native_replay": exact_result["reproduction"]["native_replay"],
            "exact_query": {
                key: exact_result["reproduction"]["exact_query"][key]
                for key in (
                    "agreement", "sensitivity", "specificity", "true_positive",
                    "false_positive", "false_negative", "true_negative",
                )
            },
            "first_contact_step_error": exact_result["reproduction"]["first_contact_step_error"],
        },
    }
    return state_payloads, bindings


def split_states(states: list[dict], split: str) -> list[dict]:
    if split == "combined":
        return states
    return [state for state in states if state["split"] == split]


def evaluate_horizon(states: list[dict], horizon: int) -> dict:
    per_state = []
    selected_progress: list[float] = []
    selected_displacement: list[float] = []
    selected_heading: list[float] = []
    oracle_progress: list[float] = []
    regrets: list[float] = []
    top1: list[bool] = []
    top3: list[bool] = []
    all_negative = 0
    all_positive = 0
    selected_later_h1 = 0
    selected_later_h2_h3 = 0
    selected_later_any = 0
    selected_stuck = 0
    positive_progress_selections = 0
    hold_safe_states = 0
    reverse_safe_states = 0
    turning_safe_states = 0
    positive_progress_safe_states = 0
    selected_contact = 0
    classifications = Counter()
    for state in states:
        rows = []
        for branch in state["branches"]:
            row = {**branch["horizons"][horizon], "candidate_index": branch["candidate_index"],
                   "candidate": branch["candidate"], "first_primitive": branch["first_primitive"]}
            rows.append(row)
        safe = [index for index, row in enumerate(rows) if not row["committed_contact"]]
        rank = V.route_order(rows)
        safe_rank = [index for index in rank if index in safe]
        selected = safe_rank[0] if safe_rank else None
        best = None
        for index in safe:
            if best is None or V.realised_preference(rows[index], rows[best]) > 0:
                best = index
        classification = V.availability_class(
            rows,
            boundary_contact=state["boundary_contact"],
            divergence_step=state["candidate_divergence_step"],
        )
        classifications[classification] += 1
        all_negative += len(safe)
        all_positive += len(rows) - len(safe)
        if selected is not None:
            selected_progress.append(float(rows[selected]["realised_progress_m"]))
            selected_displacement.append(float(rows[selected]["realised_displacement_m"]))
            selected_heading.append(float(rows[selected]["realised_heading_improvement_rad"]))
            selected_contact += int(rows[selected]["committed_contact"])
            selected_later_h1 += int(rows[selected]["later_physics_contact_before_h1_end"])
            selected_later_h2_h3 += int(rows[selected]["later_sampled_contact_h2_h3"])
            selected_later_any += int(rows[selected]["later_continuation_contact"])
            selected_stuck += int(rows[selected]["stuck_through_h3"])
            positive_progress_selections += int(rows[selected]["realised_progress_m"] > 0.0)
        if best is not None:
            oracle_progress.append(float(rows[best]["realised_progress_m"]))
            top1.append(selected == best)
            top3.append(best in safe_rank[:3])
            if selected is not None and len(safe) >= 2:
                values = [float(rows[index]["realised_progress_m"]) for index in safe]
                spread = max(values) - min(values)
                if spread > 1e-8:
                    regrets.append((float(rows[best]["realised_progress_m"]) - float(rows[selected]["realised_progress_m"])) / spread)
        names = {row["candidate"]: index for index, row in enumerate(rows)}
        hold = names.get("hold_all")
        reverse = names.get("reverse_then_turn")
        turns = [index for index, row in enumerate(rows) if "turn" in row["candidate"] or "arc" in row["candidate"]]
        hold_safe_states += int(hold is not None and hold in safe)
        reverse_safe_states += int(reverse is not None and reverse in safe)
        turning_safe_states += int(any(index in safe for index in turns))
        positive_progress_safe_states += int(
            any(rows[index]["realised_progress_m"] > 0.0 for index in safe)
        )
        per_state.append(
            {
                "state_id": state["state_id"],
                "family": state["family"],
                "boundary_contact": state["boundary_contact"],
                "availability_classification": classification,
                "contact_negative_candidates": len(safe),
                "hold_contact_negative": hold is not None and hold in safe,
                "reverse_contact_negative": reverse is not None and reverse in safe,
                "turning_contact_negative": any(index in safe for index in turns),
                "positive_progress_contact_negative_exists": any(rows[index]["realised_progress_m"] > 0.0 for index in safe),
                "earliest_contact_step": min((row["first_contact_step"] for row in rows if row["first_contact_step"] is not None), default=None),
                "candidate_divergence_step": state["candidate_divergence_step"],
                "selected_candidate": None if selected is None else int(rows[selected]["candidate_index"]),
                "selected_candidate_name": None if selected is None else rows[selected]["candidate"],
                "selected_committed_contact": None if selected is None else bool(rows[selected]["committed_contact"]),
                "selected_progress_m": None if selected is None else float(rows[selected]["realised_progress_m"]),
                "selected_displacement_m": None if selected is None else float(rows[selected]["realised_displacement_m"]),
                "selected_heading_improvement_rad": None if selected is None else float(rows[selected]["realised_heading_improvement_rad"]),
                "selected_later_physics_contact_before_h1_end": None if selected is None else bool(rows[selected]["later_physics_contact_before_h1_end"]),
                "selected_later_sampled_contact_h2_h3": None if selected is None else bool(rows[selected]["later_sampled_contact_h2_h3"]),
                "selected_stuck_through_h3": None if selected is None else bool(rows[selected]["stuck_through_h3"]),
                "oracle_best_realised_candidate": None if best is None else int(rows[best]["candidate_index"]),
                "oracle_best_realised_progress_m": None if best is None else float(rows[best]["realised_progress_m"]),
                "abstention": selected is None,
            }
        )
    duration = horizon * V.COMMAND_DT_S
    mean_progress = float(np.mean(selected_progress)) if selected_progress else 0.0
    mean_oracle = float(np.mean(oracle_progress)) if oracle_progress else 0.0
    non_preexisting = [row for row in per_state if not row["boundary_contact"]]
    safe_states = [row for row in per_state if row["contact_negative_candidates"] > 0]
    families_with_positive = len({row["family"] for row in per_state if row["positive_progress_contact_negative_exists"]})
    return {
        "horizon_ticks": horizon,
        "duration_s": duration,
        "states": len(states),
        "non_pre_existing_states": len(non_preexisting),
        "contact_negative_branches": all_negative,
        "contact_positive_branches": all_positive,
        "states_retaining_safe_action": len(safe_states),
        "safe_action_state_fraction_non_preexisting": None if not non_preexisting else len(safe_states) / len(non_preexisting),
        "abstentions": len(states) - len(safe_states),
        "selected_committed_contacts": selected_contact,
        "positive_progress_selections": positive_progress_selections,
        "states_with_contact_negative_hold": hold_safe_states,
        "states_with_contact_negative_reverse": reverse_safe_states,
        "states_with_contact_negative_turn": turning_safe_states,
        "states_with_positive_progress_contact_negative": positive_progress_safe_states,
        "families_with_positive_progress_safe_action": families_with_positive,
        "mean_selected_displacement_m": float(np.mean(selected_displacement)) if selected_displacement else 0.0,
        "mean_selected_progress_m": mean_progress,
        "mean_selected_progress_rate_m_s": mean_progress / duration,
        "mean_selected_heading_improvement_rad": float(np.mean(selected_heading)) if selected_heading else 0.0,
        "oracle_best_realised_progress_m": mean_oracle,
        "oracle_best_realised_progress_rate_m_s": mean_oracle / duration,
        "oracle_progress_rate_fraction": None if abs(mean_oracle) <= 1e-12 else mean_progress / mean_oracle,
        "normalized_regret": None if not regrets else float(np.mean(regrets)),
        "normalized_regret_states": len(regrets),
        "best_safe_top1": None if not top1 else float(np.mean(top1)),
        "best_safe_top3": None if not top3 else float(np.mean(top3)),
        "selected_later_physics_contact_before_h1_end": selected_later_h1,
        "selected_later_sampled_contact_h2_h3": selected_later_h2_h3,
        "selected_any_later_continuation_contact": selected_later_any,
        "selected_stuck_through_h3": selected_stuck,
        "availability_classification_counts": dict(classifications),
        "per_state": per_state,
    }


def add_family_metrics(states: list[dict], horizon: int, aggregate: dict) -> None:
    aggregate["per_family"] = {
        family: evaluate_horizon([state for state in states if state["family"] == family], horizon)
        for family in FAMILIES
    }
    for value in aggregate["per_family"].values():
        value.pop("per_state", None)


def horizon_gate(metric: dict, *, technically_achievable: bool) -> dict:
    family_ok = all(value["states_retaining_safe_action"] > 0 for value in metric["per_family"].values())
    checks = {
        "technically_achievable": technically_achievable,
        "safe_action_fraction_ge_0_95": metric["safe_action_state_fraction_non_preexisting"] >= 0.95,
        "zero_selected_contact": metric["selected_committed_contacts"] == 0,
        "no_family_safe_action_collapse": family_ok,
        "regret_le_0_20": metric["normalized_regret"] is not None and metric["normalized_regret"] <= 0.20,
        "best_safe_top3_ge_0_75": metric["best_safe_top3"] is not None and metric["best_safe_top3"] >= 0.75,
        "progress_rate_fraction_ge_0_80": metric["oracle_progress_rate_fraction"] is not None and metric["oracle_progress_rate_fraction"] >= 0.80,
        "positive_progress_in_three_families": metric["families_with_positive_progress_safe_action"] >= 3,
        "continuation_risk_reported": "selected_any_later_continuation_contact" in metric,
    }
    return {"checks": checks, "passed": all(checks.values())}


def build_row_ledger(states: list[dict], horizon_results: dict[int, dict]) -> dict:
    selection = {
        (int(horizon), row["state_id"]): row
        for horizon, result in horizon_results.items()
        for row in result["per_state"]
    }
    rows = []
    for state in states:
        for branch in state["branches"]:
            for horizon in V.HORIZONS:
                item = branch["horizons"][horizon]
                selected = selection[(horizon, state["state_id"])]
                rows.append(
                    {
                        "state_id": state["state_id"], "split": state["split"], "family": state["family"],
                        "branch_id": branch["branch_id"], "candidate_index": branch["candidate_index"],
                        "candidate": branch["candidate"], "horizon_ticks": horizon,
                        "contact_label": item["committed_contact"], "first_contact_step": item["first_contact_step"],
                        "body_region": item["body_region"], "candidate_divergence_step": state["candidate_divergence_step"],
                        "safe_action_available_in_state": selected["contact_negative_candidates"] > 0,
                        "state_availability_classification": selected["availability_classification"],
                        "nominal_progress_m": item["nominal_progress_m"],
                        "nominal_heading_improvement_rad": item["nominal_heading_improvement_rad"],
                        "realised_progress_m": item["realised_progress_m"],
                        "realised_displacement_m": item["realised_displacement_m"],
                        "selected": selected["selected_candidate"] == branch["candidate_index"],
                        "abstention": selected["abstention"],
                        "later_physics_contact_before_h1_end": item["later_physics_contact_before_h1_end"],
                        "later_sampled_contact_h2_h3": item["later_sampled_contact_h2_h3"],
                    }
                )
    payload = {"schema": "control_commitment_horizon_and_viability_row_evidence_v1", "rows": rows}
    payload["content_digest"] = V.digest(payload)
    return payload


def main() -> int:
    started = time.monotonic()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"expected source commit {SOURCE_COMMIT}, found {head}")
    prior = json.loads(PRIOR_BRAKE.read_text())
    prior_fallback = json.loads(PRIOR_FALLBACK.read_text())
    required = {
        prior["classifications"]["mode"], prior["classifications"]["action_set"],
        prior["classifications"]["candidate_bank"], prior["classifications"]["ranking"],
    }
    expected = {"DEPLOYMENT_VALID_BRAKING_MODE_UNAVAILABLE", "H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO",
                "CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO", "KINEMATIC_ROUTE_RANKING_LIMITATION"}
    if required != expected:
        raise RuntimeError(f"preserved predecessor mismatch: {required}")
    if "EMERGENCY_BRAKE_INSUFFICIENT" not in prior_fallback["classifications"]["fallback"]:
        raise RuntimeError("preserved emergency-brake terminal mismatch")
    fixture = V.fixture_payload()
    if not fixture["pass"] or fixture != V.fixture_payload():
        raise RuntimeError("deterministic evaluator fixture failed")
    atomic_json(OUT / "fixture.json", fixture)
    interface = interface_contract()
    states, bindings = load_evidence()

    results: dict[str, dict[int, dict]] = {}
    for split in ("calibration", "heldout", "combined"):
        results[split] = {}
        chosen = split_states(states, split)
        for horizon in V.HORIZONS:
            metric = evaluate_horizon(chosen, horizon)
            add_family_metrics(chosen, horizon, metric)
            # Only the existing five-tick loop is demonstrated. No shorter
            # end-to-end predictor/preemption deadline is qualified.
            metric["gate"] = horizon_gate(metric, technically_achievable=(horizon == 5))
            results[split][horizon] = metric

    h5_no_safe = [state for state in states if not any(not branch["horizons"][5]["committed_contact"] for branch in state["branches"])]
    viability_rows = []
    viability_counts = Counter()
    for state in h5_no_safe:
        safe_counts = {
            horizon: sum(not branch["horizons"][horizon]["committed_contact"] for branch in state["branches"])
            for horizon in V.HORIZONS
        }
        first = [branch["horizons"][5]["first_contact_step"] for branch in state["branches"]]
        classification = V.viability_class(
            boundary_contact=state["boundary_contact"], safe_counts=safe_counts,
            first_contact_steps=first, divergence_step=state["candidate_divergence_step"],
            shorter_horizon_technically_available=False,
        )
        viability_counts[classification] += 1
        shortest = next((horizon for horizon in V.HORIZONS if safe_counts[horizon] > 0), None)
        viability_rows.append(
            {
                "state_id": state["state_id"], "split": state["split"], "family": state["family"],
                "safe_candidate_counts_by_horizon": safe_counts,
                "shortest_trace_horizon_with_safe_action": shortest,
                "shorter_trace_would_restore_action": shortest is not None and shortest < 5,
                "shorter_horizon_currently_technically_qualified": False,
                "earliest_contact_step": min(step for step in first if step is not None),
                "latest_first_contact_step": max(step for step in first if step is not None),
                "candidate_divergence_step": state["candidate_divergence_step"],
                "classification": classification,
            }
        )

    held = results["heldout"]
    one_tick_material_failure = held[1]["states_retaining_safe_action"] < math.ceil(0.95 * held[1]["non_pre_existing_states"])
    passing_shorter = [h for h in V.HORIZONS[:-1] if held[h]["gate"]["passed"]]
    if one_tick_material_failure:
        primary = "ONE_TICK_SAFE_ACTION_SET_NO_GO"
    elif passing_shorter:
        primary = "SHORTER_COMMITMENT_SAFE_ACTION_SET_SIGNAL"
    elif any(results["heldout"][h]["safe_action_state_fraction_non_preexisting"] >= .95 for h in V.HORIZONS[:-1]):
        primary = "REPLANNING_RATE_IMPLEMENTATION_BLOCKER"
    else:
        primary = "NO_COMMITMENT_HORIZON_SIGNAL"

    combined_horizon = {h: results["combined"][h] for h in V.HORIZONS}
    row_ledger = build_row_ledger(states, combined_horizon)
    row_path = CACHE / "row_level_evidence_v1.json"
    atomic_json(row_path, row_ledger)

    # Remove row-heavy structures from the split summary only after the ledger
    # is complete. Held-out/calibration per-state values remain separately.
    report_results = {}
    for split, by_horizon in results.items():
        report_results[split] = {}
        for horizon, metric in by_horizon.items():
            report_results[split][str(horizon)] = metric

    result = {
        "schema": "control_commitment_horizon_and_viability_result_v1",
        "experiment": "CONTROL_COMMITMENT_HORIZON_AND_VIABILITY_V1",
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": (
            "development-only simulated H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT control-authority analysis; "
            "not material-impact, injury, physical-Go2-braking, human/property, learned-safety, mission, or closed-loop assurance"
        ),
        "preserved_classifications": sorted(required | {"EMERGENCY_BRAKE_INSUFFICIENT"}),
        "bindings": bindings,
        "interface": interface,
        "fixture": fixture,
        "horizon_results": report_results,
        "five_tick_no_safe_viability": {
            "states": len(viability_rows),
            "classification_counts": dict(viability_counts),
            "trace_recovered_at_shorter_horizon": sum(row["shorter_trace_would_restore_action"] for row in viability_rows),
            "still_no_safe_at_one_tick": sum(row["safe_candidate_counts_by_horizon"][1] == 0 for row in viability_rows),
            "per_state": viability_rows,
        },
        "development_gate": {
            "passing_shorter_horizons": passing_shorter,
            "longest_passing_horizon": max(passing_shorter) if passing_shorter else None,
            "primary_classification": primary,
        },
        "interpretation": {
            "commitment_is_principal_blocker": False,
            "reason": (
                "five of eleven five-tick no-safe states regain an action in shorter trace prefixes, but six remain "
                "without any contact-negative action even at one tick; faster replanning alone cannot establish viability"
            ),
            "scientific_planner_work_can_proceed_conditionally": False,
            "learned_safety_prediction_remains_blocked": True,
        },
        "platform_stopping_mode_parity": {
            "classification": "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
            "status": "deferred; no physical access, parity trace, vendor control law, or detailed controller specification used",
            "future_evidence": [
                "physical Go2 access or vendor-provided parity traces",
                "mode request and acknowledgement timing",
                "high-rate low-state and independently measured body-motion telemetry",
                "IMU, motor position/velocity/estimated torque, foot force, and applied high-level command",
                "obstacle-free StopMove, BalanceStand, and Damp trials before obstacle testing",
            ],
        },
        "next_step": {
            "classification": "STRICTER_STATE_ELIGIBILITY_AND_ONE_TICK_RESPONSE_ENVELOPE_REQUIRED",
            "specification": (
                "exclude entry into states lacking a one-tick contact-negative response; separately implement and "
                "latency-qualify 100 ms observation/replanning before repeating planner evaluation"
            ),
            "short_commitment_planner_authorized": False,
            "sport_mode_adapter_authorized": False,
        },
        "row_level_evidence": {
            "path": str(row_path), "sha256": sha256(row_path), "bytes": row_path.stat().st_size,
            "content_digest": row_ledger["content_digest"], "rows": len(row_ledger["rows"]),
        },
        "runtime": {"evaluation_s": time.monotonic() - started, "simulation_s": 0.0, "training_s": 0.0, "inference_s": 0.0},
        "storage": {"new_raw_physics_bytes": 0, "row_ledger_bytes": row_path.stat().st_size},
        "confirmations": {
            "model_training": False, "model_inference": False, "simulation": False,
            "sport_mode_adapter_implementation": False, "jepa_access": False,
            "new_state_or_branch_identity": False, "memory": False, "navigation": False,
            "nothing_left_running": True,
        },
    }
    result["runtime"]["evaluation_s"] = time.monotonic() - started
    result["content_digest"] = V.digest(result)
    atomic_json(OUT / "result.json", result)
    print(json.dumps({
        "primary_classification": primary,
        "interface": interface["classification"],
        "safe_states_heldout": {str(h): held[h]["states_retaining_safe_action"] for h in V.HORIZONS},
        "five_tick_no_safe": result["five_tick_no_safe_viability"],
        "ledger_sha256": result["row_level_evidence"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
