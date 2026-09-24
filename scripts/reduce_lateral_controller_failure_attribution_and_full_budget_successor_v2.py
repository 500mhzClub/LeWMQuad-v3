#!/usr/bin/env python3
"""Persist the bounded V2 controller/viability result without rerunning it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/lateral_controller_failure_attribution_and_full_budget_successor_v2"
SOURCE_COMMIT = "004ef60c81d98f744e5dad0206d4c6a618707196"


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    audit = read(OUT / "attribution_result.json")
    training = read(OUT / "successor_training_result.json")
    v1_qualification = read(OUT / "qualification_v1_requalification.json")
    qualification = read(OUT / "qualification_successor_final.json")
    science = read(OUT / "scientific_viability_result.json")
    receipt = read(OUT / "scientific_collection_receipt.json")
    states = [read(path) for path in sorted((OUT / "scientific_states").glob("*.json"))]
    rollouts = [read(path) for path in sorted((OUT / "scientific_rollouts").glob("*.json"))]

    runtime_s = sum(
        (
            float(audit["runtime_s"]),
            float(v1_qualification["runtime_s"]),
            float(training["runtime_s"]),
            float(qualification["runtime_s"]),
            float(receipt["wall_runtime_s"]),
        )
    )
    checkpoint = Path(training["checkpoint"]["path"])
    result = {
        "schema": "lateral_controller_failure_attribution_and_full_budget_successor_v2_result",
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": (
            "simulation-only oracle viability under H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT; "
            "no learned planner-safety or physical Go2 qualification"
        ),
        "determinism_localisation": audit["determinism_localisation"],
        "command_reward_audit": {
            "bindings": audit["command_reward_path"]["bindings"],
            "reward_interpretability": audit["reward_interpretability"],
            "policy_sensitivity": audit["policy_sensitivity"],
            "v1_failure_classification": audit["v1_failure_classification"],
        },
        "successor_path": audit["successor_path"],
        "training": {
            "seed": training["seed"],
            "updates": training["updates"],
            "start_iteration": training["start_iteration"],
            "final_iteration": training["final_iteration"],
            "parallel_environments": training["contract"]["parallel_environments"],
            "command_mixture": training["contract"]["command_mixture"],
            "vy_range_m_s": training["contract"]["vy_range_m_s"],
            "correction": training["contract"]["correction"],
            "checkpoint": training["checkpoint"],
            "checkpoint_verified": sha256(checkpoint) == training["checkpoint"]["sha256"],
            "monitoring_records": training["monitoring_records"],
            "first_monitoring": training["monitoring"][0],
            "final_monitoring": training["monitoring"][-1],
        },
        "controller_qualification": {
            "classification": qualification["classification"],
            "pass": qualification["pass"],
            "route_non_regression": {
                "pass": qualification["route_non_regression"]["pass"],
                "source": qualification["route_non_regression"]["source"],
                "successor": qualification["route_non_regression"]["successor"],
                "allowed": qualification["route_non_regression"]["allowed"],
            },
            "lateral_tracking": {
                "pass": qualification["lateral_tracking"]["pass"],
                "correct_sign_every_fixture_at_0_2_s": qualification["lateral_tracking"]["correct_sign_every_fixture_at_0_2_s"],
                "measurable_every_fixture_at_0_2_s": qualification["lateral_tracking"]["measurable_every_fixture_at_0_2_s"],
                "median_0_5_s_velocity_fraction_for_abs_0_2": qualification["lateral_tracking"]["median_0_5_s_velocity_fraction_for_abs_0_2"],
                "determinism": qualification["lateral_tracking"]["determinism"],
            },
            "mode_transition": {
                "pass": qualification["mode_transition"]["pass"],
                "safe": qualification["mode_transition"]["safe"],
                "route_tracking_resumes": qualification["mode_transition"]["route_tracking_resumes"],
                "determinism": qualification["mode_transition"]["determinism"],
            },
        },
        "scientific_viability": {
            "primary_classification": science["primary_classification"],
            "full_panel": science["full_panel"],
            "rollouts": {key: value for key, value in science["rollouts"].items() if key != "per_state"},
            "gate": science["gate"],
            "generated_branches": science["generated_branches"],
        },
        "preserved_terminals": [
            "LATERAL_CONTROLLER_QUALIFICATION_NO_GO",
            "LATERAL_TRACKING_AUTHORITY_NO_GO",
            "SUPPORTED_VX_YAW_CONTROL_AUTHORITY_NO_GO",
            "CANDIDATE_BANK_ONE_TICK_VIABILITY_NO_GO",
            "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION",
            "ONE_TICK_FULL_JEPA_COMPUTE_NO_GO",
            "TWO_TICK_COMPUTE_FEASIBLE_INTERFACE_UNQUALIFIED",
            "REPLANNING_INTERFACE_UNRESOLVED",
            "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        ],
        "runtime_s": runtime_s,
        "evidence_counts": {"full_panel_states": len(states), "rollouts": len(rollouts)},
        "training_paths_run": 1,
        "jepa_predictor_opened_or_executed": False,
    }
    write_json(OUT / "aggregate_result.json", result)

    CACHE.mkdir(parents=True, exist_ok=True)
    ledger = CACHE / "row_level_evidence_v2.jsonl"
    with ledger.open("w") as stream:
        for row in audit["command_reward_path"]["rows"].items():
            family, values = row
            for value in values:
                stream.write(json.dumps({"record_type": "command_reward", "family": family, **value}, sort_keys=True) + "\n")
        for value in training["monitoring"]:
            stream.write(json.dumps({"record_type": "training_monitor", **value}, sort_keys=True) + "\n")
        for family in ("route_non_regression", "lateral_tracking", "mode_transition"):
            for value in qualification[family]["rows"]:
                stream.write(json.dumps({"record_type": "controller_qualification", "family": family, **value}, sort_keys=True) + "\n")
        for value in states:
            stream.write(json.dumps({"record_type": "scientific_current_state", **value}, sort_keys=True) + "\n")
        for value in rollouts:
            stream.write(json.dumps({"record_type": "scientific_rollout", **value}, sort_keys=True) + "\n")

    evidence = [
        OUT / "attribution_result.json",
        OUT / "qualification_v1_requalification.json",
        OUT / "successor_training_result.json",
        OUT / "qualification_successor_final.json",
        OUT / "scientific_collection_receipt.json",
        OUT / "scientific_viability_result.json",
        OUT / "aggregate_result.json",
        checkpoint,
        ledger,
    ]
    digests = {
        str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in evidence
    }
    write_json(CACHE / "content_digests.json", digests)
    print(json.dumps({"aggregate": result, "ledger": str(ledger), "ledger_sha256": sha256(ledger), "digests": digests}, indent=2))


if __name__ == "__main__":
    main()
