#!/usr/bin/env python3
"""Run the no-simulation platform-mode binding gate and persist its receipt."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import deployment_valid_strong_braking_mode_v1 as S

SOURCE_COMMIT = "ccfce9444a0ce49e837e61b5e2da4ddfcbedf5be"
OUT = ROOT / ".generated/deployment_valid_strong_braking_mode_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/deployment_valid_strong_braking_mode_v1")
PRIOR = ROOT / ".generated/h1_safe_action_set_successor_v1/result.json"
SOURCES = {
    "local_mode_manager": ROOT / "lewm_go2_control/nodes/mode_manager",
    "genesis_rollout": ROOT / "lewm_genesis/lewm_genesis/rollout.py",
    "champ_velocity_smoother": ROOT / "third_party/unitree_go2_ros2/champ_base/config/velocity_smoother/velocity_smoother.yaml",
    "ros_effort_controller": ROOT / "third_party/unitree_go2_ros2/unitree_go2_sim/config/ros_control/ros_control.yaml",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def main() -> int:
    start = time.monotonic()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError(f"expected source commit {SOURCE_COMMIT}, found {head}")

    texts = {name: path.read_text() for name, path in SOURCES.items()}
    source_checks = {
        "stop_stand_hold_recovery_publish_only_zero_twist": (
            'requested in {"stand", "hold", "stop", "recovery_stand"}' in texts["local_mode_manager"]
            and "self._publish_zero_twist()" in texts["local_mode_manager"]
        ),
        "recovery_explicitly_lacks_sport_primitive": (
            "no sport-mode fall recovery primitive is available" in texts["local_mode_manager"]
        ),
        "genesis_uses_policy_position_targets": (
            "joint_targets = self.policy.act(obs)" in texts["genesis_rollout"]
            and "robot.control_dofs_position" in texts["genesis_rollout"]
        ),
        "genesis_velocity_write_is_reset_only": (
            texts["genesis_rollout"].count("set_dofs_velocity(") == 1
            and "def _reset_robot_to_spawn" in texts["genesis_rollout"]
        ),
        "champ_smoother_is_not_active_mode": (
            "decel_factor:" in texts["champ_velocity_smoother"]
            and "StopMove" not in texts["champ_velocity_smoother"]
        ),
        "ros_effort_controller_has_no_named_platform_mode": (
            "command_interfaces:" in texts["ros_effort_controller"]
            and "StopMove" not in texts["ros_effort_controller"]
            and "BalanceStand" not in texts["ros_effort_controller"]
        ),
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"local controller binding changed: {source_checks}")

    prior = json.loads(PRIOR.read_text())
    preserved = {
        "action_set": prior["classifications"]["action_set"],
        "fallback": prior["classifications"]["fallback"],
        "candidate_bank": prior["classifications"]["candidate_bank"],
        "ranking": prior["classifications"]["ranking"],
        "successor_action_availability": prior["successor_action_availability"],
        "prior_result_sha256": sha256(PRIOR),
        "prior_content_digest": prior["content_digest"],
    }
    expected = {
        "H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO",
        "EMERGENCY_BRAKE_INSUFFICIENT",
        "CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO",
        "KINEMATIC_ROUTE_RANKING_LIMITATION",
    }
    actual = {
        preserved["action_set"],
        *preserved["fallback"],
        preserved["candidate_bank"]["classification"],
        preserved["ranking"],
    }
    if not expected.issubset(actual):
        raise RuntimeError(f"required predecessor terminals not reproduced: {actual}")

    fixture = S.fixture_payload()
    if not fixture["pass"] or S.digest({k: v for k, v in fixture.items() if k != "content_digest"}) != fixture["content_digest"]:
        raise RuntimeError("binding fixture is not deterministic")
    bindings = fixture["bindings"]
    primary = S.choose_primary_mode(S.frozen_mode_bindings())
    assert primary is None

    # Section 5 terminal: an empty eligible set forbids physics fixtures and
    # all scientific-state execution.  "Not evaluated" is kept distinct from
    # a zero-event physical result.
    fixture_matrix = []
    for row in bindings:
        fixture_matrix.append(
            {
                "mode": row["experiment_name"],
                "official_concept": row["official_concept"],
                "binding_eligibility_passed": False,
                "physics_fixture_runs": 0,
                "two_repeat_determinism": "not_evaluated",
                "stop_and_stability_outcome": "not_evaluated",
                "reason": row["exclusion_reason"],
            }
        )

    envelope_example = S.stopping_envelope(
        mode_qualified=False,
        planar_speed_m_s=0.4,
        yaw_rate_rad_s=0.2,
        current_command=(0.4, 0.0, 0.2),
        candidate_command=(0.0, 0.0, 0.0),
        stopping_distance_m=None,
        stopping_time_s=None,
        uncertainty_margin_m=0.10,
    )
    row_ledger = {
        "schema": "deployment_valid_strong_braking_mode_row_evidence_v1",
        "source_commit": SOURCE_COMMIT,
        "mode_bindings": bindings,
        "fixture_matrix": fixture_matrix,
        "selected_primary_mode": None,
        "eligible_mode_count": 0,
        "scientific_state_rows": [],
        "predecessor_rows": [],
        "stopping_envelope_unqualified_example": envelope_example,
    }
    row_ledger["content_digest"] = S.digest(row_ledger)
    ledger_path = CACHE / "row_level_evidence_v1.json"
    atomic_json(ledger_path, row_ledger)

    sdk_presence = {
        "unitree_sdk2py": importlib.util.find_spec("unitree_sdk2py") is not None,
        "unitree_sdk2": importlib.util.find_spec("unitree_sdk2") is not None,
    }
    result = {
        "schema": "deployment_valid_strong_braking_mode_result_v1",
        "experiment": S.EXPERIMENT,
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": S.CLAIM_BOUNDARY,
        "preserved_results": preserved,
        "official_interfaces": {
            "source": "Unitree SDK2 public Go2 SportClient client API and examples",
            "methods": ["StopMove", "BalanceStand", "Damp", "StandDown", "RecoveryStand"],
            "server_controller_semantics_available_locally": False,
            "references": [
                "https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/sport/sport_client.hpp",
                "https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_sport_client.cpp",
                "https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py",
            ],
        },
        "local_mode_bindings": bindings,
        "local_sdk_modules": sdk_presence,
        "source_checks": source_checks,
        "source_bindings": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in SOURCES.items()
        },
        "fixture": {
            "binding_fixture": fixture,
            "matrix": fixture_matrix,
            "physics_fixture_modes": 0,
            "physics_fixture_runs": 0,
            "reason_not_run": "no local mode passed the pre-fixture platform-equivalence and implementation gate",
        },
        "primary_mode": None,
        "mode_acknowledgement_latency": {"status": "not_measured", "reason": "no eligible local controller mode"},
        "stopping_time_distribution": {"count": 0, "status": "not_evaluated"},
        "stopping_distance_distribution": {"count": 0, "status": "not_evaluated"},
        "scientific_execution": {
            "frozen_states_bound": 48,
            "new_primary_mode_branches": 0,
            "predecessor_branches": 0,
            "contact_outcomes": "not_evaluated",
            "fall_outcomes": "not_evaluated",
            "stability_outcomes": "not_evaluated",
            "historical_no_safe_state_outcomes": "not_reexecuted; 11/48 remain without a qualified fallback",
        },
        "classifications": {
            "mode": S.UNAVAILABLE,
            "cause": ["SIMULATOR_CONTROLLER_LIMITATION", "MISSING_PLATFORM_MODE_IMPLEMENTATION"],
            "action_set": "H1_SAFE_ACTION_SET_SUCCESSOR_NO_GO",
            "candidate_bank": "CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO",
            "ranking": "KINEMATIC_ROUTE_RANKING_LIMITATION",
        },
        "stopping_envelope": {
            "status": "UNDEFINED_UNTIL_MODE_QUALIFIED",
            "indexed_by": [
                "current planar speed", "yaw rate", "current command", "candidate command",
                "stopping mode", "stopping distance", "stopping time", "uncertainty margin",
            ],
            "required_clearance": "validated_stop_distance(mode,state,command) + uncertainty_margin",
            "one_cycle_guard_rule": (
                "reject a route action that can enter a next state with neither a contact-negative route response "
                "nor a qualified stop inside available clearance"
            ),
            "qualification_requirements": [
                "monotone conservative stopping envelope over speed and absolute yaw rate",
                "mode-request-to-acknowledgement latency included",
                "controller and plant variance included in uncertainty margin",
                "contact-free and stable fixture validation before frozen-state evaluation",
            ],
            "example": envelope_example,
        },
        "exact_next_experiment": {
            "name": "GENESIS_GO2_SPORT_MODE_ADAPTER_V1",
            "purpose": "implement and parity-qualify exactly one genuine Go2 StopMove or BalanceStand controller transition before repeating stopping qualification",
            "requirements": [
                "obtain documented or black-box physical Go2 mode request, acknowledgement, joint-response, stopping-time and stopping-distance traces",
                "bind an explicit Genesis mode state machine rather than a zero Twist alias",
                "drive only physically simulated joint position/velocity/torque interfaces with validated platform-equivalent parameters",
                "preserve body momentum, gravity, collision, actuator limits and stability dynamics",
                "qualify training-only fixtures before requesting authority for the frozen 48-state panel",
            ],
            "learned_safety_blocked": True,
        },
        "row_level_evidence": {
            "path": str(ledger_path),
            "sha256": sha256(ledger_path),
            "bytes": ledger_path.stat().st_size,
            "content_digest": row_ledger["content_digest"],
        },
        "runtime": {
            "binding_and_fixture_s": time.monotonic() - start,
            "simulation_s": 0.0,
            "model_training_s": 0.0,
            "learned_inference_s": 0.0,
        },
        "storage": {
            "new_raw_physics_bytes": 0,
            "row_ledger_bytes": ledger_path.stat().st_size,
        },
        "confirmations": {
            "model_training": False,
            "learned_inference": False,
            "simulation": False,
            "jepa_access": False,
            "new_scientific_panel": False,
            "memory": False,
            "navigation": False,
            "nothing_left_running": True,
        },
    }
    result["runtime"]["binding_and_fixture_s"] = time.monotonic() - start
    result["content_digest"] = S.digest(result)
    atomic_json(OUT / "result.json", result)
    print(
        json.dumps(
            {
                "classification": result["classifications"]["mode"],
                "eligible_modes": 0,
                "scientific_branches": 0,
                "row_ledger_sha256": result["row_level_evidence"]["sha256"],
                "content_digest": result["content_digest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
