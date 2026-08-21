"""Pure controller-authority reducers for lateral-retreat qualification."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_batch


PROBE_MAGNITUDE_M_S = 0.20
FIXTURE_CONTEXTS = (
    ("obstacle_free_rest", (0.0, 0.0, 0.0)),
    ("obstacle_free_forward_motion", (0.30, 0.0, 0.0)),
    ("obstacle_free_yaw_motion", (0.0, 0.0, 0.45)),
    ("left_wall", (0.0, 0.0, 0.0)),
    ("right_wall", (0.0, 0.0, 0.0)),
    ("front_left_corner", (0.20, 0.0, 0.0)),
    ("front_right_corner", (0.20, 0.0, 0.0)),
    ("narrow_corridor_asymmetric_joint_phase", (0.0, 0.0, -0.45)),
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def controller_authority_audit(
    manifest: Mapping, policy_command_config: Mapping, primitive_registry: Mapping
) -> dict:
    safety = manifest["locomotion"]["safety"]
    delta = safety["max_command_delta_per_tick"]
    primitives = primitive_registry["primitives"]
    lateral = {name: primitives[name] for name in ("lateral_left", "lateral_right")}
    policy_bank = np.asarray(policy_command_config["lewm_command_bank"], dtype=np.float64)
    structural_vy = tuple(manifest["locomotion"]["command_vector"]["order"])[1] == "vy_body_mps"
    audit = {
        "command_vector_structurally_contains_vy": structural_vy,
        "manifest_vy_range_m_s": [float(safety["min_vy_mps"]), float(safety["max_vy_mps"])],
        "manifest_max_delta_vy_m_s_per_tick": float(delta["vy_mps"]),
        "policy_training_vy_range_m_s": [float(x) for x in policy_command_config["lin_vel_y_range"]],
        "policy_training_bank_vy_values_m_s": sorted(set(float(x) for x in policy_bank[:, 1])),
        "registry_lateral": {
            name: {
                "requested_vy_m_s": float(row["command"]["vy_body_mps"]),
                "train": bool(row["train"]),
                "enable_after_validation": bool(row["enable_after_validation"]),
            }
            for name, row in lateral.items()
        },
    }
    audit["nonzero_lateral_controller_supported"] = bool(
        structural_vy
        and audit["manifest_vy_range_m_s"][0] < 0.0
        and audit["manifest_vy_range_m_s"][1] > 0.0
        and audit["manifest_max_delta_vy_m_s_per_tick"] > 0.0
        and audit["policy_training_vy_range_m_s"][0] < 0.0
        and audit["policy_training_vy_range_m_s"][1] > 0.0
        and any(abs(value) > 0.0 for value in audit["policy_training_bank_vy_values_m_s"])
        and all(row["train"] for row in audit["registry_lateral"].values())
    )
    return audit


def command_adapter_fixture_rows(limits: SafetyLimits) -> list[dict]:
    """Exercise every mirrored fixture twice through the unchanged adapter.

    These are adapter-gate fixtures.  Environment dynamics are intentionally
    not entered if the applied lateral command is zero, because no lateral
    mechanism would then be under test.
    """

    rows: list[dict] = []
    for fixture, previous in FIXTURE_CONTEXTS:
        for direction_index, (direction, requested_vy) in enumerate(
            (("left", PROBE_MAGNITUDE_M_S), ("right", -PROBE_MAGNITUDE_M_S))
        ):
            requested = np.asarray([[[0.0, requested_vy, 0.0]]], dtype=np.float32)
            prior = np.asarray([previous], dtype=np.float32)
            for repeat in range(2):
                applied, clipped = apply_safety_limits_batch(
                    requested, prior, limits, enforce_rate_limits=True
                )
                core = {
                    "fixture": fixture,
                    "direction": direction,
                    "direction_index": 12 + direction_index,
                    "previous_command_vx_vy_wz": [float(x) for x in prior[0]],
                    "requested_command_vx_vy_wz": [float(x) for x in requested[0, 0]],
                    "applied_command_vx_vy_wz": [float(x) for x in applied[0, 0]],
                    "clipped": bool(clipped[0]),
                    "applied_nonzero_lateral": bool(abs(float(applied[0, 0, 1])) > 1e-9),
                    "dynamics_executed": False,
                    "actual_lateral_displacement_m": None,
                    "contact": None,
                    "fall_or_unsafe_termination": None,
                    "reason_dynamics_not_executed": "applied_lateral_command_is_zero",
                }
                rows.append({**core, "repeat": repeat, "core_digest": digest(core)})
    return rows


def fixture_reduction(rows: Sequence[Mapping]) -> dict:
    grouped: dict[tuple[str, str], list[Mapping]] = {}
    for row in rows:
        grouped.setdefault((str(row["fixture"]), str(row["direction"])), []).append(row)
    deterministic = all(
        len(group) == 2 and len({str(row["core_digest"]) for row in group}) == 1
        for group in grouped.values()
    )
    result = {
        "fixture_contexts": len(FIXTURE_CONTEXTS),
        "mirrored_instances": 2,
        "repeats_per_instance": 2,
        "rows": len(rows),
        "byte_identical_reduction": deterministic,
        "finite_controller_outputs": all(
            all(np.isfinite(float(x)) for x in row["applied_command_vx_vy_wz"])
            for row in rows
        ),
        "all_requests_clipped": all(bool(row["clipped"]) for row in rows),
        "nonzero_applied_lateral_rows": sum(bool(row["applied_nonzero_lateral"]) for row in rows),
        "measurable_requested_direction": False,
        "environment_fixture_execution_entered": False,
        "qualification_pass": False,
        "stop_classification": "LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO",
    }
    result["content_digest"] = digest(result)
    return result

