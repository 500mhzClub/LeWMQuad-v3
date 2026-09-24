"""Pure binding gates for DEPLOYMENT_VALID_STRONG_BRAKING_MODE_V1.

This module deliberately contains no simulator or robot-control calls.  It
formalises the pre-fixture eligibility gate: a named platform mode must have a
local behavioural implementation, an explicit acknowledgement path, and
known command semantics before it can be exercised as a scientific fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


EXPERIMENT = "DEPLOYMENT_VALID_STRONG_BRAKING_MODE_V1"
UNAVAILABLE = "DEPLOYMENT_VALID_BRAKING_MODE_UNAVAILABLE"
CLAIM_BOUNDARY = (
    "simulated/platform-equivalent stopping fallback for the "
    "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT proxy only; not material-impact, "
    "injury, property-damage, human, fragile-infrastructure, learned-safety, "
    "mission-safety, or closed-loop assurance"
)


@dataclass(frozen=True)
class ModeBinding:
    experiment_name: str
    official_concept: str
    official_interface: str
    local_path: str | None
    local_control_level: str | None
    local_behavior: str | None
    active_balance: str
    command_mechanism: str
    acknowledgement: str
    transition_latency: str
    platform_equivalent: bool
    eligible_for_fixtures: bool
    exclusion_reason: str


def frozen_mode_bindings() -> tuple[ModeBinding, ...]:
    """Return the prospectively frozen mode binding, before state outcomes."""

    mode_manager = "lewm_go2_control/nodes/mode_manager"
    rollout = "lewm_genesis/lewm_genesis/rollout.py"
    return (
        ModeBinding(
            "ACTIVE_STOP",
            "StopMove",
            "official Go2 SportClient high-level RPC",
            mode_manager,
            "CHAMP high-level velocity topic",
            "mode flag followed by one zero Twist publication",
            "not provided as a distinct local controller state",
            "zero desired planar/yaw velocity only",
            "service response acknowledges the local flag, not a platform controller transition",
            "unmeasured because no local StopMove controller exists",
            False,
            False,
            "prohibited zero-velocity retry; no active StopMove-equivalent behavior is implemented",
        ),
        ModeBinding(
            "BALANCE_STAND_TRANSITION",
            "BalanceStand",
            "official Go2 SportClient high-level RPC",
            mode_manager,
            "CHAMP high-level velocity topic",
            "stand/hold aliases publish one zero Twist; no distinct balance transition",
            "CHAMP may continue stance control, but no BalanceStand state is bound or acknowledged",
            "zero desired velocity through the ordinary CHAMP seam",
            "service response acknowledges only the local mode string",
            "unmeasured because no BalanceStand transition exists",
            False,
            False,
            "no distinct local stationary-balance transition; using the alias would retry zero velocity",
        ),
        ModeBinding(
            "DAMPING_MODE",
            "Damp",
            "official Go2 SportClient high-level RPC",
            rollout,
            "Genesis position-target locomotion policy",
            "no runtime damping-mode transition; URDF passive damping is not a command mode",
            "not established; public client API does not provide server-side controller gains/semantics",
            "no local joint velocity/torque damping command path",
            "none",
            "unavailable",
            False,
            False,
            "implementing unknown gains would invent a non-platform low-level controller",
        ),
        ModeBinding(
            "STAND_DOWN",
            "StandDown",
            "official Go2 SportClient high-level RPC",
            None,
            None,
            None,
            "not established as a moving emergency stop",
            "server-side behavior unavailable locally",
            "none",
            "unavailable",
            False,
            False,
            "not locally implemented and not eligible as an ordinary moving brake",
        ),
        ModeBinding(
            "RECOVERY_STAND",
            "RecoveryStand",
            "official Go2 SportClient high-level RPC",
            mode_manager,
            "CHAMP high-level velocity topic",
            "explicitly mapped to zero-velocity CHAMP stance",
            "no sport-mode recovery primitive in the local backend",
            "zero desired velocity only",
            "service response acknowledges the alias, not recovery-controller state",
            "unmeasured",
            False,
            False,
            "recovery behavior is absent and the local alias is the prohibited zero-velocity path",
        ),
    )


def choose_primary_mode(bindings: Iterable[ModeBinding]) -> ModeBinding | None:
    """Apply the preregistered eligibility gate before fixture comparison."""

    eligible = [binding for binding in bindings if binding.eligible_for_fixtures]
    if not eligible:
        return None
    # Outcome-based ordering is intentionally outside this source-only gate.
    # This deterministic order is used only if fixture-qualified records are
    # supplied by a later, authorised controller implementation.
    order = {"ACTIVE_STOP": 0, "BALANCE_STAND_TRANSITION": 1, "DAMPING_MODE": 2}
    return sorted(eligible, key=lambda item: order[item.experiment_name])[0]


def stopping_envelope(
    *,
    mode_qualified: bool,
    planar_speed_m_s: float,
    yaw_rate_rad_s: float,
    current_command: tuple[float, float, float],
    candidate_command: tuple[float, float, float],
    stopping_distance_m: float | None,
    stopping_time_s: float | None,
    uncertainty_margin_m: float,
) -> dict:
    """Construct the prospective guard value without inventing brake data."""

    if min(planar_speed_m_s, uncertainty_margin_m) < 0:
        raise ValueError("speed and uncertainty margin must be non-negative")
    if stopping_distance_m is not None and stopping_distance_m < 0:
        raise ValueError("stopping distance must be non-negative")
    if stopping_time_s is not None and stopping_time_s < 0:
        raise ValueError("stopping time must be non-negative")
    defined = mode_qualified and stopping_distance_m is not None and stopping_time_s is not None
    return {
        "schema": "one_cycle_stopping_envelope_guard_v1",
        "mode_qualified": bool(mode_qualified),
        "planar_speed_m_s": float(planar_speed_m_s),
        "absolute_yaw_rate_rad_s": float(abs(yaw_rate_rad_s)),
        "current_command_vx_vy_yaw": [float(x) for x in current_command],
        "candidate_command_vx_vy_yaw": [float(x) for x in candidate_command],
        "stopping_distance_m": None if stopping_distance_m is None else float(stopping_distance_m),
        "stopping_time_s": None if stopping_time_s is None else float(stopping_time_s),
        "uncertainty_margin_m": float(uncertainty_margin_m),
        "required_clearance_m": (
            float(stopping_distance_m + uncertainty_margin_m) if defined else None
        ),
        "guard_defined": defined,
        "route_authorisation": (
            "evaluate successor safe-response envelope"
            if defined
            else "blocked: no qualified deployment-valid stopping mode"
        ),
    }


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fixture_payload() -> dict:
    """Deterministic pre-execution binding fixture."""

    bindings = frozen_mode_bindings()
    envelope = stopping_envelope(
        mode_qualified=False,
        planar_speed_m_s=0.4,
        yaw_rate_rad_s=-0.2,
        current_command=(0.4, 0.0, -0.2),
        candidate_command=(0.0, 0.0, 0.0),
        stopping_distance_m=None,
        stopping_time_s=None,
        uncertainty_margin_m=0.10,
    )
    tests = {
        "five_modes_bound": len(bindings) == 5,
        "no_mode_fixture_eligible": not any(x.eligible_for_fixtures for x in bindings),
        "no_mode_platform_equivalent": not any(x.platform_equivalent for x in bindings),
        "primary_mode_absent": choose_primary_mode(bindings) is None,
        "envelope_not_fabricated": not envelope["guard_defined"] and envelope["required_clearance_m"] is None,
        "zero_velocity_path_explicitly_excluded": "zero-velocity" in bindings[0].exclusion_reason,
    }
    payload = {
        "schema": "deployment_valid_strong_braking_mode_binding_fixture_v1",
        "bindings": [asdict(x) for x in bindings],
        "tests": tests,
        "pass": all(tests.values()),
    }
    payload["content_digest"] = digest(payload)
    return payload
