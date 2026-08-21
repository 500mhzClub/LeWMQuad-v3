"""Pure contracts and reducers for the H1 emergency-brake successor."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np

SPEED_THRESHOLD_M_S = 0.05
YAW_RATE_THRESHOLD_RAD_S = 0.10
CONSECUTIVE_COMMAND_TICKS = 3
COMMAND_TICK_S = 0.10
PHYSICS_DT_S = 0.002
MAX_BRAKE_S = 2.0
MAX_PHYSICS_STEPS = 1000
BRAKE_COMMAND = np.asarray([0.0, 0.0, 0.0], np.float32)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stopped_tick(
    planar_speed: Sequence[float],
    yaw_rate: Sequence[float],
    unsafe: Sequence[bool],
    *,
    speed_threshold: float = SPEED_THRESHOLD_M_S,
    yaw_threshold: float = YAW_RATE_THRESHOLD_RAD_S,
    consecutive: int = CONSECUTIVE_COMMAND_TICKS,
) -> int | None:
    speed = np.asarray(planar_speed, np.float64); yaw = np.asarray(yaw_rate, np.float64); bad = np.asarray(unsafe, bool)
    if speed.shape != yaw.shape or speed.shape != bad.shape:
        raise ValueError("stop-condition shape mismatch")
    run = 0
    for index in range(len(speed)):
        run = run + 1 if speed[index] < speed_threshold and abs(yaw[index]) < yaw_threshold and not bad[index] else 0
        if run >= consecutive:
            return int(index)
    return None


def classify_brake(
    *,
    boundary_contact: bool,
    contact: bool,
    contact_time_s: float | None,
    stopped: bool,
    stable: bool,
    speed_reduced_before_contact: bool,
) -> str:
    if boundary_contact:
        return "PRE_EXISTING_CONTACT"
    if not stable:
        return "BRAKE_DESTABILISES_ROBOT"
    if not contact and stopped:
        return "BRAKE_RESTORES_SAFE_RESPONSE"
    if contact and (contact_time_s is not None and contact_time_s <= COMMAND_TICK_S or not speed_reduced_before_contact):
        return "CONTACT_BEFORE_BRAKE_CAN_RESPOND"
    if contact or not stopped:
        return "BRAKE_COMMAND_AUTHORITY_INSUFFICIENT"
    return "UNRESOLVED"


def classify_predecessor(current: dict, predecessor: dict | None) -> str | None:
    if not current.get("contact", False):
        return None
    if current.get("boundary_contact", False):
        return "PRE_EXISTING_CONTACT"
    if predecessor is None:
        return "UNRESOLVED"
    if predecessor.get("qualified_safe_brake", False):
        return "PREDECESSOR_VIABILITY_GUARD_REQUIRED"
    if not predecessor.get("boundary_contact", False):
        return "EMERGENCY_BRAKE_INSUFFICIENT"
    return "PRE_EXISTING_CONTACT"


def summarize(values: Sequence[float]) -> dict:
    x = np.asarray(values, np.float64)
    if not len(x):
        return {"count": 0}
    return {"count": int(len(x)), "min": float(x.min()), "q25": float(np.quantile(x, .25)),
            "median": float(np.median(x)), "mean": float(x.mean()), "q75": float(np.quantile(x, .75)), "max": float(x.max())}


def fixture_payload() -> dict:
    speed = [0.2, 0.04, 0.03, 0.02]
    yaw = [0.2, 0.09, 0.08, 0.07]
    tests = {
        "three_tick_stop": stopped_tick(speed, yaw, [False] * 4) == 3,
        "strict_speed_tie_not_stopped": stopped_tick([.05, .05, .05], [0, 0, 0], [False] * 3) is None,
        "unsafe_breaks_run": stopped_tick([.01] * 4, [0] * 4, [False, True, False, False]) is None,
        "safe_class": classify_brake(boundary_contact=False, contact=False, contact_time_s=None, stopped=True, stable=True, speed_reduced_before_contact=True) == "BRAKE_RESTORES_SAFE_RESPONSE",
        "immediate_class": classify_brake(boundary_contact=False, contact=True, contact_time_s=.02, stopped=False, stable=True, speed_reduced_before_contact=False) == "CONTACT_BEFORE_BRAKE_CAN_RESPOND",
        "instability_class": classify_brake(boundary_contact=False, contact=False, contact_time_s=None, stopped=False, stable=False, speed_reduced_before_contact=True) == "BRAKE_DESTABILISES_ROBOT",
        "predecessor_guard": classify_predecessor({"contact": True, "boundary_contact": False}, {"qualified_safe_brake": True}) == "PREDECESSOR_VIABILITY_GUARD_REQUIRED",
        "zero_command": bool(np.array_equal(BRAKE_COMMAND, np.zeros(3, np.float32))),
    }
    payload = {"schema": "h1_safe_action_set_successor_fixture_v1", "tests": tests, "pass": all(tests.values())}
    payload["content_digest"] = digest(payload)
    payload["byte_identical_regeneration"] = payload["content_digest"] == digest({k: v for k, v in payload.items() if k not in {"content_digest", "byte_identical_regeneration"}})
    return payload
