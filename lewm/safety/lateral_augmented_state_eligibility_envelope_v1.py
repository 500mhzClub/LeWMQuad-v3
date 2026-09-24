"""Deterministic reducers for the lateral-augmented eligibility experiment."""
from __future__ import annotations

import hashlib
import json


SOURCE_COMMIT = "7d6672e53e567a2b07e51df506be5db4d6b2d04c"
RESIDUAL_IDS = (
    "wide-cal-0-02",
    "wide-cal-0-05",
    "wide-held-0-05",
    "wide-held-2-04",
    "wide-held-3-03",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def stable_probe(record: dict) -> bool:
    rows = record.get("selected", [])
    return bool(
        record.get("completed_cycles", 0) >= 3
        and len(rows) >= 3
        and not any(row.get("abstained", False) for row in rows[:3])
        and not any(row.get("selected_first_tick_contact", False) for row in rows[:3])
        and not any(row.get("selected_successor_viable") is False for row in rows[:3])
        and all(int(row.get("selected_successor_safe_action_count", 0)) >= 2 for row in rows[:3])
        and not any(any(row.get("termination", {}).values()) for row in rows[:3])
    )


def classify_residual(
    *, stable_depth: int | None, any_viable_depth: bool, pre_existing: bool,
    contact_before_authority: bool, predecessor_available: bool,
) -> str:
    if not predecessor_available:
        return "HISTORICAL_PREDECESSOR_UNAVAILABLE"
    if pre_existing:
        return "PRE_EXISTING_CONTACT"
    if stable_depth is not None:
        return "LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT"
    if contact_before_authority:
        return "CONTACT_BEFORE_CONTROL_AUTHORITY"
    if any_viable_depth:
        return "INTERMITTENT_NO_STABLE_ENVELOPE"
    return "PERSISTENT_AUGMENTED_ACTION_SET_FAILURE"


def experiment_classification(*, gate_pass: bool, recovered: int, persistent: int) -> str:
    if gate_pass:
        return "LATERAL_AUGMENTED_STATE_ELIGIBILITY_SIGNAL"
    if recovered > persistent and persistent > 0:
        return "STATE_ELIGIBILITY_SIGNAL_RESIDUAL_ACTION_SET_NO_GO"
    return "LATERAL_AUGMENTED_STATE_ELIGIBILITY_NO_GO"


def fixture_payload() -> dict:
    stable = {
        "completed_cycles": 3,
        "selected": [
            {
                "abstained": False,
                "selected_first_tick_contact": False,
                "selected_successor_viable": True,
                "selected_successor_safe_action_count": 2,
                "termination": {"fall": False},
            }
            for _ in range(3)
        ],
    }
    tests = {
        "five_residuals_frozen": len(RESIDUAL_IDS) == 5,
        "stable_probe_accepts_margin_two": stable_probe(stable),
        "stable_probe_rejects_margin_one": not stable_probe({
            **stable,
            "selected": [{**row, "selected_successor_safe_action_count": 1}
                         for row in stable["selected"]],
        }),
        "stable_classification": classify_residual(
            stable_depth=2, any_viable_depth=True, pre_existing=False,
            contact_before_authority=False, predecessor_available=True,
        ) == "LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT",
        "persistent_classification": classify_residual(
            stable_depth=None, any_viable_depth=False, pre_existing=False,
            contact_before_authority=False, predecessor_available=True,
        ) == "PERSISTENT_AUGMENTED_ACTION_SET_FAILURE",
    }
    payload = {"schema": "lateral_augmented_state_eligibility_fixture_v1", "tests": tests}
    payload["pass"] = all(tests.values())
    payload["content_digest"] = digest(payload)
    return payload
