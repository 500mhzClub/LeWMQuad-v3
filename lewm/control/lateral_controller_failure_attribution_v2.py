"""Frozen reducers for lateral-controller failure attribution V2."""
from __future__ import annotations

import hashlib
import json
import math


SOURCE_COMMIT = "004ef60c81d98f744e5dad0206d4c6a618707196"
V1_SEED = 2026082014
SUCCESSOR_SEED = 2026082015
SUCCESSOR_UPDATES = 500
VY_LIMIT_M_S = 0.20
ORIGINAL_TRACKING_SIGMA = 0.25
CORRECTED_LATERAL_TRACKING_SIGMA = VY_LIMIT_M_S**2
DETERMINISM_TOLERANCE = 1e-4


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tracking_reward(error_m_s: float, sigma: float = ORIGINAL_TRACKING_SIGMA) -> float:
    return math.exp(-(float(error_m_s) ** 2) / float(sigma))


def error_for_reward(reward: float, sigma: float = ORIGINAL_TRACKING_SIGMA) -> float:
    if not 0.0 < reward <= 1.0:
        raise ValueError("reward must be in (0, 1]")
    return math.sqrt(-float(sigma) * math.log(float(reward)))


def choose_successor_path(audit: dict) -> str:
    """Apply the frozen A/B/C/stop decision without outcome-dependent tuning."""
    if audit.get("v1_requalification_pass"):
        return "PATH_A_REQUALIFICATION_ONLY"
    if audit.get("plant_or_gait_authority_absent"):
        return "STOP_LATERAL_LOCOMOTION_ARCHITECTURE_NO_GO"
    if audit.get("concrete_reward_or_binding_defect"):
        return "PATH_C_CORRECTED_SUCCESSOR_TRAINING"
    if (
        audit.get("bindings_correct")
        and audit.get("policy_command_sensitive")
        and audit.get("v1_failure_classification") == "LIKELY_UNDERTRAINED"
    ):
        return "PATH_B_SAME_SEED_FULL_BUDGET_CONTINUATION"
    return "STOP_LATERAL_LOCOMOTION_ARCHITECTURE_NO_GO"


def fixture_payload() -> dict:
    zero_response_reward = tracking_reward(VY_LIMIT_M_S)
    tests = {
        "frozen_source": len(SOURCE_COMMIT) == 40,
        "v1_seed": V1_SEED == 2026082014,
        "successor_seed": SUCCESSOR_SEED == 2026082015,
        "successor_budget": SUCCESSOR_UPDATES == 500,
        "broad_original_reward": zero_response_reward > 0.85,
        "corrected_zero_response_reward": abs(
            tracking_reward(VY_LIMIT_M_S, CORRECTED_LATERAL_TRACKING_SIGMA) - math.exp(-1.0)
        ) < 1e-12,
        "deterministic": DETERMINISM_TOLERANCE == 1e-4,
    }
    payload = {"schema": "lateral_controller_failure_attribution_v2_fixture", "tests": tests}
    payload["pass"] = all(tests.values())
    payload["content_digest"] = digest(payload)
    return payload
