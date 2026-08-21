"""Frozen command-mixture contracts for lateral PPO continuation V1."""
from __future__ import annotations

import hashlib
import json

SEED = 2026082014
ORIGINAL_UPDATES = 501
CONTINUATION_UPDATES = min(1000, int(ORIGINAL_UPDATES * 0.25))
VY_LIMIT = 0.20
VY_MIN_NONZERO = 0.05
TRANSITION_DELAY_STEPS = 50  # 1.0 s at the frozen 50 Hz policy rate.
NUM_ENVS = 4096


def contract() -> dict:
    value = {
        "schema": "lateral_recovery_locomotion_controller_training_contract_v1",
        "seed": SEED,
        "original_updates": ORIGINAL_UPDATES,
        "continuation_updates": CONTINUATION_UPDATES,
        "continuation_fraction_rule": "floor(0.25 * 501), capped at 1000",
        "parallel_environments": NUM_ENVS,
        "vy_range_m_s": [-VY_LIMIT, VY_LIMIT],
        "nonzero_vy_magnitude_range_m_s": [VY_MIN_NONZERO, VY_LIMIT],
        "transition_delay_policy_steps": TRANSITION_DELAY_STEPS,
        "mixture": {"historical_route": 0.50, "pure_lateral": 0.25, "route_to_lateral": 0.25},
        "category_assignment": "environment_index_modulo_4; categories persist across reset",
        "mirroring": "lateral sign alternates by floor(environment_index/4), equal for env counts divisible by 8",
        "final_update_only": True,
        "second_seed": False,
        "hyperparameter_sweep": False,
    }
    value["content_digest"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def fixture() -> dict:
    c = contract()
    tests = {
        "single_seed": c["seed"] == 2026082014,
        "budget": c["continuation_updates"] == 125,
        "mixture_sums_to_one": abs(sum(c["mixture"].values()) - 1.0) < 1e-12,
        "mirrored": c["parallel_environments"] % 8 == 0,
        "bounded_vy": c["vy_range_m_s"] == [-0.2, 0.2],
    }
    return {"tests": tests, "pass": all(tests.values())}
