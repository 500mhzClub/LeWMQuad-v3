from lewm.safety import h1_candidate_bank_viability_successor_v1 as subject
from lewm_genesis.lewm_contract import SafetyLimits


def frozen_limits():
    return SafetyLimits(
        min_vx_mps=-0.3,
        max_vx_mps=0.3,
        min_vy_mps=0.0,
        max_vy_mps=0.0,
        max_yaw_rate_radps=0.5,
        max_delta_vx_mps=0.25,
        max_delta_vy_mps=0.0,
        max_delta_yaw_rate_radps=0.35,
    )


def test_mirrored_fixture_is_deterministic_and_clipped():
    rows = subject.command_adapter_fixture_rows(frozen_limits())
    result = subject.fixture_reduction(rows)
    assert len(rows) == 32
    assert result["byte_identical_reduction"]
    assert result["nonzero_applied_lateral_rows"] == 0
    assert result["stop_classification"] == "LATERAL_RETREAT_CONTROLLER_AUTHORITY_NO_GO"


def test_authority_requires_manifest_training_and_registry_support():
    manifest = {
        "locomotion": {
            "command_vector": {"order": ["vx_body_mps", "vy_body_mps", "yaw_rate_radps"]},
            "safety": {
                "min_vy_mps": 0.0,
                "max_vy_mps": 0.0,
                "max_command_delta_per_tick": {"vy_mps": 0.0},
            },
        }
    }
    policy = {"lin_vel_y_range": [0.0, 0.0], "lewm_command_bank": [[0.0, 0.0, 0.0]]}
    registry = {
        "primitives": {
            "lateral_left": {"train": False, "enable_after_validation": True,
                             "command": {"vy_body_mps": 0.2}},
            "lateral_right": {"train": False, "enable_after_validation": True,
                              "command": {"vy_body_mps": -0.2}},
        }
    }
    audit = subject.controller_authority_audit(manifest, policy, registry)
    assert not audit["nonzero_lateral_controller_supported"]
    assert audit["policy_training_bank_vy_values_m_s"] == [0.0]

