from __future__ import annotations

from types import SimpleNamespace

from scripts import (
    go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_lifecycle
    as lifecycle,
)


def _engine() -> SimpleNamespace:
    return SimpleNamespace(
        SCHEMA_PREFIX="test_v16",
        V12_GATE_CHECK_NAMES=tuple(f"check_{index}" for index in range(24)),
        CONTROL_NAMES=("true", "wrong_rgb", "wrong_action", "persistence"),
        CONTROL_CHECK_NAMES=("finite", "state_preserved", "positive"),
        _validate_physical_summary=lambda value: value,
    )


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    engine = _engine()
    v12 = {
        "checks": {name: True for name in engine.V12_GATE_CHECK_NAMES},
        "passed": True,
    }
    physical = {
        "passed_margin_count": 94,
        "total_shortfall": 35.0,
        "rough_motion": {
            "depth_p95_m": 1.3,
            "ground_balanced_accuracy": 0.70,
            "pixel_balanced_accuracy": 0.88,
        },
    }
    controls = {
        name: {check: True for check in engine.CONTROL_CHECK_NAMES}
        for name in engine.CONTROL_NAMES
    }
    return v12, physical, controls


def test_extension_eligibility_requires_every_preregistered_conjunct() -> None:
    engine = _engine()
    v12, physical, controls = _inputs()
    decision = lifecycle.evaluate_extension_eligibility_v16(
        v12,
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert decision["passed"] is True
    assert decision["automatic_execution_authorized"] is False

    physical["rough_motion"]["depth_p95_m"] = 1.45
    failed = lifecycle.evaluate_extension_eligibility_v16(
        v12,
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert failed["passed"] is False
    assert failed["checks"]["rough_depth_p95_strictly_below_1_45_m"] is False


def test_update400_gate_uses_only_the_frozen_v16_conjunction() -> None:
    engine = _engine()
    _, physical, controls = _inputs()
    physical.update(
        passed_margin_count=72,
        total_shortfall=68.0,
    )
    physical["rough_motion"].update(
        depth_p95_m=1.8,
        ground_balanced_accuracy=0.1,
        pixel_balanced_accuracy=0.1,
    )
    decision = lifecycle.evaluate_update400_gate_v16(
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert decision["passed"] is True
    assert set(decision["checks"]) == {
        "structural_integrity_pass",
        "all_twelve_causal_control_checks_true",
        "passed_physical_margin_count_at_least_72",
        "total_physical_shortfall_strictly_below_v14_v15_update400",
        "rough_depth_p95_strictly_below_v14_v15_update400",
    }


def test_update400_gate_preserves_strict_threshold_boundaries() -> None:
    engine = _engine()
    _, physical, controls = _inputs()
    physical.update(
        passed_margin_count=72,
        total_shortfall=lifecycle.UPDATE400_THRESHOLDS_V16[
            "total_shortfall_strictly_less_than"
        ],
    )
    physical["rough_motion"]["depth_p95_m"] = 1.0
    shortfall_boundary = lifecycle.evaluate_update400_gate_v16(
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert shortfall_boundary["passed"] is False

    physical["total_shortfall"] = 1.0
    physical["rough_motion"]["depth_p95_m"] = lifecycle.UPDATE400_THRESHOLDS_V16[
        "rough_depth_p95_m_strictly_less_than"
    ]
    depth_boundary = lifecycle.evaluate_update400_gate_v16(
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert depth_boundary["passed"] is False


def test_extension_eligibility_rejects_one_failed_control() -> None:
    engine = _engine()
    v12, physical, controls = _inputs()
    controls["wrong_rgb"]["positive"] = False
    decision = lifecycle.evaluate_extension_eligibility_v16(
        v12,
        physical,
        controls,
        integrity_pass=True,
        engine=engine,
    )
    assert decision["passed"] is False
    assert decision["checks"]["all_twelve_causal_control_checks_true"] is False
