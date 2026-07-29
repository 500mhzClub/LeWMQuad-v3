from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as v13,
)
from scripts import execute_go2_rgb_unified_ray_survival_joint_jepa_v14 as v14


ROOT = Path(__file__).resolve().parents[2]


def _summary(
    *,
    passed: int,
    shortfall: float,
    pixel: float,
    ground: float,
    depth: float,
) -> dict[str, object]:
    return {
        "complete_physical_scope_count": 0,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -0.5,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _positive_controls() -> dict[str, dict[str, bool]]:
    return {
        name: {check: True for check in v14.CONTROL_CHECK_NAMES}
        for name in v14.CONTROL_NAMES
    }


def test_private_adapter_preserves_process_global_v13_defaults() -> None:
    assert v14._engine is not v13
    assert v14.PRIVATE_V13_MODULE_NAME not in sys.modules
    assert v13.SCHEMA_PREFIX == (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13"
    )
    assert v13.PREREGISTRATION_COMMIT == (
        "a285129651a0c418467d95c7c1e3d7a1767453d2"
    )
    assert v13.MODEL_CLASS_NAME == (
        "GeometryAnchoredSweptProgressSurvivalJointJepaV13"
    )
    assert v13.MATCHED_UPDATE400_THRESHOLDS is None
    assert v14.SCHEMA_PREFIX == "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14"
    assert v14.PREREGISTRATION_COMMIT == (
        "456d864b9e03a46f3f79ef413a1bd29ae88b6ace"
    )
    assert v14.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_rgb_unified_ray_survival_joint_jepa_v14/attempt_v1"
    )
    assert v14.MODEL_CLASS_NAME == (
        "GeometryAnchoredSweptProgressSurvivalJointJepaV14"
    )
    receipt = v14.private_adapter_receipt_v14()
    assert receipt["public_v13_loaded_by_adapter"] is False
    assert receipt["private_module_registered"] is False
    assert receipt["execution_authorized"] is False


def test_private_launcher_compatibility_hooks_are_reexported() -> None:
    pairs = (
        ("validate_content_bound_v13", "validate_content_bound_v14"),
        ("validate_bound_sources_v13", "validate_bound_sources_v14"),
        (
            "validate_future_execution_prerequisites_v13",
            "validate_future_execution_prerequisites_v14",
        ),
        ("reserve_attempt_v13", "reserve_attempt_v14"),
        ("terminalize_failure_v13", "terminalize_failure_v14"),
        ("flatten_physical_metrics_v13", "flatten_physical_metrics_v14"),
        ("registered_wrong_rgb_mapping_v13", "registered_wrong_rgb_mapping_v14"),
        ("run_future_authorized_engine_v13", "run_future_authorized_engine_v14"),
    )
    for compatibility_name, v14_name in pairs:
        assert getattr(v14, compatibility_name) is getattr(v14, v14_name)
    assert v14._write_immutable_json_v13.__globals__ is v14._engine.__dict__
    assert v14.run_future_authorized_engine_v13.__globals__ is v14._engine.__dict__


def test_v14_preregistration_binding_replaces_only_v13_preregistration() -> None:
    old_path = v13.PREREGISTRATION_PATH
    assert old_path not in v14.BOUND_PARENT_SOURCES
    assert v14.BOUND_PARENT_SOURCES[v14.PREREGISTRATION_PATH] == (
        v14.PREREGISTRATION_FILE_SHA256,
        v14.PREREGISTRATION_BYTE_COUNT,
    )
    assert {
        path: binding
        for path, binding in v13.BOUND_PARENT_SOURCES.items()
        if path != old_path
    } == {
        path: binding
        for path, binding in v14.BOUND_PARENT_SOURCES.items()
        if path != v14.PREREGISTRATION_PATH
    }
    receipt = v14.validate_bound_sources_v14(ROOT)
    assert receipt["schema"] == f"{v14.SCHEMA_PREFIX}_parent_source_validation_v1"
    assert v14.PREREGISTRATION_PATH in receipt["validated_paths"]
    assert old_path not in receipt["validated_paths"]
    assert receipt["execution_authority_granted"] is False


def test_v13_default_update400_gate_has_no_successor_checks() -> None:
    before = _summary(
        passed=60,
        shortfall=80.0,
        pixel=0.70,
        ground=0.60,
        depth=2.1,
    )
    after = _summary(
        passed=61,
        shortfall=79.0,
        pixel=0.71,
        ground=0.61,
        depth=2.0,
    )
    decision = v13.evaluate_update400_gate_v13(
        before,
        after,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is True
    assert len(decision["checks"]) == 5
    assert "matched_update400_thresholds" not in decision


def test_v14_update400_gate_requires_all_three_matched_beats() -> None:
    before = _summary(
        passed=60,
        shortfall=80.0,
        pixel=0.70,
        ground=0.60,
        depth=2.1,
    )
    after = _summary(
        passed=72,
        shortfall=71.0,
        pixel=0.71,
        ground=0.61,
        depth=1.9,
    )
    decision = v14.evaluate_update400_gate_v14(
        before,
        after,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is True
    assert decision["matched_update400_thresholds"] == (
        v14.MATCHED_UPDATE400_THRESHOLDS
    )
    assert len(decision["checks"]) == 8
    assert all(
        decision["checks"][name]
        for name in (
            "passed_physical_margin_count_strictly_above_matched_update400",
            "total_physical_shortfall_strictly_below_matched_update400",
            "rough_depth_p95_strictly_below_matched_update400",
        )
    )
    assert v14.evaluate_update400_gate_v14.__globals__ is (
        v14.run_future_authorized_engine_v14.__globals__
    )
    assert v14.run_future_authorized_engine_v14.__globals__[
        "MATCHED_UPDATE400_THRESHOLDS"
    ] == v14.MATCHED_UPDATE400_THRESHOLDS


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        (
            "passed_margin_count",
            71,
            "passed_physical_margin_count_strictly_above_matched_update400",
        ),
        (
            "total_shortfall",
            71.67935936391197,
            "total_physical_shortfall_strictly_below_matched_update400",
        ),
        (
            "depth_p95_m",
            1.936374711990354,
            "rough_depth_p95_strictly_below_matched_update400",
        ),
    ),
)
def test_v14_matched_update400_threshold_equalities_fail_strictly(
    field: str,
    value: int | float,
    failed_check: str,
) -> None:
    before = _summary(
        passed=60,
        shortfall=80.0,
        pixel=0.70,
        ground=0.60,
        depth=2.1,
    )
    after = _summary(
        passed=72,
        shortfall=71.0,
        pixel=0.71,
        ground=0.61,
        depth=1.9,
    )
    if field == "depth_p95_m":
        after["rough_motion"][field] = value  # type: ignore[index]
    else:
        after[field] = value
    decision = v14.evaluate_update400_gate_v14(
        before,
        after,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["checks"][failed_check] is False
    assert decision["passed"] is False
    assert decision["next_update"] is None


def test_v14_model_validator_uses_v14_class_and_counts() -> None:
    from lewm.models import (
        geometry_anchored_swept_progress_survival_joint_jepa_v14_unified_ray_survival
        as model_module,
    )

    receipt = v14.validate_model_api_v14(model_module)
    assert receipt == {
        "model_class": v14.MODEL_CLASS_NAME,
        "method_count": len(v14.MODEL_REQUIRED_METHODS),
        "online_trainable_parameter_count": 3_383_917,
    }


def test_denial_shell_uses_only_v14_identity(capsys: pytest.CaptureFixture[str]) -> None:
    assert v14.main([]) == 4
    receipt = v14.validate_content_bound_v14(
        __import__("json").loads(capsys.readouterr().out)
    )
    assert receipt["schema"] == f"{v14.SCHEMA_PREFIX}_current_execution_denial_v1"
    assert receipt["preregistration_commit"] == v14.PREREGISTRATION_COMMIT
    assert receipt["output_root"] == v14.OUTPUT_ROOT_RELATIVE_PATH
    assert receipt["scientific_payload_opened"] is False
    assert receipt["reservation_created"] is False
