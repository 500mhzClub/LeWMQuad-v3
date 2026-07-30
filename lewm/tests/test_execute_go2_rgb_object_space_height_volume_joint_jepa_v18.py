from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

v14 = importlib.import_module(
    "scripts.execute_go2_rgb_unified_ray_survival_joint_jepa_v14"
)
v18 = importlib.import_module(
    "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
)


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


def _controls(value: bool = True) -> dict[str, dict[str, bool]]:
    return {
        name: {check: value for check in v18.CONTROL_CHECK_NAMES}
        for name in v18.CONTROL_NAMES
    }


def test_private_adapter_preserves_public_v14_and_is_denied(capsys) -> None:
    assert v18._base is not v14
    assert v18.PRIVATE_V14_MODULE_NAME not in sys.modules
    assert v14.SCHEMA_PREFIX == "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14"
    assert v14.MODEL_CLASS_NAME == "GeometryAnchoredSweptProgressSurvivalJointJepaV14"
    assert v18.SCHEMA_PREFIX.endswith(
        "object_space_height_volume_joint_jepa_v18_integrity_replacement_v1"
    )
    assert v18.MODEL_CLASS_NAME == "GeometryAnchoredSweptProgressSurvivalJointJepaV18"
    assert v18.main([]) == 4
    denial = v18.validate_content_bound_v18(__import__("json").loads(capsys.readouterr().out))
    assert denial["status"] == "DENIED_SOURCE_ONLY"
    assert denial["scientific_payload_opened"] is False
    assert denial["reservation_created"] is False


def test_model_validator_binds_v18_counts_prefixes_and_native_constants() -> None:
    model_module = importlib.import_module(
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
        "object_space_height_volume"
    )
    receipt = v18.validate_model_api_v18(model_module)
    assert receipt == {
        "model_class": v18.MODEL_CLASS_NAME,
        "method_count": len(v18.MODEL_REQUIRED_METHODS),
        "online_trainable_parameter_count": 3_439_403,
    }
    assert model_module.REPRESENTATION_PARAMETER_PREFIXES_V13 == (
        "bev_lift.point_projection.",
        "bev_lift.volume_block.",
        "semantic_head.",
    )


def test_parent_bindings_include_exact_prereg_and_scientific_witnesses() -> None:
    receipt = v18.validate_bound_sources_v18(ROOT)
    assert receipt["execution_authority_granted"] is False
    expected = {
        v18.PREREGISTRATION_PATH: (
            v18.PREREGISTRATION_FILE_SHA256,
            v18.PREREGISTRATION_BYTE_COUNT,
        ),
        v18.ORIGINAL_V18_PREREGISTRATION_PATH: (
            v18.ORIGINAL_V18_PREREGISTRATION_FILE_SHA256,
            v18.ORIGINAL_V18_PREREGISTRATION_BYTE_COUNT,
        ),
        v18.V18_TERMINAL_FAILURE_RESULT_PATH: (
            v18.V18_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            v18.V18_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        v18.V14_RESULT_PATH: (
            v18.V14_RESULT_FILE_SHA256,
            v18.V14_RESULT_BYTE_COUNT,
        ),
        v18.V15_RESULT_PATH: (
            v18.V15_RESULT_FILE_SHA256,
            v18.V15_RESULT_BYTE_COUNT,
        ),
        v18.V17_RESULT_PATH: (
            v18.V17_RESULT_FILE_SHA256,
            v18.V17_RESULT_BYTE_COUNT,
        ),
    }
    for path, binding in expected.items():
        assert v18.BOUND_PARENT_SOURCES[path] == binding
        assert path in receipt["validated_paths"]


def test_inherited_runtime_executor_surface_is_complete_and_exact() -> None:
    required_constants = {
        "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
        "MODEL_CLASS_NAME",
        "OUTPUT_ROOT_RELATIVE_PATH",
        "REGISTERED_FAMILIES",
        "RUNTIME_INPUT_BINDING_NAMES",
        "SCOPES",
        "V12_GATE_CHECK_NAMES",
    }
    required_callables = {
        "_canonical_json_bytes",
        "_write_immutable_json_v13",
        "flatten_physical_metrics_v13",
        "registered_wrong_rgb_mapping_v13",
        "reserve_attempt_v13",
        "run_future_authorized_engine_v13",
        "terminalize_failure_v13",
        "validate_bound_sources_v13",
        "validate_content_bound_v13",
        "validate_future_execution_prerequisites_v13",
    }
    assert all(hasattr(v18, name) for name in required_constants)
    assert all(callable(getattr(v18, name, None)) for name in required_callables)
    assert (
        v18.registered_wrong_rgb_mapping_v13
        is v18.registered_wrong_rgb_mapping_v18
        is v18._engine.registered_wrong_rgb_mapping_v13
    )
    assert v18.registered_wrong_rgb_mapping_v13.__globals__ is v18._engine.__dict__
    assert (
        v18.flatten_physical_metrics_v13
        is v18.flatten_physical_metrics_v18
        is v18._engine.flatten_physical_metrics_v13
    )
    assert v18.flatten_physical_metrics_v13.__globals__ is v18._engine.__dict__


def test_update400_gate_uses_only_registered_five_authoritative_checks() -> None:
    # Deliberately make all inherited update100->400 direction checks false;
    # V18 update 100 is health-only, while the three fixed comparators pass.
    before = _summary(
        passed=100,
        shortfall=20.0,
        pixel=0.99,
        ground=0.99,
        depth=0.5,
    )
    after = _summary(
        passed=73,
        shortfall=68.0,
        pixel=0.80,
        ground=0.70,
        depth=1.8,
    )
    decision = v18.evaluate_update400_gate_v18(
        before,
        after,
        _controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is True
    assert set(decision["checks"]) == {
        "structural_integrity_pass",
        "all_twelve_causal_control_checks_true",
        "passed_physical_margin_count_strictly_above_matched_update400",
        "total_physical_shortfall_strictly_below_matched_update400",
        "rough_depth_p95_strictly_below_matched_update400",
    }
    assert not all(decision["diagnostic_direction_checks"].values())
    assert decision["next_update"] == 1_000


@pytest.mark.parametrize(
    ("field", "value", "failed"),
    (
        (
            "passed_margin_count",
            72,
            "passed_physical_margin_count_strictly_above_matched_update400",
        ),
        (
            "total_shortfall",
            68.96954700805838,
            "total_physical_shortfall_strictly_below_matched_update400",
        ),
        (
            "depth_p95_m",
            1.8582415819168085,
            "rough_depth_p95_strictly_below_matched_update400",
        ),
    ),
)
def test_update400_comparator_equalities_fail(
    field: str,
    value: int | float,
    failed: str,
) -> None:
    before = _summary(
        passed=50,
        shortfall=100.0,
        pixel=0.60,
        ground=0.55,
        depth=3.0,
    )
    after = _summary(
        passed=73,
        shortfall=68.0,
        pixel=0.80,
        ground=0.70,
        depth=1.8,
    )
    if field == "depth_p95_m":
        after["rough_motion"][field] = value  # type: ignore[index]
    else:
        after[field] = value
    decision = v18.evaluate_update400_gate_v18(
        before,
        after,
        _controls(),
        integrity_pass=True,
    )
    assert decision["checks"][failed] is False
    assert decision["passed"] is False
    assert decision["next_update"] is None


def test_update100_integrity_requires_active_representation_route(monkeypatch) -> None:
    monkeypatch.setattr(
        v18,
        "_original_validate_update_integrity",
        lambda *args, **kwargs: {"base": "pass"},
    )
    result = SimpleNamespace(
        gradient_routes={
            "representation": SimpleNamespace(preclip_l2=0.0),
        }
    )
    with pytest.raises(RuntimeError, match="representation gradient route is zero"):
        v18.validate_update_integrity_v18(
            object(),
            object(),
            result,
            update=100,
            access_receipt={},
        )
    result.gradient_routes["representation"].preclip_l2 = 0.25
    receipt = v18.validate_update_integrity_v18(
        object(),
        object(),
        result,
        update=100,
        access_receipt={},
    )
    assert receipt["v18_height_volume"]["representation_route_active"] is True
