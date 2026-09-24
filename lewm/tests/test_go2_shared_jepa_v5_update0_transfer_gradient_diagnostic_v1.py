from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_update0_transfer_gradient_diagnostic_v1 as contract


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_update0_diagnostic_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(sources: dict[str, str]) -> dict:
    return contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/independent_update0_reviewer",
        "reviewed_sources": sources,
        "science_contract": contract.science_contract(),
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def _scope(*, physical: bool = True, jepa: bool = True) -> dict:
    physical_metrics = {
        "pixel_first_hit_balanced_accuracy": 0.99 if physical else 0.20,
        "depth_median_error_m": 0.01,
        "depth_p95_error_m": 0.02,
        "ground_clear_balanced_accuracy": 0.99,
        "distance_group_balanced_accuracy": [0.99],
        "derived_raster_nll": 0.01,
        "derived_raster_balanced_accuracy": 0.99,
        "present_class_recall": {"free": 0.99, "occupied": 0.99},
        "wrong_rgb_pixel_balanced_accuracy_drop": 0.20,
        "wrong_rgb_depth_median_error_increase_m": 0.20,
        "wrong_rgb_depth_p95_error_increase_m": 0.30,
        "wrong_rgb_ground_balanced_accuracy_drop": 0.20,
        "wrong_rgb_raster_nll_increase": 0.20,
        "wrong_rgb_raster_balanced_accuracy_drop": 0.20,
    }
    jepa_metrics = {
        "prediction_valid_cell_count": 10,
        "target_cross_sample_std_mean": 0.10 if jepa else 0.0,
        "target_cross_sample_effective_rank": 5.0,
        "warped_persistence_target_change": 0.1,
        "prediction_to_warped_persistence_ratio": 0.5,
        "wrong_action_advantage_over_target_change": 0.2,
        "wrong_commanded_delta_advantage_over_target_change": 0.2,
        "wrong_action_prediction_sensitivity": 0.2,
        "wrong_commanded_delta_prediction_sensitivity": 0.2,
    }
    return {"physical": physical_metrics, "jepa": jepa_metrics}


def test_exact_source_closure_is_25_v4_plus_audit_and_three_new_sources() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert len(contract._v4.SOURCE_PATHS) == 25
    assert len(bindings) == 29
    assert bindings[contract.V4_TERMINAL_AUDIT_RELATIVE_PATH] == contract.V4_TERMINAL_AUDIT_BINDING["file_sha256"]
    assert set(contract.V4_SOURCE_SHA256.items()) <= set(bindings.items())


def test_runner_import_is_stdlib_only_and_defers_exact_overlay_and_torch() -> None:
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"import importlib.util,sys; p={str(path)!r}; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules; assert callable(m._load_v4_stack)"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = inspect.getsource(_runner().run_parent)
    assert source.index("_reserve(") < source.index("_predecessor_inputs(") < source.index("_rehash_v4_terminal(")


def test_review_validation_is_exact_and_source_only() -> None:
    sources = {"a": "0" * 64}
    value = _review(sources)
    assert contract.validate_review(value, expected_sources=sources) == value
    changed = {**value, "source_only": False}
    with pytest.raises(PermissionError):
        contract.validate_review(changed, expected_sources=sources)


def test_authorization_is_one_narrow_diagnostic_attempt() -> None:
    review = _review({"a": "0" * 64})
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    review_binding = contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, review_raw, content_sha256=review["content_sha256"])
    value = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_development_diagnostic_attempt",
        "authorizer": "/root/update0_authorizer",
        "independent_review": review_binding,
        "predecessor_audit": dict(contract.V4_TERMINAL_AUDIT_BINDING),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    })
    assert contract.validate_authorization(value, review_binding=review_binding) == value
    with pytest.raises(PermissionError):
        contract.validate_authorization({**value, "status": "authorized_training"}, review_binding=review_binding)


def test_update0_uses_scope_evaluator_without_illegal_checkpoint_selector() -> None:
    scopes = {name: _scope() for name in contract._v1.SCOPES}
    scopes["visual_sensor_stress"] = _scope(physical=False, jepa=True)
    scopes["rough_local_dynamics"] = _scope(physical=True, jepa=False)
    result = contract.evaluate_scopes(scopes)
    assert result["physical_pass_count"] == 8
    assert result["jepa_pass_count"] == 8
    assert result["full_gate_pass_count"] == 7
    assert result["all_nine_full_gate_pass"] is False
    with pytest.raises(ValueError, match="checkpoint update"):
        contract._v1.evaluate_checkpoint_candidate({"update": 0, "scopes": scopes})


def test_gradient_interaction_reports_dot_cosine_sum_norm_and_clip() -> None:
    runner = _runner()

    class Vector:
        def __init__(self, values):
            self.values = tuple(float(item) for item in values)

        def square(self):
            return Vector(item * item for item in self.values)

        def sum(self):
            return sum(self.values)

        def __mul__(self, other):
            return Vector(left * right for left, right in zip(self.values, other.values, strict=True))

        def __add__(self, other):
            return Vector(left + right for left, right in zip(self.values, other.values, strict=True))

    names = [prefix + "weight" for prefix in contract.GRADIENT_COMPONENT_PREFIXES.values()]
    camera = {name: Vector([1.0]) for name in names}
    jepa = {name: Vector([1.0]) for name in names}
    result = runner._interaction(camera, jepa)["components_and_global"]["global"]
    assert result["dot"] == 5.0
    assert result["cosine"] == pytest.approx(1.0)
    assert result["camera_plus_jepa_norm"] == pytest.approx(math.sqrt(20.0))
    assert result["camera_plus_jepa_counterfactual_clip_factor"] == pytest.approx(1.0 / (math.sqrt(20.0) + 1e-6))
    assert runner._component("occupancy_head.weight") == "occupancy_head_expected_zero"
    with pytest.raises(PermissionError):
        runner._component("escaped.weight")


def test_warning_collector_accepts_only_exact_v4_grid_warning() -> None:
    collector = contract.CompactDeterminismWarnings()
    collector(contract._v4.GRID_WARNING, UserWarning)
    assert collector.receipt()["warning_count"] == 1
    with pytest.raises(RuntimeError, match="unexpected training warning"):
        collector("different warning", UserWarning)
    with pytest.raises(RuntimeError, match="warning category"):
        collector(contract._v4.GRID_WARNING, RuntimeWarning)


def test_tmp_lifecycle_is_exclusive_canonical_and_failure_is_append_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    review = contract.with_content_sha256({"schema": "synthetic_review"})
    authorization = contract.with_content_sha256({"schema": "synthetic_authorization", "raw": {}, "camera": {}})
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    output = tmp_path / "attempt"
    reservation, reservation_raw = runner._reserve(output, review, review_raw, authorization, authorization_raw, {"synthetic": "0" * 64}, {"synthetic": True})
    assert reservation["status"] == "reserved_before_torch_v4_terminal_camera_raw_or_rgb"
    assert reservation["v4_terminal_opened_before_reservation"] is False
    access, access_raw = runner._publish_json(output / "access.json", {"schema": contract.ACCESS_SCHEMA, "synthetic": True})
    result, result_raw = runner._publish_json(output / "result.json", {"schema": contract.RESULT_SCHEMA, "synthetic": True})
    runner._terminal_failure(output, reservation, reservation_raw, "synthetic_stage", RuntimeError("synthetic"), {"access.json": (access, access_raw), "result.json": (result, result_raw)})
    assert sorted(item.name for item in output.iterdir()) == ["access.json", "failed.json", "reservation.json", "result.json"]
    failed = contract.parse_canonical_json((output / "failed.json").read_bytes(), name="synthetic failure")
    assert failed["published_prefix"] == ["reservation.json", "access.json", "result.json"]
    assert failed["retry_authorized"] is False
    with pytest.raises(RuntimeError, match="already reserved"):
        runner._reserve(output, review, review_raw, authorization, authorization_raw, {}, {})
    failed_root = tmp_path / "reservation_commit_failure"
    original_publish = runner._publish_json
    failed_once = False

    def fail_reservation_once(path, core):
        nonlocal failed_once
        if path.name == "reservation.json" and not failed_once:
            failed_once = True
            raise OSError("synthetic reservation commit failure")
        return original_publish(path, core)

    monkeypatch.setattr(runner, "_publish_json", fail_reservation_once)
    with pytest.raises(OSError, match="synthetic reservation"):
        runner._reserve(failed_root, review, review_raw, authorization, authorization_raw, {}, {})
    assert [item.name for item in failed_root.iterdir()] == ["reservation_failed.json"]
    reservation_failed = contract.parse_canonical_json((failed_root / "reservation_failed.json").read_bytes(), name="synthetic reservation failure")
    assert reservation_failed["status"] == "failed_reservation_commit"
    assert reservation_failed["torch_imported"] is False
    assert reservation_failed["v4_terminal_opened"] is False
    assert reservation_failed["camera_raw_or_rgb_opened"] is False
    assert reservation_failed["retry_authorized"] is False
