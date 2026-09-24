from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_adaptation_v1 as contract


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_protected_camera_adaptation_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(sources: dict[str, str], *, reviewer: str = "/root/protected_camera_reviewer") -> dict:
    return contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "reviewed_sources": sources,
        "science_contract": contract.science_contract(),
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def _review_binding(review: dict) -> dict:
    raw = contract.canonical_json_bytes(review) + b"\n"
    return contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, raw, content_sha256=review["content_sha256"])


def _authorization(review: dict, *, authorizer: str = "/root/protected_camera_authorizer") -> dict:
    return contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_protected_camera_adaptation_attempt",
        "authorizer": authorizer,
        "independent_review": _review_binding(review),
        "predecessor": contract.predecessor_contract(),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    })


def _physical(*, passes: bool = True) -> dict:
    return {
        "pixel_first_hit_balanced_accuracy": 0.99 if passes else 0.20,
        "ground_clear_balanced_accuracy": 0.99,
        "derived_raster_balanced_accuracy": 0.99,
        "wrong_rgb_pixel_balanced_accuracy_drop": 0.20,
        "wrong_rgb_depth_median_error_increase_m": 0.20,
        "wrong_rgb_depth_p95_error_increase_m": 0.30,
        "wrong_rgb_ground_balanced_accuracy_drop": 0.20,
        "wrong_rgb_raster_nll_increase": 0.20,
        "wrong_rgb_raster_balanced_accuracy_drop": 0.20,
        "depth_median_error_m": 0.01,
        "depth_p95_error_m": 0.02,
        "derived_raster_nll": 0.01,
        "distance_group_balanced_accuracy": [0.99] * 6,
        "present_class_recall": {"free": 0.99, "occupied": 0.99, "unknown": 0.99},
    }


def test_exact_source_closure_adds_only_audit_and_three_new_sources() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert len(contract._diagnostic.SOURCE_PATHS) == 29
    assert len(bindings) == 33
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert bindings[contract.UPDATE0_AUDIT_RELATIVE_PATH] == contract.UPDATE0_AUDIT_BINDING["file_sha256"]
    assert set(contract.DIAGNOSTIC_SOURCE_SHA256.items()) <= set(bindings.items())


def test_runner_import_defers_torch_payload_and_terminals_until_reservation() -> None:
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"import importlib.util,sys; p={str(path)!r}; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = inspect.getsource(_runner().run_parent)
    assert source.index("_reserve(") < source.index("_update0_terminal(") < source.index("_load_diagnostic_runner(") < source.index("_load_runtime(")


def test_science_contract_is_one_bounded_camera_only_attempt() -> None:
    science = contract.science_contract()
    optimizer = science["optimizer"]
    assert science["schedule"]["use_exact_prefix_updates"] == 4_000
    assert science["schedule"]["presentation_count"] == 64_000
    assert science["schedule"]["checkpoint_updates"] == [100, 400, 1_000, 2_000, 4_000]
    assert science["camera_loss"]["jepa_objective_count"] == 0
    assert science["camera_loss"]["jepa_backward_count"] == 0
    assert optimizer["group_order"] == ["evidence_head", "encoder"]
    assert optimizer["encoder_learning_rate_scale"] == 0.01
    assert optimizer["independent_group_clip_norm"] == 1.0
    assert optimizer["post_clip_norm_assertion_tolerance"] == 1e-5
    assert science["expected_parameter_tensor_counts"] == {"encoder": 78, "evidence_head": 14}
    assert all(value is False for value in contract.DOWNSTREAM_DENIALS.values())
    assert contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[4_000] == "14e83952c758c2ee4118d38c116625feb351813bc24b017d7b47f53426df47ab"
    head, encoder = contract.learning_rates(1)
    assert encoder == pytest.approx(0.01 * head)
    with pytest.raises(ValueError):
        contract.learning_rates(0)
    with pytest.raises(ValueError):
        contract.learning_rates(4_001)


def test_parameter_partition_is_exhaustive_and_rejects_escape() -> None:
    expected = {
        "encoder.weight": "encoder",
        "evidence_head.weight": "evidence_head",
        "bev_decoder.weight": "bev_decoder",
        "predictor.weight": "predictor",
        "occupancy_head.weight": "occupancy_head",
        "target_encoder.weight": "target_encoder",
        "target_bev_decoder.weight": "target_bev_decoder",
    }
    assert {name: contract.parameter_partition(name) for name in expected} == expected
    with pytest.raises(ValueError, match="escaped"):
        contract.parameter_partition("new_architecture.weight")
    with pytest.raises(ValueError):
        contract.parameter_partition("encoder")


def test_physical_gate_uses_exact_nine_scope_v1_margins() -> None:
    scopes = {name: _physical() for name in contract._v1.SCOPES}
    result = contract.evaluate_physical_scopes(scopes)
    assert result["physical_pass_count"] == 9
    assert result["all_nine_physical_pass"] is True
    scopes["visual_sensor_stress"] = _physical(passes=False)
    failed = contract.evaluate_physical_scopes(scopes)
    assert failed["physical_pass_count"] == 8
    assert failed["all_nine_physical_pass"] is False
    assert failed["scope_evaluations"]["visual_sensor_stress"]["passes"] is False
    with pytest.raises(ValueError, match="order"):
        contract.evaluate_physical_scopes(dict(reversed(list(scopes.items()))))


def test_bound_update0_terminal_audit_validates_and_tamper_fails() -> None:
    raw = (ROOT / contract.UPDATE0_AUDIT_RELATIVE_PATH).read_bytes()
    value = contract.validate_update0_audit(raw)
    assert value["content_sha256"] == contract.UPDATE0_AUDIT_BINDING["content_sha256"]
    assert value["successor_decision"]["decision"] == "SEPARATE_CAMERA_ADAPTATION_BEFORE_FROZEN_CAMERA_JEPA_TRAINING"
    with pytest.raises(PermissionError, match="byte binding"):
        contract.validate_update0_audit(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_review_validation_binds_exact_sources_and_has_no_authority() -> None:
    sources = {"synthetic.py": "0" * 64}
    review = _review(sources)
    assert contract.validate_review(review, expected_sources=sources) == review
    assert review["authority"]["output_root_observed_absent_at_authorization"] is False
    with pytest.raises(PermissionError):
        contract.validate_review(review, expected_sources={"synthetic.py": "1" * 64})
    with pytest.raises(PermissionError):
        contract.validate_review({**review, "source_only": False}, expected_sources=sources)


def test_authorization_is_identity_separated_and_exactly_one_attempt() -> None:
    review = _review({"synthetic.py": "0" * 64})
    authorization = _authorization(review)
    binding = _review_binding(review)
    assert contract.validate_authorization(authorization, review_binding=binding, reviewer=review["reviewer"]) == authorization
    assert authorization["authority"]["output_root_observed_absent_at_authorization"] is True
    assert authorization["authority"]["requires_absent_output_root_before_reservation"] is True
    for identity in (contract.IMPLEMENTATION_AUTHOR, review["reviewer"]):
        changed = {**authorization, "authorizer": identity}
        with pytest.raises(PermissionError):
            contract.validate_authorization(changed, review_binding=binding, reviewer=review["reviewer"])


def test_reservation_is_exclusive_fail_closed_and_records_preopen_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    review = contract.with_content_sha256({"reviewed_sources": {}})
    authorization = contract.with_content_sha256({"authority": {"output_root_observed_absent_at_authorization": True}})
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    output = tmp_path / "attempt"
    reservation, _ = runner._reserve(output, review, review_raw, authorization, authorization_raw, {"torch_module_absent": True})
    assert reservation["output_root_absent_before_reservation"] is True
    assert reservation["torch_imported_before_reservation"] is False
    assert reservation["v4_terminal_opened_before_reservation"] is False
    assert reservation["update0_terminal_opened_before_reservation"] is False
    assert reservation["camera_raw_or_rgb_opened_before_reservation"] is False
    with pytest.raises(RuntimeError, match="already reserved"):
        runner._reserve(output, review, review_raw, authorization, authorization_raw, {})
    dangling = tmp_path / "dangling_artifacts"
    dangling.mkdir()
    synthetic_reservation, _ = runner._publish_json(dangling / "reservation.json", {"schema": contract.RESERVATION_SCHEMA, "attempt_identity": "synthetic"})
    runner._write_exclusive(dangling / "checkpoints/update_100.pt", b"synthetic checkpoint")
    runner._write_exclusive(dangling / "training_trace.jsonl", b'{"update":100}\n')
    runner._terminal_failure(dangling, synthetic_reservation, "synthetic_stage", RuntimeError("synthetic"), {})
    failed = contract.parse_canonical_json((dangling / "failed.json").read_bytes(), name="dangling failure")
    assert failed["published_prefix"] == ["reservation.json", "checkpoints/update_100.pt", "training_trace.jsonl"]
    assert set(failed["artifacts"]) == {"reservation.json", "checkpoints/update_100.pt", "training_trace.jsonl"}
    assert failed["artifacts"]["reservation.json"]["content_sha256"] == synthetic_reservation["content_sha256"]
    assert failed["all_existing_regular_artifacts_bound"] is True
    assert failed["caller_ledger_paths"] == []
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
        runner._reserve(failed_root, review, review_raw, authorization, authorization_raw, {})
    assert [item.name for item in failed_root.iterdir()] == ["reservation_failed.json"]
    failure = contract.parse_canonical_json((failed_root / "reservation_failed.json").read_bytes(), name="reservation failure")
    assert failure["status"] == "failed_reservation_commit"
    assert failure["torch_imported"] is False
    assert failure["v4_terminal_opened"] is False
    assert failure["update0_terminal_opened"] is False
    assert failure["camera_raw_or_rgb_opened"] is False
    assert failure["retry_authorized"] is False


def test_runner_statically_enforces_camera_only_two_group_training_and_hard_failure() -> None:
    source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text(encoding="utf-8")
    train_source = inspect.getsource(_runner()._train)
    pair_source = inspect.getsource(_runner()._camera_pair)
    assert source.count("clip_grad_norm_(") == 2
    assert "clip_grad_norm_(model.parameters()" not in source
    assert "combine_joint_losses" not in source
    assert "forward_training_pair" not in source
    assert "update_ema_target_after_optimizer_step" not in source
    assert "optimizer_state_dict" not in source
    assert "observable_camera_ray_v4_loss_v4" in train_source
    assert "camera.total / 4.0" in train_source
    assert "jepa=None" in pair_source
    assert '"jepa_objective_count": 0' in source
    assert '"global_clip_invocation_count": 0' in source
    assert '"failed_numeric_physical_gate"' in source
    assert "no fixed checkpoint passed all nine physical scopes by update 4000" in source
    assert "closest_or_soft_promotion" in source
    assert "return 2" in inspect.getsource(_runner().run_parent)

    class Truth:
        def __init__(self, value):
            self.value = bool(value)

        def all(self):
            return self

        def item(self):
            return self.value

    class Gradient:
        def __init__(self, value):
            self.value = float(value)

        def detach(self):
            return self

        def float(self):
            return self

        def square(self):
            return Gradient(self.value * self.value)

        def sum(self):
            return self

        def cpu(self):
            return self

        def __float__(self):
            return self.value

    class Torch:
        @staticmethod
        def isfinite(value):
            return Truth(math.isfinite(value.value))

        @staticmethod
        def stack(values):
            class Stack:
                def __init__(self, items):
                    self.items = items

                def all(self):
                    return Truth(all(item.value for item in self.items))

                def sum(self):
                    return Gradient(sum(item.value for item in self.items))

            return Stack(list(values))

    class Runtime:
        torch = Torch()

    class Parameter:
        def __init__(self, gradient):
            self.grad = gradient

    head = [Parameter(Gradient(0.1)) for _ in range(14)]
    assert _runner()._gradient_group_norm(Runtime(), head, "evidence_head", maximum=1.0) == pytest.approx(math.sqrt(0.14))
    head[0].grad = None
    with pytest.raises(RuntimeError, match="no gradient"):
        _runner()._gradient_group_norm(Runtime(), head, "evidence_head")
    head[0].grad = Gradient(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _runner()._gradient_group_norm(Runtime(), head, "evidence_head")
    with pytest.raises(RuntimeError, match="tensor count"):
        _runner()._gradient_group_norm(Runtime(), head[:-1], "evidence_head")
    within = [Parameter(Gradient((1.0 + 0.5e-5) / math.sqrt(14.0))) for _ in range(14)]
    assert _runner()._gradient_group_norm(Runtime(), within, "evidence_head", maximum=1.0) == pytest.approx(1.0 + 0.5e-5)
    too_large = [Parameter(Gradient((1.0 + 2e-5) / math.sqrt(14.0))) for _ in range(14)]
    with pytest.raises(RuntimeError, match="post-clip"):
        _runner()._gradient_group_norm(Runtime(), too_large, "evidence_head", maximum=1.0)
