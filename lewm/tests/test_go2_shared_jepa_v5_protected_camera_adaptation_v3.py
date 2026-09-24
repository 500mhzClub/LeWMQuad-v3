from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_adaptation_v3 as contract


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_protected_camera_adaptation_v3_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(sources: dict[str, str], *, reviewer: str = "/root/protected_camera_v3_reviewer") -> dict:
    return contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "reviewed_sources": sources,
        "predecessor": contract.predecessor_contract(),
        "science_contract": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def _review_binding(review: dict) -> dict:
    raw = contract.canonical_json_bytes(review) + b"\n"
    return contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, raw, content_sha256=review["content_sha256"])


def _authorization(review: dict, *, authorizer: str = "/root/protected_camera_v3_authorizer") -> dict:
    return contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_final_protected_camera_adaptation_v3_attempt",
        "authorizer": authorizer,
        "independent_review": _review_binding(review),
        "predecessor": contract.predecessor_contract(),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
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


def _metric(update: int, *, passes: bool = True) -> dict:
    scopes = {name: _physical(passes=passes) for name in contract._v1.SCOPES}
    return {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": 495,
        "unique_endpoint_count": 924,
        "scopes": scopes,
        "aggregate_complete_v4_loss": 1.0,
        "evaluation": contract.evaluate_physical_scopes(scopes),
        "state_sha256_before": "0" * 64,
        "state_sha256_after": "0" * 64,
        "frozen_state_sha256_before_and_after": "1" * 64,
        "state_mutation_count": 0,
    }


def _checkpoint(update: int) -> dict:
    return {
        "path": f"checkpoints/update_{update}.pt",
        "file_sha256": "2" * 64,
        "content_sha256": "3" * 64,
        "byte_count": 123,
        "state_sha256": "4" * 64,
        "frozen_state_sha256": "5" * 64,
        "trainable_state_sha256": "6" * 64,
    }


def _leaf_differences(left, right, path=()):
    if type(left) is dict and type(right) is dict and set(left) == set(right):
        result = []
        for key in sorted(left):
            result.extend(_leaf_differences(left[key], right[key], (*path, key)))
        return result
    if type(left) is list and type(right) is list and len(left) == len(right):
        result = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            result.extend(_leaf_differences(a, b, (*path, index)))
        return result
    return [] if left == right else [(path, left, right)]


def _decision(update: int, passed: int, shortfall: float, worst: float, loss: float, *, all_nine: bool = False) -> dict:
    return contract.control_decision_from_progress(
        update=update,
        passed_margin_count=passed,
        total_shortfall=shortfall,
        worst_margin=worst,
        aggregate_complete_v4_loss=loss,
        all_nine_physical_pass=all_nine,
    )


def test_exact_v2_v1_derivation_source_closure_and_terminal_precondition() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert set(contract.V2_SOURCE_SHA256.items()) <= set(bindings.items())
    assert set(contract.V1_SOURCE_SHA256.items()) <= set(bindings.items())
    assert bindings[contract.V2_TERMINAL_AUDIT_RELATIVE_PATH] == contract.V2_TERMINAL_AUDIT_BINDING["file_sha256"]
    raw = (ROOT / contract.V2_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    audit = contract.validate_v2_terminal_audit(raw)
    assert audit["verdict"] == contract.V2_TERMINAL_AUDIT_BINDING["verdict"]
    assert audit["successor_decision"]["qualified_camera_checkpoint_exists"] is False
    assert audit["successor_decision"]["frozen_camera_jepa_training_may_start"] is False
    assert audit["successor_decision"]["retry_or_extension_authorized"] is False
    with pytest.raises(PermissionError, match="byte binding"):
        contract.validate_v2_terminal_audit(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_encoder_lr_scale_is_the_only_science_leaf_delta_and_starts_update0() -> None:
    differences = _leaf_differences(contract._v2_contract.science_contract(), contract.science_contract())
    assert differences == [(("optimizer", "encoder_learning_rate_scale"), 0.10, 1.0)]
    assert contract.science_delta()["other_science_changes"] == []
    for update in (1, 100, 400, 1_000, 2_000, 4_000):
        v2_head, v2_encoder = contract._v2_contract.learning_rates(update)
        head, encoder = contract.learning_rates(update)
        assert head == v2_head
        assert encoder == pytest.approx(10.0 * v2_encoder)
        assert encoder == pytest.approx(head)
    assert contract.OPTIMIZER_CONTRACT == {**contract._v2_contract.OPTIMIZER_CONTRACT, "encoder_learning_rate_scale": 1.0}
    assert contract.science_contract()["initial_state_sha256"] == "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"
    assert contract.science_contract()["optimizer"]["maximum_updates"] == 4_000


def test_update_1000_stop_boundary_is_exact_conjunction() -> None:
    threshold = contract.UPDATE1000_SHORTFALL_STOP_ABOVE
    assert _decision(1_000, 79, threshold, -20.0, 9.0)["action"] == contract.CONTROL_ACTION_CONTINUE
    stopped = _decision(1_000, 79, math.nextafter(threshold, math.inf), -20.0, 9.0)
    assert stopped["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert stopped["terminal_stage"] == "predeclared_numeric_progress_cutoff_at_update_1000"
    assert _decision(1_000, 80, threshold * 2.0, -20.0, 9.0)["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _decision(1_000, 0, threshold, -20.0, 9.0)["action"] == contract.CONTROL_ACTION_CONTINUE


def test_update_2000_continue_boundary_requires_weak_pareto_with_one_strict_and_guards() -> None:
    p = contract.UPDATE2000_PASSED_MARGIN_FLOOR
    s = contract.UPDATE2000_SHORTFALL_CEILING
    w = contract.UPDATE2000_WORST_MARGIN_FLOOR
    loss = contract.UPDATE2000_LOSS_CEILING
    assert _decision(2_000, p, s, w, loss)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _decision(2_000, p + 1, s, w, loss)["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _decision(2_000, p, math.nextafter(s, -math.inf), w, loss)["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _decision(2_000, p - 1, s - 1.0, w, loss)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _decision(2_000, p + 1, math.nextafter(s, math.inf), w, loss)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _decision(2_000, p + 1, s, math.nextafter(w, -math.inf), loss)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _decision(2_000, p + 1, s, w, math.nextafter(loss, math.inf))["action"] == contract.CONTROL_ACTION_STOP_PROGRESS


def test_all_nine_overrides_controls_and_update_4000_never_soft_qualifies() -> None:
    for update in contract.CHECKPOINT_UPDATES:
        assert _decision(update, 0, 1_000.0, -1_000.0, 1_000.0, all_nine=True)["action"] == contract.CONTROL_ACTION_QUALIFY
    assert _decision(100, 0, 1_000.0, -1_000.0, 1_000.0)["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _decision(400, 0, 1_000.0, -1_000.0, 1_000.0)["action"] == contract.CONTROL_ACTION_CONTINUE
    final = _decision(4_000, 188, 0.0001, -0.0001, 0.0001)
    assert final["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    assert final["qualifies"] is False
    with pytest.raises(FloatingPointError):
        _decision(1_000, 79, float("nan"), -1.0, 1.0)


def test_reporting_review_and_authorization_bind_exact_control_and_deny_downstream() -> None:
    reporting = contract.reporting_contract()
    assert reporting["fixed_checkpoint_updates"] == [100, 400, 1_000, 2_000, 4_000]
    assert reporting["one_inline_physical_evaluation_per_published_sidecar"] is True
    assert reporting["read_only_observers_must_not_rerun_evaluation"] is True
    assert reporting["metric_controlled_stop_other_than_earliest_all_nine_pass"] is True
    assert reporting["numeric_progress_cutoff_updates"] == [1_000, 2_000]
    sources = {"synthetic.py": "0" * 64}
    review = _review(sources)
    assert contract.validate_review(review, expected_sources=sources) == review
    authorization = _authorization(review)
    assert contract.validate_authorization(authorization, review_binding=_review_binding(review), reviewer=review["reviewer"]) == authorization
    for key in ("jepa_training_authorized", "g2_authorized", "navigation_authorized", "heldout_authorized", "retry_authorized", "training_extension_authorized", "soft_promotion_authorized"):
        assert authorization["authority"][key] is False
    assert authorization["authority"]["mutation_scope"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    with pytest.raises(PermissionError):
        contract.validate_review({**review, "control_contract": {}}, expected_sources=sources)


def test_runner_import_defers_torch_and_binds_exact_v2_runner() -> None:
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"import importlib.util,sys; p={str(path)!r}; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert hashlib.sha256((ROOT / contract.V2_RUNNER_RELATIVE_PATH).read_bytes()).hexdigest() == contract.V2_SOURCE_SHA256[contract.V2_RUNNER_RELATIVE_PATH]


def test_v3_contract_covers_every_attribute_used_by_rebound_v1_and_v2_runners() -> None:
    required = set()
    for relative in (contract.V1_RUNNER_RELATIVE_PATH, contract.V2_RUNNER_RELATIVE_PATH):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        required.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "contract"
        )
    assert sorted(name for name in required if not hasattr(contract, name)) == []


def test_actual_v2_terminal_mixed_artifact_groups_flatten_and_rehash_all_exact_files() -> None:
    runner = _runner()
    audit, records = runner._v2_terminal()
    assert audit["content_sha256"] == contract.V2_TERMINAL_AUDIT_BINDING["content_sha256"]
    assert len(records) == len(contract.V2_TERMINAL_EXACT_PATHS) == 15
    assert [item["path"] for item in records] == list(contract.V2_TERMINAL_EXACT_PATHS)
    assert {item["kind"] for item in records} == {
        "access", "checkpoint_metrics", "checkpoints", "failed", "reservation", "sidecars", "training_trace",
    }
    assert sum(item["kind"] == "checkpoints" for item in records) == 5
    assert sum(item["kind"] == "sidecars" for item in records) == 5


def test_v2_terminal_artifact_flatten_rejects_malformed_and_duplicate_paths() -> None:
    runner = _runner()
    binding = {"path": "one.json", "file_sha256": "0" * 64, "byte_count": 1}
    assert runner._flatten_terminal_artifact_bindings({"single": binding, "many": [{**binding, "path": "two.json"}]}) == [
        {"kind": "single", **binding},
        {"kind": "many", **binding, "path": "two.json"},
    ]
    with pytest.raises(PermissionError, match="group is malformed"):
        runner._flatten_terminal_artifact_bindings({"bad": []})
    with pytest.raises(PermissionError, match="malformed or duplicated"):
        runner._flatten_terminal_artifact_bindings({"a": binding, "b": [dict(binding)]})
    with pytest.raises(PermissionError, match="binding is malformed"):
        runner._flatten_terminal_artifact_bindings({"bad": [{**binding, "kind": "spoofed"}]})


def test_sidecar_is_canonical_atomic_exclusive_read_only_and_binds_control(tmp_path: Path) -> None:
    runner = _runner()
    metric = _metric(100, passes=False)
    checkpoint = _checkpoint(100)
    binding = runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=checkpoint, metric=metric)
    path = tmp_path / binding["path"]
    raw = path.read_bytes()
    value = contract.parse_canonical_json(raw, name="test sidecar")
    assert raw == contract.canonical_json_bytes(value) + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert not list((tmp_path / "checkpoints").glob(".*.publishing"))
    assert value["continuation"] == contract.checkpoint_control_decision(metric)
    assert value["continuation"]["action"] == contract.CONTROL_ACTION_CONTINUE
    assert contract.validate_metric_sidecar(value, update=100, checkpoint=checkpoint, metric=metric) == value
    with pytest.raises(FileExistsError):
        runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=checkpoint, metric=metric)
    tampered = {**value, "continuation": {**value["continuation"], "action": contract.CONTROL_ACTION_QUALIFY}}
    with pytest.raises(PermissionError):
        contract.validate_metric_sidecar(tampered)


def test_checkpoint_order_is_snapshot_evaluate_publish_then_only_predeclared_branch() -> None:
    runner = _runner()
    source = inspect.getsource(runner._train)
    checkpoint_region = source[source.index("checkpoint = _v1_runner._snapshot") :]
    assert checkpoint_region.index("_snapshot") < checkpoint_region.index("_evaluate") < checkpoint_region.index("_publish_metric_sidecar") < checkpoint_region.index("metrics.append") < checkpoint_region.index("checkpoint_control_decision")
    assert source.count("_v1_runner._evaluate") == 1
    assert "physical_pass_count" not in source
    assert "observable_camera_ray_v4_loss_v4" in source
    assert "update_ema_target_after_optimizer_step" not in source
    assert "combine_joint_losses" not in source
    assert "_BASE_TRAIN" not in source


def test_final_metrics_collate_existing_sidecars_without_evaluation(tmp_path: Path) -> None:
    runner = _runner()
    metric = _metric(100, passes=False)
    sidecar = runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=_checkpoint(100), metric=metric)
    decision = contract.checkpoint_control_decision(metric)
    runner._ACTIVE_SIDECARS[:] = [sidecar]
    runner._ACTIVE_CONTROL_DECISIONS[:] = [decision]
    runner._ACTIVE_TERMINAL_CONTROL = None
    assert "_evaluate" not in inspect.getsource(runner._publish_training)
    _, metrics_binding = runner._publish_training(tmp_path, [{"schema": "synthetic_trace", "update": 1}], [metric])
    final = contract.parse_canonical_json((tmp_path / "checkpoint_metrics.json").read_bytes(), name="final metrics")
    assert final["rows"] == [metric]
    assert final["sidecars"] == [sidecar]
    assert final["checkpoint_controls"] == [decision]
    assert final["inline_evaluation_count"] == 1
    assert final["observer_evaluation_rerun_count"] == 0
    assert metrics_binding["content_sha256"] == final["content_sha256"]


def test_progress_cutoff_failure_is_explicit_bound_and_has_no_downstream_authority(tmp_path: Path) -> None:
    runner = _runner()
    reservation, _ = runner._publish_json(tmp_path / "reservation.json", {"schema": contract.RESERVATION_SCHEMA, "attempt_identity": "synthetic"})
    runner._write_exclusive(tmp_path / "checkpoints/update_1000.pt", b"synthetic")
    metric = _metric(1_000, passes=False)
    sidecar = runner._publish_metric_sidecar(tmp_path, update=1_000, checkpoint=_checkpoint(1_000), metric=metric)
    decision = _decision(1_000, 79, math.nextafter(contract.UPDATE1000_SHORTFALL_STOP_ABOVE, math.inf), -9.0, 1.0)
    runner._ACTIVE_TERMINAL_CONTROL = decision
    runner._terminal_failure(tmp_path, reservation, "scientific_numeric_physical_gate", RuntimeError("generic base message"), {sidecar["path"]: sidecar}, numeric=True)
    failed = contract.parse_canonical_json((tmp_path / "failed.json").read_bytes(), name="test failure")
    assert failed["status"] == "failed_predeclared_numeric_progress_cutoff"
    assert failed["stage"] == "predeclared_numeric_progress_cutoff_at_update_1000"
    assert failed["checkpoint_control"] == decision
    assert failed["numeric_progress_cutoff_applied"] is True
    assert failed["caller_error"] == {
        "type": "BaseLifecycleSelectedCheckpointAbsentTrigger",
        "message": "base lifecycle requested numeric terminalization after the predeclared stop at update 1000; update 4000 was not reached",
    }
    assert "by update 4000" not in failed["caller_error"]["message"]
    assert failed["downstream_authority_granted"] is False
    assert failed["extension_or_retry_authorized"] is False
    assert sidecar["path"] in failed["artifacts"]
    assert failed["exact_terminal_directories_including_root"] == [".", "checkpoints"]


def test_runner_hooks_are_narrow_restore_exact_v2_lifecycle_and_rehash_v2_then_v1() -> None:
    runner = _runner()
    source = inspect.getsource(runner.run_parent)
    for hook in ("_train", "_publish_training", "_update0_terminal_with_v1", "_access_receipt", "_terminal_failure"):
        assert f"_v2.{hook} =" in source
    assert "finally:" in source
    assert "_BASE_V2_RUN_PARENT" in source
    update0_source = inspect.getsource(runner._update0_terminal_with_v2)
    assert update0_source.index("_v2_terminal") < update0_source.index("_BASE_V2_UPDATE0_TERMINAL")
    assert contract.TERMINAL_DIRECTORIES_INCLUDING_ROOT == (".", "checkpoints")
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("protected_camera_adaptation_v3")
