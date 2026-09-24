from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_adaptation_v2 as contract


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_protected_camera_adaptation_v2_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review(sources: dict[str, str], *, reviewer: str = "/root/protected_camera_v2_reviewer") -> dict:
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
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def _review_binding(review: dict) -> dict:
    raw = contract.canonical_json_bytes(review) + b"\n"
    return contract.artifact_binding(contract.REVIEW_RELATIVE_PATH, raw, content_sha256=review["content_sha256"])


def _authorization(review: dict, *, authorizer: str = "/root/protected_camera_v2_authorizer") -> dict:
    return contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_protected_camera_adaptation_v2_attempt",
        "authorizer": authorizer,
        "independent_review": _review_binding(review),
        "predecessor": contract.predecessor_contract(),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "reporting_contract": contract.reporting_contract(),
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


def test_exact_v1_derivation_and_source_closure() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert len(contract._v1_contract.SOURCE_PATHS) == 33
    assert len(bindings) == 37
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert set(contract.V1_SOURCE_SHA256.items()) <= set(bindings.items())
    assert bindings[contract.V1_TERMINAL_AUDIT_RELATIVE_PATH] == contract.V1_TERMINAL_AUDIT_BINDING["file_sha256"]


def test_v1_terminal_audit_is_exact_numeric_no_pass_precondition() -> None:
    raw = (ROOT / contract.V1_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    value = contract.validate_v1_terminal_audit(raw)
    assert value["verdict"] == contract.V1_TERMINAL_AUDIT_BINDING["verdict"]
    assert value["successor_decision"]["qualified_camera_checkpoint_exists"] is False
    assert value["successor_decision"]["automatic_successor_authorized"] is False
    assert value["successor_decision"]["frozen_camera_jepa_training_may_start"] is False
    assert value["successor_decision"]["training_extension_or_retry_authorized"] is False
    with pytest.raises(PermissionError, match="byte binding"):
        contract.validate_v1_terminal_audit(raw[:-1] + bytes([raw[-1] ^ 1]))


def test_encoder_lr_scale_is_the_only_science_leaf_delta() -> None:
    differences = _leaf_differences(contract._v1_contract.science_contract(), contract.science_contract())
    assert differences == [(('optimizer', 'encoder_learning_rate_scale'), 0.01, 0.10)]
    assert contract.science_delta()["other_science_changes"] == []
    for update in (1, 100, 400, 1_000, 2_000, 4_000):
        v1_head, v1_encoder = contract._v1_contract.learning_rates(update)
        head, encoder = contract.learning_rates(update)
        assert head == v1_head
        assert encoder == pytest.approx(10.0 * v1_encoder)
        assert encoder == pytest.approx(0.10 * head)
    assert contract.OPTIMIZER_CONTRACT == {
        **contract._v1_contract.OPTIMIZER_CONTRACT,
        "encoder_learning_rate_scale": 0.10,
    }
    assert contract.science_contract()["initial_state_sha256"] == "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"


def test_reporting_is_fixed_sidecars_without_a_numeric_progress_cutoff() -> None:
    reporting = contract.reporting_contract()
    assert reporting["fixed_checkpoint_updates"] == [100, 400, 1_000, 2_000, 4_000]
    assert reporting["sidecar_paths"] == [
        "checkpoints/update_100.metrics.json",
        "checkpoints/update_400.metrics.json",
        "checkpoints/update_1000.metrics.json",
        "checkpoints/update_2000.metrics.json",
        "checkpoints/update_4000.metrics.json",
    ]
    assert reporting["numeric_progress_cutoff_at_update_400"] is False
    assert reporting["metric_controlled_stop_other_than_earliest_all_nine_pass"] is False
    assert reporting["read_only_observers_must_not_rerun_evaluation"] is True
    with pytest.raises(ValueError):
        contract.metric_sidecar_path(401)


def test_review_and_authorization_are_strict_identity_separated_and_deny_downstream() -> None:
    sources = {"synthetic.py": "0" * 64}
    review = _review(sources)
    assert contract.validate_review(review, expected_sources=sources) == review
    authorization = _authorization(review)
    binding = _review_binding(review)
    assert contract.validate_authorization(authorization, review_binding=binding, reviewer=review["reviewer"]) == authorization
    for key in ("jepa_training_authorized", "g2_authorized", "navigation_authorized", "heldout_authorized", "retry_authorized", "training_extension_authorized", "soft_promotion_authorized"):
        assert authorization["authority"][key] is False
    assert authorization["authority"]["mutation_scope"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    assert authorization["authority"]["checkpoint_metric_sidecar_publication_authorized"] is True
    with pytest.raises(PermissionError):
        contract.validate_review({**review, "reporting_contract": {}}, expected_sources=sources)
    for identity in (contract.IMPLEMENTATION_AUTHOR, review["reviewer"]):
        with pytest.raises(PermissionError):
            contract.validate_authorization({**authorization, "authorizer": identity}, review_binding=binding, reviewer=review["reviewer"])


def test_runner_import_defers_torch_and_exact_v1_runner_is_bound() -> None:
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"import importlib.util,sys; p={str(path)!r}; s=importlib.util.spec_from_file_location('r',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert hashlib.sha256((ROOT / contract.V1_RUNNER_RELATIVE_PATH).read_bytes()).hexdigest() == contract.V1_SOURCE_SHA256[contract.V1_RUNNER_RELATIVE_PATH]


def test_v2_contract_covers_every_attribute_used_by_exact_rebound_v1_runner() -> None:
    tree = ast.parse((ROOT / contract.V1_RUNNER_RELATIVE_PATH).read_text(encoding="utf-8"))
    required = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "contract"
    }
    missing = sorted(name for name in required if not hasattr(contract, name))
    assert len(required) == 44
    assert missing == []


def test_sidecar_is_canonical_atomic_exclusive_read_only_and_strict(tmp_path: Path) -> None:
    runner = _runner()
    metric = _metric(100)
    checkpoint = _checkpoint(100)
    binding = runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=checkpoint, metric=metric)
    path = tmp_path / binding["path"]
    raw = path.read_bytes()
    value = contract.parse_canonical_json(raw, name="test sidecar")
    assert raw == contract.canonical_json_bytes(value) + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert not list((tmp_path / "checkpoints").glob(".*.publishing"))
    assert binding["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert contract.validate_metric_sidecar(value, update=100, checkpoint=checkpoint, metric=metric) == value
    with pytest.raises(FileExistsError):
        runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=checkpoint, metric=metric)
    tampered = {**value, "continuation": "stop_at_update_400"}
    with pytest.raises(PermissionError):
        contract.validate_metric_sidecar(tampered)


def test_publication_occurs_immediately_after_inline_evaluation_before_only_selection_branch() -> None:
    runner = _runner()
    wrapper_source = inspect.getsource(runner._train)
    observer_source = wrapper_source[wrapper_source.index("def evaluate_then_publish") :]
    assert observer_source.index("metric = original_evaluate") < observer_source.index("_publish_metric_sidecar") < observer_source.index("return metric")
    base_source = inspect.getsource(runner._BASE_TRAIN)
    assert base_source.index("metric = _evaluate") < base_source.index("metrics.append(metric)") < base_source.index('metric["evaluation"]["all_nine_physical_pass"]')
    assert base_source.count('metric["evaluation"]') == 1
    assert "physical_pass_count" not in base_source
    assert "update == 400" not in base_source
    assert "observable_camera_ray_v4_loss_v4" in base_source
    assert "update_ema_target_after_optimizer_step" not in base_source
    assert "combine_joint_losses" not in base_source


def test_train_hook_publishes_sidecar_and_preserves_base_success_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    metric = _metric(100, passes=False)
    checkpoint = _checkpoint(100)
    monkeypatch.setattr(runner._base, "_snapshot", lambda *args, **kwargs: dict(checkpoint))
    monkeypatch.setattr(runner._base, "_evaluate", lambda *args, **kwargs: dict(metric))

    def fake_base_train(*args, **kwargs):
        artifact = runner._base._snapshot(update=100)
        observed = runner._base._evaluate(update=100)
        return [{"schema": "trace", "update": 1}], [observed], [artifact], None, {}, {"operation_counts": {}}

    monkeypatch.setattr(runner, "_BASE_TRAIN", fake_base_train)
    trace, metrics, artifacts, selected, _, state = runner._train(*([None] * 12), tmp_path)
    assert trace[0]["update"] == 1
    assert metrics == [metric]
    assert selected is None
    sidecar = next(item for item in artifacts if item["path"].endswith(".metrics.json"))
    assert sidecar["path"] == "checkpoints/update_100.metrics.json"
    assert state["operation_counts"]["checkpoint_metric_sidecar_publication_count"] == 1
    runner._write_exclusive(tmp_path / "reservation.json", b"reservation")
    runner._write_exclusive(tmp_path / checkpoint["path"], b"checkpoint")
    for name in ("training_trace.jsonl", "checkpoint_metrics.json", "access.json", "result.json"):
        runner._write_exclusive(tmp_path / name, name.encode())
    files, directories = runner._base._terminal_paths(tmp_path)
    assert directories == [".", "checkpoints"]
    assert files == sorted([
        "reservation.json", checkpoint["path"], sidecar["path"], "training_trace.jsonl",
        "checkpoint_metrics.json", "access.json", "result.json",
    ])


def test_final_metrics_collate_existing_sidecars_without_evaluation(tmp_path: Path) -> None:
    runner = _runner()
    metric = _metric(100, passes=False)
    sidecar = runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=_checkpoint(100), metric=metric)
    runner._ACTIVE_SIDECARS[:] = [sidecar]
    trace_binding, metrics_binding = runner._publish_training(tmp_path, [{"schema": "synthetic_trace", "update": 1}], [metric])
    assert trace_binding["row_count"] == 1
    final = contract.parse_canonical_json((tmp_path / "checkpoint_metrics.json").read_bytes(), name="final metrics")
    assert final["rows"] == [metric]
    assert final["sidecars"] == [sidecar]
    assert final["inline_evaluation_count"] == 1
    assert final["observer_evaluation_rerun_count"] == 0
    assert final["numeric_progress_cutoff_at_update_400"] is False
    assert metrics_binding["content_sha256"] == final["content_sha256"]


def test_success_access_receipt_rehashes_every_published_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    output = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    sidecar = runner._publish_metric_sidecar(output, update=100, checkpoint=_checkpoint(100), metric=_metric(100, passes=False))
    runner._ACTIVE_SIDECARS[:] = [sidecar]
    runner._ACTIVE_V1_RECORDS[:] = [{"path": path} for path in contract.V1_TERMINAL_EXACT_PATHS]
    monkeypatch.setattr(runner, "_ACTIVE_V1_AUDIT", {"verdict": contract.V1_TERMINAL_AUDIT_BINDING["verdict"]})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_BASE_ACCESS_RECEIPT", lambda *args, **kwargs: {"schema": contract.ACCESS_SCHEMA})
    receipt = runner._access_receipt()
    assert receipt["checkpoint_metric_sidecars"] == {
        "records": [sidecar],
        "count": 1,
        "all_rehashed": True,
        "observer_evaluation_rerun_count": 0,
    }


def test_failure_inventory_binds_sidecars_directories_and_failure_path(tmp_path: Path) -> None:
    runner = _runner()
    reservation, _ = runner._publish_json(tmp_path / "reservation.json", {"schema": contract.RESERVATION_SCHEMA, "attempt_identity": "synthetic"})
    runner._write_exclusive(tmp_path / "checkpoints/update_100.pt", b"synthetic")
    sidecar = runner._publish_metric_sidecar(tmp_path, update=100, checkpoint=_checkpoint(100), metric=_metric(100, passes=False))
    runner._terminal_failure(tmp_path, reservation, "scientific_numeric_physical_gate", RuntimeError("no pass"), {sidecar["path"]: sidecar}, numeric=True)
    failed = contract.parse_canonical_json((tmp_path / "failed.json").read_bytes(), name="test failure")
    assert failed["failure_path"] == "failed.json"
    assert failed["exact_terminal_directories_including_root"] == [".", "checkpoints"]
    assert failed["exact_terminal_files"] == sorted(["reservation.json", "checkpoints/update_100.pt", sidecar["path"], "failed.json"])
    assert sidecar["path"] in failed["artifacts"]
    assert failed["artifacts"][sidecar["path"]]["file_sha256"] == sidecar["file_sha256"]
    assert failed["status"] == "failed_numeric_physical_gate"
    assert failed["extension_or_retry_authorized"] is False


def test_runner_hooks_are_narrow_and_restore_exact_v1_lifecycle() -> None:
    runner = _runner()
    source = inspect.getsource(runner.run_parent)
    for hook in ("_train", "_publish_training", "_update0_terminal", "_access_receipt", "_terminal_failure"):
        assert f"_base.{hook} =" in source
    assert "finally:" in source
    assert "_BASE_RUN_PARENT" in source
    assert runner.contract.TERMINAL_DIRECTORIES_INCLUDING_ROOT == (".", "checkpoints")
    assert runner.contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("protected_camera_adaptation_v2")
