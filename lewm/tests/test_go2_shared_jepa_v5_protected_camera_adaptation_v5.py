from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lewm.benchmarks import go2_shared_jepa_v5_protected_camera_adaptation_v5 as contract


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("_test_protected_camera_adaptation_v5_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schedule_indices() -> list[int]:
    raw = (ROOT / contract.SCHEDULE_RELATIVE_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == contract.SCHEDULE_FILE_SHA256
    schedule = contract.parse_canonical_json(raw, name="bound native schedule")
    assert schedule["content_sha256"] == contract.SCHEDULE_CONTENT_SHA256
    return schedule["presentation_indices"]


def _progress(
    update: int,
    *,
    passed: int,
    shortfall: float,
    worst: float,
    loss: float = 1.0,
    all_nine: bool = False,
    update_4000_control_baseline: dict | None = None,
) -> dict:
    return contract.control_decision_from_progress(
        update=update,
        passed_margin_count=passed,
        total_shortfall=shortfall,
        worst_margin=worst,
        aggregate_complete_v4_loss=loss,
        all_nine_physical_pass=all_nine,
        update_4000_control_baseline=update_4000_control_baseline,
    )


def _review(sources: dict[str, str]) -> dict:
    return contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS",
        "implementation_author": contract.IMPLEMENTATION_AUTHOR,
        "reviewer": "/root/camera_v5_roundtrip_reviewer",
        "reviewed_sources": sources,
        "predecessor": contract.predecessor_contract(),
        "science_contract": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "evidence": contract.evidence_contract(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
        "source_only": True,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def test_exact_v3_science_extension_native_learning_rate_and_source_closure() -> None:
    expected = copy.deepcopy(contract._v3_contract.science_contract())
    expected["schedule"].update({
        "use_exact_prefix_updates": 8_000,
        "presentation_count": 128_000,
        "presentation_indices_sha256": contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[8_000],
        "checkpoint_updates": list(contract.CHECKPOINT_UPDATES),
        "checkpoint_prefix_sha256": {
            str(update): digest
            for update, digest in contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256.items()
        },
    })
    expected["optimizer"]["maximum_updates"] = 8_000
    assert contract.science_contract() == expected
    assert contract.CHECKPOINT_UPDATES == (100, 400, 1_000, 4_000, 6_000, 8_000)
    assert contract.MAXIMUM_UPDATE == 8_000
    assert contract.science_delta()["other_training_science_changes"] == []
    for update in (1, 100, 400, 1_000, 4_000, 6_000, 8_000):
        expected_lr = contract._v3_contract._v1.learning_rate(update)
        assert contract.learning_rates(update) == (expected_lr, expected_lr)
    assert contract.learning_rates(1)[0] == pytest.approx(1e-6)
    assert contract.learning_rates(400)[0] == pytest.approx(1e-4)
    assert contract.learning_rates(8_000)[0] == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        contract.learning_rates(8_001)
    bindings = contract.current_source_bindings(ROOT)
    assert set(bindings) == set(contract.SOURCE_PATHS)
    assert set(contract.V3_SOURCE_SHA256.items()) <= set(bindings.items())
    assert set(contract.FIXED_EVIDENCE_SHA256.items()) <= set(bindings.items())


def test_schedule_requires_exact_128000_presentations_and_every_fixed_prefix() -> None:
    runner = _runner()
    indices = _schedule_indices()
    assert len(indices) == 128_000
    runner._validate_schedule(indices)
    changed = list(indices)
    changed[0] = (changed[0] + 1) % 4_262
    with pytest.raises(PermissionError, match="prefix"):
        runner._validate_schedule(changed)
    with pytest.raises(PermissionError, match="length|128000|exact"):
        runner._validate_schedule(indices[:-1])
    with pytest.raises(PermissionError, match="length|128000|exact"):
        runner._validate_schedule([*indices, indices[0]])


def test_controls_are_loss_free_exact_and_bind_update6000_to_same_run_update4000() -> None:
    p1, s1, w1 = 106, 49.09939462151839, -7.944758415222166
    p4, s4, w4 = 134, 19.869159033399846, -4.920835733413693
    for loss in (0.0, 1e30):
        assert _progress(1_000, passed=p1, shortfall=s1, worst=w1, loss=loss)["action"] == contract.CONTROL_ACTION_CONTINUE
        assert _progress(4_000, passed=p4, shortfall=s4, worst=w4, loss=loss)["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(1_000, passed=p1 - 1, shortfall=s1, worst=w1)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _progress(1_000, passed=p1, shortfall=math.nextafter(s1, math.inf), worst=w1)["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    assert _progress(4_000, passed=p4, shortfall=s4, worst=math.nextafter(w4, -math.inf))["action"] == contract.CONTROL_ACTION_STOP_PROGRESS

    baseline = {
        "update": 4_000,
        "path": contract.metric_sidecar_path(4_000),
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "passed_margin_count": p4,
        "total_shortfall": s4,
        "worst_margin": w4,
    }
    equal = _progress(
        6_000,
        passed=p4,
        shortfall=s4,
        worst=w4,
        update_4000_control_baseline=baseline,
    )
    assert equal["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    improved = _progress(
        6_000,
        passed=p4 + 1,
        shortfall=s4,
        worst=w4,
        loss=1e30,
        update_4000_control_baseline=baseline,
    )
    assert improved["action"] == contract.CONTROL_ACTION_CONTINUE
    assert _progress(
        6_000,
        passed=p4 + 1,
        shortfall=math.nextafter(s4, math.inf),
        worst=w4,
        update_4000_control_baseline=baseline,
    )["action"] == contract.CONTROL_ACTION_STOP_PROGRESS
    with pytest.raises(PermissionError, match="baseline|4000"):
        _progress(6_000, passed=p4 + 1, shortfall=s4, worst=w4)
    assert _progress(8_000, passed=189, shortfall=0.0, worst=0.0)["action"] == contract.CONTROL_ACTION_STOP_MAXIMUM
    assert _progress(8_000, passed=189, shortfall=0.0, worst=0.0, all_nine=True)["action"] == contract.CONTROL_ACTION_QUALIFY


def test_update100_integrity_precedes_apparent_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = {
        "update": 100,
        "passed_margin_count": 189,
        "total_shortfall": 0.0,
        "worst_margin": 0.1,
        "aggregate_complete_v4_loss": 1.0,
        "all_nine_physical_pass": True,
    }
    monkeypatch.setattr(contract, "checkpoint_progress", lambda metric: progress)
    unchanged = {
        "state_sha256_before": contract.UPDATE0_STATE_SHA256,
        "state_sha256_after": contract.UPDATE0_STATE_SHA256,
        "frozen_state_sha256_before_and_after": "1" * 64,
        "state_mutation_count": 0,
    }
    with pytest.raises(PermissionError, match="movement|integrity"):
        contract.checkpoint_control_decision(unchanged)
    moved = {**unchanged, "state_sha256_before": "2" * 64, "state_sha256_after": "2" * 64}
    assert contract.checkpoint_control_decision(moved)["action"] == contract.CONTROL_ACTION_QUALIFY


def test_canonical_review_and_authorization_round_trip_and_tamper() -> None:
    sources = contract.current_source_bindings(ROOT)
    review = _review(sources)
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    parsed = contract.parse_canonical_json(review_raw, name="round-trip V5 review")
    assert contract.validate_review(parsed, expected_sources=sources) == parsed
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "authorized_one_exact_protected_camera_adaptation_v5_native_schedule_completion_attempt",
        "authorizer": "/root/camera_v5_roundtrip_authorizer",
        "independent_review": review_binding,
        "predecessor": contract.predecessor_contract(),
        "raw": contract.expected_raw_authority(),
        "camera": contract.expected_camera_authority(),
        "experiment": contract.science_contract(),
        "science_delta": contract.science_delta(),
        "evidence": contract.evidence_contract(),
        "reporting_contract": contract.reporting_contract(),
        "control_contract": contract.control_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    })
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    parsed_authorization = contract.parse_canonical_json(
        authorization_raw, name="round-trip V5 authorization"
    )
    assert contract.validate_authorization(
        parsed_authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
    ) == parsed_authorization
    with pytest.raises(PermissionError):
        contract.validate_authorization(
            {**authorization, "content_sha256": "0" * 64},
            review_binding=review_binding,
            reviewer=review["reviewer"],
        )


def test_runner_import_is_nonmutating_accelerator_free_and_does_not_reserve() -> None:
    output = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation = output / "reservation.json"
    before = {
        "root": output.exists(),
        "reservation": reservation.read_bytes() if reservation.is_file() else None,
    }
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    code = f"""
import importlib.util,json,sys
p={str(path)!r}; s=importlib.util.spec_from_file_location('_isolated_camera_v5',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(json.dumps(sorted(set(sys.modules)&{{'torch','numpy','PIL','cv2'}})))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == "" and json.loads(completed.stdout) == []
    assert output.exists() is before["root"]
    assert (reservation.read_bytes() if reservation.is_file() else None) == before["reservation"]


def test_runner_restores_hooks_and_loss_and_finite_snapshot_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner()
    outer_originals = (
        runner._v3.contract,
        runner._v3._train,
        runner._v3._publish_metric_sidecar,
        runner._v3._publish_training,
        runner._v3._access_receipt,
        runner._v3._terminal_failure,
        runner._v1_runner._validate_schedule,
    )

    def fail_parent(**kwargs):
        assert kwargs == {
            "review_file_sha256": "a" * 64,
            "authorization_file_sha256": "b" * 64,
        }
        assert runner._v3.contract is runner.contract
        assert runner._v3._train is runner._train
        assert runner._v3._publish_metric_sidecar is runner._publish_metric_sidecar
        assert runner._v3._publish_training is runner._publish_training
        assert runner._v3._access_receipt is runner._access_receipt
        assert runner._v3._terminal_failure is runner._terminal_failure
        assert runner._v1_runner._validate_schedule is runner._validate_schedule
        raise RuntimeError("synthetic V5 parent failure")

    monkeypatch.setattr(runner, "_BASE_V3_RUN_PARENT", fail_parent)
    with pytest.raises(RuntimeError, match="synthetic V5 parent failure"):
        runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
        )
    assert (
        runner._v3.contract,
        runner._v3._train,
        runner._v3._publish_metric_sidecar,
        runner._v3._publish_training,
        runner._v3._access_receipt,
        runner._v3._terminal_failure,
        runner._v1_runner._validate_schedule,
    ) == outer_originals

    original_loss = object()
    loss_adapter = SimpleNamespace(observable_camera_ray_v4_loss_v4=original_loss)
    original_snapshot = runner._v1_runner._snapshot
    finite_false = SimpleNamespace(
        all=lambda: SimpleNamespace(item=lambda: False)
    )
    fake_torch = SimpleNamespace(isfinite=lambda value: finite_false)
    runtime = SimpleNamespace(loss_adapter=loss_adapter, torch=fake_torch)

    def fail(*args, **kwargs):
        assert loss_adapter.observable_camera_ray_v4_loss_v4 is original_loss
        assert runner._v1_runner._snapshot is not original_snapshot
        raise RuntimeError("synthetic V5 train failure")

    monkeypatch.setattr(runner, "_BASE_V3_TRAIN", fail)
    with pytest.raises(RuntimeError, match="synthetic V5 train failure"):
        runner._train(runtime, None, None, [], [], [], [], [], [], [], None, None, tmp_path)
    assert loss_adapter.observable_camera_ray_v4_loss_v4 is original_loss
    assert runner._v1_runner._snapshot is original_snapshot

    called = False

    def base(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    snapshot_runtime = SimpleNamespace(torch=fake_torch)
    checked = runner._finite_snapshot(base, snapshot_runtime)
    nonfinite = SimpleNamespace(
        is_floating_point=lambda: True,
        is_complex=lambda: False,
    )
    model = SimpleNamespace(state_dict=lambda: {"encoder.bad": nonfinite})
    with pytest.raises(FloatingPointError, match="encoder.bad"):
        checked(snapshot_runtime, model, tmp_path, update=100, frozen_sha="1" * 64)
    assert called is False
