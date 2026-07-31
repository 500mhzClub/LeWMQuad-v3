from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/run_go2_world_model_substrate_authorized_v1.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("substrate_supervisor_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    supervisor = _load_supervisor()
    assert supervisor.strict_json_bytes(b'{"a": 1}', label="fixture") == {"a": 1}
    with pytest.raises(supervisor.SupervisionError, match="duplicate JSON key"):
        supervisor.strict_json_bytes(b'{"a": 1, "a": 2}', label="fixture")
    with pytest.raises(supervisor.SupervisionError, match="non-finite"):
        supervisor.strict_json_bytes(b'{"a": NaN}', label="fixture")


def test_gpu_selector_correction_binding_is_exact() -> None:
    supervisor = _load_supervisor()
    expected = supervisor.EXPECTED_SELECTOR_CORRECTION
    actual = supervisor.file_binding(REPO_ROOT / expected["path"])
    assert actual["byte_count"] == expected["byte_count"]
    assert actual["sha256"] == expected["sha256"]


def test_follow_on_review_requires_exact_non_authorizing_pass() -> None:
    supervisor = _load_supervisor()
    source_commit = "a" * 40
    review = {
        "schema": supervisor.FOLLOW_ON_REVIEW_SCHEMA,
        "status": supervisor.FOLLOW_ON_REVIEW_STATUS,
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": source_commit,
        "remaining_findings": [],
    }
    assert supervisor.validate_follow_on_source_review(
        review, source_commit=source_commit
    ) is review
    for field, replacement in (
        ("status", "PASS"),
        ("authority_granted_by_this_document", True),
        ("reviewed_source_commit", "b" * 40),
        ("remaining_findings", ["unresolved"]),
    ):
        changed = dict(review)
        changed[field] = replacement
        with pytest.raises(supervisor.SupervisionError, match="exact PASS"):
            supervisor.validate_follow_on_source_review(
                changed, source_commit=source_commit
            )


def test_device_preflight_rejects_old_selector_before_torch_access(monkeypatch) -> None:
    supervisor = _load_supervisor()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    with pytest.raises(supervisor.SupervisionError, match="must itself start"):
        supervisor.validate_corrected_device_preflight()


def test_device_preflight_accepts_exact_r9700_contract(monkeypatch) -> None:
    supervisor = _load_supervisor()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    properties = types.SimpleNamespace(
        name="AMD Radeon AI PRO R9700",
        gcnArchName="gfx1201",
        total_memory=34_208_743_424,
    )
    fake_torch = types.SimpleNamespace(
        __version__="test-rocm",
        version=types.SimpleNamespace(hip="test-hip"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: properties,
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    receipt = supervisor.validate_corrected_device_preflight()
    assert receipt["selector"] == "0"
    assert receipt["device_name"] == "AMD Radeon AI PRO R9700"
    assert receipt["gcn_arch"] == "gfx1201"


def test_retention_parity_compares_only_shared_finite_scalars() -> None:
    supervisor = _load_supervisor()
    predecessor = {
        "spatial_retention": {
            "evaluation": {"a": 1.0, "nested": {"b": 2}, "left_only": 4.0}
        }
    }
    update_zero = {
        "spatial_retention": {
            "evaluation": {"a": 1.0 + 5e-8, "nested": {"b": 2}, "right_only": 7.0}
        }
    }
    differences = supervisor.retention_parity_differences(predecessor, update_zero)
    assert set(differences) == {"a", "nested.b"}
    assert differences["a"] == pytest.approx(5e-8)
    assert differences["nested.b"] == 0.0


def test_retention_parity_fails_without_shared_metrics() -> None:
    supervisor = _load_supervisor()
    with pytest.raises(supervisor.SupervisionError, match="no shared"):
        supervisor.retention_parity_differences(
            {"spatial_retention": {"evaluation": {"a": 1.0}}},
            {"spatial_retention": {"evaluation": {"b": 1.0}}},
        )


def test_execute_requested_block_uses_every_policy_step_and_updates_history() -> None:
    from lewm_genesis.rollout import RolloutRunner, _BlockTrajectory

    runner = RolloutRunner.__new__(RolloutRunner)
    runner.n_envs = 2
    runner._block_size = 5
    runner._policy_steps_per_command_tick = 5
    runner._last_executed = np.zeros((2, 3), dtype=np.float32)
    executed = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    requested = executed + 100.0
    runner._clip_block = lambda value: _BlockTrajectory(  # type: ignore[method-assign]
        requested=np.asarray(value),
        executed=executed,
        clipped=np.asarray([True, False]),
    )
    stepped: list[np.ndarray] = []
    callbacks: list[tuple[int, int]] = []
    runner._step_policy_step = lambda target: stepped.append(target.copy())  # type: ignore[method-assign]

    result = runner.execute_requested_block(
        requested,
        after_policy_step=lambda tick, policy: callbacks.append((tick, policy)),
    )

    assert result.executed is executed
    assert len(stepped) == 25
    assert callbacks == [(tick, policy) for tick in range(5) for policy in range(5)]
    for tick in range(5):
        np.testing.assert_array_equal(stepped[tick * 5], executed[:, tick])
    np.testing.assert_array_equal(runner._last_executed, executed[:, -1])


def test_execute_requested_block_rejects_wrong_shape_before_clipping() -> None:
    from lewm_genesis.rollout import RolloutRunner

    runner = RolloutRunner.__new__(RolloutRunner)
    runner.n_envs = 2
    runner._block_size = 5
    with pytest.raises(ValueError, match="expected"):
        runner.execute_requested_block(np.zeros((1, 5, 3), dtype=np.float32))
