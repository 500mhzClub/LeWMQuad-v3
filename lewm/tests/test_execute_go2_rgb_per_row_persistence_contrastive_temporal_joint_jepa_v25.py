from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)
TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25.py"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _executor(name: str) -> Any:
    return _load(EXECUTOR_PATH, name)


def _diagnostics(executor: Any) -> dict[str, Any]:
    prediction = tuple(0.2 + 0.01 * index for index in range(16))
    persistence = tuple(0.25 + 0.005 * index for index in range(16))
    gaps = tuple(a - b for a, b in zip(prediction, persistence, strict=True))
    row_loss = tuple(executor._stable_softplus(value) / math.log(2.0) for value in gaps)
    legacy = tuple(
        (sum(prediction[start : start + 4]) / 4.0)
        / max(sum(persistence[start : start + 4]) / 4.0, 1e-6)
        for start in range(0, 16, 4)
    )
    result: dict[str, Any] = {
        "mechanism": "per_row_persistence_contrastive_temporal_v1",
        "prediction_energy_per_row": prediction,
        "persistence_energy_per_row": persistence,
        "gap_per_row": gaps,
        "row_loss_per_row": row_loss,
        "negative_gap_count": sum(value < 0.0 for value in gaps),
        "negative_gap_fraction": sum(value < 0.0 for value in gaps) / 16.0,
        "legacy_global_ratio_per_microbatch": legacy,
        "log2_normalizer": math.log(2.0),
        "softplus_beta": 1.0,
        "softplus_threshold": 20.0,
        "denominator_used": False,
    }
    for stem, values in (
        ("prediction_energy", prediction),
        ("persistence_energy", persistence),
        ("gap", gaps),
        ("row_loss", row_loss),
        ("legacy_global_ratio", legacy),
    ):
        result[f"{stem}_count"] = len(values)
        result[f"{stem}_sum"] = sum(values)
        result[f"{stem}_mean"] = sum(values) / len(values)
        result[f"{stem}_minimum"] = min(values)
        result[f"{stem}_maximum"] = max(values)
    return result


class _FakeCuda:
    def __init__(self) -> None:
        self.state = torch.arange(8, dtype=torch.uint8)

    def get_rng_state_all(self) -> list[torch.Tensor]:
        return [self.state.clone()]


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online = torch.nn.Linear(2, 2)
        self.target = torch.nn.Linear(2, 2)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("ema_update_count", torch.tensor(400, dtype=torch.long))


class _MemoryPublisher:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.order: list[str] = []

    @staticmethod
    def _binding(path: str, raw: bytes) -> dict[str, Any]:
        return {
            "path": path,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }

    def publish_json(self, relative: str, core: dict[str, Any]) -> dict[str, Any]:
        if relative in self.files:
            raise FileExistsError(relative)
        value = executor_for_publisher._engine._content_bound(core)
        raw = executor_for_publisher._engine._canonical_json_bytes(value) + b"\n"
        self.files[relative] = raw
        self.order.append(relative)
        return {"value": value, "binding": self._binding(relative, raw)}

    def publish_bytes(self, relative: str, raw: bytes) -> dict[str, Any]:
        if relative in self.files:
            raise FileExistsError(relative)
        self.files[relative] = raw
        self.order.append(relative)
        return self._binding(relative, raw)


executor_for_publisher: Any


def _recovery_fixture(executor: Any):
    global executor_for_publisher
    executor_for_publisher = executor
    model = _ToyModel()
    optimizer = torch.optim.AdamW(model.online.parameters(), lr=1e-3)
    loss = model.online(torch.ones((1, 2))).square().sum()
    loss.backward()
    optimizer.step()
    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        random=torch.random,
        cuda=_FakeCuda(),
        save=torch.save,
    )
    runtime = SimpleNamespace(torch=fake_torch, np=np)
    accounting = executor._expected_accounting_v25(400)
    authority = {
        "frozen_source_and_review_commit": "1" * 40,
        "recursive_source_closure_manifest_sha256": "2" * 64,
        "independent_source_review_sha256": "3" * 64,
        "clean_export_certification_sha256": "4" * 64,
        "execution_binding_commit": "5" * 40,
        "runtime_inputs": {"schedule": {"file_sha256": "6" * 64}},
    }
    gate = {"passed": True, "action": "CONTINUE_TO_UPDATE_1000"}
    metrics = tuple(
        {"path": f"metrics/update_{update}.json", "file_sha256": str(index) * 64}
        for index, update in zip((6, 7, 8), (0, 100, 400), strict=True)
    )
    trace_prefix = {"row_count": 402, "byte_count": 123, "file_prefix_sha256": "8" * 64}
    return runtime, model, optimizer, accounting, authority, gate, metrics, trace_prefix


def test_denied_shell_and_actual_training_api_bind_v25(capsys) -> None:
    executor = _executor("_v25_executor_denial")
    assert executor.main([]) == 4
    assert "DENIED_SOURCE_ONLY" in capsys.readouterr().out
    assert executor.PREREGISTRATION_COMMIT == (
        "f00e20df3b429f9242516ac38f67fea587e04b22"
    )
    training = _load(TRAINING_PATH, "_v25_training_for_executor")
    receipt = executor.validate_training_api_v25(training)
    assert receipt["temporal_objective"] == (
        "P25_per_row_softplus_energy_gap_over_log2"
    )
    assert receipt["new_batch_fields_over_v24"] == 0
    assert receipt["j24_bit_identical_to_v24"] is True


def test_diagnostics_bind_all_rows_equations_and_detached_legacy_summary() -> None:
    executor = _executor("_v25_executor_diagnostics")
    diagnostics = _diagnostics(executor)
    validated = executor._validate_p25_diagnostics(diagnostics)
    assert len(validated["row_loss_per_row"]) == 16
    assert len(validated["legacy_global_ratio_per_microbatch"]) == 4
    assert validated["denominator_used"] is False
    corrupted = dict(diagnostics)
    corrupted["gap_per_row"] = (*diagnostics["gap_per_row"][:-1], 10.0)
    with pytest.raises(RuntimeError, match="gap equation"):
        executor._validate_p25_diagnostics(corrupted)


def test_recovery_writer_is_pass_gated_complete_write_once_and_nonmutating() -> None:
    executor = _executor("_v25_executor_recovery")
    fixture = _recovery_fixture(executor)
    runtime, model, optimizer, accounting, authority, gate, metrics, trace_prefix = fixture
    publisher = _MemoryPublisher()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_before = executor._tree_sha256_v25(
        runtime.torch, runtime.np, optimizer.state_dict()
    )
    snapshot, binding = executor._publish_update400_recovery_v25(
        authority=authority,
        runtime=runtime,
        publisher=publisher,
        model=model,
        optimizer=optimizer,
        accounting=accounting,
        gate_decision=gate,
        metric_bindings=metrics,
        trace_prefix_identity=trace_prefix,
        publication_state={},
    )
    assert publisher.order == [
        executor.RECOVERY_STATE_RELATIVE_PATH,
        executor.RECOVERY_BINDING_RELATIVE_PATH,
    ]
    assert snapshot["path"] == executor.RECOVERY_STATE_RELATIVE_PATH
    assert binding["path"] == executor.RECOVERY_BINDING_RELATIVE_PATH
    payload = torch.load(
        io.BytesIO(publisher.files[executor.RECOVERY_STATE_RELATIVE_PATH]),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["next_update"] == 401
    assert payload["next_schedule_position"] == 6_400
    assert payload["ema_update_count"] == 400
    assert payload["accounting"] == accounting
    assert payload["optimizer_state_dict"]["state"]
    assert payload["parameter_gradients"]
    assert payload["rng_states"]["visible_rocm_devices"]
    assert payload["scientific_identity"]["trace_prefix"] == trace_prefix
    assert payload["dataset_or_rgb_payload_included"] is False
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in model_before.items())
    assert executor._tree_sha256_v25(
        runtime.torch, runtime.np, optimizer.state_dict()
    ) == optimizer_before
    with pytest.raises(FileExistsError):
        executor._publish_update400_recovery_v25(
            authority=authority,
            runtime=runtime,
            publisher=publisher,
            model=model,
            optimizer=optimizer,
            accounting=accounting,
            gate_decision=gate,
            metric_bindings=metrics,
            trace_prefix_identity=trace_prefix,
            publication_state={},
        )


def test_failed_gate_or_preexisting_target_fails_closed_without_binding() -> None:
    executor = _executor("_v25_executor_fail_closed")
    runtime, model, optimizer, accounting, authority, gate, metrics, trace_prefix = (
        _recovery_fixture(executor)
    )
    failed = dict(gate, passed=False, action="FAIL_TERMINAL_NO_RETRY_NO_RESUME")
    publisher = _MemoryPublisher()
    with pytest.raises(PermissionError, match="passed update-400 gate"):
        executor._publish_update400_recovery_v25(
            authority=authority,
            runtime=runtime,
            publisher=publisher,
            model=model,
            optimizer=optimizer,
            accounting=accounting,
            gate_decision=failed,
            metric_bindings=metrics,
            trace_prefix_identity=trace_prefix,
            publication_state={},
        )
    assert publisher.files == {}
    publisher.files[executor.RECOVERY_STATE_RELATIVE_PATH] = b"preexisting"
    with pytest.raises(FileExistsError):
        executor._publish_update400_recovery_v25(
            authority=authority,
            runtime=runtime,
            publisher=publisher,
            model=model,
            optimizer=optimizer,
            accounting=accounting,
            gate_decision=gate,
            metric_bindings=metrics,
            trace_prefix_identity=trace_prefix,
            publication_state={},
        )
    assert executor.RECOVERY_BINDING_RELATIVE_PATH not in publisher.files


def test_source_orders_recovery_after_gate_before_continuation_and_has_no_reader() -> None:
    source = EXECUTOR_PATH.read_text(encoding="utf-8")
    branch = source[source.index("if update == RECOVERY_UPDATE:") : source.index(
        "elif update == 1_000:"
    )]
    positions = tuple(
        branch.index(marker)
        for marker in (
            "evaluate_update400_gate_v13",
            '"event": "update400_control"',
            "_trace_prefix_identity_v25(trace)",
            "_publish_update400_recovery_v25(",
            '"event": "update400_recovery_written"',
        )
    )
    assert positions == tuple(sorted(positions))
    assert "if not scientific_decision[\"passed\"]" in branch
    assert branch.index("if not scientific_decision") < branch.index(
        "_publish_update400_recovery_v25("
    )
    assert "torch.load" not in source
    assert "load_state_dict" not in source
    assert "resume_from" not in source
    assert source.count(RECOVERY_STATE_LITERAL := "recovery/update_400_training_state.pt") == 1
    assert RECOVERY_STATE_LITERAL in source
