from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "scripts/run_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V6_"
    "PHASE_SEPARATED_FROZEN_STATE_PREDICTION_PREFLIGHT_JSON"
)


def _load(name: str = "_direct_bev_v6_phase_runner_test") -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_isolated_import_is_stdlib_only_and_exactly_rebound() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_v6_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert not any(name.startswith('numpy.') for name in sys.modules)
assert 'PIL' not in sys.modules
owners = (
    module._V5, module._V5._V4, module._V5._V4._V3,
    module._V5._V4._V3._V2, module._V5._V4._V3._V2._V1,
)
assert all(owner.contract is module.contract for owner in owners)
assert all(owner.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r} for owner in owners)
assert all(Path(owner.__file__).resolve() == path for owner in owners)
deepest = module._V5._V4._V3._V2._V1
assert deepest._initialize_model is module._v6_initialize_model
assert deepest._build_optimizer is module._v6_build_optimizer
assert deepest._gradient_integrity_probe is module._v6_gradient_integrity_probe
assert deepest._evaluate_observation_impl is module._v6_evaluate_observation_impl
assert deepest._train_probe is module._v6_train_probe
assert deepest._write_training_trace is module._v6_write_training_trace
assert deepest._snapshot_model is module._v6_snapshot_model
assert deepest._terminal_failure is module._v6_terminal_failure
assert deepest.contract.validate_failure_status_chain is module.contract.validate_failure_status_chain
args = module.parse_args([
    '--run',
    '--review-sha256', '0' * 64,
    '--authorization-sha256', '1' * 64,
])
assert args.review_sha256 == '0' * 64
assert args.authorization_sha256 == '1' * 64
print('PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("update", "expected"),
    (
        (0, (0, 0, 0, 0, 0, 0)),
        (100, (100, 100, 0, 100, 0, 0)),
        (400, (400, 400, 0, 400, 1, 0)),
        (401, (401, 400, 1, 400, 1, 1)),
        (1_000, (1_000, 400, 600, 400, 1, 600)),
    ),
)
def test_phase_accounting_is_exact(update: int, expected: tuple[int, ...]) -> None:
    runner = _load(f"_v6_phase_accounting_{update}")
    receipt = runner._phase_accounting_for_update(update)
    assert tuple(receipt.values()) == expected
    assert receipt == {
        "target_update_callback_count": expected[0],
        "perception_optimizer_updates": expected[1],
        "predictor_optimizer_updates": expected[2],
        "ema_arithmetic_updates": expected[3],
        "boundary_hard_sync_count": expected[4],
        "phase_two_target_noop_count": expected[5],
    }


def test_phase_accounting_rejects_invalid_updates() -> None:
    runner = _load("_v6_phase_accounting_invalid")
    for value in (-1, 1_001, 1.0, True):
        with pytest.raises(ValueError):
            runner._phase_accounting_for_update(value)


class _ReceiptModel:
    def __init__(self, update: int) -> None:
        self.update = update
        self._v6_optimizer_for_integrity_probe = object()

    def phase_counters_v6(self) -> dict[str, int | bool]:
        return {
            "phase_policy_armed": True,
            "global_target_update_callback_count": self.update,
            "target_update_callback_count": self.update,
            "ema_arithmetic_update_count": min(self.update, 400),
            "boundary_hard_sync_count": int(self.update >= 400),
            "phase_two_target_noop_count": max(self.update - 400, 0),
            "perception_optimizer_update_count": min(self.update, 400),
            "predictor_optimizer_update_count": max(self.update - 400, 0),
        }


def test_trace_rows_replace_ambiguous_ema_count(monkeypatch) -> None:
    runner = _load("_v6_trace_accounting")
    captured: dict[str, Any] = {}

    def frozen(output_root, rows):
        captured["root"] = output_root
        captured["rows"] = rows
        return {"path": "training_trace.json"}

    monkeypatch.setattr(runner, "_FROZEN_WRITE_TRAINING_TRACE", frozen)
    result = runner._v6_write_training_trace(
        Path("unused"),
        [{"update": 401, "ema_update_count": 401, "mean_G": 0.5}],
    )
    assert result == {"path": "training_trace.json"}
    row = captured["rows"][0]
    assert "ema_update_count" not in row
    assert row["mean_G"] == 0.5
    assert row["target_update_callback_count"] == 401
    assert row["ema_arithmetic_updates"] == 400
    assert row["boundary_hard_sync_count"] == 1
    assert row["phase_two_target_noop_count"] == 1


def test_snapshot_metadata_replaces_ambiguous_ema_count(monkeypatch) -> None:
    runner = _load("_v6_snapshot_accounting")
    captured: dict[str, Any] = {}

    def frozen(runtime, model, output_root, *, update, metadata):
        captured.update({"update": update, "metadata": metadata})
        return {"update": update}

    monkeypatch.setattr(runner, "_FROZEN_SNAPSHOT_MODEL", frozen)
    result = runner._v6_snapshot_model(
        object(),
        _ReceiptModel(1_000),
        Path("unused"),
        update=1_000,
        metadata={"ema_updates": 1_000, "gate": {"passed": True}},
    )
    assert result == {"update": 1_000}
    metadata = captured["metadata"]
    assert "ema_updates" not in metadata
    assert metadata["target_update_callback_count"] == 1_000
    assert metadata["ema_arithmetic_updates"] == 400
    assert metadata["phase_two_target_noop_count"] == 600


def test_terminal_probe_publishes_actual_ema_and_global_callbacks(
    monkeypatch,
) -> None:
    runner = _load("_v6_terminal_accounting")
    model = _ReceiptModel(1_000)
    source = {
        "updates": 1_000,
        "ema_updates": 1_000,
        "presentations": 16_000,
    }
    monkeypatch.setattr(
        runner,
        "_FROZEN_TRAIN_PROBE",
        lambda *args, **kwargs: (model, dict(source)),
    )
    returned_model, receipt = runner._v6_train_probe()
    assert returned_model is model
    assert receipt["global_target_update_callback_count"] == 1_000
    assert receipt["target_update_callback_count"] == 1_000
    assert receipt["ema_updates"] == 400
    assert receipt["ema_arithmetic_updates"] == 400
    assert receipt["perception_optimizer_updates"] == 400
    assert receipt["predictor_optimizer_updates"] == 600
    assert receipt["phase_two_target_noop_count"] == 600


def test_operational_failure_translates_partial_phase_counts(monkeypatch) -> None:
    runner = _load("_v6_failure_accounting")
    captured: dict[str, Any] = {}

    def frozen(output_root, reservation, reservation_raw, *, error, progress):
        captured.update(progress)

    monkeypatch.setattr(runner, "_FROZEN_TERMINAL_FAILURE", frozen)
    runner._v6_terminal_failure(
        Path("unused"),
        {"status": "reserved"},
        b"reservation",
        error=RuntimeError("expected test failure"),
        progress={
            "updates": 700,
            "optimizer_updates": 701,
            "ema_updates": 700,
        },
    )
    assert captured["global_target_update_callback_count"] == 700
    assert captured["target_update_callback_count"] == 700
    assert captured["perception_optimizer_updates"] == 400
    assert captured["predictor_optimizer_updates"] == 301
    assert captured["ema_updates"] == 400
    assert captured["ema_arithmetic_updates"] == 400
    assert captured["boundary_hard_sync_count"] == 1
    assert captured["phase_two_target_noop_count"] == 300


def test_normalized_online_and_target_hash_namespaces_match() -> None:
    runner = _load("_v6_normalized_state_hash")

    class Model:
        def state_dict(self):
            return {
                "encoder.weight": torch.tensor([1.0, 2.0]),
                "bev_decoder.bias": torch.tensor([3.0]),
                "state_head.weight": torch.tensor([4.0]),
                "target_encoder.weight": torch.tensor([1.0, 2.0]),
                "target_bev_decoder.bias": torch.tensor([3.0]),
                "target_state_head.weight": torch.tensor([4.0]),
                "predictor.weight": torch.tensor([5.0]),
            }

    def tensor_state_dict_sha256(state):
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            tensor_state_dict_sha256=tensor_state_dict_sha256
        ),
    )
    model = Model()
    online = runner._normalized_state_sha256(
        runtime, model, runner._ONLINE_PERCEPTION_PREFIXES
    )
    target = runner._normalized_state_sha256(
        runtime, model, runner._TARGET_PERCEPTION_PREFIXES
    )
    predictor = runner._normalized_state_sha256(
        runtime, model, runner._PREDICTOR_PREFIXES
    )
    assert online == target
    assert predictor != online
    assert len(online) == len(predictor) == 64


def test_optimizer_hash_is_stable_and_sensitive() -> None:
    runner = _load("_v6_optimizer_hash")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    before = runner._optimizer_sha256(optimizer)
    assert runner._optimizer_sha256(optimizer) == before
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    after = runner._optimizer_sha256(optimizer)
    assert after != before
