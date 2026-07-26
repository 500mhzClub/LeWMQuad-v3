from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from lewm.models import (
    direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor
    as v3_model,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT / "scripts/run_go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("_test_direct_bev_v4_hook_runner")


def test_isolated_import_is_source_only_and_deeply_rebound() -> None:
    program = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(RUNNER_PATH)!r})
spec = importlib.util.spec_from_file_location('_v4_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
assert module._V3.contract is module.contract
assert module._V3._V2.contract is module.contract
assert module._V3._V2._V1.contract is module.contract
assert module._V3._V2._V1._gradient_integrity_probe is module._v4_gradient_integrity_probe
assert module._V3._V2._V1._initialize_model is module._v4_initialize_model
assert pathlib.Path(module._V3._V2._V1.__file__).resolve() == path
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


def test_entrypoint_delegates_after_rebinding(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run_parent(
        *, review_file_sha256: str, authorization_file_sha256: str
    ) -> int:
        assert runner._V3.contract is runner.contract
        assert runner._V3._V2._V1._gradient_integrity_probe is (
            runner._v4_gradient_integrity_probe
        )
        calls.append((review_file_sha256, authorization_file_sha256))
        return 23

    monkeypatch.setattr(runner._V3, "run_parent", fake_run_parent)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 23
    assert calls == [("a" * 64, "b" * 64)]


class _FakeHead:
    in_channels = 16
    out_channels = 3
    kernel_size = (3, 3)
    padding = (1, 1)
    bias = object()


class _FakePredictor:
    def __init__(self) -> None:
        self.residual_head = _FakeHead()
        self.net = (self.residual_head,)


class _FakeModel:
    def __init__(self) -> None:
        self.predictor = _FakePredictor()
        self.marker = object()

    def parameters(self):
        return (self.marker,)

    def training_objective(self, **_kwargs):
        return self.marker


def test_model_view_changes_only_predictor_hook_witness() -> None:
    real = _FakeModel()
    view = runner._ResidualHeadHookModelView(real)
    assert view.predictor is real.predictor.residual_head
    assert view.marker is real.marker
    assert view.parameters.__self__ is real
    assert view.parameters.__func__ is real.parameters.__func__
    assert view.training_objective.__self__ is real
    assert view.training_objective.__func__ is real.training_objective.__func__
    with pytest.raises(AttributeError):
        view.marker = object()


def test_actual_v3_all_actions_outer_zero_residual_head_one_and_no_mutation() -> None:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(1701)
        predictor = v3_model._CoordinateAwareFilmUnetPredictorV3()
    finally:
        torch.random.set_rng_state(caller_rng)
    state = torch.linspace(-1.0, 1.0, 2 * 3 * 64 * 64).reshape(
        2, 3, 64, 64
    )
    before_state = {
        name: value.detach().clone()
        for name, value in predictor.state_dict().items()
    }
    rng_before_call = torch.random.get_rng_state().clone()
    with torch.no_grad():
        expected = predictor.predict_all_actions(state)
    counts = {"outer": 0, "residual_head": 0}
    outer_handle = predictor.register_forward_hook(
        lambda *_args: counts.__setitem__("outer", counts["outer"] + 1)
    )
    head_handle = predictor.residual_head.register_forward_hook(
        lambda *_args: counts.__setitem__(
            "residual_head", counts["residual_head"] + 1
        )
    )
    try:
        with torch.no_grad():
            observed = predictor.predict_all_actions(state)
    finally:
        outer_handle.remove()
        head_handle.remove()
    assert counts == {"outer": 0, "residual_head": 1}
    assert torch.equal(observed, expected)
    assert torch.equal(torch.random.get_rng_state(), rng_before_call)
    assert all(
        torch.equal(predictor.state_dict()[name], value)
        for name, value in before_state.items()
    )


def test_redirected_frozen_probe_observes_one_without_model_mutation(
    monkeypatch,
) -> None:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(1702)
        predictor = v3_model._CoordinateAwareFilmUnetPredictorV3()
    finally:
        torch.random.set_rng_state(caller_rng)

    class Real:
        def __init__(self) -> None:
            self.predictor = predictor
            self.marker = object()

        def parameters(self):
            return tuple(self.predictor.parameters())

    real = Real()
    state = torch.zeros(2, 3, 64, 64)

    def fake_frozen(_runtime, model, _partition, batch):
        assert isinstance(model, runner._ResidualHeadHookModelView)
        assert model.parameters.__self__ is real
        count = 0

        def counted(*_args):
            nonlocal count
            count += 1

        handle = model.predictor.register_forward_hook(counted)
        try:
            real.predictor.predict_all_actions(batch["state"])
        finally:
            handle.remove()
        return {
            "training_objective_call_counts": {
                "online_state_stack": 3,
                "predictor": count,
                "target_state_stack": 3,
            },
            "six_call_graph_isolation_exact": count == 1,
        }

    monkeypatch.setattr(
        runner,
        "_FROZEN_GRADIENT_INTEGRITY_PROBE",
        fake_frozen,
    )
    result = runner._v4_gradient_integrity_probe(
        None,
        real,
        {},
        {"state": state},
    )
    assert result == {
        "training_objective_call_counts": {
            "online_state_stack": 3,
            "predictor": 1,
            "target_state_stack": 3,
        },
        "six_call_graph_isolation_exact": True,
    }
    assert real.predictor is predictor


def test_frozen_probe_uses_model_predictor_only_as_one_hook_witness() -> None:
    tree = ast.parse(inspect.getsource(runner._FROZEN_GRADIENT_INTEGRITY_PROBE))
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "model"
        and node.attr == "predictor"
    ]
    assert len(uses) == 1
    assert isinstance(uses[0].ctx, ast.Load)


def test_fresh_initial_state_identity_is_fail_closed(monkeypatch) -> None:
    model = object()
    partition = {"groups": object()}
    good = {
        "complete_initial_state_sha256": (
            runner.contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
        )
    }
    monkeypatch.setattr(
        runner,
        "_FROZEN_INITIALIZE_MODEL",
        lambda *_args: (model, partition, dict(good)),
    )
    assert runner._v4_initialize_model(None, None, None, None) == (
        model,
        partition,
        good,
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_INITIALIZE_MODEL",
        lambda *_args: (
            model,
            partition,
            {"complete_initial_state_sha256": "0" * 64},
        ),
    )
    with pytest.raises(RuntimeError, match="fresh V4 initial model"):
        runner._v4_initialize_model(None, None, None, None)
