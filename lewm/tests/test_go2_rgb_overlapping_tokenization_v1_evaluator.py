from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_overlapping_tokenization_v1.py"


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class _Model:
    def __init__(self) -> None:
        self.mode_calls: list[str] = []

    def eval(self):
        self.mode_calls.append("eval")
        return self

    def train(self):
        self.mode_calls.append("train")
        return self


def test_evaluator_uses_static_physical_metrics_without_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_static_evaluator")
    frozen_sha256 = "4" * 64
    model_sha256 = "5" * 64
    model = _Model()
    selection_pairs = ({"dataset_role": "checkpoint_selection"},)
    physical = {
        scope: {"scope": scope} for scope in module.contract.SCOPES
    }
    calls: list[dict[str, object]] = []

    def physical_metrics(
        observed_model,
        observed_pairs,
        observed_device,
        *,
        arm,
        stage,
    ):
        calls.append({
            "model": observed_model,
            "pairs": observed_pairs,
            "device": observed_device,
            "arm": arm,
            "stage": stage,
        })
        return physical, 1.25

    monkeypatch.setattr(
        module,
        "_state_sha",
        lambda _runtime, _model: model_sha256,
    )
    monkeypatch.setattr(
        module._BASE,
        "_subset_sha",
        lambda _runtime, _model, _prefixes: frozen_sha256,
    )
    evaluation = {"synthetic_gate": "frozen"}
    monkeypatch.setattr(
        module.contract,
        "evaluate_physical_scopes",
        lambda observed: evaluation if observed is physical else None,
    )
    trainer = SimpleNamespace(physical_metrics=physical_metrics)

    metric = module._evaluate(
        SimpleNamespace(),
        trainer,
        model,
        selection_pairs,
        "cpu",
        update=400,
        frozen_sha256=frozen_sha256,
    )

    assert model.mode_calls == ["eval", "train"]
    assert calls == [{
        "model": model,
        "pairs": selection_pairs,
        "device": "cpu",
        "arm": module.ARM_NAME,
        "stage": "inline_checkpoint_selection_update_400",
    }]
    assert set(metric) == {
        "update",
        "role",
        "pair_count",
        "unique_endpoint_count",
        "scopes",
        "aggregate_complete_v4_tail_depth_loss",
        "evaluation",
        "preledger_model_state_checks_pass",
        "state_sha256_before",
        "state_sha256_after",
        "frozen_state_sha256_before_and_after",
        "state_mutation_count",
    }
    assert metric["scopes"] is physical
    assert metric["evaluation"] is evaluation
    assert metric["aggregate_complete_v4_tail_depth_loss"] == pytest.approx(
        1.25
    )
    assert metric["preledger_model_state_checks_pass"] is True
    assert metric["state_mutation_count"] == 0
    module._assert_static_payload(
        metric, name="synthetic inline evaluation"
    )


def test_evaluator_rejects_dynamic_metric_fields_from_physical_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_overlap_static_evaluator_guard")
    frozen_sha256 = "4" * 64
    monkeypatch.setattr(
        module, "_state_sha", lambda _runtime, _model: "5" * 64
    )
    monkeypatch.setattr(
        module._BASE,
        "_subset_sha",
        lambda _runtime, _model, _prefixes: frozen_sha256,
    )
    monkeypatch.setattr(
        module.contract,
        "evaluate_physical_scopes",
        lambda _physical: {"temporal_population": {}},
    )
    trainer = SimpleNamespace(
        physical_metrics=lambda *_args, **_kwargs: (
            {scope: {} for scope in module.contract.SCOPES},
            1.0,
        )
    )
    with pytest.raises(PermissionError, match="retained dynamic field"):
        module._evaluate(
            SimpleNamespace(),
            trainer,
            _Model(),
            (),
            "cpu",
            update=100,
            frozen_sha256=frozen_sha256,
        )
