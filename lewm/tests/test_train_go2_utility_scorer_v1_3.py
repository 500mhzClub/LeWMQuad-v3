"""Focused source/synthetic tests for the one-shot oracle-v1.3 scorer."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as CONTRACT


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "scripts/train_go2_utility_scorer_v1_3.py"


def _source() -> str:
    return SOURCE_PATH.read_text()


def _function_source(name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    node = next(value for value in tree.body
                if isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef))
                and value.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def _trainer():
    pytest.importorskip("torch")
    return importlib.import_module("scripts.train_go2_utility_scorer_v1_3")


def test_wrapper_reuses_frozen_model_training_metrics_and_packager():
    source = _source()
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.ClassDef)
                   and node.name == "UtilityScorer" for node in tree.body)
    for call in (
        "FROZEN_TRAINER.features(",
        "FROZEN_TRAINER.register_initialisation(",
        "FROZEN_TRAINER.train_registered_model(",
        "FROZEN_TRAINER.evaluate_model(",
        "FROZEN_TRAINER.qualification_criteria(",
        "FROZEN_TRAINER._paired_baseline_diagnostics(",
        "FROZEN_TRAINER._write_once_torch(",
    ):
        assert call in source
    assert "torch.optim" not in source


def test_degeneracy_training_and_evaluation_gates_are_in_order():
    source = _function_source("train_and_qualify")
    degeneracy = source.index("full_bank_v2_completion_degeneracy(")
    features = source.index("FROZEN_TRAINER.features(")
    registration = source.index("FROZEN_TRAINER.register_initialisation(")
    training_authorisation = source.index(
        "label=\"v1.3 one-shot training authorisation\"")
    train_call = source.index("FROZEN_TRAINER.train_registered_model(")
    evaluation_authorisation = source.index(
        "label=\"v1.3 one-shot qualification evaluation authorisation\"")
    evaluate_call = source.index("FROZEN_TRAINER.evaluate_model(")
    assert degeneracy < features < registration < training_authorisation
    assert training_authorisation < train_call
    assert train_call < evaluation_authorisation < evaluate_call
    assert source.index("evaluation_authorisation_path(root)") < degeneracy


def test_one_shot_terminal_never_opens_or_authorises_downstream_artifacts():
    source = _source()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports
    assert "final_eval" not in source
    assert '"predictor_artifact_access_authorised": True' not in source
    assert source.count('"predictor_artifact_access_authorised": False') >= 2
    main = _function_source("main")
    assert '"predictor_utility_shards_opened": 0' in main
    assert '"final_200_state_corpus_generated": False' in main


def test_all_training_outputs_are_registered_under_the_closed_contract_root():
    expected = {
        CONTRACT.QUALIFICATION_PATH,
        CONTRACT.SCORER_PACKAGE_PATH,
        CONTRACT.SCORER_PACKAGE_RECEIPT_PATH,
        CONTRACT.NO_LATENT_BASELINE_PATH,
        CONTRACT.NO_LATENT_BASELINE_RECEIPT_PATH,
        CONTRACT.FAILED_SCORER_PATH,
        CONTRACT.TRAINING_EXECUTION_AUTHORISATION_PATH,
        CONTRACT.QUALIFICATION_EVALUATION_AUTHORISATION_PATH,
    }
    assert all(CONTRACT.GENERATED_ROOT in path.parents for path in expected)
    assert set(CONTRACT.OUTPUT_PATHS.values()) >= {str(path) for path in expected}
    assert CONTRACT.contract()[
        "qualification_pass_authorises_predictor_open_in_this_workflow"] is False
    assert CONTRACT.contract()["final_200_state_benchmark_authorised"] is False


def test_frozen_budget_thresholds_and_execution_counts_when_torch_available():
    trainer = _trainer()
    budget = trainer.frozen_training_budget()
    assert budget["epochs"] == 60
    assert budget["batch"] == 64
    assert budget["seed"] == 20260811
    assert trainer.validate_frozen_qualification_thresholds() \
        == CONTRACT.QUALIFICATION_THRESHOLDS
    counts = trainer.training_execution_counts()
    assert counts["optimizer_updates_per_epoch"] == 18
    assert counts["optimizer_updates_per_model"] == 1_080
    assert counts["example_presentations_per_model"] == 69_120
    assert counts["models"] == [
        "shared_true_latent_scorer", "no_latent_baseline"]


def test_qualification_delegates_exactly_once_when_torch_available(monkeypatch):
    trainer = _trainer()
    sentinel = ({"gate": True}, {"detail": 1}, 0.05)
    calls = []

    def frozen(*arguments):
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(trainer, "validate_frozen_qualification_thresholds",
                        lambda: dict(CONTRACT.QUALIFICATION_THRESHOLDS))
    monkeypatch.setattr(trainer.FROZEN_TRAINER, "qualification_criteria", frozen)
    inputs = ({"latent": 1}, {"baseline": 2}, {"fit": 3}, {"cal": 4})
    assert trainer.qualification_criteria(*inputs) == sentinel
    assert calls == [inputs]


def test_immutable_json_publication_is_idempotent_not_replaceable(
        tmp_path: Path):
    trainer = _trainer()
    path = tmp_path / "terminal.json"
    payload = {"schema": "synthetic", "qualified": False}
    trainer.publish_json_once(path, payload, label="synthetic terminal")
    trainer.publish_json_once(path, payload, label="synthetic terminal")
    assert json.loads(path.read_text()) == payload
    with pytest.raises(trainer.V13TrainingError):
        trainer.publish_json_once(
            path, {**payload, "qualified": True}, label="synthetic terminal")
