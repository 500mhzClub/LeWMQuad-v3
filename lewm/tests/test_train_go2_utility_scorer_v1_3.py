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
        "issue_training_integrity_replacement_authorisation(")
    train_call = source.index("FROZEN_TRAINER.train_registered_model(")
    evaluation_authorisation = source.index(
        "label=\"v1.3 one-shot qualification evaluation authorisation\"")
    evaluate_call = source.index("FROZEN_TRAINER.evaluate_model(")
    assert degeneracy < registration < training_authorisation < features
    assert features < train_call
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
    replacement = {
        CONTRACT.INVALID_TRAINING_ATTEMPT_RECEIPT_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_AUTHORISATION_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_QUALIFICATION_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_EVALUATION_AUTHORISATION_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_PACKAGE_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_PACKAGE_RECEIPT_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_BASELINE_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_BASELINE_RECEIPT_PATH,
        CONTRACT.SCORER_TRAINING_REPLACEMENT_FAILED_SCORER_PATH,
    }
    assert all(CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT_ROOT in path.parents
               for path in replacement)
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


def test_replacement_amendment_does_not_change_frozen_science_contract():
    assert CONTRACT.contract_digest() == (
        "93532f22a0cbc0e57ccdab3d5c01419cd824bc402d637738c5004eb621c23a89")
    amendment = CONTRACT.SCORER_TRAINING_INTEGRITY_REPLACEMENT
    assert amendment["invalid_original_attempt"]["status"] == (
        "INVALID_TECHNICAL_PREQUALIFICATION_ADAMW_SCALAR_STATE_SERIALIZATION")
    assert amendment["replacement_attempt"] == 1
    assert amendment["maximum_authorised_replacement_attempts"] == 1
    assert amendment["performance_based_authorisation"] is False
    assert amendment["further_replacement_automatically_permitted"] is False


def test_replacement_authorisation_binds_lineage_and_forbids_another_attempt():
    trainer = _trainer()
    initialisations = {
        "latent": {"path": "latent.pt", "sha256": "a" * 64,
                   "initial_state_digest": "b" * 64},
        "no_latent": {"path": "no_latent.pt", "sha256": "c" * 64,
                      "initial_state_digest": "d" * 64},
    }
    invalid = {trainer.INVALID_ATTEMPT_SELF_KEY: "e" * 64,
               "status": CONTRACT.INVALID_TRAINING_ATTEMPT_STATUS,
               "exception": {"type": "RuntimeError", "message": "scalar"}}
    binding = {
        "architecture": {}, "normalisation": {}, "utility_weights": {},
        "training": {}, "training_execution_counts": {},
        "qualification_thresholds": {}, "learning_rate_schedule": "constant",
        "final_epoch_only": True, "epoch_selection_permitted": False,
        "model_specific_calibration": None,
    }
    value = trainer._replacement_authorisation_payload(
        invalid_attempt=invalid, source={"source_commit": "f" * 40},
        science={"training_view_digest": "1" * 64}, binding=binding,
        binding_digest="2" * 64, training_run_digest="3" * 64,
        initialisations=initialisations, data_order={"seed": 20260811},
        smoke={"passed": True})
    assert trainer._validate_signed(
        value, trainer.REPLACEMENT_AUTHORISATION_SELF_KEY,
        "synthetic replacement") == value
    assert value["replacement_attempt_number"] == 1
    assert value["maximum_authorised_replacement_attempts"] == 1
    assert value["reuse_original_eighteen_updates"] is False
    assert value["authorised_because_of_model_performance"] is False
    assert value["further_replacement_automatically_permitted"] is False


def test_replacement_preflight_accepts_only_preserved_attempt_zero(tmp_path: Path):
    trainer = _trainer()
    model_root = (tmp_path / CONTRACT.SCORER_TRAINING_REPLACEMENT_CHECKPOINTS_ROOT
                  / "latent")
    attempt = model_root / "attempt_000"
    attempt.mkdir(parents=True)
    (attempt / "attempt.json").write_text("{}")
    common = dict(
        use_latent=True, registration={}, training_run_digest="a" * 64,
        device=trainer.torch.device("cpu"), budget={}, training_rows=1152,
        root=tmp_path)
    trainer._preflight_registered_training("latent", **common)
    (model_root / "attempt_001").mkdir()
    with pytest.raises(trainer.V13TrainingError, match="attempt_000"):
        trainer._preflight_registered_training("latent", **common)


def test_replacement_loader_bypasses_only_old_live_source_equality():
    source = _function_source(
        "load_preserved_encoded_training_view_for_replacement")
    bridge = _function_source(
        "_validate_preserved_workflow_inputs_for_replacement")
    assert "load_and_validate_encoded_training_view_for_consumption" not in source
    assert "root=None" in bridge
    for validation in (
        "validate_equivalence_receipt(",
        "validate_fresh_selection_attempt(",
        "validate_fresh_selection_terminal(",
        "validate_fresh_calibration_manifest(",
        "validate_latent_index(",
    ):
        assert validation in source + bridge
