"""Focused source/synthetic tests for the exploratory ViT-g scorer."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / (
    "scripts/train_go2_utility_scorer_vjepa2_1_vitg_ablation_v1.py")


def _source() -> str:
    return SOURCE_PATH.read_text()


def _function_source(name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    node = next(value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def _trainer():
    pytest.importorskip("torch")
    return importlib.import_module(
        "scripts.train_go2_utility_scorer_vjepa2_1_vitg_ablation_v1")


def test_source_has_no_predictor_planner_or_final_corpus_route():
    source = _source()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports
    assert "planner" not in imports
    assert "counterfactual" not in imports
    assert "final_200_state_corpus_generated\": True" not in source
    assert "qualified_scorer_package_published\": True" not in source
    assert "EXPLORATORY_ENCODER_SCALE_ABLATION" in source


def test_only_the_latent_input_projection_changes_shape():
    trainer = _trainer()
    trainer.FROZEN.configure_determinism(trainer.SCORER_SEED)
    vitl = trainer.FROZEN.UtilityScorer(use_latent=True)
    source_state = trainer.FROZEN._cpu_state(vitl)
    target, receipt = trainer.build_dimension_aware_initial_state(source_state)
    changed = []
    for name, source in source_state.items():
        value = target[name]
        if source.shape == value.shape:
            assert source.dtype == value.dtype
            assert source.equal(value)
        else:
            changed.append(name)
    assert changed == ["per_horizon.0.weight"]
    assert target["per_horizon.0.weight"].shape == (512, 1408)
    assert receipt["dimension_changed_parameters"] == changed
    assert receipt["parameter_count_increase"] == 512 * (1408 - 1024)


def test_dimension_aware_projection_is_repeatable_and_keyed_by_shape():
    trainer = _trainer()
    trainer.FROZEN.configure_determinism(trainer.SCORER_SEED)
    source = trainer.FROZEN._cpu_state(
        trainer.FROZEN.UtilityScorer(use_latent=True))
    first, first_receipt = trainer.build_dimension_aware_initial_state(source)
    second, second_receipt = trainer.build_dimension_aware_initial_state(source)
    assert trainer.FROZEN.state_dict_digest(first) \
        == trainer.FROZEN.state_dict_digest(second)
    assert first["per_horizon.0.weight"].equal(
        second["per_horizon.0.weight"])
    assert first_receipt == second_receipt
    source_digest = trainer.FROZEN.state_dict_digest(source)
    expected = trainer.dimension_aware_projection_key(
        source_state_digest=source_digest, token_dim=1408)
    changed_shape = trainer.dimension_aware_projection_key(
        source_state_digest=source_digest, token_dim=1409)
    changed_source = trainer.dimension_aware_projection_key(
        source_state_digest="f" * 64, token_dim=1408)
    assert expected["key_digest"] != changed_shape["key_digest"]
    assert expected["key_digest"] != changed_source["key_digest"]


def test_materialised_features_keep_frozen_action_goal_and_target_contract():
    trainer = _trainer()
    torch = pytest.importorskip("torch")

    class Store:
        def __getitem__(self, positions):
            count = len(np.asarray(positions).reshape(-1))
            value = np.zeros(
                (count, trainer.HORIZONS, trainer.TOKENS, trainer.TOKEN_DIM),
                dtype=np.float16)
            for index in range(count):
                value[index] = index + 1
            return value

    rows = [{
        "_latent_index": index,
        "action_blocks": [[float(block)] * 10 for block in range(4)],
        "goal_binding_input": [0.0, 1.0, 2.0],
        "progress": 0.1,
        "safety": 0.2,
        "completion": 0.0,
    } for index in range(2)]
    latent, action_goal, targets = trainer.materialise_features(
        rows, Store(), torch.device("cpu"), latent_chunk=2)
    assert latent.shape == (2, 4, 1408)
    assert action_goal.shape == (2, 43)
    assert set(targets) == {"progress", "safety", "completion"}
    assert all(value.dtype == torch.float32 for value in targets.values())


def test_frozen_budget_thresholds_and_baseline_are_not_retrained():
    trainer = _trainer()
    budget = trainer.frozen_budget()
    assert (budget["epochs"], budget["batch"], budget["seed"]) \
        == (60, 64, 20260811)
    assert trainer.TOTAL_UPDATES == 1080
    assert trainer.PRESENTATIONS == 69120
    assert trainer.PRIMARY_THRESHOLDS == {
        "safety_auc_min": 0.75,
        "latent_over_baseline_pairwise_gain_min": 0.05,
    }
    run_source = _function_source("run_once")
    assert "train_latent_once(" in run_source
    assert run_source.count("train_latent_once(") == 1
    assert "UtilityScorer(use_latent=False)" not in run_source
    assert "no_latent" not in _function_source("train_latent_once")


def test_exact_completed_vitl_and_baseline_lineage_constants():
    trainer = _trainer()
    assert trainer.FROZEN_V13_TERMINAL_DIGEST == (
        "441f52d4199ba152825f30a9f5422b80537f68b9f7a3633f4e01610f964de419")
    assert trainer.FROZEN_BASELINE_RECEIPT_DIGEST == (
        "454bc81c3077d62cac661a4ccac7212b3eb3860eda3177f9b8879f27632abc25")
    assert trainer.FROZEN_BASELINE_SHA256 == (
        "cfd07d2ad739ef884f3d8ebc3faa01a0b807ef6f19049874eb7fc6ecc9c418ca")
    assert trainer.FROZEN_BASELINE_STATE_DIGEST == (
        "33e7bcffbfab16371fb8e7e233490c33c442336edac823c19733214fa87d91d1")


@pytest.mark.parametrize(
    ("auc", "gain", "old_auc", "old_gain", "expected"), (
        (0.76, 0.06, 0.70, 0.03, "STRONG_SCALING_SIGNAL"),
        (0.76, 0.04, 0.70, 0.03, "MIXED_SIGNAL"),
        (0.72, 0.04, 0.70, 0.03, "NO_SCALING_SIGNAL"),
        (0.76, 0.06, 0.77, 0.03, "NO_SCALING_SIGNAL"),
    ))
def test_predeclared_decision_is_not_redefined(
        auc: float, gain: float, old_auc: float, old_gain: float,
        expected: str):
    trainer = _trainer()
    result = trainer.exploratory_decision(
        safety_auc=auc, pairwise_gain=gain,
        vitl_safety_auc=old_auc, vitl_pairwise_gain=old_gain)
    assert result["classification"] == expected


def test_calibration_features_are_materialised_only_after_training_and_auth():
    source = _function_source("run_once")
    training = source.index("train_latent_once(")
    authorisation = source.index("_authorise_evaluation(")
    calibration = source.index("corpus[\"calibration_rows\"]")
    evaluate = source.index("calibration_metrics, predictions")
    assert training < authorisation < calibration < evaluate
    assert source.count("calibration_metrics, predictions") == 1


def test_output_paths_are_all_under_the_separate_vitg_scorer_namespace():
    trainer = _trainer()
    root = Path("/synthetic/repository")
    expected_root = root / (
        ".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1/scorer")
    for path in (
        trainer.contract_path(root), trainer.initialisation_path(root),
        trainer.attempt_root(root), trainer.final_checkpoint_path(root),
        trainer.evaluation_authorisation_path(root),
        trainer.terminal_path(root), trainer.technical_failure_path(root),
    ):
        assert path == expected_root or expected_root in path.parents
