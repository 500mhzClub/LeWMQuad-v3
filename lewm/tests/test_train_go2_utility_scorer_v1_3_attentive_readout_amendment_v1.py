"""Focused tests for the amended sole attentive-readout execution."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from scripts import (
    train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1 as RUNNER,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py")


def _function_source(name: str) -> str:
    source = SOURCE.read_text()
    node = next(value for value in ast.parse(source).body
                if isinstance(value, ast.FunctionDef) and value.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def test_runtime_namespace_is_distinct_from_unconsumed_original() -> None:
    synthetic = Path("/tmp/synthetic-attentive-amendment")
    assert RUNNER.runtime_root(synthetic).parts[-1] == (
        "attentive_readout_amendment_v1")
    assert RUNNER.runtime_root(synthetic) != (
        synthetic / RUNNER.CONTRACT.GENERATED_ROOT / "attentive_readout")
    assert RUNNER.attempt_root(synthetic).parts[-2:] == (
        "training", "attempt_000")


def test_device_contract_is_exactly_cuda0_r9700_gfx1201_two_hip_devices() -> None:
    source = _function_source("device_preflight")
    for value in (
        'count == 2', 'torch.device("cuda:0")',
        'name == "AMD Radeon AI PRO R9700"',
        'architecture == "gfx1201"',
    ):
        assert value in source
    assert "cpu" not in source.lower()


def test_smoke_is_fit_only_one_real_update_and_discards_state() -> None:
    source = _function_source("run_production_smoke")
    assert "_fit_only_smoke_fixture(root)" in source
    assert "_load_corpus(" not in source
    assert 'corpus["calibration_rows"]' not in source
    assert "BASE._token_batch(" in source
    assert source.count("optimiser.step()") == 1
    assert "gradient_norm > 0.0" in source
    assert "structured_digest(optimiser_state)" in source
    assert "reload_optimiser.load_state_dict" in source
    assert '"calibration_latent_rows_opened": 0' in source
    assert '"calibration_evaluations": 0' in source
    assert '"smoke_model_and_optimizer_discarded": True' in source
    assert '"smoke_state_reuse_permitted": False' in source


def test_fit_only_smoke_fixture_opens_only_four_bound_fit_files() -> None:
    rows, store, binding = RUNNER._fit_only_smoke_fixture(ROOT)
    assert len(rows) == 4 and store.shape == (4, 4, 768, 1024)
    assert binding["fixture_digest"] == RUNNER.SMOKE_FIT_FIXTURE_DIGEST
    assert binding["row_record_files_opened"] == 4
    assert binding["fit_latent_shards_opened"] == 4
    assert binding["calibration_rows_materialized"] == 0
    assert binding["calibration_label_rows_opened"] == 0
    assert binding["calibration_latent_shards_opened"] == 0


def test_smoke_binds_projection_attention_and_distinct_query_gradients() -> None:
    source = _function_source("_smoke_gradient_evidence")
    assert "model.token_projection.weight.grad" in source
    assert '"pooler.cross_attention_block.xattn."' in source
    assert '".attn."' in source
    assert "model.pooler.query_tokens.grad" in source
    assert 'tuple(query.shape) == (1, 3, 512)' in source
    assert "all_pairwise_distinct" in source


def test_scientific_initialisation_is_fresh_and_never_loads_smoke_state() -> None:
    source = _function_source("build_initialisation")
    assert "_fresh_model_state()" in source
    assert '"smoke_state_reused": False' in source
    assert "smoke_checkpoint_path" not in source
    training = _function_source("train_once")
    assert '"smoke_checkpoint_used": False' in training
    assert '"smoke_state_used": False' in training


def test_training_contract_is_exact_microbatch_accumulation_and_final_epoch() -> None:
    source = _function_source("train_once")
    assert "range(1, EPOCHS + 1)" in source
    assert "range(0, FIT_ROWS, EFFECTIVE_BATCH)" in source
    assert "range(0, EFFECTIVE_BATCH, MICROBATCH)" in source
    assert "loss.backward()" in source
    assert "clip_grad_norm_" in source
    assert source.count("optimiser.step()") == 1
    assert "completed_updates == TOTAL_UPDATES" in source
    assert '"final_epoch_only_no_selection"' in source
    assert '"performance_metric_inspected": False' in source
    assert '"calibration_opened": False' in source


def test_scientific_fit_loader_never_reads_calibration_latent_shards() -> None:
    source = _function_source("_load_fit_training_corpus")
    assert "load_preserved_encoded_training_view_for_replacement" not in source
    assert 'if V13.ENCODER._row_role(row) != "fit"' in source
    assert "_valid_latent_record(" in source
    assert '"calibration_latent_shards_read": 0' in source
    run = _function_source("_execute_once")
    assert run.index("_load_fit_training_corpus(root)") < run.index(
        "train_once(")
    assert run.index("publish_json(evaluation_path(root)") < run.index(
        "corpus = _load_corpus(root)")


def test_calibration_is_forwarded_once_then_closed_evidence_is_published() -> None:
    run = _function_source("_execute_once")
    validate = _function_source("validate_result_for_consumption")
    assert run.count("BASE._evaluate_streaming(") == 1
    assert run.index("evaluation_path(root)") < run.index(
        "BASE._evaluate_streaming(")
    assert run.index("BASE._evaluate_streaming(") < run.index(
        "publish_json(evidence_path(root)")
    assert "_evaluate_streaming(" not in validate
    assert "metrics_from_evidence(" in validate
    assert "model(" not in validate
    assert run.index('custody["calibration_evaluations"] = 1') < run.index(
        "BASE._evaluate_streaming(")
    assert run.index("BASE._evaluate_streaming(") < run.index(
        'custody["calibration_evaluation_completed"] = True')


def _synthetic_evidence(rows):
    payload = {
        "schema": RUNNER.EVIDENCE_SCHEMA,
        "status": RUNNER.STATUS,
        "complete": True,
        "execution_bindings": {"binding": "x"},
        "evaluation_authorisation_digest": "e" * 64,
        "final_checkpoint_sha256": "c" * 64,
        "final_state_digest": "s" * 64,
        "row_count": len(rows),
        "training_view_row_order_digest": RUNNER.AMENDMENT.digest([
            row["training_view_row_digest"] for row in rows]),
        "training_view_row_identity_set_digest": RUNNER.AMENDMENT.digest(sorted(
            row["training_view_row_digest"] for row in rows)),
        "branch_identity_set_digest": RUNNER.AMENDMENT.digest(sorted(
            row["branch_identity_digest"] for row in rows)),
        "rows": rows,
        "calibration_evaluation_session_count": 1,
        "model_forward_batch_count": RUNNER.CALIBRATION_FORWARD_BATCHES,
        "raw_latent_persisted": False,
        "predictor_material_accessed": False,
    }
    return RUNNER.signed(payload, RUNNER.EVIDENCE_SELF_KEY)


def test_closed_evidence_rejects_duplicate_or_changed_frozen_rows(
        monkeypatch) -> None:
    monkeypatch.setattr(RUNNER, "CALIBRATION_ROWS", 2)
    template = {
        "training_view_row_digest": "a" * 64,
        "branch_identity_digest": "b" * 64,
        "state_id": "state-0", "family": "family-a", "stratum": "s",
        "candidate_index": 0,
        "target": {"progress": 0.0, "safety": 0.0,
                   "completion": 0.0, "utility": 0.0},
        "prediction": {"progress": 0.1, "safety": 0.2,
                       "completion": 0.3, "utility": 0.4},
    }
    duplicate = _synthetic_evidence([copy.deepcopy(template),
                                     copy.deepcopy(template)])
    with pytest.raises(RUNNER.AttentiveAmendmentError,
                       match="calibration evidence binding changed"):
        RUNNER.metrics_from_evidence(
            corpus_rows=[], evidence=duplicate,
            bindings={"binding": "x"}, evaluation_digest="e" * 64,
            checkpoint_sha256="c" * 64, final_state_digest="s" * 64)

    second = copy.deepcopy(template)
    second["training_view_row_digest"] = "d" * 64
    second["branch_identity_digest"] = "f" * 64
    second["state_id"] = "state-1"
    second["candidate_index"] = 1
    evidence = _synthetic_evidence([copy.deepcopy(template), second])
    frozen = []
    for row in (template, second):
        frozen.append({
            key: row[key] for key in (
                "training_view_row_digest", "branch_identity_digest",
                "state_id", "family", "stratum", "candidate_index")
        })
        frozen[-1].update(row["target"])
    tampered = copy.deepcopy(evidence)
    tampered["rows"][1]["target"]["safety"] = 1.0
    unsigned = dict(tampered)
    unsigned.pop(RUNNER.EVIDENCE_SELF_KEY)
    tampered[RUNNER.EVIDENCE_SELF_KEY] = RUNNER.AMENDMENT.digest(unsigned)
    with pytest.raises(RUNNER.AttentiveAmendmentError,
                       match="row changed from frozen corpus"):
        RUNNER.metrics_from_evidence(
            corpus_rows=frozen, evidence=tampered,
            bindings={"binding": "x"}, evaluation_digest="e" * 64,
            checkpoint_sha256="c" * 64, final_state_digest="s" * 64)


def test_metric_replay_uses_closed_predictions_and_all_groupings() -> None:
    source = _function_source("metrics_from_evidence")
    assert "FROZEN._evaluate_arrays(" in source
    assert 'FROZEN._grouped_calibration(\n        corpus_rows, targets, predicted_arrays, "family")' in source
    assert 'FROZEN._grouped_calibration(\n        corpus_rows, targets, predicted_arrays, "stratum")' in source
    assert "_token_batch" not in source
    assert "_evaluate_streaming" not in source
    assert "model(" not in source


def test_section9_decision_uses_all_original_gates_and_full_vitl_values() -> None:
    assert RUNNER.ORIGINAL_SAFETY_AUC == 0.7043234198736978
    assert RUNNER.ORIGINAL_PAIRWISE_GAIN == 0.0317880794701987
    passing = {f"criterion_{index}": True for index in range(8)}
    strong = RUNNER.exploratory_decision(
        criteria=passing, safety_auc=0.8, pairwise_gain=0.08)
    assert strong["classification"] == "STRONG_READOUT_SIGNAL"
    failed_other = dict(passing)
    failed_other["criterion_7"] = False
    mixed = RUNNER.exploratory_decision(
        criteria=failed_other, safety_auc=0.8, pairwise_gain=0.08)
    assert mixed["classification"] == "MIXED_READOUT_SIGNAL"
    one_primary = RUNNER.exploratory_decision(
        criteria=failed_other, safety_auc=0.8, pairwise_gain=0.04)
    assert one_primary["classification"] == "MIXED_READOUT_SIGNAL"
    no_signal = RUNNER.exploratory_decision(
        criteria=failed_other, safety_auc=0.7, pairwise_gain=0.03)
    assert no_signal["classification"] == "NO_READOUT_SIGNAL"
    assert no_signal["per_family_consistency_is_report_only"] is True


def test_decimal_pairwise_gate_does_not_use_binary_rounding() -> None:
    criteria = {f"criterion_{index}": True for index in range(8)}
    at_gate = RUNNER.exploratory_decision(
        criteria=criteria, safety_auc=0.8, pairwise_gain=0.05)
    below = RUNNER.exploratory_decision(
        criteria=criteria, safety_auc=0.8,
        pairwise_gain=float("0.04999999999999999"))
    assert at_gate["latent_over_baseline_pairwise_gain_gate_met"] is True
    assert below["latent_over_baseline_pairwise_gain_gate_met"] is False


def test_per_family_is_reported_but_never_gates_decision() -> None:
    decision = _function_source("exploratory_decision")
    assert "family_report" not in decision
    assert "per_family_consistency" not in decision.split("return", 1)[0]
    run = _function_source("_execute_once")
    assert "BASE.per_family_primary_consistency(" in run
    assert '"per_family_consistency_is_report_only": True' in run


def test_no_predictor_planner_package_or_second_attempt_route() -> None:
    source = SOURCE.read_text()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports.lower()
    for forbidden in (
        "scorer_package_path(", "load_predictor", "open_predictor",
        "apply_predictor", "score_predictor",
        "final_200_state_corpus_generated\": True", "attempt_001",
        "best_epoch", "resume_from",
    ):
        assert forbidden not in source
    assert '"qualified_scorer_package_published": False' in source
    assert '"predictor_checkpoints_opened_for_utility": 0' in source


def test_terminal_validator_closes_all_frozen_lineage_and_training_receipts() -> None:
    source = _function_source("validate_result_for_consumption")
    for required in (
        "initial_path.is_file() and not initial_path.is_symlink()",
        'training.get("byte_count") == checkpoint_path.stat().st_size',
        'training.get("technical_validity") is True',
        'checkpoint.get("status") == STATUS',
        'checkpoint.get("attempt_number") == 1',
        'training.get("attempt_digest") == checkpoint.get("attempt_digest")',
        '== baseline["vitl_terminal_digest"]',
        '== vitg["latent_over_baseline_pairwise_gain"]',
        '== vitg["exploratory_decision"]["classification"]',
        "not technical_failure_path(root).exists()",
    ):
        assert required in source
    smoke = _function_source("run_production_smoke")
    assert "smoke_updates = 0" in smoke
    assert "smoke_updates = 1" in smoke
    assert "updates=smoke_updates" in smoke
