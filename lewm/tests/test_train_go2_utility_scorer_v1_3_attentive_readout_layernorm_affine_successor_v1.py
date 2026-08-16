from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lewm.oracle import (
    go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract
    as C,
)
from scripts import (
    train_go2_utility_scorer_v1_3_attentive_readout_layernorm_affine_successor_v1
    as R,
)
from scripts import train_go2_utility_scorer_v1_2 as FROZEN


ROOT = Path(__file__).resolve().parents[2]


def source() -> str:
    return (ROOT / C.NEW_SOURCE_PATHS[2]).read_text()


def test_factory_preserves_state_and_exact_context_boundary() -> None:
    model = R.model_factory()
    assert FROZEN.state_dict_digest(FROZEN._cpu_state(model)) == \
        C.INITIAL_STATE_DIGEST
    inventory = R.implementation_inventory(model)
    assert inventory["compatible_paths"] == list(C.LN.LAYER_NORM_PATHS)
    assert inventory["native_modules_preserved_outside_forward_context"] is True
    module = R._module_at(model, C.LN.LAYER_NORM_PATHS[0])
    original = module.forward
    with R.LN_DIAGNOSTIC.externalised_layernorms(model):
        assert module.forward != original
    assert module.forward == original
    assert list(model.state_dict()) == inventory["state_dict_keys"]


def test_training_context_covers_forward_and_backward() -> None:
    text = source()
    body = text[text.index("def train_once("):text.index("def strict_checkpoint_reload")]
    assert "with LN_DIAGNOSTIC.externalised_layernorms(model):" in body
    context = body.index("with LN_DIAGNOSTIC.externalised_layernorms(model):")
    forward = body.index("outputs = model", context)
    backward = body.index("loss.backward()", forward)
    assert context < forward < backward
    loop = body[body.index("for epoch in range"):]
    assert "_load_corpus" not in loop
    assert "_evaluate_streaming" not in loop


def test_fit_loader_never_parses_global_view_or_index() -> None:
    text = source()
    body = text[text.index("def load_fit_only_training_corpus"):
                text.index("def fresh_initialisation")]
    assert "CONTRACT.fit_only_ledger()" in body
    assert "load_full_bank_v2_branch_runtime_authority" not in body
    assert "json.loads(index_path.read_text())" not in body
    assert "_load_corpus" not in body
    assert "_materialise_preserved_training_view" not in body
    assert '"calibration_row_records_opened": 0' in body
    assert '"calibration_overlay_records_opened": 0' in body
    assert '"calibration_latent_shards_opened": 0' in body
    assert '"global_latent_index_bytes_read": False' in body
    assert "SOURCE_KIND_V2_VALID" in body


def test_calibration_is_after_signed_final_checkpoint_authority() -> None:
    text = source()
    body = text[text.index("def execute_once"):text.index("def validate_result")]
    checkpoint = body.index("strict_checkpoint_reload")
    authority = body.index("publish_json_once(evaluation_path")
    full_corpus = body.index("PREVIOUS._load_corpus")
    forward = body.index("BASE._evaluate_streaming")
    evidence = body.index("publish_json_once(evidence_path")
    assert checkpoint < authority < full_corpus < forward < evidence
    context = body.rfind("with LN_DIAGNOSTIC.externalised_layernorms", 0, forward)
    assert authority < context < forward


def test_loss_and_interpretation_are_frozen() -> None:
    outputs = (torch.tensor([1.0, 2.0]), torch.tensor([0.1, -0.2]),
               torch.tensor([0.4, -0.3]))
    targets = {"progress": torch.tensor([0.0, 1.0]),
               "safety": torch.tensor([1.0, 0.0]),
               "completion": torch.tensor([0.0, 1.0])}
    indices = torch.tensor([0, 1])
    expected = (torch.nn.functional.mse_loss(
        outputs[0], targets["progress"], reduction="sum")
        + torch.nn.functional.binary_cross_entropy_with_logits(
            outputs[1], targets["safety"], reduction="sum")
        + torch.nn.functional.binary_cross_entropy_with_logits(
            outputs[2], targets["completion"], reduction="sum")) / 64
    assert torch.equal(R._loss(outputs, targets, indices), expected)
    all_pass = {f"gate_{index}": True for index in range(8)}
    assert R.decision(all_pass, 0.76, 0.06)["classification"] == \
        "STRONG_READOUT_SIGNAL"
    assert R.decision(all_pass, 0.70, 0.04)["classification"] == \
        "NO_READOUT_SIGNAL"
    mixed = dict(all_pass)
    mixed["gate_0"] = False
    assert R.decision(mixed, 0.76, 0.06)["classification"] == \
        "MIXED_READOUT_SIGNAL"


def test_validator_replays_evidence_without_model_forward() -> None:
    text = source()
    body = text[text.index("def validate_result"):text.index("def run_once")]
    assert "metrics_from_evidence" in body
    assert "_evaluate_streaming" not in body
    assert ".forward(" not in body
    assert "predictor_checkpoints_opened_for_utility" in body


def test_no_predictor_or_qualification_route_exists() -> None:
    text = source()
    assert "qualified_scorer_package_published\": False" in text
    assert "predictor_checkpoints_opened_for_utility\": 0" in text
    assert "final_200_state_corpus_generated\": False" in text
    assert "predictor checkpoint" not in text.lower()


def test_complete_technical_trace_and_terminal_fields_are_validated() -> None:
    trace = R.expected_technical_trace()
    assert len(trace) == 60
    assert trace[0] == {
        "epoch": 1, "completed_optimizer_updates": 18,
        "technical_finite": True, "performance_metric_inspected": False,
        "calibration_opened": False,
    }
    assert trace[-1]["completed_optimizer_updates"] == 1_080
    changed = [dict(row) for row in trace]
    changed[-1]["completed_optimizer_updates"] -= 1
    assert changed != R.expected_technical_trace()

    body = source()[source().index("def validate_result"):
                    source().index("def validate_failure")]
    for required in (
        'result.get("label") == CONTRACT.RESULT_LABEL',
        'result.get("implementation_name")',
        'checkpoint.get("registered_seed") == CONTRACT.SCORER_SEED',
        'checkpoint.get("data_order_seed") == CONTRACT.DATA_ORDER_SEED',
        'checkpoint.get("technical_trace") == expected_technical_trace()',
        'attempt.get("effective_batch") == CONTRACT.EFFECTIVE_BATCH',
        'attempt.get("fixed_final_epoch") == CONTRACT.EPOCHS',
        '"calibration_model_forwards_before_authorisation") == 0',
        '"calibration_predictions_before_authorisation") == 0',
        '"calibration_metrics_before_authorisation") == 0',
        '"persist_closed_prediction_target_evidence") is True',
        'result.get("predictor_retrained") is False',
        'result.get("final_200_state_corpus_generated") is False',
        '"nothing_left_running_by_this_process_after_exit") is True',
    ):
        assert required in body


def test_existing_failure_refuses_any_second_execution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "runtime_root", lambda root: tmp_path)
    (tmp_path / "technical_failure.json").write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr(R, "validate_failure",
                        lambda root: calls.append("validated_failure") or {})
    monkeypatch.setattr(R, "execute_once",
                        lambda root, custody: calls.append("executed") or {})
    monkeypatch.setattr(R, "_record_failure",
                        lambda root, stage, error, custody:
                        calls.append("preserved_existing_failure"))
    with pytest.raises(R.ScientificSuccessorError,
                       match="terminally failed"):
        R.run_once(tmp_path)
    assert calls == ["validated_failure", "preserved_existing_failure"]


def test_result_validation_failure_is_terminally_receipted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "runtime_root", lambda root: tmp_path)
    (tmp_path / "result.json").write_text("{}")
    observed: dict[str, object] = {}

    def fail_validation(root: Path) -> dict[str, object]:
        raise R.ScientificSuccessorError("tampered result")

    def record(root: Path, stage: str, error: BaseException,
               custody: dict[str, object]) -> None:
        observed.update(stage=stage, error=str(error))

    monkeypatch.setattr(R, "validate_result", fail_validation)
    monkeypatch.setattr(R, "_record_failure", record)
    with pytest.raises(R.ScientificSuccessorError, match="tampered result"):
        R.run_once(tmp_path)
    assert observed == {"stage": "result_validation", "error": "tampered result"}
