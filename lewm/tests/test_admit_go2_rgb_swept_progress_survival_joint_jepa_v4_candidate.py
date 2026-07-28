from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

from scripts import admit_go2_rgb_swept_progress_survival_joint_jepa_v4_candidate as admission


class _Prediction(NamedTuple):
    predicted_latents: torch.Tensor
    survival_logits: torch.Tensor


class _StubModel(torch.nn.Module):
    def __init__(self, encoder_state: dict[str, torch.Tensor], masks: torch.Tensor) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.predictor = torch.nn.Module()
        self.predictor.swept_progress_head = torch.nn.Module()
        self.predictor.swept_progress_head.register_buffer("sweep_masks", masks.clone())

    @property
    def action_vocabulary(self) -> tuple[str, ...]:
        return admission.ACTION_VOCABULARY

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.zeros((rgb.shape[0], 64, 64, 64), dtype=torch.float32)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros((latent.shape[0], 3, 64, 64), dtype=torch.float32)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> _Prediction:
        return _Prediction(
            torch.zeros((latent.shape[0], 9, 64, 64, 64), dtype=torch.float32),
            torch.zeros((latent.shape[0], 9, 16), dtype=torch.float32),
        )


def _payload() -> dict[str, Any]:
    model = _StubModel(
        {"weight": torch.tensor([[1.0]])},
        torch.ones((9, 16, 64, 64), dtype=torch.bool),
    )
    return {
        "schema": admission.CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "experiment_seed": 20_260_728,
        "initialization_source": "exact_n320_encoder_only",
        "predecessor_experiment_checkpoint_read": False,
        "auxiliary_objective": admission.AUXILIARY_OBJECTIVE,
        "initial_semantic_decoder": admission.INITIAL_DECODER_RECEIPT,
        "accounting": admission.ACCOUNTING,
        "model_state_dict": model.state_dict(),
    }


def _canonical_result(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    checks = {name: True for name in admission.GATE_CHECKS}
    core = {
        "schema": admission.RESULT_SCHEMA,
        "status": "PASS_FULL_ARM",
        "gate": {
            "status": "PASS_FULL_ARM", "passed": True, "checks": checks,
            "failed_checks": [], "thresholds": admission.GATE_THRESHOLDS,
        },
        "caps": {"updates": 1_000, "presentations": 16_000},
        "seeds": {
            "inherited_fresh_component_constructor": 20_260_712,
            "semantic_decoder": 20_260_713,
            "experiment_and_stochastic_execution": 20_260_728,
            "bootstrap": 20_260_728,
        },
        "scientific_change_from_v3": {
            "initial_semantic_decoder": admission.INITIAL_DECODER_RECEIPT,
            "auxiliary_objective_unchanged": admission.AUXILIARY_OBJECTIVE,
        },
        "training": {
            "accounting": admission.ACCOUNTING,
            "joint_from_update_one": True,
            "separate_head_or_predictor_training": False,
            "checkpoint": checkpoint,
        },
        "access": {"forbidden_input_count": 0, "g2_navigation_final_evaluation_open_count": 0},
        "authority": {
            "development_only": True,
            "g2_navigation_final_evaluation_opened": False,
            "heldout_or_sealed_opened": False,
            "checkpoint_qualified": False,
            "promotion_performed": False,
            "retry_or_resume_authorized": False,
        },
    }
    value = admission._with_content_hash(core)
    return value, admission._canonical_bytes(value) + b"\n"


def _fixture_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, Path]:
    source_bindings = []
    for index, role in enumerate(("v4_model", "v4_executor")):
        relative = f"source/{index}.py"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# {role}\n".encode()
        path.write_bytes(raw)
        source_bindings.append({
            "role": role, "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        })
    checkpoint_buffer = io.BytesIO()
    torch.save(_payload(), checkpoint_buffer)
    checkpoint_raw = checkpoint_buffer.getvalue()
    checkpoint_binding = {
        "path": "checkpoint_update_1000.pt",
        "byte_count": len(checkpoint_raw),
        "file_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
    }
    _, result_raw = _canonical_result(checkpoint_binding)
    attempt = tmp_path / "fixture/attempt_v1"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_bytes(result_raw)
    (attempt / checkpoint_binding["path"]).write_bytes(checkpoint_raw)
    monkeypatch.setattr(admission, "SOURCE_BINDINGS", tuple(source_bindings))
    monkeypatch.setattr(admission, "RESULT_RELATIVE_PATH", "fixture/attempt_v1/result.json")
    monkeypatch.setattr(admission, "OUTPUT_RELATIVE_PATH", "admission/attempt_v1")
    monkeypatch.setattr(admission, "EXPECTED_RESULT_FILE_SHA256", hashlib.sha256(result_raw).hexdigest())
    monkeypatch.setattr(admission, "EXPECTED_RESULT_CONTENT_SHA256", admission._parse_canonical(result_raw, name="fixture")["content_sha256"])
    monkeypatch.setattr(admission, "_model_class", lambda: _StubModel)
    return checkpoint_raw, attempt


def test_admits_one_bound_checkpoint_with_weights_only_cpu_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_raw, _ = _fixture_tree(tmp_path, monkeypatch)
    original_load = torch.load
    calls: list[dict[str, Any]] = []

    def observed_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(admission.torch, "load", observed_load)
    result = admission.admit_candidate(repository_root=tmp_path)
    output = tmp_path / "admission/attempt_v1"
    assert result["status"] == "ADMITTED_PRE_G2_CANDIDATE"
    assert (output / "candidate_checkpoint.pt").read_bytes() == checkpoint_raw
    assert (output / "candidate_receipt.json").is_file()
    assert not (output / "failure.json").exists()
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert result["checkpoint"]["strict_load"]["passed"] is True
    assert result["checkpoint"]["synthetic_inference"]["state_mutated"] is False
    assert result["access"]["checkpoint_file_reads"] == 1
    assert result["access"]["checkpoint_deserializations"] == 1
    assert result["access"]["dataset_reads"] == result["access"]["g2_operations"] == 0
    assert result["authority"] == admission._authority(True)


def test_hash_mismatch_fails_before_deserialization_or_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, attempt = _fixture_tree(tmp_path, monkeypatch)
    checkpoint = attempt / "checkpoint_update_1000.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    monkeypatch.setattr(
        admission.torch, "load",
        lambda *args, **kwargs: pytest.fail("checkpoint was deserialized before hash validation"),
    )
    result = admission.admit_candidate(repository_root=tmp_path)
    output = tmp_path / "admission/attempt_v1"
    assert result["status"] == "FAILED_CLOSED"
    assert (output / "failure.json").is_file()
    assert not (output / "candidate_checkpoint.pt").exists()
    assert not (output / "candidate_receipt.json").exists()
    assert result["access"]["checkpoint_file_reads"] == 1
    assert result["access"]["checkpoint_deserializations"] == 0
    assert result["authority"] == admission._authority(False)


def test_failed_deserialization_is_counted_in_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    result = admission.admit_candidate(repository_root=tmp_path)
    assert result["status"] == "FAILED_CLOSED"
    assert result["access"]["checkpoint_file_reads"] == 1
    assert result["access"]["checkpoint_deserializations"] == 1
    assert result["access"]["synthetic_batches"] == 0
