from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts import run_go2_matched_branch_successor_screen_v1 as runner


def _panel(batch: int = 2, width: int = 8) -> torch.Tensor:
    torch.manual_seed(31)
    return F.normalize(torch.randn(batch, 9, 256, width), dim=-1)


def _synthetic_index() -> runner.ScreenIndexV1:
    contexts = []
    targets = []
    histories = []
    for state in range(128):
        offset = state * 12
        contexts.append((offset, offset + 1, offset + 2))
        targets.append(tuple(range(offset + 3, offset + 12)))
        histories.append((state % 9, (state + 1) % 9))
    return runner.ScreenIndexV1(
        state_ids=tuple(f"state-{state}" for state in range(128)),
        family_ids=tuple(f"family-{state % 8}" for state in range(128)),
        scene_ids=tuple(f"scene-{state // 8}" for state in range(128)),
        artifact_ids=tuple(f"artifact-{index}" for index in range(1_536)),
        context_indices=torch.tensor(contexts, dtype=torch.long),
        target_indices=torch.tensor(targets, dtype=torch.long),
        history_actions=torch.tensor(histories, dtype=torch.long),
        index_sha256="0" * 64,
    )


def test_distance_matrix_and_metrics_retrieve_exact_successors() -> None:
    targets = _panel()
    predictions = targets.clone()
    last_context = F.normalize(torch.randn(2, 256, 8), dim=-1)

    matrix = runner.cosine_distance_matrix_v1(predictions, targets)
    metrics = runner.screen_metrics_from_panels_v1(
        predictions, targets, last_context
    )

    assert matrix.shape == (2, 9, 9)
    assert torch.allclose(
        matrix.diagonal(dim1=1, dim2=2), torch.zeros(2, 9), atol=2.0e-6
    )
    assert metrics["branch_retrieval_accuracy"] == 1.0
    assert metrics["matched_cosine_error"] == pytest.approx(0.0, abs=2.0e-6)
    assert metrics["action_intervention_margin"] > 0.5


def test_common_objective_prefers_correct_over_cyclic_assignment() -> None:
    targets = _panel()
    correct, correct_terms = runner.common_objective_v1(targets, targets)
    shifted = torch.roll(targets, shifts=1, dims=1)
    wrong, wrong_terms = runner.common_objective_v1(shifted, targets)

    assert correct < wrong
    assert correct_terms["matched"] < wrong_terms["matched"]
    assert correct_terms["contrastive"] < wrong_terms["contrastive"]


def test_screen_config_keeps_the_preregistered_four_arm_panel() -> None:
    config = runner.screen_config_v1()
    assert config["arms"] == list(runner.ARM_NAMES)
    assert config["seed"] == 2_026_080_301
    assert config["updates"] == 800
    assert config["maximum_error_to_persistence_ratio"] == 0.8
    assert config["retrieval_threshold"] == 0.5
    assert config["rssm_kl_reduction"] == "batchmean_after_latent_sum"


def test_feature_preprocessing_contracts_are_explicit_and_distinct() -> None:
    dino = runner.feature_preprocessing_contract_v1("dinov2")
    vjepa = runner.feature_preprocessing_contract_v1("vjepa2_1")

    assert dino["decoded_input"] == {
        "format": "PNG",
        "mode": "RGB",
        "size": [224, 224],
    }
    assert dino["encoder_output_grid"] == [16, 16]
    assert dino["spatial_conversion"] == "identity"
    assert vjepa["image_geometry"]["resize"] == [438, 438]
    assert vjepa["image_geometry"]["center_crop"] == [384, 384]
    assert vjepa["encoder_output_grid"] == [24, 24]
    assert vjepa["spatial_conversion"] == "torch_area_24x24_to_16x16"
    assert dino["token_conversion"] == vjepa["token_conversion"]


def test_mocked_feature_extraction_binds_preprocessing_and_artifact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeDino(torch.nn.Module):
        def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = inputs.shape[0]
            values = torch.arange(1, 385, dtype=torch.float32).view(1, 1, 384)
            return {"x_norm_patchtokens": values.expand(batch, 256, -1)}

    monkeypatch.setattr(runner, "ARTIFACT_COUNT", 2)
    monkeypatch.setattr(runner, "_load_dino_encoder", lambda *_args: _FakeDino())
    monkeypatch.setattr(
        runner.screen_data,
        "preprocess_dinov2_png_bytes_v1",
        lambda _raw: torch.ones(3, 224, 224),
    )
    opened: list[str] = []

    def fake_read(_bundle: object, artifact_id: str) -> bytes:
        opened.append(artifact_id)
        return b"bound-rgb"

    monkeypatch.setattr(runner, "read_bound_rgb_bytes_v1", fake_read)
    index = runner.ScreenIndexV1(
        state_ids=(),
        family_ids=(),
        scene_ids=(),
        artifact_ids=("train-a", "train-b"),
        context_indices=torch.empty((0, 3), dtype=torch.long),
        target_indices=torch.empty((0, 9), dtype=torch.long),
        history_actions=torch.empty((0, 2), dtype=torch.long),
        index_sha256="1" * 64,
    )
    authority = {
        "config": {"feature_batches": {"dinov2": 2}},
        "encoder_sources": {"dinov2": {"checkpoint_binding": "bound"}},
    }
    bundle = SimpleNamespace(manifest_binding={"sha256": "2" * 64})
    cache_path = tmp_path / "dinov2.pt"
    receipt = runner.extract_feature_cache_v1(
        bundle,
        index,
        encoder_name="dinov2",
        authority=authority,
        device=torch.device("cpu"),
        output_path=cache_path,
    )

    assert opened == ["train-a", "train-b"]
    assert receipt["train_artifact_open_count"] == 2
    assert receipt["eval_artifact_open_count"] == 0
    assert receipt["preprocessing"] == runner.feature_preprocessing_contract_v1(
        "dinov2"
    )
    assert len(receipt["artifact_order_sha256"]) == 64
    loaded = runner._load_feature_cache(  # noqa: SLF001
        receipt, expected_encoder="dinov2", index=index
    )
    assert loaded.shape == (2, 256, 384)

    changed = {**receipt, "preprocessing": {}}
    with pytest.raises(runner.ScreenError, match="receipt changed"):
        runner._load_feature_cache(  # noqa: SLF001
            changed, expected_encoder="dinov2", index=index
        )


def test_authority_rejects_a_changed_caller_binding(tmp_path: Path) -> None:
    authority = tmp_path / "authority.json"
    authority.write_text("{}\n")
    with pytest.raises(runner.ScreenError, match="caller binding"):
        runner._read_authority(  # noqa: SLF001
            authority,
            expected_sha256="0" * 64,
            expected_byte_count=3,
        )


def test_one_update_synthetic_train_core_writes_bound_checkpoint(
    tmp_path: Path,
) -> None:
    torch.manual_seed(32)
    features = F.normalize(torch.randn(1_536, 256, 4), dim=-1).to(torch.float16)
    result = runner.train_arm_v1(
        "dense_dinov2",
        features,
        _synthetic_index(),
        config=runner.screen_config_v1(),
        device=torch.device("cpu"),
        output_path=tmp_path / "checkpoint.pt",
        updates=1,
        trace_updates=(0, 1),
    )

    assert result["arm"] == "dense_dinov2"
    assert result["updates"] == 1
    assert len(result["traces"]) == 2
    assert result["nonfinite_count"] == 0
    assert result["deterministic_repeat_passed"] is True
    assert runner.file_binding_v1(tmp_path / "checkpoint.pt") == result[
        "checkpoint_binding"
    ]


def test_one_update_rssm_uses_the_stochastic_posterior_branch(tmp_path: Path) -> None:
    torch.manual_seed(33)
    features = F.normalize(torch.randn(1_536, 256, 4), dim=-1).to(torch.float16)
    result = runner.train_arm_v1(
        "rssm_vjepa2_1",
        features,
        _synthetic_index(),
        config=runner.screen_config_v1(),
        device=torch.device("cpu"),
        output_path=tmp_path / "rssm.pt",
        updates=1,
        trace_updates=(0, 1),
    )

    objective = result["traces"][-1]["objective"]
    assert objective["posterior_reconstruction"] >= 0.0
    assert objective["kl"] >= 0.0
    assert result["deterministic_repeat_passed"] is True


def test_preexisting_output_is_not_contaminated_by_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve\n")
    monkeypatch.setattr(
        runner,
        "_read_authority",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        raise runner.ScreenError("expected failure")

    monkeypatch.setattr(runner, "execute_v1", fail)
    with pytest.raises(runner.ScreenError, match="expected failure"):
        runner.main(
            [
                "--authority",
                str(tmp_path / "unused.json"),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )
    assert sentinel.read_text() == "preserve\n"
    assert not (output / "terminal.json").exists()


def test_new_output_gets_a_terminal_on_midrun_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "new-output"
    monkeypatch.setattr(
        runner,
        "_read_authority",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        output.mkdir()
        raise runner.ScreenError("expected midrun failure")

    monkeypatch.setattr(runner, "execute_v1", fail)
    with pytest.raises(runner.ScreenError, match="expected midrun failure"):
        runner.main(
            [
                "--authority",
                str(tmp_path / "unused.json"),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )
    terminal = runner.json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE"
    assert terminal["citable_as_scientific_evidence"] is False
