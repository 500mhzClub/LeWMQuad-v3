from __future__ import annotations

from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.nn as nn

from lewm.models.go2_world_model_progression_v1 import (
    SpatialLatentDisplacementActionDecoderV1,
    normalized_spatial_energy_v1,
    predict_dynamic_spatial_tokens_v1,
)
from scripts import dev_train_go2_world_model_progression_v1 as runner


class _FakeArm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            spatial_token_count=4,
            feature_dim=6,
            action_count=3,
            time_embedding_count=3,
            temporal_hidden_dim=6,
            normalization_epsilon=1.0e-8,
        )
        self.predictor_position = nn.Parameter(torch.randn(4, 6))
        self.predictor_mask_token = nn.Parameter(torch.randn(1, 1, 6))
        self.predictor_blocks = nn.ModuleList((nn.Identity(),))
        self.predictor_norm = nn.LayerNorm(6)
        self.predictor_output = nn.Linear(6, 6)
        self.action_embedding = nn.Embedding(3, 6)
        self.time_embedding = nn.Embedding(3, 6)
        self.temporal_gru = nn.GRU(6, 6, batch_first=True)


def test_dynamic_predictor_supports_masked_and_full_grid_outputs() -> None:
    torch.manual_seed(4)
    arm = _FakeArm()
    history = torch.randn(2, 3, 4, 6)
    actions = torch.tensor(((0, 1, 2), (2, 1, 0)), dtype=torch.long)
    masked = torch.tensor(((0, 3), (1, 2)), dtype=torch.long)
    full = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(2, -1)

    masked_output = predict_dynamic_spatial_tokens_v1(
        arm, history, actions, masked
    )
    full_output = predict_dynamic_spatial_tokens_v1(
        arm, history, actions, full
    )

    assert masked_output.raw.shape == (2, 2, 6)
    assert full_output.raw.shape == (2, 4, 6)
    assert torch.allclose(full_output.normalized.norm(dim=-1), torch.ones(2, 4))


def test_candidate_blind_route_is_invariant_to_final_action() -> None:
    torch.manual_seed(5)
    arm = _FakeArm()
    history = torch.randn(2, 3, 4, 6)
    actions = torch.tensor(((0, 1, 0), (2, 1, 2)), dtype=torch.long)
    indices = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(2, -1)

    first = predict_dynamic_spatial_tokens_v1(
        arm, history, actions, indices, candidate_blind=True
    ).raw
    changed = actions.clone()
    changed[:, -1] = torch.tensor((2, 0))
    second = predict_dynamic_spatial_tokens_v1(
        arm, history, changed, indices, candidate_blind=True
    ).raw

    assert torch.equal(first, second)


def test_spatial_delta_decoder_backpropagates_to_predicted_future() -> None:
    torch.manual_seed(6)
    decoder = SpatialLatentDisplacementActionDecoderV1(
        feature_dim=6,
        hidden_dim=8,
        spatial_token_count=4,
        action_count=3,
    )
    current = torch.randn(3, 2, 6)
    future = torch.randn(3, 2, 6, requires_grad=True)
    indices = torch.tensor(((0, 1), (1, 3), (0, 2)), dtype=torch.long)
    labels = torch.tensor((0, 1, 2), dtype=torch.long)

    logits = decoder(current, future, indices)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (3, 3)
    assert future.grad is not None
    assert torch.isfinite(future.grad).all()
    assert float(future.grad.abs().sum()) > 0.0


def test_partial_delta_panel_requires_explicit_indices() -> None:
    decoder = SpatialLatentDisplacementActionDecoderV1(
        feature_dim=6,
        hidden_dim=8,
        spatial_token_count=4,
        action_count=3,
    )
    with pytest.raises(ValueError, match="explicit token_indices"):
        decoder(torch.zeros(2, 2, 6), torch.zeros(2, 2, 6))


def test_dynamic_energy_matches_manual_normalized_half_squared_distance() -> None:
    prediction = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]], dtype=torch.float32)
    target = torch.tensor([[[0.0, 2.0], [0.0, 5.0]]], dtype=torch.float32)
    energy = normalized_spatial_energy_v1(prediction, target)
    manual = 0.5 * torch.tensor((2.0, 0.0)).mean().reshape(1)
    assert torch.allclose(energy, manual)


def test_full_grid_output_is_structurally_reentrant() -> None:
    torch.manual_seed(7)
    arm = _FakeArm()
    history = torch.randn(2, 3, 4, 6)
    actions = torch.tensor(((0, 1, 2), (2, 1, 0)), dtype=torch.long)
    full = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(2, -1)

    first = predict_dynamic_spatial_tokens_v1(arm, history, actions, full).raw
    shifted_history = torch.cat((history[:, 1:], first.unsqueeze(1)), dim=1)
    shifted_actions = torch.cat(
        (actions[:, 1:], torch.tensor(((1,), (2,)), dtype=torch.long)), dim=1
    )
    second = predict_dynamic_spatial_tokens_v1(
        arm, shifted_history, shifted_actions, full
    ).raw

    assert second.shape == first.shape
    assert torch.isfinite(second).all()


def test_frozen_true_delta_decoder_passes_gradient_only_to_future() -> None:
    torch.manual_seed(8)
    decoder = SpatialLatentDisplacementActionDecoderV1(
        feature_dim=6,
        hidden_dim=8,
        spatial_token_count=4,
        action_count=3,
    )
    decoder.requires_grad_(False)
    current = torch.randn(3, 4, 6)
    future = torch.randn(3, 4, 6, requires_grad=True)
    labels = torch.tensor((0, 1, 2), dtype=torch.long)

    nn.functional.cross_entropy(decoder(current, future), labels).backward()

    assert future.grad is not None
    assert float(future.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_runner_panel_is_factorial_and_dropout_is_paired_within_grid_mode() -> None:
    assert runner.ARM_NAMES == (
        "masked_plain",
        "masked_delta",
        "full_plain",
        "full_delta",
    )
    assert runner.DELTA_ARM_NAMES == {"masked_delta", "full_delta"}
    assert runner.FULL_ARM_NAMES == {"full_plain", "full_delta"}
    assert runner._paired_dropout_seed(11, 4, 2) == runner._paired_dropout_seed(
        11, 4, 2
    )
    assert runner._paired_dropout_seed(11, 4, 2) != runner._paired_dropout_seed(
        10_011, 4, 2
    )


def test_next_batch_wraps_without_dropping_the_epoch_tail() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    order = torch.tensor((4, 3, 2, 1, 0), dtype=torch.long)
    batch, fresh, cursor = runner._next_batch(
        generator=generator,
        order=order,
        cursor=3,
        row_count=5,
        batch_size=4,
    )

    assert batch[:2].tolist() == [1, 0]
    assert batch[2:].tolist() == fresh[:2].tolist()
    assert cursor == 2


def test_progression_snapshot_is_current_loader_compatible_and_cpu_only(
    tmp_path,
) -> None:
    arm = _FakeArm()
    decoder = SpatialLatentDisplacementActionDecoderV1(
        feature_dim=6,
        hidden_dim=8,
        spatial_token_count=4,
        action_count=3,
    )
    path = tmp_path / "snapshot.pt"

    runner._snapshot(
        path=path,
        name="masked_delta",
        seed=1,
        update=2,
        arm=arm,
        decoder=decoder,
        metrics={"probe": 1.0},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema"] == runner.SNAPSHOT_SCHEMA
    assert payload["status"] == "COMPLETE"
    assert payload["citable_as_scientific_evidence"] is False
    assert payload["authorizes_retry_or_resume"] is False
    assert all(value.device.type == "cpu" for value in payload["arm_state_dict"].values())
    assert all(
        value.device.type == "cpu" for value in payload["decoder_state_dict"].values()
    )


def test_scene_clustered_balanced_accuracy_interval_is_deterministic() -> None:
    labels = np.tile(np.arange(9, dtype=np.int64), 5)
    predictions = labels.copy()
    scenes = [f"scene-{row // 9}" for row in range(len(labels))]
    first = runner._scene_clustered_balanced_accuracy_interval(
        labels, predictions, scenes, resamples=200, seed=3
    )
    second = runner._scene_clustered_balanced_accuracy_interval(
        labels, predictions, scenes, resamples=200, seed=3
    )
    assert first == second
    assert first["point"] == first["lower_95"] == first["upper_95"] == 1.0
