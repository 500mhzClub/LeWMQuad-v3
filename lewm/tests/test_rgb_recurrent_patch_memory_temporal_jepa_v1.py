from __future__ import annotations

import pytest
import torch

from lewm.models.encoders import VisionEncoder
from lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1 import (
    RGBRecurrentPatchMemoryTemporalJepaV1,
    RGBRecurrentPatchMemoryTemporalJepaV1Config,
    temporal_v1_accepts_predecessor_key,
)
from lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (
    SingleFrameMultiblockMaskedSpatialJepaV1,
)
from scripts import (
    evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as evaluation,
)
from scripts import run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as training


@pytest.fixture(scope="module")
def predecessor_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(1701)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        encoder_state = {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
        predecessor = SingleFrameMultiblockMaskedSpatialJepaV1(encoder_state)
        state = {
            name: value.detach().clone()
            for name, value in predecessor.state_dict().items()
        }
        # These entries are deliberately stale.  Temporal V1 must reject them
        # from migration and establish a fresh hard-synchronized EMA target.
        for name, value in state.items():
            if name.startswith("target_encoder."):
                value.fill_(7.0)
        state["ema_update_count"].fill_(91)
        return state
    finally:
        torch.random.set_rng_state(caller_rng)


def _model(
    predecessor_state: dict[str, torch.Tensor],
) -> RGBRecurrentPatchMemoryTemporalJepaV1:
    return RGBRecurrentPatchMemoryTemporalJepaV1(predecessor_state)


def _target_indices(batch: int) -> torch.Tensor:
    return torch.arange(0, 256, 4, dtype=torch.long).unsqueeze(0).expand(
        batch, -1
    ).clone()


def _context_and_future(
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(9321)
    context = torch.randn(
        batch,
        3,
        3,
        112,
        112,
        generator=generator,
        dtype=torch.float32,
    )
    future = torch.randn(
        batch,
        3,
        112,
        112,
        generator=generator,
        dtype=torch.float32,
    )
    return context, future


def test_actual_model_matches_trainer_and_evaluator_contracts(
    predecessor_state: dict[str, torch.Tensor],
) -> None:
    model = _model(predecessor_state).eval()
    partition = training.partition_parameters_v1(model)
    assert partition.encoder and partition.predictor and partition.memory
    context, _future = _context_and_future(1)
    actions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target_indices = _target_indices(1)
    fields = evaluation._predict_future(
        model,
        context,
        actions,
        target_indices,
    )
    assert fields.prediction.shape == (1, 64, 192)
    assert fields.memory.shape == (1, 256, 192)
    assert fields.step_states.shape == (1, 3, 256, 192)


def test_temporal_v1_exact_warm_start_fresh_target_and_rng_boundary(
    predecessor_state: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(401)
    caller_rng = torch.random.get_rng_state().clone()
    model = _model(predecessor_state)

    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert model.action_embedding.weight.shape == (9, 192)
    assert model.time_embedding.weight.shape == (3, 192)
    assert model.temporal_gru.input_size == 192
    assert model.temporal_gru.hidden_size == 192
    assert model.temporal_gru.num_layers == 1
    assert model.temporal_gru.batch_first

    state = model.state_dict()
    migrated = {
        name for name in predecessor_state if temporal_v1_accepts_predecessor_key(name)
    }
    assert migrated
    for name in migrated:
        assert torch.equal(state[name], predecessor_state[name])

    assert int(model.ema_update_count) == 0
    assert not model.target_encoder.training
    assert not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    )
    for name, online in model.encoder.state_dict().items():
        target_name = f"target_encoder.{name}"
        assert torch.equal(online, state[target_name])
        assert not torch.equal(state[target_name], predecessor_state[target_name])

    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        parameter.requires_grad for parameter in model.predictor_blocks.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.temporal_gru.parameters()
    )

    with pytest.raises(ValueError, match="constants cannot change"):
        RGBRecurrentPatchMemoryTemporalJepaV1Config(context_length=4)


def test_temporal_v1_zero_reset_recurrence_and_320_token_future_decoder(
    predecessor_state: dict[str, torch.Tensor],
) -> None:
    model = _model(predecessor_state).eval()
    context, future = _context_and_future(1)
    actions = torch.tensor([[0, 4, 8]], dtype=torch.long)
    target_indices = _target_indices(1)
    captured: dict[str, torch.Tensor] = {}

    def capture_gru_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured["stream"] = inputs[0].detach().clone()
        captured["initial"] = inputs[1].detach().clone()

    handle = model.temporal_gru.register_forward_pre_hook(capture_gru_input)
    try:
        output = model(
            context,
            actions,
            future,
            target_indices,
            capture_intermediates=True,
        )
    finally:
        handle.remove()

    prediction = output.prediction
    assert captured["stream"].shape == (256, 3, 192)
    assert captured["initial"].shape == (1, 256, 192)
    assert torch.count_nonzero(captured["initial"]) == 0
    assert prediction.raw_predicted_target_tokens.shape == (1, 64, 192)
    assert prediction.normalized_predicted_target_tokens.shape == (1, 64, 192)
    assert prediction.encoded_history.shape == (1, 3, 256, 192)
    assert prediction.recurrent_step_states.shape == (1, 3, 256, 192)
    assert prediction.recurrent_memory.shape == (1, 256, 192)
    assert torch.equal(
        prediction.recurrent_memory,
        prediction.recurrent_step_states[:, -1],
    )
    assert torch.equal(
        prediction.time_indices,
        torch.tensor([[0, 1, 2]], dtype=torch.long),
    )
    assert prediction.predictor_input is not None
    assert prediction.predictor_input.shape == (1, 320, 192)
    assert len(prediction.predictor_block_outputs) == 2
    assert all(
        value.shape == (1, 320, 192)
        for value in prediction.predictor_block_outputs
    )

    expected_memory = (
        prediction.recurrent_memory + model.predictor_position.unsqueeze(0)
    )
    expected_queries = (
        model.predictor_mask_token.expand(1, 64, -1)
        + model.predictor_position[target_indices]
    )
    assert torch.equal(prediction.predictor_input[:, :256], expected_memory)
    assert torch.equal(prediction.predictor_input[:, 256:], expected_queries)
    assert output.target.raw_target_tokens.shape == (1, 64, 192)
    assert not output.target.raw_target_tokens.requires_grad
    assert output.target.raw_target_tokens.grad_fn is None
    assert torch.allclose(
        prediction.normalized_predicted_target_tokens.norm(dim=-1),
        torch.ones(1, 64),
        atol=1e-5,
    )
    registered_loss = 0.5 * (
        prediction.normalized_predicted_target_tokens
        - output.target.normalized_target_tokens
    ).square().sum(dim=-1).mean()
    assert torch.equal(output.loss, registered_loss)

    changed_actions = actions.clone()
    changed_actions[:, 1] = 5
    changed = model.predict_from_encoded_history(
        prediction.encoded_history,
        changed_actions,
        target_indices,
    )
    assert not torch.equal(changed.recurrent_memory, prediction.recurrent_memory)
    assert not torch.equal(
        changed.raw_predicted_target_tokens,
        prediction.raw_predicted_target_tokens,
    )

    current = model.predict_current_only(
        context[:, 2],
        actions[:, 2],
        target_indices,
    )
    assert current.recurrent_step_states.shape == (1, 1, 256, 192)
    assert torch.equal(current.time_indices, torch.tensor([[2]]))


def test_temporal_v1_sole_future_loss_trains_every_online_role_and_ema(
    predecessor_state: dict[str, torch.Tensor],
) -> None:
    model = _model(predecessor_state).train()
    context, future = _context_and_future(1)
    context.requires_grad_(True)
    future.requires_grad_(True)
    actions = torch.tensor([[1, 3, 7]], dtype=torch.long)
    target_indices = _target_indices(1)

    output = model(context, actions, future, target_indices)
    output.loss.backward()

    assert context.grad is not None
    assert torch.count_nonzero(context.grad) > 0
    assert future.grad is None
    online_roles = {
        "encoder": tuple(model.encoder.parameters()),
        "predictor": (
            model.predictor_position,
            model.predictor_mask_token,
            *tuple(model.predictor_blocks.parameters()),
            *tuple(model.predictor_norm.parameters()),
            *tuple(model.predictor_output.parameters()),
        ),
        "memory": (
            *tuple(model.action_embedding.parameters()),
            *tuple(model.time_embedding.parameters()),
            *tuple(model.temporal_gru.parameters()),
        ),
    }
    for parameters in online_roles.values():
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
        assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)
    assert not any(
        parameter.grad is not None for parameter in model.target_encoder.parameters()
    )
    assert not model.target_encoder.training

    online = model.encoder.patch_embed.weight
    target = model.target_encoder.patch_embed.weight
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(0.25)
    expected = target_before.clone()
    expected.mul_(0.996).add_(online.detach(), alpha=0.004)
    model.update_target_ema()
    assert int(model.ema_update_count) == 1
    assert torch.equal(target, expected)
    assert not model.target_encoder.training


def test_temporal_v1_rejects_state_and_input_contract_drift(
    predecessor_state: dict[str, torch.Tensor],
) -> None:
    missing = dict(predecessor_state)
    missing.pop("predictor_mask_token")
    with pytest.raises(ValueError, match="predecessor state inventory changed"):
        _model(missing)

    extra = dict(predecessor_state)
    extra["predictor_blocks.unregistered"] = torch.zeros(1)
    with pytest.raises(ValueError, match="predecessor state inventory changed"):
        _model(extra)

    model = _model(predecessor_state).eval()
    context, future = _context_and_future(1)
    target_indices = _target_indices(1)
    with pytest.raises(TypeError, match="actions must be long"):
        model(
            context,
            torch.zeros(1, 3, dtype=torch.float32),
            future,
            target_indices,
        )
    with pytest.raises(ValueError, match=r"closed range \[0,8\]"):
        model(
            context,
            torch.tensor([[0, 1, 9]], dtype=torch.long),
            future,
            target_indices,
        )
    with pytest.raises(ValueError, match="context_rgb must have shape"):
        model(
            context[:, :2],
            torch.zeros(1, 3, dtype=torch.long),
            future,
            target_indices,
        )
