from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
)
from lewm.models.memory_role_spatial_contrastive_joint_jepa_v3 import (
    MemoryRoleSpatialContrastiveJointJepaV3,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1 import (
    INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1,
    MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1,
    PROJECTION_INITIALIZATION_SEED_V13,
    SpatialTokenDelayLineCausalConvolutionPredictorV1,
    V18SpatialTokenDelayLineCausalConvolutionJointJepaV1,
    V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config,
)


def _memory_harness() -> V18SpatialTokenDelayLineCausalConvolutionJointJepaV1:
    model = object.__new__(
        V18SpatialTokenDelayLineCausalConvolutionJointJepaV1
    )
    nn.Module.__init__(model)
    model.config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
    model.memory_config = (
        V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()
    )
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(
            INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1
        )
        model.memory_predictor = (
            SpatialTokenDelayLineCausalConvolutionPredictorV1(
                model.memory_config
            )
        )
    finally:
        torch.random.set_rng_state(caller_rng)
    return model


def _normalized_tokens(batch: int, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(
        torch.randn((batch, length, 64, 16, 16), generator=generator),
        dim=2,
    )


def _one_hot(indices: list[list[int]] | list[int]) -> torch.Tensor:
    return F.one_hot(torch.tensor(indices), num_classes=9).to(torch.float32)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


def _fitted_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(31_001)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


def test_frozen_config_and_v13_projection_compatibility_export() -> None:
    config = V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()

    assert PROJECTION_INITIALIZATION_SEED_V13 == 20_260_729
    assert config.source_shape == (64, 64, 64)
    assert config.token_shape == (64, 16, 16)
    assert config.history_slots == 4
    assert config.rollout_horizon == 4
    assert config.masked_current_loss_weight == 0.5
    with pytest.raises(ValueError, match="cannot change"):
        V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config(
            rollout_horizon=3
        )


def test_predictor_is_exact_newest_tap_identity_with_zero_film() -> None:
    model = _memory_harness()
    predictor = model.memory_predictor
    depthwise = predictor.depthwise_causal
    pointwise = predictor.pointwise

    assert depthwise.kernel_size == (4, 3, 3)
    assert depthwise.groups == 65
    assert depthwise.in_channels == depthwise.out_channels == 65
    expected_depthwise = torch.zeros_like(depthwise.weight)
    expected_depthwise[:, 0, 0, 1, 1] = 1.0
    assert torch.equal(depthwise.weight, expected_depthwise)
    expected_pointwise = torch.zeros_like(pointwise.weight)
    indices = torch.arange(64)
    expected_pointwise[indices, indices, 0, 0] = 1.0
    assert torch.equal(pointwise.weight, expected_pointwise)
    assert torch.count_nonzero(pointwise.bias) == 0
    assert torch.count_nonzero(predictor.age_embeddings) == 0
    assert torch.count_nonzero(predictor.action_film[2].weight) == 0
    assert torch.count_nonzero(predictor.action_film[2].bias) == 0

    tokens = _normalized_tokens(2, 3, 31_002)
    history_actions = _one_hot([[0, 1], [2, 3]])
    future_actions = _one_hot([[4, 5, 6, 7], [8, 7, 6, 5]])
    state = model.build_history_state(tokens, history_actions)
    predictions, _ = model._rollout_from_state(state, future_actions)

    expected = tokens[:, -1, None].expand_as(predictions)
    assert torch.allclose(predictions, expected, rtol=0.0, atol=2.0e-7)
    assert torch.allclose(
        torch.linalg.vector_norm(predictions, dim=2),
        torch.ones((2, 4, 16, 16)),
        rtol=0.0,
        atol=2.0e-6,
    )


def test_history_build_predict_and_reset_preserve_exact_fifo_order() -> None:
    model = _memory_harness()
    tokens = _normalized_tokens(2, 3, 31_003)
    history_actions = _one_hot([[0, 1], [2, 3]])
    original_tokens = tokens.clone()
    state = model.build_history_state(tokens, history_actions)

    assert torch.equal(state.tokens[:, 0], tokens[:, 2])
    assert torch.equal(state.tokens[:, 1], tokens[:, 1])
    assert torch.equal(state.tokens[:, 2], tokens[:, 0])
    assert torch.count_nonzero(state.tokens[:, 3]) == 0
    assert torch.equal(
        state.valid,
        torch.tensor(
            [[True, True, True, False], [True, True, True, False]]
        ),
    )
    assert torch.equal(state.actions[:, 0], history_actions[:, 1])
    assert torch.equal(state.actions[:, 1], history_actions[:, 0])
    assert torch.count_nonzero(state.actions[:, 2:]) == 0

    captured_action_tapes: list[torch.Tensor] = []

    def capture_action_tape(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured_action_tapes.append(inputs[2].detach().clone())

    handle = model.memory_predictor.register_forward_pre_hook(capture_action_tape)
    try:
        first = model.predict_from_state(state, _one_hot([4, 5]))
        second = model.predict_from_state(first.state, _one_hot([6, 7]))
    finally:
        handle.remove()

    assert torch.equal(captured_action_tapes[0][:, 0], _one_hot([4, 5]))
    assert torch.equal(captured_action_tapes[0][:, 1], history_actions[:, 1])
    assert torch.equal(captured_action_tapes[0][:, 2], history_actions[:, 0])
    assert torch.count_nonzero(captured_action_tapes[0][:, 3]) == 0
    assert torch.equal(captured_action_tapes[1][:, 0], _one_hot([6, 7]))
    assert torch.equal(captured_action_tapes[1][:, 1], _one_hot([4, 5]))
    assert torch.equal(captured_action_tapes[1][:, 2], history_actions[:, 1])
    assert torch.equal(captured_action_tapes[1][:, 3], history_actions[:, 0])
    assert torch.equal(tokens, original_tokens)
    rebuilt = model.build_history_state(tokens, history_actions)
    assert torch.equal(state.tokens, rebuilt.tokens)

    reset = model.reset_history_state(
        second.state,
        torch.tensor([True, False]),
    )
    assert reset.valid[0].tolist() == [True, False, False, False]
    assert torch.equal(reset.tokens[0, 0], second.state.tokens[0, 0])
    assert torch.count_nonzero(reset.tokens[0, 1:]) == 0
    assert torch.count_nonzero(reset.actions[0]) == 0
    assert torch.equal(reset.tokens[1], second.state.tokens[1])
    assert torch.equal(reset.valid[1], second.state.valid[1])
    assert torch.equal(reset.actions[1], second.state.actions[1])


def test_newest_mask_drops_exactly_half_in_contiguous_four_by_four_blocks() -> None:
    model = _memory_harness()
    mask = model.deterministic_newest_keep_mask(
        3,
        device=torch.device("cpu"),
    )

    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (3, 1, 16, 16)
    assert torch.equal(mask[0], mask[1])
    assert int(mask[0].sum()) == 128
    block_values = []
    for row in range(0, 16, 4):
        for column in range(0, 16, 4):
            block = mask[0, 0, row : row + 4, column : column + 4]
            assert bool((block == block[0, 0]).all())
            block_values.append(bool(block[0, 0]))
    assert sum(block_values) == 8


class _ForwardHarness(
    V18SpatialTokenDelayLineCausalConvolutionJointJepaV1
):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        self.memory_config = (
            V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()
        )
        self.memory_predictor = (
            SpatialTokenDelayLineCausalConvolutionPredictorV1(
                self.memory_config
            )
        )
        self.online_source = nn.Parameter(_normalized_tokens(1, 3, 31_004))
        self.register_buffer(
            "target_source",
            _normalized_tokens(1, 4, 31_005),
        )

    def encode_online_memory_sequence(
        self,
        rgb_sequence: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(
            self.online_source.expand(rgb_sequence.shape[0], -1, -1, -1, -1),
            dim=2,
        )

    @torch.no_grad()
    def encode_target_memory_sequence(
        self,
        rgb_sequence: torch.Tensor,
    ) -> torch.Tensor:
        return self.target_source.expand(
            rgb_sequence.shape[0],
            -1,
            -1,
            -1,
            -1,
        ).detach()


def test_forward_memory_has_recursive_full_masked_and_stopgrad_target_branches() -> None:
    model = _ForwardHarness()
    history_rgb = torch.zeros((1, 3, 3, 112, 112), dtype=torch.float32)
    future_rgb = torch.ones((1, 4, 3, 112, 112), dtype=torch.float32)
    actions = _one_hot([[0, 1, 2, 3, 4, 5]])

    output = model.forward_memory(history_rgb, actions, future_rgb)

    assert tuple(output.online_history_tokens.shape) == (1, 3, 64, 16, 16)
    assert tuple(output.target_future_tokens.shape) == (1, 4, 64, 16, 16)
    assert tuple(output.full_predictions.shape) == (1, 4, 64, 16, 16)
    assert tuple(output.masked_current_predictions.shape) == (
        1,
        4,
        64,
        16,
        16,
    )
    expected_persistence = output.online_history_tokens[:, -1, None].expand_as(
        output.full_predictions
    )
    assert torch.allclose(
        output.full_predictions,
        expected_persistence,
        rtol=0.0,
        atol=2.0e-7,
    )
    assert output.target_future_tokens.requires_grad is False
    assert output.newest_keep_mask.dtype == torch.bool
    assert int(output.newest_keep_mask.sum()) == 128
    assert torch.allclose(
        output.loss,
        output.full_loss + 0.5 * output.masked_current_loss,
        rtol=0.0,
        atol=0.0,
    )

    output.loss.backward()
    assert model.online_source.grad is not None
    memory_gradients = [
        parameter.grad
        for parameter in model.memory_predictor.parameters()
        if parameter.grad is not None
    ]
    assert memory_gradients
    assert any(int(torch.count_nonzero(gradient)) > 0 for gradient in memory_gradients)
    assert model.target_source.grad is None


class _SequenceEncodingHarness(
    V18SpatialTokenDelayLineCausalConvolutionJointJepaV1
):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        self.memory_config = (
            V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()
        )
        self.memory_predictor = (
            SpatialTokenDelayLineCausalConvolutionPredictorV1(
                self.memory_config
            )
        )
        latent = torch.arange(
            64 * 64 * 64,
            dtype=torch.float32,
        ).reshape(1, 64, 64, 64)
        self.online_latent = nn.Parameter(latent / float(latent.numel()))
        self.register_buffer("target_latent", -self.online_latent.detach().clone())

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.online_latent.expand(rgb.shape[0], -1, -1, -1)

    @torch.no_grad()
    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.target_latent.expand(rgb.shape[0], -1, -1, -1)


def test_sequence_encoders_apply_fixed_pooling_normalization_and_target_stopgrad() -> None:
    model = _SequenceEncodingHarness()
    rgb = torch.zeros((2, 3, 3, 112, 112), dtype=torch.float32)

    online = model.encode_online_memory_sequence(rgb)
    target = model.encode_target_memory_sequence(rgb[:, :2])
    expected_online = F.normalize(
        F.avg_pool2d(model.online_latent, kernel_size=4, stride=4),
        dim=1,
        eps=1.0e-6,
    )
    expected_target = F.normalize(
        F.avg_pool2d(model.target_latent, kernel_size=4, stride=4),
        dim=1,
        eps=1.0e-6,
    )

    assert tuple(online.shape) == (2, 3, 64, 16, 16)
    assert torch.equal(online[0, 0], expected_online[0])
    assert torch.equal(online[1, 2], expected_online[0])
    assert online.requires_grad
    assert tuple(target.shape) == (2, 2, 64, 16, 16)
    assert torch.equal(target[0, 0], expected_target[0])
    assert target.requires_grad is False


def test_real_model_freezes_diagnostic_roles_and_covers_trainable_inventory() -> None:
    fitted = _fitted_model()
    caller_rng = torch.random.get_rng_state().clone()
    model = V18SpatialTokenDelayLineCausalConvolutionJointJepaV1(
        fitted,
        _sweep_masks(),
    ).train()

    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert isinstance(model, MemoryRoleSpatialContrastiveJointJepaV3)
    for name in ("role_factorizer", "place_predictor", "local_predictor"):
        module = getattr(model, name)
        assert not module.training
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert not model.target_role_factorizer.training
    assert not any(
        parameter.requires_grad
        for module in model.target_modules()
        for parameter in module.parameters()
    )
    assert model.memory_predictor.training
    assert all(
        parameter.requires_grad for parameter in model.memory_predictor.parameters()
    )
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(
            INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1
        )
        expected_predictor = (
            SpatialTokenDelayLineCausalConvolutionPredictorV1(
                model.memory_config
            )
        )
    finally:
        torch.random.set_rng_state(caller_rng)
    actual_state = model.memory_predictor.state_dict()
    expected_state = expected_predictor.state_dict()
    assert actual_state.keys() == expected_state.keys()
    assert all(
        torch.equal(value, expected_state[name])
        for name, value in actual_state.items()
    )

    groups = model.trainable_parameter_groups_delay_line_v1()
    assert groups.memory_predictor
    assert all(
        name.startswith(MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1)
        for name, _ in groups.memory_predictor
    )
    assert {id(parameter) for _, parameter in groups.online} == {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert len({id(parameter) for _, parameter in groups.online}) == len(
        groups.online
    )
