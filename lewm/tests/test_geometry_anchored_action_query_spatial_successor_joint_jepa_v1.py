from __future__ import annotations

import copy
import inspect
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder
from lewm.models import (
    geometry_anchored_action_query_spatial_successor_joint_jepa_v1 as api,
)


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(9917)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def model(
    n320_encoder_state: dict[str, torch.Tensor],
) -> api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1:
    return api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1(
        n320_encoder_state
    )


def _predictor() -> api.ActionQuerySpatialSuccessorPredictorV1:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(20260712)
        return api.ActionQuerySpatialSuccessorPredictorV1(
            api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1Config()
        )
    finally:
        torch.random.set_rng_state(caller_rng)


def _normalized_latent(seed: int, batch: int = 1) -> torch.Tensor:
    value = torch.randn(
        batch, 64, 64, 64, generator=torch.Generator().manual_seed(seed)
    )
    return api.normalize_latent_per_cell_v1(value)


def test_exact_position_formula_row_major_indexing_and_predictor_inventory() -> None:
    predictor = _predictor()
    position = predictor.position_encoding
    assert position.shape == (256, 128)
    assert position.dtype == torch.float32
    assert not position.requires_grad
    for u, v, i in ((0, 0, 0), (3, 5, 7), (15, 15, 31)):
        q = 16 * u + v
        frequency = 10000.0 ** (-i / 32.0)
        assert position[q, 2 * i].item() == pytest.approx(
            math.sin(u * frequency), abs=2e-6
        )
        assert position[q, 2 * i + 1].item() == pytest.approx(
            math.cos(u * frequency), abs=2e-6
        )
        assert position[q, 64 + 2 * i].item() == pytest.approx(
            math.sin(v * frequency), abs=2e-6
        )
        assert position[q, 64 + 2 * i + 1].item() == pytest.approx(
            math.cos(v * frequency), abs=2e-6
        )
    assert tuple(name for name, _ in predictor.named_parameters()) == (
        api.PREDICTOR_ORDERED_PARAMETER_NAMES_V1
    )
    assert len(tuple(predictor.parameters())) == 34
    assert sum(value.numel() for value in predictor.parameters()) == 504_384
    assert predictor.current_downsampler.kernel_size == (4, 4)
    assert predictor.current_downsampler.stride == (4, 4)


def test_blocks_are_separate_exact_cross_attention_mlp_and_memory_is_immutable() -> None:
    predictor = _predictor().eval()
    assert len(predictor.blocks) == 2
    first, second = predictor.blocks
    assert first is not second
    assert len({id(first.query_norm), id(first.memory_norm), id(first.ffn_norm)}) == 3
    assert len({id(second.query_norm), id(second.memory_norm), id(second.ffn_norm)}) == 3
    assert not set(map(id, first.parameters())) & set(map(id, second.parameters()))
    for block in predictor.blocks:
        assert block.attention.embed_dim == 128
        assert block.attention.num_heads == 4
        assert block.attention.dropout == 0.0
        assert block.linear1.in_features == 128
        assert block.linear1.out_features == 256
        assert block.linear2.in_features == 256
        assert block.linear2.out_features == 128

    memory_pointers: list[int] = []
    key_value_identity: list[bool] = []
    handles = []
    for block in predictor.blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, args: memory_pointers.append(args[1].data_ptr())
            )
        )
        handles.append(
            block.attention.register_forward_pre_hook(
                lambda _module, args: key_value_identity.append(args[1] is args[2])
            )
        )
    predictor(_normalized_latent(9))
    for handle in handles:
        handle.remove()
    assert len(memory_pointers) == 2 and memory_pointers[0] == memory_pointers[1]
    assert key_value_identity == [True, True]


def test_vectorized_shapes_one_action_slice_residual_arithmetic_and_permutation() -> None:
    predictor = _predictor().eval()
    latent = _normalized_latent(13, batch=2)
    all_successors = predictor(latent)
    residuals = predictor.predict_residuals_all_actions(latent)
    assert all_successors.shape == (2, 9, 64, 64, 64)
    assert residuals.shape == all_successors.shape
    torch.testing.assert_close(
        all_successors, latent[:, None] + residuals, rtol=0.0, atol=0.0
    )
    actions = torch.tensor((2, 7), dtype=torch.long)
    selected_slice = api.select_action_successor_v1(all_successors, actions)
    assert torch.equal(selected_slice, all_successors[torch.arange(2), actions])

    permutation = torch.tensor((4, 0, 8, 2, 6, 1, 7, 3, 5))
    permuted = copy.deepcopy(predictor)
    with torch.no_grad():
        permuted.action_embedding.weight.copy_(
            predictor.action_embedding.weight[permutation]
        )
    torch.testing.assert_close(
        permuted(latent), all_successors[:, permutation], rtol=0.0, atol=0.0
    )

    identity_predictor = copy.deepcopy(predictor)
    with torch.no_grad():
        identity_predictor.output_head.weight.zero_()
        identity_predictor.output_head.bias.zero_()
    assert torch.equal(
        identity_predictor(latent), latent[:, None].expand(-1, 9, -1, -1, -1)
    )


def test_selected_action_path_matches_vectorized_all_action_slice() -> None:
    predictor = _predictor().eval()
    latent = _normalized_latent(37, batch=2)
    actions = torch.tensor((1, 8), dtype=torch.long)
    with torch.no_grad():
        all_successors = predictor(latent)
        selected = latent + predictor.predict_residuals_selected_actions(
            latent, actions
        )
    torch.testing.assert_close(
        selected,
        all_successors[torch.arange(2), actions],
        rtol=1e-5,
        atol=1e-6,
    )


def test_nonzero_distinct_initialization_and_pairwise_distinct_head_inputs() -> None:
    predictor = _predictor().eval()
    action = predictor.action_embedding.weight.detach()
    assert torch.isfinite(action).all() and float(action.norm()) > 0.0
    assert torch.isfinite(predictor.future_queries).all()
    assert float(predictor.future_queries.detach().norm()) > 0.0
    assert torch.isfinite(predictor.output_head.weight).all()
    assert float(predictor.output_head.weight.detach().norm()) > 0.0
    assert torch.equal(
        predictor.output_head.bias, torch.zeros_like(predictor.output_head.bias)
    )
    for first in range(9):
        for second in range(first + 1, 9):
            assert not torch.equal(action[first], action[second])

    head_inputs = predictor.head_inputs_all_actions(_normalized_latent(20260727))
    assert head_inputs.shape == (1, 9, 128, 16, 16)
    for first in range(9):
        for second in range(first + 1, 9):
            assert not torch.equal(head_inputs[:, first], head_inputs[:, second])


def test_constructor_restores_rng_is_deterministic_and_target_isolated(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.random.default_generator.manual_seed(731)
    caller_rng = torch.random.get_rng_state().clone()
    first = api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    second = api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    for first_module, second_module in (
        (first.bev_lift, second.bev_lift),
        (first.semantic_head, second.semantic_head),
        (first.predictor, second.predictor),
    ):
        assert all(
            torch.equal(value, second_module.state_dict()[name])
            for name, value in first_module.state_dict().items()
        )
    assert tuple(first.target_modules()) == (
        first.target_encoder,
        first.target_bev_lift,
    )
    assert all(
        not parameter.requires_grad
        for module in first.target_modules()
        for parameter in module.parameters()
    )
    assert not hasattr(first, "target_predictor")
    assert not hasattr(first, "target_semantic_head")
    optimizer_ids = {
        id(parameter)
        for parameter in first.parameters()
        if parameter.requires_grad
    }
    assert not optimizer_ids & {
        id(parameter)
        for module in first.target_modules()
        for parameter in module.parameters()
    }


def test_local_energies_scales_and_targets_are_exactly_detached() -> None:
    generator = torch.Generator().manual_seed(41)
    current = torch.randn(1, 64, 64, 64, generator=generator, requires_grad=True)
    predictions = torch.randn(
        1, 9, 64, 64, 64, generator=generator, requires_grad=True
    )
    target = torch.randn(
        1, 64, 64, 64, generator=generator, requires_grad=True
    )
    negative = torch.randn(
        1, 64, 64, 64, generator=generator, requires_grad=True
    )
    actions = torch.tensor((3,), dtype=torch.long)
    terms = api.action_query_joint_objective_v1(
        predictions, target, negative, current, actions
    )
    manual_cells = F.smooth_l1_loss(
        predictions,
        target.detach()[:, None].expand_as(predictions),
        beta=1.0,
        reduction="none",
    ).mean(2)
    manual = F.avg_pool2d(manual_cells, 4, stride=4).flatten(2)
    torch.testing.assert_close(terms.positive, manual, rtol=0.0, atol=0.0)
    assert not terms.action_scale.requires_grad
    assert not terms.target_scale.requires_grad
    assert terms.local_action_ce.shape == (1, 256)
    assert terms.local_target_ce.shape == (1, 256)
    terms.dynamics.backward()
    assert predictions.grad is not None and float(predictions.grad.abs().sum()) > 0.0
    assert target.grad is None
    assert negative.grad is None
    # Persistence is reporting-only and is not part of the displayed dynamics sum.
    assert current.grad is None


def test_local_first_action_ce_counterexample_and_corrected_ssm_chance_sign() -> None:
    tied = torch.full((2, 256), 1.0)
    torch.testing.assert_close(
        api.smooth_spatial_soft_min_v1(tied),
        torch.ones(2),
        rtol=0.0,
        atol=2e-7,
    )
    nonnegative = torch.full((1, 256), 5.0)
    nonnegative[:, :32] = 0.0
    localized = api.smooth_spatial_soft_min_v1(nonnegative)
    assert 0.0 < localized.item() < 5.0

    positive = torch.full((1, 9, 256), 100.0)
    positive[:, 0, :32] = 0.0
    positive[:, 1, 32:] = 0.0
    assert positive[:, 0].mean() > positive[:, 1].mean()
    local_ce = api.local_action_cross_entropy_v1(
        positive, torch.tensor((0,), dtype=torch.long)
    )
    local_first_score = api.smooth_spatial_soft_min_v1(
        local_ce / math.log(9.0)
    )
    assert local_first_score.item() < 1.0
    tied_positive = torch.ones(1, 9, 256)
    tied_ce = api.local_action_cross_entropy_v1(
        tied_positive, torch.tensor((0,), dtype=torch.long)
    )
    torch.testing.assert_close(
        api.smooth_spatial_soft_min_v1(tied_ce / math.log(9.0)),
        torch.ones(1),
        rtol=0.0,
        atol=2e-7,
    )


def test_reporting_helpers_use_energies_and_exact_ce_conventions() -> None:
    positive = torch.ones(2, 9, 256)
    negative = torch.ones(2, 9, 256) * 2.0
    actions = torch.tensor((0, 8), dtype=torch.long)
    scores = api.reporting_energy_helpers_v1(positive, negative, actions)
    assert scores.action.shape == (2, 9)
    assert scores.correct.shape == (2,)
    assert scores.deranged.shape == (2,)
    assert torch.equal(
        api.reporting_action_cross_entropy_v1(scores.action, actions),
        torch.full((2,), math.log(9.0)),
    )
    assert bool(
        (api.reporting_target_cross_entropy_v1(scores.correct, scores.deranged)
        < math.log(2.0)).all()
    )
    assert tuple(inspect.signature(_predictor().forward).parameters) == (
        "normalized_current_latent",
    )


def test_synthetic_joint_gradients_reach_every_route_and_not_target(
    model: api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1,
) -> None:
    model.eval()
    model.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(1201)
    current_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    next_rgb = torch.rand(1, 3, 112, 112, generator=generator)
    negative_rgb = torch.rand(1, 3, 112, 112, generator=generator)

    with torch.no_grad():
        online_same = model.encode_online(current_rgb)
        target_same = model.encode_target(current_rgb)
    assert torch.equal(online_same, target_same)
    assert not target_same.requires_grad

    model.train()
    online_current = model.encode_online(current_rgb)
    online_next = model.encode_online(next_rgb)
    current = api.normalize_latent_per_cell_v1(online_current)
    target = api.normalize_latent_per_cell_v1(model.encode_target(next_rgb))
    negative = api.normalize_latent_per_cell_v1(model.encode_target(negative_rgb))
    predictions = model.predict_all_actions(current)
    terms = api.action_query_joint_objective_v1(
        predictions,
        target,
        negative,
        current,
        torch.tensor((5,), dtype=torch.long),
    )
    labels = (
        torch.arange(64 * 64, dtype=torch.long).reshape(1, 64, 64) % 3
    )
    semantic = 0.5 * api.final_class_macro_nll_per_row(
        model.semantic_logits_from_latent(online_current), labels
    ).mean() + 0.5 * api.final_class_macro_nll_per_row(
        model.semantic_logits_from_latent(online_next), labels.roll(1, dims=2)
    ).mean()

    encoder_parameters = tuple(model.encoder.parameters())
    lift_parameters = tuple(model.bev_lift.parameters())
    shared_parameters = encoder_parameters + lift_parameters

    def assert_route_gradients(loss: torch.Tensor) -> None:
        gradients = torch.autograd.grad(
            loss,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        encoder_gradients = [
            value for value in gradients[: len(encoder_parameters)] if value is not None
        ]
        lift_gradients = [
            value for value in gradients[len(encoder_parameters) :] if value is not None
        ]
        for route in (encoder_gradients, lift_gradients):
            assert route
            assert all(bool(torch.isfinite(value).all()) for value in route)
            assert sum(float(value.detach().abs().sum()) for value in route) > 0.0

    assert_route_gradients(semantic / math.log(3.0))
    assert_route_gradients(terms.dynamics)
    (semantic / math.log(3.0) + terms.dynamics).backward()

    def assert_nonzero(module_or_parameter: nn.Module | torch.Tensor) -> None:
        parameters = (
            (module_or_parameter,)
            if isinstance(module_or_parameter, torch.Tensor)
            else tuple(module_or_parameter.parameters())
        )
        gradients = [value.grad for value in parameters if value.grad is not None]
        assert gradients
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0.0

    assert_nonzero(model.encoder)
    assert_nonzero(model.bev_lift)
    assert_nonzero(model.semantic_head)
    assert_nonzero(model.predictor.current_downsampler)
    assert_nonzero(model.predictor.action_embedding)
    assert_nonzero(model.predictor.future_queries)
    for block in model.predictor.blocks:
        assert_nonzero(block.attention)
        assert_nonzero(block.linear1)
        assert_nonzero(block.linear2)
    assert_nonzero(model.predictor.output_head)
    assert all(
        parameter.grad is None
        for module in model.target_modules()
        for parameter in module.parameters()
    )


def test_no_forbidden_predictor_branch_and_complete_component_receipts(
    model: api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1,
) -> None:
    predictor = model.predictor
    assert sum(isinstance(module, nn.MultiheadAttention) for module in predictor.modules()) == 2
    assert sum(isinstance(module, nn.Conv2d) for module in predictor.modules()) == 2
    assert not any(
        isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveMaxPool1d,
                nn.AdaptiveMaxPool2d,
            ),
        )
        for module in predictor.modules()
    )
    for forbidden in (
        "flow",
        "warp",
        "transport",
        "event",
        "mode",
        "codebook",
        "inverse_head",
        "action_classifier",
        "future_encoder",
        "pose",
        "odometry",
        "goal",
    ):
        assert not hasattr(predictor, forbidden)
    receipt = model.predictor_component_parameter_receipt()
    assert tuple(receipt) == (
        "downsampler",
        "action_embedding",
        "future_queries",
        "block_0_attention",
        "block_0_mlp",
        "block_1_attention",
        "block_1_mlp",
        "output_head",
    )
    names = [
        name
        for component in receipt.values()
        for name in component["ordered_parameter_names"]
    ]
    assert len(names) == len(set(names)) == 34
    assert sum(component["parameter_count"] for component in receipt.values()) == 504_384
    assert api.GeometryAnchoredDeformableBevLiftJointJepaV1 is (
        api.GeometryAnchoredActionQuerySpatialSuccessorJointJepaV1
    )
