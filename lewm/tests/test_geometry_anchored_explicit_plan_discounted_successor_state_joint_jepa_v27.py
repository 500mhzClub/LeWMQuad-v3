from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    ONLINE_TRAINABLE_PARAMETER_COUNT_V18,
    PREDICTOR_GROUP_PARAMETER_COUNT_V18,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
    SHARED_ROUTE_PARAMETER_COUNT_V18,
    GeometryAnchoredSweptProgressSurvivalJointJepaV18,
)
from lewm.models.geometry_anchored_explicit_plan_discounted_successor_state_joint_jepa_v27 import (
    ONLINE_TRAINABLE_PARAMETER_COUNT_V27,
    PLAN_GRADIENT_ROUTE_PARAMETER_COUNT_V27,
    PLAN_INITIALIZATION_SEED_V27,
    PLAN_PREDICTOR_PARAMETER_COUNT_V27,
    PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27,
    PREDICTOR_GROUP_PARAMETER_COUNT_V27,
    ExplicitPlanDiscountedSuccessorStatePredictorV27,
    GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def n320_fit_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(27_001)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def matched_models(
    n320_fit_model: ObservableCameraRayEvidenceV4Model,
) -> tuple[
    GeometryAnchoredSweptProgressSurvivalJointJepaV18,
    GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27,
]:
    caller_rng = torch.random.get_rng_state().clone()
    v18 = GeometryAnchoredSweptProgressSurvivalJointJepaV18(
        n320_fit_model,
        _sweep_masks(),
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    v27 = GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27(
        n320_fit_model,
        _sweep_masks(),
    ).eval()
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    return v18, v27


@pytest.fixture(scope="module")
def plan_predictor() -> ExplicitPlanDiscountedSuccessorStatePredictorV27:
    return ExplicitPlanDiscountedSuccessorStatePredictorV27().eval()


def _weighted_layers(
    predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
) -> tuple[nn.Module, ...]:
    return (
        *predictor.action_embeddings,
        predictor.plan_linear1,
        predictor.plan_linear2,
        predictor.state_projection,
        predictor.fusion_projection,
        predictor.output_projection,
    )


def test_exact_plan_head_architecture_initialization_and_rng_restoration() -> None:
    torch.random.default_generator.manual_seed(27_101)
    first_caller_rng = torch.random.get_rng_state().clone()
    first = ExplicitPlanDiscountedSuccessorStatePredictorV27()
    assert torch.equal(torch.random.get_rng_state(), first_caller_rng)

    torch.random.default_generator.manual_seed(27_102)
    second_caller_rng = torch.random.get_rng_state().clone()
    second = ExplicitPlanDiscountedSuccessorStatePredictorV27()
    assert torch.equal(torch.random.get_rng_state(), second_caller_rng)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )

    assert len(first.action_embeddings) == 4
    assert all(
        isinstance(embedding, nn.Embedding)
        and embedding.num_embeddings == 9
        and embedding.embedding_dim == 16
        for embedding in first.action_embeddings
    )
    assert (first.plan_linear1.in_features, first.plan_linear1.out_features) == (
        64,
        128,
    )
    assert (first.plan_linear2.in_features, first.plan_linear2.out_features) == (
        128,
        128,
    )
    assert first.state_projection.kernel_size == (3, 3)
    assert first.fusion_projection.kernel_size == (3, 3)
    assert first.output_projection.kernel_size == (1, 1)
    assert first.activation.approximate == "none"
    assert not any(
        isinstance(module, (nn.Dropout, nn.LayerNorm, nn.BatchNorm2d))
        for module in first.modules()
    )

    generator = torch.Generator(device="cpu").manual_seed(
        PLAN_INITIALIZATION_SEED_V27
    )
    for layer in _weighted_layers(first):
        expected = torch.empty_like(layer.weight)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(layer.weight, expected)
        if getattr(layer, "bias", None) is not None:
            assert torch.count_nonzero(layer.bias) == 0
    parameters = tuple(first.parameters())
    assert len(parameters) == PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27 == 14
    assert (
        sum(parameter.numel() for parameter in parameters)
        == PLAN_PREDICTOR_PARAMETER_COUNT_V27
        == 103_424
    )


def test_ordered_action_embeddings_feed_the_frozen_four_segments(
    plan_predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
) -> None:
    captured: list[torch.Tensor] = []

    def capture_input(_module: nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        captured.append(values[0].detach().clone())

    hook = plan_predictor.plan_linear1.register_forward_pre_hook(capture_input)
    latent = torch.randn(
        (2, 64, 64, 64),
        generator=torch.Generator().manual_seed(27_201),
    )
    plans = torch.tensor(((0, 2, 4, 6), (1, 3, 5, 7)), dtype=torch.long)
    try:
        with torch.no_grad():
            result = plan_predictor(latent, plans)
    finally:
        hook.remove()

    assert tuple(result.shape) == (2, 64, 64, 64)
    assert bool(torch.isfinite(result).all())
    assert len(captured) == 1
    expected = torch.cat(
        tuple(
            embedding(plans[:, position])
            for position, embedding in enumerate(plan_predictor.action_embeddings)
        ),
        dim=1,
    )
    assert torch.equal(captured[0], expected)
    assert all(
        torch.equal(
            captured[0][:, 16 * position : 16 * (position + 1)],
            plan_predictor.action_embeddings[position](plans[:, position]),
        )
        for position in range(4)
    )


def test_plan_head_is_absolute_and_has_complete_gradients(
    plan_predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
) -> None:
    absolute = copy.deepcopy(plan_predictor)
    with torch.no_grad():
        absolute.output_projection.weight.zero_()
        absolute.output_projection.bias.zero_()
        latent = torch.randn(
            (1, 64, 64, 64),
            generator=torch.Generator().manual_seed(27_301),
        )
        output = absolute(latent, torch.tensor(((1, 2, 3, 4),)))
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(latent) > 0

    predictor = copy.deepcopy(plan_predictor).train()
    latent = torch.randn(
        (4, 64, 64, 64),
        generator=torch.Generator().manual_seed(27_302),
        requires_grad=True,
    )
    plans = torch.tensor(
        ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)),
        dtype=torch.long,
    )
    coefficients = torch.randn(
        latent.shape,
        generator=torch.Generator().manual_seed(27_303),
    )
    loss = (predictor(latent, plans) * coefficients).mean()
    gradients = torch.autograd.grad(loss, (latent, *tuple(predictor.parameters())))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert int(torch.count_nonzero(gradients[0])) > 0
    assert all(int(torch.count_nonzero(gradient)) > 0 for gradient in gradients[1:])


@pytest.mark.parametrize(
    ("latent", "error"),
    (
        (torch.zeros((64, 64, 64)), ValueError),
        (torch.zeros((0, 64, 64, 64)), ValueError),
        (torch.zeros((1, 63, 64, 64)), ValueError),
        (torch.zeros((1, 64, 64, 64), dtype=torch.float64), TypeError),
        (torch.full((1, 64, 64, 64), float("nan")), FloatingPointError),
        (torch.full((1, 64, 64, 64), float("inf")), FloatingPointError),
    ),
)
def test_plan_head_rejects_invalid_latents(
    plan_predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
    latent: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        plan_predictor(latent, torch.zeros((1, 4), dtype=torch.long))


@pytest.mark.parametrize(
    ("plan", "error"),
    (
        (torch.zeros((1, 3), dtype=torch.long), ValueError),
        (torch.zeros((2, 4), dtype=torch.long), ValueError),
        (torch.zeros((1, 4), dtype=torch.int32), TypeError),
        (torch.zeros((1, 4), dtype=torch.float32), TypeError),
        (torch.tensor(((-1, 0, 1, 2),), dtype=torch.long), ValueError),
        (torch.tensor(((0, 1, 2, 9),), dtype=torch.long), ValueError),
    ),
)
def test_plan_head_rejects_invalid_action_plans(
    plan_predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
    plan: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        plan_predictor(torch.zeros((1, 64, 64, 64)), plan)


def test_plan_head_rejects_device_and_output_finiteness_mismatches(
    plan_predictor: ExplicitPlanDiscountedSuccessorStatePredictorV27,
) -> None:
    meta_plan = torch.empty((1, 4), dtype=torch.long, device="meta")
    with pytest.raises(TypeError, match="share a device"):
        plan_predictor(torch.zeros((1, 64, 64, 64)), meta_plan)

    broken = copy.deepcopy(plan_predictor)
    with torch.no_grad():
        broken.output_projection.bias.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="output is nonfinite"):
        broken(
            torch.zeros((1, 64, 64, 64)),
            torch.zeros((1, 4), dtype=torch.long),
        )


def test_v27_preserves_every_inherited_v18_tensor_and_predictor(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV18,
        GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27,
    ],
) -> None:
    v18, v27 = matched_models
    assert isinstance(v27, GeometryAnchoredSweptProgressSurvivalJointJepaV18)
    inherited_state = v18.state_dict()
    v27_state = v27.state_dict()
    assert set(inherited_state).issubset(v27_state)
    assert all(
        torch.equal(value, v27_state[name])
        for name, value in inherited_state.items()
    )
    assert not hasattr(v27, "target_plan_predictor")
    assert int(v27.target_hard_sync_count.item()) == 1
    assert int(v27.ema_update_count.item()) == 0

    latent = torch.randn(
        (1, 64, 64, 64),
        generator=torch.Generator().manual_seed(27_401),
    )
    with torch.no_grad():
        old = v18.predict_all_actions_with_survival(latent)
        retained = v27.predict_all_actions_with_survival(latent)
        plan = torch.tensor(((0, 1, 2, 3),), dtype=torch.long)
        direct = v27.plan_predictor(latent, plan)
        public = v27.predict_plan_successor(latent, plan)
    assert torch.equal(old.predicted_latents, retained.predicted_latents)
    assert torch.equal(old.survival_logits, retained.survival_logits)
    assert torch.equal(direct, public)


def test_extended_inventory_preserves_the_exact_old_predictor_view(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV18,
        GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27,
    ],
) -> None:
    _v18, model = matched_models
    inherited = model.trainable_parameter_groups_v18()
    extended = model.trainable_parameter_groups_v27()
    assert extended.inherited_v18 == inherited
    assert tuple(
        sum(parameter.numel() for _, parameter in group)
        for group in inherited
    ) == (
        SHARED_ROUTE_PARAMETER_COUNT_V18,
        REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
        PREDICTOR_GROUP_PARAMETER_COUNT_V18,
    )
    assert len(inherited.predictor) == 15
    assert (
        sum(parameter.numel() for _, parameter in inherited.predictor)
        == PREDICTOR_GROUP_PARAMETER_COUNT_V18
        == 259_073
    )
    assert all(name.startswith("predictor.") for name, _ in inherited.predictor)
    assert all(
        not name.startswith("plan_predictor.")
        for group in inherited
        for name, _ in group
    )

    assert len(extended.plan_predictor) == 14
    assert sum(parameter.numel() for _, parameter in extended.plan_predictor) == (
        PLAN_PREDICTOR_PARAMETER_COUNT_V27
    )
    assert all(
        name.startswith("plan_predictor.")
        for name, _ in extended.plan_predictor
    )
    assert (
        sum(parameter.numel() for _, parameter in extended.predictor)
        == PREDICTOR_GROUP_PARAMETER_COUNT_V27
        == 362_497
    )
    assert (
        sum(parameter.numel() for _, parameter in extended.online)
        == ONLINE_TRAINABLE_PARAMETER_COUNT_V27
        == 3_542_827
    )
    assert ONLINE_TRAINABLE_PARAMETER_COUNT_V27 == (
        ONLINE_TRAINABLE_PARAMETER_COUNT_V18 + PLAN_PREDICTOR_PARAMETER_COUNT_V27
    )
    selected_ids = [id(parameter) for _, parameter in extended.online]
    assert len(selected_ids) == len(set(selected_ids))
    assert set(selected_ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    volume = tuple(
        (name, parameter)
        for name, parameter in extended.representation
        if name.startswith(
            ("bev_lift.point_projection.", "bev_lift.volume_block.")
        )
    )
    plan_route = extended.shared + volume + extended.plan_predictor
    assert (
        sum(parameter.numel() for _, parameter in plan_route)
        == PLAN_GRADIENT_ROUTE_PARAMETER_COUNT_V27
        == 3_209_768
    )
    assert not any(
        name.startswith(("semantic_head.", "predictor.", "target_"))
        for name, _ in plan_route
    )


def test_parameter_inventory_fails_closed_on_an_unknown_trainable_name(
    matched_models: tuple[
        GeometryAnchoredSweptProgressSurvivalJointJepaV18,
        GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27,
    ],
) -> None:
    _v18, model = matched_models
    model.register_parameter(
        "unexpected_v27_parameter",
        nn.Parameter(torch.zeros(1)),
    )
    try:
        with pytest.raises(RuntimeError, match="inherited V18 parameter view"):
            model.trainable_parameter_groups_v18()
        with pytest.raises(RuntimeError):
            model.trainable_parameter_groups_v27()
    finally:
        delattr(model, "unexpected_v27_parameter")
