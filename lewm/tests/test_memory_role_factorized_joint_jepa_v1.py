from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.memory_role_factorized_joint_jepa_v1 import (
    LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
    PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
    ActionConditionedLocalControlPredictorV1,
    MemoryRoleFactorizedJointJepaV1,
    MemoryRoleFactorizerV1,
    PlaceKeyPredictorV1,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


@pytest.fixture(scope="module")
def model() -> MemoryRoleFactorizedJointJepaV1:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(28_901)
        fitted = ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)
    return MemoryRoleFactorizedJointJepaV1(fitted, _sweep_masks()).eval()


def test_factorizer_shapes_normalization_and_joint_gradients() -> None:
    factorizer = MemoryRoleFactorizerV1().train()
    place_predictor = PlaceKeyPredictorV1().train()
    local_predictor = ActionConditionedLocalControlPredictorV1().train()
    latent = torch.randn(
        (3, 64, 64, 64),
        generator=torch.Generator().manual_seed(28_902),
        requires_grad=True,
    )
    current = factorizer(latent)
    predicted_place = place_predictor(current.place_key)
    predicted_local = local_predictor(current.local_control, torch.eye(9)[[0, 4, 8]])

    assert tuple(current.place_key.shape) == (
        3,
        PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
    )
    assert tuple(current.local_control.shape) == (
        3,
        *LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(current.place_key, dim=1),
        torch.ones(3),
        rtol=0.0,
        atol=1.0e-5,
    )

    place_target = torch.nn.functional.normalize(
        torch.randn(
            predicted_place.shape,
            generator=torch.Generator().manual_seed(28_903),
        )
    )
    local_target = torch.randn(
        predicted_local.shape,
        generator=torch.Generator().manual_seed(28_904),
    )
    loss = (1.0 - (predicted_place * place_target).sum(dim=1)).mean()
    loss = loss + torch.nn.functional.smooth_l1_loss(predicted_local, local_target)
    parameters = (
        *factorizer.parameters(),
        *place_predictor.parameters(),
        *local_predictor.parameters(),
    )
    gradients = torch.autograd.grad(loss, (latent, *parameters))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(int(torch.count_nonzero(gradient)) > 0 for gradient in gradients)


def test_only_local_prediction_is_action_conditioned() -> None:
    place_predictor = PlaceKeyPredictorV1().eval()
    local_predictor = ActionConditionedLocalControlPredictorV1().eval()
    generator = torch.Generator().manual_seed(28_905)
    place = torch.nn.functional.normalize(torch.randn((1, 64), generator=generator))
    local = torch.randn((1, 32, 16, 16), generator=generator)

    assert torch.equal(place_predictor(place), place_predictor(place))
    assert not torch.equal(
        local_predictor(local, torch.eye(9)[[0]]),
        local_predictor(local, torch.eye(9)[[1]]),
    )


def test_model_binds_frozen_ema_role_target_and_complete_online_groups(
    model: MemoryRoleFactorizedJointJepaV1,
) -> None:
    online = model.role_factorizer.state_dict()
    target = model.target_role_factorizer.state_dict()
    assert online.keys() == target.keys()
    assert all(torch.equal(value, target[name]) for name, value in online.items())
    assert not any(
        parameter.requires_grad
        for module in model.target_modules()
        for parameter in module.parameters()
    )

    selected = model.trainable_parameter_groups_memory_role_factorized_v1().online
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    assert {id(parameter) for _, parameter in selected} == {
        id(parameter) for parameter in trainable
    }
    assert len({id(parameter) for _, parameter in selected}) == len(selected)

    model.train()
    assert model.role_factorizer.training
    assert not model.target_role_factorizer.training


def test_role_factorizer_target_moves_only_through_the_shared_ema_update(
    model: MemoryRoleFactorizedJointJepaV1,
) -> None:
    online = next(model.role_factorizer.parameters())
    target = next(model.target_role_factorizer.parameters())
    before_target = target.detach().clone()
    before_count = int(model.ema_update_count.item())
    with torch.no_grad():
        online.add_(1.0)
    model.update_target_ema_after_optimizer_step()

    assert int(model.ema_update_count.item()) == before_count + 1
    assert not torch.equal(target, before_target)
    assert target.grad is None
    assert target.requires_grad is False


def test_model_construction_is_deterministic_and_restores_rng() -> None:
    torch.random.default_generator.manual_seed(28_906)
    fitted = ObservableCameraRayEvidenceV4Model().eval()

    torch.random.default_generator.manual_seed(28_907)
    first_rng = torch.random.get_rng_state().clone()
    first = MemoryRoleFactorizedJointJepaV1(fitted, _sweep_masks()).eval()
    assert torch.equal(torch.random.get_rng_state(), first_rng)

    torch.random.default_generator.manual_seed(28_908)
    second_rng = torch.random.get_rng_state().clone()
    second = MemoryRoleFactorizedJointJepaV1(fitted, _sweep_masks()).eval()
    assert torch.equal(torch.random.get_rng_state(), second_rng)
    for prefix in ("role_factorizer", "place_predictor", "local_predictor"):
        first_state = getattr(first, prefix).state_dict()
        second_state = getattr(second, prefix).state_dict()
        assert all(
            torch.equal(value, second_state[name])
            for name, value in first_state.items()
        )


@pytest.mark.parametrize(
    "action",
    (
        torch.zeros((1, 9)),
        torch.ones((1, 9)),
        torch.full((1, 9), 1.0 / 9.0),
        torch.eye(9, dtype=torch.float64)[[0]],
        torch.eye(9)[:2],
    ),
)
def test_local_predictor_rejects_invalid_action_rows(action: torch.Tensor) -> None:
    predictor = ActionConditionedLocalControlPredictorV1().eval()
    local = torch.zeros((1, 32, 16, 16))
    with pytest.raises((TypeError, ValueError)):
        predictor(local, action)
