from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from lewm.models.go2_task_coupled_recurrent_dynamics_v1 import (
    CANDIDATE_WIDTH,
    CONTEXT_STEPS,
    MOTION_WIDTH,
    OUTPUT_WIDTH,
    TOKEN_COUNT,
    VISUAL_WIDTH,
    TaskCoupledRecurrentDynamicsV1,
    initialize_task_coupled_recurrent_dynamics_v1,
    recurrent_dynamics_state_identity_v1,
)


def _inputs(
    *, batch: int = 2, actions: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(2_026_080_401)
    visual = torch.randn(
        batch,
        CONTEXT_STEPS,
        TOKEN_COUNT,
        VISUAL_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    motion = torch.randn(
        batch,
        CONTEXT_STEPS,
        MOTION_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    motion[:, 0] = 0.0
    candidates = torch.randn(
        batch,
        actions,
        CANDIDATE_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    return visual, motion, candidates


def test_forward_has_exact_shape_finite_values_and_finite_gradients() -> None:
    model = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_402)
    visual, motion, candidates = _inputs()

    prediction = model(visual, motion, candidates)

    assert prediction.shape == (2, 4, OUTPUT_WIDTH)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    (prediction - 1.0).square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_seeded_initialization_and_state_identity_are_exact_and_rng_isolated() -> None:
    torch.manual_seed(73)
    rng_before = torch.random.get_rng_state().clone()
    first = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_403)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    second = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_403)
    changed_seed = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_404)

    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    first_identity = recurrent_dynamics_state_identity_v1(first)
    assert len(first_identity) == 64
    assert first_identity == recurrent_dynamics_state_identity_v1(second)
    assert first_identity != recurrent_dynamics_state_identity_v1(changed_seed)


def test_matched_arms_are_identical_under_identical_inputs() -> None:
    visual_arm = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_405)
    no_vision_arm = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_405)
    visual, motion, candidates = _inputs()
    identical_visual_slots = torch.zeros_like(visual)

    visual_prediction = visual_arm(identical_visual_slots, motion, candidates)
    no_vision_prediction = no_vision_arm(
        identical_visual_slots, motion, candidates
    )

    assert recurrent_dynamics_state_identity_v1(
        visual_arm
    ) == recurrent_dynamics_state_identity_v1(no_vision_arm)
    assert torch.equal(visual_prediction, no_vision_prediction)


def test_nonzero_visual_context_changes_prediction() -> None:
    model = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_406)
    visual, motion, candidates = _inputs()
    # Scientific initialization makes the decoded residual exactly zero.  Once
    # the direct head receives its first gradient, the recurrent visual path
    # must influence the output.
    with torch.no_grad():
        model.query_output.weight.fill_(0.1)

    with_visual = model(visual, motion, candidates)
    without_visual = model(torch.zeros_like(visual), motion, candidates)

    assert torch.isfinite(with_visual).all()
    assert torch.isfinite(without_visual).all()
    visual_delta = (with_visual - without_visual).detach().abs().max().item()
    assert visual_delta > 1.0e-6


def _nonfinite_visual(
    visual: torch.Tensor,
    motion: torch.Tensor,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    changed = visual.clone()
    changed[0, 0, 0, 0] = float("nan")
    return changed, motion, candidates


@pytest.mark.parametrize(
    ("mutator", "error"),
    (
        (lambda v, m, c: (v[:, :, :-1], m, c), ValueError),
        (lambda v, m, c: (v, m[:, :-1], c), ValueError),
        (lambda v, m, c: (v, m, c[:, :0]), ValueError),
        (lambda v, m, c: (v, m, c[:, :, :-1]), ValueError),
        (lambda v, m, c: (v[:0], m[:0], c[:0]), ValueError),
        (lambda v, m, c: (v.to(torch.float64), m, c), TypeError),
        (_nonfinite_visual, FloatingPointError),
    ),
)
def test_invalid_forward_inputs_are_rejected(
    mutator: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    error: type[Exception],
) -> None:
    model = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_407)
    inputs = mutator(*_inputs())

    with pytest.raises(error):
        model(*inputs)


@pytest.mark.parametrize("seed", (True, -1, 2**63, 1.5))
def test_invalid_initialization_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError):
        initialize_task_coupled_recurrent_dynamics_v1(seed)  # type: ignore[arg-type]


def test_state_identity_rejects_inventory_dtype_and_nonfinite_changes() -> None:
    model = initialize_task_coupled_recurrent_dynamics_v1(2_026_080_408)
    state = dict(model.state_dict())

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="inventory"):
        recurrent_dynamics_state_identity_v1(missing)

    wrong_dtype = dict(state)
    name = next(iter(wrong_dtype))
    wrong_dtype[name] = wrong_dtype[name].to(torch.float64)
    with pytest.raises(ValueError, match="finite float32"):
        recurrent_dynamics_state_identity_v1(wrong_dtype)

    nonfinite = dict(state)
    name = next(iter(nonfinite))
    nonfinite[name] = nonfinite[name].clone()
    nonfinite[name].reshape(-1)[0] = float("inf")
    with pytest.raises(ValueError, match="finite float32"):
        recurrent_dynamics_state_identity_v1(nonfinite)

    non_tensor = dict(state)
    non_tensor[name] = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="non-tensor"):
        recurrent_dynamics_state_identity_v1(non_tensor)


def test_direct_construction_obeys_the_public_forward_contract() -> None:
    model = TaskCoupledRecurrentDynamicsV1()
    visual, motion, candidates = _inputs(batch=1, actions=1)

    prediction = model(visual, motion, candidates)

    assert prediction.shape == (1, 1, OUTPUT_WIDTH)
    assert torch.isfinite(prediction).all()
