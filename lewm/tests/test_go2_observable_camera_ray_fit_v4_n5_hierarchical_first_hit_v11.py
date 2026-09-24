"""CPU-only author and adversarial tests for Camera-ray N5 V11.

Tests use hand-constructed probabilities, tensors, source text, and temporary
directories only. They never open the canonical experiment data, RGB, output,
checkpoint, metric, gate, accelerator, G2, held-out, runtime, hardware, or
production namespaces.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
    hierarchical_first_hit_nll_breakdown_v9,
    hierarchical_first_hit_nll_from_log_probabilities_v9,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    ObservableCameraRayEvidenceV4Targets,
)


EXPECTED_SCHEDULE_SHA256 = (
    "fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380"
)


def _direct_breakdown(
    *,
    hit_probabilities: list[list[float]],
    no_hit_probabilities: list[float],
    target_bins: list[int | None],
    requires_grad: bool = False,
):
    hit = torch.tensor(hit_probabilities, dtype=torch.float64).transpose(0, 1)
    hit = hit.reshape(1, hit.shape[0], 1, hit.shape[1])
    no_hit = torch.tensor(no_hit_probabilities, dtype=torch.float64).reshape(
        1, 1, -1
    )
    hit_log = hit.log().requires_grad_(requires_grad)
    no_hit_log = no_hit.log().requires_grad_(requires_grad)
    hit_mask = torch.tensor(
        [value is not None for value in target_bins], dtype=torch.bool
    ).reshape(1, 1, -1)
    bins = torch.tensor(
        [0 if value is None else value for value in target_bins], dtype=torch.long
    ).reshape(1, 1, -1)
    return hierarchical_first_hit_nll_from_log_probabilities_v9(
        hit_log_probabilities=hit_log,
        no_hit_log_probability=no_hit_log,
        pixel_in_range_hit_mask=hit_mask,
        pixel_no_hit_mask=~hit_mask,
        pixel_hit_bin_index=bins,
    )


def _targets(hit_bins: list[int | None]) -> ObservableCameraRayEvidenceV4Targets:
    hit_mask = torch.tensor(
        [value is not None for value in hit_bins], dtype=torch.bool
    ).reshape(1, 1, -1)
    bins = torch.tensor(
        [0 if value is None else value for value in hit_bins], dtype=torch.long
    ).reshape(1, 1, -1)
    return ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=hit_mask,
        pixel_no_hit_mask=~hit_mask,
        pixel_hit_bin_index=bins,
        pixel_within_bin_offset_m=torch.zeros_like(hit_mask, dtype=torch.float64),
        ground_in_frustum=torch.zeros((1, 1, 1, 5), dtype=torch.bool),
        ground_clear_to_target=torch.zeros((1, 1, 1, 5), dtype=torch.bool),
    )


def _categorical_probabilities_to_ordered_hazards(
    *,
    hit_probabilities: list[list[float]],
    no_hit_probabilities: list[float],
) -> torch.Tensor:
    """Invert the ordered first-hit distribution into per-bin hazards."""

    rows = []
    for hit, no_hit in zip(hit_probabilities, no_hit_probabilities, strict=True):
        if sum(hit) + no_hit != pytest.approx(1.0):
            raise ValueError("categorical first-hit probabilities must sum to one")
        hazards = []
        for depth_bin, probability in enumerate(hit):
            remaining = no_hit + sum(hit[depth_bin:])
            hazard = probability / remaining
            hazards.append(math.log(hazard) - math.log1p(-hazard))
        rows.append(hazards)
    value = torch.tensor(rows, dtype=torch.float64).transpose(0, 1)
    return value.reshape(1, value.shape[0], 1, value.shape[1])


def test_v11_hierarchical_first_hit_matches_hand_computation() -> None:
    breakdown = _direct_breakdown(
        hit_probabilities=[
            [0.1, 0.1],
            [0.3, 0.1],
            [0.6, 0.15],
            [0.125, 0.375],
        ],
        no_hit_probabilities=[0.8, 0.6, 0.25, 0.5],
        target_bins=[None, None, 0, 1],
    )
    no_hit = (-math.log(0.8) - math.log(0.6)) / 2.0
    hit = (-math.log(0.75) - math.log(0.5)) / 2.0
    presence = 0.5 * (no_hit + hit)
    conditional = 0.5 * (-math.log(0.8) - math.log(0.75))
    assert breakdown.no_hit_presence_nll.item() == pytest.approx(no_hit)
    assert breakdown.hit_presence_nll.item() == pytest.approx(hit)
    assert breakdown.presence_nll.item() == pytest.approx(presence)
    assert breakdown.conditional_depth_nll.item() == pytest.approx(conditional)
    assert breakdown.total.item() == pytest.approx(
        0.5 * presence + 0.5 * conditional
    )
    assert breakdown.no_hit_count == 2
    assert breakdown.hit_count == 2
    assert breakdown.hit_distance_bin_counts == (1, 1)
    assert breakdown.nonempty_presence_group_count == 2
    assert breakdown.nonempty_conditional_depth_group_count == 2


def test_v11_presence_does_not_retain_old_one_over_groups_weight() -> None:
    breakdown = _direct_breakdown(
        hit_probabilities=[
            [0.0625, 0.0625, 0.0625, 0.0625],
            [0.8, 0.05, 0.05, 0.05],
            [0.05, 0.8, 0.05, 0.05],
            [0.05, 0.05, 0.8, 0.05],
            [0.05, 0.05, 0.05, 0.8],
        ],
        no_hit_probabilities=[0.75, 0.05, 0.05, 0.05, 0.05],
        target_bins=[None, 0, 1, 2, 3],
    )
    expected_no_hit = -math.log(0.75)
    expected_hit = -math.log(0.95)
    assert breakdown.presence_nll.item() == pytest.approx(
        0.5 * (expected_no_hit + expected_hit)
    )
    old_group_weighted_presence = (
        expected_no_hit + 4.0 * expected_hit
    ) / 5.0
    assert breakdown.presence_nll.item() != pytest.approx(
        old_group_weighted_presence
    )


def test_v11_presence_and_conditional_depth_invariances() -> None:
    target_bins: list[int | None] = [0, 1]
    first_logits = _categorical_probabilities_to_ordered_hazards(
        hit_probabilities=[[0.6, 0.15], [0.15, 0.6]],
        no_hit_probabilities=[0.25, 0.25],
    )
    redistributed_logits = _categorical_probabilities_to_ordered_hazards(
        hit_probabilities=[[0.675, 0.075], [0.075, 0.675]],
        no_hit_probabilities=[0.25, 0.25],
    )
    lower_presence_logits = _categorical_probabilities_to_ordered_hazards(
        hit_probabilities=[[0.4, 0.1], [0.1, 0.4]],
        no_hit_probabilities=[0.5, 0.5],
    )
    first = hierarchical_first_hit_nll_breakdown_v9(
        first_logits,
        _targets(target_bins),
    )
    redistributed = hierarchical_first_hit_nll_breakdown_v9(
        redistributed_logits,
        _targets(target_bins),
    )
    lower_presence = hierarchical_first_hit_nll_breakdown_v9(
        lower_presence_logits,
        _targets(target_bins),
    )
    assert first.presence_nll.item() == pytest.approx(
        redistributed.presence_nll.item()
    )
    assert first.conditional_depth_nll.item() != pytest.approx(
        redistributed.conditional_depth_nll.item()
    )
    assert first.conditional_depth_nll.item() == pytest.approx(
        lower_presence.conditional_depth_nll.item()
    )
    assert first.presence_nll.item() != pytest.approx(
        lower_presence.presence_nll.item()
    )


def test_v11_extreme_logits_have_finite_gradients_for_every_state_and_bin() -> None:
    hit_bins: list[int | None] = [None, 0, 1, 2, 3]
    logits = torch.tensor(
        [
            [80.0, -80.0, 25.0, -25.0, 0.0],
            [-80.0, 80.0, -25.0, 25.0, 0.0],
            [40.0, -40.0, 80.0, -80.0, 0.0],
            [-40.0, 40.0, -80.0, 80.0, 0.0],
        ],
        dtype=torch.float64,
    ).reshape(1, 4, 1, 5)
    logits.requires_grad_(True)
    breakdown = hierarchical_first_hit_nll_breakdown_v9(
        logits,
        _targets(hit_bins),
    )
    breakdown.total.backward()
    assert math.isfinite(breakdown.total.item())
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())
    assert bool((logits.grad.abs().sum(dim=1) > 0.0).all().item())
    assert bool((logits.grad.abs().sum(dim=(0, 2, 3)) > 0.0).all().item())


def test_v11_plus_minus_ten_thousand_logits_remain_finite() -> None:
    logits = torch.tensor(
        [
            [10000.0, -10000.0, 10000.0],
            [-10000.0, 10000.0, -10000.0],
            [10000.0, 10000.0, -10000.0],
        ],
        dtype=torch.float64,
    ).reshape(1, 3, 1, 3)
    logits.requires_grad_(True)
    breakdown = hierarchical_first_hit_nll_breakdown_v9(
        logits,
        _targets([None, 0, 2]),
    )
    breakdown.total.backward()
    assert bool(torch.isfinite(breakdown.total).item())
    assert bool(torch.isfinite(breakdown.presence_nll).item())
    assert bool(torch.isfinite(breakdown.conditional_depth_nll).item())
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())


@pytest.mark.parametrize("hit_bins", [[None, None], [0, 1]])
def test_v11_empty_groups_keep_a_finite_zero_gradient_term(
    hit_bins: list[int | None],
) -> None:
    logits = torch.zeros((1, 2, 1, 2), dtype=torch.float64, requires_grad=True)
    breakdown = hierarchical_first_hit_nll_breakdown_v9(
        logits,
        _targets(hit_bins),
    )
    breakdown.total.backward()
    assert bool(torch.isfinite(breakdown.total).item())
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all().item())
    if all(value is None for value in hit_bins):
        assert breakdown.conditional_depth_nll.item() == 0.0
        assert breakdown.nonempty_conditional_depth_group_count == 0
    else:
        assert breakdown.no_hit_presence_nll.item() == 0.0
        assert breakdown.nonempty_presence_group_count == 1


def test_v11_rejects_unnormalized_or_nonfinite_log_probabilities() -> None:
    hit = torch.log(torch.tensor([[[[0.4]], [[0.4]]]], dtype=torch.float64))
    no_hit = torch.log(torch.tensor([[[0.4]]], dtype=torch.float64))
    mask = torch.tensor([[[True]]])
    bins = torch.tensor([[[0]]], dtype=torch.long)
    with pytest.raises(ValueError, match="not normalized"):
        hierarchical_first_hit_nll_from_log_probabilities_v9(
            hit_log_probabilities=hit,
            no_hit_log_probability=no_hit,
            pixel_in_range_hit_mask=mask,
            pixel_no_hit_mask=~mask,
            pixel_hit_bin_index=bins,
        )
    bad = hit.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        hierarchical_first_hit_nll_from_log_probabilities_v9(
            hit_log_probabilities=bad,
            no_hit_log_probability=torch.log(
                torch.tensor([[[0.2]]], dtype=torch.float64)
            ),
            pixel_in_range_hit_mask=mask,
            pixel_no_hit_mask=~mask,
            pixel_hit_bin_index=bins,
        )


def test_v11_frozen_full_panel_schedule_is_exact_and_deterministic() -> None:
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    first = base._deterministic_training_batches(
        frame_count=5,
        batch_size=5,
        steps=4000,
        seed=20260710,
    )
    second = base._deterministic_training_batches(
        frame_count=5,
        batch_size=5,
        steps=4000,
        seed=20260710,
    )
    assert first == second
    assert len(first) == 4000
    assert sum(len(batch) for batch in first) == 20000
    assert all(
        len(batch) == 5 and set(batch) == set(range(5)) for batch in first
    )
    assert base.canonical_json_sha256(first) == EXPECTED_SCHEDULE_SHA256
    expected_trace = (1, *range(100, 4001, 100))
    assert len(expected_trace) == 41
    assert expected_trace[0] == 1 and expected_trace[-1] == 4000


def test_v11_trainer_and_verifier_preserve_all_non_first_hit_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.models import (
        observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 as loss_module,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base
    from scripts import (
        train_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11
        as trainer,
    )
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11
        as verifier,
    )

    target = SimpleNamespace(ground_in_frustum=torch.tensor([True]))
    raw = SimpleNamespace(
        ground_query_in_frustum=torch.tensor([True]),
        pixel_first_hit_hazard_logits=torch.tensor(0.0),
        ground_clear_to_target_logits=torch.tensor(0.0),
        ground_target_distance_m=torch.tensor(0.0),
    )
    soft = object()
    monkeypatch.setattr(
        base,
        "derive_observable_camera_ray_evidence_v4_targets",
        lambda **_kwargs: target,
    )
    monkeypatch.setattr(
        loss_module,
        "hierarchical_first_hit_nll_breakdown_v9",
        lambda *_args: SimpleNamespace(total=torch.tensor(2.0)),
    )
    monkeypatch.setattr(
        base,
        "_skew_balanced_pixel_offset_loss",
        lambda *_args: torch.tensor(4.0),
    )
    monkeypatch.setattr(
        base,
        "balanced_ground_clear_bce_v4",
        lambda *_args: torch.tensor(6.0),
    )
    monkeypatch.setattr(
        base,
        "soft_rasterize_observable_camera_ray_evidence_v4",
        lambda *_args, **_kwargs: soft,
    )
    monkeypatch.setattr(
        base,
        "hierarchical_raster_cross_entropy_v4",
        lambda *_args: SimpleNamespace(total=torch.tensor(8.0)),
    )
    batch = SimpleNamespace(
        pixel_hit_mask=None,
        pixel_first_hit_distance_m=None,
        ground_support_in_frustum=None,
        ground_support_clear_to_target=None,
        image=None,
        camera_origin_body_m=None,
        camera_basis_body_fru=None,
        ground_plane_z_body_m=None,
        target_raster_labels=None,
    )
    for compute in (
        trainer.compute_four_equal_v9_losses,
        verifier.compute_four_equal_v9_losses_for_verification,
    ):
        total, components, returned_raw, returned_target, returned_soft = compute(
            lambda *_args: raw,
            batch,
        )
        assert total.item() == pytest.approx(5.0)
        assert tuple(components) == (
            "hierarchical_first_hit_nll",
            "target_bin_offset_smooth_l1",
            "ground_clear_distance_state_balanced_bce",
            "derived_raster_hierarchical_bce",
        )
        assert [value.item() for value in components.values()] == [2.0, 4.0, 6.0, 8.0]
        assert returned_raw is raw
        assert returned_target is target
        assert returned_soft is soft


def test_v11_standalone_verifier_is_compute_only() -> None:
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v11
        as verifier,
    )

    with pytest.raises(PermissionError, match="compute-only"):
        verifier.main([])
