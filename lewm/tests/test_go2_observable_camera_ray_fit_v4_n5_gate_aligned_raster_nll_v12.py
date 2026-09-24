"""CPU-only author and adversarial tests for Camera-ray N5 V12.

Tests use hand-constructed probabilities, tensors, source text, and temporary
directories only. They never open the canonical experiment data, RGB, output,
checkpoint, metric, gate, accelerator, G2, held-out, runtime, hardware, or
production namespaces.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
    hierarchical_first_hit_nll_breakdown_v9,
    hierarchical_first_hit_nll_from_log_probabilities_v9,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    ObservableCameraRayEvidenceV4Targets,
    SoftObservableCameraRayRasterV4,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4RawOutput,
)
from lewm.models.observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12 import (
    branch_reduction_decomposition_v12,
    compose_gate_aligned_objective_v12,
    derived_raster_cell_nll_v12,
    merge_raster_nll_diagnostics_v12,
    raster_nll_diagnostics_v12,
    validate_raster_nll_diagnostics_v12,
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


def test_v12_hierarchical_first_hit_matches_hand_computation() -> None:
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


def test_v12_presence_does_not_retain_old_one_over_groups_weight() -> None:
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


def test_v12_presence_and_conditional_depth_invariances() -> None:
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


def test_v12_extreme_logits_have_finite_gradients_for_every_state_and_bin() -> None:
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


def test_v12_plus_minus_ten_thousand_logits_remain_finite() -> None:
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
def test_v12_empty_groups_keep_a_finite_zero_gradient_term(
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


def test_v12_rejects_unnormalized_or_nonfinite_log_probabilities() -> None:
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


def test_v12_frozen_full_panel_schedule_is_exact_and_deterministic() -> None:
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


def test_v12_trainer_and_verifier_preserve_all_non_first_hit_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.models import (
        observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 as loss_module,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base
    from scripts import (
        train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12
        as trainer,
    )
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12
        as verifier,
    )

    target = SimpleNamespace(ground_in_frustum=torch.tensor([True]))
    raw = SimpleNamespace(
        ground_query_in_frustum=torch.tensor([True]),
        pixel_first_hit_hazard_logits=torch.tensor(0.0),
        ground_clear_to_target_logits=torch.tensor(0.0),
        ground_target_distance_m=torch.tensor(0.0),
    )
    probabilities = torch.tensor(
        [[[[0.7]], [[0.2]], [[0.1]]]], dtype=torch.float32
    )
    soft = SimpleNamespace(class_probabilities=probabilities)
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
        target_raster_labels=torch.tensor([[[0]]], dtype=torch.long),
    )
    for compute in (
        trainer.compute_gate_aligned_v12_losses,
        verifier.compute_gate_aligned_v12_losses_for_verification,
    ):
        total, v11_base, components, returned_raw, returned_target, returned_soft = (
            compute(lambda *_args: raw, batch)
        )
        cell_nll = -math.log(0.7)
        assert v11_base.item() == pytest.approx(5.0)
        assert total.item() == pytest.approx(5.0 + 0.25 * cell_nll)
        assert tuple(components) == (
            "hierarchical_first_hit_nll",
            "target_bin_offset_smooth_l1",
            "ground_clear_distance_state_balanced_bce",
            "derived_raster_hierarchical_bce",
            "derived_raster_cell_nll",
        )
        assert [value.item() for value in components.values()] == pytest.approx(
            [2.0, 4.0, 6.0, 8.0, cell_nll]
        )
        assert returned_raw is raw
        assert returned_target is target
        assert returned_soft is soft


def test_v12_standalone_verifier_is_compute_only() -> None:
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12
        as verifier,
    )

    with pytest.raises(PermissionError, match="compute-only"):
        verifier.main([])


def test_v12_exact_cell_nll_formula_gradients_and_nonmutation() -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0], [0.5, 0.25]],
                [[0.0, 2.0], [-0.5, 0.75]],
                [[-1.0, 0.0], [1.5, -0.25]],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    probabilities = torch.softmax(logits, dim=1)
    labels = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.int16)
    probability_before = probabilities.detach().clone()
    label_before = labels.clone()
    result = derived_raster_cell_nll_v12(probabilities, labels)
    expected = -probabilities.gather(
        1, labels.to(dtype=torch.long)[:, None]
    ).squeeze(1).clamp_min(torch.finfo(torch.float32).eps).log().mean()
    assert torch.equal(result, expected)
    result.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())
    assert bool((logits.grad != 0.0).any().item())
    assert torch.equal(probabilities.detach(), probability_before)
    assert torch.equal(labels, label_before)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        (torch.ones((1, 3, 1, 1), dtype=torch.float64) / 3, torch.zeros((1, 1, 1), dtype=torch.long), "float32"),
        (torch.ones((1, 3, 1, 1), dtype=torch.float32) / 3, torch.zeros((1, 1), dtype=torch.long), "shape"),
        (torch.ones((1, 3, 1, 1), dtype=torch.float32) / 3, torch.full((1, 1, 1), 3, dtype=torch.long), "lie in"),
        (torch.tensor([[[[0.4]], [[0.4]], [[0.4]]]], dtype=torch.float32), torch.zeros((1, 1, 1), dtype=torch.long), "normalized"),
    ],
)
def test_v12_cell_nll_rejects_schema_crossing(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        derived_raster_cell_nll_v12(probabilities, labels)


def _metric_accumulator_raster_nll(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (
        ObservableCameraRayFitV4MetricAccumulator,
    )

    batch = probabilities.shape[0]
    hazard = torch.zeros((batch, 1, 1, 1), dtype=torch.float32)
    ground_shape = (batch, 1, 1, 5)
    raw = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazard,
        pixel_within_bin_offset_m=torch.zeros_like(hazard),
        ground_clear_to_target_logits=torch.zeros(ground_shape),
        ground_query_in_frustum=torch.zeros(ground_shape, dtype=torch.bool),
        ground_query_uv_px=torch.zeros((*ground_shape, 2)),
        ground_target_distance_m=torch.zeros(ground_shape),
    )
    hit = torch.zeros((batch, 1, 1), dtype=torch.bool)
    targets = ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=hit,
        pixel_no_hit_mask=~hit,
        pixel_hit_bin_index=torch.zeros_like(hit, dtype=torch.long),
        pixel_within_bin_offset_m=torch.zeros_like(hit, dtype=torch.float32),
        ground_in_frustum=torch.zeros(ground_shape, dtype=torch.bool),
        ground_clear_to_target=torch.zeros(ground_shape, dtype=torch.bool),
    )
    raster = SoftObservableCameraRayRasterV4(
        source_free_probability=torch.zeros(
            (batch, labels.shape[-2], labels.shape[-1])
        ),
        free_given_not_occupied_probability=torch.zeros(
            (batch, labels.shape[-2], labels.shape[-1])
        ),
        occupied_probability=torch.zeros(
            (batch, labels.shape[-2], labels.shape[-1])
        ),
        class_probabilities=probabilities,
    )
    accumulator = ObservableCameraRayFitV4MetricAccumulator()
    accumulator.update(
        raw_output=raw,
        targets=targets,
        soft_raster=raster,
        target_raster_labels=labels,
        families=tuple(f"family_{index}" for index in range(batch)),
    )
    return accumulator.finalize()["derived_raster"]


def test_v12_batch_five_five_by_one_and_retained_accumulator_parity() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260710)
    probabilities = torch.softmax(
        torch.randn((5, 3, 4, 6), generator=generator, dtype=torch.float32),
        dim=1,
    )
    labels = torch.randint(0, 3, (5, 4, 6), generator=generator)
    batch_five = derived_raster_cell_nll_v12(probabilities, labels)
    five_by_one = torch.stack(
        [
            derived_raster_cell_nll_v12(
                probabilities[index : index + 1], labels[index : index + 1]
            )
            for index in range(5)
        ]
    ).mean()
    assert torch.isclose(batch_five, five_by_one, rtol=0.0, atol=2e-7)
    metric = _metric_accumulator_raster_nll(probabilities, labels)
    assert metric["count"] == labels.numel()
    assert float(metric["nll_sum"]) / int(metric["count"]) == pytest.approx(
        batch_five.item(), abs=2e-7
    )


def test_v12_objective_keeps_all_four_retained_coefficients_and_adds_one() -> None:
    values = {
        "hierarchical_first_hit_nll": torch.tensor(0.8),
        "target_bin_offset_smooth_l1": torch.tensor(0.02),
        "ground_clear_distance_state_balanced_bce": torch.tensor(0.04),
        "derived_raster_hierarchical_bce": torch.tensor(0.2),
    }
    cell_nll = torch.tensor(0.1)
    snapshot = {key: value.clone() for key, value in values.items()}
    objective = compose_gate_aligned_objective_v12(values, cell_nll)
    assert objective.v11_base_total.item() == pytest.approx(0.265)
    assert objective.total.item() == pytest.approx(0.29)
    assert all(torch.equal(values[key], snapshot[key]) for key in values)
    with pytest.raises(ValueError, match="order or fields"):
        compose_gate_aligned_objective_v12(dict(reversed(tuple(values.items()))), cell_nll)


def test_v12_rouf_frozen_counts_and_missing_class_cases() -> None:
    labels = torch.cat(
        (
            torch.zeros(16123, dtype=torch.long),
            torch.ones(4259, dtype=torch.long),
            torch.full((98,), 2, dtype=torch.long),
        )
    ).reshape(1, 128, 160)
    probabilities = torch.tensor([0.72, 0.23, 0.05], dtype=torch.float32).view(
        1, 3, 1, 1
    ).expand(1, 3, 128, 160).contiguous()
    decomposition = branch_reduction_decomposition_v12(probabilities, labels)
    assert decomposition["R"]["count"] == 20382
    assert decomposition["O"]["count"] == 98
    assert decomposition["U"]["count"] == 16123
    assert decomposition["F"]["count"] == 4259
    assert decomposition["cell_micro_mean"] == pytest.approx(
        derived_raster_cell_nll_v12(probabilities, labels).item(), abs=2e-6
    )
    missing = raster_nll_diagnostics_v12(
        torch.tensor(
            [[[[0.9, 0.8]], [[0.08, 0.15]], [[0.02, 0.05]]]],
            dtype=torch.float32,
        ),
        torch.zeros((1, 1, 2), dtype=torch.long),
        ("only_family",),
    )
    assert missing["by_target_class"]["FREE"] == {
        "count": 0,
        "nll_sum": 0.0,
        "mean": None,
    }
    assert missing["by_target_class"]["OCCUPIED"]["mean"] is None


def test_v12_diagnostics_partition_merge_nonmutation_and_rejection() -> None:
    probabilities = torch.tensor(
        [
            [[[0.8, 0.1]], [[0.15, 0.7]], [[0.05, 0.2]]],
            [[[0.2, 0.2]], [[0.3, 0.2]], [[0.5, 0.6]]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1]], [[2, 2]]], dtype=torch.long)
    before_probabilities = probabilities.clone()
    before_labels = labels.clone()
    whole = raster_nll_diagnostics_v12(
        probabilities, labels, ("alpha", "beta")
    )
    rows = [
        raster_nll_diagnostics_v12(
            probabilities[index : index + 1],
            labels[index : index + 1],
            (("alpha", "beta")[index],),
        )
        for index in range(2)
    ]
    merged = merge_raster_nll_diagnostics_v12(rows)
    assert merged == whole
    assert torch.equal(probabilities, before_probabilities)
    assert torch.equal(labels, before_labels)
    validate_raster_nll_diagnostics_v12(whole)
    for mutation in ("boolean_count", "nan_sum", "bad_mean", "bad_partition"):
        changed = copy.deepcopy(whole)
        if mutation == "boolean_count":
            changed["overall"]["count"] = True
        elif mutation == "nan_sum":
            changed["by_family"]["alpha"]["nll_sum"] = float("nan")
        elif mutation == "bad_mean":
            changed["by_target_class"]["UNKNOWN"]["mean"] = 99.0
        else:
            changed["by_family"]["alpha"]["count"] -= 1
        with pytest.raises(ValueError):
            validate_raster_nll_diagnostics_v12(changed)


def test_v12_native_record_adapter_and_actual_26_check_gate() -> None:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12 as policy,
    )
    from lewm.tests.n5_gate_aligned_raster_nll_v12_synthetic_execution import (
        complete_retained_gate_evaluation_v12,
    )

    evaluation = complete_retained_gate_evaluation_v12()
    before = copy.deepcopy(evaluation)
    policy.validate_evaluation_structure(evaluation)
    adapted = policy.adapt_native_v12_evaluation_for_retained_v4_gate(evaluation)
    assert evaluation == before
    assert "native_v12_objective" not in adapted["matched_rgb"]
    assert "raster_nll_diagnostics" not in adapted["matched_rgb"]
    assert adapted["matched_rgb"]["losses"]["ordered_first_hit_nll"] == 0.8
    assert "hierarchical_first_hit_nll" not in adapted["matched_rgb"]["losses"]
    _matched, _wrong, _signature, numeric = policy.reconstruct_retained_v4_gate(
        evaluation, fit_size=5
    )
    assert numeric["check_count"] == 26
    assert len(numeric["checks"]) == 26

    high_accuracy_failed_nll = copy.deepcopy(evaluation)
    row = high_accuracy_failed_nll["matched_rgb"]
    count = row["metrics"]["derived_raster"]["count"]
    row["metrics"]["derived_raster"]["nll"] = 0.07
    row["metrics"]["derived_raster"]["nll_sum"] = 0.07 * count
    row["native_v12_objective"]["derived_raster_cell_nll"] = 0.07
    row["native_v12_objective"]["total"] = (
        row["native_v12_objective"]["v11_base_total"] + 0.25 * 0.07
    )
    row["raster_nll_diagnostics"]["overall"].update(
        {"nll_sum": 0.07 * count, "mean": 0.07}
    )
    for partition in ("by_target_class", "by_family"):
        for diagnostic in row["raster_nll_diagnostics"][partition].values():
            diagnostic["nll_sum"] = 0.07 * diagnostic["count"]
            diagnostic["mean"] = None if diagnostic["count"] == 0 else 0.07
    policy.validate_evaluation_structure(high_accuracy_failed_nll)
    _m, _w, _s, failed = policy.reconstruct_retained_v4_gate(
        high_accuracy_failed_nll, fit_size=5
    )
    assert failed["passes"] is False
    assert [check["name"] for check in failed["failed_checks"]] == [
        "matched.raster_nll"
    ]


def test_v12_verifier_recomputes_g_without_trainer_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import (
        train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12
        as trainer,
    )
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12
        as verifier,
    )

    probabilities = torch.tensor(
        [[[[0.6]], [[0.3]], [[0.1]]]], dtype=torch.float32
    )
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    expected = -math.log(0.6)
    monkeypatch.setattr(
        trainer,
        "compute_gate_aligned_v12_losses",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trainer scalar reused")
        ),
    )
    actual = verifier.derived_raster_cell_nll_for_verification_v12(
        probabilities, labels
    )
    assert actual.item() == pytest.approx(expected)


def test_v12_sources_forbid_v11_checkpoint_input_and_exact_root_is_absent() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = (
        root
        / "scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py",
        root
        / "scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py",
        root
        / "scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v12.py",
    )
    forbidden = ".generated/go2_observable_camera_ray_fit_v4/n5_hierarchical_first_hit_v11/attempts"
    assert all(forbidden not in path.read_text(encoding="utf-8") for path in sources)
    output = (
        root
        / ".generated/go2_observable_camera_ray_fit_v4/n5_gate_aligned_raster_nll_v12"
    )
    assert not output.exists()
