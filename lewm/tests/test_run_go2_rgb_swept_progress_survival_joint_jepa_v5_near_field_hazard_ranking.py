from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from lewm.tests import (
    test_run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux
    as v2_fixtures,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking.py"
)


def _load_runner() -> Any:
    name = "_test_go2_swept_progress_survival_v5_hazard_ranking_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _near_indices() -> list[tuple[int, int]]:
    return [
        tuple(map(int, index))
        for index in runner._near_field_mask_v5(torch, device=torch.device("cpu"))
        .nonzero()
        .tolist()
    ]


def _empty_view(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((batch, 3, 64, 64), dtype=torch.float64),
        torch.zeros((batch, 64, 64), dtype=torch.long),
    )


def _set_hazard_score(logits: torch.Tensor, row: int, cell: tuple[int, int], score: float) -> None:
    # unknown=free=0 makes occupied=score+log(2) produce the requested
    # occupied-vs-rest log odds exactly up to floating arithmetic.
    logits[row, 2, cell[0], cell[1]] = score + math.log(2.0)


def _eligible_view(score_difference: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits, labels = _empty_view()
    occupied, free = _near_indices()[:2]
    labels[0, occupied[0], occupied[1]] = runner.OCCUPIED_CLASS_INDEX
    labels[0, free[0], free[1]] = runner.FREE_CLASS_INDEX
    _set_hazard_score(logits, 0, occupied, 0.0)
    _set_hazard_score(logits, 0, free, score_difference)
    return logits, labels


def test_fixed_near_field_grid_and_inherited_science_constants() -> None:
    mask = runner._near_field_mask_v5(torch, device=torch.device("cpu"))
    assert runner.RASTER_SIZE == runner.RASTER_SIDE == 64
    assert runner.FORWARD_MIN_M == -0.95
    assert runner.FORWARD_MAX_M == 5.35
    assert runner.LEFT_MIN_M == -3.15
    assert runner.LEFT_MAX_M == 3.15
    assert runner.NEAR_FIELD_RANGE_M == 2.0
    assert runner.NEAR_FIELD_CELL_COUNT == 1_016
    assert tuple(mask.shape) == (64, 64)
    assert mask.dtype == torch.bool
    assert int(mask.sum()) == 1_016
    assert runner.HAZARD_RANKING_COEFFICIENT == 1.0
    assert runner.HAZARD_RANKING_NORMALIZATION == math.log(2.0)
    assert runner.OCCUPIED_SAFETY_AUX_COEFFICIENT == 0.5
    assert runner.OCCUPIED_SAFETY_AUX_NORMALIZATION == math.log(2.0)
    assert runner.OCCUPIED_SAFETY_AUX_COEFFICIENT is runner._v3.OCCUPIED_SAFETY_AUX_COEFFICIENT


def test_equal_hazard_scores_have_exact_unit_normalized_pair_loss() -> None:
    current, current_labels = _empty_view(batch=2)
    next_logits, next_labels = _empty_view(batch=2)
    cells = _near_indices()[:7]
    current_labels[0, cells[0][0], cells[0][1]] = runner.OCCUPIED_CLASS_INDEX
    for cell in cells[1:4]:
        current_labels[0, cell[0], cell[1]] = runner.FREE_CLASS_INDEX
    for cell in cells[4:6]:
        current_labels[1, cell[0], cell[1]] = runner.OCCUPIED_CLASS_INDEX
    current_labels[1, cells[6][0], cells[6][1]] = runner.FREE_CLASS_INDEX

    observed = runner.near_field_hazard_ranking_loss_v5(
        current, current_labels, next_logits, next_labels
    )

    assert torch.equal(
        observed.current_per_eligible_row,
        torch.ones((2,), dtype=torch.float64),
    )
    assert observed.next_per_eligible_row.numel() == 0
    assert observed.loss.item() == 1.0
    assert observed.current_eligible_row_count == 2
    assert observed.current_ranked_pair_count == 5
    assert observed.active is True


def test_h_is_mean_of_complete_pair_means_per_sample_not_pooled_pairs() -> None:
    logits, labels = _empty_view(batch=2)
    next_logits, next_labels = _empty_view(batch=2)
    cells = _near_indices()[:6]
    # Sample zero contributes one easy pair.
    labels[0, cells[0][0], cells[0][1]] = runner.OCCUPIED_CLASS_INDEX
    labels[0, cells[1][0], cells[1][1]] = runner.FREE_CLASS_INDEX
    _set_hazard_score(logits, 0, cells[0], 2.0)
    _set_hazard_score(logits, 0, cells[1], 0.0)
    # Sample one contributes three difficult pairs, but still one sample mean.
    labels[1, cells[2][0], cells[2][1]] = runner.OCCUPIED_CLASS_INDEX
    _set_hazard_score(logits, 1, cells[2], -1.0)
    for cell in cells[3:6]:
        labels[1, cell[0], cell[1]] = runner.FREE_CLASS_INDEX
        _set_hazard_score(logits, 1, cell, 1.0)

    observed = runner.near_field_hazard_ranking_loss_v5(
        logits, labels, next_logits, next_labels
    )
    easy = F.softplus(torch.tensor(-2.0, dtype=torch.float64)) / math.log(2.0)
    difficult = F.softplus(torch.tensor(2.0, dtype=torch.float64)) / math.log(2.0)
    sample_mean = 0.5 * (easy + difficult)
    pooled_pair_mean = (easy + 3.0 * difficult) / 4.0

    assert torch.allclose(observed.loss, sample_mean, rtol=1e-14, atol=1e-14)
    assert not torch.allclose(observed.loss, pooled_pair_mean, rtol=1e-3, atol=1e-3)
    assert observed.current_eligible_row_count == 2
    assert observed.current_ranked_pair_count == 4


def test_current_only_next_only_and_both_view_means_follow_exact_rule() -> None:
    current, current_labels = _eligible_view(-1.5)
    next_logits, next_labels = _eligible_view(0.75)
    empty_logits, empty_labels = _empty_view()
    current_value = F.softplus(torch.tensor(-1.5, dtype=torch.float64)) / math.log(2.0)
    next_value = F.softplus(torch.tensor(0.75, dtype=torch.float64)) / math.log(2.0)

    current_only = runner.near_field_hazard_ranking_loss_v5(
        current, current_labels, empty_logits, empty_labels
    )
    next_only = runner.near_field_hazard_ranking_loss_v5(
        empty_logits, empty_labels, next_logits, next_labels
    )
    both = runner.near_field_hazard_ranking_loss_v5(
        current, current_labels, next_logits, next_labels
    )

    assert torch.allclose(current_only.loss, current_value, rtol=1e-14, atol=1e-14)
    assert torch.allclose(next_only.loss, next_value, rtol=1e-14, atol=1e-14)
    assert torch.allclose(
        both.loss, 0.5 * (current_value + next_value), rtol=1e-14, atol=1e-14
    )
    assert (current_only.current_eligible_row_count, current_only.next_eligible_row_count) == (1, 0)
    assert (next_only.current_eligible_row_count, next_only.next_eligible_row_count) == (0, 1)
    assert (both.current_ranked_pair_count, both.next_ranked_pair_count) == (1, 1)


def test_inactive_h_is_exact_zero_graph_connected_to_both_views() -> None:
    current, current_labels = _empty_view()
    next_logits, next_labels = _empty_view()
    current.requires_grad_()
    next_logits.requires_grad_()

    observed = runner.near_field_hazard_ranking_loss_v5(
        current, current_labels, next_logits, next_labels
    )
    observed.loss.backward()

    assert observed.loss.item() == 0.0
    assert observed.active is False
    assert observed.current_eligible_row_count == observed.next_eligible_row_count == 0
    assert observed.current_ranked_pair_count == observed.next_ranked_pair_count == 0
    assert current.grad is not None and torch.count_nonzero(current.grad) == 0
    assert next_logits.grad is not None and torch.count_nonzero(next_logits.grad) == 0


def test_h_gradients_reach_all_semantic_logits_without_common_shift_effect() -> None:
    current, labels = _eligible_view(0.4)
    next_logits, next_labels = _eligible_view(-0.2)
    current.requires_grad_()
    next_logits.requires_grad_()
    base = runner.near_field_hazard_ranking_loss_v5(
        current, labels, next_logits, next_labels
    ).loss
    shifted = runner.near_field_hazard_ranking_loss_v5(
        current + 13.0, labels, next_logits - 7.0, next_labels
    ).loss
    assert torch.allclose(base, shifted, rtol=1e-13, atol=1e-13)

    base.backward()
    assert current.grad is not None and next_logits.grad is not None
    assert bool((current.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
    assert bool((next_logits.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
    assert torch.allclose(
        current.grad.sum(dim=1), torch.zeros_like(current.grad[:, 0]), atol=1e-14
    )


class _V5TinyJointModel(v2_fixtures._TinyJointModel):
    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        coarse = super().semantic_logits_from_latent(latent)
        return F.interpolate(coarse, size=(64, 64), mode="nearest")


def _joint_microbatches() -> list[dict[str, torch.Tensor]]:
    batches = v2_fixtures._microbatches()
    cells = _near_indices()[:4]
    for batch in batches:
        current = torch.zeros((4, 64, 64), dtype=torch.long)
        next_labels = torch.zeros_like(current)
        for row in range(4):
            current[row, cells[0][0], cells[0][1]] = runner.FREE_CLASS_INDEX
            current[row, cells[1][0], cells[1][1]] = runner.OCCUPIED_CLASS_INDEX
            next_labels[row, cells[2][0], cells[2][1]] = runner.FREE_CLASS_INDEX
            next_labels[row, cells[3][0], cells[3][1]] = runner.OCCUPIED_CLASS_INDEX
        batch[runner.CURRENT_LABELS_KEY] = current
        batch[runner.NEXT_LABELS_KEY] = next_labels
    return batches


def test_joint_update_adds_h_to_v3_loss_without_changing_r_accounting() -> None:
    torch.manual_seed(91)
    model_v3 = _V5TinyJointModel()
    torch.manual_seed(91)
    model_v5 = _V5TinyJointModel()
    model_v5.load_state_dict(model_v3.state_dict())
    optimizer_v3 = runner.build_frozen_optimizer_v1(
        runner.partition_parameters_v1(model_v3)
    )
    optimizer_v5 = runner.build_frozen_optimizer_v1(
        runner.partition_parameters_v1(model_v5)
    )
    batches = _joint_microbatches()

    result_v3 = runner._v3.joint_training_update_v3(
        model_v3, optimizer_v3, batches
    )
    result_v5 = runner.joint_training_update_v5(model_v5, optimizer_v5, batches)

    assert result_v5.accounting == result_v3.accounting
    assert set(result_v5.mean_losses) == {"S", "P", "U", "R", "O", "H", "L"}
    for name in ("S", "P", "U", "R", "O"):
        assert result_v5.mean_losses[name] == result_v3.mean_losses[name]
    assert runner._v3.OCCUPIED_SAFETY_AUX_COEFFICIENT == 0.5
    assert result_v5.mean_losses["H"] > 0.0
    assert math.isclose(
        result_v5.mean_losses["L"],
        result_v3.mean_losses["L"] + result_v5.mean_losses["H"],
        rel_tol=2e-6,
        abs_tol=2e-6,
    )
    assert result_v5.ranking_active_microbatches == result_v3.ranking_active_microbatches
    assert result_v5.ranking_eligible_pairs == result_v3.ranking_eligible_pairs
    assert result_v5.survival_supervised_decisions == result_v3.survival_supervised_decisions
    assert len(result_v5.hazard_ranking_microbatches) == 4
    assert all(row["hazard_active"] for row in result_v5.hazard_ranking_microbatches)
    assert model_v5.semantic_head.weight.grad is not None
    assert not torch.equal(
        model_v5.semantic_head.weight.grad, model_v3.semantic_head.weight.grad
    )


def test_fixed_driver_records_microbatch_update_window_and_aggregate_h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = runner.freeze_role_labels_v1(
        v2_fixtures._label_rows(), role="train", np=__import__("numpy")
    )
    pair = {
        "dataset_role": "train",
        "content_sha256": "a" * 64,
        "current_endpoint_sha256": "b" * 64,
        "next_endpoint_sha256": "c" * 64,
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
        "primitive": "arc_left",
    }
    built = updates = 0

    def fake_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal built
        del args, kwargs
        built += 1
        return {}

    def fake_update(
        model: Any,
        optimizer: Any,
        microbatches: Any,
        *,
        accounting: runner.JointTrainingAccountingV1,
    ) -> runner.JointUpdateResultV5:
        nonlocal updates
        del model, optimizer
        updates += 1
        assert len(microbatches) == 4
        losses = {name: 1.0 for name in ("S", "P", "U", "R", "O")}
        losses.update(H=float(updates), L=5.0 + float(updates))
        receipts = tuple(
            {
                "H": float(updates),
                "hazard_active": True,
                "hazard_current_eligible_row_count": 2,
                "hazard_next_eligible_row_count": 3,
                "hazard_current_ranked_pair_count": 5,
                "hazard_next_ranked_pair_count": 7,
            }
            for _ in range(4)
        )
        return runner.JointUpdateResultV5(
            accounting=runner._v3._v2._v1._base._advanced_accounting(accounting),
            mean_losses=losses,
            gradient_l2={
                name: 1.0 for name in ("encoder", "lift_semantic", "predictor")
            },
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_pairs=3,
            survival_supervised_decisions=4,
            hazard_ranking_microbatches=receipts,
        )

    monkeypatch.setattr(runner, "build_microbatch_v1", fake_build)
    monkeypatch.setattr(runner, "joint_training_update_v5", fake_update)
    accounting, trace, diagnostics = runner.run_fixed_training_v5(
        object(), object(), object(), (pair,), labels, (0,) * 16_000, object()
    )

    assert accounting.updates == updates == 1_000
    assert accounting.presentations == 16_000
    assert built == 4_000
    assert len(trace) == 1_000
    assert trace[0]["losses"]["H"] == 1.0
    assert trace[-1]["losses"]["H"] == 1_000.0
    first_activity = trace[0]["hazard_ranking_activity"]
    assert len(first_activity["microbatches"]) == 4
    assert first_activity["hazard_active_microbatch_count"] == 4
    activity = diagnostics["hazard_ranking_activity"]
    assert activity["hazard_microbatch_count"] == 4_000
    assert activity["hazard_active_microbatch_count"] == 4_000
    assert activity["hazard_current_eligible_row_count"] == 8_000
    assert activity["hazard_next_eligible_row_count"] == 12_000
    assert activity["hazard_current_ranked_pair_count"] == 20_000
    assert activity["hazard_next_ranked_pair_count"] == 28_000
    assert activity["hazard_ranked_pair_count"] == 48_000
    windows = activity["hazard_windows_100_updates"]
    assert len(windows) == 10
    assert windows[0]["first_update"] == 1
    assert windows[0]["last_update"] == 100
    assert windows[0]["hazard_mean_microbatch_H"] == 50.5
    assert windows[-1]["hazard_mean_microbatch_H"] == 950.5
    # Existing JEPA progress-ranking receipts remain a distinct unchanged path.
    assert diagnostics["ranking_active_microbatch_count"] == 2_000
    assert diagnostics["ranking_eligible_pair_count"] == 3_000


@pytest.mark.parametrize(
    ("logits", "labels", "error"),
    [
        (torch.zeros((1, 3, 63, 64)), torch.zeros((1, 63, 64), dtype=torch.long), ValueError),
        (torch.zeros((1, 3, 64, 64)), torch.zeros((1, 64, 64)), TypeError),
        (
            torch.zeros((1, 3, 64, 64)),
            torch.full((1, 64, 64), 3, dtype=torch.long),
            ValueError,
        ),
    ],
)
def test_h_rejects_nonfrozen_semantic_contract(
    logits: torch.Tensor, labels: torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        runner.near_field_hazard_ranking_loss_v5(logits, labels, logits, labels)
