import pytest
import torch

from lewm.benchmarks.phase3b_reachability import (
    build_pairwise_reachability_target,
    build_reachability_target,
    cell_to_row_col,
    egocentric_frontier_cells,
    reachability_prediction_losses,
    select_reachability_target_cells,
    stack_reachability_targets,
)
from lewm.models.phase3b_reachability import (
    Phase3BReachabilityConditionedValueMapPlannerHead,
    Phase3BReachabilityHead,
    reachability_feature_tensor,
)


def _value_at(
    tensor: torch.Tensor,
    cell: tuple[int, int],
    *,
    memory_size: int,
) -> float:
    row_col = cell_to_row_col(cell, memory_size=memory_size)
    assert row_col is not None
    return float(tensor[0, row_col[0], row_col[1]].item())


def _bool_at(tensor: torch.Tensor, cell: tuple[int, int], *, memory_size: int) -> bool:
    row_col = cell_to_row_col(cell, memory_size=memory_size)
    assert row_col is not None
    return bool(tensor[0, row_col[0], row_col[1]].item())


def test_reachability_target_routes_to_remembered_marker() -> None:
    memory = {
        "free": {(0, 0), (1, 0), (2, 0), (0, 1)},
        "blocked": {(1, 1)},
        "marker": (2, 0),
    }

    target = build_reachability_target(memory, memory_size=7, gamma=0.9)

    assert target.target_cells == ((2, 0),)
    assert _bool_at(target.reachable_mask, (2, 0), memory_size=7)
    assert _bool_at(target.target_mask, (2, 0), memory_size=7)
    assert _value_at(target.current_distance, (2, 0), memory_size=7) == 2.0
    assert _value_at(target.target_distance, (0, 0), memory_size=7) == 2.0
    assert _value_at(target.target_value, (0, 0), memory_size=7) == pytest.approx(
        0.9 ** 2
    )
    assert _value_at(target.target_value, (2, 0), memory_size=7) == pytest.approx(1.0)


def test_reachability_target_falls_back_to_frontier_without_marker() -> None:
    memory = {
        "free": {(0, 0), (1, 0)},
        "blocked": {(0, 1), (0, -1), (-1, 0)},
        "marker": None,
    }

    assert egocentric_frontier_cells(memory) == ((1, 0),)
    assert select_reachability_target_cells(memory) == ((1, 0),)

    target = build_reachability_target(memory, memory_size=7, gamma=0.5)

    assert target.target_cells == ((1, 0),)
    assert _bool_at(target.frontier_mask, (1, 0), memory_size=7)
    assert _value_at(target.target_distance, (0, 0), memory_size=7) == 1.0
    assert _value_at(target.target_value, (0, 0), memory_size=7) == pytest.approx(0.5)


def test_reachability_target_masks_unreachable_explicit_target() -> None:
    memory = {
        "free": {(0, 0), (0, 1)},
        "blocked": {(1, 0)},
        "marker": None,
    }

    target = build_reachability_target(
        memory,
        memory_size=7,
        target_cells=[(2, 0)],
        gamma=0.9,
    )

    assert target.target_cells == ((2, 0),)
    assert not _bool_at(target.reachable_mask, (2, 0), memory_size=7)
    assert _value_at(target.target_distance, (0, 0), memory_size=7) == 49.0
    assert _value_at(target.target_value, (0, 0), memory_size=7) == 0.0


def test_pairwise_reachability_target_records_shortest_paths() -> None:
    memory = {
        "free": {(0, 0), (1, 0), (2, 0), (0, 1)},
        "blocked": {(1, 1)},
        "marker": None,
    }

    target = build_pairwise_reachability_target(
        memory,
        memory_size=7,
        cells=[(0, 0), (2, 0), (0, 1)],
    )

    assert target.cells == ((0, 0), (2, 0), (0, 1))
    assert torch.equal(target.reachable_mask, torch.ones(3, 3, dtype=torch.bool))
    assert target.distances[0, 1].item() == 2.0
    assert target.distances[1, 2].item() == 3.0
    assert target.distances[2, 2].item() == 0.0


def test_reachability_head_and_losses_have_stable_shapes() -> None:
    memories = [
        {
            "free": {(0, 0), (1, 0), (2, 0)},
            "blocked": set(),
            "marker": (2, 0),
        },
        {
            "free": {(0, 0), (0, 1)},
            "blocked": {(1, 0)},
            "marker": None,
        },
    ]
    targets = stack_reachability_targets(
        [
            build_reachability_target(memory, memory_size=7, gamma=0.9)
            for memory in memories
        ]
    )
    head = Phase3BReachabilityHead(memory_size=7, hidden_dim=8)
    memory_tensor = torch.zeros(2, 3, 7, 7)

    predictions = head(memory_tensor)
    losses = reachability_prediction_losses(predictions, targets)

    assert predictions["reachable_logits"].shape == (2, 1, 7, 7)
    assert predictions["current_distance"].shape == (2, 1, 7, 7)
    assert predictions["target_distance"].shape == (2, 1, 7, 7)
    assert predictions["target_value_logits"].shape == (2, 1, 7, 7)
    assert targets.reachable_mask.shape == (2, 1, 7, 7)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["reachable_loss"])
    assert torch.isfinite(losses["current_distance_loss"])
    assert torch.isfinite(losses["target_distance_loss"])
    assert torch.isfinite(losses["target_value_loss"])


def test_reachability_head_dilated_variant_forward_shape() -> None:
    head = Phase3BReachabilityHead(
        memory_size=7,
        hidden_dim=8,
        architecture="dilated",
    )

    predictions = head(torch.zeros(2, 3, 7, 7))

    assert predictions["reachable_logits"].shape == (2, 1, 7, 7)


def test_reachability_feature_tensor_bounds_head_outputs() -> None:
    head = Phase3BReachabilityHead(memory_size=7, hidden_dim=8)
    predictions = head(torch.zeros(2, 3, 7, 7))

    features = reachability_feature_tensor(predictions, memory_size=7)

    assert features.shape == (2, 4, 7, 7)
    assert torch.isfinite(features).all()
    assert float(features.min().item()) >= 0.0
    assert float(features.max().item()) <= 1.0


def test_reachability_conditioned_planner_forward_shape() -> None:
    reachability_head = Phase3BReachabilityHead(memory_size=7, hidden_dim=8)
    planner = Phase3BReachabilityConditionedValueMapPlannerHead(
        memory_size=7,
        hidden_dim=8,
    )
    memory = torch.zeros(2, 3, 7, 7)
    target = torch.zeros(2, 1, 7, 7)
    sparse = torch.tensor([0.0, 1.0])
    reachability = reachability_feature_tensor(
        reachability_head(memory),
        memory_size=7,
    )

    logits = planner(memory, target, sparse, reachability)

    assert logits.shape == (2, 1, 7, 7)


def test_reachability_conditioned_planner_dilated_variant_forward_shape() -> None:
    planner = Phase3BReachabilityConditionedValueMapPlannerHead(
        memory_size=7,
        hidden_dim=8,
        architecture="dilated",
    )

    logits = planner(
        torch.zeros(2, 3, 7, 7),
        torch.zeros(2, 1, 7, 7),
        torch.tensor([0.0, 1.0]),
        torch.zeros(2, 4, 7, 7),
    )

    assert logits.shape == (2, 1, 7, 7)
