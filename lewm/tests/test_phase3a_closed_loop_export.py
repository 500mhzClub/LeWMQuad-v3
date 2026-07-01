import torch

from lewm.models.phase3a_latent_map import (
    Phase3AEgocentricMemoryPolicy,
    Phase3AEgocentricMemoryUpdate,
    Phase3AEgocentricValueFieldHead,
    Phase3AValueFieldActionHead,
    Phase3AValueFieldExtractorHead,
    Phase3AValueMapPlannerHead,
    Phase3AValueMapRouterHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (
    _egocentric_has_contiguous_side_wall,
    _roll_egocentric_cell,
    _roll_egocentric_frontier_memory,
    _select_egocentric_learned_value_field_action,
    _select_egocentric_learned_value_map_action,
    _select_egocentric_frontier_action,
    _select_egocentric_value_field_action,
)


def test_egocentric_roll_conventions_match_marker_memory_actions() -> None:
    assert _roll_egocentric_cell((2, 1), "forward", collision=False) == (1, 1)
    assert _roll_egocentric_cell((2, 1), "forward", collision=True) == (2, 1)
    assert _roll_egocentric_cell((0, 1), "turn_left", collision=False) == (1, 0)
    assert _roll_egocentric_cell((0, -1), "turn_right", collision=False) == (1, 0)


def test_egocentric_marker_selection_uses_left_right_coordinates() -> None:
    memory = {
        "free": {(0, 0), (0, 1)},
        "blocked": set(),
        "marker": (0, 1),
        "radius": 5,
    }
    assert _select_egocentric_frontier_action(memory) == "turn_left"

    memory = {
        "free": {(0, 0), (0, -1)},
        "blocked": set(),
        "marker": (0, -1),
        "radius": 5,
    }
    assert _select_egocentric_frontier_action(memory) == "turn_right"


def test_egocentric_side_wall_gate_requires_contiguous_lateral_blockers() -> None:
    assert _egocentric_has_contiguous_side_wall(
        {
            "free": {(0, 0), (1, 0)},
            "blocked": {(0, 1), (0, 2), (0, 3), (1, -1)},
            "marker": None,
            "radius": 5,
        }
    )
    assert not _egocentric_has_contiguous_side_wall(
        {
            "free": {(0, 0), (1, 0)},
            "blocked": {(0, 1), (0, 3), (1, -1), (-1, 1)},
            "marker": None,
            "radius": 5,
        }
    )


def test_egocentric_frontier_memory_rolls_and_records_collision() -> None:
    memory = {
        "free": {(0, 0), (1, 0), (2, 0)},
        "blocked": set(),
        "marker": (2, 0),
        "radius": 5,
    }
    _roll_egocentric_frontier_memory(memory, "forward", collision=False)
    assert memory["marker"] == (1, 0)
    assert (0, 0) in memory["free"]
    assert (1, 0) in memory["free"]

    _roll_egocentric_frontier_memory(memory, "forward", collision=True)
    assert memory["marker"] == (1, 0)
    assert (1, 0) in memory["blocked"]
    assert (1, 0) not in memory["free"]


def test_recurrent_memory_updater_forward_shape() -> None:
    updater = Phase3AEgocentricMemoryUpdate(memory_size=7, hidden_dim=8)
    previous = torch.zeros(2, 3, 7, 7)
    evidence = torch.zeros(2, 3, 7, 7)
    actions = torch.tensor([0, 1])
    collisions = torch.tensor([0.0, 0.0])

    logits = updater(previous, evidence, actions, collisions)

    assert logits.shape == (2, 3, 7, 7)


def test_no_prior_recurrent_memory_updater_forward_shape() -> None:
    updater = Phase3AEgocentricMemoryUpdate(
        memory_size=7,
        hidden_dim=8,
        use_geometric_prior=False,
        learned_transition_hidden_dim=16,
    )
    previous = torch.zeros(2, 3, 7, 7)
    evidence = torch.zeros(2, 3, 7, 7)
    actions = torch.tensor([0, 1])
    collisions = torch.tensor([0.0, 1.0])

    logits = updater(previous, evidence, actions, collisions)

    assert logits.shape == (2, 3, 7, 7)


def test_no_prior_direct_transition_updater_forward_shape() -> None:
    updater = Phase3AEgocentricMemoryUpdate(
        memory_size=7,
        hidden_dim=8,
        use_geometric_prior=False,
        learned_transition_hidden_dim=0,
    )
    previous = torch.zeros(2, 3, 7, 7)
    evidence = torch.zeros(2, 3, 7, 7)
    actions = torch.tensor([0, 1])
    collisions = torch.tensor([0.0, 1.0])

    logits = updater(previous, evidence, actions, collisions)

    assert logits.shape == (2, 3, 7, 7)


def test_recurrent_memory_policy_forward_shape() -> None:
    policy = Phase3AEgocentricMemoryPolicy(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)

    logits = policy(memory)

    assert logits.shape == (2, 4)


def test_recurrent_memory_conv_policy_forward_shape() -> None:
    policy = Phase3AEgocentricMemoryPolicy(
        memory_size=7,
        hidden_dim=16,
        architecture="conv",
    )
    memory = torch.zeros(2, 3, 7, 7)

    logits = policy(memory)

    assert logits.shape == (2, 4)


def test_recurrent_value_field_head_forward_shape() -> None:
    head = Phase3AEgocentricValueFieldHead(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)

    logits = head(memory)

    assert logits.shape == (2, 1, 7, 7)


def test_value_field_extractor_head_forward_shape() -> None:
    head = Phase3AValueFieldExtractorHead(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)

    logits = head(memory)

    assert logits.shape == (2,)


def test_value_field_action_head_forward_shape() -> None:
    head = Phase3AValueFieldActionHead(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)
    target = torch.zeros(2, 1, 7, 7)
    sparse = torch.tensor([0.0, 1.0])

    logits = head(memory, target, sparse)

    assert logits.shape == (2, 4)


def test_value_map_planner_head_forward_shape() -> None:
    head = Phase3AValueMapPlannerHead(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)
    target = torch.zeros(2, 1, 7, 7)
    sparse = torch.tensor([0.0, 1.0])

    logits = head(memory, target, sparse)

    assert logits.shape == (2, 1, 7, 7)


def test_value_map_router_head_forward_shape() -> None:
    head = Phase3AValueMapRouterHead(memory_size=7, hidden_dim=16)
    memory = torch.zeros(2, 3, 7, 7)

    logits = head(memory)

    assert logits.shape == (2,)


def test_value_map_planner_head_architecture_variants_forward_shape() -> None:
    memory = torch.zeros(2, 3, 7, 7)
    target = torch.zeros(2, 1, 7, 7)
    sparse = torch.tensor([0.0, 1.0])
    for architecture in ("dilated", "recurrent"):
        head = Phase3AValueMapPlannerHead(
            memory_size=7,
            hidden_dim=16,
            architecture=architecture,
            refinement_steps=2,
        )

        logits = head(memory, target, sparse)

        assert logits.shape == (2, 1, 7, 7)


def test_egocentric_value_field_routes_to_marker() -> None:
    memory = {
        "free": {(0, 0), (1, 0), (2, 0)},
        "blocked": set(),
        "marker": (2, 0),
        "radius": 5,
    }

    action, mode = _select_egocentric_value_field_action(memory)

    assert action == "forward"
    assert mode == "latent_recurrent_value_marker"


def test_egocentric_value_field_routes_to_frontier() -> None:
    memory = {
        "free": {(0, 0), (0, 1)},
        "blocked": {(1, 0), (0, -1), (-1, 0)},
        "marker": None,
        "radius": 5,
    }

    action, mode = _select_egocentric_value_field_action(memory)

    assert action == "turn_left"
    assert mode == "latent_recurrent_value_frontier"


def test_egocentric_learned_value_field_routes_to_target() -> None:
    memory = {
        "free": {(0, 0), (0, -1)},
        "blocked": {(1, 0), (0, 1), (-1, 0)},
        "marker": None,
        "radius": 3,
    }
    target_probs = torch.zeros(7, 7)
    radius = 3
    target_probs[radius, radius - 1] = 0.9

    action, mode = _select_egocentric_learned_value_field_action(
        memory,
        target_probs,
        threshold=0.5,
        top_k=4,
    )

    assert action == "turn_right"
    assert mode == "latent_recurrent_learned_value_field"


def test_egocentric_learned_value_map_routes_to_local_maximum() -> None:
    memory = {
        "free": {(0, 0), (0, -1)},
        "blocked": {(1, 0), (0, 1), (-1, 0)},
        "marker": None,
        "radius": 3,
    }
    value_probs = torch.zeros(7, 7)
    radius = 3
    value_probs[radius, radius - 1] = 0.9

    action, mode = _select_egocentric_learned_value_map_action(memory, value_probs)

    assert action == "turn_right"
    assert mode == "latent_recurrent_learned_value_map"
