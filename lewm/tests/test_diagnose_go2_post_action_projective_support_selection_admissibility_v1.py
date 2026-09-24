from __future__ import annotations

from scripts import (
    diagnose_go2_post_action_projective_support_selection_admissibility_v1
    as diagnostic,
)


def _state_rows(
    state_index: int,
    *,
    family: str,
    prefixes: dict[str, int],
    inadmissible: set[str],
) -> list[dict[str, object]]:
    non_hold_prefixes = {
        action: prefixes[action] for action in diagnostic.NON_HOLD_ACTIONS
    }
    primary = not inadmissible
    best = max(non_hold_prefixes.values())
    distinct = len(set(non_hold_prefixes.values())) >= 2
    informative = primary and best > 0 and distinct
    eligible_pairs = sum(
        int(first > second)
        for first in non_hold_prefixes.values()
        for second in non_hold_prefixes.values()
    )
    return [
        {
            "dataset_role": diagnostic.SELECTION_ROLE,
            "role_state_index": state_index,
            "family": family,
            "scene_id": f"{family}_scene",
            "action_index": action_index,
            "action": action,
            "immediate_primitive": {"feasible": action not in inadmissible},
            "blind_bridge": {"feasible": True},
            "remote_safe_prefix_length": prefixes.get(action, 0),
            "primary_subset_eligible": primary,
            "oracle_remote_safe_prefix_length": best,
            "eligible_ordered_ranking_pair_count": eligible_pairs,
            "informative_state": informative,
        }
        for action_index, action in enumerate(diagnostic.labels.ACTION_ORDER)
    ]


def test_aggregation_zeroes_inadmissible_prefixes_and_recovers_proposed_counts() -> None:
    actions = diagnostic.NON_HOLD_ACTIONS
    rows = [
        *_state_rows(
            0,
            family="family_a",
            prefixes={action: (2 if index == 0 else 1) for index, action in enumerate(actions)},
            inadmissible=set(),
        ),
        *_state_rows(
            1,
            family="family_a",
            prefixes={action: (3 if index == 0 else 2 if index == 1 else 0) for index, action in enumerate(actions)},
            inadmissible={actions[0]},
        ),
        *_state_rows(
            2,
            family="family_b",
            prefixes={action: (4 if index == 0 else 0) for index, action in enumerate(actions)},
            inadmissible={actions[0]},
        ),
    ]
    census = diagnostic.aggregate_selection_rows_v1(
        rows, families=("family_a", "family_b")
    )

    family_a = census["families"]["family_a"]
    assert family_a["original_conjunct_counts"] == {
        "primary_subset_eligible": 1,
        "positive_best_remote_safe_prefix": 2,
        "at_least_two_distinct_nonhold_remote_safe_prefixes": 2,
        "informative_state": 1,
    }
    assert family_a["proposed_conjunct_counts"] == {
        "positive_best_admissible_prefix": 2,
        "at_least_two_distinct_nonhold_admissible_prefixes": 2,
        "informative_state": 2,
    }
    assert family_a["informative_transition_counts"] == {
        "original_and_proposed": 1,
        "proposed_only": 1,
        "original_only": 0,
        "neither": 0,
    }
    family_b = census["families"]["family_b"]
    assert family_b["original_conjunct_counts"]["informative_state"] == 0
    assert family_b["proposed_conjunct_counts"]["informative_state"] == 0

    aggregate = census["aggregate"]
    first_action = aggregate["actions"][actions[0]]
    assert first_action["admissible_count"] == 1
    assert first_action["positive_admissible_prefix_count"] == 1
    assert first_action["admissible_prefix_histogram_0_through_11"] == [
        2,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    second_action = aggregate["actions"][actions[1]]
    assert second_action["admissible_count"] == 3
    assert second_action["positive_admissible_prefix_count"] == 2
