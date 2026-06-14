from __future__ import annotations

from scripts.build_jepa_counterfactual_render_plans import _selected_candidate_indices


def test_bounded_candidate_selection_has_exact_size_and_includes_oracle() -> None:
    row = {
        "counterfactual_candidates": [{} for _ in range(81)],
        "counterfactual_oracle_index": 37,
    }

    selected = _selected_candidate_indices(row, 9)

    assert len(selected) == 9
    assert 37 in selected
    assert selected == sorted(selected)


def test_outcome_stratified_selection_balances_candidate_classes() -> None:
    candidates = []
    for index in range(30):
        candidates.append(
            {
                "enters_grid_unsafe": 10 <= index < 20,
                "ends_grid_unsafe": False,
                "target_progress_m": 0.1 if index < 10 else -0.1,
                "target_recoverable": True,
            }
        )
    row = {
        "counterfactual_candidates": candidates,
        "counterfactual_oracle_index": 4,
    }

    selected = _selected_candidate_indices(row, 9, "outcome_stratified")
    buckets = [
        "unsafe"
        if candidates[index]["enters_grid_unsafe"]
        else "progress"
        if candidates[index]["target_progress_m"] > 0.0
        else "other"
        for index in selected
    ]

    assert len(selected) == 9
    assert selected == sorted(selected)
    assert 4 in selected
    assert buckets.count("progress") == 3
    assert buckets.count("unsafe") == 3
    assert buckets.count("other") == 3
