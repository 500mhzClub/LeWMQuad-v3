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
