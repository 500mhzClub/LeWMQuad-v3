from __future__ import annotations

from lewm.benchmarks.counterfactual import Pose2D
from scripts.build_jepa_counterfactual_render_plans import (
    _frame,
    _selected_candidate_indices,
)


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


def test_render_plan_frame_context_propagates_phase2d_lineage() -> None:
    row = {
        "benchmark_schema": "jepa_counterfactual_decision_v0",
        "scene_id": "scene",
        "start_frame": "start.png",
        "topology_seed": 123,
        "visual_seed": 456,
        "phase2d_source_state_lineage": {"lineage_verified": True},
        "start_base_pose_world": {"position": {"x": 0.0, "y": 0.0, "z": 0.4}},
        "start_base_rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "block_size": 5,
        "command_dt_s": 0.1,
        "counterfactual_oracle_index": 0,
        "counterfactual_candidates": [
            {"primitive_sequence": ["hold", "forward"]}
        ],
    }

    frame = _frame(
        frame_index=0,
        row=row,
        source_index=9,
        candidate_index=0,
        block_index=1,
        endpoint=Pose2D(x_m=1.0, y_m=2.0, yaw_rad=0.3),
    )

    context = frame["counterfactual_context"]
    assert context["source_index"] == 9
    assert context["scene_id"] == "scene"
    assert context["topology_seed"] == 123
    assert context["visual_seed"] == 456
    assert context["phase2d_lineage_verified"]
