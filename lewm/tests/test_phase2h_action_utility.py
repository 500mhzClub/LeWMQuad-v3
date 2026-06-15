from __future__ import annotations

import pytest

from lewm.benchmarks.phase2h_action_utility import (
    evaluate_action_only_baseline,
    fit_action_utility_priors,
    phase2h_action_utility_audit,
    utility_label_audit,
)


def _labels(
    *,
    progress: float,
    enters_unsafe: bool = False,
    ends_unsafe: bool = False,
    recoverable: bool = True,
) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": 0.2,
        "unsafe_sample_fraction": 0.0,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": ends_unsafe,
        "target_recoverable": recoverable,
        "target_heading_error_rad": 0.0,
    }


def _row(
    *,
    scene_id: str,
    source_index: int,
    sequence: tuple[str, str],
    progress: float | None,
) -> dict:
    row = {
        "scene_id": scene_id,
        "family": "family",
        "source_index": source_index,
        "primitive_sequence": list(sequence),
        "active_blocks": [[1.0], [2.0]],
        "start_frame": "start.png",
        "future_frames": ["future_0.png", "future_1.png"],
    }
    if progress is not None:
        row["consequence_labels"] = _labels(progress=progress)
    return row


def test_utility_label_audit_tracks_valid_source_local_spread() -> None:
    rows = [
        _row(
            scene_id="scene_a",
            source_index=1,
            sequence=("forward", "forward"),
            progress=0.3,
        ),
        _row(
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "backward"),
            progress=-0.3,
        ),
        _row(
            scene_id="scene_a",
            source_index=2,
            sequence=("forward", "forward"),
            progress=None,
        ),
    ]

    audit = utility_label_audit(rows, split_name="train")

    assert audit["input_rows"] == 3
    assert audit["source_states"] == 2
    assert audit["valid_utility_rows"] == 2
    assert audit["valid_utility_source_states"] == 1
    assert audit["minimum_valid_candidate_rows_per_source"] == 2
    assert audit["source_records"][0]["oracle_sequence"] == [
        "forward",
        "forward",
    ]
    assert audit["source_records"][0]["utility_range"] > 0.0


def test_action_only_baseline_uses_train_priors_for_validation_selection() -> None:
    train_rows = [
        _row(
            scene_id="train_scene",
            source_index=1,
            sequence=("forward", "forward"),
            progress=0.3,
        ),
        _row(
            scene_id="train_scene",
            source_index=1,
            sequence=("backward", "backward"),
            progress=-0.3,
        ),
    ]
    validation_rows = [
        _row(
            scene_id="validation_scene",
            source_index=1,
            sequence=("forward", "forward"),
            progress=0.3,
        ),
        _row(
            scene_id="validation_scene",
            source_index=1,
            sequence=("backward", "backward"),
            progress=-0.3,
        ),
        _row(
            scene_id="validation_scene",
            source_index=2,
            sequence=("forward", "forward"),
            progress=-0.3,
        ),
        _row(
            scene_id="validation_scene",
            source_index=2,
            sequence=("backward", "backward"),
            progress=0.3,
        ),
    ]

    priors = fit_action_utility_priors(train_rows)
    summary = evaluate_action_only_baseline(
        validation_rows,
        priors,
        split_name="validation",
        baseline="full_sequence_mean",
    )

    assert priors["valid_training_rows"] == 2
    assert summary["source_state_count"] == 2
    assert summary["top1_match_rate"] == 0.5
    assert summary["first_primitive_match_rate"] == 0.5
    assert summary["uniform_random_expected_top1_rate"] == 0.5
    assert summary["mean_target_utility_regret"] > 0.0


def test_phase2h_audit_requires_valid_train_targets() -> None:
    with pytest.raises(ValueError, match="valid targets"):
        phase2h_action_utility_audit(
            train_rows=[
                _row(
                    scene_id="train_scene",
                    source_index=1,
                    sequence=("forward", "forward"),
                    progress=None,
                )
            ],
            validation_rows=[
                _row(
                    scene_id="validation_scene",
                    source_index=1,
                    sequence=("forward", "forward"),
                    progress=0.1,
                )
            ],
        )
