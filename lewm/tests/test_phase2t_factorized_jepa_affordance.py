from __future__ import annotations

import torch

from lewm.benchmarks.phase2m_primitive_affordance import (
    build_primitive_affordance_examples,
)
from lewm.benchmarks.phase2t_factorized_jepa_affordance import (
    factorized_sequence_primitive_selection_records,
    factorized_sequence_primitive_selection_summary,
    materialize_phase2t_sequence_factor_targets,
    phase2t_sequence_factor_target_audit,
)


def _labels(progress: float, *, unsafe: bool = False) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": 0.2,
        "minimum_swept_configuration_clearance_m": 0.3,
        "unsafe_sample_fraction": 1.0 if unsafe else 0.0,
        "enters_grid_unsafe": unsafe,
        "ends_grid_unsafe": False,
        "target_recoverable": not unsafe,
        "target_heading_error_rad": 0.0,
    }


def _row(sequence: tuple[str, str], progress: float, *, unsafe: bool = False) -> dict:
    return {
        "scene_id": "scene",
        "source_index": 1,
        "start_frame": "start.png",
        "primitive_sequence": list(sequence),
        "active_blocks": [[0.0], [0.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": _labels(progress, unsafe=unsafe),
    }


def test_sequence_factor_targets_and_selection_summary() -> None:
    rows = [
        _row(("left", "hold"), 0.3),
        _row(("left", "right"), 0.2),
        _row(("right", "hold"), -0.2, unsafe=True),
        _row(("right", "left"), -0.1),
    ]
    primitive_examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("left", "right"),
    )
    targets, mask = materialize_phase2t_sequence_factor_targets(rows)
    audit = phase2t_sequence_factor_target_audit(rows)

    records = factorized_sequence_primitive_selection_records(
        rows,
        targets,
        primitive_examples,
        seed=7,
        split_name="validation",
        scorer_name="unit_test",
    )
    summary = factorized_sequence_primitive_selection_summary(records)

    assert targets.shape == (4, 6)
    assert mask.shape == (4, 6)
    assert audit["core_factor_target_values"] == 20
    assert len(records) == 1
    assert records[0]["selected_primitive"] == "left"
    assert records[0]["primitive_match"]
    assert summary["primitive_match_rate"] == 1.0


def test_sequence_selection_rejects_wrong_row_count() -> None:
    rows = [_row(("left", "hold"), 0.3), _row(("right", "hold"), -0.1)]
    primitive_examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("left", "right"),
    )
    try:
        factorized_sequence_primitive_selection_records(
            rows,
            torch.zeros((1, 6)),
            primitive_examples,
            seed=7,
            split_name="validation",
            scorer_name="unit_test",
        )
    except ValueError as error:
        assert "row count" in str(error)
    else:
        raise AssertionError("expected row-count validation failure")
