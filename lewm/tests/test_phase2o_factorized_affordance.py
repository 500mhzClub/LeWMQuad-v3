from __future__ import annotations

import pytest

from lewm.benchmarks.phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    build_factorized_primitive_affordance_examples,
    factorized_affordance_dataset_audit,
    factorized_candidate_targets,
)


def _labels(
    *,
    progress: float | None = 0.0,
    clearance_gain: float | None = None,
    p05: float = 0.2,
    minimum: float = 0.3,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
    ends_unsafe: bool = False,
    recoverable: bool = True,
    heading_error: float = 0.0,
) -> dict:
    labels = {
        "p05_swept_configuration_clearance_m": p05,
        "minimum_swept_configuration_clearance_m": minimum,
        "unsafe_sample_fraction": unsafe_fraction,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": ends_unsafe,
        "target_recoverable": recoverable,
        "target_heading_error_rad": heading_error,
    }
    if progress is not None:
        labels["target_progress_m"] = progress
    if clearance_gain is not None:
        labels["clearance_gain_m"] = clearance_gain
    return labels


def _row(
    *,
    scene_id: str,
    source_index: int,
    sequence: tuple[str, str],
    labels: dict,
) -> dict:
    return {
        "scene_id": scene_id,
        "family": "family",
        "source_index": source_index,
        "start_frame": "start.png",
        "primitive_sequence": list(sequence),
        "active_blocks": [[1.0], [2.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": labels,
    }


def test_factorized_candidate_targets_encode_safety_and_geometry() -> None:
    values, mask = factorized_candidate_targets(
        {
            "consequence_labels": _labels(
                progress=0.15,
                p05=-0.2,
                minimum=0.4,
                unsafe_fraction=0.25,
                enters_unsafe=True,
                ends_unsafe=False,
                recoverable=True,
                heading_error=0.0,
            )
        }
    )

    assert len(values) == len(FACTORIZED_AFFORDANCE_FACTOR_NAMES)
    assert mask == (True, True, True, True, True, True)
    assert values[:6] == pytest.approx((
        0.0,
        0.5,
        0.0,
        0.5,
        0.25,
        1.0,
    ))


def test_factorized_examples_use_utility_best_continuation_per_primitive() -> None:
    rows = [
        _row(
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            labels=_labels(progress=0.0, p05=0.0),
        ),
        _row(
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "yaw_left"),
            labels=_labels(progress=0.3, p05=0.2),
        ),
        _row(
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            labels=_labels(
                progress=-0.3,
                enters_unsafe=True,
                unsafe_fraction=1.0,
            ),
        ),
    ]

    examples = build_factorized_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )
    audit = factorized_affordance_dataset_audit(examples, split_name="train")

    assert len(examples) == 1
    example = examples[0]
    assert example.oracle_primitive == "forward_slow"
    assert example.oracle_sequence == ("forward_slow", "yaw_left")
    assert example.factor_targets[0][0] == 1.0
    assert example.factor_targets[1][0] == 0.0
    assert audit["core_factors_complete"]
    assert audit["all_factors_complete"]
    assert audit["safe_positive_counts_by_primitive"] == {"forward_slow": 1}
