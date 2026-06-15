from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.phase2_data import (
    build_hard_negative_index,
    confirmatory_data_gate,
    load_spatial_future_rows,
    materialized_frame_paths,
    pairwise_split_overlap,
    phase2_dataset_audit,
    source_grouped_batches,
    transition_validity,
)


def _row(
    *,
    source_index: int,
    primitives: list[str],
    actions: list[list[float]],
    valid: list[bool],
    scene_id: str = "scene_a",
    split: str = "train",
) -> dict:
    return {
        "scene_id": scene_id,
        "family": "family",
        "split": split,
        "source_index": source_index,
        "start_frame": "start.png",
        "primitive_sequence": primitives,
        "active_blocks": actions,
        "future_frames": [
            f"future_{source_index}_{step}.png" if item else None
            for step, item in enumerate(valid)
        ],
        "future_observations": [
            {
                "rgb_path": f"future_{source_index}_{step}.png" if item else None,
                "observation_valid": item,
            }
            for step, item in enumerate(valid)
        ],
        "complete_valid_future_sequence": all(valid),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_transition_validity_requires_valid_current_and_future_observations() -> None:
    assert transition_validity(
        _row(
            source_index=0,
            primitives=["forward", "arc_left"],
            actions=[[1.0], [2.0]],
            valid=[True, False],
        )
    ) == (True, False)
    assert transition_validity(
        _row(
            source_index=0,
            primitives=["forward", "arc_left"],
            actions=[[1.0], [2.0]],
            valid=[False, True],
        )
    ) == (False, False)


def test_load_rows_retains_partial_sequences_and_records_selection(tmp_path: Path) -> None:
    rows = [
        _row(
            source_index=0,
            primitives=["forward", "hold"],
            actions=[[1.0], [0.0]],
            valid=[True, False],
        ),
        _row(
            source_index=1,
            primitives=["forward", "arc_left"],
            actions=[[1.0], [2.0]],
            valid=[True, True],
        ),
    ]
    path = tmp_path / "data.jsonl"
    _write_rows(path, rows)

    all_rows, all_audit = load_spatial_future_rows(path, mode="all")
    partial_rows, partial_audit = load_spatial_future_rows(path, mode="any_transition")
    complete_rows, complete_audit = load_spatial_future_rows(path, mode="complete")

    assert len(all_rows) == 2
    assert len(partial_rows) == 2
    assert len(complete_rows) == 1
    assert all_audit["excluded_by_mode"] == 0
    assert partial_audit["excluded_by_mode"] == 0
    assert complete_audit["excluded_by_mode"] == 1


def test_materialized_paths_substitute_invalid_slots_but_mask_them() -> None:
    row = _row(
        source_index=0,
        primitives=["forward", "arc_left"],
        actions=[[1.0], [2.0]],
        valid=[True, False],
    )

    paths, mask = materialized_frame_paths(row)

    assert paths == (
        Path("start.png"),
        Path("future_0_0.png"),
        Path("future_0_0.png"),
    )
    assert mask == (True, False)


def test_hard_negatives_are_same_source_unique_and_nonidentical() -> None:
    rows = [
        _row(
            source_index=0,
            primitives=["forward"],
            actions=[[1.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["forward_duplicate"],
            actions=[[1.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["arc_left"],
            actions=[[2.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["hold"],
            actions=[[0.0]],
            valid=[True],
        ),
        _row(
            source_index=1,
            primitives=["arc_right"],
            actions=[[3.0]],
            valid=[True],
        ),
    ]

    index = build_hard_negative_index(rows, step=0)

    assert index.candidates[0] == (2, 3)
    assert index.candidates[1] == (2, 3)
    assert 3 not in index.candidates
    assert 4 not in index.candidates
    assert index.audit["identical_negative_count"] == 0


def test_source_grouped_batches_keep_all_candidates_from_a_source_together() -> None:
    rows = [
        _row(
            source_index=source,
            primitives=[f"action_{candidate}"],
            actions=[[float(candidate)]],
            valid=[True],
        )
        for source in range(3)
        for candidate in range(2)
    ]

    batches = source_grouped_batches(
        rows,
        source_states_per_batch=2,
        seed=17,
        shuffle=False,
    )

    assert batches == ((0, 1, 2, 3), (4, 5))
    assert source_grouped_batches(
        rows,
        source_states_per_batch=2,
        seed=17,
    ) == source_grouped_batches(
        rows,
        source_states_per_batch=2,
        seed=17,
    )


def test_dataset_audit_reports_complete_filter_and_legacy_control_contamination() -> None:
    rows = [
        _row(
            source_index=0,
            primitives=["forward"],
            actions=[[1.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["forward_duplicate"],
            actions=[[1.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["arc_left"],
            actions=[[2.0]],
            valid=[False],
        ),
    ]

    audit = phase2_dataset_audit(rows, legacy_batch_size=2)

    assert audit["rows"] == 3
    assert audit["complete_valid_rows"] == 2
    assert audit["hard_negative_by_step"][0]["identical_negative_count"] == 0
    assert (
        audit["legacy_complete_subset_rolling_control"]["same_action_fraction_by_step"][0]
        == 1.0
    )


def test_split_overlap_is_not_hidden_by_different_split_labels() -> None:
    train = [
        _row(
            source_index=0,
            primitives=["forward"],
            actions=[[1.0]],
            valid=[True],
            split="train",
        )
    ]
    evaluation = [
        _row(
            source_index=0,
            primitives=["forward"],
            actions=[[1.0]],
            valid=[True],
            split="test",
        )
    ]

    overlap = pairwise_split_overlap({"train": train, "test": evaluation})

    assert overlap["test__train"]["scene_ids"] == ["scene_a"]
    assert overlap["test__train"]["source_keys"] == [["scene_a", 0]]


def test_confirmatory_gate_requires_all_splits_sizes_balance_and_lineage() -> None:
    rows = [
        _row(
            source_index=0,
            primitives=["forward"],
            actions=[[1.0]],
            valid=[True],
        ),
        _row(
            source_index=0,
            primitives=["arc_left"],
            actions=[[2.0]],
            valid=[True],
        ),
    ]

    gate = confirmatory_data_gate({"train": rows})

    assert not gate["passed"]
    assert gate["missing_splits"] == ["test_hard", "test_id", "validation"]
    assert not gate["checks"]["all_required_splits_present"]
    assert not gate["checks"]["all_split_requirements_passed"]
    assert not gate["checks"]["artifact_and_seed_lineage_verified"]
