from __future__ import annotations

import pytest

from lewm.benchmarks.phase2d_generation import (
    PHASE2D_EXPECTED_SEQUENCE_COUNT,
    PHASE2D_HORIZON_BLOCKS,
    PHASE2D_PRIMITIVE_NAMES,
    factorial_primitive_sequences,
    phase2d_lineage_fields,
    sequence_grid_audit,
)


def test_factorial_grid_matches_registered_phase2d_sequence_count() -> None:
    sequences = factorial_primitive_sequences(
        PHASE2D_PRIMITIVE_NAMES,
        horizon_blocks=PHASE2D_HORIZON_BLOCKS,
    )
    audit = sequence_grid_audit(
        primitive_names=PHASE2D_PRIMITIVE_NAMES,
        horizon_blocks=PHASE2D_HORIZON_BLOCKS,
        sequences=sequences,
    )

    assert len(sequences) == PHASE2D_EXPECTED_SEQUENCE_COUNT
    assert len(set(sequences)) == PHASE2D_EXPECTED_SEQUENCE_COUNT
    assert audit["expected_sequence_count"] == 81
    assert audit["observed_sequence_count"] == 81
    assert audit["first_action_count"] == 9
    assert all(count == 9 for count in audit["first_action_counts"].values())
    assert audit["full_factorial_passed"]
    assert audit["phase2d_full_81_two_block_grid"]


def test_factorial_grid_rejects_duplicate_primitive_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        factorial_primitive_sequences(("hold", "hold"), horizon_blocks=2)


def test_sequence_grid_audit_reports_missing_and_unexpected_sequences() -> None:
    audit = sequence_grid_audit(
        primitive_names=("hold", "forward"),
        horizon_blocks=2,
        sequences=(("hold", "hold"), ("hold", "turn")),
    )

    assert not audit["full_factorial_passed"]
    assert audit["missing_sequence_count"] == 3
    assert audit["unexpected_sequence_count"] == 1
    assert audit["duplicate_sequence_count"] == 0


def test_phase2d_lineage_fields_prefer_row_values() -> None:
    fields = phase2d_lineage_fields(
        {
            "scene_id": "scene",
            "source_index": 4,
            "start_frame": "start.png",
            "topology_seed": 123,
            "scene_metadata": {"visual_seed": 456},
        },
        scene_manifest={"topology_seed": 999, "visual_seed": 888},
    )

    assert fields["topology_seed"] == 123
    assert fields["visual_seed"] == 456
    audit = fields["phase2d_source_state_lineage"]
    assert audit["lineage_verified"]
    assert audit["field_sources"] == {
        "topology_seed": "row",
        "visual_seed": "row.scene_metadata",
    }


def test_phase2d_lineage_fields_fall_back_to_scene_manifest() -> None:
    fields = phase2d_lineage_fields(
        {"scene_id": "scene", "source_index": 1, "start_frame": "start.png"},
        scene_manifest={"topology_seed": 12, "visual_seed": 34},
    )

    assert fields["topology_seed"] == 12
    assert fields["visual_seed"] == 34
    assert fields["phase2d_source_state_lineage"]["lineage_verified"]


def test_phase2d_lineage_fields_reports_missing_fields() -> None:
    fields = phase2d_lineage_fields(
        {"scene_id": "scene", "source_index": 1, "start_frame": "start.png"},
    )

    assert fields["topology_seed"] is None
    assert fields["visual_seed"] is None
    assert fields["phase2d_source_state_lineage"]["missing_fields"] == [
        "topology_seed",
        "visual_seed",
    ]
    assert not fields["phase2d_source_state_lineage"]["lineage_verified"]
