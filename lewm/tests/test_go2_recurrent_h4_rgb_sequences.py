from __future__ import annotations

from pathlib import Path

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import FAMILIES
from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import SourceRef
from lewm.datasets.go2_recurrent_h4_rgb_sequences import (
    Endpoint,
    H6Window,
    SCHEMA,
    SequenceContractError,
    _coverage,
    _interleave_families,
    _source_directory,
    canonical_row_bytes,
    validate_selected_rgb,
)
import pytest


def _window(*, family: str = FAMILIES[0], scene: str | None = None) -> H6Window:
    scene_id = scene or f"{family}_0123456789ab"
    endpoints = tuple(
        Endpoint(frame_index=17 + 48 * index, env_index=17, episode_step=index + 1, timestamp_ns=index)
        for index in range(7)
    )
    return H6Window(
        rank="0" * 64,
        role="train",
        family=family,
        scene_id=scene_id,
        endpoints=endpoints,
        actions=(0, 1, 2, 3, 4, 5),
    )


def test_window_serializes_only_rgb_action_and_public_identity() -> None:
    row = _window().to_row()
    assert set(row) == {"schema", "role", "family", "scene_id", "rgb", "actions"}
    assert row["schema"] == SCHEMA
    assert len(row["rgb"]) == 7
    assert row["rgb"][0].endswith("/rgb/frame_000017_env_17.png")
    assert row["actions"] == [0, 1, 2, 3, 4, 5]
    assert canonical_row_bytes(_window()).endswith(b"\n")


def test_family_interleave_is_rectangular_and_batch_balanced() -> None:
    by_family = {family: [_window(family=family)] * 2 for family in FAMILIES}
    result = _interleave_families(by_family)
    assert len(result) == 16
    assert {item.family for item in result[:8]} == set(FAMILIES)
    assert {item.family for item in result[8:]} == set(FAMILIES)


def test_selected_rgb_validation_uses_exact_leaves(tmp_path: Path) -> None:
    window = _window()
    rgb_dir = (
        tmp_path
        / ".generated"
        / "datagen_full"
        / "render_textured_v03"
        / window.scene_id
        / "rgb"
    )
    rgb_dir.mkdir(parents=True)
    for relative in window.to_row()["rgb"]:
        (rgb_dir / Path(relative).name).write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    result = validate_selected_rgb(tmp_path, [window])
    assert result == {"unique_rgb_count": 7, "unique_rgb_byte_count": 105}


def test_future_action_coverage_reports_missing_cells() -> None:
    result = _coverage([_window()])
    assert result["row_count"] == 1
    assert result["missing_action_position_cells"]


def test_public_source_scan_cannot_escape_the_allowlisted_roles(tmp_path: Path) -> None:
    source = SourceRef(
        role="/tmp/escape",
        family=FAMILIES[0],
        chunk="chunk_0000",
        sequence=f"000000_{FAMILIES[0]}_0123456789ab",
        byte_count=1,
        ordinal=1,
    )
    with pytest.raises(SequenceContractError, match="allowlist"):
        _source_directory(tmp_path, source)
