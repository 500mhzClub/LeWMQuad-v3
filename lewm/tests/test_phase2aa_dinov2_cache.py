from __future__ import annotations

from pathlib import Path

from PIL import Image

from lewm.benchmarks.phase2aa_dinov2_cache import (
    phase2aa_frame_cache_audit,
    phase2aa_unique_frame_records,
)


def _image(path: Path, value: int) -> str:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
    return str(path)


def test_phase2aa_frame_records_use_start_and_valid_future_frames(
    tmp_path: Path,
) -> None:
    start = _image(tmp_path / "start.png", 10)
    future_0 = _image(tmp_path / "future_0.png", 20)
    future_1 = _image(tmp_path / "future_1.png", 30)
    rows = [
        {
            "scene_id": "scene_a",
            "source_index": 1,
            "start_frame": start,
            "active_blocks": [[1.0], [2.0]],
            "future_frames": [future_0, future_1],
            "future_observations": [
                {"rgb_path": future_0, "observation_valid": True},
                {"rgb_path": future_1, "observation_valid": False},
            ],
        },
        {
            "scene_id": "scene_a",
            "source_index": 2,
            "start_frame": start,
            "active_blocks": [[3.0]],
            "future_frames": [future_0],
            "future_observations": [
                {"rgb_path": future_0, "observation_valid": True},
            ],
        },
    ]

    records = phase2aa_unique_frame_records(rows)
    audit = phase2aa_frame_cache_audit(rows, records, split_name="train")

    assert [Path(record.frame_path).name for record in records] == [
        "future_0.png",
        "start.png",
    ]
    assert records[0].roles == ("future_step_0",)
    assert records[0].row_count == 2
    assert records[1].roles == ("start",)
    assert records[1].row_count == 2
    assert audit["source_rows"] == 2
    assert audit["transition_slots"] == 3
    assert audit["valid_transitions"] == 2
    assert audit["unique_frames"] == 2
    assert audit["all_frames_exist"]
