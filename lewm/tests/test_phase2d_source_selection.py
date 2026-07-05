from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.phase2d_source_selection import select_phase2d_source_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_select_phase2d_source_rows_is_deterministic_and_bounded(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"png")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scene_id": "manifest",
                "family": "family",
                "topology_seed": 1,
                "visual_seed": 2,
            }
        )
    )
    rows = []
    for scene_index in range(4):
        for row_index in range(6):
            rows.append(
                {
                    "scene_id": f"scene_{scene_index}",
                    "family": "family",
                    "split": "train",
                    "scene_manifest": str(manifest),
                    "start_frame": f"scene_{scene_index}/frame_{row_index}.png",
                    "local_target_frame": str(target),
                    "route_target_id": 1,
                    "oracle_next_cell_id": 1,
                }
            )
    source = tmp_path / "source.jsonl"
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    _write_jsonl(source, rows)

    summary_a = select_phase2d_source_rows(
        split_name="train",
        source_path=source,
        output_path=out_a,
        scene_count=3,
        source_states_per_scene=4,
        seed=7,
    )
    summary_b = select_phase2d_source_rows(
        split_name="train",
        source_path=source,
        output_path=out_b,
        scene_count=3,
        source_states_per_scene=4,
        seed=7,
    )

    selected = [json.loads(line) for line in out_a.read_text().splitlines()]
    assert out_a.read_text() == out_b.read_text()
    assert summary_a["selected_scene_ids"] == summary_b["selected_scene_ids"]
    assert summary_a["selected_scene_count"] == 3
    assert summary_a["selected_source_rows"] == 12
    assert {row["scene_id"] for row in selected} == set(
        summary_a["selected_scene_ids"]
    )
    assert all(count == 4 for count in summary_a["selected_rows_by_scene"].values())
