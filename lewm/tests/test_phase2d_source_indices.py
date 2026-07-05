from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS
from lewm.benchmarks.phase2d_source_indices import audit_phase2d_source_indices


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _manifest(path: Path, *, split_index: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene_id": f"manifest_{split_index}",
                "family": "family",
                "topology_seed": 1000 + split_index,
                "visual_seed": 2000 + split_index,
            }
        )
    )
    return path


def _rows(
    *,
    split: str,
    manifest: Path,
    target: Path,
    scenes: int,
    source_rows_per_scene: int,
    missing_target_rows: int = 0,
) -> list[dict]:
    rows = []
    row_index = 0
    for scene_index in range(scenes):
        scene_id = f"{split}_scene_{scene_index:03d}"
        for source_index in range(source_rows_per_scene):
            local_target = None if row_index < missing_target_rows else str(target)
            rows.append(
                {
                    "scene_id": scene_id,
                    "family": "family",
                    "split": "val" if split == "validation" else split,
                    "scene_manifest": str(manifest),
                    "start_frame": f"{scene_id}/start_{source_index:04d}.png",
                    "local_target_frame": local_target,
                    "route_target_id": 7,
                    "oracle_next_cell_id": 7,
                    "decision_types": ["branch"],
                }
            )
            row_index += 1
    return rows


def test_source_index_readiness_requires_all_registered_splits(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"png")
    manifest = _manifest(tmp_path / "manifest.json", split_index=0)
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        _rows(
            split="train",
            manifest=manifest,
            target=target,
            scenes=1,
            source_rows_per_scene=1,
        ),
    )

    report = audit_phase2d_source_indices({"train": train})

    assert not report["ready_for_counterfactual_generation"]
    assert report["missing_splits"] == ["test_hard", "test_id", "validation"]
    assert not report["checks"]["all_required_source_indices_present"]


def test_source_index_readiness_passes_complete_synthetic_indices(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"png")
    paths = {}
    for split_index, (split, requirements) in enumerate(
        CONFIRMATORY_SPLIT_REQUIREMENTS.items()
    ):
        manifest = _manifest(tmp_path / f"{split}_manifest.json", split_index=split_index)
        path = tmp_path / f"{split}.jsonl"
        _write_jsonl(
            path,
            _rows(
                split=split,
                manifest=manifest,
                target=target,
                scenes=requirements["minimum_scenes"],
                source_rows_per_scene=requirements[
                    "minimum_source_states_per_scene"
                ],
            ),
        )
        paths[split] = path

    report = audit_phase2d_source_indices(paths)

    assert report["ready_for_counterfactual_generation"]
    assert report["checks"]["no_scene_source_or_lineage_overlap"]
    assert report["splits"]["train"]["minimum_scene_count_passed"]
    assert report["splits"]["validation"]["split_label_mismatch_rows"] == 0


def test_source_index_readiness_counts_missing_local_target_rows(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"png")
    requirements = CONFIRMATORY_SPLIT_REQUIREMENTS["validation"]
    manifest = _manifest(tmp_path / "manifest.json", split_index=1)
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation,
        _rows(
            split="validation",
            manifest=manifest,
            target=target,
            scenes=requirements["minimum_scenes"],
            source_rows_per_scene=requirements["minimum_source_states_per_scene"],
            missing_target_rows=1,
        ),
    )

    report = audit_phase2d_source_indices({"validation": validation})

    assert report["splits"]["validation"]["skipped_missing_local_target_frame"] == 1
    assert not report["splits"]["validation"][
        "minimum_source_states_per_scene_passed"
    ]
    assert not report["ready_for_counterfactual_generation"]
