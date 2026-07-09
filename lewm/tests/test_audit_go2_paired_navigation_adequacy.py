from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lewm.datasets.go2_paired_navigation import sha256_file
from scripts.audit_go2_paired_navigation_adequacy import (
    AdequacyError,
    audit_paired_navigation_adequacy,
)


def _write_shard(
    path: Path,
    *,
    rows: int,
    labels: np.ndarray,
    supervision: np.ndarray,
    observed: np.ndarray,
) -> dict[str, object]:
    np.savez_compressed(
        path,
        next_labels=labels.astype(np.uint8),
        next_supervision_mask=supervision.astype(bool),
        next_observed_mask=observed.astype(bool),
    )
    return {"scene_id": path.stem, "path": str(path), "sha256": sha256_file(path), "rows": rows}


def _dataset(
    tmp_path: Path,
    *,
    calibration_labels: np.ndarray | None = None,
    observed_fraction: float = 1.0,
) -> Path:
    grid = (3, 4, 4)
    full = np.ones(grid, dtype=bool)
    base_labels = np.zeros(grid, dtype=np.uint8)
    base_labels[:, 0, :] = 1
    base_labels[:, 1, 0] = 2
    observed = np.ones(grid, dtype=bool)
    hidden_rows = int(round(grid[0] * (1.0 - observed_fraction)))
    if hidden_rows:
        observed[:hidden_rows] = False
    if calibration_labels is None:
        calibration_labels = base_labels
    shards = []
    for name, labels in (
        ("scene_train", base_labels),
        ("scene_selection", base_labels),
        ("scene_calibration", calibration_labels),
    ):
        shards.append(
            _write_shard(
                tmp_path / f"{name}.npz",
                rows=grid[0],
                labels=labels,
                supervision=full,
                observed=observed,
            )
        )
    manifest = {
        "schema": "lewm_go2_paired_navigation_dataset_v2",
        "scene_roles": {
            "assignments": {
                "scene_train": "train",
                "scene_selection": "checkpoint_selection",
                "scene_calibration": "probability_calibration",
                "scene_untouched": "g2_evaluation",
            }
        },
        # scene_untouched deliberately has no shard on disk: opening it fails.
        "shards": shards,
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_adequate_dataset_passes_without_opening_g2(tmp_path: Path) -> None:
    manifest_path = _dataset(tmp_path)
    report = audit_paired_navigation_adequacy(
        manifest_path,
        minimum_calibration_free_cells=4,
        minimum_calibration_occupied_cells=1,
    )
    assert report["passed"] is True
    assert report["untouched_g2_shards_opened"] == 0
    assert report["untouched_g2_scene_count"] == 1
    assert report["loaded_shards_opened"] == 3
    counts = report["per_role"]["probability_calibration"][
        "supervised_next_cell_counts"
    ]
    assert counts == {"unknown": 33, "free": 12, "occupied": 3}


def test_missing_calibration_class_fails(tmp_path: Path) -> None:
    labels = np.zeros((3, 4, 4), dtype=np.uint8)
    labels[:, 0, :] = 1  # free present, occupied absent
    manifest_path = _dataset(tmp_path, calibration_labels=labels)
    report = audit_paired_navigation_adequacy(
        manifest_path,
        minimum_calibration_free_cells=4,
        minimum_calibration_occupied_cells=1,
    )
    assert report["passed"] is False
    assert report["checks"]["calibration_role_has_all_three_classes"] is False
    assert report["checks"]["calibration_occupied_cells_at_floor"] is False


def test_low_next_observed_fraction_fails(tmp_path: Path) -> None:
    manifest_path = _dataset(tmp_path, observed_fraction=2 / 3)
    report = audit_paired_navigation_adequacy(
        manifest_path,
        minimum_calibration_free_cells=4,
        minimum_calibration_occupied_cells=1,
    )
    assert report["passed"] is False
    assert report["checks"]["next_observed_row_fraction_at_floor"] is False
    assert report["combined_next_observed_row_fraction"] == pytest.approx(2 / 3)


def test_shard_hash_mismatch_raises(tmp_path: Path) -> None:
    manifest_path = _dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AdequacyError, match="hash mismatch"):
        audit_paired_navigation_adequacy(
            manifest_path,
            minimum_calibration_free_cells=4,
            minimum_calibration_occupied_cells=1,
        )
