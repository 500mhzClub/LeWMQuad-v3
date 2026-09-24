from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lewm.datasets.go2_paired_navigation import sha256_file
from scripts import audit_go2_paired_navigation_adequacy as adequacy_script
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
    schema: str = "lewm_go2_paired_navigation_dataset_v2",
    include_untouched_shard_record: bool = False,
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
        "schema": schema,
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
    if schema == "lewm_go2_paired_navigation_dataset_v3":
        manifest["label_semantics"] = {
            "label_contract": "observable_physical_occupancy_v3",
            "target_occupancy_space": "observable_physical_occupancy",
            "per_frame_configuration_classes_supervised": False,
            "post_memory_configuration_derivation_is_evaluation_only": True,
        }
    if include_untouched_shard_record:
        manifest["shards"].append(
            {
                "scene_id": "scene_untouched",
                "path": str(tmp_path / "scene_untouched.npz"),
                "sha256": "0" * 64,
                "rows": 1,
            }
        )
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
    assert report["dataset_schema"] == "lewm_go2_paired_navigation_dataset_v2"
    assert report["label_contract"] == "center_visible_configuration_v2"
    assert report["target_occupancy_space"] == (
        "body_inflated_configuration_space"
    )
    counts = report["per_role"]["probability_calibration"][
        "supervised_next_cell_counts"
    ]
    assert counts == {"unknown": 33, "free": 12, "occupied": 3}


def test_observable_physical_v3_passes_and_records_semantics(tmp_path: Path) -> None:
    manifest_path = _dataset(
        tmp_path,
        schema="lewm_go2_paired_navigation_dataset_v3",
        include_untouched_shard_record=True,
    )
    report = audit_paired_navigation_adequacy(
        manifest_path,
        minimum_calibration_free_cells=4,
        minimum_calibration_occupied_cells=1,
    )
    assert report["passed"] is True
    assert report["dataset_schema"] == "lewm_go2_paired_navigation_dataset_v3"
    assert report["label_contract"] == "observable_physical_occupancy_v3"
    assert report["target_occupancy_space"] == "observable_physical_occupancy"
    assert report["untouched_g2_shards_opened"] == 0


def test_untouched_g2_shard_is_never_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _dataset(tmp_path, include_untouched_shard_record=True)
    real_load = adequacy_script.np.load
    opened: list[Path] = []

    def recording_load(path: str | Path, *args: object, **kwargs: object):
        opened.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(adequacy_script.np, "load", recording_load)
    report = audit_paired_navigation_adequacy(
        manifest_path,
        minimum_calibration_free_cells=4,
        minimum_calibration_occupied_cells=1,
    )
    assert report["untouched_g2_shards_opened"] == 0
    assert tmp_path / "scene_untouched.npz" not in opened


def test_v3_semantic_mismatch_is_rejected_before_shards_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _dataset(
        tmp_path, schema="lewm_go2_paired_navigation_dataset_v3"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["label_semantics"]["target_occupancy_space"] = (
        "body_inflated_configuration_space"
    )
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        adequacy_script.np,
        "load",
        lambda *args, **kwargs: pytest.fail("semantic rejection opened a shard"),
    )
    with pytest.raises(AdequacyError, match="label semantics disagree"):
        audit_paired_navigation_adequacy(manifest_path)


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
