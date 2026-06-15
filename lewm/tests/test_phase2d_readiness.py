from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.experiment_manifest import (
    build_experiment_manifest,
    write_json,
)
from lewm.benchmarks.phase2d_readiness import (
    build_phase2d_split_manifest,
    canonical_split_name,
    phase2d_run_readiness,
    phase2d_training_start_readiness,
    source_state_lineage_audit,
    verify_phase2d_split_manifest,
)


def _row(
    *,
    split: str = "train",
    source_index: int = 0,
    candidate_index: int = 0,
    include_lineage: bool = False,
) -> dict:
    row = {
        "scene_id": f"scene_{split}",
        "family": "family",
        "split": split,
        "source_index": source_index,
        "candidate_index": candidate_index,
        "start_frame": "start.png",
        "primitive_sequence": ["hold", "forward"],
        "active_blocks": [[0.0], [1.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "future_observations": [
            {"rgb_path": "future_0.png", "observation_valid": True},
            {"rgb_path": "future_1.png", "observation_valid": True},
        ],
    }
    if include_lineage:
        row.update({"topology_seed": 123, "visual_seed": 456})
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _ready_split_manifest(path: Path, dataset: Path) -> Path:
    record = {
        "path": str(dataset.resolve()),
        "sha256": "unused",
        "size_bytes": dataset.stat().st_size,
    }
    # Use build_experiment_manifest once to get a real artifact record without
    # depending on private helpers in this test.
    manifest = build_experiment_manifest(
        experiment_id="artifact",
        repository_root=path.parent,
        inputs={"dataset": dataset},
    )
    record = manifest["inputs"]["dataset"]
    split_manifest = {
        "schema": "jepa_phase2d_split_manifest_v0",
        "splits": {
            name: {
                "dataset_file": record,
                "image_artifacts": {},
            }
            for name in ("train", "validation", "test_id", "test_hard")
        },
        "lineage": {"lineage_verified": True},
        "confirmatory_gate": {"passed": True},
    }
    write_json(path, split_manifest)
    return path


def _cell_manifest(path: Path, checkpoint: Path, *, cell: str) -> Path:
    checkpoint.write_text(f"checkpoint-{cell}")
    manifest = build_experiment_manifest(
        experiment_id=f"{cell}-manifest",
        repository_root=path.parent,
        inputs={},
        artifacts={"selected_checkpoint": checkpoint},
        config={
            "cell": cell,
            "run_class": "confirmatory",
            "checkpoint_rule": "registered_phase2d_validation_v0",
        },
    )
    write_json(path, manifest)
    return path


def test_canonical_split_name_accepts_registered_aliases() -> None:
    assert canonical_split_name("val") == "validation"
    assert canonical_split_name("test-id") == "test_id"
    assert canonical_split_name("test_hard") == "test_hard"


def test_source_state_lineage_requires_topology_and_visual_seed() -> None:
    missing = source_state_lineage_audit([_row()], split_name="train")
    present = source_state_lineage_audit(
        [_row(include_lineage=True)],
        split_name="train",
    )

    assert not missing["lineage_verified"]
    assert missing["missing_lineage_field_counts"] == {
        "topology_seed": 1,
        "visual_seed": 1,
    }
    assert present["lineage_verified"]


def test_split_manifest_records_strict_gate_failure(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    _write_rows(train, [_row()])

    manifest = build_phase2d_split_manifest({"train": train})
    verification = verify_phase2d_split_manifest(manifest)

    assert not manifest["lineage"]["lineage_verified"]
    assert not manifest["confirmatory_gate"]["passed"]
    assert not verification["passes"]
    assert verification["files"]["splits.train.dataset_file"]["matches"]


def test_run_readiness_requires_all_cells_and_blocks_test_hard_until_test_id_report(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_rows(dataset, [_row(include_lineage=True)])
    split_manifest = _ready_split_manifest(tmp_path / "splits.json", dataset)
    cell_manifests = {
        cell: _cell_manifest(
            tmp_path / f"{cell}.json",
            tmp_path / f"{cell}.pt",
            cell=cell,
        )
        for cell in ("C0", "C1", "C2")
    }

    ready = phase2d_run_readiness(
        split_manifest_path=split_manifest,
        cell_manifest_paths=cell_manifests,
        requested_stage="test_id",
    )
    missing = phase2d_run_readiness(
        split_manifest_path=split_manifest,
        cell_manifest_paths={key: value for key, value in cell_manifests.items() if key != "C2"},
        requested_stage="test_id",
    )
    blocked_hard = phase2d_run_readiness(
        split_manifest_path=split_manifest,
        cell_manifest_paths=cell_manifests,
        requested_stage="test_hard",
    )
    test_id_report = build_experiment_manifest(
        experiment_id="test-id-report",
        repository_root=tmp_path,
        inputs={},
        artifacts={"test_id_report": dataset},
    )
    test_id_manifest = tmp_path / "test_id_report_manifest.json"
    write_json(test_id_manifest, test_id_report)
    hard_ready = phase2d_run_readiness(
        split_manifest_path=split_manifest,
        cell_manifest_paths=cell_manifests,
        requested_stage="test_hard",
        test_id_report_manifest_path=test_id_manifest,
    )

    assert ready["passed"]
    assert not missing["passed"]
    assert missing["missing_cells"] == ["C2"]
    assert not blocked_hard["passed"]
    assert not blocked_hard["checks"]["test_stage_access_rule_passed"]
    assert hard_ready["passed"]


def test_training_start_readiness_requires_manifest_paths_and_primary_cell(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_rows(train, [_row(split="train", include_lineage=True)])
    _write_rows(validation, [_row(split="validation", include_lineage=True)])
    split_manifest = tmp_path / "splits.json"
    train_record = build_experiment_manifest(
        experiment_id="train-artifact",
        repository_root=tmp_path,
        inputs={"train": train},
    )["inputs"]["train"]
    validation_record = build_experiment_manifest(
        experiment_id="validation-artifact",
        repository_root=tmp_path,
        inputs={"validation": validation},
    )["inputs"]["validation"]
    manifest = {
        "schema": "jepa_phase2d_split_manifest_v0",
        "splits": {
            "train": {"dataset_file": train_record, "image_artifacts": {}},
            "validation": {
                "dataset_file": validation_record,
                "image_artifacts": {},
            },
            "test_id": {"dataset_file": validation_record, "image_artifacts": {}},
            "test_hard": {"dataset_file": validation_record, "image_artifacts": {}},
        },
        "lineage": {"lineage_verified": True},
        "confirmatory_gate": {"passed": True},
    }
    write_json(split_manifest, manifest)

    ready = phase2d_training_start_readiness(
        split_manifest_path=split_manifest,
        cell="C2",
        requested_run_class="confirmatory",
        train_data_path=train,
        validation_data_path=validation,
    )
    diagnostic = phase2d_training_start_readiness(
        split_manifest_path=split_manifest,
        cell="state_only",
        requested_run_class="confirmatory",
        train_data_path=train,
        validation_data_path=validation,
    )
    mismatch = phase2d_training_start_readiness(
        split_manifest_path=split_manifest,
        cell="C2",
        requested_run_class="confirmatory",
        train_data_path=validation,
        validation_data_path=validation,
    )
    manifest["confirmatory_gate"] = {"passed": False}
    write_json(split_manifest, manifest)
    blocked_gate = phase2d_training_start_readiness(
        split_manifest_path=split_manifest,
        cell="C2",
        requested_run_class="confirmatory",
        train_data_path=train,
        validation_data_path=validation,
    )

    assert ready["passed"]
    assert not diagnostic["passed"]
    assert not diagnostic["checks"][
        "confirmatory_cell_participates_in_primary_comparison"
    ]
    assert not mismatch["passed"]
    assert not mismatch["checks"]["train_data_matches_manifest"]
    assert not blocked_gate["passed"]
    assert not blocked_gate["checks"]["confirmatory_data_gate_required_and_passed"]
