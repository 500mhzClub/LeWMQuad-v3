from __future__ import annotations

import json
from pathlib import Path

from lewm.benchmarks.experiment_manifest import (
    build_experiment_manifest,
    verify_manifest_files,
    write_json,
)
from lewm.benchmarks.phase2d_run_manifest import (
    PRIMARY_CELLS,
    REGISTERED_OPTIMIZATION_SEEDS,
    create_phase2d_training_run_manifests,
    phase2d_epoch_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _split_manifest(path: Path, train: Path, validation: Path) -> Path:
    train_record = build_experiment_manifest(
        experiment_id="train",
        repository_root=path.parent,
        inputs={"train": train},
    )["inputs"]["train"]
    validation_record = build_experiment_manifest(
        experiment_id="validation",
        repository_root=path.parent,
        inputs={"validation": validation},
    )["inputs"]["validation"]
    manifest = {
        "schema": "jepa_phase2d_split_manifest_v0",
        "splits": {
            "train": {
                "dataset_file": train_record,
                "dataset_audit": {"source_states": 512},
                "image_artifacts": {},
            },
            "validation": {
                "dataset_file": validation_record,
                "dataset_audit": {"source_states": 256},
                "image_artifacts": {},
            },
            "test_id": {
                "dataset_file": validation_record,
                "dataset_audit": {"source_states": 256},
                "image_artifacts": {},
            },
            "test_hard": {
                "dataset_file": validation_record,
                "dataset_audit": {"source_states": 256},
                "image_artifacts": {},
            },
        },
        "lineage": {"lineage_verified": True},
        "confirmatory_gate": {"passed": True},
    }
    write_json(path, manifest)
    return path


def test_phase2d_epoch_schedule_uses_source_grouped_epochs() -> None:
    schedule = phase2d_epoch_schedule(
        train_source_states=512,
        source_states_per_batch=2,
        epochs=3,
    )

    assert schedule["steps_per_epoch"] == 256
    assert schedule["optimization_steps"] == 768
    assert schedule["evaluation_interval"] == 256


def test_create_phase2d_training_run_manifests_writes_registered_matrix(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n")
    validation.write_text("{}\n")
    split_manifest = _split_manifest(tmp_path / "split_manifest.json", train, validation)

    summary = create_phase2d_training_run_manifests(
        repository_root=REPO_ROOT,
        split_manifest_path=split_manifest,
        train_data_path=train,
        validation_data_path=validation,
        output_dir=tmp_path / "manifests",
        checkpoint_dir=tmp_path / "checkpoints",
        python_executable="python",
        device="cpu",
    )

    assert summary["cells"] == list(PRIMARY_CELLS)
    assert summary["seeds"] == list(REGISTERED_OPTIMIZATION_SEEDS)
    assert summary["schedule"]["optimization_steps"] == 768
    assert len(summary["manifests"]) == 9

    first = Path(
        summary["manifests"]["C2_seed_20260614"]["manifest_path"]
    )
    manifest = json.loads(first.read_text())

    assert verify_manifest_files(manifest)["passes"]
    assert manifest["config"]["cell"] == "C2"
    assert manifest["config"]["run_class"] == "confirmatory"
    assert manifest["config"]["checkpoint_rule"]
    assert "inputs.trainer_script" in verify_manifest_files(manifest)["files"]
    assert "inputs.phase2d_training_module" in verify_manifest_files(manifest)["files"]
    assert "inputs.rollout_diagnostics_module" in verify_manifest_files(manifest)["files"]
    assert "--optimization-steps 768" in manifest["run_command"]
    assert "--evaluation-interval 256" in manifest["run_command"]
    assert "--source-states-per-batch 2" in manifest["run_command"]


def test_create_phase2d_training_run_manifests_freezes_optimizer_amendments(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n")
    validation.write_text("{}\n")
    split_manifest = _split_manifest(tmp_path / "split_manifest.json", train, validation)

    summary = create_phase2d_training_run_manifests(
        repository_root=REPO_ROOT,
        split_manifest_path=split_manifest,
        train_data_path=train,
        validation_data_path=validation,
        output_dir=tmp_path / "manifests",
        checkpoint_dir=tmp_path / "checkpoints",
        cells=("C2",),
        seeds=(20260614,),
        python_executable="python",
        device="cpu",
        learning_rate=1e-4,
        max_grad_norm=1.0,
        detach_action_control_state=True,
        target_geometry="slot",
        num_target_slots=8,
        consequence_loss_lambda=0.25,
        action_utility_loss_lambda=0.75,
        action_utility_regression_weight=0.2,
    )
    manifest_path = Path(
        summary["manifests"]["C2_seed_20260614"]["manifest_path"]
    )
    manifest = json.loads(manifest_path.read_text())

    assert summary["optimizer"]["learning_rate"] == 1e-4
    assert summary["optimizer"]["max_grad_norm"] == 1.0
    assert summary["model_amendments"]["detach_action_control_state"]
    assert summary["model_amendments"]["target_geometry"] == "slot"
    assert summary["model_amendments"]["num_target_slots"] == 8
    assert summary["model_amendments"]["consequence_dim"] == 9
    assert summary["model_amendments"]["consequence_loss_lambda"] == 0.25
    assert summary["model_amendments"]["action_utility_loss_lambda"] == 0.75
    assert summary["model_amendments"]["action_utility_regression_weight"] == 0.2
    assert summary["model_amendments"]["action_utility_target_version"]
    assert manifest["config"]["optimizer"]["learning_rate"] == 1e-4
    assert manifest["config"]["optimizer"]["max_grad_norm"] == 1.0
    assert manifest["config"]["model_amendments"]["detach_action_control_state"]
    assert manifest["config"]["model_amendments"]["target_geometry"] == "slot"
    assert manifest["config"]["model_amendments"]["num_target_slots"] == 8
    assert manifest["config"]["model_amendments"]["consequence_dim"] == 9
    assert (
        manifest["config"]["model_amendments"]["consequence_loss_lambda"]
        == 0.25
    )
    assert (
        manifest["config"]["model_amendments"]["action_utility_loss_lambda"]
        == 0.75
    )
    assert (
        manifest["config"]["model_amendments"]["action_utility_regression_weight"]
        == 0.2
    )
    assert manifest["config"]["model_amendments"]["action_utility_target_version"]
    assert "--lr 0.0001" in manifest["run_command"]
    assert "--max-grad-norm 1" in manifest["run_command"]
    assert "--detach-action-control-state" in manifest["run_command"]
    assert "--target-geometry slot" in manifest["run_command"]
    assert "--num-target-slots 8" in manifest["run_command"]
    assert "--consequence-loss-lambda 0.25" in manifest["run_command"]
    assert "--action-utility-loss-lambda 0.75" in manifest["run_command"]
    assert "--action-utility-regression-weight 0.2" in manifest["run_command"]
