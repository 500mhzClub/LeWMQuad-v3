from __future__ import annotations

from pathlib import Path

from lewm.benchmarks.experiment_manifest import (
    artifact_record,
    build_experiment_manifest,
    sha256_file,
    sha256_json,
    verify_manifest_files,
)


def test_artifact_record_is_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("research\n")

    record = artifact_record(path)

    assert record["sha256"] == sha256_file(path)
    assert record["size_bytes"] == len("research\n")


def test_manifest_verification_detects_artifact_change(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    artifact_path = tmp_path / "result.json"
    input_path.write_text("{}\n")
    artifact_path.write_text("{}\n")
    manifest = build_experiment_manifest(
        experiment_id="phase2_test",
        repository_root=tmp_path,
        inputs={"train": input_path},
        artifacts={"result": artifact_path},
        config={"cell": "C0"},
        seeds=[1, 2, 3],
        run_command="python train.py",
    )

    assert verify_manifest_files(manifest)["passes"]

    artifact_path.write_text('{"changed": true}\n')
    verification = verify_manifest_files(manifest)

    assert not verification["passes"]
    assert not verification["files"]["artifacts.result"]["matches"]


def test_manifest_records_research_dependencies_and_scene_splits(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "geometry.json"
    corpus = tmp_path / "corpus.json"
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "input.jsonl"
    for path in (geometry, corpus, checkpoint, input_path):
        path.write_text(path.name)

    manifest = build_experiment_manifest(
        experiment_id="generalization_gate",
        repository_root=tmp_path,
        inputs={"input": input_path},
        config={"threshold": 0.9},
        scene_splits={"train": ["scene-a"], "development": ["scene-b"]},
        geometry_contract=geometry,
        corpus_plan=corpus,
        checkpoints={"encoder": checkpoint},
        runtime_contract={"inputs": ["rgb", "odometry"]},
    )

    assert manifest["schema"] == "lewm_experiment_manifest_v1"
    assert manifest["config_sha256"] == sha256_json({"threshold": 0.9})
    assert manifest["scene_splits"]["sha256"] == sha256_json(
        {"train": ["scene-a"], "development": ["scene-b"]}
    )
    assert manifest["runtime_contract"]["inputs"] == ["rgb", "odometry"]
    assert verify_manifest_files(manifest)["passes"]


def test_manifest_rejects_scene_leakage(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n")

    try:
        build_experiment_manifest(
            experiment_id="leaked",
            repository_root=tmp_path,
            inputs={"input": input_path},
            scene_splits={"train": ["same"], "validation": ["same"]},
        )
    except ValueError as exc:
        assert "appears in both" in str(exc)
    else:
        raise AssertionError("scene leakage was accepted")
