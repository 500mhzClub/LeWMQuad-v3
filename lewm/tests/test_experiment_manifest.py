from __future__ import annotations

from pathlib import Path

from lewm.benchmarks.experiment_manifest import (
    artifact_record,
    build_experiment_manifest,
    sha256_file,
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
