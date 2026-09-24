from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import finalize_go2_dynamic_cartesian_n32 as finalizer


def _write_json(path: Path, value: object) -> str:
    data = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_finalizer_import_is_torch_free() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(finalizer.REPOSITORY_ROOT)
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; "
            "from scripts import finalize_go2_dynamic_cartesian_n32; "
            "assert 'torch' not in sys.modules",
        ),
        cwd=finalizer.REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_path_rejects_relative_and_alias_components(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical absolute"):
        finalizer._canonical_absolute_path(Path("relative.json"), name="test")
    aliased = tmp_path / "directory" / ".." / "result.json"
    with pytest.raises(ValueError, match="canonical absolute"):
        finalizer._canonical_absolute_path(aliased, name="test")


def test_regular_reader_rejects_symlinked_file_and_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    payload = real_parent / "payload.json"
    payload.write_bytes(b"{}\n")

    file_link = real_parent / "file-link.json"
    file_link.symlink_to(payload)
    with pytest.raises(OSError):
        finalizer._read_regular_file(file_link, name="file link")

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        finalizer._read_regular_file(
            parent_link / payload.name,
            name="parent link",
        )


def test_seed_specific_primary_arguments_are_fail_closed() -> None:
    empty = argparse.Namespace(
        seed_20260710_result=None,
        expected_seed_20260710_result_sha256=None,
        seed_20260710_attempt_marker=None,
        expected_seed_20260710_attempt_marker_sha256=None,
    )
    finalizer._validate_primary_arguments(empty, seed=20260710)
    with pytest.raises(ValueError, match="requires"):
        finalizer._validate_primary_arguments(empty, seed=20260711)

    supplied = argparse.Namespace(
        seed_20260710_result=Path("/tmp/seed10.json"),
        expected_seed_20260710_result_sha256="a" * 64,
        seed_20260710_attempt_marker=Path("/tmp/seed10-attempt.json"),
        expected_seed_20260710_attempt_marker_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="rejects"):
        finalizer._validate_primary_arguments(supplied, seed=20260710)
    finalizer._validate_primary_arguments(supplied, seed=20260711)


def test_seed11_finalizer_validates_primary_first_and_creates_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    implementation_path = tmp_path / "implementation.json"
    primary_path = tmp_path / "seed10.json"
    replication_path = tmp_path / "seed11.json"
    primary_attempt_path = tmp_path / "seed10-attempt.json"
    replication_attempt_path = tmp_path / "seed11-attempt.json"
    implementation_sha = _write_json(implementation_path, {"manifest": True})
    primary_sha = _write_json(
        primary_path,
        {"content_sha256": "1" * 64, "seed": 20260710},
    )
    replication_sha = _write_json(
        replication_path,
        {"content_sha256": "2" * 64, "seed": 20260711},
    )
    primary_attempt_sha = _write_json(
        primary_attempt_path,
        {"content_sha256": "3" * 64, "seed": 20260710},
    )
    replication_attempt_sha = _write_json(
        replication_attempt_path,
        {"content_sha256": "4" * 64, "seed": 20260711},
    )
    monkeypatch.setattr(finalizer, "IMPLEMENTATION_MANIFEST_PATH", implementation_path)
    monkeypatch.setattr(
        finalizer,
        "CANONICAL_RESULT_PATHS",
        {20260710: primary_path, 20260711: replication_path},
    )
    monkeypatch.setattr(
        finalizer,
        "CANONICAL_ATTEMPT_MARKER_PATHS",
        {20260710: primary_attempt_path, 20260711: replication_attempt_path},
    )
    monkeypatch.setattr(
        finalizer.contract,
        "validate_implementation_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(
        finalizer,
        "_verify_manifest_sources",
        lambda _manifest: {"finalizer": "f" * 64},
    )
    calls: list[tuple[object, ...]] = []

    def validate_result(
        value: dict,
        expected_seed: int,
        implementation_manifest: dict,
        **kwargs: object,
    ) -> dict:
        calls.append(
            (
                "result",
                expected_seed,
                kwargs.get("implementation_manifest_file_sha256"),
                kwargs.get("attempt_marker"),
                kwargs.get("attempt_marker_file_sha256"),
                kwargs.get("primary_result"),
                kwargs.get("primary_file_sha256"),
                kwargs.get("primary_attempt_marker"),
                kwargs.get("primary_attempt_marker_file_sha256"),
            )
        )
        return {
            **value,
            "decision": {
                "classification": "favorable",
                "favorable": True,
            },
        }

    def validate_pair(
        primary: dict,
        replication: dict,
        implementation_manifest: dict,
        implementation_manifest_file_sha256: str,
        external_primary_sha256: str,
        primary_attempt_marker: dict,
        primary_attempt_marker_file_sha256: str,
        replication_attempt_marker: dict,
        replication_attempt_marker_file_sha256: str,
    ) -> dict:
        calls.append(
            (
                "pair",
                primary["seed"],
                replication["seed"],
                implementation_manifest_file_sha256,
                external_primary_sha256,
                primary_attempt_marker_file_sha256,
                replication_attempt_marker_file_sha256,
            )
        )
        return {"shared_jepa_construction_licensed": True}

    monkeypatch.setattr(
        finalizer.contract,
        "validate_authoritative_result",
        validate_result,
    )
    monkeypatch.setattr(finalizer.contract, "validate_seed_pair", validate_pair)
    before = sorted(tmp_path.iterdir())
    assert (
        finalizer.main(
            (
                "--result",
                str(replication_path),
                "--expected-result-sha256",
                replication_sha,
                "--implementation-manifest",
                str(implementation_path),
                "--expected-implementation-manifest-sha256",
                implementation_sha,
                "--attempt-marker",
                str(replication_attempt_path),
                "--expected-attempt-marker-sha256",
                replication_attempt_sha,
                "--seed-20260710-result",
                str(primary_path),
                "--expected-seed-20260710-result-sha256",
                primary_sha,
                "--seed-20260710-attempt-marker",
                str(primary_attempt_path),
                "--expected-seed-20260710-attempt-marker-sha256",
                primary_attempt_sha,
            )
        )
        == 0
    )
    assert calls[0] == (
        "result",
        20260710,
        implementation_sha,
        json.loads(primary_attempt_path.read_text()),
        primary_attempt_sha,
        None,
        None,
        None,
        None,
    )
    assert calls[1] == (
        "result",
        20260711,
        implementation_sha,
        json.loads(replication_attempt_path.read_text()),
        replication_attempt_sha,
        json.loads(primary_path.read_text()),
        primary_sha,
        json.loads(primary_attempt_path.read_text()),
        primary_attempt_sha,
    )
    assert calls[2] == (
        "pair",
        20260710,
        20260711,
        implementation_sha,
        primary_sha,
        primary_attempt_sha,
        replication_attempt_sha,
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation_only"] is True
    assert summary["output_file_created"] is False
    assert summary["shared_jepa_construction_licensed"] is True
    assert sorted(tmp_path.iterdir()) == before


def test_expected_result_file_hash_is_checked_before_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    _write_json(path, {"seed": 20260710})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        finalizer._read_expected_json(
            path,
            expected_sha256="0" * 64,
            name="result",
        )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        finalizer._strict_json(b'{"seed":20260710,"seed":20260711}', name="result")
