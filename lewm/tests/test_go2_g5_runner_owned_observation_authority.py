from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts import run_go2_g5_runner_owned_observation_v1 as runner


def test_production_identities_are_hard_unset_and_no_output_is_promoted() -> None:
    assert (
        runner.CANONICAL_G5_RUNNER_SOURCE_SHA256,
        runner.CANONICAL_G3_RUNNER_OUTCOME_FILE_SHA256,
        runner.CANONICAL_G5_RGB_FRAME_FILE_SHA256,
        runner.CANONICAL_G5_OBSERVATION_HEAD_CHECKPOINT_FILE_SHA256,
        runner.CANONICAL_G5_EPISODE_AUTHORITY_FILE_SHA256,
        runner.CANONICAL_G5_PROMOTED_OUTPUT_FILE_SHA256,
    ) == (None,) * 6
    assert set(inspect.signature(runner.run_one_shot).parameters) == set()


def test_pending_authority_fails_before_any_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("pending G5 authority attempted filesystem access")

    monkeypatch.setattr(Path, "open", forbidden_open)
    monkeypatch.setattr(Path, "read_bytes", forbidden_open)
    monkeypatch.setattr(Path, "write_bytes", forbidden_open)
    with pytest.raises(PermissionError, match="pending reviewed identities"):
        runner.run_one_shot()


def test_fresh_process_import_and_no_argument_run_precede_path_construction() -> None:
    probe = textwrap.dedent(
        """
        import importlib
        from pathlib import Path

        def forbidden_path(*args, **kwargs):
            raise AssertionError("pending G5 authority attempted path access")

        Path.__new__ = staticmethod(forbidden_path)
        Path.resolve = forbidden_path
        Path.open = forbidden_path
        Path.read_bytes = forbidden_path
        Path.write_bytes = forbidden_path

        runner = importlib.import_module(
            "scripts.run_go2_g5_runner_owned_observation_v1"
        )
        assert (
            runner.CANONICAL_G5_RUNNER_SOURCE_SHA256,
            runner.CANONICAL_G3_RUNNER_OUTCOME_FILE_SHA256,
            runner.CANONICAL_G5_RGB_FRAME_FILE_SHA256,
            runner.CANONICAL_G5_OBSERVATION_HEAD_CHECKPOINT_FILE_SHA256,
            runner.CANONICAL_G5_EPISODE_AUTHORITY_FILE_SHA256,
            runner.CANONICAL_G5_PROMOTED_OUTPUT_FILE_SHA256,
        ) == (None,) * 6
        try:
            runner.main()
        except PermissionError as error:
            assert "pending reviewed identities" in str(error)
        else:
            raise AssertionError("pending G5 authority did not fail closed")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_raw_interface_excludes_caller_selected_authority() -> None:
    fields = set(runner.RAW_OBSERVATION_INTERFACE_FIELDS)
    forbidden = set(runner.FORBIDDEN_CALLER_AUTHORITY_FIELDS)
    assert not fields & forbidden
    assert fields == {
        "schema",
        "scene_id",
        "episode_id",
        "tick",
        "rgb_frame_file_sha256",
        "g3_runner_outcome_file_sha256",
        "pose_receipt_sha256",
        "camera_calibration_sha256",
        "observation_head_checkpoint_file_sha256",
        "runner_source_sha256",
    }
