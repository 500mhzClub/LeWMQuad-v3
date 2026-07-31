from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
JOINER_PATH = ROOT / "scripts/join_go2_world_model_counterfactual_pilot_v1.py"


def _load_joiner():
    spec = importlib.util.spec_from_file_location(
        "counterfactual_pilot_joiner_blocked_v1", JOINER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_join_fails_closed_before_opening_any_input() -> None:
    joiner = _load_joiner()
    with pytest.raises(
        joiner.PilotJoinBlocked,
        match="160-branch calibration collection",
    ):
        joiner.join_pilot(
            collection_path=Path("/must/not/be/opened/collection.json"),
            expected_collection_sha256="0" * 64,
            expected_collection_byte_count=1,
            calibration_receipt_path=Path("/must/not/be/opened/calibration.json"),
            expected_calibration_sha256="1" * 64,
            expected_calibration_byte_count=1,
        )


def test_pilot_join_cli_fails_closed_before_opening_any_input() -> None:
    joiner = _load_joiner()
    with pytest.raises(
        joiner.PilotJoinBlocked,
        match="160-branch calibration collection",
    ):
        joiner.main(
            [
                "--collection",
                "/must/not/be/opened/collection.json",
                "--expected-collection-sha256",
                "0" * 64,
                "--expected-collection-byte-count",
                "1",
                "--calibration-receipt",
                "/must/not/be/opened/calibration.json",
                "--expected-calibration-sha256",
                "1" * 64,
                "--expected-calibration-byte-count",
                "1",
            ]
        )
