from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import (
    launch_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1 as launcher,
)


def _bound(value: dict) -> dict:
    raw = launcher._canonical_bytes(value)
    return {**value, "content_sha256": hashlib.sha256(raw).hexdigest()}


def _write_json(path: Path, value: dict) -> dict:
    raw = launcher._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": launcher.CERTIFICATION_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "content_sha256": value["content_sha256"],
    }


def test_certified_source_rehashes_exact_inventory(tmp_path: Path) -> None:
    source = tmp_path / "lewm" / "models" / "tiny.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    raw = source.read_bytes()
    binding = {
        "path": "lewm/models/tiny.py",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    certification = _bound(
        {
            "schema": (
                "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
                "clean_export_certification_v1"
            ),
            "status": "PASS_NARROW_CLEAN_EXPORT_CERTIFIED",
            "certified_source_root": str(tmp_path.resolve()),
            "pinned_source_and_review_commit": "a" * 40,
            "source_bindings": [binding],
            "bindings_sha256": hashlib.sha256(
                launcher._canonical_bytes([binding])
            ).hexdigest(),
        }
    )
    identity = _write_json(
        tmp_path / launcher.CERTIFICATION_RELATIVE_PATH, certification
    )
    receipt = launcher.validate_certified_source_v1(
        tmp_path,
        {
            "pinned_source_and_review_commit": "a" * 40,
            "clean_export_certification": identity,
        },
    )
    assert receipt["status"] == "PASS_CERTIFIED_SOURCE_REHASH"
    assert receipt["validated_path_count"] == 1


def test_certified_source_rejects_changed_bytes(tmp_path: Path) -> None:
    source = tmp_path / "scripts" / "tiny.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    raw = source.read_bytes()
    binding = {
        "path": "scripts/tiny.py",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    certification = _bound(
        {
            "schema": (
                "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
                "clean_export_certification_v1"
            ),
            "status": "PASS_NARROW_CLEAN_EXPORT_CERTIFIED",
            "certified_source_root": str(tmp_path.resolve()),
            "pinned_source_and_review_commit": "b" * 40,
            "source_bindings": [binding],
            "bindings_sha256": hashlib.sha256(
                launcher._canonical_bytes([binding])
            ).hexdigest(),
        }
    )
    identity = _write_json(
        tmp_path / launcher.CERTIFICATION_RELATIVE_PATH, certification
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="changed"):
        launcher.validate_certified_source_v1(
            tmp_path,
            {
                "pinned_source_and_review_commit": "b" * 40,
                "clean_export_certification": identity,
            },
        )


def test_gpu_guard_requires_one_r9700_without_allocating() -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "AMD Radeon AI PRO R9700"

    class Torch:
        cuda = Cuda()

    receipt = launcher.validate_gpu_v1(Torch())
    assert receipt["status"] == "PASS_EXACTLY_ONE_VISIBLE_R9700"
    assert receipt["tensor_allocation_count"] == 0
