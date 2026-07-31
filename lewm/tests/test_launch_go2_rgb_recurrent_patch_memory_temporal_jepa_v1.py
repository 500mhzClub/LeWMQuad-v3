from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import (
    launch_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as launcher,
)


def _bound(value: dict) -> dict:
    return {
        **value,
        "content_sha256": hashlib.sha256(
            launcher._canonical_bytes(value)
        ).hexdigest(),
    }


def _write_certification(path: Path, value: dict) -> dict:
    raw = launcher._canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": launcher.CERTIFICATION_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "byte_count": len(raw),
    }


def _certification(
    tmp_path: Path,
    source_binding: dict,
    *,
    commit: str,
) -> tuple[dict, dict]:
    bindings = [source_binding]
    value = _bound(
        {
            "schema": (
                f"{launcher.executor.SCHEMA_PREFIX}_"
                "clean_export_certification_v1"
            ),
            "status": "PASS_NARROW_CLEAN_EXPORT_CERTIFIED",
            "certified_source_root": str(tmp_path.resolve()),
            "pinned_source_and_review_commit": commit,
            "source_bindings": bindings,
            "bindings_sha256": hashlib.sha256(
                launcher._canonical_bytes(bindings)
            ).hexdigest(),
        }
    )
    identity = _write_certification(
        tmp_path / launcher.CERTIFICATION_RELATIVE_PATH,
        value,
    )
    return value, identity


def test_certified_temporal_source_rehashes_exact_safe_inventory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "lewm" / "models" / "tiny.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    raw = source.read_bytes()
    binding = {
        "path": "lewm/models/tiny.py",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    commit = "a" * 40
    _value, identity = _certification(
        tmp_path,
        binding,
        commit=commit,
    )
    receipt = launcher.validate_certified_source_v1(
        tmp_path,
        {
            "pinned_source_and_review_commit": commit,
            "clean_export_certification": identity,
        },
    )
    assert receipt["status"] == "PASS_CERTIFIED_SOURCE_REHASH"
    assert receipt["validated_path_count"] == 1


def test_certified_temporal_source_rejects_mutation_and_unsafe_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scripts" / "tiny.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    raw = source.read_bytes()
    binding = {
        "path": "scripts/tiny.py",
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    commit = "b" * 40
    _value, identity = _certification(
        tmp_path,
        binding,
        commit=commit,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="changed"):
        launcher.validate_certified_source_v1(
            tmp_path,
            {
                "pinned_source_and_review_commit": commit,
                "clean_export_certification": identity,
            },
        )
    with pytest.raises(PermissionError, match="unsafe"):
        launcher._safe_source_path("sealed_hidden/source.py")


def test_gpu_guard_requires_one_visible_amd_r9700_without_allocation() -> None:
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

    class Version:
        hip = "7.1.1"

    class Torch:
        cuda = Cuda()
        version = Version()

    receipt = launcher.validate_gpu_v1(Torch())
    assert receipt["status"] == "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700"
    assert receipt["tensor_allocation_count"] == 0
    Torch.version.hip = None
    with pytest.raises(RuntimeError, match="AMD R9700"):
        launcher.validate_gpu_v1(Torch())
