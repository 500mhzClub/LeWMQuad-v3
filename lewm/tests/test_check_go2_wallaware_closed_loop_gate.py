from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "claim_checker_test_support_wallaware",
    ROOT / "lewm/tests/test_go2_claim_checker_manifest_binding.py",
)
assert SUPPORT_SPEC is not None and SUPPORT_SPEC.loader is not None
SUPPORT = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)


def test_wallaware_checker_accepts_exact_external_manifest_map(
    tmp_path: Path,
) -> None:
    manifest = SUPPORT._manifest(task_count=1)
    code, report = SUPPORT._run_checker(
        tmp_path, "wallaware", SUPPORT._canonical_result(manifest), manifest
    )
    assert code == 0, report
    binding = next(
        item
        for item in report["checks"]
        if item["name"] == "exact_scene_manifest_mapping"
    )
    assert binding["passed"] is True


@pytest.mark.parametrize("mutation", SUPPORT.MUTATIONS)
def test_wallaware_checker_rejects_each_canonical_claim_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest = SUPPORT._manifest(task_count=1)
    result, supplied_manifest = SUPPORT._mutate(
        SUPPORT._canonical_result(manifest), manifest, mutation
    )
    code, report = SUPPORT._run_checker(
        tmp_path, "wallaware", result, supplied_manifest
    )
    assert code == 1, (mutation, report)
    assert report["passed"] is False
