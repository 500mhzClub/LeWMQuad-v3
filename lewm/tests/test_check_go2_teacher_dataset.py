from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "claim_checker_test_support_teacher",
    ROOT / "lewm/tests/test_go2_claim_checker_manifest_binding.py",
)
assert SUPPORT_SPEC is not None and SUPPORT_SPEC.loader is not None
SUPPORT = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(SUPPORT)


def test_teacher_checker_accepts_external_manifest_recomputation(tmp_path: Path) -> None:
    manifest = SUPPORT._manifest(task_count=4)
    code, report = SUPPORT._run_checker(
        tmp_path, "teacher", SUPPORT._canonical_result(manifest), manifest
    )
    assert code == 0, report
    assert report["gates"]["canonical_physical_claims"] is True
    assert report["gates"]["scene_manifest_match"] is True


@pytest.mark.parametrize("mutation", SUPPORT.MUTATIONS)
def test_teacher_checker_rejects_each_canonical_claim_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest = SUPPORT._manifest(task_count=4)
    result, supplied_manifest = SUPPORT._mutate(
        SUPPORT._canonical_result(manifest), manifest, mutation
    )
    code, report = SUPPORT._run_checker(
        tmp_path, "teacher", result, supplied_manifest
    )
    assert code == 1, (mutation, report)
    assert not (
        report["gates"]["canonical_physical_claims"]
        and report["gates"]["scene_manifest_match"]
    )
