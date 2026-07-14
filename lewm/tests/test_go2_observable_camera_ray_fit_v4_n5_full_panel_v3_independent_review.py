"""Independent lifecycle conformance review for full-panel V3.

All dynamic checks use temporary paths.  The canonical review, output, exact
training, dataset, RGB, protected roles, accelerators, and hardware stay closed.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v3 as policy,
)
from lewm.tests.n5_full_panel_v3_synthetic_execution import SyntheticExecutionV3
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3 as executor


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_plan_v2_qa"
FROZEN_ARTIFACTS = {
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_implementation_handoff_2026-07-13.md": (
        "c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db"
    ),
    ROOT / "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3"
    ),
    ROOT / "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66"
    ),
    ROOT / "lewm/tests/n5_full_panel_v3_synthetic_execution.py": (
        "83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91"
    ),
    ROOT / "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee"
    ),
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review_2026-07-13.md": (
        "24953fc64da151a6ff1f4ad89e5465e1caae300223556702e0f5c8430d47ee04"
    ),
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review_block_2026-07-13.json": (
        "ddca89e467e4cc30e52bacf57b28c040465e712843fde465f472f3cc8b38fc73"
    ),
    ROOT
    / "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review.py": (
        "a53c5e5d351784ff2a4824231998194e15040597897411c91e7727ec73a95e69"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _caller_reservation(directory: Path) -> executor.AttemptReservation:
    raw = b'{"caller_supplied":true}\n'
    return executor.AttemptReservation(
        directory=directory,
        value={"content_sha256": "1" * 64},
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _caller_result() -> dict[str, Any]:
    return {
        "schema": "caller_supplied_result",
        "caller_supplied": True,
        "content_sha256": "2" * 64,
    }


def _review_bytes() -> tuple[bytes, str]:
    sources = {
        relative: {
            "path": relative,
            "file_sha256": _sha(ROOT / relative),
        }
        for relative in policy.SUCCESSOR_SOURCE_PATHS
    }
    core = policy.expected_source_review_core(
        reviewer=REVIEWER,
        successor_sources=sources,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    return raw, hashlib.sha256(raw).hexdigest()


def test_v3_independent_review_frozen_candidate_and_parent_hashes() -> None:
    for path, expected in FROZEN_ARTIFACTS.items():
        assert _sha(path) == expected
    assert policy.V1_BLOCK_CONTENT_SHA256 == (
        "99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7"
    )
    assert policy.V2_BLOCK_CONTENT_SHA256 == (
        "c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a"
    )


def test_v3_canonical_review_and_output_remain_absent_and_exact_fails_closed() -> None:
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
    with pytest.raises(PermissionError, match="requires isolation"):
        executor.execute_exact("0" * 64, rgb_workers=5)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_v3_production_module_exposes_no_constructible_stage_evidence_object() -> None:
    exposed = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type)
        and value.__module__ == executor.__name__
        and any(term in name.casefold() for term in ("reservation", "completion", "attempt"))
    }
    assert exposed == set(), (
        "the documented single-operation surface exposes constructible stage "
        f"evidence classes: {sorted(exposed)}"
    )


def test_v3_reservation_copies_reconstruction_and_mutation_are_not_accepted(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "attempt"
    directory.mkdir()
    original = _caller_reservation(directory)
    copies = (
        copy.copy(original),
        copy.deepcopy(original),
        replace(original),
        executor.AttemptReservation(
            directory=original.directory,
            value=dict(original.value),
            raw=original.raw,
            file_sha256=original.file_sha256,
        ),
    )
    mutable_value = dict(original.value)
    mutable = executor.AttemptReservation(
        directory=directory,
        value=mutable_value,
        raw=original.raw,
        file_sha256=original.file_sha256,
    )
    mutable_value["content_sha256"] = "3" * 64
    accepted = [item.binding for item in (*copies, mutable)]
    assert accepted == [], "reconstructed or caller-mutated reservations were accepted"


def test_v3_completion_writer_rejects_caller_supplied_stage_inputs(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "attempt"
    directory.mkdir()
    reservation = _caller_reservation(directory)
    with pytest.raises(PermissionError, match="isolated|lifecycle|provenance"):
        executor._publish_success(
            reservation,
            checkpoint_raw=b"caller checkpoint",
            checkpoint_content_sha256="4" * 64,
            result=_caller_result(),
        )
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("kind", ("metric", "gate"))
def test_v3_metric_and_gate_writers_reject_caller_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    metric = output / "metric_verifications" / "seed_20260710_n5.json"
    gate = output / "gates" / "seed_20260710_n5.json"
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    monkeypatch.setattr(policy, "CANONICAL_METRIC_RECEIPT_PATH", metric)
    monkeypatch.setattr(policy, "CANONICAL_GATE_PATH", gate)
    path = metric if kind == "metric" else gate
    value = {
        "schema": f"caller_supplied_{kind}",
        "caller_supplied": True,
        "content_sha256": "5" * 64,
    }
    with pytest.raises(PermissionError, match="isolated|lifecycle|provenance"):
        executor._write_canonical_json(path, value)
    assert not path.exists()


def test_v3_publication_requires_the_original_claimed_directory_identity(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    reservation = _caller_reservation(attempt)
    moved = tmp_path / "moved-original-attempt"
    attempt.rename(moved)
    attempt.mkdir()
    try:
        executor._publish_success(
            reservation,
            checkpoint_raw=b"caller checkpoint",
            checkpoint_content_sha256="6" * 64,
            result=_caller_result(),
        )
    except (PermissionError, RuntimeError, ValueError):
        return
    assert list(attempt.iterdir()) == [], (
        "publication continued after the claimed directory identity changed"
    )


def test_v3_review_preflight_rejects_a_canonical_leaf_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, digest = _review_bytes()
    outside = tmp_path / "review-storage" / "review.json"
    outside.parent.mkdir()
    outside.write_bytes(raw)
    canonical = tmp_path / "canonical-review.json"
    canonical.symlink_to(outside)
    monkeypatch.setattr(policy, "CANONICAL_SOURCE_REVIEW_PATH", canonical)
    try:
        policy.preflight_source_review(canonical, digest)
    except PermissionError:
        return
    assert not canonical.is_symlink(), "a leaf alias was accepted as the canonical review"


def test_v3_source_reader_rejects_parent_identity_change_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    parent = root / "source"
    parent.mkdir(parents=True)
    source = parent / "candidate.py"
    source.write_bytes(b"reviewed source bytes\n")
    original_open = os.open
    changed = False

    def change_parent_then_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal changed
        if not changed and Path(path) == source:
            changed = True
            outside = tmp_path / "moved-source-parent"
            parent.rename(outside)
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(policy.os, "open", change_parent_then_open)
    try:
        policy.read_regular_bytes(source, name="reviewed source")
    except (PermissionError, RuntimeError, OSError):
        return
    assert source.resolve(strict=True).is_relative_to(root), (
        "source read continued after its parent identity changed"
    )


def test_v3_synthetic_lifecycle_remains_separate_recoverable_and_single_use(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV3(tmp_path / "synthetic")
    staging = operation.prepare_complete_staging()
    reservation = operation.claim()
    assert not staging.exists()
    assert reservation.directory == operation.attempt
    assert reservation.value["production_eligible"] is False
    with pytest.raises(FileExistsError, match="already claimed"):
        copy.deepcopy(operation).claim()
    for forbidden in (ROOT, ROOT / "lewm", policy.CANONICAL_OUTPUT_ROOT, ROOT.parent):
        with pytest.raises(PermissionError):
            SyntheticExecutionV3(forbidden)


def test_v3_production_claim_source_has_rename_fsync_and_inode_checks() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    reserve = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reserve_exact_attempt"
    )
    source = ast.unparse(reserve)
    assert "owned_claim_identity" in source
    assert "os.rename(active_staging, attempt_path)" in source
    assert source.index("os.rename(active_staging, attempt_path)") < source.index(
        "_fsync_directory(seed_root)"
    )
    assert "attempt_metadata.st_dev" in source
    assert "attempt_metadata.st_ino" in source


def test_v3_retained_training_rehashes_inputs_rgb_and_sources_before_publication() -> None:
    retained_path = ROOT / policy.RETAINED_V1_SOURCE_BINDINGS[2][0]
    source = retained_path.read_text()
    post_inputs = source.index("base.revalidate_exact_inputs_after_training(")
    post_rgb = source.index("revalidate_selected_rgb_before_publication(", post_inputs)
    checkpoint = source.index("base._checkpoint_bytes(", post_rgb)
    publication = source.index("_publish_success(", checkpoint)
    assert post_inputs < post_rgb < checkpoint < publication

    executor_source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert "policy.preflight_source_review(" in executor_source
    assert "finally:" in executor_source
    assert "retained.policy = original[\"policy\"]" in executor_source
    assert "verifier.policy = original_policy" in executor_source


def test_v3_isolated_launcher_keeps_gpu_and_process_contract_fixed() -> None:
    source = inspect.getsource(executor._isolated_child)
    assert '[sys.executable, "-I", "-B"' in source
    assert 'environment["HIP_VISIBLE_DEVICES"] = "0"' in source
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in source
    assert "PYTHONPATH" in source
    assert executor.run_cpu_contract_smoke() == {
        "schedule_sha256": policy.EXPECTED_SCHEDULE_SHA256,
        "update_count": 400,
        "frame_exposures": 2000,
        "every_update_is_full_panel": True,
        "losses": {
            "ordered_first_hit_nll": 0.8,
            "target_bin_offset_smooth_l1": 0.02,
            "ground_clear_distance_state_balanced_bce": 0.04,
            "derived_raster_hierarchical_bce": 0.2,
            "total": 0.265,
        },
    }


def test_v3_review_did_not_create_canonical_state() -> None:
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
