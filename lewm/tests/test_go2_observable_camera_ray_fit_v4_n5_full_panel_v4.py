"""Author closure for the additive N5 full-panel V4 executor."""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v4 as policy,
)
from lewm.tests.n5_full_panel_v4_synthetic_execution import SyntheticExecutionV4
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4 as executor


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V3 = {
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_implementation_handoff_2026-07-13.md": "c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": "b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3",
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": "8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66",
    "lewm/tests/n5_full_panel_v3_synthetic_execution.py": "83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": "730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_independent_review.py": "b7d3669135f22311e13c840e04c4ec2ed583365fc77f7fce6c5c0ecc4e512395",
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_independent_review_2026-07-13.md": "d28eadce56668b0cf793806bb98e7c793eb9d874b7ca818d4d9b3c3205fe53e7",
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_independent_review_block_2026-07-13.json": "d1f859aea2a80f090c3ee09df5194f5b4bcfca22865f323de543f3b216b3e168",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporary_review(root: Path) -> tuple[bytes, str]:
    sources: dict[str, dict[str, str]] = {}
    for index, relative in enumerate(policy.SUCCESSOR_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"temporary reviewed source {index}\n".encode("ascii"))
        sources[relative] = {"path": relative, "file_sha256": _sha(path)}
    core = policy.expected_source_review_core(
        reviewer="/root/different_agent",
        successor_sources=sources,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    return raw, hashlib.sha256(raw).hexdigest()


def test_v4_frozen_v3_and_all_static_parents_rehash() -> None:
    for relative, expected in FROZEN_V3.items():
        assert _sha(ROOT / relative) == expected
    static = policy.preflight_static_authority()
    assert static["v1_block_content_sha256"] == policy.V1_BLOCK_CONTENT_SHA256
    assert static["v2_block_content_sha256"] == policy.V2_BLOCK_CONTENT_SHA256
    assert static["v3_block_content_sha256"] == policy.V3_BLOCK_CONTENT_SHA256


def test_v4_import_exposes_no_callable_operation_and_performs_no_lifecycle_work() -> None:
    defined_functions = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, FunctionType) and value.__module__ == executor.__name__
    }
    defined_classes = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type) and value.__module__ == executor.__name__
    }
    assert defined_functions == set()
    assert defined_classes == set()
    assert not hasattr(executor, "main")
    assert not hasattr(executor, "execute_exact")
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_v4_production_source_defines_partial_stages_only_inside_script_entry() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    top_functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert top_functions == []
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    nested = {
        node.name
        for node in ast.walk(entry)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {
        "AttemptReservationV4",
        "_reserve_exact_attempt",
        "_publish_success",
        "_write_canonical_json",
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
        "execute_exact",
    } <= nested


def test_v4_no_constructible_stage_evidence_class_is_importable() -> None:
    exposed = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type)
        and value.__module__ == executor.__name__
        and any(term in name.casefold() for term in ("reservation", "attempt", "completion"))
    }
    assert exposed == set()


def test_v4_reservation_copy_reconstruction_and_mutation_surface_is_absent() -> None:
    assert not hasattr(executor, "AttemptReservationV4")
    assert not any("reservation" in name.casefold() for name in vars(executor))


def test_v4_completion_writer_is_not_importable() -> None:
    assert not hasattr(executor, "_publish_success")
    assert not hasattr(executor, "_terminate_failure")


@pytest.mark.parametrize("name", ["_write_canonical_json", "_write_bytes_exclusive"])
def test_v4_metric_and_gate_writers_are_not_importable(name: str) -> None:
    assert not hasattr(executor, name)


def test_v4_publication_rejects_replaced_claim_directory(tmp_path: Path) -> None:
    operation = SyntheticExecutionV4(tmp_path / "synthetic")
    reservation = operation.claim()
    moved = tmp_path / "moved-original-attempt"
    operation.attempt.rename(moved)
    operation.attempt.mkdir()
    try:
        with pytest.raises(PermissionError, match="claim identity"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        assert list(operation.attempt.iterdir()) == []
        assert sorted(path.name for path in moved.iterdir()) == ["reservation.json"]
    finally:
        operation.close(reservation)


def test_v4_review_preflight_rejects_canonical_leaf_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_root = tmp_path / "repo"
    temporary_root.mkdir()
    raw, digest = _temporary_review(temporary_root)
    storage = temporary_root / "storage/review.json"
    storage.parent.mkdir()
    storage.write_bytes(raw)
    canonical = temporary_root / policy.SOURCE_REVIEW_RELATIVE_PATH
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(storage)
    monkeypatch.setattr(policy, "ROOT", temporary_root)
    monkeypatch.setattr(policy, "CANONICAL_SOURCE_REVIEW_PATH", canonical)

    with pytest.raises((PermissionError, OSError)):
        policy.preflight_source_review(canonical, digest)
    assert canonical.is_symlink()


def test_v4_review_preflight_accepts_only_the_regular_canonical_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_root = tmp_path / "repo"
    temporary_root.mkdir()
    raw, digest = _temporary_review(temporary_root)
    canonical = temporary_root / policy.SOURCE_REVIEW_RELATIVE_PATH
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(raw)
    monkeypatch.setattr(policy, "ROOT", temporary_root)
    monkeypatch.setattr(policy, "CANONICAL_SOURCE_REVIEW_PATH", canonical)

    review, reread = policy.preflight_source_review(canonical, digest)
    assert reread == raw
    assert review["reviewer"] == "/root/different_agent"


def test_v4_source_reader_rejects_parent_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    parent = root / "source"
    parent.mkdir(parents=True)
    (parent / "candidate.py").write_bytes(b"reviewed source bytes\n")
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
        if not changed and path == "candidate.py" and dir_fd is not None:
            changed = True
            moved = tmp_path / "moved-source-parent"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(policy.os, "open", change_parent_then_open)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, "source/candidate.py", name="reviewed source")
    assert changed is True


def test_v4_source_reader_rejects_symlink_hardlink_and_fifo(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = root / "source.py"
    original.write_bytes(b"source\n")
    alias = root / "alias.py"
    alias.symlink_to(original)
    with pytest.raises((PermissionError, OSError)):
        policy.read_regular_bytes_at(root, "alias.py", name="leaf alias")

    hardlink = root / "hardlink.py"
    os.link(original, hardlink)
    with pytest.raises(PermissionError, match="singly-linked"):
        policy.read_regular_bytes_at(root, "source.py", name="hardlinked source")

    fifo = root / "source.fifo"
    os.mkfifo(fifo)
    with pytest.raises(PermissionError, match="singly-linked"):
        policy.read_regular_bytes_at(root, "source.fifo", name="FIFO source")


@pytest.mark.parametrize(
    "unsafe",
    [ROOT, ROOT / "lewm", policy.CANONICAL_OUTPUT_ROOT, ROOT.parent],
)
def test_v4_synthetic_operation_rejects_production_namespaces(unsafe: Path) -> None:
    with pytest.raises(PermissionError, match="production|repository"):
        SyntheticExecutionV4(unsafe)


def test_v4_synthetic_recovery_is_single_use_and_descriptor_bound(tmp_path: Path) -> None:
    operation = SyntheticExecutionV4(tmp_path / "safe")
    staged = operation.prepare_complete_staging()
    reservation = operation.claim()
    try:
        assert not staged.exists()
        assert reservation.value["production_eligible"] is False
        operation.publish(reservation, b'{"synthetic":true}\n')
        assert sorted(path.name for path in operation.attempt.iterdir()) == [
            "completed.json",
            "reservation.json",
        ]
        with pytest.raises(FileExistsError, match="already claimed"):
            copy.deepcopy(operation).claim()
    finally:
        operation.close(reservation)


def test_v4_concurrent_synthetic_claims_choose_exactly_one(tmp_path: Path) -> None:
    operations = [SyntheticExecutionV4(tmp_path / "safe") for _ in range(2)]

    def claim(operation: SyntheticExecutionV4) -> object:
        try:
            return operation.claim()
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, operations))
    reservations = [value for value in outcomes if value != "rejected"]
    assert len(reservations) == 1
    SyntheticExecutionV4.close(reservations[0])  # type: ignore[arg-type]


def test_v4_preclaim_cleanup_and_postclaim_failure_are_durable(tmp_path: Path) -> None:
    before = SyntheticExecutionV4(tmp_path / "before")
    with pytest.raises(RuntimeError, match="before atomic"):
        before.claim(failure_injection="before_atomic_claim")
    assert not before.attempt.exists()

    after = SyntheticExecutionV4(tmp_path / "after")
    with pytest.raises(RuntimeError, match="after atomic"):
        after.claim(failure_injection="after_atomic_claim")
    assert sorted(path.name for path in after.attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    with pytest.raises(FileExistsError, match="already claimed"):
        SyntheticExecutionV4(after.root).claim()


def test_v4_claim_fd_and_final_close_are_present_in_production_source() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    reserve = functions["_reserve_exact_attempt"]
    assert "os.open(active_staging" in reserve
    assert "os.fstat(claimed_directory_fd)" in reserve
    assert reserve.index("os.open(active_staging") < reserve.index(
        "os.rename(active_staging, attempt_path)"
    )
    assert reserve.index("os.rename(active_staging, attempt_path)") < reserve.index(
        "_fsync_directory(seed_root)"
    )
    assert "directory_fd=claimed_directory_fd" in reserve
    assert "_write_claim_file_exclusive" in functions["_publish_success"]
    assert "_read_claim_file" in functions["_artifact_args"]
    assert "os.close(reservation.directory_fd)" in functions["execute_exact"]


def test_v4_frozen_science_and_gpu0_launcher_are_unchanged() -> None:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    assert policy.experiment_contract() == retained.EXPERIMENT
    assert policy.authority_bindings() == retained.AUTHORITY_BINDINGS
    schedule = base._deterministic_training_batches(
        frame_count=5,
        batch_size=5,
        steps=400,
        seed=20260710,
    )
    assert base.canonical_json_sha256(schedule) == policy.EXPECTED_SCHEDULE_SHA256
    assert len(schedule) == 400
    assert sum(len(batch) for batch in schedule) == 2000
    assert all(len(batch) == 5 and set(batch) == set(range(5)) for batch in schedule)

    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert '[sys.executable, "-I", "-B"' in source
    assert 'environment["HIP_VISIBLE_DEVICES"] = "0"' in source
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in source
    assert 'environment[name] = "1"' in source


def test_v4_review_contract_binds_only_end_to_end_sources_and_v3_block() -> None:
    sources = {
        relative: {"path": relative, "file_sha256": _sha(ROOT / relative)}
        for relative in policy.SUCCESSOR_SOURCE_PATHS
    }
    core = policy.expected_source_review_core(
        reviewer="/root/different_agent",
        successor_sources=sources,
    )
    assert core["implementation_author"] == policy.IMPLEMENTATION_AUTHOR
    assert core["reviewer"] != policy.IMPLEMENTATION_AUTHOR
    assert core["v3_block_evidence"]["block_content_sha256"] == (
        policy.V3_BLOCK_CONTENT_SHA256
    )
    assert core["execution_contract"]["importable_partial_stage"] is False
    assert core["execution_contract"]["stage_values_constructed_inside_script_entry"] is True
    assert core["reservation_contract"][
        "claimed_directory_descriptor_retained_end_to_end"
    ] is True
    assert set(core["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
