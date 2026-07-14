"""Independent conformance review for camera full-panel V4.

Dynamic checks use temporary paths only. Exact training, experiment data, RGB,
accelerators, protected roles, and canonical experiment output remain closed.
"""
from __future__ import annotations

import ast
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
REVIEWER = "/root/coordinator_v2_qa"
FROZEN_ARTIFACTS = {
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": (
        "ff291b94b1546ae9ccf0b85de5f96b87edce4ad5b7992ca16bbbf13dcd1d4485"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": (
        "19cbdc5692911b31b3b44883b0cfefcc81daa4afc16250b89c1317dd9b66afe4"
    ),
    "lewm/tests/n5_full_panel_v4_synthetic_execution.py": (
        "01e49c303d0e2c8e76e7ecbdbd2d0cf159948a5f36a4dc6248d0e014d9c69fb5"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": (
        "299fd18b88a869916a916adc4e8848235e955447e9a1f245aeaeec6e7ee69688"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "implementation_handoff_2026-07-13.md": (
        "4e0aa7e2efa266feb774a4b095cbddca105cfd046aac7a0da7f942f1b2b6925e"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66"
    ),
    "lewm/tests/n5_full_panel_v3_synthetic_execution.py": (
        "83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py": (
        "730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "independent_review.py": (
        "b7d3669135f22311e13c840e04c4ec2ed583365fc77f7fce6c5c0ecc4e512395"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "implementation_handoff_2026-07-13.md": (
        "c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "independent_review_2026-07-13.md": (
        "d28eadce56668b0cf793806bb98e7c793eb9d874b7ca818d4d9b3c3205fe53e7"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_"
    "independent_review_block_2026-07-13.json": (
        "d1f859aea2a80f090c3ee09df5194f5b4bcfca22865f323de543f3b216b3e168"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_bytes(root: Path) -> tuple[bytes, str]:
    sources: dict[str, dict[str, str]] = {}
    for index, relative in enumerate(policy.SUCCESSOR_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"independent source {index}\n".encode("ascii"))
        sources[relative] = {"path": relative, "file_sha256": _sha(path)}
    core = policy.expected_source_review_core(
        reviewer=REVIEWER,
        successor_sources=sources,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    return raw, hashlib.sha256(raw).hexdigest()


def _function_nodes() -> dict[str, ast.FunctionDef]:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def test_v4_independent_freezes_candidate_and_v3_block_inputs() -> None:
    assert {
        relative: _sha(ROOT / relative)
        for relative in FROZEN_ARTIFACTS
    } == FROZEN_ARTIFACTS
    assert policy.V1_BLOCK_CONTENT_SHA256 == (
        "99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7"
    )
    assert policy.V2_BLOCK_CONTENT_SHA256 == (
        "c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a"
    )
    assert policy.V3_BLOCK_CONTENT_SHA256 == (
        "d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f"
    )


def test_v4_independent_import_has_no_partial_lifecycle_surface() -> None:
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
    for name in (
        "AttemptReservationV4",
        "_reserve_exact_attempt",
        "_publish_success",
        "_terminate_failure",
        "_write_canonical_json",
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
        "execute_exact",
    ):
        assert not hasattr(executor, name)


def test_v4_independent_stage_definitions_are_inside_only_script_entry() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
    entries = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    assert len(entries) == 1
    nested = {
        node.name
        for node in ast.walk(entries[0])
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {
        "AttemptReservationV4",
        "_reserve_exact_attempt",
        "_publish_success",
        "_write_canonical_json",
        "execute_exact",
        "dispatch",
    } <= nested


def test_v4_independent_v3_reservation_reconstruction_surface_is_absent() -> None:
    assert not any("reservation" in name.casefold() for name in vars(executor))


def test_v4_independent_v3_completion_writer_is_absent() -> None:
    assert not hasattr(executor, "_publish_success")
    assert not hasattr(executor, "_terminate_failure")


@pytest.mark.parametrize("name", ("_write_canonical_json", "_write_bytes_exclusive"))
def test_v4_independent_v3_metric_and_gate_writers_are_absent(name: str) -> None:
    assert not hasattr(executor, name)


def test_v4_independent_v3_claim_replacement_is_rejected(tmp_path: Path) -> None:
    operation = SyntheticExecutionV4(tmp_path / "operation")
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


def test_v4_independent_v3_canonical_review_alias_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    raw, digest = _review_bytes(root)
    storage = root / "storage/review.json"
    storage.parent.mkdir()
    storage.write_bytes(raw)
    canonical = root / policy.SOURCE_REVIEW_RELATIVE_PATH
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.symlink_to(storage)
    monkeypatch.setattr(policy, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_SOURCE_REVIEW_PATH", canonical)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.preflight_source_review(canonical, digest)


def test_v4_independent_v3_source_parent_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    parent = root / "source"
    parent.mkdir(parents=True)
    source = parent / "candidate.py"
    source.write_bytes(b"reviewed source\n")
    original_open = os.open
    changed = False

    def replace_parent_before_leaf_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal changed
        if not changed and path == source.name and dir_fd is not None:
            changed = True
            moved = tmp_path / "moved-source-parent"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(policy.os, "open", replace_parent_before_leaf_open)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, "source/candidate.py", name="source")
    assert changed


def test_v4_independent_repository_root_walk_rejects_transient_ancestor_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "v4-root-anchor"
    root = anchor / "repo"
    source = root / "source/candidate.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed source\n")
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_root_ancestor_during_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        attacks_absolute_root = Path(path) == root and dir_fd is None
        attacks_component_walk = path == anchor.name and dir_fd is not None
        if not replaced and (attacks_absolute_root or attacks_component_walk):
            replaced = True
            moved = tmp_path / "v4-moved-root-anchor"
            anchor.rename(moved)
            anchor.symlink_to(moved, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                anchor.unlink()
                moved.rename(anchor)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_reads(descriptor: int, length: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, length)

    monkeypatch.setattr(policy.os, "open", replace_root_ancestor_during_open)
    monkeypatch.setattr(policy.os, "read", count_reads)
    rejected = False
    try:
        policy.read_regular_bytes_at(root, "source/candidate.py", name="source")
    except (PermissionError, RuntimeError, OSError):
        rejected = True
    assert {
        "replacement_exercised": replaced,
        "rejected": rejected,
        "read_calls": reads,
    } == {
        "replacement_exercised": True,
        "rejected": True,
        "read_calls": 0,
    }


def test_v4_independent_claim_rejects_transient_output_ancestor_alias(
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "v4-claim-anchor"
    operation = SyntheticExecutionV4(anchor / "operation")
    reservation = operation.claim()
    moved = tmp_path / "v4-moved-claim-anchor"
    anchor.rename(moved)
    anchor.symlink_to(moved, target_is_directory=True)
    rejected = False
    try:
        try:
            operation.publish(reservation, b'{"status":"completed"}\n')
        except (PermissionError, RuntimeError, OSError):
            rejected = True
    finally:
        operation.close(reservation)
        anchor.unlink()
        moved.rename(anchor)
    assert rejected, "publication accepted a canonical path containing a new ancestor alias"


def test_v4_independent_post_training_failures_are_terminalized() -> None:
    execute = _function_nodes()["execute_exact"]
    lifecycle_try = next(
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Try)
        and {
            "_run_independent_verification",
            "_run_finalization",
        } <= _called_names(ast.Module(body=node.body, type_ignores=[]))
    )
    handler_calls = {
        name
        for handler in lifecycle_try.handlers
        for name in _called_names(handler)
    }
    assert lifecycle_try.handlers, (
        "verification/finalization errors only close the claim descriptor and leave "
        "the sole completed attempt without terminal failure handling"
    )
    assert "_terminate_failure" in handler_calls


def test_v4_independent_claim_descriptor_spans_all_publication_stages() -> None:
    functions = _function_nodes()
    reserve_source = ast.unparse(functions["_reserve_exact_attempt"])
    execute_source = ast.unparse(functions["execute_exact"])
    assert reserve_source.index("os.open(active_staging") < reserve_source.index(
        "os.rename(active_staging, attempt_path)"
    )
    assert "directory_fd=claimed_directory_fd" in reserve_source
    for name in (
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
    ):
        assert name in execute_source
    assert execute_source.rindex("os.close(reservation.directory_fd)") > (
        execute_source.index("_run_finalization")
    )


def test_v4_independent_frozen_schedule_rehash_and_gpu_launcher() -> None:
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

    isolated = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert '[sys.executable, "-I", "-B"' in isolated
    assert 'environment["HIP_VISIBLE_DEVICES"] = "0"' in isolated
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in isolated
    assert 'environment[name] = "1"' in isolated


def test_v4_independent_source_and_rgb_rehash_order_is_retained() -> None:
    retained_path = ROOT / policy.RETAINED_V1_SOURCE_BINDINGS[2][0]
    retained_source = retained_path.read_text()
    post_inputs = retained_source.index("base.revalidate_exact_inputs_after_training(")
    post_rgb = retained_source.index(
        "revalidate_selected_rgb_before_publication(",
        post_inputs,
    )
    checkpoint = retained_source.index("base._checkpoint_bytes(", post_rgb)
    publication = retained_source.index("_publish_success(", checkpoint)
    assert post_inputs < post_rgb < checkpoint < publication

    executor_source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert "policy.preflight_source_review(" in executor_source
    assert 'retained.policy = original["policy"]' in executor_source
    assert "verifier.policy = original_verifier_policy" in executor_source
    assert "finalizer.policy = original_finalizer_policy" in executor_source


def test_v4_independent_recovery_and_fail_closed_exact_state(tmp_path: Path) -> None:
    operation = SyntheticExecutionV4(tmp_path / "recovery")
    staged = operation.prepare_complete_staging()
    reservation = operation.claim()
    try:
        assert not staged.exists()
        operation.publish(reservation, b'{"production_eligible":false}\n')
        with pytest.raises(FileExistsError, match="already claimed"):
            SyntheticExecutionV4(operation.root).claim()
    finally:
        operation.close(reservation)

    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
