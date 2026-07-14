"""Author closure for the additive N5 full-panel V5 executor."""
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
    go2_observable_camera_ray_fit_v4_n5_full_panel_v5 as policy,
)
from lewm.tests.n5_full_panel_v5_synthetic_execution import SyntheticExecutionV5
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5 as executor


ROOT = Path(__file__).resolve().parents[2]
FROZEN_V4 = {
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_implementation_handoff_2026-07-13.md": "4e0aa7e2efa266feb774a4b095cbddca105cfd046aac7a0da7f942f1b2b6925e",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": "ff291b94b1546ae9ccf0b85de5f96b87edce4ad5b7992ca16bbbf13dcd1d4485",
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": "19cbdc5692911b31b3b44883b0cfefcc81daa4afc16250b89c1317dd9b66afe4",
    "lewm/tests/n5_full_panel_v4_synthetic_execution.py": "01e49c303d0e2c8e76e7ecbdbd2d0cf159948a5f36a4dc6248d0e014d9c69fb5",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py": "299fd18b88a869916a916adc4e8848235e955447e9a1f245aeaeec6e7ee69688",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_independent_review.py": "2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f",
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_independent_review_2026-07-13.md": "7edeff73d6022a4086706907b03084ff080c9ad1d52ae91e8659fc6ecdc6b18c",
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_independent_review_block_2026-07-13.json": "d2224049a4ee2b793737802d06d91757c17d20b0457c1624517467638173c507",
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


def test_v5_frozen_v4_block_and_all_static_parents_rehash() -> None:
    for relative, expected in FROZEN_V4.items():
        assert _sha(ROOT / relative) == expected
    static = policy.preflight_static_authority()
    assert static["v1_block_content_sha256"] == policy.V1_BLOCK_CONTENT_SHA256
    assert static["v2_block_content_sha256"] == policy.V2_BLOCK_CONTENT_SHA256
    assert static["v3_block_content_sha256"] == policy.V3_BLOCK_CONTENT_SHA256
    assert static["v4_block_content_sha256"] == policy.V4_BLOCK_CONTENT_SHA256


def test_v5_import_exposes_no_callable_operation_and_performs_no_lifecycle_work() -> None:
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


def test_v5_production_source_defines_partial_stages_only_inside_script_entry() -> None:
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
        "AttemptReservationV5",
        "_reserve_exact_attempt",
        "_publish_success",
        "_write_canonical_json",
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
        "execute_exact",
    } <= nested


def test_v5_no_constructible_stage_evidence_class_is_importable() -> None:
    exposed = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type)
        and value.__module__ == executor.__name__
        and any(term in name.casefold() for term in ("reservation", "attempt", "completion"))
    }
    assert exposed == set()


def test_v5_reservation_copy_reconstruction_and_mutation_surface_is_absent() -> None:
    assert not hasattr(executor, "AttemptReservationV5")
    assert not any("reservation" in name.casefold() for name in vars(executor))


def test_v5_completion_writer_is_not_importable() -> None:
    assert not hasattr(executor, "_publish_success")
    assert not hasattr(executor, "_terminate_failure")


@pytest.mark.parametrize("name", ["_write_canonical_json", "_write_bytes_exclusive"])
def test_v5_metric_and_gate_writers_are_not_importable(name: str) -> None:
    assert not hasattr(executor, name)


def test_v5_publication_rejects_replaced_claim_directory(tmp_path: Path) -> None:
    operation = SyntheticExecutionV5(tmp_path / "synthetic")
    reservation = operation.claim()
    moved = tmp_path / "moved-original-attempt"
    operation.attempt.rename(moved)
    operation.attempt.mkdir()
    try:
        with pytest.raises(PermissionError, match="directory identity|claim identity"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        assert list(operation.attempt.iterdir()) == []
        assert sorted(path.name for path in moved.iterdir()) == ["reservation.json"]
    finally:
        operation.close(reservation)


def test_v5_review_preflight_rejects_canonical_leaf_alias(
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


def test_v5_review_preflight_accepts_only_the_regular_canonical_entry(
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


def test_v5_source_reader_rejects_parent_identity_change(
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


def test_v5_source_reader_walks_from_filesystem_root_without_absolute_repo_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical-root"
    root.mkdir()
    (root / "candidate.py").write_bytes(b"reviewed source bytes\n")
    original_open = os.open
    opened: list[object] = []

    def record_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        opened.append(path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(policy.os, "open", record_open)
    assert policy.read_regular_bytes_at(
        root,
        "candidate.py",
        name="reviewed source",
    ) == b"reviewed source bytes\n"
    assert opened[0] == Path(root.anchor)
    assert root not in opened
    assert all(path == Path(root.anchor) or isinstance(path, str) for path in opened)


def test_v5_source_reader_rejects_transient_root_component_alias_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical-root"
    root.mkdir()
    (root / "candidate.py").write_bytes(b"reviewed source bytes\n")
    moved = tmp_path / "moved-root"
    original_open = os.open
    original_read = os.read
    swapped = False
    reads = 0

    def swap_root_then_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == root.name and dir_fd is not None:
            swapped = True
            root.rename(moved)
            root.symlink_to(moved, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "open", swap_root_then_open)
    monkeypatch.setattr(policy.os, "read", count_read)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, "candidate.py", name="reviewed source")
    assert swapped is True
    assert reads == 0


def test_v5_source_reader_rejects_restored_transient_ancestor_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "v5-root-anchor"
    root = anchor / "repo"
    source = root / "source/candidate.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"reviewed source\n")
    original_open = os.open
    original_read = os.read
    replaced = False
    reads = 0

    def replace_ancestor_during_component_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == anchor.name and dir_fd is not None:
            replaced = True
            moved = tmp_path / "v5-moved-root-anchor"
            anchor.rename(moved)
            anchor.symlink_to(moved, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                anchor.unlink()
                moved.rename(anchor)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def count_reads(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "open", replace_ancestor_during_component_open)
    monkeypatch.setattr(policy.os, "read", count_reads)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, "source/candidate.py", name="source")
    assert replaced is True
    assert reads == 0


def test_v5_source_reader_rejects_same_inode_parent_metadata_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    parent = root / "source"
    parent.mkdir(parents=True)
    (parent / "candidate.py").write_bytes(b"reviewed source bytes\n")
    original_read = os.read
    changed = False

    def change_mode_then_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            os.chmod(parent, 0o750)
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "read", change_mode_then_read)
    with pytest.raises((PermissionError, RuntimeError), match="changed|mutation"):
        policy.read_regular_bytes_at(root, "source/candidate.py", name="reviewed source")
    assert changed is True


def test_v5_source_reader_rejects_symlink_hardlink_and_fifo(tmp_path: Path) -> None:
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
def test_v5_synthetic_operation_rejects_production_namespaces(unsafe: Path) -> None:
    with pytest.raises(PermissionError, match="production|repository"):
        SyntheticExecutionV5(unsafe)


def test_v5_synthetic_recovery_is_single_use_and_descriptor_bound(tmp_path: Path) -> None:
    operation = SyntheticExecutionV5(tmp_path / "safe")
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


def test_v5_success_rejects_replaced_output_ancestor_but_fd_failure_terminalizes(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV5(tmp_path / "safe")
    reservation = operation.claim()
    operation.publish_claim_artifact(reservation, "checkpoint.pt", b"owned checkpoint")
    operation.publish_derived_artifact(reservation, "metric.json", b"owned metric")
    moved_root = tmp_path / "moved-safe"
    operation.root.rename(moved_root)
    operation.root.symlink_to(moved_root, target_is_directory=True)
    try:
        with pytest.raises(PermissionError, match="directory identity"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        failure = operation.terminate(
            reservation,
            RuntimeError("verification failed"),
            stage="verification",
        )
        moved_attempt = moved_root / "attempts/seed_20260710/n5"
        assert sorted(path.name for path in moved_attempt.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        assert list((moved_root / "derived").iterdir()) == []
        assert failure["retry_authorized"] is False
    finally:
        operation.close(reservation)


@pytest.mark.parametrize(
    ("stage", "derived_names"),
    [
        ("verification", ("metric.json",)),
        ("finalization", ("metric.json", "gate.json")),
    ],
)
def test_v5_post_training_failure_removes_only_owned_partials_and_is_no_retry(
    tmp_path: Path,
    stage: str,
    derived_names: tuple[str, ...],
) -> None:
    operation = SyntheticExecutionV5(tmp_path / stage)
    reservation = operation.claim()
    try:
        for name in ("checkpoint.pt", "result.json", "completed.json"):
            operation.publish_claim_artifact(reservation, name, name.encode("ascii"))
        for name in derived_names:
            operation.publish_derived_artifact(reservation, name, name.encode("ascii"))
        failure = operation.terminate(
            reservation,
            RuntimeError(f"injected {stage} failure"),
            stage=stage,
        )
        assert failure["failure_stage"] == stage
        assert failure["retry_authorized"] is False
        assert {item["outcome"] for item in failure["artifact_cleanup"]} == {
            "removed_owned"
        }
        assert sorted(path.name for path in operation.attempt.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        assert list((operation.root / "derived").iterdir()) == []
        with pytest.raises(FileExistsError, match="already claimed"):
            SyntheticExecutionV5(operation.root).claim()
    finally:
        operation.close(reservation)


def test_v5_terminalization_preserves_mutated_artifacts_as_invalid(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV5(tmp_path / "safe")
    reservation = operation.claim()
    try:
        operation.publish_claim_artifact(reservation, "checkpoint.pt", b"owned")
        operation.publish_derived_artifact(reservation, "metric.json", b"owned")
        (operation.attempt / "checkpoint.pt").write_bytes(b"foreign replacement bytes")
        (operation.root / "derived/metric.json").write_bytes(
            b"foreign replacement bytes"
        )
        failure = operation.terminate(
            reservation,
            RuntimeError("injected verification failure"),
            stage="verification",
        )
        mismatches = {
            item["artifact"]
            for item in failure["artifact_cleanup"]
            if item["outcome"] == "ownership_mismatch_preserved_invalid"
        }
        assert mismatches == {"checkpoint.pt", "metric.json"}
        assert (operation.attempt / "checkpoint.pt").read_bytes() == (
            b"foreign replacement bytes"
        )
        assert (operation.root / "derived/metric.json").read_bytes() == (
            b"foreign replacement bytes"
        )
        assert (operation.attempt / "failed.json").is_file()
    finally:
        operation.close(reservation)


def test_v5_concurrent_synthetic_claims_choose_exactly_one(tmp_path: Path) -> None:
    operations = [SyntheticExecutionV5(tmp_path / "safe") for _ in range(2)]

    def claim(operation: SyntheticExecutionV5) -> object:
        try:
            return operation.claim()
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, operations))
    reservations = [value for value in outcomes if value != "rejected"]
    assert len(reservations) == 1
    SyntheticExecutionV5.close(reservations[0])  # type: ignore[arg-type]


def test_v5_preclaim_cleanup_and_postclaim_failure_are_durable(tmp_path: Path) -> None:
    before = SyntheticExecutionV5(tmp_path / "before")
    with pytest.raises(RuntimeError, match="before atomic"):
        before.claim(failure_injection="before_atomic_claim")
    assert not before.attempt.exists()

    after = SyntheticExecutionV5(tmp_path / "after")
    with pytest.raises(RuntimeError, match="after atomic"):
        after.claim(failure_injection="after_atomic_claim")
    assert sorted(path.name for path in after.attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    with pytest.raises(FileExistsError, match="already claimed"):
        SyntheticExecutionV5(after.root).claim()


def test_v5_retained_source_output_and_claim_chains_are_present_in_production() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    reserve = functions["_reserve_exact_attempt"]
    chain = functions["_open_canonical_directory_chain"]
    writer = functions["_write_canonical_json"]
    assert "os.open(filesystem_root, _directory_flags())" in chain
    assert "dir_fd=parent_fd" in chain
    assert "anchor_fingerprint" in chain
    assert "_open_canonical_directory_chain(seed_root)" in reserve
    assert "dir_fd=seed_root_fd" in reserve
    assert "os.fstat(claimed_directory_fd)" in reserve
    assert reserve.index("os.open(active_staging.name") < reserve.index(
        "os.rename(active_staging.name, attempt_path.name"
    )
    assert reserve.index(
        "os.rename(active_staging.name, attempt_path.name"
    ) < reserve.index(
        "os.fsync(seed_root_fd)"
    )
    assert "directory_fd=claimed_directory_fd" in reserve
    assert "directory_chain=claimed_chain" in reserve
    assert "dir_fd=parent_fd" in writer
    assert "reservation.output_root_fd" in writer
    assert "os.open(path" not in writer
    assert "_write_claim_file_exclusive" in functions["_publish_success"]
    assert "_read_claim_file" in functions["_artifact_args"]


def test_v5_post_training_exception_terminalizes_before_all_descriptors_close() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    execute = functions["execute_exact"]
    terminate = functions["_terminate_failure"]
    cleanup = functions["_cleanup_owned_artifacts"]
    assert "stage = 'verification'" in execute
    assert "stage = 'finalization'" in execute
    assert "except BaseException as error" in execute
    assert "_terminate_failure(reservation, error, stage=stage)" in execute
    assert execute.index("_terminate_failure(reservation, error, stage=stage)") < (
        execute.index("os.close(reservation.directory_fd)")
    )
    assert "_close_directory_chain(reservation.directory_chain)" in execute
    assert "_assert_claim_fd_owned(reservation)" in terminate
    assert "_assert_owned_claim(reservation)" not in terminate
    assert "follow_symlinks=False" in cleanup
    assert "_stable_fingerprint(current) != artifact.fingerprint" in cleanup


def test_v5_frozen_science_and_gpu0_launcher_are_unchanged() -> None:
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


def test_v5_review_contract_binds_end_to_end_sources_and_v4_block_closures() -> None:
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
    assert core["v4_block_evidence"]["block_content_sha256"] == (
        policy.V4_BLOCK_CONTENT_SHA256
    )
    assert core["execution_contract"]["importable_partial_stage"] is False
    assert core["execution_contract"]["stage_values_constructed_inside_script_entry"] is True
    assert core["execution_contract"]["filesystem_root_anchored_source_walk"] is True
    assert core["execution_contract"]["post_training_failure_terminalization"] is True
    assert core["reservation_contract"][
        "claimed_directory_descriptor_retained_end_to_end"
    ] is True
    assert core["reservation_contract"][
        "canonical_claim_parent_chain_retained_end_to_end"
    ] is True
    assert core["reservation_contract"][
        "owned_derived_partials_removed_before_failure_terminalization"
    ] is True
    assert set(core["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
