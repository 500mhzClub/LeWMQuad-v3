"""Author closure for the additive N5 full-panel V8 executor."""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any
import uuid

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v8 as policy,
)
from lewm.tests.n5_full_panel_v8_synthetic_execution import (
    LOCK_NAME as SYNTHETIC_LOCK_NAME,
    SyntheticExecutionV8,
)
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v8 as executor


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
FROZEN_V5 = {
    policy.V5_POLICY_RELATIVE_PATH: policy.V5_POLICY_FILE_SHA256,
    policy.V5_EXECUTOR_RELATIVE_PATH: policy.V5_EXECUTOR_FILE_SHA256,
    policy.V5_SYNTHETIC_RELATIVE_PATH: policy.V5_SYNTHETIC_FILE_SHA256,
    policy.V5_TEST_RELATIVE_PATH: policy.V5_TEST_FILE_SHA256,
    policy.V5_HANDOFF_RELATIVE_PATH: policy.V5_HANDOFF_FILE_SHA256,
    policy.V5_REVIEW_RELATIVE_PATH: policy.V5_REVIEW_FILE_SHA256,
    policy.V5_REVIEW_RECORD_RELATIVE_PATH: policy.V5_REVIEW_RECORD_FILE_SHA256,
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


def _load_nested_executor() -> ModuleType:
    source_path = ROOT / policy.EXECUTOR_RELATIVE_PATH
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    body: list[ast.stmt] = []
    for node in tree.body:
        if node is entry:
            body.extend(item for item in entry.body if not isinstance(item, ast.Raise))
        else:
            body.append(node)
    module_name = f"_lewm_camera_v8_author_{uuid.uuid4().hex}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                str(source_path),
                "exec",
            ),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _runtime_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Any, Path]:
    runtime = _load_nested_executor()
    root = tmp_path / "repository"
    generated = root / ".generated"
    output = generated / "camera-v8"
    seed_root = output / "attempts/seed_20260710"
    attempt = seed_root / "n5"
    metric_parent = output / "metric_verifications"
    gate_parent = output / "gates"
    attempt.mkdir(parents=True)
    metric_parent.mkdir()
    gate_parent.mkdir()
    for directory in (
        output,
        seed_root.parent,
        seed_root,
        attempt,
        metric_parent,
        gate_parent,
    ):
        directory.chmod(0o700)
    reservation_raw = b'{"content_sha256":"' + b"0" * 64 + b'"}\n'
    (attempt / "reservation.json").write_bytes(reservation_raw)
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    monkeypatch.setattr(policy, "CANONICAL_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(
        policy,
        "CANONICAL_METRIC_RECEIPT_PATH",
        metric_parent / "seed_20260710_n5.json",
    )
    monkeypatch.setattr(
        policy,
        "CANONICAL_GATE_PATH",
        gate_parent / "seed_20260710_n5.json",
    )
    chain = runtime._open_canonical_directory_chain(seed_root)
    runtime._open_chain_child(chain, output, metric_parent.name)
    runtime._open_chain_child(chain, output, gate_parent.name)
    seed_fd = chain.path_fds[seed_root]
    directory_fd = os.open(attempt.name, runtime._directory_flags(), dir_fd=seed_fd)
    chain.journal.watch_directory(directory_fd, label=str(attempt))
    metadata = os.fstat(directory_fd)
    reservation = runtime.AttemptReservationV8(
        directory=attempt,
        value={"source_review": {}, "content_sha256": "0" * 64},
        raw=reservation_raw,
        file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
        directory_fd=directory_fd,
        directory_identity=(metadata.st_dev, metadata.st_ino),
        directory_fingerprint=runtime._stable_fingerprint(metadata),
        directory_chain=chain,
    )
    runtime._assert_owned_claim(reservation)
    return runtime, reservation, generated


def _close_runtime_reservation(runtime: ModuleType, reservation: Any) -> None:
    os.close(reservation.directory_fd)
    runtime._close_directory_chain(reservation.directory_chain)
    sys.modules.pop(runtime.__name__, None)


def _create_remove(parent_fd: int, name: str = "foreign-restored-child") -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=parent_fd,
    )
    os.close(descriptor)
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def test_v8_claim_transaction_rejects_frozen_v6_create_delete_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    original = reservation.journal._finish

    def interleaved(state: Any, before: Any, **kwargs: Any) -> Any:
        _create_remove(state.directory_fd)
        return original(state, before, **kwargs)

    reservation.journal._finish = interleaved
    try:
        with pytest.raises(PermissionError, match="event sequence"):
            runtime._write_claim_file_exclusive(
                reservation,
                "checkpoint.pt",
                b"owned checkpoint",
                role="training_checkpoint",
            )
        assert reservation.journal.poisoned is True
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_derived_transaction_rejects_frozen_v6_create_delete_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    original = reservation.journal._finish

    def interleaved(state: Any, before: Any, **kwargs: Any) -> Any:
        _create_remove(state.directory_fd)
        return original(state, before, **kwargs)

    reservation.journal._finish = interleaved
    core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    try:
        with pytest.raises(PermissionError, match="event sequence"):
            runtime._write_canonical_json(
                reservation,
                policy.CANONICAL_METRIC_RECEIPT_PATH,
                value,
            )
        assert reservation.journal.poisoned is True
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_owned_claim_and_derived_transactions_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    try:
        runtime._write_claim_file_exclusive(
            reservation,
            "checkpoint.pt",
            b"owned checkpoint",
            role="training_checkpoint",
        )
        runtime._write_canonical_json(
            reservation,
            policy.CANONICAL_METRIC_RECEIPT_PATH,
            value,
        )
        runtime._assert_owned_claim(reservation)
        assert reservation.journal.poisoned is False
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_production_cleanup_and_failure_receipt_are_journaled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    try:
        runtime._write_claim_file_exclusive(
            reservation,
            "checkpoint.pt",
            b"partial checkpoint",
            role="training_checkpoint",
        )
        runtime._write_canonical_json(
            reservation,
            policy.CANONICAL_METRIC_RECEIPT_PATH,
            value,
        )
        runtime._terminate_failure(
            reservation,
            RuntimeError("synthetic verification failure"),
            stage="verification",
        )
        failed = policy.parse_json(
            runtime._read_claim_file(
                reservation,
                "failed.json",
                require_canonical=False,
            ),
            name="synthetic production failure",
        )
        assert failed["owned_directory_journal"]["integrity"] == "intact"
        assert {item["outcome"] for item in failed["artifact_cleanup"]} == {
            "removed_owned"
        }
        assert sorted(os.listdir(reservation.directory_fd)) == [
            "failed.json",
            "reservation.json",
        ]
        metric_parent_fd = reservation.directory_chain.path_fds[
            policy.CANONICAL_METRIC_RECEIPT_PATH.parent
        ]
        assert os.listdir(metric_parent_fd) == []
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_claim_commit_is_registered_before_postcommit_stat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    original_create = reservation.journal.create_file
    original_stat = os.stat
    committed = False
    failed = False

    def tracked_create(parent_fd: int, name: str, payload: bytes, **kwargs: Any) -> Any:
        nonlocal committed
        fingerprint = original_create(parent_fd, name, payload, **kwargs)
        if name == "checkpoint.pt":
            committed = True
        return fingerprint

    def fail_first_postcommit_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if committed and not failed and path == "checkpoint.pt":
            failed = True
            raise OSError("injected postcommit stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(reservation.journal, "create_file", tracked_create)
    monkeypatch.setattr(runtime.os, "stat", fail_first_postcommit_stat)
    try:
        with pytest.raises(OSError, match="postcommit stat"):
            runtime._write_claim_file_exclusive(
                reservation,
                "checkpoint.pt",
                b"committed checkpoint",
                role="training_checkpoint",
            )
        assert "checkpoint.pt" in reservation.owned_claim_artifacts
        monkeypatch.setattr(runtime.os, "stat", original_stat)
        runtime._terminate_failure(
            reservation,
            RuntimeError("postcommit claim failure"),
            stage="training",
        )
        assert sorted(os.listdir(reservation.directory_fd)) == [
            "failed.json",
            "reservation.json",
        ]
    finally:
        monkeypatch.setattr(runtime.os, "stat", original_stat)
        _close_runtime_reservation(runtime, reservation)


def test_v8_derived_commit_is_registered_before_postcommit_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    original_fsync = os.fsync
    failed = False

    def fail_output_root_fsync(descriptor: int) -> None:
        nonlocal failed
        if descriptor == reservation.output_root_fd and not failed:
            failed = True
            raise OSError("injected postcommit output-root fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(runtime.os, "fsync", fail_output_root_fsync)
    relative = str(
        policy.CANONICAL_METRIC_RECEIPT_PATH.relative_to(
            policy.CANONICAL_OUTPUT_ROOT
        )
    )
    try:
        with pytest.raises(OSError, match="postcommit output-root fsync"):
            runtime._write_canonical_json(
                reservation,
                policy.CANONICAL_METRIC_RECEIPT_PATH,
                value,
            )
        assert relative in reservation.owned_derived_artifacts
        monkeypatch.setattr(runtime.os, "fsync", original_fsync)
        runtime._terminate_failure(
            reservation,
            RuntimeError("postcommit derived failure"),
            stage="verification",
        )
        metric_parent_fd = reservation.directory_chain.path_fds[
            policy.CANONICAL_METRIC_RECEIPT_PATH.parent
        ]
        assert os.listdir(metric_parent_fd) == []
        assert sorted(os.listdir(reservation.directory_fd)) == [
            "failed.json",
            "reservation.json",
        ]
    finally:
        monkeypatch.setattr(runtime.os, "fsync", original_fsync)
        _close_runtime_reservation(runtime, reservation)


def test_v8_poisoned_journal_writes_identity_only_failure_and_never_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    _create_remove(reservation.directory_fd)
    try:
        runtime._terminate_failure(
            reservation,
            PermissionError("synthetic external history"),
            stage="training",
        )
        failed = policy.parse_json(
            runtime._read_claim_file(
                reservation,
                "failed.json",
                require_canonical=False,
            ),
            name="poisoned synthetic production failure",
        )
        assert failed["owned_directory_journal"] == {
            "integrity": "failed",
            "poison_reason": "event occurred outside an owned transaction",
            "success_eligibility_restored": False,
        }
        assert reservation.journal.poisoned is True
        with pytest.raises(PermissionError):
            reservation.journal.create_file(
                reservation.directory_fd,
                "never-success.json",
                b"{}\n",
            )
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_poison_during_failure_receipt_preserves_invalid_and_returns_no_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    original_finish = reservation.journal._finish

    def interleave_terminal_create(state: Any, before: Any, **kwargs: Any) -> Any:
        expected = kwargs.get("expected_events", ())
        if any(event.name == "failed.json" for event in expected):
            _create_remove(reservation.directory_fd, "terminal-foreign-history")
        return original_finish(state, before, **kwargs)

    reservation.journal._finish = interleave_terminal_create
    try:
        with pytest.raises(PermissionError, match="event sequence"):
            runtime._terminate_failure(
                reservation,
                RuntimeError("terminal publication race"),
                stage="training",
            )
        assert reservation.journal.poisoned is True
        assert sorted(os.listdir(reservation.directory_fd)) == [
            "failed.json",
            "reservation.json",
        ]
        assert "failed.json" not in reservation.owned_claim_artifacts
        with pytest.raises(PermissionError):
            reservation.journal.create_file(
                reservation.directory_fd,
                "never-success.json",
                b"{}\n",
            )
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_post_training_ownership_failure_terminalizes_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    from scripts import train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained

    review_binding = {
        "path": policy.SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda *_args, **_kwargs: ({"content_sha256": "b" * 64}, b"review\n"),
    )
    monkeypatch.setattr(
        policy,
        "source_review_binding",
        lambda *_args, **_kwargs: dict(review_binding),
    )
    monkeypatch.setattr(
        runtime,
        "_reserve_exact_attempt",
        lambda _source_review_sha256: reservation,
    )

    def returned_training_with_queued_mutation(
        authority: object,
        *,
        rgb_workers: int,
    ) -> dict[str, Any]:
        assert rgb_workers == 1
        assert retained._reserve_attempt(authority) is reservation
        _create_remove(reservation.directory_fd, "post-training-history")
        return {"status": "returned"}

    monkeypatch.setattr(
        retained,
        "_run_training",
        returned_training_with_queued_mutation,
    )
    with pytest.raises(
        PermissionError,
        match="claimed directory changed|owned-directory journal rejected",
    ):
        runtime._run_frozen_training("a" * 64, rgb_workers=1)
    failed = policy.parse_json(
        (reservation.directory / "failed.json").read_bytes(),
        name="post-training ownership failure",
    )
    assert failed["owned_directory_journal"]["integrity"] == "failed"
    assert reservation.journal.poisoned is True
    assert reservation.directory_chain.closed is True
    with pytest.raises(OSError):
        os.fstat(reservation.directory_fd)
    sys.modules.pop(runtime.__name__, None)


def test_v8_move_in_out_restoration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _runtime_reservation(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "foreign-move").write_bytes(b"foreign")
    outside_fd = os.open(outside, runtime._directory_flags())
    original = reservation.journal._finish

    def interleaved(state: Any, before: Any, **kwargs: Any) -> Any:
        os.rename(
            "foreign-move",
            "foreign-move",
            src_dir_fd=outside_fd,
            dst_dir_fd=state.directory_fd,
        )
        os.rename(
            "foreign-move",
            "foreign-move",
            src_dir_fd=state.directory_fd,
            dst_dir_fd=outside_fd,
        )
        os.fsync(state.directory_fd)
        return original(state, before, **kwargs)

    reservation.journal._finish = interleaved
    try:
        with pytest.raises(PermissionError, match="event sequence"):
            runtime._write_claim_file_exclusive(
                reservation,
                "checkpoint.pt",
                b"owned checkpoint",
            )
        assert (outside / "foreign-move").read_bytes() == b"foreign"
        assert reservation.journal.poisoned is True
    finally:
        os.close(outside_fd)
        _close_runtime_reservation(runtime, reservation)


@pytest.mark.parametrize("history", ["create_delete", "move_in_out"])
@pytest.mark.parametrize("hook", ["pre_drain", "post_snapshot", "post_drain"])
def test_v8_restoration_history_rejects_at_every_transaction_hook(
    tmp_path: Path,
    history: str,
    hook: str,
) -> None:
    runtime = _load_nested_executor()
    watched = tmp_path / "watched"
    outside = tmp_path / "outside"
    watched.mkdir()
    outside.mkdir()
    (outside / "foreign-move").write_bytes(b"foreign")
    watched_fd = os.open(watched, runtime._directory_flags())
    outside_fd = os.open(outside, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(watched_fd, label="watched")

    def restore_history() -> None:
        if history == "create_delete":
            _create_remove(watched_fd, "foreign-restored")
            return
        os.rename(
            "foreign-move",
            "foreign-move",
            src_dir_fd=outside_fd,
            dst_dir_fd=watched_fd,
        )
        os.rename(
            "foreign-move",
            "foreign-move",
            src_dir_fd=watched_fd,
            dst_dir_fd=outside_fd,
        )
        os.fsync(watched_fd)
        os.fsync(outside_fd)

    original_snapshots = journal._snapshots
    snapshot_calls = 0

    def snapshots_with_hook() -> Any:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshots = original_snapshots()
        if (
            (hook == "pre_drain" and snapshot_calls == 2)
            or (hook == "post_snapshot" and snapshot_calls == 3)
        ):
            restore_history()
        return snapshots

    original_validate = journal._validate_snapshots

    def validate_with_post_drain_hook(*args: Any, **kwargs: Any) -> None:
        original_validate(*args, **kwargs)
        if hook == "post_drain":
            restore_history()

    journal._snapshots = snapshots_with_hook
    journal._validate_snapshots = validate_with_post_drain_hook
    try:
        if hook == "post_drain":
            journal.create_file(watched_fd, "owned", b"payload")
            with pytest.raises(PermissionError):
                journal.assert_clean()
        else:
            with pytest.raises(PermissionError):
                journal.create_file(watched_fd, "owned", b"payload")
        assert journal.poisoned is True
        assert (outside / "foreign-move").read_bytes() == b"foreign"
    finally:
        journal.close()
        os.close(outside_fd)
        os.close(watched_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_kernel_exact_sequences_cover_all_journal_primitives(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(parent_fd, label="parent")
    child_fd: int | None = None
    try:
        journal.mkdir(parent_fd, "child")
        child_fd = os.open("child", runtime._directory_flags(), dir_fd=parent_fd)
        journal.watch_directory(child_fd, label="child")
        journal.create_file(child_fd, "empty", b"")
        journal.create_file(child_fd, "payload", b"first")
        journal.replace_file(child_fd, "payload", b"second")
        rows = dict(journal.baseline(child_fd).inventory)
        journal.unlink(child_fd, "empty", expected_fingerprint=rows["empty"])
        rows = dict(journal.baseline(child_fd).inventory)
        journal.unlink(child_fd, "payload", expected_fingerprint=rows["payload"])
        journal.rename_directory(
            parent_fd,
            "child",
            "claimed",
            directory_fd=child_fd,
        )
        journal.rmdir(
            parent_fd,
            "claimed",
            directory_fd=child_fd,
        )
        child_fd = None
        journal.assert_clean()
        assert os.listdir(parent_fd) == []
    finally:
        journal.close()
        if child_fd is not None:
            try:
                os.close(child_fd)
            except OSError:
                pass
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_no_replace_claim_race_preserves_foreign_destination(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(parent_fd, label="parent")
    journal.mkdir(parent_fd, "staging")
    source_fd = os.open("staging", runtime._directory_flags(), dir_fd=parent_fd)
    source_identity = (os.fstat(source_fd).st_dev, os.fstat(source_fd).st_ino)
    source_snapshot = journal._snapshot(source_fd)
    journal.watch_directory(
        source_fd,
        label="staging",
        expected_snapshot=source_snapshot,
    )
    original_rename = journal._rename_noreplace
    raced_identity: tuple[int, int] | None = None

    def race_destination(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal raced_identity
        os.mkdir(destination, 0o700, dir_fd=destination_parent_fd)
        raced = os.stat(
            destination,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        raced_identity = (raced.st_dev, raced.st_ino)
        original_rename(
            source_parent_fd,
            source,
            destination_parent_fd,
            destination,
        )

    journal._rename_noreplace = race_destination
    try:
        with pytest.raises(PermissionError):
            journal.rename_directory(
                parent_fd,
                "staging",
                "n5",
                directory_fd=source_fd,
            )
        source = os.stat("staging", dir_fd=parent_fd, follow_symlinks=False)
        destination = os.stat("n5", dir_fd=parent_fd, follow_symlinks=False)
        assert (source.st_dev, source.st_ino) == source_identity
        assert (destination.st_dev, destination.st_ino) == raced_identity
        assert source_identity != raced_identity
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(source_fd)
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_directory_rename_binds_source_to_retained_descriptor(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    (tmp_path / "source-a").mkdir()
    (tmp_path / "source-b").mkdir()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    source_a_fd = os.open("source-a", runtime._directory_flags(), dir_fd=parent_fd)
    journal = runtime.OwnedDirectoryJournalV8()
    parent_snapshot = journal._snapshot(parent_fd)
    journal.watch_directory(
        parent_fd,
        label="parent",
        expected_snapshot=parent_snapshot,
    )
    source_snapshot = journal._snapshot(source_a_fd)
    journal.watch_directory(
        source_a_fd,
        label="source-a",
        expected_snapshot=source_snapshot,
    )
    invoked = False

    def forbidden_rename(*_args: Any) -> None:
        nonlocal invoked
        invoked = True
        raise AssertionError("mismatched source reached renameat2")

    journal._rename_noreplace = forbidden_rename
    try:
        with pytest.raises(PermissionError, match="source/destination changed"):
            journal.rename_directory(
                parent_fd,
                "source-b",
                "claimed",
                directory_fd=source_a_fd,
            )
        assert invoked is False
        assert (tmp_path / "source-a").is_dir()
        assert (tmp_path / "source-b").is_dir()
        assert not (tmp_path / "claimed").exists()
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(source_a_fd)
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_unlink_syscall_boundary_race_poisons_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    directory_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(directory_fd, label="watched")
    expected = journal.create_file(directory_fd, "owned", b"expected")
    original_unlink = os.unlink
    original_rename = os.rename

    def raced_unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name == "owned":
            original_rename(
                "owned",
                "expected-preserved",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            descriptor = os.open(
                "owned",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=dir_fd,
            )
            os.close(descriptor)
        original_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "unlink", raced_unlink)
    try:
        with pytest.raises(PermissionError):
            journal.unlink(
                directory_fd,
                "owned",
                expected_fingerprint=expected,
            )
        assert (tmp_path / "expected-preserved").read_bytes() == b"expected"
        assert journal.poisoned is True
        with pytest.raises(PermissionError):
            journal.create_file(directory_fd, "never-success", b"")
    finally:
        monkeypatch.setattr(runtime.os, "unlink", original_unlink)
        journal.close()
        os.close(directory_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_replace_syscall_boundary_race_poisons_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    directory_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(directory_fd, label="watched")
    journal.create_file(directory_fd, "owned", b"expected")
    original_replace = os.replace
    original_rename = os.rename

    def raced_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if destination == "owned":
            original_rename(
                "owned",
                "expected-preserved",
                src_dir_fd=dst_dir_fd,
                dst_dir_fd=dst_dir_fd,
            )
            descriptor = os.open(
                "owned",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=dst_dir_fd,
            )
            os.close(descriptor)
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(runtime.os, "replace", raced_replace)
    try:
        with pytest.raises(PermissionError):
            journal.replace_file(directory_fd, "owned", b"replacement")
        assert (tmp_path / "expected-preserved").read_bytes() == b"expected"
        assert journal.poisoned is True
        with pytest.raises(PermissionError):
            journal.create_file(directory_fd, "never-success", b"")
    finally:
        monkeypatch.setattr(runtime.os, "replace", original_replace)
        journal.close()
        os.close(directory_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_rmdir_syscall_boundary_race_poisons_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(parent_fd, label="parent")
    journal.mkdir(parent_fd, "owned")
    child_fd = os.open("owned", runtime._directory_flags(), dir_fd=parent_fd)
    child_snapshot = journal._snapshot(child_fd)
    journal.watch_directory(
        child_fd,
        label="owned",
        expected_snapshot=child_snapshot,
    )
    expected_identity = (os.fstat(child_fd).st_dev, os.fstat(child_fd).st_ino)
    original_rmdir = os.rmdir
    original_rename = os.rename
    original_mkdir = os.mkdir

    def raced_rmdir(name: str, *, dir_fd: int | None = None) -> None:
        if name == "owned":
            original_rename(
                "owned",
                "expected-preserved",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            original_mkdir("owned", 0o700, dir_fd=dir_fd)
        original_rmdir(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "rmdir", raced_rmdir)
    try:
        with pytest.raises(PermissionError):
            journal.rmdir(parent_fd, "owned", directory_fd=child_fd)
        child_fd = -1
        preserved = os.stat(
            "expected-preserved",
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        assert (preserved.st_dev, preserved.st_ino) == expected_identity
        assert journal.poisoned is True
        with pytest.raises(PermissionError):
            journal.mkdir(parent_fd, "never-success")
    finally:
        monkeypatch.setattr(runtime.os, "rmdir", original_rmdir)
        journal.close()
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_watch_rejects_mutation_before_installation(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    directory_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    expected = journal._snapshot(directory_fd)
    transient = tmp_path / "transient"
    transient.write_bytes(b"history")
    transient.unlink()
    try:
        with pytest.raises(PermissionError, match="before its watch was installed"):
            journal.watch_directory(
                directory_fd,
                label="watched",
                expected_snapshot=expected,
            )
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(directory_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_watch_rejects_mutation_during_installation(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    directory_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    expected = journal._snapshot(directory_fd)
    original_add_watch = journal._add_watch

    def add_watch_then_mutate(*args: Any) -> int:
        watch = int(original_add_watch(*args))
        (tmp_path / "intruder").write_bytes(b"mutation")
        return watch

    journal._add_watch = add_watch_then_mutate
    try:
        with pytest.raises(PermissionError, match="while its watch was installed"):
            journal.watch_directory(
                directory_fd,
                label="watched",
                expected_snapshot=expected,
            )
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(directory_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_exclusive_root_and_descendants_are_created_by_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    root = tmp_path / "repository"
    generated = root / ".generated"
    generated.mkdir(parents=True)
    output = generated / "camera-v8"
    seed_root = output / "attempts/seed_20260710"
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    chain = runtime._open_canonical_directory_chain(seed_root)
    try:
        assert seed_root.is_dir()
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        chain.journal.assert_clean()
    finally:
        runtime._close_directory_chain(chain)
        sys.modules.pop(runtime.__name__, None)


def test_v8_exclusive_root_allows_unrelated_named_shared_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    original_mkdir = os.mkdir
    original_rmdir = os.rmdir

    def mkdir_with_unrelated_churn(
        name: str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if name == "exclusive-v8":
            original_mkdir("unrelated", 0o700, dir_fd=dir_fd)
            original_rmdir("unrelated", dir_fd=dir_fd)
        original_mkdir(name, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "mkdir", mkdir_with_unrelated_churn)
    try:
        fingerprint = journal.create_exclusive_root(parent_fd, "exclusive-v8")
        assert runtime._stable_fingerprint(
            os.stat("exclusive-v8", dir_fd=parent_fd, follow_symlinks=False)
        ) == fingerprint
        journal.assert_clean()
    finally:
        journal.close()
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_exclusive_root_rejects_shared_parent_self_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    original_mkdir = os.mkdir
    original_mode = stat.S_IMODE(tmp_path.stat().st_mode)

    def mkdir_after_parent_mode_history(
        name: str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        os.chmod(tmp_path, original_mode ^ 0o010)
        os.chmod(tmp_path, original_mode)
        original_mkdir(name, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "mkdir", mkdir_after_parent_mode_history)
    try:
        with pytest.raises(PermissionError, match="shared parent self"):
            journal.create_exclusive_root(parent_fd, "exclusive-v8")
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_exclusive_root_rejects_unknown_shared_event_mask(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    original_events = journal._events

    def events_with_unknown_mask() -> list[Any]:
        events = original_events()
        if any(event.name == "exclusive-v8" for event in events):
            events.append(
                runtime.JournalEventV8(
                    events[0].watch,
                    0x00000001,
                    0,
                    "unrelated",
                )
            )
        return events

    journal._events = events_with_unknown_mask
    try:
        with pytest.raises(PermissionError, match="unknown shared-parent event mask"):
            journal.create_exclusive_root(parent_fd, "exclusive-v8")
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def _patch_temporary_reservation_authority(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "temporary-repository"
    root.mkdir()
    output = root / ".generated/camera-v8"
    seed_root = output / "attempts/seed_20260710"
    attempt = seed_root / "n5"
    review_binding = {
        "path": policy.SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    monkeypatch.setattr(policy, "CANONICAL_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(
        policy,
        "CANONICAL_METRIC_RECEIPT_PATH",
        output / "metric_verifications/seed_20260710_n5.json",
    )
    monkeypatch.setattr(
        policy,
        "CANONICAL_GATE_PATH",
        output / "gates/seed_20260710_n5.json",
    )
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda *_args, **_kwargs: ({"content_sha256": "b" * 64}, b"review\n"),
    )
    monkeypatch.setattr(
        policy,
        "source_review_binding",
        lambda *_args, **_kwargs: dict(review_binding),
    )
    return seed_root, attempt, review_binding


def _materialize_existing_recovery_scaffold(
    runtime: ModuleType,
    seed_root: Path,
    *,
    lock_payload: bytes = b"",
) -> tuple[Path, Path, Path]:
    output = policy.CANONICAL_OUTPUT_ROOT
    metric = policy.CANONICAL_METRIC_RECEIPT_PATH.parent
    gate = policy.CANONICAL_GATE_PATH.parent
    seed_root.mkdir(parents=True)
    metric.mkdir()
    gate.mkdir()
    for directory in (output, seed_root.parent, seed_root, metric, gate):
        directory.chmod(0o700)
    lock = seed_root / runtime.LOCK_NAME
    lock.write_bytes(lock_payload)
    lock.chmod(0o600)
    return metric, gate, lock


def _prepare_complete_recovery_stagings(
    runtime: ModuleType,
    seed_root: Path,
    attempt: Path,
    review: dict[str, str],
    *,
    count: int,
) -> list[Path]:
    chain = runtime._open_canonical_directory_chain(seed_root)
    caller_fds: list[int] = []
    paths: list[Path] = []
    try:
        runtime._open_chain_child(
            chain,
            policy.CANONICAL_OUTPUT_ROOT,
            policy.CANONICAL_METRIC_RECEIPT_PATH.parent.name,
        )
        runtime._open_chain_child(
            chain,
            policy.CANONICAL_OUTPUT_ROOT,
            policy.CANONICAL_GATE_PATH.parent.name,
        )
        seed_fd = chain.path_fds[seed_root]
        with runtime._locked_seed_root(
            seed_fd,
            chain.journal,
            allow_create=chain.output_root_created,
        ):
            for index in range(count):
                staging = runtime._new_staging(seed_root, seed_fd, chain.journal)
                caller_fds.append(staging.directory_fd)
                paths.append(staging.path)
                recovery = [
                    {
                        "staging_name": staging.path.name,
                        "classification": "new_unique_private",
                        "action": "complete_then_atomic_claim",
                        "inventory_sha256": policy.canonical_json_sha256([index]),
                    }
                ]
                pending = runtime._reservation(
                    review,
                    attempt_path=attempt,
                    recovery_events=recovery,
                )
                runtime._prepare_new_staging(
                    staging,
                    chain.journal,
                    reservation=pending,
                    attempt_path=attempt,
                )
        return paths
    finally:
        for descriptor in caller_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        runtime._close_directory_chain(chain)


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    return {**core, "content_sha256": policy.canonical_json_sha256(core)}


def _temporary_verifier_request(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    attempt = tmp_path / "attempts/seed_20260710/n5"
    attempt.mkdir(parents=True)
    names = {
        "reservation": "reservation.json",
        "result": "result.json",
        "checkpoint": "checkpoint.pt",
        "completion": "completed.json",
    }
    artifacts: dict[str, str] = {}
    for role, name in names.items():
        path = attempt / name
        raw = f"synthetic {role}\n".encode("ascii")
        path.write_bytes(raw)
        artifacts[role] = f"{path.resolve()}:{hashlib.sha256(raw).hexdigest()}"
    source_review = {
        "path": policy.SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    sources = {
        relative: {"path": relative, "file_sha256": character * 64}
        for relative, character in zip(policy.SUCCESSOR_SOURCE_PATHS, ("c", "d"))
    }
    review = {"successor_sources": sources, "content_sha256": "b" * 64}
    monkeypatch.setattr(policy, "CANONICAL_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda *_args, **_kwargs: (review, b"review\n"),
    )
    monkeypatch.setattr(
        policy,
        "source_review_binding",
        lambda *_args, **_kwargs: dict(source_review),
    )
    monkeypatch.setattr(policy, "preflight_static_authority", lambda: {})
    return _rehash(
        {
            "schema": policy.VERIFICATION_REQUEST_SCHEMA,
            "nonce": "e" * 64,
            "source_review": source_review,
            "sources": sources,
            "artifacts": artifacts,
            "process": {
                "parent_pid": os.getpid(),
                "expected_executable": str(Path(sys.executable).resolve()),
                "expected_executor": str(
                    (ROOT / policy.EXECUTOR_RELATIVE_PATH).resolve()
                ),
                "expected_child_mode": "verification_child",
                "expected_isolated": True,
                "expected_no_bytecode": True,
            },
            "environment": runtime._expected_child_environment(),
            "contract": policy.isolated_verifier_contract(),
        }
    )


def _synthetic_verifier_response(
    runtime: ModuleType,
    request: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    return _rehash(
        {
            "schema": policy.VERIFICATION_RESPONSE_SCHEMA,
            "status": "verified_compute_only",
            "nonce": request["nonce"],
            "request_content_sha256": request["content_sha256"],
            "process": {
                "child_pid": os.getpid() + 100000,
                "parent_pid": os.getpid(),
                "executable": str(Path(sys.executable).resolve()),
                "executor": str((ROOT / policy.EXECUTOR_RELATIVE_PATH).resolve()),
                "mode": "verification_child",
                "isolated": True,
                "no_bytecode": True,
            },
            "environment": runtime._expected_child_environment(),
            "sources": request["sources"],
            "artifacts": request["artifacts"],
            "receipt": receipt,
            "receipt_content_sha256": receipt["content_sha256"],
            "publication_performed": False,
        }
    )


def test_v8_verifier_child_environment_removes_all_conflicting_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        monkeypatch.setenv(name, "conflicting")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "wrong")
    environment = runtime._verification_child_environment()
    assert runtime._child_environment_projection(environment) == (
        runtime._expected_child_environment()
    )
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert all(
        name not in environment
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
        )
    )
    sys.modules.pop(runtime.__name__, None)


def test_v8_exact_relaunch_removes_all_conflicting_device_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        monkeypatch.setenv(name, "conflicting")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "wrong")
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    argv = ["--source-review-sha256", "a" * 64]
    assert runtime._isolated_child(argv) == 0
    assert captured["command"] == [
        sys.executable,
        "-I",
        "-B",
        str((ROOT / policy.EXECUTOR_RELATIVE_PATH).resolve()),
        *argv,
    ]
    assert captured["env"]["HIP_VISIBLE_DEVICES"] == "0"
    assert all(
        name not in captured["env"]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
        )
    )
    assert all(
        captured["env"][name] == "1" for name in policy.THREAD_ENVIRONMENT
    )
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize(
    "mutation",
    ["nonce", "environment", "process", "sources", "artifact", "contract"],
)
def test_v8_verifier_request_binds_every_authority_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime = _load_nested_executor()
    request = _temporary_verifier_request(runtime, tmp_path, monkeypatch)
    assert runtime._validate_verification_request(request) == request
    changed = copy.deepcopy(request)
    if mutation == "nonce":
        changed["nonce"] = "not-a-sha"
    elif mutation == "environment":
        changed["environment"]["hsa_visible_devices"] = "0"
    elif mutation == "process":
        changed["process"]["expected_executor"] = str(tmp_path / "other.py")
    elif mutation == "sources":
        changed["sources"][policy.POLICY_RELATIVE_PATH]["file_sha256"] = "f" * 64
    elif mutation == "artifact":
        changed["artifacts"]["checkpoint"] = changed["artifacts"]["result"]
    else:
        changed["contract"]["child_publication_authorized"] = True
    changed = _rehash(changed)
    with pytest.raises((PermissionError, ValueError)):
        runtime._validate_verification_request(changed)
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize(
    "mutation",
    ["nonce", "environment", "process", "sources", "publication", "receipt"],
)
def test_v8_verifier_response_rejects_binding_or_publication_tamper(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime = _load_nested_executor()
    request = {
        "nonce": "a" * 64,
        "content_sha256": "b" * 64,
        "source_review": {"file_sha256": "c" * 64},
        "sources": {"source": "binding"},
        "artifacts": {"artifact": "binding"},
    }
    receipt = {"content_sha256": "d" * 64}
    response = _synthetic_verifier_response(runtime, request, receipt)
    monkeypatch.setattr(
        runtime,
        "_validate_child_metric_receipt",
        lambda *_args, **_kwargs: dict(receipt),
    )
    assert runtime._validate_verification_response(
        object(), request, response
    ) == receipt
    changed = copy.deepcopy(response)
    if mutation == "nonce":
        changed["nonce"] = "e" * 64
    elif mutation == "environment":
        changed["environment"]["hsa_visible_devices"] = "0"
    elif mutation == "process":
        changed["process"]["isolated"] = False
    elif mutation == "sources":
        changed["sources"] = {"different": "source"}
    elif mutation == "publication":
        changed["publication_performed"] = True
    else:
        changed["receipt_content_sha256"] = "f" * 64
    changed = _rehash(changed)
    with pytest.raises((PermissionError, ValueError)):
        runtime._validate_verification_response(object(), request, changed)
    sys.modules.pop(runtime.__name__, None)


def test_v8_parent_spawns_exact_compute_child_then_publishes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    request = {
        "nonce": "a" * 64,
        "content_sha256": "b" * 64,
    }
    request_raw = policy.canonical_json_bytes(request) + b"\n"
    response_raw = b'{"synthetic":true}\n'
    receipt = {"content_sha256": "c" * 64}
    captured: dict[str, Any] = {}
    publications: list[tuple[Any, Path, Any]] = []

    monkeypatch.setattr(
        runtime,
        "_verification_request",
        lambda *_args, **_kwargs: (dict(request), request_raw),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_verification_response",
        lambda *_args, **_kwargs: dict(receipt),
    )
    monkeypatch.setattr(runtime, "_assert_owned_claim", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_write_canonical_json",
        lambda reservation, path, value: publications.append(
            (reservation, path, value)
        ),
    )

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=response_raw, stderr=b"")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    reservation = object()
    assert runtime._run_independent_verification(reservation, "d" * 64) == receipt
    assert captured["command"] == [
        sys.executable,
        "-I",
        "-B",
        str((ROOT / policy.EXECUTOR_RELATIVE_PATH).resolve()),
        "--verification-child",
    ]
    assert captured["input"] == request_raw
    assert captured["stdout"] is runtime.subprocess.PIPE
    assert captured["stderr"] is runtime.subprocess.PIPE
    assert captured["cwd"] == runtime.ROOT
    assert captured["check"] is False
    assert captured["timeout"] == policy.VERIFICATION_TIMEOUT_SECONDS
    assert captured["close_fds"] is True
    assert runtime._child_environment_projection(captured["env"]) == (
        runtime._expected_child_environment()
    )
    assert publications == [
        (reservation, policy.CANONICAL_METRIC_RECEIPT_PATH, receipt)
    ]
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "nonzero",
        "signal",
        "stderr",
        "empty",
        "oversize",
        "malformed",
        "noncanonical",
        "extra_stdout",
        "binding",
    ],
)
def test_v8_verifier_child_failures_never_publish_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    runtime = _load_nested_executor()
    request = {"nonce": "a" * 64, "content_sha256": "b" * 64}
    request_raw = policy.canonical_json_bytes(request) + b"\n"
    publications: list[Any] = []
    fallbacks: list[Any] = []
    monkeypatch.setattr(
        runtime,
        "_verification_request",
        lambda *_args, **_kwargs: (dict(request), request_raw),
    )
    monkeypatch.setattr(runtime, "_assert_owned_claim", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_write_canonical_json",
        lambda *_args, **_kwargs: publications.append(True),
    )
    monkeypatch.setattr(
        runtime,
        "_compute_verification_receipt_child",
        lambda *_args, **_kwargs: fallbacks.append(True),
    )
    if failure == "binding":
        monkeypatch.setattr(
            runtime,
            "_validate_verification_response",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("injected response binding mismatch")
            ),
        )
    else:
        monkeypatch.setattr(
            runtime,
            "_validate_verification_response",
            lambda *_args, **_kwargs: {"content_sha256": "c" * 64},
        )
    monkeypatch.setattr(policy, "VERIFICATION_MAX_RESPONSE_BYTES", 32)

    def fake_run(*_args: Any, **_kwargs: Any) -> Any:
        if failure == "timeout":
            raise runtime.subprocess.TimeoutExpired("verifier", 1)
        returncode = -9 if failure == "signal" else (7 if failure == "nonzero" else 0)
        stderr = b"warning" if failure == "stderr" else b""
        stdout = b'{}\n'
        if failure == "empty":
            stdout = b""
        elif failure == "oversize":
            stdout = b"x" * 33
        elif failure == "malformed":
            stdout = b"not-json\n"
        elif failure == "noncanonical":
            stdout = b'{"x": 1}\n'
        elif failure == "extra_stdout":
            stdout = b'{}\n{}\n'
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    with pytest.raises((RuntimeError, ValueError, PermissionError)):
        runtime._run_independent_verification(object(), "d" * 64)
    assert publications == []
    assert fallbacks == []
    sys.modules.pop(runtime.__name__, None)


def test_v8_verifier_child_is_the_only_frozen_interop_setter_path() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    child = functions["_compute_verification_receipt_child"]
    parent = functions["_run_independent_verification"]
    dispatch = functions["dispatch"]
    assert "configure_determinism" not in child
    assert child.count("verifier._compute_receipt(token, bundle)") == 1
    assert "compatibility.write_exclusive = forbid_publication" in child
    assert "_write_canonical_json" not in child
    assert "_compute_verification_receipt_child" not in parent
    assert "no fallback" in parent
    assert parent.index("_validate_verification_response") < parent.index(
        "_write_canonical_json"
    )
    assert "raw_argv == ['--verification-child']" in dispatch


def test_v8_full_temporary_reservation_journals_lock_staging_and_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    _seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    reservation = runtime._reserve_exact_attempt("a" * 64)
    try:
        runtime._assert_owned_claim(reservation)
        assert attempt.is_dir()
        assert sorted(os.listdir(reservation.directory_fd)) == ["reservation.json"]
        assert runtime.LOCK_NAME in os.listdir(reservation.seed_root_fd)
        assert not any(
            name.startswith(runtime.STAGING_PREFIX)
            for name in os.listdir(reservation.seed_root_fd)
        )
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_transferred_claim_fd_cleanup_failure_still_closes_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    _seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    chains: list[Any] = []
    transferred_fds: list[int] = []
    original_open_chain = runtime._open_canonical_directory_chain

    def tracked_open_chain(path: Path) -> Any:
        chain = original_open_chain(path)
        chains.append(chain)
        return chain

    def fail_after_transfer(
        journal: Any,
        parent_fd: int,
        source: str,
        destination: str,
        *,
        directory_fd: int,
    ) -> Any:
        del parent_fd, source, destination
        transferred_fds.append(directory_fd)
        os.close(directory_fd)
        journal._fail("injected failure after claim-fd transfer")

    monkeypatch.setattr(runtime, "_open_canonical_directory_chain", tracked_open_chain)
    monkeypatch.setattr(
        runtime.OwnedDirectoryJournalV8,
        "rename_directory",
        fail_after_transfer,
    )
    with pytest.raises(
        RuntimeError,
        match="reservation failure cleanup or terminalization failed",
    ):
        runtime._reserve_exact_attempt("a" * 64)
    assert transferred_fds
    for descriptor in transferred_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert chains and all(chain.closed for chain in chains)
    for chain in chains:
        for descriptor in chain.descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_complete_staging_recovers_by_rehash_then_atomic_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    chain = runtime._open_canonical_directory_chain(seed_root)
    staging_fd: int | None = None
    try:
        runtime._open_chain_child(
            chain,
            policy.CANONICAL_OUTPUT_ROOT,
            policy.CANONICAL_METRIC_RECEIPT_PATH.parent.name,
        )
        runtime._open_chain_child(
            chain,
            policy.CANONICAL_OUTPUT_ROOT,
            policy.CANONICAL_GATE_PATH.parent.name,
        )
        seed_fd = chain.path_fds[seed_root]
        with runtime._locked_seed_root(
            seed_fd,
            chain.journal,
            allow_create=chain.output_root_created,
        ):
            staging = runtime._new_staging(
                seed_root,
                seed_fd,
                chain.journal,
            )
            staging_fd = staging.directory_fd
            recovery = [
                {
                    "staging_name": staging.path.name,
                    "classification": "new_unique_private",
                    "action": "complete_then_atomic_claim",
                    "inventory_sha256": policy.canonical_json_sha256([]),
                }
            ]
            pending = runtime._reservation(
                review,
                attempt_path=attempt,
                recovery_events=recovery,
            )
            runtime._prepare_new_staging(
                staging,
                chain.journal,
                reservation=pending,
                attempt_path=attempt,
            )
        os.close(staging_fd)
        staging_fd = None
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        runtime._close_directory_chain(chain)

    reservation = runtime._reserve_exact_attempt("a" * 64)
    try:
        runtime._assert_owned_claim(reservation)
        classifications = [
            item["classification"] for item in reservation.value["preclaim_recovery"]
        ]
        assert classifications == ["complete"]
        assert reservation.value["preclaim_recovery"][0]["action"] == (
            "resume_after_rehash"
        )
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_existing_exact_scaffold_without_staging_blocks_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    _metric, _gate, lock = _materialize_existing_recovery_scaffold(
        runtime, seed_root
    )
    with pytest.raises(PermissionError, match="lacks one complete staging"):
        runtime._reserve_exact_attempt("a" * 64)
    assert lock.read_bytes() == b""
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize("missing", ["metric", "gate"])
def test_v8_missing_derived_scaffold_blocks_valid_complete_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    staging = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=1
    )[0]
    missing_path = {
        "metric": policy.CANONICAL_METRIC_RECEIPT_PATH.parent,
        "gate": policy.CANONICAL_GATE_PATH.parent,
    }[missing]
    missing_path.rmdir()
    with pytest.raises(PermissionError, match="missing a derived directory"):
        runtime._reserve_exact_attempt("a" * 64)
    assert not missing_path.exists()
    assert staging.is_dir()
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize("lock_state", ["missing", "nonempty"])
def test_v8_invalid_lock_blocks_valid_complete_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_state: str,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    staging = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=1
    )[0]
    lock = seed_root / runtime.LOCK_NAME
    if lock_state == "missing":
        lock.unlink()
        expected = "missing its exact lock leaf"
    else:
        lock.write_bytes(b"foreign lock payload")
        lock.chmod(0o600)
        expected = "reservation lock is not private"
    with pytest.raises(PermissionError, match=expected):
        runtime._reserve_exact_attempt("a" * 64)
    if lock_state == "missing":
        assert not lock.exists()
    else:
        assert lock.read_bytes() == b"foreign lock payload"
    assert staging.is_dir()
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize("mutation", ["nonprivate", "oversize"])
def test_v8_insecure_recovered_leaf_is_preserved_and_blocks_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    staging = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=1
    )[0]
    reservation_leaf = staging / "reservation.json"
    if mutation == "nonprivate":
        reservation_leaf.chmod(0o644)
    else:
        reservation_leaf.write_bytes(b"x" * (1024 * 1024 + 1))
        reservation_leaf.chmod(0o600)
    with pytest.raises(PermissionError, match="preserved invalid"):
        runtime._reserve_exact_attempt("a" * 64)
    assert staging.is_dir()
    assert reservation_leaf.is_file()
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_claim_manifest_alias_is_preserved_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    staging = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=1
    )[0]
    (staging / "staging.json").rename(staging / "claim.json")
    with pytest.raises(PermissionError, match="preserved invalid"):
        runtime._reserve_exact_attempt("a" * 64)
    assert (staging / "claim.json").is_file()
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_multiple_equivalent_complete_stagings_resume_lexical_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    stagings = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=2
    )
    reservation = runtime._reserve_exact_attempt("a" * 64)
    try:
        runtime._assert_owned_claim(reservation)
        assert not any(path.exists() for path in stagings)
        classifications = {
            item["classification"]
            for item in reservation.value["preclaim_recovery"]
        }
        assert classifications == {"complete", "complete_equivalent_duplicate"}
    finally:
        _close_runtime_reservation(runtime, reservation)


def test_v8_conflicting_authority_core_cannot_pass_complete_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    stagings = _prepare_complete_recovery_stagings(
        runtime, seed_root, attempt, review, count=2
    )
    conflicting = stagings[1]
    value = json.loads((conflicting / "reservation.json").read_bytes())
    value["seed"] = 20260711
    core = dict(value)
    core.pop("content_sha256")
    value["content_sha256"] = policy.canonical_json_sha256(core)
    raw = policy.canonical_json_bytes(value) + b"\n"
    (conflicting / "reservation.json").write_bytes(raw)
    conflicting_reservation = runtime.AttemptReservationV8(
        directory=attempt,
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    manifest = runtime._manifest_value(
        staging=conflicting,
        attempt_path=attempt,
        reservation=conflicting_reservation,
    )
    (conflicting / "staging.json").write_bytes(
        policy.canonical_json_bytes(manifest) + b"\n"
    )
    for leaf in conflicting.iterdir():
        leaf.chmod(0o600)
    with pytest.raises(PermissionError, match="preserved invalid"):
        runtime._reserve_exact_attempt("a" * 64)
    assert all(path.is_dir() for path in stagings)
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_exact_name_invalid_staging_closes_opened_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    _materialize_existing_recovery_scaffold(runtime, seed_root)
    staging = seed_root / f"{runtime.STAGING_PREFIX}{'c' * 32}"
    staging.mkdir(mode=0o700)
    marker = staging / "foreign-user-data"
    marker.write_bytes(b"preserve exact-name invalid staging")
    marker.chmod(0o600)
    opened_fds: list[int] = []
    original_open_staging = runtime._open_staging

    def tracked_open_staging(*args: Any, **kwargs: Any) -> Any:
        opened = original_open_staging(*args, **kwargs)
        opened_fds.append(opened.directory_fd)
        return opened

    monkeypatch.setattr(runtime, "_open_staging", tracked_open_staging)
    with pytest.raises(PermissionError, match="preserved invalid"):
        runtime._reserve_exact_attempt("a" * 64)
    assert marker.read_bytes() == b"preserve exact-name invalid staging"
    assert not attempt.exists()
    assert opened_fds
    for descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    sys.modules.pop(runtime.__name__, None)


def test_v8_foreign_staging_prefix_is_preserved_and_blocks_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    foreign = seed_root / f"{runtime.STAGING_PREFIX}foreign"
    foreign.mkdir(parents=True, mode=0o700)
    metric_parent = policy.CANONICAL_METRIC_RECEIPT_PATH.parent
    gate_parent = policy.CANONICAL_GATE_PATH.parent
    metric_parent.mkdir()
    gate_parent.mkdir()
    for directory in (
        policy.CANONICAL_OUTPUT_ROOT,
        seed_root.parent,
        seed_root,
        foreign,
        metric_parent,
        gate_parent,
    ):
        directory.chmod(0o700)
    lock = seed_root / runtime.LOCK_NAME
    lock.write_bytes(b"")
    lock.chmod(0o600)
    marker = foreign / "foreign-user-data"
    marker.write_bytes(b"preserve me")
    opened_staging_fds: list[int] = []
    original_open_staging = runtime._open_staging

    def tracked_open_staging(*args: Any, **kwargs: Any) -> Any:
        staging = original_open_staging(*args, **kwargs)
        opened_staging_fds.append(staging.directory_fd)
        return staging

    monkeypatch.setattr(runtime, "_open_staging", tracked_open_staging)
    with pytest.raises(PermissionError, match="unproved recovery inventory"):
        runtime._reserve_exact_attempt("a" * 64)
    assert marker.read_bytes() == b"preserve me"
    assert foreign.is_dir()
    assert not attempt.exists()
    assert not opened_staging_fds
    for descriptor in opened_staging_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize("location", ["output", "seed", "metric"])
def test_v8_historical_foreign_inventory_blocks_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    output = policy.CANONICAL_OUTPUT_ROOT
    attempts = seed_root.parent
    metric = policy.CANONICAL_METRIC_RECEIPT_PATH.parent
    gate = policy.CANONICAL_GATE_PATH.parent
    seed_root.mkdir(parents=True)
    metric.mkdir()
    gate.mkdir()
    for directory in (output, attempts, seed_root, metric, gate):
        directory.chmod(0o700)
    parent = {"output": output, "seed": seed_root, "metric": metric}[location]
    marker = parent / "foreign-marker"
    marker.write_bytes(b"historical foreign bytes")
    with pytest.raises(PermissionError, match="unproved recovery inventory"):
        runtime._reserve_exact_attempt("a" * 64)
    assert marker.read_bytes() == b"historical foreign bytes"
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_existing_nonprivate_exclusive_root_blocks_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root, attempt, _review = _patch_temporary_reservation_authority(
        runtime, tmp_path, monkeypatch
    )
    output = policy.CANONICAL_OUTPUT_ROOT
    seed_root.mkdir(parents=True)
    for directory in (output, seed_root.parent, seed_root):
        directory.chmod(0o700)
    output.chmod(0o777)
    with pytest.raises(PermissionError, match="canonical component changed"):
        runtime._reserve_exact_attempt("a" * 64)
    assert stat.S_IMODE(output.stat().st_mode) == 0o777
    assert not attempt.exists()
    sys.modules.pop(runtime.__name__, None)


def test_v8_recovered_staging_rejects_leaf_rewrite_before_watch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    staging_name = f"{runtime.STAGING_PREFIX}{'a' * 32}"
    staging_path = seed_root / staging_name
    staging_path.mkdir(mode=0o700)
    leaf = staging_path / "reservation.json"
    leaf.write_bytes(b"original")
    seed_fd = os.open(seed_root, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(seed_fd, label=str(seed_root))
    original_watch = journal.watch_directory

    def rewrite_then_watch(
        directory_fd: int,
        *,
        label: str,
        expected_snapshot: Any = None,
    ) -> None:
        if os.fstat(directory_fd).st_ino == staging_path.stat().st_ino:
            descriptor = os.open(
                "reservation.json",
                os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                assert os.write(descriptor, b"changed!") == len(b"changed!")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        original_watch(
            directory_fd,
            label=label,
            expected_snapshot=expected_snapshot,
        )

    monkeypatch.setattr(journal, "watch_directory", rewrite_then_watch)
    try:
        with pytest.raises(PermissionError, match="before its watch was installed"):
            runtime._open_staging(
                seed_root,
                seed_fd,
                journal,
                staging_name,
            )
        assert leaf.read_bytes() == b"changed!"
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(seed_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_all_production_child_watches_bind_expected_snapshots() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "watch_directory"
    ]
    assert len(calls) == 4
    assert all(
        any(keyword.arg == "expected_snapshot" for keyword in call.keywords)
        for call in calls
    )


@pytest.mark.parametrize(
    ("kind", "mask"),
    [
        ("overflow", 0x00004000),
        ("unmount", 0x00002000),
        ("move_self", 0x00000800),
        ("delete_self", 0x00000400),
        ("ignored", 0x00008000),
        ("unknown_mask", 0x00000001),
    ],
)
def test_v8_special_or_unknown_event_permanently_poisons_success(
    tmp_path: Path,
    kind: str,
    mask: int,
) -> None:
    runtime = _load_nested_executor()
    directory_fd = os.open(tmp_path, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(directory_fd, label="watched")
    state = journal._state(directory_fd)
    original = journal._events
    event_watch = -1 if kind == "overflow" else state.watch
    journal._events = lambda: [
        runtime.JournalEventV8(event_watch, mask, 0, "")
    ]
    try:
        with pytest.raises(PermissionError):
            journal.assert_clean()
        journal._events = original
        with pytest.raises(PermissionError):
            journal.mkdir(directory_fd, "never-success")
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(directory_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_unknown_watch_and_watch_descriptor_reuse_poison(
    tmp_path: Path,
) -> None:
    runtime = _load_nested_executor()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_fd = os.open(first, runtime._directory_flags())
    second_fd = os.open(second, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(first_fd, label="first")
    original_events = journal._events
    state = journal._state(first_fd)
    journal._events = lambda: [
        runtime.JournalEventV8(state.watch + 1000, journal.IN_CREATE, 0, "x")
    ]
    try:
        with pytest.raises(PermissionError, match="unknown or reused"):
            journal.assert_clean()
    finally:
        journal._events = original_events
        journal.close()
        os.close(first_fd)
        os.close(second_fd)
        sys.modules.pop(runtime.__name__, None)

    runtime = _load_nested_executor()
    first_fd = os.open(first, runtime._directory_flags())
    second_fd = os.open(second, runtime._directory_flags())
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(first_fd, label="first")
    active_watch = journal._state(first_fd).watch
    journal._add_watch = lambda *_args: active_watch
    try:
        with pytest.raises(PermissionError, match="reused before retirement"):
            journal.watch_directory(second_fd, label="second")
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(first_fd)
        os.close(second_fd)
        sys.modules.pop(runtime.__name__, None)


@pytest.mark.parametrize("corruption", ["cookie", "order"])
def test_v8_directory_move_cookie_and_event_order_are_exact(
    tmp_path: Path,
    corruption: str,
) -> None:
    runtime = _load_nested_executor()
    (tmp_path / "child").mkdir()
    parent_fd = os.open(tmp_path, runtime._directory_flags())
    child_fd = os.open("child", runtime._directory_flags(), dir_fd=parent_fd)
    journal = runtime.OwnedDirectoryJournalV8()
    journal.watch_directory(parent_fd, label="parent")
    journal.watch_directory(child_fd, label="child")
    original = journal._events

    def corrupted() -> list[Any]:
        events = original()
        if len(events) == 3 and events[0].mask & journal.IN_MOVED_FROM:
            if corruption == "cookie":
                events[1] = runtime.JournalEventV8(
                    events[1].watch,
                    events[1].mask,
                    events[1].cookie + 1,
                    events[1].name,
                )
            else:
                events[0], events[1] = events[1], events[0]
        return events

    journal._events = corrupted
    try:
        with pytest.raises(PermissionError, match="rename event sequence"):
            journal.rename_directory(
                parent_fd,
                "child",
                "claimed",
                directory_fd=child_fd,
            )
        assert journal.poisoned is True
    finally:
        journal.close()
        os.close(child_fd)
        os.close(parent_fd)
        sys.modules.pop(runtime.__name__, None)


def test_v8_frozen_v4_block_and_all_static_parents_rehash() -> None:
    for relative, expected in FROZEN_V4.items():
        assert _sha(ROOT / relative) == expected
    for relative, expected in FROZEN_V5.items():
        assert _sha(ROOT / relative) == expected
    static = policy.preflight_static_authority()
    assert static["v1_block_content_sha256"] == policy.V1_BLOCK_CONTENT_SHA256
    assert static["v2_block_content_sha256"] == policy.V2_BLOCK_CONTENT_SHA256
    assert static["v3_block_content_sha256"] == policy.V3_BLOCK_CONTENT_SHA256
    assert static["v4_block_content_sha256"] == policy.V4_BLOCK_CONTENT_SHA256
    assert static["v5_terminal"] == {
        "source_review_content_sha256": policy.V5_REVIEW_RECORD_CONTENT_SHA256,
        "reservation_content_sha256": policy.V5_RESERVATION_CONTENT_SHA256,
        "failure_content_sha256": policy.V5_FAILURE_CONTENT_SHA256,
        "numeric_payload_survived": False,
        "retry_authorized": False,
    }
    assert static["v7_terminal"] == policy.v7_terminal_failure_binding()
    assert policy.retained_v7_artifact_bindings() == dict(
        policy.RETAINED_V7_ARTIFACT_BINDINGS
    )
    assert policy.ISOLATED_VERIFIER_AMENDMENT_FILE_SHA256 == (
        "9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211"
    )


def test_v8_public_exports_include_isolated_verifier_and_v7_closure() -> None:
    assert {
        "isolated_verifier_contract",
        "retained_v7_artifact_bindings",
        "v7_terminal_failure_binding",
    } <= set(policy.__all__)


def test_v8_recovery_reservation_validates_against_frozen_science() -> None:
    reservation = json.loads((ROOT / policy.V5_RESERVATION_RELATIVE_PATH).read_text())
    review = {
        "path": policy.SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    reservation["schema"] = policy.RESERVATION_SCHEMA
    reservation["scope"] = (
        "one_exclusive_fresh_infrastructure_replacement_attempt"
    )
    reservation["experiment"] = policy.experiment_contract()
    reservation["authority_bindings"] = policy.authority_bindings()
    reservation["source_review"] = review
    reservation["inputs"]["source_review_file_sha256"] = "a" * 64
    reservation["inputs"]["source_review_content_sha256"] = "b" * 64
    core = dict(reservation)
    core.pop("content_sha256")
    reservation["content_sha256"] = policy.canonical_json_sha256(core)
    assert policy.validate_reservation_structure(
        reservation,
        expected_source_review=review,
    ) == reservation


def test_v8_import_exposes_no_callable_operation_and_performs_no_lifecycle_work() -> None:
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


def test_v8_production_source_defines_partial_stages_only_inside_script_entry() -> None:
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
        "AttemptReservationV8",
        "_reserve_exact_attempt",
        "_publish_success",
        "_write_canonical_json",
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
        "execute_exact",
    } <= nested


def test_v8_no_constructible_stage_evidence_class_is_importable() -> None:
    exposed = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type)
        and value.__module__ == executor.__name__
        and any(term in name.casefold() for term in ("reservation", "attempt", "completion"))
    }
    assert exposed == set()


def test_v8_reservation_copy_reconstruction_and_mutation_surface_is_absent() -> None:
    assert not hasattr(executor, "AttemptReservationV8")
    assert not any("reservation" in name.casefold() for name in vars(executor))


def test_v8_completion_writer_is_not_importable() -> None:
    assert not hasattr(executor, "_publish_success")
    assert not hasattr(executor, "_terminate_failure")


@pytest.mark.parametrize("name", ["_write_canonical_json", "_write_bytes_exclusive"])
def test_v8_metric_and_gate_writers_are_not_importable(name: str) -> None:
    assert not hasattr(executor, name)


def test_v8_publication_rejects_replaced_claim_directory(tmp_path: Path) -> None:
    operation = SyntheticExecutionV8(tmp_path / "synthetic")
    reservation = operation.claim()
    moved = tmp_path / "moved-original-attempt"
    operation.attempt.rename(moved)
    operation.attempt.mkdir()
    try:
        with pytest.raises(
            PermissionError,
            match="claimed directory|directory identity|claim identity|undeclared mutation",
        ):
            operation.publish(reservation, b'{"status":"completed"}\n')
        assert list(operation.attempt.iterdir()) == []
        assert sorted(path.name for path in moved.iterdir()) == ["reservation.json"]
    finally:
        operation.close(reservation)


def test_v8_shared_generated_direct_child_churn_does_not_abort_claim(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "repo/.generated"
    operation = SyntheticExecutionV8(generated / "exclusive-v8")
    reservation = operation.claim()
    unrelated = generated / "unrelated-concurrent-test"
    try:
        unrelated.mkdir()
        unrelated.rmdir()
        operation.publish(reservation, b'{"status":"completed"}\n')
        assert sorted(path.name for path in operation.attempt.iterdir()) == [
            "completed.json",
            "reservation.json",
        ]
    finally:
        operation.close(reservation)


def test_v8_real_chain_allows_shared_churn_and_blocks_alias_and_owned_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _load_nested_executor()
    root = tmp_path / "repo"
    generated = root / ".generated"
    output = generated / "camera-output-v8"
    seed_root = output / "attempts/seed_20260710"
    seed_root.mkdir(parents=True)
    for directory in (output, seed_root.parent, seed_root):
        directory.chmod(0o700)
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    chain = runtime._open_canonical_directory_chain(seed_root)
    try:
        unrelated = generated / "unrelated-direct-child"
        unrelated.mkdir()
        unrelated.rmdir()
        runtime._assert_directory_chain(chain)

        foreign = output / "foreign-owned-child"
        foreign.mkdir()
        with pytest.raises(PermissionError, match="journal|event|mutation"):
            runtime._assert_directory_chain(chain)
        assert chain.journal.poisoned is True
    finally:
        runtime._close_directory_chain(chain)
        sys.modules.pop(runtime.__name__, None)


def test_v8_shared_generated_alias_swap_is_rejected(tmp_path: Path) -> None:
    generated = tmp_path / "repo/.generated"
    operation = SyntheticExecutionV8(generated / "exclusive-v8")
    reservation = operation.claim()
    moved = tmp_path / "moved-generated"
    generated.rename(moved)
    generated.symlink_to(moved, target_is_directory=True)
    try:
        with pytest.raises(PermissionError, match="directory identity|undeclared mutation"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        failure = operation.terminate(
            reservation,
            RuntimeError("shared ancestor alias replacement"),
            stage="training",
        )
        assert failure["retry_authorized"] is False
        assert (moved / "exclusive-v8/attempts/seed_20260710/n5/failed.json").is_file()
    finally:
        operation.close(reservation)


def test_v8_shared_generated_permission_change_is_rejected(tmp_path: Path) -> None:
    generated = tmp_path / "repo/.generated"
    operation = SyntheticExecutionV8(generated / "exclusive-v8")
    reservation = operation.claim()
    os.chmod(generated, 0o750)
    try:
        with pytest.raises(PermissionError, match="directory identity|undeclared mutation"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        operation.terminate(
            reservation,
            RuntimeError("shared ancestor security mutation"),
            stage="training",
        )
    finally:
        os.chmod(generated, 0o755)
        operation.close(reservation)


def test_v8_unowned_exclusive_subtree_mutation_is_rejected(tmp_path: Path) -> None:
    operation = SyntheticExecutionV8(tmp_path / "repo/.generated/exclusive-v8")
    reservation = operation.claim()
    foreign = operation.root / "foreign-unowned-child"
    foreign.mkdir()
    try:
        with pytest.raises(
            PermissionError,
            match="directory identity|undeclared mutation",
        ):
            operation.publish(reservation, b'{"status":"completed"}\n')
        operation.terminate(
            reservation,
            RuntimeError("exclusive subtree mutation"),
            stage="training",
        )
    finally:
        operation.close(reservation)


def test_v8_unowned_claim_child_is_rejected_and_terminalized(tmp_path: Path) -> None:
    operation = SyntheticExecutionV8(tmp_path / "repo/.generated/exclusive-v8")
    reservation = operation.claim()
    foreign = operation.attempt / "foreign-unowned-child"
    foreign.write_bytes(b"not owned by the lifecycle")
    try:
        with pytest.raises(PermissionError, match="claimed directory|undeclared mutation"):
            operation.publish(reservation, b'{"status":"completed"}\n')
        failure = operation.terminate(
            reservation,
            RuntimeError("claim inventory mutation"),
            stage="training",
        )
        assert failure["retry_authorized"] is False
        assert foreign.read_bytes() == b"not owned by the lifecycle"
        assert (operation.attempt / "failed.json").is_file()
    finally:
        operation.close(reservation)


def test_v8_review_preflight_rejects_canonical_leaf_alias(
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


def test_v8_review_preflight_accepts_only_the_regular_canonical_entry(
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


def test_v8_source_reader_rejects_parent_identity_change(
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


def test_v8_source_reader_walks_from_filesystem_root_without_absolute_repo_open(
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


def test_v8_source_reader_rejects_transient_root_component_alias_before_any_read(
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


def test_v8_source_reader_rejects_restored_transient_ancestor_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "v8-root-anchor"
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
            moved = tmp_path / "v8-moved-root-anchor"
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


def test_v8_source_reader_rejects_same_inode_parent_metadata_change(
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


def test_v8_source_reader_rejects_symlink_hardlink_and_fifo(tmp_path: Path) -> None:
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
def test_v8_synthetic_operation_rejects_production_namespaces(unsafe: Path) -> None:
    with pytest.raises(PermissionError, match="production|repository"):
        SyntheticExecutionV8(unsafe)


def test_v8_synthetic_recovery_is_single_use_and_descriptor_bound(tmp_path: Path) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
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


def test_v8_synthetic_existing_scaffold_without_staging_blocks_claim(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
    seed_root = operation.attempt.parent
    attempts = seed_root.parent
    derived = operation.root / "derived"
    seed_root.mkdir(parents=True)
    derived.mkdir()
    for directory in (operation.root, attempts, seed_root, derived):
        directory.chmod(0o700)
    lock = seed_root / SYNTHETIC_LOCK_NAME
    lock.write_bytes(b"")
    lock.chmod(0o600)
    with pytest.raises(PermissionError, match="lacks one complete staging"):
        operation.claim()
    assert lock.read_bytes() == b""
    assert not operation.attempt.exists()


def test_v8_synthetic_multiple_equivalent_stagings_resume_one(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
    stagings = [operation.prepare_complete_staging() for _ in range(2)]
    reservation = operation.claim()
    try:
        assert not any(staging.exists() for staging in stagings)
        classifications = {
            item["classification"] for item in reservation.value["recovery"]
        }
        assert classifications == {"complete", "complete_equivalent_duplicate"}
    finally:
        operation.close(reservation)


def test_v8_synthetic_foreign_staging_is_preserved_and_blocks_claim(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
    complete = operation.prepare_complete_staging()
    foreign = operation.attempt.parent / ".n5.synthetic-v8-000-foreign"
    foreign.mkdir(mode=0o700)
    marker = foreign / "foreign-user-data"
    marker.write_bytes(b"preserve")
    with pytest.raises(PermissionError, match="recovery inventory is unproved"):
        operation.claim()
    assert marker.read_bytes() == b"preserve"
    assert complete.is_dir()
    assert not operation.attempt.exists()


def test_v8_synthetic_historical_foreign_inventory_blocks_claim(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
    marker = operation.root / "foreign-marker"
    marker.write_bytes(b"historical")
    with pytest.raises(PermissionError, match="recovery inventory is unproved"):
        operation.claim()
    assert marker.read_bytes() == b"historical"
    assert not operation.attempt.exists()


def test_v8_success_rejects_replaced_output_ancestor_but_fd_failure_terminalizes(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
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
            "checkpoint.pt",
            "failed.json",
            "reservation.json",
        ]
        assert sorted(path.name for path in (moved_root / "derived").iterdir()) == [
            "metric.json"
        ]
        assert failure["owned_directory_journal"]["integrity"] == "failed"
        assert {
            item["outcome"] for item in failure["artifact_cleanup"]
        } == {"journal_integrity_failed_preserved_invalid"}
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
def test_v8_post_training_failure_removes_only_owned_partials_and_is_no_retry(
    tmp_path: Path,
    stage: str,
    derived_names: tuple[str, ...],
) -> None:
    operation = SyntheticExecutionV8(tmp_path / stage)
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
            SyntheticExecutionV8(operation.root).claim()
    finally:
        operation.close(reservation)


def test_v8_terminalization_preserves_mutated_artifacts_as_invalid(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV8(tmp_path / "safe")
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


def test_v8_concurrent_synthetic_claims_choose_exactly_one(tmp_path: Path) -> None:
    operations = [SyntheticExecutionV8(tmp_path / "safe") for _ in range(2)]

    def claim(operation: SyntheticExecutionV8) -> object:
        try:
            return operation.claim()
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, operations))
    reservations = [value for value in outcomes if value != "rejected"]
    assert len(reservations) == 1
    SyntheticExecutionV8.close(reservations[0])  # type: ignore[arg-type]


def test_v8_preclaim_cleanup_and_postclaim_failure_are_durable(tmp_path: Path) -> None:
    before = SyntheticExecutionV8(tmp_path / "before")
    with pytest.raises(RuntimeError, match="before atomic"):
        before.claim(failure_injection="before_atomic_claim")
    assert not before.attempt.exists()

    after = SyntheticExecutionV8(tmp_path / "after")
    with pytest.raises(RuntimeError, match="after atomic"):
        after.claim(failure_injection="after_atomic_claim")
    assert sorted(path.name for path in after.attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    with pytest.raises(FileExistsError, match="already claimed"):
        SyntheticExecutionV8(after.root).claim()


def test_v8_retained_source_output_and_claim_chains_are_present_in_production() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    reserve = functions["_reserve_exact_attempt"]
    chain = functions["_open_canonical_directory_chain"]
    identity = functions["_identity_security_fingerprint"]
    full = functions["_stable_fingerprint"]
    writer = functions["_write_canonical_json"]
    assert "os.open(filesystem_root, _directory_flags())" in chain
    assert "dir_fd=parent_fd" in chain
    assert "anchor_identity_security" in chain
    assert "_identity_security_fingerprint" in chain
    assert "exclusive" in chain
    assert "policy.CANONICAL_OUTPUT_ROOT" in chain
    for field in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid"):
        assert field in identity
    for unrelated_child_field in ("st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"):
        assert unrelated_child_field not in identity
        assert unrelated_child_field in full
    assert "_open_canonical_directory_chain(seed_root)" in reserve
    assert "dir_fd=seed_root_fd" in reserve
    assert "os.fstat(claimed_directory_fd)" in reserve
    assert "claimed_directory_fd = active_staging.directory_fd" in reserve
    assert "journal.rename_directory" in reserve
    assert reserve.index("claimed_directory_fd = active_staging.directory_fd") < (
        reserve.index("journal.rename_directory")
    )
    assert "directory_fd=claimed_directory_fd" in reserve
    assert "directory_fingerprint=_stable_fingerprint(staging_metadata)" in reserve
    assert "directory_chain=claimed_chain" in reserve
    assert "reservation.journal.create_file" in writer
    assert "parent_fd" in writer
    assert "reservation.output_root_fd" in writer
    assert "os.open(path" not in writer
    assert "_write_claim_file_exclusive" in functions["_publish_success"]
    assert "_read_claim_file" in functions["_artifact_args"]
    assert "os.listdir(reservation.directory_fd)" in functions["_assert_owned_claim"]
    claim_writer = functions["_write_claim_file_exclusive"]
    assert "reservation.journal.create_file" in claim_writer
    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert "_refresh_directory_chain" not in source
    assert "_refresh_claim_directory" not in source
    assert "mutable_fds" not in source
    assert "shutil.rmtree" not in source


def test_v8_post_training_exception_terminalizes_before_all_descriptors_close() -> None:
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


def test_v8_exact_entry_rejects_a_claim_but_preserves_preclaim_recovery() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    execute = functions["execute_exact"]
    reservation = functions["_reservation_core"]
    assert "policy.CANONICAL_ATTEMPT_PATH.exists()" in execute
    assert "policy.CANONICAL_ATTEMPT_PATH.is_symlink()" in execute
    assert "recovery attempt is already claimed" in execute
    assert "policy.CANONICAL_OUTPUT_ROOT.exists()" not in execute
    assert "one_exclusive_fresh_infrastructure_replacement_attempt" in reservation
    assert policy.OUTPUT_ROOT_RELATIVE_PATH.endswith("n5_full_panel_recovery_v8")
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_v8_frozen_science_and_gpu0_launcher_are_unchanged() -> None:
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    experiment = policy.experiment_contract()
    retained_experiment = dict(retained.EXPERIMENT)
    assert {
        key: value for key, value in experiment.items() if key != "output_path"
    } == {
        key: value for key, value in retained_experiment.items() if key != "output_path"
    }
    assert experiment["output_path"] == str(policy.CANONICAL_ATTEMPT_PATH)
    assert experiment["output_path"] != retained_experiment["output_path"]
    bindings = policy.authority_bindings()
    assert {
        key: bindings[key] for key in retained.AUTHORITY_BINDINGS
    } == retained.AUTHORITY_BINDINGS
    assert bindings["lifecycle_recovery_amendment"]["file_sha256"] == (
        policy.RECOVERY_AMENDMENT_FILE_SHA256
    )
    assert bindings["v5_terminal_reservation"]["file_sha256"] == (
        policy.V5_RESERVATION_FILE_SHA256
    )
    assert bindings["v5_terminal_failure"]["file_sha256"] == (
        policy.V5_FAILURE_FILE_SHA256
    )
    assert bindings["v8_isolated_verifier_amendment"]["file_sha256"] == (
        policy.ISOLATED_VERIFIER_AMENDMENT_FILE_SHA256
    )
    assert bindings["v7_terminal_reservation"]["file_sha256"] == (
        policy.V7_RESERVATION_FILE_SHA256
    )
    assert bindings["v7_terminal_failure"]["file_sha256"] == (
        policy.V7_FAILURE_FILE_SHA256
    )
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
    assert '"HSA_VISIBLE_DEVICES",' in source
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in source
    assert 'environment[name] = "1"' in source


def test_v8_review_contract_binds_end_to_end_sources_and_v4_block_closures() -> None:
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
    assert core["infrastructure_replacement_authorized"] is True
    assert core["scientific_retry_authorized"] is False
    assert core["v5_numeric_payload_inspected"] is False
    assert core["v6_numeric_payload_inspected"] is False
    assert core["v7_numeric_payload_inspected"] is False
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
    assert core["execution_contract"]["shared_ancestor_child_churn_tolerated"] is True
    assert core["execution_contract"]["shared_ancestor_identity_and_security_bound"] is True
    assert core["execution_contract"]["exclusive_subtree_full_metadata_bound"] is True
    assert core["reservation_contract"][
        "claimed_directory_descriptor_retained_end_to_end"
    ] is True
    assert core["reservation_contract"][
        "canonical_claim_parent_chain_retained_end_to_end"
    ] is True
    assert core["reservation_contract"][
        "owned_derived_partials_removed_before_failure_terminalization"
    ] is True
    assert core["reservation_contract"]["new_exclusive_output_namespace"] is True
    assert core["v5_terminal_evidence"]["numeric_payload_survived"] is False
    assert core["v5_terminal_evidence"]["numeric_payload_inspected"] is False
    assert core["v5_terminal_evidence"]["retry_authorized"] is False
    assert core["v7_terminal_failure_evidence"] == (
        policy.v7_terminal_failure_binding()
    )
    assert core["retained_v7_artifacts"] == (
        policy.retained_v7_artifact_bindings()
    )
    assert core["authority_boundary"][
        "v5_v6_or_v7_numeric_state_authorized"
    ] is False
    assert core["isolated_verifier_contract"] == (
        policy.isolated_verifier_contract()
    )
    assert core["execution_contract"]["fresh_isolated_verifier_child"] is True
    assert core["execution_contract"]["verifier_child_compute_only"] is True
    assert core["execution_contract"][
        "verifier_child_canonical_publication"
    ] is False
    assert core["execution_contract"][
        "verifier_parent_publication_only"
    ] is True
    assert core["execution_contract"]["verifier_in_process_fallback"] is False
    assert core["execution_contract"]["verifier_request_response_bound"] is True
    assert set(core["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
