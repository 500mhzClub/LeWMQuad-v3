"""Different-agent adversarial review of camera full-panel V6.

All lifecycle operations use temporary synthetic namespaces.  The only V5
payloads read are the two explicitly admitted terminal JSON receipts.  No
numeric experiment payload, exact attempt, accelerator, or canonical V6 output
is opened.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any, Iterator, Mapping
import uuid

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v6 as policy,
)
from lewm.tests.n5_full_panel_v6_synthetic_execution import SyntheticExecutionV6
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6 as executor


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_auditor_v1_independent"
FROZEN_V6 = {
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_lifecycle_recovery_amendment_2026-07-13.md": (
        "1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py": (
        "75b987dc97c21e2689caea8df4fb316a80b6602cf8a612e47abe02bf14a5d549"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py": (
        "791103400c6093c40abed5c87009d4a18feceda1c5155c2d06dae97b2bb38a3d"
    ),
    "lewm/tests/n5_full_panel_v6_synthetic_execution.py": (
        "8df835debcc24f7fd1b77f5cc0f559215023c9111d3c2ff5ae367129296a496f"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py": (
        "2af8b43439ce2b72cc9c22cd1a3d48028c66e3b18cd2b2b742ddf0b147ce017b"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_implementation_handoff_2026-07-13.md": (
        "4ca14a5d8392d88c4d9779d82ef4eb3f1655317ed61c8e51490651877e3e57e1"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_nested_executor() -> ModuleType:
    source_path = ROOT / policy.EXECUTOR_RELATIVE_PATH
    tree = ast.parse(source_path.read_text(encoding="ascii"), filename=str(source_path))
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
    name = f"_lewm_camera_v6_independent_{uuid.uuid4().hex}"
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[name] = module
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
        sys.modules.pop(name, None)
        raise
    return module


def _production_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Any, Path, Path]:
    runtime = _load_nested_executor()
    root = tmp_path / "repo"
    generated = root / ".generated"
    output = generated / "camera-v6"
    seed_root = output / "attempts" / "seed_20260710"
    attempt = seed_root / "n5"
    metric_parent = output / "metric_verifications"
    gate_parent = output / "gates"
    metric = metric_parent / "seed_20260710_n5.json"
    gate = gate_parent / "seed_20260710_n5.json"
    attempt.mkdir(parents=True)
    metric_parent.mkdir()
    gate_parent.mkdir()
    reservation_raw = b'{"content_sha256":"' + b"0" * 64 + b'"}\n'
    (attempt / "reservation.json").write_bytes(reservation_raw)
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    monkeypatch.setattr(policy, "CANONICAL_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(policy, "CANONICAL_METRIC_RECEIPT_PATH", metric)
    monkeypatch.setattr(policy, "CANONICAL_GATE_PATH", gate)
    chain = runtime._open_canonical_directory_chain(seed_root)
    runtime._open_chain_child(chain, output, metric_parent.name)
    runtime._open_chain_child(chain, output, gate_parent.name)
    seed_fd = chain.path_fds[seed_root]
    directory_fd = os.open(
        attempt.name,
        runtime._directory_flags(),
        dir_fd=seed_fd,
    )
    metadata = os.fstat(directory_fd)
    reservation = runtime.AttemptReservationV6(
        directory=attempt,
        value={"source_review": {}},
        raw=reservation_raw,
        file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
        directory_fd=directory_fd,
        directory_identity=(metadata.st_dev, metadata.st_ino),
        directory_fingerprint=runtime._stable_fingerprint(metadata),
        directory_chain=chain,
    )
    runtime._assert_owned_claim(reservation)
    return runtime, reservation, generated, output


def _close_runtime(runtime: ModuleType, reservation: Any) -> None:
    os.close(reservation.directory_fd)
    if reservation.directory_chain is not None:
        runtime._close_directory_chain(reservation.directory_chain)
    sys.modules.pop(runtime.__name__, None)


def _entry_copy(metadata: os.stat_result, **changes: int) -> SimpleNamespace:
    fields = {
        name: getattr(metadata, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def _temporary_review(root: Path) -> tuple[Path, str]:
    sources: dict[str, dict[str, str]] = {}
    for index, relative in enumerate(policy.SUCCESSOR_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"independent V6 source {index}\n".encode("ascii"))
        sources[relative] = {"path": relative, "file_sha256": _sha(path)}
    core = policy.expected_source_review_core(
        reviewer=REVIEWER,
        successor_sources=sources,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    review = root / policy.SOURCE_REVIEW_RELATIVE_PATH
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_bytes(raw)
    return review, hashlib.sha256(raw).hexdigest()


def test_v6_independent_frozen_candidate_and_v5_terminal_receipts_rehash() -> None:
    assert {relative: _sha(ROOT / relative) for relative in FROZEN_V6} == FROZEN_V6
    assert _sha(ROOT / policy.V5_RESERVATION_RELATIVE_PATH) == (
        policy.V5_RESERVATION_FILE_SHA256
    )
    assert _sha(ROOT / policy.V5_FAILURE_RELATIVE_PATH) == policy.V5_FAILURE_FILE_SHA256
    assert _sha(ROOT / policy.V5_REVIEW_RECORD_RELATIVE_PATH) == (
        policy.V5_REVIEW_RECORD_FILE_SHA256
    )


def test_v6_independent_v5_admits_only_terminal_receipts_no_numeric_payload() -> None:
    value = policy._validate_v5_terminal_incident()
    assert value == {
        "source_review_content_sha256": policy.V5_REVIEW_RECORD_CONTENT_SHA256,
        "reservation_content_sha256": policy.V5_RESERVATION_CONTENT_SHA256,
        "failure_content_sha256": policy.V5_FAILURE_CONTENT_SHA256,
        "numeric_payload_survived": False,
        "retry_authorized": False,
    }
    attempt = (ROOT / policy.V5_RESERVATION_RELATIVE_PATH).parent
    assert sorted(item.name for item in attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]


def test_v6_independent_import_has_no_partial_lifecycle_surface() -> None:
    functions = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, FunctionType) and value.__module__ == executor.__name__
    }
    classes = {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type) and value.__module__ == executor.__name__
    }
    assert functions == set()
    assert classes == set()
    for name in (
        "execute_exact",
        "_reserve_exact_attempt",
        "_run_frozen_training",
        "_run_independent_verification",
        "_run_finalization",
        "_publish_success",
        "_terminate_failure",
    ):
        assert not hasattr(executor, name)


def test_v6_independent_review_and_sources_are_exactly_path_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    review, digest = _temporary_review(root)
    monkeypatch.setattr(policy, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_SOURCE_REVIEW_PATH", review)
    value, reread = policy.preflight_source_review(review, digest)
    assert value["reviewer"] == REVIEWER
    assert hashlib.sha256(reread).hexdigest() == digest
    with pytest.raises(PermissionError, match="not canonical"):
        policy.preflight_source_review(review.parent / "other.json", digest)
    source_parent = root / Path(policy.POLICY_RELATIVE_PATH).parent
    moved = tmp_path / "moved-policy-parent"
    source_parent.rename(moved)
    source_parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(PermissionError, match="real directory|component"):
        policy.preflight_source_review(review, digest)


def test_v6_independent_shared_direct_child_churn_passes(tmp_path: Path) -> None:
    generated = tmp_path / "repo" / ".generated"
    operation = SyntheticExecutionV6(generated / "camera-v6")
    reservation = operation.claim()
    unrelated = generated / "unrelated"
    try:
        unrelated.mkdir()
        unrelated.rmdir()
        operation.publish(reservation, b'{"status":"completed"}\n')
        assert (operation.attempt / "completed.json").is_file()
    finally:
        operation.close(reservation)


@pytest.mark.parametrize("mutation", ("alias", "inode", "type", "mode"))
def test_v6_independent_shared_ancestor_mutations_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    runtime, reservation, generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    moved = tmp_path / "moved-generated"
    original_mode = stat.S_IMODE(generated.stat().st_mode)
    try:
        if mutation == "mode":
            os.chmod(generated, original_mode ^ stat.S_IXGRP)
        else:
            generated.rename(moved)
            if mutation == "alias":
                generated.symlink_to(moved, target_is_directory=True)
            elif mutation == "inode":
                generated.mkdir()
            else:
                generated.write_bytes(b"not a directory")
        with pytest.raises(PermissionError, match="component"):
            runtime._assert_directory_chain(reservation.directory_chain)
    finally:
        if generated.is_symlink() or generated.is_file():
            generated.unlink()
        elif mutation == "inode" and generated.exists():
            generated.rmdir()
        elif mutation == "mode" and generated.exists():
            os.chmod(generated, original_mode)
        if moved.exists():
            moved.rename(generated)
        _close_runtime(runtime, reservation)


def test_v6_independent_transient_restored_alias_is_observed_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, reservation, generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    moved = tmp_path / "transient-generated"
    original = runtime._entry_metadata
    injected = False

    def transient(parent_fd: int, name: str) -> os.stat_result:
        nonlocal injected
        if not injected and name == generated.name:
            injected = True
            generated.rename(moved)
            generated.symlink_to(moved, target_is_directory=True)
            metadata = original(parent_fd, name)
            generated.unlink()
            moved.rename(generated)
            return metadata
        return original(parent_fd, name)

    runtime._entry_metadata = transient
    try:
        with pytest.raises(PermissionError, match="component"):
            runtime._assert_directory_chain(reservation.directory_chain)
    finally:
        _close_runtime(runtime, reservation)


@pytest.mark.parametrize("field", ("st_uid", "st_gid"))
def test_v6_independent_shared_owner_and_group_mutation_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    runtime, reservation, generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    original = runtime._entry_metadata

    def changed(parent_fd: int, name: str) -> Any:
        metadata = original(parent_fd, name)
        if name == generated.name:
            return _entry_copy(metadata, **{field: getattr(metadata, field) + 1})
        return metadata

    runtime._entry_metadata = changed
    try:
        with pytest.raises(PermissionError, match="component"):
            runtime._assert_directory_chain(reservation.directory_chain)
    finally:
        _close_runtime(runtime, reservation)


def test_v6_independent_unowned_output_and_claim_children_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, reservation, _generated, output = _production_reservation(
        tmp_path, monkeypatch
    )
    foreign_output = output / "foreign-output-child"
    foreign_claim = reservation.directory / "foreign-claim-child"
    try:
        foreign_output.mkdir()
        with pytest.raises(PermissionError, match="component"):
            runtime._assert_directory_chain(reservation.directory_chain)
        foreign_output.rmdir()
        runtime._refresh_directory_chain(
            reservation.directory_chain,
            mutable_fds={reservation.output_root_fd},
        )
        foreign_claim.write_bytes(b"foreign")
        with pytest.raises(PermissionError, match="claimed directory"):
            runtime._assert_owned_claim(reservation)
    finally:
        _close_runtime(runtime, reservation)


def test_v6_independent_exact_owned_mutations_refresh_narrowly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, reservation, _generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    before_claim = reservation.directory_fingerprint
    metric_parent = policy.CANONICAL_METRIC_RECEIPT_PATH.parent
    metric_entry = next(
        item
        for item in reservation.directory_chain.entries
        if item.child_fd == reservation.directory_chain.path_fds[metric_parent]
    )
    before_metric = metric_entry.full_fingerprint
    try:
        runtime._write_claim_file_exclusive(
            reservation,
            "checkpoint.pt",
            b"owned checkpoint",
            role="training_checkpoint",
        )
        metric_core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
        metric = {
            **metric_core,
            "content_sha256": policy.canonical_json_sha256(metric_core),
        }
        runtime._write_canonical_json(
            reservation,
            policy.CANONICAL_METRIC_RECEIPT_PATH,
            metric,
        )
        runtime._assert_owned_claim(reservation)
        assert reservation.directory_fingerprint != before_claim
        assert metric_entry.full_fingerprint != before_metric
    finally:
        _close_runtime(runtime, reservation)


def test_v6_block_claim_refresh_absorbs_interleaved_unowned_create_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, reservation, _generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    original = runtime._refresh_claim_directory

    def inject_foreign_history(value: Any) -> None:
        descriptor = os.open(
            "foreign-restored-child",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=value.directory_fd,
        )
        os.close(descriptor)
        os.unlink("foreign-restored-child", dir_fd=value.directory_fd)
        os.fsync(value.directory_fd)
        original(value)

    runtime._refresh_claim_directory = inject_foreign_history
    try:
        with pytest.raises(PermissionError, match="unexpected|mutation|inventory"):
            runtime._write_claim_file_exclusive(
                reservation,
                "checkpoint.pt",
                b"owned checkpoint",
                role="training_checkpoint",
            )
    finally:
        _close_runtime(runtime, reservation)


def test_v6_block_derived_refresh_absorbs_interleaved_unowned_create_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, reservation, _generated, _output = _production_reservation(
        tmp_path, monkeypatch
    )
    original = runtime._refresh_directory_chain

    def inject_foreign_history(chain: Any, *, mutable_fds: set[int]) -> None:
        for parent_fd in mutable_fds:
            descriptor = os.open(
                "foreign-restored-child",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            os.close(descriptor)
            os.unlink("foreign-restored-child", dir_fd=parent_fd)
            os.fsync(parent_fd)
        original(chain, mutable_fds=mutable_fds)

    runtime._refresh_directory_chain = inject_foreign_history
    metric_core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    metric = {
        **metric_core,
        "content_sha256": policy.canonical_json_sha256(metric_core),
    }
    try:
        with pytest.raises(PermissionError, match="unexpected|mutation|identity"):
            runtime._write_canonical_json(
                reservation,
                policy.CANONICAL_METRIC_RECEIPT_PATH,
                metric,
            )
    finally:
        _close_runtime(runtime, reservation)


def test_v6_independent_terminal_cleanup_preserves_foreign_and_is_no_retry(
    tmp_path: Path,
) -> None:
    operation = SyntheticExecutionV6(tmp_path / "camera-v6")
    reservation = operation.claim()
    try:
        operation.publish_claim_artifact(reservation, "checkpoint.pt", b"owned")
        operation.publish_derived_artifact(reservation, "metric.json", b"owned")
        (operation.attempt / "checkpoint.pt").write_bytes(b"foreign mutation")
        failure = operation.terminate(
            reservation,
            RuntimeError("verification failure"),
            stage="verification",
        )
        assert failure["retry_authorized"] is False
        assert (operation.attempt / "failed.json").is_file()
        assert (operation.attempt / "checkpoint.pt").read_bytes() == b"foreign mutation"
        assert not (operation.root / "derived" / "metric.json").exists()
        with pytest.raises(FileExistsError, match="already claimed"):
            SyntheticExecutionV6(operation.root).claim()
    finally:
        operation.close(reservation)


def test_v6_independent_frozen_science_gpu0_and_scope() -> None:
    experiment = policy.experiment_contract()
    assert experiment == {
        "seed": 20260710,
        "fit_size": 5,
        "fresh_model_initialization": True,
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "optimizer": "AdamW",
        "optimizer_updates": 400,
        "training_batch_size": 5,
        "frame_exposures": 2000,
        "evaluation_batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": {name: 0.25 for name in policy.LOSS_COMPONENTS},
        "schedule_algorithm": policy.SCHEDULE_ALGORITHM,
        "schedule_sha256": "62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634",
        "checkpoint_selection": "final_update_only",
        "evaluation_controls": ["matched_rgb", "wrong_rgb_with_target_calibration"],
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "raphael_igpu_forbidden": True,
        "rgb_worker_count_max": 5,
        "native_threads_per_process": 1,
        "attempt_count": 1,
        "output_path": str(policy.CANONICAL_ATTEMPT_PATH),
    }
    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text(encoding="ascii")
    assert '[sys.executable, "-I", "-B"' in source
    assert 'environment["HIP_VISIBLE_DEVICES"] = "0"' in source
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in source
    assert "GPU1" not in source
    licenses = policy.licenses()
    assert licenses["authorizes_one_fresh_n5_full_panel_infrastructure_replacement"]
    assert licenses["authorizes_metric_verification_only_checkpoint_use"]
    assert licenses["authorizes_stage_finalization"]
    assert all(
        value is False
        for name, value in licenses.items()
        if name
        not in {
            "authorizes_one_fresh_n5_full_panel_infrastructure_replacement",
            "authorizes_metric_verification_only_checkpoint_use",
            "authorizes_stage_finalization",
        }
    )


def test_v6_independent_canonical_review_and_output_remain_absent() -> None:
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.is_symlink()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.is_symlink()

