"""Independent directory-consistency regression for camera full-panel V6.

Every lifecycle operation is isolated below ``tmp_path``. No canonical camera
output, experiment input, checkpoint, model, training path, or accelerator is
opened.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import uuid

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v6 as policy,
)
from lewm.tests.n5_full_panel_v6_synthetic_execution import SyntheticExecutionV6


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_auditor_author"
FROZEN_V6 = {
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
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_"
    "implementation_handoff_2026-07-13.md": (
        "4ca14a5d8392d88c4d9779d82ef4eb3f1655317ed61c8e51490651877e3e57e1"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_nested_executor() -> ModuleType:
    source_path = ROOT / policy.EXECUTOR_RELATIVE_PATH
    tree = ast.parse(
        source_path.read_text(encoding="ascii"), filename=str(source_path)
    )
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
    name = f"_lewm_camera_v6_directory_consistency_{uuid.uuid4().hex}"
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


def _reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Any, Path]:
    runtime = _load_nested_executor()
    root = tmp_path / "repository"
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
    directory_fd = os.open(attempt.name, runtime._directory_flags(), dir_fd=seed_fd)
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
    return runtime, reservation, generated


def _close(runtime: ModuleType, reservation: Any) -> None:
    os.close(reservation.directory_fd)
    runtime._close_directory_chain(reservation.directory_chain)
    sys.modules.pop(runtime.__name__, None)


def _create_remove(parent_fd: int) -> None:
    descriptor = os.open(
        "foreign-restored-child",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=parent_fd,
    )
    os.close(descriptor)
    os.unlink("foreign-restored-child", dir_fd=parent_fd)
    os.fsync(parent_fd)


def test_v6_directory_consistency_frozen_candidate_rehashes() -> None:
    assert {relative: _sha256(ROOT / relative) for relative in FROZEN_V6} == FROZEN_V6


def test_v6_claim_refresh_rejects_interleaved_foreign_create_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _reservation(tmp_path, monkeypatch)
    original = runtime._refresh_claim_directory

    def interleaved(value: Any) -> None:
        _create_remove(value.directory_fd)
        original(value)

    runtime._refresh_claim_directory = interleaved
    try:
        with pytest.raises(PermissionError, match="unexpected|mutation|inventory"):
            runtime._write_claim_file_exclusive(
                reservation,
                "checkpoint.pt",
                b"owned checkpoint",
                role="training_checkpoint",
            )
    finally:
        _close(runtime, reservation)


def test_v6_derived_refresh_rejects_interleaved_foreign_create_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _reservation(tmp_path, monkeypatch)
    original = runtime._refresh_directory_chain

    def interleaved(chain: Any, *, mutable_fds: set[int]) -> None:
        for parent_fd in mutable_fds:
            _create_remove(parent_fd)
        original(chain, mutable_fds=mutable_fds)

    runtime._refresh_directory_chain = interleaved
    core = {"schema": policy.METRIC_RECEIPT_SCHEMA, "status": "synthetic"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    try:
        with pytest.raises(PermissionError, match="unexpected|mutation|identity"):
            runtime._write_canonical_json(
                reservation,
                policy.CANONICAL_METRIC_RECEIPT_PATH,
                value,
            )
    finally:
        _close(runtime, reservation)


def test_v6_normal_owned_claim_write_still_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, reservation, _generated = _reservation(tmp_path, monkeypatch)
    try:
        runtime._write_claim_file_exclusive(
            reservation,
            "checkpoint.pt",
            b"owned checkpoint",
            role="training_checkpoint",
        )
        runtime._assert_owned_claim(reservation)
    finally:
        _close(runtime, reservation)


def test_v6_shared_parent_create_delete_still_passes(tmp_path: Path) -> None:
    generated = tmp_path / "repository" / ".generated"
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
