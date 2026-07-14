"""Different-agent conformance review for camera full-panel V5.

All dynamic checks use temporary, production-ineligible paths.  The executor's
nested functions are compiled from the frozen source without its final CLI
dispatch so the real descriptor and terminalization code can be exercised.
No experiment input, RGB, accelerator, or canonical output is opened.
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
from typing import Any, Iterator
import uuid

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained_v1,
)
from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v5 as policy,
)
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5 as executor
from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base_trainer


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/camera_v5_independent"
FROZEN_ARTIFACTS = {
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py": (
        "cc28934be4fe1109feae3a31803e9e09502e968591268f80fc7124ba0a63f2c1"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py": (
        "5dcc77a7434b64d3ae759b563b16db95e909bec9d1751dacc7657f6a740ac2e1"
    ),
    "lewm/tests/n5_full_panel_v5_synthetic_execution.py": (
        "7601341cd92beb1a9a6738d2534e6f654a4058fe7d84b07547ac75f674fef608"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v5.py": (
        "80f51db295cad4d2a8494d1c61a1f605dac12cf558b5137d0eeee15611d88264"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v5_"
    "implementation_handoff_2026-07-13.md": (
        "df3d58eff6b582a113beb9d558c3e210f7a22acd38763f55037ae86609dc8b5c"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review_2026-07-13.md": (
        "7edeff73d6022a4086706907b03084ff080c9ad1d52ae91e8659fc6ecdc6b18c"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review.py": (
        "2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v4_"
    "independent_review_block_2026-07-13.json": (
        "d2224049a4ee2b793737802d06d91757c17d20b0457c1624517467638173c507"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_nested_executor() -> ModuleType:
    """Load frozen nested definitions without invoking dispatch."""

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
    review_tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    module_name = f"_lewm_camera_v5_independent_{uuid.uuid4().hex}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(review_tree, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def runtime() -> Iterator[ModuleType]:
    module = _load_nested_executor()
    try:
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


def _patch_paths(
    runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    output = root / "camera-output"
    attempt = output / "attempts/seed_20260710/n5"
    metric = output / "metric_verifications/seed_20260710_n5.json"
    gate = output / "gates/seed_20260710_n5.json"
    monkeypatch.setattr(runtime, "ROOT", root)
    monkeypatch.setattr(policy, "ROOT", root)
    monkeypatch.setattr(policy, "CANONICAL_OUTPUT_ROOT", output)
    monkeypatch.setattr(policy, "CANONICAL_ATTEMPT_PATH", attempt)
    monkeypatch.setattr(policy, "CANONICAL_METRIC_RECEIPT_PATH", metric)
    monkeypatch.setattr(policy, "CANONICAL_GATE_PATH", gate)
    return output, attempt, metric, gate


def _make_reservation(
    runtime: ModuleType,
    output: Path,
    attempt: Path,
    metric: Path,
    gate: Path,
) -> tuple[Any, list[int]]:
    chain = runtime._open_canonical_directory_chain(attempt.parent)
    runtime._open_chain_child(chain, output, metric.parent.name)
    runtime._open_chain_child(chain, output, gate.parent.name)
    seed_fd = chain.path_fds[attempt.parent]
    os.mkdir(attempt.name, 0o700, dir_fd=seed_fd)
    os.fsync(seed_fd)
    runtime._refresh_directory_chain(chain, mutable_fds={seed_fd})
    claim_fd = os.open(attempt.name, runtime._directory_flags(), dir_fd=seed_fd)
    claim_stat = os.fstat(claim_fd)
    core = {"schema": policy.RESERVATION_SCHEMA, "status": "reserved"}
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    reservation = runtime.AttemptReservationV5(
        directory=attempt,
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        directory_fd=claim_fd,
        directory_identity=(claim_stat.st_dev, claim_stat.st_ino),
        directory_chain=chain,
    )
    runtime._write_claim_file_exclusive(
        reservation,
        "reservation.json",
        raw,
        role="reservation",
    )
    return reservation, [claim_fd, *chain.descriptors]


def _derived_value(schema: str, *, marker: str) -> dict[str, Any]:
    core = {"schema": schema, "status": marker}
    return {**core, "content_sha256": policy.canonical_json_sha256(core)}


def _close(runtime: ModuleType, reservation: Any) -> None:
    try:
        os.close(reservation.directory_fd)
    except OSError:
        pass
    if reservation.directory_chain is not None:
        try:
            runtime._close_directory_chain(reservation.directory_chain)
        except OSError:
            pass


def _assert_descriptors_closed(descriptors: list[int]) -> None:
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_v5_independent_frozen_candidate_and_v4_block_rehash() -> None:
    assert {relative: _sha(ROOT / relative) for relative in FROZEN_ARTIFACTS} == (
        FROZEN_ARTIFACTS
    )
    block = json.loads(
        (ROOT / policy.V4_BLOCK_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert block["content_sha256"] == policy.V4_BLOCK_CONTENT_SHA256
    assert str(block["status"]).startswith("blocked_")
    static = policy.preflight_static_authority()
    assert static["v4_block_content_sha256"] == policy.V4_BLOCK_CONTENT_SHA256


def test_v5_independent_import_exposes_no_lifecycle_and_opens_no_output() -> None:
    assert {
        name
        for name, value in vars(executor).items()
        if isinstance(value, FunctionType) and value.__module__ == executor.__name__
    } == set()
    assert {
        name
        for name, value in vars(executor).items()
        if isinstance(value, type) and value.__module__ == executor.__name__
    } == set()
    for name in (
        "AttemptReservationV5",
        "_reserve_exact_attempt",
        "_publish_success",
        "_terminate_failure",
        "_write_canonical_json",
        "execute_exact",
    ):
        assert not hasattr(executor, name)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_v5_independent_lifecycle_definitions_exist_only_under_script_entry() -> None:
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
        "CanonicalDirectoryChainV5",
        "AttemptReservationV5",
        "_open_canonical_directory_chain",
        "_reserve_exact_attempt",
        "_terminate_failure",
        "execute_exact",
        "dispatch",
    } <= nested


def test_v5_independent_source_walk_rejects_restored_ancestor_alias_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = tmp_path / "source-anchor"
    root = anchor / "repo"
    source = root / "nested/source.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen source\n")
    moved = tmp_path / "moved-source-anchor"
    original_open = os.open
    original_read = os.read
    attacked = False
    reads = 0

    def attacked_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if not attacked and path == anchor.name and dir_fd is not None:
            attacked = True
            anchor.rename(moved)
            anchor.symlink_to(moved, target_is_directory=True)
            try:
                return original_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                anchor.unlink()
                moved.rename(anchor)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def counted_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "open", attacked_open)
    monkeypatch.setattr(policy.os, "read", counted_read)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, "nested/source.py", name="source")
    assert attacked is True
    assert reads == 0


def test_v5_independent_source_walk_rejects_same_inode_component_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    parent = root / "nested"
    parent.mkdir(parents=True)
    (parent / "source.py").write_bytes(b"source\n")
    original_read = os.read
    mutated = False

    def mutate_then_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        if not mutated:
            mutated = True
            os.chmod(parent, 0o750)
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "read", mutate_then_read)
    with pytest.raises((PermissionError, RuntimeError), match="changed|mutation"):
        policy.read_regular_bytes_at(root, "nested/source.py", name="source")
    assert mutated


@pytest.mark.parametrize("failure", [False, True])
def test_v5_independent_source_walk_closes_every_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
) -> None:
    root = tmp_path / "repo"
    source = root / "nested/source.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source\n")
    original_open = os.open
    original_read = os.read
    descriptors: list[int] = []
    injected = False

    def record_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        descriptors.append(descriptor)
        return descriptor

    def maybe_fail(descriptor: int, size: int) -> bytes:
        nonlocal injected
        if failure and not injected:
            injected = True
            raise RuntimeError("injected source read failure")
        return original_read(descriptor, size)

    monkeypatch.setattr(policy.os, "open", record_open)
    monkeypatch.setattr(policy.os, "read", maybe_fail)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            policy.read_regular_bytes_at(root, "nested/source.py", name="source")
    else:
        assert policy.read_regular_bytes_at(root, "nested/source.py", name="source") == (
            b"source\n"
        )
    assert descriptors
    _assert_descriptors_closed(descriptors)


@pytest.mark.parametrize("leaf_kind", ["symlink", "hardlink", "fifo"])
def test_v5_independent_source_walk_rejects_nonexclusive_leaf(
    tmp_path: Path,
    leaf_kind: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    original = root / "original.py"
    original.write_bytes(b"source\n")
    candidate = root / "candidate.py"
    if leaf_kind == "symlink":
        candidate.symlink_to(original)
    elif leaf_kind == "hardlink":
        os.link(original, candidate)
    else:
        os.mkfifo(candidate)
    with pytest.raises((PermissionError, RuntimeError, OSError)):
        policy.read_regular_bytes_at(root, candidate.name, name="candidate")


def test_v5_independent_real_output_chain_rejects_ancestor_alias(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    reservation, _descriptors = _make_reservation(
        runtime, output, attempt, metric, gate
    )
    moved = output.with_name("moved-camera-output")
    output.rename(moved)
    output.symlink_to(moved, target_is_directory=True)
    try:
        with pytest.raises(PermissionError, match="directory component"):
            runtime._assert_owned_claim(reservation)
    finally:
        _close(runtime, reservation)


def test_v5_independent_real_claim_check_rejects_replacement(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    reservation, _descriptors = _make_reservation(
        runtime, output, attempt, metric, gate
    )
    moved = attempt.with_name("n5-owned-moved")
    attempt.rename(moved)
    attempt.mkdir()
    try:
        with pytest.raises(PermissionError, match="directory|claim"):
            runtime._assert_owned_claim(reservation)
    finally:
        _close(runtime, reservation)


@pytest.mark.parametrize("stage", ["verification", "finalization"])
def test_v5_independent_real_terminalization_removes_exact_owned_partials(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    reservation, descriptors = _make_reservation(
        runtime, output, attempt, metric, gate
    )
    try:
        for name in ("checkpoint.pt", "result.json", "completed.json"):
            runtime._write_claim_file_exclusive(
                reservation, name, name.encode("ascii"), role=name
            )
        runtime._write_canonical_json(
            reservation,
            metric,
            _derived_value(policy.METRIC_RECEIPT_SCHEMA, marker="verified"),
        )
        if stage == "finalization":
            runtime._write_canonical_json(
                reservation,
                gate,
                _derived_value(policy.GATE_SCHEMA, marker="finalized"),
            )
        failure = runtime._terminate_failure(
            reservation, RuntimeError("injected post-training failure"), stage=stage
        )
        value = policy.parse_json(
            (attempt / "failed.json").read_bytes(), name="failure"
        )
        assert value["failure_stage"] == stage
        assert value["retry_authorized"] is False
        assert value["partial_artifacts_removed"] is True
        assert {item["outcome"] for item in value["artifact_cleanup"]} == {
            "removed_owned"
        }
        assert sorted(path.name for path in attempt.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        assert not metric.exists()
        assert not gate.exists()
        assert failure["path"] == "failed.json"
    finally:
        _close(runtime, reservation)
    _assert_descriptors_closed(descriptors)


def test_v5_independent_real_terminalization_preserves_changed_and_foreign(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    reservation, _descriptors = _make_reservation(
        runtime, output, attempt, metric, gate
    )
    try:
        runtime._write_claim_file_exclusive(
            reservation, "checkpoint.pt", b"owned checkpoint", role="checkpoint"
        )
        runtime._write_claim_file_exclusive(
            reservation, "result.json", b"owned result", role="result"
        )
        runtime._write_canonical_json(
            reservation,
            metric,
            _derived_value(policy.METRIC_RECEIPT_SCHEMA, marker="verified"),
        )
        (attempt / "checkpoint.pt").write_bytes(b"changed checkpoint bytes")
        metric.write_bytes(b"changed metric bytes")
        (attempt / "foreign.bin").write_bytes(b"foreign")
        gate.write_bytes(b"foreign gate")
        runtime._terminate_failure(
            reservation, RuntimeError("verification failure"), stage="verification"
        )
        value = policy.parse_json(
            (attempt / "failed.json").read_bytes(), name="failure"
        )
        mismatches = {
            item["artifact"]
            for item in value["artifact_cleanup"]
            if item["outcome"] == "ownership_mismatch_preserved_invalid"
        }
        assert mismatches == {
            "checkpoint.pt",
            "metric_verifications/seed_20260710_n5.json",
        }
        assert value["partial_artifacts_removed"] is False
        assert not (attempt / "result.json").exists()
        assert (attempt / "checkpoint.pt").read_bytes() == b"changed checkpoint bytes"
        assert metric.read_bytes() == b"changed metric bytes"
        assert (attempt / "foreign.bin").read_bytes() == b"foreign"
        assert gate.read_bytes() == b"foreign gate"
    finally:
        _close(runtime, reservation)


def test_v5_independent_failure_uses_claim_fd_after_canonical_replacement(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    reservation, _descriptors = _make_reservation(
        runtime, output, attempt, metric, gate
    )
    runtime._write_claim_file_exclusive(
        reservation, "checkpoint.pt", b"owned checkpoint", role="checkpoint"
    )
    moved = attempt.with_name("n5-owned-moved")
    attempt.rename(moved)
    attempt.mkdir()
    try:
        runtime._terminate_failure(
            reservation, RuntimeError("claim replacement"), stage="verification"
        )
        assert sorted(path.name for path in moved.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        assert list(attempt.iterdir()) == []
    finally:
        _close(runtime, reservation)


@pytest.mark.parametrize(
    ("stage", "after_publication"),
    [
        ("verification", False),
        ("verification", True),
        ("finalization", False),
        ("finalization", True),
    ],
)
def test_v5_independent_execute_terminalizes_each_post_training_boundary(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    after_publication: bool,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        runtime, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=True))
    )
    monkeypatch.setattr(policy, "preflight_static_authority", lambda: {})
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda _path, _sha256: ({"content_sha256": "a" * 64}, b"review\n"),
    )

    def fake_training(_review_sha: str, *, rgb_workers: int) -> tuple[dict[str, Any], Any]:
        assert rgb_workers == 1
        reservation, descriptors = _make_reservation(
            runtime, output, attempt, metric, gate
        )
        for name in ("checkpoint.pt", "result.json", "completed.json"):
            runtime._write_claim_file_exclusive(
                reservation, name, name.encode("ascii"), role=name
            )
        holder.update(reservation=reservation, descriptors=descriptors)
        return {"synthetic": True}, reservation

    def write_metric(reservation: Any) -> dict[str, Any]:
        value = _derived_value(policy.METRIC_RECEIPT_SCHEMA, marker="verified")
        runtime._write_canonical_json(reservation, metric, value)
        return value

    def fake_verification(reservation: Any, _review_sha: str) -> dict[str, Any]:
        if stage == "verification" and not after_publication:
            raise RuntimeError("injected verification before publication")
        value = write_metric(reservation)
        if stage == "verification":
            raise RuntimeError("injected verification after publication")
        return value

    def fake_finalization(reservation: Any, _review_sha: str) -> dict[str, Any]:
        if not after_publication:
            raise RuntimeError("injected finalization before publication")
        value = _derived_value(policy.GATE_SCHEMA, marker="finalized")
        runtime._write_canonical_json(reservation, gate, value)
        raise RuntimeError("injected finalization after publication")

    monkeypatch.setattr(runtime, "_run_frozen_training", fake_training)
    monkeypatch.setattr(runtime, "_run_independent_verification", fake_verification)
    monkeypatch.setattr(runtime, "_run_finalization", fake_finalization)
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        runtime.execute_exact("a" * 64, rgb_workers=1)

    failure = policy.parse_json((attempt / "failed.json").read_bytes(), name="failure")
    assert failure["failure_stage"] == stage
    assert failure["retry_authorized"] is False
    assert sorted(path.name for path in attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    assert not metric.exists()
    assert not gate.exists()
    _assert_descriptors_closed(holder["descriptors"])
    with pytest.raises(FileExistsError, match="already claimed"):
        runtime.execute_exact("a" * 64, rgb_workers=1)


def test_v5_independent_secondary_terminalization_failure_still_closes_descriptors(
    runtime: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, attempt, metric, gate = _patch_paths(
        runtime, monkeypatch, tmp_path / "repo"
    )
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        runtime, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=True))
    )
    monkeypatch.setattr(policy, "preflight_static_authority", lambda: {})
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda _path, _sha256: ({"content_sha256": "a" * 64}, b"review\n"),
    )

    def fake_training(_review_sha: str, *, rgb_workers: int) -> tuple[dict[str, Any], Any]:
        assert rgb_workers == 1
        reservation, descriptors = _make_reservation(
            runtime, output, attempt, metric, gate
        )
        holder.update(reservation=reservation, descriptors=descriptors)
        return {"synthetic": True}, reservation

    monkeypatch.setattr(runtime, "_run_frozen_training", fake_training)
    monkeypatch.setattr(
        runtime,
        "_run_independent_verification",
        lambda _reservation, _sha256: (_ for _ in ()).throw(
            RuntimeError("verification failed")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_terminate_failure",
        lambda _reservation, _error, *, stage: (_ for _ in ()).throw(
            OSError(f"terminalization failed at {stage}")
        ),
    )
    with pytest.raises(RuntimeError, match="terminal receipt could not be written"):
        runtime.execute_exact("a" * 64, rgb_workers=1)
    _assert_descriptors_closed(holder["descriptors"])


def test_v5_independent_frozen_science_and_gpu_contract_are_unchanged() -> None:
    assert policy.experiment_contract() == retained_v1.EXPERIMENT
    assert policy.authority_bindings() == retained_v1.AUTHORITY_BINDINGS
    schedule = base_trainer._deterministic_training_batches(
        frame_count=5,
        batch_size=5,
        steps=400,
        seed=20260710,
    )
    assert base_trainer.canonical_json_sha256(schedule) == (
        policy.EXPECTED_SCHEDULE_SHA256
    )
    assert len(schedule) == 400
    assert sum(map(len, schedule)) == 2000
    assert all(len(batch) == 5 and set(batch) == set(range(5)) for batch in schedule)
    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    assert '[sys.executable, "-I", "-B"' in source
    assert 'environment["HIP_VISIBLE_DEVICES"] = "0"' in source
    assert 'environment.pop("HSA_OVERRIDE_GFX_VERSION", None)' in source
    assert 'environment[name] = "1"' in source


def test_v5_independent_review_core_is_exact_and_canonical_paths_are_absent() -> None:
    sources = {
        relative: {"path": relative, "file_sha256": _sha(ROOT / relative)}
        for relative in policy.SUCCESSOR_SOURCE_PATHS
    }
    core = policy.expected_source_review_core(
        reviewer=REVIEWER,
        successor_sources=sources,
    )
    assert core["reviewer"] == REVIEWER
    assert core["reviewer"] != policy.IMPLEMENTATION_AUTHOR
    assert set(core["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)
    assert core["execution_contract"]["filesystem_root_anchored_source_walk"] is True
    assert core["execution_contract"]["post_training_failure_terminalization"] is True
    assert core["reservation_contract"][
        "canonical_claim_parent_chain_retained_end_to_end"
    ] is True
    assert core["reservation_contract"][
        "owned_derived_partials_removed_before_failure_terminalization"
    ] is True
    assert core["licenses"]["authorizes_retry"] is False
    assert core["licenses"]["authorizes_n16_execution"] is False
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
