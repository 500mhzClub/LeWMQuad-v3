"""Adversarial source, lifecycle, and durability closure for full-panel V3."""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import inspect
import os
from pathlib import Path
import shutil
from types import FunctionType, ModuleType
from typing import Any

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v3 as policy,
)
from lewm.tests.n5_full_panel_v3_synthetic_execution import SyntheticExecutionV3
from scripts import execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3 as executor


ROOT = Path(__file__).resolve().parents[2]


def _sha(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_v1_v2_sources_handoffs_and_block_evidence_are_immutable() -> None:
    for relative, digest in policy.RETAINED_V1_SOURCE_BINDINGS:
        assert _sha(relative) == digest
    for relative, digest in policy.RETAINED_V2_SOURCE_BINDINGS:
        assert _sha(relative) == digest
    assert _sha(policy.V1_REVIEW_RELATIVE_PATH) == policy.V1_REVIEW_FILE_SHA256
    assert _sha(policy.V1_BLOCK_RELATIVE_PATH) == policy.V1_BLOCK_FILE_SHA256
    assert _sha(policy.V1_EXPLOIT_TEST_RELATIVE_PATH) == policy.V1_EXPLOIT_TEST_FILE_SHA256
    assert _sha(policy.V1_HANDOFF_RELATIVE_PATH) == policy.V1_HANDOFF_FILE_SHA256
    assert _sha(policy.V2_REVIEW_RELATIVE_PATH) == policy.V2_REVIEW_FILE_SHA256
    assert _sha(policy.V2_BLOCK_RELATIVE_PATH) == policy.V2_BLOCK_FILE_SHA256
    assert _sha(policy.V2_EXPLOIT_TEST_RELATIVE_PATH) == policy.V2_EXPLOIT_TEST_FILE_SHA256
    assert _sha(policy.V2_HANDOFF_RELATIVE_PATH) == policy.V2_HANDOFF_FILE_SHA256
    static = policy.preflight_static_authority()
    assert static["v1_block_content_sha256"] == policy.V1_BLOCK_CONTENT_SHA256
    assert static["v2_block_content_sha256"] == policy.V2_BLOCK_CONTENT_SHA256


def test_v3_has_no_authority_capability_registry_or_execution_object() -> None:
    forbidden = {
        "verify_authority",
        "require_verified_authority",
        "transition_authority",
        "create_test_authority_capability",
        "VerifiedAuthority",
        "VerifiedAuthorityV2",
        "TestAuthorityCapabilityV2",
        "_AuthorityRecord",
    }
    assert forbidden.isdisjoint(vars(policy))
    for module in (policy, executor):
        for name, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                assert not any(word in name.casefold() for word in ("authority", "capability", "issuer", "token", "record"))
    assert "authority" not in inspect.signature(executor.execute_exact).parameters
    assert set(inspect.signature(executor._reserve_exact_attempt).parameters) == {
        "source_review_file_sha256"
    }


def test_function_defaults_closures_and_defined_referents_expose_no_lifecycle_state() -> None:
    for module in (policy, executor):
        functions = [
            value
            for value in vars(module).values()
            if isinstance(value, FunctionType) and value.__module__ == module.__name__
        ]
        for function in functions:
            values = list(function.__defaults__ or ()) + list((function.__kwdefaults__ or {}).values())
            values.extend(cell.cell_contents for cell in function.__closure__ or ())
            assert not any(isinstance(value, (dict, list, set)) for value in values), function.__name__

            seen: set[int] = set()
            frontier: list[tuple[Any, int]] = [(function, 0)]
            while frontier:
                value, depth = frontier.pop()
                if id(value) in seen or depth > 3:
                    continue
                seen.add(id(value))
                if isinstance(value, dict):
                    keys = {str(key).casefold() for key in value}
                    assert not ({"state", "issuance_digest", "authority"} <= keys)
                    assert not any(isinstance(key, int) for key in value)
                if isinstance(value, ModuleType) or isinstance(value, type):
                    continue
                for referent in gc.get_referents(value):
                    if isinstance(referent, (str, bytes, int, float, bool, type(None))):
                        continue
                    frontier.append((referent, depth + 1))


def test_production_source_has_no_test_path_injection_or_heavy_import() -> None:
    tree = ast.parse((ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text())
    top_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not ({"torch", "numpy", "PIL"} & top_imports)
    exact = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_exact")
    assert [argument.arg for argument in exact.args.args + exact.args.kwonlyargs] == [
        "source_review_file_sha256",
        "rgb_workers",
    ]
    reserve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_reserve_exact_attempt")
    assert [argument.arg for argument in reserve.args.args] == ["source_review_file_sha256"]
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_nonisolated_exact_entry_rejects_before_output() -> None:
    with pytest.raises(PermissionError, match="requires isolation"):
        executor.execute_exact("0" * 64, rgb_workers=5)
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


@pytest.mark.parametrize(
    "unsafe",
    [ROOT, ROOT / "lewm", policy.CANONICAL_OUTPUT_ROOT, ROOT.parent],
)
def test_synthetic_executor_structurally_rejects_production_namespaces(unsafe: Path) -> None:
    with pytest.raises(PermissionError, match="cannot target production|repository"):
        SyntheticExecutionV3(unsafe)


def test_synthetic_copy_deepcopy_reconstruction_and_replay_cannot_reclaim(tmp_path: Path) -> None:
    operation = SyntheticExecutionV3(tmp_path / "safe")
    copies = (copy.copy(operation), copy.deepcopy(operation), SyntheticExecutionV3(operation.root))
    reservation = operation.claim()
    assert reservation.directory == operation.attempt
    for clone in copies:
        with pytest.raises(FileExistsError, match="already claimed"):
            clone.claim()


def test_concurrent_synthetic_operations_claim_exactly_once(tmp_path: Path) -> None:
    operations = [SyntheticExecutionV3(tmp_path / "safe") for _ in range(2)]

    def claim(operation: SyntheticExecutionV3) -> str:
        try:
            operation.claim()
            return "claimed"
        except FileExistsError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, operations))
    assert sorted(outcomes) == ["claimed", "rejected"]
    assert sorted(path.name for path in operations[0].attempt.iterdir()) == ["reservation.json"]


def test_complete_staging_is_rehashed_resumed_and_cross_root_transfer_rejected(tmp_path: Path) -> None:
    first = SyntheticExecutionV3(tmp_path / "first")
    staged = first.prepare_complete_staging()
    reservation = first.claim()
    assert not staged.exists()
    assert "complete" in {item["classification"] for item in reservation.value["recovery"]}

    second = SyntheticExecutionV3(tmp_path / "second")
    foreign = first.prepare_complete_staging()
    second.attempt.parent.mkdir(parents=True, exist_ok=True)
    transferred = second.attempt.parent / foreign.name
    shutil.move(foreign, transferred)
    second_reservation = second.claim()
    assert not transferred.exists()
    assert "mutated" in {item["classification"] for item in second_reservation.value["recovery"]}


@pytest.mark.parametrize("kind", ["incomplete", "foreign", "mutated"])
def test_stale_synthetic_states_are_cleaned_without_stranding(tmp_path: Path, kind: str) -> None:
    operation = SyntheticExecutionV3(tmp_path / "safe")
    operation.attempt.parent.mkdir(parents=True, exist_ok=True)
    staging = operation.attempt.parent / f".n5.synthetic-v3-{kind}"
    if kind == "foreign":
        staging.write_bytes(b"foreign")
    else:
        staging.mkdir(mode=0o700)
        (staging / "reservation.json").write_bytes(b"{}" if kind == "incomplete" else b'{"bad":true}')
        if kind == "mutated":
            (staging / "staging.json").write_bytes(b"{}")
    reservation = operation.claim()
    assert operation.attempt.is_dir()
    assert not staging.exists()
    classes = {item["classification"] for item in reservation.value["recovery"]}
    assert kind in classes or (kind == "foreign" and "foreign" in classes)


def test_preclaim_failure_cleans_and_postclaim_failure_is_terminal_and_durable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = SyntheticExecutionV3(tmp_path / "before")
    with pytest.raises(RuntimeError, match="before atomic"):
        before.claim(failure_injection="before_atomic_claim")
    assert not before.attempt.exists()
    assert not any(path.name.startswith(".n5.synthetic-v3-") for path in before.attempt.parent.iterdir())

    after = SyntheticExecutionV3(tmp_path / "after")
    events: list[tuple[str, Path]] = []
    original = __import__("lewm.tests.n5_full_panel_v3_synthetic_execution", fromlist=["_fsync"])
    real_fsync = original._fsync
    real_rename = original.os.rename

    def fsync(path: Path) -> None:
        events.append(("fsync", Path(path)))
        real_fsync(path)

    def rename(source: Path, target: Path) -> None:
        events.append(("rename", Path(target)))
        real_rename(source, target)

    monkeypatch.setattr(original, "_fsync", fsync)
    monkeypatch.setattr(original.os, "rename", rename)
    with pytest.raises(RuntimeError, match="after atomic"):
        after.claim(failure_injection="after_atomic_claim")
    rename_index = events.index(("rename", after.attempt))
    assert events[rename_index + 1] == ("fsync", after.attempt.parent)
    assert events[-2:] == [("fsync", after.attempt), ("fsync", after.attempt.parent)]
    assert sorted(path.name for path in after.attempt.iterdir()) == ["failed.json", "reservation.json"]
    with pytest.raises(FileExistsError, match="already claimed"):
        SyntheticExecutionV3(after.root).claim()


def test_production_rename_is_immediately_followed_by_parent_fsync() -> None:
    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    reserve = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_reserve_exact_attempt")

    def statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
        rows: list[list[ast.stmt]] = []
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                rows.append(value)
                for item in value:
                    rows.extend(statement_lists(item))
            elif isinstance(value, ast.AST):
                rows.extend(statement_lists(value))
        return rows

    found = False
    for statements in statement_lists(reserve):
        for index, statement in enumerate(statements[:-1]):
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "rename"
            ):
                continue
            following = statements[index + 1]
            found = (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Call)
                and isinstance(following.value.func, ast.Name)
                and following.value.func.id == "_fsync_directory"
                and isinstance(following.value.args[0], ast.Name)
                and following.value.args[0].id == "seed_root"
            )
    assert found


def test_frozen_scientific_contract_and_cpu_smoke_are_unchanged() -> None:
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained

    assert policy.experiment_contract() == retained.EXPERIMENT
    assert policy.authority_bindings() == retained.AUTHORITY_BINDINGS
    assert policy.frozen_source_bindings() == retained.FROZEN_SOURCE_BINDINGS
    smoke = executor.run_cpu_contract_smoke()
    assert smoke["schedule_sha256"] == policy.EXPECTED_SCHEDULE_SHA256
    assert smoke["update_count"] == 400
    assert smoke["frame_exposures"] == 2000
    assert smoke["every_update_is_full_panel"] is True


def test_source_review_contract_demands_different_agent_and_binds_only_end_to_end_sources() -> None:
    sources = {
        relative: {"path": relative, "file_sha256": _sha(relative)}
        for relative in policy.SUCCESSOR_SOURCE_PATHS
    }
    core = policy.expected_source_review_core(
        reviewer="/root/different_agent",
        successor_sources=sources,
    )
    assert core["implementation_author"] == policy.IMPLEMENTATION_AUTHOR
    assert core["reviewer"] != policy.IMPLEMENTATION_AUTHOR
    assert core["execution_contract"]["caller_held_authority"] is False
    assert core["execution_contract"]["caller_held_capability"] is False
    assert core["execution_contract"]["single_isolated_end_to_end_operation"] is True
    assert set(core["successor_sources"]) == set(policy.SUCCESSOR_SOURCE_PATHS)
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
