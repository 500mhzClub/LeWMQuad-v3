"""Adversarial authority, recovery, and durability closure for full-panel V2."""
from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import os
from pathlib import Path

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)
from lewm.tests.n5_full_panel_v2_test_support import (
    active_test_authority,
    write_source_review,
)
from scripts import train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as trainer


ROOT = Path(__file__).resolve().parents[2]


def _new_authority(
    capability: policy.TestAuthorityCapabilityV2,
    root: Path,
    attempt: Path,
) -> policy.VerifiedAuthorityV2:
    review = root / "review.json"
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    authority = capability.issue(review, digest, target_path=attempt)
    capability.transition(
        authority,
        target_path=attempt,
        from_states=("issued",),
        to_state="active",
    )
    return authority


def test_frozen_v1_block_and_science_are_unchanged() -> None:
    assert hashlib.sha256((ROOT / policy.V1_REVIEW_RELATIVE_PATH).read_bytes()).hexdigest() == (
        policy.V1_REVIEW_FILE_SHA256
    )
    assert hashlib.sha256((ROOT / policy.V1_BLOCK_RELATIVE_PATH).read_bytes()).hexdigest() == (
        policy.V1_BLOCK_FILE_SHA256
    )
    assert hashlib.sha256(
        (ROOT / policy.V1_EXPLOIT_TEST_RELATIVE_PATH).read_bytes()
    ).hexdigest() == policy.V1_EXPLOIT_TEST_FILE_SHA256
    for relative, expected in policy.RETAINED_V1_SOURCE_BINDINGS.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert policy.EXPERIMENT == policy.retained_v1.EXPERIMENT
    smoke = trainer.run_cpu_contract_smoke()
    assert smoke["schedule_sha256"] == policy.EXPECTED_SCHEDULE_SHA256
    assert smoke["update_count"] == 400
    assert smoke["frame_exposures"] == 2000
    assert smoke["every_update_is_full_panel"] is True


def test_authority_rejects_construction_shell_copy_clone_and_serialization(
    tmp_path: Path,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    with pytest.raises(TypeError, match="issued only"):
        policy.VerifiedAuthorityV2()
    forged = object.__new__(policy.VerifiedAuthorityV2)
    with pytest.raises(PermissionError, match="forged|cloned|reconstructed"):
        capability.validate(forged, target_path=attempt, allowed_states=("active",))
    with pytest.raises(TypeError, match="noncopyable"):
        copy.copy(authority)
    with pytest.raises(TypeError, match="noncopyable"):
        copy.deepcopy(authority)
    with pytest.raises(TypeError, match="dataclass"):
        replace(authority)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="serialized"):
        authority.__reduce_ex__(4)


def test_authority_object_graph_exposes_no_issuer_registry_or_mutable_state(
    tmp_path: Path,
) -> None:
    capability, authority, _attempt = active_test_authority(tmp_path)
    assert "_issuer" not in policy.VerifiedAuthorityV2.__slots__
    assert policy.TestAuthorityCapabilityV2.__slots__ == ()
    assert not hasattr(policy, "_CANONICAL_AUTHORITY_ISSUER")
    assert not hasattr(policy, "_AuthorityIssuerV2")
    with pytest.raises(TypeError):
        vars(authority)
    with pytest.raises(TypeError):
        vars(capability)
    for slot in policy.VerifiedAuthorityV2.__slots__:
        value = object.__getattribute__(authority, slot)
        assert not isinstance(value, (dict, list, set))
        assert "record" not in type(value).__name__.casefold()
        assert "issuer" not in type(value).__name__.casefold()
    record = policy._AuthorityRecord(authority, "0" * 64, "issued")
    with pytest.raises(FrozenInstanceError):
        record.state = "active"  # type: ignore[misc]


def test_authority_mutation_and_test_promotion_are_rejected(tmp_path: Path) -> None:
    capability, authority, attempt = active_test_authority(tmp_path / "a")
    object.__setattr__(authority, "_purpose", "exact_run")
    with pytest.raises(PermissionError, match="mutated"):
        capability.validate(authority, target_path=attempt, allowed_states=("active",))

    capability2, authority2, attempt2 = active_test_authority(tmp_path / "b")
    object.__setattr__(authority2, "_test_only", False)
    with pytest.raises(PermissionError, match="mutated|production|forged|cross-issuer"):
        policy.require_verified_authority(
            authority2,
            purpose="exact_run",
            target_path=attempt2,
            allowed_states=("active",),
        )
    with pytest.raises(PermissionError, match="mutated"):
        capability2.validate(authority2, target_path=attempt2, allowed_states=("active",))


def test_cross_issuer_transfer_and_review_mutation_are_rejected(tmp_path: Path) -> None:
    first, authority, attempt = active_test_authority(tmp_path / "first")
    second, _other, second_attempt = active_test_authority(tmp_path / "second")
    with pytest.raises(PermissionError, match="cross-issuer|forged|cloned"):
        second.validate(authority, target_path=second_attempt, allowed_states=("active",))

    review_path = first.root / "review.json"
    original = review_path.read_bytes()
    review_path.write_bytes(original + b" ")
    with pytest.raises((PermissionError, ValueError), match="changed|SHA-256"):
        first.validate(authority, target_path=attempt, allowed_states=("active",))


def test_one_authority_one_bound_attempt_and_no_production_test_promotion(
    tmp_path: Path,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    reservation = trainer._reserve_attempt_for_test(
        authority,
        test_capability=capability,
        attempt_path=attempt,
    )
    assert reservation.directory == attempt
    with pytest.raises(PermissionError, match="consumed|replayed|one use"):
        trainer._reserve_attempt_for_test(
            authority,
            test_capability=capability,
            attempt_path=attempt,
        )
    with pytest.raises(PermissionError, match="path binding"):
        capability.validate(
            authority,
            target_path=tmp_path / "other/seed_20260710/n5",
            allowed_states=("claimed",),
        )
    with pytest.raises(PermissionError, match="production|forged|exact verified"):
        trainer._reserve_attempt(authority)
    assert "attempt_path" not in inspect.signature(trainer._reserve_attempt).parameters


def test_authority_consumption_is_atomic_under_concurrent_callers(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    attempt = (tmp_path / "attempts/seed_20260710/n5").resolve()
    review = tmp_path / "review.json"
    digest = write_source_review(review)
    capability = policy.create_test_authority_capability(tmp_path)
    authority = capability.issue(review, digest, target_path=attempt)

    def consume() -> str:
        try:
            capability.transition(
                authority,
                target_path=attempt,
                from_states=("issued",),
                to_state="active",
            )
            return "consumed"
        except PermissionError:
            return "replay_rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: consume(), range(2)))
    assert sorted(outcomes) == ["consumed", "replay_rejected"]


def test_concurrent_claimers_create_exactly_one_attempt(tmp_path: Path) -> None:
    capability, first, attempt = active_test_authority(tmp_path)
    second = _new_authority(capability, tmp_path, attempt)

    def claim(authority: policy.VerifiedAuthorityV2) -> str:
        try:
            trainer._reserve_attempt_for_test(
                authority,
                test_capability=capability,
                attempt_path=attempt,
            )
            return "claimed"
        except FileExistsError:
            return "already_claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(claim, (first, second)))
    assert sorted(outcomes) == ["already_claimed", "claimed"]
    assert sorted(path.name for path in attempt.iterdir()) == ["reservation.json"]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("legacy", "incomplete_legacy_v1"),
        ("incomplete", "incomplete"),
        ("foreign", "foreign"),
        ("unreadable", "foreign"),
        ("mutated", "mutated"),
    ],
)
def test_stale_preclaim_states_are_classified_cleaned_and_do_not_strand(
    tmp_path: Path,
    kind: str,
    expected: str,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    attempt.parent.mkdir(parents=True, exist_ok=True)
    if kind == "legacy":
        staging = attempt.parent / trainer.LEGACY_STAGING_NAME
        staging.mkdir()
        (staging / "interrupted-write").write_bytes(b"pre-rename process death")
    elif kind == "foreign":
        staging = attempt.parent / f"{trainer.STAGING_PREFIX}foreign"
        staging.write_bytes(b"not a directory")
    elif kind == "unreadable":
        staging = attempt.parent / f"{trainer.STAGING_PREFIX}unreadable"
        staging.mkdir(mode=0o700)
        (staging / "foreign").write_bytes(b"unreviewed")
        staging.chmod(0o000)
    else:
        staging = attempt.parent / f"{trainer.STAGING_PREFIX}{kind}"
        staging.mkdir(mode=0o700)
        if kind == "incomplete":
            (staging / "reservation.json").write_bytes(b"partial")
        else:
            (staging / "reservation.json").write_bytes(b"{}")
            (staging / "staging.json").write_bytes(b"{}")
    reservation = trainer._reserve_attempt_for_test(
        authority,
        test_capability=capability,
        attempt_path=attempt,
    )
    assert attempt.is_dir()
    assert not staging.exists()
    assert expected in {
        row["classification"] for row in reservation.value["preclaim_recovery"]
    }


def test_complete_staging_is_rehashed_resumed_and_claimed_once(tmp_path: Path) -> None:
    capability, first, attempt = active_test_authority(tmp_path)
    capability.transition(
        first,
        target_path=attempt,
        from_states=("active",),
        to_state="claiming",
    )
    attempt.parent.mkdir(parents=True, exist_ok=True)
    staging = trainer._new_staging(attempt.parent, attempt.name)
    staged = trainer._reservation(
        first,
        attempt_path=attempt,
        recovery_events=(
            {
                "staging_name": staging.name,
                "classification": "new_unique_private",
                "action": "complete_then_atomic_claim",
                "inventory_sha256": policy.canonical_json_sha256([]),
            },
        ),
        test_capability=capability,
    )
    trainer._prepare_new_staging(
        staging,
        reservation=staged,
        attempt_path=attempt,
    )

    resumed_authority = _new_authority(capability, tmp_path, attempt)
    resumed = trainer._reserve_attempt_for_test(
        resumed_authority,
        test_capability=capability,
        attempt_path=attempt,
    )
    assert attempt.is_dir()
    assert sorted(path.name for path in attempt.iterdir()) == ["reservation.json"]
    assert "complete" in {
        row["classification"] for row in resumed.value["preclaim_recovery"]
    }
    second = _new_authority(capability, tmp_path, attempt)
    with pytest.raises(FileExistsError, match="already claimed"):
        trainer._reserve_attempt_for_test(
            second,
            test_capability=capability,
            attempt_path=attempt,
        )


def test_multiple_complete_equivalent_staging_resumes_one_and_removes_rest(
    tmp_path: Path,
) -> None:
    capability, first, attempt = active_test_authority(tmp_path)
    capability.transition(
        first,
        target_path=attempt,
        from_states=("active",),
        to_state="claiming",
    )
    attempt.parent.mkdir(parents=True, exist_ok=True)
    stagings = [trainer._new_staging(attempt.parent, attempt.name) for _ in range(2)]
    for index, staging in enumerate(stagings):
        staged = trainer._reservation(
            first,
            attempt_path=attempt,
            recovery_events=(
                {
                    "staging_name": staging.name,
                    "classification": "new_unique_private",
                    "action": "complete_then_atomic_claim",
                    "inventory_sha256": policy.canonical_json_sha256([index]),
                },
            ),
            test_capability=capability,
        )
        trainer._prepare_new_staging(
            staging,
            reservation=staged,
            attempt_path=attempt,
        )
    resumed_authority = _new_authority(capability, tmp_path, attempt)
    resumed = trainer._reserve_attempt_for_test(
        resumed_authority,
        test_capability=capability,
        attempt_path=attempt,
    )
    assert attempt.is_dir()
    assert not any(staging.exists() for staging in stagings)
    classifications = {
        row["classification"] for row in resumed.value["preclaim_recovery"]
    }
    assert "complete" in classifications
    assert "complete_equivalent_duplicate" in classifications


def test_preclaim_failure_cleans_unique_staging_without_claim(tmp_path: Path) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    with pytest.raises(RuntimeError, match="before atomic"):
        trainer._reserve_attempt_for_test(
            authority,
            test_capability=capability,
            attempt_path=attempt,
            failure_injection="before_atomic_claim",
        )
    assert not attempt.exists()
    assert not any(
        child.name.startswith(trainer.STAGING_PREFIX)
        or child.name == trainer.LEGACY_STAGING_NAME
        for child in attempt.parent.iterdir()
    )
    with pytest.raises(PermissionError, match="consumed|replayed|one use"):
        trainer._reserve_attempt_for_test(
            authority,
            test_capability=capability,
            attempt_path=attempt,
        )


def test_postrename_parent_fsync_precedes_failure_and_terminal_parent_is_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    events: list[tuple[str, Path]] = []
    original_fsync = trainer._fsync_directory
    original_rename = trainer.os.rename

    def recording_fsync(path: Path) -> None:
        events.append(("fsync", Path(path)))
        original_fsync(path)

    def recording_rename(source: Path, target: Path) -> None:
        events.append(("rename", Path(target)))
        original_rename(source, target)

    monkeypatch.setattr(trainer, "_fsync_directory", recording_fsync)
    monkeypatch.setattr(trainer.os, "rename", recording_rename)
    with pytest.raises(RuntimeError, match="after atomic"):
        trainer._reserve_attempt_for_test(
            authority,
            test_capability=capability,
            attempt_path=attempt,
            failure_injection="after_atomic_claim",
        )
    rename_index = events.index(("rename", attempt))
    assert events[rename_index + 1] == ("fsync", attempt.parent)
    assert events[-2:] == [("fsync", attempt), ("fsync", attempt.parent)]
    assert sorted(path.name for path in attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]

    source = (ROOT / policy.TRAINER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    reserve = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reserve_bound_attempt"
    )
    calls = [
        node
        for node in ast.walk(reserve)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rename"
    ]
    assert len(calls) == 1

    def statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
        result: list[list[ast.stmt]] = []
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(
                isinstance(item, ast.stmt) for item in value
            ):
                result.append(value)
                for item in value:
                    result.extend(statement_lists(item))
            elif isinstance(value, ast.AST):
                result.extend(statement_lists(value))
        return result

    adjacency_found = False
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
            adjacency_found = (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Call)
                and isinstance(following.value.func, ast.Name)
                and following.value.func.id == "_fsync_directory"
                and isinstance(following.value.args[0], ast.Name)
                and following.value.args[0].id == "seed_root"
            )
    assert adjacency_found


def test_v2_reservation_schema_retains_frozen_scope_and_recovery_ledger(
    tmp_path: Path,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    reservation = trainer._reserve_attempt_for_test(
        authority,
        test_capability=capability,
        attempt_path=attempt,
    )
    assert reservation.value["experiment"] == policy.EXPERIMENT
    assert reservation.value["licenses"]["retry_authorized"] is False
    assert reservation.value["licenses"]["holdout_authorized"] is False
    assert reservation.value["licenses"]["g2_authorized"] is False
    assert reservation.value["preclaim_recovery"]


def test_v2_sources_are_import_safe_and_exact_output_remains_disabled() -> None:
    for relative in policy.SUCCESSOR_SOURCE_PATHS:
        tree = ast.parse((ROOT / relative).read_text())
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
    assert not policy.CANONICAL_SOURCE_REVIEW_PATH.exists()
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()
    with pytest.raises(PermissionError, match="isolated launcher"):
        trainer.run_exact(object(), rgb_workers=5)  # type: ignore[arg-type]


def test_source_review_requires_different_agent_and_exact_source_hashes(
    tmp_path: Path,
) -> None:
    self_review = tmp_path / "self.json"
    self_sha = write_source_review(self_review, reviewer=policy.IMPLEMENTATION_AUTHOR)
    capability = policy.create_test_authority_capability(tmp_path)
    with pytest.raises(PermissionError, match="different agent"):
        capability.issue(
            self_review,
            self_sha,
            target_path=tmp_path / "attempts/seed_20260710/n5",
        )
    corrupt = tmp_path / "corrupt.json"
    corrupt_sha = write_source_review(
        corrupt,
        corrupt_source=policy.TRAINER_RELATIVE_PATH,
    )
    with pytest.raises(PermissionError, match="source changed"):
        capability.issue(
            corrupt,
            corrupt_sha,
            target_path=tmp_path / "other/seed_20260710/n5",
        )
