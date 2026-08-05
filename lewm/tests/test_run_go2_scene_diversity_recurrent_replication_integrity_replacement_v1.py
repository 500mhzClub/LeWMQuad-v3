from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_integrity_replacement_v1
    as runner,
)


def _replacement_plan() -> dict:
    path = runner.REPO_ROOT / (
        "docs/lewm_go2_scene_diversity_recurrent_replication_"
        "integrity_replacement_v1_exact_plan_2026-08-04.json"
    )
    return json.loads(path.read_text())


def test_wrapper_extends_the_complete_frozen_closure_and_changes_only_identity() -> None:
    assert runner.AUTHORITY_SCHEMA == (
        "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
        "execution_authority_v1"
    )
    assert runner.AUTHORITY_STATUS == (
        "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
        "INTEGRITY_REPLACEMENT_V1"
    )
    assert runner.DEFAULT_COLLECTION_ROOT.parent == runner.DEFAULT_ATTEMPT_ROOT
    assert set(runner.frozen_runner.SOURCE_PATHS) < set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["replacement_runner"] == Path(runner.__file__).resolve()
    assert runner.SOURCE_PATHS["replacement_runner_test"] == Path(__file__).resolve()
    assert runner.SOURCE_PATHS["predecessor_exact_plan"] == (
        runner.plan_builder.FROZEN_V1_EXACT_PLAN
    )
    assert runner.SOURCE_PATHS["predecessor_failure_terminal"] == (
        runner.PREDECESSOR_TERMINAL
    )
    assert runner.SOURCE_PATHS["predecessor_terminal_review"] == (
        runner.PREDECESSOR_TERMINAL_REVIEW
    )
    assert {
        "replacement_plan_builder",
        "replacement_collector",
        "replacement_runner",
        "replacement_authority_builder",
        "replacement_plan_test",
        "replacement_collector_test",
        "replacement_runner_test",
        "replacement_authority_test",
        "predecessor_exact_plan",
        "predecessor_failure_terminal",
        "predecessor_terminal_review",
    } <= set(runner.SOURCE_PATHS)
    assert runner.ContextOnlyLedgerV1 is runner.frozen_runner.ContextOnlyLedgerV1
    assert runner.RoleRuntimeDataV1 is runner.frozen_runner.RoleRuntimeDataV1
    assert runner.benchmark is runner.frozen_runner.benchmark
    assert runner.collector.EXPECTED_CAPS == (
        runner.frozen_runner.collector.EXPECTED_CAPS
    )
    assert runner.collector.EXPECTED_PERMISSIONS == (
        runner.frozen_runner.collector.EXPECTED_PERMISSIONS
    )


def test_predecessor_failure_evidence_is_exact_and_non_authorizing() -> None:
    evidence = runner.predecessor_failure_bindings_v1()

    assert evidence["predecessor_failure_terminal"] == {
        "path": str(runner.PREDECESSOR_TERMINAL.resolve()),
        "sha256": runner.PREDECESSOR_TERMINAL_SHA256,
        "byte_count": runner.PREDECESSOR_TERMINAL_BYTE_COUNT,
    }
    assert evidence["predecessor_terminal_review"] == {
        "path": str(runner.PREDECESSOR_TERMINAL_REVIEW.resolve()),
        "sha256": runner.PREDECESSOR_TERMINAL_REVIEW_SHA256,
        "byte_count": runner.PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT,
    }


def test_runtime_plan_validation_rejects_valid_scientific_drift() -> None:
    plan = _replacement_plan()
    authority = {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
    runner._validate_plan_v1(plan, authority)  # noqa: SLF001

    changed = copy.deepcopy(plan)
    changed["execution_contract"]["seed"] += 1
    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="not science-identical",
    ):
        runner._validate_plan_v1(changed, authority)  # noqa: SLF001


def test_scoped_configuration_patches_only_replacement_seams_and_restores() -> None:
    overrides = runner._configuration_overrides_v1()  # noqa: SLF001
    expected_keys = {
        "collector",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "SOURCE_REVIEW_SCHEMA",
        "SOURCE_REVIEW_STATUS",
        "RESULT_SCHEMA",
        "TERMINAL_SCHEMA",
        "RESERVATION_SCHEMA",
        "PREREGISTRATION",
        "SCENE_PANEL",
        "SCENE_PANEL_SHA256",
        "SCENE_PANEL_BYTE_COUNT",
        "SOURCE_REVIEW",
        "DEFAULT_ATTEMPT_ROOT",
        "DEFAULT_COLLECTION_ROOT",
        "SOURCE_PATHS",
        "_load_physics_index_v1",
    }
    assert set(overrides) == expected_keys
    original = {name: getattr(runner.frozen_runner, name) for name in overrides}

    with runner._configured_frozen_runner_v1():  # noqa: SLF001
        assert all(
            getattr(runner.frozen_runner, name) is value
            if name in {"collector", "SOURCE_PATHS"}
            else getattr(runner.frozen_runner, name) == value
            for name, value in overrides.items()
        )
        assert runner.frozen_runner._save_checkpoint_exclusive is (  # noqa: SLF001
            runner._save_checkpoint_exclusive  # noqa: SLF001
        )
        assert runner.frozen_runner.ContextOnlyLedgerV1 is runner.ContextOnlyLedgerV1

    assert all(getattr(runner.frozen_runner, name) is value for name, value in original.items())


def test_execute_delegates_to_frozen_custody_path_under_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "predecessor_failure_bindings_v1", lambda: {})

    def execute(
        authority: object, *, authority_binding: object, plan: object
    ) -> dict[str, object]:
        observed.update(
            {
                "authority": authority,
                "authority_binding": authority_binding,
                "plan": plan,
                "collector": runner.frozen_runner.collector,
                "result_schema": runner.frozen_runner.RESULT_SCHEMA,
                "attempt_root": runner.frozen_runner.DEFAULT_ATTEMPT_ROOT,
                "ledger": runner.frozen_runner.ContextOnlyLedgerV1,
            }
        )
        return {"status": "TEST"}

    monkeypatch.setattr(runner.frozen_runner, "execute_v1", execute)
    plan = _replacement_plan()
    authority = {"attempt_id": plan["attempt_id"]}
    binding = {"path": "/authority", "sha256": "a" * 64, "byte_count": 1}

    result = runner.execute_v1(authority, authority_binding=binding, plan=plan)

    assert result == {"status": "TEST"}
    assert observed == {
        "authority": authority,
        "authority_binding": binding,
        "plan": plan,
        "collector": runner.collector,
        "result_schema": runner.RESULT_SCHEMA,
        "attempt_root": runner.DEFAULT_ATTEMPT_ROOT,
        "ledger": runner.ContextOnlyLedgerV1,
    }


def test_split_process_evidence_is_validated_at_the_physics_loader_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    physics = {"status": "PHYSICS_COMPLETE"}

    def frozen_loader(*_args: object) -> dict[str, object]:
        events.append("frozen-combined-result")
        return dict(physics)

    def validator(
        value: object, **_kwargs: object
    ) -> dict[str, bool]:
        assert value == physics
        events.append("replacement-split-evidence")
        return {
            "validated": True,
            "workers_exact": True,
            "fixed_seed_exact": True,
            "release_barrier_exact": True,
            "join_exact": True,
        }

    monkeypatch.setattr(runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", frozen_loader)
    monkeypatch.setattr(
        runner.collector, "validate_split_collection_evidence_v1", validator
    )

    result = runner._load_replacement_physics_index_v1(  # noqa: SLF001
        {}, {}, {}
    )

    assert events == ["frozen-combined-result", "replacement-split-evidence"]
    assert result["_replacement_split_process_validation"]["validated"] is True


@pytest.mark.parametrize(
    "failed_field",
    (
        "validated",
        "workers_exact",
        "fixed_seed_exact",
        "release_barrier_exact",
        "join_exact",
    ),
)
def test_split_process_evidence_drift_fails_before_scientific_route(
    failed_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_FROZEN_LOAD_PHYSICS_INDEX_V1",
        lambda *_args: {"status": "PHYSICS_COMPLETE"},
    )
    evidence = {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barrier_exact": True,
        "join_exact": True,
    }
    evidence[failed_field] = False
    monkeypatch.setattr(
        runner.collector,
        "validate_split_collection_evidence_v1",
        lambda *_args, **_kwargs: dict(evidence),
    )

    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="did not pass exactly",
    ):
        runner._load_replacement_physics_index_v1({}, {}, {})  # noqa: SLF001


def test_split_validation_failure_prevents_downstream_execute_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(runner, "predecessor_failure_bindings_v1", lambda: {})

    def frozen_loader(*_args: object) -> dict[str, object]:
        events.append("frozen-loader")
        return {"status": "PHYSICS_COMPLETE"}

    def validator(*_args: object, **_kwargs: object) -> dict[str, bool]:
        events.append("split-validator")
        return {
            "validated": False,
            "workers_exact": True,
            "fixed_seed_exact": True,
            "release_barrier_exact": True,
            "join_exact": True,
        }

    def frozen_execute(
        authority: object, *, authority_binding: object, plan: object
    ) -> dict[str, object]:
        events.append("execute-entry")
        runner.frozen_runner._load_physics_index_v1(  # noqa: SLF001
            authority, authority_binding, plan
        )
        events.append("downstream-scientific-work")
        return {"status": "UNREACHABLE"}

    monkeypatch.setattr(runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", frozen_loader)
    monkeypatch.setattr(
        runner.collector, "validate_split_collection_evidence_v1", validator
    )
    monkeypatch.setattr(runner.frozen_runner, "execute_v1", frozen_execute)

    plan = _replacement_plan()
    authority = {"attempt_id": plan["attempt_id"]}
    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="did not pass exactly",
    ):
        runner.execute_v1(authority, authority_binding={}, plan=plan)

    assert events == ["execute-entry", "frozen-loader", "split-validator"]


def test_frozen_context_ledger_still_requires_checkpoint_before_eval() -> None:
    ledger = runner.ContextOnlyLedgerV1()
    with pytest.raises(runner.SceneDiversityRunnerError, match="custody stage"):
        ledger.load_receipts("eval")
    ledger.load_receipts("train")
    with pytest.raises(runner.SceneDiversityRunnerError, match="structurally forbidden"):
        ledger.open_rgb("train", "successor", "forbidden-successor")
    ledger.checkpoint()
    ledger.load_receipts("eval")
    assert ledger.rgb_opens["train_successor"] == 0
    assert ledger.rgb_opens["eval_successor"] == 0
