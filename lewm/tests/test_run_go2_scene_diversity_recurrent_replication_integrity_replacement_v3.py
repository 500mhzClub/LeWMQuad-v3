from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_integrity_replacement_v3
    as runner,
)


def _replacement_plan() -> dict:
    return json.loads(runner.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())


def test_v3_wrapper_extends_v2_closure_and_changes_only_identity() -> None:
    assert runner.AUTHORITY_SCHEMA == (
        "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_"
        "execution_authority_v1"
    )
    assert runner.AUTHORITY_STATUS.endswith("INTEGRITY_REPLACEMENT_V3")
    assert runner.DEFAULT_COLLECTION_ROOT.parent == runner.DEFAULT_ATTEMPT_ROOT
    assert set(runner.predecessor_runner.SOURCE_PATHS) < set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["replacement_v3_runner"] == Path(
        runner.__file__
    ).resolve()
    assert runner.SOURCE_PATHS["replacement_v3_runner_test"] == Path(
        __file__
    ).resolve()
    assert runner.SOURCE_PATHS[
        "predecessor_replacement_v2_failure_terminal"
    ] == runner.PREDECESSOR_V2_TERMINAL
    assert runner.SOURCE_PATHS[
        "predecessor_replacement_v2_terminal_review"
    ] == runner.PREDECESSOR_V2_TERMINAL_REVIEW
    assert runner.PROCESS_RESET_DEPENDENCY_PATHS is (
        runner.predecessor_runner.PROCESS_RESET_DEPENDENCY_PATHS
    )
    assert runner.ContextOnlyLedgerV1 is runner.frozen_runner.ContextOnlyLedgerV1
    assert runner.collector.EXPECTED_CAPS == runner.frozen_runner.collector.EXPECTED_CAPS
    assert (
        runner.collector.EXPECTED_PERMISSIONS
        == runner.frozen_runner.collector.EXPECTED_PERMISSIONS
    )


def test_all_predecessor_failures_are_exact_and_non_authorizing() -> None:
    evidence = runner.predecessor_failure_bindings_v3()

    assert set(evidence) == {
        "predecessor_failure_terminal",
        "predecessor_terminal_review",
        "predecessor_replacement_v1_failure_terminal",
        "predecessor_replacement_v1_terminal_review",
        "predecessor_replacement_v2_failure_terminal",
        "predecessor_replacement_v2_terminal_review",
    }
    assert evidence["predecessor_replacement_v2_failure_terminal"] == {
        "path": str(runner.PREDECESSOR_V2_TERMINAL.resolve()),
        "sha256": runner.PREDECESSOR_V2_TERMINAL_SHA256,
        "byte_count": runner.PREDECESSOR_V2_TERMINAL_BYTE_COUNT,
    }
    assert evidence["predecessor_replacement_v2_terminal_review"] == {
        "path": str(runner.PREDECESSOR_V2_TERMINAL_REVIEW.resolve()),
        "sha256": runner.PREDECESSOR_V2_TERMINAL_REVIEW_SHA256,
        "byte_count": runner.PREDECESSOR_V2_TERMINAL_REVIEW_BYTE_COUNT,
    }


def test_runtime_plan_validation_rejects_valid_scientific_drift() -> None:
    plan = _replacement_plan()
    authority = {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
    runner._validate_plan_v3(plan, authority)  # noqa: SLF001

    changed = copy.deepcopy(plan)
    changed["execution_contract"]["seed"] += 1
    with pytest.raises(runner.SceneDiversityRunnerError, match="science-identical"):
        runner._validate_plan_v3(changed, authority)  # noqa: SLF001


def test_scoped_runner_overlay_is_narrow_and_restored() -> None:
    overrides = runner._configuration_overrides_v3()  # noqa: SLF001
    original = {
        name: getattr(runner.predecessor_runner, name) for name in overrides
    }

    with runner._configured_predecessor_runner_v3():  # noqa: SLF001
        assert all(
            getattr(runner.predecessor_runner, name) is value
            for name, value in overrides.items()
        )
        nested = runner.predecessor_runner._configuration_overrides_v2()  # noqa: SLF001
        assert nested["collector"] is runner.collector
        assert nested["plan_builder"] is runner.plan_builder
        assert nested["SOURCE_PATHS"] is runner.SOURCE_PATHS
        assert nested["_load_replacement_physics_index_v1"] is (
            runner._load_replacement_physics_index_v3  # noqa: SLF001
        )

    assert all(
        getattr(runner.predecessor_runner, name) is value
        for name, value in original.items()
    )


def test_execute_delegates_to_v2_with_no_scientific_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "_validate_plan_v3", lambda *_args: None)
    monkeypatch.setattr(runner, "predecessor_failure_bindings_v3", lambda: {})

    def execute(authority, *, authority_binding, plan):
        observed.update(
            {
                "authority": authority,
                "authority_binding": authority_binding,
                "plan": plan,
                "collector": runner.predecessor_runner.collector,
                "plan_builder": runner.predecessor_runner.plan_builder,
                "result_schema": runner.predecessor_runner.RESULT_SCHEMA,
            }
        )
        return {"status": "TEST"}

    monkeypatch.setattr(runner.predecessor_runner, "execute_v2", execute)
    plan = _replacement_plan()
    authority = {"attempt_id": plan["attempt_id"]}
    binding = {"path": "/authority", "sha256": "a" * 64, "byte_count": 1}

    assert runner.execute_v3(
        authority, authority_binding=binding, plan=plan
    ) == {"status": "TEST"}
    assert observed == {
        "authority": authority,
        "authority_binding": binding,
        "plan": plan,
        "collector": runner.collector,
        "plan_builder": runner.plan_builder,
        "result_schema": runner.RESULT_SCHEMA,
    }


def test_physics_loader_is_exact_v2_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def load(authority, authority_binding, plan):
        observed.update(
            {
                "authority": authority,
                "authority_binding": authority_binding,
                "plan": plan,
                "collector": runner.predecessor_runner.collector,
            }
        )
        return {"status": "PHYSICS_COMPLETE"}

    monkeypatch.setattr(runner, "_V2_LOAD_PHYSICS_INDEX", load)
    assert runner._load_replacement_physics_index_v3(  # noqa: SLF001
        {"a": 1}, {"b": 2}, {"c": 3}
    ) == {"status": "PHYSICS_COMPLETE"}
    assert observed == {
        "authority": {"a": 1},
        "authority_binding": {"b": 2},
        "plan": {"c": 3},
        "collector": runner.collector,
    }


def test_preregistration_contains_final_vulkan_hard_stop() -> None:
    text = runner.PREREGISTRATION.read_text()
    assert "last authorized Vulkan/Genesis identity-only replacement" in text
    assert "there is no V4" in text
    assert "materially different backend" in text
    assert "a host reboot is not a V3" in text
    assert "prerequisite" in text
