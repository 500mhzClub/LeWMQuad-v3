from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import (
    run_go2_scene_diversity_recurrent_replication_integrity_replacement_v2
    as runner,
)


@pytest.fixture(autouse=True)
def _stub_pre_science_plan_input_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.collector.pilot, "require_plan_bindings", lambda _plan: None
    )
    monkeypatch.setattr(
        runner.collector.bounded,
        "_validate_bound_scenes",
        lambda _plan: None,
    )
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_closure_v2",
        lambda *_args, **_kwargs: _passing_closure(),
        raising=False,
    )


def _replacement_plan() -> dict:
    return json.loads(runner.plan_builder.DEFAULT_PLAN_OUTPUT.read_text())


def _passing_evidence() -> dict[str, bool]:
    return {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barriers_exact": True,
        "join_exact": True,
    }


def _passing_closure() -> dict[str, bool]:
    return {
        "validated": True,
        "evidence_validated": True,
        "closure_rehashed": True,
        "scene_results_rehashed": True,
        "state_receipts_rehashed": True,
        "render_receipts_rehashed": True,
        "derived_meshes_rehashed": True,
        "plan_scene_input_bindings_rehashed": True,
    }


def _physics_stub() -> dict:
    return {
        "status": "PHYSICS_COMPLETE",
        "plan_binding": {"path": "/plan", "sha256": "a" * 64, "byte_count": 1},
        "scene_process_evidence": {
            "process_reset_equivalence_audit": (
                runner.collector.PROCESS_RESET_EQUIVALENCE_AUDIT_V2
            )
        },
    }


def _authority_stub() -> dict[str, str]:
    return {"collection_root": str(runner.REPO_ROOT)}


def test_v2_wrapper_extends_predecessor_closure_and_identity() -> None:
    assert runner.AUTHORITY_SCHEMA == (
        "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
        "execution_authority_v1"
    )
    assert runner.AUTHORITY_STATUS.endswith("INTEGRITY_REPLACEMENT_V2")
    assert runner.DEFAULT_COLLECTION_ROOT.parent == runner.DEFAULT_ATTEMPT_ROOT
    assert set(runner.predecessor_runner.SOURCE_PATHS) < set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["replacement_v2_runner"] == Path(
        runner.__file__
    ).resolve()
    assert runner.SOURCE_PATHS["replacement_v2_runner_test"] == Path(
        __file__
    ).resolve()
    assert runner.SOURCE_PATHS[
        "predecessor_replacement_v1_failure_terminal"
    ] == runner.PREDECESSOR_REPLACEMENT_TERMINAL
    assert runner.SOURCE_PATHS[
        "predecessor_replacement_v1_terminal_review"
    ] == runner.PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW
    assert set(runner.PROCESS_RESET_DEPENDENCY_PATHS) == {
        "replacement_v2_dependency_genesis_init",
        "replacement_v2_dependency_genesis_misc",
        "replacement_v2_dependency_genesis_scene",
        "replacement_v2_dependency_genesis_rigid_solver",
        "replacement_v2_dependency_genesis_rigid_entity",
        "replacement_v2_dependency_genesis_engine_mesh",
        "replacement_v2_dependency_genesis_mesh",
        "replacement_v2_dependency_genesis_options_misc",
        "replacement_v2_dependency_genesis_rasterizer_context",
        "replacement_v2_dependency_rsl_on_policy_runner",
        "replacement_v2_dependency_rsl_ppo",
        "replacement_v2_dependency_rsl_mlp_model",
    }
    assert all(
        runner.SOURCE_PATHS[name] == path and path.is_file()
        for name, path in runner.PROCESS_RESET_DEPENDENCY_PATHS.items()
    )
    assert runner.ContextOnlyLedgerV1 is runner.frozen_runner.ContextOnlyLedgerV1
    assert runner.collector.EXPECTED_CAPS == runner.frozen_runner.collector.EXPECTED_CAPS
    assert (
        runner.collector.EXPECTED_PERMISSIONS
        == runner.frozen_runner.collector.EXPECTED_PERMISSIONS
    )


def test_both_predecessor_failures_are_exact_and_non_authorizing() -> None:
    evidence = runner.predecessor_failure_bindings_v2()

    assert set(evidence) == {
        "predecessor_failure_terminal",
        "predecessor_terminal_review",
        "predecessor_replacement_v1_failure_terminal",
        "predecessor_replacement_v1_terminal_review",
    }
    assert evidence["predecessor_replacement_v1_failure_terminal"] == {
        "path": str(runner.PREDECESSOR_REPLACEMENT_TERMINAL.resolve()),
        "sha256": runner.PREDECESSOR_REPLACEMENT_TERMINAL_SHA256,
        "byte_count": runner.PREDECESSOR_REPLACEMENT_TERMINAL_BYTE_COUNT,
    }
    assert evidence["predecessor_replacement_v1_terminal_review"] == {
        "path": str(runner.PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW.resolve()),
        "sha256": runner.PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_SHA256,
        "byte_count": runner.PREDECESSOR_REPLACEMENT_TERMINAL_REVIEW_BYTE_COUNT,
    }


def test_runtime_plan_validation_rejects_valid_scientific_drift() -> None:
    plan = _replacement_plan()
    authority = {"attempt_id": runner.plan_builder.DEFAULT_ATTEMPT_ID}
    runner._validate_plan_v2(plan, authority)  # noqa: SLF001

    changed = copy.deepcopy(plan)
    changed["execution_contract"]["seed"] += 1
    with pytest.raises(runner.SceneDiversityRunnerError, match="science-identical"):
        runner._validate_plan_v2(changed, authority)  # noqa: SLF001


def test_scoped_wrapper_overlay_is_narrow_and_restored() -> None:
    overrides = runner._configuration_overrides_v2()  # noqa: SLF001
    original = {
        name: getattr(runner.predecessor_runner, name) for name in overrides
    }

    with runner._configured_predecessor_runner_v2():  # noqa: SLF001
        assert all(
            getattr(runner.predecessor_runner, name) is value
            for name, value in overrides.items()
        )
        nested = runner.predecessor_runner._configuration_overrides_v1()  # noqa: SLF001
        assert nested["collector"] is runner.collector
        assert nested["SOURCE_PATHS"] is runner.SOURCE_PATHS
        assert nested["_load_physics_index_v1"] is (
            runner._load_replacement_physics_index_v2  # noqa: SLF001
        )

    assert all(
        getattr(runner.predecessor_runner, name) is value
        for name, value in original.items()
    )


def test_execute_delegates_to_predecessor_custody_under_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(runner, "predecessor_failure_bindings_v2", lambda: {})

    def execute(
        authority: object, *, authority_binding: object, plan: object
    ) -> dict[str, object]:
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

    monkeypatch.setattr(runner.predecessor_runner, "execute_v1", execute)
    plan = _replacement_plan()
    authority = {"attempt_id": plan["attempt_id"]}
    binding = {"path": "/authority", "sha256": "a" * 64, "byte_count": 1}

    assert runner.execute_v2(authority, authority_binding=binding, plan=plan) == {
        "status": "TEST"
    }
    assert observed == {
        "authority": authority,
        "authority_binding": binding,
        "plan": plan,
        "collector": runner.collector,
        "plan_builder": runner.plan_builder,
        "result_schema": runner.RESULT_SCHEMA,
    }


def test_scene_process_evidence_and_reset_audit_gate_physics_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _physics_stub()
    events: list[str] = []

    def frozen_loader(*_args: object) -> dict:
        events.append("frozen-combined-result")
        return copy.deepcopy(physics)

    def validator(value: object, **_kwargs: object) -> dict[str, bool]:
        assert value == physics
        events.append("v2-process-evidence")
        return _passing_evidence()

    monkeypatch.setattr(runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", frozen_loader)
    monkeypatch.setattr(
        runner.collector, "validate_scene_process_evidence_v2", validator
    )
    monkeypatch.setattr(
        runner.collector.pilot,
        "require_plan_bindings",
        lambda _plan: events.append("plan-input-rehash"),
    )
    monkeypatch.setattr(
        runner.collector.bounded,
        "_validate_bound_scenes",
        lambda _plan: events.append("bound-scene-check"),
    )

    def closure_validator(*_args: object, **_kwargs: object) -> dict[str, bool]:
        events.append("generated-output-rehash")
        return _passing_closure()

    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_closure_v2",
        closure_validator,
    )

    result = runner._load_replacement_physics_index_v2(  # noqa: SLF001
        _authority_stub(), {}, {}
    )

    assert events == [
        "frozen-combined-result",
        "v2-process-evidence",
        "plan-input-rehash",
        "bound-scene-check",
        "generated-output-rehash",
    ]
    assert result["_replacement_v2_scene_process_validation"]["validated"] is True
    assert result["_replacement_v2_plan_input_closure"] == {
        "all_plan_input_bindings_rehashed": True,
        "bound_scene_inputs_absolute_and_non_generated": True,
    }
    assert result["_replacement_v2_generated_output_closure"] == {
        "validated": True,
        "evidence_validated": True,
        "closure_rehashed": True,
        "scene_results_rehashed": True,
        "state_receipts_rehashed": True,
        "render_receipts_rehashed": True,
        "derived_meshes_rehashed": True,
        "plan_scene_input_bindings_rehashed": True,
        "all_scene_result_receipt_mesh_bindings_rehashed": True,
    }


@pytest.mark.parametrize(
    "failed_field",
    (
        "validated",
        "workers_exact",
        "fixed_seed_exact",
        "release_barriers_exact",
        "join_exact",
    ),
)
def test_scene_process_evidence_drift_fails_before_science(
    failed_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "_FROZEN_LOAD_PHYSICS_INDEX_V1",
        lambda *_args: _physics_stub(),
    )
    evidence = _passing_evidence()
    evidence[failed_field] = False
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_evidence_v2",
        lambda *_args, **_kwargs: dict(evidence),
    )

    with pytest.raises(runner.SceneDiversityRunnerError, match="did not pass exactly"):
        runner._load_replacement_physics_index_v2(  # noqa: SLF001
            _authority_stub(), {}, {}
        )


def test_reset_equivalence_audit_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics = _physics_stub()
    physics["scene_process_evidence"]["process_reset_equivalence_audit"] = {}
    monkeypatch.setattr(
        runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", lambda *_args: physics
    )
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_evidence_v2",
        lambda *_args, **_kwargs: _passing_evidence(),
    )

    with pytest.raises(runner.SceneDiversityRunnerError, match="did not pass exactly"):
        runner._load_replacement_physics_index_v2(  # noqa: SLF001
            _authority_stub(), {}, {}
        )


def test_plan_input_rehash_failure_prevents_generated_closure_and_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", lambda *_args: _physics_stub()
    )
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_evidence_v2",
        lambda *_args, **_kwargs: _passing_evidence(),
    )

    def fail_rehash(_plan: object) -> None:
        events.append("plan-input-rehash")
        raise RuntimeError("changed input")

    monkeypatch.setattr(
        runner.collector.pilot, "require_plan_bindings", fail_rehash
    )
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_closure_v2",
        lambda *_args, **_kwargs: events.append("unreachable"),
    )

    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="plan input closure changed",
    ):
        runner._load_replacement_physics_index_v2(  # noqa: SLF001
            _authority_stub(), {}, {}
        )
    assert events == ["plan-input-rehash"]


@pytest.mark.parametrize("failed_field", tuple(_passing_closure()))
def test_generated_filesystem_closure_drift_fails_before_science(
    failed_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "_FROZEN_LOAD_PHYSICS_INDEX_V1", lambda *_args: _physics_stub()
    )
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_evidence_v2",
        lambda *_args, **_kwargs: _passing_evidence(),
    )
    closure = _passing_closure()
    closure[failed_field] = False
    monkeypatch.setattr(
        runner.collector,
        "validate_scene_process_closure_v2",
        lambda *_args, **_kwargs: closure,
    )

    with pytest.raises(
        runner.SceneDiversityRunnerError,
        match="filesystem closure did not pass exactly",
    ):
        runner._load_replacement_physics_index_v2(  # noqa: SLF001
            _authority_stub(), {}, {}
        )


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
