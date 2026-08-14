from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from scripts import run_go2_small_completion_global_exact_v1 as RUNNER


C40 = "2" * 40
GENESIS_CONFIG = b"fake-genesis-config\n"
ROCM_CONFIG = b"fake-rocm-config\n"
FIXTURE_IDS = [
    "KNOWN_FEASIBLE",
    "KNOWN_INFEASIBLE",
    "MULTIPLE_FEASIBLE_OLD_CANONICAL_MASK_FAIL_LATER_JOINT_VALID",
    "FIT_CALIBRATION_CONSTRAINTS",
    "RESIDUAL_CANDIDATE_FREQUENCY_CONSTRAINTS",
]
BOUNDARY_ID = FIXTURE_IDS[2]
BOUNDARY_PREDICATES = {
    "at_least_two_hard_margin_feasible_rotation_vectors": True,
    "old_identity_ordered_canonical_vector_mask_passes": False,
    "later_hard_feasible_vector_mask_passes": True,
    "old_and_new_methods_agree_underlying_hard_margin_feasibility": True,
    "new_global_model_returns_mask_valid_solution": True,
    "new_solution_may_differ_from_old_canonical_vector": True,
    "every_scientific_constraint_still_validates": True,
}
FIXTURE_CONTRACT = {
    "status": "MANDATORY_BEFORE_SCIENTIFIC_SOLVE",
    "control_method": "TRACTABLE_EXHAUSTIVE_TRUSTED_CONTROL_ENUMERATION",
    "required_fixture_ids": FIXTURE_IDS,
    "requirements": {
        "new_and_exhaustive_control_agree_on_feasibility": True,
        "returned_solutions_satisfy_every_frozen_constraint": True,
        "infeasible_fixtures_proved_infeasible": True,
        "repeated_runs_same_solution_and_digest": True,
        "same_lexicographically_earliest_solution_required": False,
        "candidate_outcomes_consumed": False,
    },
    "mandatory_boundary_fixture": {
        "fixture_id": BOUNDARY_ID,
        **BOUNDARY_PREDICATES,
    },
}


class FakeAuthority:
    REPORT_SELF_KEY = "coupling_report_digest"
    AMENDMENT_SELF_KEY = "execution_amendment_digest"
    FIXTURE_VALIDATION_CONTRACT = FIXTURE_CONTRACT
    GENESIS_DOWNSTREAM_INTERPRETER_RELATIVE_PATH = (
        ".generated/venvs/fake_genesis/bin/python")
    GENESIS_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH = (
        ".generated/venvs/fake_genesis/pyvenv.cfg")
    ROCM_DOWNSTREAM_INTERPRETER_RELATIVE_PATH = (
        ".generated/venvs/fake_rocm/bin/python")
    ROCM_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH = (
        ".generated/venvs/fake_rocm/pyvenv.cfg")
    DOWNSTREAM_RUNTIME_CONTRACTS = {
        "genesis": {
            "role": "genesis_branch_generation",
            "interpreter_relative_path":
                GENESIS_DOWNSTREAM_INTERPRETER_RELATIVE_PATH,
            "pyvenv_config_relative_path":
                GENESIS_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH,
            "pyvenv_config_sha256": hashlib.sha256(
                GENESIS_CONFIG).hexdigest(),
            "pyvenv_config_byte_count": len(GENESIS_CONFIG),
            "python_version": "3.12.3",
            "genesis_version": "0.3.14",
            "torch_version": "2.12.0+cu130",
            "torch_cuda_runtime": "13.0",
            "torch_hip_runtime": None,
            "accelerator_available": False,
            "accelerator_device_count": 0,
            "accelerator_devices": [],
        },
        "rocm": {
            "role": "rocm_encoding_training_and_development",
            "interpreter_relative_path":
                ROCM_DOWNSTREAM_INTERPRETER_RELATIVE_PATH,
            "pyvenv_config_relative_path":
                ROCM_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH,
            "pyvenv_config_sha256": hashlib.sha256(ROCM_CONFIG).hexdigest(),
            "pyvenv_config_byte_count": len(ROCM_CONFIG),
            "python_version": "3.12.3",
            "torch_version": "2.12.0+rocm7.2",
            "torch_cuda_runtime": None,
            "torch_hip_runtime": "7.2.53211",
            "accelerator_available": True,
            "accelerator_device_count": 2,
            "accelerator_devices": [
                {
                    "index": 0,
                    "name": "AMD Radeon AI PRO R9700",
                    "capability": [12, 0],
                    "gcn_arch_name": "gfx1201",
                    "multi_processor_count": 32,
                },
                {
                    "index": 1,
                    "name": "AMD Ryzen 9 9950X3D 16-Core Processor",
                    "capability": [10, 3],
                    "gcn_arch_name": "gfx1036",
                    "multi_processor_count": 1,
                },
            ],
        },
    }
    DOWNSTREAM_STAGE_RUNTIME_ROLES = {
        "six_branch_smoke": "genesis",
        "smoke_encoding": "rocm",
        "full_720_branch_corpus": "genesis",
        "full_latent_encoding": "rocm",
        "scorer_training_and_qualification": "rocm",
        "development_transfer": "rocm",
        "qualification_validation": "rocm",
        "development_validation": "rocm",
    }
    NEW_RUNTIME_OUTPUT_PATHS = (
        ("global_exact_model_plan",
         Path(".generated/scorer_fit/"
              "small_completion_global_exact_model_plan_v1.json"),
         "file"),
        ("global_exact_terminal_result",
         Path(".generated/scorer_fit/"
              "small_completion_global_exact_terminal_result_v1.json"),
         "file"),
        ("global_exact_terminal_infeasibility",
         Path(".generated/scorer_fit/"
              "small_completion_global_exact_terminal_infeasibility_v1.json"),
         "file"),
    )


class FakeModel:
    FIXTURE_SUITE_SCHEMA = "fixture-suite"
    FIXTURE_SUITE_DIGEST_KEY = "fixture_suite_digest"
    FROZEN_FIXTURE_SUITE_RESULT_DIGEST = ""
    OBJECTIVE_CONTRACT_DIGEST = "e" * 64
    SOLVER_CONTRACT_DIGEST = "d" * 64
    EXECUTION_PLAN_DIGEST_KEY = "execution_plan_digest"
    EXECUTION_RESULT_DIGEST_KEY = "execution_result_digest"
    ALLOCATION_RESULT_DIGEST_KEY = "materialized_allocation_digest"
    EXECUTION_PASS_STATUS = "PASS_EXACT_GLOBAL_ALLOCATION_FOUND"
    EXECUTION_INFEASIBLE_STATUS = "EXACT_GLOBAL_ALLOCATION_INFEASIBLE"
    FROZEN_SOLVER_RUNTIME_IDENTITY = {
        "schema": "fake-frozen-solver-runtime",
        "runtime_digest": "f" * 64,
    }

    def __init__(self, events: list[str], *, feasible: bool = True) -> None:
        self.events = events
        self.feasible = feasible
        self.solve_count = 0
        self.live_plan_validations = 0
        self.live_result_validations = 0
        self.solve_free_plan_validations = 0
        self.solve_free_result_validations = 0
        self.runtime_record_validations = 0
        self.live_validation_allowed = True
        self.FROZEN_FIXTURE_SUITE_RESULT_DIGEST = self._fixture_payload()[
            self.FIXTURE_SUITE_DIGEST_KEY]

    canonical_digest = staticmethod(RUNNER.canonical_digest)

    def _fixture_payload(self) -> dict[str, Any]:
        rows = []
        for index, fixture_id in enumerate(FIXTURE_IDS):
            feasible = fixture_id != "KNOWN_INFEASIBLE"
            rows.append({
                "fixture_id": fixture_id,
                "semantic_spec_digest": f"{index + 1:x}" * 64,
                "model_digest": f"{index + 6:x}" * 64,
                "solver_feasible": feasible,
                "control_feasible": feasible,
                "control_valid_assignment_count": 2 if feasible else 0,
                "deterministic_optimal_objective_value": (
                    index if feasible else None),
                "repeated_runs_identical_bytes": True,
                "exact_result_digest": f"{index + 10:x}" * 64,
                "all_returned_constraints_directly_validated": True,
                "boundary_predicates": (
                    dict(BOUNDARY_PREDICATES)
                    if fixture_id == BOUNDARY_ID else None),
                "candidate_outcomes_consumed": False,
            })
        return RUNNER._signed({
            "schema": self.FIXTURE_SUITE_SCHEMA,
            "status": "PASS_MANDATORY_SYNTHETIC_FIXTURE_SUITE",
            "fixture_validation_contract": FIXTURE_CONTRACT,
            "objective_contract_digest": self.OBJECTIVE_CONTRACT_DIGEST,
            "solver_contract_digest": self.SOLVER_CONTRACT_DIGEST,
            "solver_runtime_identity": self.FROZEN_SOLVER_RUNTIME_IDENTITY,
            "fixtures": rows,
            "candidate_outcomes_consumed": False,
        }, self.FIXTURE_SUITE_DIGEST_KEY)

    def build_fixture_suite_result(self) -> dict[str, Any]:
        self.events.append("fixtures")
        return self._fixture_payload()

    def validate_fixture_suite_result(
            self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not self.live_validation_allowed:
            raise AssertionError("live fixture validator was called")
        expected = self._fixture_payload()
        if dict(value) != expected:
            raise ValueError("fixture suite changed")
        return expected

    def validate_solver_runtime_identity_record(
            self, value: Mapping[str, Any]) -> dict[str, Any]:
        self.runtime_record_validations += 1
        if dict(value) != self.FROZEN_SOLVER_RUNTIME_IDENTITY:
            raise ValueError("solver runtime record changed")
        return dict(self.FROZEN_SOLVER_RUNTIME_IDENTITY)

    def validate_production_instance(
            self, value: Mapping[str, Any]) -> dict[str, Any]:
        expected = {"schema": "instance", "masks": "frozen", "outcomes": False}
        if dict(value) != expected:
            raise ValueError("instance changed")
        return dict(expected)

    def build_execution_plan(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_production_instance(instance)
        return RUNNER._signed({
            "schema": "model-plan",
            "instance_digest": RUNNER.canonical_digest(instance),
            "solver_threads": 1,
        }, self.EXECUTION_PLAN_DIGEST_KEY)

    def _validate_plan(self, instance: Mapping[str, Any],
                       value: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.build_execution_plan(instance)
        if dict(value) != expected:
            raise ValueError("model plan changed")
        return expected

    def validate_execution_plan(
            self, instance: Mapping[str, Any], value: Mapping[str, Any]
            ) -> dict[str, Any]:
        if not self.live_validation_allowed:
            raise AssertionError("live plan validator was called")
        self.live_plan_validations += 1
        return self._validate_plan(instance, value)

    def validate_execution_plan_solve_free(
            self, instance: Mapping[str, Any], value: Mapping[str, Any]
            ) -> dict[str, Any]:
        self.solve_free_plan_validations += 1
        return self._validate_plan(instance, value)

    def _materialized(self) -> dict[str, Any]:
        scene_ids = [f"scene-{index}" for index in range(5)]
        rows = [{
            "selected_scene_id": scene_id,
            "selected_ordinal": index,
        } for index, scene_id in enumerate(scene_ids)]
        return RUNNER._signed({
            "schema": "materialized-allocation",
            "selected_scene_ids": scene_ids,
            "selected_scene_rows": rows,
        }, self.ALLOCATION_RESULT_DIGEST_KEY)

    def solve_once(self, instance: Mapping[str, Any],
                   plan: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_execution_plan(instance, plan)
        self.events.append("solve")
        self.solve_count += 1
        status = (self.EXECUTION_PASS_STATUS if self.feasible
                  else self.EXECUTION_INFEASIBLE_STATUS)
        return RUNNER._signed({
            "schema": "model-result", "status": status,
            "exact_model_result": {"solver_status": "optimal"},
            "materialized_allocation": (
                self._materialized() if self.feasible else None),
            "selected_scene_indices": list(range(5)) if self.feasible else [],
            "selected_scene_ids": (
                [f"scene-{index}" for index in range(5)]
                if self.feasible else []),
        }, self.EXECUTION_RESULT_DIGEST_KEY)

    def _validate_result(
            self, instance: Mapping[str, Any], plan: Mapping[str, Any],
            value: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_plan(instance, plan)
        result = dict(value)
        if result.get("status") not in {
                self.EXECUTION_PASS_STATUS,
                self.EXECUTION_INFEASIBLE_STATUS}:
            raise ValueError("result status changed")
        digest = result.get(self.EXECUTION_RESULT_DIGEST_KEY)
        if digest != RUNNER.canonical_digest({
                key: item for key, item in result.items()
                if key != self.EXECUTION_RESULT_DIGEST_KEY}):
            raise ValueError("result digest changed")
        materialized = result.get("materialized_allocation")
        if ((result["status"] == self.EXECUTION_PASS_STATUS)
                is not isinstance(materialized, Mapping)):
            raise ValueError("result allocation changed")
        return result

    def validate_execution_result(
            self, instance: Mapping[str, Any], plan: Mapping[str, Any],
            value: Mapping[str, Any]) -> dict[str, Any]:
        if not self.live_validation_allowed:
            raise AssertionError("live result validator was called")
        self.live_result_validations += 1
        return self._validate_result(instance, plan, value)

    def validate_execution_result_solve_free(
            self, instance: Mapping[str, Any], plan: Mapping[str, Any],
            value: Mapping[str, Any]) -> dict[str, Any]:
        self.solve_free_result_validations += 1
        return self._validate_result(instance, plan, value)


class FakeBuilder:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.context_modes: list[bool] = []
        self.finalize_count = 0
        self.finalizer_arguments: dict[str, Any] | None = None
        self.report = {
            "status": "REPORT_ISSUED", "classification": "COUPLED",
            "selected_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
            FakeAuthority.REPORT_SELF_KEY: "3" * 64,
        }
        self.amendment = {
            "status": "AMENDMENT_ISSUED",
            "source_repository_commit": C40,
            "selected_execution_method": {
                "method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"},
            "issuance_boundary": {"solver_invoked": False},
            "supersession": {
                "status": "SUPERSEDED_PRE_OUTCOME_UNNECESSARY_"
                          "LEXICOGRAPHIC_EXHAUSTION"},
            "v1_disposition": (
                "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"),
            "v2_backend_disposition": (
                "NOT_AUTHORISED_FOR_SCIENTIFIC_SEARCH_"
                "AFTER_V2_PERFORMANCE_FAILURE"),
            "immutable_predecessor_lineage_digest": "4" * 64,
            "continuation_authority": {
                "downstream_runtime_contracts": json.loads(json.dumps(
                    FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS)),
                "downstream_stage_runtime_roles": dict(
                    FakeAuthority.DOWNSTREAM_STAGE_RUNTIME_ROLES),
                "downstream_uses_global_solver_interpreter": {
                    "genesis": False, "rocm": False},
            },
            FakeAuthority.AMENDMENT_SELF_KEY: "5" * 64,
        }

    def issue_global_exact_coupling_report(self) -> dict[str, Any]:
        self.events.append("issue-report")
        return dict(self.report)

    def issue_global_exact_execution_amendment(self) -> dict[str, Any]:
        self.events.append("issue-amendment")
        return dict(self.amendment)

    def load_global_exact_execution_context(
            self, *, attach_scientific_masks: bool) -> dict[str, Any]:
        self.events.append(f"context:{attach_scientific_masks}")
        self.context_modes.append(attach_scientific_masks)
        context: dict[str, Any] = {
            "predecessor_scientific_input_bindings": {
                "schema": "predecessor"},
            "coupling_report": dict(self.report),
            "coupling_report_binding": {"sha256": "d" * 64},
            "execution_amendment": json.loads(json.dumps(self.amendment)),
            "scientific_contract_bindings": {"science": "6" * 64},
            "preoutcome_input_bindings": {"inputs": "7" * 64},
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": attach_scientific_masks,
        }
        if attach_scientific_masks:
            context["inputs"] = {"fixed": 115, "optional": 17}
            context["preserved_vectors"] = {
                f"state-{index}": [True] * 12 for index in range(7)}
        return context

    def build_global_exact_production_instance(
            self, context: Mapping[str, Any]) -> dict[str, Any]:
        assert context["scientific_masks_accessed"] is True
        self.events.append("instance")
        return {"schema": "instance", "masks": "frozen", "outcomes": False}

    def finalize_global_exact_feasible_allocation(self, **kwargs: Any
                                                  ) -> dict[str, Any]:
        self.events.append("finalize")
        self.finalize_count += 1
        self.finalizer_arguments = kwargs
        return {
            "state_manifest_digest": "8" * 64,
            "global_exact_successor_scorer_contract_digest": "9" * 64,
        }


def _runtime_root(tmp_path: Path) -> Path:
    (tmp_path / ".generated/scorer_fit").mkdir(parents=True)
    for role, config_bytes in (
            ("genesis", GENESIS_CONFIG), ("rocm", ROCM_CONFIG)):
        contract = FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS[role]
        interpreter = tmp_path / contract["interpreter_relative_path"]
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"#!/bin/sh\n")
        interpreter.chmod(0o755)
        config = tmp_path / contract["pyvenv_config_relative_path"]
        config.write_bytes(config_bytes)
    return tmp_path


class FakeRuntimeProbeInvoker:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, role: str, root: Path, downstream_python: Path,
                 authority: Any) -> Mapping[str, Any]:
        self.events.append(f"probe:{role}")
        self.calls.append((role, downstream_python))
        assert downstream_python == root / (
            authority.DOWNSTREAM_RUNTIME_CONTRACTS[role][
                "interpreter_relative_path"])
        return RUNNER._build_downstream_runtime_probe_receipt(
            runtime_role=role,
            observation=RUNNER._runtime_observation_from_contract(
                authority.DOWNSTREAM_RUNTIME_CONTRACTS[role]),
            authority=authority)


class FakeValidationInvoker:
    def __init__(self, events: list[str], *, qualified: bool = True,
                 amendment_digest: str = "5" * 64,
                 successor_digest: str = "9" * 64) -> None:
        self.events = events
        self.qualified = qualified
        self.amendment_digest = amendment_digest
        self.successor_digest = successor_digest
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, artifact_kind: str, root: Path,
                 downstream_python: Path, authority: Any
                 ) -> Mapping[str, Any]:
        self.events.append(f"validate:{artifact_kind}")
        self.calls.append((artifact_kind, downstream_python))
        assert downstream_python == root / Path(
            authority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"][
                "interpreter_relative_path"])
        common = {
            "qualification_report_digest": (
                "a" if self.qualified else "b") * 64,
            "global_exact_execution_amendment_digest": self.amendment_digest,
            "global_exact_successor_scorer_contract_digest":
                self.successor_digest,
        }
        if artifact_kind == "qualification":
            projection = {"qualified": self.qualified, **common}
        elif artifact_kind == "development":
            projection = {
                "complete": True,
                "development_transfer_result_digest": "c" * 64,
                **common,
            }
        else:
            raise AssertionError("unexpected validation kind")
        return RUNNER._build_downstream_validation_receipt(
            artifact_kind=artifact_kind, projection=projection,
            authority=authority)


def _resign_terminal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    result.pop(RUNNER.TERMINAL_SELF_KEY, None)
    result[RUNNER.TERMINAL_SELF_KEY] = RUNNER.canonical_digest(result)
    return result


class RecordingCommands:
    def __init__(self, events: list[str], *,
                 qualification_passes: bool = True) -> None:
        self.events = events
        self.commands: list[list[str]] = []
        self.qualification_passes = qualification_passes

    def __call__(self, command: Sequence[str], root: Path) -> int:
        del root
        row = [str(value) for value in command]
        stage = RUNNER._DOWNSTREAM_COMMAND_STAGE_NAMES[
            len(self.commands) % len(RUNNER._DOWNSTREAM_COMMAND_STAGE_NAMES)]
        self.events.append(f"command:{stage}")
        self.commands.append(row)
        if Path(row[1]).name == "train_go2_utility_scorer_v1_2.py":
            return 0 if self.qualification_passes else 1
        return 0


def _run(tmp_path: Path, *, feasible: bool = True,
         qualification_passes: bool = True):
    root = _runtime_root(tmp_path)
    events: list[str] = []
    builder = FakeBuilder(events)
    model = FakeModel(events, feasible=feasible)
    commands = RecordingCommands(
        events, qualification_passes=qualification_passes)
    probes = FakeRuntimeProbeInvoker(events)
    validation = FakeValidationInvoker(
        events, qualified=qualification_passes)
    result = RUNNER.solve_and_continue(
        root=root, builder=builder, authority=FakeAuthority, model=model,
        command_runner=commands, runtime_probe_invoker=probes,
        validation_invoker=validation)
    return (result, root, events, builder, model, commands, probes,
            validation)


def test_issue_stages_do_not_open_masks_or_solve(
        capsys: pytest.CaptureFixture[str]) -> None:
    events: list[str] = []
    builder = FakeBuilder(events)
    report = RUNNER.issue_report(builder=builder, authority=FakeAuthority)
    amendment = RUNNER.issue_amendment(
        builder=builder, authority=FakeAuthority)
    capsys.readouterr()
    assert report["classification"] == "COUPLED"
    assert amendment["issuance_boundary"]["solver_invoked"] is False
    assert events == ["issue-report", "issue-amendment"]


def test_feasible_dual_runtime_orchestration_and_terminal_reuse(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ((code, summary), root, events, builder, model, commands, probes,
     validation) = _run(tmp_path)
    capsys.readouterr()
    assert code == 0
    assert summary["status"] == "COMPLETE_AUTHORISED_DEVELOPMENT_TRANSFER"
    assert events[:4] == ["context:False", "fixtures", "context:True", "instance"]
    assert events.count("solve") == 1
    finalize_index = events.index("finalize")
    assert events[finalize_index + 1:finalize_index + 3] == [
        "probe:genesis", "probe:rocm"]
    assert events[finalize_index + 3].startswith("command:")
    assert model.solve_count == 1
    assert builder.finalize_count == 1
    assert [role for role, _path in probes.calls] == ["genesis", "rocm"]
    assert [kind for kind, _path in validation.calls] == [
        "qualification", "development"]
    assert len(commands.commands) == 6
    genesis = root / FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["genesis"][
        "interpreter_relative_path"]
    rocm = root / FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"][
        "interpreter_relative_path"]
    assert [Path(row[0]) for row in commands.commands] == [
        genesis, rocm, genesis, rocm, rocm, rocm]
    assert all("final_eval" not in part
               for command in commands.commands for part in command)

    plan_path = root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[0][1]
    terminal_path = root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[1][1]
    assert plan_path.stat().st_mode & 0o222 == 0
    assert terminal_path.stat().st_mode & 0o222 == 0
    plan = json.loads(plan_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    assert set(plan) == RUNNER._RUNNER_PLAN_KEYS
    assert set(terminal) == RUNNER._FEASIBLE_TERMINAL_KEYS
    assert terminal["selected_scene_ids"] == [f"scene-{i}" for i in range(5)]

    live_plan_calls = model.live_plan_validations
    live_result_calls = model.live_result_validations
    model.live_validation_allowed = False
    context = builder.load_global_exact_execution_context(
        attach_scientific_masks=True)
    RUNNER.validate_runner_plan(
        plan, execution_context=context,
        instance={"schema": "instance", "masks": "frozen", "outcomes": False},
        authority=FakeAuthority, model=model)
    RUNNER.validate_runner_terminal(
        terminal, execution_context=context,
        instance={"schema": "instance", "masks": "frozen", "outcomes": False},
        runner_plan=plan, authority=FakeAuthority, model=model)
    assert model.live_plan_validations == live_plan_calls
    assert model.live_result_validations == live_result_calls
    assert model.solve_free_plan_validations > 0
    assert model.solve_free_result_validations > 0
    model.live_validation_allowed = True

    RUNNER.solve_and_continue(
        root=root, builder=builder, authority=FakeAuthority, model=model,
        command_runner=commands, runtime_probe_invoker=probes,
        validation_invoker=validation)
    capsys.readouterr()
    assert model.solve_count == 1
    assert builder.finalize_count == 2


def test_exact_infeasibility_stops_without_downstream(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ((code, summary), root, _events, builder, model, commands, probes,
     validation) = _run(tmp_path, feasible=False)
    capsys.readouterr()
    assert code == 2
    assert summary["exact_infeasibility_proved"] is True
    assert builder.finalize_count == 0
    assert commands.commands == []
    assert probes.calls == []
    assert validation.calls == []
    assert model.solve_count == 1
    path = root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[2][1]
    terminal = json.loads(path.read_text())
    plan = json.loads(
        (root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[0][1]).read_text())
    assert set(terminal) == RUNNER._INFEASIBLE_TERMINAL_KEYS
    for field, changed in (
            ("exact_infeasibility_proved", False),
            ("scientific_conditions_relaxed", True),
            ("automatic_selector_revision", True),
            ("candidate_outcomes_consumed", True),
            ("scientific_masks_accessed", False),
            ("branch_labels_read", True),
            ("scorer_or_predictor_accessed", True)):
        mutated = dict(terminal)
        mutated[field] = changed
        with pytest.raises(RUNNER.GlobalExactRunnerError):
            RUNNER.validate_runner_terminal(
                _resign_terminal(mutated),
                execution_context=builder.load_global_exact_execution_context(
                    attach_scientific_masks=True),
                instance={
                    "schema": "instance", "masks": "frozen",
                    "outcomes": False},
                runner_plan=plan, authority=FakeAuthority, model=model)


def test_qualification_failure_stops_before_development(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ((code, summary), _root, _events, _builder, _model, commands, _probes,
     validation) = _run(tmp_path, qualification_passes=False)
    capsys.readouterr()
    assert code == 4
    assert summary["downstream"]["status"] == (
        "STOP_FROZEN_SCORER_QUALIFICATION_FAILURE")
    assert len(commands.commands) == 5
    assert [kind for kind, _path in validation.calls] == ["qualification"]


def test_parent_never_imports_torch_producers_for_artifact_validation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(_root: Path) -> dict[str, Any]:
        raise AssertionError("parent imported a Torch producer validator")

    monkeypatch.setattr(RUNNER, "_load_qualification", forbidden)
    monkeypatch.setattr(RUNNER, "_load_development_result", forbidden)
    ((code, _summary), root, _events, _builder, _model, _commands, _probes,
     validation) = _run(tmp_path)
    assert code == 0
    rocm = root / FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"][
        "interpreter_relative_path"]
    assert validation.calls == [
        ("qualification", rocm), ("development", rocm)]


def test_terminal_rejects_resigned_outcome_field(tmp_path: Path) -> None:
    ((_code, _summary), root, _events, builder, model, _commands, _probes,
     _validation) = _run(tmp_path)
    plan = json.loads(
        (root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[0][1]).read_text())
    terminal = json.loads(
        (root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[1][1]).read_text())
    terminal["candidate_branch_outcome"] = "forbidden"
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER.validate_runner_terminal(
            _resign_terminal(terminal),
            execution_context=builder.load_global_exact_execution_context(
                attach_scientific_masks=True),
            instance={"schema": "instance", "masks": "frozen", "outcomes": False},
            runner_plan=plan, authority=FakeAuthority, model=model)


def test_persisted_fixture_rejects_self_resigned_runtime_and_row(
        tmp_path: Path) -> None:
    del tmp_path
    model = FakeModel([])
    fixture = model.build_fixture_suite_result()
    for mutation in ("runtime", "row"):
        changed = json.loads(json.dumps(fixture))
        if mutation == "runtime":
            changed["solver_runtime_identity"]["runtime_digest"] = "0" * 64
        else:
            changed["fixtures"][0]["control_valid_assignment_count"] += 1
        changed[model.FIXTURE_SUITE_DIGEST_KEY] = RUNNER.canonical_digest(
            {key: value for key, value in changed.items()
             if key != model.FIXTURE_SUITE_DIGEST_KEY})
        with pytest.raises(RUNNER.GlobalExactRunnerError):
            RUNNER._validate_fixture_suite_solve_free(
                changed, authority=FakeAuthority, model=model)


def test_divergent_existing_plan_is_never_overwritten(tmp_path: Path) -> None:
    root = _runtime_root(tmp_path)
    plan_path = root / FakeAuthority.NEW_RUNTIME_OUTPUT_PATHS[0][1]
    plan_path.write_text("{}\n")
    plan_path.chmod(0o444)
    events: list[str] = []
    builder = FakeBuilder(events)
    model = FakeModel(events)
    with pytest.raises(RUNNER.GlobalExactRunnerError, match="differs"):
        RUNNER.solve_and_continue(
            root=root, builder=builder, authority=FakeAuthority, model=model,
            command_runner=RecordingCommands(events),
            runtime_probe_invoker=FakeRuntimeProbeInvoker(events),
            validation_invoker=FakeValidationInvoker(events))
    assert model.solve_count == 0


def test_fixed_command_sequence_uses_exact_dual_runtime_routing(
        tmp_path: Path) -> None:
    commands = RUNNER.downstream_command_sequence(tmp_path)
    flattened = " ".join(part for command in commands for part in command)
    assert "benchmark" not in flattened
    assert "final_eval" not in flattened
    assert "small-completion-search" not in flattened
    genesis = tmp_path / RUNNER.AUTHORITY.DOWNSTREAM_RUNTIME_CONTRACTS[
        "genesis"]["interpreter_relative_path"]
    rocm = tmp_path / RUNNER.AUTHORITY.DOWNSTREAM_RUNTIME_CONTRACTS[
        "rocm"]["interpreter_relative_path"]
    assert [Path(command[0]) for command in commands] == [
        genesis, rocm, genesis, rocm, rocm, rocm]
    assert all(command[0] != sys.executable for command in commands)


@pytest.mark.parametrize("relative", [
    Path("sealed/data.json"),
    Path("nested/sealed/value.json"),
    Path("nested/sealed_test.json"),
    Path("nested/sealed_future/value.json"),
])
def test_every_runner_path_guard_rejects_sealed_components_before_access(
        tmp_path: Path, relative: Path) -> None:
    root = tmp_path.resolve()
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._pinned_relative(root, relative, label="adversarial path")
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._require_managed_file(
            root, str(relative), label="adversarial managed file")


def test_runtime_output_and_runtime_contract_paths_reject_sealed_components(
        tmp_path: Path) -> None:
    root = _runtime_root(tmp_path)

    class SealedOutputAuthority(FakeAuthority):
        NEW_RUNTIME_OUTPUT_PATHS = (
            ("global_exact_model_plan", Path("x/sealed/model.json"), "file"),
        )

    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._pinned_relative(
            root,
            RUNNER._runtime_relative(
                SealedOutputAuthority, "global_exact_model_plan"),
            label="sealed runtime output")

    builder = FakeBuilder([])
    context = builder.load_global_exact_execution_context(
        attach_scientific_masks=True)
    context["execution_amendment"]["continuation_authority"][
        "downstream_runtime_contracts"]["genesis"][
            "interpreter_relative_path"] = "x/sealed_runtime/python"
    class SealedRuntimeAuthority(FakeAuthority):
        DOWNSTREAM_RUNTIME_CONTRACTS = json.loads(json.dumps(
            FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS))
    SealedRuntimeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["genesis"][
        "interpreter_relative_path"] = "x/sealed_runtime/python"
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._bound_downstream_interpreters(
            root=root, execution_context=context,
            authority=SealedRuntimeAuthority)


def test_registered_managed_root_alias_is_pinned_but_descendant_alias_rejected(
        tmp_path: Path) -> None:
    logical = tmp_path / RUNNER.AUTHORITY.GENERATED_ROOT_RELATIVE_PATH
    physical = tmp_path / "physical" / logical.name
    (physical / "scorer_fit").mkdir(parents=True)
    logical.parent.mkdir(parents=True)
    logical.symlink_to(physical, target_is_directory=True)
    artifact = physical / "scorer_fit" / "receipt.json"
    artifact.write_text("{}\n")
    pinned = RUNNER._pinned_relative(
        tmp_path,
        RUNNER.AUTHORITY.GENERATED_ROOT_RELATIVE_PATH /
        "scorer_fit/receipt.json",
        label="registered managed alias artifact")
    assert pinned == artifact

    target = physical / "real.json"
    target.write_text("{}\n")
    descendant_alias = physical / "scorer_fit" / "alias.json"
    descendant_alias.symlink_to(target)
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._pinned_relative(
            tmp_path,
            RUNNER.AUTHORITY.GENERATED_ROOT_RELATIVE_PATH /
            "scorer_fit/alias.json",
            label="descendant alias artifact")


def test_runtime_and_validation_receipts_are_closed_and_self_bound() -> None:
    runtime = RUNNER._build_downstream_runtime_probe_receipt(
        runtime_role="rocm",
        observation=RUNNER._runtime_observation_from_contract(
            FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"]),
        authority=FakeAuthority)
    runtime["candidate_outcome"] = "forbidden"
    runtime[RUNNER.DOWNSTREAM_RUNTIME_PROBE_SELF_KEY] = RUNNER.canonical_digest(
        {key: value for key, value in runtime.items()
         if key != RUNNER.DOWNSTREAM_RUNTIME_PROBE_SELF_KEY})
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._validate_downstream_runtime_probe_receipt(
            runtime, runtime_role="rocm", authority=FakeAuthority)

    validation = RUNNER._build_downstream_validation_receipt(
        artifact_kind="qualification", projection={
            "qualified": True,
            "qualification_report_digest": "a" * 64,
            "global_exact_execution_amendment_digest": "5" * 64,
            "global_exact_successor_scorer_contract_digest": "9" * 64,
        }, authority=FakeAuthority)
    validation["branch_label"] = "forbidden"
    validation[RUNNER.DOWNSTREAM_VALIDATION_SELF_KEY] = RUNNER.canonical_digest(
        {key: value for key, value in validation.items()
         if key != RUNNER.DOWNSTREAM_VALIDATION_SELF_KEY})
    with pytest.raises(RUNNER.GlobalExactRunnerError):
        RUNNER._validate_downstream_validation_receipt(
            validation, artifact_kind="qualification",
            authority=FakeAuthority)


@pytest.mark.parametrize("runtime_role", ["genesis", "rocm"])
def test_internal_runtime_probe_checks_existence_config_and_exact_identity(
        tmp_path: Path, runtime_role: str, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    root = _runtime_root(tmp_path)
    contract = FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS[runtime_role]

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return bool(contract["accelerator_available"])

        @staticmethod
        def device_count() -> int:
            return int(contract["accelerator_device_count"])

        @staticmethod
        def get_device_name(index: int) -> str:
            return str(contract["accelerator_devices"][index]["name"])

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            return tuple(contract["accelerator_devices"][index]["capability"])

        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            device = contract["accelerator_devices"][index]
            return SimpleNamespace(
                gcnArchName=device["gcn_arch_name"],
                multi_processor_count=device["multi_processor_count"])

    fake_torch = SimpleNamespace(
        __version__=contract["torch_version"],
        version=SimpleNamespace(
            cuda=contract["torch_cuda_runtime"],
            hip=contract["torch_hip_runtime"]),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        RUNNER.platform, "python_version",
        lambda: contract["python_version"])
    monkeypatch.setattr(
        metadata, "version",
        lambda distribution: (
            contract["genesis_version"]
            if distribution == "genesis-world" else "unexpected"))
    interpreter = root / contract["interpreter_relative_path"]
    monkeypatch.setattr(RUNNER.sys, "executable", str(interpreter))

    assert RUNNER._emit_downstream_runtime_probe(
        runtime_role, root=root, authority=FakeAuthority) == 0
    receipt = json.loads(capsys.readouterr().out)
    validated = RUNNER._validate_downstream_runtime_probe_receipt(
        receipt, runtime_role=runtime_role, authority=FakeAuthority)
    assert validated["observed_runtime_identity"] == (
        RUNNER._runtime_observation_from_contract(contract))


def test_default_artifact_validator_subprocess_is_bound_to_rocm_interpreter(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rocm = tmp_path / FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"][
        "interpreter_relative_path"]
    receipt = RUNNER._build_downstream_validation_receipt(
        artifact_kind="qualification", projection={
            "qualified": True,
            "qualification_report_digest": "a" * 64,
            "global_exact_execution_amendment_digest": "5" * 64,
            "global_exact_successor_scorer_contract_digest": "9" * 64,
        }, authority=FakeAuthority)
    observed: list[list[str]] = []

    def fake_run(command: Sequence[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        observed.append(list(command))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt), stderr="")

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    validated = RUNNER._default_downstream_validation_invoker(
        "qualification", tmp_path, rocm, FakeAuthority)
    assert validated == receipt
    assert observed == [[
        str(rocm), str(tmp_path / RUNNER.RUNNER_RELATIVE_PATH),
        "--stage", "internal-validate-qualification",
    ]]


def test_internal_artifact_validation_reopens_manifest_before_metric_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    root = _runtime_root(tmp_path)
    contract = FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS["rocm"]
    interpreter = root / contract["interpreter_relative_path"]
    amendment_digest = "5" * 64
    successor_digest = "9" * 64
    historical_digest = "7" * 64
    events: list[str] = []

    def boundary(_root: Path) -> dict[str, Any]:
        events.append("manifest")
        return {
            "global_exact_execution_amendment_digest": amendment_digest,
            "global_exact_successor_scorer_contract_digest": successor_digest,
            "scientific_predecessor_scorer_contract_v1_2_digest":
                historical_digest,
        }

    def qualification(
            _root: Path, *, expected_execution_amendment_digest: str,
            expected_successor_contract_digest: str,
            expected_scientific_predecessor_scorer_contract_digest: str,
            ) -> dict[str, Any]:
        assert events == ["manifest"]
        assert expected_execution_amendment_digest == amendment_digest
        assert expected_successor_contract_digest == successor_digest
        assert (expected_scientific_predecessor_scorer_contract_digest
                == historical_digest)
        events.append("qualification")
        return {
            "qualified": True,
            "qualification_report_digest": "a" * 64,
            "global_exact_execution_amendment_digest": amendment_digest,
            "global_exact_successor_scorer_contract_digest": successor_digest,
        }

    monkeypatch.setattr(
        RUNNER, "_observe_current_downstream_runtime",
        lambda role: RUNNER._runtime_observation_from_contract(
            FakeAuthority.DOWNSTREAM_RUNTIME_CONTRACTS[role]))
    monkeypatch.setattr(
        RUNNER, "_validate_downstream_manifest_boundary", boundary)
    monkeypatch.setattr(RUNNER, "_load_qualification", qualification)
    monkeypatch.setattr(RUNNER.sys, "executable", str(interpreter))
    assert RUNNER._emit_downstream_validation(
        "qualification", root=root, authority=FakeAuthority) == 0
    receipt = json.loads(capsys.readouterr().out)
    RUNNER._validate_downstream_validation_receipt(
        receipt, artifact_kind="qualification", authority=FakeAuthority)
    assert events == ["manifest", "qualification"]
