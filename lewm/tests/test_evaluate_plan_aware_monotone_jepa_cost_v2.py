from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lewm.safety import plan_aware_monotone_jepa_cost_v2_contract as contract
from scripts import evaluate_plan_aware_monotone_jepa_cost_v2 as evaluator


def _synthetic_attempt_root() -> Path:
    return contract.OUTPUT_ROOT.parent / f".{contract.OUTPUT_ROOT.name}.attempt-v2-1-1"


def _extended_stat_row(path: Path) -> dict[str, Any]:
    observed = os.stat(path, follow_symlinks=False)
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "nlink": observed.st_nlink,
        "file_type": (
            "DIRECTORY" if stat.S_ISDIR(observed.st_mode) else "REGULAR_FILE"
        ),
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _synthetic_process_identity(
    *, argv: list[str], pid: int = 70_001, ppid: int = 70_000
) -> dict[str, Any]:
    return {
        "pid": pid,
        "start_time_ticks": 101,
        "ppid": ppid,
        "argv": argv,
        "cwd": str(contract.REPO_ROOT),
        "executable": str(contract.PYTHON_EXECUTABLE),
    }


def _synthetic_namespace_receipt(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    output = tmp_path / "result.json"
    monkeypatch.setattr(contract, "OUTPUT_ROOT", output)
    attempt_id = contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID
    receipt = {
        "attempt_id": attempt_id,
        "attempt_root": str(
            output.parent / f".{output.name}.attempt-{attempt_id}"
        ),
        "content_digest": "0" * 64,
    }

    def validate(
        value: dict[str, Any], *, expected_attempt_id: str
    ) -> dict[str, Any]:
        assert value == receipt
        assert expected_attempt_id == attempt_id
        return dict(receipt)

    monkeypatch.setattr(
        contract, "validate_namespace_and_nonreuse_receipt", validate
    )
    return receipt


def _run_tinyquad_harness(
    source: str, *arguments: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(contract.PYTHON_EXECUTABLE),
            "-E",
            "-s",
            "-u",
            "-c",
            textwrap.dedent(source),
            *arguments,
        ],
        cwd=contract.REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    value = json.loads(completed.stdout.decode("utf-8"))
    assert type(value) is dict
    assert completed.stdout == contract.canonical_json_bytes(value) + b"\n"
    return value


def _function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(contract.EVALUATOR_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _ordered_call_names(function_name: str) -> list[str]:
    function = _function_node(function_name)
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        calls.append((node.lineno, node.col_offset, name))
    return [name for _line, _column, name in sorted(calls)]


def _function_source(name: str) -> str:
    return ast.unparse(_function_node(name))


def _contract_function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(contract.__file__).read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _contract_function_source(name: str) -> str:
    return ast.unparse(_contract_function_node(name))


def _static_main_mode_branches() -> set[str]:
    function = _function_node("main")
    return {
        comparator.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance((comparator := node.comparators[0]), ast.Attribute)
        and isinstance(comparator.value, ast.Name)
        and comparator.value.id == "V2"
    }


def _parser_argv_for_mode(mode: str) -> list[str]:
    root = str(_synthetic_attempt_root())
    digest = "0" * 64
    if mode == contract.PUBLIC_EXECUTE_MODE:
        return [mode]
    if mode == contract.PHASE1_SUPERVISOR_MODE:
        return [
            mode,
            "--attempt-id",
            contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID,
        ]
    if mode == contract.PRE_ROOT_PHASE1_FAILURE_CUSTODY_MODE:
        return [
            mode,
            "--custody-path",
            str(contract.PRE_ROOT_FAILURE_CUSTODY_PATH),
            "--supervisor-observation-content-digest",
            digest,
        ]
    if mode == contract.TERMINAL_PUBLICATION_SUPERVISOR_MODE:
        return [
            mode,
            "--attempt-root",
            root,
            "--resume-boundary",
            "AFTER_PHASE_1",
            "--publication-attempt-number",
            "1",
            "--publication-attempt-id",
            "v2-presentation-1-1",
            "--tracked-attempt-number",
            "1",
            "--tracked-attempt-id",
            "v2-tracked-1-1",
            "--phase-1-custody-content-digest",
            digest,
        ]
    if mode == contract.PHASE1_CONTEXT_STATE_WORKER_MODE:
        return [mode, "--attempt-root", root, "--state-index", "0"]
    if mode == contract.PHASE1_PREDICTION_SOURCE_WORKER_MODE:
        return [mode, "--attempt-root", root, "--source-id", "P1"]
    result = [mode, "--attempt-root", root]
    if mode == contract.PHASE1_TERMINAL_CUSTODY_MODE:
        result.extend(("--supervisor-observation-content-digest", digest))
    elif mode == contract.PHASE3_PUBLISHER_MODE:
        result.extend(
            (
                "--publication-attempt-number",
                "1",
                "--publication-attempt-id",
                "v2-presentation-1-1",
            )
        )
    return result


def test_parser_exposes_only_the_frozen_public_and_internal_modes() -> None:
    parser = evaluator.build_parser()
    parsed_modes = []
    for mode in contract.ALL_EVALUATOR_SUBCOMMAND_MODES:
        parsed = parser.parse_args(_parser_argv_for_mode(mode))
        assert parsed.mode == mode
        parsed_modes.append(parsed.mode)

    assert tuple(parsed_modes) == contract.ALL_EVALUATOR_SUBCOMMAND_MODES
    assert len(parsed_modes) == len(set(parsed_modes))

    with pytest.raises(SystemExit):
        parser.parse_args(["execute-scientific"])


def test_mode_roles_and_phase_routing_are_exact() -> None:
    expected_mode_to_role = {
        contract.PUBLIC_EXECUTE_MODE: contract.PROCESS_ROLES[0],
        contract.PHASE1_PROPRIO_HELPER_MODE: contract.PROCESS_ROLES[1],
        contract.PHASE1_STAGE_C_HELPER_MODE: contract.PROCESS_ROLES[2],
        contract.PHASE2_VALIDATOR_MODE: contract.PROCESS_ROLES[3],
        contract.PHASE3_PUBLISHER_MODE: contract.PROCESS_ROLES[4],
        contract.PHASE1_CONTEXT_STATE_WORKER_MODE: contract.PROCESS_ROLES[5],
        contract.PHASE1_PREDICTION_SOURCE_WORKER_MODE: contract.PROCESS_ROLES[6],
        contract.PHASE1_TERMINAL_CUSTODY_MODE: contract.PROCESS_ROLES[7],
        contract.PRE_ROOT_PHASE1_FAILURE_CUSTODY_MODE: (
            contract.PROCESS_ROLES[8]
        ),
        contract.PHASE1_SUPERVISOR_MODE: contract.PROCESS_ROLES[9],
        contract.TERMINAL_PUBLICATION_SUPERVISOR_MODE: (
            contract.PROCESS_ROLES[10]
        ),
    }
    assert set(contract.MODE_TO_ROLE) == set(
        contract.ALL_EVALUATOR_SUBCOMMAND_MODES
    )
    assert len(contract.MODE_TO_ROLE) == len(
        contract.ALL_EVALUATOR_SUBCOMMAND_MODES
    )
    assert set(contract.MODE_TO_ROLE.values()) == set(contract.PROCESS_ROLES)
    assert len(set(contract.MODE_TO_ROLE.values())) == len(contract.PROCESS_ROLES)
    assert contract.MODE_TO_ROLE == expected_mode_to_role
    assert evaluator._phase_for_mode(contract.PUBLIC_EXECUTE_MODE) == contract.PHASE_IDS[0]
    assert evaluator._phase_for_mode(contract.PHASE1_PROPRIO_HELPER_MODE) == contract.PHASE_IDS[0]
    assert evaluator._phase_for_mode(contract.PHASE1_STAGE_C_HELPER_MODE) == contract.PHASE_IDS[0]
    assert evaluator._phase_for_mode(contract.PHASE2_VALIDATOR_MODE) == contract.PHASE_IDS[1]
    assert evaluator._phase_for_mode(contract.PHASE3_PUBLISHER_MODE) == contract.PHASE_IDS[2]


def test_v2_source_has_no_call_to_forbidden_v1_driver_or_replay_symbol() -> None:
    tree = ast.parse(contract.EVALUATOR_PATH.read_text(encoding="utf-8"))
    forbidden = set(contract.FORBIDDEN_V1_REPLAY_OR_DRIVER_SYMBOLS)
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    assert called.isdisjoint(forbidden)


def test_stage_execution_preserves_all_three_prospective_branches() -> None:
    stage_a_only = contract.build_stage_execution(
        stage_a_gate_pass=False,
        proprio_contribution_supported=None,
    )
    assert stage_a_only["stage_b"]["executed"] is False
    assert stage_a_only["stage_c"]["executed"] is False
    assert stage_a_only["disposition"] == "STAGE_A_FAILED_B_AND_C_NOT_RUN"

    stage_a_b = contract.build_stage_execution(
        stage_a_gate_pass=True,
        proprio_contribution_supported=False,
    )
    assert stage_a_b["stage_b"]["rows"] == 2_304
    assert stage_a_b["stage_c"]["executed"] is False

    full = contract.build_stage_execution(
        stage_a_gate_pass=True,
        proprio_contribution_supported=True,
    )
    assert full["stage_a"]["rows"] == 576
    assert full["stage_a"]["raw_cost_rows"] == 1_728
    assert full["stage_b"]["rows"] == 2_304
    assert full["stage_c"]["rows"] == 1_728
    assert full["stage_c"]["direct_fidelity_rows"] == 6_912
    assert full["stage_c"]["action_sensitivity_rows"] == 576
    assert full["forced_stage_a_pass"] is False
    assert full["correction_replay_control_flow_used"] is False


def test_phase2_calls_exact_raw_target_action_and_counter_regenerators() -> None:
    source = ast.parse(contract.EVALUATOR_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_regenerate_scientific_payload"
    )
    calls = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {
        "_training_target_rows",
        "_predecessor_raw_goal_score_maps",
        "_validate_raw_cost_merged_scores",
        "_validate_persisted_action_plans",
    } <= calls
    direct_calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_regenerate_phase1_scientific_counters" in direct_calls


def test_phase1_scientific_counters_are_independently_reconstructed() -> None:
    fake_v1 = SimpleNamespace(FIT="fit", CALIBRATION="calibration", HELDOUT="heldout")
    datasets = {
        "fit": {"f0": [{}] * 12, "f1": [{}] * 12},
        "calibration": {"c0": [{}] * 12},
        "heldout": {"h0": [{}] * 12},
    }
    training = {
        "checkpoint_bindings": {"no": {}, "latent": {}},
        "training_history": {
            "no": [{"optimizer_steps": 2}, {"optimizer_steps": 2}],
            "latent": [{"optimizer_steps": 2}, {"optimizer_steps": 2}],
        },
    }
    counters = evaluator._regenerate_phase1_scientific_counters(
        v1=fake_v1,
        datasets=datasets,
        training_receipt=training,
        stage_a_rows=[{}] * 48,
        raw_rows=[{}] * 144,
        stage_b_rows=[{}] * 192,
        stage_c_rows=[{}] * 144,
        fidelity_rows=[{}] * 576,
        action_rows=[{}] * 48,
        candidate_selection_rows=[{}] * 4,
        proprio_helper_manifest={"logical_records": 96},
        proprio_helper_receipt={
            "counters": {"predictor_checkpoints_opened": 2}
        },
        stage_c_helper_manifest={"logical_records": 144},
        stage_c_helper_receipt={
            "counters": {"predictor_checkpoints_opened": 3}
        },
    )
    assert tuple(counters) == contract.SCIENTIFIC_COUNTER_FIELDS
    assert counters["fit_states_opened"] == 2
    assert counters["fit_rows_opened"] == 24
    assert counters["predicted_latents_opened"] == 240
    assert counters["predictor_checkpoints_opened"] == 5
    assert counters["optimizer_updates"] == 8
    assert counters["training_epochs"] == 4
    assert counters["scientific_score_rows"] == 1_152
    assert counters["failed_output_files_opened_for_reuse"] == 0
    assert counters["correction_replay_receipts_read"] == 0
    assert counters["future_proprioception_reads"] == 0
    assert counters["route_outcome_reads"] == 0
    assert counters["raw_goal_cosine_executions"] == 0


def test_terminal_supervisor_mode_is_dispatched_with_every_frozen_argument() -> None:
    function = _function_node("main")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "supervise_v2_terminal_publication"
    ]
    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "attempt_root",
        "resume_boundary",
        "publication_attempt_number",
        "publication_attempt_id",
        "tracked_attempt_number",
        "tracked_attempt_id",
        "phase_1_custody_content_digest",
        "phase_2_custody_content_digest",
        "phase_3_custody_content_digest",
        "presentation_retry_authority_content_digest",
        "tracked_retry_authority_content_digest",
    }
    assert "TERMINAL_PUBLICATION_SUPERVISOR_MODE" in _static_main_mode_branches()


def test_resumed_phase3_uses_anchored_initial_preflight_from_resume_receipt() -> None:
    source = _function_source("supervise_v2_terminal_publication")
    assert "observe_terminal_publication_supervisor_resume_preflight" in source
    assert "write_terminal_publication_supervisor_resume_preflight_exclusive_fsync" in source
    assert "else None" in source
    assert "resume_preflight['terminal_publication_supervisor_preflight']" not in source


def test_attempt2_full_retry_authority_is_loaded_before_and_passed_into_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _synthetic_attempt_root()
    identity = {
        "argv": ["terminal-supervisor"],
        "cwd": str(contract.REPO_ROOT),
    }
    phase_1 = {
        "preexecution_receipt": {
            "attempt_root_fd_custody": {"attempt_id": "v2-1-1"}
        }
    }
    authority = {
        "schema": contract.PRESENTATION_RETRY_AUTHORITY_SCHEMA,
        "content_digest": "1" * 64,
        "prior_failed_attempt_custodies": [
            {"publication_attempt_id": "v2-presentation-1-1"}
        ],
    }
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(evaluator, "_process_identity", lambda: identity)
    monkeypatch.setattr(
        contract,
        "expected_terminal_publication_supervisor_argv",
        lambda *_args, **_kwargs: identity["argv"],
    )
    monkeypatch.setattr(
        evaluator, "_load_terminal_phase_custody", lambda **_kwargs: phase_1
    )
    monkeypatch.setattr(
        evaluator,
        "_load_presentation_retry_authority_before_reopen",
        lambda **_kwargs: authority,
    )

    class ReopenObserved(RuntimeError):
        pass

    def reopen(**kwargs: Any) -> tuple[int, int, dict[str, Any]]:
        observed.append(kwargs)
        raise ReopenObserved("stop after retry-authority handoff")

    monkeypatch.setattr(
        contract,
        "reopen_attempt_root_for_terminal_publication_supervisor",
        reopen,
    )
    with pytest.raises(ReopenObserved, match="retry-authority handoff"):
        evaluator.supervise_v2_terminal_publication(
            attempt_root=root,
            resume_boundary="AFTER_PHASE_2_VALIDATION",
            publication_attempt_number=2,
            publication_attempt_id="v2-presentation-2-1",
            tracked_attempt_number=1,
            tracked_attempt_id="v2-tracked-1-1",
            phase_1_custody_content_digest="2" * 64,
            phase_2_custody_content_digest="3" * 64,
            phase_3_custody_content_digest=None,
            presentation_retry_authority_content_digest="1" * 64,
            tracked_retry_authority_content_digest=None,
        )
    assert len(observed) == 1
    assert observed[0]["presentation_retry_authority"] is authority
    assert observed[0]["presentation_retry_authority_content_digest"] == (
        authority["content_digest"]
    )
    calls = _ordered_call_names("supervise_v2_terminal_publication")
    assert calls.index(
        "_load_presentation_retry_authority_before_reopen"
    ) < calls.index(
        "reopen_attempt_root_for_terminal_publication_supervisor"
    )


def test_phase2_always_persists_terminal_failure_custody() -> None:
    source = _function_source("_run_terminal_publication_child")
    assert (
        "phase_id == V2.PHASE_IDS[1] or "
        "observation['observation_complete'] is True"
    ) in source
    calls = _ordered_call_names("_run_terminal_publication_child")
    assert calls.index("observe_terminal_publication_child_exit") < calls.index(
        "write_terminal_publication_child_exit_observation_exclusive_fsync"
    ) < calls.index("build_terminal_publication_phase_custody") < calls.index(
        "write_terminal_publication_phase_custody_exclusive_fsync"
    )


def test_pre_phase2_supervisor_failure_custody_is_durable_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    observed_keywords: dict[str, Any] = {}

    def observe(**kwargs: Any) -> dict[str, Any]:
        events.append("observe")
        observed_keywords.update(kwargs)
        return {"observation": True}

    def build(**kwargs: Any) -> dict[str, Any]:
        events.append("build")
        assert kwargs["failure_observation"] == {"observation": True}
        return {"custody": True}

    def write(**kwargs: Any) -> dict[str, Any]:
        events.append("write")
        assert kwargs["failure_custody"] == {"custody": True}
        return {"binding": True}

    def validate(value: dict[str, Any]) -> dict[str, Any]:
        events.append("validate")
        assert value == {"custody": True}
        return value

    monkeypatch.setattr(
        contract, "observe_terminal_publication_supervisor_failure", observe
    )
    monkeypatch.setattr(
        contract,
        "build_terminal_publication_supervisor_failure_custody",
        build,
    )
    monkeypatch.setattr(
        contract,
        "write_terminal_publication_supervisor_failure_custody_exclusive_fsync",
        write,
    )
    monkeypatch.setattr(
        contract,
        "validate_terminal_publication_supervisor_failure_custody",
        validate,
    )
    try:
        raise RuntimeError("injected terminal supervisor failure")
    except RuntimeError as error:
        result = evaluator._persist_pre_phase2_supervisor_failure(
            phase_1_custody={"phase": 1},
            supervisor_process_identity={"pid": 1},
            publication_attempt_id="v2-presentation-1-1",
            tracked_attempt_id="v2-tracked-1-1",
            failure_stage="ROOT_REOPEN",
            error=error,
            failure_started_monotonic_ns=1,
        )
    assert result == {"custody": True}
    assert events == ["observe", "build", "write", "validate"]
    exception = observed_keywords["structured_exception"]
    assert exception["type"] == "RuntimeError"
    assert exception["message"] == "injected terminal supervisor failure"
    assert "RuntimeError: injected terminal supervisor failure" in (
        exception["traceback"]
    )
    assert exception["traceback_sha256"] == hashlib.sha256(
        exception["traceback"].encode("utf-8")
    ).hexdigest()


def test_rooted_phase1_supervisor_failure_custody_is_durable_before_raise(
    tmp_path: Path,
) -> None:
    result = _run_tinyquad_harness(
        """
        import hashlib
        import json
        import os
        import stat
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        from lewm.safety import plan_aware_monotone_jepa_cost_v2_contract as contract
        from scripts import evaluate_plan_aware_monotone_jepa_cost_v2 as evaluator

        contract.OUTPUT_ROOT = Path(sys.argv[1]) / "result.json"
        supervisor = evaluator._process_identity()
        producer = {"pid": 53, "start_time_ticks": 59}
        process = SimpleNamespace(pid=53, poll=lambda: 0)
        evaluator._cleanup_remaining_process_group = lambda _pid: []
        evaluator._process_group_members = lambda _pid: []
        evaluator._active_v2_role_rows = lambda: [{"pid": supervisor["pid"]}]
        evaluator._identity_is_live = lambda _identity: False
        evaluator._producer_resource_rows = (
            lambda *_args, **_kwargs: ([], [], [], [])
        )
        contract.observe_phase1_post_cleanup_start_namespace = (
            lambda **_kwargs: {"attempt_root_absent_after_cleanup": False}
        )
        attempt_id = contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID
        attempt_root = contract.OUTPUT_ROOT.parent / (
            f".{contract.OUTPUT_ROOT.name}.attempt-{attempt_id}"
        )
        failure_observation = contract.attach_self_digest({
            "schema": contract.ROOTED_PHASE1_SUPERVISOR_FAILURE_OBSERVATION_SCHEMA,
            "attempt_id": attempt_id,
            "attempt_root": str(attempt_root),
            "supervisor_process_identity": supervisor,
            "root_and_artifact_inventory": {"artifact_inventory_rows": []},
            "producer_process_absence_observation": {
                "exact_historical_process_absent": True
            },
            "cleanup_actions": [],
            "process_group_members_after_cleanup": [],
            "all_known_children_and_resources_absent": True,
        })
        observed = {}

        def observe(**kwargs):
            observed.update(kwargs)
            return failure_observation

        contract.observe_rooted_phase1_supervisor_failure = observe
        contract.validate_rooted_phase1_supervisor_failure_observation = (
            lambda value: dict(value)
        )
        result = evaluator._persist_rooted_phase1_supervisor_failure(
            attempt_id=attempt_id,
            prelaunch_namespace_custody={"attempt_root": str(attempt_root)},
            prelaunch_handoff_memfd_custody={"handoff": True},
            supervisor_process_identity=supervisor,
            producer_process_identity=producer,
            attempt_root_fd_custody={"root": True},
            supervisor_root_reopen_custody={"reopen": True},
            producer_environment_names=["A"],
            producer_environment_sha256="1" * 64,
            supervisor_environment_names=["B"],
            supervisor_environment_sha256="2" * 64,
            process=process,
            captured_stdout=b"raw stdout",
            captured_stderr=b"raw stderr",
            failure_stage="PRODUCER_SOURCE_STREAM_LOAD",
            failure_errors=[RuntimeError("injected rooted failure")],
            supervisor_started_monotonic_ns=1,
            popen_started_monotonic_ns=2,
            cleanup_actions=[],
            communication_attempted=True,
            communicate_completed=True,
        )
        custody_path = Path(result["custody_path"])
        parent_fd = os.open(
            "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        leaf_fd = -1
        try:
            for component in custody_path.parent.parts[1:]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
                os.close(parent_fd)
                parent_fd = next_fd
            leaf_fd = os.open(
                custody_path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            opened = os.fstat(leaf_fd)
            chunks = []
            while True:
                chunk = os.read(leaf_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            if leaf_fd >= 0:
                os.close(leaf_fd)
            os.close(parent_fd)
        expected_payload = contract.canonical_json_bytes(result) + b"\\n"
        reloaded = json.loads(payload.decode("utf-8"))
        validated = contract.validate_rooted_phase1_supervisor_failure_custody(
            reloaded
        )
        disposition = contract.validate_terminal_scientific_attempt_disposition(
            result["scientific_attempt_disposition"]
        )
        envelope = {
            "custody_content_digest": result["content_digest"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "expected_payload_sha256": hashlib.sha256(expected_payload).hexdigest(),
            "payload_bytes": len(payload),
            "expected_payload_bytes": len(expected_payload),
            "persisted_mode": stat.S_IMODE(opened.st_mode),
            "persisted_nlink": opened.st_nlink,
            "reloaded_and_validated_equal": validated == result,
            "failure_stage": observed["failure_stage"],
            "captured_stdout_sha256": hashlib.sha256(
                observed["captured_stdout"]
            ).hexdigest(),
            "captured_stderr_sha256": hashlib.sha256(
                observed["captured_stderr"]
            ).hexdigest(),
            "failure_exception_type": observed["failure_exceptions"][0][
                "exception"
            ]["type"],
            "exit_observation_schema": disposition["exit_observation_schema"],
            "disposition": disposition["disposition"],
            "later_phase_continuation_authorized": disposition[
                "later_phase_continuation_authorized"
            ],
        }
        os.write(1, contract.canonical_json_bytes(envelope) + b"\\n")
        """,
        str(tmp_path),
    )
    assert result["payload_sha256"] == result["expected_payload_sha256"]
    assert result["payload_bytes"] == result["expected_payload_bytes"]
    assert result["persisted_mode"] == contract.RUNTIME_FILE_MODE
    assert result["persisted_nlink"] == 1
    assert result["reloaded_and_validated_equal"] is True
    assert result["failure_stage"] == "PRODUCER_SOURCE_STREAM_LOAD"
    assert result["captured_stdout_sha256"] == hashlib.sha256(
        b"raw stdout"
    ).hexdigest()
    assert result["captured_stderr_sha256"] == hashlib.sha256(
        b"raw stderr"
    ).hexdigest()
    assert result["failure_exception_type"] == "RuntimeError"
    assert result["exit_observation_schema"] == (
        contract.ROOTED_PHASE1_SUPERVISOR_FAILURE_OBSERVATION_SCHEMA
    )
    assert result["disposition"] == "UNKNOWN_NO_RETRY"
    assert result["later_phase_continuation_authorized"] is False


def test_phase1_execute_observes_complete_foreground_runtime_custody_before_science(
    tmp_path: Path,
) -> None:
    observed = _run_tinyquad_harness(
        """
        import os
        import sys
        from pathlib import Path

        from lewm.safety import plan_aware_monotone_jepa_cost_v2_contract as contract
        from scripts import evaluate_plan_aware_monotone_jepa_cost_v2 as evaluator

        class StartupBoundaryReached(RuntimeError):
            pass

        genuine_identity = evaluator._process_identity()
        original_live_gate = contract._require_exact_live_process_identity
        assert original_live_gate(genuine_identity) == genuine_identity
        identity = {
            **genuine_identity,
            "argv": list(contract.PUBLIC_EXECUTE_ARGV),
        }

        def exact_public_argv_adapter(value):
            assert original_live_gate(genuine_identity) == genuine_identity
            assert value == identity
            assert value["argv"] == list(contract.PUBLIC_EXECUTE_ARGV)
            assert {
                key: value[key] for key in value if key != "argv"
            } == {
                key: genuine_identity[key]
                for key in genuine_identity
                if key != "argv"
            }
            return dict(value)

        contract._require_exact_live_process_identity = (
            exact_public_argv_adapter
        )
        evaluator._process_identity = lambda: identity
        contract.OUTPUT_ROOT = Path(sys.argv[1]) / "result.json"
        attempt_id = contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID
        attempt_root = contract.OUTPUT_ROOT.parent / (
            f".result.json.attempt-{attempt_id}"
        )
        contract.consume_supervised_phase1_prelaunch_handoff = (
            lambda **_kwargs: {"attempt_id": attempt_id}
        )
        evaluator._require_clean_source_freeze = lambda: "0" * 40
        contract.build_scientific_invariance_receipt = lambda: {}
        contract.validate_scientific_invariance_receipt = (
            lambda value: value
        )
        evaluator._new_attempt_root = lambda *_args: (
            attempt_id, attempt_root, {}, {}
        )
        evaluator._require_attempt_io = lambda _root: (-1, {}, None)
        evaluator._write_last_stage = lambda *_args: None
        contract.write_phase1_supervisor_prelaunch_handoff_receipt_exclusive_fsync = (
            lambda **_kwargs: {}
        )
        evaluator._validate_interpreter_and_resource_paths = lambda: True
        evaluator._environment_custody = lambda: (["A"], "1" * 64)
        actual_observer = (
            contract.observe_phase1_foreground_runtime_import_custody
        )
        observed = []

        def observe(**kwargs):
            receipt = actual_observer(**kwargs)
            observed.append(receipt)
            return receipt

        contract.observe_phase1_foreground_runtime_import_custody = observe
        evaluator._write_json_exclusive = lambda *_args: None
        evaluator._begin_static_audit = lambda **_kwargs: (
            (_ for _ in ()).throw(
                StartupBoundaryReached("startup custody completed")
            )
        )
        evaluator._v1 = lambda: (_ for _ in ()).throw(
            AssertionError("scientific entrypoint was reached")
        )
        try:
            evaluator.execute_phase_1()
        except StartupBoundaryReached:
            pass
        else:
            raise AssertionError("Phase-1 startup crossed the sentinel")
        assert len(observed) == 1
        receipt = contract.validate_phase1_foreground_runtime_import_custody(
            observed[0]
        )
        os.write(1, contract.canonical_json_bytes(receipt) + b"\\n")
        """,
        str(tmp_path),
    )
    custody = contract.validate_phase1_foreground_runtime_import_custody(
        observed
    )
    for authority in (
        contract.FOREGROUND_RUNTIME_IMPORT_AUTHORITY["modules"]
    ):
        assert {"mode", "uid", "gid", "nlink"} <= set(authority)
    assert custody["observed_executable_binding"]["bytes"] > 0
    assert custody["observed_module_rows"]
    assert custody["pass"] is True
    source = _function_source("execute_phase_1")
    assert source.index(
        "observe_phase1_foreground_runtime_import_custody"
    ) < source.index("_begin_static_audit") < source.index("_v1()")


def test_corrected_phase1_prelaunch_binds_failed_startup_zero_reuse_and_fresh_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_id = contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID
    with pytest.raises(contract.V2ContractError, match="corrected"):
        contract.observe_phase1_supervisor_prelaunch_namespace(
            contract.FAILED_TECHNICAL_STARTUP_ATTEMPT_ID
        )
    amendment = contract.build_technical_correction_amendment_authority()
    assert contract.validate_technical_correction_amendment_authority(
        amendment
    ) == amendment
    failed_nonreuse = (
        contract.observe_failed_technical_startup_nonreuse_custody()
    )
    assert contract.validate_failed_technical_startup_nonreuse_custody(
        failed_nonreuse, reverify_live=True
    ) == failed_nonreuse
    assert failed_nonreuse["failed_attempt_id"] == (
        contract.FAILED_TECHNICAL_STARTUP_ATTEMPT_ID
    )
    assert failed_nonreuse["corrected_attempt_id"] == attempt_id
    assert failed_nonreuse["scientific_inputs_opened"] == 0
    assert failed_nonreuse["model_initializations"] == 0
    assert failed_nonreuse["failed_root_bytes_adopted_or_reused"] is False
    assert failed_nonreuse["preexecution_receipt_absent"] is True
    assert failed_nonreuse["scientific_attempt_boundary_entry_absent"] is True
    assert failed_nonreuse["first_scientific_open_receipt_absent"] is True
    prelaunch = contract.observe_phase1_supervisor_prelaunch_namespace(
        attempt_id
    )
    assert contract.validate_phase1_supervisor_prelaunch_namespace(
        prelaunch
    ) == prelaunch
    assert prelaunch["matching_attempt_roots_before"] == [
        str(contract.FAILED_TECHNICAL_STARTUP_ROOT)
    ]
    assert prelaunch["failed_technical_startup_nonreuse_custody"] == (
        failed_nonreuse
    )
    assert prelaunch["technical_correction_amendment_authority"] == amendment
    supervisor = _synthetic_process_identity(
        argv=contract.expected_phase1_supervisor_argv(attempt_id),
        pid=os.getpid(),
    )
    monkeypatch.setattr(
        contract,
        "_require_exact_live_process_identity",
        lambda value: dict(value),
    )
    descriptor = -1
    try:
        descriptor, handoff = (
            contract.create_phase1_supervisor_prelaunch_handoff_memfd(
                attempt_id=attempt_id,
                prelaunch_namespace_custody=prelaunch,
                supervisor_process_identity=supervisor,
            )
        )
        payload = handoff["payload"]
        assert payload["attempt_id"] == attempt_id
        assert payload["prelaunch_namespace_custody"] == prelaunch
        assert payload["technical_correction_amendment_authority"] == (
            amendment
        )
        assert payload["failed_technical_startup_nonreuse_content_digest"] == (
            failed_nonreuse["content_digest"]
        )
        assert contract.supervised_phase1_prelaunch_pass_fds(handoff) == (
            descriptor,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@pytest.mark.parametrize(
    ("injected_branch", "expected_stage"),
    (
        ("root_reopen", "ATTEMPT_ROOT_REOPEN"),
        ("missing_stream", "PRODUCER_SOURCE_STREAM_LOAD"),
        ("bundle_write", "SUPERVISOR_BUNDLE_PUBLICATION"),
        ("attestor", "TERMINAL_CUSTODY_PERSISTENCE"),
    ),
)
def test_phase1_post_root_abrupt_failures_never_fall_back_to_pre_root_custody(
    monkeypatch: pytest.MonkeyPatch,
    injected_branch: str,
    expected_stage: str,
) -> None:
    attempt_id = contract.CORRECTED_TECHNICAL_STARTUP_ATTEMPT_ID
    root = _synthetic_attempt_root()
    supervisor = {
        "pid": 41,
        "start_time_ticks": 43,
        "argv": ["phase-1-supervisor"],
        "cwd": str(contract.REPO_ROOT),
    }
    producer = {
        "pid": 53,
        "start_time_ticks": 59,
        "ppid": 41,
        "argv": list(contract.PUBLIC_EXECUTE_ARGV),
    }
    historical = {"producer_process_identity": producer}
    reopened = {"root": "reopened"}
    persisted: list[dict[str, Any]] = []

    class FakeProcess:
        pid = 53
        returncode = 0

        def communicate(self) -> tuple[bytes, bytes]:
            return b"producer stdout", b""

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float = 0) -> int:
            return 0

    monkeypatch.setattr(evaluator, "_process_identity", lambda: supervisor)
    monkeypatch.setattr(
        contract,
        "expected_phase1_supervisor_argv",
        lambda _attempt_id: supervisor["argv"],
    )
    amendment = {"authority": True}
    failed_nonreuse = {
        "amendment_authority": amendment,
        "content_digest": "1" * 64,
    }
    prelaunch = {
        "attempt_root": str(root),
        "technical_correction_amendment_authority": amendment,
        "failed_technical_startup_nonreuse_custody": failed_nonreuse,
        "failed_technical_startup_nonreuse_content_digest": "1" * 64,
    }
    monkeypatch.setattr(
        contract,
        "observe_phase1_supervisor_prelaunch_namespace",
        lambda _attempt_id: prelaunch,
    )
    monkeypatch.setattr(
        contract,
        "validate_technical_correction_amendment_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        contract,
        "validate_failed_technical_startup_nonreuse_custody",
        lambda value, *, reverify_live: value,
    )
    handoff_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    monkeypatch.setattr(
        contract,
        "create_phase1_supervisor_prelaunch_handoff_memfd",
        lambda **_kwargs: (handoff_fd, {"handoff": True}),
    )
    monkeypatch.setattr(
        contract,
        "build_supervised_phase1_environment",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        contract, "supervised_phase1_prelaunch_pass_fds", lambda _value: ()
    )
    monkeypatch.setattr(
        evaluator.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )
    monkeypatch.setattr(
        evaluator,
        "_process_identity_for_pid",
        lambda *_args, **_kwargs: producer,
    )
    monkeypatch.setattr(
        evaluator, "_cleanup_remaining_process_group", lambda _pid: []
    )
    monkeypatch.setattr(evaluator, "_process_group_members", lambda _pid: [])
    monkeypatch.setattr(
        evaluator,
        "_active_v2_role_rows",
        lambda: [{"pid": supervisor["pid"]}],
    )
    monkeypatch.setattr(evaluator, "_identity_is_live", lambda _value: False)
    monkeypatch.setattr(
        evaluator,
        "_producer_resource_rows",
        lambda *_args, **_kwargs: ([], [], [], []),
    )
    monkeypatch.setattr(
        contract,
        "observe_phase1_post_cleanup_start_namespace",
        lambda **_kwargs: {"attempt_root_absent_after_cleanup": False},
    )
    monkeypatch.setattr(
        contract,
        "require_rooted_phase1_supervisor_failure_custody_absent",
        lambda _attempt_id: None,
    )
    monkeypatch.setattr(
        evaluator,
        "_load_attempt_root_custody_for_reopen",
        lambda _root: historical,
    )

    if injected_branch == "root_reopen":
        monkeypatch.setattr(
            contract,
            "reopen_phase1_attempt_root_for_supervisor",
            lambda **_kwargs: (_ for _ in ()).throw(
                OSError("injected root reopen failure")
            ),
        )
    else:
        monkeypatch.setattr(
            contract,
            "reopen_phase1_attempt_root_for_supervisor",
            lambda **_kwargs: (
                os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC),
                os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC),
                reopened,
            ),
        )
        monkeypatch.setattr(
            contract,
            "read_v2_attempt_bytes_no_follow",
            (
                (lambda **_kwargs: (_ for _ in ()).throw(
                    FileNotFoundError("injected producer stream absence")
                ))
                if injected_branch == "missing_stream"
                else (lambda **_kwargs: b"source stream")
            ),
        )
        monkeypatch.setattr(
            contract,
            "phase1_terminal_observation_runtime_paths",
            lambda: {
                key: f"terminal/{key}"
                for key in (
                    "stdout", "stderr", "traceback", "exception", "last_stage"
                )
            },
        )
        monkeypatch.setattr(
            contract,
            "build_phase1_supervisor_observation",
            lambda **_kwargs: {"content_digest": "3" * 64},
        )
        monkeypatch.setattr(
            contract,
            "write_phase1_supervisor_observation_bundle_exclusive_fsync",
            (
                (lambda **_kwargs: (_ for _ in ()).throw(
                    OSError("injected bundle write failure")
                ))
                if injected_branch == "bundle_write"
                else (lambda **_kwargs: {"binding": True})
            ),
        )
        if injected_branch == "attestor":
            attestor_error = RuntimeError("injected attestor failure")
            monkeypatch.setattr(
                contract,
                "expected_internal_argv",
                lambda *_args, **_kwargs: ["terminal-attestor"],
            )
            monkeypatch.setattr(
                evaluator,
                "_run_terminal_attestor",
                lambda _argv: (_ for _ in ()).throw(
                    evaluator.V2TerminalAttestorProcessError(
                        "injected terminal attestor failure",
                        failure_stage="TERMINAL_CUSTODY_PERSISTENCE",
                        lifecycle={"popen_succeeded": False},
                        primary_error=attestor_error,
                    )
                ),
            )

    def persist_rooted(**kwargs: Any) -> dict[str, Any]:
        persisted.append(kwargs)
        return {"content_digest": "4" * 64}

    monkeypatch.setattr(
        evaluator, "_persist_rooted_phase1_supervisor_failure", persist_rooted
    )
    monkeypatch.setattr(
        contract,
        "build_pre_root_phase1_supervisor_observation",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rooted failure fell into pre-root custody")
        ),
    )
    with pytest.raises(evaluator.V2EvaluationError, match="UNKNOWN_NO_RETRY"):
        evaluator.supervise_phase_1_execution(attempt_id)
    assert len(persisted) == 1
    assert persisted[0]["failure_stage"] == expected_stage
    assert persisted[0]["captured_stdout"] == b"producer stdout"
    if injected_branch == "attestor":
        assert persisted[0]["terminal_attestor_lifecycle"] == {
            "popen_succeeded": False
        }


@pytest.mark.parametrize("failure_stage", ("ROOT_REOPEN", "INITIAL_PREFLIGHT"))
def test_terminal_supervisor_root_and_preflight_failures_freeze_sibling_custody(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    root = _synthetic_attempt_root()
    identity = {"argv": ["terminal-supervisor"], "cwd": str(contract.REPO_ROOT)}
    phase_1 = {
        "preexecution_receipt": {
            "attempt_root_fd_custody": {"attempt_id": "v2-1-1"}
        }
    }
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(evaluator, "_process_identity", lambda: identity)
    monkeypatch.setattr(
        contract,
        "expected_terminal_publication_supervisor_argv",
        lambda *_args, **_kwargs: identity["argv"],
    )
    monkeypatch.setattr(
        evaluator, "_load_terminal_phase_custody", lambda **_kwargs: phase_1
    )
    monkeypatch.setattr(
        contract,
        "require_terminal_publication_supervisor_failure_custody_absent",
        lambda _attempt_id: None,
    )
    reopened = {"root": "reopened"}

    if failure_stage == "ROOT_REOPEN":
        monkeypatch.setattr(
            contract,
            "reopen_attempt_root_for_terminal_publication_supervisor",
            lambda **_kwargs: (_ for _ in ()).throw(
                OSError("injected root reopen failure")
            ),
        )
    else:
        monkeypatch.setattr(
            contract,
            "reopen_attempt_root_for_terminal_publication_supervisor",
            lambda **_kwargs: (100, 101, reopened),
        )
        monkeypatch.setattr(evaluator, "_activate_attempt_io", lambda **_kwargs: None)
        monkeypatch.setattr(evaluator, "_load_canonical_json", lambda _path: {})
        monkeypatch.setattr(
            contract,
            "validate_phase1_terminal_publication_receipt",
            lambda _value, **_kwargs: {},
        )
        monkeypatch.setattr(
            evaluator,
            "_load_terminal_retry_authorities",
            lambda **_kwargs: (None, None),
        )
        monkeypatch.setattr(
            contract,
            "observe_terminal_publication_supervisor_preflight",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected preflight failure")
            ),
        )

    def persist(**kwargs: Any) -> dict[str, Any]:
        observed.append(kwargs)
        return {"failure_stage": kwargs["failure_stage"]}

    monkeypatch.setattr(
        evaluator, "_persist_pre_phase2_supervisor_failure", persist
    )
    result = evaluator.supervise_v2_terminal_publication(
        attempt_root=root,
        resume_boundary="AFTER_PHASE_1",
        publication_attempt_number=1,
        publication_attempt_id="v2-presentation-1-1",
        tracked_attempt_number=1,
        tracked_attempt_id="v2-tracked-1-1",
        phase_1_custody_content_digest="0" * 64,
        phase_2_custody_content_digest=None,
        phase_3_custody_content_digest=None,
        presentation_retry_authority_content_digest=None,
        tracked_retry_authority_content_digest=None,
    )
    assert result == {"failure_stage": failure_stage}
    assert len(observed) == 1
    assert observed[0]["failure_stage"] == failure_stage
    assert observed[0].get(
        "terminal_publication_supervisor_root_reopen_custody"
    ) == (
        None if failure_stage == "ROOT_REOPEN" else reopened
    )


@pytest.mark.parametrize(
    "failure_stage",
    (
        "CHILD_HANDOFF",
        "CHILD_POPEN",
        "CHILD_IDENTITY_OR_LIFECYCLE",
        "CHILD_EXIT_OBSERVATION_CONSTRUCTION",
    ),
)
def test_phase2_child_supervisor_failures_freeze_sibling_custody(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    root = _synthetic_attempt_root()
    supervisor = {"pid": 41, "start_time_ticks": 43}
    reopened = {
        "supervisor_process_identity": supervisor,
        "publication_attempt_id": "v2-presentation-1-1",
        "tracked_attempt_id": "v2-tracked-1-1",
    }
    preflight = {"preflight": True}
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        contract,
        "expected_internal_argv",
        lambda *_args, **_kwargs: ["phase-2-child"],
    )
    monkeypatch.setattr(
        evaluator,
        "_terminal_child_paths",
        lambda **_kwargs: {
            "stdout": "runtime/stdout",
            "stderr": "runtime/stderr",
            "traceback": "runtime/traceback",
            "exception": "runtime/exception",
            "last_stage": "runtime/last-stage",
            "handoff": "runtime/handoff",
            "produced": "runtime/produced",
        },
    )

    def persist(**kwargs: Any) -> dict[str, Any]:
        observed.append(kwargs)
        return {"failure_stage": kwargs["failure_stage"]}

    monkeypatch.setattr(
        evaluator, "_persist_pre_phase2_supervisor_failure", persist
    )
    if failure_stage == "CHILD_HANDOFF":
        monkeypatch.setattr(
            contract,
            "create_terminal_publication_child_handoff_memfd",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected handoff failure")
            ),
        )
    else:
        handoff_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        monkeypatch.setattr(
            contract,
            "create_terminal_publication_child_handoff_memfd",
            lambda **_kwargs: (handoff_fd, {"handoff": True}),
        )
        monkeypatch.setattr(
            contract,
            "build_terminal_publication_child_environment",
            lambda **_kwargs: {},
        )
        monkeypatch.setattr(
            contract,
            "terminal_publication_child_handoff_pass_fds",
            lambda _custody: (),
        )
        if failure_stage == "CHILD_POPEN":
            monkeypatch.setattr(
                evaluator.subprocess,
                "Popen",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    OSError("injected Popen failure")
                ),
            )
        else:
            class FakeProcess:
                pid = 53
                returncode = 0 if failure_stage == (
                    "CHILD_EXIT_OBSERVATION_CONSTRUCTION"
                ) else 1

                def communicate(self) -> tuple[bytes, bytes]:
                    return b"", b""

                def poll(self) -> int:
                    return self.returncode

                def wait(self, timeout: float = 0) -> int:
                    return self.returncode

            monkeypatch.setattr(
                evaluator.subprocess,
                "Popen",
                lambda *_args, **_kwargs: FakeProcess(),
            )
            if failure_stage == "CHILD_IDENTITY_OR_LIFECYCLE":
                monkeypatch.setattr(
                    evaluator,
                    "_process_identity_for_pid",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("injected identity failure")
                    ),
                )
                monkeypatch.setattr(
                    evaluator,
                    "_bounded_process_drain",
                    lambda *_args, **_kwargs: (b"", b"", True),
                )
            else:
                monkeypatch.setattr(
                    evaluator,
                    "_process_identity_for_pid",
                    lambda *_args, **_kwargs: {
                        "pid": 53,
                        "start_time_ticks": 59,
                    },
                )
                monkeypatch.setattr(
                    evaluator,
                    "_persist_terminal_child_bytes",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("injected exit construction failure")
                    ),
                )
            monkeypatch.setattr(
                evaluator, "_cleanup_remaining_process_group", lambda _pid: []
            )
            monkeypatch.setattr(
                evaluator, "_process_group_members", lambda _pid: []
            )

    result, phase_custody = evaluator._run_terminal_publication_child(
        phase_id=contract.PHASE_IDS[1],
        attempt_root=root,
        phase_1_custody={"phase": 1},
        phase_2_custody=None,
        supervisor_root_reopen_custody=reopened,
        supervisor_preflight=preflight,
        supervisor_resume_preflight=None,
        presentation_retry_authority=None,
    )
    assert result == {"failure_stage": failure_stage}
    assert phase_custody is None
    assert len(observed) == 1
    assert observed[0]["failure_stage"] == failure_stage


@pytest.mark.parametrize(
    "failure_stage",
    (
        "CHILD_HANDOFF_CONSTRUCTION",
        "CHILD_HANDOFF_PERSISTENCE",
        "CHILD_POPEN",
        "CHILD_IDENTITY_OR_LIFECYCLE",
        "CHILD_EXIT_OBSERVATION_CONSTRUCTION",
        "CHILD_EXIT_OBSERVATION_PERSISTENCE",
        "PHASE_CUSTODY_CONSTRUCTION",
        "PHASE_CUSTODY_PERSISTENCE",
    ),
)
def test_phase3_uncanonicalized_failures_freeze_raw_preservation_before_return(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    root = _synthetic_attempt_root()
    supervisor = {"pid": 41, "start_time_ticks": 43}
    reopened = {
        "supervisor_process_identity": supervisor,
        "publication_attempt_number": 1,
        "publication_attempt_id": "v2-presentation-1-1",
        "tracked_attempt_id": "v2-tracked-1-1",
    }
    observed: list[dict[str, Any]] = []
    paths = {
        "stdout": "presentation_attempts/v2-presentation-1-1/stdout",
        "stderr": "presentation_attempts/v2-presentation-1-1/stderr",
        "traceback": "presentation_attempts/v2-presentation-1-1/traceback",
        "exception": "presentation_attempts/v2-presentation-1-1/exception",
        "last_stage": "presentation_attempts/v2-presentation-1-1/last-stage",
        "handoff": "presentation_attempts/v2-presentation-1-1/handoff",
        "produced": "presentation_attempts/v2-presentation-1-1/manifest",
    }
    monkeypatch.setattr(
        contract,
        "expected_internal_argv",
        lambda *_args, **_kwargs: ["phase-3-child"],
    )
    monkeypatch.setattr(
        evaluator, "_terminal_child_paths", lambda **_kwargs: paths
    )

    def persist_raw(**kwargs: Any) -> dict[str, Any]:
        observed.append(kwargs)
        return {
            "preservation_receipt": {
                "schema": contract.RAW_PHASE3_FAILURE_PRESERVATION_RECEIPT_SCHEMA,
                "failure_stage": kwargs["failure_stage"],
            }
        }

    monkeypatch.setattr(
        evaluator, "_persist_raw_phase3_supervisor_failure", persist_raw
    )
    if failure_stage == "CHILD_HANDOFF_CONSTRUCTION":
        monkeypatch.setattr(
            contract,
            "create_terminal_publication_child_handoff_memfd",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected handoff construction failure")
            ),
        )
    else:
        handoff_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        monkeypatch.setattr(
            contract,
            "create_terminal_publication_child_handoff_memfd",
            lambda **_kwargs: (handoff_fd, {"handoff": True}),
        )
        monkeypatch.setattr(
            contract,
            "build_terminal_publication_child_environment",
            lambda **_kwargs: {},
        )
        if failure_stage == "CHILD_HANDOFF_PERSISTENCE":
            monkeypatch.setattr(
                contract,
                "terminal_publication_child_handoff_pass_fds",
                lambda _value: (_ for _ in ()).throw(
                    RuntimeError("injected handoff persistence failure")
                ),
            )
        else:
            monkeypatch.setattr(
                contract,
                "terminal_publication_child_handoff_pass_fds",
                lambda _value: (),
            )
            if failure_stage == "CHILD_POPEN":
                monkeypatch.setattr(
                    evaluator.subprocess,
                    "Popen",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        OSError("injected Phase-3 Popen failure")
                    ),
                )
            else:
                class FakeProcess:
                    pid = 53
                    returncode = 0

                    def communicate(self) -> tuple[bytes, bytes]:
                        return b"", b""

                    def poll(self) -> int:
                        return self.returncode

                    def wait(self, timeout: float = 0) -> int:
                        return self.returncode

                monkeypatch.setattr(
                    evaluator.subprocess,
                    "Popen",
                    lambda *_args, **_kwargs: FakeProcess(),
                )
                if failure_stage == "CHILD_IDENTITY_OR_LIFECYCLE":
                    monkeypatch.setattr(
                        evaluator,
                        "_process_identity_for_pid",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("injected Phase-3 identity failure")
                        ),
                    )
                    monkeypatch.setattr(
                        evaluator,
                        "_bounded_process_drain",
                        lambda *_args, **_kwargs: (b"", b"", True),
                    )
                else:
                    monkeypatch.setattr(
                        evaluator,
                        "_process_identity_for_pid",
                        lambda *_args, **_kwargs: {
                            "pid": 53,
                            "start_time_ticks": 59,
                        },
                    )
                    if failure_stage == (
                        "CHILD_EXIT_OBSERVATION_CONSTRUCTION"
                    ):
                        monkeypatch.setattr(
                            evaluator,
                            "_persist_terminal_child_bytes",
                            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                                RuntimeError(
                                    "injected exit-observation construction failure"
                                )
                            ),
                        )
                    else:
                        monkeypatch.setattr(
                            evaluator,
                            "_persist_terminal_child_bytes",
                            lambda *_args, **_kwargs: None,
                        )
                        monkeypatch.setattr(
                            evaluator,
                            "_terminal_child_last_stage",
                            lambda **_kwargs: "COMPLETE",
                        )
                        monkeypatch.setattr(
                            evaluator,
                            "_load_terminal_child_handoff_if_present",
                            lambda **_kwargs: (
                                {"handoff": True},
                                {"binding": True},
                            ),
                        )
                        monkeypatch.setattr(
                            evaluator,
                            "_artifact_binding",
                            lambda *_args, **_kwargs: {"artifact": True},
                        )
                        monkeypatch.setattr(
                            evaluator,
                            "_require_attempt_io",
                            lambda _root: (71, {"root": True}, reopened),
                        )
                        monkeypatch.setattr(
                            contract,
                            "observe_terminal_publication_child_exit",
                            lambda **_kwargs: {"observation_complete": True},
                        )
                        monkeypatch.setattr(
                            contract,
                            "write_terminal_publication_child_exit_observation_exclusive_fsync",
                            (
                                lambda **_kwargs: (_ for _ in ()).throw(
                                    RuntimeError(
                                        "injected exit-observation persistence failure"
                                    )
                                )
                                if failure_stage
                                == "CHILD_EXIT_OBSERVATION_PERSISTENCE"
                                else lambda **_kwargs: {"binding": True}
                            ),
                        )
                        monkeypatch.setattr(
                            contract,
                            "build_terminal_publication_phase_custody",
                            (
                                lambda **_kwargs: (_ for _ in ()).throw(
                                    RuntimeError(
                                        "injected phase-custody construction failure"
                                    )
                                )
                                if failure_stage == "PHASE_CUSTODY_CONSTRUCTION"
                                else lambda **_kwargs: {"custody": True}
                            ),
                        )
                        monkeypatch.setattr(
                            contract,
                            "write_terminal_publication_phase_custody_exclusive_fsync",
                            (
                                lambda **_kwargs: (_ for _ in ()).throw(
                                    RuntimeError(
                                        "injected phase-custody persistence failure"
                                    )
                                )
                                if failure_stage == "PHASE_CUSTODY_PERSISTENCE"
                                else lambda **_kwargs: {"binding": True}
                            ),
                        )
                monkeypatch.setattr(
                    evaluator,
                    "_cleanup_remaining_process_group",
                    lambda _pid: [],
                )
                monkeypatch.setattr(
                    evaluator, "_process_group_members", lambda _pid: []
                )

    result, phase_custody = evaluator._run_terminal_publication_child(
        phase_id=contract.PHASE_IDS[2],
        attempt_root=root,
        phase_1_custody={"phase": 1},
        phase_2_custody={"phase": 2},
        supervisor_root_reopen_custody=reopened,
        supervisor_preflight={"preflight": True},
        supervisor_resume_preflight=None,
        presentation_retry_authority=None,
    )
    assert result == {
        "schema": contract.RAW_PHASE3_FAILURE_PRESERVATION_RECEIPT_SCHEMA,
        "failure_stage": failure_stage,
    }
    assert phase_custody is None
    assert len(observed) == 1
    assert observed[0]["failure_stage"] == failure_stage


def test_raw_phase3_clean_absent_failure_may_feed_explicit_retry_but_surviving_child_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    number = 1
    publication_id = "v2-presentation-1-1"
    before = {
        "inventory_stage": "RAW_FAILURE_BEFORE_CUSTODY_WRITE",
        "publication_attempt_number": number,
        "publication_attempt_id": publication_id,
        "attempt_root": str(_synthetic_attempt_root()),
        "namespace_state": "ABSENT",
        "pass": True,
        "observation_completed_monotonic_ns": 1,
    }
    after = {
        **before,
        "inventory_stage": "RAW_FAILURE_AFTER_CUSTODY_WRITE",
        "observation_completed_monotonic_ns": 2,
    }
    preserved = {
        **before,
        "inventory_stage": "RAW_FAILURE_PRESERVED_READ_ONLY",
        "observation_completed_monotonic_ns": 3,
    }
    monkeypatch.setattr(
        contract,
        "validate_raw_phase3_supervisor_failure_custody",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        contract,
        "validate_raw_phase3_namespace_inventory",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        contract,
        "_raw_phase3_inventory_bytes_and_inode_equal",
        lambda _left, _right: True,
    )

    def build(surviving: bool) -> dict[str, Any]:
        failure_observation = {
            "child_process_absence_observation": {
                "exact_historical_process_absent": not surviving
            },
            "active_other_v2_role_rows": (
                [{"pid": 53}] if surviving else []
            ),
            "child_process_group_rows": [],
            "other_process_open_attempt_root_rows": [],
            "other_v2_gpu_or_kfd_rows": [],
            "other_v2_lock_observation": {"matching_rows": []},
            "process_group_members_after_cleanup": [],
            "cleanup_complete": not surviving,
            "popen_succeeded": False,
            "process_reaped": False,
            "communicate_completed": False,
        }
        custody = {
            "attempt_id": "v2-1-1",
            "attempt_root": str(_synthetic_attempt_root()),
            "publication_attempt_number": number,
            "publication_attempt_id": publication_id,
            "failure_custody_relative_path": (
                contract.raw_phase3_supervisor_failure_custody_relative_path(
                    number, publication_id
                )
            ),
            "namespace_inventory_before_failure_custody_write": before,
            "failure_observation": failure_observation,
            "known_pre_popen_namespace_absent": True,
            "content_digest": "0" * 64,
        }
        payload = contract.canonical_json_bytes(custody) + b"\n"
        binding = contract.build_artifact_binding(
            path=custody["failure_custody_relative_path"],
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes_count=len(payload),
            content_digest=custody["content_digest"],
            rows=None,
        )
        return contract.build_raw_phase3_failure_preservation_receipt(
            failure_custody=custody,
            failure_custody_binding=binding,
            inventory_after_failure_custody_write=after,
            preserved_inventory=preserved,
            preservation_action_rows=[
                {
                    "path": contract.publication_attempt_relative_root(
                        number, publication_id
                    ),
                    "action": "NAMESPACE_ABSENT_NO_ACTION",
                    "completed": True,
                    "observed_stat": None,
                }
            ],
            preservation_error_rows=[],
            preservation_completed_monotonic_ns=4,
        )

    clean = build(False)
    assert clean["failure_child_process_absent"] is True
    assert clean["failure_residual_resources_zero"] is True
    assert clean["failure_cleanup_complete"] is True
    assert clean["preservation_complete"] is True
    assert clean[
        "explicit_new_presentation_attempt_authority_possible"
    ] is True
    assert clean["automatic_retry"] is False

    surviving = build(True)
    assert surviving["failure_child_process_absent"] is False
    assert surviving["failure_residual_resources_zero"] is False
    assert surviving["failure_cleanup_complete"] is False
    assert surviving["preservation_complete"] is False
    assert surviving[
        "explicit_new_presentation_attempt_authority_possible"
    ] is False
    assert surviving["external_repair_and_live_revalidation_required"] is True


@pytest.mark.parametrize(
    ("entrypoint", "failure_type", "failure_message"),
    (
        ("phase2", FileNotFoundError, "missing sealed handoff fd"),
        ("phase2", RuntimeError, "mutated sealed handoff payload"),
        ("phase2", OSError, "closed sealed handoff fd"),
        ("phase3", FileNotFoundError, "missing sealed handoff fd"),
        ("phase3", RuntimeError, "mutated sealed handoff payload"),
        ("phase3", OSError, "closed sealed handoff fd"),
    ),
)
def test_terminal_child_handoff_failure_precedes_root_reopen_and_payload_read(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    failure_type: type[BaseException],
    failure_message: str,
) -> None:
    events: list[str] = []
    expected_argv = ["frozen-v2-child"]
    monkeypatch.setattr(
        evaluator,
        "_process_identity",
        lambda: {
            "pid": 17,
            "start_time_ticks": 19,
            "ppid": 11,
            "argv": expected_argv,
            "cwd": str(contract.REPO_ROOT),
            "executable": str(contract.PYTHON_EXECUTABLE),
        },
    )
    monkeypatch.setattr(
        contract, "expected_internal_argv", lambda *_args, **_kwargs: expected_argv
    )

    def fail_handoff(**_kwargs: Any) -> dict[str, Any]:
        events.append("consume_handoff")
        raise failure_type(failure_message)

    def unexpected_reopen(**_kwargs: Any) -> dict[str, Any]:
        events.append("reopen_root")
        raise AssertionError("attempt root reopened after handoff failure")

    def unexpected_payload_read(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("payload_read")
        raise AssertionError("payload read after handoff failure")

    monkeypatch.setattr(
        contract, "consume_terminal_publication_child_handoff", fail_handoff
    )
    monkeypatch.setattr(evaluator, "_activate_reopened_attempt_io", unexpected_reopen)
    monkeypatch.setattr(evaluator, "_load_canonical_json", unexpected_payload_read)
    root = _synthetic_attempt_root()
    with pytest.raises(failure_type, match=failure_message):
        if entrypoint == "phase2":
            evaluator.validate_immutable_payload(root)
        else:
            evaluator.publish_validated_presentation(
                root, 1, "v2-presentation-1-1", None
            )
    assert events == ["consume_handoff"]


def test_phase2_rebuilds_conditional_components_without_producer_deep_copy() -> None:
    source = _function_source("_regenerate_scientific_payload")
    assert "copy.deepcopy(persisted_stage_b)" not in source
    assert "copy.deepcopy(persisted_stage_c)" not in source
    assert "stage_b != persisted_stage_b" in source
    assert "stage_c != persisted_stage_c" in source
    calls = set(_ordered_call_names("_regenerate_scientific_payload"))
    assert {
        "_load_stage_b_helper_outputs",
        "_load_stage_c_helper_outputs",
        "_replay_stage_a_metrics",
        "_replay_conditional_metrics",
        "_stage_a_decisions",
        "_stage_b_decisions",
        "_stage_c_decisions",
        "build_scientific_invariance_receipt",
        "build_immutable_scientific_payload",
    } <= calls


def test_phase1_first_open_witness_is_durable_before_split_and_stage_a() -> None:
    calls = _ordered_call_names("execute_phase_1")
    for earlier, later in zip(
        (
            "write_scientific_attempt_boundary_entry_exclusive_fsync",
            "write_first_scientific_open_initiation_exclusive_fsync",
            "split_ids",
            "observe_first_scientific_open_event_in_flight",
            "build_first_scientific_open_receipt",
        ),
        (
            "write_first_scientific_open_initiation_exclusive_fsync",
            "split_ids",
            "observe_first_scientific_open_event_in_flight",
            "build_first_scientific_open_receipt",
            "_run_stage_a",
        ),
    ):
        assert calls.index(earlier) < calls.index(later)


def test_static_audit_stream_uses_unfiltered_rows_and_contract_correlation() -> None:
    finish_source = _function_source("_finish_static_audit")
    hook_source = _function_source("_static_audit_hook")
    assert "raw_audit_open_rows=capture['rows']" in finish_source
    assert "contract_openat_audit_ledger=ledger" in finish_source
    assert "dropped_event_count=0" in finish_source
    assert "current_contract_openat_audit_correlation" in hook_source
    assert "correlation_token" in hook_source


def test_nested_workers_share_helper_group_and_persist_failure_custody() -> None:
    source = _function_source("_run_phase1_worker_process")
    assert "start_new_session=False" in source
    assert "_terminate_nested_worker" in source
    assert "build_phase1_worker_exit_custody" in source
    assert "paths['exit_custody']" in source
    assert source.index("paths['exit_custody']") < source.index(
        "raise V2Phase1WorkerProcessError"
    )


def test_git_commit_spawn_failure_uses_the_exact_failure_union() -> None:
    function = _function_node("_commit_tracked_publication")
    handlers = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "OSError"
    ]
    assert len(handlers) == 1
    source = ast.unparse(handlers[0])
    assert "spawn_error" in source
    assert "structured_exception" in source
    function_source = ast.unparse(function)
    assert "returncode=returncode" in function_source
    assert "spawn_error=spawn_error" in function_source
    assert "structured_exception=structured_exception" in function_source


def test_reused_staging_skips_a_second_git_index_mutation() -> None:
    source = _function_source("_publish_terminal_tracked_result")
    assert "TRACKED_PUBLICATION_REUSED_STAGING_RECEIPT_SCHEMA" in source
    assert "git_index_receipt = staging['prior_git_index_receipt']" in source
    assert "git_index_binding = staging['prior_git_index_receipt_binding']" in source
    assert source.index("else:") < source.index(
        "stage_tracked_publication_git_index_transactional"
    )


def test_v2_registered_regular_file_nlink_one_round_trip() -> None:
    authority = contract.registered_static_input_file_authority_rows(
        "PHASE1_STAGE_A"
    )[0]
    path = Path(authority["path"])
    stat_row = _extended_stat_row(path)
    identity = _synthetic_process_identity(
        argv=list(contract.PUBLIC_EXECUTE_ARGV)
    )
    binding = contract.build_artifact_binding(
        path=str(path),
        sha256=authority["sha256"],
        bytes_count=authority["bytes"],
        content_digest=authority["content_digest"],
        rows=None,
    )
    receipt = contract.build_static_input_open_event(
        stage_id="PHASE1_STAGE_A",
        process_identity=identity,
        path=str(path),
        audit_event_name="open",
        audit_path=str(path),
        open_started_monotonic_ns=11,
        open_completed_monotonic_ns=12,
        stat_before=stat_row,
        stat_opened=stat_row,
        stat_after=stat_row,
        observed_binding=binding,
    )
    assert contract.validate_static_input_open_event(receipt) == receipt
    assert receipt["stat_opened"]["nlink"] == 1
    assert receipt["post_open_bytes_rehashed_no_follow"] is True


def test_v2_attempt_root_custody_survives_authorized_directory_growth_and_rejects_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonreuse = _synthetic_namespace_receipt(
        tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    producer = _synthetic_process_identity(
        argv=list(contract.PUBLIC_EXECUTE_ARGV)
    )
    parent_fd = root_fd = reopened_parent_fd = reopened_root_fd = -1
    original_root = Path(nonreuse["attempt_root"])
    displaced_root = original_root.with_name(original_root.name + ".original")
    try:
        parent_fd, root_fd, custody = contract.create_fresh_attempt_root_anchored(
            namespace_and_nonreuse_receipt=nonreuse,
            producer_process_identity=producer,
        )
        (original_root / "authorized-growth").mkdir(mode=0o700)
        assert contract.validate_attempt_root_fd_custody(
            custody, parent_fd=parent_fd, root_fd=root_fd, reverify_live=True
        ) == custody

        phase2_identity = _synthetic_process_identity(
            argv=contract.expected_internal_argv(
                contract.PHASE2_VALIDATOR_MODE, original_root
            ),
            pid=70_002,
        )
        monkeypatch.setattr(
            contract,
            "_require_exact_live_process_identity",
            lambda value: dict(value),
        )
        reopened_parent_fd, reopened_root_fd, reopened = (
            contract.reopen_attempt_root_anchored(
                attempt_root_fd_custody=custody,
                phase_id=contract.PHASE_IDS[1],
                phase_process_identity=phase2_identity,
            )
        )
        assert reopened["same_historical_root_identity"] is True
        os.close(reopened_root_fd)
        reopened_root_fd = -1
        os.close(reopened_parent_fd)
        reopened_parent_fd = -1
        os.close(root_fd)
        root_fd = -1
        os.close(parent_fd)
        parent_fd = -1

        original_root.rename(displaced_root)
        original_root.mkdir(mode=0o700)
        with pytest.raises(contract.V2ContractError, match="identity drift"):
            contract.reopen_attempt_root_anchored(
                attempt_root_fd_custody=custody,
                phase_id=contract.PHASE_IDS[1],
                phase_process_identity=phase2_identity,
            )
    finally:
        for descriptor in (
            reopened_root_fd,
            reopened_parent_fd,
            root_fd,
            parent_fd,
        ):
            if descriptor >= 0:
                os.close(descriptor)
        if original_root.exists():
            shutil.rmtree(original_root)
        if displaced_root.exists():
            shutil.rmtree(displaced_root)


def test_phase1_supervisor_bootstrap_allows_root_owned_ancestors_and_rejects_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonreuse = _synthetic_namespace_receipt(
        tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    producer = _synthetic_process_identity(
        argv=list(contract.PUBLIC_EXECUTE_ARGV)
    )
    parent_fd = root_fd = -1
    original_root = Path(nonreuse["attempt_root"])
    displaced_root = original_root.with_name(original_root.name + ".original")
    try:
        parent_fd, root_fd, custody = (
            contract.create_fresh_attempt_root_anchored(
                namespace_and_nonreuse_receipt=nonreuse,
                producer_process_identity=producer,
            )
        )
        assert any(
            ancestor != Path("/")
            and os.stat(ancestor, follow_symlinks=False).st_uid != os.getuid()
            for ancestor in original_root.parents
        )
        assert evaluator._load_attempt_root_custody_for_reopen(
            original_root
        ) == custody

        os.close(root_fd)
        root_fd = -1
        os.close(parent_fd)
        parent_fd = -1
        original_root.rename(displaced_root)
        original_root.mkdir(mode=contract.ATTEMPT_ROOT_DIRECTORY_MODE)
        (original_root / "receipts").mkdir(
            mode=contract.RUNTIME_DIRECTORY_MODE
        )
        stale_receipt = displaced_root / contract.RUNTIME_PATHS[
            "attempt_root_fd_custody"
        ]
        replacement_receipt = original_root / contract.RUNTIME_PATHS[
            "attempt_root_fd_custody"
        ]
        shutil.copyfile(stale_receipt, replacement_receipt)
        replacement_receipt.chmod(contract.RUNTIME_FILE_MODE)
        with pytest.raises(
            evaluator.V2EvaluationError,
            match="historical attempt-root custody drift",
        ):
            evaluator._load_attempt_root_custody_for_reopen(original_root)
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        if original_root.exists():
            shutil.rmtree(original_root)
        if displaced_root.exists():
            shutil.rmtree(displaced_root)


def test_v2_presentation_attempt_id_gate_rejects_canonical_or_raw_failure_and_accepts_fresh_attempt2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonreuse = _synthetic_namespace_receipt(
        tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    producer = _synthetic_process_identity(
        argv=list(contract.PUBLIC_EXECUTE_ARGV)
    )
    observer = _synthetic_process_identity(
        argv=["terminal-supervisor"], pid=70_002
    )
    monkeypatch.setattr(
        contract,
        "_require_exact_live_process_identity",
        lambda value: dict(value),
    )
    parent_fd = root_fd = -1
    attempt_root = Path(nonreuse["attempt_root"])
    try:
        parent_fd, root_fd, custody = contract.create_fresh_attempt_root_anchored(
            namespace_and_nonreuse_receipt=nonreuse,
            producer_process_identity=producer,
        )
        consumed = (
            (
                "v2-presentation-2-1",
                contract.publication_attempt_runtime_paths(
                    2, "v2-presentation-2-1"
                )["failure_custody"],
            ),
            (
                "v2-presentation-2-2",
                contract.raw_phase3_supervisor_failure_custody_relative_path(
                    2, "v2-presentation-2-2"
                ),
            ),
        )
        for publication_id, relative_path in consumed:
            contract.write_v2_attempt_bytes_exclusive_fsync(
                attempt_root_fd_custody=custody,
                root_fd=root_fd,
                relative_path=relative_path,
                payload=b"{}\n",
                content_digest=None,
                rows=None,
            )
            with pytest.raises(
                contract.V2ContractError,
                match="already has failure custody",
            ):
                contract.observe_presentation_attempt_failure_custody_absence(
                    attempt_root_fd_custody=custody,
                    root_fd=root_fd,
                    observer_process_identity=observer,
                    publication_attempt_number=2,
                    publication_attempt_id=publication_id,
                )

        fresh = contract.observe_presentation_attempt_failure_custody_absence(
            attempt_root_fd_custody=custody,
            root_fd=root_fd,
            observer_process_identity=observer,
            publication_attempt_number=2,
            publication_attempt_id="v2-presentation-2-3",
        )
        assert contract.validate_presentation_attempt_failure_custody_absence(
            fresh
        ) == fresh
        assert fresh["requested_presentation_attempt_id_unconsumed"] is True
        assert fresh["failure_custody_paths_checked"] == 4
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        if attempt_root.exists():
            shutil.rmtree(attempt_root)


def test_v2_pre_root_identity_failure_after_root_or_boundary_creation_is_unknown_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _synthetic_process_identity(
        argv=contract.expected_phase1_supervisor_argv("v2-1-1")
    )
    attestor = _synthetic_process_identity(
        argv=contract.expected_pre_root_phase1_failure_custody_argv("0" * 64),
        pid=70_002,
        ppid=supervisor["pid"],
    )
    observed = {
        "attempt_id": "v2-1-1",
        "content_digest": "0" * 64,
        "supervisor_process_identity": supervisor,
        "producer_process_identity": None,
        "spawned_pid": 70_003,
        "spawned_process_group_id": 70_003,
        "popen_succeeded": True,
        "identity_observation_error": {"type": "ProcessLookupError"},
        "exact_outer_supervisor_argv": supervisor["argv"],
        "stream_bindings": {
            key: {"path": key, "sha256": "0" * 64, "bytes": 0}
            for key in ("stdout", "stderr", "traceback", "exception", "last_stage")
        },
        "structured_exception": {"type": "ProcessLookupError"},
        "last_stage": "CHILD_IDENTITY",
        "returncode": None,
        "termination_signal": None,
        "popen_started_monotonic_ns": 1,
        "ended_monotonic_ns": 2,
        "cleanup_actions": ["SIGKILL_PROCESS_GROUP"],
        "process_group_members_after_cleanup": [],
        "active_v2_role_rows_after_cleanup": [],
        "gpu_process_rows_after_cleanup": [],
        "kfd_rows_after_cleanup": [],
        "lock_rows_after_cleanup": [],
        "open_file_rows_after_cleanup": [],
        "communication_attempted": True,
        "communicate_completed": False,
        "process_reaped": False,
        "post_cleanup_start_namespace_observation": {},
        "start_state_classification": "UNKNOWN_START_STATE",
        "zero_science_proved": False,
        "attempt_root_present_after_cleanup": True,
        "scientific_attempt_reserved": None,
        "scientific_attempt_consumed": None,
        "residual_resources_zero": False,
    }
    monkeypatch.setattr(
        contract,
        "validate_pre_root_phase1_supervisor_observation",
        lambda value: dict(value),
    )
    monkeypatch.setattr(contract, "_validate_process_identity", lambda value: value)
    monkeypatch.setattr(
        contract,
        "_build_terminal_scientific_attempt_disposition",
        lambda **kwargs: {"disposition": kwargs["disposition"]},
    )
    custody = contract.build_pre_root_phase1_failure_custody(
        supervisor_observation=observed,
        terminal_attestor_process_identity=attestor,
    )
    assert custody["attempt_accounting"]["state"] == "UNKNOWN_NO_RETRY"
    assert custody["scientific_attempt_disposition"]["disposition"] == (
        "UNKNOWN_NO_RETRY"
    )
    assert custody["scientific_counters"] is None

    false_zero_science = dict(custody)
    false_zero_science["scientific_counters"] = contract.zero_scientific_counters()
    false_zero_science = contract.attach_self_digest(false_zero_science)
    with pytest.raises(contract.V2ContractError, match="drift"):
        contract.validate_pre_root_phase1_failure_custody(false_zero_science)


def test_v2_exhaustive_open_stream_rejects_forbidden_unregistered_or_dropped_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _synthetic_process_identity(
        argv=contract.expected_internal_argv(
            contract.PHASE2_VALIDATOR_MODE, _synthetic_attempt_root()
        )
    )
    monkeypatch.setattr(
        contract, "_validate_static_stage_event_process", lambda **_kwargs: identity
    )
    monkeypatch.setattr(
        contract, "validate_static_input_snapshot", lambda value: value
    )
    monkeypatch.setattr(
        contract, "validate_static_input_open_event", lambda value: value
    )
    classification = "UNREGISTERED_OPEN"

    def classify(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = []
        for sequence, raw in enumerate(kwargs["raw_audit_open_rows"]):
            rows.append(
                {
                    **raw,
                    "sequence": sequence,
                    "classification": classification,
                    "classified_path": raw["audit_path"],
                }
            )
        return rows, dict(kwargs["contract_openat_audit_ledger"])

    monkeypatch.setattr(contract, "_classify_and_bind_static_audit_rows", classify)
    pre = {
        "stage_id": "PHASE2_REPLAY",
        "snapshot_kind": "PRE_STAGE",
        "stage_process_identity": identity,
        "attempt_root_fd_custody_content_digest": "0" * 64,
        "reopened_attempt_root_fd_custody_content_digest": "1" * 64,
        "started_monotonic_ns": 2,
        "ended_monotonic_ns": 5,
        "registered_file_authority_rows": [],
        "observed_file_rows": [],
    }
    post = {
        **pre,
        "snapshot_kind": "POST_STAGE",
        "started_monotonic_ns": 25,
        "ended_monotonic_ns": 28,
    }
    raw = {
        "sequence": 0,
        "audit_event_name": "open",
        "audit_path": "/forbidden",
        "audit_mode": "r",
        "audit_flags": 0,
        "audit_cwd": str(contract.REPO_ROOT),
        "correlation_token": None,
        "correlation_context": None,
        "open_started_monotonic_ns": 11,
        "open_completed_monotonic_ns": 12,
    }
    for classification_value, dropped, raw_rows in (
        ("FORBIDDEN_FAILED_OUTPUT", 0, [raw]),
        ("FORBIDDEN_CORRECTION_OR_RESULT_REPLAY", 0, [raw]),
        ("UNREGISTERED_SCIENTIFIC_INPUT", 0, [raw]),
        ("UNREGISTERED_OPEN", 0, [raw]),
        ("AUTHORIZED_TECHNICAL_RUNTIME", 1, []),
    ):
        classification = classification_value
        audit = contract.build_static_input_audit_stream_receipt(
            stage_id="PHASE2_REPLAY",
            process_identity=identity,
            attempt_root=_synthetic_attempt_root(),
            audit_hook_installed_monotonic_ns=1,
            audit_stream_completed_monotonic_ns=30,
            raw_audit_open_rows=raw_rows,
            registered_static_open_events=[],
            contract_openat_audit_ledger={"rows": []},
            dropped_event_count=dropped,
        )
        assert audit["pass"] is False
        with pytest.raises(
            contract.V2ContractError,
            match="static-input exhaustive audit custody drift",
        ):
            contract.build_static_input_stage_custody(
                pre_snapshot=pre,
                post_snapshot=post,
                static_input_audit_stream=audit,
                stage_started_monotonic_ns=10,
                stage_ended_monotonic_ns=20,
            )


def test_v2_static_stage_custody_rejects_same_argv_different_pid_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = contract.expected_internal_argv(
        contract.PHASE2_VALIDATOR_MODE, _synthetic_attempt_root()
    )
    identity = _synthetic_process_identity(argv=argv)
    other_identity = _synthetic_process_identity(argv=argv, pid=70_002)
    monkeypatch.setattr(
        contract, "validate_static_input_snapshot", lambda value: value
    )
    monkeypatch.setattr(
        contract, "validate_static_input_audit_stream_receipt", lambda value: value
    )
    pre = {
        "stage_id": "PHASE2_REPLAY",
        "snapshot_kind": "PRE_STAGE",
        "stage_process_identity": identity,
        "attempt_root_fd_custody_content_digest": "0" * 64,
        "reopened_attempt_root_fd_custody_content_digest": "1" * 64,
        "started_monotonic_ns": 2,
        "ended_monotonic_ns": 5,
        "registered_file_authority_rows": [],
        "observed_file_rows": [],
    }
    post = {**pre, "snapshot_kind": "POST_STAGE", "started_monotonic_ns": 25}
    audit = {
        "stage_id": "PHASE2_REPLAY",
        "process_identity": other_identity,
        "audit_hook_installed_monotonic_ns": 1,
        "audit_stream_completed_monotonic_ns": 30,
        "dropped_event_count": 0,
        "success": True,
        "pass": True,
    }
    with pytest.raises(
        contract.V2ContractError,
        match="static-input exhaustive audit custody drift",
    ):
        contract.build_static_input_stage_custody(
            pre_snapshot=pre,
            post_snapshot=post,
            static_input_audit_stream=audit,
            stage_started_monotonic_ns=10,
            stage_ended_monotonic_ns=20,
        )


def test_v2_pre_snapshot_after_first_scientific_open_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _synthetic_process_identity(argv=list(contract.PUBLIC_EXECUTE_ARGV))
    monkeypatch.setattr(
        contract, "validate_static_input_snapshot", lambda value: value
    )
    monkeypatch.setattr(
        contract, "validate_static_input_audit_stream_receipt", lambda value: value
    )
    monkeypatch.setattr(contract, "validate_preexecution_receipt", lambda value: value)
    monkeypatch.setattr(
        contract,
        "validate_scientific_attempt_boundary_entry_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        contract,
        "validate_first_scientific_open_receipt",
        lambda value, **_kwargs: value,
    )
    path = "/registered/split.json"
    pre = {
        "content_digest": "0" * 64,
        "stage_id": "PHASE1_STAGE_A",
        "snapshot_kind": "PRE_STAGE",
        "stage_process_identity": identity,
        "attempt_root_fd_custody_content_digest": "1" * 64,
        "reopened_attempt_root_fd_custody_content_digest": None,
        "started_monotonic_ns": 19,
        "ended_monotonic_ns": 20,
        "registered_file_authority_rows": [{"path": path}],
        "observed_file_rows": [],
    }
    post = {**pre, "snapshot_kind": "POST_STAGE", "started_monotonic_ns": 50}
    preexecution = {
        "producer_process_identity": identity,
        "static_input_metadata_snapshot": pre,
        "static_input_metadata_snapshot_content_digest": pre["content_digest"],
        "completed_monotonic_ns": 10,
    }
    boundary = {"boundary_entered_monotonic_ns": 25}
    first_open = {
        "first_open_completed_monotonic_ns": 30,
        "first_scientific_open_event": {"path": path},
    }
    audit = {
        "stage_id": "PHASE1_STAGE_A",
        "process_identity": identity,
        "audit_hook_installed_monotonic_ns": 1,
        "audit_stream_completed_monotonic_ns": 60,
        "dropped_event_count": 0,
        "success": True,
        "pass": True,
        "registered_static_open_events": [],
    }
    with pytest.raises(contract.V2ContractError, match="first-open order drift"):
        contract.build_static_input_stage_custody(
            pre_snapshot=pre,
            post_snapshot=post,
            static_input_audit_stream=audit,
            stage_started_monotonic_ns=40,
            stage_ended_monotonic_ns=45,
            phase1_preexecution_receipt=preexecution,
            phase1_boundary_entry_receipt=boundary,
            first_scientific_open_receipt=first_open,
        )


def test_v2_stage_b_gate_rejects_inconsistent_derangement_or_mismatched_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preexecution = contract.attach_self_digest(
        {"source_freeze_commit": "0" * 40}
    )
    evidence = {
        "stage_a_decisions": {"derangement": {"pass": True}},
        "prerequisite_artifact_bindings": {
            "evaluation_contract": {"path": "evaluation"},
            "latent_checkpoint": {"path": "latent"},
        },
    }
    evidence_binding = {"path": contract.RUNTIME_PATHS["stage_a_gate_evidence"]}
    monkeypatch.setattr(contract, "validate_preexecution_receipt", lambda value: value)
    monkeypatch.setattr(
        contract, "validate_stage_a_gate_evidence", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(
        contract, "validate_artifact_binding", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(contract, "_validate_json_component_binding", lambda *_args: None)
    receipt = contract.build_stage_b_gate_receipt(
        preexecution_receipt=preexecution,
        stage_a_gate_evidence=evidence,
        stage_a_gate_evidence_binding=evidence_binding,
    )
    assert contract.validate_stage_b_gate_receipt(
        receipt, preexecution_receipt=preexecution
    ) == receipt
    for field, value in (
        ("stage_a_decisions", {"derangement": {"pass": False}}),
        ("latent_ranker_checkpoint_binding", {"path": "other"}),
    ):
        tampered = dict(receipt)
        tampered[field] = value
        tampered = contract.attach_self_digest(tampered)
        with pytest.raises(contract.V2ContractError, match="receipt drift"):
            contract.validate_stage_b_gate_receipt(
                tampered, preexecution_receipt=preexecution
            )


def test_v2_terminal_bundle_inventory_rejects_extra_symlink_or_hardlink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "_validate_process_identity", lambda value: value)
    monkeypatch.setattr(contract, "_validate_extended_stat_row", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        contract, "_presentation_inventory_leaf_authority", lambda *_args: ({}, set())
    )
    inventory_core = {"directory_stat": {}, "leaf_names": [], "leaf_rows": []}
    receipt = contract.attach_self_digest(
        {
            "schema": contract.PRESENTATION_ATTEMPT_NAMESPACE_INVENTORY_SCHEMA,
            "experiment_id": contract.EXPERIMENT_ID,
            "attempt_root": str(_synthetic_attempt_root()),
            "publication_attempt_number": 1,
            "publication_attempt_id": "v2-presentation-1-1",
            "publication_attempt_relative_root": contract.publication_attempt_relative_root(
                1, "v2-presentation-1-1"
            ),
            "inventory_stage": "FAILED_BEFORE_FAILURE_CUSTODY_WRITE",
            "observer_process_identity": {},
            "observation_started_monotonic_ns": 1,
            "observation_completed_monotonic_ns": 2,
            "attempt_root_fd_custody_content_digest": "0" * 64,
            "reopened_attempt_root_fd_custody_content_digest": "1" * 64,
            "directory_stat": {},
            "leaf_names_before": [],
            "leaf_names_after": [],
            "leaf_rows": [],
            "leaf_count": 0,
            "total_leaf_bytes": 0,
            "namespace_inventory_sha256": contract.canonical_json_sha256(
                inventory_core
            ),
            "symlinks": 0,
            "hardlinks": 0,
            "devices_or_nonregular_leaves": 0,
            "extra_paths": [],
            "complete_componentwise_no_follow_inventory": True,
            "pass": True,
        }
    )
    assert contract.validate_presentation_attempt_namespace_inventory(receipt) == receipt
    for field in ("symlinks", "hardlinks"):
        tampered = dict(receipt)
        tampered[field] = 1
        tampered = contract.attach_self_digest(tampered)
        with pytest.raises(contract.V2ContractError, match="inventory receipt drift"):
            contract.validate_presentation_attempt_namespace_inventory(tampered)


@pytest.mark.parametrize("live_role", ("terminal_attestor", "supervisor"))
def test_v2_phase2_preflight_rejects_live_terminal_attestor_or_supervisor(
    monkeypatch: pytest.MonkeyPatch, live_role: str
) -> None:
    root = _synthetic_attempt_root()
    producer = _synthetic_process_identity(argv=list(contract.PUBLIC_EXECUTE_ARGV))
    attestor = _synthetic_process_identity(argv=["attestor"], pid=70_002)
    phase1_supervisor = _synthetic_process_identity(argv=["phase1-supervisor"], pid=70_003)
    terminal_supervisor = _synthetic_process_identity(argv=["terminal-supervisor"], pid=70_004)
    validator = _synthetic_process_identity(
        argv=contract.expected_internal_argv(contract.PHASE2_VALIDATOR_MODE, root),
        pid=70_005,
        ppid=terminal_supervisor["pid"],
    )
    phase1 = {
        "phase_id": contract.PHASE_IDS[0],
        "pass": True,
        "attempt_root": str(root),
        "content_digest": "1" * 64,
        "process_identity": producer,
        "observer_process_identity": attestor,
        "phase1_external_exit_observation": {
            "supervisor_process_identity": phase1_supervisor,
            "supervisor_observation": {"spawned_process_group_id": None},
        },
    }
    supervisor_root = {
        "supervisor_process_identity": terminal_supervisor,
        "resume_boundary": "AFTER_PHASE_1",
        "phase_1_custody_content_digest": phase1["content_digest"],
        "process_local_root_fd": 999_999,
        "content_digest": "2" * 64,
    }
    preflight = {
        "supervisor_process_identity": terminal_supervisor,
        "supervisor_root_reopen_custody_content_digest": supervisor_root[
            "content_digest"
        ],
    }
    handoff = {
        "phase_id": contract.PHASE_IDS[1],
        "phase_1_custody": phase1,
        "phase_2_custody": None,
        "prior_phase_custody": phase1,
        "supervisor_root_reopen_custody": supervisor_root,
        "terminal_publication_supervisor_preflight": preflight,
    }
    handoff_receipt = {
        "payload": handoff,
        "child_process_identity": validator,
        "content_digest": "3" * 64,
    }
    handoff_bytes = contract.canonical_json_bytes(handoff_receipt) + b"\n"
    handoff_binding = contract.build_artifact_binding(
        path=contract.RUNTIME_PATHS["phase_2_supervisor_handoff"],
        sha256=hashlib.sha256(handoff_bytes).hexdigest(),
        bytes_count=len(handoff_bytes),
        content_digest=handoff_receipt["content_digest"],
        rows=None,
    )
    monkeypatch.setattr(contract, "validate_phase_custody", lambda value: value)
    monkeypatch.setattr(
        contract,
        "validate_phase1_terminal_publication_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        contract, "_require_exact_live_process_identity", lambda value: value
    )
    monkeypatch.setattr(
        contract,
        "validate_terminal_publication_child_handoff_receipt",
        lambda value: value,
    )
    monkeypatch.setattr(
        contract, "validate_artifact_binding", lambda value, **_kwargs: value
    )
    target = {"terminal_attestor": attestor, "supervisor": phase1_supervisor}[
        live_role
    ]
    monkeypatch.setattr(
        contract,
        "_observe_historical_process_absence",
        lambda value: {
            "exact_historical_process_absent": value != target,
        },
    )
    monkeypatch.setattr(
        contract,
        "_active_v2_process_rows",
        lambda **_kwargs: [
            {
                "pid": terminal_supervisor["pid"],
                "start_time_ticks": terminal_supervisor["start_time_ticks"],
                "argv": terminal_supervisor["argv"],
            }
        ],
    )
    monkeypatch.setattr(
        contract,
        "_observe_other_v2_lock_rows",
        lambda **_kwargs: {"matching_rows": []},
    )
    receipt = contract.observe_phase2_prior_process_absence_custody(
        producer_phase_exit_custody=phase1,
        phase1_terminal_publication_receipt={"published_monotonic_ns": 0},
        terminal_publication_child_handoff_receipt=handoff_receipt,
        terminal_publication_child_handoff_binding=handoff_binding,
        validator_process_identity=validator,
    )
    assert receipt["historical_process_absence_observations"][live_role][
        "exact_historical_process_absent"
    ] is False
    assert receipt["pass"] is False
    with pytest.raises(contract.V2ContractError, match="absence schema drift"):
        contract.validate_phase2_prior_process_absence_custody(
            receipt, producer_phase_exit_custody=phase1
        )


def test_v2_orphaned_boundary_initiation_or_first_open_presence_is_rejected() -> None:
    path_by_key = dict(contract._PHASE1_BOUNDARY_ARTIFACT_PATHS)
    cases = (
        {
            "preexecution_receipt": False,
            "scientific_attempt_boundary_entry_receipt": True,
            "first_scientific_open_initiation_receipt": False,
            "first_scientific_open_receipt": False,
        },
        {
            "preexecution_receipt": True,
            "scientific_attempt_boundary_entry_receipt": False,
            "first_scientific_open_initiation_receipt": True,
            "first_scientific_open_receipt": False,
        },
        {
            "preexecution_receipt": True,
            "scientific_attempt_boundary_entry_receipt": True,
            "first_scientific_open_initiation_receipt": False,
            "first_scientific_open_receipt": True,
        },
    )
    orphan_keys = (
        "scientific_attempt_boundary_entry_receipt",
        "first_scientific_open_initiation_receipt",
        "first_scientific_open_receipt",
    )
    for presence, orphan_key in zip(cases, orphan_keys):
        with pytest.raises(contract.V2ContractError, match="implication drift"):
            contract._validate_phase1_orphaned_boundary_artifact_bindings(
                [], presence=presence
            )
        orphan_binding = contract.build_artifact_binding(
            path=path_by_key[orphan_key],
            sha256="0" * 64,
            bytes_count=1,
            content_digest=None,
            rows=None,
        )
        assert contract._validate_phase1_orphaned_boundary_artifact_bindings(
            [orphan_binding], presence=presence
        ) == [orphan_binding]
        forged_validated_receipt = contract.build_artifact_binding(
            path=path_by_key[orphan_key],
            sha256="0" * 64,
            bytes_count=1,
            content_digest="1" * 64,
            rows=None,
        )
        with pytest.raises(contract.V2ContractError, match="content digest"):
            contract._validate_phase1_orphaned_boundary_artifact_bindings(
                [forged_validated_receipt], presence=presence
            )


@pytest.mark.parametrize("fault_point", ("FIRST_FSTAT", "FCHMOD"))
def test_v2_generic_transaction_fault_restores_exact_leaf_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    parent_fd = os.open(
        tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    real_fstat = os.fstat
    real_fchmod = os.fchmod
    injected = False

    def is_staging_descriptor(descriptor: int) -> bool:
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            return False
        return Path(target).name.startswith(".leaf.json.staging-")

    def injected_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        if (
            fault_point == "FIRST_FSTAT"
            and not injected
            and is_staging_descriptor(descriptor)
        ):
            injected = True
            raise OSError("injected first fstat failure")
        return real_fstat(descriptor)

    def injected_fchmod(descriptor: int, mode: int) -> None:
        nonlocal injected
        if fault_point == "FCHMOD" and is_staging_descriptor(descriptor):
            injected = True
            raise OSError("injected fchmod failure")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(contract.os, "fstat", injected_fstat)
    monkeypatch.setattr(contract.os, "fchmod", injected_fchmod)
    try:
        with pytest.raises(OSError, match="injected"):
            contract._transactional_write_regular_no_replace_at(
                parent_fd=parent_fd,
                final_name="leaf.json",
                payload=b"payload\n",
                mode=0o600,
            )
        assert injected is True
        assert list(tmp_path.iterdir()) == []
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("fault_point", ("FIRST_FSTAT", "FCHMOD"))
def test_v2_tracked_transaction_fault_records_identity_and_rolls_back_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_point: str,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source_freeze = "0" * 40
    payloads = {
        path: f"{Path(path).name}\n".encode("utf-8")
        for path in contract.PRESENTATION_PUBLICATION_PATHS
    }
    bindings = {
        path: contract.build_artifact_binding(
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes_count=len(payload),
            content_digest=None,
            rows=None,
        )
        for path, payload in payloads.items()
    }
    candidate = contract.attach_self_digest(
        {
            "tracked_attempt_number": 1,
            "tracked_attempt_id": "v2-tracked-1-1",
            "source_freeze_commit": source_freeze,
            "candidate_bindings": bindings,
            "tracked_publication_retry_authority": None,
        }
    )
    candidate_bytes = contract.canonical_json_bytes(candidate) + b"\n"
    candidate_paths = contract.tracked_publication_attempt_runtime_paths(
        1, "v2-tracked-1-1"
    )
    candidate_binding = contract.build_artifact_binding(
        path=candidate_paths["candidate_custody"],
        sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        bytes_count=len(candidate_bytes),
        content_digest=candidate["content_digest"],
        rows=None,
    )
    monkeypatch.setattr(contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        contract,
        "validate_tracked_publication_candidate_custody",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        contract, "_require_live_terminal_publication_owner", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        contract,
        "read_v2_attempt_bytes_no_follow",
        lambda **_kwargs: candidate_bytes,
    )
    monkeypatch.setattr(
        contract,
        "write_tracked_publication_failure_custody_exclusive_fsync",
        lambda **_kwargs: {"path": "failure"},
    )

    def fake_git(_root: Path, arguments: tuple[str, ...]) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            return f"{source_freeze}\n".encode("ascii")
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return b""
        raise AssertionError(f"unexpected Git read: {arguments!r}")

    monkeypatch.setattr(contract, "_git_bytes", fake_git)
    real_fstat = os.fstat
    real_fchmod = os.fchmod
    first_leaf = Path(contract.PRESENTATION_PUBLICATION_PATHS[0]).name
    injected = False

    def is_first_tracked_descriptor(descriptor: int) -> bool:
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            return False
        return Path(target).name == first_leaf

    def injected_fstat(descriptor: int) -> os.stat_result:
        nonlocal injected
        if (
            fault_point == "FIRST_FSTAT"
            and not injected
            and is_first_tracked_descriptor(descriptor)
        ):
            injected = True
            raise OSError("injected tracked first fstat failure")
        return real_fstat(descriptor)

    def injected_fchmod(descriptor: int, mode: int) -> None:
        nonlocal injected
        if fault_point == "FCHMOD" and is_first_tracked_descriptor(descriptor):
            injected = True
            raise OSError("injected tracked fchmod failure")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(contract.os, "fstat", injected_fstat)
    monkeypatch.setattr(contract.os, "fchmod", injected_fchmod)
    failure, _binding = contract.stage_tracked_publication_payloads_transactional(
        candidate_custody=candidate,
        published_payloads=payloads,
        immutable_payload={"source_freeze_commit": source_freeze},
        independent_validation_receipt={},
        phase_1_custody={},
        phase_2_custody={},
        phase_3_custody={},
        candidate_custody_binding=candidate_binding,
        attempt_root_fd_custody={},
        reopened_attempt_root_fd_custody={},
        root_fd=0,
        repo_root=tmp_path,
    )
    assert injected is True
    assert failure["failure_stage"] == "PRECOMMIT_STAGING"
    assert failure["rollback_complete"] is True
    assert failure["all_tracked_paths_absent_after_rollback"] is True
    assert failure["worktree_clean_after_rollback"] is True
    assert failure["retry_scope"] == "TRACKED_STAGING_ONLY"
    assert failure["validated_payload_reuse_policy"] == (
        "SAME_VALIDATED_PAYLOAD_TRACKED_RETRY_ALLOWED"
    )
    assert failure["rollback_error_rows"] == []
    assert failure["status_observation"] == {
        "argv": [
            "git", "status", "--porcelain", "--untracked-files=all"
        ],
        "completed": True,
        "stdout_hex": "",
        "exception": None,
    }
    assert all(
        row["present"] is False
        and row["observed_stat"] is None
        and row["observation_error"] is None
        for row in failure["partial_path_inventory_rows"]
    )
    assert list(docs.iterdir()) == []
    if fault_point == "FIRST_FSTAT":
        assert failure["created_path_rows"] == []
        assert failure["rollback_rows"] == []
        observation = failure["unregistered_creation_observation"]
        assert observation is not None
        assert observation["path"] == (
            contract.PRESENTATION_PUBLICATION_PATHS[0]
        )
        assert observation["unlink_completed"] is True
        assert observation["parent_fsync_completed"] is True
        assert observation["descriptor_close_completed"] is True
        assert observation["path_absence_proved"] is True
        assert observation["cleanup_error_rows"] == []
        assert observation["cleanup_complete"] is True
    else:
        assert failure["unregistered_creation_observation"] is None
        assert len(failure["created_path_rows"]) == 1
        created = failure["created_path_rows"][0]
        assert created["path"] == (
            contract.PRESENTATION_PUBLICATION_PATHS[0]
        )
        assert created["inode"] > 0
        assert len(failure["rollback_rows"]) == 1
        rollback = failure["rollback_rows"][0]
        assert rollback["path"] == created["path"]
        assert rollback["expected_identity"] == {
            "device": created["device"], "inode": created["inode"]
        }
        assert rollback["retained_fd"] >= 0
        assert rollback["retained_fd_stat"] is not None
        assert rollback["path_stat_before_action"] is not None
        assert rollback["action"] == "UNLINKED_EXACT"
        assert rollback["unlink_completed"] is True
        assert rollback["path_absence_proved"] is True
        assert rollback["error_rows"] == []


@pytest.mark.parametrize(
    ("outcome", "expected_reason", "identity_unknown"),
    (
        ("SPAWN", "HEAD_PROCESS_SPAWN_FAILED", True),
        ("NONZERO", "HEAD_PROCESS_NONZERO_EXIT", True),
        ("MALFORMED", "MALFORMED_HEAD_OUTPUT", True),
        (
            "SOURCE_FREEZE",
            "HEAD_DID_NOT_ADVANCE_FROM_SOURCE_FREEZE",
            False,
        ),
    ),
)
def test_v2_postcommit_head_failures_are_persistable_and_never_recommit(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_reason: str,
    identity_unknown: bool,
) -> None:
    source_freeze = "0" * 40
    candidate = {
        "source_freeze_commit": source_freeze,
        "tracked_attempt_number": 1,
        "tracked_attempt_id": "v2-tracked-1-1",
    }
    commit_success = {"candidate_custody": candidate}
    commit_binding = {
        "path": "tracked_publication_attempts/v2-tracked-1-1/"
        "git_commit_success_custody.json"
    }
    monkeypatch.setattr(
        contract,
        "_validate_tracked_publication_candidate_structure",
        lambda value: value,
    )
    monkeypatch.setattr(
        contract, "validate_git_commit_success_custody", lambda value: value
    )
    monkeypatch.setattr(
        contract,
        "_git_commit_success_custody_binding",
        lambda _value: commit_binding,
    )
    monkeypatch.setattr(
        contract,
        "_validate_postcommit_observation_candidate_chain",
        lambda **_kwargs: None,
    )
    if outcome == "SPAWN":
        monkeypatch.setattr(
            contract.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected HEAD spawn failure")
            ),
        )
    else:
        completed = {
            "NONZERO": SimpleNamespace(stdout=b"", stderr=b"failed\n", returncode=3),
            "MALFORMED": SimpleNamespace(
                stdout=b"not-an-object-id\n", stderr=b"", returncode=0
            ),
            "SOURCE_FREEZE": SimpleNamespace(
                stdout=f"{source_freeze}\n".encode("ascii"),
                stderr=b"",
                returncode=0,
            ),
        }[outcome]
        monkeypatch.setattr(
            contract.subprocess, "run", lambda *_args, **_kwargs: completed
        )
    failure, _stdout, _stderr = (
        contract.observe_postcommit_head_after_successful_commit(
            candidate_custody=candidate,
            git_commit_success_custody=commit_success,
            git_commit_success_custody_binding=commit_binding,
            repo_root=contract.REPO_ROOT,
        )
    )
    assert failure["failure_stage"] == "POSTCOMMIT_HEAD_OBSERVATION"
    assert failure["head_mismatch_reason"] == expected_reason
    assert failure["result_commit_identity_outcome_unknown"] is identity_unknown
    assert failure["git_commit_rerun_performed"] is False
    assert failure["git_commit_rerun_authorized"] is False
    assert failure["retry_scope"] == "POSTCOMMIT_OBSERVATION_ONLY"
    assert contract.validate_tracked_publication_failure_custody(failure) == failure


def test_v2_postcommit_retry_scope_is_absorbing_and_rejects_chain_splicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postcommit = {
        "failure_stage": "POSTCOMMIT_HEAD_OBSERVATION",
        "retry_scope": "POSTCOMMIT_OBSERVATION_ONLY",
        "commit": "original",
    }
    downgraded = {
        "failure_stage": "GIT_COMMIT",
        "retry_scope": "GIT_COMMIT_ONLY",
        "commit": "original",
    }
    with pytest.raises(contract.V2ContractError, match="regressed"):
        contract._validate_postcommit_retry_chain_absorbing(
            [postcommit, downgraded]
        )
    monkeypatch.setattr(
        contract,
        "_postcommit_failure_commit_success_pair",
        lambda failure: (failure["commit"], {"path": failure["commit"]}),
    )
    spliced = {
        "failure_stage": "POSTCOMMIT_VALIDATION",
        "retry_scope": "POSTCOMMIT_OBSERVATION_ONLY",
        "commit": "different",
    }
    with pytest.raises(contract.V2ContractError, match="spliced"):
        contract._validate_postcommit_retry_chain_absorbing(
            [postcommit, spliced]
        )


def test_v2_postcommit_retry_never_restages_readds_or_recommits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retry = {"retry_scope": "POSTCOMMIT_OBSERVATION_ONLY"}
    monkeypatch.setattr(
        contract,
        "validate_tracked_publication_retry_authority",
        lambda value: value,
    )
    with pytest.raises(contract.V2ContractError, match="cannot stage"):
        contract._reject_postcommit_scoped_git_mutation(
            {"tracked_publication_retry_authority": retry},
            operation="stage tracked paths",
        )
    source = _function_source("_publish_terminal_tracked_result")
    branch = source.index(
        "if reopened['resume_boundary'] == 'POSTCOMMIT_OBSERVATION_ONLY'"
    )
    observation_return = source.index(
        "return _observe_and_validate_postcommit", branch
    )
    staging = source.index("staging, staging_binding =", observation_return)
    assert branch < observation_return < staging


def test_v2_rc_zero_commit_is_durable_before_any_head_observation() -> None:
    commit_calls = _ordered_call_names("_commit_tracked_publication")
    assert commit_calls.index("build_git_commit_success_custody") < (
        commit_calls.index("write_git_commit_success_custody_exclusive_fsync")
    ) < commit_calls.index("_observe_and_validate_postcommit")
    postcommit_calls = _ordered_call_names("_observe_and_validate_postcommit")
    assert postcommit_calls.index(
        "observe_postcommit_head_after_successful_commit"
    ) < postcommit_calls.index(
        "write_postcommit_head_observation_exclusive_fsync"
    ) < postcommit_calls.index("validate_presentation_result_commit")
    assert "git_commit_success_custody" in _function_source(
        "_observe_and_validate_postcommit"
    )


def test_v2_claim_critical_read_custody_is_stable_after_five_sibling_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    docs = repo_root / "docs"
    docs.mkdir(parents=True)
    frozen = docs / "frozen.json"
    frozen_payload = b'{"frozen":true}\n'
    frozen.write_bytes(frozen_payload)
    monkeypatch.setattr(contract, "REPO_ROOT", repo_root)
    expected_sha256 = hashlib.sha256(frozen_payload).hexdigest()

    payload_before, observation_before = (
        contract._read_claim_critical_regular_file_anchored_no_follow(
            repo_root=repo_root,
            relative_path="docs/frozen.json",
            expected_sha256=expected_sha256,
            expected_bytes=len(frozen_payload),
        )
    )
    docs_stat_before = os.stat(docs, follow_symlinks=False)
    for index in range(5):
        (docs / f"authority-{index}.json").write_bytes(
            f'{{"authority":{index}}}\n'.encode("ascii")
        )
    os.utime(
        docs,
        ns=(
            docs_stat_before.st_atime_ns,
            docs_stat_before.st_mtime_ns + 10_000_000_000,
        ),
        follow_symlinks=False,
    )
    docs_stat_after = os.stat(docs, follow_symlinks=False)
    assert docs_stat_after.st_mtime_ns != docs_stat_before.st_mtime_ns

    payload_after, observation_after = (
        contract._read_claim_critical_regular_file_anchored_no_follow(
            repo_root=repo_root,
            relative_path="docs/frozen.json",
            expected_sha256=expected_sha256,
            expected_bytes=len(frozen_payload),
        )
    )
    assert payload_after == payload_before == frozen_payload
    assert observation_after == observation_before
    assert contract._validate_claim_critical_read_observation(
        observation_after,
        expected_path="docs/frozen.json",
    ) == observation_after
    expected_identity_keys = {
        "device",
        "inode",
        "file_type",
        "mode",
        "uid",
        "gid",
    }
    for row in observation_after["parent_component_rows"]:
        assert set(row["stable_identity"]) == expected_identity_keys
        assert row["directory_nlink_at_least_two_at_observation"] is True
