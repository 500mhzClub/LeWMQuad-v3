"""Narrow, versioned correction for the frozen PREEXECUTION diagnostic.

The fa80 forensic contract is historical and byte-exact.  This additive module
changes only the terminal source-closure binding validator from the frozen
four-key shape to the exact row-counted five-key shape.  It delegates the
launcher, wrapper, streams, serializer, synthetic fixtures, startup ledger,
cleanup, and read guard to the frozen implementation.
"""

from __future__ import annotations

import ast
import argparse
import base64
import copy
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lewm.safety import (
    plan_aware_monotone_jepa_cost_v1_forensic_contract as BASE,
)


class ForensicCorrectionContractError(RuntimeError):
    """Raised when the versioned technical correction drifts."""


class CorrectionPrecommitStagingTransactionError(
    ForensicCorrectionContractError
):
    """Carries the fact that the exact result staging transaction began."""

    tracked_publication_attempted = True


SOURCE_FORENSIC_FREEZE_COMMIT = "fa80f01599e99a5fd5721481f00ab9d177e2f00f"
CORRECTION_FREEZE_COMMIT_SUBJECT = (
    "Correct strict row-counted closure binding validation"
)
CORRECTION_RESULT_COMMIT_SUBJECT = (
    "Complete corrected plan-aware JEPA PREEXECUTION diagnostic"
)
ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID = (
    "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1"
)
PREEXECUTION_FORENSIC_CORRECTION_FREEZE_SUBCOMMAND = (
    "freeze-preexecution-forensic-correction-1"
)
PREEXECUTION_DIAGNOSTIC_CORRECTION_SUBCOMMAND = (
    "diagnose-preexecution-correction-1"
)
PREEXECUTION_DIAGNOSTIC_CORRECTION_CHILD_SUBCOMMAND = (
    "diagnose-preexecution-correction-1-child"
)
PREEXECUTION_DIAGNOSTIC_CORRECTION_FINALIZER_SUBCOMMAND = (
    "finalize-preexecution-forensic-correction-1"
)
PREEXECUTION_DIAGNOSTIC_CORRECTION_CHECKER_SUBCOMMAND = (
    "check-preexecution-forensic-correction-1"
)
PREEXECUTION_DIAGNOSTIC_CORRECTION_COORDINATOR_SUBCOMMAND = (
    "coordinate-preexecution-forensic-correction-1-terminal-phases"
)
PREEXECUTION_FORENSIC_CORRECTION_RESULT_COMMIT_VALIDATION_SUBCOMMAND = (
    "validate-preexecution-forensic-correction-1-result-commit"
)
CORRECTION_FINALIZER_ROLE = (
    "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_FINALIZER"
)
CORRECTION_CHECKER_ROLE = "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_CHECKER"
CORRECTION_REPO_ROOT = Path("/home/andrewknowles/Workspace/LeWMQuad-v3")
CORRECTION_EVALUATOR_SCRIPT = (
    CORRECTION_REPO_ROOT
    / "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
)
CORRECTION_WRAPPER_SCRIPT = (
    CORRECTION_REPO_ROOT / BASE.PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH
)
CORRECTION_CONTRACT_MODULE = (
    "lewm.safety."
    "plan_aware_monotone_jepa_cost_v1_forensic_correction_1_contract"
)
CORRECTION_CONTRACT_SCRIPT = (
    CORRECTION_REPO_ROOT
    / "lewm/safety/"
    "plan_aware_monotone_jepa_cost_v1_forensic_correction_1_contract.py"
)
FAILED_PREEXECUTION_DIAGNOSTIC_ROOT = BASE.PREEXECUTION_DIAGNOSTIC_ROOT
PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_correction_1"
)
CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_"
    "correction_1_terminal_failure_custody"
)
CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH = (
    CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT
    / "terminal_phase_failure_custody.json"
)
FAILED_DIAGNOSTIC_ROOT_AUTHORITY = {
    "root": str(FAILED_PREEXECUTION_DIAGNOSTIC_ROOT),
    "file_count": 87,
    "total_file_bytes": 269_462,
    "manifest_sha256": (
        "c2b9c4dc241eead083a8eb2a8eda074cb86e2b40a1a78d0a8f289482884c1385"
    ),
    "manifest_row_keys": ["path", "sha256", "bytes"],
    "technical_only": True,
    "immutable_and_nonreusable": True,
}
_FAILED_MAIN_FILES = (
    "receipts/child_exception.json", "receipts/command.json",
    "receipts/environment.json", "receipts/forensic_freeze_custody.json",
    "receipts/heartbeat.jsonl", "receipts/invocation.json",
    "receipts/last_stage.json", "receipts/os_evidence.json",
    "receipts/preexecution_only.json", "receipts/read_guard_events.jsonl",
    "receipts/read_guard_manifest.json", "receipts/startup_stage.jsonl",
    "receipts/synthetic_results.json", "streams/child.stderr",
    "streams/child.stdout", "streams/child.traceback",
)
_FAILED_SYNTHETIC_FILES = (
    "receipts/child_exception.json", "receipts/command.json",
    "receipts/environment.json", "receipts/heartbeat.jsonl",
    "receipts/last_stage.json", "receipts/read_guard_events.jsonl",
    "receipts/read_guard_manifest.json", "receipts/startup_stage.jsonl",
    "receipts/synthetic_technical_lifecycle.jsonl", "streams/child.stderr",
    "streams/child.stdout", "streams/child.traceback",
)
FAILED_TECHNICAL_FILE_PATHS = tuple(sorted((
    *_FAILED_MAIN_FILES,
    *(
        f"synthetic/{fixture}/{relative}"
        for fixture in BASE.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
        for relative in _FAILED_SYNTHETIC_FILES
        if not (
            fixture == "MISSING_PATH"
            and relative == "receipts/startup_stage.jsonl"
        )
    ),
)))
FAILED_TECHNICAL_DIRECTORY_PATHS = tuple(sorted((
    "receipts", "streams", "synthetic",
    *(
        (f"synthetic/{fixture}" if not suffix else f"synthetic/{fixture}/{suffix}")
        for fixture in BASE.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
        for suffix in ("", "receipts", "streams")
    ),
)))
if len(FAILED_TECHNICAL_FILE_PATHS) != 87:
    raise RuntimeError("failed technical file-path authority drift")
PRESERVED_LINEAGE = {
    "scientific_source": "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0",
    "scientific_freeze": "9c1c3adcfb8382c33e8da8895dc345e006e92e43",
    "correction_1": "14625958c0fcc21af05b33fd30c6cc2fc8537745",
    "correction_2_base": "1c18af8c3c45f9b14992362f7e50a35b651c6997",
    "forensic_freeze": SOURCE_FORENSIC_FREEZE_COMMIT,
}
FROZEN_FORENSIC_CONTRACT_PATH = (
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_forensic_contract.py"
)
FROZEN_FORENSIC_CONTRACT_SHA256 = (
    "a57c2d01d309cc28cbdd08c51cac55615c1e97370b5f4b38ab715ec123e578e6"
)
FROZEN_FORENSIC_CONTRACT_BLOB_OID = "c2de82b82fc413253007f9176cacdfe2459bea17"
FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256 = (
    "1111ad326d17ea14f019b4bbbc1c62bb5622edbebbedbc751313366f2c2aa9ab"
)
COMMITTED_ROOT_CAUSE_PROOF_CONTENT_DIGEST = (
    "e143a8b6762dc9c5a8f773d271e0fb17da11814e46b3dc595f8ca4cb35d01d31"
)
FUTURE_V2_SPEC_ONLY_COMMAND = {
    "argv": [
        "/home/andrewknowles/TinyQuadJEPA/bin/python",
        "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
        "evaluate_plan_aware_monotone_jepa_cost_v2.py",
        "execute",
    ],
    "cwd": str(CORRECTION_REPO_ROOT),
    "shell_rendering": (
        "/home/andrewknowles/TinyQuadJEPA/bin/python "
        "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
        "evaluate_plan_aware_monotone_jepa_cost_v2.py execute"
    ),
    "status": "REQUIRED_FUTURE_V2_IMPLEMENTATION_NOT_PRESENT_OR_RUNNABLE",
    "specification_only": True,
    "execution_authorized": False,
    "do_not_execute": True,
    "separate_explicit_scientific_authority_required": True,
}

CORRECTION_SCIENTIFIC_COUNTER_FIELDS = (
    "fit_states_opened",
    "fit_rows_opened",
    "calibration_states_opened",
    "calibration_rows_opened",
    "heldout_states_opened",
    "heldout_rows_opened",
    "route_labels_opened",
    "true_future_latents_opened",
    "predicted_latents_opened",
    "predictor_checkpoints_opened",
    "route_cost_checkpoints_opened",
    "model_initializations",
    "optimizer_updates",
    "training_epochs",
    "scientific_score_rows",
    "candidate_selections",
    "scientific_metrics",
    "scientific_payloads",
)
CORRECTION_ZERO_SCIENTIFIC_COUNTERS = {
    key: 0 for key in CORRECTION_SCIENTIFIC_COUNTER_FIELDS
}
CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING = {
    "technical_diagnostic_authorized": 1,
    "technical_diagnostic_consumed": 1,
    "technical_diagnostic_remaining": 0,
    "automatic_retry": False,
    "further_correction_or_diagnostic_authorized": False,
    "scientific_attempt_consumed": False,
}
CORRECTION_TECHNICAL_DIAGNOSTIC_AUTHORITY_ACCOUNTING = {
    "technical_diagnostic_authorized": 1,
    "technical_diagnostic_consumed": 0,
    "technical_diagnostic_remaining": 1,
    "automatic_retry": False,
    "further_correction_or_diagnostic_authorized": False,
    "scientific_attempt_consumed": False,
}

TRACKED_CORRECTION_AMENDMENT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_amendment.json"
)
TRACKED_CORRECTION_FIXTURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_evaluator_fixture.json"
)
TRACKED_CORRECTION_ALLOW_LIST_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_allow_list.json"
)
TRACKED_CORRECTION_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_source_closure.json"
)
TRACKED_CORRECTION_RESULT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_result.json"
)
TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_diagnostic_custody.json"
)
TRACKED_CORRECTION_FAILED_ROOT_CUSTODY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_failed_diagnostic_custody.json"
)
TRACKED_CORRECTION_FIXTURE_RESULTS_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_fixture_results.json"
)
TRACKED_CORRECTION_FINAL_NAMESPACE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_final_namespace_inventory.json"
)
TRACKED_CORRECTION_FINALIZER_CUSTODY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_finalizer_custody.json"
)
TRACKED_CORRECTION_CHECKER_CUSTODY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_"
    "forensic_correction_1_checker_custody.json"
)
CORRECTION_RESULT_PATHS = (
    str(TRACKED_CORRECTION_RESULT_PATH),
    str(TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH),
    str(TRACKED_CORRECTION_FAILED_ROOT_CUSTODY_PATH),
    str(TRACKED_CORRECTION_FIXTURE_RESULTS_PATH),
    str(TRACKED_CORRECTION_FINAL_NAMESPACE_PATH),
    str(TRACKED_CORRECTION_FINALIZER_CUSTODY_PATH),
    str(TRACKED_CORRECTION_CHECKER_CUSTODY_PATH),
    str(BASE.TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH),
    str(BASE.TRACKED_CONDITIONAL_V2_SPEC_PATH),
)
CORRECTION_CORE_RESULT_PATHS = tuple(
    path
    for path in CORRECTION_RESULT_PATHS
    if path not in {
        str(TRACKED_CORRECTION_FINALIZER_CUSTODY_PATH),
        str(TRACKED_CORRECTION_CHECKER_CUSTODY_PATH),
    }
)
CORRECTION_CODE_AND_TEST_PATHS = (
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_forensic_"
    "correction_1_contract.py",
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "scripts/run_plan_aware_monotone_jepa_cost_v1_"
    "preexecution_diagnostic_child.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_v1_"
    "forensic_correction_1_contract.py",
    "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1_forensics.py",
)
CORRECTION_GENERATED_AUTHORITY_PATHS = (
    str(TRACKED_CORRECTION_AMENDMENT_PATH),
    str(TRACKED_CORRECTION_FIXTURE_PATH),
    str(TRACKED_CORRECTION_ALLOW_LIST_PATH),
    str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
)
CORRECTION_REQUIRED_CHANGED_PATHS = (
    *CORRECTION_CODE_AND_TEST_PATHS,
    *CORRECTION_GENERATED_AUTHORITY_PATHS,
)
CORRECTION_SOURCE_CLOSURE_PATHS = (
    str(BASE.TRACKED_FORENSIC_SOURCE_CLOSURE_PATH),
    *CORRECTION_CODE_AND_TEST_PATHS,
    str(TRACKED_CORRECTION_AMENDMENT_PATH),
    str(TRACKED_CORRECTION_FIXTURE_PATH),
    str(TRACKED_CORRECTION_ALLOW_LIST_PATH),
)
CORRECTION_SOURCE_CLOSURE_ROW_COUNT = 9
if len(CORRECTION_SOURCE_CLOSURE_PATHS) != CORRECTION_SOURCE_CLOSURE_ROW_COUNT:
    raise RuntimeError("correction source-closure row authority drift")

CORRECTED_CHILD_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_child_result."
    "row_counted_correction_1.v1"
)
CORRECTION_DIAGNOSTIC_CUSTODY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_diagnostic_custody."
    "row_counted_correction_1.v1"
)
CORRECTION_FREEZE_CUSTODY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_correction_1_freeze_custody.v1"
)
CORRECTION_FREEZE_PREPARATION_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_freeze_preparation.v1"
)
CORRECTION_AUTHORITY_WRITE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_authority_write.v1"
)
CORRECTION_SOURCE_CLOSURE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_correction_1_source_closure.v1"
)
CORRECTION_FINAL_NAMESPACE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_final_namespace."
    "row_counted_correction_1.v1"
)
CORRECTION_RUNTIME_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_result."
    "row_counted_correction_1.v1"
)
CORRECTION_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_correction_1_result.v1"
)
CORRECTION_FINALIZER_PROSPECTIVE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_finalizer_prospective.v1"
)
CORRECTION_EXTERNAL_CHECKER_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_external_checker.v1"
)
CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_phase_exit_custody.v1"
)
CORRECTION_PHASE_WRAPPER_ENVELOPE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_phase_wrapper_envelope.v1"
)
CORRECTION_PHASE_ENVIRONMENT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_phase_environment.v1"
)
CORRECTION_TERMINAL_COORDINATOR_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_terminal_coordinator.v1"
)
CORRECTION_FRESH_TECHNICAL_MANIFEST_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_fresh_technical_manifest.v1"
)
CORRECTION_ROLE_SCAN_CUSTODY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1."
    "forensic_correction_1_role_scan_custody.v1"
)
CORRECTION_V2_SPECIFICATION_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v2."
    "technical_specification.row_counted_correction_1.v1"
)
CORRECTION_PHASE_IDENTITY_TIMEOUT_SECONDS = 5.0
CORRECTION_PHASE_RUNTIME_TIMEOUT_SECONDS = 30.0
CORRECTION_PHASE_TERM_GRACE_SECONDS = 5.0
CORRECTION_PHASE_KILL_GRACE_SECONDS = 5.0
CORRECTION_PHASE_STAGE_IDS = {
    "FINALIZER": (
        "STARTED", "LOAD_TERMINAL_BUNDLE", "BUILD_PROSPECTIVE_BYTES",
        "COMPLETE",
    ),
    "CHECKER": (
        "STARTED", "LOAD_TERMINAL_BUNDLE", "REBUILD_PROSPECTIVE_BYTES",
        "COMPLETE",
    ),
}
ROW_COUNTED_FIXTURE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.row_counted_binding_fixtures.v1"
)
ROW_COUNTED_TERMINAL_RECEIPT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.row_counted_terminal_binding.v1"
)
_LOWERCASE_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _authority_bytes(value: Mapping[str, Any]) -> bytes:
    return BASE.canonical_json_bytes(value) + b"\n"


def _runtime_json_binding(key: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if key not in BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS:
        raise ForensicCorrectionContractError("unknown BASE runtime key")
    BASE.validate_self_digest(value)
    payload = _authority_bytes(value)
    return BASE._artifact_binding(
        BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
        payload,
        content_digest=value["content_digest"],
    )


def _future_v2_script_absent() -> bool:
    path = Path(FUTURE_V2_SPEC_ONLY_COMMAND["argv"][1])
    parent_fd = BASE._open_absolute_directory_no_follow(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(parent_fd)


def _anchored_leaf_absent(path: Path) -> bool:
    """Prove lexical leaf absence without following any parent or leaf link."""

    absolute = path.absolute()
    try:
        parent_fd = BASE._open_absolute_directory_no_follow(absolute.parent)
    except FileNotFoundError:
        return True
    try:
        try:
            os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(parent_fd)


def _validate_correction_freeze_commit_git_custody(
    repo_root: Path, freeze_commit: str
) -> dict[str, Any]:
    if not isinstance(freeze_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", freeze_commit
    ):
        raise ForensicCorrectionContractError(
            "correction freeze commit identity drift"
        )
    parents = BASE._git_text(
        repo_root, ("rev-list", "--parents", "-n", "1", freeze_commit)
    ).split()
    subject = BASE._git_text(
        repo_root, ("show", "-s", "--format=%s", freeze_commit)
    )
    changed = sorted(
        row
        for row in BASE._git_text(
            repo_root,
            (
                "diff",
                "--name-only",
                SOURCE_FORENSIC_FREEZE_COMMIT,
                freeze_commit,
            ),
        ).splitlines()
        if row
    )
    if (
        len(parents) != 2
        or parents[1] != SOURCE_FORENSIC_FREEZE_COMMIT
        or subject != CORRECTION_FREEZE_COMMIT_SUBJECT
        or changed != sorted(CORRECTION_REQUIRED_CHANGED_PATHS)
    ):
        raise ForensicCorrectionContractError(
            "correction freeze commit Git custody drift"
        )
    return {
        "commit": freeze_commit,
        "sole_parent": SOURCE_FORENSIC_FREEZE_COMMIT,
        "subject": subject,
        "changed_paths": changed,
        "pass": True,
    }


def expected_correction_launcher_argv() -> list[str]:
    return [
        str(BASE.FORENSIC_INTERPRETER),
        str(CORRECTION_EVALUATOR_SCRIPT),
        PREEXECUTION_DIAGNOSTIC_CORRECTION_SUBCOMMAND,
    ]


def expected_correction_child_inner_argv(
    *, launcher_pid: int, launcher_start_time_ticks: int
) -> list[str]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (launcher_pid, launcher_start_time_ticks)
    ):
        raise ForensicCorrectionContractError("correction launcher identity drift")
    return [
        str(CORRECTION_EVALUATOR_SCRIPT),
        PREEXECUTION_DIAGNOSTIC_CORRECTION_CHILD_SUBCOMMAND,
        "--launcher-pid",
        str(launcher_pid),
        "--launcher-start-time-ticks",
        str(launcher_start_time_ticks),
        "--diagnostic-root",
        str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
    ]


def expected_correction_finalizer_inner_argv() -> list[str]:
    return [
        str(CORRECTION_EVALUATOR_SCRIPT),
        PREEXECUTION_DIAGNOSTIC_CORRECTION_FINALIZER_SUBCOMMAND,
    ]


def expected_correction_finalizer_outer_argv() -> list[str]:
    return [
        str(BASE.FORENSIC_INTERPRETER),
        *BASE.PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS,
        str(CORRECTION_WRAPPER_SCRIPT),
        "--mode",
        PREEXECUTION_DIAGNOSTIC_CORRECTION_FINALIZER_SUBCOMMAND,
        "--diagnostic-root",
        str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
    ]


def expected_correction_finalizer_argv() -> list[str]:
    """Return the exact root-invoked wrapper argv, never the inner dispatch."""

    return expected_correction_finalizer_outer_argv()


def expected_correction_checker_inner_argv() -> list[str]:
    return [
        str(CORRECTION_EVALUATOR_SCRIPT),
        PREEXECUTION_DIAGNOSTIC_CORRECTION_CHECKER_SUBCOMMAND,
    ]


def expected_correction_checker_outer_argv() -> list[str]:
    return [
        str(BASE.FORENSIC_INTERPRETER),
        *BASE.PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS,
        str(CORRECTION_WRAPPER_SCRIPT),
        "--mode",
        PREEXECUTION_DIAGNOSTIC_CORRECTION_CHECKER_SUBCOMMAND,
        "--diagnostic-root",
        str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
    ]


def expected_correction_checker_argv() -> list[str]:
    """Return the exact root-invoked wrapper argv, never the inner dispatch."""

    return expected_correction_checker_outer_argv()


def expected_correction_terminal_coordinator_argv() -> list[str]:
    """Return the exact root-owned, out-of-producer-scan coordinator argv."""

    return [
        str(BASE.FORENSIC_INTERPRETER),
        *BASE.PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS,
        "-m",
        CORRECTION_CONTRACT_MODULE,
        PREEXECUTION_DIAGNOSTIC_CORRECTION_COORDINATOR_SUBCOMMAND,
    ]


def expected_correction_result_commit_validation_argv() -> list[str]:
    """Return the exact postcommit-only replay command."""

    return [
        str(BASE.FORENSIC_INTERPRETER),
        *BASE.PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS,
        "-m",
        CORRECTION_CONTRACT_MODULE,
        PREEXECUTION_FORENSIC_CORRECTION_RESULT_COMMIT_VALIDATION_SUBCOMMAND,
    ]


def _correction_proc_argv(pid: int) -> list[str]:
    payload = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(b"\0")
        if item
    ]


def _correction_proc_stat(pid: int) -> tuple[int, int, int]:
    payload = (Path("/proc") / str(pid) / "stat").read_text(
        encoding="utf-8"
    )
    try:
        _prefix, suffix = payload.rsplit(")", 1)
        fields = suffix.strip().split()
        return int(fields[1]), int(fields[2]), int(fields[19])
    except (ValueError, IndexError) as exc:
        raise ForensicCorrectionContractError(
            "malformed correction /proc stat record"
        ) from exc


def _observe_correction_contract_process_identity(
    pid: int,
) -> dict[str, Any]:
    argv = _correction_proc_argv(pid)
    _ppid, process_group_id, start_time_ticks = _correction_proc_stat(pid)
    executable = str(
        (Path("/proc") / str(pid) / "exe").resolve(strict=True)
    )
    executable_exact = (
        Path(executable).resolve() == BASE.FORENSIC_INTERPRETER_RESOLVED
    )
    if executable_exact and argv == expected_correction_terminal_coordinator_argv():
        role = "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_ROOT_COORDINATOR"
    elif executable_exact and argv == expected_correction_result_commit_validation_argv():
        role = "PREEXECUTION_FORENSIC_CORRECTION_1_POSTCOMMIT_REPLAY"
    else:
        role = "INVALID_CORRECTION_CONTRACT_ARGV"
    return BASE.validate_process_identity(
        {
            "pid": pid,
            "process_group_id": process_group_id,
            "start_time_ticks": start_time_ticks,
            "argv": argv,
            "argv_sha256": BASE.canonical_json_sha256(argv),
            "executable": executable,
            "role": role,
        }
    )


def _observe_correction_terminal_coordinator_identity(
    pid: int, *, require_exact: bool
) -> dict[str, Any]:
    identity = _observe_correction_contract_process_identity(pid)
    if (
        require_exact
        and identity["role"]
        != "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_ROOT_COORDINATOR"
    ):
        raise ForensicCorrectionContractError(
            "terminal coordinator exact process identity drift"
        )
    return identity


def validate_correction_result_commit_validation_identity(
    value: Mapping[str, Any], *, require_live: bool
) -> dict[str, Any]:
    identity = BASE.validate_process_identity(value)
    if (
        identity["role"]
        != "PREEXECUTION_FORENSIC_CORRECTION_1_POSTCOMMIT_REPLAY"
        or identity["argv"]
        != expected_correction_result_commit_validation_argv()
        or Path(identity["executable"]).resolve()
        != BASE.FORENSIC_INTERPRETER_RESOLVED
    ):
        raise ForensicCorrectionContractError(
            "correction postcommit replay identity drift"
        )
    if require_live:
        observed = _observe_correction_contract_process_identity(
            identity["pid"]
        )
        if observed != identity:
            raise ForensicCorrectionContractError(
                "correction postcommit replay live identity drift"
            )
    return copy.deepcopy(identity)


def validate_correction_terminal_coordinator_identity(
    value: Mapping[str, Any], *, require_live: bool
) -> dict[str, Any]:
    identity = BASE.validate_process_identity(value)
    if (
        identity["role"]
        != "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_ROOT_COORDINATOR"
        or identity["argv"] != expected_correction_terminal_coordinator_argv()
        or Path(identity["executable"]).resolve()
        != BASE.FORENSIC_INTERPRETER_RESOLVED
    ):
        raise ForensicCorrectionContractError(
            "terminal coordinator identity receipt drift"
        )
    if require_live:
        observed = _observe_correction_terminal_coordinator_identity(
            identity["pid"], require_exact=True
        )
        if observed != identity:
            raise ForensicCorrectionContractError(
                "terminal coordinator live identity drift"
            )
    return copy.deepcopy(identity)


def require_live_correction_terminal_coordinator(
    *, pid: int, start_time_ticks: int
) -> dict[str, Any]:
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_time_ticks, bool)
        or not isinstance(start_time_ticks, int)
        or start_time_ticks <= 0
    ):
        raise ForensicCorrectionContractError(
            "terminal coordinator PID/start input drift"
        )
    identity = _observe_correction_terminal_coordinator_identity(
        pid, require_exact=True
    )
    if identity["start_time_ticks"] != start_time_ticks:
        raise ForensicCorrectionContractError(
            "terminal coordinator PID reuse custody drift"
        )
    return identity


def _observe_correction_process_cwd(identity: Mapping[str, Any]) -> str:
    validated = BASE.validate_process_identity(identity)
    try:
        observed = str(
            (Path("/proc") / str(validated["pid"]) / "cwd").resolve(
                strict=True
            )
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as exc:
        raise ForensicCorrectionContractError(
            "correction process cwd custody unavailable"
        ) from exc
    if observed != str(CORRECTION_REPO_ROOT):
        raise ForensicCorrectionContractError(
            "correction process cwd drift"
        )
    return observed


def _correction_argv_references_contract_script(
    argv: Sequence[str], *, process_cwd: Path
) -> bool:
    target = CORRECTION_CONTRACT_SCRIPT.resolve(strict=True)
    for item in argv:
        if not isinstance(item, str) or not item or "\x00" in item:
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = process_cwd / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, PermissionError, OSError, RuntimeError):
            continue
        if resolved == target:
            return True
    return False


def _correction_argv_has_direct_contract_reference(
    argv: Sequence[str],
) -> bool:
    relative_contract_script = str(
        CORRECTION_CONTRACT_SCRIPT.relative_to(CORRECTION_REPO_ROOT)
    )
    return any(
        token in argv
        for token in (
            CORRECTION_CONTRACT_MODULE,
            str(CORRECTION_CONTRACT_SCRIPT),
            relative_contract_script,
            PREEXECUTION_DIAGNOSTIC_CORRECTION_COORDINATOR_SUBCOMMAND,
            PREEXECUTION_FORENSIC_CORRECTION_RESULT_COMMIT_VALIDATION_SUBCOMMAND,
        )
    )


def _active_correction_terminal_coordinators(
    *, exclude_pids: Iterable[int] = ()
) -> list[dict[str, Any]]:
    excluded = {int(pid) for pid in exclude_pids}
    output: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        pid = int(entry.name)
        try:
            argv = _correction_proc_argv(pid)
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
        ):
            continue
        direct_reference = _correction_argv_has_direct_contract_reference(
            argv
        )
        path_reference = False
        if not direct_reference:
            try:
                process_cwd = (
                    Path("/proc") / str(pid) / "cwd"
                ).resolve(strict=True)
                path_reference = _correction_argv_references_contract_script(
                    argv, process_cwd=process_cwd
                )
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                OSError,
                RuntimeError,
            ):
                path_reference = False
        if not direct_reference and not path_reference:
            continue
        try:
            output.append(
                _observe_correction_contract_process_identity(pid)
            )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
            ForensicCorrectionContractError,
        ):
            continue
    return sorted(output, key=lambda row: row["pid"])


_CORRECTION_ROLE_SCAN_GATES = {
    "POSTCOMMIT_BEFORE_RESULT_READS",
    "POSTCOMMIT_AFTER_RESULT_READS",
    "POSTCOMMIT_REPLAY_FAILURE",
    "COORDINATOR_BOOTSTRAP_FAILURE_AFTER_ROLLBACK",
    "BOOTSTRAP_CUSTODY_DOUBLE_FAILURE_AFTER_ROLLBACK",
    "FAILURE_PERSISTENCE_ERROR",
    "TERMINAL_COORDINATOR_RECEIPT",
}


def _observe_correction_role_scan_custody(
    repo_root: str | Path,
    *,
    gate: str,
    excluded_pids: Iterable[int],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    raw_excluded = list(excluded_pids)
    if any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in raw_excluded
    ):
        raise ForensicCorrectionContractError(
            "correction role-scan excluded PID drift"
        )
    excluded = sorted(set(raw_excluded))
    if (
        root != CORRECTION_REPO_ROOT
        or gate not in _CORRECTION_ROLE_SCAN_GATES
    ):
        raise ForensicCorrectionContractError(
            "correction role-scan input drift"
        )
    base_matches = BASE._active_forensic_or_scientific_processes(
        root, exclude_pids=set(excluded)
    )
    correction_matches = _active_correction_terminal_coordinators(
        exclude_pids=set(excluded)
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_ROLE_SCAN_CUSTODY_SCHEMA,
            "gate": gate,
            "repo_root": str(root),
            "excluded_pids": excluded,
            "base_forensic_or_scientific_process_matches": base_matches,
            "correction_contract_process_matches": correction_matches,
            "nonexcluded_match_count": (
                len(base_matches) + len(correction_matches)
            ),
            "all_nonexcluded_roles_zero": not (
                base_matches or correction_matches
            ),
            "pass_meaning": (
                "ROLE_SCAN_CUSTODY_COMPLETE_NOT_A_ZERO_CLAIM"
            ),
            "pass": True,
        }
    )


def _validate_correction_role_scan_custody(
    value: Mapping[str, Any],
    *,
    expected_gate: str,
    expected_excluded_pids: Iterable[int],
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "gate", "repo_root", "excluded_pids",
        "base_forensic_or_scientific_process_matches",
        "correction_contract_process_matches", "nonexcluded_match_count",
        "all_nonexcluded_roles_zero", "pass_meaning", "pass",
        "content_digest",
    }
    raw_excluded = list(expected_excluded_pids)
    if any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in raw_excluded
    ):
        raise ForensicCorrectionContractError(
            "correction role-scan expected PID drift"
        )
    excluded = sorted(set(raw_excluded))
    base_matches = value.get(
        "base_forensic_or_scientific_process_matches"
    )
    correction_matches = value.get("correction_contract_process_matches")
    if type(base_matches) is not list or type(correction_matches) is not list:
        raise ForensicCorrectionContractError(
            "correction role-scan process rows drift"
        )
    validated_base = [
        BASE.validate_process_identity(row) for row in base_matches
    ]
    validated_correction = [
        BASE.validate_process_identity(row) for row in correction_matches
    ]
    all_rows = [*validated_base, *validated_correction]
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or expected_gate not in _CORRECTION_ROLE_SCAN_GATES
        or value.get("schema") != CORRECTION_ROLE_SCAN_CUSTODY_SCHEMA
        or value.get("gate") != expected_gate
        or value.get("repo_root") != str(CORRECTION_REPO_ROOT)
        or value.get("excluded_pids") != excluded
        or any(row["pid"] in excluded for row in all_rows)
        or validated_base
        != sorted(validated_base, key=lambda row: row["pid"])
        or validated_correction
        != sorted(validated_correction, key=lambda row: row["pid"])
        or value.get("nonexcluded_match_count") != len(all_rows)
        or value.get("all_nonexcluded_roles_zero") is not (not all_rows)
        or value.get("pass_meaning")
        != "ROLE_SCAN_CUSTODY_COMPLETE_NOT_A_ZERO_CLAIM"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction role-scan custody drift"
        )
    if reverify_live:
        live = _observe_correction_role_scan_custody(
            CORRECTION_REPO_ROOT,
            gate=expected_gate,
            excluded_pids=excluded,
        )
        if dict(value) != live:
            raise ForensicCorrectionContractError(
                "correction live role-scan custody drift"
            )
    return copy.deepcopy(dict(value))


def validate_correction_zero_scientific_counters(
    value: Mapping[str, Any],
) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(CORRECTION_SCIENTIFIC_COUNTER_FIELDS)
        or any(
            isinstance(value.get(key), bool)
            or not isinstance(value.get(key), int)
            or value[key] != 0
            for key in CORRECTION_SCIENTIFIC_COUNTER_FIELDS
        )
    ):
        raise ForensicCorrectionContractError("correction zero-counter drift")
    return copy.deepcopy(CORRECTION_ZERO_SCIENTIFIC_COUNTERS)


def validate_row_counted_content_binding(
    value: Mapping[str, Any],
    *,
    expected_path: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Validate exactly {path, sha256, bytes, content_digest, rows}."""

    required = {"path", "sha256", "bytes", "content_digest", "rows"}
    if type(value) is not dict or set(value) != required:
        raise ForensicCorrectionContractError(
            "row-counted content binding key-set drift"
        )
    ordinary = {
        key: value[key]
        for key in ("path", "sha256", "bytes", "content_digest")
    }
    try:
        BASE.validate_artifact_binding(
            ordinary, content_digest_required=True
        )
    except BASE.ForensicContractError as exc:
        raise ForensicCorrectionContractError(str(exc)) from exc
    if (
        not value["path"]
        or "\x00" in value["path"]
        or _LOWERCASE_SHA256_RE.fullmatch(value["sha256"]) is None
        or _LOWERCASE_SHA256_RE.fullmatch(value["content_digest"]) is None
        or isinstance(value["rows"], bool)
        or not isinstance(value["rows"], int)
        or value["rows"] < 0
        or (expected_path is not None and value["path"] != expected_path)
        or (expected_rows is not None and value["rows"] != expected_rows)
    ):
        raise ForensicCorrectionContractError(
            "row-counted content binding value drift"
        )
    return copy.deepcopy(dict(value))


def validate_correction_technical_diagnostic_attempt_accounting(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or value != CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
        or any(
            isinstance(value[key], bool) or not isinstance(value[key], int)
            for key in (
                "technical_diagnostic_authorized",
                "technical_diagnostic_consumed",
                "technical_diagnostic_remaining",
            )
        )
    ):
        raise ForensicCorrectionContractError(
            "correction technical diagnostic attempt accounting drift"
        )
    return copy.deepcopy(dict(value))


STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1 = (
    validate_row_counted_content_binding
)
ROW_COUNTED_CONTENT_BINDING_VALIDATOR_AUTHORITY = {
    "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
    "exported_symbol": "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1",
    "implementation": "validate_row_counted_content_binding",
    "exact_keys": ["path", "sha256", "bytes", "content_digest", "rows"],
    "rows": "CANONICAL_NONNEGATIVE_INTEGER",
    "sha256_and_content_digest": "LOWERCASE_HEX_64",
    "ordinary_four_key_validator_unchanged": True,
}


def produce_row_counted_content_binding(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_row_counted_content_binding(
        value,
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )


def validate_terminal_row_counted_content_binding(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_row_counted_content_binding(
        value,
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )


def check_external_row_counted_content_binding(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_row_counted_content_binding(
        value,
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )


def build_row_counted_terminal_binding_receipt(
    binding: Mapping[str, Any]
) -> dict[str, Any]:
    produced = produce_row_counted_content_binding(binding)
    validated = validate_terminal_row_counted_content_binding(produced)
    return BASE.attach_self_digest(
        {
            "schema": ROW_COUNTED_TERMINAL_RECEIPT_SCHEMA,
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "source_closure": validated,
            "producer_validated": True,
            "terminal_validated": True,
            "external_checker_validation_required": True,
            "pass": True,
        }
    )


def validate_row_counted_terminal_binding_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    binding = validate_terminal_row_counted_content_binding(
        value.get("source_closure")
    )
    expected = BASE.attach_self_digest(
        {
            "schema": ROW_COUNTED_TERMINAL_RECEIPT_SCHEMA,
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "source_closure": binding,
            "producer_validated": True,
            "terminal_validated": True,
            "external_checker_validation_required": True,
            "pass": True,
        }
    )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "row-counted terminal receipt drift"
        )
    return copy.deepcopy(dict(value))


def check_external_terminal_binding_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = validate_row_counted_terminal_binding_receipt(value)
    check_external_row_counted_content_binding(
        receipt["source_closure"]
    )
    return receipt


ROW_COUNTED_BINDING_FIXTURE_IDS = (
    "ORDINARY_FOUR_KEY_ACCEPT",
    "ROW_COUNTED_FIVE_KEY_ACCEPT",
    "FIVE_KEY_REJECTED_BY_ORDINARY",
    "FOUR_KEY_REJECTED_BY_ROW_COUNTED",
    "ROWS_MISSING",
    "EXTRA_SIXTH_FIELD",
    "ROWS_NEGATIVE",
    "ROWS_FLOAT",
    "ROWS_STRING",
    "ROWS_BOOLEAN",
    "SHA256_MALFORMED",
    "CONTENT_DIGEST_MALFORMED",
    "PATH_EMPTY",
    "TERMINAL_RECEIPT_REPEAT_IDENTICAL",
    "UTF8_SELF_DIGEST_REGENERATION",
    "PRODUCER_TERMINAL_CHECKER_AGREEMENT",
)
ROW_COUNTED_BINDING_FIXTURE_EXPECTATIONS = {
    "ORDINARY_FOUR_KEY_ACCEPT": "ACCEPT",
    "ROW_COUNTED_FIVE_KEY_ACCEPT": "ACCEPT",
    "FIVE_KEY_REJECTED_BY_ORDINARY": "REJECT",
    "FOUR_KEY_REJECTED_BY_ROW_COUNTED": "REJECT",
    "ROWS_MISSING": "REJECT",
    "EXTRA_SIXTH_FIELD": "REJECT",
    "ROWS_NEGATIVE": "REJECT",
    "ROWS_FLOAT": "REJECT",
    "ROWS_STRING": "REJECT",
    "ROWS_BOOLEAN": "REJECT",
    "SHA256_MALFORMED": "REJECT",
    "CONTENT_DIGEST_MALFORMED": "REJECT",
    "PATH_EMPTY": "REJECT",
    "TERMINAL_RECEIPT_REPEAT_IDENTICAL": "PASS/BYTE_IDENTICAL",
    "UTF8_SELF_DIGEST_REGENERATION": "PASS/CANONICAL_BYTE_IDENTICAL",
    "PRODUCER_TERMINAL_CHECKER_AGREEMENT": "PASS/EXACT",
}
UTF8_SELF_DIGEST_FIXTURE_LITERAL = "H1–H4"
UTF8_SELF_DIGEST_FIXTURE_HEX = "48 31 e2 80 93 48 34"


def _fixture_call(callable_value: Any, value: Mapping[str, Any]) -> str:
    try:
        callable_value(value)
    except Exception:
        return "REJECT"
    return "ACCEPT"


def validate_ordinary_four_key_content_binding(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    return BASE.validate_artifact_binding(
        value, content_digest_required=True
    )


def build_row_counted_binding_fixture_receipt() -> dict[str, Any]:
    four = {
        "path": "docs/ordinary.json",
        "sha256": "1" * 64,
        "bytes": 17,
        "content_digest": "2" * 64,
    }
    five = {
        **four,
        "path": str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        "rows": CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    }
    cases: dict[str, tuple[Any, dict[str, Any]]] = {
        "ORDINARY_FOUR_KEY_ACCEPT": (
            validate_ordinary_four_key_content_binding,
            four,
        ),
        "ROW_COUNTED_FIVE_KEY_ACCEPT": (
            validate_row_counted_content_binding,
            five,
        ),
        "FIVE_KEY_REJECTED_BY_ORDINARY": (
            validate_ordinary_four_key_content_binding,
            five,
        ),
        "FOUR_KEY_REJECTED_BY_ROW_COUNTED": (
            validate_row_counted_content_binding,
            four,
        ),
        "ROWS_MISSING": (
            validate_row_counted_content_binding,
            {key: value for key, value in five.items() if key != "rows"},
        ),
        "EXTRA_SIXTH_FIELD": (
            validate_row_counted_content_binding,
            {**five, "extra": False},
        ),
        "ROWS_NEGATIVE": (
            validate_row_counted_content_binding,
            {**five, "rows": -1},
        ),
        "ROWS_FLOAT": (
            validate_row_counted_content_binding,
            {**five, "rows": 9.0},
        ),
        "ROWS_STRING": (
            validate_row_counted_content_binding,
            {**five, "rows": "9"},
        ),
        "ROWS_BOOLEAN": (
            validate_row_counted_content_binding,
            {**five, "rows": True},
        ),
        "SHA256_MALFORMED": (
            validate_row_counted_content_binding,
            {**five, "sha256": "A" * 64},
        ),
        "CONTENT_DIGEST_MALFORMED": (
            validate_row_counted_content_binding,
            {**five, "content_digest": "G" * 64},
        ),
        "PATH_EMPTY": (
            validate_row_counted_content_binding,
            {**five, "path": ""},
        ),
    }
    rows: list[dict[str, Any]] = []
    for fixture_id in ROW_COUNTED_BINDING_FIXTURE_IDS[:13]:
        callable_value, supplied = cases[fixture_id]
        observed = _fixture_call(callable_value, supplied)
        expected = ROW_COUNTED_BINDING_FIXTURE_EXPECTATIONS[fixture_id]
        rows.append(
            {
                "fixture_id": fixture_id,
                "expected": expected,
                "observed": observed,
                "pass": observed == expected,
            }
        )
    first_terminal = build_row_counted_terminal_binding_receipt(five)
    second_terminal = build_row_counted_terminal_binding_receipt(five)
    repeated = _authority_bytes(first_terminal) == _authority_bytes(second_terminal)
    rows.append(
        {
            "fixture_id": "TERMINAL_RECEIPT_REPEAT_IDENTICAL",
            "expected": "PASS/BYTE_IDENTICAL",
            "observed": "PASS/BYTE_IDENTICAL" if repeated else "REJECT",
            "pass": repeated,
        }
    )
    unicode_value = BASE.attach_self_digest(
        {"fixture": UTF8_SELF_DIGEST_FIXTURE_LITERAL}
    )
    unicode_bytes = _authority_bytes(unicode_value)
    regenerated = BASE.attach_self_digest(
        json.loads(unicode_bytes.decode("utf-8"))
    )
    unicode_pass = (
        UTF8_SELF_DIGEST_FIXTURE_LITERAL.encode("utf-8").hex(" ")
        == UTF8_SELF_DIGEST_FIXTURE_HEX
        and unicode_bytes == _authority_bytes(regenerated)
    )
    rows.append(
        {
            "fixture_id": "UTF8_SELF_DIGEST_REGENERATION",
            "expected": "PASS/CANONICAL_BYTE_IDENTICAL",
            "observed": (
                "PASS/CANONICAL_BYTE_IDENTICAL" if unicode_pass else "REJECT"
            ),
            "literal": UTF8_SELF_DIGEST_FIXTURE_LITERAL,
            "utf8_hex": UTF8_SELF_DIGEST_FIXTURE_HEX,
            "pass": unicode_pass,
        }
    )
    producer = produce_row_counted_content_binding(five)
    terminal = validate_terminal_row_counted_content_binding(producer)
    checker = check_external_row_counted_content_binding(terminal)
    agreement = producer == terminal == checker
    rows.append(
        {
            "fixture_id": "PRODUCER_TERMINAL_CHECKER_AGREEMENT",
            "expected": "PASS/EXACT",
            "observed": "PASS/EXACT" if agreement else "REJECT",
            "pass": agreement,
        }
    )
    zero_boundary = validate_row_counted_content_binding(
        {**five, "rows": 0}
    )["rows"] == 0
    if not all(row["pass"] is True for row in rows) or not zero_boundary:
        raise ForensicCorrectionContractError("row-counted fixture failed")
    return BASE.attach_self_digest(
        {
            "schema": ROW_COUNTED_FIXTURE_SCHEMA,
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "fixture_ids": list(ROW_COUNTED_BINDING_FIXTURE_IDS),
            "rows": rows,
            "row_count": 16,
            "rows_zero_boundary": {
                "fixture_id": "ROWS_ZERO_ACCEPT",
                "expected": "ACCEPT",
                "observed": "ACCEPT",
                "pass": True,
            },
            "rows_zero_accepted_by_generic_row_counted_schema": True,
            "all_pass": True,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_row_counted_binding_fixture_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    expected = build_row_counted_binding_fixture_receipt()
    if dict(value) != expected:
        raise ForensicCorrectionContractError("row-counted fixture drift")
    return copy.deepcopy(dict(value))


def build_corrected_preexecution_child_result(
    *,
    repo_root: str | Path,
    launcher_process_identity: Mapping[str, Any],
    child_process_identity: Mapping[str, Any],
    repo_head: str,
    repo_clean: bool,
    scientific_contract_digest: str,
    forensic_authority_source_closure: Mapping[str, Any],
    forensic_freeze_custody: Mapping[str, Any],
    output_namespace_before: Sequence[Mapping[str, str]],
    output_namespace_after: Sequence[Mapping[str, str]],
    mismatch_evidence: Mapping[str, Any],
    committed_source_root_cause_proof: Mapping[str, Any],
    current_runtime_context: Mapping[str, Any],
    scientific_counters: Mapping[str, Any],
) -> dict[str, Any]:
    launcher = BASE.validate_process_identity(launcher_process_identity)
    child = BASE.validate_process_identity(child_process_identity)
    closure = validate_row_counted_content_binding(
        forensic_authority_source_closure,
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )
    freeze = validate_correction_freeze_custody_receipt(
        forensic_freeze_custody,
        repo_root=repo_root,
        reverify_failed_root=False,
    )
    mismatch = BASE.validate_archive_schema_mismatch_evidence(
        mismatch_evidence
    )
    BASE.validate_self_digest(committed_source_root_cause_proof)
    before = BASE._validate_namespace_snapshot(list(output_namespace_before))
    after = BASE._validate_namespace_snapshot(list(output_namespace_after))
    context = BASE.validate_current_runtime_context(current_runtime_context)
    counters = validate_correction_zero_scientific_counters(
        scientific_counters
    )
    fixture = build_row_counted_binding_fixture_receipt()
    if (
        not isinstance(repo_head, str)
        or not re.fullmatch(r"[0-9a-f]{40}", repo_head)
        or repo_clean is not True
        or repo_head != freeze["repo_head"]
        or scientific_contract_digest
        != BASE.BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or closure != freeze["correction_source_closure"]
        or dict(committed_source_root_cause_proof)
        != freeze["committed_source_root_cause_proof"]
        or before != after
    ):
        raise ForensicCorrectionContractError(
            "corrected child result custody drift"
        )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTED_CHILD_RESULT_SCHEMA,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC_CORRECTION_1",
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "repo_head": repo_head,
            "repo_clean": True,
            "scientific_contract_digest": scientific_contract_digest,
            "forensic_authority_source_closure": closure,
            "forensic_freeze_custody": _runtime_json_binding(
                "forensic_freeze_custody", freeze
            ),
            "row_counted_binding_fixtures": fixture,
            "terminal_binding_receipt": (
                build_row_counted_terminal_binding_receipt(closure)
            ),
            "output_namespace_before": before,
            "output_namespace_after": after,
            "namespace_unchanged": True,
            "attempt_namespace_created": False,
            "attempt_reservation_created": False,
            "mismatch_evidence": mismatch,
            "committed_source_root_cause_proof": copy.deepcopy(
                dict(committed_source_root_cause_proof)
            ),
            "current_runtime_context": context,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": counters,
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_corrected_preexecution_child_result(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    forensic_freeze_custody: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_corrected_preexecution_child_result(
        repo_root=repo_root,
        launcher_process_identity=value.get("launcher_process_identity"),
        child_process_identity=value.get("child_process_identity"),
        repo_head=value.get("repo_head"),
        repo_clean=value.get("repo_clean"),
        scientific_contract_digest=value.get("scientific_contract_digest"),
        forensic_authority_source_closure=value.get(
            "forensic_authority_source_closure"
        ),
        forensic_freeze_custody=forensic_freeze_custody,
        output_namespace_before=value.get("output_namespace_before"),
        output_namespace_after=value.get("output_namespace_after"),
        mismatch_evidence=value.get("mismatch_evidence"),
        committed_source_root_cause_proof=value.get(
            "committed_source_root_cause_proof"
        ),
        current_runtime_context=value.get("current_runtime_context"),
        scientific_counters=value.get("scientific_counters"),
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError("corrected child result drift")
    return copy.deepcopy(dict(value))


def _failed_root_metadata_preflight() -> dict[str, Any]:
    """Reject any unknown/type/link entry before opening a single leaf."""

    root = FAILED_PREEXECUTION_DIAGNOSTIC_ROOT
    root_fd = BASE._open_absolute_directory_no_follow(root.absolute())
    root_info = os.fstat(root_fd)
    directory_paths: set[str] = set()
    file_stats: dict[str, dict[str, Any]] = {}
    stat_rows: dict[str, dict[str, Any]] = {
        ".": {
            "path": ".", "kind": "DIRECTORY",
            "mode": stat.S_IMODE(root_info.st_mode),
            "uid": int(root_info.st_uid), "gid": int(root_info.st_gid),
            "nlink": int(root_info.st_nlink),
            "device": int(root_info.st_dev), "inode": int(root_info.st_ino),
        }
    }
    seen: set[tuple[int, int]] = {
        (int(os.fstat(root_fd).st_dev), int(os.fstat(root_fd).st_ino))
    }
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    def visit(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise ForensicCorrectionContractError(
                    "failed-root manifest entry-name drift"
                )
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            inode = (int(before.st_dev), int(before.st_ino))
            if inode in seen:
                raise ForensicCorrectionContractError(
                    "failed-root duplicate inode/hardlink"
                )
            seen.add(inode)
            relative = name if not prefix else f"{prefix}/{name}"
            if stat.S_ISDIR(before.st_mode):
                directory_paths.add(relative)
                stat_rows[relative] = {
                    "path": relative, "kind": "DIRECTORY",
                    "mode": stat.S_IMODE(before.st_mode),
                    "uid": int(before.st_uid), "gid": int(before.st_gid),
                    "nlink": int(before.st_nlink),
                    "device": int(before.st_dev), "inode": int(before.st_ino),
                }
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    if not BASE._same_opened_stat(before, os.fstat(child_fd)):
                        raise ForensicCorrectionContractError(
                            "failed-root directory race"
                        )
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ForensicCorrectionContractError(
                    "failed-root symlink/nonregular/multilink"
                )
            file_stats[relative] = {
                "path": relative,
                "kind": "FILE",
                "mode": stat.S_IMODE(before.st_mode),
                "uid": int(before.st_uid),
                "gid": int(before.st_gid),
                "nlink": int(before.st_nlink),
                "bytes": int(before.st_size),
                "device": int(before.st_dev),
                "inode": int(before.st_ino),
            }
            stat_rows[relative] = file_stats[relative]

    try:
        visit(root_fd, "")
    finally:
        os.close(root_fd)
    if (
        tuple(sorted(directory_paths)) != FAILED_TECHNICAL_DIRECTORY_PATHS
        or tuple(sorted(file_stats)) != FAILED_TECHNICAL_FILE_PATHS
    ):
        raise ForensicCorrectionContractError(
            "failed-root exact technical namespace drift"
        )
    return {
        "file_stats": file_stats,
        "stat_rows": stat_rows,
        "all_inodes": sorted(seen),
    }


def _anchored_manifest_rows(root: Path) -> list[dict[str, Any]]:
    if root.absolute() != FAILED_PREEXECUTION_DIAGNOSTIC_ROOT:
        raise ForensicCorrectionContractError("failed-root manifest root drift")
    preflight = _failed_root_metadata_preflight()
    rows: list[dict[str, Any]] = []
    for relative in FAILED_TECHNICAL_FILE_PATHS:
        payload = BASE._read_no_follow_bound_file(
            root,
            relative,
            expected_stat_rows=preflight["stat_rows"],
        )
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    if _failed_root_metadata_preflight() != preflight:
        raise ForensicCorrectionContractError(
            "failed-root metadata changed across admitted reads"
        )
    return rows


def build_failed_diagnostic_root_custody() -> dict[str, Any]:
    metadata = _failed_root_metadata_preflight()
    rows = _anchored_manifest_rows(FAILED_PREEXECUTION_DIAGNOSTIC_ROOT)
    manifest = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    observed = {
        "file_count": len(rows),
        "total_file_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    if observed != {
        "file_count": FAILED_DIAGNOSTIC_ROOT_AUTHORITY["file_count"],
        "total_file_bytes": FAILED_DIAGNOSTIC_ROOT_AUTHORITY[
            "total_file_bytes"
        ],
        "manifest_sha256": FAILED_DIAGNOSTIC_ROOT_AUTHORITY[
            "manifest_sha256"
        ],
    }:
        raise ForensicCorrectionContractError(
            "failed diagnostic root immutable manifest drift"
        )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "failed_diagnostic_root_custody.correction_1.v1"
            ),
            "authority": copy.deepcopy(FAILED_DIAGNOSTIC_ROOT_AUTHORITY),
            "observed": observed,
            "metadata_stat_row_count": len(metadata["stat_rows"]),
            "metadata_stat_inventory_sha256": BASE.canonical_json_sha256(
                [metadata["stat_rows"][key] for key in sorted(metadata["stat_rows"])]
            ),
            "anchored_component_no_follow_reads": True,
            "manifest_rows_not_persisted": True,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_failed_diagnostic_root_custody(
    value: Mapping[str, Any], *, reverify_live: bool = True
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    required = {
        "schema", "authority", "observed", "metadata_stat_row_count",
        "metadata_stat_inventory_sha256", "anchored_component_no_follow_reads",
        "manifest_rows_not_persisted", "scientific_counters", "files_reused",
        "pass", "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "failed_diagnostic_root_custody.correction_1.v1"
        )
        or value.get("authority") != FAILED_DIAGNOSTIC_ROOT_AUTHORITY
        or value.get("observed") != {
            "file_count": FAILED_DIAGNOSTIC_ROOT_AUTHORITY["file_count"],
            "total_file_bytes": FAILED_DIAGNOSTIC_ROOT_AUTHORITY[
                "total_file_bytes"
            ],
            "manifest_sha256": FAILED_DIAGNOSTIC_ROOT_AUTHORITY[
                "manifest_sha256"
            ],
        }
        or value.get("metadata_stat_row_count")
        != 1 + len(FAILED_TECHNICAL_DIRECTORY_PATHS) + len(
            FAILED_TECHNICAL_FILE_PATHS
        )
        or not isinstance(value.get("metadata_stat_inventory_sha256"), str)
        or _LOWERCASE_SHA256_RE.fullmatch(
            value["metadata_stat_inventory_sha256"]
        ) is None
        or value.get("anchored_component_no_follow_reads") is not True
        or value.get("manifest_rows_not_persisted") is not True
        or value.get("files_reused") != 0
        or value.get("pass") is not True
        or (reverify_live and dict(value) != build_failed_diagnostic_root_custody())
    ):
        raise ForensicCorrectionContractError("failed-root custody drift")
    return copy.deepcopy(dict(value))


def build_correction_inode_nonreuse_proof() -> dict[str, Any]:
    failed = _failed_root_metadata_preflight()
    fresh = BASE._observe_final_diagnostic_namespace(
        PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
        inventory_receipt_must_exist=True,
    )
    failed_inodes = {tuple(row) for row in failed["all_inodes"]}
    fresh_inodes = {
        (int(row["device"]), int(row["inode"])) for row in fresh["rows"]
    }
    shared = sorted(failed_inodes & fresh_inodes)
    if shared:
        raise ForensicCorrectionContractError("failed-root inode was reused")
    return BASE.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.inode_nonreuse.correction_1.v1",
            "failed_file_count": len(FAILED_TECHNICAL_FILE_PATHS),
            "fresh_inventory_row_count": fresh["row_count"],
            "shared_inodes": [],
            "shared_inode_count": 0,
            "files_reused": 0,
            "pass": True,
        }
    )


def _ast_digest(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _assignment_names(node: ast.AST) -> list[str]:
    targets: list[ast.AST] = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = (
            list(node.targets) if isinstance(node, ast.Assign) else [node.target]
        )
    if not targets or any(not isinstance(target, ast.Name) for target in targets):
        return []
    return sorted(target.id for target in targets)


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    return sorted(
        alias.asname or alias.name.split(".", 1)[0]
        for alias in node.names
    )


def _ast_surface(raw: bytes) -> dict[str, Any]:
    try:
        module = ast.parse(raw.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ForensicCorrectionContractError("overlay source parse drift") from exc
    imports: list[str] = []
    assignments: dict[str, str] = {}
    functions: dict[str, str] = {}
    classes: dict[str, str] = {}
    other: list[str] = []
    bound_names: set[str] = set()
    for node in module.body:
        digest = _ast_digest(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = _import_names(node)
            if any(name in bound_names for name in names):
                raise ForensicCorrectionContractError(
                    "duplicate top-level bound name"
                )
            bound_names.update(names)
            imports.append(digest)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assignment_names(node)
            if not names:
                other.append(digest)
                continue
            if any(name in bound_names for name in names):
                raise ForensicCorrectionContractError(
                    "duplicate top-level bound name"
                )
            bound_names.update(names)
            for name in names:
                if name in assignments:
                    raise ForensicCorrectionContractError(
                        f"duplicate top-level assignment: {name}"
                    )
                assignments[name] = digest
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in bound_names:
                raise ForensicCorrectionContractError(
                    f"duplicate top-level bound name: {node.name}"
                )
            bound_names.add(node.name)
            functions[node.name] = digest
        elif isinstance(node, ast.ClassDef):
            if node.name in bound_names:
                raise ForensicCorrectionContractError(
                    f"duplicate top-level bound name: {node.name}"
                )
            bound_names.add(node.name)
            classes[node.name] = digest
        else:
            other.append(digest)
    return {
        "imports": imports,
        "assignments": assignments,
        "functions": functions,
        "classes": classes,
        "other": other,
    }


def _surface_delta(base: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for category in ("assignments", "functions", "classes"):
        before = base[category]
        after = live[category]
        output[category] = {
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(
                name for name in set(before) & set(after)
                if before[name] != after[name]
            ),
            "live_hashes": {
                name: after[name]
                for name in sorted(set(after) - set(before) | {
                    name for name in set(before) & set(after)
                    if before[name] != after[name]
                })
            },
        }
    output["imports"] = {
        "base": base["imports"],
        "live": live["imports"],
        "changed": base["imports"] != live["imports"],
    }
    output["other"] = {
        "base": base["other"],
        "live": live["other"],
        "changed": base["other"] != live["other"],
    }
    return output


CORRECTION_OVERLAY_AST_EXPECTED: dict[str, Any] = {'scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py': {'base_content_sha256': '5fc42f354ca112b7a7eb72a8f5b32c02ffb5b35b9347fe0015c5bfcddfdf973a',
                                                          'base_git_blob_oid': '3aeb63d46f2b3d518bc8aedd6b837d819df37ba1',
                                                          'live_content_sha256': '8d145c94a38a67bb0ce7a0b99a50aa0cd902c294adbe66bdc073ec10813b4684',
                                                          'live_bytes': 491885,
                                                          'changed_legacy_function_ast_sha256': {'_active_forensic_processes': '10a977c4b98f158b88033924576f1a8bfb9acfcabafa7a4f841ec7893cf1a530',
                                                                                                 '_classify_experiment_argv': '23ecd4264c04e208002a4edb35d0c20b53ccda7a9cb5ac207d9579bdbb57b1c4',
                                                                                                 '_classify_forensic_argv': '9862db38d511c98e1d5fc83ef92861e7d1ed3e5450cd958c35a3977b52820425',
                                                                                                 '_execute_preexecution_diagnostic_under_umask': 'c2ba051853d9bbf30177199a7993d7b82cdd7759ecf24896db57224b9464c570',
                                                                                                 '_execute_preexecution_only_diagnostic_child': 'dc687898ce8cd42904fd69920cb77520f81e2b39883180fea7bfc36a38f94ce9',
                                                                                                 '_require_synthetic_diagnostic_root': '3ed9c830fd432d6d1a9768b6ceae5659272ca0c3417ec63945651331ee598870',
                                                                                                 '_validate_preexecution_child_payload': '62017f2ac2a6f58ae99c9c2947625405fc1bb90c4cfc08d436bdc0132c92c93a',
                                                                                                 'main': 'c4769e39d14e7b4756473ceedb986b41338db2b742defd3fc0cf87cb29e2d25b'},
                                                          'added_function_ast_sha256': {'_build_preexecution_child_result_for_root': '55057d63b72e8fd2e48b1760cff48b1160b4b8856424e39d872c2568328735a2',
                                                                                        '_correction_contract_for_diagnostic_root': 'b1cc826a7dbec805965a729aea2ec7d493a390c5404abf7569d56854201295e0',
                                                                                        '_correction_terminal_coordinator_from_environment': 'de79bca30ad39c6d16b460da24591453f7609f987d6a07b6f1fb05c0c4b29ce7',
                                                                                        '_expected_preexecution_diagnostic_correction_checker_argv': 'c9a478917a3008fb8090e9887909e6fcbb4a3c4c5310051653ab22ce6828b339',
                                                                                        '_expected_preexecution_diagnostic_correction_child_argv': '37dc4f867d727ee4f430d8b2d1af281f0f105f232399281bf0239748332a7058',
                                                                                        '_expected_preexecution_diagnostic_correction_finalizer_argv': 'd7b89471e472e8db01ae0854a7e32216a8cd23762366dd9a32a3d730d1b2772d',
                                                                                        '_expected_preexecution_diagnostic_correction_internal_argv': 'cbc36486419e6f60ed7e7ae32d20e649548bd465c6fb83a95397d92710661389',
                                                                                        '_expected_preexecution_diagnostic_correction_launcher_argv': 'b1e13e4bc0827d8d85facb03194b5c3a7418edf91e78259a3188ee6e53ba9e3f',
                                                                                        '_expected_preexecution_forensic_correction_freeze_argv': 'd75a44cc868301408628b0c22e04abc86464b68122e5956dac16d4e76f436ff2',
                                                                                        '_forensic_correction_1_contract': 'b858585d0008dca758073b77d1de99268a6308646b6b495092804b741e8ced6e',
                                                                                        '_load_forensic_correction_source_closure': 'dafa5f8adf4e763e0e2b52482e1a638b907b72cde7e270cc2f3200e6209bdfd2',
                                                                                        '_require_exact_live_preexecution_diagnostic_correction_launcher': 'f0b644e5bb0e11eb363a55286207970f377ffa9b5bb0353b2e811b544f8c9053',
                                                                                        '_validate_forensic_correction_1_constant_alignment': '9bec0fe2f2c1498842502d8983b49aa9f2ca3e08f141f6ea5ecb0d3c2c82a635',
                                                                                        '_validate_preexecution_active_freeze': '99cf6885ad954bab9d39f2b0e3ad5edb8bbd048c4edb8af9f2df8c5865cebde6',
                                                                                        '_validate_preexecution_child_result_for_root': '5b6202379ee397f8fb24dcd7da91a0e8f3f2cbf8c74e692a0149358d30ad6657',
                                                                                        'check_preexecution_forensic_correction_1': '1bdc02edfe1c2adb47c11d4689786a3b39f8f57a03dda39615a6541509ac455f',
                                                                                        'execute_preexecution_correction_1': 'a0af0a58f703caa6778ef6bea291cafa60214f2658916221f0d55755c336a4fc',
                                                                                        'finalize_preexecution_forensic_correction_1': '376e4a4c44d23d6a4a95d1b92181c46e02ba19133a4a3ecaecd409e20288e27b',
                                                                                        'freeze_preexecution_forensic_correction_1_contract': '6407fdfc2ede8446ef855b55fc6bedb4f96cd587b1f853070f9fbab2bf872a03'},
                                                          'added_assignment_ast_sha256': {'PREEXECUTION_DIAGNOSTIC_CORRECTION_CHECKER_SUBCOMMAND': 'f7d7b20de0c79d6866d9c0232014741ae86f09812a4fd1b342a65e7239ccdbf7',
                                                                                          'PREEXECUTION_DIAGNOSTIC_CORRECTION_CHILD_SUBCOMMAND': '41424a504b3e0e1bfe70df49bb311ef7e577d870641c33d5cd15f5d0c568d42b',
                                                                                          'PREEXECUTION_DIAGNOSTIC_CORRECTION_FINALIZER_SUBCOMMAND': '3cee4e0e24114fce6c24df53d886e13a83dcce9f3198e3fd6457e3f60cd29e4e',
                                                                                          'PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT': 'ff7a1efa196ff5eae7ae4416eb6deb186b1652173135ec5c7a041308e27b42f1',
                                                                                          'PREEXECUTION_DIAGNOSTIC_CORRECTION_SUBCOMMAND': '26f5b3655f3998718feff9b5e29531db672cae37e2a347836109bab36203d5ba',
                                                                                          'PREEXECUTION_FORENSIC_CORRECTION_FREEZE_SUBCOMMAND': '2444e11c500cdaa6cae97b57fc0f660cb8e9b7ac6902395306902715d460e05a'},
                                                          'added_class_ast_sha256': {},
                                                          'base_import_count': 26,
                                                          'base_import_ast_set_sha256': 'e493701095d14592c627283c10e826b8c9c453c30d4bf7e02da3712f8b2ed23b',
                                                          'live_import_count': 26,
                                                          'live_import_ast_set_sha256': 'e493701095d14592c627283c10e826b8c9c453c30d4bf7e02da3712f8b2ed23b',
                                                          'top_level_other_count': 4,
                                                          'top_level_other_ast_set_sha256': 'a97d9f3bb54959a7ae62a6dfea1deb49cd5a0557968b966d29ed05ab641f8ec0'},
 'scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py': {'base_content_sha256': '0f6f6d0c1e040dfb6661ffefeec6680029e360b407a4d185e8bc14898d937e96',
                                                                                   'base_git_blob_oid': 'e9cd3d6df6b46e14e89ffa2b86a3914e86360a9f',
                                                                                   'live_content_sha256': '1926e9bf4020ceb9875d6adcb829e04a80e3bcf40fc600be0965937b433de9fe',
                                                                                   'live_bytes': 36906,
                                                                                   'changed_legacy_function_ast_sha256': {'_arguments': 'b08128b0bc1e85a862a6ab596d5f5465050e7f8bb86d3a54f381f0372d13dba0',
                                                                                                                          '_inner_argv': '70559e323b561008a3e7952d9908abb72da22fc19777765b61b819ab692f1baa',
                                                                                                                          'main': '8ae0e84c4b4b122a7408e7514b848461cb8e21dc6b68cd210d6d01be6632370a'},
                                                                                   'added_function_ast_sha256': {'_correction_phase_arguments': '5d62b26e6f35e218d3f63d54e31bfeeabed5671d4d7cc487b3eb67fec6ac42d8',
                                                                                                                 '_correction_phase_contract': '48b3ea023451dba3922cedea50a04cc3a4049991222ab5af1941cee6c1f82a2f',
                                                                                                                 '_correction_phase_coordinator': 'e44e2e7f581a15b86502970bba378d372089e469489c5c633da6602e723805a8',
                                                                                                                 '_correction_phase_inner_receipt': 'c447c1c4092b9642881b7cb7049ef86cab7ebe6c1a72490b082db1c7daa0a94f',
                                                                                                                 '_correction_phase_mode': '8c0307138759b70158ffec2d5ac08c154c034df37dd8c572fadac94b7464801f',
                                                                                                                 '_correction_phase_process_identity': '2837a7754a95af74406b532261b970e2e0b5df109aaca61cf074dcaf2ae1a516',
                                                                                                                 '_run_correction_phase_wrapper': 'a4507756a8a59e46fc0f7dc6f047a8cb0d46e2214ecfd444f5e01d5bb2dcb2a2'},
                                                                                   'added_assignment_ast_sha256': {'CORRECTION_CHECKER_MODE': '7cfa89208e6bb44e71e336d2d2b886b20a2aa8d56f7a098f580b544ad87e92e6',
                                                                                                                   'CORRECTION_DIAGNOSTIC_ROOT': '6af0cb597847831ee31eff15a2b4d52739480a31e809b1c0d6bde9ca0131e6fb',
                                                                                                                   'CORRECTION_FINALIZER_MODE': '814efeeffccf3443426c0f8bbbcac31877d9878948e6bca5d8ab9bfc20f39141',
                                                                                                                   'CORRECTION_PHASE_MODES': '9dc7286a88933579596dcb8c1820d5b77fe9b471f682ca808b8c454db7718026',
                                                                                                                   'CORRECTION_REAL_MODE': 'c9fafa653b09096b45531a010e0b6b03fec2e2f38c09c94b322f07c317b4911f'},
                                                                                   'added_class_ast_sha256': {},
                                                                                   'base_import_count': 13,
                                                                                   'base_import_ast_set_sha256': '40117883c009d3d7e4a4fa7140ae99e9d746ed44740261297bc5e2c9f5c40b06',
                                                                                   'live_import_count': 16,
                                                                                   'live_import_ast_set_sha256': '17ae9e3f6afa1eb8622498221696854509d58696ee7101ac2fbc4f14d82e5dc9',
                                                                                   'top_level_other_count': 3,
                                                                                   'top_level_other_ast_set_sha256': '5d01a98550c19728fd22dc847f232ee85203e8a44234ed6a2d3345a9d186a916'}}
CORRECTION_OVERLAY_SEMANTIC_ALLOW_LIST: dict[str, Any] = {'scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py': {'changed_legacy_functions': ['_active_forensic_processes',
                                                                                       '_classify_experiment_argv',
                                                                                       '_classify_forensic_argv',
                                                                                       '_execute_preexecution_diagnostic_under_umask',
                                                                                       '_execute_preexecution_only_diagnostic_child',
                                                                                       '_require_synthetic_diagnostic_root',
                                                                                       '_validate_preexecution_child_payload',
                                                                                       'main'],
                                                          'added_functions': ['_build_preexecution_child_result_for_root',
                                                                              '_correction_contract_for_diagnostic_root',
                                                                              '_correction_terminal_coordinator_from_environment',
                                                                              '_expected_preexecution_diagnostic_correction_checker_argv',
                                                                              '_expected_preexecution_diagnostic_correction_child_argv',
                                                                              '_expected_preexecution_diagnostic_correction_finalizer_argv',
                                                                              '_expected_preexecution_diagnostic_correction_internal_argv',
                                                                              '_expected_preexecution_diagnostic_correction_launcher_argv',
                                                                              '_expected_preexecution_forensic_correction_freeze_argv',
                                                                              '_forensic_correction_1_contract',
                                                                              '_load_forensic_correction_source_closure',
                                                                              '_require_exact_live_preexecution_diagnostic_correction_launcher',
                                                                              '_validate_forensic_correction_1_constant_alignment',
                                                                              '_validate_preexecution_active_freeze',
                                                                              '_validate_preexecution_child_result_for_root',
                                                                              'check_preexecution_forensic_correction_1',
                                                                              'execute_preexecution_correction_1',
                                                                              'finalize_preexecution_forensic_correction_1',
                                                                              'freeze_preexecution_forensic_correction_1_contract'],
                                                          'added_assignments': ['PREEXECUTION_DIAGNOSTIC_CORRECTION_CHECKER_SUBCOMMAND',
                                                                                'PREEXECUTION_DIAGNOSTIC_CORRECTION_CHILD_SUBCOMMAND',
                                                                                'PREEXECUTION_DIAGNOSTIC_CORRECTION_FINALIZER_SUBCOMMAND',
                                                                                'PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT',
                                                                                'PREEXECUTION_DIAGNOSTIC_CORRECTION_SUBCOMMAND',
                                                                                'PREEXECUTION_FORENSIC_CORRECTION_FREEZE_SUBCOMMAND'],
                                                          'added_classes': [],
                                                          'imports_changed': False,
                                                          'live_import_count': 26,
                                                          'live_import_ast_set_sha256': 'e493701095d14592c627283c10e826b8c9c453c30d4bf7e02da3712f8b2ed23b'},
 'scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py': {'changed_legacy_functions': ['_arguments',
                                                                                                                '_inner_argv',
                                                                                                                'main'],
                                                                                   'added_functions': ['_correction_phase_arguments',
                                                                                                       '_correction_phase_contract',
                                                                                                       '_correction_phase_coordinator',
                                                                                                       '_correction_phase_inner_receipt',
                                                                                                       '_correction_phase_mode',
                                                                                                       '_correction_phase_process_identity',
                                                                                                       '_run_correction_phase_wrapper'],
                                                                                   'added_assignments': ['CORRECTION_CHECKER_MODE',
                                                                                                         'CORRECTION_DIAGNOSTIC_ROOT',
                                                                                                         'CORRECTION_FINALIZER_MODE',
                                                                                                         'CORRECTION_PHASE_MODES',
                                                                                                         'CORRECTION_REAL_MODE'],
                                                                                   'added_classes': [],
                                                                                   'imports_changed': True,
                                                                                   'live_import_count': 16,
                                                                                   'live_import_ast_set_sha256': '17ae9e3f6afa1eb8622498221696854509d58696ee7101ac2fbc4f14d82e5dc9'}}
_MAXIMUM_ALLOWED_CHANGED_LEGACY_FUNCTIONS = {
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py": {
        "_active_forensic_processes",
        "_classify_experiment_argv",
        "_classify_forensic_argv",
        "_execute_preexecution_diagnostic_under_umask",
        "_execute_preexecution_only_diagnostic_child",
        "_require_synthetic_diagnostic_root",
        "_validate_preexecution_child_payload",
        "main",
    },
    "scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py": {
        "_arguments",
        "_inner_argv",
        "main",
    },
}


def build_correction_overlay_ast_proof(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    frozen_contract, contract_oid = BASE._git_blob_bytes(
        root, SOURCE_FORENSIC_FREEZE_COMMIT, FROZEN_FORENSIC_CONTRACT_PATH
    )
    if (
        hashlib.sha256(frozen_contract).hexdigest()
        != FROZEN_FORENSIC_CONTRACT_SHA256
        or contract_oid != FROZEN_FORENSIC_CONTRACT_BLOB_OID
        or BASE._read_no_follow_bound_file(
            root, FROZEN_FORENSIC_CONTRACT_PATH, expected_stat_rows=None
        )
        != frozen_contract
    ):
        raise ForensicCorrectionContractError(
            "frozen forensic contract byte custody drift"
        )
    module = ast.parse(frozen_contract.decode("utf-8"))
    validators = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_artifact_binding"
    ]
    if len(validators) != 1 or _ast_digest(validators[0]) != (
        FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256
    ):
        raise ForensicCorrectionContractError(
            "frozen four-key validator AST drift"
        )
    files: dict[str, Any] = {}
    expected_overlay_paths = set(
        _MAXIMUM_ALLOWED_CHANGED_LEGACY_FUNCTIONS
    )
    if (
        not CORRECTION_OVERLAY_SEMANTIC_ALLOW_LIST
        or set(CORRECTION_OVERLAY_SEMANTIC_ALLOW_LIST)
        != expected_overlay_paths
        or not CORRECTION_OVERLAY_AST_EXPECTED
        or set(CORRECTION_OVERLAY_AST_EXPECTED) != expected_overlay_paths
    ):
        raise ForensicCorrectionContractError(
            "correction overlay authority path set is not frozen"
        )
    for relative in _MAXIMUM_ALLOWED_CHANGED_LEGACY_FUNCTIONS:
        base_raw, base_oid = BASE._git_blob_bytes(
            root, SOURCE_FORENSIC_FREEZE_COMMIT, relative
        )
        live_raw = BASE._read_no_follow_bound_file(
            root, relative, expected_stat_rows=None
        )
        delta = _surface_delta(_ast_surface(base_raw), _ast_surface(live_raw))
        for category in ("assignments", "functions", "classes"):
            if delta[category]["removed"]:
                raise ForensicCorrectionContractError(
                    f"overlay removed legacy {category}: {relative}"
                )
        if (
            delta["assignments"]["changed"]
            or delta["classes"]["changed"]
            or delta["other"]["changed"]
        ):
            raise ForensicCorrectionContractError(
                f"overlay changed legacy assignment/class: {relative}"
            )
        if delta["functions"]["changed"] != sorted(
            _MAXIMUM_ALLOWED_CHANGED_LEGACY_FUNCTIONS[relative]
        ):
            raise ForensicCorrectionContractError(
                f"overlay changed legacy function set drift: {relative}"
            )
        semantic = CORRECTION_OVERLAY_SEMANTIC_ALLOW_LIST.get(relative)
        observed_semantic = {
            "changed_legacy_functions": delta["functions"]["changed"],
            "added_functions": delta["functions"]["added"],
            "added_assignments": delta["assignments"]["added"],
            "added_classes": delta["classes"]["added"],
            "imports_changed": delta["imports"]["changed"],
            "live_import_count": len(delta["imports"]["live"]),
            "live_import_ast_set_sha256": BASE.canonical_json_sha256(
                delta["imports"]["live"]
            ),
        }
        if semantic != observed_semantic:
            raise ForensicCorrectionContractError(
                f"overlay semantic allow-list drift: {relative}"
            )
        observed_file = {
            "base_content_sha256": hashlib.sha256(base_raw).hexdigest(),
            "base_git_blob_oid": base_oid,
            "live_content_sha256": hashlib.sha256(live_raw).hexdigest(),
            "live_bytes": len(live_raw),
            "changed_legacy_function_ast_sha256": {
                name: delta["functions"]["live_hashes"][name]
                for name in delta["functions"]["changed"]
            },
            "added_function_ast_sha256": {
                name: delta["functions"]["live_hashes"][name]
                for name in delta["functions"]["added"]
            },
            "added_assignment_ast_sha256": {
                name: delta["assignments"]["live_hashes"][name]
                for name in delta["assignments"]["added"]
            },
            "added_class_ast_sha256": {
                name: delta["classes"]["live_hashes"][name]
                for name in delta["classes"]["added"]
            },
            "base_import_count": len(delta["imports"]["base"]),
            "base_import_ast_set_sha256": BASE.canonical_json_sha256(
                delta["imports"]["base"]
            ),
            "live_import_count": len(delta["imports"]["live"]),
            "live_import_ast_set_sha256": BASE.canonical_json_sha256(
                delta["imports"]["live"]
            ),
            "top_level_other_count": len(delta["other"]["live"]),
            "top_level_other_ast_set_sha256": (
                BASE.canonical_json_sha256(delta["other"]["live"])
            ),
        }
        if observed_file != CORRECTION_OVERLAY_AST_EXPECTED[relative]:
            raise ForensicCorrectionContractError(
                f"overlay AST/hash authority drift: {relative}"
            )
        files[relative] = observed_file
    if files != CORRECTION_OVERLAY_AST_EXPECTED:
        raise ForensicCorrectionContractError("correction overlay AST drift")
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_overlay_ast.v1"
            ),
            "source_commit": SOURCE_FORENSIC_FREEZE_COMMIT,
            "frozen_forensic_contract": {
                "path": FROZEN_FORENSIC_CONTRACT_PATH,
                "sha256": FROZEN_FORENSIC_CONTRACT_SHA256,
                "git_blob_oid": FROZEN_FORENSIC_CONTRACT_BLOB_OID,
            },
            "frozen_four_key_validator_ast_sha256": (
                FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256
            ),
            "files": files,
            "removed_legacy_nodes": [],
            "all_unlisted_legacy_nodes_unchanged": True,
            "scientific_design_unchanged": True,
            "pass": True,
        }
    )


def build_forensic_correction_amendment() -> dict[str, Any]:
    return BASE.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.forensic_correction_1_amendment.v1",
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "source_forensic_freeze_commit": SOURCE_FORENSIC_FREEZE_COMMIT,
            "failure": {
                "phase": "POST_PREEXECUTION_STAGE_SEQUENCE_DURING_CHILD_RESULT_BUILD",
                "exception": "artifact binding key-set drift",
                "original_primary": "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED",
                "original_secondary": "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH",
                "correction_bug": "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH",
                "scientific_attempt_consumed": False,
            },
            "correction": copy.deepcopy(
                ROW_COUNTED_CONTENT_BINDING_VALIDATOR_AUTHORITY
            ),
            "old_validator_contract": {
                "exported_symbol": "validate_artifact_binding",
                "exact_keys_when_content_digest_required": [
                    "path", "sha256", "bytes", "content_digest"
                ],
                "frozen_module_path": FROZEN_FORENSIC_CONTRACT_PATH,
                "frozen_module_sha256": FROZEN_FORENSIC_CONTRACT_SHA256,
                "frozen_module_git_blob_oid": (
                    FROZEN_FORENSIC_CONTRACT_BLOB_OID
                ),
                "function_ast_sha256": (
                    FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256
                ),
                "unchanged": True,
            },
            "new_validator_contract": copy.deepcopy(
                ROW_COUNTED_CONTENT_BINDING_VALIDATOR_AUTHORITY
            ),
            "scope": {
                "exactly_one_versioned_technical_correction": True,
                "exactly_one_fresh_preexecution_only_diagnostic": True,
                "automatic_retry": False,
                "further_correction_or_diagnostic_authorized": False,
                "scientific_attempt_authorized": False,
                "v2_execution_authorized": False,
            },
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_AUTHORITY_ACCOUNTING
            ),
            "mechanics": {
                "base_launcher_wrapper_child_path_reused": True,
                "base_serializer_streams_cleanup_instrumentation_reused": True,
                "base_six_synthetic_fixtures_and_roles_reused": True,
                "base_four_key_validator_unchanged": True,
                "only_corrected_consumer_selects_five_key_validator": True,
                "semantic_change_limited_to_strict_five_key_selection": True,
                "bounded_root_coordinator_sequences_finalizer_then_checker": True,
                "coordinator_stages_only_exact_nine_result_paths": True,
                "postcommit_replay_is_sole_success_literal_emitter": True,
                "conditional_v2_spec_overlay_preserves_prior_archive_identity_and_new_five_key_corrections": True,
                "frozen_base_v2_specification_bytes_unchanged": True,
            },
            "failed_diagnostic_root": copy.deepcopy(
                FAILED_DIAGNOSTIC_ROOT_AUTHORITY
            ),
            "failed_root_files_reused": 0,
            "fresh_diagnostic_root": str(
                PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
            ),
            "preserved_lineage": copy.deepcopy(PRESERVED_LINEAGE),
            "future_v2_spec_only_command": copy.deepcopy(
                FUTURE_V2_SPEC_ONLY_COMMAND
            ),
            "conditional_success_literals": (
                "DEFINED_AND_EMITTED_ONLY_BY_POSTCOMMIT_REPLAY"
            ),
            "overlay_expected_sha256": BASE.canonical_json_sha256(
                CORRECTION_OVERLAY_AST_EXPECTED
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def build_forensic_correction_allow_list() -> dict[str, Any]:
    return BASE.attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.forensic_correction_1_allow_list.v1",
            "source_forensic_freeze_commit": SOURCE_FORENSIC_FREEZE_COMMIT,
            "exact_changed_paths": list(CORRECTION_REQUIRED_CHANGED_PATHS),
            "allowed_effects": [
                "add strict five-key validator and correction-only callsite",
                "add exact correction freeze/launcher/child/finalizer/checker routing",
                "select a fresh fixed diagnostic root",
                "add one bounded root coordinator that sequences finalizer then checker and captures their exit or failure custody",
                "stage exactly nine correction result paths before the Git publication boundary",
                "add one postcommit-only replay gate as the sole emitter of technical success literals",
                "publish a correction V2 specification overlay that preserves the frozen BASE specification and both exact custody corrections while execution remains false",
            ],
            "required_reuse": [
                "BASE launcher-wrapper-child isolation mechanics",
                "BASE canonical serializer and runtime path names",
                "BASE stdout/stderr/traceback/exception/heartbeat custody",
                "BASE startup ledger/read guard/termination/cleanup",
                "BASE six synthetic fixture IDs, roles, and aggregate schema",
            ],
            "forbidden_effects": [
                "change frozen validate_artifact_binding",
                "change the frozen BASE V2 specification builder or bytes",
                "change V1 science, models, targets, metrics, gates, or data",
                "add a parallel diagnostic launcher, child, synthetic-fixture, stream, I/O, or cleanup engine",
                "use the root coordinator for scientific or diagnostic producer work",
                "read or reuse failed scientific payloads",
                "execute science or V2",
            ],
            "pass": True,
        }
    )


def forensic_correction_authority_payloads() -> dict[Path, bytes]:
    return {
        TRACKED_CORRECTION_AMENDMENT_PATH: _authority_bytes(
            build_forensic_correction_amendment()
        ),
        TRACKED_CORRECTION_FIXTURE_PATH: _authority_bytes(
            build_row_counted_binding_fixture_receipt()
        ),
        TRACKED_CORRECTION_ALLOW_LIST_PATH: _authority_bytes(
            build_forensic_correction_allow_list()
        ),
    }


def build_forensic_correction_source_closure(
    repo_root: str | Path,
    *,
    prospective_payloads: Mapping[Path, bytes] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    supplied = dict(prospective_payloads or {})
    allowed = {Path(path) for path in CORRECTION_SOURCE_CLOSURE_PATHS}
    if (
        any(not isinstance(path, Path) for path in supplied)
        or set(supplied) - allowed
        or any(not isinstance(payload, bytes) for payload in supplied.values())
    ):
        raise ForensicCorrectionContractError(
            "correction prospective closure payload drift"
        )
    rows: list[dict[str, Any]] = []
    for relative_text in CORRECTION_SOURCE_CLOSURE_PATHS:
        relative = Path(relative_text)
        payload = supplied.get(relative)
        if payload is None:
            payload = BASE._read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
        rows.append(
            {
                "path": relative_text,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    if [row["path"] for row in rows] != list(CORRECTION_SOURCE_CLOSURE_PATHS):
        raise ForensicCorrectionContractError("correction closure order drift")
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_SOURCE_CLOSURE_SCHEMA,
            "source_forensic_freeze_commit": SOURCE_FORENSIC_FREEZE_COMMIT,
            "rows": rows,
            "row_count": CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
            "all_rows_technical_or_static": True,
            "scientific_payloads_opened": 0,
            "pass": True,
        }
    )


def validate_forensic_correction_source_closure(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rows = value.get("rows")
    required = {
        "schema", "source_forensic_freeze_commit", "rows", "row_count",
        "all_rows_technical_or_static", "scientific_payloads_opened", "pass",
        "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema") != CORRECTION_SOURCE_CLOSURE_SCHEMA
        or value.get("source_forensic_freeze_commit")
        != SOURCE_FORENSIC_FREEZE_COMMIT
        or not isinstance(rows, list)
        or len(rows) != CORRECTION_SOURCE_CLOSURE_ROW_COUNT
        or value.get("row_count") != CORRECTION_SOURCE_CLOSURE_ROW_COUNT
        or [row.get("path") for row in rows]
        != list(CORRECTION_SOURCE_CLOSURE_PATHS)
        or len({row.get("path") for row in rows})
        != CORRECTION_SOURCE_CLOSURE_ROW_COUNT
        or any(
            type(row) is not dict
            or set(row) != {"path", "sha256", "bytes"}
            or not isinstance(row["sha256"], str)
            or _LOWERCASE_SHA256_RE.fullmatch(row["sha256"]) is None
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
            for row in rows
        )
        or value.get("all_rows_technical_or_static") is not True
        or value.get("scientific_payloads_opened") != 0
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError("correction source closure drift")
    if repo_root is not None and dict(value) != build_forensic_correction_source_closure(
        repo_root
    ):
        raise ForensicCorrectionContractError("live correction closure drift")
    return copy.deepcopy(dict(value))


def correction_source_closure_binding(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    closure = validate_forensic_correction_source_closure(value)
    return validate_row_counted_content_binding(
        BASE._artifact_binding(
            str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
            _authority_bytes(closure),
            content_digest=closure["content_digest"],
            rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
        ),
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )


def load_and_validate_forensic_correction_source_closure(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    payload = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH, expected_stat_rows=None
    )
    closure = BASE._canonical_json_from_bound_bytes(
        payload, label="correction source closure"
    )
    if payload != _authority_bytes(closure):
        raise ForensicCorrectionContractError(
            "correction source closure canonical-byte drift"
        )
    return validate_forensic_correction_source_closure(
        closure, repo_root=root
    )


def _validate_correction_prelaunch_namespace_custody(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "scientific_output_root", "scientific_output_root_absent",
        "scientific_attempt_namespaces", "failed_diagnostic_root",
        "failed_diagnostic_root_immutable_and_nonreusable",
        "fresh_diagnostic_root", "fresh_diagnostic_root_absent",
        "terminal_failure_custody_root",
        "terminal_failure_custody_root_absent",
        "tracked_result_paths_present",
        "other_exact_scientific_or_forensic_process_matches",
        "correction_terminal_coordinator_process_matches",
        "caller_pid_excluded_from_other_process_scan", "pass",
    }
    excluded_pid = value.get(
        "caller_pid_excluded_from_other_process_scan"
    ) if isinstance(value, Mapping) else None
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("scientific_output_root")
        != str(BASE.BASE.OUTPUT_ROOT)
        or value.get("scientific_output_root_absent") is not True
        or value.get("scientific_attempt_namespaces") != []
        or value.get("failed_diagnostic_root")
        != str(FAILED_PREEXECUTION_DIAGNOSTIC_ROOT)
        or value.get("failed_diagnostic_root_immutable_and_nonreusable")
        is not True
        or value.get("fresh_diagnostic_root")
        != str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT)
        or value.get("fresh_diagnostic_root_absent") is not True
        or value.get("terminal_failure_custody_root")
        != str(CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT)
        or value.get("terminal_failure_custody_root_absent") is not True
        or value.get("tracked_result_paths_present") != []
        or value.get(
            "other_exact_scientific_or_forensic_process_matches"
        ) != []
        or value.get("correction_terminal_coordinator_process_matches")
        != []
        or (
            excluded_pid is not None
            and (
                isinstance(excluded_pid, bool)
                or not isinstance(excluded_pid, int)
                or excluded_pid <= 0
            )
        )
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction prelaunch namespace custody drift"
        )
    return copy.deepcopy(dict(value))


def validate_correction_freeze_custody_receipt(
    value: Mapping[str, Any], *, repo_root: str | Path,
    reverify_failed_root: bool,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    BASE.validate_self_digest(value)
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    fixture = validate_row_counted_binding_fixture_receipt(
        value.get("row_counted_binding_fixtures")
    )
    failed = validate_failed_diagnostic_root_custody(
        value.get("failed_diagnostic_root_custody"),
        reverify_live=reverify_failed_root,
    )
    closure = validate_row_counted_content_binding(
        value.get("correction_source_closure"),
        expected_path=str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        expected_rows=CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
    )
    prelaunch_namespace = _validate_correction_prelaunch_namespace_custody(
        value.get("prelaunch_namespace_and_process_custody")
    )
    required = {
        "schema", "experiment_id", "validator_id", "repo_head",
        "sole_parent", "commit_subject", "preserved_lineage",
        "committed_source_root_cause_proof", "overlay_ast_proof",
        "correction_source_closure", "row_counted_binding_fixtures",
        "failed_diagnostic_root_custody", "fresh_diagnostic_root",
        "prelaunch_namespace_and_process_custody",
        "fresh_root_absence_required_by_call", "future_v2_script_present",
        "scientific_counters", "scientific_inputs_opened", "files_reused",
        "pass", "content_digest",
    }
    proof = value.get("committed_source_root_cause_proof")
    overlay = value.get("overlay_ast_proof")
    BASE.validate_self_digest(proof)
    BASE.validate_self_digest(overlay)
    expected_proof = BASE.build_committed_source_root_cause_proof(root)
    expected_overlay = build_correction_overlay_ast_proof(root)
    closure_payload = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH, expected_stat_rows=None
    )
    try:
        closure_authority = json.loads(closure_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicCorrectionContractError(
            "correction closure authority parse drift"
        ) from exc
    if closure_payload != _authority_bytes(closure_authority):
        raise ForensicCorrectionContractError(
            "correction closure authority byte drift"
        )
    validate_forensic_correction_source_closure(
        closure_authority, repo_root=root
    )
    expected_closure = correction_source_closure_binding(closure_authority)
    if (
        set(value) != required
        or value.get("schema") != CORRECTION_FREEZE_CUSTODY_SCHEMA
        or value.get("experiment_id") != BASE.FORENSIC_EXPERIMENT_ID
        or not isinstance(value.get("repo_head"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", value["repo_head"])
        or value.get("validator_id") != ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID
        or value.get("sole_parent") != SOURCE_FORENSIC_FREEZE_COMMIT
        or value.get("commit_subject") != CORRECTION_FREEZE_COMMIT_SUBJECT
        or value.get("preserved_lineage") != PRESERVED_LINEAGE
        or value.get("row_counted_binding_fixtures") != fixture
        or value.get("failed_diagnostic_root_custody") != failed
        or value.get("prelaunch_namespace_and_process_custody")
        != prelaunch_namespace
        or value.get("correction_source_closure") != closure
        or value.get("fresh_diagnostic_root")
        != str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT)
        or value.get("fresh_root_absence_required_by_call") is not True
        or proof != expected_proof
        or proof.get("content_digest")
        != COMMITTED_ROOT_CAUSE_PROOF_CONTENT_DIGEST
        or overlay != expected_overlay
        or value.get("correction_source_closure") != expected_closure
        or value.get("future_v2_script_present") is not False
        or value.get("scientific_inputs_opened") != 0
        or value.get("files_reused") != 0
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError("correction freeze custody drift")
    return copy.deepcopy(dict(value))


def validate_forensic_correction_freeze_custody(
    repo_root: str | Path,
    *,
    require_fresh_root_absent: bool,
    exclude_current_process: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if require_fresh_root_absent is not True:
        raise ForensicCorrectionContractError(
            "pristine correction freeze must require fresh-root absence"
        )
    head = BASE._git_text(root, ("rev-parse", "HEAD"))
    parents = BASE._git_text(root, ("rev-list", "--parents", "-n", "1", head)).split()
    subject = BASE._git_text(root, ("show", "-s", "--format=%s", head))
    status = BASE._git_text(root, ("status", "--porcelain=v1"))
    changed = [
        line.split("\t", 1)[1]
        for line in BASE._git_text(
            root, ("diff", "--name-status", SOURCE_FORENSIC_FREEZE_COMMIT, head)
        ).splitlines()
    ]
    if (
        status
        or len(parents) != 2
        or parents[1] != SOURCE_FORENSIC_FREEZE_COMMIT
        or subject != CORRECTION_FREEZE_COMMIT_SUBJECT
        or sorted(changed) != sorted(CORRECTION_REQUIRED_CHANGED_PATHS)
        or not _anchored_leaf_absent(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT)
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError("correction freeze repository drift")
    for ancestor in PRESERVED_LINEAGE.values():
        if not BASE._git_is_ancestor(root, ancestor, head):
            raise ForensicCorrectionContractError("correction lineage command drift")
    closure_raw = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH, expected_stat_rows=None
    )
    closure = json.loads(closure_raw)
    if closure_raw != _authority_bytes(closure):
        raise ForensicCorrectionContractError("correction closure byte drift")
    validate_forensic_correction_source_closure(closure, repo_root=root)
    source_root_cause_proof = BASE.build_committed_source_root_cause_proof(root)
    prelaunch_namespace = _validate_correction_prepublication_namespaces(
        root, exclude_current_process=exclude_current_process
    )
    value = BASE.attach_self_digest(
        {
            "schema": CORRECTION_FREEZE_CUSTODY_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "repo_head": head,
            "sole_parent": SOURCE_FORENSIC_FREEZE_COMMIT,
            "commit_subject": subject,
            "preserved_lineage": copy.deepcopy(PRESERVED_LINEAGE),
            "committed_source_root_cause_proof": copy.deepcopy(
                source_root_cause_proof
            ),
            "overlay_ast_proof": build_correction_overlay_ast_proof(root),
            "correction_source_closure": correction_source_closure_binding(closure),
            "row_counted_binding_fixtures": build_row_counted_binding_fixture_receipt(),
            "failed_diagnostic_root_custody": build_failed_diagnostic_root_custody(),
            "fresh_diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
            "prelaunch_namespace_and_process_custody": (
                prelaunch_namespace
            ),
            "fresh_root_absence_required_by_call": require_fresh_root_absent,
            "future_v2_script_present": False,
            "scientific_counters": copy.deepcopy(CORRECTION_ZERO_SCIENTIFIC_COUNTERS),
            "scientific_inputs_opened": 0,
            "files_reused": 0,
            "pass": True,
        }
    )
    return validate_correction_freeze_custody_receipt(
        value, repo_root=root, reverify_failed_root=True
    )


def _correction_dirty_paths(repo_root: Path) -> list[str]:
    return BASE._forensic_dirty_paths(repo_root)


def _validate_correction_authority_files(repo_root: Path) -> dict[str, Any]:
    payloads = forensic_correction_authority_payloads()
    for relative, payload in payloads.items():
        if BASE._read_no_follow_bound_file(
            repo_root, relative, expected_stat_rows=None
        ) != payload:
            raise ForensicCorrectionContractError(
                f"correction authority byte drift: {relative}"
            )
    closure_raw = BASE._read_no_follow_bound_file(
        repo_root, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH,
        expected_stat_rows=None,
    )
    closure = BASE._canonical_json_from_bound_bytes(
        closure_raw, label="correction source closure"
    )
    validate_forensic_correction_source_closure(closure, repo_root=repo_root)
    return {
        "correction_source_closure": correction_source_closure_binding(closure),
        "authority_count": len(payloads) + 1,
        "pass": True,
    }


def validate_active_forensic_correction_freeze_custody(
    repo_root: str | Path,
    *,
    forensic_freeze_custody: Mapping[str, Any],
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    """Adapt the frozen BASE active-child checks to the correction root."""

    root = Path(repo_root).resolve()
    active_root = Path(diagnostic_root).absolute()
    freeze = validate_correction_freeze_custody_receipt(
        forensic_freeze_custody,
        repo_root=root,
        reverify_failed_root=False,
    )
    tracked_terminal_paths = sorted(
        {
            *CORRECTION_RESULT_PATHS,
            *(str(path) for path in BASE._forensic_result_paths()),
        }
    )
    present_terminal_paths = [
        path
        for path in tracked_terminal_paths
        if not _anchored_leaf_absent(
            Path(path) if Path(path).is_absolute() else root / path
        )
    ]
    active_coordinators = _active_correction_terminal_coordinators(
        exclude_pids=set()
    )
    terminal_failure_root_absent = _anchored_leaf_absent(
        CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT
    )
    if (
        BASE._git_text(root, ("rev-parse", "HEAD")) != freeze["repo_head"]
        or BASE._git_text(root, ("status", "--porcelain=v1"))
        or active_root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
        or active_root.is_symlink()
        or not active_root.is_dir()
        or BASE.BASE.OUTPUT_ROOT.exists()
        or BASE.BASE.OUTPUT_ROOT.is_symlink()
        or list(
            BASE.BASE.OUTPUT_ROOT.parent.glob(
                f".{BASE.BASE.OUTPUT_ROOT.name}.attempt-*"
            )
        )
        or not _future_v2_script_absent()
        or not terminal_failure_root_absent
        or present_terminal_paths
        or active_coordinators
    ):
        raise ForensicCorrectionContractError(
            "active correction freeze namespace/Git drift"
        )
    authorities = _validate_correction_authority_files(root)
    if authorities["correction_source_closure"] != freeze[
        "correction_source_closure"
    ]:
        raise ForensicCorrectionContractError(
            "active correction authority closure drift"
        )
    base = BASE._validate_base_authorities_read_only(root)
    archives = BASE._validate_archive_custody_metadata_only()
    failed_metadata = _failed_root_metadata_preflight()
    failed_receipt = freeze["failed_diagnostic_root_custody"]
    failed_stat_digest = BASE.canonical_json_sha256(
        [
            failed_metadata["stat_rows"][key]
            for key in sorted(failed_metadata["stat_rows"])
        ]
    )
    if (
        failed_receipt["metadata_stat_row_count"]
        != len(failed_metadata["stat_rows"])
        or failed_receipt["metadata_stat_inventory_sha256"]
        != failed_stat_digest
    ):
        raise ForensicCorrectionContractError(
            "active failed-root metadata custody drift"
        )
    inventory = BASE._validate_active_diagnostic_namespace(active_root)
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "active_forensic_correction_1_freeze_custody.v1"
            ),
            "source_freeze_commit": freeze["repo_head"],
            "forensic_freeze_custody": copy.deepcopy(freeze),
            "correction_authority_validated": True,
            "base_authorities_read_only_validated": base["pass"],
            "archive_custody_metadata_only_validated": archives["pass"],
            "failed_root_metadata_only_validated": True,
            "active_diagnostic_root": str(active_root),
            "pristine_root_absence_bound_by_prelaunch_freeze": (
                freeze["fresh_root_absence_required_by_call"] is True
            ),
            "active_diagnostic_namespace_inventory": inventory,
            "active_terminal_namespace_custody": {
                "terminal_failure_custody_root": str(
                    CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT
                ),
                "terminal_failure_custody_root_absent": True,
                "tracked_result_paths_present": [],
                "correction_terminal_coordinator_process_matches": [],
                "future_v2_script_present": False,
                "pass": True,
            },
            "scientific_output_namespace_absent": True,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "scientific_inputs_opened": 0,
            "files_reused": 0,
            "pass": True,
        }
    )


def _validate_correction_prepublication_namespaces(
    repo_root: Path, *, exclude_current_process: bool
) -> dict[str, Any]:
    scientific_root = BASE.BASE.OUTPUT_ROOT
    attempts = sorted(
        str(path.absolute())
        for path in scientific_root.parent.glob(
            f".{scientific_root.name}.attempt-*"
        )
    )
    tracked = sorted(
        {
            *CORRECTION_RESULT_PATHS,
            *(str(path) for path in BASE._forensic_result_paths()),
        }
    )
    present = [
        path
        for path in tracked
        if (repo_root / path).exists() or (repo_root / path).is_symlink()
    ]
    excluded = {os.getpid()} if exclude_current_process else set()
    processes = BASE._active_forensic_or_scientific_processes(
        repo_root, exclude_pids=excluded
    )
    coordinators = _active_correction_terminal_coordinators(
        exclude_pids=excluded
    )
    if (
        scientific_root.exists()
        or scientific_root.is_symlink()
        or attempts
        or not _anchored_leaf_absent(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT)
        or not _anchored_leaf_absent(CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT)
        or present
        or processes
        or coordinators
    ):
        raise ForensicCorrectionContractError(
            "correction prepublication namespace/process drift"
        )
    return {
        "scientific_output_root": str(scientific_root),
        "scientific_output_root_absent": True,
        "scientific_attempt_namespaces": [],
        "failed_diagnostic_root": str(FAILED_PREEXECUTION_DIAGNOSTIC_ROOT),
        "failed_diagnostic_root_immutable_and_nonreusable": True,
        "fresh_diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
        "fresh_diagnostic_root_absent": True,
        "terminal_failure_custody_root": str(
            CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT
        ),
        "terminal_failure_custody_root_absent": True,
        "tracked_result_paths_present": [],
        "other_exact_scientific_or_forensic_process_matches": [],
        "correction_terminal_coordinator_process_matches": [],
        "caller_pid_excluded_from_other_process_scan": (
            os.getpid() if exclude_current_process else None
        ),
        "pass": True,
    }


def validate_forensic_correction_freeze_preparation(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate prospective correction bytes before any authority write."""

    root = Path(repo_root).resolve()
    head = BASE._git_text(root, ("rev-parse", "HEAD"))
    changed = _correction_dirty_paths(root)
    if (
        head != SOURCE_FORENSIC_FREEZE_COMMIT
        or changed != sorted(CORRECTION_CODE_AND_TEST_PATHS)
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError(
            "correction freeze preparation repository drift"
        )
    stale = [
        str(path)
        for path in CORRECTION_GENERATED_AUTHORITY_PATHS
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if stale:
        raise ForensicCorrectionContractError(
            f"correction authority paths are stale: {stale}"
        )
    for relative in CORRECTION_CODE_AND_TEST_PATHS:
        BASE._read_no_follow_bound_file(root, relative, expected_stat_rows=None)
    namespace = _validate_correction_prepublication_namespaces(
        root, exclude_current_process=True
    )
    failed = build_failed_diagnostic_root_custody()
    overlay = build_correction_overlay_ast_proof(root)
    proof = BASE.build_committed_source_root_cause_proof(root)
    payloads = forensic_correction_authority_payloads()
    closure = build_forensic_correction_source_closure(
        root, prospective_payloads=payloads
    )
    authorities = {
        str(path): BASE._artifact_binding(str(path), payload)
        for path, payload in payloads.items()
    }
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_FREEZE_PREPARATION_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "freeze_mode": "PREEXECUTION_FORENSIC_CORRECTION_1_PREPARATION",
            "base_head": head,
            "changed_paths": changed,
            "prospective_authorities": authorities,
            "prospective_source_closure": closure,
            "namespace_and_process_custody": namespace,
            "failed_diagnostic_root_custody": failed,
            "committed_source_root_cause_proof": proof,
            "overlay_ast_proof": overlay,
            "preserved_lineage": copy.deepcopy(PRESERVED_LINEAGE),
            "one_versioned_correction_authorized": True,
            "one_fresh_preexecution_diagnostic_authorized": True,
            "automatic_retry_authorized": False,
            "future_v2_script_present": False,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "scientific_inputs_opened": 0,
            "files_reused": 0,
            "required_enclosing_commit_subject": (
                CORRECTION_FREEZE_COMMIT_SUBJECT
            ),
            "pass": True,
        }
    )


def write_forensic_correction_authorities(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Exclusively write the four correction authorities all-or-absent."""

    root = Path(repo_root).resolve()
    payloads = forensic_correction_authority_payloads()
    closure = build_forensic_correction_source_closure(
        root, prospective_payloads=payloads
    )
    closure_payload = _authority_bytes(closure)
    intended = [*payloads, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH]
    stale = [
        str(path)
        for path in intended
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if stale:
        raise ForensicCorrectionContractError(
            f"correction authority writer refuses stale paths: {stale}"
        )
    expected = {**payloads, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH: closure_payload}
    try:
        for relative, payload in expected.items():
            BASE._exclusive_write(root / relative, payload)
        for relative, payload in expected.items():
            if BASE._read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            ) != payload:
                raise ForensicCorrectionContractError(
                    f"correction authority post-write byte drift: {relative}"
                )
    except BaseException as original:
        try:
            BASE._rollback_relative_paths_no_follow(
                root, intended, label="forensic correction authority publication"
            )
        except BaseException as cleanup_exc:
            raise ForensicCorrectionContractError(
                "correction authority publication failed and rollback could "
                "not prove all intended paths absent"
            ) from cleanup_exc
        raise
    return {
        "authorities": {
            str(path): BASE._artifact_binding(str(path), payload)
            for path, payload in payloads.items()
        },
        "source_closure": correction_source_closure_binding(closure),
    }


def rollback_forensic_correction_authorities(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if BASE._git_text(root, ("rev-parse", "HEAD")) != SOURCE_FORENSIC_FREEZE_COMMIT:
        raise ForensicCorrectionContractError(
            "correction authority rollback is allowed only at fa80"
        )
    removed = BASE._rollback_relative_paths_no_follow(
        root,
        [Path(path) for path in CORRECTION_GENERATED_AUTHORITY_PATHS],
        label="forensic correction authority postflight",
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_authority_rollback.v1"
            ),
            "base_head": SOURCE_FORENSIC_FREEZE_COMMIT,
            "intended_paths": list(CORRECTION_GENERATED_AUTHORITY_PATHS),
            "removed_paths": removed,
            "all_intended_paths_absent": True,
            "pass": True,
        }
    )


def validate_forensic_correction_authority_write(
    repo_root: str | Path,
    *,
    preparation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    BASE.validate_self_digest(preparation_receipt)
    required = {
        "schema", "experiment_id", "freeze_mode", "base_head",
        "changed_paths", "prospective_authorities",
        "prospective_source_closure", "namespace_and_process_custody",
        "failed_diagnostic_root_custody", "committed_source_root_cause_proof",
        "overlay_ast_proof", "preserved_lineage",
        "one_versioned_correction_authorized",
        "one_fresh_preexecution_diagnostic_authorized",
        "automatic_retry_authorized", "future_v2_script_present",
        "scientific_counters", "scientific_inputs_opened", "files_reused",
        "required_enclosing_commit_subject", "pass", "content_digest",
    }
    payloads = forensic_correction_authority_payloads()
    authority_bindings = {
        str(path): BASE._artifact_binding(str(path), payload)
        for path, payload in payloads.items()
    }
    closure_payload = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH, expected_stat_rows=None
    )
    try:
        closure = json.loads(closure_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicCorrectionContractError(
            "written correction closure parse drift"
        ) from exc
    if closure_payload != _authority_bytes(closure):
        raise ForensicCorrectionContractError(
            "written correction closure byte drift"
        )
    validate_forensic_correction_source_closure(closure, repo_root=root)
    for relative, payload in payloads.items():
        if BASE._read_no_follow_bound_file(
            root, relative, expected_stat_rows=None
        ) != payload:
            raise ForensicCorrectionContractError(
                f"written correction authority drift: {relative}"
            )
    current_closure = build_forensic_correction_source_closure(root)
    failed = build_failed_diagnostic_root_custody()
    overlay = build_correction_overlay_ast_proof(root)
    proof = BASE.build_committed_source_root_cause_proof(root)
    namespace = _validate_correction_prepublication_namespaces(
        root, exclude_current_process=True
    )
    if (
        set(preparation_receipt) != required
        or preparation_receipt.get("schema")
        != CORRECTION_FREEZE_PREPARATION_SCHEMA
        or preparation_receipt.get("experiment_id") != BASE.FORENSIC_EXPERIMENT_ID
        or preparation_receipt.get("freeze_mode")
        != "PREEXECUTION_FORENSIC_CORRECTION_1_PREPARATION"
        or preparation_receipt.get("base_head") != SOURCE_FORENSIC_FREEZE_COMMIT
        or preparation_receipt.get("changed_paths")
        != sorted(CORRECTION_CODE_AND_TEST_PATHS)
        or preparation_receipt.get("prospective_authorities")
        != authority_bindings
        or preparation_receipt.get("prospective_source_closure") != closure
        or closure != current_closure
        or preparation_receipt.get("namespace_and_process_custody")
        != namespace
        or preparation_receipt.get("failed_diagnostic_root_custody") != failed
        or preparation_receipt.get("committed_source_root_cause_proof") != proof
        or preparation_receipt.get("overlay_ast_proof") != overlay
        or preparation_receipt.get("preserved_lineage") != PRESERVED_LINEAGE
        or preparation_receipt.get("one_versioned_correction_authorized") is not True
        or preparation_receipt.get(
            "one_fresh_preexecution_diagnostic_authorized"
        ) is not True
        or preparation_receipt.get("automatic_retry_authorized") is not False
        or preparation_receipt.get("future_v2_script_present") is not False
        or preparation_receipt.get("scientific_inputs_opened") != 0
        or preparation_receipt.get("files_reused") != 0
        or preparation_receipt.get("required_enclosing_commit_subject")
        != CORRECTION_FREEZE_COMMIT_SUBJECT
        or preparation_receipt.get("pass") is not True
        or BASE._git_text(root, ("rev-parse", "HEAD"))
        != SOURCE_FORENSIC_FREEZE_COMMIT
        or _correction_dirty_paths(root)
        != sorted(CORRECTION_REQUIRED_CHANGED_PATHS)
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError(
            "correction authority post-write custody drift"
        )
    validate_correction_zero_scientific_counters(
        preparation_receipt.get("scientific_counters")
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_AUTHORITY_WRITE_SCHEMA,
            "base_head": SOURCE_FORENSIC_FREEZE_COMMIT,
            "changed_paths": sorted(CORRECTION_REQUIRED_CHANGED_PATHS),
            "authorities": authority_bindings,
            "source_closure": correction_source_closure_binding(closure),
            "source_closure_rows": CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
            "authority_custody": {
                "authority_count": len(payloads) + 1,
                "all_bytes_exact": True,
                "prospective_and_written_closure_equal": True,
                "pass": True,
            },
            "failed_diagnostic_root_custody": failed,
            "overlay_ast_proof": overlay,
            "scientific_inputs_opened": 0,
            "scientific_archive_payload_files_opened": 0,
            "files_reused": 0,
            "required_enclosing_commit_subject": (
                CORRECTION_FREEZE_COMMIT_SUBJECT
            ),
            "pass": True,
        }
    )


def freeze_preexecution_forensic_correction_1_contract(
    repo_root: str | Path = CORRECTION_REPO_ROOT,
) -> dict[str, Any]:
    """Perform only the authorized correction-authority publication."""

    preparation = validate_forensic_correction_freeze_preparation(repo_root)
    try:
        written = write_forensic_correction_authorities(repo_root)
        post = validate_forensic_correction_authority_write(
            repo_root, preparation_receipt=preparation
        )
        if (
            written["authorities"] != post["authorities"]
            or written["source_closure"] != post["source_closure"]
        ):
            raise ForensicCorrectionContractError(
                "correction authority publication cross-binding drift"
            )
    except BaseException as original:
        try:
            rollback_forensic_correction_authorities(repo_root)
        except BaseException as cleanup_exc:
            raise ForensicCorrectionContractError(
                "correction authority rollback failed"
            ) from cleanup_exc
        raise
    return {
        "preparation": preparation,
        "written": written,
        "postwrite": post,
        "pass": True,
    }


def derive_correction_zero_scientific_counters(
    *,
    base_scientific_counters: Mapping[str, Any],
    read_guard_event_bytes: int,
    scientific_namespace_paths: Sequence[str],
) -> dict[str, int]:
    if (
        set(base_scientific_counters) != set(BASE.ZERO_SCIENTIFIC_COUNTERS)
        or any(
            isinstance(base_scientific_counters[key], bool)
            or not isinstance(base_scientific_counters[key], int)
            or base_scientific_counters[key] != 0
            for key in BASE.ZERO_SCIENTIFIC_COUNTERS
        )
        or isinstance(read_guard_event_bytes, bool)
        or read_guard_event_bytes != 0
        or list(scientific_namespace_paths) != []
    ):
        raise ForensicCorrectionContractError(
            "correction zero-science projection drift"
        )
    return copy.deepcopy(CORRECTION_ZERO_SCIENTIFIC_COUNTERS)


def build_correction_diagnostic_custody_receipt(
    *,
    repo_root: str | Path,
    source_commit: str,
    correction_source_closure: Mapping[str, Any],
    diagnostic_root: str | Path,
    launcher_process_identity: Mapping[str, Any],
    child_process_identity: Mapping[str, Any],
    forensic_freeze_custody_receipt: Mapping[str, Any],
    umask_custody_receipt: Mapping[str, Any],
    invocation_receipt: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    command_receipt: Mapping[str, Any],
    read_guard_manifest: Mapping[str, Any],
    os_evidence_receipt: Mapping[str, Any],
    preexecution_only_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    corrected_child_result: Mapping[str, Any],
    last_stage_marker_value: Mapping[str, Any],
    startup_stage_rows: Sequence[Mapping[str, Any]],
    stream_bindings: Mapping[str, Any],
    exception_observed: bool,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    launcher = BASE.validate_process_identity(launcher_process_identity)
    child = BASE.validate_process_identity(child_process_identity)
    freeze = validate_correction_freeze_custody_receipt(
        forensic_freeze_custody_receipt,
        repo_root=repo_root,
        reverify_failed_root=False,
    )
    closure = validate_forensic_correction_source_closure(
        correction_source_closure
    )
    umask = BASE.validate_diagnostic_umask_custody(umask_custody_receipt)
    invocation = BASE.validate_invocation_receipt(invocation_receipt)
    environment = BASE.validate_environment_receipt(environment_receipt)
    command = BASE.validate_command_receipt(command_receipt)
    guard = BASE.validate_read_guard_manifest(read_guard_manifest, repo_root=repo_root)
    os_evidence = BASE.validate_os_evidence_receipt(os_evidence_receipt)
    preexecution = BASE.validate_preexecution_only_receipt(
        preexecution_only_receipt
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        synthetic_results_receipt
    )
    corrected = validate_corrected_preexecution_child_result(
        corrected_child_result,
        repo_root=repo_root,
        forensic_freeze_custody=freeze,
    )
    marker = BASE.validate_last_stage_marker(last_stage_marker_value)
    rows = BASE.validate_startup_stage_rows(startup_stage_rows, require_complete=True)
    streams = BASE._validate_stream_bindings(stream_bindings)
    expected_internal_argv = expected_correction_child_inner_argv(
        launcher_pid=launcher["pid"],
        launcher_start_time_ticks=launcher["start_time_ticks"],
    )
    counters = derive_correction_zero_scientific_counters(
        base_scientific_counters=preexecution["scientific_counters"],
        read_guard_event_bytes=streams["read_guard_events"]["bytes"],
        scientific_namespace_paths=[],
    )
    if (
        root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
        or not isinstance(source_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or source_commit != freeze["repo_head"]
        or invocation["source_commit"] != source_commit
        or invocation["diagnostic_root"] != str(root)
        or invocation["launcher_process_identity"] != launcher
        or invocation["child_process_identity"] != child
        or command["outer_exact_argv"] != child["argv"]
        or command["internal_exact_argv"] != expected_internal_argv
        or invocation["internal_exact_argv"] != expected_internal_argv
        or command["internal_exact_argv"]
        != invocation["internal_exact_argv"]
        or command["fd_custody"] != invocation["fd_custody"]
        or command["cwd"] != str(Path(repo_root).resolve())
        or command["popen_environment"]
        != environment["result_environment_custody"]
        or os_evidence["launcher_process_identity"] != launcher
        or os_evidence["child_process_identity"] != child
        or os_evidence["namespace_before"] != preexecution["namespace_before"]
        or os_evidence["namespace_after"] != preexecution["namespace_after"]
        or os_evidence["returncode"] != 0
        or os_evidence["pass"] is not True
        or preexecution["pass"] is not True
        or synthetic["pass"] is not True
        or corrected["launcher_process_identity"] != launcher
        or corrected["child_process_identity"] != child
        or corrected["repo_head"] != source_commit
        or corrected["forensic_authority_source_closure"]
        != freeze["correction_source_closure"]
        or corrected["forensic_freeze_custody"]
        != _runtime_json_binding("forensic_freeze_custody", freeze)
        or correction_source_closure_binding(closure)
        != freeze["correction_source_closure"]
        or invocation["technical_runtime_context"]
        != os_evidence["current_runtime_context"]
        or invocation["technical_runtime_context"]["umask"]["value"]
        != BASE.PREEXECUTION_DIAGNOSTIC_UMASK
        or corrected["current_runtime_context"]["umask"]["value"]
        != BASE.PREEXECUTION_DIAGNOSTIC_UMASK
        or marker["producer_role"] != "PREEXECUTION_DIAGNOSTIC_CHILD"
        or marker["stage_id"] != "COMPLETE"
        or marker["event"] != "COMPLETED"
        or marker["pid"] != child["pid"]
        or not isinstance(exception_observed, bool)
        or exception_observed is not False
        or streams["read_guard_events"]["bytes"] != 0
        or streams["heartbeat"]["bytes"] <= 0
        or streams["stderr"]["bytes"] != 0
        or streams["traceback"]["bytes"] != 0
        or streams["exception"]["bytes"] != 0
        or corrected["scientific_counters"] != counters
    ):
        raise ForensicCorrectionContractError(
            "correction diagnostic custody cross-binding drift"
        )
    child_payload = _authority_bytes(corrected)
    if (
        streams["stdout"]["sha256"] != hashlib.sha256(child_payload).hexdigest()
        or streams["stdout"]["bytes"] != len(child_payload)
    ):
        raise ForensicCorrectionContractError("corrected child stdout drift")
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_DIAGNOSTIC_CUSTODY_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC_CORRECTION_1",
            "source_commit": source_commit,
            "forensic_contract": copy.deepcopy(BASE.FORENSIC_CONTRACT_BINDING),
            "forensic_source_closure": copy.deepcopy(closure),
            "diagnostic_root": str(root),
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "exact_argv": copy.deepcopy(child["argv"]),
            "receipts": {
                "forensic_freeze_custody": _runtime_json_binding(
                    "forensic_freeze_custody", freeze
                ),
                "umask_custody": _runtime_json_binding("umask_custody", umask),
                "invocation": _runtime_json_binding("invocation", invocation),
                "environment": _runtime_json_binding("environment", environment),
                "command": _runtime_json_binding("command", command),
                "read_guard_manifest": _runtime_json_binding(
                    "read_guard_manifest", guard
                ),
                "os_evidence": _runtime_json_binding("os_evidence", os_evidence),
                "preexecution_only": _runtime_json_binding(
                    "preexecution_only", preexecution
                ),
                "synthetic_results": _runtime_json_binding(
                    "synthetic_results", synthetic
                ),
                "startup_stage_ledger": BASE._startup_ledger_binding(rows),
                "last_stage_marker": _runtime_json_binding(
                    "last_stage_marker", marker
                ),
            },
            "streams": streams,
            "preexecution_child_result": corrected,
            "last_stage_marker_value": marker,
            "launcher_technical_runtime_context": copy.deepcopy(
                invocation["technical_runtime_context"]
            ),
            "child_technical_runtime_context": copy.deepcopy(
                corrected["current_runtime_context"]
            ),
            "exception_observed": False,
            "started_monotonic_ns": os_evidence["started_monotonic_ns"],
            "ended_monotonic_ns": os_evidence["ended_monotonic_ns"],
            "runtime_ns": os_evidence["runtime_ns"],
            "last_started_stage": BASE.PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1],
            "last_completed_stage": BASE.PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1],
            "returncode": 0,
            "termination": copy.deepcopy(os_evidence["termination"]),
            "cleanup": copy.deepcopy(os_evidence["cleanup"]),
            "process_state_scope_at_custody_write": {
                "launcher_process_identity": copy.deepcopy(launcher),
                "launcher_live_at_custody_write": True,
                "cleanup_scope": os_evidence["cleanup"]["cleanup_scope"],
                "child_process_group_and_nonlauncher_roles_zero": True,
                "literal_zero_all_forensic_roles_claimed": False,
                "all_forensic_role_zero_validation": (
                    "REQUIRED_AFTER_LAUNCHER_EXIT_BY_POSTCOMMIT_VALIDATOR"
                ),
            },
            "attempt_namespace_created": False,
            "attempt_reservation_created": False,
            "canonical_or_tracked_written": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": counters,
            "files_reused": 0,
            "namespace_before": copy.deepcopy(preexecution["namespace_before"]),
            "namespace_after": copy.deepcopy(preexecution["namespace_after"]),
            "namespace_unchanged": True,
            "pass_meaning": (
                "TECHNICAL_CUSTODY_COMPLETE_NOT_SCIENTIFIC_SUCCESS"
            ),
            "pass": True,
        }
    )


def validate_correction_diagnostic_custody_receipt(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    source_commit: str,
    correction_source_closure: Mapping[str, Any],
    diagnostic_root: str | Path,
    launcher_process_identity: Mapping[str, Any],
    child_process_identity: Mapping[str, Any],
    forensic_freeze_custody_receipt: Mapping[str, Any],
    umask_custody_receipt: Mapping[str, Any],
    invocation_receipt: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    command_receipt: Mapping[str, Any],
    read_guard_manifest: Mapping[str, Any],
    os_evidence_receipt: Mapping[str, Any],
    preexecution_only_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    corrected_child_result: Mapping[str, Any],
    last_stage_marker_value: Mapping[str, Any],
    startup_stage_rows: Sequence[Mapping[str, Any]],
    stream_bindings: Mapping[str, Any],
    exception_observed: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_diagnostic_custody_receipt(
        repo_root=repo_root,
        source_commit=source_commit,
        correction_source_closure=correction_source_closure,
        diagnostic_root=diagnostic_root,
        launcher_process_identity=launcher_process_identity,
        child_process_identity=child_process_identity,
        forensic_freeze_custody_receipt=forensic_freeze_custody_receipt,
        umask_custody_receipt=umask_custody_receipt,
        invocation_receipt=invocation_receipt,
        environment_receipt=environment_receipt,
        command_receipt=command_receipt,
        read_guard_manifest=read_guard_manifest,
        os_evidence_receipt=os_evidence_receipt,
        preexecution_only_receipt=preexecution_only_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
        corrected_child_result=corrected_child_result,
        last_stage_marker_value=last_stage_marker_value,
        startup_stage_rows=startup_stage_rows,
        stream_bindings=stream_bindings,
        exception_observed=exception_observed,
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError("correction custody drift")
    return copy.deepcopy(dict(value))


def _validate_correction_diagnostic_custody_envelope(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the persisted envelope without reopening its bound sources."""

    BASE.validate_self_digest(value)
    base_fields = {
        "schema", "experiment_id", "mode", "source_commit",
        "forensic_contract", "forensic_source_closure", "diagnostic_root",
        "launcher_process_identity", "child_process_identity", "exact_argv",
        "receipts", "streams", "preexecution_child_result",
        "last_stage_marker_value", "launcher_technical_runtime_context",
        "child_technical_runtime_context", "exception_observed",
        "started_monotonic_ns", "ended_monotonic_ns", "runtime_ns",
        "last_started_stage", "last_completed_stage", "returncode",
        "termination", "cleanup", "process_state_scope_at_custody_write",
        "attempt_namespace_created", "attempt_reservation_created",
        "canonical_or_tracked_written",
        "technical_diagnostic_attempt_accounting", "scientific_counters",
        "files_reused",
        "namespace_before", "namespace_after", "namespace_unchanged",
        "pass_meaning", "pass", "content_digest",
    }
    child = value.get("preexecution_child_result")
    if not isinstance(child, Mapping):
        raise ForensicCorrectionContractError("corrected custody child absent")
    BASE.validate_self_digest(child)
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    if (
        set(value) != base_fields
        or value.get("schema") != CORRECTION_DIAGNOSTIC_CUSTODY_SCHEMA
        or value.get("experiment_id") != BASE.FORENSIC_EXPERIMENT_ID
        or value.get("mode") != "PREEXECUTION_ONLY_DIAGNOSTIC_CORRECTION_1"
        or value.get("forensic_contract") != BASE.FORENSIC_CONTRACT_BINDING
        or value.get("diagnostic_root")
        != str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT)
        or child.get("schema") != CORRECTED_CHILD_RESULT_SCHEMA
        or child.get("validator_id") != ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID
        or value.get("attempt_namespace_created") is not False
        or value.get("attempt_reservation_created") is not False
        or value.get("canonical_or_tracked_written") is not False
        or value.get("namespace_before") != value.get("namespace_after")
        or value.get("namespace_unchanged") is not True
        or value.get("returncode") != 0
        or value.get("exception_observed") is not False
        or value.get("files_reused") != 0
        or value.get("pass_meaning")
        != "TECHNICAL_CUSTODY_COMPLETE_NOT_SCIENTIFIC_SUCCESS"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction diagnostic custody envelope drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_conditional_v2_decision(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    custody = _validate_correction_diagnostic_custody_envelope(
        diagnostic_custody_receipt
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        synthetic_results_receipt
    )
    child = custody["preexecution_child_result"]
    fixtures = validate_row_counted_binding_fixture_receipt(
        child["row_counted_binding_fixtures"]
    )
    terminal = validate_row_counted_terminal_binding_receipt(
        child["terminal_binding_receipt"]
    )
    counters = validate_correction_zero_scientific_counters(
        custody["scientific_counters"]
    )
    cleanup = custody["cleanup"]
    conditions = {
        "forensic_authority_and_source_closure_frozen": (
            child["forensic_authority_source_closure"]
            == terminal["source_closure"]
            and child["forensic_freeze_custody"]
            == custody["receipts"]["forensic_freeze_custody"]
        ),
        "preexecution_diagnostic_complete": custody["pass"] is True,
        "startup_stage_sequence_complete": (
            custody["last_completed_stage"]
            == BASE.PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1]
        ),
        "invocation_os_stream_and_exception_custody_complete": (
            custody["exception_observed"] is False
            and custody["streams"]["stderr"]["bytes"] == 0
            and custody["streams"]["traceback"]["bytes"] == 0
            and custody["streams"]["exception"]["bytes"] == 0
            and custody["streams"]["heartbeat"]["bytes"] > 0
            and custody["streams"]["read_guard_events"]["bytes"] == 0
        ),
        "all_synthetic_failure_modes_captured_and_cleaned": (
            synthetic["row_count"]
            == len(BASE.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES)
            and synthetic["all_capture_and_cleanup_pass"] is True
        ),
        "exact_technical_root_cause_reproduced": (
            child["mismatch_evidence"]["pass"] is True
            and child["mismatch_evidence"]["identity_projection_equality"] is True
            and child["mismatch_evidence"]["raw_record_equality"] is False
        ),
        "cause_specific_correction_specified_and_regression_passes": (
            fixtures["all_pass"] is True and terminal["pass"] is True
        ),
        "no_scientific_attempt_or_scientific_reservation_namespace_created": (
            custody["attempt_namespace_created"] is False
            and custody["attempt_reservation_created"] is False
        ),
        "no_canonical_or_tracked_scientific_output_written": (
            custody["canonical_or_tracked_written"] is False
        ),
        "zero_scientific_inputs_outcomes_tensors_models_and_training": (
            all(value == 0 for value in counters.values())
        ),
        "original_scientific_contract_byte_and_digest_immutable": (
            child["scientific_contract_digest"]
            == BASE.BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        ),
        "all_failed_archive_files_reused_zero": custody["files_reused"] == 0,
        "scientific_namespace_and_child_process_state_clean_launcher_disclosed_live": (
            custody["namespace_unchanged"] is True
            and cleanup["process_group_members_after_wait"] == []
            and cleanup["exact_nonlauncher_forensic_role_matches_after_wait"] == []
            and cleanup["scoped_dev_kfd_holders_after_wait"] == []
            and cleanup["current_launcher_excluded_from_role_scan"] is True
            and cleanup["literal_zero_all_forensic_roles_claimed"] is False
        ),
        "explicit_human_or_stakeholder_authorization": (
            BASE.USER_CONDITIONAL_V2_SPEC_AUTHORITY[
                "specification_authorized_conditionally"
            ] is True
            and BASE.USER_CONDITIONAL_V2_SPEC_AUTHORITY[
                "execution_authorized"
            ] is False
            and BASE.USER_CONDITIONAL_V2_SPEC_AUTHORITY[
                "scientific_attempt_authorized"
            ] is False
        ),
    }
    if tuple(conditions) != tuple(BASE.CONDITIONAL_V2_REQUIRED_CONDITIONS):
        raise ForensicCorrectionContractError("correction V2 condition order drift")
    gate = BASE.evaluate_conditional_v2_gate(
        root_cause_classification=BASE.FORENSIC_PRIMARY_CLASSIFICATION,
        conditions=conditions,
    )
    if gate["v2_spec_authorized"] is not True or synthetic["pass"] is not True:
        raise ForensicCorrectionContractError("correction V2 specification gate failed")
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "conditional_v2_spec_decision.correction_1.v1"
            ),
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "gate_authority": copy.deepcopy(BASE.CONDITIONAL_V2_GATE_BINDING),
            "user_specification_authority": copy.deepcopy(
                BASE.USER_CONDITIONAL_V2_SPEC_AUTHORITY
            ),
            "primary_forensic_classification": (
                "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED"
            ),
            "secondary_mechanism": "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH",
            "correction_bug": (
                "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
            ),
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "gate_result": gate,
            "v2_spec_authorized": True,
            "automatic_execution_authorized": False,
            "scientific_contract_change_authorized": False,
            "scientific_classification_authorized": False,
            "pass": True,
        }
    )


def validate_correction_conditional_v2_decision(
    value: Mapping[str, Any],
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_conditional_v2_decision(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError("correction V2 decision drift")
    return copy.deepcopy(dict(value))


def _correction_v2_technical_corrections() -> list[dict[str, Any]]:
    return [
        {
            "correction_id": BASE.FORENSIC_DEFECT_ID,
            "original_invalid_contract": (
                "WHOLE_RECORD_ARCHIVE_CUSTODY_EQUALITY"
            ),
            "required_contract": "STABLE_ARCHIVE_IDENTITY_PROJECTION",
            "preserved_from_frozen_base_v2_specification": True,
            "scientific_payload_or_logic_changed": False,
        },
        {
            "correction_id": (
                "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
            ),
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "validator_symbol": (
                "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1"
            ),
            "exact_keys": [
                "path", "sha256", "bytes", "content_digest", "rows"
            ],
            "expected_path": str(TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
            "expected_rows": CORRECTION_SOURCE_CLOSURE_ROW_COUNT,
            "production_selectors": {
                "corrected_child_producer": (
                    "produce_row_counted_content_binding"
                ),
                "terminal_validator": (
                    "validate_terminal_row_counted_content_binding"
                ),
                "independent_external_checker": (
                    "check_external_row_counted_content_binding"
                ),
                "terminal_receipt_external_checker": (
                    "check_external_terminal_binding_receipt"
                ),
            },
            "ordinary_four_key_validator_unchanged": True,
            "scientific_payload_or_logic_changed": False,
        },
    ]


def build_correction_conditional_v2_spec_authority(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(decision)
    if (
        decision.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "conditional_v2_spec_decision.correction_1.v1"
        )
        or decision.get("v2_spec_authorized") is not True
        or decision.get("automatic_execution_authorized") is not False
        or decision.get("scientific_contract_change_authorized") is not False
    ):
        raise ForensicCorrectionContractError(
            "correction V2 specification decision drift"
        )
    frozen_base = BASE.build_conditional_v2_spec_authority(decision)
    BASE.validate_conditional_v2_spec_authority(
        frozen_base, decision=decision
    )
    frozen_base_markdown = BASE.build_conditional_v2_spec_markdown(
        frozen_base
    )
    BASE.validate_conditional_v2_spec_markdown(
        frozen_base_markdown, specification=frozen_base
    )
    clauses = copy.deepcopy(frozen_base["clauses"])
    matching = [
        index
        for index, clause in enumerate(clauses)
        if clause.get("id") == "EXACT_TECHNICAL_CORRECTION_ONLY"
    ]
    if matching != [8]:
        raise ForensicCorrectionContractError(
            "frozen V2 technical-correction clause position drift"
        )
    clauses[matching[0]] = {
        "id": "EXACT_TECHNICAL_CORRECTION_ONLY",
        "requirement": (
            "Preserve the prior replacement of incompatible whole-record "
            "archive-custody equality with the stable archive-identity "
            "projection, and additionally require the exact row-counted "
            "five-key {path, sha256, bytes, content_digest, rows} source-"
            "closure binding at the corrected child producer, terminal "
            "validator, and independent external checker. Models, targets, "
            "metrics, gates, and all other scientific logic remain unchanged."
        ),
    }
    technical_corrections = _correction_v2_technical_corrections()
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_V2_SPECIFICATION_SCHEMA,
            "source_experiment": BASE.FORENSIC_EXPERIMENT_ID,
            "gate_decision": copy.deepcopy(dict(decision)),
            "frozen_base_specification_authority": frozen_base,
            "frozen_base_specification_content_digest": frozen_base[
                "content_digest"
            ],
            "frozen_base_specification_markdown": frozen_base_markdown,
            "frozen_base_specification_markdown_sha256": hashlib.sha256(
                frozen_base_markdown.encode("utf-8")
            ).hexdigest(),
            "base_specification_bytes_unchanged": True,
            "user_specification_authority": copy.deepcopy(
                BASE.USER_CONDITIONAL_V2_SPEC_AUTHORITY
            ),
            "primary_forensic_classification": (
                BASE.FORENSIC_PRIMARY_CLASSIFICATION
            ),
            "secondary_mechanism": BASE.FORENSIC_SECONDARY_MECHANISM,
            "technical_defect_id": BASE.FORENSIC_DEFECT_ID,
            "correction_defect_id": (
                "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
            ),
            "clauses": clauses,
            "clause_ids": [row["id"] for row in clauses],
            "exact_technical_corrections": technical_corrections,
            "exact_technical_correction_count": len(technical_corrections),
            "row_counted_validator_authority": copy.deepcopy(
                ROW_COUNTED_CONTENT_BINDING_VALIDATOR_AUTHORITY
            ),
            "preserved_v1_scientific_dimensions": copy.deepcopy(
                frozen_base["preserved_v1_scientific_dimensions"]
            ),
            "conditional_stage_c_semantics": frozen_base[
                "conditional_stage_c_semantics"
            ],
            "future_v2_spec_only_command": copy.deepcopy(
                FUTURE_V2_SPEC_ONLY_COMMAND
            ),
            "future_v2_script_present": False,
            "execution_authorized": False,
            "scientific_attempt_authorized": False,
            "automatic_execution": False,
        }
    )


def validate_correction_conditional_v2_spec_authority(
    value: Mapping[str, Any], *, decision: Mapping[str, Any]
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_conditional_v2_spec_authority(decision)
    if dict(value) != rebuilt or not _future_v2_script_absent():
        raise ForensicCorrectionContractError(
            "correction V2 specification authority drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_conditional_v2_spec_markdown(
    specification: Mapping[str, Any],
) -> str:
    validated = validate_correction_conditional_v2_spec_authority(
        specification,
        decision=specification.get("gate_decision"),
    )
    lines = [
        "# Conditional plan-aware JEPA V2 technical specification",
        "",
        (
            "Status: specification authorized by the forensic gate; "
            "execution and a scientific attempt are not authorized."
        ),
        "",
        (
            "This correction overlay preserves the frozen BASE V2 "
            "specification and both exact technical custody corrections."
        ),
        "",
        f"Primary forensic classification: `{BASE.FORENSIC_PRIMARY_CLASSIFICATION}`.",
        f"Secondary mechanism: `{BASE.FORENSIC_SECONDARY_MECHANISM}`.",
        f"Prior defect: `{BASE.FORENSIC_DEFECT_ID}`.",
        (
            "Correction defect: "
            "`ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH`."
        ),
        "",
    ]
    for index, clause in enumerate(validated["clauses"], start=1):
        lines.extend(
            [
                f"## {index}. {clause['id']}",
                "",
                str(clause["requirement"]),
                "",
            ]
        )
    row_counted = validated["exact_technical_corrections"][1]
    lines.extend(
        [
            "## Exact row-counted custody selector",
            "",
            (
                "The strict selector is "
                f"`{row_counted['validator_symbol']}` with exact keys "
                "`path`, `sha256`, `bytes`, `content_digest`, and `rows`."
            ),
            (
                "The source-closure callsites require path "
                f"`{row_counted['expected_path']}` and exactly "
                f"`{row_counted['expected_rows']}` rows."
            ),
            "",
            "V2 execution remains false and the future V2 script is absent.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_correction_conditional_v2_spec_markdown(
    value: str, *, specification: Mapping[str, Any]
) -> str:
    if not isinstance(value, str):
        raise ForensicCorrectionContractError(
            "correction V2 Markdown type drift"
        )
    expected = build_correction_conditional_v2_spec_markdown(specification)
    if value != expected:
        raise ForensicCorrectionContractError(
            "correction V2 Markdown projection drift"
        )
    return value


def build_correction_runtime_result(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    custody = _validate_correction_diagnostic_custody_envelope(
        diagnostic_custody_receipt
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        synthetic_results_receipt
    )
    decision = build_correction_conditional_v2_decision(
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_RUNTIME_RESULT_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC_CORRECTION_1",
            "source_freeze_commit": custody["source_commit"],
            "forensic_classification": "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED",
            "forensic_secondary_mechanism": (
                "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH"
            ),
            "correction_bug": (
                "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
            ),
            "correction_disposition": (
                "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_STAGED_"
                "EXTERNAL_CHECK_PENDING"
            ),
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "scientific_result": False,
            "scientific_classification": None,
            "diagnostic_custody": _runtime_json_binding(
                "diagnostic_custody", custody
            ),
            "process_state_scope_at_prepublication": copy.deepcopy(
                custody["process_state_scope_at_custody_write"]
            ),
            "launcher_live_during_prepublication": True,
            "literal_zero_all_forensic_roles_claimed": False,
            "conditional_v2_decision": decision,
            "v2_specification_conditionally_authorized": True,
            "v2_specification_publication": (
                "PENDING_PROSPECTIVE_FINALIZER_OUTPUT"
            ),
            "external_checker_gate_pass_claimed": False,
            "future_v2_spec_only_command": copy.deepcopy(
                FUTURE_V2_SPEC_ONLY_COMMAND
            ),
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_attempt_authorized": False,
            "scientific_attempt_consumed": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "technical_diagnostic_consumed_by_launcher_process_identity": (
                copy.deepcopy(custody["launcher_process_identity"])
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "TECHNICAL_CORRECTION_STAGED_EXTERNAL_CHECK_PENDING_"
                "NOT_A_SCIENTIFIC_RESULT"
            ),
            "pass": True,
        }
    )


def validate_correction_runtime_result(
    value: Mapping[str, Any],
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_runtime_result(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError("correction runtime result drift")
    return copy.deepcopy(dict(value))


def build_correction_final_namespace_inventory(
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    witness_relative = BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
        "final_namespace_inventory"
    ]
    witness = root / witness_relative
    if (
        root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
        or root.is_symlink()
        or not root.is_dir()
        or witness.exists()
        or witness.is_symlink()
    ):
        raise ForensicCorrectionContractError(
            "correction final namespace pre-witness drift"
        )
    observed = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=False
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_FINAL_NAMESPACE_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "CORRECTION_1_TERMINAL_DIAGNOSTIC_BUNDLE_COMPLETE",
            "diagnostic_root": str(root),
            **observed,
            "self_exclusion": {
                "path": witness_relative,
                "reason": "AVOID_SELF_REFERENTIAL_BYTE_AND_DIGEST_FIXED_POINT",
                "required_kind": "FILE",
                "required_mode": 0o644,
                "required_uid": 1000,
                "required_gid": 1000,
                "required_nlink": 1,
                "bytes_and_inode_excluded": True,
            },
            "symlinks": 0,
            "nonregular": 0,
            "multi_link_files": 0,
            "duplicate_file_inodes": 0,
            "unexpected_paths": [],
            "pass": True,
        }
    )


def validate_correction_final_namespace_inventory(
    value: Mapping[str, Any],
    *,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    root = Path(diagnostic_root).absolute()
    witness = root / BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
        "final_namespace_inventory"
    ]
    if (
        root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
        or witness.is_symlink()
        or not witness.is_file()
    ):
        raise ForensicCorrectionContractError(
            "correction final namespace witness absent"
        )
    observed = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    expected = BASE.attach_self_digest(
        {
            "schema": CORRECTION_FINAL_NAMESPACE_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "CORRECTION_1_TERMINAL_DIAGNOSTIC_BUNDLE_COMPLETE",
            "diagnostic_root": str(root),
            **observed,
            "self_exclusion": {
                "path": BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                    "final_namespace_inventory"
                ],
                "reason": "AVOID_SELF_REFERENTIAL_BYTE_AND_DIGEST_FIXED_POINT",
                "required_kind": "FILE",
                "required_mode": 0o644,
                "required_uid": 1000,
                "required_gid": 1000,
                "required_nlink": 1,
                "bytes_and_inode_excluded": True,
            },
            "symlinks": 0,
            "nonregular": 0,
            "multi_link_files": 0,
            "duplicate_file_inodes": 0,
            "unexpected_paths": [],
            "pass": True,
        }
    )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "correction final namespace inventory drift"
        )
    return copy.deepcopy(dict(value))


def write_correction_final_namespace_inventory(
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    value = build_correction_final_namespace_inventory(root)
    directory_rows = {
        str(row["path"]): row
        for row in value["rows"]
        if row["kind"] == "DIRECTORY"
    }
    BASE._exclusive_write_relative_no_follow(
        root,
        BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
            "final_namespace_inventory"
        ],
        _authority_bytes(value),
        expected_directory_rows=directory_rows,
        mode=0o644,
    )
    return validate_correction_final_namespace_inventory(
        value, diagnostic_root=root
    )


def write_correction_terminal_bundle(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    if root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT:
        raise ForensicCorrectionContractError("correction terminal root drift")
    result = build_correction_runtime_result(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    root_fd = BASE._open_absolute_directory_no_follow(root)
    try:
        info = os.fstat(root_fd)
    finally:
        os.close(root_fd)
    root_row = {
        "path": ".", "kind": "DIRECTORY",
        "mode": stat.S_IMODE(info.st_mode), "uid": int(info.st_uid),
        "gid": int(info.st_gid), "nlink": int(info.st_nlink),
        "device": int(info.st_dev), "inode": int(info.st_ino),
    }
    BASE._exclusive_write_relative_no_follow(
        root,
        BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["result"],
        _authority_bytes(result),
        expected_directory_rows={".": root_row},
        mode=0o644,
    )
    inventory = write_correction_final_namespace_inventory(root)
    return {
        "runtime_result": result,
        "final_namespace_inventory": inventory,
        "v2_specification_conditionally_authorized": True,
        "tracked_publication_completed": False,
        "external_checker_gate_pass_claimed": False,
        "pass": True,
    }


def load_and_validate_correction_diagnostic_bundle(
    *,
    repo_root: str | Path,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
    repository_mode: str = "PREPUBLICATION_FREEZE_HEAD",
) -> dict[str, Any]:
    """Load the BASE-shaped correction bundle through BASE no-follow custody."""

    repo = Path(repo_root).resolve()
    root = Path(diagnostic_root).absolute()
    if (
        repository_mode not in {
            "PREPUBLICATION_FREEZE_HEAD",
            "PRECOMMIT_STAGED_RESULT_PATHS",
            "POSTCOMMIT_RESULT_HEAD",
        }
        or
        root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ForensicCorrectionContractError("correction diagnostic root drift")
    preflight = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    preflight_rows = {str(row["path"]): row for row in preflight["rows"]}
    loaded = {
        key: BASE._read_no_follow_bound_file(
            root,
            BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
            expected_stat_rows=preflight_rows,
        )
        for key in BASE.PREEXECUTION_DIAGNOSTIC_FINAL_MAIN_FILE_KEYS
    }

    def load_json_key(key: str) -> dict[str, Any]:
        return BASE._canonical_json_from_bound_bytes(loaded[key], label=key)

    environment = BASE.validate_environment_receipt(load_json_key("environment"))
    command = BASE.validate_command_receipt(load_json_key("command"))
    invocation = BASE.validate_invocation_receipt(load_json_key("invocation"))
    os_evidence = BASE.validate_os_evidence_receipt(load_json_key("os_evidence"))
    preexecution = BASE.validate_preexecution_only_receipt(
        load_json_key("preexecution_only")
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        load_json_key("synthetic_results")
    )
    marker = BASE.validate_last_stage_marker(load_json_key("last_stage_marker"))
    guard = BASE.validate_read_guard_manifest(
        load_json_key("read_guard_manifest"), repo_root=repo
    )
    rows = BASE._startup_rows_from_bound_bytes(loaded["startup_stage_ledger"])
    streams = {
        label: {
            "path": BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
            "sha256": hashlib.sha256(loaded[key]).hexdigest(),
            "bytes": len(loaded[key]),
        }
        for label, key in {
            "stdout": "child_stdout", "stderr": "child_stderr",
            "traceback": "child_traceback", "exception": "child_exception",
            "heartbeat": "heartbeat", "read_guard_events": "read_guard_events",
        }.items()
    }
    freeze = validate_correction_freeze_custody_receipt(
        load_json_key("forensic_freeze_custody"),
        repo_root=repo,
        reverify_failed_root=True,
    )
    freeze_git_custody = _validate_correction_freeze_commit_git_custody(
        repo, freeze["repo_head"]
    )
    current_head = BASE._git_text(repo, ("rev-parse", "HEAD"))
    dirty_paths = _correction_dirty_paths(repo)
    exact_result_commit = False
    if repository_mode == "POSTCOMMIT_RESULT_HEAD":
        parents = BASE._git_text(
            repo, ("rev-list", "--parents", "-n", "1", current_head)
        ).split()
        changed = sorted(
            row for row in BASE._git_text(
                repo, ("diff", "--name-only", freeze["repo_head"], current_head)
            ).splitlines() if row
        )
        exact_result_commit = (
            len(parents) == 2
            and parents[1] == freeze["repo_head"]
            and BASE._git_text(
                repo, ("show", "-s", "--format=%s", current_head)
            ) == CORRECTION_RESULT_COMMIT_SUBJECT
            and changed == sorted(CORRECTION_RESULT_PATHS)
        )
    if (
        (
            repository_mode == "PREPUBLICATION_FREEZE_HEAD"
            and current_head != freeze["repo_head"]
        )
        or (
            repository_mode == "PRECOMMIT_STAGED_RESULT_PATHS"
            and (
                current_head != freeze["repo_head"]
                or dirty_paths != sorted(CORRECTION_RESULT_PATHS)
            )
        )
        or (
            repository_mode == "POSTCOMMIT_RESULT_HEAD"
            and not exact_result_commit
        )
        or (
            repository_mode != "PRECOMMIT_STAGED_RESULT_PATHS"
            and dirty_paths
        )
        or not _future_v2_script_absent()
        or not _anchored_leaf_absent(CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT)
    ):
        raise ForensicCorrectionContractError(
            "correction terminal repository custody drift"
        )
    closure_raw = BASE._read_no_follow_bound_file(
        repo, TRACKED_CORRECTION_SOURCE_CLOSURE_PATH, expected_stat_rows=None
    )
    closure = BASE._canonical_json_from_bound_bytes(
        closure_raw, label="correction_source_closure"
    )
    validate_forensic_correction_source_closure(closure, repo_root=repo)
    child = validate_corrected_preexecution_child_result(
        load_json_key("child_stdout"),
        repo_root=repo,
        forensic_freeze_custody=freeze,
    )
    umask = BASE.validate_diagnostic_umask_custody(load_json_key("umask_custody"))
    rebuilt = build_correction_diagnostic_custody_receipt(
        repo_root=repo,
        source_commit=invocation["source_commit"],
        correction_source_closure=closure,
        diagnostic_root=root,
        launcher_process_identity=invocation["launcher_process_identity"],
        child_process_identity=invocation["child_process_identity"],
        forensic_freeze_custody_receipt=freeze,
        umask_custody_receipt=umask,
        invocation_receipt=invocation,
        environment_receipt=environment,
        command_receipt=command,
        read_guard_manifest=guard,
        os_evidence_receipt=os_evidence,
        preexecution_only_receipt=preexecution,
        synthetic_results_receipt=synthetic,
        corrected_child_result=child,
        last_stage_marker_value=marker,
        startup_stage_rows=rows,
        stream_bindings=streams,
        exception_observed=streams["exception"]["bytes"] > 0,
    )
    persisted = load_json_key("diagnostic_custody")
    validate_correction_diagnostic_custody_receipt(
        persisted,
        repo_root=repo,
        source_commit=invocation["source_commit"],
        correction_source_closure=closure,
        diagnostic_root=root,
        launcher_process_identity=invocation["launcher_process_identity"],
        child_process_identity=invocation["child_process_identity"],
        forensic_freeze_custody_receipt=freeze,
        umask_custody_receipt=umask,
        invocation_receipt=invocation,
        environment_receipt=environment,
        command_receipt=command,
        read_guard_manifest=guard,
        os_evidence_receipt=os_evidence,
        preexecution_only_receipt=preexecution,
        synthetic_results_receipt=synthetic,
        corrected_child_result=child,
        last_stage_marker_value=marker,
        startup_stage_rows=rows,
        stream_bindings=streams,
        exception_observed=streams["exception"]["bytes"] > 0,
    )
    if persisted != rebuilt:
        raise ForensicCorrectionContractError(
            "correction custody persisted/rebuilt drift"
        )
    for key, receipt in {
        "environment": environment, "command": command,
        "invocation": invocation, "os_evidence": os_evidence,
        "preexecution_only": preexecution, "synthetic_results": synthetic,
        "last_stage_marker": marker, "read_guard_manifest": guard,
        "forensic_freeze_custody": freeze, "umask_custody": umask,
    }.items():
        observed = {
            "path": BASE.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
            "sha256": hashlib.sha256(loaded[key]).hexdigest(),
            "bytes": len(loaded[key]),
            "content_digest": receipt["content_digest"],
        }
        if persisted["receipts"][key] != observed:
            raise ForensicCorrectionContractError(
                f"correction runtime binding drift: {key}"
            )
    if persisted["receipts"]["startup_stage_ledger"] != BASE._startup_ledger_binding(
        rows
    ):
        raise ForensicCorrectionContractError(
            "correction startup ledger binding drift"
        )
    runtime_result = validate_correction_runtime_result(
        load_json_key("result"),
        diagnostic_custody_receipt=persisted,
        synthetic_results_receipt=synthetic,
    )
    inventory = validate_correction_final_namespace_inventory(
        load_json_key("final_namespace_inventory"), diagnostic_root=root
    )
    postflight = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    if postflight != preflight:
        raise ForensicCorrectionContractError(
            "correction diagnostic bundle changed across reads"
        )
    return {
        "environment": environment,
        "command": command,
        "invocation": invocation,
        "os_evidence": os_evidence,
        "preexecution_only": preexecution,
        "synthetic_results": synthetic,
        "preexecution_child_result": child,
        "last_stage_marker": marker,
        "read_guard_manifest": guard,
        "startup_stage_rows": rows,
        "streams": streams,
        "diagnostic_custody": persisted,
        "forensic_freeze_custody": freeze,
        "forensic_freeze_commit_git_custody": freeze_git_custody,
        "umask_custody": umask,
        "result": runtime_result,
        "final_namespace_inventory": inventory,
        "preflight_namespace_inventory": preflight,
        "postflight_namespace_inventory": postflight,
        "failed_root_inode_nonreuse": build_correction_inode_nonreuse_proof(),
        "repository_mode": repository_mode,
    }


def _tracked_correction_json_binding(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    return BASE._artifact_binding(
        str(path), _authority_bytes(value), content_digest=value["content_digest"]
    )


def build_correction_fresh_technical_root_manifest(
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
) -> dict[str, Any]:
    """Hash only the exact BASE terminal technical namespace after preflight."""

    root = Path(diagnostic_root).absolute()
    if root != PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT:
        raise ForensicCorrectionContractError(
            "correction technical manifest root drift"
        )
    preflight = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    _expected_directories, expected_files = (
        BASE._expected_final_diagnostic_namespace()
    )
    rows: list[dict[str, Any]] = []
    for relative in sorted(expected_files):
        payload = BASE._read_no_follow_bound_file(
            root, relative, expected_stat_rows=None
        )
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    postflight = BASE._observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    if preflight != postflight:
        raise ForensicCorrectionContractError(
            "correction technical manifest namespace changed during hashing"
        )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_FRESH_TECHNICAL_MANIFEST_SCHEMA,
            "diagnostic_root": str(root),
            "metadata_path_set_validated_before_content_reads": True,
            "stat_inventory_sha256": preflight["stat_inventory_sha256"],
            "rows": rows,
            "file_count": len(rows),
            "total_file_bytes": sum(row["bytes"] for row in rows),
            "manifest_sha256": BASE.canonical_json_sha256(rows),
            "manifest_row_keys": ["path", "sha256", "bytes"],
            "technical_only": True,
            "scientific_inputs_or_outputs_opened": False,
            "preflight_equals_postflight": True,
            "pass": True,
        }
    )


def validate_correction_fresh_technical_root_manifest(
    value: Mapping[str, Any], *, reverify_live: bool
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    if reverify_live:
        expected = build_correction_fresh_technical_root_manifest()
    else:
        rows = value.get("rows")
        _expected_directories, expected_files = (
            BASE._expected_final_diagnostic_namespace()
        )
        if (
            type(rows) is not list
            or any(
                type(row) is not dict
                or set(row) != {"path", "sha256", "bytes"}
                or not isinstance(row["path"], str)
                or not row["path"]
                or not isinstance(row["sha256"], str)
                or not _LOWERCASE_SHA256_RE.fullmatch(row["sha256"])
                or isinstance(row["bytes"], bool)
                or not isinstance(row["bytes"], int)
                or row["bytes"] < 0
                for row in rows
            )
            or rows != sorted(rows, key=lambda row: row["path"])
            or len({row["path"] for row in rows}) != len(rows)
            or [row["path"] for row in rows] != sorted(expected_files)
            or len(rows) != len(expected_files)
            or not isinstance(value.get("stat_inventory_sha256"), str)
            or not _LOWERCASE_SHA256_RE.fullmatch(
                value["stat_inventory_sha256"]
            )
        ):
            raise ForensicCorrectionContractError(
                "correction technical manifest row drift"
            )
        expected = BASE.attach_self_digest(
            {
                "schema": CORRECTION_FRESH_TECHNICAL_MANIFEST_SCHEMA,
                "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
                "metadata_path_set_validated_before_content_reads": True,
                "stat_inventory_sha256": value.get("stat_inventory_sha256"),
                "rows": rows,
                "file_count": len(rows),
                "total_file_bytes": sum(row["bytes"] for row in rows),
                "manifest_sha256": BASE.canonical_json_sha256(rows),
                "manifest_row_keys": ["path", "sha256", "bytes"],
                "technical_only": True,
                "scientific_inputs_or_outputs_opened": False,
                "preflight_equals_postflight": True,
                "pass": True,
            }
        )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "correction fresh technical manifest drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_result(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    failed_root_custody_receipt: Mapping[str, Any],
    fixture_receipt: Mapping[str, Any],
    final_namespace_inventory: Mapping[str, Any],
    conditional_v2_decision: Mapping[str, Any],
    conditional_v2_spec_authority: Mapping[str, Any],
    conditional_v2_spec_markdown: str,
    inode_nonreuse_proof: Mapping[str, Any],
) -> dict[str, Any]:
    custody = _validate_correction_diagnostic_custody_envelope(
        diagnostic_custody_receipt
    )
    failed = validate_failed_diagnostic_root_custody(
        failed_root_custody_receipt, reverify_live=True
    )
    fixtures = validate_row_counted_binding_fixture_receipt(fixture_receipt)
    inventory = validate_correction_final_namespace_inventory(
        final_namespace_inventory,
        diagnostic_root=PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        synthetic_results_receipt
    )
    decision = validate_correction_conditional_v2_decision(
        conditional_v2_decision,
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    specification = validate_correction_conditional_v2_spec_authority(
        conditional_v2_spec_authority, decision=decision
    )
    markdown = validate_correction_conditional_v2_spec_markdown(
        conditional_v2_spec_markdown, specification=specification
    )
    BASE.validate_self_digest(inode_nonreuse_proof)
    if (
        inode_nonreuse_proof.get("shared_inodes") != []
        or inode_nonreuse_proof.get("shared_inode_count") != 0
        or inode_nonreuse_proof.get("files_reused") != 0
        or inode_nonreuse_proof.get("pass") is not True
        or custody["preexecution_child_result"]["row_counted_binding_fixtures"]
        != fixtures
        or custody["receipts"]["synthetic_results"]
        != _runtime_json_binding("synthetic_results", synthetic)
        or decision["v2_spec_authorized"] is not True
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError(
            "correction prospective result input drift"
        )
    tracked = {
        "diagnostic_custody": _tracked_correction_json_binding(
            TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH, custody
        ),
        "failed_diagnostic_root_custody": _tracked_correction_json_binding(
            TRACKED_CORRECTION_FAILED_ROOT_CUSTODY_PATH, failed
        ),
        "row_counted_binding_fixtures": _tracked_correction_json_binding(
            TRACKED_CORRECTION_FIXTURE_RESULTS_PATH, fixtures
        ),
        "final_namespace_inventory": _tracked_correction_json_binding(
            TRACKED_CORRECTION_FINAL_NAMESPACE_PATH, inventory
        ),
        "conditional_v2_specification_authority": (
            _tracked_correction_json_binding(
                BASE.TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH,
                specification,
            )
        ),
        "conditional_v2_specification_markdown": BASE._artifact_binding(
            str(BASE.TRACKED_CONDITIONAL_V2_SPEC_PATH),
            markdown.encode("utf-8"),
        ),
    }
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_RESULT_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "lineage": {
                **copy.deepcopy(PRESERVED_LINEAGE),
                "forensic_correction_1_freeze": custody["source_commit"],
                "correction_result_commit": None,
                "correction_result_commit_subject": (
                    CORRECTION_RESULT_COMMIT_SUBJECT
                ),
                "result_commit_self_binding": (
                    "EXTERNAL_POSTCOMMIT_VALIDATION_NOT_SELF_EMBEDDED"
                ),
            },
            "source_freeze_commit": custody["source_commit"],
            "forensic_classification": "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED",
            "forensic_secondary_mechanism": (
                "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH"
            ),
            "correction_bug": (
                "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
            ),
            "validator_id": ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
            "correction_disposition": (
                "PROSPECTIVE_RESULT_EXTERNAL_CHECK_AND_PUBLICATION_PENDING"
            ),
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "external_checker_gate_pass_claimed": False,
            "postcommit_gate_pass_claimed": False,
            "scientific_result": False,
            "scientific_classification": None,
            "scientific_metrics": None,
            "scientific_claims": [],
            "diagnostic": {
                "root": custody["diagnostic_root"],
                "launcher_process_identity": copy.deepcopy(
                    custody["launcher_process_identity"]
                ),
                "child_process_identity": copy.deepcopy(
                    custody["child_process_identity"]
                ),
                "returncode": custody["returncode"],
                "termination": copy.deepcopy(custody["termination"]),
                "cleanup": copy.deepcopy(custody["cleanup"]),
                "base_runtime_paths_and_stream_mechanics_reused": True,
                "final_namespace_inventory_scope": (
                    "LAUNCHER_STAGED_BASE_BUNDLE_ONLY_"
                    "PHASE_CUSTODY_TRACKED_SEPARATELY"
                ),
                "final_namespace_inventory": tracked[
                    "final_namespace_inventory"
                ],
            },
            "row_counted_binding": {
                "source_closure": copy.deepcopy(
                    custody["preexecution_child_result"][
                        "forensic_authority_source_closure"
                    ]
                ),
                "terminal_receipt": copy.deepcopy(
                    custody["preexecution_child_result"][
                        "terminal_binding_receipt"
                    ]
                ),
                "fixture_receipt": tracked[
                    "row_counted_binding_fixtures"
                ],
            },
            "failed_diagnostic_root": {
                "custody": tracked["failed_diagnostic_root_custody"],
                "inode_nonreuse_proof": copy.deepcopy(
                    dict(inode_nonreuse_proof)
                ),
                "files_reused": 0,
            },
            "tracked_receipts": tracked,
            "conditional_v2_decision": decision,
            "conditional_v2_specification": {
                "authority": tracked[
                    "conditional_v2_specification_authority"
                ],
                "markdown": tracked[
                    "conditional_v2_specification_markdown"
                ],
                "specification_conditionally_authorized": True,
                "publication_pending": True,
            },
            "future_v2_spec_only_command": copy.deepcopy(
                FUTURE_V2_SPEC_ONLY_COMMAND
            ),
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_attempt_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "technical_diagnostic_consumed_by_launcher_process_identity": (
                copy.deepcopy(custody["launcher_process_identity"])
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "PROSPECTIVE_TECHNICAL_RESULT_PENDING_EXTERNAL_CHECK_"
                "PUBLICATION_AND_POSTCOMMIT_REPLAY"
            ),
            "pass": True,
        }
    )


def build_correction_core_result_artifacts_from_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "synthetic_results", "diagnostic_custody",
        "forensic_freeze_custody", "final_namespace_inventory",
        "failed_root_inode_nonreuse",
    }
    if not required.issubset(bundle):
        raise ForensicCorrectionContractError(
            "correction terminal bundle is incomplete"
        )
    custody = _validate_correction_diagnostic_custody_envelope(
        bundle["diagnostic_custody"]
    )
    synthetic = BASE.validate_synthetic_results_receipt(
        bundle["synthetic_results"]
    )
    failed = validate_failed_diagnostic_root_custody(
        bundle["forensic_freeze_custody"]["failed_diagnostic_root_custody"],
        reverify_live=True,
    )
    fixtures = validate_row_counted_binding_fixture_receipt(
        custody["preexecution_child_result"]["row_counted_binding_fixtures"]
    )
    decision = build_correction_conditional_v2_decision(
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    specification = build_correction_conditional_v2_spec_authority(decision)
    validate_correction_conditional_v2_spec_authority(
        specification, decision=decision
    )
    markdown = build_correction_conditional_v2_spec_markdown(specification)
    validate_correction_conditional_v2_spec_markdown(
        markdown, specification=specification
    )
    result = build_correction_result(
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
        failed_root_custody_receipt=failed,
        fixture_receipt=fixtures,
        final_namespace_inventory=bundle["final_namespace_inventory"],
        conditional_v2_decision=decision,
        conditional_v2_spec_authority=specification,
        conditional_v2_spec_markdown=markdown,
        inode_nonreuse_proof=bundle["failed_root_inode_nonreuse"],
    )
    values: dict[Path, Mapping[str, Any] | str] = {
        TRACKED_CORRECTION_RESULT_PATH: result,
        TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH: custody,
        TRACKED_CORRECTION_FAILED_ROOT_CUSTODY_PATH: failed,
        TRACKED_CORRECTION_FIXTURE_RESULTS_PATH: fixtures,
        TRACKED_CORRECTION_FINAL_NAMESPACE_PATH: bundle[
            "final_namespace_inventory"
        ],
        BASE.TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH: specification,
        BASE.TRACKED_CONDITIONAL_V2_SPEC_PATH: markdown,
    }
    payloads = {
        path: (
            _authority_bytes(value)
            if isinstance(value, Mapping)
            else value.encode("utf-8")
        )
        for path, value in values.items()
    }
    if {str(path) for path in payloads} != set(CORRECTION_CORE_RESULT_PATHS):
        raise ForensicCorrectionContractError(
            "correction core result path-set drift"
        )
    return {
        "result": result,
        "diagnostic_custody": custody,
        "failed_root_custody": failed,
        "fixture_receipt": fixtures,
        "final_namespace_inventory": bundle["final_namespace_inventory"],
        "conditional_v2_decision": decision,
        "conditional_v2_spec_authority": specification,
        "conditional_v2_spec_markdown": markdown,
        "tracked_payloads": payloads,
    }


def _correction_artifact_binding_map(
    payloads: Mapping[Path, bytes],
) -> dict[str, Any]:
    return {
        str(path): BASE._artifact_binding(str(path), payloads[path])
        for path in sorted(payloads, key=str)
    }


def _validate_correction_phase_identity(
    value: Mapping[str, Any], *, phase: str, require_current: bool
) -> dict[str, Any]:
    identity = BASE.validate_process_identity(value)
    if phase == "FINALIZER":
        expected_argv = expected_correction_finalizer_outer_argv()
        expected_role = CORRECTION_FINALIZER_ROLE
    elif phase == "CHECKER":
        expected_argv = expected_correction_checker_outer_argv()
        expected_role = CORRECTION_CHECKER_ROLE
    else:
        raise ForensicCorrectionContractError("unknown correction phase")
    if (
        identity["argv"] != expected_argv
        or identity["role"] != expected_role
        or Path(identity["executable"]).resolve()
        != BASE.FORENSIC_INTERPRETER_RESOLVED
        or (require_current and identity["pid"] != os.getpid())
    ):
        raise ForensicCorrectionContractError(
            f"correction {phase.lower()} process identity drift"
        )
    return identity


def _require_correction_terminal_role_zero_before_read_or_spawn(
    repo_root: Path,
    *,
    current_phase_pid: int | None,
    coordinator_pid: int,
    gate: str,
) -> None:
    excluded_base = (
        {current_phase_pid} if current_phase_pid is not None else set()
    )
    excluded_correction = {coordinator_pid}
    if current_phase_pid is not None:
        excluded_correction.add(current_phase_pid)
    base_matches = BASE._active_forensic_or_scientific_processes(
        repo_root, exclude_pids=excluded_base
    )
    correction_matches = _active_correction_terminal_coordinators(
        exclude_pids=excluded_correction
    )
    if base_matches or correction_matches:
        raise ForensicCorrectionContractError(
            f"correction terminal role-zero gate failed before {gate}"
        )


def build_correction_finalizer_prospective_receipt(
    *,
    repo_root: str | Path,
    finalizer_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    identity = _validate_correction_phase_identity(
        finalizer_process_identity, phase="FINALIZER", require_current=True
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity, require_live=True
    )
    if coordinator["pid"] == identity["pid"]:
        raise ForensicCorrectionContractError(
            "finalizer/coordinator identity collision"
        )
    _require_correction_terminal_role_zero_before_read_or_spawn(
        root,
        current_phase_pid=identity["pid"],
        coordinator_pid=coordinator["pid"],
        gate="finalizer bundle read",
    )
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode="PREPUBLICATION_FREEZE_HEAD",
    )
    artifacts = build_correction_core_result_artifacts_from_bundle(bundle)
    bindings = _correction_artifact_binding_map(artifacts["tracked_payloads"])
    _require_correction_terminal_role_zero_before_read_or_spawn(
        root,
        current_phase_pid=identity["pid"],
        coordinator_pid=coordinator["pid"],
        gate="finalizer receipt completion",
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_FINALIZER_PROSPECTIVE_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "FINALIZER_PROSPECTIVE_BYTES_ONLY",
            "finalizer_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "root_coordinator_live_and_explicitly_excluded": True,
            "outer_exact_argv": expected_correction_finalizer_outer_argv(),
            "inner_evaluator_argv": expected_correction_finalizer_inner_argv(),
            "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
            "diagnostic_final_namespace": _runtime_json_binding(
                "final_namespace_inventory",
                bundle["final_namespace_inventory"],
            ),
            "prospective_artifact_bindings": bindings,
            "prospective_artifact_count": len(CORRECTION_CORE_RESULT_PATHS),
            "prospective_artifact_set_sha256": BASE.canonical_json_sha256(
                bindings
            ),
            "full_frozen_v2_json_and_markdown_included": True,
            "tracked_paths_written": [],
            "external_checker_gate_pass_claimed": False,
            "postcommit_gate_pass_claimed": False,
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "finalizer_live_at_receipt_build": True,
            "literal_zero_all_correction_roles_claimed": False,
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_correction_finalizer_prospective_receipt(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    require_current_identity: bool,
    repository_mode: str,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    identity = _validate_correction_phase_identity(
        value.get("finalizer_process_identity"),
        phase="FINALIZER",
        require_current=require_current_identity,
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=require_current_identity,
    )
    root = Path(repo_root).resolve()
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode=repository_mode,
    )
    artifacts = build_correction_core_result_artifacts_from_bundle(bundle)
    bindings = _correction_artifact_binding_map(artifacts["tracked_payloads"])
    expected = BASE.attach_self_digest(
        {
            "schema": CORRECTION_FINALIZER_PROSPECTIVE_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "FINALIZER_PROSPECTIVE_BYTES_ONLY",
            "finalizer_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "root_coordinator_live_and_explicitly_excluded": True,
            "outer_exact_argv": expected_correction_finalizer_outer_argv(),
            "inner_evaluator_argv": expected_correction_finalizer_inner_argv(),
            "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
            "diagnostic_final_namespace": _runtime_json_binding(
                "final_namespace_inventory",
                bundle["final_namespace_inventory"],
            ),
            "prospective_artifact_bindings": bindings,
            "prospective_artifact_count": len(CORRECTION_CORE_RESULT_PATHS),
            "prospective_artifact_set_sha256": BASE.canonical_json_sha256(
                bindings
            ),
            "full_frozen_v2_json_and_markdown_included": True,
            "tracked_paths_written": [],
            "external_checker_gate_pass_claimed": False,
            "postcommit_gate_pass_claimed": False,
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "finalizer_live_at_receipt_build": True,
            "literal_zero_all_correction_roles_claimed": False,
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "correction finalizer prospective receipt drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_external_checker_receipt(
    *,
    repo_root: str | Path,
    checker_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    identity = _validate_correction_phase_identity(
        checker_process_identity, phase="CHECKER", require_current=True
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity, require_live=True
    )
    if coordinator["pid"] == identity["pid"]:
        raise ForensicCorrectionContractError(
            "checker/coordinator identity collision"
        )
    _require_correction_terminal_role_zero_before_read_or_spawn(
        root,
        current_phase_pid=identity["pid"],
        coordinator_pid=coordinator["pid"],
        gate="external checker bundle read",
    )
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode="PREPUBLICATION_FREEZE_HEAD",
    )
    corrected_child = bundle["preexecution_child_result"]
    externally_checked_terminal = check_external_terminal_binding_receipt(
        corrected_child["terminal_binding_receipt"]
    )
    externally_checked_closure = check_external_row_counted_content_binding(
        corrected_child["forensic_authority_source_closure"]
    )
    if (
        externally_checked_terminal["source_closure"]
        != externally_checked_closure
    ):
        raise ForensicCorrectionContractError(
            "external checker terminal/source-closure disagreement"
        )
    artifacts = build_correction_core_result_artifacts_from_bundle(bundle)
    bindings = _correction_artifact_binding_map(artifacts["tracked_payloads"])
    _require_correction_terminal_role_zero_before_read_or_spawn(
        root,
        current_phase_pid=identity["pid"],
        coordinator_pid=coordinator["pid"],
        gate="external checker receipt completion",
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_EXTERNAL_CHECKER_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "EXTERNAL_CHECKER_PROSPECTIVE_VALIDATION",
            "checker_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "root_coordinator_live_and_explicitly_excluded": True,
            "outer_exact_argv": expected_correction_checker_outer_argv(),
            "inner_evaluator_argv": expected_correction_checker_inner_argv(),
            "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
            "externally_validated_terminal_binding_receipt": (
                externally_checked_terminal
            ),
            "externally_validated_source_closure_binding": (
                externally_checked_closure
            ),
            "terminal_source_closure_exact_agreement": True,
            "external_five_key_selector_called": (
                "check_external_terminal_binding_receipt"
            ),
            "independently_rebuilt_artifact_bindings": bindings,
            "prospective_artifact_count": len(CORRECTION_CORE_RESULT_PATHS),
            "prospective_artifact_set_sha256": BASE.canonical_json_sha256(
                bindings
            ),
            "finalizer_output_comparison_required_before_publication": True,
            "all_other_producer_roles_zero": True,
            "checker_live_at_receipt_build": True,
            "literal_zero_all_correction_roles_claimed": False,
            "tracked_paths_written": [],
            "postcommit_gate_pass_claimed": False,
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_correction_external_checker_receipt(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    require_current_identity: bool,
    repository_mode: str,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    identity = _validate_correction_phase_identity(
        value.get("checker_process_identity"),
        phase="CHECKER",
        require_current=require_current_identity,
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=require_current_identity,
    )
    root = Path(repo_root).resolve()
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode=repository_mode,
    )
    corrected_child = bundle["preexecution_child_result"]
    externally_checked_terminal = check_external_terminal_binding_receipt(
        corrected_child["terminal_binding_receipt"]
    )
    externally_checked_closure = check_external_row_counted_content_binding(
        corrected_child["forensic_authority_source_closure"]
    )
    if (
        externally_checked_terminal["source_closure"]
        != externally_checked_closure
    ):
        raise ForensicCorrectionContractError(
            "external checker terminal/source-closure disagreement"
        )
    artifacts = build_correction_core_result_artifacts_from_bundle(bundle)
    bindings = _correction_artifact_binding_map(artifacts["tracked_payloads"])
    expected = BASE.attach_self_digest(
        {
            "schema": CORRECTION_EXTERNAL_CHECKER_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": "EXTERNAL_CHECKER_PROSPECTIVE_VALIDATION",
            "checker_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "root_coordinator_live_and_explicitly_excluded": True,
            "outer_exact_argv": expected_correction_checker_outer_argv(),
            "inner_evaluator_argv": expected_correction_checker_inner_argv(),
            "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT),
            "externally_validated_terminal_binding_receipt": (
                externally_checked_terminal
            ),
            "externally_validated_source_closure_binding": (
                externally_checked_closure
            ),
            "terminal_source_closure_exact_agreement": True,
            "external_five_key_selector_called": (
                "check_external_terminal_binding_receipt"
            ),
            "independently_rebuilt_artifact_bindings": bindings,
            "prospective_artifact_count": len(CORRECTION_CORE_RESULT_PATHS),
            "prospective_artifact_set_sha256": BASE.canonical_json_sha256(
                bindings
            ),
            "finalizer_output_comparison_required_before_publication": True,
            "all_other_producer_roles_zero": True,
            "checker_live_at_receipt_build": True,
            "literal_zero_all_correction_roles_claimed": False,
            "tracked_paths_written": [],
            "postcommit_gate_pass_claimed": False,
            "technical_success_classification": None,
            "next_pass_authorization": None,
            "future_v2_script_present": False,
            "v2_execution_authorized": False,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "correction external checker receipt drift"
        )
    return copy.deepcopy(dict(value))


def _correction_phase_role(phase: str) -> str:
    if phase == "FINALIZER":
        return CORRECTION_FINALIZER_ROLE
    if phase == "CHECKER":
        return CORRECTION_CHECKER_ROLE
    raise ForensicCorrectionContractError("unknown correction phase")


def build_correction_phase_environment_custody(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    if (
        type(environment) is not dict
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
    ):
        raise ForensicCorrectionContractError(
            "correction phase environment map drift"
        )
    copied = dict(environment)
    names = sorted(copied)
    venv = str(BASE.FORENSIC_INTERPRETER.parent.parent)
    path_prefix = str(BASE.FORENSIC_INTERPRETER.parent)
    if (
        copied.get("PYTHONNOUSERSITE") != "1"
        or copied.get("PYTHONUNBUFFERED") != "1"
        or copied.get("PYTHONFAULTHANDLER") != "1"
        or copied.get("VIRTUAL_ENV") != venv
        or not copied.get("PATH", "").startswith(path_prefix)
    ):
        raise ForensicCorrectionContractError(
            "correction phase environment policy drift"
        )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_PHASE_ENVIRONMENT_SCHEMA,
            "key_names": names,
            "key_count": len(names),
            "python_key_names": sorted(
                key for key in names if key.startswith("PYTHON")
            ),
            "required_python_environment": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1",
            },
            "virtual_env": venv,
            "path_prepend": path_prefix,
            "cwd": str(CORRECTION_REPO_ROOT),
            "canonical_map_sha256": BASE.canonical_json_sha256(copied),
            "values_persisted": False,
            "pass": True,
        }
    )


def validate_correction_phase_environment_custody(
    value: Mapping[str, Any],
    *,
    expected_environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    required = {
        "schema", "key_names", "key_count", "python_key_names",
        "required_python_environment", "virtual_env", "path_prepend", "cwd",
        "canonical_map_sha256", "values_persisted", "pass",
        "content_digest",
    }
    names = value.get("key_names")
    if (
        set(value) != required
        or value.get("schema") != CORRECTION_PHASE_ENVIRONMENT_SCHEMA
        or not isinstance(names, list)
        or names != sorted(set(names))
        or any(not isinstance(name, str) for name in names)
        or value.get("key_count") != len(names)
        or value.get("python_key_names")
        != sorted(name for name in names if name.startswith("PYTHON"))
        or value.get("required_python_environment")
        != {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
        }
        or value.get("virtual_env")
        != str(BASE.FORENSIC_INTERPRETER.parent.parent)
        or value.get("path_prepend") != str(BASE.FORENSIC_INTERPRETER.parent)
        or value.get("cwd") != str(CORRECTION_REPO_ROOT)
        or not isinstance(value.get("canonical_map_sha256"), str)
        or not _LOWERCASE_SHA256_RE.fullmatch(value["canonical_map_sha256"])
        or value.get("values_persisted") is not False
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction phase environment custody drift"
        )
    if expected_environment is not None:
        rebuilt = build_correction_phase_environment_custody(
            expected_environment
        )
        if dict(value) != rebuilt:
            raise ForensicCorrectionContractError(
                "correction phase environment/map binding drift"
            )
    return copy.deepcopy(dict(value))


def build_correction_phase_heartbeat_row(
    *,
    phase: str,
    pid: int,
    sequence: int,
    event: str,
    monotonic_ns: int,
) -> dict[str, Any]:
    _correction_phase_role(phase)
    expected_event = {
        0: "WRAPPER_STARTED",
        1: {"WRAPPER_COMPLETED", "WRAPPER_FAILED"},
    }
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or sequence not in expected_event
        or (
            event != expected_event[sequence]
            if sequence == 0
            else event not in expected_event[sequence]
        )
        or isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns <= 0
    ):
        raise ForensicCorrectionContractError(
            "correction phase heartbeat input drift"
        )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_phase_heartbeat.v1"
            ),
            "phase": phase,
            "producer_role": _correction_phase_role(phase),
            "pid": pid,
            "sequence": sequence,
            "event": event,
            "monotonic_ns": monotonic_ns,
        }
    )


def validate_correction_phase_heartbeat_row(
    value: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_phase_heartbeat_row(
        phase=phase,
        pid=value.get("pid"),
        sequence=value.get("sequence"),
        event=value.get("event"),
        monotonic_ns=value.get("monotonic_ns"),
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError(
            "correction phase heartbeat drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_phase_structured_exception(
    *,
    phase: str,
    exception_type: str,
    exception_message: str,
    traceback_bytes: bytes,
    capture_source: str,
) -> dict[str, Any]:
    _correction_phase_role(phase)
    if (
        not isinstance(exception_type, str)
        or not exception_type
        or not isinstance(exception_message, str)
        or not isinstance(traceback_bytes, bytes)
        or not traceback_bytes
        or capture_source
        not in {"STDLIB_PHASE_WRAPPER", "EXTERNAL_COORDINATOR_FALLBACK"}
    ):
        raise ForensicCorrectionContractError(
            "correction phase structured exception input drift"
        )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_phase_exception.v1"
            ),
            "phase": phase,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "traceback_sha256": hashlib.sha256(traceback_bytes).hexdigest(),
            "traceback_bytes": len(traceback_bytes),
            "capture_source": capture_source,
            "pass": True,
        }
    )


def validate_correction_phase_structured_exception(
    value: Mapping[str, Any],
    *,
    phase: str,
    expected_traceback_bytes: bytes | None,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    required = {
        "schema", "phase", "exception_type", "exception_message",
        "traceback_sha256", "traceback_bytes", "capture_source", "pass",
        "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_phase_exception.v1"
        )
        or value.get("phase") != phase
        or not isinstance(value.get("exception_type"), str)
        or not value["exception_type"]
        or not isinstance(value.get("exception_message"), str)
        or not isinstance(value.get("traceback_sha256"), str)
        or not _LOWERCASE_SHA256_RE.fullmatch(value["traceback_sha256"])
        or isinstance(value.get("traceback_bytes"), bool)
        or not isinstance(value.get("traceback_bytes"), int)
        or value["traceback_bytes"] <= 0
        or value.get("capture_source")
        not in {"STDLIB_PHASE_WRAPPER", "EXTERNAL_COORDINATOR_FALLBACK"}
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction phase structured exception drift"
        )
    if expected_traceback_bytes is not None and (
        hashlib.sha256(expected_traceback_bytes).hexdigest()
        != value["traceback_sha256"]
        or len(expected_traceback_bytes) != value["traceback_bytes"]
    ):
        raise ForensicCorrectionContractError(
            "correction phase exception/traceback binding drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_phase_last_stage(
    *,
    phase: str,
    pid: int,
    stage_id: str,
    event: str,
    monotonic_ns: int,
) -> dict[str, Any]:
    if (
        phase not in CORRECTION_PHASE_STAGE_IDS
        or stage_id not in CORRECTION_PHASE_STAGE_IDS[phase]
        or event not in {"STARTED", "COMPLETED", "FAILED"}
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns <= 0
    ):
        raise ForensicCorrectionContractError(
            "correction phase last-stage input drift"
        )
    role = (
        CORRECTION_FINALIZER_ROLE if phase == "FINALIZER"
        else CORRECTION_CHECKER_ROLE
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_phase_last_stage.v1"
            ),
            "phase": phase,
            "producer_role": role,
            "stage_id": stage_id,
            "event": event,
            "pid": pid,
            "monotonic_ns": monotonic_ns,
        }
    )


def validate_correction_phase_last_stage(
    value: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    rebuilt = build_correction_phase_last_stage(
        phase=phase,
        pid=value.get("pid"),
        stage_id=value.get("stage_id"),
        event=value.get("event"),
        monotonic_ns=value.get("monotonic_ns"),
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError(
            "correction phase last-stage drift"
        )
    return copy.deepcopy(dict(value))


def build_correction_phase_wrapper_envelope(
    *,
    phase: str,
    phase_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
    environment_custody: Mapping[str, Any],
    observed_cwd: str,
    wrapper_heartbeat: Sequence[Mapping[str, Any]],
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    last_stage_marker: Mapping[str, Any],
    phase_receipt: Mapping[str, Any] | None,
    structured_exception: Mapping[str, Any] | None,
    envelope_source: str,
) -> dict[str, Any]:
    identity = _validate_correction_phase_identity(
        phase_process_identity, phase=phase, require_current=False
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity, require_live=False
    )
    environment = validate_correction_phase_environment_custody(
        environment_custody, expected_environment=None
    )
    heartbeat = [
        validate_correction_phase_heartbeat_row(row, phase=phase)
        for row in wrapper_heartbeat
    ]
    marker = validate_correction_phase_last_stage(
        last_stage_marker, phase=phase
    )
    if (
        len(heartbeat) != 2
        or [row["sequence"] for row in heartbeat] != [0, 1]
        or heartbeat[0]["event"] != "WRAPPER_STARTED"
        or any(row["pid"] != identity["pid"] for row in heartbeat)
        or marker["pid"] != identity["pid"]
        or isinstance(started_monotonic_ns, bool)
        or not isinstance(started_monotonic_ns, int)
        or isinstance(ended_monotonic_ns, bool)
        or not isinstance(ended_monotonic_ns, int)
        or not (
            0 < started_monotonic_ns
            <= heartbeat[0]["monotonic_ns"]
            <= marker["monotonic_ns"]
            <= heartbeat[1]["monotonic_ns"]
            <= ended_monotonic_ns
        )
        or envelope_source
        not in {"STDLIB_PHASE_WRAPPER", "EXTERNAL_COORDINATOR_FALLBACK"}
        or observed_cwd != str(CORRECTION_REPO_ROOT)
    ):
        raise ForensicCorrectionContractError(
            "correction phase wrapper envelope timing drift"
        )
    phase_success = phase_receipt is not None
    if phase_success:
        if structured_exception is not None:
            raise ForensicCorrectionContractError(
                "successful correction phase has exception evidence"
            )
        BASE.validate_self_digest(phase_receipt)
        if (
            heartbeat[1]["event"] != "WRAPPER_COMPLETED"
            or marker["stage_id"] != "COMPLETE"
            or marker["event"] != "COMPLETED"
            or envelope_source != "STDLIB_PHASE_WRAPPER"
        ):
            raise ForensicCorrectionContractError(
                "successful correction phase wrapper envelope drift"
            )
        exception_value = None
        receipt_value = copy.deepcopy(dict(phase_receipt))
    else:
        if structured_exception is None:
            raise ForensicCorrectionContractError(
                "failed correction phase lacks structured exception"
            )
        exception_value = validate_correction_phase_structured_exception(
            structured_exception,
            phase=phase,
            expected_traceback_bytes=None,
        )
        if (
            heartbeat[1]["event"] != "WRAPPER_FAILED"
            or marker["event"] != "FAILED"
        ):
            raise ForensicCorrectionContractError(
                "failed correction phase wrapper envelope drift"
            )
        receipt_value = None
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_PHASE_WRAPPER_ENVELOPE_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": phase,
            "phase_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "outer_exact_argv": (
                expected_correction_finalizer_outer_argv()
                if phase == "FINALIZER"
                else expected_correction_checker_outer_argv()
            ),
            "inner_evaluator_argv": (
                expected_correction_finalizer_inner_argv()
                if phase == "FINALIZER"
                else expected_correction_checker_inner_argv()
            ),
            "environment_custody": environment,
            "observed_cwd": observed_cwd,
            "wrapper_heartbeat": heartbeat,
            "started_monotonic_ns": started_monotonic_ns,
            "ended_monotonic_ns": ended_monotonic_ns,
            "runtime_ns": ended_monotonic_ns - started_monotonic_ns,
            "last_stage_marker": marker,
            "phase_receipt": receipt_value,
            "structured_exception": exception_value,
            "phase_success": phase_success,
            "read_guard_events": [],
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "envelope_source": envelope_source,
            "pass_meaning": (
                "COMPLETE_WRAPPER_EVIDENCE_INDEPENDENT_OF_PHASE_SUCCESS"
            ),
            "pass": True,
        }
    )


def validate_correction_phase_wrapper_envelope(
    value: Mapping[str, Any],
    *,
    phase: str,
    expected_process_identity: Mapping[str, Any],
    expected_coordinator_identity: Mapping[str, Any],
    expected_environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    identity = _validate_correction_phase_identity(
        expected_process_identity, phase=phase, require_current=False
    )
    environment = validate_correction_phase_environment_custody(
        value.get("environment_custody"),
        expected_environment=expected_environment,
    )
    rebuilt = build_correction_phase_wrapper_envelope(
        phase=phase,
        phase_process_identity=identity,
        coordinator_process_identity=expected_coordinator_identity,
        environment_custody=environment,
        observed_cwd=value.get("observed_cwd"),
        wrapper_heartbeat=value.get("wrapper_heartbeat"),
        started_monotonic_ns=value.get("started_monotonic_ns"),
        ended_monotonic_ns=value.get("ended_monotonic_ns"),
        last_stage_marker=value.get("last_stage_marker"),
        phase_receipt=value.get("phase_receipt"),
        structured_exception=value.get("structured_exception"),
        envelope_source=value.get("envelope_source"),
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError(
            "correction phase wrapper envelope drift"
        )
    return copy.deepcopy(dict(value))


def _correction_process_group_members(process_group_id: int) -> list[int]:
    if (
        isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
    ):
        raise ForensicCorrectionContractError(
            "correction process-group identity drift"
        )
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            payload = (entry / "stat").read_text(encoding="utf-8")
            _prefix, suffix = payload.rsplit(")", 1)
            fields = suffix.strip().split()
            if int(fields[2]) == process_group_id:
                members.append(int(entry.name))
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
            ValueError,
            IndexError,
        ):
            continue
    return sorted(members)


def build_correction_phase_cleanup(
    *,
    repo_root: str | Path,
    phase_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
    require_coordinator_live: bool,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    identity = BASE.validate_process_identity(phase_process_identity)
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity,
        require_live=require_coordinator_live,
    )
    if require_coordinator_live and coordinator["pid"] != os.getpid():
        raise ForensicCorrectionContractError(
            "phase cleanup is not owned by the bound coordinator"
        )
    if Path(f"/proc/{identity['pid']}").exists():
        raise ForensicCorrectionContractError(
            "correction phase process remains live after wait"
        )
    group_members = _correction_process_group_members(
        identity["process_group_id"]
    )
    matches = BASE._active_forensic_or_scientific_processes(
        root, exclude_pids={os.getpid()}
    )
    coordinators = _active_correction_terminal_coordinators(
        exclude_pids={os.getpid(), coordinator["pid"]}
    )
    if group_members or matches or coordinators:
        raise ForensicCorrectionContractError(
            "correction role remains live after phase exit"
        )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_phase_cleanup.v1"
            ),
            "phase_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "coordinator_was_live_at_original_cleanup_build": True,
            "phase_process_absent_after_wait": True,
            "phase_process_group_members_after_wait": [],
            "exact_other_forensic_or_scientific_role_matches_after_wait": [],
            "other_correction_terminal_coordinator_matches_after_wait": [],
            "current_root_orchestrator_excluded_from_scan": True,
            "literal_zero_all_producer_roles": True,
            "pass": True,
        }
    )


def _correction_phase_stream_binding(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise ForensicCorrectionContractError(
            "correction phase stream is not raw bytes"
        )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "base64": base64.b64encode(payload).decode("ascii"),
        "raw_bytes_preserved": True,
    }


def _correction_phase_stream_bytes(value: Mapping[str, Any]) -> bytes:
    if (
        type(value) is not dict
        or set(value)
        != {"sha256", "bytes", "base64", "raw_bytes_preserved"}
        or not isinstance(value.get("sha256"), str)
        or not _LOWERCASE_SHA256_RE.fullmatch(value["sha256"])
        or isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
        or not isinstance(value.get("base64"), str)
        or value.get("raw_bytes_preserved") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction phase stream binding drift"
        )
    try:
        payload = base64.b64decode(value["base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ForensicCorrectionContractError(
            "correction phase stream base64 drift"
        ) from exc
    if (
        len(payload) != value["bytes"]
        or hashlib.sha256(payload).hexdigest() != value["sha256"]
    ):
        raise ForensicCorrectionContractError(
            "correction phase stream byte binding drift"
        )
    return payload


def _correction_prior_phase_binding(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    BASE.validate_self_digest(value)
    payload = _authority_bytes(value)
    return {
        "phase": value.get("phase"),
        "content_digest": value["content_digest"],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def build_correction_phase_exit_custody(
    *,
    repo_root: str | Path,
    phase: str,
    phase_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
    require_coordinator_live: bool,
    phase_environment: Mapping[str, str] | None,
    parent_observed_phase_cwd: str,
    parent_observed_phase_ppid: int,
    wrapper_envelope: Mapping[str, Any],
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    traceback_bytes: bytes,
    parent_spawn_requested_monotonic_ns: int,
    parent_spawned_monotonic_ns: int,
    parent_popen_returned_monotonic_ns: int,
    parent_identity_bound_monotonic_ns: int,
    parent_ended_monotonic_ns: int,
    deadline_expired: bool,
    returncode: int,
    termination: Mapping[str, Any],
    prior_phase_exit_custody: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = _validate_correction_phase_identity(
        phase_process_identity, phase=phase, require_current=False
    )
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity,
        require_live=require_coordinator_live,
    )
    envelope = validate_correction_phase_wrapper_envelope(
        wrapper_envelope,
        phase=phase,
        expected_process_identity=identity,
        expected_coordinator_identity=coordinator,
        expected_environment=phase_environment,
    )
    term = BASE._validate_termination(termination)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                parent_spawned_monotonic_ns,
                parent_spawn_requested_monotonic_ns,
                parent_popen_returned_monotonic_ns,
                parent_identity_bound_monotonic_ns,
                parent_ended_monotonic_ns,
                returncode,
            )
        )
        or not isinstance(deadline_expired, bool)
        or not (
            0 < parent_spawn_requested_monotonic_ns
            == parent_spawned_monotonic_ns
            <= parent_popen_returned_monotonic_ns
            <= parent_identity_bound_monotonic_ns
            <= parent_ended_monotonic_ns
        )
        or envelope["started_monotonic_ns"] < parent_spawned_monotonic_ns
        or envelope["ended_monotonic_ns"] > parent_ended_monotonic_ns
        or parent_observed_phase_cwd != str(CORRECTION_REPO_ROOT)
        or envelope["observed_cwd"] != parent_observed_phase_cwd
        or parent_observed_phase_ppid != coordinator["pid"]
        or int(identity["process_group_id"]) != int(identity["pid"])
        or envelope["root_coordinator_process_identity"] != coordinator
    ):
        raise ForensicCorrectionContractError(
            "correction phase parent-observed custody drift"
        )
    expected_stdout = _authority_bytes(envelope)
    if (
        envelope["phase_receipt"] is not None
        and envelope["phase_receipt"].get(
            "root_coordinator_process_identity"
        )
        != coordinator
    ):
        raise ForensicCorrectionContractError(
            "phase receipt/coordinator cross-binding drift"
        )
    phase_success = bool(
        envelope["phase_success"] and returncode == 0 and not deadline_expired
    )
    if phase == "FINALIZER":
        if prior_phase_exit_custody is not None:
            raise ForensicCorrectionContractError(
                "finalizer has a future prior-phase dependency"
            )
        prior_binding = None
    else:
        if prior_phase_exit_custody is None:
            raise ForensicCorrectionContractError(
                "checker lacks finalizer exit-custody dependency"
            )
        BASE.validate_self_digest(prior_phase_exit_custody)
        if (
            prior_phase_exit_custody.get("phase") != "FINALIZER"
            or prior_phase_exit_custody.get("phase_success") is not True
            or prior_phase_exit_custody.get(
                "root_coordinator_process_identity"
            )
            != coordinator
            or prior_phase_exit_custody.get("parent_ended_monotonic_ns")
            >= parent_spawned_monotonic_ns
            or prior_phase_exit_custody.get("phase_process_identity", {}).get(
                "pid"
            )
            == identity["pid"]
        ):
            raise ForensicCorrectionContractError(
                "checker/finalizer phase ordering drift"
            )
        prior_binding = _correction_prior_phase_binding(
            prior_phase_exit_custody
        )
    structured = envelope["structured_exception"]
    if phase_success:
        if (
            stdout_bytes != expected_stdout
            or stderr_bytes
            or traceback_bytes
            or structured is not None
            or envelope["phase_receipt"] is None
            or term
            != {
                "kind": "EXIT",
                "exit_code_or_null": 0,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            }
        ):
            raise ForensicCorrectionContractError(
                "successful correction phase exit custody drift"
            )
    else:
        if envelope["phase_receipt"] is not None or structured is None:
            raise ForensicCorrectionContractError(
                "failed correction phase receipt ambiguity"
            )
        if returncode == 0 or term == {
            "kind": "EXIT",
            "exit_code_or_null": 0,
            "signal_number_or_null": None,
            "signal_name_or_null": None,
        }:
            raise ForensicCorrectionContractError(
                "failed correction phase exited successfully"
            )
        validate_correction_phase_structured_exception(
            structured,
            phase=phase,
            expected_traceback_bytes=traceback_bytes,
        )
        if not traceback_bytes:
            raise ForensicCorrectionContractError(
                "failed correction phase lacks traceback custody"
            )
    cleanup = build_correction_phase_cleanup(
        repo_root=repo_root,
        phase_process_identity=identity,
        coordinator_process_identity=coordinator,
        require_coordinator_live=require_coordinator_live,
    )
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "phase": phase,
            "phase_process_identity": identity,
            "root_coordinator_process_identity": coordinator,
            "phase_parent_pid": coordinator["pid"],
            "outer_exact_argv": (
                expected_correction_finalizer_outer_argv()
                if phase == "FINALIZER"
                else expected_correction_checker_outer_argv()
            ),
            "inner_evaluator_argv": (
                expected_correction_finalizer_inner_argv()
                if phase == "FINALIZER"
                else expected_correction_checker_inner_argv()
            ),
            "environment_custody": envelope["environment_custody"],
            "parent_observed_phase_cwd": parent_observed_phase_cwd,
            "parent_observed_phase_ppid": parent_observed_phase_ppid,
            "wrapper_envelope": envelope,
            "streams": {
                "stdout": _correction_phase_stream_binding(stdout_bytes),
                "stderr": _correction_phase_stream_binding(stderr_bytes),
                "traceback": _correction_phase_stream_binding(
                    traceback_bytes
                ),
            },
            "structured_exception": structured,
            "last_stage_marker": envelope["last_stage_marker"],
            "parent_spawned_monotonic_ns": parent_spawned_monotonic_ns,
            "parent_spawn_requested_monotonic_ns": (
                parent_spawn_requested_monotonic_ns
            ),
            "parent_popen_returned_monotonic_ns": (
                parent_popen_returned_monotonic_ns
            ),
            "parent_identity_bound_monotonic_ns": (
                parent_identity_bound_monotonic_ns
            ),
            "parent_ended_monotonic_ns": parent_ended_monotonic_ns,
            "start_new_session": True,
            "phase_process_group_leader": True,
            "runtime_timeout_seconds": (
                CORRECTION_PHASE_RUNTIME_TIMEOUT_SECONDS
            ),
            "term_grace_seconds": CORRECTION_PHASE_TERM_GRACE_SECONDS,
            "kill_grace_seconds": CORRECTION_PHASE_KILL_GRACE_SECONDS,
            "deadline_expired": deadline_expired,
            "returncode": returncode,
            "termination": term,
            "cleanup": cleanup,
            "phase_receipt": envelope["phase_receipt"],
            "prior_phase_exit_custody_binding": prior_binding,
            "phase_success": phase_success,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "COMPLETE_EXTERNAL_PHASE_CUSTODY_INDEPENDENT_OF_PHASE_SUCCESS"
            ),
            "pass": True,
        }
    )


def validate_correction_phase_exit_custody(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_phase: str,
    expected_phase_receipt: Mapping[str, Any] | None,
    prior_phase_exit_custody: Mapping[str, Any] | None,
    require_coordinator_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    streams = value.get("streams")
    if type(streams) is not dict or set(streams) != {
        "stdout", "stderr", "traceback"
    }:
        raise ForensicCorrectionContractError(
            "correction phase stream set drift"
        )
    stdout = _correction_phase_stream_bytes(streams["stdout"])
    stderr = _correction_phase_stream_bytes(streams["stderr"])
    traceback_payload = _correction_phase_stream_bytes(streams["traceback"])
    envelope = value.get("wrapper_envelope")
    if envelope.get("phase_receipt") != expected_phase_receipt:
        raise ForensicCorrectionContractError(
            "correction phase expected receipt drift"
        )
    environment = value.get("environment_custody")
    validate_correction_phase_environment_custody(
        environment, expected_environment=None
    )
    rebuilt = build_correction_phase_exit_custody(
        repo_root=repo_root,
        phase=expected_phase,
        phase_process_identity=value.get("phase_process_identity"),
        coordinator_process_identity=value.get(
            "root_coordinator_process_identity"
        ),
        require_coordinator_live=require_coordinator_live,
        phase_environment=None,
        parent_observed_phase_cwd=value.get("parent_observed_phase_cwd"),
        parent_observed_phase_ppid=value.get("parent_observed_phase_ppid"),
        wrapper_envelope=envelope,
        stdout_bytes=stdout,
        stderr_bytes=stderr,
        traceback_bytes=traceback_payload,
        parent_spawn_requested_monotonic_ns=value.get(
            "parent_spawn_requested_monotonic_ns"
        ),
        parent_spawned_monotonic_ns=value.get(
            "parent_spawned_monotonic_ns"
        ),
        parent_popen_returned_monotonic_ns=value.get(
            "parent_popen_returned_monotonic_ns"
        ),
        parent_identity_bound_monotonic_ns=value.get(
            "parent_identity_bound_monotonic_ns"
        ),
        parent_ended_monotonic_ns=value.get("parent_ended_monotonic_ns"),
        deadline_expired=value.get("deadline_expired"),
        returncode=value.get("returncode"),
        termination=value.get("termination"),
        prior_phase_exit_custody=prior_phase_exit_custody,
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError(
            "correction phase exit custody drift"
        )
    return copy.deepcopy(dict(value))


def correction_result_artifact_payloads(
    *,
    repo_root: str | Path,
    finalizer_phase_custody: Mapping[str, Any],
    checker_phase_custody: Mapping[str, Any],
    repository_mode: str,
) -> dict[str, Any]:
    """Rebuild the exact nine tracked bytes without writing them."""

    root = Path(repo_root).resolve()
    finalizer_receipt = finalizer_phase_custody.get("phase_receipt")
    checker_receipt = checker_phase_custody.get("phase_receipt")
    finalizer = validate_correction_finalizer_prospective_receipt(
        finalizer_receipt,
        repo_root=root,
        require_current_identity=False,
        repository_mode=repository_mode,
    )
    checker = validate_correction_external_checker_receipt(
        checker_receipt,
        repo_root=root,
        require_current_identity=False,
        repository_mode=repository_mode,
    )
    finalizer_custody = validate_correction_phase_exit_custody(
        finalizer_phase_custody,
        repo_root=root,
        expected_phase="FINALIZER",
        expected_phase_receipt=finalizer,
        prior_phase_exit_custody=None,
        require_coordinator_live=False,
    )
    checker_custody = validate_correction_phase_exit_custody(
        checker_phase_custody,
        repo_root=root,
        expected_phase="CHECKER",
        expected_phase_receipt=checker,
        prior_phase_exit_custody=finalizer_custody,
        require_coordinator_live=False,
    )
    if (
        finalizer_custody["phase_success"] is not True
        or checker_custody["phase_success"] is not True
        or finalizer["prospective_artifact_bindings"]
        != checker["independently_rebuilt_artifact_bindings"]
        or finalizer["prospective_artifact_set_sha256"]
        != checker["prospective_artifact_set_sha256"]
    ):
        raise ForensicCorrectionContractError(
            "finalizer/checker prospective artifact disagreement"
        )
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode=repository_mode,
    )
    core = build_correction_core_result_artifacts_from_bundle(bundle)
    observed = _correction_artifact_binding_map(core["tracked_payloads"])
    if observed != finalizer["prospective_artifact_bindings"]:
        raise ForensicCorrectionContractError(
            "phase receipts do not bind rebuilt core artifacts"
        )
    payloads = {
        **core["tracked_payloads"],
        TRACKED_CORRECTION_FINALIZER_CUSTODY_PATH: _authority_bytes(
            finalizer_custody
        ),
        TRACKED_CORRECTION_CHECKER_CUSTODY_PATH: _authority_bytes(
            checker_custody
        ),
    }
    if {str(path) for path in payloads} != set(CORRECTION_RESULT_PATHS):
        raise ForensicCorrectionContractError(
            "correction tracked result path-set drift"
        )
    return {
        **core,
        "finalizer_phase_custody": finalizer_custody,
        "checker_phase_custody": checker_custody,
        "tracked_payloads": payloads,
        "tracked_bindings": _correction_artifact_binding_map(payloads),
        "success_literals_emitted": False,
        "pass": True,
    }


def _correction_result_path_absence_receipt(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    rows = [
        {
            "path": path,
            "absent_by_anchored_no_follow_stat": (
                BASE._relative_leaf_absent_no_follow(root, Path(path))
            ),
        }
        for path in sorted(CORRECTION_RESULT_PATHS)
    ]
    if any(
        row["absent_by_anchored_no_follow_stat"] is not True
        for row in rows
    ):
        raise ForensicCorrectionContractError(
            "correction result path absence proof failed"
        )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_result_path_absence.v1"
            ),
            "repo_root": str(root),
            "paths": rows,
            "path_count": len(rows),
            "all_absent_including_dangling_symlinks": True,
            "pass": True,
        }
    )


def validate_correction_result_path_absence_receipt(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    root = Path(repo_root).resolve()
    rows = [
        {
            "path": path,
            "absent_by_anchored_no_follow_stat": True,
        }
        for path in sorted(CORRECTION_RESULT_PATHS)
    ]
    expected = BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_result_path_absence.v1"
            ),
            "repo_root": str(root),
            "paths": rows,
            "path_count": len(rows),
            "all_absent_including_dangling_symlinks": True,
            "pass": True,
        }
    )
    if dict(value) != expected:
        raise ForensicCorrectionContractError(
            "correction result path absence receipt drift"
        )
    if reverify_live and _correction_result_path_absence_receipt(root) != expected:
        raise ForensicCorrectionContractError(
            "correction result path live absence drift"
        )
    return copy.deepcopy(dict(value))


def rollback_correction_result_artifacts(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Remove only the exact nine staged result paths and prove absence."""

    root = Path(repo_root).resolve()
    BASE._rollback_relative_paths_no_follow(
        root,
        [Path(path) for path in sorted(CORRECTION_RESULT_PATHS)],
        label="forensic correction result precommit staging",
    )
    return _correction_result_path_absence_receipt(root)


def write_correction_result_artifacts(
    *,
    repo_root: str | Path,
    coordinator_process_identity: Mapping[str, Any],
    finalizer_phase_custody: Mapping[str, Any],
    checker_phase_custody: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage nine paths with catchable-failure rollback after both phases."""

    root = Path(repo_root).resolve()
    coordinator = validate_correction_terminal_coordinator_identity(
        coordinator_process_identity, require_live=True
    )
    if coordinator["pid"] != os.getpid() or (
        _active_correction_terminal_coordinators(
            exclude_pids={coordinator["pid"]}
        )
    ):
        raise ForensicCorrectionContractError(
            "correction result staging coordinator custody drift"
        )
    artifacts = correction_result_artifact_payloads(
        repo_root=root,
        finalizer_phase_custody=finalizer_phase_custody,
        checker_phase_custody=checker_phase_custody,
        repository_mode="PREPUBLICATION_FREEZE_HEAD",
    )
    if (
        artifacts["finalizer_phase_custody"][
            "root_coordinator_process_identity"
        ]
        != coordinator
        or artifacts["checker_phase_custody"][
            "root_coordinator_process_identity"
        ]
        != coordinator
    ):
        raise ForensicCorrectionContractError(
            "phase custody/current staging coordinator drift"
        )
    payloads: Mapping[Path, bytes] = artifacts["tracked_payloads"]
    freeze_head = artifacts["diagnostic_custody"]["source_commit"]
    if (
        BASE._git_text(root, ("rev-parse", "HEAD")) != freeze_head
        or BASE._git_text(root, ("status", "--porcelain=v1"))
        or BASE._active_forensic_or_scientific_processes(
            root, exclude_pids={os.getpid()}
        )
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError(
            "correction result prepublication custody drift"
        )
    publication_order = sorted(payloads, key=str)
    stale = [
        str(path)
        for path in publication_order
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if stale:
        raise ForensicCorrectionContractError(
            f"correction result paths are stale: {stale}"
        )
    try:
        for relative in publication_order:
            BASE._exclusive_write(root / relative, payloads[relative])
        for relative in publication_order:
            observed = BASE._read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
            if observed != payloads[relative]:
                raise ForensicCorrectionContractError(
                    f"correction result byte drift: {relative}"
                )
    except BaseException as original:
        try:
            BASE._rollback_relative_paths_no_follow(
                root,
                publication_order,
                label="forensic correction result publication",
            )
        except BaseException as cleanup_exc:
            raise ForensicCorrectionContractError(
                "correction result publication rollback failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            ) from original
        raise
    postwrite_dirty_paths = _correction_dirty_paths(root)
    postwrite_roles = BASE._active_forensic_or_scientific_processes(
        root, exclude_pids={coordinator["pid"]}
    )
    postwrite_coordinators = _active_correction_terminal_coordinators(
        exclude_pids={coordinator["pid"]}
    )
    if (
        postwrite_dirty_paths != sorted(CORRECTION_RESULT_PATHS)
        or postwrite_roles
        or postwrite_coordinators
    ):
        try:
            BASE._rollback_relative_paths_no_follow(
                root,
                publication_order,
                label="forensic correction result postwrite validation",
            )
        except BaseException as cleanup_exc:
            raise ForensicCorrectionContractError(
                "correction result postwrite rollback failed"
            ) from cleanup_exc
        raise ForensicCorrectionContractError(
            "correction result postwrite custody drift"
        )
    return {
        "source_freeze_commit": freeze_head,
        "result": artifacts["result"],
        "tracked_paths": [str(path) for path in publication_order],
        "tracked_bindings": _correction_artifact_binding_map(payloads),
        "postwrite_dirty_paths": postwrite_dirty_paths,
        "postwrite_other_role_matches": [],
        "postwrite_other_coordinator_matches": [],
        "staging_semantics": (
            "PRECOMMIT_STAGING_WITH_ROLLBACK_FOR_CATCHABLE_FAILURES"
        ),
        "publication_boundary": (
            "EXACT_GIT_RESULT_COMMIT_AND_POSTCOMMIT_REPLAY_ONLY"
        ),
        "required_commit_subject": CORRECTION_RESULT_COMMIT_SUBJECT,
        "success_literals_emitted": False,
        "v2_execution_authorized": False,
        "pass": True,
    }


def validate_correction_result_commit(
    *,
    repo_root: str | Path,
    postcommit_process_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the exact result commit and emit the only success disposition."""

    root = Path(repo_root).resolve()
    attestor = validate_correction_result_commit_validation_identity(
        postcommit_process_identity, require_live=True
    )
    if attestor["pid"] != os.getpid() or _observe_correction_process_cwd(
        attestor
    ) != str(root):
        raise ForensicCorrectionContractError(
            "postcommit replay current attestor custody drift"
        )
    pre_read_role_scan = _observe_correction_role_scan_custody(
        root,
        gate="POSTCOMMIT_BEFORE_RESULT_READS",
        excluded_pids={attestor["pid"]},
    )
    if pre_read_role_scan["all_nonexcluded_roles_zero"] is not True:
        raise ForensicCorrectionContractError(
            "postcommit correction role drift before result reads"
        )
    head = BASE._git_text(root, ("rev-parse", "HEAD"))
    parents = BASE._git_text(
        root, ("rev-list", "--parents", "-n", "1", head)
    ).split()
    subject = BASE._git_text(root, ("show", "-s", "--format=%s", head))
    if len(parents) != 2:
        raise ForensicCorrectionContractError(
            "correction result commit is not a sole-parent commit"
        )
    freeze_head = parents[1]
    freeze_git_custody = _validate_correction_freeze_commit_git_custody(
        root, freeze_head
    )
    changed = sorted(
        row for row in BASE._git_text(
            root, ("diff", "--name-only", freeze_head, head)
        ).splitlines() if row
    )
    if (
        subject != CORRECTION_RESULT_COMMIT_SUBJECT
        or changed != sorted(CORRECTION_RESULT_PATHS)
        or BASE._git_text(root, ("status", "--porcelain=v1"))
        or not _future_v2_script_absent()
    ):
        raise ForensicCorrectionContractError(
            "correction result commit Git custody drift"
        )
    for ancestor in PRESERVED_LINEAGE.values():
        if not BASE._git_is_ancestor(root, ancestor, freeze_head):
            raise ForensicCorrectionContractError(
                "correction result lineage drift"
            )
    finalizer_payload = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_FINALIZER_CUSTODY_PATH,
        expected_stat_rows=None,
    )
    checker_payload = BASE._read_no_follow_bound_file(
        root, TRACKED_CORRECTION_CHECKER_CUSTODY_PATH,
        expected_stat_rows=None,
    )
    finalizer_custody = BASE._canonical_json_from_bound_bytes(
        finalizer_payload, label="correction finalizer custody"
    )
    checker_custody = BASE._canonical_json_from_bound_bytes(
        checker_payload, label="correction checker custody"
    )
    artifacts = correction_result_artifact_payloads(
        repo_root=root,
        finalizer_phase_custody=finalizer_custody,
        checker_phase_custody=checker_custody,
        repository_mode="POSTCOMMIT_RESULT_HEAD",
    )
    if artifacts["diagnostic_custody"]["source_commit"] != freeze_head:
        raise ForensicCorrectionContractError(
            "correction result direct-parent binding drift"
        )
    for relative, expected in artifacts["tracked_payloads"].items():
        if BASE._read_no_follow_bound_file(
            root, relative, expected_stat_rows=None
        ) != expected:
            raise ForensicCorrectionContractError(
                f"committed correction result byte drift: {relative}"
            )
    post_read_role_scan = _observe_correction_role_scan_custody(
        root,
        gate="POSTCOMMIT_AFTER_RESULT_READS",
        excluded_pids={attestor["pid"]},
    )
    scientific_root = BASE.BASE.OUTPUT_ROOT
    attempts = list(
        scientific_root.parent.glob(f".{scientific_root.name}.attempt-*")
    )
    if (
        post_read_role_scan["all_nonexcluded_roles_zero"] is not True
        or scientific_root.exists()
        or scientific_root.is_symlink()
        or attempts
    ):
        raise ForensicCorrectionContractError(
            "postcommit correction process/science namespace drift"
        )
    disposition = BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_postcommit_disposition.v1"
            ),
            "result_commit": head,
            "source_freeze_commit": freeze_head,
            "source_freeze_commit_git_custody": freeze_git_custody,
            "commit_subject": subject,
            "changed_paths": changed,
            "postcommit_attestor_process_identity": attestor,
            "postcommit_attestor_live_at_disposition_build": True,
            "pre_result_read_role_scan_custody": pre_read_role_scan,
            "post_result_read_role_scan_custody": post_read_role_scan,
            "technical_success_classification": (
                "V2_PREEXECUTION_PATH_QUALIFIED"
            ),
            "next_pass_authorization": (
                "PLAN_AWARE_MONOTONE_JEPA_COST_V2_"
                "TECHNICALLY_AUTHORISED_FOR_NEXT_PASS"
            ),
            "external_checker_exited": True,
            "literal_zero_all_other_forensic_and_scientific_roles": True,
            "literal_zero_all_roles_claimed_while_attestor_live": False,
            "worktree_clean": True,
            "tracked_bytes_equal_rebuilt_artifact_set": True,
            "future_v2_script_present": False,
            "future_v2_spec_only_command": copy.deepcopy(
                FUTURE_V2_SPEC_ONLY_COMMAND
            ),
            "v2_execution_authorized": False,
            "scientific_attempt_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "technical_diagnostic_consumed_by_launcher_process_identity": (
                copy.deepcopy(
                    artifacts["diagnostic_custody"][
                        "launcher_process_identity"
                    ]
                )
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )
    return disposition


def _correction_terminal_phase_environment(
    coordinator: Mapping[str, Any],
) -> dict[str, str]:
    identity = validate_correction_terminal_coordinator_identity(
        coordinator, require_live=True
    )
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("PYTHON"):
            environment.pop(key, None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "VIRTUAL_ENV": str(BASE.FORENSIC_INTERPRETER.parent.parent),
            "LEWM_CORRECTION_COORDINATOR_PID": str(identity["pid"]),
            "LEWM_CORRECTION_COORDINATOR_START_TIME_TICKS": str(
                identity["start_time_ticks"]
            ),
        }
    )
    inherited_path = environment.get("PATH", "")
    environment["PATH"] = str(BASE.FORENSIC_INTERPRETER.parent) + (
        os.pathsep + inherited_path if inherited_path else ""
    )
    build_correction_phase_environment_custody(environment)
    return environment


def _observe_correction_phase_process(
    *, pid: int, phase: str
) -> dict[str, Any]:
    argv = _correction_proc_argv(pid)
    ppid, process_group_id, start_time_ticks = _correction_proc_stat(pid)
    executable = str(
        (Path("/proc") / str(pid) / "exe").resolve(strict=True)
    )
    cwd = str((Path("/proc") / str(pid) / "cwd").resolve(strict=True))
    expected_argv = (
        expected_correction_finalizer_outer_argv()
        if phase == "FINALIZER"
        else expected_correction_checker_outer_argv()
    )
    identity = _validate_correction_phase_identity(
        {
            "pid": pid,
            "process_group_id": process_group_id,
            "start_time_ticks": start_time_ticks,
            "argv": argv,
            "argv_sha256": BASE.canonical_json_sha256(argv),
            "executable": executable,
            "role": _correction_phase_role(phase),
        },
        phase=phase,
        require_current=False,
    )
    if argv != expected_argv or cwd != str(CORRECTION_REPO_ROOT):
        raise ForensicCorrectionContractError(
            "correction phase /proc argv/cwd drift"
        )
    return {
        "process_identity": identity,
        "parent_pid": ppid,
        "cwd": cwd,
    }


def _correction_phase_termination(
    *,
    returncode: int,
    timed_out: bool,
    timeout_cleanup_actions: Sequence[str],
) -> dict[str, Any]:
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ForensicCorrectionContractError(
            "correction phase returncode drift"
        )
    if timed_out and timeout_cleanup_actions:
        signal_number = (
            int(signal.SIGKILL)
            if any("SIGKILL" in action for action in timeout_cleanup_actions)
            else int(signal.SIGTERM)
        )
        return {
            "kind": "TIMEOUT",
            "exit_code_or_null": None,
            "signal_number_or_null": signal_number,
            "signal_name_or_null": signal.Signals(signal_number).name,
        }
    if returncode >= 0:
        return {
            "kind": "EXIT",
            "exit_code_or_null": returncode,
            "signal_number_or_null": None,
            "signal_name_or_null": None,
        }
    signal_number = -returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"SIGNAL_{signal_number}"
    return {
        "kind": "SIGNAL",
        "exit_code_or_null": None,
        "signal_number_or_null": signal_number,
        "signal_name_or_null": signal_name,
    }


def _terminate_correction_phase_group(
    process: subprocess.Popen[bytes],
) -> list[str]:
    actions: list[str] = []
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            actions.append("SIGTERM")
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=CORRECTION_PHASE_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                actions.append("SIGKILL")
            except ProcessLookupError:
                pass
            process.wait(timeout=CORRECTION_PHASE_KILL_GRACE_SECONDS)
    members = _correction_process_group_members(process.pid)
    if members:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            actions.append("SIGKILL_REMAINING_GROUP")
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + CORRECTION_PHASE_KILL_GRACE_SECONDS
        while time.monotonic() < deadline:
            if not _correction_process_group_members(process.pid):
                break
            time.sleep(0.01)
    if _correction_process_group_members(process.pid):
        raise ForensicCorrectionContractError(
            "correction phase process group survived bounded cleanup"
        )
    return actions


def _canonical_phase_wrapper_envelope_from_stdout(
    stdout_bytes: bytes,
    *,
    phase: str,
    phase_process_identity: Mapping[str, Any],
    coordinator_process_identity: Mapping[str, Any],
    phase_environment: Mapping[str, str],
) -> dict[str, Any]:
    try:
        decoded = stdout_bytes.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicCorrectionContractError(
            "correction wrapper stdout envelope parse drift"
        ) from exc
    if type(value) is not dict or stdout_bytes != _authority_bytes(value):
        raise ForensicCorrectionContractError(
            "correction wrapper stdout envelope is not canonical"
        )
    return validate_correction_phase_wrapper_envelope(
        value,
        phase=phase,
        expected_process_identity=phase_process_identity,
        expected_coordinator_identity=coordinator_process_identity,
        expected_environment=phase_environment,
    )


def _build_coordinator_observed_phase_failure(
    *,
    phase: str,
    coordinator: Mapping[str, Any],
    fresh_technical_manifest: Mapping[str, Any],
    phase_environment: Mapping[str, str],
    outer_argv: Sequence[str],
    process_observation: Mapping[str, Any] | None,
    spawned_process_group_id: int | None,
    spawn_requested_monotonic_ns: int | None,
    spawned_monotonic_ns: int | None,
    popen_returned_monotonic_ns: int | None,
    ended_monotonic_ns: int,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    traceback_bytes: bytes,
    returncode: int | None,
    termination: Mapping[str, Any] | None,
    cleanup_actions: Sequence[str],
    deadline_expired: bool,
    exception_type: str,
    exception_message: str,
) -> dict[str, Any]:
    traceback_payload = traceback_bytes or (
        f"{exception_type}: {exception_message}\n".encode("utf-8")
    )
    structured = build_correction_phase_structured_exception(
        phase=phase,
        exception_type=exception_type,
        exception_message=exception_message,
        traceback_bytes=traceback_payload,
        capture_source="EXTERNAL_COORDINATOR_FALLBACK",
    )
    manifest = validate_correction_fresh_technical_root_manifest(
        fresh_technical_manifest,
        reverify_live=True,
    )
    result_path_absence = _correction_result_path_absence_receipt(
        CORRECTION_REPO_ROOT
    )
    failure_classification = (
        "TERMINAL_CUSTODY_NO_GO"
        if phase == "FINALIZER"
        else "INDEPENDENT_CHECKER_NO_GO"
    )
    process_group_members_after_cleanup = (
        _correction_process_group_members(
            spawned_process_group_id
        )
        if spawned_process_group_id is not None else []
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_coordinator_observed_phase_failure.v1"
            ),
            "phase": phase,
            "root_coordinator_process_identity": copy.deepcopy(
                dict(coordinator)
            ),
            "fresh_diagnostic_technical_manifest": manifest,
            "failure_classification": failure_classification,
            "outer_exact_argv": list(outer_argv),
            "inner_evaluator_argv": (
                expected_correction_finalizer_inner_argv()
                if phase == "FINALIZER"
                else expected_correction_checker_inner_argv()
            ),
            "phase_environment_custody": (
                build_correction_phase_environment_custody(
                    phase_environment
                )
            ),
            "parent_requested_cwd": str(CORRECTION_REPO_ROOT),
            "process_observation": (
                copy.deepcopy(dict(process_observation))
                if process_observation is not None else None
            ),
            "spawned_process_group_id": spawned_process_group_id,
            "spawn_requested_monotonic_ns": spawn_requested_monotonic_ns,
            "spawned_monotonic_ns": spawned_monotonic_ns,
            "popen_returned_monotonic_ns": popen_returned_monotonic_ns,
            "ended_monotonic_ns": ended_monotonic_ns,
            "streams": {
                "stdout": _correction_phase_stream_binding(stdout_bytes),
                "stderr": _correction_phase_stream_binding(stderr_bytes),
                "traceback": _correction_phase_stream_binding(
                    traceback_payload
                ),
            },
            "structured_exception": structured,
            "returncode": returncode,
            "termination": (
                copy.deepcopy(dict(termination))
                if termination is not None else None
            ),
            "bounded_cleanup_actions": list(cleanup_actions),
            "deadline_exceeded": deadline_expired,
            "process_group_members_after_cleanup": (
                process_group_members_after_cleanup
            ),
            "cleanup_complete": not process_group_members_after_cleanup,
            "normal_phase_exit_custody_built": False,
            "tracked_publication_attempted": False,
            "result_path_absence_receipt": result_path_absence,
            "result_paths_absent": True,
            "phase_receipt": None,
            "phase_success": False,
            "no_retry_authorized": True,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "COMPLETE_COORDINATOR_FAILURE_CUSTODY_NOT_PHASE_SUCCESS"
            ),
            "pass": True,
        }
    )


def _run_one_correction_terminal_phase(
    *,
    repo_root: Path,
    phase: str,
    coordinator: Mapping[str, Any],
    fresh_technical_manifest: Mapping[str, Any],
    prior_phase_exit_custody: Mapping[str, Any] | None,
    not_before_monotonic_ns: int,
) -> dict[str, Any]:
    outer = (
        expected_correction_finalizer_outer_argv()
        if phase == "FINALIZER"
        else expected_correction_checker_outer_argv()
    )
    environment = _correction_terminal_phase_environment(coordinator)
    process: subprocess.Popen[bytes] | None = None
    process_observation: dict[str, Any] | None = None
    spawn_requested_ns: int | None = None
    spawned_ns: int | None = None
    popen_returned_ns: int | None = None
    ended_ns = time.monotonic_ns()
    stdout_bytes = b""
    stderr_bytes = b""
    traceback_bytes = b""
    cleanup_actions: list[str] = []
    timed_out = False
    communicated = False
    try:
        _require_correction_terminal_role_zero_before_read_or_spawn(
            repo_root,
            current_phase_pid=None,
            coordinator_pid=coordinator["pid"],
            gate=f"{phase.lower()} spawn",
        )
        spawn_requested_ns = time.monotonic_ns()
        if spawn_requested_ns <= not_before_monotonic_ns:
            raise ForensicCorrectionContractError(
                "correction phase spawn request ordering drift"
            )
        phase_deadline = (
            time.monotonic() + CORRECTION_PHASE_RUNTIME_TIMEOUT_SECONDS
        )
        spawned_ns = spawn_requested_ns
        process = subprocess.Popen(
            outer,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        popen_returned_ns = time.monotonic_ns()
        identity_deadline = min(
            phase_deadline,
            time.monotonic() + CORRECTION_PHASE_IDENTITY_TIMEOUT_SECONDS,
        )
        while time.monotonic() < identity_deadline:
            if process.poll() is not None:
                break
            try:
                candidate = _observe_correction_phase_process(
                    pid=process.pid, phase=phase
                )
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                OSError,
                ForensicCorrectionContractError,
            ):
                time.sleep(0.01)
                continue
            if candidate["parent_pid"] != coordinator["pid"]:
                raise ForensicCorrectionContractError(
                    "correction phase PPID/coordinator drift"
                )
            process_observation = candidate
            identity_bound_ns = time.monotonic_ns()
            break
        else:
            identity_bound_ns = time.monotonic_ns()
        if process_observation is None:
            raise ForensicCorrectionContractError(
                "correction phase exact identity was never observed"
            )
        try:
            remaining_seconds = phase_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise subprocess.TimeoutExpired(
                    outer, CORRECTION_PHASE_RUNTIME_TIMEOUT_SECONDS
                )
            stdout_bytes, stderr_bytes = process.communicate(
                timeout=remaining_seconds
            )
            communicated = True
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_actions = _terminate_correction_phase_group(process)
            stdout_bytes, stderr_bytes = process.communicate()
            communicated = True
        ended_ns = time.monotonic_ns()
        if process.poll() is None:
            cleanup_actions.extend(_terminate_correction_phase_group(process))
        elif _correction_process_group_members(process.pid):
            cleanup_actions.extend(_terminate_correction_phase_group(process))
        returncode = int(process.returncode)
        termination = _correction_phase_termination(
            returncode=returncode,
            timed_out=timed_out,
            timeout_cleanup_actions=cleanup_actions,
        )
        traceback_bytes = stderr_bytes
        if cleanup_actions:
            raise ForensicCorrectionContractError(
                "correction phase required forced process-group cleanup"
            )
        try:
            envelope = _canonical_phase_wrapper_envelope_from_stdout(
                stdout_bytes,
                phase=phase,
                phase_process_identity=process_observation["process_identity"],
                coordinator_process_identity=coordinator,
                phase_environment=environment,
            )
        except ForensicCorrectionContractError as exc:
            coordinator_traceback_bytes = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ).encode("utf-8", errors="backslashreplace")
            failure = _build_coordinator_observed_phase_failure(
                phase=phase,
                coordinator=coordinator,
                fresh_technical_manifest=fresh_technical_manifest,
                phase_environment=environment,
                outer_argv=outer,
                process_observation=process_observation,
                spawned_process_group_id=process.pid,
                spawn_requested_monotonic_ns=spawn_requested_ns,
                spawned_monotonic_ns=spawned_ns,
                popen_returned_monotonic_ns=popen_returned_ns,
                ended_monotonic_ns=ended_ns,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                traceback_bytes=coordinator_traceback_bytes,
                returncode=returncode,
                termination=termination,
                cleanup_actions=cleanup_actions,
                deadline_expired=timed_out,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return {"phase_exit_custody": None, "raw_failure": failure}
        phase_receipt = envelope["phase_receipt"]
        if phase_receipt is not None:
            if phase == "FINALIZER":
                phase_receipt = validate_correction_finalizer_prospective_receipt(
                    phase_receipt,
                    repo_root=repo_root,
                    require_current_identity=False,
                    repository_mode="PREPUBLICATION_FREEZE_HEAD",
                )
            else:
                phase_receipt = validate_correction_external_checker_receipt(
                    phase_receipt,
                    repo_root=repo_root,
                    require_current_identity=False,
                    repository_mode="PREPUBLICATION_FREEZE_HEAD",
                )
        exit_custody = build_correction_phase_exit_custody(
            repo_root=repo_root,
            phase=phase,
            phase_process_identity=process_observation["process_identity"],
            coordinator_process_identity=coordinator,
            require_coordinator_live=True,
            phase_environment=environment,
            parent_observed_phase_cwd=process_observation["cwd"],
            parent_observed_phase_ppid=process_observation["parent_pid"],
            wrapper_envelope=envelope,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            traceback_bytes=traceback_bytes,
            parent_spawn_requested_monotonic_ns=spawn_requested_ns,
            parent_spawned_monotonic_ns=spawned_ns,
            parent_popen_returned_monotonic_ns=popen_returned_ns,
            parent_identity_bound_monotonic_ns=identity_bound_ns,
            parent_ended_monotonic_ns=ended_ns,
            deadline_expired=timed_out,
            returncode=returncode,
            termination=termination,
            prior_phase_exit_custody=prior_phase_exit_custody,
        )
        return {"phase_exit_custody": exit_custody, "raw_failure": None}
    except BaseException as exc:
        if process is not None:
            try:
                cleanup_actions.extend(
                    _terminate_correction_phase_group(process)
                )
                if not communicated:
                    stdout_bytes, stderr_bytes = process.communicate()
                    communicated = True
            except BaseException as cleanup_exc:
                cleanup_actions.append(
                    f"CLEANUP_ERROR:{type(cleanup_exc).__name__}:{cleanup_exc}"
                )
        ended_ns = time.monotonic_ns()
        returncode = (
            int(process.returncode)
            if process is not None and process.returncode is not None
            else None
        )
        termination = (
            _correction_phase_termination(
                returncode=returncode,
                timed_out=timed_out,
                timeout_cleanup_actions=cleanup_actions,
            )
            if returncode is not None else None
        )
        traceback_bytes = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).encode("utf-8", errors="backslashreplace")
        failure = _build_coordinator_observed_phase_failure(
            phase=phase,
            coordinator=coordinator,
            fresh_technical_manifest=fresh_technical_manifest,
            phase_environment=environment,
            outer_argv=outer,
            process_observation=process_observation,
            spawned_process_group_id=(process.pid if process is not None else None),
            spawn_requested_monotonic_ns=spawn_requested_ns,
            spawned_monotonic_ns=spawned_ns,
            popen_returned_monotonic_ns=popen_returned_ns,
            ended_monotonic_ns=ended_ns,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            traceback_bytes=traceback_bytes,
            returncode=returncode,
            termination=termination,
            cleanup_actions=cleanup_actions,
            deadline_expired=timed_out,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        return {"phase_exit_custody": None, "raw_failure": failure}


def _persist_correction_terminal_failure_custody(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    root = CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT
    parent_fd = BASE._open_absolute_directory_no_follow(root.parent)
    root_fd: int | None = None
    root_created_by_this_call = False
    try:
        os.mkdir(root.name, mode=0o755, dir_fd=parent_fd)
        root_created_by_this_call = True
        os.fsync(parent_fd)
        for open_attempt in range(2):
            try:
                root_fd = os.open(
                    root.name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                break
            except OSError:
                if not root_created_by_this_call or open_attempt:
                    raise
        if root_fd is None:
            raise ForensicCorrectionContractError(
                "terminal failure custody root could not be retained"
            )
        root_info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != 0o755
            or root_info.st_uid != os.getuid()
            or root_info.st_gid != os.getgid()
            or os.listdir(root_fd)
        ):
            raise ForensicCorrectionContractError(
                "terminal failure custody root ownership/emptiness drift"
            )
        payload = _authority_bytes(value)
        leaf = CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH.name
        for write_attempt in range(2):
            fd: int | None = None
            try:
                fd = os.open(
                    leaf,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o644,
                    dir_fd=root_fd,
                )
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
                os.fsync(fd)
                os.close(fd)
                fd = None
                os.fsync(root_fd)
                break
            except BaseException:
                if fd is not None:
                    os.close(fd)
                try:
                    os.unlink(leaf, dir_fd=root_fd)
                    os.fsync(root_fd)
                except FileNotFoundError:
                    pass
                if not root_created_by_this_call or write_attempt:
                    raise
        os.fsync(parent_fd)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
    observed = BASE._read_no_follow_bound_file(
        root,
        CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH.name,
        expected_stat_rows=None,
    )
    if observed != _authority_bytes(value):
        raise ForensicCorrectionContractError(
            "terminal failure custody durable-byte drift"
        )
    return {
        "path": str(CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH),
        "sha256": hashlib.sha256(observed).hexdigest(),
        "bytes": len(observed),
        "created_exclusive": True,
        "file_and_parent_directories_fsynced": True,
        "no_retry_authorized": True,
        "pass": True,
    }


def _build_correction_precommit_staging_failure(
    *,
    coordinator: Mapping[str, Any],
    coordinator_observed_cwd: str,
    fresh_technical_manifest: Mapping[str, Any],
    finalizer_phase: Mapping[str, Any],
    checker_phase: Mapping[str, Any],
    exception: BaseException,
    rollback_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    coordinator_identity = validate_correction_terminal_coordinator_identity(
        coordinator, require_live=True
    )
    manifest = validate_correction_fresh_technical_root_manifest(
        fresh_technical_manifest, reverify_live=True
    )
    rollback = validate_correction_result_path_absence_receipt(
        rollback_receipt,
        repo_root=CORRECTION_REPO_ROOT,
        reverify_live=True,
    )
    if (
        coordinator_identity["pid"] != os.getpid()
        or coordinator_observed_cwd != str(CORRECTION_REPO_ROOT)
        or rollback.get("all_absent_including_dangling_symlinks") is not True
    ):
        raise ForensicCorrectionContractError(
            "precommit staging failure custody drift"
        )
    traceback_text = "".join(
        traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_precommit_staging_failure.v1"
            ),
            "phase": "PRECOMMIT_STAGING",
            "failure_classification": "UNRESOLVED_TECHNICAL_FAILURE",
            "root_coordinator_process_identity": coordinator_identity,
            "coordinator_observed_cwd": coordinator_observed_cwd,
            "fresh_diagnostic_technical_manifest": manifest,
            "finalizer_outer_exact_argv": (
                expected_correction_finalizer_outer_argv()
            ),
            "checker_outer_exact_argv": (
                expected_correction_checker_outer_argv()
            ),
            "finalizer_phase_exit_custody": copy.deepcopy(
                dict(finalizer_phase)
            ),
            "checker_phase_exit_custody": copy.deepcopy(dict(checker_phase)),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback_text,
            "traceback_sha256": hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest(),
            "tracked_publication_attempted": True,
            "staging_semantics": (
                "PRECOMMIT_STAGING_WITH_ROLLBACK_FOR_CATCHABLE_FAILURES"
            ),
            "rollback_required": True,
            "rollback_completed": True,
            "result_path_absence_receipt": copy.deepcopy(
                dict(rollback)
            ),
            "result_paths_absent": True,
            "automatic_retry_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "COMPLETE_PRECOMMIT_STAGING_FAILURE_CUSTODY_NOT_SUCCESS"
            ),
            "pass": True,
        }
    )


def _validate_correction_precommit_staging_failure(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_coordinator: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_finalizer: Mapping[str, Any],
    expected_checker: Mapping[str, Any],
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=reverify_live,
    )
    expected_keys = {
        "schema", "phase", "failure_classification",
        "root_coordinator_process_identity", "coordinator_observed_cwd",
        "fresh_diagnostic_technical_manifest",
        "finalizer_outer_exact_argv", "checker_outer_exact_argv",
        "finalizer_phase_exit_custody", "checker_phase_exit_custody",
        "exception_type", "exception_message", "traceback",
        "traceback_sha256", "tracked_publication_attempted",
        "staging_semantics", "rollback_required", "rollback_completed",
        "result_path_absence_receipt", "result_paths_absent",
        "automatic_retry_authorized",
        "technical_diagnostic_attempt_accounting", "scientific_counters",
        "files_reused", "pass_meaning", "pass", "content_digest",
    }
    manifest = validate_correction_fresh_technical_root_manifest(
        value.get("fresh_diagnostic_technical_manifest"),
        reverify_live=reverify_live,
    )
    absence = validate_correction_result_path_absence_receipt(
        value.get("result_path_absence_receipt"),
        repo_root=repo_root,
        reverify_live=reverify_live,
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    traceback_text = value.get("traceback")
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_precommit_staging_failure.v1"
        )
        or value.get("phase") != "PRECOMMIT_STAGING"
        or value.get("failure_classification")
        != "UNRESOLVED_TECHNICAL_FAILURE"
        or coordinator != expected_coordinator
        or (reverify_live and coordinator["pid"] != os.getpid())
        or value.get("coordinator_observed_cwd")
        != str(CORRECTION_REPO_ROOT)
        or manifest != expected_manifest
        or value.get("finalizer_outer_exact_argv")
        != expected_correction_finalizer_outer_argv()
        or value.get("checker_outer_exact_argv")
        != expected_correction_checker_outer_argv()
        or value.get("finalizer_phase_exit_custody")
        != expected_finalizer
        or value.get("checker_phase_exit_custody") != expected_checker
        or not isinstance(value.get("exception_type"), str)
        or not value["exception_type"]
        or not isinstance(value.get("exception_message"), str)
        or not isinstance(traceback_text, str)
        or not traceback_text
        or hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
        != value.get("traceback_sha256")
        or value.get("tracked_publication_attempted") is not True
        or value.get("staging_semantics")
        != "PRECOMMIT_STAGING_WITH_ROLLBACK_FOR_CATCHABLE_FAILURES"
        or value.get("rollback_required") is not True
        or value.get("rollback_completed") is not True
        or value.get("result_path_absence_receipt") != absence
        or value.get("result_paths_absent") is not True
        or value.get("automatic_retry_authorized") is not False
        or value.get("files_reused") != 0
        or value.get("pass_meaning")
        != "COMPLETE_PRECOMMIT_STAGING_FAILURE_CUSTODY_NOT_SUCCESS"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "precommit staging failure receipt drift"
        )
    return copy.deepcopy(dict(value))


def validate_correction_precommit_staging_failure(
    value: Mapping[str, Any],
    *,
    reverify_live: bool,
) -> dict[str, Any]:
    if reverify_live:
        correction_result_artifact_payloads(
            repo_root=CORRECTION_REPO_ROOT,
            finalizer_phase_custody=value.get(
                "finalizer_phase_exit_custody"
            ),
            checker_phase_custody=value.get(
                "checker_phase_exit_custody"
            ),
            repository_mode="PREPUBLICATION_FREEZE_HEAD",
        )
    return _validate_correction_precommit_staging_failure(
        value,
        repo_root=CORRECTION_REPO_ROOT,
        expected_coordinator=value.get("root_coordinator_process_identity"),
        expected_manifest=value.get(
            "fresh_diagnostic_technical_manifest"
        ),
        expected_finalizer=value.get("finalizer_phase_exit_custody"),
        expected_checker=value.get("checker_phase_exit_custody"),
        reverify_live=reverify_live,
    )


def _build_correction_terminal_coordinator_receipt(
    *,
    coordinator: Mapping[str, Any],
    require_coordinator_live: bool,
    coordinator_observed_cwd: str,
    launcher_absence_observed_monotonic_ns: int,
    diagnostic_custody: Mapping[str, Any],
    fresh_technical_manifest: Mapping[str, Any],
    finalizer_phase: Mapping[str, Any] | None,
    checker_phase: Mapping[str, Any] | None,
    failure_phase: str | None,
    failure_custody: Mapping[str, Any] | None,
    result_path_absence_receipt: Mapping[str, Any] | None,
    precommit_staging: Mapping[str, Any] | None,
) -> dict[str, Any]:
    coordinator_identity = validate_correction_terminal_coordinator_identity(
        coordinator, require_live=require_coordinator_live
    )
    if (
        (require_coordinator_live and coordinator_identity["pid"] != os.getpid())
        or coordinator_observed_cwd != str(CORRECTION_REPO_ROOT)
    ):
        raise ForensicCorrectionContractError(
            "terminal coordinator cwd drift"
        )
    manifest = validate_correction_fresh_technical_root_manifest(
        fresh_technical_manifest,
        reverify_live=require_coordinator_live,
    )
    sequence_success = (
        finalizer_phase is not None
        and checker_phase is not None
        and failure_phase is None
        and failure_custody is None
        and precommit_staging is not None
    )
    terminal_role_scan = _observe_correction_role_scan_custody(
        CORRECTION_REPO_ROOT,
        gate="TERMINAL_COORDINATOR_RECEIPT",
        excluded_pids={coordinator_identity["pid"]},
    )
    if (
        sequence_success
        and terminal_role_scan["all_nonexcluded_roles_zero"] is not True
    ):
        raise ForensicCorrectionContractError(
            "successful coordinator has a nonzero terminal role scan"
        )
    failure_classification = {
        None: None,
        "FINALIZER": "TERMINAL_CUSTODY_NO_GO",
        "CHECKER": "INDEPENDENT_CHECKER_NO_GO",
        "PRECOMMIT_STAGING": "UNRESOLVED_TECHNICAL_FAILURE",
    }.get(failure_phase)
    if failure_phase not in {
        None, "FINALIZER", "CHECKER", "PRECOMMIT_STAGING"
    }:
        raise ForensicCorrectionContractError(
            "terminal coordinator failure phase drift"
        )
    staging_attempted = (
        precommit_staging is not None or failure_phase == "PRECOMMIT_STAGING"
    )
    rollback_required = failure_phase == "PRECOMMIT_STAGING"
    rollback_completed = bool(
        rollback_required
        and failure_custody is not None
        and failure_custody.get("rollback_completed") is True
    )
    if sequence_success:
        if result_path_absence_receipt is not None:
            raise ForensicCorrectionContractError(
                "successful coordinator has failure path-absence receipt"
            )
        absence = None
    else:
        absence = validate_correction_result_path_absence_receipt(
            result_path_absence_receipt,
            repo_root=CORRECTION_REPO_ROOT,
            reverify_live=require_coordinator_live,
        )
    result_paths_absent_after_failure = absence is not None
    return BASE.attach_self_digest(
        {
            "schema": CORRECTION_TERMINAL_COORDINATOR_SCHEMA,
            "experiment_id": BASE.FORENSIC_EXPERIMENT_ID,
            "root_coordinator_process_identity": copy.deepcopy(
                dict(coordinator_identity)
            ),
            "exact_coordinator_argv": (
                expected_correction_terminal_coordinator_argv()
            ),
            "coordinator_observed_cwd": coordinator_observed_cwd,
            "coordinator_outside_evaluator_wrapper_producer_scan": True,
            "diagnostic_custody_binding": _tracked_correction_json_binding(
                TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH,
                diagnostic_custody,
            ),
            "fresh_diagnostic_technical_manifest": manifest,
            "terminal_role_scan_custody": terminal_role_scan,
            "diagnostic_launcher_pid": diagnostic_custody[
                "launcher_process_identity"
            ]["pid"],
            "diagnostic_launcher_absent_before_finalizer_spawn": True,
            "launcher_absence_observed_monotonic_ns": (
                launcher_absence_observed_monotonic_ns
            ),
            "phase_order": ["FINALIZER", "CHECKER"],
            "finalizer_phase_exit_custody": (
                copy.deepcopy(dict(finalizer_phase))
                if finalizer_phase is not None else None
            ),
            "checker_phase_exit_custody": (
                copy.deepcopy(dict(checker_phase))
                if checker_phase is not None else None
            ),
            "failure_phase": failure_phase,
            "failure_classification": failure_classification,
            "failure_custody": (
                copy.deepcopy(dict(failure_custody))
                if failure_custody is not None else None
            ),
            "precommit_staging_receipt": (
                copy.deepcopy(dict(precommit_staging))
                if precommit_staging is not None else None
            ),
            "precommit_staging_attempted": staging_attempted,
            "precommit_staging_and_receipt_validation_completed": (
                sequence_success
            ),
            "coordinator_stdout_flush_required_to_complete_transaction": (
                sequence_success
            ),
            "precommit_staging_transaction_completed_at_receipt_build": False,
            "precommit_staging_rollback_required": rollback_required,
            "precommit_staging_rollback_completed": rollback_completed,
            "result_path_absence_receipt": absence,
            "result_paths_absent_after_failure": (
                result_paths_absent_after_failure
            ),
            "git_commit_is_the_only_publication_boundary": True,
            "success_literals_emitted": False,
            "automatic_retry_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "technical_diagnostic_consumed_by_launcher_process_identity": (
                copy.deepcopy(
                    diagnostic_custody["launcher_process_identity"]
                )
            ),
            "sequence_success": sequence_success,
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass_meaning": (
                "COMPLETE_COORDINATOR_CUSTODY_INDEPENDENT_OF_SEQUENCE_SUCCESS"
            ),
            "pass": True,
        }
    )


def _validate_correction_coordinator_observed_phase_failure(
    value: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_coordinator: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=reverify_live,
    )
    expected_keys = {
        "schema", "phase", "root_coordinator_process_identity",
        "fresh_diagnostic_technical_manifest", "failure_classification",
        "outer_exact_argv", "inner_evaluator_argv",
        "phase_environment_custody", "parent_requested_cwd",
        "process_observation", "spawned_process_group_id",
        "spawn_requested_monotonic_ns",
        "spawned_monotonic_ns", "popen_returned_monotonic_ns",
        "ended_monotonic_ns", "streams", "structured_exception",
        "returncode", "termination", "bounded_cleanup_actions",
        "deadline_exceeded", "process_group_members_after_cleanup",
        "cleanup_complete", "normal_phase_exit_custody_built",
        "tracked_publication_attempted", "result_path_absence_receipt",
        "result_paths_absent", "phase_receipt", "phase_success",
        "no_retry_authorized", "technical_diagnostic_attempt_accounting",
        "scientific_counters", "files_reused", "pass_meaning", "pass",
        "content_digest",
    }
    if expected_phase not in {"FINALIZER", "CHECKER"}:
        raise ForensicCorrectionContractError(
            "coordinator-observed failure phase drift"
        )
    streams = value.get("streams")
    if type(streams) is not dict or set(streams) != {
        "stdout", "stderr", "traceback"
    }:
        raise ForensicCorrectionContractError(
            "coordinator-observed failure stream set drift"
        )
    traceback_payload = _correction_phase_stream_bytes(
        streams["traceback"]
    )
    _correction_phase_stream_bytes(streams["stdout"])
    _correction_phase_stream_bytes(streams["stderr"])
    structured = validate_correction_phase_structured_exception(
        value.get("structured_exception"),
        phase=expected_phase,
        expected_traceback_bytes=traceback_payload,
    )
    environment = validate_correction_phase_environment_custody(
        value.get("phase_environment_custody"),
        expected_environment=None,
    )
    observation = value.get("process_observation")
    spawned_process_group_id = value.get("spawned_process_group_id")
    if (
        spawned_process_group_id is not None
        and (
            isinstance(spawned_process_group_id, bool)
            or not isinstance(spawned_process_group_id, int)
            or spawned_process_group_id <= 0
        )
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed spawned process-group drift"
        )
    if observation is not None:
        if type(observation) is not dict or set(observation) != {
            "process_identity", "parent_pid", "cwd"
        }:
            raise ForensicCorrectionContractError(
                "coordinator-observed process observation drift"
            )
        phase_identity = _validate_correction_phase_identity(
            observation["process_identity"],
            phase=expected_phase,
            require_current=False,
        )
        if (
            observation["parent_pid"] != expected_coordinator["pid"]
            or observation["cwd"] != str(CORRECTION_REPO_ROOT)
            or phase_identity["process_group_id"] != phase_identity["pid"]
            or spawned_process_group_id != phase_identity["process_group_id"]
        ):
            raise ForensicCorrectionContractError(
                "coordinator-observed process custody drift"
            )
    termination = value.get("termination")
    returncode = value.get("returncode")
    if (
        returncode is not None
        and (isinstance(returncode, bool) or not isinstance(returncode, int))
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed failure returncode drift"
        )
    timestamp_values = [
        value.get("spawn_requested_monotonic_ns"),
        value.get("spawned_monotonic_ns"),
        value.get("popen_returned_monotonic_ns"),
    ]
    populated = [item for item in timestamp_values if item is not None]
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in populated)
        or populated != sorted(populated)
        or (
            timestamp_values[0] is None
            and any(item is not None for item in timestamp_values[1:])
        )
        or (
            timestamp_values[0] is not None
            and timestamp_values[1] != timestamp_values[0]
        )
        or (
            observation is not None
            and any(item is None for item in timestamp_values)
        )
        or isinstance(value.get("ended_monotonic_ns"), bool)
        or not isinstance(value.get("ended_monotonic_ns"), int)
        or value["ended_monotonic_ns"] <= 0
        or (populated and populated[-1] > value["ended_monotonic_ns"])
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed failure timing drift"
        )
    expected_classification = (
        "TERMINAL_CUSTODY_NO_GO"
        if expected_phase == "FINALIZER"
        else "INDEPENDENT_CHECKER_NO_GO"
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    manifest = validate_correction_fresh_technical_root_manifest(
        value.get("fresh_diagnostic_technical_manifest"),
        reverify_live=False,
    )
    absence = value.get("result_path_absence_receipt")
    validated_absence = validate_correction_result_path_absence_receipt(
        absence,
        repo_root=CORRECTION_REPO_ROOT,
        reverify_live=False,
    )
    expected_outer = (
        expected_correction_finalizer_outer_argv()
        if expected_phase == "FINALIZER"
        else expected_correction_checker_outer_argv()
    )
    expected_inner = (
        expected_correction_finalizer_inner_argv()
        if expected_phase == "FINALIZER"
        else expected_correction_checker_inner_argv()
    )
    actions = value.get("bounded_cleanup_actions")
    if (
        type(actions) is not list
        or any(
            not isinstance(action, str)
            or (
                action not in {
                    "SIGTERM", "SIGKILL", "SIGKILL_REMAINING_GROUP"
                }
                and not action.startswith("CLEANUP_ERROR:")
            )
            for action in actions
        )
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed cleanup action drift"
        )
    expected_termination = (
        _correction_phase_termination(
            returncode=returncode,
            timed_out=value.get("deadline_exceeded"),
            timeout_cleanup_actions=actions,
        )
        if returncode is not None else None
    )
    members = value.get("process_group_members_after_cleanup")
    if (
        reverify_live
        and spawned_process_group_id is not None
        and _correction_process_group_members(spawned_process_group_id)
        != members
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed cleanup replay drift"
        )
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_coordinator_observed_phase_failure.v1"
        )
        or value.get("phase") != expected_phase
        or coordinator != expected_coordinator
        or (reverify_live and coordinator["pid"] != os.getpid())
        or manifest != expected_manifest
        or value.get("failure_classification") != expected_classification
        or value.get("outer_exact_argv") != expected_outer
        or value.get("inner_evaluator_argv") != expected_inner
        or environment != value.get("phase_environment_custody")
        or value.get("parent_requested_cwd") != str(CORRECTION_REPO_ROOT)
        or value.get("structured_exception") != structured
        or not isinstance(value.get("deadline_exceeded"), bool)
        or value.get("termination") != expected_termination
        or (observation is not None and spawned_process_group_id is None)
        or type(members) is not list
        or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            for pid in members
        )
        or members != sorted(set(members))
        or value.get("cleanup_complete") is not (not members)
        or value.get("normal_phase_exit_custody_built") is not False
        or value.get("tracked_publication_attempted") is not False
        or validated_absence != absence
        or value.get("result_paths_absent") is not True
        or value.get("phase_receipt") is not None
        or value.get("phase_success") is not False
        or value.get("no_retry_authorized") is not True
        or value.get("files_reused") != 0
        or value.get("pass_meaning")
        != "COMPLETE_COORDINATOR_FAILURE_CUSTODY_NOT_PHASE_SUCCESS"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "coordinator-observed phase failure custody drift"
        )
    return copy.deepcopy(dict(value))


def _validate_correction_precommit_staging_receipt(
    value: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "source_freeze_commit", "result", "tracked_paths",
        "tracked_bindings", "postwrite_dirty_paths",
        "postwrite_other_role_matches",
        "postwrite_other_coordinator_matches", "staging_semantics",
        "publication_boundary", "required_commit_subject",
        "success_literals_emitted", "v2_execution_authorized", "pass",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("source_freeze_commit")
        != artifacts["diagnostic_custody"]["source_commit"]
        or value.get("result") != artifacts["result"]
        or value.get("tracked_paths") != sorted(CORRECTION_RESULT_PATHS)
        or value.get("tracked_bindings") != artifacts["tracked_bindings"]
        or value.get("postwrite_dirty_paths")
        != sorted(CORRECTION_RESULT_PATHS)
        or value.get("postwrite_other_role_matches") != []
        or value.get("postwrite_other_coordinator_matches") != []
        or value.get("staging_semantics")
        != "PRECOMMIT_STAGING_WITH_ROLLBACK_FOR_CATCHABLE_FAILURES"
        or value.get("publication_boundary")
        != "EXACT_GIT_RESULT_COMMIT_AND_POSTCOMMIT_REPLAY_ONLY"
        or value.get("required_commit_subject")
        != CORRECTION_RESULT_COMMIT_SUBJECT
        or value.get("success_literals_emitted") is not False
        or value.get("v2_execution_authorized") is not False
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "correction precommit staging receipt drift"
        )
    return copy.deepcopy(dict(value))


def validate_correction_terminal_coordinator_receipt(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    require_coordinator_live: bool,
) -> dict[str, Any]:
    """Replay exact phase order, failure exclusivity, and staging custody."""

    BASE.validate_self_digest(value)
    root = Path(repo_root).resolve()
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=require_coordinator_live,
    )
    if require_coordinator_live and coordinator["pid"] != os.getpid():
        raise ForensicCorrectionContractError(
            "coordinator receipt is not owned by current coordinator"
        )
    manifest = validate_correction_fresh_technical_root_manifest(
        value.get("fresh_diagnostic_technical_manifest"),
        reverify_live=require_coordinator_live,
    )
    sequence_success = value.get("sequence_success") is True
    repository_mode = (
        "PRECOMMIT_STAGED_RESULT_PATHS"
        if sequence_success
        else "PREPUBLICATION_FREEZE_HEAD"
    )
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode=repository_mode,
    )
    custody = bundle["diagnostic_custody"]
    if (
        value.get("diagnostic_custody_binding")
        != _tracked_correction_json_binding(
            TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH, custody
        )
        or value.get("diagnostic_launcher_pid")
        != custody["launcher_process_identity"]["pid"]
        or value.get("diagnostic_launcher_absent_before_finalizer_spawn")
        is not True
        or isinstance(
            value.get("launcher_absence_observed_monotonic_ns"), bool
        )
        or not isinstance(
            value.get("launcher_absence_observed_monotonic_ns"), int
        )
        or value["launcher_absence_observed_monotonic_ns"]
        <= custody["ended_monotonic_ns"]
    ):
        raise ForensicCorrectionContractError(
            "coordinator diagnostic/launcher custody drift"
        )
    finalizer = value.get("finalizer_phase_exit_custody")
    checker = value.get("checker_phase_exit_custody")
    failure_phase = value.get("failure_phase")
    failure = value.get("failure_custody")
    staging = value.get("precommit_staging_receipt")
    absence = value.get("result_path_absence_receipt")
    if sequence_success:
        artifacts = correction_result_artifact_payloads(
            repo_root=root,
            finalizer_phase_custody=finalizer,
            checker_phase_custody=checker,
            repository_mode=repository_mode,
        )
        pre_read_dirty_paths = _correction_dirty_paths(root)
        for relative, expected_payload in artifacts["tracked_payloads"].items():
            observed_payload = BASE._read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
            if observed_payload != expected_payload:
                raise ForensicCorrectionContractError(
                    f"staged correction result byte drift: {relative}"
                )
        post_read_dirty_paths = _correction_dirty_paths(root)
        if (
            pre_read_dirty_paths != sorted(CORRECTION_RESULT_PATHS)
            or post_read_dirty_paths != pre_read_dirty_paths
        ):
            raise ForensicCorrectionContractError(
                "staged correction result path set changed during replay"
            )
        _validate_correction_precommit_staging_receipt(
            staging, artifacts=artifacts
        )
        if failure_phase is not None or failure is not None or absence is not None:
            raise ForensicCorrectionContractError(
                "successful coordinator carries failure custody"
            )
    else:
        if staging is not None or failure is None or absence is None:
            raise ForensicCorrectionContractError(
                "failed coordinator success/failure exclusivity drift"
            )
        validate_correction_result_path_absence_receipt(
            absence,
            repo_root=root,
            reverify_live=require_coordinator_live,
        )
        if failure_phase == "FINALIZER":
            if finalizer is not None or checker is not None:
                raise ForensicCorrectionContractError(
                    "finalizer failure has later phase custody"
                )
            if failure.get("schema") == CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA:
                validate_correction_phase_exit_custody(
                    failure,
                    repo_root=root,
                    expected_phase="FINALIZER",
                    expected_phase_receipt=None,
                    prior_phase_exit_custody=None,
                    require_coordinator_live=False,
                )
            else:
                _validate_correction_coordinator_observed_phase_failure(
                    failure,
                    expected_phase="FINALIZER",
                    expected_coordinator=coordinator,
                    expected_manifest=manifest,
                    reverify_live=require_coordinator_live,
                )
        elif failure_phase == "CHECKER":
            if finalizer is None or checker is not None:
                raise ForensicCorrectionContractError(
                    "checker failure phase dependency drift"
                )
            finalizer_receipt = validate_correction_finalizer_prospective_receipt(
                finalizer.get("phase_receipt"),
                repo_root=root,
                require_current_identity=False,
                repository_mode=repository_mode,
            )
            validate_correction_phase_exit_custody(
                finalizer,
                repo_root=root,
                expected_phase="FINALIZER",
                expected_phase_receipt=finalizer_receipt,
                prior_phase_exit_custody=None,
                require_coordinator_live=False,
            )
            if failure.get("schema") == CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA:
                validate_correction_phase_exit_custody(
                    failure,
                    repo_root=root,
                    expected_phase="CHECKER",
                    expected_phase_receipt=None,
                    prior_phase_exit_custody=finalizer,
                    require_coordinator_live=False,
                )
            else:
                _validate_correction_coordinator_observed_phase_failure(
                    failure,
                    expected_phase="CHECKER",
                    expected_coordinator=coordinator,
                    expected_manifest=manifest,
                    reverify_live=require_coordinator_live,
                )
        elif failure_phase == "PRECOMMIT_STAGING":
            if finalizer is None or checker is None:
                raise ForensicCorrectionContractError(
                    "precommit failure lacks completed phase custody"
                )
            correction_result_artifact_payloads(
                repo_root=root,
                finalizer_phase_custody=finalizer,
                checker_phase_custody=checker,
                repository_mode=repository_mode,
            )
            _validate_correction_precommit_staging_failure(
                failure,
                repo_root=root,
                expected_coordinator=coordinator,
                expected_manifest=manifest,
                expected_finalizer=finalizer,
                expected_checker=checker,
                reverify_live=require_coordinator_live,
            )
        else:
            raise ForensicCorrectionContractError(
                "coordinator failure phase drift"
            )
    rebuilt = _build_correction_terminal_coordinator_receipt(
        coordinator=coordinator,
        require_coordinator_live=require_coordinator_live,
        coordinator_observed_cwd=value.get("coordinator_observed_cwd"),
        launcher_absence_observed_monotonic_ns=value.get(
            "launcher_absence_observed_monotonic_ns"
        ),
        diagnostic_custody=custody,
        fresh_technical_manifest=manifest,
        finalizer_phase=finalizer,
        checker_phase=checker,
        failure_phase=failure_phase,
        failure_custody=failure,
        result_path_absence_receipt=absence,
        precommit_staging=staging,
    )
    if dict(value) != rebuilt:
        raise ForensicCorrectionContractError(
            "correction terminal coordinator receipt drift"
        )
    return copy.deepcopy(dict(value))


def run_correction_terminal_phase_sequence(
    repo_root: str | Path = CORRECTION_REPO_ROOT,
) -> dict[str, Any]:
    """Run finalizer then checker, stage nine bytes, and never run science."""

    root = Path(repo_root).resolve()
    if root != CORRECTION_REPO_ROOT:
        raise ForensicCorrectionContractError(
            "terminal coordinator repository root drift"
        )
    coordinator = _observe_correction_terminal_coordinator_identity(
        os.getpid(), require_exact=True
    )
    coordinator_cwd = _observe_correction_process_cwd(coordinator)
    other_coordinators = _active_correction_terminal_coordinators(
        exclude_pids={coordinator["pid"]}
    )
    if other_coordinators:
        raise ForensicCorrectionContractError(
            "another or malformed terminal coordinator is live"
        )
    _require_correction_terminal_role_zero_before_read_or_spawn(
        root,
        current_phase_pid=None,
        coordinator_pid=coordinator["pid"],
        gate="coordinator terminal manifest and bundle read",
    )
    fresh_manifest = build_correction_fresh_technical_root_manifest()
    bundle = load_and_validate_correction_diagnostic_bundle(
        repo_root=root,
        repository_mode="PREPUBLICATION_FREEZE_HEAD",
    )
    custody = bundle["diagnostic_custody"]
    launcher_pid = int(custody["launcher_process_identity"]["pid"])
    if Path(f"/proc/{launcher_pid}").exists():
        raise ForensicCorrectionContractError(
            "diagnostic launcher remains live before finalizer"
        )
    remaining_roles = BASE._active_forensic_or_scientific_processes(
        root, exclude_pids={coordinator["pid"]}
    )
    remaining_coordinators = _active_correction_terminal_coordinators(
        exclude_pids={coordinator["pid"]}
    )
    if remaining_roles or remaining_coordinators:
        raise ForensicCorrectionContractError(
            "producer/coordinator remains live before finalizer"
        )
    launcher_absence_ns = time.monotonic_ns()
    if launcher_absence_ns <= int(custody["ended_monotonic_ns"]):
        raise ForensicCorrectionContractError(
            "launcher-exit/finalizer monotonic ordering drift"
        )
    finalizer_run = _run_one_correction_terminal_phase(
        repo_root=root,
        phase="FINALIZER",
        coordinator=coordinator,
        fresh_technical_manifest=fresh_manifest,
        prior_phase_exit_custody=None,
        not_before_monotonic_ns=launcher_absence_ns,
    )
    finalizer = finalizer_run["phase_exit_custody"]
    if finalizer is None or finalizer.get("phase_success") is not True:
        failure = finalizer_run["raw_failure"] or finalizer
        absence = _correction_result_path_absence_receipt(root)
        receipt = _build_correction_terminal_coordinator_receipt(
            coordinator=coordinator,
            require_coordinator_live=True,
            coordinator_observed_cwd=coordinator_cwd,
            launcher_absence_observed_monotonic_ns=launcher_absence_ns,
            diagnostic_custody=custody,
            fresh_technical_manifest=fresh_manifest,
            finalizer_phase=None,
            checker_phase=None,
            failure_phase="FINALIZER",
            failure_custody=failure,
            result_path_absence_receipt=absence,
            precommit_staging=None,
        )
        return validate_correction_terminal_coordinator_receipt(
            receipt, repo_root=root, require_coordinator_live=True
        )
    checker_run = _run_one_correction_terminal_phase(
        repo_root=root,
        phase="CHECKER",
        coordinator=coordinator,
        fresh_technical_manifest=fresh_manifest,
        prior_phase_exit_custody=finalizer,
        not_before_monotonic_ns=finalizer["parent_ended_monotonic_ns"],
    )
    checker = checker_run["phase_exit_custody"]
    if checker is None or checker.get("phase_success") is not True:
        failure = checker_run["raw_failure"] or checker
        absence = _correction_result_path_absence_receipt(root)
        receipt = _build_correction_terminal_coordinator_receipt(
            coordinator=coordinator,
            require_coordinator_live=True,
            coordinator_observed_cwd=coordinator_cwd,
            launcher_absence_observed_monotonic_ns=launcher_absence_ns,
            diagnostic_custody=custody,
            fresh_technical_manifest=fresh_manifest,
            finalizer_phase=finalizer,
            checker_phase=None,
            failure_phase="CHECKER",
            failure_custody=failure,
            result_path_absence_receipt=absence,
            precommit_staging=None,
        )
        return validate_correction_terminal_coordinator_receipt(
            receipt, repo_root=root, require_coordinator_live=True
        )
    try:
        staging = write_correction_result_artifacts(
            repo_root=root,
            coordinator_process_identity=coordinator,
            finalizer_phase_custody=finalizer,
            checker_phase_custody=checker,
        )
        receipt = _build_correction_terminal_coordinator_receipt(
            coordinator=coordinator,
            require_coordinator_live=True,
            coordinator_observed_cwd=coordinator_cwd,
            launcher_absence_observed_monotonic_ns=launcher_absence_ns,
            diagnostic_custody=custody,
            fresh_technical_manifest=fresh_manifest,
            finalizer_phase=finalizer,
            checker_phase=checker,
            failure_phase=None,
            failure_custody=None,
            result_path_absence_receipt=None,
            precommit_staging=staging,
        )
        return validate_correction_terminal_coordinator_receipt(
            receipt, repo_root=root, require_coordinator_live=True
        )
    except BaseException as exc:
        try:
            rollback = rollback_correction_result_artifacts(root)
            failure = _build_correction_precommit_staging_failure(
                coordinator=coordinator,
                coordinator_observed_cwd=coordinator_cwd,
                fresh_technical_manifest=fresh_manifest,
                finalizer_phase=finalizer,
                checker_phase=checker,
                exception=exc,
                rollback_receipt=rollback,
            )
            receipt = _build_correction_terminal_coordinator_receipt(
                coordinator=coordinator,
                require_coordinator_live=True,
                coordinator_observed_cwd=coordinator_cwd,
                launcher_absence_observed_monotonic_ns=launcher_absence_ns,
                diagnostic_custody=custody,
                fresh_technical_manifest=fresh_manifest,
                finalizer_phase=finalizer,
                checker_phase=checker,
                failure_phase="PRECOMMIT_STAGING",
                failure_custody=failure,
                result_path_absence_receipt=rollback,
                precommit_staging=None,
            )
            return validate_correction_terminal_coordinator_receipt(
                receipt, repo_root=root, require_coordinator_live=True
            )
        except BaseException as custody_exc:
            raise CorrectionPrecommitStagingTransactionError(
                "precommit staging failure custody could not be completed: "
                f"{type(custody_exc).__name__}: {custody_exc}"
            ) from custody_exc


def _build_correction_coordinator_bootstrap_failure(
    *,
    exception: BaseException,
) -> dict[str, Any]:
    coordinator = _observe_correction_terminal_coordinator_identity(
        os.getpid(), require_exact=True
    )
    coordinator_cwd = _observe_correction_process_cwd(coordinator)
    manifest = build_correction_fresh_technical_root_manifest()
    dirty_result_paths = sorted(
        set(_correction_dirty_paths(CORRECTION_REPO_ROOT))
        & set(CORRECTION_RESULT_PATHS)
    )
    attempted = bool(
        getattr(exception, "tracked_publication_attempted", False)
        or dirty_result_paths
    )
    absence = rollback_correction_result_artifacts(CORRECTION_REPO_ROOT)
    role_scan = _observe_correction_role_scan_custody(
        CORRECTION_REPO_ROOT,
        gate="COORDINATOR_BOOTSTRAP_FAILURE_AFTER_ROLLBACK",
        excluded_pids={coordinator["pid"]},
    )
    traceback_text = "".join(
        traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_coordinator_bootstrap_failure.v1"
            ),
            "phase": (
                "PRECOMMIT_STAGING"
                if attempted else "COORDINATOR_BOOTSTRAP"
            ),
            "failure_classification": "UNRESOLVED_TECHNICAL_FAILURE",
            "root_coordinator_process_identity": coordinator,
            "coordinator_observed_cwd": coordinator_cwd,
            "exact_coordinator_argv": (
                expected_correction_terminal_coordinator_argv()
            ),
            "coordinator_environment_custody": (
                build_correction_phase_environment_custody(
                    _correction_terminal_phase_environment(coordinator)
                )
            ),
            "fresh_diagnostic_technical_manifest": manifest,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback_text,
            "traceback_sha256": hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest(),
            "tracked_publication_attempted": attempted,
            "dirty_result_paths_before_rollback": dirty_result_paths,
            "rollback_required": attempted,
            "rollback_completed": True,
            "result_path_absence_receipt": absence,
            "result_paths_absent": True,
            "failure_role_scan_custody": role_scan,
            "success_literals_emitted": False,
            "automatic_retry_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "sequence_success": False,
            "pass_meaning": "COMPLETE_BOOTSTRAP_FAILURE_CUSTODY",
            "pass": True,
        }
    )


def validate_correction_coordinator_bootstrap_failure(
    value: Mapping[str, Any],
    *,
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "phase", "failure_classification",
        "root_coordinator_process_identity", "coordinator_observed_cwd",
        "exact_coordinator_argv", "coordinator_environment_custody",
        "fresh_diagnostic_technical_manifest", "exception_type",
        "exception_message", "traceback", "traceback_sha256",
        "tracked_publication_attempted", "dirty_result_paths_before_rollback",
        "rollback_required", "rollback_completed",
        "result_path_absence_receipt", "result_paths_absent",
        "failure_role_scan_custody",
        "success_literals_emitted", "automatic_retry_authorized",
        "technical_diagnostic_attempt_accounting", "scientific_counters",
        "files_reused", "sequence_success", "pass_meaning", "pass",
        "content_digest",
    }
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=reverify_live,
    )
    manifest = validate_correction_fresh_technical_root_manifest(
        value.get("fresh_diagnostic_technical_manifest"),
        reverify_live=reverify_live,
    )
    environment = validate_correction_phase_environment_custody(
        value.get("coordinator_environment_custody"),
        expected_environment=None,
    )
    absence = validate_correction_result_path_absence_receipt(
        value.get("result_path_absence_receipt"),
        repo_root=CORRECTION_REPO_ROOT,
        reverify_live=reverify_live,
    )
    role_scan = _validate_correction_role_scan_custody(
        value.get("failure_role_scan_custody"),
        expected_gate="COORDINATOR_BOOTSTRAP_FAILURE_AFTER_ROLLBACK",
        expected_excluded_pids={coordinator["pid"]},
        reverify_live=False,
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    traceback_text = value.get("traceback")
    attempted = value.get("tracked_publication_attempted")
    dirty = value.get("dirty_result_paths_before_rollback")
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_coordinator_bootstrap_failure.v1"
        )
        or value.get("phase")
        != ("PRECOMMIT_STAGING" if attempted else "COORDINATOR_BOOTSTRAP")
        or value.get("failure_classification")
        != "UNRESOLVED_TECHNICAL_FAILURE"
        or (reverify_live and coordinator["pid"] != os.getpid())
        or value.get("coordinator_observed_cwd")
        != str(CORRECTION_REPO_ROOT)
        or value.get("exact_coordinator_argv")
        != expected_correction_terminal_coordinator_argv()
        or environment != value.get("coordinator_environment_custody")
        or manifest != value.get("fresh_diagnostic_technical_manifest")
        or not isinstance(value.get("exception_type"), str)
        or not value["exception_type"]
        or not isinstance(value.get("exception_message"), str)
        or not isinstance(traceback_text, str)
        or not traceback_text
        or hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
        != value.get("traceback_sha256")
        or not isinstance(attempted, bool)
        or type(dirty) is not list
        or dirty != sorted(set(dirty))
        or any(path not in CORRECTION_RESULT_PATHS for path in dirty)
        or value.get("rollback_required") is not attempted
        or value.get("rollback_completed") is not True
        or value.get("result_path_absence_receipt") != absence
        or value.get("result_paths_absent") is not True
        or value.get("failure_role_scan_custody") != role_scan
        or value.get("success_literals_emitted") is not False
        or value.get("automatic_retry_authorized") is not False
        or value.get("files_reused") != 0
        or value.get("sequence_success") is not False
        or value.get("pass_meaning")
        != "COMPLETE_BOOTSTRAP_FAILURE_CUSTODY"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "coordinator bootstrap failure receipt drift"
        )
    return copy.deepcopy(dict(value))


def _build_correction_bootstrap_custody_failure(
    *,
    primary_exception: BaseException,
    custody_exception: BaseException,
    tracked_publication_attempted: bool,
    rollback_receipt: Mapping[str, Any] | None,
    rollback_error: str | None,
) -> dict[str, Any]:
    if not isinstance(tracked_publication_attempted, bool):
        raise ForensicCorrectionContractError(
            "bootstrap-custody publication-attempt type drift"
        )
    validated_rollback: dict[str, Any] | None
    if rollback_receipt is None:
        validated_rollback = None
        if not isinstance(rollback_error, str) or not rollback_error:
            raise ForensicCorrectionContractError(
                "bootstrap-custody rollback failure was not preserved"
            )
    else:
        if rollback_error is not None:
            raise ForensicCorrectionContractError(
                "bootstrap-custody rollback outcome is ambiguous"
            )
        validated_rollback = validate_correction_result_path_absence_receipt(
            rollback_receipt,
            repo_root=CORRECTION_REPO_ROOT,
            reverify_live=True,
        )
    primary_traceback = "".join(
        traceback.format_exception(
            type(primary_exception),
            primary_exception,
            primary_exception.__traceback__,
        )
    )
    custody_traceback = "".join(
        traceback.format_exception(
            type(custody_exception),
            custody_exception,
            custody_exception.__traceback__,
        )
    )
    role_scan = _observe_correction_role_scan_custody(
        CORRECTION_REPO_ROOT,
        gate="BOOTSTRAP_CUSTODY_DOUBLE_FAILURE_AFTER_ROLLBACK",
        excluded_pids={os.getpid()},
    )
    receipt = BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_bootstrap_custody_failure.v1"
            ),
            "failure_classification": "UNRESOLVED_TECHNICAL_FAILURE",
            "exact_coordinator_argv": (
                expected_correction_terminal_coordinator_argv()
            ),
            "primary_exception_type": type(primary_exception).__name__,
            "primary_exception_message": str(primary_exception),
            "primary_traceback": primary_traceback,
            "primary_traceback_sha256": hashlib.sha256(
                primary_traceback.encode("utf-8")
            ).hexdigest(),
            "custody_exception_type": type(custody_exception).__name__,
            "custody_exception_message": str(custody_exception),
            "custody_traceback": custody_traceback,
            "custody_traceback_sha256": hashlib.sha256(
                custody_traceback.encode("utf-8")
            ).hexdigest(),
            "tracked_publication_attempted": (
                tracked_publication_attempted
            ),
            "rollback_required": tracked_publication_attempted,
            "rollback_completed": validated_rollback is not None,
            "rollback_receipt": validated_rollback,
            "rollback_error": rollback_error,
            "result_paths_absent": validated_rollback is not None,
            "failure_role_scan_custody": role_scan,
            "fresh_diagnostic_technical_manifest_unavailable": True,
            "success_literals_emitted": False,
            "automatic_retry_authorized": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "sequence_success": False,
            "pass_meaning": (
                "COMPLETE_FAILURE_OF_BOOTSTRAP_CUSTODY_"
                "NOT_TECHNICAL_SUCCESS"
            ),
            "pass": True,
        }
    )
    return _validate_correction_bootstrap_custody_failure(receipt)


def _validate_correction_bootstrap_custody_failure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "failure_classification", "exact_coordinator_argv",
        "primary_exception_type", "primary_exception_message",
        "primary_traceback", "primary_traceback_sha256",
        "custody_exception_type", "custody_exception_message",
        "custody_traceback", "custody_traceback_sha256",
        "tracked_publication_attempted", "rollback_required",
        "rollback_completed", "rollback_receipt", "rollback_error",
        "result_paths_absent", "failure_role_scan_custody",
        "fresh_diagnostic_technical_manifest_unavailable",
        "success_literals_emitted", "automatic_retry_authorized",
        "technical_diagnostic_attempt_accounting", "scientific_counters",
        "files_reused", "sequence_success", "pass_meaning", "pass",
        "content_digest",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ForensicCorrectionContractError(
            "bootstrap-custody failure key-set drift"
        )
    validate_correction_technical_diagnostic_attempt_accounting(
        value["technical_diagnostic_attempt_accounting"]
    )
    validate_correction_zero_scientific_counters(
        value["scientific_counters"]
    )
    attempted = value["tracked_publication_attempted"]
    rollback_receipt = value["rollback_receipt"]
    rollback_error = value["rollback_error"]
    if rollback_receipt is not None:
        validated_rollback = validate_correction_result_path_absence_receipt(
            rollback_receipt,
            repo_root=CORRECTION_REPO_ROOT,
            reverify_live=False,
        )
        rollback_succeeded = True
    else:
        validated_rollback = None
        rollback_succeeded = False
    primary_traceback = value["primary_traceback"]
    custody_traceback = value["custody_traceback"]
    role_scan = _validate_correction_role_scan_custody(
        value["failure_role_scan_custody"],
        expected_gate="BOOTSTRAP_CUSTODY_DOUBLE_FAILURE_AFTER_ROLLBACK",
        expected_excluded_pids={os.getpid()},
        reverify_live=False,
    )
    if (
        value["schema"]
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_bootstrap_custody_failure.v1"
        )
        or value["failure_classification"]
        != "UNRESOLVED_TECHNICAL_FAILURE"
        or value["exact_coordinator_argv"]
        != expected_correction_terminal_coordinator_argv()
        or not isinstance(value["primary_exception_type"], str)
        or not value["primary_exception_type"]
        or not isinstance(value["primary_exception_message"], str)
        or not isinstance(primary_traceback, str)
        or not primary_traceback
        or hashlib.sha256(primary_traceback.encode("utf-8")).hexdigest()
        != value["primary_traceback_sha256"]
        or not isinstance(value["custody_exception_type"], str)
        or not value["custody_exception_type"]
        or not isinstance(value["custody_exception_message"], str)
        or not isinstance(custody_traceback, str)
        or not custody_traceback
        or hashlib.sha256(custody_traceback.encode("utf-8")).hexdigest()
        != value["custody_traceback_sha256"]
        or not isinstance(attempted, bool)
        or value["rollback_required"] is not attempted
        or value["rollback_completed"] is not rollback_succeeded
        or value["rollback_receipt"] != validated_rollback
        or (
            rollback_succeeded
            and rollback_error is not None
        )
        or (
            not rollback_succeeded
            and (
                not isinstance(rollback_error, str)
                or not rollback_error
            )
        )
        or value["result_paths_absent"] is not rollback_succeeded
        or value["failure_role_scan_custody"] != role_scan
        or value["fresh_diagnostic_technical_manifest_unavailable"]
        is not True
        or value["success_literals_emitted"] is not False
        or value["automatic_retry_authorized"] is not False
        or value["files_reused"] != 0
        or value["sequence_success"] is not False
        or value["pass_meaning"]
        != (
            "COMPLETE_FAILURE_OF_BOOTSTRAP_CUSTODY_"
            "NOT_TECHNICAL_SUCCESS"
        )
        or value["pass"] is not True
    ):
        raise ForensicCorrectionContractError(
            "bootstrap-custody failure receipt drift"
        )
    return copy.deepcopy(dict(value))


def _validate_correction_failed_coordinator_receipt_for_persistence(
    value: Mapping[str, Any],
    *,
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "experiment_id", "root_coordinator_process_identity",
        "exact_coordinator_argv", "coordinator_observed_cwd",
        "coordinator_outside_evaluator_wrapper_producer_scan",
        "diagnostic_custody_binding", "fresh_diagnostic_technical_manifest",
        "terminal_role_scan_custody",
        "diagnostic_launcher_pid",
        "diagnostic_launcher_absent_before_finalizer_spawn",
        "launcher_absence_observed_monotonic_ns", "phase_order",
        "finalizer_phase_exit_custody", "checker_phase_exit_custody",
        "failure_phase", "failure_classification", "failure_custody",
        "precommit_staging_receipt", "precommit_staging_attempted",
        "precommit_staging_and_receipt_validation_completed",
        "coordinator_stdout_flush_required_to_complete_transaction",
        "precommit_staging_transaction_completed_at_receipt_build",
        "precommit_staging_rollback_required",
        "precommit_staging_rollback_completed",
        "result_path_absence_receipt",
        "result_paths_absent_after_failure",
        "git_commit_is_the_only_publication_boundary",
        "success_literals_emitted", "automatic_retry_authorized",
        "technical_diagnostic_attempt_accounting",
        "technical_diagnostic_consumed_by_launcher_process_identity",
        "sequence_success", "scientific_counters", "files_reused",
        "pass_meaning", "pass", "content_digest",
    }
    coordinator = validate_correction_terminal_coordinator_identity(
        value.get("root_coordinator_process_identity"),
        require_live=reverify_live,
    )
    launcher = BASE.validate_process_identity(
        value.get(
            "technical_diagnostic_consumed_by_launcher_process_identity"
        )
    )
    manifest = validate_correction_fresh_technical_root_manifest(
        value.get("fresh_diagnostic_technical_manifest"),
        reverify_live=reverify_live,
    )
    terminal_role_scan = _validate_correction_role_scan_custody(
        value.get("terminal_role_scan_custody"),
        expected_gate="TERMINAL_COORDINATOR_RECEIPT",
        expected_excluded_pids={coordinator["pid"]},
        reverify_live=False,
    )
    absence = validate_correction_result_path_absence_receipt(
        value.get("result_path_absence_receipt"),
        repo_root=CORRECTION_REPO_ROOT,
        reverify_live=reverify_live,
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    binding = BASE.validate_artifact_binding(
        value.get("diagnostic_custody_binding"),
        content_digest_required=True,
    )
    failure_phase = value.get("failure_phase")
    expected_classification = {
        "FINALIZER": "TERMINAL_CUSTODY_NO_GO",
        "CHECKER": "INDEPENDENT_CHECKER_NO_GO",
        "PRECOMMIT_STAGING": "UNRESOLVED_TECHNICAL_FAILURE",
    }.get(failure_phase)
    finalizer = value.get("finalizer_phase_exit_custody")
    checker = value.get("checker_phase_exit_custody")
    failure = value.get("failure_custody")
    if failure_phase == "FINALIZER":
        if finalizer is not None or checker is not None:
            raise ForensicCorrectionContractError(
                "persisted finalizer failure phase union drift"
            )
        if failure.get("schema") == CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA:
            validate_correction_phase_exit_custody(
                failure,
                repo_root=CORRECTION_REPO_ROOT,
                expected_phase="FINALIZER",
                expected_phase_receipt=None,
                prior_phase_exit_custody=None,
                require_coordinator_live=reverify_live,
            )
        else:
            _validate_correction_coordinator_observed_phase_failure(
                failure,
                expected_phase="FINALIZER",
                expected_coordinator=coordinator,
                expected_manifest=manifest,
                reverify_live=reverify_live,
            )
    elif failure_phase == "CHECKER":
        if finalizer is None or checker is not None:
            raise ForensicCorrectionContractError(
                "persisted checker failure phase union drift"
            )
        BASE.validate_self_digest(finalizer.get("phase_receipt"))
        validate_correction_phase_exit_custody(
            finalizer,
            repo_root=CORRECTION_REPO_ROOT,
            expected_phase="FINALIZER",
            expected_phase_receipt=finalizer["phase_receipt"],
            prior_phase_exit_custody=None,
            require_coordinator_live=reverify_live,
        )
        if failure.get("schema") == CORRECTION_PHASE_EXIT_CUSTODY_SCHEMA:
            validate_correction_phase_exit_custody(
                failure,
                repo_root=CORRECTION_REPO_ROOT,
                expected_phase="CHECKER",
                expected_phase_receipt=None,
                prior_phase_exit_custody=finalizer,
                require_coordinator_live=reverify_live,
            )
        else:
            _validate_correction_coordinator_observed_phase_failure(
                failure,
                expected_phase="CHECKER",
                expected_coordinator=coordinator,
                expected_manifest=manifest,
                reverify_live=reverify_live,
            )
    elif failure_phase == "PRECOMMIT_STAGING":
        if finalizer is None or checker is None:
            raise ForensicCorrectionContractError(
                "persisted precommit failure phase union drift"
            )
        validate_correction_precommit_staging_failure(
            failure, reverify_live=reverify_live
        )
    else:
        raise ForensicCorrectionContractError(
            "persisted coordinator failure phase drift"
        )
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema") != CORRECTION_TERMINAL_COORDINATOR_SCHEMA
        or value.get("experiment_id") != BASE.FORENSIC_EXPERIMENT_ID
        or (reverify_live and coordinator["pid"] != os.getpid())
        or value.get("exact_coordinator_argv")
        != expected_correction_terminal_coordinator_argv()
        or value.get("coordinator_observed_cwd")
        != str(CORRECTION_REPO_ROOT)
        or value.get("coordinator_outside_evaluator_wrapper_producer_scan")
        is not True
        or binding["path"]
        != str(TRACKED_CORRECTION_DIAGNOSTIC_CUSTODY_PATH)
        or value.get("terminal_role_scan_custody") != terminal_role_scan
        or value.get("diagnostic_launcher_pid") != launcher["pid"]
        or value.get("diagnostic_launcher_absent_before_finalizer_spawn")
        is not True
        or value.get("phase_order") != ["FINALIZER", "CHECKER"]
        or expected_classification is None
        or value.get("failure_classification") != expected_classification
        or failure is None
        or value.get("precommit_staging_receipt") is not None
        or value.get("precommit_staging_attempted")
        is not (failure_phase == "PRECOMMIT_STAGING")
        or value.get("precommit_staging_and_receipt_validation_completed")
        is not False
        or value.get(
            "coordinator_stdout_flush_required_to_complete_transaction"
        )
        is not False
        or value.get(
            "precommit_staging_transaction_completed_at_receipt_build"
        )
        is not False
        or value.get("precommit_staging_rollback_required")
        is not (failure_phase == "PRECOMMIT_STAGING")
        or value.get("precommit_staging_rollback_completed")
        is not (failure_phase == "PRECOMMIT_STAGING")
        or value.get("result_path_absence_receipt") != absence
        or value.get("result_paths_absent_after_failure") is not True
        or value.get("git_commit_is_the_only_publication_boundary")
        is not True
        or value.get("success_literals_emitted") is not False
        or value.get("automatic_retry_authorized") is not False
        or value.get("sequence_success") is not False
        or value.get("files_reused") != 0
        or value.get("pass_meaning")
        != "COMPLETE_COORDINATOR_CUSTODY_INDEPENDENT_OF_SEQUENCE_SUCCESS"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "failed coordinator receipt persistence union drift"
        )
    return copy.deepcopy(dict(value))


def _validate_correction_failure_receipt_for_persistence(
    value: Mapping[str, Any],
    *,
    reverify_live: bool,
) -> dict[str, Any]:
    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == CORRECTION_TERMINAL_COORDINATOR_SCHEMA:
        if reverify_live:
            validated = validate_correction_terminal_coordinator_receipt(
                value,
                repo_root=CORRECTION_REPO_ROOT,
                require_coordinator_live=True,
            )
            if validated.get("sequence_success") is not False:
                raise ForensicCorrectionContractError(
                    "successful coordinator receipt cannot be persisted as failure"
                )
            return validated
        return _validate_correction_failed_coordinator_receipt_for_persistence(
            value, reverify_live=reverify_live
        )
    if schema == (
        "plan_aware_monotone_jepa_cost_v1."
        "forensic_correction_1_coordinator_bootstrap_failure.v1"
    ):
        return validate_correction_coordinator_bootstrap_failure(
            value, reverify_live=reverify_live
        )
    if schema == (
        "plan_aware_monotone_jepa_cost_v1."
        "forensic_correction_1_precommit_staging_failure.v1"
    ):
        return validate_correction_precommit_staging_failure(
            value, reverify_live=reverify_live
        )
    if schema == (
        "plan_aware_monotone_jepa_cost_v1."
        "forensic_correction_1_bootstrap_custody_failure.v1"
    ):
        return _validate_correction_bootstrap_custody_failure(value)
    raise ForensicCorrectionContractError(
        "unsupported correction failure receipt persistence schema"
    )


def _validate_correction_persisted_failure_envelope(
    value: Mapping[str, Any],
    *,
    expected_failure_receipt: Mapping[str, Any],
    reverify_live: bool,
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "failure_receipt", "durable_failure_custody",
        "sequence_success", "no_retry_authorized", "pass",
        "content_digest",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ForensicCorrectionContractError(
            "persisted correction failure envelope key-set drift"
        )
    validated_expected = (
        _validate_correction_failure_receipt_for_persistence(
            expected_failure_receipt, reverify_live=False
        )
    )
    validated_failure = (
        _validate_correction_failure_receipt_for_persistence(
            value["failure_receipt"], reverify_live=False
        )
    )
    durable = value["durable_failure_custody"]
    durable_keys = {
        "path", "sha256", "bytes", "created_exclusive",
        "file_and_parent_directories_fsynced", "no_retry_authorized",
        "pass",
    }
    payload = _authority_bytes(validated_failure)
    if (
        type(durable) is not dict
        or set(durable) != durable_keys
        or durable.get("path")
        != str(CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH)
        or _LOWERCASE_SHA256_RE.fullmatch(
            durable.get("sha256", "")
            if isinstance(durable.get("sha256"), str) else ""
        )
        is None
        or durable.get("sha256") != hashlib.sha256(payload).hexdigest()
        or isinstance(durable.get("bytes"), bool)
        or not isinstance(durable.get("bytes"), int)
        or durable.get("bytes") != len(payload)
        or durable.get("created_exclusive") is not True
        or durable.get("file_and_parent_directories_fsynced") is not True
        or durable.get("no_retry_authorized") is not True
        or durable.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "persisted correction failure durable binding drift"
        )
    if reverify_live:
        observed = BASE._read_no_follow_bound_file(
            CORRECTION_TERMINAL_FAILURE_CUSTODY_ROOT,
            CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH.name,
            expected_stat_rows=None,
        )
        if observed != payload:
            raise ForensicCorrectionContractError(
                "persisted correction failure live bytes drift"
            )
    if (
        value["schema"]
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_persisted_failure.v1"
        )
        or validated_failure != validated_expected
        or value["failure_receipt"] != validated_failure
        or value["sequence_success"] is not False
        or value["no_retry_authorized"] is not True
        or value["pass"] is not True
    ):
        raise ForensicCorrectionContractError(
            "persisted correction failure envelope drift"
        )
    return copy.deepcopy(dict(value))


def _correction_failure_publication_attempted(
    value: Mapping[str, Any],
) -> bool:
    schema = value["schema"]
    if schema == CORRECTION_TERMINAL_COORDINATOR_SCHEMA:
        attempted = value["precommit_staging_attempted"]
    else:
        attempted = value["tracked_publication_attempted"]
    if not isinstance(attempted, bool):
        raise ForensicCorrectionContractError(
            "failure receipt publication-attempt type drift"
        )
    return attempted


def _correction_failure_result_paths_absent(
    value: Mapping[str, Any],
) -> bool:
    schema = value["schema"]
    if schema == CORRECTION_TERMINAL_COORDINATOR_SCHEMA:
        absent = value["result_paths_absent_after_failure"]
    else:
        absent = value["result_paths_absent"]
    if not isinstance(absent, bool):
        raise ForensicCorrectionContractError(
            "failure receipt result-path absence type drift"
        )
    return absent


def _build_correction_failure_persistence_error(
    *,
    failure_receipt: Mapping[str, Any],
    exception: BaseException,
) -> dict[str, Any]:
    validated_failure = (
        _validate_correction_failure_receipt_for_persistence(
            failure_receipt, reverify_live=False
        )
    )
    traceback_text = "".join(
        traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
    )
    role_scan = _observe_correction_role_scan_custody(
        CORRECTION_REPO_ROOT,
        gate="FAILURE_PERSISTENCE_ERROR",
        excluded_pids={os.getpid()},
    )
    receipt = BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_failure_persistence_error.v1"
            ),
            "failure_classification": "UNRESOLVED_TECHNICAL_FAILURE",
            "exact_coordinator_argv": (
                expected_correction_terminal_coordinator_argv()
            ),
            "failure_receipt": validated_failure,
            "failure_receipt_content_digest": validated_failure[
                "content_digest"
            ],
            "tracked_publication_attempted": (
                _correction_failure_publication_attempted(validated_failure)
            ),
            "result_paths_absent_claimed": (
                _correction_failure_result_paths_absent(validated_failure)
            ),
            "persistence_exception_type": type(exception).__name__,
            "persistence_exception_message": str(exception),
            "persistence_traceback": traceback_text,
            "persistence_traceback_sha256": hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest(),
            "durable_failure_custody_path": str(
                CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH
            ),
            "durable_failure_custody_proven": False,
            "failure_role_scan_custody": role_scan,
            "success_literals_emitted": False,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "sequence_success": False,
            "no_retry_authorized": True,
            "pass_meaning": (
                "COMPLETE_FAILURE_PERSISTENCE_ERROR_CUSTODY_"
                "NOT_TECHNICAL_SUCCESS"
            ),
            "pass": True,
        }
    )
    return _validate_correction_failure_persistence_error(receipt)


def _validate_correction_failure_persistence_error(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    expected_keys = {
        "schema", "failure_classification", "exact_coordinator_argv",
        "failure_receipt", "failure_receipt_content_digest",
        "tracked_publication_attempted", "result_paths_absent_claimed",
        "persistence_exception_type", "persistence_exception_message",
        "persistence_traceback", "persistence_traceback_sha256",
        "durable_failure_custody_path", "durable_failure_custody_proven",
        "failure_role_scan_custody", "success_literals_emitted",
        "technical_diagnostic_attempt_accounting", "scientific_counters",
        "files_reused", "sequence_success", "no_retry_authorized",
        "pass_meaning", "pass", "content_digest",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ForensicCorrectionContractError(
            "failure-persistence-error key-set drift"
        )
    failure = _validate_correction_failure_receipt_for_persistence(
        value["failure_receipt"], reverify_live=False
    )
    validate_correction_technical_diagnostic_attempt_accounting(
        value["technical_diagnostic_attempt_accounting"]
    )
    validate_correction_zero_scientific_counters(
        value["scientific_counters"]
    )
    traceback_text = value["persistence_traceback"]
    role_scan = _validate_correction_role_scan_custody(
        value["failure_role_scan_custody"],
        expected_gate="FAILURE_PERSISTENCE_ERROR",
        expected_excluded_pids={os.getpid()},
        reverify_live=False,
    )
    if (
        value["schema"]
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_failure_persistence_error.v1"
        )
        or value["failure_classification"]
        != "UNRESOLVED_TECHNICAL_FAILURE"
        or value["exact_coordinator_argv"]
        != expected_correction_terminal_coordinator_argv()
        or value["failure_receipt"] != failure
        or value["failure_receipt_content_digest"]
        != failure["content_digest"]
        or value["tracked_publication_attempted"]
        is not _correction_failure_publication_attempted(failure)
        or value["result_paths_absent_claimed"]
        is not _correction_failure_result_paths_absent(failure)
        or not isinstance(value["persistence_exception_type"], str)
        or not value["persistence_exception_type"]
        or not isinstance(value["persistence_exception_message"], str)
        or not isinstance(traceback_text, str)
        or not traceback_text
        or hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
        != value["persistence_traceback_sha256"]
        or value["durable_failure_custody_path"]
        != str(CORRECTION_TERMINAL_FAILURE_CUSTODY_PATH)
        or value["durable_failure_custody_proven"] is not False
        or value["failure_role_scan_custody"] != role_scan
        or value["success_literals_emitted"] is not False
        or value["files_reused"] != 0
        or value["sequence_success"] is not False
        or value["no_retry_authorized"] is not True
        or value["pass_meaning"]
        != (
            "COMPLETE_FAILURE_PERSISTENCE_ERROR_CUSTODY_"
            "NOT_TECHNICAL_SUCCESS"
        )
        or value["pass"] is not True
    ):
        raise ForensicCorrectionContractError(
            "failure-persistence-error receipt drift"
        )
    return copy.deepcopy(dict(value))


def _persisted_correction_failure_envelope(
    failure_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _validate_correction_failure_receipt_for_persistence(
        failure_receipt, reverify_live=True
    )
    persisted = _persist_correction_terminal_failure_custody(validated)
    envelope = BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_persisted_failure.v1"
            ),
            "failure_receipt": validated,
            "durable_failure_custody": persisted,
            "sequence_success": False,
            "no_retry_authorized": True,
            "pass": True,
        }
    )
    return _validate_correction_persisted_failure_envelope(
        envelope,
        expected_failure_receipt=validated,
        reverify_live=True,
    )


def build_correction_postcommit_replay_failure(
    exception: BaseException,
) -> dict[str, Any]:
    traceback_text = "".join(
        traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
    )
    role_scan = _observe_correction_role_scan_custody(
        CORRECTION_REPO_ROOT,
        gate="POSTCOMMIT_REPLAY_FAILURE",
        excluded_pids={os.getpid()},
    )
    return BASE.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_correction_1_postcommit_replay_failure.v1"
            ),
            "failure_classification": "UNRESOLVED_TECHNICAL_FAILURE",
            "exact_argv": _correction_proc_argv(os.getpid()),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback_text,
            "traceback_sha256": hashlib.sha256(
                traceback_text.encode("utf-8")
            ).hexdigest(),
            "failure_role_scan_custody": role_scan,
            "technical_diagnostic_attempt_accounting": copy.deepcopy(
                CORRECTION_TECHNICAL_DIAGNOSTIC_ATTEMPT_ACCOUNTING
            ),
            "scientific_counters": copy.deepcopy(
                CORRECTION_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "success_literals_emitted": False,
            "no_retry_authorized": True,
            "pass_meaning": (
                "COMPLETE_POSTCOMMIT_REPLAY_FAILURE_CUSTODY"
            ),
            "pass": True,
        }
    )


def validate_correction_postcommit_replay_failure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    BASE.validate_self_digest(value)
    validate_correction_technical_diagnostic_attempt_accounting(
        value.get("technical_diagnostic_attempt_accounting")
    )
    validate_correction_zero_scientific_counters(
        value.get("scientific_counters")
    )
    traceback_text = value.get("traceback")
    role_scan = _validate_correction_role_scan_custody(
        value.get("failure_role_scan_custody"),
        expected_gate="POSTCOMMIT_REPLAY_FAILURE",
        expected_excluded_pids={os.getpid()},
        reverify_live=False,
    )
    expected_keys = {
        "schema", "failure_classification", "exact_argv",
        "exception_type", "exception_message", "traceback",
        "traceback_sha256", "failure_role_scan_custody",
        "technical_diagnostic_attempt_accounting",
        "scientific_counters", "files_reused", "success_literals_emitted",
        "no_retry_authorized", "pass_meaning", "pass", "content_digest",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "forensic_correction_1_postcommit_replay_failure.v1"
        )
        or value.get("failure_classification")
        != "UNRESOLVED_TECHNICAL_FAILURE"
        or value.get("exact_argv")
        != expected_correction_result_commit_validation_argv()
        or not isinstance(value.get("exception_type"), str)
        or not value["exception_type"]
        or not isinstance(value.get("exception_message"), str)
        or not isinstance(traceback_text, str)
        or not traceback_text
        or hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
        != value.get("traceback_sha256")
        or value.get("failure_role_scan_custody") != role_scan
        or value.get("files_reused") != 0
        or value.get("success_literals_emitted") is not False
        or value.get("no_retry_authorized") is not True
        or value.get("pass_meaning")
        != "COMPLETE_POSTCOMMIT_REPLAY_FAILURE_CUSTODY"
        or value.get("pass") is not True
    ):
        raise ForensicCorrectionContractError(
            "postcommit replay failure receipt drift"
        )
    return copy.deepcopy(dict(value))


def _correction_contract_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            PREEXECUTION_DIAGNOSTIC_CORRECTION_COORDINATOR_SUBCOMMAND,
            PREEXECUTION_FORENSIC_CORRECTION_RESULT_COMMIT_VALIDATION_SUBCOMMAND,
        ),
    )
    args = parser.parse_args()
    if (
        args.command
        == PREEXECUTION_FORENSIC_CORRECTION_RESULT_COMMIT_VALIDATION_SUBCOMMAND
    ):
        try:
            identity = _observe_correction_contract_process_identity(
                os.getpid()
            )
            disposition = validate_correction_result_commit(
                repo_root=CORRECTION_REPO_ROOT,
                postcommit_process_identity=identity,
            )
            sys.stdout.buffer.write(_authority_bytes(disposition))
            sys.stdout.buffer.flush()
            return 0
        except BaseException as exc:
            failure = validate_correction_postcommit_replay_failure(
                build_correction_postcommit_replay_failure(exc)
            )
            sys.stdout.buffer.write(_authority_bytes(failure))
            sys.stdout.buffer.flush()
            return 1
    try:
        receipt = run_correction_terminal_phase_sequence()
    except BaseException as exc:
        try:
            receipt = validate_correction_coordinator_bootstrap_failure(
                _build_correction_coordinator_bootstrap_failure(
                    exception=exc
                ),
                reverify_live=True,
            )
        except BaseException as custody_exc:
            attempted = bool(
                getattr(exc, "tracked_publication_attempted", False)
                or (
                    set(_correction_dirty_paths(CORRECTION_REPO_ROOT))
                    & set(CORRECTION_RESULT_PATHS)
                )
            )
            rollback_value: Mapping[str, Any] | None = None
            rollback_error: str | None = None
            try:
                rollback_value = rollback_correction_result_artifacts(
                    CORRECTION_REPO_ROOT
                )
            except BaseException as rollback_exc:
                rollback_error = (
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
            receipt = _build_correction_bootstrap_custody_failure(
                primary_exception=exc,
                custody_exception=custody_exc,
                tracked_publication_attempted=attempted,
                rollback_receipt=rollback_value,
                rollback_error=rollback_error,
            )
    if receipt.get("sequence_success") is not True:
        try:
            output = _persisted_correction_failure_envelope(receipt)
        except BaseException as persistence_exc:
            output = _build_correction_failure_persistence_error(
                failure_receipt=receipt,
                exception=persistence_exc,
            )
        sys.stdout.buffer.write(_authority_bytes(output))
        sys.stdout.buffer.flush()
        return 1
    try:
        sys.stdout.buffer.write(_authority_bytes(receipt))
        sys.stdout.buffer.flush()
        return 0
    except BaseException as stdout_exc:
        rollback = rollback_correction_result_artifacts(
            CORRECTION_REPO_ROOT
        )
        finalizer = receipt["finalizer_phase_exit_custody"]
        checker = receipt["checker_phase_exit_custody"]
        failure = _build_correction_precommit_staging_failure(
            coordinator=receipt["root_coordinator_process_identity"],
            coordinator_observed_cwd=receipt["coordinator_observed_cwd"],
            fresh_technical_manifest=receipt[
                "fresh_diagnostic_technical_manifest"
            ],
            finalizer_phase=finalizer,
            checker_phase=checker,
            exception=stdout_exc,
            rollback_receipt=rollback,
        )
        try:
            output = _persisted_correction_failure_envelope(failure)
        except BaseException as persistence_exc:
            output = _build_correction_failure_persistence_error(
                failure_receipt=failure,
                exception=persistence_exc,
            )
        try:
            sys.stdout.buffer.write(_authority_bytes(output))
            sys.stdout.buffer.flush()
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(_correction_contract_cli())
