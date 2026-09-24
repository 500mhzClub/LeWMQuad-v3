"""Non-scientific failure-forensics authority for the plan-aware JEPA run.

This module is a separate technical-custody overlay.  It does not alter or
rebuild the frozen scientific contract, output schema, model, targets,
metrics, gates, or classifications.  Importing it opens no panel, predictor,
checkpoint, tensor, metric, route row, or outcome payload.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lewm.safety import plan_aware_monotone_jepa_cost_v1_contract as BASE


FORENSIC_EXPERIMENT_ID = "PLAN_AWARE_MONOTONE_JEPA_COST_V1_FAILURE_FORENSICS"
DATE = "2026-08-27"
SOURCE_COMMIT = "1c18af8c3c45f9b14992362f7e50a35b651c6997"
FORENSIC_LINEAGE = {
    "scientific_source_commit": BASE.SOURCE_COMMIT,
    "scientific_contract_freeze_commit": BASE.INITIAL_EXECUTION_FREEZE_COMMIT,
    "execution_correction_1_commit": (
        BASE.INITIAL_EXECUTION_CORRECTION_FREEZE_COMMIT
    ),
    "execution_correction_2_base_commit": SOURCE_COMMIT,
}
FORENSIC_FREEZE_COMMIT_SUBJECT = (
    "Freeze plan-aware JEPA failure-forensics diagnostic"
)
FORENSIC_RESULT_COMMIT_SUBJECT = "Document plan-aware JEPA failure forensics"
FORENSIC_CONTRACT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_forensic_contract.v1"
)
ARCHIVE_INVENTORY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_archive_inventory.v1"
)
OS_EVIDENCE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.os_evidence_invocation_schema.v1"
)
FORENSIC_FIXTURE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_forensic_fixture.v1"
)
FORENSIC_SOURCE_CLOSURE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_forensic_source_closure.v1"
)
STARTUP_STAGE_ROW_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_startup_stage_row.v1"
)
LAST_STAGE_MARKER_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_last_stage.v1"
)
INVOCATION_RECEIPT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_invocation.v1"
)
TRACKED_INVOCATION_COMPARISON_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_invocation_comparison.v1"
)
TRACKED_OS_EVIDENCE_RECEIPT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_combined_os_evidence.v1"
)
DIAGNOSTIC_CUSTODY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_diagnostic_custody.v1"
)
PREEXECUTION_CHILD_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_only_diagnostic_child.v1"
)
PREEXECUTION_RUNTIME_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_forensic_runtime_result.v1"
)
CONDITIONAL_V2_GATE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.conditional_v2_spec_gate.v1"
)
FORENSIC_RESULT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_forensic_result.v1"
)
FORENSIC_RESULT_COMMIT_VALIDATION_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_result_commit_validation.v1"
)
FORENSIC_FREEZE_PREPARATION_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_freeze_preparation.v1"
)
FORENSIC_AUTHORITY_WRITE_VALIDATION_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.forensic_authority_write_validation.v1"
)
FINAL_DIAGNOSTIC_NAMESPACE_INVENTORY_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.final_diagnostic_namespace_inventory.v1"
)

PREEXECUTION_DIAGNOSTIC_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_v1"
)
PREDECESSOR_SCIENTIFIC_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "jepa_local_waypoint_planning_cost_qualification_v1"
)
FORENSIC_INTERPRETER = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
FORENSIC_INTERPRETER_RESOLVED = Path("/usr/bin/python3.12")
PREEXECUTION_DIAGNOSTIC_INTERPRETER = FORENSIC_INTERPRETER
PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS = {
    "startup_stage_ledger": "receipts/startup_stage.jsonl",
    "child_stdout": "streams/child.stdout",
    "child_stderr": "streams/child.stderr",
    "child_traceback": "streams/child.traceback",
    "child_exception": "receipts/child_exception.json",
    "invocation": "receipts/invocation.json",
    "os_evidence": "receipts/os_evidence.json",
    "environment": "receipts/environment.json",
    "command": "receipts/command.json",
    "heartbeat": "receipts/heartbeat.jsonl",
    "synthetic_technical_lifecycle": (
        "receipts/synthetic_technical_lifecycle.jsonl"
    ),
    "read_guard_manifest": "receipts/read_guard_manifest.json",
    "read_guard_events": "receipts/read_guard_events.jsonl",
    "forensic_freeze_custody": "receipts/forensic_freeze_custody.json",
    "umask_custody": "receipts/umask_custody.json",
    "last_stage_marker": "receipts/last_stage.json",
    "synthetic_results": "receipts/synthetic_results.json",
    "preexecution_only": "receipts/preexecution_only.json",
    "diagnostic_custody": "receipts/diagnostic_custody.json",
    "result": "result.json",
    "final_namespace_inventory": "receipts/final_namespace_inventory.json",
}
PREEXECUTION_DIAGNOSTIC_FINAL_MAIN_FILE_KEYS = tuple(
    key
    for key in PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS
    if key != "synthetic_technical_lifecycle"
)
PREEXECUTION_DIAGNOSTIC_STAGE_IDS = (
    "REQUIRE_LIVE_LAUNCHER",
    "BIND_CHILD_IDENTITY",
    "VALIDATE_FORENSIC_AUTHORITY",
    "VALIDATE_BASE_AUTHORITIES_READ_ONLY",
    "VALIDATE_ARCHIVE_CUSTODY_METADATA_ONLY",
    "ASSERT_NAMESPACE_UNCHANGED",
    "COMPLETE",
)
PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS = (
    "CREATE_TECHNICAL_RESERVATION_DIRECTORY",
    "ACQUIRE_TECHNICAL_LOCK",
    "REGISTER_PROCESS_STATE",
    "ENTER_PREEXECUTION_BOUNDARY",
    "CLEANUP_TECHNICAL_RESOURCES",
)
PREEXECUTION_DIAGNOSTIC_SUBCOMMAND = "diagnose-preexecution"
PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND = "freeze-preexecution-forensic"
PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND = "diagnose-preexecution-child"
PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND = (
    "diagnose-preexecution-synthetic-child"
)
PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES = (
    "PASS",
    "RAISE",
    "EXIT_NONZERO",
    "SIGTERM",
    "MISSING_PATH",
    "UNICODE_RAISE",
)
PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH = Path(
    "scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py"
)
PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS = ("-E", "-s", "-u")
PREEXECUTION_DIAGNOSTIC_UMASK = 0o022
PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK = 0o002
PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY = {
    "wall_clock_timeout_s": 30.0,
    "term_grace_s": 5.0,
    "kill_grace_s": 5.0,
    "timeout_termination_kind": "TIMEOUT",
    "cleanup_sequence": [
        "SEND_SIGTERM_TO_CHILD_PROCESS_GROUP",
        "WAIT_TERM_GRACE",
        "SEND_SIGKILL_TO_CHILD_PROCESS_GROUP_IF_STILL_LIVE",
        "WAIT_KILL_GRACE",
        "ASSERT_PROCESS_GROUP_ROLE_AND_KFD_QUIESCENCE",
    ],
    "unbounded_wait_forbidden": True,
    "timeout_is_technical_diagnostic_failure": True,
    "timeout_consumes_scientific_attempt": False,
}

ROOT_CAUSE_CLASSIFICATIONS = (
    "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED",
    "FINAL_CHILD_LAUNCHER_PATH_DEFECT_IDENTIFIED",
    "FINAL_CHILD_ENVIRONMENT_OR_RESOURCE_FAILURE_IDENTIFIED",
    "TRACEBACK_CUSTODY_DEFECT_CONFIRMED_ROOT_CAUSE_UNRESOLVED",
    "FAILURE_FORENSICS_INCONCLUSIVE",
)
FORENSIC_PRIMARY_CLASSIFICATION = "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED"
FORENSIC_SECONDARY_MECHANISM = "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH"
FORENSIC_DEFECT_ID = (
    "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
)
SCIENTIFIC_DISPOSITION = (
    "PLAN_AWARE_MONOTONE_JEPA_COST_V1_TECHNICAL_NON_RESULT"
)

THIRD_FAILURE_ARCHIVE = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    ".plan_aware_monotone_jepa_cost_v1.failed-1787850081983564058-690965"
)
THIRD_FAILURE_ARCHIVE_INVENTORY = {
    "files": 2,
    "bytes": 3_023,
    "manifest_sha256": (
        "d775d83656de3ab718d031c57a91ebba92b83b2b0eb8e42fc2bf8b4c0a71d6a5"
    ),
    "rows": [
        {
            "path": "receipts/failure.json",
            "sha256": (
                "22f0663136243a9ea8329d1088f6a678e6d6beef7cc17cef967a66dee7701a5d"
            ),
            "bytes": 934,
            "content_digest": (
                "5b4f5a19a1a3fc4b070f49191e308fb2d07fe74204b5bb876b6ba7c1834b91f3"
            ),
        },
        {
            "path": "receipts/terminal_failure_finalization.json",
            "sha256": (
                "d5c9ca34035e22b3b7d3b329379afbd8c936fa1d677f72776e4bcaa48f7a1b62"
            ),
            "bytes": 2_089,
            "content_digest": (
                "c6c766a45cd0c3ea7a9dad4216f32663a5584987ed68eecaf5a89aca94329627"
            ),
        },
    ],
}
ARCHIVE_STAT_MANIFEST_ROW_SCHEMA = ["path", "bytes", "mode", "nlink"]
ARCHIVE_STAT_MANIFEST_SHA256_BY_ORDINAL = {
    1: "a53a718c7a3f9c25a02da6be595c2e37818622bf0288279cf411599bc2747b5e",
    2: "d428e9795ec94dd949161167a7f372b1472642a859964a018346492c4e0d46b2",
    3: "357fb8a697f1e449e8818bc2de1d63a377a8463c7772da75fc4a8b856b56254d",
}
ARCHIVE_ROOT_STAT_AUTHORITY = {
    1: {
        "namespace_timestamp_ns": 1_787_837_464_936_614_634,
        "namespace_timestamp_local": "2026-08-27T14:31:04.936614634+01:00",
        "mode": 0o755,
        "nlink": 8,
        "uid": 1000,
        "gid": 1000,
        "size": 4096,
        "birth": "2026-08-27 14:30:01.704750744 +0100",
        "mtime": "2026-08-27 14:31:04.930864021 +0100",
        "ctime": "2026-08-27 14:31:04.935864030 +0100",
    },
    2: {
        "namespace_timestamp_ns": 1_787_840_583_990_734_770,
        "namespace_timestamp_local": "2026-08-27T15:23:03.990734770+01:00",
        "mode": 0o755,
        "nlink": 9,
        "uid": 1000,
        "gid": 1000,
        "size": 4096,
        "birth": "2026-08-27 15:18:56.031421638 +0100",
        "mtime": "2026-08-27 15:22:30.607793538 +0100",
        "ctime": "2026-08-27 15:23:03.989851396 +0100",
    },
    3: {
        "namespace_timestamp_ns": 1_787_850_081_983_564_058,
        "namespace_timestamp_local": "2026-08-27T18:01:21.983564058+01:00",
        "mode": 0o775,
        "nlink": 3,
        "uid": 1000,
        "gid": 1000,
        "size": 4096,
        "birth": "2026-08-27 18:01:21.982376185 +0100",
        "mtime": "2026-08-27 18:01:21.984376189 +0100",
        "ctime": "2026-08-27 18:01:21.984376189 +0100",
    },
}

ORIGINAL_FINAL_CHILD_RECONSTRUCTION = {
    "schema": (
        "plan_aware_monotone_jepa_cost_v1."
        "original_final_child_reconstruction.v1"
    ),
    "source_receipt": copy.deepcopy(
        THIRD_FAILURE_ARCHIVE_INVENTORY["rows"][1]
    ),
    "launcher_process_identity": {
        "pid": 690_965,
        "process_group_id": 690_965,
        "start_time_ticks": 17_110_176,
        "argv": [
            "/home/andrewknowles/TinyQuadJEPA/bin/python",
            (
                "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
                "evaluate_plan_aware_monotone_jepa_cost_v1.py"
            ),
            "execute",
        ],
        "argv_sha256": (
            "2792a62c325cc75bcd40c6984a5208df05746e1de846dc9fa0b7a6da18209934"
        ),
        "executable": "/usr/bin/python3.12",
        "role": "NONSCIENTIFIC_LAUNCHER",
    },
    "child_process_identity": {
        "pid": 691_063,
        "process_group_id": 691_063,
        "start_time_ticks": 17_110_501,
        "argv": [
            "/home/andrewknowles/TinyQuadJEPA/bin/python",
            (
                "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
                "evaluate_plan_aware_monotone_jepa_cost_v1.py"
            ),
            "execute-scientific",
            "--launcher-pid",
            "690965",
            "--launcher-start-time-ticks",
            "17110176",
        ],
        "argv_sha256": (
            "209cf659109978d21a77ee0f740c2e1516fad937e5d1645fb179857360b2a22f"
        ),
        "executable": "/usr/bin/python3.12",
        "role": "SCIENTIFIC_EVALUATOR",
    },
    "child_returncode": 1,
    "evidence_roles": {
        "launcher_process_identity": "RECEIPT_PROVED",
        "child_process_identity": "RECEIPT_PROVED",
        "child_returncode": "RECEIPT_PROVED",
        "child_cwd": "SOURCE_PROVED",
        "child_stdio_and_session_semantics": "SOURCE_PROVED",
        "scientific_output_root": "SOURCE_PROVED",
        "attempt_root_created": "RECEIPT_AND_NAMESPACE_PROVED_FALSE",
        "launcher_uid_gid_groups_umask_permissions": (
            "RETROSPECTIVELY_UNAVAILABLE"
        ),
        "child_uid_gid_groups_umask_permissions": "RETROSPECTIVELY_UNAVAILABLE",
        "cpu_gpu_visibility_environment": "RETROSPECTIVELY_UNAVAILABLE",
        "child_resource_limits": "RETROSPECTIVELY_UNAVAILABLE",
        "temporary_root_environment": "RETROSPECTIVELY_UNAVAILABLE",
    },
    "committed_launcher_semantics": {
        "cwd": "/home/andrewknowles/Workspace/LeWMQuad-v3",
        "stdin": "INHERITED",
        "stdout": "PIPE",
        "stderr": "STDOUT",
        "close_fds": "SUBPROCESS_DEFAULT_TRUE",
        "start_new_session": True,
        "child_session_id_equals_pid": True,
        "child_process_group_id_equals_pid": True,
        "scientific_output_root": str(BASE.OUTPUT_ROOT),
        "attempt_root_created": False,
    },
    "retrospectively_unavailable": [
        "launcher_cwd",
        "launcher_session_id",
        "launcher_exact_environment",
        "launcher_open_file_descriptors",
        "child_inherited_exact_environment",
        "child_original_resource_limits",
        "launcher_uid_euid_gid_egid_groups_umask_permissions",
        "child_uid_euid_gid_egid_groups_umask_permissions",
        "original_cpu_and_gpu_visibility_environment",
        "original_temporary_root_environment_and_permissions",
    ],
    "non_inference": (
        "archive file ownership or current-process observations are not proof of "
        "the expired launcher or child process identity, environment, limits, or "
        "permissions"
    ),
    "standalone_child_comparator_boundary": (
        "No prior standalone isolated execute-scientific child completed "
        "successfully. Correction 1 executed science directly in the outer "
        "evaluator; comparisons are limited to that direct invocation plus "
        "test/mocked isolated paths."
    ),
}

HISTORICAL_OS_EVIDENCE_AUTHORITY = {
    "schema": (
        "plan_aware_monotone_jepa_cost_v1."
        "historical_final_child_os_evidence.v1"
    ),
    "incident_timestamp_local": "2026-08-27T18:01:21.983564058+01:00",
    "query_window": {
        "local_inclusive_start": "2026-08-27T17:59:00+01:00",
        "local_inclusive_end": "2026-08-27T18:03:30+01:00",
        "utc_inclusive_start": "2026-08-27T16:59:00Z",
        "utc_inclusive_end": "2026-08-27T17:03:30Z",
    },
    "journal_queries": [
        {"selector": "_PID=690965", "rows": 0},
        {"selector": "_PID=691063", "rows": 0},
        {"selector": "_COMM=python3.12", "rows": 0},
        {"selector": "PRIORITY=warning..emerg", "rows": 0},
        {"selector": "kernel relevant regex", "rows": 0},
    ],
    "file_logs": {
        "/var/log/syslog": {
            "rows": 8,
            "relevance": "UNRELATED_SERVICE_OR_NETWORK_ONLY",
            "summaries": [
                "DHCP at 17:59:54",
                "NetworkManager dispatcher",
                "logrotate at 18:00:04",
                "tailscaled timeout at 18:03:27",
            ],
            "pid_python_oom_filesystem_or_audit_rows": 0,
        },
        "/var/log/kern.log": {"rows": 0},
    },
    "unavailable_or_denied_sources": {
        "coredumpctl": "UNAVAILABLE",
        "ausearch": "UNAVAILABLE",
        "dmesg": "PERMISSION_DENIED",
        "audit_log": "UNAVAILABLE",
    },
    "later_current_cgroup_observation": {
        "query_start": "2026-08-27T20:27:23.696952118+01:00",
        "query_end": "2026-08-27T20:27:31.218507143+01:00",
        "proc_self_cgroup": (
            "0::/user.slice/user-1000.slice/user@1000.service/"
            "tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope"
        ),
        "cgroup": (
            "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/"
            "tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope"
        ),
        "memory.events": {
            "path": (
                "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/"
                "tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope/"
                "memory.events"
            ),
            "status": "READABLE",
            "mode": 0o444,
            "uid": 1000,
            "gid": 1000,
            "low": 0,
            "high": 0,
            "max": 0,
            "oom": 0,
            "oom_kill": 0,
            "oom_group_kill": 0,
            "sock_throttled": 0,
        },
        "pids.events": {
            "path": (
                "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/"
                "tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope/"
                "pids.events"
            ),
            "status": "READABLE",
            "mode": 0o444,
            "uid": 1000,
            "gid": 1000,
            "max": 0,
        },
        "cpu.stat": {
            "path": (
                "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/"
                "tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope/"
                "cpu.stat"
            ),
            "status": "READABLE",
            "mode": 0o444,
            "uid": 1000,
            "gid": 1000,
            "usage_usec": 8_987_337_183,
            "user_usec": 7_006_038_972,
            "system_usec": 1_981_298_211,
            "nice_usec": 9_000,
            "core_sched.force_idle_usec": 0,
            "nr_periods": 0,
            "nr_throttled": 0,
            "throttled_usec": 0,
            "nr_bursts": 0,
            "burst_usec": 0,
        },
        "role": (
            "LATER_CURRENT_AGENT_CGROUP_REFERENCE_ONLY_NOT_ORIGINAL_CHILD_EVIDENCE"
        ),
    },
    "interpretation_limits": [
        "absence of a logged event is not proof of absence of a failure",
        "the original child environment and resource levels are unrecoverable",
        "current process limits do not establish original child limits",
        "unrelated syslog rows are not causal evidence",
    ],
}

TRACKED_ATTEMPT_ACCOUNTING_POLICY_PATH = Path(
    "docs/lewm_experiment_attempt_accounting_policy_v2.md"
)
TRACKED_FORENSIC_CONTRACT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_contract.json"
)
TRACKED_ARCHIVE_INVENTORY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_archive_inventory.json"
)
TRACKED_OS_EVIDENCE_SCHEMA_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_os_evidence_schema.json"
)
TRACKED_CONDITIONAL_V2_GATE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_conditional_v2_spec_gate.json"
)
TRACKED_FORENSIC_FIXTURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_evaluator_fixture.json"
)
TRACKED_FORENSIC_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_source_closure.json"
)
TRACKED_FORENSIC_RESULT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_result.json"
)
TRACKED_FORENSIC_REPORT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_forensic_report.md"
)
TRACKED_INVOCATION_RECEIPT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_invocation_receipt.json"
)
TRACKED_OS_EVIDENCE_RECEIPT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_os_evidence_receipt.json"
)
TRACKED_SYNTHETIC_RESULTS_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_synthetic_results.json"
)
TRACKED_PREEXECUTION_ONLY_RECEIPT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_only_receipt.json"
)
TRACKED_DIAGNOSTIC_CUSTODY_RECEIPT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_custody_receipt.json"
)
TRACKED_CONDITIONAL_V2_DECISION_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_conditional_v2_spec_decision.json"
)
TRACKED_CONDITIONAL_V2_SPEC_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v2_specification.md"
)
TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v2_specification.json"
)

BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING = {
    "path": (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
        "execution_correction_2_source_closure.json"
    ),
    "sha256": "b715e328a20c8e18b71fd42f27627d837ed8a9b62febd5347187f582036cbf81",
    "bytes": 25_144,
    "content_digest": (
        "a0d0b2e397cb1142ad6a8e1d63c4c6b27604f1659a10793b8ccc27df016382f5"
    ),
    "rows": 96,
}

FORENSIC_GENERATED_AUTHORITY_PATHS = (
    str(TRACKED_ATTEMPT_ACCOUNTING_POLICY_PATH),
    str(TRACKED_FORENSIC_CONTRACT_PATH),
    str(TRACKED_ARCHIVE_INVENTORY_PATH),
    str(TRACKED_OS_EVIDENCE_SCHEMA_PATH),
    str(TRACKED_CONDITIONAL_V2_GATE_PATH),
    str(TRACKED_FORENSIC_FIXTURE_PATH),
    str(TRACKED_FORENSIC_SOURCE_CLOSURE_PATH),
)
FORENSIC_CODE_AND_TEST_PATHS = (
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_forensic_contract.py",
    str(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH),
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_v1_forensic_contract.py",
    "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1_forensics.py",
)
FORENSIC_REQUIRED_CHANGED_PATHS = (
    *FORENSIC_CODE_AND_TEST_PATHS,
    *FORENSIC_GENERATED_AUTHORITY_PATHS,
)

ATTEMPT_ACCOUNTING_POLICY = {
    "schema": "lewm.experiment_attempt_accounting_policy.v2",
    "attempt_classes": {
        "technical_startup_attempt": (
            "is recorded when the exact child command receipt, external stream "
            "and inherited-FD custody files, and the exclusive external "
            "diagnostic-root/path reservation are "
            "durably created before Popen; child creation then binds those "
            "prelaunch records. Complete diagnostic custody is finalized only "
            "after child exit. It is not a consumed scientific attempt."
        ),
        "scientific_attempt": (
            "begins only after PREEXECUTION passes and the first scientific input "
            "or model is opened"
        ),
        "publication_attempt": (
            "begins when an independently validated immutable scientific payload "
            "enters the authorized durable publication/finalization path"
        ),
    },
    "class_specific_recording_and_consumption": {
        "technical_startup_attempt": (
            "recorded by the pre-Popen exact command receipt plus preopened "
            "external stream/FD custody and exclusive diagnostic-root/path "
            "reservation; the complete "
            "diagnostic-custody receipt is post-exit and never consumes the "
            "scientific retry budget"
        ),
        "scientific_attempt": (
            "consumed only after PREEXECUTION passes and the first scientific "
            "input or model is opened under its durable reservation/namespace"
        ),
        "publication_attempt": (
            "recorded only when immutable-payload publication custody is durably "
            "reserved; it cannot recompute scientific content"
        ),
    },
    "scientific_completion_boundary": (
        "The scientific attempt completes only when the immutable Stage-A, "
        "Stage-B, and frozen conditional-Stage-C payload has been independently "
        "validated before narrative or report generation."
    ),
    "publication_retry_rule": (
        "After scientific completion, presentation, tracked JSON or Markdown, "
        "and Git publication failures are publication attempts. A correction "
        "may use only the byte-exact independently validated payload, with no "
        "training, predictor or model inference, metric recomputation, or new "
        "scientific attempt."
    ),
    "failed_artifact_and_static_input_boundary": {
        "failed_attempt_artifacts_reusable": False,
        "forbidden_failed_artifacts": [
            "checkpoints",
            "scores",
            "candidate_selections",
            "metrics",
            "reports",
        ],
        "independently_frozen_static_inputs_may_remain": [
            "panel",
            "split",
            "route_labels",
            "true_future_tensors",
            "predictor_tensors",
            "hyperparameters",
        ],
    },
    "not_attempts": [
        "an invocation rejected before process creation",
        "a PREEXECUTION_ONLY_DIAGNOSTIC invocation is not a scientific or publication attempt",
        "process start without a scientific reservation or first scientific open is not a scientific attempt",
        "a technical incident archive without evidence of attempt reservation",
    ],
    "failure_incident_policy": (
        "Every launched child failure is retained as an incident even when it did "
        "not consume an experiment attempt."
    ),
    "ambiguity_policy": "FAIL_CLOSED_PENDING_DURABLE_RESERVATION_EVIDENCE",
    "third_incident_provisional_accounting": {
        "attempt_namespace_created": False,
        "attempt_reservation_created": False,
        "scientific_counters_zero": True,
        "attempt_consumed": False,
        "technical_startup_incident": True,
        "technical_startup_attempt_recorded": False,
        "scientific_attempt_consumed": False,
        "publication_attempt_recorded": False,
        "basis": (
            "technical failure receipts and namespace custody; this finding does "
            "not itself authorize another execution"
        ),
    },
    "retry_authority": {
        "current_no_further_retry_authority_remains_in_force": True,
        "automatic_retry": False,
        "v2_requires_passing_conditional_gate": True,
        "v2_requires_explicit_human_or_stakeholder_authorization": True,
    },
    "mandatory_rules": [
        (
            "A failure before PREEXECUTION completion and before any scientific "
            "input or model open is not a scientific attempt."
        ),
        (
            "Technical startup correction and publication-only correction are "
            "accounted separately from outcome-bearing scientific retries."
        ),
        (
            "A publication-only correction may republish only an immutable, "
            "independently validated scientific payload; it may not recompute it."
        ),
        (
            "Every child stdout, stderr, and traceback stream is preopened and "
            "retained outside the attempt namespace."
        ),
        (
            "A final-retry limit is enforced only after diagnostic custody has "
            "qualified the failure phase and attempt class."
        ),
        (
            "Any change informed by scientific outcomes requires a new experiment "
            "version and a new prospectively frozen scientific contract."
        ),
        (
            "An identical technical recovery may not change models, inputs, "
            "targets, metrics, gates, seeds, or scientific decision logic."
        ),
    ],
    "nonretroactivity": (
        "This policy does not retroactively reclassify completed scientific "
        "experiments."
    ),
}

COMMITTED_SOURCE_ROOT_CAUSE_BINDINGS = {
    "commit": SOURCE_COMMIT,
    "evaluator": {
        "path": "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
        "git_blob_sha1": "d93612885e72af4895e02dd38b687482a1c26c8d",
        "sha256": "7eb8a43d75c1296332a36d4c99d426a0c9074a144b0d46233c9a78022914a01b",
    },
    "base_contract": {
        "path": "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py",
        "git_blob_sha1": "d2a0506b7b6c7f7a22d7f77e5bf854c05e627571",
        "sha256": "5588b2345bcc432ef1403a103cac04fb652e3355b270aaf9eaf3a1f72dfc4d47",
    },
    "ast_requirements": {
        "_runtime_execution_correction_2_custody": {
            "validate_execution_correction_2_freeze_custody_calls": 1,
            "verify_full_archives_literal": True,
            "build_execution_correction_2_runtime_custody_calls": 1,
        },
        "_new_attempt": {
            "validate_execution_correction_2_freeze_custody_calls": 1,
            "verify_full_archives_literal": False,
            "whole_failed_archives_inequality_comparisons": 1,
        },
        "validate_execution_correction_2_archive": {
            "full_inventory_verified_assignments_from_verify_full_inventory": 1,
        },
        "execution_correction_2_failed_archive_custody": {
            "minimal_row_count": 2,
        },
        "validate_execution_correction_archive": {
            "detailed_return_dict_count": 1,
        },
    },
}

SCIENTIFIC_IMPLEMENTATION_INVARIANT_BINDINGS = {
    "contract": {
        "path": "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py",
        "git_blob_sha1": "d2a0506b7b6c7f7a22d7f77e5bf854c05e627571",
        "sha256": "5588b2345bcc432ef1403a103cac04fb652e3355b270aaf9eaf3a1f72dfc4d47",
    },
    "model": {
        "path": "lewm/safety/plan_aware_monotone_jepa_cost_v1.py",
        "git_blob_sha1": "840621fc6f8a6f7f62e18be56c10220c4ae47faf",
        "sha256": "df48ce1a4c53c114ccc605c98802ce425005686bc4a4df12475dc7f89546f1a7",
    },
    "metrics": {
        "path": "lewm/safety/plan_aware_monotone_jepa_cost_metrics_v1.py",
        "git_blob_sha1": "09ade75ebc257e9166cc1b286b0be384485b759f",
        "sha256": "3b9603638652fe3cac86a7bd289c85105148c60d077ad253351eb486664a74d3",
    },
}

CONDITIONAL_V2_REQUIRED_CONDITIONS = (
    "forensic_authority_and_source_closure_frozen",
    "preexecution_diagnostic_complete",
    "startup_stage_sequence_complete",
    "invocation_os_stream_and_exception_custody_complete",
    "all_synthetic_failure_modes_captured_and_cleaned",
    "exact_technical_root_cause_reproduced",
    "cause_specific_correction_specified_and_regression_passes",
    "no_scientific_attempt_or_scientific_reservation_namespace_created",
    "no_canonical_or_tracked_scientific_output_written",
    "zero_scientific_inputs_outcomes_tensors_models_and_training",
    "original_scientific_contract_byte_and_digest_immutable",
    "all_failed_archive_files_reused_zero",
    "scientific_namespace_and_child_process_state_clean_launcher_disclosed_live",
    "explicit_human_or_stakeholder_authorization",
)

ROOT_CAUSE_DEFINITIONS = {
    "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED": (
        "A reproducible technical cause in the final scientific-child startup "
        "path is identified, bounded, and independently evidenced."
    ),
    "FINAL_CHILD_LAUNCHER_PATH_DEFECT_IDENTIFIED": (
        "A reproducible defect in launcher/finalizer isolation machinery is "
        "identified: spawn, argv, working directory, interpreter, environment, "
        "stdio, session, or cleanup. It expressly excludes evaluator or base-"
        "contract validation logic."
    ),
    "FINAL_CHILD_ENVIRONMENT_OR_RESOURCE_FAILURE_IDENTIFIED": (
        "A reproducible interpreter, environment, device, memory, process, or "
        "other operating-resource cause is identified before scientific input."
    ),
    "TRACEBACK_CUSTODY_DEFECT_CONFIRMED_ROOT_CAUSE_UNRESOLVED": (
        "The missing traceback or stream custody defect is confirmed, but the "
        "underlying final-child failure cause is not identified."
    ),
    "FAILURE_FORENSICS_INCONCLUSIVE": (
        "The admitted technical evidence is insufficient to identify the cause."
    ),
}


class ForensicContractError(RuntimeError):
    """Raised when technical-forensic custody differs from this authority."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ForensicContractError("value is not canonical JSON") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_self_digest(
    value: Mapping[str, Any], key: str = "content_digest"
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(key, None)
    payload[key] = canonical_json_sha256(payload)
    return payload


def validate_self_digest(
    value: Mapping[str, Any], key: str = "content_digest"
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    declared = payload.pop(key, None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ForensicContractError(f"{key} is absent or malformed")
    if canonical_json_sha256(payload) != declared:
        raise ForensicContractError(f"{key} mismatch")
    return copy.deepcopy(dict(value))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _artifact_binding(path: str, payload: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        **copy.deepcopy(extra),
    }


def validate_artifact_binding(
    value: Mapping[str, Any], *, content_digest_required: bool = False
) -> dict[str, Any]:
    required = {"path", "sha256", "bytes"}
    if content_digest_required:
        required.add("content_digest")
    if set(value) != required:
        raise ForensicContractError("artifact binding key-set drift")
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or isinstance(value["bytes"], bool)
        or not isinstance(value["bytes"], int)
        or value["bytes"] < 0
        or (
            content_digest_required
            and (
                not isinstance(value["content_digest"], str)
                or len(value["content_digest"]) != 64
            )
        )
    ):
        raise ForensicContractError("artifact binding value drift")
    return copy.deepcopy(dict(value))


PROCESS_IDENTITY_FIELDS = {
    "pid",
    "process_group_id",
    "start_time_ticks",
    "argv",
    "argv_sha256",
    "executable",
    "role",
}


def validate_process_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != PROCESS_IDENTITY_FIELDS:
        raise ForensicContractError("process identity key-set drift")
    for key in ("pid", "process_group_id", "start_time_ticks"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise ForensicContractError(f"process identity {key} drift")
    argv = value["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or value["argv_sha256"] != canonical_json_sha256(argv)
        or not isinstance(value["executable"], str)
        or not Path(value["executable"]).is_absolute()
        or not isinstance(value["role"], str)
        or not value["role"]
    ):
        raise ForensicContractError("process identity value drift")
    return copy.deepcopy(dict(value))


def build_startup_stage_row(
    *, sequence: int, stage_id: str, event: str, monotonic_ns: int
) -> dict[str, Any]:
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or stage_id not in PREEXECUTION_DIAGNOSTIC_STAGE_IDS
        or event not in ("STARTED", "COMPLETED")
        or isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns <= 0
    ):
        raise ForensicContractError("startup-stage row input drift")
    return attach_self_digest(
        {
            "schema": STARTUP_STAGE_ROW_SCHEMA,
            "sequence": sequence,
            "stage_id": stage_id,
            "event": event,
            "monotonic_ns": monotonic_ns,
        }
    )


def validate_startup_stage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(row)
    if set(row) != {
        "schema",
        "sequence",
        "stage_id",
        "event",
        "monotonic_ns",
        "content_digest",
    }:
        raise ForensicContractError("startup-stage row key-set drift")
    if (
        row.get("schema") != STARTUP_STAGE_ROW_SCHEMA
        or isinstance(row.get("sequence"), bool)
        or not isinstance(row.get("sequence"), int)
        or row["sequence"] < 0
        or row.get("stage_id") not in PREEXECUTION_DIAGNOSTIC_STAGE_IDS
        or row.get("event") not in ("STARTED", "COMPLETED")
        or isinstance(row.get("monotonic_ns"), bool)
        or not isinstance(row.get("monotonic_ns"), int)
        or row["monotonic_ns"] <= 0
    ):
        raise ForensicContractError("startup-stage row value drift")
    return copy.deepcopy(dict(row))


def validate_startup_stage_rows(
    rows: Sequence[Mapping[str, Any]], *, require_complete: bool
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ForensicContractError("startup-stage ledger must be a sequence")
    validated: list[dict[str, Any]] = []
    expected: list[tuple[str, str]] = []
    for stage_id in PREEXECUTION_DIAGNOSTIC_STAGE_IDS:
        expected.extend([(stage_id, "STARTED"), (stage_id, "COMPLETED")])
    if len(rows) > len(expected) or (require_complete and len(rows) != len(expected)):
        raise ForensicContractError("startup-stage ledger cardinality drift")
    previous_time = 0
    for sequence, row in enumerate(rows):
        validate_startup_stage_row(row)
        if (
            row.get("schema") != STARTUP_STAGE_ROW_SCHEMA
            or row.get("sequence") != sequence
            or (row.get("stage_id"), row.get("event")) != expected[sequence]
            or not isinstance(row.get("monotonic_ns"), int)
            or row["monotonic_ns"] <= previous_time
        ):
            raise ForensicContractError("startup-stage ledger ordering drift")
        previous_time = row["monotonic_ns"]
        validated.append(copy.deepcopy(dict(row)))
    return validated


def build_last_stage_marker(
    *, producer_role: str, stage_id: str, event: str, pid: int, monotonic_ns: int
) -> dict[str, Any]:
    value = attach_self_digest(
        {
            "schema": LAST_STAGE_MARKER_SCHEMA,
            "producer_role": producer_role,
            "stage_id": stage_id,
            "event": event,
            "pid": pid,
            "monotonic_ns": monotonic_ns,
        }
    )
    return validate_last_stage_marker(value)


def validate_last_stage_marker(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "producer_role",
        "stage_id",
        "event",
        "pid",
        "monotonic_ns",
        "content_digest",
    }
    roles = {
        "PREEXECUTION_DIAGNOSTIC_LAUNCHER",
        "PREEXECUTION_DIAGNOSTIC_WRAPPER",
        "PREEXECUTION_DIAGNOSTIC_CHILD",
    }
    wrapper_pairs = {
        ("WRAPPER_BOOTSTRAP_STARTED", "COMPLETED"),
        ("WRAPPER_BOOTSTRAP_COMPLETED", "COMPLETED"),
        ("WRAPPER_ARGPARSE_COMPLETED", "COMPLETED"),
        ("BEFORE_EVALUATOR_IMPORT_AND_ARGPARSE_HANDOFF", "COMPLETED"),
        ("WRAPPER_ARGPARSE_EXCEPTION", "COMPLETED"),
    }
    launcher_pairs = {("LAUNCHER_PRESPAWN", "COMPLETED")}
    child_pairs = {
        (stage_id, event)
        for stage_id in PREEXECUTION_DIAGNOSTIC_STAGE_IDS
        for event in ("STARTED", "COMPLETED")
    }
    pair = (value.get("stage_id"), value.get("event"))
    role = value.get("producer_role")
    role_pair_valid = (
        role == "PREEXECUTION_DIAGNOSTIC_LAUNCHER" and pair in launcher_pairs
    ) or (
        role == "PREEXECUTION_DIAGNOSTIC_WRAPPER" and pair in wrapper_pairs
    ) or (
        role == "PREEXECUTION_DIAGNOSTIC_CHILD" and pair in child_pairs
    )
    if (
        set(value) != required
        or value.get("schema") != LAST_STAGE_MARKER_SCHEMA
        or role not in roles
        or not role_pair_valid
        or isinstance(value.get("pid"), bool)
        or not isinstance(value.get("pid"), int)
        or value["pid"] <= 0
        or isinstance(value.get("monotonic_ns"), bool)
        or not isinstance(value.get("monotonic_ns"), int)
        or value["monotonic_ns"] <= 0
    ):
        raise ForensicContractError("preexecution last-stage marker drift")
    return copy.deepcopy(dict(value))


def build_archive_inventory_authority() -> dict[str, Any]:
    """Bind all incidents without reopening prior scientific archive payloads."""

    first_failure_row = next(
        copy.deepcopy(row)
        for row in BASE.EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS
        if row["path"] == "receipts/failure.json"
    )
    first_failure_row["content_digest"] = (
        BASE.EXECUTION_CORRECTION_RECEIPT_CONTENT_DIGESTS[
            "receipts/failure.json"
        ]
    )
    incident_details = {
        1: {
            "attempt_identity": (
                ".plan_aware_monotone_jepa_cost_v1."
                "attempt-9c1c3adcfb83-1787837401705662496-641931"
            ),
            "original_attempt_path": str(
                BASE.OUTPUT_ROOT.parent
                / (
                    ".plan_aware_monotone_jepa_cost_v1."
                    "attempt-9c1c3adcfb83-1787837401705662496-641931"
                )
            ),
            "attempt_identity_provenance": "FROZEN_PRIOR_CUSTODY_AUTHORITY",
            "stage_reached": "CONDITIONAL_STAGE_B_AND_C",
            "process_ids": {"evaluator_pid": 641_931},
            "returncode": "NOT_PERSISTED",
            "stdout": {"path": "NOT_PERSISTED", "bytes": "NOT_PERSISTED"},
            "stderr": {"path": "NOT_PERSISTED", "bytes": "NOT_PERSISTED"},
            "available_technical_logs": [
                {
                    "path": "logs/stage_b_proprio_predictor_materialisation.log",
                    "bytes": 9_305,
                    "role": "NOT_DEDICATED_LAUNCHER_STDOUT_OR_STDERR",
                    "content_opened_by_forensics": False,
                }
            ],
            "failure_receipt": first_failure_row,
            "terminal_receipt": "ABSENT_FROM_BOUND_ARCHIVE_INVENTORY",
            "rows_opened": {"calibration": 96, "heldout": 96},
            "scientific_payload_exists": True,
            "attempt_namespace_created": True,
            "scientific_attempt_consumed": True,
        },
        2: {
            "attempt_identity": (
                ".plan_aware_monotone_jepa_cost_v1."
                "attempt-14625958c0fc-1787840336032855487-659206"
            ),
            "original_attempt_path": str(
                BASE.OUTPUT_ROOT.parent
                / (
                    ".plan_aware_monotone_jepa_cost_v1."
                    "attempt-14625958c0fc-1787840336032855487-659206"
                )
            ),
            "attempt_identity_provenance": "FROZEN_PRIOR_CUSTODY_AUTHORITY",
            "stage_reached": "PERSISTENCE",
            "process_ids": {"evaluator_pid": 659_206},
            "returncode": "NOT_PERSISTED",
            "stdout": {"path": "NOT_PERSISTED", "bytes": "NOT_PERSISTED"},
            "stderr": {"path": "NOT_PERSISTED", "bytes": "NOT_PERSISTED"},
            "available_technical_logs": [
                {
                    "path": "logs/stage_b_proprio_predictor_materialisation.log",
                    "bytes": 0,
                    "role": "NOT_DEDICATED_LAUNCHER_STDOUT_OR_STDERR",
                    "content_opened_by_forensics": False,
                }
            ],
            "failure_receipt": copy.deepcopy(
                BASE.EXECUTION_CORRECTION_2_ARCHIVE_KEY_BINDINGS["failure"]
            ),
            "terminal_receipt": "ABSENT_FROM_BOUND_KEY_RECEIPTS",
            "rows_opened": {"calibration": 96, "heldout": 96},
            "scientific_payload_exists": True,
            "attempt_namespace_created": True,
            "scientific_attempt_consumed": True,
        },
        3: {
            "attempt_identity": "NO_ATTEMPT_NAMESPACE_CREATED",
            "original_attempt_path": "NOT_CREATED",
            "stage_reached": "PREATTEMPT_NEW_ATTEMPT_CUSTODY_VALIDATION",
            "process_ids": {
                "launcher_pid": 690_965,
                "scientific_child_pid": 691_063,
            },
            "returncode": 1,
            "stdout": {"path": "NOT_PERSISTED", "bytes": "NOT_PERSISTED"},
            "stderr": {
                "path": "REDIRECTED_TO_STDOUT_NOT_PERSISTED",
                "bytes": "NOT_PERSISTED",
            },
            "available_technical_logs": [],
            "failure_receipt": copy.deepcopy(
                THIRD_FAILURE_ARCHIVE_INVENTORY["rows"][0]
            ),
            "terminal_receipt": copy.deepcopy(
                THIRD_FAILURE_ARCHIVE_INVENTORY["rows"][1]
            ),
            "rows_opened": {"calibration": 0, "heldout": 0},
            "scientific_payload_exists": False,
            "attempt_namespace_created": False,
            "scientific_attempt_consumed": False,
        },
    }

    archives = [
        {
            "ordinal": 1,
            "archive_path": str(BASE.EXECUTION_CORRECTION_FAILED_ARCHIVE),
            "source_freeze_commit": BASE.INITIAL_EXECUTION_FREEZE_COMMIT,
            "inventory": copy.deepcopy(BASE.EXECUTION_CORRECTION_ARCHIVE_INVENTORY),
            "verification": "EXISTING_BOUND_MANIFEST_AND_DIRECTORY_STAT_ONLY",
            "archive_payload_files_opened_by_forensic_author": 0,
            "files_reused": 0,
            "nonreusable": True,
            "stat_manifest": {
                "row_fields": ARCHIVE_STAT_MANIFEST_ROW_SCHEMA,
                "sha256": ARCHIVE_STAT_MANIFEST_SHA256_BY_ORDINAL[1],
            },
            "root_stat_authority": copy.deepcopy(ARCHIVE_ROOT_STAT_AUTHORITY[1]),
            "incident_accounting": incident_details[1],
        },
        {
            "ordinal": 2,
            "archive_path": str(BASE.EXECUTION_CORRECTION_2_FAILED_ARCHIVE),
            "source_freeze_commit": BASE.INITIAL_EXECUTION_CORRECTION_FREEZE_COMMIT,
            "inventory": copy.deepcopy(BASE.EXECUTION_CORRECTION_2_ARCHIVE_INVENTORY),
            "verification": "EXISTING_BOUND_MANIFEST_AND_DIRECTORY_STAT_ONLY",
            "archive_payload_files_opened_by_forensic_author": 0,
            "files_reused": 0,
            "nonreusable": True,
            "stat_manifest": {
                "row_fields": ARCHIVE_STAT_MANIFEST_ROW_SCHEMA,
                "sha256": ARCHIVE_STAT_MANIFEST_SHA256_BY_ORDINAL[2],
            },
            "root_stat_authority": copy.deepcopy(ARCHIVE_ROOT_STAT_AUTHORITY[2]),
            "incident_accounting": incident_details[2],
        },
        {
            "ordinal": 3,
            "archive_path": str(THIRD_FAILURE_ARCHIVE),
            "source_freeze_commit": SOURCE_COMMIT,
            "inventory": {
                "files": THIRD_FAILURE_ARCHIVE_INVENTORY["files"],
                "bytes": THIRD_FAILURE_ARCHIVE_INVENTORY["bytes"],
                "manifest_sha256": THIRD_FAILURE_ARCHIVE_INVENTORY[
                    "manifest_sha256"
                ],
            },
            "technical_receipts": copy.deepcopy(
                THIRD_FAILURE_ARCHIVE_INVENTORY["rows"]
            ),
            "verification": (
                "EXACT_TWO_TECHNICAL_RECEIPTS_MAY_BE_VERIFIED;_NO_SCIENTIFIC_PAYLOAD"
            ),
            "scientific_payload_files": 0,
            "files_reused": 0,
            "nonreusable": True,
            "stat_manifest": {
                "row_fields": ARCHIVE_STAT_MANIFEST_ROW_SCHEMA,
                "sha256": ARCHIVE_STAT_MANIFEST_SHA256_BY_ORDINAL[3],
            },
            "root_stat_authority": copy.deepcopy(ARCHIVE_ROOT_STAT_AUTHORITY[3]),
            "incident_accounting": incident_details[3],
        },
    ]
    return attach_self_digest(
        {
            "schema": ARCHIVE_INVENTORY_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "forensic_base_source_commit": SOURCE_COMMIT,
            "lineage": copy.deepcopy(FORENSIC_LINEAGE),
            "archives": archives,
            "archive_count": 3,
            "scientific_payload_values_opened_or_interpreted": False,
            "metadata_only_scope_for_prior_scientific_archives": True,
            "all_files_reused": 0,
        }
    )


def validate_archive_inventory_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    expected = build_archive_inventory_authority()
    if dict(value) != expected:
        raise ForensicContractError("forensic archive inventory authority drift")
    return copy.deepcopy(dict(value))


ARCHIVE_INVENTORY_AUTHORITY = build_archive_inventory_authority()
ARCHIVE_INVENTORY_AUTHORITY_BYTES = canonical_json_bytes(
    ARCHIVE_INVENTORY_AUTHORITY
) + b"\n"
ARCHIVE_INVENTORY_AUTHORITY_BINDING = _artifact_binding(
    str(TRACKED_ARCHIVE_INVENTORY_PATH),
    ARCHIVE_INVENTORY_AUTHORITY_BYTES,
    content_digest=ARCHIVE_INVENTORY_AUTHORITY["content_digest"],
)


def verify_third_failure_technical_receipts() -> dict[str, Any]:
    """Verify only the two admitted technical receipts in the third incident."""

    root = THIRD_FAILURE_ARCHIVE.absolute()
    observed: list[dict[str, Any]] = []
    for expected in THIRD_FAILURE_ARCHIVE_INVENTORY["rows"]:
        relative = Path(str(expected["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ForensicContractError("technical receipt path is unsafe")
        try:
            raw = _read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
        except (OSError, ForensicContractError) as exc:
            raise ForensicContractError("third technical receipt read failed") from exc
        sha256 = hashlib.sha256(raw).hexdigest()
        size = len(raw)
        if sha256 != expected["sha256"] or size != expected["bytes"]:
            raise ForensicContractError("third technical receipt byte drift")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ForensicContractError("third technical receipt parse drift") from exc
        validate_self_digest(payload)
        if payload["content_digest"] != expected["content_digest"]:
            raise ForensicContractError("third technical receipt digest drift")
        observed.append(copy.deepcopy(expected))
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.third_incident_receipt_check.v1",
            "archive_path": str(root),
            "technical_receipts": observed,
            "distinct_technical_receipt_paths": 2,
            "technical_file_open_operations_this_validation": 2,
            "scientific_payload_files_opened": 0,
            "pass": True,
        }
    )


def build_archive_verification_record_equality_fixture() -> dict[str, Any]:
    """Pure synthetic proof of the incompatible whole-record schema defect."""

    shared_rows = []
    for ordinal in (1, 2):
        shared_rows.append(
            {
                "archive_path": f"/synthetic/immutable-failed-archive-{ordinal}",
                "source_freeze_commit": str(ordinal) * 40,
                "inventory": {
                    "files": ordinal,
                    "bytes": ordinal * 10,
                    "manifest_sha256": str(ordinal) * 64,
                },
                "failure_receipt": {
                    "path": "receipts/failure.json",
                    "sha256": "f" * 64,
                    "bytes": 1,
                    "content_digest": "e" * 64,
                },
                "files_reused": 0,
            }
        )
    minimal_rows = [
        {
            **copy.deepcopy(shared_rows[0]),
            "partial_artifacts_reusable": False,
        },
        {
            **copy.deepcopy(shared_rows[1]),
            "partial_artifacts_reusable": False,
        },
    ]
    detailed_rows = [
        {
            **copy.deepcopy(shared_rows[0]),
            "source_closure_snapshot": {"synthetic": True},
            "stage_b_gate_receipt": {"synthetic": True},
            "nothing_running": True,
            "pass": True,
        },
        {
            **copy.deepcopy(shared_rows[1]),
            "persistence_receipt": {"synthetic": True},
            "stage_c_executed": False,
            "partial_artifacts_reusable": False,
            "full_inventory_verified": False,
            "pass": True,
        },
    ]
    projection_fields = (
        "archive_path",
        "source_freeze_commit",
        "inventory",
        "failure_receipt",
        "files_reused",
    )
    minimal_projection = [
        {key: copy.deepcopy(row[key]) for key in projection_fields}
        for row in minimal_rows
    ]
    detailed_projection = [
        {key: copy.deepcopy(row[key]) for key in projection_fields}
        for row in detailed_rows
    ]
    key_differences = []
    for ordinal, (minimal, detailed) in enumerate(
        zip(minimal_rows, detailed_rows), start=1
    ):
        key_differences.append(
            {
                "ordinal": ordinal,
                "minimal_only": sorted(set(minimal) - set(detailed)),
                "detailed_only": sorted(set(detailed) - set(minimal)),
                "shared": sorted(set(minimal) & set(detailed)),
            }
        )
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "archive_verification_record_equality_fixture.v1"
            ),
            "input_kind": "SYNTHETIC_TECHNICAL_METADATA_ONLY",
            "scientific_inputs_opened": 0,
            "defect_id": (
                "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
            ),
            "runtime_minimal_records": minimal_rows,
            "new_attempt_detailed_records": detailed_rows,
            "per_row_key_differences": key_differences,
            "raw_records_equal": minimal_rows == detailed_rows,
            "identity_projection_fields": list(projection_fields),
            "identity_projections_equal": (
                minimal_projection == detailed_projection
            ),
            "defect_reproduced": minimal_rows != detailed_rows,
            "corrected_identity_comparison_passes": (
                minimal_projection == detailed_projection
            ),
            "full_inventory_verified_is_one_detail_not_sole_cause": True,
        }
    )


def build_forensic_contract() -> dict[str, Any]:
    return attach_self_digest(
        {
            "schema": FORENSIC_CONTRACT_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "date": DATE,
            "forensic_base_source_commit": SOURCE_COMMIT,
            "lineage": copy.deepcopy(FORENSIC_LINEAGE),
            "freeze_commit_subject": FORENSIC_FREEZE_COMMIT_SUBJECT,
            "result_commit_subject": FORENSIC_RESULT_COMMIT_SUBJECT,
            "scientific_authority": {
                "freeze_commit": BASE.INITIAL_EXECUTION_FREEZE_COMMIT,
                "contract_sha256": BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
                "mutation_authorized": False,
            },
            "scope": {
                "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
                "technical_failure_forensics_only": True,
                "scientific_execution": False,
                "scientific_payload_reads": 0,
                "training_steps": 0,
                "canonical_or_tracked_scientific_publication": False,
                "root_cause_classification_count": 1,
            },
            "freeze_authority_publication": {
                "subcommand": PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND,
                "preflight_api": "validate_forensic_freeze_preparation",
                "writer_api": "write_forensic_authorities",
                "postflight_api": "validate_forensic_authority_write",
                "rollback_api": "rollback_forensic_authorities",
                "intended_paths": list(FORENSIC_GENERATED_AUTHORITY_PATHS),
                "preflight_complete_source_closure_bound": True,
                "preexisting_source_rows_exact_across_preflight_and_postflight": (
                    True
                ),
                "generated_authority_rows_bound_to_in_memory_bytes": True,
                "writer_failure_semantics": (
                    "FULL_INTENDED_SET_DURABLY_ABSENT_OR_EXPLICIT_"
                    "CLEANUP_FAILURE"
                ),
                "postflight_failure_semantics": (
                    "CALLER_MUST_ROLL_BACK_FULL_GENERATED_AUTHORITY_SET"
                ),
                "legacy_scientific_freeze_contract_path_used": False,
            },
            "attempt_accounting_policy": copy.deepcopy(
                ATTEMPT_ACCOUNTING_POLICY
            ),
            "archive_inventory": copy.deepcopy(
                ARCHIVE_INVENTORY_AUTHORITY_BINDING
            ),
            "diagnostic": {
                "root": str(PREEXECUTION_DIAGNOSTIC_ROOT),
                "interpreter": {
                    "lexical": str(FORENSIC_INTERPRETER),
                    "resolved_executable": str(FORENSIC_INTERPRETER_RESOLVED),
                    "separately_frozen_for_technical_diagnostic": True,
                    "may_equal_scientific_interpreter": True,
                    "original_final_child_scientific_interpreter": (
                        ORIGINAL_FINAL_CHILD_RECONSTRUCTION[
                            "child_process_identity"
                        ]["argv"][0]
                    ),
                    "lexically_equal_to_original_final_child_interpreter": (
                        str(FORENSIC_INTERPRETER)
                        == ORIGINAL_FINAL_CHILD_RECONSTRUCTION[
                            "child_process_identity"
                        ]["argv"][0]
                    ),
                },
                "runtime_paths": copy.deepcopy(PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS),
                "timeout_policy": copy.deepcopy(
                    PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY
                ),
                "stage_ids": list(PREEXECUTION_DIAGNOSTIC_STAGE_IDS),
                "launcher_subcommand": PREEXECUTION_DIAGNOSTIC_SUBCOMMAND,
                "freeze_subcommand": PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND,
                "child_subcommand": PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND,
                "synthetic_child_subcommand": (
                    PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND
                ),
                "synthetic_fixtures": list(
                    PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
                ),
                "synthetic_technical_lifecycle_stage_ids": list(
                    PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS
                ),
                "outer_child_wrapper": {
                    "path": str(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH),
                    "interpreter_flags": list(
                        PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS
                    ),
                    "real_argv_grammar": [
                        "FORENSIC_INTERPRETER",
                        "-E",
                        "-s",
                        "-u",
                        str(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH),
                        "--mode",
                        PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND,
                        "--launcher-pid",
                        "POSITIVE_DECIMAL",
                        "--launcher-start-time-ticks",
                        "POSITIVE_DECIMAL",
                        "--diagnostic-root",
                        str(PREEXECUTION_DIAGNOSTIC_ROOT),
                        "--traceback-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--exception-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--heartbeat-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--read-guard-manifest",
                        str(
                            PREEXECUTION_DIAGNOSTIC_ROOT
                            / "receipts/read_guard_manifest.json"
                        ),
                        "--read-guard-events-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                    ],
                    "synthetic_argv_grammar": [
                        "FORENSIC_INTERPRETER",
                        "-E",
                        "-s",
                        "-u",
                        str(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH),
                        "--mode",
                        PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND,
                        "--fixture-id",
                        "ONE_OF_SYNTHETIC_FIXTURES",
                        "--diagnostic-root",
                        (
                            "ABSOLUTE_PYTEST_TEMP_ROOT_OR_EXACT_"
                            "FORENSIC_ROOT_SYNTHETIC_FIXTURE_SUBTREE"
                        ),
                        "--traceback-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--exception-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--heartbeat-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                        "--read-guard-manifest",
                        "ABSOLUTE_ROOT_RECEIPTS_READ_GUARD_MANIFEST",
                        "--read-guard-events-fd",
                        "CANONICAL_DECIMAL_FD_AT_LEAST_3",
                    ],
                    "outer_wrapper_is_stdlib_only_and_import_safe": True,
                },
                "environment": {
                    "remove_every_inherited_key_with_prefix": "PYTHON",
                    "set": {
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONFAULTHANDLER": "1",
                    },
                    "raw_streams_preopened_by_launcher": True,
                    "traceback_exception_and_heartbeat_fds_passed": True,
                },
            },
            "root_cause_classifications": list(ROOT_CAUSE_CLASSIFICATIONS),
            "root_cause_definitions": copy.deepcopy(ROOT_CAUSE_DEFINITIONS),
            "candidate_root_cause": {
                "classification_if_all_proofs_pass": (
                    "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED"
                ),
                "technical_mechanism": (
                    "evaluator _new_attempt pre-attempt raw whole-list equality "
                    "compares minimal runtime-custody archive records with detailed "
                    "validator archive records that have incompatible key sets"
                ),
                "secondary_mechanism": "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH",
                "defect_id": (
                    "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
                ),
                "not_yet_a_result": True,
                "required_proofs": [
                    "exact committed-source blob and AST proof at 1c18af8",
                    "pure synthetic incompatible-schema whole-record fixture",
                    "external zero-input preexecution diagnostic",
                    "complete raw stream and OS custody",
                    "independent regression of identity-projection correction",
                ],
            },
            "committed_source_root_cause_authority": copy.deepcopy(
                COMMITTED_SOURCE_ROOT_CAUSE_BINDINGS
            ),
            "original_final_child_reconstruction": copy.deepcopy(
                ORIGINAL_FINAL_CHILD_RECONSTRUCTION
            ),
            "historical_os_evidence": copy.deepcopy(
                HISTORICAL_OS_EVIDENCE_AUTHORITY
            ),
            "conditional_v2": {
                "authorization_is_boolean_not_classification": True,
                "required_conditions": list(CONDITIONAL_V2_REQUIRED_CONDITIONS),
                "automatic_execution": False,
            },
            "claims_boundary": {
                "outcome_or_metric_values_inspected_or_interpreted": False,
                "scientific_archive_payloads_opened": False,
                "technical_receipts_and_filesystem_metadata_only": True,
                "no_safety_deployment_or_model_claim": True,
            },
        }
    )


FORENSIC_CONTRACT = build_forensic_contract()
FORENSIC_CONTRACT_BYTES = canonical_json_bytes(FORENSIC_CONTRACT) + b"\n"
FORENSIC_CONTRACT_SHA256 = FORENSIC_CONTRACT["content_digest"]
FORENSIC_CONTRACT_BINDING = _artifact_binding(
    str(TRACKED_FORENSIC_CONTRACT_PATH),
    FORENSIC_CONTRACT_BYTES,
    content_digest=FORENSIC_CONTRACT_SHA256,
)


def validate_forensic_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    if dict(value) != FORENSIC_CONTRACT:
        raise ForensicContractError("forensic contract drift")
    return copy.deepcopy(dict(value))


def _git_blob_bytes(repo_root: Path, commit: str, path: str) -> tuple[bytes, str]:
    try:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path}"],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ForensicContractError("cannot bind committed forensic source") from exc
    return payload, blob


def _ast_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(rows) != 1:
        raise ForensicContractError(f"committed function cardinality drift: {name}")
    return rows[0]


def _keyword_literal(call: ast.Call, name: str) -> Any:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1 or not isinstance(matches[0], ast.Constant):
        raise ForensicContractError(f"committed keyword literal drift: {name}")
    return matches[0].value


def _calls_named(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    output: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and target.attr == name:
            output.append(node)
        elif isinstance(target, ast.Name) and target.id == name:
            output.append(node)
    return output


def _literal_dict_keys(node: ast.Dict) -> list[str]:
    keys: list[str] = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ForensicContractError("committed return dict has nonliteral key")
        keys.append(key.value)
    return keys


def _single_return_dict_keys(function: ast.FunctionDef) -> list[str]:
    rows = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    if len(rows) != 1:
        raise ForensicContractError(
            f"committed return-dict cardinality drift: {function.name}"
        )
    return _literal_dict_keys(rows[0])


def _single_return_list_dict_keys(function: ast.FunctionDef) -> list[list[str]]:
    rows = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.List)
    ]
    if len(rows) != 1 or any(not isinstance(item, ast.Dict) for item in rows[0].elts):
        raise ForensicContractError(
            f"committed return-list cardinality drift: {function.name}"
        )
    return [_literal_dict_keys(item) for item in rows[0].elts]


def _is_failed_archives_get(node: ast.AST, receiver: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "failed_archives"
    )


def build_committed_source_root_cause_proof(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Prove the exact 1c18af8 control-path mismatch without opening data."""

    root = Path(repo_root).resolve()
    bound_sources: dict[str, dict[str, Any]] = {}
    parsed: dict[str, ast.Module] = {}
    for label in ("evaluator", "base_contract"):
        expected = COMMITTED_SOURCE_ROOT_CAUSE_BINDINGS[label]
        payload, blob = _git_blob_bytes(root, SOURCE_COMMIT, expected["path"])
        observed_sha = hashlib.sha256(payload).hexdigest()
        if blob != expected["git_blob_sha1"] or observed_sha != expected["sha256"]:
            raise ForensicContractError("committed source binding drift")
        try:
            parsed[label] = ast.parse(payload.decode("utf-8"))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise ForensicContractError("committed source AST parse failed") from exc
        bound_sources[label] = {
            **copy.deepcopy(expected),
            "bytes": len(payload),
            "parsed_as_python_ast": True,
        }

    runtime = _ast_function(
        parsed["evaluator"], "_runtime_execution_correction_2_custody"
    )
    new_attempt = _ast_function(parsed["evaluator"], "_new_attempt")
    archive_validator = _ast_function(
        parsed["base_contract"], "validate_execution_correction_2_archive"
    )
    first_archive_validator = _ast_function(
        parsed["base_contract"], "validate_execution_correction_archive"
    )
    minimal_archive_builder = _ast_function(
        parsed["base_contract"], "execution_correction_2_failed_archive_custody"
    )
    runtime_custody_builder = _ast_function(
        parsed["base_contract"], "build_execution_correction_2_runtime_custody"
    )
    isolated_runner = _ast_function(
        parsed["evaluator"], "_run_exact_isolated_process"
    )
    launcher_lifecycle = _ast_function(
        parsed["evaluator"], "_execute_launcher_lifecycle"
    )

    runtime_calls = _calls_named(
        runtime, "validate_execution_correction_2_freeze_custody"
    )
    attempt_calls = _calls_named(
        new_attempt, "validate_execution_correction_2_freeze_custody"
    )
    runtime_builder_calls = _calls_named(
        runtime, "build_execution_correction_2_runtime_custody"
    )
    minimal_builder_calls = _calls_named(
        runtime_custody_builder, "execution_correction_2_failed_archive_custody"
    )
    if len(runtime_calls) != 1 or _keyword_literal(
        runtime_calls[0], "verify_full_archives"
    ) is not True:
        raise ForensicContractError("runtime full archive proof-strength drift")
    if len(attempt_calls) != 1 or _keyword_literal(
        attempt_calls[0], "verify_full_archives"
    ) is not False:
        raise ForensicContractError("attempt reduced archive proof-strength drift")
    if len(runtime_builder_calls) != 1 or len(minimal_builder_calls) != 1:
        raise ForensicContractError("runtime minimal archive-custody provenance drift")

    failed_archive_comparisons = 0
    comparison_lines: list[int] = []
    for node in ast.walk(new_attempt):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotEq)
            and len(node.comparators) == 1
            and _is_failed_archives_get(
                node.left, "execution_correction_2_custody"
            )
            and _is_failed_archives_get(node.comparators[0], "runtime")
        ):
            failed_archive_comparisons += 1
            comparison_lines.append(int(node.lineno))
    if failed_archive_comparisons != 1:
        raise ForensicContractError("whole failed-archive comparison drift")

    assignments = 0
    assignment_lines: list[int] = []
    for node in ast.walk(archive_validator):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "full_inventory_verified"
                and isinstance(value, ast.Name)
                and value.id == "verify_full_inventory"
            ):
                assignments += 1
                assignment_lines.append(int(key.lineno))
    if assignments != 1:
        raise ForensicContractError("archive proof-strength return field drift")

    popen_calls = _calls_named(isolated_runner, "Popen")
    if len(popen_calls) != 1:
        raise ForensicContractError("committed isolated Popen cardinality drift")
    popen_keywords = {
        keyword.arg: keyword.value
        for keyword in popen_calls[0].keywords
        if keyword.arg is not None
    }
    required_popen_keywords = {
        "cwd",
        "env",
        "text",
        "stdout",
        "stderr",
        "start_new_session",
    }
    if set(popen_keywords) != required_popen_keywords:
        raise ForensicContractError("committed isolated Popen keyword-set drift")
    if (
        not isinstance(popen_keywords["cwd"], ast.Name)
        or popen_keywords["cwd"].id != "ROOT"
        or not isinstance(popen_keywords["text"], ast.Constant)
        or popen_keywords["text"].value is not True
        or not isinstance(popen_keywords["start_new_session"], ast.Constant)
        or popen_keywords["start_new_session"].value is not True
        or not isinstance(popen_keywords["stdout"], ast.Attribute)
        or popen_keywords["stdout"].attr != "PIPE"
        or not isinstance(popen_keywords["stderr"], ast.Attribute)
        or popen_keywords["stderr"].attr != "STDOUT"
        or not isinstance(popen_keywords["env"], ast.IfExp)
    ):
        raise ForensicContractError("committed isolated Popen semantics drift")
    scientific_launch_calls = []
    for call in _calls_named(launcher_lifecycle, "_run_exact_isolated_process"):
        if (
            call.args
            and isinstance(call.args[0], ast.Call)
            and isinstance(call.args[0].func, ast.Name)
            and call.args[0].func.id == "_expected_scientific_argv"
        ):
            scientific_launch_calls.append(call)
    if len(scientific_launch_calls) != 1 or any(
        keyword.arg == "environment"
        for keyword in scientific_launch_calls[0].keywords
    ):
        raise ForensicContractError("committed scientific child env call drift")

    minimal_key_sets = _single_return_list_dict_keys(minimal_archive_builder)
    first_detailed_keys = _single_return_dict_keys(first_archive_validator)
    second_detailed_keys = _single_return_dict_keys(archive_validator)
    expected_minimal = [
        [
            "archive_path",
            "source_freeze_commit",
            "inventory",
            "failure_receipt",
            "files_reused",
            "partial_artifacts_reusable",
        ],
        [
            "archive_path",
            "source_freeze_commit",
            "inventory",
            "failure_receipt",
            "files_reused",
            "partial_artifacts_reusable",
        ],
    ]
    expected_first_detailed = [
        "archive_path",
        "inventory",
        "source_freeze_commit",
        "failure_receipt",
        "source_closure_snapshot",
        "stage_b_gate_receipt",
        "files_reused",
        "nothing_running",
        "pass",
    ]
    expected_second_detailed = [
        "archive_path",
        "source_freeze_commit",
        "inventory",
        "failure_receipt",
        "persistence_receipt",
        "stage_c_executed",
        "files_reused",
        "partial_artifacts_reusable",
        "full_inventory_verified",
        "pass",
    ]
    if (
        minimal_key_sets != expected_minimal
        or first_detailed_keys != expected_first_detailed
        or second_detailed_keys != expected_second_detailed
    ):
        raise ForensicContractError("committed archive custody schema key-set drift")

    per_row_key_differences = []
    for ordinal, (minimal, detailed) in enumerate(
        zip(minimal_key_sets, [first_detailed_keys, second_detailed_keys]),
        start=1,
    ):
        per_row_key_differences.append(
            {
                "ordinal": ordinal,
                "minimal_runtime_only": sorted(set(minimal) - set(detailed)),
                "detailed_validator_only": sorted(set(detailed) - set(minimal)),
                "shared": sorted(set(minimal) & set(detailed)),
            }
        )

    evidence = {
        "runtime_full_verification_call": {
            "function": runtime.name,
            "line": int(runtime_calls[0].lineno),
            "call_count": len(runtime_calls),
            "verify_full_archives": True,
        },
        "runtime_minimal_custody_builder": {
            "evaluator_function": runtime.name,
            "call_line": int(runtime_builder_calls[0].lineno),
            "base_builder_function": runtime_custody_builder.name,
            "minimal_archive_builder_function": minimal_archive_builder.name,
            "minimal_builder_call_line": int(minimal_builder_calls[0].lineno),
            "row_key_sets": minimal_key_sets,
        },
        "attempt_reduced_verification_call": {
            "function": new_attempt.name,
            "line": int(attempt_calls[0].lineno),
            "call_count": len(attempt_calls),
            "verify_full_archives": False,
        },
        "whole_record_inequality": {
            "function": new_attempt.name,
            "lines": comparison_lines,
            "comparison_count": failed_archive_comparisons,
            "field": "failed_archives",
        },
        "verification_strength_return_field": {
            "function": archive_validator.name,
            "lines": assignment_lines,
            "assignment_count": assignments,
            "field": "full_inventory_verified",
            "source_parameter": "verify_full_inventory",
        },
        "detailed_validator_return_key_sets": {
            "first_archive_function": first_archive_validator.name,
            "first_archive": first_detailed_keys,
            "second_archive_function": archive_validator.name,
            "second_archive": second_detailed_keys,
        },
        "per_row_key_differences": per_row_key_differences,
        "defect_id": (
            "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
        ),
        "full_inventory_verified_is_one_detail_not_sole_cause": True,
        "original_isolated_child_launch_semantics": {
            "runner_function": isolated_runner.name,
            "popen_line": int(popen_calls[0].lineno),
            "cwd": "ROOT",
            "env": (
                "None when runner environment argument is None; otherwise a copy"
            ),
            "scientific_call_environment_keyword_present": False,
            "scientific_child_environment": "INHERITED_FROM_LAUNCHER",
            "stdin": "UNSPECIFIED_SUBPROCESS_DEFAULT_INHERITED",
            "stdout": "subprocess.PIPE",
            "stderr": "subprocess.STDOUT",
            "close_fds": "UNSPECIFIED_SUBPROCESS_DEFAULT_TRUE",
            "text": True,
            "start_new_session": True,
            "child_session_and_process_group_equal_pid": True,
        },
    }
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "committed_source_root_cause_ast_proof.v1"
            ),
            "commit": SOURCE_COMMIT,
            "bound_sources": bound_sources,
            "evidence": evidence,
            "archive_payloads_opened": 0,
            "outcome_metric_or_tensor_values_opened": 0,
            "defect_proved": True,
            "pass": True,
        }
    )


def build_attempt_accounting_policy_markdown() -> str:
    rules = "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(
            ATTEMPT_ACCOUNTING_POLICY["mandatory_rules"], start=1
        )
    )
    return "\n".join(
        [
            "# Experiment attempt accounting policy V2",
            "",
            f"Date: {DATE}",
            "",
            "## Purpose",
            "",
            (
                "This policy separates technical startup, scientific execution, "
                "and publication so a process-control failure is not silently "
                "counted as a scientific model attempt. It does not authorize an "
                "additional experiment."
            ),
            "",
            "## Attempt classes",
            "",
            "- `technical_startup_attempt`: "
            + ATTEMPT_ACCOUNTING_POLICY["attempt_classes"][
                "technical_startup_attempt"
            ],
            "- `scientific_attempt`: "
            + ATTEMPT_ACCOUNTING_POLICY["attempt_classes"]["scientific_attempt"],
            "- `publication_attempt`: "
            + ATTEMPT_ACCOUNTING_POLICY["attempt_classes"]["publication_attempt"],
            "",
            "## Mandatory rules",
            "",
            rules,
            "",
            ATTEMPT_ACCOUNTING_POLICY["nonretroactivity"],
            "",
            "## Durable reservation boundary",
            "",
            "- Technical: "
            + ATTEMPT_ACCOUNTING_POLICY[
                "class_specific_recording_and_consumption"
            ]["technical_startup_attempt"],
            "- Scientific: "
            + ATTEMPT_ACCOUNTING_POLICY[
                "class_specific_recording_and_consumption"
            ]["scientific_attempt"],
            "- Publication: "
            + ATTEMPT_ACCOUNTING_POLICY[
                "class_specific_recording_and_consumption"
            ]["publication_attempt"],
            "",
            (
                "A diagnostic invocation, syntactic rejection, child process start, "
                "or technical incident archive is not by itself evidence that a "
                "scientific attempt was consumed. Missing reservation evidence is "
                "handled fail-closed."
            ),
            "",
            "## Current incident",
            "",
            (
                "The third incident has exact technical receipts, no attempt or "
                "reservation namespace, and zero scientific counters. It is "
                "accounted as a technical-startup incident, not a scientific "
                "attempt. This incident accounting alone authorizes neither a "
                "V2 specification nor V2 execution. A V2 specification requires "
                "the complete frozen forensic gate and the bound current-user "
                "specification-only authority; V2 execution and a scientific "
                "attempt remain unauthorized."
            ),
            "",
            "## Authority boundary",
            "",
            (
                f"The scientific contract remains frozen at "
                f"`{BASE.INITIAL_EXECUTION_FREEZE_COMMIT}` with digest "
                f"`{BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256}`. The present "
                "forensic overlay changes no model, input, target, metric, gate, "
                "seed, safety scope, or scientific decision."
            ),
            "",
        ]
    )


ATTEMPT_ACCOUNTING_POLICY_BYTES = (
    build_attempt_accounting_policy_markdown().encode("utf-8")
)
ATTEMPT_ACCOUNTING_POLICY_BINDING = _artifact_binding(
    str(TRACKED_ATTEMPT_ACCOUNTING_POLICY_PATH),
    ATTEMPT_ACCOUNTING_POLICY_BYTES,
)


def build_os_evidence_schema() -> dict[str, Any]:
    return attach_self_digest(
        {
            "schema": OS_EVIDENCE_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
            "runtime_paths": copy.deepcopy(PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS),
            "diagnostic_timeout_policy": copy.deepcopy(
                PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY
            ),
            "process_identity_fields": sorted(PROCESS_IDENTITY_FIELDS),
            "invocation_receipt": {
                "schema": INVOCATION_RECEIPT_SCHEMA,
                "required_fields": [
                    "schema",
                    "experiment_id",
                    "mode",
                    "source_commit",
                    "forensic_contract",
                    "diagnostic_root",
                    "launcher_process_identity",
                    "child_process_identity",
                    "outer_exact_argv",
                    "internal_exact_argv",
                    "environment",
                    "fd_custody",
                    "technical_runtime_context",
                    "started_monotonic_ns",
                    "namespace_before",
                    "attempt_namespace_created",
                    "attempt_reservation_created",
                    "content_digest",
                ],
            },
            "os_evidence_receipt": {
                "required_fields": [
                    "schema",
                    "experiment_id",
                    "launcher_process_identity",
                    "child_process_identity",
                    "started_monotonic_ns",
                    "ended_monotonic_ns",
                    "runtime_ns",
                    "returncode",
                    "termination",
                    "cleanup",
                    "namespace_before",
                    "namespace_after",
                    "namespace_unchanged",
                    "current_runtime_context",
                    "pass",
                    "content_digest",
                ],
                "termination": {
                    "kind": ["EXIT", "SIGNAL", "TIMEOUT"],
                    "fields": [
                        "kind",
                        "exit_code_or_null",
                        "signal_number_or_null",
                        "signal_name_or_null",
                    ],
                },
                "cleanup_required_exact_empty": [
                    "process_group_members_after_wait",
                    "exact_nonlauncher_forensic_role_matches_after_wait",
                    "scoped_dev_kfd_holders_after_wait",
                ],
                "cleanup_scope": (
                    "TERMINATED_CHILD_PROCESS_GROUP_AND_NONLAUNCHER_"
                    "FORENSIC_ROLES"
                ),
                "current_launcher_excluded_and_disclosed_live": True,
                "literal_zero_all_forensic_roles_claimed": False,
            },
            "external_stream_custody": {
                "preopened_before_child": [
                    "child_stdout",
                    "child_stderr",
                    "child_traceback",
                    "child_exception",
                    "heartbeat",
                    "read_guard_events",
                ],
                "distinct_paths": [
                    "child_stdout",
                    "child_stderr",
                    "child_traceback",
                    "child_exception",
                    "heartbeat",
                    "read_guard_events",
                ],
                "exclusive_creation": True,
                "fsync_before_child": True,
                    "wrapper_fds": [
                        "traceback",
                        "exception",
                        "heartbeat",
                        "read_guard_events",
                    ],
                    "read_guard_installed_before_evaluator_runpy_import": True,
            },
            "diagnostic_custody_receipt": {
                "schema": DIAGNOSTIC_CUSTODY_SCHEMA,
                "pass_means": "TECHNICAL_CUSTODY_COMPLETE_NOT_SCIENTIFIC_SUCCESS",
                "required_zero_counters": [
                    "scientific_inputs_opened",
                    "outcome_rows_opened",
                    "tensor_reads",
                    "model_loads",
                    "training_steps",
                    "files_reused",
                ],
                "required_false": [
                    "attempt_namespace_created",
                    "attempt_reservation_created",
                    "canonical_or_tracked_written",
                ],
            },
            "startup_stage_ledger": {
                "schema": STARTUP_STAGE_ROW_SCHEMA,
                "stage_ids": list(PREEXECUTION_DIAGNOSTIC_STAGE_IDS),
                "events_per_stage": ["STARTED", "COMPLETED"],
                "complete_rows": 2 * len(PREEXECUTION_DIAGNOSTIC_STAGE_IDS),
                "append_only_and_fsynced": True,
            },
            "claims_boundary": {
                "scientific_archive_payloads": "NEVER_OPENED_BY_DIAGNOSTIC",
                "admitted_technical_archive_receipts": [
                    row["path"] for row in THIRD_FAILURE_ARCHIVE_INVENTORY["rows"]
                ],
                "scientific_inputs": "NEVER_OPENED_BY_DIAGNOSTIC",
                "technical_receipts": "BOUND_SEPARATELY",
            },
        }
    )


OS_EVIDENCE_SCHEMA_AUTHORITY = build_os_evidence_schema()
OS_EVIDENCE_SCHEMA_BYTES = canonical_json_bytes(OS_EVIDENCE_SCHEMA_AUTHORITY) + b"\n"
OS_EVIDENCE_SCHEMA_BINDING = _artifact_binding(
    str(TRACKED_OS_EVIDENCE_SCHEMA_PATH),
    OS_EVIDENCE_SCHEMA_BYTES,
    content_digest=OS_EVIDENCE_SCHEMA_AUTHORITY["content_digest"],
)


def validate_os_evidence_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    if dict(value) != OS_EVIDENCE_SCHEMA_AUTHORITY:
        raise ForensicContractError("OS-evidence schema drift")
    return copy.deepcopy(dict(value))


USER_CONDITIONAL_V2_SPEC_AUTHORITY = attach_self_digest(
    {
        "schema": (
            "plan_aware_monotone_jepa_cost_v1."
            "conditional_v2_spec_user_authority.v1"
        ),
        "source": "CURRENT_USER_INSTRUCTION",
        "instruction": (
            "Create and commit a V2 specification if and only if every frozen "
            "technical gate passes; this does not authorize execution or a "
            "scientific attempt."
        ),
        "scope": "SPECIFY_V2_ONLY_IF_ALL_TECHNICAL_GATES_PASS",
        "specification_authorized_conditionally": True,
        "execution_authorized": False,
        "scientific_attempt_authorized": False,
        "automatic_execution": False,
    }
)


def build_conditional_v2_gate_authority() -> dict[str, Any]:
    return attach_self_digest(
        {
            "schema": CONDITIONAL_V2_GATE_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "authorization_field": "v2_spec_authorized",
            "authorization_type": "boolean",
            "not_a_classification": True,
            "required_conditions": list(CONDITIONAL_V2_REQUIRED_CONDITIONS),
            "eligible_root_cause_classifications": [
                "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED",
                "FINAL_CHILD_LAUNCHER_PATH_DEFECT_IDENTIFIED",
                "FINAL_CHILD_ENVIRONMENT_OR_RESOURCE_FAILURE_IDENTIFIED",
            ],
            "ineligible_root_cause_classifications": [
                "TRACEBACK_CUSTODY_DEFECT_CONFIRMED_ROOT_CAUSE_UNRESOLVED",
                "FAILURE_FORENSICS_INCONCLUSIVE",
            ],
            "all_conditions_must_be_literal_true": True,
            "automatic_execution": False,
            "existing_no_further_retry_authority_overridden_by_gate_alone": False,
            "explicit_human_or_stakeholder_authorization_required": True,
            "bound_user_specification_authority": copy.deepcopy(
                USER_CONDITIONAL_V2_SPEC_AUTHORITY
            ),
            "scientific_contract_change_allowed": False,
        }
    )


CONDITIONAL_V2_GATE_AUTHORITY = build_conditional_v2_gate_authority()
CONDITIONAL_V2_GATE_BYTES = canonical_json_bytes(CONDITIONAL_V2_GATE_AUTHORITY) + b"\n"
CONDITIONAL_V2_GATE_BINDING = _artifact_binding(
    str(TRACKED_CONDITIONAL_V2_GATE_PATH),
    CONDITIONAL_V2_GATE_BYTES,
    content_digest=CONDITIONAL_V2_GATE_AUTHORITY["content_digest"],
)


def evaluate_conditional_v2_gate(
    *, root_cause_classification: str, conditions: Mapping[str, Any]
) -> dict[str, Any]:
    if root_cause_classification not in ROOT_CAUSE_CLASSIFICATIONS:
        raise ForensicContractError("unknown forensic root-cause classification")
    if set(conditions) != set(CONDITIONAL_V2_REQUIRED_CONDITIONS):
        raise ForensicContractError("conditional V2 condition key-set drift")
    normalized = {
        key: value if isinstance(value, bool) else False
        for key, value in conditions.items()
    }
    eligible = root_cause_classification in set(
        CONDITIONAL_V2_GATE_AUTHORITY["eligible_root_cause_classifications"]
    )
    passed = eligible and all(
        normalized[key] is True for key in CONDITIONAL_V2_REQUIRED_CONDITIONS
    )
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.conditional_v2_gate_result.v1",
            "root_cause_classification": root_cause_classification,
            "conditions": {
                key: normalized[key] for key in CONDITIONAL_V2_REQUIRED_CONDITIONS
            },
            "eligible_root_cause": eligible,
            "v2_spec_authorized": passed,
            "automatic_execution": False,
        }
    )


def build_forensic_evaluator_fixture() -> dict[str, Any]:
    return attach_self_digest(
        {
            "schema": FORENSIC_FIXTURE_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "forensic_contract_sha256": FORENSIC_CONTRACT_SHA256,
            "authority_bindings": {
                "contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
                "archive_inventory": copy.deepcopy(
                    ARCHIVE_INVENTORY_AUTHORITY_BINDING
                ),
                "os_evidence_schema": copy.deepcopy(OS_EVIDENCE_SCHEMA_BINDING),
                "conditional_v2_gate": copy.deepcopy(
                    CONDITIONAL_V2_GATE_BINDING
                ),
                "attempt_accounting_policy": copy.deepcopy(
                    ATTEMPT_ACCOUNTING_POLICY_BINDING
                ),
            },
            "runtime_paths": copy.deepcopy(PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS),
            "stage_ids": list(PREEXECUTION_DIAGNOSTIC_STAGE_IDS),
            "synthetic_stage_ids": list(
                PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS
            ),
            "synthetic_fixtures": list(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES),
            "diagnostic_timeout_policy": copy.deepcopy(
                PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY
            ),
            "root_cause_fixture": build_archive_verification_record_equality_fixture(),
            "committed_source_proof": copy.deepcopy(
                COMMITTED_SOURCE_ROOT_CAUSE_BINDINGS
            ),
            "expected_zero_scientific_counters": {
                "scientific_inputs_opened": 0,
                "outcome_rows_opened": 0,
                "tensor_reads": 0,
                "model_loads": 0,
                "training_steps": 0,
                "files_reused": 0,
            },
        }
    )


FORENSIC_EVALUATOR_FIXTURE = build_forensic_evaluator_fixture()
FORENSIC_EVALUATOR_FIXTURE_BYTES = canonical_json_bytes(
    FORENSIC_EVALUATOR_FIXTURE
) + b"\n"
FORENSIC_EVALUATOR_FIXTURE_BINDING = _artifact_binding(
    str(TRACKED_FORENSIC_FIXTURE_PATH),
    FORENSIC_EVALUATOR_FIXTURE_BYTES,
    content_digest=FORENSIC_EVALUATOR_FIXTURE["content_digest"],
)


def _validate_monotonic_ns(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForensicContractError(f"{label} must be a positive integer")
    return value


def _validate_namespace_snapshot(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ForensicContractError("namespace snapshot must be a list")
    output: list[dict[str, str]] = []
    previous = ""
    for row in value:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "kind"}
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or row["kind"] not in {"FILE", "DIRECTORY", "SYMLINK", "OTHER"}
            or row["path"] <= previous
        ):
            raise ForensicContractError("namespace snapshot drift")
        previous = row["path"]
        output.append(dict(row))
    return output


def build_environment_receipt(
    *,
    inherited_python_keys_removed: Sequence[str],
    inherited_environment_key_names: Sequence[str],
    result_environment: Mapping[str, str],
    virtual_env: str,
    path_prepend: str,
) -> dict[str, Any]:
    removed = sorted(str(key) for key in inherited_python_keys_removed)
    inherited_names = sorted(str(key) for key in inherited_environment_key_names)
    result_map = {str(key): str(value) for key, value in result_environment.items()}
    result_names = sorted(result_map)
    expected_names = sorted(
        (set(inherited_names) - {key for key in inherited_names if key.startswith("PYTHON")})
        | {
            "PYTHONNOUSERSITE",
            "PYTHONUNBUFFERED",
            "PYTHONFAULTHANDLER",
            "VIRTUAL_ENV",
            "PATH",
        }
    )
    if (
        len(removed) != len(set(removed))
        or any(not key.startswith("PYTHON") for key in removed)
        or inherited_names != sorted(set(inherited_names))
        or removed != sorted(key for key in inherited_names if key.startswith("PYTHON"))
        or result_names != expected_names
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in result_environment.items())
        or result_map.get("PYTHONNOUSERSITE") != "1"
        or result_map.get("PYTHONUNBUFFERED") != "1"
        or result_map.get("PYTHONFAULTHANDLER") != "1"
        or result_map.get("VIRTUAL_ENV") != virtual_env
        or not result_map.get("PATH", "").startswith(path_prepend)
        or not Path(virtual_env).is_absolute()
        or not Path(path_prepend).is_absolute()
    ):
        raise ForensicContractError("diagnostic environment input drift")
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_environment.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "inherited_python_keys_removed": removed,
            "inherited_environment_key_names": inherited_names,
            "inherited_python_keys_remaining": [],
            "set_python_environment": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1",
            },
            "interpreter_flags": list(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS),
            "virtual_env": virtual_env,
            "path_prepend": path_prepend,
            "result_environment_custody": {
                "key_names": result_names,
                "key_count": len(result_names),
                "canonical_map_sha256": canonical_json_sha256(result_map),
                "path_sha256": hashlib.sha256(
                    result_map["PATH"].encode("utf-8")
                ).hexdigest(),
                "path_bytes": len(result_map["PATH"].encode("utf-8")),
                "values_persisted": False,
                "popen_map_source": "EXACT_SANITIZED_RESULT_ENVIRONMENT",
            },
            "pass": True,
        }
    )


def validate_environment_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "experiment_id",
        "inherited_python_keys_removed",
        "inherited_environment_key_names",
        "inherited_python_keys_remaining",
        "set_python_environment",
        "interpreter_flags",
        "virtual_env",
        "path_prepend",
        "result_environment_custody",
        "pass",
        "content_digest",
    }
    removed = value.get("inherited_python_keys_removed")
    inherited_names = value.get("inherited_environment_key_names")
    custody = value.get("result_environment_custody")
    if (
        set(value) != required
        or value.get("schema")
        != "plan_aware_monotone_jepa_cost_v1.preexecution_environment.v1"
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or not isinstance(removed, list)
        or removed != sorted(set(removed))
        or any(not isinstance(key, str) or not key.startswith("PYTHON") for key in removed)
        or not isinstance(inherited_names, list)
        or inherited_names != sorted(set(inherited_names))
        or removed != sorted(key for key in inherited_names if key.startswith("PYTHON"))
        or value.get("inherited_python_keys_remaining") != []
        or value.get("set_python_environment")
        != {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
        }
        or value.get("interpreter_flags")
        != list(PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_FLAGS)
        or not isinstance(value.get("virtual_env"), str)
        or not Path(value["virtual_env"]).is_absolute()
        or not isinstance(value.get("path_prepend"), str)
        or not Path(value["path_prepend"]).is_absolute()
        or not isinstance(custody, Mapping)
        or set(custody)
        != {
            "key_names",
            "key_count",
            "canonical_map_sha256",
            "path_sha256",
            "path_bytes",
            "values_persisted",
            "popen_map_source",
        }
        or custody.get("key_names") != sorted(set(custody.get("key_names", [])))
        or custody.get("key_count") != len(custody.get("key_names", []))
        or not isinstance(custody.get("canonical_map_sha256"), str)
        or len(custody["canonical_map_sha256"]) != 64
        or not isinstance(custody.get("path_sha256"), str)
        or len(custody["path_sha256"]) != 64
        or not isinstance(custody.get("path_bytes"), int)
        or custody["path_bytes"] <= 0
        or custody.get("values_persisted") is not False
        or custody.get("popen_map_source")
        != "EXACT_SANITIZED_RESULT_ENVIRONMENT"
        or value.get("pass") is not True
    ):
        raise ForensicContractError("diagnostic environment receipt drift")
    return copy.deepcopy(dict(value))


def build_diagnostic_umask_custody(
    *,
    previous_umask: int,
    restored_umask: int,
    set_before_root_creation: bool,
    inherited_by_all_children: bool,
    restoration_verified: bool,
) -> dict[str, Any]:
    value = attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "preexecution_diagnostic_umask_custody.v1"
            ),
            "previous_umask": previous_umask,
            "effective_diagnostic_umask": PREEXECUTION_DIAGNOSTIC_UMASK,
            "set_before_root_creation": set_before_root_creation,
            "set_before_any_diagnostic_artifact": set_before_root_creation,
            "inherited_by_all_children": inherited_by_all_children,
            "restored_umask": restored_umask,
            "restoration_verified": restoration_verified,
            "receipt_written_after_restoration": True,
            "pass": True,
        }
    )
    return validate_diagnostic_umask_custody(value)


def validate_diagnostic_umask_custody(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    if (
        set(value)
        != {
            "schema",
            "previous_umask",
            "effective_diagnostic_umask",
            "set_before_root_creation",
            "set_before_any_diagnostic_artifact",
            "inherited_by_all_children",
            "restored_umask",
            "restoration_verified",
            "receipt_written_after_restoration",
            "pass",
            "content_digest",
        }
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "preexecution_diagnostic_umask_custody.v1"
        )
        or value.get("previous_umask")
        != PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK
        or value.get("effective_diagnostic_umask")
        != PREEXECUTION_DIAGNOSTIC_UMASK
        or value.get("restored_umask")
        != PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK
        or any(
            value.get(key) is not True
            for key in (
                "set_before_root_creation",
                "set_before_any_diagnostic_artifact",
                "inherited_by_all_children",
                "restoration_verified",
                "receipt_written_after_restoration",
                "pass",
            )
        )
    ):
        raise ForensicContractError("diagnostic umask custody drift")
    return copy.deepcopy(dict(value))


def _validate_fd_custody(value: Mapping[str, Any], diagnostic_root: Path) -> dict[str, Any]:
    expected_names = {
        "traceback_fd",
        "exception_fd",
        "heartbeat_fd",
        "read_guard_events_fd",
        "paths",
        "pass",
    }
    if set(value) != expected_names or value.get("pass") is not True:
        raise ForensicContractError("diagnostic FD custody key-set drift")
    fd_keys = (
        "traceback_fd",
        "exception_fd",
        "heartbeat_fd",
        "read_guard_events_fd",
    )
    for key in fd_keys:
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 3:
            raise ForensicContractError("diagnostic inherited FD drift")
    if len({value[key] for key in fd_keys}) != len(fd_keys):
        raise ForensicContractError("diagnostic inherited FDs are not distinct")
    paths = value.get("paths")
    expected_paths = {
        "traceback": str(
            (diagnostic_root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["child_traceback"]).absolute()
        ),
        "exception": str(
            (diagnostic_root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["child_exception"]).absolute()
        ),
        "heartbeat": str(
            (diagnostic_root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["heartbeat"]).absolute()
        ),
        "read_guard_events": str(
            (
                diagnostic_root
                / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["read_guard_events"]
            ).absolute()
        ),
    }
    if paths != expected_paths:
        raise ForensicContractError("diagnostic inherited FD path drift")
    return copy.deepcopy(dict(value))


def scientific_forbidden_bindings(repo_root: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    bindings = {
        "generated_root": str((root / ".generated").resolve(strict=False)),
        "predecessor_root": str(PREDECESSOR_SCIENTIFIC_ROOT.resolve(strict=False)),
        "scientific_output_root": str(BASE.OUTPUT_ROOT.resolve(strict=False)),
        "failed_archive_1": str(
            BASE.EXECUTION_CORRECTION_FAILED_ARCHIVE.resolve(strict=False)
        ),
        "failed_archive_2": str(
            BASE.EXECUTION_CORRECTION_2_FAILED_ARCHIVE.resolve(strict=False)
        ),
        "failed_archive_3": str(THIRD_FAILURE_ARCHIVE.resolve(strict=False)),
        "tracked_result": str(
            (root / BASE.TRACKED_RESULT_PATH).resolve(strict=False)
        ),
        "tracked_report": str(
            (root / BASE.TRACKED_REPORT_PATH).resolve(strict=False)
        ),
        "encoder": str(
            Path(str(BASE.ENCODER_BINDING["path"])).resolve(strict=False)
        ),
    }
    for label, record in BASE.CHECKPOINT_BINDINGS.items():
        bindings[f"checkpoint:{label}"] = str(
            Path(str(record["path"])).resolve(strict=False)
        )
    for label, record in BASE.PANEL_BINDINGS.items():
        bindings[f"panel:{label}"] = str(
            (root / str(record["path"])).resolve(strict=False)
        )
    for label, record in BASE.PREDECESSOR_RESULT_BINDINGS.items():
        if isinstance(record, Mapping) and isinstance(record.get("path"), str):
            bindings[f"predecessor_result:{label}"] = str(
                (root / str(record["path"])).resolve(strict=False)
            )
    predecessor_candidate = BASE.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
    bindings["predecessor_candidate_evidence"] = str(
        Path(str(predecessor_candidate["path"])).resolve(strict=False)
    )
    bindings["predecessor_route_role_authority"] = str(
        Path(str(BASE.ROUTE_ROLE_AUTHORITY_BINDING["path"])).resolve(strict=False)
    )
    for label, record in BASE.PROPRIO_INPUT_BINDINGS.items():
        if isinstance(record, Mapping) and isinstance(record.get("path"), str):
            path = Path(str(record["path"]))
            bindings[f"proprio_input:{label}"] = str(
                (path if path.is_absolute() else root / path).resolve(strict=False)
            )
    return dict(sorted(bindings.items()))


def build_read_guard_manifest(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    scientific_bindings = scientific_forbidden_bindings(root)
    forbidden = set(scientific_bindings.values())
    admitted = sorted(
        str(
            (
                THIRD_FAILURE_ARCHIVE
                / str(record["path"])
            ).resolve(strict=False)
        )
        for record in THIRD_FAILURE_ARCHIVE_INVENTORY["rows"]
    )
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_read_guard.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "forbidden_path_prefixes": sorted(forbidden),
            "scientific_forbidden_bindings": scientific_bindings,
            "admitted_exact_read_only_technical_receipts": admitted,
            "admission_precedence": (
                "EXACT_ADMITTED_PATH_CHECK_BEFORE_FORBIDDEN_PREFIX_CHECK"
            ),
            "admitted_open_modes": ["r", "rb"],
            "admitted_write_create_truncate_append_or_update": False,
            "admission_does_not_apply_to_parent_or_sibling_paths": True,
            "installed_by_stdlib_wrapper_before_evaluator_runpy": True,
            "forbidden_open_event_action": (
                "APPEND_CANONICAL_EVENT_FSYNC_THEN_RAISE_PERMISSION_ERROR"
            ),
            "expected_event_rows": 0,
        }
    )


def validate_read_guard_manifest(
    value: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    validate_self_digest(value)
    if dict(value) != build_read_guard_manifest(repo_root):
        raise ForensicContractError("preexecution read-guard manifest drift")
    return copy.deepcopy(dict(value))


def build_command_receipt(
    *,
    diagnostic_root: str | Path,
    outer_exact_argv: Sequence[str],
    internal_exact_argv: Sequence[str],
    fd_custody: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    cwd: str | Path,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    outer = [str(item) for item in outer_exact_argv]
    internal = [str(item) for item in internal_exact_argv]
    if not outer or not internal or not Path(cwd).is_absolute():
        raise ForensicContractError("diagnostic command input drift")
    validated_fds = _validate_fd_custody(fd_custody, root)
    environment = validate_environment_receipt(environment_receipt)
    inherited_fds = [
        int(validated_fds[key])
        for key in (
            "traceback_fd",
            "exception_fd",
            "heartbeat_fd",
            "read_guard_events_fd",
        )
    ]
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_command.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "diagnostic_root": str(root),
            "outer_exact_argv": outer,
            "outer_argv_sha256": canonical_json_sha256(outer),
            "internal_exact_argv": internal,
            "internal_argv_sha256": canonical_json_sha256(internal),
            "fd_custody": copy.deepcopy(dict(fd_custody)),
            "popen_environment": copy.deepcopy(
                environment["result_environment_custody"]
            ),
            "cwd": str(cwd),
            "popen_semantics": {
                "stdin": {"destination": "DEVNULL", "child_fd": 0},
                "stdout": {
                    "destination": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                        "child_stdout"
                    ],
                    "child_fd": 1,
                },
                "stderr": {
                    "destination": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                        "child_stderr"
                    ],
                    "child_fd": 2,
                },
                "close_fds": True,
                "pass_fds": inherited_fds,
                "start_new_session": True,
                "child_session_and_process_group_leader": True,
                "shell": False,
            },
            "timeout_policy": copy.deepcopy(
                PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY
            ),
            "pass": True,
        }
    )


def validate_command_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "experiment_id",
        "diagnostic_root",
        "outer_exact_argv",
        "outer_argv_sha256",
        "internal_exact_argv",
        "internal_argv_sha256",
        "fd_custody",
        "popen_environment",
        "cwd",
        "popen_semantics",
        "timeout_policy",
        "pass",
        "content_digest",
    }
    if set(value) != required:
        raise ForensicContractError("diagnostic command receipt key-set drift")
    root = Path(str(value.get("diagnostic_root")))
    outer = value.get("outer_exact_argv")
    internal = value.get("internal_exact_argv")
    validated_fds = _validate_fd_custody(value["fd_custody"], root)
    expected_pass_fds = [
        int(validated_fds[key])
        for key in (
            "traceback_fd",
            "exception_fd",
            "heartbeat_fd",
            "read_guard_events_fd",
        )
    ]
    if (
        value.get("schema")
        != "plan_aware_monotone_jepa_cost_v1.preexecution_command.v1"
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or not root.is_absolute()
        or not isinstance(outer, list)
        or not outer
        or any(not isinstance(item, str) or not item for item in outer)
        or value.get("outer_argv_sha256") != canonical_json_sha256(outer)
        or not isinstance(internal, list)
        or not internal
        or any(not isinstance(item, str) or not item for item in internal)
        or value.get("internal_argv_sha256") != canonical_json_sha256(internal)
        or not isinstance(value.get("cwd"), str)
        or not Path(value["cwd"]).is_absolute()
        or value.get("pass") is not True
        or value.get("timeout_policy")
        != PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY
        or value.get("popen_semantics")
        != {
            "stdin": {"destination": "DEVNULL", "child_fd": 0},
            "stdout": {
                "destination": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                    "child_stdout"
                ],
                "child_fd": 1,
            },
            "stderr": {
                "destination": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                    "child_stderr"
                ],
                "child_fd": 2,
            },
            "close_fds": True,
            "pass_fds": expected_pass_fds,
            "start_new_session": True,
            "child_session_and_process_group_leader": True,
            "shell": False,
        }
    ):
        raise ForensicContractError("diagnostic command receipt drift")
    custody = value.get("popen_environment")
    if (
        not isinstance(custody, Mapping)
        or set(custody)
        != {
            "key_names",
            "key_count",
            "canonical_map_sha256",
            "path_sha256",
            "path_bytes",
            "values_persisted",
            "popen_map_source",
        }
    ):
        raise ForensicContractError("diagnostic Popen environment custody drift")
    return copy.deepcopy(dict(value))


def build_invocation_receipt(
    *,
    source_commit: str,
    diagnostic_root: str | Path,
    launcher_process_identity: Mapping[str, Any],
    child_process_identity: Mapping[str, Any],
    outer_exact_argv: Sequence[str],
    internal_exact_argv: Sequence[str],
    environment: Mapping[str, Any],
    fd_custody: Mapping[str, Any],
    technical_runtime_context: Mapping[str, Any],
    started_monotonic_ns: int,
    namespace_before: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    launcher = validate_process_identity(launcher_process_identity)
    child = validate_process_identity(child_process_identity)
    validate_environment_receipt(environment)
    _validate_fd_custody(fd_custody, root)
    before = _validate_namespace_snapshot(list(namespace_before))
    _validate_monotonic_ns(started_monotonic_ns, "invocation start")
    outer = [str(item) for item in outer_exact_argv]
    internal = [str(item) for item in internal_exact_argv]
    if child["argv"] != outer:
        raise ForensicContractError("child identity does not bind outer argv")
    return attach_self_digest(
        {
            "schema": INVOCATION_RECEIPT_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
            "source_commit": source_commit,
            "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
            "diagnostic_root": str(root),
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "outer_exact_argv": outer,
            "internal_exact_argv": internal,
            "environment": copy.deepcopy(dict(environment)),
            "fd_custody": copy.deepcopy(dict(fd_custody)),
            "technical_runtime_context": validate_current_runtime_context(
                technical_runtime_context
            ),
            "started_monotonic_ns": started_monotonic_ns,
            "namespace_before": before,
            "attempt_namespace_created": False,
            "attempt_reservation_created": False,
        }
    )


def validate_invocation_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "experiment_id",
        "mode",
        "source_commit",
        "forensic_contract",
        "diagnostic_root",
        "launcher_process_identity",
        "child_process_identity",
        "outer_exact_argv",
        "internal_exact_argv",
        "environment",
        "fd_custody",
        "technical_runtime_context",
        "started_monotonic_ns",
        "namespace_before",
        "attempt_namespace_created",
        "attempt_reservation_created",
        "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema") != INVOCATION_RECEIPT_SCHEMA
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or value.get("mode") != "PREEXECUTION_ONLY_DIAGNOSTIC"
        or not isinstance(value.get("source_commit"), str)
        or len(value["source_commit"]) != 40
        or value.get("forensic_contract") != FORENSIC_CONTRACT_BINDING
        or value.get("attempt_namespace_created") is not False
        or value.get("attempt_reservation_created") is not False
    ):
        raise ForensicContractError("diagnostic invocation receipt drift")
    root = Path(str(value.get("diagnostic_root")))
    if not root.is_absolute():
        raise ForensicContractError("diagnostic invocation root drift")
    launcher = validate_process_identity(value["launcher_process_identity"])
    child = validate_process_identity(value["child_process_identity"])
    if child["argv"] != value.get("outer_exact_argv"):
        raise ForensicContractError("diagnostic outer argv identity drift")
    if not isinstance(value.get("internal_exact_argv"), list):
        raise ForensicContractError("diagnostic internal argv drift")
    validate_environment_receipt(value["environment"])
    validate_current_runtime_context(value["technical_runtime_context"])
    _validate_fd_custody(value["fd_custody"], root)
    _validate_monotonic_ns(value.get("started_monotonic_ns"), "invocation start")
    _validate_namespace_snapshot(value.get("namespace_before"))
    del launcher
    return copy.deepcopy(dict(value))


def _validate_termination(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "kind",
        "exit_code_or_null",
        "signal_number_or_null",
        "signal_name_or_null",
    }
    if set(value) != required or value.get("kind") not in {"EXIT", "SIGNAL", "TIMEOUT"}:
        raise ForensicContractError("diagnostic termination key-set drift")
    if value["kind"] == "EXIT":
        if (
            isinstance(value.get("exit_code_or_null"), bool)
            or not isinstance(value.get("exit_code_or_null"), int)
            or value.get("signal_number_or_null") is not None
            or value.get("signal_name_or_null") is not None
        ):
            raise ForensicContractError("diagnostic exit termination drift")
    else:
        if (
            value.get("exit_code_or_null") is not None
            or isinstance(value.get("signal_number_or_null"), bool)
            or not isinstance(value.get("signal_number_or_null"), int)
            or value["signal_number_or_null"] <= 0
            or not isinstance(value.get("signal_name_or_null"), str)
            or not value["signal_name_or_null"]
            or (
                value["kind"] == "TIMEOUT"
                and value["signal_number_or_null"]
                not in {int(signal.SIGTERM), int(signal.SIGKILL)}
            )
        ):
            raise ForensicContractError("diagnostic signal termination drift")
    return copy.deepcopy(dict(value))


def _validate_cleanup(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "process_group_members_after_wait",
        "exact_nonlauncher_forensic_role_matches_after_wait",
        "scoped_dev_kfd_holders_after_wait",
        "current_launcher_excluded_from_role_scan",
        "cleanup_scope",
        "literal_zero_all_forensic_roles_claimed",
        "pass",
    }
    if (
        set(value) != required
        or value.get("process_group_members_after_wait") != []
        or value.get("exact_nonlauncher_forensic_role_matches_after_wait") != []
        or value.get("scoped_dev_kfd_holders_after_wait") != []
        or value.get("current_launcher_excluded_from_role_scan") is not True
        or value.get("cleanup_scope")
        != "TERMINATED_CHILD_PROCESS_GROUP_AND_NONLAUNCHER_FORENSIC_ROLES"
        or value.get("literal_zero_all_forensic_roles_claimed") is not False
        or value.get("pass") is not True
    ):
        raise ForensicContractError("diagnostic cleanup drift")
    return copy.deepcopy(dict(value))


def _validate_stream_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "stdout",
        "stderr",
        "traceback",
        "exception",
        "heartbeat",
        "read_guard_events",
    }
    if set(value) != expected:
        raise ForensicContractError("diagnostic stream-binding key-set drift")
    paths: list[str] = []
    output: dict[str, Any] = {}
    for label in sorted(expected):
        row = validate_artifact_binding(value[label])
        paths.append(row["path"])
        output[label] = row
    if len(paths) != len(set(paths)):
        raise ForensicContractError("diagnostic external streams are not distinct")
    return output


def build_current_runtime_context(
    *,
    captured_at_ns: int,
    identity: Mapping[str, Any],
    cwd: str,
    umask: Mapping[str, Any],
    rlimits: Mapping[str, Any],
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
    temp: Mapping[str, Any],
) -> dict[str, Any]:
    value = attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "current_technical_runtime_context.v1"
            ),
            "captured_at_ns": captured_at_ns,
            "identity": copy.deepcopy(dict(identity)),
            "cwd": cwd,
            "umask": copy.deepcopy(dict(umask)),
            "rlimits": copy.deepcopy(dict(rlimits)),
            "cpu": copy.deepcopy(dict(cpu)),
            "gpu": copy.deepcopy(dict(gpu)),
            "temp": copy.deepcopy(dict(temp)),
            "role": (
                "CURRENT_TECHNICAL_DIAGNOSTIC_ONLY_NOT_HISTORICAL_CHILD_CONTEXT"
            ),
        }
    )
    return validate_current_runtime_context(value)


def validate_current_runtime_context(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "captured_at_ns",
        "identity",
        "cwd",
        "umask",
        "rlimits",
        "cpu",
        "gpu",
        "temp",
        "role",
        "content_digest",
    }
    identity = value.get("identity")
    umask = value.get("umask")
    cpu = value.get("cpu")
    gpu = value.get("gpu")
    temp = value.get("temp")
    if (
        set(value) != required
        or value.get("schema")
        != "plan_aware_monotone_jepa_cost_v1.current_technical_runtime_context.v1"
        or isinstance(value.get("captured_at_ns"), bool)
        or not isinstance(value.get("captured_at_ns"), int)
        or value["captured_at_ns"] <= 0
        or not isinstance(identity, Mapping)
        or set(identity) != {"uid", "euid", "gid", "egid", "groups"}
        or any(
            isinstance(identity[key], bool) or not isinstance(identity[key], int)
            for key in ("uid", "euid", "gid", "egid")
        )
        or not isinstance(identity.get("groups"), list)
        or identity["groups"] != sorted(set(identity["groups"]))
        or any(not isinstance(item, int) or isinstance(item, bool) for item in identity["groups"])
        or not isinstance(value.get("cwd"), str)
        or not Path(value["cwd"]).is_absolute()
        or not isinstance(umask, Mapping)
        or set(umask) != {"value", "sampled_and_restored"}
        or not isinstance(umask.get("value"), int)
        or isinstance(umask.get("value"), bool)
        or not (0 <= umask["value"] <= 0o777)
        or umask.get("sampled_and_restored") is not True
        or not isinstance(value.get("rlimits"), Mapping)
        or not value["rlimits"]
        or not isinstance(cpu, Mapping)
        or set(cpu) != {"affinity", "cpu_count"}
        or not isinstance(cpu.get("affinity"), list)
        or cpu["affinity"] != sorted(set(cpu["affinity"]))
        or not isinstance(cpu.get("cpu_count"), int)
        or cpu["cpu_count"] <= 0
        or not isinstance(gpu, Mapping)
        or set(gpu)
        != {"visibility_environment", "device_metadata", "device_files_opened"}
        or set(gpu.get("visibility_environment", {}))
        != {
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
        }
        or not isinstance(gpu.get("device_metadata"), list)
        or gpu.get("device_files_opened") != 0
        or not isinstance(temp, Mapping)
        or set(temp) != {"environment", "path_metadata"}
        or set(temp.get("environment", {})) != {"TMPDIR", "TMP", "TEMP"}
        or not isinstance(temp.get("path_metadata"), list)
        or value.get("role")
        != "CURRENT_TECHNICAL_DIAGNOSTIC_ONLY_NOT_HISTORICAL_CHILD_CONTEXT"
    ):
        raise ForensicContractError("current technical runtime context drift")
    for name, row in value["rlimits"].items():
        if (
            not isinstance(name, str)
            or not isinstance(row, Mapping)
            or set(row) != {"soft", "hard"}
            or any(
                not (
                    isinstance(row[key], int)
                    and not isinstance(row[key], bool)
                    or row[key] == "INFINITY"
                )
                for key in ("soft", "hard")
            )
        ):
            raise ForensicContractError("current rlimit custody drift")
    for row in gpu["device_metadata"] + temp["path_metadata"]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"path", "exists", "kind", "mode_or_null", "uid_or_null", "gid_or_null", "writable_or_null"}
            or not isinstance(row.get("path"), str)
            or not Path(row["path"]).is_absolute()
            or not isinstance(row.get("exists"), bool)
            or row.get("kind") not in {"FILE", "DIRECTORY", "CHAR_DEVICE", "ABSENT", "OTHER"}
        ):
            raise ForensicContractError("current runtime path metadata drift")
    return copy.deepcopy(dict(value))


def build_os_evidence_receipt(
    *,
    launcher_process_identity: Mapping[str, Any],
    child_process_identity: Mapping[str, Any],
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    returncode: int,
    termination: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    namespace_before: Sequence[Mapping[str, str]],
    namespace_after: Sequence[Mapping[str, str]],
    current_runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    launcher = validate_process_identity(launcher_process_identity)
    child = validate_process_identity(child_process_identity)
    start = _validate_monotonic_ns(started_monotonic_ns, "OS evidence start")
    end = _validate_monotonic_ns(ended_monotonic_ns, "OS evidence end")
    if end <= start or isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ForensicContractError("OS evidence timing/returncode drift")
    term = _validate_termination(termination)
    termination_matches = (
        (
            term["kind"] == "EXIT"
            and returncode >= 0
            and term["exit_code_or_null"] == returncode
        )
        or (
            term["kind"] in {"SIGNAL", "TIMEOUT"}
            and returncode == -int(term["signal_number_or_null"])
            and term["signal_name_or_null"]
            == signal.Signals(int(term["signal_number_or_null"])).name
        )
    )
    if not termination_matches:
        raise ForensicContractError("OS evidence returncode/termination mismatch")
    before = _validate_namespace_snapshot(list(namespace_before))
    after = _validate_namespace_snapshot(list(namespace_after))
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_os_evidence.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "started_monotonic_ns": start,
            "ended_monotonic_ns": end,
            "runtime_ns": end - start,
            "returncode": returncode,
            "termination": term,
            "cleanup": _validate_cleanup(cleanup),
            "namespace_before": before,
            "namespace_after": after,
            "namespace_unchanged": before == after,
            "current_runtime_context": validate_current_runtime_context(
                current_runtime_context
            ),
            "pass": before == after,
        }
    )


def validate_os_evidence_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "experiment_id",
        "launcher_process_identity",
        "child_process_identity",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "runtime_ns",
        "returncode",
        "termination",
        "cleanup",
        "namespace_before",
        "namespace_after",
        "namespace_unchanged",
        "current_runtime_context",
        "pass",
        "content_digest",
    }
    if set(value) != required:
        raise ForensicContractError("OS evidence receipt key-set drift")
    rebuilt = build_os_evidence_receipt(
        launcher_process_identity=value["launcher_process_identity"],
        child_process_identity=value["child_process_identity"],
        started_monotonic_ns=value["started_monotonic_ns"],
        ended_monotonic_ns=value["ended_monotonic_ns"],
        returncode=value["returncode"],
        termination=value["termination"],
        cleanup=value["cleanup"],
        namespace_before=value["namespace_before"],
        namespace_after=value["namespace_after"],
        current_runtime_context=value["current_runtime_context"],
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("OS evidence receipt drift")
    return copy.deepcopy(dict(value))


ZERO_SCIENTIFIC_COUNTERS = {
    "scientific_inputs_opened": 0,
    "outcome_rows_opened": 0,
    "tensor_reads": 0,
    "model_loads": 0,
    "training_steps": 0,
    "files_reused": 0,
}
PREEXECUTION_CHILD_ZERO_SCIENTIFIC_COUNTERS = {
    "scientific_inputs_opened": 0,
    "failed_scientific_payloads_opened": 0,
    "outcome_rows_opened": 0,
    "tensor_reads": 0,
    "model_loads": 0,
    "training_steps": 0,
}


def build_preexecution_only_receipt(
    *,
    source_commit: str,
    namespace_before: Sequence[Mapping[str, str]],
    namespace_after: Sequence[Mapping[str, str]],
    scientific_counters: Mapping[str, Any],
) -> dict[str, Any]:
    before = _validate_namespace_snapshot(list(namespace_before))
    after = _validate_namespace_snapshot(list(namespace_after))
    if dict(scientific_counters) != ZERO_SCIENTIFIC_COUNTERS:
        raise ForensicContractError("PREEXECUTION-only scientific counter drift")
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_only.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
            "source_commit": source_commit,
            "attempt_namespace_created": False,
            "attempt_reservation_created": False,
            "canonical_or_tracked_written": False,
            "namespace_before": before,
            "namespace_after": after,
            "namespace_unchanged": before == after,
            "scientific_counters": copy.deepcopy(ZERO_SCIENTIFIC_COUNTERS),
            "scientific_attempt_consumed": False,
            "technical_startup_incident_only": True,
            "pass": before == after,
        }
    )


def validate_preexecution_only_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_preexecution_only_receipt(
        source_commit=value.get("source_commit"),
        namespace_before=value.get("namespace_before"),
        namespace_after=value.get("namespace_after"),
        scientific_counters=value.get("scientific_counters"),
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("PREEXECUTION-only receipt drift")
    return copy.deepcopy(dict(value))


def validate_archive_schema_mismatch_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-free whole-record/schema mismatch projection."""

    fixture = build_archive_verification_record_equality_fixture()
    expected_minimal = [
        sorted(row) for row in fixture["runtime_minimal_records"]
    ]
    expected_detailed = [
        sorted(row) for row in fixture["new_attempt_detailed_records"]
    ]
    required = {
        "fixture",
        "raw_record_equality",
        "identity_projection_equality",
        "mismatch_mechanism",
        "minimal_key_sets",
        "detailed_key_sets",
        "stable_identity_fields",
        "scientific_payloads_opened",
        "pass",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("fixture")
        != "MINIMAL_RUNTIME_VERSUS_DETAILED_VALIDATOR_ARCHIVE_CUSTODY"
        or value.get("raw_record_equality") is not False
        or value.get("identity_projection_equality") is not True
        or value.get("mismatch_mechanism")
        != "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
        or value.get("minimal_key_sets") != expected_minimal
        or value.get("detailed_key_sets") != expected_detailed
        or value.get("stable_identity_fields")
        != fixture["identity_projection_fields"]
        or value.get("scientific_payloads_opened") != 0
        or value.get("pass") is not True
    ):
        raise ForensicContractError("archive-schema mismatch evidence drift")
    return copy.deepcopy(dict(value))


def build_preexecution_child_result(
    *,
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
    """Build the real child's exact PREEXECUTION-only terminal payload."""

    launcher = validate_process_identity(launcher_process_identity)
    child = validate_process_identity(child_process_identity)
    closure = validate_artifact_binding(
        forensic_authority_source_closure, content_digest_required=True
    )
    freeze = validate_forensic_freeze_custody_receipt(forensic_freeze_custody)
    freeze_binding = _runtime_json_binding("forensic_freeze_custody", freeze)
    mismatch = validate_archive_schema_mismatch_evidence(mismatch_evidence)
    validate_self_digest(committed_source_root_cause_proof)
    before = _validate_namespace_snapshot(list(output_namespace_before))
    after = _validate_namespace_snapshot(list(output_namespace_after))
    context = validate_current_runtime_context(current_runtime_context)
    if (
        not isinstance(repo_head, str)
        or len(repo_head) != 40
        or repo_clean is not True
        or repo_head != freeze["repo_head"]
        or scientific_contract_digest
        != BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or freeze["scientific_contract_digest"] != scientific_contract_digest
        or closure != freeze["forensic_authority_source_closure"]
        or dict(committed_source_root_cause_proof)
        != freeze["committed_source_root_cause_proof"]
        or before != after
        or dict(scientific_counters)
        != PREEXECUTION_CHILD_ZERO_SCIENTIFIC_COUNTERS
    ):
        raise ForensicContractError("PREEXECUTION child result custody drift")
    return attach_self_digest(
        {
            "schema": PREEXECUTION_CHILD_RESULT_SCHEMA,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "repo_head": repo_head,
            "repo_clean": True,
            "scientific_contract_digest": scientific_contract_digest,
            "forensic_authority_source_closure": closure,
            "forensic_freeze_custody": freeze_binding,
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
            "scientific_counters": copy.deepcopy(
                PREEXECUTION_CHILD_ZERO_SCIENTIFIC_COUNTERS
            ),
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_preexecution_child_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and validate the real PREEXECUTION-only child payload."""

    validate_self_digest(value)
    required = {
        "schema",
        "mode",
        "launcher_process_identity",
        "child_process_identity",
        "repo_head",
        "repo_clean",
        "scientific_contract_digest",
        "forensic_authority_source_closure",
        "forensic_freeze_custody",
        "output_namespace_before",
        "output_namespace_after",
        "namespace_unchanged",
        "attempt_namespace_created",
        "attempt_reservation_created",
        "mismatch_evidence",
        "committed_source_root_cause_proof",
        "current_runtime_context",
        "scientific_counters",
        "files_reused",
        "pass",
        "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema") != PREEXECUTION_CHILD_RESULT_SCHEMA
        or value.get("mode") != "PREEXECUTION_ONLY_DIAGNOSTIC"
        or value.get("repo_clean") is not True
        or value.get("namespace_unchanged") is not True
        or value.get("attempt_namespace_created") is not False
        or value.get("attempt_reservation_created") is not False
        or value.get("files_reused") != 0
        or value.get("pass") is not True
    ):
        raise ForensicContractError("PREEXECUTION child result key/value drift")
    validate_process_identity(value["launcher_process_identity"])
    validate_process_identity(value["child_process_identity"])
    validate_artifact_binding(
        value["forensic_authority_source_closure"],
        content_digest_required=True,
    )
    validate_artifact_binding(
        value["forensic_freeze_custody"], content_digest_required=True
    )
    before = _validate_namespace_snapshot(value["output_namespace_before"])
    after = _validate_namespace_snapshot(value["output_namespace_after"])
    if before != after:
        raise ForensicContractError("PREEXECUTION child namespace drift")
    validate_archive_schema_mismatch_evidence(value["mismatch_evidence"])
    validate_self_digest(value["committed_source_root_cause_proof"])
    validate_current_runtime_context(value["current_runtime_context"])
    if (
        value.get("scientific_contract_digest")
        != BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or value.get("scientific_counters")
        != PREEXECUTION_CHILD_ZERO_SCIENTIFIC_COUNTERS
    ):
        raise ForensicContractError("PREEXECUTION child science-boundary drift")
    return copy.deepcopy(dict(value))


SYNTHETIC_TECHNICAL_STAGE_ROW_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.preexecution_synthetic_technical_stage.v1"
)


def validate_synthetic_technical_lifecycle_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != 2 * len(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS)
    ):
        raise ForensicContractError("synthetic technical lifecycle cardinality drift")
    expected = [
        (stage_id, event)
        for stage_id in PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS
        for event in ("STARTED", "COMPLETED")
    ]
    output: list[dict[str, Any]] = []
    previous_ns = 0
    for sequence, (row, expected_pair) in enumerate(zip(rows, expected)):
        validate_self_digest(row)
        if (
            set(row)
            != {
                "schema",
                "sequence",
                "stage_id",
                "event",
                "monotonic_ns",
                "content_digest",
            }
            or row.get("schema") != SYNTHETIC_TECHNICAL_STAGE_ROW_SCHEMA
            or row.get("sequence") != sequence
            or (row.get("stage_id"), row.get("event")) != expected_pair
            or isinstance(row.get("monotonic_ns"), bool)
            or not isinstance(row.get("monotonic_ns"), int)
            or row["monotonic_ns"] <= previous_ns
        ):
            raise ForensicContractError("synthetic technical lifecycle row drift")
        previous_ns = row["monotonic_ns"]
        output.append(copy.deepcopy(dict(row)))
    return output


def build_synthetic_technical_lifecycle_binding(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_synthetic_technical_lifecycle_rows(rows)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in validated)
    return _artifact_binding(
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["synthetic_technical_lifecycle"],
        payload,
        rows=len(validated),
    )


def build_synthetic_results_receipt(
    *, fixture_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(fixture_rows, Sequence) or isinstance(fixture_rows, (str, bytes)):
        raise ForensicContractError("synthetic result rows must be a sequence")
    if len(fixture_rows) != len(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES):
        raise ForensicContractError("synthetic result cardinality drift")
    expectations = {
        "PASS": {
            "termination": {
                "kind": "EXIT",
                "exit_code_or_null": 0,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            },
            "exception": False,
            "traceback_nonempty": False,
        },
        "RAISE": {
            "termination": {
                "kind": "EXIT",
                "exit_code_or_null": 1,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            },
            "exception": True,
            "traceback_nonempty": True,
        },
        "EXIT_NONZERO": {
            "termination": {
                "kind": "EXIT",
                "exit_code_or_null": 23,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            },
            "exception": True,
            "traceback_nonempty": True,
        },
        "SIGTERM": {
            "termination": {
                "kind": "SIGNAL",
                "exit_code_or_null": None,
                "signal_number_or_null": int(signal.SIGTERM),
                "signal_name_or_null": signal.Signals(signal.SIGTERM).name,
            },
            "exception": False,
            "traceback_nonempty": False,
        },
        "MISSING_PATH": {
            "termination": {
                "kind": "EXIT",
                "exit_code_or_null": 1,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            },
            "exception": True,
            "traceback_nonempty": True,
        },
        "UNICODE_RAISE": {
            "termination": {
                "kind": "EXIT",
                "exit_code_or_null": 1,
                "signal_number_or_null": None,
                "signal_name_or_null": None,
            },
            "exception": True,
            "traceback_nonempty": True,
        },
    }
    rows: list[dict[str, Any]] = []
    for expected_id, row in zip(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES, fixture_rows):
        if not isinstance(row, Mapping):
            raise ForensicContractError("synthetic result row drift")
        required = {
            "fixture_id",
            "termination",
            "exception_observed",
            "unicode_exception_preserved",
            "missing_path_preserved",
            "streams",
            "cleanup",
            "scientific_counters",
            "expected_behavior_observed",
            "technical_lifecycle",
            "technical_lifecycle_stage_sequence",
            "technical_resources_cleaned",
            "last_stage_marker",
            "last_stage_marker_value",
            "startup_stage_prefix",
            "startup_stage_prefix_rows",
        }
        if set(row) != required or row.get("fixture_id") != expected_id:
            raise ForensicContractError("synthetic result fixture order drift")
        termination = _validate_termination(row["termination"])
        streams = _validate_stream_bindings(row["streams"])
        cleanup = _validate_cleanup(row["cleanup"])
        expected = expectations[expected_id]
        lifecycle = row.get("technical_lifecycle")
        marker = validate_last_stage_marker(row.get("last_stage_marker_value", {}))
        marker_binding = validate_artifact_binding(
            row.get("last_stage_marker", {}), content_digest_required=True
        )
        marker_payload = canonical_json_bytes(marker) + b"\n"
        expected_marker = (
            ("PREEXECUTION_DIAGNOSTIC_CHILD", "COMPLETE", "COMPLETED")
            if expected_id == "PASS"
            else (
                (
                    "PREEXECUTION_DIAGNOSTIC_WRAPPER",
                    "BEFORE_EVALUATOR_IMPORT_AND_ARGPARSE_HANDOFF",
                    "COMPLETED",
                )
                if expected_id == "MISSING_PATH"
                else (
                    "PREEXECUTION_DIAGNOSTIC_CHILD",
                    "VALIDATE_FORENSIC_AUTHORITY",
                    "STARTED",
                )
            )
        )
        startup_rows_raw = row.get("startup_stage_prefix_rows")
        if not isinstance(startup_rows_raw, list):
            raise ForensicContractError("synthetic startup prefix rows drift")
        startup_rows = validate_startup_stage_rows(
            startup_rows_raw, require_complete=expected_id == "PASS"
        )
        expected_startup_count = 14 if expected_id == "PASS" else (
            0 if expected_id == "MISSING_PATH" else 5
        )
        startup_binding = row.get("startup_stage_prefix")
        if startup_rows:
            if not isinstance(startup_binding, Mapping):
                raise ForensicContractError("synthetic startup prefix binding absent")
            startup_binding = copy.deepcopy(dict(startup_binding))
            startup_payload = b"".join(
                canonical_json_bytes(item) + b"\n" for item in startup_rows
            )
            if (
                set(startup_binding) != {"path", "sha256", "bytes", "rows"}
                or startup_binding.get("sha256")
                != hashlib.sha256(startup_payload).hexdigest()
                or startup_binding.get("bytes") != len(startup_payload)
                or startup_binding.get("rows") != len(startup_rows)
            ):
                raise ForensicContractError("synthetic startup prefix binding drift")
        elif startup_binding is not None:
            raise ForensicContractError("absent synthetic prefix has a binding")
        expected_lifecycle_sequence = [
            [stage_id, event]
            for stage_id in PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS
            for event in ("STARTED", "COMPLETED")
        ]
        if (
            not isinstance(row.get("exception_observed"), bool)
            or not isinstance(row.get("unicode_exception_preserved"), bool)
            or not isinstance(row.get("missing_path_preserved"), bool)
            or dict(row.get("scientific_counters", {})) != ZERO_SCIENTIFIC_COUNTERS
            or row.get("expected_behavior_observed") is not True
            or termination != expected["termination"]
            or row["exception_observed"] is not expected["exception"]
            or row["unicode_exception_preserved"]
            is not (expected_id == "UNICODE_RAISE")
            or row["missing_path_preserved"]
            is not (expected_id == "MISSING_PATH")
            or (streams["exception"]["bytes"] > 0)
            is not expected["exception"]
            or (streams["traceback"]["bytes"] > 0)
            is not expected["traceback_nonempty"]
            or streams["heartbeat"]["bytes"] <= 0
            or streams["read_guard_events"]["bytes"] != 0
            or not isinstance(lifecycle, Mapping)
            or set(lifecycle) != {"path", "sha256", "bytes", "rows"}
            or lifecycle.get("path")
            != PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                "synthetic_technical_lifecycle"
            ]
            or not isinstance(lifecycle.get("sha256"), str)
            or len(lifecycle["sha256"]) != 64
            or not isinstance(lifecycle.get("bytes"), int)
            or lifecycle["bytes"] <= 0
            or lifecycle.get("rows")
            != 2 * len(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS)
            or row.get("technical_lifecycle_stage_sequence")
            != expected_lifecycle_sequence
            or row.get("technical_resources_cleaned") is not True
            or marker_binding["sha256"]
            != hashlib.sha256(marker_payload).hexdigest()
            or marker_binding["bytes"] != len(marker_payload)
            or marker_binding["content_digest"] != marker["content_digest"]
            or (
                marker["producer_role"], marker["stage_id"], marker["event"]
            )
            != expected_marker
            or len(startup_rows) != expected_startup_count
        ):
            raise ForensicContractError("synthetic result evidence drift")
        rows.append(
            attach_self_digest(
                {
                    "fixture_id": expected_id,
                    "termination": termination,
                    "exception_observed": row["exception_observed"],
                    "unicode_exception_preserved": row[
                        "unicode_exception_preserved"
                    ],
                    "missing_path_preserved": row["missing_path_preserved"],
                    "streams": streams,
                    "cleanup": cleanup,
                    "scientific_counters": copy.deepcopy(
                        ZERO_SCIENTIFIC_COUNTERS
                    ),
                    "expected_behavior_observed": True,
                    "technical_lifecycle": copy.deepcopy(dict(lifecycle)),
                    "technical_lifecycle_stage_sequence": (
                        expected_lifecycle_sequence
                    ),
                    "technical_resources_cleaned": True,
                    "last_stage_marker": marker_binding,
                    "last_stage_marker_value": marker,
                    "startup_stage_prefix": startup_binding,
                    "startup_stage_prefix_rows": startup_rows,
                }
            )
        )
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.preexecution_synthetic_results.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "rows": rows,
            "row_count": len(rows),
            "fixture_ids": list(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES),
            "all_capture_and_cleanup_pass": True,
            "scientific_inputs_opened": 0,
            "pass": True,
        }
    )


def validate_synthetic_results_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ForensicContractError("synthetic result rows absent")
    raw_rows = []
    for row in rows:
        validate_self_digest(row)
        raw = copy.deepcopy(dict(row))
        raw.pop("content_digest")
        raw_rows.append(raw)
    rebuilt = build_synthetic_results_receipt(fixture_rows=raw_rows)
    if dict(value) != rebuilt:
        raise ForensicContractError("synthetic result receipt drift")
    return copy.deepcopy(dict(value))


def _runtime_json_binding(key: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if key not in PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS:
        raise ForensicContractError("unknown diagnostic runtime binding key")
    validate_self_digest(value)
    payload = canonical_json_bytes(value) + b"\n"
    return _artifact_binding(
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
        payload,
        content_digest=value["content_digest"],
    )


def _startup_ledger_binding(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validated = validate_startup_stage_rows(rows, require_complete=True)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in validated)
    return _artifact_binding(
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["startup_stage_ledger"],
        payload,
        rows=len(validated),
    )


def build_diagnostic_custody_receipt(
    *,
    repo_root: str | Path,
    source_commit: str,
    forensic_source_closure: Mapping[str, Any],
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
    preexecution_child_result: Mapping[str, Any],
    last_stage_marker_value: Mapping[str, Any],
    startup_stage_rows: Sequence[Mapping[str, Any]],
    stream_bindings: Mapping[str, Any],
    exception_observed: bool,
) -> dict[str, Any]:
    root = Path(diagnostic_root).absolute()
    launcher = validate_process_identity(launcher_process_identity)
    child = validate_process_identity(child_process_identity)
    freeze_custody = validate_forensic_freeze_custody_receipt(
        forensic_freeze_custody_receipt
    )
    umask_custody = validate_diagnostic_umask_custody(umask_custody_receipt)
    invocation = validate_invocation_receipt(invocation_receipt)
    environment = validate_environment_receipt(environment_receipt)
    command = validate_command_receipt(command_receipt)
    read_guard = validate_read_guard_manifest(
        read_guard_manifest, repo_root=repo_root
    )
    os_evidence = validate_os_evidence_receipt(os_evidence_receipt)
    preexecution = validate_preexecution_only_receipt(preexecution_only_receipt)
    synthetic = validate_synthetic_results_receipt(synthetic_results_receipt)
    child_result = validate_preexecution_child_result(preexecution_child_result)
    marker = validate_last_stage_marker(last_stage_marker_value)
    rows = validate_startup_stage_rows(startup_stage_rows, require_complete=True)
    streams = _validate_stream_bindings(stream_bindings)
    validate_self_digest(forensic_source_closure)
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or freeze_custody["repo_head"] != source_commit
        or invocation["source_commit"] != source_commit
        or invocation["diagnostic_root"] != str(root)
        or invocation["launcher_process_identity"] != launcher
        or invocation["child_process_identity"] != child
        or command["outer_exact_argv"] != child["argv"]
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
        or child_result["launcher_process_identity"] != launcher
        or child_result["child_process_identity"] != child
        or child_result["repo_head"] != source_commit
        or child_result["forensic_authority_source_closure"]
        != freeze_custody["forensic_authority_source_closure"]
        or child_result["forensic_freeze_custody"]
        != _runtime_json_binding("forensic_freeze_custody", freeze_custody)
        or invocation["technical_runtime_context"]
        != os_evidence["current_runtime_context"]
        or invocation["technical_runtime_context"]["umask"]["value"]
        != PREEXECUTION_DIAGNOSTIC_UMASK
        or child_result["current_runtime_context"]["umask"]["value"]
        != PREEXECUTION_DIAGNOSTIC_UMASK
        or marker["producer_role"] != "PREEXECUTION_DIAGNOSTIC_CHILD"
        or marker["stage_id"] != "COMPLETE"
        or marker["event"] != "COMPLETED"
        or marker["pid"] != child["pid"]
        or not isinstance(exception_observed, bool)
        or exception_observed is not False
        or streams["read_guard_events"]["bytes"] != 0
    ):
        raise ForensicContractError("diagnostic custody cross-binding drift")
    child_payload = canonical_json_bytes(child_result) + b"\n"
    if (
        streams["stdout"]["sha256"]
        != hashlib.sha256(child_payload).hexdigest()
        or streams["stdout"]["bytes"] != len(child_payload)
    ):
        raise ForensicContractError("PREEXECUTION child stdout/result drift")
    return attach_self_digest(
        {
            "schema": DIAGNOSTIC_CUSTODY_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
            "source_commit": source_commit,
            "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
            "forensic_source_closure": copy.deepcopy(dict(forensic_source_closure)),
            "diagnostic_root": str(root),
            "launcher_process_identity": launcher,
            "child_process_identity": child,
            "exact_argv": copy.deepcopy(child["argv"]),
            "receipts": {
                "forensic_freeze_custody": _runtime_json_binding(
                    "forensic_freeze_custody", freeze_custody
                ),
                "umask_custody": _runtime_json_binding(
                    "umask_custody", umask_custody
                ),
                "invocation": _runtime_json_binding("invocation", invocation),
                "environment": _runtime_json_binding("environment", environment),
                "command": _runtime_json_binding("command", command),
                "read_guard_manifest": _runtime_json_binding(
                    "read_guard_manifest", read_guard
                ),
                "os_evidence": _runtime_json_binding("os_evidence", os_evidence),
                "preexecution_only": _runtime_json_binding(
                    "preexecution_only", preexecution
                ),
                "synthetic_results": _runtime_json_binding(
                    "synthetic_results", synthetic
                ),
                "startup_stage_ledger": _startup_ledger_binding(rows),
                "last_stage_marker": _runtime_json_binding(
                    "last_stage_marker", marker
                ),
            },
            "streams": streams,
            "preexecution_child_result": child_result,
            "last_stage_marker_value": marker,
            "launcher_technical_runtime_context": copy.deepcopy(
                invocation["technical_runtime_context"]
            ),
            "child_technical_runtime_context": copy.deepcopy(
                child_result["current_runtime_context"]
            ),
            "exception_observed": exception_observed,
            "started_monotonic_ns": os_evidence["started_monotonic_ns"],
            "ended_monotonic_ns": os_evidence["ended_monotonic_ns"],
            "runtime_ns": os_evidence["runtime_ns"],
            "last_started_stage": PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1],
            "last_completed_stage": PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1],
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
            "scientific_counters": copy.deepcopy(ZERO_SCIENTIFIC_COUNTERS),
            "files_reused": 0,
            "namespace_before": copy.deepcopy(preexecution["namespace_before"]),
            "namespace_after": copy.deepcopy(preexecution["namespace_after"]),
            "namespace_unchanged": True,
            "pass_meaning": "TECHNICAL_CUSTODY_COMPLETE_NOT_SCIENTIFIC_SUCCESS",
            "pass": True,
        }
    )


def validate_diagnostic_custody_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    required = {
        "schema",
        "experiment_id",
        "mode",
        "source_commit",
        "forensic_contract",
        "forensic_source_closure",
        "diagnostic_root",
        "launcher_process_identity",
        "child_process_identity",
        "exact_argv",
        "receipts",
        "streams",
        "preexecution_child_result",
        "last_stage_marker_value",
        "launcher_technical_runtime_context",
        "child_technical_runtime_context",
        "exception_observed",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "runtime_ns",
        "last_started_stage",
        "last_completed_stage",
        "returncode",
        "termination",
        "cleanup",
        "process_state_scope_at_custody_write",
        "attempt_namespace_created",
        "attempt_reservation_created",
        "canonical_or_tracked_written",
        "scientific_counters",
        "files_reused",
        "namespace_before",
        "namespace_after",
        "namespace_unchanged",
        "pass_meaning",
        "pass",
        "content_digest",
    }
    if (
        set(value) != required
        or value.get("schema") != DIAGNOSTIC_CUSTODY_SCHEMA
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or value.get("mode") != "PREEXECUTION_ONLY_DIAGNOSTIC"
        or value.get("forensic_contract") != FORENSIC_CONTRACT_BINDING
        or value.get("attempt_namespace_created") is not False
        or value.get("attempt_reservation_created") is not False
        or value.get("canonical_or_tracked_written") is not False
        or value.get("scientific_counters") != ZERO_SCIENTIFIC_COUNTERS
        or value.get("files_reused") != 0
        or value.get("namespace_unchanged") is not True
        or value.get("namespace_before") != value.get("namespace_after")
        or value.get("last_started_stage")
        != PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1]
        or value.get("last_completed_stage")
        != PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1]
        or value.get("returncode") != 0
        or value.get("exception_observed") is not False
        or value.get("pass_meaning")
        != "TECHNICAL_CUSTODY_COMPLETE_NOT_SCIENTIFIC_SUCCESS"
        or value.get("pass") is not True
    ):
        raise ForensicContractError("diagnostic custody receipt drift")
    validate_process_identity(value["launcher_process_identity"])
    child = validate_process_identity(value["child_process_identity"])
    if value.get("exact_argv") != child["argv"]:
        raise ForensicContractError("diagnostic custody argv drift")
    _validate_termination(value["termination"])
    if value["termination"] != {
        "kind": "EXIT",
        "exit_code_or_null": 0,
        "signal_number_or_null": None,
        "signal_name_or_null": None,
    }:
        raise ForensicContractError("diagnostic successful termination drift")
    _validate_cleanup(value["cleanup"])
    process_scope = value.get("process_state_scope_at_custody_write")
    if process_scope != {
        "launcher_process_identity": value["launcher_process_identity"],
        "launcher_live_at_custody_write": True,
        "cleanup_scope": (
            "TERMINATED_CHILD_PROCESS_GROUP_AND_NONLAUNCHER_FORENSIC_ROLES"
        ),
        "child_process_group_and_nonlauncher_roles_zero": True,
        "literal_zero_all_forensic_roles_claimed": False,
        "all_forensic_role_zero_validation": (
            "REQUIRED_AFTER_LAUNCHER_EXIT_BY_POSTCOMMIT_VALIDATOR"
        ),
    }:
        raise ForensicContractError("diagnostic custody process-scope drift")
    before = _validate_namespace_snapshot(value["namespace_before"])
    after = _validate_namespace_snapshot(value["namespace_after"])
    if before != after:
        raise ForensicContractError("diagnostic namespace cross-binding drift")
    streams = _validate_stream_bindings(value["streams"])
    child_result = validate_preexecution_child_result(
        value["preexecution_child_result"]
    )
    marker = validate_last_stage_marker(value["last_stage_marker_value"])
    launcher_runtime_context = validate_current_runtime_context(
        value["launcher_technical_runtime_context"]
    )
    child_runtime_context = validate_current_runtime_context(
        value["child_technical_runtime_context"]
    )
    if (
        streams["exception"]["bytes"] != 0
        or streams["traceback"]["bytes"] != 0
        or streams["read_guard_events"]["bytes"] != 0
        or streams["heartbeat"]["bytes"] <= 0
        or child_result["launcher_process_identity"]
        != value["launcher_process_identity"]
        or child_result["child_process_identity"] != child
        or child_result["current_runtime_context"] != child_runtime_context
        or marker["producer_role"] != "PREEXECUTION_DIAGNOSTIC_CHILD"
        or marker["stage_id"] != "COMPLETE"
        or marker["event"] != "COMPLETED"
        or marker["pid"] != child["pid"]
    ):
        raise ForensicContractError("diagnostic successful stream custody drift")
    child_payload = canonical_json_bytes(child_result) + b"\n"
    if (
        streams["stdout"]["sha256"]
        != hashlib.sha256(child_payload).hexdigest()
        or streams["stdout"]["bytes"] != len(child_payload)
    ):
        raise ForensicContractError("diagnostic child result/stdout binding drift")
    _validate_monotonic_ns(value["started_monotonic_ns"], "custody start")
    _validate_monotonic_ns(value["ended_monotonic_ns"], "custody end")
    if (
        value["ended_monotonic_ns"] <= value["started_monotonic_ns"]
        or value["runtime_ns"]
        != value["ended_monotonic_ns"] - value["started_monotonic_ns"]
    ):
        raise ForensicContractError("diagnostic custody timing drift")
    validate_forensic_source_closure(
        value["forensic_source_closure"], require_complete=True
    )
    receipts = value.get("receipts")
    if not isinstance(receipts, Mapping) or set(receipts) != {
        "invocation",
        "forensic_freeze_custody",
        "umask_custody",
        "environment",
        "command",
        "read_guard_manifest",
        "os_evidence",
        "preexecution_only",
        "synthetic_results",
        "last_stage_marker",
        "startup_stage_ledger",
    }:
        raise ForensicContractError("diagnostic custody receipt binding drift")
    for key in (
        "invocation",
        "forensic_freeze_custody",
        "umask_custody",
        "environment",
        "command",
        "read_guard_manifest",
        "os_evidence",
        "preexecution_only",
        "synthetic_results",
        "last_stage_marker",
    ):
        validate_artifact_binding(receipts[key], content_digest_required=True)
    if (
        receipts["last_stage_marker"]
        != _runtime_json_binding("last_stage_marker", marker)
        or child_result["forensic_freeze_custody"]
        != receipts["forensic_freeze_custody"]
    ):
        raise ForensicContractError("diagnostic marker/freeze binding drift")
    ledger = receipts["startup_stage_ledger"]
    if (
        set(ledger) != {"path", "sha256", "bytes", "rows"}
        or ledger.get("rows") != 2 * len(PREEXECUTION_DIAGNOSTIC_STAGE_IDS)
    ):
        raise ForensicContractError("diagnostic startup ledger binding drift")
    return copy.deepcopy(dict(value))


def _binding_for_file(path: Path, relative_path: str, **extra: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ForensicContractError(f"diagnostic artifact absent: {relative_path}")
    sha256, size = _sha256_file(path)
    return {
        "path": relative_path,
        "sha256": sha256,
        "bytes": size,
        **copy.deepcopy(extra),
    }


def _load_startup_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ForensicContractError("diagnostic startup ledger absent")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line in stream:
                if not line.endswith("\n") or not line.strip():
                    raise ForensicContractError("diagnostic startup ledger framing drift")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ForensicContractError("diagnostic startup row is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicContractError("diagnostic startup ledger parse drift") from exc
    return validate_startup_stage_rows(rows, require_complete=True)


def _same_opened_stat(before: os.stat_result, opened: os.stat_result) -> bool:
    return (
        (before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_gid, before.st_nlink)
        == (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid, opened.st_gid, opened.st_nlink)
    )


def _open_absolute_directory_no_follow(path: Path) -> int:
    """Anchor an absolute directory path from `/` without following symlinks."""

    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts:
        raise ForensicContractError("secure directory path drift")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            before = os.stat(component, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ForensicContractError("secure directory component drift")
            child_fd = os.open(component, flags, dir_fd=fd)
            opened = os.fstat(child_fd)
            if not _same_opened_stat(before, opened):
                os.close(child_fd)
                raise ForensicContractError("secure directory component race")
            os.close(fd)
            fd = child_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _stat_matches_inventory_row(
    info: os.stat_result, row: Mapping[str, Any], *, kind: str
) -> bool:
    common = {
        "kind": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
    }
    if kind == "FILE":
        common["bytes"] = int(info.st_size)
    return all(row.get(key) == value for key, value in common.items())


def _read_no_follow_bound_file(
    root: Path,
    relative_path: str | Path,
    *,
    expected_stat_rows: Mapping[str, Mapping[str, Any]] | None,
) -> bytes:
    """Open every path component from a verified dirfd before reading a leaf."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ForensicContractError("secure bundle relative path drift")
    directory_fd = _open_absolute_directory_no_follow(root)
    current_relative = "."
    try:
        if expected_stat_rows is not None:
            root_row = expected_stat_rows.get(".")
            if root_row is None or not _stat_matches_inventory_row(
                os.fstat(directory_fd), root_row, kind="DIRECTORY"
            ):
                raise ForensicContractError("secure bundle root stat drift")
        for component in relative.parts[:-1]:
            before = os.stat(
                component, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ForensicContractError("secure bundle parent drift")
            flags = (
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            if not _same_opened_stat(before, opened):
                os.close(child_fd)
                raise ForensicContractError("secure bundle parent race")
            current_relative = (
                component
                if current_relative == "."
                else f"{current_relative}/{component}"
            )
            if expected_stat_rows is not None:
                row = expected_stat_rows.get(current_relative)
                if row is None or not _stat_matches_inventory_row(
                    opened, row, kind="DIRECTORY"
                ):
                    os.close(child_fd)
                    raise ForensicContractError("secure bundle parent stat drift")
            os.close(directory_fd)
            directory_fd = child_fd
        leaf = relative.parts[-1]
        before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ForensicContractError("secure bundle leaf type/link drift")
        leaf_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(leaf_fd)
            if not _same_opened_stat(before, opened) or opened.st_size != before.st_size:
                raise ForensicContractError("secure bundle leaf race")
            if expected_stat_rows is not None:
                row = expected_stat_rows.get(str(relative))
                self_excluded = (
                    str(relative)
                    == PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
                        "final_namespace_inventory"
                    ]
                )
                if row is None and not self_excluded:
                    raise ForensicContractError(
                        "secure bundle leaf missing from preflight inventory"
                    )
                if row is not None and not _stat_matches_inventory_row(
                    opened, row, kind="FILE"
                ):
                    raise ForensicContractError("secure bundle leaf stat drift")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(leaf_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(leaf_fd)
            if not _same_opened_stat(opened, after) or after.st_size != opened.st_size:
                raise ForensicContractError("secure bundle leaf changed during read")
            payload = b"".join(chunks)
            if len(payload) != opened.st_size:
                raise ForensicContractError("secure bundle short read")
            return payload
        finally:
            os.close(leaf_fd)
    finally:
        os.close(directory_fd)


def _exclusive_write_relative_no_follow(
    root: Path,
    relative_path: str | Path,
    payload: bytes,
    *,
    expected_directory_rows: Mapping[str, Mapping[str, Any]],
    mode: int = 0o644,
) -> None:
    """Create a leaf through an anchored, preflight-bound directory chain."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ForensicContractError("secure write relative path drift")
    directory_fd = _open_absolute_directory_no_follow(root)
    current_relative = "."
    try:
        root_row = expected_directory_rows.get(".")
        if root_row is None or not _stat_matches_inventory_row(
            os.fstat(directory_fd), root_row, kind="DIRECTORY"
        ):
            raise ForensicContractError("secure write root stat drift")
        for component in relative.parts[:-1]:
            before = os.stat(
                component, dir_fd=directory_fd, follow_symlinks=False
            )
            flags = (
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISDIR(before.st_mode)
                or not _same_opened_stat(before, opened)
            ):
                os.close(child_fd)
                raise ForensicContractError("secure write parent race")
            current_relative = (
                component
                if current_relative == "."
                else f"{current_relative}/{component}"
            )
            row = expected_directory_rows.get(current_relative)
            if row is None or not _stat_matches_inventory_row(
                opened, row, kind="DIRECTORY"
            ):
                os.close(child_fd)
                raise ForensicContractError("secure write parent stat drift")
            os.close(directory_fd)
            directory_fd = child_fd
        leaf = relative.parts[-1]
        try:
            os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForensicContractError("secure write refuses stale leaf")
        leaf_fd = os.open(
            leaf,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(leaf_fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            created = os.fstat(leaf_fd)
            if (
                not stat.S_ISREG(created.st_mode)
                or created.st_nlink != 1
                or stat.S_IMODE(created.st_mode) != mode
                or created.st_uid != 1000
                or created.st_gid != 1000
                or created.st_size != len(payload)
            ):
                raise ForensicContractError("secure write leaf stat drift")
        finally:
            os.close(leaf_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _unlink_relative_no_follow(root: Path, relative_path: str | Path) -> None:
    """Remove one regular leaf through an anchored directory chain and fsync it."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ForensicContractError("secure unlink relative path drift")
    directory_fd = _open_absolute_directory_no_follow(root)
    try:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in relative.parts[:-1]:
            before = os.stat(
                component, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ForensicContractError("secure unlink parent drift")
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            if not _same_opened_stat(before, opened):
                os.close(child_fd)
                raise ForensicContractError("secure unlink parent race")
            os.close(directory_fd)
            directory_fd = child_fd
        leaf = relative.parts[-1]
        before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ForensicContractError("secure unlink leaf type drift")
        os.unlink(leaf, dir_fd=directory_fd)
        os.fsync(directory_fd)
        try:
            os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForensicContractError("secure unlink did not remove leaf")
    finally:
        os.close(directory_fd)


def _relative_leaf_absent_no_follow(
    root: Path, relative_path: str | Path
) -> bool:
    """Prove a relative leaf absent without following any directory component."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ForensicContractError("secure absence relative path drift")
    directory_fd = _open_absolute_directory_no_follow(root)
    try:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for component in relative.parts[:-1]:
            try:
                before = os.stat(
                    component, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                return True
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ForensicContractError("secure absence parent drift")
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            opened = os.fstat(child_fd)
            if not _same_opened_stat(before, opened):
                os.close(child_fd)
                raise ForensicContractError("secure absence parent race")
            os.close(directory_fd)
            directory_fd = child_fd
        try:
            os.stat(
                relative.parts[-1],
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(directory_fd)


def _rollback_relative_paths_no_follow(
    root: Path,
    relative_paths: Iterable[str | Path],
    *,
    label: str,
) -> list[str]:
    """Remove and durably prove absence for an exact intended leaf set."""

    normalized = [Path(value) for value in relative_paths]
    if len(normalized) != len({str(path) for path in normalized}):
        raise ForensicContractError(f"{label} rollback path-set repeats a path")
    failures: list[str] = []
    removed: list[str] = []
    for relative in reversed(normalized):
        try:
            absent = _relative_leaf_absent_no_follow(root, relative)
        except BaseException as exc:
            failures.append(
                f"{relative}: initial absence check failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if absent:
            continue
        try:
            _unlink_relative_no_follow(root, relative)
            removed.append(str(relative))
        except BaseException as exc:
            failures.append(
                f"{relative}: unlink/fsync failed: {type(exc).__name__}: {exc}"
            )
    for relative in normalized:
        try:
            absent = _relative_leaf_absent_no_follow(root, relative)
        except BaseException as exc:
            failures.append(
                f"{relative}: final absence proof failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not absent:
            failures.append(f"{relative}: remains after rollback")
    if failures:
        raise ForensicContractError(
            f"{label} rollback could not durably prove the full intended "
            f"path set absent: {failures}"
        )
    return removed


def _canonical_json_from_bound_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicContractError(f"{label} JSON parse drift") from exc
    if (
        not isinstance(value, dict)
        or payload != canonical_json_bytes(value) + b"\n"
    ):
        raise ForensicContractError(f"{label} canonical JSON byte drift")
    return value


def _startup_rows_from_bound_bytes(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
        for line in text.splitlines(keepends=True):
            if not line.endswith("\n") or not line.strip():
                raise ForensicContractError(
                    "diagnostic startup ledger framing drift"
                )
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or line.encode("utf-8") != canonical_json_bytes(value) + b"\n"
            ):
                raise ForensicContractError(
                    "diagnostic startup row canonical-byte drift"
                )
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForensicContractError("diagnostic startup ledger parse drift") from exc
    return validate_startup_stage_rows(rows, require_complete=True)


def build_preexecution_runtime_result(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the external pre-publication terminal; it is not the tracked result."""

    custody = validate_diagnostic_custody_receipt(
        diagnostic_custody_receipt
    )
    synthetic = validate_synthetic_results_receipt(synthetic_results_receipt)
    decision = build_conditional_v2_decision(
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    return attach_self_digest(
        {
            "schema": PREEXECUTION_RUNTIME_RESULT_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
            "source_freeze_commit": custody["source_commit"],
            "forensic_classification": FORENSIC_PRIMARY_CLASSIFICATION,
            "forensic_secondary_mechanism": FORENSIC_SECONDARY_MECHANISM,
            "technical_defect_id": FORENSIC_DEFECT_ID,
            "scientific_disposition": SCIENTIFIC_DISPOSITION,
            "scientific_result": False,
            "scientific_classification_authorized": False,
            "scientific_classification": None,
            "diagnostic_custody": _runtime_json_binding(
                "diagnostic_custody", custody
            ),
            "process_state_scope_at_prepublication": copy.deepcopy(
                custody["process_state_scope_at_custody_write"]
            ),
            "launcher_live_during_prepublication": True,
            "literal_zero_all_forensic_roles_claimed": False,
            "all_forensic_role_zero_validation": (
                "PENDING_POSTCOMMIT_VALIDATOR_AFTER_LAUNCHER_EXIT"
            ),
            "conditional_v2_decision": decision,
            "v2_specification_publication": (
                "REQUIRED_PENDING_TRACKED_RESULT_PUBLICATION"
                if decision["v2_spec_authorized"] is True
                else "NOT_AUTHORIZED"
            ),
            "v2_spec_written": False,
            "automatic_v2_execution_authorized": False,
            "scientific_attempt_authorized": False,
            "scientific_attempt_consumed": False,
            "scientific_inputs_opened": 0,
            "pass_meaning": (
                "TECHNICAL_CUSTODY_COMPLETE_PREPUBLICATION_"
                "NOT_A_SCIENTIFIC_RESULT"
            ),
            "pass": True,
        }
    )


def validate_preexecution_runtime_result(
    value: Mapping[str, Any],
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_preexecution_runtime_result(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("preexecution runtime result drift")
    return copy.deepcopy(dict(value))


def load_and_validate_diagnostic_bundle(
    *, repo_root: str | Path, diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT
) -> dict[str, Any]:
    """No-follow load the terminal technical bundle after a full lstat preflight."""

    repo = Path(repo_root).resolve()
    root = Path(diagnostic_root).absolute()
    if root != PREEXECUTION_DIAGNOSTIC_ROOT or root.is_symlink() or not root.is_dir():
        raise ForensicContractError("diagnostic result root drift")
    # Security ordering is intentional: no bundle byte is read until the exact
    # complete namespace has passed an lstat-only, no-symlink inventory.
    preflight = _observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    preflight_rows = {
        str(row["path"]): row
        for row in preflight["rows"]
    }
    paths = {
        key: root / relative
        for key, relative in PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS.items()
    }
    loaded_bytes = {
        key: _read_no_follow_bound_file(
            root,
            PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
            expected_stat_rows=preflight_rows,
        )
        for key in PREEXECUTION_DIAGNOSTIC_FINAL_MAIN_FILE_KEYS
    }

    def load_json_key(key: str) -> dict[str, Any]:
        return _canonical_json_from_bound_bytes(loaded_bytes[key], label=key)

    environment = validate_environment_receipt(load_json_key("environment"))
    command = validate_command_receipt(load_json_key("command"))
    invocation = validate_invocation_receipt(load_json_key("invocation"))
    os_evidence = validate_os_evidence_receipt(load_json_key("os_evidence"))
    preexecution = validate_preexecution_only_receipt(
        load_json_key("preexecution_only")
    )
    synthetic = validate_synthetic_results_receipt(
        load_json_key("synthetic_results")
    )
    child_result = validate_preexecution_child_result(
        load_json_key("child_stdout")
    )
    last_stage_marker = validate_last_stage_marker(
        load_json_key("last_stage_marker")
    )
    read_guard = validate_read_guard_manifest(
        load_json_key("read_guard_manifest"), repo_root=repo
    )
    stage_rows = _startup_rows_from_bound_bytes(
        loaded_bytes["startup_stage_ledger"]
    )
    streams = {
        label: {
            "path": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[path_key],
            "sha256": hashlib.sha256(loaded_bytes[path_key]).hexdigest(),
            "bytes": len(loaded_bytes[path_key]),
        }
        for label, path_key in {
            "stdout": "child_stdout",
            "stderr": "child_stderr",
            "traceback": "child_traceback",
            "exception": "child_exception",
            "heartbeat": "heartbeat",
            "read_guard_events": "read_guard_events",
        }.items()
    }
    freeze_custody = validate_forensic_freeze_custody_receipt(
        load_json_key("forensic_freeze_custody")
    )
    if (
        _git_text(repo, ["rev-parse", "HEAD"]) != freeze_custody["repo_head"]
        or _git_text(repo, ["status", "--porcelain=v1"])
    ):
        raise ForensicContractError("terminal bundle repository custody drift")
    closure = _canonical_json_from_bound_bytes(
        _read_no_follow_bound_file(
            repo,
            TRACKED_FORENSIC_SOURCE_CLOSURE_PATH,
            expected_stat_rows=None,
        ),
        label="forensic_source_closure",
    )
    validate_forensic_source_closure(closure, require_complete=True)
    umask_custody = validate_diagnostic_umask_custody(
        load_json_key("umask_custody")
    )
    rebuilt = build_diagnostic_custody_receipt(
        repo_root=repo,
        source_commit=invocation["source_commit"],
        forensic_source_closure=closure,
        diagnostic_root=root,
        launcher_process_identity=invocation["launcher_process_identity"],
        child_process_identity=invocation["child_process_identity"],
        forensic_freeze_custody_receipt=freeze_custody,
        umask_custody_receipt=umask_custody,
        invocation_receipt=invocation,
        environment_receipt=environment,
        command_receipt=command,
        read_guard_manifest=read_guard,
        os_evidence_receipt=os_evidence,
        preexecution_only_receipt=preexecution,
        synthetic_results_receipt=synthetic,
        preexecution_child_result=child_result,
        last_stage_marker_value=last_stage_marker,
        startup_stage_rows=stage_rows,
        stream_bindings=streams,
        exception_observed=streams["exception"]["bytes"] > 0,
    )
    persisted = validate_diagnostic_custody_receipt(
        load_json_key("diagnostic_custody")
    )
    if rebuilt != persisted:
        raise ForensicContractError("diagnostic custody persisted/rebuilt drift")
    for key, receipt in {
        "environment": environment,
        "command": command,
        "invocation": invocation,
        "os_evidence": os_evidence,
        "preexecution_only": preexecution,
        "synthetic_results": synthetic,
        "last_stage_marker": last_stage_marker,
        "read_guard_manifest": read_guard,
        "forensic_freeze_custody": freeze_custody,
        "umask_custody": umask_custody,
    }.items():
        expected = persisted["receipts"][key]
        observed = {
            "path": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key],
            "sha256": hashlib.sha256(loaded_bytes[key]).hexdigest(),
            "bytes": len(loaded_bytes[key]),
            "content_digest": receipt["content_digest"],
        }
        if expected != observed:
            raise ForensicContractError(f"diagnostic receipt binding drift: {key}")
    if persisted["receipts"]["startup_stage_ledger"] != _startup_ledger_binding(
        stage_rows
    ):
        raise ForensicContractError("diagnostic startup ledger binding drift")
    result = validate_preexecution_runtime_result(
        load_json_key("result"),
        diagnostic_custody_receipt=persisted,
        synthetic_results_receipt=synthetic,
    )
    if loaded_bytes["result"] != canonical_json_bytes(result) + b"\n":
        raise ForensicContractError("external forensic result byte drift")
    final_inventory = validate_final_diagnostic_namespace_inventory(
        load_json_key("final_namespace_inventory"), diagnostic_root=root
    )
    postflight = _observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    if postflight != preflight:
        raise ForensicContractError("diagnostic bundle changed across reads")
    return {
        "environment": environment,
        "command": command,
        "invocation": invocation,
        "os_evidence": os_evidence,
        "preexecution_only": preexecution,
        "synthetic_results": synthetic,
        "preexecution_child_result": child_result,
        "last_stage_marker": last_stage_marker,
        "read_guard_manifest": read_guard,
        "startup_stage_rows": stage_rows,
        "streams": streams,
        "diagnostic_custody": persisted,
        "forensic_freeze_custody": freeze_custody,
        "umask_custody": umask_custody,
        "result": result,
        "final_namespace_inventory": final_inventory,
        "preflight_namespace_inventory": preflight,
        "postflight_namespace_inventory": postflight,
    }


def build_tracked_invocation_comparison_receipt(
    *,
    invocation_receipt: Mapping[str, Any],
    command_receipt: Mapping[str, Any],
    preexecution_child_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine the failed-child reconstruction with the new diagnostic call."""

    invocation = validate_invocation_receipt(invocation_receipt)
    command = validate_command_receipt(command_receipt)
    child_result = validate_preexecution_child_result(preexecution_child_result)
    if (
        invocation["outer_exact_argv"] != command["outer_exact_argv"]
        or invocation["internal_exact_argv"] != command["internal_exact_argv"]
        or invocation["fd_custody"] != command["fd_custody"]
        or invocation["child_process_identity"]
        != child_result["child_process_identity"]
        or invocation["launcher_process_identity"]
        != child_result["launcher_process_identity"]
    ):
        raise ForensicContractError("tracked invocation cross-binding drift")
    comparison = [
        {
            "field": "purpose_and_scientific_payload",
            "failed_final_child": "SCIENTIFIC_EVALUATOR_BEFORE_ATTEMPT_CREATION",
            "new_diagnostic_child": "PREEXECUTION_ONLY_ZERO_SCIENTIFIC_INPUT",
            "difference": "DIAGNOSTIC_NEVER_ENTERS_SCIENTIFIC_PATH",
        },
        {
            "field": "child_argv",
            "failed_final_child": copy.deepcopy(
                ORIGINAL_FINAL_CHILD_RECONSTRUCTION["child_process_identity"][
                    "argv"
                ]
            ),
            "new_diagnostic_child": copy.deepcopy(command["outer_exact_argv"]),
            "difference": "STDLIB_WRAPPER_AND_DIAGNOSTIC_SUBCOMMAND_ADDED",
        },
        {
            "field": "cwd",
            "failed_final_child": ORIGINAL_FINAL_CHILD_RECONSTRUCTION[
                "committed_launcher_semantics"
            ]["cwd"],
            "new_diagnostic_child": command["cwd"],
            "difference": "NONE_REPOSITORY_ROOT_FOR_BOTH",
        },
        {
            "field": "stdin_stdout_stderr",
            "failed_final_child": {
                "stdin": "INHERITED",
                "stdout": "PIPE",
                "stderr": "STDOUT",
            },
            "new_diagnostic_child": {
                "stdin": command["popen_semantics"]["stdin"],
                "stdout": command["popen_semantics"]["stdout"],
                "stderr": command["popen_semantics"]["stderr"],
            },
            "difference": "DISTINCT_PREOPENED_EXTERNAL_STREAM_CUSTODY",
        },
        {
            "field": "inherited_file_descriptors",
            "failed_final_child": "RETROSPECTIVELY_UNAVAILABLE",
            "new_diagnostic_child": copy.deepcopy(
                command["popen_semantics"]["pass_fds"]
            ),
            "difference": "EXACT_FD_SET_BOUND_ONLY_FOR_NEW_DIAGNOSTIC",
        },
        {
            "field": "environment_and_resource_limits",
            "failed_final_child": "RETROSPECTIVELY_UNAVAILABLE",
            "new_diagnostic_child": {
                "environment": copy.deepcopy(command["popen_environment"]),
                "launcher_runtime_context": copy.deepcopy(
                    invocation["technical_runtime_context"]
                ),
            },
            "difference": "CURRENT_TECHNICAL_CONTEXT_NOT_HISTORICAL_INFERENCE",
        },
        {
            "field": "session",
            "failed_final_child": {
                "start_new_session": True,
                "child_session_and_process_group_equal_pid": True,
            },
            "new_diagnostic_child": {
                "start_new_session": command["popen_semantics"][
                    "start_new_session"
                ],
                "child_session_and_process_group_leader": command[
                    "popen_semantics"
                ]["child_session_and_process_group_leader"],
            },
            "difference": "NONE_IN_SESSION_ISOLATION_SEMANTICS",
        },
    ]
    return attach_self_digest(
        {
            "schema": TRACKED_INVOCATION_COMPARISON_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "lineage": copy.deepcopy(FORENSIC_LINEAGE),
            "original_failed_final_launcher_and_child": copy.deepcopy(
                ORIGINAL_FINAL_CHILD_RECONSTRUCTION
            ),
            "correction_1_direct_outer_comparator": {
                "commit": BASE.INITIAL_EXECUTION_CORRECTION_FREEZE_COMMIT,
                "mode": "SCIENCE_EXECUTED_DIRECTLY_IN_OUTER_EVALUATOR",
                "standalone_isolated_scientific_child_completed": False,
                "exact_expired_process_environment": (
                    "RETROSPECTIVELY_UNAVAILABLE"
                ),
            },
            "correction_2_failed_isolated_path": {
                "commit": SOURCE_COMMIT,
                "mode": "LAUNCHER_TO_ISOLATED_SCIENTIFIC_CHILD",
                "reconstruction": copy.deepcopy(
                    ORIGINAL_FINAL_CHILD_RECONSTRUCTION
                ),
            },
            "test_and_mock_comparator": {
                "synthetic_fixture_ids": list(
                    PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
                ),
                "direct_evaluator_test_invocation": {
                    "path": "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1.py",
                    "call": (
                        "execute_scientific(launcher_pid=123, "
                        "launcher_start_time_ticks=456)"
                    ),
                    "execution_form": "DIRECT_IN_PROCESS_FUNCTION_CALL",
                    "authorities": "MONKEYPATCHED_SYNTHETIC_ONLY",
                    "subprocess_or_exact_argv": False,
                    "real_panel_opened": False,
                },
                "isolated_process_test_invocation": {
                    "path": "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1.py",
                    "runner": "_run_exact_isolated_process",
                    "child": "SYNTHETIC_TEMP_EVALUATOR_SCRIPT",
                    "real_scientific_evaluator": False,
                },
                "scientific_execution": False,
                "previous_successful_isolated_scientific_child": False,
                "comparison_limit": (
                    "NEITHER_TEST_PATH_IS_A_SUCCESSFUL_REAL_ISOLATED_"
                    "SCIENTIFIC_CHILD"
                ),
            },
            "new_preexecution_diagnostic": {
                "invocation": invocation,
                "command": command,
                "child_result_binding": {
                    "content_digest": child_result["content_digest"],
                    "child_process_identity": copy.deepcopy(
                        child_result["child_process_identity"]
                    ),
                    "scientific_counters": copy.deepcopy(
                        child_result["scientific_counters"]
                    ),
                },
            },
            "explicit_field_differences": comparison,
            "historical_unknowns_not_inferred": copy.deepcopy(
                ORIGINAL_FINAL_CHILD_RECONSTRUCTION[
                    "retrospectively_unavailable"
                ]
            ),
            "scientific_payloads_opened_for_comparison": 0,
            "pass": True,
        }
    )


def validate_tracked_invocation_comparison_receipt(
    value: Mapping[str, Any],
    *,
    invocation_receipt: Mapping[str, Any],
    command_receipt: Mapping[str, Any],
    preexecution_child_result: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_tracked_invocation_comparison_receipt(
        invocation_receipt=invocation_receipt,
        command_receipt=command_receipt,
        preexecution_child_result=preexecution_child_result,
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("tracked invocation comparison drift")
    return copy.deepcopy(dict(value))


def build_tracked_os_evidence_receipt(
    *,
    os_evidence_receipt: Mapping[str, Any],
    preexecution_child_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep retrospective incident evidence distinct from current diagnostics."""

    os_evidence = validate_os_evidence_receipt(os_evidence_receipt)
    child_result = validate_preexecution_child_result(preexecution_child_result)
    if (
        os_evidence["child_process_identity"]
        != child_result["child_process_identity"]
    ):
        raise ForensicContractError("combined OS evidence child drift")
    return attach_self_digest(
        {
            "schema": TRACKED_OS_EVIDENCE_RECEIPT_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "historical_final_child_evidence": copy.deepcopy(
                HISTORICAL_OS_EVIDENCE_AUTHORITY
            ),
            "historical_final_child_reconstruction": copy.deepcopy(
                ORIGINAL_FINAL_CHILD_RECONSTRUCTION
            ),
            "current_diagnostic_os_evidence": os_evidence,
            "current_diagnostic_child_runtime_context": copy.deepcopy(
                child_result["current_runtime_context"]
            ),
            "contexts_are_distinct_and_not_cross_inferred": True,
            "absence_of_historical_log_rows_is_not_causal_proof": True,
            "scientific_payloads_opened": 0,
            "pass": True,
        }
    )


def validate_tracked_os_evidence_receipt(
    value: Mapping[str, Any],
    *,
    os_evidence_receipt: Mapping[str, Any],
    preexecution_child_result: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_tracked_os_evidence_receipt(
        os_evidence_receipt=os_evidence_receipt,
        preexecution_child_result=preexecution_child_result,
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("tracked combined OS evidence drift")
    return copy.deepcopy(dict(value))


def build_conditional_v2_conditions(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, bool]:
    """Build the exact boolean V2 gate inputs from technical custody only."""

    custody = validate_diagnostic_custody_receipt(
        diagnostic_custody_receipt
    )
    validate_self_digest(USER_CONDITIONAL_V2_SPEC_AUTHORITY)
    synthetic = validate_synthetic_results_receipt(synthetic_results_receipt)
    if custody["receipts"]["synthetic_results"] != _runtime_json_binding(
        "synthetic_results", synthetic
    ):
        raise ForensicContractError("conditional V2 synthetic binding drift")
    cleanup = custody["cleanup"]
    zero = custody["scientific_counters"] == ZERO_SCIENTIFIC_COUNTERS
    conditions = {
        "forensic_authority_and_source_closure_frozen": True,
        "preexecution_diagnostic_complete": custody["pass"] is True,
        "startup_stage_sequence_complete": (
            custody["last_completed_stage"]
            == PREEXECUTION_DIAGNOSTIC_STAGE_IDS[-1]
        ),
        "invocation_os_stream_and_exception_custody_complete": (
            custody["exception_observed"] is False
            and custody["streams"]["read_guard_events"]["bytes"] == 0
        ),
        "all_synthetic_failure_modes_captured_and_cleaned": (
            synthetic["row_count"]
            == len(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES)
            and synthetic["all_capture_and_cleanup_pass"] is True
        ),
        "exact_technical_root_cause_reproduced": (
            custody["preexecution_child_result"]["mismatch_evidence"][
                "identity_projection_equality"
            ]
            is True
            and custody["preexecution_child_result"]["mismatch_evidence"][
                "raw_record_equality"
            ]
            is False
        ),
        "cause_specific_correction_specified_and_regression_passes": (
            custody["preexecution_child_result"]["mismatch_evidence"][
                "pass"
            ]
            is True
        ),
        "no_scientific_attempt_or_scientific_reservation_namespace_created": (
            custody["attempt_namespace_created"] is False
            and custody["attempt_reservation_created"] is False
        ),
        "no_canonical_or_tracked_scientific_output_written": (
            custody["canonical_or_tracked_written"] is False
        ),
        "zero_scientific_inputs_outcomes_tensors_models_and_training": zero,
        "original_scientific_contract_byte_and_digest_immutable": (
            custody["preexecution_child_result"]["scientific_contract_digest"]
            == BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
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
            USER_CONDITIONAL_V2_SPEC_AUTHORITY[
                "specification_authorized_conditionally"
            ]
            is True
            and USER_CONDITIONAL_V2_SPEC_AUTHORITY["execution_authorized"]
            is False
            and USER_CONDITIONAL_V2_SPEC_AUTHORITY[
                "scientific_attempt_authorized"
            ]
            is False
        ),
    }
    if set(conditions) != set(CONDITIONAL_V2_REQUIRED_CONDITIONS):
        raise ForensicContractError("derived conditional V2 key-set drift")
    return {
        key: conditions[key] for key in CONDITIONAL_V2_REQUIRED_CONDITIONS
    }


def build_conditional_v2_decision(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    conditions = build_conditional_v2_conditions(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    decision = evaluate_conditional_v2_gate(
        root_cause_classification=FORENSIC_PRIMARY_CLASSIFICATION,
        conditions=conditions,
    )
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "conditional_v2_spec_decision.v1"
            ),
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "gate_authority": copy.deepcopy(CONDITIONAL_V2_GATE_BINDING),
            "user_specification_authority": copy.deepcopy(
                USER_CONDITIONAL_V2_SPEC_AUTHORITY
            ),
            "primary_forensic_classification": (
                FORENSIC_PRIMARY_CLASSIFICATION
            ),
            "secondary_mechanism": FORENSIC_SECONDARY_MECHANISM,
            "technical_defect_id": FORENSIC_DEFECT_ID,
            "gate_result": decision,
            "v2_spec_authorized": decision["v2_spec_authorized"],
            "automatic_execution_authorized": False,
            "scientific_contract_change_authorized": False,
            "scientific_classification_authorized": False,
            "pass": True,
        }
    )


def validate_conditional_v2_decision(
    value: Mapping[str, Any],
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    gate = value.get("gate_result")
    if not isinstance(gate, Mapping):
        raise ForensicContractError("conditional V2 decision lacks gate result")
    conditions = gate.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ForensicContractError("conditional V2 decision lacks conditions")
    if conditions.get("explicit_human_or_stakeholder_authorization") is not True:
        raise ForensicContractError("bound user V2 specification authority drift")
    rebuilt = build_conditional_v2_decision(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("conditional V2 decision drift")
    return copy.deepcopy(dict(value))


def build_conditional_v2_spec_authority(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(decision)
    if decision.get("v2_spec_authorized") is not True:
        raise ForensicContractError(
            "conditional V2 specification cannot exist while gate is false"
        )
    clauses = [
        {
            "id": "PRESERVE_V1_SCIENTIFIC_DESIGN",
            "requirement": (
                "Preserve the V1 scientific design unless a separate change is "
                "prospectively justified and explicitly approved."
            ),
        },
        {
            "id": "IMMUTABLE_MACHINE_READABLE_SCIENTIFIC_PAYLOAD",
            "requirement": (
                "After Stage A, Stage B, and the frozen conditional Stage-C "
                "disposition (entering Stage C only when its gate authorizes), "
                "freeze an "
                "immutable machine-readable payload binding checkpoint-state "
                "digests, raw score ledgers, candidate selections, metrics, gate "
                "decisions, and stage disposition."
            ),
        },
        {
            "id": "INDEPENDENT_PAYLOAD_VALIDATION_COMPLETES_SCIENCE",
            "requirement": (
                "Independently validate the immutable payload before narrative "
                "or report generation; that validation completes the scientific "
                "attempt."
            ),
        },
        {
            "id": "SEPARATE_PUBLICATION_ATTEMPT",
            "requirement": (
                "Presentation, Markdown, tracked JSON, and Git publication form "
                "a separate publication attempt. A publication correction may "
                "reuse only the exact independently validated immutable payload, "
                "with no retraining, predictor or model inference, metric "
                "recomputation, or new scientific attempt."
            ),
        },
        {
            "id": "THREE_ATTEMPT_BOUNDARIES",
            "requirement": (
                "Define technical_startup_attempt, scientific_attempt, and "
                "publication_attempt separately. Science begins only after "
                "PREEXECUTION passes and the first scientific input or model "
                "state opens; pre-boundary failure does not consume science."
            ),
        },
        {
            "id": "MANDATORY_EXTERNAL_CHILD_CUSTODY",
            "requirement": (
                "Retain external stdout, stderr, traceback, exact command, "
                "environment custody, and an atomic last-stage marker for every "
                "child."
            ),
        },
        {
            "id": "FAILED_ARCHIVE_NONREUSE_AND_STATIC_INPUT_DISTINCTION",
            "requirement": (
                "All three failed archives are nonreusable: no failed checkpoint, "
                "score, selection, metric, or report may be reused. Independently "
                "frozen static panel, split, route labels, true-future tensors, "
                "predictor tensors, and hyperparameters may remain inputs."
            ),
        },
        {
            "id": "DEVELOPMENT_ONLY_OUTCOME_OBSERVED_BOUNDARY",
            "requirement": (
                "The V1 development panel is already outcome-observed. V2 remains "
                "development-only and non-claim-bearing."
            ),
        },
        {
            "id": "EXACT_TECHNICAL_CORRECTION_ONLY",
            "requirement": (
                "Replace incompatible whole-record archive-custody equality with "
                "the stable archive-identity projection. Models, targets, metrics, "
                "and gates remain unchanged."
            ),
        },
        {
            "id": "SPECIFICATION_ONLY_NO_EXECUTION",
            "requirement": (
                "This is a specification only. No V2 execution or scientific "
                "attempt is authorized now."
            ),
        },
    ]
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v2.technical_specification.v1"
            ),
            "source_experiment": FORENSIC_EXPERIMENT_ID,
            "gate_decision": copy.deepcopy(dict(decision)),
            "user_specification_authority": copy.deepcopy(
                USER_CONDITIONAL_V2_SPEC_AUTHORITY
            ),
            "primary_forensic_classification": (
                FORENSIC_PRIMARY_CLASSIFICATION
            ),
            "secondary_mechanism": FORENSIC_SECONDARY_MECHANISM,
            "technical_defect_id": FORENSIC_DEFECT_ID,
            "clauses": clauses,
            "clause_ids": [row["id"] for row in clauses],
            "preserved_v1_scientific_dimensions": [
                "scientific_contract_and_output_schema",
                "model_architecture_and_initialization",
                "optimizer_seed_epochs_and_loss",
                "residual_weights",
                "development_panel_splits_waypoints_and_route_target",
                "predictor_and_latent_bindings",
                "metrics_estimators_thresholds_and_gates",
                "scientific_classification_vocabularies_and_logic",
                "evidence_cardinalities",
                "scientific_prohibitions",
            ],
            "conditional_stage_c_semantics": (
                "STAGE_C_ENTERED_ONLY_IF_FROZEN_GATE_AUTHORIZES_OTHERWISE_"
                "DISPOSITION_IS_NOT_ENTERED"
            ),
            "execution_authorized": False,
            "scientific_attempt_authorized": False,
            "automatic_execution": False,
        }
    )


def validate_conditional_v2_spec_authority(
    value: Mapping[str, Any], *, decision: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_conditional_v2_spec_authority(decision)
    if dict(value) != rebuilt:
        raise ForensicContractError("conditional V2 specification authority drift")
    return copy.deepcopy(dict(value))


def build_conditional_v2_spec_markdown(
    specification: Mapping[str, Any],
) -> str:
    validate_self_digest(specification)
    if (
        specification.get("schema")
        != "plan_aware_monotone_jepa_cost_v2.technical_specification.v1"
        or specification.get("execution_authorized") is not False
        or specification.get("scientific_attempt_authorized") is not False
    ):
        raise ForensicContractError("conditional V2 Markdown input drift")
    lines = [
        "# Conditional plan-aware JEPA V2 technical specification",
        "",
        "Status: specification authorized by the forensic gate; execution and a scientific attempt are not authorized.",
        "",
        f"Primary forensic classification: `{FORENSIC_PRIMARY_CLASSIFICATION}`.",
        f"Secondary mechanism: `{FORENSIC_SECONDARY_MECHANISM}`.",
        f"Defect: `{FORENSIC_DEFECT_ID}`.",
        "",
    ]
    for index, clause in enumerate(specification["clauses"], start=1):
        lines.extend(
            [
                f"## {index}. {clause['id']}",
                "",
                str(clause["requirement"]),
                "",
            ]
        )
    return "\n".join(lines)


def validate_conditional_v2_spec_markdown(
    value: str, *, specification: Mapping[str, Any]
) -> str:
    expected = build_conditional_v2_spec_markdown(specification)
    if value != expected:
        raise ForensicContractError("conditional V2 Markdown projection drift")
    return value


def _tracked_json_binding(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    payload = canonical_json_bytes(value) + b"\n"
    return _artifact_binding(
        str(path), payload, content_digest=value["content_digest"]
    )


def build_forensic_result(
    *,
    source_freeze_commit: str,
    invocation_comparison_receipt: Mapping[str, Any],
    combined_os_evidence_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    preexecution_only_receipt: Mapping[str, Any],
    diagnostic_custody_receipt: Mapping[str, Any],
    conditional_v2_decision: Mapping[str, Any],
    conditional_v2_spec_authority: Mapping[str, Any],
    conditional_v2_spec_markdown: str,
) -> dict[str, Any]:
    """Build the technical forensic result; it is explicitly not science."""

    invocation = copy.deepcopy(dict(invocation_comparison_receipt))
    combined_os = copy.deepcopy(dict(combined_os_evidence_receipt))
    synthetic = validate_synthetic_results_receipt(synthetic_results_receipt)
    preexecution = validate_preexecution_only_receipt(
        preexecution_only_receipt
    )
    custody = validate_diagnostic_custody_receipt(
        diagnostic_custody_receipt
    )
    decision = validate_conditional_v2_decision(
        conditional_v2_decision,
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    specification = validate_conditional_v2_spec_authority(
        conditional_v2_spec_authority, decision=decision
    )
    specification_markdown = validate_conditional_v2_spec_markdown(
        conditional_v2_spec_markdown, specification=specification
    )
    validate_tracked_invocation_comparison_receipt(
        invocation,
        invocation_receipt=invocation["new_preexecution_diagnostic"][
            "invocation"
        ],
        command_receipt=invocation["new_preexecution_diagnostic"]["command"],
        preexecution_child_result=custody["preexecution_child_result"],
    )
    validate_tracked_os_evidence_receipt(
        combined_os,
        os_evidence_receipt=combined_os["current_diagnostic_os_evidence"],
        preexecution_child_result=custody["preexecution_child_result"],
    )
    if (
        not isinstance(source_freeze_commit, str)
        or len(source_freeze_commit) != 40
        or source_freeze_commit != custody["source_commit"]
        or custody["receipts"]["preexecution_only"]
        != _runtime_json_binding("preexecution_only", preexecution)
        or custody["receipts"]["synthetic_results"]
        != _runtime_json_binding("synthetic_results", synthetic)
        or custody["receipts"]["invocation"]
        != _runtime_json_binding(
            "invocation",
            invocation["new_preexecution_diagnostic"]["invocation"],
        )
        or custody["receipts"]["command"]
        != _runtime_json_binding(
            "command",
            invocation["new_preexecution_diagnostic"]["command"],
        )
        or custody["receipts"]["os_evidence"]
        != _runtime_json_binding(
            "os_evidence", combined_os["current_diagnostic_os_evidence"]
        )
        or decision["primary_forensic_classification"]
        != FORENSIC_PRIMARY_CLASSIFICATION
        or decision["secondary_mechanism"]
        != FORENSIC_SECONDARY_MECHANISM
        or decision["technical_defect_id"] != FORENSIC_DEFECT_ID
    ):
        raise ForensicContractError("forensic result input custody drift")
    receipt_bindings = {
        "invocation_comparison": _tracked_json_binding(
            TRACKED_INVOCATION_RECEIPT_PATH, invocation
        ),
        "combined_os_evidence": _tracked_json_binding(
            TRACKED_OS_EVIDENCE_RECEIPT_PATH, combined_os
        ),
        "synthetic_results": _tracked_json_binding(
            TRACKED_SYNTHETIC_RESULTS_PATH, synthetic
        ),
        "preexecution_only": _tracked_json_binding(
            TRACKED_PREEXECUTION_ONLY_RECEIPT_PATH, preexecution
        ),
        "diagnostic_custody": _tracked_json_binding(
            TRACKED_DIAGNOSTIC_CUSTODY_RECEIPT_PATH, custody
        ),
        "conditional_v2_decision": _tracked_json_binding(
            TRACKED_CONDITIONAL_V2_DECISION_PATH, decision
        ),
        "conditional_v2_specification_authority": _tracked_json_binding(
            TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH, specification
        ),
        "conditional_v2_specification_markdown": _artifact_binding(
            str(TRACKED_CONDITIONAL_V2_SPEC_PATH),
            specification_markdown.encode("utf-8"),
        ),
    }
    child_result = custody["preexecution_child_result"]
    return attach_self_digest(
        {
            "schema": FORENSIC_RESULT_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "date": DATE,
            "lineage": {
                **copy.deepcopy(FORENSIC_LINEAGE),
                "forensic_diagnostic_freeze_commit": source_freeze_commit,
                "forensic_result_commit": None,
                "forensic_result_commit_subject": (
                    FORENSIC_RESULT_COMMIT_SUBJECT
                ),
                "result_commit_self_binding": (
                    "EXTERNAL_POST_COMMIT_VALIDATION_NOT_SELF_EMBEDDED"
                ),
            },
            "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
            "source_freeze_commit": source_freeze_commit,
            "forensic_classification": FORENSIC_PRIMARY_CLASSIFICATION,
            "forensic_secondary_mechanism": FORENSIC_SECONDARY_MECHANISM,
            "technical_defect_id": FORENSIC_DEFECT_ID,
            "scientific_disposition": SCIENTIFIC_DISPOSITION,
            "scientific_result": False,
            "scientific_classification_authorized": False,
            "scientific_classification": None,
            "scientific_metrics": None,
            "scientific_claims": [],
            "technical_result": True,
            "root_cause_evidence": {
                "mismatch_evidence": copy.deepcopy(
                    child_result["mismatch_evidence"]
                ),
                "committed_source_root_cause_proof": copy.deepcopy(
                    child_result["committed_source_root_cause_proof"]
                ),
                "full_inventory_verified_is_one_detail_not_sole_cause": True,
            },
            "archive_inventory": copy.deepcopy(
                ARCHIVE_INVENTORY_AUTHORITY
            ),
            "archive_validation_custody": {
                "distinct_admitted_technical_paths": 2,
                "validation_phases": 2,
                "technical_file_open_operations": 4,
                "scientific_archive_payload_file_opens": 0,
                "files_reused": 0,
            },
            "attempt_accounting": copy.deepcopy(ATTEMPT_ACCOUNTING_POLICY),
            "diagnostic": {
                "mode": "PREEXECUTION_ONLY_DIAGNOSTIC",
                "root": custody["diagnostic_root"],
                "launcher_process_identity": copy.deepcopy(
                    custody["launcher_process_identity"]
                ),
                "child_process_identity": copy.deepcopy(
                    custody["child_process_identity"]
                ),
                "launcher_technical_runtime_context": copy.deepcopy(
                    custody["launcher_technical_runtime_context"]
                ),
                "child_technical_runtime_context": copy.deepcopy(
                    custody["child_technical_runtime_context"]
                ),
                "returncode": custody["returncode"],
                "termination": copy.deepcopy(custody["termination"]),
                "cleanup": copy.deepcopy(custody["cleanup"]),
                "process_state_scope_at_publication": copy.deepcopy(
                    custody["process_state_scope_at_custody_write"]
                ),
                "launcher_live_during_tracked_publication": True,
                "literal_zero_all_forensic_roles_claimed": False,
                "all_forensic_role_zero_validation": (
                    "PENDING_POSTCOMMIT_VALIDATOR_AFTER_LAUNCHER_EXIT"
                ),
                "startup_stages_completed": list(
                    PREEXECUTION_DIAGNOSTIC_STAGE_IDS
                ),
                "synthetic_fixture_ids": list(
                    PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
                ),
                "scientific_counters": copy.deepcopy(
                    custody["scientific_counters"]
                ),
                "python_open_guard_events": custody["streams"][
                    "read_guard_events"
                ]["bytes"],
                "python_audit_hook_boundary": (
                    "PYTHON_OPEN_EVENTS_ONLY_NOT_KERNEL_OR_NATIVE_SYSCALL_SANDBOX"
                ),
                "final_namespace_inventory_required": True,
                "final_namespace_inventory_self_excludes_only_its_own_bytes_and_inode": True,
            },
            "tracked_receipts": receipt_bindings,
            "conditional_v2": decision,
            "conditional_v2_specification": {
                "authority": receipt_bindings[
                    "conditional_v2_specification_authority"
                ],
                "markdown": receipt_bindings[
                    "conditional_v2_specification_markdown"
                ],
                "bound_in_authorized_fail_closed_publication_set": True,
                "publication_pending_until_writer_completes": True,
            },
            "automatic_execution_authorized": False,
            "no_further_retry_authority_changed": False,
            "claims_boundary": {
                "scientific_or_metric_values_inspected_or_interpreted": False,
                "scientific_payloads_opened": 0,
                "outcome_rows_opened": 0,
                "tensor_reads": 0,
                "model_loads": 0,
                "training_steps": 0,
                "python_open_guard_events": 0,
                "universal_os_level_read_prevention_claimed": False,
            },
            "pass_meaning": (
                "TECHNICAL_FAILURE_ROOT_CAUSE_AND_CUSTODY_COMPLETE_"
                "NOT_A_SCIENTIFIC_RESULT"
            ),
            "pass": True,
        }
    )


def validate_forensic_result(
    value: Mapping[str, Any],
    *,
    source_freeze_commit: str,
    invocation_comparison_receipt: Mapping[str, Any],
    combined_os_evidence_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    preexecution_only_receipt: Mapping[str, Any],
    diagnostic_custody_receipt: Mapping[str, Any],
    conditional_v2_decision: Mapping[str, Any],
    conditional_v2_spec_authority: Mapping[str, Any],
    conditional_v2_spec_markdown: str,
) -> dict[str, Any]:
    validate_self_digest(value)
    rebuilt = build_forensic_result(
        source_freeze_commit=source_freeze_commit,
        invocation_comparison_receipt=invocation_comparison_receipt,
        combined_os_evidence_receipt=combined_os_evidence_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
        preexecution_only_receipt=preexecution_only_receipt,
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        conditional_v2_decision=conditional_v2_decision,
        conditional_v2_spec_authority=conditional_v2_spec_authority,
        conditional_v2_spec_markdown=conditional_v2_spec_markdown,
    )
    if dict(value) != rebuilt:
        raise ForensicContractError("forensic result drift")
    return copy.deepcopy(dict(value))


def build_forensic_report_markdown(result: Mapping[str, Any]) -> str:
    validate_self_digest(result)
    if (
        result.get("schema") != FORENSIC_RESULT_SCHEMA
        or result.get("scientific_disposition") != SCIENTIFIC_DISPOSITION
        or result.get("scientific_result") is not False
        or result.get("scientific_classification_authorized") is not False
        or result.get("scientific_classification") is not None
    ):
        raise ForensicContractError("forensic report input is not a non-result")
    lines = [
        "# Plan-aware JEPA failure forensics",
        "",
        f"Technical disposition: `{SCIENTIFIC_DISPOSITION}`.",
        "",
        "This is a technical failure-forensics result. It is not a scientific result, reports no scientific metric, and authorizes no scientific classification or automatic retry.",
        "",
        "## Root cause",
        "",
        f"- Primary forensic classification: `{FORENSIC_PRIMARY_CLASSIFICATION}`.",
        f"- Secondary mechanism: `{FORENSIC_SECONDARY_MECHANISM}`.",
        f"- Exact defect: `{FORENSIC_DEFECT_ID}`.",
        "- The pre-attempt evaluator compared minimal runtime failed-archive custody rows against detailed validator rows as complete records. Their schemas differ in multiple fields; `full_inventory_verified=False` is one difference, not the sole cause.",
        "",
        "## Lineage",
        "",
        f"- Scientific source: `{result['lineage']['scientific_source_commit']}`.",
        f"- Scientific contract freeze: `{result['lineage']['scientific_contract_freeze_commit']}`.",
        f"- Execution correction 1: `{result['lineage']['execution_correction_1_commit']}`.",
        f"- Execution correction 2 / forensic base: `{result['lineage']['execution_correction_2_base_commit']}`.",
        f"- Diagnostic freeze: `{result['lineage']['forensic_diagnostic_freeze_commit']}`.",
        f"- Result commit subject: `{FORENSIC_RESULT_COMMIT_SUBJECT}`; the hash is validated after commit and is not self-embedded.",
        "",
        "## Failed-archive custody",
        "",
        "| Incident | Archive | Stage | Process IDs | Return code | Calibration / heldout rows | Scientific payload | Reused files |",
        "|---:|---|---|---|---|---:|---|---:|",
    ]
    for row in result["archive_inventory"]["archives"]:
        incident = row["incident_accounting"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ordinal"]),
                    f"`{row['archive_path']}`",
                    str(incident["stage_reached"]),
                    ", ".join(
                        f"{key}={value}"
                        for key, value in incident["process_ids"].items()
                    ),
                    str(incident["returncode"]),
                    (
                        f"{incident['rows_opened']['calibration']} / "
                        f"{incident['rows_opened']['heldout']}"
                    ),
                    "yes" if incident["scientific_payload_exists"] else "no",
                    str(row["files_reused"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The first two archives were checked by frozen content manifests plus a no-open filesystem-stat inventory. The third archive admitted only its two exact technical receipts. Across the prelaunch and active-child phases that is two distinct paths, four technical file-open operations, zero scientific archive-payload opens, and zero reused files.",
            "",
            "## Invocation and OS evidence",
            "",
            "The tracked invocation receipt reconstructs the failed final launcher/child, the correction-1 direct outer invocation, the correction-2 isolated path, test/mock comparators, and the new diagnostic call. Historical UID/GID/groups/umask, environment, inherited FDs, resource limits, GPU visibility, and temporary-root context remain retrospectively unavailable where the original receipts did not persist them.",
            "",
            "Historical incident log queries and later/current runtime observations are separate. Zero journal or kernel rows in the query window is not proof that no environmental failure occurred; current cgroup and diagnostic process state is not attributed to the expired child.",
            "",
            "## Diagnostic custody",
            "",
            f"- All {len(PREEXECUTION_DIAGNOSTIC_STAGE_IDS)} PREEXECUTION stages completed.",
            f"- All {len(PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES)} synthetic fixtures produced externally retained stdout, stderr, traceback, exception, heartbeat, read-guard, cleanup, and atomic last-stage custody.",
            "- The real diagnostic created no scientific attempt or reservation namespace and opened no scientific input, outcome row, tensor, or model.",
            "- The read guard covers Python audit-hook open events. It is not claimed to be a kernel/native-I/O sandbox; source-byte/AST custody and zero guard events supplement that boundary.",
            "- The terminal namespace is validated with a pre-read lstat inventory, no-follow per-file reads, and a post-read inventory. Its dedicated witness excludes only its own bytes and inode to avoid self-reference.",
            "- Child-process-group and nonlauncher forensic roles are quiescent. The diagnostic launcher is intentionally still live while it publishes this tracked evidence; literal zero across all forensic roles is required only after launcher exit by the post-commit validator.",
            "",
            "## Conditional V2 gate",
            "",
            f"- V2 specification authorized: `{str(result['conditional_v2']['v2_spec_authorized']).lower()}`.",
            "- Automatic execution authorized: `false`.",
            "- The existing no-further-retry authority is unchanged.",
            "- A V2 specification Markdown file exists only if every frozen condition, including explicit human or stakeholder authorization, is literally true.",
            "",
        ]
    )
    return "\n".join(lines)


def build_forensic_result_artifacts(
    *,
    invocation_receipt: Mapping[str, Any],
    command_receipt: Mapping[str, Any],
    os_evidence_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    preexecution_only_receipt: Mapping[str, Any],
    diagnostic_custody_receipt: Mapping[str, Any],
    preexecution_child_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all external/tracked result bytes without writing any path."""

    custody = validate_diagnostic_custody_receipt(
        diagnostic_custody_receipt
    )
    child_result = validate_preexecution_child_result(
        preexecution_child_result
    )
    if custody["preexecution_child_result"] != child_result:
        raise ForensicContractError("result child/custody cross-binding drift")
    invocation_comparison = build_tracked_invocation_comparison_receipt(
        invocation_receipt=invocation_receipt,
        command_receipt=command_receipt,
        preexecution_child_result=child_result,
    )
    combined_os = build_tracked_os_evidence_receipt(
        os_evidence_receipt=os_evidence_receipt,
        preexecution_child_result=child_result,
    )
    synthetic = validate_synthetic_results_receipt(synthetic_results_receipt)
    preexecution = validate_preexecution_only_receipt(
        preexecution_only_receipt
    )
    decision = build_conditional_v2_decision(
        diagnostic_custody_receipt=custody,
        synthetic_results_receipt=synthetic,
    )
    specification = build_conditional_v2_spec_authority(decision)
    conditional_spec = build_conditional_v2_spec_markdown(specification)
    result = build_forensic_result(
        source_freeze_commit=custody["source_commit"],
        invocation_comparison_receipt=invocation_comparison,
        combined_os_evidence_receipt=combined_os,
        synthetic_results_receipt=synthetic,
        preexecution_only_receipt=preexecution,
        diagnostic_custody_receipt=custody,
        conditional_v2_decision=decision,
        conditional_v2_spec_authority=specification,
        conditional_v2_spec_markdown=conditional_spec,
    )
    report = build_forensic_report_markdown(result)
    values: dict[Path, Mapping[str, Any] | str] = {
        TRACKED_INVOCATION_RECEIPT_PATH: invocation_comparison,
        TRACKED_OS_EVIDENCE_RECEIPT_PATH: combined_os,
        TRACKED_SYNTHETIC_RESULTS_PATH: synthetic,
        TRACKED_PREEXECUTION_ONLY_RECEIPT_PATH: preexecution,
        TRACKED_DIAGNOSTIC_CUSTODY_RECEIPT_PATH: custody,
        TRACKED_CONDITIONAL_V2_DECISION_PATH: decision,
        TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH: specification,
        TRACKED_CONDITIONAL_V2_SPEC_PATH: conditional_spec,
        TRACKED_FORENSIC_RESULT_PATH: result,
        TRACKED_FORENSIC_REPORT_PATH: report,
    }
    payloads = {
        path: (
            canonical_json_bytes(value) + b"\n"
            if isinstance(value, Mapping)
            else value.encode("utf-8")
        )
        for path, value in values.items()
    }
    return {
        "invocation_comparison": invocation_comparison,
        "combined_os_evidence": combined_os,
        "synthetic_results": synthetic,
        "preexecution_only": preexecution,
        "diagnostic_custody": custody,
        "conditional_v2_decision": decision,
        "conditional_v2_spec_authority": specification,
        "conditional_v2_spec_markdown": conditional_spec,
        "result": result,
        "report_markdown": report,
        "tracked_payloads": payloads,
    }


def build_forensic_result_artifacts_from_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "invocation",
        "command",
        "os_evidence",
        "synthetic_results",
        "preexecution_only",
        "preexecution_child_result",
        "diagnostic_custody",
    }
    if not required.issubset(bundle):
        raise ForensicContractError("terminal diagnostic bundle is incomplete")
    return build_forensic_result_artifacts(
        invocation_receipt=bundle["invocation"],
        command_receipt=bundle["command"],
        os_evidence_receipt=bundle["os_evidence"],
        synthetic_results_receipt=bundle["synthetic_results"],
        preexecution_only_receipt=bundle["preexecution_only"],
        diagnostic_custody_receipt=bundle["diagnostic_custody"],
        preexecution_child_result=bundle["preexecution_child_result"],
    )


def write_forensic_result_artifacts(
    *,
    repo_root: str | Path,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Publish tracked technical evidence fail-closed, with all-or-absent rollback."""

    repo = Path(repo_root).resolve()
    bundle = load_and_validate_diagnostic_bundle(
        repo_root=repo, diagnostic_root=diagnostic_root
    )
    artifacts = build_forensic_result_artifacts_from_bundle(bundle)
    payloads: Mapping[Path, bytes] = artifacts["tracked_payloads"]
    expected_paths = set(_forensic_result_paths())
    if set(payloads) != expected_paths:
        raise ForensicContractError("tracked forensic result path-set drift")
    stale = [
        str(path)
        for path in sorted(expected_paths, key=str)
        if (repo / path).exists() or (repo / path).is_symlink()
    ]
    if stale:
        raise ForensicContractError(
            f"tracked forensic result paths are stale: {stale}"
        )
    publication_order = [
        path
        for path in sorted(expected_paths, key=str)
        if path not in {TRACKED_FORENSIC_RESULT_PATH, TRACKED_FORENSIC_REPORT_PATH}
    ] + [TRACKED_FORENSIC_RESULT_PATH, TRACKED_FORENSIC_REPORT_PATH]
    observed_bindings: dict[str, Any] = {}
    try:
        for relative in publication_order:
            _exclusive_write(repo / relative, payloads[relative])
        # Verification is part of the same rollback scope as publication.  This
        # is deliberately described as fail-closed all-or-absent publication,
        # not as an atomic multi-file filesystem transaction.
        for relative in publication_order:
            observed = _read_no_follow_bound_file(
                repo, relative, expected_stat_rows=None
            )
            if observed != payloads[relative]:
                raise ForensicContractError(
                    f"tracked forensic result byte drift: {relative}"
                )
            observed_bindings[str(relative)] = _artifact_binding(
                str(relative), observed
            )
    except BaseException as original:
        try:
            _rollback_relative_paths_no_follow(
                repo,
                publication_order,
                label="tracked forensic result publication",
            )
        except BaseException as cleanup_exc:
            raise ForensicContractError(
                "tracked forensic result publication failed and rollback did "
                "not durably prove the full intended path set absent: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            ) from original
        raise
    return {
        "result": artifacts["result"],
        "report_markdown": artifacts["report_markdown"],
        "conditional_v2_spec_authority": artifacts[
            "conditional_v2_spec_authority"
        ],
        "conditional_v2_spec_markdown": artifacts[
            "conditional_v2_spec_markdown"
        ],
        "tracked_bindings": observed_bindings,
        "tracked_paths": [str(path) for path in publication_order],
        "publication_semantics": "FAIL_CLOSED_ALL_OR_ABSENT_WITH_ROLLBACK",
        "v2_spec_written": True,
        "automatic_execution_authorized": False,
        "pass": True,
    }


FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS = (
    BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING["path"],
    *FORENSIC_CODE_AND_TEST_PATHS,
    str(TRACKED_ATTEMPT_ACCOUNTING_POLICY_PATH),
    str(TRACKED_FORENSIC_CONTRACT_PATH),
    str(TRACKED_ARCHIVE_INVENTORY_PATH),
    str(TRACKED_OS_EVIDENCE_SCHEMA_PATH),
    str(TRACKED_CONDITIONAL_V2_GATE_PATH),
    str(TRACKED_FORENSIC_FIXTURE_PATH),
)


def _assemble_forensic_source_closure(
    *,
    selected: Sequence[Path],
    rows: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
) -> dict[str, Any]:
    return attach_self_digest(
        {
            "schema": FORENSIC_SOURCE_CLOSURE_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "forensic_base_source_commit": SOURCE_COMMIT,
            "scientific_contract_freeze_commit": (
                BASE.INITIAL_EXECUTION_FREEZE_COMMIT
            ),
            "scientific_authority_contract_sha256": (
                BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "base_execution_correction_2_source_closure": copy.deepcopy(
                BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING
            ),
            "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
            "declared_paths": [str(path) for path in selected],
            "rows": [copy.deepcopy(dict(row)) for row in rows],
            "row_count": len(rows),
            "missing_paths": list(missing),
            "complete": not missing,
            "self_path_excluded_to_avoid_circular_digest": str(
                TRACKED_FORENSIC_SOURCE_CLOSURE_PATH
            ),
            "archive_custody": {
                "prior_scientific_archives_opened": False,
                "prior_bound_manifests_reused_as_metadata_authority": True,
                "third_incident_two_technical_receipts_only": True,
                "scientific_archive_payload_values_inspected": False,
            },
            "scientific_value_handling": {
                "outcome_metric_or_tensor_values_inspected": False,
                "outcome_metric_or_tensor_values_interpreted": False,
                "outcome_informed_scientific_change": False,
            },
            "diagnostic_freeze_commit_binding_policy": (
                "bound by the enclosing Git diagnostic-freeze commit after byte "
                "finalization; no circular future commit hash is embedded"
            ),
        }
    )


def build_forensic_source_closure(
    repo_root: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    selected = [
        Path(value)
        for value in (
            FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS if paths is None else paths
        )
    ]
    if len(selected) != len({str(path) for path in selected}):
        raise ForensicContractError("forensic source closure repeats a path")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in selected:
        if relative.is_absolute() or ".." in relative.parts:
            raise ForensicContractError(
                "forensic source-closure paths must be repository-relative"
            )
        try:
            payload = _read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
        except FileNotFoundError:
            missing.append(str(relative))
            continue
        rows.append(
            {
                "path": str(relative),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    if require_complete and missing:
        raise ForensicContractError(
            f"forensic source closure is incomplete: {missing}"
        )
    return _assemble_forensic_source_closure(
        selected=selected, rows=rows, missing=missing
    )


def build_prospective_forensic_source_closure(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the exact complete closure using in-memory authority payloads."""

    root = Path(repo_root).resolve()
    selected = [Path(value) for value in FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS]
    virtual = {
        str(path): payload for path, payload in forensic_authority_payloads().items()
    }
    rows: list[dict[str, Any]] = []
    for relative in selected:
        if relative.is_absolute() or ".." in relative.parts:
            raise ForensicContractError(
                "forensic source-closure paths must be repository-relative"
            )
        payload = virtual.get(str(relative))
        if payload is not None:
            rows.append(
                {
                    "path": str(relative),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
            continue
        try:
            payload = _read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            )
        except FileNotFoundError as exc:
            raise ForensicContractError(
                f"prospective forensic closure source absent: {relative}"
            ) from exc
        rows.append(
            {
                "path": str(relative),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    value = _assemble_forensic_source_closure(
        selected=selected, rows=rows, missing=[]
    )
    validate_forensic_source_closure(value, require_complete=True)
    return value


def validate_forensic_source_closure(
    value: Mapping[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    validate_self_digest(value)
    declared = value.get("declared_paths")
    rows = value.get("rows")
    missing = value.get("missing_paths")
    if (
        value.get("schema") != FORENSIC_SOURCE_CLOSURE_SCHEMA
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or value.get("forensic_base_source_commit") != SOURCE_COMMIT
        or value.get("scientific_contract_freeze_commit")
        != BASE.INITIAL_EXECUTION_FREEZE_COMMIT
        or value.get("scientific_authority_contract_sha256")
        != BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or value.get("base_execution_correction_2_source_closure")
        != BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING
        or value.get("forensic_contract") != FORENSIC_CONTRACT_BINDING
        or declared != list(FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS)
        or len(declared) != len(set(declared))
        or not isinstance(rows, list)
        or value.get("row_count") != len(rows)
        or [row.get("path") for row in rows]
        != [path for path in declared if path not in set(missing or [])]
        or not isinstance(missing, list)
    ):
        raise ForensicContractError("forensic source closure structural drift")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "bytes"}
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
        ):
            raise ForensicContractError("forensic source closure row drift")
    if require_complete and (
        value.get("complete") is not True or missing != []
    ):
        raise ForensicContractError("forensic source closure is not complete")
    return copy.deepcopy(dict(value))


def forensic_source_closure_bytes(value: Mapping[str, Any]) -> bytes:
    validate_forensic_source_closure(value, require_complete=True)
    return canonical_json_bytes(value) + b"\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.tmp-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exclusive_write(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    """Create, fsync, and publish one leaf or durably remove it on failure."""

    absolute = Path(path).absolute()
    absolute.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = _open_absolute_directory_no_follow(absolute.parent)
    leaf_fd: int | None = None
    created = False
    try:
        try:
            os.stat(absolute.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ForensicContractError(
                f"forensic artifact cannot overwrite existing path: {absolute}"
            )
        try:
            leaf_fd = os.open(
                absolute.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                mode,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise ForensicContractError(
                f"forensic artifact cannot overwrite existing path: {absolute}"
            ) from exc
        created = True
        os.fchmod(leaf_fd, mode)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(leaf_fd, remaining)
            if written <= 0:
                raise ForensicContractError("forensic exclusive write stalled")
            remaining = remaining[written:]
        os.fsync(leaf_fd)
        observed = os.fstat(leaf_fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != mode
            or observed.st_size != len(payload)
        ):
            raise ForensicContractError("forensic exclusive-write leaf drift")
        os.close(leaf_fd)
        leaf_fd = None
        os.fsync(directory_fd)
    except BaseException as original:
        cleanup_failures: list[str] = []
        if leaf_fd is not None:
            try:
                os.close(leaf_fd)
            except BaseException as exc:
                cleanup_failures.append(
                    f"leaf close failed: {type(exc).__name__}: {exc}"
                )
            leaf_fd = None
        if created:
            try:
                os.unlink(absolute.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_failures.append(
                    f"leaf unlink failed: {type(exc).__name__}: {exc}"
                )
            try:
                os.fsync(directory_fd)
            except BaseException as exc:
                cleanup_failures.append(
                    f"parent fsync after cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            try:
                os.stat(
                    absolute.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_failures.append(
                    f"leaf absence proof failed: {type(exc).__name__}: {exc}"
                )
            else:
                cleanup_failures.append("leaf remains after exclusive-write failure")
        if cleanup_failures:
            raise ForensicContractError(
                "forensic exclusive-write cleanup failed: "
                f"{cleanup_failures}"
            ) from original
        raise
    finally:
        try:
            if leaf_fd is not None:
                os.close(leaf_fd)
        finally:
            os.close(directory_fd)


def forensic_authority_payloads() -> dict[Path, bytes]:
    return {
        TRACKED_ATTEMPT_ACCOUNTING_POLICY_PATH: ATTEMPT_ACCOUNTING_POLICY_BYTES,
        TRACKED_FORENSIC_CONTRACT_PATH: FORENSIC_CONTRACT_BYTES,
        TRACKED_ARCHIVE_INVENTORY_PATH: ARCHIVE_INVENTORY_AUTHORITY_BYTES,
        TRACKED_OS_EVIDENCE_SCHEMA_PATH: OS_EVIDENCE_SCHEMA_BYTES,
        TRACKED_CONDITIONAL_V2_GATE_PATH: CONDITIONAL_V2_GATE_BYTES,
        TRACKED_FORENSIC_FIXTURE_PATH: FORENSIC_EVALUATOR_FIXTURE_BYTES,
    }


def write_forensic_authorities(repo_root: str | Path) -> dict[str, Any]:
    """Write prospective authorities fail-closed with all-or-absent rollback."""

    root = Path(repo_root).resolve()
    authority_payloads = forensic_authority_payloads()
    all_paths = [*authority_payloads, TRACKED_FORENSIC_SOURCE_CLOSURE_PATH]
    stale = [
        str(path)
        for path in all_paths
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if stale:
        raise ForensicContractError(
            f"forensic authority writer refuses stale paths: {stale}"
        )
    try:
        for relative, payload in authority_payloads.items():
            _exclusive_write(root / relative, payload)
        closure = build_forensic_source_closure(root, require_complete=True)
        closure_payload = forensic_source_closure_bytes(closure)
        _exclusive_write(
            root / TRACKED_FORENSIC_SOURCE_CLOSURE_PATH, closure_payload
        )
        expected_payloads = {
            **authority_payloads,
            TRACKED_FORENSIC_SOURCE_CLOSURE_PATH: closure_payload,
        }
        for relative, expected in expected_payloads.items():
            if _read_no_follow_bound_file(
                root, relative, expected_stat_rows=None
            ) != expected:
                raise ForensicContractError(
                    f"forensic authority post-write byte drift: {relative}"
                )
    except BaseException as original:
        try:
            _rollback_relative_paths_no_follow(
                root, all_paths, label="forensic authority publication"
            )
        except BaseException as cleanup_exc:
            raise ForensicContractError(
                "forensic authority publication failed and rollback did not "
                "durably prove the full intended path set absent: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            ) from original
        raise
    return {
        "authorities": {
            str(path): _artifact_binding(str(path), payload)
            for path, payload in authority_payloads.items()
        },
        "source_closure": _artifact_binding(
            str(TRACKED_FORENSIC_SOURCE_CLOSURE_PATH),
            closure_payload,
            content_digest=closure["content_digest"],
            rows=closure["row_count"],
        ),
    }


def rollback_forensic_authorities(repo_root: str | Path) -> dict[str, Any]:
    """Rollback only uncommitted forensic-freeze authorities at the base HEAD."""

    root = Path(repo_root).resolve()
    if _git_text(root, ("rev-parse", "HEAD")) != SOURCE_COMMIT:
        raise ForensicContractError(
            "forensic authority rollback is permitted only at the base HEAD"
        )
    removed = _rollback_relative_paths_no_follow(
        root,
        FORENSIC_GENERATED_AUTHORITY_PATHS,
        label="forensic authority postflight",
    )
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "forensic_authority_rollback.v1"
            ),
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "base_head": SOURCE_COMMIT,
            "intended_paths": list(FORENSIC_GENERATED_AUTHORITY_PATHS),
            "removed_paths": removed,
            "all_intended_paths_absent": True,
            "pass": True,
        }
    )


def _forensic_dirty_paths(repo_root: Path) -> list[str]:
    return sorted(
        {
            row
            for args in (
                ("diff", "--name-only"),
                ("diff", "--cached", "--name-only"),
                ("ls-files", "--others", "--exclude-standard"),
            )
            for row in _git_text(repo_root, args).splitlines()
            if row
        }
    )


def _validate_forensic_prepublication_namespaces(
    repo_root: Path, *, exclude_current_process: bool
) -> dict[str, Any]:
    scientific_output = BASE.OUTPUT_ROOT
    attempts = sorted(
        str(path.absolute())
        for path in scientific_output.parent.glob(
            f".{scientific_output.name}.attempt-*"
        )
    )
    diagnostic_present = (
        PREEXECUTION_DIAGNOSTIC_ROOT.exists()
        or PREEXECUTION_DIAGNOSTIC_ROOT.is_symlink()
    )
    result_present = sorted(
        str(path)
        for path in _forensic_result_paths()
        if (repo_root / path).exists() or (repo_root / path).is_symlink()
    )
    process_matches = _active_forensic_or_scientific_processes(
        repo_root,
        exclude_pids=({os.getpid()} if exclude_current_process else ()),
    )
    if (
        scientific_output.exists()
        or scientific_output.is_symlink()
        or attempts
        or diagnostic_present
        or result_present
        or process_matches
    ):
        raise ForensicContractError(
            "forensic authority preparation namespace/process drift"
        )
    return {
        "scientific_output_root": str(scientific_output),
        "scientific_output_root_absent": True,
        "scientific_attempt_namespaces": [],
        "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_ROOT),
        "diagnostic_root_absent": True,
        "forensic_result_paths_present": [],
        "other_exact_scientific_or_forensic_process_matches": [],
        "caller_pid_excluded_from_other_process_scan": (
            os.getpid() if exclude_current_process else None
        ),
        "caller_role_validation_owned_by_evaluator_entrypoint": (
            exclude_current_process
        ),
        "pass": True,
    }


def validate_forensic_freeze_preparation(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate prospective dirty bytes before writing any forensic authority."""

    root = Path(repo_root).resolve()
    head = _git_text(root, ("rev-parse", "HEAD"))
    if head != SOURCE_COMMIT:
        raise ForensicContractError("forensic preparation base HEAD drift")
    changed = _forensic_dirty_paths(root)
    if changed != sorted(FORENSIC_CODE_AND_TEST_PATHS):
        raise ForensicContractError("forensic preparation dirty path-set drift")
    for relative in FORENSIC_CODE_AND_TEST_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ForensicContractError(
                f"forensic preparation source path drift: {relative}"
            )
    stale_authorities = [
        path
        for path in FORENSIC_GENERATED_AUTHORITY_PATHS
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if stale_authorities:
        raise ForensicContractError(
            f"forensic preparation authority paths are stale: {stale_authorities}"
        )
    namespace = _validate_forensic_prepublication_namespaces(
        root, exclude_current_process=True
    )
    prospective_closure = build_forensic_source_closure(
        root, require_complete=False
    )
    prospective_complete_closure = build_prospective_forensic_source_closure(
        root
    )
    expected_missing = [str(path) for path in forensic_authority_payloads()]
    if (
        prospective_closure["missing_paths"] != expected_missing
        or prospective_closure["complete"] is not False
        or prospective_closure["row_count"]
        != len(FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS) - len(expected_missing)
    ):
        raise ForensicContractError(
            "forensic prospective source-closure phase drift"
        )
    source_proof = build_committed_source_root_cause_proof(root)
    scientific_invariants = _validate_scientific_argv_builder_invariants(root)
    prospective_authorities = {
        str(path): _artifact_binding(str(path), payload)
        for path, payload in forensic_authority_payloads().items()
    }
    return attach_self_digest(
        {
            "schema": FORENSIC_FREEZE_PREPARATION_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "freeze_mode": "PREEXECUTION_FORENSIC_AUTHORITY_PREPARATION",
            "base_head": head,
            "changed_paths": changed,
            "prospective_authorities": prospective_authorities,
            "prospective_source_closure": prospective_closure,
            "prospective_complete_source_closure": (
                prospective_complete_closure
            ),
            "namespace_and_process_custody": namespace,
            "committed_source_root_cause_proof": source_proof,
            "scientific_implementation_invariants": scientific_invariants,
            "scientific_inputs_opened": 0,
            "scientific_archive_payload_files_opened": 0,
            "files_reused": 0,
            "required_enclosing_commit_subject": FORENSIC_FREEZE_COMMIT_SUBJECT,
            "pass": True,
        }
    )


def validate_forensic_authority_write(
    repo_root: str | Path,
    *,
    preparation_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact authority/closure bytes after the prospective write."""

    root = Path(repo_root).resolve()
    validate_self_digest(preparation_receipt)
    expected_authorities = {
        str(path): _artifact_binding(str(path), payload)
        for path, payload in forensic_authority_payloads().items()
    }
    partial = preparation_receipt.get("prospective_source_closure")
    complete = preparation_receipt.get("prospective_complete_source_closure")
    if not isinstance(partial, Mapping) or not isinstance(complete, Mapping):
        raise ForensicContractError("forensic preparation closure custody absent")
    validate_forensic_source_closure(partial, require_complete=False)
    validate_forensic_source_closure(complete, require_complete=True)
    expected_missing = [str(path) for path in forensic_authority_payloads()]
    partial_rows = [copy.deepcopy(dict(row)) for row in partial["rows"]]
    complete_rows = [copy.deepcopy(dict(row)) for row in complete["rows"]]
    complete_by_path = {row["path"]: row for row in complete_rows}
    expected_generated_rows = {
        path: {
            "path": path,
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
        }
        for path, binding in expected_authorities.items()
    }
    expected_receipt_keys = {
        "schema",
        "experiment_id",
        "freeze_mode",
        "base_head",
        "changed_paths",
        "prospective_authorities",
        "prospective_source_closure",
        "prospective_complete_source_closure",
        "namespace_and_process_custody",
        "committed_source_root_cause_proof",
        "scientific_implementation_invariants",
        "scientific_inputs_opened",
        "scientific_archive_payload_files_opened",
        "files_reused",
        "required_enclosing_commit_subject",
        "pass",
        "content_digest",
    }
    current_complete = build_prospective_forensic_source_closure(root)
    current_namespace = _validate_forensic_prepublication_namespaces(
        root, exclude_current_process=True
    )
    current_source_proof = build_committed_source_root_cause_proof(root)
    current_scientific_invariants = _validate_scientific_argv_builder_invariants(
        root
    )
    if (
        set(preparation_receipt) != expected_receipt_keys
        or preparation_receipt.get("schema")
        != FORENSIC_FREEZE_PREPARATION_SCHEMA
        or preparation_receipt.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or preparation_receipt.get("freeze_mode")
        != "PREEXECUTION_FORENSIC_AUTHORITY_PREPARATION"
        or preparation_receipt.get("base_head") != SOURCE_COMMIT
        or preparation_receipt.get("changed_paths")
        != sorted(FORENSIC_CODE_AND_TEST_PATHS)
        or preparation_receipt.get("required_enclosing_commit_subject")
        != FORENSIC_FREEZE_COMMIT_SUBJECT
        or preparation_receipt.get("prospective_authorities")
        != expected_authorities
        or partial.get("missing_paths") != expected_missing
        or partial.get("complete") is not False
        or partial_rows
        != [row for row in complete_rows if row["path"] not in expected_missing]
        or {
            path: complete_by_path.get(path) for path in expected_generated_rows
        }
        != expected_generated_rows
        or complete != current_complete
        or preparation_receipt.get("namespace_and_process_custody")
        != current_namespace
        or preparation_receipt.get("committed_source_root_cause_proof")
        != current_source_proof
        or preparation_receipt.get("scientific_implementation_invariants")
        != current_scientific_invariants
        or preparation_receipt.get("scientific_inputs_opened") != 0
        or preparation_receipt.get("scientific_archive_payload_files_opened")
        != 0
        or preparation_receipt.get("files_reused") != 0
        or preparation_receipt.get("pass") is not True
    ):
        raise ForensicContractError("forensic preparation receipt drift")
    head = _git_text(root, ("rev-parse", "HEAD"))
    changed = _forensic_dirty_paths(root)
    if head != SOURCE_COMMIT or changed != sorted(FORENSIC_REQUIRED_CHANGED_PATHS):
        raise ForensicContractError("forensic authority write path/HEAD drift")
    namespace = _validate_forensic_prepublication_namespaces(
        root, exclude_current_process=True
    )
    authority_custody = _validate_forensic_authority_files(root)
    closure = build_forensic_source_closure(root, require_complete=True)
    validate_forensic_source_closure(closure, require_complete=True)
    if (
        closure["row_count"] != len(FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS)
        or closure != complete
    ):
        raise ForensicContractError(
            "forensic authority closure changed after preparation"
        )
    return attach_self_digest(
        {
            "schema": FORENSIC_AUTHORITY_WRITE_VALIDATION_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "freeze_mode": "PREEXECUTION_FORENSIC_AUTHORITY_PREPARATION",
            "base_head": head,
            "changed_paths": changed,
            "preparation_receipt_content_digest": preparation_receipt[
                "content_digest"
            ],
            "authorities": expected_authorities,
            "source_closure": authority_custody[
                "forensic_authority_source_closure"
            ],
            "authority_custody": authority_custody,
            "source_closure_rows": closure["row_count"],
            "namespace_and_process_custody": namespace,
            "scientific_inputs_opened": 0,
            "scientific_archive_payload_files_opened": 0,
            "files_reused": 0,
            "required_enclosing_commit_subject": FORENSIC_FREEZE_COMMIT_SUBJECT,
            "pass": True,
        }
    )


def _validate_bound_repo_file(
    repo_root: Path, binding: Mapping[str, Any], label: str
) -> None:
    relative = Path(str(binding["path"]))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ForensicContractError(f"{label} repository path drift")
    try:
        payload = _read_no_follow_bound_file(
            repo_root, relative, expected_stat_rows=None
        )
    except FileNotFoundError as exc:
        raise ForensicContractError(f"{label} is absent") from exc
    if (
        hashlib.sha256(payload).hexdigest() != binding["sha256"]
        or len(payload) != binding["bytes"]
    ):
        raise ForensicContractError(f"{label} byte binding drift")
    if "content_digest" in binding:
        value = _canonical_json_from_bound_bytes(payload, label=label)
        digest_field_names = (
            "content_digest",
            "contract_sha256",
            "output_schema_sha256",
            "fixture_sha256",
        )
        matching_fields = [
            name
            for name in digest_field_names
            if value.get(name) == binding["content_digest"]
        ]
        if len(matching_fields) != 1:
            raise ForensicContractError(f"{label} content binding drift")


def _validate_base_authorities_read_only(repo_root: Path) -> dict[str, Any]:
    if BASE.CONTRACT_SHA256 != BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256:
        raise ForensicContractError("live scientific contract digest drift")
    safe_original = {
        name: binding
        for name, binding in BASE.BASE_SCIENTIFIC_AUTHORITY_BINDINGS.items()
        if name != "route_role_authority"
    }
    groups = {
        "original_scientific": safe_original,
        "first_execution_correction": (
            BASE.EXECUTION_CORRECTION_2_BASE_AMENDMENT_AUTHORITY_BINDINGS
        ),
        "second_execution_correction": {
            "amendment": BASE.EXECUTION_CORRECTION_2_AMENDMENT_BINDING,
            "output_schema": BASE.EXECUTION_CORRECTION_2_OUTPUT_SCHEMA_BINDING,
            "fixture": BASE.EXECUTION_CORRECTION_2_FIXTURE_BINDING,
            "source_closure": BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING,
        },
    }
    for group, bindings in groups.items():
        for name, binding in bindings.items():
            _validate_bound_repo_file(repo_root, binding, f"{group}:{name}")
    return {
        "scientific_contract_digest": BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
        "scientific_contract_freeze_commit": BASE.INITIAL_EXECUTION_FREEZE_COMMIT,
        "original_scientific_authorities": len(groups["original_scientific"]),
        "outcome_derived_route_role_authority_files_opened": 0,
        "route_role_authority_binding_reused_from_prior_source_closure_metadata": (
            copy.deepcopy(BASE.ROUTE_ROLE_RECEIPT_BINDING)
        ),
        "first_execution_correction_authorities": len(
            groups["first_execution_correction"]
        ),
        "second_execution_correction_authorities": len(
            groups["second_execution_correction"]
        ),
        "pass": True,
    }


def _validate_archive_custody_metadata_only() -> dict[str, Any]:
    records = build_archive_inventory_authority()["archives"]
    output: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["archive_path"])
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise ForensicContractError("failed archive directory is absent") from exc
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ForensicContractError("failed archive root custody drift")
        expected_root_stat = record["root_stat_authority"]
        root_scalar_stat = {
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": int(info.st_nlink),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "size": int(info.st_size),
        }
        if root_scalar_stat != {
            key: expected_root_stat[key]
            for key in ("mode", "nlink", "uid", "gid", "size")
        }:
            raise ForensicContractError("failed archive root scalar-stat drift")
        try:
            stat_times = subprocess.run(
                ["stat", "--printf=%w\\n%y\\n%z\\n", str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "LC_ALL": "C", "TZ": "Europe/London"},
            ).stdout.splitlines()
        except subprocess.CalledProcessError as exc:
            raise ForensicContractError("failed archive root time-stat failed") from exc
        if stat_times != [
            expected_root_stat["birth"],
            expected_root_stat["mtime"],
            expected_root_stat["ctime"],
        ]:
            raise ForensicContractError("failed archive root time-stat drift")
        stat_rows: list[dict[str, Any]] = []
        symlinks: list[str] = []
        nonregular: list[str] = []
        multi_link: list[str] = []
        for directory, directory_names, file_names in os.walk(
            path, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            for name in list(directory_names):
                candidate = directory_path / name
                candidate_info = os.lstat(candidate)
                if stat.S_ISLNK(candidate_info.st_mode):
                    symlinks.append(str(candidate.relative_to(path)))
                    directory_names.remove(name)
                elif not stat.S_ISDIR(candidate_info.st_mode):
                    nonregular.append(str(candidate.relative_to(path)))
                    directory_names.remove(name)
            for name in file_names:
                candidate = directory_path / name
                candidate_info = os.lstat(candidate)
                relative = str(candidate.relative_to(path))
                if stat.S_ISLNK(candidate_info.st_mode):
                    symlinks.append(relative)
                    continue
                if not stat.S_ISREG(candidate_info.st_mode):
                    nonregular.append(relative)
                    continue
                if candidate_info.st_nlink != 1:
                    multi_link.append(relative)
                stat_rows.append(
                    {
                        "path": relative,
                        "bytes": int(candidate_info.st_size),
                        "mode": stat.S_IMODE(candidate_info.st_mode),
                        "nlink": int(candidate_info.st_nlink),
                    }
                )
        stat_rows.sort(key=lambda row: row["path"])
        observed = {
            "files": len(stat_rows),
            "bytes": sum(row["bytes"] for row in stat_rows),
        }
        expected_counts = {
            "files": int(record["inventory"]["files"]),
            "bytes": int(record["inventory"]["bytes"]),
        }
        stat_manifest_sha256 = canonical_json_sha256(stat_rows)
        if (
            observed != expected_counts
            or stat_manifest_sha256
            != record["stat_manifest"]["sha256"]
            or symlinks
            or nonregular
            or multi_link
        ):
            raise ForensicContractError("failed archive stat inventory drift")
        output.append(
            {
                "ordinal": record["ordinal"],
                "archive_path": record["archive_path"],
                "inventory": copy.deepcopy(record["inventory"]),
                "directory_stat_only": True,
                "root_stat": {
                    **root_scalar_stat,
                    "birth": stat_times[0],
                    "mtime": stat_times[1],
                    "ctime": stat_times[2],
                    "namespace_timestamp_ns": expected_root_stat[
                        "namespace_timestamp_ns"
                    ],
                    "namespace_timestamp_local": expected_root_stat[
                        "namespace_timestamp_local"
                    ],
                    "namespace_timestamp_role": (
                        "ARCHIVE_NAME_ENCODING_NOT_FILESYSTEM_BIRTH_TIME"
                    ),
                    "filesystem_stat_role": (
                        "CURRENT_MUTABLE_CUSTODY_NOT_HISTORICAL_PROCESS_EVIDENCE"
                    ),
                },
                "observed_stat_inventory": observed,
                "stat_manifest_sha256": stat_manifest_sha256,
                "stat_manifest_role": (
                    "FILESYSTEM_METADATA_ONLY_NOT_HISTORICAL_CONTENT_MANIFEST"
                ),
                "symlinks": [],
                "nonregular_files": [],
                "multi_link_files": [],
                "archive_files_opened": 0,
                "files_reused": 0,
                "pass": True,
            }
        )
    third_technical_receipts = verify_third_failure_technical_receipts()
    return {
        "archives": output,
        "archive_count": 3,
        "third_incident_technical_receipts": third_technical_receipts,
        "third_incident_technical_receipts_verified": True,
        "scientific_archive_payload_files_opened": 0,
        "technical_receipt_distinct_paths": 2,
        "technical_file_open_operations_this_validation": 2,
        "files_reused": 0,
        "pass": True,
    }


def _git_text(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ForensicContractError(f"git custody command failed: {args}") from exc


def _validate_scientific_argv_builder_invariants(repo_root: Path) -> dict[str, Any]:
    path = "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
    committed, _ = _git_blob_bytes(repo_root, SOURCE_COMMIT, path)
    live = _read_no_follow_bound_file(
        repo_root, path, expected_stat_rows=None
    )
    try:
        committed_tree = ast.parse(committed.decode("utf-8"))
        live_tree = ast.parse(live.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ForensicContractError("scientific argv invariant AST parse failed") from exc
    committed_functions = {
        node.name: node
        for node in committed_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    live_functions = {
        node.name: node
        for node in live_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    committed_imports = [
        ast.dump(node, include_attributes=False)
        for node in committed_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    live_imports = [
        ast.dump(node, include_attributes=False)
        for node in live_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    new_import_nodes = [
        ast.Import(names=[ast.alias(name=name)])
        for name in ("faulthandler", "importlib", "resource", "stat", "traceback")
    ]
    allowed_new_imports = {
        ast.dump(node, include_attributes=False) for node in new_import_nodes
    }
    observed_new_imports = [
        value for value in live_imports if value not in set(committed_imports)
    ]
    observed_existing_imports = [
        value for value in live_imports if value in set(committed_imports)
    ]
    if (
        observed_existing_imports != committed_imports
        or set(observed_new_imports) != allowed_new_imports
        or len(observed_new_imports) != len(allowed_new_imports)
    ):
        raise ForensicContractError("evaluator top-level import authority drift")

    def assignment_names(node: ast.AST) -> tuple[str, ...]:
        targets: list[ast.AST]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            return ()
        names: list[str] = []
        for target in targets:
            for item in ast.walk(target):
                if isinstance(item, ast.Name):
                    names.append(item.id)
        return tuple(names)

    committed_assignments = [
        (assignment_names(node), ast.dump(node, include_attributes=False))
        for node in committed_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    live_assignments = [
        (assignment_names(node), ast.dump(node, include_attributes=False))
        for node in live_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    committed_assignment_names = {name for names, _ in committed_assignments for name in names}
    observed_existing_assignments = [
        row
        for row in live_assignments
        if any(name in committed_assignment_names for name in row[0])
    ]
    if observed_existing_assignments != committed_assignments:
        raise ForensicContractError("pre-existing evaluator assignment drift")
    expected_new_assignment_source = """
FORENSIC_INTERPRETER = Path('/home/andrewknowles/TinyQuadJEPA/bin/python')
PREEXECUTION_DIAGNOSTIC_WRAPPER_SCRIPT = (ROOT / 'scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py').absolute()
PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND = 'freeze-preexecution-forensic'
PREEXECUTION_DIAGNOSTIC_SUBCOMMAND = 'diagnose-preexecution'
PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND = 'diagnose-preexecution-child'
PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND = 'diagnose-preexecution-synthetic-child'
"""
    expected_new_assignment_tree = ast.parse(expected_new_assignment_source)
    expected_new_assignments = {
        (assignment_names(node), ast.dump(node, include_attributes=False))
        for node in expected_new_assignment_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    }
    observed_new_assignments = {
        row
        for row in live_assignments
        if not any(name in committed_assignment_names for name in row[0])
    }
    if observed_new_assignments != expected_new_assignments:
        raise ForensicContractError("new evaluator forensic assignment drift")
    allowed_diagnostic_cli_extensions = {
        "_classify_experiment_argv",
        "main",
    }
    missing = sorted(set(committed_functions) - set(live_functions))
    if missing:
        raise ForensicContractError(
            f"pre-existing evaluator function removed: {missing}"
        )
    rows = []
    for name in sorted(set(committed_functions) - allowed_diagnostic_cli_extensions):
        committed_node = committed_functions[name]
        live_node = live_functions[name]
        committed_dump = ast.dump(committed_node, include_attributes=False)
        live_dump = ast.dump(live_node, include_attributes=False)
        if committed_dump != live_dump:
            raise ForensicContractError(
                f"existing scientific argv builder changed: {name}"
            )
        rows.append(
            {
                "function": name,
                "ast_sha256": hashlib.sha256(
                    committed_dump.encode("utf-8")
                ).hexdigest(),
                "unchanged": True,
            }
        )
    changed_allowed = []

    def statement_dump(source: str) -> str:
        tree = ast.parse(source)
        if len(tree.body) != 1:
            raise ForensicContractError("internal CLI authority statement drift")
        return ast.dump(tree.body[0], include_attributes=False)

    def branch_dump(source: str) -> str:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.If):
            raise ForensicContractError("internal CLI authority branch drift")
        node = tree.body[0]
        return ast.dump(
            ast.If(test=node.test, body=node.body, orelse=[]),
            include_attributes=False,
        )

    permitted_statement_dumps = {
        statement_dump(
            "subparsers.add_parser(PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND)"
        ),
        statement_dump(
            "subparsers.add_parser(PREEXECUTION_DIAGNOSTIC_SUBCOMMAND)"
        ),
        statement_dump(
            "diagnostic_child_parser = subparsers.add_parser("
            "PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND)"
        ),
        statement_dump(
            "diagnostic_child_parser.add_argument("
            "'--launcher-pid', type=int, required=True)"
        ),
        statement_dump(
            "diagnostic_child_parser.add_argument("
            "'--launcher-start-time-ticks', type=int, required=True)"
        ),
        statement_dump(
            "diagnostic_child_parser.add_argument("
            "'--diagnostic-root', type=Path, required=True)"
        ),
        statement_dump(
            "diagnostic_synthetic_parser = subparsers.add_parser("
            "PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND)"
        ),
        statement_dump(
            "diagnostic_synthetic_parser.add_argument("
            "'--fixture-id', type=str, required=True)"
        ),
        statement_dump(
            "diagnostic_synthetic_parser.add_argument("
            "'--diagnostic-root', type=Path, required=True)"
        ),
    }
    permitted_branch_dumps = {
        branch_dump(
            "if subcommand in {PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND, "
            "PREEXECUTION_DIAGNOSTIC_SUBCOMMAND}:\n"
            "    return None"
        ),
        branch_dump(
            "if args.command == PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND:\n"
            "    value = freeze_preexecution_forensic_contract()"
        ),
        branch_dump(
            "if args.command == PREEXECUTION_DIAGNOSTIC_SUBCOMMAND:\n"
            "    value = execute_preexecution_diagnostic()"
        ),
        branch_dump(
            "if args.command == PREEXECUTION_DIAGNOSTIC_CHILD_SUBCOMMAND:\n"
            "    value = _execute_preexecution_only_diagnostic_child("
            "launcher_pid=args.launcher_pid, "
            "launcher_start_time_ticks=args.launcher_start_time_ticks, "
            "diagnostic_root=args.diagnostic_root)"
        ),
        branch_dump(
            "if args.command == PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD_SUBCOMMAND:\n"
            "    value = _execute_preexecution_diagnostic_synthetic_child("
            "fixture_id=args.fixture_id, diagnostic_root=args.diagnostic_root)"
        ),
    }

    class StripDiagnosticCli(ast.NodeTransformer):
        def __init__(self) -> None:
            self.removed_statements: list[str] = []
            self.removed_branches: list[str] = []

        def visit_If(self, node: ast.If) -> Any:  # noqa: N802
            signature = ast.dump(
                ast.If(test=node.test, body=node.body, orelse=[]),
                include_attributes=False,
            )
            if signature in permitted_branch_dumps:
                self.removed_branches.append(signature)
                replacement: list[ast.stmt] = []
                for child_node in node.orelse:
                    visited = self.visit(child_node)
                    if isinstance(visited, list):
                        replacement.extend(visited)
                    elif visited is not None:
                        replacement.append(visited)
                return replacement
            return self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: N802
            signature = ast.dump(node, include_attributes=False)
            if signature in permitted_statement_dumps:
                self.removed_statements.append(signature)
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node: ast.Expr) -> Any:  # noqa: N802
            signature = ast.dump(node, include_attributes=False)
            if signature in permitted_statement_dumps:
                self.removed_statements.append(signature)
                return None
            return self.generic_visit(node)

    for name in sorted(allowed_diagnostic_cli_extensions):
        before = ast.dump(committed_functions[name], include_attributes=False)
        stripper = StripDiagnosticCli()
        stripped = stripper.visit(copy.deepcopy(live_functions[name]))
        ast.fix_missing_locations(stripped)
        after = ast.dump(stripped, include_attributes=False)
        if before != after:
            raise ForensicContractError(
                f"diagnostic CLI extension exceeds authority: {name}"
            )
        changed_allowed.append(
            {
                "function": name,
                "live_changed": ast.dump(
                    live_functions[name], include_attributes=False
                ) != before,
                "normalized_ast_equal": True,
                "removed_statement_ast_dumps": sorted(
                    stripper.removed_statements
                ),
                "removed_branch_ast_dumps": sorted(stripper.removed_branches),
                "change_authority": "EXACT_DIAGNOSTIC_CLI_EXTENSION_ONLY",
            }
        )
    observed_removed_statements = {
        item
        for row in changed_allowed
        for item in row["removed_statement_ast_dumps"]
    }
    observed_removed_branches = {
        item
        for row in changed_allowed
        for item in row["removed_branch_ast_dumps"]
    }
    if (
        observed_removed_statements != permitted_statement_dumps
        or observed_removed_branches != permitted_branch_dumps
        or sum(
            len(row["removed_statement_ast_dumps"]) for row in changed_allowed
        )
        != len(permitted_statement_dumps)
        or sum(len(row["removed_branch_ast_dumps"]) for row in changed_allowed)
        != len(permitted_branch_dumps)
    ):
        raise ForensicContractError("exact diagnostic CLI addition custody drift")
    new_functions = sorted(set(live_functions) - set(committed_functions))
    expected_new_functions = {
        "_active_forensic_processes",
        "_append_synthetic_technical_cleanup_row",
        "_assert_forensic_process_group_quiescent",
        "_atomic_preexecution_json",
        "_atomic_preexecution_last_stage",
        "_build_preexecution_read_guard_manifest",
        "_classify_forensic_argv",
        "_correction_2_failed_archive_identity_matches",
        "_correction_2_failed_archive_identity_projection",
        "_diagnostic_forbidden_scientific_roots",
        "_diagnostic_stage",
        "_ensure_synthetic_technical_cleanup",
        "_exclusive_bytes",
        "_execute_preexecution_diagnostic_synthetic_child",
        "_execute_preexecution_diagnostic_under_umask",
        "_execute_preexecution_only_diagnostic_child",
        "_expected_preexecution_diagnostic_child_argv",
        "_expected_preexecution_diagnostic_internal_argv",
        "_expected_preexecution_diagnostic_launcher_argv",
        "_expected_preexecution_forensic_freeze_argv",
        "_expected_preexecution_diagnostic_synthetic_child_argv",
        "_expected_preexecution_diagnostic_synthetic_internal_argv",
        "_forensic_child_environment",
        "_forensic_contract",
        "_forensic_process_identity",
        "_forensic_process_identity_is_live",
        "_forensic_runtime_paths",
        "_freeze_preexecution_forensic_authorities",
        "_install_preexecution_diagnostic_read_guard",
        "_load_diagnostic_stage_rows",
        "_load_forensic_source_closure",
        "_load_preexecution_child_exception",
        "_load_preexecution_last_stage_marker",
        "_load_synthetic_technical_lifecycle_rows",
        "_preexecution_output_namespace_entries",
        "_publish_preexecution_diagnostic_terminal",
        "_require_exact_live_preexecution_diagnostic_launcher",
        "_require_synthetic_diagnostic_root",
        "_run_forensic_child_with_external_stream_custody",
        "_run_synthetic_preexecution_diagnostic_fixture",
        "_run_synthetic_preexecution_diagnostic_fixture_under_umask",
        "_synthetic_archive_validation_projection_evidence",
        "_synthetic_fixture_result_row",
        "_technical_runtime_context",
        "_terminate_forensic_process_group_bounded",
        "_validate_forensic_constant_alignment",
        "_validate_preexecution_child_payload",
        "_wait_for_forensic_child_with_timeout",
        "execute_preexecution_diagnostic",
        "freeze_preexecution_forensic_contract",
    }
    if set(new_functions) != expected_new_functions:
        raise ForensicContractError("non-diagnostic evaluator function added")
    new_function_ast = {
        name: hashlib.sha256(
            ast.dump(live_functions[name], include_attributes=False).encode("utf-8")
        ).hexdigest()
        for name in new_functions
    }

    implementation_rows = []
    for label, binding in SCIENTIFIC_IMPLEMENTATION_INVARIANT_BINDINGS.items():
        committed_payload, blob = _git_blob_bytes(
            repo_root, SOURCE_COMMIT, binding["path"]
        )
        live_payload = _read_no_follow_bound_file(
            repo_root, binding["path"], expected_stat_rows=None
        )
        if (
            blob != binding["git_blob_sha1"]
            or hashlib.sha256(committed_payload).hexdigest() != binding["sha256"]
            or live_payload != committed_payload
        ):
            raise ForensicContractError(
                f"frozen scientific {label} implementation byte drift"
            )
        implementation_rows.append(
            {"label": label, **copy.deepcopy(binding), "byte_exact": True}
        )
    return {
        "unchanged_preexisting_functions": rows,
        "unchanged_preexisting_function_count": len(rows),
        "allowed_changed_functions": changed_allowed,
        "new_diagnostic_functions": new_functions,
        "new_diagnostic_function_ast_sha256": new_function_ast,
        "unchanged_top_level_import_count": len(committed_imports),
        "new_forensic_imports": sorted(allowed_new_imports),
        "unchanged_top_level_assignment_count": len(committed_assignments),
        "new_forensic_assignments": [
            {"targets": list(names), "ast": dump}
            for names, dump in sorted(expected_new_assignments)
        ],
        "scientific_implementation_bytes": implementation_rows,
        "scientific_contract_digest": BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
        "model_metric_gate_or_target_change": False,
        "pass": True,
    }


def _forensic_result_paths() -> tuple[Path, ...]:
    return (
        TRACKED_INVOCATION_RECEIPT_PATH,
        TRACKED_OS_EVIDENCE_RECEIPT_PATH,
        TRACKED_SYNTHETIC_RESULTS_PATH,
        TRACKED_PREEXECUTION_ONLY_RECEIPT_PATH,
        TRACKED_DIAGNOSTIC_CUSTODY_RECEIPT_PATH,
        TRACKED_CONDITIONAL_V2_DECISION_PATH,
        TRACKED_CONDITIONAL_V2_SPEC_AUTHORITY_PATH,
        TRACKED_CONDITIONAL_V2_SPEC_PATH,
        TRACKED_FORENSIC_RESULT_PATH,
        TRACKED_FORENSIC_REPORT_PATH,
    )


def _active_forensic_or_scientific_processes(
    repo_root: Path,
    *,
    exclude_pids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Return exact-element experiment/diagnostic producer matches in `/proc`."""

    evaluator = str(
        (repo_root / "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py").absolute()
    )
    helper = str(
        (
            repo_root
            / "scripts/materialize_plan_aware_proprio_predictor_substitution_v1.py"
        ).absolute()
    )
    wrapper = str(
        (repo_root / PREEXECUTION_DIAGNOSTIC_CHILD_WRAPPER_PATH).absolute()
    )
    evaluator_roles = {
        "execute": "NONSCIENTIFIC_LAUNCHER",
        "execute-scientific": "SCIENTIFIC_EVALUATOR",
        "finalize-correction-2": "TERMINAL_FINALIZER",
        "check": "POST_FINALIZER_CHECKER",
        PREEXECUTION_FORENSIC_FREEZE_SUBCOMMAND: (
            "PREEXECUTION_FORENSIC_FREEZE"
        ),
        PREEXECUTION_DIAGNOSTIC_SUBCOMMAND: (
            "PREEXECUTION_DIAGNOSTIC_LAUNCHER"
        ),
    }
    excluded = {int(pid) for pid in exclude_pids}
    matches: list[dict[str, Any]] = []
    for process in sorted(
        (entry for entry in Path("/proc").iterdir() if entry.name.isdigit()),
        key=lambda entry: int(entry.name),
    ):
        pid = int(process.name)
        if pid in excluded:
            continue
        try:
            raw = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        argv = [
            part.decode("utf-8", errors="surrogateescape")
            for part in raw.split(b"\0")
            if part
        ]
        if not argv:
            continue
        role: str | None = None
        if evaluator in argv:
            role = "INVALID_EXPERIMENT_ARGV"
            if len(argv) >= 3 and argv[1] == evaluator:
                role = evaluator_roles.get(argv[2], role)
        elif helper in argv:
            role = "CONDITIONAL_SCIENTIFIC_HELPER"
        elif wrapper in argv:
            role = "PREEXECUTION_DIAGNOSTIC_CHILD"
        if role is None:
            continue
        try:
            executable = str((process / "exe").resolve(strict=True))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            executable = "UNAVAILABLE_DURING_SCAN"
        matches.append(
            {
                "pid": pid,
                "role": role,
                "argv": argv,
                "argv_sha256": canonical_json_sha256(argv),
                "executable": executable,
            }
        )
    return matches


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0


def validate_forensic_result_commit(
    repo_root: str | Path,
    *,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Validate the clean post-diagnostic evidence commit without opening science."""

    root = Path(repo_root).resolve()
    head = _git_text(root, ["rev-parse", "HEAD"])
    if _git_text(root, ["status", "--porcelain=v1"]):
        raise ForensicContractError("forensic result repository is dirty")
    lineage = _git_text(root, ["rev-list", "--parents", "-n", "1", head]).split()
    if len(lineage) != 2 or lineage[0] != head:
        raise ForensicContractError("forensic result commit parent-count drift")
    freeze_commit = lineage[1]
    if _git_text(root, ["show", "-s", "--format=%s", head]) != (
        FORENSIC_RESULT_COMMIT_SUBJECT
    ):
        raise ForensicContractError("forensic result commit subject drift")

    freeze_lineage = _git_text(
        root, ["rev-list", "--parents", "-n", "1", freeze_commit]
    ).split()
    if (
        len(freeze_lineage) != 2
        or freeze_lineage[0] != freeze_commit
        or freeze_lineage[1] != SOURCE_COMMIT
        or _git_text(root, ["show", "-s", "--format=%s", freeze_commit])
        != FORENSIC_FREEZE_COMMIT_SUBJECT
    ):
        raise ForensicContractError("forensic diagnostic freeze lineage drift")
    freeze_changed = sorted(
        row
        for row in _git_text(
            root, ["diff", "--name-only", SOURCE_COMMIT, freeze_commit]
        ).splitlines()
        if row
    )
    if freeze_changed != sorted(FORENSIC_REQUIRED_CHANGED_PATHS):
        raise ForensicContractError("forensic diagnostic freeze path-set drift")

    status_rows = [
        row.split("\t", 1)
        for row in _git_text(
            root, ["diff", "--name-status", freeze_commit, head]
        ).splitlines()
        if row
    ]
    expected_result_paths = sorted(str(path) for path in _forensic_result_paths())
    if (
        len(status_rows) != len(expected_result_paths)
        or sorted(path for status, path in status_rows if status == "A")
        != expected_result_paths
        or any(status != "A" for status, _path in status_rows)
    ):
        raise ForensicContractError("forensic result commit path/status drift")

    required_ancestors = list(
        dict.fromkeys([*FORENSIC_LINEAGE.values(), freeze_commit])
    )
    absent_ancestors = [
        ancestor
        for ancestor in required_ancestors
        if not _git_is_ancestor(root, ancestor, head)
    ]
    if absent_ancestors:
        raise ForensicContractError(
            f"forensic result required ancestors absent: {absent_ancestors}"
        )

    authority_custody = _validate_forensic_authority_files(root)
    bundle = load_and_validate_diagnostic_bundle(
        repo_root=root, diagnostic_root=diagnostic_root
    )
    artifacts = build_forensic_result_artifacts_from_bundle(bundle)
    if (
        artifacts["result"].get("source_freeze_commit") != freeze_commit
        or bundle["forensic_freeze_custody"].get("repo_head") != freeze_commit
    ):
        raise ForensicContractError("forensic result/freeze cross-binding drift")
    archive_custody = _validate_archive_custody_metadata_only()
    tracked_payloads: Mapping[Path, bytes] = artifacts["tracked_payloads"]
    if set(tracked_payloads) != set(_forensic_result_paths()):
        raise ForensicContractError("rebuilt forensic result path-set drift")
    tracked_bindings: dict[str, Any] = {}
    for relative in _forensic_result_paths():
        observed = _read_no_follow_bound_file(
            root, relative, expected_stat_rows=None
        )
        expected = tracked_payloads[relative]
        if observed != expected:
            raise ForensicContractError(
                f"tracked forensic result byte drift: {relative}"
            )
        tracked_bindings[str(relative)] = _artifact_binding(
            str(relative), observed
        )
    validate_conditional_v2_spec_authority(
        artifacts["conditional_v2_spec_authority"],
        decision=artifacts["conditional_v2_decision"],
    )
    validate_conditional_v2_spec_markdown(
        artifacts["conditional_v2_spec_markdown"],
        specification=artifacts["conditional_v2_spec_authority"],
    )

    scientific_output_root = BASE.OUTPUT_ROOT
    attempt_namespaces = sorted(
        str(path.absolute())
        for path in scientific_output_root.parent.glob(
            f".{scientific_output_root.name}.attempt-*"
        )
    )
    process_matches = _active_forensic_or_scientific_processes(root)
    if (
        scientific_output_root.exists()
        or scientific_output_root.is_symlink()
        or attempt_namespaces
        or process_matches
    ):
        raise ForensicContractError(
            "forensic result has live scientific/diagnostic execution state"
        )

    return attach_self_digest(
        {
            "schema": FORENSIC_RESULT_COMMIT_VALIDATION_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "result_commit": head,
            "sole_parent_diagnostic_freeze_commit": freeze_commit,
            "result_commit_subject": FORENSIC_RESULT_COMMIT_SUBJECT,
            "diagnostic_freeze_commit_subject": FORENSIC_FREEZE_COMMIT_SUBJECT,
            "result_changed_paths": expected_result_paths,
            "result_changed_path_status": "ADDED_ONLY",
            "required_ancestors": required_ancestors,
            "repo_clean": True,
            "authority_custody": authority_custody,
            "archive_custody": archive_custody,
            "diagnostic_bundle": {
                "root": str(Path(diagnostic_root).absolute()),
                "runtime_result_content_digest": bundle["result"][
                    "content_digest"
                ],
                "final_namespace_inventory_content_digest": bundle[
                    "final_namespace_inventory"
                ]["content_digest"],
                "preflight_postflight_equal": (
                    bundle["preflight_namespace_inventory"]
                    == bundle["postflight_namespace_inventory"]
                ),
                "pass": True,
            },
            "tracked_bindings": tracked_bindings,
            "tracked_bytes_equal_rebuilt_artifact_set": True,
            "v2_specification_authority_validated": True,
            "v2_specification_markdown_projection_validated": True,
            "scientific_disposition": SCIENTIFIC_DISPOSITION,
            "scientific_result": False,
            "scientific_classification": None,
            "scientific_output_root": str(scientific_output_root),
            "scientific_output_root_absent": True,
            "scientific_attempt_namespaces": [],
            "exact_scientific_or_forensic_process_matches": [],
            "scientific_inputs_opened_by_commit_validator": 0,
            "scientific_attempt_authorized": False,
            "automatic_execution_authorized": False,
            "pass": True,
        }
    )


def _validate_forensic_authority_files(repo_root: Path) -> dict[str, Any]:
    expected = forensic_authority_payloads()
    for relative, payload in expected.items():
        if _read_no_follow_bound_file(
            repo_root, relative, expected_stat_rows=None
        ) != payload:
            raise ForensicContractError(f"forensic authority byte drift: {relative}")
    closure_payload = _read_no_follow_bound_file(
        repo_root,
        TRACKED_FORENSIC_SOURCE_CLOSURE_PATH,
        expected_stat_rows=None,
    )
    closure = _canonical_json_from_bound_bytes(
        closure_payload, label="forensic source closure"
    )
    validate_forensic_source_closure(closure, require_complete=True)
    live = build_forensic_source_closure(repo_root, require_complete=True)
    if closure != live:
        raise ForensicContractError("forensic source closure does not match live bytes")
    payload = canonical_json_bytes(closure) + b"\n"
    if closure_payload != payload:
        raise ForensicContractError("forensic source closure canonical bytes drift")
    return {
        "forensic_contract": copy.deepcopy(FORENSIC_CONTRACT_BINDING),
        "forensic_authority_source_closure": _artifact_binding(
            str(TRACKED_FORENSIC_SOURCE_CLOSURE_PATH),
            payload,
            content_digest=closure["content_digest"],
            rows=closure["row_count"],
        ),
        "authority_count": len(expected) + 1,
        "pass": True,
    }


def validate_forensic_freeze_custody(repo_root: str | Path) -> dict[str, Any]:
    """Validate the clean committed diagnostic freeze without opening science."""

    root = Path(repo_root).resolve()
    head = _git_text(root, ["rev-parse", "HEAD"])
    if _git_text(root, ["status", "--porcelain=v1"]):
        raise ForensicContractError("forensic diagnostic freeze tree is dirty")
    parents = _git_text(root, ["rev-list", "--parents", "-n", "1", head]).split()
    subject = _git_text(root, ["show", "-s", "--format=%s", head])
    if (
        len(parents) != 2
        or parents[1] != SOURCE_COMMIT
        or subject != FORENSIC_FREEZE_COMMIT_SUBJECT
    ):
        raise ForensicContractError("forensic diagnostic freeze lineage drift")
    changed = sorted(
        line
        for line in _git_text(
            root, ["diff", "--name-only", SOURCE_COMMIT, head]
        ).splitlines()
        if line
    )
    if changed != sorted(FORENSIC_REQUIRED_CHANGED_PATHS):
        raise ForensicContractError("forensic diagnostic freeze path-set drift")

    authorities = _validate_forensic_authority_files(root)
    base = _validate_base_authorities_read_only(root)
    archives = _validate_archive_custody_metadata_only()
    committed_source_proof = build_committed_source_root_cause_proof(root)
    argv_invariants = _validate_scientific_argv_builder_invariants(root)

    output_root = BASE.OUTPUT_ROOT
    attempts = sorted(
        str(path.absolute())
        for path in output_root.parent.glob(f".{output_root.name}.attempt-*")
    )
    result_paths_present = [
        str(path)
        for path in _forensic_result_paths()
        if (root / path).exists() or (root / path).is_symlink()
    ]
    if (
        output_root.exists()
        or output_root.is_symlink()
        or attempts
        or PREEXECUTION_DIAGNOSTIC_ROOT.exists()
        or PREEXECUTION_DIAGNOSTIC_ROOT.is_symlink()
        or result_paths_present
    ):
        raise ForensicContractError("forensic prediagnostic output namespace is not empty")
    output_namespace = {
        "scientific_output_root": str(output_root),
        "scientific_output_root_absent": True,
        "scientific_attempt_namespaces": [],
        "diagnostic_root": str(PREEXECUTION_DIAGNOSTIC_ROOT),
        "diagnostic_root_absent": True,
        "forensic_result_paths_present": [],
        "pass": True,
    }
    return attach_self_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.forensic_freeze_custody.v1",
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "repo_head": head,
            "repo_clean": True,
            "sole_parent": SOURCE_COMMIT,
            "commit_subject": subject,
            "changed_paths": changed,
            "forensic_authority_validated": True,
            "base_authorities_read_only_validated": True,
            "archive_custody_metadata_only_validated": True,
            "scientific_contract_digest": (
                BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "forensic_authority_source_closure": authorities[
                "forensic_authority_source_closure"
            ],
            "authority_custody": authorities,
            "base_authority_custody": base,
            "archive_custody": archives,
            "committed_source_root_cause_proof": committed_source_proof,
            "scientific_argv_builder_invariants": argv_invariants,
            "output_namespace": output_namespace,
            "scientific_archive_payload_files_opened": 0,
            "scientific_inputs_opened": 0,
            "files_reused": 0,
            "pass": True,
        }
    )


def validate_forensic_freeze_custody_receipt(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    validate_self_digest(value)
    required_true = (
        "repo_clean",
        "forensic_authority_validated",
        "base_authorities_read_only_validated",
        "archive_custody_metadata_only_validated",
        "pass",
    )
    if (
        value.get("schema")
        != "plan_aware_monotone_jepa_cost_v1.forensic_freeze_custody.v1"
        or value.get("experiment_id") != FORENSIC_EXPERIMENT_ID
        or value.get("sole_parent") != SOURCE_COMMIT
        or value.get("commit_subject") != FORENSIC_FREEZE_COMMIT_SUBJECT
        or value.get("changed_paths") != sorted(FORENSIC_REQUIRED_CHANGED_PATHS)
        or value.get("scientific_contract_digest")
        != BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or any(value.get(key) is not True for key in required_true)
        or value.get("scientific_archive_payload_files_opened") != 0
        or value.get("scientific_inputs_opened") != 0
        or value.get("files_reused") != 0
    ):
        raise ForensicContractError("forensic freeze custody receipt drift")
    validate_self_digest(value["committed_source_root_cause_proof"])
    archives = value.get("archive_custody")
    if (
        not isinstance(archives, Mapping)
        or archives.get("archive_count") != 3
        or archives.get("third_incident_technical_receipts_verified") is not True
        or archives.get("technical_receipt_distinct_paths") != 2
        or archives.get("technical_file_open_operations_this_validation") != 2
        or archives.get("scientific_archive_payload_files_opened") != 0
    ):
        raise ForensicContractError("forensic archive custody receipt drift")
    validate_self_digest(archives["third_incident_technical_receipts"])
    return copy.deepcopy(dict(value))


def _expected_active_diagnostic_namespace() -> tuple[set[str], set[str]]:
    directories = {"receipts", "streams", "synthetic"}
    main_file_keys = {
        "startup_stage_ledger",
        "child_stdout",
        "child_stderr",
        "child_traceback",
        "child_exception",
        "invocation",
        "environment",
        "command",
        "heartbeat",
        "read_guard_manifest",
        "read_guard_events",
        "forensic_freeze_custody",
        "last_stage_marker",
        "synthetic_results",
    }
    files = {
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key] for key in main_file_keys
    }
    synthetic_file_keys = {
        "child_stdout",
        "child_stderr",
        "child_traceback",
        "child_exception",
        "environment",
        "command",
        "heartbeat",
        "synthetic_technical_lifecycle",
        "read_guard_manifest",
        "read_guard_events",
        "last_stage_marker",
    }
    for fixture_id in PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES:
        prefix = f"synthetic/{fixture_id}"
        directories.update({prefix, f"{prefix}/receipts", f"{prefix}/streams"})
        files.update(
            f"{prefix}/{PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key]}"
            for key in synthetic_file_keys
        )
        if fixture_id != "MISSING_PATH":
            files.add(
                f"{prefix}/"
                f"{PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS['startup_stage_ledger']}"
            )
    return directories, files


def _validate_active_diagnostic_namespace(root: Path) -> dict[str, Any]:
    """Validate only the exact pre-child-authority technical namespace."""

    observed_umask = os.umask(PREEXECUTION_DIAGNOSTIC_UMASK)
    os.umask(observed_umask)
    if observed_umask != PREEXECUTION_DIAGNOSTIC_UMASK:
        raise ForensicContractError("active diagnostic child umask drift")

    expected_directories, expected_files = _expected_active_diagnostic_namespace()
    observed_directories: dict[str, os.stat_result] = {}
    observed_files: dict[str, os.stat_result] = {}
    inode_rows: set[tuple[int, int]] = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        if directory_path != root:
            relative_directory = str(directory_path.relative_to(root))
            observed_directories[relative_directory] = os.lstat(directory_path)
        for name in list(directory_names):
            candidate = directory_path / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ForensicContractError("active diagnostic directory type drift")
        for name in file_names:
            candidate = directory_path / name
            relative = str(candidate.relative_to(root))
            info = os.lstat(candidate)
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise ForensicContractError("active diagnostic file type/link drift")
            inode = (int(info.st_dev), int(info.st_ino))
            if inode in inode_rows:
                raise ForensicContractError("active diagnostic hardlink reuse")
            inode_rows.add(inode)
            observed_files[relative] = info
    if (
        set(observed_directories) != expected_directories
        or set(observed_files) != expected_files
    ):
        raise ForensicContractError("active diagnostic phase path-set drift")
    root_info = os.lstat(root)
    directory_stats = {".": root_info, **observed_directories}
    for relative, info in directory_stats.items():
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o755
            or info.st_uid != 1000
            or info.st_gid != 1000
        ):
            raise ForensicContractError(
                f"active diagnostic directory ownership/mode drift: {relative}"
            )
    for relative, info in observed_files.items():
        expected_mode = (
            0o600
            if relative
            == (
                "synthetic/MISSING_PATH/"
                + PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["last_stage_marker"]
            )
            else 0o644
        )
        if (
            stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_uid != 1000
            or info.st_gid != 1000
        ):
            raise ForensicContractError(
                f"active diagnostic file ownership/mode drift: {relative}"
            )
    rows = [
        {
            "path": relative,
            "kind": "DIRECTORY",
            "mode": stat.S_IMODE(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "nlink": int(info.st_nlink),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
        }
        for relative, info in sorted(directory_stats.items())
    ] + [
        {
            "path": relative,
            "kind": "FILE",
            "mode": stat.S_IMODE(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "nlink": int(info.st_nlink),
            "bytes": int(info.st_size),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
        }
        for relative, info in sorted(observed_files.items())
    ]
    return {
        "phase": "BEFORE_VALIDATE_FORENSIC_AUTHORITY_ACTION",
        "effective_umask": observed_umask,
        "directories": len(directory_stats),
        "files": len(observed_files),
        "rows": rows,
        "stat_inventory_sha256": canonical_json_sha256(rows),
        "symlinks": 0,
        "nonregular": 0,
        "multi_link_files": 0,
        "unexpected_paths": [],
        "pass": True,
    }


def _expected_final_diagnostic_namespace() -> tuple[set[str], set[str]]:
    """Return the exact terminal technical namespace, including its witness."""

    directories, files = _expected_active_diagnostic_namespace()
    files.update(
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key]
        for key in (
            "os_evidence",
            "umask_custody",
            "preexecution_only",
            "diagnostic_custody",
            "result",
            "final_namespace_inventory",
        )
    )
    return directories, files


def _observe_final_diagnostic_namespace(
    root: Path, *, inventory_receipt_must_exist: bool
) -> dict[str, Any]:
    expected_directories, expected_files = _expected_final_diagnostic_namespace()
    self_path = PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
        "final_namespace_inventory"
    ]
    observed_directories: dict[str, os.stat_result] = {}
    observed_files: dict[str, os.stat_result] = {}
    all_inodes: set[tuple[int, int]] = set()
    root_fd = _open_absolute_directory_no_follow(root)
    root_info = os.fstat(root_fd)
    all_inodes.add((int(root_info.st_dev), int(root_info.st_ino)))

    def visit(directory_fd: int, prefix: str) -> None:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for name in sorted(os.listdir(directory_fd)):
            if not name or "/" in name or name in {".", ".."}:
                raise ForensicContractError("final diagnostic entry-name drift")
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = name if not prefix else f"{prefix}/{name}"
            inode = (int(before.st_dev), int(before.st_ino))
            if inode in all_inodes:
                raise ForensicContractError(
                    "final diagnostic duplicate inode/hardlink"
                )
            all_inodes.add(inode)
            if stat.S_ISDIR(before.st_mode):
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if not _same_opened_stat(before, opened):
                        raise ForensicContractError(
                            "final diagnostic directory race"
                        )
                    observed_directories[relative] = opened
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                observed_files[relative] = before
            else:
                raise ForensicContractError(
                    "final diagnostic symlink/nonregular/multilink entry"
                )

    try:
        visit(root_fd, "")
    finally:
        os.close(root_fd)
    expected_observed_files = set(expected_files)
    if not inventory_receipt_must_exist:
        expected_observed_files.remove(self_path)
    if (
        set(observed_directories) != expected_directories
        or set(observed_files) != expected_observed_files
    ):
        raise ForensicContractError("final diagnostic phase path-set drift")
    directory_stats = {".": root_info, **observed_directories}
    for relative, info in directory_stats.items():
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o755
            or info.st_uid != 1000
            or info.st_gid != 1000
        ):
            raise ForensicContractError(
                f"final diagnostic directory ownership/mode drift: {relative}"
            )
    for relative, info in observed_files.items():
        expected_mode = (
            0o600
            if relative
            == (
                "synthetic/MISSING_PATH/"
                + PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["last_stage_marker"]
            )
            else 0o644
        )
        if (
            stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_uid != 1000
            or info.st_gid != 1000
        ):
            raise ForensicContractError(
                f"final diagnostic file ownership/mode drift: {relative}"
            )
    rows = [
        {
            "path": relative,
            "kind": "DIRECTORY",
            "mode": stat.S_IMODE(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "nlink": int(info.st_nlink),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
        }
        for relative, info in sorted(directory_stats.items())
    ] + [
        {
            "path": relative,
            "kind": "FILE",
            "mode": stat.S_IMODE(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "nlink": int(info.st_nlink),
            "bytes": int(info.st_size),
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
        }
        for relative, info in sorted(observed_files.items())
        if relative != self_path
    ]
    return {
        "rows": rows,
        "row_count": len(rows),
        "stat_inventory_sha256": canonical_json_sha256(rows),
        "expected_directory_count_including_root": len(expected_directories) + 1,
        "expected_file_count_including_self_excluded_witness": len(expected_files),
    }


def build_final_diagnostic_namespace_inventory(
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Build the terminal path/stat witness before exclusively creating it."""

    root = Path(diagnostic_root).absolute()
    if root != PREEXECUTION_DIAGNOSTIC_ROOT or root.is_symlink() or not root.is_dir():
        raise ForensicContractError("final diagnostic result root drift")
    witness_path = root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
        "final_namespace_inventory"
    ]
    if witness_path.exists() or witness_path.is_symlink():
        raise ForensicContractError("final diagnostic inventory witness is stale")
    observed = _observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=False
    )
    return attach_self_digest(
        {
            "schema": FINAL_DIAGNOSTIC_NAMESPACE_INVENTORY_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "phase": "TERMINAL_DIAGNOSTIC_BUNDLE_COMPLETE",
            "diagnostic_root": str(root),
            **observed,
            "self_exclusion": {
                "path": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
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


def validate_final_diagnostic_namespace_inventory(
    value: Mapping[str, Any],
    *,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Recompute the exact terminal inventory while self-excluding its receipt."""

    validate_self_digest(value)
    root = Path(diagnostic_root).absolute()
    witness_path = root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
        "final_namespace_inventory"
    ]
    if root != PREEXECUTION_DIAGNOSTIC_ROOT or witness_path.is_symlink() or not witness_path.is_file():
        raise ForensicContractError("final diagnostic inventory witness absent")
    observed = _observe_final_diagnostic_namespace(
        root, inventory_receipt_must_exist=True
    )
    expected = attach_self_digest(
        {
            "schema": FINAL_DIAGNOSTIC_NAMESPACE_INVENTORY_SCHEMA,
            "experiment_id": FORENSIC_EXPERIMENT_ID,
            "phase": "TERMINAL_DIAGNOSTIC_BUNDLE_COMPLETE",
            "diagnostic_root": str(root),
            **observed,
            "self_exclusion": {
                "path": PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
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
        raise ForensicContractError("final diagnostic namespace inventory drift")
    return copy.deepcopy(dict(value))


def write_final_diagnostic_namespace_inventory(
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Exclusively publish the self-excluding terminal namespace witness."""

    root = Path(diagnostic_root).absolute()
    value = build_final_diagnostic_namespace_inventory(root)
    payload = canonical_json_bytes(value) + b"\n"
    directory_rows = {
        str(row["path"]): row
        for row in value["rows"]
        if row["kind"] == "DIRECTORY"
    }
    _exclusive_write_relative_no_follow(
        root,
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[
            "final_namespace_inventory"
        ],
        payload,
        expected_directory_rows=directory_rows,
        mode=0o644,
    )
    return validate_final_diagnostic_namespace_inventory(
        value, diagnostic_root=root
    )


def write_preexecution_terminal_bundle(
    *,
    diagnostic_custody_receipt: Mapping[str, Any],
    synthetic_results_receipt: Mapping[str, Any],
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Write the pre-publication runtime result, then the final inventory."""

    root = Path(diagnostic_root).absolute()
    if root != PREEXECUTION_DIAGNOSTIC_ROOT:
        raise ForensicContractError("preexecution terminal root drift")
    result = build_preexecution_runtime_result(
        diagnostic_custody_receipt=diagnostic_custody_receipt,
        synthetic_results_receipt=synthetic_results_receipt,
    )
    result_path = root / PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["result"]
    root_fd = _open_absolute_directory_no_follow(root)
    try:
        root_info = os.fstat(root_fd)
    finally:
        os.close(root_fd)
    root_row = {
        "path": ".",
        "kind": "DIRECTORY",
        "mode": stat.S_IMODE(root_info.st_mode),
        "uid": int(root_info.st_uid),
        "gid": int(root_info.st_gid),
        "nlink": int(root_info.st_nlink),
        "device": int(root_info.st_dev),
        "inode": int(root_info.st_ino),
    }
    _exclusive_write_relative_no_follow(
        root,
        PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS["result"],
        canonical_json_bytes(result) + b"\n",
        expected_directory_rows={".": root_row},
        mode=0o644,
    )
    inventory = write_final_diagnostic_namespace_inventory(root)
    return {
        "runtime_result": result,
        "final_namespace_inventory": inventory,
        "v2_specification_publication": result[
            "v2_specification_publication"
        ],
        "tracked_publication_completed": False,
        "automatic_execution_authorized": False,
        "pass": True,
    }


def validate_active_diagnostic_freeze_custody(
    repo_root: str | Path,
    receipt: Mapping[str, Any],
    *,
    diagnostic_root: str | Path = PREEXECUTION_DIAGNOSTIC_ROOT,
) -> dict[str, Any]:
    """Child-side revalidation after the launcher has created only its root."""

    root = Path(repo_root).resolve()
    value = validate_forensic_freeze_custody_receipt(receipt)
    if (
        _git_text(root, ["rev-parse", "HEAD"]) != value["repo_head"]
        or _git_text(root, ["status", "--porcelain=v1"])
    ):
        raise ForensicContractError("active diagnostic Git custody drift")
    active_root = Path(diagnostic_root).absolute()
    if (
        active_root != PREEXECUTION_DIAGNOSTIC_ROOT
        or active_root.is_symlink()
        or not active_root.is_dir()
        or BASE.OUTPUT_ROOT.exists()
        or BASE.OUTPUT_ROOT.is_symlink()
        or list(
            BASE.OUTPUT_ROOT.parent.glob(
                f".{BASE.OUTPUT_ROOT.name}.attempt-*"
            )
        )
    ):
        raise ForensicContractError("active diagnostic namespace drift")
    authorities = _validate_forensic_authority_files(root)
    if (
        authorities["forensic_authority_source_closure"]
        != value["forensic_authority_source_closure"]
    ):
        raise ForensicContractError("active diagnostic authority closure drift")
    base = _validate_base_authorities_read_only(root)
    archives = _validate_archive_custody_metadata_only()
    if archives != value["archive_custody"]:
        raise ForensicContractError("active diagnostic archive custody drift")
    active_inventory = _validate_active_diagnostic_namespace(active_root)
    return attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "active_diagnostic_freeze_custody.v1"
            ),
            "source_freeze_commit": value["repo_head"],
            "forensic_freeze_custody": copy.deepcopy(dict(value)),
            "forensic_authority_validated": True,
            "base_authorities_read_only_validated": base["pass"],
            "archive_custody_metadata_only_validated": archives["pass"],
            "active_diagnostic_root": str(active_root),
            "pristine_root_absence_bound_by_prelaunch_freeze": (
                value["output_namespace"]["diagnostic_root_absent"] is True
            ),
            "active_diagnostic_namespace_inventory": active_inventory,
            "scientific_output_namespace_absent": True,
            "scientific_inputs_opened": 0,
            "files_reused": 0,
            "pass": True,
        }
    )
