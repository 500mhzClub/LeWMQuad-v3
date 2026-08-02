#!/usr/bin/env python3
"""Supervise one exact post-calibration 2,304-branch WM-A pilot.

The supervisor validates a caller-pinned committed authority, performs the
same exact graphics preflight as calibration, consumes the attempt before it
starts the collector, checks physics receipts, joins the frozen
train/eval pilot, and checks the joined manifest.  It provides no retry,
resume, refill, screening, model-training, or scientific-promotion path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded_collector  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as calibration_supervisor  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as plan_contract  # noqa: E402


TERMINAL_SCHEMA = (
    "lewm_go2_world_model_bounded_branch_experiment_supervision_terminal_v1"
)
COLLECTOR_RELATIVE = (
    "scripts/collect_go2_world_model_bounded_branch_experiment_authorized_v1.py"
)
CHECKER_RELATIVE = "scripts/check_go2_world_model_counterfactual_pilot_v1.py"
JOINER_RELATIVE = "scripts/join_go2_world_model_counterfactual_pilot_v1.py"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class BoundedBranchSupervisionError(RuntimeError):
    """Raised when the one-shot bounded pilot fails closed."""


def _owned_reservation(
    attempt_root: Path,
    *,
    nonce: str,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    supervisor_pid: int | None = None,
) -> bool:
    path = attempt_root / "reservation.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, Mapping)
        and (
            supervisor_pid is None
            or set(value)
            == {
                "schema",
                "status",
                "attempt",
                "plan_binding",
                "authority_binding",
                "supervisor_nonce",
                "supervisor_pid",
                "root_creation_consumes_attempt",
                "reservation_records_consumed_attempt",
                "retry_authorized",
                "resume_authorized",
                "overwrite_authorized",
                "refill_authorized",
            }
        )
        and value.get("schema")
        == "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
        and value.get("status") == "RESERVED_ATTEMPT_CONSUMED"
        and value.get("attempt") == authority["attempt"]
        and value.get("plan_binding") == plan_binding
        and value.get("authority_binding") == authority_binding
        and value.get("supervisor_nonce") == nonce
        and value.get("root_creation_consumes_attempt") is True
        and value.get("reservation_records_consumed_attempt") is True
        and (
            supervisor_pid is None
            or value.get("supervisor_pid") == supervisor_pid
        )
        and value.get("retry_authorized") is False
        and value.get("resume_authorized") is False
        and value.get("overwrite_authorized") is False
        and value.get("refill_authorized") is False
    )


def _reserve_attempt_v1(
    attempt_root: Path,
    *,
    nonce: str,
    supervisor_pid: int,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically create the attempt root, then consume its sole reservation."""

    if _SHA.fullmatch(nonce) is None:
        raise BoundedBranchSupervisionError("supervisor ownership nonce is invalid")
    if (
        isinstance(supervisor_pid, bool)
        or not isinstance(supervisor_pid, int)
        or supervisor_pid <= 1
        or supervisor_pid != os.getpid()
    ):
        raise BoundedBranchSupervisionError("supervisor PID identity is invalid")
    try:
        development_root = (REPO_ROOT / ".generated/dev").resolve(strict=True)
        selected = Path(os.path.abspath(attempt_root))
        relative = selected.relative_to(development_root)
        if not relative.parts:
            raise BoundedBranchSupervisionError(
                "attempt root cannot equal .generated/dev"
            )
        cursor = development_root
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise BoundedBranchSupervisionError(
                    "attempt root traverses a symlink"
                )
            if cursor.exists() and not cursor.is_dir():
                raise BoundedBranchSupervisionError(
                    "attempt path has a non-directory component"
                )
        if selected.exists() or selected.is_symlink():
            raise BoundedBranchSupervisionError("attempt root is not fresh")
        output_root = pilot.fresh_development_output_root(
            selected, development_root=development_root
        )
        binding = pilot.write_json_exclusive(
            output_root / "reservation.json",
            {
                "schema": (
                    "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
                ),
                "status": "RESERVED_ATTEMPT_CONSUMED",
                "attempt": dict(authority["attempt"]),
                "plan_binding": dict(plan_binding),
                "authority_binding": dict(authority_binding),
                "supervisor_nonce": nonce,
                "supervisor_pid": supervisor_pid,
                "root_creation_consumes_attempt": True,
                "reservation_records_consumed_attempt": True,
                "retry_authorized": False,
                "resume_authorized": False,
                "overwrite_authorized": False,
                "refill_authorized": False,
            },
        )
    except (
        BoundedBranchSupervisionError,
        OSError,
        ValueError,
        pilot.PilotContractError,
    ) as exc:
        raise BoundedBranchSupervisionError(
            f"could not consume exclusive supervisor reservation: {exc}"
        ) from exc
    return {
        "path": "reservation.json",
        "file_sha256": str(binding["file_sha256"]),
        "byte_count": int(binding["byte_count"]),
    }


def _run_collector_once_with_vram_ceiling(
    argv: Sequence[str],
    *,
    timeout: float,
    env: Mapping[str, str],
    sampler: Any,
    ceiling_bytes: int,
    enforcement: dict[str, Any],
) -> dict[str, Any]:
    """Run exactly once and terminate the collector on an observed VRAM breach."""

    if timeout <= 0.0:
        raise BoundedBranchSupervisionError("hard wall ceiling exhausted")
    if ceiling_bytes <= 0:
        raise BoundedBranchSupervisionError("VRAM ceiling is invalid")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        start_new_session=True,
        env=dict(env),
    )
    enforcement["collector_pid"] = process.pid
    enforcement["collector_started"] = True
    try:
        while True:
            peak = int(sampler.peak_used_bytes)
            read_errors = int(sampler.read_errors)
            enforcement["peak_observed_during_collector_bytes"] = max(
                int(enforcement["peak_observed_during_collector_bytes"]), peak
            )
            if read_errors != 0:
                enforcement["collector_terminated"] = True
                enforcement["termination_reason"] = "vram_counter_read_error"
                calibration_supervisor._terminate_process_group(process)  # noqa: SLF001
                raise BoundedBranchSupervisionError(
                    "active selected-device VRAM monitor had read errors"
                )
            if peak > ceiling_bytes:
                enforcement["collector_terminated"] = True
                enforcement["termination_reason"] = "vram_ceiling_exceeded"
                calibration_supervisor._terminate_process_group(process)  # noqa: SLF001
                raise BoundedBranchSupervisionError(
                    "active selected-device VRAM ceiling exceeded "
                    f"({peak} > {ceiling_bytes})"
                )
            returncode = process.poll()
            if returncode is not None:
                break
            if time.monotonic() - started >= timeout:
                enforcement["collector_terminated"] = True
                enforcement["termination_reason"] = "wall_ceiling_exceeded"
                calibration_supervisor._terminate_process_group(process)  # noqa: SLF001
                raise BoundedBranchSupervisionError(
                    "supervised collector exceeded wall ceiling"
                )
            time.sleep(min(0.02, max(0.001, timeout / 100.0)))
    except BaseException:
        if process.poll() is None:
            enforcement["collector_terminated"] = True
            if enforcement.get("termination_reason") is None:
                enforcement["termination_reason"] = "supervision_interrupted"
            calibration_supervisor._terminate_process_group(process)  # noqa: SLF001
        raise
    elapsed = time.monotonic() - started
    enforcement["collector_exit_code"] = int(returncode)
    if returncode != 0:
        raise BoundedBranchSupervisionError(
            f"supervised collector exited with status {returncode}"
        )
    return {
        "argv": list(argv),
        "elapsed_seconds": elapsed,
        "exit_code": int(returncode),
        "active_vram_ceiling_enforced": True,
    }


def _check_report(
    path: Path,
    *,
    expected_phase: str,
    expected_purpose: str,
    expected_manifest_binding: Mapping[str, Any],
) -> dict[str, Any]:
    binding = pilot.file_binding(path)
    report, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label=f"bounded pilot {expected_phase} check",
    )
    if (
        actual != binding
        or not isinstance(report, Mapping)
        or report.get("schema") != checker.REPORT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("phase") != expected_phase
        or report.get("purpose") != expected_purpose
        or report.get("manifest_binding") != dict(expected_manifest_binding)
        or report.get("runtime_payloads_opened") is not False
        or report.get("rgb_bytes_opened") is not False
        or report.get("checkpoints_opened") is not False
    ):
        raise BoundedBranchSupervisionError(
            f"receipt-only checker did not pass {expected_phase} exactly"
        )
    return binding


def _load_physics_result_if_present(
    attempt_root: Path,
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Bind either collector terminal status without opening any RGB leaf."""

    path = attempt_root / "physics_result.json"
    if not path.exists():
        return None, None
    if path.is_symlink() or not path.is_file():
        raise BoundedBranchSupervisionError(
            "physics result is not a regular non-symlink file"
        )
    binding = pilot.file_binding(path)
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label="bounded branch physics result",
    )
    status = value.get("status") if isinstance(value, Mapping) else None
    failure = value.get("failure") if isinstance(value, Mapping) else None
    if (
        actual != binding
        or not isinstance(value, Mapping)
        or value.get("schema") != pilot.PHYSICS_RESULT_SCHEMA
        or status not in {"PHYSICS_COMPLETE", "FAILED"}
        or value.get("attempt_id") != plan["attempt_id"]
        or value.get("purpose") != "bounded_wm_a_pilot"
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("reservation_binding") != dict(reservation_binding)
        or value.get("review_binding") != authority["review_binding"]
        or value.get("source_bindings") != authority["source_bindings"]
        or value.get("caps") != authority["caps"]
        or value.get("citable_as_scientific_evidence") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or (status == "PHYSICS_COMPLETE" and failure is not None)
        or (status == "FAILED" and (not isinstance(failure, Mapping) or not failure))
    ):
        raise BoundedBranchSupervisionError(
            "collector physics result is not an exact terminal receipt"
        )
    return dict(value), binding


def supervise_v1(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        plan_contract._reject_protected_path(  # noqa: SLF001
            authority_path, label="bounded branch authority"
        )
    except plan_contract.BoundedBranchPlanError as exc:
        raise BoundedBranchSupervisionError(str(exc)) from exc
    raw_authority, authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="bounded branch authority",
    )
    plan_binding = raw_authority.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise BoundedBranchSupervisionError("authority plan binding is absent")
    authority, actual_authority_binding, plan, actual_plan_binding = (
        bounded_collector.load_and_validate_v1(
            plan_path=Path(str(plan_binding["path"])),
            expected_plan_byte_count=int(plan_binding["byte_count"]),
            expected_plan_sha256=str(plan_binding["file_sha256"]),
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
        )
    )
    if actual_authority_binding != authority_binding or actual_plan_binding != dict(plan_binding):
        raise BoundedBranchSupervisionError("authority/plan binding changed")
    invocation = str(calibration_supervisor._validate_python_invocation(plan))  # noqa: SLF001
    child_env = calibration_supervisor._child_environment(plan)  # noqa: SLF001
    attempt_root = calibration_supervisor._require_fresh_attempt_root(  # noqa: SLF001
        plan["output_root"]
    )
    wall_ceiling = float(authority["caps"]["wall_seconds"])
    wall_started = time.monotonic()
    graphics = calibration_supervisor._run_graphics_preflight(  # noqa: SLF001
        plan,
        child_env=child_env,
        wall_started=wall_started,
        wall_ceiling=wall_ceiling,
    )
    used, total, vendor, device = calibration_supervisor._selected_gpu_memory_files(plan)  # noqa: SLF001
    vram_ceiling = int(authority["caps"]["selected_device_vram_byte_ceiling"])
    sampler = calibration_supervisor._GlobalVramSampler(  # noqa: SLF001
        used,
        total,
        vendor_id=vendor,
        device_id=device,
        interval_seconds=0.02,
    )
    if int(sampler.baseline_used_bytes) > vram_ceiling:
        raise BoundedBranchSupervisionError(
            "selected-device VRAM baseline already exceeds the authority ceiling"
        )
    nonce = secrets.token_hex(32)
    supervisor_pid = os.getpid()
    reservation_binding = _reserve_attempt_v1(
        attempt_root,
        nonce=nonce,
        supervisor_pid=supervisor_pid,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=actual_plan_binding,
    )
    phases: list[dict[str, Any]] = [graphics]
    physics_binding = None
    physics_check_binding = None
    joined_manifest_binding = None
    joined_check_binding = None
    gpu_measurement = None
    failure = None
    sampler_started = False
    vram_enforcement = {
        "enabled": True,
        "scope": "selected_device_global_vram_not_process_attributed",
        "ceiling_bytes": vram_ceiling,
        "sample_interval_seconds": float(sampler.interval_seconds),
        "collector_started": False,
        "collector_pid": None,
        "collector_exit_code": None,
        "collector_terminated": False,
        "termination_reason": None,
        "peak_observed_during_collector_bytes": int(sampler.baseline_used_bytes),
    }
    collector_argv = [
        invocation,
        str((REPO_ROOT / COLLECTOR_RELATIVE).resolve()),
        "--plan",
        str(Path(actual_plan_binding["path"]).resolve()),
        "--expected-plan-byte-count",
        str(actual_plan_binding["byte_count"]),
        "--expected-plan-sha256",
        str(actual_plan_binding["file_sha256"]),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--expected-authority-sha256",
        str(authority_binding["file_sha256"]),
        "--supervisor-nonce",
        nonce,
        "--supervisor-pid",
        str(supervisor_pid),
    ]
    try:
        sampler.start()
        sampler_started = True
        phases.append(_run_collector_once_with_vram_ceiling(
            collector_argv,
            timeout=calibration_supervisor._remaining_wall(  # noqa: SLF001
                started=wall_started, ceiling=wall_ceiling
            ),
            env=child_env,
            sampler=sampler,
            ceiling_bytes=vram_ceiling,
            enforcement=vram_enforcement,
        ))
        physics_path = attempt_root / "physics_result.json"
        physics_result, physics_binding = _load_physics_result_if_present(
            attempt_root,
            plan=plan,
            plan_binding=actual_plan_binding,
            authority=authority,
            authority_binding=authority_binding,
            reservation_binding=reservation_binding,
        )
        if physics_result is None or physics_binding is None:
            raise BoundedBranchSupervisionError(
                "collector did not emit a bound physics result"
            )
        if physics_result["status"] != "PHYSICS_COMPLETE":
            raise BoundedBranchSupervisionError(
                "collector emitted a terminal failed physics result"
            )
        physics_check_path = attempt_root / "physics_receipt_check.json"
        phases.append(calibration_supervisor._run_once(  # noqa: SLF001
            [
                invocation,
                str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
                "--manifest", str(physics_path),
                "--expected-file-sha256", str(physics_binding["file_sha256"]),
                "--expected-byte-count", str(physics_binding["byte_count"]),
                "--output", str(physics_check_path),
            ],
            timeout=calibration_supervisor._remaining_wall(  # noqa: SLF001
                started=wall_started, ceiling=wall_ceiling
            ),
            env=child_env,
        ))
        physics_check_binding = _check_report(
            physics_check_path,
            expected_phase="physics_collection",
            expected_purpose="bounded_wm_a_pilot",
            expected_manifest_binding=physics_binding,
        )
        gate_binding = authority["calibration_gate_binding"]
        gate, _ = pilot.read_bound_json(
            Path(str(gate_binding["path"])),
            expected_sha256=str(gate_binding["file_sha256"]),
            expected_byte_count=int(gate_binding["byte_count"]),
            label="bounded branch calibration gate",
        )
        calibration_binding = gate["calibration_receipt_binding"]
        phases.append(calibration_supervisor._run_once(  # noqa: SLF001
            [
                invocation,
                str((REPO_ROOT / JOINER_RELATIVE).resolve()),
                "--collection", str(physics_path),
                "--expected-collection-sha256", str(physics_binding["file_sha256"]),
                "--expected-collection-byte-count", str(physics_binding["byte_count"]),
                "--calibration-receipt", str(calibration_binding["path"]),
                "--expected-calibration-sha256", str(calibration_binding["file_sha256"]),
                "--expected-calibration-byte-count", str(calibration_binding["byte_count"]),
            ],
            timeout=calibration_supervisor._remaining_wall(  # noqa: SLF001
                started=wall_started, ceiling=wall_ceiling
            ),
            env=child_env,
        ))
        joined_path = attempt_root / "manifest.json"
        joined_manifest_binding = pilot.file_binding(joined_path)
        joined_check_path = attempt_root / "joined_receipt_check.json"
        phases.append(calibration_supervisor._run_once(  # noqa: SLF001
            [
                invocation,
                str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
                "--manifest", str(joined_path),
                "--expected-file-sha256", str(joined_manifest_binding["file_sha256"]),
                "--expected-byte-count", str(joined_manifest_binding["byte_count"]),
                "--output", str(joined_check_path),
            ],
            timeout=calibration_supervisor._remaining_wall(  # noqa: SLF001
                started=wall_started, ceiling=wall_ceiling
            ),
            env=child_env,
        ))
        joined_check_binding = _check_report(
            joined_check_path,
            expected_phase="joined_pilot",
            expected_purpose="bounded_wm_a_pilot",
            expected_manifest_binding=joined_manifest_binding,
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if physics_binding is None:
            try:
                _physics_result, physics_binding = _load_physics_result_if_present(
                    attempt_root,
                    plan=plan,
                    plan_binding=actual_plan_binding,
                    authority=authority,
                    authority_binding=authority_binding,
                    reservation_binding=reservation_binding,
                )
            except BaseException as receipt_exc:
                failure = (
                    f"{failure}; physics result binding failed: "
                    f"{type(receipt_exc).__name__}: {receipt_exc}"
                )
    finally:
        try:
            if sampler_started:
                gpu_measurement = sampler.stop()
            if gpu_measurement is not None and gpu_measurement["read_errors"] != 0:
                sampler_failure = "BoundedBranchSupervisionError: GPU sampler had read errors"
                failure = sampler_failure if failure is None else f"{failure}; {sampler_failure}"
            elif gpu_measurement is not None and int(
                gpu_measurement["peak_used_bytes"]
            ) > vram_ceiling:
                sampler_failure = (
                    "BoundedBranchSupervisionError: selected-device VRAM ceiling "
                    "was exceeded"
                )
                failure = sampler_failure if failure is None else f"{failure}; {sampler_failure}"
        except BaseException as exc:
            sampler_failure = f"{type(exc).__name__}: {exc}"
            failure = sampler_failure if failure is None else f"{failure}; {sampler_failure}"
    elapsed = time.monotonic() - wall_started
    if failure is None and elapsed > wall_ceiling:
        failure = (
            "BoundedBranchSupervisionError: terminal validation exceeded hard wall "
            f"ceiling ({elapsed:.6f} > {wall_ceiling:.6f})"
        )
    owned = _owned_reservation(
        attempt_root,
        nonce=nonce,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=actual_plan_binding,
        supervisor_pid=supervisor_pid,
    )
    if attempt_root.exists() and not owned:
        ownership_failure = "BoundedBranchSupervisionError: reservation ownership changed"
        failure = ownership_failure if failure is None else f"{failure}; {ownership_failure}"
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": (
            "COMPLETE_PENDING_INDEPENDENT_TERMINAL_REVIEW"
            if failure is None
            else "CONSUMED_TERMINAL_FAILURE"
        ),
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authorizes_refill_or_screening": False,
        "scientific_verdict_emitted": False,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "authority_binding": authority_binding,
        "plan_binding": actual_plan_binding,
        "reservation_binding": reservation_binding,
        "calibration_gate_binding": authority["calibration_gate_binding"],
        "source_commit": authority["source_commit"],
        "attempt_root": str(attempt_root),
        "wall_elapsed_seconds": elapsed,
        "wall_ceiling_seconds": wall_ceiling,
        "phase_receipts": phases,
        "physics_result_binding": physics_binding,
        "physics_receipt_check_binding": physics_check_binding,
        "joined_manifest_binding": joined_manifest_binding,
        "joined_receipt_check_binding": joined_check_binding,
        "gpu_memory_measurement": gpu_measurement,
        "active_vram_ceiling_enforcement": vram_enforcement,
        "same_sensor_visual_domain_contract": {
            "enforced": True,
            "task_valid_near_wall_or_low_texture_observations_screened": False,
            "camera_or_render_failure_causes_terminal_failure": True,
        },
        "evaluation_contract": authority["evaluation_contract"],
        "failure": failure,
        "terminal_reviewer": authority["external_supervisor"]["terminal_reviewer"],
        "supervisor_nonce": nonce,
        "supervisor_pid": supervisor_pid,
    }
    terminal_binding = None
    if owned:
        terminal_binding = pilot.write_json_exclusive(
            attempt_root / "terminal_supervision.json", terminal
        )
    return terminal, terminal_binding


def _signal(signum: int, _frame: Any) -> None:
    raise BoundedBranchSupervisionError(f"supervisor received signal {signum}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    args = parser.parse_args(argv)
    if args.expected_authority_byte_count <= 0 or _SHA.fullmatch(args.expected_authority_sha256) is None:
        raise SystemExit("caller authority binding is malformed")
    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)
    terminal, binding = supervise_v1(
        args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
    )
    if binding is None:
        print(
            "reserved attempt lost terminal ownership; do not retry or resume",
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": terminal["status"], "terminal": binding}, sort_keys=True))
    return 0 if terminal["status"] == "COMPLETE_PENDING_INDEPENDENT_TERMINAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BoundedBranchSupervisionError",
    "_load_physics_result_if_present",
    "_reserve_attempt_v1",
    "_run_collector_once_with_vram_ceiling",
    "supervise_v1",
]
