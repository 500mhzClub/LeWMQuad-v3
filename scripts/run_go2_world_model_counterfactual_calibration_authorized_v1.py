#!/usr/bin/env python3
"""Supervise one exactly authorized 160-branch counterfactual calibration.

This source creates no authority.  It validates one caller-bound, committed
calibration authority and its reviewed source closure, installs the exact
plan-bound runtime selector environment, and starts the collector once under
one hard wall clock.  If collection succeeds, the same wall clock covers the
receipt-only checker and calibration analyzer.  No retry or resume path is
provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import analyze_go2_world_model_counterfactual_calibration_v1 as analyzer  # noqa: E402
from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector  # noqa: E402


PURPOSE = "sizing_calibration_only"
AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_160_BRANCH_CALIBRATION"
TERMINAL_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_supervision_terminal_v1"
)
COLLECTOR_RELATIVE = Path(
    "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
)
CHECKER_RELATIVE = Path(
    "scripts/check_go2_world_model_counterfactual_pilot_v1.py"
)
ANALYZER_RELATIVE = Path(
    "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
)
DEVELOPMENT_ROOT = REPO_ROOT / ".generated" / "dev"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CalibrationSupervisionError(RuntimeError):
    """Raised when the one-shot calibration boundary fails closed."""


def _selected_gpu_memory_files(
    plan: Mapping[str, Any],
    *,
    drm_root: Path = Path("/sys/class/drm"),
) -> tuple[Path, Path, str, str]:
    """Resolve the one DRM device named by the bound graphics expectation."""

    expectation = plan["execution_contract"]["graphics_preflight"]
    vendor_id = str(expectation["vulkan_vendor_id"]).lower()
    device_id = str(expectation["vulkan_device_id"]).lower()
    matches: list[tuple[Path, Path]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card.name) is None:
            continue
        device = card / "device"
        try:
            observed_vendor = (device / "vendor").read_text().strip().lower()
            observed_device = (device / "device").read_text().strip().lower()
        except OSError:
            continue
        if observed_vendor == vendor_id and observed_device == device_id:
            matches.append(
                (device / "mem_info_vram_used", device / "mem_info_vram_total")
            )
    if len(matches) != 1 or any(not path.is_file() for path in matches[0]):
        raise CalibrationSupervisionError(
            "bound GPU does not expose one exact sysfs VRAM counter pair"
        )
    used_path, total_path = matches[0]
    return used_path, total_path, vendor_id, device_id


class _GlobalVramSampler:
    """Sample selected-device global VRAM; values are not process-attributed."""

    def __init__(
        self,
        used_path: Path,
        total_path: Path,
        *,
        vendor_id: str,
        device_id: str,
        interval_seconds: float = 0.05,
    ) -> None:
        self.used_path = used_path
        self.total_path = total_path
        self.vendor_id = vendor_id
        self.device_id = device_id
        self.interval_seconds = interval_seconds
        self.total_bytes = self._read_counter(total_path)
        self.baseline_used_bytes = self._read_counter(used_path)
        self.peak_used_bytes = self.baseline_used_bytes
        self.sample_count = 1
        self.read_errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    @staticmethod
    def _read_counter(path: Path) -> int:
        try:
            value = int(path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise CalibrationSupervisionError(
                f"invalid GPU memory counter: {path}"
            ) from exc
        if value < 0:
            raise CalibrationSupervisionError(f"negative GPU memory counter: {path}")
        return value

    def _sample_once(self) -> None:
        try:
            used = self._read_counter(self.used_path)
        except CalibrationSupervisionError:
            self.read_errors += 1
            return
        self.sample_count += 1
        self.peak_used_bytes = max(self.peak_used_bytes, used)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(1.0, 4.0 * self.interval_seconds))
        if self._thread.is_alive():
            raise CalibrationSupervisionError("GPU memory sampler did not stop")
        self._sample_once()
        return {
            "scope": "selected_device_global_vram_not_process_attributed",
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "used_counter_path": str(self.used_path),
            "total_counter_path": str(self.total_path),
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
            "read_errors": self.read_errors,
            "baseline_used_bytes": self.baseline_used_bytes,
            "peak_used_bytes": self.peak_used_bytes,
            "peak_delta_above_baseline_bytes": max(
                0, self.peak_used_bytes - self.baseline_used_bytes
            ),
            "device_total_bytes": self.total_bytes,
            "attribution_limitation": (
                "global selected-device usage can include concurrent processes"
            ),
        }


def _child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    """Remove ambient selectors and install only the plan-bound mapping."""

    expected = plan["execution_contract"]["environment"]
    if expected != pilot.EXECUTION_ENVIRONMENT:
        raise CalibrationSupervisionError("execution environment contract changed")
    child = dict(os.environ)
    for key in collector._SANITIZED_SELECTOR_KEYS:  # noqa: SLF001
        child.pop(key, None)
    child.update({str(key): str(value) for key, value in expected.items()})
    return child


def _validate_python_invocation(plan: Mapping[str, Any]) -> Path:
    invocation = Path(str(plan["execution_contract"]["python_invocation_path"]))
    target_binding = plan["runtime_bindings"]["python_executable_target"]
    environment_binding = plan["runtime_bindings"]["python_environment_config"]
    try:
        pilot.require_binding(target_binding, label="Python executable target")
        pilot.require_binding(environment_binding, label="Python environment config")
    except pilot.PilotContractError as exc:
        raise CalibrationSupervisionError(str(exc)) from exc
    target = Path(str(target_binding["path"]))
    environment_config = Path(str(environment_binding["path"]))
    if (
        not invocation.is_absolute()
        or invocation.resolve(strict=True) != target.resolve(strict=True)
        or environment_config.name != "pyvenv.cfg"
        or invocation.parent.parent != environment_config.parent
    ):
        raise CalibrationSupervisionError(
            "Python invocation does not resolve inside the bound environment"
        )
    return invocation


def _run_graphics_preflight(
    plan: Mapping[str, Any],
    *,
    child_env: Mapping[str, str],
    wall_started: float,
    wall_ceiling: float,
) -> dict[str, Any]:
    """Verify the exact EGL/Vulkan selection before consuming the attempt."""

    runtime = plan["runtime_bindings"]
    expectation = plan["execution_contract"]["graphics_preflight"]
    if expectation != pilot.GRAPHICS_PREFLIGHT_EXPECTATION:
        raise CalibrationSupervisionError("graphics preflight expectation changed")
    try:
        eglinfo = pilot.require_binding(
            runtime["eglinfo_executable"], label="eglinfo executable"
        )
        vulkaninfo = pilot.require_binding(
            runtime["vulkaninfo_executable"], label="vulkaninfo executable"
        )
    except pilot.PilotContractError as exc:
        raise CalibrationSupervisionError(str(exc)) from exc

    vulkan_timeout = min(
        90.0,
        _remaining_wall(started=wall_started, ceiling=wall_ceiling),
    )
    if vulkan_timeout <= 0.0:
        raise CalibrationSupervisionError("hard wall ceiling exhausted in preflight")
    try:
        vulkan = subprocess.run(
            [str(vulkaninfo["path"]), "--summary"],
            cwd=REPO_ROOT,
            env=dict(child_env),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=vulkan_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CalibrationSupervisionError("Vulkan preflight exceeded wall ceiling") from exc
    if vulkan.returncode != 0:
        raise CalibrationSupervisionError(
            f"Vulkan preflight exited with status {vulkan.returncode}"
        )
    gpu_pattern = re.compile(
        rf"GPU{int(expectation['vulkan_device_index'])}:.*?"
        rf"vendorID\s*=\s*{re.escape(str(expectation['vulkan_vendor_id']))}.*?"
        rf"deviceID\s*=\s*{re.escape(str(expectation['vulkan_device_id']))}.*?"
        rf"deviceName\s*=\s*{re.escape(str(expectation['vulkan_device_name']))}(?:\n|$)",
        re.DOTALL,
    )
    if gpu_pattern.search(vulkan.stdout) is None:
        raise CalibrationSupervisionError(
            "selected Vulkan GPU is not the exact bound device"
        )
    if re.search(r"^GPU1:", vulkan.stdout, re.MULTILINE) is not None:
        raise CalibrationSupervisionError(
            "Vulkan selector did not reduce enumeration to GPU0"
        )

    egl_timeout = min(
        90.0,
        _remaining_wall(started=wall_started, ceiling=wall_ceiling),
    )
    if egl_timeout <= 0.0:
        raise CalibrationSupervisionError("hard wall ceiling exhausted in preflight")
    try:
        egl = subprocess.run(
            [str(eglinfo["path"]), "-B"],
            cwd=REPO_ROOT,
            env=dict(child_env),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=egl_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CalibrationSupervisionError("EGL preflight exceeded wall ceiling") from exc
    if egl.returncode != expectation["eglinfo_expected_exit_code"]:
        raise CalibrationSupervisionError(
            f"EGL preflight exited with status {egl.returncode}"
        )
    device_index = int(expectation["egl_device_index"])
    match = re.search(
        rf"Device #{device_index}:\s*\n(?P<section>.*?)(?=\nDevice #\d+:|\Z)",
        egl.stdout,
        re.DOTALL,
    )
    if (
        match is None
        or str(expectation["egl_renderer_name_contains"])
        not in match.group("section")
    ):
        raise CalibrationSupervisionError(
            "selected EGL device is not the exact bound device"
        )
    return {
        "phase": "graphics_preflight",
        "status": "PASS",
        "environment": dict(plan["execution_contract"]["environment"]),
        "expectation": dict(expectation),
        "vulkan_stdout_sha256": hashlib.sha256(
            vulkan.stdout.encode("utf-8")
        ).hexdigest(),
        "egl_stdout_sha256": hashlib.sha256(egl.stdout.encode("utf-8")).hexdigest(),
        "egl_stderr_sha256": hashlib.sha256(egl.stderr.encode("utf-8")).hexdigest(),
        "egl_exit_code": int(egl.returncode),
    }


def _require_fresh_attempt_root(path_text: object) -> Path:
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise CalibrationSupervisionError("attempt root must be absolute")
    development_root = DEVELOPMENT_ROOT.resolve(strict=True)
    candidate = Path(path_text)
    normalized = candidate.resolve(strict=False)
    try:
        relative = normalized.relative_to(development_root)
    except ValueError as exc:
        raise CalibrationSupervisionError("attempt root escapes .generated/dev") from exc
    if not relative.parts:
        raise CalibrationSupervisionError("attempt root cannot equal .generated/dev")
    cursor = development_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise CalibrationSupervisionError(
                f"attempt path contains a symlink: {cursor}"
            )
        if not cursor.exists():
            break
    if candidate.exists() or candidate.is_symlink():
        raise CalibrationSupervisionError(f"attempt root is not fresh: {candidate}")
    return candidate


def _source_binding(
    authority: Mapping[str, Any], *, name: str
) -> Mapping[str, Any]:
    matches = [
        row["binding"]
        for row in authority["source_bindings"]
        if row["name"] == name
    ]
    if len(matches) != 1:
        raise CalibrationSupervisionError(
            f"reviewed source closure has {len(matches)} entries for {name}"
        )
    return matches[0]


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the authority, exact plan, source review, and committed closure."""

    try:
        raw_authority, authority_binding = pilot.read_bound_json(
            authority_path,
            expected_sha256=expected_authority_sha256,
            expected_byte_count=expected_authority_byte_count,
            label="counterfactual calibration authority",
        )
        if not isinstance(raw_authority, Mapping):
            raise pilot.PilotContractError("calibration authority must be an object")
        raw_plan_binding = pilot._validate_binding_shape(  # noqa: SLF001
            raw_authority.get("plan_binding"), label="authority plan binding"
        )
        raw_plan, plan_binding = pilot.read_bound_json(
            Path(str(raw_plan_binding["path"])),
            expected_sha256=str(raw_plan_binding["file_sha256"]),
            expected_byte_count=int(raw_plan_binding["byte_count"]),
            label="counterfactual calibration plan",
        )
        if plan_binding != raw_plan_binding:
            raise pilot.PilotContractError("authority plan binding changed")
        plan = pilot.validate_plan(raw_plan)
        if plan["purpose"] != PURPOSE:
            raise pilot.PilotContractError(
                "calibration supervisor requires sizing_calibration_only"
            )
        authority = collector._validate_authority_for_plan(  # noqa: SLF001
            raw_authority,
            plan=plan,
            plan_binding=plan_binding,
        )
        if (
            authority["schema"] != AUTHORITY_SCHEMA
            or authority["status"] != AUTHORITY_STATUS
        ):
            raise pilot.PilotContractError("calibration authority identity changed")
        review_binding = authority["review_binding"]
        raw_review, actual_review_binding = pilot.read_bound_json(
            Path(str(review_binding["path"])),
            expected_sha256=str(review_binding["file_sha256"]),
            expected_byte_count=int(review_binding["byte_count"]),
            label="independent calibration source review",
        )
        if actual_review_binding != review_binding:
            raise pilot.PilotContractError("source review binding changed")
        pilot.validate_source_review(raw_review, authority=authority)
        collector._validate_git_authority_boundary(  # noqa: SLF001
            plan_binding=plan_binding,
            authority_binding=authority_binding,
            authority=authority,
        )
        pilot.require_plan_bindings(plan)
        expected_supervisor = _source_binding(
            authority, name="external_supervisor"
        )
        actual_supervisor = pilot.file_binding(Path(__file__))
        if expected_supervisor != actual_supervisor:
            raise pilot.PilotContractError(
                "authority binds a different calibration supervisor"
            )
    except (OSError, pilot.PilotContractError) as exc:
        raise CalibrationSupervisionError(str(exc)) from exc
    return authority, authority_binding, plan, plan_binding


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_once(
    argv: Sequence[str],
    *,
    timeout: float,
    env: Mapping[str, str],
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> dict[str, Any]:
    if timeout <= 0.0:
        raise CalibrationSupervisionError("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        start_new_session=True,
        env=dict(env),
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise CalibrationSupervisionError(
            "supervised command exceeded wall ceiling"
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if returncode not in accepted_returncodes:
        raise CalibrationSupervisionError(
            f"supervised command exited with status {returncode}"
        )
    return {
        "argv": list(argv),
        "elapsed_seconds": elapsed,
        "exit_code": returncode,
    }


def _remaining_wall(*, started: float, ceiling: float) -> float:
    return ceiling - (time.monotonic() - started)


def _owned_reservation(
    attempt_root: Path,
    *,
    supervisor_nonce: str,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
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
        and value.get("schema")
        == "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
        and value.get("status") == "RESERVED_ATTEMPT_CONSUMED"
        and value.get("attempt") == authority["attempt"]
        and value.get("plan_binding") == plan_binding
        and value.get("authority_binding") == authority_binding
        and value.get("supervisor_nonce") == supervisor_nonce
        and value.get("retry_authorized") is False
        and value.get("resume_authorized") is False
        and value.get("overwrite_authorized") is False
        and value.get("refill_authorized") is False
    )


def supervise(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authority, authority_binding, plan, plan_binding = load_and_validate_authority(
        authority_path,
        expected_authority_byte_count=expected_authority_byte_count,
        expected_authority_sha256=expected_authority_sha256,
    )
    invocation = str(_validate_python_invocation(plan))
    child_env = _child_environment(plan)
    attempt_root = _require_fresh_attempt_root(plan["output_root"])
    wall_ceiling = float(authority["caps"]["wall_seconds"])
    wall_started = time.monotonic()
    graphics_preflight = _run_graphics_preflight(
        plan,
        child_env=child_env,
        wall_started=wall_started,
        wall_ceiling=wall_ceiling,
    )
    used_path, total_path, vendor_id, device_id = _selected_gpu_memory_files(plan)
    gpu_sampler = _GlobalVramSampler(
        used_path,
        total_path,
        vendor_id=vendor_id,
        device_id=device_id,
    )
    gpu_sampler.start()
    supervisor_nonce = secrets.token_hex(32)
    phases: list[dict[str, Any]] = [graphics_preflight]
    physics_binding: dict[str, Any] | None = None
    check_binding: dict[str, Any] | None = None
    calibration_binding: dict[str, Any] | None = None
    decision: str | None = None
    failure: str | None = None
    gpu_memory_measurement: dict[str, Any] | None = None

    collector_argv = [
        invocation,
        str((REPO_ROOT / COLLECTOR_RELATIVE).resolve()),
        "--plan",
        str(Path(plan_binding["path"]).resolve()),
        "--expected-plan-byte-count",
        str(plan_binding["byte_count"]),
        "--expected-plan-sha256",
        str(plan_binding["file_sha256"]),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--expected-authority-sha256",
        str(authority_binding["file_sha256"]),
        "--supervisor-nonce",
        supervisor_nonce,
    ]
    try:
        phases.append(
            _run_once(
                collector_argv,
                timeout=_remaining_wall(started=wall_started, ceiling=wall_ceiling),
                env=child_env,
            )
        )
        physics_path = attempt_root / "physics_result.json"
        physics_binding = pilot.file_binding(physics_path)
        check_path = attempt_root / "receipt_check.json"
        checker_argv = [
            invocation,
            str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
            "--manifest",
            str(physics_path),
            "--expected-file-sha256",
            str(physics_binding["file_sha256"]),
            "--expected-byte-count",
            str(physics_binding["byte_count"]),
            "--output",
            str(check_path),
        ]
        phases.append(
            _run_once(
                checker_argv,
                timeout=_remaining_wall(started=wall_started, ceiling=wall_ceiling),
                env=child_env,
            )
        )
        check_binding = pilot.file_binding(check_path)
        check_report, actual_check_binding = pilot.read_bound_json(
            check_path,
            expected_sha256=str(check_binding["file_sha256"]),
            expected_byte_count=int(check_binding["byte_count"]),
            label="calibration receipt check",
        )
        if (
            actual_check_binding != check_binding
            or not isinstance(check_report, Mapping)
            or check_report.get("schema") != checker.REPORT_SCHEMA
            or check_report.get("status") != "PASS"
            or check_report.get("phase") != "physics_collection"
            or check_report.get("purpose") != PURPOSE
            or check_report.get("manifest_binding") != physics_binding
        ):
            raise CalibrationSupervisionError(
                "receipt-only checker did not exactly pass calibration"
            )

        calibration_path = attempt_root / "calibration_receipt.json"
        analyzer_argv = [
            invocation,
            str((REPO_ROOT / ANALYZER_RELATIVE).resolve()),
            "--collection",
            str(physics_path),
            "--expected-collection-sha256",
            str(physics_binding["file_sha256"]),
            "--expected-collection-byte-count",
            str(physics_binding["byte_count"]),
            "--output",
            str(calibration_path),
        ]
        phases.append(
            _run_once(
                analyzer_argv,
                timeout=_remaining_wall(started=wall_started, ceiling=wall_ceiling),
                env=child_env,
                accepted_returncodes=frozenset({0, 2}),
            )
        )
        calibration_binding = pilot.file_binding(calibration_path)
        calibration_receipt, actual_calibration_binding, _raw = (
            analyzer.load_bound_calibration_receipt_v1(
                calibration_path,
                expected_sha256=str(calibration_binding["file_sha256"]),
                expected_byte_count=int(calibration_binding["byte_count"]),
            )
        )
        if actual_calibration_binding != calibration_binding:
            raise CalibrationSupervisionError("calibration receipt binding changed")
        decision = str(calibration_receipt["decision"])
        expected_analyzer_returncode = 0 if decision == "FREEZE_PILOT_CONTRACT" else 2
        if phases[-1]["exit_code"] != expected_analyzer_returncode:
            raise CalibrationSupervisionError(
                "analyzer exit code and calibration decision disagree"
            )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            gpu_memory_measurement = gpu_sampler.stop()
            if gpu_memory_measurement["read_errors"] != 0:
                sampler_failure = (
                    "CalibrationSupervisionError: GPU memory sampler had read errors"
                )
                failure = (
                    sampler_failure
                    if failure is None
                    else f"{failure}; {sampler_failure}"
                )
        except BaseException as exc:
            sampler_failure = f"{type(exc).__name__}: {exc}"
            failure = (
                sampler_failure
                if failure is None
                else f"{failure}; {sampler_failure}"
            )

    wall_elapsed = time.monotonic() - wall_started
    if failure is None and wall_elapsed > wall_ceiling:
        failure = (
            "CalibrationSupervisionError: terminal validation exceeded the hard "
            f"wall ceiling ({wall_elapsed:.6f} > {wall_ceiling:.6f} seconds)"
        )
    reservation_owned = _owned_reservation(
        attempt_root,
        supervisor_nonce=supervisor_nonce,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
    )
    if attempt_root.exists() and not reservation_owned:
        ownership_failure = "CalibrationSupervisionError: reservation ownership changed"
        failure = ownership_failure if failure is None else f"{failure}; {ownership_failure}"
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": (
            "COMPLETE_PENDING_TERMINAL_REVIEW"
            if failure is None
            else "CONSUMED_TERMINAL_FAILURE"
        ),
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "scientific_verdict_emitted": False,
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
        "source_commit": authority["source_commit"],
        "attempt_root": str(attempt_root),
        "wall_elapsed_seconds": wall_elapsed,
        "wall_ceiling_seconds": wall_ceiling,
        "phase_receipts": phases,
        "physics_result_binding": physics_binding,
        "receipt_check_binding": check_binding,
        "calibration_receipt_binding": calibration_binding,
        "calibration_decision": decision,
        "gpu_memory_measurement": gpu_memory_measurement,
        "failure": failure,
        "terminal_reviewer": authority["external_supervisor"]["terminal_reviewer"],
        "supervisor_nonce": supervisor_nonce,
    }
    terminal_binding = None
    if reservation_owned:
        terminal_binding = pilot.write_json_exclusive(
            attempt_root / "terminal_supervision.json", terminal
        )
    return terminal, terminal_binding


def _raise_on_signal(signum: int, _frame: Any) -> None:
    raise CalibrationSupervisionError(f"supervisor received signal {signum}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument(
        "--expected-authority-byte-count", required=True, type=int
    )
    parser.add_argument("--expected-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_authority_byte_count <= 0:
        raise SystemExit("authority byte count must be positive")
    if _SHA256.fullmatch(args.expected_authority_sha256) is None:
        raise SystemExit("authority SHA-256 must be lowercase hexadecimal")
    signal.signal(signal.SIGINT, _raise_on_signal)
    signal.signal(signal.SIGTERM, _raise_on_signal)
    terminal, terminal_binding = supervise(
        args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
    )
    if terminal_binding is None:
        print(
            "pre-reservation supervision failure; no attempt was consumed",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "calibration_decision": terminal["calibration_decision"],
                "terminal_supervision": terminal_binding,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == "COMPLETE_PENDING_TERMINAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationSupervisionError",
    "_child_environment",
    "_require_fresh_attempt_root",
    "_run_once",
    "load_and_validate_authority",
    "supervise",
]
