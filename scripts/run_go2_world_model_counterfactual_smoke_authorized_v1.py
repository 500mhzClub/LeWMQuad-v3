#!/usr/bin/env python3
"""Supervise one exactly authorized counterfactual source-integration smoke.

This script does not create authority.  It accepts one caller-bound, committed
authority record, verifies the authority/plan/source boundary without opening
any simulator input, and starts the bound collector exactly once.  The
collector reserves the fresh attempt root before it opens runtime inputs.  A
hard external wall timer covers the collector and the receipt-only checker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402


DEVELOPMENT_ROOT = REPO_ROOT / ".generated" / "dev"
AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_smoke_execution_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_SOURCE_INTEGRATION_SMOKE"
PLAN_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_plan_v1"
PHYSICS_RESULT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_pilot_physics_result_v1"
)
CHECK_REPORT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_pilot_receipt_check_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_world_model_counterfactual_smoke_supervision_terminal_v1"
)
COLLECTOR_RELATIVE = Path(
    "scripts/collect_go2_world_model_counterfactual_pilot_v1.py"
)
CHECKER_RELATIVE = Path(
    "scripts/check_go2_world_model_counterfactual_pilot_v1.py"
)

_SANITIZED_SELECTOR_KEYS = {
    "AMD_VULKAN_ICD",
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
    "DRI_PRIME",
    "EGL_DEVICE_ID",
    "GS_BACKEND",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "MESA_VK_DEVICE_SELECT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYOPENGL_PLATFORM",
    "ROCR_VISIBLE_DEVICES",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "WAYLAND_DISPLAY",
}


class SmokeSupervisionError(RuntimeError):
    """Raised when the one-shot authority or supervised run fails closed."""


def _reject_protected_path(path: Path, *, label: str) -> None:
    lowered_parts = [part.lower() for part in Path(path).parts]
    if any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out"}
        or part.startswith("heldout_")
        or part.startswith("held_out_")
        or part.startswith("held-out-")
        for part in lowered_parts
    ):
        raise SmokeSupervisionError(f"{label} path is custody-protected")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeSupervisionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-finite values."""

    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SmokeSupervisionError(
                    f"non-finite JSON value in {label}: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeSupervisionError(f"invalid JSON in {label}") from exc


def file_binding(path: Path) -> dict[str, Any]:
    """Hash one regular, non-symlink file with a stable inode/size check."""

    selected = Path(path)
    _reject_protected_path(selected, label="bound file")
    if selected.is_symlink() or not selected.is_file():
        raise SmokeSupervisionError(
            f"bound file is absent, non-regular, or a symlink: {selected}"
        )
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise SmokeSupervisionError(f"bound file changed while read: {selected}")
    return {
        "path": str(selected.resolve()),
        "file_sha256": digest.hexdigest(),
        "byte_count": int(after.st_size),
    }


def _binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise SmokeSupervisionError(f"{label} binding shape changed")
    path = value["path"]
    digest = value["file_sha256"]
    byte_count = value["byte_count"]
    if not isinstance(path, str) or not path:
        raise SmokeSupervisionError(f"{label} path is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise SmokeSupervisionError(f"{label} SHA-256 is invalid")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise SmokeSupervisionError(f"{label} byte count is invalid")
    return {
        "path": path,
        "file_sha256": digest,
        "byte_count": byte_count,
    }


def _resolve_bound_path(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else REPO_ROOT / selected


def verify_binding(value: Any, *, label: str) -> dict[str, Any]:
    expected = _binding_shape(value, label=label)
    actual = file_binding(_resolve_bound_path(expected["path"]))
    if actual["byte_count"] != expected["byte_count"]:
        raise SmokeSupervisionError(f"{label} byte count changed")
    if actual["file_sha256"] != expected["file_sha256"]:
        raise SmokeSupervisionError(f"{label} SHA-256 changed")
    return actual


def _read_bound_json(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _binding_shape(value, label=label)
    path = _resolve_bound_path(binding["path"])
    actual = verify_binding(binding, label=label)
    document = strict_json_bytes(path.read_bytes(), label=label)
    if not isinstance(document, dict):
        raise SmokeSupervisionError(f"{label} must be a JSON object")
    return document, actual


def _git_output(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _git_head() -> str:
    value = _git_output("rev-parse", "HEAD")
    assert isinstance(value, str)
    return value


def _require_commit_ancestor(commit: str, *, label: str) -> None:
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise SmokeSupervisionError(f"{label} commit is invalid")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise SmokeSupervisionError(f"{label} commit is not an ancestor of HEAD")


def _require_binding_at_commit(
    binding: Mapping[str, Any], *, commit: str, label: str
) -> None:
    path = _resolve_bound_path(str(binding["path"])).resolve(strict=True)
    try:
        relative = path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SmokeSupervisionError(f"{label} must be inside the repository") from exc
    try:
        payload = _git_output("show", f"{commit}:{relative.as_posix()}", binary=True)
    except subprocess.CalledProcessError as exc:
        raise SmokeSupervisionError(
            f"{label} is absent from commit {commit}"
        ) from exc
    assert isinstance(payload, bytes)
    if len(payload) != int(binding["byte_count"]):
        raise SmokeSupervisionError(f"committed {label} byte count changed")
    if hashlib.sha256(payload).hexdigest() != str(binding["file_sha256"]):
        raise SmokeSupervisionError(f"committed {label} SHA-256 changed")


def _require_fresh_development_root(path_text: str) -> Path:
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise SmokeSupervisionError("attempt root must be absolute")
    development_root = DEVELOPMENT_ROOT.resolve(strict=True)
    candidate = Path(path_text)
    normalized = candidate.resolve(strict=False)
    try:
        relative = normalized.relative_to(development_root)
    except ValueError as exc:
        raise SmokeSupervisionError("attempt root escapes .generated/dev") from exc
    if not relative.parts:
        raise SmokeSupervisionError("attempt root cannot equal .generated/dev")
    cursor = development_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise SmokeSupervisionError(f"attempt path contains a symlink: {cursor}")
        if not cursor.exists():
            break
    if candidate.exists() or candidate.is_symlink():
        raise SmokeSupervisionError(f"attempt root is not fresh: {candidate}")
    return candidate


def _child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    """Return a deterministic selector environment without ambient overrides."""

    expected = plan["execution_contract"]["environment"]
    if expected != pilot.EXECUTION_ENVIRONMENT:
        raise SmokeSupervisionError("execution environment contract changed")
    child = dict(os.environ)
    for key in _SANITIZED_SELECTOR_KEYS:
        child.pop(key, None)
    child.update({str(key): str(value) for key, value in expected.items()})
    return child


def _validate_python_invocation_before_launch(plan: Mapping[str, Any]) -> Path:
    invocation = Path(str(plan["execution_contract"]["python_invocation_path"]))
    target_binding = plan["runtime_bindings"]["python_executable_target"]
    environment_binding = plan["runtime_bindings"]["python_environment_config"]
    try:
        pilot.require_binding(target_binding, label="Python executable target")
        pilot.require_binding(environment_binding, label="Python environment config")
    except pilot.PilotContractError as exc:
        raise SmokeSupervisionError(str(exc)) from exc
    target = Path(str(target_binding["path"]))
    environment_config = Path(str(environment_binding["path"]))
    if (
        not invocation.is_absolute()
        or invocation.resolve(strict=True) != target.resolve(strict=True)
        or environment_config.name != "pyvenv.cfg"
        or invocation.parent.parent != environment_config.parent
    ):
        raise SmokeSupervisionError(
            "Python invocation does not resolve inside the bound environment"
        )
    return invocation


def _run_graphics_preflight(
    plan: Mapping[str, Any], *, child_env: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the exact EGL/Vulkan selectors before consuming the attempt."""

    runtime = plan["runtime_bindings"]
    expectation = plan["execution_contract"]["graphics_preflight"]
    if expectation != pilot.GRAPHICS_PREFLIGHT_EXPECTATION:
        raise SmokeSupervisionError("graphics preflight expectation changed")
    try:
        eglinfo = pilot.require_binding(
            runtime["eglinfo_executable"], label="eglinfo executable"
        )
        vulkaninfo = pilot.require_binding(
            runtime["vulkaninfo_executable"], label="vulkaninfo executable"
        )
    except pilot.PilotContractError as exc:
        raise SmokeSupervisionError(str(exc)) from exc

    vulkan = subprocess.run(
        [str(vulkaninfo["path"]), "--summary"],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    if vulkan.returncode != 0:
        raise SmokeSupervisionError(
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
        raise SmokeSupervisionError("selected Vulkan GPU is not the exact R9700")
    if re.search(r"^GPU1:", vulkan.stdout, re.MULTILINE) is not None:
        raise SmokeSupervisionError("Vulkan selector did not reduce enumeration to GPU0")

    egl = subprocess.run(
        [str(eglinfo["path"]), "-B"],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    if egl.returncode != expectation["eglinfo_expected_exit_code"]:
        raise SmokeSupervisionError(
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
        raise SmokeSupervisionError("selected EGL device is not the exact R9700")
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


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the committed authority and plan before the collector runs."""

    actual_authority = file_binding(authority_path)
    if actual_authority["byte_count"] != expected_byte_count:
        raise SmokeSupervisionError("authority byte count disagrees with caller")
    if actual_authority["file_sha256"] != expected_sha256:
        raise SmokeSupervisionError("authority SHA-256 disagrees with caller")
    authority = strict_json_bytes(authority_path.read_bytes(), label="authority")
    if not isinstance(authority, dict):
        raise SmokeSupervisionError("authority must be a JSON object")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authority_granted_by_this_document") is not True
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("network_access") is not False
    ):
        raise SmokeSupervisionError("authority semantic grant is invalid")
    source_commit = authority.get("source_commit")
    _require_commit_ancestor(source_commit, label="authorized source")
    _require_binding_at_commit(
        actual_authority, commit="HEAD", label="execution authority"
    )

    plan, actual_plan = _read_bound_json(authority.get("plan_binding"), label="plan")
    _require_binding_at_commit(actual_plan, commit="HEAD", label="plan")
    try:
        plan = pilot.validate_plan(plan)
        authority = pilot.validate_authority(
            authority,
            plan=plan,
            plan_binding=actual_plan,
        )
    except pilot.PilotContractError as exc:
        raise SmokeSupervisionError(
            f"full authority/plan contract validation failed: {exc}"
        ) from exc
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("purpose") != "source_integration_smoke"
        or plan.get("citable_as_scientific_evidence") is not False
        or plan.get("authorizes_retry_or_resume") is not False
    ):
        raise SmokeSupervisionError("bound plan is not the source-integration smoke")
    attempt = authority.get("attempt")
    if not isinstance(attempt, Mapping):
        raise SmokeSupervisionError("authority attempt contract is absent")
    expected_attempt = {
        "id": plan.get("attempt_id"),
        "root": plan.get("output_root"),
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    if dict(attempt) != expected_attempt:
        raise SmokeSupervisionError("authority attempt boundary changed")
    caps = authority.get("caps")
    wall_seconds = caps.get("wall_seconds") if isinstance(caps, Mapping) else None
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
    ):
        raise SmokeSupervisionError("authority wall cap is invalid")
    execution = plan.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise SmokeSupervisionError("plan execution contract is absent")
    invocation = execution.get("python_invocation_path")
    if not isinstance(invocation, str) or not Path(invocation).is_absolute():
        raise SmokeSupervisionError("bound Python invocation path is invalid")

    source_bindings = authority.get("source_bindings")
    if not isinstance(source_bindings, list):
        raise SmokeSupervisionError("authority source closure is absent")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in source_bindings:
        if not isinstance(row, Mapping) or set(row) != {"name", "binding"}:
            raise SmokeSupervisionError("authority source binding shape changed")
        name = row["name"]
        if not isinstance(name, str) or name in by_name:
            raise SmokeSupervisionError("authority source names are invalid")
        binding = _binding_shape(row["binding"], label=f"source {name}")
        by_name[name] = binding
        verify_binding(binding, label=f"source {name}")
        _require_binding_at_commit(
            binding, commit=str(source_commit), label=f"source {name}"
        )
    for required_name in ("collector", "checker", "pilot_consumer", "external_supervisor"):
        if required_name not in by_name:
            raise SmokeSupervisionError(
                f"authority source closure omits {required_name}"
            )
    if _resolve_bound_path(str(by_name["collector"]["path"])).resolve() != (
        REPO_ROOT / COLLECTOR_RELATIVE
    ).resolve():
        raise SmokeSupervisionError("authority binds a different collector")
    if _resolve_bound_path(str(by_name["checker"]["path"])).resolve() != (
        REPO_ROOT / CHECKER_RELATIVE
    ).resolve():
        raise SmokeSupervisionError("authority binds a different checker")

    external = authority.get("external_supervisor")
    if (
        not isinstance(external, Mapping)
        or set(external) != {"source_binding", "terminal_reviewer"}
        or not isinstance(external.get("terminal_reviewer"), str)
        or not external["terminal_reviewer"].strip()
    ):
        raise SmokeSupervisionError("external supervisor contract is invalid")
    external_binding = _binding_shape(
        external["source_binding"], label="external supervisor"
    )
    if external_binding != by_name["external_supervisor"]:
        raise SmokeSupervisionError(
            "external supervisor binding differs from reviewed source closure"
        )
    actual_supervisor = verify_binding(
        external_binding, label="external supervisor"
    )
    if Path(actual_supervisor["path"]) != Path(__file__).resolve():
        raise SmokeSupervisionError("authority binds a different supervisor source")

    review, actual_review = _read_bound_json(
        authority.get("review_binding"), label="independent source review"
    )
    _require_binding_at_commit(
        actual_review, commit="HEAD", label="independent source review"
    )
    if (
        review.get("schema")
        != "lewm_go2_world_model_follow_on_independent_source_review_v1"
        or review.get("status") != "PASS_SOURCE_ONLY_NOT_AUTHORITY"
        or review.get("authority_granted_by_this_document") is not False
        or review.get("reviewed_source_commit") != source_commit
        or review.get("reviewed_source_bindings") != source_bindings
        or review.get("remaining_findings") != []
    ):
        raise SmokeSupervisionError("independent source review is not an exact pass")
    try:
        pilot.validate_source_review(review, authority=authority)
    except pilot.PilotContractError as exc:
        raise SmokeSupervisionError(
            f"full independent source review validation failed: {exc}"
        ) from exc
    return authority, actual_authority, plan, actual_plan


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


def _raise_on_termination_signal(signum: int, _frame: Any) -> None:
    """Turn supervisor termination into an exception so children are reaped."""

    raise SmokeSupervisionError(f"supervisor received signal {signum}")


def _run_once(
    argv: Sequence[str], *, timeout: float, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    if timeout <= 0.0:
        raise SmokeSupervisionError("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        start_new_session=True,
        env=None if env is None else dict(env),
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise SmokeSupervisionError("supervised command exceeded wall ceiling") from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if returncode != 0:
        _terminate_process_group(process)
        raise SmokeSupervisionError(
            f"supervised command exited with status {returncode}"
        )
    return {"argv": list(argv), "elapsed_seconds": elapsed, "exit_code": 0}


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(
        dict(payload), allow_nan=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(path)


def _write_terminal(
    attempt_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        return None
    return _write_json_exclusive(
        attempt_root / "terminal_supervision.json", payload
    )


def _owned_reservation(
    attempt_root: Path, *, supervisor_nonce: str
) -> dict[str, Any] | None:
    path = attempt_root / "reservation.json"
    if not path.is_file() or path.is_symlink():
        return None
    binding = file_binding(path)
    value = strict_json_bytes(path.read_bytes(), label="attempt reservation")
    expected_keys = {
        "schema",
        "status",
        "attempt",
        "plan_binding",
        "authority_binding",
        "supervisor_nonce",
        "retry_authorized",
        "resume_authorized",
        "overwrite_authorized",
        "refill_authorized",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("supervisor_nonce") != supervisor_nonce
    ):
        return None
    return {"document": value, "binding": binding}


def supervise(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authority, authority_binding, plan, plan_binding = load_and_validate_authority(
        authority_path,
        expected_byte_count=expected_authority_byte_count,
        expected_sha256=expected_authority_sha256,
    )
    invocation_path = _validate_python_invocation_before_launch(plan)
    child_env = _child_environment(plan)
    graphics_preflight = _run_graphics_preflight(plan, child_env=child_env)
    attempt_root = _require_fresh_development_root(str(plan["output_root"]))
    wall_cap = float(authority["caps"]["wall_seconds"])
    wall_started = time.monotonic()
    phases: list[dict[str, Any]] = [graphics_preflight]
    physics_binding: dict[str, Any] | None = None
    physics_result: Any = None
    check_binding: dict[str, Any] | None = None
    failure: str | None = None
    supervisor_nonce = secrets.token_hex(32)
    invocation = str(invocation_path)
    collector_argv = [
        invocation,
        str((REPO_ROOT / COLLECTOR_RELATIVE).resolve()),
        "--plan",
        str(_resolve_bound_path(str(authority["plan_binding"]["path"])).resolve()),
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
        phases.append(_run_once(collector_argv, timeout=wall_cap, env=child_env))
        physics_path = attempt_root / "physics_result.json"
        physics_binding = file_binding(physics_path)
        physics_result = strict_json_bytes(
            physics_path.read_bytes(), label="physics result"
        )
        if (
            not isinstance(physics_result, dict)
            or physics_result.get("schema") != PHYSICS_RESULT_SCHEMA
            or physics_result.get("status") != "PHYSICS_COMPLETE"
            or physics_result.get("physics_validated") is not False
            or physics_result.get("citable_as_scientific_evidence") is not False
            or physics_result.get("attempt_id") != plan.get("attempt_id")
            or physics_result.get("plan_binding") != authority.get("plan_binding")
            or physics_result.get("authority_binding") != authority_binding
            or physics_result.get("review_binding") != authority.get("review_binding")
            or physics_result.get("source_bindings") != authority.get("source_bindings")
            or physics_result.get("caps") != authority.get("caps")
            or physics_result.get("runtime_bindings")
            != authority.get("runtime_bindings")
            or physics_result.get("execution_contract") != authority.get("execution")
        ):
            raise SmokeSupervisionError("collector physics result is not an exact success")
        remaining = wall_cap - (time.monotonic() - wall_started)
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
        phases.append(_run_once(checker_argv, timeout=remaining, env=child_env))
        check_binding = file_binding(check_path)
        check_report = strict_json_bytes(check_path.read_bytes(), label="receipt check")
        if (
            not isinstance(check_report, dict)
            or check_report.get("schema") != CHECK_REPORT_SCHEMA
            or check_report.get("status") != "PASS"
            or check_report.get("phase") != "physics_collection"
            or check_report.get("purpose") != "source_integration_smoke"
            or check_report.get("can_freeze_pilot_contract") is not False
            or check_report.get("runtime_payloads_opened") is not False
            or check_report.get("rgb_bytes_opened") is not False
            or check_report.get("checkpoints_opened") is not False
            or check_report.get("manifest_binding") != physics_binding
        ):
            raise SmokeSupervisionError("receipt-only checker did not exactly pass")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"

    owned_reservation = _owned_reservation(
        attempt_root, supervisor_nonce=supervisor_nonce
    )
    if owned_reservation is not None:
        reservation_document = owned_reservation["document"]
        owned_reservation_binding = dict(owned_reservation["binding"])
        owned_reservation_binding["path"] = "reservation.json"
        if (
            reservation_document.get("schema")
            != "lewm_go2_world_model_counterfactual_smoke_reservation_v1"
            or reservation_document.get("status") != "RESERVED_ATTEMPT_CONSUMED"
            or reservation_document.get("attempt") != authority.get("attempt")
            or reservation_document.get("plan_binding") != plan_binding
            or reservation_document.get("authority_binding") != authority_binding
            or reservation_document.get("retry_authorized") is not False
            or reservation_document.get("resume_authorized") is not False
            or reservation_document.get("overwrite_authorized") is not False
            or reservation_document.get("refill_authorized") is not False
            or not isinstance(physics_result, dict)
            or physics_result.get("reservation_binding")
            != owned_reservation_binding
        ):
            failure = "SmokeSupervisionError: owned reservation contract changed"

    blocked_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked_signals)
    try:
        wall_elapsed = time.monotonic() - wall_started
        if failure is None and wall_elapsed > wall_cap:
            failure = (
                "SmokeSupervisionError: terminal validation exceeded the hard wall "
                f"ceiling ({wall_elapsed:.6f} > {wall_cap:.6f} seconds)"
            )
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": (
                "COMPLETE_PENDING_TERMINAL_REVIEW"
                if failure is None
                else "CONSUMED_TERMINAL_FAILURE"
            ),
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
            "authority_binding": authority_binding,
            "plan_binding": plan_binding,
            "source_commit": authority["source_commit"],
            "execution_head": _git_head(),
            "attempt_root": str(attempt_root),
            "wall_elapsed_seconds": wall_elapsed,
            "wall_ceiling_seconds": wall_cap,
            "phase_receipts": phases,
            "physics_result_binding": physics_binding,
            "receipt_check_binding": check_binding,
            "failure": failure,
            "terminal_reviewer": authority["external_supervisor"][
                "terminal_reviewer"
            ],
            "scientific_verdict_emitted": False,
            "pilot_contract_frozen": False,
            "supervisor_nonce": supervisor_nonce,
        }
        terminal_binding = (
            _write_terminal(attempt_root, terminal)
            if owned_reservation is not None
            else None
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return terminal, terminal_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument(
        "--expected-authority-byte-count", type=int, required=True
    )
    parser.add_argument("--expected-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_authority_byte_count <= 0:
        parser.error("authority byte count must be positive")
    if (
        len(args.expected_authority_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in args.expected_authority_sha256
        )
    ):
        parser.error("authority SHA-256 must be lowercase hexadecimal")
    signal.signal(signal.SIGINT, _raise_on_termination_signal)
    signal.signal(signal.SIGTERM, _raise_on_termination_signal)
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
                "terminal_supervision": terminal_binding,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == "COMPLETE_PENDING_TERMINAL_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
