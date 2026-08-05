#!/usr/bin/env python3
"""Qualify Genesis 0.4.6 ``gs.amdgpu`` on two exact full scenes.

The qualification is deliberately non-scientific.  It runs frozen scene 12
and then scene 0 in fresh process groups, exercises the complete physical and
textured-v03 render route, and permanently forbids probe reuse.  This source
does not issue its own authority and must not be run without a separately
reviewed one-shot qualification authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as collector  # noqa: E402


pilot = collector.pilot
predecessor = collector.predecessor
kernel = collector.kernel

QUALIFICATION_AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "qualification_authority_v1"
)
QUALIFICATION_AUTHORITY_STATUS = (
    "AUTHORIZED_GENESIS_ROCM_BACKEND_V1_QUALIFICATION_ONLY"
)
QUALIFICATION_AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS)
    | {"qualification_contract", "predecessor_cpu_terminal_review_binding"}
)
QUALIFICATION_RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "qualification_reservation_v1"
)
QUALIFICATION_SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "qualification_scene_result_v1"
)
QUALIFICATION_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "qualification_result_v1"
)
QUALIFICATION_TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_"
    "qualification_terminal_v1"
)
QUALIFICATION_RESULT_STATUS = "PASS_GENESIS_ROCM_BACKEND_V1_QUALIFICATION"
QUALIFICATION_PROBE_ORDER = tuple(plan_builder.QUALIFICATION_SCENE_INDICES)
WORKER_TIMEOUT_SECONDS = float(
    plan_builder.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS = float(
    plan_builder.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)

QUALIFICATION_CONTRACT = {
    "backend": "amdgpu",
    "backend_api": "gs.amdgpu",
    "genesis_version": "0.4.6",
    "hip_device_name": "AMD Radeon AI PRO R9700",
    "hip_arch_name": "gfx1201",
    "probe_scene_indices_in_order": list(QUALIFICATION_PROBE_ORDER),
    "fresh_worker_processes": 2,
    "states_per_worker": 4,
    "candidate_actions_per_state": 9,
    "branches_per_worker": 36,
    "context_frames_per_worker": 12,
    "target_frames_per_worker": 36,
    "stored_rgb_frames_per_worker": 48,
    "auxiliary_depth_validation_renders_per_worker": 48,
    "full_history_and_successor_horizon": True,
    "worker_process_group_watchdog_seconds": WORKER_TIMEOUT_SECONDS,
    "existing_integrity_render_count_and_nonfinite_gates_required": True,
    "selected_device_vram_cap_unchanged": True,
    "new_amdgpu_ring_timeout_or_reset_forbidden": True,
    "hsa_override_gfx_version_forbidden": True,
    "contact_force_reads_forbidden": True,
    "role_local_cache_required": True,
    "timing_gate_formula": "64 * max(worker_elapsed_seconds) + 900 <= 7200",
    "probe_output_scientific_reuse_authorized": False,
    "qualification_failure_authorizes_retry": False,
}

QUALIFICATION_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "attempt_id",
        "backend",
        "backend_api",
        "qualification_contract",
        "authority_binding",
        "plan_binding",
        "rocm_egl_preflight",
        "probe_order",
        "probes",
        "kernel_reset_audit",
        "timing_gate",
        "contact_force_route_audit",
        "all_existing_scene_gates_passed",
        "exact_v03_renderer_compatibility_passed",
        "scientific_attempt_root_absent",
        "probe_output_scientific_reuse_authorized",
        "authorizes_scientific_authority_consideration",
        "authorizes_retry_or_resume",
    }
)

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_preregistration_2026-08-04.md"
)
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_source_review_2026-08-04.json"
)
QUALIFICATION_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_qualification_authority_2026-08-04.json"
)
QUALIFICATION_RESULT_PATH = (
    plan_builder.QUALIFICATION_ATTEMPT_ROOT / "qualification_result.json"
)

_ROCM_WORKER_ARGV = collector._worker_argv_rocm  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


class GenesisRocmBackendQualificationError(RuntimeError):
    """Raised when the bounded ROCm qualification fails closed."""


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def _all_numbers_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _source_paths() -> Mapping[str, Path]:
    """Return the runner-owned complete successor source closure lazily."""

    from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as runner

    return runner.SOURCE_PATHS


def _worker_argv_qualification(**kwargs: Any) -> list[str]:
    argv = _ROCM_WORKER_ARGV(**kwargs)
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(collector.__file__).resolve():
        raise GenesisRocmBackendQualificationError(
            "ROCm worker entry point changed"
        )
    argv[1] = str(Path(__file__).resolve())
    return argv


def _read_qualification_reservation(
    *,
    collection_root: Path,
    expected_sha256: str,
    expected_byte_count: int,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    orchestrator_nonce: str,
    orchestrator_pid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, binding = pilot.read_bound_json(
        collection_root / "reservation.json",
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="ROCm qualification reservation",
    )
    if (
        value.get("schema") != QUALIFICATION_RESERVATION_SCHEMA
        or value.get("status") != "RESERVED_TWO_PROBE_QUALIFICATION_CONSUMED"
        or value.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("plan_binding") != predecessor._standard_binding(plan_binding)  # noqa: SLF001
        or value.get("orchestrator_nonce") != orchestrator_nonce
        or value.get("orchestrator_pid") != orchestrator_pid
        or value.get("probe_scene_indices_in_order")
        != list(QUALIFICATION_PROBE_ORDER)
        or value.get("qualification_contract") != QUALIFICATION_CONTRACT
        or value.get("probe_output_scientific_reuse_authorized") is not False
        or value.get("retry_resume_overwrite_authorized") is not False
    ):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification reservation changed"
        )
    return value, binding


@contextmanager
def _configured_qualification_collector() -> Iterator[None]:
    overrides = {
        "AUTHORITY_FIELDS": QUALIFICATION_AUTHORITY_FIELDS,
        "AUTHORITY_SCHEMA": QUALIFICATION_AUTHORITY_SCHEMA,
        "AUTHORITY_STATUS": QUALIFICATION_AUTHORITY_STATUS,
        "ATTEMPT_ID": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "RESERVATION_SCHEMA": QUALIFICATION_RESERVATION_SCHEMA,
        "SCENE_RESULT_SCHEMA": QUALIFICATION_SCENE_RESULT_SCHEMA,
        "_read_collection_reservation_rocm": _read_qualification_reservation,
        "_worker_argv_rocm": _worker_argv_qualification,
    }
    with _CONFIGURATION_LOCK:
        originals = {name: getattr(collector, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(collector, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(collector, name, value)


def validate_qualification_authority(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a separately issued authority; this function never issues one."""

    authority_raw, historical = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="ROCm qualification authority",
    )
    authority_binding = _standard_binding(historical)
    plan_binding = authority_raw.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification plan binding is absent"
        )
    plan_raw, observed_plan_binding = pilot.read_bound_json(
        Path(str(plan_binding["path"])),
        expected_sha256=str(plan_binding["sha256"]),
        expected_byte_count=int(plan_binding["byte_count"]),
        label="ROCm qualification plan",
    )
    if _standard_binding(observed_plan_binding) != dict(plan_binding):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification plan binding changed"
        )
    plan = plan_builder.validate_rocm_plan(
        plan_raw,
        expected_attempt_id=plan_builder.QUALIFICATION_ATTEMPT_ID,
        expected_output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
        plan_role="qualification",
    )

    sources = _source_paths()
    bound_sources = authority_raw.get("source_bindings")
    if not isinstance(bound_sources, Mapping) or set(bound_sources) != set(sources):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification direct source closure changed"
        )
    if any(dict(bound_sources[name]) != _binding(path) for name, path in sources.items()):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification source binding changed"
        )
    prereg_binding = authority_raw.get("preregistration_binding")
    review_binding = authority_raw.get("source_review_binding")
    predecessor_review = authority_raw.get(
        "predecessor_cpu_terminal_review_binding"
    )
    if (
        not isinstance(prereg_binding, Mapping)
        or not isinstance(review_binding, Mapping)
        or dict(prereg_binding) != _binding(PREREGISTRATION)
        or dict(review_binding) != _binding(SOURCE_REVIEW)
        or predecessor_review
        != _standard_binding(plan_builder.CPU_TERMINAL_REVIEW_BINDING)
    ):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification reviewed evidence changed"
        )
    review = json.loads(SOURCE_REVIEW.read_bytes())
    if (
        set(authority_raw) != QUALIFICATION_AUTHORITY_FIELDS
        or authority_raw.get("schema") != QUALIFICATION_AUTHORITY_SCHEMA
        or authority_raw.get("status") != QUALIFICATION_AUTHORITY_STATUS
        or authority_raw.get("attempt_id")
        != plan_builder.QUALIFICATION_ATTEMPT_ID
        or authority_raw.get("attempt_root")
        != str(plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve())
        or authority_raw.get("collection_root")
        != str(plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve())
        or authority_raw.get("qualification_contract")
        != QUALIFICATION_CONTRACT
        or authority_raw.get("caps") != collector.EXPECTED_CAPS
        or authority_raw.get("permissions") != collector.EXPECTED_PERMISSIONS
        or review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or review.get("findings") != []
        or review.get("protected_material_opened") is not False
        or review.get("qualification_plan_binding") != dict(plan_binding)
        or review.get("source_bindings") != bound_sources
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification authority contract changed"
        )
    return dict(authority_raw), authority_binding, plan, dict(plan_binding)


def _reserve_qualification(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    nonce: str,
) -> tuple[Path, dict[str, Any]]:
    namespace = plan_builder.QUALIFICATION_ATTEMPT_ROOT.parent
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    if namespace.parent != development or namespace.is_symlink():
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification namespace changed"
        )
    namespace.mkdir(mode=0o700, exist_ok=True)
    os.mkdir(plan_builder.QUALIFICATION_ATTEMPT_ROOT, mode=0o700)
    pilot.write_json_exclusive(
        plan_builder.QUALIFICATION_ATTEMPT_ROOT / "reservation.json",
        {
            "schema": QUALIFICATION_RESERVATION_SCHEMA,
            "status": "CONSUMED_ONE_SHOT_QUALIFICATION_ATTEMPT",
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "scientific_attempt_root": str(
                plan_builder.DEFAULT_ATTEMPT_ROOT.resolve()
            ),
            "scientific_attempt_root_must_remain_absent": True,
            "retry_resume_overwrite_authorized": False,
        },
    )
    collection = plan_builder.QUALIFICATION_OUTPUT_ROOT
    os.mkdir(collection, mode=0o700)
    os.mkdir(collection / "scenes", mode=0o700)
    os.mkdir(collection / "scene_results", mode=0o700)
    reservation = {
        "schema": QUALIFICATION_RESERVATION_SCHEMA,
        "status": "RESERVED_TWO_PROBE_QUALIFICATION_CONSUMED",
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "attempt_root": str(plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve()),
        "collection_root": str(collection.resolve()),
        "plan_binding": predecessor._standard_binding(plan_binding),  # noqa: SLF001
        "authority_binding": dict(authority_binding),
        "orchestrator_nonce": nonce,
        "orchestrator_pid": os.getpid(),
        "probe_scene_indices_in_order": list(QUALIFICATION_PROBE_ORDER),
        "qualification_contract": QUALIFICATION_CONTRACT,
        "probe_output_scientific_reuse_authorized": False,
        "retry_resume_overwrite_authorized": False,
    }
    binding = pilot.write_json_exclusive(
        collection / "reservation.json", reservation
    )
    return collection.resolve(), binding


def _child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = plan["execution_contract"]["environment"]
    role_expected = plan_builder.rocm_execution_environment("qualification")
    if expected != role_expected:
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification environment contract changed"
        )
    sanitized = set(kernel._SANITIZED_SELECTOR_KEYS) | set(  # noqa: SLF001
        collector.ROCM_ADDITIONAL_SANITIZED_KEYS
    )
    child = dict(os.environ)
    for key in sanitized:
        child.pop(key, None)
    child.update({str(key): str(value) for key, value in expected.items()})
    if (
        "HSA_OVERRIDE_GFX_VERSION" in child
        or "LD_LIBRARY_PATH" in expected
        or child.get("GS_CACHE_FILE_PATH")
        != str((plan_builder.QUALIFICATION_ATTEMPT_ROOT / "quadrants_cache").resolve())
    ):
        raise GenesisRocmBackendQualificationError(
            "ROCm qualification selector/cache isolation changed"
        )
    return child


def _run_rocm_egl_preflight(
    plan: Mapping[str, Any], *, child_env: Mapping[str, str]
) -> dict[str, Any]:
    """Verify compiler, HIP identity, Genesis API, and EGL before workers."""

    runtime = plan["runtime_bindings"]
    expectation = plan["execution_contract"]["graphics_preflight"]
    if expectation != plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION:
        raise GenesisRocmBackendQualificationError(
            "ROCm preflight expectation changed"
        )
    for name in (
        "eglinfo_executable",
        "rocminfo_executable",
        "rocm_lld_executable",
        "python_executable_target",
    ):
        pilot.require_binding(runtime[name], label=f"ROCm preflight {name}")

    bound_lld = Path(str(runtime["rocm_lld_executable"]["path"])).resolve(
        strict=True
    )
    path_lld = shutil.which("ld.lld", path=str(child_env["PATH"]))
    rocm_lld = (
        Path(str(child_env["ROCM_PATH"])) / "lib/llvm/bin/ld.lld"
    )
    if (
        path_lld is None
        or Path(path_lld).resolve(strict=True) != bound_lld
        or rocm_lld.resolve(strict=True) != bound_lld
    ):
        raise GenesisRocmBackendQualificationError(
            "PATH/ROCM_PATH do not resolve the bound AMD LLD"
        )
    lld = subprocess.run(
        [str(bound_lld), "--version"],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    if lld.returncode != 0 or "AMD LLD 20." not in lld.stdout:
        raise GenesisRocmBackendQualificationError("bound AMD LLD preflight failed")

    rocminfo = subprocess.run(
        [str(runtime["rocminfo_executable"]["path"])],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60.0,
    )
    if rocminfo.returncode != 0 or "gfx1201" not in rocminfo.stdout:
        raise GenesisRocmBackendQualificationError("rocminfo gfx1201 preflight failed")

    query_code = """
import json, os
from pathlib import Path
import genesis as gs
import numpy
import PIL
import torch
p = torch.cuda.get_device_properties(0)
print(json.dumps({
    'torch_version': torch.__version__,
    'torch_hip_version': torch.version.hip,
    'visible_device_count': torch.cuda.device_count(),
    'device_name': torch.cuda.get_device_name(0),
    'arch_name': str(getattr(p, 'gcnArchName', '')),
    'genesis_version': gs.__version__,
    'genesis_backend_symbol': 'gs.amdgpu' if int(gs.amdgpu) == 3 else 'unexpected',
    'hsa_override_present': 'HSA_OVERRIDE_GFX_VERSION' in os.environ,
    'genesis_file': str(Path(gs.__file__).resolve()),
    'torch_file': str(Path(torch.__file__).resolve()),
    'numpy_file': str(Path(numpy.__file__).resolve()),
    'pillow_file': str(Path(PIL.__file__).resolve()),
}, sort_keys=True))
"""
    hip = subprocess.run(
        [str(plan["execution_contract"]["python_invocation_path"]), "-I", "-B", "-c", query_code],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    identity_lines = [line.strip() for line in hip.stdout.splitlines() if line.strip()]
    try:
        identity = json.loads(identity_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GenesisRocmBackendQualificationError(
            "HIP identity preflight emitted malformed output"
        ) from exc
    if not isinstance(identity, dict):
        raise GenesisRocmBackendQualificationError(
            "HIP identity preflight emitted a non-object"
        )
    try:
        module_paths = {
            name: Path(str(identity[name])).resolve(strict=True)
            for name in (
                "genesis_file",
                "torch_file",
                "numpy_file",
                "pillow_file",
            )
        }
        world_model_site = Path(
            str(runtime["torch_distribution_metadata"]["path"])
        ).parent.parent.resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise GenesisRocmBackendQualificationError(
            "HIP module-path identity preflight failed"
        ) from exc
    if (
        hip.returncode != 0
        or identity.get("torch_version") != "2.12.0+rocm7.2"
        or not str(identity.get("torch_hip_version", "")).startswith("7.2")
        or identity.get("visible_device_count") != expectation["hip_visible_device_count"]
        or identity.get("device_name") != expectation["hip_device_name"]
        or not str(identity.get("arch_name", "")).startswith(
            str(expectation["hip_arch_name"])
        )
        or identity.get("genesis_version") != "0.4.6"
        or identity.get("genesis_backend_symbol") != "gs.amdgpu"
        or identity.get("hsa_override_present") is not False
        or module_paths["genesis_file"]
        != Path(str(runtime["genesis_init_source"]["path"])).resolve(strict=True)
        or any(
            not module_paths[name].is_relative_to(world_model_site)
            for name in ("torch_file", "numpy_file", "pillow_file")
        )
    ):
        raise GenesisRocmBackendQualificationError(
            "HIP/Genesis R9700 identity preflight failed"
        )

    egl = subprocess.run(
        [str(runtime["eglinfo_executable"]["path"]), "-B"],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    device_index = int(expectation["egl_device_index"])
    device_sections = list(
        re.finditer(
            rf"Device #{device_index}:\s*\n(?P<section>.*?)(?=\nDevice #\d+:|\Z)",
            egl.stdout,
            re.DOTALL,
        )
    )
    if (
        egl.returncode != expectation["eglinfo_expected_exit_code"]
        or len(device_sections) != 1
        or expectation["egl_renderer_name_contains"]
        not in device_sections[0].group("section")
    ):
        raise GenesisRocmBackendQualificationError("EGL R9700 preflight failed")
    return {
        "status": "PASS_EXACT_ROCM_HIP_AND_EGL_R9700",
        # Never persist the inherited host environment: it can include
        # credentials and is not qualification evidence.
        "environment": dict(plan["execution_contract"]["environment"]),
        "expectation": copy.deepcopy(expectation),
        "identity": identity,
        "egl_device_index": device_index,
        "path_ld_lld": str(Path(path_lld).resolve(strict=True)),
        "rocm_path_ld_lld": str(rocm_lld.resolve(strict=True)),
        "lld_stdout_sha256": hashlib.sha256(lld.stdout.encode()).hexdigest(),
        "rocminfo_stdout_sha256": hashlib.sha256(
            rocminfo.stdout.encode()
        ).hexdigest(),
        "egl_stdout_sha256": hashlib.sha256(egl.stdout.encode()).hexdigest(),
        "egl_stderr_sha256": hashlib.sha256(egl.stderr.encode()).hexdigest(),
        "egl_exit_code": egl.returncode,
    }


def _run_worker_with_watchdog(
    argv: Sequence[str],
    *,
    scene: Mapping[str, Any],
    child_env: Mapping[str, str],
    used_path: Path,
    ceiling_bytes: int,
) -> dict[str, Any]:
    baseline = predecessor._read_vram_counter_v2(used_path)  # noqa: SLF001
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv), cwd=REPO_ROOT, env=dict(child_env), start_new_session=True
    )
    pgid = os.getpgid(process.pid)
    peak = baseline
    timeout = False
    cap_breach = False
    while process.poll() is None:
        elapsed = time.monotonic() - started
        used = predecessor._read_vram_counter_v2(used_path)  # noqa: SLF001
        peak = max(peak, used)
        cap_breach = cap_breach or used > ceiling_bytes
        if elapsed > WORKER_TIMEOUT_SECONDS or cap_breach:
            timeout = elapsed > WORKER_TIMEOUT_SECONDS
            os.killpg(pgid, signal.SIGKILL)
            process.wait()
            break
        time.sleep(0.02)
    returncode = int(process.wait())
    return {
        "scene_index": int(scene["scene_index"]),
        "role": str(scene["role"]),
        "scene_id": str(scene["scene_id"]),
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "process_group_id": pgid,
        "fresh_process_group": pgid == process.pid,
        "sys_executable": str(Path(sys.executable).resolve(strict=True)),
        "prelaunch_baseline_used_bytes": baseline,
        "peak_selected_device_vram_bytes": peak,
        "selected_device_vram_cap_breached": cap_breach,
        "watchdog_timeout": timeout,
        "exit_code": returncode,
        "elapsed_seconds": time.monotonic() - started,
    }


def _kernel_events_since(epoch_seconds: float) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "/usr/bin/journalctl",
            "-k",
            "--since",
            f"@{epoch_seconds:.6f}",
            "--no-pager",
            "-o",
            "short-unix",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15.0,
    )
    if completed.returncode != 0:
        raise GenesisRocmBackendQualificationError(
            "kernel reset audit was unavailable"
        )
    pattern = re.compile(
        r"amdgpu.*(?:ring .* timeout|ring .* reset|device wedged|GPU reset)",
        re.IGNORECASE,
    )
    events = [line for line in completed.stdout.splitlines() if pattern.search(line)]
    return {
        "query_succeeded": True,
        "new_amdgpu_ring_timeout_or_reset_count": len(events),
        "matching_lines_sha256": hashlib.sha256(
            "\n".join(events).encode("utf-8")
        ).hexdigest(),
    }


def execute_qualification(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute exactly two non-scientific full-scene probes."""

    with _configured_qualification_collector(), collector._configured_predecessor_collector_rocm():  # noqa: SLF001
        nonce = hashlib.sha256(os.urandom(32)).hexdigest()
        collection_root, absolute_reservation = _reserve_qualification(
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            nonce=nonce,
        )
        relative_reservation = kernel._relative_output_binding(  # noqa: SLF001
            absolute_reservation, output_root=collection_root
        )
        worker_plan_binding = pilot.file_binding(
            Path(str(plan_binding["path"]))
        )
        if _standard_binding(worker_plan_binding) != dict(plan_binding):
            raise GenesisRocmBackendQualificationError(
                "ROCm qualification plan changed before worker launch"
            )
        started_epoch = time.time()
        child_env = _child_environment(plan)
        preflight = _run_rocm_egl_preflight(plan, child_env=child_env)
        used_path, _total, _vendor, _device = (
            collector._selected_gpu_memory_files_rocm(plan)  # noqa: SLF001
        )
        ceiling = int(collector.EXPECTED_CAPS["selected_device_vram_byte_ceiling"])
        scenes = predecessor._scene_slices_v2(plan)  # noqa: SLF001
        probes: list[dict[str, Any]] = []
        for index in QUALIFICATION_PROBE_ORDER:
            scene = scenes[index]
            argv = _worker_argv_qualification(
                scene_index=index,
                plan_path=Path(str(plan_binding["path"])),
                expected_plan_byte_count=int(plan_binding["byte_count"]),
                expected_plan_sha256=str(plan_binding["sha256"]),
                authority_path=Path(str(authority_binding["path"])),
                expected_authority_byte_count=int(authority_binding["byte_count"]),
                expected_authority_sha256=str(authority_binding["sha256"]),
                reservation_binding=absolute_reservation,
                orchestrator_nonce=nonce,
            )
            worker = _run_worker_with_watchdog(
                argv,
                scene=scene,
                child_env=child_env,
                used_path=used_path,
                ceiling_bytes=ceiling,
            )
            barrier = predecessor._wait_for_vram_release_v2(  # noqa: SLF001
                used_path,
                baseline_used_bytes=int(worker["prelaunch_baseline_used_bytes"]),
                ceiling_bytes=ceiling,
            )
            if (
                worker["exit_code"] != 0
                or worker["watchdog_timeout"] is not False
                or worker["selected_device_vram_cap_breached"] is not False
                or worker["fresh_process_group"] is not True
            ):
                raise GenesisRocmBackendQualificationError(
                    f"ROCm qualification worker {index} failed"
                )
            scene_result, scene_binding = predecessor._load_scene_result_v2(  # noqa: SLF001
                collection_root=collection_root,
                scene=scene,
                authority=authority,
                authority_binding=authority_binding,
                plan=plan,
                # Workers record the no-follow pilot binding shape.  The
                # authority and top-level receipt retain the standard shape.
                plan_binding=worker_plan_binding,
                reservation_binding=relative_reservation,
                worker_receipt=worker,
            )
            counts = scene_result["observed_counts"]
            if counts != predecessor._scene_expected_counts_v2(str(scene["role"])):  # noqa: SLF001
                raise GenesisRocmBackendQualificationError(
                    "ROCm qualification counts changed"
                )
            if not _all_numbers_finite(scene_result):
                raise GenesisRocmBackendQualificationError(
                    "ROCm qualification metric is nonfinite"
                )
            probes.append(
                {
                    "scene_index": index,
                    "role": str(scene["role"]),
                    "scene_id": str(scene["scene_id"]),
                    "worker": worker,
                    "release_barrier": barrier,
                    "scene_result_binding": scene_binding,
                    "observed_counts": counts,
                    "existing_scene_validation_passed": True,
                    "probe_output_scientific_reuse_authorized": False,
                }
            )
        kernel_audit = _kernel_events_since(started_epoch)
        maximum = max(float(row["worker"]["elapsed_seconds"]) for row in probes)
        projected = 64.0 * maximum + SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS
        passed = (
            kernel_audit["new_amdgpu_ring_timeout_or_reset_count"] == 0
            and projected <= float(collector.EXPECTED_CAPS["wall_seconds"])
        )
        if not passed:
            raise GenesisRocmBackendQualificationError(
                "ROCm qualification timing/kernel gate failed"
            )
        result = {
            "schema": QUALIFICATION_RESULT_SCHEMA,
            "status": QUALIFICATION_RESULT_STATUS,
            "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
            "backend": "amdgpu",
            "backend_api": "gs.amdgpu",
            "qualification_contract": copy.deepcopy(QUALIFICATION_CONTRACT),
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "rocm_egl_preflight": preflight,
            "probe_order": list(QUALIFICATION_PROBE_ORDER),
            "probes": probes,
            "kernel_reset_audit": kernel_audit,
            "timing_gate": {
                "cold_scene_12_worker_elapsed_seconds": float(
                    probes[0]["worker"]["elapsed_seconds"]
                ),
                "warm_scene_0_worker_elapsed_seconds": float(
                    probes[1]["worker"]["elapsed_seconds"]
                ),
                "maximum_worker_elapsed_seconds": maximum,
                "projected_scientific_wall_seconds": projected,
                "wall_ceiling_seconds": collector.EXPECTED_CAPS["wall_seconds"],
                "passed": True,
            },
            "contact_force_route_audit": copy.deepcopy(
                collector.CONTACT_FORCE_ROUTE_AUDIT
            ),
            "all_existing_scene_gates_passed": True,
            "exact_v03_renderer_compatibility_passed": True,
            "scientific_attempt_root_absent": not plan_builder.DEFAULT_ATTEMPT_ROOT.exists(),
            "probe_output_scientific_reuse_authorized": False,
            "authorizes_scientific_authority_consideration": True,
            "authorizes_retry_or_resume": False,
        }
        if set(result) != QUALIFICATION_RESULT_FIELDS:
            raise GenesisRocmBackendQualificationError(
                "ROCm qualification result field closure changed"
            )
        pilot.write_json_exclusive(QUALIFICATION_RESULT_PATH, result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = collector.build_parser()
    parser.description = (
        "Run the separately authorized two-scene Genesis 0.4.6 ROCm/HIP "
        "qualification; outputs are permanently ineligible for science."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if "--worker-scene-index" in raw:
        with _configured_qualification_collector():
            return collector.main(raw)
    args = build_parser().parse_args(raw)
    try:
        authority, authority_binding, plan, plan_binding = (
            validate_qualification_authority(
                args.authority,
                expected_sha256=args.expected_authority_sha256,
                expected_byte_count=args.expected_authority_byte_count,
            )
        )
        if (
            args.plan.resolve(strict=True)
            != Path(str(plan_binding["path"])).resolve(strict=True)
            or args.expected_plan_sha256 != plan_binding["sha256"]
            or args.expected_plan_byte_count != plan_binding["byte_count"]
        ):
            raise GenesisRocmBackendQualificationError(
                "ROCm qualification CLI plan pins differ from authority"
            )
        result = execute_qualification(
            authority,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(QUALIFICATION_RESULT_PATH),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        if plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_dir():
            terminal = {
                "schema": QUALIFICATION_TERMINAL_SCHEMA,
                "status": "FAIL_GENESIS_ROCM_BACKEND_V1_QUALIFICATION_HARD_STOP",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "qualification_result_binding": None,
                "authorizes_scientific_authority": False,
                "authorizes_retry_or_resume": False,
            }
            terminal_path = plan_builder.QUALIFICATION_ATTEMPT_ROOT / "terminal.json"
            if not terminal_path.exists():
                pilot.write_json_exclusive(terminal_path, terminal)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GenesisRocmBackendQualificationError",
    "QUALIFICATION_AUTHORITY_FIELDS",
    "QUALIFICATION_AUTHORITY_SCHEMA",
    "QUALIFICATION_AUTHORITY_STATUS",
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_PROBE_ORDER",
    "QUALIFICATION_RESULT_FIELDS",
    "QUALIFICATION_RESULT_PATH",
    "QUALIFICATION_RESULT_SCHEMA",
    "QUALIFICATION_RESULT_STATUS",
    "SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS",
    "WORKER_TIMEOUT_SECONDS",
    "execute_qualification",
    "validate_qualification_authority",
]
