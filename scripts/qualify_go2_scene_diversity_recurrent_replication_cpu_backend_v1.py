#!/usr/bin/env python3
"""Run the separately authorized two-scene CPU-backend qualification.

This stage is non-scientific and cannot populate the scientific attempt.  It
executes exactly scene 12 then scene 0 in fresh, watchdog-owned process groups
and publishes only a qualification decision.  Probe outputs are permanently
ineligible for scientific reuse.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_cpu_backend_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as collector  # noqa: E402


pilot = collector.pilot
predecessor = collector.predecessor
calibration = predecessor.calibration_supervisor

QUALIFICATION_AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "qualification_authority_v1"
)
QUALIFICATION_AUTHORITY_STATUS = "AUTHORIZED_CPU_BACKEND_QUALIFICATION_ONLY"
QUALIFICATION_AUTHORITY_FIELDS = frozenset(
    set(predecessor.AUTHORITY_FIELDS) | {"qualification_contract"}
)
QUALIFICATION_RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "qualification_reservation_v1"
)
QUALIFICATION_SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "qualification_scene_result_v1"
)
QUALIFICATION_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "qualification_result_v1"
)
QUALIFICATION_TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_"
    "qualification_terminal_v1"
)
QUALIFICATION_RESULT_STATUS = "PASS_CPU_BACKEND_QUALIFICATION"
QUALIFICATION_PROBE_ORDER = (12, 0)
WORKER_TIMEOUT_SECONDS = 180.0
SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS = 900.0
QUALIFICATION_RESULT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "attempt_id",
        "backend",
        "qualification_contract",
        "authority_binding",
        "plan_binding",
        "graphics_preflight",
        "probe_order",
        "probes",
        "kernel_reset_audit",
        "timing_gate",
        "all_existing_scene_gates_passed",
        "scientific_attempt_root_absent",
        "probe_output_scientific_reuse_authorized",
        "authorizes_cpu_scientific_authority_consideration",
        "authorizes_retry_or_resume",
    }
)
QUALIFICATION_PROBE_FIELDS = frozenset(
    {
        "scene_index",
        "role",
        "scene_id",
        "worker",
        "release_barrier",
        "scene_result_binding",
        "observed_counts",
        "existing_scene_validation_passed",
        "probe_output_scientific_reuse_authorized",
    }
)
QUALIFICATION_WORKER_FIELDS = frozenset(
    {
        "scene_index",
        "role",
        "scene_id",
        "pid",
        "parent_pid",
        "process_group_id",
        "fresh_process_group",
        "sys_executable",
        "prelaunch_baseline_used_bytes",
        "peak_selected_device_vram_bytes",
        "selected_device_vram_cap_breached",
        "watchdog_timeout",
        "exit_code",
        "elapsed_seconds",
    }
)
QUALIFICATION_RELEASE_BARRIER_FIELDS = frozenset(
    {
        "status",
        "read_only",
        "counter_path",
        "baseline_used_bytes",
        "release_margin_bytes",
        "release_ceiling_bytes",
        "absolute_vram_ceiling_bytes",
        "required_consecutive_samples",
        "sample_interval_seconds",
        "sample_count",
        "minimum_used_bytes",
        "maximum_used_bytes",
        "final_used_bytes",
        "final_consecutive_samples",
        "elapsed_seconds",
    }
)
QUALIFICATION_GRAPHICS_PREFLIGHT_FIELDS = frozenset(
    {
        "phase",
        "status",
        "environment",
        "expectation",
        "vulkan_stdout_sha256",
        "egl_stdout_sha256",
        "egl_stderr_sha256",
        "egl_exit_code",
    }
)
QUALIFICATION_KERNEL_AUDIT_FIELDS = frozenset(
    {
        "query_succeeded",
        "new_amdgpu_ring_timeout_or_reset_count",
        "matching_lines_sha256",
    }
)
QUALIFICATION_TIMING_GATE_FIELDS = frozenset(
    {
        "maximum_worker_elapsed_seconds",
        "projected_scientific_wall_seconds",
        "wall_ceiling_seconds",
        "passed",
    }
)
QUALIFICATION_CONTRACT = {
    "backend": "cpu",
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
    "timing_gate_formula": "64*max(worker_elapsed_seconds)+900<=7200",
    "probe_output_scientific_reuse_authorized": False,
    "cpu_backend_failure_authorizes_retry": False,
}

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_preregistration_2026-08-04.md"
)
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_source_review_2026-08-04.json"
)
QUALIFICATION_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification_authority_2026-08-04.json"
)
QUALIFICATION_RESULT_PATH = (
    plan_builder.QUALIFICATION_ATTEMPT_ROOT / "qualification_result.json"
)

_CPU_WORKER_ARGV = collector._worker_argv_cpu  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


class CpuBackendQualificationError(RuntimeError):
    """Raised when the bounded CPU qualification fails closed."""


def _binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
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


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def _worker_argv_qualification(**kwargs: Any) -> list[str]:
    argv = _CPU_WORKER_ARGV(**kwargs)
    if len(argv) < 2 or Path(argv[1]).resolve() != Path(collector.__file__).resolve():
        raise CpuBackendQualificationError("CPU worker entry point changed")
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
        label="CPU qualification reservation",
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
        raise CpuBackendQualificationError("qualification reservation changed")
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
        "_read_collection_reservation_cpu": _read_qualification_reservation,
        "_worker_argv_cpu": _worker_argv_qualification,
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


def _source_paths() -> Mapping[str, Path]:
    from scripts import run_go2_scene_diversity_recurrent_replication_cpu_backend_v1 as runner

    return runner.SOURCE_PATHS


def validate_qualification_authority(
    authority_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority_raw, historical = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="CPU qualification authority",
    )
    authority_binding = _standard_binding(historical)
    plan_binding = authority_raw.get("plan_binding")
    if not isinstance(plan_binding, Mapping):
        raise CpuBackendQualificationError("qualification plan binding absent")
    plan_raw, observed_plan_binding = pilot.read_bound_json(
        Path(str(plan_binding["path"])),
        expected_sha256=str(plan_binding["sha256"]),
        expected_byte_count=int(plan_binding["byte_count"]),
        label="CPU qualification plan",
    )
    if _standard_binding(observed_plan_binding) != dict(plan_binding):
        raise CpuBackendQualificationError("qualification plan binding changed")
    plan = plan_builder.validate_cpu_plan(
        plan_raw,
        expected_attempt_id=plan_builder.QUALIFICATION_ATTEMPT_ID,
        expected_output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
    )
    sources = _source_paths()
    bound_sources = authority_raw.get("source_bindings")
    if not isinstance(bound_sources, Mapping) or set(bound_sources) != set(sources):
        raise CpuBackendQualificationError("qualification source closure changed")
    if any(
        dict(bound_sources[name]) != _binding(path)
        for name, path in sources.items()
    ):
        raise CpuBackendQualificationError("qualification source binding changed")
    review_binding = authority_raw.get("source_review_binding")
    prereg_binding = authority_raw.get("preregistration_binding")
    if (
        not isinstance(review_binding, Mapping)
        or not isinstance(prereg_binding, Mapping)
        or dict(prereg_binding) != _binding(PREREGISTRATION)
        or dict(review_binding) != _binding(SOURCE_REVIEW)
    ):
        raise CpuBackendQualificationError("qualification reviewed documents changed")
    review = json.loads(SOURCE_REVIEW.read_bytes())
    if (
        set(authority_raw) != QUALIFICATION_AUTHORITY_FIELDS
        or authority_raw.get("schema") != QUALIFICATION_AUTHORITY_SCHEMA
        or authority_raw.get("status") != QUALIFICATION_AUTHORITY_STATUS
        or authority_raw.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or authority_raw.get("attempt_root")
        != str(plan_builder.QUALIFICATION_ATTEMPT_ROOT.resolve())
        or authority_raw.get("collection_root")
        != str(plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve())
        or authority_raw.get("qualification_contract") != QUALIFICATION_CONTRACT
        or authority_raw.get("caps") != collector.EXPECTED_CAPS
        or authority_raw.get("permissions") != collector.EXPECTED_PERMISSIONS
        or review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_cpu_backend_v1_source_review_v1"
        or review.get("status") != "PASS_INDEPENDENT_SOURCE_REVIEW"
        or review.get("protected_material_opened") is not False
        or review.get("findings") != []
        or review.get("preregistration_binding") != dict(prereg_binding)
        or review.get("qualification_plan_binding") != dict(plan_binding)
        or review.get("source_bindings") != bound_sources
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise CpuBackendQualificationError("qualification authority contract changed")
    return dict(authority_raw), authority_binding, plan, dict(plan_binding)


def _reserve_qualification(
    *, authority: Mapping[str, Any], authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any], nonce: str,
) -> tuple[Path, dict[str, Any]]:
    namespace = plan_builder.QUALIFICATION_ATTEMPT_ROOT.parent
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    if namespace.parent != development or namespace.is_symlink():
        raise CpuBackendQualificationError("qualification namespace changed")
    namespace.mkdir(mode=0o700, exist_ok=True)
    os.mkdir(plan_builder.QUALIFICATION_ATTEMPT_ROOT, mode=0o700)
    pilot.write_json_exclusive(
        plan_builder.QUALIFICATION_ATTEMPT_ROOT / "reservation.json",
        {
            "schema": QUALIFICATION_RESERVATION_SCHEMA,
            "status": "CONSUMED_ONE_SHOT_QUALIFICATION_ATTEMPT",
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "scientific_attempt_root": str(plan_builder.DEFAULT_ATTEMPT_ROOT.resolve()),
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
    binding = pilot.write_json_exclusive(collection / "reservation.json", reservation)
    return collection.resolve(), binding


def _run_worker_with_watchdog(
    argv: Sequence[str], *, scene: Mapping[str, Any], child_env: Mapping[str, str],
    used_path: Path, ceiling_bytes: int,
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
        if used > ceiling_bytes:
            cap_breach = True
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
    command = [
        "/usr/bin/journalctl", "-k", "--since", f"@{epoch_seconds:.6f}",
        "--no-pager", "-o", "short-unix",
    ]
    completed = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=15.0,
    )
    if completed.returncode != 0:
        raise CpuBackendQualificationError("kernel reset audit was unavailable")
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
    authority: Mapping[str, Any], *, authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any], plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute exactly two non-scientific full-scene probes."""

    with _configured_qualification_collector(), collector._configured_predecessor_collector_cpu():  # noqa: SLF001
        # Consume the one-shot qualification before any graphics, resource, or
        # worker action.  A failure in any mandatory preflight must therefore
        # leave a terminalized attempt, never a rerunnable authority.
        nonce = hashlib.sha256(os.urandom(32)).hexdigest()
        collection_root, absolute_reservation = _reserve_qualification(
            authority=authority, authority_binding=authority_binding,
            plan_binding=plan_binding, nonce=nonce,
        )
        relative_reservation = predecessor.kernel._relative_output_binding(  # noqa: SLF001
            absolute_reservation, output_root=collection_root
        )
        started_epoch = time.time()
        child_env = calibration._child_environment(plan)  # noqa: SLF001
        wall_started = time.monotonic()
        graphics = calibration._run_graphics_preflight(  # noqa: SLF001
            plan, child_env=child_env, wall_started=wall_started,
            wall_ceiling=float(collector.EXPECTED_CAPS["wall_seconds"]),
        )
        used_path, _total, _vendor, _device = calibration._selected_gpu_memory_files(plan)  # noqa: SLF001
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
                argv, scene=scene, child_env=child_env,
                used_path=used_path, ceiling_bytes=ceiling,
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
                raise CpuBackendQualificationError(
                    f"CPU qualification worker {index} failed"
                )
            # Reuse V2's complete scene/result/render/receipt/mesh validator.
            scene_result, scene_binding = predecessor._load_scene_result_v2(  # noqa: SLF001
                collection_root=collection_root,
                scene=scene,
                authority=authority,
                authority_binding=authority_binding,
                plan=plan,
                plan_binding=plan_binding,
                reservation_binding=relative_reservation,
                worker_receipt=worker,
            )
            counts = scene_result["observed_counts"]
            if counts != predecessor._scene_expected_counts_v2(str(scene["role"])):  # noqa: SLF001
                raise CpuBackendQualificationError("qualification counts changed")
            if not _all_numbers_finite(scene_result):
                raise CpuBackendQualificationError("qualification metric is nonfinite")
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
            raise CpuBackendQualificationError("CPU qualification gate failed")
        result = {
            "schema": QUALIFICATION_RESULT_SCHEMA,
            "status": QUALIFICATION_RESULT_STATUS,
            "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
            "backend": "cpu",
            "qualification_contract": QUALIFICATION_CONTRACT,
            "authority_binding": dict(authority_binding),
            "plan_binding": dict(plan_binding),
            "graphics_preflight": graphics,
            "probe_order": list(QUALIFICATION_PROBE_ORDER),
            "probes": probes,
            "kernel_reset_audit": kernel_audit,
            "timing_gate": {
                "maximum_worker_elapsed_seconds": maximum,
                "projected_scientific_wall_seconds": projected,
                "wall_ceiling_seconds": collector.EXPECTED_CAPS["wall_seconds"],
                "passed": True,
            },
            "all_existing_scene_gates_passed": True,
            "scientific_attempt_root_absent": not plan_builder.DEFAULT_ATTEMPT_ROOT.exists(),
            "probe_output_scientific_reuse_authorized": False,
            "authorizes_cpu_scientific_authority_consideration": True,
            "authorizes_retry_or_resume": False,
        }
        if set(result) != QUALIFICATION_RESULT_FIELDS:
            raise CpuBackendQualificationError(
                "qualification result field closure changed"
            )
        pilot.write_json_exclusive(QUALIFICATION_RESULT_PATH, result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = collector.build_parser()
    parser.description = (
        "Run the separate two-scene CPU-backend qualification; its outputs "
        "are ineligible for scientific reuse."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Worker invocations use the exact collector parser/contract.
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
            args.plan.resolve(strict=True) != Path(str(plan_binding["path"])).resolve(strict=True)
            or args.expected_plan_sha256 != plan_binding["sha256"]
            or args.expected_plan_byte_count != plan_binding["byte_count"]
        ):
            raise CpuBackendQualificationError(
                "qualification CLI plan pins differ from authority"
            )
        result = execute_qualification(
            authority, authority_binding=authority_binding,
            plan=plan, plan_binding=plan_binding,
        )
        print(json.dumps({"status": result["status"], "result": str(QUALIFICATION_RESULT_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        if plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_dir():
            terminal = {
                "schema": QUALIFICATION_TERMINAL_SCHEMA,
                "status": "FAIL_CPU_BACKEND_QUALIFICATION_HARD_STOP",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "qualification_result_binding": None,
                "authorizes_cpu_scientific_authority": False,
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
    "CpuBackendQualificationError",
    "QUALIFICATION_AUTHORITY_FIELDS",
    "QUALIFICATION_AUTHORITY_SCHEMA",
    "QUALIFICATION_AUTHORITY_STATUS",
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_PROBE_ORDER",
    "QUALIFICATION_GRAPHICS_PREFLIGHT_FIELDS",
    "QUALIFICATION_KERNEL_AUDIT_FIELDS",
    "QUALIFICATION_PROBE_FIELDS",
    "QUALIFICATION_RELEASE_BARRIER_FIELDS",
    "QUALIFICATION_RESULT_PATH",
    "QUALIFICATION_RESULT_FIELDS",
    "QUALIFICATION_RESULT_SCHEMA",
    "QUALIFICATION_RESULT_STATUS",
    "QUALIFICATION_TIMING_GATE_FIELDS",
    "QUALIFICATION_WORKER_FIELDS",
    "SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS",
    "WORKER_TIMEOUT_SECONDS",
    "execute_qualification",
    "validate_qualification_authority",
]
