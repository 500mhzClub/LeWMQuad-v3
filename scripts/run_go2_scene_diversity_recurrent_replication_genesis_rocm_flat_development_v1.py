#!/usr/bin/env python3
"""Run the flat two-scene Genesis/ROCm development qualification.

This module deliberately owns its coordinator and worker interfaces.  It does
not enter any historical ``_configured_*`` context and never changes an
imported module attribute.  Historical plan, source-input, and textured-v03
parity validation completes before any root, GPU, Genesis, ROCm, HIP, or EGL
operation.  Runtime differences are carried in the exact child environment
and a worker-local runtime dictionary.

Qualification output is development-only and may not be reused by science.
Only the small PASS decision emitted after every gate succeeds is eligible to
be bound by a later, separately reviewed scientific plan.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import subprocess
import sys
import time
from types import FunctionType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (
    REPO_ROOT,
    REPO_ROOT / "lewm_genesis",
    REPO_ROOT / "lewm_worlds",
):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_flat_development_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1 as rocm_core  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v2 as scene_core  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as kernel  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as historical_preflight  # noqa: E402


PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_preregistration_2026-08-05.md"
)
PREREGISTRATION_SHA256 = (
    "5712c934dee265d301eb86cf7b06faeda52ff10a14945d0a81420523438b95b5"
)
PREREGISTRATION_BYTE_COUNT = 9_076

QUALIFICATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_result_v1"
)
DECISION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_decision_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_terminal_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_reservation_v1"
)
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_scene_result_v1"
)

PASS_STATUS = "PASS_GENESIS_ROCM_FLAT_DEVELOPMENT_V1_QUALIFICATION"
FAIL_STATUS = "FAIL_GENESIS_ROCM_FLAT_DEVELOPMENT_V1_QUALIFICATION"
SCENE_PASS_STATUS = "SCENE_PHYSICS_COMPLETE"
PROBE_ORDER = tuple(plan_builder.QUALIFICATION_SCENE_INDICES)
WORKER_TIMEOUT_SECONDS = float(
    plan_builder.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
NONCOLLECTION_RESERVE_SECONDS = float(
    plan_builder.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)
VRAM_CEILING_BYTES = int(
    scene_core.EXPECTED_CAPS["selected_device_vram_byte_ceiling"]
)
POLL_SECONDS = 0.02
GROUP_EXIT_TIMEOUT_SECONDS = 5.0

EXPECTED_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "EGL_DEVICE_ID",
        "GS_BACKEND",
        "GS_CACHE_FILE_PATH",
        "GS_ENABLE_FASTCACHE",
        "GS_ENABLE_NDARRAY",
        "GS_ENABLE_ZEROCOPY",
        "GS_PARA_LEVEL",
        "HIP_VISIBLE_DEVICES",
        "PATH",
        "PYOPENGL_PLATFORM",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "ROCM_PATH",
        "ROCR_VISIBLE_DEVICES",
    }
)

QUALIFICATION_CONTRACT = {
    "backend": "amdgpu",
    "backend_api": "gs.amdgpu",
    "genesis_version": "0.4.6",
    "probe_scene_indices_in_order": list(PROBE_ORDER),
    "fresh_worker_processes": 2,
    "states_per_worker": 4,
    "candidate_actions_per_state": 9,
    "branches_per_worker": 36,
    "context_frames_per_worker": 12,
    "target_frames_per_worker": 36,
    "stored_rgb_frames_per_worker": 48,
    "auxiliary_depth_validation_renders_per_worker": 48,
    "worker_process_group_watchdog_seconds": WORKER_TIMEOUT_SECONDS,
    "selected_device_vram_byte_ceiling": VRAM_CEILING_BYTES,
    "vram_sample_interval_seconds": POLL_SECONDS,
    "release_after_every_worker_including_final": True,
    "leaked_process_group_forbidden": True,
    "new_amdgpu_reset_wedge_or_hsa_exception_forbidden": True,
    "timing_gate_formula": "64 * max(worker_elapsed_seconds) + 900 <= 7200",
    "probe_output_scientific_reuse_authorized": False,
    "retry_resume_overwrite_authorized": False,
}


class FlatQualificationError(RuntimeError):
    """Raised when the flat qualification must fail closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "sha256": str(value.get("sha256", value.get("file_sha256"))),
        "byte_count": int(value["byte_count"]),
    }


def _relative_binding(
    value: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    return kernel._relative_output_binding(value, output_root=output_root)  # noqa: SLF001


def _preregistration_binding() -> dict[str, Any]:
    expected = {
        "path": str(PREREGISTRATION.resolve(strict=True)),
        "sha256": PREREGISTRATION_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }
    observed = _standard_binding(pilot.file_binding(PREREGISTRATION))
    if observed != expected:
        raise FlatQualificationError("flat preregistration changed")
    return expected


def _shared_state_snapshot() -> tuple[object, object, bytes, bytes]:
    return (
        pilot.EXECUTION_ENVIRONMENT,
        pilot.GRAPHICS_PREFLIGHT_EXPECTATION,
        _canonical(pilot.EXECUTION_ENVIRONMENT),
        _canonical(pilot.GRAPHICS_PREFLIGHT_EXPECTATION),
    )


def _require_shared_state_unchanged(
    snapshot: tuple[object, object, bytes, bytes]
) -> None:
    environment, graphics, environment_bytes, graphics_bytes = snapshot
    if (
        pilot.EXECUTION_ENVIRONMENT is not environment
        or pilot.GRAPHICS_PREFLIGHT_EXPECTATION is not graphics
        or _canonical(pilot.EXECUTION_ENVIRONMENT) != environment_bytes
        or _canonical(pilot.GRAPHICS_PREFLIGHT_EXPECTATION) != graphics_bytes
    ):
        raise FlatQualificationError("historical pilot shared state changed")


def _read_and_validate_plan(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    snapshot = _shared_state_snapshot()
    raw, binding = pilot.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="flat qualification exact plan",
    )
    plan = plan_builder.validate_flat_plan(raw, role="qualification")
    pilot.require_plan_bindings(plan)
    bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    bounded._validate_bound_scenes(plan)  # noqa: SLF001
    scenes = scene_core._scene_slices_v2(plan)  # noqa: SLF001
    _preregistration_binding()
    _require_shared_state_unchanged(snapshot)
    expected_environment = plan_builder.rocm_execution_environment(
        "qualification"
    )
    if (
        set(expected_environment) != EXPECTED_ENVIRONMENT_KEYS
        or plan["execution_contract"]["environment"] != expected_environment
        or tuple(PROBE_ORDER) != (12, 0)
        or len(scenes) != 64
        or plan.get("output_root")
        != str(plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve(strict=False))
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.exists()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or plan_builder.QUALIFICATION_OUTPUT_ROOT.exists()
        or plan_builder.QUALIFICATION_OUTPUT_ROOT.is_symlink()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        or plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    ):
        raise FlatQualificationError("flat qualification static contract changed")
    return plan, _standard_binding(binding), scenes


def _read_worker_plan(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Read the parent-validated exact plan without post-reservation parity.

    The parent has already rehashed every input and completed historical
    parity before creating the reservation.  A worker authenticates only the
    immutable plan bytes it was given and the small identity fields needed to
    select its scene; it must not repeat that historical validation after the
    one-shot root exists.
    """

    raw, binding = pilot.read_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_byte_count=expected_byte_count,
        label="flat qualification worker exact plan",
    )
    if (
        not isinstance(raw, dict)
        or raw.get("schema")
        != "lewm_go2_world_model_counterfactual_pilot_plan_v1"
        or raw.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or raw.get("output_root")
        != str(plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve(strict=True))
        or raw.get("execution_contract", {}).get("environment")
        != plan_builder.rocm_execution_environment("qualification")
        or raw.get("expected_counts") != scene_core.EXPECTED_COUNTS
        or not plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_dir()
        or plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_symlink()
        or not plan_builder.QUALIFICATION_OUTPUT_ROOT.is_dir()
        or plan_builder.QUALIFICATION_OUTPUT_ROOT.is_symlink()
    ):
        raise FlatQualificationError("worker exact-plan identity changed")
    scenes = scene_core._scene_slices_v2(raw)  # noqa: SLF001
    return raw, _standard_binding(binding), scenes


def _validate_child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(plan["execution_contract"]["environment"])
    if (
        set(expected) != EXPECTED_ENVIRONMENT_KEYS
        or expected != plan_builder.rocm_execution_environment("qualification")
        or "HSA_OVERRIDE_GFX_VERSION" in expected
        or "MESA_VK_DEVICE_SELECT" in expected
    ):
        raise FlatQualificationError("flat ROCm child environment changed")
    return expected


def _require_worker_environment(plan: Mapping[str, Any]) -> None:
    expected = _validate_child_environment(plan)
    observed = dict(os.environ)
    # CPython locale coercion adds this key after exec even when Popen receives
    # the exact 17-key mapping.  It is not an ambient selector or launch input.
    coerced_locale = observed.pop("LC_CTYPE", None)
    if coerced_locale not in {None, "C.UTF-8"} or observed != expected:
        extra = sorted(set(observed) - set(expected))
        missing = sorted(set(expected) - set(observed))
        changed = sorted(
            key
            for key in set(expected) & set(observed)
            if observed[key] != expected[key]
        )
        raise FlatQualificationError(
            "worker environment is not exact: "
            f"extra={extra}, missing={missing}, changed={changed}"
        )


def _reserve_qualification(
    *, plan_binding: Mapping[str, Any], nonce: str
) -> tuple[Path, dict[str, Any]]:
    attempt = plan_builder.QUALIFICATION_ATTEMPT_ROOT
    collection = plan_builder.QUALIFICATION_OUTPUT_ROOT
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    namespace = attempt.parent
    if (
        attempt.resolve(strict=False).parent != namespace.resolve(strict=False)
        or not attempt.resolve(strict=False).is_relative_to(development)
        or namespace.is_symlink()
        or attempt.exists()
        or attempt.is_symlink()
        or collection.exists()
        or collection.is_symlink()
    ):
        raise FlatQualificationError("qualification root is not exact and fresh")
    if not namespace.exists():
        os.mkdir(namespace, mode=0o700)
    if not namespace.is_dir() or namespace.is_symlink():
        raise FlatQualificationError("qualification namespace changed")
    os.mkdir(attempt, mode=0o700)
    os.mkdir(collection, mode=0o700)
    os.mkdir(collection / "scenes", mode=0o700)
    os.mkdir(collection / "scenes" / "train", mode=0o700)
    os.mkdir(collection / "scenes" / "eval", mode=0o700)
    os.mkdir(collection / "scene_results", mode=0o700)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_FLAT_TWO_PROBE_QUALIFICATION_CONSUMED",
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "attempt_root": str(attempt.resolve(strict=True)),
        "collection_root": str(collection.resolve(strict=True)),
        "plan_binding": dict(plan_binding),
        "preregistration_binding": _preregistration_binding(),
        "orchestrator_nonce": nonce,
        "orchestrator_pid": os.getpid(),
        "probe_scene_indices_in_order": list(PROBE_ORDER),
        "qualification_contract": copy.deepcopy(QUALIFICATION_CONTRACT),
        "root_creation_consumes_attempt": True,
        "probe_output_scientific_reuse_authorized": False,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    binding = pilot.write_json_exclusive(
        collection / "reservation.json", reservation
    )
    return collection.resolve(strict=True), _standard_binding(binding)


def _read_reservation(
    *,
    binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    nonce: str,
    expected_orchestrator_pid: int,
) -> dict[str, Any]:
    value, actual = pilot.read_bound_json(
        Path(str(binding["path"])),
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label="flat qualification reservation",
    )
    if (
        _standard_binding(actual) != dict(binding)
        or value.get("schema") != RESERVATION_SCHEMA
        or value.get("status")
        != "RESERVED_FLAT_TWO_PROBE_QUALIFICATION_CONSUMED"
        or value.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("orchestrator_nonce") != nonce
        or value.get("orchestrator_pid") != expected_orchestrator_pid
        or value.get("probe_scene_indices_in_order") != list(PROBE_ORDER)
        or value.get("qualification_contract") != QUALIFICATION_CONTRACT
        or value.get("probe_output_scientific_reuse_authorized") is not False
        or any(
            value.get(key) is not False
            for key in (
                "retry_authorized",
                "resume_authorized",
                "overwrite_authorized",
                "refill_authorized",
            )
        )
    ):
        raise FlatQualificationError("flat qualification reservation changed")
    return value


def _clone_worker_runtime(
    runtime: Mapping[str, Any], *, scene_root: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Clone the renderer globals and alter only a worker-local runtime map."""

    local = dict(runtime)
    cached_box_obj = local.get("cached_box_obj")
    render_builder = local.get("build_textured_v03_scene")
    render_globals = getattr(render_builder, "__globals__", None)
    rollout_config = local.get("RolloutConfig")
    if (
        not callable(cached_box_obj)
        or not callable(render_builder)
        or not isinstance(render_globals, dict)
        or not callable(rollout_config)
    ):
        raise FlatQualificationError("worker-local runtime seam changed")
    original_global_cached_box_obj = render_globals.get("cached_box_obj")
    cache_root = scene_root / "derived_meshes"

    def scene_local_cached_box_obj(
        size_xyz_m: Sequence[float], *, tiles_per_m: float = 0.7
    ) -> str:
        return cached_box_obj(
            tuple(float(value) for value in size_xyz_m),
            tiles_per_m=tiles_per_m,
            cache_dir=cache_root,
        )

    cloned_globals = dict(render_globals)
    cloned_globals["cached_box_obj"] = scene_local_cached_box_obj
    cloned_builder = FunctionType(
        render_builder.__code__,
        cloned_globals,
        name=render_builder.__name__,
        argdefs=render_builder.__defaults__,
        closure=render_builder.__closure__,
    )
    cloned_builder.__kwdefaults__ = copy.copy(render_builder.__kwdefaults__)
    cloned_builder.__annotations__ = dict(render_builder.__annotations__)
    cloned_builder.__dict__.update(render_builder.__dict__)
    cloned_builder.__doc__ = render_builder.__doc__
    cloned_builder.__module__ = render_builder.__module__
    cloned_builder.__qualname__ = render_builder.__qualname__

    def safe_rollout_config(**kwargs: Any) -> Any:
        if "foot_contact_source" in kwargs:
            raise FlatQualificationError(
                "caller attempted to override zero-contact rollout"
            )
        return rollout_config(foot_contact_source="zero", **kwargs)

    local["cached_box_obj"] = scene_local_cached_box_obj
    local["build_textured_v03_scene"] = cloned_builder
    local["RolloutConfig"] = safe_rollout_config
    if render_globals.get("cached_box_obj") is not original_global_cached_box_obj:
        raise FlatQualificationError("renderer module globals were mutated")
    return local, cache_root, {
        "renderer_code_identity_preserved": (
            cloned_builder.__code__ is render_builder.__code__
        ),
        "renderer_globals_cloned": cloned_builder.__globals__ is not render_globals,
        "original_renderer_globals_unchanged": True,
        "rollout_config_worker_local": True,
        "foot_contact_source": "zero",
    }


def _collect_scene_worker(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_index: int,
    orchestrator_pid: int,
    nonce: str,
) -> dict[str, Any]:
    if os.getppid() != orchestrator_pid or orchestrator_pid <= 1:
        raise FlatQualificationError("worker parent ownership changed")
    _require_worker_environment(plan)
    scenes = scene_core._scene_slices_v2(plan)  # noqa: SLF001
    if type(scene_index) is not int or scene_index not in PROBE_ORDER:
        raise FlatQualificationError("worker scene index changed")
    scene = scenes[scene_index]
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    states = list(scene["states"])
    collection_root = plan_builder.QUALIFICATION_OUTPUT_ROOT.resolve(strict=True)
    reservation = _read_reservation(
        binding=reservation_binding,
        plan_binding=plan_binding,
        nonce=nonce,
        expected_orchestrator_pid=orchestrator_pid,
    )
    if reservation.get("orchestrator_pid") != orchestrator_pid:
        raise FlatQualificationError("worker reservation ownership changed")
    scene_root = collection_root / "scenes" / role / scene_id
    result_path = collection_root / "scene_results" / f"{scene_index:03d}.json"
    if scene_root.exists() or scene_root.is_symlink() or result_path.exists():
        raise FlatQualificationError("worker output is not fresh")
    os.mkdir(scene_root, mode=0o700)
    os.mkdir(scene_root / "state_receipts", mode=0o700)
    started = time.perf_counter()

    kernel._validate_python_runtime(plan)  # noqa: SLF001
    versions = kernel._capture_runtime_versions()  # noqa: SLF001
    imported_runtime = kernel._runtime_imports(textured_v03=True)  # noqa: SLF001
    runtime, cache_root, local_runtime_audit = _clone_worker_runtime(
        imported_runtime, scene_root=scene_root
    )
    platform = runtime["load_platform_manifest"](
        plan["runtime_bindings"]["platform_manifest"]["path"]
    )
    resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), REPO_ROOT)
    if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
        raise FlatQualificationError("platform resolved a different Go2 URDF")
    registry = runtime["PrimitiveRegistry"].from_yaml(
        plan["runtime_bindings"]["primitive_registry"]["path"]
    )
    action_blocks = kernel._load_action_blocks(  # noqa: SLF001
        plan=plan,
        registry=registry,
        expand=runtime["expand_primitive_to_block"],
    )
    genesis_initialization = rocm_core._initialize_from_plan_first_scene_rocm(  # noqa: SLF001
        plan=plan
    )
    receipts, frames, quality, sentinels, metrics = kernel._collect_scene(  # noqa: SLF001
        plan=plan,
        states=states,
        runtime=runtime,
        platform=platform,
        registry=registry,
        action_blocks=action_blocks,
    )
    ordered_ids = [str(row["state_id"]) for row in states]
    if [str(row["state"]["state_id"]) for row in receipts] != ordered_ids:
        raise FlatQualificationError("worker changed planned state order")
    stored_rgb_bytes = sum(int(frame["byte_count"]) for frame in frames)
    if stored_rgb_bytes > int(
        scene_core.EXPECTED_CAPS["stored_rgb_byte_ceiling"]
    ):
        raise FlatQualificationError("worker stored RGB ceiling exceeded")
    render_receipt = {
        "schema": pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA,
        "attempt_id": str(plan["attempt_id"]),
        "status": "RENDER_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "scene": {
            "role": role,
            "scene_id": scene_id,
            "family": str(states[0]["family"]),
            "scene_manifest_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                states[0],
                binding_name="scene_manifest_binding",
                output_root=collection_root,
            ),
            "scene_genesis_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                states[0],
                binding_name="scene_genesis_binding",
                output_root=collection_root,
            ),
        },
        **bounded._validated_render_receipt_identity_v1(  # noqa: SLF001
            plan=plan, metrics=metrics
        ),
        "frame_receipts": frames,
        "quality_audits": quality,
        "render_sentinel_audits": sentinels,
    }
    render_binding = pilot.write_json_exclusive(
        scene_root / "live_render_receipt.json", render_receipt
    )
    relative_render = _relative_binding(
        render_binding, output_root=collection_root
    )
    state_bindings: list[dict[str, Any]] = []
    for receipt in receipts:
        state_id = str(receipt["state"]["state_id"])
        receipt["render_receipt_binding"] = relative_render
        binding = pilot.write_json_exclusive(
            scene_root / "state_receipts" / f"{state_id}.json", receipt
        )
        state_bindings.append(_relative_binding(binding, output_root=collection_root))
    observed = scene_core._observed_scene_counts_v2(receipts, role=role)  # noqa: SLF001
    expected = scene_core._scene_expected_counts_v2(role)  # noqa: SLF001
    if observed != expected:
        raise FlatQualificationError("worker observed counts changed")
    for name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if int(metrics[name]) != scene_core.STORED_FRAMES_PER_SCENE:
            raise FlatQualificationError(f"worker {name} changed")
    mesh_cache = scene_core._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
        metrics,
        cache_root=cache_root,
        collection_root=collection_root,
    )
    result = {
        "schema": SCENE_RESULT_SCHEMA,
        "status": SCENE_PASS_STATUS,
        "attempt_id": str(plan["attempt_id"]),
        "scene_index": scene_index,
        "role": role,
        "scene_id": scene_id,
        "worker_pid": os.getpid(),
        "orchestrator_pid": orchestrator_pid,
        "sys_executable": str(Path(sys.executable).absolute()),
        "fresh_process": True,
        "execution_seed": int(plan["execution_contract"]["seed"]),
        "genesis_initialization": genesis_initialization,
        "local_runtime_audit": local_runtime_audit,
        "scene_local_mesh_cache": mesh_cache,
        "plan_binding": dict(plan_binding),
        "reservation_binding": dict(reservation_binding),
        "runtime_versions": versions,
        "expected_counts": expected,
        "observed_counts": observed,
        "ordered_state_ids": ordered_ids,
        "state_receipt_bindings": state_bindings,
        "render_receipt_binding": relative_render,
        "scene_metric": metrics,
        "stored_rgb_bytes": stored_rgb_bytes,
        "collection_wall_seconds": time.perf_counter() - started,
        "failure": None,
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "probe_output_scientific_reuse_authorized": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "allows_adaptive_batching": False,
    }
    if (
        not math.isfinite(float(result["collection_wall_seconds"]))
        or result["collection_wall_seconds"] <= 0.0
        or not _all_numbers_finite(result)
    ):
        raise FlatQualificationError("worker emitted nonfinite evidence")
    pilot.write_json_exclusive(result_path, result)
    return result


def _all_numbers_finite(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_worker_with_watchdog(
    argv: Sequence[str],
    *,
    scene: Mapping[str, Any],
    child_env: Mapping[str, str],
    used_path: Path,
) -> dict[str, Any]:
    baseline = scene_core._read_vram_counter_v2(used_path)  # noqa: SLF001
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(child_env),
        start_new_session=True,
    )
    pgid = os.getpgid(process.pid)
    peak = baseline
    samples = 1
    read_errors = 0
    timeout = False
    cap_breach = False
    forced_kill = False
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            try:
                used = scene_core._read_vram_counter_v2(used_path)  # noqa: SLF001
            except Exception:
                read_errors += 1
                forced_kill = True
                os.killpg(pgid, signal.SIGKILL)
                break
            samples += 1
            peak = max(peak, used)
            cap_breach = used > VRAM_CEILING_BYTES
            timeout = elapsed > WORKER_TIMEOUT_SECONDS
            if cap_breach or timeout:
                forced_kill = True
                os.killpg(pgid, signal.SIGKILL)
                break
            time.sleep(POLL_SECONDS)
    finally:
        returncode = int(process.wait())
    deadline = time.monotonic() + GROUP_EXIT_TIMEOUT_SECONDS
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = _group_exists(pgid)
    if leaked:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            leaked = False
    return {
        "scene_index": int(scene["scene_index"]),
        "role": str(scene["role"]),
        "scene_id": str(scene["scene_id"]),
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "process_group_id": pgid,
        "fresh_process_group": pgid == process.pid,
        "process_group_gone_after_exit": not leaked,
        "prelaunch_baseline_used_bytes": baseline,
        "peak_selected_device_vram_bytes": peak,
        "vram_sample_count": samples,
        "vram_read_errors": read_errors,
        "selected_device_vram_cap_breached": cap_breach,
        "watchdog_timeout": timeout,
        "forced_kill": forced_kill,
        "exit_code": returncode,
        "elapsed_seconds": time.monotonic() - started,
    }


def _validate_worker_receipt(worker: Mapping[str, Any]) -> None:
    if (
        worker.get("fresh_process_group") is not True
        or worker.get("process_group_gone_after_exit") is not True
        or worker.get("exit_code") != 0
        or worker.get("watchdog_timeout") is not False
        or worker.get("selected_device_vram_cap_breached") is not False
        or worker.get("vram_read_errors") != 0
        or int(worker.get("vram_sample_count", 0)) < 2
        or int(worker.get("peak_selected_device_vram_bytes", -1))
        > VRAM_CEILING_BYTES
    ):
        raise FlatQualificationError(
            f"worker {worker.get('scene_index')} process/VRAM gate failed"
        )


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
        raise FlatQualificationError("kernel reset audit was unavailable")
    pattern = re.compile(
        r"(?:amdgpu.*(?:ring .* timeout|ring .* reset|device wedged|GPU reset)"
        r"|(?:hsa|kfd).*(?:exception|queue.*error|memory fault))",
        re.IGNORECASE,
    )
    events = [line for line in completed.stdout.splitlines() if pattern.search(line)]
    return {
        "query_succeeded": True,
        "new_amdgpu_reset_wedge_or_hsa_exception_count": len(events),
        "matching_lines_sha256": hashlib.sha256(
            "\n".join(events).encode()
        ).hexdigest(),
    }


def _worker_argv(
    *,
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_index: int,
    nonce: str,
) -> list[str]:
    return [
        str(plan_builder.ROCM_PYTHON.absolute()),
        str(Path(__file__).resolve()),
        "--worker-scene-index",
        str(scene_index),
        "--plan",
        str(plan_binding["path"]),
        "--expected-plan-sha256",
        str(plan_binding["sha256"]),
        "--expected-plan-byte-count",
        str(plan_binding["byte_count"]),
        "--reservation",
        str(reservation_binding["path"]),
        "--expected-reservation-sha256",
        str(reservation_binding["sha256"]),
        "--expected-reservation-byte-count",
        str(reservation_binding["byte_count"]),
        "--orchestrator-pid",
        str(os.getpid()),
        "--orchestrator-nonce",
        nonce,
    ]


def _read_scene_result_after_release(
    *,
    scene: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (
        plan_builder.QUALIFICATION_OUTPUT_ROOT
        / "scene_results"
        / f"{int(scene['scene_index']):03d}.json"
    )
    binding = _standard_binding(pilot.file_binding(path))
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label=f"flat qualification scene {scene['scene_index']} result",
    )
    expected_counts = scene_core._scene_expected_counts_v2(  # noqa: SLF001
        str(scene["role"])
    )
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    ordered_ids = [str(row["state_id"]) for row in scene["states"]]
    state_bindings = value.get("state_receipt_bindings")
    render_binding = value.get("render_receipt_binding")
    metric = value.get("scene_metric")
    mesh_cache = value.get("scene_local_mesh_cache")
    initialization = value.get("genesis_initialization")
    if (
        _standard_binding(actual) != binding
        or value.get("schema") != SCENE_RESULT_SCHEMA
        or value.get("status") != SCENE_PASS_STATUS
        or value.get("attempt_id") != plan_builder.QUALIFICATION_ATTEMPT_ID
        or value.get("scene_index") != scene["scene_index"]
        or value.get("role") != role
        or value.get("scene_id") != scene_id
        or value.get("worker_pid") != worker_receipt.get("pid")
        or value.get("orchestrator_pid") != os.getpid()
        or value.get("fresh_process") is not True
        or value.get("execution_seed") != plan["execution_contract"]["seed"]
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("reservation_binding") != dict(reservation_binding)
        or value.get("expected_counts") != expected_counts
        or value.get("observed_counts") != expected_counts
        or value.get("ordered_state_ids") != ordered_ids
        or not isinstance(state_bindings, list)
        or len(state_bindings) != 4
        or not isinstance(render_binding, Mapping)
        or not isinstance(metric, Mapping)
        or metric.get("role") != role
        or metric.get("scene_id") != scene_id
        or metric.get("states") != 4
        or not isinstance(mesh_cache, Mapping)
        or not isinstance(initialization, Mapping)
        or initialization.get("source")
        != "full_plan_first_scene_bound_manifest"
        or initialization.get("full_physics_seed")
        != rocm_core.PLAN_FIRST_PHYSICS_SEED
        or initialization.get("effective_genesis_seed")
        != rocm_core.PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or initialization.get("backend") != "amdgpu"
        or initialization.get("backend_api") != "gs.amdgpu"
        or initialization.get("hsa_override_gfx_version_present") is not False
        or value.get("local_runtime_audit", {}).get(
            "original_renderer_globals_unchanged"
        )
        is not True
        or value.get("local_runtime_audit", {}).get(
            "renderer_code_identity_preserved"
        )
        is not True
        or value.get("local_runtime_audit", {}).get("renderer_globals_cloned")
        is not True
        or value.get("local_runtime_audit", {}).get(
            "rollout_config_worker_local"
        )
        is not True
        or value.get("local_runtime_audit", {}).get("foot_contact_source")
        != "zero"
        or value.get("probe_output_scientific_reuse_authorized") is not False
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
        or value.get("allows_adaptive_batching") is not False
        or value.get("failure") is not None
        or not _all_numbers_finite(value)
    ):
        raise FlatQualificationError(
            f"scene {scene['scene_index']} result gate failed"
        )
    for declared, state_id, raw_binding in zip(
        scene["states"], ordered_ids, state_bindings, strict=True
    ):
        checked = scene_core._rehash_relative_binding_v2(  # noqa: SLF001
            raw_binding,
            collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
            label="flat qualification state receipt",
        )
        expected_path = PurePosixPath(
            "scenes", role, scene_id, "state_receipts", f"{state_id}.json"
        )
        if PurePosixPath(str(checked["path"])) != expected_path:
            raise FlatQualificationError("state receipt order/path changed")
        receipt_path = plan_builder.QUALIFICATION_OUTPUT_ROOT.joinpath(
            *expected_path.parts
        )
        receipt, reopened = pilot.read_bound_json(
            receipt_path,
            expected_sha256=str(checked["file_sha256"]),
            expected_byte_count=int(checked["byte_count"]),
            label=f"flat qualification state receipt {state_id}",
        )
        state = receipt.get("state", {})
        if (
            reopened.get("file_sha256") != checked["file_sha256"]
            or reopened.get("byte_count") != checked["byte_count"]
            or receipt.get("status") != "PHYSICS_COMPLETE"
            or state.get("state_id") != state_id
            or state.get("role") != role
            or state.get("scene_id") != scene_id
            or state.get("family") != declared.get("family")
            or state.get("group_index") != declared.get("group_index")
            or receipt.get("render_receipt_binding") != render_binding
            or len(receipt.get("branches", [])) != 9
        ):
            raise FlatQualificationError("state receipt content changed")
    checked_render = scene_core._rehash_relative_binding_v2(  # noqa: SLF001
        render_binding,
        collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
        label="flat qualification render receipt",
    )
    expected_render_path = PurePosixPath(
        "scenes", role, scene_id, "live_render_receipt.json"
    )
    if PurePosixPath(str(checked_render["path"])) != expected_render_path:
        raise FlatQualificationError("render receipt path changed")
    render, reopened_render = pilot.read_bound_json(
        plan_builder.QUALIFICATION_OUTPUT_ROOT.joinpath(
            *expected_render_path.parts
        ),
        expected_sha256=str(checked_render["file_sha256"]),
        expected_byte_count=int(checked_render["byte_count"]),
        label="flat qualification render receipt",
    )
    if (
        reopened_render.get("file_sha256") != checked_render["file_sha256"]
        or reopened_render.get("byte_count") != checked_render["byte_count"]
        or render.get("status") != "RENDER_COMPLETE"
        or render.get("scene", {}).get("role") != role
        or render.get("scene", {}).get("scene_id") != scene_id
        or len(render.get("frame_receipts", [])) != 48
        or any(
            not isinstance(frame, Mapping)
            or type(frame.get("byte_count")) is not int
            or int(frame["byte_count"]) <= 0
            for frame in render.get("frame_receipts", [])
        )
    ):
        raise FlatQualificationError("render receipt content changed")
    observed_mesh = scene_core._validate_scene_local_mesh_bindings_v2(  # noqa: SLF001
        metric,
        cache_root=(
            plan_builder.QUALIFICATION_OUTPUT_ROOT
            / "scenes"
            / role
            / scene_id
            / "derived_meshes"
        ),
        collection_root=plan_builder.QUALIFICATION_OUTPUT_ROOT,
    )
    if observed_mesh != dict(mesh_cache):
        raise FlatQualificationError("scene-local mesh cache changed")
    relative_result = _relative_binding(
        actual, output_root=plan_builder.QUALIFICATION_OUTPUT_ROOT
    )
    if PurePosixPath(str(relative_result["path"])) != PurePosixPath(
        "scene_results", f"{int(scene['scene_index']):03d}.json"
    ):
        raise FlatQualificationError("scene result index path changed")
    return value, relative_result


def execute_qualification(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    child_env = _validate_child_environment(plan)
    snapshot = _shared_state_snapshot()
    nonce = secrets.token_hex(32)
    collection_root, reservation_binding = _reserve_qualification(
        plan_binding=plan_binding, nonce=nonce
    )
    _read_reservation(
        binding=reservation_binding,
        plan_binding=plan_binding,
        nonce=nonce,
        expected_orchestrator_pid=os.getpid(),
    )
    started_epoch = time.time()
    preflight = historical_preflight._run_rocm_egl_preflight(  # noqa: SLF001
        plan, child_env=child_env
    )
    _require_shared_state_unchanged(snapshot)
    used_path, _total_path, vendor, device = (
        rocm_core._selected_gpu_memory_files_rocm(plan)  # noqa: SLF001
    )
    if scene_core._read_vram_counter_v2(used_path) > VRAM_CEILING_BYTES:  # noqa: SLF001
        raise FlatQualificationError("selected-device VRAM exceeds cap")
    probes: list[dict[str, Any]] = []
    for scene_index in PROBE_ORDER:
        scene = scenes[scene_index]
        worker = _run_worker_with_watchdog(
            _worker_argv(
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                scene_index=scene_index,
                nonce=nonce,
            ),
            scene=scene,
            child_env=child_env,
            used_path=used_path,
        )
        release = scene_core._wait_for_vram_release_v2(  # noqa: SLF001
            used_path,
            baseline_used_bytes=int(worker["prelaunch_baseline_used_bytes"]),
            ceiling_bytes=VRAM_CEILING_BYTES,
        )
        _validate_worker_receipt(worker)
        scene_result, scene_result_binding = _read_scene_result_after_release(
            scene=scene,
            plan=plan,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            worker_receipt=worker,
        )
        probes.append(
            {
                "scene_index": scene_index,
                "role": scene["role"],
                "scene_id": scene["scene_id"],
                "worker": worker,
                "release_barrier": release,
                "scene_result_binding": scene_result_binding,
                "observed_counts": scene_result["observed_counts"],
                "existing_scene_validation_passed": True,
                "probe_output_scientific_reuse_authorized": False,
            }
        )
    kernel_audit = _kernel_events_since(started_epoch)
    maximum_worker_seconds = max(
        float(probe["worker"]["elapsed_seconds"]) for probe in probes
    )
    projected_total_seconds = (
        64.0 * maximum_worker_seconds + NONCOLLECTION_RESERVE_SECONDS
    )
    timing_gate = {
        "maximum_worker_elapsed_seconds": maximum_worker_seconds,
        "scene_count": 64,
        "noncollection_reserve_seconds": NONCOLLECTION_RESERVE_SECONDS,
        "projected_total_seconds": projected_total_seconds,
        "wall_ceiling_seconds": float(scene_core.EXPECTED_CAPS["wall_seconds"]),
        "passed": projected_total_seconds
        <= float(scene_core.EXPECTED_CAPS["wall_seconds"]),
    }
    passed = (
        [probe["scene_index"] for probe in probes] == list(PROBE_ORDER)
        and all(
            probe["existing_scene_validation_passed"] is True
            for probe in probes
        )
        and timing_gate["passed"] is True
        and kernel_audit[
            "new_amdgpu_reset_wedge_or_hsa_exception_count"
        ]
        == 0
        and not plan_builder.DEFAULT_ATTEMPT_ROOT.exists()
        and not plan_builder.DEFAULT_ATTEMPT_ROOT.is_symlink()
    )
    if not passed:
        raise FlatQualificationError("flat qualification terminal gate failed")
    result = {
        "schema": QUALIFICATION_SCHEMA,
        "status": PASS_STATUS,
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "backend": "amdgpu",
        "backend_api": "gs.amdgpu",
        "qualification_contract": copy.deepcopy(QUALIFICATION_CONTRACT),
        "preregistration_binding": _preregistration_binding(),
        "plan_binding": dict(plan_binding),
        "reservation_binding": dict(reservation_binding),
        "rocm_egl_preflight": preflight,
        "selected_device": {"vendor_id": vendor, "device_id": device},
        "probe_order": list(PROBE_ORDER),
        "probes": probes,
        "kernel_reset_audit": kernel_audit,
        "timing_gate": timing_gate,
        "contact_force_route_audit": copy.deepcopy(
            rocm_core.CONTACT_FORCE_ROUTE_AUDIT
        ),
        "all_existing_scene_gates_passed": True,
        "exact_v03_renderer_compatibility_passed": True,
        "scientific_attempt_root_absent": True,
        "probe_output_scientific_reuse_authorized": False,
        "authorizes_scientific_plan_release": False,
        "authorizes_retry_or_resume": False,
    }
    result_binding = _standard_binding(
        pilot.write_json_exclusive(
            plan_builder.QUALIFICATION_ATTEMPT_ROOT / "qualification_result.json",
            result,
        )
    )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "PASS_QUALIFICATION_GATES_CLOSED_NONAUTHORIZING",
        "result_binding": result_binding,
        "decision_binding": None,
        "decision_must_be_emitted_last_and_bind_this_terminal": True,
        "authorizes_scientific_plan_release": False,
        "authorizes_retry_or_resume": False,
        "failure": None,
    }
    terminal_binding = _standard_binding(
        pilot.write_json_exclusive(
            plan_builder.QUALIFICATION_ATTEMPT_ROOT / "terminal.json", terminal
        )
    )
    decision = {
        "schema": DECISION_SCHEMA,
        "status": PASS_STATUS,
        "attempt_id": plan_builder.QUALIFICATION_ATTEMPT_ID,
        "plan_binding": dict(plan_binding),
        "qualification_result_binding": result_binding,
        "qualification_terminal_binding": terminal_binding,
        "probe_order": list(PROBE_ORDER),
        "all_scene_gates_passed": True,
        "timing_gate_passed": True,
        "kernel_gate_passed": True,
        "vram_and_release_gates_passed": True,
        "scientific_attempt_root_absent": True,
        "qualification_payload_reuse_authorized": False,
        "authorizes_scientific_plan_release": True,
        "authorizes_retry_or_resume": False,
    }
    decision_binding = _standard_binding(
        pilot.write_json_exclusive(
            plan_builder.QUALIFICATION_ATTEMPT_ROOT
            / "qualification_decision.json",
            decision,
        )
    )
    return {
        **result,
        "result_binding": result_binding,
        "decision_binding": decision_binding,
        "collection_root": str(collection_root),
    }


def _write_failure_terminal(error: Exception) -> None:
    attempt = plan_builder.QUALIFICATION_ATTEMPT_ROOT
    terminal = attempt / "terminal.json"
    if not attempt.is_dir() or terminal.exists() or terminal.is_symlink():
        return
    try:
        pilot.write_json_exclusive(
            terminal,
            {
                "schema": TERMINAL_SCHEMA,
                "status": FAIL_STATUS,
                "result_binding": None,
                "decision_binding": None,
                "authorizes_scientific_plan_release": False,
                "authorizes_retry_or_resume": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-byte-count", type=int, required=True)
    parser.add_argument("--worker-scene-index", type=int)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--expected-reservation-sha256")
    parser.add_argument("--expected-reservation-byte-count", type=int)
    parser.add_argument("--orchestrator-pid", type=int)
    parser.add_argument("--orchestrator-nonce")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker_scene_index is not None:
            if (
                args.reservation is None
                or args.expected_reservation_sha256 is None
                or args.expected_reservation_byte_count is None
                or args.orchestrator_pid is None
                or args.orchestrator_nonce is None
            ):
                raise FlatQualificationError("worker invocation is incomplete")
            plan, plan_binding, _scenes = _read_worker_plan(
                args.plan,
                expected_sha256=args.expected_plan_sha256,
                expected_byte_count=args.expected_plan_byte_count,
            )
            reservation_binding = {
                "path": str(args.reservation.resolve(strict=True)),
                "sha256": args.expected_reservation_sha256,
                "byte_count": args.expected_reservation_byte_count,
            }
            _collect_scene_worker(
                plan=plan,
                plan_binding=plan_binding,
                reservation_binding=reservation_binding,
                scene_index=args.worker_scene_index,
                orchestrator_pid=args.orchestrator_pid,
                nonce=args.orchestrator_nonce,
            )
            return 0
        plan, plan_binding, scenes = _read_and_validate_plan(
            args.plan,
            expected_sha256=args.expected_plan_sha256,
            expected_byte_count=args.expected_plan_byte_count,
        )
        if any(
            value is not None
            for value in (
                args.reservation,
                args.expected_reservation_sha256,
                args.expected_reservation_byte_count,
                args.orchestrator_pid,
                args.orchestrator_nonce,
            )
        ):
            raise FlatQualificationError("parent received worker-only arguments")
        result = execute_qualification(
            plan=plan, plan_binding=plan_binding, scenes=scenes
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result_binding": result["result_binding"],
                    "decision_binding": result["decision_binding"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        _write_failure_terminal(error)
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DECISION_SCHEMA",
    "EXPECTED_ENVIRONMENT_KEYS",
    "FAIL_STATUS",
    "FlatQualificationError",
    "PASS_STATUS",
    "PROBE_ORDER",
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_SCHEMA",
    "SCENE_RESULT_SCHEMA",
    "TERMINAL_SCHEMA",
    "VRAM_CEILING_BYTES",
    "_clone_worker_runtime",
    "_read_and_validate_plan",
    "_require_worker_environment",
    "_run_worker_with_watchdog",
    "execute_qualification",
]
