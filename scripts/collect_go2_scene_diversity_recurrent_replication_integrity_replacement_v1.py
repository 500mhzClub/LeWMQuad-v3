#!/usr/bin/env python3
"""Collect the V1 replication in two clean, fixed-role worker processes.

The scientific scene kernel is the frozen V1 implementation.  This integrity
replacement changes only its lifetime: one worker collects all 32 train
scenes, exits, and a second worker collects all 32 evaluation scenes.  The
parent writes the standard combined physics result only after both exact role
indexes and the intervening VRAM-release barrier pass.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as frozen  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as kernel  # noqa: E402
from scripts import run_go2_world_model_counterfactual_calibration_authorized_v1 as calibration_supervisor  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "INTEGRITY_REPLACEMENT_V1"
)
AUTHORITY_FIELDS = frozen.AUTHORITY_FIELDS
EXPECTED_COUNTS = frozen.EXPECTED_COUNTS
EXPECTED_CAPS = frozen.EXPECTED_CAPS
EXPECTED_HISTORY_PANEL = frozen.EXPECTED_HISTORY_PANEL
EXPECTED_PERMISSIONS = frozen.EXPECTED_PERMISSIONS

ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-integrity-replacement-v1"
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "collection_reservation_v1"
)
RESERVATION_STATUS = "RESERVED_ONE_SHOT_SPLIT_COLLECTION_CONSUMED"
ROLE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "role_physics_result_v1"
)
ROLE_RESULT_STATUS = "ROLE_PHYSICS_COMPLETE"
SPLIT_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_"
    "split_process_evidence_v1"
)
SPLIT_EVIDENCE_STATUS = "PASS_EXACT_TWO_PROCESS_COLLECTION_AND_JOIN"

ROLE_ORDER = ("train", "eval")
ROLE_STATE_COUNT = 128
ROLE_SCENE_COUNT = 32
ROLE_BRANCH_COUNT = 1152
ROLE_CONTEXT_FRAME_COUNT = 384
ROLE_STORED_FRAME_COUNT = 1536
PLAN_FIRST_PHYSICS_SEED = 14_102_849_992_353_107_924
PLAN_FIRST_EFFECTIVE_GENESIS_SEED = 315_871_188
VRAM_RELEASE_MARGIN_BYTES = 512 * 1024 * 1024
VRAM_RELEASE_TIMEOUT_SECONDS = 60.0
VRAM_RELEASE_POLL_SECONDS = 0.05
VRAM_RELEASE_CONSECUTIVE_SAMPLES = 3


class SplitCollectionError(RuntimeError):
    """Raised when the fixed split-process collection contract changes."""


_validate_scene_diversity_plan_v1 = frozen._validate_scene_diversity_plan_v1


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if "file_sha256" in value:
        return frozen._standard_binding(value)  # noqa: SLF001
    return {
        "path": str(value["path"]),
        "sha256": str(value["sha256"]),
        "byte_count": int(value["byte_count"]),
    }


def _role_expected_counts_v1(role: str) -> dict[str, Any]:
    if role not in ROLE_ORDER:
        raise SplitCollectionError("collector role changed")
    return {
        "scenes": ROLE_SCENE_COUNT,
        "states": ROLE_STATE_COUNT,
        "roles": {role: ROLE_STATE_COUNT},
        "actions": 9,
        "candidate_branches": ROLE_BRANCH_COUNT,
        "sentinel_branches": 0,
        "total_branches": ROLE_BRANCH_COUNT,
        "context_frames": ROLE_CONTEXT_FRAME_COUNT,
        "target_frames": ROLE_BRANCH_COUNT,
    }


def _validate_output_roots_v1(
    *, authority: Mapping[str, Any], plan: Mapping[str, Any], reserved: bool
) -> tuple[Path, Path]:
    attempt = Path(os.path.abspath(str(authority.get("attempt_root", ""))))
    collection = Path(os.path.abspath(str(authority.get("collection_root", ""))))
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    frozen._reject_protected_path(attempt, label="replacement attempt root")  # noqa: SLF001
    frozen._reject_protected_path(collection, label="replacement collection root")  # noqa: SLF001
    try:
        attempt.relative_to(development)
        collection.relative_to(attempt)
    except ValueError as exc:
        raise SplitCollectionError("replacement roots escape development custody") from exc
    if (
        collection.parent != attempt
        or str(collection) != str(plan.get("output_root"))
        or not attempt.is_dir()
        or attempt.is_symlink()
    ):
        raise SplitCollectionError("replacement attempt/collection roots changed")
    if reserved:
        if not collection.is_dir() or collection.is_symlink():
            raise SplitCollectionError("replacement collection root is not reserved")
    elif collection.exists() or collection.is_symlink():
        raise SplitCollectionError("replacement collection root is not fresh")
    return attempt.resolve(strict=True), (
        collection.resolve(strict=True) if reserved else collection
    )


def _validate_authority_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reserved: bool,
) -> dict[str, Any]:
    if (
        not isinstance(authority, Mapping)
        or set(authority) != AUTHORITY_FIELDS
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("attempt_id") != ATTEMPT_ID
        or authority.get("attempt_id") != plan.get("attempt_id")
        or authority.get("plan_binding") != _standard_binding(plan_binding)
        or authority.get("config") != benchmark.config_v1()
        or authority.get("caps") != EXPECTED_CAPS
        or authority.get("permissions") != EXPECTED_PERMISSIONS
        or not isinstance(authority.get("preregistration_binding"), Mapping)
        or not isinstance(authority.get("source_review_binding"), Mapping)
        or not isinstance(authority.get("source_bindings"), Mapping)
        or not isinstance(authority.get("dino"), Mapping)
        or not isinstance(authority_binding, Mapping)
    ):
        raise SplitCollectionError("replacement execution authority changed")
    _validate_output_roots_v1(authority=authority, plan=plan, reserved=reserved)
    return dict(authority)


def load_and_validate_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
    _collection_reserved: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen._reject_protected_path(plan_path, label="replacement exact plan")  # noqa: SLF001
    frozen._reject_protected_path(authority_path, label="replacement authority")  # noqa: SLF001
    raw_plan, plan_binding = pilot.read_bound_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="replacement exact plan",
    )
    plan = _validate_scene_diversity_plan_v1(
        copy.deepcopy(pilot.validate_plan(raw_plan))
    )
    bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    if (
        plan.get("attempt_id") != ATTEMPT_ID
        or [row.get("role") for row in plan["states"][:ROLE_STATE_COUNT]]
        != ["train"] * ROLE_STATE_COUNT
        or [row.get("role") for row in plan["states"][ROLE_STATE_COUNT:]]
        != ["eval"] * ROLE_STATE_COUNT
        or [row.get("group_index") for row in plan["states"]]
        != list(range(EXPECTED_COUNTS["states"]))
    ):
        raise SplitCollectionError("replacement plan role order or identity changed")
    raw_authority, historical_authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="replacement execution authority",
    )
    authority_binding = _standard_binding(historical_authority_binding)
    authority = _validate_authority_v1(
        raw_authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        reserved=_collection_reserved,
    )
    return authority, authority_binding, plan, plan_binding


load_and_validate_replacement_v1 = load_and_validate_v1


def _create_collection_root_v1(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    orchestrator_nonce: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    _attempt, collection = _validate_output_roots_v1(
        authority=authority, plan=plan, reserved=False
    )
    try:
        os.mkdir(collection, mode=0o700)
        os.mkdir(collection / "scenes", mode=0o700)
        os.mkdir(collection / "role_results", mode=0o700)
    except OSError as exc:
        raise SplitCollectionError("could not reserve split collection root") from exc
    collection = collection.resolve(strict=True)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": RESERVATION_STATUS,
        "attempt_id": ATTEMPT_ID,
        "attempt_root": str(authority["attempt_root"]),
        "collection_root": str(collection),
        "plan_binding": _standard_binding(plan_binding),
        "authority_binding": dict(authority_binding),
        "orchestrator_nonce": orchestrator_nonce,
        "orchestrator_pid": os.getpid(),
        "fixed_role_order": list(ROLE_ORDER),
        "root_creation_consumes_attempt": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
        "partial_artifact_reuse_authorized": False,
    }
    binding = pilot.write_json_exclusive(collection / "reservation.json", reservation)
    return collection, reservation, kernel._relative_output_binding(  # noqa: SLF001
        binding, output_root=collection
    )


def _read_collection_reservation_v1(
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
        label="split collection reservation",
    )
    if (
        value.get("schema") != RESERVATION_SCHEMA
        or value.get("status") != RESERVATION_STATUS
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("plan_binding") != _standard_binding(plan_binding)
        or value.get("orchestrator_nonce") != orchestrator_nonce
        or value.get("orchestrator_pid") != orchestrator_pid
        or value.get("fixed_role_order") != list(ROLE_ORDER)
        or any(
            value.get(name) is not False
            for name in (
                "retry_authorized",
                "resume_authorized",
                "overwrite_authorized",
                "refill_authorized",
                "partial_artifact_reuse_authorized",
            )
        )
    ):
        raise SplitCollectionError("split collection reservation changed")
    return value, binding


def _role_states_v1(plan: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    start = 0 if role == "train" else ROLE_STATE_COUNT
    states = list(plan["states"][start : start + ROLE_STATE_COUNT])
    if len(states) != ROLE_STATE_COUNT or any(row.get("role") != role for row in states):
        raise SplitCollectionError(f"{role} fixed state slice changed")
    return states


def _initialize_genesis_v1(*, backend: str, seed: int) -> None:
    from lewm_genesis.scene_builder import initialize_genesis

    initialize_genesis(backend=backend, seed=seed)


def _install_role_local_mesh_cache_v1(
    runtime: dict[str, Any], *, role_root: Path
) -> Path:
    """Route every deterministic OBJ cache write into this worker's role root."""

    cache_root = role_root / "derived_meshes"
    original_cached_box_obj = runtime.get("cached_box_obj")
    render_builder = runtime.get("build_textured_v03_scene")
    render_globals = getattr(render_builder, "__globals__", None)
    if not callable(original_cached_box_obj) or not isinstance(render_globals, dict):
        raise SplitCollectionError("textured-v03 mesh cache seam changed")

    def role_local_cached_box_obj(
        size_xyz_m: Sequence[float], *, tiles_per_m: float = 0.7
    ) -> str:
        return original_cached_box_obj(
            tuple(float(value) for value in size_xyz_m),
            tiles_per_m=tiles_per_m,
            cache_dir=cache_root,
        )

    runtime["cached_box_obj"] = role_local_cached_box_obj
    render_globals["cached_box_obj"] = role_local_cached_box_obj
    return cache_root


def _validate_role_local_mesh_bindings_v1(
    scene_metrics: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path,
    collection_root: Path,
) -> dict[str, Any]:
    bindings_by_path: dict[str, dict[str, Any]] = {}
    for metric in scene_metrics:
        bindings = metric.get("derived_mesh_bindings")
        if not isinstance(bindings, list):
            raise SplitCollectionError("role-local derived mesh bindings are absent")
        for raw_binding in bindings:
            try:
                binding = pilot.require_binding(
                    raw_binding, label="role-local derived OBJ mesh"
                )
                Path(str(binding["path"])).resolve(strict=True).relative_to(
                    cache_root.resolve(strict=True)
                )
            except (OSError, ValueError, pilot.PilotContractError) as exc:
                raise SplitCollectionError(
                    "derived mesh escaped or changed outside the role cache"
                ) from exc
            bindings_by_path[str(binding["path"])] = binding
    bindings = [bindings_by_path[path] for path in sorted(bindings_by_path)]
    if not cache_root.is_dir() or cache_root.is_symlink() or not bindings:
        raise SplitCollectionError("role-local derived mesh cache is incomplete")
    return {
        "path": cache_root.relative_to(collection_root).as_posix(),
        "cross_role_reuse_authorized": False,
        "mesh_count": len(bindings),
        "bindings_identity_sha256": hashlib.sha256(
            pilot.canonical_json_bytes(bindings)
        ).hexdigest(),
    }


def _initialize_from_plan_first_scene_v1(
    *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    state = plan["states"][0]
    binding = state["scene_manifest_binding"]
    manifest, actual = pilot.read_bound_json(
        Path(str(binding["path"])),
        expected_sha256=str(binding["file_sha256"]),
        expected_byte_count=int(binding["byte_count"]),
        label="plan-first Genesis seed manifest",
    )
    full_seed = manifest.get("physics_seed")
    effective = int(full_seed) & 0x7FFF_FFFF if type(full_seed) is int else None
    backend = str(plan["execution_contract"]["backend"])
    if (
        actual != binding
        or state.get("scene_id") != "large_enclosed_maze_8a6599d5327d"
        or full_seed != PLAN_FIRST_PHYSICS_SEED
        or effective != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or backend != "vulkan"
    ):
        raise SplitCollectionError("plan-first process-global Genesis seed changed")
    _initialize_genesis_v1(backend=backend, seed=int(full_seed))
    return {
        "source": "full_plan_first_scene_bound_manifest",
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "manifest_binding": dict(binding),
        "backend": backend,
        "full_physics_seed": int(full_seed),
        "effective_genesis_seed": int(effective),
    }


def _observed_counts_v1(
    receipts: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
    scene_ids = {str(row["state"]["scene_id"]) for row in receipts}
    candidate = sum(
        branch.get("kind") == "candidate"
        for row in receipts
        for branch in row["branches"]
    )
    sentinel = sum(
        branch.get("kind") == "sentinel"
        for row in receipts
        for branch in row["branches"]
    )
    contexts = sum(len(row["context"]["frame_identities"]) for row in receipts)
    return {
        "scenes": len(scene_ids),
        "states": len(receipts),
        "roles": {role: len(receipts)},
        "actions": 9,
        "candidate_branches": candidate,
        "sentinel_branches": sentinel,
        "total_branches": candidate + sentinel,
        "context_frames": contexts,
        "target_frames": candidate + sentinel,
    }


def _collect_role_v1(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    role: str,
    orchestrator_pid: int,
) -> tuple[dict[str, Any], Path]:
    if role not in ROLE_ORDER or orchestrator_pid <= 1 or os.getppid() != orchestrator_pid:
        raise SplitCollectionError("role worker ownership changed")
    collection_root = Path(str(authority["collection_root"])).resolve(strict=True)
    role_root = collection_root / "scenes" / role
    result_path = collection_root / "role_results" / f"{role}.json"
    if role_root.exists() or role_root.is_symlink() or result_path.exists():
        raise SplitCollectionError(f"{role} worker root is not fresh")
    os.mkdir(role_root, mode=0o700)

    states = _role_states_v1(plan, role)
    state_bindings: dict[str, dict[str, Any]] = {}
    receipts_by_id: dict[str, dict[str, Any]] = {}
    render_bindings: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    stored_rgb_bytes = 0
    started = time.perf_counter()

    pilot.require_plan_bindings(plan)
    kernel._validate_python_runtime(plan)  # noqa: SLF001
    kernel._validate_execution_environment(plan)  # noqa: SLF001
    bounded._validate_bound_scenes(plan)  # noqa: SLF001
    runtime_versions = kernel._capture_runtime_versions()  # noqa: SLF001
    runtime = kernel._runtime_imports(textured_v03=True)  # noqa: SLF001
    mesh_cache_root = _install_role_local_mesh_cache_v1(runtime, role_root=role_root)
    platform = runtime["load_platform_manifest"](
        plan["runtime_bindings"]["platform_manifest"]["path"]
    )
    resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), REPO_ROOT)
    if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
        raise SplitCollectionError("platform resolves a different Go2 URDF")
    registry = runtime["PrimitiveRegistry"].from_yaml(
        plan["runtime_bindings"]["primitive_registry"]["path"]
    )
    action_blocks = kernel._load_action_blocks(  # noqa: SLF001
        plan=plan, registry=registry, expand=runtime["expand_primitive_to_block"]
    )
    genesis_initialization = _initialize_from_plan_first_scene_v1(plan=plan)

    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        by_scene[str(state["scene_id"])].append(state)
    if len(by_scene) != ROLE_SCENE_COUNT:
        raise SplitCollectionError(f"{role} scene count changed")

    for scene_id, scene_states in by_scene.items():
        if len(scene_states) != 4:
            raise SplitCollectionError("role worker scene state count changed")
        receipts, frames, quality, sentinels, metrics = kernel._collect_scene(  # noqa: SLF001
            plan=plan,
            states=scene_states,
            runtime=runtime,
            platform=platform,
            registry=registry,
            action_blocks=action_blocks,
        )
        if [str(row["state"]["state_id"]) for row in receipts] != [
            str(row["state_id"]) for row in scene_states
        ]:
            raise SplitCollectionError("role worker changed planned state order")
        stored_rgb_bytes += sum(int(frame["byte_count"]) for frame in frames)
        if stored_rgb_bytes > EXPECTED_CAPS["stored_rgb_byte_ceiling"]:
            raise SplitCollectionError("role worker stored RGB ceiling exceeded")
        render_receipt = {
            "schema": pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA,
            "attempt_id": str(plan["attempt_id"]),
            "status": "RENDER_COMPLETE",
            "physics_validated": False,
            "citable_as_scientific_evidence": False,
            "scene": {
                "role": role,
                "scene_id": scene_id,
                "family": str(scene_states[0]["family"]),
                "scene_manifest_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                    scene_states[0],
                    binding_name="scene_manifest_binding",
                    output_root=collection_root,
                ),
                "scene_genesis_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                    scene_states[0],
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
            role_root / scene_id / "live_render_receipt.json", render_receipt
        )
        relative_render = kernel._relative_output_binding(  # noqa: SLF001
            render_binding, output_root=collection_root
        )
        render_bindings.append(relative_render)
        for receipt in receipts:
            state_id = str(receipt["state"]["state_id"])
            receipt["render_receipt_binding"] = relative_render
            binding = pilot.write_json_exclusive(
                role_root / scene_id / "state_receipts" / f"{state_id}.json",
                receipt,
            )
            state_bindings[state_id] = kernel._relative_output_binding(  # noqa: SLF001
                binding, output_root=collection_root
            )
            receipts_by_id[state_id] = receipt
        scene_metrics.append(metrics)
        if time.perf_counter() - started > EXPECTED_CAPS["wall_seconds"]:
            raise SplitCollectionError("role worker exceeded global wall cap")

    ordered_ids = [str(row["state_id"]) for row in states]
    if set(state_bindings) != set(ordered_ids):
        raise SplitCollectionError("role worker state receipt closure changed")
    ordered_receipts = [receipts_by_id[state_id] for state_id in ordered_ids]
    observed = _observed_counts_v1(ordered_receipts, role=role)
    expected = _role_expected_counts_v1(role)
    if observed != expected:
        raise SplitCollectionError("role worker observed counts changed")
    for metric_name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if sum(int(row[metric_name]) for row in scene_metrics) != ROLE_STORED_FRAME_COUNT:
            raise SplitCollectionError(f"role worker {metric_name} changed")
    mesh_cache = _validate_role_local_mesh_bindings_v1(
        scene_metrics,
        cache_root=mesh_cache_root,
        collection_root=collection_root,
    )

    result = {
        "schema": ROLE_RESULT_SCHEMA,
        "status": ROLE_RESULT_STATUS,
        "attempt_id": ATTEMPT_ID,
        "role": role,
        "role_index": ROLE_ORDER.index(role),
        "worker_pid": os.getpid(),
        "orchestrator_pid": orchestrator_pid,
        "sys_executable": str(Path(sys.executable).resolve(strict=True)),
        "fresh_process": True,
        "execution_seed": int(plan["execution_contract"]["seed"]),
        "genesis_initialization": genesis_initialization,
        "role_local_mesh_cache": mesh_cache,
        "plan_binding": dict(plan_binding),
        "authority_binding": dict(authority_binding),
        "collection_reservation_binding": dict(reservation_binding),
        "caps": dict(authority["caps"]),
        "runtime_versions": runtime_versions,
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": dict(authority["source_bindings"]),
        "expected_counts": expected,
        "observed_counts": observed,
        "ordered_state_ids": ordered_ids,
        "state_receipt_bindings": [state_bindings[state_id] for state_id in ordered_ids],
        "render_receipt_bindings": render_bindings,
        "scene_metrics": scene_metrics,
        "stored_rgb_bytes": stored_rgb_bytes,
        "collection_wall_seconds": time.perf_counter() - started,
        "failure": None,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
    }
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


def _read_vram_counter_v1(path: Path) -> int:
    selected = Path(path)
    if selected.is_symlink():
        raise SplitCollectionError("selected-device VRAM counter is a symlink")
    try:
        raw = selected.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SplitCollectionError("selected-device VRAM counter read failed") from exc
    if not raw.isdigit():
        raise SplitCollectionError("selected-device VRAM counter is malformed")
    return int(raw)


def _wait_for_vram_release_v1(
    used_path: Path, *, baseline_used_bytes: int, ceiling_bytes: int
) -> dict[str, Any]:
    release_ceiling = baseline_used_bytes + VRAM_RELEASE_MARGIN_BYTES
    if baseline_used_bytes < 0 or release_ceiling >= ceiling_bytes:
        raise SplitCollectionError("VRAM release baseline leaves no bounded margin")
    started = time.monotonic()
    samples = 0
    consecutive = 0
    minimum: int | None = None
    maximum: int | None = None
    final = baseline_used_bytes
    while time.monotonic() - started <= VRAM_RELEASE_TIMEOUT_SECONDS:
        final = _read_vram_counter_v1(used_path)
        samples += 1
        minimum = final if minimum is None else min(minimum, final)
        maximum = final if maximum is None else max(maximum, final)
        consecutive = consecutive + 1 if final <= release_ceiling else 0
        if consecutive >= VRAM_RELEASE_CONSECUTIVE_SAMPLES:
            return {
                "status": "PASSED",
                "read_only": True,
                "counter_path": str(used_path.resolve(strict=True)),
                "baseline_used_bytes": baseline_used_bytes,
                "release_margin_bytes": VRAM_RELEASE_MARGIN_BYTES,
                "release_ceiling_bytes": release_ceiling,
                "absolute_vram_ceiling_bytes": ceiling_bytes,
                "required_consecutive_samples": VRAM_RELEASE_CONSECUTIVE_SAMPLES,
                "sample_interval_seconds": VRAM_RELEASE_POLL_SECONDS,
                "sample_count": samples,
                "minimum_used_bytes": minimum,
                "maximum_used_bytes": maximum,
                "final_used_bytes": final,
                "final_consecutive_samples": consecutive,
                "elapsed_seconds": time.monotonic() - started,
            }
        time.sleep(VRAM_RELEASE_POLL_SECONDS)
    raise SplitCollectionError("VRAM did not return through the release barrier")


def _worker_argv_v1(
    *,
    role: str,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
    reservation_binding: Mapping[str, Any],
    orchestrator_nonce: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-role",
        role,
        "--plan",
        str(plan_path),
        "--expected-plan-byte-count",
        str(expected_plan_byte_count),
        "--expected-plan-sha256",
        expected_plan_sha256,
        "--authority",
        str(authority_path),
        "--expected-authority-byte-count",
        str(expected_authority_byte_count),
        "--expected-authority-sha256",
        expected_authority_sha256,
        "--expected-reservation-byte-count",
        str(reservation_binding["byte_count"]),
        "--expected-reservation-sha256",
        str(reservation_binding["file_sha256"]),
        "--orchestrator-nonce",
        orchestrator_nonce,
        "--orchestrator-pid",
        str(os.getpid()),
    ]


def _run_worker_process_v1(argv: Sequence[str], *, role: str) -> dict[str, Any]:
    if list(argv[:1]) != [sys.executable] or "--worker-role" not in argv:
        raise SplitCollectionError("role worker invocation changed")
    started = time.monotonic()
    # Deliberately inherit the outer collector's process group: the frozen
    # supervisor's killpg must cover this worker on a wall/VRAM breach.
    process = subprocess.Popen(list(argv), cwd=REPO_ROOT)
    returncode = process.wait()
    receipt = {
        "role": role,
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "argv": list(argv),
        "sys_executable": str(Path(sys.executable).resolve(strict=True)),
        "fresh_process": True,
        "inherited_outer_process_group": True,
        "exit_code": int(returncode),
        "elapsed_seconds": time.monotonic() - started,
    }
    if returncode != 0:
        raise SplitCollectionError(f"{role} worker exited with status {returncode}")
    return receipt


def _binding_shape_v1(value: object, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "file_sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or PurePosixPath(str(value["path"])).is_absolute()
        or ".." in PurePosixPath(str(value["path"])).parts
        or not isinstance(value.get("file_sha256"), str)
        or len(str(value["file_sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise SplitCollectionError(f"{label} binding changed")
    return dict(value)


def _rehash_relative_binding_v1(
    value: object, *, collection_root: Path, label: str
) -> dict[str, Any]:
    binding = _binding_shape_v1(value, label=label)
    relative = PurePosixPath(str(binding["path"]))
    selected = collection_root.joinpath(*relative.parts)
    try:
        selected.resolve(strict=True).relative_to(collection_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SplitCollectionError(f"{label} escapes or is absent") from exc
    if selected.is_symlink() or not selected.is_file():
        raise SplitCollectionError(f"{label} is not a regular no-follow file")
    actual = pilot.file_binding(selected)
    if (
        actual["file_sha256"] != binding["file_sha256"]
        or actual["byte_count"] != binding["byte_count"]
    ):
        raise SplitCollectionError(f"{label} content binding changed")
    return binding


def _load_role_result_v1(
    *,
    collection_root: Path,
    role: str,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = collection_root / "role_results" / f"{role}.json"
    absolute = pilot.file_binding(path)
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(absolute["file_sha256"]),
        expected_byte_count=int(absolute["byte_count"]),
        label=f"{role} role result",
    )
    expected_states = _role_states_v1(plan, role)
    ordered_ids = [str(row["state_id"]) for row in expected_states]
    state_bindings = value.get("state_receipt_bindings")
    render_bindings = value.get("render_receipt_bindings")
    metrics = value.get("scene_metrics")
    mesh_cache = value.get("role_local_mesh_cache")
    if (
        actual != absolute
        or value.get("schema") != ROLE_RESULT_SCHEMA
        or value.get("status") != ROLE_RESULT_STATUS
        or value.get("attempt_id") != ATTEMPT_ID
        or value.get("role") != role
        or value.get("role_index") != ROLE_ORDER.index(role)
        or value.get("worker_pid") != worker_receipt.get("pid")
        or value.get("orchestrator_pid") != os.getpid()
        or value.get("fresh_process") is not True
        or value.get("execution_seed") != plan["execution_contract"]["seed"]
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("collection_reservation_binding") != dict(reservation_binding)
        or value.get("caps") != authority["caps"]
        or value.get("runtime_bindings") != plan["runtime_bindings"]
        or value.get("source_bindings") != authority["source_bindings"]
        or value.get("expected_counts") != _role_expected_counts_v1(role)
        or value.get("observed_counts") != _role_expected_counts_v1(role)
        or value.get("ordered_state_ids") != ordered_ids
        or not isinstance(state_bindings, list)
        or len(state_bindings) != ROLE_STATE_COUNT
        or not isinstance(render_bindings, list)
        or len(render_bindings) != ROLE_SCENE_COUNT
        or not isinstance(metrics, list)
        or len(metrics) != ROLE_SCENE_COUNT
        or not isinstance(mesh_cache, Mapping)
        or value.get("failure") is not None
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
    ):
        raise SplitCollectionError(f"{role} role result contract changed")
    initialization = value.get("genesis_initialization")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("full_physics_seed") != PLAN_FIRST_PHYSICS_SEED
        or initialization.get("effective_genesis_seed")
        != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or initialization.get("source") != "full_plan_first_scene_bound_manifest"
    ):
        raise SplitCollectionError(f"{role} Genesis initialization changed")
    expected_scenes: list[str] = []
    for state in expected_states:
        if str(state["scene_id"]) not in expected_scenes:
            expected_scenes.append(str(state["scene_id"]))
    for state_id, state, raw_binding in zip(ordered_ids, expected_states, state_bindings, strict=True):
        binding = _rehash_relative_binding_v1(
            raw_binding,
            collection_root=collection_root,
            label=f"{role} state receipt",
        )
        expected_path = PurePosixPath(
            "scenes", role, str(state["scene_id"]), "state_receipts", f"{state_id}.json"
        )
        if PurePosixPath(binding["path"]) != expected_path:
            raise SplitCollectionError(f"{role} state receipt order/path changed")
    for scene_id, raw_binding in zip(expected_scenes, render_bindings, strict=True):
        binding = _rehash_relative_binding_v1(
            raw_binding,
            collection_root=collection_root,
            label=f"{role} render receipt",
        )
        expected_path = PurePosixPath(
            "scenes", role, scene_id, "live_render_receipt.json"
        )
        if PurePosixPath(binding["path"]) != expected_path:
            raise SplitCollectionError(f"{role} render receipt order/path changed")
    for scene_id, row in zip(expected_scenes, metrics, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("role") != role
            or row.get("scene_id") != scene_id
            or row.get("states") != 4
        ):
            raise SplitCollectionError(f"{role} scene metric order changed")
    observed_mesh_cache = _validate_role_local_mesh_bindings_v1(
        metrics,
        cache_root=collection_root / "scenes" / role / "derived_meshes",
        collection_root=collection_root,
    )
    if observed_mesh_cache != dict(mesh_cache):
        raise SplitCollectionError(f"{role} role-local mesh cache changed")
    relative = kernel._relative_output_binding(absolute, output_root=collection_root)  # noqa: SLF001
    return value, relative


def _ordered_binding_identity_v1(values: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(pilot.canonical_json_bytes(list(values))).hexdigest()


def _build_split_evidence_v1(
    *,
    plan: Mapping[str, Any],
    role_results: Mapping[str, Mapping[str, Any]],
    role_result_bindings: Mapping[str, Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    release_barrier: Mapping[str, Any],
) -> dict[str, Any]:
    workers = []
    for role, receipt in zip(ROLE_ORDER, worker_receipts, strict=True):
        part = role_results[role]
        workers.append(
            {
                "role": role,
                "fresh_process": True,
                "pid": int(receipt["pid"]),
                "parent_pid": int(receipt["parent_pid"]),
                "sys_executable": str(receipt["sys_executable"]),
                "inherited_outer_process_group": True,
                "exit_code": 0,
                "execution_seed": int(part["execution_seed"]),
                "full_genesis_seed": PLAN_FIRST_PHYSICS_SEED,
                "effective_genesis_seed": PLAN_FIRST_EFFECTIVE_GENESIS_SEED,
                "role_local_mesh_cache": copy.deepcopy(
                    part["role_local_mesh_cache"]
                ),
                "expected_counts": copy.deepcopy(part["expected_counts"]),
                "observed_counts": copy.deepcopy(part["observed_counts"]),
                "role_result_binding": dict(role_result_bindings[role]),
            }
        )
    state_bindings = [
        row
        for role in ROLE_ORDER
        for row in role_results[role]["state_receipt_bindings"]
    ]
    render_bindings = [
        row
        for role in ROLE_ORDER
        for row in role_results[role]["render_receipt_bindings"]
    ]
    return {
        "schema": SPLIT_EVIDENCE_SCHEMA,
        "status": SPLIT_EVIDENCE_STATUS,
        "workers": workers,
        "train_exit_observed_before_eval_launch": True,
        "release_barrier": dict(release_barrier),
        "join": {
            "exact_join": True,
            "ordered_roles": list(ROLE_ORDER),
            "expected_counts": copy.deepcopy(EXPECTED_COUNTS),
            "observed_counts": copy.deepcopy(EXPECTED_COUNTS),
            "state_receipt_count": len(state_bindings),
            "render_receipt_count": len(render_bindings),
            "scene_metric_count": sum(
                len(role_results[role]["scene_metrics"]) for role in ROLE_ORDER
            ),
            "ordered_state_binding_identity_sha256": _ordered_binding_identity_v1(
                state_bindings
            ),
            "ordered_render_binding_identity_sha256": _ordered_binding_identity_v1(
                render_bindings
            ),
            "plan_ordered_state_ids_sha256": hashlib.sha256(
                pilot.canonical_json_bytes(
                    [str(row["state_id"]) for row in plan["states"]]
                )
            ).hexdigest(),
        },
    }


def validate_split_collection_evidence_v1(
    result: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, bool]:
    evidence = result.get("split_process_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema",
        "status",
        "workers",
        "train_exit_observed_before_eval_launch",
        "release_barrier",
        "join",
    }:
        raise SplitCollectionError("split-process evidence fields changed")
    workers = evidence.get("workers")
    worker_fields = {
        "role",
        "fresh_process",
        "pid",
        "parent_pid",
        "sys_executable",
        "inherited_outer_process_group",
        "exit_code",
        "execution_seed",
        "full_genesis_seed",
        "effective_genesis_seed",
        "role_local_mesh_cache",
        "expected_counts",
        "observed_counts",
        "role_result_binding",
    }
    if (
        evidence.get("schema") != SPLIT_EVIDENCE_SCHEMA
        or evidence.get("status") != SPLIT_EVIDENCE_STATUS
        or evidence.get("train_exit_observed_before_eval_launch") is not True
        or not isinstance(workers, list)
        or len(workers) != 2
        or [row.get("role") for row in workers if isinstance(row, Mapping)]
        != list(ROLE_ORDER)
        or [row.get("exit_code") for row in workers if isinstance(row, Mapping)]
        != [0, 0]
        or any(
            not isinstance(row, Mapping)
            or set(row) != worker_fields
            or row.get("fresh_process") is not True
            or row.get("inherited_outer_process_group") is not True
            or type(row.get("pid")) is not int
            or int(row["pid"]) <= 1
            or type(row.get("parent_pid")) is not int
            or int(row["parent_pid"]) <= 1
            or not isinstance(row.get("sys_executable"), str)
            or row["sys_executable"]
            != str(
                Path(plan["execution_contract"]["python_invocation_path"]).resolve(
                    strict=True
                )
            )
            or row.get("execution_seed") != plan["execution_contract"]["seed"]
            or row.get("full_genesis_seed") != PLAN_FIRST_PHYSICS_SEED
            or row.get("effective_genesis_seed")
            != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
            or not isinstance(row.get("role_local_mesh_cache"), Mapping)
            or row["role_local_mesh_cache"].get("path")
            != f"scenes/{row.get('role')}/derived_meshes"
            or row["role_local_mesh_cache"].get("cross_role_reuse_authorized")
            is not False
            or type(row["role_local_mesh_cache"].get("mesh_count")) is not int
            or row["role_local_mesh_cache"]["mesh_count"] <= 0
            or not isinstance(
                row["role_local_mesh_cache"].get("bindings_identity_sha256"), str
            )
            or len(row["role_local_mesh_cache"]["bindings_identity_sha256"]) != 64
            or row.get("expected_counts")
            != _role_expected_counts_v1(str(row.get("role")))
            or row.get("observed_counts")
            != _role_expected_counts_v1(str(row.get("role")))
            for row in workers
        )
        or workers[0].get("pid") == workers[1].get("pid")
        or workers[0].get("parent_pid") != workers[1].get("parent_pid")
    ):
        raise SplitCollectionError("exact two-worker evidence changed")
    barrier = evidence.get("release_barrier")
    barrier_fields = {
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
    if (
        not isinstance(barrier, Mapping)
        or set(barrier) != barrier_fields
        or barrier.get("status") != "PASSED"
        or barrier.get("read_only") is not True
        or barrier.get("release_margin_bytes") != VRAM_RELEASE_MARGIN_BYTES
        or barrier.get("required_consecutive_samples")
        != VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or barrier.get("sample_interval_seconds") != VRAM_RELEASE_POLL_SECONDS
        or type(barrier.get("baseline_used_bytes")) is not int
        or barrier.get("release_ceiling_bytes")
        != barrier["baseline_used_bytes"] + VRAM_RELEASE_MARGIN_BYTES
        or type(barrier.get("absolute_vram_ceiling_bytes")) is not int
        or barrier["absolute_vram_ceiling_bytes"] != EXPECTED_CAPS["selected_device_vram_byte_ceiling"]
        or barrier["release_ceiling_bytes"] >= barrier["absolute_vram_ceiling_bytes"]
        or type(barrier.get("sample_count")) is not int
        or barrier["sample_count"] < VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or type(barrier.get("minimum_used_bytes")) is not int
        or type(barrier.get("maximum_used_bytes")) is not int
        or type(barrier.get("final_used_bytes")) is not int
        or not isinstance(barrier.get("elapsed_seconds"), (int, float))
        or isinstance(barrier.get("elapsed_seconds"), bool)
        or not math.isfinite(float(barrier["elapsed_seconds"]))
        or not 0.0 <= float(barrier["elapsed_seconds"]) <= VRAM_RELEASE_TIMEOUT_SECONDS
        or not isinstance(barrier.get("counter_path"), str)
        or not barrier["counter_path"]
        or barrier["minimum_used_bytes"] > barrier["final_used_bytes"]
        or barrier["maximum_used_bytes"] < barrier["final_used_bytes"]
        or barrier.get("final_consecutive_samples", 0)
        < VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or barrier.get("final_used_bytes", math.inf) > barrier["release_ceiling_bytes"]
    ):
        raise SplitCollectionError("VRAM release-barrier evidence changed")
    join = evidence.get("join")
    state_bindings = result.get("state_receipt_bindings")
    render_bindings = result.get("render_receipt_bindings")
    scene_metrics = result.get("scene_metrics")
    role_result_bindings = result.get("role_result_bindings")
    join_fields = {
        "exact_join",
        "ordered_roles",
        "expected_counts",
        "observed_counts",
        "state_receipt_count",
        "render_receipt_count",
        "scene_metric_count",
        "ordered_state_binding_identity_sha256",
        "ordered_render_binding_identity_sha256",
        "plan_ordered_state_ids_sha256",
    }
    if (
        not isinstance(join, Mapping)
        or set(join) != join_fields
        or join.get("exact_join") is not True
        or join.get("ordered_roles") != list(ROLE_ORDER)
        or join.get("expected_counts") != EXPECTED_COUNTS
        or join.get("observed_counts") != EXPECTED_COUNTS
        or join.get("state_receipt_count") != ROLE_STATE_COUNT * 2
        or join.get("render_receipt_count") != ROLE_SCENE_COUNT * 2
        or join.get("scene_metric_count") != ROLE_SCENE_COUNT * 2
        or not isinstance(state_bindings, list)
        or not isinstance(render_bindings, list)
        or not isinstance(scene_metrics, list)
        or len(scene_metrics) != EXPECTED_COUNTS["scenes"]
        or not isinstance(role_result_bindings, list)
        or len(role_result_bindings) != 2
        or join.get("ordered_state_binding_identity_sha256")
        != _ordered_binding_identity_v1(state_bindings)
        or join.get("ordered_render_binding_identity_sha256")
        != _ordered_binding_identity_v1(render_bindings)
        or join.get("plan_ordered_state_ids_sha256")
        != hashlib.sha256(
            pilot.canonical_json_bytes([str(row["state_id"]) for row in plan["states"]])
        ).hexdigest()
        or result.get("authority_binding") != dict(authority_binding)
        or result.get("plan_binding") != dict(plan_binding)
    ):
        raise SplitCollectionError("exact split join evidence changed")
    expected_state_paths: list[PurePosixPath] = []
    expected_render_paths: list[PurePosixPath] = []
    expected_scene_rows: list[tuple[str, str]] = []
    seen_scenes: set[tuple[str, str]] = set()
    for state in plan["states"]:
        role = str(state["role"])
        scene_id = str(state["scene_id"])
        state_id = str(state["state_id"])
        expected_state_paths.append(
            PurePosixPath(
                "scenes", role, scene_id, "state_receipts", f"{state_id}.json"
            )
        )
        scene_key = (role, scene_id)
        if scene_key not in seen_scenes:
            seen_scenes.add(scene_key)
            expected_render_paths.append(
                PurePosixPath("scenes", role, scene_id, "live_render_receipt.json")
            )
            expected_scene_rows.append(scene_key)
    observed_state_paths = [
        PurePosixPath(_binding_shape_v1(value, label="joined state receipt")["path"])
        for value in state_bindings
    ]
    observed_render_paths = [
        PurePosixPath(_binding_shape_v1(value, label="joined render receipt")["path"])
        for value in render_bindings
    ]
    if (
        observed_state_paths != expected_state_paths
        or observed_render_paths != expected_render_paths
        or [
            (str(row.get("role")), str(row.get("scene_id")))
            for row in scene_metrics
            if isinstance(row, Mapping)
        ]
        != expected_scene_rows
        or [
            PurePosixPath(
                _binding_shape_v1(value, label="joined role result")["path"]
            )
            for value in role_result_bindings
        ]
        != [
            PurePosixPath("role_results/train.json"),
            PurePosixPath("role_results/eval.json"),
        ]
        or [row["role_result_binding"] for row in workers]
        != role_result_bindings
    ):
        raise SplitCollectionError("joined role/state/scene order or paths changed")
    physics_binding = result.get("_binding")
    if isinstance(physics_binding, Mapping) and isinstance(
        physics_binding.get("path"), str
    ):
        collection_root = Path(str(physics_binding["path"])).parent.resolve(strict=True)
        for role, binding in zip(ROLE_ORDER, role_result_bindings, strict=True):
            _rehash_relative_binding_v1(
                binding,
                collection_root=collection_root,
                label=f"joined {role} role result",
            )
    return {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barrier_exact": True,
        "join_exact": True,
    }


validate_split_process_evidence_v1 = validate_split_collection_evidence_v1


def _join_role_results_v1(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan_receipt_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    role_results: Mapping[str, Mapping[str, Any]],
    role_result_bindings: Mapping[str, Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    release_barrier: Mapping[str, Any],
    collection_wall_seconds: float,
) -> dict[str, Any]:
    if [row.get("role") for row in worker_receipts] != list(ROLE_ORDER):
        raise SplitCollectionError("worker completion order changed")
    state_bindings = [
        row
        for role in ROLE_ORDER
        for row in role_results[role]["state_receipt_bindings"]
    ]
    render_bindings = [
        row
        for role in ROLE_ORDER
        for row in role_results[role]["render_receipt_bindings"]
    ]
    scene_metrics = [
        row for role in ROLE_ORDER for row in role_results[role]["scene_metrics"]
    ]
    paths = [str(row["path"]) for row in (*state_bindings, *render_bindings)]
    if (
        len(state_bindings) != EXPECTED_COUNTS["states"]
        or len(render_bindings) != EXPECTED_COUNTS["scenes"]
        or len(scene_metrics) != EXPECTED_COUNTS["scenes"]
        or len(paths) != len(set(paths))
        or sum(int(row["stored_rgb_bytes"]) for row in role_results.values())
        > EXPECTED_CAPS["stored_rgb_byte_ceiling"]
        or role_results["train"]["runtime_versions"]
        != role_results["eval"]["runtime_versions"]
    ):
        raise SplitCollectionError("combined role closure changed")
    for name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if sum(int(row[name]) for row in scene_metrics) != EXPECTED_CAPS[name]:
            raise SplitCollectionError(f"combined {name} changed")
    evidence = _build_split_evidence_v1(
        plan=plan,
        role_results=role_results,
        role_result_bindings=role_result_bindings,
        worker_receipts=worker_receipts,
        release_barrier=release_barrier,
    )
    result = {
        "schema": pilot.PHYSICS_RESULT_SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "purpose": str(plan["purpose"]),
        "status": "PHYSICS_COMPLETE",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "plan_binding": dict(plan_binding),
        "plan_receipt_binding": dict(plan_receipt_binding),
        "authority_binding": dict(authority_binding),
        "reservation_binding": dict(reservation_binding),
        "caps": dict(authority["caps"]),
        "execution_contract": dict(plan["execution_contract"]),
        "runtime_versions": dict(role_results["train"]["runtime_versions"]),
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": dict(authority["source_bindings"]),
        "expected_counts": copy.deepcopy(EXPECTED_COUNTS),
        "observed_counts": copy.deepcopy(EXPECTED_COUNTS),
        "scene_materialization": None,
        "state_receipt_bindings": state_bindings,
        "render_receipt_bindings": render_bindings,
        "scene_metrics": scene_metrics,
        "visual_domain_limitation": (
            "textured-v03 exact historical RGB call plus a separate transient "
            "depth-only quality render; observations are not screened or refilled"
        ),
        "collection_wall_seconds": collection_wall_seconds,
        "failure": None,
        "role_result_bindings": [dict(role_result_bindings[role]) for role in ROLE_ORDER],
        "split_process_evidence": evidence,
    }
    validate_split_collection_evidence_v1(
        result,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        plan=plan,
    )
    return result


def collect_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], Path]:
    authority, authority_binding, plan, plan_binding = load_and_validate_v1(
        plan_path=plan_path,
        expected_plan_byte_count=expected_plan_byte_count,
        expected_plan_sha256=expected_plan_sha256,
        authority_path=authority_path,
        expected_authority_byte_count=expected_authority_byte_count,
        expected_authority_sha256=expected_authority_sha256,
    )
    started = time.perf_counter()
    nonce = secrets.token_hex(32)
    collection_root, _reservation, reservation_binding = _create_collection_root_v1(
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        orchestrator_nonce=nonce,
    )
    absolute_reservation = pilot.file_binding(collection_root / "reservation.json")
    plan_receipt_binding = kernel._copy_exact_plan_receipt(  # noqa: SLF001
        plan_binding, output_root=collection_root
    )
    used_path, _total_path, _vendor, _device = (
        calibration_supervisor._selected_gpu_memory_files(plan)  # noqa: SLF001
    )
    baseline = _read_vram_counter_v1(used_path)
    ceiling = int(authority["caps"]["selected_device_vram_byte_ceiling"])
    if baseline + VRAM_RELEASE_MARGIN_BYTES >= ceiling:
        raise SplitCollectionError("pre-train VRAM baseline leaves no release margin")

    role_results: dict[str, Mapping[str, Any]] = {}
    role_bindings: dict[str, Mapping[str, Any]] = {}
    worker_receipts: list[Mapping[str, Any]] = []
    for role in ROLE_ORDER:
        if role == "eval":
            release_barrier = _wait_for_vram_release_v1(
                used_path,
                baseline_used_bytes=baseline,
                ceiling_bytes=ceiling,
            )
        argv = _worker_argv_v1(
            role=role,
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
            reservation_binding=absolute_reservation,
            orchestrator_nonce=nonce,
        )
        worker = _run_worker_process_v1(argv, role=role)
        worker_receipts.append(worker)
        role_result, role_binding = _load_role_result_v1(
            collection_root=collection_root,
            role=role,
            authority=authority,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            worker_receipt=worker,
        )
        role_results[role] = role_result
        role_bindings[role] = role_binding
    elapsed = time.perf_counter() - started
    if elapsed > EXPECTED_CAPS["wall_seconds"]:
        raise SplitCollectionError("split collection exceeded global wall cap")
    result = _join_role_results_v1(
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        plan_receipt_binding=plan_receipt_binding,
        reservation_binding=reservation_binding,
        role_results=role_results,
        role_result_bindings=role_bindings,
        worker_receipts=worker_receipts,
        release_barrier=release_barrier,
        collection_wall_seconds=elapsed,
    )
    result_path = collection_root / "physics_result.json"
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


def _worker_main(args: argparse.Namespace) -> int:
    authority, authority_binding, plan, plan_binding = load_and_validate_v1(
        plan_path=args.plan,
        expected_plan_byte_count=args.expected_plan_byte_count,
        expected_plan_sha256=args.expected_plan_sha256,
        authority_path=args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
        _collection_reserved=True,
    )
    collection_root = Path(str(authority["collection_root"])).resolve(strict=True)
    _reservation, absolute_reservation = _read_collection_reservation_v1(
        collection_root=collection_root,
        expected_sha256=args.expected_reservation_sha256,
        expected_byte_count=args.expected_reservation_byte_count,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        orchestrator_nonce=args.orchestrator_nonce,
        orchestrator_pid=args.orchestrator_pid,
    )
    relative_reservation = kernel._relative_output_binding(  # noqa: SLF001
        absolute_reservation, output_root=collection_root
    )
    result, path = _collect_role_v1(
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        reservation_binding=relative_reservation,
        role=args.worker_role,
        orchestrator_pid=args.orchestrator_pid,
    )
    print(json.dumps({"status": result["status"], "role": args.worker_role, "result": str(path)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-byte-count", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--worker-role", choices=ROLE_ORDER)
    parser.add_argument("--expected-reservation-byte-count", type=int)
    parser.add_argument("--expected-reservation-sha256")
    parser.add_argument("--orchestrator-nonce")
    parser.add_argument("--orchestrator-pid", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker_role is not None:
            if (
                args.expected_reservation_byte_count is None
                or args.expected_reservation_sha256 is None
                or args.orchestrator_nonce is None
                or args.orchestrator_pid is None
            ):
                raise SplitCollectionError("worker reservation pins are incomplete")
            return _worker_main(args)
        if any(
            value is not None
            for value in (
                args.expected_reservation_byte_count,
                args.expected_reservation_sha256,
                args.orchestrator_nonce,
                args.orchestrator_pid,
            )
        ):
            raise SplitCollectionError("parent invocation contains worker-only pins")
        result, path = collect_v1(
            plan_path=args.plan,
            expected_plan_byte_count=args.expected_plan_byte_count,
            expected_plan_sha256=args.expected_plan_sha256,
            authority_path=args.authority,
            expected_authority_byte_count=args.expected_authority_byte_count,
            expected_authority_sha256=args.expected_authority_sha256,
        )
        print(json.dumps({"status": result["status"], "physics_result": str(path)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_FIELDS",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "PLAN_FIRST_EFFECTIVE_GENESIS_SEED",
    "PLAN_FIRST_PHYSICS_SEED",
    "ROLE_ORDER",
    "SplitCollectionError",
    "_validate_scene_diversity_plan_v1",
    "collect_v1",
    "load_and_validate_replacement_v1",
    "load_and_validate_v1",
    "validate_split_collection_evidence_v1",
    "validate_split_process_evidence_v1",
]
