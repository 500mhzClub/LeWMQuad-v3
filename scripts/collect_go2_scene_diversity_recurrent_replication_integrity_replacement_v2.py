#!/usr/bin/env python3
"""Collect the frozen scene-diversity replication in 64 clean processes.

This is a science-identical infrastructure replacement.  The 64 scenes are
collected, in frozen plan order, by exactly one fresh Genesis process apiece.
The parent admits no batching or refill and writes the conventional combined
``physics_result.json`` only after every scene index, every referenced receipt,
and every post-process VRAM-release barrier have been checked.
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
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_"
    "INTEGRITY_REPLACEMENT_V2"
)
AUTHORITY_FIELDS = frozen.AUTHORITY_FIELDS
EXPECTED_COUNTS = frozen.EXPECTED_COUNTS
EXPECTED_CAPS = frozen.EXPECTED_CAPS
EXPECTED_HISTORY_PANEL = frozen.EXPECTED_HISTORY_PANEL
EXPECTED_PERMISSIONS = frozen.EXPECTED_PERMISSIONS

ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-integrity-replacement-v2"
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "collection_reservation_v1"
)
RESERVATION_STATUS = "RESERVED_ONE_SHOT_PER_SCENE_COLLECTION_CONSUMED"
SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "scene_physics_result_v1"
)
SCENE_RESULT_STATUS = "SCENE_PHYSICS_COMPLETE"
SCENE_EVIDENCE_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
    "scene_process_evidence_v1"
)
SCENE_EVIDENCE_STATUS = "PASS_EXACT_64_PROCESS_COLLECTION_AND_JOIN"

ROLE_ORDER = ("train", "eval")
SCENE_COUNT = 64
TRAIN_SCENE_COUNT = 32
EVAL_SCENE_COUNT = 32
STATES_PER_SCENE = 4
BRANCHES_PER_SCENE = 36
CONTEXT_FRAMES_PER_SCENE = 12
STORED_FRAMES_PER_SCENE = 48
PLAN_FIRST_PHYSICS_SEED = 14_102_849_992_353_107_924
PLAN_FIRST_EFFECTIVE_GENESIS_SEED = 315_871_188
VRAM_RELEASE_MARGIN_BYTES = 512 * 1024 * 1024
VRAM_RELEASE_TIMEOUT_SECONDS = 60.0
VRAM_RELEASE_POLL_SECONDS = 0.05
VRAM_RELEASE_CONSECUTIVE_SAMPLES = 3

# Version-bound source audit for the sole scientific-equivalence subtlety
# introduced by clean per-scene processes.  ``gs.init`` resets the process RNG,
# but the selected collection path has no outcome-affecting post-init random
# draw: manifests bind geometry/physics/camera, the rollout runner is put into
# exact clones and consumes caller-provided blocks, texture choice has its own
# scene-keyed ``random.Random``, and the only reachable Genesis global-NumPy
# draw assigns collision-debug RGBA.  That RGBA is neither a rigid-body input
# nor an observed surface (physical robot rendering is disabled; replay boxes
# have explicit bound textures).  ``RolloutRunner.__init__`` constructs an
# EpisodeScheduler and consumes plan-seeded private RNG draws, but the selected
# ``execute_requested_block`` route never reads its policy or assignments.
# No stochastic sensors, IK, path planning, force fields, or spawn sampling are
# invoked.
PROCESS_RESET_EQUIVALENCE_AUDIT_V2 = {
    "schema": (
        "lewm_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_"
        "process_reset_equivalence_source_audit_v1"
    ),
    "status": "PASS_NO_OUTCOME_AFFECTING_POST_INIT_RANDOM_DRAW",
    "audit_scope": "frozen_physics_and_textured_v03_render_call_graph",
    "manifest_binds_scene_geometry_physics_camera": True,
    "exact_clone_initialization_overwrites_dynamic_state": True,
    "requested_action_blocks_are_plan_bound": True,
    "spawn_randomization_disabled_and_bypassed": True,
    "collector_scheduler_policy_and_assignment_route_not_consumed": True,
    "texture_rng_is_scene_keyed_local_rng": True,
    "stochastic_sensors_ik_planners_force_fields_not_invoked": True,
    "reachable_global_rng_draw_affects_only_unobserved_collision_debug_rgba": True,
    "stored_rgb_uses_explicit_bound_texture_surfaces": True,
    "source_functions": [
        "lewm_genesis.scene_builder.initialize_genesis",
        "lewm_genesis.scene_builder.build_scene_from_pack",
        "scripts.collect_go2_world_model_counterfactual_pilot_v1._initialize_exact_clones",
        "lewm.benchmarks.go2_world_model_counterfactual_pilot_v1.execute_lockstep_trial",
        "lewm_genesis.rollout.RolloutRunner.execute_requested_block",
        "lewm_genesis.textures.select_scene_textures",
        "scripts.render_replay_v03.build_scene",
    ],
}


class SceneProcessCollectionError(RuntimeError):
    """Raised when the exact per-scene process contract changes."""


# The runner overlay and source review deliberately reuse the frozen scientific
# plan validator; only process lifetime and output namespace differ.
_validate_scene_diversity_plan_v1 = frozen._validate_scene_diversity_plan_v1


def _standard_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if "file_sha256" in value:
        return frozen._standard_binding(value)  # noqa: SLF001
    return {
        "path": str(value["path"]),
        "sha256": str(value["sha256"]),
        "byte_count": int(value["byte_count"]),
    }


def _scene_expected_counts_v2(role: str) -> dict[str, Any]:
    if role not in ROLE_ORDER:
        raise SceneProcessCollectionError("scene role changed")
    return {
        "scenes": 1,
        "states": STATES_PER_SCENE,
        "roles": {role: STATES_PER_SCENE},
        "actions": 9,
        "candidate_branches": BRANCHES_PER_SCENE,
        "sentinel_branches": 0,
        "total_branches": BRANCHES_PER_SCENE,
        "context_frames": CONTEXT_FRAMES_PER_SCENE,
        "target_frames": BRANCHES_PER_SCENE,
    }


def _scene_slices_v2(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    states = plan.get("states")
    if not isinstance(states, list) or len(states) != EXPECTED_COUNTS["states"]:
        raise SceneProcessCollectionError("plan state closure changed")
    grouped: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, Mapping):
            raise SceneProcessCollectionError("plan state changed")
        key = (str(state.get("role")), str(state.get("scene_id")))
        if not grouped or grouped[-1]["key"] != key:
            grouped.append({"key": key, "states": []})
        grouped[-1]["states"].append(state)
    if len(grouped) != SCENE_COUNT:
        raise SceneProcessCollectionError("plan scene closure changed")
    result: list[dict[str, Any]] = []
    for scene_index, row in enumerate(grouped):
        role, scene_id = row["key"]
        scene_states = list(row["states"])
        expected_role = "train" if scene_index < TRAIN_SCENE_COUNT else "eval"
        if (
            role != expected_role
            or len(scene_states) != STATES_PER_SCENE
            or any(str(state.get("role")) != role for state in scene_states)
            or any(str(state.get("scene_id")) != scene_id for state in scene_states)
        ):
            raise SceneProcessCollectionError("plan scene order or membership changed")
        result.append(
            {
                "scene_index": scene_index,
                "role": role,
                "scene_id": scene_id,
                "states": scene_states,
            }
        )
    if len({(row["role"], row["scene_id"]) for row in result}) != SCENE_COUNT:
        raise SceneProcessCollectionError("plan contains repeated scene identities")
    return result


def _validate_output_roots_v2(
    *, authority: Mapping[str, Any], plan: Mapping[str, Any], reserved: bool
) -> tuple[Path, Path]:
    attempt = Path(os.path.abspath(str(authority.get("attempt_root", ""))))
    collection = Path(os.path.abspath(str(authority.get("collection_root", ""))))
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    frozen._reject_protected_path(attempt, label="V2 replacement attempt root")  # noqa: SLF001
    frozen._reject_protected_path(collection, label="V2 replacement collection root")  # noqa: SLF001
    try:
        attempt.relative_to(development)
        collection.relative_to(attempt)
    except ValueError as exc:
        raise SceneProcessCollectionError("V2 roots escape development custody") from exc
    if (
        collection.parent != attempt
        or str(collection) != str(plan.get("output_root"))
        or not attempt.is_dir()
        or attempt.is_symlink()
    ):
        raise SceneProcessCollectionError("V2 attempt/collection roots changed")
    if reserved:
        if not collection.is_dir() or collection.is_symlink():
            raise SceneProcessCollectionError("V2 collection root is not reserved")
    elif collection.exists() or collection.is_symlink():
        raise SceneProcessCollectionError("V2 collection root is not fresh")
    return attempt.resolve(strict=True), (
        collection.resolve(strict=True) if reserved else collection
    )


def _validate_authority_v2(
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
        raise SceneProcessCollectionError("V2 execution authority changed")
    _validate_output_roots_v2(authority=authority, plan=plan, reserved=reserved)
    return dict(authority)


def load_and_validate_v2(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
    _collection_reserved: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen._reject_protected_path(plan_path, label="V2 replacement exact plan")  # noqa: SLF001
    frozen._reject_protected_path(authority_path, label="V2 replacement authority")  # noqa: SLF001
    raw_plan, plan_binding = pilot.read_bound_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="V2 replacement exact plan",
    )
    plan = _validate_scene_diversity_plan_v1(
        copy.deepcopy(pilot.validate_plan(raw_plan))
    )
    bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    if (
        plan.get("attempt_id") != ATTEMPT_ID
        or [row.get("group_index") for row in plan["states"]]
        != list(range(EXPECTED_COUNTS["states"]))
    ):
        raise SceneProcessCollectionError("V2 plan identity changed")
    _scene_slices_v2(plan)
    raw_authority, historical_authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="V2 replacement execution authority",
    )
    authority_binding = _standard_binding(historical_authority_binding)
    authority = _validate_authority_v2(
        raw_authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        reserved=_collection_reserved,
    )
    return authority, authority_binding, plan, plan_binding


load_and_validate_replacement_v2 = load_and_validate_v2


def _create_collection_root_v2(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    orchestrator_nonce: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    _attempt, collection = _validate_output_roots_v2(
        authority=authority, plan=plan, reserved=False
    )
    try:
        os.mkdir(collection, mode=0o700)
        os.mkdir(collection / "scenes", mode=0o700)
        os.mkdir(collection / "scene_results", mode=0o700)
    except OSError as exc:
        raise SceneProcessCollectionError("could not reserve V2 collection root") from exc
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
        "fixed_scene_process_count": SCENE_COUNT,
        "fixed_scene_order": "full_frozen_plan_order",
        "release_after_every_worker_including_final": True,
        "root_creation_consumes_attempt": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
        "adaptive_batching_authorized": False,
        "partial_artifact_reuse_authorized": False,
    }
    binding = pilot.write_json_exclusive(collection / "reservation.json", reservation)
    return collection, reservation, kernel._relative_output_binding(  # noqa: SLF001
        binding, output_root=collection
    )


def _read_collection_reservation_v2(
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
        label="V2 collection reservation",
    )
    if (
        value.get("schema") != RESERVATION_SCHEMA
        or value.get("status") != RESERVATION_STATUS
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("plan_binding") != _standard_binding(plan_binding)
        or value.get("orchestrator_nonce") != orchestrator_nonce
        or value.get("orchestrator_pid") != orchestrator_pid
        or value.get("fixed_scene_process_count") != SCENE_COUNT
        or value.get("fixed_scene_order") != "full_frozen_plan_order"
        or value.get("release_after_every_worker_including_final") is not True
        or any(
            value.get(name) is not False
            for name in (
                "retry_authorized",
                "resume_authorized",
                "overwrite_authorized",
                "refill_authorized",
                "adaptive_batching_authorized",
                "partial_artifact_reuse_authorized",
            )
        )
    ):
        raise SceneProcessCollectionError("V2 collection reservation changed")
    return value, binding


def _initialize_genesis_v2(*, backend: str, seed: int) -> None:
    from lewm_genesis.scene_builder import initialize_genesis

    initialize_genesis(backend=backend, seed=seed)


def _initialize_from_plan_first_scene_v2(
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
        raise SceneProcessCollectionError("plan-first process-global Genesis seed changed")
    _initialize_genesis_v2(backend=backend, seed=int(full_seed))
    return {
        "source": "full_plan_first_scene_bound_manifest",
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "manifest_binding": dict(binding),
        "backend": backend,
        "full_physics_seed": int(full_seed),
        "effective_genesis_seed": int(effective),
    }


def _install_scene_local_mesh_cache_v2(
    runtime: dict[str, Any], *, scene_root: Path
) -> Path:
    """Route deterministic OBJ writes to this worker's role/scene namespace."""

    cache_root = scene_root / "derived_meshes"
    original_cached_box_obj = runtime.get("cached_box_obj")
    render_builder = runtime.get("build_textured_v03_scene")
    render_globals = getattr(render_builder, "__globals__", None)
    if not callable(original_cached_box_obj) or not isinstance(render_globals, dict):
        raise SceneProcessCollectionError("textured-v03 mesh cache seam changed")

    def scene_local_cached_box_obj(
        size_xyz_m: Sequence[float], *, tiles_per_m: float = 0.7
    ) -> str:
        return original_cached_box_obj(
            tuple(float(value) for value in size_xyz_m),
            tiles_per_m=tiles_per_m,
            cache_dir=cache_root,
        )

    runtime["cached_box_obj"] = scene_local_cached_box_obj
    render_globals["cached_box_obj"] = scene_local_cached_box_obj
    return cache_root


def _validate_scene_local_mesh_bindings_v2(
    metrics: Mapping[str, Any], *, cache_root: Path, collection_root: Path
) -> dict[str, Any]:
    raw_bindings = metrics.get("derived_mesh_bindings")
    if not isinstance(raw_bindings, list):
        raise SceneProcessCollectionError("scene-local derived mesh bindings are absent")
    bindings_by_path: dict[str, dict[str, Any]] = {}
    for raw_binding in raw_bindings:
        try:
            binding = pilot.require_binding(raw_binding, label="scene-local OBJ mesh")
            Path(str(binding["path"])).resolve(strict=True).relative_to(
                cache_root.resolve(strict=True)
            )
        except (OSError, ValueError, pilot.PilotContractError) as exc:
            raise SceneProcessCollectionError(
                "derived mesh escaped or changed outside the scene cache"
            ) from exc
        bindings_by_path[str(binding["path"])] = binding
    bindings = [bindings_by_path[path] for path in sorted(bindings_by_path)]
    if not cache_root.is_dir() or cache_root.is_symlink() or not bindings:
        raise SceneProcessCollectionError("scene-local derived mesh cache is incomplete")
    try:
        cache_children = list(cache_root.iterdir())
    except OSError as exc:
        raise SceneProcessCollectionError("scene-local mesh cache cannot be listed") from exc
    if any(child.is_symlink() or not child.is_file() for child in cache_children):
        raise SceneProcessCollectionError("scene-local mesh cache contains an unbound entry")
    observed_paths = sorted(str(child.resolve(strict=True)) for child in cache_children)
    if observed_paths != sorted(bindings_by_path):
        raise SceneProcessCollectionError("scene-local mesh file closure changed")
    return {
        "path": cache_root.relative_to(collection_root).as_posix(),
        "cross_scene_reuse_authorized": False,
        "mesh_count": len(bindings),
        "bindings_identity_sha256": hashlib.sha256(
            pilot.canonical_json_bytes(bindings)
        ).hexdigest(),
    }


def _observed_scene_counts_v2(
    receipts: Sequence[Mapping[str, Any]], *, role: str
) -> dict[str, Any]:
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
    return {
        "scenes": len({str(row["state"]["scene_id"]) for row in receipts}),
        "states": len(receipts),
        "roles": {role: len(receipts)},
        "actions": 9,
        "candidate_branches": candidate,
        "sentinel_branches": sentinel,
        "total_branches": candidate + sentinel,
        "context_frames": sum(
            len(row["context"]["frame_identities"]) for row in receipts
        ),
        "target_frames": candidate + sentinel,
    }


def _collect_scene_worker_v2(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_index: int,
    orchestrator_pid: int,
) -> tuple[dict[str, Any], Path]:
    scenes = _scene_slices_v2(plan)
    if (
        type(scene_index) is not int
        or not 0 <= scene_index < SCENE_COUNT
        or orchestrator_pid <= 1
        or os.getppid() != orchestrator_pid
    ):
        raise SceneProcessCollectionError("scene worker ownership/index changed")
    scene = scenes[scene_index]
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    states = list(scene["states"])
    collection_root = Path(str(authority["collection_root"])).resolve(strict=True)
    scene_root = collection_root / "scenes" / role / scene_id
    result_path = collection_root / "scene_results" / f"{scene_index:03d}.json"
    if scene_root.exists() or scene_root.is_symlink() or result_path.exists():
        raise SceneProcessCollectionError("scene worker output is not fresh")
    started = time.perf_counter()

    pilot.require_plan_bindings(plan)
    kernel._validate_python_runtime(plan)  # noqa: SLF001
    kernel._validate_execution_environment(plan)  # noqa: SLF001
    bounded._validate_bound_scenes(plan)  # noqa: SLF001
    runtime_versions = kernel._capture_runtime_versions()  # noqa: SLF001
    runtime = kernel._runtime_imports(textured_v03=True)  # noqa: SLF001
    mesh_cache_root = _install_scene_local_mesh_cache_v2(
        runtime, scene_root=scene_root
    )
    platform = runtime["load_platform_manifest"](
        plan["runtime_bindings"]["platform_manifest"]["path"]
    )
    resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), REPO_ROOT)
    if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
        raise SceneProcessCollectionError("platform resolves a different Go2 URDF")
    registry = runtime["PrimitiveRegistry"].from_yaml(
        plan["runtime_bindings"]["primitive_registry"]["path"]
    )
    action_blocks = kernel._load_action_blocks(  # noqa: SLF001
        plan=plan, registry=registry, expand=runtime["expand_primitive_to_block"]
    )
    genesis_initialization = _initialize_from_plan_first_scene_v2(plan=plan)
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
        raise SceneProcessCollectionError("scene worker changed planned state order")
    stored_rgb_bytes = sum(int(frame["byte_count"]) for frame in frames)
    if stored_rgb_bytes > EXPECTED_CAPS["stored_rgb_byte_ceiling"]:
        raise SceneProcessCollectionError("scene worker stored RGB ceiling exceeded")
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
    relative_render = kernel._relative_output_binding(  # noqa: SLF001
        render_binding, output_root=collection_root
    )
    state_bindings: list[dict[str, Any]] = []
    for receipt in receipts:
        state_id = str(receipt["state"]["state_id"])
        receipt["render_receipt_binding"] = relative_render
        binding = pilot.write_json_exclusive(
            scene_root / "state_receipts" / f"{state_id}.json", receipt
        )
        state_bindings.append(
            kernel._relative_output_binding(binding, output_root=collection_root)  # noqa: SLF001
        )
    observed = _observed_scene_counts_v2(receipts, role=role)
    expected = _scene_expected_counts_v2(role)
    if observed != expected:
        raise SceneProcessCollectionError("scene worker observed counts changed")
    for metric_name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if int(metrics[metric_name]) != STORED_FRAMES_PER_SCENE:
            raise SceneProcessCollectionError(f"scene worker {metric_name} changed")
    mesh_cache = _validate_scene_local_mesh_bindings_v2(
        metrics,
        cache_root=mesh_cache_root,
        collection_root=collection_root,
    )
    result = {
        "schema": SCENE_RESULT_SCHEMA,
        "status": SCENE_RESULT_STATUS,
        "attempt_id": ATTEMPT_ID,
        "scene_index": scene_index,
        "role": role,
        "scene_id": scene_id,
        "worker_pid": os.getpid(),
        "orchestrator_pid": orchestrator_pid,
        "sys_executable": str(Path(sys.executable).resolve(strict=True)),
        "fresh_process": True,
        "execution_seed": int(plan["execution_contract"]["seed"]),
        "genesis_initialization": genesis_initialization,
        "process_reset_equivalence_audit": copy.deepcopy(
            PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        ),
        "scene_local_mesh_cache": mesh_cache,
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
        "state_receipt_bindings": state_bindings,
        "render_receipt_binding": relative_render,
        "scene_metric": metrics,
        "stored_rgb_bytes": stored_rgb_bytes,
        "collection_wall_seconds": time.perf_counter() - started,
        "failure": None,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "allows_adaptive_batching": False,
    }
    if result["collection_wall_seconds"] > EXPECTED_CAPS["wall_seconds"]:
        raise SceneProcessCollectionError("scene worker exceeded global wall cap")
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


def _read_vram_counter_v2(path: Path) -> int:
    selected = Path(path)
    if selected.is_symlink():
        raise SceneProcessCollectionError("selected-device VRAM counter is a symlink")
    try:
        raw = selected.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SceneProcessCollectionError("selected-device VRAM counter read failed") from exc
    if not raw.isdigit():
        raise SceneProcessCollectionError("selected-device VRAM counter is malformed")
    return int(raw)


def _wait_for_vram_release_v2(
    used_path: Path, *, baseline_used_bytes: int, ceiling_bytes: int
) -> dict[str, Any]:
    release_ceiling = baseline_used_bytes + VRAM_RELEASE_MARGIN_BYTES
    if baseline_used_bytes < 0 or release_ceiling >= ceiling_bytes:
        raise SceneProcessCollectionError("VRAM release baseline leaves no bounded margin")
    started = time.monotonic()
    samples = 0
    consecutive = 0
    minimum: int | None = None
    maximum: int | None = None
    final = baseline_used_bytes
    while time.monotonic() - started <= VRAM_RELEASE_TIMEOUT_SECONDS:
        final = _read_vram_counter_v2(used_path)
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
    raise SceneProcessCollectionError("VRAM did not return through release barrier")


def _worker_argv_v2(
    *,
    scene_index: int,
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
        "--worker-scene-index",
        str(scene_index),
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


def _run_worker_process_v2(
    argv: Sequence[str],
    *,
    scene_index: int,
    role: str,
    scene_id: str,
    used_path: Path,
    ceiling_bytes: int,
) -> dict[str, Any]:
    if (
        list(argv[:1]) != [sys.executable]
        or "--worker-scene-index" not in argv
        or type(ceiling_bytes) is not int
        or ceiling_bytes != EXPECTED_CAPS["selected_device_vram_byte_ceiling"]
    ):
        raise SceneProcessCollectionError("scene worker invocation changed")
    started = time.monotonic()
    # Deliberately inherit the outer collector process group.  The existing
    # 20-ms global monitor can therefore kill this worker on either cap.
    prelaunch_baseline_used_bytes = _read_vram_counter_v2(used_path)
    if prelaunch_baseline_used_bytes + VRAM_RELEASE_MARGIN_BYTES >= ceiling_bytes:
        raise SceneProcessCollectionError(
            "scene pre-launch VRAM baseline leaves no release margin"
        )
    process = subprocess.Popen(list(argv), cwd=REPO_ROOT)
    parent_process_group_id = os.getpgrp()
    try:
        child_process_group_id = os.getpgid(process.pid)
    except OSError as exc:
        child_process_group_id = None
        process_group_observation_error = f"{type(exc).__name__}: {exc}"
    else:
        process_group_observation_error = None
    returncode = process.wait()
    receipt = {
        "scene_index": scene_index,
        "role": role,
        "scene_id": scene_id,
        "pid": process.pid,
        "parent_pid": os.getpid(),
        "sys_executable": str(Path(sys.executable).resolve(strict=True)),
        "fresh_process": True,
        "inherited_outer_process_group": (
            child_process_group_id == parent_process_group_id
        ),
        "parent_process_group_id": parent_process_group_id,
        "child_process_group_id": child_process_group_id,
        "process_group_equality_observed": (
            child_process_group_id == parent_process_group_id
        ),
        "process_group_observation_error": process_group_observation_error,
        "prelaunch_baseline_used_bytes": prelaunch_baseline_used_bytes,
        "exit_code": int(returncode),
        "elapsed_seconds": time.monotonic() - started,
    }
    return receipt


def _validate_completed_worker_v2(receipt: Mapping[str, Any]) -> None:
    """Reject a completed worker only after its release barrier has run."""

    if (
        receipt.get("exit_code") != 0
        or receipt.get("process_group_observation_error") is not None
        or receipt.get("process_group_equality_observed") is not True
        or type(receipt.get("parent_process_group_id")) is not int
        or type(receipt.get("child_process_group_id")) is not int
        or receipt.get("child_process_group_id")
        != receipt.get("parent_process_group_id")
    ):
        raise SceneProcessCollectionError(
            f"scene worker {receipt.get('scene_index')} completion contract failed"
        )


def _binding_shape_v2(value: object, *, label: str) -> dict[str, Any]:
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
        raise SceneProcessCollectionError(f"{label} binding changed")
    return dict(value)


def _rehash_relative_binding_v2(
    value: object, *, collection_root: Path, label: str
) -> dict[str, Any]:
    binding = _binding_shape_v2(value, label=label)
    relative = PurePosixPath(str(binding["path"]))
    selected = collection_root.joinpath(*relative.parts)
    try:
        selected.resolve(strict=True).relative_to(collection_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SceneProcessCollectionError(f"{label} escapes or is absent") from exc
    if selected.is_symlink() or not selected.is_file():
        raise SceneProcessCollectionError(f"{label} is not a regular no-follow file")
    actual = pilot.file_binding(selected)
    if (
        actual["file_sha256"] != binding["file_sha256"]
        or actual["byte_count"] != binding["byte_count"]
    ):
        raise SceneProcessCollectionError(f"{label} content binding changed")
    return binding


def _load_scene_result_v2(
    *,
    collection_root: Path,
    scene: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    worker_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_index = int(scene["scene_index"])
    role = str(scene["role"])
    scene_id = str(scene["scene_id"])
    path = collection_root / "scene_results" / f"{scene_index:03d}.json"
    absolute = pilot.file_binding(path)
    value, actual = pilot.read_bound_json(
        path,
        expected_sha256=str(absolute["file_sha256"]),
        expected_byte_count=int(absolute["byte_count"]),
        label=f"scene {scene_index:03d} result",
    )
    ordered_ids = [str(row["state_id"]) for row in scene["states"]]
    state_bindings = value.get("state_receipt_bindings")
    render_binding = value.get("render_receipt_binding")
    metric = value.get("scene_metric")
    mesh_cache = value.get("scene_local_mesh_cache")
    if (
        actual != absolute
        or value.get("schema") != SCENE_RESULT_SCHEMA
        or value.get("status") != SCENE_RESULT_STATUS
        or value.get("attempt_id") != ATTEMPT_ID
        or value.get("scene_index") != scene_index
        or value.get("role") != role
        or value.get("scene_id") != scene_id
        or value.get("worker_pid") != worker_receipt.get("pid")
        or value.get("orchestrator_pid") != os.getpid()
        or value.get("fresh_process") is not True
        or value.get("execution_seed") != plan["execution_contract"]["seed"]
        or value.get("process_reset_equivalence_audit")
        != PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        or value.get("plan_binding") != dict(plan_binding)
        or value.get("authority_binding") != dict(authority_binding)
        or value.get("collection_reservation_binding") != dict(reservation_binding)
        or value.get("caps") != authority["caps"]
        or value.get("runtime_bindings") != plan["runtime_bindings"]
        or value.get("source_bindings") != authority["source_bindings"]
        or value.get("expected_counts") != _scene_expected_counts_v2(role)
        or value.get("observed_counts") != _scene_expected_counts_v2(role)
        or value.get("ordered_state_ids") != ordered_ids
        or not isinstance(state_bindings, list)
        or len(state_bindings) != STATES_PER_SCENE
        or not isinstance(render_binding, Mapping)
        or not isinstance(metric, Mapping)
        or metric.get("role") != role
        or metric.get("scene_id") != scene_id
        or metric.get("states") != STATES_PER_SCENE
        or not isinstance(mesh_cache, Mapping)
        or value.get("failure") is not None
        or value.get("authorizes_retry_or_resume") is not False
        or value.get("allows_refill") is not False
        or value.get("allows_overwrite") is not False
        or value.get("allows_adaptive_batching") is not False
    ):
        raise SceneProcessCollectionError("scene result contract changed")
    initialization = value.get("genesis_initialization")
    if (
        not isinstance(initialization, Mapping)
        or initialization.get("full_physics_seed") != PLAN_FIRST_PHYSICS_SEED
        or initialization.get("effective_genesis_seed")
        != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
        or initialization.get("source") != "full_plan_first_scene_bound_manifest"
    ):
        raise SceneProcessCollectionError("scene Genesis initialization changed")
    for state_id, raw_binding in zip(ordered_ids, state_bindings, strict=True):
        binding = _rehash_relative_binding_v2(
            raw_binding, collection_root=collection_root, label="scene state receipt"
        )
        expected_path = PurePosixPath(
            "scenes", role, scene_id, "state_receipts", f"{state_id}.json"
        )
        if PurePosixPath(binding["path"]) != expected_path:
            raise SceneProcessCollectionError("scene state receipt order/path changed")
    checked_render = _rehash_relative_binding_v2(
        render_binding, collection_root=collection_root, label="scene render receipt"
    )
    if PurePosixPath(checked_render["path"]) != PurePosixPath(
        "scenes", role, scene_id, "live_render_receipt.json"
    ):
        raise SceneProcessCollectionError("scene render receipt path changed")
    observed_mesh = _validate_scene_local_mesh_bindings_v2(
        metric,
        cache_root=collection_root / "scenes" / role / scene_id / "derived_meshes",
        collection_root=collection_root,
    )
    if observed_mesh != dict(mesh_cache):
        raise SceneProcessCollectionError("scene-local mesh cache changed")
    relative = kernel._relative_output_binding(absolute, output_root=collection_root)  # noqa: SLF001
    if PurePosixPath(relative["path"]) != PurePosixPath(
        "scene_results", f"{scene_index:03d}.json"
    ):
        raise SceneProcessCollectionError("scene-result index path changed")
    return value, relative


def _ordered_binding_identity_v2(values: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(pilot.canonical_json_bytes(list(values))).hexdigest()


def _barrier_with_identity_v2(
    barrier: Mapping[str, Any], *, scene: Mapping[str, Any], worker_pid: int
) -> dict[str, Any]:
    return {
        "scene_index": int(scene["scene_index"]),
        "role": str(scene["role"]),
        "scene_id": str(scene["scene_id"]),
        "after_worker_pid": worker_pid,
        **dict(barrier),
    }


def _build_scene_process_evidence_v2(
    *,
    plan: Mapping[str, Any],
    scene_results: Sequence[Mapping[str, Any]],
    scene_result_bindings: Sequence[Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    release_barriers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenes = _scene_slices_v2(plan)
    workers: list[dict[str, Any]] = []
    for scene, result, binding, receipt in zip(
        scenes, scene_results, scene_result_bindings, worker_receipts, strict=True
    ):
        workers.append(
            {
                "scene_index": int(scene["scene_index"]),
                "role": str(scene["role"]),
                "scene_id": str(scene["scene_id"]),
                "fresh_process": True,
                "pid": int(receipt["pid"]),
                "parent_pid": int(receipt["parent_pid"]),
                "sys_executable": str(receipt["sys_executable"]),
                "inherited_outer_process_group": True,
                "parent_process_group_id": int(
                    receipt["parent_process_group_id"]
                ),
                "child_process_group_id": int(
                    receipt["child_process_group_id"]
                ),
                "process_group_equality_observed": True,
                "prelaunch_baseline_used_bytes": int(
                    receipt["prelaunch_baseline_used_bytes"]
                ),
                "exit_code": 0,
                "execution_seed": int(result["execution_seed"]),
                "full_genesis_seed": PLAN_FIRST_PHYSICS_SEED,
                "effective_genesis_seed": PLAN_FIRST_EFFECTIVE_GENESIS_SEED,
                "scene_local_mesh_cache": copy.deepcopy(
                    result["scene_local_mesh_cache"]
                ),
                "expected_counts": copy.deepcopy(result["expected_counts"]),
                "observed_counts": copy.deepcopy(result["observed_counts"]),
                "scene_result_binding": dict(binding),
            }
        )
    state_bindings = [
        binding
        for result in scene_results
        for binding in result["state_receipt_bindings"]
    ]
    render_bindings = [result["render_receipt_binding"] for result in scene_results]
    return {
        "schema": SCENE_EVIDENCE_SCHEMA,
        "status": SCENE_EVIDENCE_STATUS,
        "process_reset_equivalence_audit": copy.deepcopy(
            PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        ),
        "workers": workers,
        "release_barriers": [dict(row) for row in release_barriers],
        "sequential_launch": {
            "exact_process_count": SCENE_COUNT,
            "plan_order": True,
            "each_exit_observed_before_release_barrier": True,
            "each_release_barrier_passed_before_next_launch": True,
            "final_release_barrier_passed_before_join": True,
            "adaptive_batching_used": False,
            "fallback_used": False,
            "refill_used": False,
        },
        "join": {
            "exact_join": True,
            "expected_counts": copy.deepcopy(EXPECTED_COUNTS),
            "observed_counts": copy.deepcopy(EXPECTED_COUNTS),
            "scene_result_count": len(scene_result_bindings),
            "state_receipt_count": len(state_bindings),
            "render_receipt_count": len(render_bindings),
            "scene_metric_count": len(scene_results),
            "ordered_scene_result_binding_identity_sha256": _ordered_binding_identity_v2(
                scene_result_bindings
            ),
            "ordered_state_binding_identity_sha256": _ordered_binding_identity_v2(
                state_bindings
            ),
            "ordered_render_binding_identity_sha256": _ordered_binding_identity_v2(
                render_bindings
            ),
            "plan_ordered_state_ids_sha256": hashlib.sha256(
                pilot.canonical_json_bytes(
                    [str(row["state_id"]) for row in plan["states"]]
                )
            ).hexdigest(),
        },
    }


def _validate_release_barrier_shape_v2(
    barrier: Mapping[str, Any], *, scene: Mapping[str, Any], worker_pid: int
) -> None:
    fields = {
        "scene_index",
        "role",
        "scene_id",
        "after_worker_pid",
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
        set(barrier) != fields
        or barrier.get("scene_index") != scene["scene_index"]
        or barrier.get("role") != scene["role"]
        or barrier.get("scene_id") != scene["scene_id"]
        or barrier.get("after_worker_pid") != worker_pid
        or barrier.get("status") != "PASSED"
        or barrier.get("read_only") is not True
        or barrier.get("release_margin_bytes") != VRAM_RELEASE_MARGIN_BYTES
        or barrier.get("required_consecutive_samples")
        != VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or barrier.get("sample_interval_seconds") != VRAM_RELEASE_POLL_SECONDS
        or type(barrier.get("baseline_used_bytes")) is not int
        or barrier.get("release_ceiling_bytes")
        != barrier["baseline_used_bytes"] + VRAM_RELEASE_MARGIN_BYTES
        or barrier.get("absolute_vram_ceiling_bytes")
        != EXPECTED_CAPS["selected_device_vram_byte_ceiling"]
        or barrier["release_ceiling_bytes"] >= barrier["absolute_vram_ceiling_bytes"]
        or type(barrier.get("sample_count")) is not int
        or barrier["sample_count"] < VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or type(barrier.get("minimum_used_bytes")) is not int
        or type(barrier.get("maximum_used_bytes")) is not int
        or type(barrier.get("final_used_bytes")) is not int
        or type(barrier.get("final_consecutive_samples")) is not int
        or barrier["final_consecutive_samples"] < VRAM_RELEASE_CONSECUTIVE_SAMPLES
        or barrier["final_used_bytes"] > barrier["release_ceiling_bytes"]
        or barrier["minimum_used_bytes"] > barrier["final_used_bytes"]
        or barrier["maximum_used_bytes"] < barrier["final_used_bytes"]
        or not isinstance(barrier.get("counter_path"), str)
        or not barrier["counter_path"]
        or not isinstance(barrier.get("elapsed_seconds"), (int, float))
        or isinstance(barrier.get("elapsed_seconds"), bool)
        or not math.isfinite(float(barrier["elapsed_seconds"]))
        or not 0.0 <= float(barrier["elapsed_seconds"]) <= VRAM_RELEASE_TIMEOUT_SECONDS
    ):
        raise SceneProcessCollectionError("per-scene VRAM release evidence changed")


def validate_scene_process_evidence_v2(
    result: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, bool]:
    """Purely validate the 64-process partition, barriers, and ordered join."""

    evidence = result.get("scene_process_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema",
        "status",
        "process_reset_equivalence_audit",
        "workers",
        "release_barriers",
        "sequential_launch",
        "join",
    }:
        raise SceneProcessCollectionError("scene-process evidence fields changed")
    scenes = _scene_slices_v2(plan)
    workers = evidence.get("workers")
    barriers = evidence.get("release_barriers")
    worker_fields = {
        "scene_index",
        "role",
        "scene_id",
        "fresh_process",
        "pid",
        "parent_pid",
        "sys_executable",
        "inherited_outer_process_group",
        "parent_process_group_id",
        "child_process_group_id",
        "process_group_equality_observed",
        "prelaunch_baseline_used_bytes",
        "exit_code",
        "execution_seed",
        "full_genesis_seed",
        "effective_genesis_seed",
        "scene_local_mesh_cache",
        "expected_counts",
        "observed_counts",
        "scene_result_binding",
    }
    if (
        evidence.get("schema") != SCENE_EVIDENCE_SCHEMA
        or evidence.get("status") != SCENE_EVIDENCE_STATUS
        or evidence.get("process_reset_equivalence_audit")
        != PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        or not isinstance(workers, list)
        or len(workers) != SCENE_COUNT
        or not isinstance(barriers, list)
        or len(barriers) != SCENE_COUNT
    ):
        raise SceneProcessCollectionError("exact 64-worker evidence changed")
    pids: list[int] = []
    parent_pids: list[int] = []
    scene_result_bindings: list[dict[str, Any]] = []
    expected_python = str(
        Path(plan["execution_contract"]["python_invocation_path"]).resolve(strict=True)
    )
    for scene, worker, barrier in zip(scenes, workers, barriers, strict=True):
        role = str(scene["role"])
        expected_cache_path = (
            f"scenes/{role}/{scene['scene_id']}/derived_meshes"
        )
        cache = worker.get("scene_local_mesh_cache") if isinstance(worker, Mapping) else None
        if (
            not isinstance(worker, Mapping)
            or set(worker) != worker_fields
            or worker.get("scene_index") != scene["scene_index"]
            or worker.get("role") != role
            or worker.get("scene_id") != scene["scene_id"]
            or worker.get("fresh_process") is not True
            or worker.get("inherited_outer_process_group") is not True
            or worker.get("process_group_equality_observed") is not True
            or type(worker.get("parent_process_group_id")) is not int
            or type(worker.get("child_process_group_id")) is not int
            or worker["parent_process_group_id"] <= 1
            or worker["child_process_group_id"]
            != worker["parent_process_group_id"]
            or type(worker.get("prelaunch_baseline_used_bytes")) is not int
            or worker["prelaunch_baseline_used_bytes"] < 0
            or worker.get("exit_code") != 0
            or type(worker.get("pid")) is not int
            or worker["pid"] <= 1
            or type(worker.get("parent_pid")) is not int
            or worker["parent_pid"] <= 1
            or worker.get("sys_executable") != expected_python
            or worker.get("execution_seed") != plan["execution_contract"]["seed"]
            or worker.get("full_genesis_seed") != PLAN_FIRST_PHYSICS_SEED
            or worker.get("effective_genesis_seed")
            != PLAN_FIRST_EFFECTIVE_GENESIS_SEED
            or not isinstance(cache, Mapping)
            or cache.get("path") != expected_cache_path
            or cache.get("cross_scene_reuse_authorized") is not False
            or type(cache.get("mesh_count")) is not int
            or cache["mesh_count"] <= 0
            or not isinstance(cache.get("bindings_identity_sha256"), str)
            or len(cache["bindings_identity_sha256"]) != 64
            or worker.get("expected_counts") != _scene_expected_counts_v2(role)
            or worker.get("observed_counts") != _scene_expected_counts_v2(role)
        ):
            raise SceneProcessCollectionError("scene worker identity/seed changed")
        binding = _binding_shape_v2(
            worker["scene_result_binding"], label="scene result"
        )
        if PurePosixPath(binding["path"]) != PurePosixPath(
            "scene_results", f"{scene['scene_index']:03d}.json"
        ):
            raise SceneProcessCollectionError("scene-result plan order changed")
        pids.append(int(worker["pid"]))
        parent_pids.append(int(worker["parent_pid"]))
        scene_result_bindings.append(binding)
        _validate_release_barrier_shape_v2(
            barrier, scene=scene, worker_pid=int(worker["pid"])
        )
        if (
            barrier["baseline_used_bytes"]
            != worker["prelaunch_baseline_used_bytes"]
        ):
            raise SceneProcessCollectionError(
                "release barrier does not bind its worker pre-launch baseline"
            )
    if len(set(pids)) != SCENE_COUNT or len(set(parent_pids)) != 1:
        raise SceneProcessCollectionError("fresh scene-process identity changed")
    sequential = evidence.get("sequential_launch")
    if sequential != {
        "exact_process_count": SCENE_COUNT,
        "plan_order": True,
        "each_exit_observed_before_release_barrier": True,
        "each_release_barrier_passed_before_next_launch": True,
        "final_release_barrier_passed_before_join": True,
        "adaptive_batching_used": False,
        "fallback_used": False,
        "refill_used": False,
    }:
        raise SceneProcessCollectionError("sequential launch contract changed")
    state_bindings = result.get("state_receipt_bindings")
    render_bindings = result.get("render_receipt_bindings")
    metrics = result.get("scene_metrics")
    result_scene_bindings = result.get("scene_result_bindings")
    join = evidence.get("join")
    join_fields = {
        "exact_join",
        "expected_counts",
        "observed_counts",
        "scene_result_count",
        "state_receipt_count",
        "render_receipt_count",
        "scene_metric_count",
        "ordered_scene_result_binding_identity_sha256",
        "ordered_state_binding_identity_sha256",
        "ordered_render_binding_identity_sha256",
        "plan_ordered_state_ids_sha256",
    }
    if (
        not isinstance(join, Mapping)
        or set(join) != join_fields
        or join.get("exact_join") is not True
        or join.get("expected_counts") != EXPECTED_COUNTS
        or join.get("observed_counts") != EXPECTED_COUNTS
        or join.get("scene_result_count") != SCENE_COUNT
        or join.get("state_receipt_count") != EXPECTED_COUNTS["states"]
        or join.get("render_receipt_count") != EXPECTED_COUNTS["scenes"]
        or join.get("scene_metric_count") != EXPECTED_COUNTS["scenes"]
        or not isinstance(state_bindings, list)
        or len(state_bindings) != EXPECTED_COUNTS["states"]
        or not isinstance(render_bindings, list)
        or len(render_bindings) != EXPECTED_COUNTS["scenes"]
        or not isinstance(metrics, list)
        or len(metrics) != EXPECTED_COUNTS["scenes"]
        or result_scene_bindings != scene_result_bindings
        or join.get("ordered_scene_result_binding_identity_sha256")
        != _ordered_binding_identity_v2(scene_result_bindings)
        or join.get("ordered_state_binding_identity_sha256")
        != _ordered_binding_identity_v2(state_bindings)
        or join.get("ordered_render_binding_identity_sha256")
        != _ordered_binding_identity_v2(render_bindings)
        or join.get("plan_ordered_state_ids_sha256")
        != hashlib.sha256(
            pilot.canonical_json_bytes([str(row["state_id"]) for row in plan["states"]])
        ).hexdigest()
        or result.get("authority_binding") != dict(authority_binding)
        or result.get("plan_binding") != dict(plan_binding)
    ):
        raise SceneProcessCollectionError("exact per-scene join evidence changed")
    expected_state_paths: list[PurePosixPath] = []
    expected_render_paths: list[PurePosixPath] = []
    expected_metrics: list[tuple[str, str]] = []
    for scene in scenes:
        role = str(scene["role"])
        scene_id = str(scene["scene_id"])
        expected_state_paths.extend(
            PurePosixPath(
                "scenes", role, scene_id, "state_receipts", f"{state['state_id']}.json"
            )
            for state in scene["states"]
        )
        expected_render_paths.append(
            PurePosixPath("scenes", role, scene_id, "live_render_receipt.json")
        )
        expected_metrics.append((role, scene_id))
    if (
        [
            PurePosixPath(_binding_shape_v2(row, label="joined state receipt")["path"])
            for row in state_bindings
        ]
        != expected_state_paths
        or [
            PurePosixPath(_binding_shape_v2(row, label="joined render receipt")["path"])
            for row in render_bindings
        ]
        != expected_render_paths
        or [
            (str(row.get("role")), str(row.get("scene_id")))
            for row in metrics
            if isinstance(row, Mapping)
        ]
        != expected_metrics
    ):
        raise SceneProcessCollectionError("joined state/scene plan order changed")
    return {
        "validated": True,
        "workers_exact": True,
        "fixed_seed_exact": True,
        "release_barriers_exact": True,
        "join_exact": True,
    }


def validate_scene_process_closure_v2(
    result: Mapping[str, Any],
    *,
    collection_root: Path,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, bool]:
    """Reopen the complete V2 output and scene-input closure before DINO.

    The pure evidence validator proves metadata structure.  This second pass is
    intentionally filesystem-backed and is called by the runner after it opens
    ``physics_result.json`` but before any state receipt, RGB, DINO, checkpoint,
    or model path.  It closes the collector-to-runner TOCTOU interval.
    """

    evidence_report = validate_scene_process_evidence_v2(
        result,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        plan=plan,
    )
    try:
        root = Path(collection_root).resolve(strict=True)
    except OSError as exc:
        raise SceneProcessCollectionError("collection root is absent at closure audit") from exc
    if root.is_symlink() or not root.is_dir():
        raise SceneProcessCollectionError("collection root changed at closure audit")
    frozen._reject_protected_path(root, label="V2 collection closure")  # noqa: SLF001
    scenes = _scene_slices_v2(plan)
    scene_bindings = result.get("scene_result_bindings")
    state_bindings = result.get("state_receipt_bindings")
    render_bindings = result.get("render_receipt_bindings")
    metrics = result.get("scene_metrics")
    workers = result["scene_process_evidence"]["workers"]
    if (
        not isinstance(scene_bindings, list)
        or len(scene_bindings) != SCENE_COUNT
        or not isinstance(state_bindings, list)
        or len(state_bindings) != EXPECTED_COUNTS["states"]
        or not isinstance(render_bindings, list)
        or len(render_bindings) != EXPECTED_COUNTS["scenes"]
        or not isinstance(metrics, list)
        or len(metrics) != EXPECTED_COUNTS["scenes"]
    ):
        raise SceneProcessCollectionError("output closure counts changed")

    manifest_paths: set[str] = set()
    genesis_paths: set[str] = set()
    for scene in scenes:
        state0 = scene["states"][0]
        for field, paths in (
            ("scene_manifest_binding", manifest_paths),
            ("scene_genesis_binding", genesis_paths),
        ):
            binding = state0.get(field)
            try:
                actual = pilot.require_binding(
                    binding, label=f"closure {field} scene {scene['scene_index']:03d}"
                )
            except (OSError, pilot.PilotContractError) as exc:
                raise SceneProcessCollectionError(
                    "plan scene input binding changed before DINO"
                ) from exc
            if actual != binding:
                raise SceneProcessCollectionError(
                    "plan scene input identity changed before DINO"
                )
            paths.add(str(actual["path"]))
    if len(manifest_paths) != SCENE_COUNT or len(genesis_paths) != SCENE_COUNT:
        raise SceneProcessCollectionError("plan scene input closure is not 64+64")

    for scene, raw_scene_binding, metric, worker in zip(
        scenes, scene_bindings, metrics, workers, strict=True
    ):
        scene_index = int(scene["scene_index"])
        role = str(scene["role"])
        scene_id = str(scene["scene_id"])
        scene_binding = _rehash_relative_binding_v2(
            raw_scene_binding,
            collection_root=root,
            label="pre-DINO scene result",
        )
        selected = root.joinpath(*PurePosixPath(scene_binding["path"]).parts)
        scene_result, actual = pilot.read_bound_json(
            selected,
            expected_sha256=str(scene_binding["file_sha256"]),
            expected_byte_count=int(scene_binding["byte_count"]),
            label=f"pre-DINO scene result {scene_index:03d}",
        )
        state_start = scene_index * STATES_PER_SCENE
        expected_scene_states = state_bindings[
            state_start : state_start + STATES_PER_SCENE
        ]
        if (
            actual["file_sha256"] != scene_binding["file_sha256"]
            or actual["byte_count"] != scene_binding["byte_count"]
            or scene_result.get("schema") != SCENE_RESULT_SCHEMA
            or scene_result.get("status") != SCENE_RESULT_STATUS
            or scene_result.get("attempt_id") != ATTEMPT_ID
            or scene_result.get("scene_index") != scene_index
            or scene_result.get("role") != role
            or scene_result.get("scene_id") != scene_id
            or scene_result.get("plan_binding") != dict(plan_binding)
            or scene_result.get("authority_binding") != dict(authority_binding)
            or scene_result.get("process_reset_equivalence_audit")
            != PROCESS_RESET_EQUIVALENCE_AUDIT_V2
            or scene_result.get("ordered_state_ids")
            != [str(row["state_id"]) for row in scene["states"]]
            or scene_result.get("state_receipt_bindings")
            != expected_scene_states
            or scene_result.get("render_receipt_binding")
            != render_bindings[scene_index]
            or scene_result.get("scene_metric") != metric
            or scene_result.get("scene_local_mesh_cache")
            != worker.get("scene_local_mesh_cache")
            or scene_result.get("failure") is not None
            or scene_result.get("authorizes_retry_or_resume") is not False
            or scene_result.get("allows_refill") is not False
            or scene_result.get("allows_overwrite") is not False
            or scene_result.get("allows_adaptive_batching") is not False
        ):
            raise SceneProcessCollectionError(
                "scene-result lineage changed before DINO"
            )
        for binding in expected_scene_states:
            _rehash_relative_binding_v2(
                binding,
                collection_root=root,
                label="pre-DINO state receipt",
            )
        _rehash_relative_binding_v2(
            render_bindings[scene_index],
            collection_root=root,
            label="pre-DINO render receipt",
        )
        cache_summary = _validate_scene_local_mesh_bindings_v2(
            metric,
            cache_root=root / "scenes" / role / scene_id / "derived_meshes",
            collection_root=root,
        )
        if cache_summary != scene_result.get("scene_local_mesh_cache"):
            raise SceneProcessCollectionError(
                "derived mesh closure changed before DINO"
            )
    if set(evidence_report.values()) != {True}:
        raise SceneProcessCollectionError("pure scene-process evidence did not pass")
    return {
        "validated": True,
        "evidence_validated": True,
        "closure_rehashed": True,
        "scene_results_rehashed": True,
        "state_receipts_rehashed": True,
        "render_receipts_rehashed": True,
        "derived_meshes_rehashed": True,
        "plan_scene_input_bindings_rehashed": True,
    }


def _join_scene_results_v2(
    *,
    collection_root: Path,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    plan_receipt_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    scene_results: Sequence[Mapping[str, Any]],
    scene_result_bindings: Sequence[Mapping[str, Any]],
    worker_receipts: Sequence[Mapping[str, Any]],
    release_barriers: Sequence[Mapping[str, Any]],
    collection_wall_seconds: float,
) -> dict[str, Any]:
    scenes = _scene_slices_v2(plan)
    if not (
        len(scene_results)
        == len(scene_result_bindings)
        == len(worker_receipts)
        == len(release_barriers)
        == SCENE_COUNT
    ):
        raise SceneProcessCollectionError("scene result closure changed")
    # Rehash all 64 indexes and all referenced receipt files at the final join,
    # after the final release barrier and before constructing physics_result.
    for scene, scene_result, scene_binding in zip(
        scenes, scene_results, scene_result_bindings, strict=True
    ):
        _rehash_relative_binding_v2(
            scene_binding,
            collection_root=collection_root,
            label="final scene-result join",
        )
        for binding in scene_result["state_receipt_bindings"]:
            _rehash_relative_binding_v2(
                binding,
                collection_root=collection_root,
                label="final state-receipt join",
            )
        _rehash_relative_binding_v2(
            scene_result["render_receipt_binding"],
            collection_root=collection_root,
            label="final render-receipt join",
        )
        observed_mesh_cache = _validate_scene_local_mesh_bindings_v2(
            scene_result["scene_metric"],
            cache_root=(
                collection_root
                / "scenes"
                / str(scene["role"])
                / str(scene["scene_id"])
                / "derived_meshes"
            ),
            collection_root=collection_root,
        )
        if (
            scene_result.get("scene_index") != scene["scene_index"]
            or scene_result.get("role") != scene["role"]
            or scene_result.get("scene_id") != scene["scene_id"]
            or observed_mesh_cache != scene_result.get("scene_local_mesh_cache")
        ):
            raise SceneProcessCollectionError("final scene-result order changed")
    state_bindings = [
        row for result in scene_results for row in result["state_receipt_bindings"]
    ]
    render_bindings = [result["render_receipt_binding"] for result in scene_results]
    scene_metrics = [result["scene_metric"] for result in scene_results]
    paths = [
        str(row["path"])
        for row in (*scene_result_bindings, *state_bindings, *render_bindings)
    ]
    if (
        len(state_bindings) != EXPECTED_COUNTS["states"]
        or len(render_bindings) != EXPECTED_COUNTS["scenes"]
        or len(scene_metrics) != EXPECTED_COUNTS["scenes"]
        or len(paths) != len(set(paths))
        or sum(int(row["stored_rgb_bytes"]) for row in scene_results)
        > EXPECTED_CAPS["stored_rgb_byte_ceiling"]
        or any(
            row["runtime_versions"] != scene_results[0]["runtime_versions"]
            for row in scene_results[1:]
        )
    ):
        raise SceneProcessCollectionError("combined scene closure changed")
    for name in (
        "native_render_calls",
        "rgb_render_calls",
        "auxiliary_depth_render_calls",
        "stored_rgb_frames",
    ):
        if sum(int(row[name]) for row in scene_metrics) != EXPECTED_CAPS[name]:
            raise SceneProcessCollectionError(f"combined {name} changed")
    evidence = _build_scene_process_evidence_v2(
        plan=plan,
        scene_results=scene_results,
        scene_result_bindings=scene_result_bindings,
        worker_receipts=worker_receipts,
        release_barriers=release_barriers,
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
        "runtime_versions": dict(scene_results[0]["runtime_versions"]),
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
        "process_reset_equivalence_audit": copy.deepcopy(
            PROCESS_RESET_EQUIVALENCE_AUDIT_V2
        ),
        "scene_result_bindings": [dict(row) for row in scene_result_bindings],
        "scene_process_evidence": evidence,
    }
    validate_scene_process_evidence_v2(
        result,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        plan=plan,
    )
    return result


def collect_v2(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], Path]:
    authority, authority_binding, plan, plan_binding = load_and_validate_v2(
        plan_path=plan_path,
        expected_plan_byte_count=expected_plan_byte_count,
        expected_plan_sha256=expected_plan_sha256,
        authority_path=authority_path,
        expected_authority_byte_count=expected_authority_byte_count,
        expected_authority_sha256=expected_authority_sha256,
    )
    started = time.perf_counter()
    nonce = secrets.token_hex(32)
    collection_root, _reservation, reservation_binding = _create_collection_root_v2(
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
    ceiling = int(authority["caps"]["selected_device_vram_byte_ceiling"])
    scenes = _scene_slices_v2(plan)
    scene_results: list[Mapping[str, Any]] = []
    scene_bindings: list[Mapping[str, Any]] = []
    worker_receipts: list[Mapping[str, Any]] = []
    release_barriers: list[Mapping[str, Any]] = []
    for scene in scenes:
        scene_index = int(scene["scene_index"])
        argv = _worker_argv_v2(
            scene_index=scene_index,
            plan_path=plan_path,
            expected_plan_byte_count=expected_plan_byte_count,
            expected_plan_sha256=expected_plan_sha256,
            authority_path=authority_path,
            expected_authority_byte_count=expected_authority_byte_count,
            expected_authority_sha256=expected_authority_sha256,
            reservation_binding=absolute_reservation,
            orchestrator_nonce=nonce,
        )
        worker = _run_worker_process_v2(
            argv,
            scene_index=scene_index,
            role=str(scene["role"]),
            scene_id=str(scene["scene_id"]),
            used_path=used_path,
            ceiling_bytes=ceiling,
        )
        worker_receipts.append(worker)
        barrier = _wait_for_vram_release_v2(
            used_path,
            baseline_used_bytes=int(worker["prelaunch_baseline_used_bytes"]),
            ceiling_bytes=ceiling,
        )
        release_barriers.append(
            _barrier_with_identity_v2(
                barrier, scene=scene, worker_pid=int(worker["pid"])
            )
        )
        _validate_completed_worker_v2(worker)
        scene_result, scene_binding = _load_scene_result_v2(
            collection_root=collection_root,
            scene=scene,
            authority=authority,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            worker_receipt=worker,
        )
        scene_results.append(scene_result)
        scene_bindings.append(scene_binding)
    elapsed = time.perf_counter() - started
    if elapsed > EXPECTED_CAPS["wall_seconds"]:
        raise SceneProcessCollectionError("per-scene collection exceeded global wall cap")
    result = _join_scene_results_v2(
        collection_root=collection_root,
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        plan_receipt_binding=plan_receipt_binding,
        reservation_binding=reservation_binding,
        scene_results=scene_results,
        scene_result_bindings=scene_bindings,
        worker_receipts=worker_receipts,
        release_barriers=release_barriers,
        collection_wall_seconds=elapsed,
    )
    result_path = collection_root / "physics_result.json"
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


# The frozen runner calls ``collector.collect_v1``.  A thin scoped overlay may
# install this module without changing the runner's scientific path.
collect_v1 = collect_v2


def _worker_main(args: argparse.Namespace) -> int:
    authority, authority_binding, plan, plan_binding = load_and_validate_v2(
        plan_path=args.plan,
        expected_plan_byte_count=args.expected_plan_byte_count,
        expected_plan_sha256=args.expected_plan_sha256,
        authority_path=args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
        _collection_reserved=True,
    )
    collection_root = Path(str(authority["collection_root"])).resolve(strict=True)
    _reservation, absolute_reservation = _read_collection_reservation_v2(
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
    result, path = _collect_scene_worker_v2(
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
        reservation_binding=relative_reservation,
        scene_index=args.worker_scene_index,
        orchestrator_pid=args.orchestrator_pid,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "scene_index": args.worker_scene_index,
                "result": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-byte-count", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--worker-scene-index", type=int)
    parser.add_argument("--expected-reservation-byte-count", type=int)
    parser.add_argument("--expected-reservation-sha256")
    parser.add_argument("--orchestrator-nonce")
    parser.add_argument("--orchestrator-pid", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker_scene_index is not None:
            if (
                not 0 <= args.worker_scene_index < SCENE_COUNT
                or args.expected_reservation_byte_count is None
                or args.expected_reservation_sha256 is None
                or args.orchestrator_nonce is None
                or args.orchestrator_pid is None
            ):
                raise SceneProcessCollectionError("worker reservation/index pins are incomplete")
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
            raise SceneProcessCollectionError("parent invocation contains worker-only pins")
        result, path = collect_v2(
            plan_path=args.plan,
            expected_plan_byte_count=args.expected_plan_byte_count,
            expected_plan_sha256=args.expected_plan_sha256,
            authority_path=args.authority,
            expected_authority_byte_count=args.expected_authority_byte_count,
            expected_authority_sha256=args.expected_authority_sha256,
        )
        print(
            json.dumps(
                {"status": result["status"], "physics_result": str(path)},
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_ID",
    "AUTHORITY_FIELDS",
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "PLAN_FIRST_EFFECTIVE_GENESIS_SEED",
    "PLAN_FIRST_PHYSICS_SEED",
    "PROCESS_RESET_EQUIVALENCE_AUDIT_V2",
    "ROLE_ORDER",
    "SCENE_COUNT",
    "SCENE_EVIDENCE_SCHEMA",
    "SCENE_EVIDENCE_STATUS",
    "SceneProcessCollectionError",
    "_validate_scene_diversity_plan_v1",
    "collect_v1",
    "collect_v2",
    "load_and_validate_replacement_v2",
    "load_and_validate_v2",
    "validate_scene_process_closure_v2",
    "validate_scene_process_evidence_v2",
]
