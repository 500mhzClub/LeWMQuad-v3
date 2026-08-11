#!/usr/bin/env python3
"""Branch corpora for the v1.2 utility scorer and the final evaluation.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  No predictor checkpoint is opened here.

Two pools, one generator:

* ``scorer_fit``  120 states (15/family), 6 candidates each  -> 720 branches
* ``final_eval``  200 states (25/family), 12 candidates each -> 2400 branches

The snapshot mechanism, the canonical boundary, the candidate bank and oracle
v1.2 are imported unchanged.  What is new here is only the corpus: state
selection with strata, the frozen candidate rotation, the proprioceptive and
control history, and the textured_v03 renders at the three context slots and the
four horizons.

Stages::

    --stage states    resolve + freeze the identities (no branch, no render)
    --stage branches  execute, render and label every allocated branch
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1 as V1
import run_go2_oracle_branch_pilot_v1_2 as V12
from lewm.oracle.go2_branch_oracle_v1_2 import (
    GeodesicField,
    progress_digest, safety_digest, oracle_digest as v12_oracle_digest,
)
from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC
from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as INVALID_IDS
from lewm.oracle.go2_textured_v03_renderer import (
    BasePose,
    TexturedV03Renderer,
    capture_base_pose,
    renderer_contract_digest as textured_v03_renderer_contract_digest,
)
from lewm.oracle.go2_scorer_contract_v1_2 import (
    CORPUS_SELECTION_CONTRACT,
    SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
    TARGET_ENCODER,
    clean_source_binding,
    contract as scorer_contract,
    contract_digest as scorer_contract_digest,
    preprocess_contract_digest,
    render_contract_digest,
    target_encoder_digest,
)
from scripts import dev_action_slew_reconstruction_v1 as SLEW
from scripts import build_dev_v03_proprio_action_manifest_v1 as M

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
CORPUS = V1.CORPUS
SPLITS = ("train", "val", "test_id", "test_hard")

CONTEXT_SLOTS = 3
SAMPLES_PER_SLOT = M.SAMPLES_PER_SLOT           # 5
PROPRIO_HISTORY = CONTEXT_SLOTS * SAMPLES_PER_SLOT   # 15 trailing 10 Hz samples
HORIZONS = 4
EXPECTED_FAMILIES = 8
FACTORIAL_ROOT = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1"
)
FACTORIAL_MANIFEST = FACTORIAL_ROOT / "factorial_manifest.json"
FACTORIAL_MANIFEST_FILE_SHA256 = (
    "8bf59020d24e02fdb11948f3732220df839aa1c3bc8612392ce6baab6b8d629c"
)
FACTORIAL_MANIFEST_DIGEST = (
    "6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0"
)
FACTORIAL_ROWS = FACTORIAL_ROOT / "proprio_rows.jsonl"
FACTORIAL_ROWS_SHA256 = (
    "7b79d12830f12175c591a87982a20e5df7a8d64cfc40e99dd9cee2dc1ae2543e"
)
V11_IDENTITY_MANIFEST_DIGEST = (
    "015eb0bb4ccb9da28ce4b055771975fc68ac0c986e462d9c3af0a61ef45a9ea2"
)
V12_IDENTITY_MANIFEST_DIGEST = (
    "5f380bf7f49ef10437c7d9644f04dbef065f0550dfd30d0ec36208cda25d08cf"
)

# ---- frozen strata (scorer-fit only), snapshot-time geometry only -------------
STRATA = ("general", "safety_enriched", "completion_enriched")
SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M = 0.10
COMPLETION_ENRICHED_MAX_GEODESIC_M = 0.75
COMPLETION_ENRICHED_MAX_BEARING_RAD = math.radians(75.0)

POOLS = {
    "scorer_fit": {
        "states_per_family": 15, "candidates_per_state": 6,
        "strata": {"general": 5, "safety_enriched": 5, "completion_enriched": 5},
        "calibration_per_stratum_per_family": 1,
    },
    "final_eval": {
        "states_per_family": 25, "candidates_per_state": 12,
        "strata": None,
        "calibration_per_stratum_per_family": 0,
    },
}

SELECTION = dict(CORPUS_SELECTION_CONTRACT)
WARMUP_BLOCKS_MIN, WARMUP_BLOCKS_MAX = SELECTION["warmup_blocks"]
PRE_IDENTITY_VALIDATION_NAME = "pre_identity_allocation_validation.json"
LAUNCH_RECEIPT_NAME = "clean_source_launch_receipt.json"
SCORER_CONTRACT_ARTIFACT_PATH = (
    ROOT / ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json"
)
LAUNCH_BINDING_KEYS = (
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
)


def selection_digest() -> str:
    # The scorer contract, not a mutually generated artifact, freezes selection.
    return str(scorer_contract()["corpus_selection_digest"])


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(V1._jsonable(payload), sort_keys=True).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(V1._jsonable(payload), indent=2,
                                    sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _load_pre_identity_allocation_validation() -> dict[str, Any]:
    path = OUT_ROOT / "scorer_fit" / PRE_IDENTITY_VALIDATION_NAME
    if not path.is_file():
        raise RuntimeError(
            "state identity selection is gated on the frozen pre-identity "
            "allocation validation artifact"
        )
    artifact = json.loads(path.read_text())
    ALLOC.validate_pre_identity_structural_validation(artifact)
    return artifact


def _load_issued_scorer_contract() -> dict[str, Any]:
    if not SCORER_CONTRACT_ARTIFACT_PATH.is_file():
        raise RuntimeError("clean-source scorer contract must be issued before preflight")
    artifact = json.loads(SCORER_CONTRACT_ARTIFACT_PATH.read_text())
    _verify_self_digest(artifact, "contract_artifact_digest", "scorer contract artifact")
    if (artifact.get("complete") is not True
            or artifact.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or artifact.get("source_repository_clean") is not True):
        raise RuntimeError("issued scorer contract is not the current clean-source contract")
    current_source = clean_source_binding()
    if (artifact.get("clean_source_binding") != current_source
            or artifact.get("clean_source_binding_digest")
            != canonical_digest(current_source)):
        raise RuntimeError("issued scorer contract source binding differs from current HEAD")
    return artifact


def _build_clean_source_launch_receipt(
        pre_identity: dict[str, Any]) -> dict[str, Any]:
    scorer_artifact = _load_issued_scorer_contract()
    source = scorer_artifact["clean_source_binding"]
    receipt = {
        "schema": "go2_utility_scorer_v1_2_clean_source_launch_receipt",
        "status": STATUS,
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "source_repository_clean": True,
        "clean_source_binding_digest":
            scorer_artifact["clean_source_binding_digest"],
        "bound_implementations_digest": source["bound_implementations_digest"],
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "scorer_contract_artifact_digest":
            scorer_artifact["contract_artifact_digest"],
        "scorer_contract_artifact_sha256":
            file_sha256(SCORER_CONTRACT_ARTIFACT_PATH),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        "pre_identity_allocation_validation_digest":
            pre_identity["pre_identity_validation_digest"],
    }
    receipt["clean_source_launch_receipt_digest"] = canonical_digest(receipt)
    return receipt


def _load_clean_source_launch_receipt() -> dict[str, Any]:
    path = OUT_ROOT / "scorer_fit" / LAUNCH_RECEIPT_NAME
    if not path.is_file():
        raise RuntimeError("state identity selection requires a clean-source launch receipt")
    receipt = json.loads(path.read_text())
    _verify_self_digest(
        receipt, "clean_source_launch_receipt_digest", "clean-source launch receipt")
    expected = _build_clean_source_launch_receipt(
        _load_pre_identity_allocation_validation())
    if receipt != expected:
        raise RuntimeError("clean-source launch receipt differs from current clean HEAD")
    return receipt


def issue_pre_identity_allocation_validation(out: Path) -> int:
    """Issue the deterministic structural table before any state identity."""

    if out.name != "scorer_fit":
        raise RuntimeError("allocation preflight is defined only for scorer_fit")
    # Verifies clean git HEAD, exact bound source bytes and the issued contract
    # before any pre-identity artifact is retained or created.
    _load_issued_scorer_contract()
    amendment_path = ROOT / ALLOC.AMENDMENT_ARTIFACT_PATH
    ALLOC.validate_allocation_amendment_artifact(
        json.loads(amendment_path.read_text())
    )
    artifact = ALLOC.build_pre_identity_structural_validation()
    path = out / PRE_IDENTITY_VALIDATION_NAME
    retained = False
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            ALLOC.validate_pre_identity_structural_validation(existing)
            if existing == artifact:
                retained = True
        except Exception:
            pass
        if not retained and _outcome_generation_started(out):
            raise RuntimeError(
                "pre-identity allocation validation changed after outcomes started"
            )
        if not retained:
            _preserve_invalid(path, out, "pre-identity-validation-mismatch")
    if not retained:
        atomic_json(path, artifact)

    launch = _build_clean_source_launch_receipt(artifact)
    launch_path = out / LAUNCH_RECEIPT_NAME
    if launch_path.is_file():
        existing_launch = json.loads(launch_path.read_text())
        if existing_launch != launch:
            if _outcome_generation_started(out):
                raise RuntimeError("clean-source launch binding changed after outcomes started")
            _preserve_invalid(launch_path, out, "clean-source-launch-mismatch")
    atomic_json(launch_path, launch)
    print(json.dumps({
        "recovery": ("retained_valid_pre_identity_validation" if retained
                     else "issued_pre_identity_validation"),
        "path": str(path),
        "clean_source_launch_receipt_path": str(launch_path),
        "clean_source_launch_receipt_digest":
            launch["clean_source_launch_receipt_digest"],
        "source_repository_commit": launch["source_repository_commit"],
        "pre_identity_validation_digest":
            artifact["pre_identity_validation_digest"],
        "state_slots": artifact["global"]["state_slot_count"],
        "candidate_slots": artifact["global"]["candidate_slot_count"],
        "goal_type_validation_status":
            artifact["goal_type_validation"]["status"],
    }, indent=2, sort_keys=True))
    return 0


def _verify_self_digest(payload: dict[str, Any], key: str, label: str) -> None:
    expected = canonical_digest({name: value for name, value in payload.items()
                                 if name != key})
    if payload.get(key) != expected:
        raise RuntimeError(f"{label} self digest mismatch")


def _preserve_invalid(path: Path, out: Path, reason: str) -> Path:
    invalid_root = out / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path) if path.is_file() else "not-a-file"
    target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.invalid"
    counter = 0
    while target.exists():
        counter += 1
        target = invalid_root / (
            f"{path.name}.{digest[:16]}.{reason}.{counter}.invalid")
    path.rename(target)
    return target


def _factorial_scene_exclusions() -> tuple[set[str], dict[str, Any]]:
    if file_sha256(FACTORIAL_MANIFEST) != FACTORIAL_MANIFEST_FILE_SHA256:
        raise RuntimeError("frozen factorial manifest file digest changed")
    if file_sha256(FACTORIAL_ROWS) != FACTORIAL_ROWS_SHA256:
        raise RuntimeError("frozen factorial base-row digest changed")
    factorial = json.loads(FACTORIAL_MANIFEST.read_text())
    _verify_self_digest(factorial, "digest", "factorial manifest")
    if factorial["digest"] != FACTORIAL_MANIFEST_DIGEST:
        raise RuntimeError("frozen factorial manifest identity changed")
    if factorial.get("base_manifest_rows_sha256") != FACTORIAL_ROWS_SHA256:
        raise RuntimeError("factorial manifest does not bind the expected base rows")
    base_rows = [json.loads(line) for line in FACTORIAL_ROWS.read_text().splitlines()
                 if line.strip()]
    scenes: set[str] = set()
    for row in factorial["rows"]:
        index = int(row["manifest_row_index"])
        if not 0 <= index < len(base_rows):
            raise RuntimeError("factorial manifest row index is out of bounds")
        scenes.add(str(base_rows[index]["scene"]))
    if len(scenes) != 80:
        raise RuntimeError(f"expected 80 factorial scenes, recovered {len(scenes)}")
    binding = {
        "factorial_manifest_digest": FACTORIAL_MANIFEST_DIGEST,
        "factorial_manifest_file_sha256": FACTORIAL_MANIFEST_FILE_SHA256,
        "factorial_rows_sha256": FACTORIAL_ROWS_SHA256,
        "scene_count": 80,
        "scene_ids_digest": canonical_digest(sorted(scenes)),
    }
    return scenes, binding


def _pilot_scene_exclusions() -> tuple[set[str], dict[str, Any]]:
    v11_path = V1.OUT_DIR / "identity_manifest.json"
    v11 = json.loads(v11_path.read_text())
    _verify_self_digest(v11, "identity_manifest_digest", "oracle-v1.1 identity manifest")
    if v11["identity_manifest_digest"] != V11_IDENTITY_MANIFEST_DIGEST:
        raise RuntimeError("oracle-v1.1 pilot identity manifest changed")
    v12_path = V12.OUT_DIR / "state_manifest.json"
    v12 = json.loads(v12_path.read_text())
    _verify_self_digest(v12, "state_manifest_digest", "oracle-v1.2 state manifest")
    if v12["state_manifest_digest"] != V12_IDENTITY_MANIFEST_DIGEST:
        raise RuntimeError("oracle-v1.2 pilot identity manifest changed")
    scenes = ({str(row["scene_id"]) for row in v11["pilot_states"]}
              | {str(row["scene_id"]) for row in v11["replay_states"]}
              | {str(row["scene_id"]) for row in v12["states"]})
    binding = {
        "oracle_v1_1_identity_manifest_digest": V11_IDENTITY_MANIFEST_DIGEST,
        "oracle_v1_2_identity_manifest_digest": V12_IDENTITY_MANIFEST_DIGEST,
        "scene_count": len(scenes),
        "scene_ids_digest": canonical_digest(sorted(scenes)),
    }
    return scenes, binding


def _state_identity_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items()
            if key not in {"state_identity_digest", "state_index",
                           "candidate_indices", "candidate_rotation_index",
                           "branch_identities"}}


def _state_identity_digest(state: dict[str, Any]) -> str:
    return canonical_digest({
        "schema": "go2_branch_state_identity_v1_2",
        "selection_digest": selection_digest(),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "state": _state_identity_payload(state),
    })


# ------------------------------------------------------------------ rendering --
def write_png_atomic(array: np.ndarray, path: Path, out: Path) -> tuple[str, int]:
    """Write an exact 224-square PNG without overwriting a differing artifact."""

    from PIL import Image
    image = np.asarray(array)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"invalid historical RGB array {image.shape}/{image.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    Image.fromarray(image).save(temporary, format="PNG")
    digest = file_sha256(temporary)
    byte_count = temporary.stat().st_size
    if path.exists():
        if (path.is_file() and path.stat().st_size == byte_count
                and file_sha256(path) == digest):
            temporary.unlink()
            return digest, byte_count
        _preserve_invalid(path, out, "frame-mismatch")
    os.replace(temporary, path)
    return digest, byte_count


# ------------------------------------------------------------- proprioception --
def proprio_sample(ctx: V1.BranchContext) -> list[float]:
    """The frozen 30-D sensed-state vector, in the corpus channel order."""

    from lewm_genesis.rollout import _roll_from_quat_wxyz, _pitch_from_quat_wxyz
    runner = ctx.runner
    robot = ctx.build.robot
    quat_wxyz = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64)
    ang_world = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64)
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz[None, :]
    if ang_world.ndim == 1:
        ang_world = ang_world[None, :]
    qw, qx, qy, qz = (float(v) for v in quat_wxyz[0])
    quat_xyzw = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    gyro = runner._world_to_body(ang_world[0], quat_xyzw)
    gravity = M.projected_gravity(_roll_from_quat_wxyz(qw, qx, qy, qz),
                                  _pitch_from_quat_wxyz(qw, qx, qy, qz))
    feature = [g - o for g, o in zip(gravity, M.GRAVITY_OFFSET)]
    joint_pos = np.asarray(runner._as_np(
        robot.get_dofs_position(runner._leg_dof_idx.tolist())), dtype=np.float64)
    joint_vel = np.asarray(runner._as_np(
        robot.get_dofs_velocity(runner._leg_dof_idx.tolist())), dtype=np.float64)
    if joint_pos.ndim == 2:
        joint_pos, joint_vel = joint_pos[0], joint_vel[0]
    return ([float(v) for v in feature] + [float(v) for v in gyro]
            + [float(v) for v in joint_pos] + [float(v) for v in joint_vel])


def control_sample(previous_applied: Sequence[float]) -> list[float]:
    """Efference copy: the applied command at the tick BEFORE this sample."""

    return [float(previous_applied[c]) for c in SLEW.ACTIVE_CHANNELS]


def action_block_10d(executed_block: np.ndarray) -> list[float]:
    """The frozen 10-D five-tick post-slew action for one block."""

    return SLEW.flatten([[float(v) for v in tick] for tick in executed_block])


# ------------------------------------------------------------------- driving ---
def drive_block_with_probe(ctx: V1.BranchContext,
                           probe: Callable[[int, Sequence[float]], None]) -> Any:
    """V1.drive_one_block, with a per-command-tick probe.

    Replicated rather than patched so the frozen driver stays untouched.
    """

    runner = ctx.runner
    requested, _choices = runner._collect_block()
    executed = np.asarray(
        runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
        dtype=np.float64)
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    previous = np.asarray(runner._last_executed, dtype=np.float64)[0].copy()
    carry = {"prev": previous}

    def after_policy_step(tick_idx: int, step_idx: int) -> None:
        if step_idx != steps_per_tick - 1:
            return
        probe(int(tick_idx), carry["prev"])
        carry["prev"] = executed[0, tick_idx].copy()

    block = runner.execute_requested_block(requested,
                                           after_policy_step=after_policy_step)
    for _ in range(runner._block_size):
        for state in runner.episode_states:
            state.step()
    runner._blocks_in_episode += 1
    ctx.ticks_executed += runner._block_size
    ctx.episode_ticks += runner._block_size
    ctx.policy_steps += runner._block_size * steps_per_tick
    ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()

    before = int(runner.episode_states[0].reset_count)
    from lewm_genesis import ros_msg_adapter as adapter
    runner._check_and_reset_fallen_envs(V1._NullWriter(), adapter)
    runner._check_and_reset_completed_envs(V1._NullWriter(), adapter)
    after = int(runner.episode_states[0].reset_count)
    ctx.reset_in_last_block = after != before
    if ctx.reset_in_last_block:
        ctx.episode_ticks = 0
        ctx.last_block_executed = None
        ctx.episode_start_reset_count = after
    return block


# ------------------------------------------------------------------ eligible --
def landmark_bearing_range(ctx: V1.BranchContext, landmark_xy: Sequence[float]
                           ) -> tuple[float, float]:
    """The planning-time observable goal binding: body bearing and range."""

    from lewm_worlds.scene_graph import wrap_angle_pi
    (x, y), yaw, _z = ctx.pose()
    dx = float(landmark_xy[0]) - x
    dy = float(landmark_xy[1]) - y
    return float(wrap_angle_pi(math.atan2(dy, dx) - yaw)), float(math.hypot(dx, dy))


_FIELD_CACHE: dict[tuple[int, int], GeodesicField] = {}


def geodesic_field(ctx: V1.BranchContext, landmark_cell: int,
                   blocked: frozenset) -> GeodesicField:
    """Dijkstra depends only on the scene and the goal, so cache it per scene."""

    key = (id(ctx.scene_graph), int(landmark_cell))
    field = _FIELD_CACHE.get(key)
    if field is None:
        field = GeodesicField(ctx.scene_graph, int(landmark_cell),
                              transit_blocked=blocked)
        _FIELD_CACHE[key] = field
    return field


def classify_state(ctx: V1.BranchContext, topology: dict[str, Any]
                   ) -> tuple[dict[str, Any], GeodesicField, set[str]] | str:
    """Common eligibility plus the strata this state qualifies for."""

    try:
        boundary = V1.assert_canonical_boundary(ctx)
    except V1.BoundaryRefused as exc:
        return f"boundary_refused: {str(exc)[:50]}"
    if ctx.episode_ticks < PROPRIO_HISTORY - 1:
        return "insufficient_proprioceptive_history"
    (x, y), _yaw, _z = ctx.pose()
    hit = ctx.scene_graph.locate((x, y))
    if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
        return "locate_distance_gt_2m"
    if V12._contact_count(ctx, topology) > 0:
        return "already_in_disallowed_contact"

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    cell = int(hit.cell_id)
    best: tuple[float, str, int, int, GeodesicField] | None = None
    for name, landmark_cell in sorted(graph.landmark_cells, key=lambda kv: str(kv[0])):
        hops = graph.bfs_distance(cell, int(landmark_cell), transit_blocked=blocked)
        if hops is None or int(hops) < 1:
            continue
        field = geodesic_field(ctx, int(landmark_cell), blocked)
        distance = field.remaining_distance((x, y), cell)
        if not math.isfinite(distance):
            continue
        key = (float(distance), str(name), int(landmark_cell), int(hops))
        if best is None or key < best[:4]:
            best = (*key, field)
    if best is None:
        return "no_reachable_landmark"
    distance, name, landmark_cell, hops, field = best

    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    (_x, _y), yaw, _z2 = ctx.pose()
    body_clearance = float(_body_probe_configuration_clearance_m(
        ctx.grid, [x, y], yaw,
        body_forward_m=V1.CONTACT_BODY_FORWARD_M,
        body_half_width_m=V1.CONTACT_BODY_HALF_WIDTH_M,
        body_probe_margin_m=V1.CONTACT_BODY_PROBE_MARGIN_M))
    centre = graph.cell_center(int(landmark_cell))
    bearing, range_m = landmark_bearing_range(ctx, centre)
    materials = sorted({str(obj.material_id) for obj in ctx.pack.static_objects
                        if str(obj.object_id) == str(name)})
    if len(materials) != 1 or not materials[0].startswith("landmark_"):
        return "bound_landmark_material_missing_or_ambiguous"
    goal_type = materials[0]

    strata: set[str] = set()
    if hops >= 2:
        strata.add("general")
        if body_clearance <= SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M:
            strata.add("safety_enriched")
    if (distance <= COMPLETION_ENRICHED_MAX_GEODESIC_M
            and abs(bearing) <= COMPLETION_ENRICHED_MAX_BEARING_RAD):
        strata.add("completion_enriched")
    if not strata:
        return "no_stratum"

    record = {
        "boundary": boundary, "cell_id": cell,
        "goal": {"landmark_id": name, "landmark_cell": int(landmark_cell),
                 "material_id": goal_type,
                 "graph_edges": int(hops), "start_geodesic_m": float(distance),
                 "bearing_body_rad": float(bearing), "range_m": float(range_m),
                 "landmark_xy_m": [float(centre[0]), float(centre[1])]},
        "body_clearance_m": body_clearance,
        "clearance_m": float(graph.clearance_to_walls((x, y))),
    }
    return record, field, strata


# ------------------------------------------------------------------- stage A --
def scene_pool(pool_name: str) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    factorial_scenes, factorial_binding = _factorial_scene_exclusions()
    pilot_scenes, pilot_binding = _pilot_scene_exclusions()
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    excluded = (set(factorial_scenes) | set(pilot_scenes)
                | set(invalid_identity_index.scene_ids))
    scorer_binding: dict[str, Any] | None = None
    if pool_name == "final_eval":
        scorer_path = OUT_ROOT / "scorer_fit/state_manifest.json"
        scorer_manifest = json.loads(scorer_path.read_text())
        _verify_self_digest(scorer_manifest, "state_manifest_digest",
                            "scorer-fit state manifest")
        if (scorer_manifest.get("pool") != "scorer_fit"
                or len(scorer_manifest.get("states", [])) != 120
                or scorer_manifest.get("scorer_contract_v1_2_digest")
                != scorer_contract_digest()):
            raise RuntimeError("final selection requires the complete current scorer-fit identity manifest")
        scorer_scenes = {str(row["scene_id"]) for row in scorer_manifest["states"]}
        if len(scorer_scenes) != 120:
            raise RuntimeError("scorer-fit identity manifest is not scene-disjoint")
        excluded |= scorer_scenes
        scorer_binding = {
            "state_manifest_digest": scorer_manifest["state_manifest_digest"],
            "scene_count": len(scorer_scenes),
            "scene_ids_digest": canonical_digest(sorted(scorer_scenes)),
        }

    families: dict[str, list[Path]] = {}
    for split in SPLITS:
        root = CORPUS / split
        if not root.is_dir():
            continue
        for family_dir in sorted(root.iterdir()):
            if not family_dir.is_dir():
                continue
            for scene_dir in sorted(family_dir.iterdir()):
                if scene_dir.name in excluded:
                    continue
                if not (scene_dir / "manifest.json").is_file():
                    continue
                families.setdefault(family_dir.name, []).append(scene_dir)
    for family in families:
        families[family].sort(key=lambda path: path.name)
    if len(families) != EXPECTED_FAMILIES:
        raise RuntimeError(f"expected eight scene families, found {sorted(families)}")
    allow_list = {family: [path.name for path in paths]
                  for family, paths in sorted(families.items())}
    exclusion_binding = {
        "factorial": factorial_binding,
        "oracle_pilots": pilot_binding,
        "invalid_scorer_identity_attempt": invalid_identity_index.binding(),
        "scorer_fit": scorer_binding,
        "excluded_scene_count": len(excluded),
        "excluded_scene_ids_digest": canonical_digest(sorted(excluded)),
        "allow_list_scene_count": sum(len(values) for values in allow_list.values()),
        "allow_list_digest": canonical_digest(allow_list),
    }
    return families, exclusion_binding


def _state_shard_bindings(args: argparse.Namespace, exclusion: dict[str, Any],
                          family_allow_list: list[str]) -> dict[str, Any]:
    scorer = scorer_contract()
    pre_identity = _load_pre_identity_allocation_validation()
    launch = _load_clean_source_launch_receipt()
    return {
        "selection_digest": selection_digest(),
        "scorer_fit_allocation_design_digest":
            scorer["scorer_fit_allocation_design_digest"],
        "candidate_allocator_contract_digest": ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            ALLOC.allocation_amendment_digest(),
        "pre_identity_allocation_validation_digest":
            pre_identity["pre_identity_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            INVALID_IDS.invalid_identity_exclusion_digest(),
        "clean_source_launch_receipt_digest":
            launch["clean_source_launch_receipt_digest"],
        "source_repository_commit": launch["source_repository_commit"],
        "clean_source_binding_digest": launch["clean_source_binding_digest"],
        "bound_implementations_digest": launch["bound_implementations_digest"],
        "scorer_contract_artifact_digest":
            launch["scorer_contract_artifact_digest"],
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": v12_oracle_digest(),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "boundary_digest": V1.BOUNDARY_DIGEST,
        "render_contract_digest": render_contract_digest(),
        "textured_v03_renderer_contract_digest":
            textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": preprocess_contract_digest(),
        "preprocessing_digest": TARGET_ENCODER["preprocessing_identity_sha256"],
        "target_encoder_digest": target_encoder_digest(),
        "target_encoder_checkpoint_sha256": TARGET_ENCODER["checkpoint_sha256"],
        "genesis_backend": args.backend,
        "exclusion_binding": exclusion,
        "family_allow_list_digest": canonical_digest(family_allow_list),
    }


def resolve_states(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend != "cpu":
        raise RuntimeError("the frozen branch backend is cpu")
    # This must happen before scene discovery or simulator construction.
    _load_clean_source_launch_receipt()
    spec = POOLS[args.pool]
    pool, exclusion = scene_pool(args.pool)
    if args.family is None:
        raise RuntimeError("state identity resolution must be sharded by one family")
    if args.family not in pool:
        raise RuntimeError(f"unknown family {args.family!r}")
    family = args.family
    scenes = pool[family]
    shared = V1._load_shared(args.backend)
    states: list[dict[str, Any]] = []
    rejections: dict[str, dict[str, int]] = {}
    need = ({"evaluation": spec["states_per_family"]}
            if spec["strata"] is None else dict(spec["strata"]))
    found = {key: 0 for key in need}

    for scene_dir in scenes:
        if all(found[key] >= need[key] for key in need):
            break
        seed = V1._drive_seed(scene_dir.name)
        ctx = V1.build_context(scene_dir, seed=seed, backend=args.backend,
                               shared=shared)
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        reasons: dict[str, int] = {}
        chosen: dict[str, Any] | None = None
        for block_idx in range(WARMUP_BLOCKS_MAX):
            ctx.drive_one_block()
            if block_idx + 1 < WARMUP_BLOCKS_MIN:
                continue
            verdict = classify_state(ctx, topology)
            if isinstance(verdict, str):
                key = verdict.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            record, _field, strata = verdict
            if spec["strata"] is None:
                wanted = (["evaluation"] if found["evaluation"] < need["evaluation"]
                          else [])
            else:
                wanted = [name for name in STRATA
                          if name in strata and found[name] < need[name]]
            if not wanted:
                reasons["stratum_already_full"] = reasons.get(
                    "stratum_already_full", 0) + 1
                continue
            stratum = wanted[0]
            ordinal = found[stratum]
            split_role = ("evaluation" if args.pool == "final_eval"
                          else ("calibration" if ordinal == 0 else "fit"))
            scene_manifest_path = scene_dir / "manifest.json"
            state_id = f"{args.pool}-{family}-{stratum}-{ordinal:02d}"
            chosen = {
                "state_id": state_id,
                "family": family,
                "scene_id": scene_dir.name,
                "scene_dir": str(scene_dir.resolve()),
                "scene_manifest_sha256": file_sha256(scene_manifest_path),
                "scene_manifest_byte_count": scene_manifest_path.stat().st_size,
                "split": scene_dir.parent.parent.name,
                "drive_seed": int(seed),
                "stratum": stratum,
                "split_role": split_role,
                "warmup_blocks": block_idx + 1,
                "source_step": int(record["boundary"]["source_step"]),
                "episode_id": int(ctx.runner.episode_states[0].episode_id),
                "episode_cluster_id": (
                    f"{scene_dir.name}/env0/ep"
                    f"{int(ctx.runner.episode_states[0].episode_id)}"
                ),
                "cell_id": int(record["cell_id"]),
                "boundary": record["boundary"],
                "goal": record["goal"],
                "goal_type": record["goal"]["material_id"],
                "body_clearance_m": float(record["body_clearance_m"]),
                "clearance_m": float(record["clearance_m"]),
            }
            chosen["state_identity_digest"] = _state_identity_digest(chosen)
            found[stratum] += 1
            break
        rejections[scene_dir.name] = reasons
        _FIELD_CACHE.clear()
        del ctx
        gc.collect()
        if chosen is not None:
            states.append(chosen)
            print(f"[states] {args.pool} {family[:22]:22s} {chosen['stratum'][:20]:20s} "
                  f"{scene_dir.name} blocks={chosen['warmup_blocks']} "
                  f"edges={chosen['goal']['graph_edges']} "
                  f"d0={chosen['goal']['start_geodesic_m']:.2f}m", flush=True)

    incomplete = {key: [found[key], need[key]] for key in need
                  if found[key] != need[key]}
    if incomplete:
        raise RuntimeError(f"could not resolve frozen state quota for {family}: {incomplete}")
    states.sort(key=lambda state: (
        STRATA.index(state["stratum"]) if state["stratum"] in STRATA else 0,
        state["scene_id"],
    ))
    if len(states) != spec["states_per_family"]:
        raise RuntimeError("state shard count mismatch")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError("state shard reuses a scene")
    INVALID_IDS.assert_disjoint(states, label=f"{family} state shard")
    bindings = _state_shard_bindings(args, exclusion, [path.name for path in scenes])
    shard = {
        "schema": "go2_branch_corpus_v1_2_state_shard",
        "status": STATUS,
        "complete": True,
        "pool": args.pool,
        "family": family,
        "spec": spec,
        "selection": SELECTION,
        **bindings,
        "states": states,
        "scene_rejection_reasons": rejections,
    }
    shard["state_shard_digest"] = canonical_digest(shard)
    return shard


# ------------------------------------------------------------------- stage B --
def _validate_state_manifest(manifest: dict[str, Any], pool: str) -> None:
    _verify_self_digest(manifest, "state_manifest_digest", "state manifest")
    expected_states = EXPECTED_FAMILIES * POOLS[pool]["states_per_family"]
    expected_branches = expected_states * POOLS[pool]["candidates_per_state"]
    if (manifest.get("schema") != "go2_branch_corpus_v1_2_state_manifest"
            or manifest.get("complete") is not True
            or manifest.get("pool") != pool
            or manifest.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or manifest.get("selection_digest") != selection_digest()
            or manifest.get("candidate_allocator_contract_digest")
            != ALLOC.allocation_contract_digest()
            or manifest.get("candidate_allocation_amendment_digest")
            != ALLOC.allocation_amendment_digest()
            or manifest.get("invalid_scorer_identity_exclusion_digest")
            != INVALID_IDS.invalid_identity_exclusion_digest()
            or manifest.get("pre_identity_allocation_validation_digest")
            != _load_pre_identity_allocation_validation()[
                "pre_identity_validation_digest"]
            or manifest.get("candidate_bank_digest") != V1.bank_digest()
            or manifest.get("progress_contract_digest") != progress_digest()
            or manifest.get("safety_contract_digest") != safety_digest()
            or manifest.get("oracle_v1_2_digest") != v12_oracle_digest()
            or manifest.get("render_contract_digest") != render_contract_digest()
            or manifest.get("textured_v03_renderer_contract_digest")
            != textured_v03_renderer_contract_digest()
            or manifest.get("preprocess_contract_digest")
            != preprocess_contract_digest()
            or manifest.get("preprocessing_digest")
            != TARGET_ENCODER["preprocessing_identity_sha256"]
            or manifest.get("target_encoder_digest") != target_encoder_digest()
            or manifest.get("target_encoder_checkpoint_sha256")
            != TARGET_ENCODER["checkpoint_sha256"]
            or manifest.get("boundary_digest") != V1.BOUNDARY_DIGEST
            or manifest.get("genesis_backend") != "cpu"
            or len(manifest.get("states", [])) != expected_states
            or manifest.get("attempted_branch_count_registered") != expected_branches):
        raise RuntimeError("state manifest is incomplete or bound to another contract")
    launch = _load_clean_source_launch_receipt()
    for key in LAUNCH_BINDING_KEYS:
        if manifest.get(key) != launch[key]:
            raise RuntimeError(f"state manifest clean-source binding mismatch: {key}")
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    if manifest.get("exclusion_binding", {}).get(
            "invalid_scorer_identity_attempt") != invalid_identity_index.binding():
        raise RuntimeError("state manifest invalid45 exclusion binding mismatch")
    INVALID_IDS.assert_disjoint(
        manifest["states"], label="state manifest", index=invalid_identity_index)
    identities = [identity for state in manifest["states"]
                  for identity in state["branch_identities"]]
    if (len(identities) != expected_branches
            or canonical_digest(sorted(row["branch_identity_digest"]
                                       for row in identities))
            != manifest["branch_identity_set_digest"]):
        raise RuntimeError("state manifest branch identity set is inconsistent")
    if len({state["scene_id"] for state in manifest["states"]}) != expected_states:
        raise RuntimeError("state manifest is not scene-disjoint")
    if any(_state_identity_digest(state) != state["state_identity_digest"]
           for state in manifest["states"]):
        raise RuntimeError("state manifest contains a changed state identity")
    identity_bindings = {
        "pool": pool,
        **{key: manifest[key] for key in (
            "candidate_bank_digest", "oracle_v1_2_digest",
            "scorer_contract_v1_2_digest", "render_contract_digest",
            "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
            "target_encoder_digest", "candidate_allocation_amendment_digest",
            "candidate_allocation_post_identity_validation_digest",
            "pre_identity_allocation_validation_digest",
            "invalid_scorer_identity_exclusion_digest",
            *LAUNCH_BINDING_KEYS,
        )},
    }
    for state in manifest["states"]:
        for candidate_index in state["candidate_indices"]:
            expected = _branch_identity(state, int(candidate_index), identity_bindings)
            if _identity_for(state, int(candidate_index)) != expected:
                raise RuntimeError("state manifest contains a changed branch identity")
    allocation_path = OUT_ROOT / pool / "candidate_allocation_manifest.json"
    allocation = json.loads(allocation_path.read_text())
    if allocation.get("allocation_manifest_digest") \
            != manifest["candidate_allocation_manifest_digest"]:
        raise RuntimeError("candidate allocation artifact digest mismatch")
    if pool == "scorer_fit":
        ALLOC.validate_allocation_manifest(
            allocation,
            expected_source_identity_manifest_digest=
                manifest["pre_allocation_identity_manifest_digest"])
        if (allocation["post_identity_pre_outcome_validation"][
                "post_identity_validation_digest"]
                != manifest[
                    "candidate_allocation_post_identity_validation_digest"]):
            raise RuntimeError(
                "post-identity allocation validation digest mismatch"
            )
    elif allocation["allocation_manifest_digest"] != canonical_digest({
            key: value for key, value in allocation.items()
            if key != "allocation_manifest_digest"}):
        raise RuntimeError("final candidate allocation self digest mismatch")
    assignment = {row["state_id"]: row["candidate_indices"]
                  for row in allocation["assignments"]}
    if any(list(state["candidate_indices"]) != list(assignment.get(state["state_id"], []))
           for state in manifest["states"]):
        raise RuntimeError("state manifest candidate assignments changed")


def _identity_for(state: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    matches = [row for row in state["branch_identities"]
               if int(row["candidate_index"]) == int(candidate_index)]
    if len(matches) != 1:
        raise RuntimeError("state manifest candidate identity lookup is ambiguous")
    return matches[0]


def _row_path(out: Path, identity: dict[str, Any]) -> Path:
    return out / "row_records" / f"{identity['branch_identity_digest']}.json"


def _resolve_corpus_file(out: Path, relative: str) -> Path:
    path = (out / relative).resolve()
    if out.resolve() not in path.parents:
        raise RuntimeError(f"corpus artifact escapes output root: {relative}")
    return path


def _validate_frame_record(out: Path, record: dict[str, Any]) -> None:
    path = _resolve_corpus_file(out, str(record["path"]))
    if (not path.is_file()
            or path.stat().st_size != int(record["byte_count"])
            or file_sha256(path) != record["sha256"]
            or record.get("shape") != [224, 224, 3]
            or record.get("dtype") != "uint8"):
        raise RuntimeError(f"frame receipt mismatch for {path}")


def _validate_branch_row(row: dict[str, Any], state: dict[str, Any],
                         identity: dict[str, Any], manifest: dict[str, Any],
                         out: Path) -> None:
    _verify_self_digest(row, "branch_row_digest",
                        f"branch row {state['state_id']}|{identity['candidate']}")
    expected = {
        "pool": manifest["pool"],
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digest": identity["branch_identity_digest"],
        "candidate": identity["candidate"],
        "candidate_index": int(identity["candidate_index"]),
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "candidate_allocator_contract_digest":
            manifest["candidate_allocator_contract_digest"],
        "candidate_allocation_amendment_digest":
            manifest["candidate_allocation_amendment_digest"],
        "candidate_allocation_post_identity_validation_digest":
            manifest["candidate_allocation_post_identity_validation_digest"],
        "pre_identity_allocation_validation_digest":
            manifest["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            manifest["invalid_scorer_identity_exclusion_digest"],
        **{key: manifest[key] for key in LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": manifest["candidate_bank_digest"],
        "progress_contract_digest": manifest["progress_contract_digest"],
        "safety_contract_digest": manifest["safety_contract_digest"],
        "oracle_v1_2_digest": manifest["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest": manifest["scorer_contract_v1_2_digest"],
        "selection_digest": manifest["selection_digest"],
        "boundary_digest": manifest["boundary_digest"],
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
    }
    if row.get("schema") != "go2_branch_corpus_v1_2_branch_row" \
            or row.get("record_complete") is not True:
        raise RuntimeError("branch row is not a completion record")
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"branch row {key} mismatch")
    for key in ("state_index", "split_role", "stratum", "scene_id", "family",
                "split", "episode_cluster_id", "episode_id", "source_step"):
        if row.get(key) != state[key]:
            raise RuntimeError(f"branch row state field {key} mismatch")
    if row.get("primitives") != identity["primitives"]:
        raise RuntimeError("branch row primitive sequence mismatch")
    if row.get("goal") != state["goal"]:
        raise RuntimeError("branch row goal binding mismatch")
    INVALID_IDS.assert_disjoint(
        [row], label="branch row", index=INVALID_IDS.load_invalid_identity_index())
    goal = state["goal"]
    expected_goal_binding = [
        math.sin(float(goal["bearing_body_rad"])),
        math.cos(float(goal["bearing_body_rad"])),
        float(goal["range_m"]),
    ]
    if not np.allclose(np.asarray(row.get("goal_binding_input"), dtype=np.float64),
                       np.asarray(expected_goal_binding, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row numeric goal binding mismatch")

    previous = np.asarray(row.get("previous_applied_command"), dtype=np.float64)
    if previous.shape != (3,) or not np.all(np.isfinite(previous)):
        raise RuntimeError("branch row previous applied command is malformed")
    candidate = V1.CANDIDATE_BANK[int(identity["candidate_index"])]
    requested, post_slew_plan, action_blocks = candidate_planning_trajectory(
        candidate, previous.tolist())
    if row.get("requested") != requested:
        raise RuntimeError("branch row requested candidate plan mismatch")
    if not np.allclose(np.asarray(row.get("candidate_post_slew_plan"),
                                  dtype=np.float64),
                       np.asarray(post_slew_plan, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row post-slew candidate plan mismatch")
    if not np.allclose(np.asarray(row.get("action_blocks"), dtype=np.float64),
                       np.asarray(action_blocks, dtype=np.float64),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError("branch row scorer action blocks mismatch")
    realised_prefix = np.asarray(row.get("post_slew"), dtype=np.float64)
    if (realised_prefix.ndim != 3 or realised_prefix.shape[1:] != (5, 3)
            or realised_prefix.shape[0] > HORIZONS
            or not np.all(np.isfinite(realised_prefix))
            or not np.allclose(realised_prefix,
                               np.asarray(post_slew_plan[:len(realised_prefix)],
                                          dtype=np.float64),
                               rtol=0.0, atol=1e-6)):
        raise RuntimeError("branch row realised post-slew prefix mismatch")
    action_context = np.asarray(row.get("action_context_blocks"), dtype=np.float64)
    proprio = np.asarray(row.get("proprio"), dtype=np.float64)
    control = np.asarray(row.get("control"), dtype=np.float64)
    if (action_context.shape != (CONTEXT_SLOTS, SLEW.ACTION_DIM)
            or proprio.shape != (PROPRIO_HISTORY, 30)
            or control.shape != (PROPRIO_HISTORY, 2)
            or not np.all(np.isfinite(action_context))
            or not np.all(np.isfinite(proprio))
            or not np.all(np.isfinite(control))):
        raise RuntimeError("branch row planning histories are malformed")
    context = row.get("context_frames", [])
    horizons = row.get("horizon_frames", [])
    if row.get("context_paths") != [frame.get("path") for frame in context] \
            or row.get("horizon_paths") != [frame.get("path") for frame in horizons]:
        raise RuntimeError("branch row frame-path projection mismatch")
    if row.get("valid"):
        if len(context) != CONTEXT_SLOTS or len(horizons) != HORIZONS:
            raise RuntimeError("valid branch row lacks exact H=1..4 renders")
        if any(row.get(key) is None for key in ("progress", "safety",
                                                "completion", "utility")):
            raise RuntimeError("valid branch row lacks oracle labels")
    elif not isinstance(row.get("invalid_reason"), str) or not row["invalid_reason"]:
        raise RuntimeError("invalid branch row lacks a reason code")
    for frame in context + horizons:
        _validate_frame_record(out, frame)


def _completed_rows(manifest: dict[str, Any], out: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for state in manifest["states"]:
        for candidate_index in state["candidate_indices"]:
            identity = _identity_for(state, int(candidate_index))
            path = _row_path(out, identity)
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text())
                _validate_branch_row(row, state, identity, manifest, out)
            except Exception as exc:
                preserved = _preserve_invalid(path, out, "row-validation-failed")
                print(f"[recovery] preserved invalid row {preserved}: {exc}", flush=True)
                continue
            completed[(state["state_id"], int(candidate_index))] = row
    return completed


def _frame_receipt(result: Any, path: Path, out: Path, *, index_key: str,
                   index_value: int) -> dict[str, Any]:
    digest, byte_count = write_png_atomic(result.image, path, out)
    return {
        index_key: int(index_value),
        "path": str(path.relative_to(out)),
        "sha256": digest,
        "byte_count": byte_count,
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "camera_pose_world": result.camera_pose_world,
        "render_runtime_s": round(float(result.runtime_s), 6),
    }


def _row_bindings(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_allocation_manifest_digest", "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "candidate_allocation_post_identity_validation_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest", "candidate_bank_digest",
        *LAUNCH_BINDING_KEYS,
        "progress_contract_digest", "safety_contract_digest",
        "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "selection_digest",
        "boundary_digest", "render_contract_digest",
        "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
        "preprocessing_digest", "target_encoder_digest",
        "target_encoder_checkpoint_sha256",
    )
    return {key: manifest[key] for key in keys}


def _redrive_mismatch(entry: dict[str, Any], record: dict[str, Any],
                      ctx: V1.BranchContext) -> str | None:
    comparisons = {
        "source_step": int(record["boundary"]["source_step"]) == int(entry["source_step"]),
        "boundary": record["boundary"] == entry["boundary"],
        "episode_id": int(ctx.runner.episode_states[0].episode_id)
                      == int(entry["episode_id"]),
        "cell_id": int(record["cell_id"]) == int(entry["cell_id"]),
        "goal": record["goal"] == entry["goal"],
    }
    failed = [name for name, passed in comparisons.items() if not passed]
    return None if not failed else "redrive_" + "_".join(failed) + "_mismatch"


def candidate_planning_trajectory(candidate: tuple[str, tuple[str, ...]],
                                  previous_applied: Sequence[float]
                                  ) -> tuple[list[list[list[float]]],
                                             list[list[list[float]]],
                                             list[list[float]]]:
    """Frozen full four-block request/post-slew plan, without future state."""

    previous = tuple(float(value) for value in previous_applied)
    requested: list[list[list[float]]] = []
    post_slew: list[list[list[float]]] = []
    action_blocks: list[list[float]] = []
    for primitive in candidate[1]:
        requested_block = np.asarray(V1.block_for(primitive), dtype=np.float64).tolist()
        reconstructed, previous = SLEW.reconstruct_block(primitive, previous)
        requested.append(requested_block)
        post_slew.append([[float(value) for value in tick]
                          for tick in reconstructed])
        action_blocks.append(action_block_10d(np.asarray(reconstructed,
                                                         dtype=np.float64)))
    if len(action_blocks) != HORIZONS or any(len(block) != SLEW.ACTION_DIM
                                             for block in action_blocks):
        raise RuntimeError("candidate planning action shape changed")
    return requested, post_slew, action_blocks


def _invalid_completed_row(entry: dict[str, Any], identity: dict[str, Any],
                           manifest: dict[str, Any], reason: str,
                           runtime_s: float) -> dict[str, Any]:
    row = {
        "schema": "go2_branch_corpus_v1_2_branch_row",
        "status": STATUS,
        "record_complete": True,
        "pool": manifest["pool"],
        "state_id": entry["state_id"],
        "state_index": int(entry["state_index"]),
        "state_identity_digest": entry["state_identity_digest"],
        "branch_identity_digest": identity["branch_identity_digest"],
        "split_role": entry["split_role"],
        "stratum": entry["stratum"],
        "scene_id": entry["scene_id"],
        "family": entry["family"],
        "split": entry["split"],
        "episode_cluster_id": entry["episode_cluster_id"],
        "episode_id": entry["episode_id"],
        "source_step": entry["source_step"],
        "candidate": identity["candidate"],
        "candidate_index": int(identity["candidate_index"]),
        "primitives": identity["primitives"],
        "goal": entry["goal"],
        "requested": None,
        "post_slew": None,
        "action_blocks": None,
        "action_context_blocks": None,
        "previous_applied_command": None,
        "context_frames": [],
        "horizon_frames": [],
        "context_paths": [],
        "horizon_paths": [],
        "proprio": None,
        "control": None,
        "valid": False,
        "invalid_reason": reason,
        "blocks_completed": 0,
        "truncated_at_block": None,
        "snapshot_digest": None,
        "wall_time_s": round(float(runtime_s), 6),
        "storage_bytes": 0,
        "state_manifest_digest": manifest["state_manifest_digest"],
        **_row_bindings(manifest),
    }
    for key in ("start_geodesic_m", "final_geodesic_m", "progress",
                "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
                "safety", "completion", "utility", "min_clearance_m",
                "evaluation_points"):
        row[key] = None
    row["branch_row_digest"] = canonical_digest(row)
    return row


def _write_row(out: Path, identity: dict[str, Any], row: dict[str, Any]) -> None:
    path = _row_path(out, identity)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing == row:
            return
        _preserve_invalid(path, out, "row-overwrite-refused")
    atomic_json(path, row)


def _write_invalid_attempt_row(out: Path, identity: dict[str, Any],
                               row: dict[str, Any], reason: str) -> Path:
    root = out / "invalid_attempts/redrive_records"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{identity['branch_identity_digest']}.{reason.replace(':', '-')}.invalid.json"
    path = root / stem
    counter = 0
    while path.exists():
        counter += 1
        path = root / f"{stem}.{counter}"
    atomic_json(path, row)
    return path


def _compiled_receipt(
        manifest: dict[str, Any], out: Path, ordered: list[dict[str, Any]],
        completed_states: int, rows_text: str,
        invocation_runtime_s: float) -> dict[str, Any]:
    """Build the exact derived receipt without mutating the corpus.

    Keeping this calculation independent of the on-disk ledger lets a
    zero-new resume prove that both existing derived artifacts are already
    exact, then retain their bytes rather than rewriting operational timing.
    """

    expected_states = len(manifest["states"])
    expected_branches = int(manifest["attempted_branch_count_registered"])
    valid_count = sum(bool(row["valid"]) for row in ordered)
    complete = len(ordered) == expected_branches and completed_states == expected_states
    rows_bytes = rows_text.encode()
    branch_rows_sha = hashlib.sha256(rows_bytes).hexdigest()
    frame_sizes: dict[str, int] = {}
    for row in ordered:
        for frame in row.get("context_frames", []) + row.get("horizon_frames", []):
            relative = str(frame["path"])
            byte_count = int(frame["byte_count"])
            if relative in frame_sizes and frame_sizes[relative] != byte_count:
                raise RuntimeError("shared frame receipts disagree on byte count")
            frame_sizes[relative] = byte_count
    frame_storage_bytes = sum(frame_sizes.values())
    row_record_storage_bytes = sum(
        _row_path(out, _identity_for(state, int(candidate_index))).stat().st_size
        for state in manifest["states"]
        for candidate_index in state["candidate_indices"]
        if _row_path(out, _identity_for(state, int(candidate_index))).is_file()
    )
    ledger_storage_bytes = len(rows_bytes)
    storage_bytes = (frame_storage_bytes + row_record_storage_bytes
                     + ledger_storage_bytes)
    runtime_total = sum(float(row.get("wall_time_s") or 0.0) for row in ordered)
    payload = {
        "schema": "go2_branch_corpus_v1_2_corpus_identity",
        "pool": manifest["pool"],
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "branch_identity_set_digest": manifest["branch_identity_set_digest"],
        "branch_rows_sha256": branch_rows_sha,
        "branch_row_digests": [row["branch_row_digest"] for row in ordered],
        "state_count": expected_states,
        "attempted_branch_count": len(ordered),
        "valid_branch_count": valid_count,
        "invalid_branch_count": len(ordered) - valid_count,
        "complete": complete,
        "bound_digests": _row_bindings(manifest),
    }
    receipt = {
        "schema": "go2_branch_corpus_v1_2_completion_receipt",
        "status": STATUS,
        "pool": manifest["pool"],
        "complete": complete,
        "states": expected_states,
        "state_count": expected_states,
        "completed_states": completed_states,
        "expected_branches": expected_branches,
        "attempted_branches": len(ordered),
        "attempted_count": len(ordered),
        "rows": len(ordered),
        "valid_branches": valid_count,
        "valid_count": valid_count,
        "invalid_branches": len(ordered) - valid_count,
        "invalid_count": len(ordered) - valid_count,
        "state_manifest_digest": manifest["state_manifest_digest"],
        "candidate_allocation_manifest_digest":
            manifest["candidate_allocation_manifest_digest"],
        "branch_rows_sha256": branch_rows_sha,
        **_row_bindings(manifest),
        "corpus_digest_payload": payload,
        "corpus_digest": canonical_digest(payload),
        "runtime_s_completed_rows": round(runtime_total, 6),
        "runtime_s_this_invocation": round(float(invocation_runtime_s), 6),
        "storage_bytes": storage_bytes,
        "storage_components_bytes": {
            "unique_rendered_frames": frame_storage_bytes,
            "row_records": row_record_storage_bytes,
            "branch_rows_ledger": ledger_storage_bytes,
        },
    }
    return receipt


def _load_exact_compiled_receipt(
        manifest: dict[str, Any], out: Path, ordered: list[dict[str, Any]],
        completed_states: int, rows_text: str) -> dict[str, Any] | None:
    """Return an existing byte-valid ledger/receipt pair, otherwise ``None``."""

    rows_path = out / "branch_rows.jsonl"
    receipt_path = out / "corpus_receipt.json"
    if not rows_path.is_file() or not receipt_path.is_file():
        return None
    try:
        if rows_path.read_bytes() != rows_text.encode():
            return None
        existing = json.loads(receipt_path.read_text())
        runtime = existing.get("runtime_s_this_invocation")
        if (isinstance(runtime, bool) or not isinstance(runtime, (int, float))
                or not math.isfinite(float(runtime)) or float(runtime) < 0.0):
            return None
        expected = _compiled_receipt(
            manifest, out, ordered, completed_states, rows_text, float(runtime))
        if existing != expected:
            return None
        return existing
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _compile_corpus(manifest: dict[str, Any], out: Path,
                    invocation_runtime_s: float) -> dict[str, Any]:
    completed = _completed_rows(manifest, out)
    ordered: list[dict[str, Any]] = []
    completed_states = 0
    for state in manifest["states"]:
        state_rows = []
        for candidate_index in state["candidate_indices"]:
            row = completed.get((state["state_id"], int(candidate_index)))
            if row is not None:
                state_rows.append(row)
                ordered.append(row)
        if len(state_rows) == len(state["candidate_indices"]):
            completed_states += 1
    rows_text = "".join(json.dumps(V1._jsonable(row), sort_keys=True) + "\n"
                        for row in ordered)
    retained = _load_exact_compiled_receipt(
        manifest, out, ordered, completed_states, rows_text)
    if retained is not None:
        return retained

    rows_path = out / "branch_rows.jsonl"
    receipt_path = out / "corpus_receipt.json"
    if not rows_path.is_file() or rows_path.read_bytes() != rows_text.encode():
        if rows_path.exists():
            _preserve_invalid(
                rows_path, out, "superseded-or-invalid-compilation")
        atomic_text(rows_path, rows_text)
    receipt = _compiled_receipt(
        manifest, out, ordered, completed_states, rows_text,
        invocation_runtime_s)
    if receipt_path.exists():
        try:
            existing = json.loads(receipt_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        if existing == receipt:
            return existing
        _preserve_invalid(receipt_path, out, "superseded-or-invalid-compilation")
    atomic_json(receipt_path, receipt)
    return receipt


def _build_smoke_branch_receipt(
        manifest: dict[str, Any], rows: list[dict[str, Any]], *,
        corpus_digest: str, replay_check: dict[str, Any]) -> dict[str, Any]:
    """Build the six-branch replay receipt from already validated rows."""

    state = manifest["states"][0]
    receipt = {
        "schema": "go2_scorer_fit_branch_smoke_receipt_v1_2",
        "status": STATUS,
        "pass": bool(
            len(rows) == 6 and all(row["valid"] for row in rows)
            and replay_check.get("exact_repeat") is True
            and all(len(row["context_frames"]) == 3
                    and len(row["horizon_frames"]) == 4 for row in rows)
        ),
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "branch_identity_digests": sorted(row["branch_identity_digest"]
                                           for row in rows),
        "branch_row_digests": sorted(row["branch_row_digest"] for row in rows),
        "state_manifest_digest": manifest["state_manifest_digest"],
        "corpus_bound_digests": _row_bindings(manifest),
        **_row_bindings(manifest),
        "scorer_contract_v1_2_digest": scorer_contract_digest(),
        "corpus_digest": corpus_digest,
        "render_contract_digest": manifest["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest": manifest["preprocess_contract_digest"],
        "preprocessing_digest": manifest["preprocessing_digest"],
        "target_encoder_digest": manifest["target_encoder_digest"],
        "target_encoder_checkpoint_sha256":
            manifest["target_encoder_checkpoint_sha256"],
        "replay_check": replay_check,
    }
    receipt["smoke_branch_receipt_digest"] = canonical_digest(receipt)
    return receipt


def _load_valid_smoke_branch_receipt(
        manifest: dict[str, Any], out: Path,
        rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and reuse the original exact-replay proof on zero-new smoke."""

    path = out / "smoke_branch_receipt.json"
    if not path.is_file():
        raise RuntimeError(
            "six completed smoke rows exist without a replay receipt; refusing "
            "to downgrade or fabricate the exact-replay check"
        )
    try:
        receipt = json.loads(path.read_text())
        _verify_self_digest(
            receipt, "smoke_branch_receipt_digest", "branch smoke receipt")
        corpus_digest = str(receipt["corpus_digest"])
        if (len(corpus_digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in corpus_digest)):
            raise RuntimeError("branch smoke receipt corpus digest is malformed")
        replay_check = receipt["replay_check"]
        if (not isinstance(replay_check, dict)
                or replay_check.get("state_id") != manifest["states"][0]["state_id"]
                or replay_check.get("exact_repeat") is not True
                or replay_check.get("separate_render_scene_physically_inert") is not True):
            raise RuntimeError("branch smoke replay proof is missing or failed")
        matching = [row for row in rows
                    if row.get("candidate") == replay_check.get("candidate")]
        if (len(matching) != 1
                or matching[0].get("snapshot_digest")
                != replay_check.get("snapshot_digest")):
            raise RuntimeError("branch smoke replay proof no longer matches its row")
        expected = _build_smoke_branch_receipt(
            manifest, rows, corpus_digest=corpus_digest,
            replay_check=replay_check)
        if receipt != expected or receipt.get("pass") is not True:
            raise RuntimeError("branch smoke receipt differs from current rows")
        return receipt
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"branch smoke receipt validation failed: {exc}") from exc


def _final_identifiability_gate(manifest: dict[str, Any], out: Path,
                                receipt: dict[str, Any]) -> dict[str, Any] | None:
    if manifest["pool"] != "final_eval" or not receipt["complete"]:
        return None
    rows = [json.loads(line) for line in (out / "branch_rows.jsonl").read_text().splitlines()
            if line.strip()]
    statistics = V1.identifiability(rows)
    verdict = V1.gate_verdict(statistics)
    report = {
        "schema": "go2_final_evaluation_oracle_identifiability_gate_v1_2",
        "status": STATUS,
        "state_manifest_digest": manifest["state_manifest_digest"],
        "corpus_digest": receipt["corpus_digest"],
        "tie_tolerance": V1.TIE_TOLERANCE,
        "statistics": statistics,
        "gate": verdict,
        "predictor_checkpoint_loading_authorized": bool(verdict["pass"]),
    }
    report["final_gate_digest"] = canonical_digest(report)
    atomic_json(out / "final_gate.json", report)
    return report


def stage_branches(args: argparse.Namespace, *, smoke: bool = False) -> int:
    out = OUT_ROOT / args.pool
    manifest = json.loads((out / "state_manifest.json").read_text())
    _validate_state_manifest(manifest, args.pool)
    if args.backend != "cpu":
        raise RuntimeError("the frozen qualified branch backend is cpu")
    if smoke and args.pool != "scorer_fit":
        raise RuntimeError("end-to-end smoke is defined only for scorer_fit")
    if not smoke:
        smoke_path = OUT_ROOT / "scorer_fit/smoke_encoding_receipt.json"
        if not smoke_path.is_file():
            raise RuntimeError("full branch generation is gated on the encoded six-branch smoke")
        smoke_receipt = json.loads(smoke_path.read_text())
        _verify_self_digest(smoke_receipt, "smoke_receipt_digest", "smoke receipt")
        if (not smoke_receipt.get("pass")
                or smoke_receipt.get("state_manifest_digest")
                != (manifest["state_manifest_digest"] if args.pool == "scorer_fit"
                    else json.loads((OUT_ROOT / "scorer_fit/state_manifest.json").read_text())[
                        "state_manifest_digest"])
                or smoke_receipt.get("scorer_contract_v1_2_digest")
                != scorer_contract_digest()):
            raise RuntimeError("encoded smoke receipt is not valid for this scorer contract")

    completed = _completed_rows(manifest, out)
    states = ([manifest["states"][0]] if smoke else
              manifest["states"][args.state_offset:args.state_offset + args.state_limit])
    frames_dir = out / "frames"
    invocation_started = time.time()
    new_rows = 0
    replay_check: dict[str, Any] | None = None

    if smoke:
        state = manifest["states"][0]
        smoke_rows = [row for row in completed.values()
                      if row["state_id"] == state["state_id"]]
        if len(smoke_rows) == len(state["candidate_indices"]):
            # A zero-new smoke validates its durable proof and never starts
            # Genesis.  The compile call is itself byte-idempotent.
            receipt = _compile_corpus(
                manifest, out, time.time() - invocation_started)
            retained_smoke = _load_valid_smoke_branch_receipt(
                manifest, out, smoke_rows)
            if retained_smoke["corpus_digest"] != receipt["corpus_digest"]:
                retained_smoke = _build_smoke_branch_receipt(
                    manifest, smoke_rows,
                    corpus_digest=receipt["corpus_digest"],
                    replay_check=retained_smoke["replay_check"])
                atomic_json(out / "smoke_branch_receipt.json", retained_smoke)
            print(json.dumps({
                **retained_smoke,
                "recovery": "retained_valid_zero_new_replay_receipt",
            }, indent=2, sort_keys=True))
            return 0

    shared = None

    for entry in states:
        missing = [int(index) for index in entry["candidate_indices"]
                   if (entry["state_id"], int(index)) not in completed]
        if not missing:
            print(f"[branches] retain complete {entry['state_id']}", flush=True)
            continue
        state_started = time.time()
        print(f"[branches] {entry['state_id']} ({entry['scene_id']}); missing={missing}",
              flush=True)
        if shared is None:
            shared = V1._load_shared(args.backend)
        scene_dir = Path(entry["scene_dir"])
        scene_manifest_path = scene_dir / "manifest.json"
        if (file_sha256(scene_manifest_path) != entry["scene_manifest_sha256"]
                or scene_manifest_path.stat().st_size
                != int(entry["scene_manifest_byte_count"])):
            raise RuntimeError("registered scene manifest changed after identity freeze")
        ctx = V1.build_context(scene_dir, seed=int(entry["drive_seed"]),
                               backend=args.backend, shared=shared)
        topology = V12.link_topology(ctx)
        ctx.begin_episode()
        proprio_log: list[list[float]] = []
        control_log: list[list[float]] = []
        context_poses: list[BasePose] = []
        action_context_blocks: list[list[float]] = []
        warmup = int(entry["warmup_blocks"])

        def probe(_tick_idx: int, previous_applied: Sequence[float]) -> None:
            proprio_log.append(proprio_sample(ctx))
            control_log.append(control_sample(previous_applied))

        for block_index in range(warmup):
            driven = drive_block_with_probe(ctx, probe)
            if block_index >= warmup - CONTEXT_SLOTS:
                action_context_blocks.append(action_block_10d(
                    np.asarray(driven.executed, dtype=np.float64)[0]))
                context_poses.append(capture_base_pose(ctx))

        verdict = classify_state(ctx, topology)
        redrive_reason: str | None = None
        if isinstance(verdict, str):
            redrive_reason = f"redrive_failed:{verdict}"
        elif len(proprio_log) < PROPRIO_HISTORY:
            redrive_reason = "redrive_failed:short_proprio_history"
        else:
            record, field, _strata = verdict
            redrive_reason = _redrive_mismatch(entry, record, ctx)
        if redrive_reason is not None:
            for candidate_index in missing:
                identity = _identity_for(entry, candidate_index)
                row = _invalid_completed_row(
                    entry, identity, manifest, redrive_reason,
                    time.time() - state_started)
                _write_invalid_attempt_row(out, identity, row, redrive_reason)
            _compile_corpus(manifest, out, time.time() - invocation_started)
            del ctx
            gc.collect()
            raise RuntimeError(
                f"registered state {entry['state_id']} could not be redriven exactly: "
                f"{redrive_reason}")

        raw_manifest = json.loads(scene_manifest_path.read_text())
        import genesis as gs
        renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
        if renderer.contract_digest != manifest["textured_v03_renderer_contract_digest"]:
            raise RuntimeError("runtime historical renderer contract changed")
        context_frames: list[dict[str, Any]] = []
        for slot, pose in enumerate(context_poses):
            result = renderer.render_pose(pose)
            path = (frames_dir / entry["family"]
                    / f"{entry['state_identity_digest']}_ctx{slot}.png")
            context_frames.append(_frame_receipt(
                result, path, out, index_key="slot", index_value=slot))
        proprio = np.asarray(proprio_log[-PROPRIO_HISTORY:], dtype=np.float32)
        control = np.asarray(control_log[-PROPRIO_HISTORY:], dtype=np.float32)
        if proprio.shape != (PROPRIO_HISTORY, 30) or control.shape != (
                PROPRIO_HISTORY, 2):
            raise RuntimeError(f"planning history shape changed: {proprio.shape}/{control.shape}")
        previous_applied = np.asarray(ctx.runner._last_executed,
                                      dtype=np.float64)[0].tolist()
        snapshot = V1.capture_branch_state(
            ctx, goal=entry["goal"],
            identity={
                "state_id": entry["state_id"],
                "state_identity_digest": entry["state_identity_digest"],
                "scene_id": entry["scene_id"],
                "family": entry["family"],
                "split": entry["split"],
                "block_index": warmup,
                "source_step": entry["source_step"],
                "episode_id": int(entry["episode_id"]),
            })

        for candidate_index in missing:
            identity = _identity_for(entry, candidate_index)
            candidate = V1.CANDIDATE_BANK[candidate_index]
            branch_started = time.time()
            horizon_poses: list[BasePose] = []
            requested_plan, post_slew_plan, action_plan_blocks = (
                candidate_planning_trajectory(candidate, previous_applied))

            def on_block_end(_block_index: int) -> None:
                horizon_poses.append(capture_base_pose(ctx))

            branch = _execute_and_render(ctx, snapshot, candidate, field=field,
                                         topology=topology,
                                         on_block_end=on_block_end)
            scored = V12.score_branch_v12(branch)
            actual_post_slew = branch["post_slew"]
            prefix_plan = post_slew_plan[:len(actual_post_slew)]
            if not np.allclose(np.asarray(actual_post_slew, dtype=np.float64),
                               np.asarray(prefix_plan, dtype=np.float64),
                               rtol=0.0, atol=1e-6):
                raise RuntimeError("runtime post-slew actions disagree with frozen planning reconstruction")
            invalid_reason = None
            if scored is None:
                invalid_reason = ("solver_nan" if branch["nan"]
                                  else "unlocatable_or_unreachable_geodesic")
            horizon_frames: list[dict[str, Any]] = []
            for horizon_index, pose in enumerate(horizon_poses, start=1):
                result = renderer.render_pose(pose)
                path = (frames_dir / entry["family"]
                        / f"{identity['branch_identity_digest']}_h{horizon_index}.png")
                horizon_frames.append(_frame_receipt(
                    result, path, out, index_key="horizon",
                    index_value=horizon_index))
            valid = bool(scored is not None and len(horizon_frames) == HORIZONS)
            if scored is not None and not valid:
                invalid_reason = "truncated_before_h4_render"
            row = {
                "schema": "go2_branch_corpus_v1_2_branch_row",
                "status": STATUS,
                "record_complete": True,
                "pool": args.pool,
                "state_id": entry["state_id"],
                "state_index": int(entry["state_index"]),
                "state_identity_digest": entry["state_identity_digest"],
                "branch_identity_digest": identity["branch_identity_digest"],
                "split_role": entry["split_role"],
                "stratum": entry["stratum"],
                "scene_id": entry["scene_id"],
                "family": entry["family"],
                "split": entry["split"],
                "episode_cluster_id": entry["episode_cluster_id"],
                "episode_id": int(snapshot.identity["episode_id"]),
                "source_step": int(snapshot.identity["source_step"]),
                "candidate": candidate[0],
                "candidate_index": int(candidate_index),
                "primitives": list(candidate[1]),
                "requested": requested_plan,
                "realised_requested_prefix": branch["requested"],
                "post_slew": branch["post_slew"],
                "candidate_post_slew_plan": post_slew_plan,
                "action_blocks": action_plan_blocks,
                "action_context_blocks": action_context_blocks,
                "previous_applied_command": previous_applied,
                "goal": entry["goal"],
                "goal_binding_input": [
                    math.sin(float(entry["goal"]["bearing_body_rad"])),
                    math.cos(float(entry["goal"]["bearing_body_rad"])),
                    float(entry["goal"]["range_m"]),
                ],
                "context_frames": context_frames,
                "horizon_frames": horizon_frames,
                "context_paths": [frame["path"] for frame in context_frames],
                "horizon_paths": [frame["path"] for frame in horizon_frames],
                "proprio": proprio.tolist(),
                "control": control.tolist(),
                "masks": {
                    "context_rgb_valid": [True] * CONTEXT_SLOTS,
                    "observed_proprio_valid": [True] * PROPRIO_HISTORY,
                    "observed_control_valid": [True] * PROPRIO_HISTORY,
                    "future_proprio_available": [False] * HORIZONS,
                    "target_rgb_valid": [index < len(horizon_frames)
                                         for index in range(HORIZONS)],
                },
                "timing": {
                    "command_hz": 10,
                    "ticks_per_block": 5,
                    "seconds_per_block": 0.5,
                    "context_boundary_offsets_blocks": [-2, -1, 0],
                    "target_horizons_blocks": [1, 2, 3, 4],
                },
                "valid": valid,
                "invalid_reason": invalid_reason,
                "blocks_completed": branch["blocks_completed"],
                "truncated_at_block": branch["truncated_at_block"],
                "snapshot_digest": snapshot.digest,
                "state_manifest_digest": manifest["state_manifest_digest"],
                **_row_bindings(manifest),
                "wall_time_s": round(time.time() - branch_started, 6),
                "storage_bytes": sum(frame["byte_count"] for frame in
                                     context_frames + horizon_frames),
            }
            row.update({key: (None if scored is None else scored[key]) for key in (
                "start_geodesic_m", "final_geodesic_m", "progress",
                "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
                "safety", "completion", "utility", "min_clearance_m",
                "evaluation_points")})
            row["branch_row_digest"] = canonical_digest(row)
            _write_row(out, identity, row)
            completed[(entry["state_id"], candidate_index)] = row
            new_rows += 1

            if smoke and replay_check is None:
                repeat = _execute_and_render(ctx, snapshot, candidate, field=field,
                                             topology=topology, on_block_end=None)
                again = V12.score_branch_v12(repeat)
                replay_check = {
                    "state_id": entry["state_id"],
                    "candidate": candidate[0],
                    "snapshot_digest": snapshot.digest,
                    "exact_repeat": bool(
                        scored is not None and again is not None
                        and canonical_digest(scored) == canonical_digest(again)
                        and branch["requested"] == repeat["requested"]
                        and branch["post_slew"] == repeat["post_slew"]),
                    "separate_render_scene_physically_inert": True,
                }
        print(f"    done in {time.time() - state_started:.1f}s", flush=True)
        del renderer, ctx
        gc.collect()

    receipt = _compile_corpus(manifest, out, time.time() - invocation_started)
    if smoke:
        state = manifest["states"][0]
        rows = [row for row in _completed_rows(manifest, out).values()
                if row["state_id"] == state["state_id"]]
        if replay_check is None:
            raise RuntimeError("new smoke branches did not produce an exact-replay check")
        smoke_receipt = _build_smoke_branch_receipt(
            manifest, rows, corpus_digest=receipt["corpus_digest"],
            replay_check=replay_check)
        atomic_json(out / "smoke_branch_receipt.json", smoke_receipt)
        print(json.dumps(smoke_receipt, indent=2, sort_keys=True))
        return 0 if smoke_receipt["pass"] else 1

    if args.pool == "scorer_fit" and receipt["complete"]:
        # The exact-replay proof is first issued against the six-row smoke
        # corpus.  Rebind that unchanged proof once to the immutable complete
        # corpus so the full encoder can validate it fail-closed.
        state = manifest["states"][0]
        smoke_rows = [row for row in _completed_rows(manifest, out).values()
                      if row["state_id"] == state["state_id"]]
        prior_smoke = _load_valid_smoke_branch_receipt(
            manifest, out, smoke_rows)
        refreshed_smoke = _build_smoke_branch_receipt(
            manifest, smoke_rows, corpus_digest=receipt["corpus_digest"],
            replay_check=prior_smoke["replay_check"])
        if refreshed_smoke != prior_smoke:
            atomic_json(out / "smoke_branch_receipt.json", refreshed_smoke)

    gate = _final_identifiability_gate(manifest, out, receipt)
    summary = {
        "pool": args.pool,
        "new_rows": new_rows,
        "attempted_rows": receipt["attempted_branches"],
        "valid_rows": receipt["valid_branches"],
        "complete": receipt["complete"],
        "corpus_digest": receipt["corpus_digest"],
        "final_gate": None if gate is None else gate["gate"],
        "wall_time_s": round(time.time() - invocation_started, 3),
    }
    atomic_json(out / f"branch_summary_{args.state_offset}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _execute_and_render(ctx, snapshot, candidate, *, field, topology, on_block_end):
    """V12.execute_branch_v12 with an optional per-block render hook."""

    from lewm_worlds.labels.derived import (DerivedLabelComputer, DerivedLabelConfig,
                                            PoseStep)
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step),
             "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed_cmd):
        (x, y), yaw, z = ctx.pose()
        label = label_computer.step(PoseStep(
            timestamp_ns=int(state["stamp_ns"]), env_idx=0, episode_id=episode_id,
            episode_step=int(state["episode_step"]), position_xy_world=(x, y),
            yaw_world_rad=float(yaw),
            last_command=(float(executed_cmd[0]), float(executed_cmd[1]),
                          float(executed_cmd[2]))))
        flags = V1._termination_flags(ctx)
        hit = ctx.scene_graph.locate((x, y))
        located = bool(float(hit.distance_m) <= V1.LOCATE_MAX_DISTANCE_M)
        cell = int(hit.cell_id)
        return {"xy": [x, y], "yaw": yaw, "z": z, "cell_id": cell, "located": located,
                "geodesic_m": float(field.remaining_distance((x, y), cell)
                                    if located else math.inf),
                "at_goal_cell": bool(cell == goal_cell),
                "clearance_m": float(label.clearance_m),
                "stuck": bool(label.stuck_label),
                "disallowed_contacts": int(V12._contact_count(ctx, topology)),
                "terminated": bool(flags["fall"] or flags["out_of_bounds"]
                                   or flags["tipped"]),
                "nan": bool(flags["nan"])}

    start_row = sample(np.asarray(runner._last_executed, dtype=np.float64)[0])
    tick_rows, requested_all, executed_all = [], [], []
    truncated_at_block, nan_seen = None, False

    for block_idx, primitive in enumerate(candidate[1]):
        requested = V1.block_for(primitive)[None, ...]
        executed_block = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64)

        def after_policy_step(tick_idx, step_idx, _b=executed_block, _i=block_idx):
            if step_idx != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_b[0, tick_idx])
            row["block"] = _i
            row["tick"] = int(tick_idx)
            tick_rows.append(row)

        block = runner.execute_requested_block(requested,
                                               after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        if on_block_end is not None:
            on_block_end(block_idx)
        if tick_rows and tick_rows[-1]["nan"]:
            nan_seen, truncated_at_block = True, block_idx
            break
        if tick_rows and tick_rows[-1]["terminated"]:
            truncated_at_block = block_idx
            break

    return {"candidate": candidate[0], "primitives": list(candidate[1]),
            "requested": requested_all, "post_slew": executed_all,
            "blocks_completed": len(executed_all),
            "truncated_at_block": truncated_at_block, "nan": nan_seen,
            "start": start_row, "ticks": tick_rows}


def _branch_identity(state: dict[str, Any], candidate_index: int,
                     manifest_bindings: dict[str, Any]) -> dict[str, Any]:
    candidate = V1.CANDIDATE_BANK[candidate_index]
    payload = {
        "schema": "go2_branch_identity_v1_2",
        "pool": manifest_bindings["pool"],
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "episode_cluster_id": state["episode_cluster_id"],
        "source_step": state["source_step"],
        "goal": state["goal"],
        "candidate_index": int(candidate_index),
        "candidate": candidate[0],
        "primitives": list(candidate[1]),
        "candidate_allocation_amendment_digest":
            manifest_bindings["candidate_allocation_amendment_digest"],
        "candidate_allocation_post_identity_validation_digest":
            manifest_bindings[
                "candidate_allocation_post_identity_validation_digest"
            ],
        "pre_identity_allocation_validation_digest":
            manifest_bindings["pre_identity_allocation_validation_digest"],
        "invalid_scorer_identity_exclusion_digest":
            manifest_bindings["invalid_scorer_identity_exclusion_digest"],
        **{key: manifest_bindings[key] for key in LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": manifest_bindings["candidate_bank_digest"],
        "oracle_v1_2_digest": manifest_bindings["oracle_v1_2_digest"],
        "scorer_contract_v1_2_digest":
            manifest_bindings["scorer_contract_v1_2_digest"],
        "render_contract_digest": manifest_bindings["render_contract_digest"],
        "textured_v03_renderer_contract_digest":
            manifest_bindings["textured_v03_renderer_contract_digest"],
        "preprocess_contract_digest":
            manifest_bindings["preprocess_contract_digest"],
        "target_encoder_digest": manifest_bindings["target_encoder_digest"],
    }
    return {**payload, "branch_identity_digest": canonical_digest(payload)}


def _validate_state_shard(payload: dict[str, Any], path: Path,
                          expected_pool: str) -> None:
    _verify_self_digest(payload, "state_shard_digest", f"state shard {path.name}")
    spec = POOLS[expected_pool]
    if (payload.get("schema") != "go2_branch_corpus_v1_2_state_shard"
            or payload.get("pool") != expected_pool
            or payload.get("complete") is not True
            or payload.get("spec") != spec
            or payload.get("selection") != SELECTION
            or payload.get("selection_digest") != selection_digest()
            or payload.get("candidate_allocator_contract_digest")
            != ALLOC.allocation_contract_digest()
            or payload.get("candidate_allocation_amendment_digest")
            != ALLOC.allocation_amendment_digest()
            or payload.get("pre_identity_allocation_validation_digest")
            != _load_pre_identity_allocation_validation()[
                "pre_identity_validation_digest"]
            or payload.get("invalid_scorer_identity_exclusion_digest")
            != INVALID_IDS.invalid_identity_exclusion_digest()
            or payload.get("scorer_contract_v1_2_digest")
            != scorer_contract_digest()
            or payload.get("candidate_bank_digest") != V1.bank_digest()
            or payload.get("progress_contract_digest") != progress_digest()
            or payload.get("safety_contract_digest") != safety_digest()
            or payload.get("oracle_v1_2_digest") != v12_oracle_digest()
            or payload.get("render_contract_digest") != render_contract_digest()
            or payload.get("textured_v03_renderer_contract_digest")
            != textured_v03_renderer_contract_digest()
            or payload.get("preprocess_contract_digest")
            != preprocess_contract_digest()
            or payload.get("preprocessing_digest")
            != TARGET_ENCODER["preprocessing_identity_sha256"]
            or payload.get("target_encoder_digest") != target_encoder_digest()
            or payload.get("target_encoder_checkpoint_sha256")
            != TARGET_ENCODER["checkpoint_sha256"]
            or payload.get("boundary_digest") != V1.BOUNDARY_DIGEST
            or payload.get("genesis_backend") != "cpu"):
        raise RuntimeError(f"state shard {path.name} is bound to another contract")
    launch = _load_clean_source_launch_receipt()
    for key in LAUNCH_BINDING_KEYS:
        if payload.get(key) != launch[key]:
            raise RuntimeError(f"state shard {path.name} clean-source {key} mismatch")
    states = payload.get("states", [])
    if len(states) != spec["states_per_family"]:
        raise RuntimeError(f"state shard {path.name} has the wrong state count")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError(f"state shard {path.name} reuses a scene")
    invalid_identity_index = INVALID_IDS.load_invalid_identity_index()
    if payload.get("exclusion_binding", {}).get(
            "invalid_scorer_identity_attempt") != invalid_identity_index.binding():
        raise RuntimeError(f"state shard {path.name} invalid45 binding mismatch")
    INVALID_IDS.assert_disjoint(
        states, label=f"state shard {path.name}", index=invalid_identity_index)
    if any(_state_identity_digest(state) != state.get("state_identity_digest")
           for state in states):
        raise RuntimeError(f"state shard {path.name} has an identity digest mismatch")


def _outcome_generation_started(out: Path) -> bool:
    row_root = out / "row_records"
    frame_root = out / "frames"
    return bool(
        (out / "branch_rows.jsonl").exists()
        or (out / "corpus_receipt.json").exists()
        or (out / "latents_index.json").exists()
        or (row_root.is_dir() and any(path.is_file() for path in row_root.iterdir()))
        or (frame_root.is_dir()
            and any(path.is_file() for family in frame_root.iterdir()
                    if family.is_dir() for path in family.iterdir()))
    )


def merge_states(out: Path) -> int:
    """Merge exactly eight completed shards and freeze all branch identities."""

    pool_name = out.name
    if pool_name not in POOLS:
        raise RuntimeError(f"unknown output pool {pool_name!r}")
    paths = sorted(out.glob("state_shard_*.json"))
    if len(paths) != EXPECTED_FAMILIES:
        raise RuntimeError(f"expected eight state shards, found {len(paths)}")
    shards: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        _validate_state_shard(payload, path, pool_name)
        shards.append(payload)
    families = [str(shard["family"]) for shard in shards]
    if len(set(families)) != EXPECTED_FAMILIES:
        raise RuntimeError("state shards do not represent eight unique families")

    states = [dict(state) for shard in shards for state in shard["states"]]
    states.sort(key=lambda state: (
        state["family"],
        STRATA.index(state["stratum"]) if state["stratum"] in STRATA else 0,
        state["scene_id"],
    ))
    spec = POOLS[pool_name]
    expected_states = EXPECTED_FAMILIES * spec["states_per_family"]
    if len(states) != expected_states:
        raise RuntimeError(f"expected {expected_states} states, found {len(states)}")
    if len({state["scene_id"] for state in states}) != len(states):
        raise RuntimeError("merged state identities are not scene-disjoint")
    if len({state["episode_cluster_id"] for state in states}) != len(states):
        raise RuntimeError("merged state identities are not episode-cluster-disjoint")
    if len({state["state_identity_digest"] for state in states}) != len(states):
        raise RuntimeError("merged state identity digests are not unique")
    for index, state in enumerate(states):
        state["state_index"] = index

    common_keys = (
        "selection_digest", "scorer_fit_allocation_design_digest",
        "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest", "candidate_bank_digest",
        *LAUNCH_BINDING_KEYS,
        "progress_contract_digest", "safety_contract_digest",
        "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "boundary_digest",
        "render_contract_digest", "preprocess_contract_digest",
        "textured_v03_renderer_contract_digest",
        "preprocessing_digest", "target_encoder_digest",
        "target_encoder_checkpoint_sha256", "genesis_backend",
    )
    common = {key: shards[0][key] for key in common_keys}
    for shard in shards[1:]:
        if any(shard[key] != common[key] for key in common_keys):
            raise RuntimeError("state shards contain mixed contract bindings")

    pre_allocation_payload = {
        "schema": "go2_branch_corpus_v1_2_pre_allocation_identity_manifest",
        "pool": pool_name,
        "spec": spec,
        **common,
        "state_identities": [{
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "family": state["family"],
            "stratum": state["stratum"],
            "split_role": state["split_role"],
            "goal_type": state["goal_type"],
        } for state in states],
    }
    pre_allocation_digest = canonical_digest(pre_allocation_payload)

    allocation_path = out / "candidate_allocation_manifest.json"
    allocation_digest: str
    if pool_name == "scorer_fit":
        projection = [{
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "family": state["family"],
            "stratum": state["stratum"],
            "split_role": state["split_role"],
            "goal_type": state["goal_type"],
        } for state in states]
        allocation = ALLOC.build_allocation_manifest(
            projection, source_identity_manifest_digest=pre_allocation_digest)
        ALLOC.validate_allocation_manifest(
            allocation,
            expected_source_identity_manifest_digest=pre_allocation_digest)
        if allocation_path.is_file():
            existing = json.loads(allocation_path.read_text())
            if existing != allocation:
                if _outcome_generation_started(out):
                    raise RuntimeError("candidate allocation changed after branch generation")
                _preserve_invalid(allocation_path, out, "allocation-mismatch")
        atomic_json(allocation_path, allocation)
        allocation_digest = allocation["allocation_manifest_digest"]
        assigned = {row["state_id"]: row for row in allocation["assignments"]}
        for state in states:
            state["candidate_indices"] = list(assigned[state["state_id"]][
                "candidate_indices"])
            state["candidate_rotation_index"] = int(assigned[state["state_id"]][
                "rotation_index"])
    else:
        allocation = {
            "schema": "go2_final_eval_all_candidate_allocation_v1_2",
            "source_identity_manifest_digest": pre_allocation_digest,
            "candidate_bank_digest": V1.bank_digest(),
            "assignments": [{
                "state_id": state["state_id"],
                "state_identity_digest": state["state_identity_digest"],
                "candidate_indices": list(range(len(V1.CANDIDATE_BANK))),
            } for state in states],
        }
        allocation["allocation_manifest_digest"] = canonical_digest(allocation)
        if allocation_path.is_file():
            existing = json.loads(allocation_path.read_text())
            if existing != allocation:
                if _outcome_generation_started(out):
                    raise RuntimeError("final allocation changed after branch generation")
                _preserve_invalid(allocation_path, out, "allocation-mismatch")
        atomic_json(allocation_path, allocation)
        allocation_digest = allocation["allocation_manifest_digest"]
        for state in states:
            state["candidate_indices"] = list(range(len(V1.CANDIDATE_BANK)))

    post_identity_validation_digest = (
        allocation.get("post_identity_pre_outcome_validation", {}).get(
            "post_identity_validation_digest"
        )
    )
    if pool_name == "scorer_fit" and not isinstance(
            post_identity_validation_digest, str):
        raise RuntimeError(
            "scorer-fit allocation lacks post-identity/pre-outcome validation"
        )
    manifest_bindings = {
        "pool": pool_name,
        **common,
        "candidate_allocation_post_identity_validation_digest":
            post_identity_validation_digest,
    }
    branch_identities: list[dict[str, Any]] = []
    candidate_counts = {name: 0 for name, _sequence in V1.CANDIDATE_BANK}
    for state in states:
        identities = [_branch_identity(state, int(candidate_index), manifest_bindings)
                      for candidate_index in state["candidate_indices"]]
        state["branch_identities"] = identities
        branch_identities.extend(identities)
        for identity in identities:
            candidate_counts[identity["candidate"]] += 1
    expected_branches = expected_states * spec["candidates_per_state"]
    if len(branch_identities) != expected_branches:
        raise RuntimeError("registered branch identity count mismatch")
    branch_digests = [row["branch_identity_digest"] for row in branch_identities]
    if len(set(branch_digests)) != expected_branches:
        raise RuntimeError("registered branch identities are not unique")
    invalid_identity_disjointness = INVALID_IDS.assert_disjoint(
        states,
        label="merged pre-outcome state and branch identities",
        index=INVALID_IDS.load_invalid_identity_index(),
    )

    exclusion_bindings = [shard["exclusion_binding"] for shard in shards]
    if any(value != exclusion_bindings[0] for value in exclusion_bindings[1:]):
        raise RuntimeError("state shards disagree on exclusions")
    rejections = {shard["family"]: shard["scene_rejection_reasons"]
                  for shard in shards}
    manifest = {
        "schema": "go2_branch_corpus_v1_2_state_manifest",
        "status": STATUS,
        "complete": True,
        "pool": pool_name,
        "spec": spec,
        "selection": SELECTION,
        **common,
        "pre_allocation_identity_manifest_digest": pre_allocation_digest,
        "candidate_allocation_manifest_digest": allocation_digest,
        "candidate_allocation_post_identity_validation_digest":
            post_identity_validation_digest,
        "branch_identity_set_digest": canonical_digest(sorted(branch_digests)),
        "exclusion_binding": exclusion_bindings[0],
        "state_shard_digests": {shard["family"]: shard["state_shard_digest"]
                                for shard in shards},
        "states": states,
        "candidate_appearances": candidate_counts,
        "attempted_branch_count_registered": expected_branches,
        "disjointness": {
            "state_count": expected_states,
            "unique_scene_count": len({state["scene_id"] for state in states}),
            "unique_episode_cluster_count": len({state["episode_cluster_id"]
                                                  for state in states}),
            "unique_state_identity_count": len({state["state_identity_digest"]
                                                 for state in states}),
            "unique_branch_identity_count": len(set(branch_digests)),
            "scene_episode_state_branch_disjoint": True,
            "invalid_scorer_identity_attempt": invalid_identity_disjointness,
        },
        "scene_rejection_reasons": rejections,
        "recovery_provenance": {
            "interrupted_attempt_witnesses": [
                witness["path"] for witness in
                INVALID_IDS.INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"]
            ],
            "invalid_scorer_identity_exclusion_digest":
                INVALID_IDS.invalid_identity_exclusion_digest(),
            "invalid_scene_ids_digest": (
                INVALID_IDS.INVALID_SCORER_IDENTITY_EXCLUSION[
                    "derived_identity_bindings"]["scene_ids_digest"]
            ),
            "superseded_pre_run_contract_artifact":
                SUPERSEDED_PRE_RUN_CONTRACT_ARTIFACT,
            "decision": (
                "preserved the incomplete three-of-eight-family 45-state "
                "pre-outcome identity attempt under its superseded contract and "
                "selection; no branch, render, latent or outcome existed; the "
                "exact 45 scenes and all descendant identity namespaces are "
                "excluded from this corpus; no invalid artifact is mixed"
            ),
        },
    }
    manifest["state_manifest_digest"] = canonical_digest(manifest)
    manifest_path = out / "state_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            if _outcome_generation_started(out):
                raise RuntimeError("state identity manifest changed after branch generation")
            _preserve_invalid(manifest_path, out, "identity-mismatch")
    atomic_json(manifest_path, manifest)

    from collections import Counter
    print(json.dumps({
        "state_manifest_digest": manifest["state_manifest_digest"],
        "pre_allocation_identity_manifest_digest": pre_allocation_digest,
        "candidate_allocation_manifest_digest": allocation_digest,
        "states": len(states),
        "branches": len(branch_identities),
        "per_family": dict(Counter(state["family"] for state in states)),
        "per_stratum": dict(Counter(state["stratum"] for state in states)),
        "split_roles": dict(Counter(state["split_role"] for state in states)),
        "candidate_appearances": candidate_counts,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=sorted(POOLS), required=True)
    parser.add_argument("--stage",
                        choices=["allocation-preflight", "states", "merge-states",
                                 "smoke", "branches"],
                        required=True)
    parser.add_argument("--family", default=None,
                        help="resolve one family only; shards merge via merge-states")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=10**6)
    args = parser.parse_args()
    if args.state_offset < 0 or args.state_limit < 1:
        raise SystemExit("state offset must be nonnegative and limit positive")

    out = OUT_ROOT / args.pool
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "allocation-preflight":
        return issue_pre_identity_allocation_validation(out)
    if args.stage == "merge-states":
        return merge_states(out)
    if args.stage == "states":
        if args.family is None:
            raise SystemExit("--stage states requires exactly one --family shard")
        shard_path = out / f"state_shard_{args.family}.json"
        if shard_path.is_file():
            try:
                existing = json.loads(shard_path.read_text())
                _validate_state_shard(existing, shard_path, args.pool)
                print(json.dumps({
                    "recovery": "retained_valid_completed_identity_shard",
                    "path": str(shard_path),
                    "state_shard_digest": existing["state_shard_digest"],
                    "states": len(existing["states"]),
                }, indent=2, sort_keys=True))
                return 0
            except Exception as exc:
                preserved = _preserve_invalid(shard_path, out,
                                              "identity-shard-validation-failed")
                print(f"[recovery] preserved invalid shard {preserved}: {exc}",
                      flush=True)
        manifest = resolve_states(args)
        atomic_json(shard_path, manifest)
        from collections import Counter
        print(json.dumps({
            "state_shard_digest": manifest["state_shard_digest"],
            "states": len(manifest["states"]),
            "per_family": dict(Counter(s["family"] for s in manifest["states"])),
            "per_stratum": dict(Counter(s["stratum"] for s in manifest["states"])),
            "split_roles": dict(Counter(s["split_role"] for s in manifest["states"])),
        }, indent=2, sort_keys=True))
        return 0
    return stage_branches(args, smoke=args.stage == "smoke")


if __name__ == "__main__":
    raise SystemExit(main())
