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
import subprocess
import sys
import time
from functools import lru_cache
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
    GeodesicField, HORIZON_S, V_MAX_MPS,
    progress_digest, safety_digest, oracle_digest as v12_oracle_digest,
)
from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC
from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as INVALID_IDS
from lewm.oracle import go2_scorer_state_selector_amendment_v1 as STATE_SELECTOR
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
COMPLETION_ENRICHED_MAX_GEODESIC_M = STATE_SELECTOR.COMPLETION_MAX_GEODESIC_M
COMPLETION_ENRICHED_MAX_BEARING_RAD = math.radians(
    STATE_SELECTOR.COMPLETION_MAX_ABS_BEARING_DEG)
COMPLETION_HORIZON_S = STATE_SELECTOR.HORIZON_S
COMPLETION_MAX_TRANSLATION_M = STATE_SELECTOR.MAX_TRANSLATION_M
if (COMPLETION_HORIZON_S != HORIZON_S
        or STATE_SELECTOR.V_MAX_MPS != V_MAX_MPS
        or tuple(STATE_SELECTOR.SCORER_FIT_SELECTION_PRIORITY) != STRATA):
    raise RuntimeError("state-selector amendment changed a preserved oracle binding")

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
SELECTOR_FEASIBILITY_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_receipt_v1"
)
SELECTOR_FEASIBILITY_RECEIPT_NAME = "state_selector_feasibility_receipt.json"
SELECTOR_FEASIBILITY_PASS_STATUS = "PASS_OUTCOME_FREE_ALL_SCENE_FEASIBILITY"
SELECTOR_FEASIBILITY_REDUCER_VERSION = (
    "go2_scorer_fit_state_selector_feasibility_scene_isolated_reducer_v1"
)
SELECTOR_FEASIBILITY_TASK_CENSUS_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_task_census_v1"
)
SELECTOR_FEASIBILITY_TASK_CENSUS_NAME = (
    "state_selector_feasibility_task_census.json"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_scene_shard_v1"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS = (
    "COMPLETE_OUTCOME_FREE_SCENE_CENSUS_NO_ELIGIBILITY_VERDICT"
)
SELECTOR_FEASIBILITY_SCENE_SHARD_ROOT = (
    "state_selector_feasibility_scene_shards"
)
SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS = (
    "selected_state_identities_created", "candidate_outcomes_loaded",
    "branch_identities_created", "branches_attempted", "frames_rendered",
    "target_latents_encoded", "scorer_training_started",
)
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
ACTIVE_SELECTOR_BINDING_KEYS = (
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest",
    "preserved_state_revalidation_receipt_digest",
)


def selection_digest() -> str:
    # The scorer contract, not a mutually generated artifact, freezes selection.
    return str(scorer_contract()["corpus_selection_digest"])


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(V1._jsonable(payload), sort_keys=True).encode()
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _assert_unsealed_path(path: Path) -> None:
    """Reject sealed custody names and symlinks before path traversal."""

    if any(part == ".." or part == "sealed_test.json" or part == "sealed"
           or part.startswith("sealed_") for part in path.parts):
        raise RuntimeError("sealed benchmark paths are inaccessible")
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("symlinked corpus paths are inaccessible")


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
    selector_receipts = _load_state_selector_preconditions(
        source_commit=str(source["source_repository_commit"]),
        successor_selection_digest=selection_digest(),
    )
    if (scorer_artifact.get("state_selector_feasibility_receipt_digest")
            != selector_receipts["state_selector_feasibility_receipt_digest"]
            or scorer_artifact.get(
                "preserved_state_precontract_revalidation_receipt_digest")
            != selector_receipts[
                "preserved_state_precontract_revalidation_receipt_digest"]):
        raise RuntimeError(
            "issued scorer contract differs from active selector receipts"
        )
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
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        **selector_receipts,
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


def _load_state_selector_preconditions(
        *, source_commit: str, successor_selection_digest: str
        ) -> dict[str, str]:
    """Load and validate the outcome-free pre-identity feasibility gate."""

    feasibility_path = ROOT / STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH
    precontract_path = (
        ROOT
        / STATE_SELECTOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH
    )
    if not feasibility_path.is_file():
        raise RuntimeError("state-selector all-family feasibility receipt is missing")
    if not precontract_path.is_file():
        raise RuntimeError(
            "preserved-state precontract revalidation receipt is missing"
        )
    feasibility = json.loads(feasibility_path.read_text())
    precontract = json.loads(precontract_path.read_text())
    STATE_SELECTOR.validate_state_selector_feasibility_receipt(
        feasibility,
        expected_source_commit=source_commit,
        expected_successor_selection_digest=successor_selection_digest,
    )
    feasibility_digest = str(
        feasibility["state_selector_feasibility_receipt_digest"]
    )
    STATE_SELECTOR.validate_preserved_state_precontract_revalidation_receipt(
        precontract,
        expected_source_commit=source_commit,
        expected_successor_selection_digest=successor_selection_digest,
        expected_feasibility_receipt_digest=feasibility_digest,
        root=ROOT,
    )
    return {
        "state_selector_feasibility_receipt_digest": feasibility_digest,
        "preserved_state_precontract_revalidation_receipt_digest": str(
            precontract[
                "preserved_state_precontract_revalidation_receipt_digest"
            ]
        ),
    }


@lru_cache(maxsize=1)
def _preserved_states_by_digest() -> dict[str, dict[str, Any]]:
    shards = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    states = {
        str(state["state_identity_digest"]): dict(state)
        for shard in shards.values() for state in shard["states"]
    }
    if len(states) != 45:
        raise RuntimeError("preserved predecessor state identity count changed")
    return states


def _state_identity_matches_active_or_preserved(state: dict[str, Any]) -> bool:
    digest = str(state.get("state_identity_digest", ""))
    preserved = _preserved_states_by_digest().get(digest)
    if preserved is not None:
        return _state_identity_payload(state) == _state_identity_payload(preserved)
    return _state_identity_digest(state) == digest


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


def _snapshot_task_status(ctx: V1.BranchContext, goal_cell: int) -> dict[str, Any]:
    """Return the production task flags at one pre-branch canonical boundary.

    The active collector is the sole owner of route claims.  Truncation is
    false by construction: this selector runs between complete production
    command blocks, before a candidate branch or invocation-level limit exists.
    """

    runner = ctx.runner
    active = runner._scheduler.policy_for(0)
    visited = getattr(active, "visited_landmark_cells", None)
    claimed = (frozenset(int(cell) for cell in visited(0))
               if callable(visited) else frozenset())
    # Match RolloutRunner._check_and_reset_completed_envs exactly: its route
    # completion universe is the runner's landmark-cell lookup, not a newly
    # inferred or filtered goal list.
    all_goal_cells = frozenset(
        int(cell) for cell in runner._landmark_cell_to_id
    )
    from lewm_genesis.rollout import _MIN_BLOCKS_BEFORE_COMPLETE_RESET
    reset_evidence = {
        "minimum_block_guard_pass": (
            int(runner._blocks_in_episode[0])
            >= _MIN_BLOCKS_BEFORE_COMPLETE_RESET),
        "scene_graph_available": runner._scene_graph is not None,
        "active_collector_route_like": callable(visited),
        "active_collector_non_revisit": not bool(
            getattr(active, "revisit_after_arrival", False)),
        "scene_landmark_cells_nonempty": bool(all_goal_cells),
        "all_scene_landmark_cells_claimed": bool(
            all_goal_cells and all_goal_cells.issubset(claimed)),
    }
    task_completed = all(reset_evidence.values())
    termination_flags = {
        str(key): bool(value) for key, value in V1._termination_flags(ctx).items()
    }
    return {
        "task_completed": task_completed,
        "goal_claimed": int(goal_cell) in claimed,
        "production_claim_evidence": {
            "active_collector_visited_accessor_callable": callable(visited),
            "active_collector_claimed_cells": sorted(claimed),
            "designated_goal_cell": int(goal_cell),
        },
        "production_task_completion_reset_evidence": reset_evidence,
        "terminated": any(termination_flags.values()),
        "truncated": False,
        "termination_flags": termination_flags,
    }


def completion_enriched_eligibility(
        *, graph_hops: int, reachable: bool, continuous_geodesic_m: float,
        bearing_body_rad: float, task_status: dict[str, Any]) -> dict[str, Any]:
    """Pure successor predicate; it has no branch or outcome input.

    ``graph_hops`` is retained as diagnostic evidence only.  In particular,
    zero hops is not itself completion: the production task claim/completion
    flags remain authoritative and must both be false.
    """

    reasons: list[str] = []
    if not bool(reachable) or not math.isfinite(float(continuous_geodesic_m)):
        reasons.append("completion_unreachable")
    elif float(continuous_geodesic_m) > COMPLETION_ENRICHED_MAX_GEODESIC_M:
        reasons.append("completion_geodesic_gt_0_75m")
    if (not math.isfinite(float(bearing_body_rad))
            or abs(float(bearing_body_rad)) > COMPLETION_ENRICHED_MAX_BEARING_RAD):
        reasons.append("completion_bearing_gt_75deg")
    for key in ("task_completed", "goal_claimed", "terminated", "truncated"):
        if key not in task_status or not isinstance(task_status[key], bool):
            reasons.append(f"completion_snapshot_{key}_unavailable")
        elif task_status[key]:
            reasons.append(f"completion_snapshot_{key}")
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "reachable": bool(reachable),
        "continuous_geodesic_m": float(continuous_geodesic_m),
        "max_geodesic_m": COMPLETION_ENRICHED_MAX_GEODESIC_M,
        "bearing_body_rad": float(bearing_body_rad),
        "abs_bearing_rad": abs(float(bearing_body_rad)),
        "max_abs_bearing_rad": COMPLETION_ENRICHED_MAX_BEARING_RAD,
        "horizon_s": COMPLETION_HORIZON_S,
        "max_translation_m": COMPLETION_MAX_TRANSLATION_M,
        "graph_hops_diagnostic": int(graph_hops),
        "task_status": dict(task_status),
    }


def _oracle_completion_target_unchanged() -> bool:
    """Verify the frozen future completion label, not a snapshot task flag."""

    return bool(
        v12_oracle_digest() == STATE_SELECTOR.ORACLE_V1_2_DIGEST
        and HORIZONS == 4
        and COMPLETION_HORIZON_S == HORIZON_S == 2.0
    )


def _snapshot_claim_semantics_unchanged(task_status: dict[str, Any]) -> bool:
    evidence = task_status.get("production_claim_evidence", {})
    cells = evidence.get("active_collector_claimed_cells")
    goal_cell = evidence.get("designated_goal_cell")
    return bool(
        isinstance(evidence.get(
            "active_collector_visited_accessor_callable"), bool)
        and isinstance(cells, list)
        and isinstance(goal_cell, int)
        and isinstance(task_status.get("goal_claimed"), bool)
        and task_status["goal_claimed"] is (goal_cell in cells)
    )


def _production_task_reset_semantics_unchanged(
        task_status: dict[str, Any]) -> bool:
    evidence = task_status.get("production_task_completion_reset_evidence", {})
    required = (
        "minimum_block_guard_pass", "scene_graph_available",
        "active_collector_route_like", "active_collector_non_revisit",
        "scene_landmark_cells_nonempty", "all_scene_landmark_cells_claimed",
    )
    return bool(
        all(isinstance(evidence.get(key), bool) for key in required)
        and task_status.get("task_completed") is all(evidence[key] for key in required)
    )


def _goal_material(ctx: V1.BranchContext, name: str) -> str | None:
    materials = sorted({str(obj.material_id) for obj in ctx.pack.static_objects
                        if str(obj.object_id) == str(name)})
    if len(materials) != 1 or not materials[0].startswith("landmark_"):
        return None
    return materials[0]


def _state_record(*, boundary: dict[str, Any], cell: int, name: str,
                  landmark_cell: int, goal_type: str, hops: int, distance: float,
                  bearing: float, range_m: float, centre: Sequence[float],
                  body_clearance: float, clearance: float,
                  completion_eligibility: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    record = {
        "boundary": boundary, "cell_id": int(cell),
        "goal": {"landmark_id": str(name), "landmark_cell": int(landmark_cell),
                 "material_id": str(goal_type),
                 "graph_edges": int(hops), "start_geodesic_m": float(distance),
                 "bearing_body_rad": float(bearing), "range_m": float(range_m),
                 "landmark_xy_m": [float(centre[0]), float(centre[1])]},
        "body_clearance_m": float(body_clearance),
        "clearance_m": float(clearance),
    }
    if completion_eligibility is not None:
        record["completion_eligibility"] = completion_eligibility
        record["snapshot_task_status"] = completion_eligibility["task_status"]
    return record


def classify_state(ctx: V1.BranchContext, topology: dict[str, Any], *,
                   requested_stratum: str | None = None,
                   diagnostics: dict[str, int] | None = None
                   ) -> tuple[dict[str, Any], GeodesicField, set[str]] | str:
    """Classify one snapshot, optionally binding the goal for one stratum.

    General and safety retain the original nearest-landmark ``hops >= 2``
    semantics.  The successor changes only completion goal enumeration: its
    hop count is diagnostic, while unchanged continuous geometry and exact
    production task-status guards determine eligibility.
    """

    if requested_stratum not in (None, *STRATA):
        raise ValueError(f"unknown requested stratum {requested_stratum!r}")

    def reject(reason: str) -> str:
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
        return reason

    try:
        boundary = V1.assert_canonical_boundary(ctx)
    except V1.BoundaryRefused as exc:
        return reject(f"boundary_refused: {str(exc)[:50]}")
    if ctx.episode_ticks < PROPRIO_HISTORY - 1:
        return reject("insufficient_proprioceptive_history")
    (x, y), yaw, _z = ctx.pose()
    hit = ctx.scene_graph.locate((x, y))
    if float(hit.distance_m) > V1.LOCATE_MAX_DISTANCE_M:
        return reject("locate_distance_gt_2m")
    if V12._contact_count(ctx, topology) > 0:
        return reject("already_in_disallowed_contact")

    graph = ctx.scene_graph
    blocked = getattr(graph, "nav_blocked_cells", frozenset())
    cell = int(hit.cell_id)
    from analyze_go2_closed_loop_quality import _body_probe_configuration_clearance_m
    body_clearance = float(_body_probe_configuration_clearance_m(
        ctx.grid, [x, y], yaw,
        body_forward_m=V1.CONTACT_BODY_FORWARD_M,
        body_half_width_m=V1.CONTACT_BODY_HALF_WIDTH_M,
        body_probe_margin_m=V1.CONTACT_BODY_PROBE_MARGIN_M))
    clearance = float(graph.clearance_to_walls((x, y)))

    # Completion uses a dedicated designation pass.  Unlike the general/safety
    # pass below, no hop floor is applied before the unchanged reachability,
    # continuous-distance, bearing, horizon and task-state checks.
    if requested_stratum == "completion_enriched":
        eligible: list[tuple[tuple[float, str, int, int], dict[str, Any],
                                  GeodesicField]] = []
        saw_reachable = False
        for name, landmark_cell in sorted(graph.landmark_cells,
                                          key=lambda kv: str(kv[0])):
            hops = graph.bfs_distance(cell, int(landmark_cell),
                                      transit_blocked=blocked)
            if hops is None:
                if diagnostics is not None:
                    diagnostics["completion_unreachable"] = diagnostics.get(
                        "completion_unreachable", 0) + 1
                continue
            saw_reachable = True
            field = geodesic_field(ctx, int(landmark_cell), blocked)
            distance = field.remaining_distance((x, y), cell)
            centre = graph.cell_center(int(landmark_cell))
            bearing, range_m = landmark_bearing_range(ctx, centre)
            task_status = _snapshot_task_status(ctx, int(landmark_cell))
            evidence = completion_enriched_eligibility(
                graph_hops=int(hops), reachable=math.isfinite(distance),
                continuous_geodesic_m=float(distance),
                bearing_body_rad=float(bearing), task_status=task_status)
            if diagnostics is not None:
                for reason in evidence["rejection_reasons"]:
                    diagnostics[reason] = diagnostics.get(reason, 0) + 1
            goal_type = _goal_material(ctx, str(name))
            if goal_type is None:
                if diagnostics is not None:
                    diagnostics["bound_landmark_material_missing_or_ambiguous"] = \
                        diagnostics.get(
                            "bound_landmark_material_missing_or_ambiguous", 0) + 1
                continue
            if not evidence["eligible"]:
                continue
            record = _state_record(
                boundary=boundary, cell=cell, name=str(name),
                landmark_cell=int(landmark_cell), goal_type=goal_type,
                hops=int(hops), distance=float(distance), bearing=float(bearing),
                range_m=float(range_m), centre=centre,
                body_clearance=body_clearance, clearance=clearance,
                completion_eligibility=evidence)
            key = (float(distance), str(name), int(landmark_cell), int(hops))
            eligible.append((key, record, field))
        if not eligible:
            return reject("no_completion_enriched_goal" if saw_reachable
                          else "no_reachable_landmark")
        _key, record, field = min(eligible, key=lambda row: row[0])
        return record, field, {"completion_enriched"}

    # Original general/safety designation and ordering, byte-for-byte in
    # substance: the closest reachable landmark at one or more graph edges.
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
        return reject("no_reachable_landmark")
    distance, name, landmark_cell, hops, field = best
    centre = graph.cell_center(int(landmark_cell))
    bearing, range_m = landmark_bearing_range(ctx, centre)
    goal_type = _goal_material(ctx, str(name))
    if goal_type is None:
        return reject("bound_landmark_material_missing_or_ambiguous")

    strata: set[str] = set()
    if hops >= 2:
        strata.add("general")
        if body_clearance <= SAFETY_ENRICHED_MAX_BODY_CLEARANCE_M:
            strata.add("safety_enriched")
    # Preserve the pre-successor default path for final-evaluation callers.
    if (requested_stratum is None
            and distance <= COMPLETION_ENRICHED_MAX_GEODESIC_M
            and abs(bearing) <= COMPLETION_ENRICHED_MAX_BEARING_RAD):
        strata.add("completion_enriched")
    if requested_stratum is not None and requested_stratum not in strata:
        return reject("no_stratum")
    if not strata:
        return reject("no_stratum")
    record = _state_record(
        boundary=boundary, cell=cell, name=str(name),
        landmark_cell=int(landmark_cell), goal_type=goal_type, hops=int(hops),
        distance=float(distance), bearing=float(bearing), range_m=float(range_m),
        centre=centre, body_clearance=body_clearance, clearance=clearance)
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
        _assert_unsealed_path(root)
        if not root.is_dir():
            continue
        for family_dir in sorted(root.iterdir()):
            _assert_unsealed_path(family_dir)
            if not family_dir.is_dir():
                continue
            for scene_dir in sorted(family_dir.iterdir()):
                _assert_unsealed_path(scene_dir)
                if scene_dir.name in excluded:
                    continue
                manifest = scene_dir / "manifest.json"
                genesis_scene = scene_dir / "genesis_scene.json"
                _assert_unsealed_path(manifest)
                _assert_unsealed_path(genesis_scene)
                if (not manifest.is_file() or not genesis_scene.is_file()):
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


def _metric_distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values
                         if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "q1": None, "median": None,
                "mean": None, "q3": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "q1": float(np.quantile(finite, 0.25)),
        "median": float(np.quantile(finite, 0.5)),
        "mean": float(np.mean(finite)),
        "q3": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def build_selector_feasibility_summary(
        *, family: str, allowed_scene_count: int,
        requested_strata: Sequence[str], scene_evidence: Sequence[dict[str, Any]],
        rejection_counts: dict[str, int]) -> dict[str, Any]:
    """Pure reducer for an identity-free, outcome-free family dry-run."""

    strata = tuple(str(value) for value in requested_strata)
    if not strata or any(value not in STRATA for value in strata):
        raise ValueError("dry-run requested unknown or empty strata")
    evidence_by_stratum: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in strata
    }
    seen_pairs: set[tuple[str, str]] = set()
    for row in scene_evidence:
        # Intentionally enumerate the permitted keys: a branch outcome, oracle
        # label or candidate result has no read path through this reducer.
        row_family = str(row["family"])
        scene_id = str(row["scene_id"])
        stratum = str(row["stratum"])
        if row_family != family or stratum not in evidence_by_stratum:
            raise ValueError("dry-run evidence family/stratum mismatch")
        pair = (scene_id, stratum)
        if pair in seen_pairs:
            raise ValueError("dry-run evidence must be first-eligible per scene/stratum")
        seen_pairs.add(pair)
        evidence_by_stratum[stratum].append({
            "scene_id": scene_id,
            "first_eligible_block": int(row["first_eligible_block"]),
            "continuous_geodesic_m": float(row["continuous_geodesic_m"]),
            "abs_bearing_rad": float(row["abs_bearing_rad"]),
            "graph_hops_diagnostic": int(row["graph_hops_diagnostic"]),
            "body_clearance_m": float(row["body_clearance_m"]),
        })

    per_stratum: dict[str, Any] = {}
    for stratum in strata:
        rows = sorted(evidence_by_stratum[stratum],
                      key=lambda row: (row["scene_id"], row["first_eligible_block"]))
        count = len(rows)
        required = int(POOLS["scorer_fit"]["strata"][stratum])
        per_stratum[stratum] = {
            "required_distinct_scenes": required,
            "eligible_distinct_scenes": count,
            "quota_pass": count >= required,
            "distributions": {
                "continuous_geodesic_m": _metric_distribution(
                    [row["continuous_geodesic_m"] for row in rows]),
                "abs_bearing_rad": _metric_distribution(
                    [row["abs_bearing_rad"] for row in rows]),
                "graph_hops_diagnostic": _metric_distribution(
                    [row["graph_hops_diagnostic"] for row in rows]),
                "first_eligible_block": _metric_distribution(
                    [row["first_eligible_block"] for row in rows]),
                "body_clearance_m": _metric_distribution(
                    [row["body_clearance_m"] for row in rows]),
            },
            "scene_evidence": rows,
        }
    return {
        "family": str(family),
        "allowed_scene_count": int(allowed_scene_count),
        "scanned_scene_count": int(allowed_scene_count),
        "requested_strata": list(strata),
        "per_stratum": per_stratum,
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejection_counts.items())
        },
        "all_requested_quotas_pass": all(
            row["quota_pass"] for row in per_stratum.values()),
    }


def _selector_scene_evidence(family: str, scene_id: str, stratum: str,
                             block_index: int,
                             record: dict[str, Any]) -> dict[str, Any]:
    goal = record["goal"]
    return {
        "family": str(family),
        "scene_id": str(scene_id),
        "stratum": str(stratum),
        "first_eligible_block": int(block_index),
        "continuous_geodesic_m": float(goal["start_geodesic_m"]),
        "abs_bearing_rad": abs(float(goal["bearing_body_rad"])),
        "graph_hops_diagnostic": int(goal["graph_edges"]),
        "body_clearance_m": float(record["body_clearance_m"]),
    }


def _scan_selector_scene(*, family: str, scene_dir: Path,
                         requested_strata: Sequence[str], ctx: Any
                         ) -> dict[str, Any]:
    """Scan one exact scene; a native crash cannot yield a result object."""

    evidence: list[dict[str, Any]] = []
    rejections: dict[str, int] = {}
    topology = V12.link_topology(ctx)
    ctx.begin_episode()
    found_in_scene: set[str] = set()
    for block_idx in range(WARMUP_BLOCKS_MAX):
        ctx.drive_one_block()
        if block_idx + 1 < WARMUP_BLOCKS_MIN:
            continue
        for stratum in requested_strata:
            if stratum in found_in_scene:
                continue
            local_diagnostics: dict[str, int] = {}
            verdict = classify_state(
                ctx, topology, requested_stratum=stratum,
                diagnostics=local_diagnostics)
            for reason, count in local_diagnostics.items():
                key = reason.split(":")[0]
                rejections[key] = rejections.get(key, 0) + int(count)
            if isinstance(verdict, str):
                continue
            record, _field, _eligible = verdict
            evidence.append(_selector_scene_evidence(
                family, scene_dir.name, stratum, block_idx + 1, record))
            found_in_scene.add(stratum)
        if len(found_in_scene) == len(requested_strata):
            break
    return {
        "family": family,
        "scene_id": scene_dir.name,
        "scene_evidence": evidence,
        "rejection_counts": {
            str(key): int(value) for key, value in sorted(rejections.items())
        },
    }


def _selector_feasibility_scene_task(
        family: str, scene_dir: Path, task_index: int) -> dict[str, Any]:
    manifest = scene_dir / "manifest.json"
    genesis_scene = scene_dir / "genesis_scene.json"
    _assert_unsealed_path(scene_dir)
    _assert_unsealed_path(manifest)
    _assert_unsealed_path(genesis_scene)
    payload = {
        "schema": "go2_scorer_fit_selector_feasibility_scene_task_v1",
        "family": family,
        "task_index_within_family": int(task_index),
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir.resolve()),
        "split": scene_dir.parent.parent.name,
        "drive_seed": int(V1._drive_seed(scene_dir.name)),
        "scene_manifest_sha256": file_sha256(manifest),
        "scene_manifest_byte_count": manifest.stat().st_size,
        "genesis_scene_sha256": file_sha256(genesis_scene),
        "genesis_scene_byte_count": genesis_scene.stat().st_size,
        "requested_strata": list(STRATA),
    }
    payload["scene_task_digest"] = canonical_digest(payload)
    return payload


def build_selector_feasibility_task_census(
        *, pool: dict[str, Sequence[Path]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Freeze every allowed scene task before any isolated worker starts."""

    if set(pool) != set(STATE_SELECTOR.REQUIRED_FAMILIES):
        raise RuntimeError("selector-feasibility task census family set changed")
    families: list[dict[str, Any]] = []
    all_task_digests: list[str] = []
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        scenes = sorted(pool[family], key=lambda path: path.name)
        if len({scene.name for scene in scenes}) != len(scenes):
            raise RuntimeError(
                f"selector-feasibility task census repeats a scene in {family}")
        tasks = [
            _selector_feasibility_scene_task(family, scene, index)
            for index, scene in enumerate(scenes)
        ]
        all_task_digests.extend(task["scene_task_digest"] for task in tasks)
        families.append({
            "family": family,
            "allowed_scene_count": len(tasks),
            "tasks": tasks,
            "family_task_set_digest": canonical_digest(
                [task["scene_task_digest"] for task in tasks]),
        })
    payload = {
        "schema": SELECTOR_FEASIBILITY_TASK_CENSUS_SCHEMA,
        "status": "FROZEN_OUTCOME_FREE_EXHAUSTIVE_SCENE_TASK_CENSUS",
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "exclusion_binding_digest": exclusion_binding_digest,
        "family_count": len(families),
        "scene_task_count": len(all_task_digests),
        "families": families,
        "scene_task_set_digest": canonical_digest(all_task_digests),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_task_census_digest"] = \
        canonical_digest(payload)
    return payload


def _validate_selector_feasibility_task_census(
        census: dict[str, Any], *, pool: dict[str, Sequence[Path]],
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> None:
    _verify_self_digest(
        census, "state_selector_feasibility_task_census_digest",
        "state-selector feasibility task census")
    expected = build_selector_feasibility_task_census(
        pool=pool, source=source,
        successor_selection_digest=successor_selection_digest,
        exclusion_binding_digest=exclusion_binding_digest)
    if census != expected:
        raise RuntimeError(
            "selector-feasibility task census differs from the exact allow-list")


def _issue_selector_feasibility_task_census(
        *, out: Path, pool: dict[str, Sequence[Path]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    path = out / SELECTOR_FEASIBILITY_TASK_CENSUS_NAME
    expected = build_selector_feasibility_task_census(
        pool=pool, source=source,
        successor_selection_digest=successor_selection_digest,
        exclusion_binding_digest=exclusion_binding_digest)
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            _validate_selector_feasibility_task_census(
                existing, pool=pool, source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest)
        except Exception:
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "selector-feasibility task census changed after outcomes")
            _preserve_invalid(path, out, "selector-feasibility-task-census-invalid")
        else:
            return existing
    elif path.exists():
        if _outcome_generation_started(out):
            raise RuntimeError(
                "selector-feasibility task census path changed after outcomes")
        _preserve_invalid(path, out, "selector-feasibility-task-census-invalid")
    atomic_json(path, expected)
    return expected


def _selector_feasibility_family_tasks(
        census: dict[str, Any], family: str) -> list[dict[str, Any]]:
    matches = [row for row in census["families"] if row["family"] == family]
    if len(matches) != 1:
        raise RuntimeError(f"task census family lookup is ambiguous for {family}")
    return list(matches[0]["tasks"])


def _selector_feasibility_scene_shard_path(
        out: Path, task: dict[str, Any]) -> Path:
    return (out / SELECTOR_FEASIBILITY_SCENE_SHARD_ROOT / task["family"]
            / f"{task['scene_task_digest']}.json")


def _build_selector_feasibility_scene_shard(
        *, task: dict[str, Any], scene_result: dict[str, Any],
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str, exclusion_binding_digest: str,
        runtime_s: float) -> dict[str, Any]:
    payload = {
        "schema": SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA,
        "status": SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS,
        "complete": True,
        "binding_receipt": False,
        "eligibility_verdict_inferred_from_process_exit": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_task_census_digest": task_census_digest,
        "exclusion_binding_digest": exclusion_binding_digest,
        "task": task,
        "scene_result": scene_result,
        "runtime_s": round(float(runtime_s), 6),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_scene_shard_digest"] = \
        canonical_digest(payload)
    return payload


def _validate_selector_feasibility_scene_shard(
        shard: dict[str, Any], *, expected_task: dict[str, Any],
        expected_task_census_digest: str, source: dict[str, Any],
        expected_successor_selection_digest: str,
        expected_exclusion_binding_digest: str) -> None:
    _verify_self_digest(
        shard, "state_selector_feasibility_scene_shard_digest",
        f"selector-feasibility scene shard {expected_task['scene_id']}")
    if (shard.get("schema") != SELECTOR_FEASIBILITY_SCENE_SHARD_SCHEMA
            or shard.get("status") != SELECTOR_FEASIBILITY_SCENE_SHARD_STATUS
            or shard.get("complete") is not True
            or shard.get("binding_receipt") is not False
            or shard.get("eligibility_verdict_inferred_from_process_exit") is not False
            or shard.get("source_repository_commit")
            != source["source_repository_commit"]
            or shard.get("clean_source_binding_digest") != canonical_digest(source)
            or shard.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or shard.get("successor_selection_digest")
            != expected_successor_selection_digest
            or shard.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or shard.get("state_selector_feasibility_task_census_digest")
            != expected_task_census_digest
            or shard.get("exclusion_binding_digest")
            != expected_exclusion_binding_digest
            or shard.get("task") != expected_task
            or any(shard.get(key) not in (False, 0)
                   for key in SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS)):
        raise RuntimeError(
            f"selector-feasibility scene shard {expected_task['scene_id']} binding failed")
    runtime_s = shard.get("runtime_s")
    if (isinstance(runtime_s, bool) or not isinstance(runtime_s, (int, float))
            or not math.isfinite(float(runtime_s)) or float(runtime_s) < 0.0):
        raise RuntimeError("selector-feasibility scene runtime is invalid")
    result = shard.get("scene_result")
    if (not isinstance(result, dict)
            or set(result) != {
                "family", "scene_id", "scene_evidence", "rejection_counts"}
            or result.get("family") != expected_task["family"]
            or result.get("scene_id") != expected_task["scene_id"]
            or not isinstance(result.get("scene_evidence"), list)
            or not isinstance(result.get("rejection_counts"), dict)):
        raise RuntimeError("selector-feasibility scene result is malformed")
    seen: set[str] = set()
    evidence_keys = {
        "family", "scene_id", "stratum", "first_eligible_block",
        "continuous_geodesic_m", "abs_bearing_rad", "graph_hops_diagnostic",
        "body_clearance_m",
    }
    for evidence in result["scene_evidence"]:
        if (not isinstance(evidence, dict)
                or set(evidence) != evidence_keys
                or evidence.get("family") != expected_task["family"]
                or evidence.get("scene_id") != expected_task["scene_id"]
                or evidence.get("stratum") not in STRATA
                or evidence["stratum"] in seen):
            raise RuntimeError("selector-feasibility scene evidence is malformed")
        seen.add(str(evidence["stratum"]))
    if any(not isinstance(key, str)
           or isinstance(value, bool) or not isinstance(value, int) or value < 0
           for key, value in result["rejection_counts"].items()):
        raise RuntimeError("selector-feasibility scene rejections are malformed")


def _load_completed_selector_feasibility_scene_shard(
        path: Path, *, expected_task: dict[str, Any],
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        shard = json.loads(path.read_text())
        _validate_selector_feasibility_scene_shard(
            shard, expected_task=expected_task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        return shard
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def _execute_selector_feasibility_scene_worker(
        *, args: argparse.Namespace, task: dict[str, Any], path: Path,
        task_census_digest: str, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Scan and atomically receipt one scene before releasing native state."""

    started = time.time()
    shared = V1._load_shared(args.backend)
    scene_dir = Path(task["scene_dir"])
    ctx = V1.build_context(
        scene_dir, seed=int(task["drive_seed"]), backend=args.backend,
        shared=shared)
    try:
        result = _scan_selector_scene(
            family=str(task["family"]), scene_dir=scene_dir,
            requested_strata=STRATA, ctx=ctx)
        payload = _build_selector_feasibility_scene_shard(
            task=task, scene_result=result,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest,
            runtime_s=time.time() - started)
        _validate_selector_feasibility_scene_shard(
            payload, expected_task=task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        atomic_json(path, payload)
        return payload
    finally:
        # Native teardown has historically SIGSEGV'd.  A complete scene census
        # must be durable before either reference is released or GC is forced.
        _FIELD_CACHE.clear()
        del ctx
        gc.collect()


def _selector_feasibility_family_row(
        summary: dict[str, Any], requested_strata: Sequence[str]
        ) -> dict[str, Any]:
    """Project one exhaustive family scan into the binding receipt schema."""

    strata: dict[str, Any] = {}
    for stratum in requested_strata:
        evidence = summary["per_stratum"][stratum]
        strata[stratum] = {
            "required_distinct_scenes": evidence["required_distinct_scenes"],
            "eligible_distinct_scenes": evidence["eligible_distinct_scenes"],
            "verdict": "PASS" if evidence["quota_pass"] else "FAIL",
            "distributions": evidence["distributions"],
            "scene_evidence": evidence["scene_evidence"],
        }
    return {
        "family": summary["family"],
        "allowed_scene_count": summary["allowed_scene_count"],
        "scanned_scene_count": summary["scanned_scene_count"],
        "all_allowed_scenes_scanned": (
            summary["scanned_scene_count"] == summary["allowed_scene_count"]),
        "verdict": "PASS" if summary["all_requested_quotas_pass"] else "FAIL",
        "strata": strata,
        "rejection_counts": summary["rejection_counts"],
    }


def build_selector_feasibility_receipt_from_family_reductions(
        *, reductions: Sequence[dict[str, Any]], source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> dict[str, Any]:
    """Pure deterministic reducer over eight validated exhaustive censuses."""

    _verify_self_digest(
        task_census, "state_selector_feasibility_task_census_digest",
        "state-selector feasibility task census")
    if (task_census.get("source_repository_commit")
            != source["source_repository_commit"]
            or task_census.get("clean_source_binding_digest")
            != canonical_digest(source)
            or task_census.get("bound_implementations_digest")
            != source["bound_implementations_digest"]
            or task_census.get("successor_selection_digest")
            != successor_selection_digest
            or task_census.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or task_census.get("exclusion_binding_digest")
            != exclusion_binding_digest):
        raise RuntimeError("selector-feasibility reducer task census binding failed")
    by_family: dict[str, dict[str, Any]] = {}
    task_census_digest = str(
        task_census["state_selector_feasibility_task_census_digest"])
    for reduction in reductions:
        family = str(reduction.get("family", ""))
        if family in by_family:
            raise RuntimeError("selector-feasibility reducer received a repeated family")
        _verify_self_digest(
            reduction, "family_reduction_digest",
            f"selector-feasibility family reduction {family}")
        tasks = _selector_feasibility_family_tasks(task_census, family)
        scene_bindings = reduction.get("scene_shards")
        exact_scene_coverage = (
            isinstance(scene_bindings, list)
            and len(scene_bindings) == len(tasks)
            and all(
                isinstance(row, dict)
                and set(row) == {
                    "family", "scene_id", "scene_task_digest",
                    "scene_shard_digest"}
                and row["family"] == family
                and row["scene_id"] == task["scene_id"]
                and row["scene_task_digest"] == task["scene_task_digest"]
                and _is_sha256(row["scene_shard_digest"])
                for row, task in zip(scene_bindings, tasks)))
        family_result = reduction.get("family_result", {})
        if (reduction.get("schema")
                != "go2_scorer_fit_selector_feasibility_family_reduction_v1"
                or reduction.get("task_census_digest") != task_census_digest
                or reduction.get("scene_task_count") != len(tasks)
                or not exact_scene_coverage
                or family_result.get("family") != family
                or family_result.get("allowed_scene_count") != len(tasks)
                or family_result.get("scanned_scene_count") != len(tasks)
                or family_result.get("all_allowed_scenes_scanned") is not True
                or isinstance(reduction.get("runtime_s"), bool)
                or not isinstance(reduction.get("runtime_s"), (int, float))
                or not math.isfinite(float(reduction["runtime_s"]))
                or float(reduction["runtime_s"]) < 0.0):
            raise RuntimeError(
                f"selector-feasibility family reduction {family} is malformed")
        by_family[family] = reduction
    required_families = tuple(STATE_SELECTOR.REQUIRED_FAMILIES)
    if set(by_family) != set(required_families):
        raise RuntimeError("selector-feasibility reducer requires all eight families")
    ordered = [by_family[family] for family in required_families]
    family_rows = [reduction["family_result"] for reduction in ordered]
    scene_shard_lineage = [
        row for reduction in ordered for row in reduction["scene_shards"]
    ]
    passed = all(row["verdict"] == "PASS" for row in family_rows)
    payload = {
        "schema": SELECTOR_FEASIBILITY_SCHEMA,
        "status": (SELECTOR_FEASIBILITY_PASS_STATUS if passed
                   else "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"),
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_task_census_digest": task_census_digest,
        "scene_task_count": task_census["scene_task_count"],
        "scene_shard_count": len(scene_shard_lineage),
        "scene_shard_lineage": scene_shard_lineage,
        "scene_shard_lineage_digest": canonical_digest(scene_shard_lineage),
        "family_count": len(family_rows),
        "strata": list(STRATA),
        "required_distinct_scenes_per_stratum": 5,
        "families": family_rows,
        "exclusion_binding_digest": exclusion_binding_digest,
        "runtime_s": round(math.fsum(
            float(reduction["runtime_s"]) for reduction in ordered), 6),
        "reducer_version": SELECTOR_FEASIBILITY_REDUCER_VERSION,
        "family_reduction_digests": {
            family: by_family[family][
                "family_reduction_digest"]
            for family in required_families
        },
        "scene_subprocess_isolation": True,
        "resume_reuses_only_valid_complete_scene_shards": True,
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_receipt_digest"] = canonical_digest(payload)
    return payload


def _run_selector_feasibility_scene_subprocess(
        args: argparse.Namespace, task: dict[str, Any]) -> int:
    """Run exactly one pre-bound scene task in a fresh native process."""

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--pool", "scorer_fit", "--stage", "selector-feasibility",
        "--family", str(task["family"]),
        "--selector-scene-id", str(task["scene_id"]),
        "--backend", str(args.backend),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, check=False)
    return int(completed.returncode)


def _collect_selector_feasibility_scene_shards(
        *, args: argparse.Namespace, out: Path,
        tasks: Sequence[dict[str, Any]], task_census_digest: str,
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> list[dict[str, Any]]:
    """Resume exact complete scenes; a process exit is never eligibility data."""

    shards: list[dict[str, Any]] = []
    for task in tasks:
        path = _selector_feasibility_scene_shard_path(out, task)
        shard = _load_completed_selector_feasibility_scene_shard(
            path, expected_task=task, task_census_digest=task_census_digest,
            source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest)
        if shard is None:
            if _outcome_generation_started(out):
                state = "invalid" if path.exists() else "missing"
                raise RuntimeError(
                    f"selector-feasibility scene shard is {state} after outcomes")
            if path.exists():
                preserved = _preserve_invalid(
                    path, out, "selector-feasibility-scene-invalid")
                print(f"[recovery] preserved invalid scene census {preserved}",
                      flush=True)
            print(
                "[selector-feasibility] isolated exhaustive scene census: "
                f"{task['family']}/{task['scene_id']}", flush=True)
            return_code = _run_selector_feasibility_scene_subprocess(args, task)
            shard = _load_completed_selector_feasibility_scene_shard(
                path, expected_task=task, task_census_digest=task_census_digest,
                source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest)
            if shard is None:
                raise RuntimeError(
                    "isolated selector-feasibility scene "
                    f"{task['family']}/{task['scene_id']} exited {return_code} "
                    "without a valid durable census; no eligibility conclusion "
                    "was recorded")
            if return_code != 0:
                print(
                    "[recovery] retained valid atomic scene census despite "
                    f"worker return code {return_code}: "
                    f"{task['family']}/{task['scene_id']}", flush=True)
        else:
            print(
                "[selector-feasibility] retained valid exhaustive scene census: "
                f"{task['family']}/{task['scene_id']}", flush=True)
        shards.append(shard)
    return shards


def _reduce_selector_feasibility_family_scene_shards(
        *, family: str, tasks: Sequence[dict[str, Any]],
        shards: Sequence[dict[str, Any]], task_census_digest: str,
        source: dict[str, Any], successor_selection_digest: str,
        exclusion_binding_digest: str) -> dict[str, Any]:
    """Deterministically reduce the exact scene census for one family."""

    if len(tasks) != len(shards):
        raise RuntimeError(f"selector-feasibility family {family} scene count changed")
    by_task_digest: dict[str, dict[str, Any]] = {}
    for shard in shards:
        digest = str(shard.get("task", {}).get("scene_task_digest", ""))
        if digest in by_task_digest:
            raise RuntimeError(
                f"selector-feasibility family {family} repeats a scene shard")
        by_task_digest[digest] = shard
    expected_task_digests = [str(task["scene_task_digest"]) for task in tasks]
    if set(by_task_digest) != set(expected_task_digests):
        raise RuntimeError(
            f"selector-feasibility family {family} scene task coverage changed")
    scene_evidence: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    scene_bindings: list[dict[str, Any]] = []
    runtime_values: list[float] = []
    for task in tasks:
        shard = by_task_digest[str(task["scene_task_digest"])]
        _validate_selector_feasibility_scene_shard(
            shard, expected_task=task,
            expected_task_census_digest=task_census_digest, source=source,
            expected_successor_selection_digest=successor_selection_digest,
            expected_exclusion_binding_digest=exclusion_binding_digest)
        result = shard["scene_result"]
        scene_evidence.extend(result["scene_evidence"])
        for reason, count in result["rejection_counts"].items():
            rejection_counts[reason] = rejection_counts.get(reason, 0) + int(count)
        runtime_values.append(float(shard["runtime_s"]))
        scene_bindings.append({
            "family": family,
            "scene_id": task["scene_id"],
            "scene_task_digest": task["scene_task_digest"],
            "scene_shard_digest":
                shard["state_selector_feasibility_scene_shard_digest"],
        })
    summary = build_selector_feasibility_summary(
        family=family, allowed_scene_count=len(tasks),
        requested_strata=STRATA, scene_evidence=scene_evidence,
        rejection_counts=rejection_counts)
    payload = {
        "schema": "go2_scorer_fit_selector_feasibility_family_reduction_v1",
        "family": family,
        "task_census_digest": task_census_digest,
        "scene_task_count": len(tasks),
        "scene_shards": scene_bindings,
        "family_result": _selector_feasibility_family_row(summary, STRATA),
        "runtime_s": round(math.fsum(runtime_values), 6),
    }
    payload["family_reduction_digest"] = canonical_digest(payload)
    return payload


def _reduce_selector_feasibility_families(
        *, args: argparse.Namespace, out: Path, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce exact scene shards family-by-family without loading Genesis."""

    reductions: list[dict[str, Any]] = []
    task_census_digest = str(
        task_census["state_selector_feasibility_task_census_digest"])
    for family in STATE_SELECTOR.REQUIRED_FAMILIES:
        tasks = _selector_feasibility_family_tasks(task_census, family)
        scene_shards = _collect_selector_feasibility_scene_shards(
            args=args, out=out, tasks=tasks,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest)
        reductions.append(_reduce_selector_feasibility_family_scene_shards(
            family=family, tasks=tasks, shards=scene_shards,
            task_census_digest=task_census_digest, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest))
    return reductions


def _load_completed_selector_feasibility(
        path: Path, *, source: dict[str, Any],
        successor_selection_digest: str,
        exclusion_binding_digest: str,
        task_census: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the binding receipt from every current durable scene shard."""

    if not path.is_file():
        return None
    try:
        existing = json.loads(path.read_text())
        task_census_digest = str(
            task_census["state_selector_feasibility_task_census_digest"])
        reductions: list[dict[str, Any]] = []
        for family in STATE_SELECTOR.REQUIRED_FAMILIES:
            tasks = _selector_feasibility_family_tasks(task_census, family)
            shards: list[dict[str, Any]] = []
            for task in tasks:
                shard = _load_completed_selector_feasibility_scene_shard(
                    _selector_feasibility_scene_shard_path(path.parent, task),
                    expected_task=task,
                    task_census_digest=task_census_digest, source=source,
                    successor_selection_digest=successor_selection_digest,
                    exclusion_binding_digest=exclusion_binding_digest)
                if shard is None:
                    return None
                shards.append(shard)
            reductions.append(_reduce_selector_feasibility_family_scene_shards(
                family=family, tasks=tasks, shards=shards,
                task_census_digest=task_census_digest, source=source,
                successor_selection_digest=successor_selection_digest,
                exclusion_binding_digest=exclusion_binding_digest))
        expected = build_selector_feasibility_receipt_from_family_reductions(
            reductions=reductions, source=source,
            successor_selection_digest=successor_selection_digest,
            exclusion_binding_digest=exclusion_binding_digest,
            task_census=task_census)
        if existing != expected:
            return None
        if existing["status"] == SELECTOR_FEASIBILITY_PASS_STATUS:
            STATE_SELECTOR.validate_state_selector_feasibility_receipt(
                existing,
                expected_source_commit=str(source["source_repository_commit"]),
                expected_successor_selection_digest=successor_selection_digest,
            )
        return existing
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def stage_selector_feasibility(args: argparse.Namespace) -> int:
    """Run the outcome-free all-scene feasibility gate or a scoped diagnostic."""

    if args.pool != "scorer_fit":
        raise RuntimeError("selector feasibility is defined only for scorer_fit")
    if args.backend != "cpu":
        raise RuntimeError("the frozen selector feasibility backend is cpu")
    STATE_SELECTOR.validate_authority_artifacts()
    source = clean_source_binding()
    if source.get("source_repository_clean") is not True:
        raise RuntimeError("selector feasibility requires a clean source repository")
    successor_digest = selection_digest()
    selector_scene_id = getattr(args, "selector_scene_id", None)
    if (selector_scene_id is not None
            and (args.family is None or args.stratum is not None)):
        raise RuntimeError(
            "--selector-scene-id requires one --family and no --stratum")
    requested_families = ([str(args.family)] if args.family is not None
                          else list(STATE_SELECTOR.REQUIRED_FAMILIES))
    requested_strata = ([str(args.stratum)] if args.stratum is not None
                        else list(STATE_SELECTOR.REQUIRED_STRATA))
    binding_run = (
        args.family is None and args.stratum is None
        and selector_scene_id is None)
    scene_worker_run = selector_scene_id is not None
    out = OUT_ROOT / "scorer_fit"
    out.mkdir(parents=True, exist_ok=True)
    if binding_run:
        path = out / SELECTOR_FEASIBILITY_RECEIPT_NAME
    elif scene_worker_run:
        path = None
    else:
        path = out / ("state_selector_feasibility_diagnostic_"
                      + "-".join(requested_families) + "_"
                      + "-".join(requested_strata) + ".json")

    pool, exclusion = scene_pool("scorer_fit")
    unknown = sorted(set(requested_families) - set(pool))
    if unknown:
        raise RuntimeError(f"unknown selector-feasibility families: {unknown}")
    exclusion_digest = canonical_digest(exclusion)
    census: dict[str, Any] | None = None
    if not scene_worker_run:
        census = _issue_selector_feasibility_task_census(
            out=out, pool=pool, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
    elif scene_worker_run:
        census_path = out / SELECTOR_FEASIBILITY_TASK_CENSUS_NAME
        if not census_path.is_file():
            raise RuntimeError(
                "isolated scene worker requires the frozen task census")
        census = json.loads(census_path.read_text())
        _validate_selector_feasibility_task_census(
            census, pool=pool, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)

    if binding_run:
        assert census is not None and path is not None
        existing = _load_completed_selector_feasibility(
            path, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return (0 if existing.get("status")
                    == SELECTOR_FEASIBILITY_PASS_STATUS else 1)

    if binding_run:
        assert census is not None and path is not None
        reductions = _reduce_selector_feasibility_families(
            args=args, out=out, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        payload = build_selector_feasibility_receipt_from_family_reductions(
            reductions=reductions, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest,
            task_census=census)
        if path.exists():
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "selector-feasibility receipt changed after outcomes started")
            _preserve_invalid(path, out, "selector-feasibility-superseded")
        atomic_json(path, payload)
        passed = payload["status"] == SELECTOR_FEASIBILITY_PASS_STATUS
        if passed:
            STATE_SELECTOR.validate_state_selector_feasibility_receipt(
                payload,
                expected_source_commit=str(source["source_repository_commit"]),
                expected_successor_selection_digest=successor_digest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    if scene_worker_run:
        assert census is not None
        family = requested_families[0]
        matches = [
            task for task in _selector_feasibility_family_tasks(census, family)
            if task["scene_id"] == selector_scene_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"scene task lookup is ambiguous for {family}/{selector_scene_id}")
        task = matches[0]
        path = _selector_feasibility_scene_shard_path(out, task)
        census_digest = str(
            census["state_selector_feasibility_task_census_digest"])
        existing = _load_completed_selector_feasibility_scene_shard(
            path, expected_task=task, task_census_digest=census_digest,
            source=source, successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        if _outcome_generation_started(out):
            state = "invalid" if path.exists() else "missing"
            raise RuntimeError(
                f"selector-feasibility scene shard is {state} after outcomes")
        if path.exists():
            _preserve_invalid(path, out, "selector-feasibility-scene-invalid")
        payload = _execute_selector_feasibility_scene_worker(
            args=args, task=task, path=path,
            task_census_digest=census_digest, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Scoped diagnostics reuse the same exact per-scene workers and merely
    # reduce a requested view; they cannot satisfy the binding all-family gate.
    assert census is not None
    census_digest = str(
        census["state_selector_feasibility_task_census_digest"])
    families: list[dict[str, Any]] = []
    runtime_values: list[float] = []
    for family in requested_families:
        tasks = _selector_feasibility_family_tasks(census, family)
        scene_shards = _collect_selector_feasibility_scene_shards(
            args=args, out=out, tasks=tasks,
            task_census_digest=census_digest, source=source,
            successor_selection_digest=successor_digest,
            exclusion_binding_digest=exclusion_digest)
        evidence = [
            row for shard in scene_shards
            for row in shard["scene_result"]["scene_evidence"]
            if row["stratum"] in requested_strata
        ]
        rejections: dict[str, int] = {}
        for shard in scene_shards:
            runtime_values.append(float(shard["runtime_s"]))
            for reason, count in shard["scene_result"]["rejection_counts"].items():
                rejections[reason] = rejections.get(reason, 0) + int(count)
        families.append(build_selector_feasibility_summary(
            family=family, allowed_scene_count=len(tasks),
            requested_strata=requested_strata, scene_evidence=evidence,
            rejection_counts=rejections))
    family_rows = [
        _selector_feasibility_family_row(summary, requested_strata)
        for summary in families
    ]
    passed = all(row["verdict"] == "PASS" for row in family_rows)
    payload = {
        "schema": "go2_scorer_fit_state_selector_feasibility_diagnostic_v1",
        "status": ("PASS_OUTCOME_FREE_SCOPED_FEASIBILITY" if passed
                   else "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"),
        "complete": True,
        "binding_receipt": False,
        "source_repository_commit": source["source_repository_commit"],
        "clean_source_binding_digest": canonical_digest(source),
        "bound_implementations_digest": source["bound_implementations_digest"],
        "successor_selection_digest": successor_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "family_count": len(family_rows),
        "strata": list(requested_strata),
        "required_distinct_scenes_per_stratum": 5,
        "families": family_rows,
        "exclusion_binding_digest": exclusion_digest,
        "runtime_s": round(math.fsum(runtime_values), 6),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_diagnostic_digest"] = \
        canonical_digest(payload)
    assert path is not None
    if path.exists():
        _preserve_invalid(path, out, "selector-feasibility-superseded")
    atomic_json(path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


def stage_preserved_state_precontract_revalidation(
        args: argparse.Namespace) -> int:
    """Exactly redrive the valid paused 45 identities without branch outcomes."""

    if args.pool != "scorer_fit" or args.family is not None or args.stratum is not None:
        raise RuntimeError(
            "preserved-state precontract revalidation is one all-family scorer-fit gate"
        )
    if args.backend != "cpu":
        raise RuntimeError("preserved-state revalidation requires the CPU backend")
    STATE_SELECTOR.validate_authority_artifacts()
    oracle_completion_target_unchanged = _oracle_completion_target_unchanged()
    source = clean_source_binding()
    if source.get("source_repository_clean") is not True:
        raise RuntimeError("preserved-state revalidation requires clean source")
    successor_digest = selection_digest()
    feasibility_path = ROOT / STATE_SELECTOR.STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH
    if not feasibility_path.is_file():
        raise RuntimeError("selector feasibility must complete before revalidation")
    feasibility = json.loads(feasibility_path.read_text())
    STATE_SELECTOR.validate_state_selector_feasibility_receipt(
        feasibility,
        expected_source_commit=str(source["source_repository_commit"]),
        expected_successor_selection_digest=successor_digest,
    )
    out = OUT_ROOT / "scorer_fit"
    path = (
        ROOT
        / STATE_SELECTOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH
    )
    if path.parent != out:
        raise RuntimeError("precontract revalidation receipt escaped scorer-fit pool")
    if path.is_file():
        existing = json.loads(path.read_text())
        try:
            STATE_SELECTOR.validate_preserved_state_precontract_revalidation_receipt(
                existing,
                expected_source_commit=str(source["source_repository_commit"]),
                expected_successor_selection_digest=successor_digest,
                expected_feasibility_receipt_digest=feasibility[
                    "state_selector_feasibility_receipt_digest"],
                root=ROOT,
            )
        except Exception:
            # A complete, self-bound scientific failure is terminal and must
            # never be converted into a retry merely because it did not pass.
            try:
                _verify_self_digest(
                    existing,
                    "preserved_state_precontract_revalidation_receipt_digest",
                    "failed precontract revalidation receipt")
            except Exception:
                if _outcome_generation_started(out):
                    raise RuntimeError(
                        "invalid precontract receipt exists after outcomes"
                    )
                _preserve_invalid(path, out, "precontract-revalidation-invalid")
            else:
                if (existing.get("status")
                        == "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"
                        and existing.get("complete") is True
                        and existing.get("source_repository_commit")
                        == source["source_repository_commit"]
                        and existing.get("successor_selection_digest")
                        == successor_digest):
                    print(json.dumps(existing, indent=2, sort_keys=True))
                    return 1
                if _outcome_generation_started(out):
                    raise RuntimeError(
                        "mismatched precontract receipt exists after outcomes"
                    )
                _preserve_invalid(path, out, "precontract-revalidation-mismatch")
        else:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    predecessor_shards = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    pool, _exclusion = scene_pool("scorer_fit")
    allowed = {
        family: {scene.name: scene for scene in scenes}
        for family, scenes in pool.items()
    }
    shared = V1._load_shared(args.backend)
    shard_rows: list[dict[str, Any]] = []
    all_state_digests: list[str] = []
    global_failures: list[dict[str, Any]] = []
    for expected in STATE_SELECTOR.PRESERVED_STATE_SHARDS:
        family = str(expected["family"])
        shard = predecessor_shards[family]
        state_checks: list[dict[str, Any]] = []
        for entry in shard["states"]:
            check = {
                "state_id": str(entry["state_id"]),
                "state_identity_digest": str(entry["state_identity_digest"]),
                "exclusion_checks_pass": False,
                "exact_redrive_pass": False,
                "amended_classification_pass": False,
                "goal_binding_unchanged": False,
                "oracle_completion_target_unchanged": False,
                "snapshot_production_designated_goal_claim_unchanged": False,
                "production_task_completion_reset_unchanged": False,
                "completion_state_task_status_all_false": False,
                "failure_reason": None,
            }
            ctx = None
            try:
                scene_dir = allowed.get(family, {}).get(str(entry["scene_id"]))
                if scene_dir is None:
                    raise RuntimeError("scene is absent from strict successor allow-list")
                if (scene_dir.resolve() != Path(str(entry["scene_dir"])).resolve()
                        or scene_dir.parent.parent.name != str(entry["split"])
                        or int(V1._drive_seed(scene_dir.name))
                        != int(entry["drive_seed"])):
                    raise RuntimeError("scene path, split, or drive seed changed")
                INVALID_IDS.assert_disjoint(
                    [entry], label=f"preserved revalidation {entry['state_id']}")
                if (file_sha256(scene_dir / "manifest.json")
                        != entry["scene_manifest_sha256"]
                        or (scene_dir / "manifest.json").stat().st_size
                        != int(entry["scene_manifest_byte_count"])):
                    raise RuntimeError("scene manifest changed")
                check["exclusion_checks_pass"] = True
                ctx = V1.build_context(
                    scene_dir, seed=int(entry["drive_seed"]), backend=args.backend,
                    shared=shared)
                topology = V12.link_topology(ctx)
                ctx.begin_episode()
                for _block_index in range(int(entry["warmup_blocks"])):
                    ctx.drive_one_block()
                verdict = classify_state(
                    ctx, topology, requested_stratum=str(entry["stratum"]))
                if isinstance(verdict, str):
                    raise RuntimeError(f"amended classification failed: {verdict}")
                record, _field, eligible = verdict
                check["amended_classification_pass"] = (
                    str(entry["stratum"]) in eligible)
                check["goal_binding_unchanged"] = record["goal"] == entry["goal"]
                mismatch = _redrive_mismatch(entry, record, ctx)
                check["exact_redrive_pass"] = mismatch is None
                semantic_status = _snapshot_task_status(
                    ctx, int(entry["goal"]["landmark_cell"]))
                record_status_matches = (
                    str(entry["stratum"]) != "completion_enriched"
                    or record.get("snapshot_task_status") == semantic_status)
                check["oracle_completion_target_unchanged"] = \
                    oracle_completion_target_unchanged
                check[
                    "snapshot_production_designated_goal_claim_unchanged"
                ] = bool(
                    record_status_matches
                    and _snapshot_claim_semantics_unchanged(semantic_status))
                check["production_task_completion_reset_unchanged"] = bool(
                    record_status_matches
                    and _production_task_reset_semantics_unchanged(
                        semantic_status))
                if str(entry["stratum"]) == "completion_enriched":
                    task_clear = all(
                        semantic_status.get(key) is False for key in (
                            "task_completed", "goal_claimed", "terminated", "truncated"
                        )
                    )
                    check["completion_state_task_status_all_false"] = task_clear
                else:
                    check["completion_state_task_status_all_false"] = True
                if not all(check[key] is True for key in (
                        "exclusion_checks_pass", "exact_redrive_pass",
                        "amended_classification_pass", "goal_binding_unchanged",
                        "oracle_completion_target_unchanged",
                        "snapshot_production_designated_goal_claim_unchanged",
                        "production_task_completion_reset_unchanged",
                        "completion_state_task_status_all_false")):
                    raise RuntimeError(mismatch or "one or more revalidation checks failed")
            except Exception as exc:
                check["failure_reason"] = f"{type(exc).__name__}:{str(exc)[:200]}"
                global_failures.append(dict(check))
            finally:
                _FIELD_CACHE.clear()
                if ctx is not None:
                    del ctx
                gc.collect()
            state_checks.append(check)

        state_digests = sorted(
            str(state["state_identity_digest"]) for state in shard["states"])
        all_state_digests.extend(state_digests)
        failed = [row for row in state_checks if row["failure_reason"] is not None]
        shard_rows.append({
            **dict(expected),
            "revalidated_state_count": len(state_checks),
            "unchanged_state_identity_count": len(state_checks),
            "failed_state_count": len(failed),
            "exact_redrive_pass": not failed and all(
                row["exact_redrive_pass"] for row in state_checks),
            "amended_classification_pass": not failed and all(
                row["amended_classification_pass"] for row in state_checks),
            "completion_state_task_status_all_false": not failed and all(
                row["completion_state_task_status_all_false"]
                for row in state_checks),
            "exclusion_checks_pass": not failed and all(
                row["exclusion_checks_pass"] for row in state_checks),
            "goal_binding_unchanged": not failed and all(
                row["goal_binding_unchanged"] for row in state_checks),
            "oracle_completion_target_unchanged": not failed and all(
                row["oracle_completion_target_unchanged"]
                for row in state_checks),
            "snapshot_production_designated_goal_claim_unchanged":
                not failed and all(
                    row[
                        "snapshot_production_designated_goal_claim_unchanged"
                    ] for row in state_checks),
            "production_task_completion_reset_unchanged": not failed and all(
                row["production_task_completion_reset_unchanged"]
                for row in state_checks),
            "candidate_outcomes_loaded": False,
            "state_identity_digests": state_digests,
            "state_identity_set_digest": canonical_digest(state_digests),
            "state_checks": state_checks,
        })

    passed = not global_failures
    payload = {
        "schema": STATE_SELECTOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
        "status": ("PASS_PRECONTRACT_IDENTITY_REVALIDATION" if passed
                   else "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"),
        "complete": True,
        "source_repository_commit": source["source_repository_commit"],
        "successor_selection_digest": successor_digest,
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            feasibility["state_selector_feasibility_receipt_digest"],
        "predecessor_selection_digest":
            STATE_SELECTOR.PREDECESSOR_SELECTION_DIGEST,
        "predecessor_scorer_contract_digest":
            STATE_SELECTOR.PREDECESSOR_SCORER_CONTRACT_DIGEST,
        "candidate_outcomes_loaded": False,
        "candidate_allocation_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "preserved_state_count": len(all_state_digests),
        "state_identity_set_digest": canonical_digest(sorted(all_state_digests)),
        "shards": shard_rows,
        "failure_count": len(global_failures),
        "failures": global_failures,
    }
    payload["preserved_state_precontract_revalidation_receipt_digest"] = \
        canonical_digest(payload)
    atomic_json(path, payload)
    if passed:
        STATE_SELECTOR.validate_preserved_state_precontract_revalidation_receipt(
            payload,
            expected_source_commit=str(source["source_repository_commit"]),
            expected_successor_selection_digest=successor_digest,
            expected_feasibility_receipt_digest=feasibility[
                "state_selector_feasibility_receipt_digest"],
            root=ROOT,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


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
        "state_selector_amendment_digest":
            STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            launch["state_selector_feasibility_receipt_digest"],
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
            if spec["strata"] is None:
                verdict = classify_state(ctx, topology)
                if isinstance(verdict, str):
                    key = verdict.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
                record, _field, strata = verdict
                wanted = (["evaluation"] if found["evaluation"] < need["evaluation"]
                          else [])
            else:
                wanted = []
                attempted: list[str] = []
                record = None
                for name in STRATA:
                    if found[name] >= need[name]:
                        continue
                    verdict = classify_state(
                        ctx, topology, requested_stratum=name)
                    if isinstance(verdict, str):
                        attempted.append(f"{name}:{verdict.split(':')[0]}")
                        continue
                    record, _field, _eligible = verdict
                    wanted = [name]
                    break
                if not wanted:
                    key = ("no_requested_stratum" if not attempted
                           else "|".join(attempted))
                    reasons[key] = reasons.get(key, 0) + 1
                    continue
            if not wanted:
                reasons["stratum_already_full"] = reasons.get(
                    "stratum_already_full", 0) + 1
                continue
            stratum = wanted[0]
            assert record is not None
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
            if stratum == "completion_enriched":
                # These snapshot-time facts are part of the frozen identity,
                # not a later branch/outcome annotation.
                chosen["completion_eligibility"] = record[
                    "completion_eligibility"
                ]
                chosen["snapshot_task_status"] = record[
                    "snapshot_task_status"
                ]
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
            or manifest.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or manifest.get("state_selector_feasibility_receipt_digest")
            != _load_clean_source_launch_receipt()[
                "state_selector_feasibility_receipt_digest"]
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
    _validate_state_shard_provenance(manifest, pool=pool)
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
    if any(not _state_identity_matches_active_or_preserved(state)
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
            *ACTIVE_SELECTOR_BINDING_KEYS,
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
        revalidation_path = (
            ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
        )
        if not revalidation_path.is_file():
            raise RuntimeError("post-allocation state revalidation receipt is missing")
        revalidation = json.loads(revalidation_path.read_text())
        launch = _load_clean_source_launch_receipt()
        STATE_SELECTOR.validate_preserved_state_revalidation_receipt(
            revalidation,
            allocation_manifest=allocation,
            expected_source_commit=str(launch["source_repository_commit"]),
            expected_successor_selection_digest=selection_digest(),
            expected_feasibility_receipt_digest=str(
                launch["state_selector_feasibility_receipt_digest"]),
            expected_precontract_revalidation_receipt_digest=str(
                launch[
                    "preserved_state_precontract_revalidation_receipt_digest"
                ]),
            root=ROOT,
        )
        if (manifest.get("preserved_state_revalidation_receipt_digest")
                != revalidation[
                    "preserved_state_revalidation_receipt_digest"]):
            raise RuntimeError("state manifest phase-2 revalidation digest mismatch")
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
        **{key: manifest[key] for key in ACTIVE_SELECTOR_BINDING_KEYS},
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
        *ACTIVE_SELECTOR_BINDING_KEYS,
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
        "body_clearance": float(record["body_clearance_m"])
                          == float(entry["body_clearance_m"]),
        "clearance": float(record["clearance_m"])
                     == float(entry["clearance_m"]),
    }
    if entry.get("stratum") == "completion_enriched":
        if "completion_eligibility" in entry:
            comparisons.update({
                "completion_eligibility": record.get("completion_eligibility")
                                          == entry.get("completion_eligibility"),
                "snapshot_task_status": record.get("snapshot_task_status")
                                        == entry.get("snapshot_task_status"),
            })
        else:
            # The 45 phase-1 predecessor identities predate evidence fields;
            # their exact payloads cannot be changed.  They are admissible only
            # through the byte-bound predecessor set, and current redrive must
            # independently pass the successor predicate with all task flags
            # false before any branch is attempted.
            status = record.get("snapshot_task_status", {})
            comparisons.update({
                "preserved_completion_identity":
                    str(entry.get("state_identity_digest"))
                    in _preserved_states_by_digest(),
                "completion_successor_eligible": bool(
                    record.get("completion_eligibility", {}).get("eligible")),
                "completion_task_status": all(
                    status.get(key) is False for key in (
                        "task_completed", "goal_claimed", "terminated", "truncated"
                    )
                ),
            })
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

        verdict = classify_state(
            ctx, topology,
            requested_stratum=(entry["stratum"]
                               if manifest["pool"] == "scorer_fit" else None))
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
        **{key: manifest_bindings[key] for key in ACTIVE_SELECTOR_BINDING_KEYS},
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
            or payload.get("state_selector_amendment_digest")
            != STATE_SELECTOR.state_selector_amendment_digest()
            or payload.get("state_selector_feasibility_receipt_digest")
            != _load_clean_source_launch_receipt()[
                "state_selector_feasibility_receipt_digest"]
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


def _issue_preserved_state_revalidation(
        out: Path, allocation: dict[str, Any]) -> dict[str, Any]:
    """Issue phase 2 only after all 120 identities receive frozen masks."""

    launch = _load_clean_source_launch_receipt()
    expected = STATE_SELECTOR.build_preserved_state_revalidation_receipt(
        allocation_manifest=allocation,
        source_repository_commit=str(launch["source_repository_commit"]),
        successor_selection_digest=selection_digest(),
        state_selector_feasibility_receipt_digest=str(
            launch["state_selector_feasibility_receipt_digest"]),
        preserved_state_precontract_revalidation_receipt_digest=str(
            launch["preserved_state_precontract_revalidation_receipt_digest"]),
        root=ROOT,
    )
    path = ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
    if path.parent != out:
        raise RuntimeError("final preserved-state receipt path escaped scorer-fit pool")
    if path.is_file():
        existing = json.loads(path.read_text())
        try:
            STATE_SELECTOR.validate_preserved_state_revalidation_receipt(
                existing,
                allocation_manifest=allocation,
                expected_source_commit=str(launch["source_repository_commit"]),
                expected_successor_selection_digest=selection_digest(),
                expected_feasibility_receipt_digest=str(
                    launch["state_selector_feasibility_receipt_digest"]),
                expected_precontract_revalidation_receipt_digest=str(
                    launch[
                        "preserved_state_precontract_revalidation_receipt_digest"
                    ]),
                root=ROOT,
            )
        except Exception as exc:
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "final preserved-state revalidation changed after outcomes"
                ) from exc
            _preserve_invalid(path, out, "post-allocation-revalidation-invalid")
        else:
            if existing == expected:
                return existing
            if _outcome_generation_started(out):
                raise RuntimeError(
                    "final preserved-state revalidation changed after outcomes"
                )
            _preserve_invalid(path, out, "post-allocation-revalidation-mismatch")
    atomic_json(path, expected)
    return expected


def _build_state_shard_provenance(
        paths: Sequence[Path], shards: Sequence[dict[str, Any]],
        *, pool_name: str) -> list[dict[str, Any]]:
    """Bind mixed pre-outcome shard bytes without promoting old bindings."""

    by_family = {str(shard["family"]): (path, shard)
                 for path, shard in zip(paths, shards, strict=True)}
    preserved = {str(row["family"]): row
                 for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}
    rows: list[dict[str, Any]] = []
    for family in sorted(by_family):
        path, shard = by_family[family]
        row = {
            "family": family,
            "path": str(path.relative_to(ROOT)),
            "state_shard_digest": str(shard["state_shard_digest"]),
            "raw_sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
            "selection_provenance": (
                "PREDECESSOR_BYTE_EXACT_REVALIDATED"
                if pool_name == "scorer_fit" and family in preserved
                else "SUCCESSOR_SELECTOR_AMENDMENT_V1"
            ),
        }
        if family in preserved and pool_name == "scorer_fit":
            expected = preserved[family]
            for key in ("path", "state_shard_digest", "raw_sha256", "byte_count"):
                if row[key] != expected[key]:
                    raise RuntimeError(
                        f"preserved shard provenance {family}/{key} changed"
                    )
        rows.append(row)
    if len(rows) != EXPECTED_FAMILIES:
        raise RuntimeError("mixed state-shard provenance must cover eight families")
    return rows


def _validate_state_shard_provenance(
        manifest: dict[str, Any], *, pool: str) -> None:
    rows = manifest.get("state_shard_provenance")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FAMILIES:
        raise RuntimeError("state manifest lacks eight-row shard provenance")
    preserved = {str(row["family"]): row
                 for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS}
    seen: set[str] = set()
    observed_digests: dict[str, str] = {}
    for row in rows:
        family = str(row.get("family", ""))
        if family in seen:
            raise RuntimeError("state-shard provenance repeats a family")
        seen.add(family)
        path = (ROOT / str(row.get("path", ""))).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file():
            raise RuntimeError("state-shard provenance path is missing or escapes root")
        payload = json.loads(path.read_text())
        expected_provenance = (
            "PREDECESSOR_BYTE_EXACT_REVALIDATED"
            if pool == "scorer_fit" and family in preserved
            else "SUCCESSOR_SELECTOR_AMENDMENT_V1"
        )
        if (row.get("selection_provenance") != expected_provenance
                or row.get("raw_sha256") != file_sha256(path)
                or row.get("byte_count") != path.stat().st_size
                or row.get("state_shard_digest")
                != payload.get("state_shard_digest")):
            raise RuntimeError(f"state-shard provenance failed for {family}")
        if expected_provenance == "PREDECESSOR_BYTE_EXACT_REVALIDATED":
            expected = preserved[family]
            if any(row.get(key) != expected[key] for key in (
                    "path", "state_shard_digest", "raw_sha256", "byte_count")):
                raise RuntimeError(f"predecessor provenance changed for {family}")
        else:
            _validate_state_shard(payload, path, pool)
        observed_digests[family] = str(row["state_shard_digest"])
    if (seen != set(manifest.get("state_shard_digests", {}))
            or observed_digests != manifest.get("state_shard_digests")):
        raise RuntimeError("state-shard digest map and provenance disagree")


def merge_states(out: Path) -> int:
    """Merge exactly eight completed shards and freeze all branch identities."""

    pool_name = out.name
    if pool_name not in POOLS:
        raise RuntimeError(f"unknown output pool {pool_name!r}")
    paths = sorted(out.glob("state_shard_*.json"))
    if len(paths) != EXPECTED_FAMILIES:
        raise RuntimeError(f"expected eight state shards, found {len(paths)}")
    preserved_by_family = STATE_SELECTOR.load_preserved_state_shards(ROOT)
    preserved_families = set(preserved_by_family)
    shards: list[dict[str, Any]] = []
    successor_shards: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        family = str(payload.get("family", ""))
        if pool_name == "scorer_fit" and family in preserved_families:
            if payload != preserved_by_family[family]:
                raise RuntimeError(
                    f"preserved predecessor shard {family} changed before merge"
                )
        else:
            _validate_state_shard(payload, path, pool_name)
            successor_shards.append(payload)
        shards.append(payload)
    if pool_name == "scorer_fit" and len(successor_shards) != 5:
        raise RuntimeError(
            "scorer-fit merge requires exactly three byte-bound predecessor "
            "shards and five successor shards"
        )
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
    if any(not _state_identity_matches_active_or_preserved(state)
           for state in states):
        raise RuntimeError("merged state identities changed across selector phases")
    for index, state in enumerate(states):
        state["state_index"] = index

    common_keys = (
        "selection_digest", "scorer_fit_allocation_design_digest",
        "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "pre_identity_allocation_validation_digest",
        "invalid_scorer_identity_exclusion_digest",
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest", "candidate_bank_digest",
        *LAUNCH_BINDING_KEYS,
        "progress_contract_digest", "safety_contract_digest",
        "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "boundary_digest",
        "render_contract_digest", "preprocess_contract_digest",
        "textured_v03_renderer_contract_digest",
        "preprocessing_digest", "target_encoder_digest",
        "target_encoder_checkpoint_sha256", "genesis_backend",
    )
    active_shards = successor_shards if pool_name == "scorer_fit" else shards
    common = {key: active_shards[0][key] for key in common_keys}
    for shard in active_shards[1:]:
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
    if pool_name == "scorer_fit":
        preserved_revalidation = _issue_preserved_state_revalidation(
            out, allocation)
    else:
        preserved_path = (
            ROOT / STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH
        )
        if not preserved_path.is_file():
            raise RuntimeError(
                "final-evaluation identities require scorer-fit phase-2 revalidation"
            )
        preserved_revalidation = json.loads(preserved_path.read_text())
    manifest_bindings = {
        "pool": pool_name,
        **common,
        "candidate_allocation_post_identity_validation_digest":
            post_identity_validation_digest,
        "preserved_state_revalidation_receipt_digest":
            preserved_revalidation[
                "preserved_state_revalidation_receipt_digest"
            ],
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

    # Predecessor exclusion bindings are provenance only.  The active corpus
    # exclusion is the one shared by all five successor shards.
    exclusion_bindings = [shard["exclusion_binding"] for shard in active_shards]
    if any(value != exclusion_bindings[0] for value in exclusion_bindings[1:]):
        raise RuntimeError("state shards disagree on exclusions")
    rejections = {shard["family"]: shard["scene_rejection_reasons"]
                  for shard in shards}
    shard_provenance = _build_state_shard_provenance(
        paths, shards, pool_name=pool_name)
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
        "preserved_state_revalidation_receipt_digest":
            preserved_revalidation[
                "preserved_state_revalidation_receipt_digest"
            ],
        "branch_identity_set_digest": canonical_digest(sorted(branch_digests)),
        "exclusion_binding": exclusion_bindings[0],
        "state_shard_digests": {shard["family"]: shard["state_shard_digest"]
                                for shard in shards},
        "state_shard_provenance": shard_provenance,
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
            "phase1_preserved_valid_state_identity_count": 45,
            "phase1_preserved_valid_family_count": 3,
            "preserved_state_precontract_revalidation_receipt_digest":
                _load_clean_source_launch_receipt()[
                    "preserved_state_precontract_revalidation_receipt_digest"
                ],
            "preserved_state_revalidation_receipt_digest":
                preserved_revalidation[
                    "preserved_state_revalidation_receipt_digest"
                ],
            "valid_predecessor_state_shards": [
                dict(row) for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
            ],
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
            "invalid_attempt_decision": (
                "preserved the incomplete three-of-eight-family 45-state "
                "pre-outcome identity attempt under its superseded contract and "
                "selection; no branch, render, latent or outcome existed; the "
                "exact 45 scenes and all descendant identity namespaces are "
                "excluded from this corpus; no invalid artifact is mixed"
            ),
            "valid_paused_identity_decision": (
                "retained the separate byte-bound 45-state pre-outcome paused "
                "identity set after exact phase-1 redrive and selector checks; "
                "the active 120-state allocation and exact six-candidate masks "
                "were verified by phase 2 before branch identities were issued; "
                "predecessor shard and contract bindings remain provenance only"
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
                        choices=["allocation-preflight", "selector-feasibility",
                                 "revalidate-preserved",
                                 "states", "merge-states", "smoke", "branches"],
                        required=True)
    parser.add_argument("--family", default=None,
                        help="resolve one family only; shards merge via merge-states")
    parser.add_argument("--stratum", choices=STRATA, default=None,
                        help="scope selector-feasibility diagnostics to one stratum")
    parser.add_argument(
        "--selector-scene-id", default=None,
        help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=10**6)
    args = parser.parse_args()
    if args.state_offset < 0 or args.state_limit < 1:
        raise SystemExit("state offset must be nonnegative and limit positive")
    if (args.selector_scene_id is not None
            and args.stage != "selector-feasibility"):
        raise SystemExit(
            "--selector-scene-id is internal to --stage selector-feasibility")

    out = OUT_ROOT / args.pool
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "allocation-preflight":
        return issue_pre_identity_allocation_validation(out)
    if args.stage == "selector-feasibility":
        return stage_selector_feasibility(args)
    if args.stage == "revalidate-preserved":
        return stage_preserved_state_precontract_revalidation(args)
    if args.stage == "merge-states":
        return merge_states(out)
    if args.stage == "states":
        if args.family is None:
            raise SystemExit("--stage states requires exactly one --family shard")
        shard_path = out / f"state_shard_{args.family}.json"
        if (args.pool == "scorer_fit"
                and args.family in {
                    row["family"] for row in STATE_SELECTOR.PRESERVED_STATE_SHARDS
                }):
            # These exact source shards passed the phase-1 identity-only gate.
            # Never rewrite or regenerate them under the successor wrapper.
            _load_clean_source_launch_receipt()
            preserved = STATE_SELECTOR.load_preserved_state_shards(ROOT)
            if (not shard_path.is_file()
                    or json.loads(shard_path.read_text())
                    != preserved[args.family]):
                raise RuntimeError(
                    f"byte-bound preserved state shard {args.family} is missing "
                    "or changed; refusing replacement selection"
                )
            print(json.dumps({
                "recovery": "retained_phase1_revalidated_predecessor_identity_shard",
                "path": str(shard_path),
                "state_shard_digest":
                    preserved[args.family]["state_shard_digest"],
                "states": len(preserved[args.family]["states"]),
                "preserved_state_precontract_revalidation_receipt_digest":
                    _load_clean_source_launch_receipt()[
                        "preserved_state_precontract_revalidation_receipt_digest"
                    ],
            }, indent=2, sort_keys=True))
            return 0
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
