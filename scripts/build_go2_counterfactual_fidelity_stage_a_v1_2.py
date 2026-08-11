#!/usr/bin/env python3
"""Stage A: replay the frozen oracle-v1.2 pilot with corrected V03 evidence.

This is a narrow data-recovery/replay driver.  It does not select states, alter
the oracle, train a scorer, or open a world-model checkpoint.  The already
frozen successful twenty-state oracle-v1.2 manifest and its 240-row outcome
ledger are immutable witnesses.  Each registered branch is executed once on
the qualified CPU backend while the same execution captures H=1..4 base poses.
Those poses are rendered afterwards in the separate historical ``textured_v03``
scene, so rendering cannot perturb the physical branch.

Stages are deliberately sequential and resumable::

    --stage prepare   validate witnesses and issue the pre-execution identity
    --stage smoke     first state, exact candidate indices 0..5
    --stage branches  all 240 identities, retaining verified smoke rows

All durable records are atomic.  A record is complete only after its own
digest, bound source outcome, render receipts, and exact oracle equality have
validated.  A malformed partial record is preserved under ``invalid_attempts``
and only that exact registered identity is regenerated.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1 as V1  # noqa: E402
import run_go2_oracle_branch_pilot_v1_2 as V12  # noqa: E402
from lewm.oracle.go2_branch_oracle_v1_2 import (  # noqa: E402
    GeodesicField,
    oracle_digest as oracle_v1_2_digest,
    progress_digest,
    safety_digest,
)
from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    RENDER_CONTRACT,
    TARGET_ENCODER,
    preprocess_contract_digest,
    render_contract_digest,
    target_encoder_digest,
)
from lewm.oracle.go2_textured_v03_renderer import (  # noqa: E402
    BasePose,
    TexturedV03Renderer,
    capture_base_pose,
    raw_manifest_digest,
    renderer_contract_digest,
)
from scripts import build_dev_v03_proprio_action_manifest_v1 as M  # noqa: E402
from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402


STATUS = "FROZEN_ORACLE_V1_2_COUNTERFACTUAL_FIDELITY_STAGE_A"
OUT_ROOT = ROOT / ".generated/go2_counterfactual_fidelity_v1_2"
SOURCE_ROOT = ROOT / ".generated/go2_oracle_branch_pilot_v1_2"

SOURCE_STATE_MANIFEST_DIGEST = (
    "5f380bf7f49ef10437c7d9644f04dbef065f0550dfd30d0ec36208cda25d08cf"
)
SOURCE_FILES = {
    "state_manifest": {
        "path": ".generated/go2_oracle_branch_pilot_v1_2/state_manifest.json",
        "sha256": "1f76afa94a66eaec0049559f9a47d48a4b50543c0ad4c6cec5060ff5b5ab0d9e",
        "byte_count": 20_147,
    },
    "pilot_branches": {
        "path": ".generated/go2_oracle_branch_pilot_v1_2/pilot_branches.jsonl",
        "sha256": "761c0de85296db70e044a177a75cbd1f12181c506a375a8827946468c8a6ce4c",
        "byte_count": 654_272,
        "record_count": 240,
    },
    "gate_report": {
        "path": ".generated/go2_oracle_branch_pilot_v1_2/gate_report.json",
        "sha256": "e77bf3b27551aeeca2d5a2bfe92d04b949e08b3bf5e4e1f2168387a50c832834",
        "byte_count": 3_468,
    },
    "smoke": {
        "path": ".generated/go2_oracle_branch_pilot_v1_2/smoke.json",
        "sha256": "92f8cdf190eb687321ee1d8e342670b22ec675fb8ee8cb081ba70286d1863b9e",
        "byte_count": 1_159,
    },
}

CONTEXT_SLOTS = 3
SAMPLES_PER_SLOT = M.SAMPLES_PER_SLOT
PROPRIO_HISTORY = CONTEXT_SLOTS * SAMPLES_PER_SLOT
HORIZONS = 4
EXPECTED_STATES = 20
EXPECTED_CANDIDATES = 12
EXPECTED_BRANCHES = EXPECTED_STATES * EXPECTED_CANDIDATES
PREPROCESSING_DIGEST = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)

OUTCOME_FIELDS = (
    "state_id", "scene_id", "family", "split", "episode_id", "source_step",
    "warmup_blocks", "landmark_id", "landmark_cell", "graph_edges_to_landmark",
    "candidate", "primitives", "requested", "post_slew", "clipped",
    "blocks_completed", "truncated_at_block", "valid", "invalid_reason",
    "snapshot_digest", "candidate_bank_digest", "progress_contract_digest",
    "safety_contract_digest", "oracle_v1_2_digest", "state_manifest_digest",
    "start_geodesic_m", "final_geodesic_m", "progress", "contact_fraction",
    "clearance_cost", "stuck_fraction", "fall", "safety", "completion",
    "utility", "min_clearance_m", "evaluation_points",
)
ORACLE_LABEL_FIELDS = (
    "start_geodesic_m", "final_geodesic_m", "progress", "contact_fraction",
    "clearance_cost", "stuck_fraction", "fall", "safety", "completion",
    "utility", "min_clearance_m", "evaluation_points",
)


def canonical_digest(payload: Any) -> str:
    """Repository-standard canonical JSON SHA-256 used by Stage A."""

    return hashlib.sha256(json.dumps(
        V1._jsonable(payload), sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        sink.write(json.dumps(V1._jsonable(payload), indent=2, sort_keys=True) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        sink.write(value)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist a rename itself, not only the bytes preceding it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_self_digest(payload: Mapping[str, Any], key: str, label: str) -> None:
    expected = canonical_digest({name: value for name, value in payload.items()
                                 if name != key})
    if payload.get(key) != expected:
        raise RuntimeError(f"{label} self digest mismatch")


def _preserve_invalid(path: Path, out: Path, reason: str) -> Path:
    invalid_root = out / "invalid_attempts"
    invalid_root.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(path) if path.is_file() else "not-a-file"
    target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.invalid"
    suffix = 0
    while target.exists():
        suffix += 1
        target = invalid_root / f"{path.name}.{digest[:16]}.{reason}.{suffix}.invalid"
    path.rename(target)
    _fsync_directory(path.parent)
    if target.parent != path.parent:
        _fsync_directory(target.parent)
    return target


def _outcome_generation_started(out: Path) -> bool:
    """A pre-execution identity may be superseded only before this turns true."""

    durable_files = (
        "branch_rows.jsonl", "corpus_receipt.json", "smoke_receipt.json",
        "smoke_encoding_receipt.json", "latents_index.json", "encoding_summary.json",
    )
    if any((out / name).exists() for name in durable_files):
        return True
    for name in ("state_records", "row_records", "frames", "latents"):
        root = out / name
        if root.is_dir() and any(root.iterdir()):
            return True
    return False


def _assert_binding(path: Path, binding: Mapping[str, Any], label: str) -> None:
    if (not path.is_file() or path.stat().st_size != int(binding["byte_count"])
            or file_sha256(path) != binding["sha256"]):
        raise RuntimeError(f"frozen source witness changed: {label}")


@dataclass(frozen=True)
class AcceptedOutcome:
    row: dict[str, Any]
    line_index: int
    line_sha256: str
    line_byte_count: int


@dataclass(frozen=True)
class SourceEvidence:
    manifest: dict[str, Any]
    gate: dict[str, Any]
    outcomes: dict[tuple[str, int], AcceptedOutcome]
    witnesses: dict[str, Any]


def _outcome_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in OUTCOME_FIELDS}


def load_source_evidence() -> SourceEvidence:
    """Validate immutable pilot bytes and return an exact identity index."""

    for label, binding in SOURCE_FILES.items():
        _assert_binding(ROOT / str(binding["path"]), binding, label)
    manifest_path = ROOT / SOURCE_FILES["state_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    # The frozen pilot predates Stage A and used json.dumps(..., sort_keys=True)
    # with the default ensure_ascii=True.  Validate it with its own published
    # rule; Stage-A self digests use canonical_digest() above.
    source_manifest_digest = hashlib.sha256(json.dumps(
        {name: value for name, value in manifest.items()
         if name != "state_manifest_digest"}, sort_keys=True,
    ).encode()).hexdigest()
    if manifest.get("state_manifest_digest") != source_manifest_digest:
        raise RuntimeError("source state manifest self digest mismatch")
    if (manifest.get("state_manifest_digest") != SOURCE_STATE_MANIFEST_DIGEST
            or manifest.get("candidate_bank_digest") != V1.bank_digest()
            or manifest.get("progress_contract_digest") != progress_digest()
            or manifest.get("safety_contract_digest") != safety_digest()
            or manifest.get("oracle_v1_2_digest") != oracle_v1_2_digest()
            or manifest.get("genesis_backend") != "cpu"
            or len(manifest.get("states", [])) != EXPECTED_STATES):
        raise RuntimeError("source oracle-v1.2 manifest is not the frozen successful one")

    gate = json.loads((ROOT / SOURCE_FILES["gate_report"]["path"]).read_text())
    if (gate.get("state_manifest_digest") != SOURCE_STATE_MANIFEST_DIGEST
            or gate.get("gate", {}).get("pass") is not True
            or gate.get("statistics", {}).get("attempted") != EXPECTED_BRANCHES
            or gate.get("statistics", {}).get("valid") != EXPECTED_BRANCHES):
        raise RuntimeError("source oracle-v1.2 gate witness is not the successful gate")

    raw_lines = (ROOT / SOURCE_FILES["pilot_branches"]["path"]).read_bytes().splitlines(
        keepends=True)
    if len(raw_lines) != EXPECTED_BRANCHES:
        raise RuntimeError("source pilot branch ledger record count changed")
    state_order = {state["state_id"]: index
                   for index, state in enumerate(manifest["states"])}
    candidate_order = {candidate[0]: index
                       for index, candidate in enumerate(V1.CANDIDATE_BANK)}
    outcomes: dict[tuple[str, int], AcceptedOutcome] = {}
    for line_index, raw_line in enumerate(raw_lines):
        row = json.loads(raw_line)
        state_id = str(row.get("state_id"))
        candidate = str(row.get("candidate"))
        if state_id not in state_order or candidate not in candidate_order:
            raise RuntimeError("source pilot ledger contains an unregistered identity")
        candidate_index = candidate_order[candidate]
        expected_position = state_order[state_id] * EXPECTED_CANDIDATES + candidate_index
        if line_index != expected_position:
            raise RuntimeError("source pilot ledger order changed")
        key = (state_id, candidate_index)
        if key in outcomes:
            raise RuntimeError("source pilot ledger duplicates a branch")
        if (row.get("valid") is not True
                or row.get("state_manifest_digest") != SOURCE_STATE_MANIFEST_DIGEST
                or row.get("candidate_bank_digest") != V1.bank_digest()
                or row.get("oracle_v1_2_digest") != oracle_v1_2_digest()
                or row.get("primitives") != list(V1.CANDIDATE_BANK[candidate_index][1])):
            raise RuntimeError("source pilot outcome is invalid or relabelled")
        outcomes[key] = AcceptedOutcome(
            row=row,
            line_index=line_index,
            line_sha256=hashlib.sha256(raw_line).hexdigest(),
            line_byte_count=len(raw_line),
        )
    if len(outcomes) != EXPECTED_BRANCHES:
        raise RuntimeError("source pilot outcome index is incomplete")
    witnesses = copy.deepcopy(SOURCE_FILES)
    witnesses["state_manifest"]["identity_manifest_digest"] = (
        SOURCE_STATE_MANIFEST_DIGEST)
    witnesses["pilot_branches"]["line_digest_rule"] = (
        "SHA256 of each exact JSONL record including its terminating newline")
    witnesses["gate_report"]["gate_pass"] = True
    return SourceEvidence(manifest, gate, outcomes, witnesses)


def source_bindings() -> dict[str, Any]:
    """Exact implementation identities bound before Stage-A execution."""

    paths = (
        "scripts/build_go2_counterfactual_fidelity_stage_a_v1_2.py",
        "scripts/encode_go2_counterfactual_fidelity_stage_a_v1_2.py",
        "scripts/run_go2_oracle_branch_pilot_v1.py",
        "scripts/run_go2_oracle_branch_pilot_v1_2.py",
        "lewm/oracle/go2_branch_oracle_v1_2.py",
        "lewm/oracle/go2_scorer_contract_v1_2.py",
        "scripts/render_replay_v03.py",
        "lewm/oracle/go2_textured_v03_renderer.py",
        "scripts/dev_action_slew_reconstruction_v1.py",
        "scripts/dev_frozen_dense_representation_encoders_v1.py",
        "scripts/build_dev_v03_proprio_action_manifest_v1.py",
        "scripts/analyze_go2_closed_loop_quality.py",
        "lewm_genesis/lewm_genesis/rollout.py",
        "lewm_genesis/lewm_genesis/scene_loader.py",
        "lewm_genesis/lewm_genesis/scene_builder.py",
        "lewm_genesis/lewm_genesis/render_replay.py",
        "lewm_worlds/lewm_worlds/labels/derived.py",
        "lewm_worlds/lewm_worlds/planning_grid.py",
        "config/go2_platform_manifest.yaml",
        "config/go2_primitive_registry.yaml",
    )
    return {path: _file_binding(ROOT / path) for path in paths}


def assay_spec() -> dict[str, Any]:
    return {
        "schema": "go2_counterfactual_fidelity_stage_a_assay_contract_v1_2",
        "source": {
            "identity_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
            "state_count": EXPECTED_STATES,
            "candidates_per_state": EXPECTED_CANDIDATES,
            "branch_count": EXPECTED_BRANCHES,
            "reuse": "the exact successful oracle-v1.2 pilot identities only",
        },
        "execution": {
            "backend": "cpu",
            "single_physical_redrive_per_invocation_state": True,
            "single_branch_execution_for_oracle_and_horizon_poses": True,
            "candidate_order": "frozen candidate-bank indices 0..11",
            "smoke": "state index 0, candidate indices 0..5; retained by full run",
            "resume": "validated completed branch identities are never regenerated",
        },
        "context": {
            "rgb_slots": CONTEXT_SLOTS,
            "boundary_offsets_blocks": [-2, -1, 0],
            "action_context_shape": [CONTEXT_SLOTS, SLEW.ACTION_DIM],
            "proprio_shape": [PROPRIO_HISTORY, 30],
            "control_shape": [PROPRIO_HISTORY, 2],
            "previous_applied_command_shape": [3],
            "proprio_channels": "frozen deployment-valid V03 30-D order",
            "control_channels": list(SLEW.ACTIVE_CHANNEL_NAMES),
        },
        "candidate": {
            "requested_shape": [HORIZONS, 5, 3],
            "post_slew_shape": [HORIZONS, 5, 3],
            "action_shape": [HORIZONS, SLEW.ACTION_DIM],
            "action_order": "block-major, tick-major, vx then yaw-rate",
        },
        "targets": {
            "horizons_blocks": [1, 2, 3, 4],
            "horizon_seconds": [0.5, 1.0, 1.5, 2.0],
            "base_pose": "world XYZ plus quaternion WXYZ at each block end",
            "rgb_shape_hwc": [224, 224, 3],
            "latent_shape": [HORIZONS, 768, 1024],
            "latent_dtype": "float16",
            "token_order": TARGET_ENCODER["token_order"],
            "latent_storage": (
                "raw final-block encoder tokens rounded to float16; consumers "
                "reload as float32 then apply F.layer_norm over the 1024-D token axis"
            ),
        },
        "outcome_equality": {
            "fields": list(OUTCOME_FIELDS),
            "excluded_source_field": "wall_time_s only",
            "rule": "canonical projections must be byte-for-byte digest equal",
        },
        "receipt_schemas": {
            "identity_manifest": (
                "go2_counterfactual_fidelity_stage_a_identity_manifest_v1_2"),
            "state_record": (
                "go2_counterfactual_fidelity_stage_a_state_record_v1_2"),
            "branch_row": (
                "go2_counterfactual_fidelity_stage_a_branch_row_v1_2"),
            "completion_receipt": (
                "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2"),
            "latents_index": (
                "go2_counterfactual_fidelity_stage_a_latents_index_v1_2"),
        },
        "render": {
            "render_contract_digest": render_contract_digest(),
            "renderer_contract_digest": renderer_contract_digest(),
            "contract": RENDER_CONTRACT,
        },
        "preprocess": {
            "preprocess_contract_digest": preprocess_contract_digest(),
            "preprocessing_digest": PREPROCESSING_DIGEST,
        },
        "target_encoder": {
            "target_encoder_digest": target_encoder_digest(),
            "checkpoint_sha256": TARGET_ENCODER["checkpoint_sha256"],
            "checkpoint_byte_count": TARGET_ENCODER["checkpoint_byte_count"],
            "tokens": TARGET_ENCODER["tokens"],
            "token_dim": TARGET_ENCODER["token_dim"],
            "token_grid": TARGET_ENCODER["token_grid"],
        },
    }


def assay_spec_digest() -> str:
    return canonical_digest(assay_spec())


def _scene_binding(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(entry["scene_dir"])) / "manifest.json"
    raw = json.loads(path.read_text())
    if str(raw.get("scene_id")) != str(entry["scene_id"]):
        raise RuntimeError(f"scene manifest identity mismatch for {entry['state_id']}")
    return {
        "scene_manifest_path": str(path),
        "scene_manifest_sha256": file_sha256(path),
        "scene_manifest_byte_count": path.stat().st_size,
        "scene_manifest_canonical_digest": canonical_digest(raw),
        "raw_manifest_digest": raw_manifest_digest(raw),
    }


def _state_identity_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in state.items()
            if name not in {"state_identity_digest", "branch_identities"}}


def _branch_identity(state: Mapping[str, Any], candidate_index: int) -> dict[str, Any]:
    candidate = V1.CANDIDATE_BANK[candidate_index]
    payload = {
        "schema": "go2_counterfactual_fidelity_stage_a_branch_identity_v1_2",
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "episode_cluster_id": state["episode_cluster_id"],
        "source_step": state["source_step"],
        "goal_designation": state["goal_designation"],
        "candidate_index": int(candidate_index),
        "candidate": candidate[0],
        "primitives": list(candidate[1]),
        "source_state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
        "candidate_bank_digest": V1.bank_digest(),
        "oracle_v1_2_digest": oracle_v1_2_digest(),
        "assay_spec_digest": assay_spec_digest(),
    }
    return {**payload, "branch_identity_digest": canonical_digest(payload)}


def build_identity_manifest(source: SourceEvidence | None = None) -> dict[str, Any]:
    """Issue all Stage-A identities without executing a candidate branch."""

    source = load_source_evidence() if source is None else source
    bindings = source_bindings()
    states: list[dict[str, Any]] = []
    for state_index, original in enumerate(source.manifest["states"]):
        state = {
            "schema": "go2_counterfactual_fidelity_stage_a_state_identity_v1_2",
            "state_index": int(state_index),
            "state_id": original["state_id"],
            "family": original["family"],
            "split": original["split"],
            "scene_id": original["scene_id"],
            "scene_dir": original["scene_dir"],
            "episode_cluster_id": (
                f"{original['split']}:{original['scene_id']}:seed-{original['drive_seed']}"),
            "drive_seed": int(original["drive_seed"]),
            "warmup_blocks": int(original["warmup_blocks"]),
            "source_step": int(original["source_step"]),
            "cell_id": int(original["cell_id"]),
            "goal_designation": {
                "landmark_id": original["landmark_id"],
                "landmark_cell": int(original["landmark_cell"]),
                "graph_edges_to_landmark": int(original["graph_edges_to_landmark"]),
                "start_geodesic_m": float(original["start_geodesic_m"]),
            },
            "clearance_m": float(original["clearance_m"]),
            **_scene_binding(original),
            "source_state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
            "assay_spec_digest": assay_spec_digest(),
        }
        state["state_identity_digest"] = canonical_digest(_state_identity_payload(state))
        state["branch_identities"] = [
            _branch_identity(state, candidate_index)
            for candidate_index in range(EXPECTED_CANDIDATES)
        ]
        states.append(state)
    manifest = {
        "schema": "go2_counterfactual_fidelity_stage_a_identity_manifest_v1_2",
        "status": STATUS,
        "complete": True,
        "output_root": str(OUT_ROOT),
        "source_witnesses": source.witnesses,
        "source_state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
        "source_pilot_branch_ledger_sha256": SOURCE_FILES["pilot_branches"]["sha256"],
        "source_gate_report_sha256": SOURCE_FILES["gate_report"]["sha256"],
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": oracle_v1_2_digest(),
        "boundary_digest": V1.BOUNDARY_DIGEST,
        "genesis_backend": "cpu",
        "render_contract_digest": render_contract_digest(),
        "textured_v03_renderer_contract_digest": renderer_contract_digest(),
        "preprocess_contract_digest": preprocess_contract_digest(),
        "preprocessing_digest": PREPROCESSING_DIGEST,
        "target_encoder_digest": target_encoder_digest(),
        "target_encoder_checkpoint_sha256": TARGET_ENCODER["checkpoint_sha256"],
        "target_encoder_checkpoint_byte_count": TARGET_ENCODER["checkpoint_byte_count"],
        "target_token_layout": {
            "grid_hw": TARGET_ENCODER["token_grid"],
            "tokens": TARGET_ENCODER["tokens"],
            "token_dim": TARGET_ENCODER["token_dim"],
            "order": TARGET_ENCODER["token_order"],
        },
        "assay_spec": assay_spec(),
        "assay_spec_digest": assay_spec_digest(),
        "source_bindings": bindings,
        "state_count_registered": EXPECTED_STATES,
        "candidate_count_per_state_registered": EXPECTED_CANDIDATES,
        "attempted_branch_count_registered": EXPECTED_BRANCHES,
        "states": states,
        "branch_identity_set_digest": canonical_digest(sorted(
            identity["branch_identity_digest"]
            for state in states for identity in state["branch_identities"])),
    }
    manifest["stage_a_identity_manifest_digest"] = canonical_digest(manifest)
    return manifest


def validate_identity_manifest(manifest: Mapping[str, Any],
                               source: SourceEvidence | None = None) -> None:
    source = load_source_evidence() if source is None else source
    _verify_self_digest(manifest, "stage_a_identity_manifest_digest",
                        "Stage-A identity manifest")
    expected = build_identity_manifest(source)
    if manifest != expected:
        raise RuntimeError("Stage-A identity manifest differs from frozen construction")


def _base_pose_payload(pose: BasePose) -> dict[str, Any]:
    return {
        "position_world_xyz": [float(value) for value in pose.position_world_xyz],
        "quaternion_world_wxyz": [
            float(value) for value in pose.quaternion_world_wxyz],
        "quaternion_order": "wxyz",
    }


def _frame_camera_contract() -> dict[str, Any]:
    return {
        "fov_axis": "horizontal",
        "fov_deg": 78.323,
        "near_m": 0.05,
        "far_m": 200.0,
        "resolution_wh": [224, 224],
    }


def _resolve_output_path(out: Path, relative: str) -> Path:
    path = (out / relative).resolve()
    if out.resolve() not in path.parents:
        raise RuntimeError(f"Stage-A artifact escapes output root: {relative}")
    return path


def write_png_atomic(image: np.ndarray, path: Path, out: Path) -> tuple[str, int]:
    from PIL import Image
    array = np.asarray(image)
    if array.shape != (224, 224, 3) or array.dtype != np.uint8:
        raise RuntimeError(f"invalid corrected V03 frame {array.shape}/{array.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    Image.fromarray(array).save(temporary, format="PNG")
    digest, byte_count = file_sha256(temporary), temporary.stat().st_size
    if path.exists():
        if (path.is_file() and path.stat().st_size == byte_count
                and file_sha256(path) == digest):
            temporary.unlink()
            return digest, byte_count
        _preserve_invalid(path, out, "frame-mismatch")
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return digest, byte_count


def _frame_receipt(result: Any, pose: BasePose, path: Path, out: Path, *,
                   index_key: str, index_value: int,
                   renderer: TexturedV03Renderer) -> dict[str, Any]:
    digest, byte_count = write_png_atomic(result.image, path, out)
    return {
        index_key: int(index_value),
        "path": str(path.relative_to(out)),
        "sha256": digest,
        "byte_count": byte_count,
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "pixel_sha256": hashlib.sha256(
            np.ascontiguousarray(result.image).tobytes()).hexdigest(),
        "base_pose": _base_pose_payload(pose),
        "camera_pose_world": result.camera_pose_world,
        "camera": _frame_camera_contract(),
        "raw_manifest_digest": renderer.raw_manifest_digest,
        "textured_v03_renderer_contract_digest": renderer.contract_digest,
        "render_runtime_s": round(float(result.runtime_s), 6),
    }


def _validate_pose(value: Mapping[str, Any]) -> None:
    xyz = np.asarray(value.get("position_world_xyz"), dtype=np.float64)
    quat = np.asarray(value.get("quaternion_world_wxyz"), dtype=np.float64)
    if (xyz.shape != (3,) or quat.shape != (4,) or value.get("quaternion_order") != "wxyz"
            or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat))):
        raise RuntimeError("malformed immutable base pose")


def _validate_frame(out: Path, frame: Mapping[str, Any], *, index_key: str,
                    index_value: int, raw_digest: str) -> None:
    path = _resolve_output_path(out, str(frame["path"]))
    if (int(frame.get(index_key, -1)) != index_value
            or frame.get("shape") != [224, 224, 3]
            or frame.get("dtype") != "uint8"
            or frame.get("camera") != _frame_camera_contract()
            or frame.get("raw_manifest_digest") != raw_digest
            or frame.get("textured_v03_renderer_contract_digest")
            != renderer_contract_digest()
            or not path.is_file()
            or path.stat().st_size != int(frame["byte_count"])
            or file_sha256(path) != frame["sha256"]):
        raise RuntimeError(f"invalid corrected V03 frame receipt {path}")
    from PIL import Image
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() \
            != frame.get("pixel_sha256"):
        raise RuntimeError("stored PNG does not reproduce the bound raw pixels")
    _validate_pose(frame["base_pose"])
    camera_pose = frame.get("camera_pose_world", {})
    for key in ("position", "lookat", "up"):
        values = np.asarray(camera_pose.get(key), dtype=np.float64)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise RuntimeError("malformed exact camera pose")


def proprio_sample(ctx: V1.BranchContext) -> list[float]:
    from lewm_genesis.rollout import _pitch_from_quat_wxyz, _roll_from_quat_wxyz
    runner, robot = ctx.runner, ctx.build.robot
    quat = np.asarray(runner._as_np(robot.get_quat()), dtype=np.float64)
    angular = np.asarray(runner._as_np(robot.get_ang()), dtype=np.float64)
    if quat.ndim == 1:
        quat = quat[None, :]
    if angular.ndim == 1:
        angular = angular[None, :]
    qw, qx, qy, qz = (float(value) for value in quat[0])
    gyro = runner._world_to_body(angular[0], np.asarray([qx, qy, qz, qw]))
    gravity = M.projected_gravity(_roll_from_quat_wxyz(qw, qx, qy, qz),
                                  _pitch_from_quat_wxyz(qw, qx, qy, qz))
    joint_pos = np.asarray(runner._as_np(
        robot.get_dofs_position(runner._leg_dof_idx.tolist())), dtype=np.float64)
    joint_vel = np.asarray(runner._as_np(
        robot.get_dofs_velocity(runner._leg_dof_idx.tolist())), dtype=np.float64)
    if joint_pos.ndim == 2:
        joint_pos, joint_vel = joint_pos[0], joint_vel[0]
    return ([float(g - offset) for g, offset in zip(gravity, M.GRAVITY_OFFSET)]
            + [float(value) for value in gyro]
            + [float(value) for value in joint_pos]
            + [float(value) for value in joint_vel])


def control_sample(previous_applied: Sequence[float]) -> list[float]:
    return [float(previous_applied[channel]) for channel in SLEW.ACTIVE_CHANNELS]


def action_block_10d(block: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for value in SLEW.flatten(block)]


def drive_block_with_probe(ctx: V1.BranchContext,
                           probe: Callable[[int, Sequence[float]], None]) -> Any:
    """Read-only probes added to the otherwise identical frozen drive block."""

    runner = ctx.runner
    requested, _choices = runner._collect_block()
    planned = np.asarray(
        runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
        dtype=np.float64)
    previous = np.asarray(runner._last_executed, dtype=np.float64)[0].copy()
    carry = {"previous": previous}
    steps_per_tick = int(runner._policy_steps_per_command_tick)

    def after_policy_step(tick_index: int, step_index: int) -> None:
        if step_index != steps_per_tick - 1:
            return
        probe(int(tick_index), carry["previous"])
        carry["previous"] = planned[0, tick_index].copy()

    block = runner.execute_requested_block(requested,
                                           after_policy_step=after_policy_step)
    if not np.array_equal(np.asarray(block.executed, dtype=np.float32),
                          np.asarray(planned, dtype=np.float32)):
        raise RuntimeError("read-only drive probe changed post-slew actions")
    for _ in range(runner._block_size):
        for episode_state in runner.episode_states:
            episode_state.step()
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


def candidate_planning_trajectory(candidate: tuple[str, tuple[str, ...]],
                                  previous_applied: Sequence[float]) -> tuple[
                                      list[Any], list[Any], list[Any]]:
    previous = tuple(float(value) for value in previous_applied)
    requested, post_slew, actions = [], [], []
    for primitive in candidate[1]:
        request = np.asarray(V1.block_for(primitive), dtype=np.float64).tolist()
        realised, previous = SLEW.reconstruct_block(primitive, previous)
        requested.append(request)
        post_slew.append([[float(value) for value in tick] for tick in realised])
        actions.append(action_block_10d(realised))
    return requested, post_slew, actions


def _goal_binding(ctx: V1.BranchContext, goal: Mapping[str, Any]) -> tuple[float, float]:
    from lewm_worlds.scene_graph import wrap_angle_pi
    (x, y), yaw, _z = ctx.pose()
    gx, gy = (float(value) for value in goal["landmark_xy_m"])
    dx, dy = gx - x, gy - y
    return float(wrap_angle_pi(math.atan2(dy, dx) - yaw)), float(math.hypot(dx, dy))


def execute_branch_with_pose_capture(ctx: V1.BranchContext, snapshot: Any,
                                     candidate: tuple[str, tuple[str, ...]], *,
                                     field: GeodesicField,
                                     topology: Mapping[str, Any]) -> tuple[dict[str, Any],
                                                                          list[BasePose]]:
    """V12 execution plus a read-only base-pose capture after every block."""

    from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep
    V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    goal_cell = int(snapshot.goal["landmark_cell"])
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    state = {"episode_step": int(runner.episode_states[0].episode_step),
             "stamp_ns": int(runner._sim_time_ns)}

    def sample(executed: Sequence[float]) -> dict[str, Any]:
        (x, y), yaw, z = ctx.pose()
        label = label_computer.step(PoseStep(
            timestamp_ns=state["stamp_ns"], env_idx=0, episode_id=episode_id,
            episode_step=state["episode_step"], position_xy_world=(x, y),
            yaw_world_rad=yaw,
            last_command=(float(executed[0]), float(executed[1]), float(executed[2]))))
        flags = V1._termination_flags(ctx)
        hit = ctx.scene_graph.locate((x, y))
        located = bool(float(hit.distance_m) <= V1.LOCATE_MAX_DISTANCE_M)
        cell = int(hit.cell_id)
        return {
            "xy": [x, y], "yaw": yaw, "z": z, "cell_id": cell,
            "located": located,
            "geodesic_m": float(field.remaining_distance((x, y), cell)
                                if located else math.inf),
            "at_goal_cell": bool(cell == goal_cell),
            "clearance_m": float(label.clearance_m),
            "stuck": bool(label.stuck_label),
            "disallowed_contacts": int(V12._contact_count(ctx, topology)),
            "terminated": bool(flags["fall"] or flags["out_of_bounds"]
                               or flags["tipped"]),
            "nan": bool(flags["nan"]),
        }

    start = sample(np.asarray(runner._last_executed, dtype=np.float64)[0])
    ticks: list[dict[str, Any]] = []
    requested_all: list[Any] = []
    executed_all: list[Any] = []
    poses: list[BasePose] = []
    clipped = False
    truncated_at_block = None
    nan_seen = False
    for block_index, primitive in enumerate(candidate[1]):
        requested = V1.block_for(primitive)[None, ...]
        planned = np.asarray(
            runner._clip_block(np.asarray(requested, dtype=np.float32)).executed,
            dtype=np.float64)

        def after_policy_step(tick_index: int, step_index: int,
                              _planned=planned, _block=block_index) -> None:
            if step_index != steps_per_tick - 1:
                return
            state["episode_step"] += 1
            state["stamp_ns"] += int(runner._command_dt_ns)
            row = sample(_planned[0, tick_index])
            row["block"] = _block
            row["tick"] = int(tick_index)
            ticks.append(row)

        block = runner.execute_requested_block(requested,
                                               after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested)[0].tolist())
        executed_all.append(np.asarray(block.executed)[0].tolist())
        clipped = clipped or bool(np.asarray(block.clipped)[0])
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
        poses.append(capture_base_pose(ctx))
        if ticks and ticks[-1]["nan"]:
            nan_seen, truncated_at_block = True, block_index
            break
        if ticks and ticks[-1]["terminated"]:
            truncated_at_block = block_index
            break
    return ({
        "candidate": candidate[0], "primitives": list(candidate[1]),
        "requested": requested_all, "post_slew": executed_all,
        "clipped": clipped, "blocks_completed": len(executed_all),
        "truncated_at_block": truncated_at_block, "nan": nan_seen,
        "start": start, "ticks": ticks,
    }, poses)


def _identity_for(state: Mapping[str, Any], candidate_index: int) -> dict[str, Any]:
    identities = [identity for identity in state["branch_identities"]
                  if int(identity["candidate_index"]) == int(candidate_index)]
    if len(identities) != 1:
        raise RuntimeError("branch identity lookup is ambiguous")
    return identities[0]


def _state_record_path(out: Path, state: Mapping[str, Any]) -> Path:
    return out / "state_records" / f"{state['state_identity_digest']}.json"


def _row_path(out: Path, identity: Mapping[str, Any]) -> Path:
    return out / "row_records" / f"{identity['branch_identity_digest']}.json"


def _manifest_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stage_a_identity_manifest_digest", "assay_spec_digest",
        "source_state_manifest_digest", "source_pilot_branch_ledger_sha256",
        "source_gate_report_sha256", "candidate_bank_digest",
        "progress_contract_digest", "safety_contract_digest", "oracle_v1_2_digest",
        "boundary_digest", "render_contract_digest",
        "textured_v03_renderer_contract_digest", "preprocess_contract_digest",
        "preprocessing_digest", "target_encoder_digest",
        "target_encoder_checkpoint_sha256",
    )
    return {key: manifest[key] for key in keys}


def _redrive_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "state_id", "state_identity_digest", "scene_id", "episode_cluster_id",
        "episode_id", "source_step", "warmup_blocks", "boundary", "goal",
        "goal_binding_input", "previous_applied_command", "action_context_blocks",
        "proprio", "control", "context_base_poses", "snapshot_digest",
    )
    return {key: payload[key] for key in keys}


def _validate_state_record(record: Mapping[str, Any], state: Mapping[str, Any],
                           manifest: Mapping[str, Any], out: Path) -> None:
    _verify_self_digest(record, "state_record_digest", "Stage-A state record")
    if (record.get("schema")
            != "go2_counterfactual_fidelity_stage_a_state_record_v1_2"
            or record.get("record_complete") is not True
            or record.get("state_id") != state["state_id"]
            or record.get("state_identity_digest") != state["state_identity_digest"]
            or record.get("scene_manifest_sha256") != state["scene_manifest_sha256"]
            or record.get("raw_manifest_digest") != state["raw_manifest_digest"]
            or record.get("redrive_projection_digest")
            != canonical_digest(_redrive_projection(record))):
        raise RuntimeError("Stage-A state record identity or redrive digest mismatch")
    for key, expected in _manifest_bindings(manifest).items():
        if record.get(key) != expected:
            raise RuntimeError(f"Stage-A state record {key} mismatch")
    proprio = np.asarray(record.get("proprio"), dtype=np.float64)
    control = np.asarray(record.get("control"), dtype=np.float64)
    actions = np.asarray(record.get("action_context_blocks"), dtype=np.float64)
    if (proprio.shape != (PROPRIO_HISTORY, 30)
            or control.shape != (PROPRIO_HISTORY, 2)
            or actions.shape != (CONTEXT_SLOTS, SLEW.ACTION_DIM)
            or not np.all(np.isfinite(proprio)) or not np.all(np.isfinite(control))
            or not np.all(np.isfinite(actions))):
        raise RuntimeError("Stage-A planning-time histories are malformed")
    poses = record.get("context_base_poses", [])
    frames = record.get("context_frames", [])
    if len(poses) != CONTEXT_SLOTS or len(frames) != CONTEXT_SLOTS:
        raise RuntimeError("Stage-A state record lacks three context poses/frames")
    for index, (pose, frame) in enumerate(zip(poses, frames)):
        _validate_pose(pose)
        if frame.get("base_pose") != pose:
            raise RuntimeError("context frame/base pose binding mismatch")
        _validate_frame(out, frame, index_key="slot", index_value=index,
                        raw_digest=state["raw_manifest_digest"])


def _accepted_witness(outcome: AcceptedOutcome) -> dict[str, Any]:
    return {
        "source_path": SOURCE_FILES["pilot_branches"]["path"],
        "source_ledger_sha256": SOURCE_FILES["pilot_branches"]["sha256"],
        "source_line_index": outcome.line_index,
        "source_line_sha256": outcome.line_sha256,
        "source_line_byte_count": outcome.line_byte_count,
        "outcome_projection_digest": canonical_digest(_outcome_projection(outcome.row)),
    }


def _validate_branch_row(row: Mapping[str, Any], state: Mapping[str, Any],
                         identity: Mapping[str, Any], manifest: Mapping[str, Any],
                         source: SourceEvidence, out: Path) -> None:
    _verify_self_digest(row, "branch_row_digest", "Stage-A branch row")
    outcome = source.outcomes[(state["state_id"], int(identity["candidate_index"]))]
    if (row.get("schema") != "go2_counterfactual_fidelity_stage_a_branch_row_v1_2"
            or row.get("record_complete") is not True
            or row.get("state_id") != state["state_id"]
            or row.get("state_identity_digest") != state["state_identity_digest"]
            or row.get("branch_identity_digest") != identity["branch_identity_digest"]
            or row.get("candidate_index") != int(identity["candidate_index"])
            or row.get("candidate") != identity["candidate"]
            or row.get("context_key") != state["state_id"]
            or row.get("valid") is not True
            or row.get("oracle_outcome_equal") is not True
            or row.get("accepted_outcome_witness") != _accepted_witness(outcome)
            or row.get("accepted_oracle_outcome") != outcome.row
            or row.get("reexecuted_outcome_projection_digest")
            != canonical_digest(_outcome_projection(outcome.row))):
        raise RuntimeError("Stage-A branch row identity/outcome witness mismatch")
    for key, expected in _manifest_bindings(manifest).items():
        if row.get(key) != expected:
            raise RuntimeError(f"Stage-A branch row {key} mismatch")
    state_record = json.loads(_state_record_path(out, state).read_text())
    _validate_state_record(state_record, state, manifest, out)
    if row.get("state_record_digest") != state_record["state_record_digest"]:
        raise RuntimeError("Stage-A branch row binds another context record")
    for key in (
        "scene_dir", "scene_manifest_sha256", "scene_manifest_byte_count",
        "scene_manifest_canonical_digest", "raw_manifest_digest", "episode_cluster_id",
        "goal", "goal_binding_input", "previous_applied_command",
        "action_context_blocks", "proprio", "control", "masks", "timing",
        "context_base_poses", "context_frames", "snapshot_digest",
    ):
        if row.get(key) != state_record.get(key):
            raise RuntimeError(f"Stage-A row/context projection mismatch: {key}")
    previous = np.asarray(row["previous_applied_command"], dtype=np.float64)
    candidate = V1.CANDIDATE_BANK[int(identity["candidate_index"])]
    requested, post_slew, action_blocks = candidate_planning_trajectory(
        candidate, previous.tolist())
    if (row.get("requested") != requested
            or not np.allclose(np.asarray(row.get("candidate_post_slew_plan")),
                               np.asarray(post_slew), rtol=0.0, atol=1e-12)
            or not np.allclose(np.asarray(row.get("action_blocks")),
                               np.asarray(action_blocks), rtol=0.0, atol=1e-12)):
        raise RuntimeError("Stage-A candidate planning representation changed")
    horizon_poses = row.get("horizon_base_poses", [])
    horizon_frames = row.get("horizon_frames", [])
    if len(horizon_poses) != HORIZONS or len(horizon_frames) != HORIZONS:
        raise RuntimeError("Stage-A row lacks exact H=1..4 pose/render evidence")
    for index, (pose, frame) in enumerate(zip(horizon_poses, horizon_frames), start=1):
        _validate_pose(pose)
        if frame.get("base_pose") != pose:
            raise RuntimeError("horizon frame/base pose binding mismatch")
        _validate_frame(out, frame, index_key="horizon", index_value=index,
                        raw_digest=state["raw_manifest_digest"])
    if any(row.get(field) != outcome.row.get(field) for field in ORACLE_LABEL_FIELDS):
        raise RuntimeError("Stage-A row oracle labels differ from accepted pilot")


def _recover_row_records_from_ledger(manifest: Mapping[str, Any],
                                     source: SourceEvidence, out: Path) -> int:
    """Restore missing/corrupt row shards from a fully validated durable ledger."""

    ledger = out / "branch_rows.jsonl"
    receipt_path = out / "corpus_receipt.json"
    if not ledger.is_file() or not receipt_path.is_file():
        return 0
    try:
        rows = [json.loads(line) for line in ledger.read_text().splitlines()
                if line.strip()]
        canonical_text = "".join(
            json.dumps(V1._jsonable(row), sort_keys=True) + "\n" for row in rows)
        if ledger.read_text() != canonical_text:
            raise RuntimeError("Stage-A corpus ledger is not canonical JSONL")
        receipt = json.loads(receipt_path.read_text())
        _validate_corpus_receipt(receipt, manifest, out, rows)
        registered = {
            identity["branch_identity_digest"]: (state, identity)
            for state in manifest["states"] for identity in state["branch_identities"]
        }
        validated: list[tuple[dict[str, Any], Mapping[str, Any],
                              Mapping[str, Any]]] = []
        for row in rows:
            digest = str(row.get("branch_identity_digest"))
            if digest not in registered:
                raise RuntimeError("Stage-A ledger invents a branch identity")
            state, identity = registered[digest]
            if (row.get("state_id") != state["state_id"]
                    or int(row.get("candidate_index", -1))
                    != int(identity["candidate_index"])):
                raise RuntimeError("Stage-A ledger branch registration changed")
            _validate_branch_row(row, state, identity, manifest, source, out)
            validated.append((row, state, identity))
    except Exception as exc:
        preserved: list[str] = []
        for path in (receipt_path, ledger):
            if path.exists():
                preserved.append(str(_preserve_invalid(
                    path, out, "ledger-row-recovery-validation-failed")))
        print(f"[recovery] preserved invalid ledger/receipt {preserved}: {exc}",
              flush=True)
        return 0

    recovered = 0
    for row, state, identity in validated:
        path = _row_path(out, identity)
        if path.is_file():
            try:
                stored = json.loads(path.read_text())
                _validate_branch_row(stored, state, identity, manifest, source, out)
                if stored != row:
                    raise RuntimeError("row shard differs from validated ledger row")
                continue
            except Exception as exc:
                preserved = _preserve_invalid(
                    path, out, "row-shard-ledger-recovery-validation-failed")
                print(f"[recovery] preserved {preserved}: {exc}", flush=True)
        atomic_json(path, row)
        recovered += 1
    if recovered:
        print(f"[recovery] restored {recovered} exact row shard(s) from ledger",
              flush=True)
    return recovered


def _completed_rows(manifest: Mapping[str, Any], source: SourceEvidence,
                    out: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for state in manifest["states"]:
        for candidate_index in range(EXPECTED_CANDIDATES):
            identity = _identity_for(state, candidate_index)
            path = _row_path(out, identity)
            if not path.exists():
                continue
            try:
                row = json.loads(path.read_text())
                _validate_branch_row(row, state, identity, manifest, source, out)
            except Exception as exc:
                preserved = _preserve_invalid(path, out, "row-validation-failed")
                print(f"[recovery] preserved {preserved}: {exc}", flush=True)
                continue
            completed[(state["state_id"], candidate_index)] = row
    return completed


def _validate_corpus_receipt(receipt: Mapping[str, Any],
                             manifest: Mapping[str, Any], out: Path,
                             expected_rows: Sequence[Mapping[str, Any]] | None = None
                             ) -> None:
    """Validate a corpus receipt, its ledger, and their immutable bindings."""

    _verify_self_digest(receipt, "completion_receipt_digest",
                        "Stage-A completion receipt")
    ledger = out / "branch_rows.jsonl"
    if not ledger.is_file():
        raise RuntimeError("Stage-A corpus ledger is missing")
    payload = receipt.get("corpus_digest_payload")
    if (not isinstance(payload, Mapping)
            or canonical_digest(payload) != receipt.get("corpus_digest")):
        raise RuntimeError("Stage-A corpus identity payload mismatch")
    bindings = _manifest_bindings(manifest)
    branch_digests = receipt.get("branch_row_digests")
    attempted = int(receipt.get("attempted_branch_count", -1))
    completed_states = int(receipt.get("completed_state_count", -1))
    complete = bool(attempted == EXPECTED_BRANCHES
                    and completed_states == EXPECTED_STATES)
    checks = (
        receipt.get("schema")
        == "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2",
        receipt.get("status") == STATUS,
        receipt.get("stage_a_identity_manifest_digest")
        == manifest["stage_a_identity_manifest_digest"],
        receipt.get("assay_spec_digest") == manifest["assay_spec_digest"],
        all(receipt.get(key) == value for key, value in bindings.items()),
        receipt.get("branch_rows_sha256") == file_sha256(ledger),
        isinstance(branch_digests, list) and len(branch_digests) == attempted,
        int(receipt.get("state_count", -1)) == EXPECTED_STATES,
        int(receipt.get("expected_branch_count", -1)) == EXPECTED_BRANCHES,
        int(receipt.get("valid_branch_count", -1)) == attempted,
        int(receipt.get("oracle_equal_branch_count", -1)) == attempted,
        int(receipt.get("invalid_branch_count", -1)) == 0,
        receipt.get("complete") is complete,
        payload.get("schema")
        == "go2_counterfactual_fidelity_stage_a_corpus_identity_v1_2",
        payload.get("stage_a_identity_manifest_digest")
        == manifest["stage_a_identity_manifest_digest"],
        payload.get("assay_spec_digest") == manifest["assay_spec_digest"],
        payload.get("source_state_manifest_digest") == SOURCE_STATE_MANIFEST_DIGEST,
        payload.get("source_pilot_branch_ledger_sha256")
        == SOURCE_FILES["pilot_branches"]["sha256"],
        payload.get("branch_identity_set_digest")
        == manifest["branch_identity_set_digest"],
        payload.get("branch_rows_sha256") == receipt.get("branch_rows_sha256"),
        payload.get("branch_row_digests") == branch_digests,
        int(payload.get("state_count", -1)) == EXPECTED_STATES,
        int(payload.get("completed_state_count", -1)) == completed_states,
        int(payload.get("attempted_branch_count", -1)) == attempted,
        int(payload.get("valid_branch_count", -1)) == attempted,
        int(payload.get("oracle_equal_branch_count", -1)) == attempted,
        payload.get("complete") is complete,
        payload.get("bound_digests") == bindings,
    )
    if not all(checks):
        raise RuntimeError("Stage-A completion receipt reconciliation failed")
    if expected_rows is not None:
        expected_digests = [row["branch_row_digest"] for row in expected_rows]
        if (branch_digests != expected_digests
                or attempted != len(expected_rows)
                or int(receipt.get("valid_branch_count", -1))
                != sum(bool(row.get("valid")) for row in expected_rows)
                or int(receipt.get("oracle_equal_branch_count", -1))
                != sum(bool(row.get("oracle_outcome_equal"))
                       for row in expected_rows)):
            raise RuntimeError("Stage-A receipt does not bind the expected rows")


def _corpus_identity_payload(manifest: Mapping[str, Any],
                             rows: Sequence[Mapping[str, Any]],
                             branch_rows_sha256: str) -> dict[str, Any]:
    present = {(str(row["state_id"]), int(row["candidate_index"])) for row in rows}
    completed_states = sum(
        all((state["state_id"], index) in present
            for index in range(EXPECTED_CANDIDATES))
        for state in manifest["states"]
    )
    complete = len(rows) == EXPECTED_BRANCHES and completed_states == EXPECTED_STATES
    return {
        "schema": "go2_counterfactual_fidelity_stage_a_corpus_identity_v1_2",
        "stage_a_identity_manifest_digest": manifest["stage_a_identity_manifest_digest"],
        "assay_spec_digest": manifest["assay_spec_digest"],
        "source_state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
        "source_pilot_branch_ledger_sha256": SOURCE_FILES["pilot_branches"]["sha256"],
        "branch_identity_set_digest": manifest["branch_identity_set_digest"],
        "branch_rows_sha256": branch_rows_sha256,
        "branch_row_digests": [row["branch_row_digest"] for row in rows],
        "state_count": EXPECTED_STATES,
        "completed_state_count": completed_states,
        "attempted_branch_count": len(rows),
        "valid_branch_count": sum(bool(row["valid"]) for row in rows),
        "oracle_equal_branch_count": sum(bool(row["oracle_outcome_equal"])
                                         for row in rows),
        "complete": complete,
        "bound_digests": _manifest_bindings(manifest),
    }


def _compile_corpus(manifest: Mapping[str, Any], source: SourceEvidence, out: Path,
                    invocation_runtime_s: float) -> dict[str, Any]:
    completed = _completed_rows(manifest, source, out)
    ordered = [completed[(state["state_id"], candidate_index)]
               for state in manifest["states"]
               for candidate_index in range(EXPECTED_CANDIDATES)
               if (state["state_id"], candidate_index) in completed]
    text = "".join(json.dumps(V1._jsonable(row), sort_keys=True) + "\n"
                   for row in ordered)
    ledger = out / "branch_rows.jsonl"
    expected_ledger_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    prior_receipt_path = out / "corpus_receipt.json"
    prior_receipt: dict[str, Any] | None = None
    if prior_receipt_path.exists():
        try:
            prior_receipt = json.loads(prior_receipt_path.read_text())
            _validate_corpus_receipt(prior_receipt, manifest, out)
        except Exception as exc:
            preserved = _preserve_invalid(
                prior_receipt_path, out, "corpus-receipt-validation-failed")
            print(f"[recovery] preserved {preserved}: {exc}", flush=True)
            prior_receipt = None
    completed_states = sum(all((state["state_id"], index) in completed
                               for index in range(EXPECTED_CANDIDATES))
                           for state in manifest["states"])
    complete = len(ordered) == EXPECTED_BRANCHES and completed_states == EXPECTED_STATES
    payload = _corpus_identity_payload(manifest, ordered, expected_ledger_sha256)
    corpus_digest = canonical_digest(payload)
    expected_row_digests = [row["branch_row_digest"] for row in ordered]
    if (prior_receipt is not None
            and prior_receipt.get("branch_rows_sha256") == expected_ledger_sha256
            and prior_receipt.get("branch_row_digests") == expected_row_digests
            and prior_receipt.get("corpus_digest") == corpus_digest
            and prior_receipt.get("complete") is complete):
        _validate_corpus_receipt(prior_receipt, manifest, out, ordered)
        return prior_receipt
    if ledger.exists() and file_sha256(ledger) != expected_ledger_sha256:
        if prior_receipt is None:
            preserved = _preserve_invalid(ledger, out, "corpus-ledger-validation-failed")
            print(f"[recovery] preserved {preserved}", flush=True)
        atomic_text(ledger, text)
    elif not ledger.exists():
        atomic_text(ledger, text)
    receipt = {
        "schema": "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2",
        "status": STATUS,
        "complete": complete,
        "state_count": EXPECTED_STATES,
        "completed_state_count": completed_states,
        "expected_branch_count": EXPECTED_BRANCHES,
        "attempted_branch_count": len(ordered),
        "valid_branch_count": sum(bool(row["valid"]) for row in ordered),
        "oracle_equal_branch_count": sum(bool(row["oracle_outcome_equal"])
                                         for row in ordered),
        "invalid_branch_count": sum(not bool(row["valid"]) for row in ordered),
        "branch_rows_sha256": expected_ledger_sha256,
        "branch_row_digests": expected_row_digests,
        **_manifest_bindings(manifest),
        "corpus_digest_payload": payload,
        "corpus_digest": corpus_digest,
        "runtime_s_completed_rows": round(sum(float(row.get("wall_time_s") or 0.0)
                                                for row in ordered), 6),
        "runtime_s_this_invocation": round(float(invocation_runtime_s), 6),
        "storage_bytes": ledger.stat().st_size + sum(
            _row_path(out, _identity_for(state, index)).stat().st_size
            for state in manifest["states"] for index in range(EXPECTED_CANDIDATES)
            if _row_path(out, _identity_for(state, index)).is_file()),
    }
    receipt["completion_receipt_digest"] = canonical_digest(receipt)
    atomic_json(out / "corpus_receipt.json", receipt)
    return receipt


def _state_redrive(ctx: V1.BranchContext, state: Mapping[str, Any]) -> tuple[
        dict[str, Any], GeodesicField, Mapping[str, Any], Any, list[BasePose]]:
    topology = V12.link_topology(ctx)
    ctx.begin_episode()
    proprio: list[list[float]] = []
    control: list[list[float]] = []
    context_poses: list[BasePose] = []
    action_context: list[list[float]] = []

    def probe(_tick_index: int, previous: Sequence[float]) -> None:
        proprio.append(proprio_sample(ctx))
        control.append(control_sample(previous))

    warmup = int(state["warmup_blocks"])
    for block_index in range(warmup):
        block = drive_block_with_probe(ctx, probe)
        if block_index >= warmup - CONTEXT_SLOTS:
            action_context.append(action_block_10d(np.asarray(block.executed)[0]))
            context_poses.append(capture_base_pose(ctx))
    verdict = V12.eligible_here(ctx, topology)
    if isinstance(verdict, str):
        raise RuntimeError(f"registered source state redrive ineligible: {verdict}")
    record, field = verdict
    comparisons = {
        "source_step": int(record["boundary"]["source_step"]) == int(state["source_step"]),
        "cell": int(record["cell_id"]) == int(state["cell_id"]),
        "landmark_id": record["goal"]["landmark_id"]
                       == state["goal_designation"]["landmark_id"],
        "landmark_cell": int(record["goal"]["landmark_cell"])
                         == int(state["goal_designation"]["landmark_cell"]),
        "graph_edges": int(record["goal"]["graph_edges"])
                       == int(state["goal_designation"]["graph_edges_to_landmark"]),
        "start_geodesic": float(record["goal"]["start_geodesic_m"])
                          == float(state["goal_designation"]["start_geodesic_m"]),
    }
    failed = [key for key, passed in comparisons.items() if not passed]
    if failed:
        raise RuntimeError(f"registered source state redrive mismatch: {failed}")
    if (len(proprio) < PROPRIO_HISTORY or len(control) < PROPRIO_HISTORY
            or len(context_poses) != CONTEXT_SLOTS
            or len(action_context) != CONTEXT_SLOTS):
        raise RuntimeError("registered source state lacks planning-time history")
    bearing, range_m = _goal_binding(ctx, record["goal"])
    previous = np.asarray(ctx.runner._last_executed, dtype=np.float64)[0].tolist()
    episode_id = int(ctx.runner.episode_states[0].episode_id)
    snapshot = V1.capture_branch_state(
        ctx, goal=record["goal"], identity={
            "state_id": state["state_id"], "scene_id": state["scene_id"],
            "family": state["family"], "split": state["split"],
            "block_index": warmup, "source_step": state["source_step"],
            "episode_id": episode_id,
        })
    dynamic = {
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "scene_id": state["scene_id"],
        "episode_cluster_id": state["episode_cluster_id"],
        "episode_id": episode_id,
        "source_step": int(state["source_step"]),
        "warmup_blocks": warmup,
        "boundary": record["boundary"],
        "goal": record["goal"],
        "goal_binding_input": [math.sin(bearing), math.cos(bearing), range_m],
        "previous_applied_command": [float(value) for value in previous],
        "action_context_blocks": action_context,
        "proprio": np.asarray(proprio[-PROPRIO_HISTORY:], dtype=np.float32).tolist(),
        "control": np.asarray(control[-PROPRIO_HISTORY:], dtype=np.float32).tolist(),
        "context_base_poses": [_base_pose_payload(pose) for pose in context_poses],
        "snapshot_digest": snapshot.digest,
    }
    return dynamic, field, topology, snapshot, context_poses


def _get_or_create_state_record(out: Path, manifest: Mapping[str, Any],
                                state: Mapping[str, Any], dynamic: Mapping[str, Any],
                                context_poses: list[BasePose], renderer: TexturedV03Renderer
                                ) -> dict[str, Any]:
    path = _state_record_path(out, state)
    if path.exists():
        try:
            record = json.loads(path.read_text())
            _validate_state_record(record, state, manifest, out)
        except Exception as exc:
            preserved = _preserve_invalid(path, out, "state-record-validation-failed")
            raise RuntimeError(f"preserved invalid state record {preserved}: {exc}") from exc
        live_digest = canonical_digest(dynamic)
        if record["redrive_projection_digest"] != live_digest:
            mismatch_root = out / "invalid_attempts/state_redrive_mismatches"
            mismatch_root.mkdir(parents=True, exist_ok=True)
            mismatch = {
                "schema":
                    "go2_counterfactual_fidelity_stage_a_state_redrive_mismatch_v1_2",
                "status": "INVALID_LIVE_REDRIVE_MISMATCH_RETAINED_DURABLE_STATE",
                "state_id": state["state_id"],
                "state_identity_digest": state["state_identity_digest"],
                "stage_a_identity_manifest_digest":
                    manifest["stage_a_identity_manifest_digest"],
                "assay_spec_digest": manifest["assay_spec_digest"],
                "retained_state_record_path": str(path.relative_to(out)),
                "retained_state_record_digest": record["state_record_digest"],
                "retained_redrive_projection_digest":
                    record["redrive_projection_digest"],
                "live_redrive_projection_digest": live_digest,
                "live_redrive_projection": V1._jsonable(dynamic),
            }
            mismatch["state_redrive_mismatch_digest"] = canonical_digest(mismatch)
            evidence = mismatch_root / (
                f"{state['state_identity_digest']}.{time.time_ns()}.json")
            atomic_json(evidence, mismatch)
            raise RuntimeError(
                "current redrive differs from retained valid state record; "
                f"retained record unchanged and wrote {evidence}")
        return record
    frames = []
    for slot, pose in enumerate(context_poses):
        result = renderer.render_pose(pose)
        frame_path = (out / "frames" / state["family"]
                      / f"{state['state_identity_digest']}_ctx{slot}.png")
        frames.append(_frame_receipt(result, pose, frame_path, out,
                                     index_key="slot", index_value=slot,
                                     renderer=renderer))
    record = {
        "schema": "go2_counterfactual_fidelity_stage_a_state_record_v1_2",
        "status": STATUS,
        "record_complete": True,
        **dynamic,
        "family": state["family"],
        "split": state["split"],
        "scene_dir": state["scene_dir"],
        "scene_manifest_path": state["scene_manifest_path"],
        "scene_manifest_sha256": state["scene_manifest_sha256"],
        "scene_manifest_byte_count": state["scene_manifest_byte_count"],
        "scene_manifest_canonical_digest": state["scene_manifest_canonical_digest"],
        "raw_manifest_digest": state["raw_manifest_digest"],
        "context_key": state["state_id"],
        "context_frames": frames,
        "context_paths": [frame["path"] for frame in frames],
        "masks": {
            "context_rgb_valid": [True] * CONTEXT_SLOTS,
            "observed_proprio_valid": [True] * PROPRIO_HISTORY,
            "observed_control_valid": [True] * PROPRIO_HISTORY,
            "future_proprio_available": [False] * HORIZONS,
            "target_rgb_valid": [True] * HORIZONS,
        },
        "timing": {
            "command_hz": 10,
            "ticks_per_block": 5,
            "seconds_per_block": 0.5,
            "context_boundary_offsets_blocks": [-2, -1, 0],
            "target_horizons_blocks": [1, 2, 3, 4],
        },
        "renderer": {
            "raw_manifest_digest": renderer.raw_manifest_digest,
            "camera_mount_body": renderer.camera_mount_body,
            "contract_digest": renderer.contract_digest,
            "scene_build_runtime_s": round(renderer.scene_build_runtime_s, 6),
        },
        **_manifest_bindings(manifest),
    }
    record["redrive_projection_digest"] = canonical_digest(dynamic)
    record["state_record_digest"] = canonical_digest(record)
    atomic_json(path, record)
    return record


def _reexecuted_outcome(state: Mapping[str, Any], snapshot: Any,
                        candidate: tuple[str, tuple[str, ...]], branch: Mapping[str, Any],
                        scored: Mapping[str, Any] | None) -> dict[str, Any]:
    row = {
        "state_id": state["state_id"], "scene_id": state["scene_id"],
        "family": state["family"], "split": state["split"],
        "episode_id": int(snapshot.identity["episode_id"]),
        "source_step": int(snapshot.identity["source_step"]),
        "warmup_blocks": int(state["warmup_blocks"]),
        "landmark_id": state["goal_designation"]["landmark_id"],
        "landmark_cell": int(state["goal_designation"]["landmark_cell"]),
        "graph_edges_to_landmark": int(
            state["goal_designation"]["graph_edges_to_landmark"]),
        "candidate": candidate[0], "primitives": list(candidate[1]),
        "requested": branch["requested"], "post_slew": branch["post_slew"],
        "clipped": branch["clipped"], "blocks_completed": branch["blocks_completed"],
        "truncated_at_block": branch["truncated_at_block"],
        "valid": scored is not None,
        "invalid_reason": (None if scored is not None else
                           ("solver_nan" if branch["nan"] else
                            "unlocatable_or_unreachable_geodesic")),
        "snapshot_digest": snapshot.digest,
        "candidate_bank_digest": V1.bank_digest(),
        "progress_contract_digest": progress_digest(),
        "safety_contract_digest": safety_digest(),
        "oracle_v1_2_digest": oracle_v1_2_digest(),
        "state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
    }
    row.update({field: None if scored is None else scored[field]
                for field in ORACLE_LABEL_FIELDS})
    return row


def _write_mismatch(out: Path, identity: Mapping[str, Any], payload: Mapping[str, Any]) -> Path:
    path = out / "invalid_attempts/oracle_mismatches" \
        / f"{identity['branch_identity_digest']}.json"
    if path.exists():
        path = path.with_name(f"{path.stem}.{int(time.time_ns())}.json")
    atomic_json(path, payload)
    return path


def _validate_raw_frame_identity(raw: Mapping[str, Any],
                                 state: Mapping[str, Any],
                                 row: Mapping[str, Any]) -> None:
    frame = row["horizon_frames"][0]
    pose = row["horizon_base_poses"][0]
    checks = (
        raw.get("state_id") == state["state_id"],
        raw.get("candidate") == row["candidate"],
        raw.get("branch_identity_digest") == row["branch_identity_digest"],
        int(raw.get("horizon", -1)) == 1,
        raw.get("captured_base_pose") == pose,
        raw.get("first_pixel_sha256") == frame["pixel_sha256"],
        raw.get("repeat_pixel_sha256") == frame["pixel_sha256"],
        raw.get("shape") == [224, 224, 3],
        raw.get("dtype") == "uint8",
        raw.get("identical") is True,
        raw.get("renderer_contract_digest") == renderer_contract_digest(),
        raw.get("raw_manifest_digest") == state["raw_manifest_digest"],
    )
    if not all(checks):
        raise RuntimeError("raw repeat-render evidence does not bind the retained H1 frame")


def _load_prior_smoke_receipt(path: Path, out: Path,
                              manifest: Mapping[str, Any],
                              state: Mapping[str, Any],
                              rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        receipt = json.loads(path.read_text())
        _verify_self_digest(receipt, "smoke_receipt_digest",
                            "prior Stage-A smoke receipt")
        expected_branch_ids = [row["branch_identity_digest"] for row in rows]
        expected_row_digests = [row["branch_row_digest"] for row in rows]
        if (len(rows) != 6
                or receipt.get("schema")
                != "go2_counterfactual_fidelity_stage_a_smoke_receipt_v1_2"
                or receipt.get("status") != STATUS
                or receipt.get("pass") is not True
                or receipt.get("state_id") != state["state_id"]
                or receipt.get("state_identity_digest")
                != state["state_identity_digest"]
                or receipt.get("branch_identity_digests") != expected_branch_ids
                or receipt.get("branch_row_digests") != expected_row_digests
                or receipt.get("stage_a_identity_manifest_digest")
                != manifest["stage_a_identity_manifest_digest"]
                or receipt.get("assay_spec_digest") != manifest["assay_spec_digest"]):
            raise RuntimeError("prior smoke receipt identity bindings changed")
        raw = receipt.get("raw_frame_identity")
        if not isinstance(raw, Mapping):
            raise RuntimeError("prior smoke receipt lacks raw repeat-render evidence")
        matching = [row for row in rows
                    if row["branch_identity_digest"]
                    == raw.get("branch_identity_digest")]
        if len(matching) != 1:
            raise RuntimeError("prior raw repeat-render branch is not registered")
        _validate_raw_frame_identity(raw, state, matching[0])
        return receipt
    except Exception as exc:
        preserved = _preserve_invalid(path, out, "smoke-receipt-validation-failed")
        print(f"[recovery] preserved {preserved}: {exc}", flush=True)
        return {}


def _raw_identity_from_repeat(state: Mapping[str, Any], row: Mapping[str, Any],
                              renderer: TexturedV03Renderer, repeat: Any
                              ) -> dict[str, Any]:
    frame = row["horizon_frames"][0]
    pose_payload = row["horizon_base_poses"][0]
    repeat_pixel_sha = hashlib.sha256(
        np.ascontiguousarray(repeat.image).tobytes()).hexdigest()
    raw = {
        "state_id": state["state_id"],
        "candidate": row["candidate"],
        "branch_identity_digest": row["branch_identity_digest"],
        "horizon": 1,
        "captured_base_pose": pose_payload,
        "first_pixel_sha256": frame["pixel_sha256"],
        "repeat_pixel_sha256": repeat_pixel_sha,
        "shape": list(repeat.image.shape),
        "dtype": str(repeat.image.dtype),
        "identical": bool(frame["pixel_sha256"] == repeat_pixel_sha),
        "renderer_contract_digest": renderer.contract_digest,
        "raw_manifest_digest": renderer.raw_manifest_digest,
    }
    _validate_raw_frame_identity(raw, state, row)
    return raw


def _recover_raw_frame_identity(out: Path, state: Mapping[str, Any],
                                row: Mapping[str, Any], shared: Any
                                ) -> dict[str, Any]:
    """Repeat-render a retained H1 pose without executing its branch again."""

    scene_manifest_path = Path(state["scene_manifest_path"])
    if (file_sha256(scene_manifest_path) != state["scene_manifest_sha256"]
            or scene_manifest_path.stat().st_size
            != int(state["scene_manifest_byte_count"])):
        raise RuntimeError("registered scene manifest changed during smoke recovery")
    ctx = V1.build_context(Path(state["scene_dir"]), seed=int(state["drive_seed"]),
                           backend="cpu", shared=shared)
    raw_manifest = json.loads(scene_manifest_path.read_text())
    import genesis as gs
    renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
    if (renderer.contract_digest
            != state.get("textured_v03_renderer_contract_digest",
                         renderer_contract_digest())
            or renderer.raw_manifest_digest != state["raw_manifest_digest"]):
        raise RuntimeError("smoke recovery renderer binding changed")
    pose_payload = row["horizon_base_poses"][0]
    _validate_pose(pose_payload)
    pose = BasePose(
        tuple(float(value) for value in pose_payload["position_world_xyz"]),
        tuple(float(value) for value in pose_payload["quaternion_world_wxyz"]),
    )
    repeat = renderer.render_pose(pose)
    raw = _raw_identity_from_repeat(state, row, renderer, repeat)
    del renderer, ctx
    gc.collect()
    return raw


def _assert_branch_smoke_scope(
        manifest: Mapping[str, Any],
        completed: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    state_id = str(manifest["states"][0]["state_id"])
    allowed = {(state_id, index) for index in range(6)}
    extras = sorted(key for key in completed if key not in allowed)
    if extras:
        raise RuntimeError(
            "Stage-A smoke is unavailable after full-branch progress; resume "
            f"--stage branches instead (first extra durable row: {extras[0]})")


def stage_branches(manifest: Mapping[str, Any], source: SourceEvidence, *,
                   smoke: bool, state_offset: int, state_limit: int) -> int:
    out = OUT_ROOT
    if smoke:
        # Refuse before touching an evolving progress receipt/index.  Recovery
        # may restore a missing first-six row, so check both before and after it.
        completed = _completed_rows(manifest, source, out)
        _assert_branch_smoke_scope(manifest, completed)
        _recover_row_records_from_ledger(manifest, source, out)
    else:
        _recover_row_records_from_ledger(manifest, source, out)
    completed = _completed_rows(manifest, source, out)
    if smoke:
        _assert_branch_smoke_scope(manifest, completed)
    previous_smoke: dict[str, Any] = {}
    smoke_path = out / "smoke_receipt.json"
    smoke_state = manifest["states"][0]
    retained_smoke_rows = [
        completed[(smoke_state["state_id"], index)]
        for index in range(6)
        if (smoke_state["state_id"], index) in completed
    ]
    if smoke and smoke_path.is_file():
        previous_smoke = _load_prior_smoke_receipt(
            smoke_path, out, manifest, smoke_state, retained_smoke_rows)
    if not smoke:
        # Reconcile only immutable durable row records before deciding whether
        # execution may proceed.  This repairs a stale/interrupted ledger or
        # receipt without loading Genesis or generating an outcome.
        _compile_corpus(manifest, source, out, 0.0)
        _validate_full_run_smoke_gate(manifest, out, completed)
    selected_states = ([manifest["states"][0]] if smoke else
                       manifest["states"][state_offset:state_offset + state_limit])
    shared = V1._load_shared("cpu")
    invocation_started = time.time()
    new_rows = 0
    raw_frame_identity = previous_smoke.get("raw_frame_identity")
    for state in selected_states:
        allowed = range(6) if smoke else range(EXPECTED_CANDIDATES)
        missing = [index for index in allowed
                   if (state["state_id"], index) not in completed]
        if not missing:
            if smoke and raw_frame_identity is None:
                retained = [completed[(state["state_id"], index)]
                            for index in range(6)]
                raw_frame_identity = _recover_raw_frame_identity(
                    out, state, retained[0], shared)
            print(f"[stage-a] retain complete {state['state_id']}", flush=True)
            continue
        print(f"[stage-a] {state['state_id']} missing={missing}", flush=True)
        scene_manifest_path = Path(state["scene_manifest_path"])
        if (file_sha256(scene_manifest_path) != state["scene_manifest_sha256"]
                or scene_manifest_path.stat().st_size
                != int(state["scene_manifest_byte_count"])):
            raise RuntimeError("registered scene manifest changed after identity issue")
        ctx = V1.build_context(Path(state["scene_dir"]), seed=int(state["drive_seed"]),
                               backend="cpu", shared=shared)
        dynamic, field, topology, snapshot, context_poses = _state_redrive(ctx, state)
        raw_manifest = json.loads(scene_manifest_path.read_text())
        import genesis as gs
        renderer = TexturedV03Renderer(ctx, gs=gs, raw_manifest=raw_manifest)
        if (renderer.contract_digest
                != manifest["textured_v03_renderer_contract_digest"]
                or renderer.raw_manifest_digest != state["raw_manifest_digest"]):
            raise RuntimeError("runtime corrected renderer binding changed")
        state_record = _get_or_create_state_record(
            out, manifest, state, dynamic, context_poses, renderer)

        for candidate_index in missing:
            started = time.time()
            identity = _identity_for(state, candidate_index)
            candidate = V1.CANDIDATE_BANK[candidate_index]
            accepted = source.outcomes[(state["state_id"], candidate_index)]
            requested, planned_post_slew, action_blocks = candidate_planning_trajectory(
                candidate, state_record["previous_applied_command"])
            branch, horizon_poses = execute_branch_with_pose_capture(
                ctx, snapshot, candidate, field=field, topology=topology)
            scored = V12.score_branch_v12(branch)
            reexecuted = _reexecuted_outcome(state, snapshot, candidate, branch, scored)
            accepted_projection = _outcome_projection(accepted.row)
            reexecuted_projection = _outcome_projection(reexecuted)
            if canonical_digest(reexecuted_projection) != canonical_digest(
                    accepted_projection):
                mismatch = _write_mismatch(out, identity, {
                    "schema": "go2_counterfactual_fidelity_stage_a_oracle_mismatch_v1_2",
                    "state_id": state["state_id"], "candidate": candidate[0],
                    "branch_identity_digest": identity["branch_identity_digest"],
                    "accepted_outcome_witness": _accepted_witness(accepted),
                    "accepted_projection": accepted_projection,
                    "reexecuted_projection": reexecuted_projection,
                })
                raise RuntimeError(f"oracle replay differs from accepted pilot: {mismatch}")
            if (len(horizon_poses) != HORIZONS
                    or not np.allclose(np.asarray(branch["post_slew"], dtype=np.float64),
                                       np.asarray(planned_post_slew, dtype=np.float64),
                                       rtol=0.0, atol=1e-6)):
                raise RuntimeError("complete accepted branch lacks H1..4 or exact action plan")
            frames = []
            for horizon, pose in enumerate(horizon_poses, start=1):
                result = renderer.render_pose(pose)
                frame_path = (out / "frames" / state["family"]
                              / f"{identity['branch_identity_digest']}_h{horizon}.png")
                frames.append(_frame_receipt(
                    result, pose, frame_path, out, index_key="horizon",
                    index_value=horizon, renderer=renderer))
            if smoke and raw_frame_identity is None:
                repeat = renderer.render_pose(horizon_poses[0])
                first_pixel_sha = frames[0]["pixel_sha256"]
                repeat_pixel_sha = hashlib.sha256(
                    np.ascontiguousarray(repeat.image).tobytes()).hexdigest()
                raw_frame_identity = {
                    "state_id": state["state_id"],
                    "candidate": candidate[0],
                    "branch_identity_digest": identity["branch_identity_digest"],
                    "horizon": 1,
                    "captured_base_pose": _base_pose_payload(horizon_poses[0]),
                    "first_pixel_sha256": first_pixel_sha,
                    "repeat_pixel_sha256": repeat_pixel_sha,
                    "shape": list(repeat.image.shape),
                    "dtype": str(repeat.image.dtype),
                    "identical": bool(first_pixel_sha == repeat_pixel_sha),
                    "renderer_contract_digest": renderer.contract_digest,
                    "raw_manifest_digest": renderer.raw_manifest_digest,
                }
            row = {
                "schema": "go2_counterfactual_fidelity_stage_a_branch_row_v1_2",
                "status": STATUS,
                "record_complete": True,
                "state_id": state["state_id"],
                "state_index": int(state["state_index"]),
                "state_identity_digest": state["state_identity_digest"],
                "state_record_digest": state_record["state_record_digest"],
                "branch_identity_digest": identity["branch_identity_digest"],
                "context_key": state["state_id"],
                "family": state["family"], "split": state["split"],
                "scene_id": state["scene_id"], "scene_dir": state["scene_dir"],
                "scene_manifest_path": state["scene_manifest_path"],
                "scene_manifest_sha256": state["scene_manifest_sha256"],
                "scene_manifest_byte_count": state["scene_manifest_byte_count"],
                "scene_manifest_canonical_digest": state["scene_manifest_canonical_digest"],
                "raw_manifest_digest": state["raw_manifest_digest"],
                "episode_cluster_id": state["episode_cluster_id"],
                "episode_id": state_record["episode_id"],
                "source_step": state_record["source_step"],
                "candidate_index": candidate_index,
                "candidate": candidate[0], "primitives": list(candidate[1]),
                "requested": requested,
                "realised_requested_prefix": branch["requested"],
                "candidate_post_slew_plan": planned_post_slew,
                "post_slew": branch["post_slew"],
                "action_blocks": action_blocks,
                "action_context_blocks": state_record["action_context_blocks"],
                "previous_applied_command": state_record["previous_applied_command"],
                "goal": state_record["goal"],
                "goal_binding_input": state_record["goal_binding_input"],
                "proprio": state_record["proprio"],
                "control": state_record["control"],
                "masks": state_record["masks"],
                "timing": state_record["timing"],
                "context_base_poses": state_record["context_base_poses"],
                "context_frames": state_record["context_frames"],
                "context_paths": state_record["context_paths"],
                "horizon_base_poses": [_base_pose_payload(pose) for pose in horizon_poses],
                "horizon_frames": frames,
                "horizon_paths": [frame["path"] for frame in frames],
                "blocks_completed": branch["blocks_completed"],
                "truncated_at_block": branch["truncated_at_block"],
                "snapshot_digest": snapshot.digest,
                "valid": True,
                "invalid_reason": None,
                "oracle_outcome_equal": True,
                "accepted_outcome_witness": _accepted_witness(accepted),
                "accepted_oracle_outcome": accepted.row,
                "reexecuted_outcome_projection_digest": canonical_digest(
                    reexecuted_projection),
                **_manifest_bindings(manifest),
                "wall_time_s": round(time.time() - started, 6),
                "storage_bytes": sum(frame["byte_count"] for frame in frames),
            }
            row.update({field: scored[field] for field in ORACLE_LABEL_FIELDS})
            row["branch_row_digest"] = canonical_digest(row)
            path = _row_path(out, identity)
            if path.exists():
                _preserve_invalid(path, out, "row-overwrite-refused")
            atomic_json(path, row)
            _validate_branch_row(row, state, identity, manifest, source, out)
            completed[(state["state_id"], candidate_index)] = row
            new_rows += 1
        del renderer, ctx
        gc.collect()
    receipt = _compile_corpus(
        manifest, source, out, time.time() - invocation_started)
    if smoke:
        state = manifest["states"][0]
        rows = [completed[(state["state_id"], index)] for index in range(6)
                if (state["state_id"], index) in completed]
        smoke_text = "".join(
            json.dumps(V1._jsonable(row), sort_keys=True) + "\n" for row in rows)
        smoke_ledger_sha256 = hashlib.sha256(smoke_text.encode("utf-8")).hexdigest()
        smoke_corpus_digest = canonical_digest(_corpus_identity_payload(
            manifest, rows, smoke_ledger_sha256))
        smoke_receipt = {
            "schema": "go2_counterfactual_fidelity_stage_a_smoke_receipt_v1_2",
            "status": STATUS,
            "pass": bool(len(rows) == 6 and all(row["valid"]
                                                and row["oracle_outcome_equal"]
                                                and len(row["horizon_frames"]) == HORIZONS
                                                for row in rows)
                         and isinstance(raw_frame_identity, Mapping)
                         and raw_frame_identity.get("identical") is True),
            "resume_only_verified": bool(new_rows == 0),
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "branch_identity_digests": [row["branch_identity_digest"] for row in rows],
            "branch_row_digests": [row["branch_row_digest"] for row in rows],
            "stage_a_identity_manifest_digest":
                manifest["stage_a_identity_manifest_digest"],
            "assay_spec_digest": manifest["assay_spec_digest"],
            "partial_corpus_digest": smoke_corpus_digest,
            "new_rows_this_invocation": new_rows,
            "raw_frame_identity": raw_frame_identity,
        }
        smoke_receipt["smoke_receipt_digest"] = canonical_digest(smoke_receipt)
        atomic_json(out / "smoke_receipt.json", smoke_receipt)
        print(json.dumps(smoke_receipt, indent=2, sort_keys=True))
        return 0 if smoke_receipt["pass"] else 1
    print(json.dumps({
        "complete": receipt["complete"],
        "new_rows_this_invocation": new_rows,
        "attempted_branch_count": receipt["attempted_branch_count"],
        "valid_branch_count": receipt["valid_branch_count"],
        "oracle_equal_branch_count": receipt["oracle_equal_branch_count"],
        "corpus_digest": receipt["corpus_digest"],
    }, indent=2, sort_keys=True))
    return 0 if receipt["complete"] else 1


def _validate_latents_index(index: Mapping[str, Any], manifest: Mapping[str, Any],
                            receipt: Mapping[str, Any],
                            states: Sequence[Mapping[str, Any]],
                            rows: Sequence[Mapping[str, Any]], *,
                            complete: bool) -> None:
    _verify_self_digest(index, "latents_index_digest", "Stage-A latent index")
    expected_state_ids = {state["state_id"] for state in states}
    expected_state_digests = {state["state_identity_digest"] for state in states}
    expected_branch_ids = {row["branch_identity_digest"] for row in rows}
    expected_row_digests = {row["branch_row_digest"] for row in rows}
    context_records = index.get("context_records", [])
    horizon_records = index.get("horizon_records", [])
    digest_bindings = (
        "stage_a_identity_manifest_digest", "assay_spec_digest",
        "candidate_bank_digest", "oracle_v1_2_digest",
        "render_contract_digest", "textured_v03_renderer_contract_digest",
        "preprocess_contract_digest", "preprocessing_digest",
        "target_encoder_digest", "target_encoder_checkpoint_sha256",
        "source_state_manifest_digest", "source_pilot_branch_ledger_sha256",
    )
    checks = (
        index.get("schema")
        == "go2_counterfactual_fidelity_stage_a_latents_index_v1_2",
        index.get("status") == STATUS,
        index.get("complete") is complete,
        int(index.get("state_count", -1)) == len(states),
        int(index.get("branch_count", -1)) == len(rows),
        index.get("context_shape")
        == [len(states), CONTEXT_SLOTS, TARGET_ENCODER["tokens"],
            TARGET_ENCODER["token_dim"]],
        index.get("horizon_shape")
        == [len(rows), HORIZONS, TARGET_ENCODER["tokens"],
            TARGET_ENCODER["token_dim"]],
        index.get("dtype") == "float16",
        index.get("encoder_compute_dtype") == "float32",
        int(index.get("tokens", -1)) == TARGET_ENCODER["tokens"],
        int(index.get("token_dim", -1)) == TARGET_ENCODER["token_dim"],
        int(index.get("context_slots", -1)) == CONTEXT_SLOTS,
        int(index.get("horizons", -1)) == HORIZONS,
        index.get("corpus_digest") == receipt["corpus_digest"],
        index.get("branch_rows_sha256") == receipt["branch_rows_sha256"],
        all(index.get(key) == manifest[key] for key in digest_bindings),
        isinstance(context_records, list) and len(context_records) == len(states),
        isinstance(horizon_records, list) and len(horizon_records) == len(rows),
        {record.get("state_id") for record in context_records} == expected_state_ids,
        {record.get("state_identity_digest") for record in context_records}
        == expected_state_digests,
        {record.get("branch_identity_digest") for record in horizon_records}
        == expected_branch_ids,
        {record.get("branch_row_digest") for record in horizon_records}
        == expected_row_digests,
    )
    if not all(checks):
        raise RuntimeError("Stage-A latent index does not bind the current corpus")


def _validate_full_run_smoke_gate(
        manifest: Mapping[str, Any], out: Path,
        completed: Mapping[tuple[str, int], Mapping[str, Any]]) -> None:
    """Admit an initial strict 1/6 smoke or a fully completed 20/240 no-op."""

    ordered = [completed[(state["state_id"], candidate_index)]
               for state in manifest["states"]
               for candidate_index in range(EXPECTED_CANDIDATES)
               if (state["state_id"], candidate_index) in completed]
    receipt_path = out / "corpus_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("full Stage A requires a corpus receipt")
    receipt = json.loads(receipt_path.read_text())
    ledger_rows = [json.loads(line) for line in
                   (out / "branch_rows.jsonl").read_text().splitlines()
                   if line.strip()]
    _validate_corpus_receipt(receipt, manifest, out, ledger_rows)
    for ledger_row in ledger_rows:
        key = (str(ledger_row.get("state_id")),
               int(ledger_row.get("candidate_index", -1)))
        if key not in completed or completed[key] != ledger_row:
            raise RuntimeError("Stage-A corpus ledger differs from retained row records")

    if receipt.get("complete") is True:
        if len(ordered) != EXPECTED_BRANCHES or len(ledger_rows) != EXPECTED_BRANCHES:
            raise RuntimeError("completed Stage-A receipt lacks all registered rows")
        # Branch completion is independently durable.  Immediately after the
        # first 240-row run, the encoder index is legitimately still the 1/6
        # smoke index; the encoder owns its later 20/240 completion.
        return

    smoke_state = manifest["states"][0]
    smoke_rows = [completed[(smoke_state["state_id"], index)]
                  for index in range(6)
                  if (smoke_state["state_id"], index) in completed]
    if len(smoke_rows) != 6 or len(ordered) < 6 or ledger_rows != ordered:
        raise RuntimeError(
            "full Stage A requires the registered smoke rows and reconciled progress")
    smoke_text = "".join(
        json.dumps(V1._jsonable(row), sort_keys=True) + "\n" for row in smoke_rows)
    smoke_ledger_sha256 = hashlib.sha256(smoke_text.encode("utf-8")).hexdigest()
    smoke_corpus_digest = canonical_digest(_corpus_identity_payload(
        manifest, smoke_rows, smoke_ledger_sha256))

    branch_path = out / "smoke_receipt.json"
    encoding_path = out / "smoke_encoding_receipt.json"
    if not branch_path.is_file() or not encoding_path.is_file():
        raise RuntimeError("full Stage A requires branch and encoding smoke receipts")
    index_path = out / "latents_index.json"
    if not index_path.is_file():
        raise RuntimeError("initial full Stage A requires the smoke latent index")
    index = json.loads(index_path.read_text())
    branch = json.loads(branch_path.read_text())
    encoding = json.loads(encoding_path.read_text())
    _verify_self_digest(branch, "smoke_receipt_digest", "Stage-A smoke receipt")
    _verify_self_digest(encoding, "smoke_encoding_receipt_digest",
                        "Stage-A smoke encoding receipt")
    expected_branch_ids = [row["branch_identity_digest"] for row in smoke_rows]
    expected_row_digests = [row["branch_row_digest"] for row in smoke_rows]
    common = {
        "stage_a_identity_manifest_digest": manifest["stage_a_identity_manifest_digest"],
        "assay_spec_digest": manifest["assay_spec_digest"],
        "state_id": smoke_state["state_id"],
        "branch_identity_digests": expected_branch_ids,
        "branch_row_digests": expected_row_digests,
        "partial_corpus_digest": smoke_corpus_digest,
    }
    checks = (
        branch.get("schema")
        == "go2_counterfactual_fidelity_stage_a_smoke_receipt_v1_2",
        encoding.get("schema")
        == "go2_counterfactual_fidelity_stage_a_smoke_encoding_receipt_v1_2",
        branch.get("pass") is True,
        encoding.get("pass") is True,
        branch.get("resume_only_verified") is True,
        encoding.get("resume_only_verified") is True,
        int(branch.get("new_rows_this_invocation", -1)) == 0,
        int(encoding.get("new_context_shards_this_invocation", -1)) == 0,
        int(encoding.get("new_horizon_shards_this_invocation", -1)) == 0,
        branch.get("state_identity_digest") == smoke_state["state_identity_digest"],
        branch.get("raw_frame_identity", {}).get("identical") is True,
        encoding.get("raw_frame_identity") == branch.get("raw_frame_identity"),
        all(branch.get(key) == value for key, value in common.items()),
        all(encoding.get(key) == value for key, value in common.items()),
    )
    if not all(checks):
        raise RuntimeError(
            "full Stage A is gated on cross-bound zero-regeneration smoke reruns")
    raw = branch["raw_frame_identity"]
    matching = [row for row in smoke_rows
                if row["branch_identity_digest"] == raw.get("branch_identity_digest")]
    if len(matching) != 1:
        raise RuntimeError("smoke raw repeat evidence binds an unknown branch")
    _validate_raw_frame_identity(raw, smoke_state, matching[0])
    smoke_corpus_binding = {
        "corpus_digest": smoke_corpus_digest,
        "branch_rows_sha256": smoke_ledger_sha256,
    }
    _validate_latents_index(
        index, manifest, smoke_corpus_binding, [smoke_state], smoke_rows,
        complete=False)
    if encoding.get("latents_index_digest") != index.get("latents_index_digest"):
        raise RuntimeError("smoke encoding receipt does not bind the current latent index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("digests", "prepare", "smoke", "branches"),
                        default="digests")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-limit", type=int, default=EXPECTED_STATES)
    args = parser.parse_args()
    if args.backend != "cpu":
        raise SystemExit("Stage A is frozen to the qualified CPU backend")
    if args.state_offset < 0 or args.state_limit < 1:
        raise SystemExit("state slice must be non-negative and non-empty")
    if args.stage == "digests":
        print(json.dumps({
            "source_state_manifest_digest": SOURCE_STATE_MANIFEST_DIGEST,
            "source_pilot_branch_ledger_sha256": SOURCE_FILES["pilot_branches"]["sha256"],
            "candidate_bank_digest": V1.bank_digest(),
            "oracle_v1_2_digest": oracle_v1_2_digest(),
            "assay_spec_digest": assay_spec_digest(),
            "render_contract_digest": render_contract_digest(),
            "textured_v03_renderer_contract_digest": renderer_contract_digest(),
            "preprocess_contract_digest": preprocess_contract_digest(),
            "preprocessing_digest": PREPROCESSING_DIGEST,
            "target_encoder_digest": target_encoder_digest(),
        }, indent=2, sort_keys=True))
        return 0
    source = load_source_evidence()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_ROOT / "stage_a_identity_manifest.json"
    if args.stage == "prepare":
        expected = build_identity_manifest(source)
        if manifest_path.exists():
            try:
                observed = json.loads(manifest_path.read_text())
                validate_identity_manifest(observed, source)
            except Exception:
                if _outcome_generation_started(OUT_ROOT):
                    raise RuntimeError(
                        "Stage-A contract changed after outcome generation began")
                _preserve_invalid(
                    manifest_path, OUT_ROOT, "preexecution-contract-superseded")
            else:
                print(json.dumps({
                    "retained": True,
                    "stage_a_identity_manifest_digest":
                        observed["stage_a_identity_manifest_digest"],
                    "states": observed["state_count_registered"],
                    "branches": observed["attempted_branch_count_registered"],
                }, indent=2, sort_keys=True))
                return 0
        atomic_json(manifest_path, expected)
        print(json.dumps({
            "retained": False,
            "stage_a_identity_manifest_digest":
                expected["stage_a_identity_manifest_digest"],
            "states": expected["state_count_registered"],
            "branches": expected["attempted_branch_count_registered"],
        }, indent=2, sort_keys=True))
        return 0
    if not manifest_path.is_file():
        raise SystemExit("run --stage prepare before executing Stage-A branches")
    manifest = json.loads(manifest_path.read_text())
    validate_identity_manifest(manifest, source)
    return stage_branches(manifest, source, smoke=args.stage == "smoke",
                          state_offset=args.state_offset, state_limit=args.state_limit)


if __name__ == "__main__":
    raise SystemExit(main())
