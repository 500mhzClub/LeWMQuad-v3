#!/usr/bin/env python3
"""Frozen-probe H=1..4 occupancy diagnostic on oracle-v1.2 pilot branches.

This diagnostic is deliberately separate from the planning-utility pipeline.  It
does not fit a spatial head, alter the frozen H=1 probe, or execute a simulator.
It consumes the exact Stage-A replay products under
``.generated/go2_counterfactual_fidelity_v1_2``:

* ``labels`` builds renderer-aligned V4 observable camera-ray rasters from the
  already-captured horizon poses and immutable scene manifests;
* ``assay`` first applies the pre-existing final-epoch probe to true target
  latents and freezes an independent >=0.35 occupied-IoU decision for H=1..4;
* only then are the already-produced output shards from the 32 frozen epoch-21
  predictors opened, and only qualified horizons are reported.  This script
  never opens a predictor checkpoint.

Every branch label and checkpoint score record is an atomic/resumable shard.
An existing artifact is reused only after its self digest, byte digest, identity
and all cross-stage bindings verify.  Invalid attempts are preserved rather
than overwritten or mixed with a new contract.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.
"""
from __future__ import annotations

import argparse
import ast
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
for _source_root in (ROOT, ROOT / "lewm_worlds", ROOT / "lewm_genesis"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from lewm.benchmarks import go2_dynamic_cell_square_projection as dynamic_projection  # noqa: E402
from lewm.benchmarks import go2_observable_camera_ray_evidence_v4 as ray_v4  # noqa: E402
from lewm.oracle import go2_textured_v03_renderer as textured_v03  # noqa: E402
from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict  # noqa: E402
from scripts import build_go2_observable_camera_ray_fit_v4 as v4_builder  # noqa: E402
from scripts import run_dev_frozen_dense_representation_screen_v1 as dense_probe  # noqa: E402


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
STAGE_ROOT = ROOT / ".generated/go2_counterfactual_fidelity_v1_2"
LABEL_ROOT = STAGE_ROOT / "occupancy_labels"
RESULT_ROOT = STAGE_ROOT / "occupancy_results"

STAGE_A_MANIFEST = STAGE_ROOT / "stage_a_identity_manifest.json"
STAGE_A_ROWS = STAGE_ROOT / "branch_rows.jsonl"
STAGE_A_RECEIPT = STAGE_ROOT / "corpus_receipt.json"
STAGE_A_LATENTS = STAGE_ROOT / "latents_index.json"

PROBE_PACKAGE_PATH = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "factorial_v1/spatial_retention/probe_package.json"
)
PROBE_WEIGHTS_PATH = PROBE_PACKAGE_PATH.with_name("probe_final_epoch.pt")
PROBE_PACKAGE_DIGEST = (
    "b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686"
)
PROBE_PACKAGE_FILE_SHA256 = (
    "3d216f4e60851861d521705397ae0f43f783a8ceb1852685f42ab27ff0260c75"
)
PROBE_WEIGHTS_SHA256 = (
    "95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322"
)
PROBE_SPECIFICATION_DIGEST = (
    "646073a9b0a43d7a6c3230f55b3d68026d0632af70726c196603cb7ccf182478"
)

EXPECTED_STATES = 20
EXPECTED_BRANCHES = 240
EXPECTED_CANDIDATES = 12
EXPECTED_FAMILIES = 8
HORIZONS = 4
CONTEXT_SLOTS = 3
TOKENS = 768
TOKEN_DIM = 1024
TOKEN_GRID = (24, 32)
LABEL_SHAPE = (HORIZONS, 64, 64)
LABEL_DTYPE = np.dtype("uint8")
QUALIFICATION_IOU_FLOOR = 0.35
STAGE_A_TARGET_NORMALISATION = (
    "raw final-block tokens rounded to float16; consumers reload float16 as "
    "float32 and apply F.layer_norm over the 1024-D token dimension"
)
STAGE_A_ASSAY_SPEC_DIGEST = (
    "39545af7599da2f2a1bf171c050489eea9f8637137bc1a9c0af3a193d1aaaf3a"
)
STAGE_A_IDENTITY_MANIFEST_DIGEST = (
    "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a"
)
FROZEN_CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)
FROZEN_PROGRESS_CONTRACT_DIGEST = (
    "840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d"
)
FROZEN_SAFETY_CONTRACT_DIGEST = (
    "5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f"
)
FROZEN_ORACLE_V1_2_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
)
FROZEN_RENDER_CONTRACT_DIGEST = (
    "2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17"
)
FROZEN_PREPROCESS_CONTRACT_DIGEST = (
    "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"
)
FROZEN_PREPROCESSING_DIGEST = (
    "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"
)
FROZEN_TARGET_ENCODER_DIGEST = (
    "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"
)
FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256 = (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
STAGE_BC_PREDICTION_NORMALISATION = (
    "scripts.dev_proprio_predictor_v1.unroll applies the frozen "
    "run_dev_v03_temporal_action_jepa_v1.normalise after every step"
)
STAGE_BC_PREDICTION_REPRESENTATION = (
    "autoregressive H1-H4 normalized predictor tokens, rounded float16 "
    "exactly as the frozen direct evaluator"
)
FROZEN_SCORE_BATCH = 12
FROZEN_SEED_COUNT = 8
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
CELLS = (
    "rgb_one_step", "rgb_rollout", "proprio_one_step", "proprio_rollout"
)
SEEDS = (
    2_026_080_901, 2_026_080_902, 2_026_080_903, 2_026_080_904,
    2_026_080_905, 2_026_080_906, 2_026_080_907, 2_026_080_908,
)
FROZEN_CONFIRMATORY_COMMIT = "443e5914694a533534486b629e95ec15f8df9b7a"
FROZEN_RUN_PACKAGE_DIGEST = (
    "cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991"
)

LABEL_SCHEMA = "go2_counterfactual_fidelity_occupancy_labels_v1_2"
LABEL_RECORD_SCHEMA = "go2_counterfactual_fidelity_occupancy_label_record_v1_2"
LABEL_INDEX_SCHEMA = "go2_counterfactual_fidelity_occupancy_label_index_v1_2"
LABEL_RECEIPT_SCHEMA = "go2_counterfactual_fidelity_occupancy_label_receipt_v1_2"
TRUE_GATE_SCHEMA = "go2_counterfactual_fidelity_occupancy_true_target_gate_v1_2"
SCORE_RECORD_SCHEMA = "go2_counterfactual_fidelity_occupancy_score_record_v1_2"
SCORE_RECEIPT_SCHEMA = "go2_counterfactual_fidelity_occupancy_score_receipt_v1_2"
RESULT_SCHEMA = "go2_counterfactual_fidelity_occupancy_result_v1_2"

PRE_FIX_SOURCE_SHA256 = (
    "366c36f766bcf064cb7e68a46c7cb7922cf6e4138394b6cb98e1656668bd71c9"
)
PRE_FIX_SOURCE_BYTE_COUNT = 121_919
PRE_FIX_SOURCE_GIT_BLOB_OID = "8f97a8de468e44365cc6956cf304c1da9edf4914"
SOURCE_PRESERVATION_RECEIPT_DIGEST = (
    "ec66410e3d0f08f4c0bcc911684c9cd2c4f436cd33b00db6b574592379afd318"
)
PROTECTED_SCIENTIFIC_AST_DIGEST = (
    "0536504c46422a69733853786e45f906a0fa63defa9af7e4f7a63f1789fa1365"
)
PROTECTED_SCIENTIFIC_FUNCTIONS = (
    "label_contract", "assay_spec", "base_pose_values",
    "_quaternion_yaw_xyzw", "_rotation_world_from_wxyz",
    "validate_recorded_camera", "_scene_payload", "_make_label_payload",
    "_write_label", "validate_probe_package_metadata", "tensor_state_digest",
    "load_probe", "occupied_counts", "_load_label_array",
    "_load_horizon_latent", "_true_record_valid", "score_true_targets",
)
FROZEN_LABEL_IMPLEMENTATION_BINDINGS = {
    "lewm/benchmarks/go2_dynamic_cell_square_projection.py":
        "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py":
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85",
    "lewm/oracle/go2_textured_v03_renderer.py":
        "392439be92c128f639c8c9682627530b34660168229c24c9944d847372524aba",
    "lewm_worlds/lewm_worlds/manifest.py":
        "5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888",
    "scripts/build_go2_observable_camera_ray_fit_v4.py":
        "4efb0517130df39a1953539755d82289b16e89b314bba5713d6d9d944acf1d16",
    "scripts/run_dev_frozen_dense_representation_screen_v1.py":
        "6402883b211f7cb40a923e78e9ba78c9510bb0310b25df0c60ce9b73cba530cb",
    "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py":
        PRE_FIX_SOURCE_SHA256,
    "scripts/run_go2_representation_qualification_probe_v1.py":
        "75ddc9e7674549e385a56f6866e9c2a39c034512f9b48af85cb3acb937c75b9a",
}


class OccupancyAssayRefused(RuntimeError):
    """A frozen identity, alignment, completeness or ordering check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OccupancyAssayRefused(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def without(payload: Mapping[str, Any], *fields: str) -> dict[str, Any]:
    omitted = set(fields)
    return {key: value for key, value in payload.items() if key not in omitted}


def verify_self_digest(payload: Mapping[str, Any], field: str, label: str) -> str:
    observed = payload.get(field)
    _require(isinstance(observed, str) and len(observed) == 64,
             f"{label} has no valid {field}")
    expected = canonical_digest(without(payload, field))
    _require(observed == expected,
             f"{label} {field} mismatch: {observed} != {expected}")
    return observed


def read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OccupancyAssayRefused(f"cannot read {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing {label}: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            _require(bool(line.strip()), f"blank {label} line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OccupancyAssayRefused(
                    f"malformed {label} line {line_number}: {exc}") from exc
            _require(isinstance(record, dict),
                     f"non-object {label} line {line_number}")
            records.append(record)
    return records


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True, allow_nan=False)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def atomic_bytes(path: Path, payload: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
    digest = file_sha256(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, path)
    _fsync_dir(path.parent)
    return digest, size


def atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
    atomic_bytes(path, payload)


def _custody_safe(path: Path) -> None:
    parts = path.resolve().parts
    _require(path.name != "sealed_test.json", f"sealed material is inaccessible: {path}")
    _require(not any(part == "sealed" or part.startswith("sealed_") for part in parts),
             f"sealed material is inaccessible: {path}")


def _safe_under(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    resolved = (path if path.is_absolute() else root / path).resolve()
    _custody_safe(resolved)
    _require(resolved == root.resolve() or root.resolve() in resolved.parents,
             f"{label} escapes {root}: {value}")
    return resolved


def preserve_invalid(path: Path, reason: str, recovery: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    destination_root = path.parent / "invalid_attempts"
    destination_root.mkdir(parents=True, exist_ok=True)
    before = file_sha256(path) if path.is_file() else None
    suffix = f"{time.time_ns()}-{hashlib.sha256(reason.encode()).hexdigest()[:12]}"
    destination = destination_root / f"{path.name}.{suffix}.invalid"
    os.replace(path, destination)
    _fsync_dir(destination_root)
    recovery.append({
        "source": str(path), "preserved_as": str(destination),
        "source_sha256": before, "reason": reason,
    })


def _function_ast_projection(source: str) -> tuple[dict[str, str], str]:
    tree = ast.parse(source)
    projection = {
        node.name: ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in PROTECTED_SCIENTIFIC_FUNCTIONS
    }
    _require(set(projection) == set(PROTECTED_SCIENTIFIC_FUNCTIONS),
             "source-equivalence projection lacks a protected function")
    return projection, canonical_digest(projection)


def _top_level_function_digests(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    return {
        node.name: hashlib.sha256(ast.dump(
            node, annotate_fields=True, include_attributes=False,
        ).encode("utf-8")).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def source_equivalence_receipt() -> dict[str, Any]:
    """Prove the interrupted labels/true scores are reusable after the fix.

    The exact pre-fix source was preserved before editing.  The protected AST
    projection covers label construction, probe loading/inference, true-target
    normalization and true-record production.  Only downstream undefined-
    metric propagation and its recovery plumbing may differ.
    """

    recovery_root = RESULT_ROOT / "recovery/source_equivalence"
    preserved_path = recovery_root / "pre_fix_366c36f766bcf064.py"
    preservation_path = recovery_root / "source_preservation_receipt.json"
    preservation = read_json(
        preservation_path, "pre-fix source preservation receipt")
    verify_self_digest(
        preservation, "source_preservation_receipt_digest",
        "pre-fix source preservation receipt")
    _require(preservation["source_preservation_receipt_digest"]
             == SOURCE_PRESERVATION_RECEIPT_DIGEST
             and preservation.get("sha256") == PRE_FIX_SOURCE_SHA256
             and int(preservation.get("byte_count", -1))
             == PRE_FIX_SOURCE_BYTE_COUNT
             and preservation.get("git_blob_oid") == PRE_FIX_SOURCE_GIT_BLOB_OID
             and preservation.get("labels_corpus_digest")
             == "a402ee134a0ec854b9936699e42e0a2c715ea70ac99a2c0393ee09ba6ac41a27"
             and int(preservation.get("true_target_records_completed", -1))
             == EXPECTED_BRANCHES
             and int(preservation.get("prediction_artifacts_opened_by_stage_d", -1))
             == 0,
             "pre-fix source preservation provenance differs")
    _require(preserved_path.is_file()
             and preserved_path.stat().st_size == PRE_FIX_SOURCE_BYTE_COUNT
             and file_sha256(preserved_path) == PRE_FIX_SOURCE_SHA256,
             "preserved pre-fix source bytes differ")

    active_path = ROOT / "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py"
    old_source = preserved_path.read_text(encoding="utf-8")
    new_source = active_path.read_text(encoding="utf-8")
    _old_projection, old_projection_digest = _function_ast_projection(old_source)
    _new_projection, new_projection_digest = _function_ast_projection(new_source)
    _require(old_projection_digest == new_projection_digest
             == PROTECTED_SCIENTIFIC_AST_DIGEST,
             "label/probe/true-target scientific AST changed during recovery")
    old_functions = _top_level_function_digests(old_source)
    new_functions = _top_level_function_digests(new_source)
    changed_existing = sorted(
        name for name in set(old_functions) & set(new_functions)
        if old_functions[name] != new_functions[name]
    )
    added = sorted(set(new_functions) - set(old_functions))
    removed = sorted(set(old_functions) - set(new_functions))
    _require(not removed,
             f"source-equivalence recovery removed functions: {removed}")
    allowed_changed = {
        "source_bindings", "build_labels", "episode_then_family",
        "freeze_true_target_gate", "_analysis_for_estimator",
        "analyse_occupancy", "run_assay",
    }
    allowed_added = {
        "_function_ast_projection", "_top_level_function_digests",
        "source_equivalence_receipt", "_seed_summary_or_unavailable",
    }
    _require(set(changed_existing).issubset(allowed_changed)
             and set(added).issubset(allowed_added),
             "source recovery changed functions outside estimator/provenance plumbing")
    unified_diff = "".join(difflib.unified_diff(
        old_source.splitlines(keepends=True),
        new_source.splitlines(keepends=True),
        fromfile=f"pre-fix/{PRE_FIX_SOURCE_SHA256}",
        tofile=f"active/{file_sha256(active_path)}",
    ))
    receipt: dict[str, Any] = {
        "schema": "go2_counterfactual_occupancy_source_equivalence_v1_2",
        "status": STATUS,
        "complete": True,
        "recovery_scope": (
            "undefined secondary per-family/equal-family occupied-IoU "
            "propagation only; pooled gate unchanged"
        ),
        "pre_fix_source_sha256": PRE_FIX_SOURCE_SHA256,
        "pre_fix_source_byte_count": PRE_FIX_SOURCE_BYTE_COUNT,
        "pre_fix_source_git_blob_oid": PRE_FIX_SOURCE_GIT_BLOB_OID,
        "source_preservation_receipt_digest":
            SOURCE_PRESERVATION_RECEIPT_DIGEST,
        "active_source_sha256": file_sha256(active_path),
        "active_source_byte_count": active_path.stat().st_size,
        "unified_diff_sha256": hashlib.sha256(
            unified_diff.encode("utf-8")).hexdigest(),
        "unified_diff_byte_count": len(unified_diff.encode("utf-8")),
        "unified_diff": unified_diff,
        "changed_existing_functions": changed_existing,
        "added_functions": added,
        "removed_functions": removed,
        "protected_scientific_functions": list(PROTECTED_SCIENTIFIC_FUNCTIONS),
        "protected_scientific_ast_digest": new_projection_digest,
        "label_generation_changed": False,
        "probe_inference_changed": False,
        "true_target_record_generation_changed": False,
        "pre_fix_labels_corpus_digest":
            "a402ee134a0ec854b9936699e42e0a2c715ea70ac99a2c0393ee09ba6ac41a27",
        "pre_fix_true_target_records_retained": EXPECTED_BRANCHES,
        "pre_fix_prediction_artifacts_opened_by_stage_d": 0,
        "estimator_rule": {
            "equal_family": (
                "unavailable if any of the eight families has no defined rows; "
                "never average seven families and never impute"
            ),
            "per_family": "unavailable when that family has no defined rows",
            "corpus_weighted": "mean over explicitly defined branch rows",
            "gate": "whole-pilot pooled occupied intersection/union only",
        },
    }
    receipt["source_equivalence_receipt_digest"] = canonical_digest(receipt)
    path = recovery_root / "source_equivalence_receipt.json"
    if path.exists():
        existing = read_json(path, "source-equivalence receipt")
        verify_self_digest(existing, "source_equivalence_receipt_digest",
                           "source-equivalence receipt")
        _require(existing == receipt,
                 "existing source-equivalence receipt differs")
        return existing
    atomic_json(path, receipt)
    return receipt


def source_bindings() -> dict[str, str]:
    # Existing label rows were produced by the exact preserved pre-fix source.
    # Reuse is permitted only after the explicit AST/diff equivalence proof.
    source_equivalence_receipt()
    for relative, expected in FROZEN_LABEL_IMPLEMENTATION_BINDINGS.items():
        if relative == "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py":
            continue
        path = ROOT / relative
        _require(path.is_file() and file_sha256(path) == expected,
                 f"frozen label implementation changed: {relative}")
    return dict(FROZEN_LABEL_IMPLEMENTATION_BINDINGS)


def label_contract() -> dict[str, Any]:
    """The fully prospective label and alignment definition."""

    return {
        "schema": LABEL_SCHEMA,
        "status": STATUS,
        "source": "Stage-A exact replay poses; no branch reexecution",
        "horizons": [1, 2, 3, 4],
        "base_frame": "per-horizon base-position/stored-yaw frame",
        "camera": {
            "composition": (
                "go2_dynamic_cell_square_projection.compose_yaw_aligned_camera"
                "(base quaternion XYZW, quaternion-derived stored yaw)"
            ),
            "origin_body_m": list(dynamic_projection.CAMERA_XYZ_BODY_M),
            "horizontal_fov_deg": ray_v4.CAMERA_HORIZONTAL_FOV_DEG,
            "vertical_fov_deg": ray_v4.CAMERA_VERTICAL_FOV_DEG,
            "near_m": ray_v4.CAMERA_NEAR_M,
            "native_crop_shape_hw": list(ray_v4.CAMERA_IMAGE_SHAPE),
            "pixel_ray_shape_hw": list(ray_v4.PIXEL_RAY_SHAPE),
            "pixel_ray_stride_px": ray_v4.PIXEL_RAY_STRIDE_PX,
        },
        "raster": {
            "implementation": (
                "build_frame_evidence_v4 then "
                "rasterize_observable_camera_ray_evidence_v4"
            ),
            "shape_hw": list(ray_v4.OUTPUT_SHAPE),
            "cell_size_m": ray_v4.OUTPUT_CELL_SIZE_M,
            "forward_min_edge_m": ray_v4.OUTPUT_FORWARD_MIN_EDGE_M,
            "left_min_edge_m": ray_v4.OUTPUT_LEFT_MIN_EDGE_M,
            "class_order": {"unknown": 0, "free": 1, "occupied": 2},
        },
        "renderer_object_parity": {
            "included": ["walls", "obstacles", "landmarks"],
            "excluded": ["visual_randomization.distractor_objects"],
            "reason": (
                "the exact textured_v03 branch renderer and CPU branch physics "
                "exclude distractors; labels may not contain invisible objects"
            ),
        },
        "token_alignment": {
            "target_tokens": [TOKENS, TOKEN_DIM],
            "token_grid_hw": list(TOKEN_GRID),
            "stored_target_latents": "raw frozen encoder output rounded float16",
            "stage_a_target_normalisation_contract":
                STAGE_A_TARGET_NORMALISATION,
            "stage_a_encoder_compute_dtype": "float32",
            "true_target_probe_input": (
                "float32 reload then F.layer_norm over the 1024-D token dimension"
            ),
            "predicted_probe_input": (
                "B/C normalized predictor output rounded float16; no second "
                "normalisation"
            ),
            "preprocess": (
                "224x224 RGB rows 28:196 -> bicubic 512x384 -> frozen V03 encoder"
            ),
        },
        "render_contract_digest": textured_v03.renderer_contract_digest(),
    }


LABEL_CONTRACT_DIGEST = canonical_digest(label_contract())


def assay_spec() -> dict[str, Any]:
    return {
        "schema": "go2_counterfactual_fidelity_occupancy_assay_spec_v1_2",
        "status": STATUS,
        "claim_bearing": False,
        "utility_or_spatial_training_permitted": False,
        "probe": {
            "package_digest": PROBE_PACKAGE_DIGEST,
            "package_file_sha256": PROBE_PACKAGE_FILE_SHA256,
            "weights_sha256": PROBE_WEIGHTS_SHA256,
            "specification_digest": PROBE_SPECIFICATION_DIGEST,
            "architecture": "SharedTokenToBev(1024) -> 64x64 x 3",
            "epoch": "fixed final epoch 12; no best-epoch selection",
            "modification_or_refit_permitted": False,
        },
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "qualification": {
            "order": "true target latents before predictor checkpoint access",
            "per_horizon_independent": True,
            "metric": "whole-pilot observable occupied IoU",
            "floor": QUALIFICATION_IOU_FLOOR,
            "failed_horizon_policy": "unavailable; do not score predictor latents",
        },
        "inference": {
            "probe_mode": "eval; argmax over unchanged three-class logits",
            "batch_size": FROZEN_SCORE_BATCH,
            "arithmetic_device": (
                "recorded in every record; a different device is a different "
                "attempt and cannot resume or replace a frozen gate"
            ),
            "threshold_or_calibration_change_permitted": False,
            "normalisation_distinction": {
                "true_targets": (
                    "raw Stage-A f16 -> float32 -> one F.layer_norm"
                ),
                "predictions": (
                    "already-normalized B/C predictor f16 -> float32; no second "
                    "F.layer_norm"
                ),
            },
        },
        "estimators": {
            "row": "observable occupied IoU; undefined union is NaN",
            "primary": (
                "rows within episode cluster, clusters within family, unweighted "
                "mean over the eight families"
            ),
            "secondary": "unweighted mean over all branch rows with defined IoU",
            "whole_pilot": "pooled observable occupied intersection / union",
            "seed_replication_units": 8,
            "rollout_benefit": "rollout IoU minus one-step IoU",
            "interaction": "B_prop minus B_RGB",
        },
    }


ASSAY_SPEC_DIGEST = canonical_digest(assay_spec())


def _count_from(receipt: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = receipt.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for section_name in ("expected", "actual", "counts"):
        section = receipt.get(section_name)
        if isinstance(section, Mapping):
            found = _count_from(section, *names)
            if found is not None:
                return found
    return None


def _row_key(row: Mapping[str, Any]) -> str:
    value = row.get("branch_identity_digest")
    _require(isinstance(value, str) and len(value) == 64,
             "Stage-A row has no branch_identity_digest")
    return value


@dataclass(frozen=True)
class StageABundle:
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    index: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    state_by_id: Mapping[str, dict[str, Any]]
    context_records: Mapping[str, dict[str, Any]]
    horizon_records: Mapping[str, dict[str, Any]]
    identity_digest: str
    corpus_digest: str
    branch_rows_sha256: str
    latents_index_digest: str


def _latent_record_key(record: Mapping[str, Any], kind: str) -> str:
    if kind == "context":
        key = record.get("state_id", record.get("key"))
    else:
        key = record.get("branch_identity_digest",
                         record.get("branch_key", record.get("key")))
    _require(isinstance(key, str) and key,
             f"Stage-A {kind} latent record has no identity key")
    return key


def _verify_latent_record(record: dict[str, Any], *, kind: str,
                          expected_shape: Sequence[int]) -> Path:
    _require(record.get("schema")
             == "go2_counterfactual_fidelity_stage_a_latent_shard_receipt_v1_2"
             and record.get("record_complete") is True
             and record.get("kind") == kind,
             f"Stage-A {kind} latent receipt is incomplete or has another schema")
    verify_self_digest(record, "latent_shard_receipt_digest",
                       f"Stage-A {kind} latent shard receipt")
    _require(record.get("shape") == list(expected_shape),
             f"Stage-A {kind} latent shape differs: {record.get('shape')}")
    _require(record.get("dtype", "float16") in ("float16", "<f2"),
             f"Stage-A {kind} latent dtype differs")
    path = _safe_under(STAGE_ROOT, record.get("relative_path", record.get("path")),
                       f"Stage-A {kind} latent")
    _require(path.is_file(), f"missing Stage-A {kind} latent: {path}")
    expected_bytes = int(np.prod(expected_shape)) * np.dtype(np.float16).itemsize
    observed_bytes = record.get("byte_count", record.get("bytes"))
    _require(int(observed_bytes if observed_bytes is not None else -1)
             == path.stat().st_size == expected_bytes,
             f"Stage-A {kind} latent byte count differs: {path}")
    _require(record.get("sha256") == file_sha256(path),
             f"Stage-A {kind} latent digest differs: {path}")
    record["_resolved_path"] = str(path)
    return path


def load_stage_a() -> StageABundle:
    """Validate complete Stage A without opening any predictor checkpoint."""

    manifest = read_json(STAGE_A_MANIFEST, "Stage-A identity manifest")
    _require(manifest.get("schema")
             == "go2_counterfactual_fidelity_stage_a_identity_manifest_v1_2",
             "Stage-A identity manifest schema differs")
    identity_digest = verify_self_digest(
        manifest, "stage_a_identity_manifest_digest", "Stage-A identity manifest")
    _require(identity_digest == STAGE_A_IDENTITY_MANIFEST_DIGEST,
             "Stage-A identity manifest is not the prepare-frozen pilot")
    spec = manifest.get("assay_spec")
    _require(isinstance(spec, Mapping)
             and canonical_digest(spec) == manifest.get("assay_spec_digest")
             == STAGE_A_ASSAY_SPEC_DIGEST,
             "Stage-A prospective assay specification differs")
    expected_frozen_bindings = {
        "candidate_bank_digest": FROZEN_CANDIDATE_BANK_DIGEST,
        "progress_contract_digest": FROZEN_PROGRESS_CONTRACT_DIGEST,
        "safety_contract_digest": FROZEN_SAFETY_CONTRACT_DIGEST,
        "oracle_v1_2_digest": FROZEN_ORACLE_V1_2_DIGEST,
        "render_contract_digest": FROZEN_RENDER_CONTRACT_DIGEST,
        "textured_v03_renderer_contract_digest":
            textured_v03.renderer_contract_digest(),
        "preprocess_contract_digest": FROZEN_PREPROCESS_CONTRACT_DIGEST,
        "preprocessing_digest": FROZEN_PREPROCESSING_DIGEST,
        "target_encoder_digest": FROZEN_TARGET_ENCODER_DIGEST,
        "target_encoder_checkpoint_sha256":
            FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256,
    }
    for field, expected in expected_frozen_bindings.items():
        _require(manifest.get(field) == expected,
                 f"Stage-A identity manifest {field} differs")
    source = manifest.get("source_bindings")
    _require(isinstance(source, Mapping) and source,
             "Stage-A identity manifest has no source bindings")
    for relative, binding in source.items():
        _require(isinstance(relative, str) and isinstance(binding, Mapping)
                 and binding.get("path") == relative
                 and isinstance(binding.get("sha256"), str)
                 and isinstance(binding.get("byte_count"), int),
                 f"Stage-A source binding is malformed: {relative}")
        relative_path = Path(relative)
        candidate = (ROOT / relative_path).resolve()
        _custody_safe(candidate)
        _require(not relative_path.is_absolute()
                 and ROOT.resolve() in candidate.parents
                 and candidate.is_file()
                 and candidate.stat().st_size == binding["byte_count"]
                 and file_sha256(candidate) == binding["sha256"],
                 f"Stage-A implementation changed: {relative}")
    states = manifest.get("states")
    _require(isinstance(states, list) and len(states) == EXPECTED_STATES,
             "Stage-A identity manifest must contain exactly 20 states")
    state_by_id: dict[str, dict[str, Any]] = {}
    for state in states:
        _require(isinstance(state, dict), "Stage-A state identity is not an object")
        state_id = state.get("state_id")
        _require(isinstance(state_id, str) and state_id not in state_by_id,
                 "Stage-A state identities are absent or duplicated")
        state_by_id[state_id] = state
    _require(collections.Counter(str(x.get("family")) for x in states).keys()
             == set(FAMILIES), "Stage-A does not contain exactly the eight families")

    receipt = read_json(STAGE_A_RECEIPT, "Stage-A completion receipt")
    _require(receipt.get("schema")
             == "go2_counterfactual_fidelity_stage_a_completion_receipt_v1_2",
             "Stage-A completion receipt schema differs")
    receipt_digest_field = next((field for field in (
        "completion_receipt_digest", "corpus_receipt_digest", "receipt_digest")
        if field in receipt), "receipt_digest")
    verify_self_digest(receipt, receipt_digest_field, "Stage-A completion receipt")
    _require(receipt.get("complete") is True,
             "Stage-A completion receipt is not complete")
    _require(receipt.get("stage_a_identity_manifest_digest") == identity_digest,
             "Stage-A receipt binds another identity manifest")
    _require(_count_from(receipt, "state_count", "states") == EXPECTED_STATES,
             "Stage-A receipt does not certify 20 states")
    _require(_count_from(receipt, "attempted_branch_count", "attempted_branches",
                         "branch_count", "rows") == EXPECTED_BRANCHES,
             "Stage-A receipt does not certify 240 branches")
    valid_count = _count_from(receipt, "valid_branch_count", "valid_branches", "valid")
    _require(valid_count == EXPECTED_BRANCHES,
             "Stage-A exact pilot replay is not 240/240 valid")

    branch_rows_sha = file_sha256(STAGE_A_ROWS)
    _require(receipt.get("branch_rows_sha256") == branch_rows_sha,
             "Stage-A branch ledger bytes differ from receipt")
    corpus_payload = receipt.get("corpus_digest_payload")
    corpus_digest = receipt.get("corpus_digest")
    _require(isinstance(corpus_payload, dict)
             and isinstance(corpus_digest, str)
             and canonical_digest(corpus_payload) == corpus_digest,
             "Stage-A corpus digest is not independently reproducible")
    rows = read_jsonl(STAGE_A_ROWS, "Stage-A branch ledger")
    _require(len(rows) == EXPECTED_BRANCHES,
             "Stage-A branch ledger must contain exactly 240 records")
    seen: set[str] = set()
    per_state: collections.Counter[str] = collections.Counter()
    for row in rows:
        _require(row.get("schema")
                 == "go2_counterfactual_fidelity_stage_a_branch_row_v1_2",
                 "Stage-A branch-row schema differs")
        verify_self_digest(row, "branch_row_digest", "Stage-A branch row")
        key = _row_key(row)
        _require(key not in seen, f"duplicate Stage-A branch identity {key}")
        seen.add(key)
        state_id = str(row.get("state_id"))
        _require(state_id in state_by_id, f"unknown Stage-A state {state_id}")
        _require(row.get("record_complete") is True
                 and row.get("valid") is True
                 and row.get("oracle_outcome_equal") is True,
                 f"Stage-A branch {key} is not a complete equal replay")
        _require(row.get("stage_a_identity_manifest_digest") == identity_digest,
                 f"Stage-A branch {key} binds another identity manifest")
        per_state[state_id] += 1
    _require(set(per_state.values()) == {EXPECTED_CANDIDATES},
             "Stage-A does not contain twelve branches per state")

    index = read_json(STAGE_A_LATENTS, "Stage-A latent index")
    _require(index.get("schema")
             == "go2_counterfactual_fidelity_stage_a_latents_index_v1_2",
             "Stage-A latent-index schema differs")
    latents_digest = verify_self_digest(
        index, "latents_index_digest", "Stage-A latent index")
    _require(index.get("complete") is True,
             "Stage-A latent index is not complete")
    for field, expected in (
        ("stage_a_identity_manifest_digest", identity_digest),
        ("branch_rows_sha256", branch_rows_sha),
        ("corpus_digest", corpus_digest),
    ):
        _require(index.get(field) == expected,
                 f"Stage-A latent index {field} differs")
    _require(int(index.get("tokens", TOKENS)) == TOKENS
             and int(index.get("token_dim", TOKEN_DIM)) == TOKEN_DIM
             and int(index.get("horizons", HORIZONS)) == HORIZONS
             and int(index.get("context_slots", CONTEXT_SLOTS)) == CONTEXT_SLOTS,
             "Stage-A latent layout differs from 3/4 x 768 x 1024")
    _require(index.get("context_shape")
             == [EXPECTED_STATES, CONTEXT_SLOTS, TOKENS, TOKEN_DIM]
             and index.get("horizon_shape")
             == [EXPECTED_BRANCHES, HORIZONS, TOKENS, TOKEN_DIM],
             "Stage-A aggregate latent shapes differ")
    context_list = index.get("context_records")
    horizon_list = index.get("horizon_records")
    _require(isinstance(context_list, list) and len(context_list) == EXPECTED_STATES,
             "Stage-A context latent records are incomplete")
    _require(isinstance(horizon_list, list) and len(horizon_list) == EXPECTED_BRANCHES,
             "Stage-A horizon latent records are incomplete")
    contexts: dict[str, dict[str, Any]] = {}
    for original in context_list:
        _require(isinstance(original, dict), "Stage-A context record is malformed")
        record = dict(original)
        key = _latent_record_key(record, "context")
        _require(key in state_by_id and key not in contexts,
                 f"Stage-A context identity differs: {key}")
        _verify_latent_record(record, kind="context",
                              expected_shape=(CONTEXT_SLOTS, TOKENS, TOKEN_DIM))
        contexts[key] = record
    pair_to_digest = {
        f"{row['state_id']}|{row['candidate']}": _row_key(row) for row in rows
    }
    horizons: dict[str, dict[str, Any]] = {}
    for original in horizon_list:
        _require(isinstance(original, dict), "Stage-A horizon record is malformed")
        record = dict(original)
        source_key = _latent_record_key(record, "horizon")
        key = pair_to_digest.get(source_key, source_key)
        _require(key in seen and key not in horizons,
                 f"Stage-A horizon identity differs: {key}")
        _verify_latent_record(record, kind="horizon",
                              expected_shape=(HORIZONS, TOKENS, TOKEN_DIM))
        horizons[key] = record
    _require(set(contexts) == set(state_by_id),
             "Stage-A context latent identity set differs")
    _require(set(horizons) == seen,
             "Stage-A horizon latent identity set differs")

    binding_fields = (
        "render_contract_digest", "textured_v03_renderer_contract_digest",
        "preprocess_contract_digest", "preprocessing_digest",
        "target_encoder_digest", "target_encoder_checkpoint_sha256",
        "candidate_bank_digest", "oracle_v1_2_digest", "assay_spec_digest",
    )
    index_direct_bindings = set(binding_fields)
    for field in binding_fields:
        expected = manifest.get(field)
        _require(expected is not None, f"Stage-A manifest omits binding {field}")
        _require(receipt.get(field) == expected,
                 f"Stage-A receipt binding {field} differs")
        if field in index_direct_bindings:
            _require(index.get(field) == expected,
                     f"Stage-A latent-index binding {field} differs")
    common_latent_bindings = {
        field: manifest[field] for field in binding_fields
    }
    for state_id, record in contexts.items():
        state = state_by_id[state_id]
        _require(record.get("state_id") == state_id
                 and record.get("state_identity_digest")
                 == state.get("state_identity_digest")
                 and all(record.get(field) == expected
                         for field, expected in common_latent_bindings.items())
                 and record.get("encoder_compute_dtype") == "float32"
                 and record.get("target_normalisation")
                 == STAGE_A_TARGET_NORMALISATION,
                 f"Stage-A context latent receipt binding differs: {state_id}")
    rows_by_digest = {_row_key(row): row for row in rows}
    for branch_digest, record in horizons.items():
        row = rows_by_digest[branch_digest]
        _require(record.get("branch_identity_digest") == branch_digest
                 and record.get("branch_row_digest") == row.get("branch_row_digest")
                 and record.get("branch_key")
                 == f"{row['state_id']}|{row['candidate']}"
                 and record.get("state_id") == row.get("state_id")
                 and record.get("candidate") == row.get("candidate")
                 and int(record.get("candidate_index", -1))
                 == int(row.get("candidate_index", -2))
                 and record.get("source_frame_set_digest")
                 == canonical_digest(row.get("horizon_frames"))
                 and record.get("source_frame_sha256")
                 == [frame["sha256"] for frame in row.get("horizon_frames", [])]
                 and all(record.get(field) == expected
                         for field, expected in common_latent_bindings.items())
                 and record.get("encoder_compute_dtype") == "float32"
                 and record.get("target_normalisation")
                 == STAGE_A_TARGET_NORMALISATION,
                 f"Stage-A horizon latent receipt binding differs: {branch_digest}")
    _require(index.get("target_normalisation") == STAGE_A_TARGET_NORMALISATION,
             "Stage-A latent index has another target normalisation")
    _require(index.get("encoder_compute_dtype") == "float32",
             "Stage-A latent index has another encoder compute dtype")
    _require(manifest.get("textured_v03_renderer_contract_digest")
             == textured_v03.renderer_contract_digest(),
             "Stage-A uses another textured-v03 renderer contract")
    # Stage A has its own prospective direct-fidelity assay specification.  It
    # is propagated as a source binding; the occupancy specification is frozen
    # here, after Stage A, and must never be substituted for it.
    return StageABundle(
        manifest=manifest, receipt=receipt, index=index, rows=tuple(rows),
        state_by_id=state_by_id, context_records=contexts,
        horizon_records=horizons, identity_digest=identity_digest,
        corpus_digest=str(corpus_digest), branch_rows_sha256=branch_rows_sha,
        latents_index_digest=latents_digest,
    )


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    _require(isinstance(value, (list, tuple)) and len(value) == length,
             f"{label} must contain {length} values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OccupancyAssayRefused(f"{label} contains a non-number") from exc
    _require(all(math.isfinite(item) for item in result),
             f"{label} contains a non-finite value")
    return result


def base_pose_values(pose: Mapping[str, Any]) -> tuple[
        tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the locked Stage-A WXYZ pose without silently guessing order."""

    _require(isinstance(pose, Mapping), "Stage-A base pose is not an object")
    position_raw = pose.get("position_world_xyz")
    if position_raw is None and isinstance(pose.get("position"), Mapping):
        position_raw = [pose["position"][axis] for axis in ("x", "y", "z")]
    quaternion_raw = pose.get("quaternion_world_wxyz",
                              pose.get("quat_world_wxyz"))
    _require(quaternion_raw is not None,
             "Stage-A base pose must explicitly name its WXYZ quaternion")
    position = _finite_vector(position_raw, 3, "base position world XYZ")
    wxyz = _finite_vector(quaternion_raw, 4, "base quaternion world WXYZ")
    norm = math.sqrt(sum(value * value for value in wxyz))
    _require(abs(norm - 1.0) <= dynamic_projection.QUATERNION_NORM_TOLERANCE,
             "Stage-A base quaternion is not unit length")
    return position, wxyz


def _quaternion_yaw_xyzw(xyzw: Sequence[float]) -> float:
    qx, qy, qz, qw = (float(value) for value in xyzw)
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _rotation_world_from_wxyz(wxyz: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = (float(value) for value in wxyz)
    return np.asarray((
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
         2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
         1 - 2 * (qx * qx + qy * qy)),
    ), dtype=np.float64)


def validate_recorded_camera(frame: Mapping[str, Any], position: Sequence[float],
                             wxyz: Sequence[float]) -> None:
    camera = frame.get("camera")
    _require(isinstance(camera, Mapping), "Stage-A horizon frame has no camera contract")
    _require(camera.get("fov_axis") == textured_v03.FOV_AXIS
             and float(camera.get("fov_deg", math.nan)) == textured_v03.FOV_DEG
             and float(camera.get("near_m", math.nan)) == textured_v03.NEAR_M
             and float(camera.get("far_m", math.nan)) == textured_v03.FAR_M
             and list(camera.get("resolution_wh", ()))
             == list(textured_v03.RESOLUTION_WH),
             "Stage-A horizon camera differs from exact textured-v03")
    recorded = frame.get("camera_pose_world")
    _require(isinstance(recorded, Mapping),
             "Stage-A horizon frame has no camera_pose_world")
    rotation = _rotation_world_from_wxyz(wxyz)
    expected_position = np.asarray(position) + rotation @ np.asarray(
        dynamic_projection.CAMERA_XYZ_BODY_M)
    expected_forward = rotation[:, 0]
    expected_up = rotation[:, 2]
    expected = {
        "position": expected_position,
        "lookat": expected_position + expected_forward,
        "up": expected_up,
    }
    for field, vector in expected.items():
        observed = np.asarray(_finite_vector(recorded.get(field), 3,
                                             f"camera_pose_world.{field}"))
        _require(np.allclose(observed, vector, rtol=0.0, atol=5e-5),
                 f"Stage-A recorded camera {field} disagrees with base pose")


def _scene_payload(row: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Path]:
    scene_dir = Path(str(row.get("scene_dir", ""))).resolve()
    _custody_safe(scene_dir)
    manifest_path = scene_dir / "manifest.json"
    _custody_safe(manifest_path)
    _require(manifest_path.is_file(), f"missing scene manifest {manifest_path}")
    expected_bytes = row.get("scene_manifest_byte_count",
                             row.get("scene_manifest_bytes"))
    _require(int(expected_bytes if expected_bytes is not None else -1)
             == manifest_path.stat().st_size,
             f"scene manifest byte count differs: {manifest_path}")
    _require(row.get("scene_manifest_sha256") == file_sha256(manifest_path),
             f"scene manifest byte digest differs: {manifest_path}")
    payload = read_json(manifest_path, "Stage-A scene manifest")
    parsed = parse_scene_manifest_dict(payload)
    _require(canonical_digest(payload)
             == row.get("scene_manifest_canonical_digest"),
             f"scene manifest canonical digest differs: {manifest_path}")
    if isinstance(payload.get("manifest_sha256"), str):
        _require(manifest_sha256(parsed) == payload["manifest_sha256"],
                 f"scene manifest embedded semantic digest differs: {manifest_path}")
    _require(textured_v03.raw_manifest_digest(payload)
             == row.get("raw_manifest_digest"),
             f"scene raw-manifest digest differs: {manifest_path}")
    _require(parsed.scene_id == row.get("scene_id")
             and parsed.family == row.get("family"),
             f"scene manifest identity differs: {manifest_path}")
    return payload, parsed, manifest_path


def _make_label_payload(task: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure worker-side four-horizon label generation for one Stage-A row."""

    row = task["row"]
    _payload, scene, scene_path = _scene_payload(row)
    # Exact renderer parity: branch RGB and branch physics omit distractors.
    raw_boxes = tuple((*scene.walls, *scene.obstacles, *scene.landmarks))
    distractor_count = (
        len(scene.visual_randomization.distractor_objects)
        if scene.visual_randomization is not None else 0
    )
    poses = row.get("horizon_base_poses")
    frames = row.get("horizon_frames")
    _require(isinstance(poses, list) and len(poses) == HORIZONS,
             "Stage-A row does not carry four horizon base poses")
    _require(isinstance(frames, list) and len(frames) == HORIZONS,
             "Stage-A row does not carry four horizon frame receipts")
    labels: list[np.ndarray] = []
    alignments: list[dict[str, Any]] = []
    for offset, (pose, frame) in enumerate(zip(poses, frames)):
        horizon = offset + 1
        _require(int(frame.get("horizon", -1)) == horizon,
                 f"Stage-A horizon frame index differs at H={horizon}")
        position, wxyz = base_pose_values(pose)
        validate_recorded_camera(frame, position, wxyz)
        xyzw = (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
        stored_yaw = _quaternion_yaw_xyzw(xyzw)
        camera = dynamic_projection.compose_yaw_aligned_camera(xyzw, stored_yaw)
        boxes = tuple(
            v4_builder._box_in_yaw_body(
                box, base_position_world=position, stored_yaw_rad=stored_yaw)
            for box in raw_boxes
        )
        frame_input = v4_builder.FrameBuildInputV4(
            frame_key={
                "branch_identity_digest": row["branch_identity_digest"],
                "horizon": horizon,
            },
            camera_origin_body_m=tuple(camera.origin_xyz),
            camera_basis_body_fru=v4_builder._normalized_camera_basis_fru(camera),
            ground_plane_z_body_m=-float(position[2]),
            rendered_boxes_body=boxes,
            image_path_metadata_only=str(frame.get("path", f"H{horizon}")),
            image_sha256=str(frame["sha256"]),
            sidecar_row_identity_sha256=str(row["branch_row_digest"]),
        )
        evidence = v4_builder.build_frame_evidence_v4(frame_input)
        # The reviewed builder loads the pure evidence implementation under a
        # neutral module name.  Use its paired rasterizer so the dataclass
        # identity and the exact H1 label implementation cannot diverge.
        raster = v4_builder.rasterize_observable_camera_ray_evidence_v4(evidence)
        array = np.ascontiguousarray(raster.output_labels, dtype=LABEL_DTYPE)
        _require(array.shape == (64, 64)
                 and set(np.unique(array)).issubset({0, 1, 2}),
                 f"malformed V4 raster at H={horizon}")
        labels.append(array)
        alignments.append({
            "horizon": horizon,
            "base_pose_digest": canonical_digest(pose),
            "camera_pose_world_digest": canonical_digest(frame["camera_pose_world"]),
            "horizon_frame_sha256": frame["sha256"],
            "source_branch_row_digest": row["branch_row_digest"],
            "evidence_content_sha256": evidence.content_sha256(),
            "raster_content_sha256": raster.content_sha256(),
            "rendered_box_count": len(raw_boxes),
            "excluded_distractor_count": distractor_count,
            "class_counts": {
                "unknown": int((array == 0).sum()),
                "free": int((array == 1).sum()),
                "occupied": int((array == 2).sum()),
            },
        })
    stacked = np.stack(labels, axis=0)
    provenance = {
        "scene_manifest_path": str(scene_path),
        "scene_manifest_sha256": row["scene_manifest_sha256"],
        "scene_manifest_canonical_digest": row["scene_manifest_canonical_digest"],
        "raw_manifest_digest": row["raw_manifest_digest"],
        "renderer_object_parity": label_contract()["renderer_object_parity"],
        "included_rendered_box_count": len(raw_boxes),
        "excluded_distractor_count": distractor_count,
        "horizons": alignments,
    }
    return stacked, provenance


def _label_paths(branch_digest: str) -> tuple[Path, Path]:
    return (LABEL_ROOT / "shards" / f"{branch_digest}.u1",
            LABEL_ROOT / "row_records" / f"{branch_digest}.json")


def _label_record_valid(record_path: Path, shard_path: Path,
                        row: Mapping[str, Any], bundle: StageABundle,
                        implementation: Mapping[str, str]) -> dict[str, Any]:
    record = read_json(record_path, "occupancy label record")
    _require(record.get("schema") == LABEL_RECORD_SCHEMA,
             "occupancy label-record schema differs")
    verify_self_digest(record, "label_record_digest", "occupancy label record")
    key = _row_key(row)
    _require(record.get("complete") is True
             and record.get("branch_identity_digest") == key
             and record.get("source_branch_row_digest") == row.get("branch_row_digest")
             and record.get("stage_a_identity_manifest_digest") == bundle.identity_digest
             and record.get("stage_a_corpus_digest") == bundle.corpus_digest
             and record.get("stage_a_latents_index_digest")
             == bundle.latents_index_digest
             and record.get("label_contract_digest") == LABEL_CONTRACT_DIGEST
             and record.get("implementation_bindings") == implementation,
             f"occupancy label record binding differs: {key}")
    _payload, scene, scene_path = _scene_payload(row)
    expected_boxes = len(scene.walls) + len(scene.obstacles) + len(scene.landmarks)
    expected_distractors = (
        len(scene.visual_randomization.distractor_objects)
        if scene.visual_randomization is not None else 0
    )
    provenance = record.get("provenance")
    horizon_provenance = provenance.get("horizons") \
        if isinstance(provenance, Mapping) else None
    _require(isinstance(provenance, Mapping)
             and provenance.get("scene_manifest_path") == str(scene_path)
             and provenance.get("scene_manifest_sha256")
             == row.get("scene_manifest_sha256")
             and provenance.get("scene_manifest_canonical_digest")
             == row.get("scene_manifest_canonical_digest")
             and provenance.get("raw_manifest_digest")
             == row.get("raw_manifest_digest")
             and provenance.get("renderer_object_parity")
             == label_contract()["renderer_object_parity"]
             and int(provenance.get("included_rendered_box_count", -1))
             == expected_boxes
             and int(provenance.get("excluded_distractor_count", -1))
             == expected_distractors
             and isinstance(horizon_provenance, list)
             and len(horizon_provenance) == HORIZONS,
             f"occupancy label provenance differs: {key}")
    poses = row.get("horizon_base_poses", [])
    frames = row.get("horizon_frames", [])
    _require(isinstance(poses, list) and len(poses) == HORIZONS
             and isinstance(frames, list) and len(frames) == HORIZONS,
             f"occupancy label source horizons differ: {key}")
    for horizon, item in enumerate(horizon_provenance, 1):
        _require(isinstance(item, Mapping)
                 and int(item.get("horizon", -1)) == horizon
                 and item.get("base_pose_digest")
                 == canonical_digest(poses[horizon - 1])
                 and item.get("camera_pose_world_digest")
                 == canonical_digest(frames[horizon - 1]["camera_pose_world"])
                 and item.get("horizon_frame_sha256")
                 == frames[horizon - 1]["sha256"]
                 and item.get("source_branch_row_digest")
                 == row.get("branch_row_digest")
                 and isinstance(item.get("evidence_content_sha256"), str)
                 and len(item["evidence_content_sha256"]) == 64
                 and isinstance(item.get("raster_content_sha256"), str)
                 and len(item["raster_content_sha256"]) == 64
                 and int(item.get("rendered_box_count", -1)) == expected_boxes
                 and int(item.get("excluded_distractor_count", -1))
                 == expected_distractors,
                 f"occupancy label H={horizon} alignment differs: {key}")
    _require(shard_path.is_file()
             and shard_path.stat().st_size == int(np.prod(LABEL_SHAPE))
             and record.get("label_sha256") == file_sha256(shard_path)
             and int(record.get("label_byte_count", -1)) == shard_path.stat().st_size
             and record.get("shape") == list(LABEL_SHAPE)
             and record.get("dtype") == "uint8",
             f"occupancy label shard differs: {key}")
    values = np.fromfile(shard_path, dtype=LABEL_DTYPE).reshape(LABEL_SHAPE)
    _require(set(np.unique(values)).issubset({0, 1, 2}),
             f"occupancy label shard contains an unknown class: {key}")
    for horizon, item in enumerate(horizon_provenance):
        expected_counts = {
            "unknown": int((values[horizon] == 0).sum()),
            "free": int((values[horizon] == 1).sum()),
            "occupied": int((values[horizon] == 2).sum()),
        }
        _require(item.get("class_counts") == expected_counts,
                 f"occupancy label H={horizon + 1} class counts differ: {key}")
    return record


def _write_label(task: Mapping[str, Any]) -> dict[str, Any]:
    row = task["row"]
    bundle_fields = task["bundle"]
    implementation = task["implementation"]
    values, provenance = _make_label_payload(task)
    key = _row_key(row)
    shard_path, record_path = _label_paths(key)
    digest, byte_count = atomic_bytes(shard_path, values.tobytes(order="C"))
    record: dict[str, Any] = {
        "schema": LABEL_RECORD_SCHEMA,
        "status": STATUS,
        "complete": True,
        "branch_identity_digest": key,
        "state_id": row["state_id"],
        "family": row["family"],
        "episode_cluster_id": row["episode_cluster_id"],
        "candidate": row["candidate"],
        "candidate_index": row["candidate_index"],
        "source_branch_row_digest": row["branch_row_digest"],
        "accepted_outcome_witness": row.get("accepted_outcome_witness"),
        "stage_a_identity_manifest_digest": bundle_fields["identity_digest"],
        "stage_a_corpus_digest": bundle_fields["corpus_digest"],
        "stage_a_branch_rows_sha256": bundle_fields["branch_rows_sha256"],
        "stage_a_latents_index_digest": bundle_fields["latents_index_digest"],
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "implementation_bindings": implementation,
        "shape": list(values.shape),
        "dtype": "uint8",
        "path": str(shard_path.relative_to(STAGE_ROOT)),
        "label_sha256": digest,
        "label_byte_count": byte_count,
        "provenance": provenance,
    }
    record["label_record_digest"] = canonical_digest(record)
    atomic_json(record_path, record)
    return record


def build_labels(bundle: StageABundle, workers: int) -> dict[str, Any]:
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    implementation = source_bindings()
    recovery: list[dict[str, Any]] = []
    receipt_path = LABEL_ROOT / "labels_receipt.json"
    prior_receipt: dict[str, Any] | None = None
    if receipt_path.is_file():
        try:
            prior_receipt = read_json(receipt_path, "existing occupancy label receipt")
            verify_self_digest(prior_receipt, "receipt_digest",
                               "existing occupancy label receipt")
            historical = prior_receipt.get("recovery", [])
            _require(isinstance(historical, list),
                     "existing occupancy label recovery is malformed")
            recovery.extend(historical)
        except (OccupancyAssayRefused, OSError, ValueError) as exc:
            preserve_invalid(receipt_path, str(exc), recovery)
            prior_receipt = None
    records: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    bundle_fields = {
        "identity_digest": bundle.identity_digest,
        "corpus_digest": bundle.corpus_digest,
        "branch_rows_sha256": bundle.branch_rows_sha256,
        "latents_index_digest": bundle.latents_index_digest,
    }
    for row in bundle.rows:
        key = _row_key(row)
        shard_path, record_path = _label_paths(key)
        if shard_path.exists() or record_path.exists():
            try:
                _require(shard_path.is_file() and record_path.is_file(),
                         "label shard/receipt pair is incomplete")
                records[key] = _label_record_valid(
                    record_path, shard_path, row, bundle, implementation)
                continue
            except (OccupancyAssayRefused, OSError, ValueError) as exc:
                preserve_invalid(record_path, str(exc), recovery)
                preserve_invalid(shard_path, str(exc), recovery)
        pending.append({
            "row": row, "bundle": bundle_fields, "implementation": implementation,
        })

    _require(not pending or file_sha256(
        ROOT / "scripts/run_go2_counterfactual_occupancy_assay_v1_2.py"
    ) == PRE_FIX_SOURCE_SHA256,
        "recovery source may retain verified pre-fix labels but may not generate "
        "or replace a label; restore the preserved exact generator for that item")
    if pending and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_write_label, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records[str(record["branch_identity_digest"])] = record
                print(f"[occupancy labels] {len(records)}/{EXPECTED_BRANCHES}",
                      flush=True)
    else:
        for task in pending:
            record = _write_label(task)
            records[str(record["branch_identity_digest"])] = record
            print(f"[occupancy labels] {len(records)}/{EXPECTED_BRANCHES}", flush=True)

    ordered = [records[_row_key(row)] for row in bundle.rows]
    _require(len(ordered) == EXPECTED_BRANCHES,
             "occupancy label record set is incomplete")
    index: dict[str, Any] = {
        "schema": LABEL_INDEX_SCHEMA,
        "status": STATUS,
        "complete": True,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_branch_rows_sha256": bundle.branch_rows_sha256,
        "stage_a_latents_index_digest": bundle.latents_index_digest,
        "label_contract": label_contract(),
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "implementation_bindings": implementation,
        "shape_per_branch": list(LABEL_SHAPE),
        "dtype": "uint8",
        "records": [{
            key: record[key] for key in (
                "branch_identity_digest", "state_id", "family",
                "episode_cluster_id", "candidate", "candidate_index",
                "source_branch_row_digest", "path", "shape", "dtype",
                "label_sha256", "label_byte_count", "label_record_digest",
            )
        } for record in ordered],
    }
    index["labels_index_digest"] = canonical_digest(index)
    index_path = LABEL_ROOT / "labels_index.json"
    if index_path.is_file():
        try:
            existing_index = read_json(index_path, "existing occupancy label index")
            verify_self_digest(existing_index, "labels_index_digest",
                               "existing occupancy label index")
            _require(existing_index == index,
                     "existing occupancy label index differs")
        except (OccupancyAssayRefused, OSError, ValueError) as exc:
            preserve_invalid(index_path, str(exc), recovery)
            atomic_json(index_path, index)
    else:
        atomic_json(index_path, index)
    aggregate = {
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_latents_index_digest": bundle.latents_index_digest,
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "labels_index_digest": index["labels_index_digest"],
        "ordered_label_record_digests": [
            record["label_record_digest"] for record in ordered],
        "ordered_label_sha256": [record["label_sha256"] for record in ordered],
        "record_count": len(ordered),
        "complete": True,
    }
    receipt: dict[str, Any] = {
        "schema": LABEL_RECEIPT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "state_count": EXPECTED_STATES,
        "branch_count": EXPECTED_BRANCHES,
        "horizon_label_count": EXPECTED_BRANCHES * HORIZONS,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_branch_rows_sha256": bundle.branch_rows_sha256,
        "stage_a_latents_index_digest": bundle.latents_index_digest,
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "labels_index_digest": index["labels_index_digest"],
        "labels_index_sha256": file_sha256(index_path),
        "labels_corpus_digest_payload": aggregate,
        "labels_corpus_digest": canonical_digest(aggregate),
        "recovery": recovery,
        "implementation_bindings": implementation,
        "storage_bytes": sum(record["label_byte_count"] for record in ordered),
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    if prior_receipt is not None and receipt_path.is_file():
        if prior_receipt == receipt:
            return prior_receipt
        preserve_invalid(
            receipt_path,
            "existing complete occupancy label receipt differs from reconstruction",
            recovery,
        )
        receipt["recovery"] = recovery
        receipt["receipt_digest"] = canonical_digest(
            without(receipt, "receipt_digest"))
    atomic_json(receipt_path, receipt)
    return receipt


@dataclass(frozen=True)
class LabelBundle:
    index: dict[str, Any]
    receipt: dict[str, Any]
    records: Mapping[str, dict[str, Any]]
    shards: Mapping[str, Path]
    corpus_digest: str
    index_digest: str


def load_labels(bundle: StageABundle) -> LabelBundle:
    index_path = LABEL_ROOT / "labels_index.json"
    receipt_path = LABEL_ROOT / "labels_receipt.json"
    index = read_json(index_path, "occupancy label index")
    _require(index.get("schema") == LABEL_INDEX_SCHEMA
             and index.get("complete") is True,
             "occupancy label index is not complete v1.2")
    index_digest = verify_self_digest(
        index, "labels_index_digest", "occupancy label index")
    receipt = read_json(receipt_path, "occupancy label receipt")
    _require(receipt.get("schema") == LABEL_RECEIPT_SCHEMA
             and receipt.get("complete") is True,
             "occupancy label receipt is not complete v1.2")
    verify_self_digest(receipt, "receipt_digest", "occupancy label receipt")
    for payload, label in ((index, "label index"), (receipt, "label receipt")):
        for field, expected in (
            ("stage_a_identity_manifest_digest", bundle.identity_digest),
            ("stage_a_corpus_digest", bundle.corpus_digest),
            ("stage_a_branch_rows_sha256", bundle.branch_rows_sha256),
            ("stage_a_latents_index_digest", bundle.latents_index_digest),
            ("label_contract_digest", LABEL_CONTRACT_DIGEST),
            ("assay_spec_digest", ASSAY_SPEC_DIGEST),
        ):
            _require(payload.get(field) == expected,
                     f"occupancy {label} {field} differs")
    implementation = source_bindings()
    _require(index.get("label_contract") == label_contract()
             and index.get("implementation_bindings") == implementation
             and receipt.get("implementation_bindings") == implementation,
             "occupancy label contract or implementation binding differs")
    _require(int(receipt.get("state_count", -1)) == EXPECTED_STATES
             and int(receipt.get("branch_count", -1)) == EXPECTED_BRANCHES
             and int(receipt.get("horizon_label_count", -1))
             == EXPECTED_BRANCHES * HORIZONS,
             "occupancy label receipt counts differ")
    _require(receipt.get("labels_index_digest") == index_digest
             and receipt.get("labels_index_sha256") == file_sha256(index_path),
             "occupancy label receipt binds another index")
    corpus_payload = receipt.get("labels_corpus_digest_payload")
    corpus_digest = receipt.get("labels_corpus_digest")
    _require(isinstance(corpus_payload, dict)
             and canonical_digest(corpus_payload) == corpus_digest,
             "occupancy label corpus digest is not reproducible")
    listed = index.get("records")
    _require(isinstance(listed, list) and len(listed) == EXPECTED_BRANCHES,
             "occupancy label index does not contain 240 records")
    rows_by_key = {_row_key(row): row for row in bundle.rows}
    records: dict[str, dict[str, Any]] = {}
    shards: dict[str, Path] = {}
    for entry in listed:
        _require(isinstance(entry, dict), "occupancy label index record is malformed")
        key = str(entry.get("branch_identity_digest", ""))
        _require(key in rows_by_key and key not in records,
                 f"occupancy label index identity differs: {key}")
        shard_path, record_path = _label_paths(key)
        record = _label_record_valid(
            record_path, shard_path, rows_by_key[key], bundle, implementation)
        for field in (
            "state_id", "family", "episode_cluster_id", "candidate",
            "candidate_index", "source_branch_row_digest", "path", "shape",
            "dtype", "label_sha256", "label_byte_count", "label_record_digest",
        ):
            _require(entry.get(field) == record.get(field),
                     f"occupancy label index/record {field} differs: {key}")
        records[key] = record
        shards[key] = shard_path
    _require(set(records) == set(rows_by_key),
             "occupancy label identity set differs from Stage A")
    ordered = [records[_row_key(row)] for row in bundle.rows]
    expected_aggregate = {
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_latents_index_digest": bundle.latents_index_digest,
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "labels_index_digest": index_digest,
        "ordered_label_record_digests": [
            record["label_record_digest"] for record in ordered],
        "ordered_label_sha256": [record["label_sha256"] for record in ordered],
        "record_count": len(ordered),
        "complete": True,
    }
    _require(corpus_payload == expected_aggregate
             and corpus_digest == canonical_digest(expected_aggregate)
             and int(receipt.get("storage_bytes", -1))
             == sum(record["label_byte_count"] for record in ordered),
             "occupancy label aggregate receipt differs from its records")
    return LabelBundle(
        index=index, receipt=receipt, records=records, shards=shards,
        corpus_digest=str(corpus_digest), index_digest=index_digest,
    )


def validate_probe_package_metadata() -> dict[str, Any]:
    """Validate the frozen package before (but do not yet load) its weights."""

    _custody_safe(PROBE_PACKAGE_PATH)
    _custody_safe(PROBE_WEIGHTS_PATH)
    _require(PROBE_PACKAGE_PATH.is_file() and PROBE_WEIGHTS_PATH.is_file(),
             "frozen occupancy probe package or weights are missing")
    package_file_sha = file_sha256(PROBE_PACKAGE_PATH)
    weights_sha = file_sha256(PROBE_WEIGHTS_PATH)
    _require(package_file_sha == PROBE_PACKAGE_FILE_SHA256,
             "frozen occupancy probe package bytes differ")
    _require(weights_sha == PROBE_WEIGHTS_SHA256,
             "frozen occupancy probe weights differ")
    package = read_json(PROBE_PACKAGE_PATH, "frozen occupancy probe package")
    verify_self_digest(package, "package_digest", "frozen occupancy probe package")
    _require(package.get("package_digest") == PROBE_PACKAGE_DIGEST,
             "frozen occupancy probe package digest differs")
    _require(package.get("probe_weights_sha256") == weights_sha,
             "frozen occupancy probe package binds other weights")
    _require(package.get("specification_digest") == PROBE_SPECIFICATION_DIGEST,
             "frozen occupancy probe specification differs")
    probe = package.get("probe")
    qualification = package.get("qualification")
    _require(isinstance(probe, Mapping)
             and probe.get("architecture")
             == "SharedTokenToBev(1024) -> 64x64 x 3"
             and int(probe.get("epochs", -1)) == 12
             and probe.get("epoch_taken") == "final"
             and probe.get("best_epoch_selection") is False,
             "frozen occupancy probe is not its fixed final-epoch architecture")
    _require(package.get("fit_rows") == 3137
             and package.get("calibration_rows") == 785
             and package.get("selection_rows_inspected_during_fit_or_qualification") == 0,
             "frozen occupancy probe fit/calibration provenance differs")
    _require(isinstance(qualification, Mapping)
             and qualification.get("qualified") is True
             and float(qualification.get("observed_iou", -1.0))
             == 0.4991258441602962
             and float(qualification.get("observed_iou", -1.0))
             >= QUALIFICATION_IOU_FLOOR,
             "frozen occupancy probe did not pass its original qualification")
    return {
        "path": str(PROBE_PACKAGE_PATH),
        "package_digest": PROBE_PACKAGE_DIGEST,
        "package_file_sha256": package_file_sha,
        "weights_path": str(PROBE_WEIGHTS_PATH),
        "weights_sha256": weights_sha,
        "weights_bytes": PROBE_WEIGHTS_PATH.stat().st_size,
        "specification_digest": PROBE_SPECIFICATION_DIGEST,
        "original_qualification": qualification,
    }


def tensor_state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(canonical_bytes(list(value.shape)) + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_probe(device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load exactly the already-qualified weights.  There is no optimiser."""

    state = torch.load(PROBE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping), "frozen occupancy probe weights are malformed")
    model = dense_probe.SharedTokenToBev(TOKEN_DIM).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise OccupancyAssayRefused(
            f"frozen occupancy probe architecture/weights mismatch: {exc}") from exc
    model.eval()
    return model, tensor_state_digest(model)


def occupied_counts(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.uint8)
    truth = np.asarray(truth, dtype=np.uint8)
    _require(prediction.shape == truth.shape,
             "occupancy prediction/target shapes differ")
    observable = truth != ray_v4.UNKNOWN_CLASS
    actual = truth[observable] == ray_v4.OCCUPIED_CLASS
    chosen = prediction[observable] == ray_v4.OCCUPIED_CLASS
    intersection = int(np.logical_and(actual, chosen).sum())
    union = int(np.logical_or(actual, chosen).sum())
    return {
        "observable_cells": int(observable.sum()),
        "occupied_support": int(actual.sum()),
        "occupied_predicted": int(chosen.sum()),
        "occupied_intersection": intersection,
        "occupied_union": union,
        "observable_occupied_iou": intersection / union if union else None,
    }


def _validate_occupied_count_record(record: Mapping[str, Any], label: str) -> None:
    names = (
        "observable_cells", "occupied_support", "occupied_predicted",
        "occupied_intersection", "occupied_union",
    )
    _require(all(isinstance(record.get(name), int)
                 and not isinstance(record.get(name), bool)
                 and int(record[name]) >= 0 for name in names),
             f"{label} has malformed occupied counts")
    observable = int(record["observable_cells"])
    support = int(record["occupied_support"])
    predicted = int(record["occupied_predicted"])
    intersection = int(record["occupied_intersection"])
    union = int(record["occupied_union"])
    _require(support <= observable and predicted <= observable
             and intersection <= min(support, predicted)
             and union == support + predicted - intersection,
             f"{label} occupied counts are internally inconsistent")
    observed = record.get("observable_occupied_iou")
    expected = intersection / union if union else None
    _require((observed is None and expected is None)
             or (isinstance(observed, (int, float))
                 and math.isfinite(float(observed))
                 and float(observed) == expected),
             f"{label} occupied IoU differs from its counts")


def pooled_iou(records: Sequence[Mapping[str, Any]]) -> float | None:
    intersection = sum(int(record["occupied_intersection"]) for record in records)
    union = sum(int(record["occupied_union"]) for record in records)
    return float(intersection / union) if union else None


def episode_then_family(values: Sequence[float | None], rows: Sequence[Mapping[str, Any]]) \
        -> dict[str, Any]:
    _require(len(values) == len(rows), "occupancy estimator row lengths differ")
    by_cluster: dict[str, list[float]] = collections.defaultdict(list)
    cluster_family: dict[str, str] = {}
    for value, row in zip(values, rows):
        cluster = str(row["episode_cluster_id"])
        family = str(row["family"])
        if cluster in cluster_family:
            _require(cluster_family[cluster] == family,
                     "one episode cluster crosses occupancy families")
        cluster_family[cluster] = family
        if value is not None and math.isfinite(float(value)):
            by_cluster[cluster].append(float(value))
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    for cluster, cluster_values in by_cluster.items():
        if cluster_values:
            by_family[cluster_family[cluster]].append(float(np.mean(cluster_values)))
    per_family = {
        family: (float(np.mean(by_family[family]))
                 if by_family[family] else None)
        for family in FAMILIES
    }
    missing = [family for family in FAMILIES if per_family[family] is None]
    finite_values = [float(value) for value in values
                     if value is not None and math.isfinite(float(value))]
    total_clusters_per_family = collections.Counter(cluster_family.values())
    return {
        "equal_family": (
            float(np.mean([per_family[family] for family in FAMILIES]))
            if not missing else None
        ),
        "equal_family_available": not missing,
        "equal_family_defined_family_count": len(FAMILIES) - len(missing),
        "equal_family_required_family_count": len(FAMILIES),
        "equal_family_missing_families": missing,
        "equal_family_unavailable_reason": (
            None if not missing else
            "at least one frozen family has no defined occupied-IoU rows; "
            "seven-family averaging and imputation are forbidden"
        ),
        "corpus_weighted": (
            float(np.mean(finite_values)) if finite_values else None
        ),
        "per_family": per_family,
        "per_family_available": {
            family: per_family[family] is not None for family in FAMILIES
        },
        "per_family_defined_cluster_count": {
            family: len(by_family[family]) for family in FAMILIES
        },
        "defined_rows": len(finite_values),
        "undefined_rows": len(values) - len(finite_values),
        "episode_clusters": len(cluster_family),
        "clusters_per_family": {
            family: int(total_clusters_per_family[family]) for family in FAMILIES
        },
    }


def _load_label_array(labels: LabelBundle, branch_digest: str) -> np.ndarray:
    return np.asarray(np.memmap(
        labels.shards[branch_digest], mode="r", dtype=LABEL_DTYPE,
        shape=LABEL_SHAPE), dtype=np.uint8)


def _load_horizon_latent(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(np.memmap(
        Path(str(record["_resolved_path"])), mode="r", dtype=np.float16,
        shape=(HORIZONS, TOKENS, TOKEN_DIM)), dtype=np.float32)


def _true_record_paths(branch_digest: str) -> Path:
    return RESULT_ROOT / "true_target_records" / f"{branch_digest}.json"


def _true_record_valid(path: Path, row: Mapping[str, Any], bundle: StageABundle,
                       labels: LabelBundle, probe_state_digest: str,
                       device: torch.device) -> dict[str, Any]:
    record = read_json(path, "true-target occupancy record")
    verify_self_digest(record, "true_target_record_digest",
                       "true-target occupancy record")
    _require(record.get("schema") == "go2_counterfactual_occupancy_true_record_v1_2"
             and record.get("complete") is True
             and record.get("branch_identity_digest") == _row_key(row)
             and record.get("state_id") == row.get("state_id")
             and record.get("family") == row.get("family")
             and record.get("episode_cluster_id") == row.get("episode_cluster_id")
             and record.get("candidate") == row.get("candidate")
             and int(record.get("candidate_index", -1))
             == int(row.get("candidate_index", -2))
             and record.get("source_branch_row_digest") == row.get("branch_row_digest")
             and record.get("stage_a_identity_manifest_digest")
             == bundle.identity_digest
             and record.get("stage_a_latents_index_digest")
             == bundle.latents_index_digest
             and record.get("horizon_latent_sha256")
             == bundle.horizon_records[_row_key(row)]["sha256"]
             and record.get("labels_corpus_digest") == labels.corpus_digest
             and record.get("label_sha256")
             == labels.records[_row_key(row)]["label_sha256"]
             and record.get("probe_package_digest") == PROBE_PACKAGE_DIGEST
             and record.get("probe_weights_sha256") == PROBE_WEIGHTS_SHA256
             and record.get("probe_state_digest") == probe_state_digest
             and record.get("probe_inference_device") == str(device)
             and record.get("label_contract_digest") == LABEL_CONTRACT_DIGEST
             and record.get("assay_spec_digest") == ASSAY_SPEC_DIGEST,
             f"true-target occupancy record binding differs: {_row_key(row)}")
    horizons = record.get("horizons")
    _require(isinstance(horizons, list) and len(horizons) == HORIZONS
             and [int(item.get("horizon", -1)) for item in horizons]
             == [1, 2, 3, 4],
             "true-target occupancy record horizon set differs")
    for item in horizons:
        _validate_occupied_count_record(item, "true-target occupancy record")
    return record


@torch.no_grad()
def score_true_targets(bundle: StageABundle, labels: LabelBundle,
                       probe: torch.nn.Module, probe_state_digest: str,
                       device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    recovery: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in bundle.rows:
        key = _row_key(row)
        path = _true_record_paths(key)
        if path.exists():
            try:
                completed[key] = _true_record_valid(
                    path, row, bundle, labels, probe_state_digest, device)
                continue
            except (OccupancyAssayRefused, OSError, ValueError) as exc:
                preserve_invalid(path, str(exc), recovery)
        pending.append(row)

    for start in range(0, len(pending), batch_size):
        selected = pending[start:start + batch_size]
        latent = np.stack([
            _load_horizon_latent(bundle.horizon_records[_row_key(row)])
            for row in selected
        ], axis=0)
        target_labels = np.stack([
            _load_label_array(labels, _row_key(row)) for row in selected
        ], axis=0)
        predictions: list[np.ndarray] = []
        for horizon in range(HORIZONS):
            tokens = torch.from_numpy(latent[:, horizon]).to(device, torch.float32)
            tokens = F.layer_norm(tokens, (TOKEN_DIM,))
            logits = probe(tokens, TOKEN_GRID)
            predictions.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
        for offset, row in enumerate(selected):
            key = _row_key(row)
            horizons = []
            for horizon in range(HORIZONS):
                counts = occupied_counts(
                    predictions[horizon][offset], target_labels[offset, horizon])
                horizons.append({"horizon": horizon + 1, **counts})
            record: dict[str, Any] = {
                "schema": "go2_counterfactual_occupancy_true_record_v1_2",
                "status": STATUS,
                "complete": True,
                "branch_identity_digest": key,
                "state_id": row["state_id"],
                "family": row["family"],
                "episode_cluster_id": row["episode_cluster_id"],
                "candidate": row["candidate"],
                "candidate_index": row["candidate_index"],
                "source_branch_row_digest": row["branch_row_digest"],
                "stage_a_identity_manifest_digest": bundle.identity_digest,
                "stage_a_latents_index_digest": bundle.latents_index_digest,
                "horizon_latent_sha256": bundle.horizon_records[key]["sha256"],
                "labels_corpus_digest": labels.corpus_digest,
                "label_sha256": labels.records[key]["label_sha256"],
                "probe_package_digest": PROBE_PACKAGE_DIGEST,
                "probe_weights_sha256": PROBE_WEIGHTS_SHA256,
                "probe_state_digest": probe_state_digest,
                "probe_inference_device": str(device),
                "label_contract_digest": LABEL_CONTRACT_DIGEST,
                "assay_spec_digest": ASSAY_SPEC_DIGEST,
                "horizons": horizons,
            }
            record["true_target_record_digest"] = canonical_digest(record)
            atomic_json(_true_record_paths(key), record)
            completed[key] = record
        print(f"[occupancy true targets] {len(completed)}/{EXPECTED_BRANCHES}",
              flush=True)
    ordered = [completed[_row_key(row)] for row in bundle.rows]
    _require(len(ordered) == EXPECTED_BRANCHES,
             "true-target occupancy record set is incomplete")
    if recovery:
        recovery_path = RESULT_ROOT / "true_target_recovery.json"
        payload = {"records": recovery, "assay_spec_digest": ASSAY_SPEC_DIGEST}
        payload["recovery_digest"] = canonical_digest(payload)
        atomic_json(recovery_path, payload)
    return ordered


def _metrics_for_horizon(records: Sequence[Mapping[str, Any]], horizon: int,
                         rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [record["horizons"][horizon - 1]["observable_occupied_iou"]
              for record in records]
    aggregate = episode_then_family(values, rows)
    pooled_records = [record["horizons"][horizon - 1] for record in records]
    aggregate["whole_pilot_observable_occupied_iou"] = pooled_iou(pooled_records)
    aggregate["pooled_counts"] = {
        name: sum(int(record[name]) for record in pooled_records)
        for name in (
            "observable_cells", "occupied_support", "occupied_predicted",
            "occupied_intersection", "occupied_union",
        )
    }
    return aggregate


def freeze_true_target_gate(bundle: StageABundle, labels: LabelBundle,
                            records: Sequence[Mapping[str, Any]],
                            probe_provenance: Mapping[str, Any],
                            probe_state_digest: str,
                            device: torch.device) -> dict[str, Any]:
    equivalence = source_equivalence_receipt()
    horizon_reports: dict[str, Any] = {}
    for horizon in range(1, HORIZONS + 1):
        metrics = _metrics_for_horizon(records, horizon, bundle.rows)
        observed = metrics["whole_pilot_observable_occupied_iou"]
        qualified = observed is not None and float(observed) >= QUALIFICATION_IOU_FLOOR
        horizon_reports[str(horizon)] = {
            "horizon": horizon,
            "qualification_metric": "whole_pilot_observable_occupied_iou",
            "qualification_floor": QUALIFICATION_IOU_FLOOR,
            "observed_iou": observed,
            "qualified": qualified,
            "availability": "available" if qualified else "unavailable",
            "metrics": metrics,
        }
    gate: dict[str, Any] = {
        "schema": TRUE_GATE_SCHEMA,
        "status": STATUS,
        "complete": True,
        "frozen_before_predicted_latent_access": True,
        "predictor_checkpoints_opened_by_stage_d": 0,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "stage_a_corpus_digest": bundle.corpus_digest,
        "stage_a_latents_index_digest": bundle.latents_index_digest,
        "labels_corpus_digest": labels.corpus_digest,
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "probe": dict(probe_provenance),
        "probe_state_digest": probe_state_digest,
        "probe_inference_device": str(device),
        "assay_spec": assay_spec(),
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "source_equivalence_receipt_digest":
            equivalence["source_equivalence_receipt_digest"],
        "source_recovery_scope": equivalence["recovery_scope"],
        "true_target_record_digests": [
            record["true_target_record_digest"] for record in records],
        "horizons": horizon_reports,
        "qualified_horizons": [
            horizon for horizon in range(1, HORIZONS + 1)
            if horizon_reports[str(horizon)]["qualified"]
        ],
        "unavailable_horizons": [
            horizon for horizon in range(1, HORIZONS + 1)
            if not horizon_reports[str(horizon)]["qualified"]
        ],
    }
    gate["true_target_gate_digest"] = canonical_digest(gate)
    path = RESULT_ROOT / "true_target_gate.json"
    if path.exists():
        try:
            existing = read_json(path, "existing true-target occupancy gate")
            verify_self_digest(existing, "true_target_gate_digest",
                               "existing true-target occupancy gate")
            _require(existing == gate,
                     "existing true-target occupancy gate differs from recomputation")
            return existing
        except (OccupancyAssayRefused, OSError, ValueError) as exc:
            recovery: list[dict[str, Any]] = []
            preserve_invalid(path, str(exc), recovery)
            recovery_payload = {"records": recovery}
            recovery_payload["recovery_digest"] = canonical_digest(recovery_payload)
            atomic_json(RESULT_ROOT / "gate_recovery.json", recovery_payload)
    atomic_json(path, gate)
    return gate


@dataclass(frozen=True)
class PredictionIndex:
    seed_index: int
    seed: int
    cell: str
    checkpoint_sha256: str
    index: dict[str, Any]
    index_digest: str
    branch_records: Mapping[str, dict[str, Any]]
    state_shards: Mapping[str, dict[str, Any]]
    state_paths: Mapping[str, Path]
    checkpoint_receipt: dict[str, Any]


def _sequence_digest(values: Sequence[Any]) -> str:
    return canonical_digest(list(values))


def _predictor_index_dir(seed: int, cell: str) -> Path:
    return (STAGE_ROOT / "predictor_assay" / "prediction_ledgers"
            / f"seed_{seed}_{cell}")


def validate_stage_bc_assay_spec(bundle: StageABundle) -> dict[str, Any]:
    path = STAGE_ROOT / "predictor_assay" / "assay_spec.json"
    spec = read_json(path, "Stage-B/C prospective assay specification")
    digest = verify_self_digest(
        spec, "assay_spec_digest", "Stage-B/C prospective assay specification")
    _require(spec.get("schema") == "go2_counterfactual_predictor_assay_spec_v1_2"
             and spec.get("prospective") is True
             and spec.get("created_before_target_latent_scoring") is True
             and spec.get("utility_scorer_used") is False,
             "Stage-B/C assay specification is not prospective direct inference")
    for field, expected in (
        ("stage_a_identity_manifest_digest", bundle.identity_digest),
        ("stage_a_corpus_digest", bundle.corpus_digest),
        ("stage_a_branch_rows_sha256", bundle.branch_rows_sha256),
        ("stage_a_latents_index_digest", bundle.latents_index_digest),
        ("scientific_run_package_digest", FROZEN_RUN_PACKAGE_DIGEST),
        ("confirmatory_commit", FROZEN_CONFIRMATORY_COMMIT),
    ):
        _require(spec.get(field) == expected,
                 f"Stage-B/C assay specification {field} differs")
    _require(spec.get("seeds") == list(SEEDS)
             and spec.get("cells") == list(CELLS)
             and int(spec.get("checkpoint_epoch", -1)) == 21,
             "Stage-B/C assay seed/cell/checkpoint scope differs")
    normalisation = spec.get("normalisation")
    _require(isinstance(normalisation, Mapping)
             and normalisation.get("context_and_targets")
             == STAGE_A_TARGET_NORMALISATION
             and normalisation.get("predictions")
             == STAGE_BC_PREDICTION_NORMALISATION,
             "Stage-B/C assay latent-normalisation contract differs")
    source = spec.get("source_bindings")
    _require(isinstance(source, Mapping) and source,
             "Stage-B/C assay specification has no source bindings")
    for relative, observed in source.items():
        relative_path = Path(str(relative))
        candidate = (ROOT / relative_path).resolve()
        _custody_safe(candidate)
        _require(not relative_path.is_absolute()
                 and ROOT.resolve() in candidate.parents
                 and candidate.is_file()
                 and file_sha256(candidate) == observed,
                 f"Stage-B/C assay implementation changed: {relative}")
    return {**spec, "_verified_path": str(path), "_verified_digest": digest}


def _validate_prediction_index(seed_index: int, seed: int, cell: str,
                               bundle: StageABundle) -> PredictionIndex:
    """Validate already-produced B/C shards; never open a checkpoint."""

    directory = _predictor_index_dir(seed, cell)
    index_path = directory / "predictions_index.json"
    index = read_json(index_path, f"prediction index {seed}/{cell}")
    _require(index.get("schema")
             == "go2_counterfactual_predictor_predictions_index_v1_2"
             and index.get("complete") is True
             and index.get("utility_scorer_used") is False,
             f"prediction index {seed}/{cell} is not complete direct inference")
    index_digest = verify_self_digest(
        index, "predictions_index_digest", f"prediction index {seed}/{cell}")
    _require(int(index.get("seed_index", -1)) == seed_index
             and int(index.get("seed", -1)) == seed
             and index.get("cell") == cell
             and int(index.get("checkpoint_epoch", -1)) == 21,
             f"prediction index {seed}/{cell} checkpoint identity differs")
    checkpoint_sha = index.get("checkpoint_sha256")
    _require(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64,
             f"prediction index {seed}/{cell} has no checkpoint SHA-256")
    _require(index.get("stage_a_identity_manifest_digest") == bundle.identity_digest
             and index.get("stage_a_corpus_digest") == bundle.corpus_digest
             and index.get("stage_a_latents_index_digest")
             == bundle.latents_index_digest,
             f"prediction index {seed}/{cell} binds another Stage A")
    _require(index.get("scientific_run_package_digest")
             == FROZEN_RUN_PACKAGE_DIGEST
             and index.get("confirmatory_commit") == FROZEN_CONFIRMATORY_COMMIT,
             f"prediction index {seed}/{cell} lineage differs")
    _require(int(index.get("states", -1)) == EXPECTED_STATES
             and int(index.get("branches", -1)) == EXPECTED_BRANCHES,
             f"prediction index {seed}/{cell} scope differs")
    _require(index.get("prediction_representation")
             == STAGE_BC_PREDICTION_REPRESENTATION,
             f"prediction index {seed}/{cell} representation differs")
    source = index.get("source_bindings")
    _require(isinstance(source, Mapping) and source,
             f"prediction index {seed}/{cell} has no source bindings")
    for relative, digest in source.items():
        _require(isinstance(relative, str) and isinstance(digest, str),
                 f"prediction index {seed}/{cell} source binding is malformed")
        relative_path = Path(relative)
        _require(not relative_path.is_absolute(),
                 f"prediction source binding is not repository-relative: {relative}")
        path = (ROOT / relative_path).resolve()
        _custody_safe(path)
        _require(ROOT.resolve() in path.parents and path.is_file()
                 and file_sha256(path) == digest,
                 f"prediction source changed after B/C inference: {relative}")

    expected_rows = {_row_key(row): row for row in bundle.rows}
    branch_list = index.get("branch_records")
    _require(isinstance(branch_list, list) and len(branch_list) == EXPECTED_BRANCHES,
             f"prediction index {seed}/{cell} branch records are incomplete")
    branches: dict[str, dict[str, Any]] = {}
    ordered_digests: list[str] = []
    for position, record in enumerate(branch_list):
        _require(isinstance(record, dict),
                 f"prediction index {seed}/{cell} branch record is malformed")
        key = str(record.get("branch_identity_digest", ""))
        _require(key in expected_rows and key not in branches,
                 f"prediction index {seed}/{cell} branch identity differs: {key}")
        _require(key == _row_key(bundle.rows[position]),
                 f"prediction index {seed}/{cell} branch order differs at {position}")
        row = expected_rows[key]
        _require(int(record.get("position", -1)) == position
                 and int(record.get("seed_index", -1)) == seed_index
                 and int(record.get("seed", -1)) == seed
                 and record.get("cell") == cell
                 and record.get("checkpoint_sha256") == checkpoint_sha
                 and record.get("state_id") == row.get("state_id")
                 and record.get("family") == row.get("family")
                 and record.get("candidate") == row.get("candidate")
                 and int(record.get("candidate_index", -1))
                 == int(row.get("candidate_index", -2))
                 and record.get("branch_shape") == [HORIZONS, TOKENS, TOKEN_DIM]
                 and record.get("state_shard_shape")
                 == [EXPECTED_CANDIDATES, HORIZONS, TOKENS, TOKEN_DIM]
                 and record.get("branch_slice")
                 == [int(row["candidate_index"]), 0, 0, 0]
                 and record.get("dtype") == "float16",
                 f"prediction index {seed}/{cell} branch binding differs: {key}")
        branches[key] = record
        ordered_digests.append(key)
    _require(set(branches) == set(expected_rows),
             f"prediction index {seed}/{cell} branch identity set differs")
    _require(index.get("ordered_branch_identity_set_digest")
             == _sequence_digest(ordered_digests),
             f"prediction index {seed}/{cell} ordered branch digest differs")

    shard_list = index.get("state_shards")
    _require(isinstance(shard_list, list) and len(shard_list) == EXPECTED_STATES,
             f"prediction index {seed}/{cell} state shards are incomplete")
    shards: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    predictor_result_root = STAGE_ROOT / "predictor_assay"
    expected_bytes = EXPECTED_CANDIDATES * HORIZONS * TOKENS * TOKEN_DIM * 2
    for shard in shard_list:
        _require(isinstance(shard, dict),
                 f"prediction index {seed}/{cell} state shard is malformed")
        verify_self_digest(shard, "receipt_digest",
                           f"prediction shard receipt {seed}/{cell}")
        state_id = str(shard.get("state_id", ""))
        _require(state_id in bundle.state_by_id and state_id not in shards,
                 f"prediction shard {seed}/{cell} state identity differs: {state_id}")
        expected_candidates = [
            str(row["candidate"]) for row in bundle.rows
            if str(row["state_id"]) == state_id
        ]
        expected_candidates = [name for _, name in sorted(zip(
            [int(row["candidate_index"]) for row in bundle.rows
             if str(row["state_id"]) == state_id], expected_candidates))]
        _require(shard.get("schema")
                 == "go2_counterfactual_prediction_state_shard_receipt_v1_2"
                 and shard.get("complete") is True
                 and shard.get("checkpoint_sha256") == checkpoint_sha
                 and int(shard.get("seed", -1)) == seed
                 and shard.get("cell") == cell
                 and shard.get("candidate_names") == expected_candidates
                 and shard.get("shape")
                 == [EXPECTED_CANDIDATES, HORIZONS, TOKENS, TOKEN_DIM]
                 and shard.get("dtype") == "float16"
                 and int(shard.get("byte_count", -1)) == expected_bytes,
                 f"prediction shard receipt {seed}/{cell}/{state_id} differs")
        path = _safe_under(
            predictor_result_root, shard.get("relative_path"),
            f"prediction shard {seed}/{cell}/{state_id}")
        _require(path.is_file() and path.stat().st_size == expected_bytes
                 and shard.get("sha256") == file_sha256(path),
                 f"prediction shard bytes differ {seed}/{cell}/{state_id}")
        for key, record in branches.items():
            if record["state_id"] != state_id:
                continue
            _require(record.get("relative_path") == shard.get("relative_path")
                     and record.get("sha256") == shard.get("sha256")
                     and int(record.get("byte_count", -1)) == expected_bytes,
                     f"prediction branch/shard binding differs: {seed}/{cell}/{key}")
        shards[state_id] = shard
        paths[state_id] = path
    _require(set(shards) == set(bundle.state_by_id),
             f"prediction index {seed}/{cell} state identity set differs")

    receipt_path = (STAGE_ROOT / "predictor_assay" / "prediction_ledgers"
                    / f"seed_{seed}_{cell}.receipt.json")
    receipt = read_json(receipt_path, f"checkpoint prediction receipt {seed}/{cell}")
    verify_self_digest(receipt, "receipt_digest",
                       f"checkpoint prediction receipt {seed}/{cell}")
    _require(receipt.get("schema")
             == "go2_counterfactual_predictor_checkpoint_receipt_v1_2"
             and receipt.get("complete") is True
             and receipt.get("checkpoint_sha256") == checkpoint_sha
             and int(receipt.get("seed_index", -1)) == seed_index
             and int(receipt.get("seed", -1)) == seed
             and receipt.get("cell") == cell
             and int(receipt.get("states_completed", -1)) == EXPECTED_STATES
             and receipt.get("predictions_index_digest") == index_digest,
             f"checkpoint prediction receipt {seed}/{cell} differs")
    return PredictionIndex(
        seed_index=seed_index, seed=seed, cell=cell,
        checkpoint_sha256=str(checkpoint_sha), index=index,
        index_digest=index_digest, branch_records=branches,
        state_shards=shards, state_paths=paths, checkpoint_receipt=receipt,
    )


def load_prediction_indices(bundle: StageABundle) -> list[PredictionIndex]:
    """Called only after the true-target gate is durably frozen."""

    bc_spec = validate_stage_bc_assay_spec(bundle)
    indices = [
        _validate_prediction_index(seed_index, seed, cell, bundle)
        for seed_index, seed in enumerate(SEEDS) for cell in CELLS
    ]
    _require(len(indices) == 32
             and len({(item.seed, item.cell) for item in indices}) == 32,
             "prediction index inventory is not the frozen 32 checkpoints")
    _require(all(item.index.get("assay_spec_digest")
                 == bc_spec["_verified_digest"] for item in indices),
             "one or more prediction indices bind another B/C assay specification")
    return indices


def _checkpoint_result_dir(index: PredictionIndex) -> Path:
    return RESULT_ROOT / "checkpoints" / f"seed_{index.seed}_{index.cell}"


def _predicted_state_record_valid(path: Path, index: PredictionIndex,
                                  state_id: str, bundle: StageABundle,
                                  labels: LabelBundle,
                                  qualified_horizons: Sequence[int],
                                  probe_state_digest: str,
                                  device: torch.device) -> dict[str, Any]:
    record = read_json(path, "predicted occupancy state record")
    verify_self_digest(record, "predicted_state_record_digest",
                       "predicted occupancy state record")
    _require(record.get("schema") == SCORE_RECORD_SCHEMA
             and record.get("complete") is True
             and int(record.get("seed_index", -1)) == index.seed_index
             and record.get("seed") == index.seed
             and record.get("cell") == index.cell
             and record.get("checkpoint_sha256") == index.checkpoint_sha256
             and record.get("predictions_index_digest") == index.index_digest
             and record.get("state_id") == state_id
             and record.get("prediction_shard_sha256")
             == index.state_shards[state_id]["sha256"]
             and int(record.get("state_index", -1))
             == int(bundle.state_by_id[state_id]["state_index"])
             and record.get("family") == bundle.state_by_id[state_id]["family"]
             and record.get("episode_cluster_id")
             == bundle.state_by_id[state_id]["episode_cluster_id"]
             and record.get("stage_a_identity_manifest_digest")
             == bundle.identity_digest
             and record.get("labels_corpus_digest") == labels.corpus_digest
             and record.get("probe_package_digest") == PROBE_PACKAGE_DIGEST
             and record.get("probe_weights_sha256") == PROBE_WEIGHTS_SHA256
             and record.get("probe_state_digest") == probe_state_digest
             and record.get("probe_inference_device") == str(device)
             and record.get("label_contract_digest") == LABEL_CONTRACT_DIGEST
             and record.get("assay_spec_digest") == ASSAY_SPEC_DIGEST
             and record.get("qualified_horizons") == list(qualified_horizons),
             f"predicted occupancy state record binding differs: {index.seed}/"
             f"{index.cell}/{state_id}")
    branches = record.get("branches")
    _require(isinstance(branches, list) and len(branches) == EXPECTED_CANDIDATES,
             "predicted occupancy state record lacks twelve branches")
    expected_rows = sorted(
        (row for row in bundle.rows if str(row["state_id"]) == state_id),
        key=lambda row: int(row["candidate_index"]),
    )
    for branch, row in zip(branches, expected_rows):
        key = _row_key(row)
        _require(branch.get("branch_identity_digest") == key
                 and branch.get("candidate") == row.get("candidate")
                 and int(branch.get("candidate_index", -1))
                 == int(row.get("candidate_index", -2))
                 and branch.get("label_sha256") == labels.records[key]["label_sha256"]
                 and [int(item.get("horizon", -1))
                      for item in branch.get("horizons", [])]
                 == list(qualified_horizons),
                 f"predicted occupancy branch binding differs: {index.seed}/"
                 f"{index.cell}/{key}")
        for item in branch["horizons"]:
            _validate_occupied_count_record(
                item, f"predicted occupancy {index.seed}/{index.cell}/{key}")
    return record


@torch.no_grad()
def score_prediction_index(index: PredictionIndex, bundle: StageABundle,
                           labels: LabelBundle, probe: torch.nn.Module,
                           probe_state_digest: str, device: torch.device,
                           qualified_horizons: Sequence[int]) -> tuple[
                               list[dict[str, Any]], dict[str, Any]]:
    result_dir = _checkpoint_result_dir(index)
    state_dir = result_dir / "state_records"
    recovery: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    rows_by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in bundle.rows:
        rows_by_state[str(row["state_id"])].append(row)
    for state_id in rows_by_state:
        rows_by_state[state_id].sort(key=lambda row: int(row["candidate_index"]))

    for state_id in sorted(rows_by_state,
                           key=lambda value: int(rows_by_state[value][0]["state_index"])):
        path = state_dir / f"{hashlib.sha256(state_id.encode()).hexdigest()}.json"
        if path.exists():
            try:
                completed[state_id] = _predicted_state_record_valid(
                    path, index, state_id, bundle, labels,
                    qualified_horizons, probe_state_digest, device)
                continue
            except (OccupancyAssayRefused, OSError, ValueError) as exc:
                preserve_invalid(path, str(exc), recovery)

        shape = (EXPECTED_CANDIDATES, HORIZONS, TOKENS, TOKEN_DIM)
        prediction = np.memmap(index.state_paths[state_id], mode="r",
                               dtype=np.float16, shape=shape)
        state_rows = rows_by_state[state_id]
        target = np.stack([
            _load_label_array(labels, _row_key(row)) for row in state_rows
        ], axis=0)
        per_horizon_predictions: dict[int, np.ndarray] = {}
        for horizon in qualified_horizons:
            tokens = torch.from_numpy(
                np.asarray(prediction[:, horizon - 1], dtype=np.float32)).to(device)
            per_horizon_predictions[horizon] = (
                probe(tokens, TOKEN_GRID).argmax(1).cpu().numpy().astype(np.uint8)
            )
        branches: list[dict[str, Any]] = []
        for offset, row in enumerate(state_rows):
            branch_horizons = []
            for horizon in qualified_horizons:
                branch_horizons.append({
                    "horizon": horizon,
                    **occupied_counts(
                        per_horizon_predictions[horizon][offset],
                        target[offset, horizon - 1]),
                })
            branches.append({
                "branch_identity_digest": _row_key(row),
                "candidate": row["candidate"],
                "candidate_index": row["candidate_index"],
                "label_sha256": labels.records[_row_key(row)]["label_sha256"],
                "horizons": branch_horizons,
            })
        del prediction
        record: dict[str, Any] = {
            "schema": SCORE_RECORD_SCHEMA,
            "status": STATUS,
            "complete": True,
            "seed_index": index.seed_index,
            "seed": index.seed,
            "cell": index.cell,
            "checkpoint_sha256": index.checkpoint_sha256,
            "predictions_index_digest": index.index_digest,
            "state_id": state_id,
            "state_index": state_rows[0]["state_index"],
            "family": state_rows[0]["family"],
            "episode_cluster_id": state_rows[0]["episode_cluster_id"],
            "prediction_shard_sha256": index.state_shards[state_id]["sha256"],
            "stage_a_identity_manifest_digest": bundle.identity_digest,
            "labels_corpus_digest": labels.corpus_digest,
            "probe_package_digest": PROBE_PACKAGE_DIGEST,
            "probe_weights_sha256": PROBE_WEIGHTS_SHA256,
            "probe_state_digest": probe_state_digest,
            "probe_inference_device": str(device),
            "label_contract_digest": LABEL_CONTRACT_DIGEST,
            "assay_spec_digest": ASSAY_SPEC_DIGEST,
            "qualified_horizons": list(qualified_horizons),
            "unqualified_horizons_scored": [],
            "branches": branches,
        }
        record["predicted_state_record_digest"] = canonical_digest(record)
        atomic_json(path, record)
        completed[state_id] = record
        print(f"[occupancy predicted] seed {index.seed} {index.cell}: "
              f"{len(completed)}/{EXPECTED_STATES}", flush=True)

    ordered_states = sorted(completed.values(), key=lambda item: int(item["state_index"]))
    flat: list[dict[str, Any]] = []
    for state in ordered_states:
        for branch in state["branches"]:
            flat.append({
                "state_id": state["state_id"],
                "state_index": state["state_index"],
                "family": state["family"],
                "episode_cluster_id": state["episode_cluster_id"],
                **branch,
            })
    _require(len(flat) == EXPECTED_BRANCHES,
             "predicted occupancy checkpoint result is incomplete")
    receipt: dict[str, Any] = {
        "schema": SCORE_RECEIPT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "seed_index": index.seed_index,
        "seed": index.seed,
        "cell": index.cell,
        "checkpoint_sha256": index.checkpoint_sha256,
        "predictions_index_digest": index.index_digest,
        "stage_a_identity_manifest_digest": bundle.identity_digest,
        "labels_corpus_digest": labels.corpus_digest,
        "probe_package_digest": PROBE_PACKAGE_DIGEST,
        "probe_weights_sha256": PROBE_WEIGHTS_SHA256,
        "probe_state_digest": probe_state_digest,
        "probe_inference_device": str(device),
        "label_contract_digest": LABEL_CONTRACT_DIGEST,
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "qualified_horizons": list(qualified_horizons),
        "states_completed": len(ordered_states),
        "branches_completed": len(flat),
        "state_record_digest_set": _sequence_digest([
            item["predicted_state_record_digest"] for item in ordered_states]),
        "recovery": recovery,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    receipt_path = result_dir / "receipt.json"
    if receipt_path.exists():
        try:
            existing = read_json(receipt_path, "predicted occupancy checkpoint receipt")
            verify_self_digest(existing, "receipt_digest",
                               "predicted occupancy checkpoint receipt")
            _require(without(existing, "receipt_digest", "recovery")
                     == without(receipt, "receipt_digest", "recovery"),
                     "existing predicted occupancy checkpoint receipt differs")
            return flat, existing
        except (OccupancyAssayRefused, OSError, ValueError) as exc:
            preserve_invalid(receipt_path, str(exc), recovery)
            receipt["recovery"] = recovery
            receipt["receipt_digest"] = canonical_digest(
                without(receipt, "receipt_digest"))
            atomic_json(receipt_path, receipt)
    else:
        atomic_json(receipt_path, receipt)
    return flat, receipt


def t_summary(values: Sequence[float]) -> dict[str, Any]:
    _require(len(values) == FROZEN_SEED_COUNT,
             "occupancy paired inference requires exactly eight seed values")
    array = np.asarray(values, dtype=np.float64)
    _require(bool(np.isfinite(array).all()),
             "occupancy seed vector contains a non-finite value")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = 2.3646242510102993  # exact t_(0.975, 7)
    half = critical * sd / math.sqrt(FROZEN_SEED_COUNT)
    return {
        "values": [float(value) for value in array],
        "n": FROZEN_SEED_COUNT,
        "mean": mean,
        "sample_standard_deviation": sd,
        "t_critical_df7": critical,
        "two_sided_95_t_interval": [mean - half, mean + half],
    }


def _horizon_entry(record: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    matches = [item for item in record.get("horizons", [])
               if int(item.get("horizon", -1)) == horizon]
    _require(len(matches) == 1,
             f"occupancy result does not contain exactly one H={horizon} entry")
    return matches[0]


def aggregate_prediction(records: Sequence[Mapping[str, Any]], horizon: int,
                         bundle: StageABundle) -> dict[str, Any]:
    by_key = {str(record["branch_identity_digest"]): record for record in records}
    _require(len(by_key) == EXPECTED_BRANCHES,
             "predicted occupancy aggregation lacks 240 branches")
    ordered = [by_key[_row_key(row)] for row in bundle.rows]
    entries = [_horizon_entry(record, horizon) for record in ordered]
    aggregate = episode_then_family(
        [entry["observable_occupied_iou"] for entry in entries], bundle.rows)
    aggregate["whole_pilot_observable_occupied_iou"] = pooled_iou(entries)
    aggregate["pooled_counts"] = {
        name: sum(int(entry[name]) for entry in entries)
        for name in (
            "observable_cells", "occupied_support", "occupied_predicted",
            "occupied_intersection", "occupied_union",
        )
    }
    return aggregate


def _cell_effects(values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    _require(set(values) == set(CELLS)
             and all(len(values[cell]) == FROZEN_SEED_COUNT for cell in CELLS),
             "occupancy cell/seed matrix differs")
    rgb = [float(values["rgb_rollout"][index])
           - float(values["rgb_one_step"][index])
           for index in range(FROZEN_SEED_COUNT)]
    prop = [float(values["proprio_rollout"][index])
            - float(values["proprio_one_step"][index])
            for index in range(FROZEN_SEED_COUNT)]
    main = [(rgb[index] + prop[index]) / 2.0
            for index in range(FROZEN_SEED_COUNT)]
    interaction = [prop[index] - rgb[index]
                   for index in range(FROZEN_SEED_COUNT)]
    return {
        "direction": "positive means rollout improves observable occupied IoU",
        "B_RGB": t_summary(rgb),
        "B_prop": t_summary(prop),
        "M_main_rollout": t_summary(main),
        "J_proprioception_by_rollout": t_summary(interaction),
    }


def _seed_summary_or_unavailable(values: Sequence[float | None],
                                 reason: str) -> dict[str, Any]:
    serialized = [
        float(value) if value is not None and math.isfinite(float(value)) else None
        for value in values
    ]
    defined = sum(value is not None for value in serialized)
    if len(serialized) == FROZEN_SEED_COUNT and defined == FROZEN_SEED_COUNT:
        return {"available": True, **t_summary([float(x) for x in serialized])}
    return {
        "available": False,
        "values": serialized,
        "defined_seeds": defined,
        "required_seeds": FROZEN_SEED_COUNT,
        "reason": reason,
    }


def _analysis_for_estimator(
        target: float | None,
        by_cell_seed: Mapping[str, Sequence[float | None]],
        unavailable_reason: str | None = None) -> dict[str, Any]:
    target_available = target is not None and math.isfinite(float(target))
    result = {
        "true_target": float(target) if target_available else None,
        "true_target_available": target_available,
        "unavailable_reason": unavailable_reason if not target_available else None,
        "four_cells": {},
    }
    complete_values: dict[str, list[float]] = {}
    for cell in CELLS:
        predicted_summary = _seed_summary_or_unavailable(
            by_cell_seed[cell],
            "one or more seed estimates are undefined; no imputation permitted",
        )
        if predicted_summary["available"]:
            complete_values[cell] = [float(value) for value in by_cell_seed[cell]]
        if target_available and predicted_summary["available"]:
            gaps: Sequence[float | None] = [
                float(target) - value for value in complete_values[cell]
            ]
            gap_summary = _seed_summary_or_unavailable(gaps, "")
        else:
            gap_summary = {
                "available": False,
                "values": [None] * FROZEN_SEED_COUNT,
                "defined_seeds": 0,
                "required_seeds": FROZEN_SEED_COUNT,
                "reason": (
                    unavailable_reason
                    if not target_available else predicted_summary["reason"]
                ),
            }
        result["four_cells"][cell] = {
            "predicted": predicted_summary,
            "true_target_minus_predicted_gap": gap_summary,
        }
    if set(complete_values) == set(CELLS):
        result["rollout_effects"] = {
            "available": True, **_cell_effects(complete_values),
        }
    else:
        result["rollout_effects"] = {
            "available": False,
            "reason": (
                "at least one cell/seed estimate is undefined; paired rollout "
                "effects and interactions are unavailable without imputation"
            ),
            "available_cells": sorted(complete_values),
            "required_cells": list(CELLS),
        }
    return result


def analyse_occupancy(bundle: StageABundle, gate: Mapping[str, Any],
                      checkpoint_metrics: Mapping[
                          tuple[int, str], Mapping[int, Mapping[str, Any]]],
                      prediction_indices: Sequence[PredictionIndex],
                      prediction_receipts: Sequence[Mapping[str, Any]],
                      probe_provenance: Mapping[str, Any],
                      probe_state_digest: str,
                      labels: LabelBundle,
                      runtime: Mapping[str, float]) -> dict[str, Any]:
    equivalence = source_equivalence_receipt()
    _require(gate.get("source_equivalence_receipt_digest")
             == equivalence["source_equivalence_receipt_digest"],
             "true-target gate binds another recovery source")
    horizons: dict[str, Any] = {}
    qualified = set(int(value) for value in gate["qualified_horizons"])
    for horizon in range(1, HORIZONS + 1):
        target = gate["horizons"][str(horizon)]["metrics"]
        if horizon not in qualified:
            horizons[str(horizon)] = {
                "horizon": horizon,
                "available": False,
                "reason": (
                    "true-target frozen probe observable occupied IoU failed "
                    "the independently fixed 0.35 floor"
                ),
                "true_target_gate": gate["horizons"][str(horizon)],
                "predictor_latents_scored": False,
            }
            continue
        matrix: dict[str, list[Mapping[str, Any]]] = {
            cell: [checkpoint_metrics[(seed, cell)][horizon] for seed in SEEDS]
            for cell in CELLS
        }
        estimator_reports: dict[str, Any] = {}
        for estimator in (
            "equal_family", "corpus_weighted",
            "whole_pilot_observable_occupied_iou",
        ):
            values = {
                cell: [item[estimator] for item in matrix[cell]]
                for cell in CELLS
            }
            reason = (
                target.get("equal_family_unavailable_reason")
                if estimator == "equal_family" else
                f"true-target H={horizon} {estimator} is undefined"
            )
            estimator_reports[estimator] = _analysis_for_estimator(
                target[estimator], values, reason)

        per_family: dict[str, Any] = {}
        for family in FAMILIES:
            values = {
                cell: [item["per_family"][family] for item in matrix[cell]]
                for cell in CELLS
            }
            per_family[family] = _analysis_for_estimator(
                target["per_family"][family], values,
                (f"family {family} has no defined true-target occupied-IoU "
                 "rows; no imputation permitted"),
            )
            if family == "local_composite_motifs":
                per_family[family]["interpretation"] = "diagnostic only"
        horizons[str(horizon)] = {
            "horizon": horizon,
            "available": True,
            "true_target_gate": gate["horizons"][str(horizon)],
            "predictor_latents_scored": True,
            "primary_equal_family": estimator_reports["equal_family"],
            "secondary_corpus_weighted": estimator_reports["corpus_weighted"],
            "whole_pilot_pooled_diagnostic": estimator_reports[
                "whole_pilot_observable_occupied_iou"],
            "per_family": per_family,
        }

    checkpoints = [{
        "seed_index": item.seed_index,
        "seed": item.seed,
        "cell": item.cell,
        "checkpoint_epoch": 21,
        "checkpoint_sha256": item.checkpoint_sha256,
        "predictions_index_digest": item.index_digest,
        "source_checkpoint_opened_by_stage_d": False,
    } for item in prediction_indices]
    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "claim_bearing": False,
        "complete": True,
        "assay": "pilot-branch H=1..4 frozen occupancy-probe portability",
        "explicitly_not": [
            "planning utility", "a spatial-head refit", "new spatial training",
            "a reinterpretation or rerun of the prior H=1 occupancy assay",
        ],
        "sequential_order": {
            "labels_and_true_target_latents_complete_before_probe_load": True,
            "true_target_gate_frozen_before_predicted_latent_access": True,
            "predictor_checkpoints_opened_by_stage_d": 0,
            "predicted_latents_consumed_from_stage_bc": True,
            "unqualified_horizons_scored": [],
        },
        "stage_a": {
            "identity_manifest_digest": bundle.identity_digest,
            "corpus_digest": bundle.corpus_digest,
            "branch_rows_sha256": bundle.branch_rows_sha256,
            "latents_index_digest": bundle.latents_index_digest,
            "source_assay_spec_digest": bundle.manifest["assay_spec_digest"],
        },
        "labels": {
            "label_contract": label_contract(),
            "label_contract_digest": LABEL_CONTRACT_DIGEST,
            "labels_index_digest": labels.index_digest,
            "labels_corpus_digest": labels.corpus_digest,
            "renderer_object_parity": label_contract()["renderer_object_parity"],
            "excluded_distractor_counts_by_branch": {
                key: int(record["provenance"]["excluded_distractor_count"])
                for key, record in labels.records.items()
                if int(record["provenance"]["excluded_distractor_count"]) > 0
            },
        },
        "probe": {
            **dict(probe_provenance),
            "loaded_state_digest": probe_state_digest,
            "inference_device": gate["probe_inference_device"],
            "refit": False,
            "modified": False,
        },
        "assay_spec": assay_spec(),
        "assay_spec_digest": ASSAY_SPEC_DIGEST,
        "source_equivalence": {
            "receipt_digest":
                equivalence["source_equivalence_receipt_digest"],
            "pre_fix_source_sha256": equivalence["pre_fix_source_sha256"],
            "active_source_sha256": equivalence["active_source_sha256"],
            "unified_diff_sha256": equivalence["unified_diff_sha256"],
            "protected_scientific_ast_digest":
                equivalence["protected_scientific_ast_digest"],
            "changed_existing_functions":
                equivalence["changed_existing_functions"],
            "label_generation_changed": False,
            "probe_inference_changed": False,
            "true_target_record_generation_changed": False,
        },
        "true_target_gate_digest": gate["true_target_gate_digest"],
        "qualified_horizons": sorted(qualified),
        "unavailable_horizons": gate["unavailable_horizons"],
        "checkpoints": checkpoints,
        "prediction_receipt_digests": [
            receipt["receipt_digest"] for receipt in prediction_receipts],
        "horizons": horizons,
        "runtime_seconds": {key: round(float(value), 3)
                            for key, value in runtime.items()},
        "storage": {
            "label_shards_bytes": sum(path.stat().st_size
                                       for path in labels.shards.values()),
            "source_prediction_shards_bytes_read_only": sum(
                path.stat().st_size for item in prediction_indices
                for path in item.state_paths.values()),
            "scope": "explicit digest-bound Stage-D and B/C files only",
        },
    }
    report["report_digest"] = canonical_digest(report)
    return report


def _write_or_reuse_report(report: Mapping[str, Any]) -> dict[str, Any]:
    path = RESULT_ROOT / "result.json"
    if path.exists():
        existing = read_json(path, "existing occupancy result")
        verify_self_digest(existing, "report_digest", "existing occupancy result")
        _require(without(existing, "report_digest", "runtime_seconds")
                 == without(report, "report_digest", "runtime_seconds"),
                 "existing complete occupancy result differs from recomputation")
        return existing
    atomic_json(path, report)
    return dict(report)


def run_assay(bundle: StageABundle, labels: LabelBundle,
              device: torch.device, batch_size: int) -> dict[str, Any]:
    total_started = time.time()
    # Metadata and every source true-latent/label byte were validated above.
    # Only now may the immutable probe weights be opened.
    probe_provenance = validate_probe_package_metadata()
    probe, probe_state = load_probe(device)
    true_started = time.time()
    true_records = score_true_targets(
        bundle, labels, probe, probe_state, device, batch_size)
    _require(tensor_state_digest(probe) == probe_state,
             "frozen occupancy probe changed during true-target scoring")
    gate = freeze_true_target_gate(
        bundle, labels, true_records, probe_provenance, probe_state, device)
    true_seconds = time.time() - true_started

    qualified = [int(value) for value in gate["qualified_horizons"]]
    checkpoint_metrics: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    indices: list[PredictionIndex] = []
    receipts: list[dict[str, Any]] = []
    prediction_seconds = 0.0
    if qualified:
        # This is the first predicted-latent access in Stage D.  The B/C shard
        # validator never loads (and this script cannot load) predictor weights.
        indices = load_prediction_indices(bundle)
        prediction_started = time.time()
        for index in indices:
            records, receipt = score_prediction_index(
                index, bundle, labels, probe, probe_state, device, qualified)
            checkpoint_metrics[(index.seed, index.cell)] = {
                horizon: aggregate_prediction(records, horizon, bundle)
                for horizon in qualified
            }
            receipts.append(receipt)
        prediction_seconds = time.time() - prediction_started
    _require(tensor_state_digest(probe) == probe_state,
             "frozen occupancy probe changed during predicted-latent scoring")
    report = analyse_occupancy(
        bundle, gate, checkpoint_metrics, indices, receipts,
        probe_provenance, probe_state, labels,
        {
            "true_target_and_gate": true_seconds,
            "predicted_latent_application": prediction_seconds,
            "total": time.time() - total_started,
        },
    )
    return _write_or_reuse_report(report)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        _require(torch.cuda.is_available(), "--device cuda requested but unavailable")
    return torch.device(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("labels", "assay", "all"), default="all")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, choices=(FROZEN_SCORE_BATCH,),
                        default=FROZEN_SCORE_BATCH)
    args = parser.parse_args()
    _require(args.workers > 0, "--workers must be positive")
    bundle = load_stage_a()
    if args.stage in ("labels", "all"):
        build_labels(bundle, args.workers)
    if args.stage == "labels":
        receipt = read_json(LABEL_ROOT / "labels_receipt.json",
                            "occupancy label receipt")
        print(json.dumps({
            "labels_complete": receipt["complete"],
            "labels_corpus_digest": receipt["labels_corpus_digest"],
            "label_contract_digest": LABEL_CONTRACT_DIGEST,
        }, indent=2))
        return 0
    labels = load_labels(bundle)
    report = run_assay(bundle, labels, resolve_device(args.device), args.batch)
    print(json.dumps({
        "complete": report["complete"],
        "report_digest": report["report_digest"],
        "probe_package_digest": PROBE_PACKAGE_DIGEST,
        "qualified_horizons": report["qualified_horizons"],
        "unavailable_horizons": report["unavailable_horizons"],
        "result": str(RESULT_ROOT / "result.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
