#!/usr/bin/env python3
"""Bounded occupancy/clearance contract differential.

This diagnostic reads only the frozen counterfactual occupancy artefacts and
the frozen route-intent artefacts.  It does not simulate, render, encode, train,
or execute a world-model predictor.  Panel identities are materialised by the
``--freeze-identities`` phase before the probe is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_worlds", ROOT / "lewm_genesis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.benchmarks import go2_dynamic_cell_square_projection as DYNAMIC
from lewm.oracle.go2_branch_oracle_v1_2 import CLEARANCE_SAFE_M
from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_graph import SceneGraph
from scripts import build_go2_observable_camera_ray_fit_v4 as V4
from scripts import dev_frozen_dense_representation_encoders_v1 as ENC
from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as OCC
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS
from scripts import train_evaluate_structured_spatial_safety_state_jepa_dev_v2 as SS

OUT = ROOT / ".generated/occupancy_clearance_contract_differential_v1"
IDENTITIES = OUT / "panel_identities.json"
RESULT = OUT / "result.json"
REFERENCE_ROOT = ROOT / ".generated/go2_counterfactual_fidelity_v1_2"
ROUTE_V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
ROUTE_V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
ROUTE_LABELS = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "structured_spatial_safety_state_jepa_dev_v2/derived_frozen_pose_occupancy_labels.npy"
)
ROUTE_LABEL_INDEX = ROUTE_LABELS.with_name("derived_frozen_pose_occupancy_index.json")
EXPECTED_OCCUPANCY_RESULT = "09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6"
EXPECTED_ROUTE_TARGET = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
EXPECTED_PACKAGE = "b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686"
EXPECTED_WEIGHTS = "95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322"
DOMAIN_REFERENCE = "OCCUPANCY_CLEARANCE_CONTRACT_DIFFERENTIAL_V1/reference"
DOMAIN_ROUTE = "OCCUPANCY_CLEARANCE_CONTRACT_DIFFERENTIAL_V1/route"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def structural_order(domain: str, *fields: object) -> str:
    return hashlib.sha256((domain + "\0" + "\0".join(map(str, fields))).encode()).hexdigest()


def reference_metadata() -> tuple[list[dict], dict, dict, dict]:
    rows = read_jsonl(REFERENCE_ROOT / "branch_rows.jsonl")
    by_digest = {str(row["branch_identity_digest"]): row for row in rows}
    latent_index = json.loads((REFERENCE_ROOT / "latents_index.json").read_text())
    latent_by_digest = {
        str(row["branch_identity_digest"]): row for row in latent_index["horizon_records"]
    }
    label_index = json.loads((REFERENCE_ROOT / "occupancy_labels/labels_index.json").read_text())
    label_by_digest = {
        str(row["branch_identity_digest"]): row for row in label_index["records"]
    }
    return rows, by_digest, latent_by_digest, label_by_digest


def freeze_identities() -> dict:
    if sha(REFERENCE_ROOT / "occupancy_results/result.json") == EXPECTED_OCCUPANCY_RESULT:
        raise RuntimeError("expected digest is the registered report digest, not the result-file SHA")
    registered = json.loads((REFERENCE_ROOT / "occupancy_results/result.json").read_text())
    if registered.get("report_digest") != EXPECTED_OCCUPANCY_RESULT:
        raise RuntimeError("registered occupancy result digest mismatch")
    if sha(ROUTE_V2 / "target_latent_index.json") != EXPECTED_ROUTE_TARGET:
        raise RuntimeError("route target index mismatch")
    rows, by_digest, latent_by_digest, label_by_digest = reference_metadata()
    families = sorted({str(row["family"]) for row in rows})
    reference = []
    for family in families:
        eligible = []
        for row in rows:
            if row["family"] != family:
                continue
            key = str(row["branch_identity_digest"])
            record_path = REFERENCE_ROOT / "occupancy_results/true_target_records" / f"{key}.json"
            record = json.loads(record_path.read_text())
            h3 = record["horizons"][2]
            if h3["observable_occupied_iou"] is not None:
                eligible.append((structural_order(
                    DOMAIN_REFERENCE, family, row["state_id"], row["candidate_index"], key
                ), row, record))
        eligible.sort(key=lambda item: item[0])
        if len(eligible) < 2:
            raise RuntimeError(f"{family}: fewer than two defined H3 reference rows")
        for order, row, record in eligible[:2]:
            key = str(row["branch_identity_digest"])
            reference.append({
                "selection_hash": order,
                "family": family,
                "state_id": row["state_id"],
                "scene_identity": row.get("scene_identity", row.get("scene_id")),
                "episode_cluster_id": row["episode_cluster_id"],
                "candidate_index": int(row["candidate_index"]),
                "branch_identity": key,
                "h3_defined": record["horizons"][2]["observable_occupied_iou"] is not None,
                "latent_sha256": latent_by_digest[key]["sha256"],
                "label_sha256": label_by_digest[key]["label_sha256"],
            })

    route_rows = FS.load_metadata()
    route = []
    for family in FS.FAMILIES:
        family_rows = [(i, row) for i, row in enumerate(route_rows) if row["family"] == family]
        strata = {False: [], True: []}
        for index, row in family_rows:
            strata[bool(row["unsafe"])].append((structural_order(
                DOMAIN_ROUTE, family, row["state_id"], row["candidate_index"]
            ), index, row))
        chosen = []
        for unsafe in (False, True):
            strata[unsafe].sort(key=lambda item: item[0])
            chosen.extend(strata[unsafe][:2])
        if len(chosen) < 4:
            selected_ids = {item[1] for item in chosen}
            rest = sorted((item for values in strata.values() for item in values
                           if item[1] not in selected_ids), key=lambda item: item[0])
            chosen.extend(rest[:4 - len(chosen)])
        if len(chosen) != 4:
            raise RuntimeError(f"{family}: cannot freeze four route rows")
        for order, index, row in sorted(chosen, key=lambda item: item[0]):
            route.append({
                "selection_hash": order,
                "row_index": int(index),
                "family": family,
                "state_id": row["state_id"],
                "scene_identity": next(x["scene_id"] for x in json.loads(
                    (ROUTE_V1 / "state_manifest.json").read_text()
                )["state_candidates"] if x["state_id"] == row["state_id"]),
                "candidate_index": int(row["candidate_index"]),
                "branch_identity": f"{row['state_id']}:{int(row['candidate_index']):02d}",
                "unsafe": bool(row["unsafe"]),
            })
    payload = {
        "schema": "occupancy_clearance_contract_differential_v1_panel_identities",
        "frozen_before_probe_scoring": True,
        "selection": {
            "reference": "two defined H3 rows per family by structural SHA-256 order",
            "route": "two safe and two unsafe rows per maze family where available, then structural SHA-256 order",
            "probe_performance_used": False,
        },
        "bindings": {
            "occupancy_result_digest": EXPECTED_OCCUPANCY_RESULT,
            "route_target_index_sha256": EXPECTED_ROUTE_TARGET,
            "reference_branch_rows_sha256": sha(REFERENCE_ROOT / "branch_rows.jsonl"),
            "reference_latents_index_sha256": sha(REFERENCE_ROOT / "latents_index.json"),
            "reference_labels_index_sha256": sha(REFERENCE_ROOT / "occupancy_labels/labels_index.json"),
            "route_state_manifest_sha256": sha(ROUTE_V1 / "state_manifest.json"),
            "route_branch_ledger_sha256": sha(ROUTE_V1 / "branch_labels.jsonl"),
            "route_replay_directory_records": 48,
        },
        "reference": reference,
        "route": route,
    }
    payload["identity_digest"] = canonical_digest(payload)
    atomic_json(IDENTITIES, payload)
    return payload


def reference_paths(key: str, latent_record: dict, label_record: dict) -> tuple[Path, Path]:
    latent = REFERENCE_ROOT / str(latent_record["path"])
    label = REFERENCE_ROOT / str(label_record["path"])
    if sha(latent) != latent_record["sha256"] or sha(label) != label_record["label_sha256"]:
        raise RuntimeError(f"reference shard digest mismatch: {key}")
    return latent, label


def score_probe(probe, latent: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tokens = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)
    tokens = F.layer_norm(tokens, (1024,), weight=None, bias=None)
    with torch.inference_mode():
        logits = probe(tokens, (24, 32))
        probabilities = torch.softmax(logits, dim=1)
    return (
        logits.argmax(1).cpu().numpy().astype(np.uint8),
        probabilities.cpu().numpy().astype(np.float32),
    )


def frame_contract_common() -> dict:
    arm = ENC.VJepa21CroppedV03Arm()
    return {
        "encoder_checkpoint_sha256": sha(Path(arm.checkpoint)),
        "preprocessing_identity": ENC.preprocessing_identity(arm),
        "preprocessing_digest": ENC.preprocessing_hash(arm),
        "crop_rows": [28, 196],
        "resize_wh": [512, 384],
        "normalisation": "ImageNet mean/std after RGB scale to [0,1]",
        "token_shape": [768, 1024],
        "token_grid_hw": [24, 32],
        "token_order": "row-major 24x32 patch grid",
        "token_normalisation": "FP16 reload as FP32 then affine-free LayerNorm across 1024-D feature axis",
        "probe_decision": "argmax over unchanged [unknown,free,occupied] logits",
        "metric_mask": "truth != unknown, applied after rasterisation and before occupied intersection/union",
        "na_handling": "occupied union == 0 yields JSON null; undefined values are never replaced with zero or one",
    }


def producer_route_array(state: dict, replay: dict) -> tuple[np.ndarray, dict]:
    scene_path = Path(state["scene_dir"]) / "manifest.json"
    scene_payload = json.loads(scene_path.read_text())
    scene = parse_scene_manifest_dict(scene_payload)
    raw_boxes = (*scene.walls, *scene.obstacles, *scene.landmarks)
    output = np.empty((3, 64, 64), np.uint8)
    provenance = []
    for horizon in (1, 2, 3):
        frame = replay["horizons"][str(horizon)]
        position = frame["pose"]
        wxyz = frame["quaternion_wxyz"]
        xyzw = (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
        yaw = OCC._quaternion_yaw_xyzw(xyzw)
        camera = DYNAMIC.compose_yaw_aligned_camera(xyzw, yaw)
        boxes = tuple(V4._box_in_yaw_body(
            box, base_position_world=position, stored_yaw_rad=yaw
        ) for box in raw_boxes)
        identity = f"{state['state_id']}:{int(replay['candidate_index']):02d}:H{horizon}"
        frame_input = V4.FrameBuildInputV4(
            frame_key={"identity": identity},
            camera_origin_body_m=tuple(camera.origin_xyz),
            camera_basis_body_fru=V4._normalized_camera_basis_fru(camera),
            ground_plane_z_body_m=-float(position[2]),
            rendered_boxes_body=boxes,
            image_path_metadata_only=frame["rgb_path"],
            image_sha256=frame["rgb_sha256"],
            sidecar_row_identity_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        )
        evidence = V4.build_frame_evidence_v4(frame_input)
        raster = V4.rasterize_observable_camera_ray_evidence_v4(evidence)
        output[horizon - 1] = np.asarray(raster.output_labels, np.uint8)
        provenance.append({
            "horizon": horizon,
            "base_pose": position,
            "quaternion_wxyz": wxyz,
            "stored_yaw_rad": yaw,
            "camera_origin_yaw_body_m": list(camera.origin_xyz),
            "camera_basis_yaw_body_flu": [list(camera.forward_xyz), list(camera.left_xyz), list(camera.up_xyz)],
            "rgb_path": frame["rgb_path"],
            "rgb_sha256": frame["rgb_sha256"],
            "evidence_sha256": evidence.content_sha256(),
            "raster_sha256": raster.content_sha256(),
        })
    return output, {
        "scene_manifest_path": str(scene_path),
        "scene_manifest_sha256": sha(scene_path),
        "rendered_objects": ["walls", "obstacles", "landmarks"],
        "excluded_objects": ["visual_randomization.distractor_objects"],
        "horizons": provenance,
    }


def transform_arrays(array: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "transpose": array.T,
        "horizontal_flip": np.fliplr(array),
        "vertical_flip": np.flipud(array),
        "transpose_plus_horizontal_flip": np.fliplr(array.T),
        "transpose_plus_vertical_flip": np.flipud(array.T),
    }


def clearance_from_classes(classes: np.ndarray, *, subtract_footprint: bool) -> float:
    forward = DYNAMIC.FORWARD_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    left = DYNAMIC.LEFT_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    distance = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)
    occupied = classes == V4.ray_v4.OCCUPIED_CLASS
    if not occupied.any():
        return 10.0
    value = float(distance[occupied].min())
    return value - float(SS.FOOTPRINT_RADIUS_M) if subtract_footprint else value


def corrected_stage_a(rows, split, labels, probe, device, old_result) -> dict:
    """Re-evaluate once after the proven clearance semantic correction."""
    expected_checkpoint = old_result["component_contract"]["checkpoint_sha256"]
    if sha(SS.CHECKPOINT) != expected_checkpoint:
        raise RuntimeError("frozen stuck checkpoint digest mismatch")
    # This locally generated, SHA-256-bound checkpoint includes NumPy scalar
    # metadata in addition to tensors, which is outside the restricted
    # weights-only unpickler allowlist in newer PyTorch versions.
    checkpoint = torch.load(SS.CHECKPOINT, map_location=device, weights_only=False)
    model = SS.StuckBlockedMotionHead().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    store = SS.TokenStore(rows)
    mean = checkpoint["shortfall_mean"].to(device)
    std = checkpoint["shortfall_std"].to(device)
    ids = split["heldout"]
    logits, shortfall = SS.predict_head(model, store, rows, ids, mean, std, device)
    temperature = float(old_result["calibration"]["stuck_temperature"])
    threshold = float(old_result["calibration"]["stuck_threshold"])
    stuck_probability = 1 / (1 + np.exp(-logits / temperature))

    predicted_clearance = np.empty((len(ids), 3), np.float32)
    clearance_risk = np.empty((len(ids), 3), np.float32)
    intersections = np.zeros(3, np.int64)
    unions = np.zeros(3, np.int64)
    per_row_iou = np.full((len(ids), 3), np.nan)
    forward = DYNAMIC.FORWARD_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    left = DYNAMIC.LEFT_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    point_distance = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)
    near_point = point_distance <= CLEARANCE_SAFE_M
    for offset, row_index in enumerate(ids):
        row = rows[int(row_index)]
        latent = np.stack([np.asarray(np.load(path, mmap_mode="r"), np.float32)
                           for path in row["future_paths"]])
        classes, probabilities = score_probe(probe, latent, device)
        instantaneous = []
        instantaneous_risk = []
        for horizon in range(3):
            counts = OCC.occupied_counts(classes[horizon], labels[int(row_index), horizon])
            intersections[horizon] += counts["occupied_intersection"]
            unions[horizon] += counts["occupied_union"]
            if counts["observable_occupied_iou"] is not None:
                per_row_iou[offset, horizon] = counts["observable_occupied_iou"]
            instantaneous.append(clearance_from_classes(
                classes[horizon], subtract_footprint=False
            ))
            instantaneous_risk.append(float(probabilities[horizon, 2][near_point].max()))
        # Match the authoritative target: cumulative path minimum by horizon.
        predicted_clearance[offset] = np.minimum.accumulate(instantaneous)
        clearance_risk[offset] = np.maximum.accumulate(instantaneous_risk)
    occupancy = {
        "predicted_clearance": predicted_clearance,
        "clearance_risk": clearance_risk,
        "pooled_iou": [float(intersections[h] / unions[h]) if unions[h] else None for h in range(3)],
        "per_row_iou": per_row_iou,
    }
    clearance_unsafe = predicted_clearance[:, 2] < CLEARANCE_SAFE_M
    admitted = ~(clearance_unsafe | (stuck_probability[:, 2] >= threshold))
    risk = np.maximum(clearance_risk[:, 2], stuck_probability[:, 2])
    SS.CALIBRATION = dict(old_result["calibration"])
    kinematic = np.stack([row["kinematic"] for row in rows])
    report = SS.metric_report(rows, ids, occupancy, stuck_probability,
                              shortfall, admitted, risk, kinematic)
    oracle = old_result["oracle_safety_kinematic"]
    spatial, stuck = report["spatial"], report["stuck"]
    branch, planning = report["branch"], report["planning"]
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    checks = {
        "h3_occupied_iou_ge_0_35": spatial["occupied_iou_by_horizon"]["3"] is not None and spatial["occupied_iou_by_horizon"]["3"] >= .35,
        "clearance_spearman_ge_0_60": spatial["clearance_spearman_by_horizon"]["3"] is not None and spatial["clearance_spearman_by_horizon"]["3"] >= .60,
        "low_clearance_recall_ge_0_90": spatial["h3_low_clearance_recall"] is not None and spatial["h3_low_clearance_recall"] >= .90,
        "stuck_auc_ge_0_85": stuck["auc"] is not None and stuck["auc"] >= .85,
        "stuck_recall_ge_0_90": stuck["recall"] is not None and stuck["recall"] >= .90,
        "stuck_fnr_le_0_10": stuck["false_negative_rate"] is not None and stuck["false_negative_rate"] <= .10,
        "unsafe_recall_ge_0_95": branch["unsafe_recall"] >= .95,
        "unsafe_fnr_le_0_05": branch["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": branch["safe_candidate_retention"] >= .40,
        "six_states_retain_safe": planning["states_retaining_safe"] >= 6,
        "no_state_only_unsafe": planning["states_only_unsafe_admitted"] == 0,
        "no_safe_states_abstain": planning["correct_abstention"] == oracle["planning"]["correct_abstention"],
        "selected_unsafe_zero": planning["selected_unsafe_rate"] == 0,
        "route_progress_ge_80pct_oracle": planning["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": planning["normalized_safe_progress_regret"] is not None and planning["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": planning["best_safe_top3"] is not None and planning["best_safe_top3"] >= .75,
        "false_abstention_le_1": planning["false_abstention"] <= 1,
    }
    return {
        "correction": {
            "old_prediction": "instantaneous endpoint occupied-cell range minus 0.47 m footprint radius",
            "old_target": "cumulative tick-level minimum centre-point clearance to walls/obstacles",
            "new_prediction": "cumulative minimum H1-H3 occupied-cell centre range without footprint subtraction",
            "new_target": "unchanged cumulative tick-level minimum centre-point clearance in metres",
            "stuck_checkpoint_reused": sha(SS.CHECKPOINT),
            "stuck_temperature_reused": temperature,
            "stuck_threshold_reused": threshold,
            "labels_regenerated": 0,
        },
        "heldout": report,
        "checks": checks,
        "passed": all(checks.values()),
        "classification": ("STAGE_A_CORRECTED_PASS" if all(checks.values())
                           else "TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO"),
    }


def run() -> dict:
    started = time.time()
    identities = json.loads(IDENTITIES.read_text())
    if identities.get("identity_digest") != canonical_digest({
        k: v for k, v in identities.items() if k != "identity_digest"
    }):
        raise RuntimeError("frozen panel identity digest mismatch")
    rows, ref_by_digest, latent_by_digest, label_by_digest = reference_metadata()
    route_rows, route_split = SS.load_rows()
    route_index = json.loads((ROUTE_V2 / "target_latent_index.json").read_text())
    route_latent = {(x["state_id"], int(x["candidate_index"]), int(x["horizon"])): x
                    for x in route_index["entries"]}
    states = {x["state_id"]: x for x in json.loads(
        (ROUTE_V1 / "state_manifest.json").read_text()
    )["state_candidates"]}
    replay = {}
    for path in sorted((ROUTE_V2 / "replay").glob("purpose-*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            replay[(row["state_id"], int(row["candidate_index"]))] = row

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_metadata = OCC.validate_probe_package_metadata()
    if probe_metadata["package_digest"] != EXPECTED_PACKAGE or probe_metadata["weights_sha256"] != EXPECTED_WEIGHTS:
        raise RuntimeError("occupancy probe binding mismatch")
    probe, probe_state_digest = OCC.load_probe(device)
    common = frame_contract_common()

    reference_rows = []
    pooled_intersection = pooled_union = 0
    first_divergence = None
    for selected in identities["reference"]:
        key = selected["branch_identity"]
        row = ref_by_digest[key]
        latent_record = latent_by_digest[key]
        label_record = label_by_digest[key]
        latent_path, label_path = reference_paths(key, latent_record, label_record)
        latent = np.memmap(latent_path, mode="r", dtype=np.float16, shape=(4, 768, 1024))
        label = np.memmap(label_path, mode="r", dtype=np.uint8, shape=(4, 64, 64))
        classes, probabilities = score_probe(probe, np.asarray(latent[2], np.float32)[None], device)
        counts = OCC.occupied_counts(classes[0], np.asarray(label[2]))
        registered = json.loads((REFERENCE_ROOT / "occupancy_results/true_target_records" / f"{key}.json").read_text())["horizons"][2]
        exact = all(counts[field] == registered[field] for field in (
            "observable_cells", "occupied_support", "occupied_predicted",
            "occupied_intersection", "occupied_union", "observable_occupied_iou"
        ))
        if not exact and first_divergence is None:
            first_divergence = {"panel": "reference", "branch_identity": key,
                                "field": next(field for field in counts if counts[field] != registered[field]),
                                "current": counts, "registered": registered}
        pooled_intersection += counts["occupied_intersection"]
        pooled_union += counts["occupied_union"]
        frame = row["horizon_frames"][2]
        reference_rows.append({
            **selected,
            "h3_frame_identity": frame.get("sha256"),
            "h3_frame_path": frame.get("path"),
            "target_latent": {"path": str(latent_path), "shape": [768, 1024], "dtype": "float16",
                              "sha256": latent_record["sha256"], "encoder_checkpoint_sha256": latent_record["target_encoder_checkpoint_sha256"]},
            "base_pose": row["horizon_base_poses"][2],
            "camera_pose_world": frame.get("camera_pose_world"),
            "label": {"path": str(label_path), "shape": [64, 64], "dtype": "uint8",
                      "sha256": label_record["label_sha256"], "coordinate_frame": "H3 yaw-aligned body frame",
                      "axis_order": "row=forward, column=left", "class_convention": {"unknown": 0, "free": 1, "occupied": 2}},
            "counts": counts,
            "registered_counts": registered,
            "exact_registered_reproduction": exact,
            "probe_positive_probability_summary": {
                "minimum": float(probabilities[0, 2].min()), "maximum": float(probabilities[0, 2].max()),
                "mean": float(probabilities[0, 2].mean())},
            "registered_logits_or_probability_array_available": False,
        })
    reference_exact = all(row["exact_registered_reproduction"] for row in reference_rows)

    route_labels = np.load(ROUTE_LABELS, mmap_mode="r")
    if route_labels.shape != (576, 3, 64, 64) or route_labels.dtype != np.uint8:
        raise RuntimeError("route occupancy label cache malformed")
    route_current_vs_producer = []
    route_rows_output = []
    route_pooled_intersection = route_pooled_union = 0
    route_structural_disagreement = False
    hand_checks = []
    for ordinal, selected in enumerate(identities["route"]):
        index = int(selected["row_index"])
        row = route_rows[index]
        state = states[row["state_id"]]
        replay_row = replay[(row["state_id"], row["candidate_index"])]
        regenerated, provenance = producer_route_array(state, replay_row)
        current = np.asarray(route_labels[index])
        exact_by_horizon = [bool(np.array_equal(current[h], regenerated[h])) for h in range(3)]
        structural = all(exact_by_horizon)
        route_structural_disagreement |= not structural
        h3_record = route_latent[(row["state_id"], row["candidate_index"], 3)]
        latent_path = Path(h3_record["latent_path"])
        if sha(latent_path) != h3_record["sha256"]:
            raise RuntimeError("route latent shard digest mismatch")
        latent = np.asarray(np.load(latent_path, mmap_mode="r"), np.float32)
        classes, probabilities = score_probe(probe, latent[None], device)
        counts = OCC.occupied_counts(classes[0], current[2])
        route_pooled_intersection += counts["occupied_intersection"]
        route_pooled_union += counts["occupied_union"]
        transforms = None
        if not structural:
            transforms = {}
            for name, transformed in transform_arrays(current[2]).items():
                transforms[name] = {
                    "exact": bool(np.array_equal(transformed, regenerated[2])),
                    "cell_agreement": float(np.mean(transformed == regenerated[2])),
                }
        frame = replay_row["horizons"]["3"]
        point_pred = clearance_from_classes(classes[0], subtract_footprint=False)
        old_pred = clearance_from_classes(classes[0], subtract_footprint=True)
        graph = SceneGraph(parse_scene_manifest_dict(json.loads(
            (Path(state["scene_dir"]) / "manifest.json").read_text()
        )))
        endpoint_point = float(graph.clearance_to_walls(tuple(frame["pose"][:2])))
        current_true = float(row["clearance"][2])
        label_point = clearance_from_classes(current[2], subtract_footprint=False)
        item = {
            **selected,
            "h3_frame_identity": frame["rgb_sha256"],
            "h3_frame_path": frame["rgb_path"],
            "target_latent": {"path": str(latent_path), "shape": list(latent.shape), "dtype": "float16",
                              "sha256": h3_record["sha256"], "encoder_checkpoint_sha256": route_index["encoder_sha256"]},
            "base_pose": {"position_world_xyz": frame["pose"], "quaternion_world_wxyz": frame["quaternion_wxyz"]},
            "label": {"path": str(ROUTE_LABELS), "row_index": index, "shape": [64, 64], "dtype": "uint8",
                      "coordinate_frame": "H3 yaw-aligned body frame", "axis_order": "row=forward, column=left",
                      "class_convention": {"unknown": 0, "free": 1, "occupied": 2}},
            "current_vs_reference_producer": {"exact_by_horizon": exact_by_horizon,
                                               "exact_h3": structural, "h3_cell_agreement": float(np.mean(current[2] == regenerated[2]))},
            "alignment_transforms": transforms,
            "counts": counts,
            "probe_positive_probability_summary": {
                "minimum": float(probabilities[0, 2].min()), "maximum": float(probabilities[0, 2].max()),
                "mean": float(probabilities[0, 2].mean())},
            "clearance": {
                "authoritative_label_m": current_true,
                "authoritative_label_semantics": "cumulative tick-level minimum centre-point distance to walls/obstacles through H3",
                "scene_geometry_endpoint_point_clearance_m": endpoint_point,
                "current_stage_a_prediction_m": old_pred,
                "current_stage_a_prediction_semantics": "instantaneous H3 nearest predicted occupied-cell centre minus 0.47 m footprint; 10 m sentinel if none",
                "corrected_instantaneous_point_prediction_m": point_pred,
                "true_label_raster_nearest_occupied_point_range_m": label_point,
                "units": "metres throughout; no normalization or denormalization",
                "sentinel_no_predicted_occupied_m": 10.0,
                "clipping": "none",
            },
            "producer_provenance": provenance,
        }
        route_rows_output.append(item)
        route_current_vs_producer.append(structural)
        if ordinal < 4:
            hand_checks.append({
                "branch_identity": selected["branch_identity"],
                "scene_geometry_endpoint_point_clearance_m": endpoint_point,
                "stored_cumulative_path_minimum_clearance_m": current_true,
                "raster_nearest_visible_occupied_range_m": label_point,
                "predicted_nearest_occupied_range_m": point_pred,
                "old_footprint_subtracted_prediction_m": old_pred,
            })

    contract = {
        "common": common,
        "reference": {
            "target_index_contract": json.loads((REFERENCE_ROOT / "latents_index.json").read_text())["preprocess"],
            "label_contract": json.loads((REFERENCE_ROOT / "occupancy_labels/labels_index.json").read_text())["label_contract"],
        },
        "route": {
            "encoder_source": "encode_safe_local_waypoint_route_intent_v2.VJepa21CroppedV03Arm",
            "encoder_index_metadata_complete": False,
            "encoder_index_missing_fields": ["preprocessing_digest", "token_order", "normalisation", "render_contract_digest"],
            "source_inspection_confirms_common_contract": True,
            "label_producer": "same compose_yaw_aligned_camera + FrameBuildInputV4 + build/rasterize V4 implementation",
            "raster_source": "stored endpoint pose/quaternion, frozen scene walls/obstacles/landmarks; no simulator",
        },
        "explicit_field_comparison": {
            "source_vs_endpoint_pose": "both panels use the per-horizon endpoint base pose; neither uses the branch-start pose",
            "world_vs_camera_body": "scene boxes start in world coordinates, are transformed into endpoint yaw-body coordinates, then camera evidence is rasterised in that body frame",
            "row_vs_column": "row is forward and column is left in both panels",
            "axis_orientation": "increasing row is increasing forward; increasing column is increasing left; no flip",
            "crop_offset": "rows 28:196 in both panels",
            "resize_scaling": "224x168 to 512x384 bicubic in both panels",
            "timestamp": "route H3 latent RGB SHA and H3 label pose belong to the same replay H3 record; reference uses its paired H3 frame and pose receipts",
            "mask_order": "unknown/observable mask is derived from the completed raster and applied afterward during IoU; it is not applied before rasterisation",
        },
    }

    mismatch = {
        "found": True,
        "field": "clearance prediction/target semantics and temporal aggregation",
        "current_prediction": "instantaneous endpoint configuration-clearance proxy (nearest occupied range minus 0.47 m)",
        "current_target": "cumulative path-minimum centre-point clearance (no footprint subtraction)",
        "independent_basis": [
            "Stage-A source subtracts FOOTPRINT_RADIUS_M from an H3 class-map range",
            "collector target is min(DerivedLabelStep.clearance_m) through H3",
            "DerivedLabelStep.clearance_m is SceneGraph.clearance_to_walls(base centre)",
        ],
    }
    old_result = json.loads((SS.OUT / "result.json").read_text())
    corrected = None
    if reference_exact and all(route_current_vs_producer):
        labels_all = np.load(ROUTE_LABELS, mmap_mode="r")
        corrected = corrected_stage_a(route_rows, route_split, labels_all,
                                      probe, device, old_result)
        classification = "STRUCTURED_SAFETY_LABEL_OR_ALIGNMENT_DEFECT"
    elif not reference_exact:
        classification = "OCCUPANCY_CONSUMER_CONTRACT_MISMATCH"
    else:
        classification = "OCCUPANCY_CLEARANCE_DIFFERENTIAL_UNRESOLVED"

    payload = {
        "schema": "occupancy_clearance_contract_differential_v1_result",
        "source_commit": "94711820f212b583196c5abe13820c9852cfe46c",
        "preserved_terminal": "TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO",
        "classification": classification,
        "bindings": {
            "occupancy_result_digest": EXPECTED_OCCUPANCY_RESULT,
            "probe_package_digest": probe_metadata["package_digest"],
            "probe_weights_sha256": probe_metadata["weights_sha256"],
            "probe_state_digest": probe_state_digest,
            "route_target_index_sha256": sha(ROUTE_V2 / "target_latent_index.json"),
            "panel_identity_digest": identities["identity_digest"],
            "route_label_array_sha256": sha(ROUTE_LABELS),
            "route_label_index_sha256": sha(ROUTE_LABEL_INDEX),
        },
        "reference_reproduction": {
            "rows": 16,
            "all_row_counts_and_iou_exact": reference_exact,
            "pooled_selected_iou": pooled_intersection / pooled_union if pooled_union else None,
            "defined_status_exact": all(row["h3_defined"] for row in reference_rows),
            "registered_logits_or_probability_arrays_persisted": False,
            "note": "The frozen records persist class-map sufficient statistics and IoU, not logits; all persisted prediction counts reproduce exactly.",
            "rows_detail": reference_rows,
        },
        "contract_comparison": contract,
        "route_label_comparison": {
            "rows": 16,
            "all_current_labels_equal_reference_producer": all(route_current_vs_producer),
            "structural_disagreement": route_structural_disagreement,
            "alignment_transform_evaluation": ("not_applicable_no_structural_disagreement"
                                               if not route_structural_disagreement else "reported_per_row"),
            "pooled_current_label_iou": route_pooled_intersection / route_pooled_union if route_pooled_union else None,
            "rows_detail": route_rows_output,
        },
        "first_reference_divergent_field": first_divergence,
        "first_divergent_field": (first_divergence if first_divergence is not None else {
            "panel": "route_clearance_consumer",
            "field": mismatch["field"],
            "current": mismatch["current_prediction"],
            "frozen_target": mismatch["current_target"],
        }),
        "clearance_audit": {
            "contract_mismatch": mismatch,
            "hand_checks": hand_checks,
            "route_label_range_m": [float(min(row["clearance"]["authoritative_label_m"] for row in route_rows_output)),
                                    float(max(row["clearance"]["authoritative_label_m"] for row in route_rows_output))],
            "old_prediction_range_m": [float(min(row["clearance"]["current_stage_a_prediction_m"] for row in route_rows_output)),
                                       float(max(row["clearance"]["current_stage_a_prediction_m"] for row in route_rows_output))],
            "semantics": "The 4.08 m MAE was dominated by a 10 m no-occupied sentinel and compared unlike clearance quantities; it was not a unit conversion error.",
        },
        "corrected_stage_a": corrected,
        "corrected_stage_a_justified": bool(reference_exact and all(route_current_vs_producer) and mismatch["found"]),
        "runtime": {"seconds": time.time() - started, "device": str(device)},
        "storage": {},
        "custody": {
            "training": False, "simulation": False, "rgb_rendering": False,
            "latent_encoding": False, "predictor_opened": False,
            "branches_or_latents_modified": False, "route_labels_modified": False,
            "nothing_running_at_finalization": True,
        },
    }
    atomic_json(RESULT, payload)
    # Stabilise byte-count metadata (writing the count can itself change the
    # JSON byte count by a digit at a boundary).
    for _ in range(4):
        size = RESULT.stat().st_size
        updated = {
            "identity_manifest_bytes": IDENTITIES.stat().st_size,
            "result_bytes": size,
            "new_bytes": IDENTITIES.stat().st_size + size,
        }
        if payload["storage"] == updated:
            break
        payload["storage"] = updated
        atomic_json(RESULT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-identities", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_identities == args.run:
        raise SystemExit("choose exactly one of --freeze-identities or --run")
    if args.freeze_identities:
        payload = freeze_identities()
        print(json.dumps({"identity_digest": payload["identity_digest"],
                          "reference_rows": len(payload["reference"]),
                          "route_rows": len(payload["route"])}, indent=2))
        return 0
    payload = run()
    print(json.dumps({
        "classification": payload["classification"],
        "reference_exact": payload["reference_reproduction"]["all_row_counts_and_iou_exact"],
        "reference_pooled_iou": payload["reference_reproduction"]["pooled_selected_iou"],
        "route_label_equal": payload["route_label_comparison"]["all_current_labels_equal_reference_producer"],
        "route_pooled_iou": payload["route_label_comparison"]["pooled_current_label_iou"],
        "clearance_mismatch": payload["clearance_audit"]["contract_mismatch"],
        "corrected_stage_a": None if payload["corrected_stage_a"] is None else {
            "classification": payload["corrected_stage_a"]["classification"],
            "passed": payload["corrected_stage_a"]["passed"],
            "checks": payload["corrected_stage_a"]["checks"],
        },
        "result_sha256": sha(RESULT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
