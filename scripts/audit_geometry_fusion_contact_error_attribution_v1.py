#!/usr/bin/env python3
"""Read-only attribution of frozen geometry-fusion contact errors."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
OUT = ROOT / ".generated/geometry_fusion_contact_error_attribution_v1"
SOURCE = ROOT / ".generated/geometry_modality_safety_sufficiency_v1"
RESULT = SOURCE / "result.json"
SCALE_INDEX = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/sensor_index.json"
SCALE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
EXPECTED = {
    "DEPTH_ONLY": "6edd8b4c754f631759e343f40bf88502bada9c4b8923584696de87235d7ea4b0",
    "LIDAR_ONLY": "e225ac245e8e625750a59cd17cf8608e507b44a4d693946548e03f7db807d26b",
    "DEPTH_PLUS_EMBODIED": "8c51342d431c20496a60a69675851005cd9cc0d88f1440c9a583f1ae6d465204",
}
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
PROXIMITY_M = 0.35


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def ledger(condition: str, source: dict) -> dict[str, np.ndarray]:
    evidence = source["conditions"][condition]["row_level_evidence"]; path = Path(evidence["file"])
    if sha(path) != evidence["sha256"]: raise RuntimeError(f"{condition}: ledger SHA mismatch")
    with np.load(path, allow_pickle=False) as loaded: arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    if len(arrays["branch_id"]) != 576 or len(set(arrays["branch_id"].tolist())) != 576: raise RuntimeError(f"{condition}: ledger cardinality mismatch")
    return arrays


def load_branch_metadata() -> tuple[dict[str, dict], dict[str, dict], dict]:
    # This imports data reducers but executes no model/checkpoint forward.
    from scripts import train_evaluate_geometry_modality_safety_sufficiency_v1 as experiment
    rows, geometry_index = experiment.load_rows()
    panel_states = {row["state_id"]: row for row in json.loads(SCALE_PANEL.read_text())["states"]}
    held = {}
    for row in rows:
        if row["split"] == "heldout":
            row["start_pose"] = panel_states[row["state_id"]]["start_pose"]
            held[row["branch_id"]] = row
    scale = json.loads(SCALE_INDEX.read_text()); records = {row["state_id"]: row for row in scale["state_records"]}
    if len(held) != 288: raise RuntimeError("held-out metadata cardinality mismatch")
    return held, records, geometry_index


def subset_ledger(arrays: dict[str, np.ndarray], split="heldout") -> dict[str, np.ndarray]:
    ids = np.flatnonzero(arrays["split"] == split)
    return {key: value[ids] for key, value in arrays.items()}


def reproduce(source, ledgers, rows):
    from scripts import train_evaluate_geometry_modality_safety_sufficiency_v1 as experiment
    report = {}
    ordered_rows = [rows[branch] for branch in subset_ledger(ledgers["DEPTH_ONLY"])["branch_id"]]
    for condition, full in ledgers.items():
        arrays = subset_ledger(full); committed = source["conditions"][condition]
        if not np.array_equal(arrays["branch_id"], subset_ledger(ledgers["DEPTH_ONLY"])["branch_id"]): raise RuntimeError("cross-condition row mismatch")
        calibration = committed["calibration"]; metrics = experiment.contact_metrics(ordered_rows, arrays["raw_logits"], calibration["temperature"], calibration["threshold"])
        decision = experiment.decision_metrics(ordered_rows, arrays["calibrated_probability"][:, -1, 1], calibration["threshold"], True)
        for key, value in committed["heldout"]["contact"].items():
            if value is None: assert metrics[key] is None
            elif not math.isclose(float(metrics[key]), float(value), rel_tol=1e-12, abs_tol=1e-12): raise RuntimeError(f"{condition}: first metric divergence {key}")
        expected_state = committed["heldout"]["filtering_and_planning"]["per_state"]
        if [row["selected_candidate"] for row in decision["per_state"]] != [row["selected_candidate"] for row in expected_state]: raise RuntimeError(f"{condition}: selected-candidate divergence")
        report[condition] = {"checkpoint_sha256_expected": EXPECTED[condition], "checkpoint_sha256_actual": sha(Path(committed["training"]["checkpoint"])),
            "ledger_sha256": committed["row_level_evidence"]["sha256"], "metrics": metrics,
            "selected_candidates_exact": True, "metric_tolerance": {"rtol": 1e-12, "atol": 1e-12}}
        if report[condition]["checkpoint_sha256_actual"] != EXPECTED[condition]: raise RuntimeError(f"{condition}: checkpoint SHA mismatch")
    return report, ordered_rows


def threshold_frontier(source, arrays, rows):
    from scripts import train_evaluate_factorised_micro_safety_world_model_v1 as base
    from scripts import train_evaluate_geometry_modality_safety_sufficiency_v1 as experiment
    probability = arrays["calibrated_probability"][:, -1, 1]; raw = arrays["raw_logits"]
    temperature = source["conditions"]["DEPTH_PLUS_EMBODIED"]["calibration"]["temperature"]
    fields = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "admitted_contact_negative_count",
        "admitted_contact_positive_count", "states_retaining_contact_negative", "states_only_contact_positive_admitted",
        "selected_contact_count", "false_abstentions", "mean_selected_route_progress_m", "normalized_route_progress_regret",
        "best_safe_top1", "best_safe_top3", "oracle_progress_fraction")
    values = defaultdict(list); complete = []
    for threshold in base.threshold_values(probability):
        threshold = float(threshold); decision = experiment.decision_metrics(rows, probability, threshold, True)
        contact = experiment.contact_metrics(rows, raw, temperature, threshold)
        families = experiment.per_family(rows, raw, {"temperature": temperature, "threshold": threshold})
        gate = experiment.gate(contact, decision, families)
        values["threshold"].append(threshold)
        for field in fields: values[field].append(np.nan if decision[field] is None else decision[field])
        values["event_tick_recall"].append(contact["event_tick_recall"])
        values["complete_gate"].append(int(gate["passed"]))
        for family in FAMILIES:
            item = families[family]["filtering_and_planning"]
            values[f"{family}__recall"].append(families[family]["contact"]["recall"])
            values[f"{family}__retention"].append(families[family]["contact"]["contact_negative_retention"])
            values[f"{family}__states_retaining"].append(item["states_retaining_contact_negative"])
            values[f"{family}__selected_contact"].append(item["selected_contact_count"])
            values[f"{family}__progress"].append(item["mean_selected_route_progress_m"])
        if gate["passed"]: complete.append({"threshold": threshold, "contact": contact, "decision": decision, "per_family": families})
    arrays_out = {key: np.asarray(value, np.float64) for key, value in values.items()}; path = OUT / "heldout_threshold_frontier_v1.npz"; atomic_npz(path, **arrays_out)
    recall95 = arrays_out["contact_recall"] >= .95
    zero_selected = arrays_out["selected_contact_count"] == 0
    best_retention = float(np.max(arrays_out["contact_negative_retention"][recall95])) if recall95.any() else None
    best_progress = float(np.max(arrays_out["mean_selected_route_progress_m"][zero_selected])) if zero_selected.any() else None
    def point(index):
        names = ("threshold", "contact_recall", "contact_false_negative_rate", "contact_negative_retention",
            "admitted_contact_negative_count", "admitted_contact_positive_count", "states_retaining_contact_negative",
            "states_only_contact_positive_admitted", "selected_contact_count", "false_abstentions",
            "mean_selected_route_progress_m", "normalized_route_progress_regret", "best_safe_top1", "best_safe_top3",
            "oracle_progress_fraction", "event_tick_recall")
        return {name: None if np.isnan(arrays_out[name][index]) else float(arrays_out[name][index]) for name in names}
    best_retention_index = int(np.flatnonzero(recall95)[np.argmax(arrays_out["contact_negative_retention"][recall95])]) if recall95.any() else None
    best_progress_index = int(np.flatnonzero(zero_selected)[np.argmax(arrays_out["mean_selected_route_progress_m"][zero_selected])]) if zero_selected.any() else None
    return {"post_hoc_held_out_threshold_frontier": True, "thresholds": len(arrays_out["threshold"]),
        "complete_gate_operating_point_exists": bool(complete), "complete_gate_operating_points": complete,
        "capacity_finding": "HELD_OUT_OPERATING_POINT_EXISTS" if complete else "FUSION_SCORE_FRONTIER_NO_GO",
        "max_contact_negative_retention_at_recall_ge_0_95": best_retention,
        "max_states_retaining_at_recall_ge_0_95": int(np.max(arrays_out["states_retaining_contact_negative"][recall95])) if recall95.any() else None,
        "max_progress_with_zero_selected_contact": best_progress,
        "min_regret_at_recall_ge_0_95": float(np.nanmin(arrays_out["normalized_route_progress_regret"][recall95])) if recall95.any() else None,
        "best_retention_point_at_recall_ge_0_95": None if best_retention_index is None else point(best_retention_index),
        "best_progress_point_with_zero_selected_contact": None if best_progress_index is None else point(best_progress_index),
        "frontier_file": str(path), "frontier_sha256": sha(path), "frontier_content_digest": array_digest(arrays_out)}


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key]); h.update(key.encode()); h.update(value.dtype.str.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def pose_deviation(row, state_record, onset):
    with np.load(state_record["shard_path"], allow_pickle=False) as loaded: poses = np.asarray(loaded["poses"], np.float64)[int(row["candidate_index"])]
    pose = poses[-1]; (sx, sy), syaw, _sz = row["start_pose"]; dx, dy = float(pose[0] - sx), float(pose[1] - sy)
    actual_body = np.asarray([math.cos(syaw) * dx + math.sin(syaw) * dy, -math.sin(syaw) * dx + math.cos(syaw) * dy])
    nominal = np.asarray(row["kinematic"][:2], np.float64); nominal_yaw = float(math.atan2(row["kinematic"][2], row["kinematic"][3]))
    actual_yaw = math.atan2(math.sin(float(pose[2]) - syaw), math.cos(float(pose[2]) - syaw))
    previous_xy = np.asarray([sx, sy]) if onset == 0 else poses[onset - 1, :2]
    base_speed = float(np.linalg.norm(poses[onset, :2] - previous_xy) / .1)
    return {"actual_endpoint_world_xy_yaw": pose[:3].tolist(), "actual_endpoint_body_xy": actual_body.tolist(),
        "nominal_endpoint_body_xy": nominal.tolist(), "endpoint_displacement_error_m": float(np.linalg.norm(actual_body - nominal)),
        "nominal_yaw_change_rad": nominal_yaw, "actual_yaw_change_rad": actual_yaw,
        "absolute_yaw_error_rad": abs(math.atan2(math.sin(actual_yaw - nominal_yaw), math.cos(actual_yaw - nominal_yaw))),
        "base_translational_speed_at_contact_tick_m_s": base_speed,
        "relative_contact_speed": None, "relative_contact_speed_unavailable_reason": "contact point/object velocity was not persisted"}


def group_changes(row, onset):
    groups = {"acceleration": slice(0, 3), "angular_velocity": slice(3, 6), "joint_acceleration": slice(33, 45),
        "torque": slice(45, 57), "calf_net_contact_force": slice(57, 61), "command_response_joint_velocity": slice(21, 33)}
    current, future = np.asarray(row["current_enhanced"]), np.asarray(row["future_enhanced"])
    output = {}
    for name, indices in groups.items():
        delta = np.linalg.norm(future[:, indices] - current[indices], axis=1)
        output[name] = {"pre_onset_peak_change": float(delta[:onset].max()) if onset else None, "event_tick_change": float(delta[onset])}
    return output


def sensor_visibility(row, onset):
    depth = np.asarray(row["future_depth"]); lidar = np.asarray(row["future_lidar"])
    depth_sector = depth[:, depth.shape[1] // 4:3 * depth.shape[1] // 4, depth.shape[2] // 4:3 * depth.shape[2] // 4]
    # Scan starts at -pi. Forward body-path sector is +/-45 degrees.
    azimuth = np.linspace(-math.pi, math.pi, lidar.shape[-1], endpoint=False); forward = np.abs(azimuth) <= math.pi / 4
    lidar_sector = lidar[..., forward]
    dmin = depth_sector.min(axis=(1, 2)); lmin = lidar_sector.min(axis=(1, 2)); lall = lidar.min(axis=(1, 2))
    def one(values):
        crossings = np.flatnonzero(values < PROXIMITY_M); before = bool(np.any(values[:onset] < PROXIMITY_M)); at = bool(values[onset] < PROXIMITY_M)
        return {"visible_before_onset_by_proximity_proxy": before, "visible_only_at_onset_by_proximity_proxy": bool(at and not before),
            "minimum_range_m": float(values.min()), "minimum_range_before_or_at_onset_m": float(values[:onset + 1].min()),
            "first_potentially_informative_tick": None if not len(crossings) else int(crossings[0] + 1)}
    return {"front_depth": {**one(dmin), "contact_point_inside_horizontal_vertical_fov": None, "occluded": None, "outside_fov": None,
            "limitation": "contact point/link not persisted; central planned-path sector is a proxy"},
        "lidar": {**one(lmin), "minimum_full_scan_range_m": float(lall.min()), "contact_point_inside_scan_coverage": None,
            "occluded_or_missing": None, "limitation": "contact point/link not persisted; forward sector and full scan are proxies"}}


def error_class(visibility, embodied):
    depth = visibility["front_depth"]; lidar = visibility["lidar"]
    if depth["visible_before_onset_by_proximity_proxy"]: return "GEOMETRY_VISIBLE_MODEL_MISS"
    if depth["visible_only_at_onset_by_proximity_proxy"]: return "ONLY_EVENT_TICK_OBSERVABLE"
    if lidar["visible_before_onset_by_proximity_proxy"] or lidar["visible_only_at_onset_by_proximity_proxy"]: return "MULTIPLE"
    signal = max(value["event_tick_change"] for value in embodied.values())
    return "EMBODIED_SIGNAL_WEAK_OR_ABSENT" if signal < 1e-3 else "UNRESOLVED"


def inventory(source, ledgers, rows, state_records):
    fusion = subset_ledger(ledgers["DEPTH_PLUS_EMBODIED"]); lidar = subset_ledger(ledgers["LIDAR_ONLY"])
    fprob = fusion["calibrated_probability"][:, -1, 1]; lprob = lidar["calibrated_probability"][:, -1, 1]
    ft = source["conditions"]["DEPTH_PLUS_EMBODIED"]["calibration"]["threshold"]
    lt = source["conditions"]["LIDAR_ONLY"]["calibration"]["threshold"]
    positives = fusion["contact_labels"][:, -1, 1].astype(bool); ffn = positives & (fprob < ft); admitted_positive = ffn
    selected = fusion["selected"].astype(bool); selected_positive = positives & selected
    all_rows = []
    for index in np.flatnonzero(positives):
        row = rows[index]; active = fusion["contact_labels"][index, :, 0].astype(bool); ticks = np.flatnonzero(active); onset = int(ticks[0]); duration = int(active.sum())
        vis = sensor_visibility(row, onset); embodied = group_changes(row, onset); plan = np.asarray(row["post_slew"], np.float64).reshape(-1, 3)
        item = {"branch_id": str(fusion["branch_id"][index]), "family": str(fusion["family"][index]), "scene_id": row["scene_id"],
            "state_id": str(fusion["state_id"][index]), "candidate_index": int(fusion["candidate_index"][index]),
            "candidate": row.get("candidate"), "action_primitives": row.get("primitives"), "contact_onset_tick": onset + 1,
            "contact_duration_ticks": duration, "transient_or_persistent": "transient" if duration == 1 else "persistent",
            "body_link_or_region": None, "contact_position_relative_robot": None, "relative_speed_at_contact": None,
            "commanded_speed_at_contact_m_s": float(np.linalg.norm(plan[onset, :2])), "commanded_yaw_rate_at_contact_rad_s": float(plan[onset, 2]),
            "nominal_translation_body_xy_m": row["kinematic"][:2].astype(float).tolist(),
            "nominal_rotation_rad": float(math.atan2(row["kinematic"][2], row["kinematic"][3])),
            "peak_contact_force_or_impulse": None, "penetration_or_overlap_depth": None,
            "physical_fields_unavailable_reason": "frozen labels contain boolean disallowed-contact ticks, not link/contact-point/force/impulse/penetration telemetry",
            "pose_evidence": pose_deviation(row, state_records[row["state_id"]], onset), "followed_by_or_overlaps_stuck": bool(row["stuck"]),
            "realised_route_progress_m": float(row["p_d"]), "meaningful_route_progress": bool(abs(float(row["p_d"])) > .03),
            "fusion_probability": float(fprob[index]), "lidar_probability": float(lprob[index]),
            "fusion_false_negative": bool(ffn[index]), "fusion_admitted_contact_positive": bool(admitted_positive[index]),
            "fusion_selected_contact_positive": bool(selected_positive[index]), "lidar_rejected_at_frozen_threshold": bool(lprob[index] >= lt),
            "sensor_visibility": vis, "embodied_changes": embodied, "false_negative_attribution": None,
            "prospective_severity_category": "SEVERITY_UNRESOLVED", "severity_basis": "force, impulse, body-region damage relevance, object fragility, and calibrated separation evidence are unavailable"}
        if item["fusion_false_negative"]: item["false_negative_attribution"] = error_class(vis, embodied)
        all_rows.append(item)
    false_rows = [row for row in all_rows if row["fusion_false_negative"]]
    selected_rows = [row for row in all_rows if row["fusion_selected_contact_positive"]]
    # Mechanically matched samples: same family, then closest onset/duration,
    # without replacement. Labels and scores do not influence the match key.
    rejected = [row for row in all_rows if not row["fusion_false_negative"]]; used = set(); matched_rejected = []
    for target in false_rows:
        candidates = [row for row in rejected if row["branch_id"] not in used and row["family"] == target["family"]]
        if candidates:
            pick = min(candidates, key=lambda row: (abs(row["contact_onset_tick"] - target["contact_onset_tick"]), abs(row["contact_duration_ticks"] - target["contact_duration_ticks"]), row["branch_id"]))
            used.add(pick["branch_id"]); matched_rejected.append(pick["branch_id"])
    negatives = np.flatnonzero(~positives & (fprob < ft)); matched_negative = []
    for target in false_rows:
        candidates = [i for i in negatives if str(fusion["branch_id"][i]) not in matched_negative and str(fusion["family"][i]) == target["family"]]
        if candidates:
            pick = min(candidates, key=lambda i: (abs(int(fusion["candidate_index"][i]) - target["candidate_index"]), str(fusion["branch_id"][i])))
            matched_negative.append(str(fusion["branch_id"][pick]))
    output = {"schema": "geometry_fusion_contact_error_inventory_v1", "heldout_contact_positive_count": int(positives.sum()),
        "fusion_false_negative_count": int(ffn.sum()), "fusion_admitted_contact_positive_count": int(admitted_positive.sum()),
        "fusion_selected_contact_positive_count": int(selected_positive.sum()), "all_contact_positive_branches": all_rows,
        "fusion_false_negative_branch_ids": [row["branch_id"] for row in false_rows],
        "selected_contact_diagnoses": selected_rows, "matched_correctly_rejected_contact_positive_branch_ids": matched_rejected,
        "matched_retained_contact_negative_branch_ids": matched_negative,
        "unavailable_fields": ["body_link", "body_region", "contact_point", "relative_speed", "contact_force", "impulse", "penetration", "object_fragility"]}
    path = OUT / "contact_error_inventory.json"; atomic_json(path, output)
    output["file"] = str(path); output["sha256"] = sha(path); return output


def complementarity(source, ledgers):
    fusion = subset_ledger(ledgers["DEPTH_PLUS_EMBODIED"]); lidar = subset_ledger(ledgers["LIDAR_ONLY"])
    label = fusion["contact_labels"][:, -1, 1].astype(bool); fp = fusion["calibrated_probability"][:, -1, 1]; lp = lidar["calibrated_probability"][:, -1, 1]
    ft = source["conditions"]["DEPTH_PLUS_EMBODIED"]["calibration"]["threshold"]; lt = source["conditions"]["LIDAR_ONLY"]["calibration"]["threshold"]
    ffn = label & (fp < ft); lfn = label & (lp < lt); selected_contact = label & fusion["selected"].astype(bool)
    def distribution(values): return {"min": float(values.min()), "p05": float(np.quantile(values, .05)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}
    return {"fusion_false_negatives": int(ffn.sum()), "fusion_false_negatives_flagged_by_lidar": int((ffn & (lp >= lt)).sum()),
        "fusion_false_negative_lidar_conditional_recall": float((lp[ffn] >= lt).mean()),
        "lidar_false_negatives": int(lfn.sum()), "lidar_false_negatives_flagged_by_fusion": int((lfn & (fp >= ft)).sum()),
        "lidar_false_negative_fusion_conditional_recall": float((fp[lfn] >= ft).mean()),
        "pearson_score_correlation": float(np.corrcoef(fp, lp)[0, 1]),
        "score_distributions": {"fusion_contact_positive": distribution(fp[label]), "fusion_contact_negative": distribution(fp[~label]),
            "lidar_contact_positive": distribution(lp[label]), "lidar_contact_negative": distribution(lp[~label])},
        "selected_fusion_contacts": int(selected_contact.sum()), "selected_fusion_contacts_rejected_by_lidar": int((selected_contact & (lp >= lt)).sum()),
        "selected_contact_branch_ids": fusion["branch_id"][selected_contact].tolist(),
        "lidar_complementary_tendency_rule": "at least 25% of fusion false negatives rejected by frozen LiDAR",
        "lidar_complementary_tendency": bool((lp[ffn] >= lt).mean() >= .25)}


def severity_summary(inventory):
    rows = inventory["all_contact_positive_branches"]
    return {"contact_positive_branches": len(rows), "transient": sum(row["transient_or_persistent"] == "transient" for row in rows),
        "persistent": sum(row["transient_or_persistent"] == "persistent" for row in rows),
        "followed_by_or_overlaps_stuck": sum(row["followed_by_or_overlaps_stuck"] for row in rows),
        "meaningful_route_progress": sum(row["meaningful_route_progress"] for row in rows),
        "prospective_categories": {"MATERIAL_HAZARDOUS_CONTACT": 0, "RECOVERABLE_LOW_ENERGY_CONTACT": 0, "SEVERITY_UNRESOLVED": len(rows)},
        "category_assignment_is_descriptive_not_relabeling": True,
        "required_evidence": ["link/body-region resolved disallowed contact", "calibrated contact force and impulse", "relative contact speed",
            "penetration/overlap or required separation", "body-region damage model", "object identity and fragility", "contact duration and task interruption consequence"],
        "definitions": {"MATERIAL_HAZARDOUS_CONTACT": "contact whose calibrated energy/impulse, region-specific damage model, separation violation, or fragile-object consequence exceeds a prospectively frozen hazard limit",
            "RECOVERABLE_LOW_ENERGY_CONTACT": "contact prospectively demonstrated below those limits with no persistent damage, loss of control, unsafe separation, or material task interruption",
            "SEVERITY_UNRESOLVED": "the frozen evidence cannot establish either definition"}}


def main() -> int:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); source = json.loads(RESULT.read_text())
    ledgers = {condition: ledger(condition, source) for condition in EXPECTED}; metadata, state_records, geometry_index = load_branch_metadata()
    reproduction, rows = reproduce(source, ledgers, metadata)
    fusion_held = subset_ledger(ledgers["DEPTH_PLUS_EMBODIED"])
    frontier = threshold_frontier(source, fusion_held, rows)
    inventory_result = inventory(source, ledgers, rows, state_records); complement = complementarity(source, ledgers); severity = severity_summary(inventory_result)
    false_rows = [row for row in inventory_result["all_contact_positive_branches"] if row["fusion_false_negative"]]
    attribution_counts = {key: sum(row["false_negative_attribution"] == key for row in false_rows) for key in
        ("FRONT_DEPTH_OUT_OF_FOV", "GEOMETRY_OCCLUDED", "GEOMETRY_VISIBLE_MODEL_MISS", "ONLY_EVENT_TICK_OBSERVABLE", "EMBODIED_SIGNAL_WEAK_OR_ABSENT", "MULTIPLE", "UNRESOLVED")}
    if frontier["complete_gate_operating_point_exists"]:
        classification = "CALIBRATION_PANEL_OR_METHOD_BOTTLENECK"; recommendation = "Preserve the frozen fusion model and validate prospectively specified decision-level calibration on a larger fresh calibration panel."
    else:
        unresolved_fraction = severity["prospective_categories"]["SEVERITY_UNRESOLVED"] / max(severity["contact_positive_branches"], 1)
        if unresolved_fraction >= .5:
            classification = "CONTACT_REQUIREMENT_ONTOLOGY_REQUIRES_REVISION"
            recommendation = "Complete and prospectively freeze a contact-hazard ontology with link, force/impulse, relative-speed, separation, damage, and object-fragility evidence before training or collecting another model panel."
        elif complement["lidar_complementary_tendency"]:
            classification = "WIDE_AREA_GEOMETRY_REQUIRED"; recommendation = "Prospectively specify LIDAR_PLUS_EMBODIED_CONTACT_MODEL_V1 under an explicit changed deployment sensor contract."
        else:
            classification = "GEOMETRY_FUSION_SCORE_NO_GO"; recommendation = "Close the current fusion architecture rather than tuning it."
    secondary = [frontier["capacity_finding"]]
    if complement["lidar_complementary_tendency"]: secondary.append("LIDAR_COMPLEMENTARY_TENDENCY")
    if attribution_counts["ONLY_EVENT_TICK_OBSERVABLE"] / max(len(false_rows), 1) >= .25: secondary.append("EVENT_TICK_ONLY_CONTACT_EVIDENCE")
    result = {"schema": "geometry_fusion_contact_error_attribution_v1_result", "experiment": "GEOMETRY_FUSION_CONTACT_ERROR_ATTRIBUTION_V1",
        "status": "POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC", "source_commit": "06b1ffb8232476456f21fa8fd56284230f26d7c8",
        "preserved_result": "GEOMETRY_MODALITY_POSITIVE_TENDENCY", "frozen_metric_reproduction": reproduction,
        "heldout_threshold_frontier": frontier, "contact_error_inventory": {key: value for key, value in inventory_result.items() if key != "all_contact_positive_branches"},
        "false_negative_attribution_counts": attribution_counts, "selected_contact_diagnoses": inventory_result["selected_contact_diagnoses"],
        "lidar_fusion_complementarity": complement, "contact_severity_and_requirement_audit": severity,
        "primary_classification": classification, "secondary_findings": secondary, "single_recommended_next_action": recommendation,
        "claims_boundary": {"heldout_threshold_not_adopted": True, "labels_not_changed": True, "contact_point_visibility_not_inferred": True,
            "operational_safety_and_task_progress_separate": True, "post_outcome_not_claim_bearing": True},
        "runtime": {"seconds": time.time() - started, "new_storage_bytes": sum(path.stat().st_size for path in OUT.glob("*"))},
        "custody": {"training": False, "checkpoint_inference": False, "simulation": False, "rendering": False, "encoding": False,
            "new_branches_or_sensor_observations": False, "jepa_predictor_opened": False}}
    atomic_json(OUT / "result.json", result); print(json.dumps({"reproduction": "PASS", "thresholds": frontier["thresholds"],
        "complete_gate_operating_point_exists": frontier["complete_gate_operating_point_exists"], "false_negative_attribution": attribution_counts,
        "complementarity": complement, "classification": classification, "secondary": secondary, "runtime": result["runtime"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
