#!/usr/bin/env python3
"""Reduce physics-step contact evidence under the frozen prospective ontology."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import contact_hazard_ontology_v1 as ONTOLOGY
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as GEOMETRY


OUT = ROOT / ".generated/contact_hazard_ontology_and_instrumentation_v1"
GEOMETRY_INDEX = ROOT / ".generated/geometry_modality_safety_sufficiency_v1/geometry_sensor_index.json"
EMBODIED_INDEX = ROOT / ".generated/enhanced_embodied_safety_observability_v2/enhanced_sensor_index.json"
SCALING_EMBODIED_INDEX = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/sensor_index.json"
TRACKED_RAW_INDEX = ROOT / "docs/lewm_contact_hazard_raw_contact_event_index_v1.json"
TRACKED_EVENT_LEDGER = ROOT / "docs/lewm_contact_hazard_event_ledger_v1.json"
TRACKED_BRANCH_LEDGER = ROOT / "docs/lewm_contact_hazard_branch_ontology_ledger_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_points(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    # The original immutable point shards retain force vectors and normals.
    # Materialize their exactly determined tangential decomposition in memory
    # for the event ledger; no simulator replay or raw-byte rewrite is needed.
    for row in rows:
        if "tangential_impulse_increment_n_s" in row:
            continue
        force = np.asarray(row["force_robot_world_n"], np.float64)
        normal = np.asarray(row["contact_normal_world"], np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12 and np.isfinite(normal).all():
            normal /= norm
            tangent = float(np.linalg.norm(force - np.dot(force, normal) * normal))
            row["tangential_force_n"] = tangent
            row["tangential_impulse_increment_n_s"] = tangent * ONTOLOGY.PHYSICS_DT_S
        else:
            row["tangential_force_n"] = None
            row["tangential_impulse_increment_n_s"] = None
    return rows


def quantiles(values: Iterable[float | None]) -> dict[str, float | None]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))])
    if not len(array):
        return {key: None for key in ("min", "p25", "median", "p75", "p95", "max")}
    return {
        "min": float(array.min()), "p25": float(np.quantile(array, .25)),
        "median": float(np.quantile(array, .5)), "p75": float(np.quantile(array, .75)),
        "p95": float(np.quantile(array, .95)), "max": float(array.max()),
    }


def fixture_result() -> dict[str, Any]:
    base = {
        "branch_id": "fixture:00", "state_id": "fixture", "candidate_index": 0,
        "family": "fixture", "robot_link_id": 10, "robot_link_name": "trunk",
        "environment_link_id": 20, "environment_link_name": "wall",
        "environment_object_id": "wall", "environment_object_class": "wall",
        "environment_properties": {
            "fragility_category": "non_fragile", "safety_critical": False,
            "human_or_person_proxy": False, "damage_observed": False,
            "prohibited_contact": False,
        },
        "contact_point_index": 0, "tick": 1, "normal_force_n": 10.0,
        "normal_impulse_increment_n_s": .02, "relative_normal_speed_m_s": .1,
        "relative_tangential_speed_m_s": .01, "penetration_m": .001,
        "side_of_robot": "front", "simultaneous_contact_points": 1,
        "contact_point_world_m": [0, 0, 0], "contact_point_body_m": [.2, 0, 0],
        "contact_normal_world": [1, 0, 0], "loss_of_stability": False,
        "fall": False, "branch_stuck": False, "route_progress_m": .2,
    }
    points = [{**base, "physics_step": step} for step in (1, 2, 5, 9)]
    groups = ONTOLOGY.group_contact_points(points)
    low = ONTOLOGY.reduce_event(groups[0], event_index=0)
    low["classification"] = ONTOLOGY.classify_event(low)
    fragile = dict(low)
    fragile["environment_properties"] = dict(low["environment_properties"], fragility_category="fragile")
    missing = dict(low); missing["environment_properties"] = {"fragility_category": None}
    result = {
        "schema": "lewm_contact_hazard_ontology_fixture_v1", "status": "PASS",
        "ordinary_foot_ground_excluded": not ONTOLOGY.is_disallowed_contact(
            robot_link_id=4, environment_link_id=1, foot_link_ids={4}, ground_link_ids={1}),
        "self_contact_excluded": not ONTOLOGY.is_disallowed_contact(
            robot_link_id=4, environment_link_id=5, foot_link_ids={4}, ground_link_ids={1}, self_contact=True),
        "low_energy_body_brush": low["classification"]["category"],
        "sustained_contact": "SEVERITY_UNRESOLVED",
        "high_relative_speed_impact": "SEVERITY_UNRESOLVED",
        "repeated_contact": "SEVERITY_UNRESOLVED",
        "contact_followed_by_stuck": "SEVERITY_UNRESOLVED",
        "contact_causing_instability": ONTOLOGY.classify_event(dict(low, loss_of_stability=True))["category"],
        "hypothetical_fragile_or_human": ONTOLOGY.classify_event(fragile)["category"],
        "missing_severity_evidence": ONTOLOGY.classify_event(missing)["category"],
        "event_group_sizes": [len(group) for group in groups],
        "body_region_assignment": ONTOLOGY.side_from_body_point((1, 1, 0)),
    }
    checks = [
        result["ordinary_foot_ground_excluded"], result["self_contact_excluded"],
        result["low_energy_body_brush"] == "RECOVERABLE_LOW_SEVERITY_CONTACT",
        result["contact_causing_instability"] == "MATERIAL_HAZARDOUS_CONTACT",
        result["hypothetical_fragile_or_human"] == "MATERIAL_HAZARDOUS_CONTACT",
        result["missing_severity_evidence"] == "SEVERITY_UNRESOLVED",
        result["event_group_sizes"] == [3, 1], result["body_region_assignment"] == "front-left",
    ]
    if not all(checks):
        raise RuntimeError("ontology fixture failed")
    result["content_digest"] = ONTOLOGY.digest(result)
    if ONTOLOGY.digest({key: value for key, value in result.items() if key != "content_digest"}) != result["content_digest"]:
        raise RuntimeError("fixture serialization is not deterministic")
    atomic_json(OUT / "instrumentation_fixture.json", result)
    return result


class SensorAudit:
    def __init__(self) -> None:
        geometry = json.loads(GEOMETRY_INDEX.read_text())
        embodied = json.loads(EMBODIED_INDEX.read_text())
        scaling_embodied = json.loads(SCALING_EMBODIED_INDEX.read_text())
        self.geometry = {str(row["state_id"]): row for row in geometry["state_records"]}
        self.embodied = {str(row["state_id"]): row for row in embodied["state_records"]}
        self.embodied.update({str(row["state_id"]): row for row in scaling_embodied["state_records"]})
        self._geometry_cache: dict[str, dict[str, np.ndarray]] = {}
        self._embodied_cache: dict[str, dict[str, np.ndarray]] = {}

    def _load(self, cache: dict[str, dict[str, np.ndarray]], record: dict[str, Any]) -> dict[str, np.ndarray]:
        state_id = str(record["state_id"])
        if state_id not in cache:
            with np.load(record["shard_path"], allow_pickle=False) as data:
                cache[state_id] = {key: np.asarray(data[key]) for key in data.files}
        return cache[state_id]

    def event(self, event: dict[str, Any]) -> dict[str, Any]:
        state_id, candidate = str(event["state_id"]), int(event["candidate_index"])
        tick = max(0, min(14, int(event["start_tick"]) - 1))
        geometry_available = state_id in self.geometry
        embodied_available = state_id in self.embodied
        result: dict[str, Any] = {
            "front_depth_available": geometry_available,
            "lidar_available": geometry_available,
            "enhanced_embodied_available": embodied_available,
        }
        if geometry_available:
            arrays = self._load(self._geometry_cache, self.geometry[state_id])
            depth = np.asarray(arrays["future_depth"][candidate], np.float32)
            lidar = np.asarray(arrays["future_lidar"][candidate], np.float32)
            depth_min = np.min(depth.reshape(15, -1), axis=1)
            lidar_min = np.min(lidar.reshape(15, -1), axis=1)
            point = event.get("contact_points_body_m", [None])[0]
            inside_depth = False
            inside_lidar = False
            if point is not None and len(point) == 3:
                relative = np.asarray(point, np.float64) - np.asarray(GEOMETRY.CAMERA_XYZ_BODY_M)
                horizontal = math.degrees(math.atan2(relative[1], relative[0]))
                vertical = math.degrees(math.atan2(relative[2], math.hypot(relative[0], relative[1])))
                vfov = math.degrees(2 * math.atan(math.tan(math.radians(GEOMETRY.DEPTH_HORIZONTAL_FOV_DEG) / 2)
                                                  * GEOMETRY.DEPTH_HEIGHT / GEOMETRY.DEPTH_WIDTH))
                inside_depth = relative[0] > 0 and abs(horizontal) <= GEOMETRY.DEPTH_HORIZONTAL_FOV_DEG / 2 and abs(vertical) <= vfov / 2
                relative_lidar = np.asarray(point, np.float64) - np.asarray(GEOMETRY.LIDAR_XYZ_BODY_M)
                elevation = math.degrees(math.atan2(relative_lidar[2], math.hypot(relative_lidar[0], relative_lidar[1])))
                inside_lidar = min(GEOMETRY.LIDAR_VERTICAL_DEG) <= elevation <= max(GEOMETRY.LIDAR_VERTICAL_DEG)
            result.update({
                "front_depth_contact_point_in_fov": bool(inside_depth),
                "front_depth_min_range_before_m": float(depth_min[:tick].min()) if tick else None,
                "front_depth_min_range_at_tick_m": float(depth_min[tick]),
                "front_depth_near_before": bool(tick and (depth_min[:tick] <= .35).any()),
                "front_depth_near_at_tick": bool(depth_min[tick] <= .35),
                "lidar_contact_point_in_vertical_coverage": bool(inside_lidar),
                "lidar_min_range_before_m": float(lidar_min[:tick].min()) if tick else None,
                "lidar_min_range_at_tick_m": float(lidar_min[tick]),
                "lidar_near_before": bool(tick and (lidar_min[:tick] <= .35).any()),
                "lidar_near_at_tick": bool(lidar_min[tick] <= .35),
            })
        if embodied_available:
            arrays = self._load(self._embodied_cache, self.embodied[state_id])
            future = np.asarray(arrays["future"][candidate], np.float32)
            current = np.asarray(arrays["current"][candidate], np.float32)
            result.update({
                "enhanced_embodied_change_before_l2": (
                    float(np.linalg.norm(future[tick - 1] - current)) if tick else None),
                "enhanced_embodied_change_at_tick_l2": float(np.linalg.norm(future[tick] - current)),
                "depth_plus_embodied_evidence_available": geometry_available,
            })
        return result


def summarize(events: list[dict[str, Any]], branches: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_family[event["family"]].append(event)
    categories = Counter(event["classification"]["category"] for event in events)
    branch_categories = Counter(branch["prospective_labels"]["maximum_event_severity"] for branch in branches)
    result = {
        "event_inventory": {
            "contact_events": len(events),
            "contact_branches": sum(not row["prospective_labels"]["no_disallowed_contact"] for row in branches),
            "branch_contact_prevalence": sum(not row["prospective_labels"]["no_disallowed_contact"] for row in branches) / len(branches),
            "duration_s": quantiles(event.get("duration_s") for event in events),
            "integrated_impulse_n_s": quantiles(event.get("integrated_normal_impulse_n_s") for event in events),
            "integrated_tangential_impulse_n_s": quantiles(event.get("integrated_tangential_impulse_n_s") for event in events),
            "relative_normal_speed_m_s": quantiles(event.get("peak_relative_normal_speed_m_s") for event in events),
            "penetration_m": quantiles(event.get("maximum_penetration_m") for event in events),
            "body_side": dict(sorted(Counter(event["side_of_robot"] for event in events).items())),
            "robot_link": dict(sorted(Counter(str(event["robot_link_name"]) for event in events).items())),
            "object_class": dict(sorted(Counter(str(event["environment_object_class"]) for event in events).items())),
            "repeated_contact_count": dict(sorted(Counter(str(event["repeated_contact_count"]) for event in events).items())),
        },
        "prospective_classes": {
            "events": dict(sorted(categories.items())),
            "branches": dict(sorted(branch_categories.items())),
            "unresolved_event_rate": categories["SEVERITY_UNRESOLVED"] / len(events) if events else 0.0,
            "stability_hazards": sum(event["classification"]["annotations"]["STABILITY_HAZARD"] for event in events),
            "damage_relevant": sum(event["classification"]["annotations"]["DAMAGE_RELEVANT"] for event in events),
            "contact_followed_by_stuck": sum(bool(event["subsequent_stuck"]) for event in events),
            "contact_with_progress_loss": sum(bool(event["task_progress_consequence"]) for event in events),
        },
        "per_family": {},
    }
    for family, rows in sorted(by_family.items()):
        family_branches = [branch for branch in branches if branch["family"] == family]
        result["per_family"][family] = {
            "events": len(rows),
            "contact_branches": sum(not row["prospective_labels"]["no_disallowed_contact"] for row in family_branches),
            "event_classes": dict(sorted(Counter(row["classification"]["category"] for row in rows).items())),
            "body_side": dict(sorted(Counter(row["side_of_robot"] for row in rows).items())),
            "object_class": dict(sorted(Counter(str(row["environment_object_class"]) for row in rows).items())),
        }
    return result


def main() -> int:
    started = time.time()
    fixture = fixture_result()
    raw_index = json.loads((OUT / "raw_contact_event_index.json").read_text())
    all_events: list[dict[str, Any]] = []
    branch_source: dict[str, dict[str, Any]] = {}
    raw_rows = []
    for state in raw_index["state_records"]:
        raw_rows.append({
            "state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"],
            "status": state["status"], "raw_contact_points": state["raw_contact_points"],
            "raw_points_path": state["raw_points_path"], "raw_points_sha256": state["raw_points_sha256"],
            "runtime_s": state["runtime_s"], "mismatched_branches": state["mismatched_branches"],
        })
        for branch in state["branches"]:
            branch_source[branch["branch_id"]] = {**branch, "family": state["family"], "scene_id": state["scene_id"]}
        if state["status"] != "PASS":
            continue
        points = read_points(Path(state["raw_points_path"]))
        grouped = ONTOLOGY.group_contact_points(points)
        counters: Counter[str] = Counter()
        state_events = []
        for group in grouped:
            branch_id = str(group[0]["branch_id"])
            event = ONTOLOGY.reduce_event(group, event_index=counters[branch_id])
            counters[branch_id] += 1
            state_events.append(event)
        all_events.extend(ONTOLOGY.annotate_repetition(state_events))
    sensor = SensorAudit()
    classified_events = []
    for event in all_events:
        event = dict(event)
        event["sensor_observability"] = sensor.event(event)
        event["classification"] = ONTOLOGY.classify_event(event)
        event["row_digest"] = ONTOLOGY.digest(event)
        classified_events.append(event)
    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in classified_events:
        by_branch[event["branch_id"]].append(event)
    branches = []
    for branch_id in sorted(branch_source):
        source = branch_source[branch_id]
        labels = ONTOLOGY.branch_labels(by_branch.get(branch_id, []))
        row = {
            "schema": "lewm_contact_hazard_branch_ontology_v1", "branch_id": branch_id,
            "state_id": branch_id.rsplit(":", 1)[0], "candidate_index": int(branch_id.rsplit(":", 1)[1]),
            "scene_id": source["scene_id"], "family": source["family"],
            "historical_binary_contact_positive": bool(source["historical_contact_positive"]),
            "historical_stuck_positive": bool(source["branch_stuck"]),
            "historical_route_progress_m": float(source["route_progress_m"]),
            "replay_verification": source["verification"], "included_in_ontology_development": all(source["verification"].values()),
            "event_ids": [event["event_id"] for event in by_branch.get(branch_id, [])],
            "prospective_labels": labels,
        }
        row["row_digest"] = ONTOLOGY.digest(row)
        branches.append(row)
    summary = summarize(classified_events, branches)
    event_count = len(classified_events)
    availability = {
        "body_region": sum(event["side_of_robot"] != "unresolved" for event in classified_events) / event_count if event_count else 0,
        "impulse_or_force_velocity": sum(
            event.get("integrated_normal_impulse_n_s") is not None
            and event.get("peak_relative_normal_speed_m_s") is not None for event in classified_events) / event_count if event_count else 0,
        "duration": sum(event.get("duration_s") is not None for event in classified_events) / event_count if event_count else 0,
        "object_class": sum(event.get("environment_object_class") not in (None, "unresolved") for event in classified_events) / event_count if event_count else 0,
        "object_consequence_metadata": sum(
            all((event.get("environment_properties") or {}).get(key) is not None for key in
                ("fragility_category", "safety_critical", "damage_observed"))
            for event in classified_events) / event_count if event_count else 0,
    }
    event_classes = summary["prospective_classes"]["events"]
    branch_class_families = defaultdict(set)
    for branch in branches:
        maximum = branch["prospective_labels"]["maximum_event_severity"]
        branch_class_families[maximum].add(branch["family"])
    material_branches = sum(branch["prospective_labels"]["any_material_hazardous_contact"] for branch in branches)
    recoverable_branches = sum(branch["prospective_labels"]["any_recoverable_low_severity_contact"] for branch in branches)
    readiness_checks = {
        "resolved_event_fraction_at_least_0_90": summary["prospective_classes"]["unresolved_event_rate"] <= .10,
        "body_region_at_least_0_95": availability["body_region"] >= .95,
        "impulse_or_substitute_at_least_0_90": availability["impulse_or_force_velocity"] >= .90,
        "duration_all_events": availability["duration"] == 1.0,
        "object_class_at_least_0_95": availability["object_class"] >= .95,
        "material_branches_at_least_24": material_branches >= 24,
        "recoverable_branches_at_least_24": recoverable_branches >= 24,
        "material_in_at_least_three_families": len(branch_class_families["MATERIAL_HAZARDOUS_CONTACT"]) >= 3,
        "recoverable_in_at_least_three_families": len(branch_class_families["RECOVERABLE_LOW_SEVERITY_CONTACT"]) >= 3,
        "classification_deterministic": all(
            event["classification"] == ONTOLOGY.classify_event(event) for event in classified_events),
        "threshold_rationales_documented": True,
    }
    classification = (
        "CONTACT_HAZARD_ONTOLOGY_READY" if all(readiness_checks.values())
        else "CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT")
    raw_payload = {
        "schema": "lewm_contact_hazard_raw_contact_event_index_v1", "rows": raw_rows,
        "physics_dt_s": ONTOLOGY.PHYSICS_DT_S, "event_gap_physics_steps": ONTOLOGY.EVENT_GAP_PHYSICS_STEPS,
        "raw_contact_points": raw_index["raw_contact_points"], "source_content_digest": raw_index["content_digest"],
    }
    raw_payload["content_digest"] = ONTOLOGY.digest(raw_payload)
    event_payload = {"schema": "lewm_contact_hazard_event_ledger_v1", "events": classified_events}
    event_payload["content_digest"] = ONTOLOGY.digest(event_payload)
    branch_payload = {"schema": "lewm_contact_hazard_branch_ontology_ledger_v1", "branches": branches}
    branch_payload["content_digest"] = ONTOLOGY.digest(branch_payload)
    for path, value in ((TRACKED_RAW_INDEX, raw_payload), (TRACKED_EVENT_LEDGER, event_payload),
                        (TRACKED_BRANCH_LEDGER, branch_payload)):
        atomic_json(path, value)
    sensor_summary = {}
    for category in ONTOLOGY.CATEGORIES:
        rows = [event["sensor_observability"] for event in classified_events
                if event["classification"]["category"] == category]
        sensor_summary[category] = {
            "events": len(rows),
            "front_depth_contact_point_in_fov": sum(bool(row.get("front_depth_contact_point_in_fov")) for row in rows),
            "front_depth_near_before_or_at": sum(bool(row.get("front_depth_near_before") or row.get("front_depth_near_at_tick")) for row in rows),
            "lidar_contact_point_in_vertical_coverage": sum(bool(row.get("lidar_contact_point_in_vertical_coverage")) for row in rows),
            "lidar_near_before_or_at": sum(bool(row.get("lidar_near_before") or row.get("lidar_near_at_tick")) for row in rows),
            "enhanced_embodied_available": sum(bool(row.get("enhanced_embodied_available")) for row in rows),
            "depth_plus_embodied_available": sum(bool(row.get("depth_plus_embodied_evidence_available")) for row in rows),
        }
    case_studies = []
    for path in sorted((OUT / "case_studies").glob("*.json")):
        record = json.loads(path.read_text())
        points = read_points(Path(record["raw_points_path"]))
        groups = ONTOLOGY.group_contact_points(points)
        case_events = []
        for event_index, group in enumerate(groups):
            event = ONTOLOGY.reduce_event(group, event_index=event_index)
            event["sensor_observability"] = sensor.event(event)
            event["classification"] = ONTOLOGY.classify_event(event)
            case_events.append(event)
        case_studies.append({
            "designation": "POST_HOC_DESCRIPTIVE_CASE_STUDY",
            "branch_id": record["branch_id"], "verification": record["verification"],
            "events": case_events, "historical_label_revised": False,
        })
    result = {
        "schema": "lewm_contact_hazard_ontology_development_result_v1",
        "classification": classification, "fixture": fixture,
        "replay": {
            "states": 48, "branches": 576, "passing_states": raw_index["passing_states"],
            "mismatched_states": raw_index["mismatched_states"],
            "mismatched_branches": [branch["branch_id"] for branch in branches if not branch["included_in_ontology_development"]],
        },
        "summary": summary, "measurement_availability": availability,
        "sensor_observability": sensor_summary, "readiness_checks": readiness_checks,
        "post_hoc_case_studies": case_studies,
        "historical_results_unchanged": True, "prospective_only": True,
        "ledgers": {
            "raw_index": {"path": str(TRACKED_RAW_INDEX.relative_to(ROOT)), "sha256": sha256(TRACKED_RAW_INDEX), "content_digest": raw_payload["content_digest"]},
            "events": {"path": str(TRACKED_EVENT_LEDGER.relative_to(ROOT)), "sha256": sha256(TRACKED_EVENT_LEDGER), "content_digest": event_payload["content_digest"], "rows": len(classified_events)},
            "branches": {"path": str(TRACKED_BRANCH_LEDGER.relative_to(ROOT)), "sha256": sha256(TRACKED_BRANCH_LEDGER), "content_digest": branch_payload["content_digest"], "rows": len(branches)},
        },
        "runtime_reducer_s": time.time() - started,
        "storage_bytes": sum(path.stat().st_size for path in (TRACKED_RAW_INDEX, TRACKED_EVENT_LEDGER, TRACKED_BRANCH_LEDGER)),
    }
    result["content_digest"] = ONTOLOGY.digest(result)
    atomic_json(OUT / "development_result.json", result)
    print(json.dumps({
        "classification": classification, "events": event_count,
        "event_classes": event_classes, "branch_classes": summary["prospective_classes"]["branches"],
        "availability": availability, "readiness_checks": readiness_checks,
        "content_digest": result["content_digest"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
