#!/usr/bin/env python3
"""Derive and gate route-intent labels for the frozen V1 waypoint panel."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
sys.path[:0] = [str(ROOT / "lewm_worlds"), str(ROOT / "lewm_genesis")]
from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_graph import SceneGraph

DELTA_D = 0.03
DELTA_THETA = math.radians(5.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def route_heading(entry: dict) -> float:
    payload = json.loads((Path(entry["scene_dir"]) / "manifest.json").read_text())
    graph = SceneGraph(parse_scene_manifest_dict(payload))
    path = entry["waypoint_path_cells"]
    start, nxt = graph.cell_center(int(path[0])), graph.cell_center(int(path[1]))
    return math.atan2(float(nxt[1]) - float(start[1]), float(nxt[0]) - float(start[0]))


def preference(a: dict, b: dict) -> int:
    """Return 1 when a>b, -1 when b>a and 0 when unordered."""
    if a["safe"] != b["safe"]:
        return 1 if a["safe"] else -1
    if not a["safe"]:
        return 0
    if a["completed"] != b["completed"]:
        return 1 if a["completed"] else -1
    dd = a["p_d"] - b["p_d"]
    if abs(dd) > DELTA_D:
        return 1 if dd > 0 else -1
    dt = a["p_theta_rad"] - b["p_theta_rad"]
    if abs(dt) > DELTA_THETA:
        return 1 if dt > 0 else -1
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    split = json.loads((V1 / "split.json").read_text())
    split_of = {sid: name for name in ("fit", "calibration", "heldout") for sid in split[name]}
    entries = {entry["state_id"]: entry for entry in manifest["state_candidates"]}
    headings = {sid: route_heading(entry) for sid, entry in entries.items()}
    start_yaws = {sid: float(entry["start_pose"][1]) for sid, entry in entries.items()}
    rows = [json.loads(line) for line in (V1 / "branch_labels.jsonl").read_text().splitlines() if line]
    derived = []
    for row in rows:
        sid = row["state_id"]
        horizons = {}
        for horizon in ("1", "2", "3"):
            source = row["horizons"][horizon]
            start_error = abs(wrap(headings[sid] - start_yaws[sid]))
            end_error = abs(wrap(headings[sid] - float(source["pose"][2])))
            horizons[horizon] = {
                "p_d": float(source["progress"]),
                "p_theta_rad": float(start_error - end_error),
                "p_theta_deg": math.degrees(start_error - end_error),
                "completed": bool(source["completed"]),
                "safe": not bool(source["unsafe"]),
                "route_heading_world_rad": headings[sid],
                "heading_error_start_rad": start_error,
                "heading_error_end_rad": end_error,
            }
        derived.append({"branch_id": f"{sid}:{int(row['candidate_index']):02d}",
                        "state_id": sid, "candidate_index": int(row["candidate_index"]),
                        "family": row["family"], "split": split_of[sid], "horizons": horizons})

    by_state = defaultdict(list)
    for row in derived:
        by_state[row["state_id"]].append(row)
    classes = {}
    pair_counts = {}
    for sid, state_rows in by_state.items():
        labels = [row["horizons"]["3"] for row in state_rows]
        safe = [label for label in labels if label["safe"]]
        if not safe:
            state_class = "NO_SAFE_CANDIDATE"
        elif any(label["p_d"] > DELTA_D for label in safe):
            state_class = "TRANSLATIONAL_PROGRESS_AVAILABLE"
        elif any(label["p_theta_rad"] > DELTA_THETA for label in safe):
            state_class = "ALIGNMENT_PROGRESS_AVAILABLE"
        else:
            state_class = "SAFE_HOLD_OR_ABSTAIN"
        classes[sid] = state_class
        pair_counts[sid] = sum(preference(labels[i], labels[j]) != 0
                               for i in range(len(labels)) for j in range(i + 1, len(labels)))

    class_counts = Counter(classes.values())
    split_classes = {name: Counter(classes[sid] for sid in split[name]) for name in ("fit", "calibration", "heldout")}
    family_classes = defaultdict(Counter)
    for sid, state_class in classes.items():
        family_classes[entries[sid]["family"]][state_class] += 1

    split_rows = defaultdict(list)
    for row in derived:
        split_rows[row["split"]].append(row["horizons"]["3"])
    split_checks = {}
    for name, values in split_rows.items():
        p_d = [v["p_d"] for v in values]
        p_t = [v["p_theta_rad"] for v in values]
        safe = [v["safe"] for v in values]
        split_checks[name] = {
            "rows": len(values), "safe": sum(safe), "unsafe": len(safe) - sum(safe),
            "p_d_range": [min(p_d), max(p_d)], "p_theta_deg_range": [math.degrees(min(p_t)), math.degrees(max(p_t))],
            "safe_unsafe_nondegenerate": bool(any(safe) and not all(safe)),
            "progress_nondegenerate": bool(max(p_d) > min(p_d) or max(p_t) > min(p_t)),
        }
    improving = {"fit": sum(classes[s] in ("TRANSLATIONAL_PROGRESS_AVAILABLE", "ALIGNMENT_PROGRESS_AVAILABLE") for s in split["fit"]),
                 "heldout": sum(classes[s] in ("TRANSLATIONAL_PROGRESS_AVAILABLE", "ALIGNMENT_PROGRESS_AVAILABLE") for s in split["heldout"])}
    heldout_align_or_abstain = sum(classes[s] in ("ALIGNMENT_PROGRESS_AVAILABLE", "SAFE_HOLD_OR_ABSTAIN", "NO_SAFE_CANDIDATE") for s in split["heldout"])
    family_improving = {family: sum(count for cls, count in counts.items() if cls in ("TRANSLATIONAL_PROGRESS_AVAILABLE", "ALIGNMENT_PROGRESS_AVAILABLE"))
                        for family, counts in family_classes.items()}
    family_unsafe = {family: any(not row["horizons"]["3"]["safe"] for row in derived if row["family"] == family)
                     for family in family_classes}
    checks = {
        "split_safe_unsafe_nondegenerate": all(v["safe_unsafe_nondegenerate"] for v in split_checks.values()),
        "split_progress_nondegenerate": all(v["progress_nondegenerate"] for v in split_checks.values()),
        "fit_improving_states_at_least_20": improving["fit"] >= 20,
        "heldout_improving_states_at_least_4": improving["heldout"] >= 4,
        "heldout_alignment_or_abstention_at_least_2": heldout_align_or_abstain >= 2,
        "every_family_route_improving": all(value >= 1 for value in family_improving.values()),
        "every_family_has_unsafe": all(family_unsafe.values()),
    }
    audit = {
        "schema": "safe_local_waypoint_route_intent_v2_data_audit",
        "status": "POST_OUTCOME_DEVELOPMENT_SUCCESSOR",
        "source": {"manifest_sha256": sha(V1 / "state_manifest.json"),
                   "ledger_sha256": sha(V1 / "branch_labels.jsonl"), "split_sha256": sha(V1 / "split.json")},
        "margins": {"delta_d_m": DELTA_D, "delta_theta_deg": 5.0},
        "rows": len(derived), "states": len(by_state),
        "class_counts": dict(class_counts),
        "split_class_counts": {name: dict(value) for name, value in split_classes.items()},
        "family_class_counts": {name: dict(value) for name, value in family_classes.items()},
        "split_checks": split_checks, "improving_state_counts": improving,
        "heldout_alignment_or_abstention_states": heldout_align_or_abstain,
        "family_improving_states": family_improving, "family_has_unsafe": family_unsafe,
        "ordered_pair_counts": {"total": sum(pair_counts.values()), "per_state": pair_counts},
        "safety_component_audit": {
            "path_unsafe": sum(not row["horizons"]["3"]["safe"] for row in derived),
            "path_safe": sum(row["horizons"]["3"]["safe"] for row in derived),
            "component_fields_present": ["min_clearance", "path_unsafe"],
            "component_fields_missing": ["collision_or_disallowed_contact", "clearance_violation", "stuck", "fall", "unsafe_termination", "combinations"],
            "aggregate_consistency": "NOT_VERIFIABLE_FROM_V1_LEDGER; deterministic replay required before training",
        },
        "checks": checks, "passed": all(checks.values()),
        "classification_if_failed": "ROUTE_INTENT_DATA_INSUFFICIENT",
    }
    with (OUT / "route_intent_labels.jsonl").open("w") as handle:
        for row in derived:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
