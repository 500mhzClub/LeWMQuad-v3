#!/usr/bin/env python3
"""Enumerate the frozen contact/stuck decision-threshold frontier without inference."""
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
from scipy.stats import beta, ks_2samp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_COMMIT = "03de8a59eaeb87b50644ae528016803c4ce4e399"
OUT = ROOT / ".generated/joint_decision_level_safety_calibration_frontier_v1"
RECOVERY = ROOT / ".generated/mechanism_specific_safety_composition_inference_recovery_v1"
LEDGER = RECOVERY / "row_level_component_predictions_v1.npz"
LEDGER_INDEX = RECOVERY / "row_level_component_predictions_v1_index.json"
PRIOR_RESULT = RECOVERY / "result.json"
EXPECTED_CONTENT_DIGEST = "e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de"
EXPECTED_LEDGER_SHA = "a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08"
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
DELTA_D = .03
DELTA_THETA = math.radians(5.)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def array_content_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_ledger() -> tuple[dict[str, np.ndarray], dict]:
    if sha(LEDGER) != EXPECTED_LEDGER_SHA:
        raise RuntimeError("frozen row ledger SHA-256 mismatch")
    index = json.loads(LEDGER_INDEX.read_text())
    with np.load(LEDGER, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if array_content_digest(arrays) != EXPECTED_CONTENT_DIGEST:
        raise RuntimeError("frozen row ledger decoded-content digest mismatch")
    if index["array_content_digest"] != EXPECTED_CONTENT_DIGEST or index["file_sha256"] != EXPECTED_LEDGER_SHA:
        raise RuntimeError("frozen row ledger index mismatch")
    required = {
        "branch_id", "state_id", "candidate_index", "split", "family", "labels",
        "primary_contact_probability", "primary_stuck_probability", "primary_admitted",
    }
    if not required.issubset(arrays) or len(arrays["branch_id"]) != 576:
        raise RuntimeError("frozen row ledger schema/cardinality mismatch")
    return arrays, index


def integrate(post_slew: list, waypoint: list[float]) -> np.ndarray:
    x = y = yaw = 0.
    for vx, vy, wz in [tick for block in post_slew[:3] for tick in block]:
        x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * .1
        y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * .1
        yaw = math.atan2(math.sin(yaw + float(wz) * .1), math.cos(yaw + float(wz) * .1))
    wx, wy, sg, cg = waypoint
    goal_heading = math.atan2(sg, cg)
    p_d = math.hypot(wx, wy) - math.hypot(wx - x, wy - y)
    p_theta = abs(math.atan2(math.sin(goal_heading), math.cos(goal_heading))) - abs(
        math.atan2(math.sin(goal_heading - yaw), math.cos(goal_heading - yaw)))
    return np.asarray([x, y, math.sin(yaw), math.cos(yaw), p_d, p_theta], np.float32)


def load_route_rows() -> list[dict]:
    branches = [json.loads(line) for line in (V1 / "branch_labels.jsonl").read_text().splitlines()]
    route = [json.loads(line) for line in (V2 / "route_intent_labels.jsonl").read_text().splitlines()]
    route_by = {row["branch_id"]: row for row in route}
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    states = {row["state_id"]: row for row in manifest["state_candidates"]}
    rows = []
    for branch in branches:
        state_id = branch["state_id"]
        candidate = int(branch["candidate_index"])
        route_row = route_by[f"{state_id}:{candidate:02d}"]
        state = states[state_id]
        start_yaw = float(state["start_pose"][1])
        body = state.get("waypoint_body_xy")
        if body is None:
            worlds_root = str(ROOT / "lewm_worlds")
            if worlds_root not in sys.path:
                sys.path.insert(0, worlds_root)
            from lewm_worlds.manifest import parse_scene_manifest_dict
            from lewm_worlds.scene_graph import SceneGraph
            graph = SceneGraph(parse_scene_manifest_dict(json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())))
            waypoint_x, waypoint_y = graph.cell_center(int(state["waypoint_path_cells"][2]))
            start_x, start_y = map(float, state["start_pose"][0])
            delta_x, delta_y = waypoint_x - start_x, waypoint_y - start_y
            body = [math.cos(start_yaw) * delta_x + math.sin(start_yaw) * delta_y,
                    -math.sin(start_yaw) * delta_x + math.cos(start_yaw) * delta_y]
        heading = float(route_row["horizons"]["3"]["route_heading_world_rad"]) - start_yaw
        waypoint = [float(body[0]), float(body[1]), math.sin(heading), math.cos(heading)]
        rows.append({
            "state_id": state_id, "candidate_index": candidate, "family": branch["family"],
            "split": route_row["split"], "kinematic": integrate(branch["post_slew"], waypoint),
            "p_d": float(route_row["horizons"]["3"]["p_d"]),
            "p_theta": float(route_row["horizons"]["3"]["p_theta_rad"]),
            "unsafe": bool(branch["horizons"]["3"]["unsafe"]),
        })
    if len(rows) != 576:
        raise RuntimeError("frozen route row cardinality mismatch")
    return rows


def preference(a: dict, b: dict) -> int:
    if a["unsafe"] != b["unsafe"]:
        return 1 if not a["unsafe"] else -1
    if a["unsafe"]:
        return 0
    distance = a["p_d"] - b["p_d"]
    if abs(distance) > DELTA_D:
        return 1 if distance > 0 else -1
    heading = a["p_theta"] - b["p_theta"]
    if abs(heading) > DELTA_THETA:
        return 1 if heading > 0 else -1
    return 0


def best_safe(ids: list[int], rows: list[dict]) -> int | None:
    safe = [index for index in ids if not rows[index]["unsafe"]]
    if not safe:
        return None
    best = safe[0]
    for index in safe[1:]:
        if preference(rows[index], rows[best]) > 0:
            best = index
    return best


def route_order(ids: list[int], kinematic: np.ndarray, rows: list[dict]) -> list[int]:
    remaining = list(ids)
    order = []
    while remaining:
        best = max(float(kinematic[index, 4]) for index in remaining)
        near = [index for index in remaining if best - float(kinematic[index, 4]) <= DELTA_D]
        pick = min(near, key=lambda index: (-float(kinematic[index, 5]), rows[index]["candidate_index"]))
        order.append(pick)
        remaining.remove(pick)
    return order


def bind_route_rows(arrays: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    route_rows = load_route_rows()
    route_by_id = {f"{row['state_id']}:{int(row['candidate_index']):02d}": i for i, row in enumerate(route_rows)}
    route_indices = np.asarray([route_by_id[str(value)] for value in arrays["branch_id"]], dtype=np.int32)
    for ledger_index, route_index in enumerate(route_indices):
        route = route_rows[int(route_index)]
        checks = (
            str(arrays["state_id"][ledger_index]) == route["state_id"],
            int(arrays["candidate_index"][ledger_index]) == int(route["candidate_index"]),
            str(arrays["split"][ledger_index]) == route["split"],
            str(arrays["family"][ledger_index]) == route["family"],
            bool(arrays["labels"][ledger_index, -1, 4]) == bool(route["unsafe"]),
        )
        if not all(checks):
            raise RuntimeError(f"row alignment mismatch at {arrays['branch_id'][ledger_index]}")
    split_indices = {
        split: np.asarray([i for i, value in enumerate(arrays["split"]) if str(value) == split], dtype=np.int32)
        for split in ("fit", "calibration", "heldout")
    }
    report = {
        "rows": 576,
        "states": len(set(map(str, arrays["state_id"]))),
        "split_rows": {key: int(len(value)) for key, value in split_indices.items()},
        "route_identity_matches": 576,
        "component_and_aggregate_label_matches": 576,
        "ledger_sha256": EXPECTED_LEDGER_SHA,
        "ledger_content_digest": EXPECTED_CONTENT_DIGEST,
    }
    return route_rows, route_indices, {**split_indices, "report": report}


def threshold_values(probability: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(probability, dtype=np.float64))
    return np.concatenate((np.asarray([np.nextafter(0.0, -np.inf)]), values,
                           np.asarray([np.nextafter(1.0, np.inf)])))


def clopper_pearson(k: int, n: int, confidence: float = .95) -> list[float | None]:
    if n == 0:
        return [None, None]
    alpha = 1. - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2., k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1. - alpha / 2., k + 1, n - k))
    return [lower, upper]


def one_sided_zero_miss_sample_size(upper: float = .05, confidence: float = .95) -> int:
    return int(math.ceil(math.log(1. - confidence) / math.log(1. - upper)))


def prepare_states(route_rows: list[dict], global_indices: np.ndarray, kinematic: np.ndarray) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for route_index in global_indices:
        grouped[route_rows[int(route_index)]["state_id"]].append(int(route_index))
    states = []
    for state_id in sorted(grouped, key=lambda value: int(value.split("-")[1])):
        ids = grouped[state_id]
        safe = [index for index in ids if not route_rows[index]["unsafe"]]
        best = best_safe(ids, route_rows)
        states.append({
            "state_id": state_id,
            "family": route_rows[ids[0]]["family"],
            "ids": ids,
            "rank_order": route_order(ids, kinematic, route_rows),
            "safe": safe,
            "best": best,
        })
    return states


def evaluate_pair(contact: np.ndarray, stuck: np.ndarray, contact_labels: np.ndarray,
                  stuck_labels: np.ndarray, aggregate_labels: np.ndarray,
                  ledger_indices: np.ndarray, route_indices: np.ndarray, route_rows: list[dict],
                  states: list[dict], t_contact: float, t_stuck: float,
                  include_state_rows: bool = False) -> dict:
    admitted = (contact < t_contact) & (stuck < t_stuck)
    unsafe = aggregate_labels.astype(bool)
    contact_positive = contact_labels.astype(bool)
    stuck_positive = stuck_labels.astype(bool)
    admitted_by_route = {int(route_index): bool(admitted[local]) for local, route_index in enumerate(route_indices)}
    selected_progress: list[float] = []
    selected_heading: list[float] = []
    selected_unsafe: list[bool] = []
    regrets: list[float] = []
    top1: list[bool] = []
    top3: list[bool] = []
    state_rows = []
    false_abstention = correct_abstention = 0
    for state in states:
        admitted_rank = [index for index in state["rank_order"] if admitted_by_route[index]]
        pick = admitted_rank[0] if admitted_rank else None
        if pick is None:
            if state["safe"]:
                false_abstention += 1
            else:
                correct_abstention += 1
        else:
            selected_progress.append(float(route_rows[pick]["p_d"]))
            selected_heading.append(math.degrees(float(route_rows[pick]["p_theta"])))
            selected_unsafe.append(bool(route_rows[pick]["unsafe"]))
        if state["best"] is not None:
            top1.append(pick == state["best"])
            top3.append(state["best"] in admitted_rank[:3])
            if pick is not None and not route_rows[pick]["unsafe"] and len(state["safe"]) >= 2:
                values = [float(route_rows[index]["p_d"]) for index in state["safe"]]
                spread = max(values) - min(values)
                if spread > 1e-8:
                    regrets.append((float(route_rows[state["best"]]["p_d"]) - float(route_rows[pick]["p_d"])) / spread)
        if include_state_rows:
            admitted_ids = [index for index in state["ids"] if admitted_by_route[index]]
            state_rows.append({
                "state_id": state["state_id"], "family": state["family"],
                "admitted": len(admitted_ids),
                "admitted_safe": sum(not route_rows[index]["unsafe"] for index in admitted_ids),
                "admitted_unsafe": sum(route_rows[index]["unsafe"] for index in admitted_ids),
                "selected_candidate": None if pick is None else int(route_rows[pick]["candidate_index"]),
                "selected_safe": None if pick is None else not bool(route_rows[pick]["unsafe"]),
                "selected_p_d": None if pick is None else float(route_rows[pick]["p_d"]),
                "selected_p_theta_deg": None if pick is None else math.degrees(float(route_rows[pick]["p_theta"])),
            })
    rejected = ~admitted
    result = {
        "contact_recall": float(np.mean(rejected[contact_positive])) if contact_positive.any() else None,
        "contact_false_negative_rate": float(np.mean(admitted[contact_positive])) if contact_positive.any() else None,
        "stuck_recall": float(np.mean(rejected[stuck_positive])) if stuck_positive.any() else None,
        "stuck_false_negative_rate": float(np.mean(admitted[stuck_positive])) if stuck_positive.any() else None,
        "aggregate_unsafe_recall": float(np.mean(rejected[unsafe])) if unsafe.any() else None,
        "aggregate_false_negative_rate": float(np.mean(admitted[unsafe])) if unsafe.any() else None,
        "admitted_safe_count": int(np.sum(admitted & ~unsafe)),
        "admitted_unsafe_count": int(np.sum(admitted & unsafe)),
        "safe_candidate_retention": float(np.mean(admitted[~unsafe])) if (~unsafe).any() else None,
        "states_retaining_safe": sum(row["admitted_safe"] > 0 for row in state_rows) if include_state_rows else sum(
            any(admitted_by_route[index] for index in state["safe"]) for state in states),
        "states_only_unsafe_admitted": sum(
            any(admitted_by_route[index] for index in state["ids"])
            and not any(admitted_by_route[index] for index in state["safe"]) for state in states),
        "states_no_admitted": sum(not any(admitted_by_route[index] for index in state["ids"]) for state in states),
        "selected_unsafe_count": int(sum(selected_unsafe)),
        "selected_unsafe_rate": float(np.mean(selected_unsafe)) if selected_unsafe else 0.0,
        "false_abstentions": false_abstention,
        "correct_abstentions": correct_abstention,
        "mean_selected_route_progress_m": float(np.mean(selected_progress)) if selected_progress else 0.0,
        "mean_selected_heading_improvement_deg": float(np.mean(selected_heading)) if selected_heading else 0.0,
        "normalized_safe_progress_regret": float(np.mean(regrets)) if regrets else None,
        "normalized_regret_states": len(regrets),
        "best_safe_top1": float(np.mean(top1)) if top1 else None,
        "best_safe_top3": float(np.mean(top3)) if top3 else None,
        "selected_state_count": len(selected_progress),
    }
    if include_state_rows:
        result["per_state"] = state_rows
    return result


FRONTIER_FIELDS = (
    "contact_recall", "stuck_recall", "aggregate_unsafe_recall", "aggregate_false_negative_rate",
    "admitted_safe_count", "admitted_unsafe_count", "safe_candidate_retention", "states_retaining_safe",
    "states_only_unsafe_admitted", "states_no_admitted", "selected_unsafe_count", "false_abstentions",
    "correct_abstentions", "mean_selected_route_progress_m", "normalized_safe_progress_regret",
    "best_safe_top1", "best_safe_top3",
)


def enumerate_frontier(contact: np.ndarray, stuck: np.ndarray, contact_labels: np.ndarray,
                       stuck_labels: np.ndarray, aggregate_labels: np.ndarray,
                       ledger_indices: np.ndarray, route_indices: np.ndarray, route_rows: list[dict],
                       states: list[dict]) -> dict[str, np.ndarray]:
    contact_thresholds = threshold_values(contact)
    stuck_thresholds = threshold_values(stuck)
    size = len(contact_thresholds) * len(stuck_thresholds)
    arrays = {
        "contact_threshold": np.empty(size, dtype=np.float64),
        "stuck_threshold": np.empty(size, dtype=np.float64),
    }
    for field in FRONTIER_FIELDS:
        arrays[field] = np.empty(size, dtype=np.float64)
    cursor = 0
    for t_contact in contact_thresholds:
        for t_stuck in stuck_thresholds:
            result = evaluate_pair(contact, stuck, contact_labels, stuck_labels, aggregate_labels,
                                   ledger_indices, route_indices, route_rows, states,
                                   float(t_contact), float(t_stuck))
            arrays["contact_threshold"][cursor] = t_contact
            arrays["stuck_threshold"][cursor] = t_stuck
            for field in FRONTIER_FIELDS:
                value = result[field]
                arrays[field][cursor] = np.nan if value is None else value
            cursor += 1
    return arrays


def eligible_mask(frontier: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (frontier["aggregate_unsafe_recall"] >= .95)
        & (frontier["aggregate_false_negative_rate"] <= .05)
        & (frontier["contact_recall"] >= .90)
        & (frontier["stuck_recall"] >= .90)
        & (frontier["states_only_unsafe_admitted"] == 0)
        & (frontier["selected_unsafe_count"] == 0)
    )


def choose_primary(frontier: dict[str, np.ndarray]) -> int | None:
    eligible = np.flatnonzero(eligible_mask(frontier))
    if not len(eligible):
        return None
    def key(index: int) -> tuple:
        regret = frontier["normalized_safe_progress_regret"][index]
        return (
            frontier["states_retaining_safe"][index],
            frontier["safe_candidate_retention"][index],
            frontier["mean_selected_route_progress_m"][index],
            -regret if np.isfinite(regret) else -math.inf,
            -frontier["false_abstentions"][index],
            frontier["best_safe_top3"][index],
            -frontier["contact_threshold"][index],
            -frontier["stuck_threshold"][index],
        )
    return int(max(eligible, key=key))


def complete_gate(result: dict, oracle_progress: float, per_family: dict) -> dict:
    no_family_collapse = all(
        row["states_retaining_safe"] >= 1 and row["selected_state_count"] >= 1
        for row in per_family.values()
    )
    checks = {
        "aggregate_recall_ge_0_95": result["aggregate_unsafe_recall"] >= .95,
        "aggregate_fnr_le_0_05": result["aggregate_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": result["safe_candidate_retention"] >= .40,
        "contact_recall_ge_0_90": result["contact_recall"] >= .90,
        "stuck_recall_ge_0_90": result["stuck_recall"] >= .90,
        "six_states_retain_safe": result["states_retaining_safe"] >= 6,
        "no_state_only_unsafe_admitted": result["states_only_unsafe_admitted"] == 0,
        "selected_unsafe_rate_zero": result["selected_unsafe_rate"] == 0.,
        "false_abstention_le_1": result["false_abstentions"] <= 1,
        "progress_ge_80pct_oracle": result["mean_selected_route_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": result["normalized_safe_progress_regret"] is not None and result["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": result["best_safe_top3"] is not None and result["best_safe_top3"] >= .75,
        "no_complete_family_collapse": no_family_collapse,
    }
    return {"passed": all(checks.values()), "checks": checks}


def pareto_mask(x: np.ndarray, y: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if eligible is not None:
        valid &= eligible
    indices = np.flatnonzero(valid)
    mask = np.zeros(len(x), dtype=bool)
    for index in indices:
        dominated = np.any((x[indices] >= x[index]) & (y[indices] >= y[index])
                           & ((x[indices] > x[index]) | (y[indices] > y[index])))
        mask[index] = not dominated
    return mask


def result_at(frontier: dict[str, np.ndarray], index: int) -> dict:
    result = {key: float(value[index]) for key, value in frontier.items() if key not in ("risk_retention_pareto", "risk_progress_pareto")}
    for field in ("admitted_safe_count", "admitted_unsafe_count", "states_retaining_safe", "states_only_unsafe_admitted",
                  "states_no_admitted", "selected_unsafe_count", "false_abstentions", "correct_abstentions"):
        result[field] = int(result[field])
    if math.isnan(result["normalized_safe_progress_regret"]):
        result["normalized_safe_progress_regret"] = None
    return result


def family_metrics(contact: np.ndarray, stuck: np.ndarray, contact_labels: np.ndarray, stuck_labels: np.ndarray,
                   aggregate_labels: np.ndarray, ledger_indices: np.ndarray, route_indices: np.ndarray,
                   route_rows: list[dict], kinematic: np.ndarray, t_contact: float, t_stuck: float) -> dict:
    output = {}
    families = np.asarray([route_rows[int(index)]["family"] for index in route_indices])
    for family in FAMILIES:
        mask = families == family
        sub_ledger = ledger_indices[mask]
        sub_route = route_indices[mask]
        states = prepare_states(route_rows, sub_route, kinematic)
        output[family] = evaluate_pair(contact[mask], stuck[mask], contact_labels[mask], stuck_labels[mask],
                                       aggregate_labels[mask], sub_ledger, sub_route, route_rows, states,
                                       t_contact, t_stuck, include_state_rows=True)
    return output


def bootstrap_state_intervals(per_state: list[dict], seed: int = 2026082010, draws: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(per_state)
    retain = np.asarray([row["admitted_safe"] > 0 for row in per_state], float)
    false_abstain = np.asarray([row["selected_candidate"] is None and row["admitted_safe"] == 0 for row in per_state], float)
    progress = np.asarray([0. if row["selected_p_d"] is None else row["selected_p_d"] for row in per_state], float)
    samples = rng.integers(0, n, size=(draws, n))
    def interval(values: np.ndarray) -> list[float]:
        distribution = values[samples].mean(axis=1)
        return [float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))]
    return {"draws": draws, "seed": seed, "state_retention_fraction_95pct": interval(retain),
            "false_abstention_fraction_95pct": interval(false_abstain),
            "progress_with_abstention_as_zero_m_95pct": interval(progress)}


def score_shift(calibration: np.ndarray, heldout: np.ndarray, calibration_labels: np.ndarray,
                heldout_labels: np.ndarray) -> dict:
    output = {}
    for name, cal_mask, held_mask in (
        ("all", np.ones(len(calibration), bool), np.ones(len(heldout), bool)),
        ("positive", calibration_labels.astype(bool), heldout_labels.astype(bool)),
        ("negative", ~calibration_labels.astype(bool), ~heldout_labels.astype(bool)),
    ):
        cal_values, held_values = calibration[cal_mask], heldout[held_mask]
        output[name] = {
            "calibration_n": len(cal_values), "heldout_n": len(held_values),
            "calibration_mean": float(np.mean(cal_values)), "heldout_mean": float(np.mean(held_values)),
            "calibration_median": float(np.median(cal_values)), "heldout_median": float(np.median(held_values)),
            "ks_statistic": float(ks_2samp(cal_values, held_values).statistic),
        }
    return output


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, ledger_index = load_ledger()
    route_rows, route_index_all, split = bind_route_rows(arrays)
    kinematic = np.stack([row["kinematic"] for row in route_rows])
    contact_all = arrays["primary_contact_probability"][:, -1, 2].astype(np.float64)
    stuck_all = arrays["primary_stuck_probability"][:, -1, 3].astype(np.float64)
    contact_labels_all = arrays["labels"][:, -1, 2].astype(bool)
    stuck_labels_all = arrays["labels"][:, -1, 3].astype(bool)
    unsafe_all = arrays["labels"][:, -1, 4].astype(bool)

    prior = json.loads(PRIOR_RESULT.read_text())
    prior_contact_threshold = prior["component_calibration"]["ENHANCED_EMBODIED"]["contact"]["threshold"]
    prior_stuck_threshold = prior["component_calibration"]["ACTION_CONTROL_ONLY"]["stuck"]["threshold"]
    held = split["heldout"]
    held_route = route_index_all[held]
    held_states = prepare_states(route_rows, held_route, kinematic)
    reproduction = evaluate_pair(contact_all[held], stuck_all[held], contact_labels_all[held], stuck_labels_all[held],
                                 unsafe_all[held], held, held_route, route_rows, held_states,
                                 prior_contact_threshold, prior_stuck_threshold, include_state_rows=True)
    expected = prior["primary_composition"]
    reproduction_checks = {
        "aggregate_unsafe_recall": np.isclose(reproduction["aggregate_unsafe_recall"], expected["aggregate"]["unsafe_recall"]),
        "aggregate_fnr": np.isclose(reproduction["aggregate_false_negative_rate"], expected["aggregate"]["unsafe_false_negative_rate"]),
        "safe_retention": np.isclose(reproduction["safe_candidate_retention"], expected["aggregate"]["safe_candidate_retention"]),
        "states_retaining_safe": reproduction["states_retaining_safe"] == expected["planning"]["states_retaining_safe"],
        "selected_unsafe_rate": reproduction["selected_unsafe_rate"] == expected["planning"]["selected_unsafe_rate"],
        "selected_progress": np.isclose(reproduction["mean_selected_route_progress_m"], expected["planning"]["mean_selected_distance_progress_m"]),
        "normalized_regret": np.isclose(reproduction["normalized_safe_progress_regret"], expected["planning"]["normalized_safe_progress_regret"]),
        "best_safe_top3": np.isclose(reproduction["best_safe_top3"], expected["planning"]["best_safe_top3"]),
        "selected_candidates": [row["selected_candidate"] for row in reproduction["per_state"]] == [row["selected_candidate"] for row in expected["planning"]["per_state"]],
    }
    if not all(reproduction_checks.values()):
        raise RuntimeError(f"committed specialist composition did not reproduce: {reproduction_checks}")

    cal = split["calibration"]
    cal_route = route_index_all[cal]
    cal_states = prepare_states(route_rows, cal_route, kinematic)
    calibration_frontier = enumerate_frontier(contact_all[cal], stuck_all[cal], contact_labels_all[cal],
                                               stuck_labels_all[cal], unsafe_all[cal], cal, cal_route,
                                               route_rows, cal_states)
    chosen_index = choose_primary(calibration_frontier)
    calibration_feasible = chosen_index is not None
    selected_thresholds = None if chosen_index is None else {
        "contact": float(calibration_frontier["contact_threshold"][chosen_index]),
        "stuck": float(calibration_frontier["stuck_threshold"][chosen_index]),
    }
    calibration_selected = None if chosen_index is None else result_at(calibration_frontier, chosen_index)

    oracle_probability = np.asarray([float(route_rows[int(index)]["unsafe"]) for index in held_route])
    oracle = evaluate_pair(oracle_probability, np.zeros(len(held), dtype=float),
                           unsafe_all[held], np.zeros(len(held), dtype=bool), unsafe_all[held],
                           held, held_route, route_rows, held_states, .5, np.nextafter(1., np.inf),
                           include_state_rows=True)
    oracle_progress = oracle["mean_selected_route_progress_m"]
    heldout_primary = heldout_family = heldout_gate = None
    if selected_thresholds is not None:
        heldout_primary = evaluate_pair(contact_all[held], stuck_all[held], contact_labels_all[held], stuck_labels_all[held],
                                        unsafe_all[held], held, held_route, route_rows, held_states,
                                        selected_thresholds["contact"], selected_thresholds["stuck"], include_state_rows=True)
        heldout_primary["oracle_progress_fraction"] = heldout_primary["mean_selected_route_progress_m"] / oracle_progress
        heldout_family = family_metrics(contact_all[held], stuck_all[held], contact_labels_all[held], stuck_labels_all[held],
                                        unsafe_all[held], held, held_route, route_rows, kinematic,
                                        selected_thresholds["contact"], selected_thresholds["stuck"])
        heldout_gate = complete_gate(heldout_primary, oracle_progress, heldout_family)

    heldout_frontier = enumerate_frontier(contact_all[held], stuck_all[held], contact_labels_all[held],
                                          stuck_labels_all[held], unsafe_all[held], held, held_route,
                                          route_rows, held_states)
    recall_mask = heldout_frontier["aggregate_unsafe_recall"] >= .95
    zero_selected_unsafe = heldout_frontier["selected_unsafe_count"] == 0
    heldout_frontier["risk_retention_pareto"] = pareto_mask(heldout_frontier["aggregate_unsafe_recall"], heldout_frontier["safe_candidate_retention"])
    heldout_frontier["risk_progress_pareto"] = pareto_mask(heldout_frontier["aggregate_unsafe_recall"], heldout_frontier["mean_selected_route_progress_m"], zero_selected_unsafe)
    calibration_frontier["risk_retention_pareto"] = pareto_mask(calibration_frontier["aggregate_unsafe_recall"], calibration_frontier["safe_candidate_retention"])
    calibration_frontier["risk_progress_pareto"] = pareto_mask(calibration_frontier["aggregate_unsafe_recall"], calibration_frontier["mean_selected_route_progress_m"], calibration_frontier["selected_unsafe_count"] == 0)

    full_gate_indices = []
    for index in range(len(heldout_frontier["contact_threshold"])):
        candidate = result_at(heldout_frontier, index)
        if candidate["aggregate_unsafe_recall"] < .95:
            continue
        detailed = evaluate_pair(contact_all[held], stuck_all[held], contact_labels_all[held], stuck_labels_all[held],
                                 unsafe_all[held], held, held_route, route_rows, held_states,
                                 candidate["contact_threshold"], candidate["stuck_threshold"], include_state_rows=True)
        family = family_metrics(contact_all[held], stuck_all[held], contact_labels_all[held], stuck_labels_all[held],
                                unsafe_all[held], held, held_route, route_rows, kinematic,
                                candidate["contact_threshold"], candidate["stuck_threshold"])
        if complete_gate(detailed, oracle_progress, family)["passed"]:
            full_gate_indices.append(index)

    def argmax(mask: np.ndarray, field: str) -> dict | None:
        indices = np.flatnonzero(mask & np.isfinite(heldout_frontier[field]))
        return None if not len(indices) else result_at(heldout_frontier, int(indices[np.argmax(heldout_frontier[field][indices])]))
    def argmin(mask: np.ndarray, field: str) -> dict | None:
        indices = np.flatnonzero(mask & np.isfinite(heldout_frontier[field]))
        return None if not len(indices) else result_at(heldout_frontier, int(indices[np.argmin(heldout_frontier[field][indices])]))
    oracle_summary = {
        "status": "POST_HOC_ORACLE_FRONTIER_DIAGNOSTIC",
        "threshold_pairs": len(heldout_frontier["contact_threshold"]),
        "maximum_safe_retention_at_recall_ge_0_95": argmax(recall_mask, "safe_candidate_retention"),
        "maximum_states_retaining_safe_at_recall_ge_0_95": argmax(recall_mask, "states_retaining_safe"),
        "maximum_route_progress_with_zero_selected_unsafe": argmax(zero_selected_unsafe, "mean_selected_route_progress_m"),
        "minimum_regret_at_recall_ge_0_95": argmin(recall_mask, "normalized_safe_progress_regret"),
        "maximum_best_safe_top3_at_recall_ge_0_95": argmax(recall_mask, "best_safe_top3"),
        "complete_gate_pair_count": len(full_gate_indices),
        "any_complete_gate_pair": bool(full_gate_indices),
        "complete_gate_examples": [result_at(heldout_frontier, index) for index in full_gate_indices[:10]],
        "risk_retention_pareto_points": int(heldout_frontier["risk_retention_pareto"].sum()),
        "risk_progress_pareto_points": int(heldout_frontier["risk_progress_pareto"].sum()),
    }

    if heldout_gate is not None and heldout_gate["passed"]:
        classification = "JOINT_DECISION_CALIBRATION_SIGNAL"
        recommendation = "Validate the frozen joint decision rule on a small fresh, prospectively frozen panel; do not train a new safety model yet."
    elif oracle_summary["any_complete_gate_pair"]:
        classification = "CALIBRATION_DATA_OR_SELECTION_BOTTLENECK"
        recommendation = "Use a larger fresh calibration panel with prospectively specified decision-level or action-conditional conformal calibration while preserving both specialists."
    else:
        classification = "SPECIALIST_SCORE_FRONTIER_NO_GO"
        recommendation = "Prospectively train FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1 with separate contact and stuck temporal states, row-level persistence, and a fresh frozen panel."

    frontier_path = OUT / "joint_threshold_frontiers_v1.npz"
    frontier_arrays = {f"calibration_{key}": value for key, value in calibration_frontier.items()}
    frontier_arrays.update({f"heldout_{key}": value for key, value in heldout_frontier.items()})
    atomic_npz(frontier_path, **frontier_arrays)
    frontier_index = {
        "schema": "joint_decision_threshold_frontiers_v1",
        "calibration_threshold_pairs": len(calibration_frontier["contact_threshold"]),
        "heldout_threshold_pairs": len(heldout_frontier["contact_threshold"]),
        "array_content_digest": array_content_digest(frontier_arrays),
        "file_sha256": sha(frontier_path), "storage_bytes": frontier_path.stat().st_size,
        "heldout_status": "POST_HOC_ORACLE_FRONTIER_DIAGNOSTIC",
    }
    atomic_json(OUT / "joint_threshold_frontiers_v1_index.json", frontier_index)

    selected_uncertainty = None
    shifts = {
        "contact": score_shift(contact_all[cal], contact_all[held], contact_labels_all[cal], contact_labels_all[held]),
        "stuck": score_shift(stuck_all[cal], stuck_all[held], stuck_labels_all[cal], stuck_labels_all[held]),
    }
    if heldout_primary is not None:
        misses = heldout_primary["admitted_unsafe_count"]
        unsafe_count = int(unsafe_all[held].sum())
        cal_misses = calibration_selected["admitted_unsafe_count"]
        cal_unsafe_count = int(unsafe_all[cal].sum())
        selected_uncertainty = {
            "calibration_branch_counts": {"total": len(cal), "unsafe": cal_unsafe_count, "safe": int((~unsafe_all[cal]).sum()), "unsafe_misses": cal_misses},
            "heldout_branch_counts": {"total": len(held), "unsafe": unsafe_count, "safe": int((~unsafe_all[held]).sum()), "unsafe_misses": misses},
            "calibration_fnr_clopper_pearson_95pct": clopper_pearson(cal_misses, cal_unsafe_count),
            "heldout_fnr_clopper_pearson_95pct": clopper_pearson(misses, unsafe_count),
            "heldout_state_bootstrap": bootstrap_state_intervals(heldout_primary["per_state"]),
            "zero_miss_one_sided_95pct_upper_0_05_required_unsafe_examples": one_sided_zero_miss_sample_size(),
            "finite_sample_safety_guarantee": False,
        }

    result = {
        "schema": "joint_decision_level_safety_calibration_frontier_v1_result",
        "source_commit": SOURCE_COMMIT,
        "status": "POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC",
        "preserved": "MECHANISM_SPECIFIC_SAFETY_COMPOSITION_NO_SIGNAL",
        "ledger": {**split["report"], "index_sha256": sha(LEDGER_INDEX), "storage_bytes": LEDGER.stat().st_size},
        "ledger_reproduction": {"passed": all(reproduction_checks.values()), "checks": reproduction_checks,
                                "thresholds": {"contact": prior_contact_threshold, "stuck": prior_stuck_threshold},
                                "metrics": reproduction},
        "specialists": {"contact": "ENHANCED_EMBODIED contact score", "stuck": "ACTION_CONTROL_ONLY stuck score",
                        "probabilities_or_temperatures_changed": False},
        "calibration_frontier": {"threshold_pairs": len(calibration_frontier["contact_threshold"]),
                                 "eligible_pairs": int(eligible_mask(calibration_frontier).sum()),
                                 "feasibility": calibration_feasible,
                                 "failure_label_if_infeasible": None if calibration_feasible else "JOINT_CALIBRATION_FEASIBILITY_FAILURE",
                                 "selected_thresholds": selected_thresholds, "selected_metrics": calibration_selected,
                                 "selection_rule": ["states_retaining_safe max", "safe_retention max", "route_progress max",
                                                    "normalized_regret min", "false_abstentions min", "top3 max",
                                                    "contact threshold conservative", "stuck threshold conservative"]},
        "heldout_primary": heldout_primary,
        "heldout_per_family": heldout_family,
        "heldout_gate": heldout_gate,
        "heldout_oracle_frontier": oracle_summary,
        "oracle_safety_kinematic": oracle,
        "frontier_artifact": frontier_index,
        "uncertainty": selected_uncertainty,
        "score_distribution_shift": shifts,
        "classification": classification,
        "bottleneck": "component decision calibration" if classification == "JOINT_DECISION_CALIBRATION_SIGNAL" else (
            "calibration data or selection" if classification == "CALIBRATION_DATA_OR_SELECTION_BOTTLENECK" else "specialist score quality/frontier"),
        "interpretation": "A safety filter is not useful merely because it attains high unsafe recall. It must retain enough safe actions to permit the safety-related task to proceed.",
        "claim_boundary": "Eight calibration states cannot provide a finite-sample safety guarantee; held-out oracle-frontier results are post-hoc diagnostics only.",
        "recommendation": recommendation,
        "runtime": {"total_s": time.time() - started, "frontier_storage_bytes": frontier_path.stat().st_size, "result_storage_bytes": 0},
        "custody": {"models_trained": 0, "model_inference": False, "checkpoint_access": False, "simulation": False,
                    "rendering": False, "encoding": False, "jepa_predictor_access": False, "specialists_changed": False,
                    "route_ranker_changed": False},
    }
    result_path = OUT / "result.json"
    atomic_json(result_path, result)
    result["runtime"]["result_storage_bytes"] = result_path.stat().st_size
    atomic_json(result_path, result)
    print(json.dumps({"classification": classification, "calibration_pairs": len(calibration_frontier["contact_threshold"]),
                      "heldout_pairs": len(heldout_frontier["contact_threshold"]), "selected_thresholds": selected_thresholds,
                      "oracle_complete_gate_pairs": len(full_gate_indices), "result_sha256": sha(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
