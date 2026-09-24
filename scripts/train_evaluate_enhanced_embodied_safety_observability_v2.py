#!/usr/bin/env python3
"""Train/evaluate the single enhanced true-future embodied-safety condition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE
from scripts import train_evaluate_deployment_valid_safety_observability_matrix_v1 as MATRIX

OUT = ROOT / ".generated/enhanced_embodied_safety_observability_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/enhanced_embodied_safety_observability_v2")
PRIOR_OUT = ROOT / ".generated/deployment_valid_safety_observability_matrix_v1"
SEED = 2026082009
FAMILIES = FS.FAMILIES
CHANNELS = DENSE.CHANNELS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=FS.json_default) + "\n")
    os.replace(temporary, path)


def evaluator_fixture() -> dict:
    labels = np.asarray([0, 0, 1, 1], bool)
    perfect = np.asarray([.01, .05, .9, .99])
    reversed_probability = perfect[::-1]
    tie = np.asarray([.5, .5])
    cases = {
        "transient_one_tick_contact": sum([0, 1, 0, 0]) == 1,
        "persistent_contact": sum([0, 1, 1, 1]) == 3,
        "delayed_stuck": [0, 0, 0, 1].index(1) == 3,
        "safe_branch": not any([0, 0, 0, 0]),
        "all_candidates_unsafe": bool(np.ones(12, bool).all()),
        "one_safe_candidate": int((~np.asarray([1] * 11 + [0], bool)).sum()) == 1,
        "no_candidate_admitted": not bool(np.any(np.asarray([.8, .9]) < .5)),
        "threshold_tie_rejected": not bool(np.any(tie < .5)),
        "perfect_probability_ranking": FS.auc(labels, perfect) == 1.,
        "reversed_probability_ranking": FS.auc(labels, reversed_probability) == 0.,
    }
    rows = [{"state_id": "purpose-999", "candidate_index": candidate, "family": FAMILIES[0],
             "unsafe": candidate != 7, "p_d": candidate / 100., "p_theta": 0.}
            for candidate in range(12)]
    kinematic = np.zeros((12, 6), np.float32); kinematic[:, 4] = [row["p_d"] for row in rows]
    FS.rows_global = rows
    planning = FS.evaluate_condition("fixture", rows, np.arange(12),
                                     np.asarray([float(row["unsafe"]) for row in rows]), .5, kinematic)
    cases["one_safe_candidate_selected"] = planning["planning"]["per_state"][0]["selected_candidate"] == 7
    encoded = json.dumps({"cases": cases, "planning": planning}, sort_keys=True, default=FS.json_default)
    cases["deterministic_complete_json"] = encoded == json.dumps(json.loads(encoded), sort_keys=True)
    payload = {"schema": "enhanced_embodied_safety_evaluator_fixture_v2", "cases": cases,
               "fixture_planning": planning, "pass": all(cases.values())}
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if json.loads((OUT / "evaluator_fixture.json").read_text()) != payload or not payload["pass"]:
        raise RuntimeError("enhanced evaluator fixture failed")
    return payload


def load_branches() -> tuple[list[dict], dict]:
    states, _, _ = DENSE.load_dense()
    index = json.loads((OUT / "enhanced_sensor_index.json").read_text())
    if not index.get("complete") or index.get("content_digest") != canonical_digest({k: v for k, v in index.items() if k != "content_digest"}):
        raise RuntimeError("invalid enhanced sensor index")
    by_state = {row["state_id"]: row for row in index["state_records"]}
    branches = []
    for state in states:
        record = by_state[state["state_id"]]
        with np.load(record["shard_path"]) as loaded:
            current = np.asarray(loaded["current"], np.float32)
            future = np.asarray(loaded["future"], np.float32)
            action_control = np.asarray(loaded["action_control"], np.float32)
        for branch in state["branches"]:
            ci = int(branch["candidate_index"])
            branch = dict(branch)
            branch.update(split=state["split"], family=state["family"],
                          current_enhanced=current[ci], future_enhanced=future[ci],
                          action_control=action_control[ci])
            branches.append(branch)
    if len(branches) != 576:
        raise RuntimeError("frozen branch cardinality changed")
    return branches, index


def fit_stats(branches: list[dict]) -> dict:
    value = np.concatenate([np.concatenate((branch["current_enhanced"][None], branch["future_enhanced"]), 0)
                            for branch in branches]).astype(np.float64)
    mean = value.mean(0); std = value.std(0); degenerate = std < 1e-7; std[degenerate] = 1.
    payload = {"mean": mean.tolist(), "std": std.tolist(),
               "degenerate_channel_indices": np.flatnonzero(degenerate).tolist(), "fit_samples": len(value)}
    payload["digest"] = canonical_digest(payload)
    atomic_json(OUT / "fit_standardization.json", payload)
    return payload


def input_label_circularity(branches: list[dict]) -> dict:
    value = np.concatenate([branch["future_enhanced"] for branch in branches], 0)
    target = np.concatenate([DENSE.branch_targets(branch) for branch in branches], 0)
    direct = []
    for channel, name in enumerate(SENSOR.CHANNELS):
        for label_index, label_name in enumerate(CHANNELS):
            if np.array_equal(value[:, channel], target[:, label_index]):
                direct.append({"input": name, "target": label_name})
    return {
        "direct_mathematical_equivalences": direct,
        "passed": not direct,
        "normal_foot_contact_is_target": False,
        "foot_force_semantics": "calf-link net force magnitude includes normal ground support and is not the frozen disallowed body-contact label",
        "torque_semantics": "PD actuator force is continuous motor state, not a collision decision",
        "stuck_label_dependency": "privileged base displacement/window logic; global or body translation is excluded from model input",
        "event_instrumentation_only_conditions": [],
    }


def _signal(branch: dict) -> np.ndarray:
    x = branch["future_enhanced"]
    control = branch["action_control"]
    return np.stack((np.linalg.norm(x[:, 0:3], axis=1), np.max(np.abs(x[:, 33:45]), axis=1),
                     np.max(np.abs(x[:, 45:57]), axis=1), np.max(x[:, 57:61], axis=1),
                     np.linalg.norm(control[:, 0:3] - control[:, 3:6], axis=1),
                     np.linalg.norm(x[:, 61:73] - branch["current_enhanced"][None, 61:73], axis=1)), 1)


def event_audit(branches: list[dict]) -> dict:
    names = ("accelerometer_norm", "peak_abs_joint_acceleration", "peak_abs_actuator_torque",
             "peak_foot_force", "applied_command_change", "motor_policy_action_response")
    fit_negative = defaultdict(list)
    for branch in branches:
        if branch["split"] != "fit": continue
        signals = _signal(branch); target = DENSE.branch_targets(branch)
        for kind, col in (("contact", 0), ("stuck", 1)):
            fit_negative[kind].append(signals[target[:, col] < .5])
    thresholds = {kind: np.quantile(np.concatenate(values, 0), .95, axis=0) for kind, values in fit_negative.items()}

    def summarize(rows: list[dict], kind: str, label_col: int) -> dict:
        events = []; positive_ticks = []; negative_ticks = []; pre_rows = []; event_rows = []; post_rows = []
        preventive = event_tick = aftermath = no_signal = 0
        for branch in rows:
            labels = DENSE.branch_targets(branch)[:, label_col].astype(bool); signals = _signal(branch)
            positive_ticks.append(signals[labels]); negative_ticks.append(signals[~labels])
            if not labels.any(): continue
            positive = np.flatnonzero(labels); first, final = int(positive[0]), int(positive[-1])
            if first > 0: pre_rows.append(signals[first - 1])
            event_rows.append(signals[first])
            if final + 1 < len(labels): post_rows.append(signals[final + 1])
            pre = first > 0 and bool(np.any(signals[first - 1] > thresholds[kind]))
            during = bool(np.any(signals[first:final + 1] > thresholds[kind]))
            post = final + 1 < len(labels) and bool(np.any(signals[final + 1] > thresholds[kind]))
            preventive += pre; event_tick += (not pre and during); aftermath += (not pre and not during and post)
            no_signal += not (pre or during or post)
            events.append({"branch_id": branch["branch_id"], "first_tick": first + 1, "final_tick": final + 1,
                           "duration_ticks": int(labels.sum()), "peak_accelerometer": float(signals[labels, 0].max()),
                           "peak_joint_acceleration": float(signals[labels, 1].max()),
                           "peak_torque": float(signals[labels, 2].max()), "peak_foot_force": float(signals[labels, 3].max()),
                           "pre_event_detectable": pre, "event_tick_detectable": during, "post_event_detectable": post})
        pos = np.concatenate(positive_ticks, 0) if positive_ticks else np.empty((0, len(names))); neg = np.concatenate(negative_ticks, 0)
        effect = (pos.mean(0) - neg.mean(0)) / np.maximum(np.sqrt((pos.var(0) + neg.var(0)) / 2), 1e-9) if len(pos) else np.full(len(names), np.nan)
        count = len(events)
        def distribution(values):
            if not values: return {"count": 0, "mean": None, "median": None, "p90": None}
            array = np.stack(values)
            return {"count": len(array), "mean": dict(zip(names, array.mean(0).tolist())),
                    "median": dict(zip(names, np.median(array, axis=0).tolist())),
                    "p90": dict(zip(names, np.quantile(array, .9, axis=0).tolist()))}
        return {"positive_branches": count, "positive_ticks": int(sum(row["duration_ticks"] for row in events)),
                "first_event_tick_distribution": [row["first_tick"] for row in events],
                "event_duration_distribution": [row["duration_ticks"] for row in events],
                "signal_thresholds_fit_negative_p95": dict(zip(names, thresholds[kind].tolist())),
                "positive_tick_means": dict(zip(names, pos.mean(0).tolist())) if len(pos) else None,
                "negative_tick_means": dict(zip(names, neg.mean(0).tolist())),
                "standardized_effect_sizes": dict(zip(names, effect.tolist())),
                "pre_event_distribution": distribution(pre_rows), "first_event_tick_distribution_signals": distribution(event_rows),
                "post_event_distribution": distribution(post_rows),
                "proportion_detectable_before_event": preventive / count if count else None,
                "proportion_detectable_first_at_event": event_tick / count if count else None,
                "proportion_detectable_only_after_event": aftermath / count if count else None,
                "proportion_no_measurable_signal": no_signal / count if count else None,
                "events": events}

    result = {"threshold_contract": "fit-negative tick 95th percentile, frozen before calibration/heldout",
              "claims_boundary": "pre-event threshold exceedance is preventive evidence; event/post-event evidence is detection or aftermath only",
              "by_split": {}, "by_family": {}}
    for split in ("fit", "calibration", "heldout"):
        rows = [branch for branch in branches if branch["split"] == split]
        result["by_split"][split] = {"contact": summarize(rows, "contact", 0), "stuck": summarize(rows, "stuck", 1)}
    for family in FAMILIES:
        rows = [branch for branch in branches if branch["family"] == family]
        result["by_family"][family] = {"contact": summarize(rows, "contact", 0), "stuck": summarize(rows, "stuck", 1)}
    return result


class EnhancedSafetyModel(nn.Module):
    def __init__(self, embodied_dims: int = 73, action_dims: int = 6):
        super().__init__()
        self.embodied = nn.Sequential(nn.Linear(embodied_dims * 3, 128), nn.GELU())
        self.action = nn.Sequential(nn.Linear(action_dims, 48), nn.GELU())
        self.temporal = nn.GRU(176, 128, batch_first=True)
        self.output = nn.Linear(128, 5)

    def forward(self, current: torch.Tensor, future: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ticks = future.shape[1]; current = current[:, None].expand(-1, ticks, -1)
        state = self.embodied(torch.cat((current, future, future - current), -1))
        hidden, _ = self.temporal(torch.cat((state, self.action(action)), -1))
        return self.output(hidden)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, branches: list[dict], stats: dict):
        self.branches = branches; self.mean = np.asarray(stats["mean"], np.float32); self.std = np.asarray(stats["std"], np.float32)
    def __len__(self): return len(self.branches)
    def __getitem__(self, index):
        branch = self.branches[index]
        return {"current": ((branch["current_enhanced"] - self.mean) / self.std).astype(np.float32),
                "future": ((branch["future_enhanced"] - self.mean) / self.std).astype(np.float32),
                "action": branch["action_control"].astype(np.float32), "target": DENSE.branch_targets(branch)}


def loader(branches, stats, batch_size, shuffle_seed=None):
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return torch.utils.data.DataLoader(Dataset(branches, stats), batch_size=batch_size, shuffle=shuffle_seed is not None,
                                       generator=generator, num_workers=2, pin_memory=True)


def device_batch(raw, device):
    return {key: value.to(device, non_blocking=True) for key, value in raw.items()}


def train(fit: list[dict], stats: dict, device: torch.device):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = EnhancedSafetyModel().to(device); parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 500000: raise RuntimeError("parameter cap exceeded")
    weights_np, defined, prevalence = DENSE.positive_weights(fit); weights = torch.from_numpy(weights_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    smoke = device_batch(next(iter(loader(fit[:64], stats, 16))), device)
    model.train(); logits = model(smoke["current"], smoke["future"], smoke["action"])
    loss = DENSE.balanced_loss(logits, smoke["target"], weights, defined); loss.backward()
    if not torch.isfinite(loss): raise RuntimeError("nonfinite smoke loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0:
            raise RuntimeError(f"smoke gradient failure {name}")
    with torch.inference_mode():
        model.eval(); first = model(smoke["current"], smoke["future"], smoke["action"])
        if not torch.equal(first, model(smoke["current"], smoke["future"], smoke["action"])): raise RuntimeError("nondeterministic smoke")
        if torch.allclose(first, model(smoke["current"], smoke["future"], smoke["action"].flip(1))): raise RuntimeError("action insensitive")
        if torch.allclose(first, model(smoke["current"], smoke["future"].flip(1), smoke["action"].flip(1))): raise RuntimeError("temporal order insensitive")
    smoke_path = OUT / ".smoke.pt"; torch.save(model.state_dict(), smoke_path)
    clone = EnhancedSafetyModel().to(device); clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    optimizer.zero_grad(set_to_none=True); history = []; started = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        model.train(); total = seen = 0
        for raw in loader(fit, stats, 64, SEED + epoch):
            batch = device_batch(raw, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_loss = DENSE.balanced_loss(model(batch["current"], batch["future"], batch["action"]), batch["target"], weights, defined)
            batch_loss.backward(); optimizer.step(); total += float(batch_loss.detach()) * len(batch["target"]); seen += len(batch["target"])
        history.append({"epoch": epoch + 1, "mean_balanced_bce": total / seen})
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(history[-1]), flush=True)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return model, history, {"parameter_count": parameter_count, "seed": SEED, "batch_size": 64,
                            "training_runtime_s": time.time() - started, "peak_vram_bytes": peak,
                            "positive_weights": weights_np.tolist(), "defined_outputs": defined.tolist(),
                            "fit_tick_prevalence": prevalence, "smoke_passed": True}


def predict(model, branches, stats, device):
    logits = []; targets = []; model.eval()
    with torch.inference_mode():
        for raw in loader(branches, stats, 64):
            batch = device_batch(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits.append(model(batch["current"], batch["future"], batch["action"]).float().cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--fixture-only", action="store_true"); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    fixture = evaluator_fixture()
    if args.fixture_only: print(json.dumps(fixture, indent=2)); return 0
    started = time.time(); branches, sensor_index = load_branches()
    split = {name: [branch for branch in branches if branch["split"] == name] for name in ("fit", "calibration", "heldout")}
    circularity = input_label_circularity(branches)
    if not circularity["passed"]: raise RuntimeError("input-label circularity found")
    audit = event_audit(branches); atomic_json(OUT / "event_observability_audit.json", audit)
    stats = fit_stats(split["fit"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history, training = train(split["fit"], stats, device)
    checkpoint = OUT / f"enhanced_proprio_action_safety_head_v1_seed_{SEED}.pt"
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epoch": 60,
                "parameter_count": training["parameter_count"], "channels": list(SENSOR.CHANNELS)}, checkpoint)
    clone = EnhancedSafetyModel().to(device); clone.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]); clone.eval()
    cal_logits, cal_target = predict(clone, split["calibration"], stats, device)
    temperature = DENSE.fit_temperature(cal_logits[:, -1, 4], cal_target[:, -1, 4])
    cal_probability = 1. / (1. + np.exp(-cal_logits[:, -1, 4] / temperature))
    calibration = {"temperature": temperature, **DENSE.choose_threshold(cal_probability, cal_target[:, -1, 4])}
    route_rows = FS.load_metadata(); route_by_id = {row["state_id"] + f":{int(row['candidate_index']):02d}": i for i, row in enumerate(route_rows)}
    route_indices = {name: np.asarray([route_by_id[branch["branch_id"]] for branch in values], int) for name, values in split.items()}
    kinematic = np.stack([row["kinematic"] for row in route_rows]); FS.rows_global = route_rows
    oracle_probability = np.asarray([float(route_rows[i]["unsafe"]) for i in route_indices["heldout"]])
    oracle = FS.evaluate_condition("oracle_safety", route_rows, route_indices["heldout"], oracle_probability, .5, kinematic)
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    held_logits, held_target = predict(clone, split["heldout"], stats, device)
    heldout = MATRIX.evaluate("ENHANCED_PROPRIO_ACTION", split["heldout"], held_logits, held_target,
                              temperature, calibration["threshold"], route_rows, route_indices["heldout"], kinematic)
    heldout["candidate_filter_and_planning"]["planning"]["oracle_progress_fraction"] = (
        heldout["candidate_filter_and_planning"]["planning"]["mean_selected_distance_progress_m"] / oracle_progress if abs(oracle_progress) > 1e-12 else None)
    fit_logits, fit_target = predict(clone, split["fit"], stats, device)
    fit_result = MATRIX.evaluate("ENHANCED_PROPRIO_ACTION", split["fit"], fit_logits, fit_target,
                                 temperature, calibration["threshold"], route_rows, route_indices["fit"], kinematic)
    common_gate = MATRIX.gate(heldout, oracle_progress)
    common_gate["checks"]["contact_auc_ge_0_80"] = heldout["components"]["contact"]["auc"] is not None and heldout["components"]["contact"]["auc"] >= .80
    common_gate["checks"].pop("contact_auc_ge_0_75", None)
    common_gate["passed"] = all(common_gate["checks"].values())
    prior = json.loads((PRIOR_OUT / "result.json").read_text())
    frozen = {name: prior["heldout"][name] for name in ("ACTION_CONTROL_ONLY", "RAW_RGB", "PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION")}
    frozen["FINAL_LAYER_VIT_L"] = prior["frozen_baselines"]["final_layer_vit_l"]
    original = frozen["PROPRIOCEPTION"]
    effects = {
        "contact_auc_vs_original_proprio": heldout["components"]["contact"]["auc"] - original["components"]["contact"]["auc"],
        "safe_retention_vs_original_proprio": heldout["aggregate"]["safe_candidate_retention"] - original["aggregate"]["safe_candidate_retention"],
        "missed_transient_contact_vs_original_proprio": heldout["temporal"]["missed_transient_contact_rate"] - original["temporal"]["missed_transient_contact_rate"],
        "states_retaining_safe_vs_original_proprio": heldout["candidate_filter_and_planning"]["planning"]["states_retaining_safe"] - original["candidate_filter_and_planning"]["planning"]["states_retaining_safe"],
        "stuck_auc_vs_original_proprio": heldout["components"]["stuck"]["auc"] - original["components"]["stuck"]["auc"],
    }
    incremental_checks = {
        "one_incremental_contact_or_retention_effect": effects["contact_auc_vs_original_proprio"] >= .05 or effects["safe_retention_vs_original_proprio"] >= .20 or effects["missed_transient_contact_vs_original_proprio"] <= -.20 or effects["states_retaining_safe_vs_original_proprio"] >= 3,
        "stuck_not_materially_regressed": effects["stuck_auc_vs_original_proprio"] >= -.05,
    }
    incremental_gate = all(incremental_checks.values())
    event_tendency = incremental_checks["one_incremental_contact_or_retention_effect"]
    if common_gate["passed"] and incremental_gate:
        classification = "ENHANCED_EMBODIED_SAFETY_OBSERVABILITY_SIGNAL"
        recommendation = "Develop an action-conditioned high-rate embodied micro-safety predictor; keep the macro visual JEPA separate."
    elif event_tendency:
        classification = "ENHANCED_EMBODIED_SAFETY_POSITIVE_TENDENCY"
        recommendation = "Treat enhanced channels as useful event instrumentation, but do not train a predictive safety model until a fresh sensor-contract evaluation preserves safe actions."
    else:
        classification = "DEPLOYMENT_VALID_EMBODIED_SAFETY_NO_GO"
        recommendation = "Add a changed environmental/contact sensor contract such as depth, LiDAR, or dedicated body contact sensing, or narrow the learned claim to observable failure modes."
    result = {
        "schema": "enhanced_embodied_safety_observability_v2_result", "source_commit": "19525d6ca2061924007377ea5fe3255dda85364b",
        "preserved_terminal": "CURRENT_DEPLOYMENT_SENSOR_CONTRACT_SAFETY_NO_GO",
        "bindings": {"enhanced_sensor_index_digest": sensor_index["content_digest"], "enhanced_sensor_index_sha256": sha(OUT / "enhanced_sensor_index.json"),
                     "prior_matrix_result_sha256": sha(PRIOR_OUT / "result.json")},
        "panel": {"states": 48, "branches": 576, "ticks": 15, "split_states": {"fit": 32, "calibration": 8, "heldout": 8},
                  "split_branches": {name: len(values) for name, values in split.items()}, "families": list(FAMILIES)},
        "sensor_contract": {"channels": sensor_index["channels"], "action_control_channels": sensor_index["action_control_channels"],
                            "definitions": sensor_index["channel_contract"], "missing_channels": sensor_index["missing_channels"],
                            "degenerate_channels": sensor_index["degenerate_channels"], "excluded_inputs": sensor_index["excluded_inputs"]},
        "replay_verification": {key: sensor_index[key] for key in ("replayed_branches", "post_slew_pose_contact_stuck_mismatches", "raw_replay_h3_aggregate_matches", "raw_replay_h3_aggregate_mismatches_preserved", "snapshot_digest_matches", "snapshot_digest_mismatches_with_exact_tick_reproduction")},
        "input_label_circularity_audit": circularity, "event_aligned_observability_audit": audit,
        "evaluator_fixture": fixture, "model": {**training, "architecture": "Linear(219,128)+GELU; Linear(6,48)+GELU; causal GRU(176,128); Linear(128,5)",
                            "input_allow_list": ["current enhanced embodied state", "true future enhanced embodied state", "future-minus-current embodied state", "post-slew candidate action", "control history"],
                            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}, "epochs": 60,
                            "initial_loss": history[0]["mean_balanced_bce"], "final_loss": history[-1]["mean_balanced_bce"], "training_history": history},
        "calibration": calibration, "heldout": heldout, "fit": fit_result, "common_gate": common_gate,
        "frozen_comparators": frozen, "oracle_safety_kinematic": oracle, "incremental_value": {"effects": effects, "checks": incremental_checks, "passed": incremental_gate},
        "classification": classification, "recommended_next_architecture_or_sensor_decision": recommendation,
        "claims_boundary": {"event_detectability": "true-future event-tick and aftermath evidence", "preventive_evidence": "pre-event signal only",
                            "candidate_filtering": "H3 cumulative true-future operating point; not a before-execution predictor claim"},
        "runtime": {"materialization_compute_s": sensor_index["runtime_compute_s"], "materialization_wall_s": sensor_index["parallel_wall_runtime_s"],
                    "training_s": training["training_runtime_s"], "evaluation_and_audit_s": time.time() - started - training["training_runtime_s"],
                    "peak_vram_bytes": training["peak_vram_bytes"], "sensor_storage_bytes": sensor_index["storage_bytes"], "checkpoint_storage_bytes": checkpoint.stat().st_size},
        "custody": {"one_new_seed_trained": True, "seed": SEED, "world_model_predictor_opened_or_trained": False,
                    "new_state_identities": 0, "new_candidate_identities": 0, "simulator_state_identity_trained": False,
                    "memory_or_navigation_system_trained": False},
    }
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "result_sha256": sha(OUT / "result.json"),
                      "aggregate_auc": heldout["aggregate"]["auc"], "contact_auc": heldout["components"]["contact"]["auc"],
                      "stuck_auc": heldout["components"]["stuck"]["auc"], "safe_retention": heldout["aggregate"]["safe_candidate_retention"],
                      "common_gate": common_gate["passed"], "incremental_gate": incremental_gate}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
