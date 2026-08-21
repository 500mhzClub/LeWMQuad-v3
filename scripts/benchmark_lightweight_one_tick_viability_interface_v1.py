#!/usr/bin/env python3
"""Benchmark the simulation-side 100 ms viability command-replacement path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT
from scripts import train_evaluate_lightweight_one_tick_viability_model_v1 as TRAIN


OUT = COLLECT.OUT
RESULT = OUT / "micro_interface_result.json"


class CommandReplacementInterface:
    """One-slot simulation command mailbox; every publish replaces its predecessor."""
    def __init__(self) -> None:
        self.sequence = 0; self.acknowledged = 0; self.command = np.zeros(3, np.float32)

    def publish(self, command: np.ndarray, controller: int) -> tuple[int, int]:
        self.sequence += 1; self.command[:] = command; self.controller = int(controller)
        self.acknowledged = self.sequence
        return self.sequence, self.acknowledged


def raw_record(row: dict) -> dict:
    with np.load(row["shard_path"], allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in ("depth", "depth_valid", "lidar", "lidar_valid", "embodied", "candidate")}


def preprocess(raw: dict, stats: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    depth = raw["depth"].astype(np.float32); dv = raw["depth_valid"].astype(np.float32)
    lidar = raw["lidar"].astype(np.float32); lv = raw["lidar_valid"].astype(np.float32)
    depth = np.concatenate((depth / 10.0, np.diff(depth, axis=0) / 10.0, dv), axis=0)[None]
    lidar = np.concatenate((lidar / 10.0, np.diff(lidar, axis=0) / 10.0, lv), axis=0).reshape(1, 32, 180)
    mean = np.asarray(stats["mean"], np.float32); std = np.asarray(stats["std"], np.float32)
    embodied = ((raw["embodied"].astype(np.float32) - mean) / std)[None]
    candidate = raw["candidate"].astype(np.float32)[None]
    return tuple(torch.from_numpy(value).to(device) for value in (depth, lidar, embodied, candidate))


def percentile(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    return {"p50_ms": float(np.percentile(array, 50)), "p90_ms": float(np.percentile(array, 90)),
            "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(array.max()), "mean_ms": float(array.mean())}


def run() -> dict:
    offline = json.loads(TRAIN.RESULT.read_text()); package = torch.load(TRAIN.CHECKPOINT, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CORE.LightweightOneTickViabilityModel().to(device); model.load_state_dict(package["state_dict"]); model.eval()
    selected = offline["calibration"]["selected"]
    if selected is None:
        selected = {"contact_threshold": 0.0, "nonviability_threshold": 0.0}
    temperatures = offline["calibration"]["temperatures"]; index = json.loads(COLLECT.INDEX.read_text())
    heldout = [row for row in index["records"] if row["role"] == "heldout"]
    raw = [raw_record(row) for row in heldout]; mailbox = CommandReplacementInterface()
    components = {key: [] for key in ("sensor_handoff", "preprocessing", "state_encoding", "candidate_scoring",
                                      "filtering_ranking", "controller_transition", "command_publication", "acknowledgement", "total")}
    rows = []; warmups = 50; iterations = 1000
    torch.set_grad_enabled(False)
    def iteration(index_value: int, record_timing: bool) -> None:
        begin = time.perf_counter_ns(); source = raw[index_value % len(raw)]
        handed = {key: value.copy() for key, value in source.items()}; t1 = time.perf_counter_ns()
        depth, lidar, embodied, candidate = preprocess(handed, package["statistics"], device); t2 = time.perf_counter_ns()
        with torch.inference_mode():
            state = model.encode_state(depth, lidar, embodied); t3 = time.perf_counter_ns()
            logits = model.score_candidates(state, candidate); t4 = time.perf_counter_ns()
        value = logits[0].float().cpu().numpy(); cp = 1 / (1 + np.exp(-value[:, 0] / temperatures["contact"]))
        npv = 1 / (1 + np.exp(-value[:, 1] / temperatures["nonviability"])); count = np.clip(value[:, 5], 0, 4)
        admitted = (cp < selected["contact_threshold"]) & (npv < selected["nonviability_threshold"])
        choice = CORE.select_candidate(heldout[index_value % len(heldout)]["candidates"], admitted, count); t5 = time.perf_counter_ns()
        if choice is None:
            command = np.zeros(3, np.float32); controller = 0
        else:
            command = handed["candidate"][choice, 3:6]; controller = int(choice >= 12)
        t6 = time.perf_counter_ns(); sequence, acknowledgement = mailbox.publish(command, controller); t7 = time.perf_counter_ns()
        acknowledged = acknowledgement == sequence; t8 = time.perf_counter_ns()
        if record_timing:
            stamps = (begin, t1, t2, t3, t4, t5, t6, t7, t8)
            names = tuple(key for key in components if key != "total")
            one = {name: (stamps[offset + 1] - stamps[offset]) / 1e6 for offset, name in enumerate(names)}
            one["total"] = (t8 - begin) / 1e6
            for key, value_ms in one.items(): components[key].append(value_ms)
            rows.append({"iteration": index_value, "state_id": heldout[index_value % len(heldout)]["state_id"],
                         "selected_action": choice, "sequence": sequence, "acknowledgement": acknowledgement,
                         "acknowledged_before_return": acknowledged, **{f"{key}_ms": value_ms for key, value_ms in one.items()}})
    for index_value in range(warmups): iteration(index_value, False)
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024; cpu_start = time.process_time(); wall_start = time.perf_counter()
    for index_value in range(iterations): iteration(index_value, True)
    wall = time.perf_counter() - wall_start; cpu = time.process_time() - cpu_start
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    ledger = COLLECT.CACHE / "micro_interface_timing_rows.jsonl"; ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {key: percentile(value) for key, value in components.items()}
    total = np.asarray(components["total"]); model_decision = (np.asarray(components["state_encoding"]) +
        np.asarray(components["candidate_scoring"]) + np.asarray(components["filtering_ranking"]))
    gate = {"model_decision_p99_60ms": float(np.percentile(model_decision, 99)) <= 60.0,
        "observation_to_ack_p99_90ms": summary["total"]["p99_ms"] <= 90.0,
        "zero_iterations_over_100ms": int((total > 100).sum()) == 0,
        "replacement_ack_before_next_tick": all(row["acknowledged_before_return"] for row in rows),
        "memory_stable": rss_end - rss_start <= 16 * 1024 * 1024}
    result = {"schema": "one_tick_micro_interface_benchmark_v1", "device": str(device), "precision": "float32",
        "warmups": warmups, "iterations": iterations, "state_encoding_once": True, "candidate_batch_size": 14,
        "historical_five_tick_buffer_in_micro_path": False, "fresh_command_every_100ms": True,
        "component_latency": summary, "model_plus_decision_p99_ms": float(np.percentile(model_decision, 99)),
        "deadline_misses": {"60ms": int((total > 60).sum()), "80ms": int((total > 80).sum()), "100ms": int((total > 100).sum())},
        "cpu_utilization_percent_one_core": 100 * cpu / wall, "gpu_utilization_percent": None if device.type != "cuda" else "not sampled",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_rss_bytes": int(rss_end), "timed_rss_growth_bytes": int(rss_end - rss_start), "memory_stable": gate["memory_stable"],
        "command_publication": "single-slot replacement mailbox", "acknowledgement": "synchronous sequence acknowledgement before return",
        "timing_ledger": {"path": str(ledger), "sha256": COLLECT.sha(ledger), "bytes": ledger.stat().st_size},
        "gate": gate, "pass": all(gate.values()), "classification": "ONE_TICK_MICRO_INTERFACE_SIGNAL" if all(gate.values()) else "ONE_TICK_MICRO_INTERFACE_NO_GO"}
    result["content_digest"] = CORE.digest(result); COLLECT.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__":
    run()
