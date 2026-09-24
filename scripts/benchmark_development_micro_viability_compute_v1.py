#!/usr/bin/env python3
"""Benchmark the frozen development micro-viability production compute path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import train_evaluate_lightweight_one_tick_viability_model_v1 as BASE
from scripts import run_development_micro_viability_model_screen_v1 as SCREEN

WARMUPS = 30
ITERATIONS = 1000


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), quantile))


def run() -> dict:
    offline = json.loads(SCREEN.RESULT.read_text()); split = offline["split"]
    records = SCREEN.records_for_split(split); state = records["heldout"][0]
    device = torch.device(offline["device"]); package = torch.load(SCREEN.CHECKPOINT, map_location=device, weights_only=True)
    model = CORE.LightweightOneTickViabilityModel().to(device); model.load_state_dict(package["state_dict"]); model.eval()
    stats = package["statistics"]; calibration = offline["calibration"]; temperatures = calibration["temperatures"]
    selected = calibration["selected"] or {"contact_threshold": 0.0, "nonviability_threshold": 0.0}
    process = psutil.Process(); rows = []; rss = []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    def iteration(timed: bool) -> None:
        started = time.perf_counter_ns()
        arrays = BASE.one_tensor(state, stats); load_end = time.perf_counter_ns()
        tensors = [torch.from_numpy(value[None]).to(device) for value in arrays[:4]]; tensor_end = time.perf_counter_ns()
        with torch.inference_mode(): logits = model(*tensors).float().cpu().numpy()[0]
        inference_end = time.perf_counter_ns()
        cp, vp, count = BASE.probabilities(logits[None], temperatures); cp, vp, count = cp[0], vp[0], count[0]
        admitted = (cp < selected["contact_threshold"]) & (vp < selected["nonviability_threshold"])
        choice = CORE.select_candidate(state["candidates"], admitted, count); decision_end = time.perf_counter_ns()
        payload = json.dumps({"state_id": state["state_id"], "candidate": choice,
            "controller": None if choice is None else state["candidates"][choice]["controller"],
            "command": None if choice is None else state["candidates"][choice]["applied_first_tick_action"]},
            sort_keys=True, separators=(",", ":")).encode()
        if not payload: raise RuntimeError("serialization failed")
        ended = time.perf_counter_ns()
        if timed:
            scale = 1e-6; rows.append({"sensor_loading_and_normalization_ms": (load_end - started) * scale,
                "tensor_construction_ms": (tensor_end - load_end) * scale, "batched_inference_ms": (inference_end - tensor_end) * scale,
                "calibration_filter_selection_ms": (decision_end - inference_end) * scale,
                "command_serialization_ms": (ended - decision_end) * scale, "total_ms": (ended - started) * scale})
            rss.append(process.memory_info().rss)

    for _ in range(WARMUPS): iteration(False)
    cpu_before = process.cpu_times(); wall_before = time.perf_counter()
    for _ in range(ITERATIONS): iteration(True)
    wall = time.perf_counter() - wall_before; cpu_after = process.cpu_times()
    totals = [row["total_ms"] for row in rows]
    trace = SCREEN.CACHE / "micro_compute_timing_rows.jsonl"; trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {"schema": "development_micro_viability_compute_benchmark_v1", "device": str(device), "production_precision": "float32",
        "warmups": WARMUPS, "iterations": ITERATIONS, "state_encoding_once": True, "candidates_batched": 14,
        "included_path": ["sensor-row loading", "normalization", "tensor construction", "shared state encoding", "batched candidate scoring",
                          "probability calibration", "joint thresholding", "deterministic selection", "command serialization"],
        "latency_ms": {"p50": percentile(totals, 50), "p90": percentile(totals, 90), "p95": percentile(totals, 95),
                       "p99": percentile(totals, 99), "maximum": max(totals), "mean": float(np.mean(totals))},
        "component_mean_ms": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "total_ms"},
        "deadline_misses": {"50_ms": sum(value > 50 for value in totals), "80_ms": sum(value > 80 for value in totals),
                            "100_ms": sum(value > 100 for value in totals)},
        "cpu_utilization_percent_one_core_equivalent": 100 * ((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)) / wall,
        "gpu_utilization": "not_available_no_cuda" if device.type != "cuda" else "not sampled by installed runtime",
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_rss_bytes": max(rss), "rss_growth_bytes": rss[-1] - rss[0],
        "memory_stable": rss[-1] - rss[0] <= 8 * 1024 * 1024,
        "trace_path": str(trace), "trace_bytes": trace.stat().st_size, "trace_sha256": SCREEN.sha(trace),
        "wall_runtime_s": wall, "replanning_interface_classification": "REPLANNING_INTERFACE_UNRESOLVED"}
    if summary["latency_ms"]["p99"] <= 50 and not summary["deadline_misses"]["80_ms"] and summary["memory_stable"]:
        classification = "MICRO_VIABILITY_COMPUTE_SIGNAL"
    elif summary["latency_ms"]["p99"] <= 80 and not summary["deadline_misses"]["100_ms"]:
        classification = "MICRO_VIABILITY_COMPUTE_POSITIVE_TENDENCY"
    else: classification = "MICRO_VIABILITY_COMPUTE_NO_GO"
    summary["classification"] = classification; summary["content_digest"] = CORE.digest(summary)
    atomic_json(SCREEN.OUT / "compute_benchmark.json", summary); print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
