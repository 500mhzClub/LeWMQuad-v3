#!/usr/bin/env python3
"""Timing-only benchmark of the frozen RGB predictor's planning-time path.

This deliberately computes no prediction-quality metric and does not load any
learned safety model.  Command publication is represented by the current
in-process serialization seam; control replacement remains unqualified.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_frozen_dense_representation_encoders_v1 as E
from scripts import dev_proprio_predictor_v1 as P
from scripts import dev_action_slew_reconstruction_v1 as SLEW
from scripts import run_dev_v03_temporal_action_jepa_v1 as T


SOURCE_COMMIT = "481253b5a504b0cd9fd05b14f5ad662b496fa0a8"
CHECKPOINT = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/seed_2026080901/seed_2026080901_rgb_rollout_epoch21.pt")
CHECKPOINT_SHA = "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4"
TEMPORAL_ROWS = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/temporal_rows.jsonl")
CTX0 = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/temporal_action_jepa_v1/evaluation/frozen_ctx0.f16")
CTX1 = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/temporal_action_jepa_v1/evaluation/frozen_ctx1.f16")
PROPRIO_ROWS = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_rows.jsonl")
STATS = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_norm_stats.json")
WIDE_STATE = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/states/wide-cal-0-00.json"
OUT = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/latency_benchmark.json"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/one_tick_viability_constrained_mpc_v1")
WARMUP = 30
ITERATIONS = 500
FAMILIES = {
    "large_enclosed_maze", "medium_enclosed_maze",
    "small_enclosed_maze", "loop_alias_stress",
}


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float]) -> dict:
    array = np.asarray(values, np.float64)
    return {
        "mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)), "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)), "maximum_ms": float(array.max()),
    }


def preprocess_bytes(blob: bytes) -> torch.Tensor:
    with Image.open(BytesIO(blob)) as decoded:
        image = decoded.convert("RGB")
        if image.size != (224, 224):
            raise RuntimeError(f"unexpected timing frame size {image.size}")
        image = image.crop((0, 28, 224, 196)).resize((512, 384), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(E.IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(E.IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def gpu_busy_path() -> Path | None:
    candidates = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
    candidates.sort(
        key=lambda path: int((path.parent / "mem_info_vram_total").read_text())
        if (path.parent / "mem_info_vram_total").is_file() else 0,
        reverse=True,
    )
    for path in candidates:
        try:
            if path.read_text().strip().isdigit():
                return path
        except OSError:
            pass
    return None


def main() -> int:
    started = time.time()
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("frozen predictor checkpoint binding mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("production ROCm device unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["model_config"] != {"cell": "rgb_rollout", "use_proprio": False, "rollout": True, "width": 384}:
        raise RuntimeError("frozen predictor architecture binding mismatch")
    predictor = P.build_paired(2026080901, use_proprio=False, width=384, depth=6, heads=6)
    predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictor.to(device).eval().requires_grad_(False)
    del checkpoint

    arm = E.VJepa21CroppedV03Arm()
    encoder = arm.build(device, torch.float32)

    # Two previous context tokens are persistent planner state.  The current
    # frame is decoded and encoded afresh on every benchmark iteration.
    count0 = CTX0.stat().st_size // (T.TOKENS * T.TOKEN_DIM * 2)
    count1 = CTX1.stat().st_size // (T.TOKENS * T.TOKEN_DIM * 2)
    old0 = np.memmap(CTX0, dtype=np.float16, mode="r", shape=(count0, T.TOKENS, T.TOKEN_DIM))[0]
    old1 = np.memmap(CTX1, dtype=np.float16, mode="r", shape=(count1, T.TOKENS, T.TOKEN_DIM))[0]
    old0_gpu = T.normalise(torch.from_numpy(np.array(old0, copy=True)).to(device=device, dtype=torch.bfloat16))
    old1_gpu = T.normalise(torch.from_numpy(np.array(old1, copy=True)).to(device=device, dtype=torch.bfloat16))

    temporal = []
    family_paths = {}
    with TEMPORAL_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] in FAMILIES and row["family"] not in family_paths and Path(row["context_paths"][-1]).is_file():
                family_paths[row["family"]] = row["context_paths"][-1]
            if len(family_paths) >= 4:
                break
    if len(family_paths) < 4:
        raise RuntimeError("fewer than four representative frozen frame families")
    temporal = [Path(path) for _, path in sorted(family_paths.items())]

    wide = json.loads(WIDE_STATE.read_text())
    action_blocks = []
    for horizon in range(3):
        action_blocks.append(torch.tensor(
            np.asarray([SLEW.flatten(np.asarray(branch["post_slew"][horizon], np.float32)) for branch in wide["branches"]]),
            dtype=torch.float32, device=device,
        ))
    stats = json.loads(STATS.read_text())
    first_proprio = json.loads(PROPRIO_ROWS.open().readline())
    raw_control = torch.tensor(first_proprio["control"], dtype=torch.float32, device=device).reshape(1, 3, 5, 2)
    c_mean = torch.tensor(stats["control_mean"], dtype=torch.float32, device=device)
    c_std = torch.tensor(stats["control_std"], dtype=torch.float32, device=device)
    control = ((raw_control - c_mean) / c_std).repeat(12, 1, 1, 1)

    stages = {name: [] for name in ("observation", "preprocessing", "encoder", "predictor", "scoring", "command_publication", "total")}
    vram_samples: list[int] = []
    busy_samples: list[float] = []
    stop_sampler = threading.Event()
    busy = gpu_busy_path()

    def sample_busy() -> None:
        while not stop_sampler.is_set():
            if busy is not None:
                try:
                    busy_samples.append(float(busy.read_text().strip()))
                except (OSError, ValueError):
                    pass
            time.sleep(.01)

    sampler = threading.Thread(target=sample_busy, daemon=True)
    sampler.start()
    torch.cuda.reset_peak_memory_stats(device)
    process_cpu_start = time.process_time()
    wall_timed_start = None

    @torch.no_grad()
    def iteration(frame: Path, record: bool) -> None:
        torch.cuda.synchronize(); total_start = time.perf_counter()
        start = total_start; blob = frame.read_bytes(); observation_end = time.perf_counter()
        pixels = preprocess_bytes(blob)
        pixels = pixels.unsqueeze(0).to(device=device, dtype=torch.float32)
        torch.cuda.synchronize(); preprocessing_end = time.perf_counter(); encoder_start = preprocessing_end
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            current = arm.tokens(pixels)[0]
        current = T.normalise(current)
        torch.cuda.synchronize(); encoder_end = time.perf_counter()
        context = torch.stack([old0_gpu, old1_gpu, current], dim=0).unsqueeze(0).repeat(12, 1, 1, 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = P.unroll(predictor, context, action_blocks, control=control, max_h=3)
        torch.cuda.synchronize(); predictor_end = time.perf_counter()
        # The production route score consumes the predicted H3 endpoint.  Its
        # exact scientific values are deliberately discarded in this timing run.
        pooled = outputs[-1].float().mean(dim=(1, 2))
        nominal = np.asarray([branch["p_d"] for branch in wide["branches"]], np.float32)
        _selected = int(np.lexsort((np.arange(12), -(nominal + pooled.detach().cpu().numpy() * 0.0)))[0])
        scoring_end = time.perf_counter()
        _publication_payload = np.asarray(wide["branches"][_selected]["post_slew"][0][0], np.float32).tobytes()
        publication_end = time.perf_counter()
        if record:
            stages["observation"].append((observation_end - start) * 1000)
            stages["preprocessing"].append((preprocessing_end - observation_end) * 1000)
            stages["encoder"].append((encoder_end - encoder_start) * 1000)
            stages["predictor"].append((predictor_end - encoder_end) * 1000)
            stages["scoring"].append((scoring_end - predictor_end) * 1000)
            stages["command_publication"].append((publication_end - scoring_end) * 1000)
            stages["total"].append((publication_end - total_start) * 1000)
            vram_samples.append(int(torch.cuda.memory_allocated(device)))

    for index in range(WARMUP):
        iteration(temporal[index % len(temporal)], False)
    torch.cuda.synchronize()
    process_cpu_start = time.process_time()
    wall_timed_start = time.perf_counter()
    for index in range(ITERATIONS):
        iteration(temporal[index % len(temporal)], True)
    torch.cuda.synchronize()
    wall_timed = time.perf_counter() - wall_timed_start
    process_cpu = time.process_time() - process_cpu_start
    stop_sampler.set(); sampler.join(timeout=1)

    trace = CACHE / "latency_trace_v1.npz"
    np.savez_compressed(trace, **{key: np.asarray(value, np.float64) for key, value in stages.items()})
    total = percentile(stages["total"])
    memory_stable = max(vram_samples) - min(vram_samples) <= 16 * 1024 * 1024
    one_tick_compute = total["p99_ms"] <= 80.0 and total["maximum_ms"] <= 100.0 and memory_stable
    two_tick_compute = total["p99_ms"] <= 180.0 and total["maximum_ms"] <= 200.0 and memory_stable
    if one_tick_compute:
        compute_class = "ONE_TICK_REPLANNING_COMPUTE_SIGNAL"
    elif two_tick_compute:
        compute_class = "TWO_TICK_REPLANNING_COMPUTE_SIGNAL"
    else:
        compute_class = "REPLANNING_COMPUTE_LATENCY_NO_GO"
    # The current source contract has neither per-tick RGB delivery nor a
    # command-preemption acknowledgement.  Timing cannot manufacture that seam.
    loop_class = "REPLANNING_INTERFACE_IMPLEMENTATION_BLOCKER" if one_tick_compute or two_tick_compute else compute_class
    result = {
        "schema": "one_tick_observation_prediction_control_loop_benchmark_v1",
        "source_commit": SOURCE_COMMIT,
        "purpose": "timing only; no predictor-quality or learned-safety evaluation",
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha(CHECKPOINT), "epoch": 21, "seed": 2026080901},
        "device": {"name": properties.name, "total_memory_bytes": properties.total_memory, "torch": torch.__version__, "hip": torch.version.hip},
        "precision": "FP32 encoder/predictor weights under BF16 autocast",
        "workload": {"warmup_iterations": WARMUP, "timed_iterations": ITERATIONS, "batched_candidates": 12, "rollout_horizons": 3,
                     "representative_cached_families": sorted(family_paths), "live_sensor_acquisition": False,
                     "command_publication": "serialization-only current in-process seam; no controller acknowledgement"},
        "latency": {key: percentile(value) for key, value in stages.items()},
        "deadline_misses": {"100_ms": int(sum(value > 100.0 for value in stages["total"])), "200_ms": int(sum(value > 200.0 for value in stages["total"]))},
        "utilization": {"gpu_busy_percent_mean": None if not busy_samples else float(np.mean(busy_samples)),
                        "gpu_busy_percent_p95": None if not busy_samples else float(np.percentile(busy_samples, 95)),
                        "process_cpu_percent_of_host": 100.0 * process_cpu / wall_timed / max(1, os.cpu_count())},
        "memory": {"peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "steady_vram_min_bytes": min(vram_samples), "steady_vram_max_bytes": max(vram_samples),
                   "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                   "stable": memory_stable},
        "compute_classification": compute_class,
        "loop_rate_classification": loop_class,
        "interface_blockers": ["production RGB is block-final rather than every 100 ms", "current planner executes five buffered command ticks", "no measured command replacement acknowledgement"],
        "trace": {"path": str(trace), "sha256": sha(trace), "bytes": trace.stat().st_size},
        "runtime_s": time.time() - started,
    }
    result["content_digest"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_json(OUT, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
