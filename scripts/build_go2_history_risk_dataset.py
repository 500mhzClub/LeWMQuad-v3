#!/usr/bin/env python3
"""Build a history+action-conditioned per-primitive risk dataset for Go2.

Joins closed-loop result logs (proprioceptive tick features) with re-rendered
per-tick frames (export_go2_result_pose_risk_frames.py --counterfactual-pose-rows),
encodes frames with a frozen JEPA encoder, and labels every tick with the
counterfactual swept body clearance of each primitive from the frame pose.

Storage is per-run: latents (T,D), proprio (T,F), per-primitive blocked labels
(T,P) and clearance (T,P), plus tick indices; windows are assembled at train
time. Labels are privileged offline occupancy checks; every input is available
nonprivileged at runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis" / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.benchmarks.go2_primitive_outcome import (  # noqa: E402
    primitive_body_clearance_and_progress,
)
from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402
from lewm_contract import PrimitiveRegistry  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from build_go2_proprio_contact_dataset import (  # noqa: E402
    FEATURE_DIM as PROPRIO_FEATURE_DIM,
    _tick_features,
)
from train_go2_hidden_target_memory_probe import _load_image, _resolve_device  # noqa: E402

PRIMITIVE_VOCAB = (
    "forward_medium",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
)


def _load_log_by_tick(result_path: Path) -> tuple[dict[int, dict], str]:
    payload = json.loads(result_path.read_text())
    result = payload.get("result") or {}
    log = payload.get("log") or []
    by_tick: dict[int, dict] = {}
    for row in log:
        if isinstance(row, dict) and row.get("tick") is not None:
            by_tick[int(row["tick"])] = row
    return by_tick, str(result.get("scene", ""))


def _process_run(
    result_path: Path,
    rows_path: Path,
    *,
    encoder: torch.nn.Module,
    image_size: int,
    device: torch.device,
    registry: PrimitiveRegistry,
    grids: dict[str, InflatedOccupancyGrid],
    cell_size_m: float,
    inflation_m: float,
    body_forward_m: float,
    body_half_width_m: float,
    clearance_source: str,
    batch_size: int,
) -> dict[str, np.ndarray] | None:
    log_by_tick, scene_id = _load_log_by_tick(result_path)
    rows = []
    with rows_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if int(r.get("variant", 0)) == 0]
    rows.sort(key=lambda r: int(r["tick"]))
    if not rows:
        return None
    manifest_path = str(rows[0].get("scene_manifest", ""))
    if not manifest_path:
        return None
    if manifest_path not in grids:
        manifest_payload = json.loads(Path(manifest_path).read_text())
        grids[manifest_path] = InflatedOccupancyGrid(
            parse_scene_manifest_dict(manifest_payload),
            cell_size_m=float(cell_size_m),
            inflation_m=float(inflation_m),
        )
    grid = grids[manifest_path]

    ticks: list[int] = []
    frame_paths: list[Path] = []
    poses: list[tuple[float, float, float]] = []
    proprio: list[np.ndarray] = []
    prev_log_row: dict | None = None
    for row in rows:
        tick = int(row["tick"])
        log_row = log_by_tick.get(tick)
        if log_row is None:
            prev_log_row = None
            continue
        frame = Path(str(row["start_frame"]))
        if not frame.is_file():
            prev_log_row = None
            continue
        features = _tick_features(log_row, prev_log_row)
        prev_log_row = log_row
        if features is None:
            continue
        position = row["start_base_pose_world"]["position"]
        rpy = row["start_base_rpy_rad"]
        ticks.append(tick)
        frame_paths.append(frame)
        poses.append((float(position["x"]), float(position["y"]), float(rpy["yaw"])))
        proprio.append(features)
    if len(ticks) < 32:
        return None

    encoder.eval()
    latents: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), int(batch_size)):
            batch = torch.stack(
                [
                    _load_image(path, image_size=int(image_size))
                    for path in frame_paths[start : start + int(batch_size)]
                ]
            ).to(device)
            latents.append(encoder(batch).cpu())
    latent_arr = torch.cat(latents).numpy().astype(np.float32)

    command_dt_s = float(registry.command_dt_s)
    clearance = np.zeros((len(ticks), len(PRIMITIVE_VOCAB)), dtype=np.float32)
    progress = np.zeros((len(ticks), len(PRIMITIVE_VOCAB)), dtype=np.float32)
    for row_idx, (x_m, y_m, yaw_rad) in enumerate(poses):
        for prim_idx, primitive in enumerate(PRIMITIVE_VOCAB):
            (
                swept_clearance_m,
                after_start_clearance_m,
                _final_clearance_m,
                progress_m,
            ) = primitive_body_clearance_and_progress(
                registry=registry,
                primitive=primitive,
                grid=grid,
                x_m=float(x_m),
                y_m=float(y_m),
                yaw_rad=float(yaw_rad),
                command_dt_s=command_dt_s,
                body_forward_m=float(body_forward_m),
                body_half_width_m=float(body_half_width_m),
                clearance_source=str(clearance_source),
            )
            clearance[row_idx, prim_idx] = float(after_start_clearance_m)
            progress[row_idx, prim_idx] = float(progress_m)
    return {
        "ticks": np.asarray(ticks, dtype=np.int64),
        "latents": latent_arr,
        "proprio": np.stack(proprio).astype(np.float32),
        "clearance": clearance,
        "progress": progress,
        "scene_id": np.asarray([scene_id] * len(ticks), dtype=object),
        "executed_primitive": np.asarray(
            [str(log_by_tick[t].get("primitive", "")) for t in ticks], dtype=object
        ),
        "contact_label": np.asarray(
            [1 if log_by_tick[t].get("body_clearance_violation") else 0 for t in ticks],
            dtype=np.int64,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs",
        nargs="+",
        help="result_json::rows_jsonl pairs",
    )
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--body-forward-m", type=float, default=0.40)
    parser.add_argument("--body-half-width-m", type=float, default=0.24)
    parser.add_argument("--clearance-source", default="obstacle")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = _resolve_device(str(args.device))
    encoder, encoder_ck = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint, device=device, freeze=True
    )
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry)
    grids: dict[str, InflatedOccupancyGrid] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema": "go2_history_risk_dataset_v0",
        "primitive_vocab": list(PRIMITIVE_VOCAB),
        "proprio_feature_dim": int(PROPRIO_FEATURE_DIM),
        "latent_dim": int(encoder_ck.get("latent_dim", 192)),
        "image_size": int(args.image_size),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "clearance_source": str(args.clearance_source),
        "inflation_m": float(args.inflation_m),
        "runs": {},
    }
    for pair in args.pairs:
        result_str, rows_str = pair.split("::", 1)
        result_path, rows_path = Path(result_str), Path(rows_str)
        tag = rows_path.stem
        out_path = args.output_dir / f"{tag}.npz"
        if out_path.is_file():
            print(f"SKIP {tag}", flush=True)
            continue
        data = _process_run(
            result_path,
            rows_path,
            encoder=encoder,
            image_size=int(args.image_size),
            device=device,
            registry=registry,
            grids=grids,
            cell_size_m=float(args.cell_size_m),
            inflation_m=float(args.inflation_m),
            body_forward_m=float(args.body_forward_m),
            body_half_width_m=float(args.body_half_width_m),
            clearance_source=str(args.clearance_source),
            batch_size=int(args.batch_size),
        )
        if data is None:
            print(f"EMPTY {tag}", flush=True)
            continue
        np.savez_compressed(out_path, **data)
        summary["runs"][tag] = {
            "ticks": int(len(data["ticks"])),
            "contact_ticks": int(data["contact_label"].sum()),
        }
        print(f"WROTE {tag} ticks={len(data['ticks'])}", flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"runs": len(summary["runs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
