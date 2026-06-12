#!/usr/bin/env python3
"""Encode a scored task-aligned decision index with a frozen LeWM checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _encode_frames, _load_image_tensor  # noqa: E402


def _read_rows(path: Path, max_rows: int) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def _candidate_matrix(rows: list[dict], field: str, primitive_names: list[str]) -> np.ndarray:
    values = []
    for row in rows:
        by_name = {
            candidate["primitive_name"]: candidate
            for candidate in row["counterfactual_candidates"]
        }
        values.append([by_name[name][field] for name in primitive_names])
    return np.asarray(values)


@torch.no_grad()
def _encode_spatial_frames(
    model,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    spatial_grid: int,
) -> np.ndarray:
    descriptors = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        array = (
            np.stack([_load_image_tensor(path) for path in batch_paths]).astype(np.float32)
            / 255.0
        )
        vision = torch.from_numpy(array).to(device)
        tokens = model.encoder.vis_enc.forward_tokens(vision)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        patch_grid = int(round(patches.shape[1] ** 0.5))
        if patch_grid * patch_grid != patches.shape[1]:
            raise RuntimeError(f"patch-token count {patches.shape[1]} is not square")
        pooled = F.adaptive_avg_pool2d(
            patches.reshape(len(batch_paths), patch_grid, patch_grid, -1).permute(0, 3, 1, 2),
            (spatial_grid, spatial_grid),
        )
        descriptor = torch.cat([cls, pooled.flatten(1)], dim=1)
        descriptors.append(descriptor.cpu().float().numpy())
    return np.concatenate(descriptors, axis=0)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--spatial-grid",
        type=int,
        default=0,
        help="Also store CLS plus an NxN average-pooled patch-token descriptor.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=0,
        help="Also store terminal/mean/delta raw descriptors over the last H frames.",
    )
    parser.add_argument(
        "--goal-frame-field",
        choices=("target_frame", "route_target_frame", "local_target_frame", "none"),
        default="target_frame",
        help="Which row field supplies the goal image. v2 contract: "
        "local_target_frame is the matched local-subgoal image, "
        "route_target_frame is the final-route image, none zeros the goal.",
    )
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    model, _config = load_model(
        SimpleNamespace(
            checkpoint=args.checkpoint.resolve(),
            max_seq_len=args.max_seq_len,
            sigreg_lambda=args.sigreg_lambda,
        ),
        device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = _read_rows(args.input, args.max_rows)
    if not rows:
        raise SystemExit("input index is empty")
    primitive_names = [
        str(candidate["primitive_name"])
        for candidate in rows[0]["counterfactual_candidates"]
    ]
    primitive_actions = np.asarray(
        [
            candidate["active_block"]
            for candidate in rows[0]["counterfactual_candidates"]
        ],
        dtype=np.float32,
    )
    for row in rows:
        names = [
            str(candidate["primitive_name"])
            for candidate in row["counterfactual_candidates"]
        ]
        if names != primitive_names:
            raise RuntimeError("counterfactual primitive order differs between rows")

    start_paths = [Path(row["start_frame"]) for row in rows]
    goal_field = args.goal_frame_field
    target_paths = [
        Path(row[goal_field])
        if goal_field != "none" and row.get(goal_field) is not None
        else None
        for row in rows
    ]
    history_paths = [
        [Path(path) for path in row.get("history_frames", [])[-args.history_frames :]]
        if args.history_frames > 0
        else []
        for row in rows
    ]
    unique_paths = list(
        dict.fromkeys(
            start_paths
            + [path for path in target_paths if path is not None]
            + [path for history in history_paths for path in history]
        )
    )
    missing = [str(path) for path in unique_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} feature-input frames are missing; first={missing[0]}")
    print(
        f"encoding {len(unique_paths)} unique frames for {len(rows)} decisions "
        f"on {device}",
        flush=True,
    )
    raw_unique, proj_unique = _encode_frames(model, unique_paths, device, args.batch_size)
    spatial_unique = None
    if args.spatial_grid > 0:
        print(f"encoding {args.spatial_grid}x{args.spatial_grid} spatial descriptors", flush=True)
        spatial_unique = _encode_spatial_frames(
            model,
            unique_paths,
            device,
            args.batch_size,
            args.spatial_grid,
        )
    path_index = {path: index for index, path in enumerate(unique_paths)}
    start_index = np.asarray([path_index[path] for path in start_paths], dtype=np.int64)
    goal_present = np.asarray([path is not None for path in target_paths], dtype=np.bool_)
    latent_dim = int(raw_unique.shape[1])
    goal_raw = np.zeros((len(rows), latent_dim), dtype=np.float32)
    goal_proj = np.zeros((len(rows), latent_dim), dtype=np.float32)
    goal_spatial = (
        np.zeros((len(rows), spatial_unique.shape[1]), dtype=np.float32)
        if spatial_unique is not None
        else None
    )
    start_history = (
        np.zeros((len(rows), latent_dim * 3), dtype=np.float32)
        if args.history_frames > 0
        else None
    )
    goal_history = (
        np.zeros((len(rows), latent_dim * 3), dtype=np.float32)
        if args.history_frames > 0
        else None
    )
    for index, path in enumerate(target_paths):
        if path is not None:
            goal_raw[index] = raw_unique[path_index[path]]
            goal_proj[index] = proj_unique[path_index[path]]

    progress = _candidate_matrix(rows, "target_progress_m", primitive_names)
    heading = _candidate_matrix(rows, "heading_error_rad", primitive_names)
    progress = np.where(progress == None, 0.0, progress).astype(np.float32)  # noqa: E711
    heading = np.where(heading == None, 0.0, heading).astype(np.float32)  # noqa: E711
    collision = _candidate_matrix(rows, "collided", primitive_names).astype(np.bool_)
    clearance = _candidate_matrix(rows, "clearance_m", primitive_names).astype(np.float32)
    cost = _candidate_matrix(rows, "cost", primitive_names).astype(np.float32)
    relative_goal_vector_body = []
    for row in rows:
        v = row.get("relative_goal_vector_body")
        relative_goal_vector_body.append(v if v is not None else [0.0, 0.0])
    relative_goal_vector_body = np.asarray(relative_goal_vector_body, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema": np.asarray("task_aligned_feature_dataset_v0"),
        "goal_frame_field": np.asarray(str(args.goal_frame_field)),
        "source_index": np.asarray(str(args.input.resolve())),
        "source_checkpoint": np.asarray(str(args.checkpoint.resolve())),
        "primitive_names": np.asarray(primitive_names),
        "primitive_actions": primitive_actions,
        "start_raw": raw_unique[start_index].astype(np.float32),
        "start_proj": proj_unique[start_index].astype(np.float32),
        "goal_raw": goal_raw,
        "goal_proj": goal_proj,
        "goal_present": goal_present,
        "relative_goal_vector_body": relative_goal_vector_body,
        "collision": collision,
        "progress": progress,
        "heading": heading,
        "clearance": clearance,
        "cost": cost,
        "scene_id": np.asarray([str(row["scene_id"]) for row in rows]),
        "family": np.asarray([str(row["family"]) for row in rows]),
        "decision_types": np.asarray(
            ["+".join(sorted(row["decision_types"])) for row in rows]
        ),
        "logged_primitive": np.asarray([str(row["primitive_name"]) for row in rows]),
    }
    if spatial_unique is not None:
        for index, path in enumerate(target_paths):
            if path is not None:
                goal_spatial[index] = spatial_unique[path_index[path]]
        arrays["start_spatial"] = spatial_unique[start_index].astype(np.float32)
        arrays["goal_spatial"] = goal_spatial
    if args.history_frames > 0:
        for index, history in enumerate(history_paths):
            if not history:
                history = [start_paths[index]]
            history_latents = raw_unique[
                np.asarray([path_index[path] for path in history], dtype=np.int64)
            ]
            terminal = raw_unique[start_index[index]]
            start_history[index] = np.concatenate(
                [terminal, history_latents.mean(axis=0), terminal - history_latents[0]]
            )
            if target_paths[index] is not None:
                goal = raw_unique[path_index[target_paths[index]]]
                goal_history[index] = np.concatenate([goal, goal, np.zeros_like(goal)])
        arrays["start_history"] = start_history
        arrays["goal_history"] = goal_history
    np.savez_compressed(args.output, **arrays)
    summary = {
        "schema": "task_aligned_feature_dataset_summary_v0",
        "output": str(args.output.resolve()),
        "source_index": str(args.input.resolve()),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "row_count": len(rows),
        "scene_count": len({str(row["scene_id"]) for row in rows}),
        "unique_encoded_frames": len(unique_paths),
        "spatial_grid": args.spatial_grid,
        "spatial_dim": int(spatial_unique.shape[1]) if spatial_unique is not None else None,
        "history_frames": args.history_frames,
        "history_dim": int(start_history.shape[1]) if start_history is not None else None,
        "goal_present_rows": int(goal_present.sum()),
        "primitive_names": primitive_names,
        "candidate_collision_rate": float(collision.mean()),
        "mean_random_regret": float((cost.mean(axis=1) - cost.min(axis=1)).mean()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
