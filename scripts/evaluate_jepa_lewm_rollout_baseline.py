#!/usr/bin/env python3
"""Evaluate the existing pooled LeWM rollout planner on the JEPA benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _encode_frames  # noqa: E402


def _load_rows(path: Path, max_rows: int) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def _selection_metrics(rows: list[dict], selected_indices: list[int]) -> dict:
    selected = [
        row["counterfactual_candidates"][index]
        for row, index in zip(rows, selected_indices, strict=True)
    ]
    return {
        "row_count": len(rows),
        "selected_enters_grid_unsafe_rate": float(
            np.mean([candidate["enters_grid_unsafe"] for candidate in selected])
        ),
        "selected_ends_grid_unsafe_rate": float(
            np.mean([candidate["ends_grid_unsafe"] for candidate in selected])
        ),
        "selected_target_recoverable_rate": float(
            np.mean(
                [
                    candidate["target_recoverable"]
                    for candidate in selected
                    if candidate["target_recoverable"] is not None
                ]
            )
        ),
        "mean_target_progress_m": float(
            np.mean(
                [
                    candidate["target_progress_m"]
                    for candidate in selected
                    if candidate["target_progress_m"] is not None
                ]
            )
        ),
        "safe_positive_progress_rate": float(
            np.mean(
                [
                    candidate["target_progress_m"] is not None
                    and candidate["target_progress_m"] > 0.0
                    and not candidate["enters_grid_unsafe"]
                    and not candidate["ends_grid_unsafe"]
                    and candidate["target_recoverable"] is not False
                    for candidate in selected
                ]
            )
        ),
        "oracle_sequence_match_rate": float(
            np.mean(
                [
                    index == int(row["counterfactual_oracle_index"])
                    for row, index in zip(rows, selected_indices, strict=True)
                ]
            )
        ),
    }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
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
            max_seq_len=None,
            sigreg_lambda=None,
        ),
        device,
    )
    model.eval()
    rows = _load_rows(args.input, args.max_rows)
    if not rows:
        raise SystemExit("benchmark input is empty")
    input_rows = len(rows)
    targetless_rows = sum(row.get("counterfactual_target_cell_id") is None for row in rows)
    missing_matched_goal_rows = sum(
        row.get("counterfactual_target_cell_id") is not None
        and row.get("local_target_frame") is None
        for row in rows
    )
    if missing_matched_goal_rows:
        raise SystemExit(
            f"{missing_matched_goal_rows} targeted rows lack matched local-target frames"
        )
    rows = [row for row in rows if row.get("counterfactual_target_cell_id") is not None]
    if not rows:
        raise SystemExit("benchmark contains no matched local-target rows")

    paths = list(
        dict.fromkeys(
            [Path(row["start_frame"]) for row in rows]
            + [Path(row["local_target_frame"]) for row in rows]
        )
    )
    raw, proj = _encode_frames(model, paths, device, args.batch_size)
    path_index = {path: index for index, path in enumerate(paths)}
    selected_indices = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        start_raw = torch.from_numpy(
            np.stack([raw[path_index[Path(row["start_frame"])]] for row in batch])
        ).float().to(device)
        goal_proj = torch.from_numpy(
            np.stack([proj[path_index[Path(row["local_target_frame"])]] for row in batch])
        ).float().to(device)
        action_sequences = torch.tensor(
            [
                [candidate["active_blocks"] for candidate in row["counterfactual_candidates"]]
                for row in batch
            ],
            dtype=torch.float32,
            device=device,
        )
        groups, candidates, horizon, action_dim = action_sequences.shape
        predicted = model.plan_rollout(
            start_raw[:, None, :].expand(groups, candidates, start_raw.shape[-1]).reshape(
                groups * candidates, start_raw.shape[-1]
            ),
            action_sequences.reshape(groups * candidates, horizon, action_dim),
        )
        predicted_last = predicted[:, -1] if predicted.dim() == 3 else predicted
        costs = torch.linalg.vector_norm(
            predicted_last
            - goal_proj[:, None, :]
            .expand(groups, candidates, goal_proj.shape[-1])
            .reshape(groups * candidates, goal_proj.shape[-1]),
            dim=-1,
        ).reshape(groups, candidates)
        selected_indices.extend(costs.argmin(dim=1).cpu().tolist())
        print(f"rows {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)

    report = {
        "schema": "jepa_lewm_rollout_baseline_v0",
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "input_rows": input_rows,
        "targetless_recovery_rows_excluded": targetless_rows,
        "metrics": _selection_metrics(rows, selected_indices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
