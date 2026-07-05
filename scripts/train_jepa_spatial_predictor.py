#!/usr/bin/env python3
"""Train and evaluate the Phase 2A frozen-encoder spatial-token predictor."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.predictor import TransformerPredictor  # noqa: E402
from lewm.models.spatial_predictor import (  # noqa: E402
    SpatialTokenPredictor,
    trainable_parameter_count,
)
from probe_lewm_checkpoint import load_model  # noqa: E402


def _load_rows(path: Path, max_rows: int) -> tuple[list[dict], dict]:
    input_rows = []
    with path.open() as stream:
        for line in stream:
            input_rows.append(json.loads(line))
    complete_valid_rows = [
        row
        for row in input_rows
        if row.get("complete_valid_future_sequence", True)
        and all(path is not None for path in row["future_frames"])
    ]
    usable_rows = complete_valid_rows
    if max_rows > 0:
        usable_rows = usable_rows[:max_rows]
    if not usable_rows:
        raise ValueError(f"spatial future dataset is empty: {path}")
    audit = {
        "input_candidate_sequences": len(input_rows),
        "complete_valid_sequences_before_limit": len(complete_valid_rows),
        "usable_complete_valid_sequences": len(usable_rows),
        "excluded_incomplete_or_invalid_sequences": (
            len(input_rows) - len(complete_valid_rows)
        ),
        "excluded_by_max_rows": len(complete_valid_rows) - len(usable_rows),
        "usable_fraction": len(usable_rows) / len(input_rows),
    }
    return usable_rows, audit


def _image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(array)


@torch.no_grad()
def _encode_paths(
    encoder,
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[Path, torch.Tensor]:
    encoded = {}
    encoder.eval()
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack([_image_tensor(path) for path in batch_paths]).to(device)
        tokens = encoder.forward_tokens(images)[:, 1:].cpu()
        encoded.update(zip(batch_paths, tokens, strict=True))
        print(f"encoded {min(offset + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
    return encoded


def _token_cache(rows: list[dict], encoder, device: torch.device, batch_size: int) -> dict:
    paths = []
    for row in rows:
        paths.append(Path(row["start_frame"]))
        paths.extend(Path(path) for path in row["future_frames"])
        if row.get("goal_frame") is not None:
            paths.append(Path(row["goal_frame"]))
    unique = list(dict.fromkeys(paths))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} spatial dataset images are missing: {missing[0]}")
    return _encode_paths(encoder, unique, device=device, batch_size=batch_size)


def _batch(rows: list[dict], indices: list[int], cache: dict, device: torch.device):
    selected = [rows[index] for index in indices]
    start = torch.stack([cache[Path(row["start_frame"])] for row in selected]).to(device)
    future = torch.stack(
        [
            torch.stack([cache[Path(path)] for path in row["future_frames"]])
            for row in selected
        ]
    ).to(device)
    actions = torch.tensor(
        [row["active_blocks"] for row in selected],
        dtype=torch.float32,
        device=device,
    )
    return selected, start, actions, future


def _safe_positive_progress(labels: dict) -> bool:
    progress = labels["target_progress_m"]
    return (
        progress is not None
        and float(progress) > 0.0
        and not bool(labels["enters_grid_unsafe"])
        and not bool(labels["ends_grid_unsafe"])
        and labels["target_recoverable"] is not False
    )


def _selection_metrics(rows: list[dict], costs: list[float]) -> dict:
    grouped: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    for row, cost in zip(rows, costs, strict=True):
        if row.get("goal_present") and row.get("goal_frame") is not None:
            grouped[int(row["source_index"])].append((row, float(cost)))
    selected = [min(candidates, key=lambda item: item[1])[0] for candidates in grouped.values()]
    if not selected:
        return {"target_groups": 0}
    labels = [row["consequence_labels"] for row in selected]
    group_sizes = [len(candidates) for candidates in grouped.values()]
    return {
        "target_groups": len(selected),
        "candidate_sequences_considered": len(costs),
        "minimum_candidates_per_target_group": min(group_sizes),
        "mean_candidates_per_target_group": float(np.mean(group_sizes)),
        "maximum_candidates_per_target_group": max(group_sizes),
        "conditional_on_complete_valid_future_observations": True,
        "selected_enters_grid_unsafe_rate": float(
            np.mean([label["enters_grid_unsafe"] for label in labels])
        ),
        "selected_ends_grid_unsafe_rate": float(
            np.mean([label["ends_grid_unsafe"] for label in labels])
        ),
        "selected_target_recoverable_rate": float(
            np.mean([label["target_recoverable"] for label in labels])
        ),
        "mean_target_progress_m": float(
            np.mean([label["target_progress_m"] for label in labels])
        ),
        "safe_positive_progress_rate": float(
            np.mean([_safe_positive_progress(label) for label in labels])
        ),
        "oracle_sequence_match_rate": float(
            np.mean([row["is_oracle_candidate"] for row in selected])
        ),
    }


@torch.no_grad()
def _evaluate(
    predictor: SpatialTokenPredictor,
    rows: list[dict],
    cache: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    predictor.eval()
    rollout_losses = []
    teacher_losses = []
    persistence_losses = []
    rollout_step_losses: list[list[float]] = []
    teacher_step_losses: list[list[float]] = []
    persistence_step_losses: list[list[float]] = []
    goal_costs = []
    for offset in range(0, len(rows), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(rows))))
        selected, start, actions, future = _batch(rows, indices, cache, device)
        rollout = predictor.rollout(start, actions)
        teacher = predictor.rollout(
            start,
            actions,
            teacher_tokens=future,
            teacher_prob=1.0,
        )
        rollout_losses.extend(
            (rollout - future).square().mean(dim=(1, 2, 3)).cpu().tolist()
        )
        teacher_losses.extend(
            (teacher - future).square().mean(dim=(1, 2, 3)).cpu().tolist()
        )
        persistence = start[:, None].expand_as(future)
        persistence_losses.extend(
            (persistence - future).square().mean(dim=(1, 2, 3)).cpu().tolist()
        )
        for storage, values in (
            (rollout_step_losses, (rollout - future).square().mean(dim=(2, 3))),
            (teacher_step_losses, (teacher - future).square().mean(dim=(2, 3))),
            (
                persistence_step_losses,
                (persistence - future).square().mean(dim=(2, 3)),
            ),
        ):
            while len(storage) < values.shape[1]:
                storage.append([])
            for step in range(values.shape[1]):
                storage[step].extend(values[:, step].cpu().tolist())
        for row, predicted_final in zip(selected, rollout[:, -1], strict=True):
            if row.get("goal_present") and row.get("goal_frame") is not None:
                goal = cache[Path(row["goal_frame"])].to(device)
                goal_costs.append(float((predicted_final - goal).square().mean()))
            else:
                goal_costs.append(float("inf"))
    free_running_mse = float(np.mean(rollout_losses))
    persistence_mse = float(np.mean(persistence_losses))
    return {
        "teacher_forced_token_mse": float(np.mean(teacher_losses)),
        "free_running_token_mse": free_running_mse,
        "persistence_token_mse": persistence_mse,
        "free_running_vs_persistence_mse_ratio": (
            free_running_mse / persistence_mse
            if persistence_mse > 0.0
            else float("inf")
        ),
        "free_running_beats_persistence": free_running_mse < persistence_mse,
        "per_horizon_step": [
            {
                "step": step + 1,
                "teacher_forced_token_mse": float(np.mean(teacher_step_losses[step])),
                "free_running_token_mse": float(np.mean(rollout_step_losses[step])),
                "persistence_token_mse": float(np.mean(persistence_step_losses[step])),
                "free_running_beats_persistence": (
                    float(np.mean(rollout_step_losses[step]))
                    < float(np.mean(persistence_step_losses[step]))
                ),
            }
            for step in range(len(rollout_step_losses))
        ],
        "selection": _selection_metrics(rows, goal_costs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--allow-scene-overlap", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    train_rows, train_input_audit = _load_rows(args.train_data, args.max_train_rows)
    eval_rows, eval_input_audit = _load_rows(args.eval_data, args.max_eval_rows)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    eval_scenes = {str(row["scene_id"]) for row in eval_rows}
    overlap = sorted(train_scenes & eval_scenes)
    if overlap and not args.allow_scene_overlap:
        raise SystemExit(f"train/eval scene overlap: {overlap[:8]}")

    model, _config = load_model(
        SimpleNamespace(
            checkpoint=args.encoder_checkpoint.resolve(),
            max_seq_len=None,
            sigreg_lambda=None,
        ),
        device,
    )
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    combined_rows = train_rows + eval_rows
    cache = _token_cache(
        combined_rows,
        model.encoder.vis_enc,
        device,
        args.encode_batch_size,
    )
    sample_tokens = next(iter(cache.values()))
    predictor = SpatialTokenPredictor(
        latent_dim=int(sample_tokens.shape[-1]),
        cmd_dim=len(train_rows[0]["active_blocks"][0]),
        num_spatial_tokens=int(sample_tokens.shape[-2]),
        n_layers=args.layers,
        n_heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(args.epochs):
        predictor.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset : offset + args.batch_size]
            _selected, start, actions, future = _batch(train_rows, indices, cache, device)
            predicted = predictor.rollout(
                start,
                actions,
                teacher_tokens=future,
                teacher_prob=args.teacher_prob,
            )
            loss = (predicted - future).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        evaluation = _evaluate(
            predictor,
            eval_rows,
            cache,
            batch_size=args.batch_size,
            device=device,
        )
        record = {
            "epoch": epoch + 1,
            "train_token_mse": float(np.mean(losses)),
            **evaluation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    pooled_reference = TransformerPredictor(
        latent_dim=int(sample_tokens.shape[-1]),
        cmd_dim=len(train_rows[0]["active_blocks"][0]),
        n_layers=args.layers,
        n_heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    )
    report = {
        "schema": "jepa_spatial_predictor_training_v0",
        "phase": "2A_frozen_existing_encoder_patch_token_diagnostic",
        "train_data": str(args.train_data.resolve()),
        "eval_data": str(args.eval_data.resolve()),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_input_audit": train_input_audit,
        "eval_input_audit": eval_input_audit,
        "scene_overlap": overlap,
        "device": str(device),
        "spatial_predictor_parameters": trainable_parameter_count(predictor),
        "pooled_predictor_reference_parameters": trainable_parameter_count(
            pooled_reference
        ),
        "training_uses_privileged_consequence_labels": False,
        "final": history[-1],
        "history": history,
        "limitations": [
            "encoder is frozen and was originally trained through a pooled objective",
            "kinematic future observations are not physics-validated",
            "selection cost is direct position-aligned patch-token MSE to the goal image",
            (
                "token loss and selection metrics exclude renderer-invalid future "
                "observations; they are representation diagnostics, not safety claims"
            ),
        ],
    }
    payload = {
        "report": report,
        "predictor_state_dict": {
            name: value.detach().cpu() for name, value in predictor.state_dict().items()
        },
        "model_config": {
            "latent_dim": predictor.latent_dim,
            "num_spatial_tokens": predictor.num_spatial_tokens,
            "cmd_dim": len(train_rows[0]["active_blocks"][0]),
            "layers": args.layers,
            "heads": args.heads,
            "dim_head": args.dim_head,
            "mlp_dim": args.mlp_dim,
            "dropout": args.dropout,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
