#!/usr/bin/env python3
"""Train a marker-return action head from closed-loop DAgger states."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    ACTION_NAMES,
    read_jsonl,
)
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AValueFieldActionHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _infer_scene_seed,
)
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_action_correction import (  # noqa: E402
    _build_correction_examples,
    _json_safe_arg,
)
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_action import (  # noqa: E402
    _load_value_action_head,
    _load_value_extractor_head,
)
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _load_latent_memory_updater,
    _load_value_field_head,
)
from scripts.train_jepa_phase3a_value_map_planner import (  # noqa: E402
    _load_value_map_planner_head,
)


@torch.no_grad()
def _evaluate_action_head(
    action_head: Phase3AValueFieldActionHead,
    examples,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    action_head.eval()
    dataset = TensorDataset(
        examples.memories,
        examples.target_fields,
        examples.sparse_probabilities,
        examples.planned_actions,
        examples.actions,
        examples.weights,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    action_matches = 0
    planned_matches = 0
    correction_total = 0
    correction_matches = 0
    keep_total = 0
    keep_matches = 0
    action_counts = {name: 0 for name in ACTION_NAMES}
    planned_counts = {name: 0 for name in ACTION_NAMES}
    predicted_counts = {name: 0 for name in ACTION_NAMES}
    for memory, target, sparse_prob, planned_action, action, weight in loader:
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        planned_action = planned_action.to(device)
        action = action.to(device)
        weight = weight.to(device)
        logits = action_head(memory, target, sparse_prob)
        loss_items = F.cross_entropy(logits, action, reduction="none")
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        pred = logits.argmax(dim=1)
        match = pred == action
        planned_match = planned_action == action
        correction_mask = planned_action != action
        keep_mask = planned_action == action
        total += int(memory.shape[0])
        loss_total += float(loss.item()) * int(memory.shape[0])
        action_matches += int(match.sum().item())
        planned_matches += int(planned_match.sum().item())
        correction_total += int(correction_mask.sum().item())
        correction_matches += int((match & correction_mask).sum().item())
        keep_total += int(keep_mask.sum().item())
        keep_matches += int((match & keep_mask).sum().item())
        for action_index in action.detach().cpu().tolist():
            action_counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in planned_action.detach().cpu().tolist():
            planned_counts[ACTION_NAMES[int(action_index)]] += 1
        for action_index in pred.detach().cpu().tolist():
            predicted_counts[ACTION_NAMES[int(action_index)]] += 1
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "action_match": action_matches / max(total, 1),
        "planned_action_match": planned_matches / max(total, 1),
        "correction_examples": correction_total,
        "correction_action_match": correction_matches / max(correction_total, 1),
        "keep_examples": keep_total,
        "keep_action_match": keep_matches / max(keep_total, 1),
        "action_counts": action_counts,
        "planned_action_counts": planned_counts,
        "predicted_action_counts": predicted_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
    parser.add_argument("--latent-value-field-head", type=Path, required=True)
    parser.add_argument("--latent-value-extractor-head", type=Path, required=True)
    parser.add_argument("--latent-value-map-planner-head", type=Path, required=True)
    parser.add_argument("--rollout-value-action-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--max-candidates-per-state", type=int, default=None)
    parser.add_argument("--example-filter", choices=("marker_return", "latent_marker_seen"), default="marker_return")
    parser.add_argument("--turn-oscillation-breaker", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mismatch-weight", type=float, default=4.0)
    parser.add_argument("--regret-weight", type=float, default=1.0)
    parser.add_argument("--latent-map-marker-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--target-top-k", type=int, default=16)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--sparse-target-top-k", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=256)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.horizon < 1:
        raise SystemExit("--horizon must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit("--label-smoothing must be in [0, 1)")
    if args.mismatch_weight < 0.0:
        raise SystemExit("--mismatch-weight must be non-negative")
    if args.regret_weight < 0.0:
        raise SystemExit("--regret-weight must be non-negative")
    if not 0.0 <= args.latent_map_marker_threshold <= 1.0:
        raise SystemExit("--latent-map-marker-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_blocked_threshold <= 1.0:
        raise SystemExit("--latent-memory-blocked-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_free_threshold <= 1.0:
        raise SystemExit("--latent-memory-free-threshold must be in [0, 1]")
    if not 0.0 <= args.latent_memory_marker_threshold <= 1.0:
        raise SystemExit("--latent-memory-marker-threshold must be in [0, 1]")
    if args.target_top_k < 1:
        raise SystemExit("--target-top-k must be positive")
    if args.sparse_target_top_k < 1:
        raise SystemExit("--sparse-target-top-k must be positive")
    if args.max_candidates_per_state is not None and args.max_candidates_per_state < 1:
        raise SystemExit("--max-candidates-per-state must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_seed = _infer_scene_seed(args.train_data)
    validation_seed = _infer_scene_seed(args.validation_data)
    if train_seed is None or validation_seed is None:
        raise SystemExit("could not infer train/validation scene seeds")
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)

    base_model, base_report = load_model(args.base_checkpoint, device=device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    latent_map_head, latent_map_report = _load_latent_map_head(
        args.latent_map_head,
        base_model=base_model,
        device=device,
    )
    latent_memory_updater, latent_memory_report = _load_latent_memory_updater(
        args.latent_memory_updater,
        model=base_model,
        device=device,
    )
    value_field_head, value_field_report = _load_value_field_head(
        args.latent_value_field_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    extractor_head, extractor_report = _load_value_extractor_head(
        args.latent_value_extractor_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    value_map_planner_head, planner_report = _load_value_map_planner_head(
        args.latent_value_map_planner_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    rollout_value_action_head, rollout_action_report = _load_value_action_head(
        args.rollout_value_action_head,
        fallback_memory_size=args.memory_size,
        device=device,
    )
    for name, size in (
        ("latent memory updater", latent_memory_updater.memory_size),
        ("value field head", value_field_head.memory_size),
        ("value extractor head", extractor_head.memory_size),
        ("value-map planner head", value_map_planner_head.memory_size),
        ("rollout value-action head", rollout_value_action_head.memory_size),
    ):
        if int(size) != int(args.memory_size):
            raise SystemExit(f"--memory-size must match {name} size ({size})")

    common_build_args = {
        "width": args.width_cells,
        "height": args.height_cells,
        "view_size": args.view_size,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "base_model": base_model,
        "latent_map_head": latent_map_head,
        "latent_memory_updater": latent_memory_updater,
        "value_field_head": value_field_head,
        "extractor_head": extractor_head,
        "value_map_planner_head": value_map_planner_head,
        "value_action_head": rollout_value_action_head,
        "marker_action_return": True,
        "marker_action_require_local_evidence": False,
        "turn_oscillation_breaker": bool(args.turn_oscillation_breaker),
        "blocked_threshold": args.latent_memory_blocked_threshold,
        "free_threshold": args.latent_memory_free_threshold,
        "marker_threshold": args.latent_memory_marker_threshold,
        "latent_map_marker_threshold": args.latent_map_marker_threshold,
        "target_threshold": args.target_threshold,
        "target_top_k": args.target_top_k,
        "extractor_threshold": args.extractor_threshold,
        "sparse_target_top_k": args.sparse_target_top_k,
        "regret_weight": args.regret_weight,
        "mismatch_weight": args.mismatch_weight,
        "max_candidates_per_state": args.max_candidates_per_state,
        "example_filter": args.example_filter,
        "device": device,
    }
    train_examples = _build_correction_examples(
        train_rows,
        scene_seed=train_seed,
        max_episodes=args.max_train_episodes,
        **common_build_args,
    )
    validation_examples = _build_correction_examples(
        validation_rows,
        scene_seed=validation_seed,
        max_episodes=args.max_validation_episodes,
        **common_build_args,
    )

    train_dataset = TensorDataset(
        train_examples.memories,
        train_examples.target_fields,
        train_examples.sparse_probabilities,
        train_examples.actions,
        train_examples.weights,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    action_head = Phase3AValueFieldActionHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
        action_dim=len(ACTION_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(
        action_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = (-1.0, -1.0, -1.0, float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, target, sparse_prob, action, weight = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, target, sparse_prob, action, weight = next(iterator)
        memory = memory.to(device)
        target = target.to(device)
        sparse_prob = sparse_prob.to(device)
        action = action.to(device)
        weight = weight.to(device)
        logits = action_head(memory, target, sparse_prob)
        loss_items = F.cross_entropy(
            logits,
            action,
            reduction="none",
            label_smoothing=args.label_smoothing,
        )
        loss = (loss_items * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate_action_head(
                action_head,
                validation_examples,
                batch_size=args.batch_size,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["action_match"]),
                float(metrics["correction_action_match"]),
                float(metrics["keep_action_match"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in action_head.state_dict().items()
                }

    final_validation = _evaluate_action_head(
        action_head,
        validation_examples,
        batch_size=args.batch_size,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        action_head.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation

    report = {
        "schema": "jepa_phase3a_marker_action_dagger_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "latent_value_field_head": str(args.latent_value_field_head.resolve()),
        "latent_value_field_completed_steps": value_field_report.get("completed_steps"),
        "latent_value_extractor_head": str(args.latent_value_extractor_head.resolve()),
        "latent_value_extractor_completed_steps": extractor_report.get("completed_steps"),
        "latent_value_map_planner_head": str(
            args.latent_value_map_planner_head.resolve()
        ),
        "latent_value_map_planner_completed_steps": planner_report.get(
            "completed_steps"
        ),
        "rollout_value_action_head": str(args.rollout_value_action_head.resolve()),
        "rollout_value_action_completed_steps": rollout_action_report.get(
            "completed_steps"
        ),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_examples.actions)),
        "validation_examples": int(len(validation_examples.actions)),
        "example_filter": args.example_filter,
        "completed_steps": args.optimization_steps,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "logs": logs,
        "args": {
            key: _json_safe_arg(value)
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
            "action_dim": len(ACTION_NAMES),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_head_state_dict": action_head.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
