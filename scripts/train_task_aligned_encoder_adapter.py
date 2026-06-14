#!/usr/bin/env python3
"""Fine-tune final LeWM vision blocks with direct task-aligned candidate targets."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.action_ranker import (  # noqa: E402
    TaskAlignedCandidateScorer,
    task_aligned_candidate_loss,
)
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import _load_image_tensor  # noqa: E402
from train_task_aligned_candidate_scorer import (  # noqa: E402
    _controls,
    _denormalize,
    _normalize,
    _selection_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_task_aligned_encoder_adapter")


class DecisionImageDataset(Dataset):
    def __init__(self, path: Path, max_rows: int = 0) -> None:
        self.rows = []
        with path.open() as stream:
            for line in stream:
                if max_rows > 0 and len(self.rows) >= max_rows:
                    break
                self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"no rows in {path}")
        self.primitive_names = [
            str(item["primitive_name"])
            for item in self.rows[0]["counterfactual_candidates"]
        ]
        self.actions = torch.tensor(
            [
                item["active_block"]
                for item in self.rows[0]["counterfactual_candidates"]
            ],
            dtype=torch.float32,
        )
        self.scene_ids = [str(row["scene_id"]) for row in self.rows]
        self.logged_primitives = [str(row["primitive_name"]) for row in self.rows]
        self.goal_present = torch.tensor(
            [row.get("target_frame") is not None for row in self.rows],
            dtype=torch.bool,
        )
        self.collision = self._candidate_tensor("collided", torch.bool)
        self.progress = self._candidate_tensor("target_progress_m", torch.float32)
        self.heading = self._candidate_tensor("heading_error_rad", torch.float32)
        self.clearance = self._candidate_tensor("clearance_m", torch.float32)
        self.cost = self._candidate_tensor("cost", torch.float32)

    def _candidate_tensor(self, field: str, dtype: torch.dtype) -> torch.Tensor:
        rows = []
        for row in self.rows:
            by_name = {
                str(item["primitive_name"]): item
                for item in row["counterfactual_candidates"]
            }
            rows.append(
                [
                    0.0 if by_name[name][field] is None else by_name[name][field]
                    for name in self.primitive_names
                ]
            )
        return torch.tensor(rows, dtype=dtype)

    def control_view(self) -> dict:
        return {
            "primitive_names": self.primitive_names,
            "cost": self.cost,
            "collision": self.collision,
            "progress": self.progress,
            "goal_present": self.goal_present,
            "scene_ids": self.scene_ids,
            "logged_primitives": self.logged_primitives,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        start = torch.from_numpy(
            np.array(_load_image_tensor(Path(row["start_frame"])), copy=True)
        ).float().div_(255.0)
        target_frame = row.get("target_frame")
        goal = (
            torch.from_numpy(
                np.array(_load_image_tensor(Path(target_frame)), copy=True)
            ).float().div_(255.0)
            if target_frame is not None
            else torch.zeros_like(start)
        )
        return {
            "index": index,
            "start": start,
            "goal": goal,
            "goal_present": self.goal_present[index],
            "collision": self.collision[index],
            "progress": self.progress[index],
            "heading": self.heading[index],
            "clearance": self.clearance[index],
        }


def _stats(dataset: DecisionImageDataset) -> dict[str, tuple[float, float]]:
    goal_mask = dataset.goal_present.unsqueeze(1).expand_as(dataset.progress)
    result = {}
    for name in ("progress", "heading"):
        values = getattr(dataset, name)[goal_mask]
        result[name] = (float(values.mean()), max(float(values.std()), 1e-6))
    values = dataset.clearance
    result["clearance"] = (float(values.mean()), max(float(values.std()), 1e-6))
    return result


def _enable_adapter(model: torch.nn.Module, unfreeze_blocks: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = model.encoder.vis_enc.blocks
    if unfreeze_blocks <= 0 or unfreeze_blocks > len(blocks):
        raise ValueError(f"unfreeze_blocks must be in [1, {len(blocks)}]")
    for block in blocks[-unfreeze_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.encoder.vis_enc.norm.parameters():
        parameter.requires_grad_(True)
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _forward(
    model,
    head: TaskAlignedCandidateScorer,
    batch: dict,
    actions: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    start = batch["start"].to(device, non_blocking=True)
    goal = batch["goal"].to(device, non_blocking=True)
    goal_present = batch["goal_present"].to(device, non_blocking=True)
    images = torch.cat([start, goal], dim=0)
    latents = model.encoder(images, None)
    z_start, z_goal = latents.chunk(2, dim=0)
    z_goal = z_goal * goal_present.unsqueeze(1).to(z_goal.dtype)
    groups, latent_dim = z_start.shape
    primitives, cmd_dim = actions.shape
    return head(
        z_start[:, None, :].expand(groups, primitives, latent_dim),
        z_goal[:, None, :].expand(groups, primitives, latent_dim),
        goal_present[:, None].expand(groups, primitives),
        actions[None, :, :].expand(groups, primitives, cmd_dim),
    )


def _run_epoch(
    model,
    head: TaskAlignedCandidateScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    actions: torch.Tensor,
    stats: dict[str, tuple[float, float]],
    collision_pos_weight: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    head.train()
    total = 0.0
    count = 0
    for batch in loader:
        predictions = _forward(model, head, batch, actions, device)
        targets = {
            "collision": batch["collision"].to(device, non_blocking=True),
            "progress": _normalize(batch["progress"].to(device, non_blocking=True), stats["progress"]),
            "heading": _normalize(batch["heading"].to(device, non_blocking=True), stats["heading"]),
            "clearance": _normalize(batch["clearance"].to(device, non_blocking=True), stats["clearance"]),
        }
        loss, _components = task_aligned_candidate_loss(
            predictions,
            targets,
            goal_present=batch["goal_present"].to(device, non_blocking=True),
            collision_pos_weight=collision_pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch["index"])
        count += len(batch["index"])
    return total / count


@torch.no_grad()
def _evaluate(
    model,
    head: TaskAlignedCandidateScorer,
    loader: DataLoader,
    dataset: DecisionImageDataset,
    actions: torch.Tensor,
    stats: dict[str, tuple[float, float]],
    *,
    progress_weight: float,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    heading_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    head.eval()
    selected = torch.empty(len(dataset), dtype=torch.long)
    for batch in loader:
        predictions = _forward(model, head, batch, actions, device)
        progress = _denormalize(predictions["progress"], stats["progress"])
        heading = _denormalize(predictions["heading"], stats["heading"])
        clearance = _denormalize(predictions["clearance"], stats["clearance"])
        goal_present = batch["goal_present"].to(device).unsqueeze(1)
        task_cost = torch.where(
            goal_present,
            -progress_weight * progress + heading_weight * heading,
            torch.zeros_like(progress),
        )
        predicted_cost = (
            task_cost
            + collision_penalty * predictions["collision_logit"].sigmoid()
            + clearance_penalty * (clearance_target_m - clearance).clamp_min(0.0)
        )
        selected[batch["index"]] = predicted_cost.argmin(dim=1).cpu()
    return _selection_metrics(dataset.control_view(), selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--eval-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--progress-weight", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=3.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--clearance-penalty", type=float, default=0.5)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    train = DecisionImageDataset(args.train_index, args.max_train_rows)
    evaluation = DecisionImageDataset(args.eval_index, args.max_eval_rows)
    if train.primitive_names != evaluation.primitive_names:
        raise SystemExit("train/eval primitive order differs")
    model, _config = load_model(
        SimpleNamespace(
            checkpoint=args.checkpoint.resolve(),
            max_seq_len=args.max_seq_len,
            sigreg_lambda=args.sigreg_lambda,
        ),
        device,
    )
    trainable_encoder = _enable_adapter(model, args.unfreeze_blocks)
    head = TaskAlignedCandidateScorer(
        latent_dim=model.latent_dim,
        cmd_dim=train.actions.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    actions = train.actions.to(device)
    stats = _stats(train)
    positives = train.collision.float().sum()
    collision_pos_weight = (
        (train.collision.numel() - positives) / positives.clamp_min(1.0)
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                "lr": args.encoder_lr,
            },
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        evaluation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    controls = _controls(train.control_view(), evaluation.control_view())
    best_score = float("inf")
    best_epoch = -1
    best_metrics = None
    best_head = None
    best_adapter = None
    for epoch in range(args.epochs):
        loss = _run_epoch(
            model,
            head,
            train_loader,
            optimizer,
            actions,
            stats,
            collision_pos_weight,
            device,
        )
        metrics = _evaluate(
            model,
            head,
            eval_loader,
            evaluation,
            actions,
            stats,
            progress_weight=args.progress_weight,
            collision_penalty=args.collision_penalty,
            clearance_target_m=args.clearance_target_m,
            clearance_penalty=args.clearance_penalty,
            heading_weight=args.heading_weight,
            device=device,
        )
        score = (
            metrics["mean_regret"]
            + metrics["selected_collision_rate"]
            - metrics["mean_target_progress_m"]
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            best_adapter = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in trainable_encoder
            }
        logger.info(
            "ep=%d loss=%.4f regret=%.4f collision=%.3f progress=%+.4f",
            epoch,
            loss,
            metrics["mean_regret"],
            metrics["selected_collision_rate"],
            metrics["mean_target_progress_m"],
        )

    assert best_metrics is not None and best_head is not None and best_adapter is not None
    action_only = controls["action_only"]
    random = controls["random"]
    minimum_validity_passed = bool(
        best_metrics["mean_regret"] < action_only["mean_regret"]
        and best_metrics["selected_collision_rate"] < action_only["selected_collision_rate"]
        and best_metrics["mean_target_progress_m"] > action_only["mean_target_progress_m"]
    )
    promotion_gate_passed = bool(
        minimum_validity_passed
        and best_metrics["regret_ratio_vs_random"] <= 0.5
        and best_metrics["selected_collision_rate"] <= 0.05
        and best_metrics["mean_target_progress_m"] > random["mean_target_progress_m"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_head,
            "adapter_state_dict": best_adapter,
            "trainable_encoder_parameters": trainable_encoder,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "regression_stats": stats,
            "best_epoch": best_epoch,
            "best_eval_metrics": best_metrics,
            "controls": controls,
            "minimum_validity_passed": minimum_validity_passed,
            "promotion_gate_passed": promotion_gate_passed,
            "config": vars(args),
        },
        args.output,
    )
    report = {
        "schema": "task_aligned_encoder_adapter_report_v0",
        "checkpoint": str(args.output.resolve()),
        "seed": args.seed,
        "unfreeze_blocks": args.unfreeze_blocks,
        "trainable_encoder_parameter_tensors": len(trainable_encoder),
        "best_epoch": best_epoch,
        "best_eval_metrics": best_metrics,
        "controls": controls,
        "minimum_validity_passed": minimum_validity_passed,
        "promotion_gate_passed": promotion_gate_passed,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "eval_scene_count": len(set(evaluation.scene_ids)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
