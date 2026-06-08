#!/usr/bin/env python3
"""Compare trained candidate scorers by goal source, sliced by decision type.

Each (label, checkpoint, eval-npz) triple is evaluated with the same deployed
selection rule as the trainer, then regret/collision/progress are reported on
subsets (all / goal-present / branch / recovery / branch&goal-present) so the
goal-conditioning effect is not diluted by goal-less recovery rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pathlib import PosixPath

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.action_ranker import TaskAlignedCandidateScorer  # noqa: E402
from train_task_aligned_candidate_scorer import _load_dataset, _predict  # noqa: E402


def _deployed_selection(
    dataset: dict,
    predictions: dict[str, torch.Tensor],
    *,
    progress_weight: float,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    heading_weight: float,
) -> torch.Tensor:
    goal_present = dataset["goal_present"].unsqueeze(1)
    task_cost = torch.where(
        goal_present,
        -progress_weight * predictions["progress"] + heading_weight * predictions["heading"],
        torch.zeros_like(predictions["progress"]),
    )
    predicted_cost = (
        task_cost
        + collision_penalty * predictions["collision_probability"]
        + clearance_penalty * (clearance_target_m - predictions["clearance"]).clamp_min(0.0)
    )
    return predicted_cost.argmin(dim=1)


def _subset_metrics(dataset: dict, selected: torch.Tensor, mask: torch.Tensor) -> dict:
    if int(mask.sum()) == 0:
        return {"rows": 0}
    idx = selected.long().unsqueeze(1)
    selected_cost = dataset["cost"].gather(1, idx).squeeze(1)
    oracle_cost = dataset["cost"].min(dim=1).values
    random_cost = dataset["cost"].mean(dim=1)
    regret = (selected_cost - oracle_cost)[mask]
    random_regret = (random_cost - oracle_cost)[mask]
    collision = dataset["collision"].gather(1, idx).squeeze(1)[mask].float()
    goal_present = dataset["goal_present"][mask]
    progress = dataset["progress"].gather(1, idx).squeeze(1)[mask]
    mean_rr = random_regret.mean().clamp_min(1e-8)
    out = {
        "rows": int(mask.sum()),
        "mean_regret": float(regret.mean()),
        "regret_ratio_vs_random": float(regret.mean() / mean_rr),
        "selected_collision_rate": float(collision.mean()),
        "goal_present_rows": int(goal_present.sum()),
    }
    if int(goal_present.sum()) > 0:
        out["mean_target_progress_m"] = float(progress[goal_present].mean())
    return out


def _decision_mask(dataset: dict, token: str) -> torch.Tensor:
    return torch.tensor([token in dt for dt in dataset["decision_types"]], dtype=torch.bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL:CHECKPOINT:EVAL_NPZ",
        help="Repeatable. Each is label:checkpoint.pt:eval.npz",
    )
    parser.add_argument("--latent-space", default="spatial")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-weight", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=3.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--clearance-penalty", type=float, default=0.5)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.serialization.add_safe_globals([PosixPath])
    report = {"schema": "task_aligned_goal_controls_v0", "models": []}
    for spec in args.model:
        label, checkpoint, eval_npz = spec.split(":", 2)
        ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        dataset = _load_dataset(Path(eval_npz), args.latent_space)
        head = TaskAlignedCandidateScorer(
            latent_dim=int(ckpt["latent_dim"]),
            cmd_dim=int(ckpt["cmd_dim"]),
            hidden=int(ckpt["hidden"]),
            dropout=float(ckpt["dropout"]),
        ).to(device)
        head.load_state_dict(ckpt["head_state_dict"])
        predictions = _predict(head, dataset, ckpt["regression_stats"], batch_size=512, device=device)
        selected = _deployed_selection(
            dataset,
            predictions,
            progress_weight=args.progress_weight,
            collision_penalty=args.collision_penalty,
            clearance_target_m=args.clearance_target_m,
            clearance_penalty=args.clearance_penalty,
            heading_weight=args.heading_weight,
        )
        n = len(dataset["scene_ids"])
        all_mask = torch.ones(n, dtype=torch.bool)
        goal_mask = dataset["goal_present"]
        branch_mask = _decision_mask(dataset, "branch")
        recovery_mask = _decision_mask(dataset, "recovery")
        report["models"].append(
            {
                "label": label,
                "eval_npz": str(Path(eval_npz).resolve()),
                "checkpoint": str(Path(checkpoint).resolve()),
                "subsets": {
                    "all": _subset_metrics(dataset, selected, all_mask),
                    "goal_present": _subset_metrics(dataset, selected, goal_mask),
                    "branch": _subset_metrics(dataset, selected, branch_mask),
                    "branch_and_goal_present": _subset_metrics(dataset, selected, branch_mask & goal_mask),
                    "recovery": _subset_metrics(dataset, selected, recovery_mask),
                },
            }
        )
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
