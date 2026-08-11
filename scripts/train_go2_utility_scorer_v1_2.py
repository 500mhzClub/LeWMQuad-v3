#!/usr/bin/env python3
"""Train and qualify the shared utility scorer against oracle v1.2.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  **No predictor checkpoint is opened.**
Training and qualification use only the scorer-fit corpus's TRUE latent
trajectories.

Everything architectural, optimisational and procedural comes from the frozen
scorer contract ``d3211855…`` via its v1.2 versioning
(``lewm/oracle/go2_scorer_contract_v1_2.py``); this file only implements it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle.go2_scorer_contract_v1_2 import (  # noqa: E402
    SCORER, contract, contract_digest,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT_ROOT = ROOT / ".generated/go2_branch_corpus_v1_2"
PACKAGE_DIR = ROOT / ".generated/go2_utility_scorer_v1_2"
TOKENS, TOKEN_DIM, HORIZONS = 768, 1024, 4
ACTION_DIM, GOAL_DIM = 40, 3
TIE_TOLERANCE = 0.02          # frozen
WEIGHTS = SCORER["weights"]


# ----------------------------------------------------------------- the model --
class UtilityScorer(nn.Module):
    """Per-horizon shared trunk, attention pool over h, three separate heads."""

    def __init__(self, *, use_latent: bool, hidden: int = 512) -> None:
        super().__init__()
        self.use_latent = use_latent
        if use_latent:
            self.per_horizon = nn.Sequential(
                nn.Linear(TOKEN_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.attention = nn.Linear(hidden, 1)
        self.context = nn.Sequential(
            nn.Linear(ACTION_DIM + GOAL_DIM, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        fuse_in = hidden * (2 if use_latent else 1)
        self.fuse = nn.Sequential(nn.Linear(fuse_in, hidden), nn.SiLU())
        self.progress = nn.Linear(hidden, 1)
        self.safety = nn.Linear(hidden, 1)
        self.completion = nn.Linear(hidden, 1)

    def forward(self, latent, action_goal):
        parts = [self.context(action_goal)]
        if self.use_latent:
            per_h = self.per_horizon(latent)                    # (B, H, hidden)
            weights = torch.softmax(self.attention(per_h), dim=1)
            parts.insert(0, (per_h * weights).sum(dim=1))
        fused = self.fuse(torch.cat(parts, dim=-1))
        return (self.progress(fused).squeeze(-1),
                self.safety(fused).squeeze(-1),
                self.completion(fused).squeeze(-1))


def composite(progress, safety_logit, completion_logit):
    return (WEIGHTS["progress"] * progress
            + WEIGHTS["safety"] * torch.sigmoid(safety_logit)
            + WEIGHTS["completion"] * torch.sigmoid(completion_logit))


# ------------------------------------------------------------------- metrics --
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(target: np.ndarray, predicted: np.ndarray,
                               bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, error = 0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted >= lo) & (predicted < hi if hi < 1.0 else predicted <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        error += count * abs(float(predicted[mask].mean()) - float(target[mask].mean()))
        total += count
    return float(error / total) if total else float("nan")


def pairwise_ordering(states: list[str], true_u: np.ndarray, pred_u: np.ndarray,
                      tolerance: float = TIE_TOLERANCE) -> tuple[float, int]:
    correct = considered = 0
    by_state: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        by_state.setdefault(state, []).append(index)
    for indices in by_state.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                gap = true_u[a] - true_u[b]
                if abs(gap) <= tolerance:
                    continue
                considered += 1
                if (pred_u[a] - pred_u[b]) * gap > 0:
                    correct += 1
    return (correct / considered if considered else float("nan")), considered


def normalised_rank_regret(states: list[str], true_u: np.ndarray,
                           pred_u: np.ndarray) -> tuple[float, list[float]]:
    """(best true utility - true utility of the argmax-predicted) / spread."""

    by_state: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        by_state.setdefault(state, []).append(index)
    values = []
    for indices in by_state.values():
        if len(indices) < 2:
            continue
        truth = true_u[indices]
        chosen = indices[int(np.argmax(pred_u[indices]))]
        spread = float(truth.max() - truth.min())
        if spread <= 0:
            values.append(0.0)
            continue
        values.append(float((truth.max() - true_u[chosen]) / spread))
    return (float(np.mean(values)) if values else float("nan")), values


# ---------------------------------------------------------------- the corpus --
def load_corpus(pool: str) -> dict[str, Any]:
    out = OUT_ROOT / pool
    rows = [json.loads(l) for l in (out / "branch_rows.jsonl").read_text().splitlines()
            if l.strip()]
    index = json.loads((out / "latents_index.json").read_text())
    horizon = np.fromfile(out / "horizon.f16", dtype=np.float16).reshape(
        index["horizon_shape"])
    keys = {key: i for i, key in enumerate(index["horizon_keys"])}
    usable = []
    for row in rows:
        key = f"{row['state_id']}|{row['candidate']}"
        if row.get("valid") and key in keys:
            row["_latent_index"] = keys[key]
            usable.append(row)
    return {"rows": usable, "horizon": horizon, "index": index}


def features(rows: list[dict[str, Any]], horizon: np.ndarray, device):
    latent = torch.from_numpy(
        horizon[[r["_latent_index"] for r in rows]].astype(np.float32)).mean(dim=2)
    action = np.zeros((len(rows), ACTION_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        blocks = row["action_blocks"]
        flat = [v for block in blocks for v in block]
        action[i, :len(flat)] = flat[:ACTION_DIM]
    goal = np.asarray([[np.sin(r["goal"]["bearing_body_rad"]),
                        np.cos(r["goal"]["bearing_body_rad"]),
                        r["goal"]["range_m"]] for r in rows], dtype=np.float32)
    action_goal = torch.from_numpy(np.concatenate([action, goal], axis=-1))
    targets = {
        "progress": torch.tensor([r["progress"] for r in rows], dtype=torch.float32),
        "safety": torch.tensor([r["safety"] for r in rows], dtype=torch.float32),
        "completion": torch.tensor([r["completion"] for r in rows], dtype=torch.float32),
    }
    return (latent.to(device), action_goal.to(device),
            {k: v.to(device) for k, v in targets.items()})


def train(model, latent, action_goal, targets, *, device, budget) -> None:
    torch.manual_seed(int(budget["seed"]))
    optimiser = torch.optim.AdamW(model.parameters(), lr=budget["lr"],
                                  weight_decay=budget["weight_decay"])
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    n = latent.shape[0]
    generator = torch.Generator().manual_seed(int(budget["seed"]))
    for _epoch in range(budget["epochs"]):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, budget["batch"]):
            index = order[start:start + budget["batch"]]
            p, s, c = model(latent[index], action_goal[index])
            loss = (mse(p, targets["progress"][index])
                    + bce(s, targets["safety"][index])
                    + bce(c, targets["completion"][index]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), budget["grad_clip"])
            optimiser.step()


def evaluate(model, latent, action_goal, rows, targets) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        p, s, c = model(latent, action_goal)
        u = composite(p, s, c)
    progress = p.cpu().numpy()
    safety = torch.sigmoid(s).cpu().numpy()
    completion = torch.sigmoid(c).cpu().numpy()
    predicted_u = u.cpu().numpy()
    true_progress = targets["progress"].cpu().numpy()
    true_safety = targets["safety"].cpu().numpy()
    true_completion = targets["completion"].cpu().numpy()
    true_u = np.asarray([r["utility"] for r in rows], dtype=np.float64)
    states = [r["state_id"] for r in rows]
    accuracy, pairs = pairwise_ordering(states, true_u, predicted_u)
    regret, per_state = normalised_rank_regret(states, true_u, predicted_u)
    return {
        "rows": len(rows),
        "progress": {"spearman": spearman(true_progress, progress),
                     "mae": float(np.abs(true_progress - progress).mean())},
        "safety": {"auc_any_hazard": roc_auc((true_safety > 0).astype(float), safety),
                   "calibration_error": expected_calibration_error(true_safety, safety),
                   "mae": float(np.abs(true_safety - safety).mean())},
        "completion": {"prevalence": float(true_completion.mean()),
                       "auc": roc_auc(true_completion, completion),
                       "calibration_error": expected_calibration_error(
                           true_completion, completion)},
        "composite": {"pairwise_ordering_accuracy": accuracy,
                      "pairs_considered": pairs,
                      "normalised_rank_regret": regret,
                      "states": len(per_state)},
    }


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(key):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        return {"min": float(values.min()), "median": float(np.median(values)),
                "max": float(values.max()), "mean": float(values.mean()),
                "distinct": int(len(set(np.round(values, 6).tolist())))}
    return {"rows": len(rows),
            "progress": summary("progress"), "safety": summary("safety"),
            "utility": summary("utility"),
            "completion_positive": int(sum(r["completion"] for r in rows)),
            "completion_prevalence": float(np.mean([r["completion"] for r in rows])),
            "any_hazard_prevalence": float(np.mean([r["safety"] > 0 for r in rows]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="scorer_fit")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(args.pool)
    rows = corpus["rows"]
    fit_rows = [r for r in rows if r["split_role"] == "fit"]
    cal_rows = [r for r in rows if r["split_role"] == "calibration"]
    if not fit_rows or not cal_rows:
        raise SystemExit("empty fit or calibration split")
    fit_scenes = {r["scene_id"] for r in fit_rows}
    cal_scenes = {r["scene_id"] for r in cal_rows}
    if fit_scenes & cal_scenes:
        raise SystemExit("fit/calibration split is not scene-disjoint")

    budget = SCORER["training"]
    started = time.time()
    fit_features = features(fit_rows, corpus["horizon"], device)
    cal_features = features(cal_rows, corpus["horizon"], device)

    results = {}
    packages = {}
    for name, use_latent in (("latent", True), ("no_latent", False)):
        model = UtilityScorer(use_latent=use_latent).to(device)
        train(model, *fit_features, device=device, budget=budget)
        results[name] = {
            "fit": evaluate(model, *fit_features[:2], fit_rows, fit_features[2]),
            "calibration": evaluate(model, *cal_features[:2], cal_rows, cal_features[2]),
        }
        packages[name] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    latent_cal = results["latent"]["calibration"]
    baseline_cal = results["no_latent"]["calibration"]
    dominance = (latent_cal["composite"]["pairwise_ordering_accuracy"]
                 - baseline_cal["composite"]["pairwise_ordering_accuracy"])
    completion_degenerate = bool(
        label_distribution(fit_rows)["completion_positive"] == 0
        or label_distribution(cal_rows)["completion_positive"] == 0
        or label_distribution(fit_rows)["completion_prevalence"] == 1.0
        or label_distribution(cal_rows)["completion_prevalence"] == 1.0)

    criteria = {
        "progress_spearman_ge_0.50":
            latent_cal["progress"]["spearman"] >= 0.50,
        "safety_auc_ge_0.75": latent_cal["safety"]["auc_any_hazard"] >= 0.75,
        "safety_calibration_le_0.10":
            latent_cal["safety"]["calibration_error"] <= 0.10,
        "completion_auc_ge_0.75": latent_cal["completion"]["auc"] >= 0.75,
        "completion_calibration_le_0.10":
            latent_cal["completion"]["calibration_error"] <= 0.10,
        "composite_pairwise_ge_0.65":
            latent_cal["composite"]["pairwise_ordering_accuracy"] >= 0.65,
        "beats_no_latent_baseline_by_0.05": dominance >= 0.05,
        "completion_labels_not_degenerate": not completion_degenerate,
    }
    criteria = {k: (bool(v) if v == v else False) for k, v in criteria.items()}
    qualified = all(criteria.values())

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    package_path = PACKAGE_DIR / "scorer_package.pt"
    torch.save({"status": STATUS, "contract_digest": contract_digest(),
                "latent": packages["latent"], "no_latent": packages["no_latent"],
                "architecture": {"tokens": TOKENS, "token_dim": TOKEN_DIM,
                                 "horizons": HORIZONS, "hidden": 512,
                                 "action_dim": ACTION_DIM, "goal_dim": GOAL_DIM},
                "spatial_aggregation": "mean over the 768 tokens",
                "goal_binding": "[sin(bearing_body_rad), cos(bearing_body_rad), range_m]",
                "weights": WEIGHTS, "qualified": qualified}, package_path)
    package_digest = hashlib.sha256(package_path.read_bytes()).hexdigest()

    report = {
        "schema": "go2_utility_scorer_v1_2_qualification", "status": STATUS,
        "scorer_contract_v1_2_digest": contract_digest(),
        "state_manifest_digest": rows[0].get("state_manifest_digest"),
        "oracle_v1_2_digest": rows[0].get("oracle_v1_2_digest"),
        "encoder": corpus["index"]["encoder"]["name"],
        "fit_states": len({r["state_id"] for r in fit_rows}),
        "calibration_states": len({r["state_id"] for r in cal_rows}),
        "fit_rows": len(fit_rows), "calibration_rows": len(cal_rows),
        "scene_disjoint": True,
        "label_distributions": {"fit": label_distribution(fit_rows),
                                "calibration": label_distribution(cal_rows)},
        "latent_scorer": results["latent"],
        "no_latent_baseline": results["no_latent"],
        "baseline_dominance_pairwise": dominance,
        "criteria": criteria, "qualified": qualified,
        "scorer_package_sha256": package_digest,
        "wall_time_s": round(time.time() - started, 1),
    }
    (PACKAGE_DIR / "qualification.json").write_text(json.dumps(report, indent=2,
                                                              default=float))
    print(json.dumps(report, indent=2, default=float))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
