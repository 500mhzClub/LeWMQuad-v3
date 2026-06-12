#!/usr/bin/env python3
"""Frozen-latent closure probe for inverse-dynamics (IDM) usefulness.

This diagnostic asks a narrow question before any IDM fine-tuning: does the true
consecutive latent pair contain transition-specific information about the logged
command beyond what is predictable from the current state or an unrelated next
state?

It fits deterministic ridge readouts on held-out scenes for:
  - state:         z_t
  - true_pair:     [z_t, z_{t+1}]
  - shuffled_next: [z_t, shuffled(z_{t+1})]
  - delta:         z_{t+1} - z_t

A true-pair gain is evidence that IDM has a real signal to exploit. It is not
evidence that IDM improves goal-conditioned first-action ranking.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from train_lewm import GenesisWMDataset, make_loader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (x - mean) / scale, mean, scale


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x_train, mean, scale = _standardize_fit(x_train)
    x_eval = (x_eval.astype(np.float64, copy=False) - mean) / scale
    y_train = y_train.astype(np.float64, copy=False)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_centered = y_train - y_mean
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram, x_train.T @ y_centered)
    return x_eval @ weights + y_mean


def _pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.square(y_true - y_pred).sum()
    centered = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum()
    return float(1.0 - residual / centered) if centered > 1e-8 else float("nan")


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alphas: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Select ridge regularization without consulting the held-out scene split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    split = max(1, min(len(order) - 1, int(0.8 * len(order))))
    fit_indices, validation_indices = order[:split], order[split:]

    selected_alpha = alphas[0]
    selected_score = float("-inf")
    for alpha in alphas:
        prediction = _ridge_predict(
            x_train[fit_indices],
            y_train[fit_indices],
            x_train[validation_indices],
            alpha=alpha,
        )
        score = _pooled_r2(y_train[validation_indices], prediction)
        if score > selected_score:
            selected_alpha = alpha
            selected_score = score

    prediction = _ridge_predict(x_train, y_train, x_eval, alpha=selected_alpha)
    return prediction, selected_alpha, selected_score


def _finite_mean(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _r2_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = np.square(y_true - y_pred)
    centered = np.square(y_true - y_true.mean(axis=0, keepdims=True))
    sse = residual.sum(axis=0)
    sst = centered.sum(axis=0)
    valid = sst > 1e-8
    per_channel = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    per_channel[valid] = 1.0 - sse[valid] / sst[valid]
    pooled_sst = float(sst.sum())
    pooled = float(1.0 - residual.sum() / pooled_sst) if pooled_sst > 1e-8 else float("nan")
    report = {
        "pooled_r2": pooled,
        "mean_channel_r2": _finite_mean(per_channel),
        "per_channel_r2": [None if np.isnan(v) else float(v) for v in per_channel],
        "mae": float(np.abs(y_true - y_pred).mean()),
    }
    if y_true.shape[1] % 3 == 0:
        block = y_true.shape[1] // 3
        for index, name in enumerate(("vx", "vy", "wz")):
            values = per_channel[index * block : (index + 1) * block]
            report[f"{name}_mean_r2"] = _finite_mean(values)
            report[f"{name}_mae"] = float(
                np.abs(
                    y_true[:, index * block : (index + 1) * block]
                    - y_pred[:, index * block : (index + 1) * block]
                ).mean()
            )
    return report


def _features(z: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    current = z[:, 0]
    nxt = z[:, 1]
    rng = np.random.default_rng(seed)
    shuffled = nxt[rng.permutation(len(nxt))]
    return {
        "state": current,
        "true_pair": np.concatenate((current, nxt), axis=1),
        "shuffled_next": np.concatenate((current, shuffled), axis=1),
        "delta": nxt - current,
    }


@torch.no_grad()
def _collect(
    model,
    loader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, np.ndarray]:
    raw_parts: list[np.ndarray] = []
    proj_parts: list[np.ndarray] = []
    cmd_parts: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        vis = batch["vis_seq"].to(device)
        cmd = batch["cmd_seq"].to(device)
        z_raw, z_proj = model.encode_seq(vis, prop_seq=None)
        transitions = z_raw.shape[1] - 1
        raw_parts.append(
            torch.stack((z_raw[:, :-1], z_raw[:, 1:]), dim=2)
            .reshape(-1, 2, z_raw.shape[-1])
            .float()
            .cpu()
            .numpy()
        )
        proj_parts.append(
            torch.stack((z_proj[:, :-1], z_proj[:, 1:]), dim=2)
            .reshape(-1, 2, z_proj.shape[-1])
            .float()
            .cpu()
            .numpy()
        )
        cmd_parts.append(cmd[:, :transitions].reshape(-1, cmd.shape[-1]).float().cpu().numpy())
    if not raw_parts:
        raise RuntimeError("no batches available for IDM closure probe")
    return {
        "raw": np.concatenate(raw_parts, axis=0),
        "proj": np.concatenate(proj_parts, axis=0),
        "cmd": np.concatenate(cmd_parts, axis=0),
    }


def _make_dataset(args, config: dict, *, role: str) -> GenesisWMDataset:
    render_root = args.render_root or (
        Path(config["render_root"]) if "render_root" in config else None
    )
    return GenesisWMDataset(
        root_dir=args.data_root,
        render_root=render_root,
        seq_len=args.max_seq_len or int(config.get("max_seq_len", 4)),
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=(
            args.allow_material_color_render
            or bool(config.get("allow_material_color_render", False))
        ),
        holdout_fraction=args.holdout_fraction,
        holdout_role=role,
        holdout_seed=args.holdout_seed,
    )


def _make_probe_loader(dataset: GenesisWMDataset, args, device: torch.device):
    return make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-seed", type=int, default=20260524)
    parser.add_argument("--allow-material-color-render", action="store_true")
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(10.0, 100.0, 1000.0, 10000.0),
        help="Candidate ridge strengths selected on a training-only validation split.",
    )
    parser.add_argument("--min-pair-gain", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    ridge_alphas = tuple(sorted(set(args.ridge_alphas)))
    if not ridge_alphas or ridge_alphas[0] <= 0:
        raise SystemExit("--ridge-alphas must contain positive values")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Running IDM closure probe on %s", device)

    model, config = load_model(args, device)
    train_dataset = _make_dataset(args, config, role="train")
    eval_dataset = _make_dataset(args, config, role="eval")
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise SystemExit(
            f"empty split: train={len(train_dataset)} eval={len(eval_dataset)}; "
            "increase --max-sessions or --holdout-fraction"
        )
    train = _collect(
        model,
        _make_probe_loader(train_dataset, args, device),
        device,
        max_batches=args.max_train_batches,
    )
    evaluation = _collect(
        model,
        _make_probe_loader(eval_dataset, args, device),
        device,
        max_batches=args.max_eval_batches,
    )

    spaces = {}
    for space_index, space in enumerate(("raw", "proj")):
        train_features = _features(train[space], seed=args.seed + space_index)
        eval_features = _features(evaluation[space], seed=args.seed + 100 + space_index)
        probes = {}
        for probe_index, name in enumerate(("state", "true_pair", "shuffled_next", "delta")):
            prediction, selected_alpha, selection_r2 = _fit_probe(
                train_features[name],
                train["cmd"],
                eval_features[name],
                alphas=ridge_alphas,
                seed=args.seed + 1000 * space_index + probe_index,
            )
            probes[name] = _r2_report(evaluation["cmd"], prediction)
            probes[name]["selected_ridge_alpha"] = selected_alpha
            probes[name]["train_selection_r2"] = selection_r2
        pair_r2 = probes["true_pair"]["pooled_r2"]
        state_gain = pair_r2 - probes["state"]["pooled_r2"]
        shuffled_gain = pair_r2 - probes["shuffled_next"]["pooled_r2"]
        spaces[space] = {
            "probes": probes,
            "true_pair_gain_over_state": state_gain,
            "true_pair_gain_over_shuffled_next": shuffled_gain,
            "transition_specific_signal": bool(
                state_gain >= args.min_pair_gain and shuffled_gain >= args.min_pair_gain
            ),
        }

    report = {
        "schema": "idm_decodability_closure_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "train_transition_samples": int(len(train["cmd"])),
        "eval_transition_samples": int(len(evaluation["cmd"])),
        "holdout_fraction": args.holdout_fraction,
        "holdout_seed": args.holdout_seed,
        "ridge_alphas": ridge_alphas,
        "min_pair_gain": args.min_pair_gain,
        "spaces": spaces,
        "decision": {
            "any_transition_specific_signal": any(
                result["transition_specific_signal"] for result in spaces.values()
            ),
            "interpretation": (
                "A positive result only shows that IDM has transition-specific signal. "
                "It does not justify fine-tuning without improvement on a goal-aligned "
                "first-action metric."
            ),
        },
    }
    text = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
