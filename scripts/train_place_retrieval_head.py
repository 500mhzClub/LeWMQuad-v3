#!/usr/bin/env python3
"""Train and gate a place-retrieval head on frozen raw LeWM features."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.place_retrieval import (  # noqa: E402
    PlaceRetrievalHead,
    masked_supervised_contrastive_loss,
)
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_reachability_a3 import (  # noqa: E402
    _select,
    build_scene_bank,
    same_cell_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_place_retrieval_head")


def _aggregate(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _pair_masks(bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
    cells = bank["cells"]
    graph = bank["graph"]
    positive = cells[:, None] == cells[None, :]
    np.fill_diagonal(positive, False)
    valid = positive.copy()
    unique_cells = sorted({int(cell) for cell in cells})
    valid_cell_pairs: set[tuple[int, int]] = set()
    for offset, cell_a in enumerate(unique_cells):
        for cell_b in unique_cells[offset + 1 :]:
            distance = graph.bfs_distance(cell_a, cell_b)
            if distance is not None and distance >= 2:
                valid_cell_pairs.add((cell_a, cell_b))
                valid_cell_pairs.add((cell_b, cell_a))
    for i, cell_a in enumerate(cells):
        for j, cell_b in enumerate(cells):
            if (int(cell_a), int(cell_b)) in valid_cell_pairs:
                valid[i, j] = True
    return torch.from_numpy(positive), torch.from_numpy(valid)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _graph_distance_spearman(embedding: np.ndarray, bank: dict) -> float | None:
    cells = bank["cells"]
    graph = bank["graph"]
    cosine_distance = 1.0 - embedding @ embedding.T
    latent_distances: list[float] = []
    graph_distances: list[float] = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if cells[i] == cells[j]:
                continue
            distance = graph.bfs_distance(int(cells[i]), int(cells[j]))
            if distance is not None:
                latent_distances.append(float(cosine_distance[i, j]))
                graph_distances.append(float(distance))
    if len(latent_distances) < 3:
        return None
    latent_rank = _rank(np.asarray(latent_distances))
    graph_rank = _rank(np.asarray(graph_distances))
    return float(np.corrcoef(latent_rank, graph_rank)[0, 1])


@torch.no_grad()
def _evaluate(
    head: PlaceRetrievalHead | None,
    banks: list[dict],
    device: torch.device,
) -> dict:
    retrieval_at_1: list[float] = []
    retrieval_at_5: list[float] = []
    graph_spearman: list[float] = []
    if head is not None:
        head.eval()
    for bank in banks:
        if head is None:
            embedding = bank["z_raw"]
            normalized = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-9)
        else:
            latent = torch.from_numpy(bank["z_raw"]).float().to(device)
            embedding = head(latent).cpu().numpy().astype(np.float64)
            normalized = embedding
        retrieval = same_cell_retrieval(embedding, bank["cells"])
        if retrieval is None:
            continue
        retrieval_at_1.append(retrieval["retrieval_at_1"])
        retrieval_at_5.append(retrieval["retrieval_at_5"])
        correlation = _graph_distance_spearman(normalized, bank)
        if correlation is not None and not np.isnan(correlation):
            graph_spearman.append(correlation)
    return {
        "retrieval_at_1": _aggregate(retrieval_at_1),
        "retrieval_at_5": _aggregate(retrieval_at_5),
        "graph_distance_spearman": _aggregate(graph_spearman),
    }


def _train_seed(
    train_banks: list[dict],
    eval_banks: list[dict],
    *,
    seed: int,
    epochs: int,
    hidden: int,
    embedding_dim: int,
    dropout: float,
    temperature: float,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[PlaceRetrievalHead, dict]:
    torch.manual_seed(seed)
    latent_dim = int(train_banks[0]["z_raw"].shape[1])
    head = PlaceRetrievalHead(
        latent_dim=latent_dim,
        hidden=hidden,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    prepared = [
        (
            torch.from_numpy(bank["z_raw"]).float(),
            *_pair_masks(bank),
        )
        for bank in train_banks
    ]

    final_loss = 0.0
    for epoch in range(epochs):
        head.train()
        order = list(range(len(prepared)))
        random.Random(seed + epoch).shuffle(order)
        total_loss = 0.0
        for index in order:
            latent, positive, valid = prepared[index]
            embedding = head(latent.to(device))
            loss = masked_supervised_contrastive_loss(
                embedding,
                positive.to(device),
                valid.to(device),
                temperature=temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        final_loss = total_loss / len(order)
        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info("seed=%d epoch=%d train_loss=%.4f", seed, epoch, final_loss)

    metrics = _evaluate(head, eval_banks, device)
    metrics["seed"] = seed
    metrics["final_train_loss"] = final_loss
    return head, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    parser.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    parser.add_argument(
        "--manifest-corpus",
        type=Path,
        default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--train-scenes-per-family", type=int, default=4)
    parser.add_argument("--eval-scenes-per-family", type=int, default=4)
    parser.add_argument("--train-frames-per-scene", type=int, default=100)
    parser.add_argument("--eval-frames-per-scene", type=int, default=240)
    parser.add_argument("--train-max-per-cell", type=int, default=3)
    parser.add_argument("--eval-max-per-cell", type=int, default=8)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", default="20260606,20260607,20260608")
    parser.add_argument("--gate-recall5-improvement", type=float, default=0.15)
    parser.add_argument("--min-eval-scenes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    model, _config = load_model(args, device)

    def build_banks(split: str, per_family: int, frames: int, max_per_cell: int) -> list[dict]:
        banks = []
        for index, (family, label_file) in enumerate(
            _select(args.rollout_root, split, per_family, seeds[0])
        ):
            bank = build_scene_bank(
                model,
                label_file=label_file,
                family=family,
                split=split,
                render_root=args.render_root,
                corpus_root=args.manifest_corpus,
                device=device,
                frames_per_scene=frames,
                max_per_cell=max_per_cell,
                batch_size=args.batch_size,
                min_cells=args.min_cells,
                rng=random.Random(seeds[0] + index * 7919),
            )
            if bank is not None:
                banks.append(bank)
                logger.info(
                    "bank %s %s frames=%d cells=%d",
                    split,
                    bank["scene_id"],
                    len(bank["cells"]),
                    len(set(bank["cells"].tolist())),
                )
        return banks

    logger.info("Building frozen train banks on %s", device)
    train_banks = build_banks(
        args.train_split,
        args.train_scenes_per_family,
        args.train_frames_per_scene,
        args.train_max_per_cell,
    )
    logger.info("Building frozen evaluation banks")
    eval_banks = build_banks(
        args.eval_split,
        args.eval_scenes_per_family,
        args.eval_frames_per_scene,
        args.eval_max_per_cell,
    )
    if not train_banks or not eval_banks:
        raise SystemExit("no usable train or evaluation scene banks")

    baseline = _evaluate(None, eval_banks, device)
    seed_reports = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    head_dir = args.output.parent / f"{args.output.stem}_heads"
    head_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        head, metrics = _train_seed(
            train_banks,
            eval_banks,
            seed=seed,
            epochs=args.epochs,
            hidden=args.hidden,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            temperature=args.temperature,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        seed_reports.append(metrics)
        torch.save(
            {
                "head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                "latent_dim": head.latent_dim,
                "embedding_dim": head.embedding_dim,
                "source_checkpoint": str(args.checkpoint),
                "seed": seed,
                "metrics": metrics,
            },
            head_dir / f"place_retrieval_seed{seed}.pt",
        )

    baseline_r1 = baseline["retrieval_at_1"]["mean"]
    baseline_r5 = baseline["retrieval_at_5"]["mean"]
    seed_r1 = [report["retrieval_at_1"]["mean"] for report in seed_reports]
    seed_r5 = [report["retrieval_at_5"]["mean"] for report in seed_reports]
    learned_summary = {
        "retrieval_at_1_across_seeds": _aggregate(seed_r1),
        "retrieval_at_5_across_seeds": _aggregate(seed_r5),
        "mean_recall1_improvement": float(np.mean(seed_r1) - baseline_r1),
        "mean_recall5_improvement": float(np.mean(seed_r5) - baseline_r5),
    }
    gate_passed = bool(
        len(eval_banks) >= args.min_eval_scenes
        and learned_summary["mean_recall5_improvement"] >= args.gate_recall5_improvement
        and all(value > baseline_r5 for value in seed_r5)
        and np.mean(seed_r1) >= baseline_r1
    )
    report = {
        "schema": "lewm_minimal_phase_b_retrieval_gate_v0",
        "source_checkpoint": str(args.checkpoint),
        "train_scene_count": len(train_banks),
        "eval_scene_count": len(eval_banks),
        "baseline_raw": baseline,
        "seed_reports": seed_reports,
        "learned_summary": learned_summary,
        "gate": {
            "passed": gate_passed,
            "required_recall5_improvement": args.gate_recall5_improvement,
            "min_eval_scenes": args.min_eval_scenes,
            "requires_every_seed_recall5_improvement": True,
            "requires_mean_recall1_non_regression": True,
        },
        "config": {key: str(value) for key, value in vars(args).items()},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "config"}, indent=2))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

