#!/usr/bin/env python3
"""Train a small action-conditioned JEPA-style latent encoder on Go2 RGB slices."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_causal_memory_query_probe import _scrub_command_aux  # noqa: E402
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _aux_features,
    _feature_stats,
    _load_image,
    _load_rows,
    _primitive_vocab,
    _resolve_device,
    _seq_key,
    _split_rows_by_sequence,
)

from lewm.models.go2_jepa import (  # noqa: E402
    Go2JepaEncoder,
    Go2JepaPredictor,
    update_ema,
)


@dataclass(frozen=True)
class Go2JepaFrame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor


@dataclass(frozen=True)
class Go2JepaPair:
    source: Go2JepaFrame
    target: Go2JepaFrame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--predictor-hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--std-loss-weight", type=float, default=0.02)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--max-step-delta", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--scrub-command-aux",
        action="store_true",
        help="Zero route-teacher command fields before building predictor aux features.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows_raw = _load_rows(args.datasets)
    validation_rows_raw = (
        _load_rows(args.validation_datasets) if args.validation_datasets else None
    )
    if not train_rows_raw:
        raise SystemExit("no train rows")
    train_rows = _scrub_command_aux(train_rows_raw) if args.scrub_command_aux else train_rows_raw
    validation_rows = (
        _scrub_command_aux(validation_rows_raw)
        if args.scrub_command_aux and validation_rows_raw is not None
        else validation_rows_raw
    )
    if validation_rows is None:
        train_rows, validation_rows = _split_rows_by_sequence(train_rows)
        validation_rows_raw = validation_rows
    if not validation_rows:
        raise SystemExit("no validation rows")

    primitive_vocab = _primitive_vocab(train_rows_raw, validation_rows_raw or [])
    feature_stats = _feature_stats(train_rows, primitive_vocab=primitive_vocab)
    train_sequences = _build_frames(
        train_rows,
        primitive_vocab=primitive_vocab,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
    )
    validation_sequences = _build_frames(
        validation_rows,
        primitive_vocab=primitive_vocab,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
    )
    train_pairs = _build_pairs(train_sequences, max_step_delta=int(args.max_step_delta))
    validation_pairs = _build_pairs(
        validation_sequences,
        max_step_delta=int(args.max_step_delta),
    )
    if not train_pairs:
        raise SystemExit("no train pairs")
    if not validation_pairs:
        raise SystemExit("no validation pairs")

    device = _resolve_device(str(args.device))
    aux_dim = int(train_pairs[0].source.aux.numel())
    encoder = Go2JepaEncoder(latent_dim=int(args.latent_dim)).to(device)
    target_encoder = Go2JepaEncoder(latent_dim=int(args.latent_dim)).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)
    predictor = Go2JepaPredictor(
        latent_dim=int(args.latent_dim),
        aux_dim=aux_dim,
        hidden_dim=int(args.predictor_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=float(args.lr),
        weight_decay=1e-4,
    )

    history = []
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_epoch(
            encoder,
            target_encoder,
            predictor,
            optimizer,
            train_pairs,
            device=device,
            batch_size=int(args.batch_size),
            ema_decay=float(args.ema_decay),
            std_loss_weight=float(args.std_loss_weight),
            contrastive_loss_weight=float(args.contrastive_loss_weight),
            contrastive_temperature=float(args.contrastive_temperature),
        )
        train_metrics = _evaluate(
            encoder,
            target_encoder,
            predictor,
            train_pairs,
            device=device,
            batch_size=int(args.batch_size),
        )
        validation_metrics = _evaluate(
            encoder,
            target_encoder,
            predictor,
            validation_pairs,
            device=device,
            batch_size=int(args.batch_size),
        )
        score = float(validation_metrics["retrieval_at_1"])
        score += 0.1 * float(validation_metrics["positive_minus_best_negative_cosine"])
        score += 0.01 * min(1.0, float(validation_metrics["target_latent_std_mean"]))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if score >= best_score:
            best_score = score
            best_metrics = validation_metrics
            best_state = {
                "encoder": {
                    key: value.detach().cpu().clone()
                    for key, value in encoder.state_dict().items()
                },
                "target_encoder": {
                    key: value.detach().cpu().clone()
                    for key, value in target_encoder.state_dict().items()
                },
                "predictor": {
                    key: value.detach().cpu().clone()
                    for key, value in predictor.state_dict().items()
                },
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={train_loss:.4f}"
                f" val_retrieval_at1={validation_metrics['retrieval_at_1']:.3f}"
                f" val_pos_cos={validation_metrics['positive_cosine_mean']:.3f}"
                f" val_best_neg_cos={validation_metrics['best_negative_cosine_mean']:.3f}"
                f" val_latent_std={validation_metrics['target_latent_std_mean']:.3f}"
            )

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        target_encoder.load_state_dict(best_state["target_encoder"])
        predictor.load_state_dict(best_state["predictor"])
    final_train = _evaluate(
        encoder,
        target_encoder,
        predictor,
        train_pairs,
        device=device,
        batch_size=int(args.batch_size),
    )
    final_validation = _evaluate(
        encoder,
        target_encoder,
        predictor,
        validation_pairs,
        device=device,
        batch_size=int(args.batch_size),
    )

    checkpoint = {
        "schema": "lewm_go2_jepa_latent_encoder_checkpoint_v0",
        "encoder_state_dict": encoder.state_dict(),
        "target_encoder_state_dict": target_encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "primitive_vocab": primitive_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "latent_dim": int(args.latent_dim),
        "aux_dim": int(aux_dim),
        "predictor_hidden_dim": int(args.predictor_hidden_dim),
        "contrastive_loss_weight": float(args.contrastive_loss_weight),
        "contrastive_temperature": float(args.contrastive_temperature),
        "scrubbed_command_aux": bool(args.scrub_command_aux),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_jepa_latent_encoder_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": (
            [str(path) for path in args.validation_datasets]
            if args.validation_datasets
            else []
        ),
        "split_note": (
            "explicit validation datasets"
            if args.validation_datasets
            else "sequence holdout inside the provided dataset"
        ),
        "output": str(args.output),
        "device": str(device),
        "image_size": int(args.image_size),
        "latent_dim": int(args.latent_dim),
        "aux_dim": int(aux_dim),
        "contrastive_loss_weight": float(args.contrastive_loss_weight),
        "contrastive_temperature": float(args.contrastive_temperature),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "final_train": final_train,
        "final_validation": final_validation,
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "This is a compact action-conditioned JEPA-style Go2 RGB encoder "
            "trained by latent prediction on rendered event slices. It is a "
            "substrate checkpoint for frozen-latent memory probes, not a "
            "closed-loop Go2 navigation result."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_jepa_latent_encoder:"
        f" output={args.output}"
        f" report={report_path}"
        f" val_retrieval_at1={final_validation['retrieval_at_1']:.3f}"
        f" chance={final_validation['retrieval_at_1_chance']:.3f}"
        f" pos_minus_best_neg={final_validation['positive_minus_best_negative_cosine']:.3f}"
        f" latent_std={final_validation['target_latent_std_mean']:.3f}"
    )
    return 0


def _build_frames(
    rows: list[dict[str, Any]],
    *,
    primitive_vocab: list[str],
    feature_stats: dict[str, np.ndarray],
    image_size: int,
) -> dict[tuple[str, int, int], list[Go2JepaFrame]]:
    sequences: dict[tuple[str, int, int], list[Go2JepaFrame]] = {}
    for row in rows:
        aux = _aux_features(row, primitive_vocab=primitive_vocab)
        aux = (aux - feature_stats["mean"]) / feature_stats["std"]
        frame = Go2JepaFrame(
            seq_key=_seq_key(row),
            episode_step=int(row.get("episode_step", 0)),
            image=_load_image(Path(row["rgb_path"]), image_size=image_size),
            aux=torch.tensor(aux, dtype=torch.float32),
        )
        sequences.setdefault(frame.seq_key, []).append(frame)
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item.episode_step)
    return sequences


def _build_pairs(
    sequences: dict[tuple[str, int, int], list[Go2JepaFrame]],
    *,
    max_step_delta: int,
) -> list[Go2JepaPair]:
    pairs = []
    for sequence in sequences.values():
        for source, target in zip(sequence, sequence[1:]):
            if int(target.episode_step) <= int(source.episode_step):
                continue
            if int(target.episode_step) - int(source.episode_step) > int(max_step_delta):
                continue
            pairs.append(Go2JepaPair(source=source, target=target))
    return pairs


def _train_epoch(
    encoder: Go2JepaEncoder,
    target_encoder: Go2JepaEncoder,
    predictor: Go2JepaPredictor,
    optimizer: torch.optim.Optimizer,
    pairs: list[Go2JepaPair],
    *,
    device: torch.device,
    batch_size: int,
    ema_decay: float,
    std_loss_weight: float,
    contrastive_loss_weight: float,
    contrastive_temperature: float,
) -> float:
    encoder.train()
    predictor.train()
    target_encoder.eval()
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    total_loss = 0.0
    batches = 0
    for start in range(0, len(indices), max(1, int(batch_size))):
        batch_indices = indices[start : start + max(1, int(batch_size))]
        source_images, target_images, aux = _pair_batch(pairs, batch_indices, device=device)
        source_latent = encoder(source_images)
        with torch.no_grad():
            target_latent = target_encoder(target_images)
        predicted_latent = predictor(source_latent, aux)
        predict_loss = 1.0 - F.cosine_similarity(
            predicted_latent,
            target_latent.detach(),
            dim=-1,
        ).mean()
        smooth_loss = F.smooth_l1_loss(predicted_latent, target_latent.detach())
        std_loss = _std_loss(source_latent) + _std_loss(predicted_latent)
        contrastive_loss = _contrastive_loss(
            predicted_latent,
            target_latent.detach(),
            temperature=float(contrastive_temperature),
        )
        loss = (
            predict_loss
            + 0.25 * smooth_loss
            + float(contrastive_loss_weight) * contrastive_loss
            + float(std_loss_weight) * std_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            5.0,
        )
        optimizer.step()
        update_ema(target_encoder, encoder, decay=float(ema_decay))
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(1, batches)


def _evaluate(
    encoder: Go2JepaEncoder,
    target_encoder: Go2JepaEncoder,
    predictor: Go2JepaPredictor,
    pairs: list[Go2JepaPair],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    predicted = []
    targets = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(pairs), max(1, int(batch_size))):
            batch_indices = list(range(start, min(len(pairs), start + max(1, int(batch_size)))))
            source_images, target_images, aux = _pair_batch(pairs, batch_indices, device=device)
            source_latent = encoder(source_images)
            target_latent = target_encoder(target_images)
            predicted_latent = predictor(source_latent, aux)
            losses.append(
                float(
                    (
                        1.0
                        - F.cosine_similarity(predicted_latent, target_latent, dim=-1).mean()
                    ).cpu()
                )
            )
            predicted.append(predicted_latent.detach().cpu())
            targets.append(target_latent.detach().cpu())
    predicted_t = torch.cat(predicted, dim=0)
    target_t = torch.cat(targets, dim=0)
    predicted_n = F.normalize(predicted_t, dim=-1)
    target_n = F.normalize(target_t, dim=-1)
    similarity = predicted_n @ target_n.T
    top1 = similarity.argmax(dim=-1)
    labels = torch.arange(similarity.shape[0])
    retrieval_at_1 = float((top1 == labels).float().mean().item())
    positive = similarity.diag()
    if similarity.shape[0] > 1:
        negative_masked = similarity.clone()
        negative_masked[labels, labels] = -2.0
        best_negative = negative_masked.max(dim=-1).values
        negative_mean = float(best_negative.mean().item())
        margin = float((positive - best_negative).mean().item())
    else:
        negative_mean = 0.0
        margin = 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "retrieval_at_1": retrieval_at_1,
        "retrieval_at_1_chance": 1.0 / max(1, int(similarity.shape[0])),
        "positive_cosine_mean": float(positive.mean().item()),
        "best_negative_cosine_mean": negative_mean,
        "positive_minus_best_negative_cosine": margin,
        "target_latent_std_mean": float(target_t.std(dim=0).mean().item()),
        "predicted_latent_std_mean": float(predicted_t.std(dim=0).mean().item()),
        "pair_count": float(len(pairs)),
    }


def _pair_batch(
    pairs: list[Go2JepaPair],
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([pairs[index].source.image for index in indices]).to(device),
        torch.stack([pairs[index].target.image for index in indices]).to(device),
        torch.stack([pairs[index].source.aux for index in indices]).to(device),
    )


def _std_loss(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[0] <= 1:
        return latent.sum() * 0.0
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


def _contrastive_loss(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if predicted_latent.shape[0] <= 1:
        return predicted_latent.sum() * 0.0
    predicted_n = F.normalize(predicted_latent, dim=-1)
    target_n = F.normalize(target_latent, dim=-1)
    logits = predicted_n @ target_n.T
    logits = logits / max(1e-4, float(temperature))
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    raise SystemExit(main())
