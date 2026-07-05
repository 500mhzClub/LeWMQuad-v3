#!/usr/bin/env python3
"""Train an offline memory-conditioned Go2 primitive policy.

This is the next bridge after target selection: given the recurrent state from
the learned memory probe and a queried remembered target, predict the
route-teacher primitive for strict hidden-return frames.

The default path scrubs the current command fields before recomputing memory
states. That avoids the trivial leak where the auxiliary vector already
contains the primitive we are trying to predict.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_go2_causal_memory_target_gate import _select_object  # noqa: E402
from train_go2_causal_memory_query_probe import (  # noqa: E402
    QueryConditionedGo2MemoryProbe,
    _build_frames,
    _current_role,
    _max_landmark_slot,
    _sequence_tensors,
)
from train_go2_hidden_target_memory_probe import _load_rows, _resolve_device  # noqa: E402


@dataclass(frozen=True)
class CommandExample:
    seq_key: tuple[str, int, int]
    episode_step: int
    object_id: str
    command: str
    features: torch.Tensor


class CommandHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, command_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, command_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--memory-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--allow-command-aux-leak",
        action="store_true",
        help="Diagnostic only: keep current command fields in memory-probe aux features.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    checkpoint = _load_memory_checkpoint(args.memory_checkpoint)
    threshold = _checkpoint_threshold(checkpoint) if args.threshold is None else float(args.threshold)
    train_rows_raw = _load_rows(args.datasets)
    validation_rows_raw = _load_rows(args.validation_datasets)
    if not train_rows_raw:
        raise SystemExit("no train rows")
    if not validation_rows_raw:
        raise SystemExit("no validation rows")

    train_rows_for_memory = (
        train_rows_raw if args.allow_command_aux_leak else _scrub_command_aux(train_rows_raw)
    )
    validation_rows_for_memory = (
        validation_rows_raw
        if args.allow_command_aux_leak
        else _scrub_command_aux(validation_rows_raw)
    )
    command_vocab = _command_vocab(train_rows_raw, validation_rows_raw)
    feature_stats = {
        "mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
    }
    max_slot = _recover_max_slot(checkpoint, fallback_rows=train_rows_raw + validation_rows_raw)
    checkpoint_args = dict(checkpoint.get("args", {}))

    train_sequences = _build_frames(
        train_rows_for_memory,
        primitive_vocab=list(checkpoint["primitive_vocab"]),
        color_vocab=list(checkpoint["color_vocab"]),
        max_slot=int(max_slot),
        feature_stats=feature_stats,
        image_size=int(checkpoint["image_size"]),
        include_object_slot=bool(checkpoint_args.get("include_object_slot", False)),
        include_privileged_landmark_geometry=bool(
            checkpoint_args.get("include_privileged_landmark_geometry", False)
        ),
    )
    validation_sequences = _build_frames(
        validation_rows_for_memory,
        primitive_vocab=list(checkpoint["primitive_vocab"]),
        color_vocab=list(checkpoint["color_vocab"]),
        max_slot=int(max_slot),
        feature_stats=feature_stats,
        image_size=int(checkpoint["image_size"]),
        include_object_slot=bool(checkpoint_args.get("include_object_slot", False)),
        include_privileged_landmark_geometry=bool(
            checkpoint_args.get("include_privileged_landmark_geometry", False)
        ),
    )
    train_row_index = _row_index(train_rows_raw)
    validation_row_index = _row_index(validation_rows_raw)
    if not train_sequences or not validation_sequences:
        raise SystemExit("no evaluable sequences")

    device = _resolve_device(str(args.device))
    memory_model = QueryConditionedGo2MemoryProbe(
        aux_dim=int(checkpoint["aux_dim"]),
        query_dim=int(checkpoint["query_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    memory_model.load_state_dict(checkpoint["model_state_dict"])
    memory_model.eval()
    for param in memory_model.parameters():
        param.requires_grad_(False)

    train_examples = _oracle_command_examples(
        memory_model,
        train_sequences,
        train_row_index,
        command_vocab=command_vocab,
        device=device,
        ablation="normal",
    )
    validation_examples = _oracle_command_examples(
        memory_model,
        validation_sequences,
        validation_row_index,
        command_vocab=command_vocab,
        device=device,
        ablation="normal",
    )
    if not train_examples:
        raise SystemExit("no train command examples")
    if not validation_examples:
        raise SystemExit("no validation command examples")

    input_dim = int(train_examples[0].features.numel())
    command_head = CommandHead(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        command_count=len(command_vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(command_head.parameters(), lr=float(args.lr), weight_decay=1e-4)
    class_weights = _class_weights(train_examples, command_vocab=command_vocab).to(device)

    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_validation: dict[str, Any] | None = None
    train_x, train_y = _example_tensors(train_examples, command_vocab=command_vocab, device=device)
    val_x, val_y = _example_tensors(validation_examples, command_vocab=command_vocab, device=device)
    for epoch in range(1, int(args.epochs) + 1):
        loss = _train_epoch(
            command_head,
            optimizer,
            train_x,
            train_y,
            class_weights=class_weights,
        )
        train_metrics = _classify(command_head, train_x, train_y, command_vocab=command_vocab)
        validation_metrics = _classify(command_head, val_x, val_y, command_vocab=command_vocab)
        score = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss),
                "train": train_metrics["overall"],
                "validation": validation_metrics["overall"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if score >= best_score:
            best_score = score
            best_validation = validation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in command_head.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={loss:.4f}"
                f" val_acc={validation_metrics['overall']['accuracy']:.3f}"
                f" val_macro_f1={validation_metrics['macro_f1']:.3f}"
            )

    if best_state is not None:
        command_head.load_state_dict(best_state)

    final_train = _classify(command_head, train_x, train_y, command_vocab=command_vocab)
    final_validation = _classify(command_head, val_x, val_y, command_vocab=command_vocab)
    ablation_reports = {
        ablation: _oracle_ablation_report(
            command_head,
            memory_model,
            validation_sequences,
            validation_row_index,
            command_vocab=command_vocab,
            device=device,
            ablation=ablation,
        )
        for ablation in (
            "normal",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }
    pipeline_reports = {
        ablation: _pipeline_report(
            command_head,
            memory_model,
            validation_sequences,
            validation_row_index,
            command_vocab=command_vocab,
            device=device,
            threshold=threshold,
            ablation=ablation,
        )
        for ablation in (
            "normal",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }
    majority = _majority_baseline(
        train_examples,
        validation_examples,
        command_vocab=command_vocab,
    )
    normal_acc = float(ablation_reports["normal"]["overall"]["accuracy"])
    corrupted_best = max(
        float(ablation_reports[name]["overall"]["accuracy"])
        for name in ("reset_recurrent_state", "reverse_input_history", "shuffle_hidden_states")
    )

    checkpoint_out = {
        "schema": "lewm_go2_memory_command_policy_checkpoint_v0",
        "command_head_state_dict": command_head.state_dict(),
        "memory_checkpoint": str(args.memory_checkpoint),
        "command_vocab": command_vocab,
        "input_dim": input_dim,
        "hidden_dim": int(args.hidden_dim),
        "threshold": float(threshold),
        "scrubbed_command_aux": not bool(args.allow_command_aux_leak),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_out, args.output)

    report = {
        "schema": "lewm_go2_memory_command_policy_report_v0",
        "memory_checkpoint": str(args.memory_checkpoint),
        "output": str(args.output),
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "device": str(device),
        "threshold": float(threshold),
        "scrubbed_command_aux": not bool(args.allow_command_aux_leak),
        "command_vocab": command_vocab,
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
        "train_command_counts": _command_counts(train_examples),
        "validation_command_counts": _command_counts(validation_examples),
        "majority_baseline": majority,
        "final_train": final_train,
        "final_validation": final_validation,
        "best_validation_selected_metrics": best_validation or {},
        "validation_oracle_target_ablations": ablation_reports,
        "validation_learned_gate_pipeline_ablations": pipeline_reports,
        "normal_minus_best_corrupted_oracle_target_accuracy": normal_acc - corrupted_best,
        "history": history,
        "claim_boundary": (
            "Offline primitive-imitation bridge over strict hidden-return rows. "
            "It uses a frozen learned memory probe and, by default, scrubbed "
            "current-command aux features. Passing it supports Go2 "
            "translatability; it is not closed-loop robot navigation."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_memory_command_policy:"
        f" output={args.output}"
        f" report={report_path}"
        f" val_acc={final_validation['overall']['accuracy']:.3f}"
        f" val_macro_f1={final_validation['macro_f1']:.3f}"
        f" majority_acc={majority['accuracy']:.3f}"
        f" delta={report['normal_minus_best_corrupted_oracle_target_accuracy']:.3f}"
    )
    return 0


def _load_memory_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "lewm_go2_causal_memory_query_probe_checkpoint_v0":
        raise SystemExit(f"unsupported memory checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


def _checkpoint_threshold(checkpoint: dict[str, Any]) -> float:
    return float(dict(checkpoint.get("args", {})).get("threshold", 0.5))


def _recover_max_slot(checkpoint: dict[str, Any], *, fallback_rows: list[dict[str, Any]]) -> int:
    rows = []
    for raw_path in _checkpoint_dataset_paths(checkpoint):
        path = Path(str(raw_path))
        if path.is_file() or path.is_dir():
            rows.extend(_load_rows([path]))
    return _max_landmark_slot(rows or fallback_rows)


def _checkpoint_dataset_paths(checkpoint: dict[str, Any]) -> list[Path]:
    args = dict(checkpoint.get("args", {}))
    paths: list[Path] = []
    for key in ("datasets", "validation_datasets"):
        value = args.get(key, [])
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            value = [value]
        paths.extend(Path(str(item)) for item in value)
    return paths


def _scrub_command_aux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed = []
    for row in rows:
        item = dict(row)
        command = dict(item.get("command") or {})
        command["primitive_name"] = ""
        command["vx_body_mps"] = []
        command["vy_body_mps"] = []
        command["yaw_rate_radps"] = []
        item["command"] = command
        scrubbed.append(item)
    return scrubbed


def _command_vocab(*row_groups: list[dict[str, Any]]) -> list[str]:
    vocab = {
        str((row.get("command") or {}).get("primitive_name", ""))
        for rows in row_groups
        for row in rows
    }
    vocab.discard("")
    if not vocab:
        raise SystemExit("no command labels found")
    return sorted(vocab)


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, int, int], int], dict[str, Any]]:
    return {
        (
            (
                str(row.get("scene_id", "")),
                int(row.get("env_idx", 0)),
                int(row.get("episode_id", 0)),
            ),
            int(row.get("episode_step", 0)),
        ): row
        for row in rows
    }


def _oracle_command_examples(
    memory_model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    command_vocab: list[str],
    device: torch.device,
    ablation: str,
) -> list[CommandExample]:
    hidden_by_key = _hidden_states_by_sequence(
        memory_model,
        sequences,
        device=device,
        ablation=ablation,
    )
    examples: list[CommandExample] = []
    command_set = set(command_vocab)
    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                row = row_index.get((frame.seq_key, int(frame.episode_step)))
                if row is None:
                    continue
                command = str((row.get("command") or {}).get("primitive_name", ""))
                if command not in command_set:
                    continue
                positives = [
                    query
                    for query in frame.queries
                    if _current_role(query) and float(query.target) >= 0.5
                ]
                for query in positives:
                    examples.append(
                        CommandExample(
                            seq_key=frame.seq_key,
                            episode_step=int(frame.episode_step),
                            object_id=query.object_id,
                            command=command,
                            features=_command_features(
                                memory_model,
                                hidden[step_idx],
                                query.features,
                                device=device,
                            ),
                        )
                    )
    return examples


def _hidden_states_by_sequence(
    memory_model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    *,
    device: torch.device,
    ablation: str,
) -> dict[tuple[str, int, int], torch.Tensor]:
    hidden_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            images, aux = _sequence_tensors(sequence, device=device)
            if ablation == "normal":
                hidden = memory_model.forward_hidden(images, aux)
            elif ablation == "reset_recurrent_state":
                hidden = memory_model.forward_hidden(images, aux, reset_each_step=True)
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                hidden = memory_model.forward_hidden(images[order], aux[order]).flip(0)
            elif ablation == "shuffle_hidden_states":
                hidden = memory_model.forward_hidden(images, aux)
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            hidden_by_key[key] = hidden
    if ablation != "shuffle_hidden_states":
        return hidden_by_key
    flat_hidden = []
    spans: dict[tuple[str, int, int], tuple[int, int]] = {}
    cursor = 0
    for key in sequences:
        hidden = hidden_by_key[key]
        start = cursor
        flat_hidden.append(hidden)
        cursor += int(hidden.shape[0])
        spans[key] = (start, cursor)
    if cursor <= 1:
        return hidden_by_key
    flat = torch.cat(flat_hidden, dim=0)
    shuffled = torch.roll(flat, shifts=max(1, cursor // 2), dims=0)
    return {key: shuffled[start:end] for key, (start, end) in spans.items()}


def _command_features(
    memory_model: QueryConditionedGo2MemoryProbe,
    hidden: torch.Tensor,
    query_features: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    query = query_features.to(device)
    logit = memory_model.score_queries(hidden.unsqueeze(0), query.unsqueeze(0)).squeeze(0)
    prob = torch.sigmoid(logit)
    return torch.cat([hidden.detach(), query.detach(), prob.detach().view(1)], dim=0).cpu()


def _example_tensors(
    examples: list[CommandExample],
    *,
    command_vocab: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    command_index = {command: idx for idx, command in enumerate(command_vocab)}
    return (
        torch.stack([example.features for example in examples]).to(device),
        torch.tensor(
            [command_index[example.command] for example in examples],
            dtype=torch.long,
            device=device,
        ),
    )


def _class_weights(examples: list[CommandExample], *, command_vocab: list[str]) -> torch.Tensor:
    counts = Counter(example.command for example in examples)
    total = float(sum(counts.values()))
    weights = []
    for command in command_vocab:
        weights.append(total / max(1.0, float(len(command_vocab) * counts.get(command, 0))))
    return torch.tensor(weights, dtype=torch.float32)


def _train_epoch(
    model: CommandHead,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor,
) -> float:
    model.train()
    order = torch.randperm(features.shape[0], device=features.device)
    total_loss = 0.0
    batch_size = min(64, int(features.shape[0]))
    for start in range(0, int(features.shape[0]), batch_size):
        batch = order[start : start + batch_size]
        logits = model(features[batch])
        loss = F.cross_entropy(logits, targets[batch], weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(batch.numel())
    return total_loss / max(1, int(features.shape[0]))


def _classify(
    model: CommandHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    command_vocab: list[str],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        predictions = torch.argmax(logits, dim=-1)
    return _classification_report(
        predictions.detach().cpu().numpy().tolist(),
        targets.detach().cpu().numpy().tolist(),
        command_vocab=command_vocab,
    )


def _classification_report(
    predictions: list[int],
    targets: list[int],
    *,
    command_vocab: list[str],
) -> dict[str, Any]:
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    by_command: dict[str, dict[str, float]] = {}
    f1_values = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, target in zip(predictions, targets):
        confusion[command_vocab[target]][command_vocab[pred]] += 1
    for idx, command in enumerate(command_vocab):
        tp = sum(1 for pred, target in zip(predictions, targets) if pred == idx and target == idx)
        fp = sum(1 for pred, target in zip(predictions, targets) if pred == idx and target != idx)
        fn = sum(1 for pred, target in zip(predictions, targets) if pred != idx and target == idx)
        support = sum(1 for target in targets if target == idx)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, support)
        f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
        if support > 0:
            f1_values.append(f1)
        by_command[command] = {
            "support": float(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positive_count": float(tp),
            "false_positive_count": float(fp),
            "false_negative_count": float(fn),
        }
    return {
        "overall": {
            "accuracy": correct / max(1, len(targets)),
            "correct_count": float(correct),
            "example_count": float(len(targets)),
        },
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "by_command": by_command,
        "confusion": {
            actual: dict(sorted(predicted.items()))
            for actual, predicted in sorted(confusion.items())
        },
    }


def _oracle_ablation_report(
    command_head: CommandHead,
    memory_model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    command_vocab: list[str],
    device: torch.device,
    ablation: str,
) -> dict[str, Any]:
    examples = _oracle_command_examples(
        memory_model,
        sequences,
        row_index,
        command_vocab=command_vocab,
        device=device,
        ablation=ablation,
    )
    if not examples:
        return {"overall": {"accuracy": 0.0, "example_count": 0.0}, "macro_f1": 0.0}
    features, targets = _example_tensors(examples, command_vocab=command_vocab, device=device)
    return _classify(command_head, features, targets, command_vocab=command_vocab)


def _pipeline_report(
    command_head: CommandHead,
    memory_model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    command_vocab: list[str],
    device: torch.device,
    threshold: float,
    ablation: str,
) -> dict[str, Any]:
    hidden_by_key = _hidden_states_by_sequence(
        memory_model,
        sequences,
        device=device,
        ablation=ablation,
    )
    command_index = {command: idx for idx, command in enumerate(command_vocab)}
    positive_frames = 0
    negative_frames = 0
    selected_frames = 0
    correct_target = 0
    wrong_target = 0
    false_claim = 0
    missed_positive = 0
    correct_command_after_correct_target = 0
    pipeline_success = 0
    predicted_commands: Counter[str] = Counter()
    actual_commands: Counter[str] = Counter()

    command_head.eval()
    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                current_queries = [query for query in frame.queries if _current_role(query)]
                if not current_queries:
                    continue
                row = row_index.get((frame.seq_key, int(frame.episode_step)))
                if row is None:
                    continue
                command = str((row.get("command") or {}).get("primitive_name", ""))
                if command not in command_index:
                    continue
                query_features = torch.stack([query.features for query in current_queries]).to(device)
                hidden_rows = hidden[step_idx].repeat(len(current_queries), 1)
                probs = torch.sigmoid(memory_model.score_queries(hidden_rows, query_features))
                object_scores: dict[str, float] = {}
                object_colors: dict[str, str] = {}
                object_queries: dict[str, Any] = {}
                object_targets: dict[str, float] = {}
                for query, prob in zip(current_queries, probs.detach().cpu().numpy()):
                    object_scores[query.object_id] = max(
                        object_scores.get(query.object_id, -1.0),
                        float(prob),
                    )
                    object_colors[query.object_id] = _object_color(query.object_id)
                    object_queries[query.object_id] = query
                    object_targets[query.object_id] = max(
                        object_targets.get(query.object_id, 0.0),
                        float(query.target),
                    )
                positive_objects = {
                    object_id
                    for object_id, target in object_targets.items()
                    if float(target) >= 0.5
                }
                if positive_objects:
                    positive_frames += 1
                    actual_commands[command] += 1
                else:
                    negative_frames += 1
                selected = _select_object(
                    object_scores,
                    object_colors=object_colors,
                    default_threshold=threshold,
                    color_thresholds={},
                )
                if selected is None:
                    if positive_objects:
                        missed_positive += 1
                    continue
                selected_frames += 1
                if selected not in positive_objects:
                    if positive_objects:
                        wrong_target += 1
                    else:
                        false_claim += 1
                    continue
                correct_target += 1
                query = object_queries[selected]
                features = _command_features(
                    memory_model,
                    hidden[step_idx],
                    query.features,
                    device=device,
                ).unsqueeze(0).to(device)
                pred_idx = int(torch.argmax(command_head(features), dim=-1).item())
                predicted = command_vocab[pred_idx]
                predicted_commands[predicted] += 1
                if predicted == command:
                    correct_command_after_correct_target += 1
                    pipeline_success += 1

    return {
        "ablation": ablation,
        "positive_frame_count": float(positive_frames),
        "negative_frame_count": float(negative_frames),
        "selected_frame_count": float(selected_frames),
        "correct_target_count": float(correct_target),
        "wrong_target_count": float(wrong_target),
        "false_claim_count": float(false_claim),
        "missed_positive_count": float(missed_positive),
        "correct_command_after_correct_target_count": float(
            correct_command_after_correct_target
        ),
        "pipeline_success_count": float(pipeline_success),
        "target_recall": correct_target / max(1, positive_frames),
        "command_accuracy_after_correct_target": (
            correct_command_after_correct_target / max(1, correct_target)
        ),
        "positive_frame_pipeline_success": pipeline_success / max(1, positive_frames),
        "predicted_command_counts_after_correct_target": dict(sorted(predicted_commands.items())),
        "actual_command_counts_positive_frames": dict(sorted(actual_commands.items())),
    }


def _object_color(object_id: str) -> str:
    for color in ("red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"):
        if object_id.endswith(f"_{color}") or f"_{color}_" in object_id:
            return color
    return "unknown"


def _majority_baseline(
    train_examples: list[CommandExample],
    validation_examples: list[CommandExample],
    *,
    command_vocab: list[str],
) -> dict[str, Any]:
    counts = Counter(example.command for example in train_examples)
    majority = counts.most_common(1)[0][0]
    majority_idx = command_vocab.index(majority)
    targets = [command_vocab.index(example.command) for example in validation_examples]
    predictions = [majority_idx for _ in validation_examples]
    report = _classification_report(predictions, targets, command_vocab=command_vocab)
    return {
        "majority_command": majority,
        "accuracy": report["overall"]["accuracy"],
        "macro_f1": report["macro_f1"],
        "train_counts": dict(sorted(counts.items())),
    }


def _command_counts(examples: list[CommandExample]) -> dict[str, int]:
    return dict(sorted(Counter(example.command for example in examples).items()))


if __name__ == "__main__":
    raise SystemExit(main())
