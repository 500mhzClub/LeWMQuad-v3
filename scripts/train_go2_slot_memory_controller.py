#!/usr/bin/env python3
"""Train and evaluate an explicit differentiable Go2 slot-memory controller.

This controller is deliberately narrower than the earlier generic GRU memory
attempts. A runtime perception front end supplies per-landmark visibility,
range, and bearing observations. The learned module writes visible landmark
slots into a persistent memory vector and current hidden-target queries read
that vector. Steering remains the local runtime bearing rule.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_hidden_target_memory_probe import _resolve_device  # noqa: E402


STEERING_CLASSES = ("right", "forward", "left")


class SlotMemoryController(nn.Module):
    """Differentiable per-slot memory write/read controller."""

    def __init__(self, slot_count: int) -> None:
        super().__init__()
        self.slot_count = int(slot_count)
        self.write_logit = nn.Parameter(torch.full((self.slot_count,), 3.0))
        self.read_gain = nn.Parameter(torch.tensor(12.0))
        self.read_threshold = nn.Parameter(torch.tensor(0.5))

    def initial_memory(self, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.slot_count, dtype=torch.float32, device=device)

    def write(self, memory: torch.Tensor, visible_slots: torch.Tensor) -> torch.Tensor:
        write_prob = visible_slots.to(dtype=memory.dtype) * torch.sigmoid(self.write_logit)
        return 1.0 - (1.0 - memory) * (1.0 - write_prob)

    def read_logits(self, memory: torch.Tensor, query_slots: torch.Tensor) -> torch.Tensor:
        slot_memory = memory[query_slots]
        return torch.clamp(self.read_gain, min=1.0) * (slot_memory - self.read_threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--memory-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-threshold", type=float, default=0.5)
    parser.add_argument("--arc-threshold-rad", type=float, default=0.1)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260697)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-target-steering-success", type=float, default=0.90)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.12)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.30)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows = _load_rows(args.datasets)
    validation_rows = _load_rows(args.validation_datasets)
    if not train_rows:
        raise SystemExit("no train rows")
    if not validation_rows:
        raise SystemExit("no validation rows")
    slot_count = max(_max_slot(train_rows), _max_slot(validation_rows)) + 1
    if slot_count <= 0:
        raise SystemExit("no landmark slots found")

    device = _resolve_device(str(args.device))
    model = SlotMemoryController(slot_count=slot_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    train_sequences = _group_sequences(train_rows)
    validation_sequences = _group_sequences(validation_rows)

    history = []
    best_score = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            memory_state_loss_weight=float(args.memory_state_loss_weight),
            query_loss_weight=float(args.query_loss_weight),
        )
        validation_ablations = _evaluate_ablations(
            model,
            validation_sequences,
            device=device,
            selection_threshold=float(args.selection_threshold),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
        )
        score = _selection_score(validation_ablations)
        normal = validation_ablations["normal"]
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation": normal,
                "normal_minus_best_corrupted_target_steering_pipeline_success": (
                    float(normal["target_steering_pipeline_success"])
                    - max(
                        float(validation_ablations[name]["target_steering_pipeline_success"])
                        for name in (
                            "memory_off_abstain",
                            "reset_recurrent_state",
                            "reverse_input_history",
                            "shuffle_hidden_states",
                        )
                    )
                ),
            }
        )
        if score >= best_score:
            best_score = float(score)
            best_metrics = validation_ablations
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={train_loss:.4f}"
                f" target_steer={normal['target_steering_pipeline_success']:.3f}"
                f" false_claim={normal['false_claim_rate']:.3f}"
                f" precision={normal['target_selection_precision']:.3f}"
                f" score={score:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate(
        train_sequences,
        model=model,
        memory_by_key=_memory_trace(model, train_sequences, device=device, mode="normal"),
        device=device,
        selection_threshold=float(args.selection_threshold),
        arc_threshold_rad=float(args.arc_threshold_rad),
        yaw_threshold_rad=float(args.yaw_threshold_rad),
        hold_range_m=float(args.hold_range_m),
    )
    validation_ablations = _evaluate_ablations(
        model,
        validation_sequences,
        device=device,
        selection_threshold=float(args.selection_threshold),
        arc_threshold_rad=float(args.arc_threshold_rad),
        yaw_threshold_rad=float(args.yaw_threshold_rad),
        hold_range_m=float(args.hold_range_m),
    )
    normal = validation_ablations["normal"]
    corrupted_best = max(
        float(validation_ablations[name]["target_steering_pipeline_success"])
        for name in (
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    )
    gap = float(normal["target_steering_pipeline_success"]) - corrupted_best
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and gap >= float(args.min_corrupted_gap)
    )

    checkpoint = {
        "schema": "lewm_go2_slot_memory_controller_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "slot_count": int(slot_count),
        "selection_threshold": float(args.selection_threshold),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_slot_memory_controller_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "slot_count": int(slot_count),
        "selection_threshold": float(args.selection_threshold),
        "model_parameters": {
            "write_probability": torch.sigmoid(model.write_logit).detach().cpu().tolist(),
            "read_gain": float(torch.clamp(model.read_gain.detach().cpu(), min=1.0)),
            "read_threshold": float(model.read_threshold.detach().cpu()),
        },
        "final_train": final_train,
        "validation_ablations": validation_ablations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": gap,
        "controller_gate_pass": bool(gate_pass),
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Differentiable Go2 slot-memory controller over runtime landmark "
            "observations. It learns write/read parameters for a persistent "
            "per-landmark memory vector and uses current relative bearing for "
            "the steering rule. It is not a pure RGB latent-memory controller."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_slot_memory_controller:"
        f" output={args.output}"
        f" report={report_path}"
        f" target_steer={normal['target_steering_pipeline_success']:.3f}"
        f" false_claim={normal['false_claim_rate']:.3f}"
        f" gap={gap:.3f}"
        f" pass={bool(gate_pass)}"
    )
    return 0


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _group_sequences(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[_seq_key(row)].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda item: int(item.get("episode_step", 0)))
    return dict(sequences)


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _row_key(row: dict[str, Any]) -> tuple[tuple[str, int, int], int]:
    return (_seq_key(row), int(row.get("episode_step", 0)))


def _max_slot(rows: list[dict[str, Any]]) -> int:
    slots = []
    for row in rows:
        for landmark in row.get("landmarks", ()):
            slot = _landmark_slot(str(landmark.get("object_id", "")))
            if slot is not None:
                slots.append(slot)
    return max(slots) if slots else -1


def _landmark_slot(object_id: str) -> int | None:
    for part in str(object_id).split("_"):
        if part.isdigit():
            return int(part)
    return None


def _train_epoch(
    model: SlotMemoryController,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    device: torch.device,
    memory_state_loss_weight: float,
    query_loss_weight: float,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    trained = 0
    for key in keys:
        sequence = sequences[key]
        memory = model.initial_memory(device=device)
        target_memory = torch.zeros(model.slot_count, dtype=torch.float32, device=device)
        losses = []
        for row in sequence:
            query_slots, query_targets = _query_batch(row, device=device)
            if query_slots.numel() > 0:
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        model.read_logits(memory, query_slots),
                        query_targets,
                    )
                    * float(query_loss_weight)
                )
            visible = _visible_slots(row, slot_count=model.slot_count, device=device)
            memory = model.write(memory, visible)
            target_memory = torch.maximum(target_memory, visible)
            losses.append(
                F.binary_cross_entropy(
                    memory.clamp(1e-5, 1.0 - 1e-5),
                    target_memory,
                )
                * float(memory_state_loss_weight)
            )
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        trained += 1
    return total_loss / max(1, trained)


def _query_batch(row: dict[str, Any], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = _current_candidates(row)
    slots = []
    targets = []
    for object_id, target in sorted(candidates.items()):
        slot = _landmark_slot(object_id)
        if slot is None:
            continue
        slots.append(slot)
        targets.append(1.0 if target else 0.0)
    if not slots:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(slots, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _visible_slots(row: dict[str, Any], *, slot_count: int, device: torch.device) -> torch.Tensor:
    visible = torch.zeros(slot_count, dtype=torch.float32, device=device)
    for object_id in row.get("visible_landmark_ids", ()):
        slot = _landmark_slot(str(object_id))
        if slot is not None and 0 <= slot < slot_count:
            visible[slot] = 1.0
    return visible


def _current_candidates(row: dict[str, Any]) -> dict[str, bool]:
    candidates: dict[str, bool] = {}
    for event in row.get("go2_causal_memory_pair_selection", ()):
        role = str(event.get("pair_role", ""))
        if not role.startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        if not object_id:
            continue
        candidates[object_id] = bool(candidates.get(object_id, False)) or bool(
            event.get("seen_before", False)
        )
    return candidates


def _memory_trace(
    model: SlotMemoryController,
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    device: torch.device,
    mode: str,
) -> dict[tuple[tuple[str, int, int], int], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        if mode in ("memory_off_abstain", "reset_recurrent_state"):
            return {
                _row_key(row): model.initial_memory(device=device).detach().cpu()
                for sequence in sequences.values()
                for row in sequence
            }
        if mode == "normal":
            return _directional_trace(model, sequences, device=device, reverse=False)
        if mode == "reverse_input_history":
            return _directional_trace(model, sequences, device=device, reverse=True)
        if mode == "shuffle_hidden_states":
            normal = _directional_trace(model, sequences, device=device, reverse=False)
            return _shuffle_trace(normal)
    raise ValueError(f"unknown memory trace mode: {mode}")


def _directional_trace(
    model: SlotMemoryController,
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    device: torch.device,
    reverse: bool,
) -> dict[tuple[tuple[str, int, int], int], torch.Tensor]:
    trace = {}
    for sequence in sequences.values():
        memory = model.initial_memory(device=device)
        iterable = list(reversed(sequence)) if reverse else sequence
        for row in iterable:
            trace[_row_key(row)] = memory.detach().cpu()
            visible = _visible_slots(row, slot_count=model.slot_count, device=device)
            memory = model.write(memory, visible)
    return trace


def _shuffle_trace(
    trace: dict[tuple[tuple[str, int, int], int], torch.Tensor],
) -> dict[tuple[tuple[str, int, int], int], torch.Tensor]:
    keys = sorted(trace)
    if len(keys) <= 1:
        return dict(trace)
    shift = max(1, len(keys) // 2)
    return {key: trace[keys[(idx - shift) % len(keys)]].clone() for idx, key in enumerate(keys)}


def _evaluate_ablations(
    model: SlotMemoryController,
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    device: torch.device,
    selection_threshold: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> dict[str, dict[str, Any]]:
    return {
        mode: _evaluate(
            sequences,
            model=model,
            memory_by_key=_memory_trace(model, sequences, device=device, mode=mode),
            device=device,
            selection_threshold=selection_threshold,
            arc_threshold_rad=arc_threshold_rad,
            yaw_threshold_rad=yaw_threshold_rad,
            hold_range_m=hold_range_m,
        )
        for mode in (
            "normal",
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }


def _evaluate(
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    model: SlotMemoryController,
    memory_by_key: dict[tuple[tuple[str, int, int], int], torch.Tensor],
    device: torch.device,
    selection_threshold: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> dict[str, Any]:
    metrics = _Metrics()
    by_color: dict[str, _Metrics] = defaultdict(_Metrics)
    selected_color_counts: Counter[str] = Counter()
    for sequence in sequences.values():
        for row in sequence:
            candidates = _current_candidates(row)
            if not candidates:
                continue
            memory = memory_by_key.get(_row_key(row))
            if memory is None:
                continue
            positives = {object_id for object_id, target in candidates.items() if target}
            selected = _select(
                model,
                memory,
                candidates,
                device=device,
                probability_threshold=float(selection_threshold),
            )
            predicted_steering = (
                _target_steering(
                    row,
                    selected,
                    arc_threshold_rad=arc_threshold_rad,
                    yaw_threshold_rad=yaw_threshold_rad,
                    hold_range_m=hold_range_m,
                )
                if selected is not None
                else None
            )
            target_steering = (
                _target_steering(
                    row,
                    selected,
                    arc_threshold_rad=arc_threshold_rad,
                    yaw_threshold_rad=yaw_threshold_rad,
                    hold_range_m=hold_range_m,
                )
                if selected in positives
                else None
            )
            metrics.add(
                positives=positives,
                selected=selected,
                predicted_steering=predicted_steering,
                target_steering=target_steering,
            )
            positive_color = _object_color(next(iter(sorted(positives)), "")) if positives else "none"
            by_color[positive_color].add(
                positives=positives,
                selected=selected,
                predicted_steering=predicted_steering,
                target_steering=target_steering,
            )
            if selected is not None:
                selected_color_counts[_object_color(selected)] += 1
    result = metrics.to_dict()
    result["by_positive_target_color"] = {
        color: item.to_dict() for color, item in sorted(by_color.items())
    }
    result["selected_color_counts"] = dict(sorted(selected_color_counts.items()))
    return result


def _select(
    model: SlotMemoryController,
    memory: torch.Tensor,
    candidates: dict[str, bool],
    *,
    device: torch.device,
    probability_threshold: float,
) -> str | None:
    object_ids = []
    slots = []
    for object_id in sorted(candidates):
        slot = _landmark_slot(object_id)
        if slot is not None and 0 <= slot < int(memory.numel()):
            object_ids.append(object_id)
            slots.append(slot)
    if not slots:
        return None
    query_slots = torch.tensor(slots, dtype=torch.long, device=device)
    with torch.no_grad():
        probabilities = (
            torch.sigmoid(model.read_logits(memory.to(device), query_slots))
            .detach()
            .cpu()
            .tolist()
        )
    best_object = None
    best_score = float("-inf")
    for object_id, score in zip(object_ids, probabilities, strict=True):
        score = float(score)
        if score > best_score:
            best_object = object_id
            best_score = score
    if best_object is None or best_score < float(probability_threshold):
        return None
    return best_object


def _selection_score(evaluations: dict[str, dict[str, Any]]) -> float:
    normal = evaluations["normal"]
    corrupted_best = max(
        float(evaluations[name]["target_steering_pipeline_success"])
        for name in (
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    )
    return (
        2.0 * float(normal["target_steering_pipeline_success"])
        - 1.0 * float(normal["false_claim_rate"])
        + 0.75 * (float(normal["target_steering_pipeline_success"]) - corrupted_best)
        + 0.25 * float(normal["target_selection_precision"])
    )


def _target_steering(
    row: dict[str, Any],
    object_id: str | None,
    *,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> str:
    for landmark in row.get("landmarks", ()):
        if str(landmark.get("object_id", "")) != str(object_id):
            continue
        bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
        range_m = _finite_float(landmark.get("range_m"), 1.0)
        if range_m <= hold_range_m:
            return "forward"
        if bearing >= arc_threshold_rad or bearing >= yaw_threshold_rad:
            return "left"
        if bearing <= -arc_threshold_rad or bearing <= -yaw_threshold_rad:
            return "right"
        return "forward"
    return "forward"


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _object_color(object_id: str | None) -> str:
    text = str(object_id or "")
    for color in ("red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"):
        if color in text:
            return color
    return "unknown"


class _Metrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.selected_frames = 0
        self.correct_target = 0
        self.false_claim = 0
        self.wrong_object = 0
        self.missed_positive = 0
        self.target_steer_success = 0
        self.classifications: Counter[str] = Counter()

    def add(
        self,
        *,
        positives: set[str],
        selected: str | None,
        predicted_steering: str | None,
        target_steering: str | None,
    ) -> None:
        if positives:
            self.positive_frames += 1
        else:
            self.negative_frames += 1
        if selected is None:
            if positives:
                self.missed_positive += 1
                self.classifications["missed_positive"] += 1
            else:
                self.classifications["abstain"] += 1
            return
        self.selected_frames += 1
        if selected in positives:
            self.correct_target += 1
            self.classifications["correct_target"] += 1
            if predicted_steering == target_steering and predicted_steering is not None:
                self.target_steer_success += 1
        elif positives:
            self.wrong_object += 1
            self.classifications["wrong_object"] += 1
        else:
            self.false_claim += 1
            self.classifications["false_claim"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "selected_frame_count": float(self.selected_frames),
            "correct_target_count": float(self.correct_target),
            "missed_positive_count": float(self.missed_positive),
            "false_claim_count": float(self.false_claim),
            "wrong_object_count": float(self.wrong_object),
            "target_recall": self.correct_target / max(1, self.positive_frames),
            "false_claim_rate": self.false_claim / max(1, self.negative_frames),
            "target_selection_precision": self.correct_target / max(1, self.selected_frames),
            "target_steering_success_count": float(self.target_steer_success),
            "target_steering_pipeline_success": self.target_steer_success
            / max(1, self.positive_frames),
            "classification_counts": dict(sorted(self.classifications.items())),
        }


if __name__ == "__main__":
    raise SystemExit(main())
