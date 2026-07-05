#!/usr/bin/env python3
"""Evaluate a Go2 causal-memory probe as a target-selection gate.

The query probe reports per-object "seen before" decisions. This script
converts those query scores into the controller-facing decision we need next:
for each hidden-current frame, should a memory controller select a remembered
target object, and if so which one?

This is still an offline gate over rendered event slices. It does not execute a
Go2 policy or prove closed-loop navigation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_causal_memory_query_probe import (  # noqa: E402
    QueryConditionedGo2MemoryProbe,
    _build_frames,
    _current_role,
    _max_landmark_slot,
    _sequence_tensors,
    _scrub_command_aux,
)
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _load_rows,
    _resolve_device,
)
from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--color-threshold",
        action="append",
        default=[],
        metavar="COLOR=VALUE",
        help="Override selection/query threshold for a landmark color.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--labels",
        nargs="*",
        type=Path,
        default=None,
        help="Optional full derived labels for future-claim scoring.",
    )
    parser.add_argument("--claim-bfs-distance", type=int, default=1)
    parser.add_argument("--claim-range-m", type=float, default=1.0)
    parser.add_argument(
        "--require-hidden-claim",
        action="store_true",
        help="Count future claims only when the selected landmark is hidden.",
    )
    parser.add_argument(
        "--scrub-command-aux",
        action="store_true",
        help="Zero current command fields before building aux features.",
    )
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    threshold = _checkpoint_threshold(checkpoint) if args.threshold is None else float(args.threshold)
    color_thresholds = _parse_color_thresholds(args.color_threshold)
    rows_raw = _load_rows(args.datasets)
    rows = (
        _scrub_command_aux(rows_raw)
        if bool(args.scrub_command_aux)
        or bool(dict(checkpoint.get("args", {})).get("scrub_command_aux", False))
        else rows_raw
    )
    if not rows:
        raise SystemExit("no evaluation rows")

    max_slot = _recover_max_slot(checkpoint, fallback_rows=rows)
    feature_stats = {
        "mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
    }
    checkpoint_args = dict(checkpoint.get("args", {}))
    sequences = _build_frames(
        rows,
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
    if not sequences:
        raise SystemExit("no evaluable sequences")
    label_index = _load_label_index(args.labels or ())

    device = _resolve_device(str(args.device))
    jepa_encoder = None
    jepa_encoder_dim = None
    frozen_jepa_checkpoint = checkpoint.get("frozen_jepa_checkpoint")
    if frozen_jepa_checkpoint:
        jepa_encoder, jepa_checkpoint = load_go2_jepa_encoder(
            Path(str(frozen_jepa_checkpoint)),
            device=device,
            freeze=True,
        )
        jepa_encoder_dim = int(jepa_checkpoint.get("latent_dim", checkpoint["hidden_dim"]))

    model = QueryConditionedGo2MemoryProbe(
        aux_dim=int(checkpoint["aux_dim"]),
        query_dim=int(checkpoint["query_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        encoder=jepa_encoder,
        encoder_output_dim=jepa_encoder_dim,
        freeze_encoder=bool(frozen_jepa_checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    ablations = (
        "normal",
        "reset_recurrent_state",
        "reverse_input_history",
        "shuffle_hidden_states",
    )
    evaluations = {
        ablation: _evaluate_target_gate(
            model,
            sequences,
            device=device,
            threshold=threshold,
            color_thresholds=color_thresholds,
            ablation=ablation,
            label_index=label_index,
            claim_bfs_distance=int(args.claim_bfs_distance),
            claim_range_m=float(args.claim_range_m),
            require_hidden_claim=bool(args.require_hidden_claim),
        )
        for ablation in ablations
    }
    normal = evaluations["normal"]["overall"]
    corrupted_best = max(
        evaluations["reset_recurrent_state"]["overall"]["balanced_frame_accuracy"],
        evaluations["reverse_input_history"]["overall"]["balanced_frame_accuracy"],
        evaluations["shuffle_hidden_states"]["overall"]["balanced_frame_accuracy"],
    )
    report = {
        "schema": "lewm_go2_causal_memory_target_gate_report_v0",
        "checkpoint": str(args.checkpoint),
        "datasets": [str(path) for path in args.datasets],
        "device": str(device),
        "threshold": float(threshold),
        "color_thresholds": dict(sorted(color_thresholds.items())),
        "sequence_count": len(sequences),
        "row_count": sum(len(sequence) for sequence in sequences.values()),
        "max_landmark_slot": int(max_slot),
        "labels": [str(path) for path in (args.labels or ())],
        "future_claim_config": {
            "claim_bfs_distance": int(args.claim_bfs_distance),
            "claim_range_m": float(args.claim_range_m),
            "require_hidden_claim": bool(args.require_hidden_claim),
        },
        "evaluations": evaluations,
        "normal_minus_best_corrupted_balanced_frame_accuracy": (
            float(normal["balanced_frame_accuracy"]) - float(corrupted_best)
        ),
        "claim_boundary": (
            "Offline target-selection gate over rendered Go2 causal-memory event "
            "slices. A positive result supports learned-memory translatability; "
            "it is not yet a closed-loop Go2 control result."
        ),
    }

    out = args.out or args.checkpoint.with_suffix(".target_gate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_causal_memory_target_gate:"
        f" report={out}"
        f" frame_bal={normal['balanced_frame_accuracy']:.3f}"
        f" pos_recall={normal['positive_frame_recall']:.3f}"
        f" neg_abstain={normal['negative_frame_abstain_specificity']:.3f}"
        f" delta={report['normal_minus_best_corrupted_balanced_frame_accuracy']:.3f}"
    )
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "lewm_go2_causal_memory_query_probe_checkpoint_v0":
        raise SystemExit(f"unsupported checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


def _checkpoint_threshold(checkpoint: dict[str, Any]) -> float:
    args = dict(checkpoint.get("args", {}))
    return float(args.get("threshold", 0.5))


def _parse_color_thresholds(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--color-threshold must be COLOR=VALUE, got: {item}")
        color, value = item.split("=", 1)
        color = color.strip()
        if not color:
            raise SystemExit(f"empty color in --color-threshold: {item}")
        result[color] = float(value)
    return result


def _recover_max_slot(checkpoint: dict[str, Any], *, fallback_rows: list[dict[str, Any]]) -> int:
    rows_by_path: list[dict[str, Any]] = []
    for path in _checkpoint_dataset_paths(checkpoint):
        if path.is_file():
            rows_by_path.extend(_load_rows([path]))
    if rows_by_path:
        return _max_landmark_slot(rows_by_path)
    return _max_landmark_slot(fallback_rows)


def _checkpoint_dataset_paths(checkpoint: dict[str, Any]) -> list[Path]:
    args = dict(checkpoint.get("args", {}))
    paths: list[Path] = []
    for key in ("datasets", "validation_datasets"):
        value = args.get(key, [])
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            value = [value]
        for item in value:
            path = Path(str(item))
            if path not in paths:
                paths.append(path)
    return paths


def _load_label_index(paths: list[Path] | tuple[Path, ...]) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw_path in paths:
        path = _resolve_labels_path(raw_path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row.get("scene_id", "")),
                    int(row.get("env_idx", 0)),
                    int(row.get("episode_id", 0)),
                )
                index[key].append(row)
    for rows in index.values():
        rows.sort(key=lambda item: int(item.get("episode_step", 0)))
    return index


def _resolve_labels_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    for candidate in (path / "labels.jsonl", path / "derived_labels" / "labels.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"missing labels.jsonl under: {path}")


def _evaluate_target_gate(
    model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Any]],
    *,
    device: torch.device,
    threshold: float,
    color_thresholds: dict[str, float],
    ablation: str,
    label_index: dict[tuple[str, int, int], list[dict[str, Any]]],
    claim_bfs_distance: int,
    claim_range_m: float,
    require_hidden_claim: bool,
) -> dict[str, Any]:
    model.eval()
    hidden_by_key = _hidden_states_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
    )
    query_metrics = _BinaryMetrics()
    frame_metrics = _FrameMetrics()
    by_positive_color: dict[str, _FrameMetrics] = defaultdict(_FrameMetrics)
    by_selected_color: dict[str, _SelectionCounter] = defaultdict(_SelectionCounter)
    future_claim_metrics = _FutureClaimMetrics()

    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                current_queries = [query for query in frame.queries if _current_role(query)]
                if not current_queries:
                    continue
                query_features = torch.stack([query.features for query in current_queries]).to(device)
                hidden_rows = hidden[step_idx].repeat(len(current_queries), 1)
                probs = torch.sigmoid(model.score_queries(hidden_rows, query_features))
                probs_np = probs.detach().cpu().numpy()
                object_scores: dict[str, float] = {}
                object_colors: dict[str, str] = {}
                object_targets: dict[str, float] = {}
                for query, prob in zip(current_queries, probs_np):
                    probability = float(prob)
                    prediction = probability >= _threshold_for_color(
                        query.color,
                        default_threshold=threshold,
                        color_thresholds=color_thresholds,
                    )
                    target = float(query.target) >= 0.5
                    query_metrics.add(prediction=prediction, target=target)
                    object_scores[query.object_id] = max(
                        object_scores.get(query.object_id, -1.0),
                        probability,
                    )
                    object_colors[query.object_id] = query.color
                    object_targets[query.object_id] = max(
                        object_targets.get(query.object_id, 0.0),
                        float(query.target),
                    )

                selected_object = _select_object(
                    object_scores,
                    object_colors=object_colors,
                    default_threshold=threshold,
                    color_thresholds=color_thresholds,
                )
                positive_objects = {
                    object_id
                    for object_id, target in object_targets.items()
                    if float(target) >= 0.5
                }
                frame_result = frame_metrics.add(
                    selected_object=selected_object,
                    positive_objects=positive_objects,
                )
                selected_color = (
                    "abstain"
                    if selected_object is None
                    else object_colors.get(selected_object, "unknown")
                )
                by_selected_color[selected_color].add(frame_result)
                positive_colors = {
                    object_colors.get(object_id, "unknown")
                    for object_id in positive_objects
                }
                for color in sorted(positive_colors) or ["none"]:
                    by_positive_color[color].add_result(frame_result)
                if label_index:
                    future_claim_metrics.add(
                        selected_object=selected_object,
                        positive_objects=positive_objects,
                        future_claim=_future_claim(
                            label_index,
                            key=frame.seq_key,
                            start_step=int(frame.episode_step),
                            object_id=selected_object,
                            claim_bfs_distance=claim_bfs_distance,
                            claim_range_m=claim_range_m,
                            require_hidden_claim=require_hidden_claim,
                        )
                        if selected_object is not None
                        else None,
                    )

    result = {
        "ablation": ablation,
        "overall": frame_metrics.to_dict(query_metrics=query_metrics.to_dict()),
        "by_positive_target_color": {
            key: value.to_dict() for key, value in sorted(by_positive_color.items())
        },
        "by_selected_color": {
            key: value.to_dict() for key, value in sorted(by_selected_color.items())
        },
    }
    if label_index:
        result["future_claim_metrics"] = future_claim_metrics.to_dict()
    return result


def _hidden_states_by_sequence(
    model: QueryConditionedGo2MemoryProbe,
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
                hidden = model.forward_hidden(images, aux)
            elif ablation == "reset_recurrent_state":
                hidden = model.forward_hidden(images, aux, reset_each_step=True)
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                hidden = model.forward_hidden(images[order], aux[order]).flip(0)
            elif ablation == "shuffle_hidden_states":
                hidden = model.forward_hidden(images, aux)
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
    shift = max(1, cursor // 2)
    shuffled = torch.roll(flat, shifts=shift, dims=0)
    return {
        key: shuffled[start:end]
        for key, (start, end) in spans.items()
    }


def _select_object(
    object_scores: dict[str, float],
    *,
    object_colors: dict[str, str],
    default_threshold: float,
    color_thresholds: dict[str, float],
) -> str | None:
    if not object_scores:
        return None
    margins = {
        object_id: probability
        - _threshold_for_color(
            object_colors.get(object_id, "unknown"),
            default_threshold=default_threshold,
            color_thresholds=color_thresholds,
        )
        for object_id, probability in object_scores.items()
    }
    object_id, margin = max(margins.items(), key=lambda item: (item[1], item[0]))
    if margin < 0.0:
        return None
    return object_id


def _threshold_for_color(
    color: str,
    *,
    default_threshold: float,
    color_thresholds: dict[str, float],
) -> float:
    return float(color_thresholds.get(color, default_threshold))


def _future_claim(
    label_index: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    key: tuple[str, int, int],
    start_step: int,
    object_id: str,
    claim_bfs_distance: int,
    claim_range_m: float,
    require_hidden_claim: bool,
) -> dict[str, Any] | None:
    rows = label_index.get(key, ())
    for row in rows:
        step = int(row.get("episode_step", 0))
        if step < int(start_step):
            continue
        landmark = _landmark_by_id(row).get(object_id)
        if landmark is None:
            continue
        visible = bool(landmark.get("visible", False))
        if require_hidden_claim and visible:
            continue
        bfs = landmark.get("bfs_distance_cells")
        range_m = float(landmark.get("range_m", float("inf")))
        if (bfs is not None and int(bfs) <= int(claim_bfs_distance)) or range_m <= float(
            claim_range_m
        ):
            return {
                "claim_step": step,
                "claim_visible": visible,
                "claim_bfs_distance": None if bfs is None else int(bfs),
                "claim_range_m": range_m,
            }
    return None


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): dict(landmark)
        for landmark in row.get("landmarks", ())
    }


class _BinaryMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, *, prediction: bool, target: bool) -> None:
        if prediction and target:
            self.tp += 1
        elif prediction and not target:
            self.fp += 1
        elif not prediction and target:
            self.fn += 1
        else:
            self.tn += 1

    def to_dict(self) -> dict[str, float]:
        positive = self.tp + self.fn
        negative = self.tn + self.fp
        total = positive + negative
        recall = self.tp / max(1, positive)
        specificity = self.tn / max(1, negative)
        precision = self.tp / max(1, self.tp + self.fp)
        f1 = (2.0 * self.tp) / max(1, 2 * self.tp + self.fp + self.fn)
        return {
            "accuracy": (self.tp + self.tn) / max(1, total),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "positive_recall": recall,
            "negative_specificity": specificity,
            "precision": precision,
            "f1": f1,
            "true_positive_count": float(self.tp),
            "true_negative_count": float(self.tn),
            "false_positive_count": float(self.fp),
            "false_negative_count": float(self.fn),
            "positive_count": float(positive),
            "negative_count": float(negative),
            "target_count": float(total),
        }


class _FrameMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.correct_positive = 0
        self.correct_negative = 0
        self.wrong_object = 0
        self.false_claim = 0
        self.missed_positive = 0
        self.selected = 0

    def add(self, *, selected_object: str | None, positive_objects: set[str]) -> str:
        selected = selected_object is not None
        if selected:
            self.selected += 1
        if positive_objects:
            self.positive_frames += 1
            if selected_object in positive_objects:
                self.correct_positive += 1
                return "correct_positive"
            if selected_object is None:
                self.missed_positive += 1
                return "missed_positive"
            self.wrong_object += 1
            return "wrong_object"

        self.negative_frames += 1
        if selected:
            self.false_claim += 1
            return "false_claim"
        self.correct_negative += 1
        return "correct_negative"

    def add_result(self, result: str) -> None:
        if result == "correct_positive":
            self.positive_frames += 1
            self.correct_positive += 1
            self.selected += 1
        elif result == "missed_positive":
            self.positive_frames += 1
            self.missed_positive += 1
        elif result == "wrong_object":
            self.positive_frames += 1
            self.wrong_object += 1
            self.selected += 1
        elif result == "false_claim":
            self.negative_frames += 1
            self.false_claim += 1
            self.selected += 1
        elif result == "correct_negative":
            self.negative_frames += 1
            self.correct_negative += 1
        else:
            raise ValueError(f"unknown frame result: {result}")

    def to_dict(self, *, query_metrics: dict[str, float] | None = None) -> dict[str, Any]:
        frame_count = self.positive_frames + self.negative_frames
        positive_recall = self.correct_positive / max(1, self.positive_frames)
        negative_specificity = self.correct_negative / max(1, self.negative_frames)
        precision = self.correct_positive / max(1, self.selected)
        result: dict[str, Any] = {
            "frame_count": float(frame_count),
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "positive_frame_recall": positive_recall,
            "negative_frame_abstain_specificity": negative_specificity,
            "balanced_frame_accuracy": 0.5 * (positive_recall + negative_specificity),
            "frame_accuracy": (self.correct_positive + self.correct_negative) / max(1, frame_count),
            "target_selection_precision": precision,
            "selected_frame_count": float(self.selected),
            "correct_positive_count": float(self.correct_positive),
            "correct_negative_count": float(self.correct_negative),
            "wrong_object_count": float(self.wrong_object),
            "false_claim_count": float(self.false_claim),
            "missed_positive_count": float(self.missed_positive),
        }
        if query_metrics is not None:
            result["query_metrics"] = query_metrics
        return result


class _SelectionCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, result: str) -> None:
        self.counts[result] += 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


class _FutureClaimMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.selected_positive = 0
        self.selected_positive_future_claim = 0
        self.selected_positive_hidden_claim = 0
        self.selected_positive_visible_claim = 0
        self.selected_positive_no_future_claim = 0
        self.positive_not_selected = 0
        self.negative_selected = 0
        self.negative_selected_future_claim = 0
        self.negative_abstained = 0

    def add(
        self,
        *,
        selected_object: str | None,
        positive_objects: set[str],
        future_claim: dict[str, Any] | None,
    ) -> None:
        if positive_objects:
            self.positive_frames += 1
            if selected_object in positive_objects:
                self.selected_positive += 1
                if future_claim is None:
                    self.selected_positive_no_future_claim += 1
                else:
                    self.selected_positive_future_claim += 1
                    if bool(future_claim.get("claim_visible", False)):
                        self.selected_positive_visible_claim += 1
                    else:
                        self.selected_positive_hidden_claim += 1
            else:
                self.positive_not_selected += 1
            return

        self.negative_frames += 1
        if selected_object is None:
            self.negative_abstained += 1
            return
        self.negative_selected += 1
        if future_claim is not None:
            self.negative_selected_future_claim += 1

    def to_dict(self) -> dict[str, float]:
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "selected_positive_count": float(self.selected_positive),
            "selected_positive_future_claim_count": float(
                self.selected_positive_future_claim
            ),
            "selected_positive_hidden_claim_count": float(
                self.selected_positive_hidden_claim
            ),
            "selected_positive_visible_claim_count": float(
                self.selected_positive_visible_claim
            ),
            "selected_positive_no_future_claim_count": float(
                self.selected_positive_no_future_claim
            ),
            "positive_not_selected_count": float(self.positive_not_selected),
            "negative_selected_count": float(self.negative_selected),
            "negative_selected_future_claim_count": float(
                self.negative_selected_future_claim
            ),
            "negative_abstained_count": float(self.negative_abstained),
            "positive_future_claim_rate": (
                self.selected_positive_future_claim / max(1, self.positive_frames)
            ),
            "selected_positive_future_claim_rate": (
                self.selected_positive_future_claim / max(1, self.selected_positive)
            ),
            "negative_selection_rate": (
                self.negative_selected / max(1, self.negative_frames)
            ),
        }


if __name__ == "__main__":
    raise SystemExit(main())
