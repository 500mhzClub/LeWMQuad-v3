#!/usr/bin/env python3
"""Evaluate a deterministic Go2 runtime-observation memory controller.

The controller records landmark ids that were visible in previous frames of the
same sequence, then answers current hidden-target queries from that memory set.
It is an upper-bound/translatability check for a Go2 perception-front-end memory:
no future labels or map distances are used by the normal condition.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-target-steering-success", type=float, default=0.90)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.12)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.30)
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    sequences = _group_sequences(rows)
    memory_by_ablation = {
        "normal": _normal_memory(sequences),
        "memory_off_abstain": _empty_memory(sequences),
        "reset_recurrent_state": _empty_memory(sequences),
        "reverse_input_history": _reverse_memory(sequences),
        "shuffle_hidden_states": _shuffle_memory(_normal_memory(sequences)),
    }
    evaluations = {
        name: _evaluate(rows, memory_by_key=memory_by_key)
        for name, memory_by_key in memory_by_ablation.items()
    }
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
    gap = float(normal["target_steering_pipeline_success"]) - corrupted_best
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and gap >= float(args.min_corrupted_gap)
    )
    report = {
        "schema": "lewm_go2_runtime_observation_memory_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "row_count": len(rows),
        "sequence_count": len(sequences),
        "validation_ablations": evaluations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": gap,
        "controller_gate_pass": bool(gate_pass),
        "config": {
            "min_target_steering_success": float(args.min_target_steering_success),
            "max_false_claim_rate": float(args.max_false_claim_rate),
            "min_corrupted_gap": float(args.min_corrupted_gap),
        },
        "claim_boundary": (
            "Deterministic runtime-observation memory upper bound. The normal "
            "condition records previously visible landmark ids from a deployable "
            "object-detector style stream and uses current relative bearing for "
            "steering. It is not a learned latent memory controller."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_runtime_observation_memory:"
        f" output={args.output}"
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


def _normal_memory(
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> dict[tuple[tuple[str, int, int], int], set[str]]:
    memory_by_key = {}
    for sequence in sequences.values():
        memory: set[str] = set()
        for row in sequence:
            memory_by_key[_row_key(row)] = set(memory)
            memory.update(str(object_id) for object_id in row.get("visible_landmark_ids", ()))
    return memory_by_key


def _reverse_memory(
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> dict[tuple[tuple[str, int, int], int], set[str]]:
    memory_by_key = {}
    for sequence in sequences.values():
        memory: set[str] = set()
        for row in reversed(sequence):
            memory_by_key[_row_key(row)] = set(memory)
            memory.update(str(object_id) for object_id in row.get("visible_landmark_ids", ()))
    return memory_by_key


def _empty_memory(
    sequences: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> dict[tuple[tuple[str, int, int], int], set[str]]:
    return {_row_key(row): set() for sequence in sequences.values() for row in sequence}


def _shuffle_memory(
    memory_by_key: dict[tuple[tuple[str, int, int], int], set[str]],
) -> dict[tuple[tuple[str, int, int], int], set[str]]:
    keys = sorted(memory_by_key)
    if len(keys) <= 1:
        return dict(memory_by_key)
    shift = max(1, len(keys) // 2)
    return {key: set(memory_by_key[keys[(idx - shift) % len(keys)]]) for idx, key in enumerate(keys)}


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    memory_by_key: dict[tuple[tuple[str, int, int], int], set[str]],
) -> dict[str, Any]:
    metrics = _Metrics()
    by_color: dict[str, _Metrics] = defaultdict(_Metrics)
    for row in rows:
        candidates = _current_candidates(row)
        if not candidates:
            continue
        positives = {object_id for object_id, target in candidates.items() if target}
        memory = memory_by_key.get(_row_key(row), set())
        selected = next((object_id for object_id in sorted(candidates) if object_id in memory), None)
        selected_color = _object_color(selected) if selected is not None else "abstain"
        positive_color = _object_color(next(iter(sorted(positives)), "")) if positives else "none"
        metrics.add(positives=positives, selected=selected)
        by_color[positive_color].add(positives=positives, selected=selected)
        if selected is not None:
            by_color[f"selected:{selected_color}"].selected_frames += 1
    result = metrics.to_dict()
    result["by_positive_target_color"] = {
        color: item.to_dict() for color, item in sorted(by_color.items()) if not color.startswith("selected:")
    }
    result["selected_color_counts"] = {
        color.removeprefix("selected:"): item.selected_frames
        for color, item in sorted(by_color.items())
        if color.startswith("selected:")
    }
    return result


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
        self.classifications: Counter[str] = Counter()

    def add(self, *, positives: set[str], selected: str | None) -> None:
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
            "target_steering_success_count": float(self.correct_target),
            "target_steering_pipeline_success": self.correct_target / max(1, self.positive_frames),
            "classification_counts": dict(sorted(self.classifications.items())),
        }


if __name__ == "__main__":
    raise SystemExit(main())
