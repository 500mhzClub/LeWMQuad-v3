#!/usr/bin/env python3
"""Evaluate a direct frozen-JEPA Go2 target-selection gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402
from train_go2_causal_memory_query_probe import (  # noqa: E402
    _build_frames,
    _color_vocab,
    _max_landmark_slot,
    _scrub_command_aux,
    _scrub_runtime_aux,
)
from train_go2_frozen_jepa_target_gate import (  # noqa: E402
    DirectGo2TargetGate,
    _evaluate,
)
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _load_rows,
    _resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--selection-margin",
        type=float,
        action="append",
        default=None,
        help="Candidate logit must exceed abstain logit by this margin; repeat to sweep.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    rows_raw = _load_rows(args.datasets)
    if not rows_raw:
        raise SystemExit("no evaluation rows")
    checkpoint_args = dict(checkpoint.get("args", {}))
    if bool(checkpoint.get("scrubbed_runtime_aux", False)) or bool(
        checkpoint_args.get("scrub_runtime_aux", False)
    ):
        rows = _scrub_runtime_aux(rows_raw)
    elif bool(checkpoint.get("scrubbed_command_aux", False)) or bool(
        checkpoint_args.get("scrub_command_aux", False)
    ):
        rows = _scrub_command_aux(rows_raw)
    else:
        rows = rows_raw
    feature_stats = {
        "mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
    }
    color_vocab = list(checkpoint["color_vocab"])
    if not color_vocab:
        color_vocab = _color_vocab(rows)
    sequences = _build_frames(
        rows,
        primitive_vocab=list(checkpoint["primitive_vocab"]),
        color_vocab=color_vocab,
        max_slot=_max_landmark_slot(rows),
        feature_stats=feature_stats,
        image_size=int(checkpoint["image_size"]),
        include_object_slot=bool(checkpoint_args.get("include_object_slot", False)),
        include_privileged_landmark_geometry=False,
    )
    if not sequences:
        raise SystemExit("no evaluable sequences")

    device = _resolve_device(str(args.device))
    encoder, jepa_checkpoint = load_go2_jepa_encoder(
        checkpoint["frozen_jepa_checkpoint"],
        device=device,
        freeze=True,
    )
    model = DirectGo2TargetGate(
        encoder=encoder,
        encoder_output_dim=int(jepa_checkpoint.get("latent_dim", checkpoint["hidden_dim"])),
        aux_dim=int(checkpoint["aux_dim"]),
        query_dim=int(checkpoint["query_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    margins = args.selection_margin
    if not margins:
        margins = [float(checkpoint.get("selection_margin", 0.0))]
    evaluations: dict[str, Any] = {}
    for margin in margins:
        ablations = {
            ablation: _evaluate(
                model,
                sequences,
                device=device,
                selection_margin=float(margin),
                ablation=ablation,
            )
            for ablation in (
                "normal",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_hidden_states",
            )
        }
        normal = ablations["normal"]["overall"]
        corrupted_best = max(
            ablations["reset_recurrent_state"]["overall"]["balanced_frame_accuracy"],
            ablations["reverse_input_history"]["overall"]["balanced_frame_accuracy"],
            ablations["shuffle_hidden_states"]["overall"]["balanced_frame_accuracy"],
        )
        evaluations[str(float(margin))] = {
            "selection_margin": float(margin),
            "ablations": ablations,
            "normal_minus_best_corrupted_balanced_frame_accuracy": (
                float(normal["balanced_frame_accuracy"]) - float(corrupted_best)
            ),
        }

    report = {
        "schema": "lewm_go2_frozen_jepa_target_gate_eval_report_v0",
        "checkpoint": str(args.checkpoint),
        "datasets": [str(path) for path in args.datasets],
        "device": str(device),
        "sequence_count": len(sequences),
        "row_count": sum(len(sequence) for sequence in sequences.values()),
        "evaluations_by_margin": evaluations,
        "claim_boundary": (
            "Offline margin sweep for a direct target-selection gate over a "
            "frozen Go2 JEPA-style latent substrate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_key, best_value = max(
        evaluations.items(),
        key=lambda item: (
            item[1]["normal_minus_best_corrupted_balanced_frame_accuracy"],
            item[1]["ablations"]["normal"]["overall"]["balanced_frame_accuracy"],
        ),
    )
    best_normal = best_value["ablations"]["normal"]["overall"]
    print(
        "go2_frozen_jepa_target_gate_eval:"
        f" report={args.out}"
        f" best_margin={best_key}"
        f" frame_bal={best_normal['balanced_frame_accuracy']:.3f}"
        f" recall={best_normal['positive_frame_recall']:.3f}"
        f" abstain={best_normal['negative_frame_abstain_specificity']:.3f}"
        f" precision={best_normal['target_selection_precision']:.3f}"
        f" delta={best_value['normal_minus_best_corrupted_balanced_frame_accuracy']:.3f}"
    )
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "lewm_go2_frozen_jepa_target_gate_checkpoint_v0":
        raise SystemExit(f"unsupported checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
