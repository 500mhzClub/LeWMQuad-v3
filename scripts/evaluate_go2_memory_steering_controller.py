#!/usr/bin/env python3
"""Evaluate a Go2 memory steering controller checkpoint with margin sweeps."""

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
from train_go2_hidden_target_memory_probe import _load_rows, _resolve_device  # noqa: E402
from train_go2_memory_steering_controller import (  # noqa: E402
    Go2MemorySteeringController,
    _append_runtime_memory_geometry,
    _append_runtime_query_geometry,
    _evaluate,
    _row_index,
    _scrub_scene_aux,
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
    checkpoint_args = dict(checkpoint.get("args", {}))
    rows_raw = _load_rows(args.datasets)
    if not rows_raw:
        raise SystemExit("no evaluation rows")
    if bool(checkpoint.get("scrubbed_runtime_aux", False)) or bool(
        checkpoint_args.get("scrub_runtime_aux", False)
    ):
        rows = _scrub_runtime_aux(rows_raw)
    elif bool(checkpoint.get("scrubbed_command_aux", False)) or bool(
        checkpoint_args.get("scrub_command_aux", False)
    ):
        rows = _scrub_command_aux(rows_raw)
    elif bool(checkpoint.get("scrubbed_scene_aux", False)) or bool(
        checkpoint_args.get("scrub_scene_aux", False)
    ):
        rows = _scrub_scene_aux(rows_raw)
    else:
        rows = rows_raw

    feature_stats = {
        "mean": np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        "std": np.asarray(checkpoint["feature_std"], dtype=np.float32),
    }
    color_vocab = list(checkpoint["color_vocab"]) or _color_vocab(rows)
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
    row_index = _row_index(rows_raw)
    if bool(checkpoint_args.get("include_runtime_memory_geometry", False)):
        sequences = _append_runtime_memory_geometry(
            sequences,
            row_index,
            max_slot=_max_landmark_slot(rows),
        )
    if bool(checkpoint_args.get("include_runtime_query_geometry", False)):
        sequences = _append_runtime_query_geometry(sequences, row_index)
    if not sequences:
        raise SystemExit("no evaluable sequences")

    device = _resolve_device(str(args.device))
    encoder = None
    encoder_dim = None
    if checkpoint.get("frozen_jepa_checkpoint"):
        encoder, jepa_checkpoint = load_go2_jepa_encoder(
            Path(checkpoint["frozen_jepa_checkpoint"]),
            device=device,
            freeze=True,
        )
        encoder_dim = int(jepa_checkpoint.get("latent_dim", checkpoint["hidden_dim"]))
    model = Go2MemorySteeringController(
        aux_dim=int(checkpoint["aux_dim"]),
        query_dim=int(checkpoint["query_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        encoder=encoder,
        encoder_output_dim=encoder_dim,
        freeze_encoder=encoder is not None,
        memory_slot_count=int(checkpoint.get("memory_slot_count", 0)),
        temporal_memory_layers=int(checkpoint.get("temporal_memory_layers", 0)),
        temporal_memory_heads=int(checkpoint.get("temporal_memory_heads", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    margins = args.selection_margin
    if not margins:
        margins = [float(checkpoint_args.get("selection_margin", 0.0))]
    evaluations: dict[str, Any] = {}
    for margin in margins:
        ablations = {
            ablation: _evaluate(
                model,
                sequences,
                row_index,
                device=device,
                selection_margin=float(margin),
                arc_threshold_rad=float(checkpoint_args.get("arc_threshold_rad", 0.1)),
                yaw_threshold_rad=float(checkpoint_args.get("yaw_threshold_rad", 0.75)),
                hold_range_m=float(checkpoint_args.get("hold_range_m", 0.0)),
                ablation=ablation,
                include_current=not bool(checkpoint_args.get("exclusive_memory_state", False)),
            )
            for ablation in (
                "normal",
                "memory_off_abstain",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_hidden_states",
            )
        }
        normal = ablations["normal"]
        corrupted_best = max(
            float(ablations[name]["target_steering_pipeline_success"])
            for name in (
                "memory_off_abstain",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_hidden_states",
            )
        )
        evaluations[str(float(margin))] = {
            "selection_margin": float(margin),
            "ablations": ablations,
            "normal_minus_best_corrupted_target_steering_pipeline_success": (
                float(normal["target_steering_pipeline_success"]) - corrupted_best
            ),
        }

    best_key, best_value = max(
        evaluations.items(),
        key=lambda item: (
            item[1]["ablations"]["normal"]["target_steering_pipeline_success"],
            -item[1]["ablations"]["normal"]["false_claim_rate"],
            item[1]["normal_minus_best_corrupted_target_steering_pipeline_success"],
        ),
    )
    report = {
        "schema": "lewm_go2_memory_steering_controller_eval_report_v0",
        "checkpoint": str(args.checkpoint),
        "datasets": [str(path) for path in args.datasets],
        "device": str(device),
        "sequence_count": len(sequences),
        "row_count": sum(len(sequence) for sequence in sequences.values()),
        "evaluations_by_margin": evaluations,
        "best_margin_by_target_steering": best_key,
        "claim_boundary": (
            "Offline margin sweep for a Go2 memory steering controller over "
            "rendered event slices; not live Genesis execution."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_normal = best_value["ablations"]["normal"]
    print(
        "go2_memory_steering_controller_eval:"
        f" report={args.out}"
        f" best_margin={best_key}"
        f" recall={best_normal['target_recall']:.3f}"
        f" false_claim={best_normal['false_claim_rate']:.3f}"
        f" target_steer={best_normal['target_steering_pipeline_success']:.3f}"
        f" delta={best_value['normal_minus_best_corrupted_target_steering_pipeline_success']:.3f}"
    )
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "lewm_go2_memory_steering_controller_checkpoint_v0":
        raise SystemExit(f"unsupported checkpoint schema: {checkpoint.get('schema')}")
    return dict(checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
