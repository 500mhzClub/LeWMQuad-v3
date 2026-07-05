#!/usr/bin/env python3
"""Evaluate one frozen Phase 2 checkpoint on named complete-valid datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.benchmarks.experiment_manifest import artifact_record, write_json  # noqa: E402
from lewm.models.lewm import LeWorldModel  # noqa: E402
from lewm.models.spatial_lewm import SpatialLeWorldModel  # noqa: E402
from train_jepa_pooled_lewm_control import (  # noqa: E402
    _evaluate as evaluate_pooled,
    _pooled_context_length,
)
from train_jepa_spatial_lewm import _evaluate as evaluate_spatial  # noqa: E402
from train_jepa_spatial_predictor import _load_rows  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("dataset must use NAME=PATH")
    return name, Path(path)


def _checkpoint_kind(payload: dict) -> str:
    phase = str(payload["report"]["phase"])
    if "pooled" in phase:
        return "pooled"
    if "spatial" in phase:
        return "spatial"
    raise ValueError(f"unsupported Phase 2 checkpoint phase: {phase}")


def _value(args: dict, name: str, default):
    value = args.get(name, default)
    return default if value is None else value


def _build_model(payload: dict, rows: list[dict], device: torch.device):
    args = payload["args"]
    cmd_dim = len(rows[0]["active_blocks"][0])
    kind = _checkpoint_kind(payload)
    if kind == "spatial":
        ema_momentum = float(_value(args, "target_ema_momentum", 0.0))
        model = SpatialLeWorldModel(
            latent_dim=int(_value(args, "latent_dim", 192)),
            cmd_dim=cmd_dim,
            pred_layers=int(_value(args, "pred_layers", 6)),
            pred_heads=int(_value(args, "pred_heads", 16)),
            pred_dim_head=int(_value(args, "pred_dim_head", 64)),
            pred_mlp_dim=int(_value(args, "pred_mlp_dim", 2048)),
            encoder_depth=int(_value(args, "encoder_depth", 12)),
            encoder_heads=int(_value(args, "encoder_heads", 3)),
            encoder_mlp_ratio=int(_value(args, "encoder_mlp_ratio", 4)),
            appearance_sigreg_lambda=float(
                _value(args, "appearance_sigreg_lambda", 0.09)
            ),
            spatial_variance_lambda=float(
                _value(args, "spatial_variance_lambda", 1.0)
            ),
            spatial_target_std=float(_value(args, "spatial_target_std", 1.0)),
            sigreg_projections=int(_value(args, "sigreg_projections", 1024)),
            sigreg_knots=int(_value(args, "sigreg_knots", 17)),
            target_ema_momentum=ema_momentum if ema_momentum > 0.0 else None,
        )
        evaluate = evaluate_spatial
    else:
        model = LeWorldModel(
            latent_dim=int(_value(args, "latent_dim", 192)),
            cmd_dim=cmd_dim,
            pred_layers=int(_value(args, "pred_layers", 6)),
            pred_heads=int(_value(args, "pred_heads", 16)),
            pred_dim_head=int(_value(args, "pred_dim_head", 64)),
            pred_mlp_dim=int(_value(args, "pred_mlp_dim", 2048)),
            max_seq_len=_pooled_context_length(rows[0]),
            sigreg_lambda=float(_value(args, "sigreg_lambda", 0.09)),
            sigreg_projections=int(_value(args, "sigreg_projections", 1024)),
            sigreg_knots=int(_value(args, "sigreg_knots", 17)),
            encoder_depth=int(_value(args, "encoder_depth", 12)),
            encoder_heads=int(_value(args, "encoder_heads", 3)),
            encoder_mlp_ratio=int(_value(args, "encoder_mlp_ratio", 4)),
        )
        evaluate = evaluate_pooled
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return kind, model.to(device), evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=_named_path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(dict(args.dataset)) != len(args.dataset):
        raise SystemExit("dataset names must be unique")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    datasets = {}
    for name, path in args.dataset:
        rows, load_audit = _load_rows(path, args.max_rows)
        datasets[name] = {"path": path, "rows": rows, "load_audit": load_audit}

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    first = next(iter(datasets.values()))
    kind, model, evaluate = _build_model(payload, first["rows"], device)
    evaluations = {}
    for name, dataset in datasets.items():
        evaluations[name] = {
            "file": artifact_record(dataset["path"]),
            "load_audit": dataset["load_audit"],
            "metrics": evaluate(
                model,
                dataset["rows"],
                batch_size=args.batch_size,
                device=device,
            ),
        }
    report = {
        "schema": "jepa_phase2_frozen_checkpoint_diagnostic_v0",
        "checkpoint": artifact_record(args.checkpoint),
        "checkpoint_phase": payload["report"]["phase"],
        "checkpoint_kind": kind,
        "device": str(device),
        "datasets": evaluations,
        "limitations": [
            "diagnostic uses the historical complete-valid row contract",
            "legacy shuffled-action metrics retain the historical batch-roll control",
            "result is post-hoc diagnostic evidence and cannot retroactively pass a gate",
        ],
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "checkpoint": str(args.checkpoint.resolve()),
                "output": str(args.output.resolve()),
                "datasets": {
                    name: {
                        "step1": result["metrics"]["per_horizon_step"][0],
                        "step2": result["metrics"]["per_horizon_step"][1],
                    }
                    for name, result in evaluations.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
