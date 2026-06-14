#!/usr/bin/env python3
"""Inspect LeWM training-window collector-source mix.

This is a CPU-only planning helper for the SIGReg/source ablation. It instantiates
the same ``GenesisWMDataset`` path as training, then reports how many valid
training windows each ``CommandBlock.command_source`` contributes at the chosen
scale and holdout split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_lewm import GenesisWMDataset  # noqa: E402


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--eval-seed", type=int, default=20260524)
    parser.add_argument("--source-allow", default="ou_noise,primitive_curriculum")
    parser.add_argument("--allow-material-color-render", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = GenesisWMDataset(
        root_dir=args.data_root,
        render_root=args.render_root,
        seq_len=args.max_seq_len,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=args.allow_material_color_render,
        holdout_fraction=args.eval_holdout_fraction,
        holdout_role="train",
        holdout_seed=args.eval_seed,
    )
    counts = Counter(dataset.window_sources)
    allowed = set(_parse_csv(args.source_allow))
    allowed_count = sum(count for source, count in counts.items() if source in allowed)
    total = len(dataset.window_sources)
    record = {
        "schema": "lewm_source_mix_v0",
        "data_root": str(args.data_root),
        "render_root": str(dataset.render_root),
        "max_seq_len": int(args.max_seq_len),
        "stride": int(args.stride),
        "max_sessions": args.max_sessions,
        "eval_holdout_fraction": float(args.eval_holdout_fraction),
        "eval_seed": int(args.eval_seed),
        "num_sessions": len(dataset.sessions),
        "total_windows": int(total),
        "source_counts": dict(counts.most_common()),
        "source_allow": sorted(allowed),
        "allowed_windows": int(allowed_count),
        "allowed_fraction": float(allowed_count / total) if total else 0.0,
        "unknown_fraction": float(counts.get("unknown", 0) / total) if total else 0.0,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
