#!/usr/bin/env python3
"""Build a conservative learned-local target-policy stall patch dataset.

The output keeps the original runtime feature vectors. Labels are copied from a
baseline learned-local checkpoint everywhere except near configured stall ticks,
where the input dataset's labels are treated as offline oracle corrections.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--target-color", default="yellow")
    parser.add_argument("--policy-feature-slot", default="target:yellow")
    parser.add_argument(
        "--stall-ticks",
        default="",
        help="Comma-separated source-run ticks to replace with oracle labels.",
    )
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--correction-repeat", type=int, default=8)
    parser.add_argument("--baseline-repeat", type=int, default=1)
    parser.add_argument(
        "--label-overrides",
        default="",
        help=(
            "Optional comma-separated tick=primitive overrides for correction rows. "
            "This is for offline counterfactual patch datasets; non-correction rows "
            "still use the baseline checkpoint labels."
        ),
    )
    parser.add_argument("--clock-max-ticks", type=float, default=560.0)
    parser.add_argument("--online-map-size", type=int, default=21)
    parser.add_argument("--online-map-cell-m", type=float, default=0.45)
    parser.add_argument("--online-map-stall-displacement-m", type=float, default=0.015)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    import train_go2_closed_loop_learned_local_policy as train_policy

    with np.load(args.dataset, allow_pickle=False) as data:
        schema = str(data["schema"][0]) if "schema" in data else ""
        if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
            raise SystemExit(f"unsupported dataset schema in {args.dataset}: {schema}")
        features = np.asarray(data["features"], dtype=np.float32)
        oracle_labels = np.asarray(data["labels"], dtype=np.int64)
        primitive_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
        meta_raw = np.asarray(data["meta_json"]).astype(str)
        result_json = np.asarray(data["result_json"]).astype(str) if "result_json" in data else np.asarray([], dtype=str)
        pred_features = np.asarray(data["features"], dtype=np.float32)
        pred_features = train_policy._append_clock_features(
            data,
            pred_features,
            clock_max_ticks=float(args.clock_max_ticks),
        )
        pred_features = train_policy._append_visual_readout_features(data, pred_features)
        pred_features = train_policy._append_state_features(data, pred_features)
        pred_features = train_policy._append_online_map_features(
            data,
            pred_features,
            map_size=int(args.online_map_size),
            cell_m=float(args.online_map_cell_m),
            stall_displacement_m=float(args.online_map_stall_displacement_m),
        )

    if features.ndim != 2:
        raise SystemExit(f"features must be rank-2, got {features.shape}")
    if len(meta_raw) != int(features.shape[0]) or int(oracle_labels.shape[0]) != len(meta_raw):
        raise SystemExit("feature, label, and metadata row counts differ")

    checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_vocab = [str(item) for item in checkpoint.get("primitive_vocab", [])]
    if checkpoint_vocab != primitive_vocab:
        raise SystemExit("baseline checkpoint primitive vocabulary does not match dataset")
    if int(pred_features.shape[1]) != int(checkpoint.get("input_dim", -1)):
        raise SystemExit(
            f"baseline input_dim={checkpoint.get('input_dim')} but dataset features after append "
            f"have dim={pred_features.shape[1]}"
        )
    model_type = str(checkpoint.get("model_type", "mlp"))
    if model_type == "map_cnn":
        model = train_policy.LearnedLocalMapCnnPolicyHead(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            len(primitive_vocab),
            map_size=int(checkpoint.get("online_map_size", args.online_map_size)),
        )
    elif model_type == "mlp":
        model = train_policy.LearnedLocalPolicyHead(
            int(checkpoint["input_dim"]),
            int(checkpoint["hidden_dim"]),
            len(primitive_vocab),
        )
    else:
        raise SystemExit(f"unsupported baseline model type for row predictions: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(pred_features).float())
        baseline_labels = logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    stall_ticks = {
        int(item.strip())
        for item in str(args.stall_ticks).split(",")
        if item.strip()
    }
    label_overrides: dict[int, int] = {}
    for item in str(args.label_overrides).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid --label-overrides item {item!r}, expected tick=primitive")
        tick_text, primitive = item.split("=", 1)
        tick_key = int(tick_text.strip())
        primitive = primitive.strip()
        if primitive not in primitive_vocab:
            raise SystemExit(
                f"invalid override primitive {primitive!r}; expected one of {primitive_vocab}"
            )
        label_overrides[tick_key] = int(primitive_vocab.index(primitive))
    window = max(0, int(args.window))
    selected_indices: list[int] = []
    selected_labels: list[int] = []
    selected_meta: list[str] = []
    counts = Counter()
    baseline_repeat = max(0, int(args.baseline_repeat))
    correction_repeat = max(1, int(args.correction_repeat))
    for idx, raw in enumerate(meta_raw.tolist()):
        meta = json.loads(str(raw))
        if str(meta.get("target_color", "")) != str(args.target_color):
            counts["skip_target_color"] += 1
            continue
        if str(meta.get("policy_feature_slot", "")) != str(args.policy_feature_slot):
            counts["skip_policy_feature_slot"] += 1
            continue
        tick = int(meta.get("tick", -10**9))
        is_correction = bool(stall_ticks) and any(abs(tick - t) <= window for t in stall_ticks)
        repeats = correction_repeat if is_correction else baseline_repeat
        if repeats <= 0:
            continue
        label = int(oracle_labels[idx]) if is_correction else int(baseline_labels[idx])
        override_label = label_overrides.get(tick)
        if is_correction and override_label is not None:
            label = int(override_label)
        for rep in range(repeats):
            patched_meta = dict(meta)
            patched_meta["stall_patch_dataset"] = True
            patched_meta["stall_patch_source_index"] = int(idx)
            patched_meta["stall_patch_repeat_index"] = int(rep)
            patched_meta["stall_patch_label_source"] = "oracle" if is_correction else "baseline_policy"
            if is_correction and override_label is not None:
                patched_meta["stall_patch_label_source"] = "override"
            patched_meta["stall_patch_baseline_label"] = primitive_vocab[int(baseline_labels[idx])]
            patched_meta["stall_patch_oracle_label"] = primitive_vocab[int(oracle_labels[idx])]
            patched_meta["label"] = primitive_vocab[label]
            selected_indices.append(idx)
            selected_labels.append(label)
            selected_meta.append(json.dumps(patched_meta, sort_keys=True))
        if is_correction:
            counts["correction_rows"] += 1
            counts[f"correction_label:{primitive_vocab[label]}"] += 1
        else:
            counts["baseline_rows"] += 1
            counts[f"baseline_label:{primitive_vocab[label]}"] += 1

    if not selected_indices:
        raise SystemExit("no rows selected for patch dataset")

    out_features = features[np.asarray(selected_indices, dtype=np.int64)]
    out_labels = np.asarray(selected_labels, dtype=np.int64)
    out_meta = np.asarray(selected_meta, dtype=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray(["lewm_go2_closed_loop_learned_local_policy_dataset_v0"]),
        features=out_features.astype(np.float32, copy=False),
        labels=out_labels,
        primitive_vocab=np.asarray(primitive_vocab, dtype=str),
        meta_json=out_meta,
        result_json=result_json,
    )

    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "output": str(args.output),
        "target_color": str(args.target_color),
        "policy_feature_slot": str(args.policy_feature_slot),
        "stall_ticks": sorted(stall_ticks),
        "window": int(window),
        "baseline_repeat": int(baseline_repeat),
        "correction_repeat": int(correction_repeat),
        "label_overrides": {
            str(tick): primitive_vocab[int(label)]
            for tick, label in sorted(label_overrides.items())
        },
        "source_rows": int(features.shape[0]),
        "selected_unique_rows": int(len(set(selected_indices))),
        "output_rows": int(out_labels.shape[0]),
        "counts": dict(counts),
        "output_label_counts": {
            primitive_vocab[int(idx)]: int(count)
            for idx, count in sorted(Counter(out_labels.tolist()).items())
        },
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
