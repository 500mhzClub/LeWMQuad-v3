#!/usr/bin/env python3
"""Build a nonparametric learned-local Go2 primitive policy from safe features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument(
        "--tail-feature-count",
        type=int,
        default=48,
        help="Number of final feature columns to upweight. For pose_topology_v1 this covers target/claim/action/outcome/clock/pose.",
    )
    parser.add_argument("--tail-feature-weight", type=float, default=4.0)
    parser.add_argument(
        "--pose-clock-feature-weight",
        type=float,
        default=8.0,
        help="Additional weight for the final 8 clock/pose columns in pose_topology_v1 checkpoints.",
    )
    parser.add_argument("--max-prototypes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    primitive_vocab: list[str] | None = None
    feature_variant = "base"
    dataset_reports: list[dict] = []
    for path in args.datasets:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"][0]) if "schema" in data else ""
            if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
                raise SystemExit(f"unsupported dataset schema in {path}: {schema}")
            current_vocab = [str(item) for item in data["primitive_vocab"].tolist()]
            if primitive_vocab is None:
                primitive_vocab = current_vocab
            elif current_vocab != primitive_vocab:
                raise SystemExit(f"primitive vocab mismatch in {path}")
            current_features = np.asarray(data["features"], dtype=np.float32)
            current_labels = np.asarray(data["labels"], dtype=np.int64)
            if current_features.shape[0] != current_labels.shape[0]:
                raise SystemExit(f"feature/label length mismatch in {path}")
            features.append(current_features)
            labels.append(current_labels)
            if "result_json" in data and len(data["result_json"]) > 0:
                report = json.loads(str(data["result_json"][0]))
                dataset_reports.append(report)
                metrics = report.get("wall_metrics", {})
                if isinstance(metrics, dict):
                    feature_variant = str(
                        metrics.get("learned_local_policy_feature_variant", feature_variant)
                    )

    x = np.concatenate(features, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    if x.shape[0] == 0:
        raise SystemExit("no prototypes")
    if args.max_prototypes and int(args.max_prototypes) < x.shape[0]:
        keep = rng.choice(x.shape[0], size=int(args.max_prototypes), replace=False)
        keep.sort()
        x = x[keep]
        y = y[keep]

    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-5).astype(np.float32)
    x_norm = ((x - mean) / scale).astype(np.float32)
    feature_weights = np.ones((x.shape[1],), dtype=np.float32)
    tail_count = min(max(0, int(args.tail_feature_count)), x.shape[1])
    if tail_count:
        feature_weights[-tail_count:] *= float(args.tail_feature_weight)
    if feature_variant == "pose_topology_v1" and x.shape[1] >= 8:
        feature_weights[-8:] *= float(args.pose_clock_feature_weight)

    checkpoint = {
        "schema": "lewm_go2_closed_loop_learned_local_policy_v0",
        "model_type": "knn",
        "input_dim": int(x.shape[1]),
        "feature_variant": feature_variant,
        "primitive_vocab": list(primitive_vocab or []),
        "prototype_features": torch.from_numpy(x_norm),
        "prototype_labels": torch.from_numpy(y),
        "feature_mean": torch.from_numpy(mean),
        "feature_scale": torch.from_numpy(scale),
        "feature_weights": torch.from_numpy(feature_weights),
        "k": int(args.k),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    label_counts = Counter(int(item) for item in y.tolist())
    report = {
        "schema": "lewm_go2_closed_loop_learned_local_knn_policy_report_v0",
        "output": str(args.output),
        "datasets": [str(path) for path in args.datasets],
        "dataset_reports": dataset_reports,
        "model_type": "knn",
        "input_dim": int(x.shape[1]),
        "feature_variant": feature_variant,
        "primitive_vocab": list(primitive_vocab or []),
        "prototype_count": int(x.shape[0]),
        "k": int(args.k),
        "tail_feature_count": int(tail_count),
        "tail_feature_weight": float(args.tail_feature_weight),
        "pose_clock_feature_weight": float(args.pose_clock_feature_weight),
        "label_counts": {
            str((primitive_vocab or [])[idx]): int(label_counts.get(idx, 0))
            for idx in range(len(primitive_vocab or []))
        },
        "claim_boundary": (
            "Nonparametric learned-local policy over runtime-safe feature exemplars. "
            "Oracle or route scaffolds used to create labels are offline-only."
        ),
    }
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"learned_local_knn_policy: output={args.output} "
        f"prototypes={x.shape[0]} k={int(args.k)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
