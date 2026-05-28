#!/usr/bin/env python3
"""Aggregate corpus-level audit for the JEPA / H-JEPA pipeline.

Walks one or more roots produced by ``run_mass_datagen.sh`` (each root
containing ``rollout/<scene>/summary.json``, ``raw/<scene>/...``,
``labels/<scene>/labels.jsonl`` + ``summary.json``,
``render/<scene>/...``) and emits the histograms listed in the
mass-datagen P2 section:

- effective LeWM sequence counts (per env, per scene) after windowing
- per-family, per-primitive, per-command histograms
- local-graph-type coverage per family
- goal-image observations per (landmark, yaw_bin) bin
- strict-vs-relaxed claim split per family

Same-place positive / hard-negative / loop-closure pair counts are
deliberately deferred to a separate pass — they need a richer join
between landmark visibility and pose graph, beyond what the per-row
labels expose.

Writes the audit to ``<out>/jepa_corpus_audit.json``. Exits non-zero
on schema violations so it can be a CI gate after a full datagen run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more mass-datagen output roots (containing rollout/, raw/, labels/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".generated/audits/jepa_corpus_audit.json"),
    )
    parser.add_argument(
        "--window-ticks",
        type=int,
        default=16,
        help="History window size used for sequence counting (default 16).",
    )
    args = parser.parse_args()

    audit: dict[str, Any] = {
        "scene_count": 0,
        "label_row_count": 0,
        "effective_sequence_count": 0,
        "per_family": defaultdict(lambda: defaultdict(int)),
        "per_primitive": Counter(),
        "per_command_source": Counter(),
        "local_graph_type_counts": Counter(),
        "per_family_local_graph_counts": defaultdict(Counter),
        "goal_image_landmark_yaw_bins": defaultdict(int),
        "relaxed_claim_counts_by_family": defaultdict(int),
        "achieved_landmark_count": 0,
        "available_landmark_count": 0,
        # Per-collector marginal-isotropy aggregates. See
        # docs/lejepa_identifiability_analysis.md for what each field is
        # for; primitives are pooled across scenes so the entropy reflects
        # the corpus-wide marginal, while the yaw entropy is averaged over
        # per-scene means (each scene's mean is itself an average over
        # visited cells).
        "marginal_isotropy_by_source": defaultdict(
            lambda: {
                "blocks": 0,
                "distinct_cell_yaw_bins_sum": 0,
                "primitive_counts": Counter(),
                "per_scene_mean_yaw_entropy_nats": [],
                "max_per_cell_yaw_entropy_nats": 0.0,
            }
        ),
    }

    schema_failures: list[str] = []
    for root in args.roots:
        if not root.exists():
            print(f"missing root: {root}", file=sys.stderr)
            return 2
        rollout_root = root / "rollout"
        labels_root = root / "labels"
        if not rollout_root.is_dir() or not labels_root.is_dir():
            schema_failures.append(f"{root}: missing rollout/ or labels/")
            continue
        for scene_dir in sorted(rollout_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            rollout_summary_path = scene_dir / "summary.json"
            if not rollout_summary_path.is_file():
                schema_failures.append(f"{scene_id}: rollout summary missing")
                continue
            rollout_summary = json.loads(rollout_summary_path.read_text())
            family = (
                rollout_summary.get("family")
                or rollout_summary.get("scene_family")
                or "unknown"
            )
            stats = (
                rollout_summary.get("extra", {})
                .get("rollout_stats", {})
            )
            per_env = stats.get("per_env_metrics", [])
            for env_metric in per_env:
                audit["per_family"][family]["scene_envs"] += 1
                audit["achieved_landmark_count"] += int(
                    env_metric.get("beacons_achieved", 0)
                )
                audit["available_landmark_count"] += int(
                    env_metric.get("beacons_available", 0)
                )
                audit["relaxed_claim_counts_by_family"][family] += int(
                    env_metric.get("relaxed_claim_count", 0)
                )
                for prim, count in (env_metric.get("primitive_counts") or {}).items():
                    audit["per_primitive"][prim] += int(count)
                for src, count in (env_metric.get("command_source_counts") or {}).items():
                    audit["per_command_source"][src] += int(count)
            audit["scene_count"] += 1
            audit["per_family"][family]["scenes"] += 1

            for source, iso in (stats.get("marginal_isotropy") or {}).items():
                agg = audit["marginal_isotropy_by_source"][source]
                agg["blocks"] += int(iso.get("blocks", 0))
                agg["distinct_cell_yaw_bins_sum"] += int(
                    iso.get("distinct_cell_yaw_bins", 0)
                )
                for prim, count in (iso.get("primitive_counts") or {}).items():
                    agg["primitive_counts"][prim] += int(count)
                mean_h = float(iso.get("mean_per_cell_yaw_entropy_nats", 0.0))
                # Skip scenes where the collector saw zero cells — they
                # contribute no signal and would bias the mean toward 0.
                if iso.get("distinct_cells", 0) > 0:
                    agg["per_scene_mean_yaw_entropy_nats"].append(mean_h)
                agg["max_per_cell_yaw_entropy_nats"] = max(
                    agg["max_per_cell_yaw_entropy_nats"],
                    float(iso.get("max_per_cell_yaw_entropy_nats", 0.0)),
                )

            labels_path = labels_root / scene_id / "labels.jsonl"
            label_summary_path = labels_root / scene_id / "summary.json"
            if not labels_path.is_file() or not label_summary_path.is_file():
                schema_failures.append(f"{scene_id}: labels missing")
                continue
            label_summary = json.loads(label_summary_path.read_text())
            audit["label_row_count"] += int(label_summary.get("label_count", 0))
            # Effective sequence count = labels with at least ``window_ticks``
            # in-episode predecessors. We approximate that here by
            # counting label rows whose ``episode_step >= window_ticks``.
            with labels_path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if int(row.get("episode_step", 0)) >= int(args.window_ticks):
                        audit["effective_sequence_count"] += 1
                    lg = row.get("local_graph_type") or "unknown"
                    audit["local_graph_type_counts"][lg] += 1
                    audit["per_family_local_graph_counts"][family][lg] += 1
                    yaw_bin = row.get("yaw_bin")
                    for lm in row.get("landmarks", ()):
                        if lm.get("visible") and yaw_bin is not None:
                            bucket = f"{lm.get('object_id')}|yaw_{yaw_bin}"
                            audit["goal_image_landmark_yaw_bins"][bucket] += 1

    audit["per_family"] = {k: dict(v) for k, v in audit["per_family"].items()}
    audit["per_primitive"] = dict(audit["per_primitive"])
    audit["per_command_source"] = dict(audit["per_command_source"])
    audit["local_graph_type_counts"] = dict(audit["local_graph_type_counts"])
    audit["per_family_local_graph_counts"] = {
        k: dict(v) for k, v in audit["per_family_local_graph_counts"].items()
    }
    audit["goal_image_landmark_yaw_bins"] = dict(audit["goal_image_landmark_yaw_bins"])
    audit["relaxed_claim_counts_by_family"] = dict(audit["relaxed_claim_counts_by_family"])

    finalized_iso: dict[str, dict[str, Any]] = {}
    import math as _math

    for source, agg in audit["marginal_isotropy_by_source"].items():
        prim_total = float(sum(agg["primitive_counts"].values()))
        if prim_total > 0.0:
            ps = [c / prim_total for c in agg["primitive_counts"].values() if c > 0]
            prim_entropy = -sum(p * _math.log(p) for p in ps)
        else:
            prim_entropy = 0.0
        per_scene = agg["per_scene_mean_yaw_entropy_nats"]
        mean_yaw_entropy = (
            sum(per_scene) / len(per_scene) if per_scene else 0.0
        )
        finalized_iso[source] = {
            "blocks": int(agg["blocks"]),
            "distinct_cell_yaw_bins_sum": int(agg["distinct_cell_yaw_bins_sum"]),
            "primitive_entropy_nats": float(prim_entropy),
            "mean_per_cell_yaw_entropy_nats": float(mean_yaw_entropy),
            "max_per_cell_yaw_entropy_nats": float(agg["max_per_cell_yaw_entropy_nats"]),
            "scene_count": len(per_scene),
            "primitive_counts": dict(agg["primitive_counts"]),
        }
    audit["marginal_isotropy_by_source"] = finalized_iso
    audit["schema_violations"] = schema_failures

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "scene_count": audit["scene_count"],
                "label_row_count": audit["label_row_count"],
                "effective_sequence_count": audit["effective_sequence_count"],
                "schema_violations": len(schema_failures),
            }
        )
    )
    return 1 if schema_failures else 0


if __name__ == "__main__":
    sys.exit(main())
