#!/usr/bin/env python3
"""Select deterministic registered-minimum source rows for Phase 2D generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import write_json  # noqa: E402
from lewm.benchmarks.phase2_data import CONFIRMATORY_SPLIT_REQUIREMENTS  # noqa: E402
from lewm.benchmarks.phase2d_readiness import canonical_split_name  # noqa: E402
from lewm.benchmarks.phase2d_source_selection import (  # noqa: E402
    select_phase2d_source_rows,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("source argument must use SPLIT=PATH")
    return canonical_split_name(name), Path(path)


def _named_int(value: str) -> tuple[str, int]:
    name, separator, count = value.partition("=")
    if not separator or not name or not count:
        raise argparse.ArgumentTypeError("override must use SPLIT=COUNT")
    return canonical_split_name(name), int(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_named_path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--scene-count", action="append", type=_named_int, default=[])
    parser.add_argument(
        "--source-states-per-scene",
        action="append",
        type=_named_int,
        default=[],
    )
    parser.add_argument(
        "--require-local-target-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-summary", type=Path)
    args = parser.parse_args()
    sources = dict(args.source)
    if set(sources) != set(CONFIRMATORY_SPLIT_REQUIREMENTS):
        parser.error(
            "sources must include exactly train, validation, test_id, and test_hard"
        )
    scene_counts = {
        split: CONFIRMATORY_SPLIT_REQUIREMENTS[split]["minimum_scenes"]
        for split in CONFIRMATORY_SPLIT_REQUIREMENTS
    }
    scene_counts.update(dict(args.scene_count))
    source_counts = {
        split: CONFIRMATORY_SPLIT_REQUIREMENTS[split][
            "minimum_source_states_per_scene"
        ]
        for split in CONFIRMATORY_SPLIT_REQUIREMENTS
    }
    source_counts.update(dict(args.source_states_per_scene))

    summaries = {}
    outputs = {}
    for split_name, source_path in sorted(sources.items()):
        output_path = args.output_dir / f"{split_name}_phase2d_sources.jsonl"
        summaries[split_name] = select_phase2d_source_rows(
            split_name=split_name,
            source_path=source_path,
            output_path=output_path,
            scene_count=scene_counts[split_name],
            source_states_per_scene=source_counts[split_name],
            seed=args.seed,
            require_local_target_frame=args.require_local_target_frame,
        )
        outputs[split_name] = str(output_path.resolve())
    report = {
        "schema": "jepa_phase2d_source_row_selection_manifest_v0",
        "seed": args.seed,
        "require_local_target_frame": bool(args.require_local_target_frame),
        "outputs": outputs,
        "splits": summaries,
        "passes_registered_minimum": all(
            summary["passes_registered_minimum"] for summary in summaries.values()
        ),
    }
    summary_path = args.output_summary or args.output_dir / "summary.json"
    write_json(summary_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passes_registered_minimum"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
