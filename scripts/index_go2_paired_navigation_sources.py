#!/usr/bin/env python3
"""Build a provenance-checked source index for Go2 paired navigation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lewm.datasets.go2_navigation_source_index import build_navigation_source_index


def _labeled_commitment(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-root",
        type=Path,
        default=Path(".generated/datagen_full/render_textured_v03"),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path(".generated/datagen_full/rollout"),
    )
    parser.add_argument(
        "--scene-corpus-root",
        type=Path,
        default=Path(".generated/scene_corpus"),
    )
    parser.add_argument(
        "--exclude-scene-id-commitments",
        action="append",
        type=_labeled_commitment,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Repeatable labeled newline-delimited SHA-256(scene_id) set. "
            "Pass commitments, never a raw benchmark split file."
        ),
    )
    parser.add_argument(
        "--development-scene-id-commitments",
        type=Path,
        default=None,
        help="Compatibility alias labeled v3_development.",
    )
    parser.add_argument(
        "--sealed-scene-id-commitments",
        type=Path,
        default=None,
        help="Compatibility alias labeled v3_sealed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--family", action="append", default=[], help="Optional exact family filter."
    )
    parser.add_argument(
        "--split", action="append", default=[], help="Optional exact source split filter."
    )
    parser.add_argument(
        "--max-scenes-per-family",
        type=int,
        default=None,
        help="Hash-rank and deeply validate at most this many scenes per family.",
    )
    parser.add_argument(
        "--selection-seed",
        default="go2_navigation_source_index_v1",
        help="Stable seed for per-family SHA-256 source selection.",
    )
    parser.add_argument(
        "--require-zero-rejections",
        action="store_true",
        help="Exit unsuccessfully after writing the report when any source is rejected.",
    )
    args = parser.parse_args()

    result = build_navigation_source_index(
        render_root=args.render_root,
        rollout_root=args.rollout_root,
        scene_corpus_root=args.scene_corpus_root,
        output_dir=args.output_dir,
        exclusion_commitment_files=args.exclude_scene_id_commitments,
        development_commitments_path=args.development_scene_id_commitments,
        sealed_commitments_path=args.sealed_scene_id_commitments,
        families=args.family,
        splits=args.split,
        max_scenes_per_family=args.max_scenes_per_family,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_zero_rejections and int(result["rejected"]) != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
