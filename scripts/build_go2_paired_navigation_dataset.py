#!/usr/bin/env python3
"""Build G2 scene-disjoint RGB navigation pairs and privileged BEV targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lewm.datasets.go2_paired_navigation import (
    build_paired_navigation_dataset,
    load_scene_id_exclusions,
    load_source_index,
)
from lewm.planning.geometry_contract import (
    DEFAULT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)


def _labeled_commitment(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-index",
        type=Path,
        required=True,
        help="JSONL of explicit scene/manifest/plan/RGB paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-contract", type=Path, default=DEFAULT_GEOMETRY_CONTRACT
    )
    parser.add_argument(
        "--exclude-scene-id-commitments",
        action="append",
        type=_labeled_commitment,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Repeatable labeled exclusion commitment. PATH may be a "
            "newline-delimited SHA256(scene_id) set or a "
            "lewm_navigation_hashed_scene_roles_v1 JSON artifact; structured "
            "artifacts expand to LABEL.development and LABEL.sealed_test."
        ),
    )
    parser.add_argument(
        "--v3-development-scene-hashes",
        type=Path,
        default=None,
        help=(
            "Compatibility alias for "
            "--exclude-scene-id-commitments v3_development=PATH"
        ),
    )
    parser.add_argument(
        "--v3-sealed-scene-hashes",
        type=Path,
        default=None,
        help=(
            "Compatibility alias for "
            "--exclude-scene-id-commitments v3_sealed=PATH"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", default="go2_paired_navigation_v1")
    parser.add_argument(
        "--role-scenes-per-family",
        type=int,
        default=None,
        help=(
            "Use a label-independent direct role contract: reserve this many "
            "scenes per family for each of checkpoint selection, probability "
            "calibration, and untouched G2; all remaining scenes train."
        ),
    )
    parser.add_argument(
        "--allow-role-transition-shortfall",
        action="store_true",
        help=(
            "Diagnostic only: permit a direct-role scene to yield fewer rows "
            "than --max-transitions-per-scene."
        ),
    )
    parser.add_argument("--max-transitions-per-scene", type=int, default=512)
    parser.add_argument(
        "--selection-seed", default="go2_paired_navigation_selection_v1"
    )
    args = parser.parse_args()

    contract = load_geometry_contract(args.geometry_contract)
    legacy_paths = (
        args.v3_development_scene_hashes,
        args.v3_sealed_scene_hashes,
    )
    if (legacy_paths[0] is None) != (legacy_paths[1] is None):
        parser.error(
            "the two --v3-*-scene-hashes compatibility aliases must be passed together"
        )
    commitment_files = list(args.exclude_scene_id_commitments)
    if legacy_paths[0] is not None:
        commitment_files.extend(
            (
                ("v3_development", legacy_paths[0]),
                ("v3_sealed", legacy_paths[1]),
            )
        )
    if not commitment_files:
        parser.error("at least one held-out scene-ID commitment is required")
    exclusions = load_scene_id_exclusions(commitment_files)
    result = build_paired_navigation_dataset(
        sources=load_source_index(args.source_index),
        output_dir=args.output_dir,
        geometry_contract=contract,
        scene_exclusions=exclusions,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        role_scenes_per_family=args.role_scenes_per_family,
        allow_role_transition_shortfall=args.allow_role_transition_shortfall,
        max_transitions_per_scene=args.max_transitions_per_scene,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
