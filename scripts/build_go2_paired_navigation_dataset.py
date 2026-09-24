#!/usr/bin/env python3
"""Build G2 scene-disjoint RGB navigation pairs and privileged BEV targets."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import lewm.datasets.go2_paired_navigation as paired_navigation_module
from lewm.benchmarks.experiment_manifest import build_experiment_manifest
from lewm.datasets.go2_paired_navigation import (
    LABEL_CONTRACT_CENTER_VISIBLE_V2,
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
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
            "Permit a direct-role scene to yield fewer rows than "
            "--max-transitions-per-scene. Required when the preregistered "
            "dataset contract caps rows per scene ('at most N') and the "
            "label-independent validity filter leaves fewer; the shortfall "
            "is recorded per scene in the dataset manifest."
        ),
    )
    parser.add_argument("--max-transitions-per-scene", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of scene processes. Results are committed in sorted scene "
            "order; 1 preserves serial execution."
        ),
    )
    parser.add_argument(
        "--selection-seed", default="go2_paired_navigation_selection_v1"
    )
    parser.add_argument(
        "--label-contract",
        choices=(
            LABEL_CONTRACT_CENTER_VISIBLE_V2,
            LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
        ),
        default=LABEL_CONTRACT_CENTER_VISIBLE_V2,
        help=(
            "Label semantics and artifact schema. The observable-physical "
            "contract emits dataset/row v3 from corrected v04 RGB and defers "
            "the fixed 0.47 m morphology until after online-memory fusion; "
            "v2 remains the compatibility default."
        ),
    )
    parser.add_argument(
        "--render-audit-contract",
        type=Path,
        default=None,
        help=(
            "Required for observable physical v3: immutable "
            "lewm_go2_selected_render_audit_v1 artifact binding the corrected "
            "v04 RGB campaign to --source-index."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if (
        args.label_contract == LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3
        and args.render_audit_contract is None
    ):
        parser.error(
            "--render-audit-contract is required with "
            "--label-contract observable_physical_occupancy_v3"
        )

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
    repository_root = Path(__file__).resolve().parents[1]
    resolved_arguments = {
        "source_index": str(args.source_index.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "geometry_contract": str(args.geometry_contract.resolve()),
        "exclude_scene_id_commitments": [
            {"label": label, "path": str(path.resolve())}
            for label, path in commitment_files
        ],
        "validation_fraction": float(args.validation_fraction),
        "split_seed": str(args.split_seed),
        "role_scenes_per_family": args.role_scenes_per_family,
        "allow_role_transition_shortfall": bool(
            args.allow_role_transition_shortfall
        ),
        "max_transitions_per_scene": int(args.max_transitions_per_scene),
        "workers": int(args.workers),
        "selection_seed": str(args.selection_seed),
        "label_contract": str(args.label_contract),
        "render_audit_contract": (
            str(args.render_audit_contract.resolve())
            if args.render_audit_contract is not None
            else None
        ),
    }
    provenance_inputs = {
        "builder_source": Path(__file__).resolve(),
        "dataset_source": Path(paired_navigation_module.__file__).resolve(),
        "source_index": args.source_index.resolve(),
    }
    if args.render_audit_contract is not None:
        provenance_inputs["render_audit_contract"] = (
            args.render_audit_contract.resolve()
        )
    for index, (label, path) in enumerate(commitment_files):
        provenance_inputs[f"exclusion_{index:02d}_{label}"] = path.resolve()
    build_provenance = build_experiment_manifest(
        experiment_id=f"go2_paired_navigation_{args.label_contract}",
        repository_root=repository_root,
        inputs=provenance_inputs,
        config=resolved_arguments,
        run_command=shlex.join([sys.executable, *sys.argv]),
        geometry_contract=args.geometry_contract.resolve(),
        runtime_contract={
            "label_contract": str(args.label_contract),
            "privileged_geometry_is_offline_label_only": True,
            "runtime_model_input": "rgb_only",
        },
    )
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
        label_contract=args.label_contract,
        source_index_path=args.source_index,
        render_audit_contract_path=args.render_audit_contract,
        build_provenance=build_provenance,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
