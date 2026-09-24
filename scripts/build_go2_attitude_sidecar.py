#!/usr/bin/env python3
"""Build the frozen metadata-only Go2 dynamic-Cartesian attitude sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.datasets.go2_attitude_sidecar import (  # noqa: E402
    FROZEN_BUILD_CONTRACT,
    build_attitude_sidecar,
)


DATASET_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "dataset/dataset_manifest.json"
)
SOURCE_INDEX_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "source_index/go2_navigation_sources_v04.jsonl"
)
RENDER_AUDIT_PATH = (
    REPOSITORY_ROOT / ".generated/go2_render_selected_v04/audit_report.json"
)
DYNAMIC_GEOMETRY_PATH = (
    REPOSITORY_ROOT / "lewm/benchmarks/go2_dynamic_cell_square_projection.py"
)
OUTPUT_DIR = (
    REPOSITORY_ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1"
)
IMPLEMENTATION_MANIFEST_PATH = FROZEN_BUILD_CONTRACT.source_map_paths[
    "implementation_manifest"
]


def _source_map() -> dict[str, Path]:
    return dict(FROZEN_BUILD_CONTRACT.source_map_paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Bounded scene workers; must lie in [1,6], canonical merge is unchanged.",
    )
    parser.add_argument(
        "--implementation-manifest-sha256",
        required=True,
        help="Reviewed pre-output implementation-manifest file SHA-256.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6:
        parser.error("--workers must lie in [1,6]")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    manifest = build_attitude_sidecar(
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        source_index_path=SOURCE_INDEX_PATH,
        render_audit_path=RENDER_AUDIT_PATH,
        dynamic_geometry_path=DYNAMIC_GEOMETRY_PATH,
        output_dir=OUTPUT_DIR,
        source_map=_source_map(),
        implementation_manifest_path=IMPLEMENTATION_MANIFEST_PATH,
        expected_implementation_manifest_sha256=(
            args.implementation_manifest_sha256
        ),
        contract=FROZEN_BUILD_CONTRACT,
        workers=args.workers,
    )
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "implementation_manifest_sha256": (
                    args.implementation_manifest_sha256
                ),
                "content_sha256": manifest["content_sha256"],
                "role_counts": manifest["role_assignment"]["row_counts"],
                "distribution_summary_emitted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
