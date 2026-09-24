"""Audit canonical exact occupancy through the shared online belief map.

The command is development-only. It consumes only ``validation_scenes`` from
an explicit development manifest, loads privileged occupancy through the
standalone exact adapter, and exercises routes solely through
``OnlineBeliefMap.shortest_path``. It never imports the closed-loop benchmark
monolith or any learned model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for _source_root in (REPO_ROOT / "lewm_worlds",):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from lewm.planning.exact_occupancy_belief_adapter import (  # noqa: E402
    ExactOccupancyBeliefAdapter,
)
from lewm.planning.geometry_contract import (  # noqa: E402
    DEFAULT_GEOMETRY_CONTRACT,
    load_geometry_contract,
)
from lewm_worlds.manifest import (  # noqa: E402
    manifest_sha256,
    parse_scene_manifest_dict,
)


DEFAULT_DEVELOPMENT_MANIFEST = (
    REPO_ROOT / "config/go2_generalization_v3/development.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".generated/oracle_positive_control/generalization_v3_development"
    / "shared_map_report.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_path_guard(path: Path, *, label: str) -> None:
    lowered = "/".join(part.lower() for part in path.parts)
    forbidden = ("sealed", "final_eval", "final-test", "final_test")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{label} must be development-only, got {path}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--scene-corpus", type=Path, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=REPO_ROOT / DEFAULT_GEOMETRY_CONTRACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run_development_audit(
    *,
    development_manifest: Path,
    scene_corpus: Path,
    geometry_contract_path: Path,
    requested_scene_ids: Sequence[str] = (),
) -> dict[str, Any]:
    _development_path_guard(development_manifest, label="development manifest")
    _development_path_guard(scene_corpus, label="scene corpus")
    protocol = json.loads(development_manifest.read_text(encoding="utf-8"))
    if protocol.get("schema") != "lewm_navigation_development_manifest_v0":
        raise ValueError(
            f"unsupported development manifest schema: {protocol.get('schema')!r}"
        )
    geometry = load_geometry_contract(
        geometry_contract_path,
        repository_root=REPO_ROOT,
    )
    if str(protocol.get("geometry_contract_sha256")) != geometry.sha256:
        raise ValueError("development manifest geometry-contract SHA mismatch")
    validation = list(protocol.get("validation_scenes", ()))
    if not validation:
        raise ValueError("development manifest has no validation_scenes")
    by_id = {str(record["scene_id"]): record for record in validation}
    if len(by_id) != len(validation):
        raise ValueError("development validation_scenes contain duplicate scene ids")
    selected_ids = list(requested_scene_ids) if requested_scene_ids else list(by_id)
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(
            "scene ids are not development validation scenes: " + ", ".join(unknown)
        )

    scene_reports: list[dict[str, Any]] = []
    for scene_id in selected_ids:
        record = by_id[scene_id]
        if not bool(record.get("fully_reachable")) or str(
            record.get("failure_reason", "")
        ):
            raise ValueError(f"development scene is not geometry-valid: {scene_id}")
        family = str(record["family"])
        scene_dir = scene_corpus / "development" / family / scene_id
        manifest_path = scene_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = parse_scene_manifest_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        actual_manifest_sha = manifest_sha256(manifest)
        if actual_manifest_sha != str(record["manifest_sha256"]):
            raise ValueError(f"semantic manifest SHA mismatch: {scene_id}")
        if len(manifest.landmarks) != int(record["beacon_count"]):
            raise ValueError(f"beacon count mismatch: {scene_id}")

        adapter = ExactOccupancyBeliefAdapter(manifest, geometry)
        spawn_cell = adapter.spawn_cell
        partial_observed = {
            cell
            for cell in adapter.all_online_cells
            if abs(cell[0] - spawn_cell[0]) <= 8
            and abs(cell[1] - spawn_cell[1]) <= 8
        }
        partial_agreement = adapter.load(partial_observed)
        agreement = adapter.load()
        routes = [
            adapter.connected_claim_route(landmark)
            for landmark in manifest.landmarks
        ]
        connected_routes = [route for route in routes if route is not None]
        scene_success = bool(
            agreement.online_topology_agrees
            and agreement.resolution_is_conservative
            and partial_agreement.online_topology_agrees
            and partial_agreement.map_frontier_cells > 0
            and len(connected_routes) == len(manifest.landmarks)
        )
        scene_report = {
            "scene_id": scene_id,
            "family": family,
            "manifest_sha256": actual_manifest_sha,
            "success": scene_success,
            "agreement": agreement.to_dict(),
            "partial_frontier_probe": partial_agreement.to_dict(),
            "beacon_count": len(manifest.landmarks),
            "connected_claim_anchors": len(connected_routes),
            "missing_claim_anchor_ids": [
                str(landmark.object_id)
                for landmark, route in zip(manifest.landmarks, routes)
                if route is None
            ],
            "claim_routes": [
                route.to_dict() for route in connected_routes
            ],
        }
        scene_reports.append(scene_report)
        print(
            f"{scene_id}: component_diff="
            f"{agreement.component_symmetric_difference_cells} "
            f"frontier_diff={agreement.frontier_symmetric_difference_cells} "
            f"partial_frontiers={partial_agreement.map_frontier_cells} "
            f"partial_frontier_diff="
            f"{partial_agreement.frontier_symmetric_difference_cells} "
            f"resolution_diff={agreement.resolution_symmetric_difference_cells} "
            f"jaccard={agreement.resolution_jaccard:.6f} "
            f"anchors={len(connected_routes)}/{len(manifest.landmarks)}",
            file=sys.stderr,
            flush=True,
        )

    agreements = [report["agreement"] for report in scene_reports]
    partial_agreements = [
        report["partial_frontier_probe"] for report in scene_reports
    ]
    connected = sum(
        int(report["connected_claim_anchors"])
        for report in scene_reports
    )
    expected = sum(int(report["beacon_count"]) for report in scene_reports)
    aggregate = {
        "scene_count": len(scene_reports),
        "successful_scenes": sum(bool(report["success"]) for report in scene_reports),
        "online_topology_agreement_scenes": sum(
            bool(agreement["online_topology_agrees"])
            for agreement in agreements
        ),
        "exact_oracle_projection_agreement_scenes": sum(
            bool(agreement["oracle_projection_agrees"])
            for agreement in agreements
        ),
        "conservative_resolution_scenes": sum(
            bool(agreement["resolution_is_conservative"])
            for agreement in agreements
        ),
        "connected_claim_anchors": connected,
        "expected_claim_anchors": expected,
        "component_symmetric_difference_cells": sum(
            int(agreement["component_symmetric_difference_cells"])
            for agreement in agreements
        ),
        "frontier_symmetric_difference_cells": sum(
            int(agreement["frontier_symmetric_difference_cells"])
            for agreement in agreements
        ),
        "partial_frontier_agreement_scenes": sum(
            bool(agreement["online_topology_agrees"])
            and int(agreement["map_frontier_cells"]) > 0
            for agreement in partial_agreements
        ),
        "partial_frontier_cells": sum(
            int(agreement["map_frontier_cells"])
            for agreement in partial_agreements
        ),
        "partial_frontier_symmetric_difference_cells": sum(
            int(agreement["frontier_symmetric_difference_cells"])
            for agreement in partial_agreements
        ),
        "resolution_symmetric_difference_cells": sum(
            int(agreement["resolution_symmetric_difference_cells"])
            for agreement in agreements
        ),
        "map_only_resolution_cells": sum(
            int(agreement["map_only_resolution_cells"])
            for agreement in agreements
        ),
        "projected_oracle_only_cells": sum(
            int(agreement["projected_oracle_only_cells"])
            for agreement in agreements
        ),
        "mean_resolution_jaccard": round(
            float(np.mean([agreement["resolution_jaccard"] for agreement in agreements]))
            if agreements
            else 0.0,
            9,
        ),
        "minimum_resolution_jaccard": round(
            min(
                (float(agreement["resolution_jaccard"]) for agreement in agreements),
                default=0.0,
            ),
            9,
        ),
    }
    return {
        "schema": "lewm_go2_shared_belief_map_positive_control_v0",
        "development_only": True,
        "development_manifest": {
            "path": str(development_manifest),
            "sha256": _sha256_file(development_manifest),
            "source_key": "validation_scenes",
        },
        "scene_corpus": str(scene_corpus),
        "geometry_contract": {
            "path": str(geometry.source_path),
            "sha256": geometry.sha256,
            "status": geometry.status,
        },
        "assumptions": {
            "occupancy": "privileged exact static manifest geometry",
            "map_resolution_m": geometry.configuration_space.online_cell_size_m,
            "oracle_resolution_m": geometry.configuration_space.oracle_cell_size_m,
            "planning_connectivity": geometry.configuration_space.connectivity,
            "allow_diagonal_corner_cutting": (
                geometry.configuration_space.allow_diagonal_corner_cutting
            ),
            "routing_api": "OnlineBeliefMap.shortest_path over confirmed free only",
            "frontier_api": "OnlineBeliefMap.frontier_cells",
            "claim_endpoint": "true distance + point-geometry LOS + oracle connectivity",
            "scope": "privileged development integration control",
        },
        "aggregate": aggregate,
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    development_manifest = args.development_manifest.resolve()
    scene_corpus = (
        args.scene_corpus.resolve()
        if args.scene_corpus is not None
        else REPO_ROOT
        / ".generated/scene_corpus"
        / development_manifest.parent.name
    )
    output = args.output.resolve()
    _development_path_guard(output, label="output")
    report = run_development_audit(
        development_manifest=development_manifest,
        scene_corpus=scene_corpus,
        geometry_contract_path=args.geometry_contract.resolve(),
        requested_scene_ids=args.scene_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote shared-map positive control: {output}", file=sys.stderr)
    return 0 if report["aggregate"]["successful_scenes"] == len(report["scenes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
