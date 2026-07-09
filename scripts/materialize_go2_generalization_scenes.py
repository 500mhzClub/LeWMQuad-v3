#!/usr/bin/env python3
"""Materialize only train/development scenes from a frozen benchmark manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT, REPO_ROOT / "lewm_worlds"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lewm_worlds.exporters.to_gazebo_sdf import export_gazebo_sdf  # noqa: E402
from lewm_worlds.exporters.to_genesis import export_genesis_scene  # noqa: E402
from lewm_worlds.families import build_family_manifest  # noqa: E402
from lewm_worlds.labels.topology import topology_summary  # noqa: E402
from lewm_worlds.manifest import SceneManifest, manifest_sha256  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("config/go2_generalization_v3/development.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".generated/scene_corpus/go2_generalization_v3"),
    )
    parser.add_argument(
        "--role",
        choices=("train", "development", "both"),
        default="development",
    )
    parser.add_argument("--no-genesis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


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


def manifest_from_record(record: Mapping[str, Any]) -> SceneManifest:
    """Rebuild one exact scene and reject any generator/hash drift."""

    manifest = build_family_manifest(
        scene_seed=int(record["topology_seed"]),
        family=str(record["family"]),
        split=(
            None
            if record.get("source_split") is None
            else str(record["source_split"])
        ),
        difficulty_tier=None,
    )
    if manifest.scene_id != str(record["scene_id"]):
        raise ValueError(
            f"scene ID drift: expected {record['scene_id']}, got {manifest.scene_id}"
        )
    actual_sha256 = manifest_sha256(manifest)
    if actual_sha256 != str(record["manifest_sha256"]):
        raise ValueError(
            f"manifest hash drift for {manifest.scene_id}: "
            f"expected {record['manifest_sha256']}, got {actual_sha256}"
        )
    return manifest


def _materialize_scene(
    manifest: SceneManifest,
    *,
    role: str,
    output_dir: Path,
    emit_genesis: bool,
    overwrite: bool,
) -> dict[str, Any]:
    scene_dir = output_dir / role / manifest.family / manifest.scene_id
    manifest_path = scene_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        payload = json.loads(manifest_path.read_text())
        if payload.get("manifest_sha256") != manifest_sha256(manifest):
            raise FileExistsError(
                f"existing scene has a different manifest hash: {scene_dir}"
            )
        return {
            "role": role,
            "family": manifest.family,
            "scene_id": manifest.scene_id,
            "manifest_sha256": manifest_sha256(manifest),
            "relative_dir": str(scene_dir.relative_to(output_dir)),
            "reused": True,
        }
    export_gazebo_sdf(manifest, scene_dir)
    (scene_dir / "topology.json").write_text(
        json.dumps(topology_summary(manifest), indent=2, sort_keys=True) + "\n"
    )
    if emit_genesis:
        (scene_dir / "genesis_scene.json").write_text(
            json.dumps(export_genesis_scene(manifest), indent=2, sort_keys=True)
            + "\n"
        )
    return {
        "role": role,
        "family": manifest.family,
        "scene_id": manifest.scene_id,
        "manifest_sha256": manifest_sha256(manifest),
        "relative_dir": str(scene_dir.relative_to(output_dir)),
        "reused": False,
    }


def main() -> int:
    args = _parse_args()
    development_path = _resolve(args.development_manifest)
    output_dir = _resolve(args.output_dir)
    _development_path_guard(development_path, label="development manifest")
    _development_path_guard(output_dir, label="output directory")
    payload = json.loads(development_path.read_text())
    if payload.get("schema") != "lewm_navigation_development_manifest_v0":
        raise SystemExit("unsupported development manifest schema")
    role_keys = {
        "train": ("train_scenes",),
        "development": ("validation_scenes",),
        "both": ("train_scenes", "validation_scenes"),
    }[args.role]
    output_roles = {
        "train_scenes": "train",
        "validation_scenes": "development",
    }
    results = []
    for key in role_keys:
        role = output_roles[key]
        for record in payload[key]:
            results.append(
                _materialize_scene(
                    manifest_from_record(record),
                    role=role,
                    output_dir=output_dir,
                    emit_genesis=not bool(args.no_genesis),
                    overwrite=bool(args.overwrite),
                )
            )
    summary = {
        "schema": "lewm_go2_generalization_materialization_v0",
        "benchmark_id": payload["benchmark_id"],
        "development_manifest": {
            "path": str(development_path),
            "sha256": _sha256_file(development_path),
        },
        "geometry_contract_sha256": payload["geometry_contract_sha256"],
        "roles": list(role_keys),
        "emit_genesis": not bool(args.no_genesis),
        "scene_count": len(results),
        "scenes": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"materialization_{args.role}.json"
    if summary_path.exists() and not args.overwrite:
        previous = json.loads(summary_path.read_text())
        if previous != summary:
            raise FileExistsError(summary_path)
    else:
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print(f"materialized {len(results)} {args.role} scenes under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
