"""Materialize a :class:`CorpusPlan` into on-disk worlds and manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from lewm_worlds.exporters.to_gazebo_sdf import export_gazebo_sdf
from lewm_worlds.exporters.to_genesis import export_genesis_scene
from lewm_worlds.labels.topology import topology_summary
from lewm_worlds.manifest import SceneManifest, build_scene_manifest, manifest_sha256
from lewm_worlds.splits import CorpusPlan, SceneAssignment, plan_sha256


@dataclass(frozen=True)
class CorpusSceneResult:
    split: str
    family: str
    scene_id: str
    scene_seed: int
    relative_dir: str
    manifest_sha256: str
    node_count: int
    edge_count: int
    cycle_count: int
    obstacle_count: int
    wall_count: int
    landmark_count: int


@dataclass(frozen=True)
class CorpusBuildResult:
    out_dir: str
    plan_sha256: str
    scene_count: int
    scenes: tuple[CorpusSceneResult, ...]


def build_corpus(
    plan: CorpusPlan,
    out_dir: Path,
    *,
    emit_genesis: bool = True,
) -> CorpusBuildResult:
    """Render every scene in ``plan`` under ``out_dir``.

    Layout::

        <out_dir>/
          corpus.json
          <split>/
            <family>/
              <scene_id>/
                manifest.json
                world.sdf
                topology.json
                genesis_scene.json   # optional
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_results: list[CorpusSceneResult] = []
    for assignment in plan.assignments:
        result = _build_one(
            assignment=assignment,
            out_dir=out_dir,
            emit_genesis=emit_genesis,
        )
        scene_results.append(result)

    corpus_payload = {
        "plan": plan.to_dict(),
        "plan_sha256": plan_sha256(plan),
        "scenes": [asdict(scene) for scene in scene_results],
    }
    (out_dir / "corpus.json").write_text(
        json.dumps(corpus_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return CorpusBuildResult(
        out_dir=str(out_dir),
        plan_sha256=plan_sha256(plan),
        scene_count=len(scene_results),
        scenes=tuple(scene_results),
    )


def iter_corpus_scenes(plan: CorpusPlan) -> Iterable[tuple[SceneAssignment, SceneManifest]]:
    """Yield ``(assignment, manifest)`` pairs without writing to disk."""

    for assignment in plan.assignments:
        manifest = build_scene_manifest(
            scene_seed=assignment.scene_seed,
            family=assignment.family,
            split=assignment.split,
        )
        yield assignment, manifest


def _build_one(
    *,
    assignment: SceneAssignment,
    out_dir: Path,
    emit_genesis: bool,
) -> CorpusSceneResult:
    manifest = build_scene_manifest(
        scene_seed=assignment.scene_seed,
        family=assignment.family,
        split=assignment.split,
    )
    if manifest.scene_id != assignment.scene_id:
        raise RuntimeError(
            "scene_id drift between plan and manifest: "
            f"plan={assignment.scene_id} manifest={manifest.scene_id}"
        )

    scene_dir = out_dir / assignment.split / assignment.family / assignment.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    export_gazebo_sdf(manifest, scene_dir)

    summary = topology_summary(manifest)
    (scene_dir / "topology.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if emit_genesis:
        genesis_spec = export_genesis_scene(manifest)
        (scene_dir / "genesis_scene.json").write_text(
            json.dumps(genesis_spec, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return CorpusSceneResult(
        split=assignment.split,
        family=assignment.family,
        scene_id=assignment.scene_id,
        scene_seed=assignment.scene_seed,
        relative_dir=str(scene_dir.relative_to(out_dir)),
        manifest_sha256=manifest_sha256(manifest),
        node_count=summary["node_count"],
        edge_count=summary["edge_count"],
        cycle_count=summary["cycle_count"],
        obstacle_count=len(manifest.obstacles),
        wall_count=len(manifest.walls),
        landmark_count=len(manifest.landmarks),
    )
