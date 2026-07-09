from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lewm.datasets.go2_navigation_source_index import (
    build_navigation_source_index,
)
from lewm.datasets.go2_paired_navigation import load_source_index
from lewm_worlds.manifest import (
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
    manifest_sha256,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _scene_digest(scene_id: str) -> str:
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def _manifest(scene_id: str, *, family: str, split: str) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family=family,
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35), quat_wxyz=(1.0, 0.0, 0.0, 0.0)
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split=split,
    )


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    render = tmp_path / "render"
    rollout = tmp_path / "rollout"
    corpora = tmp_path / "scene_corpus"
    render.mkdir()
    rollout.mkdir()
    corpora.mkdir()
    return render, rollout, corpora


def _write_commitments(
    tmp_path: Path,
    *,
    development: tuple[str, ...] = ("never_development",),
    sealed: tuple[str, ...] = ("never_sealed",),
) -> tuple[Path, Path]:
    development_path = tmp_path / "development.sha256"
    sealed_path = tmp_path / "sealed.sha256"
    development_path.write_text(
        "".join(_scene_digest(scene_id) + "\n" for scene_id in development)
    )
    sealed_path.write_text(
        "".join(_scene_digest(scene_id) + "\n" for scene_id in sealed)
    )
    return development_path, sealed_path


def _write_source(
    *,
    render_root: Path,
    rollout_root: Path,
    scene_corpus_root: Path,
    scene_id: str,
    family: str = "medium_enclosed_maze",
    split: str = "train",
    chunk_index: int = 0,
) -> dict[str, Path | str]:
    corpus = scene_corpus_root / "historical_v1"
    manifest = _manifest(scene_id, family=family, split=split)
    manifest_path = corpus / split / family / scene_id / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    canonical_manifest_hash = manifest_sha256(manifest)

    chunk = rollout_root / split / family / f"chunk_{chunk_index:04d}"
    plan_dir = chunk / "plan" / f"000000_{scene_id}"
    plan_dir.mkdir(parents=True)
    frames_path = plan_dir / "frames.jsonl"
    frames = []
    for frame_index in range(2):
        frames.append(
            {
                "frame_index": frame_index,
                "env_index": frame_index,
                "timestamp_ns": (frame_index + 1) * 100_000_000,
                "episode": {
                    "episode_id": 1,
                    "episode_step": frame_index + 1,
                    "manifest_sha256": canonical_manifest_hash,
                    "split": split,
                },
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    plan_path = plan_dir / "render_replay_plan.json"
    _write_json(
        plan_path,
        {
            "schema": "lewm_render_replay_plan_v0",
            "scene_id": scene_id,
            "scene_family": family,
            "split": split,
            "manifest_sha256": canonical_manifest_hash,
            "frame_count": 2,
            "frames_jsonl": str(frames_path.resolve()),
        },
    )
    run_summary_path = chunk / "rollout" / "run_summary.json"
    _write_json(
        run_summary_path,
        {
            "schema": "lewm_genesis_bulk_rollout_run_v0",
            "family": family,
            "split": split,
            "scene_corpus": str(corpus.resolve()),
        },
    )

    render_dir = render_root / scene_id
    rgb_dir = render_dir / "rgb"
    rgb_dir.mkdir(parents=True)
    for frame in frames:
        image_name = (
            f"frame_{frame['frame_index']:06d}_env_{frame['env_index']:02d}.png"
        )
        (rgb_dir / image_name).write_bytes(b"rgb")
    summary_path = render_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "schema": "lewm_rendered_vision_v03",
            "scene_id": scene_id,
            "family": family,
            "split": split,
            "render_status": "complete",
            "frame_count": 2,
            "plan": str(plan_path.resolve()),
        },
    )
    return {
        "manifest": manifest_path,
        "manifest_sha256": canonical_manifest_hash,
        "plan": plan_path,
        "frames": frames_path,
        "summary": summary_path,
        "corpus": corpus,
    }


def _build(
    tmp_path: Path,
    *,
    render: Path,
    rollout: Path,
    corpora: Path,
    development: Path,
    sealed: Path,
    families: tuple[str, ...] = (),
    splits: tuple[str, ...] = (),
    max_scenes_per_family: int | None = None,
    selection_seed: str = "go2_navigation_source_index_v1",
) -> dict:
    return build_navigation_source_index(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        output_dir=tmp_path / "output",
        exclusion_commitment_files=(
            ("development_holdout", development),
            ("sealed_holdout", sealed),
        ),
        families=families,
        splits=splits,
        max_scenes_per_family=max_scenes_per_family,
        selection_seed=selection_seed,
    )


def test_source_index_is_deterministic_content_addressed_and_loadable(
    tmp_path: Path,
) -> None:
    render, rollout, corpora = _roots(tmp_path)
    _write_source(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        scene_id="medium_enclosed_maze_training_a",
    )
    _write_source(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        scene_id="large_enclosed_maze_filtered",
        family="large_enclosed_maze",
        chunk_index=1,
    )
    development, sealed = _write_commitments(tmp_path)

    first = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
        families=("medium_enclosed_maze",),
        splits=("train",),
    )
    second = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
        families=("medium_enclosed_maze",),
        splits=("train",),
    )

    assert first == second
    assert first["accepted"] == 1
    assert first["filtered"] == 1
    assert Path(first["index_path"]).name == (
        f"go2_navigation_sources_{first['index_sha256']}.jsonl"
    )
    sources = load_source_index(Path(first["index_path"]))
    assert [source.scene_id for source in sources] == [
        "medium_enclosed_maze_training_a"
    ]
    assert [source.family for source in sources] == ["medium_enclosed_maze"]
    report = json.loads(Path(first["report_path"]).read_text())
    assert report["counts"] == {
        "accepted": 1,
        "eligible_selection_candidates": 1,
        "filtered": 1,
        "forbidden_before_artifact_open": 0,
        "rejected": 0,
        "render_directories_discovered": 2,
        "selected_for_deep_validation": 1,
    }


def test_forbidden_scene_is_screened_before_any_artifact_is_opened(
    tmp_path: Path,
) -> None:
    render, rollout, corpora = _roots(tmp_path)
    forbidden_scene = "v3_sealed_scene"
    forbidden_dir = render / forbidden_scene
    forbidden_dir.mkdir()
    (forbidden_dir / "summary.json").write_text("not valid JSON")
    development, sealed = _write_commitments(
        tmp_path, sealed=(forbidden_scene,)
    )

    result = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
    )

    assert result["accepted"] == 0
    assert result["forbidden"] == 1
    assert result["rejected"] == 0
    report_text = Path(result["report_path"]).read_text()
    assert forbidden_scene not in report_text
    report = json.loads(report_text)
    assert report["forbidden"] == [
        {
            "scene_id_sha256": _scene_digest(forbidden_scene),
            "labels": ["sealed_holdout"],
        }
    ]


def test_manifest_hash_mismatch_rejects_source(tmp_path: Path) -> None:
    render, rollout, corpora = _roots(tmp_path)
    artifacts = _write_source(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        scene_id="mismatched_scene",
    )
    plan_path = Path(artifacts["plan"])
    plan = json.loads(plan_path.read_text())
    plan["manifest_sha256"] = "0" * 64
    _write_json(plan_path, plan)
    development, sealed = _write_commitments(tmp_path)

    result = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
    )

    assert result["accepted"] == 0
    assert result["rejected"] == 1
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["rejection_counts"] == {"frame_manifest_hash_mismatch": 1}


def test_duplicate_rollout_plan_rejects_source(tmp_path: Path) -> None:
    render, rollout, corpora = _roots(tmp_path)
    artifacts = _write_source(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        scene_id="duplicated_scene",
    )
    duplicate_dir = (
        rollout
        / "train"
        / "medium_enclosed_maze"
        / "chunk_9999"
        / "plan"
        / "000001_duplicated_scene"
    )
    duplicate_dir.mkdir(parents=True)
    duplicate_plan = duplicate_dir / "render_replay_plan.json"
    duplicate_plan.write_bytes(Path(artifacts["plan"]).read_bytes())
    development, sealed = _write_commitments(tmp_path)

    result = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
    )

    assert result["accepted"] == 0
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["rejection_counts"] == {
        "duplicate_or_missing_render_plan": 1
    }


def test_duplicate_origin_manifest_rejects_source(tmp_path: Path) -> None:
    render, rollout, corpora = _roots(tmp_path)
    artifacts = _write_source(
        render_root=render,
        rollout_root=rollout,
        scene_corpus_root=corpora,
        scene_id="duplicate_manifest_scene",
    )
    duplicate_manifest = (
        Path(artifacts["corpus"])
        / "validation"
        / "medium_enclosed_maze"
        / "duplicate_manifest_scene"
        / "manifest.json"
    )
    duplicate_manifest.parent.mkdir(parents=True)
    duplicate_manifest.write_bytes(Path(artifacts["manifest"]).read_bytes())
    development, sealed = _write_commitments(tmp_path)

    result = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
    )

    assert result["accepted"] == 0
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["rejection_counts"] == {
        "duplicate_or_missing_scene_manifest": 1
    }


def test_hash_rank_limit_is_deterministic_and_does_not_open_omitted_frames(
    tmp_path: Path,
) -> None:
    render, rollout, corpora = _roots(tmp_path)
    seed = "balanced-selection-v1"
    family = "medium_enclosed_maze"
    scene_ids = tuple(f"selection_scene_{index}" for index in range(3))
    artifacts = {
        scene_id: _write_source(
            render_root=render,
            rollout_root=rollout,
            scene_corpus_root=corpora,
            scene_id=scene_id,
            family=family,
            chunk_index=index,
        )
        for index, scene_id in enumerate(scene_ids)
    }
    ranked = sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(
            f"{seed}\0{family}\0{scene_id}".encode("utf-8")
        ).hexdigest(),
    )
    omitted = ranked[-1]
    Path(artifacts[omitted]["frames"]).write_text("invalid and deliberately unopened\n")
    development, sealed = _write_commitments(tmp_path)

    result = _build(
        tmp_path,
        render=render,
        rollout=rollout,
        corpora=corpora,
        development=development,
        sealed=sealed,
        families=(family,),
        splits=("train",),
        max_scenes_per_family=2,
        selection_seed=seed,
    )

    assert result["accepted"] == 2
    assert result["rejected"] == 0
    rows = [json.loads(line) for line in Path(result["index_path"]).read_text().splitlines()]
    assert sorted(row["scene_id"] for row in rows) == sorted(ranked[:2])
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["selection"] == {
        "candidate_by_family": {family: 3},
        "candidate_count": 3,
        "max_scenes_per_family": 2,
        "method": "sha256(seed\\0family\\0scene_id)_ascending",
        "seed": seed,
        "selected_for_deep_validation_by_family": {family: 2},
        "selected_for_deep_validation_count": 2,
    }
    omitted_ids = {
        row["scene_id"]
        for row in report["filtered"]
        if row["reason"] == "max_scenes_per_family"
    }
    assert omitted_ids == {omitted}
