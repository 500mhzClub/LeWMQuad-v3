from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot
from scripts import build_go2_world_model_visual_domain_parity_plan_v1 as plan_builder


def _binding(path: str, digit: str) -> dict[str, object]:
    return {
        "path": path,
        "file_sha256": digit * 64,
        "byte_count": 1,
    }


def _frame(pose_index: int) -> dict[str, object]:
    return {
        "frame_index": pose_index,
        "env_index": pose_index,
        "base_pose_world": {
            "position": {"x": float(pose_index), "y": 0.0, "z": 0.3},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "camera_pose_world": {
            "position": [float(pose_index), 0.0, 0.4],
            "lookat": [float(pose_index) + 1.0, 0.0, 0.4],
            "up": [0.0, 0.0, 1.0],
        },
    }


def test_source_pose_is_rederived_from_both_quaternion_encodings() -> None:
    frame = _frame(2)
    position, quaternion, camera, record_sha256 = (
        plan_builder._source_pose_from_frame(  # noqa: SLF001
            frame, scene_id="scene", pose_index=2
        )
    )
    assert position == [2.0, 0.0, 0.3]
    assert quaternion == [1.0, 0.0, 0.0, 0.0]
    assert camera == frame["camera_pose_world"]
    assert record_sha256 == plan_builder._canonical_sha256(frame)  # noqa: SLF001

    changed = copy.deepcopy(frame)
    changed["base_quat_world_xyzw"] = [0.0, 0.0, 1.0, 0.0]
    with pytest.raises(
        plan_builder.VisualDomainParityPlanError,
        match="quaternion encodings disagree",
    ):
        plan_builder._source_pose_from_frame(  # noqa: SLF001
            changed, scene_id="scene", pose_index=2
        )


def test_output_root_rejects_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_root = tmp_path / ".generated/dev"
    development_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (development_root / "redirect").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(plan_builder, "DEVELOPMENT_ROOT", development_root)

    with pytest.raises(
        plan_builder.VisualDomainParityPlanError,
        match="mutable/non-directory|canonical spelling",
    ):
        plan_builder._validate_output_root(  # noqa: SLF001
            development_root / "redirect/attempt", require_fresh=True
        )


def test_missing_cached_mesh_is_rejected_without_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_builder, "REPO_ROOT", tmp_path)
    cached_mesh_writer_called = False

    def forbidden_cached_mesh_writer(*_args, **_kwargs):
        nonlocal cached_mesh_writer_called
        cached_mesh_writer_called = True
        raise AssertionError("plan validation must not materialize a mesh")

    monkeypatch.setattr(
        plan_builder.reference_renderer,
        "cached_box_obj",
        forbidden_cached_mesh_writer,
    )
    manifest = {
        "walls": [{"kind": "wall", "size_xyz_m": [1.0, 2.0, 3.0]}],
        "obstacles": [],
        "landmarks": [],
    }

    with pytest.raises(
        plan_builder.VisualDomainParityPlanError,
        match="derived mesh is unavailable",
    ):
        plan_builder._mesh_bindings_for_manifest(  # noqa: SLF001
            manifest, scene_id="ordinary-train-scene"
        )

    assert cached_mesh_writer_called is False
    assert not (tmp_path / ".generated").exists()


def test_mesh_content_and_binding_are_derived_without_second_binding_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_builder, "REPO_ROOT", tmp_path)
    size = (1.0, 2.0, 3.0)
    tiles_per_m = (
        plan_builder.reference_renderer._textures._DEFAULT_TILES_PER_M  # noqa: SLF001
    )
    mesh_path = (
        tmp_path
        / ".generated/box_meshes"
        / f"box_1.000x2.000x3.000_t{float(tiles_per_m):.2f}.obj"
    )
    mesh_path.parent.mkdir(parents=True)
    payload = plan_builder.reference_renderer._textures.box_obj_text(  # noqa: SLF001
        size, tiles_per_m=tiles_per_m
    ).encode("utf-8")
    mesh_path.write_bytes(payload)

    def forbidden_second_binding_read(*_args, **_kwargs):
        raise AssertionError("mesh binding must come from its content-validation read")

    monkeypatch.setattr(plan_builder, "_binding", forbidden_second_binding_read)
    bindings = plan_builder._mesh_bindings_for_manifest(  # noqa: SLF001
        {
            "walls": [{"kind": "wall", "size_xyz_m": list(size)}],
            "obstacles": [],
            "landmarks": [],
        },
        scene_id="ordinary-train-scene",
    )

    assert bindings == [{
        "path": str(mesh_path),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }]


def test_source_rgb_pixels_use_one_bound_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rgb_path = tmp_path / "source.png"
    Image.new("RGB", (224, 224), color=(7, 11, 13)).save(rgb_path)
    binding = pilot.file_binding(rgb_path)

    def forbidden_rehash(*_args, **_kwargs):
        raise AssertionError("source RGB must not be rebound around its decode")

    monkeypatch.setattr(plan_builder.pilot, "file_binding", forbidden_rehash)
    raw_sha256 = plan_builder._raw_rgb_sha256(  # noqa: SLF001
        binding, label="source RGB"
    )

    expected = bytes((7, 11, 13)) * (224 * 224)
    assert raw_sha256 == hashlib.sha256(expected).hexdigest()


@pytest.fixture
def exact_plan_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    development_root = tmp_path / ".generated/dev"
    development_root.mkdir(parents=True)
    monkeypatch.setattr(plan_builder, "DEVELOPMENT_ROOT", development_root)

    texture_bindings = [
        _binding(f"/fixture/texture-{index}.png", hex(index + 1)[-1])
        for index in range(12)
    ]
    implementation_bindings = {
        str(plan_builder.REFERENCE_RENDERER): _binding(
            str(plan_builder.REFERENCE_RENDERER), "a"
        ),
        str(plan_builder.REFERENCE_TEXTURE_SOURCE): _binding(
            str(plan_builder.REFERENCE_TEXTURE_SOURCE), "b"
        ),
    }
    monkeypatch.setattr(
        plan_builder,
        "_binding",
        lambda path, *, label: copy.deepcopy(implementation_bindings[str(path)]),
    )
    monkeypatch.setattr(
        plan_builder.pilot,
        "require_binding",
        lambda value, *, label: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        plan_builder,
        "_require_exact_binding",
        lambda value, *, label: copy.deepcopy(dict(value)),
    )
    monkeypatch.setattr(
        plan_builder,
        "_texture_asset_closure",
        lambda: copy.deepcopy(texture_bindings),
    )
    runtime = {"runtime": _binding("/fixture/runtime.json", "c")}
    execution = {"backend": "vulkan"}
    monkeypatch.setattr(
        plan_builder,
        "_validate_runtime",
        lambda value: (copy.deepcopy(runtime), copy.deepcopy(execution)),
    )

    documents: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    frames_by_path: dict[str, list[dict[str, object]]] = {}
    raw_by_path: dict[str, str] = {}
    scenes: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    selected_texture_map: dict[str, dict[str, object]] = {}
    mesh_map: dict[str, list[dict[str, object]]] = {}
    genesis_map: dict[str, dict[str, object]] = {}
    summary_map: dict[str, dict[str, object]] = {}
    render_plan_map: dict[str, dict[str, object]] = {}
    frames_map: dict[str, dict[str, object]] = {}
    expected_inventory: list[dict[str, object]] = []

    for scene_index, family in enumerate(pilot.FAMILIES):
        scene_id = f"{scene_index:02d}_{family}"
        manifest_binding = _binding(f"/fixture/{scene_id}/manifest.json", "1")
        genesis_binding = _binding(f"/fixture/{scene_id}/genesis.json", "2")
        summary_binding = _binding(f"/fixture/{scene_id}/summary.json", "3")
        render_plan_binding = _binding(f"/fixture/{scene_id}/plan.json", "4")
        frames_binding = _binding(f"/fixture/{scene_id}/frames.jsonl", "5")
        mesh_binding = _binding(f"/fixture/{scene_id}/mesh.obj", "6")
        selected_textures = {
            "floor": texture_bindings[0],
            "wall": texture_bindings[4],
            "obstacle": texture_bindings[8],
        }
        manifest = {"scene_id": scene_id, "family": family, "split": "train"}
        documents[str(manifest_binding["path"])] = (manifest, manifest_binding)
        documents[str(genesis_binding["path"])] = (
            {"scene_id": scene_id},
            genesis_binding,
        )
        documents[str(summary_binding["path"])] = ({}, summary_binding)
        documents[str(render_plan_binding["path"])] = ({}, render_plan_binding)
        scene_frames = [_frame(index) for index in range(4)]
        frames_by_path[str(frames_binding["path"])] = scene_frames
        poses: list[dict[str, object]] = []
        for pose_index, frame in enumerate(scene_frames):
            pair_id = f"{scene_id}/pose_{pose_index:02d}"
            rgb_binding = _binding(
                f"/fixture/{scene_id}/rgb/{pose_index}.png", "7"
            )
            raw_sha256 = str(pose_index) * 64
            raw_by_path[str(rgb_binding["path"])] = raw_sha256
            camera = copy.deepcopy(frame["camera_pose_world"])
            identity = f"frame_{pose_index:06d}_env_{pose_index:02d}"
            poses.append({
                "pair_id": pair_id,
                "pose_index": pose_index,
                "source_frame_index": pose_index,
                "source_env_index": pose_index,
                "base_position_xyz_m": [float(pose_index), 0.0, 0.3],
                "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "historical_camera_pose_world": camera,
                "source_rgb_binding": rgb_binding,
                "source_raw_pixel_sha256": raw_sha256,
                "source_frame_record_sha256": plan_builder._canonical_sha256(  # noqa: SLF001
                    frame
                ),
                "producer_frame_identity": identity,
            })
            panel_rows.append({
                "pair_id": pair_id,
                "scene_id": scene_id,
                "family": family,
                "pose_index": pose_index,
                "camera_pose_world": camera,
                "scene_manifest_binding": manifest_binding,
                "producer_frame_identity": identity,
                "rgb_binding": rgb_binding,
                "raw_rgb_sha256": raw_sha256,
            })
        scenes.append({
            "family": family,
            "scene_id": scene_id,
            "scene_manifest_binding": manifest_binding,
            "scene_genesis_binding": genesis_binding,
            "render_summary_binding": summary_binding,
            "render_plan_binding": render_plan_binding,
            "frames_jsonl_binding": frames_binding,
            "selected_texture_asset_bindings": selected_textures,
            "mesh_asset_bindings": [mesh_binding],
            "poses": poses,
        })
        selected_texture_map[scene_id] = selected_textures
        mesh_map[scene_id] = [mesh_binding]
        genesis_map[scene_id] = genesis_binding
        summary_map[scene_id] = summary_binding
        render_plan_map[scene_id] = render_plan_binding
        frames_map[scene_id] = frames_binding
        expected_inventory.append({"family": family, "scene_id": scene_id})

    panel_rows.sort(key=lambda row: str(row["pair_id"]))
    source_panel = {
        "schema": plan_builder.SOURCE_PANEL_SCHEMA,
        "domain": plan_builder.SOURCE_DOMAIN,
        "rgb_root": str(plan_builder.SOURCE_RGB_ROOT),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "producer_source_binding": implementation_bindings[
            str(plan_builder.REFERENCE_RENDERER)
        ],
        "renderer_source_binding": implementation_bindings[
            str(plan_builder.REFERENCE_RENDERER)
        ],
        "texture_source_binding": implementation_bindings[
            str(plan_builder.REFERENCE_TEXTURE_SOURCE)
        ],
        "selected_texture_asset_bindings_by_scene": selected_texture_map,
        "mesh_asset_bindings_by_scene": mesh_map,
        "producer_lineage": {
            "schema": plan_builder.SOURCE_LINEAGE_SCHEMA,
            "scene_genesis_bindings_by_scene": genesis_map,
            "render_summary_bindings_by_scene": summary_map,
            "render_plan_bindings_by_scene": render_plan_map,
            "frames_jsonl_bindings_by_scene": frames_map,
        },
        "rows": panel_rows,
    }
    source_panel_binding = _binding("/fixture/source-panel.json", "8")
    documents[str(source_panel_binding["path"])] = (
        source_panel,
        source_panel_binding,
    )
    monkeypatch.setattr(
        plan_builder,
        "_read_bound_json",
        lambda binding, *, label: copy.deepcopy(
            documents[str(binding["path"])]
        ),
    )
    monkeypatch.setattr(
        plan_builder,
        "_read_first_bound_frames",
        lambda binding, *, scene_id: copy.deepcopy(
            frames_by_path[str(binding["path"])]
        ),
    )
    monkeypatch.setattr(
        plan_builder,
        "_raw_rgb_sha256",
        lambda binding, *, label: raw_by_path[str(binding["path"])],
    )
    monkeypatch.setattr(
        plan_builder,
        "_mesh_bindings_for_manifest",
        lambda manifest, *, scene_id: copy.deepcopy(mesh_map[scene_id]),
    )
    corpus_bindings = [_binding("/fixture/corpus.json", "9")]
    monkeypatch.setattr(
        plan_builder,
        "_ranked_source_rows",
        lambda: (copy.deepcopy(corpus_bindings), copy.deepcopy(expected_inventory)),
    )
    return {
        "schema": plan_builder.PLAN_SCHEMA,
        "status": plan_builder.PLAN_STATUS,
        "attempt_id": "parity-attempt",
        "purpose": plan_builder.PURPOSE,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "output_root": str(development_root / "parity-attempt"),
        "render_contract": dict(pilot.TEXTURED_V03_RENDER_CONTRACT),
        "comparison_contract": dict(plan_builder.COMPARISON_CONTRACT),
        "expected_counts": dict(plan_builder.EXPECTED_COUNTS),
        "runtime_bindings": runtime,
        "execution_contract": execution,
        "texture_asset_bindings": texture_bindings,
        "source_panel_binding": source_panel_binding,
        "scene_corpus_manifest_bindings": corpus_bindings,
        "scenes": scenes,
        "mesh_asset_bindings": sorted(
            [binding for values in mesh_map.values() for binding in values],
            key=lambda binding: str(binding["path"]),
        ),
    }


def test_exact_plan_validator_rederives_full_source_closure(
    exact_plan_fixture: dict[str, object],
) -> None:
    assert plan_builder.validate_plan_v1(exact_plan_fixture) == exact_plan_fixture


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["scenes"][0]["poses"][0].__setitem__(
                "base_position_xyz_m", [99.0, 0.0, 0.3]
            ),
            "pose/source lineage changed",
        ),
        (
            lambda value: value["scenes"][0]["poses"][0].__setitem__(
                "source_frame_record_sha256", "f" * 64
            ),
            "pose/source lineage changed",
        ),
        (
            lambda value: value["scene_corpus_manifest_bindings"].append(
                _binding("/fixture/foreign-corpus.json", "f")
            ),
            "corpus binding changed",
        ),
    ],
)
def test_exact_plan_validator_rejects_pose_or_selection_drift(
    exact_plan_fixture: dict[str, object], mutate, message: str
) -> None:
    changed = copy.deepcopy(exact_plan_fixture)
    mutate(changed)
    with pytest.raises(plan_builder.VisualDomainParityPlanError, match=message):
        plan_builder.validate_plan_v1(changed)
