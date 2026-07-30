from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.datasets.go2_memory_role_place_triplets_v1 import (
    MANIFEST_SCHEMA,
    PlaceTripletContractError,
    canonical_json_bytes,
    canonical_json_sha256,
    load_index,
    load_rgb_triplet,
)
from scripts import build_go2_memory_role_place_triplet_index_v1 as builder


FAMILY = "open_obstacle_field"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _relative_rgb(scene: str, index: int) -> str:
    render_scene = hashlib.sha256(scene.encode("ascii")).hexdigest()[:16]
    return (
        ".generated/go2_render_selected_v04/scenes/"
        f"scene_{render_scene}/rgb/frame_{index:06d}_env_00.png"
    )


def _source_frames_inventory(
    root: Path, *, scene: str, source_split: str = "train"
) -> dict[str, object]:
    return {
        "scene_id": scene,
        "purpose": "source_frames_jsonl",
        "path": str(
            root
            / f".generated/datagen_full/rollout/{source_split}"
            / FAMILY
            / "chunk_0000/plan"
            / f"000001_{scene}"
            / "frames.jsonl"
        ),
    }


def _raw_endpoint(root: Path, *, role: str, scene: str, index: int) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": builder.RAW_ENDPOINT_SCHEMA,
        "dataset_role": role,
        "family": FAMILY,
        "scene_id": scene,
        "endpoint_identity_sha256": _sha(f"endpoint-{role}-{index}"),
        "plan_endpoint_content_sha256": _sha(f"plan-{role}-{index}"),
        "shard_row": index,
        "image_path_metadata_only": str(root / _relative_rgb(scene, index)),
        "image_sha256_commitment_only": _sha(f"image-{role}-{index}"),
        "evidence_content_sha256": _sha(f"evidence-{role}-{index}"),
        "raster_content_sha256": _sha(f"raster-{role}-{index}"),
        "scene_shard": "shards/0123456789abcdef/shard.json",
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def test_role_is_filtered_before_any_referenced_path_validation(tmp_path: Path) -> None:
    allowed = _raw_endpoint(tmp_path, role="train", scene="scene_a", index=0)
    excluded = {
        "dataset_role": "probability_calibration",
        "image_path_metadata_only": "/must/not/be/validated/or/opened",
    }
    raw = b"".join(
        canonical_json_bytes(row) + b"\n" for row in (excluded, allowed)
    )

    endpoints, audit = builder.decode_allowed_endpoint_bytes(
        tmp_path.resolve(), raw, enforce_frozen_counts=False
    )

    assert len(endpoints) == 1
    assert endpoints[0].role == "train"
    assert audit["excluded_probability_calibration_endpoint_count"] == 1
    assert audit["excluded_role_referenced_path_dereference_count"] == 0


def test_real_render_path_and_manifest_source_frames_bind_original_labels(
    tmp_path: Path,
) -> None:
    scene = "open_obstacle_field_0123456789ab"
    endpoint = _raw_endpoint(tmp_path, role="train", scene=scene, index=7)
    endpoints, _ = builder.decode_allowed_endpoint_bytes(
        tmp_path.resolve(),
        canonical_json_bytes(endpoint) + b"\n",
        enforce_frozen_counts=False,
    )
    manifest = {
        "input_provenance": {
            "source_payload_inventory": [
                {
                    "scene_id": "calibration_scene_must_remain_unselected",
                    "purpose": "source_frames_jsonl",
                    "path": "/must/not/be/validated/or/opened",
                },
                {
                    "scene_id": "different_scene",
                    "purpose": "source_frames_jsonl",
                    "path": "/irrelevant/allowed-role/scene/must/not/be/validated",
                },
                {
                    "scene_id": scene,
                    "purpose": "render_summary",
                    "path": "/allowed/scene/non-frame/path/must/not/be/validated",
                },
                _source_frames_inventory(
                    tmp_path.resolve(), scene=scene, source_split="test_hard"
                ),
            ]
        }
    }

    bound, audit = builder.bind_allowed_label_paths(
        tmp_path.resolve(), endpoints, manifest
    )

    render_scene = hashlib.sha256(scene.encode("ascii")).hexdigest()[:16]
    assert bound[0].rgb_path == (
        ".generated/go2_render_selected_v04/scenes/"
        f"scene_{render_scene}/rgb/frame_000007_env_00.png"
    )
    assert bound[0].label_path == (
        f".generated/datagen_full/rollout/test_hard/{FAMILY}/chunk_0000/"
        f"labels/{scene}/labels.jsonl"
    )
    assert audit["allowed_scene_source_frames_binding_count"] == 1
    assert audit["unselected_scene_inventory_record_count"] == 2
    assert audit["excluded_role_referenced_path_dereference_count"] == 0


def test_label_join_reproduces_full_endpoint_identity_without_rgb_open(
    tmp_path: Path,
) -> None:
    scene = "scene_a"
    image_sha = _sha("image")
    identity = {
        "dataset_role": "train",
        "scene_id": scene,
        "episode_id": "7",
        "env_index": 0,
        "episode_step": 11,
        "frame_index": 0,
        "timestamp_ns": 12_000_000_000,
        "image_sha256": image_sha,
    }
    endpoint = builder.EndpointMetadata(
        role="train",
        family=FAMILY,
        scene_id=scene,
        endpoint_identity_sha256=canonical_json_sha256(identity),
        image_sha256=image_sha,
        rgb_path=_relative_rgb(scene, 0),
        label_path=(
            f".generated/datagen_full/rollout/train/{FAMILY}/chunk_0000/"
            f"labels/{scene}/labels.jsonl"
        ),
        frame_index=0,
        env_index=0,
    )
    label_path = tmp_path / endpoint.label_path
    label_path.parent.mkdir(parents=True)
    label_path.write_bytes(
        canonical_json_bytes(
            {
                "timestamp_ns": 12_000_000_000,
                "env_idx": 0,
                "episode_id": 7,
                "episode_step": 11,
                "scene_id": scene,
                "cell_id": 3,
                "yaw_bin": 2,
                "yaw_bin_count": 8,
            }
        )
        + b"\n"
    )

    joined, receipts = builder.join_exact_derived_labels(
        tmp_path.resolve(), (endpoint,)
    )

    assert len(joined) == len(receipts) == 1
    assert (joined[0].cell_id, joined[0].yaw_bin) == (3, 2)
    assert not (tmp_path / endpoint.rgb_path).exists()
    assert receipts[0]["selected_endpoint_count"] == 1


def _joined(role: str, scene: str, index: int, *, cell: int) -> builder.JoinedEndpoint:
    return builder.JoinedEndpoint(
        role=role,
        family=FAMILY,
        scene_id=scene,
        endpoint_identity_sha256=_sha(f"{role}-{scene}-{index}"),
        image_sha256=_sha(f"image-{role}-{scene}-{index}"),
        rgb_path=_relative_rgb(scene, index),
        frame_index=index,
        env_index=0,
        episode_id=str(index + 1),
        episode_step=index,
        timestamp_ns=index * 5_000_000_000,
        cell_id=cell,
        yaw_bin=1,
    )


def test_balanced_selection_has_unique_anchor_and_positive_keys() -> None:
    endpoints = []
    for role, count in (("train", 8), ("checkpoint_selection", 34)):
        scene = f"scene_{role}"
        endpoints.extend(
            _joined(role, scene, index, cell=index % 2) for index in range(count)
        )

    selected, support = builder.select_balanced_triplets(
        endpoints,
        families=(FAMILY,),
        targets={"train": 4, "checkpoint_selection": 32},
        require_all_checkpoint_scenes=False,
    )

    assert len(selected["train"]) == 4
    assert len(selected["checkpoint_selection"]) == 32
    for role, rows in selected.items():
        assert len({row.anchor.endpoint_identity_sha256 for row in rows}) == len(rows)
        assert len({row.positive.endpoint_identity_sha256 for row in rows}) == len(rows)
        assert support[role]["selected_anchor_identity_unique"] is True
        assert support[role]["selected_positive_identity_unique_within_scene"] is True
        assert all(row.anchor.cell_id == row.positive.cell_id for row in rows)
        assert all(row.anchor.cell_id != row.negative.cell_id for row in rows)


def test_default_schedule_matches_measured_support_without_cycling() -> None:
    quotas, schedule = builder._resolve_family_quotas(
        builder.FAMILIES,
        builder.TARGET_ROWS,
        None,
    )

    assert schedule == "measured_support_v1"
    assert quotas == builder.DEFAULT_FAMILY_QUOTAS
    assert {role: sum(values.values()) for role, values in quotas.items()} == {
        "train": 3_200,
        "checkpoint_selection": 320,
    }
    for role in builder.ALLOWED_ROLES:
        assert all(
            quotas[role][family]
            <= builder.MEASURED_CANDIDATE_CAPACITIES[role][family]
            for family in builder.FAMILIES
        )


def _write_png(path: Path, color: tuple[int, int, int]) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (224, 224), color=color)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_candidate(
    root: Path,
    *,
    role: str,
    scene: str,
    offset: int,
) -> builder.TripletCandidate:
    values = []
    for index, cell in ((offset, 1), (offset + 1, 1), (offset + 2, 2)):
        relative = _relative_rgb(scene, index)
        image_sha = _write_png(root / relative, (index % 255, 20, 30))
        values.append(
            builder.JoinedEndpoint(
                role=role,
                family=FAMILY,
                scene_id=scene,
                endpoint_identity_sha256=_sha(f"runtime-{role}-{index}"),
                image_sha256=image_sha,
                rgb_path=relative,
                frame_index=index,
                env_index=0,
                episode_id=str(index + 1),
                episode_step=index,
                timestamp_ns=index * 5_000_000_000,
                cell_id=cell,
                yaw_bin=2,
            )
        )
    return builder.TripletCandidate(*values)


def test_runtime_emits_only_rgb_tensors_and_checks_manifest_hash(tmp_path: Path) -> None:
    train = _runtime_candidate(tmp_path, role="train", scene="scene_train", offset=10)
    selection = _runtime_candidate(
        tmp_path,
        role="checkpoint_selection",
        scene="scene_selection",
        offset=20,
    )
    output = tmp_path / "index"
    receipt = builder.publish_index(
        output,
        {"train": (train,), "checkpoint_selection": (selection,)},
        {"schema": MANIFEST_SCHEMA, "status": "PASS"},
    )
    rows, audit = load_index(
        tmp_path,
        output,
        role="checkpoint_selection",
        expected_manifest_sha256=receipt["manifest_file_sha256"],
    )
    assert not hasattr(rows[0], "cell_id") and not hasattr(rows[0], "yaw_bin")
    assert audit["privileged_label_fields_emitted_to_model"] == 0
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(PlaceTripletContractError, match="Pillow and torch"):
            load_rgb_triplet(tmp_path, rows[0])
    else:
        batch = load_rgb_triplet(tmp_path, rows[0])
        assert tuple(batch.anchor_rgb.shape) == (3, 112, 112)
        assert tuple(batch.positive_rgb.shape) == (3, 112, 112)
        assert tuple(batch.negative_rgb.shape) == (3, 112, 112)
        assert not hasattr(batch, "cell_id") and not hasattr(batch, "yaw_bin")

    with pytest.raises(PlaceTripletContractError):
        load_index(
            tmp_path,
            output,
            role="checkpoint_selection",
            expected_manifest_sha256="0" * 64,
        )
