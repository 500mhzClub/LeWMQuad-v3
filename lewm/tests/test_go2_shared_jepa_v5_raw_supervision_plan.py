from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from lewm.datasets import go2_attitude_sidecar as attitude_sidecar
from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan import (
    DEVELOPMENT_ROLES,
    PRIMITIVE_VOCABULARY,
    RawSupervisionPlanError,
    canonical_json_sha256,
    endpoint_identity,
    load_frozen_development_metadata,
    load_frozen_development_source_inventory,
    plan_development_raw_supervision,
    plan_development_source_inventory,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _row(index: int, primitive: str) -> dict[str, object]:
    return {
        "schema": "lewm_go2_paired_navigation_row_v3",
        "global_row": index,
        "dataset_role": "train",
        "scene_id": "scene_a",
        "family": "family_a",
        "episode_id": "1",
        "env_index": 0,
        "reset_count": 1,
        "source_split": "train",
        "frames_jsonl_sha256": _sha("frames"),
        "scene_manifest_sha256": _sha("scene-manifest"),
        "current_episode_step": index,
        "current_frame_index": index,
        "current_timestamp_ns": index * 100,
        "current_image_path": f"/metadata/frame_{index:04d}.png",
        "current_image_sha256": _sha(f"image-{index}"),
        "next_episode_step": index + 1,
        "next_frame_index": index + 1,
        "next_timestamp_ns": (index + 1) * 100,
        "next_image_path": f"/metadata/frame_{index + 1:04d}.png",
        "next_image_sha256": _sha(f"image-{index + 1}"),
        "primitive": primitive,
        "relative_se2_current_frame": [0.1, 0.0, 0.01],
        "label_shard_path": "/metadata/scene_a.npz",
        "label_shard_sha256": _sha("label-shard"),
        "label_shard_row": index,
    }


def _sidecar(row: dict[str, object]) -> dict[str, object]:
    index = int(row["global_row"])
    core = {
        "schema": "lewm_go2_attitude_sidecar_row_v1",
        "global_row": index,
        "dataset_role": "train",
        "row_identity_sha256": attitude_sidecar.row_identity_sha256(row),
        "scene_id_sha256": hashlib.sha256(b"scene_a").hexdigest(),
        "frames_jsonl_sha256": row["frames_jsonl_sha256"],
        "env_index": 0,
        "current_frame_index": index,
        "next_frame_index": index + 1,
        "current_timestamp_ns": index * 100,
        "next_timestamp_ns": (index + 1) * 100,
        "current": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": float(index) / 10.0,
        },
        "next": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": float(index + 1) / 10.0,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _synthetic_plan():
    rows = [_row(index, primitive) for index, primitive in enumerate(PRIMITIVE_VOCABULARY)]
    sidecars = {role: [] for role in DEVELOPMENT_ROLES}
    sidecars["train"] = [_sidecar(row) for row in rows]
    return plan_development_raw_supervision(
        rows,
        sidecars,
        input_bindings={"synthetic": True},
        access_ledger={"payload_opens": 0},
        enforce_frozen_counts=False,
    )


def test_synthetic_plan_deduplicates_only_complete_endpoint_identity() -> None:
    plan = _synthetic_plan()
    assert len(plan.pairs) == len(PRIMITIVE_VOCABULARY)
    assert len(plan.endpoints) == len(PRIMITIVE_VOCABULARY) + 1
    assert plan.value["endpoint_instance_count"] == 2 * len(PRIMITIVE_VOCABULARY)
    first_next = endpoint_identity(_row(0, PRIMITIVE_VOCABULARY[0]), "next")
    second_current = endpoint_identity(_row(1, PRIMITIVE_VOCABULARY[1]), "current")
    assert first_next == second_current


def test_same_image_hash_does_not_collapse_distinct_endpoint_identity() -> None:
    first = _row(0, PRIMITIVE_VOCABULARY[0])
    second = _row(1, PRIMITIVE_VOCABULARY[1])
    second["current_image_sha256"] = first["current_image_sha256"]
    assert endpoint_identity(first, "current") != endpoint_identity(second, "current")


def test_conflicting_metadata_for_one_endpoint_is_rejected() -> None:
    rows = [_row(index, primitive) for index, primitive in enumerate(PRIMITIVE_VOCABULARY)]
    sidecars = {role: [] for role in DEVELOPMENT_ROLES}
    sidecars["train"] = [_sidecar(row) for row in rows]
    changed = copy.deepcopy(sidecars["train"][1])
    changed["current"]["stored_base_yaw_rad"] = 2.5
    changed_core = dict(changed)
    changed_core.pop("content_sha256")
    changed["content_sha256"] = canonical_json_sha256(changed_core)
    sidecars["train"][1] = changed
    with pytest.raises(RawSupervisionPlanError, match="conflicting metadata"):
        plan_development_raw_supervision(
            rows,
            sidecars,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )


def test_sidecar_join_mutation_is_rejected_before_plan() -> None:
    rows = [_row(index, primitive) for index, primitive in enumerate(PRIMITIVE_VOCABULARY)]
    sidecars = {role: [] for role in DEVELOPMENT_ROLES}
    sidecars["train"] = [_sidecar(row) for row in rows]
    sidecars["train"][0]["current_frame_index"] = 99
    core = dict(sidecars["train"][0])
    core.pop("content_sha256")
    sidecars["train"][0]["content_sha256"] = canonical_json_sha256(core)
    with pytest.raises(RawSupervisionPlanError, match="sidecar join changed"):
        plan_development_raw_supervision(
            rows,
            sidecars,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )


def test_g2_sidecar_role_cannot_enter_development_plan() -> None:
    rows = [_row(index, primitive) for index, primitive in enumerate(PRIMITIVE_VOCABULARY)]
    sidecars = {role: [] for role in DEVELOPMENT_ROLES}
    sidecars["train"] = [_sidecar(row) for row in rows]
    sidecars["g2_evaluation"] = []
    with pytest.raises(RawSupervisionPlanError, match="exactly development roles"):
        plan_development_raw_supervision(
            rows,
            sidecars,
            input_bindings={},
            access_ledger={},
            enforce_frozen_counts=False,
        )


def test_frozen_metadata_plan_matches_preregistered_population() -> None:
    plan = load_frozen_development_metadata(ROOT)
    assert plan.value["pair_counts"] == {
        "checkpoint_selection": 495,
        "probability_calibration": 415,
        "train": 4262,
    }
    assert plan.value["unique_endpoint_counts"] == {
        "checkpoint_selection": 924,
        "probability_calibration": 759,
        "train": 7777,
    }
    assert len(plan.pairs) == 5172
    assert len(plan.endpoints) == 9460
    assert plan.value["endpoint_instance_count"] == 10344
    ledger = plan.value["access_ledger"]
    assert ledger["g2_row_metadata_rows_read_for_exclusion"] == 469
    assert ledger["sidecar_g2_byte_opens"] == 0
    assert ledger["g2_geometry_or_label_payload_opens"] == 0
    assert ledger["rgb_byte_opens"] == ledger["rgb_decodes"] == 0


def test_source_inventory_retains_only_planned_scenes_without_dereference() -> None:
    plan = _synthetic_plan()
    source_row = {
        "scene_id": "scene_a",
        "family": "family_a",
        "split": "train",
        "frames_jsonl_path": str(ROOT / ".generated/metadata/scene_a/frames.jsonl"),
        "scene_manifest_path": str(ROOT / ".generated/metadata/scene_a/manifest.json"),
        "render_plan_path": str(ROOT / ".generated/metadata/scene_a/plan.json"),
        "render_summary_path": str(ROOT / ".generated/metadata/scene_a/summary.json"),
        "hashes": {
            "frames_jsonl_file_sha256": _sha("frames-a"),
            "scene_manifest_file_sha256": _sha("manifest-file-a"),
            "scene_manifest_sha256": _sha("manifest-content-a"),
            "render_plan_file_sha256": _sha("plan-a"),
            "render_summary_file_sha256": _sha("summary-a"),
        },
    }
    g2_row = copy.deepcopy(source_row)
    g2_row["scene_id"] = "scene_g2"
    inventory = plan_development_source_inventory(
        plan,
        [source_row, g2_row],
        repo_root=ROOT,
        enforce_frozen_hashes=False,
    )
    assert [row["scene_id"] for row in inventory.records] == ["scene_a"]
    assert inventory.access_ledger["g2_source_records_read_for_exclusion"] == 1
    assert inventory.access_ledger["source_frames_payload_opens"] == 0
    assert inventory.access_ledger["source_scene_manifest_payload_opens"] == 0


def test_source_inventory_rejects_missing_planned_scene() -> None:
    with pytest.raises(RawSupervisionPlanError, match="lacks a development scene"):
        plan_development_source_inventory(
            _synthetic_plan(),
            [],
            repo_root=ROOT,
            enforce_frozen_hashes=False,
        )


def test_frozen_source_inventory_matches_precommitted_metadata_reduction() -> None:
    inventory = load_frozen_development_source_inventory(ROOT)
    assert len(inventory.records) == 88
    assert inventory.hashes == {
        "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
        "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
        "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
        "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
        "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
    }
    ledger = inventory.access_ledger
    assert ledger["source_index_metadata_rows_read"] == 96
    assert ledger["development_source_records_retained"] == 88
    assert ledger["g2_source_records_read_for_exclusion"] == 8
    assert ledger["g2_payload_opens"] == 0
