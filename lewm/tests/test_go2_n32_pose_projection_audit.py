from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from lewm.benchmarks.go2_n32_pose_projection_audit import (
    FAMILIES,
    ProjectionComparison,
    QUERY_COUNT,
    compare_projection,
    ordering_decision,
    project_registered_queries,
    reconstruct_yaw_aligned_camera,
    registered_camera_geometry,
)
from lewm.models.categorical_radial_perception import (
    CAMERA_XYZ_BODY_M,
    _registered_projective_geometry,
)
from lewm.benchmarks.go2_categorical_radial_factorization import (
    build_radial_factorization,
)
from scripts import audit_go2_n32_pose_projection as audit
from scripts import extract_go2_n32_pose_fit_panel as extractor


def _synthetic_record(
    scene_root,
    *,
    frame_index: int = 17,
    env_index: int = 3,
    timestamp_ns: int = 900,
) -> dict:
    return {
        "family": "open_obstacle_field",
        "physical_dataset_role": "train",
        "scene_id": "open_obstacle_field_synthetic",
        "global_row": 1,
        "side": "current",
        "frame_index": frame_index,
        "env_index": env_index,
        "timestamp_ns": timestamp_ns,
        "episode_id": "2",
        "reset_count": 4,
        "episode_step": 11,
        "image_path_metadata_only": str(scene_root / "rgb/frame.png"),
        "image_sha256_commitment_only": "a" * 64,
        "panel_row_index": 0,
    }


def _synthetic_source_frame(record: dict) -> dict:
    return {
        "frame_index": record["frame_index"],
        "env_index": record["env_index"],
        "timestamp_ns": record["timestamp_ns"],
        "episode": {
            "split": "train",
            "episode_id": int(record["episode_id"]),
            "reset_count": record["reset_count"],
            "episode_step": record["episode_step"],
        },
        "base_pose_world": {"position": {"x": 1.0, "y": 2.0, "z": 0.3}},
        "base_rpy_rad": {"yaw": 0.4},
        "camera_pose_world": {
            "position": [1.2, 2.1, 0.4],
            "lookat": [2.2, 2.1, 0.4],
            "up": [0.0, 0.0, 1.0],
        },
    }


def _synthetic_summary(record: dict, source_path, source_sha: str) -> dict:
    return {
        "split": "train",
        "family": record["family"],
        "scene_id": record["scene_id"],
        "render_status": "complete",
        "g2_model_outputs_opened": False,
        "source": {
            "frames_jsonl": {
                "path": str(source_path),
                "sha256": source_sha,
            }
        },
        "rendered_frames": [
            {
                "frame_index": record["frame_index"],
                "env_index": record["env_index"],
                "timestamp_ns": record["timestamp_ns"],
                "image_sha256": record["image_sha256_commitment_only"],
            }
        ],
    }


def _camera(
    *,
    origin: tuple[float, float, float] = CAMERA_XYZ_BODY_M,
    forward: tuple[float, float, float] = (1.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    position = np.asarray(origin, dtype=np.float64)
    return reconstruct_yaw_aligned_camera(
        base_position_world=(0.0, 0.0, 0.0),
        base_yaw_rad=0.0,
        camera_position_world=position,
        camera_lookat_world=position + np.asarray(forward, dtype=np.float64),
        camera_up_world=up,
    )


def test_registered_level_camera_has_exact_zero_projection_mismatch() -> None:
    comparison = compare_projection(_camera())

    assert comparison.token_displacements.shape[0] > 0
    assert np.array_equal(
        comparison.token_displacements,
        np.zeros_like(comparison.token_displacements),
    )
    assert comparison.metrics["token_displacement"] == {
        "count": comparison.token_displacements.size,
        "p50_token": 0.0,
        "p95_token": 0.0,
        "maximum_token": 0.0,
        "fraction_ge_0_5_token": 0.0,
    }
    assert comparison.metrics["validity_flip_count"] == 0
    assert comparison.metrics["fixed_valid_query_count"] == comparison.metrics[
        "actual_valid_query_count"
    ]


def test_pure_registered_projection_is_bit_exact_to_width24_buffers() -> None:
    grid, validity = project_registered_queries(registered_camera_geometry())
    model_grid, model_validity, _coordinates = _registered_projective_geometry(
        build_radial_factorization()
    )

    assert np.array_equal(grid.astype(np.float32), model_grid.numpy())
    assert np.array_equal(validity, model_validity.numpy())


@pytest.mark.parametrize(
    "camera",
    (
        _camera(origin=(0.326, 0.45, 0.18)),
        _camera(forward=(1.0, 0.0, 0.35)),
    ),
)
def test_translation_and_tilt_have_nonzero_displacement_and_validity_changes(
    camera,
) -> None:
    comparison = compare_projection(camera)

    assert comparison.metrics["token_displacement"]["p50_token"] > 0.0
    assert comparison.metrics["token_displacement"]["maximum_token"] > 0.5
    assert comparison.metrics["validity_flip_count"] > 0
    assert comparison.metrics["validity_flip_rate"] > 0.0


def test_world_yaw_transform_recovers_camera_in_yaw_aligned_base_axes() -> None:
    yaw = 1.1
    base = np.asarray((2.0, -3.0, 0.7), dtype=np.float64)
    axes = np.asarray(
        (
            (math.cos(yaw), -math.sin(yaw), 0.0),
            (math.sin(yaw), math.cos(yaw), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    local_origin = np.asarray(CAMERA_XYZ_BODY_M)
    camera_world = base + axes @ local_origin
    camera = reconstruct_yaw_aligned_camera(
        base_position_world=base,
        base_yaw_rad=yaw,
        camera_position_world=camera_world,
        camera_lookat_world=camera_world + axes @ np.asarray((1.0, 0.0, 0.0)),
        camera_up_world=(0.0, 0.0, 1.0),
    )

    assert np.allclose(camera.origin_xyz, local_origin, rtol=0.0, atol=1e-15)
    assert np.allclose(camera.forward_xyz, (1.0, 0.0, 0.0), atol=1e-15)
    assert np.allclose(camera.left_xyz, (0.0, 1.0, 0.0), atol=1e-15)
    assert np.allclose(camera.up_xyz, (0.0, 0.0, 1.0), atol=1e-15)
    assert camera.forward_pitch_rad == pytest.approx(0.0, abs=1e-15)
    assert camera.up_roll_rad == pytest.approx(0.0, abs=1e-15)


def test_wrong_authorization_fails_before_any_metadata_access() -> None:
    ledger = audit.new_access_ledger()

    with pytest.raises(PermissionError, match="not authorized"):
        audit.run_authoritative_audit(authorization="0" * 64, ledger=ledger)

    assert ledger["authorization_checked"] is True
    assert ledger["authorized"] is False
    for bucket in ledger["metadata"].values():
        assert bucket["unique_files"] == 0
        assert bucket["hash_opens"] == 0
        assert bucket["parse_opens"] == 0
    assert all(value == 0 for value in ledger["forbidden"].values())


@pytest.mark.parametrize(
    "obsolete_hash",
    (
        audit.BINDING_SHA256,
        audit.AMENDMENT_SHA256,
        audit.SUPERSEDED_SCOPE_AMENDMENT_SHA256,
    ),
)
def test_older_governing_hashes_no_longer_authorize_command(
    obsolete_hash,
) -> None:
    ledger = audit.new_access_ledger()

    with pytest.raises(PermissionError, match="role-namespace amendment"):
        audit._require_authorization(obsolete_hash, ledger)

    assert ledger["authorization_checked"] is True
    assert ledger["authorized"] is False


def test_audit_runner_exposes_only_the_fit_panel_input() -> None:
    ledger = audit.new_access_ledger()

    assert not hasattr(audit, "PANEL_PATH")
    assert audit.FIT_PANEL_PATH.name == "fit_panel.json"
    assert "panel" not in ledger["metadata"]
    assert set(ledger["metadata"]) >= {
        "binding",
        "fit_panel_amendment",
        "superseded_scope_amendment",
        "role_namespace_amendment",
        "fit_panel",
    }


def test_frozen_legacy_split_map_and_frame_counts_are_exact() -> None:
    expected_hashes = {
        "scene_074f19f0608afca2/summary.json": "7a5d3b1e6ff5a8acb914ae5226326084c2b951517c110ffc19d7a99945fe0413",
        "scene_142dbd9b0428f16f/summary.json": "995e192cc1830f32bd2dc6d358da91f5bdaec48bd585ac2dadecc45517cbd2b0",
        "scene_4931dab75d2ceee8/summary.json": "7800d0d6a14ea54b9970d1dac36472446cd525af8c893736ebe1c4b4bf57cc23",
        "scene_49db95fc9ed0ce8f/summary.json": "80a035ceecf56f2c668fed3ab1dbabeeca181cb2886fedafa7116ec26bc0566d",
        "scene_4af4d0549179a705/summary.json": "bcb3866fe141c0c629368eefee8e228630ca8f3b30e1c2810b34e68fd61347b4",
        "scene_7239d51aced24ee3/summary.json": "5c6785479b9a302fcffb1d7532e450af10d2e2625a030eff872edf22b23aef6f",
        "scene_7f390beda8f5070f/summary.json": "2dc1f874130cb733be4f28eccae3359aac7bdc4e2947718391182ad651d027e7",
        "scene_9ff98ead4f1a2e96/summary.json": "203ffca9205f68dc74e6135718d3fec4bfb55e9c841bf7a4eb49964930309cc0",
        "scene_a81215e4d326a2a2/summary.json": "7b9c5dff08be0876327f8b625d225e4b1729320f98b9ccb1efcbd1c68cc2e3c1",
        "scene_b1355439db03d8f8/summary.json": "d21cd06b202422ecce81c009c08b13ab4e92be86bdc93f6571e69ac265f33fa9",
        "scene_b748962d390baeca/summary.json": "a3a90172486dc08f3e7a1728da71e43ae224aefddc22ba32e1de5b4fa6ab7f38",
        "scene_b75bb34744434970/summary.json": "64bcf8f57c55cb3456f6dd04be23bbdc417865b2ee8dbad914b5eaa387d61b6b",
        "scene_bc5a05ec9fce8d9c/summary.json": "41377a7619560162b7fd4453ca302321d2f5f22aee1a8c7397ff32626bbb1a92",
        "scene_c60650f53aaae4a6/summary.json": "be319a4b1a6e456367c3a6b4d9eee5059380ef83ebe720416b7f292a959c2d6e",
        "scene_cfcadb2bd44cce85/summary.json": "fa5a9049889a10700cd678fea78ecfb6f91545403ebfdfd304d1dc59a4b6d40a",
        "scene_d8b06cdfb1f739ed/summary.json": "6f06ee751ec3a26de741bdafcf39cb044e49734cb5a2ab1103ab2834e3edf3c2",
        "scene_ddc88df212918857/summary.json": "7b1deec174715696d4a3dd653610886e1244edfa993a8c0dc0e91176b728488f",
        "scene_df1c6b34503f2ae1/summary.json": "deed15024342195754b9022522c048624ab09a1d55e2727f615822d5b6f658e8",
        "scene_e0c2fe611e747d90/summary.json": "df2fde293612833f00f15a25a8c81c799e15e4674f5ad7f29a0d7ea06e9fd341",
        "scene_ebc33be3e6a87264/summary.json": "12b5825f4dc2388631190cc80dd42f9cea1bbbbf002f666f12ca53ddde704a35",
    }
    expected_splits = {
        "scene_074f19f0608afca2/summary.json": "train",
        "scene_142dbd9b0428f16f/summary.json": "test_hard",
        "scene_4931dab75d2ceee8/summary.json": "train",
        "scene_49db95fc9ed0ce8f/summary.json": "train",
        "scene_4af4d0549179a705/summary.json": "train",
        "scene_7239d51aced24ee3/summary.json": "test_id",
        "scene_7f390beda8f5070f/summary.json": "train",
        "scene_9ff98ead4f1a2e96/summary.json": "train",
        "scene_a81215e4d326a2a2/summary.json": "train",
        "scene_b1355439db03d8f8/summary.json": "val",
        "scene_b748962d390baeca/summary.json": "train",
        "scene_b75bb34744434970/summary.json": "test_id",
        "scene_bc5a05ec9fce8d9c/summary.json": "val",
        "scene_c60650f53aaae4a6/summary.json": "train",
        "scene_cfcadb2bd44cce85/summary.json": "train",
        "scene_d8b06cdfb1f739ed/summary.json": "train",
        "scene_ddc88df212918857/summary.json": "train",
        "scene_df1c6b34503f2ae1/summary.json": "train",
        "scene_e0c2fe611e747d90/summary.json": "train",
        "scene_ebc33be3e6a87264/summary.json": "train",
    }

    assert audit.EXPECTED_LEGACY_SOURCE_SPLIT == expected_splits
    assert audit.EXPECTED_SUMMARY_SHA256 == expected_hashes
    assert set(expected_hashes) == set(expected_splits)
    assert len(audit.EXPECTED_SUMMARY_SHA256) == 20
    assert audit.EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS == {
        "train": 244,
        "test_hard": 14,
        "test_id": 32,
        "val": 30,
    }
    assert sum(audit.EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS.values()) == 320


def test_frozen_fit_only_panel_validates_to_exactly_320_records() -> None:
    payload = json.loads(audit.FIT_PANEL_PATH.read_text())

    records = audit._validate_fit_panel(
        payload,
        panel_file_sha256=audit.FIT_PANEL_FILE_SHA256,
    )

    assert len(records) == 320
    assert {record["side"] for record in records} == {"current", "next"}
    assert {record["physical_dataset_role"] for record in records} == {"train"}
    assert all(record["family"] in audit.FAMILIES for record in records)
    assert {
        family: sum(record["family"] == family for record in records)
        for family in audit.FAMILIES
    } == {family: 64 for family in audit.FAMILIES}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(schema="wrong"), "schema"),
        (
            lambda payload: payload.update(amendment_sha256="0" * 64),
            "amendment",
        ),
        (
            lambda payload: payload["source_panel"].update(path="/wrong/panel.json"),
            "source lineage",
        ),
        (
            lambda payload: payload["access_ledger"].update(
                non_fit_rows_copied=1
            ),
            "extraction access",
        ),
        (
            lambda payload: payload.update(panels={"forbidden": []}),
            "schema keys",
        ),
    ),
)
def test_fit_only_panel_envelope_rejects_adversarial_changes(
    mutation, message, monkeypatch
) -> None:
    payload = json.loads(audit.FIT_PANEL_PATH.read_text())
    mutation(payload)
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    changed_content_sha = audit.canonical_json_sha256(content)
    payload["content_sha256"] = changed_content_sha
    monkeypatch.setattr(audit, "FIT_PANEL_CONTENT_SHA256", changed_content_sha)

    with pytest.raises(ValueError, match=message):
        audit._validate_fit_panel(
            payload,
            panel_file_sha256=audit.FIT_PANEL_FILE_SHA256,
        )


def test_fit_only_panel_rejects_unbound_file_hash() -> None:
    payload = json.loads(audit.FIT_PANEL_PATH.read_text())

    with pytest.raises(ValueError, match="file SHA-256"):
        audit._validate_fit_panel(payload, panel_file_sha256="0" * 64)


def _complete_synthetic_audit_ledger() -> dict:
    ledger = audit.new_access_ledger()
    ledger["authorization_checked"] = True
    ledger["authorized"] = True
    ledger["metadata"]["binding"] = {
        "unique_files": 1,
        "hash_opens": 2,
        "parse_opens": 0,
    }
    ledger["metadata"]["fit_panel_amendment"] = {
        "unique_files": 1,
        "hash_opens": 2,
        "parse_opens": 0,
    }
    ledger["metadata"]["superseded_scope_amendment"] = {
        "unique_files": 1,
        "hash_opens": 2,
        "parse_opens": 0,
    }
    ledger["metadata"]["role_namespace_amendment"] = {
        "unique_files": 1,
        "hash_opens": 2,
        "parse_opens": 0,
    }
    ledger["metadata"]["fit_panel"] = {
        "unique_files": 1,
        "hash_opens": 2,
        "parse_opens": 1,
    }
    summary_count = len(audit.EXPECTED_SUMMARY_SHA256)
    ledger["metadata"]["scene_summaries"] = {
        "unique_files": summary_count,
        "hash_opens": 2 * summary_count,
        "parse_opens": summary_count,
    }
    ledger["metadata"]["source_frames_jsonl"] = {
        "unique_files": 2,
        "hash_opens": 4,
        "parse_opens": 2,
        "json_records_scanned": 400,
        "requested_records": 320,
        "matched_records": 320,
    }
    ledger["metadata"]["source_code"] = {
        "unique_files": len(audit.SOURCE_PATHS),
        "hash_opens": 2 * len(audit.SOURCE_PATHS),
        "parse_opens": 0,
    }
    ledger["role_namespace"] = {
        "physical_dataset_role_train_frame_records": 320,
        "physical_dataset_nontrain_frame_records": 0,
        "legacy_source_split_frame_records": dict(
            audit.EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS
        ),
        "legacy_source_split_used_for_inclusion": False,
    }
    return ledger


def test_access_ledger_reconciles_every_exact_bucket_and_count() -> None:
    ledger = _complete_synthetic_audit_ledger()

    reconciliation = audit._validate_access_ledger(
        ledger,
        source_frames_file_count=2,
    )

    assert reconciliation["exact_bucket_and_count_reconciliation_passes"] is True
    assert reconciliation["requested_source_record_count"] == 320
    assert reconciliation["matched_source_record_count"] == 320
    assert reconciliation["physical_dataset_role_train_frame_count"] == 320
    assert reconciliation["legacy_source_split_frame_counts"] == {
        "train": 244,
        "test_hard": 14,
        "test_id": 32,
        "val": 30,
    }
    assert reconciliation["metadata_bucket_count"] == len(ledger["metadata"])


def test_access_ledger_rejects_extra_bucket_and_missing_forbidden_counter() -> None:
    extra_bucket = _complete_synthetic_audit_ledger()
    extra_bucket["metadata"]["monolithic_panel"] = {
        "unique_files": 1,
        "hash_opens": 1,
        "parse_opens": 1,
    }
    with pytest.raises(ValueError, match="bucket set"):
        audit._validate_access_ledger(extra_bucket, source_frames_file_count=2)

    missing_counter = _complete_synthetic_audit_ledger()
    missing_counter["forbidden"].pop("rgb_byte_opens")
    with pytest.raises(PermissionError, match="forbidden"):
        audit._validate_access_ledger(missing_counter, source_frames_file_count=2)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda ledger: ledger["role_namespace"][
            "legacy_source_split_frame_records"
        ].update(test_id=31),
        lambda ledger: ledger["role_namespace"].update(
            physical_dataset_nontrain_frame_records=1
        ),
        lambda ledger: ledger["role_namespace"].update(
            legacy_source_split_used_for_inclusion=True
        ),
    ),
)
def test_access_ledger_rejects_role_namespace_count_or_semantic_mismatch(
    mutation,
) -> None:
    ledger = _complete_synthetic_audit_ledger()
    mutation(ledger)

    with pytest.raises(ValueError, match="role-namespace"):
        audit._validate_access_ledger(ledger, source_frames_file_count=2)


def test_source_frame_provenance_requires_exact_key_episode_and_train_role() -> None:
    record = {
        "frame_index": 17,
        "env_index": 3,
        "timestamp_ns": 900,
        "episode_id": "2",
        "reset_count": 4,
        "episode_step": 11,
    }
    frame = {
        "frame_index": 17,
        "env_index": 3,
        "timestamp_ns": 900,
        "episode": {
            "split": "train",
            "episode_id": 2,
            "reset_count": 4,
            "episode_step": 11,
        },
        "base_pose_world": {"position": {"x": 1.0, "y": 2.0, "z": 0.3}},
        "base_rpy_rad": {"yaw": 0.4},
        "camera_pose_world": {
            "position": [1.2, 2.1, 0.4],
            "lookat": [2.2, 2.1, 0.4],
            "up": [0.0, 0.0, 1.0],
        },
    }

    extracted = audit._validate_and_extract_source_frame(frame, record)
    assert extracted["base_yaw_rad"] == pytest.approx(0.4)

    wrong_timestamp = json.loads(json.dumps(frame))
    wrong_timestamp["timestamp_ns"] = 901
    with pytest.raises(ValueError, match="timestamp"):
        audit._validate_and_extract_source_frame(wrong_timestamp, record)

    legacy_test = json.loads(json.dumps(frame))
    legacy_test["episode"]["split"] = "test_id"
    extracted_test = audit._validate_and_extract_source_frame(
        legacy_test,
        record,
        expected_legacy_source_split="test_id",
    )
    assert extracted_test["base_yaw_rad"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="legacy source split"):
        audit._validate_and_extract_source_frame(legacy_test, record)


def _summary_fixture(tmp_path, monkeypatch):
    summary_root = tmp_path / "summaries"
    scene_root = summary_root / "scene_synthetic"
    (scene_root / "rgb").mkdir(parents=True)
    frames_root = tmp_path / "rollout"
    source_path = frames_root / "train/family/scene/frames.jsonl"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}\n")
    source_sha = audit.hashlib.sha256(source_path.read_bytes()).hexdigest()
    record = _synthetic_record(scene_root)
    summary_path = scene_root / "summary.json"
    summary = _synthetic_summary(record, source_path, source_sha)
    monkeypatch.setattr(audit, "SUMMARY_ROOT", summary_root)
    monkeypatch.setattr(audit, "FRAMES_ROOT", frames_root)
    summary_key = str(summary_path.relative_to(summary_root))
    monkeypatch.setattr(
        audit,
        "EXPECTED_LEGACY_SOURCE_SPLIT",
        {summary_key: "train"},
    )
    return summary, summary_path, source_path, source_sha, record


def test_validate_summary_accepts_exact_source_and_rendered_commitments(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, source_path, source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )

    actual_path, actual_sha, legacy_split = audit._validate_summary(
        summary,
        summary_path=summary_path,
        records=[record],
    )

    assert actual_path == source_path.resolve()
    assert actual_sha == source_sha
    assert legacy_split == "train"


def test_validate_summary_accepts_frozen_nontrain_legacy_provenance(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, source_path, _source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    legacy_source = audit.FRAMES_ROOT / "test_hard/family/scene/frames.jsonl"
    legacy_source.parent.mkdir(parents=True)
    legacy_source.write_text("{}\n")
    legacy_sha = audit.hashlib.sha256(legacy_source.read_bytes()).hexdigest()
    summary_key = str(summary_path.relative_to(audit.SUMMARY_ROOT))
    summary["split"] = "test_hard"
    summary["source"]["frames_jsonl"] = {
        "path": str(legacy_source),
        "sha256": legacy_sha,
    }
    monkeypatch.setattr(
        audit,
        "EXPECTED_LEGACY_SOURCE_SPLIT",
        {summary_key: "test_hard"},
    )

    actual_path, actual_sha, legacy_split = audit._validate_summary(
        summary,
        summary_path=summary_path,
        records=[record],
    )

    assert actual_path == legacy_source.resolve()
    assert actual_sha == legacy_sha
    assert legacy_split == "test_hard"
    assert source_path != actual_path


def test_validate_summary_rejects_legacy_split_map_mismatch(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, _source_path, _source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    summary["split"] = "test_id"

    with pytest.raises(ValueError, match="legacy source split changed"):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])


def test_validate_summary_rejects_source_outside_root_and_wrong_filename(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, _source_path, source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    outside = tmp_path / "outside/frames.jsonl"
    outside.parent.mkdir()
    outside.write_text("{}\n")
    summary["source"]["frames_jsonl"]["path"] = str(outside)
    with pytest.raises(PermissionError, match="escapes"):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])

    wrong_name = audit.FRAMES_ROOT / "train/family/scene/source.jsonl"
    wrong_name.write_text("{}\n")
    summary["source"]["frames_jsonl"] = {
        "path": str(wrong_name),
        "sha256": source_sha,
    }
    with pytest.raises(PermissionError, match="named frames.jsonl"):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])


def test_validate_summary_rejects_malformed_source_commitment(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, _source_path, _source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    summary["source"]["frames_jsonl"]["sha256"] = "not-a-sha"

    with pytest.raises(ValueError, match="SHA-256 is malformed"):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda summary: summary.update(rendered_frames=[]), "exactly once"),
        (
            lambda summary: summary["rendered_frames"].append(
                copy.deepcopy(summary["rendered_frames"][0])
            ),
            "exactly once",
        ),
        (
            lambda summary: summary["rendered_frames"][0].update(timestamp_ns=901),
            "timestamp",
        ),
        (
            lambda summary: summary["rendered_frames"][0].update(
                image_sha256="b" * 64
            ),
            "image commitment",
        ),
    ),
)
def test_validate_summary_rejects_rendered_record_mismatch(
    tmp_path, monkeypatch, mutation, message
) -> None:
    summary, summary_path, _source_path, _source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    mutation(summary)

    with pytest.raises(ValueError, match=message):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])


def test_validate_summary_rejects_rendered_image_metadata_outside_scene(
    tmp_path, monkeypatch
) -> None:
    summary, summary_path, _source_path, _source_sha, record = _summary_fixture(
        tmp_path, monkeypatch
    )
    record["image_path_metadata_only"] = str(tmp_path / "other/rgb/frame.png")

    with pytest.raises(PermissionError, match="escapes its V04 scene"):
        audit._validate_summary(summary, summary_path=summary_path, records=[record])


def test_scan_requested_source_frames_matches_exactly_once(tmp_path) -> None:
    scene_root = tmp_path / "scene"
    record = _synthetic_record(scene_root)
    frame = _synthetic_source_frame(record)
    source_path = tmp_path / "frames.jsonl"
    source_path.write_text(json.dumps(frame) + "\n")
    ledger = audit.new_access_ledger()

    result = audit._scan_requested_source_frames(
        source_path,
        [record],
        ledger=ledger,
    )

    assert set(result) == {(record["frame_index"], record["env_index"])}
    assert result[(17, 3)]["base_yaw_rad"] == pytest.approx(0.4)
    source_access = ledger["metadata"]["source_frames_jsonl"]
    assert source_access["parse_opens"] == 1
    assert source_access["json_records_scanned"] == 1
    assert source_access["requested_records"] == 1
    assert source_access["matched_records"] == 1


def test_scan_requested_source_frames_binds_summary_legacy_split(tmp_path) -> None:
    record = _synthetic_record(tmp_path / "scene")
    frame = _synthetic_source_frame(record)
    frame["episode"]["split"] = "val"
    source_path = tmp_path / "frames.jsonl"
    source_path.write_text(json.dumps(frame) + "\n")

    result = audit._scan_requested_source_frames(
        source_path,
        [record],
        ledger=audit.new_access_ledger(),
        expected_legacy_source_split="val",
    )
    assert set(result) == {(17, 3)}

    with pytest.raises(ValueError, match="legacy source split changed"):
        audit._scan_requested_source_frames(
            source_path,
            [record],
            ledger=audit.new_access_ledger(),
            expected_legacy_source_split="train",
        )


@pytest.mark.parametrize("copies", (0, 2))
def test_scan_requested_source_frames_rejects_missing_and_duplicate(
    tmp_path, copies
) -> None:
    record = _synthetic_record(tmp_path / "scene")
    frame = _synthetic_source_frame(record)
    source_path = tmp_path / "frames.jsonl"
    source_path.write_text("".join(json.dumps(frame) + "\n" for _ in range(copies)))

    with pytest.raises(ValueError, match="did not match exactly once"):
        audit._scan_requested_source_frames(
            source_path,
            [record],
            ledger=audit.new_access_ledger(),
        )


def _configure_synthetic_runner_until_source_scan(tmp_path, monkeypatch):
    binding_path = tmp_path / "binding.md"
    amendment_path = tmp_path / "amendment.md"
    train_source_scope_path = tmp_path / "train-source-scope.md"
    role_namespace_path = tmp_path / "role-namespace.md"
    fit_panel_path = tmp_path / "fit_panel.json"
    binding_path.write_text("binding\n")
    amendment_path.write_text("amendment\n")
    train_source_scope_path.write_text("train source scope\n")
    role_namespace_path.write_text("role namespace\n")
    fit_panel_path.write_text("{}\n")

    summary_root = tmp_path / "summaries"
    scene_root = summary_root / "scene_synthetic"
    (scene_root / "rgb").mkdir(parents=True)
    frames_root = tmp_path / "rollout"
    source_path = frames_root / "train/family/scene/frames.jsonl"
    source_path.parent.mkdir(parents=True)
    record = _synthetic_record(scene_root)
    source_frame = _synthetic_source_frame(record)
    source_path.write_text(json.dumps(source_frame) + "\n")
    source_sha = audit.hashlib.sha256(source_path.read_bytes()).hexdigest()
    summary_path = scene_root / "summary.json"
    summary_path.write_text(
        json.dumps(_synthetic_summary(record, source_path, source_sha)) + "\n"
    )

    def file_sha(path):
        return audit.hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(audit, "BINDING_PATH", binding_path)
    monkeypatch.setattr(audit, "AMENDMENT_PATH", amendment_path)
    monkeypatch.setattr(
        audit,
        "SUPERSEDED_SCOPE_AMENDMENT_PATH",
        train_source_scope_path,
    )
    monkeypatch.setattr(
        audit,
        "ROLE_NAMESPACE_AMENDMENT_PATH",
        role_namespace_path,
    )
    monkeypatch.setattr(audit, "FIT_PANEL_PATH", fit_panel_path)
    monkeypatch.setattr(audit, "OUTPUT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(audit, "SUMMARY_ROOT", summary_root)
    monkeypatch.setattr(audit, "FRAMES_ROOT", frames_root)
    monkeypatch.setattr(audit, "BINDING_SHA256", file_sha(binding_path))
    monkeypatch.setattr(audit, "AMENDMENT_SHA256", file_sha(amendment_path))
    monkeypatch.setattr(
        audit,
        "SUPERSEDED_SCOPE_AMENDMENT_SHA256",
        file_sha(train_source_scope_path),
    )
    monkeypatch.setattr(
        audit,
        "ROLE_NAMESPACE_AMENDMENT_SHA256",
        file_sha(role_namespace_path),
    )
    monkeypatch.setattr(audit, "FIT_PANEL_FILE_SHA256", file_sha(fit_panel_path))
    monkeypatch.setattr(audit, "SOURCE_PATHS", ())
    summary_key = str(summary_path.relative_to(summary_root))
    monkeypatch.setattr(
        audit,
        "EXPECTED_SUMMARY_SHA256",
        {summary_key: file_sha(summary_path)},
    )
    monkeypatch.setattr(audit, "EXPECTED_SUMMARY_COUNT", 1)
    monkeypatch.setattr(
        audit,
        "EXPECTED_LEGACY_SOURCE_SPLIT",
        {summary_key: "train"},
    )
    monkeypatch.setattr(
        audit,
        "EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS",
        {"train": 1},
    )
    monkeypatch.setattr(
        audit,
        "_validate_fit_panel",
        lambda _payload, *, panel_file_sha256: [record],
    )
    return summary_path, source_path


def test_runner_rejects_summary_mutation_between_parse_and_rehash(
    tmp_path, monkeypatch
) -> None:
    summary_path, _source_path = _configure_synthetic_runner_until_source_scan(
        tmp_path, monkeypatch
    )
    original_load_json = audit._load_json

    def load_then_mutate(path, *, ledger, bucket):
        payload = original_load_json(path, ledger=ledger, bucket=bucket)
        if path == summary_path.resolve():
            summary_path.write_text(summary_path.read_text() + " ")
        return payload

    monkeypatch.setattr(audit, "_load_json", load_then_mutate)

    with pytest.raises(ValueError, match="V04 summary changed"):
        audit.run_authoritative_audit(
            authorization=audit.ROLE_NAMESPACE_AMENDMENT_SHA256
        )


def test_runner_rejects_source_mutation_between_scan_and_rehash(
    tmp_path, monkeypatch
) -> None:
    _summary_path, source_path = _configure_synthetic_runner_until_source_scan(
        tmp_path, monkeypatch
    )
    original_scan = audit._scan_requested_source_frames

    def scan_then_mutate(
        path, records, *, ledger, expected_legacy_source_split="train"
    ):
        payload = original_scan(
            path,
            records,
            ledger=ledger,
            expected_legacy_source_split=expected_legacy_source_split,
        )
        source_path.write_text(source_path.read_text() + " ")
        return payload

    monkeypatch.setattr(audit, "_scan_requested_source_frames", scan_then_mutate)

    with pytest.raises(ValueError, match="source frames JSONL changed"):
        audit.run_authoritative_audit(
            authorization=audit.ROLE_NAMESPACE_AMENDMENT_SHA256
        )


def test_runner_rejects_source_hash_outside_summary_commitment(
    tmp_path, monkeypatch
) -> None:
    summary_path, _source_path = _configure_synthetic_runner_until_source_scan(
        tmp_path, monkeypatch
    )
    summary = json.loads(summary_path.read_text())
    summary["source"]["frames_jsonl"]["sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n")
    summary_key = str(summary_path.relative_to(audit.SUMMARY_ROOT))
    monkeypatch.setattr(
        audit,
        "EXPECTED_SUMMARY_SHA256",
        {
            summary_key: audit.hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
        },
    )

    with pytest.raises(ValueError, match="differs from its V04 commitment"):
        audit.run_authoritative_audit(
            authorization=audit.ROLE_NAMESPACE_AMENDMENT_SHA256
        )


@pytest.mark.parametrize(
    "label",
    (
        "original audit binding",
        "fit-panel amendment",
        "superseded train-source scope amendment",
        "role-namespace amendment",
    ),
)
def test_governing_document_mutation_between_hashes_fails_closed(label) -> None:
    with pytest.raises(ValueError, match=f"{label} changed"):
        audit._require_unchanged_hash("a" * 64, "b" * 64, label=label)


def test_runner_completes_all_320_physical_train_records_with_legacy_test_hard(
    tmp_path, monkeypatch
) -> None:
    binding_path = tmp_path / "binding.md"
    fit_amendment_path = tmp_path / "fit-amendment.md"
    superseded_path = tmp_path / "superseded.md"
    role_path = tmp_path / "role.md"
    fit_panel_path = tmp_path / "fit-panel.json"
    for path, text in (
        (binding_path, "binding"),
        (fit_amendment_path, "fit amendment"),
        (superseded_path, "superseded"),
        (role_path, "role namespace"),
    ):
        path.write_text(text + "\n")
    fit_panel_path.write_text(json.dumps({"source_panel": {"synthetic": True}}))

    summary_root = tmp_path / "summaries"
    frames_root = tmp_path / "rollout"
    records = []
    expected_hashes = {}
    expected_splits = {}
    frame_index = 0
    panel_row_index = 0
    for family_index, family in enumerate(FAMILIES):
        scene_root = summary_root / f"scene_{family_index}"
        (scene_root / "rgb").mkdir(parents=True)
        source_path = (
            frames_root
            / f"test_hard/{family}/scene_{family_index}/frames.jsonl"
        )
        source_path.parent.mkdir(parents=True)
        scene_records = []
        source_frames = []
        rendered_frames = []
        for family_frame_index in range(64):
            side = "current" if family_frame_index % 2 == 0 else "next"
            record = {
                "family": family,
                "physical_dataset_role": "train",
                "scene_id": f"{family}_synthetic",
                "global_row": panel_row_index,
                "side": side,
                "frame_index": frame_index,
                "env_index": 0,
                "timestamp_ns": frame_index + 1,
                "episode_id": "2",
                "reset_count": 4,
                "episode_step": family_frame_index + 1,
                "image_path_metadata_only": str(
                    scene_root / f"rgb/frame_{frame_index}.png"
                ),
                "image_sha256_commitment_only": "a" * 64,
                "panel_row_index": panel_row_index,
            }
            scene_records.append(record)
            records.append(record)
            source_frame = _synthetic_source_frame(record)
            source_frame["episode"]["split"] = "test_hard"
            source_frames.append(source_frame)
            rendered_frames.append(
                {
                    "frame_index": frame_index,
                    "env_index": 0,
                    "timestamp_ns": frame_index + 1,
                    "image_sha256": "a" * 64,
                }
            )
            frame_index += 1
            if side == "next":
                panel_row_index += 1
        source_path.write_text(
            "".join(json.dumps(frame) + "\n" for frame in source_frames)
        )
        source_sha = audit.hashlib.sha256(source_path.read_bytes()).hexdigest()
        summary = {
            "split": "test_hard",
            "family": family,
            "scene_id": f"{family}_synthetic",
            "render_status": "complete",
            "g2_model_outputs_opened": False,
            "source": {
                "frames_jsonl": {
                    "path": str(source_path),
                    "sha256": source_sha,
                }
            },
            "rendered_frames": rendered_frames,
        }
        summary_path = scene_root / "summary.json"
        summary_path.write_text(json.dumps(summary) + "\n")
        summary_key = str(summary_path.relative_to(summary_root))
        expected_hashes[summary_key] = audit.hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()
        expected_splits[summary_key] = "test_hard"

    def file_sha(path):
        return audit.hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(audit, "BINDING_PATH", binding_path)
    monkeypatch.setattr(audit, "AMENDMENT_PATH", fit_amendment_path)
    monkeypatch.setattr(audit, "SUPERSEDED_SCOPE_AMENDMENT_PATH", superseded_path)
    monkeypatch.setattr(audit, "ROLE_NAMESPACE_AMENDMENT_PATH", role_path)
    monkeypatch.setattr(audit, "FIT_PANEL_PATH", fit_panel_path)
    monkeypatch.setattr(audit, "OUTPUT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(audit, "SUMMARY_ROOT", summary_root)
    monkeypatch.setattr(audit, "FRAMES_ROOT", frames_root)
    monkeypatch.setattr(audit, "BINDING_SHA256", file_sha(binding_path))
    monkeypatch.setattr(audit, "AMENDMENT_SHA256", file_sha(fit_amendment_path))
    monkeypatch.setattr(
        audit, "SUPERSEDED_SCOPE_AMENDMENT_SHA256", file_sha(superseded_path)
    )
    monkeypatch.setattr(audit, "ROLE_NAMESPACE_AMENDMENT_SHA256", file_sha(role_path))
    monkeypatch.setattr(audit, "FIT_PANEL_FILE_SHA256", file_sha(fit_panel_path))
    monkeypatch.setattr(audit, "SOURCE_PATHS", ())
    monkeypatch.setattr(audit, "EXPECTED_SUMMARY_COUNT", len(FAMILIES))
    monkeypatch.setattr(audit, "EXPECTED_SUMMARY_SHA256", expected_hashes)
    monkeypatch.setattr(audit, "EXPECTED_LEGACY_SOURCE_SPLIT", expected_splits)
    monkeypatch.setattr(
        audit,
        "EXPECTED_LEGACY_SOURCE_SPLIT_FRAME_COUNTS",
        {"test_hard": 320},
    )
    monkeypatch.setattr(
        audit,
        "_validate_fit_panel",
        lambda _payload, *, panel_file_sha256: records,
    )
    dummy_comparison = ProjectionComparison(
        metrics={}, token_displacements=np.asarray([0.0], dtype=np.float64)
    )
    monkeypatch.setattr(audit, "compare_projection", lambda _camera: dummy_comparison)
    monkeypatch.setattr(
        audit,
        "summarize_frame_comparisons",
        lambda comparisons: {
            "query_count_per_frame": QUERY_COUNT,
            "frame_count": len(comparisons),
        },
    )
    monkeypatch.setattr(
        audit,
        "ordering_decision",
        lambda _comparisons: {
            "material_dynamic_pose_mismatch": False,
            "next_intervention": "explicit_hierarchical_output",
        },
    )

    result = audit.run_authoritative_audit(
        authorization=audit.ROLE_NAMESPACE_AMENDMENT_SHA256
    )

    assert result["record_integrity"]["matched_record_count"] == 320
    assert result["record_integrity"]["family_frame_counts"] == {
        family: 64 for family in FAMILIES
    }
    assert result["role_namespace"]["physical_dataset_role_train_frame_records"] == 320
    assert result["role_namespace"]["physical_dataset_nontrain_frame_records"] == 0
    assert result["role_namespace"]["legacy_source_split_frame_records"] == {
        "test_hard": 320
    }
    assert result["role_namespace"]["legacy_source_split_used_for_inclusion"] is False
    assert all(
        frame["record_key"]["legacy_source_split"] == "test_hard"
        for frame in result["frames"]
    )
    assert audit.OUTPUT_PATH.exists()


def test_metadata_path_guard_rejects_images_labels_and_role_payloads(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for name in ("frame.png", "labels.npz", "sealed_test.json", "model.pt"):
        path = root / name
        path.write_bytes(b"metadata guard test")
        with pytest.raises(PermissionError):
            audit._strict_existing_path(path, root, label="test")


def test_exclusive_atomic_output_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "result.json"
    audit._exclusive_atomic_write_json(output, {"value": 1})
    assert json.loads(output.read_text()) == {"value": 1}

    with pytest.raises(FileExistsError, match="already exists"):
        audit._exclusive_atomic_write_json(output, {"value": 2})
    assert json.loads(output.read_text()) == {"value": 1}


def test_registered_query_count_is_exact() -> None:
    assert QUERY_COUNT == 5 * 64 * 256


def _ordering_inputs(*, rough: float, non_rough: float):
    return {
        family: [
            ProjectionComparison(
                metrics={
                    "token_displacement": {
                        "p50_token": rough
                        if family == "rough_local_dynamics"
                        else non_rough
                    }
                },
                token_displacements=np.asarray(
                    [rough if family == "rough_local_dynamics" else non_rough],
                    dtype=np.float64,
                ),
            )
        ]
        for family in FAMILIES
    }


@pytest.mark.parametrize(
    ("rough", "non_rough", "rough_pass", "contrast_pass", "material"),
    (
        (0.5, 0.25, True, True, True),
        (np.nextafter(0.5, -np.inf), 0.0, False, True, False),
        (0.5, np.nextafter(0.25, np.inf), True, False, False),
    ),
)
def test_ordering_decision_uses_exact_inclusive_boundaries(
    rough, non_rough, rough_pass, contrast_pass, material
) -> None:
    decision = ordering_decision(
        _ordering_inputs(rough=rough, non_rough=non_rough)
    )

    assert decision["rough_threshold_passes"] is rough_pass
    assert decision["contrast_threshold_passes"] is contrast_pass
    assert decision["material_dynamic_pose_mismatch"] is material
    assert decision["next_intervention"] == (
        "fixed_vs_recorded_pose_projective_sampling_ab"
        if material
        else "explicit_hierarchical_output"
    )


def test_fit_panel_extractor_copies_only_fit_metadata(
    tmp_path, monkeypatch
) -> None:
    rows = [
        {"family": family, "dataset_role": "train", "row": index}
        for family in extractor.FAMILIES
        for index in range(32)
    ]
    rows_sha = extractor.canonical_json_sha256(rows)
    fit = {
        "row_count": 160,
        "frame_count": 320,
        "rows_sha256": rows_sha,
        "rows": rows,
    }
    source_core = {
        "panels": {
            "fit": fit,
            "same_scene_holdout": {"forbidden_sentinel": True},
            "cross_scene_holdout": {"forbidden_sentinel": True},
        }
    }
    source_content = extractor.canonical_json_sha256(source_core)
    source = {**source_core, "content_sha256": source_content}
    source_path = tmp_path / "panel.json"
    output_path = tmp_path / "fit-panel.json"
    source_path.write_text(json.dumps(source, sort_keys=True))
    source_file_sha = extractor.hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(extractor, "SOURCE_PATH", source_path)
    monkeypatch.setattr(extractor, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(extractor, "SOURCE_FILE_SHA256", source_file_sha)
    monkeypatch.setattr(extractor, "SOURCE_CONTENT_SHA256", source_content)
    monkeypatch.setattr(extractor, "FIT_ROWS_SHA256", rows_sha)

    result = extractor.extract(authorization=extractor.AMENDMENT_SHA256)

    assert output_path.exists()
    assert result["fit"] == fit
    assert "panels" not in result
    assert "forbidden_sentinel" not in output_path.read_text()
    assert result["access_ledger"]["non_fit_rows_copied"] == 0
    assert result["content_sha256"] == extractor.canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )


def test_fit_panel_extractor_rejects_before_input_access(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(extractor, "SOURCE_PATH", tmp_path / "must-not-open.json")
    monkeypatch.setattr(extractor, "OUTPUT_PATH", tmp_path / "output.json")

    with pytest.raises(PermissionError, match="lacks the frozen amendment"):
        extractor.extract(authorization="0" * 64)
