from __future__ import annotations

import hashlib
import json

import pytest

from lewm.datasets.go2_source_camera_contract import (
    SOURCE_CAMERA_CONTRACT_SCHEMA,
    SourceCameraContractError,
    load_source_camera_contract,
    source_camera_contract_record,
    validate_source_camera_contract_for_index,
)


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_contract(path, source_index_path) -> dict:
    core = {
        "schema": SOURCE_CAMERA_CONTRACT_SCHEMA,
        "source_index": {
            "path": str(source_index_path),
            "sha256": hashlib.sha256(source_index_path.read_bytes()).hexdigest(),
        },
        "scene_count": 1,
        "source_records": [{"scene_id_sha256": "a" * 64}],
        "actual_source_projection": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 78.323,
        },
        "platform_projection_after_rectification": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.8370386364,
            "center_crop_fraction_xy": [1.0, 0.75],
        },
        "g2_images_opened": False,
    }
    payload = {**core, "content_sha256": _canonical_sha256(core)}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def test_source_camera_contract_is_content_and_index_bound(tmp_path) -> None:
    source_index = tmp_path / "sources.jsonl"
    source_index.write_text("{}\n")
    contract_path = tmp_path / "camera.json"
    payload = _write_contract(contract_path, source_index)

    contract = load_source_camera_contract(contract_path)
    validate_source_camera_contract_for_index(
        contract,
        source_index_path=source_index,
    )
    record = source_camera_contract_record(contract_path)
    assert record["content_sha256"] == payload["content_sha256"]

    source_index.write_text('{"changed":true}\n')
    with pytest.raises(SourceCameraContractError, match="another index"):
        validate_source_camera_contract_for_index(
            contract,
            source_index_path=source_index,
        )


def test_source_camera_contract_rejects_tampering_and_g2_image_contact(
    tmp_path,
) -> None:
    source_index = tmp_path / "sources.jsonl"
    source_index.write_text("{}\n")
    contract_path = tmp_path / "camera.json"
    payload = _write_contract(contract_path, source_index)
    payload["scene_count"] = 2
    contract_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(SourceCameraContractError, match="content hash mismatch"):
        load_source_camera_contract(contract_path)

    payload = _write_contract(contract_path, source_index)
    core = dict(payload)
    core.pop("content_sha256")
    core["g2_images_opened"] = True
    payload = {**core, "content_sha256": _canonical_sha256(core)}
    contract_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(SourceCameraContractError, match="must not open G2 images"):
        load_source_camera_contract(contract_path)
