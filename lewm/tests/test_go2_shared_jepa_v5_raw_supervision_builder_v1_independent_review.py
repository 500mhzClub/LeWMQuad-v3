"""Different-agent conformance review for raw-supervision builder V1.

The failing authorization-order check is the frozen BLOCK reproducer.  Tests
use synthetic prepared frames and temporary paths only; exact source payloads
and the canonical output remain unopened.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as builder


ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v1.py": (
        "df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v1_"
    "author_handoff_2026-07-13.md": (
        "9d9aee5f636069d8beef2362bcc43b9be0063207d9ffe17d9045f99e3c30d28c"
    ),
}
REQUIRED_AUTHORIZATION_ROLES = (
    "builder_source",
    "builder_cli",
    "builder_test",
    "builder_handoff",
    "builder_review",
    "auditor_source",
    "auditor_cli",
    "auditor_test",
    "auditor_review",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization(source_map: list[dict[str, str]]) -> dict[str, Any]:
    core = {
        "schema": builder.AUTHORIZATION_SCHEMA,
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": {"verdict": "PASS"},
        "auditor_review": {"verdict": "PASS"},
        "source_map": source_map,
    }
    return {**core, "content_sha256": builder.canonical_json_sha256(core)}


def test_builder_v1_independent_frozen_candidate_and_parents_rehash() -> None:
    assert {relative: _sha(ROOT / relative) for relative in FROZEN} == FROZEN
    for relative, digest in {
        **builder.FROZEN_PARENT_HASHES,
        **builder.REVIEWED_V4_SOURCES,
    }.items():
        assert _sha(ROOT / relative) == digest


def test_builder_v1_absent_authority_reaches_no_metadata_or_source_opener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", tmp_path / "output")
    monkeypatch.setattr(builder, "FAILURE_RECEIPT", tmp_path / "failed.json")
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("opened")
        raise AssertionError("metadata or source opener reached")

    monkeypatch.setattr(builder.plan_v5, "load_frozen_development_metadata", forbidden)
    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    with pytest.raises(PermissionError, match="authorization is absent"):
        builder.execute_exact_build_v1(authorization_sha256="0" * 64, workers=1)
    assert calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "failed.json").exists()


def test_builder_v1_invalid_authority_must_be_fully_validated_before_any_source_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCK: the candidate opens row one before discovering missing roles."""

    payload = _authorization(
        [
            {
                "role": "builder_source",
                "path": "arbitrary/referenced_frames.jsonl",
                "sha256": "1" * 64,
            }
        ]
    )
    opened: list[Path] = []

    def record_open(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        assert repository_root == builder.ROOT
        assert expected_sha256 == "1" * 64
        opened.append(path)
        return b"arbitrary referenced payload\n"

    monkeypatch.setattr(builder, "_read_bound_regular_file", record_open)
    with pytest.raises(builder.RawSupervisionBuildError, match="source roles changed"):
        builder._validate_authorization_payload(
            payload,
            authorization_file_sha256="2" * 64,
        )
    assert opened == [], (
        "a structurally invalid authorization caused a caller-selected repository "
        "file to open before the complete role map was accepted"
    )


def test_builder_v1_complete_authorization_source_map_is_exactly_nine_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_map = [
        {"role": role, "path": f"review/{role}.txt", "sha256": f"{index + 1:x}" * 64}
        for index, role in enumerate(REQUIRED_AUTHORIZATION_ROLES)
    ]
    opened: list[Path] = []

    def record_open(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        assert repository_root == builder.ROOT
        assert len(expected_sha256) == 64
        opened.append(path)
        return b"reviewed source\n"

    monkeypatch.setattr(builder, "_read_bound_regular_file", record_open)
    result = builder._validate_authorization_payload(
        _authorization(source_map),
        authorization_file_sha256="f" * 64,
    )
    assert result["source_map"] == source_map
    assert opened == [builder.ROOT / item["path"] for item in source_map]


def test_builder_v1_exact_population_and_metadata_bindings_are_frozen() -> None:
    source = (ROOT / FROZEN.keys().__iter__().__next__()).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    validator = functions["_validate_exact_plan_result"]
    execute = functions["execute_exact_build_v1"]
    assert "len(plan.pairs) != 5172" in validator
    assert "len(plan.endpoints) != 9460" in validator
    assert "len(inventory.records) != 88" in validator
    assert "plan_v5.SOURCE_INVENTORY_SHA256" in validator
    assert "sum((len(job.endpoints) for job in jobs)) != 9460" in execute
    assert "len(source_receipts) != 354" in execute
    assert "sum(map(len, revalidated)) != 352" in execute
    ledger = builder._exact_access_ledger(
        plan=type("Plan", (), {"value": {"access_ledger": {}}})(),
        inventory=type("Inventory", (), {"access_ledger": {}})(),
        frames_scanned=123,
    )
    assert ledger["development_scene_workers"] == 88
    assert ledger["unique_endpoint_raycasts"] == 9460
    assert ledger["pair_endpoint_references"] == 10344
    assert ledger["source_frames_selected_records"] == 9460
    assert all(
        ledger[name] == 0
        for name in (
            "g2_sidecar_byte_opens",
            "g2_source_payload_opens",
            "g2_label_payload_opens",
            "g2_rgb_byte_opens",
            "rgb_byte_opens",
            "rgb_decodes",
            "parent_label_shard_payload_opens",
            "heldout_or_sealed_opens",
        )
    )


def test_builder_v1_uses_reviewed_v4_once_per_unique_endpoint() -> None:
    source = (ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py").read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    worker = ast.unparse(functions["_write_prepared_scene_job"])
    exact_loader = ast.unparse(functions["_load_exact_scene_job"])
    validation = ast.unparse(functions["_validate_jobs_and_pairs"])
    assert worker.count("v4_builder.build_frame_evidence_v4(item.frame)") == 1
    assert worker.count(
        "v4_builder.rasterize_observable_camera_ray_evidence_v4(evidence)"
    ) == 1
    assert "one endpoint was scheduled more than once" in validation
    assert "source frames did not match every endpoint once" in exact_loader
    assert "compose_yaw_aligned_camera" in exact_loader
    assert "_box_in_yaw_body" in exact_loader
    assert "_validated_sidecar_source_attitude" in exact_loader


def test_builder_v1_array_layout_has_scalar_ground_plane_and_reviewed_raster() -> None:
    assert builder.ARRAY_LAYOUT == (
        ("camera_origin_body_m.f4", "<f4", (3,)),
        ("camera_basis_body_fru.f4", "<f4", (3, 3)),
        ("ground_plane_z_body_m.f4", "<f4", ()),
        ("ground_support_in_frustum.u1", "u1", (128, 128, 5)),
        ("ground_support_clear_to_target.u1", "u1", (128, 128, 5)),
        ("pixel_hit_mask.u1", "u1", (84, 112)),
        ("pixel_first_hit_distance_m.f4", "<f4", (84, 112)),
        ("raster_labels.u1", "u1", (64, 64)),
    )
    evidence = SimpleEvidence()
    raster = type("Raster", (), {"output_labels": np.zeros((64, 64), dtype=np.uint8)})()
    arrays = builder._endpoint_arrays(evidence, raster)
    assert arrays[2].shape == ()
    assert arrays[2].dtype == np.dtype("<f4")
    assert arrays[7].shape == (64, 64)
    assert arrays[7].dtype == np.uint8


class SimpleEvidence:
    camera_origin_body_m = (0.0, 0.0, 0.0)
    camera_basis_body_fru = np.eye(3)
    ground_plane_z_body_m = -0.35
    ground_support_in_frustum = np.zeros((128, 128, 5), dtype=bool)
    ground_support_clear_to_target = np.zeros((128, 128, 5), dtype=bool)
    pixel_hit_mask = np.zeros((84, 112), dtype=bool)
    pixel_first_hit_distance_m = np.zeros((84, 112), dtype=np.float32)


def test_builder_v1_publication_is_retained_noreplace_and_owned_cleanup_only(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    retained = builder._open_publication_parent(parent)
    try:
        os.mkdir("staging", 0o700, dir_fd=retained.parent_fd)
        retained.refresh_after_owned_mutation()
        identity = builder._named_directory_identity(retained.parent_fd, "staging")
        os.rename("staging", "moved", src_dir_fd=retained.parent_fd, dst_dir_fd=retained.parent_fd)
        os.mkdir("staging", 0o700, dir_fd=retained.parent_fd)
        (parent / "staging/foreign").write_bytes(b"keep")
        assert builder._cleanup_owned_directory(retained, "staging", identity) is False
        assert (parent / "staging/foreign").read_bytes() == b"keep"
        assert (parent / "moved").is_dir()
    finally:
        retained.close()

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "foreign").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        builder._rename_noreplace(source, destination)
    assert source.is_dir()
    assert (destination / "foreign").read_bytes() == b"keep"


def test_builder_v1_exact_loader_never_dereferences_rgb_or_parent_labels() -> None:
    source = (ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_exact_scene_job"
    )
    body = ast.unparse(function)
    assert "('frames', 'source_frames_jsonl', 'sha256')" in body
    assert "('scene_manifest', 'source_scene_manifest', 'file_sha256')" in body
    assert "('render_plan', 'render_plan', 'sha256')" in body
    assert "('render_summary', 'render_summary', 'sha256')" in body
    assert "image_path_metadata_only=str(endpoint['image_path_metadata_only'])" in body
    assert "rgb_path" not in body
    assert "label_shard" not in body
