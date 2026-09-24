from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType, SimpleNamespace

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v9 as predecessor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v10 as auditor
from lewm_worlds import manifest as manifest_semantics
from scripts import audit_go2_n32_camera_frustum_observability as source_v4


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_"
    "manifest_representation_successor_amendment_2026-07-14.md"
)
SOURCE = ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v10.py"
CLI = ROOT / "scripts/audit_go2_shared_jepa_v5_raw_supervision_v10.py"
TEST = ROOT / "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v10.py"
HANDOFF = ROOT / (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v10_"
    "author_handoff_2026-07-14.md"
)

FROZEN = {
    AMENDMENT: "02100ee004a572209866a3eb9356441600944b2da2d9b1010282ab992ad02a81",
    SOURCE: "3c87dc7878f2e0ae9c54e9b05f1183339b9839568832c0e2fcb6ce75dda984d9",
    CLI: "695653257d4aeccef162f3e8f30fd0eba88a090f29cb811481898b4680fe3866",
    TEST: "af084a3d097ae66db14f68db7c700843f1bc4007515eb07e444a5018036f177d",
    HANDOFF: "9635d4fa891e9734a6245b8cde3d6eaf8934bb8b9b7c90db9365aa6207e2c959",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _raw_manifest() -> dict[str, object]:
    box = {
        "object_id": "wall-0",
        "kind": "wall",
        "center_xyz_m": [1.0, 0.0, 0.5],
        "size_xyz_m": [0.2, 2.0, 1.0],
        "yaw_rad": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "material_id": "wall",
    }
    return {
        "scene_id": "independent_scene",
        "family": "open_obstacle_field",
        "difficulty_tier": "development",
        "topology_seed": 1,
        "visual_seed": 2,
        "physics_seed": 3,
        "world_bounds_xy_m": [[-2.0, -2.0], [2.0, 2.0]],
        "spawn": {
            "xyz_m": [0.0, 0.0, 0.4],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "graph_nodes": [],
        "graph_edges": [],
        "walls": [box],
        "obstacles": [],
        "landmarks": [],
        "camera_constraints": {
            "min_wall_thickness_m": 0.08,
            "near_m": 0.05,
            "far_m": 20.0,
            "min_camera_clearance_m": 0.1,
        },
        "visual_randomization": None,
        "physics_randomization": None,
        "camera_extrinsic_jitter": None,
    }


def _authorization_payload() -> dict[str, object]:
    source_map = []
    for role, path in auditor.SOURCE_ROLE_PATHS:
        digest = auditor.FROZEN_AUTHORITY_ROLE_SHA256.get(
            role, hashlib.sha256(role.encode("ascii")).hexdigest()
        )
        source_map.append({"role": role, "path": path, "sha256": digest})
    source_by_role = {str(row["role"]): row for row in source_map}
    review = source_by_role["auditor_review"]
    candidate = [
        source_by_role[role] for role in auditor.AUDITOR_CANDIDATE_ROLES
    ]
    core: dict[str, object] = {
        "schema": auditor.AUTHORIZATION_SCHEMA,
        "exact_audit_v10_authorized": True,
        **{field: False for field in auditor.AUTHORIZATION_FALSE_FIELDS},
        "input_dataset_path": auditor.AUTHORIZED_DATASET_PATH,
        "success_report_path": auditor.AUTHORIZED_SUCCESS_REPORT_PATH,
        "failure_report_path": auditor.AUTHORIZED_FAILURE_REPORT_PATH,
        "v9_build_authorization_file_sha256": (
            auditor.FROZEN_AUTHORITY_ROLE_SHA256["v9_build_authorization"]
        ),
        "v9_build_authorization_content_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_CONTENT_SHA256
        ),
        "v9_build_authorization_source_map_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_SOURCE_MAP_SHA256
        ),
        "v9_dataset_manifest_file_sha256": (
            auditor.FROZEN_AUTHORITY_ROLE_SHA256["v9_dataset_manifest"]
        ),
        "v9_dataset_manifest_content_sha256": (
            auditor.FROZEN_V9_DATASET_MANIFEST_CONTENT_SHA256
        ),
        "v9_audit_failure_file_sha256": (
            auditor.FROZEN_AUTHORITY_ROLE_SHA256["v9_terminal_failure"]
        ),
        "v9_audit_failure_content_sha256": (
            auditor.FROZEN_V9_AUDIT_FAILURE_CONTENT_SHA256
        ),
        "auditor_review": {
            "schema": auditor.REVIEW_BINDING_SCHEMA,
            "review_schema": auditor.AUDITOR_REVIEW_SCHEMA,
            "verdict": "PASS",
            "reviewer": "/root/raw_v10_independent_review",
            "implementation_author": auditor.V10_IMPLEMENTATION_AUTHOR,
            "path": review["path"],
            "file_sha256": review["sha256"],
            "content_sha256": "b" * 64,
            "candidate": candidate,
        },
        "source_map": source_map,
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def _transaction_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = (tmp_path / "repository").absolute()
    publication_parent = root / "generated"
    dataset = publication_parent / "dataset"
    dataset.mkdir(parents=True)
    data_payload = b"bound-v10-dataset-payload\n"
    _write(dataset / "data.bin", data_payload)
    manifest_payload = b'{"synthetic":true}\n'
    manifest_sha256 = _write(dataset / "manifest.json", manifest_payload)
    v9_failure = publication_parent / "dataset.audit_v9.failed.json"
    v9_failure_payload = b'{"status":"terminal_failed_no_dataset_authority"}\n'
    v9_failure_sha256 = _write(v9_failure, v9_failure_payload)
    authorization_path = root / "docs" / "authorization.json"
    authorization_sha256 = _write(authorization_path, b'{"synthetic":true}\n')

    plan_manifest = root / "metadata" / "manifest.json"
    plan_rows = root / "metadata" / "rows.jsonl"
    source_index = root / "metadata" / "source.json"
    plan_manifest_sha256 = _write(plan_manifest, b"manifest\n")
    plan_rows_sha256 = _write(plan_rows, b"rows\n")
    source_index_sha256 = _write(source_index, b"source\n")

    monkeypatch.setattr(auditor, "ROOT", root)
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(
        auditor, "CANONICAL_AUDIT_REPORT", publication_parent / "dataset.audit_v10.json"
    )
    monkeypatch.setattr(
        auditor,
        "CANONICAL_AUDIT_FAILURE",
        publication_parent / "dataset.audit_v10.failed.json",
    )
    monkeypatch.setattr(auditor, "AUDIT_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "FROZEN_V9_PREDECESSOR_SHA256", {})
    monkeypatch.setattr(auditor, "REVIEWED_V4_SOURCE_SHA256", {})
    monkeypatch.setattr(
        auditor, "DATASET_MANIFEST_RELATIVE_PATH", str(plan_manifest.relative_to(root))
    )
    monkeypatch.setattr(
        auditor, "DATASET_ROWS_RELATIVE_PATH", str(plan_rows.relative_to(root))
    )
    monkeypatch.setattr(
        auditor, "SOURCE_INDEX_RELATIVE_PATH", str(source_index.relative_to(root))
    )
    monkeypatch.setattr(auditor, "DATASET_MANIFEST_FILE_SHA256", plan_manifest_sha256)
    monkeypatch.setattr(auditor, "DATASET_ROWS_FILE_SHA256", plan_rows_sha256)
    monkeypatch.setattr(auditor, "SOURCE_INDEX_FILE_SHA256", source_index_sha256)

    sources = (
        auditor.SourceBindingV10(
            "v9_dataset_manifest",
            str((dataset / "manifest.json").relative_to(root)),
            manifest_sha256,
        ),
        auditor.SourceBindingV10(
            "v9_terminal_failure",
            str(v9_failure.relative_to(root)),
            v9_failure_sha256,
        ),
    )
    authorization = auditor.AcceptedAuthorizationV10(
        authorization_file_sha256="1" * 64,
        authorization_content_sha256="2" * 64,
        source_map_sha256="3" * 64,
        execution_authorization_file_sha256=authorization_sha256,
        execution_authorization_content_sha256="4" * 64,
        execution_source_map_sha256="5" * 64,
        sources=sources,
    )
    context = auditor._AuditPublicationContextV10(
        authorization=authorization,
        manifest={
            "files": [
                {
                    "path": "data.bin",
                    "byte_count": len(data_payload),
                    "file_sha256": hashlib.sha256(data_payload).hexdigest(),
                }
            ]
        },
        manifest_file_sha256=manifest_sha256,
        hashed_sources=(),
        parent_contracts=(),
    )
    result_core = {"schema": "synthetic_audit_v10", "verdict": "PASS"}
    result = {
        **result_core,
        "content_sha256": auditor.canonical_json_sha256(result_core),
    }
    retained = auditor._open_retained_directory_chain(publication_parent)
    name, descriptor, fingerprint, digest = auditor._stage_owned_audit_candidate(
        retained, result
    )
    transaction = auditor._ClosedAuditPublicationTransaction(
        context=context,
        retained=retained,
        candidate_name=name,
        candidate_descriptor=descriptor,
        candidate_fingerprint=fingerprint,
        candidate_sha256=digest,
    )
    return {
        "result": result,
        "retained": retained,
        "candidate_name": name,
        "descriptor": descriptor,
        "transaction": transaction,
        "v9_failure": v9_failure,
        "v9_failure_payload": v9_failure_payload,
        "manifest": dataset / "manifest.json",
        "manifest_payload": manifest_payload,
    }


def _close_fixture(value: dict[str, object], *, cleanup: bool = True) -> None:
    transaction = value["transaction"]
    retained = value["retained"]
    descriptor = value["descriptor"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    assert isinstance(descriptor, int)
    transaction.close()
    if cleanup:
        auditor._cleanup_owned_audit_candidate(
            retained,
            candidate_name=str(value["candidate_name"]),
            candidate_descriptor=descriptor,
            renamed=transaction.renamed,
        )
    os.close(descriptor)
    retained.close()


def test_independent_candidate_hashes_are_frozen() -> None:
    assert {path: _sha256(path) for path in FROZEN} == FROZEN


def test_independent_v9_failure_and_v10_raw_identity_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    assert isinstance(parsed.walls, tuple)
    with pytest.raises(ValueError, match="wall boxes are not a list"):
        source_v4._validate_raw_scene_object_records(parsed.to_dict())

    original = source_v4._validate_raw_scene_object_records
    observed: list[object] = []

    def spy(value: object) -> None:
        observed.append(value)
        original(value)  # type: ignore[arg-type]

    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_install_reviewed_source_semantics", lambda _digest: None
    )
    monkeypatch.setattr(source_v4, "_validate_raw_scene_object_records", spy)
    monkeypatch.setattr(source_v4, "_rendered_boxes", lambda _manifest: ())
    monkeypatch.setattr(
        source_v4,
        "labels_v3",
        SimpleNamespace(_render_object_records=lambda _manifest: []),
    )
    horizontal = 78.323
    vertical = math.degrees(
        2.0
        * math.atan(math.tan(math.radians(horizontal) * 0.5) * (168.0 / 224.0))
    )
    empty_hash = source_v4.canonical_json_sha256([])
    result = auditor._validate_sample_render_contract(
        {
            "frames_jsonl": "/tmp/v10-independent-frames.jsonl",
            "camera": {
                "native_resolution": [224, 168],
                "training_resolution": [112, 84],
                "fov_axis": "horizontal",
                "fov_deg": horizontal,
                "near_m": 0.05,
                "far_m": 20.0,
                "encoding": "rgb8",
                "mount_body": {},
            },
        },
        {
            "resolution_wh": [224, 168],
            "camera_projection": {
                "model": "pinhole",
                "renderer_fov_axis": "vertical",
                "horizontal_fov_deg": horizontal,
                "vertical_fov_deg": vertical,
                "near_m": 0.05,
                "far_m": 20.0,
                "runtime_rectification_required": False,
            },
            "object_parity": {
                "schema": "lewm_render_object_parity_v1",
                "rendered_groups": ["wall", "obstacle", "landmark", "distractor"],
                "rendered_object_count": 0,
                "rendered_object_ids": [],
                "rendered_object_ids_sha256": empty_hash,
                "rendered_object_records_sha256": empty_hash,
                "collision_distractors_rendered": True,
                "full_box_roll_pitch_yaw_rendered": True,
            },
        },
        raw,
        parsed,
        {"frames": {"path": "/tmp/v10-independent-frames.jsonl"}},
        authorization_sha256="a" * 64,
    )
    assert result == ()
    assert observed == [raw] and observed[0] is raw


@pytest.mark.parametrize(
    "value_factory",
    [
        lambda raw, _parsed: tuple(raw.items()),
        lambda raw, _parsed: MappingProxyType(raw),
        lambda raw, _parsed: (item for item in raw.items()),
        lambda _raw, parsed: parsed.to_dict(),
    ],
)
def test_independent_nonraw_representations_are_rejected(
    value_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    value = value_factory(raw, parsed)  # type: ignore[operator]
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_install_reviewed_source_semantics", lambda _digest: None
    )
    with pytest.raises((auditor.RawSupervisionAuditError, ValueError)):
        auditor._validate_sample_render_contract(
            {}, {}, value, parsed, {}, authorization_sha256="a" * 64
        )


def test_independent_ast_delta_is_closed_and_science_is_v9_identical() -> None:
    def definitions(module: object) -> dict[str, str]:
        tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    v9 = definitions(predecessor)
    v10 = definitions(auditor)
    normalize = lambda value: value.replace("V10", "V9").replace("v10", "v9")
    changed = {
        name for name in set(v9) & set(v10) if normalize(v10[name]) != v9[name]
    }
    assert changed == {
        "_audit_transaction_bindings",
        "_publish_terminal_audit_failure",
        "_recompute_one_exact_sample",
        "_require_exact_authority",
        "_review_binding",
        "_validate_authorization_phase_one",
        "_validate_authorization_phase_two",
        "_validate_sample_render_contract",
    }
    transaction = "_ClosedAuditPublicationTransaction"
    assert normalize(v10[transaction]) == v9[transaction]
    for name in (
        "_dataset_transaction_inventory",
        "_audit_fixed_dataset",
        "_exact_sample_recomputer",
        "_recompute_exact_sample_task",
        "_initialize_exact_worker",
    ):
        assert normalize(v10[name]) == v9[name]

    for old, new in (
        ("execute_exact_audit_v9", "execute_exact_audit_v10"),
        ("_final_revalidate_authorized_audit_v9", "_final_revalidate_authorized_audit_v10"),
    ):
        assert normalize(v10[new]) == v9[old]


def test_independent_authority_phase_one_is_exact_and_closed() -> None:
    payload = _authorization_payload()
    phase_one = auditor._validate_authorization_phase_one(
        payload, authorization_file_sha256="c" * 64
    )
    assert tuple(item.role for item in phase_one.sources) == auditor.SOURCE_ROLES
    assert len(phase_one.sources) == 11
    assert phase_one.auditor_review.reviewer == "/root/raw_v10_independent_review"
    for identity in (
        "/root",
        auditor.V10_IMPLEMENTATION_AUTHOR,
        auditor.V9_BUILDER_IMPLEMENTATION_AUTHOR,
        auditor.V9_AUDITOR_IMPLEMENTATION_AUTHOR,
        auditor.V9_BUILDER_REVIEWER,
        auditor.V9_AUDITOR_REVIEWER,
    ):
        mutated = deepcopy(payload)
        mutated["auditor_review"]["reviewer"] = identity  # type: ignore[index]
        core = dict(mutated)
        core.pop("content_sha256")
        mutated["content_sha256"] = auditor.canonical_json_sha256(core)
        with pytest.raises(PermissionError):
            auditor._validate_authorization_phase_one(
                mutated, authorization_file_sha256="c" * 64
            )


def test_independent_clean_v10_transaction_preserves_v9_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    retained = value["retained"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.validate(allow_final_metadata_change=True)
        transaction.validate_after_rename()
        os.fsync(retained.directory_fd)
        transaction.require_final_quiet()
        assert value["v9_failure"].read_bytes() == value["v9_failure_payload"]  # type: ignore[union-attr]
        assert value["manifest"].read_bytes() == value["manifest_payload"]  # type: ignore[union-attr]
        assert auditor.CANONICAL_AUDIT_REPORT.read_bytes() == (
            auditor.canonical_json_bytes(value["result"]) + b"\n"
        )
    finally:
        _close_fixture(value, cleanup=False)


def test_independent_unrelated_generated_sibling_file_churn_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    transaction = value["transaction"]
    retained = value["retained"]
    assert isinstance(transaction, auditor._ClosedAuditPublicationTransaction)
    assert isinstance(retained, auditor._RetainedDirectoryChain)
    transaction.validate_before_rename()
    transaction.rename_owned()
    retained.validate(allow_final_metadata_change=True)
    transaction.validate_after_rename()
    os.fsync(retained.directory_fd)
    unrelated = retained.absolute_path.parent / "unrelated-generated-sibling.tmp"
    unrelated.write_bytes(b"unrelated\n")
    unrelated.unlink()
    try:
        transaction.require_final_quiet()
    finally:
        _close_fixture(value, cleanup=False)


@pytest.mark.parametrize("target_key", ["v9_failure", "manifest"])
def test_independent_v10_transaction_rejects_bound_v9_mutation(
    target_key: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    value = _transaction_fixture(monkeypatch, tmp_path)
    target = value[target_key]
    assert isinstance(target, Path)
    target.write_bytes(b"mutated then restored\n")
    target.write_bytes(value[f"{target_key}_payload"])  # type: ignore[arg-type]
    try:
        with pytest.raises(auditor.RawSupervisionAuditError, match="poisoned"):
            value["transaction"].validate_before_rename()  # type: ignore[union-attr]
    finally:
        _close_fixture(value)


def test_independent_failure_namespace_is_terminal_and_no_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    report = tmp_path / "dataset.audit_v10.json"
    failure = tmp_path / "dataset.audit_v10.failed.json"
    v9_failure = tmp_path / "dataset.audit_v9.failed.json"
    v9_payload = b'{"status":"terminal_failed_no_dataset_authority"}\n'
    v9_failure.write_bytes(v9_payload)
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", failure)
    auditor._publish_terminal_audit_failure(
        authorization_sha256="a" * 64,
        error=RuntimeError("independent synthetic failure"),
    )
    receipt = json.loads(failure.read_bytes())
    assert receipt["status"] == "terminal_failed_no_dataset_authority"
    assert receipt["retry_authorized"] is False
    assert receipt["dataset_use_authorized"] is False
    assert receipt["training_authorized"] is False
    assert v9_failure.read_bytes() == v9_payload
    with pytest.raises(FileExistsError):
        auditor._publish_terminal_audit_failure(
            authorization_sha256="a" * 64,
            error=RuntimeError("forbidden retry"),
        )


def test_independent_production_surface_is_audit_only() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    signature = inspect.signature(auditor.execute_exact_audit_v10)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for value in signature.parameters.values()
    )
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("auditor_v9" in name or "supervision_builder" in name for name in imported)
    assert "test_hook" not in source and "importlib" not in source
    assert not any(
        token in SOURCE.name
        for token in ("trainer", "g2", "heldout", "runtime", "production")
    )
    assert auditor.ACCELERATOR_ENVIRONMENT == predecessor.ACCELERATOR_ENVIRONMENT
    assert {
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    } == set(auditor.ACCELERATOR_ENVIRONMENT)
    assert re.search(r"multiprocessing\.get_context\(\"spawn\"\)", source)
    assert "max_workers=worker_count" in source
    for value in (True, False, 0, 7, 1.0, "6", None):
        with pytest.raises(ValueError):
            auditor._strict_workers(value)
    assert tuple(auditor._strict_workers(value) for value in range(1, 7)) == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
