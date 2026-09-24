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

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v9 as predecessor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v10 as auditor
from lewm_worlds import manifest as manifest_semantics
from scripts import audit_go2_n32_camera_frustum_observability as source_v4


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
        "scene_id": "synthetic_scene",
        "family": "open_obstacle_field",
        "difficulty_tier": "development",
        "topology_seed": 1,
        "visual_seed": 2,
        "physics_seed": 3,
        "world_bounds_xy_m": [[-2.0, -2.0], [2.0, 2.0]],
        "spawn": {"xyz_m": [0.0, 0.0, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
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


def _render_contract_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    horizontal = 78.323
    vertical = math.degrees(
        2.0 * math.atan(math.tan(math.radians(horizontal) * 0.5) * (168.0 / 224.0))
    )
    frames = "/tmp/v10-author-test-frames.jsonl"
    render_plan = {
        "frames_jsonl": frames,
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
    }
    empty_hash = source_v4.canonical_json_sha256([])
    summary = {
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
    }
    source_record = {"frames": {"path": frames}}
    return render_plan, summary, source_record


def _patch_render_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_install_reviewed_source_semantics", lambda _digest: None
    )
    monkeypatch.setattr(source_v4, "_rendered_boxes", lambda _manifest: ())
    monkeypatch.setattr(
        source_v4,
        "labels_v3",
        SimpleNamespace(_render_object_records=lambda _manifest: []),
    )


def _call_render_contract(
    raw: object,
    parsed: object,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, ...]:
    _patch_render_dependencies(monkeypatch)
    render_plan, summary, source_record = _render_contract_inputs()
    return auditor._validate_sample_render_contract(
        render_plan,
        summary,
        raw,
        parsed,
        source_record,
        authorization_sha256="a" * 64,
    )


def _authorization_payload() -> dict[str, object]:
    source_map = []
    for role, path in auditor.SOURCE_ROLE_PATHS:
        digest = auditor.FROZEN_AUTHORITY_ROLE_SHA256.get(
            role, hashlib.sha256(role.encode("ascii")).hexdigest()
        )
        source_map.append({"role": role, "path": path, "sha256": digest})
    source_by_role = {str(row["role"]): row for row in source_map}
    candidate = [source_by_role[role] for role in auditor.AUDITOR_CANDIDATE_ROLES]
    review = source_by_role["auditor_review"]
    core: dict[str, object] = {
        "schema": auditor.AUTHORIZATION_SCHEMA,
        "exact_audit_v10_authorized": True,
        **{field: False for field in auditor.AUTHORIZATION_FALSE_FIELDS},
        "input_dataset_path": auditor.AUTHORIZED_DATASET_PATH,
        "success_report_path": auditor.AUTHORIZED_SUCCESS_REPORT_PATH,
        "failure_report_path": auditor.AUTHORIZED_FAILURE_REPORT_PATH,
        "v9_build_authorization_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_build_authorization"
        ],
        "v9_build_authorization_content_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_CONTENT_SHA256
        ),
        "v9_build_authorization_source_map_sha256": (
            auditor.FROZEN_V9_BUILD_AUTHORIZATION_SOURCE_MAP_SHA256
        ),
        "v9_dataset_manifest_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_dataset_manifest"
        ],
        "v9_dataset_manifest_content_sha256": (
            auditor.FROZEN_V9_DATASET_MANIFEST_CONTENT_SHA256
        ),
        "v9_audit_failure_file_sha256": auditor.FROZEN_AUTHORITY_ROLE_SHA256[
            "v9_terminal_failure"
        ],
        "v9_audit_failure_content_sha256": (
            auditor.FROZEN_V9_AUDIT_FAILURE_CONTENT_SHA256
        ),
        "auditor_review": {
            "schema": auditor.REVIEW_BINDING_SCHEMA,
            "review_schema": auditor.AUDITOR_REVIEW_SCHEMA,
            "verdict": "PASS",
            "reviewer": "/root/raw_v10_independent_reviewer",
            "implementation_author": auditor.V10_IMPLEMENTATION_AUTHOR,
            "path": review["path"],
            "file_sha256": review["sha256"],
            "content_sha256": "b" * 64,
            "candidate": candidate,
        },
        "source_map": source_map,
    }
    return {**core, "content_sha256": auditor.canonical_json_sha256(core)}


def test_v9_failure_reproduces_with_an_actual_parsed_manifest() -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    assert isinstance(parsed.walls, tuple)
    source_v4._validate_raw_scene_object_records(raw)
    with pytest.raises(ValueError, match="wall boxes are not a list"):
        source_v4._validate_raw_scene_object_records(parsed.to_dict())


def test_v10_passes_the_original_decoded_mapping_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    before = auditor.canonical_json_sha256(raw)
    original = source_v4._validate_raw_scene_object_records
    seen: list[object] = []

    def spy(value: object) -> None:
        seen.append(value)
        original(value)  # type: ignore[arg-type]

    monkeypatch.setattr(source_v4, "_validate_raw_scene_object_records", spy)
    assert _call_render_contract(raw, parsed, monkeypatch=monkeypatch) == ()
    assert seen == [raw]
    assert seen[0] is raw
    assert auditor.canonical_json_sha256(raw) == before


@pytest.mark.parametrize(
    "raw_factory",
    [
        lambda raw, _parsed: tuple(raw.items()),
        lambda raw, _parsed: MappingProxyType(raw),
        lambda raw, _parsed: (item for item in raw.items()),
    ],
)
def test_v10_rejects_nondecoded_top_level_representations(
    raw_factory: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    value = raw_factory(raw, parsed)  # type: ignore[operator]
    with pytest.raises(auditor.RawSupervisionAuditError, match="decoded JSON object"):
        _call_render_contract(value, parsed, monkeypatch=monkeypatch)


@pytest.mark.parametrize("kind", ["tuple", "generator", "reconstructed"])
def test_v10_rejects_normalized_or_reconstructed_sequences(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    if kind == "reconstructed":
        invalid = parsed.to_dict()
    else:
        invalid = deepcopy(raw)
        walls = invalid["walls"]
        assert isinstance(walls, list)
        invalid["walls"] = tuple(walls) if kind == "tuple" else (item for item in walls)
    with pytest.raises((ValueError, auditor.RawSupervisionAuditError)):
        _call_render_contract(invalid, parsed, monkeypatch=monkeypatch)


def test_v10_rejects_mutation_during_raw_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_manifest()
    parsed = manifest_semantics.parse_scene_manifest_dict(raw)
    original = source_v4._validate_raw_scene_object_records

    def mutate(value: dict[str, object]) -> None:
        original(value)
        value["unexpected_mutation"] = True

    monkeypatch.setattr(source_v4, "_validate_raw_scene_object_records", mutate)
    with pytest.raises(auditor.RawSupervisionAuditError, match="changed during raw"):
        _call_render_contract(raw, parsed, monkeypatch=monkeypatch)


def test_parsed_semantic_hash_and_rendered_geometry_are_v9_identical() -> None:
    raw = _raw_manifest()
    before = auditor.canonical_json_sha256(raw)
    first = manifest_semantics.parse_scene_manifest_dict(raw)
    second = manifest_semantics.parse_scene_manifest_dict(deepcopy(raw))
    assert manifest_semantics.manifest_sha256(first) == manifest_semantics.manifest_sha256(
        second
    )
    assert source_v4._rendered_boxes(first) == source_v4._rendered_boxes(second)
    assert auditor.canonical_json_sha256(raw) == before


def test_v10_science_region_is_exactly_v9_after_authority_rename() -> None:
    v9_source = Path(predecessor.__file__).read_text(encoding="utf-8")
    v10_source = Path(auditor.__file__).read_text(encoding="utf-8")

    def region(source: str) -> str:
        start = source.index("def _validate_integer_fields(")
        end = source.index("def _strict_canonical_json_object(", start)
        return source[start:end]

    normalized = re.sub(r"V10", "V9", region(v10_source))
    normalized = re.sub(r"v10", "v9", normalized)
    assert ast.dump(ast.parse(normalized), include_attributes=False) == ast.dump(
        ast.parse(region(v9_source)), include_attributes=False
    )


def test_v10_unmodified_replay_functions_are_exactly_v9_ast() -> None:
    names = {
        "_read_exact_source_json",
        "_source_record_for_endpoint",
        "_summary_source_entry",
        "_source_path",
        "_install_reviewed_source_semantics",
        "_find_source_frame",
        "_recompute_exact_sample_task",
        "_exact_sample_recomputer",
    }

    def definitions(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }

    expected = definitions(predecessor)
    observed = definitions(auditor)
    assert set(expected) == set(observed) == names
    for name in names:
        normalized = observed[name].replace("V10", "V9").replace("v10", "v9")
        assert normalized == expected[name]


def test_v10_transaction_methods_are_exactly_v9_ast() -> None:
    def methods(module: object) -> dict[str, str]:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        transaction = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_ClosedAuditPublicationTransaction"
        )
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in transaction.body
            if isinstance(node, ast.FunctionDef)
        }

    expected = methods(predecessor)
    observed = methods(auditor)
    assert set(expected) == set(observed)
    for name, value in observed.items():
        normalized = value.replace("V10", "V9").replace("v10", "v9")
        assert normalized == expected[name]


def test_v10_two_boundary_functions_reduce_exactly_to_v9_ast() -> None:
    def function(module: object, name: str) -> ast.FunctionDef:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        return deepcopy(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
        )

    render = function(auditor, "_validate_sample_render_contract")
    render.args.args = [
        argument for argument in render.args.args if argument.arg != "raw_scene_manifest"
    ]
    render.body = [
        statement
        for statement in render.body
        if not (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "raw_fingerprint"
                for target in statement.targets
            )
        )
        and not (
            isinstance(statement, ast.If)
            and (
                "decoded JSON object" in ast.unparse(statement)
                or "changed during raw validation" in ast.unparse(statement)
            )
        )
    ]
    for node in ast.walk(render):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_validate_raw_scene_object_records"
        ):
            node.args = [
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="scene_manifest", ctx=ast.Load()),
                        attr="to_dict",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            ]
    assert ast.dump(render, include_attributes=False) == ast.dump(
        function(predecessor, "_validate_sample_render_contract"),
        include_attributes=False,
    )

    replay = function(auditor, "_recompute_one_exact_sample")
    replay.body = [
        statement
        for statement in replay.body
        if not (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "raw_manifest_fingerprint"
                for target in statement.targets
            )
        )
        and not (
            isinstance(statement, ast.If)
            and "changed during render validation" in ast.unparse(statement)
        )
    ]
    for node in ast.walk(replay):
        if (
            isinstance(node, ast.If)
            and "sample scene manifest semantic hash changed" in ast.unparse(node)
        ):
            assert isinstance(node.test, ast.BoolOp)
            node.test.values = [
                value
                for value in node.test.values
                if "raw_manifest_fingerprint" not in ast.unparse(value)
            ]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_sample_render_contract"
        ):
            node.args.pop(2)
        if isinstance(node, ast.Name) and node.id == "raw_scene_manifest":
            node.id = "manifest_payload"
    assert ast.dump(replay, include_attributes=False) == ast.dump(
        function(predecessor, "_recompute_one_exact_sample"),
        include_attributes=False,
    )


def test_v10_boundary_source_has_only_the_permitted_raw_delta() -> None:
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    render_source = inspect.getsource(auditor._validate_sample_render_contract)
    replay_source = inspect.getsource(auditor._recompute_one_exact_sample)
    assert "scene_manifest.to_dict()" not in source
    assert "_validate_raw_scene_object_records(raw_scene_manifest)" in render_source
    assert "type(raw_scene_manifest) is not dict" in render_source
    assert "parse_scene_manifest_dict(raw_scene_manifest)" in replay_source
    assert replay_source.count("canonical_json_sha256(raw_scene_manifest)") == 3
    assert "manifest_sha256(scene_manifest)" in replay_source


def test_v10_phase_one_accepts_exact_closure_without_opening_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        nonlocal opened
        opened = True
        raise AssertionError("phase one opened a mapped target")

    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", forbidden)
    payload = _authorization_payload()
    phase_one = auditor._validate_authorization_phase_one(
        payload, authorization_file_sha256="c" * 64
    )
    assert opened is False
    assert len(phase_one.sources) == 11
    assert phase_one.auditor_review.implementation_author == auditor.V10_IMPLEMENTATION_AUTHOR


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exact_build_authorized", True),
        ("retry_authorized", True),
        ("dataset_use_authorized", True),
        ("input_dataset_path", ".generated/other"),
        ("v9_audit_failure_content_sha256", "0" * 64),
    ],
)
def test_v10_phase_one_rejects_authority_or_terminal_binding_drift(
    field: str,
    value: object,
) -> None:
    payload = _authorization_payload()
    payload[field] = value
    core = dict(payload)
    core.pop("content_sha256")
    payload["content_sha256"] = auditor.canonical_json_sha256(core)
    with pytest.raises(PermissionError):
        auditor._validate_authorization_phase_one(
            payload, authorization_file_sha256="c" * 64
        )


def test_v10_one_and_six_worker_science_bytes_are_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    arrays = tuple(
        np.zeros(shape, dtype=np.dtype(dtype))
        for _name, dtype, shape in auditor.ARRAY_LAYOUT
    )
    observed = auditor.StoredEndpointEvidence(
        endpoint_identity_sha256=digest,
        arrays=arrays,
        evidence_content_sha256="e" * 64,
        raster_content_sha256="f" * 64,
    )
    manifest = {"content_sha256": "c" * 64, "shards": [{"path": "one"}]}
    sample = [{"endpoint_identity_sha256": digest}]
    population = {"synthetic_population": 1}
    inputs = auditor.AuditInputs(
        plan=SimpleNamespace(pairs=(object(),)), inventory=SimpleNamespace()
    )
    worker_calls: list[int] = []
    monkeypatch.setattr(auditor, "_require_exact_authority", lambda _digest: object())
    monkeypatch.setattr(
        auditor, "_require_real_directory", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr(auditor, "_parse_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        auditor, "_validate_root_file_inventory", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        auditor,
        "_validate_pair_and_endpoint_indexes",
        lambda *_args, **_kwargs: ((), {digest: {}}, {}),
    )
    monkeypatch.setattr(auditor, "_derive_population", lambda *_args: population)
    monkeypatch.setattr(auditor, "_validate_frozen_population", lambda *_args: None)
    monkeypatch.setattr(auditor, "_validate_sample_precommit", lambda *_args: sample)
    monkeypatch.setattr(
        auditor,
        "_validate_shards_parallel",
        lambda *_args, **_kwargs: {digest: observed},
    )

    def recompute(
        _sample: object,
        _endpoints: object,
        _inputs: object,
        workers: int,
        **_kwargs: object,
    ) -> dict[str, tuple[np.ndarray, ...]]:
        worker_calls.append(workers)
        return {digest: arrays}

    monkeypatch.setattr(auditor, "_exact_sample_recomputer", recompute)
    monkeypatch.setattr(
        auditor, "_read_absolute_bound_payload", lambda *_args, **_kwargs: b""
    )
    one = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=1,
    )
    six = auditor._audit_fixed_dataset(
        authorization_sha256="a" * 64,
        expected_manifest_file_sha256="b" * 64,
        inputs=inputs,
        workers=6,
    )
    assert auditor.canonical_json_bytes(one) == auditor.canonical_json_bytes(six)
    assert worker_calls == [1, 6]


def test_v10_worker_wiring_reauthorizes_and_hides_accelerators() -> None:
    recomputer = inspect.getsource(auditor._exact_sample_recomputer)
    task = inspect.getsource(auditor._recompute_exact_sample_task)
    initializer = inspect.getsource(auditor._initialize_exact_worker)
    assert 'multiprocessing.get_context("spawn")' in recomputer
    assert "initializer=_initialize_exact_worker" in recomputer
    assert "_require_exact_authority(authorization_sha256)" in task
    assert "_require_exact_authority(authorization_sha256)" in initializer
    old = {name: os.environ.get(name) for name in (*auditor.THREAD_ENVIRONMENT, *auditor.ACCELERATOR_ENVIRONMENT)}
    try:
        auditor._set_worker_environment()
        assert all(os.environ[name] == "1" for name in auditor.THREAD_ENVIRONMENT)
        assert all(os.environ[name] == "" for name in auditor.ACCELERATOR_ENVIRONMENT)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_v10_failure_publication_is_additive_and_preserves_v9_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "development_raw_supervision_v1"
    dataset.mkdir()
    v9_failure = tmp_path / "development_raw_supervision_v1.audit_v9.failed.json"
    v9_bytes = b'{"status":"terminal_failed_no_dataset_authority"}\n'
    v9_failure.write_bytes(v9_bytes)
    v10_report = tmp_path / "development_raw_supervision_v1.audit_v10.json"
    v10_failure = tmp_path / "development_raw_supervision_v1.audit_v10.failed.json"
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", v10_report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", v10_failure)
    auditor._publish_terminal_audit_failure(
        authorization_sha256="a" * 64,
        error=RuntimeError("synthetic author failure"),
    )
    assert v9_failure.read_bytes() == v9_bytes
    value = json.loads(v10_failure.read_bytes())
    assert value["schema"] == auditor.AUDIT_FAILURE_SCHEMA
    assert value["status"] == "terminal_failed_no_dataset_authority"
    assert value["retry_authorized"] is False
    with pytest.raises(FileExistsError):
        auditor._publish_terminal_audit_failure(
            authorization_sha256="a" * 64,
            error=RuntimeError("second attempt"),
        )
    assert v9_failure.read_bytes() == v9_bytes


def test_v10_production_and_cli_surface_is_fixed_and_audit_only() -> None:
    signature = inspect.signature(auditor.execute_exact_audit_v10)
    assert tuple(signature.parameters) == ("authorization_sha256", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert not any(
        re.search(r"go2_shared_jepa_v5_raw_supervision_auditor_v9(?:$|\.)", module)
        or "go2_shared_jepa_v5_raw_supervision_builder" in module
        for module in imports
    )
    assert "test_hook" not in source
    assert "importlib" not in source
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert not any("skip" in name or "retry" in name for name in function_names)
    cli = Path("scripts/audit_go2_shared_jepa_v5_raw_supervision_v10.py").read_text(
        encoding="utf-8"
    )
    assert "raw_supervision_auditor_v10" in cli
    assert "execute_exact_audit_v10" in cli
    assert "--authorization-sha256" in cli and "--workers" in cli
    for forbidden in ("--path", "--retry", "--fallback", "--mode", "trainer"):
        assert forbidden not in cli
