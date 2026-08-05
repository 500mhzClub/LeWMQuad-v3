"""Source-free author tests for the standalone raw-supervision Builder V4."""
from __future__ import annotations

import ast
from collections.abc import Mapping as MappingABC
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import types
from typing import Any, Callable

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v4 as builder
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as metadata
from lewm.tests import go2_shared_jepa_v5_raw_supervision_builder_v4_test_support as support
from scripts.build_go2_observable_camera_ray_fit_v4 import synthetic_scene_jobs


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v4_"
    "authorization_successor_amendment_2026-07-13.md"
)
AMENDMENT_SHA256 = (
    "a535ee8de9a6002f5548f3c3894548ddb42cd9d077eccbb9ca922a41611ced83"
)
V3_INVALIDATION = {
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v3_"
    "authorization_successor_amendment_2026-07-13.md": (
        "501062e2eba625cf4d7ab28810f2a629652c327c770366c07f3b788f3f6f8b2b"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
    "structural_invalidation_2026-07-13.md": (
        "db86ea8bb72478b0f032068151a3c492660444b1fad21b33c700b658de33e213"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
    "author_handoff_2026-07-13.md": (
        "a3b66f150320aa790c2a9aa3c8aa0f437824cc619de12349448155559642fe23"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py": (
        "423164701e735c17dca10449434d4d96692180ee148d2a222c9af9b357a83043"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py": (
        "f1258680802be18ad77ca4cf0fa1aacef5e941d9aca40fa68a6d7d8105892445"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py": (
        "4e111e961ed3e8a7250f6c0cfbff4033c5cb6487c67cbbb9d65d389081e9fd19"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result.pop("content_sha256", None)
    return {
        **result,
        "content_sha256": builder.canonical_json_sha256(result),
    }


def _synthetic_job_and_pair() -> tuple[
    builder.PreparedSceneJobV4, dict[str, Any]
]:
    frames = synthetic_scene_jobs(2)
    scene_id = "synthetic_builder_v4_scene"
    family = "synthetic_builder_v4_family"
    endpoints: list[builder.PreparedEndpointV4] = []
    identities: list[str] = []
    for index, side in enumerate(("current", "next")):
        frame = frames[index].frames[0]
        identity = {
            "dataset_role": "train",
            "scene_id": scene_id,
            "episode_id": "synthetic_episode",
            "env_index": 0,
            "episode_step": index,
            "frame_index": index,
            "timestamp_ns": 1_000 + index,
            "image_sha256": frame.image_sha256,
        }
        identity_sha256 = builder.canonical_json_sha256(identity)
        core = {
            "schema": metadata.ENDPOINT_SCHEMA,
            "identity": identity,
            "identity_sha256": identity_sha256,
            "image_path_metadata_only": frame.image_path_metadata_only,
            "frames_jsonl_sha256": hashlib.sha256(
                f"frames:{scene_id}".encode("ascii")
            ).hexdigest(),
            "scene_manifest_sha256": hashlib.sha256(
                f"manifest:{scene_id}".encode("ascii")
            ).hexdigest(),
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        }
        endpoint = {
            **core,
            "content_sha256": builder.canonical_json_sha256(core),
        }
        endpoints.append(
            builder.PreparedEndpointV4(
                plan_endpoint=endpoint,
                family=family,
                frame=frame,
            )
        )
        identities.append(identity_sha256)
    job = builder.PreparedSceneJobV4(
        scene_id=scene_id,
        role="train",
        family=family,
        endpoints=tuple(endpoints),
    )
    pair_core = {
        "schema": metadata.PAIR_SCHEMA,
        "dataset_role": "train",
        "global_row": 0,
        "scene_id": scene_id,
        "family": family,
        "current_endpoint_sha256": identities[0],
        "next_endpoint_sha256": identities[1],
    }
    return job, {
        **pair_core,
        "content_sha256": builder.canonical_json_sha256(pair_core),
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"), key=str)
        if path.is_file()
    }


def test_v4_amendment_and_v3_invalidation_are_frozen() -> None:
    assert _sha(ROOT / AMENDMENT) == AMENDMENT_SHA256
    assert {
        relative: _sha(ROOT / relative) for relative in V3_INVALIDATION
    } == V3_INVALIDATION
    assert builder.FROZEN_PARENT_HASHES[AMENDMENT] == AMENDMENT_SHA256
    assert all(
        builder.FROZEN_PARENT_HASHES[relative] == digest
        for relative, digest in V3_INVALIDATION.items()
    )


def test_canonical_v4_roles_schemas_and_authors_are_exact() -> None:
    assert builder.AUTHORIZED_ROLE_PATHS == (
        (
            "builder_source",
            "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v4.py",
        ),
        (
            "builder_cli",
            "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v4.py",
        ),
        (
            "builder_test",
            "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v4.py",
        ),
        (
            "builder_handoff",
            "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_"
            "author_handoff_2026-07-13.md",
        ),
        (
            "builder_review",
            "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v4_"
            "independent_review_2026-07-13.json",
        ),
        (
            "auditor_source",
            "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v4.py",
        ),
        (
            "auditor_cli",
            "scripts/audit_go2_shared_jepa_v5_raw_supervision_v4.py",
        ),
        (
            "auditor_test",
            "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v4.py",
        ),
        (
            "auditor_review",
            "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v4_"
            "independent_review_2026-07-13.json",
        ),
    )
    assert builder.AUTHORIZATION_SCHEMA.endswith("build_authorization_v4")
    assert builder.REVIEW_BINDING_SCHEMA.endswith("review_binding_v4")
    assert builder.BUILDER_REVIEW_SCHEMA.endswith(
        "builder_v4_independent_review_v1"
    )
    assert builder.AUDITOR_REVIEW_SCHEMA.endswith(
        "auditor_v4_independent_review_v1"
    )
    assert builder.BUILDER_IMPLEMENTATION_AUTHOR == "/root/raw_builder_arch"
    assert builder.AUDITOR_IMPLEMENTATION_AUTHOR == "/root/raw_auditor_author"


def test_import_surface_exposes_no_legacy_builder_or_nonpure_loader() -> None:
    namespace = vars(builder)
    forbidden_names = {
        "execute_exact_build_v1",
        "execute_exact_build_v2",
        "execute_exact_build_v3",
        "load_frozen_development_metadata",
        "load_frozen_development_source_inventory",
        "plan_v5",
        "v4_builder",
        "_v1",
        "_v2",
        "_v3",
        "__getattr__",
    }
    assert forbidden_names.isdisjoint(namespace)
    legacy_modules = {
        value.__name__
        for value in namespace.values()
        if isinstance(value, types.ModuleType)
        and (
            "raw_supervision_builder_v1" in value.__name__
            or "raw_supervision_builder_v2" in value.__name__
            or "raw_supervision_builder_v3" in value.__name__
            or "raw_supervision_auditor" in value.__name__
        )
    }
    assert legacy_modules == set()
    assert set(builder.__all__) == {
        "ACCELERATOR_ENVIRONMENT",
        "ARRAY_LAYOUT",
        "AUTHORIZATION_PATH",
        "CANONICAL_OUTPUT",
        "DATASET_SCHEMA",
        "FAILURE_RECEIPT",
        "MAX_WORKERS",
        "PreparedEndpointV4",
        "PreparedSceneJobV4",
        "RawSupervisionBuildError",
        "THREAD_ENVIRONMENT",
        "canonical_json_bytes",
        "canonical_json_sha256",
        "execute_exact_build_v4",
    }


def test_production_signatures_have_no_injection_seams() -> None:
    phase_two = inspect.signature(builder._validate_authorization_phase_two)
    assert tuple(phase_two.parameters) == ("phase_one",)
    exact = inspect.signature(builder.execute_exact_build_v4)
    assert tuple(exact.parameters) == ("authorization_sha256", "workers")
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for value in exact.parameters.values()
    )
    build = inspect.signature(builder._build_exact_prepared_dataset_v4)
    forbidden = {
        "output_directory",
        "prepublication_validator",
        "callback",
        "function",
        "reader",
        "repository_root",
        "exact",
        "skip",
        "mapping",
    }
    assert forbidden.isdisjoint(build.parameters)
    for name in (
        "_run_exact_scene_load_pool",
        "_run_exact_source_revalidation_pool",
    ):
        assert forbidden.isdisjoint(inspect.signature(getattr(builder, name)).parameters)
    assert tuple(field.name for field in fields(builder.AcceptedAuthorizationV4)) == (
        "authorization_file_sha256",
        "authorization_content_sha256",
        "source_map_sha256",
    )
    receipt = builder.AcceptedAuthorizationV4("1" * 64, "2" * 64, "3" * 64)
    assert not isinstance(receipt, MappingABC)


def test_source_uses_only_fixed_worker_targets_and_authorized_initializers() -> None:
    source_path = ROOT / builder.__file__.removeprefix(str(ROOT) + "/")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        "raw_supervision_builder_v1" in module
        or "raw_supervision_builder_v2" in module
        or "raw_supervision_builder_v3" in module
        or "raw_supervision_auditor" in module
        for module in imported
    )
    submitted: list[str] = []
    initializers: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "submit"
        ):
            assert node.args and isinstance(node.args[0], ast.Name)
            submitted.append(node.args[0].id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ProcessPoolExecutor":
                keywords = {item.arg: item.value for item in node.keywords}
                assert isinstance(keywords.get("initializer"), ast.Name)
                initializers.append(keywords["initializer"].id)
                assert "initargs" in keywords
    assert set(submitted) == {
        "_write_prepared_scene_job",
        "_load_exact_scene_job",
        "_revalidate_exact_scene_sources",
    }
    assert initializers and set(initializers) == {"_initialize_exact_worker"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            assert all(
                not (
                    isinstance(target, ast.Name)
                    and target.id == "_require_exact_authority"
                )
                and not (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_require_exact_authority"
                )
                for target in targets
            )


Mutation = Callable[[dict[str, Any]], None]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["source_map"].pop(),
        lambda value: value["source_map"].append(
            {"role": "extra", "path": "review/extra.json", "sha256": "1" * 64}
        ),
        lambda value: value["source_map"][1].update(value["source_map"][0]),
        lambda value: value["source_map"][0].__setitem__(
            "path", "lewm/datasets/../unbound.py"
        ),
        lambda value: value["source_map"].__setitem__(
            slice(0, 2), [value["source_map"][1], value["source_map"][0]]
        ),
        lambda value: value["builder_review"]["candidate"][0].__setitem__(
            "sha256", "2" * 64
        ),
        lambda value: value.__setitem__("unexpected", False),
        lambda value: value["builder_review"].__setitem__(
            "implementation_author", "/synthetic/wrong_author"
        ),
        lambda value: value["auditor_review"].__setitem__(
            "reviewer", value["builder_review"]["reviewer"]
        ),
    ],
)
def test_phase_one_adversaries_reach_zero_openers(
    mutation: Mutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = support.valid_authorization()
    mutation(authority)
    authority = _rehash(authority)
    opened: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        opened.append("opened")
        raise AssertionError("phase-one rejection reached a byte opener")

    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    with pytest.raises((PermissionError, builder.RawSupervisionBuildError)):
        builder._validate_authorization_phase_one(
            authority,
            authorization_file_sha256="4" * 64,
        )
    assert opened == []


def test_synthetic_phase_two_checks_exact_nine_in_order() -> None:
    authority, raw_by_role, digests = support.valid_authorization()
    phase_one = builder._validate_authorization_phase_one(
        authority,
        authorization_file_sha256="5" * 64,
    )
    role_by_path = dict(
        (path, role) for role, path in builder.AUTHORIZED_ROLE_PATHS
    )

    def reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        role = role_by_path[path.relative_to(repository_root).as_posix()]
        assert expected_sha256 == digests[role]
        return raw_by_role[role]

    receipt, opened = support.validate_phase_two_for_tests(
        phase_one,
        repository_root=Path("/synthetic/repository"),
        reader=reader,
    )
    assert opened == tuple(role for role, _path in builder.AUTHORIZED_ROLE_PATHS)
    assert receipt.authorization_file_sha256 == "5" * 64
    assert receipt.authorization_content_sha256 == authority["content_sha256"]
    assert receipt.source_map_sha256 == builder.canonical_json_sha256(
        authority["source_map"]
    )


def test_fabricated_phase_capsule_rejects_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = support.valid_authorization()
    phase_one = builder._validate_authorization_phase_one(
        authority,
        authorization_file_sha256="6" * 64,
    )
    fabricated = replace(phase_one, source_map_sha256="7" * 64)
    opened: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        opened.append("opened")
        raise AssertionError("fabricated capsule reached a byte opener")

    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    with pytest.raises(PermissionError, match="capsule was fabricated"):
        builder._validate_authorization_phase_two(fabricated)
    assert opened == []


def test_absent_authority_reaches_no_byte_or_metadata_opener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("opened")
        raise AssertionError("absent authority reached protected data")

    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", tmp_path / "output")
    monkeypatch.setattr(builder, "FAILURE_RECEIPT", tmp_path / "failure.json")
    monkeypatch.setattr(builder, "_read_bound_regular_file", forbidden)
    monkeypatch.setattr(metadata, "load_frozen_development_metadata", forbidden)
    monkeypatch.setattr(
        metadata, "load_frozen_development_source_inventory", forbidden
    )
    with pytest.raises(PermissionError, match="authorization is absent"):
        builder.execute_exact_build_v4(
            authorization_sha256="8" * 64,
            workers=1,
        )
    assert calls == []
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "failure.json").exists()


@pytest.mark.parametrize("raw", [b"{not json}\n", b'{"x":1,"x":2}\n'])
def test_malformed_authority_opens_only_the_fixed_authority_file(
    raw: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = tmp_path / "authorization.json"
    authority.write_bytes(raw)
    opened: list[Path] = []
    metadata_calls: list[str] = []
    def tracking_reader(
        *, repository_root: Path, path: Path, expected_sha256: str
    ) -> bytes:
        assert repository_root == builder.ROOT
        assert path == authority
        assert expected_sha256 == hashlib.sha256(raw).hexdigest()
        opened.append(path)
        return raw

    monkeypatch.setattr(builder, "AUTHORIZATION_PATH", authority)
    monkeypatch.setattr(builder, "_read_bound_regular_file", tracking_reader)
    monkeypatch.setattr(
        metadata,
        "load_frozen_development_metadata",
        lambda *_args, **_kwargs: metadata_calls.append("metadata"),
    )
    with pytest.raises(builder.RawSupervisionBuildError):
        builder._require_exact_authority(hashlib.sha256(raw).hexdigest())
    assert opened == [authority]
    assert metadata_calls == []


@pytest.mark.parametrize("workers", [False, True, 0, 7, 1.0, "1"])
def test_worker_bound_rejects_before_authority(
    workers: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(_digest: str) -> Any:
        calls.append("authority")
        raise AssertionError("invalid worker count reached authority")

    monkeypatch.setattr(builder, "_require_exact_authority", forbidden)
    with pytest.raises(ValueError, match="workers"):
        builder.execute_exact_build_v4(
            authorization_sha256="9" * 64,
            workers=workers,  # type: ignore[arg-type]
        )
    assert calls == []


def test_worker_initializer_authorizes_and_hides_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        builder,
        "_require_exact_authority",
        lambda digest: calls.append(digest),
    )
    builder._initialize_exact_worker("a" * 64)
    assert calls == ["a" * 64]
    assert all(os.environ[name] == "1" for name in builder.THREAD_ENVIRONMENT)
    assert all(os.environ[name] == "" for name in builder.ACCELERATOR_ENVIRONMENT)


def test_deterministic_shard_science_and_layout_without_exact_data(
    tmp_path: Path,
) -> None:
    job, pair = _synthetic_job_and_pair()
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "shards").mkdir(parents=True)
    (second / "shards").mkdir(parents=True)
    first_result = support.write_prepared_scene_job_for_tests(job, first)
    reversed_job = builder.PreparedSceneJobV4(
        scene_id=job.scene_id,
        role=job.role,
        family=job.family,
        endpoints=tuple(reversed(job.endpoints)),
    )
    second_result = support.write_prepared_scene_job_for_tests(
        reversed_job, second
    )
    assert first_result["shard"] == second_result["shard"]
    assert _tree_hashes(first) == _tree_hashes(second)
    files = {
        item["path"]: item for item in first_result["shard"]["files"]
    }
    assert tuple(name for name, _dtype, _shape in builder.ARRAY_LAYOUT) == (
        "camera_origin_body_m.f4",
        "camera_basis_body_fru.f4",
        "ground_plane_z_body_m.f4",
        "ground_support_in_frustum.u1",
        "ground_support_clear_to_target.u1",
        "pixel_hit_mask.u1",
        "pixel_first_hit_distance_m.f4",
        "raster_labels.u1",
    )
    assert files["ground_plane_z_body_m.f4"]["shape"] == [2]
    assert files["raster_labels.u1"]["dtype"] == "|u1"
    assert files["raster_labels.u1"]["shape"] == [2, 64, 64]
    ordered_jobs, ordered_pairs = builder._validate_jobs_and_pairs(
        (job,), (pair,)
    )
    assert ordered_jobs == (job,)
    assert ordered_pairs == (pair,)


def test_join_rejects_duplicate_endpoint_and_cross_role() -> None:
    job, pair = _synthetic_job_and_pair()
    duplicate = builder.PreparedSceneJobV4(
        scene_id=job.scene_id,
        role=job.role,
        family=job.family,
        endpoints=(job.endpoints[0], job.endpoints[0]),
    )
    with pytest.raises(ValueError, match="scheduled more than once"):
        builder._validate_jobs_and_pairs((duplicate,), (pair,))
    crossed = deepcopy(pair)
    crossed["dataset_role"] = "checkpoint_selection"
    crossed = _rehash(crossed)
    with pytest.raises(ValueError, match="role, scene, or family"):
        builder._validate_jobs_and_pairs((job,), (crossed,))


def test_bound_reader_and_publication_parent_reject_alias_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(PermissionError):
        builder._read_bound_regular_file(
            repository_root=tmp_path,
            path=alias,
            expected_sha256=_sha(source),
        )
    container = tmp_path / "container"
    container.mkdir()
    retained = builder._open_publication_parent(container)
    moved = tmp_path / "moved"
    try:
        container.rename(moved)
        container.mkdir()
        with pytest.raises(builder.RawSupervisionBuildError):
            retained.validate()
    finally:
        retained.close()


def test_v2_block_reproducers_are_absent() -> None:
    assert "_run_authorized_scene_pool" not in vars(builder)
    assert "_call_v1_load_parent_contracts" not in vars(builder)
    assert "_call_v1_load_exact_scene_job" not in vars(builder)
    assert "_call_v1_revalidate_exact_scene_sources" not in vars(builder)
    assert tuple(inspect.signature(builder._validate_authorization_phase_two).parameters) == (
        "phase_one",
    )
    assert support.PRODUCTION_ELIGIBLE is False
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "from lewm.tests" not in source
    assert "_require_exact_authority =" not in source
