"""Source-free author tests for the standalone raw-supervision Builder V8."""
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
from unittest.mock import patch
from typing import Any, Callable, Mapping

import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v8 as builder
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v6 as builder_v6
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as metadata
from lewm.tests.test_go2_shared_jepa_v5_raw_supervision_builder_v6 import (
    _open_synthetic_transaction as _open_v6_synthetic_transaction,
)
from scripts.build_go2_observable_camera_ray_fit_v4 import synthetic_scene_jobs


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v8_"
    "terminal_quiet_successor_amendment_2026-07-13.md"
)
AMENDMENT_SHA256 = (
    "054de82d8648cd6be7edff01b82d549ec916700ebffad51698d4c2041edc6c88"
)
REBINDING_AMENDMENT = (
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_v8_existing_agent_identity_"
    "rebinding_amendment_2026-07-13.md"
)
REBINDING_AMENDMENT_SHA256 = (
    "392745c80ca2c6e7a103cca4a55c3614cd2c988de9a379fba950b0087df41698"
)
V7_PREDECESSOR_EVIDENCE = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py": (
        "c79e68a2dcccb0fba937a9e6cd0ab778fc267b99473163c6c3c0bdbe6d1ac2ab"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v7.py": (
        "9fdecaac622f0e5c1022c6a6298557da0ad0d7effb4b350330536da177cbf432"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v7.py": (
        "cb03351928d9d9849736a53b57d338cab10ae60bb14de062e3218e01901e99da"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
    "author_handoff_2026-07-13.md": (
        "b4fc01993b5d47e11f789192cfcf0d4e9a8ea5cce865ef0624cbce7e6379642d"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
    "independent_review_2026-07-13.json": (
        "85d1a111e10eaac865a80cebd97e771b39eaa47f6ebcf6ffe6716ed445a1ff46"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v7.py": (
        "3550917e36d1401f8ad9c895afcf591b3226b2e0c5a09f4ad427d0b04bb1490e"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v7.py": (
        "9940d35e4e33b628bf64c4947cb1f92a68e1413e20e63fd0b9080728a64f949e"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v7.py": (
        "6d123d39014fd9c3dc7b34d113e665861536010d79117a3004cb8ee1484e894f"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_"
    "author_handoff_2026-07-13.md": (
        "1351a2641025735a3a96d50283a7119f2ce02f7c49578e656133b1c48a46fd21"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v7_"
    "independent_qa.py": (
        "5d0ffc94070bdb60dfb9a90a062e5598931fa793855386e758192e6d4c3078c0"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_"
    "independent_review_2026-07-13.md": (
        "e4e17116173d2814ce202b9f7804af6dd5b8bcb36c65488b0c616c4ad8cf0efb"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_"
    "independent_review_2026-07-13.json": (
        "180cce0d86b5b226ddcdb694aea53f3cc22b4e43f75212a24950cff095bb1545"
    ),
}
V6_BLOCK = {
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_auditor_v6_"
    "authorization_successor_amendment_2026-07-13.md": (
        "09ced36b2eab16585c759e65f7eda844f76006b93de013e5f7057fb9a8e7a137"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v6.py": (
        "88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v6.py": (
        "089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6.py": (
        "acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
    "author_handoff_2026-07-13.md": (
        "d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v6_"
    "independent_qa.py": (
        "2c74e3315be3443bab11a3b7896df4df29d8b233b634b7ab539123386bc0c89a"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
    "independent_review_2026-07-13.json": (
        "55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v6.py": (
        "cf67c993427950c147860f9afe0e7661b2cb6841ccec27a867868cc34c7c00b8"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v6.py": (
        "de37e42d09d949ac5ca1cd8e4ebba2d32e757ef72cc769a151f814cc8fe84ffe"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v6.py": (
        "6cc84a493cb677437385efd3c00a8120b26748e8cabb2abd76d0f4825deaf764"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v6_"
    "author_handoff_2026-07-13.md": (
        "f7e0c1244eb55a826dfc90f7f633d88f4c3390ae3c8551949028a9757da4dc15"
    ),
}


PRODUCTION_ELIGIBLE = False
Reader = Callable[..., bytes]


def _source(role: str, path: str, digest: str) -> dict[str, str]:
    return {"role": role, "path": path, "sha256": digest}


def _hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(core))
    value.pop("content_sha256", None)
    return {**value, "content_sha256": builder.canonical_json_sha256(value)}


def _review_raw(
    *,
    kind: str,
    schema: str,
    reviewer: str,
    author: str,
    candidate: list[dict[str, str]],
) -> tuple[bytes, dict[str, Any]]:
    value = _hashed(
        {
            "schema": schema,
            "verdict": "PASS",
            "reviewer": reviewer,
            "implementation_author": author,
            "candidate": candidate,
            "authority": builder._expected_review_authority(kind),
        }
    )
    return builder.canonical_json_bytes(value) + b"\n", value


def valid_authorization() -> tuple[
    dict[str, Any], dict[str, bytes], dict[str, str]
]:
    """Return a structurally valid, non-production synthetic authority closure."""

    digest_by_role = {
        role: hashlib.sha256(f"synthetic:{role}".encode("ascii")).hexdigest()
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
    }
    source_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    builder_candidate = [
        source_by_role[role] for role in builder.BUILDER_CANDIDATE_ROLES
    ]
    auditor_candidate = [
        source_by_role[role] for role in builder.AUDITOR_CANDIDATE_ROLES
    ]
    builder_raw, builder_review = _review_raw(
        kind="builder",
        schema=builder.BUILDER_REVIEW_SCHEMA,
        reviewer="/root/synthetic_builder_reviewer",
        author=builder.BUILDER_IMPLEMENTATION_AUTHOR,
        candidate=builder_candidate,
    )
    auditor_raw, auditor_review = _review_raw(
        kind="auditor",
        schema=builder.AUDITOR_REVIEW_SCHEMA,
        reviewer="/root/synthetic_auditor_reviewer",
        author=builder.AUDITOR_IMPLEMENTATION_AUTHOR,
        candidate=auditor_candidate,
    )
    digest_by_role["builder_review"] = hashlib.sha256(builder_raw).hexdigest()
    digest_by_role["auditor_review"] = hashlib.sha256(auditor_raw).hexdigest()
    source_by_role = {
        role: _source(role, path, digest_by_role[role])
        for role, path in builder.AUTHORIZED_ROLE_PATHS
    }
    builder_candidate = [
        source_by_role[role] for role in builder.BUILDER_CANDIDATE_ROLES
    ]
    auditor_candidate = [
        source_by_role[role] for role in builder.AUDITOR_CANDIDATE_ROLES
    ]

    def binding(
        *,
        review_role: str,
        schema: str,
        review: Mapping[str, Any],
        candidate: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "schema": builder.REVIEW_BINDING_SCHEMA,
            "review_schema": schema,
            "verdict": "PASS",
            "reviewer": review["reviewer"],
            "implementation_author": review["implementation_author"],
            "path": builder.ROLE_PATH_BY_NAME[review_role],
            "file_sha256": digest_by_role[review_role],
            "content_sha256": review["content_sha256"],
            "candidate": deepcopy(candidate),
        }

    authorization = _hashed(
        {
            "schema": builder.AUTHORIZATION_SCHEMA,
            "exact_build_authorized_after_independent_reviews": True,
            "builder_review": binding(
                review_role="builder_review",
                schema=builder.BUILDER_REVIEW_SCHEMA,
                review=builder_review,
                candidate=builder_candidate,
            ),
            "auditor_review": binding(
                review_role="auditor_review",
                schema=builder.AUDITOR_REVIEW_SCHEMA,
                review=auditor_review,
                candidate=auditor_candidate,
            ),
            "source_map": [
                deepcopy(source_by_role[role])
                for role, _path in builder.AUTHORIZED_ROLE_PATHS
            ],
        }
    )
    raw_by_role = {
        role: (
            builder_raw
            if role == "builder_review"
            else auditor_raw
            if role == "auditor_review"
            else f"synthetic reviewed source {role}\n".encode("ascii")
        )
        for role, _path in builder.AUTHORIZED_ROLE_PATHS
    }
    return authorization, raw_by_role, digest_by_role


def validate_phase_two_for_tests(
    phase_one: builder.PhaseOneAuthorizationV8,
    *,
    repository_root: Path,
    reader: Reader,
) -> tuple[builder.AcceptedAuthorizationV8, tuple[str, ...]]:
    """Mirror phase two with injected bytes, outside all production modules."""

    if PRODUCTION_ELIGIBLE:
        raise AssertionError("synthetic authority helper became production eligible")
    if type(phase_one) is not builder.PhaseOneAuthorizationV8:
        raise TypeError("test phase two requires a completed V8 phase-one capsule")
    embedded = builder._strict_canonical_json_object(
        phase_one.canonical_payload + b"\n",
        name="phase-one authorization capsule",
    )
    revalidated = builder._validate_authorization_phase_one(
        embedded,
        authorization_file_sha256=phase_one.authorization_file_sha256,
    )
    if revalidated != phase_one:
        raise PermissionError("phase-one authorization capsule was fabricated")
    payload_by_role: dict[str, bytes] = {}
    opened: list[str] = []
    for source in phase_one.sources:
        payload_by_role[source.role] = reader(
            repository_root=repository_root,
            path=(repository_root / source.path).absolute(),
            expected_sha256=source.sha256,
        )
        opened.append(source.role)
    builder._validate_review_record(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    builder._validate_review_record(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    return (
        builder.AcceptedAuthorizationV8(
            authorization_file_sha256=phase_one.authorization_file_sha256,
            authorization_content_sha256=phase_one.authorization_content_sha256,
            source_map_sha256=phase_one.source_map_sha256,
        ),
        tuple(opened),
    )


def write_prepared_scene_job_for_tests(
    job: builder.PreparedSceneJobV8,
    staging_root: Path,
) -> dict[str, Any]:
    """Run one fixed construction worker with synthetic test-only authority."""

    if PRODUCTION_ELIGIBLE:
        raise AssertionError("synthetic construction helper became production eligible")
    receipt = builder.AcceptedAuthorizationV8("a" * 64, "b" * 64, "c" * 64)
    with patch.object(builder, "_require_exact_authority", return_value=receipt):
        return builder._write_prepared_scene_job(
            job,
            str(staging_root),
            "a" * 64,
        )


class _InlineFuture:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _InlineExecutor:
    def __init__(
        self,
        *,
        initializer: Any = None,
        initargs: tuple[Any, ...] = (),
        **_kwargs: Any,
    ) -> None:
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def submit(self, function: Any, *args: Any) -> _InlineFuture:
        return _InlineFuture(function(*args))


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
    builder.PreparedSceneJobV8, dict[str, Any]
]:
    frames = synthetic_scene_jobs(2)
    scene_id = "synthetic_builder_v5_scene"
    family = "synthetic_builder_v5_family"
    endpoints: list[builder.PreparedEndpointV8] = []
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
            builder.PreparedEndpointV8(
                plan_endpoint=endpoint,
                family=family,
                frame=frame,
            )
        )
        identities.append(identity_sha256)
    job = builder.PreparedSceneJobV8(
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


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _ordered_calls(source: Path, function_name: str) -> list[tuple[int, str]]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = _function(tree, function_name)
    return sorted(
        (
            (node.lineno, _call_name(node))
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        ),
        key=lambda item: item[0],
    )


def _normalized_top_level_definitions(
    path: Path, *, normalize_v7: bool
) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    if normalize_v7:
        source = source.replace("V7", "V8").replace("v7", "v8")
    tree = ast.parse(source)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _open_synthetic_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    builder._ClosedPublicationTransaction,
    builder._RetainedPublicationParent,
    dict[str, Path],
]:
    source_parent = tmp_path / "sources"
    publication_parent = tmp_path / "publication"
    source_parent.mkdir()
    publication_parent.mkdir()
    source = source_parent / "source.bin"
    source.write_bytes(b"frozen source bytes\n")
    destination = publication_parent / "dataset"
    staging_name = ".dataset.staging.synthetic"
    staging = publication_parent / staging_name
    staging.mkdir(mode=0o700)
    data = staging / "data.bin"
    data.write_bytes(b"validated staging bytes\n")
    manifest = builder._with_content_sha256(
        {"schema": "synthetic_closed_publication_transaction_v1"}
    )
    (staging / "manifest.json").write_bytes(
        builder.canonical_json_bytes(manifest) + b"\n"
    )
    expected_files = (
        {
            "path": "data.bin",
            "byte_count": data.stat().st_size,
            "file_sha256": _sha(data),
        },
    )
    retained = builder._open_publication_parent(publication_parent)
    identity = builder._named_directory_identity(retained.parent_fd, staging_name)
    monkeypatch.setattr(
        builder,
        "_exact_publication_source_hashes",
        lambda _context: {source: _sha(source)},
    )
    try:
        transaction = builder._ClosedPublicationTransaction(
            context=object(),  # type: ignore[arg-type]
            retained=retained,
            staging=staging,
            staging_name=staging_name,
            staging_identity=identity,
            destination=destination,
            expected_files=expected_files,
            manifest=manifest,
        )
    except BaseException:
        retained.close()
        raise
    return transaction, retained, {
        "source": source,
        "source_parent": source_parent,
        "staging": staging,
        "data": data,
        "destination": destination,
    }


def _publish_synthetic_transaction(
    transaction: builder._ClosedPublicationTransaction,
    retained: builder._RetainedPublicationParent,
) -> None:
    transaction.validate_before_rename()
    transaction.rename_owned()
    retained.refresh_after_owned_mutation()
    transaction.validate_after_rename()
    os.fsync(retained.parent_fd)
    transaction.require_final_quiet()


def test_v8_amendment_and_v6_block_are_frozen() -> None:
    assert _sha(ROOT / AMENDMENT) == AMENDMENT_SHA256
    assert _sha(ROOT / REBINDING_AMENDMENT) == REBINDING_AMENDMENT_SHA256
    assert {
        relative: _sha(ROOT / relative) for relative in V7_PREDECESSOR_EVIDENCE
    } == V7_PREDECESSOR_EVIDENCE
    assert {relative: _sha(ROOT / relative) for relative in V6_BLOCK} == V6_BLOCK
    assert builder.FROZEN_PARENT_HASHES[AMENDMENT] == AMENDMENT_SHA256
    assert (
        builder.FROZEN_PARENT_HASHES[REBINDING_AMENDMENT]
        == REBINDING_AMENDMENT_SHA256
    )
    assert all(
        builder.FROZEN_PARENT_HASHES[relative] == digest
        for relative, digest in V7_PREDECESSOR_EVIDENCE.items()
    )
    assert all(
        builder.FROZEN_PARENT_HASHES[relative] == digest
        for relative, digest in V6_BLOCK.items()
    )
    assert len(builder.FROZEN_PARENT_HASHES) == 69
    assert (
        builder.canonical_json_sha256(builder.FROZEN_PARENT_HASHES)
        == "79fe832122ed335188357a59bad8a031cc235449ef17e6e19ac78de9d5aff669"
    )
    assert {
        relative: _sha(ROOT / relative)
        for relative in builder.FROZEN_PARENT_HASHES
    } == builder.FROZEN_PARENT_HASHES
    assert {
        relative: _sha(ROOT / relative) for relative in builder.REVIEWED_V4_SOURCES
    } == builder.REVIEWED_V4_SOURCES
    review = json.loads(
        (
            ROOT
            / "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v6_"
            "independent_review_2026-07-13.json"
        ).read_text(encoding="utf-8")
    )
    assert review["verdict"] == "BLOCK"
    assert review["content_sha256"] == (
        "c639170b672180c8943e08efaff8d23063e8773488d1ff0f77beeb4ce44dd74b"
    )
    builder_v7_review = json.loads(
        (
            ROOT
            / "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v7_"
            "independent_review_2026-07-13.json"
        ).read_text(encoding="utf-8")
    )
    auditor_v7_review = json.loads(
        (
            ROOT
            / "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v7_"
            "independent_review_2026-07-13.json"
        ).read_text(encoding="utf-8")
    )
    assert builder_v7_review["verdict"] == "PASS"
    assert builder_v7_review["content_sha256"] == (
        "24ffe7b0c8fdba7d0e60636b865d1bf01d443e5527aa8c3db3e8eca0170e6202"
    )
    assert auditor_v7_review["verdict"] == "BLOCK"
    assert auditor_v7_review["content_sha256"] == (
        "da2715ad0d0d31b4be566fdd710ccc328a15eec72d2685d5b530c9565c80b502"
    )


def test_v8_is_mechanical_v7_definition_successor() -> None:
    v7 = _normalized_top_level_definitions(
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v7.py",
        normalize_v7=True,
    )
    v8 = _normalized_top_level_definitions(Path(builder.__file__), normalize_v7=False)
    assert len(v7) == 80
    assert set(v8) == set(v7)
    assert {
        name for name in v8 if v8[name] != v7[name]
    } == {"_review_binding"}


def test_v5_race_is_reproduced_and_v8_transaction_spans_publication() -> None:
    v5_calls = _ordered_calls(
        ROOT / "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v5.py",
        "_build_exact_prepared_dataset_v5",
    )
    v8_calls = _ordered_calls(Path(builder.__file__), "_build_exact_prepared_dataset_v8")
    v5_names = [name for _line, name in v5_calls]
    assert v5_names.count("_revalidate_exact_before_publication") == 1
    assert v5_names.count("_libc_renameat2") == 1
    assert v5_names.index("_revalidate_exact_before_publication") < v5_names.index(
        "_libc_renameat2"
    )
    assert "_validate_staging_inventory" not in v5_names[
        v5_names.index("_revalidate_exact_before_publication") + 1 :
        v5_names.index("_libc_renameat2")
    ]

    v8_names = [name for _line, name in v8_calls]
    assert v8_names.count("_ClosedPublicationTransaction") == 1
    assert v8_names.count("_revalidate_exact_before_publication") == 1
    assert v8_names.count("validate_before_rename") == 1
    assert v8_names.count("rename_owned") == 1
    assert v8_names.count("validate_after_rename") == 1
    assert v8_names.count("require_final_quiet") == 1
    assert "_libc_renameat2" not in v8_names
    transaction_index = v8_names.index("_ClosedPublicationTransaction")
    revalidate_index = v8_names.index("_revalidate_exact_before_publication")
    validate_index = v8_names.index("validate_before_rename")
    rename_index = v8_names.index("rename_owned")
    post_index = v8_names.index("validate_after_rename")
    quiet_index = v8_names.index("require_final_quiet")
    assert transaction_index < revalidate_index < validate_index < rename_index
    assert rename_index < post_index < quiet_index

    required_before_second_pass = {
        "_precommitted_audit_sample",
        "_with_content_sha256",
        "_validate_staging_inventory",
        "_write_json_exclusive",
        "_fsync_directory",
    }
    for name in required_before_second_pass:
        prepublication_occurrences = [
            line
            for line, observed_name in v8_calls
            if observed_name == name
        ]
        assert prepublication_occurrences
        transaction_line = next(
            line
            for line, observed_name in v8_calls
            if observed_name == "_ClosedPublicationTransaction"
        )
        assert max(prepublication_occurrences) < transaction_line
    assert any(
        line < transaction_line
        for line, observed_name in v8_calls
        if observed_name == "fsync"
    )


def test_v6_ancestor_gap_is_reproduced_and_v8_rejects_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v6_root = tmp_path / "v6" / "canonical"
    v6_root.mkdir(parents=True)
    v6_transaction, v6_retained, v6_paths = _open_v6_synthetic_transaction(
        v6_root, monkeypatch
    )
    v6_moved = tmp_path / "v6-moved"
    try:
        v6_transaction.validate_before_rename()
        v6_transaction.rename_owned()
        v6_retained.refresh_after_owned_mutation()
        v6_transaction.validate_after_rename()
        os.fsync(v6_retained.parent_fd)
        v6_root.rename(v6_moved)
        v6_root.mkdir()
        (v6_root / "publication").mkdir()
        v6_transaction.require_final_quiet()
        assert not v6_paths["destination"].exists()
        assert (v6_moved / "publication" / "dataset").is_dir()
    finally:
        v6_transaction.close()
        v6_retained.close()

    v8_root = tmp_path / "v8" / "canonical"
    v8_root.mkdir(parents=True)
    transaction, retained, paths = _open_synthetic_transaction(v8_root, monkeypatch)
    moved = tmp_path / "v8-moved"
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.validate_after_rename()
        os.fsync(retained.parent_fd)
        v8_root.rename(moved)
        v8_root.mkdir()
        (v8_root / "publication").mkdir()
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.require_final_quiet()
        assert not paths["destination"].exists()
        assert (moved / "publication" / "dataset").is_dir()
    finally:
        transaction.close()
        retained.close()


def test_canonical_v8_roles_schemas_and_authors_are_exact() -> None:
    assert builder.AUTHORIZED_ROLE_PATHS == (
        (
            "builder_source",
            "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v8.py",
        ),
        (
            "builder_cli",
            "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py",
        ),
        (
            "builder_test",
            "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v8.py",
        ),
        (
            "builder_handoff",
            "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_"
            "author_handoff_2026-07-13.md",
        ),
        (
            "builder_review",
            "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v8_"
            "independent_review_2026-07-13.json",
        ),
        (
            "auditor_source",
            "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v8.py",
        ),
        (
            "auditor_cli",
            "scripts/audit_go2_shared_jepa_v5_raw_supervision_v8.py",
        ),
        (
            "auditor_test",
            "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v8.py",
        ),
        (
            "auditor_review",
            "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v8_"
            "independent_review_2026-07-13.json",
        ),
    )
    assert builder.AUTHORIZATION_SCHEMA.endswith("build_authorization_v8")
    assert builder.REVIEW_BINDING_SCHEMA.endswith("review_binding_v8")
    assert builder.BUILDER_REVIEW_SCHEMA.endswith(
        "builder_v8_independent_review_v1"
    )
    assert builder.AUDITOR_REVIEW_SCHEMA.endswith(
        "auditor_v8_independent_review_v1"
    )
    assert (
        builder.BUILDER_IMPLEMENTATION_AUTHOR
        == "/root/raw_v7_successor_author/auditor_v7_author"
    )
    assert (
        builder.AUDITOR_IMPLEMENTATION_AUTHOR
        == "/root/camera_v5_independent/camera_v7_pre_freeze_review/"
        "v7_review_artifact_schema"
    )
    assert builder._expected_review_authority("builder")["retry_authorized"] is False
    assert builder._expected_review_authority("auditor")["retry_authorized"] is False


def test_v8_cli_imports_only_v8_exact_entry() -> None:
    cli = (
        ROOT
        / "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v8.py"
    ).read_text(encoding="utf-8")
    assert "go2_shared_jepa_v5_raw_supervision_builder_v8" in cli
    assert "execute_exact_build_v8" in cli
    assert "execute_exact_build_v7" not in cli
    assert "go2_shared_jepa_v5_raw_supervision_builder_v7 import" not in cli


def test_import_surface_exposes_no_legacy_builder_or_nonpure_loader() -> None:
    namespace = vars(builder)
    forbidden_names = {
        "execute_exact_build_v1",
        "execute_exact_build_v2",
        "execute_exact_build_v3",
        "execute_exact_build_v4",
        "execute_exact_build_v7",
        "load_frozen_development_metadata",
        "load_frozen_development_source_inventory",
        "plan_v5",
        "v4_builder",
        "_v1",
        "_v2",
        "_v3",
        "_v4",
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
            or "raw_supervision_builder_v4" in value.__name__
            or "raw_supervision_builder_v7" in value.__name__
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
        "PreparedEndpointV8",
        "PreparedSceneJobV8",
        "RawSupervisionBuildError",
        "THREAD_ENVIRONMENT",
        "canonical_json_bytes",
        "canonical_json_sha256",
        "execute_exact_build_v8",
    }


def test_production_signatures_have_no_injection_seams() -> None:
    phase_two = inspect.signature(builder._validate_authorization_phase_two)
    assert tuple(phase_two.parameters) == ("phase_one",)
    exact = inspect.signature(builder.execute_exact_build_v8)
    assert tuple(exact.parameters) == ("authorization_sha256", "workers")
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for value in exact.parameters.values()
    )
    build = inspect.signature(builder._build_exact_prepared_dataset_v8)
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
    assert tuple(field.name for field in fields(builder.AcceptedAuthorizationV8)) == (
        "authorization_file_sha256",
        "authorization_content_sha256",
        "source_map_sha256",
    )
    receipt = builder.AcceptedAuthorizationV8("1" * 64, "2" * 64, "3" * 64)
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
        or "raw_supervision_builder_v4" in module
        or "raw_supervision_builder_v7" in module
        or "raw_supervision_auditor" in module
        for module in imported
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"__import__", "eval", "exec"}
        for node in ast.walk(tree)
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
        lambda value: value["builder_review"].__setitem__("reviewer", "/root"),
        lambda value: value["auditor_review"].__setitem__(
            "reviewer", builder.BUILDER_IMPLEMENTATION_AUTHOR
        ),
    ],
)
def test_phase_one_adversaries_reach_zero_openers(
    mutation: Mutation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, _raw_by_role, _digests = valid_authorization()
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
    authority, raw_by_role, digests = valid_authorization()
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

    receipt, opened = validate_phase_two_for_tests(
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
    authority, _raw_by_role, _digests = valid_authorization()
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
        builder.execute_exact_build_v8(
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
        builder.execute_exact_build_v8(
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
    first_result = write_prepared_scene_job_for_tests(job, first)
    reversed_job = builder.PreparedSceneJobV8(
        scene_id=job.scene_id,
        role=job.role,
        family=job.family,
        endpoints=tuple(reversed(job.endpoints)),
    )
    second_result = write_prepared_scene_job_for_tests(
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
    duplicate = builder.PreparedSceneJobV8(
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


@pytest.mark.parametrize("mutation_scope", ["staging", "source"])
def test_full_build_rejects_mutation_during_final_source_pass(
    mutation_scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication" / "dataset"
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    source = source_parent / "source.bin"
    source.write_bytes(b"frozen source\n")
    authorization_sha256 = "a" * 64
    job, pair = _synthetic_job_and_pair()
    context = builder.ExactPrepublicationContextV8(
        plan=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        source_records=(),
        authorization_sha256=authorization_sha256,
        workers=1,
    )
    observed_staging: list[Path] = []

    monkeypatch.setattr(builder, "CANONICAL_OUTPUT", output)
    monkeypatch.setattr(builder, "_require_exact_authority", lambda _digest: None)
    monkeypatch.setattr(builder, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(
        builder,
        "_precommitted_audit_sample",
        lambda _rows: {"records": [{} for _ in range(24)]},
    )
    monkeypatch.setattr(
        builder,
        "_exact_publication_source_hashes",
        lambda _context: {source: _sha(source)},
    )

    def mutate_during_source_pass(_context: Any) -> None:
        candidates = [
            path
            for path in output.parent.iterdir()
            if path.name.startswith(f".{output.name}.staging.")
        ]
        assert len(candidates) == 1
        observed_staging.append(candidates[0])
        target = candidates[0] / "pairs.jsonl" if mutation_scope == "staging" else source
        target.write_bytes(b"mutated during final source pass\n")

    monkeypatch.setattr(
        builder,
        "_revalidate_exact_before_publication",
        mutate_during_source_pass,
    )
    with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
        builder._build_exact_prepared_dataset_v8(
            (job,),
            (pair,),
            workers=1,
            input_provenance={},
            access_ledger={},
            prepublication_context=context,
        )
    assert observed_staging
    assert not output.exists()
    assert not any(output.parent.glob(f".{output.name}.staging.*"))


def test_closed_transaction_clean_publication_has_exact_owned_rename_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        _publish_synthetic_transaction(transaction, retained)
        assert paths["destination"].is_dir()
        assert not transaction.poisoned
        assert transaction.renamed
    finally:
        transaction.close()
        retained.close()


def test_transaction_watches_complete_retained_publication_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "a" / "b"
    root.mkdir(parents=True)
    transaction, retained, _paths = _open_synthetic_transaction(root, monkeypatch)
    try:
        current = retained.anchor_path
        expected = [current]
        for component in retained.path.parts[1:]:
            current = current / component
            expected.append(current)
        assert expected[-1] == retained.path
        assert all(path in transaction._watch_by_path for path in expected)
        assert all(
            "publication_ancestor" in transaction._watch_by_path[path].roles
            for path in expected
        )
        assert not (
            builder._IN_ANCESTOR_MASK
            & (
                builder._IN_CREATE
                | builder._IN_DELETE
                | builder._IN_MOVED_FROM
                | builder._IN_MOVED_TO
                | builder._IN_MODIFY
                | builder._IN_CLOSE_WRITE
            )
        )
    finally:
        transaction.close()
        retained.close()


def test_ancestry_only_named_child_churn_is_not_a_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    transaction, retained, paths = _open_synthetic_transaction(root, monkeypatch)
    unrelated = tmp_path / "unrelated-child"
    try:
        unrelated.mkdir()
        (unrelated / "leaf").write_bytes(b"unrelated\n")
        (unrelated / "leaf").unlink()
        unrelated.rmdir()
        _publish_synthetic_transaction(transaction, retained)
        assert paths["destination"].is_dir()
        assert not transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_ancestry_event_filter_keeps_merged_strict_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, _paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        ancestry_only = next(
            binding
            for binding in transaction._watch_by_path.values()
            if binding.roles == frozenset({"publication_ancestor"})
        )
        publication = transaction._watch_by_path[retained.path]
        named_ancestor = builder._InotifyEvent(
            ancestry_only.descriptor, builder._IN_ATTRIB, 0, "unrelated"
        )
        named_publication = builder._InotifyEvent(
            publication.descriptor, builder._IN_ATTRIB, 0, "strict-child"
        )
        assert transaction._strict_events([named_ancestor]) == []
        assert transaction._strict_events([named_publication]) == [named_publication]
        for mask in (
            builder._IN_ATTRIB,
            builder._IN_DELETE_SELF,
            builder._IN_MOVE_SELF,
            builder._IN_UNMOUNT,
        ):
            self_event = builder._InotifyEvent(
                ancestry_only.descriptor, mask, 0, ""
            )
            assert transaction._strict_events([self_event]) == [self_event]
    finally:
        transaction.close()
        retained.close()


def test_ancestry_attribute_mutate_then_restore_is_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    transaction, retained, paths = _open_synthetic_transaction(
        canonical, monkeypatch
    )
    original_mode = canonical.stat().st_mode & 0o777
    try:
        canonical.chmod(0o711)
        canonical.chmod(original_mode)
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.validate_before_rename()
        assert transaction.poisoned
        assert not paths["destination"].exists()
    finally:
        canonical.chmod(original_mode)
        transaction.close()
        retained.close()


def test_terminal_close_rehashes_all_source_and_published_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, _paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.validate_after_rename()
        os.fsync(retained.parent_fd)
        expected = {leaf.descriptor for leaf in transaction._leaves}
        observed: set[int] = set()
        original = builder._sha256_fd

        def recording(descriptor: int) -> str:
            observed.add(descriptor)
            return original(descriptor)

        monkeypatch.setattr(builder, "_sha256_fd", recording)
        transaction.require_final_quiet()
        assert observed == expected
    finally:
        transaction.close()
        retained.close()


def test_terminal_close_rejects_mutation_during_inventory_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.validate_after_rename()
        os.fsync(retained.parent_fd)
        original = builder._sha256_fd
        mutated = False

        def mutate_once(descriptor: int) -> str:
            nonlocal mutated
            if not mutated:
                mutated = True
                paths["source"].write_bytes(b"terminal mutation\n")
            return original(descriptor)

        monkeypatch.setattr(builder, "_sha256_fd", mutate_once)
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.require_final_quiet()
        assert mutated
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_terminal_second_boundary_rejects_ancestor_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    transaction, retained, _paths = _open_synthetic_transaction(
        canonical, monkeypatch
    )
    moved = tmp_path / "moved-after-terminal-inventory"
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.validate_after_rename()
        os.fsync(retained.parent_fd)
        original = transaction._require_no_events

        def move_after_inventory(phase: str) -> None:
            original(phase)
            if phase == "terminal source and published inventory":
                canonical.rename(moved)
                canonical.mkdir()
                (canonical / "publication").mkdir()

        monkeypatch.setattr(transaction, "_require_no_events", move_after_inventory)
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.require_final_quiet()
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_transaction_baseline_rejects_extant_hash_change(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected\n")
    expected = _sha(source)
    source.write_bytes(b"changed before baseline\n")
    with pytest.raises(builder.RawSupervisionBuildError, match="baseline"):
        builder._open_transaction_leaf(
            source,
            expected_sha256=expected,
            namespace="source",
            relative_path=None,
        )


@pytest.mark.parametrize("namespace", ["source", "staging"])
@pytest.mark.parametrize(
    "operation",
    ["modify_restore", "create_delete", "rename_restore", "replace_restore"],
)
def test_closed_transaction_rejects_namespace_mutation_and_restoration(
    namespace: str,
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    target = paths["source"] if namespace == "source" else paths["data"]
    parent = paths["source_parent"] if namespace == "source" else paths["staging"]
    original = target.read_bytes()
    try:
        if operation == "modify_restore":
            target.write_bytes(b"transient mutation\n")
            target.write_bytes(original)
        elif operation == "create_delete":
            ephemeral = parent / "ephemeral.bin"
            ephemeral.write_bytes(b"ephemeral\n")
            ephemeral.unlink()
        elif operation == "rename_restore":
            moved = parent / "moved.bin"
            target.rename(moved)
            moved.rename(target)
        elif operation == "replace_restore":
            saved = parent / "saved.bin"
            target.rename(saved)
            target.write_bytes(b"replacement\n")
            target.unlink()
            saved.rename(target)
        else:
            raise AssertionError(operation)
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.validate_before_rename()
        assert transaction.poisoned
        assert not paths["destination"].exists()
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_rejects_mutation_during_descriptor_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    original_sha256_fd = builder._sha256_fd
    mutated = False

    def mutating_sha256_fd(descriptor: int) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            paths["source"].write_bytes(b"mutated during validation\n")
        return original_sha256_fd(descriptor)

    monkeypatch.setattr(builder, "_sha256_fd", mutating_sha256_fd)
    try:
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.validate_before_rename()
        assert mutated
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_rejects_mutation_after_validation_at_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        transaction.validate_before_rename()
        paths["source"].write_bytes(b"mutated after validation\n")
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.rename_owned()
        assert transaction.renamed
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_rejects_mutation_inside_rename_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    original_rename = builder._libc_renameat2

    def racing_rename(*args: Any) -> None:
        paths["source"].write_bytes(b"rename-time mutation\n")
        original_rename(*args)

    monkeypatch.setattr(builder, "_libc_renameat2", racing_rename)
    try:
        transaction.validate_before_rename()
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.rename_owned()
        assert transaction.renamed
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_rejects_post_rename_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        paths["source"].write_bytes(b"post-rename mutation\n")
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.validate_after_rename()
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_destination_race_is_poisoned_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    try:
        transaction.validate_before_rename()
        paths["destination"].mkdir()
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.rename_owned()
        assert transaction.poisoned
        assert not transaction.renamed
        assert paths["destination"].is_dir()
        assert paths["staging"].is_dir()
    finally:
        transaction.close()
        retained.close()


def test_post_rename_cleanup_removes_only_proven_owned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    identity = builder._named_directory_identity(
        retained.parent_fd, paths["staging"].name
    )
    replacement = paths["destination"]
    moved_owned = replacement.with_name("moved-owned")
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        replacement.rename(moved_owned)
        replacement.mkdir()
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction.validate_after_rename()
        transaction.close()
        assert not builder._cleanup_owned_directory(
            retained, replacement.name, identity
        )
        assert replacement.is_dir()
        assert moved_owned.is_dir()
    finally:
        transaction.close()
        retained.close()


def test_post_rename_cleanup_can_remove_exact_owned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    identity = builder._named_directory_identity(
        retained.parent_fd, paths["staging"].name
    )
    try:
        transaction.validate_before_rename()
        transaction.rename_owned()
        retained.refresh_after_owned_mutation()
        transaction.close()
        assert builder._cleanup_owned_directory(
            retained, paths["destination"].name, identity
        )
        assert not paths["destination"].exists()
    finally:
        transaction.close()
        retained.close()


@pytest.mark.parametrize(
    "fault",
    ["overflow", "ignored", "unmount", "unknown_watch", "unknown_mask", "malformed"],
)
def test_closed_transaction_poison_events_are_fail_closed(
    fault: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, _paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    valid_watch = next(iter(transaction._watch_by_descriptor))
    if fault == "overflow":
        payload = builder._INOTIFY_HEADER.pack(-1, builder._IN_Q_OVERFLOW, 0, 0)
    elif fault == "ignored":
        payload = builder._INOTIFY_HEADER.pack(valid_watch, builder._IN_IGNORED, 0, 0)
    elif fault == "unmount":
        payload = builder._INOTIFY_HEADER.pack(valid_watch, builder._IN_UNMOUNT, 0, 0)
    elif fault == "unknown_watch":
        payload = builder._INOTIFY_HEADER.pack(2**30, builder._IN_MODIFY, 0, 0)
    elif fault == "unknown_mask":
        payload = builder._INOTIFY_HEADER.pack(valid_watch, 0x00100000, 0, 0)
    elif fault == "malformed":
        payload = builder._INOTIFY_HEADER.pack(valid_watch, builder._IN_MODIFY, 0, 8) + b"x"
    else:
        raise AssertionError(fault)

    class FakePoll:
        def __init__(self) -> None:
            self.calls = 0

        def register(self, *_args: Any) -> None:
            return None

        def poll(self, _milliseconds: int) -> list[tuple[int, int]]:
            self.calls += 1
            return (
                [(transaction._inotify_fd, builder.select.POLLIN)]
                if self.calls == 1
                else []
            )

    monkeypatch.setattr(builder.select, "poll", FakePoll)
    monkeypatch.setattr(builder.os, "read", lambda _fd, _count: payload)
    try:
        with pytest.raises(builder.RawSupervisionBuildError, match="poisoned"):
            transaction._read_events()
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_closed_transaction_rejects_watch_descriptor_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction, retained, _paths = _open_synthetic_transaction(tmp_path, monkeypatch)
    reused = next(iter(transaction._watch_by_descriptor))
    monkeypatch.setattr(builder, "_inotify_add", lambda *_args, **_kwargs: reused)
    try:
        with pytest.raises(builder.RawSupervisionBuildError, match="reused"):
            transaction._add_watch(
                tmp_path / "distinct-watch-path",
                is_directory=False,
                role="synthetic",
            )
        assert transaction.poisoned
    finally:
        transaction.close()
        retained.close()


def test_v2_block_reproducers_are_absent() -> None:
    assert "_run_authorized_scene_pool" not in vars(builder)
    assert "_call_v1_load_parent_contracts" not in vars(builder)
    assert "_call_v1_load_exact_scene_job" not in vars(builder)
    assert "_call_v1_revalidate_exact_scene_sources" not in vars(builder)
    assert tuple(inspect.signature(builder._validate_authorization_phase_two).parameters) == (
        "phase_one",
    )
    assert PRODUCTION_ELIGIBLE is False
    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "from lewm.tests" not in source
    assert "_require_exact_authority =" not in source
