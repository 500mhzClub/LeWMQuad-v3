"""Independent adversarial review of the frozen raw-supervision auditor V1.

Every dynamic artifact is synthetic and rooted below pytest ``tmp_path``.  The
canonical dataset, development source inventory, RGB, G2, and accelerator
payloads are never opened by this suite.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_auditor_v1 as auditor
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as builder
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as plan_v5
from lewm.tests import test_go2_shared_jepa_v5_raw_supervision_auditor_v1 as author_tests
from scripts import audit_go2_shared_jepa_v5_raw_supervision_v1 as audit_cli


ROOT = Path(__file__).resolve().parents[2]
REVIEWER = "/root/raw_auditor_v1_independent"
FROZEN_CANDIDATE = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v1.py": (
        "854d433084af4bda7dca1e39bed69bc76e9904546111e9289cbb4066660c798c"
    ),
    "scripts/audit_go2_shared_jepa_v5_raw_supervision_v1.py": (
        "246a8de16a9645a0af8f0cf69e6241b16d68588d54ee9f8eb8b087519a9b908d"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v1.py": (
        "6dfe991e3f5abc7a5a7405ad1a9ad74382d05ba27e1beb5e6d087aed41351557"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v1_author_handoff_2026-07-13.md": (
        "7d693902bf4517bb19a87b6769af0c272403ba553daccb6e03d9cef88eec279d"
    ),
}
FROZEN_BUILDER = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "3bc1559776e2f8471bb6a7a1ddd8808b1f1224687dedf280fd2300820afe25ec"
    ),
    "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v1.py": (
        "df5fd60b50ba852d44fd6fe0034c7e763fc08030875488be3850e774906ceeb3"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v1.py": (
        "15767446ba45851a7f5774560db8e8f6f87d831a51fde7585acffa028f3ba2e4"
    ),
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v1_author_handoff_2026-07-13.md": (
        "9d9aee5f636069d8beef2362bcc43b9be0063207d9ffe17d9045f99e3c30d28c"
    ),
}
FROZEN_METADATA_V5 = {
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py": (
        "8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_author_handoff_2026-07-13.md": (
        "b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_independent_review_2026-07-13.md": (
        "7d7344e423492a3cf36d1cd50ca09e6c7eb6eba17c25861c840531465aaf7706"
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(auditor.canonical_json_bytes(core))
    return {**normalized, "content_sha256": auditor.canonical_json_sha256(normalized)}


def _write_manifest(root: Path, manifest: Mapping[str, Any]) -> str:
    core = dict(manifest)
    core.pop("content_sha256", None)
    value = {**core, "content_sha256": auditor.canonical_json_sha256(core)}
    raw = auditor.canonical_json_bytes(value) + b"\n"
    (root / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rebind_root_file(manifest: dict[str, Any], root: Path, relative: str) -> None:
    raw = (root / relative).read_bytes()
    record = next(item for item in manifest["files"] if item["path"] == relative)
    record["byte_count"] = len(raw)
    record["file_sha256"] = hashlib.sha256(raw).hexdigest()


def _one_endpoint_fixture(tmp_path: Path) -> tuple[
    Path,
    str,
    auditor.AuditInputs,
    dict[str, tuple[np.ndarray, ...]],
]:
    return author_tests._synthetic_fixture(tmp_path)


def _make_plan_endpoint(
    *, role: str, family: str, scene_id: str, index: int
) -> dict[str, Any]:
    image_sha = hashlib.sha256(f"independent-image:{index}".encode("ascii")).hexdigest()
    identity = {
        "dataset_role": role,
        "scene_id": scene_id,
        "episode_id": f"episode_{index:03d}",
        "env_index": index,
        "episode_step": 0,
        "frame_index": index,
        "timestamp_ns": 10_000 + index,
        "image_sha256": image_sha,
    }
    return _with_hash(
        {
            "schema": plan_v5.ENDPOINT_SCHEMA,
            "identity": identity,
            "identity_sha256": auditor.canonical_json_sha256(identity),
            "image_path_metadata_only": f"/synthetic/{role}/{family}/{index}.png",
            "frames_jsonl_sha256": hashlib.sha256(b"frames").hexdigest(),
            "scene_manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        }
    )


def _make_pair(
    endpoint: Mapping[str, Any], *, family: str, index: int
) -> dict[str, Any]:
    identity = endpoint["identity"]
    digest = endpoint["identity_sha256"]
    return _with_hash(
        {
            "schema": plan_v5.PAIR_SCHEMA,
            "dataset_role": identity["dataset_role"],
            "global_row": index,
            "scene_id": identity["scene_id"],
            "family": family,
            "episode_id": identity["episode_id"],
            "env_index": identity["env_index"],
            "reset_count": 0,
            "source_split": "independent_synthetic",
            "frames_jsonl_sha256": endpoint["frames_jsonl_sha256"],
            "scene_manifest_sha256": endpoint["scene_manifest_sha256"],
            "primitive": "hold",
            "relative_se2_current_frame": [0.0, 0.0, 0.0],
            "current_endpoint_sha256": digest,
            "next_endpoint_sha256": digest,
            "label_shard_path_metadata_only": f"/synthetic/labels/{index}.npz",
            "label_shard_sha256": hashlib.sha256(f"labels:{index}".encode()).hexdigest(),
            "label_shard_row": 0,
            "sidecar_row_identity_sha256": hashlib.sha256(
                f"sidecar:{index}".encode()
            ).hexdigest(),
        }
    )


def _caller_injected_exact_fixture(tmp_path: Path) -> tuple[
    Path,
    str,
    auditor.AuditInputs,
    dict[str, tuple[np.ndarray, ...]],
]:
    """Build only 24 endpoints but forge exact declarations on the public path."""

    base_frame = auditor.raycast_v4.synthetic_scene_jobs(1)[0].frames[0]
    evidence = auditor.raycast_v4.build_frame_evidence_v4(base_frame)
    raster = auditor.raycast_v4.rasterize_observable_camera_ray_evidence_v4(evidence)
    arrays = auditor._stored_arrays_from_evidence(evidence, raster)
    jobs: list[builder.PreparedSceneJobV1] = []
    pairs: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    replay: dict[str, tuple[np.ndarray, ...]] = {}
    index = 0
    for role in plan_v5.DEVELOPMENT_ROLES:
        for family_index in range(8):
            family = f"family_{family_index:02d}"
            scene_id = f"scene_{index:03d}"
            endpoint = _make_plan_endpoint(
                role=role,
                family=family,
                scene_id=scene_id,
                index=index,
            )
            pair = _make_pair(endpoint, family=family, index=index)
            jobs.append(
                builder.PreparedSceneJobV1(
                    scene_id=scene_id,
                    role=role,
                    family=family,
                    endpoints=(
                        builder.PreparedEndpointV1(
                            plan_endpoint=endpoint,
                            family=family,
                            frame=base_frame,
                        ),
                    ),
                )
            )
            endpoints.append(endpoint)
            pairs.append(pair)
            replay[str(endpoint["identity_sha256"])] = arrays
            index += 1
    root = tmp_path / "caller_injected_exact"
    builder.build_prepared_dataset_v1(
        jobs,
        pairs,
        output_directory=root,
        workers=1,
        input_provenance={"fixture": "caller_injected_exact"},
        access_ledger={"rgb_byte_opens": 0, "g2_payload_opens": 0},
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    manifest["pair_counts"] = {
        "train": 4262,
        "checkpoint_selection": 495,
        "probability_calibration": 415,
    }
    manifest["unique_endpoint_counts"] = {
        "train": 7777,
        "checkpoint_selection": 924,
        "probability_calibration": 759,
    }
    manifest["endpoint_instance_count"] = 10344
    manifest["scene_shard_count"] = 88
    digest = _write_manifest(root, manifest)
    plan = plan_v5.DevelopmentRawSupervisionPlan(
        value={},
        pairs=tuple(sorted(pairs, key=lambda item: item["global_row"])),
        endpoints=tuple(endpoints),
    )
    inventory = plan_v5.DevelopmentSourceInventory(
        records=(), hashes={}, access_ledger={}
    )
    return root, digest, auditor.AuditInputs(plan=plan, inventory=inventory), replay


def test_independent_frozen_source_closure() -> None:
    expected = {
        **FROZEN_CANDIDATE,
        **FROZEN_BUILDER,
        **FROZEN_METADATA_V5,
        **auditor.REVIEWED_V4_SOURCE_SHA256,
    }
    assert {relative: _sha(ROOT / relative) for relative in expected} == expected


def test_independent_literal_builder_contract_equality() -> None:
    assert tuple(
        (name, np.dtype(dtype).str, shape)
        for name, dtype, shape in auditor.ARRAY_LAYOUT
    ) == tuple(
        (name, np.dtype(dtype).str, shape)
        for name, dtype, shape in builder.ARRAY_LAYOUT
    )
    assert auditor.DATASET_SCHEMA == builder.DATASET_SCHEMA
    assert auditor.SHARD_SCHEMA == builder.SHARD_SCHEMA
    assert auditor.ENDPOINT_INDEX_SCHEMA == builder.ENDPOINT_INDEX_SCHEMA
    assert auditor.MAX_WORKERS == builder.MAX_WORKERS == 6
    assert auditor.FROZEN_PARENT_FILE_SHA256 == builder.FROZEN_PARENT_HASHES
    assert auditor.REVIEWED_V4_SOURCE_SHA256 == builder.REVIEWED_V4_SOURCES
    assert auditor.CANONICAL_DATASET == builder.CANONICAL_OUTPUT


def test_independent_sealed_entry_signature_and_cli_paths() -> None:
    signature = inspect.signature(auditor.audit_exact_dataset_v1)
    assert "sample_recomputer" not in signature.parameters
    parsed = audit_cli._parse_args(["--manifest-sha256", "a" * 64, "--workers", "6"])
    assert parsed.workers == 6
    for option in ("--dataset", "--output", "--report", "--repo-root"):
        with pytest.raises(SystemExit):
            audit_cli._parse_args(["--manifest-sha256", "a" * 64, option, "/tmp/x"])


def test_independent_authorization_precedes_metadata_and_source_openers() -> None:
    tree = ast.parse(inspect.getsource(auditor.audit_exact_dataset_v1))
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    ]
    preflight = calls.index("_preflight_exact_authorization")
    assert preflight < calls.index("load_frozen_development_metadata")
    assert preflight < calls.index("load_frozen_development_source_inventory")
    assert preflight < calls.index("_hash_complete_source_inventory")


def test_independent_worker_limits_and_visibility_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (*auditor.THREAD_ENVIRONMENT, *auditor.ACCELERATOR_ENVIRONMENT):
        monkeypatch.setenv(name, "adversarial")
    auditor._set_worker_environment()
    assert {name: os.environ[name] for name in auditor.THREAD_ENVIRONMENT} == {
        name: "1" for name in auditor.THREAD_ENVIRONMENT
    }
    assert {name: os.environ[name] for name in auditor.ACCELERATOR_ENVIRONMENT} == {
        name: "" for name in auditor.ACCELERATOR_ENVIRONMENT
    }
    for workers in (False, 0, 7):
        with pytest.raises(ValueError, match="workers"):
            auditor.audit_exact_dataset_v1(
                auditor.ROOT,
                auditor.CANONICAL_DATASET,
                expected_manifest_file_sha256="a" * 64,
                workers=workers,
            )


def test_independent_source_inventory_expands_only_88_times_four() -> None:
    records = []
    for index in range(88):
        records.append(
            {
                "scene_id": f"scene_{index:03d}",
                "frames": {"path": f"/repo/frames/{index}.jsonl", "sha256": "a" * 64},
                "scene_manifest": {
                    "path": f"/repo/manifests/{index}.json",
                    "file_sha256": "b" * 64,
                },
                "render_plan": {"path": f"/repo/plans/{index}.json", "sha256": "c" * 64},
                "render_summary": {
                    "path": f"/repo/summaries/{index}.json",
                    "sha256": "d" * 64,
                },
            }
        )
    inventory = plan_v5.DevelopmentSourceInventory(
        records=tuple(records), hashes={}, access_ledger={}
    )
    expanded = auditor._source_file_records(inventory)
    assert len(expanded) == 352
    assert Counter(item["kind"] for item in expanded) == Counter(
        {"frames": 88, "scene_manifest": 88, "render_plan": 88, "render_summary": 88}
    )
    assert {item["scene_id"] for item in expanded} == {
        f"scene_{index:03d}" for index in range(88)
    }


def test_independent_reconstructs_the_24_endpoint_precommit() -> None:
    rows = []
    expected = []
    for role in plan_v5.DEVELOPMENT_ROLES:
        for family_index in range(8):
            family = f"family_{family_index:02d}"
            candidates = []
            for suffix in range(3):
                digest = hashlib.sha256(
                    f"{role}:{family}:{suffix}".encode("ascii")
                ).hexdigest()
                rows.append(
                    {
                        "dataset_role": role,
                        "family": family,
                        "endpoint_identity_sha256": digest,
                    }
                )
                score = hashlib.sha256(
                    role.encode() + b"\0" + family.encode() + b"\0" + digest.encode()
                ).hexdigest()
                candidates.append((score, digest))
            score, digest = min(candidates)
            expected.append(
                {
                    "dataset_role": role,
                    "family": family,
                    "endpoint_identity_sha256": digest,
                    "selection_sha256": score,
                }
            )
    assert auditor._sample_records(rows) == sorted(
        expected, key=lambda item: (item["dataset_role"], item["family"])
    )
    assert len(expected) == auditor.EXPECTED_SAMPLE_COUNT == 24


def test_independent_one_and_six_workers_are_artifact_compatible(tmp_path: Path) -> None:
    root, digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    one = auditor.audit_dataset_v1(
        root,
        expected_manifest_file_sha256=digest,
        inputs=inputs,
        sample_recomputer=lambda *_args: replay,
        workers=1,
    )
    six = auditor.audit_dataset_v1(
        root,
        expected_manifest_file_sha256=digest,
        inputs=inputs,
        sample_recomputer=lambda *_args: replay,
        workers=6,
    )
    assert one == six


def test_independent_exact_replay_contains_frame_camera_rpy_raycast_raster_chain() -> None:
    replay_source = inspect.getsource(auditor._recompute_one_exact_sample)
    render_source = inspect.getsource(auditor._validate_sample_render_contract)
    for name in (
        "_find_source_frame",
        "_validated_sidecar_source_attitude",
        "parse_scene_manifest_dict",
        "compose_yaw_aligned_camera",
        "_normalized_camera_basis_fru",
        "_box_in_yaw_body",
        "build_frame_evidence_v4",
        "rasterize_observable_camera_ray_evidence_v4",
    ):
        assert name in replay_source
    assert "full_box_roll_pitch_yaw_rendered" in render_source
    assert "_rendered_boxes" in render_source
    assert "_render_object_records" in render_source


def test_independent_source_ancestor_alias_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    real = repository / "real"
    real.mkdir(parents=True)
    payload = b"allowlisted\n"
    source = real / "source.json"
    source.write_bytes(payload)
    alias = repository / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="alias"):
        auditor._read_absolute_bound_payload(
            alias / source.name,
            hashlib.sha256(payload).hexdigest(),
            repository_root=repository,
            name="independent aliased source",
        )


def test_independent_publication_ancestor_alias_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    parent = real / "publication"
    parent.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(PermissionError, match="alias"):
        auditor._ExclusiveAuditPublisher(alias / "publication")


def test_independent_publication_ancestor_swap_cleans_owned_temporary(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "root" / "ancestor"
    parent = ancestor / "nested" / "publication"
    parent.mkdir(parents=True)
    publisher = auditor._ExclusiveAuditPublisher(parent)
    moved = tmp_path / "moved_ancestor"
    try:
        ancestor.rename(moved)
        ancestor.symlink_to(moved, target_is_directory=True)
        with pytest.raises(auditor.RawSupervisionAuditError, match="directory chain"):
            publisher.publish("result.json", {"status": "PASS"})
        retained_parent = moved / "nested" / "publication"
        assert not (retained_parent / "result.json").exists()
        assert list(retained_parent.iterdir()) == []
    finally:
        publisher.__exit__()


def test_independent_true_noreplace_preserves_late_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    destination = parent / "result.json"
    original = auditor._rename_noreplace_at

    def install_late_destination(parent_fd: int, source: str, name: str) -> None:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        os.write(descriptor, b"foreign\n")
        os.close(descriptor)
        original(parent_fd, source, name)

    monkeypatch.setattr(auditor, "_rename_noreplace_at", install_late_destination)
    with auditor._ExclusiveAuditPublisher(parent) as publisher:
        with pytest.raises(FileExistsError):
            publisher.publish(destination.name, {"status": "PASS"})
    assert destination.read_bytes() == b"foreign\n"
    assert sorted(item.name for item in parent.iterdir()) == [destination.name]


def test_independent_exact_failure_is_terminal_and_retry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "publication"
    parent.mkdir()
    dataset = parent / "dataset"
    report = parent / "dataset.audit.json"
    failure = parent / "dataset.audit.failed.json"
    monkeypatch.setattr(auditor, "CANONICAL_DATASET", dataset)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_REPORT", report)
    monkeypatch.setattr(auditor, "CANONICAL_AUDIT_FAILURE", failure)

    def fail_exact(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise auditor.RawSupervisionAuditError("independent injected failure")

    monkeypatch.setattr(auditor, "audit_exact_dataset_v1", fail_exact)
    with pytest.raises(auditor.RawSupervisionAuditError, match="injected"):
        auditor.execute_exact_audit_v1(
            expected_manifest_file_sha256="a" * 64,
            workers=1,
        )
    value = json.loads(failure.read_text(encoding="ascii"))
    assert value["status"] == "terminal_failed_no_dataset_authority"
    assert value["retry_authorized"] is False
    assert not report.exists()
    with pytest.raises(FileExistsError, match="immutable audit leaf"):
        auditor.execute_exact_audit_v1(
            expected_manifest_file_sha256="a" * 64,
            workers=1,
        )


@pytest.mark.parametrize("array_name", [item[0] for item in auditor.ARRAY_LAYOUT])
def test_independent_every_committed_array_class_rejects_byte_mutation(
    tmp_path: Path, array_name: str
) -> None:
    root, digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    path = next((root / "shards").glob(f"*/{array_name}"))
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(auditor.RawSupervisionAuditError, match="bytes changed"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


@pytest.mark.parametrize("boundary", ("pair", "endpoint", "shard_index"))
def test_independent_rebound_join_mutations_reach_semantic_rejection(
    tmp_path: Path, boundary: str
) -> None:
    root, _digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    if boundary == "pair":
        path = root / "pairs.jsonl"
        row = json.loads(path.read_text(encoding="ascii"))
        row["family"] = "crossed_family"
        row = _with_hash({key: value for key, value in row.items() if key != "content_sha256"})
        path.write_bytes(auditor.canonical_json_bytes(row) + b"\n")
        _rebind_root_file(manifest, root, "pairs.jsonl")
        manifest["pair_index"]["file_sha256"] = _sha(path)
        manifest["ordered_pair_sha256"] = auditor.canonical_json_sha256(
            [row["content_sha256"]]
        )
    elif boundary == "endpoint":
        path = root / "endpoints.jsonl"
        row = json.loads(path.read_text(encoding="ascii"))
        row["family"] = "crossed_family"
        row = _with_hash({key: value for key, value in row.items() if key != "content_sha256"})
        path.write_bytes(auditor.canonical_json_bytes(row) + b"\n")
        _rebind_root_file(manifest, root, "endpoints.jsonl")
        manifest["endpoint_index"]["file_sha256"] = _sha(path)
        manifest["ordered_endpoint_sha256"] = auditor.canonical_json_sha256(
            [row["content_sha256"]]
        )
    else:
        shard_path = next((root / "shards").glob("*/shard.json"))
        index_path = shard_path.with_name("index.jsonl")
        row = json.loads(index_path.read_text(encoding="ascii"))
        row["family"] = "crossed_family"
        row = _with_hash({key: value for key, value in row.items() if key != "content_sha256"})
        index_path.write_bytes(auditor.canonical_json_bytes(row) + b"\n")
        shard = json.loads(shard_path.read_text(encoding="ascii"))
        local = next(item for item in shard["files"] if item["path"] == "index.jsonl")
        local["byte_count"] = index_path.stat().st_size
        local["file_sha256"] = _sha(index_path)
        shard = _with_hash(
            {key: value for key, value in shard.items() if key != "content_sha256"}
        )
        shard_path.write_bytes(auditor.canonical_json_bytes(shard) + b"\n")
        index_relative = str(index_path.relative_to(root))
        shard_relative = str(shard_path.relative_to(root))
        _rebind_root_file(manifest, root, index_relative)
        _rebind_root_file(manifest, root, shard_relative)
        manifest["shards"][0]["content_sha256"] = shard["content_sha256"]
    digest = _write_manifest(root, manifest)
    with pytest.raises(auditor.RawSupervisionAuditError):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_independent_missing_tree_entry_and_extra_special_entry_fail_closed(
    tmp_path: Path,
) -> None:
    root, digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    os.mkfifo(root / "unexpected.fifo")
    with pytest.raises(PermissionError, match="special entry"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


def test_block_public_exact_mode_accepts_caller_recompute_and_false_population(
    tmp_path: Path,
) -> None:
    root, digest, inputs, replay = _caller_injected_exact_fixture(tmp_path)
    callback_called = False

    def caller_recomputer(*_args: object) -> Mapping[str, tuple[np.ndarray, ...]]:
        nonlocal callback_called
        callback_called = True
        return replay

    with pytest.raises((PermissionError, auditor.RawSupervisionAuditError)):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=caller_recomputer,
            workers=1,
            exact=True,
        )
    assert callback_called is False


def test_block_invalid_authority_opens_source_before_complete_role_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    authorization_path = repository / "docs" / "authorization.json"
    selected = repository / "arbitrary" / "referenced_frames.jsonl"
    authorization_path.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"caller-selected repository payload\n")
    source_map = [
        {
            "role": "builder_source",
            "path": str(selected.relative_to(repository)),
            "sha256": _sha(selected),
        }
    ]
    authorization_core = {
        "schema": "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v1",
        "exact_build_authorized_after_independent_reviews": True,
        "builder_review": {"verdict": "PASS"},
        "auditor_review": {"verdict": "PASS"},
        "source_map": source_map,
    }
    authorization = _with_hash(authorization_core)
    authorization_raw = auditor.canonical_json_bytes(authorization) + b"\n"
    authorization_path.write_bytes(authorization_raw)
    provenance = {
        "authorization_file_sha256": hashlib.sha256(authorization_raw).hexdigest(),
        "authorization_content_sha256": authorization["content_sha256"],
        "authorization_source_map_sha256": auditor.canonical_json_sha256(source_map),
    }
    opened: list[Path] = []
    original = auditor._read_absolute_bound_payload

    def record_open(path: Path, *args: object, **kwargs: object) -> bytes:
        opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(auditor, "ROOT", repository)
    monkeypatch.setattr(auditor, "BUILD_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(auditor, "_read_absolute_bound_payload", record_open)
    with pytest.raises(auditor.RawSupervisionAuditError, match="source roles"):
        auditor._validate_exact_authorization(provenance)
    assert opened == [authorization_path]


def test_block_dataset_hard_link_alias_is_rejected(tmp_path: Path) -> None:
    root, digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    path = next((root / "shards").glob("*/camera_origin_body_m.f4"))
    outside_alias = tmp_path / "externally_mutable_alias.f4"
    os.link(path, outside_alias)
    assert path.stat().st_nlink == 2
    with pytest.raises(PermissionError, match="alias|link"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )


@pytest.mark.parametrize(
    ("field_path", "value"),
    (
        (("pair_index", "row_count"), 1.0),
        (("endpoint_index", "row_count"), True),
        (("shards", 0, "endpoint_count"), 1.0),
    ),
)
def test_block_noninteger_manifest_cardinality_is_rejected(
    tmp_path: Path, field_path: tuple[Any, ...], value: object
) -> None:
    root, _digest, inputs, replay = _one_endpoint_fixture(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    target: Any = manifest
    for component in field_path[:-1]:
        target = target[component]
    target[field_path[-1]] = value
    digest = _write_manifest(root, manifest)
    with pytest.raises(auditor.RawSupervisionAuditError, match="integer|count"):
        auditor.audit_dataset_v1(
            root,
            expected_manifest_file_sha256=digest,
            inputs=inputs,
            sample_recomputer=lambda *_args: replay,
        )
