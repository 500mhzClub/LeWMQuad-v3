from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_go2_observable_camera_ray_fit_v4 as builder


def _provenance() -> dict[str, object]:
    return {"schema": "synthetic_fit_v4_test_input", "dataset_role": "train"}


def _ledger() -> dict[str, object]:
    return {
        "rgb_byte_opens": 0,
        "fit_label_payload_byte_opens": 0,
        "selection_or_calibration_byte_opens": 0,
        "g2_byte_opens": 0,
        "model_or_checkpoint_byte_opens": 0,
    }


def _exact_build_ledger() -> dict[str, object]:
    ledger: dict[str, object] = {
        name: 0
        for name in builder.EXACT_BUILD_LEDGER_FIELDS
        if name
        not in {
            "per_shard_materialization",
            "denied_primary_reasons",
            "denied_modality_attempts",
            "denied_attempt_records",
        }
    }
    ledger["per_shard_materialization"] = []
    ledger["denied_primary_reasons"] = {
        name: 0 for name in builder.EXACT_DENIAL_PRIMARY_REASONS
    }
    ledger["denied_modality_attempts"] = {
        name: 0 for name in builder.EXACT_DENIAL_MODALITIES
    }
    ledger["denied_attempt_records"] = []
    return ledger


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): builder._sha256_file(path)
        for path in sorted(root.rglob("*"), key=str)
        if path.is_file()
    }


def test_synthetic_frame_build_retains_float32_range_authority() -> None:
    frame = builder.synthetic_scene_jobs(1)[0].frames[0]
    evidence = builder.build_frame_evidence_v4(frame)

    assert evidence.pixel_first_hit_distance_m.dtype == np.float32
    assert evidence.pixel_hit_mask.any()
    assert np.all(
        evidence.pixel_first_hit_distance_m[~evidence.pixel_hit_mask] == 0.0
    )
    assert np.isfinite(evidence.pixel_hit_xy_body_m).all()


def test_one_and_six_spawn_workers_publish_byte_identical_artifacts(
    tmp_path: Path,
) -> None:
    jobs = builder.synthetic_scene_jobs(6)
    one = tmp_path / "one"
    six = tmp_path / "six"
    first = builder.build_dataset_from_jobs(
        jobs,
        output_directory=one,
        workers=1,
        input_provenance=_provenance(),
        access_ledger=_ledger(),
    )
    second = builder.build_dataset_from_jobs(
        jobs,
        output_directory=six,
        workers=6,
        input_provenance=_provenance(),
        access_ledger=_ledger(),
    )

    assert first == second
    assert _file_hashes(one) == _file_hashes(six)
    assert first["parallel_contract"]["worker_start_method"] == "spawn"
    assert first["parallel_contract"]["maximum_workers"] == 6
    assert first["rgb_receipt"]["frame_count"] == 6
    assert first["rgb_receipt"]["rgb_byte_opens"] == 0


def test_dataset_publication_is_immutable_and_manifest_last(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest = builder.build_dataset_from_jobs(
        builder.synthetic_scene_jobs(1),
        output_directory=output,
        workers=1,
        input_provenance=_provenance(),
        access_ledger=_ledger(),
    )
    assert (output / "manifest.json").is_file()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="immutable"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=output,
            workers=1,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
        )


def test_duplicate_frame_identity_fails_before_publication(tmp_path: Path) -> None:
    job = builder.synthetic_scene_jobs(1)[0]
    duplicate = builder.SceneBuildJobV4(
        scene_key="second_scene", frames=job.frames
    )
    with pytest.raises(ValueError, match="repeat a frame key"):
        builder.build_dataset_from_jobs(
            (job, duplicate),
            output_directory=tmp_path / "dataset",
            workers=1,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
        )
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("workers", [0, 7])
def test_worker_count_is_bounded(workers: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=tmp_path / f"dataset_{workers}",
            workers=workers,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
        )


def test_manifest_carries_zero_nontrain_access_and_no_promotion(
    tmp_path: Path,
) -> None:
    manifest = builder.build_dataset_from_jobs(
        builder.synthetic_scene_jobs(1),
        output_directory=tmp_path / "dataset",
        workers=1,
        input_provenance=_provenance(),
        access_ledger=_ledger(),
    )
    assert manifest["dataset_role"] == "train"
    assert manifest["access_ledger"] == _ledger()
    assert not any(manifest["licenses"].values())


def test_unreviewed_real_implementation_manifest_cannot_open_exact_fit() -> None:
    raw = builder.IMPLEMENTATION_MANIFEST_PATH.read_bytes()
    with pytest.raises(PermissionError, match="not authorized"):
        builder._load_reviewed_implementation_manifest(
            hashlib.sha256(raw).hexdigest(),
            required_authorization="build",
        )


def test_direct_exact_job_opener_requires_reviewed_v4_authorization() -> None:
    raw = builder.IMPLEMENTATION_MANIFEST_PATH.read_bytes()
    with pytest.raises(PermissionError, match="not authorized"):
        builder.load_exact_fit_jobs(
            machine_manifest_sha256=(
                builder.SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256
            ),
            implementation_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_neutral_imports_do_not_preload_repository_semantics() -> None:
    command = (
        "import sys; "
        "from scripts import build_go2_observable_camera_ray_fit_v4; "
        "from scripts import audit_go2_observable_camera_ray_fit_v4; "
        "names=[n for n in sys.modules if n=='lewm' or n.startswith('lewm.') "
        "or n=='lewm_worlds' or n.startswith('lewm_worlds.')]; "
        "assert not names, names"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=builder.ROOT,
        env={**dict(os.environ), "PYTHONPATH": f"{builder.ROOT}:{builder.ROOT / 'lewm_worlds'}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_authorized_exact_boundary_reaches_source_gate_without_lewm_preload() -> None:
    code = """
import hashlib, json, sys, tempfile
from pathlib import Path
from scripts import build_go2_observable_camera_ray_fit_v4 as b
payload = json.loads(b.IMPLEMENTATION_MANIFEST_PATH.read_text())
payload['exact_fit_build_authorized_after_review'] = True
core = dict(payload); core.pop('content_sha256')
payload['content_sha256'] = b.canonical_json_sha256(core)
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / 'implementation.json'
    raw = b._canonical_json_bytes(payload) + b'\\n'
    path.write_bytes(raw)
    b.IMPLEMENTATION_MANIFEST_PATH = path
    def stop(name, source_path):
        names = [n for n in sys.modules if n == 'lewm' or n.startswith('lewm.')
                 or n == 'lewm_worlds' or n.startswith('lewm_worlds.')]
        assert not names, names
        raise RuntimeError('AUTHORIZED_ISOLATED')
    b._load_neutral_module = stop
    try:
        b.load_exact_fit_jobs(
            machine_manifest_sha256=b.SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256,
            implementation_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except RuntimeError as exc:
        assert str(exc) == 'AUTHORIZED_ISOLATED'
    else:
        raise AssertionError('source gate was not reached')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=builder.ROOT,
        env={
            **dict(os.environ),
            "PYTHONPATH": f"{builder.ROOT}:{builder.ROOT / 'lewm_worlds'}",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_camera_basis_matches_normalized_source_gram_construction() -> None:
    camera = SimpleNamespace(
        forward_xyz=(1.000001, 0.000002, -0.000003),
        left_xyz=(-0.000002, 0.999999, 0.000004),
        up_xyz=(0.000003, -0.000004, 1.000002),
    )
    basis = np.asarray(builder._normalized_camera_basis_fru(camera))
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(basis[1], basis[0]), basis[2], atol=1e-12)


def test_sidecar_attitude_must_match_passing_source_composition() -> None:
    quaternion = (0.0, 0.0, 0.0, 1.0)
    source = {
        "base_quat_world_xyzw": quaternion,
        "base_rpy_rad": {"yaw": 0.0},
        "camera_mount_composition": {
            "passes": True,
            "base_quat_world_xyzw": quaternion,
            "stored_base_yaw_rad": 0.0,
        },
    }
    endpoint = {
        "base_quat_world_xyzw": quaternion,
        "stored_base_yaw_rad": 0.0,
    }
    assert builder._validated_sidecar_source_attitude(source, endpoint) == (
        quaternion,
        0.0,
    )
    bad = dict(endpoint)
    bad["stored_base_yaw_rad"] = 0.1
    with pytest.raises(ValueError, match="disagrees"):
        builder._validated_sidecar_source_attitude(source, bad)
    source["camera_mount_composition"] = {
        **source["camera_mount_composition"],
        "passes": False,
    }
    with pytest.raises(ValueError, match="did not pass"):
        builder._validated_sidecar_source_attitude(source, endpoint)


def test_ground_clearance_uses_frozen_one_nanometre_tolerance() -> None:
    target = np.asarray([1.0, 1.0, 1.0])
    first = np.asarray([1.0, 1.0 - 0.5e-9, 1.0 - 2.0e-9])
    clear = builder._ground_support_clear(
        np.ones(3, dtype=bool), first, target
    )
    assert builder.GROUND_CLEARANCE_ABS_TOLERANCE_M == 1e-9
    assert clear.tolist() == [True, True, False]


def test_nontrivial_camera_and_range_match_independent_scalar_raycast() -> None:
    forward = np.asarray([0.81, 0.29, -0.51], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    basis = np.stack((forward, right, up))
    origin = np.asarray([0.37, -0.23, 1.41], dtype=np.float64)
    box_rotation = builder._rotation_xyz(0.19, -0.27, 0.31)
    box = builder.RayBoxV4(
        center_body_m=tuple(origin + 2.4 * forward),
        half_size_m=(0.42, 0.37, 0.33),
        rotation_body_from_box=tuple(tuple(row) for row in box_rotation),
    )
    frame = builder.FrameBuildInputV4(
        frame_key={
            "dataset_role": "train",
            "family": "synthetic_nontrivial",
            "global_row": 991,
        },
        camera_origin_body_m=tuple(origin),
        camera_basis_body_fru=tuple(tuple(row) for row in basis),
        ground_plane_z_body_m=-1.41,
        rendered_boxes_body=(box,),
        image_path_metadata_only=str(
            builder.ROOT / ".synthetic/nontrivial/rgb/frame.png"
        ),
        image_sha256=hashlib.sha256(b"nontrivial-rgb").hexdigest(),
        sidecar_row_identity_sha256=hashlib.sha256(
            b"nontrivial-sidecar"
        ).hexdigest(),
    )
    evidence = builder.build_frame_evidence_v4(frame)

    height, width = builder.ray_v4.CAMERA_IMAGE_SHAPE
    stride = builder.ray_v4.PIXEL_RAY_STRIDE_PX
    pixel_x = np.minimum(
        np.arange(0, width, stride, dtype=np.float64) + 0.5 * stride,
        width - 0.5,
    )
    pixel_y = np.minimum(
        np.arange(0, height, stride, dtype=np.float64) + 0.5 * stride,
        height - 0.5,
    )
    tan_h = np.tan(
        np.deg2rad(builder.ray_v4.CAMERA_HORIZONTAL_FOV_DEG / 2.0)
    )
    tan_v = np.tan(
        np.deg2rad(builder.ray_v4.CAMERA_VERTICAL_FOV_DEG / 2.0)
    )
    grid_x, grid_y = np.meshgrid(
        (2.0 * pixel_x / width - 1.0) * tan_h,
        (1.0 - 2.0 * pixel_y / height) * tan_v,
        indexing="xy",
    )
    stored_basis = np.asarray(evidence.camera_basis_body_fru, dtype=np.float64)
    directions = (
        stored_basis[0]
        + grid_x[..., None] * stored_basis[1]
        + grid_y[..., None] * stored_basis[2]
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    stored_origin = np.asarray(evidence.camera_origin_body_m, dtype=np.float64)
    local_origin = box_rotation.T @ (
        stored_origin - np.asarray(box.center_body_m)
    )

    def scalar_entry(direction: np.ndarray) -> float:
        local_direction = box_rotation.T @ direction
        lower, upper = -np.inf, np.inf
        for axis, half in enumerate(box.half_size_m):
            component = float(local_direction[axis])
            if abs(component) <= 1e-12:
                if abs(float(local_origin[axis])) > half + 1e-12:
                    return np.inf
                continue
            first = (-half - float(local_origin[axis])) / component
            second = (half - float(local_origin[axis])) / component
            lower = max(lower, min(first, second))
            upper = min(upper, max(first, second))
        entry = max(lower, 0.0)
        return entry if upper + 1e-12 >= entry else np.inf

    expected = np.asarray(
        [scalar_entry(ray) for ray in directions.reshape(-1, 3)]
    ).reshape(builder.PIXEL_RAY_SHAPE)
    expected_mask = np.isfinite(expected) & (expected > builder.CAMERA_NEAR_M)
    expected_f32 = np.zeros(builder.PIXEL_RAY_SHAPE, dtype=np.float32)
    expected_f32[expected_mask] = expected[expected_mask].astype(np.float32)
    np.testing.assert_array_equal(evidence.pixel_hit_mask, expected_mask)
    np.testing.assert_allclose(
        evidence.pixel_first_hit_distance_m,
        expected_f32,
        rtol=0.0,
        atol=2e-6,
    )
    assert 100 < int(expected_mask.sum()) < expected_mask.size
    assert float(expected_f32[expected_mask].max()) > 2.0


def test_staged_shard_rejects_extra_and_tampered_files(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = builder._write_scene_job(
        builder.synthetic_scene_jobs(1)[0], str(first_root)
    )
    first_directory = Path(first["staging_path"])
    (first_directory / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        builder._validate_shard_directory(first_directory, first["shard"])

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = builder._write_scene_job(
        builder.synthetic_scene_jobs(1)[0], str(second_root)
    )
    second_directory = Path(second["staging_path"])
    target = second_directory / "pixel_hit_mask_u8.bin"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 1
    target.write_bytes(payload)
    with pytest.raises(ValueError, match="file changed"):
        builder._validate_shard_directory(second_directory, second["shard"])


def test_required_output_root_blocks_escape_before_creation(tmp_path: Path) -> None:
    output = tmp_path / "outside" / "dataset"
    with pytest.raises(PermissionError, match="escapes"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=output,
            workers=1,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
            required_output_root=builder.ROOT,
        )
    assert not output.exists()


def _implementation_entries() -> list[dict[str, str]]:
    payload = json.loads(builder.IMPLEMENTATION_MANIFEST_PATH.read_text())
    return [dict(entry) for entry in payload["source_map"]["entries"]]


def _authorized_semantic_hashes() -> dict[str, dict[str, str]]:
    source_access = builder._load_neutral_module(
        "go2_v4_builder_test_source_access", builder.SOURCE_ACCESS_PATH
    )
    ledger = source_access.new_access_ledger()
    return source_access._source_hashes(
        source_access.AuditSpec().sources(), ledger=ledger
    )


def _source_revalidation_provenance() -> dict[str, object]:
    return {
        **_provenance(),
        "source_hashes": _authorized_semantic_hashes(),
        "implementation_manifest_file_sha256": builder._sha256_file(
            builder.IMPLEMENTATION_MANIFEST_PATH
        ),
        "source_authorization_manifest_file_sha256": (
            builder.SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256
        ),
        "sidecar_manifest_file_sha256": builder.SIDECAR_MANIFEST_FILE_SHA256,
    }


def test_worker_and_parent_revalidate_frozen_source_closure(tmp_path: Path) -> None:
    entries = _implementation_entries()
    manifest = builder.build_dataset_from_jobs(
        builder.synthetic_scene_jobs(1),
        output_directory=tmp_path / "dataset",
        workers=1,
        input_provenance=_source_revalidation_provenance(),
        access_ledger=_exact_build_ledger(),
        source_closure_entries=entries,
    )
    assert manifest["parallel_contract"]["per_worker_source_revalidation"]
    assert manifest["parallel_contract"][
        "parent_source_revalidation_before_manifest"
    ]


def test_transitive_source_hash_mutation_fails_before_worker_or_output(
    tmp_path: Path,
) -> None:
    entries = _implementation_entries()
    entries[0]["sha256"] = "0" * 64
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="implementation source changed"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=output,
            workers=1,
            input_provenance=_source_revalidation_provenance(),
            access_ledger=_exact_build_ledger(),
            source_closure_entries=entries,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".go2_ray_v4.*"))


def test_spawned_worker_detects_source_mutation_after_parent_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = builder.ROOT / ".generated"
    generated.mkdir(exist_ok=True)
    original_get_context = builder.multiprocessing.get_context
    with tempfile.TemporaryDirectory(
        prefix=".go2_v4_worker_source_test.", dir=generated
    ) as temporary:
        source_root = Path(temporary)
        entries: list[dict[str, str]] = []
        for index, role in enumerate(sorted(builder.TRANSITIVE_SOURCE_ROLES)):
            path = source_root / f"source_{index:02d}.py"
            path.write_text(f"ROLE = {role!r}\n")
            entries.append(
                {
                    "path": str(path.relative_to(builder.ROOT)),
                    "role": role,
                    "sha256": builder._sha256_file(path),
                }
            )
        mutation_target = source_root / "source_00.py"
        mutated = False

        def mutate_before_spawn(method: str):
            nonlocal mutated
            if not mutated:
                mutation_target.write_text(
                    "ROLE = 'mutated_after_parent_validation'\n"
                )
                mutated = True
            return original_get_context(method)

        monkeypatch.setattr(
            builder.multiprocessing, "get_context", mutate_before_spawn
        )
        output = tmp_path / "dataset"
        with pytest.raises(ValueError, match="implementation source changed"):
            builder.build_dataset_from_jobs(
                builder.synthetic_scene_jobs(1),
                output_directory=output,
                workers=1,
                input_provenance=_source_revalidation_provenance(),
                access_ledger=_exact_build_ledger(),
                source_closure_entries=entries,
            )
        assert mutated
        assert not output.exists()
        assert not list(tmp_path.glob(".go2_ray_v4.*"))


def test_postpublication_extra_file_fails_and_cleans_all_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = builder._fsync_directory
    injected = False

    def inject_after_publish(path: Path) -> None:
        nonlocal injected
        original_fsync(path)
        if path.parent.name == "shards" and not injected:
            (path / "unexpected.postpublish").write_bytes(b"unexpected")
            injected = True

    monkeypatch.setattr(builder, "_fsync_directory", inject_after_publish)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="inventory"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=output,
            workers=1,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
        )
    assert injected
    assert not output.exists()
    assert not list(tmp_path.glob(".go2_ray_v4.*"))


def test_postmanifest_failure_removes_manifest_extra_and_owned_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "dataset"
    manifest_path = output / "manifest.json"
    extra_path = output / "unexpected.after_manifest"
    original_write = builder._write_json_exclusive
    injected = False

    def write_then_inject(path: Path, payload) -> None:
        nonlocal injected
        original_write(path, payload)
        if path == manifest_path:
            assert manifest_path.is_file()
            extra_path.write_bytes(b"post-manifest failure injection")
            injected = True

    monkeypatch.setattr(builder, "_write_json_exclusive", write_then_inject)
    with pytest.raises(ValueError, match="root inventory"):
        builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(1),
            output_directory=output,
            workers=1,
            input_provenance=_provenance(),
            access_ledger=_ledger(),
        )
    assert injected
    assert not output.exists()
    assert not manifest_path.exists()
    assert not extra_path.exists()
    assert not list(tmp_path.glob(".go2_ray_v4.*"))


def test_denied_or_unexpected_access_receipt_fails_closed() -> None:
    base = {
        "unexpected_path_attempts": 0,
        "denied_attempts_total": 0,
        "denied_attempt_records": [],
        "denied_primary_reasons": {
            name: 0 for name in builder.EXACT_DENIAL_PRIMARY_REASONS
        },
        "denied_modality_attempts": {
            name: 0 for name in builder.EXACT_DENIAL_MODALITIES
        },
    }
    builder._assert_zero_access_denials(base)

    missing_reason = {**base, "denied_primary_reasons": {"g2": 0}}
    with pytest.raises(ValueError, match="schema changed"):
        builder._assert_zero_access_denials(missing_reason)

    inconsistent = {**base, "denied_attempts_total": 1}
    with pytest.raises(ValueError, match="totals and records"):
        builder._assert_zero_access_denials(inconsistent)

    primary = dict(base["denied_primary_reasons"])
    modality = dict(base["denied_modality_attempts"])
    primary["g2"] = 1
    modality["image"] = 1
    denied = {
        **base,
        "unexpected_path_attempts": 1,
        "denied_attempts_total": 1,
        "denied_attempt_records": [{}],
        "denied_primary_reasons": primary,
        "denied_modality_attempts": modality,
    }
    with pytest.raises(PermissionError, match="denied or unexpected"):
        builder._assert_zero_access_denials(denied)


def test_cli_separates_dry_run_and_exact_authorization() -> None:
    dry = builder._parse_args(["--dry-run"])
    assert dry.dry_run
    with pytest.raises(SystemExit):
        builder._parse_args(["--run-exact-fit"])
    with pytest.raises(SystemExit):
        builder._parse_args(
            ["--dry-run", "--machine-manifest-sha256", "0" * 64]
        )


def test_dry_run_reports_no_fit_or_gpu_access() -> None:
    result = builder.run_dry_run()
    assert result["one_vs_six_worker_byte_identical"]
    assert result["fit_payload_opened"] is False
    assert result["nontrain_payload_opened"] is False
    assert result["gpu_used"] is False
