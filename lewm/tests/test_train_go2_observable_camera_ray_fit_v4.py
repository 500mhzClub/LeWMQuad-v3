from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import train_go2_observable_camera_ray_fit_v4 as trainer


def _index_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(320):
        digest = f"{index + 1:064x}"
        rows.append(
            {
                "frame_key": {
                    "dataset_role": "train",
                    "family": trainer.FAMILIES[index % len(trainer.FAMILIES)],
                    "scene_id": f"scene_{index // 16:02d}",
                    "global_row": index,
                    "side": "current" if index % 2 == 0 else "next",
                    "image_sha256": digest,
                    "label_shard_sha256": f"{index + 321:064x}",
                    "label_row": index,
                }
            }
        )
    return rows


def test_fit_subsets_are_deterministic_nested_and_support_only_registered_sizes() -> None:
    rows = _index_rows()
    selections = {}
    for size in trainer.SUPPORTED_FIT_SIZES:
        selected, receipt = trainer.deterministic_fit_subset(rows, size)
        selections[size] = [row["frame_key"] for row in selected]
        assert len(selected) == size
        assert receipt["fit_size"] == size
        assert len(receipt["ordered_frame_key_sha256"]) == size
        counts = list(receipt["family_counts"].values())
        assert max(counts) - min(counts) <= 1
    assert selections[5] == selections[16][:5]
    assert selections[16] == selections[32][:16]
    assert selections[32] == selections[320][:32]
    reversed_selected, _ = trainer.deterministic_fit_subset(list(reversed(rows)), 32)
    assert [row["frame_key"] for row in reversed_selected] == selections[32]
    with pytest.raises(ValueError):
        trainer.deterministic_fit_subset(rows, 2)
    with pytest.raises(ValueError):
        trainer.deterministic_fit_subset(rows[:-1], 32)


def test_frozen_subset_separator_and_exact_metadata_commitments() -> None:
    assert trainer.SUBSET_RANK_SEPARATOR.hex() == "5c30"
    assert trainer.ladder_gate.EXPECTED_FIRST_FRAME_KEY_SHA256[5] == (
        "245abcda3add36f7bd066189b8d7314e7ece21b28006c51d57a3875454ee8b28"
    )
    assert trainer.ladder_gate.EXPECTED_LAST_FRAME_KEY_SHA256[320] == (
        "fde2a06b7d46f68a27179acd0b1c2e1de5eda5516cb36c58f00dcaed89b174f5"
    )
    assert trainer.ladder_gate.EXPECTED_SUBSET_CONTENT_SHA256 == {
        5: "3595dff9d24dbb44f3e73086fce3be4ec53eb8659684738defa8591c4a375f15",
        16: "3e3706c4d46476c9d6682e92bd80aa97bd7b0f0bd5bc2c9b69b9aa3605f9d4ba",
        32: "19ae70495e7a21e4ecacd7846672145ffc0187ced6b4f9296c7f9e5b4d46ed73",
        320: "be4b8863120d67132180228982f0631f5f8f6042b581ee5f8a61559fa58188b1",
    }


def test_frozen_geometry_provenance_preflight_precedes_reservation_and_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "beddb29b9826d7a21968effea863d040a6cfc9849ab0b2a78c4105d28dbb37d2"
    assert trainer.SOURCE_GEOMETRY_MANIFEST_SHA256 == expected

    attempt_root = tmp_path / "attempts"
    monkeypatch.setattr(trainer, "CANONICAL_ATTEMPT_ROOT", attempt_root)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provenance preflight crossed into payload or reservation work")

    for name in (
        "_load_dataset_frames",
        "reserve_exact_attempt",
        "validate_gpu0_r9700_runtime",
        "decode_selected_rgb",
        "train_v4_fit",
    ):
        monkeypatch.setattr(trainer, name, forbidden)

    provenance = trainer.preflight_exact_frozen_dataset_provenance(
        dataset_manifest_path=trainer.CANONICAL_DATASET_MANIFEST_PATH,
        dataset_manifest_file_sha256=(
            trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256
        ),
    )

    assert provenance["source_geometry_manifest_sha256"] == expected
    assert not attempt_root.exists()


def _write_hashed_object(path: Path, core: dict[str, object]) -> str:
    value = {**core, "content_sha256": trainer.canonical_json_sha256(core)}
    payload = trainer._canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_strict_hashed_object_checks_caller_sha_canonical_bytes_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    digest = _write_hashed_object(path, {"schema": "synthetic", "value": 3})
    value, raw = trainer._strict_hashed_object(
        path, digest, name="synthetic receipt", allowed_root=tmp_path
    )
    assert value["value"] == 3
    assert hashlib.sha256(raw).hexdigest() == digest
    with pytest.raises(ValueError, match="raw file SHA"):
        trainer._strict_hashed_object(
            path, "0" * 64, name="synthetic receipt", allowed_root=tmp_path
        )

    parsed = json.loads(path.read_text())
    noncanonical = json.dumps(parsed, indent=2).encode() + b"\n"
    path.write_bytes(noncanonical)
    with pytest.raises(ValueError, match="canonical"):
        trainer._strict_hashed_object(
            path,
            hashlib.sha256(noncanonical).hexdigest(),
            name="synthetic receipt",
            allowed_root=tmp_path,
        )

    duplicate = b'{"content_sha256":"' + b"0" * 64 + b'","schema":"x","schema":"y"}\n'
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        trainer._strict_hashed_object(
            path,
            hashlib.sha256(duplicate).hexdigest(),
            name="synthetic receipt",
            allowed_root=tmp_path,
        )


def test_determinism_warning_whitelist_is_exact() -> None:
    grid, scatter = trainer.DETERMINISM_WARNING_WHITELIST
    raw_grid = (
        grid
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)"
    )
    accepted = trainer.validate_determinism_warnings([raw_grid, scatter])
    assert accepted["warning_count"] == 2
    assert accepted["raw_messages"] == [raw_grid, scatter]
    assert accepted["normalized_messages"] == [grid, scatter]
    assert accepted["kernel_inventory"] == [
        "grid_sampler_2d_backward_cuda",
        "scatter_add_cuda_kernel",
    ]
    assert accepted["normalization"] == [
        {
            "raw": raw_grid,
            "normalized": grid,
            "context_source_line": 157,
            "trailer_removed": True,
        },
        {
            "raw": scatter,
            "normalized": scatter,
            "context_source_line": None,
            "trailer_removed": False,
        },
    ]
    assert accepted["kernel_counts"] == {
        "grid_sampler_2d_backward_cuda": 1,
        "scatter_add_cuda_kernel": 1,
    }
    trainer.validate_determinism_warnings([])


@pytest.mark.parametrize(
    "message",
    (
        "grid_sampler_2d_backward_cudb"
        + trainer._DETERMINISM_WARNING_SUFFIX
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)",
        "grid_sampler_2d_backward_cuda has a slightly different warning",
        trainer.DETERMINISM_WARNING_WHITELIST[0]
        + " (Triggered internally at /pytorch/aten/src/ATen/Other.cpp:157.)",
        trainer.DETERMINISM_WARNING_WHITELIST[0]
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:0157.)",
        trainer.DETERMINISM_WARNING_WHITELIST[0]
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)"
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:158.)",
    ),
)
def test_determinism_warning_normalizer_rejects_every_other_variation(
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match="unexpected training warning"):
        trainer.validate_determinism_warnings([message])


@dataclass
class _FakeProperties:
    total_memory: int


class _FakeCuda:
    def __init__(self, *, name: str, count: int = 1, available: bool = True) -> None:
        self.name = name
        self.count = count
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self.name

    def get_device_properties(self, index: int) -> _FakeProperties:
        assert index == 0
        return _FakeProperties(total_memory=20 * 1024**3)


def _resource_environment() -> dict[str, str]:
    return {
        "HIP_VISIBLE_DEVICES": "0",
        **{name: "1" for name in trainer.THREAD_ENVIRONMENT},
    }


def test_gpu_gate_accepts_only_gpu0_r9700_and_thread_caps() -> None:
    fake_torch = SimpleNamespace(cuda=_FakeCuda(name="AMD Radeon AI PRO R9700"))
    receipt = trainer.validate_gpu0_r9700_runtime(
        device_text="cuda:0",
        environ=_resource_environment(),
        torch_module=fake_torch,
    )
    assert receipt["device"] == "cuda:0"
    assert receipt["raphael_rejected"] is True

    with pytest.raises(PermissionError, match="Raphael"):
        trainer.validate_gpu0_r9700_runtime(
            device_text="cuda:0",
            environ=_resource_environment(),
            torch_module=SimpleNamespace(cuda=_FakeCuda(name="AMD Raphael")),
        )
    with pytest.raises(RuntimeError, match="exactly one"):
        trainer.validate_gpu0_r9700_runtime(
            device_text="cuda:0",
            environ=_resource_environment(),
            torch_module=SimpleNamespace(
                cuda=_FakeCuda(name="AMD Radeon AI PRO R9700", count=2)
            ),
        )
    bad_environment = _resource_environment()
    bad_environment["OMP_NUM_THREADS"] = "8"
    with pytest.raises(PermissionError, match="thread caps"):
        trainer.validate_gpu0_r9700_runtime(
            device_text="cuda:0",
            environ=bad_environment,
            torch_module=fake_torch,
        )
    override_environment = _resource_environment()
    override_environment["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    with pytest.raises(PermissionError, match="must be unset"):
        trainer.validate_gpu0_r9700_runtime(
            device_text="cuda:0",
            environ=override_environment,
            torch_module=fake_torch,
        )


def test_skew_balanced_offset_weights_nonempty_depth_bins_equally() -> None:
    predicted = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 3.0, 3.0, 3.0]]]])
    raw = SimpleNamespace(pixel_within_bin_offset_m=predicted)
    targets = SimpleNamespace(
        pixel_hit_bin_index=torch.tensor([[[0, 1, 1, 1]]]),
        pixel_in_range_hit_mask=torch.ones((1, 1, 4), dtype=torch.bool),
        pixel_within_bin_offset_m=torch.zeros((1, 1, 4)),
    )
    loss = trainer._skew_balanced_pixel_offset_loss(raw, targets)
    assert loss.item() == pytest.approx(1.995, abs=1e-6)


def test_four_losses_have_exact_quarter_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    target = SimpleNamespace(ground_in_frustum=torch.tensor([True]))
    raw = SimpleNamespace(
        ground_query_in_frustum=torch.tensor([True]),
        pixel_first_hit_hazard_logits=torch.tensor(0.0),
        ground_clear_to_target_logits=torch.tensor(0.0),
        ground_target_distance_m=torch.tensor(0.0),
    )
    soft = object()
    monkeypatch.setattr(
        trainer,
        "derive_observable_camera_ray_evidence_v4_targets",
        lambda **_kwargs: target,
    )
    monkeypatch.setattr(
        trainer,
        "ordered_obstacle_first_hit_nll_breakdown_v4",
        lambda *_args: SimpleNamespace(total=torch.tensor(2.0)),
    )
    monkeypatch.setattr(
        trainer, "_skew_balanced_pixel_offset_loss", lambda *_args: torch.tensor(4.0)
    )
    monkeypatch.setattr(
        trainer, "balanced_ground_clear_bce_v4", lambda *_args: torch.tensor(6.0)
    )
    monkeypatch.setattr(
        trainer,
        "soft_rasterize_observable_camera_ray_evidence_v4",
        lambda *_args, **_kwargs: soft,
    )
    monkeypatch.setattr(
        trainer,
        "hierarchical_raster_cross_entropy_v4",
        lambda *_args: SimpleNamespace(total=torch.tensor(8.0)),
    )
    batch = SimpleNamespace(
        pixel_hit_mask=None,
        pixel_first_hit_distance_m=None,
        ground_support_in_frustum=None,
        ground_support_clear_to_target=None,
        image=None,
        camera_origin_body_m=None,
        camera_basis_body_fru=None,
        ground_plane_z_body_m=None,
        target_raster_labels=None,
    )
    total, components, returned_raw, returned_target, returned_soft = (
        trainer.compute_four_equal_v4_losses(lambda *_args: raw, batch)
    )
    assert total.item() == pytest.approx(5.0)
    assert [value.item() for value in components.values()] == [2.0, 4.0, 6.0, 8.0]
    assert returned_raw is raw
    assert returned_target is target
    assert returned_soft is soft


def _fake_batch_frames() -> tuple[list[SimpleNamespace], torch.Tensor]:
    frames = []
    for index in range(2):
        evidence = SimpleNamespace(
            camera_origin_body_m=np.full((3,), index, dtype=np.float32),
            camera_basis_body_fru=np.full((3, 3), index, dtype=np.float32),
            ground_plane_z_body_m=float(index),
            pixel_hit_mask=np.full((1, 1), index, dtype=bool),
            pixel_first_hit_distance_m=np.full((1, 1), index, dtype=np.float32),
            ground_support_in_frustum=np.full((1, 1, 5), index, dtype=bool),
            ground_support_clear_to_target=np.full((1, 1, 5), index, dtype=bool),
        )
        frames.append(
            SimpleNamespace(
                evidence=evidence,
                target_raster_labels=np.full((1, 1), index, dtype=np.uint8),
                family=f"family_{index}",
            )
        )
    images = torch.stack(
        (
            torch.zeros((3, trainer.IMAGE_SIZE, trainer.IMAGE_SIZE)),
            torch.ones((3, trainer.IMAGE_SIZE, trainer.IMAGE_SIZE)),
        )
    )
    return frames, images


def test_wrong_rgb_batch_mapping_keeps_target_calibration() -> None:
    frames, images = _fake_batch_frames()
    batch = trainer._batch_from_indices(
        frames,
        images,
        (0, 1),
        image_indices=(1, 0),
    )
    assert torch.equal(batch.image[0], images[1])
    assert torch.equal(batch.image[1], images[0])
    assert batch.camera_origin_body_m[:, 0].tolist() == [0.0, 1.0]
    assert batch.ground_plane_z_body_m.tolist() == [0.0, 1.0]
    assert batch.target_raster_labels[:, 0, 0].tolist() == [0, 1]


def test_immutable_publication_is_manifest_last_and_no_replace(tmp_path: Path) -> None:
    result = {"schema": "synthetic", "content_sha256": "0" * 64}
    published = trainer.publish_immutable_development_result(
        output_root=tmp_path,
        run_name="n1_seed1_synthetic",
        checkpoint_payload=b"checkpoint",
        result=result,
        enforce_canonical_root=False,
    )
    assert Path(published["checkpoint_path"]).read_bytes() == b"checkpoint"
    assert json.loads(Path(published["result_path"]).read_text()) == result
    with pytest.raises(FileExistsError):
        trainer.publish_immutable_development_result(
            output_root=tmp_path,
            run_name="n1_seed1_synthetic",
            checkpoint_payload=b"replacement",
            result=result,
            enforce_canonical_root=False,
        )


def _patch_attempt_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    development = tmp_path / "development_fit_v2"
    monkeypatch.setattr(trainer, "CANONICAL_DEVELOPMENT_OUTPUT_ROOT", development)
    monkeypatch.setattr(trainer, "CANONICAL_ATTEMPT_ROOT", development / "attempts")


def test_v1_failure_is_immutable_and_v2_root_is_strictly_separate() -> None:
    v1 = trainer.V1_DEVELOPMENT_OUTPUT_ROOT
    v2 = trainer.CANONICAL_DEVELOPMENT_OUTPUT_ROOT
    reservation = v1 / "attempts/seed_20260710/n5/reservation.json"
    failed = v1 / "attempts/seed_20260710/n5/failed.json"
    before = {
        reservation: hashlib.sha256(reservation.read_bytes()).hexdigest(),
        failed: hashlib.sha256(failed.read_bytes()).hexdigest(),
    }

    lineage = trainer.validate_v1_failure_lineage()

    assert lineage == trainer.ladder_gate.V1_FAILURE_LINEAGE
    assert v1.resolve() != v2.resolve()
    assert not v2.exists()
    assert {path.name for path in reservation.parent.iterdir()} == {
        "reservation.json",
        "failed.json",
    }
    assert not (reservation.parent / "checkpoint.pt").exists()
    assert not (reservation.parent / "result.json").exists()
    assert not (v1 / "metric_verifications").exists()
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
    } == before


def test_exact_attempt_is_reserved_once_and_failure_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_attempt_roots(monkeypatch, tmp_path)
    reservation = trainer.reserve_exact_attempt(
        seed=20260710,
        fit_size=5,
        inputs={"synthetic": "input"},
        prerequisite_gates={
            "previous_stage_gate": None,
            "seed_20260710_gate": None,
        },
    )
    assert reservation.reservation["contract"] == trainer.LADDER_CONTRACT
    assert reservation.reservation["predecessor_failure"] == (
        trainer.ladder_gate.V1_FAILURE_LINEAGE
    )
    (reservation.directory / "checkpoint.pt").write_bytes(b"partial")
    failed = trainer.fail_reserved_exact_attempt(
        reservation,
        error=RuntimeError("raw path and payload text must not be recorded"),
    )
    assert set(path.name for path in reservation.directory.iterdir()) == {
        "reservation.json",
        "failed.json",
    }
    failure_value = json.loads(Path(failed["path"]).read_text())
    assert failure_value["failure"] == {
        "code": "execution_failure",
        "class": "runtime",
    }
    assert "raw path" not in Path(failed["path"]).read_text()
    with pytest.raises(FileExistsError):
        trainer.reserve_exact_attempt(
            seed=20260710,
            fit_size=5,
            inputs={"synthetic": "input"},
            prerequisite_gates={
                "previous_stage_gate": None,
                "seed_20260710_gate": None,
            },
        )


def test_reserved_success_completion_is_last_and_binds_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_attempt_roots(monkeypatch, tmp_path)
    reservation = trainer.reserve_exact_attempt(
        seed=20260710,
        fit_size=16,
        inputs={"synthetic": "input"},
        prerequisite_gates={
            "previous_stage_gate": {"synthetic": "gate"},
            "seed_20260710_gate": None,
        },
    )
    result_core = {"schema": "synthetic_result", "value": 16}
    result = {
        **result_core,
        "content_sha256": trainer.canonical_json_sha256(result_core),
    }
    published = trainer.publish_reserved_exact_attempt(
        reservation,
        checkpoint_payload=b"synthetic checkpoint",
        checkpoint_content_sha256="a" * 64,
        result=result,
    )
    assert set(path.name for path in reservation.directory.iterdir()) == {
        "reservation.json",
        "checkpoint.pt",
        "result.json",
        "completed.json",
    }
    completion = json.loads(Path(published["completion"]["path"]).read_text())
    assert completion["reservation"] == reservation.binding
    assert completion["checkpoint"]["file_sha256"] == hashlib.sha256(
        b"synthetic checkpoint"
    ).hexdigest()
    assert completion["result"]["content_sha256"] == result["content_sha256"]


@pytest.mark.parametrize(
    "failure_point",
    (
        "after_checkpoint_write",
        "after_result_write",
        "after_completion_write",
        "after_directory_fsync",
    ),
)
def test_reserved_publication_failure_becomes_failure_only_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _patch_attempt_roots(monkeypatch, tmp_path)
    fit_size = {
        "after_checkpoint_write": 5,
        "after_result_write": 16,
        "after_completion_write": 32,
        "after_directory_fsync": 320,
    }[failure_point]
    reservation = trainer.reserve_exact_attempt(
        seed=20260711,
        fit_size=fit_size,
        inputs={"synthetic": "input"},
        prerequisite_gates={
            "previous_stage_gate": None,
            "seed_20260710_gate": {"synthetic": "seed gate"},
        },
    )
    core = {"schema": "synthetic_result", "fit_size": fit_size}
    result = {**core, "content_sha256": trainer.canonical_json_sha256(core)}
    error = RuntimeError("not persisted")
    with pytest.raises(RuntimeError, match="injected V4 publication failure"):
        try:
            trainer.publish_reserved_exact_attempt(
                reservation,
                checkpoint_payload=b"checkpoint",
                checkpoint_content_sha256="a" * 64,
                result=result,
                failure_injection=failure_point,
            )
        except RuntimeError as caught:
            error = caught
            raise
    trainer.fail_reserved_exact_attempt(reservation, error=error)
    assert {path.name for path in reservation.directory.iterdir()} == {
        "reservation.json",
        "failed.json",
    }


def test_mutating_trainer_module_metadata_does_not_grant_exact_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = Path(trainer.__file__).read_bytes()
    preauthorization = {
        "source_map": {
            "entries": [
                {
                    "path": "scripts/train_go2_observable_camera_ray_fit_v4.py",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
        }
    }
    monkeypatch.setattr(trainer, "__name__", "_lewm_v4_ca_attacker")
    monkeypatch.setattr(
        trainer,
        "__verified_logical_name__",
        "scripts.train_go2_observable_camera_ray_fit_v4",
        raising=False,
    )
    monkeypatch.setattr(
        trainer,
        "__verified_source_sha256__",
        hashlib.sha256(payload).hexdigest(),
        raising=False,
    )
    with pytest.raises(PermissionError, match="library execution is unsupported"):
        trainer._require_captured_private_trainer(preauthorization)


def test_rgb_decode_has_no_injected_callback_and_spawn_requires_fixed_authority(
    tmp_path: Path,
) -> None:
    frames = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (16, 12), color).save(path)
        payload = path.read_bytes()
        frames.append(
            SimpleNamespace(
                rgb_path=path,
                image_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    assert not hasattr(trainer, "__verified_rgb_worker_dispatch__")
    assert not hasattr(
        trainer, "__verified_rgb_worker_authorization_file_sha256__"
    )
    with pytest.raises(PermissionError, match="fixed canonical authority bindings"):
        trainer.decode_selected_rgb(
            frames,
            maximum_workers=2,
            allowed_rgb_root=tmp_path,
        )
    images, receipt = trainer.decode_selected_rgb(
        frames,
        maximum_workers=1,
        allowed_rgb_root=tmp_path,
    )
    assert tuple(images.shape) == (2, 3, trainer.IMAGE_SIZE, trainer.IMAGE_SIZE)
    assert receipt["worker_start_method"] == "inline"
    assert receipt["worker_count"] == 1
    frames[0].rgb_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        trainer.decode_selected_rgb(
            frames,
            maximum_workers=1,
            allowed_rgb_root=tmp_path,
        )


def test_cli_modes_do_not_mix_exact_and_smoke_inputs() -> None:
    with pytest.raises(ValueError, match="may not receive exact"):
        trainer.main(
            [
                "--smoke",
                "--dataset-manifest",
                "never-open.json",
            ]
        )
    with pytest.raises(PermissionError, match="in-memory capability"):
        trainer.main([])


def _load_frozen_authorization() -> dict[str, object]:
    raw = trainer.TRAINER_AUTHORIZATION_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value, verified_raw = trainer._strict_hashed_object(
        trainer.TRAINER_AUTHORIZATION_PATH,
        digest,
        name="frozen trainer authorization",
        allowed_root=trainer.ROOT,
    )
    assert verified_raw == raw
    return value


def test_frozen_trainer_snapshot_is_complete_canonical_and_narrowly_authorized() -> None:
    authorization = _load_frozen_authorization()
    sources = trainer._validate_trainer_authorization_snapshot_schema(authorization)
    assert authorization["status"] == "independent_review_passed_authorized"
    assert authorization["authorization"] == {
        "development_fit": True,
        "development_checkpoint_creation_authorized": True,
        "checkpoint_use_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    assert (
        authorization["dataset_binding"]["file_sha256"]
        == trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256
    )
    assert (
        authorization["audit_binding"]["file_sha256"]
        == trainer.preauth_launcher.AUDIT_RECEIPT_FILE_SHA256
    )
    assert len(sources) == len(trainer.preauth_launcher.REQUIRED_SOURCE_ROLES)
    for path, expected_sha256 in sources:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    validated = trainer._validate_trainer_authorization(
        authorization,
        dataset_manifest_file_sha256=trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256,
        dataset_manifest_content_sha256=trainer.preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
        audit_receipt_file_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_FILE_SHA256,
        audit_receipt_content_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
        review_record_file_sha256=authorization["review_record"]["file_sha256"],
    )
    assert validated == sources


def test_future_authorization_requires_exact_review_and_full_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = copy.deepcopy(_load_frozen_authorization())
    review_core = {
        "schema": trainer.preauth_launcher.REVIEW_RECORD_SCHEMA,
        "status": "independent_review_passed",
        "decision": "pass",
        "reviewer": "synthetic_test_reviewer",
        "reviewed_source_map_sha256": authorization["source_map"][
            "source_map_sha256"
        ],
        "restricted_payload_opened": False,
        "findings": [],
    }
    review = {
        **review_core,
        "content_sha256": trainer.canonical_json_sha256(review_core),
    }
    review_path = (tmp_path / "review.json").resolve()
    review_payload = trainer._canonical_json_bytes(review) + b"\n"
    review_path.write_bytes(review_payload)
    review_sha256 = hashlib.sha256(review_payload).hexdigest()
    monkeypatch.setattr(trainer, "CANONICAL_REVIEW_RECORD_PATH", review_path)
    strict_loader = trainer._strict_hashed_object

    def synthetic_review_loader(
        path: Path,
        expected_file_sha256: str,
        *,
        name: str,
        allowed_root: Path | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        return strict_loader(
            path,
            expected_file_sha256,
            name=name,
            allowed_root=tmp_path if path == review_path else allowed_root,
        )

    monkeypatch.setattr(trainer, "_strict_hashed_object", synthetic_review_loader)
    authorization["status"] = "independent_review_passed_authorized"
    authorization["authorization"]["development_fit"] = True
    authorization["authorization"][
        "development_checkpoint_creation_authorized"
    ] = True
    authorization["review_record"] = {
        "path": str(review_path),
        "file_sha256": review_sha256,
        "content_sha256": review["content_sha256"],
        "status": "independent_review_passed",
    }
    core = dict(authorization)
    core.pop("content_sha256")
    authorization["content_sha256"] = trainer.canonical_json_sha256(core)
    sources = trainer._validate_trainer_authorization(
        authorization,
        dataset_manifest_file_sha256=trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256,
        dataset_manifest_content_sha256=trainer.preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
        audit_receipt_file_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_FILE_SHA256,
        audit_receipt_content_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
        review_record_file_sha256=review_sha256,
    )
    assert len(sources) == len(trainer.preauth_launcher.REQUIRED_SOURCE_ROLES)

    mismatched = copy.deepcopy(authorization)
    mismatched["dataset_binding"]["file_sha256"] = "9" * 64
    with pytest.raises(PermissionError, match="bindings"):
        trainer._validate_trainer_authorization(
            mismatched,
            dataset_manifest_file_sha256=trainer.preauth_launcher.DATASET_MANIFEST_FILE_SHA256,
            dataset_manifest_content_sha256=trainer.preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
            audit_receipt_file_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_FILE_SHA256,
            audit_receipt_content_sha256=trainer.preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
            review_record_file_sha256=review_sha256,
        )

    incomplete = copy.deepcopy(authorization)
    incomplete["source_map"]["entries"].pop()
    incomplete["source_map"]["entry_count"] -= 1
    incomplete["source_map"]["source_map_sha256"] = trainer.canonical_json_sha256(
        incomplete["source_map"]["entries"]
    )
    with pytest.raises(PermissionError, match="source closure changed"):
        trainer._validate_trainer_authorization_snapshot_schema(incomplete)
