from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import dev_frozen_dense_representation_encoders_v1 as incumbent
from scripts import vjepa2_1_vitg_frozen_encoder_ablation_v1 as ablation


def test_direct_cli_resolves_repository_imports(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(ablation.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-smoke" in result.stdout


def test_frozen_scale_only_contract_and_namespace() -> None:
    contract = ablation.encoder_contract_v1()
    assert contract["classification"] == "SCALE_ONLY"
    assert contract["source_repository_commit"] == (
        "204698b45b3712590f06245fbfba32d3be539812"
    )
    assert contract["official_source_license"] == ablation.VJEPA_LICENSE
    assert contract["official_source_license"]["spdx_identifier"] == "MIT"
    assert (
        contract["official_source_license"]["checkpoint_separate_license_statement"]
        is None
    )
    assert contract["constructor"] == "vjepa2_1_vit_giant_384"
    assert contract["checkpoint_state_key"] == "target_encoder"
    assert contract["checkpoint_byte_count"] == 16_878_318_788
    assert contract["architecture"] == {
        "width": 1408,
        "depth": 40,
        "heads": 22,
        "patch_size": 16,
        "video_tubelet_size": 2,
        "image_tokenizer_tubelet_size": 1,
        "attention": "torch.nn.functional.scaled_dot_product_attention",
        "positional_encoding": "RoPE with nonsquare interpolation",
    }
    assert contract["output"]["shape"] == ["batch", 768, 1408]
    assert contract["execution"]["probe_batch_sizes"] == [1, 2, 4]
    assert contract["execution"]["allowed_inference_dtypes"] == [
        "bfloat16",
        "float32",
    ]
    assert contract["execution"]["parameter_dtype"] == "torch.float32"
    assert contract["execution"]["input_dtype"] == "torch.float32"
    assert "autocast" in contract["execution"]["bfloat16_mode"]
    assert (
        contract["execution"]["bfloat16_mode_dense_output_dtype"]
        == "torch.float32"
    )
    assert "EMA/teacher" in contract["checkpoint_state_semantics"]
    assert ablation.EXPECTED_PARAMETER_COUNT == 1_013_267_968
    assert ablation.DEFAULT_OUTPUT_ROOT == ablation.REPO_ROOT / (
        ".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1"
    )
    assert (
        ablation.RESOURCE_SMOKE_RECEIPT_PATH.name
        == "resource_smoke_receipt.json"
    )
    assert ablation.ENCODER_CONTRACT_DIGEST == ablation.canonical_digest_v1(
        contract
    )


def test_current_encoder_projection_is_exact_scale_predecessor() -> None:
    observed = ablation.verify_current_encoder_v1(verify_checkpoint=False)
    assert observed["classification"] == "SCALE_ONLY"
    assert observed["constructor"] == "vjepa2_1_vit_large_384"
    assert observed["checkpoint_state_key"] == "ema_encoder"
    assert observed["architecture"] == {
        "width": 1024,
        "depth": 24,
        "heads": 16,
        "patch_size": 16,
        "image_tokenizer_tubelet_size": 1,
    }
    assert observed["output_shape"] == ["batch", 768, 1024]
    assert observed["change_to_registered_ablation"]["recipe_changed"] is False


def test_current_checkpoint_inspection_requires_intended_ema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "current.pt"
    checkpoint.write_bytes(b"current")
    monkeypatch.setattr(ablation, "CURRENT_CHECKPOINT", checkpoint)
    monkeypatch.setattr(
        ablation,
        "file_binding_v1",
        lambda _path: {
            "path": str(checkpoint.resolve()),
            "sha256": ablation.CURRENT_CHECKPOINT_SHA256,
            "byte_count": ablation.CURRENT_CHECKPOINT_BYTE_COUNT,
        },
    )
    keys = [f"k{index:03d}" for index in range(302)]
    ema = {key: torch.zeros(1, dtype=torch.float32) for key in keys}
    ema[keys[-1]] = torch.zeros(1).as_strided(
        (304_680_960 - 301,), (0,)
    )
    online = dict(ema)
    online[keys[0]] = torch.ones(1, dtype=torch.float32)
    payload = {
        "epoch": 40,
        "loss": 0.5,
        "ema_encoder": ema,
        "encoder": online,
    }

    def fake_load(
        path: Path, *, map_location: str, weights_only: bool, mmap: bool
    ) -> dict[str, Any]:
        assert path == checkpoint
        assert (map_location, weights_only, mmap) == ("cpu", False, True)
        return payload

    monkeypatch.setattr(torch, "load", fake_load)
    observed = ablation.verify_current_encoder_v1()
    inspection = observed["checkpoint_inspection"]
    assert inspection["ema_tensor_count"] == 302
    assert inspection["ema_value_count"] == 304_680_960
    assert inspection["ema_dtypes"] == ["torch.float32"]
    assert inspection["online_differs_from_ema"] is True
    assert inspection["first_differing_key"] == keys[0]


def test_preprocess_is_identical_to_existing_v03_crop(
    tmp_path: Path,
) -> None:
    y, x = np.indices((224, 224))
    array = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
    path = tmp_path / "frame.png"
    image = Image.fromarray(array)
    image.save(path)
    expected = incumbent.preprocess_vjepa_v03_crop(str(path))
    actual = ablation.preprocess_v03_image_v1(image)
    assert actual.shape == (3, 384, 512)
    assert actual.dtype is torch.float32
    assert torch.equal(actual, expected)
    with pytest.raises(ablation.VJepaVitGAblationError, match="224x224"):
        ablation.preprocess_v03_image_v1(Image.new("RGB", (10, 10)))


def test_official_source_commit_and_transitive_bindings_are_exact() -> None:
    observed = ablation.validate_official_source_v1()
    assert observed["commit"] == ablation.VJEPA_REPOSITORY_COMMIT
    assert set(observed["files"]) == set(ablation.SOURCE_BINDINGS)
    for relative, (sha256, byte_count) in ablation.SOURCE_BINDINGS.items():
        assert observed["files"][relative]["sha256"] == sha256
        assert observed["files"][relative]["byte_count"] == byte_count
    assert observed["license"]["identity"] == "MIT"
    assert observed["license"]["sha256"] == ablation.VJEPA_LICENSE["sha256"]
    assert observed["license"]["byte_count"] == 1_087


def test_process_preflight_reads_only_bounded_standard_metadata() -> None:
    gpu = ablation._gpu_process_inventory_v1()  # noqa: SLF001
    top = ablation._top_system_memory_consumers_v1()  # noqa: SLF001
    assert gpu["cmdline_or_environment_read"] is False
    assert gpu["fields_read"] == ["pid", "comm", "VmRSS"]
    assert gpu["process_count"] == len(gpu["processes"])
    assert top["cmdline_or_environment_read"] is False
    assert top["fields_read"] == ["pid", "comm", "VmRSS"]
    assert len(top["processes"]) <= 10
    assert [item["rss_bytes"] for item in top["processes"]] == sorted(
        [item["rss_bytes"] for item in top["processes"]], reverse=True
    )
    assert all(
        set(item) == {"pid", "comm", "rss_bytes"}
        for inventory in (gpu["processes"], top["processes"])
        for item in inventory
    )


def test_checkpoint_validation_is_exact_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "vjepa2_1_vitg_384.pt"
    checkpoint.write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    monkeypatch.setattr(ablation, "VJEPA_CHECKPOINT", checkpoint)
    monkeypatch.setattr(
        ablation,
        "file_binding_v1",
        lambda path: {
            "path": str(path.resolve()),
            "sha256": digest,
            "byte_count": ablation.EXPECTED_CHECKPOINT_BYTE_COUNT,
        },
    )
    assert ablation.validate_checkpoint_v1(digest)["sha256"] == digest
    with pytest.raises(ablation.VJepaVitGAblationError, match="digest or byte"):
        ablation.validate_checkpoint_v1("0" * 64)
    with pytest.raises(ablation.VJepaVitGAblationError, match="hexadecimal"):
        ablation.validate_checkpoint_v1("z" * 64)


class _FakeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = SimpleNamespace(use_sdpa=True)


class _FakeOfficialEncoder(torch.nn.Module):
    embed_dim = 1408
    num_heads = 22
    patch_size = 16
    tubelet_size = 2
    img_temporal_dim_size = 1
    return_hierarchical = False

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeBlock() for _ in range(40)])
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(kernel_size=(2, 16, 16))
        )
        self.patch_embed_img = SimpleNamespace(
            proj=SimpleNamespace(kernel_size=(1, 16, 16))
        )


def test_exact_official_constructor_is_used_without_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ablation, "validate_official_source_v1", lambda: {})
    predictor_module = SimpleNamespace(vit_predictor=lambda **_kwargs: pytest.fail())
    encoder = _FakeOfficialEncoder()

    def constructor(*, pretrained: bool) -> tuple[torch.nn.Module, object]:
        assert pretrained is False
        return encoder, predictor_module.vit_predictor(embed_dim=1408)

    backbones = SimpleNamespace(
        __file__=str(ablation.VJEPA_REPOSITORY / "src/hub/backbones.py"),
        ARCH_NAME_MAP={
            ablation.CONSTRUCTOR: (
                ablation.ARCH_NAME,
                "vjepa2_1_vitg_384",
            )
        },
        vjepa2_1_vit_giant_384=constructor,
    )

    def fake_import(name: str) -> object:
        if name == "src.hub.backbones":
            return backbones
        if name == "app.vjepa_2_1.models.predictor":
            return predictor_module
        raise AssertionError(name)

    monkeypatch.setattr(ablation.importlib, "import_module", fake_import)
    observed = ablation.construct_official_encoder_v1()
    assert observed is encoder
    assert predictor_module.vit_predictor.__name__ == "<lambda>"


class _LoadedEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.loaded: tuple[dict[str, torch.Tensor], bool] | None = None

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True):
        self.loaded = (state, strict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.arange(
            value.shape[0] * 768 * 1408,
            dtype=value.dtype,
            device=value.device,
        ).reshape(value.shape[0], 768, 1408)


def test_frozen_loader_uses_only_target_encoder_and_strict_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "vjepa2_1_vitg_384.pt"
    checkpoint.write_bytes(b"fixture")
    digest = "a" * 64
    binding = {
        "path": str(checkpoint.resolve()),
        "sha256": digest,
        "byte_count": ablation.EXPECTED_CHECKPOINT_BYTE_COUNT,
    }
    encoder = _LoadedEncoder()
    monkeypatch.setattr(ablation, "VJEPA_CHECKPOINT", checkpoint)
    monkeypatch.setattr(ablation, "validate_checkpoint_v1", lambda _digest: binding)
    monkeypatch.setattr(ablation, "construct_official_encoder_v1", lambda: encoder)
    opened: list[Path] = []

    def fake_load(
        path: Path, *, map_location: str, weights_only: bool
    ) -> dict[str, object]:
        opened.append(path)
        assert (map_location, weights_only) == ("cpu", False)
        return {
            "target_encoder": {"module.backbone.frozen": torch.ones(())},
            "predictor": pytest.fail,
        }

    monkeypatch.setattr(torch, "load", fake_load)
    arm = ablation.load_official_frozen_encoder_v1(
        device=torch.device("cpu"),
        dtype=torch.float32,
        expected_checkpoint_sha256=digest,
    )
    assert opened == [checkpoint]
    assert encoder.loaded is not None
    assert encoder.loaded[1] is True
    assert set(encoder.loaded[0]) == {"frozen"}
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert {parameter.dtype for parameter in encoder.parameters()} == {torch.float32}
    assert arm.checkpoint_binding == binding


def test_final_dense_tokens_keep_grid_and_token_normalisation() -> None:
    arm = ablation.VJepa21VitGFrozenEncoder()
    arm._module = _LoadedEncoder()
    batch = torch.zeros(1, 3, 384, 512)
    output = ablation.extract_final_dense_tokens_v1(arm, batch)
    assert output.shape == (1, 768, 1408)
    assert output.dtype is torch.float32
    assert torch.allclose(output.mean(dim=-1), torch.zeros(1, 768), atol=2e-6)
    with pytest.raises(ablation.VJepaVitGAblationError, match="expected"):
        arm.tokens(torch.zeros(1, 3, 384, 384))
    with pytest.raises(ablation.VJepaVitGAblationError, match="torch.float32"):
        arm.tokens(torch.zeros(1, 3, 384, 512, dtype=torch.bfloat16))


def test_bfloat16_mode_uses_cuda_autocast_with_fp32_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    class BF16Encoder(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert value.dtype is torch.float32
            return torch.zeros(value.shape[0], 768, 1408, dtype=torch.float32)

    class AutocastFixture:
        def __enter__(self) -> None:
            events.append(("enter", None))

        def __exit__(self, *_args: object) -> None:
            events.append(("exit", None))

    def fake_autocast(*, device_type: str, dtype: torch.dtype, enabled: bool):
        events.append((device_type, dtype))
        assert enabled is True
        return AutocastFixture()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    arm = ablation.VJepa21VitGFrozenEncoder()
    arm._module = BF16Encoder()
    arm.inference_dtype = torch.bfloat16
    arm.execution_mode = "bfloat16_autocast_fp32_weights"
    arm._device_type = "cuda"
    output = arm.tokens(torch.zeros(1, 3, 384, 512, dtype=torch.float32))
    assert output.dtype is torch.float32
    assert events == [
        ("cuda", torch.bfloat16),
        ("enter", None),
        ("exit", None),
    ]


def _write_frames(tmp_path: Path) -> list[tuple[str, Path]]:
    images = []
    for index, family in enumerate(("large", "large", "small", "small")):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (224, 224), (index, index + 1, index + 2)).save(path)
        images.append((family, path))
    return images


def test_smoke_images_are_four_distinct_existing_frames_from_multiple_families(
    tmp_path: Path,
) -> None:
    images = _write_frames(tmp_path)
    tensors, receipts = ablation.load_smoke_images_v1(images)
    assert len(tensors) == len(receipts) == 4
    assert [item["index"] for item in receipts] == [0, 1, 2, 3]
    assert {item["family"] for item in receipts} == {"large", "small"}
    assert all(tensor.shape == (3, 384, 512) for tensor in tensors)
    with pytest.raises(ablation.VJepaVitGAblationError, match="two families"):
        ablation.load_smoke_images_v1([("one", path) for _family, path in images])
    with pytest.raises(ablation.VJepaVitGAblationError, match="exactly 4"):
        ablation.load_smoke_images_v1(images[:3])


def _valid_receipt() -> dict[str, Any]:
    source_files = {
        relative: {
            "path": str((ablation.VJEPA_REPOSITORY / relative).resolve()),
            "sha256": sha256,
            "byte_count": byte_count,
        }
        for relative, (sha256, byte_count) in ablation.SOURCE_BINDINGS.items()
    }
    current = ablation.verify_current_encoder_v1(verify_checkpoint=False)
    current["checkpoint_binding"] = {
        "path": str(ablation.CURRENT_CHECKPOINT.resolve()),
        "sha256": ablation.CURRENT_CHECKPOINT_SHA256,
        "byte_count": ablation.CURRENT_CHECKPOINT_BYTE_COUNT,
    }
    current["checkpoint_inspection"] = {
        "ema_state_key": "ema_encoder",
        "ema_tensor_count": 302,
        "ema_value_count": 304_680_960,
        "ema_dtypes": ["torch.float32"],
        "online_differs_from_ema": True,
    }
    probes = []
    for batch_size in ablation.PROBE_BATCH_SIZES:
        probes.append(
            {
                "batch_size": batch_size,
                "status": "PASS",
                "forward_count": 3,
                "input_shape": [batch_size, 3, 384, 512],
                "input_dtype": "torch.float32",
                "output_shape": [batch_size, 768, 1408],
                "output_dtype": "torch.float32",
                "output_finite": True,
                "deterministic_repeat_max_abs_diff": 0.0,
                "warm_wall_seconds": 1.0,
                "warm_frames_per_second": float(batch_size),
                "sdpa_backend_evidence": {
                    "fused_backend_observed": True,
                    "fused_events": [
                        {
                            "key": "aten::_scaled_dot_product_flash_attention",
                            "count": 40,
                        }
                    ],
                },
                "gpu_memory": {
                    "peak_allocated_bytes": ablation.MAX_PEAK_VRAM_BYTES
                },
                "process_memory": {
                    "rusage_peak_rss_bytes": ablation.MAX_PROCESS_OR_SYSTEM_RAM_BYTES
                    - 1
                },
                "system_used_ram_bytes": ablation.MAX_PROCESS_OR_SYSTEM_RAM_BYTES
                - 1,
            }
        )
    value: dict[str, Any] = {
        "schema": ablation.RECEIPT_SCHEMA,
        "status": ablation.RECEIPT_STATUS_PASS,
        "development_only": True,
        "claim_bearing": False,
        "current_encoder_verification": current,
        "encoder_contract": ablation.encoder_contract_v1(),
        "encoder_contract_digest": ablation.ENCODER_CONTRACT_DIGEST,
        "source_binding": {
            "commit": ablation.VJEPA_REPOSITORY_COMMIT,
            "files": source_files,
            "license": {
                "path": str((ablation.VJEPA_REPOSITORY / "LICENSE").resolve()),
                "sha256": ablation.VJEPA_LICENSE["sha256"],
                "byte_count": ablation.VJEPA_LICENSE["byte_count"],
                "identity": "MIT",
                "spdx_identifier": "MIT",
                "checkpoint_separate_license_statement": None,
            },
        },
        "checkpoint_binding": {
            "path": str(ablation.VJEPA_CHECKPOINT.resolve()),
            "sha256": "a" * 64,
            "byte_count": ablation.EXPECTED_CHECKPOINT_BYTE_COUNT,
        },
        "checkpoint_state_key_opened": "target_encoder",
        "predictor_constructed": False,
        "predictor_checkpoint_state_access_count": 0,
        "scientific_labels_opened": 0,
        "corpus_frames_opened": 4,
        "inference_dtype": "torch.bfloat16",
        "execution_mode": "bfloat16_autocast_fp32_weights",
        "parameter_dtype": "torch.float32",
        "smoke_images": [
            {"index": index, "family": "large" if index < 2 else "small"}
            for index in range(4)
        ],
        "smoke_family_count": 2,
        "preflight": {
            "passes": True,
            "failures": [],
            "gpu": {"torch_hip_version": "6.4.43482"},
            "gpu_process_inventory_before_load": {
                "process_count": 0,
                "processes": [],
                "fields_read": ["pid", "comm", "VmRSS"],
                "cmdline_or_environment_read": False,
            },
            "top_system_memory_consumers_before_load": {
                "limit": 10,
                "processes": [
                    {"pid": 1, "comm": "fixture", "rss_bytes": 1024}
                ],
                "fields_read": ["pid", "comm", "VmRSS"],
                "cmdline_or_environment_read": False,
            },
            "thresholds": {
                "minimum_free_vram_bytes": ablation.MIN_FREE_VRAM_BYTES,
                "minimum_available_host_ram_bytes": ablation.MIN_AVAILABLE_RAM_BYTES,
                "minimum_destination_free_bytes": ablation.MIN_DESTINATION_FREE_BYTES,
            },
        },
        "parameter_count": ablation.EXPECTED_PARAMETER_COUNT,
        "device": {"torch_hip_version": "6.4.43482"},
        "probes": probes,
        "maximum_passing_batch_size": 4,
        "all_registered_batches_pass": True,
    }
    value["receipt_sha256"] = ablation.canonical_digest_v1(value)
    return value


def test_resource_receipt_validation_binds_smoke_sdpa_and_limits() -> None:
    receipt = _valid_receipt()
    assert (
        ablation.validate_resource_smoke_receipt_v1(
            receipt, expected_checkpoint_sha256="a" * 64
        )
        == receipt
    )
    for mutation in ("checkpoint", "determinism", "sdpa", "memory"):
        changed = deepcopy(receipt)
        if mutation == "checkpoint":
            changed["checkpoint_binding"]["byte_count"] -= 1
        elif mutation == "determinism":
            changed["probes"][0]["deterministic_repeat_max_abs_diff"] = 1e-5
        elif mutation == "sdpa":
            changed["probes"][0]["sdpa_backend_evidence"][
                "fused_backend_observed"
            ] = False
        else:
            changed["probes"][0]["gpu_memory"][
                "peak_allocated_bytes"
            ] = ablation.MAX_PEAK_VRAM_BYTES + 1
        changed["receipt_sha256"] = ablation.canonical_digest_v1(
            {key: value for key, value in changed.items() if key != "receipt_sha256"}
        )
        with pytest.raises(ablation.VJepaVitGAblationError):
            ablation.validate_resource_smoke_receipt_v1(
                changed, expected_checkpoint_sha256="a" * 64
            )


def test_resource_receipt_accepts_optional_batch_stop() -> None:
    receipt = _valid_receipt()
    receipt["probes"][1] = {
        "batch_size": 2,
        "status": "OPTIONAL_OUT_OF_MEMORY",
        "forward_count": 0,
    }
    receipt["probes"][2] = {
        "batch_size": 4,
        "status": "NOT_ATTEMPTED_AFTER_OPTIONAL_STOP",
        "forward_count": 0,
    }
    receipt["maximum_passing_batch_size"] = 1
    receipt["all_registered_batches_pass"] = False
    receipt["receipt_sha256"] = ablation.canonical_digest_v1(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    observed = ablation.validate_resource_smoke_receipt_v1(
        receipt, expected_checkpoint_sha256="a" * 64
    )
    assert observed["maximum_passing_batch_size"] == 1


def test_receipt_write_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"schema": "fixture"}
    ablation._write_receipt_exclusive_v1(path, receipt)  # noqa: SLF001
    with pytest.raises(FileExistsError):
        ablation._write_receipt_exclusive_v1(path, receipt)  # noqa: SLF001
