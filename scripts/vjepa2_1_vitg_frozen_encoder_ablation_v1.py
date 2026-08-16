#!/usr/bin/env python3
"""Exact-path resource smoke for the official frozen V-JEPA 2.1 ViT-g.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  This is a scale-only encoder ablation:
the established v03 crop, image tokenizer, dense final-token extraction and
token-wise normalisation stay fixed.  The script never downloads a checkpoint,
constructs a predictor, trains a model, or encodes a corpus.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import dev_frozen_dense_representation_encoders_v1 as frozen_encoders


VJEPA_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_REPOSITORY = Path.home() / f".cache/vjepa2-{VJEPA_REPOSITORY_COMMIT}"
VJEPA_CHECKPOINT = Path.home() / ".cache/vjepa2_1_vitg_384.pt"
VJEPA_SOURCE_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt"
VJEPA_LICENSE = {
    "identity": "MIT",
    "spdx_identifier": "MIT",
    "repository_relative_path": "LICENSE",
    "path": str(VJEPA_REPOSITORY / "LICENSE"),
    "sha256": "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173",
    "byte_count": 1_087,
    "checkpoint_separate_license_statement": None,
    "checkpoint_license_note": (
        "the official release states no separate checkpoint license"
    ),
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".generated/go2_scorer_fit_vjepa2_1_vitg_ablation_v1"
RESOURCE_SMOKE_RECEIPT_PATH = DEFAULT_OUTPUT_ROOT / "resource_smoke_receipt.json"

CONSTRUCTOR = "vjepa2_1_vit_giant_384"
CHECKPOINT_STATE_KEY = "target_encoder"
ARCH_NAME = "vit_giant_xformers"
WIDTH = 1408
DEPTH = 40
HEADS = 22
PATCH_SIZE = 16
VIDEO_TUBELET_SIZE = 2
IMAGE_TUBELET_SIZE = 1
INPUT_HW = (384, 512)
TOKEN_GRID = (24, 32)
TOKEN_COUNT = 768
TOKEN_DIM = WIDTH
PROBE_BATCH_SIZES = (1, 2, 4)
ALLOWED_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
EXPECTED_CHECKPOINT_BYTE_COUNT = 16_878_318_788
EXPECTED_PARAMETER_COUNT = 1_013_267_968
SMOKE_IMAGE_COUNT = 4
GIB = 1 << 30
MIN_FREE_VRAM_BYTES = 26 * GIB
MIN_AVAILABLE_RAM_BYTES = 40 * GIB
MIN_DESTINATION_FREE_BYTES = 50 * GIB
MAX_PEAK_VRAM_BYTES = 28 * GIB
MAX_PROCESS_OR_SYSTEM_RAM_BYTES = 80 * GIB
TOP_MEMORY_PROCESS_LIMIT = 10

CURRENT_CONSTRUCTOR = "vjepa2_1_vit_large_384"
CURRENT_CHECKPOINT = Path.home() / ".cache/vjepa2_1_vitl_dist_vitG_384.pt"
CURRENT_CHECKPOINT_LOGICAL_PATH = str(CURRENT_CHECKPOINT)
CURRENT_CHECKPOINT_SHA256 = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
CURRENT_CHECKPOINT_BYTE_COUNT = 5_151_198_524
CURRENT_ENCODER_IDENTITY_DIGEST = "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"

RECEIPT_SCHEMA = "lewm_vjepa2_1_vitg_frozen_encoder_ablation_v1_resource_receipt_v1"
RECEIPT_STATUS_PASS = "PASS_EXACT_PATH_RESOURCE_SMOKE"

# Exact transitive source files reached by the official constructor/encoder.
SOURCE_BINDINGS = {
    "src/hub/backbones.py": ("391cdde1e9a1da47cb8094bbea5fbbe8acac0135b27e82f1a6ab19c0b39cc692", 10_164),
    "app/vjepa_2_1/models/vision_transformer.py": ("d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9", 18_195),
    "app/vjepa_2_1/models/predictor.py": ("30111720eb90c6dcdde44521cea53cc736020f984e5573150fd3dc7b4acc05d8", 10_679),
    "app/vjepa_2_1/models/utils/modules.py": ("64be6a87bd9f18d385f4e44186db3347d1665e18a1f0511d51d3b305531562e2", 16_963),
    "app/vjepa_2_1/models/utils/patch_embed.py": ("29e11ab97ab3ccdef107d6a7d0d7b374b58e712076cc3561f07b7e603c9b5165", 1_883),
    "src/masks/utils.py": ("833f111a0fa5ffdbd3a6412e2dace2517c3c178f49c14f8bb631d9f6a070dfd0", 660),
    "src/utils/tensors.py": ("782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8", 1_832),
}


class VJepaVitGAblationError(RuntimeError):
    """A frozen ablation or resource-smoke contract was violated."""


def canonical_bytes_v1(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_digest_v1(value: Any) -> str:
    return hashlib.sha256(canonical_bytes_v1(value)).hexdigest()


def file_binding_v1(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while block := handle.read(1 << 22):
            digest.update(block)
            byte_count += len(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def encoder_contract_v1() -> dict[str, Any]:
    return {
        "classification": "SCALE_ONLY",
        "family": "vjepa_video_ssl",
        "release": "V-JEPA 2.1 official release",
        "source_repository": "facebookresearch/vjepa2",
        "source_repository_commit": VJEPA_REPOSITORY_COMMIT,
        "official_source_license": dict(VJEPA_LICENSE),
        "constructor": CONSTRUCTOR,
        "constructor_architecture": ARCH_NAME,
        "checkpoint_source_url": VJEPA_SOURCE_URL,
        "checkpoint_path": str(VJEPA_CHECKPOINT),
        "checkpoint_state_key": CHECKPOINT_STATE_KEY,
        "checkpoint_state_semantics": (
            "official pretrained EMA/teacher target encoder; ViT-g uses the "
            "literal target_encoder key, distinct from the ViT-L checkpoint's "
            "literal ema_encoder key"
        ),
        "checkpoint_byte_count": EXPECTED_CHECKPOINT_BYTE_COUNT,
        "architecture": {
            "width": WIDTH,
            "depth": DEPTH,
            "heads": HEADS,
            "patch_size": PATCH_SIZE,
            "video_tubelet_size": VIDEO_TUBELET_SIZE,
            "image_tokenizer_tubelet_size": IMAGE_TUBELET_SIZE,
            "attention": "torch.nn.functional.scaled_dot_product_attention",
            "positional_encoding": "RoPE with nonsquare interpolation",
        },
        "preprocessing": {
            "implementation": (
                "scripts.dev_frozen_dense_representation_encoders_v1."
                "preprocess_vjepa_v03_crop"
            ),
            "source_frame_hw": [224, 224],
            "crop_xyxy": [0, 28, 224, 196],
            "cropped_hw": [168, 224],
            "input_hw": list(INPUT_HW),
            "resample": "PIL BICUBIC",
            "normalisation": {
                "mean": list(frozen_encoders.IMAGENET_MEAN),
                "std": list(frozen_encoders.IMAGENET_STD),
            },
            "padding": "none",
        },
        "output": {
            "source_layer": "encoder final block, norms_block[-1]",
            "hierarchical_outputs": False,
            "pooling": "none",
            "token_grid_hw": list(TOKEN_GRID),
            "shape": ["batch", TOKEN_COUNT, TOKEN_DIM],
            "token_order": "row-major 24x32 image-patch grid",
            "post_normalisation": "F.layer_norm over token dimension",
        },
        "execution": {
            "frozen": True,
            "eval": True,
            "gradient": False,
            "predictor_constructed": False,
            "pretrained_constructor_download": False,
            "allowed_inference_dtypes": sorted(ALLOWED_DTYPES),
            "parameter_dtype": "torch.float32",
            "input_dtype": "torch.float32",
            "bfloat16_mode": "torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
            "bfloat16_mode_dense_output_dtype": "torch.float32",
            "float32_mode": "no autocast",
            "probe_batch_sizes": list(PROBE_BATCH_SIZES),
            "probe_forward_count_per_batch": 3,
            "probe_forward_roles": ["profile", "warm", "deterministic_repeat"],
            "smoke_image_count": SMOKE_IMAGE_COUNT,
            "smoke_minimum_family_count": 2,
            "preflight_minimums_bytes": {
                "free_vram": MIN_FREE_VRAM_BYTES,
                "available_host_ram": MIN_AVAILABLE_RAM_BYTES,
                "destination_free": MIN_DESTINATION_FREE_BYTES,
            },
            "probe_maximums_bytes": {
                "peak_vram": MAX_PEAK_VRAM_BYTES,
                "process_or_system_ram": MAX_PROCESS_OR_SYSTEM_RAM_BYTES,
            },
            "batch_1_mandatory": True,
            "batch_2_and_4_optional_stop_on_resource_failure": True,
        },
    }


ENCODER_CONTRACT_DIGEST = canonical_digest_v1(encoder_contract_v1())


def preprocess_v03_image_v1(image: Image.Image) -> torch.Tensor:
    """The existing v03 centre-crop and V-JEPA 2.1 image preprocessing."""

    converted = image.convert("RGB")
    if converted.size != (224, 224):
        raise VJepaVitGAblationError(
            f"expected one 224x224 v03 RGB frame, got {converted.size}"
        )
    cropped = converted.crop((0, 28, 224, 196)).resize(
        (512, 384), Image.Resampling.BICUBIC
    )
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(frozen_encoders.IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(frozen_encoders.IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def verify_current_encoder_v1(*, verify_checkpoint: bool = True) -> dict[str, Any]:
    """Verify the frozen ViT-L predecessor and classify the change as scale-only."""

    from lewm.oracle import go2_scorer_contract_v1_2 as current

    expected = {
        "constructor": CURRENT_CONSTRUCTOR,
        "checkpoint": CURRENT_CHECKPOINT_LOGICAL_PATH,
        "checkpoint_sha256": CURRENT_CHECKPOINT_SHA256,
        "checkpoint_byte_count": CURRENT_CHECKPOINT_BYTE_COUNT,
        "source_repository": "facebookresearch/vjepa2",
        "source_repository_commit": VJEPA_REPOSITORY_COMMIT,
        "token_grid": [24, 32],
        "tokens": 768,
        "token_dim": 1024,
    }
    for key, value in expected.items():
        if current.TARGET_ENCODER.get(key) != value:
            raise VJepaVitGAblationError(
                f"current frozen target encoder {key} binding changed"
            )
    checkpoint_binding: dict[str, Any] | None = None
    checkpoint_inspection: dict[str, Any] | None = None
    if verify_checkpoint:
        if not CURRENT_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"current frozen target encoder checkpoint is missing: {CURRENT_CHECKPOINT}"
            )
        checkpoint_binding = file_binding_v1(CURRENT_CHECKPOINT)
        if (
            checkpoint_binding["sha256"] != CURRENT_CHECKPOINT_SHA256
            or checkpoint_binding["byte_count"] != CURRENT_CHECKPOINT_BYTE_COUNT
        ):
            raise VJepaVitGAblationError(
                "current frozen target encoder checkpoint binding changed"
            )
        payload = torch.load(
            CURRENT_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("ema_encoder"), Mapping
        ):
            raise VJepaVitGAblationError(
                "current checkpoint has no intended ema_encoder state"
            )
        ema_state = payload["ema_encoder"]
        ema_tensors = {
            str(key): tensor
            for key, tensor in ema_state.items()
            if isinstance(tensor, torch.Tensor)
        }
        tensor_count = len(ema_tensors)
        value_count = sum(tensor.numel() for tensor in ema_tensors.values())
        dtypes = sorted({str(tensor.dtype) for tensor in ema_tensors.values()})
        if tensor_count != 302 or value_count != 304_680_960 or dtypes != ["torch.float32"]:
            raise VJepaVitGAblationError(
                "current ema_encoder tensor inventory changed"
            )
        online = payload.get("encoder")
        if not isinstance(online, Mapping) or set(online) != set(ema_state):
            raise VJepaVitGAblationError(
                "current online/EMA encoder state inventories changed"
            )
        differing_key: str | None = None
        compared = 0
        for key in sorted(ema_state, key=str):
            compared += 1
            if not torch.equal(ema_state[key], online[key]):
                differing_key = str(key)
                break
        if differing_key is None:
            raise VJepaVitGAblationError(
                "current checkpoint EMA and online encoder unexpectedly match"
            )
        metadata = {
            key: payload[key]
            for key in ("epoch", "loss", "batch_size", "world_size", "lr")
            if isinstance(payload.get(key), (str, int, float, bool))
        }
        checkpoint_inspection = {
            "load_mode": "torch.load_cpu_mmap_true_weights_only_false",
            "top_level_keys": sorted(str(key) for key in payload),
            "metadata": metadata,
            "ema_state_key": "ema_encoder",
            "ema_tensor_count": tensor_count,
            "ema_value_count": value_count,
            "ema_dtypes": dtypes,
            "online_state_key": "encoder",
            "online_key_set_equals_ema": True,
            "online_differs_from_ema": True,
            "first_differing_key": differing_key,
            "keys_compared_until_first_difference": compared,
        }
        del payload, ema_state, ema_tensors, online
    return {
        "classification": "SCALE_ONLY",
        "family": "V-JEPA 2.1",
        "constructor": CURRENT_CONSTRUCTOR,
        "checkpoint_state_key": "ema_encoder",
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_inspection": checkpoint_inspection,
        "source_repository": "facebookresearch/vjepa2",
        "source_repository_commit": VJEPA_REPOSITORY_COMMIT,
        "architecture": {
            "width": 1024,
            "depth": 24,
            "heads": 16,
            "patch_size": 16,
            "image_tokenizer_tubelet_size": 1,
        },
        "output_shape": ["batch", 768, 1024],
        "target_encoder_identity_digest": CURRENT_ENCODER_IDENTITY_DIGEST,
        "change_to_registered_ablation": {
            "recipe_changed": False,
            "width": [1024, WIDTH],
            "depth": [24, DEPTH],
            "heads": [16, HEADS],
            "output_shape": [["batch", 768, 1024], ["batch", TOKEN_COUNT, TOKEN_DIM]],
        },
    }


def validate_official_source_v1() -> dict[str, Any]:
    if not VJEPA_REPOSITORY.is_dir():
        raise VJepaVitGAblationError(
            f"missing pinned official source repository: {VJEPA_REPOSITORY}"
        )
    observed_commit = subprocess.run(
        ["git", "-C", str(VJEPA_REPOSITORY), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != VJEPA_REPOSITORY_COMMIT:
        raise VJepaVitGAblationError(
            f"official source commit changed: {observed_commit}"
        )
    observed: dict[str, Any] = {}
    for relative, (expected_sha256, expected_bytes) in SOURCE_BINDINGS.items():
        binding = file_binding_v1(VJEPA_REPOSITORY / relative)
        if (
            binding["sha256"] != expected_sha256
            or binding["byte_count"] != expected_bytes
        ):
            raise VJepaVitGAblationError(
                f"official source binding changed: {relative}"
            )
        observed[relative] = binding
    license_binding = file_binding_v1(VJEPA_REPOSITORY / "LICENSE")
    if (
        license_binding["sha256"] != VJEPA_LICENSE["sha256"]
        or license_binding["byte_count"] != VJEPA_LICENSE["byte_count"]
    ):
        raise VJepaVitGAblationError("official MIT license binding changed")
    return {
        "commit": observed_commit,
        "files": observed,
        "license": {
            **license_binding,
            "identity": "MIT",
            "spdx_identifier": "MIT",
            "checkpoint_separate_license_statement": None,
        },
    }


def validate_checkpoint_v1(expected_checkpoint_sha256: str) -> dict[str, Any]:
    """Bind the exact local official checkpoint without downloading it."""

    if (
        len(expected_checkpoint_sha256) != 64
        or expected_checkpoint_sha256.lower() != expected_checkpoint_sha256
    ):
        raise VJepaVitGAblationError(
            "expected checkpoint SHA-256 must be 64 lowercase hex"
        )
    try:
        int(expected_checkpoint_sha256, 16)
    except ValueError as error:
        raise VJepaVitGAblationError(
            "expected checkpoint SHA-256 is not hexadecimal"
        ) from error
    if not VJEPA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "missing official ViT-g checkpoint; download is intentionally disabled: "
            f"{VJEPA_CHECKPOINT}"
        )
    binding = file_binding_v1(VJEPA_CHECKPOINT)
    if (
        binding["sha256"] != expected_checkpoint_sha256
        or binding["byte_count"] != EXPECTED_CHECKPOINT_BYTE_COUNT
    ):
        raise VJepaVitGAblationError(
            "official ViT-g checkpoint digest or byte count mismatch"
        )
    return binding


@contextmanager
def _without_predictor_construction_v1() -> Iterator[list[dict[str, Any]]]:
    """Run the official encoder constructor while replacing its unused predictor."""

    predictor_module = importlib.import_module("app.vjepa_2_1.models.predictor")
    original = predictor_module.vit_predictor
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def predictor_not_constructed(**kwargs: Any) -> object:
        calls.append(dict(kwargs))
        return sentinel

    predictor_module.vit_predictor = predictor_not_constructed
    try:
        yield calls
    finally:
        predictor_module.vit_predictor = original


def _validate_encoder_architecture_v1(encoder: torch.nn.Module) -> None:
    checks = {
        "embed_dim": (getattr(encoder, "embed_dim", None), WIDTH),
        "num_heads": (getattr(encoder, "num_heads", None), HEADS),
        "block_count": (len(getattr(encoder, "blocks", ())), DEPTH),
        "patch_size": (getattr(encoder, "patch_size", None), PATCH_SIZE),
        "tubelet_size": (
            getattr(encoder, "tubelet_size", None),
            VIDEO_TUBELET_SIZE,
        ),
        "img_temporal_dim_size": (
            getattr(encoder, "img_temporal_dim_size", None),
            IMAGE_TUBELET_SIZE,
        ),
        "return_hierarchical": (
            getattr(encoder, "return_hierarchical", None),
            False,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise VJepaVitGAblationError(
                f"official ViT-g {label} changed: {observed!r} != {expected!r}"
            )
    blocks = tuple(encoder.blocks)
    if not blocks or not all(getattr(block.attn, "use_sdpa", False) for block in blocks):
        raise VJepaVitGAblationError("official ViT-g SDPA path is not active")
    if tuple(encoder.patch_embed.proj.kernel_size) != (2, 16, 16):
        raise VJepaVitGAblationError("official video tokenizer changed")
    if tuple(encoder.patch_embed_img.proj.kernel_size) != (1, 16, 16):
        raise VJepaVitGAblationError("official image tokenizer changed")


def construct_official_encoder_v1() -> torch.nn.Module:
    """Construct only the encoder through the exact official hub entry point."""

    validate_official_source_v1()
    source = str(VJEPA_REPOSITORY)
    if source not in sys.path:
        sys.path.insert(0, source)
    with frozen_encoders.scoped_timm_drop_path_shim_v1():
        backbones = importlib.import_module("src.hub.backbones")
        expected_origin = (VJEPA_REPOSITORY / "src/hub/backbones.py").resolve()
        if Path(backbones.__file__).resolve() != expected_origin:
            raise VJepaVitGAblationError("src.hub.backbones resolved outside the pin")
        if backbones.ARCH_NAME_MAP.get(CONSTRUCTOR) != (
            ARCH_NAME,
            "vjepa2_1_vitg_384",
        ):
            raise VJepaVitGAblationError("official ViT-g constructor mapping changed")
        with _without_predictor_construction_v1() as predictor_calls:
            encoder, predictor = getattr(backbones, CONSTRUCTOR)(pretrained=False)
    if len(predictor_calls) != 1 or predictor.__class__ is not object:
        raise VJepaVitGAblationError("predictor suppression did not follow one exact call")
    _validate_encoder_architecture_v1(encoder)
    return encoder


class VJepa21VitGFrozenEncoder:
    """Frozen ViT-g image-token encoder with the scale-ablation output contract."""

    def __init__(self, checkpoint: Path | None = None) -> None:
        self.checkpoint = VJEPA_CHECKPOINT if checkpoint is None else checkpoint
        self._module: torch.nn.Module | None = None
        self.checkpoint_binding: dict[str, Any] | None = None
        self.inference_dtype: torch.dtype | None = None
        self.execution_mode: str | None = None
        self._device_type: str | None = None

    def build(
        self,
        device: torch.device,
        dtype: torch.dtype,
        expected_checkpoint_sha256: str,
    ) -> torch.nn.Module:
        if self.checkpoint.resolve() != VJEPA_CHECKPOINT.resolve():
            raise VJepaVitGAblationError("only the exact official checkpoint path is allowed")
        if dtype not in ALLOWED_DTYPES.values():
            raise VJepaVitGAblationError(f"unsupported inference dtype: {dtype}")
        binding = validate_checkpoint_v1(expected_checkpoint_sha256)

        encoder = construct_official_encoder_v1()
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or CHECKPOINT_STATE_KEY not in checkpoint:
            raise VJepaVitGAblationError(
                f"checkpoint is missing exact state key {CHECKPOINT_STATE_KEY!r}"
            )
        raw_state = checkpoint[CHECKPOINT_STATE_KEY]
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise VJepaVitGAblationError("target_encoder state is empty or malformed")
        state = {
            str(key).replace("module.", "").replace("backbone.", ""): value
            for key, value in raw_state.items()
        }
        del checkpoint, raw_state
        encoder.load_state_dict(state, strict=True)
        del state
        encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise VJepaVitGAblationError("encoder did not remain frozen in eval mode")
        self._module = encoder
        self.checkpoint_binding = binding
        self.inference_dtype = dtype
        self.execution_mode = (
            "bfloat16_autocast_fp32_weights"
            if dtype is torch.bfloat16
            else "float32_no_autocast"
        )
        self._device_type = device.type
        return encoder

    @torch.inference_mode()
    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        if self._module is None:
            raise VJepaVitGAblationError("build the frozen encoder before inference")
        if batch.dtype is not torch.float32:
            raise VJepaVitGAblationError(
                f"preprocessed encoder inputs must remain torch.float32, got {batch.dtype}"
            )
        if tuple(batch.shape[1:]) != (3, *INPUT_HW):
            raise VJepaVitGAblationError(
                f"expected [B,3,{INPUT_HW[0]},{INPUT_HW[1]}], got {list(batch.shape)}"
            )
        if self.inference_dtype is torch.bfloat16:
            if self._device_type != "cuda":
                raise VJepaVitGAblationError(
                    "bfloat16 execution requires CUDA/ROCm autocast"
                )
            autocast = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            )
            # On the bound ROCm torch stack, the final FP32-parameter LayerNorm
            # promotes the dense result back to FP32 under BF16 autocast.
            expected_output_dtype = torch.float32
        else:
            autocast = nullcontext()
            expected_output_dtype = torch.float32
        with autocast:
            raw = self._module(batch.unsqueeze(2))
        expected = (batch.shape[0], TOKEN_COUNT, TOKEN_DIM)
        if tuple(raw.shape) != expected:
            raise VJepaVitGAblationError(
                f"official final dense-token shape changed: {list(raw.shape)}"
            )
        if raw.dtype != expected_output_dtype or not bool(torch.isfinite(raw).all()):
            raise VJepaVitGAblationError("encoder output dtype or finiteness changed")
        return F.layer_norm(raw, (TOKEN_DIM,))


def load_official_frozen_encoder_v1(
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_checkpoint_sha256: str,
) -> VJepa21VitGFrozenEncoder:
    """Load the one registered checkpoint into the one official frozen encoder."""

    arm = VJepa21VitGFrozenEncoder()
    arm.build(device, dtype, expected_checkpoint_sha256)
    return arm


def extract_final_dense_tokens_v1(
    arm: VJepa21VitGFrozenEncoder, batch: torch.Tensor
) -> torch.Tensor:
    """Extract only the final dense, token-wise-normalised image grid."""

    return arm.tokens(batch)


def _synchronise_v1(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_measurement_v1(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _process_memory_v1() -> dict[str, int]:
    values: dict[str, int] = {}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith(("VmRSS:", "VmPeak:", "VmHWM:")):
                label, raw, unit = line.split()
                if unit != "kB":
                    raise VJepaVitGAblationError("unexpected /proc memory unit")
                values[label.rstrip(":").lower() + "_bytes"] = int(raw) * 1024
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    values["rusage_peak_rss_bytes"] = peak_kib * 1024
    return values


def _process_metadata_v1(pid: int) -> dict[str, Any] | None:
    """Read only PID, comm and RSS; never a command line or environment."""

    root = Path("/proc") / str(pid)
    try:
        comm = (root / "comm").read_text(encoding="utf-8").strip()
        rss_bytes = 0
        for line in (root / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                _label, raw, unit = line.split()
                if unit != "kB":
                    raise VJepaVitGAblationError("unexpected process RSS unit")
                rss_bytes = int(raw) * 1024
                break
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return {"pid": pid, "comm": comm, "rss_bytes": rss_bytes}


def _gpu_process_inventory_v1() -> dict[str, Any]:
    """Snapshot current ROCm KFD processes from standard sysfs metadata."""

    source = Path("/sys/class/kfd/kfd/proc")
    processes: list[dict[str, Any]] = []
    if source.is_dir():
        for entry in sorted(source.iterdir(), key=lambda path: path.name):
            if not entry.name.isdigit():
                continue
            metadata = _process_metadata_v1(int(entry.name))
            if metadata is not None:
                processes.append(metadata)
    return {
        "source": str(source),
        "process_count": len(processes),
        "processes": processes,
        "fields_read": ["pid", "comm", "VmRSS"],
        "cmdline_or_environment_read": False,
    }


def _top_system_memory_consumers_v1() -> dict[str, Any]:
    """Return a bounded RSS ranking using only standard /proc metadata."""

    processes: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        metadata = _process_metadata_v1(int(entry.name))
        if metadata is not None:
            processes.append(metadata)
    processes.sort(key=lambda item: (-item["rss_bytes"], item["pid"]))
    selected = processes[:TOP_MEMORY_PROCESS_LIMIT]
    return {
        "source": "/proc/*/{comm,status}",
        "limit": TOP_MEMORY_PROCESS_LIMIT,
        "processes": selected,
        "fields_read": ["pid", "comm", "VmRSS"],
        "cmdline_or_environment_read": False,
    }


def _ram_observation_v1() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and fields[0] in {
            "MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"
        }:
            if len(fields) != 3:
                raise VJepaVitGAblationError("incomplete host RAM field")
            label, raw, unit = fields
            if unit != "kB":
                raise VJepaVitGAblationError("unexpected /proc meminfo unit")
            values[label.rstrip(":").lower() + "_bytes"] = int(raw) * 1024
    if set(values) != {
        "memtotal_bytes",
        "memavailable_bytes",
        "swaptotal_bytes",
        "swapfree_bytes",
    }:
        raise VJepaVitGAblationError("incomplete host RAM preflight")
    return values


def resource_preflight_v1(*, device: torch.device) -> dict[str, Any]:
    """Observe, but do not invent, the host/device feasibility resources."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise VJepaVitGAblationError("the resource smoke requires one CUDA/ROCm device")
    if not VJEPA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "missing official ViT-g checkpoint; download is intentionally disabled: "
            f"{VJEPA_CHECKPOINT}"
        )
    checkpoint_bytes = VJEPA_CHECKPOINT.stat().st_size
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTE_COUNT:
        raise VJepaVitGAblationError("official ViT-g checkpoint byte count mismatch")
    free_device, total_device = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    if not DEFAULT_OUTPUT_ROOT.is_symlink():
        raise VJepaVitGAblationError(
            "managed output root must be the provisioned exact symlink"
        )
    destination = DEFAULT_OUTPUT_ROOT.resolve(strict=True)
    if destination.is_relative_to(REPO_ROOT.resolve()):
        raise VJepaVitGAblationError("managed output root must resolve externally")
    disk = shutil.disk_usage(destination)
    host_ram = _ram_observation_v1()
    process = _process_memory_v1()
    system_used = host_ram["memtotal_bytes"] - host_ram["memavailable_bytes"]
    failures = []
    if not isinstance(torch.version.hip, str) or not torch.version.hip:
        failures.append("torch_rocm_hip_version_missing")
    if int(free_device) < MIN_FREE_VRAM_BYTES:
        failures.append("free_vram_below_26_gib")
    if host_ram["memavailable_bytes"] < MIN_AVAILABLE_RAM_BYTES:
        failures.append("available_host_ram_below_40_gib")
    if int(disk.free) < MIN_DESTINATION_FREE_BYTES:
        failures.append("destination_free_below_50_gib")
    observation = {
        "checkpoint": {
            "path": str(VJEPA_CHECKPOINT.resolve()),
            "observed_byte_count": checkpoint_bytes,
            "expected_byte_count": EXPECTED_CHECKPOINT_BYTE_COUNT,
        },
        "gpu": {
            "device": str(device),
            "name": properties.name,
            "torch_hip_version": torch.version.hip,
            "total_memory_bytes": int(total_device),
            "free_memory_bytes": int(free_device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        },
        "host_ram": host_ram,
        "disk": {
            "logical_path": str(DEFAULT_OUTPUT_ROOT),
            "observed_path": str(destination),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
        "process": process,
        "gpu_process_inventory_before_load": _gpu_process_inventory_v1(),
        "top_system_memory_consumers_before_load": (
            _top_system_memory_consumers_v1()
        ),
        "system_used_ram_bytes": system_used,
        "thresholds": {
            "minimum_free_vram_bytes": MIN_FREE_VRAM_BYTES,
            "minimum_available_host_ram_bytes": MIN_AVAILABLE_RAM_BYTES,
            "minimum_destination_free_bytes": MIN_DESTINATION_FREE_BYTES,
        },
        "passes": not failures,
        "failures": failures,
    }
    if failures:
        raise VJepaVitGAblationError(
            "resource preflight failed: " + ", ".join(failures)
        )
    return observation


def load_smoke_images_v1(
    images: Sequence[tuple[str, Path]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Open exactly four labelled-by-family RGB frames, never scientific labels."""

    if len(images) != SMOKE_IMAGE_COUNT:
        raise VJepaVitGAblationError(
            f"resource smoke requires exactly {SMOKE_IMAGE_COUNT} existing images"
        )
    families = [str(family).strip() for family, _path in images]
    if any(not family for family in families) or len(set(families)) < 2:
        raise VJepaVitGAblationError("resource smoke requires at least two families")
    resolved = [Path(path).resolve() for _family, path in images]
    if len(set(resolved)) != SMOKE_IMAGE_COUNT:
        raise VJepaVitGAblationError("resource smoke image paths must be distinct")

    prepared: list[torch.Tensor] = []
    receipts: list[dict[str, Any]] = []
    for index, ((family, _path), path) in enumerate(zip(images, resolved, strict=True)):
        forbidden = {
            part.lower()
            for part in path.parts
            if part.lower() == "sealed" or part.lower().startswith("sealed_")
        }
        if forbidden:
            raise VJepaVitGAblationError("sealed material is forbidden")
        if not path.is_file():
            raise FileNotFoundError(f"smoke RGB does not exist: {path}")
        raw = path.read_bytes()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            if decoded.size != (224, 224):
                raise VJepaVitGAblationError(
                    f"smoke RGB must be 224x224, got {decoded.size}: {path}"
                )
            rgb = decoded.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.uint8)
            prepared.append(preprocess_v03_image_v1(rgb))
        receipts.append(
            {
                "index": index,
                "family": str(family),
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "byte_count": len(raw),
                "source_size_wh": [224, 224],
                "source_mode_after_conversion": "RGB",
            }
        )
    return prepared, receipts


def _profile_sdpa_forward_v1(
    arm: VJepa21VitGFrozenEncoder,
    inputs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        output = arm.tokens(inputs)
        _synchronise_v1(device)
    events = [
        {"key": event.key, "count": int(event.count)}
        for event in profile.key_averages()
        if "scaled_dot_product" in event.key or "flash_attention" in event.key
    ]
    fused = [
        event
        for event in events
        if any(
            marker in event["key"]
            for marker in (
                "_scaled_dot_product_flash_attention",
                "_scaled_dot_product_efficient_attention",
                "_scaled_dot_product_cudnn_attention",
            )
        )
    ]
    if not fused:
        raise VJepaVitGAblationError(
            "profiler did not observe a fused SDPA backend"
        )
    return output, {
        "profiler_activities": ["CPU", "CUDA"],
        "matching_events": events,
        "fused_events": fused,
        "fused_backend_observed": True,
    }


def run_resource_smoke_v1(
    *,
    device: torch.device,
    dtype: torch.dtype,
    expected_checkpoint_sha256: str,
    images: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Warm, repeat and profile the exact path at each registered batch size."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise VJepaVitGAblationError("the resource smoke requires one CUDA/ROCm device")
    if dtype not in ALLOWED_DTYPES.values():
        raise VJepaVitGAblationError("resource smoke dtype must be bfloat16 or float32")
    if dtype is torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported"):
        if not torch.cuda.is_bf16_supported():
            raise VJepaVitGAblationError("selected device does not support bfloat16")

    preflight = resource_preflight_v1(device=device)
    current_encoder = verify_current_encoder_v1()
    source_started = time.perf_counter()
    source_binding = validate_official_source_v1()
    source_seconds = time.perf_counter() - source_started
    prepared, image_receipts = load_smoke_images_v1(images)
    load_started = time.perf_counter()
    arm = load_official_frozen_encoder_v1(
        device=device,
        dtype=dtype,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    assert arm._module is not None
    module = arm._module
    _synchronise_v1(device)
    load_seconds = time.perf_counter() - load_started

    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise VJepaVitGAblationError(
            f"official ViT-g parameter count changed: {parameter_count}"
        )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in module.parameters()
    )
    probes: list[dict[str, Any]] = []
    optional_stop_reason: str | None = None
    for batch_size in PROBE_BATCH_SIZES:
        if optional_stop_reason is not None:
            probes.append(
                {
                    "batch_size": batch_size,
                    "status": "NOT_ATTEMPTED_AFTER_OPTIONAL_STOP",
                    "reason": optional_stop_reason,
                    "forward_count": 0,
                }
            )
            continue
        try:
            inputs = torch.stack(prepared[:batch_size]).to(
                device=device, dtype=torch.float32
            )
            torch.cuda.reset_peak_memory_stats(device)
            _synchronise_v1(device)
            profiled_output, sdpa = _profile_sdpa_forward_v1(arm, inputs, device)

            warm_started = time.perf_counter()
            warm_output = arm.tokens(inputs)
            _synchronise_v1(device)
            warm_seconds = time.perf_counter() - warm_started
            repeat_started = time.perf_counter()
            repeat_output = arm.tokens(inputs)
            _synchronise_v1(device)
            repeat_seconds = time.perf_counter() - repeat_started
            deterministic_max_abs_diff = float(
                (warm_output.float() - repeat_output.float()).abs().max().item()
            )
            if deterministic_max_abs_diff != 0.0:
                raise VJepaVitGAblationError(
                    f"batch-{batch_size} repeat was not exactly deterministic"
                )
            gpu_memory = _gpu_measurement_v1(device)
            assert gpu_memory is not None
            process_memory = _process_memory_v1()
            host_ram = _ram_observation_v1()
            system_used_ram = (
                host_ram["memtotal_bytes"] - host_ram["memavailable_bytes"]
            )
            resource_failures = []
            if gpu_memory["peak_allocated_bytes"] > MAX_PEAK_VRAM_BYTES:
                resource_failures.append("peak_vram_above_28_gib")
            if (
                process_memory["rusage_peak_rss_bytes"]
                >= MAX_PROCESS_OR_SYSTEM_RAM_BYTES
            ):
                resource_failures.append("process_peak_rss_at_or_above_80_gib")
            if system_used_ram >= MAX_PROCESS_OR_SYSTEM_RAM_BYTES:
                resource_failures.append("system_used_ram_at_or_above_80_gib")
            status = (
                "PASS"
                if not resource_failures
                else "OPTIONAL_RESOURCE_LIMIT_EXCEEDED"
            )
            probe = {
                "batch_size": batch_size,
                "status": status,
                "forward_count": 3,
                "input_shape": list(inputs.shape),
                "input_dtype": str(inputs.dtype),
                "output_shape": list(warm_output.shape),
                "output_dtype": str(warm_output.dtype),
                "output_finite": bool(torch.isfinite(warm_output).all()),
                "profile_output_equal_to_warm": bool(
                    torch.equal(profiled_output, warm_output)
                ),
                "deterministic_repeat_max_abs_diff": deterministic_max_abs_diff,
                "warm_wall_seconds": warm_seconds,
                "warm_frames_per_second": batch_size / warm_seconds,
                "repeat_wall_seconds": repeat_seconds,
                "repeat_frames_per_second": batch_size / repeat_seconds,
                "sdpa_backend_evidence": sdpa,
                "gpu_memory": gpu_memory,
                "process_memory": process_memory,
                "system_used_ram_bytes": system_used_ram,
                "resource_failures": resource_failures,
            }
            probes.append(probe)
            if resource_failures:
                if batch_size == 1:
                    raise VJepaVitGAblationError(
                        "mandatory batch-1 probe exceeded resource limits: "
                        + ", ".join(resource_failures)
                    )
                optional_stop_reason = ",".join(resource_failures)
            del inputs, profiled_output, warm_output, repeat_output
        except torch.OutOfMemoryError as error:
            if batch_size == 1:
                raise VJepaVitGAblationError(
                    "mandatory batch-1 probe exhausted device memory"
                ) from error
            optional_stop_reason = f"batch_{batch_size}_out_of_memory"
            probes.append(
                {
                    "batch_size": batch_size,
                    "status": "OPTIONAL_OUT_OF_MEMORY",
                    "forward_count": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "gpu_memory": _gpu_measurement_v1(device),
                    "process_memory": _process_memory_v1(),
                }
            )
        except VJepaVitGAblationError as error:
            if batch_size == 1:
                raise
            optional_stop_reason = f"batch_{batch_size}_technical_failure"
            probes.append(
                {
                    "batch_size": batch_size,
                    "status": "OPTIONAL_TECHNICAL_FAILURE",
                    "forward_count": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "gpu_memory": _gpu_measurement_v1(device),
                    "process_memory": _process_memory_v1(),
                }
            )
        finally:
            torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS_PASS,
        "development_only": True,
        "claim_bearing": False,
        "current_encoder_verification": current_encoder,
        "encoder_contract": encoder_contract_v1(),
        "encoder_contract_digest": ENCODER_CONTRACT_DIGEST,
        "source_binding": source_binding,
        "checkpoint_binding": arm.checkpoint_binding,
        "checkpoint_state_key_opened": CHECKPOINT_STATE_KEY,
        "predictor_constructed": False,
        "predictor_checkpoint_state_access_count": 0,
        "scientific_labels_opened": 0,
        "corpus_frames_opened": len(image_receipts),
        "smoke_images": image_receipts,
        "smoke_family_count": len({item["family"] for item in image_receipts}),
        "preflight": preflight,
        "device": {
            "requested": str(device),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
        },
        "inference_dtype": str(dtype),
        "execution_mode": arm.execution_mode,
        "parameter_dtype": "torch.float32",
        "parameter_count": parameter_count,
        "parameter_device_bytes": parameter_bytes,
        "source_validation_seconds": source_seconds,
        "checkpoint_and_model_load_seconds": load_seconds,
        "probes": probes,
        "maximum_passing_batch_size": max(
            probe["batch_size"] for probe in probes if probe.get("status") == "PASS"
        ),
        "all_registered_batches_pass": all(
            probe.get("status") == "PASS" for probe in probes
        ),
        "optional_stop_reason": optional_stop_reason,
        "process_peak_rss_bytes": _process_memory_v1()["rusage_peak_rss_bytes"],
    }
    receipt["receipt_sha256"] = canonical_digest_v1(receipt)
    return receipt


def validate_resource_smoke_receipt_v1(
    receipt: Mapping[str, Any], *, expected_checkpoint_sha256: str
) -> dict[str, Any]:
    """Fail closed on any drift in a completed exact-path smoke receipt."""

    value = dict(receipt)
    claimed_digest = value.pop("receipt_sha256", None)
    if claimed_digest != canonical_digest_v1(value):
        raise VJepaVitGAblationError("resource receipt self-digest changed")
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") != RECEIPT_STATUS_PASS
        or value.get("encoder_contract") != encoder_contract_v1()
        or value.get("encoder_contract_digest") != ENCODER_CONTRACT_DIGEST
        or value.get("checkpoint_state_key_opened") != CHECKPOINT_STATE_KEY
        or value.get("predictor_constructed") is not False
        or value.get("predictor_checkpoint_state_access_count") != 0
        or value.get("scientific_labels_opened") != 0
        or value.get("corpus_frames_opened") != SMOKE_IMAGE_COUNT
        or value.get("parameter_dtype") != "torch.float32"
    ):
        raise VJepaVitGAblationError("resource receipt contract fields changed")
    expected_output_dtype = {
        "torch.bfloat16": "torch.float32",
        "torch.float32": "torch.float32",
    }.get(value.get("inference_dtype"))
    expected_execution_mode = {
        "torch.bfloat16": "bfloat16_autocast_fp32_weights",
        "torch.float32": "float32_no_autocast",
    }.get(value.get("inference_dtype"))
    if (
        expected_output_dtype is None
        or value.get("execution_mode") != expected_execution_mode
    ):
        raise VJepaVitGAblationError("resource receipt execution mode changed")
    checkpoint = value.get("checkpoint_binding")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != str(VJEPA_CHECKPOINT.resolve())
        or checkpoint.get("sha256") != expected_checkpoint_sha256
        or checkpoint.get("byte_count") != EXPECTED_CHECKPOINT_BYTE_COUNT
    ):
        raise VJepaVitGAblationError("resource receipt checkpoint binding changed")
    source = value.get("source_binding")
    if not isinstance(source, Mapping) or source.get("commit") != VJEPA_REPOSITORY_COMMIT:
        raise VJepaVitGAblationError("resource receipt source commit changed")
    files = source.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_BINDINGS):
        raise VJepaVitGAblationError("resource receipt source inventory changed")
    for relative, (sha256, byte_count) in SOURCE_BINDINGS.items():
        binding = files[relative]
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str((VJEPA_REPOSITORY / relative).resolve())
            or binding.get("sha256") != sha256
            or binding.get("byte_count") != byte_count
        ):
            raise VJepaVitGAblationError(
                f"resource receipt source binding changed: {relative}"
            )
    license_binding = source.get("license")
    if (
        not isinstance(license_binding, Mapping)
        or license_binding.get("path")
        != str((VJEPA_REPOSITORY / "LICENSE").resolve())
        or license_binding.get("sha256") != VJEPA_LICENSE["sha256"]
        or license_binding.get("byte_count") != VJEPA_LICENSE["byte_count"]
        or license_binding.get("identity") != "MIT"
        or license_binding.get("spdx_identifier") != "MIT"
        or license_binding.get("checkpoint_separate_license_statement") is not None
    ):
        raise VJepaVitGAblationError("resource receipt official license changed")
    current = value.get("current_encoder_verification")
    if (
        not isinstance(current, Mapping)
        or current.get("classification") != "SCALE_ONLY"
        or current.get("constructor") != CURRENT_CONSTRUCTOR
        or current.get("checkpoint_state_key") != "ema_encoder"
        or current.get("output_shape") != ["batch", 768, 1024]
        or current.get("target_encoder_identity_digest")
        != CURRENT_ENCODER_IDENTITY_DIGEST
    ):
        raise VJepaVitGAblationError("current encoder verification changed")
    current_checkpoint = current.get("checkpoint_binding")
    if (
        not isinstance(current_checkpoint, Mapping)
        or current_checkpoint.get("sha256") != CURRENT_CHECKPOINT_SHA256
        or current_checkpoint.get("byte_count") != CURRENT_CHECKPOINT_BYTE_COUNT
    ):
        raise VJepaVitGAblationError("current encoder checkpoint verification changed")
    inspection = current.get("checkpoint_inspection")
    if (
        not isinstance(inspection, Mapping)
        or inspection.get("ema_state_key") != "ema_encoder"
        or inspection.get("ema_tensor_count") != 302
        or inspection.get("ema_value_count") != 304_680_960
        or inspection.get("ema_dtypes") != ["torch.float32"]
        or inspection.get("online_differs_from_ema") is not True
    ):
        raise VJepaVitGAblationError("current encoder state inspection changed")
    smoke_images = value.get("smoke_images")
    if (
        not isinstance(smoke_images, list)
        or len(smoke_images) != SMOKE_IMAGE_COUNT
        or [item.get("index") for item in smoke_images] != list(range(SMOKE_IMAGE_COUNT))
        or len({item.get("family") for item in smoke_images}) < 2
        or value.get("smoke_family_count")
        != len({item.get("family") for item in smoke_images})
    ):
        raise VJepaVitGAblationError("resource receipt smoke-image inventory changed")
    preflight = value.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("passes") is not True
        or preflight.get("failures") != []
        or preflight.get("thresholds")
        != {
            "minimum_free_vram_bytes": MIN_FREE_VRAM_BYTES,
            "minimum_available_host_ram_bytes": MIN_AVAILABLE_RAM_BYTES,
            "minimum_destination_free_bytes": MIN_DESTINATION_FREE_BYTES,
        }
    ):
        raise VJepaVitGAblationError("resource receipt preflight changed")
    preflight_gpu = preflight.get("gpu")
    gpu_processes = preflight.get("gpu_process_inventory_before_load")
    top_memory = preflight.get("top_system_memory_consumers_before_load")
    if (
        not isinstance(preflight_gpu, Mapping)
        or not isinstance(preflight_gpu.get("torch_hip_version"), str)
        or not preflight_gpu.get("torch_hip_version")
        or not isinstance(gpu_processes, Mapping)
        or gpu_processes.get("cmdline_or_environment_read") is not False
        or gpu_processes.get("fields_read") != ["pid", "comm", "VmRSS"]
        or not isinstance(gpu_processes.get("processes"), list)
        or gpu_processes.get("process_count") != len(gpu_processes["processes"])
        or not isinstance(top_memory, Mapping)
        or top_memory.get("cmdline_or_environment_read") is not False
        or top_memory.get("fields_read") != ["pid", "comm", "VmRSS"]
        or top_memory.get("limit") != TOP_MEMORY_PROCESS_LIMIT
        or not isinstance(top_memory.get("processes"), list)
        or len(top_memory["processes"]) > TOP_MEMORY_PROCESS_LIMIT
    ):
        raise VJepaVitGAblationError("resource receipt process preflight changed")
    for inventory in (gpu_processes["processes"], top_memory["processes"]):
        if not all(
            isinstance(process, Mapping)
            and set(process) == {"pid", "comm", "rss_bytes"}
            and isinstance(process["pid"], int)
            and isinstance(process["comm"], str)
            and isinstance(process["rss_bytes"], int)
            and process["rss_bytes"] >= 0
            for process in inventory
        ):
            raise VJepaVitGAblationError(
                "resource receipt process metadata fields changed"
            )
    device = value.get("device")
    if (
        not isinstance(device, Mapping)
        or device.get("torch_hip_version") != preflight_gpu["torch_hip_version"]
    ):
        raise VJepaVitGAblationError("resource receipt ROCm version changed")
    if value.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise VJepaVitGAblationError("resource receipt parameter count changed")
    probes = value.get("probes")
    if not isinstance(probes, list) or [probe.get("batch_size") for probe in probes] != list(PROBE_BATCH_SIZES):
        raise VJepaVitGAblationError("resource receipt batch probes changed")
    if probes[0].get("status") != "PASS":
        raise VJepaVitGAblationError("mandatory batch-1 probe did not pass")
    stopped = False
    maximum_passing = 0
    for probe in probes:
        batch_size = probe["batch_size"]
        status = probe.get("status")
        if stopped:
            if status != "NOT_ATTEMPTED_AFTER_OPTIONAL_STOP" or probe.get("forward_count") != 0:
                raise VJepaVitGAblationError("optional probe stop sequence changed")
            continue
        if status in {"OPTIONAL_OUT_OF_MEMORY", "OPTIONAL_RESOURCE_LIMIT_EXCEEDED", "OPTIONAL_TECHNICAL_FAILURE"}:
            if batch_size == 1:
                raise VJepaVitGAblationError("mandatory batch-1 probe cannot be optional")
            stopped = True
            continue
        if status != "PASS":
            raise VJepaVitGAblationError(f"unknown batch-{batch_size} probe status")
        maximum_passing = batch_size
        sdpa = probe.get("sdpa_backend_evidence")
        gpu = probe.get("gpu_memory")
        process = probe.get("process_memory")
        if (
            probe.get("forward_count") != 3
            or probe.get("input_shape") != [batch_size, 3, *INPUT_HW]
            or probe.get("input_dtype") != "torch.float32"
            or probe.get("output_shape") != [batch_size, TOKEN_COUNT, TOKEN_DIM]
            or probe.get("output_dtype") != expected_output_dtype
            or probe.get("output_finite") is not True
            or probe.get("deterministic_repeat_max_abs_diff") != 0.0
            or not isinstance(probe.get("warm_wall_seconds"), (float, int))
            or probe.get("warm_wall_seconds", 0) <= 0
            or not isinstance(probe.get("warm_frames_per_second"), (float, int))
            or probe.get("warm_frames_per_second", 0) <= 0
            or not isinstance(sdpa, Mapping)
            or sdpa.get("fused_backend_observed") is not True
            or not sdpa.get("fused_events")
            or not isinstance(gpu, Mapping)
            or gpu.get("peak_allocated_bytes", MAX_PEAK_VRAM_BYTES + 1)
            > MAX_PEAK_VRAM_BYTES
            or not isinstance(process, Mapping)
            or process.get("rusage_peak_rss_bytes", MAX_PROCESS_OR_SYSTEM_RAM_BYTES)
            >= MAX_PROCESS_OR_SYSTEM_RAM_BYTES
            or probe.get("system_used_ram_bytes", MAX_PROCESS_OR_SYSTEM_RAM_BYTES)
            >= MAX_PROCESS_OR_SYSTEM_RAM_BYTES
        ):
            raise VJepaVitGAblationError(
                f"resource receipt batch-{batch_size} measurement changed"
            )
    if value.get("maximum_passing_batch_size") != maximum_passing:
        raise VJepaVitGAblationError("resource receipt maximum passing batch changed")
    if value.get("all_registered_batches_pass") is not all(
        probe.get("status") == "PASS" for probe in probes
    ):
        raise VJepaVitGAblationError("resource receipt all-pass projection changed")
    value["receipt_sha256"] = claimed_digest
    return value


def _write_receipt_exclusive_v1(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")


def _parse_smoke_image_v1(value: str) -> tuple[str, Path]:
    family, separator, raw_path = value.partition("=")
    if not separator or not family.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("smoke image must be FAMILY=/absolute/path.png")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("smoke image path must be absolute")
    return family.strip(), path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-smoke", action="store_true", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(ALLOWED_DTYPES), default="bfloat16")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--image",
        action="append",
        type=_parse_smoke_image_v1,
        required=True,
        help="repeat exactly four times as FAMILY=/absolute/existing-224x224.png",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    receipt = run_resource_smoke_v1(
        device=torch.device(args.device),
        dtype=ALLOWED_DTYPES[args.dtype],
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        images=args.image,
    )
    if args.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    else:
        if args.output.resolve() != RESOURCE_SMOKE_RECEIPT_PATH.resolve():
            parser.error(
                f"smoke receipt output must be {RESOURCE_SMOKE_RECEIPT_PATH}"
            )
        _write_receipt_exclusive_v1(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
