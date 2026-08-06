#!/usr/bin/env python3
"""Frozen dense encoders and their official preprocessing, for the screen.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Three arms, each pinned to one official checkpoint and one official dense
extraction point.  No layer sweep, no pooled variant, no size sweep.  The
preprocessing preserves the source camera geometry: the render is 224x168 and
no arm anisotropically squashes it except arm A, which must stay on its own
trained contract.

Arm A -- project ViT patch tokens from the direct-BEV ``update_400`` baseline.
Arm B -- DINOv2 ViT-L/14 ``x_norm_patchtokens`` at the render's native pixels.
Arm C -- V-JEPA 2.1 ViT-L/16-384 final normed tokens through the official
         single-frame image tokenizer (``patch_embed_img``, tubelet 1), at the
         official short-side-384 eval scale with the centre crop omitted so the
         full field of view survives.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _r in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SOURCE_FRAME_HW = (168, 224)

PROJECT_CHECKPOINT = REPO_ROOT / (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_semantic_anchor_state_v3/"
    "rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_state_probe_v3_update100_trend_gate_timing_v1/"
    "checkpoints/update_400.pt"
)
DINOV2_REPOSITORY = Path.home() / ".cache/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
DINOV2_SOURCE_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"

VJEPA_REPOSITORY = Path.home() / ".cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812"
VJEPA_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA_CHECKPOINT = Path.home() / ".cache/vjepa2_1_vitl_dist_vitG_384.pt"
VJEPA_SOURCE_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
VJEPA_FALLBACK_CHECKPOINT = Path.home() / ".cache/vjepa2_1_vitb_dist_vitG_384.pt"


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalise(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def _to_chw(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------
# preprocessing -- one function per arm, each returning a (3, H, W) tensor
# --------------------------------------------------------------------------
def preprocess_project_vit(path: str) -> torch.Tensor:
    """The trained contract of arm A: 112x112 bilinear, /255, ImageNet norm.

    This is anisotropic (224x168 -> 112x112).  It is retained because the arm
    must stay on the preprocessing its weights were trained under; the squash
    is a property of the incumbent, not of this screen.
    """
    with Image.open(path) as decoded:
        image = decoded.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
    return _normalise(_to_chw(image))


def preprocess_dinov2(path: str) -> torch.Tensor:
    """Native render pixels, 224x168.  14 divides both, so no pad and no crop."""
    with Image.open(path) as decoded:
        image = decoded.convert("RGB")
        if image.size != (224, 168):
            raise RuntimeError(f"unexpected render size {image.size} for {path}")
    return _normalise(_to_chw(image))


def preprocess_vjepa(path: str) -> torch.Tensor:
    """Official short-side-384 eval scale, centre crop omitted.

    224x168 -> 512x384 is an exact 16/7 isotropic upscale, so the aspect ratio
    and the whole field of view survive, and 16 divides both sides: no padding
    and therefore no pure-padding tokens.
    """
    with Image.open(path) as decoded:
        image = decoded.convert("RGB").resize((512, 384), Image.Resampling.BICUBIC)
    return _normalise(_to_chw(image))


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
class ProjectViTArm:
    name = "project_vit_update400"
    family = "project_task_trained"
    token_grid = (16, 16)
    token_dim = 192
    input_hw = (112, 112)
    preprocess = staticmethod(preprocess_project_vit)
    output_layer = "encoder.forward_tokens final block, CLS discarded"

    def build(self, device, dtype):
        import lewm.models.direct_egocentric_bev_state_jepa_v1 as m1

        encoder = m1._construct_n320_encoder_without_rng_draw()  # noqa: SLF001
        state = torch.load(PROJECT_CHECKPOINT, map_location="cpu", weights_only=False)
        weights = {
            k[len("encoder.") :]: v
            for k, v in state["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(weights, strict=True)
        encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._module = encoder
        return encoder

    @torch.no_grad()
    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        return self._module.forward_tokens(batch)[:, 1:, :]

    def identity(self) -> dict:
        return {
            "model_id": "project_direct_bev_vit_n320_update_400",
            "source_repository": "in-repo: lewm.models.direct_egocentric_bev_state_jepa_v1",
            "release": "local training run, checkpoint update_400.pt",
            "checkpoint_path": str(PROJECT_CHECKPOINT),
            "checkpoint_sha256": file_sha256(PROJECT_CHECKPOINT),
            "checkpoint_source_url": None,
        }


class DinoV2Arm:
    name = "dinov2_vitl14"
    family = "dinov2_image_ssl"
    token_grid = (12, 16)
    token_dim = 1024
    input_hw = (168, 224)
    preprocess = staticmethod(preprocess_dinov2)
    output_layer = "forward_features()['x_norm_patchtokens']"

    def build(self, device, dtype):
        if not DINOV2_CHECKPOINT.is_file():
            raise FileNotFoundError(f"missing official DINOv2 weights: {DINOV2_CHECKPOINT}")
        model = torch.hub.load(
            str(DINOV2_REPOSITORY), "dinov2_vitl14", source="local", pretrained=False
        )
        state = torch.load(DINOV2_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._module = model
        return model

    @torch.no_grad()
    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        return self._module.forward_features(batch)["x_norm_patchtokens"]

    def identity(self) -> dict:
        return {
            "model_id": "dinov2_vitl14 (LVD-142M)",
            "source_repository": f"facebookresearch/dinov2 @ {DINOV2_REPOSITORY_COMMIT}",
            "release": "DINOv2 official torch.hub release, no registers",
            "checkpoint_path": str(DINOV2_CHECKPOINT),
            "checkpoint_sha256": file_sha256(DINOV2_CHECKPOINT),
            "checkpoint_source_url": DINOV2_SOURCE_URL,
        }


class VJepa21Arm:
    """V-JEPA 2.1 through its official single-frame image tokenizer.

    ``img_temporal_dim_size=1`` selects ``patch_embed_img`` (tubelet 1) and the
    image modality embedding, so a genuine image token path exists and no frame
    is duplicated to manufacture a clip.  There is therefore no additional
    temporal context in this screen: exactly the labelled current frame.
    """

    name = "vjepa2_1_vitl_384"
    family = "vjepa_video_ssl"
    token_grid = (24, 32)
    token_dim = 1024
    input_hw = (384, 512)
    preprocess = staticmethod(preprocess_vjepa)
    output_layer = "encoder final block, norms_block[-1] (return_hierarchical=False)"

    def __init__(self, checkpoint: Path = VJEPA_CHECKPOINT, constructor: str = "vjepa2_1_vit_large_384"):
        self.checkpoint = checkpoint
        self.constructor = constructor
        if constructor == "vjepa2_1_vit_base_384":
            self.name = "vjepa2_1_vitb_384"
            self.token_dim = 768

    def build(self, device, dtype):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"missing official V-JEPA 2.1 weights: {self.checkpoint}")
        if str(VJEPA_REPOSITORY) not in sys.path:
            sys.path.insert(0, str(VJEPA_REPOSITORY))
        import importlib

        backbones = importlib.import_module("src.hub.backbones")
        encoder, _predictor = getattr(backbones, self.constructor)(pretrained=False)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        encoder_state = backbones._clean_backbone_key(state["ema_encoder"])  # noqa: SLF001
        encoder.load_state_dict(encoder_state, strict=True)
        del state, _predictor
        encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self._module = encoder
        return encoder

    @torch.no_grad()
    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 3, 1, H, W): T == img_temporal_dim_size routes the
        # forward through the image tokenizer rather than the 3D tubelet one.
        return self._module(batch.unsqueeze(2))

    def identity(self) -> dict:
        return {
            "model_id": f"{self.constructor} ({self.checkpoint.name})",
            "source_repository": f"facebookresearch/vjepa2 @ {VJEPA_REPOSITORY_COMMIT}",
            "release": "V-JEPA 2.1 official release, ema_encoder key",
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": file_sha256(self.checkpoint),
            "checkpoint_source_url": VJEPA_SOURCE_URL,
        }


def preprocessing_identity(arm) -> dict:
    return {
        "source_frame_hw": list(SOURCE_FRAME_HW),
        "input_hw": list(arm.input_hw),
        "resample": {
            "project_vit_update400": "PIL bilinear to 112x112 (anisotropic, trained contract)",
            "dinov2_vitl14": "none: native render pixels",
            "vjepa2_1_vitl_384": "PIL bicubic isotropic 16/7 to 512x384, no crop",
            "vjepa2_1_vitb_384": "PIL bicubic isotropic 16/7 to 512x384, no crop",
        }[arm.name],
        "scale_to_255": True,
        "normalisation": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "padding": "none",
        "pure_padding_tokens": 0,
        "token_grid_hw": list(arm.token_grid),
        "token_dim": arm.token_dim,
        "output_layer": arm.output_layer,
    }


def preprocessing_hash(arm) -> str:
    return hashlib.sha256(
        json.dumps(preprocessing_identity(arm), sort_keys=True).encode()
    ).hexdigest()
