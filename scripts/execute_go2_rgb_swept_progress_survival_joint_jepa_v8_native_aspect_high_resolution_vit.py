#!/usr/bin/env python3
"""Execute the single fresh V8 native-aspect high-resolution ViT probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_v7 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v7_"
    "hierarchical_cnn_encoder"
)
_v4 = _v7._v4
_v1 = _v7._v1
_direct = importlib.import_module(
    "scripts.run_go2_direct_egocentric_bev_state_jepa_v1"
)

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v8_"
    "native_aspect_high_resolution_vit/attempt_v1"
)
CHECKPOINT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit_checkpoint_v1"
TRACE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit_trace_v1"
RESULT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit_result_v1"
FAILURE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit_failure_v1"
PREREGISTRATION_COMMIT = "b17599fa1bb49017178f45d0e1a4c83ac8bb9314"

LABEL_ROOT_RELATIVE_PATH = _v4.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v4.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v4.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v4.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v4.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v4.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v4.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v4.ACTION_ORDER
ROLE_FILES = _v4.ROLE_FILES
MICROBATCH_SIZE = _v4.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v4.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v4.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v4.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v4.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v4.CONSTRUCTOR_INITIALIZATION_SEED
SEMANTIC_DECODER_INITIALIZATION_SEED = _v4.SEMANTIC_DECODER_INITIALIZATION_SEED
EXPERIMENT_SEED = _v4.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v4.BOOTSTRAP_SEED
CONTROL_NAMES = _v4.CONTROL_NAMES
ALL_ARM_NAMES = _v4.ALL_ARM_NAMES
GATE_THRESHOLDS = _v4.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v4.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v4.AUXILIARY_OBJECTIVE)

NATIVE_IMAGE_HEIGHT_V8 = 168
NATIVE_IMAGE_WIDTH_V8 = 224
NATIVE_TOKEN_HEIGHT_V8 = 24
NATIVE_TOKEN_WIDTH_V8 = 32
NATIVE_SPATIAL_TOKEN_COUNT_V8 = 768
NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8 = 2_845_824
NATIVE_TOKEN_CELL_RADII_XY_V8 = (4.0, 3.0)
NATIVE_POSITIONAL_PARAMETER_INCREASE_V8 = 98_304
NATIVE_LOADER_POLICY_V8 = {
    "schema": "lewm_v8_native_rgb_decode_policy_v1",
    "encoded_size_wh_required": [224, 168],
    "returned_shape_chw": [3, 168, 224],
    "decode": "Pillow_convert_RGB_direct",
    "resize": False,
    "crop": False,
    "pad": False,
    "upscale": False,
    "augmentation": False,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "dtype": "float32",
}

scientific_metrics_v8 = _v4.scientific_metrics_v4
semantic_metrics_v8 = _v4.semantic_metrics_v4
paired_control_comparison_v8 = _v4.paired_control_comparison_v4
evaluate_gate_v8 = _v4.evaluate_gate_v4


class NativeAspectDirectBevNarrowLoaderV8(_direct.DirectBevNarrowLoader):
    """The inherited narrow loader with native 224x168 RGB tensorization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._native_decode_success_count = 0
        self._native_size_mismatch_count = 0
        self._decoded_format_count: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def receipt(self) -> dict[str, Any]:
        result = super().receipt()
        result.update({
            "native_rgb_decode_policy": copy.deepcopy(NATIVE_LOADER_POLICY_V8),
            "native_rgb_decode_success_count": self._native_decode_success_count,
            "native_rgb_size_mismatch_count": self._native_size_mismatch_count,
            "native_rgb_decoded_format_count": dict(self._decoded_format_count),
            "resize_crop_pad_call_count": 0,
        })
        return result

    def image(
        self,
        endpoint_identity: str,
        *,
        role: str,
        stage: str,
        kind: str,
    ) -> Any:
        if kind not in self._IMAGE_KINDS:
            raise ValueError("V8 native RGB request kind changed")
        endpoint = self._endpoint(endpoint_identity, role=role)
        self._increment(self._counters["rgb_request_count"], kind)
        cached = self.image_cache.get(endpoint_identity)
        if cached is not None:
            self._increment(self._counters["rgb_cache_hit_count"], kind)
            self.image_cache.move_to_end(endpoint_identity)
            self._sync_progress()
            return cached

        self._increment(self._counters["rgb_cache_miss_count"], kind)
        self._increment(self._counters["rgb_physical_read_attempt_count"], kind)
        self._sync_progress()
        raw = self.inputs.read_rgb(
            str(endpoint["image_path_metadata_only"]),
            str(endpoint["image_sha256_commitment_only"]),
            role=role,
            arm="rgb_direct_egocentric_bev_state_jepa_v1",
            stage=stage,
        )
        self._increment(self._counters["rgb_physical_read_success_count"], kind)
        relative = str(endpoint["image_path_metadata_only"])
        self._physical_image_path_kind.setdefault(relative, kind)
        with self.runtime.Image.open(io.BytesIO(raw)) as decoded:
            if tuple(decoded.size) != (NATIVE_IMAGE_WIDTH_V8, NATIVE_IMAGE_HEIGHT_V8):
                self._native_size_mismatch_count += 1
                self._sync_progress()
                raise PermissionError("V8 RGB is not exact native 224x168")
            encoded_format = str(decoded.format or "UNKNOWN")
            image = decoded.convert("RGB")
            array = self.runtime.np.asarray(image, dtype=self.runtime.np.float32) / 255.0
        if tuple(array.shape) != (NATIVE_IMAGE_HEIGHT_V8, NATIVE_IMAGE_WIDTH_V8, 3):
            raise PermissionError("V8 decoded RGB shape changed")
        tensor = (
            self.runtime.torch.from_numpy(array.copy())
            .permute(2, 0, 1)
            .contiguous()
        )
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        normalized = (tensor - mean) / std
        if (
            normalized.dtype != self.runtime.torch.float32
            or tuple(normalized.shape) != (3, NATIVE_IMAGE_HEIGHT_V8, NATIVE_IMAGE_WIDTH_V8)
            or not bool(self.runtime.torch.isfinite(normalized).all())
        ):
            raise TypeError("V8 normalized native RGB contract changed")
        self._native_decode_success_count += 1
        self._decoded_format_count[encoded_format] = self._decoded_format_count.get(encoded_format, 0) + 1
        self.image_cache[endpoint_identity] = normalized
        self.image_cache.move_to_end(endpoint_identity)
        while len(self.image_cache) > self.maximum_image_cache:
            self.image_cache.popitem(last=False)
        self._sync_progress()
        return normalized


def native_loader_policy_receipt_v8() -> dict[str, Any]:
    return copy.deepcopy(NATIVE_LOADER_POLICY_V8)


def _fresh_output_root_v8(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh native-aspect-high-resolution-ViT attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _v1_training() -> Any:
    return importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1")


def _validate_training_core_v8(training_v3: Any, training_v8: Any) -> None:
    _v4._validate_training_core_v4(_v1_training(), training_v3)
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        if getattr(training_v8, name, None) != getattr(training_v3, name):
            raise PermissionError(f"V8 training wrapper changed inherited {name}")
    if (
        getattr(training_v8, "NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8", None)
        != NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8
    ):
        raise PermissionError("V8 training wrapper changed encoder parameter count")
    if not callable(getattr(training_v8, "run_fixed_training_v8", None)):
        raise PermissionError("V8 training wrapper entrypoint changed")


def _validate_model_api_v8(model_api: Any) -> None:
    for name, expected in (
        ("NATIVE_IMAGE_HEIGHT_V8", NATIVE_IMAGE_HEIGHT_V8),
        ("NATIVE_IMAGE_WIDTH_V8", NATIVE_IMAGE_WIDTH_V8),
        ("NATIVE_TOKEN_HEIGHT_V8", NATIVE_TOKEN_HEIGHT_V8),
        ("NATIVE_TOKEN_WIDTH_V8", NATIVE_TOKEN_WIDTH_V8),
        ("NATIVE_SPATIAL_TOKEN_COUNT_V8", NATIVE_SPATIAL_TOKEN_COUNT_V8),
        ("NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8", NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8),
        ("NATIVE_TOKEN_CELL_RADII_XY_V8", NATIVE_TOKEN_CELL_RADII_XY_V8),
    ):
        if getattr(model_api, name, None) != expected:
            raise PermissionError(f"V8 model changed {name}")
    for name in (
        "NativeAspectVisionEncoderV8",
        "NativeAspectGeometryAnchoredDeformableBevLiftV8",
        "GeometryAnchoredSweptProgressSurvivalJointJepaV8",
    ):
        if not callable(getattr(model_api, name, None)):
            raise PermissionError(f"V8 model API lacks {name}")


def _names_sha256_v8(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _expected_native_positional_embedding_v8(legacy: Any, *, torch: Any) -> Any:
    if tuple(legacy.shape) != (1, 257, 192) or legacy.dtype != torch.float32:
        raise RuntimeError("legacy positional embedding contract changed")
    cls = legacy[:, :1].detach().cpu().float().contiguous()
    spatial = (
        legacy[:, 1:].detach().cpu().float().reshape(1, 16, 16, 192)
        .permute(0, 3, 1, 2).contiguous()
    )
    migrated = torch.nn.functional.interpolate(
        spatial,
        size=(NATIVE_TOKEN_HEIGHT_V8, NATIVE_TOKEN_WIDTH_V8),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).permute(0, 2, 3, 1).reshape(1, NATIVE_SPATIAL_TOKEN_COUNT_V8, 192).contiguous()
    return torch.cat((cls, migrated), dim=1)


def _migration_receipt_v8(model: Any, clean_v4: Any, *, torch: Any) -> Mapping[str, Any]:
    v8_online = model.encoder.state_dict()
    v4_online = clean_v4.encoder.state_dict()
    if v8_online.keys() != v4_online.keys():
        raise RuntimeError("V8 encoder state inventory changed")
    for name in v4_online:
        if name != "pos_embed" and not torch.equal(v8_online[name], v4_online[name]):
            raise RuntimeError(f"V8 changed inherited encoder tensor {name}")
    expected_position = _expected_native_positional_embedding_v8(v4_online["pos_embed"], torch=torch)
    if not torch.equal(v8_online["pos_embed"].detach().cpu(), expected_position):
        raise RuntimeError("V8 positional interpolation changed")
    if not torch.equal(v8_online["pos_embed"][:, :1], v4_online["pos_embed"][:, :1]):
        raise RuntimeError("V8 CLS position changed")
    if model.encoder.pos_embed.numel() - clean_v4.encoder.pos_embed.numel() != NATIVE_POSITIONAL_PARAMETER_INCREASE_V8:
        raise RuntimeError("V8 positional parameter increase changed")
    v8_state = model.state_dict()
    v4_state = clean_v4.state_dict()
    if v8_state.keys() != v4_state.keys():
        raise RuntimeError("V8 whole-model state inventory changed")
    migrated_names = {"encoder.pos_embed", "target_encoder.pos_embed"}
    if any(
        not torch.equal(v8_state[name], v4_state[name])
        for name in v4_state if name not in migrated_names
    ):
        raise RuntimeError("V8 changed inherited non-positional model state")
    if model.bev_lift.state_dict().keys() != clean_v4.bev_lift.state_dict().keys() or any(
        not torch.equal(value, clean_v4.bev_lift.state_dict()[name])
        for name, value in model.bev_lift.state_dict().items()
    ):
        raise RuntimeError("V8 changed inherited BEV-lift parameter state")
    if tuple(model.bev_lift.native_token_cell_radii_xy) != NATIVE_TOKEN_CELL_RADII_XY_V8:
        raise RuntimeError("V8 native token-cell radii changed")
    with torch.no_grad():
        legacy_sampling = clean_v4.bev_lift.forward_with_sampling(torch.zeros((1, 256, 192), dtype=torch.float32))
        native_sampling = model.bev_lift.forward_with_sampling(torch.zeros((1, 768, 192), dtype=torch.float32))
        native_raw_offsets = model.bev_lift.raw_offsets[None]
        expected_native_offsets = torch.tanh(native_raw_offsets) * native_raw_offsets.new_tensor(
            NATIVE_TOKEN_CELL_RADII_XY_V8
        )
        legacy_proposed_grid = (
            clean_v4.bev_lift.anchor_grid_xy[None, ..., None, :]
            + clean_v4.bev_lift.config.offset_radius_token_cells
            * torch.tanh(clean_v4.bev_lift.raw_offsets[None])
            * (2.0 / clean_v4.bev_lift.config.token_side)
        )
        native_proposed_grid = (
            model.bev_lift.anchor_grid_xy[None, ..., None, :]
            + model.bev_lift.config.offset_radius_token_cells
            * torch.tanh(model.bev_lift.raw_offsets[None])
            * (2.0 / model.bev_lift.config.token_side)
        )
    if not torch.equal(native_sampling.offsets_token_cells, expected_native_offsets):
        raise RuntimeError("V8 native token-cell offset reporting changed")
    if not torch.equal(native_proposed_grid, legacy_proposed_grid):
        raise RuntimeError("V8 changed inherited proposed normalized sampling grid")
    for name in ("anchor_in_frustum", "sample_valid_mask", "cell_valid_mask", "sample_grid_xy", "sample_weights"):
        if not torch.equal(getattr(legacy_sampling, name), getattr(native_sampling, name)):
            raise RuntimeError(f"V8 changed inherited normalized sampling {name}")
    return {
        "source": "fresh clean V4 construction with identical N320 state and masks",
        "migrated_state_names": sorted(migrated_names),
        "all_other_state_tensors_bit_exact": True,
        "encoder_state_tensor_count": len(v8_online),
        "encoder_state_name_inventory_sha256": _names_sha256_v8(tuple(v8_online)),
        "cls_token_bit_exact": bool(torch.equal(model.encoder.cls_token, clean_v4.encoder.cls_token)),
        "cls_position_bit_exact": True,
        "spatial_position_migration": {
            "source_shape": [1, 256, 192], "target_shape": [1, 768, 192],
            "cpu_float32": True, "mode": "bicubic", "align_corners": False,
            "antialias": False, "row_major": True, "bit_exact": True,
        },
        "positional_parameter_increase": NATIVE_POSITIONAL_PARAMETER_INCREASE_V8,
        "bev_lift_parameter_state_bit_exact": True,
        "native_token_cell_radii_xy": list(NATIVE_TOKEN_CELL_RADII_XY_V8),
        "native_offsets_token_cells": {
            "formula": "tanh(raw_offsets) * [4.0, 3.0]",
            "bit_exact": True,
        },
        "proposed_normalized_sampling_grid_bit_exact_v4": True,
        "normalized_sampling_grid_and_masks_bit_exact": True,
    }


def _initial_model_receipt_v8(
    model: Any,
    partition: Any,
    migration: Mapping[str, Any],
    *,
    torch: Any,
    model_api: Any,
    inherited_semantic_method: Any,
) -> Mapping[str, Any]:
    if not isinstance(model.encoder, model_api.NativeAspectVisionEncoderV8) or not isinstance(
        model.target_encoder, model_api.NativeAspectVisionEncoderV8
    ):
        raise RuntimeError("V8 native encoder type changed")
    if not isinstance(model.bev_lift, model_api.NativeAspectGeometryAnchoredDeformableBevLiftV8) or not isinstance(
        model.target_bev_lift, model_api.NativeAspectGeometryAnchoredDeformableBevLiftV8
    ):
        raise RuntimeError("V8 native lift type changed")
    if any(
        tuple(lift.native_token_cell_radii_xy) != NATIVE_TOKEN_CELL_RADII_XY_V8
        for lift in (model.bev_lift, model.target_bev_lift)
    ):
        raise RuntimeError("V8 online/target native token-cell radii changed")
    online = tuple(model.encoder.named_parameters())
    target = tuple(model.target_encoder.named_parameters())
    if tuple(name for name, _ in online) != tuple(name for name, _ in target):
        raise RuntimeError("V8 online/target encoder inventories differ")
    if sum(parameter.numel() for _, parameter in online) != NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8:
        raise RuntimeError("V8 encoder parameter count changed")
    if any(not parameter.requires_grad for _, parameter in online) or any(
        parameter.requires_grad for _, parameter in target
    ):
        raise RuntimeError("V8 online/target trainability changed")
    if any(
        not torch.equal(left.detach(), right.detach())
        for (_, left), (_, right) in zip(online, target, strict=True)
    ):
        raise RuntimeError("V8 target encoder is not an exact initial copy")
    online_names = tuple(f"encoder.{name}" for name, _ in online)
    target_names = tuple(f"target_encoder.{name}" for name, _ in target)
    if tuple(partition.names["encoder"]) != online_names or tuple(
        name for name in partition.names["target"] if name.startswith("target_encoder.")
    ) != target_names:
        raise RuntimeError("V8 encoder partition coverage changed")
    parameter = next(model.parameters())
    probe = torch.zeros((1, 3, 168, 224), dtype=parameter.dtype, device=parameter.device)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        online_tokens = model.encoder.forward_tokens(probe)
        target_tokens = model.target_encoder.forward_tokens(probe)
    model.train(was_training)
    if tuple(online_tokens.shape) != (1, 769, 192) or not torch.equal(online_tokens, target_tokens):
        raise RuntimeError("V8 token/target contract changed")
    inherited_decoder = _v4._initial_decoder_receipt_v4(
        model, partition, torch=torch,
        inherited_semantic_method=inherited_semantic_method,
    )
    return {
        "native_loader_policy": native_loader_policy_receipt_v8(),
        "migration": dict(migration),
        "inherited_v4_decoder": inherited_decoder,
        "input_shape_chw": [3, 168, 224],
        "token_shape": [769, 192],
        "spatial_token_lattice_hw": [24, 32],
        "native_token_cell_radii_xy": list(NATIVE_TOKEN_CELL_RADII_XY_V8),
        "online_parameter_count": sum(parameter.numel() for _, parameter in online),
        "online_parameter_tensor_count": len(online),
        "target_parameter_count": sum(parameter.numel() for _, parameter in target),
        "all_online_parameters_in_encoder_partition_exactly_once": True,
        "all_target_parameters_frozen_in_target_partition_exactly_once": True,
        "target_initial_copy_exact": True,
        "initial_online_target_tokens_exact": True,
        "initial_hard_sync_count": int(model.target_hard_sync_count.item()),
        "initial_ema_update_count": int(model.ema_update_count.item()),
    }


def _physical_calibration_stage_v8(full_arm_passed: bool) -> Mapping[str, Any]:
    return _v7._physical_calibration_stage_v7(full_arm_passed)


def execute_v8(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v8(repository_root)
    initial_model: Mapping[str, Any] | None = None
    encoder_activity: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_labels_v1")
        manifest, rows_by_role = _v1.load_label_bundle_v1(repository_root, labels_api=labels_api)
        context = dict(_v1._prepare_runtime_v1(repository_root, manifest, labels_api))
        torch, np = context["torch"], context["np"]
        context["loader"] = NativeAspectDirectBevNarrowLoaderV8(
            context["runtime"], context["inputs"], progress=context["progress"]
        )
        if labels_api.summarize_preflight_v1(rows_by_role, context["schedule"]) != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_v1 = _v1_training()
        training_v3 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux")
        training_v8 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit")
        _validate_training_core_v8(training_v3, training_v8)
        frozen = {role: training_v1.freeze_role_labels_v1(rows, role=role, np=np) for role, rows in rows_by_role.items()}
        informative = {
            role: np.asarray([group[0]["informative_state"] for group in labels.state_groups], dtype=np.bool_)
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit")
        _validate_model_api_v8(model_api)
        v4_model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder")
        parent_model_api = importlib.import_module("lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1")
        survival_scoring = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1")
        metrics_api = importlib.import_module("lewm.benchmarks.go2_post_action_projective_support_metrics_v1")
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {name: value.detach().cpu().float().contiguous().clone() for name, value in context["fit"].encoder.state_dict().items()}
        masks = survival_scoring.build_swept_progress_masks_v1()
        persistence_masks = survival_scoring.build_current_frame_swept_progress_masks_v1()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV8(n320_state, masks)
        clean_v4 = v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(n320_state, masks)
        migration = _migration_receipt_v8(model, clean_v4, torch=torch)
        del clean_v4
        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v8(
            model, partition, migration, torch=torch, model_api=model_api,
            inherited_semantic_method=parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.semantic_logits_from_latent,
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(name.startswith("predictor.swept_progress_head.") for name in partition.names["predictor"]):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, diagnostics = training_v8.run_fixed_training_v8(
            model, optimizer, context["loader"], pairs["train"], frozen["train"], context["schedule"], context["device"]
        )
        accounting = dict(accounting_state.__dict__)
        encoder_activity = diagnostics["native_aspect_high_resolution_vit_encoder"]
        model.eval()
        model.requires_grad_(False)
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        checkpoint_buffer = io.BytesIO()
        torch.save({
            "schema": CHECKPOINT_SCHEMA, "development_only": True,
            "resume_authorized": False, "qualified": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
            "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
            "experiment_seed": EXPERIMENT_SEED,
            "initialization_source": "exact_n320_v4_encoder_with_one_cpu_bicubic_spatial_position_migration",
            "predecessor_experiment_checkpoint_read": False,
            "native_loader_policy": native_loader_policy_receipt_v8(),
            "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
            "initial_v8_model": initial_model,
            "native_aspect_high_resolution_vit_encoder_activity": encoder_activity,
            "training_diagnostics": diagnostics, "accounting": accounting,
            "model_state_dict": state,
        }, checkpoint_buffer)
        checkpoint_binding = _v1._atomic_write_v1(output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue())
        _, trace_binding = _v1._write_json_v1(output / "training_trace.json", {
            "schema": TRACE_SCHEMA, "status": "COMPLETE",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "initial_v8_model": initial_model,
            "native_aspect_high_resolution_vit_encoder_activity": encoder_activity,
            "training_diagnostics": diagnostics, "accounting": accounting,
            "rows": list(trace),
        })

        action_prior_m = frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64) * PROGRESS_SEGMENT_M
        scored = {
            role: _v1.score_role_v1(
                model, context["loader"], pairs[role], frozen[role], action_prior_m,
                context["device"], torch=torch, np=np, training_core=training_v1,
                current_frame_persistence_masks=persistence_masks, metrics_api=metrics_api,
            ) for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v8(
                    scored[role]["scores_m"][arm], frozen[role].prefix_lengths,
                    informative[role], frozen[role].scene_ids, frozen[role].family_ids, np=np,
                ) for arm in ALL_ARM_NAMES
            } for role in scored
        }
        selection_semantic = semantic_metrics_v8(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"], np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v8(
                selection_scores["full"], selection_scores[name], selection_labels.prefix_lengths,
                informative["checkpoint_selection"], selection_labels.scene_ids,
                selection_labels.family_ids, np=np,
            ) for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v8(role_metrics["checkpoint_selection"], selection_semantic, comparisons)
        full_arm_passed = bool(gate["passed"])
        checkpoint_access = "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED"
        calibration_stage = _physical_calibration_stage_v8(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(persistence_masks),
        }
        result, _ = _v1._write_json_v1(output / "result.json", {
            "schema": RESULT_SCHEMA,
            "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION" if full_arm_passed else "FAIL_DEVELOPMENT_FULL_ARM",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "full_arm_gate": gate, "gate": gate,
            "physical_evidence_calibration": calibration_stage,
            "caps": {"updates": MAXIMUM_UPDATES, "microbatch_graphs": 4_000, "presentations": MAXIMUM_PRESENTATIONS},
            "seeds": {
                "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "experiment_and_stochastic_execution": EXPERIMENT_SEED, "bootstrap": BOOTSTRAP_SEED,
            },
            "label_manifest": {
                "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                "content_sha256": manifest["content_sha256"], "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                "role_files": manifest["files"],
            },
            "n320": {
                "gate_content_sha256": context["n320_gate"]["content_sha256"],
                "checkpoint": context["n320_checkpoint"],
                "encoder_parameter_initialization_retained_except_spatial_position_migration": True,
                "predecessor_experiment_checkpoint_read": False,
            },
            "hardware": context["hardware"],
            "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
            "masks": mask_receipts,
            "scientific_change_from_v4": {
                "only_change": "native_224x168_patch7_vit_and_exact_rectangular_lift_input_adapter",
                "initial_v8_model": initial_model,
                "native_loader_policy": native_loader_policy_receipt_v8(),
                "same_bound_rgb_bytes": True, "dataset_identity_changed": False,
                "input_tensorization_changed": True, "model_changed": True,
                "inherited_nonreplacement_state_bit_exact": True,
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "optimizer_rules_changed": False,
                "optimizer_parameter_tensor_membership_changed": False,
                "losses_changed": False, "schedule_changed": False, "evaluation_changed": False,
            },
            "training": {
                "core": "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit",
                "accounting": accounting, "diagnostics": diagnostics,
                "native_aspect_high_resolution_vit_encoder_activity": encoder_activity,
                "joint_from_update_one": True, "separate_head_or_predictor_training": False,
                "checkpoint_access_status": checkpoint_access,
                "checkpoint": checkpoint_binding, "trace": trace_binding,
            },
            "action_prior_mean_progress_m": action_prior_m.tolist(), "roles": role_metrics,
            "selection_semantic": selection_semantic, "selection_control_comparisons": comparisons,
            "wrong_rgb_mapping_sha256": {role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored},
            "determinism": {
                "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()), "warn_only": True,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "access": access_receipt,
            "authority": {
                "development_only": True, "g2_navigation_final_evaluation_opened": False,
                "heldout_or_sealed_opened": False, "physical_evidence_gate_passed": False,
                "checkpoint_qualified": False, "promotion_performed": False,
                "retry_or_resume_authorized": False,
                "checkpoint_access_authorized_for_physical_calibration": full_arm_passed,
            },
        })
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (output / "failure.json").exists():
            try:
                _v1._write_json_v1(output / "failure.json", {
                    "schema": FAILURE_SCHEMA, "status": "FAILED_NO_RETRY_OR_RESUME",
                    "error_type": type(error).__name__, "error_message": str(error),
                    "traceback": traceback.format_exc(), "preregistration_commit": PREREGISTRATION_COMMIT,
                    "native_loader_policy": native_loader_policy_receipt_v8(),
                    "initial_v8_model": initial_model,
                    "native_aspect_high_resolution_vit_encoder_activity": encoder_activity,
                    "predecessor_experiment_checkpoint_read": False,
                    "physical_calibration_run_in_this_attempt": False,
                    "authority": {
                        "development_only": True, "g2_navigation_final_evaluation_opened": False,
                        "heldout_or_sealed_opened": False, "checkpoint_qualified": False,
                        "retry_or_resume_authorized": False,
                    },
                })
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v8(repository_root=args.repository_root)
    print(_v1._canonical_json_bytes({"status": result["status"], "result": f"{OUTPUT_RELATIVE_PATH}/result.json"}).decode("utf-8"))
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
