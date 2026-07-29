from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit.py"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _png(width: int = 224, height: int = 168) -> tuple[bytes, np.ndarray]:
    yy, xx = np.indices((height, width))
    array = np.stack(
        ((xx * 3 + yy) % 256, (xx + yy * 5) % 256, (xx * 7 + yy * 11) % 256),
        axis=-1,
    ).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG")
    return stream.getvalue(), array


class _Inputs:
    def __init__(self, rows: dict[str, tuple[bytes, str]]) -> None:
        self.rows = rows
        self.endpoints = {
            endpoint: {
                "dataset_role": "train",
                "image_path_metadata_only": f"synthetic/{endpoint}.png",
                "image_sha256_commitment_only": hashlib.sha256(raw).hexdigest(),
            }
            for endpoint, (raw, _name) in rows.items()
        }
        self.calls: list[tuple[str, str, str, str]] = []

    def read_rgb(
        self, path: str, expected_sha256: str, *, role: str, arm: str, stage: str
    ) -> bytes:
        endpoint = Path(path).stem
        raw = self.rows[endpoint][0]
        assert role == "train"
        assert expected_sha256 == hashlib.sha256(raw).hexdigest()
        self.calls.append((path, role, arm, stage))
        return raw


def _loader(module: Any, inputs: _Inputs) -> Any:
    runtime = SimpleNamespace(Image=Image, np=np, torch=torch)
    return module.NativeAspectDirectBevNarrowLoaderV8(runtime, inputs)


def test_native_loader_decodes_exact_pixels_normalizes_and_preserves_cache_counters() -> None:
    module = _load("_test_v8_native_loader")
    first_raw, first_array = _png()
    second_raw, _ = _png()
    inputs = _Inputs({"first": (first_raw, "first"), "second": (second_raw, "second")})
    loader = _loader(module, inputs)
    observed = loader.image("first", role="train", stage="training", kind="current")
    cached = loader.image("first", role="train", stage="training", kind="next")
    loader.image("second", role="train", stage="training", kind="fixed_negative")
    assert observed.data_ptr() == cached.data_ptr()
    expected = torch.from_numpy(first_array.copy()).permute(2, 0, 1).float() / 255.0
    mean = expected.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = expected.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    torch.testing.assert_close(observed, (expected - mean) / std, rtol=0.0, atol=0.0)
    assert observed.shape == (3, 168, 224) and observed.dtype == torch.float32
    receipt = loader.receipt()
    assert receipt["rgb_request_count"] == {
        "current": 1, "next": 1, "fixed_negative": 1, "endpoint": 0,
    }
    assert receipt["rgb_cache_hit_count"]["next"] == 1
    assert receipt["rgb_physical_read_success_count"]["current"] == 1
    assert receipt["rgb_physical_read_success_count"]["fixed_negative"] == 1
    assert receipt["native_rgb_decode_success_count"] == 2
    assert receipt["native_rgb_decoded_format_count"] == {"PNG": 2}
    assert receipt["resize_crop_pad_call_count"] == 0
    assert len(inputs.calls) == 2
    assert ".resize(" not in inspect.getsource(module.NativeAspectDirectBevNarrowLoaderV8.image)


def test_native_loader_rejects_non_native_dimensions_without_transforming() -> None:
    module = _load("_test_v8_native_loader_reject")
    raw, _ = _png(height=169)
    loader = _loader(module, _Inputs({"bad": (raw, "bad")}))
    with pytest.raises(PermissionError, match="224x168"):
        loader.image("bad", role="train", stage="training", kind="current")
    receipt = loader.receipt()
    assert receipt["native_rgb_size_mismatch_count"] == 1
    assert receipt["native_rgb_decode_success_count"] == 0
    assert receipt["resize_crop_pad_call_count"] == 0


def test_v8_inherits_v4_data_schedule_losses_gate_caps_and_exact_constants() -> None:
    module = _load("_test_v8_executor_bindings")
    for name in (
        "LABEL_ROOT_RELATIVE_PATH", "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256", "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT", "REQUIRED_GPU_NAME", "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER", "ROLE_FILES", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED", "SEMANTIC_DECODER_INITIALIZATION_SEED",
        "EXPERIMENT_SEED", "BOOTSTRAP_SEED", "CONTROL_NAMES", "ALL_ARM_NAMES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(module._v4, name)
    assert module.evaluate_gate_v8 is module._v4.evaluate_gate_v4
    assert module.PREREGISTRATION_COMMIT == "b17599fa1bb49017178f45d0e1a4c83ac8bb9314"
    assert (
        module.NATIVE_IMAGE_HEIGHT_V8, module.NATIVE_IMAGE_WIDTH_V8,
        module.NATIVE_TOKEN_HEIGHT_V8, module.NATIVE_TOKEN_WIDTH_V8,
        module.NATIVE_SPATIAL_TOKEN_COUNT_V8,
        module.NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8,
        module.NATIVE_TOKEN_CELL_RADII_XY_V8,
    ) == (168, 224, 24, 32, 768, 2_845_824, (4.0, 3.0))
    policy = module.native_loader_policy_receipt_v8()
    assert policy["returned_shape_chw"] == [3, 168, 224]
    assert not any(policy[name] for name in ("resize", "crop", "pad", "upscale", "augmentation"))


def test_exact_position_migration_and_inherited_state_receipt() -> None:
    module = _load("_test_v8_executor_migration")
    from lewm.models.encoders import VisionEncoder
    from lewm.models import (
        geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder as model_v4,
    )
    from lewm.models import (
        geometry_anchored_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit as model_v8,
    )

    encoder = VisionEncoder(
        image_size=112, patch_size=7, hidden_dim=192, depth=6,
        n_heads=6, mlp_ratio=4, dropout=0.0,
    )
    state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    candidate = model_v8.GeometryAnchoredSweptProgressSurvivalJointJepaV8(state, masks)
    reference = model_v4.GeometryAnchoredSweptProgressSurvivalJointJepaV4(state, masks)
    receipt = module._migration_receipt_v8(candidate, reference, torch=torch)
    assert receipt["migrated_state_names"] == ["encoder.pos_embed", "target_encoder.pos_embed"]
    assert receipt["all_other_state_tensors_bit_exact"] is True
    assert receipt["positional_parameter_increase"] == 98_304
    assert receipt["bev_lift_parameter_state_bit_exact"] is True
    assert receipt["native_token_cell_radii_xy"] == [4.0, 3.0]
    assert receipt["native_offsets_token_cells"]["bit_exact"] is True
    assert receipt["proposed_normalized_sampling_grid_bit_exact_v4"] is True
    assert receipt["normalized_sampling_grid_and_masks_bit_exact"] is True


def test_executor_validates_frozen_model_and_training_public_apis() -> None:
    module = _load("_test_v8_executor_public_apis")
    model_api = __import__(
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit",
        fromlist=["*"],
    )
    training_v8 = __import__(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v8_native_aspect_high_resolution_vit",
        fromlist=["*"],
    )
    module._validate_model_api_v8(model_api)
    module._validate_training_core_v8(training_v8._v3, training_v8)


def test_output_is_write_once_and_calibration_remains_separate(tmp_path: Path) -> None:
    module = _load("_test_v8_executor_output")
    output = module._fresh_output_root_v8(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="native-aspect-high-resolution-ViT"):
        module._fresh_output_root_v8(tmp_path)
    assert module._physical_calibration_stage_v8(True)["status"] == "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
    assert module._physical_calibration_stage_v8(False)["status"] == "CLOSED_FULL_ARM_GATE_FAILED"


def test_source_executes_v8_once_and_emits_fail_closed_scope() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v8.run_fixed_training_v8(") == 1
    assert "torch.load(" not in source
    assert "execute_v4(" not in source and "execute_v7(" not in source
    assert "GeometryAnchoredSweptProgressSurvivalJointJepaV8(" in source
    assert '"same_bound_rgb_bytes": True' in source
    assert '"input_tensorization_changed": True' in source
    assert '"losses_changed": False' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"checkpoint_access_status": checkpoint_access' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation and trace_write < evaluation
