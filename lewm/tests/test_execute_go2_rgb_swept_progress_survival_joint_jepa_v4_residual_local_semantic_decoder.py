from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT
    / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _StubInheritedModel(torch.nn.Module):
    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _StubDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Conv2d(64, 3, kernel_size=1, bias=True)
        self.local = torch.nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.activation = torch.nn.GELU()
        self.residual_output = torch.nn.Conv2d(64, 3, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.residual_output.weight)
        torch.nn.init.zeros_(self.residual_output.bias)


class _StubBevLift(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mask = torch.zeros((64, 64), dtype=torch.bool)
        mask[4:60, 3:61] = True
        self.register_buffer("anchor_in_frustum", mask, persistent=True)


class _StubV4Model(_StubInheritedModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(initialization_seed=20_260_712)
        self.semantic_head = _StubDecoder()
        self.bev_lift = _StubBevLift()


def _stub_partition(model: _StubV4Model) -> Any:
    encoder = torch.nn.Parameter(torch.ones(()))
    predictor = torch.nn.Parameter(torch.ones(()))
    target = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    return SimpleNamespace(
        encoder=(encoder,),
        lift_semantic=tuple(model.semantic_head.parameters()),
        predictor=(predictor,),
        target=(target,),
    )


def test_v4_reuses_all_frozen_v3_runtime_bindings() -> None:
    module = _load("_test_v4_executor_bindings")
    predecessor = module._v3
    for name in (
        "LABEL_ROOT_RELATIVE_PATH",
        "LABEL_MANIFEST_NAME",
        "LABEL_MANIFEST_CONTENT_SHA256",
        "LABEL_MANIFEST_FILE_SHA256",
        "LABEL_MANIFEST_BYTE_COUNT",
        "REQUIRED_GPU_NAME",
        "REQUIRED_GPU_MEMORY_BYTES",
        "ACTION_ORDER",
        "ROLE_FILES",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "CONSTRUCTOR_INITIALIZATION_SEED",
        "EXPERIMENT_SEED",
        "BOOTSTRAP_SEED",
        "CONTROL_NAMES",
        "ALL_ARM_NAMES",
        "GATE_THRESHOLDS",
    ):
        assert getattr(module, name) == getattr(predecessor, name)
    assert module.AUXILIARY_OBJECTIVE == predecessor.AUXILIARY_OBJECTIVE
    assert module.AUXILIARY_OBJECTIVE["coefficient"] == 0.5
    assert module.OUTPUT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder/"
        "attempt_v1"
    )


def test_architecture_receipt_is_exact_and_detached() -> None:
    module = _load("_test_v4_executor_architecture")
    receipt = module.semantic_decoder_architecture_receipt_v4()
    assert receipt["merge"] == "base_logits_plus_residual_logits"
    assert receipt["base"] == {
        "type": "Conv2d",
        "in_channels": 64,
        "out_channels": 3,
        "kernel_size": [1, 1],
        "bias": True,
        "identity": "exact_existing_v3_semantic_head",
    }
    assert receipt["residual"]["local"] == {
        "type": "Conv2d",
        "in_channels": 64,
        "out_channels": 64,
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1],
        "bias": True,
    }
    assert receipt["residual"]["activation"] == {
        "type": "GELU",
        "approximate": "none",
    }
    assert receipt["residual"]["output"]["weight_initialization"] == (
        "exact_zeros"
    )
    assert receipt["residual"]["output"]["bias_initialization"] == "exact_zeros"
    assert receipt["added_trainable_parameter_count"] == 37_123
    assert receipt["initialization_seed"] == 20_260_713
    assert receipt["visibility_mask"] == (
        "inherited_bev_lift_anchor_in_frustum_post_logits"
    )
    receipt["residual"]["local"]["out_channels"] = 1
    assert module.SEMANTIC_DECODER_ARCHITECTURE["residual"]["local"][
        "out_channels"
    ] == 64


def test_stubbed_initial_decoder_runtime_receipt_proves_partition_and_mask() -> None:
    module = _load("_test_v4_executor_stub_model")
    model = _StubV4Model()
    receipt = module._initial_decoder_receipt_v4(
        model,
        _stub_partition(model),
        torch=torch,
        inherited_semantic_method=_StubInheritedModel.semantic_logits_from_latent,
    )
    assert receipt["initial_residual_output_exactly_zero"] is True
    assert receipt["added_parameter_count"] == 37_123
    assert receipt["semantic_parameter_count"] == 37_318
    assert receipt["all_semantic_parameters_in_lift_semantic_exactly_once"] is True
    assert receipt["visibility_mask"]["shape"] == [64, 64]
    assert receipt["visibility_mask"]["dtype"] == "bool"
    assert receipt["visibility_mask"]["true_cell_count"] == 56 * 58
    assert len(receipt["visibility_mask"]["sha256"]) == 64
    assert receipt["visibility_mask"]["application"] == "inherited_post_logits"


def test_stubbed_decoder_runtime_guard_rejects_zero_init_or_partition_drift() -> None:
    module = _load("_test_v4_executor_stub_rejections")
    model = _StubV4Model()
    with torch.no_grad():
        model.semantic_head.residual_output.weight[0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="not exactly zero initialized"):
        module._initial_decoder_receipt_v4(
            model,
            _stub_partition(model),
            torch=torch,
            inherited_semantic_method=_StubInheritedModel.semantic_logits_from_latent,
        )

    model = _StubV4Model()
    partition = _stub_partition(model)
    partition.lift_semantic = partition.lift_semantic[:-1]
    with pytest.raises(RuntimeError, match="escaped the lift/semantic partition"):
        module._initial_decoder_receipt_v4(
            model,
            partition,
            torch=torch,
            inherited_semantic_method=_StubInheritedModel.semantic_logits_from_latent,
        )


def test_stubbed_training_core_guard_freezes_v3_half_aux_and_caps() -> None:
    module = _load("_test_v4_executor_stub_core")

    class V1:
        ACTION_ORDER = module.ACTION_ORDER
        MICROBATCH_SIZE = module.MICROBATCH_SIZE
        MICROBATCHES_PER_UPDATE = module.MICROBATCHES_PER_UPDATE
        PRESENTATIONS_PER_UPDATE = module.PRESENTATIONS_PER_UPDATE
        MAXIMUM_UPDATES = module.MAXIMUM_UPDATES
        MAXIMUM_PRESENTATIONS = module.MAXIMUM_PRESENTATIONS

    class V3(V1):
        OCCUPIED_CLASS_INDEX = 2
        OCCUPIED_SAFETY_AUX_COEFFICIENT = 0.5
        OCCUPIED_SAFETY_AUX_NORMALIZATION = module.math.log(2.0)

        @staticmethod
        def run_fixed_training_v3() -> None:
            return None

    module._validate_training_core_v4(V1, V3)
    V3.OCCUPIED_SAFETY_AUX_COEFFICIENT = 0.25
    with pytest.raises(PermissionError):
        module._validate_training_core_v4(V1, V3)


def test_stubbed_model_api_guard_freezes_class_and_constants() -> None:
    module = _load("_test_v4_executor_stub_model_api")
    stub = SimpleNamespace(
        RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4=37_123,
        RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4=1,
        GeometryAnchoredSweptProgressSurvivalJointJepaV4=lambda: None,
    )
    module._validate_model_api_v4(stub)
    stub.RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4 = 2
    with pytest.raises(PermissionError, match="model API"):
        module._validate_model_api_v4(stub)


def test_output_root_is_write_once_and_v4_schemas_are_distinct(tmp_path: Path) -> None:
    module = _load("_test_v4_executor_output")
    output = module._fresh_output_root_v4(tmp_path)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="fresh residual-local-decoder"):
        module._fresh_output_root_v4(tmp_path)
    schemas = {
        module.CHECKPOINT_SCHEMA,
        module.TRACE_SCHEMA,
        module.RESULT_SCHEMA,
        module.FAILURE_SCHEMA,
    }
    assert len(schemas) == 4
    assert all("v4_residual_local_semantic_decoder" in value for value in schemas)


def test_source_has_one_v3_run_call_and_no_predecessor_artifact_access() -> None:
    source = ENTRYPOINT.read_text()
    assert source.count("training_v3.run_fixed_training_v3(") == 1
    assert "torch.load(" not in source
    assert "execute_v3(" not in source
    for forbidden in (
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux/",
        ".generated/go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux/",
    ):
        assert forbidden not in source
    assert (
        'model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4('
        in source
    )
    assert '"initialization_source": "exact_n320_encoder_only"' in source
    assert '"predecessor_experiment_checkpoint_read": False' in source
    assert '"losses_changed": False' in source
    assert '"retry_or_resume_authorized": False' in source
    assert '"heldout_or_sealed_opened": False' in source
    assert '"status": "STAGED_ONLY_IF_FULL_ARM_PASSES"' in source
    assert '"must_use_identical_v4_decoder": True' in source
    checkpoint_write = source.index('output / "checkpoint_update_1000.pt"')
    trace_write = source.index('output / "training_trace.json"')
    evaluation = source.index("_v1.score_role_v1(")
    assert checkpoint_write < evaluation
    assert trace_write < evaluation
