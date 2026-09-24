from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


BASE_NAME = (
    "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck"
)
V18_NAME = "scripts.run_go2_rgb_object_space_height_volume_joint_jepa_v18"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _NamedParameterModel:
    def __init__(self, *, include_unknown: bool = False) -> None:
        names = [
            "encoder.weight",
            "bev_lift.evidence_head.weight",
            "bev_lift.point_projection.weight",
            "bev_lift.volume_block.conv1.weight",
            "semantic_head.output.weight",
            "predictor.weight",
            "target_encoder.weight",
            "target_bev_lift.evidence_head.weight",
            "target_bev_lift.point_projection.weight",
            "target_bev_lift.volume_block.conv1.weight",
        ]
        if include_unknown:
            names.append("unregistered.weight")
        self._named = []
        for name in names:
            parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            if name.startswith("target_"):
                parameter.requires_grad_(False)
            self._named.append((name, parameter))

    def named_parameters(self):
        return iter(self._named)

    def parameters(self):
        return (parameter for _, parameter in self._named)


class _SyntheticTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _SyntheticTorch:
    Tensor = _SyntheticTensor

    @staticmethod
    def equal(left: object, right: object) -> bool:
        return left is right


def _synthetic_microbatch(
    adapter,
    visibility: _SyntheticTensor,
) -> dict[str, object]:
    size = adapter.MICROBATCH_SIZE
    batch = {
        key: _SyntheticTensor((size,))
        for key in adapter.REQUIRED_BATCH_KEYS
    }
    batch.update(
        {
            adapter.CURRENT_CAMERA_ORIGIN_KEY: _SyntheticTensor((size, 3)),
            adapter.NEXT_CAMERA_ORIGIN_KEY: _SyntheticTensor((size, 3)),
            adapter.CURRENT_CAMERA_BASIS_KEY: _SyntheticTensor((size, 3, 3)),
            adapter.NEXT_CAMERA_BASIS_KEY: _SyntheticTensor((size, 3, 3)),
            adapter.CURRENT_GROUND_PLANE_Z_KEY: _SyntheticTensor((size,)),
            adapter.NEXT_GROUND_PLANE_Z_KEY: _SyntheticTensor((size,)),
            adapter.IMMEDIATE_FEASIBLE_KEY: _SyntheticTensor((size, 9)),
            adapter.PREFIX_LENGTHS_KEY: _SyntheticTensor((size, 9)),
            adapter.CURRENT_GROUND_IN_FRUSTUM_KEY: visibility,
            adapter.NEXT_GROUND_IN_FRUSTUM_KEY: visibility,
        }
    )
    return batch


def test_private_adapter_does_not_mutate_public_v13_training() -> None:
    public = importlib.import_module(BASE_NAME)
    original_partition = public.partition_parameters_v13
    sys.modules.pop(V18_NAME, None)
    adapter = importlib.import_module(V18_NAME)

    assert public.partition_parameters_v13 is original_partition
    assert adapter._base is not public
    assert adapter.PRIVATE_BASE_MODULE_NAME not in sys.modules
    assert adapter.joint_training_update_v13.__globals__ is adapter._base.__dict__
    assert adapter.joint_training_update_v13.__globals__["partition_parameters_v13"] is (
        adapter.partition_parameters_v18
    )


def test_v18_partition_covers_each_registered_route_once() -> None:
    adapter = importlib.import_module(V18_NAME)
    partition = adapter.partition_parameters_v18(_NamedParameterModel())

    assert partition.names["encoder"] == ("encoder.weight",)
    assert partition.names["evidence_head"] == (
        "bev_lift.evidence_head.weight",
    )
    assert partition.names["representation"] == (
        "bev_lift.point_projection.weight",
        "bev_lift.volume_block.conv1.weight",
        "semantic_head.output.weight",
    )
    assert partition.names["predictor"] == ("predictor.weight",)
    assert partition.names["target"] == (
        "target_encoder.weight",
        "target_bev_lift.evidence_head.weight",
        "target_bev_lift.point_projection.weight",
        "target_bev_lift.volume_block.conv1.weight",
    )
    identities = [id(value) for value in (*partition.online, *partition.target)]
    assert len(identities) == len(set(identities)) == 10


def test_v18_partition_fails_closed_on_an_unknown_parameter() -> None:
    adapter = importlib.import_module(V18_NAME)
    with pytest.raises(RuntimeError, match="unregistered V18 model parameter"):
        adapter.partition_parameters_v18(_NamedParameterModel(include_unknown=True))


def test_training_adapter_preserves_caps_batch_schema_and_denial_receipt() -> None:
    adapter = importlib.import_module(V18_NAME)
    receipt = adapter.private_training_adapter_receipt_v18()
    assert isinstance(receipt.pop("public_base_was_loaded_before_adapter"), bool)
    assert receipt == {
        "schema": (
            "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
            "training_adapter_v1"
        ),
        "base_training": (
            "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
            "camera_evidence_bottleneck.py"
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": False,
        "representation_parameter_prefixes": (
            "bev_lift.point_projection.",
            "bev_lift.volume_block.",
            "semantic_head.",
        ),
        "maximum_updates": 1_000,
        "maximum_presentations": 16_000,
    }
    assert adapter.MICROBATCH_SIZE == 4
    assert adapter.MICROBATCHES_PER_UPDATE == 4
    assert adapter.PRESENTATIONS_PER_UPDATE == 16
    assert adapter.MAXIMUM_UPDATES == 1_000
    assert adapter.MAXIMUM_PRESENTATIONS == 16_000
    assert tuple(adapter.REQUIRED_BATCH_KEYS) == tuple(adapter._base.REQUIRED_BATCH_KEYS)


def test_private_microbatch_validator_bridge_reaches_inherited_engine_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = importlib.import_module(V18_NAME)
    executor = importlib.import_module(
        "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
    )
    assert (
        adapter._validate_microbatches_v13
        is adapter._base._validate_microbatches_v13
    )
    assert adapter._validate_microbatches_v13.__globals__ is adapter._base.__dict__
    assert "_validate_microbatches_v13" not in adapter.__all__

    public_surface = {
        "partition_parameters_v13",
        "build_frozen_optimizer_v13",
        "validate_optimizer_v13",
        "joint_training_update_v13",
        "validate_accounting_v13",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "REQUIRED_BATCH_KEYS",
        "CURRENT_CAMERA_BASIS_KEY",
        "CURRENT_CAMERA_ORIGIN_KEY",
        "CURRENT_GROUND_CLEAR_KEY",
        "CURRENT_GROUND_IN_FRUSTUM_KEY",
        "CURRENT_GROUND_PLANE_Z_KEY",
        "CURRENT_PIXEL_DISTANCE_KEY",
        "CURRENT_PIXEL_HIT_KEY",
        "NEXT_CAMERA_BASIS_KEY",
        "NEXT_CAMERA_ORIGIN_KEY",
        "NEXT_GROUND_CLEAR_KEY",
        "NEXT_GROUND_IN_FRUSTUM_KEY",
        "NEXT_GROUND_PLANE_Z_KEY",
        "NEXT_PIXEL_DISTANCE_KEY",
        "NEXT_PIXEL_HIT_KEY",
    }
    complete_surface = {*public_surface, "_validate_microbatches_v13"}
    assert len(public_surface) == 25
    assert len(complete_surface) == 26
    assert all(hasattr(adapter, name) for name in complete_surface)
    assert public_surface.issubset(adapter.__all__)
    assert all(
        callable(getattr(adapter, name))
        for name in (
            "partition_parameters_v13",
            "build_frozen_optimizer_v13",
            "validate_optimizer_v13",
            "joint_training_update_v13",
            "validate_accounting_v13",
            "_validate_microbatches_v13",
        )
    )
    executor._engine.validate_training_api_v13(adapter)

    monkeypatch.setitem(sys.modules, "torch", _SyntheticTorch)
    size = adapter.MICROBATCH_SIZE
    visibility = _SyntheticTensor((size,))
    runtime = SimpleNamespace(torch=_SyntheticTorch, training_module=adapter)
    model = SimpleNamespace(
        bev_lift=SimpleNamespace(
            evidence_head=SimpleNamespace(
                ground_query_geometry=lambda *_args: SimpleNamespace(
                    in_frustum=visibility
                )
            )
        )
    )
    microbatches = [
        _synthetic_microbatch(adapter, visibility)
        for _ in range(adapter.MICROBATCHES_PER_UPDATE)
    ]
    executor._engine._validate_microbatches_for_engine_v13(
        runtime, model, microbatches
    )

    malformed = [dict(batch) for batch in microbatches]
    malformed[0][adapter.CURRENT_CAMERA_ORIGIN_KEY] = _SyntheticTensor(
        (size, 2)
    )
    with pytest.raises(ValueError, match="current_camera_origin_body_m"):
        executor._engine._validate_microbatches_for_engine_v13(
            runtime, model, malformed
        )
