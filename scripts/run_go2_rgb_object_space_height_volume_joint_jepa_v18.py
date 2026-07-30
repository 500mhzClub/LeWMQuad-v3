#!/usr/bin/env python3
"""Source-only V18 adapter for the reviewed V13 joint-training core.

The losses, optimizer, clipping, microbatch schedule, accounting, and EMA
ordering remain the reviewed V13 implementation.  V18 changes only the exact
parameter names assigned to the representation route because its sole JEPA
state is an object-space height volume rather than the retired FREE/OCCUPIED
plane projections.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = (
    ROOT
    / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v13_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

REPRESENTATION_PARAMETER_PREFIXES_V18 = (
    "bev_lift.point_projection.",
    "bev_lift.volume_block.",
    "semantic_head.",
)


def _load_private_base_training_v18() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("reviewed V13 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if not source:
        raise RuntimeError("reviewed V13 training source is empty")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V13 training module name is already occupied")
    module = ModuleType(PRIVATE_BASE_MODULE_NAME)
    module.__file__ = str(BASE_TRAINING_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(BASE_TRAINING_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_BASE_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_BASE_MODULE_NAME)
    return module


_base = _load_private_base_training_v18()
if (
    _base.MICROBATCH_SIZE != 4
    or _base.MICROBATCHES_PER_UPDATE != 4
    or _base.PRESENTATIONS_PER_UPDATE != 16
    or _base.MAXIMUM_UPDATES != 1_000
    or _base.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("reviewed V13 training cap or batching changed")


def partition_parameters_v18(model: Any) -> Any:
    """Resolve V18's exact Camera, volume, predictor, and EMA routes."""

    groups: dict[str, list[Any]] = {
        "encoder": [],
        "evidence_head": [],
        "representation": [],
        "predictor": [],
        "target": [],
    }
    names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith("bev_lift.evidence_head."):
            group = "evidence_head"
        elif name.startswith(REPRESENTATION_PARAMETER_PREFIXES_V18):
            group = "representation"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith(
            (
                "target_encoder.",
                "target_bev_lift.evidence_head.",
                "target_bev_lift.point_projection.",
                "target_bev_lift.volume_block.",
            )
        ):
            group = "target"
        else:
            raise RuntimeError(f"unregistered V18 model parameter {name!r}")
        groups[group].append(parameter)
        names[group].append(name)

    if any(not values for values in groups.values()):
        raise RuntimeError("V18 parameter partition contains an empty role")
    identities = [id(value) for values in groups.values() for value in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("V18 parameter partition overlaps")
    if set(identities) != {id(value) for value in model.parameters()}:
        raise RuntimeError("V18 parameter partition does not cover the model")
    if any(value.requires_grad for value in groups["target"]):
        raise RuntimeError("V18 EMA target parameter is trainable")
    if any(
        not value.requires_grad or str(value.dtype) != "torch.float32"
        for group in ("encoder", "evidence_head", "representation", "predictor")
        for value in groups[group]
    ):
        raise RuntimeError("every V18 online parameter must be trainable float32")
    return _base.ParameterPartitionV13(
        **{name: tuple(values) for name, values in groups.items()},
        names={name: tuple(values) for name, values in names.items()},
    )


# Re-export the reviewed tensor core.  Its function globals remain confined to
# the private module, where only the registered partition hook is replaced.
for _name in _base.__all__:
    globals()[_name] = getattr(_base, _name)
_base.partition_parameters_v13 = partition_parameters_v18
partition_parameters_v13 = partition_parameters_v18
joint_training_update_v13 = _base.joint_training_update_v13
build_frozen_optimizer_v13 = _base.build_frozen_optimizer_v13
validate_optimizer_v13 = _base.validate_optimizer_v13
validate_accounting_v13 = _base.validate_accounting_v13
build_frozen_optimizer_v18 = build_frozen_optimizer_v13
validate_optimizer_v18 = validate_optimizer_v13
joint_training_update_v18 = joint_training_update_v13
validate_accounting_v18 = validate_accounting_v13


def private_training_adapter_receipt_v18() -> dict[str, Any]:
    return {
        "schema": "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_training_adapter_v1",
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "representation_parameter_prefixes": REPRESENTATION_PARAMETER_PREFIXES_V18,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


__all__ = tuple(
    dict.fromkeys(
        (
            *_base.__all__,
            "REPRESENTATION_PARAMETER_PREFIXES_V18",
            "build_frozen_optimizer_v18",
            "joint_training_update_v18",
            "partition_parameters_v18",
            "private_training_adapter_receipt_v18",
            "validate_accounting_v18",
            "validate_optimizer_v18",
        )
    )
)
