#!/usr/bin/env python3
"""Source-only training-cap adapter for V15 extended-horizon joint JEPA.

The exact V13/V14 training tensor core is executed in a private module
namespace.  Only its two terminal accounting caps are extended; every model,
optimizer, loss, gradient-route, EMA, and batch operation remains the frozen
implementation.  Import performs no data discovery, experiment I/O, or
accelerator selection.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v13_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

MAXIMUM_UPDATES_V15 = 2_000
MAXIMUM_PRESENTATIONS_V15 = 32_000


def _load_private_base_training_v15() -> ModuleType:
    path = BASE_TRAINING_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("reviewed V13 training core is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("reviewed V13 training core escaped or is not regular")
    source = path.read_bytes()
    if not source:
        raise RuntimeError("reviewed V13 training core is empty")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V15 training module name is already occupied")
    module = ModuleType(PRIVATE_BASE_MODULE_NAME)
    module.__file__ = str(path)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_BASE_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_BASE_MODULE_NAME)
    return module


_training = _load_private_base_training_v15()
if (
    getattr(_training, "MAXIMUM_UPDATES", None) != 1_000
    or getattr(_training, "MAXIMUM_PRESENTATIONS", None) != 16_000
    or getattr(_training, "PRESENTATIONS_PER_UPDATE", None) != 16
    or getattr(_training, "MICROBATCH_SIZE", None) != 4
    or getattr(_training, "MICROBATCHES_PER_UPDATE", None) != 4
):
    raise RuntimeError("reviewed V13 training schedule defaults changed")
if not isinstance(getattr(_training, "__all__", None), list):
    raise RuntimeError("reviewed V13 training export surface changed")

_training.MAXIMUM_UPDATES = MAXIMUM_UPDATES_V15
_training.MAXIMUM_PRESENTATIONS = MAXIMUM_PRESENTATIONS_V15

# Re-export the exact frozen tensor-core surface.  Function globals continue
# to resolve against only the private module whose accounting caps were
# changed above.
for _name in _training.__all__:
    globals()[_name] = getattr(_training, _name)

__all__ = tuple(_training.__all__)


def private_training_adapter_receipt_v15() -> dict[str, Any]:
    """Describe the source-only isolation and the sole adapted constants."""

    return {
        "schema": (
            "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
            "private_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "public_base_was_loaded_before_adapter": (
            _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "presentations_per_update": PRESENTATIONS_PER_UPDATE,
        "scientific_change": "terminal_accounting_caps_only",
    }

