#!/usr/bin/env python3
"""Run the one fresh persistence-anchored recurrent-H4 JEPA V2 probe.

The reviewed V1 runner owns unchanged custody, schedule, evaluation, receipt,
and cap mechanics.  This thin binding selects the V2 model/output identity and
registers its two targeted auxiliary objectives without duplicating that
boilerplate.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_go2_recurrent_h4_joint_jepa_v1 as core  # noqa: E402


CORE_SOURCE = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"
CORE_SOURCE_SHA256 = "c80dc6c265219363427c88b9c7a82a5fb4c0e4d399b73acc51d27e1f0a847f9b"
CORE_SOURCE_BYTES = 67_915
MODEL_MODULE = (
    "lewm.models.go2_recurrent_h4_persistence_residual_joint_jepa_v2"
)
MODEL_SOURCE = (
    ROOT / "lewm/models/go2_recurrent_h4_persistence_residual_joint_jepa_v2.py"
)
MODEL_SOURCE_SHA256 = "21a0fdb803f598376c01b6a13a16e3966a816ae4ba3eb7eae23d751f1a8c9cdd"
MODEL_SOURCE_BYTES = 9_834
BASE_MODEL_SOURCE = ROOT / "lewm/models/go2_recurrent_h4_joint_jepa.py"
BASE_MODEL_SOURCE_SHA256 = "ddd84561aba5a36df1255ab942bb29db943cc1bf7b0e496ae41b3d1cdc218f55"
BASE_MODEL_SOURCE_BYTES = 21_166
ENCODER_SOURCE = ROOT / "lewm/models/encoders.py"
ENCODER_SOURCE_SHA256 = "5eed7bbe424d5ddd293ea67ed1596e74504c68dd8da93f8420795f216cb7599d"
ENCODER_SOURCE_BYTES = 7_028
OUTPUT_ROOT = (
    ROOT
    / ".generated/go2_recurrent_h4_persistence_residual_joint_jepa_v2/probe_v1"
)
SCHEMA = "lewm_go2_recurrent_h4_persistence_residual_joint_jepa_v2"
PASS_DECISION = "PASS_MAIN_POOL_RECURRENT_H4_PERSISTENCE_RESIDUAL_JOINT_JEPA_V2_PROBE"
STOP_DECISION = "STOP_MAIN_POOL_RECURRENT_H4_PERSISTENCE_RESIDUAL_JOINT_JEPA_V2_PROBE"


def _source_binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    core._read_regular_bound(
        path,
        expected_sha256=sha256,
        expected_bytes=byte_count,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get("LEWM_V2_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_V2_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError("external V2 wrapper byte binding is required") from error
    return {
        "v2_wrapper": _source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "shared_runner": _source_binding(
            CORE_SOURCE, CORE_SOURCE_SHA256, CORE_SOURCE_BYTES
        ),
        "v2_model": _source_binding(
            MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
        ),
        "inherited_v1_model": _source_binding(
            BASE_MODEL_SOURCE, BASE_MODEL_SOURCE_SHA256, BASE_MODEL_SOURCE_BYTES
        ),
        "encoder_dependency": _source_binding(
            ENCODER_SOURCE, ENCODER_SOURCE_SHA256, ENCODER_SOURCE_BYTES
        ),
    }


def _configure_core(
    source_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    core.MODEL_MODULE = MODEL_MODULE
    core.MODEL_SOURCE = MODEL_SOURCE
    core.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    core.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.OBJECTIVE_DESCRIPTION = (
        "V1_joint_JEPA+variance+action; "
        "persistence_ratio_0.90; detached_gate_off_history_margin_0.03"
    )
    core.ADDITIONAL_SCIENCE = {
        "state": "e2_identity_anchor_plus_zero_gated_ordered_history_correction",
        "future": "cumulative_zero_gated_action_residuals",
        "persistence_hinge": "relu(d_real-0.90*d_target_e2_persistence)",
        "history_hinge": (
            "relu(d_real+0.03*d_persistence-d_detached_gate_off_history)"
        ),
        "auxiliary_active_cell": "d_persistence>=1e-4",
        "v1_checkpoint_tensor_open_count": 0,
        "shared_runner_source": {
            "path": str(CORE_SOURCE.relative_to(ROOT)),
            "file_sha256": CORE_SOURCE_SHA256,
            "byte_count": CORE_SOURCE_BYTES,
        },
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 1
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in (source_bindings or {}).items()
    }


_CORE_DECISION = core._decision


def _v2_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    result = _CORE_DECISION(observations, updates_completed)
    selected_update = result["diagnostics"].get("selected_update")
    if selected_update is not None:
        selected = next(
            item for item in observations if item["update"] == selected_update
        )
        result["gates"].update(
            {
                "h4_action_gap_at_least_point20": (
                    selected["aggregate"]["action_gap"][3] >= 0.20
                ),
                "h4_action_gap_bootstrap_lower_at_least_point10": (
                    selected["bootstrap_lower_95"]["action_gap_h4"] >= 0.10
                ),
            }
        )
    else:
        result["gates"].update(
            {
                "h4_action_gap_at_least_point20": False,
                "h4_action_gap_bootstrap_lower_at_least_point10": False,
            }
        )
    result["failed_gates"] = sorted(
        name for name, passed in result["gates"].items() if not passed
    )
    result["decision"] = (
        PASS_DECISION if not result["failed_gates"] else STOP_DECISION
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    _configure_core(source_bindings)
    core._decision = _v2_decision
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
