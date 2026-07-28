#!/usr/bin/env python3
"""Run the science-identical factual-transition V2 schedule replacement.

This wrapper reuses the frozen V1 executable science.  It changes only the
bound train/validation index identity, row schema, output/receipt identity,
and final PASS/STOP labels required by the causal schedule-integrity repair.
Importing this module is source-only and opens no runtime input.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1
    as v1,
)


core = v1.core

V1_RUNNER_SOURCE = ROOT / (
    "scripts/"
    "run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py"
)
V1_RUNNER_SOURCE_SHA256 = (
    "693cbea45b2a49f0f3edfb7cabce347b852a67af78df1ecf5462c65be48cd977"
)
V1_RUNNER_SOURCE_BYTES = 34_730
V2_ADAPTER_SOURCE = ROOT / "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py"
V2_ADAPTER_SOURCE_SHA256 = (
    "3d49e710304ad685f9d161a84586229a6036b652f84df877772afe5b827c51ea"
)
V2_ADAPTER_SOURCE_BYTES = 21_001
V2_BUILDER_SOURCE = ROOT / "scripts/build_go2_recurrent_h4_rgb_index_v2.py"
V2_BUILDER_SOURCE_SHA256 = (
    "6d4dc0ad8626e53ab36d170d8b5d5d33af0a0c30cf68ad11ed34e6eb23831ce4"
)
V2_BUILDER_SOURCE_BYTES = 7_995

INDEX_ROOT = ROOT / (
    ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
)
TRAIN_INDEX = INDEX_ROOT / "train.jsonl"
TRAIN_INDEX_SHA256 = (
    "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
)
TRAIN_INDEX_BYTES = 10_328_000
VAL_INDEX = INDEX_ROOT / "val.jsonl"
VAL_INDEX_SHA256 = (
    "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
)
VAL_INDEX_BYTES = 1_317_888
INDEX_MANIFEST = INDEX_ROOT / "manifest.json"
INDEX_MANIFEST_SHA256 = (
    "d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e"
)
INDEX_MANIFEST_BYTES = 26_926
INDEX_ROW_SCHEMA = (
    "lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
)

OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_"
    "schedule_integrity/probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_"
    "v2_schedule_integrity"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_TRAJECTORY_"
    "H4_JEPA_V2_SCHEDULE_INTEGRITY"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_TRAJECTORY_"
    "H4_JEPA_V2_SCHEDULE_INTEGRITY"
)

_V1_DECISION = v1._factual_shared_transition_decision


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get(
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_"
        "WRAPPER_SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_"
        "WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external V2 schedule-integrity wrapper binding is required"
        ) from error

    source_binding = v1.base._source_binding
    return {
        "factual_shared_transition_v2_schedule_integrity_wrapper": (
            source_binding(Path(__file__).resolve(), wrapper_sha256, wrapper_bytes)
        ),
        "factual_shared_transition_v1_runner": source_binding(
            V1_RUNNER_SOURCE,
            V1_RUNNER_SOURCE_SHA256,
            V1_RUNNER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_adapter": source_binding(
            V2_ADAPTER_SOURCE,
            V2_ADAPTER_SOURCE_SHA256,
            V2_ADAPTER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_builder": source_binding(
            V2_BUILDER_SOURCE,
            V2_BUILDER_SOURCE_SHA256,
            V2_BUILDER_SOURCE_BYTES,
        ),
        "trajectory_h4_wrapper_dependency": source_binding(
            v1.BASE_WRAPPER_SOURCE,
            v1.BASE_WRAPPER_SOURCE_SHA256,
            v1.BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": source_binding(
            v1.base.CORE_SOURCE,
            v1.base.CORE_SOURCE_SHA256,
            v1.base.CORE_SOURCE_BYTES,
        ),
        "factual_shared_transition_trajectory_h4_model": source_binding(
            v1.MODEL_SOURCE,
            v1.MODEL_SOURCE_SHA256,
            v1.MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": source_binding(
            v1.TRAJECTORY_MODEL_SOURCE,
            v1.TRAJECTORY_MODEL_SOURCE_SHA256,
            v1.TRAJECTORY_MODEL_SOURCE_BYTES,
        ),
        "local_innovation_trajectory_h4_model_dependency": source_binding(
            v1.LOCAL_INNOVATION_MODEL_SOURCE,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_SHA256,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_BYTES,
        ),
        "dense_h4_model_dependency": source_binding(
            v1.base.DENSE_MODEL_SOURCE,
            v1.base.DENSE_MODEL_SOURCE_SHA256,
            v1.base.DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": source_binding(
            v1.base.BASE_MODEL_SOURCE,
            v1.base.BASE_MODEL_SOURCE_SHA256,
            v1.base.BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": source_binding(
            v1.base.ENCODER_SOURCE,
            v1.base.ENCODER_SOURCE_SHA256,
            v1.base.ENCODER_SOURCE_BYTES,
        ),
    }


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    """Install V1 science, then change only V2 identity and index bindings."""

    v1._configure_core(source_bindings)
    core.TRAIN_INDEX = TRAIN_INDEX
    core.TRAIN_INDEX_SHA256 = TRAIN_INDEX_SHA256
    core.TRAIN_INDEX_BYTES = TRAIN_INDEX_BYTES
    core.VAL_INDEX = VAL_INDEX
    core.VAL_INDEX_SHA256 = VAL_INDEX_SHA256
    core.VAL_INDEX_BYTES = VAL_INDEX_BYTES
    core.INDEX_ROW_SCHEMA = INDEX_ROW_SCHEMA
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }
    science = dict(core.ADDITIONAL_SCIENCE)
    science["schedule_integrity"] = {
        "replacement": "science_identical_v1_model_and_objective",
        "row_schema": INDEX_ROW_SCHEMA,
        "alignment": (
            "same_episode_previous_block_fifth_tick_to_requested_block_fifth_"
            "tick_no_destination_action_tick"
        ),
        "requested_command_action_semantics": True,
        "train_index": {
            "path": str(TRAIN_INDEX.relative_to(ROOT)),
            "file_sha256": TRAIN_INDEX_SHA256,
            "byte_count": TRAIN_INDEX_BYTES,
            "row_count": core.PRESENTATIONS,
        },
        "val_index": {
            "path": str(VAL_INDEX.relative_to(ROOT)),
            "file_sha256": VAL_INDEX_SHA256,
            "byte_count": VAL_INDEX_BYTES,
            "row_count": core.VAL_PRESENTATIONS,
        },
        "manifest": {
            "path": str(INDEX_MANIFEST.relative_to(ROOT)),
            "file_sha256": INDEX_MANIFEST_SHA256,
            "byte_count": INDEX_MANIFEST_BYTES,
        },
    }
    core.ADDITIONAL_SCIENCE = science


def _schedule_integrity_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    """Reuse every V1 gate and replace only terminal identity/authority text."""

    result = dict(_V1_DECISION(observations, updates_completed))
    failed_gates = result.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise core.ContractError("V1 factual decision failure list changed")
    expected_v1 = v1.PASS_DECISION if not failed_gates else v1.STOP_DECISION
    if result.get("decision") != expected_v1:
        raise core.ContractError("V1 factual decision identity disagrees with its gates")
    result["decision"] = PASS_DECISION if not failed_gates else STOP_DECISION
    result["authority"] = (
        "A pass establishes bounded development evidence for the unchanged "
        "factual shared-transition trajectory JEPA on candidate-valid V2 "
        "requested-action boundaries only; it grants no checkpoint access, "
        "navigation, held-out access, scale promotion, or deployment "
        "authority. A stop closes this exact one-shot schedule-integrity "
        "replacement without retry or resume."
    )
    return result


def _install_runtime_adapters() -> None:
    if core._decision is _schedule_integrity_decision:
        if (
            core._evaluate is not v1._factual_shared_transition_evaluate
            or core._run is not v1._factual_shared_transition_run
        ):
            raise core.ContractError("V2 runtime handler identity changed")
        return
    v1._install_runtime_adapters()
    if core._evaluate is not v1._factual_shared_transition_evaluate:
        raise core.ContractError("V1 factual evaluator was not preserved")
    if core._run is not v1._factual_shared_transition_run:
        raise core.ContractError("V1 factual run handler was not preserved")
    if core._decision is not _V1_DECISION:
        raise core.ContractError("V1 factual decision was not installed")
    core._decision = _schedule_integrity_decision


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != v1.base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(v1.__file__).resolve() != V1_RUNNER_SOURCE:
        raise core.ContractError("frozen V1 runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    v1.base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
