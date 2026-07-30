from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/launch_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py"
)


def _load_module(name: str):
    import sys

    for private in (
        "_lewm_v24_core_protected_private_v23_launcher",
        "_lewm_v23_scene_action_private_v21_launcher",
        "_lewm_v21_scene_innovation_private_v20_launcher",
    ):
        sys.modules.pop(private, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime(module):
    prefix = np.zeros((4_262, 9), dtype=np.int64)
    prefix[0] = np.arange(9)
    prefix[1] = np.arange(9)[::-1]
    prefix[2] = (np.arange(9) + 2) % 15
    prefix[3] = (np.arange(9)[::-1] + 3) % 15
    prior = prefix.mean(axis=0, dtype=np.float64) * 0.1
    pairs = [{"scene_id": f"scene-{index % 4}"} for index in range(4_262)]
    required_v21 = ("rgb", "scene_innovation_negative_row")
    required_v23 = (*required_v21, module.ACTION_PRIOR_M_KEY_V23)
    training = SimpleNamespace(
        REQUIRED_BATCH_KEYS_V21=required_v21,
        REQUIRED_BATCH_KEYS_V23=required_v23,
        REQUIRED_BATCH_KEYS_V24=required_v23,
        ACTION_PRIOR_M_KEY_V23=module.ACTION_PRIOR_M_KEY_V23,
        CURRENT_RGB_KEY="rgb",
    )
    return SimpleNamespace(
        np=np,
        torch=torch,
        schedule=tuple([0, 1, 2, 3] * 4_000),
        labels={"train": SimpleNamespace(prefix_lengths=prefix)},
        pairs={"train": pairs},
        action_prior_m=prior,
        training_module=training,
    )


def test_denied_shell_retargets_all_lifecycle_selectors(capsys) -> None:
    module = _load_module("_v24_launcher_denial")
    assert module.main([]) == 4
    assert "DENIED_NO_FUTURE_AUTHORITY" in capsys.readouterr().out
    assert module.PREREGISTRATION_COMMIT == (
        "475f1867149f5c5b764973bb5a371de83c29c3eb"
    )
    assert module.V23_SCIENTIFIC_RESULT_COMMIT == (
        "04b0fa48c6c4e10868c2f302bc51100394e3907e"
    )
    module._assert_configured_base_v24()
    assert module._V23_LAUNCHER._build_one_microbatch_v23 is (
        module._build_one_microbatch_v24
    )
    assert module._V21_LAUNCHER._build_one_microbatch_v21 is (
        module._build_one_microbatch_v24
    )
    assert module._BASE_LAUNCHER._build_one_microbatch_v13 is (
        module._build_one_microbatch_v24
    )


def test_v23_preflight_computation_is_republished_only_under_v24_identity() -> None:
    module = _load_module("_v24_launcher_preflight")
    runtime = _runtime(module)
    receipt = module.preflight_schedule_predictor_core_protected_survival_v24(
        runtime
    )
    assert receipt["schema"].startswith(
        "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24_"
    )
    assert "v23" not in receipt["schema"]
    assert receipt["passed"] is True
    assert receipt["microbatch_count"] == 4_000
    assert receipt["scene_zero_microbatch_count"] == 0
    assert receipt["prior_zero_microbatch_count"] == 0
    assert (
        module.preflight_schedule_predictor_core_protected_survival_v24(runtime)
        is receipt
    )
    assert getattr(runtime, module.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V24) is receipt


def test_microbatch_is_bit_exact_v23_schema_with_no_new_field() -> None:
    module = _load_module("_v24_launcher_batch")
    runtime = _runtime(module)
    prior_before = runtime.action_prior_m.copy()
    original = module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21
    module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = lambda **_: {
        "rgb": torch.zeros((4, 3, 2, 2)),
        "scene_innovation_negative_row": torch.tensor((1, 0, 3, 2)),
    }
    try:
        batch = module._build_one_microbatch_v24(
            runtime=runtime, indices=(0, 1, 2, 3), stage="test"
        )
    finally:
        module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = original
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V23
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V24
    prior = batch[module.ACTION_PRIOR_M_KEY_V23]
    assert prior.shape == (9,)
    assert prior.dtype == torch.float32
    assert prior.requires_grad is False
    assert np.array_equal(runtime.action_prior_m, prior_before)
    prior.add_(1.0)
    assert np.array_equal(runtime.action_prior_m, prior_before)


def test_adapter_receipt_adds_no_work_or_execution_authority() -> None:
    module = _load_module("_v24_launcher_receipt")
    receipt = module.private_launcher_adapter_receipt_v24()
    assert receipt["new_batch_fields_over_v23"] == 0
    assert receipt["extra_tensor_payload_reads"] == 0
    assert receipt["extra_predictor_forwards"] == 0
    assert receipt["numeric_comparisons_retained_without_rescoring"] is True
    assert receipt["schedule_preflight_inherited_exactly_from_v23"] is True
    assert receipt["microbatch_builder_inherited_exactly_from_v23"] is True
    assert receipt["public_base_loaded_by_adapter"] is False
    assert receipt["execution_authorized"] is False
