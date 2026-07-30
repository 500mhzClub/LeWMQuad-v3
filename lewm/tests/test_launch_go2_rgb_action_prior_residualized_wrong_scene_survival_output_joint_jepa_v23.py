from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)


def _load_module(name: str = "_v23_launcher_test"):
    import sys

    for private in (
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
    pairs = [
        {"scene_id": f"scene-{index % 4}"}
        for index in range(4_262)
    ]
    training = SimpleNamespace(
        REQUIRED_BATCH_KEYS_V21=("rgb", "scene_innovation_negative_row"),
        REQUIRED_BATCH_KEYS_V23=(
            "rgb",
            "scene_innovation_negative_row",
            module.ACTION_PRIOR_M_KEY_V23,
        ),
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


def test_denied_shell_and_frozen_identity(capsys) -> None:
    module = _load_module("_v23_launcher_denial")
    assert module.main([]) == 4
    output = capsys.readouterr().out
    assert "DENIED_NO_FUTURE_AUTHORITY" in output
    assert module.PREREGISTRATION_COMMIT == "a7cf9692dd93212a82cb598d3175ff1c3598941b"
    assert module.V22_SCIENTIFIC_RESULT_COMMIT == (
        "f184a41ac99b1c66ea4db1e0b0a0845f23b48bbd"
    )
    assert module._BASE_LAUNCHER._build_one_microbatch_v13 is (
        module._build_one_microbatch_v23
    )
    assert module._V21_LAUNCHER._build_one_microbatch_v21 is (
        module._build_one_microbatch_v23
    )


def test_full_schedule_preflight_has_both_axes_and_is_cached() -> None:
    module = _load_module("_v23_launcher_preflight")
    runtime = _runtime(module)
    receipt = module.preflight_schedule_state_residual_survival_v23(runtime)
    assert receipt["passed"] is True
    assert receipt["microbatch_count"] == 4_000
    assert receipt["scene_zero_microbatch_count"] == 0
    assert receipt["prior_zero_microbatch_count"] == 0
    assert receipt["scene_eligible_count_min"] > 0
    assert receipt["prior_eligible_count_min"] > 0
    assert module.preflight_schedule_state_residual_survival_v23(runtime) is receipt
    runtime.action_prior_m[0] += 0.1
    with pytest.raises(RuntimeError, match="cached schedule preflight changed"):
        module.preflight_schedule_state_residual_survival_v23(runtime)


def test_microbatch_appends_one_detached_prior_without_mutating_runtime() -> None:
    module = _load_module("_v23_launcher_batch")
    runtime = _runtime(module)
    prior_before = runtime.action_prior_m.copy()
    original = module._INHERITED_BUILD_ONE_MICROBATCH_V21
    module._INHERITED_BUILD_ONE_MICROBATCH_V21 = lambda **_: {
        "rgb": torch.zeros((4, 3, 2, 2)),
        "scene_innovation_negative_row": torch.tensor([1, 0, 3, 2]),
    }
    try:
        batch = module._build_one_microbatch_v23(
            runtime=runtime, indices=(0, 1, 2, 3), stage="test"
        )
    finally:
        module._INHERITED_BUILD_ONE_MICROBATCH_V21 = original
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V23
    prior = batch[module.ACTION_PRIOR_M_KEY_V23]
    assert prior.shape == (9,)
    assert prior.dtype == torch.float32
    assert prior.requires_grad is False
    assert np.array_equal(runtime.action_prior_m, prior_before)
    prior.add_(1.0)
    assert np.array_equal(runtime.action_prior_m, prior_before)


def test_adapter_receipt_adds_no_forward_or_payload_read() -> None:
    module = _load_module("_v23_launcher_receipt")
    receipt = module.private_launcher_adapter_receipt_v23()
    assert receipt["new_batch_fields_over_v22"] == 1
    assert receipt["extra_tensor_payload_reads"] == 0
    assert receipt["extra_predictor_forwards"] == 0
    assert receipt["numeric_comparisons_retained_without_rescoring"] is True
