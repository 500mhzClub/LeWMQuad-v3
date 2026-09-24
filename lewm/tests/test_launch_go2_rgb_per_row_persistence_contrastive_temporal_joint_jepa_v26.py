from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26.py"
)


def _load(name: str):
    prefixes = (
        "_lewm_v26_schema_compat_private_v25_launcher",
        "_lewm_v25_per_row_temporal_private_v24_launcher",
        "_lewm_v24_core_protected_private_v23_launcher",
        "_lewm_v23_scene_action_private_v21_launcher",
        "_lewm_v21_scene_innovation_private_v20_launcher",
    )
    for private in tuple(sys.modules):
        if private.startswith(prefixes):
            sys.modules.pop(private, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime(module):
    prefix = np.zeros((4_262, 9), dtype=np.int64)
    prefix[:4] = np.asarray(
        (
            np.arange(9),
            np.arange(9)[::-1],
            (np.arange(9) + 2) % 15,
            (np.arange(9)[::-1] + 3) % 15,
        )
    )
    prior = prefix.mean(axis=0, dtype=np.float64) * 0.1
    required_v21 = ("rgb", "scene_innovation_negative_row")
    required_v23 = (*required_v21, module.ACTION_PRIOR_M_KEY_V23)
    training = SimpleNamespace(
        REQUIRED_BATCH_KEYS_V21=required_v21,
        REQUIRED_BATCH_KEYS_V23=required_v23,
        REQUIRED_BATCH_KEYS_V24=required_v23,
        REQUIRED_BATCH_KEYS_V25=required_v23,
        REQUIRED_BATCH_KEYS_V26=required_v23,
        ACTION_PRIOR_M_KEY_V23=module.ACTION_PRIOR_M_KEY_V23,
        CURRENT_RGB_KEY="rgb",
    )
    return SimpleNamespace(
        np=np,
        torch=torch,
        schedule=tuple([0, 1, 2, 3] * 4_000),
        labels={"train": SimpleNamespace(prefix_lengths=prefix)},
        pairs={"train": [{"scene_id": f"scene-{i % 4}"} for i in range(4_262)]},
        action_prior_m=prior,
        training_module=training,
    )


def test_denied_shell_retargets_frozen_v25_lifecycle(capsys) -> None:
    module = _load("_v26_launcher_denial")
    assert module.main([]) == 4
    assert "DENIED_NO_FUTURE_AUTHORITY" in capsys.readouterr().out
    assert module.BASE_LAUNCHER_COMMIT == (
        "43231c689547b66de83f3cafbfac270455a7a234"
    )
    assert module.PREREGISTRATION_COMMIT == (
        "0c277fd7350931a7993d5affc2d1d4633ffed916"
    )
    assert module.V25_TERMINAL_FAILURE_RESULT_COMMIT == (
        "26c8fd902319c06d4dbf25cab36a63ec2df44081"
    )
    module._assert_configured_base_v26()


def test_v25_preflight_and_microbatch_are_republished_without_payload_change() -> None:
    module = _load("_v26_launcher_preflight")
    runtime = _runtime(module)
    receipt = module.preflight_schedule_per_row_persistence_contrastive_v26(runtime)
    assert receipt["schema"] == (
        f"{module.SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1"
    )
    assert receipt["passed"] is True
    inherited = module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21
    module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = lambda **_: {
        "rgb": torch.zeros((4, 3, 2, 2)),
        "scene_innovation_negative_row": torch.tensor((1, 0, 3, 2)),
    }
    try:
        batch = module._build_one_microbatch_v26(
            runtime=runtime, indices=(0, 1, 2, 3), stage="test"
        )
    finally:
        module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = inherited
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V25
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V26


def test_receipt_binds_inputs_and_grants_no_execution_or_recovery_read() -> None:
    module = _load("_v26_launcher_receipt")
    receipt = module.private_launcher_adapter_receipt_v26()
    assert receipt["schedule_preflight_delegated_exactly_to_v25"] is True
    assert receipt["microbatch_builder_delegated_exactly_to_v25"] is True
    assert receipt["new_batch_fields_over_v25"] == 0
    assert receipt["extra_tensor_payload_reads"] == 0
    assert receipt["extra_predictor_forwards"] == 0
    assert receipt["recovery_read_or_resume_implemented"] is False
    assert receipt["execution_authorized"] is False
    for relative, expected_sha, expected_bytes in (
        (
            module.PREREGISTRATION_RELATIVE_PATH,
            module.PREREGISTRATION_FILE_SHA256,
            module.PREREGISTRATION_BYTE_COUNT,
        ),
        (
            module.V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
            module.V25_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            module.V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
    ):
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == expected_bytes
        assert hashlib.sha256(raw).hexdigest() == expected_sha
