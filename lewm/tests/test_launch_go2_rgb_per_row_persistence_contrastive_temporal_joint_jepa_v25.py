from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25.py"
)


def _load(name: str):
    import sys

    for private in tuple(sys.modules):
        if private.startswith("_lewm_v25_per_row_temporal_private_v24_launcher"):
            sys.modules.pop(private, None)
        elif private.startswith("_lewm_v24_core_protected_private_v23_launcher"):
            sys.modules.pop(private, None)
        elif private.startswith("_lewm_v23_scene_action_private_v21_launcher"):
            sys.modules.pop(private, None)
        elif private.startswith("_lewm_v21_scene_innovation_private_v20_launcher"):
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


def test_denied_shell_retargets_every_lifecycle_selector(capsys) -> None:
    module = _load("_v25_launcher_denial")
    assert module.main([]) == 4
    assert "DENIED_NO_FUTURE_AUTHORITY" in capsys.readouterr().out
    assert module.PREREGISTRATION_COMMIT == (
        "f00e20df3b429f9242516ac38f67fea587e04b22"
    )
    assert module.V24_SCIENTIFIC_RESULT_COMMIT == (
        "2824c80c54fc7502b1413b3371fc87c9206f82a2"
    )
    module._assert_configured_base_v25()
    assert module._V24_LAUNCHER._build_one_microbatch_v24 is (
        module._build_one_microbatch_v25
    )
    assert module._BASE_LAUNCHER._build_one_microbatch_v13 is (
        module._build_one_microbatch_v25
    )


def test_preflight_is_inherited_and_republished_only_as_v25() -> None:
    module = _load("_v25_launcher_preflight")
    runtime = _runtime(module)
    receipt = module.preflight_schedule_per_row_persistence_contrastive_v25(runtime)
    assert receipt["schema"].startswith(module.SOURCE_EVIDENCE_SCHEMA_PREFIX)
    assert "v24" not in receipt["schema"]
    assert receipt["passed"] is True
    assert receipt["microbatch_count"] == 4_000
    assert (
        module.preflight_schedule_per_row_persistence_contrastive_v25(runtime)
        is receipt
    )


def test_microbatch_schema_is_bit_exact_v24_with_no_new_payload() -> None:
    module = _load("_v25_launcher_batch")
    runtime = _runtime(module)
    prior_before = runtime.action_prior_m.copy()
    inherited = module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21
    module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = lambda **_: {
        "rgb": torch.zeros((4, 3, 2, 2)),
        "scene_innovation_negative_row": torch.tensor((1, 0, 3, 2)),
    }
    try:
        batch = module._build_one_microbatch_v25(
            runtime=runtime, indices=(0, 1, 2, 3), stage="test"
        )
    finally:
        module._V23_LAUNCHER._INHERITED_BUILD_ONE_MICROBATCH_V21 = inherited
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V24
    assert tuple(batch) == runtime.training_module.REQUIRED_BATCH_KEYS_V25
    assert batch[module.ACTION_PRIOR_M_KEY_V23].requires_grad is False
    assert np.array_equal(runtime.action_prior_m, prior_before)


def test_launcher_receipt_grants_no_recovery_read_or_execution() -> None:
    module = _load("_v25_launcher_receipt")
    receipt = module.private_launcher_adapter_receipt_v25()
    assert receipt["new_batch_fields_over_v24"] == 0
    assert receipt["extra_tensor_payload_reads"] == 0
    assert receipt["extra_predictor_forwards"] == 0
    assert receipt["update400_recovery_write_owned_by_executor"] is True
    assert receipt["recovery_read_or_resume_implemented"] is False
    assert receipt["execution_authorized"] is False


def test_inherited_binary_publisher_is_exclusive_atomic_and_read_only(tmp_path) -> None:
    module = _load("_v25_launcher_publisher")
    publisher = module._BASE_LAUNCHER.V13WriteOncePublisher(tmp_path, None)
    relative = "recovery/update_400_training_state.pt"
    raw = b"write-only-recovery-state"
    binding = publisher.publish_bytes(relative, raw)
    path = tmp_path / relative
    assert binding["path"] == relative
    assert binding["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="write-once"):
        publisher.publish_bytes(relative, raw)
