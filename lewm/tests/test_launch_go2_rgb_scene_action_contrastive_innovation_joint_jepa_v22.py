from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = ROOT / (
    "scripts/launch_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORTED_BEFORE = set(sys.modules)
launcher = _load("_test_v22_scene_action_launcher", LAUNCHER_PATH)
_IMPORTED_BY_LAUNCHER = set(sys.modules) - _IMPORTED_BEFORE


def test_import_is_source_only_denied_and_selects_fresh_v22_lifecycle(capsys) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_LAUNCHER
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
            "v22_launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }
    launcher._assert_configured_base_v22()
    base = launcher._BASE_LAUNCHER
    assert base.TRAINING_MODULE_NAME.endswith(
        "rgb_scene_action_contrastive_innovation_joint_jepa_v22"
    )
    assert base.EXECUTOR_MODULE_NAME.endswith(
        "rgb_scene_action_contrastive_innovation_joint_jepa_v22"
    )
    assert base.EXPERIMENT_ARM_NAME == "scene_action_contrastive_innovation_v22"
    assert base.MAXIMUM_UPDATES == 1_000
    assert base.MAXIMUM_PRESENTATIONS == 16_000
    assert base._build_one_microbatch_v13 is launcher._build_one_microbatch_v21


def test_frozen_preregistration_result_and_exact_v21_preflight_are_bound() -> None:
    receipt = launcher.private_launcher_adapter_receipt_v22()
    assert receipt["base_launcher"] == {
        "path": (
            "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_"
            "innovation_joint_jepa_v21.py"
        ),
        "commit": "7071a006dda3851280fbdf030e156862c4f19ab3",
        "file_sha256": (
            "4cb6fb3302919d6090e3ef456068ba209890237d65c87c8515a723628ee5b486"
        ),
        "byte_count": 22_650,
    }
    assert receipt["preregistration"]["commit"] == (
        "43053ae49c28082c616f45ed857eedb727380952"
    )
    assert receipt["predecessor_scientific_result"]["commit"] == (
        "e5b5e56b30cee0c1eb818d52c4d886909f570f4d"
    )
    assert receipt["predecessor_scientific_result"]["content_sha256"] == (
        "2195025bf24e3de621e76a5a5e3ea272ced05bd9f6e4fb91302035137ab7b9ec"
    )
    assert receipt["scene_negative_preflight_inherited_exactly_from_v21"]
    assert receipt["new_batch_fields_over_v21"] == 0
    assert receipt["action_negatives_derived_from_existing_all_action_prediction"]
    assert receipt["retry_authorized"] is False
    assert receipt["resume_authorized"] is False
    assert receipt["execution_authorized"] is False
    assert launcher.first_cyclic_different_scene_rows_v21(
        ("a", "b", "b", "a")
    ) == (1, 3, 3, 1)


def test_selector_mutation_fails_closed_and_terminal_exit_codes_are_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v22", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v22",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v22",
        lambda **kwargs: {"status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 2

    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "MODEL_MODULE_NAME",
        "lewm.models.unreviewed",
    )
    with pytest.raises(PermissionError, match="changed"):
        launcher._assert_configured_base_v22()


def test_isolated_import_does_not_load_runtime_or_touch_payload() -> None:
    code = (
        "import runpy,sys;"
        f"ns=runpy.run_path({str(LAUNCHER_PATH)!r});"
        "assert ns['main']([])==4;"
        "assert 'torch' not in sys.modules;"
        "assert 'numpy' not in sys.modules;"
        "assert 'PIL' not in sys.modules"
    )
    environment = dict(os.environ)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "ZE_AFFINITY_MASK",
    ):
        environment.pop(name, None)
    environment["HIP_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
