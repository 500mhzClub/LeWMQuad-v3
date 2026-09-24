from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
V13_LAUNCHER_PATH = (
    ROOT
    / "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
V14_LAUNCHER_PATH = (
    ROOT / "scripts/launch_go2_rgb_unified_ray_survival_joint_jepa_v14.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


canonical_v13 = _load("_test_v14_adapter_canonical_v13_launcher", V13_LAUNCHER_PATH)
_IMPORTED_BEFORE = set(sys.modules)
launcher = _load("_test_v14_unified_ray_survival_launcher", V14_LAUNCHER_PATH)
_IMPORTED_BY_V14 = set(sys.modules) - _IMPORTED_BEFORE


def test_import_is_source_only_and_no_argument_cli_is_freshly_denied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_V14
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_launcher_v1",
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_private_adapter_selects_v14_without_mutating_canonical_v13() -> None:
    assert launcher._BASE_LAUNCHER is not canonical_v13
    assert canonical_v13.EXECUTOR_MODULE_NAME.endswith(
        "v13_camera_evidence_bottleneck"
    )
    assert canonical_v13.MODEL_MODULE_NAME.endswith(
        "v13_camera_evidence_bottleneck"
    )
    assert canonical_v13.SOURCE_EVIDENCE_SCHEMA_PREFIX.endswith("joint_jepa_v13")
    assert canonical_v13.EXPERIMENT_ARM_NAME == "camera_evidence_bottleneck_v13"
    assert canonical_v13.LAUNCHER_SCHEMA.endswith("joint_jepa_v13_launcher_v1")
    assert launcher._BASE_LAUNCHER.EXECUTOR_MODULE_NAME == (
        "scripts.execute_go2_rgb_unified_ray_survival_joint_jepa_v14"
    )
    assert launcher._BASE_LAUNCHER.MODEL_MODULE_NAME == (
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v14_"
        "unified_ray_survival"
    )
    assert launcher._BASE_LAUNCHER.TRAINING_MODULE_NAME.endswith(
        "v13_camera_evidence_bottleneck"
    )
    assert launcher._BASE_LAUNCHER.SOURCE_EVIDENCE_SCHEMA_PREFIX == (
        "lewm_go2_rgb_unified_ray_survival_joint_jepa_v14"
    )
    assert launcher._BASE_LAUNCHER.EXPERIMENT_ARM_NAME == (
        "unified_ray_survival_v14"
    )
    assert launcher._BASE_LAUNCHER.LAUNCHER_SCHEMA == launcher.LAUNCHER_SCHEMA
    launcher._assert_configured_base_v14()


def test_base_uses_only_explicit_dynamic_science_selectors() -> None:
    compose = inspect.getsource(launcher._BASE_LAUNCHER.compose_runtime_v13)
    initialize = inspect.getsource(
        launcher._BASE_LAUNCHER.V13ComposedRuntime.initialize_model_v13
    )
    execute = inspect.getsource(
        launcher._BASE_LAUNCHER.execute_future_authorized_v13
    )
    microbatch = inspect.getsource(launcher._BASE_LAUNCHER._build_one_microbatch_v13)
    main = inspect.getsource(launcher._BASE_LAUNCHER.main)
    assert "importlib.import_module(EXECUTOR_MODULE_NAME)" in compose
    assert "importlib.import_module(MODEL_MODULE_NAME)" in compose
    assert "importlib.import_module(TRAINING_MODULE_NAME)" in compose
    assert 'getattr(self.executor_api, "MODEL_CLASS_NAME", None)' in initialize
    assert "importlib.import_module(EXECUTOR_MODULE_NAME)" in execute
    assert microbatch.count("arm=EXPERIMENT_ARM_NAME") == 2
    assert '"schema": LAUNCHER_SCHEMA' in main


def test_private_adapter_fails_closed_if_a_selector_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "MODEL_MODULE_NAME",
        "lewm.models.unreviewed",
    )
    with pytest.raises(PermissionError, match="changed after import"):
        launcher._assert_configured_base_v14()


def test_cli_exit_codes_preserve_terminal_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v14", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v14",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v14",
        lambda **kwargs: {"status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 2


def test_isolated_import_does_not_load_runtime_or_touch_payload() -> None:
    code = (
        "import runpy,sys;"
        f"ns=runpy.run_path({str(V14_LAUNCHER_PATH)!r});"
        "assert ns['main']([])==4;"
        "assert 'torch' not in sys.modules;"
        "assert 'numpy' not in sys.modules;"
        "assert 'PIL' not in sys.modules"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
