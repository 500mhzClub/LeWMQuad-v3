from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = (
    ROOT / "scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load("_test_v18_height_volume_launcher", LAUNCHER_PATH)


def test_source_only_import_and_no_argument_denial(capsys) -> None:
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_"
            "replacement_v1_launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_private_base_has_only_exact_v18_selectors_and_caps() -> None:
    launcher._assert_configured_base_v18()
    base = launcher._BASE_LAUNCHER
    assert base.EXECUTOR_MODULE_NAME == (
        "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
    )
    assert base.MODEL_MODULE_NAME == (
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
        "object_space_height_volume"
    )
    assert base.TRAINING_MODULE_NAME == (
        "scripts.run_go2_rgb_object_space_height_volume_joint_jepa_v18"
    )
    assert base.EXPERIMENT_ARM_NAME == (
        "object_space_height_volume_v18_integrity_replacement_v1"
    )
    assert base.MAXIMUM_UPDATES == 1_000
    assert base.MAXIMUM_PRESENTATIONS == 16_000
    assert base.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert launcher.private_launcher_adapter_receipt_v18()["execution_authorized"] is False


def test_integrity_replacement_uses_fresh_evidence_paths() -> None:
    prefix = (
        "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "integrity_replacement_v1"
    )
    assert launcher.SOURCE_EVIDENCE_SCHEMA_PREFIX == prefix
    for relative in (
        launcher.AUTHORITY_RELATIVE_PATH,
        launcher.SOURCE_MANIFEST_RELATIVE_PATH,
        launcher.SOURCE_REVIEW_RELATIVE_PATH,
        launcher.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    ):
        assert "integrity_replacement_v1" in relative


def test_selector_mutation_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "MODEL_MODULE_NAME",
        "lewm.models.unreviewed",
    )
    with pytest.raises(PermissionError, match="changed after import"):
        launcher._assert_configured_base_v18()


def test_terminal_exit_codes(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v18", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v18",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v18",
        lambda **kwargs: {"status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 2


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
