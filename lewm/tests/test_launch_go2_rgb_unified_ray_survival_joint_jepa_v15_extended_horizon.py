from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
V13_LAUNCHER_PATH = ROOT / (
    "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
    "camera_evidence_bottleneck.py"
)
V15_LAUNCHER_PATH = ROOT / (
    "scripts/launch_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


canonical = _load("_test_v15_extended_horizon_canonical_launcher", V13_LAUNCHER_PATH)
_IMPORTED_BEFORE = set(sys.modules)
launcher = _load("_test_v15_extended_horizon_launcher", V15_LAUNCHER_PATH)
_IMPORTED_BY_V15 = set(sys.modules) - _IMPORTED_BEFORE


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _schedule_fixture() -> tuple[tuple[int, ...], object, object]:
    base = tuple(index % 997 for index in range(16_000))
    expected = {
        100: _canonical_hash(list(base[:1_600])),
        400: _canonical_hash(list(base[:6_400])),
        1_000: _canonical_hash(list(base)),
    }
    executor = SimpleNamespace(CHECKPOINT_SCHEDULE_PREFIX_SHA256=expected)
    labels = SimpleNamespace(
        v4=SimpleNamespace(canonical_json_sha256=_canonical_hash)
    )
    return base, executor, labels


def test_import_is_source_only_and_no_argument_cli_is_denied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_V15
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
            "integrity_replacement_v1_launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_private_adapter_does_not_mutate_the_canonical_launcher() -> None:
    assert launcher._BASE_LAUNCHER is not canonical
    assert canonical.MAXIMUM_UPDATES == 1_000
    assert canonical.MAXIMUM_PRESENTATIONS == 16_000
    assert canonical.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert canonical.MODEL_MODULE_NAME.endswith("v13_camera_evidence_bottleneck")

    adapted = launcher._BASE_LAUNCHER
    assert adapted.MAXIMUM_UPDATES == launcher.MAXIMUM_UPDATES == 2_000
    assert adapted.MAXIMUM_PRESENTATIONS == launcher.MAXIMUM_PRESENTATIONS == 32_000
    assert adapted.OBSERVATION_UPDATES == launcher.OBSERVATION_UPDATES
    assert adapted.MODEL_MODULE_NAME.endswith("v14_unified_ray_survival")
    assert adapted.TRAINING_MODULE_NAME.endswith("v15_extended_horizon")
    assert adapted._validate_schedule_v13 is launcher._validate_schedule_v15
    assert adapted._v12_observation_v13 is launcher._v12_observation_v15
    launcher._assert_configured_base_v15()


def test_integrity_replacement_uses_fresh_evidence_and_arm_identities() -> None:
    prefix = (
        "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
        "integrity_replacement_v1"
    )
    assert launcher.SOURCE_EVIDENCE_SCHEMA_PREFIX == prefix
    assert launcher.LAUNCHER_SCHEMA == f"{prefix}_launcher_v1"
    assert launcher.EXPERIMENT_ARM_NAME == (
        "unified_ray_survival_v15_extended_horizon_integrity_replacement_v1"
    )
    assert launcher.AUTHORITY_RELATIVE_PATH.endswith(
        "integrity_replacement_v1_execution_authorization_2026-07-29.json"
    )
    assert launcher.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "integrity_replacement_v1_source_manifest_2026-07-29.json"
    )
    assert launcher.SOURCE_REVIEW_RELATIVE_PATH.endswith(
        "integrity_replacement_v1_source_review_2026-07-29.json"
    )
    assert launcher.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH.endswith(
        "integrity_replacement_v1_clean_export_certification_2026-07-29.json"
    )


def test_schedule_is_validated_at_16000_then_repeated_exactly_in_memory() -> None:
    base, executor, labels = _schedule_fixture()
    extended = launcher._validate_schedule_v15(
        base,
        executor_api=executor,
        labels_api=labels,
    )
    assert len(extended) == 32_000
    assert extended[:16_000] == base
    assert extended[16_000:] == base
    assert extended[:16_000] == extended[16_000:]

    changed = list(base)
    changed[6_400] += 1
    with pytest.raises(PermissionError, match="prefix identity changed"):
        launcher._validate_schedule_v15(
            changed,
            executor_api=executor,
            labels_api=labels,
        )
    with pytest.raises(PermissionError, match="exactly 16,000"):
        launcher._validate_schedule_v15(
            base + (0,),
            executor_api=executor,
            labels_api=labels,
        )


def test_update1400_controls_are_fresh_while_provenance_keeps_actual_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_physical(runtime: object, model: object, *, update: int, **kwargs: object) -> dict[str, int]:
        runtime.physical_provenance[update] = {"evaluated_update": update}
        return {"evaluated_update": update}

    def fake_v12(runtime: object, model: object, *, update: int) -> tuple[dict[str, int], dict[str, dict[str, bool]] | None]:
        calls.append(update)
        controls = {"fresh": {"passed": True}} if update == 400 else None
        return {"evaluated_from_model": id(model)}, controls

    monkeypatch.setattr(launcher._BASE_LAUNCHER, "_physical_scopes_v13", fake_physical)
    monkeypatch.setattr(launcher, "_ORIGINAL_V12_OBSERVATION_V13", fake_v12)
    runtime = SimpleNamespace(physical_provenance={})
    model = object()
    observed = launcher._BASE_LAUNCHER.V13ComposedRuntime.observe_v13(
        runtime,
        model,
        update=1_400,
        physical_endpoint_updater=lambda **kwargs: {},
    )
    assert calls == [400]
    assert observed["controls"] == {"fresh": {"passed": True}}
    assert observed["physical_scopes"] == {"evaluated_update": 1_400}
    assert observed["physical_provenance"] == {"evaluated_update": 1_400}


def test_update1000_remains_observation_only_without_control_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_v12(runtime: object, model: object, *, update: int) -> tuple[dict[str, int], None]:
        calls.append(update)
        return {"evaluated_update": update}, None

    monkeypatch.setattr(launcher, "_ORIGINAL_V12_OBSERVATION_V13", fake_v12)
    gate, controls = launcher._v12_observation_v15(object(), object(), update=1_000)
    assert calls == [1_000]
    assert gate == {"evaluated_update": 1_000}
    assert controls is None


def test_isolated_import_does_not_load_runtime_or_touch_payload() -> None:
    code = (
        "import runpy,sys;"
        f"ns=runpy.run_path({str(V15_LAUNCHER_PATH)!r});"
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
