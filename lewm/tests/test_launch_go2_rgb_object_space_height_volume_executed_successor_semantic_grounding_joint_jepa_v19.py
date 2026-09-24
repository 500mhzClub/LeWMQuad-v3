from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = ROOT / (
    "scripts/launch_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORTED_BEFORE = set(sys.modules)
launcher = _load("_test_v19_semantic_grounding_launcher", LAUNCHER_PATH)
_IMPORTED_BY_LAUNCHER = set(sys.modules) - _IMPORTED_BEFORE


def _comparison(index: int) -> dict[str, object]:
    return {
        "scene_count": 8,
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 20_260_728,
        "equal_scene_mean_delta": 0.1 + index,
        "bootstrap_lower_95": 0.01 + index,
        "per_scene_delta": {
            f"scene_{scene}": index + scene / 10.0 for scene in range(8)
        },
        "family_deltas": {
            family: index + (family_index + 1) / 10.0
            for family_index, family in enumerate(launcher.REGISTERED_FAMILIES)
        },
        "positive_family_count": 8,
    }


def test_import_is_source_only_and_no_argument_cli_is_denied(capsys) -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_LAUNCHER
        for prefix in ("torch", "numpy", "PIL")
    )
    assert launcher.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "reservation_created": False,
        "schema": (
            "lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v20_launcher_v1"
        ),
        "scientific_payload_opened": False,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
    }


def test_private_v18_adapter_selects_only_v19_training_and_executor() -> None:
    launcher._assert_configured_base_v19()
    base = launcher._BASE_LAUNCHER
    assert base.MODEL_MODULE_NAME == (
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
        "object_space_height_volume"
    )
    assert base.TRAINING_MODULE_NAME == (
        "scripts.run_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19"
    )
    assert base.EXECUTOR_MODULE_NAME == (
        "scripts.execute_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19"
    )
    assert base.EXPERIMENT_ARM_NAME == (
        "object_space_height_volume_executed_successor_semantic_grounding_v20"
    )
    assert base.SOURCE_EVIDENCE_SCHEMA_PREFIX == (
        "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
        "grounding_joint_jepa_v20"
    )
    assert base.AUTHORITY_RELATIVE_PATH.endswith(
        "joint_jepa_v20_execution_authorization_2026-07-30.json"
    )
    assert base.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "joint_jepa_v20_source_manifest_2026-07-30.json"
    )
    assert base.SOURCE_REVIEW_RELATIVE_PATH.endswith(
        "joint_jepa_v20_source_review_2026-07-30.json"
    )
    assert base.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH.endswith(
        "joint_jepa_v20_clean_export_certification_2026-07-30.json"
    )
    assert base.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH.endswith(
        "semantic_grounding_joint_jepa_v19_source_closure.py"
    )
    assert base.MAXIMUM_UPDATES == 1_000
    assert base.MAXIMUM_PRESENTATIONS == 16_000
    assert base.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert base._v12_observation_v13 is launcher._v12_observation_v19


def test_preregistration_identity_is_bound_in_adapter_receipt() -> None:
    receipt = launcher.private_launcher_adapter_receipt_v19()
    assert receipt["preregistration"] == {
        "path": (
            "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v20_preregistration_2026-07-30.md"
        ),
        "commit": "c99837b91aeb959e07da94e898e3ba11ccbb4c04",
        "file_sha256": (
            "3f450b8949022514f82448d122de637d4cefd91829a72d0ac3f8b14a789a42bd"
        ),
        "byte_count": 9_732,
    }
    assert receipt["predecessor_terminal_failure"] == {
        "path": (
            "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v19_integrity_replacement_v1_"
            "terminal_failure_result_2026-07-30.json"
        ),
        "commit": "7105e2d9ed6e724f364c837e84177b6b4c4cd163",
        "file_sha256": (
            "1b155248194ffd6d7943f84d88c25e29843fb9c977fc5b9bd8053e381c49b886"
        ),
        "byte_count": 9_497,
        "content_sha256": (
            "fb794750c9efcc1430235478c3f4da02dcaf5211c131ca4e15084950a8cbd4e3"
        ),
    }
    assert receipt["numeric_comparisons_retained_without_rescoring"] is True
    assert receipt["certified_source_root"] == (
        "/home/andrewknowles/Workspace/"
        "LeWMQuad-v3-v20-accounting-isolation-source"
    )
    assert receipt["output_root"] == (
        ".generated/go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20/attempt_v1"
    )
    assert receipt["one_shot_attempt_count"] == 1
    assert receipt["retry_authorized"] is False
    assert receipt["resume_authorized"] is False
    assert receipt["execution_authorized"] is False


def test_registered_families_match_the_frozen_inherited_registry_exactly() -> None:
    assert launcher.REGISTERED_FAMILIES == (
        "large_enclosed_maze",
        "local_composite_motifs",
        "loop_alias_stress",
        "medium_enclosed_maze",
        "open_obstacle_field",
        "rough_local_dynamics",
        "small_enclosed_maze",
        "visual_sensor_stress",
    )


def test_comparison_sanitizer_accepts_inherited_mapping_and_rejects_change() -> None:
    inherited = _comparison(0)
    assert launcher._sanitize_comparison_v19(inherited) == inherited

    changed = _comparison(0)
    family_deltas = dict(changed["family_deltas"])
    family_deltas["structured_corridor_rooms"] = family_deltas.pop(
        "visual_sensor_stress"
    )
    changed["family_deltas"] = family_deltas
    with pytest.raises(RuntimeError, match="grouping changed"):
        launcher._sanitize_comparison_v19(changed)


@pytest.mark.parametrize("changed_registry", ["evaluator", "checkpoint_selection"])
def test_observation_binds_both_inherited_family_registries(
    changed_registry: str,
) -> None:
    stale = (*launcher.REGISTERED_FAMILIES[:-1], "structured_corridor_rooms")
    evaluator_registry = (
        stale if changed_registry == "evaluator" else launcher.REGISTERED_FAMILIES
    )
    checkpoint_registry = (
        stale
        if changed_registry == "checkpoint_selection"
        else launcher.REGISTERED_FAMILIES
    )

    def must_not_score() -> None:
        raise AssertionError("registry check must precede scoring")

    runtime = SimpleNamespace(
        executor_api=SimpleNamespace(REGISTERED_FAMILIES=checkpoint_registry),
        v1_executor=SimpleNamespace(
            CONTROL_NAMES=launcher.CONTROL_NAMES,
            REGISTERED_FAMILIES=evaluator_registry,
            paired_control_comparison_v1=must_not_score,
        ),
    )
    with pytest.raises(RuntimeError, match="family registry changed"):
        launcher._v12_observation_v19(runtime, object(), update=100)
    assert not hasattr(runtime, "causal_comparisons_v19")


def test_numeric_comparisons_are_captured_from_the_existing_four_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def paired(index: int) -> dict[str, object]:
        calls.append(index)
        return _comparison(index)

    executor = SimpleNamespace(
        CONTROL_NAMES=launcher.CONTROL_NAMES,
        REGISTERED_FAMILIES=launcher.REGISTERED_FAMILIES,
        paired_control_comparison_v1=paired,
    )
    runtime = SimpleNamespace(
        executor_api=SimpleNamespace(
            REGISTERED_FAMILIES=launcher.REGISTERED_FAMILIES
        ),
        v1_executor=executor,
    )
    controls = {
        name: {
            "positive_equal_scene_delta": True,
            "positive_bootstrap_lower_95": True,
            "positive_family_count": True,
        }
        for name in launcher.CONTROL_NAMES
    }

    def inherited(runtime_value: object, model: object, *, update: int):
        assert runtime_value is runtime
        assert update == 400
        for index in range(4):
            runtime.v1_executor.paired_control_comparison_v1(index)
        return {"unchanged_gate": True}, controls

    monkeypatch.setattr(launcher, "_ORIGINAL_V12_OBSERVATION_V13", inherited)
    original = executor.paired_control_comparison_v1
    gate, observed_controls = launcher._v12_observation_v19(
        runtime,
        object(),
        update=400,
    )
    assert calls == [0, 1, 2, 3]
    assert executor.paired_control_comparison_v1 is original
    assert gate == {"unchanged_gate": True}
    assert observed_controls is controls
    assert tuple(runtime.causal_comparisons_v19) == (400,)
    assert tuple(runtime.causal_comparisons_v19[400]) == launcher.CONTROL_NAMES
    assert runtime.causal_comparisons_v19[400]["wrong_rgb"] == _comparison(2)

    with pytest.raises(RuntimeError, match="not one-shot"):
        launcher._v12_observation_v19(runtime, object(), update=400)
    assert calls == [0, 1, 2, 3]


def test_comparison_hook_fails_closed_and_restores_inherited_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malformed() -> dict[str, object]:
        value = _comparison(0)
        value.pop("family_deltas")
        return value

    executor = SimpleNamespace(
        CONTROL_NAMES=launcher.CONTROL_NAMES,
        REGISTERED_FAMILIES=launcher.REGISTERED_FAMILIES,
        paired_control_comparison_v1=malformed,
    )
    runtime = SimpleNamespace(
        executor_api=SimpleNamespace(
            REGISTERED_FAMILIES=launcher.REGISTERED_FAMILIES
        ),
        v1_executor=executor,
    )

    def inherited(runtime_value: object, model: object, *, update: int):
        runtime_value.v1_executor.paired_control_comparison_v1()
        raise AssertionError("unreachable")

    monkeypatch.setattr(launcher, "_ORIGINAL_V12_OBSERVATION_V13", inherited)
    original = executor.paired_control_comparison_v1
    with pytest.raises(RuntimeError, match="schema changed"):
        launcher._v12_observation_v19(runtime, object(), update=100)
    assert executor.paired_control_comparison_v1 is original
    assert runtime.causal_comparisons_v19 == {}


def test_selector_or_hook_mutation_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher._BASE_LAUNCHER,
        "MODEL_MODULE_NAME",
        "lewm.models.unreviewed",
    )
    with pytest.raises(PermissionError, match="changed after import"):
        launcher._assert_configured_base_v19()


def test_terminal_exit_codes(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_load_authority_file_v19", lambda path: {})
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v19",
        lambda **kwargs: {"status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"},
    )
    assert launcher.main(["--future-authority", "authority.json"]) == 0
    monkeypatch.setattr(
        launcher,
        "execute_future_authorized_v19",
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
