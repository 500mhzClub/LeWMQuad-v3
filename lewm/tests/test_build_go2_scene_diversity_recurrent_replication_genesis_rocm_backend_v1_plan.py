from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts import (
    build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan
    as builder,
)


def _frozen() -> dict:
    return json.loads(builder.FROZEN_V1_EXACT_PLAN.read_text())


@pytest.fixture(scope="module")
def runtime_bindings() -> dict:
    return builder.build_rocm_runtime_bindings()


@pytest.fixture(scope="module")
def scientific_plan(runtime_bindings: dict) -> dict:
    return builder.build_scientific_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


@pytest.fixture(scope="module")
def qualification_plan(runtime_bindings: dict) -> dict:
    return builder.build_qualification_plan(
        frozen_plan=_frozen(), runtime_bindings=runtime_bindings
    )


def test_scientific_plan_is_only_the_exact_rocm_successor_overlay(
    scientific_plan: dict,
) -> None:
    frozen = _frozen()
    changed_top_level = {
        "attempt_id",
        "output_root",
        "runtime_bindings",
        "execution_contract",
        "successor_contract",
    }

    assert set(scientific_plan) == {*set(frozen), "successor_contract"}
    assert all(
        scientific_plan[name] == frozen[name]
        for name in set(frozen) - changed_top_level
    )
    assert scientific_plan["attempt_id"] == builder.DEFAULT_ATTEMPT_ID
    assert scientific_plan["output_root"] == str(
        builder.DEFAULT_OUTPUT_ROOT.resolve(strict=False)
    )

    frozen_execution = frozen["execution_contract"]
    execution = scientific_plan["execution_contract"]
    assert set(execution) == set(frozen_execution)
    assert all(
        execution[name] == frozen_execution[name]
        for name in set(execution)
        - {
            "backend",
            "python_invocation_path",
            "environment",
            "graphics_preflight",
        }
    )
    assert execution["backend"] == "amdgpu"
    assert execution["python_invocation_path"] == str(
        builder.ROCM_PYTHON.absolute()
    )
    assert Path(execution["python_invocation_path"]) != Path(
        scientific_plan["runtime_bindings"]["python_executable_target"]["path"]
    )
    assert Path(execution["python_invocation_path"]).resolve(strict=True) == Path(
        scientific_plan["runtime_bindings"]["python_executable_target"]["path"]
    )
    assert execution["environment"] == builder.rocm_execution_environment(
        "scientific"
    )
    assert (
        execution["graphics_preflight"]
        == builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
    )

    successor = scientific_plan["successor_contract"]
    assert successor["plan_role"] == "scientific"
    assert successor["genesis_backend_symbol"] == "gs.amdgpu"
    assert successor["qualification_scene_indices_in_order"] == []
    assert successor["scientific_execution_authorized"] is False
    assert successor["qualification_execution_authorized"] is False
    assert successor["cpu_qualification_terminal_review_binding"] == (
        builder.CPU_TERMINAL_REVIEW_BINDING
    )

    assert builder.validate_rocm_plan(
        scientific_plan,
        expected_attempt_id=builder.DEFAULT_ATTEMPT_ID,
        expected_output_root=builder.DEFAULT_OUTPUT_ROOT,
        plan_role="scientific",
    ) == scientific_plan


def test_qualification_plan_is_exactly_two_full_scenes_in_fixed_order(
    qualification_plan: dict,
) -> None:
    successor = qualification_plan["successor_contract"]
    assert qualification_plan["attempt_id"] == builder.QUALIFICATION_ATTEMPT_ID
    assert qualification_plan["output_root"] == str(
        builder.QUALIFICATION_OUTPUT_ROOT.resolve(strict=False)
    )
    assert successor["plan_role"] == "qualification"
    assert successor["qualification_scene_indices_in_order"] == [12, 0]
    assert successor["qualification_worker_watchdog_seconds"] == 300
    assert successor["qualification_timing_gate"] == {
        "scene_count": 64,
        "fixed_noncollection_reserve_seconds": 900,
        "formula": "64 * max(worker_elapsed_seconds) + 900 <= 7200",
        "scientific_wall_cap_seconds": 7200,
    }
    assert successor["qualification_execution_authorized"] is False
    assert successor["scientific_execution_authorized"] is False
    assert successor["probe_output_reuse_authorized"] is False
    assert qualification_plan["execution_contract"]["environment"] == (
        builder.rocm_execution_environment("qualification")
    )
    assert builder.validate_rocm_plan(
        qualification_plan,
        expected_attempt_id=builder.QUALIFICATION_ATTEMPT_ID,
        expected_output_root=builder.QUALIFICATION_OUTPUT_ROOT,
        plan_role="qualification",
    ) == qualification_plan


def test_emitted_plans_match_the_validated_builds(
    scientific_plan: dict, qualification_plan: dict
) -> None:
    assert json.loads(builder.DEFAULT_PLAN_OUTPUT.read_text()) == scientific_plan
    assert json.loads(builder.QUALIFICATION_PLAN_OUTPUT.read_text()) == (
        qualification_plan
    )


def test_runtime_and_environment_bind_the_exact_rocm_stack(
    runtime_bindings: dict,
) -> None:
    assert set(runtime_bindings) == set(builder.ROCM_RUNTIME_PATHS)
    assert {
        "rocminfo_executable",
        "rocm_lld_executable",
        "genesis_distribution_metadata",
        "genesis_distribution_record",
        "quadrants_distribution_metadata",
        "quadrants_distribution_record",
        "quadrants_init_source",
        "quadrants_lang_misc_source",
        "quadrants_lang_kernel_source",
        "quadrants_lib_utils_source",
        "quadrants_native_core",
        "torch_distribution_metadata",
        "torch_distribution_record",
        "torch_site_packages_link",
        "rsl_rl_distribution_metadata",
        "rsl_rl_distribution_record",
        "rsl_rl_on_policy_runner",
        "tensordict_distribution_metadata",
        "tensordict_distribution_record",
        "torchvision_distribution_metadata",
        "torchvision_distribution_record",
        "numpy_distribution_metadata",
        "numpy_distribution_record",
        "pillow_distribution_metadata",
        "pillow_distribution_record",
        "world_model_python_environment_config",
        "genesis_init_source",
        "genesis_constants_source",
        "genesis_misc_source",
        "genesis_scene_source",
        "genesis_camera_source",
        "genesis_rasterizer_source",
        "genesis_egl_platform_source",
        "quadrants_runtime_amdgpu_bitcode",
        "quadrants_rocm70_opencl_bitcode",
        "quadrants_rocm70_ockl_bitcode",
        "quadrants_rocm70_ocml_bitcode",
        "quadrants_rocm70_isa_gfx1201_bitcode",
        "quadrants_rocm70_abi_v4_bitcode",
        "quadrants_rocm70_correct_sqrt_off_bitcode",
        "quadrants_rocm70_daz_off_bitcode",
        "quadrants_rocm70_finite_only_off_bitcode",
        "quadrants_rocm70_unsafe_math_off_bitcode",
        "quadrants_rocm70_wave64_off_bitcode",
        "go2_dae_base",
        "go2_dae_calf",
        "go2_dae_calf_mirror",
        "go2_dae_foot",
        "go2_dae_hip",
        "go2_dae_thigh",
        "go2_dae_thigh_mirror",
    } <= set(runtime_bindings)
    assert all(
        Path(binding["path"]) == builder.ROCM_RUNTIME_PATHS[name].resolve()
        for name, binding in runtime_bindings.items()
    )
    science_environment = builder.rocm_execution_environment("scientific")
    qualification_environment = builder.rocm_execution_environment(
        "qualification"
    )
    assert science_environment["PATH"].split(":", maxsplit=1)[0] == str(
        builder.ROCM_LLVM_BIN
    )
    assert science_environment["GS_BACKEND"] == "amdgpu"
    assert science_environment["HIP_VISIBLE_DEVICES"] == "0"
    assert science_environment["ROCR_VISIBLE_DEVICES"] == "0"
    assert science_environment["EGL_DEVICE_ID"] == "1"
    assert science_environment["PYOPENGL_PLATFORM"] == "egl"
    assert science_environment["ROCM_PATH"] == "/opt/rocm-7.1.1"
    assert science_environment["GS_ENABLE_NDARRAY"] == "1"
    assert science_environment["GS_ENABLE_FASTCACHE"] == "1"
    assert science_environment["GS_ENABLE_ZEROCOPY"] == "1"
    assert science_environment["GS_CACHE_FILE_PATH"] == str(
        (builder.DEFAULT_ATTEMPT_ROOT / "quadrants_cache").resolve(
            strict=False
        )
    )
    assert qualification_environment["GS_CACHE_FILE_PATH"] == str(
        (builder.QUALIFICATION_ATTEMPT_ROOT / "quadrants_cache").resolve(
            strict=False
        )
    )
    assert (
        qualification_environment["GS_CACHE_FILE_PATH"]
        != science_environment["GS_CACHE_FILE_PATH"]
    )
    assert "LD_LIBRARY_PATH" not in science_environment
    assert "HSA_OVERRIDE_GFX_VERSION" not in science_environment
    assert builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION["hip_arch_name"] == (
        "gfx1201"
    )
    assert builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION["drm_vendor_id"] == (
        "0x1002"
    )
    assert builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION["drm_device_id"] == (
        "0x7551"
    )


def test_plan_invocation_path_preserves_the_rocm_venv(
    scientific_plan: dict,
) -> None:
    code = (
        "import importlib.util,json,sys;"
        "print(json.dumps({'prefix':sys.prefix,"
        "'torch':importlib.util.find_spec('torch') is not None,"
        "'genesis':importlib.util.find_spec('genesis') is not None}))"
    )
    completed = subprocess.run(
        [
            scientific_plan["execution_contract"]["python_invocation_path"],
            "-I",
            "-B",
            "-c",
            code,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "prefix": str(builder.ROCM_VENV.absolute()),
        "torch": True,
        "genesis": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda plan: plan["execution_contract"].__setitem__("backend", "gpu"),
        lambda plan: plan["execution_contract"]["environment"].__setitem__(
            "HSA_OVERRIDE_GFX_VERSION", "11.0.0"
        ),
        lambda plan: plan["execution_contract"].__setitem__("seed", 7),
        lambda plan: plan["successor_contract"].__setitem__(
            "qualification_scene_indices_in_order", [0, 12]
        ),
        lambda plan: plan["successor_contract"][
            "cpu_qualification_terminal_review_binding"
        ].__setitem__("byte_count", 1),
    ),
)
def test_any_unregistered_change_is_rejected(
    scientific_plan: dict, mutation
) -> None:
    changed = copy.deepcopy(scientific_plan)
    mutation(changed)
    with pytest.raises(
        builder.SceneDiversityGenesisRocmPlanError,
        match="changed|absent|exact",
    ):
        builder.validate_rocm_plan(
            changed,
            expected_attempt_id=builder.DEFAULT_ATTEMPT_ID,
            expected_output_root=builder.DEFAULT_OUTPUT_ROOT,
            plan_role="scientific",
        )


def test_validator_is_independent_of_mutable_predecessor_overlays(
    scientific_plan: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_validate_plan(_plan):
        raise AssertionError("mutable pilot.validate_plan was called")

    monkeypatch.setattr(builder.pilot, "validate_plan", forbidden_validate_plan)
    monkeypatch.setattr(
        builder.pilot,
        "EXECUTION_ENVIRONMENT",
        {"GS_BACKEND": "poisoned"},
    )

    assert builder.validate_rocm_plan(
        scientific_plan,
        expected_attempt_id=builder.DEFAULT_ATTEMPT_ID,
        expected_output_root=builder.DEFAULT_OUTPUT_ROOT,
        plan_role="scientific",
    ) == scientific_plan


def test_changed_frozen_science_and_nonfresh_roots_are_rejected(
    runtime_bindings: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = _frozen()
    changed["states_per_scene"] = 5
    with pytest.raises(
        builder.SceneDiversityGenesisRocmPlanError,
        match="frozen Vulkan scientific plan",
    ):
        builder.build_scientific_plan(
            frozen_plan=changed, runtime_bindings=runtime_bindings
        )

    fake_repo = tmp_path / "repo"
    development = fake_repo / ".generated/dev"
    development.mkdir(parents=True)
    attempt = development / "rocm/attempt_v1"
    collection = attempt / "collection"
    attempt.mkdir(parents=True)
    monkeypatch.setattr(builder, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(builder, "DEFAULT_ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(builder, "DEFAULT_OUTPUT_ROOT", collection)
    with pytest.raises(
        builder.SceneDiversityGenesisRocmPlanError,
        match="exact fresh development path",
    ):
        builder.build_scientific_plan(
            frozen_plan=_frozen(), runtime_bindings=runtime_bindings
        )
