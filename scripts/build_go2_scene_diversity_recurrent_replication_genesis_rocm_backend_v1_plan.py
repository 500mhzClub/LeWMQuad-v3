#!/usr/bin/env python3
"""Build prospective Genesis 0.4.6 ROCm scientific and qualification plans.

The original Vulkan V1 plan remains the sole scientific-plan input.  The
unexecuted CPU scientific plan is retained as an identity witness, and the
consumed CPU qualification terminal review is an explicit successor premise.
This builder emits metadata only: neither plan grants qualification or
scientific execution authority.

Validation is deliberately local and immutable.  It never replaces or reads
through mutable ``pilot.validate_plan`` or ``pilot.EXECUTION_ENVIRONMENT``
overlays, which caused the predecessor CPU qualification to fail before
Genesis initialization.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402


FROZEN_V1_EXACT_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "v1_exact_plan_2026-08-04.json"
)
FROZEN_V1_EXACT_PLAN_SHA256 = (
    "c34aa23303951d32dd9686a607de7b78df06db026918d868017a6a93c506a040"
)
FROZEN_V1_EXACT_PLAN_BYTE_COUNT = 346_027

FROZEN_CPU_EXACT_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_exact_plan_2026-08-04.json"
)
FROZEN_CPU_EXACT_PLAN_SHA256 = (
    "258d6bf004fa3618d492b583c56ea7fbc15b127ade36299fcba11295b147745e"
)
FROZEN_CPU_EXACT_PLAN_BYTE_COUNT = 346_045

CPU_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification_terminal_review_2026-08-04.json"
)
CPU_TERMINAL_REVIEW_SHA256 = (
    "9b0c31c05b4fb6064c67116a456d34a6f7e49cfe85ec55ed081599acb18502f0"
)
CPU_TERMINAL_REVIEW_BYTE_COUNT = 20_536

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v1"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_exact_plan_2026-08-04.json"
)

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-"
    "genesis-rocm-backend-v1-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_qualification_exact_plan_2026-08-04.json"
)

ROCM_VENV = REPO_ROOT / ".generated/venvs/genesis_rocm_0_4_6_v1"
ROCM_PYTHON = ROCM_VENV / "bin/python"
ROCM_SITE_PACKAGES = ROCM_VENV / "lib/python3.12/site-packages"
WORLD_MODEL_ROCM_SITE_PACKAGES = REPO_ROOT / (
    ".generated/venvs/world_model_rocm_7_2_1_v1/"
    "lib/python3.12/site-packages"
)
ROCM_PREFIX = Path("/opt/rocm-7.1.1")
ROCM_LLVM_BIN = ROCM_PREFIX / "lib/llvm/bin"
ROCM_EXECUTION_PATH = (
    "/opt/rocm-7.1.1/lib/llvm/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

ROCM_RUNTIME_PATHS = {
    "platform_manifest": REPO_ROOT / "config/go2_platform_manifest.yaml",
    "primitive_registry": REPO_ROOT / "config/go2_primitive_registry.yaml",
    "policy_checkpoint": REPO_ROOT / (
        ".generated/upstream_genesis/locomotion/logs/"
        "lewm-go2-contract-20260516T163413Z/model_500.pt"
    ),
    "policy_config": REPO_ROOT / (
        ".generated/upstream_genesis/locomotion/logs/"
        "lewm-go2-contract-20260516T163413Z/cfgs.pkl"
    ),
    "go2_urdf": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/urdf/go2.urdf",
    "go2_dae_base": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/base.dae",
    "go2_dae_calf": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/calf.dae",
    "go2_dae_calf_mirror": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/calf_mirror.dae",
    "go2_dae_foot": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/foot.dae",
    "go2_dae_hip": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/hip.dae",
    "go2_dae_thigh": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/thigh.dae",
    "go2_dae_thigh_mirror": ROCM_SITE_PACKAGES
    / "genesis/assets/urdf/go2/dae/thigh_mirror.dae",
    "python_executable_target": Path("/usr/bin/python3.12"),
    "python_environment_config": ROCM_VENV / "pyvenv.cfg",
    "world_model_python_environment_config": REPO_ROOT
    / ".generated/venvs/world_model_rocm_7_2_1_v1/pyvenv.cfg",
    "eglinfo_executable": Path("/usr/bin/eglinfo.x86_64-linux-gnu"),
    "rocminfo_executable": ROCM_PREFIX / "bin/rocminfo",
    # Bind the regular target, not the ld.lld symlink rejected by file_binding.
    "rocm_lld_executable": ROCM_LLVM_BIN / "lld",
    "genesis_distribution_metadata": ROCM_SITE_PACKAGES
    / "genesis_world-0.4.6.dist-info/METADATA",
    "genesis_distribution_record": ROCM_SITE_PACKAGES
    / "genesis_world-0.4.6.dist-info/RECORD",
    "genesis_init_source": ROCM_SITE_PACKAGES / "genesis/__init__.py",
    "genesis_constants_source": ROCM_SITE_PACKAGES / "genesis/constants.py",
    "genesis_misc_source": ROCM_SITE_PACKAGES / "genesis/utils/misc.py",
    "genesis_scene_source": ROCM_SITE_PACKAGES / "genesis/engine/scene.py",
    "genesis_camera_source": ROCM_SITE_PACKAGES / "genesis/vis/camera.py",
    "genesis_rasterizer_source": ROCM_SITE_PACKAGES
    / "genesis/vis/rasterizer.py",
    "genesis_egl_platform_source": ROCM_SITE_PACKAGES
    / "genesis/ext/pyrender/platforms/egl.py",
    "quadrants_distribution_metadata": ROCM_SITE_PACKAGES
    / "quadrants-0.6.2.dist-info/METADATA",
    "quadrants_distribution_record": ROCM_SITE_PACKAGES
    / "quadrants-0.6.2.dist-info/RECORD",
    "quadrants_init_source": ROCM_SITE_PACKAGES / "quadrants/__init__.py",
    "quadrants_lang_misc_source": ROCM_SITE_PACKAGES
    / "quadrants/lang/misc.py",
    "quadrants_lang_kernel_source": ROCM_SITE_PACKAGES
    / "quadrants/lang/kernel.py",
    "quadrants_lib_utils_source": ROCM_SITE_PACKAGES
    / "quadrants/_lib/utils.py",
    "quadrants_native_core": ROCM_SITE_PACKAGES
    / (
        "quadrants/_lib/core/"
        "quadrants_python.cpython-312-x86_64-linux-gnu.so"
    ),
    "quadrants_runtime_amdgpu_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime/runtime_amdgpu.bc",
    "quadrants_rocm70_opencl_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/opencl.bc",
    "quadrants_rocm70_ockl_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/ockl.bc",
    "quadrants_rocm70_ocml_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/ocml.bc",
    "quadrants_rocm70_isa_gfx1201_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_isa_version_1201.bc",
    "quadrants_rocm70_abi_v4_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_abi_version_400.bc",
    "quadrants_rocm70_correct_sqrt_off_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_correctly_rounded_sqrt_off.bc",
    "quadrants_rocm70_daz_off_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_daz_opt_off.bc",
    "quadrants_rocm70_finite_only_off_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_finite_only_off.bc",
    "quadrants_rocm70_unsafe_math_off_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_unsafe_math_off.bc",
    "quadrants_rocm70_wave64_off_bitcode": ROCM_SITE_PACKAGES
    / "quadrants/_lib/runtime_rocm70/oclc_wavefrontsize64_off.bc",
    "torch_distribution_metadata": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "torch-2.12.0+rocm7.2.dist-info/METADATA",
    "torch_distribution_record": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "torch-2.12.0+rocm7.2.dist-info/RECORD",
    "torch_site_packages_link": ROCM_SITE_PACKAGES
    / "world_model_rocm_7_2_1_v1.pth",
    "rsl_rl_distribution_metadata": ROCM_SITE_PACKAGES
    / "rsl_rl_lib-5.4.1.dist-info/METADATA",
    "rsl_rl_distribution_record": ROCM_SITE_PACKAGES
    / "rsl_rl_lib-5.4.1.dist-info/RECORD",
    "rsl_rl_on_policy_runner": ROCM_SITE_PACKAGES
    / "rsl_rl/runners/on_policy_runner.py",
    "tensordict_distribution_metadata": ROCM_SITE_PACKAGES
    / "tensordict-0.13.0.dist-info/METADATA",
    "tensordict_distribution_record": ROCM_SITE_PACKAGES
    / "tensordict-0.13.0.dist-info/RECORD",
    "torchvision_distribution_metadata": ROCM_SITE_PACKAGES
    / "torchvision-0.27.0+rocm7.2.dist-info/METADATA",
    "torchvision_distribution_record": ROCM_SITE_PACKAGES
    / "torchvision-0.27.0+rocm7.2.dist-info/RECORD",
    "numpy_distribution_metadata": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "numpy-2.4.6.dist-info/METADATA",
    "numpy_distribution_record": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "numpy-2.4.6.dist-info/RECORD",
    "pillow_distribution_metadata": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "pillow-11.3.0.dist-info/METADATA",
    "pillow_distribution_record": WORLD_MODEL_ROCM_SITE_PACKAGES
    / "pillow-11.3.0.dist-info/RECORD",
}

ROCM_EXECUTION_ENVIRONMENT_COMMON = {
    "EGL_DEVICE_ID": "1",
    "GS_BACKEND": "amdgpu",
    "GS_ENABLE_FASTCACHE": "1",
    "GS_ENABLE_NDARRAY": "1",
    "GS_ENABLE_ZEROCOPY": "1",
    "GS_PARA_LEVEL": "0",
    "HIP_VISIBLE_DEVICES": "0",
    "PATH": ROCM_EXECUTION_PATH,
    "PYOPENGL_PLATFORM": "egl",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "ROCM_PATH": str(ROCM_PREFIX),
    "ROCR_VISIBLE_DEVICES": "0",
}


def rocm_execution_environment(plan_role: str) -> dict[str, str]:
    """Return the exact role-local environment, including its fresh JIT cache."""

    if plan_role == "scientific":
        attempt_root = DEFAULT_ATTEMPT_ROOT
    elif plan_role == "qualification":
        attempt_root = QUALIFICATION_ATTEMPT_ROOT
    else:
        raise SceneDiversityGenesisRocmPlanError("ROCm plan role changed")
    return {
        **ROCM_EXECUTION_ENVIRONMENT_COMMON,
        "GS_CACHE_FILE_PATH": str(
            (attempt_root / "quadrants_cache").resolve(strict=False)
        ),
    }

ROCM_GRAPHICS_PREFLIGHT_EXPECTATION = {
    "drm_device_id": "0x7551",
    "drm_vendor_id": "0x1002",
    "egl_device_index": 1,
    "egl_renderer_name_contains": "AMD Radeon AI PRO R9700",
    "eglinfo_expected_exit_code": 2,
    "genesis_backend_symbol": "gs.amdgpu",
    "hip_arch_name": "gfx1201",
    "hip_device_index": 0,
    "hip_device_name": "AMD Radeon AI PRO R9700",
    "hip_visible_device_count": 1,
    "hsa_override_gfx_version_must_be_absent": True,
}

QUALIFICATION_SCENE_INDICES = (12, 0)
QUALIFICATION_WORKER_WATCHDOG_SECONDS = 300
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = 900
SCIENTIFIC_SCENE_COUNT = 64
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_successor_contract_v1"
)


class SceneDiversityGenesisRocmPlanError(RuntimeError):
    """Raised before a changed, unbound, or non-fresh plan can be emitted."""


def _protected(path: Path) -> bool:
    return any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        for part in Path(path).parts
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _exact_binding(path: Path, *, sha256: str, byte_count: int) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }
    try:
        actual = pilot.file_binding(path)
    except (OSError, pilot.PilotContractError) as exc:
        raise SceneDiversityGenesisRocmPlanError(str(exc)) from exc
    if actual != expected:
        raise SceneDiversityGenesisRocmPlanError(
            f"exact witness binding changed: {path}"
        )
    return expected


def _read_exact_json(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _exact_binding(path, sha256=sha256, byte_count=byte_count)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityGenesisRocmPlanError(
            f"{label} is not strict JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SceneDiversityGenesisRocmPlanError(f"{label} must be an object")
    return value, binding


def _load_immutable_witnesses() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    vulkan, vulkan_binding = _read_exact_json(
        FROZEN_V1_EXACT_PLAN,
        sha256=FROZEN_V1_EXACT_PLAN_SHA256,
        byte_count=FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
        label="frozen Vulkan scientific plan",
    )
    cpu, cpu_binding = _read_exact_json(
        FROZEN_CPU_EXACT_PLAN,
        sha256=FROZEN_CPU_EXACT_PLAN_SHA256,
        byte_count=FROZEN_CPU_EXACT_PLAN_BYTE_COUNT,
        label="frozen CPU scientific plan",
    )
    terminal, terminal_binding = _read_exact_json(
        CPU_TERMINAL_REVIEW,
        sha256=CPU_TERMINAL_REVIEW_SHA256,
        byte_count=CPU_TERMINAL_REVIEW_BYTE_COUNT,
        label="CPU qualification terminal review",
    )

    expected_cpu = copy.deepcopy(vulkan)
    expected_cpu["attempt_id"] = (
        "go2-scene-diversity-recurrent-replication-cpu-backend-v1"
    )
    expected_cpu["output_root"] = str(
        (
            REPO_ROOT
            / (
                ".generated/dev/"
                "go2_scene_diversity_recurrent_replication_cpu_backend_v1/"
                "attempt_v1/collection"
            )
        ).resolve(strict=False)
    )
    expected_cpu["execution_contract"]["backend"] = "cpu"
    expected_cpu["execution_contract"]["environment"]["GS_BACKEND"] = "cpu"
    if _canonical_bytes(cpu) != _canonical_bytes(expected_cpu):
        raise SceneDiversityGenesisRocmPlanError(
            "bound CPU scientific plan is not the exact four-field Vulkan overlay"
        )
    if (
        terminal.get("status")
        != "PASS_FAIL_CLOSED_PRE_GENESIS_QUALIFICATION_TERMINAL_REVIEW"
        or terminal.get("decision", {}).get("qualification_status")
        != "FAIL_CPU_BACKEND_QUALIFICATION_HARD_STOP"
        or terminal.get("decision", {}).get("attempt_consumed") is not True
        or terminal.get("successor_eligibility", {}).get(
            "only_eligible_next_backend_direction"
        )
        != "Genesis 0.4.6 ROCm/HIP backend under a separate fresh qualification design"
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "CPU terminal review does not permit this separately qualified direction"
        )
    return (
        copy.deepcopy(vulkan),
        copy.deepcopy(cpu),
        copy.deepcopy(terminal),
        copy.deepcopy(vulkan_binding),
        copy.deepcopy(cpu_binding),
        copy.deepcopy(terminal_binding),
    )


(
    _IMMUTABLE_FROZEN_VULKAN_PLAN,
    _IMMUTABLE_FROZEN_CPU_PLAN,
    _IMMUTABLE_CPU_TERMINAL_REVIEW,
    FROZEN_V1_EXACT_PLAN_BINDING,
    FROZEN_CPU_EXACT_PLAN_BINDING,
    CPU_TERMINAL_REVIEW_BINDING,
) = _load_immutable_witnesses()


def _binding_shape(binding: object, *, label: str) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise SceneDiversityGenesisRocmPlanError(
            f"{label} binding shape changed"
        )
    path = binding["path"]
    digest = binding["file_sha256"]
    byte_count = binding["byte_count"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or _protected(Path(path))
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise SceneDiversityGenesisRocmPlanError(f"{label} binding is invalid")
    return dict(binding)


def build_rocm_runtime_bindings() -> dict[str, dict[str, Any]]:
    """Bind the exact interpreter, compiler, ROCm, policy, and render inputs."""

    bindings: dict[str, dict[str, Any]] = {}
    for name, path in ROCM_RUNTIME_PATHS.items():
        try:
            bindings[name] = pilot.file_binding(path)
        except (OSError, pilot.PilotContractError) as exc:
            raise SceneDiversityGenesisRocmPlanError(
                f"cannot bind ROCm runtime input {name}: {exc}"
            ) from exc
    if Path(ROCM_PYTHON).resolve(strict=True) != Path(
        bindings["python_executable_target"]["path"]
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "ROCm venv Python does not resolve to the bound executable target"
        )
    bound_lld_target = Path(bindings["rocm_lld_executable"]["path"])
    path_lld = shutil.which(
        "ld.lld", path=ROCM_EXECUTION_ENVIRONMENT_COMMON["PATH"]
    )
    rocm_path_lld = (
        Path(ROCM_EXECUTION_ENVIRONMENT_COMMON["ROCM_PATH"])
        / "llvm/bin/ld.lld"
    )
    if (
        path_lld is None
        or Path(path_lld).resolve(strict=True) != bound_lld_target
        or rocm_path_lld.resolve(strict=True) != bound_lld_target
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "PATH and ROCM_PATH do not resolve to the bound ROCm LLD target"
        )
    metadata_identities = {
        "genesis_distribution_metadata": ("genesis-world", "0.4.6"),
        "quadrants_distribution_metadata": ("quadrants", "0.6.2"),
        "torch_distribution_metadata": ("torch", "2.12.0+rocm7.2"),
        "rsl_rl_distribution_metadata": ("rsl-rl-lib", "5.4.1"),
        "tensordict_distribution_metadata": ("tensordict", "0.13.0"),
        "torchvision_distribution_metadata": (
            "torchvision",
            "0.27.0+rocm7.2",
        ),
        "numpy_distribution_metadata": ("numpy", "2.4.6"),
        "pillow_distribution_metadata": ("pillow", "11.3.0"),
    }
    for binding_name, (distribution_name, version) in metadata_identities.items():
        try:
            metadata = ROCM_RUNTIME_PATHS[binding_name].read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise SceneDiversityGenesisRocmPlanError(
                f"cannot read bound distribution metadata: {binding_name}"
            ) from exc
        if (
            f"Name: {distribution_name}\n" not in metadata
            or f"Version: {version}\n" not in metadata
        ):
            raise SceneDiversityGenesisRocmPlanError(
                f"bound distribution identity changed: {binding_name}"
            )
    expected_link = f"{WORLD_MODEL_ROCM_SITE_PACKAGES.resolve(strict=True)}\n"
    try:
        link_text = ROCM_RUNTIME_PATHS["torch_site_packages_link"].read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise SceneDiversityGenesisRocmPlanError(
            "cannot read bound Torch site-packages link"
        ) from exc
    if link_text != expected_link:
        raise SceneDiversityGenesisRocmPlanError(
            "Torch site-packages link target changed"
        )
    return bindings


def _validate_rocm_runtime_bindings(
    runtime_bindings: Mapping[str, Any], *, rehash: bool
) -> dict[str, dict[str, Any]]:
    if not isinstance(runtime_bindings, Mapping) or set(runtime_bindings) != set(
        ROCM_RUNTIME_PATHS
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "ROCm runtime binding field set changed"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, expected_path in ROCM_RUNTIME_PATHS.items():
        binding = _binding_shape(runtime_bindings[name], label=name)
        try:
            resolved_expected = expected_path.resolve(strict=True)
        except OSError as exc:
            raise SceneDiversityGenesisRocmPlanError(
                f"ROCm runtime input is absent: {name}"
            ) from exc
        if Path(binding["path"]) != resolved_expected:
            raise SceneDiversityGenesisRocmPlanError(
                f"ROCm runtime binding path changed: {name}"
            )
        if rehash:
            try:
                actual = pilot.file_binding(expected_path)
            except (OSError, pilot.PilotContractError) as exc:
                raise SceneDiversityGenesisRocmPlanError(str(exc)) from exc
            if actual != binding:
                raise SceneDiversityGenesisRocmPlanError(
                    f"ROCm runtime binding content changed: {name}"
                )
        normalized[name] = binding
    return normalized


def _successor_contract(*, plan_role: str) -> dict[str, Any]:
    if plan_role not in {"scientific", "qualification"}:
        raise SceneDiversityGenesisRocmPlanError("ROCm plan role changed")
    qualification = plan_role == "qualification"
    return {
        "schema": SUCCESSOR_CONTRACT_SCHEMA,
        "plan_role": plan_role,
        "frozen_vulkan_scientific_plan_binding": copy.deepcopy(
            FROZEN_V1_EXACT_PLAN_BINDING
        ),
        "frozen_cpu_scientific_plan_binding": copy.deepcopy(
            FROZEN_CPU_EXACT_PLAN_BINDING
        ),
        "cpu_qualification_terminal_review_binding": copy.deepcopy(
            CPU_TERMINAL_REVIEW_BINDING
        ),
        "genesis_world_version": "0.4.6",
        "quadrants_version": "0.6.2",
        "torch_version": "2.12.0+rocm7.2",
        "torchvision_version": "0.27.0+rocm7.2",
        "tensordict_version": "0.13.0",
        "rsl_rl_version": "5.4.1",
        "genesis_backend_symbol": "gs.amdgpu",
        "qualification_scene_indices_in_order": (
            list(QUALIFICATION_SCENE_INDICES) if qualification else []
        ),
        "qualification_worker_watchdog_seconds": (
            QUALIFICATION_WORKER_WATCHDOG_SECONDS if qualification else None
        ),
        "qualification_timing_gate": (
            {
                "scene_count": SCIENTIFIC_SCENE_COUNT,
                "fixed_noncollection_reserve_seconds": (
                    QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
                ),
                "formula": (
                    "64 * max(worker_elapsed_seconds) + 900 <= 7200"
                ),
                "scientific_wall_cap_seconds": 7_200,
            }
            if qualification
            else None
        ),
        "qualification_execution_authorized": False,
        "scientific_execution_authorized": False,
        "probe_output_reuse_authorized": False,
    }


def _expected_rocm_plan(
    *,
    attempt_id: str,
    output_root: Path,
    plan_role: str,
    runtime_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = copy.deepcopy(_IMMUTABLE_FROZEN_VULKAN_PLAN)
    candidate["attempt_id"] = attempt_id
    candidate["output_root"] = str(output_root.resolve(strict=False))
    candidate["runtime_bindings"] = copy.deepcopy(dict(runtime_bindings))
    execution = candidate["execution_contract"]
    execution["backend"] = "amdgpu"
    # Preserve the venv launcher path.  Resolving this symlink to
    # /usr/bin/python3.12 discards the venv prefix and makes Genesis/Torch
    # unavailable even though the executable target itself is correctly
    # bound separately below.
    execution["python_invocation_path"] = str(ROCM_PYTHON.absolute())
    execution["environment"] = rocm_execution_environment(plan_role)
    execution["graphics_preflight"] = copy.deepcopy(
        ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
    )
    candidate["successor_contract"] = _successor_contract(plan_role=plan_role)
    return candidate


def validate_rocm_plan(
    plan: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    expected_output_root: Path,
    plan_role: str,
) -> dict[str, Any]:
    """Validate a ROCm plan without installing any mutable global overlay."""

    if not isinstance(plan, Mapping):
        raise SceneDiversityGenesisRocmPlanError("ROCm plan must be an object")
    candidate = copy.deepcopy(dict(plan))
    runtime = _validate_rocm_runtime_bindings(
        candidate.get("runtime_bindings", {}), rehash=True
    )
    expected = _expected_rocm_plan(
        attempt_id=expected_attempt_id,
        output_root=expected_output_root,
        plan_role=plan_role,
        runtime_bindings=runtime,
    )
    if _canonical_bytes(candidate) != _canonical_bytes(expected):
        raise SceneDiversityGenesisRocmPlanError(
            "ROCm plan changed beyond the exact successor overlay"
        )
    environment = candidate["execution_contract"]["environment"]
    if "HSA_OVERRIDE_GFX_VERSION" in environment:
        raise SceneDiversityGenesisRocmPlanError(
            "HSA_OVERRIDE_GFX_VERSION must be absent on gfx1201"
        )
    if not str(environment["PATH"]).startswith(
        f"{ROCM_LLVM_BIN}:"
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "ROCm LLVM directory must be first on PATH"
        )
    return candidate


def _require_exact_frozen_input(frozen_plan: Mapping[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(frozen_plan))
    if _canonical_bytes(candidate) != _canonical_bytes(
        _IMMUTABLE_FROZEN_VULKAN_PLAN
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "frozen Vulkan scientific plan binding or content changed"
        )
    return candidate


def _require_fresh_exact_root(
    *, output_root: Path, expected_root: Path, attempt_root: Path, label: str
) -> Path:
    selected = Path(output_root)
    development = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    resolved = selected.resolve(strict=False)
    if (
        not selected.is_absolute()
        or not resolved.is_relative_to(development)
        or resolved != expected_root.resolve(strict=False)
        or attempt_root.exists()
        or attempt_root.is_symlink()
        or selected.exists()
        or selected.is_symlink()
        or _protected(selected)
    ):
        raise SceneDiversityGenesisRocmPlanError(
            f"{label} output_root must be its exact fresh development path"
        )
    return resolved


def build_rocm_plan(
    *,
    frozen_plan: Mapping[str, Any],
    attempt_id: str,
    output_root: Path,
    expected_attempt_id: str,
    expected_output_root: Path,
    attempt_root: Path,
    plan_role: str,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if attempt_id != expected_attempt_id:
        raise SceneDiversityGenesisRocmPlanError(
            f"{plan_role} ROCm attempt identifier changed"
        )
    _require_exact_frozen_input(frozen_plan)
    selected_root = _require_fresh_exact_root(
        output_root=output_root,
        expected_root=expected_output_root,
        attempt_root=attempt_root,
        label=plan_role,
    )
    selected_runtime = (
        build_rocm_runtime_bindings()
        if runtime_bindings is None
        else _validate_rocm_runtime_bindings(runtime_bindings, rehash=True)
    )
    candidate = _expected_rocm_plan(
        attempt_id=attempt_id,
        output_root=selected_root,
        plan_role=plan_role,
        runtime_bindings=selected_runtime,
    )
    return validate_rocm_plan(
        candidate,
        expected_attempt_id=expected_attempt_id,
        expected_output_root=expected_output_root,
        plan_role=plan_role,
    )


def build_scientific_plan(
    *,
    frozen_plan: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_rocm_plan(
        frozen_plan=frozen_plan,
        attempt_id=DEFAULT_ATTEMPT_ID,
        output_root=DEFAULT_OUTPUT_ROOT,
        expected_attempt_id=DEFAULT_ATTEMPT_ID,
        expected_output_root=DEFAULT_OUTPUT_ROOT,
        attempt_root=DEFAULT_ATTEMPT_ROOT,
        plan_role="scientific",
        runtime_bindings=runtime_bindings,
    )


def build_qualification_plan(
    *,
    frozen_plan: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_rocm_plan(
        frozen_plan=frozen_plan,
        attempt_id=QUALIFICATION_ATTEMPT_ID,
        output_root=QUALIFICATION_OUTPUT_ROOT,
        expected_attempt_id=QUALIFICATION_ATTEMPT_ID,
        expected_output_root=QUALIFICATION_OUTPUT_ROOT,
        attempt_root=QUALIFICATION_ATTEMPT_ROOT,
        plan_role="qualification",
        runtime_bindings=runtime_bindings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument(
        "--qualification-plan-output",
        type=Path,
        default=QUALIFICATION_PLAN_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(
        path.exists() or path.is_symlink()
        for path in (args.plan_output, args.qualification_plan_output)
    ):
        raise SceneDiversityGenesisRocmPlanError(
            "ROCm plan outputs must be fresh"
        )
    runtime = build_rocm_runtime_bindings()
    frozen = copy.deepcopy(_IMMUTABLE_FROZEN_VULKAN_PLAN)
    science = build_scientific_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )
    qualification = build_qualification_plan(
        frozen_plan=frozen, runtime_bindings=runtime
    )
    science_binding = pilot.write_json_exclusive(args.plan_output, science)
    qualification_binding = pilot.write_json_exclusive(
        args.qualification_plan_output, qualification
    )
    print(
        json.dumps(
            {
                "scientific_plan": science_binding,
                "qualification_plan": qualification_binding,
                "frozen_vulkan_scientific_plan": FROZEN_V1_EXACT_PLAN_BINDING,
                "frozen_cpu_scientific_plan": FROZEN_CPU_EXACT_PLAN_BINDING,
                "cpu_qualification_terminal_review": (
                    CPU_TERMINAL_REVIEW_BINDING
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_TERMINAL_REVIEW",
    "CPU_TERMINAL_REVIEW_BINDING",
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PLAN_OUTPUT",
    "FROZEN_CPU_EXACT_PLAN",
    "FROZEN_V1_EXACT_PLAN",
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "QUALIFICATION_SCENE_INDICES",
    "ROCM_EXECUTION_ENVIRONMENT_COMMON",
    "ROCM_GRAPHICS_PREFLIGHT_EXPECTATION",
    "ROCM_PYTHON",
    "ROCM_RUNTIME_PATHS",
    "SceneDiversityGenesisRocmPlanError",
    "build_qualification_plan",
    "build_rocm_runtime_bindings",
    "build_scientific_plan",
    "rocm_execution_environment",
    "validate_rocm_plan",
]
