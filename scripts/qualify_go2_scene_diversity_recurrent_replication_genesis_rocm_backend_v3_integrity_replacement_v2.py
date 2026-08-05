#!/usr/bin/env python3
"""Qualify the science-identical V3 integrity replacement V2.

The fresh replacement root is consumed before this module invokes any
preflight.  V3's HOME, Python, ``ld.lld``, selectors, HIP, EGL, and both
probes remain unchanged.  A closed collector/qualifier interface is validated
before any mutable predecessor overlay or reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as collector  # noqa: E402
from scripts import qualify_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1 as predecessor  # noqa: E402


pilot = collector.pilot
kernel = collector.kernel
# The inherited scene implementation, not either ROCm identity adapter.
scene_predecessor = predecessor.scene_predecessor

QUALIFICATION_AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_authority_v1"
)
QUALIFICATION_AUTHORITY_STATUS = (
    "AUTHORIZED_GENESIS_ROCM_BACKEND_V3_INTEGRITY_REPLACEMENT_V2_QUALIFICATION_ONLY"
)
QUALIFICATION_AUTHORITY_FIELDS = frozenset(
    set(predecessor.QUALIFICATION_AUTHORITY_FIELDS)
    | {"predecessor_replacement_v1_qualification_terminal_review_binding"}
)
QUALIFICATION_RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_reservation_v1"
)
QUALIFICATION_SCENE_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_scene_result_v1"
)
QUALIFICATION_RESULT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_result_v1"
)
QUALIFICATION_TERMINAL_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_terminal_v1"
)
QUALIFICATION_RESULT_STATUS = "PASS_GENESIS_ROCM_BACKEND_V3_INTEGRITY_REPLACEMENT_V2_QUALIFICATION"
QUALIFICATION_PROBE_ORDER = tuple(plan_builder.QUALIFICATION_SCENE_INDICES)
WORKER_TIMEOUT_SECONDS = float(
    plan_builder.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS = float(
    plan_builder.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)

QUALIFICATION_CONTRACT = {
    **copy.deepcopy(predecessor.QUALIFICATION_CONTRACT),
    "ld_lld_driver_entrypoint": str(
        plan_builder.ROCM_LD_LLD_DRIVER_ENTRYPOINT
    ),
    "ld_lld_driver_link_text": plan_builder.ROCM_LD_LLD_DRIVER_LINK_TEXT,
    "ld_lld_version_stdout_prefix": (
        plan_builder.ROCM_LLD_VERSION_STDOUT_PREFIX
    ),
    "direct_bound_lld_target_invocation_forbidden": True,
    "v2_runtime_payload_reuse_authorized": False,
    "v3_runtime_payload_reuse_authorized": False,
    "replacement_v1_runtime_payload_reuse_authorized": False,
    "required_host_home": plan_builder.REQUIRED_HOST_HOME,
}
QUALIFICATION_RESULT_FIELDS = predecessor.QUALIFICATION_RESULT_FIELDS

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_preregistration_2026-08-04.md"
)
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_source_review_2026-08-04.json"
)
QUALIFICATION_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_authority_2026-08-04.json"
)
QUALIFICATION_RESULT_PATH = (
    plan_builder.QUALIFICATION_ATTEMPT_ROOT / "qualification_result.json"
)

_ROCM_WORKER_ARGV = collector._worker_argv_rocm  # noqa: SLF001
_CONFIGURATION_LOCK = threading.RLock()


class GenesisRocmBackendV3IntegrityReplacementV2QualificationError(RuntimeError):
    """Raised when fresh V3 qualification fails closed."""


GenesisRocmBackendQualificationError = GenesisRocmBackendV3IntegrityReplacementV2QualificationError


_QUALIFIER_COMMON_OVERRIDE_KEYS = frozenset(
    {
        "plan_builder",
        "collector",
        "kernel",
        "QUALIFICATION_AUTHORITY_SCHEMA",
        "QUALIFICATION_AUTHORITY_STATUS",
        "QUALIFICATION_AUTHORITY_FIELDS",
        "QUALIFICATION_RESERVATION_SCHEMA",
        "QUALIFICATION_SCENE_RESULT_SCHEMA",
        "QUALIFICATION_RESULT_SCHEMA",
        "QUALIFICATION_TERMINAL_SCHEMA",
        "QUALIFICATION_RESULT_STATUS",
        "QUALIFICATION_PROBE_ORDER",
        "QUALIFICATION_CONTRACT",
        "PREREGISTRATION",
        "SOURCE_REVIEW",
        "QUALIFICATION_AUTHORITY",
        "QUALIFICATION_RESULT_PATH",
        "_ROCM_WORKER_ARGV",
        "_source_paths",
        "_run_rocm_egl_preflight",
        "_child_environment",
        "_worker_argv_qualification",
        "GenesisRocmBackendQualificationError",
    }
)
_REPLACEMENT_V2_QUALIFIER_OVERRIDE_KEYS = frozenset(
    set(_QUALIFIER_COMMON_OVERRIDE_KEYS)
    | {"GenesisRocmBackendV3IntegrityReplacementV1QualificationError"}
)
_REPLACEMENT_V1_QUALIFIER_OVERRIDE_KEYS = frozenset(
    set(_QUALIFIER_COMMON_OVERRIDE_KEYS)
    | {"GenesisRocmBackendV3QualificationError"}
)
_V3_QUALIFIER_OVERRIDE_KEYS = frozenset(
    set(_QUALIFIER_COMMON_OVERRIDE_KEYS)
    | {"GenesisRocmBackendV2QualificationError"}
)
_V2_QUALIFIER_OVERRIDE_KEYS = frozenset(
    (set(_QUALIFIER_COMMON_OVERRIDE_KEYS) - {"_child_environment"})
    | {"predecessor"}
)
_V1_QUALIFICATION_COLLECTOR_OVERRIDE_KEYS = frozenset(
    {
        "AUTHORITY_FIELDS",
        "AUTHORITY_SCHEMA",
        "AUTHORITY_STATUS",
        "ATTEMPT_ID",
        "RESERVATION_SCHEMA",
        "SCENE_RESULT_SCHEMA",
        "_read_collection_reservation_rocm",
        "_worker_argv_rocm",
    }
)


QUALIFIER_COMPATIBILITY_MATRIX = MappingProxyType(
    {
        "v1_runner_and_replay": (
            "pilot",
            "kernel",
            "predecessor",
            "PREREGISTRATION",
            "SOURCE_REVIEW",
            "QUALIFICATION_AUTHORITY",
            "QUALIFICATION_AUTHORITY_FIELDS",
            "QUALIFICATION_AUTHORITY_SCHEMA",
            "QUALIFICATION_AUTHORITY_STATUS",
            "QUALIFICATION_CONTRACT",
            "QUALIFICATION_PROBE_ORDER",
            "QUALIFICATION_RESULT_FIELDS",
            "QUALIFICATION_RESULT_PATH",
            "QUALIFICATION_RESULT_SCHEMA",
            "QUALIFICATION_RESULT_STATUS",
            "SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS",
            "WORKER_TIMEOUT_SECONDS",
            "_all_numbers_finite",
            "_configured_qualification_collector",
            "_read_qualification_reservation",
        ),
        "v2_v3_and_replacement_runners": (
            "__file__",
            "predecessor",
            "scene_predecessor",
            "PREREGISTRATION",
            "SOURCE_REVIEW",
            "QUALIFICATION_AUTHORITY",
        ),
        "runtime_entrypoints": (
            "validate_qualification_authority",
            "execute_qualification",
            "_worker_argv_qualification",
            "_child_environment",
            "build_parser",
            "main",
        ),
        "replacement_v2_to_replacement_v1": tuple(
            sorted(_REPLACEMENT_V2_QUALIFIER_OVERRIDE_KEYS)
        ),
        "replacement_v1_to_v3": tuple(
            sorted(_REPLACEMENT_V1_QUALIFIER_OVERRIDE_KEYS)
        ),
        "v3_to_v2": tuple(sorted(_V3_QUALIFIER_OVERRIDE_KEYS)),
        "v2_to_v1": tuple(sorted(_V2_QUALIFIER_OVERRIDE_KEYS)),
        "v1_qualification_collector": tuple(
            sorted(_V1_QUALIFICATION_COLLECTOR_OVERRIDE_KEYS)
        ),
    }
)


def validate_qualifier_compatibility_facade() -> dict[str, Any]:
    collector_audit = collector.validate_collector_compatibility_facade()
    missing = {
        layer: [
            name
            for name in QUALIFIER_COMPATIBILITY_MATRIX[layer]
            if name not in globals()
        ]
        for layer in (
            "v1_runner_and_replay",
            "v2_v3_and_replacement_runners",
            "runtime_entrypoints",
        )
    }
    missing = {layer: names for layer, names in missing.items() if names}
    if missing:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            f"replacement V2 qualifier compatibility facade is incomplete: {missing}"
        )
    replacement_v1 = predecessor
    v3 = replacement_v1.predecessor
    v2 = v3.predecessor
    observed = {
        "replacement_v2_to_replacement_v1": set(
            _configuration_overrides_v3()
        ),
        "replacement_v1_to_v3": set(
            replacement_v1._configuration_overrides_v3()  # noqa: SLF001
        ),
        "v3_to_v2": set(v3._configuration_overrides_v3()),  # noqa: SLF001
        "v2_to_v1": set(v2._configuration_overrides_v2()),  # noqa: SLF001
    }
    expected = {
        "replacement_v2_to_replacement_v1": (
            _REPLACEMENT_V2_QUALIFIER_OVERRIDE_KEYS
        ),
        "replacement_v1_to_v3": _REPLACEMENT_V1_QUALIFIER_OVERRIDE_KEYS,
        "v3_to_v2": _V3_QUALIFIER_OVERRIDE_KEYS,
        "v2_to_v1": _V2_QUALIFIER_OVERRIDE_KEYS,
    }
    mismatches = {
        layer: {
            "missing": sorted(set(expected[layer]) - values),
            "unexpected": sorted(values - set(expected[layer])),
        }
        for layer, values in observed.items()
        if values != set(expected[layer])
    }
    expected_authority_fields = frozenset(
        set(predecessor.QUALIFICATION_AUTHORITY_FIELDS)
        | {"predecessor_replacement_v1_qualification_terminal_review_binding"}
    )
    if (
        QUALIFICATION_AUTHORITY_FIELDS != expected_authority_fields
        or len(QUALIFICATION_AUTHORITY_FIELDS) != 19
        or QUALIFICATION_RESULT_FIELDS != predecessor.QUALIFICATION_RESULT_FIELDS
        or len(QUALIFICATION_RESULT_FIELDS) != 20
    ):
        mismatches["field_closure"] = {
            "missing": [],
            "unexpected": ["authority_or_result_field_closure_changed"],
        }
    if mismatches:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            f"replacement V2 qualifier layered interface changed: {mismatches}"
        )
    return {
        "schema": (
            "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
            "backend_v3_integrity_replacement_v2_qualifier_compatibility_v1"
        ),
        "collector": collector_audit,
        "layers": {
            layer: list(names)
            for layer, names in QUALIFIER_COMPATIBILITY_MATRIX.items()
        },
        "passed": True,
    }


def _source_paths() -> Mapping[str, Path]:
    from scripts import run_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2 as runner

    return runner.SOURCE_PATHS


def _configuration_overrides_v3() -> dict[str, object]:
    frozen_source_paths = dict(_source_paths())

    def source_paths_before_overlay() -> Mapping[str, Path]:
        return dict(frozen_source_paths)

    return {
        "plan_builder": plan_builder,
        "collector": collector,
        "kernel": kernel,
        "QUALIFICATION_AUTHORITY_SCHEMA": QUALIFICATION_AUTHORITY_SCHEMA,
        "QUALIFICATION_AUTHORITY_STATUS": QUALIFICATION_AUTHORITY_STATUS,
        "QUALIFICATION_AUTHORITY_FIELDS": QUALIFICATION_AUTHORITY_FIELDS,
        "QUALIFICATION_RESERVATION_SCHEMA": QUALIFICATION_RESERVATION_SCHEMA,
        "QUALIFICATION_SCENE_RESULT_SCHEMA": (
            QUALIFICATION_SCENE_RESULT_SCHEMA
        ),
        "QUALIFICATION_RESULT_SCHEMA": QUALIFICATION_RESULT_SCHEMA,
        "QUALIFICATION_TERMINAL_SCHEMA": QUALIFICATION_TERMINAL_SCHEMA,
        "QUALIFICATION_RESULT_STATUS": QUALIFICATION_RESULT_STATUS,
        "QUALIFICATION_PROBE_ORDER": QUALIFICATION_PROBE_ORDER,
        "QUALIFICATION_CONTRACT": QUALIFICATION_CONTRACT,
        "PREREGISTRATION": PREREGISTRATION,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "QUALIFICATION_AUTHORITY": QUALIFICATION_AUTHORITY,
        "QUALIFICATION_RESULT_PATH": QUALIFICATION_RESULT_PATH,
        "_ROCM_WORKER_ARGV": _ROCM_WORKER_ARGV,
        "_source_paths": source_paths_before_overlay,
        "_run_rocm_egl_preflight": _run_rocm_egl_preflight,
        "_child_environment": _child_environment,
        "_worker_argv_qualification": _worker_argv_qualification,
        "GenesisRocmBackendQualificationError": (
            GenesisRocmBackendV3IntegrityReplacementV2QualificationError
        ),
        "GenesisRocmBackendV3IntegrityReplacementV1QualificationError": (
            GenesisRocmBackendV3IntegrityReplacementV2QualificationError
        ),
    }


@contextmanager
def _configured_predecessor_qualifier_v3() -> Iterator[None]:
    validate_qualifier_compatibility_facade()
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {
            name: getattr(predecessor, name) for name in overrides
        }
        try:
            for name, value in overrides.items():
                setattr(predecessor, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor, name, value)


@contextmanager
def _configured_qualification_collector() -> Iterator[None]:
    with _configured_predecessor_qualifier_v3():
        with predecessor._configured_qualification_collector():  # noqa: SLF001
            yield


def _run_rocm_egl_preflight(
    plan: Mapping[str, Any], *, child_env: Mapping[str, str]
) -> dict[str, Any]:
    """Invoke the exact unresolved ``ld.lld`` driver, then unchanged probes."""

    runtime = plan["runtime_bindings"]
    expectation = plan["execution_contract"]["graphics_preflight"]
    if expectation != plan_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 ROCm preflight expectation changed"
        )
    for name in (
        "eglinfo_executable",
        "rocminfo_executable",
        "rocm_lld_executable",
        "python_executable_target",
    ):
        pilot.require_binding(runtime[name], label=f"V3 preflight {name}")

    bound_target = Path(
        str(runtime["rocm_lld_executable"]["path"])
    )
    driver = Path(str(expectation["rocm_lld_driver_entrypoint"]))
    path_driver_text = shutil.which("ld.lld", path=str(child_env["PATH"]))
    rocm_path_driver = (
        Path(str(child_env["ROCM_PATH"])) / "lib/llvm/bin/ld.lld"
    )
    try:
        path_driver = Path(str(path_driver_text))
        path_link_text = os.readlink(path_driver)
        rocm_link_text = os.readlink(rocm_path_driver)
        resolved_path_target = path_driver.resolve(strict=True)
        resolved_rocm_target = rocm_path_driver.resolve(strict=True)
    except OSError as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 ld.lld driver entrypoint is unavailable"
        ) from exc
    if (
        path_driver_text != str(driver)
        or path_driver != driver
        or rocm_path_driver != driver
        or not driver.is_symlink()
        or path_link_text != expectation["rocm_lld_driver_link_text"]
        or rocm_link_text != expectation["rocm_lld_driver_link_text"]
        or resolved_path_target != bound_target
        or resolved_rocm_target != bound_target
        or str(bound_target)
        != expectation["rocm_lld_resolved_target_path"]
        or expectation["rocm_lld_direct_target_invocation_forbidden"]
        is not True
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 PATH/ROCM_PATH ld.lld driver identity changed"
        )

    lld_argv = [str(driver), "--version"]
    lld = subprocess.run(
        lld_argv,
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30.0,
    )
    version_prefix = str(expectation["rocm_lld_version_stdout_prefix"])
    if lld.returncode != 0 or not lld.stdout.startswith(version_prefix):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "exact ld.lld Unix-driver preflight failed"
        )

    rocminfo = subprocess.run(
        [str(runtime["rocminfo_executable"]["path"])],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60.0,
    )
    if rocminfo.returncode != 0 or "gfx1201" not in rocminfo.stdout:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "rocminfo gfx1201 preflight failed"
        )

    query_code = """
import json, os
from pathlib import Path
import genesis as gs
import numpy
import PIL
import torch
p = torch.cuda.get_device_properties(0)
print(json.dumps({
    'torch_version': torch.__version__,
    'torch_hip_version': torch.version.hip,
    'visible_device_count': torch.cuda.device_count(),
    'device_name': torch.cuda.get_device_name(0),
    'arch_name': str(getattr(p, 'gcnArchName', '')),
    'genesis_version': gs.__version__,
    'genesis_backend_symbol': 'gs.amdgpu' if int(gs.amdgpu) == 3 else 'unexpected',
    'home': os.environ.get('HOME'),
    'hsa_override_present': 'HSA_OVERRIDE_GFX_VERSION' in os.environ,
    'genesis_file': str(Path(gs.__file__).resolve()),
    'torch_file': str(Path(torch.__file__).resolve()),
    'numpy_file': str(Path(numpy.__file__).resolve()),
    'pillow_file': str(Path(PIL.__file__).resolve()),
}, sort_keys=True))
"""
    hip = subprocess.run(
        [
            str(plan["execution_contract"]["python_invocation_path"]),
            "-I",
            "-B",
            "-c",
            query_code,
        ],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    identity_lines = [
        line.strip() for line in hip.stdout.splitlines() if line.strip()
    ]
    try:
        identity = json.loads(identity_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "HIP identity preflight emitted malformed output"
        ) from exc
    if not isinstance(identity, dict):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "HIP identity preflight emitted a non-object"
        )
    try:
        module_paths = {
            name: Path(str(identity[name])).resolve(strict=True)
            for name in (
                "genesis_file",
                "torch_file",
                "numpy_file",
                "pillow_file",
            )
        }
        world_model_site = Path(
            str(runtime["torch_distribution_metadata"]["path"])
        ).parent.parent.resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "HIP module-path identity preflight failed"
        ) from exc
    if (
        hip.returncode != 0
        or identity.get("torch_version") != "2.12.0+rocm7.2"
        or not str(identity.get("torch_hip_version", "")).startswith("7.2")
        or identity.get("visible_device_count")
        != expectation["hip_visible_device_count"]
        or identity.get("device_name") != expectation["hip_device_name"]
        or not str(identity.get("arch_name", "")).startswith(
            str(expectation["hip_arch_name"])
        )
        or identity.get("genesis_version") != "0.4.6"
        or identity.get("genesis_backend_symbol") != "gs.amdgpu"
        or identity.get("home") != plan_builder.REQUIRED_HOST_HOME
        or identity.get("hsa_override_present") is not False
        or module_paths["genesis_file"]
        != Path(str(runtime["genesis_init_source"]["path"])).resolve(
            strict=True
        )
        or any(
            not module_paths[name].is_relative_to(world_model_site)
            for name in ("torch_file", "numpy_file", "pillow_file")
        )
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "HIP/Genesis R9700 identity preflight failed"
        )

    egl = subprocess.run(
        [str(runtime["eglinfo_executable"]["path"]), "-B"],
        cwd=REPO_ROOT,
        env=dict(child_env),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=90.0,
    )
    device_index = int(expectation["egl_device_index"])
    sections = list(
        re.finditer(
            rf"Device #{device_index}:\s*\n(?P<section>.*?)(?=\nDevice #\d+:|\Z)",
            egl.stdout,
            re.DOTALL,
        )
    )
    if (
        egl.returncode != expectation["eglinfo_expected_exit_code"]
        or len(sections) != 1
        or expectation["egl_renderer_name_contains"]
        not in sections[0].group("section")
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "EGL R9700 preflight failed"
        )
    return {
        "status": "PASS_EXACT_ROCM_HIP_AND_EGL_R9700",
        "environment": dict(plan["execution_contract"]["environment"]),
        "expectation": copy.deepcopy(expectation),
        "identity": identity,
        "egl_device_index": device_index,
        # New V3 evidence keeps lexical drivers separate from their target.
        # Retain the inherited resolved-target fields for the full frozen
        # qualification validator; V3 adds and independently validates the
        # exact lexical driver fields below.
        "path_ld_lld": str(bound_target),
        "rocm_path_ld_lld": str(bound_target),
        "path_ld_lld_driver": str(driver),
        "rocm_path_ld_lld_driver": str(driver),
        "lld_driver_entrypoint": str(driver),
        "lld_driver_link_text": path_link_text,
        "lld_resolved_target": str(bound_target),
        "lld_invocation_argv": lld_argv,
        "lld_version_prefix_passed": True,
        "lld_stdout_sha256": hashlib.sha256(lld.stdout.encode()).hexdigest(),
        "rocminfo_stdout_sha256": hashlib.sha256(
            rocminfo.stdout.encode()
        ).hexdigest(),
        "egl_stdout_sha256": hashlib.sha256(egl.stdout.encode()).hexdigest(),
        "egl_stderr_sha256": hashlib.sha256(egl.stderr.encode()).hexdigest(),
        "egl_exit_code": egl.returncode,
    }


def _worker_argv_qualification(**kwargs: Any) -> list[str]:
    validate_qualifier_compatibility_facade()
    argv = _ROCM_WORKER_ARGV(**kwargs)
    if (
        len(argv) < 2
        or Path(argv[1]).resolve() != Path(collector.__file__).resolve()
    ):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 ROCm worker entry point changed"
        )
    argv[1] = str(Path(__file__).resolve())
    return argv


def _read_qualification_reservation(*args: Any, **kwargs: Any):
    validate_qualifier_compatibility_facade()
    with _configured_predecessor_qualifier_v3():
        return predecessor._read_qualification_reservation(  # noqa: SLF001
            *args, **kwargs
        )


def _child_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    validate_qualifier_compatibility_facade()
    try:
        plan_role = str(plan["successor_contract"]["plan_role"])
        environment = plan["execution_contract"]["environment"]
    except (KeyError, TypeError) as exc:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 child environment contract is absent"
        ) from exc
    if not isinstance(environment, Mapping):
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 child environment contract is not an object"
        )
    expected = plan_builder.rocm_execution_environment(plan_role)
    if dict(environment) != expected:
        raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
            "V3 child environment differs from the exact role environment"
        )
    return dict(expected)


_all_numbers_finite = predecessor._all_numbers_finite  # noqa: SLF001


def validate_qualification_authority(*args: Any, **kwargs: Any):
    validate_qualifier_compatibility_facade()
    with _configured_predecessor_qualifier_v3():
        validated = predecessor.validate_qualification_authority(*args, **kwargs)
    collector._require_replacement_v1_review_binding(validated[0])  # noqa: SLF001
    return validated


def execute_qualification(*args: Any, **kwargs: Any):
    validate_qualifier_compatibility_facade()
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("qualification")
    with _configured_predecessor_qualifier_v3():
        return predecessor.execute_qualification(*args, **kwargs)


def build_parser():
    validate_qualifier_compatibility_facade()
    with _configured_predecessor_qualifier_v3():
        return predecessor.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    validate_qualifier_compatibility_facade()
    collector.require_exact_orchestrator_python()
    collector.require_exact_orchestrator_environment("qualification")
    raw = list(argv) if argv is not None else sys.argv[1:]
    if "--worker-scene-index" in raw:
        with (
            _configured_predecessor_qualifier_v3(),
            predecessor._configured_qualification_collector(),  # noqa: SLF001
        ):
            return collector.main(raw)
    args = build_parser().parse_args(raw)
    try:
        authority, authority_binding, plan, plan_binding = (
            validate_qualification_authority(
                args.authority,
                expected_sha256=args.expected_authority_sha256,
                expected_byte_count=args.expected_authority_byte_count,
            )
        )
        if (
            args.plan.resolve(strict=True)
            != Path(str(plan_binding["path"])).resolve(strict=True)
            or args.expected_plan_sha256 != plan_binding["sha256"]
            or args.expected_plan_byte_count != plan_binding["byte_count"]
        ):
            raise GenesisRocmBackendV3IntegrityReplacementV2QualificationError(
                "V3 qualification CLI plan pins differ from authority"
            )
        result = execute_qualification(
            authority,
            authority_binding=authority_binding,
            plan=plan,
            plan_binding=plan_binding,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(QUALIFICATION_RESULT_PATH),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        if plan_builder.QUALIFICATION_ATTEMPT_ROOT.is_dir():
            terminal = {
                "schema": QUALIFICATION_TERMINAL_SCHEMA,
                "status": (
                    "FAIL_GENESIS_ROCM_BACKEND_V3_INTEGRITY_REPLACEMENT_V2_QUALIFICATION_HARD_STOP"
                ),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "qualification_result_binding": None,
                "authorizes_scientific_authority": False,
                "authorizes_retry_or_resume": False,
            }
            terminal_path = (
                plan_builder.QUALIFICATION_ATTEMPT_ROOT / "terminal.json"
            )
            if not terminal_path.exists():
                pilot.write_json_exclusive(terminal_path, terminal)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GenesisRocmBackendQualificationError",
    "GenesisRocmBackendV3IntegrityReplacementV2QualificationError",
    "QUALIFICATION_AUTHORITY_FIELDS",
    "QUALIFICATION_AUTHORITY_SCHEMA",
    "QUALIFICATION_AUTHORITY_STATUS",
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_PROBE_ORDER",
    "QUALIFICATION_RESULT_FIELDS",
    "QUALIFICATION_RESULT_PATH",
    "QUALIFICATION_RESULT_SCHEMA",
    "QUALIFICATION_RESULT_STATUS",
    "SCIENTIFIC_STARTUP_ALLOWANCE_SECONDS",
    "WORKER_TIMEOUT_SECONDS",
    "execute_qualification",
    "validate_qualification_authority",
]
