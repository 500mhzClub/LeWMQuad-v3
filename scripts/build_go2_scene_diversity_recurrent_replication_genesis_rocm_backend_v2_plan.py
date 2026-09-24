#!/usr/bin/env python3
"""Build the fresh V2 ``ld.lld``-driver Genesis ROCm plans.

V2 is a material infrastructure successor, not a retry, resume, refill, or
science-identical replacement of consumed V1.  It changes the fresh attempt
identities and makes the Unix LLD driver entrypoint an explicit plan field.
All scene, data, model, evaluation, cap, and decision-gate content is rebuilt
from the same frozen Vulkan scientific plan used by V1.

This source emits metadata only and grants no execution authority.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v1_plan as predecessor  # noqa: E402


pilot = predecessor.pilot

V1_QUALIFICATION_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v1_qualification_terminal_review_2026-08-04.json"
)
V1_QUALIFICATION_TERMINAL_REVIEW_SHA256 = (
    "3e35cdb459c18d862e21df676b0a630a0496d1a26f8a97874095c71ab2facb5b"
)
V1_QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT = 14_742

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v2"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v2_exact_plan_2026-08-04.json"
)

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-"
    "genesis-rocm-backend-v2-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v2_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v2_qualification_exact_plan_2026-08-04.json"
)

ROCM_VENV = predecessor.ROCM_VENV
ROCM_PYTHON = predecessor.ROCM_PYTHON
ROCM_SITE_PACKAGES = predecessor.ROCM_SITE_PACKAGES
WORLD_MODEL_ROCM_SITE_PACKAGES = predecessor.WORLD_MODEL_ROCM_SITE_PACKAGES
ROCM_PREFIX = predecessor.ROCM_PREFIX
ROCM_LLVM_BIN = predecessor.ROCM_LLVM_BIN
ROCM_EXECUTION_PATH = predecessor.ROCM_EXECUTION_PATH
ROCM_RUNTIME_PATHS = predecessor.ROCM_RUNTIME_PATHS
ROCM_EXECUTION_ENVIRONMENT_COMMON = copy.deepcopy(
    predecessor.ROCM_EXECUTION_ENVIRONMENT_COMMON
)

ROCM_LD_LLD_DRIVER_ENTRYPOINT = ROCM_LLVM_BIN / "ld.lld"
ROCM_LD_LLD_DRIVER_LINK_TEXT = "lld"
ROCM_LLD_VERSION_STDOUT_PREFIX = "AMD LLD 20.0.0"

ROCM_GRAPHICS_PREFLIGHT_EXPECTATION = {
    **copy.deepcopy(predecessor.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION),
    "rocm_lld_driver_entrypoint": str(ROCM_LD_LLD_DRIVER_ENTRYPOINT),
    "rocm_lld_driver_link_text": ROCM_LD_LLD_DRIVER_LINK_TEXT,
    "rocm_lld_resolved_target_path": str(
        predecessor.ROCM_RUNTIME_PATHS["rocm_lld_executable"].resolve(
            strict=True
        )
    ),
    "rocm_lld_version_stdout_prefix": ROCM_LLD_VERSION_STDOUT_PREFIX,
    "rocm_lld_direct_target_invocation_forbidden": True,
}

QUALIFICATION_SCENE_INDICES = predecessor.QUALIFICATION_SCENE_INDICES
QUALIFICATION_WORKER_WATCHDOG_SECONDS = (
    predecessor.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = (
    predecessor.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)
SCIENTIFIC_SCENE_COUNT = predecessor.SCIENTIFIC_SCENE_COUNT
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v2_ld_lld_driver_successor_contract_v1"
)

# Re-export the older, still-required source premises for downstream wrappers.
CPU_TERMINAL_REVIEW = predecessor.CPU_TERMINAL_REVIEW
CPU_TERMINAL_REVIEW_SHA256 = predecessor.CPU_TERMINAL_REVIEW_SHA256
CPU_TERMINAL_REVIEW_BYTE_COUNT = predecessor.CPU_TERMINAL_REVIEW_BYTE_COUNT
CPU_TERMINAL_REVIEW_BINDING = predecessor.CPU_TERMINAL_REVIEW_BINDING


class SceneDiversityGenesisRocmV2PlanError(RuntimeError):
    """Raised before a changed or non-fresh V2 plan can be emitted."""


# Compatibility name used by the inherited collector/runner exception paths.
SceneDiversityGenesisRocmPlanError = SceneDiversityGenesisRocmV2PlanError


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _exact_review_binding() -> dict[str, Any]:
    expected = {
        "path": str(V1_QUALIFICATION_TERMINAL_REVIEW.resolve(strict=True)),
        "file_sha256": V1_QUALIFICATION_TERMINAL_REVIEW_SHA256,
        "byte_count": V1_QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT,
    }
    actual = pilot.file_binding(V1_QUALIFICATION_TERMINAL_REVIEW)
    if actual != expected:
        raise SceneDiversityGenesisRocmV2PlanError(
            "V1 qualification terminal review binding changed"
        )
    try:
        review = json.loads(V1_QUALIFICATION_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityGenesisRocmV2PlanError(
            "V1 qualification terminal review is not strict JSON"
        ) from exc
    successor = review.get("successor_eligibility", {})
    permission = review.get("permission_audit", {})
    decision = review.get("decision", {})
    if (
        review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v1_qualification_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_PRE_BACKEND_INFRASTRUCTURE_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or decision.get("attempt_consumed") is not True
        or decision.get("failure_is_source_control_flow_infrastructure")
        is not True
        or decision.get("failure_is_backend_evidence") is not False
        or successor.get(
            "separately_preregistered_fresh_v2_ld_lld_driver_entrypoint_"
            "hypothesis_eligible"
        )
        is not True
        or successor.get("v1_payload_reuse_authorized") is not False
        or successor.get("v1_runtime_metadata_as_identity_evidence_authorized")
        is not False
        or successor.get(
            "v1_terminal_review_document_as_source_evidence_authorized"
        )
        is not True
        or permission.get(
            "authorizes_v2_source_development_and_independent_review"
        )
        is not True
        or permission.get("authorizes_v2_qualification_execution") is not False
        or permission.get("authorizes_v2_scientific_execution") is not False
        or permission.get("authorizes_payload_reuse") is not False
    ):
        raise SceneDiversityGenesisRocmV2PlanError(
            "V1 terminal review does not permit this source-only V2"
        )
    return expected


V1_QUALIFICATION_TERMINAL_REVIEW_BINDING = _exact_review_binding()


def rocm_execution_environment(plan_role: str) -> dict[str, str]:
    if plan_role == "scientific":
        attempt_root = DEFAULT_ATTEMPT_ROOT
    elif plan_role == "qualification":
        attempt_root = QUALIFICATION_ATTEMPT_ROOT
    else:
        raise SceneDiversityGenesisRocmV2PlanError("V2 plan role changed")
    return {
        **ROCM_EXECUTION_ENVIRONMENT_COMMON,
        "GS_CACHE_FILE_PATH": str(
            (attempt_root / "quadrants_cache").resolve(strict=False)
        ),
    }


def _validate_driver_entrypoint(
    runtime_bindings: Mapping[str, Any], *, plan_role: str
) -> None:
    environment = rocm_execution_environment(plan_role)
    driver = ROCM_LD_LLD_DRIVER_ENTRYPOINT
    target = Path(str(runtime_bindings["rocm_lld_executable"]["path"]))
    path_driver = shutil.which("ld.lld", path=environment["PATH"])
    rocm_path_driver = (
        Path(environment["ROCM_PATH"]) / "lib/llvm/bin/ld.lld"
    )
    try:
        link_text = os.readlink(driver)
        resolved = driver.resolve(strict=True)
    except OSError as exc:
        raise SceneDiversityGenesisRocmV2PlanError(
            "exact ld.lld driver entrypoint is unavailable"
        ) from exc
    if (
        path_driver != str(driver)
        or rocm_path_driver != driver
        or not driver.is_symlink()
        or link_text != ROCM_LD_LLD_DRIVER_LINK_TEXT
        or resolved != target
        or target.is_symlink()
        or not target.is_file()
        or str(target) != ROCM_GRAPHICS_PREFLIGHT_EXPECTATION[
            "rocm_lld_resolved_target_path"
        ]
    ):
        raise SceneDiversityGenesisRocmV2PlanError(
            "ld.lld driver/target identity changed"
        )


def build_rocm_runtime_bindings() -> dict[str, dict[str, Any]]:
    bindings = predecessor.build_rocm_runtime_bindings()
    _validate_driver_entrypoint(bindings, plan_role="scientific")
    return bindings


def _validate_rocm_runtime_bindings(
    runtime_bindings: Mapping[str, Any], *, rehash: bool
) -> dict[str, dict[str, Any]]:
    try:
        validated = predecessor._validate_rocm_runtime_bindings(  # noqa: SLF001
            runtime_bindings, rehash=rehash
        )
    except predecessor.SceneDiversityGenesisRocmPlanError as exc:
        raise SceneDiversityGenesisRocmV2PlanError(str(exc)) from exc
    return validated


def _successor_contract(*, plan_role: str) -> dict[str, Any]:
    qualification = plan_role == "qualification"
    if plan_role not in {"scientific", "qualification"}:
        raise SceneDiversityGenesisRocmV2PlanError("V2 plan role changed")
    return {
        "schema": SUCCESSOR_CONTRACT_SCHEMA,
        "plan_role": plan_role,
        "frozen_vulkan_scientific_plan_binding": copy.deepcopy(
            predecessor.FROZEN_V1_EXACT_PLAN_BINDING
        ),
        "frozen_cpu_scientific_plan_binding": copy.deepcopy(
            predecessor.FROZEN_CPU_EXACT_PLAN_BINDING
        ),
        "cpu_qualification_terminal_review_binding": copy.deepcopy(
            predecessor.CPU_TERMINAL_REVIEW_BINDING
        ),
        "v1_qualification_terminal_review_binding": copy.deepcopy(
            V1_QUALIFICATION_TERMINAL_REVIEW_BINDING
        ),
        "material_infrastructure_hypothesis": (
            "invoke the exact unresolved ld.lld Unix-driver entrypoint while "
            "separately binding its resolved regular lld target"
        ),
        "rocm_lld_driver_entrypoint": str(ROCM_LD_LLD_DRIVER_ENTRYPOINT),
        "rocm_lld_direct_target_invocation_forbidden": True,
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
                "formula": "64 * max(worker_elapsed_seconds) + 900 <= 7200",
                "scientific_wall_cap_seconds": 7_200,
            }
            if qualification
            else None
        ),
        "v1_runtime_payload_reuse_authorized": False,
        "v1_runtime_metadata_as_identity_evidence_authorized": False,
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
    candidate = copy.deepcopy(predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN)  # noqa: SLF001
    candidate["attempt_id"] = attempt_id
    candidate["output_root"] = str(output_root.resolve(strict=False))
    candidate["runtime_bindings"] = copy.deepcopy(dict(runtime_bindings))
    execution = candidate["execution_contract"]
    execution["backend"] = "amdgpu"
    # Preserve the lexical venv launcher; resolving it would bypass pyvenv.cfg.
    execution["python_invocation_path"] = str(ROCM_PYTHON.absolute())
    execution["environment"] = rocm_execution_environment(plan_role)
    execution["graphics_preflight"] = copy.deepcopy(
        ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
    )
    candidate["successor_contract"] = _successor_contract(
        plan_role=plan_role
    )
    return candidate


def validate_rocm_plan(
    plan: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    expected_output_root: Path,
    plan_role: str,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise SceneDiversityGenesisRocmV2PlanError(
            "V2 ROCm plan must be an object"
        )
    candidate = copy.deepcopy(dict(plan))
    runtime = _validate_rocm_runtime_bindings(
        candidate.get("runtime_bindings", {}), rehash=True
    )
    _validate_driver_entrypoint(runtime, plan_role=plan_role)
    expected = _expected_rocm_plan(
        attempt_id=expected_attempt_id,
        output_root=expected_output_root,
        plan_role=plan_role,
        runtime_bindings=runtime,
    )
    if _canonical_bytes(candidate) != _canonical_bytes(expected):
        raise SceneDiversityGenesisRocmV2PlanError(
            "V2 ROCm plan changed beyond the exact driver successor overlay"
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
    ):
        raise SceneDiversityGenesisRocmV2PlanError(
            f"{label} output_root must be its exact fresh V2 path"
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
        raise SceneDiversityGenesisRocmV2PlanError(
            "V2 ROCm attempt identifier changed"
        )
    try:
        predecessor._require_exact_frozen_input(frozen_plan)  # noqa: SLF001
    except predecessor.SceneDiversityGenesisRocmPlanError as exc:
        raise SceneDiversityGenesisRocmV2PlanError(str(exc)) from exc
    selected_root = _require_fresh_exact_root(
        output_root=output_root,
        expected_root=expected_output_root,
        attempt_root=attempt_root,
        label=plan_role,
    )
    runtime = (
        build_rocm_runtime_bindings()
        if runtime_bindings is None
        else _validate_rocm_runtime_bindings(runtime_bindings, rehash=True)
    )
    _validate_driver_entrypoint(runtime, plan_role=plan_role)
    return validate_rocm_plan(
        _expected_rocm_plan(
            attempt_id=attempt_id,
            output_root=selected_root,
            plan_role=plan_role,
            runtime_bindings=runtime,
        ),
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
        raise SceneDiversityGenesisRocmV2PlanError(
            "V2 plan outputs must be fresh"
        )
    runtime = build_rocm_runtime_bindings()
    frozen = copy.deepcopy(predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN)  # noqa: SLF001
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
                "v1_qualification_terminal_review": (
                    V1_QUALIFICATION_TERMINAL_REVIEW_BINDING
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
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "QUALIFICATION_SCENE_INDICES",
    "ROCM_EXECUTION_ENVIRONMENT_COMMON",
    "ROCM_GRAPHICS_PREFLIGHT_EXPECTATION",
    "ROCM_LD_LLD_DRIVER_ENTRYPOINT",
    "ROCM_LD_LLD_DRIVER_LINK_TEXT",
    "ROCM_LLD_VERSION_STDOUT_PREFIX",
    "ROCM_PYTHON",
    "ROCM_RUNTIME_PATHS",
    "SceneDiversityGenesisRocmPlanError",
    "SceneDiversityGenesisRocmV2PlanError",
    "V1_QUALIFICATION_TERMINAL_REVIEW",
    "V1_QUALIFICATION_TERMINAL_REVIEW_BINDING",
    "build_qualification_plan",
    "build_rocm_runtime_bindings",
    "build_scientific_plan",
    "rocm_execution_environment",
    "validate_rocm_plan",
]
