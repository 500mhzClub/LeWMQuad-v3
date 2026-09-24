#!/usr/bin/env python3
"""Build the science-identical V3 integrity-replacement V2 plans.

Consumed replacement V1 failed before reservation because its substituted
collector did not implement the complete inherited layered interface.  This
replacement changes only fresh identity, roots, role-local caches, witnesses,
and a closed compatibility facade.  Science and runtime remain exact.

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

from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v1_plan as predecessor  # noqa: E402


pilot = predecessor.pilot

REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v1_"
    "qualification_terminal_review_2026-08-04.json"
)
REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_SHA256 = (
    "45bcd329d778dc9cba71882b398aa25943600dc160436bdba6b919868418f6fa"
)
REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT = 10_267
REQUIRED_HOST_HOME = "/home/andrewknowles"

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v3-integrity-replacement-v2"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_exact_plan_2026-08-04.json"
)

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-"
    "genesis-rocm-backend-v3-integrity-replacement-v2-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_backend_v3_integrity_replacement_v2_qualification_exact_plan_2026-08-04.json"
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
    "genesis_rocm_backend_v3_integrity_replacement_v2_"
    "science_identical_closed_interface_successor_contract_v1"
)

# Re-export the older, still-required source premises for downstream wrappers.
CPU_TERMINAL_REVIEW = predecessor.CPU_TERMINAL_REVIEW
CPU_TERMINAL_REVIEW_SHA256 = predecessor.CPU_TERMINAL_REVIEW_SHA256
CPU_TERMINAL_REVIEW_BYTE_COUNT = predecessor.CPU_TERMINAL_REVIEW_BYTE_COUNT
CPU_TERMINAL_REVIEW_BINDING = predecessor.CPU_TERMINAL_REVIEW_BINDING


class SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(RuntimeError):
    """Raised before a changed or non-fresh replacement plan can be emitted."""


# Compatibility name used by the inherited collector/runner exception paths.
SceneDiversityGenesisRocmPlanError = SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _exact_review_binding() -> dict[str, Any]:
    expected = {
        "path": str(
            REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW.resolve(strict=True)
        ),
        "file_sha256": REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_SHA256,
        "byte_count": REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT,
    }
    actual = pilot.file_binding(REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW)
    if actual != expected:
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "replacement V1 qualification terminal review binding changed"
        )
    try:
        review = json.loads(
            REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW.read_bytes()
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "replacement V1 terminal review is not strict JSON"
        ) from exc
    successor = review.get("successor_eligibility", {})
    permission = review.get("permission_audit", {})
    decision = review.get("decision", {})
    interface = review.get("layered_interface_audit", {})
    stop = review.get("explicit_stop_rule", {})
    scope = review.get("review_scope", {})
    if (
        review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_"
        "genesis_rocm_backend_v3_integrity_replacement_v1_"
        "qualification_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_PRE_RESERVATION_LAYERED_INTERFACE_"
        "TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or decision.get("qualification_invocation_consumed") is not True
        or decision.get("qualification_authority_consumed") is not True
        or decision.get("qualification_pass") is not False
        or decision.get("retry_or_resume_authorized") is not False
        or interface.get("classification")
        != "repeated_pre_reservation_layered_adapter_interface_failure"
        or interface.get("immediate_missing_attribute")
        != "_require_v1_review_binding"
        or interface.get("replacement_collector_exports_required_v2_helper")
        is not False
        or interface.get("one_alias_patch_is_sufficiently_comprehensive")
        is not False
        or scope.get("replacement_runtime_payload_opened") is not False
        or scope.get("replacement_reservation_payload_opened") is not False
        or scope.get("runtime_root_written_by_review") is not False
        or successor.get(
            "at_most_one_fresh_comprehensive_integrity_replacement_v2_"
            "source_hypothesis_eligible"
        )
        is not True
        or successor.get(
            "science_data_model_panel_seed_environment_backend_cap_"
            "evaluation_and_gate_changes_authorized"
        )
        is not False
        or successor.get(
            "replacement_v1_authority_command_runtime_or_payload_reuse_"
            "authorized"
        )
        is not False
        or stop.get("integrity_replacement_v3_authorized_after_trigger")
        is not False
        or permission.get(
            "authorizes_source_development_and_independent_review_of_at_most_"
            "one_comprehensive_integrity_replacement_v2"
        )
        is not True
        or permission.get("creates_qualification_execution_authority")
        is not False
        or permission.get("creates_scientific_execution_authority") is not False
    ):
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "replacement V1 review does not permit source-only V2"
        )
    return expected


def replacement_v1_qualification_terminal_review_binding() -> dict[str, Any]:
    return _exact_review_binding()


def rocm_execution_environment(plan_role: str) -> dict[str, str]:
    if plan_role == "scientific":
        attempt_root = DEFAULT_ATTEMPT_ROOT
    elif plan_role == "qualification":
        attempt_root = QUALIFICATION_ATTEMPT_ROOT
    else:
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError("V3 plan role changed")
    return {
        **ROCM_EXECUTION_ENVIRONMENT_COMMON,
        "HOME": REQUIRED_HOST_HOME,
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(str(exc)) from exc
    return validated


def _successor_contract(*, plan_role: str) -> dict[str, Any]:
    if plan_role not in {"scientific", "qualification"}:
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "replacement plan role changed"
        )
    contract = copy.deepcopy(
        predecessor._successor_contract(plan_role=plan_role)  # noqa: SLF001
    )
    contract.update(
        {
            "schema": SUCCESSOR_CONTRACT_SCHEMA,
            "replacement_v1_qualification_terminal_review_binding": copy.deepcopy(
                replacement_v1_qualification_terminal_review_binding()
            ),
            "material_infrastructure_hypothesis": (
                "provide and directly validate a closed compatibility facade "
                "for every inherited V1, V2, V3, and replacement V1 "
                "collector, qualifier, and runner interface"
            ),
            "replacement_v1_authority_or_command_reuse_authorized": False,
            "replacement_v1_runtime_payload_reuse_authorized": False,
            "replacement_v1_runtime_metadata_as_identity_evidence_authorized": False,
        }
    )
    return contract


def _expected_rocm_plan(
    *,
    attempt_id: str,
    output_root: Path,
    plan_role: str,
    runtime_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = predecessor._expected_rocm_plan(  # noqa: SLF001
        attempt_id=predecessor.DEFAULT_ATTEMPT_ID,
        output_root=predecessor.DEFAULT_OUTPUT_ROOT,
        plan_role=plan_role,
        runtime_bindings=runtime_bindings,
    )
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "V3 ROCm plan must be an object"
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "V3 ROCm plan changed beyond the exact driver successor overlay"
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            f"{label} output_root must be its exact fresh V3 path"
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "V3 ROCm attempt identifier changed"
        )
    try:
        predecessor.predecessor.predecessor.predecessor._require_exact_frozen_input(  # noqa: SLF001
            frozen_plan
        )
    except (
        predecessor.predecessor.predecessor.predecessor.SceneDiversityGenesisRocmPlanError
    ) as exc:
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(str(exc)) from exc
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
        raise SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError(
            "V3 plan outputs must be fresh"
        )
    runtime = build_rocm_runtime_bindings()
    frozen = copy.deepcopy(
        predecessor.predecessor.predecessor.predecessor._IMMUTABLE_FROZEN_VULKAN_PLAN  # noqa: SLF001
    )
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
                "replacement_v1_qualification_terminal_review": (
                    replacement_v1_qualification_terminal_review_binding()
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
    "SceneDiversityGenesisRocmV3IntegrityReplacementV2PlanError",
    "REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW",
    "REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_BYTE_COUNT",
    "REPLACEMENT_V1_QUALIFICATION_TERMINAL_REVIEW_SHA256",
    "build_qualification_plan",
    "build_rocm_runtime_bindings",
    "build_scientific_plan",
    "rocm_execution_environment",
    "validate_rocm_plan",
    "replacement_v1_qualification_terminal_review_binding",
]
