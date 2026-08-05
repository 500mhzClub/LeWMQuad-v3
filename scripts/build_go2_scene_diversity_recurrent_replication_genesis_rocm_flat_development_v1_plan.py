#!/usr/bin/env python3
"""Build fresh plans for the flat development-only Genesis ROCm harness.

The layered adapter chain is a historical input only.  This builder reads and
validates its final science-identical plans without entering any adapter
context, proves the immutable textured-v03 prerequisite while the historical
Vulkan globals are untouched, and changes only flat identity/root/cache and
successor-study metadata.

This source emits metadata only and grants no execution authority.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v3_integrity_replacement_v2_plan as historical_builder  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded  # noqa: E402


PREDECESSOR_TERMINAL_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "backend_v3_integrity_replacement_v2_qualification_terminal_review_"
    "2026-08-05.json"
)
PREDECESSOR_TERMINAL_REVIEW_SHA256 = (
    "b699e2f632be089a41d3ee88d30b4164809335003cba8e93e0cde33e096fc000"
)
PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT = 15_335

HISTORICAL_SCIENTIFIC_PLAN = historical_builder.DEFAULT_PLAN_OUTPUT
HISTORICAL_SCIENTIFIC_PLAN_SHA256 = (
    "806e7740613a3ef77ae05549f8157a594f4c0d09a03ac82b3fdac524ac82d3f0"
)
HISTORICAL_SCIENTIFIC_PLAN_BYTE_COUNT = 366_820
HISTORICAL_QUALIFICATION_PLAN = historical_builder.QUALIFICATION_PLAN_OUTPUT
HISTORICAL_QUALIFICATION_PLAN_SHA256 = (
    "29e9c6936eb5ebafe1e438da42f0ed5f24e439f3317494a08aaf359f1a1891da"
)
HISTORICAL_QUALIFICATION_PLAN_BYTE_COUNT = 367_074

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-genesis-rocm-flat-development-v1"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_flat_development_v1/attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_exact_plan_2026-08-05.json"
)

QUALIFICATION_ATTEMPT_ID = f"{DEFAULT_ATTEMPT_ID}-qualification"
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_"
    "genesis_rocm_flat_development_v1_qualification/attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_qualification_exact_plan_2026-08-05.json"
)

ROCM_PYTHON = historical_builder.ROCM_PYTHON
ROCM_RUNTIME_PATHS = historical_builder.ROCM_RUNTIME_PATHS
ROCM_GRAPHICS_PREFLIGHT_EXPECTATION = copy.deepcopy(
    historical_builder.ROCM_GRAPHICS_PREFLIGHT_EXPECTATION
)
QUALIFICATION_SCENE_INDICES = tuple(
    historical_builder.QUALIFICATION_SCENE_INDICES
)
QUALIFICATION_WORKER_WATCHDOG_SECONDS = (
    historical_builder.QUALIFICATION_WORKER_WATCHDOG_SECONDS
)
QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS = (
    historical_builder.QUALIFICATION_FIXED_NONCOLLECTION_RESERVE_SECONDS
)
SUCCESSOR_CONTRACT_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
    "flat_development_v1_successor_contract_v1"
)


class FlatDevelopmentPlanError(RuntimeError):
    """Raised before a changed or non-fresh flat plan can be emitted."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _standard_binding(path: Path) -> dict[str, object]:
    raw = pilot.file_binding(path)
    return {
        "path": str(raw["path"]),
        "sha256": str(raw["file_sha256"]),
        "byte_count": int(raw["byte_count"]),
    }


def predecessor_terminal_review_binding() -> dict[str, object]:
    expected = {
        "path": str(PREDECESSOR_TERMINAL_REVIEW.resolve(strict=True)),
        "sha256": PREDECESSOR_TERMINAL_REVIEW_SHA256,
        "byte_count": PREDECESSOR_TERMINAL_REVIEW_BYTE_COUNT,
    }
    if _standard_binding(PREDECESSOR_TERMINAL_REVIEW) != expected:
        raise FlatDevelopmentPlanError("predecessor terminal review changed")
    try:
        review = json.loads(PREDECESSOR_TERMINAL_REVIEW.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FlatDevelopmentPlanError(
            "predecessor terminal review is not strict JSON"
        ) from exc
    permission = review.get("permission_audit", {})
    eligibility = review.get("successor_eligibility", {})
    stop = review.get("explicit_stop_rule", {})
    if (
        review.get("schema")
        != "lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_"
        "backend_v3_integrity_replacement_v2_qualification_terminal_review_v1"
        or review.get("status")
        != "PASS_FAIL_CLOSED_POST_RESERVATION_SHARED_ENVIRONMENT_TERMINAL_REVIEW"
        or review.get("audit_passed") is not True
        or permission.get("authorizes_source_only_flat_harness_study") is not True
        or permission.get("creates_qualification_execution_authority") is not False
        or permission.get("creates_scientific_execution_authority") is not False
        or eligibility.get(
            "only_separately_scoped_flat_non_monkeypatched_harness_source_study_eligible"
        )
        is not True
        or eligibility.get(
            "study_must_validate_historical_parity_before_any_reservation_gpu_or_genesis_call"
        )
        is not True
        or stop.get("layered_adapter_chain_retired_for_this_experiment_line")
        is not True
    ):
        raise FlatDevelopmentPlanError(
            "predecessor review does not permit this flat source study"
        )
    return expected


def _read_historical_plan(*, role: str) -> dict[str, Any]:
    if role == "scientific":
        path = HISTORICAL_SCIENTIFIC_PLAN
        sha256 = HISTORICAL_SCIENTIFIC_PLAN_SHA256
        byte_count = HISTORICAL_SCIENTIFIC_PLAN_BYTE_COUNT
        attempt_id = historical_builder.DEFAULT_ATTEMPT_ID
        output_root = historical_builder.DEFAULT_OUTPUT_ROOT
    elif role == "qualification":
        path = HISTORICAL_QUALIFICATION_PLAN
        sha256 = HISTORICAL_QUALIFICATION_PLAN_SHA256
        byte_count = HISTORICAL_QUALIFICATION_PLAN_BYTE_COUNT
        attempt_id = historical_builder.QUALIFICATION_ATTEMPT_ID
        output_root = historical_builder.QUALIFICATION_OUTPUT_ROOT
    else:
        raise FlatDevelopmentPlanError("flat plan role changed")
    observed = _standard_binding(path)
    expected = {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256,
        "byte_count": byte_count,
    }
    if observed != expected:
        raise FlatDevelopmentPlanError(f"historical {role} plan changed")
    value = json.loads(path.read_bytes())
    return historical_builder.validate_rocm_plan(
        value,
        expected_attempt_id=attempt_id,
        expected_output_root=output_root,
        plan_role=role,
    )


def validate_historical_textured_parity(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every bound input and Vulkan parity without shared mutation."""

    environment_object = pilot.EXECUTION_ENVIRONMENT
    graphics_object = pilot.GRAPHICS_PREFLIGHT_EXPECTATION
    environment_before = copy.deepcopy(environment_object)
    graphics_before = copy.deepcopy(graphics_object)
    if environment_before.get("GS_BACKEND") != "vulkan":
        raise FlatDevelopmentPlanError(
            "historical parity environment is not immutable Vulkan"
        )
    pilot.require_plan_bindings(plan)
    observed = bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    if (
        pilot.EXECUTION_ENVIRONMENT is not environment_object
        or pilot.GRAPHICS_PREFLIGHT_EXPECTATION is not graphics_object
        or pilot.EXECUTION_ENVIRONMENT != environment_before
        or pilot.GRAPHICS_PREFLIGHT_EXPECTATION != graphics_before
    ):
        raise FlatDevelopmentPlanError(
            "historical parity validation mutated shared pilot state"
        )
    return observed


def rocm_execution_environment(role: str) -> dict[str, str]:
    historical = historical_builder.rocm_execution_environment(role)
    attempt = (
        DEFAULT_ATTEMPT_ROOT
        if role == "scientific"
        else QUALIFICATION_ATTEMPT_ROOT
        if role == "qualification"
        else None
    )
    if attempt is None:
        raise FlatDevelopmentPlanError("flat plan role changed")
    result = dict(historical)
    result["GS_CACHE_FILE_PATH"] = str(
        (attempt / "quadrants_cache").resolve(strict=False)
    )
    return result


def _expected_plan(*, role: str) -> dict[str, Any]:
    candidate = copy.deepcopy(_read_historical_plan(role=role))
    if role == "scientific":
        attempt_id = DEFAULT_ATTEMPT_ID
        output_root = DEFAULT_OUTPUT_ROOT
    else:
        attempt_id = QUALIFICATION_ATTEMPT_ID
        output_root = QUALIFICATION_OUTPUT_ROOT
    candidate["attempt_id"] = attempt_id
    candidate["output_root"] = str(output_root.resolve(strict=False))
    candidate["execution_contract"]["environment"] = (
        rocm_execution_environment(role)
    )
    contract = copy.deepcopy(candidate["successor_contract"])
    contract.update(
        {
            "schema": SUCCESSOR_CONTRACT_SCHEMA,
            "plan_role": role,
            "flat_harness_owned": True,
            "module_global_overlay_contexts_authorized": False,
            "historical_textured_parity_pre_reservation_required": True,
            "historical_vulkan_contract_mutation_authorized": False,
            "rocm_contract_passed_explicitly": True,
            "qualification_scene_indices_in_order": list(
                QUALIFICATION_SCENE_INDICES
            ),
            "three_arm_science_handoff_unchanged": True,
            "development_only": True,
            "promotable": False,
            "predecessor_terminal_review_binding": (
                predecessor_terminal_review_binding()
            ),
            "predecessor_authority_reservation_or_runtime_reuse_authorized": False,
        }
    )
    candidate["successor_contract"] = contract
    return candidate


def validate_flat_plan(
    plan: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise FlatDevelopmentPlanError("flat plan must be an object")
    candidate = copy.deepcopy(dict(plan))
    expected = _expected_plan(role=role)
    if _canonical(candidate) != _canonical(expected):
        raise FlatDevelopmentPlanError(
            "flat plan changed beyond identity/root/cache/study metadata"
        )
    validate_historical_textured_parity(candidate)
    return candidate


def build_plan(*, role: str) -> dict[str, Any]:
    output_root = (
        DEFAULT_OUTPUT_ROOT
        if role == "scientific"
        else QUALIFICATION_OUTPUT_ROOT
        if role == "qualification"
        else None
    )
    attempt_root = output_root.parent if output_root is not None else None
    if (
        output_root is None
        or attempt_root is None
        or attempt_root.exists()
        or attempt_root.is_symlink()
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise FlatDevelopmentPlanError(f"fresh flat {role} root changed")
    return validate_flat_plan(_expected_plan(role=role), role=role)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument(
        "--qualification-plan-output",
        type=Path,
        default=QUALIFICATION_PLAN_OUTPUT,
    )
    parser.add_argument(
        "--qualification-only",
        action="store_true",
        help=(
            "emit only the pre-runtime qualification plan; science-plan "
            "release is deferred until an exact qualification PASS decision exists"
        ),
    )
    args = parser.parse_args(argv)
    if args.qualification_only:
        if (
            args.qualification_plan_output.exists()
            or args.qualification_plan_output.is_symlink()
        ):
            raise FlatDevelopmentPlanError(
                "flat qualification plan output must be fresh"
            )
        qualification = build_plan(role="qualification")
        qualification_binding = pilot.write_json_exclusive(
            args.qualification_plan_output, qualification
        )
        print(json.dumps({
            "qualification_plan": qualification_binding,
            "scientific_plan": None,
            "scientific_plan_release_deferred_until_qualification_pass": True,
            "predecessor_terminal_review": predecessor_terminal_review_binding(),
        }, sort_keys=True))
        return 0
    if any(path.exists() or path.is_symlink() for path in (
        args.plan_output,
        args.qualification_plan_output,
    )):
        raise FlatDevelopmentPlanError("flat plan outputs must be fresh")
    science = build_plan(role="scientific")
    qualification = build_plan(role="qualification")
    science_binding = pilot.write_json_exclusive(args.plan_output, science)
    qualification_binding = pilot.write_json_exclusive(
        args.qualification_plan_output, qualification
    )
    print(json.dumps({
        "scientific_plan": science_binding,
        "qualification_plan": qualification_binding,
        "predecessor_terminal_review": predecessor_terminal_review_binding(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PLAN_OUTPUT",
    "FlatDevelopmentPlanError",
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "QUALIFICATION_SCENE_INDICES",
    "ROCM_GRAPHICS_PREFLIGHT_EXPECTATION",
    "ROCM_PYTHON",
    "build_plan",
    "predecessor_terminal_review_binding",
    "rocm_execution_environment",
    "validate_flat_plan",
    "validate_historical_textured_parity",
]
