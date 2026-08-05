#!/usr/bin/env python3
"""Build the qualified CPU-backend successor and probe plans.

The frozen scene-diversity V1 plan remains the sole scientific input.  The
scientific CPU plan has exactly four allowed differences: attempt identity,
output root, ``execution_contract.backend=cpu`` and
``execution_contract.environment.GS_BACKEND=cpu``.  The separate qualification
plan carries the same material backend delta under a non-scientific root.
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
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan as frozen_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as frozen_collector  # noqa: E402


FROZEN_V1_EXACT_PLAN = frozen_builder.FROZEN_V1_EXACT_PLAN
FROZEN_V1_EXACT_PLAN_SHA256 = frozen_builder.FROZEN_V1_EXACT_PLAN_SHA256
FROZEN_V1_EXACT_PLAN_BYTE_COUNT = frozen_builder.FROZEN_V1_EXACT_PLAN_BYTE_COUNT
FROZEN_SCENE_PANEL = frozen_builder.FROZEN_SCENE_PANEL
FROZEN_SCENE_PANEL_SHA256 = frozen_builder.FROZEN_SCENE_PANEL_SHA256
FROZEN_SCENE_PANEL_BYTE_COUNT = frozen_builder.FROZEN_SCENE_PANEL_BYTE_COUNT

DEFAULT_ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-cpu-backend-v1"
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_cpu_backend_v1/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_exact_plan_2026-08-04.json"
)

QUALIFICATION_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-cpu-backend-v1-qualification"
)
QUALIFICATION_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_cpu_backend_v1_qualification/"
    "attempt_v1"
)
QUALIFICATION_OUTPUT_ROOT = QUALIFICATION_ATTEMPT_ROOT / "collection"
QUALIFICATION_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "cpu_backend_v1_qualification_exact_plan_2026-08-04.json"
)

CPU_EXECUTION_ENVIRONMENT = {
    **pilot.EXECUTION_ENVIRONMENT,
    "GS_BACKEND": "cpu",
}
_ORIGINAL_PILOT_VALIDATE_PLAN = pilot.validate_plan


class SceneDiversityCpuBackendPlanError(RuntimeError):
    """Raised before a changed or non-fresh CPU plan can be emitted."""


def _load_immutable_frozen_plan() -> dict[str, Any]:
    """Validate the exact Vulkan witness once, before runtime overlays exist."""

    try:
        raw = json.loads(FROZEN_V1_EXACT_PLAN.read_bytes())
        validated = frozen_builder._validated_frozen_plan_v1(raw)  # noqa: SLF001
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        frozen_builder.SceneDiversityReplacementPlanError,
    ) as exc:
        raise SceneDiversityCpuBackendPlanError(str(exc)) from exc
    return copy.deepcopy(validated)


_IMMUTABLE_FROZEN_PLAN = _load_immutable_frozen_plan()


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
        raise SceneDiversityCpuBackendPlanError(
            f"{label} output_root must be its exact fresh development path"
        )
    return resolved


def _validated_frozen_plan() -> dict[str, Any]:
    try:
        raw = json.loads(FROZEN_V1_EXACT_PLAN.read_bytes())
        binding = pilot.file_binding(FROZEN_V1_EXACT_PLAN)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise SceneDiversityCpuBackendPlanError(str(exc)) from exc
    if (
        binding
        != {
            "path": str(FROZEN_V1_EXACT_PLAN.resolve()),
            "file_sha256": FROZEN_V1_EXACT_PLAN_SHA256,
            "byte_count": FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
        }
        or raw != _IMMUTABLE_FROZEN_PLAN
    ):
        raise SceneDiversityCpuBackendPlanError(
            "immutable frozen V1 plan witness changed"
        )
    return copy.deepcopy(_IMMUTABLE_FROZEN_PLAN)


def validate_cpu_plan(
    plan: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    expected_output_root: Path,
) -> dict[str, Any]:
    """Validate exactly the two identity and two backend-field changes."""

    candidate = copy.deepcopy(dict(plan))
    execution = candidate.get("execution_contract")
    if (
        candidate.get("attempt_id") != expected_attempt_id
        or candidate.get("output_root")
        != str(expected_output_root.resolve(strict=False))
        or not isinstance(execution, dict)
        or execution.get("backend") != "cpu"
        or execution.get("environment") != CPU_EXECUTION_ENVIRONMENT
    ):
        raise SceneDiversityCpuBackendPlanError(
            "CPU plan identity or exact backend delta changed"
        )
    normalized = copy.deepcopy(candidate)
    normalized["execution_contract"]["backend"] = "vulkan"
    normalized["execution_contract"]["environment"] = copy.deepcopy(
        pilot.EXECUTION_ENVIRONMENT
    )
    try:
        validated_normalized = _ORIGINAL_PILOT_VALIDATE_PLAN(normalized)
        frozen_collector._validate_scene_diversity_plan_v1(  # noqa: SLF001
            validated_normalized
        )
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityCpuBackendPlanError(str(exc)) from exc
    if validated_normalized != normalized:
        raise SceneDiversityCpuBackendPlanError("CPU plan normalization changed")

    frozen = _validated_frozen_plan()
    expected = copy.deepcopy(frozen)
    expected["attempt_id"] = expected_attempt_id
    expected["output_root"] = str(expected_output_root.resolve(strict=False))
    expected["execution_contract"]["backend"] = "cpu"
    expected["execution_contract"]["environment"] = copy.deepcopy(
        CPU_EXECUTION_ENVIRONMENT
    )
    if _canonical_bytes(candidate) != _canonical_bytes(expected):
        raise SceneDiversityCpuBackendPlanError(
            "CPU plan changed beyond identity and exact backend fields"
        )
    return candidate


def build_cpu_plan(
    *,
    frozen_plan: Mapping[str, Any],
    attempt_id: str,
    output_root: Path,
    expected_attempt_id: str,
    expected_output_root: Path,
    attempt_root: Path,
    label: str,
) -> dict[str, Any]:
    if attempt_id != expected_attempt_id:
        raise SceneDiversityCpuBackendPlanError(f"{label} attempt identifier changed")
    try:
        original = frozen_builder._validated_frozen_plan_v1(  # noqa: SLF001
            frozen_plan
        )
    except frozen_builder.SceneDiversityReplacementPlanError as exc:
        raise SceneDiversityCpuBackendPlanError(str(exc)) from exc
    selected_root = _require_fresh_exact_root(
        output_root=output_root,
        expected_root=expected_output_root,
        attempt_root=attempt_root,
        label=label,
    )
    candidate = copy.deepcopy(original)
    candidate["attempt_id"] = attempt_id
    candidate["output_root"] = str(selected_root)
    candidate["execution_contract"]["backend"] = "cpu"
    candidate["execution_contract"]["environment"] = copy.deepcopy(
        CPU_EXECUTION_ENVIRONMENT
    )
    return validate_cpu_plan(
        candidate,
        expected_attempt_id=expected_attempt_id,
        expected_output_root=expected_output_root,
    )


def build_scientific_plan(
    *, frozen_plan: Mapping[str, Any]
) -> dict[str, Any]:
    return build_cpu_plan(
        frozen_plan=frozen_plan,
        attempt_id=DEFAULT_ATTEMPT_ID,
        output_root=DEFAULT_OUTPUT_ROOT,
        expected_attempt_id=DEFAULT_ATTEMPT_ID,
        expected_output_root=DEFAULT_OUTPUT_ROOT,
        attempt_root=DEFAULT_ATTEMPT_ROOT,
        label="scientific CPU",
    )


def build_qualification_plan(
    *, frozen_plan: Mapping[str, Any]
) -> dict[str, Any]:
    return build_cpu_plan(
        frozen_plan=frozen_plan,
        attempt_id=QUALIFICATION_ATTEMPT_ID,
        output_root=QUALIFICATION_OUTPUT_ROOT,
        expected_attempt_id=QUALIFICATION_ATTEMPT_ID,
        expected_output_root=QUALIFICATION_OUTPUT_ROOT,
        attempt_root=QUALIFICATION_ATTEMPT_ROOT,
        label="CPU qualification",
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
        raise SceneDiversityCpuBackendPlanError("CPU plan outputs must be fresh")
    frozen = _validated_frozen_plan()
    science = build_scientific_plan(frozen_plan=frozen)
    qualification = build_qualification_plan(frozen_plan=frozen)
    science_binding = pilot.write_json_exclusive(args.plan_output, science)
    qualification_binding = pilot.write_json_exclusive(
        args.qualification_plan_output, qualification
    )
    print(
        json.dumps(
            {
                "scientific_plan": science_binding,
                "qualification_plan": qualification_binding,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_EXECUTION_ENVIRONMENT",
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PLAN_OUTPUT",
    "FROZEN_V1_EXACT_PLAN",
    "QUALIFICATION_ATTEMPT_ID",
    "QUALIFICATION_ATTEMPT_ROOT",
    "QUALIFICATION_OUTPUT_ROOT",
    "QUALIFICATION_PLAN_OUTPUT",
    "SceneDiversityCpuBackendPlanError",
    "build_qualification_plan",
    "build_scientific_plan",
    "validate_cpu_plan",
]
