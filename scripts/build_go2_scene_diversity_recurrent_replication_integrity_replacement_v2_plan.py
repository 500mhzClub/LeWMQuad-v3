#!/usr/bin/env python3
"""Build the science-identical one-scene-process replacement V2 plan.

The original frozen V1 exact plan is the only scientific plan input.  V2
changes only the attempt identifier and collection output root; every scene,
state, action, seed, runtime/render contract, count, cap, model, metric and gate
remains canonically identical.
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
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan as predecessor_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as frozen_collector  # noqa: E402


FROZEN_V1_ATTEMPT_ID = predecessor_builder.FROZEN_V1_ATTEMPT_ID
FROZEN_V1_COLLECTION_ROOT = predecessor_builder.FROZEN_V1_COLLECTION_ROOT
FROZEN_V1_EXACT_PLAN = predecessor_builder.FROZEN_V1_EXACT_PLAN
FROZEN_V1_EXACT_PLAN_SHA256 = predecessor_builder.FROZEN_V1_EXACT_PLAN_SHA256
FROZEN_V1_EXACT_PLAN_BYTE_COUNT = predecessor_builder.FROZEN_V1_EXACT_PLAN_BYTE_COUNT
FROZEN_SCENE_PANEL = predecessor_builder.FROZEN_SCENE_PANEL
FROZEN_SCENE_PANEL_SHA256 = predecessor_builder.FROZEN_SCENE_PANEL_SHA256
FROZEN_SCENE_PANEL_BYTE_COUNT = predecessor_builder.FROZEN_SCENE_PANEL_BYTE_COUNT

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-integrity-replacement-v2"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v2/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v2_exact_plan_2026-08-04.json"
)


class SceneDiversityReplacementV2PlanError(RuntimeError):
    """Raised before a changed or non-fresh V2 plan can be emitted."""


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


def _validate_output_root_v2(path: Path) -> Path:
    selected = Path(path)
    development_root = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    resolved = selected.resolve(strict=False)
    if (
        not selected.is_absolute()
        or not resolved.is_relative_to(development_root)
        or resolved != DEFAULT_OUTPUT_ROOT.resolve(strict=False)
        or DEFAULT_ATTEMPT_ROOT.exists()
        or DEFAULT_ATTEMPT_ROOT.is_symlink()
        or selected.exists()
        or selected.is_symlink()
        or _protected(selected)
    ):
        raise SceneDiversityReplacementV2PlanError(
            "V2 output_root must be the exact fresh replacement collection path"
        )
    return resolved


def build_plan_v2(
    *,
    frozen_plan: Mapping[str, Any],
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Return frozen V1 with only the prospective V2 identity replaced."""

    if attempt_id != DEFAULT_ATTEMPT_ID:
        raise SceneDiversityReplacementV2PlanError("V2 attempt identifier changed")
    try:
        original = predecessor_builder._validated_frozen_plan_v1(  # noqa: SLF001
            frozen_plan
        )
    except predecessor_builder.SceneDiversityReplacementPlanError as exc:
        raise SceneDiversityReplacementV2PlanError(str(exc)) from exc
    selected_root = _validate_output_root_v2(output_root)
    replacement = copy.deepcopy(original)
    replacement["attempt_id"] = attempt_id
    replacement["output_root"] = str(selected_root)
    try:
        validated = pilot.validate_plan(replacement)
        frozen_collector._validate_scene_diversity_plan_v1(validated)  # noqa: SLF001
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityReplacementV2PlanError(str(exc)) from exc
    if validated != replacement:
        raise SceneDiversityReplacementV2PlanError(
            "V2 plan normalization changed the plan"
        )
    if set(replacement) != set(original) or any(
        _canonical_bytes(replacement[field]) != _canonical_bytes(original[field])
        for field in set(original) - {"attempt_id", "output_root"}
    ):
        raise SceneDiversityReplacementV2PlanError(
            "V2 changed a frozen scientific or infrastructure field"
        )
    return replacement


build_plan_v1 = build_plan_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan_output.exists() or args.plan_output.is_symlink():
        raise SceneDiversityReplacementV2PlanError("V2 plan output must be fresh")
    try:
        frozen_plan, frozen_binding = predecessor_builder._load_frozen_json_v1(  # noqa: SLF001
            FROZEN_V1_EXACT_PLAN,
            sha256=FROZEN_V1_EXACT_PLAN_SHA256,
            byte_count=FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
            label="frozen original V1 exact plan",
        )
        _panel, panel_binding = predecessor_builder._load_frozen_json_v1(  # noqa: SLF001
            FROZEN_SCENE_PANEL,
            sha256=FROZEN_SCENE_PANEL_SHA256,
            byte_count=FROZEN_SCENE_PANEL_BYTE_COUNT,
            label="frozen V1 scene panel",
        )
    except predecessor_builder.SceneDiversityReplacementPlanError as exc:
        raise SceneDiversityReplacementV2PlanError(str(exc)) from exc
    plan = build_plan_v2(
        frozen_plan=frozen_plan,
        attempt_id=args.attempt_id,
        output_root=args.output_root,
    )
    plan_binding = pilot.write_json_exclusive(args.plan_output, plan)
    print(
        json.dumps(
            {
                "plan": plan_binding,
                "frozen_original_v1_plan": frozen_binding,
                "frozen_scene_panel": panel_binding,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ATTEMPT_ID",
    "DEFAULT_ATTEMPT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "FROZEN_SCENE_PANEL",
    "FROZEN_V1_EXACT_PLAN",
    "SceneDiversityReplacementV2PlanError",
    "build_plan_v2",
]
