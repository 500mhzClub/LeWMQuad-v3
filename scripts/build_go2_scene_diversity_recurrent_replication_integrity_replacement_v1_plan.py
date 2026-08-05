#!/usr/bin/env python3
"""Build the science-identical scene-diversity replacement plan.

The consumed V1 plan is the only scientific plan input.  This builder changes
only the attempt identifier and collection output root, requires a fresh
ordinary development root, and leaves every scene, state, action, runtime,
rendering, model, metric, gate, and resource field byte-equivalent as canonical
JSON.  It never opens a generated V1 attempt artifact.
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
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as frozen_collector  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_plan_v1 as frozen_builder  # noqa: E402


FROZEN_V1_ATTEMPT_ID = "go2-scene-diversity-recurrent-replication-v1"
FROZEN_V1_COLLECTION_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_v1/"
    "attempt_v1/collection"
)
FROZEN_V1_EXACT_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "exact_plan_2026-08-04.json"
)
FROZEN_V1_EXACT_PLAN_SHA256 = (
    "c34aa23303951d32dd9686a607de7b78df06db026918d868017a6a93c506a040"
)
FROZEN_V1_EXACT_PLAN_BYTE_COUNT = 346_027
FROZEN_SCENE_PANEL = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "scene_panel_2026-08-04.json"
)
FROZEN_SCENE_PANEL_SHA256 = (
    "df145c2d70d82243b373ef6f6d8750dc231f9de2a4d07d9f698a1831b9b84fa7"
)
FROZEN_SCENE_PANEL_BYTE_COUNT = 207_218

DEFAULT_ATTEMPT_ID = (
    "go2-scene-diversity-recurrent-replication-integrity-replacement-v1"
)
DEFAULT_ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/"
    "go2_scene_diversity_recurrent_replication_integrity_replacement_v1/"
    "attempt_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ATTEMPT_ROOT / "collection"
DEFAULT_PLAN_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_exact_plan_2026-08-04.json"
)

_MUTABLE_FIELDS = frozenset({"attempt_id", "output_root"})


class SceneDiversityReplacementPlanError(RuntimeError):
    """Raised before a changed or non-fresh replacement plan is emitted."""


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


def _validated_frozen_plan_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        frozen_binding = pilot.file_binding(FROZEN_V1_EXACT_PLAN)
        frozen_document = json.loads(FROZEN_V1_EXACT_PLAN.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityReplacementPlanError(
            "frozen V1 exact plan cannot be read exactly"
        ) from exc
    if (
        frozen_binding
        != {
            "path": str(FROZEN_V1_EXACT_PLAN.resolve()),
            "file_sha256": FROZEN_V1_EXACT_PLAN_SHA256,
            "byte_count": FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
        }
        or not isinstance(frozen_document, dict)
        or dict(value) != frozen_document
    ):
        raise SceneDiversityReplacementPlanError(
            "frozen V1 exact plan binding or content changed"
        )
    try:
        plan = pilot.validate_plan(copy.deepcopy(dict(value)))
        frozen_collector._validate_scene_diversity_plan_v1(plan)  # noqa: SLF001
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityReplacementPlanError(str(exc)) from exc
    if (
        plan != dict(value)
        or plan.get("attempt_id") != FROZEN_V1_ATTEMPT_ID
        or plan.get("output_root") != str(FROZEN_V1_COLLECTION_ROOT.resolve())
    ):
        raise SceneDiversityReplacementPlanError(
            "frozen V1 plan identity or normalization changed"
        )
    return plan


def _validate_output_root_v1(path: Path) -> Path:
    selected = Path(path)
    development_root = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    resolved = selected.resolve(strict=False)
    if (
        not selected.is_absolute()
        or not resolved.is_relative_to(development_root)
        or resolved == development_root
        or resolved != DEFAULT_OUTPUT_ROOT.resolve(strict=False)
        or DEFAULT_ATTEMPT_ROOT.exists()
        or DEFAULT_ATTEMPT_ROOT.is_symlink()
        or selected.exists()
        or selected.is_symlink()
        or _protected(selected)
    ):
        raise SceneDiversityReplacementPlanError(
            "replacement output_root must be the exact fresh replacement collection path"
        )
    return resolved


def build_plan_v1(
    *,
    frozen_plan: Mapping[str, Any],
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Return the V1 plan with only its fresh replacement identity changed."""

    if attempt_id != DEFAULT_ATTEMPT_ID:
        raise SceneDiversityReplacementPlanError(
            "replacement attempt identifier changed"
        )
    original = _validated_frozen_plan_v1(frozen_plan)
    selected_root = _validate_output_root_v1(output_root)
    replacement = copy.deepcopy(original)
    replacement["attempt_id"] = attempt_id
    replacement["output_root"] = str(selected_root)
    try:
        validated = pilot.validate_plan(replacement)
        frozen_collector._validate_scene_diversity_plan_v1(validated)  # noqa: SLF001
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityReplacementPlanError(str(exc)) from exc
    if validated != replacement:
        raise SceneDiversityReplacementPlanError(
            "replacement plan normalization changed the plan"
        )
    if set(replacement) != set(original) or any(
        _canonical_bytes(replacement[field]) != _canonical_bytes(original[field])
        for field in set(original) - _MUTABLE_FIELDS
    ):
        raise SceneDiversityReplacementPlanError(
            "replacement changed a frozen scientific or infrastructure field"
        )
    return replacement


def _load_frozen_json_v1(
    path: Path, *, sha256: str, byte_count: int, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _protected(path):
        raise SceneDiversityReplacementPlanError(f"{label} path is custody-protected")
    try:
        value, binding = pilot.read_bound_json(
            path,
            expected_sha256=sha256,
            expected_byte_count=byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise SceneDiversityReplacementPlanError(str(exc)) from exc
    if not isinstance(value, dict):
        raise SceneDiversityReplacementPlanError(f"{label} must be a JSON object")
    return value, binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan_output.exists() or args.plan_output.is_symlink():
        raise SceneDiversityReplacementPlanError(
            "replacement plan output must be fresh"
        )
    frozen_plan, frozen_binding = _load_frozen_json_v1(
        FROZEN_V1_EXACT_PLAN,
        sha256=FROZEN_V1_EXACT_PLAN_SHA256,
        byte_count=FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
        label="frozen V1 exact plan",
    )
    _panel, panel_binding = _load_frozen_json_v1(
        FROZEN_SCENE_PANEL,
        sha256=FROZEN_SCENE_PANEL_SHA256,
        byte_count=FROZEN_SCENE_PANEL_BYTE_COUNT,
        label="frozen V1 scene panel",
    )
    plan = build_plan_v1(
        frozen_plan=frozen_plan,
        attempt_id=args.attempt_id,
        output_root=args.output_root,
    )
    plan_binding = pilot.write_json_exclusive(args.plan_output, plan)
    print(
        json.dumps(
            {
                "plan": plan_binding,
                "frozen_v1_plan": frozen_binding,
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
    "SceneDiversityReplacementPlanError",
    "build_plan_v1",
]
