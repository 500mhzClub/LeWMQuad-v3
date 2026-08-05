#!/usr/bin/env python3
"""Build the one-shot scene-diversity recurrent replication authority.

The authority binds the independently reviewed source closure, the frozen
preregistration, scene panel and exact plan, the unchanged recurrent benchmark
configuration, and the exact collection limits.  It does not create an attempt
directory, collect data, train a model, or grant retry/resume authority.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as benchmark  # noqa: E402
from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_v1 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_v1 as runner  # noqa: E402


SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "preregistration_2026-08-04.md"
)
SCENE_PANEL = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "scene_panel_2026-08-04.json"
)
EXACT_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "exact_plan_2026-08-04.json"
)
SOURCE_REVIEW = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "source_review_2026-08-04.json"
)
AUTHORITY_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_v1_"
    "execution_authority_2026-08-04.json"
)
ATTEMPT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_scene_diversity_recurrent_replication_v1/attempt_v1"
)
COLLECTION_ROOT = ATTEMPT_ROOT / "collection"


class SceneDiversityAuthorityError(RuntimeError):
    """Raised before an incomplete or inconsistent authority can be emitted."""


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


def file_binding_v1(path: Path) -> dict[str, object]:
    selected = Path(path)
    if _protected(selected):
        raise SceneDiversityAuthorityError("bound path is custody-protected")
    try:
        return runner.file_binding_v1(selected)
    except (OSError, RuntimeError) as exc:
        raise SceneDiversityAuthorityError(str(exc)) from exc


def _require_binding(value: object, *, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise SceneDiversityAuthorityError(f"{label} binding is malformed")
    observed = dict(value)
    if file_binding_v1(Path(str(observed["path"]))) != observed:
        raise SceneDiversityAuthorityError(f"{label} binding changed")
    return observed


def _read_json_binding(
    value: object, *, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    binding = _require_binding(value, label=label)
    try:
        document = json.loads(Path(str(binding["path"])).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityAuthorityError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise SceneDiversityAuthorityError(f"{label} must be a JSON object")
    return document, binding


def source_bindings_v1() -> dict[str, dict[str, object]]:
    if not isinstance(runner.SOURCE_PATHS, Mapping) or not runner.SOURCE_PATHS:
        raise SceneDiversityAuthorityError("runner source closure is empty")
    bindings: dict[str, dict[str, object]] = {}
    for name, path in sorted(runner.SOURCE_PATHS.items()):
        if not isinstance(name, str) or not name or not isinstance(path, Path):
            raise SceneDiversityAuthorityError("runner source closure is malformed")
        bindings[name] = file_binding_v1(path)
    return bindings


def dino_declaration_v1() -> dict[str, object]:
    declaration = copy.deepcopy(runner.expected_dino_v1())
    checkpoint = _require_binding(
        declaration["checkpoint_binding"], label="frozen DINO checkpoint"
    )
    repository = Path(str(declaration["repository_path"]))
    if (
        _protected(repository)
        or not repository.is_dir()
        or repository.is_symlink()
        or not isinstance(declaration["repository_commit"], str)
        or len(str(declaration["repository_commit"])) != 40
    ):
        raise SceneDiversityAuthorityError("frozen DINO repository changed")
    declaration["checkpoint_binding"] = checkpoint
    return declaration


def _require_fixed_binding(
    binding: object, *, path: Path, label: str
) -> dict[str, object]:
    observed = _require_binding(binding, label=label)
    if observed["path"] != str(path.resolve()):
        raise SceneDiversityAuthorityError(f"{label} path changed")
    return observed


def build_authority_v1(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate reviewed inputs and return the authority without writing it."""

    preregistration = _require_fixed_binding(
        preregistration_binding,
        path=PREREGISTRATION,
        label="frozen preregistration",
    )
    panel = _require_fixed_binding(
        scene_panel_binding, path=SCENE_PANEL, label="frozen scene panel"
    )
    bound_plan_document, bound_plan = _read_json_binding(
        plan_binding, label="frozen exact plan"
    )
    if bound_plan["path"] != str(EXACT_PLAN.resolve()) or bound_plan_document != dict(
        plan
    ):
        raise SceneDiversityAuthorityError("frozen exact plan binding changed")
    try:
        validated_plan = collector._validate_scene_diversity_plan_v1(  # noqa: SLF001
            pilot.validate_plan(plan)
        )
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityAuthorityError(str(exc)) from exc
    if (
        validated_plan.get("attempt_id")
        != "go2-scene-diversity-recurrent-replication-v1"
        or validated_plan.get("output_root") != str(COLLECTION_ROOT.resolve())
    ):
        raise SceneDiversityAuthorityError("fixed attempt/collection identity changed")

    review_document, review_binding = _read_json_binding(
        source_review_binding, label="independent source review"
    )
    if review_binding["path"] != str(SOURCE_REVIEW.resolve()) or review_document != dict(
        source_review
    ):
        raise SceneDiversityAuthorityError("independent source review binding changed")
    sources = source_bindings_v1()
    if (
        source_review.get("schema") != SOURCE_REVIEW_SCHEMA
        or source_review.get("status") != SOURCE_REVIEW_STATUS
        or source_review.get("protected_material_opened") is not False
        or source_review.get("findings") != []
        or source_review.get("preregistration_binding") != preregistration
        or source_review.get("scene_panel_binding") != panel
        or source_review.get("plan_binding") != bound_plan
        or source_review.get("source_bindings") != sources
    ):
        raise SceneDiversityAuthorityError("independent source review changed")

    if ATTEMPT_ROOT.exists() or ATTEMPT_ROOT.is_symlink():
        raise SceneDiversityAuthorityError("one-shot attempt root is not fresh")
    if COLLECTION_ROOT.exists() or COLLECTION_ROOT.is_symlink():
        raise SceneDiversityAuthorityError("one-shot collection root is not fresh")
    authority = {
        "schema": collector.AUTHORITY_SCHEMA,
        "status": collector.AUTHORITY_STATUS,
        "attempt_id": str(validated_plan["attempt_id"]),
        "attempt_root": str(ATTEMPT_ROOT.resolve(strict=False)),
        "collection_root": str(COLLECTION_ROOT.resolve(strict=False)),
        "plan_binding": bound_plan,
        "preregistration_binding": preregistration,
        "source_review_binding": review_binding,
        "source_bindings": sources,
        "dino": dino_declaration_v1(),
        "config": benchmark.config_v1(),
        "caps": copy.deepcopy(collector.EXPECTED_CAPS),
        "permissions": copy.deepcopy(collector.EXPECTED_PERMISSIONS),
    }
    if set(authority) != collector.AUTHORITY_FIELDS:
        raise SceneDiversityAuthorityError("authority field contract changed")
    return authority


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, object]:
    selected = Path(path)
    if _protected(selected):
        raise SceneDiversityAuthorityError("authority output path is custody-protected")
    selected.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(selected, flags, 0o600)
    except OSError as exc:
        raise SceneDiversityAuthorityError("authority output is not fresh") from exc
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding_v1(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUTHORITY_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = file_binding_v1(PREREGISTRATION)
    scene_panel = file_binding_v1(SCENE_PANEL)
    plan_binding = file_binding_v1(EXACT_PLAN)
    plan, _ = _read_json_binding(plan_binding, label="frozen exact plan")
    review_binding = file_binding_v1(SOURCE_REVIEW)
    review, _ = _read_json_binding(review_binding, label="independent source review")
    authority = build_authority_v1(
        preregistration_binding=preregistration,
        scene_panel_binding=scene_panel,
        plan=plan,
        plan_binding=plan_binding,
        source_review=review,
        source_review_binding=review_binding,
    )
    binding = _write_json_exclusive(args.output, authority)
    print(json.dumps({"authority": binding}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "COLLECTION_ROOT",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "SceneDiversityAuthorityError",
    "build_authority_v1",
    "dino_declaration_v1",
    "file_binding_v1",
    "source_bindings_v1",
]
