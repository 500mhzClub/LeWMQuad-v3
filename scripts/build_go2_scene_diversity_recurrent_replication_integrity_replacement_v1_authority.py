#!/usr/bin/env python3
"""Build, but never execute, the one-shot infrastructure-replacement authority.

The authority requires an independently reviewed replacement closure, the
science-identical plan, the unchanged scene panel, configuration, DINO binding,
gates and resource caps, the exact consumed V1 failure evidence, and a fresh
replacement root.  Merely importing or running tests for this module grants no
authority and creates no attempt root.
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
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v1_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v1 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v1 as runner  # noqa: E402


SOURCE_REVIEW_SCHEMA = runner.SOURCE_REVIEW_SCHEMA
SOURCE_REVIEW_STATUS = runner.SOURCE_REVIEW_STATUS

PREREGISTRATION = runner.PREREGISTRATION
SCENE_PANEL = runner.SCENE_PANEL
EXACT_PLAN = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_exact_plan_2026-08-04.json"
)
SOURCE_REVIEW = runner.SOURCE_REVIEW
AUTHORITY_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v1_execution_authority_2026-08-04.json"
)
ATTEMPT_ROOT = runner.DEFAULT_ATTEMPT_ROOT
COLLECTION_ROOT = runner.DEFAULT_COLLECTION_ROOT


class SceneDiversityReplacementAuthorityError(RuntimeError):
    """Raised before an incomplete replacement authority can be emitted."""


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
        raise SceneDiversityReplacementAuthorityError(
            "bound path is custody-protected"
        )
    try:
        return runner.file_binding_v1(selected)
    except (OSError, RuntimeError) as exc:
        raise SceneDiversityReplacementAuthorityError(str(exc)) from exc


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
        raise SceneDiversityReplacementAuthorityError(
            f"{label} binding is malformed"
        )
    observed = dict(value)
    if file_binding_v1(Path(str(observed["path"]))) != observed:
        raise SceneDiversityReplacementAuthorityError(f"{label} binding changed")
    return observed


def _read_json_binding(
    value: object, *, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    binding = _require_binding(value, label=label)
    try:
        document = json.loads(Path(str(binding["path"])).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityReplacementAuthorityError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(document, dict):
        raise SceneDiversityReplacementAuthorityError(
            f"{label} must be a JSON object"
        )
    return document, binding


def source_bindings_v1() -> dict[str, dict[str, object]]:
    if not isinstance(runner.SOURCE_PATHS, Mapping) or not runner.SOURCE_PATHS:
        raise SceneDiversityReplacementAuthorityError(
            "replacement runner source closure is empty"
        )
    evidence = runner.predecessor_failure_bindings_v1()
    bindings: dict[str, dict[str, object]] = {}
    for name, path in sorted(runner.SOURCE_PATHS.items()):
        if not isinstance(name, str) or not name or not isinstance(path, Path):
            raise SceneDiversityReplacementAuthorityError(
                "replacement runner source closure is malformed"
            )
        bindings[name] = file_binding_v1(path)
    if any(bindings[name] != binding for name, binding in evidence.items()):
        raise SceneDiversityReplacementAuthorityError(
            "predecessor failure evidence is not exact in the source closure"
        )
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
        raise SceneDiversityReplacementAuthorityError(
            "frozen DINO repository changed"
        )
    declaration["checkpoint_binding"] = checkpoint
    return declaration


def _require_fixed_binding(
    binding: object, *, path: Path, label: str
) -> dict[str, object]:
    observed = _require_binding(binding, label=label)
    if observed["path"] != str(path.resolve()):
        raise SceneDiversityReplacementAuthorityError(f"{label} path changed")
    return observed


def _validate_science_identical_plan_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    try:
        validated = collector._validate_scene_diversity_plan_v1(  # noqa: SLF001
            pilot.validate_plan(copy.deepcopy(dict(plan)))
        )
    except (pilot.PilotContractError, RuntimeError) as exc:
        raise SceneDiversityReplacementAuthorityError(str(exc)) from exc
    if validated != dict(plan):
        raise SceneDiversityReplacementAuthorityError(
            "replacement exact plan normalization changed"
        )
    frozen_expected = {
        "path": str(plan_builder.FROZEN_V1_EXACT_PLAN.resolve()),
        "sha256": plan_builder.FROZEN_V1_EXACT_PLAN_SHA256,
        "byte_count": plan_builder.FROZEN_V1_EXACT_PLAN_BYTE_COUNT,
    }
    if file_binding_v1(plan_builder.FROZEN_V1_EXACT_PLAN) != frozen_expected:
        raise SceneDiversityReplacementAuthorityError("frozen V1 exact plan changed")
    try:
        frozen_plan = json.loads(plan_builder.FROZEN_V1_EXACT_PLAN.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneDiversityReplacementAuthorityError(
            "frozen V1 exact plan is not strict JSON"
        ) from exc
    if (
        not isinstance(frozen_plan, dict)
        or set(validated) != set(frozen_plan)
        or validated.get("attempt_id") != plan_builder.DEFAULT_ATTEMPT_ID
        or validated.get("output_root") != str(COLLECTION_ROOT.resolve())
        or any(
            runner.canonical_bytes_v1(validated[field])
            != runner.canonical_bytes_v1(frozen_plan[field])
            for field in set(frozen_plan) - {"attempt_id", "output_root"}
        )
    ):
        raise SceneDiversityReplacementAuthorityError(
            "replacement plan is not science-identical to frozen V1"
        )
    return validated


def build_authority_v1(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate reviewed inputs and return, but do not write, the authority."""

    preregistration = _require_fixed_binding(
        preregistration_binding,
        path=PREREGISTRATION,
        label="replacement preregistration",
    )
    panel = _require_fixed_binding(
        scene_panel_binding,
        path=SCENE_PANEL,
        label="frozen V1 scene panel",
    )
    if panel != {
        "path": str(SCENE_PANEL.resolve()),
        "sha256": runner.SCENE_PANEL_SHA256,
        "byte_count": runner.SCENE_PANEL_BYTE_COUNT,
    }:
        raise SceneDiversityReplacementAuthorityError(
            "frozen V1 scene panel binding changed"
        )
    bound_plan_document, bound_plan = _read_json_binding(
        plan_binding, label="replacement exact plan"
    )
    if (
        bound_plan["path"] != str(EXACT_PLAN.resolve())
        or bound_plan_document != dict(plan)
    ):
        raise SceneDiversityReplacementAuthorityError(
            "replacement exact plan binding changed"
        )
    validated_plan = _validate_science_identical_plan_v1(plan)

    review_document, review_binding = _read_json_binding(
        source_review_binding, label="independent replacement source review"
    )
    if (
        review_binding["path"] != str(SOURCE_REVIEW.resolve())
        or review_document != dict(source_review)
    ):
        raise SceneDiversityReplacementAuthorityError(
            "independent replacement source review binding changed"
        )
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
        raise SceneDiversityReplacementAuthorityError(
            "independent replacement source review changed"
        )

    if ATTEMPT_ROOT.exists() or ATTEMPT_ROOT.is_symlink():
        raise SceneDiversityReplacementAuthorityError(
            "one-shot replacement attempt root is not fresh"
        )
    if COLLECTION_ROOT.exists() or COLLECTION_ROOT.is_symlink():
        raise SceneDiversityReplacementAuthorityError(
            "one-shot replacement collection root is not fresh"
        )
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
        raise SceneDiversityReplacementAuthorityError(
            "replacement authority field contract changed"
        )
    if authority["permissions"].get("retry_resume_overwrite") is not False:
        raise SceneDiversityReplacementAuthorityError(
            "replacement authority unexpectedly permits retry or resume"
        )
    return authority


def _write_json_exclusive(
    path: Path, value: Mapping[str, Any]
) -> dict[str, object]:
    selected = Path(path)
    if _protected(selected):
        raise SceneDiversityReplacementAuthorityError(
            "authority output path is custody-protected"
        )
    selected.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(selected, flags, 0o600)
    except OSError as exc:
        raise SceneDiversityReplacementAuthorityError(
            "authority output is not fresh"
        ) from exc
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
    plan, _ = _read_json_binding(plan_binding, label="replacement exact plan")
    review_binding = file_binding_v1(SOURCE_REVIEW)
    review, _ = _read_json_binding(
        review_binding, label="independent replacement source review"
    )
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
    "SceneDiversityReplacementAuthorityError",
    "build_authority_v1",
    "dino_declaration_v1",
    "file_binding_v1",
    "source_bindings_v1",
]
