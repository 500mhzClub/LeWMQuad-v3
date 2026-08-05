#!/usr/bin/env python3
"""Build, but never execute, final identity-only replacement V3 authority.

The reviewed V2 authority builder remains the complete authority contract.
This wrapper supplies only the fresh V3 documents, identity and source
closure; all data, process, model, metric, gate, cap and permission fields are
unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v2_authority as predecessor_authority  # noqa: E402
from scripts import build_go2_scene_diversity_recurrent_replication_integrity_replacement_v3_plan as plan_builder  # noqa: E402
from scripts import collect_go2_scene_diversity_recurrent_replication_integrity_replacement_v3 as collector  # noqa: E402
from scripts import run_go2_scene_diversity_recurrent_replication_integrity_replacement_v3 as runner  # noqa: E402


SOURCE_REVIEW_SCHEMA = runner.SOURCE_REVIEW_SCHEMA
SOURCE_REVIEW_STATUS = runner.SOURCE_REVIEW_STATUS
PREREGISTRATION = runner.PREREGISTRATION
SCENE_PANEL = runner.SCENE_PANEL
EXACT_PLAN = plan_builder.DEFAULT_PLAN_OUTPUT
SOURCE_REVIEW = runner.SOURCE_REVIEW
AUTHORITY_OUTPUT = REPO_ROOT / (
    "docs/lewm_go2_scene_diversity_recurrent_replication_"
    "integrity_replacement_v3_execution_authority_2026-08-04.json"
)
ATTEMPT_ROOT = runner.DEFAULT_ATTEMPT_ROOT
COLLECTION_ROOT = runner.DEFAULT_COLLECTION_ROOT

# Exact V2 review requirements are retained, not revised for V3.
REQUIRED_RNG_EQUIVALENCE_AUDIT = (
    predecessor_authority.REQUIRED_RNG_EQUIVALENCE_AUDIT
)
REQUIRED_PROCESS_EVIDENCE_AUDIT = (
    predecessor_authority.REQUIRED_PROCESS_EVIDENCE_AUDIT
)
SceneDiversityReplacementAuthorityError = (
    predecessor_authority.SceneDiversityReplacementAuthorityError
)

_CONFIGURATION_LOCK = threading.RLock()


def _configuration_overrides_v3() -> dict[str, object]:
    return {
        "runner": runner,
        "plan_builder": plan_builder,
        "collector": collector,
        "SOURCE_REVIEW_SCHEMA": SOURCE_REVIEW_SCHEMA,
        "SOURCE_REVIEW_STATUS": SOURCE_REVIEW_STATUS,
        "PREREGISTRATION": PREREGISTRATION,
        "SCENE_PANEL": SCENE_PANEL,
        "EXACT_PLAN": EXACT_PLAN,
        "SOURCE_REVIEW": SOURCE_REVIEW,
        "AUTHORITY_OUTPUT": AUTHORITY_OUTPUT,
        "ATTEMPT_ROOT": ATTEMPT_ROOT,
        "COLLECTION_ROOT": COLLECTION_ROOT,
        "REQUIRED_RNG_EQUIVALENCE_AUDIT": REQUIRED_RNG_EQUIVALENCE_AUDIT,
        "REQUIRED_PROCESS_EVIDENCE_AUDIT": REQUIRED_PROCESS_EVIDENCE_AUDIT,
    }


@contextmanager
def _configured_predecessor_authority_v3() -> Iterator[None]:
    with _CONFIGURATION_LOCK:
        overrides = _configuration_overrides_v3()
        originals = {
            name: getattr(predecessor_authority, name) for name in overrides
        }
        try:
            for name, value in overrides.items():
                setattr(predecessor_authority, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(predecessor_authority, name, value)


def file_binding_v3(path: Path) -> dict[str, object]:
    with _configured_predecessor_authority_v3():
        return predecessor_authority.file_binding_v2(path)


file_binding_v2 = file_binding_v3
file_binding_v1 = file_binding_v3


def source_bindings_v3() -> dict[str, dict[str, object]]:
    with _configured_predecessor_authority_v3():
        return predecessor_authority.source_bindings_v2()


source_bindings_v2 = source_bindings_v3
source_bindings_v1 = source_bindings_v3


def dino_declaration_v3() -> dict[str, object]:
    with _configured_predecessor_authority_v3():
        return predecessor_authority.dino_declaration_v2()


dino_declaration_v2 = dino_declaration_v3
dino_declaration_v1 = dino_declaration_v3


def _validate_science_identical_plan_v3(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    with _configured_predecessor_authority_v3():
        return predecessor_authority._validate_science_identical_plan_v2(  # noqa: SLF001
            plan
        )


def build_authority_v3(
    *,
    preregistration_binding: Mapping[str, Any],
    scene_panel_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    source_review: Mapping[str, Any],
    source_review_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return validated V3 authority without writing or executing it."""

    with _configured_predecessor_authority_v3():
        return predecessor_authority.build_authority_v2(
            preregistration_binding=preregistration_binding,
            scene_panel_binding=scene_panel_binding,
            plan=plan,
            plan_binding=plan_binding,
            source_review=source_review,
            source_review_binding=source_review_binding,
        )


build_authority_v2 = build_authority_v3
build_authority_v1 = build_authority_v3


def build_parser():
    with _configured_predecessor_authority_v3():
        return predecessor_authority.build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    with _configured_predecessor_authority_v3():
        return predecessor_authority.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_OUTPUT",
    "COLLECTION_ROOT",
    "REQUIRED_PROCESS_EVIDENCE_AUDIT",
    "REQUIRED_RNG_EQUIVALENCE_AUDIT",
    "SOURCE_REVIEW_SCHEMA",
    "SOURCE_REVIEW_STATUS",
    "SceneDiversityReplacementAuthorityError",
    "build_authority_v1",
    "build_authority_v2",
    "build_authority_v3",
    "dino_declaration_v1",
    "dino_declaration_v2",
    "dino_declaration_v3",
    "file_binding_v1",
    "file_binding_v2",
    "file_binding_v3",
    "source_bindings_v1",
    "source_bindings_v2",
    "source_bindings_v3",
]
