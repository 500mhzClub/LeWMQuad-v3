#!/usr/bin/env python3
"""Generate and validate the requirements-only protected-contact freeze.

This entry point never imports the scientific evaluators and never opens a
sensor, calibration, held-out, JEPA, or training artifact.  Its only inputs are
the prospectively encoded requirements contract and the local source files
listed in ``SOURCE_CLOSURE_PATHS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from lewm.safety import protected_contact_scope_requirements_review_v1 as scope


DOCS = REPOSITORY / "docs"

CONTRACT_PATH = DOCS / "lewm_protected_contact_scope_contract_v1.json"
MATRIX_PATH = DOCS / "lewm_protected_contact_scope_link_object_context_matrix_v1.json"
TRACEABILITY_PATH = DOCS / "lewm_protected_contact_scope_requirement_traceability_v1.json"
ASSUMPTIONS_PATH = DOCS / "lewm_protected_contact_scope_assumptions_unresolved_v1.json"
SOURCE_CLOSURE_PATH = DOCS / "lewm_protected_contact_scope_source_closure_v1.json"

GENERATED_BUILDERS: Mapping[Path, Callable[[], dict[str, Any]]] = {
    CONTRACT_PATH: scope.build_contract_receipt,
    MATRIX_PATH: scope.build_link_object_context_matrix,
    TRACEABILITY_PATH: scope.build_traceability_matrix,
    ASSUMPTIONS_PATH: scope.build_assumption_inventory,
}

# The three predecessor results are bound only as preserved aggregate context.
# Their metric tables, error rows, and per-condition outcomes are forbidden
# Stage-A decision inputs even though their file identities are source-closed.
PRESERVED_CONTEXT_ONLY_PATHS = (
    "docs/lewm_go2_explicit_per_link_geometric_micro_state_upper_bound_v1_result_2026-08-21.md",
    "docs/lewm_go2_body_centric_range_coverage_qualification_v1_result_2026-08-25.md",
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_result_2026-08-26.md",
)

SOURCE_CLOSURE_PATHS = (
    "lewm/safety/protected_contact_scope_requirements_review_v1.py",
    "lewm/tests/test_protected_contact_scope_requirements_review_v1.py",
    "scripts/freeze_protected_contact_scope_requirements_review_v1.py",
    "docs/lewm_protected_contact_scope_requirements_review_v1.md",
    "docs/lewm_protected_contact_scope_traceability_matrix_v1.md",
    "docs/lewm_protected_contact_scope_decision_memo_v1.md",
    "docs/lewm_protected_contact_scope_assurance_fragment_v1.md",
    "docs/lewm_protected_contact_scope_contract_v1.json",
    "docs/lewm_protected_contact_scope_link_object_context_matrix_v1.json",
    "docs/lewm_protected_contact_scope_requirement_traceability_v1.json",
    "docs/lewm_protected_contact_scope_assumptions_unresolved_v1.json",
    "docs/lewm_contact_hazard_analysis_and_ontology_v1.md",
    "docs/lewm_contact_hazard_instrumentation_contract_v1.md",
    "docs/lewm_contact_hazard_ontology_development_result_v1.md",
    "docs/lewm_material_contact_safety_model_next_experiment_spec_v1.md",
    "docs/lewm_rollout_safety_and_trajectory_cleanup_2026-06-13.md",
    "docs/lewm_control_commitment_horizon_and_viability_v1_result.md",
    "docs/lewm_deployment_valid_strong_braking_mode_v1_result.md",
    "docs/lewm_go2_generalization_execution_contract_2026-07-09.md",
    "docs/lewm_safe_local_waypoint_task_spec_2026-08-19.md",
    "docs/lewm_factorised_risk_constrained_planner_design_2026-08-19.md",
    "docs/lewm_planner_design_decision_memo_2026-08-19.md",
    "docs/lewm_planner_evaluation_first_protocol_2026-08-19.md",
    "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_preregistration_2026-08-26.md",
    "docs/SAINTS_Year_1_Progression_Document-3.pdf",
    "lewm/safety/contact_hazard_ontology_v1.py",
    "lewm/safety/body_centric_range_coverage_corpus_v1.py",
    "lewm/oracle/go2_branch_oracle_v1_2.py",
    "config/go2_platform_manifest.yaml",
    "config/go2_primitive_registry.yaml",
    "lewm_worlds/lewm_worlds/manifest.py",
    "lewm_worlds/lewm_worlds/families.py",
    "third_party/unitree_go2_ros2/unitree_go2_description/urdf/const.xacro",
    "third_party/unitree_go2_ros2/unitree_go2_description/urdf/leg.xacro",
    "third_party/unitree_go2_ros2/unitree_go2_description/urdf/unitree_go2_robot.xacro",
    ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/geometry_index.json",
    ".generated/venvs/genesis_render_vulkan/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    *PRESERVED_CONTEXT_ONLY_PATHS,
)


class FreezeError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file_bytes(value: Any) -> bytes:
    return scope.canonical_json_bytes(value) + b"\n"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FreezeError(f"{path}: receipt must be a JSON object")
    return value


def build_source_closure() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    preserved = set(PRESERVED_CONTEXT_ONLY_PATHS)
    for relative in SOURCE_CLOSURE_PATHS:
        if relative in seen:
            raise FreezeError(f"duplicate source-closure path: {relative}")
        seen.add(relative)
        path = REPOSITORY / relative
        if not path.is_file():
            raise FreezeError(f"source-closure member missing: {relative}")
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "role": (
                    "PRESERVED_AGGREGATE_CONTEXT_ONLY_NO_OUTCOME_FIELD_AUTHORITY"
                    if relative in preserved
                    else "REQUIREMENTS_REVIEW_SOURCE_OR_FROZEN_ARTIFACT"
                ),
                "sha256": _sha256_file(path),
            }
        )
    return scope.with_content_digest(
        {
            "schema": "protected_contact_scope_source_closure_v1",
            "experiment_id": scope.EXPERIMENT_ID,
            "start_commit": scope.START_COMMIT,
            "files": rows,
            "file_count": len(rows),
            "outcome_fields_used_by_stage_a_scope_decision": [],
            "preserved_context_only_paths": list(PRESERVED_CONTEXT_ONLY_PATHS),
            "detailed_sensor_error_or_heldout_tables_authorized_before_freeze": False,
        }
    )


def validate_source_closure(value: Mapping[str, Any]) -> None:
    scope.validate_content_digest(value)
    if value.get("experiment_id") != scope.EXPERIMENT_ID:
        raise FreezeError("source closure experiment drift")
    if value.get("start_commit") != scope.START_COMMIT:
        raise FreezeError("source closure start commit drift")
    if value.get("outcome_fields_used_by_stage_a_scope_decision") != []:
        raise FreezeError("Stage-A source closure claims outcome use")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_CLOSURE_PATHS):
        raise FreezeError("source closure cardinality drift")
    if value.get("file_count") != len(rows):
        raise FreezeError("source closure file_count drift")
    observed_paths = [str(row.get("path")) for row in rows]
    if observed_paths != list(SOURCE_CLOSURE_PATHS):
        raise FreezeError("source closure path/order drift")
    for row in rows:
        path = REPOSITORY / str(row["path"])
        if not path.is_file():
            raise FreezeError(f"source closure member missing: {row['path']}")
        if row.get("bytes") != path.stat().st_size:
            raise FreezeError(f"source closure size drift: {row['path']}")
        if row.get("sha256") != _sha256_file(path):
            raise FreezeError(f"source closure SHA drift: {row['path']}")


def freeze() -> None:
    for path, builder in GENERATED_BUILDERS.items():
        value = builder()
        _atomic_write(path, _canonical_file_bytes(value))
    closure = build_source_closure()
    _atomic_write(SOURCE_CLOSURE_PATH, _canonical_file_bytes(closure))
    check()


def check() -> None:
    for path, builder in GENERATED_BUILDERS.items():
        if not path.is_file():
            raise FreezeError(f"frozen artifact missing: {path.relative_to(REPOSITORY)}")
        observed = path.read_bytes()
        expected = _canonical_file_bytes(builder())
        if observed != expected:
            raise FreezeError(f"frozen artifact drift: {path.relative_to(REPOSITORY)}")
    if not SOURCE_CLOSURE_PATH.is_file():
        raise FreezeError("source closure missing")
    closure = _load_json(SOURCE_CLOSURE_PATH)
    validate_source_closure(closure)
    expected_closure = _canonical_file_bytes(build_source_closure())
    if SOURCE_CLOSURE_PATH.read_bytes() != expected_closure:
        raise FreezeError("source closure byte regeneration drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "check"))
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        freeze()
    else:
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
