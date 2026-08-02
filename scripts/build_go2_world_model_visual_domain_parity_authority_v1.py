#!/usr/bin/env python3
"""Build source review metadata and one exact RGB-parity authority.

This helper cannot review its own source and never renders a frame.  Its
review template is deliberately non-passing.  A passing, independent source
review and one immutable parity plan are required before it can bind the only
authorized 8-scene, 32-pose, double-render attempt.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import build_go2_world_model_visual_domain_parity_plan_v1 as plan_builder  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as runtime_kernel  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_visual_domain_parity_generation_authority_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_8_SCENE_32_POSE_DOUBLE_RENDER"
MAX_WALL_SECONDS = 7_200.0
MAX_STORED_RGB_BYTES = 512 * 1024**2
MAX_PARITY_OUTPUT_BYTES = 1024**3
PROJECTED_PIPELINE_NEW_BYTES = 3 * 1024**3
FREE_SPACE_MARGIN_BYTES = 1024**3
REQUIRED_PREFLIGHT_FREE_BYTES = (
    PROJECTED_PIPELINE_NEW_BYTES + FREE_SPACE_MARGIN_BYTES
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")

PARITY_SOURCE_PATHS = {
    "parity_plan_builder": (
        "scripts/build_go2_world_model_visual_domain_parity_plan_v1.py"
    ),
    "parity_authority_builder": (
        "scripts/build_go2_world_model_visual_domain_parity_authority_v1.py"
    ),
    "parity_supervisor": (
        "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py"
    ),
    "parity_evaluator": (
        "scripts/evaluate_go2_world_model_visual_domain_parity_v1.py"
    ),
    "parity_scene_inventory": (
        "scripts/build_go2_world_model_bounded_branch_scene_panel_v1.py"
    ),
    "parity_runtime_contract_builder": (
        "scripts/build_go2_world_model_counterfactual_calibration_plan_v1.py"
    ),
    "parity_graphics_supervisor_helpers": (
        "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    ),
    "calibration_analyzer": (
        "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
    ),
    "historical_textured_v03_renderer": "scripts/render_replay_v03.py",
    "parity_runtime_boundary_test": (
        "lewm/tests/test_go2_world_model_visual_domain_parity_authorized_v1.py"
    ),
    "parity_plan_test": (
        "lewm/tests/test_go2_world_model_visual_domain_parity_plan_v1.py"
    ),
    "shared_textured_v03_helper_test": (
        "lewm/tests/test_go2_world_model_counterfactual_textured_v03.py"
    ),
    "parity_evaluator_lineage_test": (
        "lewm/tests/test_go2_world_model_bounded_branch_lineage_v1.py"
    ),
    "bounded_branch_runbook": (
        "docs/lewm_go2_world_model_bounded_branch_experiment_v1_runbook_2026-08-02.md"
    ),
}


class VisualDomainParityAuthorityError(RuntimeError):
    """Raised before malformed metadata can mint parity authority."""


def _nofollow_regular(path: Path, *, label: str) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    if any(
        part.lower() == "sealed_test.json"
        or part.lower() == "sealed"
        or part.lower().startswith("sealed_")
        or part.lower() in {"heldout", "held_out", "held-out", "protected"}
        or part.lower().startswith("heldout_")
        or part.lower().startswith("held_out_")
        or part.lower().startswith("held-out-")
        or part.lower().startswith("protected_")
        for part in selected.parts
    ):
        raise VisualDomainParityAuthorityError(f"{label} names protected material")
    cursor = Path(selected.anchor)
    for part in selected.parts[1:]:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise VisualDomainParityAuthorityError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise VisualDomainParityAuthorityError(f"{label} contains a symlink")
    if not selected.is_file() or selected.resolve(strict=True) != selected:
        raise VisualDomainParityAuthorityError(f"{label} is not a regular file")
    return selected


def canonical_source_paths_v1() -> dict[str, str]:
    paths = {**runtime_kernel.EXPECTED_SOURCE_PATHS, **PARITY_SOURCE_PATHS}
    paths["external_supervisor"] = PARITY_SOURCE_PATHS["parity_supervisor"]
    return dict(sorted(paths.items()))


def _require_iso8601(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualDomainParityAuthorityError(f"{label} is empty")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VisualDomainParityAuthorityError(f"{label} is not ISO-8601") from exc
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualDomainParityAuthorityError(f"{label} is empty")
    return value


def _require_commit_ancestor(source_commit: str) -> None:
    if _COMMIT.fullmatch(source_commit) is None:
        raise VisualDomainParityAuthorityError("source commit is invalid")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise VisualDomainParityAuthorityError(
            "source commit is not an ancestor of HEAD"
        ) from exc


def committed_source_bindings_v1(source_commit: str) -> list[dict[str, Any]]:
    """Bind the exact current runtime closure and prove it at one commit."""

    _require_commit_ancestor(source_commit)
    rows = []
    for name, relative in canonical_source_paths_v1().items():
        binding = pilot.file_binding(
            _nofollow_regular(
                REPO_ROOT / relative, label=f"visual-domain parity source {name}"
            )
        )
        try:
            runtime_kernel._binding_at_commit(  # noqa: SLF001
                binding,
                commit=source_commit,
                label=f"visual-domain parity source {name}",
            )
        except pilot.PilotContractError as exc:
            raise VisualDomainParityAuthorityError(str(exc)) from exc
        rows.append({"name": name, "binding": binding})
    return rows


def build_source_review_template_v1(*, source_commit: str) -> dict[str, Any]:
    return {
        "schema": pilot.SOURCE_REVIEW_SCHEMA,
        "status": "PENDING_INDEPENDENT_REVIEW",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": source_commit,
        "reviewed_source_bindings": committed_source_bindings_v1(source_commit),
        "remaining_findings": ["INDEPENDENT_REVIEW_REQUIRED"],
        "reviewer": {
            "identity": "REVIEWER_MUST_REPLACE",
            "independence_basis": "REVIEWER_MUST_REPLACE",
        },
        "reviewed_at": "REVIEWER_MUST_REPLACE_WITH_ISO8601",
        "review_method": ["REVIEWER_MUST_REPLACE"],
        "test_evidence": ["REVIEWER_MUST_REPLACE"],
        "accepted_limitations": ["REVIEWER_MUST_REPLACE"],
    }


def _read_exact_document(
    value: Mapping[str, Any], binding: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    try:
        normalized_binding = pilot.require_binding(binding, label=label)
        observed, actual = pilot.read_bound_json(
            _nofollow_regular(Path(str(normalized_binding["path"])), label=label),
            expected_sha256=str(normalized_binding["file_sha256"]),
            expected_byte_count=int(normalized_binding["byte_count"]),
            label=label,
        )
    except (OSError, pilot.PilotContractError) as exc:
        raise VisualDomainParityAuthorityError(str(exc)) from exc
    if (
        actual != normalized_binding
        or not isinstance(observed, Mapping)
        or pilot.canonical_json_bytes(observed) != pilot.canonical_json_bytes(value)
    ):
        raise VisualDomainParityAuthorityError(f"{label} document changed")
    return normalized_binding


def _expected_caps(plan: Mapping[str, Any], *, wall_seconds: float) -> dict[str, Any]:
    return {
        **dict(plan_builder.EXPECTED_COUNTS),
        "maximum_stored_rgb_bytes": MAX_STORED_RGB_BYTES,
        "maximum_parity_output_bytes": MAX_PARITY_OUTPUT_BYTES,
        "projected_pipeline_new_bytes": PROJECTED_PIPELINE_NEW_BYTES,
        "free_space_margin_bytes": FREE_SPACE_MARGIN_BYTES,
        "required_preflight_free_bytes": REQUIRED_PREFLIGHT_FREE_BYTES,
        "wall_seconds": float(wall_seconds),
    }


def validate_authority_v1(
    authority: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    require_fresh_output: bool,
) -> dict[str, Any]:
    """Reopen all authority inputs and fail closed before reservation."""

    try:
        normalized_plan = plan_builder.validate_plan_v1(
            plan, require_fresh_output=require_fresh_output
        )
    except plan_builder.VisualDomainParityPlanError as exc:
        raise VisualDomainParityAuthorityError(str(exc)) from exc
    normalized_plan_binding = _read_exact_document(
        normalized_plan, plan_binding, label="visual-domain parity plan"
    )
    normalized_review_binding = _read_exact_document(
        review, review_binding, label="visual-domain parity source review"
    )
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_authorized",
        "authorizer",
        "issued_at",
        "source_commit",
        "review_binding",
        "plan_binding",
        "source_bindings",
        "attempt",
        "caps",
        "runtime_bindings",
        "execution_contract",
        "network_access",
        "external_supervisor",
        "platform_gate_disposition",
    }
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise VisualDomainParityAuthorityError("parity authority fields changed")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authority_granted_by_this_document") is not True
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("plan_binding") != normalized_plan_binding
        or authority.get("review_binding") != normalized_review_binding
        or authority.get("network_access") is not False
    ):
        raise VisualDomainParityAuthorityError("parity authority identity changed")
    authorizer = authority.get("authorizer")
    if not isinstance(authorizer, Mapping) or set(authorizer) != {"identity", "basis"}:
        raise VisualDomainParityAuthorityError("parity authorizer fields changed")
    _require_nonempty(authorizer.get("identity"), label="authorizer identity")
    _require_nonempty(authorizer.get("basis"), label="authorizer basis")
    _require_iso8601(authority.get("issued_at"), label="issued_at")
    source_commit = authority.get("source_commit")
    if not isinstance(source_commit, str):
        raise VisualDomainParityAuthorityError("source commit is invalid")
    _require_commit_ancestor(source_commit)
    source_bindings = authority.get("source_bindings")
    expected_paths = canonical_source_paths_v1()
    if (
        not isinstance(source_bindings, list)
        or [row.get("name") if isinstance(row, Mapping) else None for row in source_bindings]
        != list(expected_paths)
    ):
        raise VisualDomainParityAuthorityError("parity source closure/order changed")
    normalized_sources = []
    for row in source_bindings:
        if not isinstance(row, Mapping) or set(row) != {"name", "binding"}:
            raise VisualDomainParityAuthorityError("parity source row changed")
        name = str(row["name"])
        try:
            binding = pilot.require_binding(
                row["binding"], label=f"visual-domain parity source {name}"
            )
            _nofollow_regular(
                Path(str(binding["path"])),
                label=f"visual-domain parity source {name}",
            )
            runtime_kernel._binding_at_commit(  # noqa: SLF001
                binding,
                commit=source_commit,
                label=f"visual-domain parity source {name}",
            )
        except pilot.PilotContractError as exc:
            raise VisualDomainParityAuthorityError(str(exc)) from exc
        if Path(str(binding["path"])) != (REPO_ROOT / expected_paths[name]).resolve():
            raise VisualDomainParityAuthorityError(
                f"visual-domain parity source {name} path changed"
            )
        normalized_sources.append({"name": name, "binding": binding})
    try:
        pilot.validate_source_review(
            review,
            authority={
                "source_commit": source_commit,
                "source_bindings": normalized_sources,
            },
        )
    except pilot.PilotContractError as exc:
        raise VisualDomainParityAuthorityError(str(exc)) from exc
    attempt = authority.get("attempt")
    expected_attempt = {
        "id": normalized_plan["attempt_id"],
        "root": normalized_plan["output_root"],
        "maximum_attempts": 1,
        "must_be_absent": True,
        "root_creation_consumes_attempt": True,
        "reservation_records_consumed_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    caps = authority.get("caps")
    wall_seconds = caps.get("wall_seconds") if isinstance(caps, Mapping) else None
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not 0.0 < float(wall_seconds) <= MAX_WALL_SECONDS
        or attempt != expected_attempt
        or caps != _expected_caps(normalized_plan, wall_seconds=float(wall_seconds))
        or authority.get("runtime_bindings") != normalized_plan["runtime_bindings"]
        or authority.get("execution_contract")
        != normalized_plan["execution_contract"]
    ):
        raise VisualDomainParityAuthorityError("parity attempt/caps changed")
    supervisor = authority.get("external_supervisor")
    supervisor_binding = next(
        row["binding"] for row in normalized_sources if row["name"] == "external_supervisor"
    )
    if (
        not isinstance(supervisor, Mapping)
        or set(supervisor) != {"source_binding", "terminal_reviewer"}
        or supervisor.get("source_binding") != supervisor_binding
        or not isinstance(supervisor.get("terminal_reviewer"), str)
        or not supervisor["terminal_reviewer"].strip()
    ):
        raise VisualDomainParityAuthorityError("parity supervisor contract changed")
    disposition = authority.get("platform_gate_disposition")
    if (
        not isinstance(disposition, Mapping)
        or set(disposition)
        != {
            "platform_hard_gates_resolved",
            "scope",
            "outputs_eligible_for_training",
            "outputs_eligible_for_scientific_claim",
            "authorizes_this_exact_generation",
            "authorizes_promotion",
            "basis",
        }
        or disposition.get("platform_hard_gates_resolved") is not True
        or disposition.get("scope") != plan_builder.PURPOSE
        or disposition.get("outputs_eligible_for_training") is not False
        or disposition.get("outputs_eligible_for_scientific_claim") is not False
        or disposition.get("authorizes_this_exact_generation") is not True
        or disposition.get("authorizes_promotion") is not False
        or not isinstance(disposition.get("basis"), str)
        or not disposition["basis"].strip()
    ):
        raise VisualDomainParityAuthorityError("platform gate disposition changed")
    normalized = dict(authority)
    normalized["plan_binding"] = normalized_plan_binding
    normalized["review_binding"] = normalized_review_binding
    normalized["source_bindings"] = normalized_sources
    return normalized


def build_authority_v1(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    authorizer_identity: str,
    authorizer_basis: str,
    issued_at: str,
    terminal_reviewer: str,
    wall_seconds: float,
    platform_basis: str,
) -> dict[str, Any]:
    try:
        normalized_plan = plan_builder.validate_plan_v1(
            plan, require_fresh_output=True
        )
    except plan_builder.VisualDomainParityPlanError as exc:
        raise VisualDomainParityAuthorityError(str(exc)) from exc
    for value, label in (
        (authorizer_identity, "authorizer identity"),
        (authorizer_basis, "authorizer basis"),
        (terminal_reviewer, "terminal reviewer"),
        (platform_basis, "platform basis"),
    ):
        _require_nonempty(value, label=label)
    _require_iso8601(issued_at, label="issued_at")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not 0.0 < float(wall_seconds) <= MAX_WALL_SECONDS
    ):
        raise VisualDomainParityAuthorityError("wall_seconds is outside the hard cap")
    source_commit = review.get("reviewed_source_commit")
    source_bindings = review.get("reviewed_source_bindings")
    if not isinstance(source_commit, str) or not isinstance(source_bindings, list):
        raise VisualDomainParityAuthorityError("source review identity changed")
    supervisor_binding = next(
        (
            row["binding"]
            for row in source_bindings
            if isinstance(row, Mapping) and row.get("name") == "external_supervisor"
        ),
        None,
    )
    if supervisor_binding is None:
        raise VisualDomainParityAuthorityError("review lacks parity supervisor")
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": AUTHORITY_STATUS,
        "authority_granted_by_this_document": True,
        "scientific_claim_authorized": False,
        "authorizer": {
            "identity": authorizer_identity,
            "basis": authorizer_basis,
        },
        "issued_at": issued_at,
        "source_commit": source_commit,
        "review_binding": dict(review_binding),
        "plan_binding": dict(plan_binding),
        "source_bindings": list(source_bindings),
        "attempt": {
            "id": normalized_plan["attempt_id"],
            "root": normalized_plan["output_root"],
            "maximum_attempts": 1,
            "must_be_absent": True,
            "root_creation_consumes_attempt": True,
            "reservation_records_consumed_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
        "caps": _expected_caps(normalized_plan, wall_seconds=float(wall_seconds)),
        "runtime_bindings": dict(normalized_plan["runtime_bindings"]),
        "execution_contract": dict(normalized_plan["execution_contract"]),
        "network_access": False,
        "external_supervisor": {
            "source_binding": supervisor_binding,
            "terminal_reviewer": terminal_reviewer,
        },
        "platform_gate_disposition": {
            "platform_hard_gates_resolved": True,
            "scope": plan_builder.PURPOSE,
            "outputs_eligible_for_training": False,
            "outputs_eligible_for_scientific_claim": False,
            "authorizes_this_exact_generation": True,
            "authorizes_promotion": False,
            "basis": platform_basis,
        },
    }
    return validate_authority_v1(
        authority,
        plan=normalized_plan,
        plan_binding=plan_binding,
        review=review,
        review_binding=review_binding,
        require_fresh_output=True,
    )


def _read_cli_document(
    path: Path, digest: str, byte_count: int, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _SHA.fullmatch(digest) is None or byte_count <= 0:
        raise VisualDomainParityAuthorityError(f"{label} binding is malformed")
    try:
        value, binding = pilot.read_bound_json(
            path,
            expected_sha256=digest,
            expected_byte_count=byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise VisualDomainParityAuthorityError(str(exc)) from exc
    return dict(value), binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review-template")
    review.add_argument("--source-commit", required=True)
    review.add_argument("--output", required=True, type=Path)
    authority = subparsers.add_parser("authority")
    authority.add_argument("--plan", required=True, type=Path)
    authority.add_argument("--expected-plan-sha256", required=True)
    authority.add_argument("--expected-plan-byte-count", required=True, type=int)
    authority.add_argument("--review", required=True, type=Path)
    authority.add_argument("--expected-review-sha256", required=True)
    authority.add_argument("--expected-review-byte-count", required=True, type=int)
    authority.add_argument("--authorizer-identity", required=True)
    authority.add_argument("--authorizer-basis", required=True)
    authority.add_argument("--issued-at", required=True)
    authority.add_argument("--terminal-reviewer", required=True)
    authority.add_argument("--wall-seconds", required=True, type=float)
    authority.add_argument("--platform-basis", required=True)
    authority.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review-template":
        document = build_source_review_template_v1(source_commit=args.source_commit)
    else:
        plan, plan_binding = _read_cli_document(
            args.plan,
            args.expected_plan_sha256,
            args.expected_plan_byte_count,
            label="visual-domain parity plan",
        )
        review, review_binding = _read_cli_document(
            args.review,
            args.expected_review_sha256,
            args.expected_review_byte_count,
            label="visual-domain parity source review",
        )
        document = build_authority_v1(
            plan=plan,
            plan_binding=plan_binding,
            review=review,
            review_binding=review_binding,
            authorizer_identity=args.authorizer_identity,
            authorizer_basis=args.authorizer_basis,
            issued_at=args.issued_at,
            terminal_reviewer=args.terminal_reviewer,
            wall_seconds=args.wall_seconds,
            platform_basis=args.platform_basis,
        )
    binding = pilot.write_json_exclusive(args.output, document)
    print(json.dumps({"document": binding}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "VisualDomainParityAuthorityError",
    "build_authority_v1",
    "build_source_review_template_v1",
    "canonical_source_paths_v1",
    "committed_source_bindings_v1",
    "validate_authority_v1",
]
