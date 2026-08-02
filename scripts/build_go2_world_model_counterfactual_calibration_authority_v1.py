#!/usr/bin/env python3
"""Build exact review and authority metadata for one calibration attempt.

The ``review-template`` subcommand emits a deliberately non-passing template
over the exact committed source closure.  An independent reviewer must finish
it outside this helper.  The ``authority`` subcommand binds that passing review
and one exact committed calibration plan.  This script does not perform a
review, resolve platform gates, or infer authorization; those claims must be
supplied explicitly by the caller and remain visible in the result.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector  # noqa: E402


PURPOSE = "sizing_calibration_only"
TEXTURED_V03_PURPOSE = "sizing_calibration_textured_v03_v3"
AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_execution_authority_v2"
)
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_160_BRANCH_CALIBRATION_V2_SUCCESSOR"
TEXTURED_V03_AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_execution_authority_v3"
)
TEXTURED_V03_AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_EXACT_160_BRANCH_TEXTURED_V03_CALIBRATION_V3"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVIEW_ONLY_SOURCE_PATHS = {
    "calibration_authority_builder": (
        "scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py"
    ),
    "calibration_plan_builder": (
        "scripts/build_go2_world_model_counterfactual_calibration_plan_v1.py"
    ),
    "calibration_analyzer_test": (
        "lewm/tests/test_analyze_go2_world_model_counterfactual_calibration_v1.py"
    ),
    "calibration_authority_builder_test": (
        "lewm/tests/test_build_go2_world_model_counterfactual_calibration_authority_v1.py"
    ),
    "calibration_plan_builder_test": (
        "lewm/tests/test_build_go2_world_model_counterfactual_calibration_plan_v1.py"
    ),
    "calibration_supervisor_test": (
        "lewm/tests/test_run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    ),
    "counterfactual_contract_test": (
        "lewm/tests/test_go2_world_model_counterfactual_pilot_v1.py"
    ),
    "counterfactual_textured_v03_test": (
        "lewm/tests/test_go2_world_model_counterfactual_textured_v03.py"
    ),
    "predecessor_terminal_failure_result": str(
        collector.CALIBRATION_PREDECESSOR_FAILURE_RELATIVE
    ),
    "pilot_consumer_test": (
        "lewm/tests/test_go2_world_model_counterfactual_consumers_v1.py"
    ),
    "pilot_joiner_test": (
        "lewm/tests/test_join_go2_world_model_counterfactual_pilot_v1.py"
    ),
    "receipt_checker_test": (
        "lewm/tests/test_check_go2_world_model_counterfactual_pilot_v1.py"
    ),
}

TEXTURED_V03_PARITY_SOURCE_PATHS = {
    "visual_domain_parity_task_relevance_evaluator": (
        "scripts/evaluate_go2_world_model_visual_domain_parity_task_relevance_v1.py"
    ),
    "visual_domain_parity_task_relevance_evaluator_test": (
        "lewm/tests/test_evaluate_go2_world_model_visual_domain_parity_task_relevance_v1.py"
    ),
    "visual_domain_parity_plan_builder": (
        "scripts/build_go2_world_model_visual_domain_parity_plan_v1.py"
    ),
    "visual_domain_parity_authority_builder": (
        "scripts/build_go2_world_model_visual_domain_parity_authority_v1.py"
    ),
    "visual_domain_parity_scene_panel_builder": (
        "scripts/build_go2_world_model_bounded_branch_scene_panel_v1.py"
    ),
    "visual_domain_parity_supervisor": (
        "scripts/run_go2_world_model_visual_domain_parity_authorized_v1.py"
    ),
    "visual_domain_parity_calibration_analyzer": (
        "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
    ),
}


class CalibrationAuthorityBuildError(RuntimeError):
    """Raised before malformed metadata can mint a review or authority file."""


def canonical_runtime_source_paths_v1(
    *, textured_v03: bool = False
) -> dict[str, str]:
    """Return the exact, name-sorted calibration runtime source closure."""

    paths = {
        **collector.EXPECTED_SOURCE_PATHS,
        **collector.NON_SMOKE_SOURCE_PATHS,
        **REVIEW_ONLY_SOURCE_PATHS,
    }
    if textured_v03:
        paths.update(TEXTURED_V03_PARITY_SOURCE_PATHS)
        paths["predecessor_terminal_failure_result"] = str(
            collector.CALIBRATION_V2_FAILURE_RELATIVE
        )
    return dict(sorted(paths.items()))


def _require_iso8601(value: str, *, label: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationAuthorityBuildError(f"{label} is not ISO-8601") from exc


def committed_source_bindings_v1(
    source_commit: str, *, textured_v03: bool = False
) -> list[dict[str, Any]]:
    """Bind each runtime source and require exact equality at source_commit."""

    if _COMMIT.fullmatch(source_commit) is None:
        raise CalibrationAuthorityBuildError("source commit is invalid")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise CalibrationAuthorityBuildError(
            "source commit is not an ancestor of HEAD"
        ) from exc
    result: list[dict[str, Any]] = []
    for name, relative in canonical_runtime_source_paths_v1(
        textured_v03=textured_v03
    ).items():
        binding = pilot.file_binding(REPO_ROOT / relative)
        try:
            collector._binding_at_commit(  # noqa: SLF001
                binding,
                commit=source_commit,
                label=f"calibration source {name}",
            )
        except pilot.PilotContractError as exc:
            raise CalibrationAuthorityBuildError(str(exc)) from exc
        result.append({"name": name, "binding": binding})
    return result


def build_source_review_template_v1(
    *, source_commit: str, textured_v03: bool = False
) -> dict[str, Any]:
    """Emit a non-passing template; only an independent reviewer may finish it."""

    review = {
        "schema": pilot.SOURCE_REVIEW_SCHEMA,
        "status": "PENDING_INDEPENDENT_REVIEW",
        "authority_granted_by_this_document": False,
        "reviewed_source_commit": source_commit,
        "reviewed_source_bindings": (
            committed_source_bindings_v1(source_commit, textured_v03=True)
            if textured_v03
            else committed_source_bindings_v1(source_commit)
        ),
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
    return review


def build_authority_v1(
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    review: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    predecessor_failure_binding: Mapping[str, Any],
    authorizer_identity: str,
    authorizer_basis: str,
    issued_at: str,
    terminal_reviewer: str,
    wall_seconds: float,
    platform_basis: str,
) -> dict[str, Any]:
    """Bind one reviewed plan into the collector's exact authority schema."""

    normalized_plan = pilot.validate_plan(plan)
    textured_v03 = normalized_plan["purpose"] == TEXTURED_V03_PURPOSE
    if normalized_plan["purpose"] not in (PURPOSE, TEXTURED_V03_PURPOSE):
        raise CalibrationAuthorityBuildError(
            "authority builder requires a supported calibration plan"
        )
    for value, label in (
        (authorizer_identity, "authorizer identity"),
        (authorizer_basis, "authorizer basis"),
        (terminal_reviewer, "terminal reviewer"),
        (platform_basis, "platform basis"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise CalibrationAuthorityBuildError(f"{label} is empty")
    _require_iso8601(issued_at, label="issued_at")
    if wall_seconds <= 0.0:
        raise CalibrationAuthorityBuildError("wall_seconds must be positive")
    source_commit = review.get("reviewed_source_commit")
    source_bindings = review.get("reviewed_source_bindings")
    if (
        not isinstance(source_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
        or not isinstance(source_bindings, list)
    ):
        raise CalibrationAuthorityBuildError("source review identity is invalid")
    expected_names = list(canonical_runtime_source_paths_v1(
        textured_v03=textured_v03
    ))
    if [row.get("name") if isinstance(row, Mapping) else None for row in source_bindings] != expected_names:
        raise CalibrationAuthorityBuildError("source review closure/order changed")
    expected_paths = canonical_runtime_source_paths_v1(
        textured_v03=textured_v03
    )
    for row in source_bindings:
        expected_path = (REPO_ROOT / expected_paths[row["name"]]).resolve()
        if Path(str(row["binding"]["path"])).resolve() != expected_path:
            raise CalibrationAuthorityBuildError(
                f"source review path changed for {row['name']}"
            )
    pilot.validate_source_review(
        review,
        authority={
            "source_commit": source_commit,
            "source_bindings": source_bindings,
        },
    )
    supervisor_binding = next(
        row["binding"] for row in source_bindings if row["name"] == "external_supervisor"
    )
    caps = {
        **collector._expected_authority_caps(normalized_plan),  # noqa: SLF001
        "wall_seconds": float(wall_seconds),
    }
    authority = {
        "schema": (
            TEXTURED_V03_AUTHORITY_SCHEMA if textured_v03 else AUTHORITY_SCHEMA
        ),
        "status": (
            TEXTURED_V03_AUTHORITY_STATUS if textured_v03 else AUTHORITY_STATUS
        ),
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
        "predecessor_failure_binding": dict(predecessor_failure_binding),
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
        "caps": caps,
        "runtime_bindings": dict(normalized_plan["runtime_bindings"]),
        "execution": dict(normalized_plan["execution_contract"]),
        "network_access": False,
        "external_supervisor": {
            "source_binding": supervisor_binding,
            "terminal_reviewer": terminal_reviewer,
        },
        "platform_gate_disposition": {
            "platform_hard_gates_resolved": True,
            "scope": normalized_plan["purpose"],
            "outputs_eligible_for_training_after_receipt_join": False,
            "outputs_eligible_for_scientific_claim": False,
            "authorizes_this_exact_generation": True,
            "authorizes_promotion": False,
            "basis": platform_basis,
        },
    }
    try:
        return collector._validate_non_smoke_authority(  # noqa: SLF001
            authority,
            plan=normalized_plan,
            plan_binding=plan_binding,
        )
    except pilot.PilotContractError as exc:
        raise CalibrationAuthorityBuildError(str(exc)) from exc


def _read_bound(path: Path, digest: str, byte_count: int, *, label: str):
    if _SHA256.fullmatch(digest) is None or byte_count <= 0:
        raise CalibrationAuthorityBuildError(f"{label} caller binding is malformed")
    try:
        return pilot.read_bound_json(
            path,
            expected_sha256=digest,
            expected_byte_count=byte_count,
            label=label,
        )
    except pilot.PilotContractError as exc:
        raise CalibrationAuthorityBuildError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review-template")
    review.add_argument("--source-commit", required=True)
    review.add_argument("--textured-v03", action="store_true")
    review.add_argument("--output", required=True, type=Path)

    authority = subparsers.add_parser("authority")
    authority.add_argument("--plan", required=True, type=Path)
    authority.add_argument("--expected-plan-sha256", required=True)
    authority.add_argument("--expected-plan-byte-count", required=True, type=int)
    authority.add_argument("--review", required=True, type=Path)
    authority.add_argument("--expected-review-sha256", required=True)
    authority.add_argument("--expected-review-byte-count", required=True, type=int)
    authority.add_argument("--predecessor-failure", required=True, type=Path)
    authority.add_argument("--expected-predecessor-failure-sha256", required=True)
    authority.add_argument(
        "--expected-predecessor-failure-byte-count", required=True, type=int
    )
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
        document = build_source_review_template_v1(
            source_commit=args.source_commit,
            textured_v03=args.textured_v03,
        )
    else:
        plan, plan_binding = _read_bound(
            args.plan,
            args.expected_plan_sha256,
            args.expected_plan_byte_count,
            label="counterfactual calibration plan",
        )
        review, review_binding = _read_bound(
            args.review,
            args.expected_review_sha256,
            args.expected_review_byte_count,
            label="counterfactual calibration source review",
        )
        _predecessor_failure, predecessor_failure_binding = _read_bound(
            args.predecessor_failure,
            args.expected_predecessor_failure_sha256,
            args.expected_predecessor_failure_byte_count,
            label="counterfactual calibration predecessor terminal-failure result",
        )
        document = build_authority_v1(
            plan=plan,
            plan_binding=plan_binding,
            review=review,
            review_binding=review_binding,
            predecessor_failure_binding=predecessor_failure_binding,
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
    "CalibrationAuthorityBuildError",
    "build_authority_v1",
    "build_source_review_template_v1",
    "canonical_runtime_source_paths_v1",
    "committed_source_bindings_v1",
]
