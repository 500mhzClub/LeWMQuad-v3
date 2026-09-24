#!/usr/bin/env python3
"""Run the registered rank-regret metric-validity study V1.

Part B1 correlates bound geometric first-action regret against bound closed-loop
progress across the completed Aug-4 policies.  Part B2 computes both one-step
metrics on the immutable V3 matched evaluation role, same-state, and correlates
them.

This runner performs no simulation, no rendering, no training, and no encoder
execution.  It reads the matched panel's state receipts and the completed
observability-ceiling assay result, and nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_observability_ceiling_assay_v1 as assay  # noqa: E402
from lewm.benchmarks import go2_rank_regret_metric_validity_v1 as validity  # noqa: E402
from scripts import run_go2_observability_ceiling_assay_v1 as ceiling_runner  # noqa: E402


STEM = "go2_rank_regret_metric_validity_v1"
ATTEMPT_ID = f"{STEM}_attempt_v1"
ATTEMPT_ROOT = REPO_ROOT / ".generated" / "dev" / STEM / "attempt_v1"
RESULT_PATH = ATTEMPT_ROOT / "result.json"
TERMINAL_PATH = ATTEMPT_ROOT / "terminal.json"

PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / f"lewm_{STEM}_preregistration_2026-08-05.md"
)

RULE_ARMS = ("geometric_endpoint", "bearing", "hold")


class MetricValidityRunnerError(RuntimeError):
    """Raised when the runner contract is violated."""


def file_binding_v1(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_json_exclusive_v1(path: Path, value: Mapping[str, Any]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)
    return file_binding_v1(path)


def _selection_from_scores(scores: np.ndarray) -> list[int]:
    return [int(np.argmin(row)) for row in scores]


def execute_study_v1(ceiling_result_path: Path) -> dict[str, Any]:
    started = time.time()

    collection_binding = file_binding_v1(ceiling_runner.COLLECTION_RESULT_PATH)
    if collection_binding["sha256"] != ceiling_runner.COLLECTION_RESULT_SHA256:
        raise MetricValidityRunnerError(
            "consumed collection result does not rehash to its registered SHA-256"
        )

    ledger = ceiling_runner.AccessLedgerV1()
    eval_groups, _receipts = ceiling_runner.load_role_v1("eval", ledger=ledger)

    ceiling_result = json.loads(ceiling_result_path.read_text())
    if ceiling_result.get("schema") != assay.SCHEMA:
        raise MetricValidityRunnerError("ceiling assay result schema changed")
    per_state = ceiling_result.get("per_state_selection")
    if not isinstance(per_state, Mapping):
        raise MetricValidityRunnerError("ceiling assay per-state selection is missing")

    panel_state_ids = [str(group.state_id) for group in eval_groups]

    arms: dict[str, dict[str, Any]] = {}

    # --- Arms carried over from the observability-ceiling assay ---
    for arm, rows in per_state.items():
        if not rows:
            continue
        if [str(row["state_id"]) for row in rows] != panel_state_ids:
            raise MetricValidityRunnerError(f"arm {arm} state order does not match the panel")
        selected = [row["selected_action_id"] for row in rows]
        if any(not isinstance(value, int) for value in selected):
            # Random expectation selects no realized action; it has no G.
            arms[arm] = {
                "source": "observability_ceiling_assay",
                "normalized_rank_regret": float(
                    np.mean([float(row["normalized_rank_regret"]) for row in rows])
                ),
                "geometric_regret_m": None,
                "note": "no realized action; excluded from the B2 correlation",
            }
            continue
        geometric = validity.geometric_regret_v1(eval_groups, selected)
        arms[arm] = {
            "source": "observability_ceiling_assay",
            "normalized_rank_regret": float(
                np.mean([float(row["normalized_rank_regret"]) for row in rows])
            ),
            "geometric_regret_m": geometric["geometric_regret_m"],
        }

    # --- Frozen rule-based arms computed here ---
    for rule in RULE_ARMS:
        scores = validity.rule_based_scores_v1(eval_groups, rule=rule)
        selected = _selection_from_scores(scores)
        report = assay.arm_report_v1(eval_groups, scores, policy="argmin")
        geometric = validity.geometric_regret_v1(eval_groups, selected)
        arms[rule] = {
            "source": "frozen_rule",
            "normalized_rank_regret": report["summary"]["normalized_rank_regret"],
            "geometric_regret_m": geometric["geometric_regret_m"],
        }

    # --- Anchors ---
    oracle_report = assay.arm_report_v1(eval_groups, None, policy="oracle")
    oracle_selected = [int(row["selected_action_id"]) for row in oracle_report["state_results"]]
    oracle_geometric = validity.geometric_regret_v1(eval_groups, oracle_selected)
    arms["physical_oracle"] = {
        "source": "anchor",
        "normalized_rank_regret": oracle_report["summary"]["normalized_rank_regret"],
        "geometric_regret_m": oracle_geometric["geometric_regret_m"],
    }

    random_report = assay.arm_report_v1(eval_groups, None, policy="random")
    arms["random_expected"] = {
        "source": "anchor",
        "normalized_rank_regret": random_report["summary"]["normalized_rank_regret"],
        "geometric_regret_m": None,
        "note": "no realized action; excluded from the B2 correlation",
    }

    # --- Registered integrity gates ---
    if arms["physical_oracle"]["normalized_rank_regret"] != 0.0:
        raise MetricValidityRunnerError("physical oracle R is not exactly zero")
    if abs(float(arms["random_expected"]["normalized_rank_regret"]) - 0.4765170304232804) > 1e-12:
        raise MetricValidityRunnerError("random expectation R does not reproduce the registered value")

    # --- Part B1 ---
    part_b1 = validity.part_b1_v1()

    # --- Part B2 ---
    paired = {
        name: value
        for name, value in arms.items()
        if value.get("geometric_regret_m") is not None
    }
    names = sorted(paired)
    rank_values = [float(paired[name]["normalized_rank_regret"]) for name in names]
    geometric_values = [float(paired[name]["geometric_regret_m"]) for name in names]
    part_b2 = {
        "description": (
            "do normalized rank regret and geometric first-action regret agree "
            "on the identical 128 matched states"
        ),
        "arms": names,
        "normalized_rank_regret": rank_values,
        "geometric_regret_m": geometric_values,
        "correlation": validity.bootstrap_correlation_v1(rank_values, geometric_values),
        "divergence": validity.divergence_summary_v1(
            eval_groups, oracle_report["state_results"], oracle_geometric
        ),
    }

    # --- Rank-order disagreement reporting (preregistration section 7.2) ---
    rank_order = {
        name: int(position)
        for position, name in enumerate(sorted(names, key=lambda item: float(paired[item]["normalized_rank_regret"])))
    }
    geometric_order = {
        name: int(position)
        for position, name in enumerate(sorted(names, key=lambda item: float(paired[item]["geometric_regret_m"])))
    }
    disagreements = {
        name: {"rank_position": rank_order[name], "geometric_position": geometric_order[name]}
        for name in names
        if abs(rank_order[name] - geometric_order[name]) >= 2
    }

    decision = validity.decide_v1(part_b1, part_b2)

    result: dict[str, Any] = {
        "schema": validity.SCHEMA,
        "attempt_id": ATTEMPT_ID,
        "development_only": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "preregistration_binding": file_binding_v1(PREREGISTRATION_PATH),
        "collection_binding": collection_binding,
        "ceiling_assay_binding": file_binding_v1(ceiling_result_path),
        "arms": arms,
        "part_b1": part_b1,
        "part_b2": part_b2,
        "rank_order_disagreements": disagreements,
        "declared_limits": [
            "B1 has seven scorers and B2 has nine arms; both are powered to "
            "distinguish a strong monotone relationship from none, not to "
            "estimate a correlation precisely",
            "B1's geometric regret is measured on each policy's own diverged "
            "states; B2 is same-state by construction",
            "B1 and B2 use different scene panels",
        ],
        "decision": decision,
        "wall_seconds": time.time() - started,
    }
    result["identity_sha256"] = validity.result_identity_v1(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ceiling-result",
        type=Path,
        default=REPO_ROOT
        / ".generated/dev/go2_observability_ceiling_assay_v1/attempt_v2/result.json",
        help="completed observability-ceiling assay result to consume",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if RESULT_PATH.exists():
        raise MetricValidityRunnerError(
            "an immutable result already exists; this attempt is not resumable"
        )
    try:
        result = execute_study_v1(arguments.ceiling_result)
    except Exception as error:  # noqa: BLE001
        ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
        if not TERMINAL_PATH.exists():
            write_json_exclusive_v1(
                TERMINAL_PATH,
                {
                    "schema": f"{validity.SCHEMA}_terminal",
                    "attempt_id": ATTEMPT_ID,
                    "status": "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION",
                    "error": f"{type(error).__name__}: {error}",
                    "citable_as_scientific_evidence": False,
                },
            )
        raise
    binding = write_json_exclusive_v1(RESULT_PATH, result)
    write_json_exclusive_v1(
        TERMINAL_PATH,
        {
            "schema": f"{validity.SCHEMA}_terminal",
            "attempt_id": ATTEMPT_ID,
            "status": result["decision"]["terminal"],
            "rho_closed_loop_link": result["decision"]["rho_closed_loop_link"],
            "rho_rank_link": result["decision"]["rho_rank_link"],
            "result_binding": binding,
            "development_only": True,
            "citable_as_scientific_evidence": False,
            "authorizes_retry_or_resume": False,
        },
    )
    print(json.dumps(result["decision"], indent=2))
    print(json.dumps(result["part_b2"]["correlation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
