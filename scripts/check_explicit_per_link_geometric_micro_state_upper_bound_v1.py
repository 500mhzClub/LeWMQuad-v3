#!/usr/bin/env python3
"""Reproduce aggregate geometry/viability metrics without simulation replay."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from scripts import evaluate_explicit_per_link_geometric_micro_state_upper_bound_v1 as E
from scripts import materialize_explicit_per_link_geometric_micro_state_upper_bound_v1 as M


def same(left, right) -> bool:
    if left is None or right is None: return left is right
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10)
    return left == right


def main() -> int:
    result = json.loads(E.RESULT.read_text()); _index, _corpus, states = E.load()
    heldout = [state for state in states if state["role"] == "heldout"]
    checks = {"source_commit": result["source_commit"] == M.SOURCE_COMMIT,
              "row_ledger_sha256": E.sha(Path(result["row_level_evidence"]["path"])) == result["row_level_evidence"]["sha256"],
              "row_ledger_count": sum(1 for _ in Path(result["row_level_evidence"]["path"]).open()) == 29470}
    for condition in M.CONDITIONS:
        threshold = result["thresholds_m"][condition]
        contacts = E.contact_summary(heldout, condition, threshold)
        decision = E.decision_metrics(heldout, condition, threshold, "unique_deployable")
        stored = result["heldout"][condition]
        for level in ("current", "successor", "combined"):
            for field in ("auc", "average_precision", "recall", "fnr", "negative_retention", "tp", "fn", "fp", "tn"):
                checks[f"{condition}_{level}_{field}"] = same(contacts[level][field], stored["contacts"][level][field])
        for field in ("states_retaining_admitted_action", "selected_immediate_contacts", "selected_oracle_nonviable_successors",
                      "false_abstentions", "correct_abstentions", "selected_h3_route_progress_m", "oracle_progress_fraction",
                      "normalized_regret", "best_admissible_top3"):
            checks[f"{condition}_decision_{field}"] = same(decision[field], stored["decisions"]["unique_deployable"][field])
        for field in ("mae", "spearman", "exact_count_accuracy", "zero_vs_nonzero_accuracy", "false_zero_rate", "false_nonzero_rate"):
            checks[f"{condition}_count_{field}"] = same(decision["safe_action_count"][field], stored["decisions"]["unique_deployable"]["safe_action_count"][field])
    value = {"schema": "explicit_per_link_geometric_micro_state_upper_bound_v1_checker",
             "pass": all(checks.values()), "checks": checks, "model_inference": 0, "simulation_replay": 0}
    value["content_digest"] = E.METRICS.digest(value); E.atomic_json(E.OUT / "result_checker.json", value)
    print(json.dumps({"pass": value["pass"], "checks": len(checks), "failures": [key for key, passed in checks.items() if not passed]}, indent=2))
    return 0 if value["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
