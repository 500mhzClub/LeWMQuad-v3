#!/usr/bin/env python3
"""Fail-closed finalizer for the V2 full-panel metric verification."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)


def _retained() -> Any:
    from scripts import (
        finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    retained.policy = policy
    return retained


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _retained().parse_args(argv)


def _validate_metric_receipt(
    authority: policy.VerifiedAuthorityV2,
    bundle: Mapping[str, Any],
    metric_bound: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    return _retained()._validate_metric_receipt(authority, bundle, metric_bound)


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = policy.verify_authority(
        args.source_review,
        args.source_review_sha256,
        purpose="finalization",
        require_unclaimed_output=False,
    )
    policy.transition_authority(
        authority,
        purpose="finalization",
        target_path=policy.CANONICAL_GATE_PATH,
        from_states=("issued",),
        to_state="active",
    )
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as verifier,
    )

    bundle = verifier._validate_attempt_bundle(authority, args)
    receipt, receipt_raw, numeric = _validate_metric_receipt(
        authority,
        bundle,
        args.metric_verification,
    )
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as frozen

    threshold_contract = {
        str(size): asdict(frozen.FIT_THRESHOLDS[size])
        for size in frozen.LADDER_FIT_SIZES
    }
    passes = bool(numeric["passes"])
    status = "passed" if passes else "terminal_numeric_failure"
    core = {
        "schema": policy.GATE_SCHEMA,
        "status": status,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "seed": 20260710,
        "fit_size": 5,
        "authority_bindings": policy.AUTHORITY_BINDINGS,
        "source_review": policy.source_review_binding(authority),
        "artifacts": {
            "reservation": bundle["reservation_binding"],
            "result": bundle["result_binding"],
            "checkpoint": bundle["checkpoint_binding"],
            "completion": bundle["completion_binding"],
            "metric_verification": policy.artifact_binding(
                "metric_verifications/seed_20260710_n5.json",
                receipt_raw,
                content_sha256=receipt["content_sha256"],
            ),
        },
        "result_content_sha256": bundle["result"]["content_sha256"],
        "metric_verification_content_sha256": receipt["content_sha256"],
        "threshold_contract_sha256": policy.canonical_json_sha256(
            threshold_contract
        ),
        "numeric_gate": numeric,
        "passes": passes,
        "failure_count": int(numeric["failure_count"]),
        "failed_checks": numeric["failed_checks"],
        "licenses": {
            "checkpoint_use_authorized": False,
            "retry_authorized": False,
            "later_rung_design_review_authorized": passes,
            "n16_execution_authorized": False,
            "second_seed_authorized": False,
            "v5_training_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "selection_authorized": False,
            "calibration_change_authorized": False,
            "runtime_authorized": False,
            "hardware_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    gate = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    policy.write_exclusive(policy.CANONICAL_GATE_PATH, gate)
    return gate


def _isolated_child(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        return _isolated_child(raw_argv)
    gate = run(parse_args(raw_argv))
    print(policy.canonical_json_bytes(gate).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
