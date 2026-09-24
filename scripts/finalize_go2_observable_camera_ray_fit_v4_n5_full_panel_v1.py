#!/usr/bin/env python3
"""Fail-closed finalizer for the V4 N5 full-panel metric verification."""
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
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--reservation", required=True, help="PATH:SHA256")
    parser.add_argument("--result", required=True, help="PATH:SHA256")
    parser.add_argument("--checkpoint", required=True, help="PATH:SHA256")
    parser.add_argument("--completion", required=True, help="PATH:SHA256")
    parser.add_argument("--metric-verification", required=True, help="PATH:SHA256")
    return parser.parse_args(argv)


def _validate_metric_receipt(
    authority: policy.VerifiedAuthority,
    bundle: Mapping[str, Any],
    metric_bound: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    metric_path, metric_sha = policy.parse_bound_path(metric_bound)
    if metric_path != policy.CANONICAL_METRIC_RECEIPT_PATH.resolve(strict=True):
        raise PermissionError("N5 full-panel metric receipt path is not canonical")
    receipt, raw = policy.load_hashed_json(
        metric_path,
        metric_sha,
        name="N5 full-panel metric verification",
    )
    expected_fields = {
        "schema",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "dataset_role",
        "seed",
        "fit_size",
        "authority_bindings",
        "source_review",
        "artifacts",
        "result_content_sha256",
        "target_partition",
        "target_partition_signature",
        "target_partition_signature_sha256",
        "recomputed_evaluation",
        "recomputed_evaluation_sha256",
        "numeric_gate",
        "verification",
        "resource",
        "access_ledger",
        "licenses",
        "content_sha256",
    }
    if set(receipt) != expected_fields or receipt.get("schema") != policy.METRIC_RECEIPT_SCHEMA:
        raise ValueError("N5 full-panel metric receipt schema changed")
    result = bundle["result"]
    review = policy.source_review_binding(authority)
    expected_artifacts = {
        "reservation": bundle["reservation_binding"],
        "result": bundle["result_binding"],
        "checkpoint": bundle["checkpoint_binding"],
        "completion": bundle["completion_binding"],
    }
    expected_verification = {
        "checkpoint_loaded": True,
        "checkpoint_state_manifest_rehashed": True,
        "checkpoint_final_update_binding_validated": True,
        "fresh_model_loaded_for_inference": True,
        "selected_train_targets_loaded": True,
        "selected_matched_rgb_loaded": True,
        "wrong_rgb_mapping_rerun": True,
        "evaluation_losses_recomputed": True,
        "evaluation_loss_arithmetic_validated": True,
        "all_confusions_recomputed": True,
        "depth_quantiles_and_sorted_commitments_recomputed": True,
        "raster_nll_recomputed": True,
        "family_metrics_recomputed": True,
        "frozen_thresholds_recomputed": True,
        "result_metrics_reused": False,
        "metric_repair_applied": False,
        "threshold_weakened": False,
    }
    expected_access = {
        "selected_rgb_count": 5,
        "selected_rgb_hash_opens": 5,
        "selected_rgb_decodes": 5,
        "checkpoint_opens": 1,
        "heldout_opens": 0,
        "g2_opens": 0,
        "selection_opens": 0,
        "calibration_opens": 0,
        "runtime_opens": 0,
        "hardware_opens": 0,
        "production_opens": 0,
        "gpu1_uses": 0,
    }
    expected_licenses = {
        "checkpoint_use_authorized_for_metric_verification_only": True,
        "development_checkpoint_use_authorized": False,
        "new_model_output_authorized": False,
        "retry_authorized": False,
        "n16_execution_authorized": False,
        "second_seed_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    evaluation = receipt.get("recomputed_evaluation")
    policy.validate_evaluation_structure(evaluation)
    if (
        receipt.get("authoritative") is not False
        or receipt.get("aggregation_eligible") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("dataset_role") != "train"
        or receipt.get("seed") != 20260710
        or receipt.get("fit_size") != 5
        or receipt.get("authority_bindings") != policy.AUTHORITY_BINDINGS
        or receipt.get("source_review") != review
        or receipt.get("artifacts") != expected_artifacts
        or receipt.get("result_content_sha256") != result["content_sha256"]
        or receipt.get("target_partition") != result["target_partition"]
        or evaluation != result["evaluation"]
        or receipt.get("recomputed_evaluation_sha256")
        != policy.canonical_json_sha256(evaluation)
        or receipt.get("verification") != expected_verification
        or receipt.get("access_ledger") != expected_access
        or receipt.get("licenses") != expected_licenses
    ):
        raise PermissionError("N5 full-panel metric receipt provenance/scope changed")

    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as frozen

    matched, wrong, signature = frozen._validated_metric_evaluation(evaluation, fit_size=5)
    numeric = frozen._gate_stage(
        {"fit_size": 5, "matched": matched, "wrong": wrong}
    )
    if (
        receipt.get("target_partition_signature") != signature
        or receipt.get("target_partition_signature_sha256")
        != policy.canonical_json_sha256(signature)
        or receipt.get("numeric_gate") != numeric
        or receipt.get("resource") != result["resource"]
    ):
        raise ValueError("N5 full-panel frozen metric decision changed")
    return receipt, raw, numeric


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = policy.verify_authority(
        args.source_review,
        args.source_review_sha256,
        require_unclaimed_output=False,
    )
    from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier

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
