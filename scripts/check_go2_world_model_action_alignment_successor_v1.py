#!/usr/bin/env python3
"""Receipt-only checker for the matched action-alignment successor."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_world_model_action_alignment_successor_v1 as metrics,
)
from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)
from scripts import (  # noqa: E402
    execute_go2_world_model_action_alignment_successor_v1 as worker,
)


CHECK_SCHEMA = "lewm_go2_world_model_action_alignment_successor_v1_receipt_check_v1"


class AlignmentCheckError(RuntimeError):
    """A terminal result or its bound metric bundle failed verification."""


def _require_passing_baseline_anchor_audit(value: Any) -> None:
    if (
        type(value) is not dict
        or value.get("exact_within_1e_15") is not True
        or type(value.get("checks")) is not dict
        or not value["checks"]
        or not all(item is True for item in value["checks"].values())
    ):
        raise AlignmentCheckError("concurrent baseline did not reproduce V3 anchors")


def _read_result(path: Path, *, digest: str, count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = worker.file_binding(path)
    expected = {
        "path": str(path.resolve(strict=True)),
        "file_sha256": digest,
        "byte_count": count,
    }
    if binding != expected:
        raise AlignmentCheckError("caller-bound result identity changed")
    try:
        raw = worker.custody._read_absolute_regular_once(binding, label="worker result")
    except Exception as error:
        raise AlignmentCheckError("could not read the bound worker result") from error
    result = worker.strict_json_bytes(raw)
    if type(result) is not dict:
        raise AlignmentCheckError("worker result must be a JSON object")
    return result, binding


def _load_metric_bundle(binding: Mapping[str, Any]) -> dict[str, Any]:
    try:
        raw = worker.custody._read_absolute_regular_once(binding, label="metric bundle")
        bundle = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise AlignmentCheckError("could not load bound metric bundle") from error
    if type(bundle) is not dict:
        raise AlignmentCheckError("metric bundle is not a dictionary")
    required = {
        "schema", "status", "authority_binding", "reservation_binding",
        "validation_row_indices", "baseline_candidate_energy",
        "baseline_factual_energy", "baseline_persistence_energy",
        "baseline_wrong_history_energy", "alignment_candidate_energy",
        "alignment_factual_energy", "alignment_persistence_energy",
        "alignment_wrong_history_energy", "alignment_rank_ratio_tail",
        "training_factual_energy", "contract_checks", "train_fit_checks",
    }
    if set(bundle) != required:
        raise AlignmentCheckError("metric bundle fields changed")
    if bundle["schema"] != worker.METRIC_BUNDLE_SCHEMA or bundle["status"] != "COMPLETE":
        raise AlignmentCheckError("metric bundle schema/status changed")
    indices = bundle["validation_row_indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or not torch.equal(indices, torch.arange(worker.EXPECTED_VALIDATION_ROWS))
    ):
        raise AlignmentCheckError("metric validation indices changed")
    for name in ("baseline", "alignment"):
        candidate = bundle[f"{name}_candidate_energy"]
        if (
            not isinstance(candidate, torch.Tensor)
            or tuple(candidate.shape) != (worker.EXPECTED_VALIDATION_ROWS, worker.ACTION_COUNT)
            or not bool(torch.isfinite(candidate).all())
            or bool((candidate < 0.0).any())
        ):
            raise AlignmentCheckError(f"{name} candidate vector changed")
        for suffix in ("factual_energy", "persistence_energy", "wrong_history_energy"):
            value = bundle[f"{name}_{suffix}"]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (worker.EXPECTED_VALIDATION_ROWS,)
                or not bool(torch.isfinite(value).all())
                or bool((value <= 0.0).any())
            ):
                raise AlignmentCheckError(f"{name} {suffix} changed")
        train = bundle["training_factual_energy"][name]
        if (
            not isinstance(train, torch.Tensor)
            or tuple(train.shape) != (worker.EXPECTED_TRAIN_ROWS,)
            or not bool(torch.isfinite(train).all())
            or bool((train <= 0.0).any())
        ):
            raise AlignmentCheckError(f"{name} training vector changed")
    ranks = bundle["alignment_rank_ratio_tail"]
    if (
        not isinstance(ranks, torch.Tensor)
        or ranks.dtype != torch.float64
        or tuple(ranks.shape) != (3,)
        or not bool(torch.isfinite(ranks).all())
        or bool((ranks < 0.0).any())
    ):
        raise AlignmentCheckError("alignment rank tail changed")
    return bundle


def check(
    *, manifest: Path, expected_sha256: str, expected_byte_count: int, output: Path
) -> dict[str, Any]:
    worker.validate_exact_child_environment()
    if output.resolve(strict=False) != (worker.ATTEMPT_ROOT / "receipt_check.json"):
        raise AlignmentCheckError("checker output path changed")
    if output.exists() or output.is_symlink():
        raise AlignmentCheckError("checker output already exists")
    worker.exact_root_inventory(worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER)
    result, result_binding = _read_result(
        manifest.resolve(strict=True), digest=expected_sha256, count=expected_byte_count
    )
    if (
        result.get("schema") != worker.RESULT_SCHEMA
        or result.get("status") != "COMPLETE_PENDING_TERMINAL_REVIEW"
        or result.get("development_evidence_complete") is not True
        or result.get("citable_as_original_factual_learnability_claim") is not False
        or result.get("claim_boundary") != worker.CLAIM_BOUNDARY
    ):
        raise AlignmentCheckError("result envelope changed")
    authority_binding = result.get("authority_binding")
    if type(authority_binding) is not dict:
        raise AlignmentCheckError("result authority binding is absent")
    authority, observed_authority_binding = worker.load_and_validate_authority(
        worker.AUTHORITY_PATH,
        expected_sha256=authority_binding["file_sha256"],
        expected_byte_count=authority_binding["byte_count"],
    )
    if observed_authority_binding != authority_binding:
        raise AlignmentCheckError("result authority binding changed")
    reservation_binding = result.get("reservation_binding")
    if (
        type(reservation_binding) is not dict
        or worker.file_binding(worker.ATTEMPT_ROOT / "reservation.json")
        != reservation_binding
    ):
        raise AlignmentCheckError("result reservation binding changed")
    if (
        result.get("source_commit") != authority["source_commit"]
        or result.get("review_commit") != authority["review_commit"]
        or result.get("execution_head") != authority["execution_head"]
        or result.get("input_bindings") != worker.EXPECTED_INPUT_BINDINGS
        or result.get("evidence_bindings") != worker.EXPECTED_EVIDENCE_BINDINGS
    ):
        raise AlignmentCheckError("result provenance changed")

    metric_binding = result.get("metric_bundle_binding")
    if type(metric_binding) is not dict or worker.file_binding(Path(metric_binding["path"])) != metric_binding:
        raise AlignmentCheckError("metric bundle binding changed")
    bundle = _load_metric_bundle(metric_binding)
    if (
        bundle["authority_binding"] != authority_binding
        or bundle["reservation_binding"] != reservation_binding
    ):
        raise AlignmentCheckError("metric bundle provenance changed")
    validation_rows, validation_audit = h6.load_bound_index(REPO_ROOT, role="val")
    if (
        len(validation_rows) != worker.EXPECTED_VALIDATION_ROWS
        or validation_audit["file_sha256"]
        != worker.EXPECTED_INPUT_BINDINGS["validation_index"]["file_sha256"]
    ):
        raise AlignmentCheckError("checker validation metadata changed")
    rank_values = bundle["alignment_rank_ratio_tail"].tolist()
    recomputed = metrics.decide_alignment_successor(
        baseline_candidate_energy=bundle["baseline_candidate_energy"],
        baseline_factual_energy=bundle["baseline_factual_energy"],
        baseline_persistence_energy=bundle["baseline_persistence_energy"],
        baseline_wrong_history_energy=bundle["baseline_wrong_history_energy"],
        treatment_candidate_energy=bundle["alignment_candidate_energy"],
        treatment_factual_energy=bundle["alignment_factual_energy"],
        treatment_persistence_energy=bundle["alignment_persistence_energy"],
        treatment_wrong_history_energy=bundle["alignment_wrong_history_energy"],
        validation_rows=validation_rows,
        treatment_rank_ratio_by_update={
            update: rank_values[index]
            for index, update in enumerate(metrics.TAIL_UPDATES)
        },
        contract_checks=bundle["contract_checks"],
        train_fit_checks=bundle["train_fit_checks"],
    )
    if recomputed != result.get("decision"):
        raise AlignmentCheckError("independently recomputed decision differs")
    baseline_audit = worker._baseline_anchor_audit(recomputed)
    if baseline_audit != result.get("baseline_v3_reproduction"):
        raise AlignmentCheckError("baseline anchor audit differs")
    _require_passing_baseline_anchor_audit(baseline_audit)
    if bundle["contract_checks"].get("baseline_v3_reproduction_exact") is not True:
        raise AlignmentCheckError("baseline reproduction is absent from contract checks")
    train_means = {
        name: float(bundle["training_factual_energy"][name].mean())
        for name in worker.ARM_NAMES
    }
    if train_means != result.get("train_fit", {}).get("full_train_factual_mean_energy"):
        raise AlignmentCheckError("full-train means differ")
    snapshots = result.get("snapshot_bindings")
    if type(snapshots) is not dict or set(snapshots) != set(worker.ARM_NAMES):
        raise AlignmentCheckError("snapshot inventory changed")
    for name, binding in snapshots.items():
        expected_path = worker.ATTEMPT_ROOT / f"{name}_update_000700.pt"
        if type(binding) is not dict or worker.file_binding(expected_path) != binding:
            raise AlignmentCheckError(f"{name} snapshot binding changed")
    forbidden = result.get("forbidden_access")
    if type(forbidden) is not dict or any(forbidden.values()):
        raise AlignmentCheckError("forbidden access was reported")
    accounting = result.get("accounting")
    if (
        type(accounting) is not dict
        or accounting.get("training_updates") != worker.TRAINING_UPDATES
        or accounting.get("total_optimizer_steps") != 1_400
        or accounting.get("rgb_open_count") != 0
        or accounting.get("data_generation_count") != 0
        or accounting.get("network_access_count") != 0
    ):
        raise AlignmentCheckError("execution accounting changed")
    receipt = {
        "schema": CHECK_SCHEMA,
        "status": "PASS",
        "result_binding": result_binding,
        "authority_binding": authority_binding,
        "reservation_binding": reservation_binding,
        "metric_bundle_binding": metric_binding,
        "decision_status": recomputed["status"],
        "decision_exactly_recomputed": True,
        "paired_bootstrap_exactly_recomputed": True,
        "baseline_anchor_audit_exactly_recomputed": True,
        "snapshot_bindings_verified_without_loading": True,
        "validation_index_open_count": 1,
        "metric_bundle_open_count": 1,
        "checkpoint_payload_open_count": 0,
        "pack_open_count": 0,
        "rgb_open_count": 0,
        "network_access_count": 0,
        "authorizes_retry_or_follow_on": False,
    }
    worker.write_immutable_json(output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-byte-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    receipt = check(
        manifest=arguments.manifest,
        expected_sha256=arguments.expected_file_sha256,
        expected_byte_count=arguments.expected_byte_count,
        output=arguments.output,
    )
    print(json.dumps({"status": receipt["status"], "decision": receipt["decision_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
