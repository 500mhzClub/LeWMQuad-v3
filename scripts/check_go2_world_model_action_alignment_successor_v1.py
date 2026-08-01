#!/usr/bin/env python3
"""Checker for the bounded fixed same-mechanism alignment continuation."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.datasets import (  # noqa: E402
    go2_explicit_plan_discounted_successor_state_v27 as h6,
)
from scripts import (  # noqa: E402
    execute_go2_world_model_action_alignment_successor_v1 as worker,
)


CHECK_SCHEMA = f"{worker.SCHEMA_PREFIX}_receipt_check_v1"
CONTRACT_CHECK_NAMES = {
    "authority_exact",
    "reservation_exact",
    "source_and_test_closure_exact",
    "runtime_exact",
    "input_bindings_exact",
    "reused_pack_exact",
    "schedule_900_and_u700_prefix_exact",
    "both_exact_u700_snapshots_loaded_once",
    "both_arm_and_own_adamw_states_restored",
    "u700_public_anchor_replay_exact_within_1e_12",
    "absolute_updates_701_through_900_only",
    "both_optimizers_reached_exact_step_900",
    "shared_candidate_route_exact",
    "alignment_coefficients_exact",
    "frozen_substrate_exact",
    "validation_no_gradient",
    "finiteness_exact",
    "no_rgb_or_data_generation",
}
TRAIN_FIT_CHECK_NAMES = {
    f"{name}_full_train_factual_energy_finite_positive_below_two"
    for name in worker.ARM_NAMES
} | {
    f"{name}_terminal_total_loss_finite" for name in worker.ARM_NAMES
}


class AlignmentCheckError(RuntimeError):
    """A terminal result or its bound metric bundle failed verification."""


def _effective_rank_from_covariance(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or tuple(value.shape)
        != (worker.RANK_FEATURE_DIMENSION, worker.RANK_FEATURE_DIMENSION)
        or not bool(torch.isfinite(value).all())
        or not torch.equal(value, value.T)
    ):
        raise AlignmentCheckError(f"{label} covariance changed")
    eigenvalues = torch.linalg.eigvalsh(value).clamp_min(0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    return float(
        (-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()).exp()
    )


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
        "validation_row_indices", "training_factual_energy", "contract_checks",
        "train_fit_checks", "alignment_rank_ratio_observations",
        "baseline_rank_ratio_observations", "rank_covariance_by_update",
        *{
            f"u{update}_{name}_{suffix}"
            for update in (700, 900)
            for name in worker.ARM_NAMES
            for suffix in (
                "candidate_energy", "factual_energy", "persistence_energy",
                "wrong_history_energy",
            )
        },
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
    for update in (700, 900):
        for name in worker.ARM_NAMES:
            candidate = bundle[f"u{update}_{name}_candidate_energy"]
            if (
                not isinstance(candidate, torch.Tensor)
                or tuple(candidate.shape)
                != (worker.EXPECTED_VALIDATION_ROWS, worker.ACTION_COUNT)
                or not bool(torch.isfinite(candidate).all())
                or bool((candidate < 0.0).any())
            ):
                raise AlignmentCheckError(
                    f"u{update} {name} candidate vector changed"
                )
            for suffix in (
                "factual_energy", "persistence_energy", "wrong_history_energy"
            ):
                value = bundle[f"u{update}_{name}_{suffix}"]
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != (worker.EXPECTED_VALIDATION_ROWS,)
                    or not bool(torch.isfinite(value).all())
                    or bool((value <= 0.0).any())
                ):
                    raise AlignmentCheckError(
                        f"u{update} {name} {suffix} changed"
                    )
        if not torch.equal(
            bundle[f"u{update}_baseline_persistence_energy"],
            bundle[f"u{update}_alignment_persistence_energy"],
        ):
            raise AlignmentCheckError(f"u{update} shared persistence vector changed")
    if (
        type(bundle["training_factual_energy"]) is not dict
        or set(bundle["training_factual_energy"]) != set(worker.ARM_NAMES)
    ):
        raise AlignmentCheckError("training vector inventory changed")
    for name in worker.ARM_NAMES:
        train = bundle["training_factual_energy"][name]
        if (
            not isinstance(train, torch.Tensor)
            or tuple(train.shape) != (worker.EXPECTED_TRAIN_ROWS,)
            or not bool(torch.isfinite(train).all())
            or bool((train <= 0.0).any())
        ):
            raise AlignmentCheckError(f"{name} training vector changed")
    for name in worker.ARM_NAMES:
        ranks = bundle[f"{name}_rank_ratio_observations"]
        if (
            not isinstance(ranks, torch.Tensor)
            or ranks.dtype != torch.float64
            or tuple(ranks.shape) != (3,)
            or not bool(torch.isfinite(ranks).all())
            or bool((ranks < 0.0).any())
        ):
            raise AlignmentCheckError(f"{name} rank observations changed")
    rank_covariances = bundle["rank_covariance_by_update"]
    if (
        type(rank_covariances) is not dict
        or set(rank_covariances) != set(worker.OBSERVATION_UPDATES)
    ):
        raise AlignmentCheckError("rank covariance update inventory changed")
    for update in worker.OBSERVATION_UPDATES:
        values = rank_covariances[update]
        if type(values) is not dict or set(values) != {"target", *worker.ARM_NAMES}:
            raise AlignmentCheckError(f"u{update} rank covariance inventory changed")
        for name, covariance in values.items():
            _effective_rank_from_covariance(
                covariance, label=f"u{update} {name} rank"
            )
    contract_checks = bundle["contract_checks"]
    if (
        type(contract_checks) is not dict
        or set(contract_checks) != CONTRACT_CHECK_NAMES
        or any(value is not True for value in contract_checks.values())
    ):
        raise AlignmentCheckError("contract-check inventory or value changed")
    train_fit_checks = bundle["train_fit_checks"]
    if (
        type(train_fit_checks) is not dict
        or set(train_fit_checks) != TRAIN_FIT_CHECK_NAMES
        or any(type(value) is not bool for value in train_fit_checks.values())
    ):
        raise AlignmentCheckError("train-fit check inventory changed")
    return bundle


def _vectors(bundle: Mapping[str, Any], *, update: int) -> worker.EvaluationVectors:
    return worker.EvaluationVectors(
        factual={
            name: bundle[f"u{update}_{name}_factual_energy"]
            for name in worker.ARM_NAMES
        },
        candidates={
            name: bundle[f"u{update}_{name}_candidate_energy"]
            for name in worker.ARM_NAMES
        },
        prediction_tokens={},
        target_tokens=torch.empty(0),
        persistence=bundle[f"u{update}_alignment_persistence_energy"],
        wrong_history={
            name: bundle[f"u{update}_{name}_wrong_history_energy"]
            for name in worker.ARM_NAMES
        },
    )


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
        set(result) != {
            "schema", "status", "development_evidence_complete",
            "citable_as_original_factual_learnability_claim",
            "citable_as_planning_usefulness_evidence", "authority_binding",
            "reservation_binding", "source_commit", "review_commit",
            "execution_head", "plan_binding", "review_binding", "attempt",
            "input_bindings", "evidence_bindings", "metric_bundle_binding",
            "snapshot_bindings", "schedule", "substrate", "restoration",
            "u700_replay_anchor_audit", "observation_measurements",
            "train_fit", "decision", "runtime", "accounting",
            "forbidden_access", "claim_boundary",
        }
        or result.get("schema") != worker.RESULT_SCHEMA
        or result.get("status") != "COMPLETE_PENDING_TERMINAL_REVIEW"
        or result.get("development_evidence_complete") is not True
        or result.get("citable_as_original_factual_learnability_claim") is not False
        or result.get("citable_as_planning_usefulness_evidence") is not False
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
        or result.get("plan_binding") != authority["plan_binding"]
        or result.get("review_binding") != authority["review_binding"]
        or result.get("input_bindings") != worker.EXPECTED_INPUT_BINDINGS
        or result.get("evidence_bindings") != worker.EXPECTED_EVIDENCE_BINDINGS
        or result.get("attempt") != {
            **authority["attempt"],
            "consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
        }
    ):
        raise AlignmentCheckError("result provenance changed")
    schedule900, audit900 = worker.base.build_bound_training_schedule(
        updates=worker.TRAINING_UPDATES
    )
    schedule700, audit700 = worker.base.build_bound_training_schedule(
        updates=worker.START_UPDATE
    )
    if (
        not torch.equal(schedule900[: worker.START_UPDATE], schedule700)
        or result.get("schedule") != {
            "source_updates": worker.START_UPDATE,
            "terminal_updates": worker.TRAINING_UPDATES,
            "source_schedule": audit700,
            "terminal_schedule": audit900,
            "prefix_tensor_exact": True,
            "trained_slice_start_zero_based": worker.START_UPDATE,
            "trained_slice_stop_exclusive": worker.TRAINING_UPDATES,
        }
    ):
        raise AlignmentCheckError("continuation schedule or prefix changed")
    restoration = result.get("restoration")
    if type(restoration) is not dict or set(restoration) != set(worker.ARM_NAMES):
        raise AlignmentCheckError("restoration receipt inventory changed")
    for name in worker.ARM_NAMES:
        receipt = restoration[name]
        if (
            type(receipt) is not dict
            or receipt.get("input_binding")
            != worker.EXPECTED_INPUT_BINDINGS[f"{name}_u700_snapshot"]
            or receipt.get("schema") != worker.PREDECESSOR_SNAPSHOT_SCHEMA
            or receipt.get("arm") != name
            or receipt.get("update") != worker.START_UPDATE
            or receipt.get("optimizer_step") != worker.START_UPDATE
            or receipt.get("optimizer_parameter_count") != 36
            or receipt.get("loaded_once") is not True
            or receipt.get("model_and_own_adamw_restored") is not True
            or type(receipt.get("model_state_sha256")) is not str
            or len(receipt["model_state_sha256"]) != 64
            or type(receipt.get("optimizer_moment_sha256")) is not str
            or len(receipt["optimizer_moment_sha256"]) != 64
        ):
            raise AlignmentCheckError(f"{name} restoration receipt changed")

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
    reported_rank_values = {
        name: bundle[f"{name}_rank_ratio_observations"].tolist()
        for name in worker.ARM_NAMES
    }
    recomputed_rank_values: dict[str, dict[int, float]] = {
        name: {} for name in worker.ARM_NAMES
    }
    for index, update in enumerate(worker.OBSERVATION_UPDATES):
        covariances = bundle["rank_covariance_by_update"][update]
        target_rank = _effective_rank_from_covariance(
            covariances["target"], label=f"u{update} target rank"
        )
        if target_rank <= 0.0:
            raise AlignmentCheckError(f"u{update} target rank is nonpositive")
        for name in worker.ARM_NAMES:
            prediction_rank = _effective_rank_from_covariance(
                covariances[name], label=f"u{update} {name} rank"
            )
            ratio = prediction_rank / target_rank
            if not math.isclose(
                ratio,
                float(reported_rank_values[name][index]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AlignmentCheckError(
                    f"u{update} {name} effective-rank ratio differs"
                )
            recomputed_rank_values[name][update] = ratio
    observations = result.get("observation_measurements")
    if (
        type(observations) is not list
        or len(observations) != len(worker.OBSERVATION_UPDATES)
        or [item.get("update") for item in observations]
        != list(worker.OBSERVATION_UPDATES)
    ):
        raise AlignmentCheckError("observation receipt inventory changed")
    for observation in observations:
        update = observation["update"]
        arms = observation.get("arms")
        if type(arms) is not dict or set(arms) != set(worker.ARM_NAMES):
            raise AlignmentCheckError(f"u{update} observation arm inventory changed")
        for name in worker.ARM_NAMES:
            if not math.isclose(
                float(arms[name].get("rank_ratio")),
                recomputed_rank_values[name][update],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AlignmentCheckError(
                    f"u{update} {name} observation rank differs"
                )
    u700_vectors = _vectors(bundle, update=700)
    u900_vectors = _vectors(bundle, update=900)
    train_fit = result.get("train_fit")
    if type(train_fit) is not dict or set(train_fit) != {
        "full_train_factual_mean_energy", "terminal_training_loss", "checks"
    }:
        raise AlignmentCheckError("train-fit receipt inventory changed")
    train_means = {
        name: float(bundle["training_factual_energy"][name].mean())
        for name in worker.ARM_NAMES
    }
    if train_means != train_fit["full_train_factual_mean_energy"]:
        raise AlignmentCheckError("full-train means differ")
    terminal_losses = train_fit["terminal_training_loss"]
    if type(terminal_losses) is not dict or set(terminal_losses) != set(
        worker.ARM_NAMES
    ):
        raise AlignmentCheckError("terminal-loss arm inventory changed")
    for name in worker.ARM_NAMES:
        values = terminal_losses[name]
        if (
            type(values) is not dict
            or set(values) != {"total", "factual", "hinge"}
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in values.values()
            )
        ):
            raise AlignmentCheckError(f"{name} terminal-loss receipt changed")
    recomputed_train_fit_checks = {
        **{
            f"{name}_full_train_factual_energy_finite_positive_below_two": (
                math.isfinite(train_means[name]) and 0.0 < train_means[name] < 2.0
            )
            for name in worker.ARM_NAMES
        },
        **{
            f"{name}_terminal_total_loss_finite": math.isfinite(
                float(terminal_losses[name]["total"])
            )
            for name in worker.ARM_NAMES
        },
    }
    if (
        recomputed_train_fit_checks != bundle["train_fit_checks"]
        or recomputed_train_fit_checks != train_fit["checks"]
    ):
        raise AlignmentCheckError("train-fit checks were not independently reproduced")
    recomputed_contract_checks = {name: True for name in CONTRACT_CHECK_NAMES}
    if recomputed_contract_checks != bundle["contract_checks"]:
        raise AlignmentCheckError("contract checks changed")
    recomputed = worker.continuation_metrics.decide_alignment_continuation(
        baseline_candidate_energy_u700=u700_vectors.candidates["baseline"],
        baseline_candidate_energy_u900=u900_vectors.candidates["baseline"],
        treatment_candidate_energy_u700=u700_vectors.candidates["alignment"],
        treatment_factual_energy_u700=u700_vectors.factual["alignment"],
        treatment_persistence_energy_u700=u700_vectors.persistence,
        treatment_wrong_history_energy_u700=u700_vectors.wrong_history["alignment"],
        treatment_candidate_energy_u900=u900_vectors.candidates["alignment"],
        treatment_factual_energy_u900=u900_vectors.factual["alignment"],
        treatment_persistence_energy_u900=u900_vectors.persistence,
        treatment_wrong_history_energy_u900=u900_vectors.wrong_history["alignment"],
        validation_rows=validation_rows,
        treatment_rank_ratio_by_update={
            update: recomputed_rank_values["alignment"][update]
            for update in worker.OBSERVATION_UPDATES
        },
        contract_checks=recomputed_contract_checks,
        train_fit_checks=recomputed_train_fit_checks,
    )
    if recomputed != result.get("decision"):
        raise AlignmentCheckError("independently recomputed decision differs")
    replay_receipt = {
        "arms": {
            name: {
                "factual_mean_energy": float(u700_vectors.factual[name].mean()),
                "rank_ratio": float(
                    bundle[f"{name}_rank_ratio_observations"][0]
                ),
            }
            for name in worker.ARM_NAMES
        }
    }
    replay_audit = worker._u700_replay_anchor_audit(
        vectors=u700_vectors,
        rows=validation_rows,
        receipt=replay_receipt,
    )
    if replay_audit != result.get("u700_replay_anchor_audit"):
        raise AlignmentCheckError("u700 replay anchor audit differs")
    if replay_audit.get("passed") is not True:
        raise AlignmentCheckError("u700 replay anchors did not reproduce")
    snapshots = result.get("snapshot_bindings")
    if type(snapshots) is not dict or set(snapshots) != set(worker.ARM_NAMES):
        raise AlignmentCheckError("snapshot inventory changed")
    for name, binding in snapshots.items():
        expected_path = worker.ATTEMPT_ROOT / f"{name}_update_000900.pt"
        if type(binding) is not dict or worker.file_binding(expected_path) != binding:
            raise AlignmentCheckError(f"{name} snapshot binding changed")
    forbidden = result.get("forbidden_access")
    if forbidden != {
        "sealed_material_opened": False,
        "heldout_material_opened": False,
        "protected_runtime_material_opened": False,
        "rgb_opened": False,
        "network_access_used": False,
        "validation_used_for_gradient_updates": False,
        "existing_pack_modified": False,
    }:
        raise AlignmentCheckError("forbidden access was reported")
    runtime = result.get("runtime")
    observed_runtime = runtime.get("observed") if type(runtime) is dict else None
    if (
        type(runtime) is not dict
        or set(runtime) != {"authorized", "observed"}
        or runtime.get("authorized") != authority["runtime"]
        or type(observed_runtime) is not dict
        or set(observed_runtime) != {
            "python_version", "torch_version", "torch_hip", "numpy_version",
            "device_name", "device_arch", "gpu_elapsed_seconds",
            "wall_elapsed_seconds", "maximum_memory_allocated_bytes",
        }
        or observed_runtime.get("python_version") != sys.version.split()[0]
        or observed_runtime.get("torch_version") != torch.__version__
        or observed_runtime.get("torch_hip") != torch.version.hip
        or observed_runtime.get("numpy_version") != worker.np.__version__
        or observed_runtime.get("device_name") != torch.cuda.get_device_name(0)
        or observed_runtime.get("device_arch")
        != str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
        or type(observed_runtime.get("gpu_elapsed_seconds")) not in (int, float)
        or not 0.0 <= float(observed_runtime["gpu_elapsed_seconds"])
        <= worker.MAXIMUM_GPU_SECONDS
        or type(observed_runtime.get("wall_elapsed_seconds")) not in (int, float)
        or not 0.0 <= float(observed_runtime["wall_elapsed_seconds"])
        <= worker.MAXIMUM_WALL_SECONDS
        or type(observed_runtime.get("maximum_memory_allocated_bytes")) is not int
        or observed_runtime["maximum_memory_allocated_bytes"] < 0
    ):
        raise AlignmentCheckError("observed runtime receipt changed")
    accounting = result.get("accounting")
    expected_accounting = {
        "source_global_update": worker.START_UPDATE,
        "terminal_global_update": worker.TRAINING_UPDATES,
        "additional_training_updates": worker.ADDITIONAL_TRAINING_UPDATES,
        "additional_optimizer_steps_per_arm": worker.ADDITIONAL_TRAINING_UPDATES,
        "additional_total_optimizer_steps": (
            worker.ADDITIONAL_TRAINING_UPDATES * len(worker.ARM_NAMES)
        ),
        "additional_schedule_presentations_per_arm": (
            worker.ADDITIONAL_TRAINING_UPDATES * worker.BATCH_SIZE
        ),
        "additional_training_head_row_presentations_per_arm": (
            worker.ADDITIONAL_TRAINING_UPDATES * worker.BATCH_SIZE * 10
        ),
        "additional_training_head_row_presentations_total": (
            worker.ADDITIONAL_TRAINING_UPDATES
            * worker.BATCH_SIZE
            * 10
            * len(worker.ARM_NAMES)
        ),
        "additional_training_shared_frame_encodings": (
            worker.ADDITIONAL_TRAINING_UPDATES * worker.BATCH_SIZE * 4
        ),
        "validation_updates": list(worker.OBSERVATION_UPDATES),
        "full_train_fit_rows_per_arm": worker.EXPECTED_TRAIN_ROWS,
        "u700_snapshot_byte_read_count": 2,
        "u700_snapshot_deserialization_count": 2,
        "predecessor_metric_bundle_byte_read_count": 0,
        "predecessor_metric_bundle_deserialization_count": 0,
        "bound_non_snapshot_input_identity_hash_reads_performed": True,
        "pack_payloads_opened_for_training_and_evaluation": True,
        "prior_attempt_write_count": 0,
        "pack_reused_read_only": True,
        "rgb_open_count": 0,
        "data_generation_count": 0,
        "network_access_count": 0,
        "sealed_open_count": 0,
        "heldout_open_count": 0,
    }
    if accounting != expected_accounting:
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
        "absolute_progress_bootstrap_exactly_recomputed": True,
        "concurrent_delta_diagnostic_exactly_recomputed": True,
        "u700_replay_anchor_audit_exactly_recomputed": True,
        "effective_rank_from_covariance_exactly_recomputed": True,
        "train_fit_checks_exactly_recomputed": True,
        "contract_inventory_exactly_validated": True,
        "runtime_and_accounting_exactly_validated": True,
        "snapshot_bindings_verified_without_loading": True,
        "result_byte_read_count": 2,
        "validation_index_identity_hash_reads_performed": True,
        "validation_index_metadata_deserialized": True,
        "metric_bundle_byte_read_count": 2,
        "metric_bundle_deserialization_count": 1,
        "u900_snapshot_identity_hash_read_count": 2,
        "u900_snapshot_deserialization_count": 0,
        "u700_snapshot_content_read_count": 0,
        "bound_pack_identity_hash_reads_performed": True,
        "pack_tensor_or_array_deserialization_count": 0,
        "bound_predecessor_evidence_identity_hash_reads_performed": True,
        "predecessor_metric_bundle_byte_read_count": 0,
        "source_test_and_runtime_identity_hash_reads_performed": True,
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
