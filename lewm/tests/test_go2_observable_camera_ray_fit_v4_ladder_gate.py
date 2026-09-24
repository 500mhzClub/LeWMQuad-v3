from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pytest

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
from scripts import verify_go2_observable_camera_ray_fit_v4_metrics as metric_verifier


ALL_KEYS = [f"{index + 1:064x}" for index in range(320)]
FROZEN_SUBSET_CONTENT_SHA256 = dict(gate.EXPECTED_SUBSET_CONTENT_SHA256)
AUTH_FILE_SHA256 = "a" * 64
AUTH_CONTENT_SHA256 = "b" * 64
REVIEW_FILE_SHA256 = "c" * 64
REVIEW_CONTENT_SHA256 = "d" * 64
RGB_RECEIPT_CONTENT_SHA256 = gate.RGB_RECEIPT_CONTENT_SHA256
PRODUCTION_FINALIZE_STAGE = gate.finalize_development_fit_stage_v4


@pytest.fixture(autouse=True)
def _synthetic_subset_commitments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metric_verifier,
        "__name__",
        "_lewm_v4_ca_test.scripts.verify_go2_observable_camera_ray_fit_v4_metrics",
    )
    monkeypatch.setattr(
        metric_verifier,
        "__verified_logical_name__",
        "scripts.verify_go2_observable_camera_ray_fit_v4_metrics",
        raising=False,
    )
    subset_hashes = {
        size: gate.canonical_json_sha256(ALL_KEYS[:size])
        for size in gate.LADDER_FIT_SIZES
    }
    family_counts = {
        size: _family_counts(size) for size in gate.LADDER_FIT_SIZES
    }
    signatures = {
        size: gate._target_partition_signature(
            _metric_bundle(size, perfect=True)
        )
        for size in gate.LADDER_FIT_SIZES
    }
    monkeypatch.setattr(
        gate,
        "EXPECTED_SUBSET_CONTENT_SHA256",
        subset_hashes,
    )
    monkeypatch.setattr(
        gate,
        "EXPECTED_FIRST_FRAME_KEY_SHA256",
        {size: ALL_KEYS[0] for size in gate.LADDER_FIT_SIZES},
    )
    monkeypatch.setattr(
        gate,
        "EXPECTED_LAST_FRAME_KEY_SHA256",
        {size: ALL_KEYS[size - 1] for size in gate.LADDER_FIT_SIZES},
    )
    monkeypatch.setattr(gate, "EXPECTED_FAMILY_COUNTS", family_counts)
    monkeypatch.setattr(
        gate,
        "EXPECTED_TARGET_PARTITION_SIGNATURES",
        signatures,
    )
    monkeypatch.setattr(
        gate,
        "EXPECTED_TARGET_PARTITION_SIGNATURE_SHA256",
        {
            size: gate.canonical_json_sha256(signatures[size])
            for size in gate.LADDER_FIT_SIZES
        },
    )

    def synthetic_partition(fit_size: int) -> dict[str, Any]:
        core = {
            "schema": "lewm_go2_observable_camera_ray_fit_v4_target_partition_binding_v1",
            "fit_size": fit_size,
            "freeze_file_sha256": gate.TARGET_PARTITION_FREEZE_FILE_SHA256,
            "freeze_content_sha256": gate.TARGET_PARTITION_FREEZE_CONTENT_SHA256,
            "verifier_file_sha256": gate.TARGET_PARTITION_VERIFIER_FILE_SHA256,
            "amendment_file_sha256": gate.TARGET_PARTITION_AMENDMENT_FILE_SHA256,
            "verified_dataset_file_count": 180,
            "family_counts": family_counts[fit_size],
            "first_frame_key_sha256": ALL_KEYS[0],
            "last_frame_key_sha256": ALL_KEYS[fit_size - 1],
            "subset_content_sha256": subset_hashes[fit_size],
            "signature_sha256": gate.canonical_json_sha256(signatures[fit_size]),
            "ordered_per_frame_target_sha256": f"{fit_size + 1000:064x}",
            "ordered_target_bytes_sha256": f"{fit_size + 2000:064x}",
        }
        return {**core, "content_sha256": gate.canonical_json_sha256(core)}

    monkeypatch.setattr(gate, "target_partition_binding_v4", synthetic_partition)

    def finalize_with_synthetic_metric_receipt(
        result: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        kwargs.setdefault(
            "metric_verification_receipt",
            _metric_receipt(result, seed=int(kwargs["expected_seed"])),
        )
        return PRODUCTION_FINALIZE_STAGE(result, *args, **kwargs)

    monkeypatch.setattr(
        gate,
        "finalize_development_fit_stage_v4",
        finalize_with_synthetic_metric_receipt,
    )


def test_frozen_exact_subset_content_hashes() -> None:
    assert FROZEN_SUBSET_CONTENT_SHA256 == {
        5: "3595dff9d24dbb44f3e73086fce3be4ec53eb8659684738defa8591c4a375f15",
        16: "3e3706c4d46476c9d6682e92bd80aa97bd7b0f0bd5bc2c9b69b9aa3605f9d4ba",
        32: "19ae70495e7a21e4ecacd7846672145ffc0187ced6b4f9296c7f9e5b4d46ed73",
        320: "be4b8863120d67132180228982f0631f5f8f6042b581ee5f8a61559fa58188b1",
    }
    assert gate.EXPECTED_RASTER_CLASS_COUNTS == {
        5: {"unknown": 16_123, "free": 4_259, "occupied": 98},
        16: {"unknown": 51_302, "free": 13_876, "occupied": 358},
        32: {"unknown": 104_108, "free": 25_975, "occupied": 989},
        320: {"unknown": 1_072_012, "free": 228_477, "occupied": 10_231},
    }


def _binary_metrics(
    count: int,
    *,
    perfect: bool,
    positive_count: int | None = None,
) -> dict[str, Any]:
    positive = count - count // 2 if positive_count is None else positive_count
    negative = count - positive
    if perfect:
        matrix = [[negative, 0], [0, positive]]
    else:
        negative_correct = negative // 2
        positive_correct = positive // 2
        matrix = [
            [negative_correct, negative - negative_correct],
            [positive - positive_correct, positive_correct],
        ]
    negative_recall = None if negative == 0 else matrix[0][0] / negative
    positive_recall = None if positive == 0 else matrix[1][1] / positive
    present = [value for value in (negative_recall, positive_recall) if value is not None]
    return {
        "confusion_target_rows_predicted_columns": matrix,
        "negative_recall": negative_recall,
        "positive_recall": positive_recall,
        "balanced_accuracy": None if not present else sum(present) / len(present),
        "count": count,
    }


def _raster_metrics(fit_size: int, *, perfect: bool) -> dict[str, Any]:
    counts = gate.EXPECTED_RASTER_CLASS_COUNTS[fit_size]
    matrix = [[0, 0, 0] for _ in range(3)]
    recalls = {}
    present = []
    for class_index, class_name in enumerate(gate.RASTER_CLASSES):
        count = counts[class_name]
        if count == 0:
            recalls[class_name] = None
            continue
        correct = count if perfect else count // 2
        matrix[class_index][class_index] = correct
        matrix[class_index][(class_index + 1) % 3] = count - correct
        recall = correct / count
        recalls[class_name] = recall
        present.append(recall)
    nll = 0.01 if perfect else 1.0
    return {
        "nll": nll,
        "nll_sum": nll * fit_size * 64 * 64,
        "confusion_target_rows_predicted_columns": matrix,
        "class_recalls": recalls,
        "balanced_accuracy": sum(present) / len(present),
        "count": fit_size * 64 * 64,
    }


def _allocate(total: int, weights: list[int]) -> list[int]:
    weight_total = sum(weights)
    values = [total * weight // weight_total for weight in weights]
    for index in range(total - sum(values)):
        values[index % len(values)] += 1
    return values


def _family_counts(fit_size: int) -> dict[str, int]:
    return {
        family: fit_size // len(gate.FAMILIES)
        + (1 if index < fit_size % len(gate.FAMILIES) else 0)
        for index, family in enumerate(gate.FAMILIES)
    }


def _metric_bundle(fit_size: int, *, perfect: bool) -> dict[str, Any]:
    pixel_count = fit_size * 84 * 112
    ground_count = fit_size * 1000
    distance_counts = [ground_count // 2, ground_count - ground_count // 2, 0, 0, 0, 0]
    family_counts = _family_counts(fit_size)
    positive_families = [family for family, count in family_counts.items() if count]
    per_family = _allocate(
        ground_count, [family_counts[family] for family in positive_families]
    )
    depth_count = fit_size * 100
    median = 0.01 if perfect else 0.50
    p95 = 0.02 if perfect else 0.80

    def quantile_record(quantile: float, value: float) -> dict[str, Any]:
        position = (depth_count - 1) * quantile
        return {
            "quantile": quantile,
            "lower_index": math.floor(position),
            "upper_index": math.ceil(position),
            "upper_weight": position - math.floor(position),
            "lower_value_m": value,
            "upper_value_m": value,
        }

    return {
        "frame_count": fit_size,
        "pixel_hit_no_hit": _binary_metrics(
            pixel_count,
            perfect=perfect,
            positive_count=depth_count,
        ),
        "pixel_hit_depth": {
            "count": depth_count,
            "median_absolute_error_m": median,
            "p95_absolute_error_m": p95,
            "absolute_error_evidence": {
                "dtype": "little_endian_float64",
                "quantile_method": "linear_interpolation_n_minus_1_v1",
                "sorted_values_sha256": "f" * 64,
                "median": quantile_record(0.5, median),
                "p95": quantile_record(0.95, p95),
            },
        },
        "ground_clear": {
            "overall": _binary_metrics(ground_count, perfect=perfect),
            "by_distance_m": {
                name: _binary_metrics(count, perfect=perfect)
                for name, count in zip(gate.GROUND_DISTANCE_GROUPS, distance_counts)
            },
            "by_family": {
                family: _binary_metrics(count, perfect=perfect)
                for family, count in zip(positive_families, per_family)
            },
        },
        "derived_raster": _raster_metrics(fit_size, perfect=perfect),
    }


def _snapshot(step: int, total: float = 0.1) -> dict[str, Any]:
    return {
        "step": step,
        "total": total,
        "components": {
            "ordered_first_hit_nll": total,
            "target_bin_offset_smooth_l1": total,
            "ground_clear_distance_state_balanced_bce": total,
            "derived_raster_hierarchical_bce": total,
        },
        "gradient_norm_before_clip": 0.5,
    }


def _result(
    fit_size: int,
    *,
    seed: int,
    previous_stage_gate: Mapping[str, Any] | None = None,
    seed_20260710_gate: Mapping[str, Any] | None = None,
    matched_perfect: bool = True,
    wrong_perfect: bool = False,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "dataset_manifest_file_sha256": gate.DATASET_MANIFEST_FILE_SHA256,
        "dataset_manifest_content_sha256": gate.DATASET_MANIFEST_CONTENT_SHA256,
        "audit_receipt_file_sha256": gate.AUDIT_RECEIPT_FILE_SHA256,
        "audit_receipt_content_sha256": gate.AUDIT_RECEIPT_CONTENT_SHA256,
        "trainer_authorization_file_sha256": AUTH_FILE_SHA256,
        "trainer_authorization_content_sha256": AUTH_CONTENT_SHA256,
        "trainer_review_record_file_sha256": REVIEW_FILE_SHA256,
        "trainer_review_record_content_sha256": REVIEW_CONTENT_SHA256,
        "rgb_receipt_content_sha256": RGB_RECEIPT_CONTENT_SHA256,
    }
    if previous_stage_gate is not None:
        inputs["previous_stage_gate"] = gate._stage_gate_binding(previous_stage_gate)
    if seed_20260710_gate is not None:
        inputs["seed_20260710_gate"] = gate._seed_gate_binding(seed_20260710_gate)
    family_counts = _family_counts(fit_size)
    subset_keys = ALL_KEYS[:fit_size]
    steps = gate.DEFAULT_STEPS[fit_size]
    target_partition = gate.target_partition_binding_v4(fit_size)
    inputs["target_partition_content_sha256"] = target_partition["content_sha256"]
    matched_metrics = _metric_bundle(fit_size, perfect=matched_perfect)
    wrong_metrics = _metric_bundle(fit_size, perfect=wrong_perfect)
    core = {
        "schema": gate.FIT_RESULT_SCHEMA,
        "mode": "exact_development_fit",
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "dataset_role": "train",
        "fit_size": fit_size,
        "attempt": {
            "attempt_index": 1,
            "maximum_attempts": 1,
            "scope": "one_frozen_attempt_per_seed_and_fit_size",
            "reservation": {
                "path": "reservation.json",
                "file_sha256": "1" * 64,
                "content_sha256": "2" * 64,
            },
            "predecessor_failure": gate.V1_FAILURE_LINEAGE,
        },
        "subset": {
            "namespace": "lewm_go2_observable_camera_ray_fit_v4_subset_v1",
            "parent_frame_count": 320,
            "fit_size": fit_size,
            "selection": (
                "registered_family_round_robin_then_namespaced_sha256_"
                "ascii_backslash_zero_rank_v1"
            ),
            "family_counts": family_counts,
            "ordered_frame_key_sha256": subset_keys,
            "content_sha256": gate.EXPECTED_SUBSET_CONTENT_SHA256[fit_size],
        },
        "target_partition": target_partition,
        "inputs": inputs,
        "model": {
            "class": "ObservableCameraRayEvidenceV4Model",
            "parameter_count": gate.MODEL_PARAMETER_COUNT,
            "checkpoint": {
                "path": "checkpoint.pt",
                "file_sha256": f"{fit_size + seed:064x}"[-64:],
                "content_sha256": f"{fit_size + seed + 1:064x}"[-64:],
                "byte_count": 123456,
                "development_only": True,
            },
        },
        "training": {
            "steps": steps,
            "batch_size": gate.TRAIN_BATCH_SIZE,
            "evaluation_batch_size": gate.EVALUATION_BATCH_SIZE,
            "learning_rate": gate.LEARNING_RATE,
            "weight_decay": gate.WEIGHT_DECAY,
            "optimizer": "AdamW",
            "precision": "float32",
            "autocast": False,
            "gradient_clip_norm": 1.0,
            "loss_weights": {
                "ordered_first_hit_nll": 0.25,
                "target_bin_offset_smooth_l1": 0.25,
                "ground_clear_distance_state_balanced_bce": 0.25,
                "derived_raster_hierarchical_bce": 0.25,
            },
            "initial": _snapshot(1),
            "final": _snapshot(steps, 0.01),
            "best_total": 0.01,
            "trace": [_snapshot(1), _snapshot(steps, 0.01)],
            "schedule_algorithm": gate.SCHEDULE_ALGORITHM,
            "schedule_sha256": gate.EXPECTED_SCHEDULE_SHA256[seed][fit_size],
        },
        "evaluation": {
            "matched_rgb": {
                "control": "matched_rgb",
                "wrong_rgb_degenerate_singleton": False,
                "image_index_mapping": list(range(fit_size)),
                "image_mapping_sha256": gate.canonical_json_sha256(
                    list(range(fit_size))
                ),
                "losses": {
                    "total": 0.01,
                    "ordered_first_hit_nll": 0.01,
                    "target_bin_offset_smooth_l1": 0.01,
                    "ground_clear_distance_state_balanced_bce": 0.01,
                    "derived_raster_hierarchical_bce": 0.01,
                },
                "metrics": matched_metrics,
            },
            "wrong_rgb_with_target_calibration": {
                "control": "wrong_rgb_with_target_calibration",
                "wrong_rgb_degenerate_singleton": False,
                "image_index_mapping": [
                    (index + 1) % fit_size for index in range(fit_size)
                ],
                "image_mapping_sha256": gate.canonical_json_sha256(
                    [(index + 1) % fit_size for index in range(fit_size)]
                ),
                "losses": {
                    "total": 0.5,
                    "ordered_first_hit_nll": 0.5,
                    "target_bin_offset_smooth_l1": 0.5,
                    "ground_clear_distance_state_balanced_bce": 0.5,
                    "derived_raster_hierarchical_bce": 0.5,
                },
                "metrics": wrong_metrics,
            },
        },
        "resource": {
            "device": "cuda:0",
            "device_name": "AMD Radeon AI PRO R9700",
            "visible_device_count": 1,
            "total_memory_bytes": 20 * 1024**3,
            "hip_visible_devices": "0",
            "hsa_override_gfx_version_unset": True,
            "raphael_rejected": True,
            "minimum_memory_bytes": 16 * 1024**3,
            "native_thread_environment": {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        },
        "determinism": {
            "seed": seed,
            "requested": "strict_deterministic_algorithms",
            "effective": "strict_where_supported_warn_on_exact_allowlisted_kernels",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "warning_count": 0,
            "raw_messages": [],
            "normalized_messages": [],
            "normalization": [],
            "whitelist": list(gate.DETERMINISM_WARNING_WHITELIST),
            "kernel_inventory": list(gate.DETERMINISM_WARNING_KERNELS),
            "kernel_counts": {
                "grid_sampler_2d_backward_cuda": 0,
                "scatter_add_cuda_kernel": 0,
            },
        },
        "access_ledger": {
            "selected_rgb_count": fit_size,
            "nonselected_rgb_opens": 0,
            "rgb_hash_opens": fit_size,
            "rgb_decodes": fit_size,
            "worker_start_method": "spawn",
            "worker_count": min(6, fit_size),
            "native_threads_per_worker": 1,
            "selected_rgb_rehashes_before_publication": fit_size,
            "heldout_opens": 0,
            "g2_opens": 0,
            "runtime_opens": 0,
            "gpu1_uses": 0,
            "dataset_root_inventory_revalidations": 1,
            "shard_directory_inventory_revalidations": 20,
            "dataset_frame_revalidations": 320,
            "dataset_file_rehashes": 181,
            "trainer_source_rehashes": gate.EXPECTED_TRAINER_SOURCE_REHASHES,
            "dataset_source_rehashes": 11,
        },
        "licenses": {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return {**core, "content_sha256": gate.canonical_json_sha256(core)}


def _rehash(result: dict[str, Any]) -> None:
    core = dict(result)
    core.pop("content_sha256", None)
    result["content_sha256"] = gate.canonical_json_sha256(core)


def _artifacts(result: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    checkpoint = result["model"]["checkpoint"]
    metric_receipt = _metric_receipt(result, seed=seed)
    return {
        "attempt_directory": f"attempts/seed_{seed}/n{result['fit_size']}",
        "reservation": dict(result["attempt"]["reservation"]),
        "result": {
            "path": "result.json",
            "file_sha256": "3" * 64,
            "content_sha256": result["content_sha256"],
        },
        "checkpoint": {
            "path": "checkpoint.pt",
            "file_sha256": checkpoint["file_sha256"],
            "content_sha256": checkpoint["content_sha256"],
            "byte_count": checkpoint["byte_count"],
        },
        "completion": {
            "path": "completed.json",
            "file_sha256": "4" * 64,
            "content_sha256": "5" * 64,
        },
        "metric_verification": {
            "path": (
                f"metric_verifications/seed_{seed}_n{result['fit_size']}.json"
            ),
            "file_sha256": "6" * 64,
            "content_sha256": metric_receipt["content_sha256"],
        },
    }


def _metric_receipt(
    result: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    return metric_verifier.build_metric_verification_receipt(
        gate_module=gate,
        authorization={
            "content_sha256": metric_verifier.AUTHORIZATION_CONTENT_SHA256
        },
        seed=seed,
        fit_size=int(result["fit_size"]),
        result_content_sha256=str(result["content_sha256"]),
        checkpoint=result["model"]["checkpoint"],
        target_partition_reproduction={
            "target_partition": result["target_partition"],
            "reproduced_before_checkpoint_inference": True,
        },
        recomputed_evaluation=result["evaluation"],
    )


def _passing_seed(
    seed: int,
    *,
    seed_20260710_gate: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stages = []
    previous = None
    for fit_size in gate.LADDER_FIT_SIZES:
        result = _result(
            fit_size,
            seed=seed,
            previous_stage_gate=previous,
            seed_20260710_gate=seed_20260710_gate,
        )
        previous = gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=seed,
            artifact_binding=_artifacts(result, seed=seed),
            previous_stage_gate=previous,
            seed_20260710_gate=seed_20260710_gate,
        )
        assert previous["passes"] is True
        stages.append(previous)
    seed_gate = gate.finalize_development_fit_seed_v4(
        stages,
        expected_seed=seed,
        seed_20260710_gate=seed_20260710_gate,
    )
    return stages, seed_gate


def test_all_four_rungs_and_both_seeds_pass_only_in_sequence() -> None:
    first_results, first_gate = _passing_seed(20260710)
    assert [result["fit_size"] for result in first_results] == [5, 16, 32, 320]
    assert first_gate["ladder_passes"] is True
    assert first_gate["seed_20260711_n5_execution_authorized"] is True

    _second_results, second_gate = _passing_seed(
        20260711, seed_20260710_gate=first_gate
    )
    combined = gate.finalize_development_fit_two_seed_v4(first_gate, second_gate)
    assert combined["both_seed_ladders_pass"] is True
    assert all(value is False for value in combined["licenses"].values())


def test_n5_present_occupied_class_is_gated() -> None:
    result = _result(5, seed=20260710)
    stage = gate.finalize_development_fit_stage_v4(
        result,
        expected_seed=20260710,
        artifact_binding=_artifacts(result, seed=20260710),
    )
    assert stage["passes"] is True
    assert any(
        check["name"] == "matched.raster_recall.occupied"
        for check in stage["numeric_gate"]["checks"]
    )


def test_numeric_failure_records_checkpoint_but_forbids_use_and_larger_rung() -> None:
    failed_result = _result(
        5,
        seed=20260710,
        matched_perfect=False,
        wrong_perfect=False,
    )
    failed = gate.finalize_development_fit_stage_v4(
        failed_result,
        expected_seed=20260710,
        artifact_binding=_artifacts(failed_result, seed=20260710),
    )
    assert failed["checkpoint_created"] is True
    assert failed["passes"] is False
    assert failed["development_checkpoint_use_authorized"] is False
    assert failed["next_rung_execution_authorized"] is False
    bypass = _result(16, seed=20260710, previous_stage_gate=failed)
    with pytest.raises(PermissionError, match="did not license"):
        gate.finalize_development_fit_stage_v4(
            bypass,
            expected_seed=20260710,
            artifact_binding=_artifacts(bypass, seed=20260710),
            previous_stage_gate=failed,
        )


def test_wrong_rgb_dependence_is_a_numeric_gate() -> None:
    n5 = _result(
        5,
        seed=20260710,
        wrong_perfect=True,
    )
    stage = gate.finalize_development_fit_stage_v4(
        n5,
        expected_seed=20260710,
        artifact_binding=_artifacts(n5, seed=20260710),
    )
    assert stage["passes"] is False
    assert any(
        failure["name"].startswith("wrong_rgb")
        for failure in stage["numeric_gate"]["failed_checks"]
    )


def test_result_gate_accepts_only_exact_context_trailer_warning_evidence() -> None:
    result = _result(5, seed=20260710)
    normalized = gate.DETERMINISM_WARNING_WHITELIST[0]
    raw = (
        normalized
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)"
    )
    result["determinism"].update(
        {
            "warning_count": 1,
            "raw_messages": [raw],
            "normalized_messages": [normalized],
            "normalization": [
                {
                    "raw": raw,
                    "normalized": normalized,
                    "context_source_line": 157,
                    "trailer_removed": True,
                }
            ],
            "kernel_counts": {
                "grid_sampler_2d_backward_cuda": 1,
                "scatter_add_cuda_kernel": 0,
            },
        }
    )
    _rehash(result)
    assert gate.finalize_development_fit_stage_v4(
        result,
        expected_seed=20260710,
        artifact_binding=_artifacts(result, seed=20260710),
    )["passes"] is True

    changed_path = copy.deepcopy(result)
    changed_path["determinism"]["raw_messages"][0] = raw.replace(
        "/ATen/Context.cpp:", "/ATen/Other.cpp:"
    )
    changed_path["determinism"]["normalization"][0]["raw"] = changed_path[
        "determinism"
    ]["raw_messages"][0]
    _rehash(changed_path)
    with pytest.raises(ValueError, match="unallowlisted"):
        gate.finalize_development_fit_stage_v4(
            changed_path,
            expected_seed=20260710,
            artifact_binding=_artifacts(changed_path, seed=20260710),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attempt", "maximum_attempts"), 2),
        (("training", "steps"), 999),
        (("training", "evaluation_batch_size"), 2),
        (("training", "learning_rate"), 2e-4),
        (("training", "schedule_sha256"), "f" * 64),
        (("model", "parameter_count"), gate.MODEL_PARAMETER_COUNT + 1),
        (("access_ledger", "rgb_decodes"), 0),
    ],
)
def test_optimizer_exposure_attempt_model_and_rgb_mutations_are_structural(
    path: tuple[str, str], value: Any
) -> None:
    result = _result(5, seed=20260710)
    result[path[0]][path[1]] = value
    _rehash(result)
    with pytest.raises((ValueError, PermissionError)):
        gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=20260710,
            artifact_binding=_artifacts(result, seed=20260710),
        )


def test_seed_20260711_n5_requires_passing_seed_20260710_gate() -> None:
    result = _result(5, seed=20260711)
    with pytest.raises(PermissionError, match="seed 20260711"):
        gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=20260711,
            artifact_binding=_artifacts(result, seed=20260711),
        )


def test_exact_raster_target_count_mutation_is_structural() -> None:
    result = _result(5, seed=20260710)
    raster = result["evaluation"]["matched_rgb"]["metrics"]["derived_raster"]
    raster["confusion_target_rows_predicted_columns"][0][0] -= 1
    raster["confusion_target_rows_predicted_columns"][1][1] += 1
    _rehash(result)
    with pytest.raises(ValueError, match="target class counts"):
        gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=20260710,
            artifact_binding=_artifacts(result, seed=20260710),
        )


def test_minimal_self_hashed_stage_and_seed_gates_are_rejected() -> None:
    stage_core = {
        "schema": gate.STAGE_GATE_SCHEMA,
        "seed": 20260710,
        "fit_size": 5,
        "passes": True,
        "development_checkpoint_use_authorized": False,
        "next_rung_execution_authorized": True,
    }
    forged_stage = {
        **stage_core,
        "content_sha256": gate.canonical_json_sha256(stage_core),
    }
    with pytest.raises(ValueError, match="schema changed"):
        gate._stage_gate_binding(forged_stage)

    seed_core = {
        "schema": gate.SEED_GATE_SCHEMA,
        "seed": 20260710,
        "ladder_passes": True,
        "seed_20260711_n5_execution_authorized": True,
    }
    forged_seed = {
        **seed_core,
        "content_sha256": gate.canonical_json_sha256(seed_core),
    }
    with pytest.raises(ValueError, match="schema changed"):
        gate._seed_gate_binding(forged_seed)


def test_larger_rung_requires_the_immediate_predecessor() -> None:
    n5 = _result(5, seed=20260710)
    first = gate.finalize_development_fit_stage_v4(
        n5,
        expected_seed=20260710,
        artifact_binding=_artifacts(n5, seed=20260710),
    )
    n32 = _result(32, seed=20260710, previous_stage_gate=first)
    with pytest.raises(PermissionError, match="immediate predecessor"):
        gate.finalize_development_fit_stage_v4(
            n32,
            expected_seed=20260710,
            artifact_binding=_artifacts(n32, seed=20260710),
            previous_stage_gate=first,
        )


def test_pure_stage_fields_check_file_hash_and_next_rung() -> None:
    result = _result(5, seed=20260710)
    stage = gate.finalize_development_fit_stage_v4(
        result,
        expected_seed=20260710,
        artifact_binding=_artifacts(result, seed=20260710),
    )
    digest = gate._gate_file_sha256(stage)
    binding = gate._validate_stage_execution_fields(
        stage,
        gate_file_sha256=digest,
        expected_seed=20260710,
        expected_next_fit_size=16,
    )
    assert binding["fit_size"] == 5
    with pytest.raises(ValueError, match="file hash"):
        gate._validate_stage_execution_fields(
            stage,
            gate_file_sha256="0" * 64,
            expected_seed=20260710,
            expected_next_fit_size=16,
        )
    with pytest.raises(PermissionError, match="immediate rung"):
        gate._validate_stage_execution_fields(
            stage,
            gate_file_sha256=digest,
            expected_seed=20260710,
            expected_next_fit_size=32,
        )


def test_second_seed_gate_must_persist_first_seed_binding() -> None:
    _first_stages, first = _passing_seed(20260710)
    _second_stages, second = _passing_seed(
        20260711,
        seed_20260710_gate=first,
    )
    forged = copy.deepcopy(second)
    forged["seed_20260710_gate"] = None
    core = dict(forged)
    core.pop("content_sha256")
    forged["content_sha256"] = gate.canonical_json_sha256(core)
    with pytest.raises((ValueError, PermissionError), match="first-seed"):
        gate._seed_gate_binding(forged)


def test_bound_trainer_snapshot_binds_exact_artifacts_sources_and_narrow_license() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
    )
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == gate._canonical_json_bytes(value) + b"\n"
    core = dict(value)
    declared = core.pop("content_sha256")
    assert declared == gate.canonical_json_sha256(core)
    assert value["status"] == "independent_review_passed_authorized"
    assert value["authorization"] == {
        "development_fit": True,
        "development_checkpoint_creation_authorized": True,
        "checkpoint_use_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    assert value["dataset_binding"]["file_sha256"] == gate.DATASET_MANIFEST_FILE_SHA256
    assert value["dataset_binding"]["content_sha256"] == gate.DATASET_MANIFEST_CONTENT_SHA256
    assert value["audit_binding"]["file_sha256"] == gate.AUDIT_RECEIPT_FILE_SHA256
    assert value["audit_binding"]["content_sha256"] == gate.AUDIT_RECEIPT_CONTENT_SHA256
    for source in value["source_map"]["entries"]:
        source_path = root / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
