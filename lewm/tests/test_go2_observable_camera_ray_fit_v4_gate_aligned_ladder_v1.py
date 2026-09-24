"""Pure/tmp and CPU contract tests for the lean Camera gate-aligned ladder."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as retained_gate
from scripts import run_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1 as ladder


ROOT = Path(__file__).resolve().parents[2]


def _self_hashed(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "content_sha256": ladder.canonical_json_sha256(core)}


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ladder.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return raw


def _dummy_binding(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(path.encode("ascii")).hexdigest(),
        "content_sha256": hashlib.sha256((path + ":content").encode("ascii")).hexdigest(),
        "byte_count": 1,
    }


def _review_binding() -> dict[str, Any]:
    return _dummy_binding(ladder.SOURCE_REVIEW_RELATIVE_PATH)


_DOWNSTREAM_DENIALS = {
    "heldout_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
}

_NUMERIC_THRESHOLD_KEYS = {
    "matched.pixel_hit_balanced_accuracy": "pixel_hit_balanced_accuracy_min",
    "matched.pixel_hit_depth_median_absolute_error_m": (
        "pixel_hit_depth_median_error_m_max"
    ),
    "matched.pixel_hit_depth_p95_absolute_error_m": (
        "pixel_hit_depth_p95_error_m_max"
    ),
    "matched.ground_overall_balanced_accuracy": (
        "ground_overall_balanced_accuracy_min"
    ),
    "matched.raster_nll": "raster_nll_max",
    "matched.raster_balanced_accuracy": "raster_balanced_accuracy_min",
    "wrong_rgb.pixel_balanced_accuracy_drop": (
        "wrong_pixel_balanced_accuracy_drop_min"
    ),
    "wrong_rgb.depth_median_error_increase_m": (
        "wrong_depth_median_error_increase_m_min"
    ),
    "wrong_rgb.depth_p95_error_increase_m": (
        "wrong_depth_p95_error_increase_m_min"
    ),
    "wrong_rgb.ground_balanced_accuracy_drop": (
        "wrong_ground_balanced_accuracy_drop_min"
    ),
    "wrong_rgb.raster_nll_increase": "wrong_raster_nll_increase_min",
    "wrong_rgb.raster_balanced_accuracy_drop": (
        "wrong_raster_balanced_accuracy_drop_min"
    ),
    **{
        f"matched.ground_distance.{group}.balanced_accuracy": (
            "ground_distance_balanced_accuracy_min"
        )
        for group in (
            "0.0_to_1.0",
            "1.0_to_2.0",
            "2.0_to_3.0",
            "3.0_to_4.0",
            "4.0_to_5.0",
            "5.0_plus",
        )
    },
    **{
        f"matched.ground_family.{family}.balanced_accuracy": (
            "ground_family_balanced_accuracy_min"
        )
        for family in (
            "open_obstacle_field",
            "rough_local_dynamics",
            "small_enclosed_maze",
            "medium_enclosed_maze",
            "large_enclosed_maze",
        )
    },
    **{
        f"matched.raster_recall.{class_name}": "raster_class_recall_min"
        for class_name in ("unknown", "free", "occupied")
    },
}
_MAXIMUM_NUMERIC_CHECKS = {
    "matched.pixel_hit_depth_median_absolute_error_m",
    "matched.pixel_hit_depth_p95_absolute_error_m",
    "matched.raster_nll",
}


def _resource() -> dict[str, Any]:
    return {
        "device": "cuda:0",
        "device_name": "AMD Radeon AI PRO R9700",
        "visible_device_count": 1,
        "native_thread_environment": {
            name: "1" for name in ladder.THREAD_ENVIRONMENT
        },
        "all_conflicting_selectors_unset": True,
    }


def _reservation(
    row: ladder.LadderRow,
    review: dict[str, Any],
    prerequisites: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt_identity = hashlib.sha256(f"attempt:{row.key}".encode()).hexdigest()
    initial_state = hashlib.sha256(f"initial:{row.key}".encode()).hexdigest()
    initialization = {
        "attempt_identity": attempt_identity,
        "initial_state_sha256": initial_state,
        "initialization_identity": ladder.initialization_identity(
            row, attempt_identity, initial_state
        ),
        "fresh_model_construction": True,
        "predecessor_checkpoint_opens": 0,
    }
    return _self_hashed(
        {
            "schema": ladder.ROW_RESERVATION_SCHEMA,
            "status": "reserved",
            "row": asdict(row) | {"key": row.key},
            "attempt_index": 1,
            "maximum_attempts": 1,
            "attempt_identity": attempt_identity,
            "source_review": review,
            "prerequisite_gates": prerequisites,
            "terminal_v16": (
                ladder.terminal_v16_summary_contract() if row.index == 0 else None
            ),
            "science_contract_sha256": ladder.canonical_json_sha256(
                ladder.science_contract()
            ),
            "inputs": {
                "data_bindings": ladder.DATA_BINDINGS,
                "subset": {
                    "content_sha256": ladder.SUBSET_CONTENT_SHA256[row.fit_size]
                },
                "target_partition": {
                    "content_sha256": ladder.TARGET_PARTITION_CONTENT_SHA256[
                        row.fit_size
                    ]
                },
            },
            "initialization": initialization,
            "resource": _resource(),
            "determinism": {
                "seed": row.seed,
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
            },
            "retry_authorized": False,
            "licenses": {
                "development_checkpoint_creation_authorized": True,
                "metric_verification_checkpoint_use_authorized": True,
                "predecessor_checkpoint_use_authorized": False,
                **_DOWNSTREAM_DENIALS,
            },
        }
    )


def _numeric_gate(row: ladder.LadderRow, *, passes: bool = True) -> dict[str, Any]:
    thresholds = asdict(retained_gate.FIT_THRESHOLDS[row.fit_size])
    checks = []
    contract = ladder.expected_numeric_check_contract(thresholds)
    for index, (name, (comparison, threshold)) in enumerate(sorted(contract.items())):
        observed = threshold
        if not passes and index == 0:
            observed = (
                threshold - 1.0
                if comparison == "greater_than_or_equal"
                else threshold + 1.0
            )
        check = {
            "name": name,
            "comparison": comparison,
            "value": observed,
            "threshold": threshold,
            "passes": (
                observed >= threshold
                if comparison == "greater_than_or_equal"
                else observed <= threshold
            ),
        }
        checks.append(check)
    failed = [check for check in checks if not check["passes"]]
    return {
        "fit_size": row.fit_size,
        "thresholds": thresholds,
        "wrong_rgb_dependence_assessable": True,
        "check_count": len(checks),
        "checks": checks,
        "failure_count": len(failed),
        "failed_checks": failed,
        "passes": not failed,
    }


def _materialize_passing_row(
    rows_root: Path,
    row: ladder.LadderRow,
    review: dict[str, Any],
    prerequisites: list[dict[str, Any]],
    *,
    numeric_passes: bool = True,
) -> dict[str, Any]:
    directory = rows_root / row.key
    directory.mkdir(parents=True)
    row_value = asdict(row) | {"key": row.key}
    reservation = _reservation(row, review, prerequisites)
    reservation_raw = _write_json(directory / "reservation.json", reservation)
    reservation_binding = ladder.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=reservation["content_sha256"],
    )
    checkpoint_raw = b"synthetic-checkpoint"
    (directory / "checkpoint.pt").write_bytes(checkpoint_raw)
    checkpoint_binding = ladder.artifact_binding(
        "checkpoint.pt",
        checkpoint_raw,
        content_sha256=hashlib.sha256(b"semantic-checkpoint").hexdigest(),
    )
    evaluation = {"synthetic_evaluation": row.key}
    gate_evaluation = {"synthetic_gate_evaluation": row.key}
    result = _self_hashed(
        {
            "schema": ladder.ROW_RESULT_SCHEMA,
            "status": "completed_training",
            "row": row_value,
            "attempt_identity": reservation["attempt_identity"],
            "source_review": review,
            "reservation": reservation_binding,
            "subset": reservation["inputs"]["subset"],
            "target_partition": reservation["inputs"]["target_partition"],
            "initialization": reservation["initialization"],
            "model": {
                "class": "ObservableCameraRayEvidenceV4Model",
                "fresh_initialization": True,
                "parameter_count": 3_105_513,
                "checkpoint": {**checkpoint_binding, "development_only": True}
            },
            "training": {
                "steps": row.updates,
                "batch_size": row.batch_size,
                "frame_exposures": row.frame_exposures,
                "schedule_sha256": row.schedule_sha256,
                "checkpoint_selection": "final_update_only",
                "fresh_model_initialization": True,
                "predecessor_checkpoint_opens": 0,
            },
            "evaluation": evaluation,
            "gate_evaluation": gate_evaluation,
            "gate_adapter": (
                "v15_native_diagnostics_excluded_then_hierarchical_to_ordered_key_v1"
            ),
            "resource": reservation["resource"],
            "determinism": reservation["determinism"],
            "access_ledger": {
                "predecessor_checkpoint_opens": 0,
                "heldout_opens": 0,
                "g2_opens": 0,
                "navigation_opens": 0,
                "runtime_opens": 0,
                "production_opens": 0,
                "gpu1_uses": 0,
            },
            "licenses": {
                "development_checkpoint_creation_authorized": True,
                "checkpoint_use_authorized": False,
                "metric_verification_only_checkpoint_use_authorized": True,
                "retry_authorized": False,
                **_DOWNSTREAM_DENIALS,
            },
        }
    )
    result_raw = _write_json(directory / "result.json", result)
    result_binding = ladder.artifact_binding(
        "result.json",
        result_raw,
        content_sha256=result["content_sha256"],
    )
    completion = _self_hashed(
        {
            "schema": ladder.ROW_COMPLETION_SCHEMA,
            "status": "completed_training",
            "row": row_value,
            "source_review": review,
            "reservation": reservation_binding,
            "checkpoint": checkpoint_binding,
            "result": result_binding,
            "inventory": [
                "checkpoint.pt",
                "completed.json",
                "gate.json",
                "metric_verification.json",
                "reservation.json",
                "result.json",
            ],
            "retry_authorized": False,
            "licenses": {
                "checkpoint_use_authorized": False,
                "metric_verification_only_checkpoint_use_authorized": True,
                **_DOWNSTREAM_DENIALS,
            },
        }
    )
    completion_raw = _write_json(directory / "completed.json", completion)
    completion_binding = ladder.artifact_binding(
        "completed.json",
        completion_raw,
        content_sha256=completion["content_sha256"],
    )
    numeric = _numeric_gate(row, passes=numeric_passes)
    target_signature = {"synthetic_target_partition": row.key}
    metric = _self_hashed(
        {
            "schema": ladder.ROW_METRIC_SCHEMA,
            "status": "verified",
            "row": row_value,
            "source_review": review,
            "artifacts": {
                "reservation": reservation_binding,
                "checkpoint": checkpoint_binding,
                "result": result_binding,
                "completion": completion_binding,
            },
            "target_partition": result["target_partition"],
            "target_partition_signature": target_signature,
            "target_partition_signature_sha256": ladder.canonical_json_sha256(
                target_signature
            ),
            "recomputed_evaluation": evaluation,
            "recomputed_evaluation_sha256": ladder.canonical_json_sha256(
                evaluation
            ),
            "recomputed_gate_evaluation": gate_evaluation,
            "recomputed_gate_evaluation_sha256": ladder.canonical_json_sha256(
                gate_evaluation
            ),
            "numeric_gate": numeric,
            "verification": {
                "checkpoint_bytes_rehashed": True,
                "checkpoint_state_manifest_rehashed": True,
                "checkpoint_semantic_hash_recomputed": True,
                "fresh_model_strict_loaded": True,
                "matched_evaluation_recomputed": True,
                "wrong_rgb_evaluation_recomputed": True,
                "result_metrics_reused": False,
                "metric_repair_applied": False,
                "threshold_weakened": False,
            },
            "resource": _resource(),
            "determinism": reservation["determinism"],
            "access_ledger": {
                "checkpoint_opens": 1,
                "heldout_opens": 0,
                "g2_opens": 0,
                "navigation_opens": 0,
                "runtime_opens": 0,
                "production_opens": 0,
                "gpu1_uses": 0,
            },
            "retry_authorized": False,
            "licenses": {
                "checkpoint_use_authorized_for_metric_verification_only": True,
                "development_checkpoint_use_authorized": False,
                "new_model_output_authorized": False,
                **_DOWNSTREAM_DENIALS,
            },
        }
    )
    metric_raw = _write_json(directory / "metric_verification.json", metric)
    metric_binding = ladder.artifact_binding(
        "metric_verification.json",
        metric_raw,
        content_sha256=metric["content_sha256"],
    )
    core = {
        "schema": ladder.ROW_GATE_SCHEMA,
        "status": "passed" if numeric_passes else "failed_numeric_gate",
        "row": row_value,
        "source_review": review,
        "prerequisite_gates": prerequisites,
        "artifacts": {
            "reservation": reservation_binding,
            "checkpoint": checkpoint_binding,
            "result": result_binding,
            "completion": completion_binding,
            "metric_verification": metric_binding,
        },
        "threshold_contract_sha256": ladder.THRESHOLD_CONTRACT_SHA256,
        "numeric_gate": numeric,
        "check_count": ladder.EXPECTED_GATE_CHECK_COUNT,
        "failure_count": numeric["failure_count"],
        "passes": numeric_passes,
        "retry_authorized": False,
    }
    gate = _self_hashed(core)
    raw = _write_json(directory / "gate.json", gate)
    return ladder.artifact_binding(
        f"rows/{row.key}/gate.json",
        raw,
        content_sha256=gate["content_sha256"],
    )


def test_row_table_is_exact_fixed_compute_topology() -> None:
    assert [(row.seed, row.fit_size) for row in ladder.LADDER_ROWS] == [
        (20260710, 5),
        (20260710, 16),
        (20260710, 32),
        (20260710, 320),
        (20260711, 5),
        (20260711, 16),
        (20260711, 32),
        (20260711, 320),
    ]
    assert all(
        (row.updates, row.batch_size, row.frame_exposures) == (4000, 5, 20000)
        for row in ladder.LADDER_ROWS
    )
    assert len({row.schedule_sha256 for row in ladder.LADDER_ROWS}) == 8
    assert all(ladder.is_sha256(row.schedule_sha256) for row in ladder.LADDER_ROWS)


def test_numeric_gate_has_exact_26_name_comparator_threshold_contract() -> None:
    assert len(_NUMERIC_THRESHOLD_KEYS) == ladder.EXPECTED_GATE_CHECK_COUNT == 26
    for row in ladder.LADDER_ROWS[:4]:
        thresholds = asdict(retained_gate.FIT_THRESHOLDS[row.fit_size])
        expected = {
            name: (
                (
                    "less_than_or_equal"
                    if name in _MAXIMUM_NUMERIC_CHECKS
                    else "greater_than_or_equal"
                ),
                float(thresholds[threshold_name]),
            )
            for name, threshold_name in _NUMERIC_THRESHOLD_KEYS.items()
        }
        assert ladder.expected_numeric_check_contract(thresholds) == expected
        ladder.validate_numeric_gate(_numeric_gate(row), row=row)


@pytest.mark.parametrize("mutation", ["name", "comparison", "threshold"])
def test_numeric_gate_rejects_check_contract_mutation(mutation: str) -> None:
    row = ladder.LADDER_ROWS[0]
    numeric = json.loads(json.dumps(_numeric_gate(row)))
    check = numeric["checks"][0]
    if mutation == "name":
        check["name"] = "forbidden.synthetic_check"
    elif mutation == "comparison":
        check["comparison"] = "less_than_or_equal"
    else:
        check["threshold"] += 0.125
    with pytest.raises(PermissionError, match="numeric check"):
        ladder.validate_numeric_gate(numeric, row=row)


def test_initialization_identity_is_attempt_bound_not_state_fabricated() -> None:
    same_state = "a" * 64
    attempts = {
        row.index: hashlib.sha256(f"attempt:{row.index}".encode()).hexdigest()
        for row in ladder.LADDER_ROWS
    }
    identities = {
        ladder.initialization_identity(row, attempts[row.index], same_state)
        for row in ladder.LADDER_ROWS
    }
    assert len(identities) == 8
    assert ladder.initialization_identity(
        ladder.LADDER_ROWS[0], attempts[0], same_state
    ) == (
        ladder.initialization_identity(ladder.LADDER_ROWS[0], attempts[0], same_state)
    )


def test_derive_next_requires_contiguous_passing_rows(tmp_path: Path) -> None:
    root = tmp_path / "ladder"
    review = _review_binding()
    assert ladder.derive_next_row(
        output_root=root, expected_source_review=review
    ) == ladder.LADDER_ROWS[0]
    rows_root = root / "rows"
    rows_root.mkdir(parents=True)
    passed: list[dict[str, Any]] = []
    for index, row in enumerate(ladder.LADDER_ROWS):
        assert ladder.derive_next_row(
            output_root=root, expected_source_review=review
        ) == row
        prerequisites = ladder._expected_prerequisite_gates(row, passed)
        passed.append(
            _materialize_passing_row(rows_root, row, review, prerequisites)
        )
        expected = ladder.LADDER_ROWS[index + 1] if index < 7 else None
        assert ladder.derive_next_row(
            output_root=root, expected_source_review=review
        ) == expected


def test_final_gate_binds_exactly_all_eight_passing_rows(tmp_path: Path) -> None:
    root = tmp_path / "ladder"
    rows = root / "rows"
    rows.mkdir(parents=True)
    review = _review_binding()
    row_gates: list[dict[str, Any]] = []
    for row in ladder.LADDER_ROWS:
        prerequisites = ladder._expected_prerequisite_gates(row, row_gates)
        row_gates.append(
            _materialize_passing_row(rows, row, review, prerequisites)
        )

    with pytest.raises(PermissionError, match="all eight"):
        ladder.publish_final_gate(
            output_root=root,
            expected_source_review=review,
            expected_row_gates=row_gates[:-1],
        )
    gate, raw = ladder.publish_final_gate(
        output_root=root,
        expected_source_review=review,
        expected_row_gates=row_gates,
    )
    assert set(gate) == {
        "schema",
        "status",
        "source_review",
        "rows",
        "row_gates",
        "threshold_contract_sha256",
        "row_count",
        "all_rows_passed",
        "retry_authorized",
        "licenses",
        "content_sha256",
    }
    assert gate["schema"] == ladder.FINAL_GATE_SCHEMA
    assert gate["status"] == "all_eight_rows_passed"
    assert gate["source_review"] == review
    assert gate["rows"] == ladder.row_contract()
    assert gate["threshold_contract_sha256"] == ladder.THRESHOLD_CONTRACT_SHA256
    assert gate["row_count"] == 8
    assert gate["all_rows_passed"] is True
    assert gate["row_gates"] == row_gates
    assert gate["retry_authorized"] is False
    assert not any(gate["licenses"].values())
    assert raw == (root / ladder.FINAL_GATE_FILENAME).read_bytes()
    ladder.validate_final_gate(
        root / ladder.FINAL_GATE_FILENAME,
        expected_source_review=review,
        expected_row_gates=row_gates,
    )
    assert ladder.derive_next_row(
        output_root=root, expected_source_review=review
    ) is None


def test_failed_or_incomplete_row_stops_ladder(tmp_path: Path) -> None:
    review = _review_binding()
    root = tmp_path / "ladder"
    row = ladder.LADDER_ROWS[0]
    directory = root / "rows" / row.key
    directory.mkdir(parents=True)
    (directory / "reservation.json").write_bytes(b"x")
    with pytest.raises(ladder.LadderStopped, match="incomplete"):
        ladder.derive_next_row(output_root=root, expected_source_review=review)

    reservation = _reservation(row, review, [])
    reservation_raw = _write_json(directory / "reservation.json", reservation)
    failure_core = {
        "schema": ladder.ROW_FAILURE_SCHEMA,
        "status": "failed",
        "row": asdict(row) | {"key": row.key},
        "source_review": review,
        "reservation": ladder.artifact_binding(
            "reservation.json",
            reservation_raw,
            content_sha256=reservation["content_sha256"],
        ),
        "failure_stage": "training_and_evaluation",
        "failure": {"class": "runtime", "code": "execution_failure"},
        "removed_owned_partials": [],
        "partial_artifacts_removed": True,
        "retry_authorized": False,
        "licenses": {
            "checkpoint_use_authorized": False,
            **_DOWNSTREAM_DENIALS,
        },
    }
    _write_json(directory / "failed.json", _self_hashed(failure_core))
    with pytest.raises(ladder.LadderStopped, match="terminal failed"):
        ladder.derive_next_row(output_root=root, expected_source_review=review)


def test_failed_numeric_gate_is_preserved_and_never_retried(tmp_path: Path) -> None:
    root = tmp_path / "ladder"
    rows = root / "rows"
    rows.mkdir(parents=True)
    review = _review_binding()
    row = ladder.LADDER_ROWS[0]
    _materialize_passing_row(rows, row, review, [], numeric_passes=False)

    with pytest.raises(ladder.LadderStopped, match="terminal numeric gate failure"):
        ladder.derive_next_row(output_root=root, expected_source_review=review)
    directory = rows / row.key
    assert sorted(path.name for path in directory.iterdir()) == [
        "checkpoint.pt",
        "completed.json",
        "gate.json",
        "metric_verification.json",
        "reservation.json",
        "result.json",
    ]
    assert not (directory / "failed.json").exists()
    for name in (
        "reservation.json",
        "completed.json",
        "metric_verification.json",
        "gate.json",
    ):
        value, _ = ladder.load_bound_json(directory / name)
        assert value["retry_authorized"] is False
    gate, _ = ladder.load_bound_json(directory / "gate.json")
    assert gate["status"] == "failed_numeric_gate"
    assert gate["failure_count"] == 1
    assert gate["passes"] is False


def test_completed_row_rehashes_actual_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "ladder"
    rows = root / "rows"
    rows.mkdir(parents=True)
    review = _review_binding()
    row = ladder.LADDER_ROWS[0]
    _materialize_passing_row(rows, row, review, [])
    assert ladder.derive_next_row(
        output_root=root,
        expected_source_review=review,
    ) == ladder.LADDER_ROWS[1]
    with (rows / row.key / "result.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(PermissionError, match="row result bytes changed"):
        ladder.derive_next_row(output_root=root, expected_source_review=review)


def test_internal_discovery_opens_candidate_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tmp_path / "rows"
    rows.mkdir()
    review = {"content_sha256": "a" * 64}
    review_raw = b"{}\n"
    review_binding = ladder.source_review_binding(review, review_raw)
    row = ladder.LADDER_ROWS[0]
    _materialize_passing_row(rows, row, review_binding, [])
    directory = rows / row.key
    (directory / "gate.json").unlink()
    (directory / "metric_verification.json").unlink()

    monkeypatch.setattr(ladder, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(ladder, "ROWS_ROOT", rows)
    monkeypatch.setattr(
        ladder,
        "validate_source_review",
        lambda _file_sha256: (review, review_raw),
    )
    original_read_regular = ladder.read_regular
    checkpoint_opens = 0

    def count_checkpoint_open(path: Path) -> bytes:
        nonlocal checkpoint_opens
        if path == directory / "checkpoint.pt":
            checkpoint_opens += 1
        return original_read_regular(path)

    monkeypatch.setattr(ladder, "read_regular", count_checkpoint_open)
    discovered, _review, _review_raw, bundle = (
        ladder.discover_internal_verification_bundle()
    )
    assert discovered == row
    assert bundle["checkpoint_raw"] == b"synthetic-checkpoint"
    assert checkpoint_opens == 1


def test_terminal_v16_validator_binds_hashes_semantics_and_inventories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v16"
    attempt = root / "attempts/seed_20260710/n5"
    attempt.mkdir(parents=True)
    (root / "gates").mkdir()
    (root / "metric_verifications").mkdir()
    lock = attempt.parent / ".n5.reservation-v15.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    reservation = _self_hashed(
        {
            "schema": ladder.V16_RESERVATION_SCHEMA,
            "scope": "one_exclusive_fresh_gate_aligned_raster_nll_v16_attempt",
            "seed": 20260710,
            "fit_size": 5,
        }
    )
    reservation_raw = _write_json(attempt / "reservation.json", reservation)
    reservation_receipt = {
        "path": "reservation.json",
        "file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "content_sha256": reservation["content_sha256"],
        "byte_count": len(reservation_raw),
    }
    failure = _self_hashed(
        {
            "schema": ladder.V16_FAILURE_SCHEMA,
            "status": "failed",
            "failure_stage": "training",
            "failure": {"class": "permission", "code": "scope_or_authorization_failure"},
            "reservation": reservation_receipt,
            "partial_artifacts_removed": True,
            "artifact_cleanup": [],
            "diagnostic_publication_succeeded": False,
            "verification_failure": None,
            "retry_authorized": False,
        }
    )
    failure_raw = _write_json(attempt / "failed.json", failure)
    visibility = _self_hashed(
        {
            "schema": (
                "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_"
                "nll_v16_gpu_visibility_preflight_v1"
            ),
            "status": "passed",
            "disposition": "pass_exactly_one_r9700",
            "zero_access_evidence": {"data": 0},
            "authority": {"training_authority": False},
            "selector_observation": {
                "hip_visible_devices": "0",
                "hsa_override_gfx_version": None,
                "conflicting_selectors": {"CUDA_VISIBLE_DEVICES": None},
            },
            "runtime_observation": {
                "enumeration_completed": True,
                "gpu1_absent": True,
                "ordered_devices": [
                    {"logical_ordinal": 0, "name": "AMD Radeon AI PRO R9700"}
                ],
                "raphael_absent": True,
                "runtime_available": True,
                "visible_device_count": 1,
            },
            "native_thread_observation": {
                "environment": {name: "1" for name in ladder.THREAD_ENVIRONMENT},
                "torch_inter_op": 1,
                "torch_intra_op": 1,
            },
        }
    )
    visibility_path = tmp_path / "visibility.json"
    visibility_raw = _write_json(visibility_path, visibility)
    bindings = {
        "reservation": {
            "path": "attempts/seed_20260710/n5/reservation.json",
            "file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
            "content_sha256": reservation["content_sha256"],
            "byte_count": len(reservation_raw),
        },
        "failure": {
            "path": "attempts/seed_20260710/n5/failed.json",
            "file_sha256": hashlib.sha256(failure_raw).hexdigest(),
            "content_sha256": failure["content_sha256"],
            "byte_count": len(failure_raw),
        },
        "lock": {
            "path": "attempts/seed_20260710/.n5.reservation-v15.lock",
            "file_sha256": hashlib.sha256(b"").hexdigest(),
            "byte_count": 0,
        },
        "attempt_inventory": ["failed.json", "reservation.json"],
        "seed_inventory": [".n5.reservation-v15.lock", "n5"],
        "root_inventory": ["attempts", "gates", "metric_verifications"],
        "visibility_receipt": {
            "path": str(visibility_path),
            "file_sha256": hashlib.sha256(visibility_raw).hexdigest(),
            "content_sha256": visibility["content_sha256"],
            "byte_count": len(visibility_raw),
        },
    }
    observed = ladder.validate_terminal_v16(
        root=root,
        bindings=bindings,
        visibility_path=visibility_path,
    )
    assert observed["numeric_outputs_observed_or_persisted"] is False
    (root / "gates/forbidden.json").write_text("{}", encoding="ascii")
    with pytest.raises(PermissionError, match="derived directory"):
        ladder.validate_terminal_v16(root=root, bindings=bindings)


def test_source_review_is_strict_and_hashes_complete_closure(tmp_path: Path) -> None:
    sources = ("runner.py", "test.py")
    proofs = ("proof.md",)
    runtime_paths = ("runtime/base.py", "runtime/gate.py")
    for relative in (*sources, *proofs, *runtime_paths):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(relative + "\n", encoding="ascii")
    source_map = {
        relative: {
            "path": relative,
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for relative in sources
    }
    proof_map = {
        relative: {
            "path": relative,
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for relative in proofs
    }
    runtime_bindings = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in runtime_paths
    }
    runtime_map = {
        relative: {"path": relative, "file_sha256": digest}
        for relative, digest in runtime_bindings.items()
    }
    original_sources = ladder.SUCCESSOR_SOURCES
    original_proofs = ladder.SUCCESSOR_PROOFS
    original_runtime = ladder.RUNTIME_SOURCE_BINDINGS
    try:
        ladder.SUCCESSOR_SOURCES = sources
        ladder.SUCCESSOR_PROOFS = proofs
        ladder.RUNTIME_SOURCE_BINDINGS = runtime_bindings
        core = {
            "schema": ladder.SOURCE_REVIEW_SCHEMA,
            "status": "different_agent_review_passed_lean_ladder_v1",
            "implementation_author": ladder.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/lean_ladder_reviewer",
            "review_completed": True,
            "source_closure_approved": True,
            "runtime_complete": True,
            "eight_serial_rows_authorized": True,
            "output_root": ladder.OUTPUT_ROOT_RELATIVE_PATH,
            "rows": ladder.row_contract(),
            "terminal_v16": ladder.TERMINAL_V16_BINDINGS,
            "successor_sources": source_map,
            "successor_proofs": proof_map,
            "runtime_sources": runtime_map,
            "science_contract": ladder.science_contract(),
            "licenses": ladder.REVIEW_LICENSES,
        }
        review = _self_hashed(core)
        raw = _write_json(tmp_path / "review.json", review)
        validated, _ = ladder.validate_source_review(
            hashlib.sha256(raw).hexdigest(),
            path=tmp_path / "review.json",
            root=tmp_path,
        )
        assert validated["runtime_complete"] is True
        forbidden = dict(review)
        forbidden["reviewer"] = ladder.IMPLEMENTATION_AUTHOR
        forbidden_core = dict(forbidden)
        forbidden_core.pop("content_sha256")
        forbidden = _self_hashed(forbidden_core)
        forbidden_raw = _write_json(tmp_path / "forbidden_review.json", forbidden)
        with pytest.raises(PermissionError, match="source review contract"):
            ladder.validate_source_review(
                hashlib.sha256(forbidden_raw).hexdigest(),
                path=tmp_path / "forbidden_review.json",
                root=tmp_path,
            )
    finally:
        ladder.SUCCESSOR_SOURCES = original_sources
        ladder.SUCCESSOR_PROOFS = original_proofs
        ladder.RUNTIME_SOURCE_BINDINGS = original_runtime


def test_cli_and_cpu_smoke_are_fail_closed_and_source_only() -> None:
    with pytest.raises(SystemExit):
        ladder.parse_args(["--next", "--seed", "20260710"])
    python = Path(sys.executable)
    completed = subprocess.run(
        [python, str(ROOT / ladder.RUNNER_RELATIVE_PATH), "--cpu-contract-smoke"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["runtime_complete"] is True


def test_exact_environment_requires_one_gpu_selector_and_six_thread_caps() -> None:
    environment = {
        "HIP_VISIBLE_DEVICES": "0",
        **{name: "1" for name in ladder.THREAD_ENVIRONMENT},
    }
    observed = ladder.validate_exact_process_environment(environment)
    assert observed["hip_visible_devices"] == "0"
    for forbidden in ladder.UNSET_DEVICE_SELECTORS:
        with pytest.raises(PermissionError, match="selectors"):
            ladder.validate_exact_process_environment(
                {**environment, forbidden: "0"}
            )


def test_runtime_sources_reproduce_all_schedule_hashes_and_checkpoint_contract() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("system test interpreter does not provide Torch")
    runtime = ladder.load_runtime_modules()
    for row in ladder.LADDER_ROWS:
        schedule = runtime.base._deterministic_training_batches(
            frame_count=row.fit_size,
            batch_size=row.batch_size,
            steps=row.updates,
            seed=row.seed,
        )
        assert runtime.base.canonical_json_sha256(schedule) == row.schedule_sha256
    model = runtime.base.ObservableCameraRayEvidenceV4Model()
    metadata = {"contract": "synthetic_cpu_only"}
    raw, content_sha256 = runtime.base._checkpoint_bytes(model, metadata=metadata)
    binding = ladder.artifact_binding(
        "checkpoint.pt",
        raw,
        content_sha256=content_sha256,
    )
    reloaded = ladder.validate_checkpoint_bytes(
        runtime,
        raw,
        expected_binding=binding,
        expected_metadata=metadata,
    )
    assert ladder.model_state_sha256(reloaded) == ladder.model_state_sha256(model)
