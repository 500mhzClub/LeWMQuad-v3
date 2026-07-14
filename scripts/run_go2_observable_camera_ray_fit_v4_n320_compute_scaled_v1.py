#!/usr/bin/env python3
"""One-shot, compute-scaled N320 successor to the terminal camera ladder row.

This is deliberately a single experiment, not another ladder.  It preserves the
frozen N320 data, objective, thresholds, seed, and batch size while increasing
training from 4,000 to 40,000 updates.  A fresh model is trained once and an
isolated child process reloads the resulting checkpoint before the terminal
numeric gate is published.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import (  # noqa: E402
    run_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1 as frozen,
)


OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1"
)
OUTPUT_ROOT = ROOT / OUTPUT_ROOT_RELATIVE_PATH
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1_"
    "prereg_review_2026-07-14.json"
)
REVIEW_PATH = ROOT / REVIEW_RELATIVE_PATH
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1.py"
)
IMPLEMENTATION_AUTHOR = "/root/n320_successor_impl"
REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1_review_v1"
)
SCHEMA_PREFIX = "lewm_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
METRIC_SCHEMA = f"{SCHEMA_PREFIX}_metric_verification_v1"
GATE_SCHEMA = f"{SCHEMA_PREFIX}_gate_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
SUMMARY_SCHEMA = f"{SCHEMA_PREFIX}_execution_summary_v1"
SUCCESS_INVENTORY = [
    "checkpoint.pt",
    "gate.json",
    "metric_verification.json",
    "reservation.json",
    "result.json",
]
FAILURE_INVENTORY = ["failed.json", "reservation.json"]
PREVERIFY_INVENTORY = ["checkpoint.pt", "reservation.json", "result.json"]

OLD_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1.py"
)
OLD_RUNNER_FILE_SHA256 = (
    "c70c07ab8e2f7de68369ade38a2da87cf66d76825a37613048e5dddc85f08eb7"
)
OLD_REVIEW_BINDING = {
    "path": frozen.SOURCE_REVIEW_RELATIVE_PATH,
    "file_sha256": "0b539d93f25ebf60d34cba9419b232d3d236c2b18bdd60f6a3ca1390ba51e961",
    "content_sha256": "81e8459b8274bce6e40893b8312be9e99f2269db90a24770b24d728017da296a",
    "byte_count": 11610,
}
PREDECESSOR_ROOT_RELATIVE_PATH = frozen.OUTPUT_ROOT_RELATIVE_PATH
PREDECESSOR_GATE_BINDING = {
    "path": "rows/row_03_seed_20260710_n320/gate.json",
    "file_sha256": "2e26b0081d51dcc19f671962b9ab00cd57f825800b7dc0b3ab65c0401b25d003",
    "content_sha256": "63fb0578cdc2d2cae1bf30aa4acba9f58dbe1c52278340770c7d6ae6d4c84711",
    "byte_count": 8099,
}
PREDECESSOR_RESERVATION_BINDING = {
    "path": "reservation.json",
    "file_sha256": "0241955da9257792ca5ffc7dceb1d45bd712ea27157c169e041dbe572b1cf347",
    "content_sha256": "0cce07f7d6b3a2310f91e3116be202b0b0b864f70305fa1369caaa5652883222",
    "byte_count": 27022,
}
PREDECESSOR_CHECKPOINT_BINDING = {
    "path": "checkpoint.pt",
    "file_sha256": "b0f5cc9105cc945bb9d3a6e68a8cf129f8467c661fee21cb3b4a0f3c8f431ab3",
    "content_sha256": "0097b825fb74e3e06abbc54c8d3cb45fd835717c750028eed6f72862dea9ff56",
    "byte_count": 13776908,
}
EXPECTED_INITIAL_STATE_SHA256 = (
    "a03f76eb539480ecb19ed4331ca4dc70eb1b3cba9f1453add4dcdc586a5ae1d2"
)


@dataclass(frozen=True)
class ComputeRow:
    seed: int = 20260710
    fit_size: int = 320
    updates: int = 40_000
    batch_size: int = 5
    frame_exposures: int = 200_000
    schedule_sha256: str = (
        "54cf287353be8942706c6904ef5d39bf227c4eeb37c1f5065a21eea8da1a7117"
    )
    first_4000_schedule_sha256: str = (
        "4084f8d5c14989cb76df4f01e4de46b0b6a88537ba607ccc4152795304bc3bd6"
    )

    @property
    def key(self) -> str:
        return "seed_20260710_n320_compute_scaled"


ROW = ComputeRow()


def row_contract() -> dict[str, Any]:
    return asdict(ROW) | {"key": ROW.key}


def predecessor_contract() -> dict[str, Any]:
    return {
        "root": PREDECESSOR_ROOT_RELATIVE_PATH,
        "old_source_review": OLD_REVIEW_BINDING,
        "terminal_row_index": 3,
        "terminal_status": "failed_numeric_gate",
        "gate": PREDECESSOR_GATE_BINDING,
        "reservation": PREDECESSOR_RESERVATION_BINDING,
        "checkpoint_bound_not_loaded": PREDECESSOR_CHECKPOINT_BINDING,
    }


def science_contract() -> dict[str, Any]:
    inherited = frozen.science_contract()
    return {
        "row": row_contract(),
        "fresh_model_initialization": True,
        "predecessor_checkpoint_opens": 0,
        "optimizer": inherited["optimizer"],
        "learning_rate": inherited["learning_rate"],
        "weight_decay": inherited["weight_decay"],
        "gradient_clip_norm": inherited["gradient_clip_norm"],
        "precision": inherited["precision"],
        "autocast": inherited["autocast"],
        "objective": inherited["objective"],
        "loss_components": inherited["loss_components"],
        "loss_weights": inherited["loss_weights"],
        "wrong_rgb_mapping": inherited["wrong_rgb_mapping"],
        "evaluation_batch_size": inherited["evaluation_batch_size"],
        "data_bindings": inherited["data_bindings"],
        "subset_content_sha256": frozen.SUBSET_CONTENT_SHA256[320],
        "target_partition_content_sha256": (
            frozen.TARGET_PARTITION_CONTENT_SHA256[320]
        ),
        "threshold_contract_sha256": frozen.THRESHOLD_CONTRACT_SHA256,
        "row_threshold_sha256": frozen.ROW_THRESHOLD_SHA256[320],
        "maximum_attempts": 1,
        "retry_authorized": False,
        "shared_v5_development_use": "authorized_only_if_numeric_gate_passes",
    }


REVIEW_LICENSES = {
    "one_development_attempt_authorized": True,
    "data_change_authorized": False,
    "threshold_change_authorized": False,
    "predecessor_checkpoint_model_use_authorized": False,
    "shared_v5_development_use_authorized_only_on_pass": True,
    "g2_authorized": False,
    "navigation_authorized": False,
    "heldout_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
}


def result_licenses() -> dict[str, bool]:
    return {
        "checkpoint_creation_authorized": True,
        "shared_v5_development_use_requires_passed_gate": True,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
    }


def metric_licenses() -> dict[str, bool]:
    return {
        "checkpoint_use_authorized_for_metric_verification_only": True,
        "shared_v5_development_use_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
    }


def gate_licenses(passes: bool) -> dict[str, bool]:
    return {
        "shared_v5_development_use_authorized": passes,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
    }


def _binding(path: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return frozen.artifact_binding(
        path, raw, content_sha256=str(value["content_sha256"])
    )


def _review_binding(review: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return _binding(REVIEW_RELATIVE_PATH, review, raw)


def review_core(root: Path = ROOT) -> dict[str, Any]:
    """Return the exact unhashed object an independent reviewer must approve."""
    return {
        "schema": REVIEW_SCHEMA,
        "status": "different_agent_review_passed_n320_compute_scaled_v1",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": None,
        "review_completed": True,
        "source_closure_approved": True,
        "runtime_complete": True,
        "one_attempt_authorized": True,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "row": row_contract(),
        "terminal_predecessor": predecessor_contract(),
        "successor_sources": {
            relative: {
                "path": relative,
                "file_sha256": hashlib.sha256(
                    frozen.read_regular(root / relative)
                ).hexdigest(),
            }
            for relative in (RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
        },
        "reused_ladder_source": {
            "path": OLD_RUNNER_RELATIVE_PATH,
            "file_sha256": OLD_RUNNER_FILE_SHA256,
        },
        "science_contract": science_contract(),
        "licenses": REVIEW_LICENSES,
    }


def validate_review(
    file_sha256: str,
    *,
    path: Path = REVIEW_PATH,
    root: Path = ROOT,
) -> tuple[dict[str, Any], bytes]:
    review, raw = frozen.load_bound_json(path, file_sha256=file_sha256)
    expected = review_core(root)
    reviewer = review.get("reviewer")
    expected["reviewer"] = reviewer
    if (
        type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or review != frozen._self_hashed(expected)
    ):
        raise PermissionError("N320 compute-scaled review contract changed")
    if hashlib.sha256(frozen.read_regular(root / OLD_RUNNER_RELATIVE_PATH)).hexdigest() != OLD_RUNNER_FILE_SHA256:
        raise PermissionError("reused committed ladder source changed")
    return review, raw


def validate_terminal_predecessor() -> dict[str, Any]:
    """Validate the committed old review and complete rows 0--3 byte chain."""
    old_review, old_raw = frozen.validate_source_review(
        OLD_REVIEW_BINDING["file_sha256"]
    )
    old_binding = frozen.source_review_binding(old_review, old_raw)
    if old_binding != OLD_REVIEW_BINDING:
        raise PermissionError("old ladder review binding changed")
    passed: list[dict[str, Any]] = []
    terminal_gate: dict[str, Any] | None = None
    terminal_binding: dict[str, Any] | None = None
    for row in frozen.LADDER_ROWS[:4]:
        prerequisites = frozen._expected_prerequisite_gates(row, passed)
        gate, gate_binding = frozen.validate_completed_row_bundle(
            frozen.ROWS_ROOT / row.key,
            row=row,
            expected_source_review=old_binding,
            expected_prerequisite_gates=prerequisites,
        )
        if row.index < 3:
            if gate.get("passes") is not True:
                raise PermissionError("terminal predecessor has an early miss")
            passed.append(gate_binding)
        else:
            terminal_gate, terminal_binding = gate, gate_binding
    assert terminal_gate is not None and terminal_binding is not None
    artifacts = terminal_gate.get("artifacts", {})
    if (
        terminal_binding != PREDECESSOR_GATE_BINDING
        or terminal_gate.get("status") != "failed_numeric_gate"
        or terminal_gate.get("passes") is not False
        or artifacts.get("reservation") != PREDECESSOR_RESERVATION_BINDING
        or artifacts.get("checkpoint") != PREDECESSOR_CHECKPOINT_BINDING
    ):
        raise PermissionError("terminal row-3 miss binding changed")
    return {
        **predecessor_contract(),
        "validated_row_gate_chain": [*passed, terminal_binding],
        "predecessor_checkpoint_opens": 0,
    }


def _attempt_identity(review: Mapping[str, Any]) -> str:
    return frozen.canonical_json_sha256(
        {
            "schema": f"{SCHEMA_PREFIX}_attempt_identity_v1",
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "row": row_contract(),
            "source_review": dict(review),
            "terminal_predecessor": predecessor_contract(),
            "attempt_index": 1,
            "maximum_attempts": 1,
        }
    )


def _json_binding(name: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return _binding(name, value, raw)


def _publish_json(name: str, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = frozen._self_hashed(core)
    raw = frozen._json_payload(value)
    frozen._write_exclusive(OUTPUT_ROOT / name, raw)
    return value, raw


def reserve(
    *,
    review: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    inputs: Any,
    target_partition: Mapping[str, Any],
    initialization: Mapping[str, Any],
    resource: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    core = {
        "schema": RESERVATION_SCHEMA,
        "status": "reserved",
        "row": row_contract(),
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": initialization["attempt_identity"],
        "source_review": dict(review),
        "terminal_predecessor": dict(predecessor),
        "science_contract_sha256": frozen.canonical_json_sha256(science_contract()),
        "inputs": {
            "data_bindings": frozen.DATA_BINDINGS,
            "subset": inputs.subset_receipt,
            "target_partition": dict(target_partition),
        },
        "initialization": dict(initialization),
        "resource": dict(resource),
        "determinism": dict(determinism),
        "retry_authorized": False,
        "licenses": REVIEW_LICENSES,
    }
    value = frozen._self_hashed(core)
    raw = frozen._json_payload(value)
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(OUTPUT_ROOT, 0o700)
    try:
        frozen._write_exclusive(OUTPUT_ROOT / "reservation.json", raw)
        frozen._fsync_directory(OUTPUT_ROOT)
    except BaseException as error:
        if (OUTPUT_ROOT / "reservation.json").exists():
            try:
                terminate_failure(
                    review=review,
                    reservation=value,
                    reservation_raw=raw,
                    stage="reservation_commit",
                    error=error,
                )
            except BaseException as terminal_error:
                raise RuntimeError(
                    "reservation commit and terminalization both failed"
                ) from terminal_error
        else:
            os.rmdir(OUTPUT_ROOT)
        raise
    return value, raw


def _failure_class(error: BaseException) -> dict[str, str]:
    return frozen._failure_classification(error)


def terminate_failure(
    *,
    review: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    removed = []
    for name in ("gate.json", "metric_verification.json", "result.json", "checkpoint.pt"):
        path = OUTPUT_ROOT / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise PermissionError("owned successor partial is not regular")
            path.unlink()
            removed.append(name)
    failed, _raw = _publish_json(
        "failed.json",
        {
            "schema": FAILURE_SCHEMA,
            "status": "failed",
            "row": row_contract(),
            "source_review": dict(review),
            "reservation": _json_binding(
                "reservation.json", reservation, reservation_raw
            ),
            "failure_stage": stage,
            "failure": _failure_class(error),
            "removed_owned_partials": sorted(removed),
            "partial_artifacts_removed": True,
            "retry_authorized": False,
            "licenses": {
                "shared_v5_development_use_authorized": False,
                "g2_authorized": False,
                "navigation_authorized": False,
                "heldout_authorized": False,
                "production_authorized": False,
                "promotion_authorized": False,
            },
        },
    )
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != FAILURE_INVENTORY:
        raise RuntimeError("infrastructure failure inventory changed")
    frozen._fsync_directory(OUTPUT_ROOT)
    return failed


def checkpoint_metadata(
    *,
    review: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": {
            "row": row_contract(),
            "science_contract_sha256": frozen.canonical_json_sha256(
                science_contract()
            ),
        },
        "source_review": dict(review),
        "attempt_reservation": dict(reservation_binding),
        "initialization": reservation["initialization"],
        "subset_content_sha256": frozen.SUBSET_CONTENT_SHA256[320],
        "target_partition_content_sha256": (
            frozen.TARGET_PARTITION_CONTENT_SHA256[320]
        ),
        "training_schedule_sha256": ROW.schedule_sha256,
        "training_schedule_first_4000_sha256": ROW.first_4000_schedule_sha256,
        "checkpoint_selection": "final_update_only",
        "loss_contract": {
            "version": "gate_aligned_raster_nll_v15",
            "components": list(frozen.LOSS_COMPONENTS),
            "weights": {name: 0.25 for name in frozen.LOSS_COMPONENTS},
            "retained_v11_components": list(frozen.RETAINED_LOSS_COMPONENTS),
            "predecessor_checkpoint_input": False,
        },
    }


def _scoped_access_is_zero(access: object) -> bool:
    return type(access) is dict and all(
        access.get(name) == 0
        for name in (
            "heldout_opens", "g2_opens", "navigation_opens",
            "production_opens", "gpu1_uses",
        )
    )


def validate_reservation_record(
    value: object,
    *,
    review: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema", "status", "row", "attempt_index", "maximum_attempts",
        "attempt_identity", "source_review", "terminal_predecessor",
        "science_contract_sha256", "inputs", "initialization", "resource",
        "determinism", "retry_authorized", "licenses", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("reservation fields changed")
    inputs = value.get("inputs")
    initialization = value.get("initialization")
    resource = value.get("resource")
    determinism = value.get("determinism")
    if (
        value.get("schema") != RESERVATION_SCHEMA
        or value.get("status") != "reserved"
        or value.get("row") != row_contract()
        or value.get("attempt_index") != 1
        or value.get("maximum_attempts") != 1
        or not frozen.is_sha256(value.get("attempt_identity"))
        or value.get("source_review") != dict(review)
        or value.get("terminal_predecessor") != dict(predecessor)
        or value.get("science_contract_sha256")
        != frozen.canonical_json_sha256(science_contract())
        or type(inputs) is not dict
        or set(inputs) != {"data_bindings", "subset", "target_partition"}
        or inputs.get("data_bindings") != frozen.DATA_BINDINGS
        or inputs.get("subset", {}).get("content_sha256")
        != frozen.SUBSET_CONTENT_SHA256[320]
        or inputs.get("target_partition", {}).get("content_sha256")
        != frozen.TARGET_PARTITION_CONTENT_SHA256[320]
        or type(initialization) is not dict
        or set(initialization) != {
            "attempt_identity", "initial_state_sha256", "initialization_identity",
            "fresh_model_construction", "predecessor_checkpoint_opens",
        }
        or initialization.get("attempt_identity") != value.get("attempt_identity")
        or initialization.get("initial_state_sha256") != EXPECTED_INITIAL_STATE_SHA256
        or initialization.get("fresh_model_construction") is not True
        or initialization.get("predecessor_checkpoint_opens") != 0
        or initialization.get("initialization_identity")
        != frozen.canonical_json_sha256(
            {
                "schema": f"{SCHEMA_PREFIX}_initialization_identity_v1",
                "attempt_identity": value["attempt_identity"],
                "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
                "fresh_model_construction": True,
            }
        )
        or type(resource) is not dict
        or resource.get("device") != "cuda:0"
        or resource.get("visible_device_count") != 1
        or "r9700" not in str(resource.get("device_name", "")).casefold()
        or resource.get("native_thread_environment")
        != {name: "1" for name in frozen.THREAD_ENVIRONMENT}
        or resource.get("all_conflicting_selectors_unset") is not True
        or type(determinism) is not dict
        or determinism.get("seed") != ROW.seed
        or determinism.get("torch_num_threads") != 1
        or determinism.get("torch_num_interop_threads") != 1
        or value.get("retry_authorized") is not False
        or value.get("licenses") != REVIEW_LICENSES
    ):
        raise PermissionError("reservation critical contract changed")
    return dict(value)


def validate_metric_record(
    value: object,
    *,
    review: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    target_partition: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    gate_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "row", "source_review", "artifacts",
        "target_partition", "target_partition_signature",
        "target_partition_signature_sha256", "recomputed_evaluation",
        "recomputed_evaluation_sha256", "recomputed_gate_evaluation",
        "recomputed_gate_evaluation_sha256", "numeric_gate", "verification",
        "resource", "determinism", "access_ledger", "retry_authorized",
        "licenses", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("metric verification fields changed")
    access = value.get("access_ledger")
    verification = value.get("verification")
    numeric = frozen.validate_numeric_gate(value.get("numeric_gate"), row=ROW)
    if (
        value.get("schema") != METRIC_SCHEMA
        or value.get("status") != "verified"
        or value.get("row") != row_contract()
        or value.get("source_review") != dict(review)
        or value.get("artifacts") != dict(artifacts)
        or (target_partition is not None and value.get("target_partition") != dict(target_partition))
        or value.get("target_partition_signature_sha256")
        != frozen.canonical_json_sha256(value.get("target_partition_signature"))
        or (evaluation is not None and value.get("recomputed_evaluation") != dict(evaluation))
        or value.get("recomputed_evaluation_sha256")
        != frozen.canonical_json_sha256(value.get("recomputed_evaluation"))
        or (gate_view is not None and value.get("recomputed_gate_evaluation") != dict(gate_view))
        or value.get("recomputed_gate_evaluation_sha256")
        != frozen.canonical_json_sha256(value.get("recomputed_gate_evaluation"))
        or verification
        != {
            "checkpoint_bytes_rehashed": True,
            "checkpoint_state_manifest_rehashed": True,
            "checkpoint_semantic_hash_recomputed": True,
            "fresh_model_strict_loaded": True,
            "matched_evaluation_recomputed": True,
            "wrong_rgb_evaluation_recomputed": True,
            "result_metrics_reused": False,
            "threshold_weakened": False,
        }
        or type(value.get("resource")) is not dict
        or value["resource"].get("native_thread_environment")
        != {name: "1" for name in frozen.THREAD_ENVIRONMENT}
        or type(access) is not dict
        or access.get("checkpoint_opens") != 1
        or access.get("predecessor_checkpoint_opens") != 0
        or not _scoped_access_is_zero(access)
        or value.get("retry_authorized") is not False
        or value.get("licenses") != metric_licenses()
        or numeric.get("check_count") != 26
    ):
        raise PermissionError("metric verification critical contract changed")
    return dict(value)


def validate_result_record(
    value: object,
    *,
    review: Mapping[str, Any],
    reservation: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema", "status", "row", "source_review", "artifacts",
        "attempt_identity", "subset", "target_partition", "initialization",
        "model", "training", "evaluation", "gate_evaluation", "resource",
        "determinism", "access_ledger", "licenses", "retry_authorized",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("result fields changed")
    model = value.get("model")
    training = value.get("training")
    access = value.get("access_ledger")
    checkpoint = artifacts.get("checkpoint")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("status") != "completed_training"
        or value.get("row") != row_contract()
        or value.get("source_review") != dict(review)
        or value.get("artifacts") != dict(artifacts)
        or value.get("attempt_identity") != reservation.get("attempt_identity")
        or value.get("subset") != reservation["inputs"]["subset"]
        or value.get("target_partition") != reservation["inputs"]["target_partition"]
        or value.get("initialization") != reservation.get("initialization")
        or model
        != {
            "class": "ObservableCameraRayEvidenceV4Model",
            "fresh_initialization": True,
            "parameter_count": 3_105_513,
            "checkpoint": {**dict(checkpoint), "development_only": True},
        }
        or type(training) is not dict
        or training.get("steps") != ROW.updates
        or training.get("batch_size") != ROW.batch_size
        or training.get("frame_exposures") != ROW.frame_exposures
        or training.get("schedule_sha256") != ROW.schedule_sha256
        or training.get("optimizer") != "AdamW"
        or training.get("learning_rate") != 1e-4
        or training.get("weight_decay") != 1e-4
        or training.get("gradient_clip_norm") != 1.0
        or training.get("precision") != "float32"
        or training.get("autocast") is not False
        or training.get("loss_weights")
        != {name: 0.25 for name in frozen.LOSS_COMPONENTS}
        or training.get("fresh_model_initialization") is not True
        or training.get("predecessor_checkpoint_opens") != 0
        or type(value.get("evaluation")) is not dict
        or set(value["evaluation"]) != {
            "matched_rgb", "wrong_rgb_with_target_calibration"
        }
        or value.get("gate_evaluation")
        != frozen.gate_evaluation_view(value["evaluation"])
        or value.get("resource") != reservation.get("resource")
        or type(value.get("determinism")) is not dict
        or type(access) is not dict
        or access.get("selected_rgb_rehashes_before_publication") != ROW.fit_size
        or access.get("predecessor_checkpoint_opens") != 0
        or access.get("new_checkpoint_verifier_opens") != 1
        or not _scoped_access_is_zero(access)
        or value.get("licenses") != result_licenses()
        or value.get("retry_authorized") is not False
    ):
        raise PermissionError("result critical contract changed")
    return dict(value)


def publish_training_bundle(
    *,
    checkpoint_raw: bytes,
    checkpoint_content_sha256: str,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    review: Mapping[str, Any],
    result_fields: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    training_artifacts = {
        "reservation": _json_binding("reservation.json", reservation, reservation_raw),
        "checkpoint": frozen.artifact_binding(
            "checkpoint.pt", checkpoint_raw, content_sha256=checkpoint_content_sha256
        ),
    }
    result = frozen._self_hashed(
        {
            "schema": RESULT_SCHEMA,
            "status": "completed_training",
            "row": row_contract(),
            "source_review": dict(review),
            "artifacts": training_artifacts,
            **dict(result_fields),
            "retry_authorized": False,
        }
    )
    validate_result_record(
        result,
        review=review,
        reservation=reservation,
        artifacts=training_artifacts,
    )
    result_raw = frozen._json_payload(result)
    frozen._write_exclusive(OUTPUT_ROOT / "checkpoint.pt", checkpoint_raw)
    frozen._write_exclusive(OUTPUT_ROOT / "result.json", result_raw)
    frozen._fsync_directory(OUTPUT_ROOT)
    return {
        **training_artifacts,
        "result": _json_binding("result.json", result, result_raw),
    }


def _load_preverification_bundle() -> tuple[
    dict[str, Any], bytes, dict[str, Any], dict[str, Any], bytes, dict[str, Any]
]:
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise PermissionError("verifier output root is not a real directory")
    if sorted(path.name for path in OUTPUT_ROOT.iterdir()) != PREVERIFY_INVENTORY:
        raise PermissionError("verifier requires the sole ungated training bundle")
    reservation, reservation_raw = frozen.load_bound_json(
        OUTPUT_ROOT / "reservation.json"
    )
    review_claim = reservation.get("source_review")
    if type(review_claim) is not dict:
        raise PermissionError("reservation review binding changed")
    review, review_raw = validate_review(str(review_claim.get("file_sha256")))
    review_binding = _review_binding(review, review_raw)
    if review_claim != review_binding:
        raise PermissionError("reservation review bytes changed")
    validate_reservation_record(
        reservation,
        review=review_binding,
        predecessor=validate_terminal_predecessor(),
    )
    result, result_raw = frozen.load_bound_json(OUTPUT_ROOT / "result.json")
    checkpoint_raw = frozen.read_regular(OUTPUT_ROOT / "checkpoint.pt")
    checkpoint_claim = result.get("model", {}).get("checkpoint")
    if type(checkpoint_claim) is not dict:
        raise PermissionError("result checkpoint claim changed")
    checkpoint_binding = dict(checkpoint_claim)
    if checkpoint_binding.pop("development_only", None) is not True:
        raise PermissionError("result checkpoint scope changed")
    if frozen.artifact_binding(
        "checkpoint.pt", checkpoint_raw,
        content_sha256=str(checkpoint_binding.get("content_sha256")),
    ) != checkpoint_binding:
        raise PermissionError("result checkpoint byte claim changed")
    training_artifacts = {
        "reservation": _json_binding("reservation.json", reservation, reservation_raw),
        "checkpoint": checkpoint_binding,
    }
    validate_result_record(
        result,
        review=review_binding,
        reservation=reservation,
        artifacts=training_artifacts,
    )
    artifacts = {
        **training_artifacts,
        "result": _json_binding("result.json", result, result_raw),
    }
    return review, review_raw, reservation, result, checkpoint_raw, artifacts


def run_internal_verifier() -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("internal verifier requires python -I -B")
    frozen.validate_exact_process_environment()
    review, review_raw, reservation, result, checkpoint_raw, artifacts = (
        _load_preverification_bundle()
    )
    runtime = frozen.load_runtime_modules()
    determinism = frozen.configure_row_runtime(runtime, ROW.seed)
    resource = frozen.validate_live_resource(runtime)
    inputs, target_partition = frozen.load_row_inputs(runtime, ROW)
    review_binding = _review_binding(review, review_raw)
    if (
        reservation["inputs"]["subset"] != inputs.subset_receipt
        or reservation["inputs"]["target_partition"] != target_partition
    ):
        raise PermissionError("verifier input reproduction changed")
    images, rgb_access = frozen.serial_decode_selected_rgb(runtime, inputs.frames)
    metadata = checkpoint_metadata(
        review=review_binding,
        reservation_binding=artifacts["reservation"],
        reservation=reservation,
    )
    model = frozen.validate_checkpoint_bytes(
        runtime,
        checkpoint_raw,
        expected_binding=artifacts["checkpoint"],
        expected_metadata=metadata,
    )
    device = runtime.torch.device("cuda:0")
    model.to(device)
    with frozen.capture_compact_determinism_warnings(runtime.base) as collector:
        matched = frozen.evaluate_row_model(
            runtime, model=model, frames=inputs.frames, images=images,
            device=device, wrong_rgb=False, independent=True,
        )
        wrong = frozen.evaluate_row_model(
            runtime, model=model, frames=inputs.frames, images=images,
            device=device, wrong_rgb=True, independent=True,
        )
    evaluation = {
        "matched_rgb": matched,
        "wrong_rgb_with_target_calibration": wrong,
    }
    gate_view = frozen.gate_evaluation_view(evaluation)
    if (
        evaluation != result.get("evaluation")
        or gate_view != result.get("gate_evaluation")
    ):
        raise ValueError("isolated evaluation differs from parent result")
    signature, numeric = frozen.reconstruct_numeric_gate(
        runtime, row=ROW, gate_evaluation=gate_view
    )
    post_input = frozen.revalidate_row_inputs(runtime, inputs)
    metric = frozen._self_hashed(
        {
            "schema": METRIC_SCHEMA,
            "status": "verified",
            "row": row_contract(),
            "source_review": reservation["source_review"],
            "artifacts": artifacts,
            "target_partition": target_partition,
            "target_partition_signature": signature,
            "target_partition_signature_sha256": frozen.canonical_json_sha256(signature),
            "recomputed_evaluation": evaluation,
            "recomputed_evaluation_sha256": frozen.canonical_json_sha256(evaluation),
            "recomputed_gate_evaluation": gate_view,
            "recomputed_gate_evaluation_sha256": frozen.canonical_json_sha256(gate_view),
            "numeric_gate": numeric,
            "verification": {
                "checkpoint_bytes_rehashed": True,
                "checkpoint_state_manifest_rehashed": True,
                "checkpoint_semantic_hash_recomputed": True,
                "fresh_model_strict_loaded": True,
                "matched_evaluation_recomputed": True,
                "wrong_rgb_evaluation_recomputed": True,
                "result_metrics_reused": False,
                "threshold_weakened": False,
            },
            "resource": resource,
            "determinism": {**determinism, **collector.receipt()},
            "access_ledger": {
                **rgb_access,
                **post_input,
                "checkpoint_opens": 1,
                "predecessor_checkpoint_opens": 0,
                "heldout_opens": 0,
                "g2_opens": 0,
                "navigation_opens": 0,
                "production_opens": 0,
                "gpu1_uses": 0,
            },
            "retry_authorized": False,
            "licenses": metric_licenses(),
        }
    )
    validate_metric_record(
        metric,
        review=review_binding,
        artifacts=artifacts,
        target_partition=target_partition,
        evaluation=evaluation,
        gate_view=gate_view,
    )
    sys.stdout.buffer.write(frozen._json_payload(metric))
    sys.stdout.buffer.flush()
    return 0


def invoke_internal_verifier(
    artifacts: Mapping[str, Any],
    *,
    review: Mapping[str, Any],
    target_partition: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    gate_view: Mapping[str, Any],
) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / RUNNER_RELATIVE_PATH), "--internal-verify"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=7200,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            "isolated successor verifier failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    metric = frozen._parse_canonical_object_bytes(
        completed.stdout, name="N320 successor metric receipt"
    )
    return validate_metric_record(
        metric,
        review=review,
        artifacts=artifacts,
        target_partition=target_partition,
        evaluation=evaluation,
        gate_view=gate_view,
    )


def publish_metric_and_gate(
    *,
    review: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_raw = frozen._json_payload(metric)
    frozen._write_exclusive(OUTPUT_ROOT / "metric_verification.json", metric_raw)
    metric_binding = _json_binding("metric_verification.json", metric, metric_raw)
    all_artifacts = {
        **dict(artifacts), "metric_verification": metric_binding,
    }
    numeric = frozen.validate_numeric_gate(metric["numeric_gate"], row=ROW)
    gate, gate_raw = _publish_json(
        "gate.json",
        {
            "schema": GATE_SCHEMA,
            "status": "passed" if numeric["passes"] else "failed_numeric_gate",
            "row": row_contract(),
            "source_review": dict(review),
            "terminal_predecessor": predecessor_contract(),
            "artifacts": all_artifacts,
            "threshold_contract_sha256": frozen.THRESHOLD_CONTRACT_SHA256,
            "numeric_gate": numeric,
            "check_count": numeric["check_count"],
            "failure_count": numeric["failure_count"],
            "passes": numeric["passes"],
            "retry_authorized": False,
            "licenses": gate_licenses(bool(numeric["passes"])),
        },
    )
    frozen._fsync_directory(OUTPUT_ROOT)
    return gate, _json_binding("gate.json", gate, gate_raw)


def validate_terminal_bundle(review: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise PermissionError("successor output root is not a real directory")
    inventory = sorted(path.name for path in OUTPUT_ROOT.iterdir())
    if inventory == FAILURE_INVENTORY:
        failed, _raw = frozen.load_bound_json(OUTPUT_ROOT / "failed.json")
        if failed.get("schema") != FAILURE_SCHEMA or failed.get("retry_authorized") is not False:
            raise PermissionError("terminal infrastructure failure changed")
        raise RuntimeError("N320 compute-scaled attempt is terminally failed")
    if inventory != SUCCESS_INVENTORY:
        raise PermissionError("terminal successor inventory changed")
    gate, gate_raw = frozen.load_bound_json(OUTPUT_ROOT / "gate.json")
    if gate.get("schema") != GATE_SCHEMA or gate.get("source_review") != dict(review):
        raise PermissionError("terminal successor gate changed")
    artifacts = gate.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != {
        "reservation", "checkpoint", "result", "metric_verification"
    }:
        raise PermissionError("terminal successor artifact map changed")
    loaded: dict[str, Any] = {}
    for role, name in (
        ("reservation", "reservation.json"),
        ("result", "result.json"),
        ("metric_verification", "metric_verification.json"),
    ):
        value, raw = frozen.load_bound_json(OUTPUT_ROOT / name)
        if artifacts[role] != _json_binding(name, value, raw):
            raise PermissionError(f"terminal {role} binding changed")
        loaded[role] = value
    checkpoint_raw = frozen.read_regular(OUTPUT_ROOT / "checkpoint.pt")
    if frozen.artifact_binding(
        "checkpoint.pt", checkpoint_raw,
        content_sha256=str(artifacts["checkpoint"].get("content_sha256")),
    ) != artifacts["checkpoint"]:
        raise PermissionError("terminal checkpoint binding changed")
    reservation = loaded["reservation"]
    result = loaded["result"]
    metric = loaded["metric_verification"]
    claimed_predecessor = reservation.get("terminal_predecessor")
    validate_reservation_record(
        reservation,
        review=review,
        predecessor=(claimed_predecessor if type(claimed_predecessor) is dict else {}),
    )
    predecessor = validate_terminal_predecessor()
    if claimed_predecessor != predecessor:
        raise PermissionError("terminal predecessor chain changed")
    training_artifacts = {
        key: artifacts[key] for key in ("reservation", "checkpoint")
    }
    validate_result_record(
        result,
        review=review,
        reservation=reservation,
        artifacts=training_artifacts,
    )
    verification_artifacts = {
        **training_artifacts,
        "result": artifacts["result"],
    }
    validate_metric_record(
        metric,
        review=review,
        artifacts=verification_artifacts,
        target_partition=result["target_partition"],
        evaluation=result["evaluation"],
        gate_view=result["gate_evaluation"],
    )
    numeric = frozen.validate_numeric_gate(metric["numeric_gate"], row=ROW)
    gate_fields = {
        "schema", "status", "row", "source_review", "terminal_predecessor",
        "artifacts", "threshold_contract_sha256", "numeric_gate", "check_count",
        "failure_count", "passes", "retry_authorized", "licenses",
        "content_sha256",
    }
    if (
        set(gate) != gate_fields
        or gate.get("schema") != GATE_SCHEMA
        or gate.get("row") != row_contract()
        or gate.get("source_review") != dict(review)
        or gate.get("terminal_predecessor") != predecessor_contract()
        or gate.get("artifacts") != artifacts
        or gate.get("threshold_contract_sha256") != frozen.THRESHOLD_CONTRACT_SHA256
        or gate.get("numeric_gate") != numeric
        or gate.get("check_count") != 26
        or gate.get("failure_count") != numeric["failure_count"]
        or gate.get("passes") is not numeric["passes"]
        or gate.get("status")
        != ("passed" if numeric["passes"] else "failed_numeric_gate")
        or gate.get("retry_authorized") is not False
        or gate.get("licenses") != gate_licenses(bool(numeric["passes"]))
    ):
        raise PermissionError("terminal successor gate contract changed")
    return gate, _json_binding("gate.json", gate, gate_raw)


def run(review: Mapping[str, Any], review_raw: bytes) -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("exact successor execution requires python -I -B")
    frozen.validate_exact_process_environment()
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError("N320 compute-scaled attempt already has terminal state")
    review_binding = _review_binding(review, review_raw)
    predecessor = validate_terminal_predecessor()
    runtime = frozen.load_runtime_modules()
    determinism = frozen.configure_row_runtime(runtime, ROW.seed)
    inputs, target_partition = frozen.load_row_inputs(runtime, ROW)
    schedule = runtime.base._deterministic_training_batches(
        frame_count=ROW.fit_size,
        batch_size=ROW.batch_size,
        steps=ROW.updates,
        seed=ROW.seed,
    )
    if (
        runtime.base.canonical_json_sha256(schedule) != ROW.schedule_sha256
        or runtime.base.canonical_json_sha256(schedule[:4000])
        != ROW.first_4000_schedule_sha256
    ):
        raise PermissionError("compute-scaled schedule changed")
    attempt_identity = _attempt_identity(review_binding)
    model = runtime.base.ObservableCameraRayEvidenceV4Model()
    initial_state = frozen.model_state_sha256(model)
    if initial_state != EXPECTED_INITIAL_STATE_SHA256:
        raise PermissionError("fresh seeded initial model state changed")
    initialization = {
        "attempt_identity": attempt_identity,
        "initial_state_sha256": initial_state,
        "initialization_identity": frozen.canonical_json_sha256(
            {
                "schema": f"{SCHEMA_PREFIX}_initialization_identity_v1",
                "attempt_identity": attempt_identity,
                "initial_state_sha256": initial_state,
                "fresh_model_construction": True,
            }
        ),
        "fresh_model_construction": True,
        "predecessor_checkpoint_opens": 0,
    }
    resource = frozen.validate_live_resource(runtime)
    reservation, reservation_raw = reserve(
        review=review_binding,
        predecessor=predecessor,
        inputs=inputs,
        target_partition=target_partition,
        initialization=initialization,
        resource=resource,
        determinism=determinism,
    )
    stage = "selected_rgb_decode"
    committed = False
    try:
        images, rgb_access = frozen.serial_decode_selected_rgb(runtime, inputs.frames)
        device = runtime.torch.device("cuda:0")
        stage = "training_and_evaluation"
        with frozen.capture_compact_determinism_warnings(runtime.base) as collector:
            training = frozen.train_row_model(
                runtime, row=ROW, model=model, frames=inputs.frames,
                images=images, device=device,
            )
            matched = frozen.evaluate_row_model(
                runtime, model=model, frames=inputs.frames, images=images,
                device=device, wrong_rgb=False, independent=False,
            )
            wrong = frozen.evaluate_row_model(
                runtime, model=model, frames=inputs.frames, images=images,
                device=device, wrong_rgb=True, independent=False,
            )
        evaluation = {
            "matched_rgb": matched,
            "wrong_rgb_with_target_calibration": wrong,
        }
        gate_view = frozen.gate_evaluation_view(evaluation)
        parent_signature, parent_numeric = frozen.reconstruct_numeric_gate(
            runtime, row=ROW, gate_evaluation=gate_view
        )
        stage = "input_revalidation_and_checkpoint"
        post_input = frozen.revalidate_row_inputs(runtime, inputs)
        runtime.base._verify_file_commitments(
            tuple((frame.rgb_path, frame.image_sha256) for frame in inputs.frames),
            name="N320 compute-scaled selected RGB before publication",
        )
        reservation_binding = _json_binding(
            "reservation.json", reservation, reservation_raw
        )
        metadata = checkpoint_metadata(
            review=review_binding,
            reservation_binding=reservation_binding,
            reservation=reservation,
        )
        checkpoint_raw, checkpoint_content_sha256 = runtime.base._checkpoint_bytes(
            model, metadata=metadata
        )
        checkpoint_binding = frozen.artifact_binding(
            "checkpoint.pt",
            checkpoint_raw,
            content_sha256=checkpoint_content_sha256,
        )
        result_fields = {
            "attempt_identity": initialization["attempt_identity"],
            "subset": inputs.subset_receipt,
            "target_partition": dict(target_partition),
            "initialization": initialization,
            "model": {
                "class": "ObservableCameraRayEvidenceV4Model",
                "fresh_initialization": True,
                "parameter_count": 3_105_513,
                "checkpoint": {**checkpoint_binding, "development_only": True},
            },
            "training": training,
            "evaluation": evaluation,
            "gate_evaluation": gate_view,
            "resource": resource,
            "determinism": {**determinism, **collector.receipt()},
            "access_ledger": {
                **rgb_access, **post_input,
                "selected_rgb_rehashes_before_publication": len(inputs.frames),
                "predecessor_checkpoint_opens": 0,
                "new_checkpoint_verifier_opens": 1,
                "heldout_opens": 0, "g2_opens": 0, "navigation_opens": 0,
                "production_opens": 0, "gpu1_uses": 0,
            },
            "licenses": result_licenses(),
        }
        stage = "training_bundle_publication"
        artifacts = publish_training_bundle(
            checkpoint_raw=checkpoint_raw,
            checkpoint_content_sha256=checkpoint_content_sha256,
            reservation=reservation,
            reservation_raw=reservation_raw,
            review=review_binding,
            result_fields=result_fields,
        )
        model.to(runtime.torch.device("cpu"))
        del model
        runtime.torch.cuda.empty_cache()
        stage = "isolated_metric_verification"
        metric = invoke_internal_verifier(
            artifacts,
            review=review_binding,
            target_partition=target_partition,
            evaluation=evaluation,
            gate_view=gate_view,
        )
        if (
            metric.get("target_partition_signature") != parent_signature
            or metric.get("numeric_gate") != parent_numeric
            or metric.get("recomputed_evaluation") != evaluation
            or metric.get("recomputed_gate_evaluation") != gate_view
            or metric.get("target_partition") != target_partition
        ):
            raise RuntimeError("isolated verifier differs from parent reconstruction")
        stage = "metric_and_gate_publication"
        gate, gate_binding = publish_metric_and_gate(
            review=review_binding,
            artifacts=artifacts,
            metric=metric,
        )
        validated_gate, validated_binding = validate_terminal_bundle(review_binding)
        if gate != validated_gate or gate_binding != validated_binding:
            raise RuntimeError("terminal successor did not byte-revalidate")
        committed = True
        summary = frozen._self_hashed(
            {
                "schema": SUMMARY_SCHEMA,
                "status": "passed" if gate["passes"] else "failed_numeric_gate",
                "row": row_contract(),
                "gate": gate_binding,
                "shared_v5_development_use_authorized": bool(gate["passes"]),
                "retry_authorized": False,
            }
        )
        print(frozen.canonical_json_bytes(summary).decode("ascii"))
        return 0 if gate["passes"] else 3
    except BaseException as error:
        if committed:
            raise
        terminate_failure(
            review=review_binding,
            reservation=reservation,
            reservation_raw=reservation_raw,
            stage=stage,
            error=error,
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--internal-verify", action="store_true")
    modes.add_argument("--cpu-contract-smoke", action="store_true")
    parser.add_argument("--review-sha256")
    args = parser.parse_args(raw)
    if args.run:
        if raw != ["--run", "--review-sha256", args.review_sha256] or not frozen.is_sha256(args.review_sha256):
            raise ValueError("--run requires only the canonical review digest")
    elif args.internal_verify:
        if raw != ["--internal-verify"]:
            raise ValueError("internal verifier accepts no caller arguments")
    elif raw != ["--cpu-contract-smoke"]:
        raise ValueError("CPU smoke accepts no other arguments")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cpu_contract_smoke:
        print(
            frozen.canonical_json_bytes(
                {
                    "schema": f"{SCHEMA_PREFIX}_cpu_contract_smoke_v1",
                    "status": "passed",
                    "row": row_contract(),
                    "science_contract_sha256": frozen.canonical_json_sha256(
                        science_contract()
                    ),
                    "success_inventory": SUCCESS_INVENTORY,
                    "failure_inventory": FAILURE_INVENTORY,
                    "runtime_complete": True,
                }
            ).decode("ascii")
        )
        return 0
    if args.internal_verify:
        return run_internal_verifier()
    review, raw = validate_review(args.review_sha256)
    return run(review, raw)


if __name__ == "__main__":
    raise SystemExit(main())
