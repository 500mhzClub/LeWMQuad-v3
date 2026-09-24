#!/usr/bin/env python3
"""CPU-only reproduction of the immutable V4 N5 structural invalidation.

This diagnostic reads fixed artifacts, invokes the frozen pure-result validator,
and prints evidence. It never loads the checkpoint, imports torch, writes an
artifact, or grants execution authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate


SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_structural_invalidation_diagnostic_v1"
N5_ATTEMPT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2/"
    "attempts/seed_20260710/n5"
)
N5_ATTEMPT_PATH = ROOT / N5_ATTEMPT_RELATIVE_PATH
GATE_SOURCE_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py"
)
GATE_SOURCE_FILE_SHA256 = (
    "aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad"
)
REPRODUCTION_SOURCE_RELATIVE_PATH = (
    "scripts/diagnose_go2_observable_camera_ray_fit_v4_reproduction.py"
)
REPRODUCTION_SOURCE_FILE_SHA256 = (
    "74b95f0e70444704ced4f685c6e341311b76ebe9ec82a8357353da627da7517f"
)
LOSS_ABSOLUTE_TOLERANCE = 1e-9
LOSS_COMPONENTS = (
    "ordered_first_hit_nll",
    "target_bin_offset_smooth_l1",
    "ground_clear_distance_state_balanced_bce",
    "derived_raster_hierarchical_bce",
)
N5_ARTIFACTS: dict[str, dict[str, object]] = {
    "checkpoint.pt": {
        "byte_count": 13_778_252,
        "file_sha256": "f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0",
        "declared_content_sha256": "589060417903167bbf9ce7605c906b25cd802edd73b79ec607c77403c6df305a",
    },
    "completed.json": {
        "byte_count": 1_111,
        "file_sha256": "4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af",
        "content_sha256": "48022dca829a73b7cbd3b665ac7679807825a9aefd56a48e752ae07e6eaa336f",
    },
    "reservation.json": {
        "byte_count": 3_740,
        "file_sha256": "f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa",
        "content_sha256": "699b4e95ed05cb13a79fe6af8507fae5d987af9ff1977b0e4684f32742aa4943",
    },
    "result.json": {
        "byte_count": 27_102_689,
        "file_sha256": "39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa",
        "content_sha256": "8c38e13f411a5cd9b03362cb5ac98379875065f284a75ac894706944ff252b61",
    },
}
SORTED_VECTOR_DIFFERENCES = {
    "matched_rgb": {
        "immutable_result_sorted_values_sha256": (
            "a8ec842a10766b724b9ee4835c0e6866ce4b2323ccb7c33757c9f9d04ac20326"
        ),
        "stable_read_only_recomputed_sorted_values_sha256": (
            "6014597b1c286c42e5e7caa0643a98141b9545809c325a40763c82caf99d9f08"
        ),
    },
    "wrong_rgb_with_target_calibration": {
        "immutable_result_sorted_values_sha256": (
            "6ec4af60dd8f684bf6ef74339e4e439e7235d1a5fdf632aca0b79e77e95e1c86"
        ),
        "stable_read_only_recomputed_sorted_values_sha256": (
            "1e161762ff2158664cee260ff65b903864e14cce3c7bc09a405336140eee5ec8"
        ),
    },
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{name} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{name} changed while read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _verified_bytes(path: Path, expected: Mapping[str, object], *, name: str) -> bytes:
    raw = _read_regular(path, name=name)
    if len(raw) != expected["byte_count"]:
        raise ValueError(f"{name} byte count changed")
    if _sha256(raw) != expected["file_sha256"]:
        raise ValueError(f"{name} file SHA-256 changed")
    return raw


def _load_verified_json(name: str) -> dict[str, Any]:
    expected = N5_ARTIFACTS[name]
    raw = _verified_bytes(N5_ATTEMPT_PATH / name, expected, name=name)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON plus newline")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if declared != expected["content_sha256"]:
        raise ValueError(f"{name} declared content SHA-256 changed")
    if gate.canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content SHA-256 is invalid")
    return value


def _exception(call: Any) -> dict[str, object]:
    try:
        call()
    except Exception as exc:  # diagnostic preserves the frozen exception verbatim
        return {
            "passed": False,
            "exception": f"{type(exc).__name__}: {exc}",
        }
    return {"passed": True, "exception": None}


def _loss_finding(losses: Mapping[str, Any], *, validator_name: str) -> dict[str, object]:
    components = {name: float(losses[name]) for name in LOSS_COMPONENTS}
    stored_total = float(losses["total"])
    computed_total = 0.25 * sum(components[name] for name in LOSS_COMPONENTS)
    delta = stored_total - computed_total
    validation = _exception(
        lambda: gate._validate_evaluation_losses(losses, name=validator_name)
    )
    return {
        "components": components,
        "weight_per_component": 0.25,
        "stored_total": stored_total,
        "computed_quarter_component_sum": computed_total,
        "stored_minus_computed_delta": delta,
        "absolute_delta": abs(delta),
        "absolute_tolerance": LOSS_ABSOLUTE_TOLERANCE,
        "within_tolerance": math.isclose(
            stored_total,
            computed_total,
            rel_tol=0.0,
            abs_tol=LOSS_ABSOLUTE_TOLERANCE,
        ),
        "frozen_validator": validation,
    }


def build_diagnostic() -> dict[str, Any]:
    gate_raw = _read_regular(ROOT / GATE_SOURCE_RELATIVE_PATH, name="frozen gate")
    if _sha256(gate_raw) != GATE_SOURCE_FILE_SHA256:
        raise ValueError("frozen gate source SHA-256 changed")
    reproduction_raw = _read_regular(
        ROOT / REPRODUCTION_SOURCE_RELATIVE_PATH,
        name="read-only reproduction source",
    )
    if _sha256(reproduction_raw) != REPRODUCTION_SOURCE_FILE_SHA256:
        raise ValueError("read-only reproduction source SHA-256 changed")

    inventory = sorted(path.name for path in N5_ATTEMPT_PATH.iterdir())
    if inventory != sorted(N5_ARTIFACTS):
        raise PermissionError("immutable N5 attempt inventory changed")
    for name, expected in N5_ARTIFACTS.items():
        if name.endswith(".json"):
            continue
        _verified_bytes(N5_ATTEMPT_PATH / name, expected, name=name)

    reservation = _load_verified_json("reservation.json")
    result = _load_verified_json("result.json")
    completion = _load_verified_json("completed.json")
    matched_losses = result["evaluation"]["matched_rgb"]["losses"]
    wrong_losses = result["evaluation"][
        "wrong_rgb_with_target_calibration"
    ]["losses"]
    matched_finding = _loss_finding(matched_losses, validator_name="matched")
    wrong_finding = _loss_finding(wrong_losses, validator_name="wrong-RGB")

    original_full_validation = _exception(
        lambda: gate._validate_result(
            result,
            expected_seed=20260710,
            previous_stage_binding=None,
            seed_20260710_binding=None,
        )
    )

    counterfactual = copy.deepcopy(result)
    counterfactual_losses = counterfactual["evaluation"]["matched_rgb"]["losses"]
    counterfactual_losses["total"] = matched_finding[
        "computed_quarter_component_sum"
    ]
    counterfactual_core = dict(counterfactual)
    original_content_sha256 = counterfactual_core.pop("content_sha256")
    counterfactual["content_sha256"] = gate.canonical_json_sha256(
        counterfactual_core
    )
    counterfactual_validation = _exception(
        lambda: gate._validate_result(
            counterfactual,
            expected_seed=20260710,
            previous_stage_binding=None,
            seed_20260710_binding=None,
        )
    )

    for evaluation_name, difference in SORTED_VECTOR_DIFFERENCES.items():
        observed = result["evaluation"][evaluation_name]["metrics"][
            "pixel_hit_depth"
        ]["absolute_error_evidence"]["sorted_values_sha256"]
        if observed != difference["immutable_result_sorted_values_sha256"]:
            raise ValueError(f"{evaluation_name} immutable sorted-vector hash changed")

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "terminal_prepublication_structural_invalidation",
        "diagnostic_only": True,
        "cpu_only": True,
        "writes_artifacts": False,
        "n5_attempt": {
            "path": N5_ATTEMPT_RELATIVE_PATH,
            "inventory": inventory,
            "artifacts": N5_ARTIFACTS,
            "reservation_content_sha256": reservation["content_sha256"],
            "result_content_sha256": result["content_sha256"],
            "completion_content_sha256": completion["content_sha256"],
        },
        "validator": {
            "path": GATE_SOURCE_RELATIVE_PATH,
            "file_sha256": GATE_SOURCE_FILE_SHA256,
            "loss_absolute_tolerance": LOSS_ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0.0,
        },
        "structural_findings": {
            "matched_rgb": matched_finding,
            "wrong_rgb_with_target_calibration": wrong_finding,
            "immutable_full_result_validation": original_full_validation,
            "counterfactual_single_field_repair": {
                "counterfactual_only": True,
                "mutation_authorized": False,
                "changed_semantic_paths": [
                    "$.evaluation.matched_rgb.losses.total"
                ],
                "enclosing_content_sha256_recomputed": True,
                "original_content_sha256": original_content_sha256,
                "counterfactual_content_sha256": counterfactual[
                    "content_sha256"
                ],
                "replacement_total": counterfactual_losses["total"],
                "full_frozen_validator": counterfactual_validation,
                "other_failing_invariants_after_counterfactual": (
                    []
                    if counterfactual_validation["passed"] is True
                    else [counterfactual_validation["exception"]]
                ),
            },
        },
        "secondary_sorted_vector_findings": {
            "classification": "secondary_observation_not_an_exception",
            "reproduction_source": {
                "path": REPRODUCTION_SOURCE_RELATIVE_PATH,
                "file_sha256": REPRODUCTION_SOURCE_FILE_SHA256,
            },
            "stable_differences": SORTED_VECTOR_DIFFERENCES,
        },
        "authority": {
            "authoritative_metric_receipt": False,
            "checkpoint_use_authorized": False,
            "finalization_authorized": False,
            "g2_authorized": False,
            "holdout_authorized": False,
            "later_rung_authorized": False,
            "promotion_authorized": False,
            "runtime_authorized": False,
            "training_authorized": False,
        },
    }
    return {**core, "content_sha256": gate.canonical_json_sha256(core)}


def main() -> int:
    print(_canonical_json_bytes(build_diagnostic()).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
