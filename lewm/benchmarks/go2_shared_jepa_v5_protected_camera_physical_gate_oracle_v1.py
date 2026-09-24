"""Zero-parameter positive control for the protected-Camera physical gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/physical_gate_oracle_impl"
PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1"
CONTRACT_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py"
RUNNER_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py"
TEST_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py"
PREREG_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_preregistration_2026-07-15.md"
PRIOR_REVIEW_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_independent_review_2026-07-15.json"
PRIOR_REVIEW_SHA256 = "3f8b184a9c9e3c149dff61c137b2d2e4c1de3be5991a4fe9881b937a94b53938"
REVIEW_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_independent_review_v2_2026-07-15.json"
AUTHORIZATION_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_execution_authorization_2026-07-15.json"
OUTPUT_ROOT = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_physical_gate_oracle_v1"
RAW_ROOT = ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
RAW_MANIFEST_PATH = f"{RAW_ROOT}/manifest.json"
RAW_AUDIT_PATH = RAW_ROOT + ".audit_v13.json"
RAW_MANIFEST_FILE_SHA256 = "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
RAW_MANIFEST_CONTENT_SHA256 = "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
RAW_AUDIT_FILE_SHA256 = "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76"
RAW_AUDIT_CONTENT_SHA256 = "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca"
EVALUATOR_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
EVALUATOR_SHA256 = "9fd912538d94944881bd8a2789023470345208abb55a94466fbecc9d82afa0be"
EVIDENCE_PATH = "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
EVIDENCE_SHA256 = "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
SOURCE_PATHS = (CONTRACT_PATH, RUNNER_PATH, TEST_PATH, PREREG_PATH, PRIOR_REVIEW_PATH, EVALUATOR_PATH, EVIDENCE_PATH)


def _load(relative: str, name: str, digest: str) -> ModuleType:
    path, raw = ROOT / relative, (ROOT / relative).read_bytes()
    if path.is_symlink() or not path.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen oracle dependency changed: {relative}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load oracle dependency: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load(EVALUATOR_PATH, "_lewm_oracle_camera_v3", EVALUATOR_SHA256)
EVIDENCE = _load(EVIDENCE_PATH, "_lewm_oracle_evidence_v4", EVIDENCE_SHA256)
CHECKPOINT_EVALUATOR = EVALUATOR._v1
FAMILIES = tuple(CHECKPOINT_EVALUATOR.FAMILIES)
SCOPES = tuple(CHECKPOINT_EVALUATOR.SCOPES)
if SCOPES != ("aggregate", *FAMILIES) or len(SCOPES) != 9:
    raise RuntimeError("frozen scope order changed")

PAIR_COUNT, ENDPOINT_COUNT, SCENE_COUNT = 495, 924, 8
MARGINS_PER_SCOPE, MARGIN_COUNT = 21, 189
REVIEW_SCHEMA, AUTHORIZATION_SCHEMA = f"{PREFIX}_independent_review_v1", f"{PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA, ACCESS_SCHEMA = f"{PREFIX}_reservation_v1", f"{PREFIX}_access_v1"
RESULT_SCHEMA, COMPLETION_SCHEMA, FAILURE_SCHEMA = f"{PREFIX}_result_v1", f"{PREFIX}_completion_v1", f"{PREFIX}_failure_v1"
SUCCESS_PATHS, FAILURE_PATHS = ("access.json", "completed.json", "reservation.json", "result.json"), ("access.json", "failed.json", "reservation.json")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
ACCELERATOR_ENV = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "HSA_VISIBLE_DEVICES")
ARRAYS = (
    ("camera_origin_body_m.f4", "<f4", (3,)), ("camera_basis_body_fru.f4", "<f4", (3, 3)),
    ("ground_plane_z_body_m.f4", "<f4", ()), ("ground_support_in_frustum.u1", "|u1", (128, 128, 5)),
    ("ground_support_clear_to_target.u1", "|u1", (128, 128, 5)), ("pixel_hit_mask.u1", "|u1", (84, 112)),
    ("pixel_first_hit_distance_m.f4", "<f4", (84, 112)), ("raster_labels.u1", "|u1", (64, 64)),
)
DISTANCE_EDGES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, float("inf"))
DENIALS = {name: False for name in (
    "training_authorized", "checkpoint_or_model_access_authorized", "rgb_decode_authorized", "data_mutation_authorized",
    "calibration_authorized", "g2_authorized", "navigation_authorized", "heldout_authorized", "runtime_authorized",
    "promotion_authorized", "retry_authorized",
)}
REVIEW_AUTHORITY = {"source_review_authorized": True, "execution_authorized": False, "generated_mutation_authorized": False, **DENIALS}
EXECUTION_AUTHORITY = {
    "one_exact_positive_control_attempt_authorized": True, "cpu_numpy_authorized": True,
    "checkpoint_selection_raw_supervision_read_authorized": True, "generated_mutation_authorized": True,
    "mutation_scope": OUTPUT_ROOT, "output_root_observed_absent_at_authorization": True, "maximum_workers": 1, **DENIALS,
}
JEPA_SENTINEL = {
    "prediction_valid_cell_count": 1, "target_cross_sample_std_mean": 1.0, "target_cross_sample_effective_rank": 8.0,
    "warped_persistence_target_change": 1.0, "prediction_to_warped_persistence_ratio": 0.0,
    "wrong_action_advantage_over_target_change": 1.0, "wrong_commanded_delta_advantage_over_target_change": 1.0,
    "wrong_action_prediction_sensitivity": 1.0, "wrong_commanded_delta_prediction_sensitivity": 1.0,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def content_value(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "content_sha256": canonical_sha(core)}


def parse_json(raw: bytes, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{name} is not one canonical line")
    value = json.loads(raw[:-1].decode("ascii"))
    if type(value) is not dict or canonical_bytes(value) + b"\n" != raw:
        raise ValueError(f"{name} is not canonical")
    core, declared = dict(value), value.get("content_sha256")
    core.pop("content_sha256", None)
    if canonical_sha(core) != declared:
        raise ValueError(f"{name} self hash changed")
    return value


def binding(path: str, raw: bytes, value: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": path, "file_sha256": hashlib.sha256(raw).hexdigest(), "content_sha256": value["content_sha256"], "byte_count": len(raw)}


def source_bindings(root: Path = ROOT) -> list[dict[str, str]]:
    rows = []
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PermissionError(f"oracle source changed: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = {PRIOR_REVIEW_PATH: PRIOR_REVIEW_SHA256, EVALUATOR_PATH: EVALUATOR_SHA256, EVIDENCE_PATH: EVIDENCE_SHA256}
        if digest != expected.get(relative, digest):
            raise PermissionError(f"frozen oracle dependency changed: {relative}")
        rows.append({"path": relative, "sha256": digest})
    return rows


def raw_bindings() -> dict[str, Any]:
    return {
        "manifest": {"path": RAW_MANIFEST_PATH, "file_sha256": RAW_MANIFEST_FILE_SHA256, "content_sha256": RAW_MANIFEST_CONTENT_SHA256},
        "audit": {"path": RAW_AUDIT_PATH, "file_sha256": RAW_AUDIT_FILE_SHA256, "content_sha256": RAW_AUDIT_CONTENT_SHA256},
    }


def experiment() -> dict[str, Any]:
    return {
        "prior_state": "BLOCK_UNPROVEN", "role": "checkpoint_selection",
        "population": {"pairs": PAIR_COUNT, "unique_endpoints": ENDPOINT_COUNT, "scenes": SCENE_COUNT, "families": list(FAMILIES)},
        "oracle": {"parameters": 0, "matched": "exact_target_supervision", "wrong": "cyclic_plus_one_supervision_within_family_sorted_identity"},
        "gate": {"scopes": list(SCOPES), "margins_per_scope": MARGINS_PER_SCOPE, "margin_count": MARGIN_COUNT, "threshold_changes": 0},
        "maximum_attempts": 1, "retry_authorized": False, "interpretation": "positive_control_only_not_learned_performance",
    }


def validate_review(raw: bytes, sources: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    value = parse_json(raw, "oracle review")
    reviewer, findings, execution = value.get("reviewer"), value.get("findings"), value.get("test_execution")
    execution_fields = {"accelerators_hidden", "bytecode_disabled", "compile", "focused_cpu_tests", "hsa_override_absent", "plugin_autoload_disabled", "pytest_cache_disabled", "thread_environment"}
    compile_evidence = execution.get("compile", {}) if type(execution) is dict else {}
    test_evidence = execution.get("focused_cpu_tests", {}) if type(execution) is dict else {}
    if (
        set(value) != {"schema", "verdict", "reviewer", "implementation_author", "candidate", "experiment", "findings", "test_execution", "authority", "content_sha256"}
        or value.get("schema") != REVIEW_SCHEMA or value.get("verdict") != "PASS_SOURCE_ONLY_NO_EXECUTION_AUTHORITY"
        or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR
        or value.get("implementation_author") != IMPLEMENTATION_AUTHOR or value.get("candidate") != list(sources)
        or value.get("experiment") != experiment() or value.get("authority") != REVIEW_AUTHORITY
        or type(findings) is not list or any(type(row) is not dict or not row.get("code") or type(row.get("severity")) is not str or row["severity"].lower() == "blocking" for row in findings)
        or type(execution) is not dict or set(execution) != execution_fields
        or execution.get("accelerators_hidden") != list(ACCELERATOR_ENV)
        or execution.get("thread_environment") != {name: "1" for name in THREAD_ENV}
        or any(execution.get(name) is not True for name in ("bytecode_disabled", "hsa_override_absent", "plugin_autoload_disabled", "pytest_cache_disabled"))
        or set(compile_evidence) != {"files", "result"} or type(compile_evidence.get("files")) is not int or compile_evidence["files"] < 3 or compile_evidence.get("result") != "PASS"
        or set(test_evidence) != {"command", "duration_s", "failed", "passed", "result"} or type(test_evidence.get("command")) is not str or not test_evidence["command"]
        or type(test_evidence.get("duration_s")) not in (int, float) or test_evidence["duration_s"] < 0
        or test_evidence.get("failed") != 0 or type(test_evidence.get("passed")) is not int or test_evidence["passed"] < 12 or test_evidence.get("result") != "PASS"
    ):
        raise PermissionError("oracle review changed")
    return value


def validate_authorization(raw: bytes, sources: Sequence[Mapping[str, str]], review: Mapping[str, Any]) -> dict[str, Any]:
    value = parse_json(raw, "oracle authorization")
    authorizer = value.get("authorizer")
    if (
        set(value) != {"schema", "status", "authorizer", "implementation_author", "independent_review", "candidate", "raw", "experiment", "authority", "content_sha256"}
        or value.get("schema") != AUTHORIZATION_SCHEMA or value.get("status") != "authorized_one_exact_positive_control_attempt"
        or type(authorizer) is not str or not authorizer.startswith("/root/") or authorizer in {IMPLEMENTATION_AUTHOR, review.get("reviewer")}
        or value.get("implementation_author") != IMPLEMENTATION_AUTHOR or value.get("independent_review") != dict(review)
        or value.get("candidate") != list(sources) or value.get("raw") != raw_bindings()
        or value.get("experiment") != experiment() or value.get("authority") != EXECUTION_AUTHORITY
    ):
        raise PermissionError("oracle authorization changed")
    return value


@dataclass(frozen=True)
class Endpoint:
    identity: str
    family: str
    pixel_hit: np.ndarray
    pixel_depth: np.ndarray
    ground_valid: np.ndarray
    ground_clear: np.ndarray
    ground_distance: np.ndarray
    raster: np.ndarray

    def __post_init__(self) -> None:
        pairs = ((self.pixel_hit, self.pixel_depth), (self.ground_valid, self.ground_clear), (self.ground_valid, self.ground_distance))
        if not self.identity or self.family not in FAMILIES or any(np.shape(a) != np.shape(b) for a, b in pairs):
            raise ValueError("oracle endpoint contract changed")
        if not np.isfinite(self.pixel_depth).all() or not np.isfinite(self.ground_distance).all() or not np.isin(self.raster, (0, 1, 2)).all():
            raise ValueError("oracle endpoint values changed")


def wrong_mapping(endpoints: Sequence[Endpoint]) -> list[tuple[str, str]]:
    rows = sorted(endpoints, key=lambda row: row.identity)
    if len(rows) < 2 or len({row.identity for row in rows}) != len(rows) or len({row.family for row in rows}) != 1:
        raise ValueError("wrong-source family population changed")
    return [(row.identity, rows[(index + 1) % len(rows)].identity) for index, row in enumerate(rows)]


def _confusion(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray, classes: int) -> np.ndarray:
    if target.shape != prediction.shape or target.shape != valid.shape:
        raise ValueError("confusion shape changed")
    return np.array([
        [np.count_nonzero(valid & (target == actual) & (prediction == predicted)) for predicted in range(classes)]
        for actual in range(classes)
    ], dtype=np.int64)


def _balanced(matrix: np.ndarray) -> float:
    totals, diagonal = matrix.sum(axis=1), np.diag(matrix)
    if not np.any(totals > 0):
        raise ValueError("empty metric population")
    return float(np.mean(diagonal[totals > 0] / totals[totals > 0]))


class _Metrics:
    def __init__(self) -> None:
        self.pixel, self.ground, self.raster = np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64), np.zeros((3, 3), dtype=np.int64)
        self.distance, self.depth = np.zeros((6, 2, 2), dtype=np.int64), []

    def update(self, target: Endpoint, prediction: Endpoint) -> None:
        hit, predicted_hit = target.pixel_hit.astype(bool), prediction.pixel_hit.astype(bool)
        self.pixel += _confusion(hit, predicted_hit, np.ones(hit.shape, dtype=bool), 2)
        if np.any(hit):
            self.depth.append(np.abs(prediction.pixel_depth.astype(np.float64)[hit] - target.pixel_depth.astype(np.float64)[hit]))
        valid, clear, predicted_clear = target.ground_valid.astype(bool), target.ground_clear.astype(bool), prediction.ground_clear.astype(bool)
        self.ground += _confusion(clear, predicted_clear, valid, 2)
        for index, (low, high) in enumerate(zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])):
            self.distance[index] += _confusion(clear, predicted_clear, valid & (target.ground_distance >= low) & (target.ground_distance < high), 2)
        self.raster += _confusion(target.raster, prediction.raster, np.ones(target.raster.shape, dtype=bool), 3)

    def merge(self, other: "_Metrics") -> None:
        self.pixel += other.pixel; self.ground += other.ground; self.raster += other.raster; self.distance += other.distance; self.depth.extend(other.depth)

    def finish(self) -> dict[str, Any]:
        if not self.depth or np.any(self.raster.sum(axis=1) == 0) or np.any(self.distance.sum(axis=(1, 2)) == 0):
            raise ValueError("scope cannot emit exactly 21 physical margins")
        errors, raster_totals = np.concatenate(self.depth), self.raster.sum(axis=1)
        mismatch = int(self.raster.sum() - np.trace(self.raster))
        return {
            "pixel": _balanced(self.pixel), "depth_median": float(np.quantile(errors, .5, method="linear")),
            "depth_p95": float(np.quantile(errors, .95, method="linear")), "ground": _balanced(self.ground),
            "distance": [_balanced(row) for row in self.distance],
            "raster_nll": mismatch * -math.log(float(np.finfo(np.float32).eps)) / int(self.raster.sum()),
            "raster": _balanced(self.raster),
            "recall": {name: float(np.diag(self.raster)[index] / raster_totals[index]) for index, name in enumerate(("unknown", "free", "occupied"))},
        }


def _physical(correct: Mapping[str, Any], wrong: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pixel_first_hit_balanced_accuracy": correct["pixel"], "depth_median_error_m": correct["depth_median"],
        "depth_p95_error_m": correct["depth_p95"], "ground_clear_balanced_accuracy": correct["ground"],
        "distance_group_balanced_accuracy": correct["distance"], "derived_raster_nll": correct["raster_nll"],
        "derived_raster_balanced_accuracy": correct["raster"], "present_class_recall": correct["recall"],
        "wrong_rgb_pixel_balanced_accuracy_drop": correct["pixel"] - wrong["pixel"],
        "wrong_rgb_depth_median_error_increase_m": wrong["depth_median"] - correct["depth_median"],
        "wrong_rgb_depth_p95_error_increase_m": wrong["depth_p95"] - correct["depth_p95"],
        "wrong_rgb_ground_balanced_accuracy_drop": correct["ground"] - wrong["ground"],
        "wrong_rgb_raster_nll_increase": wrong["raster_nll"] - correct["raster_nll"],
        "wrong_rgb_raster_balanced_accuracy_drop": correct["raster"] - wrong["raster"],
    }


def evaluate(endpoints: Sequence[Endpoint]) -> dict[str, Any]:
    grouped = {family: [row for row in endpoints if row.family == family] for family in FAMILIES}
    if any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError("oracle omitted a frozen family")
    aggregate_correct, aggregate_wrong, scopes, mapping = _Metrics(), _Metrics(), {}, []
    for family, rows in grouped.items():
        ordered, correct, wrong = sorted(rows, key=lambda row: row.identity), _Metrics(), _Metrics()
        by_identity, sources = {row.identity: row for row in ordered}, dict(wrong_mapping(ordered))
        for target in ordered:
            source = by_identity[sources[target.identity]]
            correct.update(target, target); wrong.update(target, source)
            mapping.append({"family": family, "target": target.identity, "source": source.identity})
        aggregate_correct.merge(correct); aggregate_wrong.merge(wrong)
        scopes[family] = _physical(correct.finish(), wrong.finish())
    scopes = {"aggregate": _physical(aggregate_correct.finish(), aggregate_wrong.finish()), **scopes}
    protected, rows = EVALUATOR.evaluate_physical_scopes(scopes), []
    names = (*CHECKPOINT_EVALUATOR.PHYSICAL_LOWER_THRESHOLDS, *CHECKPOINT_EVALUATOR.PHYSICAL_UPPER_THRESHOLDS,
             *(f"distance_group_balanced_accuracy[{i}]" for i in range(6)),
             *(f"present_class_recall.{name}" for name in sorted(scopes["aggregate"]["present_class_recall"])))
    for scope_index, scope in enumerate(SCOPES):
        exact = CHECKPOINT_EVALUATOR.evaluate_checkpoint_scope({"physical": scopes[scope], "jepa": JEPA_SENTINEL})["physical_margins"]
        if exact != protected["scope_evaluations"][scope]["physical_margins"] or len(exact) != 21:
            raise RuntimeError("frozen physical evaluators disagree")
        rows.extend({"index": scope_index * 21 + i, "scope": scope, "within_scope_index": i, "name": names[i], "value": float(value), "passes": value >= 0.0} for i, value in enumerate(exact))
    if len(rows) != 189:
        raise RuntimeError("physical margin vector changed")
    return {
        "per_scope_metrics": scopes, "scope_evaluations": protected["scope_evaluations"], "raw_margin_vector": rows,
        "physical_pass_by_scope": protected["physical_pass_by_scope"], "physical_pass_count": protected["physical_pass_count"],
        "all_nine_physical_pass": protected["all_nine_physical_pass"], "wrong_source_mapping_count": len(mapping),
        "wrong_source_mapping_sha256": canonical_sha(mapping),
        "wrong_source_fixed_point_count": sum(row["target"] == row["source"] for row in mapping),
    }


def ground_queries(origin: np.ndarray, basis: np.ndarray, ground_z: float) -> Any:
    return EVIDENCE.project_canonical_ground_support_v4(camera_origin_body_m=origin, camera_basis_body_fru=basis, ground_plane_z_body_m=ground_z)
