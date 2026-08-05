#!/usr/bin/env python3
"""Finalize two immutable categorical-radial N32 seed results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    CONDITIONS,
    EXECUTION_BINDING_SHA256,
    FAMILIES,
    HOLDOUT_PANELS,
    PATCH7_FINAL_STATE_SHA256,
    REFERENCE_SCHEMA,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    TWO_SEED_RESULT_SCHEMA,
    categorical_holdout_checks,
    extract_faithful_patch7_family_reference,
    fit_panel_gate_report,
    per_seed_decision,
    terminal_fit_gate_summary,
)
from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    frame_records,
    validate_panel_manifest,
)


EXPECTED_SEEDS = (20260710, 20260711)
STAGE_SCHEMA = "lewm_go2_categorical_radial_n32_stage_v1"
PANEL_REPORT_SCHEMA = "lewm_go2_categorical_radial_n32_panel_report_v1"
REGISTERED_PARAMETER_COUNT = 2_887_067
EXPECTED_SEED10_INITIAL_STATE_SHA256 = (
    "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278"
)
EXPECTED_SEED11_INITIAL_STATE_SHA256 = (
    "989e2db491d199bc544fabe2df40443a39f3ffc6e936f0d28c24625e7bd0ce13"
)
PANEL_PATH = (
    REPOSITORY_ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
)
LADDER_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v1/ladder_manifest.json"
)
V3_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_micro_overfit/v3/"
    "seed_20260710_ladder_result.json"
)
PATCH7_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_physical_micro_overfit/patch7_v1/seed_20260710_result.json"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_microfit_protocol_2026-07-10.md"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_execution_binding_2026-07-10.md"
)
EXPECTED_INPUTS = {
    "panel": (
        "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c",
        "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f",
    ),
    "ladder_manifest": (
        "967812399045b29e8be316f2f87bc16f02d681b0ea01884513c6b4f29bbe4b12",
        "00a3ad1263af16e3b858f7e7522df7b108a49301d25fa805148e82b36cb52f8e",
    ),
    "v3_result": (
        "7a5f67bacb2e3df67421bcff13b15d1fa3e00d99f3b2af52c52b0b6ce14617a8",
        "517313139077027176c471f829f57148684d3df0def6096ce7702d3bbba46ce1",
    ),
    "patch7_reference_result": (
        "6e2aacd18fe1d692fb6ad682b41132563dcbcdb95c7b7ce719f407baf6c91a8c",
        "32d848d3df68e670ddb4cc24436981f62a1aa5562b89e6d6719ecb113f66b749",
    ),
}
EXPECTED_EVIDENCE_SHA256 = {
    *(file_hash for file_hash, _content_hash in EXPECTED_INPUTS.values()),
    "ef23ee607d0976d67adf33591f5af78652da4305811a563d94bd8539abc9d404",
    EXECUTION_BINDING_SHA256,
}
BOUND_EVIDENCE = {
    PANEL_PATH: EXPECTED_INPUTS["panel"][0],
    LADDER_PATH: EXPECTED_INPUTS["ladder_manifest"][0],
    V3_RESULT_PATH: EXPECTED_INPUTS["v3_result"][0],
    PATCH7_RESULT_PATH: EXPECTED_INPUTS["patch7_reference_result"][0],
    PROTOCOL_PATH: (
        "ef23ee607d0976d67adf33591f5af78652da4305811a563d94bd8539abc9d404"
    ),
    CONTRACT_PATH: EXECUTION_BINDING_SHA256,
}
BRANCH_CONFIGS = {
    "production_faithful": {
        "updates": 2000,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
    },
    "ceiling_optimizer": {
        "updates": 5000,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    },
}
EXPECTED_PANEL_ARTIFACT_COUNTS = {
    "fit": {"images": 320, "shards": 20},
    "same_scene_holdout": {"images": 320, "shards": 20},
    "cross_scene_holdout": {"images": 320, "shards": 25},
}
RUNNER_SOURCE_BINDINGS = {
    "n32_contract": CONTRACT_PATH,
    "n32_pure": REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32.py",
    "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32.py",
}
CANONICAL_RESULT_PATHS = {
    seed: (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v1/"
        f"seed_{seed}_result.json"
    ).resolve()
    for seed in EXPECTED_SEEDS
}
RUNNER_SOURCE_PATHS = {
    "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
    "factorization": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_factorization.py"
    ),
    "ladder_contract": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_micro_overfit.py"
    ),
    "model": REPOSITORY_ROOT / "lewm/models/categorical_radial_perception.py",
    "panel_contract": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py"
    ),
    "preparer": REPOSITORY_ROOT / "scripts/prepare_go2_categorical_radial_ladder.py",
    "protocol": PROTOCOL_PATH,
    "v2_amendment": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_categorical_radial_ladder_v2_optimizer_amendment_2026-07-10.md"
    ),
    "v1_result": (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_micro_overfit/v1/"
        "seed_20260710_ladder_result.json"
    ),
    "v1_runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_ladder.py",
    "v2_runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_ladder_v2.py",
    "model_full_ray": (
        REPOSITORY_ROOT / "lewm/models/categorical_radial_perception_full_ray.py"
    ),
    "v2_result": (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_micro_overfit/v2/"
        "seed_20260710_ladder_result.json"
    ),
    "v3_amendment": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_categorical_radial_ladder_v3_full_ray_amendment_2026-07-10.md"
    ),
    "v3_runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_ladder_v3.py",
    **RUNNER_SOURCE_BINDINGS,
}
SCHEDULE_SHA256 = {
    (20260710, 2000): (
        "3de32de003991942d8e08f0d12296b6b3018831225394c12ba2da438cc94ab02"
    ),
    (20260710, 5000): (
        "0bc06fd8bef9bbf49da8459104ccb1dbb7994aa0a7e99b560b244e91a1690b8d"
    ),
    (20260711, 2000): (
        "34a5e5256c939be00c40e9594b05d2087416d5c1275d44e5904bc3dcb29d6e4b"
    ),
    (20260711, 5000): (
        "304c1d87a5719900d12b0fc6caedc3ef6abb6f3d88e035b07ff4421bfb060cc7"
    ),
}
TOP_LEVEL_FIELDS = {
    "schema",
    "authoritative",
    "aggregation_eligible",
    "promotion_eligible",
    "seed",
    "created_at_utc",
    "completed_at_utc",
    "invocation",
    "execution",
    "contract",
    "inputs",
    "source_hashes",
    "git",
    "model",
    "stages",
    "patch7_reference",
    "holdouts",
    "holdout_checks",
    "decision",
    "artifact_verification",
    "access_ledger",
    "categorical_radial_full_train_candidate_licensed",
    "content_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _strict_int(value: object, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context} must be a JSON integer >= {minimum}")
    return value


def _strict_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a JSON boolean")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite JSON number")
    return result


def _unit_interval(value: object, *, context: str) -> float:
    result = _finite_number(value, context=context)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be in [0, 1]")
    return result


def _validate_confusion(value: object, *, context: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a nonempty count matrix")
    width = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{context} row must be a nonempty count list")
        width = len(row) if width is None else width
        if len(row) != width:
            raise ValueError(f"{context} rows must have equal width")
        for column_index, count in enumerate(row):
            _strict_int(
                count,
                context=f"{context}[{row_index}][{column_index}]",
            )


def _validate_present_metric_domains(
    value: object,
    *,
    context: str,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            _validate_present_metric_domains(
                child,
                context=context,
                path=(*path, str(name)),
            )
        return
    leaf = path[-1] if path else ""
    joined = "/".join(path).lower()
    if "confusion" in joined:
        if len(path) == 1:
            _validate_confusion(value, context=f"{context}/{joined}")
        return
    if isinstance(value, list):
        return
    if value is None:
        if "posterior_quantiles" in joined:
            raise ValueError(f"{context}/{joined} must be in [0, 1]")
        return
    if "count" in leaf.lower() or "support" in leaf.lower():
        _strict_int(value, context=f"{context}/{joined}")
    elif "nll" in joined:
        if _finite_number(value, context=f"{context}/{joined}") < 0.0:
            raise ValueError(f"{context}/{joined} must be nonnegative")
    elif any(
        token in joined
        for token in (
            "accuracy",
            "recall",
            "precision",
            "average_precision",
            "posterior_quantiles",
            "probability",
        )
    ):
        _unit_interval(value, context=f"{context}/{joined}")


def _validate_metric_domains(metrics: Mapping[str, Any], *, context: str) -> None:
    nll = _finite_number(
        metrics.get("raw_hierarchical_balanced_nll"),
        context=f"{context}/raw_hierarchical_balanced_nll",
    )
    if nll < 0.0:
        raise ValueError(f"{context}/raw_hierarchical_balanced_nll must be nonnegative")
    for name in (
        "unknown_known_balanced_accuracy",
        "free_occupied_balanced_accuracy",
        "raw_joint_accuracy",
        "raw_known_free_occupied_accuracy",
        "free_average_precision",
        "occupied_average_precision",
    ):
        if name in metrics:
            _unit_interval(metrics[name], context=f"{context}/{name}")
    recalls = metrics.get("class_recall")
    if recalls is not None:
        recall_map = _mapping(recalls, context=f"{context}/class_recall")
        if set(recall_map) != {"unknown", "free", "occupied"}:
            raise ValueError(f"{context}/class_recall is incomplete")
        for name, recall in recall_map.items():
            _unit_interval(recall, context=f"{context}/class_recall/{name}")
    distance_recalls = metrics.get("distance_free_recall")
    distance_support = metrics.get("distance_free_support")
    if distance_recalls is not None:
        recall_map = _mapping(
            distance_recalls,
            context=f"{context}/distance_free_recall",
        )
        support_map = _mapping(
            distance_support,
            context=f"{context}/distance_free_support",
        )
        if set(recall_map) != set(support_map):
            raise ValueError(f"{context} distance recall/support bins differ")
        for name, recall in recall_map.items():
            support = _strict_int(
                support_map[name],
                context=f"{context}/distance_free_support/{name}",
            )
            if recall is None:
                if support != 0:
                    raise ValueError(
                        f"{context}/distance_free_recall/{name} lacks supported recall"
                    )
            else:
                _unit_interval(
                    recall,
                    context=f"{context}/distance_free_recall/{name}",
                )
                if support == 0:
                    raise ValueError(
                        f"{context}/distance_free_recall/{name} has no support"
                    )
    for name, raw_value in metrics.items():
        if "confusion" in name.lower():
            _validate_confusion(raw_value, context=f"{context}/{name}")
    _validate_present_metric_domains(metrics, context=context)


def _validate_gate_boolean_types(value: object, *, context: str) -> None:
    record = _mapping(value, context=context)
    for name, child in record.items():
        if name == "checks":
            checks = _mapping(child, context=f"{context}/checks")
            for check_name, passed in checks.items():
                _strict_bool(passed, context=f"{context}/checks/{check_name}")
        elif name == "passes" or name.startswith("requires_") or name == "ties_count_as_failure":
            _strict_bool(child, context=f"{context}/{name}")
        elif isinstance(child, Mapping):
            _validate_gate_boolean_types(child, context=f"{context}/{name}")


def _validate_zero_contact(value: object, *, context: str) -> None:
    record = _mapping(value, context=context)
    expected = {"image_byte_opens", "label_shard_byte_opens", "model_outputs"}
    if set(record) != expected or any(
        _strict_int(record[name], context=f"{context}/{name}") != 0
        for name in expected
    ):
        raise ValueError(f"{context} records forbidden access")


def _load_expected_json(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"expected result SHA-256 is malformed: {path}")
    pre_hash = _sha256_file(path)
    if pre_hash != expected_sha256:
        raise ValueError(f"result differs from its precommitted SHA-256: {path}")
    serialized = path.read_bytes()
    bytes_hash = hashlib.sha256(serialized).hexdigest()
    if bytes_hash != pre_hash:
        raise RuntimeError(f"result changed between pre-hash and read: {path}")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    post_hash = _sha256_file(path)
    if post_hash != pre_hash:
        raise RuntimeError(f"result changed during deserialization: {path}")
    return value, {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "pre_deserialization_sha256": pre_hash,
        "bytes_read_sha256": bytes_hash,
        "post_read_sha256": post_hash,
        "pre_deserialization_hash_match": True,
        "post_read_unchanged": True,
    }


def _load_bound_evidence() -> dict[str, Any]:
    pre_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if pre_hashes != {
        str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()
    }:
        raise ValueError("bound N32 evidence file SHA-256 drift")
    payloads = {
        path: json.loads(path.read_bytes())
        for path in (PANEL_PATH, LADDER_PATH, V3_RESULT_PATH, PATCH7_RESULT_PATH)
    }
    if not all(isinstance(value, dict) for value in payloads.values()):
        raise ValueError("bound N32 evidence JSON must contain objects")
    panel = payloads[PANEL_PATH]
    if (
        panel.get("schema") != "lewm_go2_physical_micro_overfit_panel_v1"
        or panel.get("content_sha256") != EXPECTED_INPUTS["panel"][1]
    ):
        raise ValueError("bound N32 panel content drift")
    panels = validate_panel_manifest(panel)
    ladder = payloads[LADDER_PATH]
    if (
        ladder.get("schema") != "lewm_go2_categorical_radial_ladder_manifest_v1"
        or ladder.get("content_sha256") != EXPECTED_INPUTS["ladder_manifest"][1]
    ):
        raise ValueError("bound N32 ladder content drift")
    v3_result = payloads[V3_RESULT_PATH]
    if (
        v3_result.get("schema") != "lewm_go2_categorical_radial_ladder_result_v3"
        or v3_result.get("content_sha256") != EXPECTED_INPUTS["v3_result"][1]
        or v3_result.get("decision", {}).get("n32_diagnostic_construction_licensed")
        is not True
    ):
        raise ValueError("bound N32 V3 decision/content drift")
    patch7_result = payloads[PATCH7_RESULT_PATH]
    if (
        patch7_result.get("schema") != "lewm_go2_physical_micro_overfit_result_v1"
        or patch7_result.get("content_sha256")
        != EXPECTED_INPUTS["patch7_reference_result"][1]
    ):
        raise ValueError("bound N32 patch7 reference content drift")
    reference = extract_faithful_patch7_family_reference(patch7_result)
    post_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 evidence changed during parsing")
    return {
        "pre_hashes": pre_hashes,
        "post_parse_hashes": post_hashes,
        "panels": panels,
        "patch7_reference": reference,
    }


def _expected_controls(
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for panel in ("fit", *HOLDOUT_PANELS):
        records = frame_records(panels[panel])
        records, role_global = attach_role_global_shuffle(
            records,
            seed=seed,
            namespace=panel,
        )
        _records, same_scene = attach_same_scene_wrong_view(
            records,
            seed=seed,
            namespace=panel,
        )
        result[panel] = {
            "role_global_shuffle": role_global,
            "same_scene_wrong_view": same_scene,
        }
    return result


def _validate_conditions(value: object, *, context: str) -> None:
    conditions = _mapping(value, context=f"{context} conditions")
    if set(conditions) != set(CONDITIONS):
        raise ValueError(f"{context} conditions are incomplete")
    for condition in CONDITIONS:
        metrics = _mapping(
            conditions[condition],
            context=f"{context}/{condition} metrics",
        )
        _validate_metric_domains(metrics, context=f"{context}/{condition}")
        if condition == "correct_rgb":
            for name in (
                "unknown_known_balanced_accuracy",
                "free_occupied_balanced_accuracy",
                "class_recall",
                "distance_free_recall",
                "distance_free_support",
            ):
                if name not in metrics:
                    raise ValueError(f"{context}/{condition} lacks {name}")


def _validate_controls(
    value: object,
    *,
    panel: str,
    seed: int,
    expected: Mapping[str, Any] | None,
) -> None:
    controls = _mapping(value, context=f"{panel} controls")
    if set(controls) != {"role_global_shuffle", "same_scene_wrong_view"}:
        raise ValueError(f"{panel} controls are incomplete")
    role = _mapping(controls["role_global_shuffle"], context=f"{panel} role control")
    same = _mapping(controls["same_scene_wrong_view"], context=f"{panel} same control")
    if (
        _strict_int(role.get("seed"), context=f"{panel} role control seed") != seed
        or role.get("namespace") != panel
        or _strict_int(
            role.get("record_count"), context=f"{panel} role record_count"
        )
        != 320
        or _strict_int(
            role.get("same_image_pairs"), context=f"{panel} role same_image_pairs"
        )
        != 0
        or _strict_int(
            role.get("same_scene_pairs"), context=f"{panel} role same_scene_pairs"
        )
        != 0
        or _strict_int(
            role.get("same_transition_pairs"),
            context=f"{panel} role same_transition_pairs",
        )
        != 0
        or not _is_sha256(role.get("permutation_sha256"))
    ):
        raise ValueError(f"{panel} role-global control contract drift")
    if (
        _strict_int(same.get("seed"), context=f"{panel} same control seed") != seed
        or same.get("namespace") != panel
        or _strict_int(
            same.get("record_count"), context=f"{panel} same record_count"
        )
        != 320
        or _strict_int(
            same.get("same_image_pairs"), context=f"{panel} same same_image_pairs"
        )
        != 0
        or _strict_int(
            same.get("same_transition_pairs"),
            context=f"{panel} same same_transition_pairs",
        )
        != 0
        or _strict_int(
            same.get("different_scene_pairs"),
            context=f"{panel} same different_scene_pairs",
        )
        != 0
        or not _is_sha256(same.get("permutation_sha256"))
    ):
        raise ValueError(f"{panel} same-scene control contract drift")
    scenes = _mapping(same.get("scenes"), context=f"{panel} same control scenes")
    for scene, raw_scene in scenes.items():
        scene_record = _mapping(raw_scene, context=f"{panel} same scene {scene}")
        for name in ("frame_count", "transition_count", "rotation"):
            _strict_int(
                scene_record.get(name),
                context=f"{panel} same scene {scene}/{name}",
                minimum=1,
            )
    if expected is not None and controls != expected:
        raise ValueError(f"{panel} exact deterministic control report drift")


def _validate_panel_report(
    value: object,
    *,
    panel: str,
    seed: int,
    require_fit_gate: bool,
    expected_controls: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    report = _mapping(value, context=f"{panel} report")
    if (
        report.get("schema") != PANEL_REPORT_SCHEMA
        or report.get("panel") != panel
        or _strict_int(report.get("frame_count"), context=f"{panel} frame_count")
        != 320
        or _strict_int(
            report.get("target_batch_size"), context=f"{panel} target_batch_size"
        )
        != 4
        or _strict_int(
            report.get("combined_model_batch_size"),
            context=f"{panel} combined_model_batch_size",
        )
        != 12
        or report.get("model_call_dtype") != "float32"
        or report.get("metric_accumulator_dtype") != "float64"
    ):
        raise ValueError(f"{panel} report execution contract drift")
    _validate_conditions(report.get("conditions"), context=f"{panel} aggregate")
    families = _mapping(report.get("families"), context=f"{panel} families")
    if set(families) != set(FAMILIES):
        raise ValueError(f"{panel} report lacks the canonical five families")
    for family in FAMILIES:
        family_record = _mapping(
            families[family],
            context=f"{panel}/{family}",
        )
        _validate_conditions(
            family_record.get("conditions"),
            context=f"{panel}/{family}",
        )
    _validate_controls(
        report.get("controls"),
        panel=panel,
        seed=seed,
        expected=expected_controls,
    )
    if require_fit_gate:
        recomputed = fit_panel_gate_report(report)
        _validate_gate_boolean_types(
            report.get("fit_gate"),
            context=f"{panel} stored fit gate",
        )
        if report.get("fit_gate") != recomputed:
            raise ValueError(f"{panel} stored aggregate/family fit gate drift")
    elif "fit_gate" in report:
        raise ValueError(f"{panel} holdout must not store an adjudicating fit gate")
    return report


def _expected_optimizer(stage_name: str) -> dict[str, Any]:
    config = BRANCH_CONFIGS[stage_name]
    return {
        "name": "AdamW",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "gradient_clip": 1.0,
        "constant_learning_rate": True,
    }


def _validate_optimizer(value: object, *, stage_name: str) -> Mapping[str, Any]:
    optimizer = _mapping(value, context=f"{stage_name} optimizer")
    if set(optimizer) != set(_expected_optimizer(stage_name)):
        raise ValueError(f"{stage_name} optimizer fields drift")
    for name in ("learning_rate", "weight_decay", "epsilon", "gradient_clip"):
        _finite_number(optimizer[name], context=f"{stage_name} optimizer/{name}")
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError(f"{stage_name} optimizer betas drift")
    for index, beta in enumerate(betas):
        _unit_interval(beta, context=f"{stage_name} optimizer/betas/{index}")
    _strict_bool(optimizer.get("amsgrad"), context=f"{stage_name} optimizer/amsgrad")
    _strict_bool(
        optimizer.get("constant_learning_rate"),
        context=f"{stage_name} optimizer/constant_learning_rate",
    )
    if optimizer != _expected_optimizer(stage_name):
        raise ValueError(f"{stage_name} optimizer contract drift")
    return optimizer


def _validate_terminal_summary(value: object, *, context: str) -> None:
    summary = _mapping(value, context=f"{context} terminal summary")
    for name in ("maximum_steps", "evaluation_interval"):
        _strict_int(summary.get(name), context=f"{context} terminal {name}", minimum=1)
    for name in ("evaluation_steps", "terminal_evaluation_steps"):
        values = summary.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{context} terminal {name} must be a nonempty list")
        for index, raw_step in enumerate(values):
            _strict_int(
                raw_step,
                context=f"{context} terminal {name}[{index}]",
                minimum=1,
            )
    for name in ("evaluation_passes", "terminal_evaluation_passes"):
        values = summary.get(name)
        if not isinstance(values, list) or not values or any(
            type(item) is not bool for item in values
        ):
            raise ValueError(f"{context} terminal {name} must contain booleans")
    for name in ("first_single_fit_gate_step", "first_three_consecutive_fit_gate_step"):
        raw_step = summary.get(name)
        if raw_step is not None:
            _strict_int(raw_step, context=f"{context} terminal {name}", minimum=1)
    for name in ("requires_exact_final_three", "passes"):
        if type(summary.get(name)) is not bool:
            raise ValueError(f"{context} terminal {name} must be boolean")


def _validate_minibatches(
    batches: object,
    *,
    updates: int,
    seed: int,
    context: str,
) -> list[list[int]]:
    if not isinstance(batches, list) or len(batches) != updates:
        raise ValueError(f"{context} minibatch schedule has the wrong length")
    normalized = []
    for batch in batches:
        if not isinstance(batch, list) or len(batch) != 4:
            raise ValueError(f"{context} minibatch schedule is malformed")
        if any(type(value) is not int for value in batch):
            raise ValueError(f"{context} minibatch indices must be JSON integers")
        indices = list(batch)
        if len(set(indices)) != 4 or any(not 0 <= value < 320 for value in indices):
            raise ValueError(f"{context} minibatch contains invalid frame indices")
        normalized.append(indices)
    for start in range(0, updates - 79, 80):
        epoch = [value for batch in normalized[start : start + 80] for value in batch]
        if sorted(epoch) != list(range(320)):
            raise ValueError(f"{context} complete epoch is not a frame permutation")
    if _canonical_json_sha256(batches) != SCHEDULE_SHA256[(seed, updates)]:
        raise ValueError(f"{context} exact seeded minibatch schedule drift")
    return normalized


def _validate_stage(
    value: object,
    *,
    stage_name: str,
    seed: int,
    expected_initial_state: str,
    expected_controls: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    stage = _mapping(value, context=f"{stage_name} stage")
    config = BRANCH_CONFIGS[stage_name]
    updates = config["updates"]
    _validate_optimizer(stage.get("optimizer"), stage_name=stage_name)
    if (
        stage.get("schema") != STAGE_SCHEMA
        or stage.get("stage") != stage_name
        or _strict_int(
            stage.get("maximum_steps"), context=f"{stage_name} maximum_steps"
        )
        != updates
        or _strict_int(
            stage.get("completed_steps"), context=f"{stage_name} completed_steps"
        )
        != updates
        or _strict_int(stage.get("batch_size"), context=f"{stage_name} batch_size")
        != 4
        or _strict_int(
            stage.get("evaluation_interval"),
            context=f"{stage_name} evaluation_interval",
        )
        != 100
        or stage.get("fixed_update_budget_consumed") is not True
        or stage.get("initial_state_sha256") != expected_initial_state
        or not _is_sha256(stage.get("final_state_sha256"))
        or not isinstance(stage.get("holdouts_evaluated"), bool)
    ):
        raise ValueError(f"{stage_name} exact budget/model/optimizer contract drift")
    batches = _validate_minibatches(
        stage.get("minibatch_indices"),
        updates=updates,
        seed=seed,
        context=stage_name,
    )
    if stage.get("minibatch_indices_sha256") != _canonical_json_sha256(
        stage.get("minibatch_indices")
    ):
        raise ValueError(f"{stage_name} minibatch content hash drift")
    curve = stage.get("learning_curve")
    if not isinstance(curve, list):
        raise ValueError(f"{stage_name} learning curve is missing")
    for point in curve:
        record = _mapping(point, context=f"{stage_name} curve point")
        _strict_int(
            record.get("step"),
            context=f"{stage_name} curve step",
            minimum=1,
        )
        _validate_panel_report(
            record.get("fit_panel"),
            panel="fit",
            seed=seed,
            require_fit_gate=True,
            expected_controls=expected_controls,
        )
        for metric_name in ("batch_loss", "gradient_norm_before_clip"):
            metric = _finite_number(
                record.get(metric_name),
                context=f"{stage_name} curve {metric_name}",
            )
            if metric < 0.0:
                raise ValueError(
                    f"{stage_name} curve {metric_name} must be nonnegative"
                )
    summary = terminal_fit_gate_summary(curve, updates, 100)
    _validate_terminal_summary(
        stage.get("terminal_fit_gate"),
        context=stage_name,
    )
    if stage.get("terminal_fit_gate") != summary:
        raise ValueError(f"{stage_name} terminal fit gate does not recompute")
    training = _mapping(stage.get("training_access"), context=f"{stage_name} training access")
    evaluation = _mapping(
        stage.get("fit_evaluation_access"),
        context=f"{stage_name} fit evaluation access",
    )
    curve_count = updates // 100
    expected_access_fields = {
        "image_requests",
        "target_requests",
        "image_decode_events",
        "label_shard_npz_open_events",
        "model_calls",
        "model_output_frames",
    }
    if set(training) != expected_access_fields or set(evaluation) != expected_access_fields:
        raise ValueError(f"{stage_name} dataset access fields drift")
    if (
        _strict_int(training.get("model_calls"), context=f"{stage_name} training model_calls")
        != updates
        or _strict_int(
            training.get("model_output_frames"),
            context=f"{stage_name} training model_output_frames",
        )
        != updates * 4
        or _strict_int(
            training.get("image_requests"),
            context=f"{stage_name} training image_requests",
        )
        != updates * 4
        or _strict_int(
            training.get("target_requests"),
            context=f"{stage_name} training target_requests",
        )
        != updates * 4
        or _strict_int(
            evaluation.get("model_calls"),
            context=f"{stage_name} evaluation model_calls",
        )
        != curve_count * 80
        or _strict_int(
            evaluation.get("model_output_frames"),
            context=f"{stage_name} evaluation model_output_frames",
        )
        != curve_count * 80 * 12
        or _strict_int(
            evaluation.get("image_requests"),
            context=f"{stage_name} evaluation image_requests",
        )
        != curve_count * 960
        or _strict_int(
            evaluation.get("target_requests"),
            context=f"{stage_name} evaluation target_requests",
        )
        != curve_count * 320
    ):
        raise ValueError(f"{stage_name} model-output access does not reconcile")
    return stage, summary


def _validate_inputs(value: object, *, seed: int) -> Mapping[str, Any]:
    inputs = _mapping(value, context="N32 inputs")
    if set(inputs) != {*EXPECTED_INPUTS, "seed_20260710_authorization"}:
        raise ValueError("N32 input provenance keys are incomplete")
    for name, (file_hash, content_hash) in EXPECTED_INPUTS.items():
        record = _mapping(inputs[name], context=f"N32 input {name}")
        expected_path = {
            "panel": PANEL_PATH,
            "ladder_manifest": LADDER_PATH,
            "v3_result": V3_RESULT_PATH,
            "patch7_reference_result": PATCH7_RESULT_PATH,
        }[name]
        if (
            Path(str(record.get("path", ""))).resolve() != expected_path.resolve()
            or record.get("sha256") != file_hash
            or record.get("content_sha256") != content_hash
        ):
            raise ValueError(f"N32 immutable input drift: {name}")
    authorization = inputs["seed_20260710_authorization"]
    if seed == 20260710 and authorization is not None:
        raise ValueError("seed 20260710 must not carry a seed authorization")
    if seed == 20260711:
        record = _mapping(authorization, context="seed 20260710 authorization")
        if (
            Path(str(record.get("path", ""))).resolve()
            != CANONICAL_RESULT_PATHS[20260710].resolve()
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError("seed 20260711 lacks immutable primary authorization")
    return inputs


def _runner_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for name, path in sorted(RUNNER_SOURCE_PATHS.items())
    }


def _validate_source_hashes(value: object) -> Mapping[str, Any]:
    sources = _mapping(value, context="N32 source hashes")
    expected_sources = _runner_source_hashes()
    if set(sources) != set(expected_sources):
        raise ValueError("N32 transitive source provenance is incomplete")
    for name, raw_record in sources.items():
        record = _mapping(raw_record, context=f"N32 source {name}")
        if not str(record.get("path", "")) or not _is_sha256(record.get("sha256")):
            raise ValueError(f"N32 source hash is malformed: {name}")
    if sources != expected_sources:
        raise ValueError("N32 transitive frozen source drift")
    return sources


def _validate_patch7_reference(value: object) -> Mapping[str, Any]:
    reference = _mapping(value, context="patch7 reference")
    if (
        reference.get("schema") != REFERENCE_SCHEMA
        or reference.get("source_stage") != "production_faithful"
        or reference.get("source_arm") != "patch7_16x16"
        or reference.get("final_state_sha256") != PATCH7_FINAL_STATE_SHA256
    ):
        raise ValueError("patch7 faithful reference identity drift")
    panels = _mapping(reference.get("panels"), context="patch7 reference panels")
    if set(panels) != set(HOLDOUT_PANELS):
        raise ValueError("patch7 faithful reference panels are incomplete")
    for panel in HOLDOUT_PANELS:
        panel_record = _mapping(panels[panel], context=f"patch7 {panel}")
        if (
            panel_record.get("panel") != panel
            or _strict_int(
                panel_record.get("frame_count"),
                context=f"patch7 {panel} frame_count",
            )
            != 320
        ):
            raise ValueError(f"patch7 {panel} identity/count drift")
        _validate_conditions(
            panel_record.get("conditions"),
            context=f"patch7 {panel} aggregate",
        )
        _validate_controls(
            {
                "role_global_shuffle": panel_record.get("role_global_shuffle"),
                "same_scene_wrong_view": panel_record.get("same_scene_wrong_view"),
            },
            panel=panel,
            seed=20260710,
            expected=None,
        )
        access = _mapping(
            panel_record.get("access"),
            context=f"patch7 {panel} access",
        )
        for name, count in access.items():
            _strict_int(count, context=f"patch7 {panel} access/{name}")
        families = _mapping(
            panel_record.get("families"),
            context=f"patch7 {panel} families",
        )
        if set(families) != set(FAMILIES):
            raise ValueError(f"patch7 {panel} families are incomplete")
        for family in FAMILIES:
            conditions = _mapping(
                _mapping(
                    families[family],
                    context=f"patch7 {panel}/{family}",
                ).get("conditions"),
                context=f"patch7 {panel}/{family} conditions",
            )
            _validate_conditions(
                conditions,
                context=f"patch7 {panel}/{family}",
            )
    fake_result = {
        "stages": {
            "production_faithful": {
                "patch7_16x16": {
                    "final_state_sha256": PATCH7_FINAL_STATE_SHA256,
                    "final_panels": reference.get("panels"),
                }
            }
        },
        "post_selection_support_audit": {
            "fit": {},
            "same_scene_holdout": {},
            "cross_scene_holdout": {},
        },
    }
    recomputed = extract_faithful_patch7_family_reference(fake_result)
    if reference != recomputed:
        raise ValueError("patch7 faithful reference metrics do not recompute")
    return reference


def _validate_artifact_verification(value: object) -> None:
    verification = _mapping(value, context="artifact verification")
    if (
        verification.get("fit_verified_before_access") is not True
        or verification.get("holdouts_verified_only_after_terminal_fit_pass")
        is not True
    ):
        raise ValueError("conditional payload verification contract drift")
    evidence = _mapping(
        verification.get("evidence_hashes"),
        context="bound evidence hashes",
    )
    expected_evidence = {
        str(path.resolve()): digest
        for path, digest in BOUND_EVIDENCE.items()
    }
    if evidence != expected_evidence:
        raise ValueError("bound evidence path/hash mapping drift")


def _validate_panel_access(
    value: object,
    *,
    holdouts_authorized: bool,
) -> None:
    access = _mapping(value, context="access ledger")
    if set(access) != {
        "panels",
        "fit_dataset_totals",
        "checkpoint_selection",
        "probability_calibration",
        "g2_evaluation",
        "non_train_image_opens",
        "non_train_label_shard_opens",
        "non_train_model_outputs",
    }:
        raise ValueError("access ledger top-level fields drift")
    panels = _mapping(access.get("panels"), context="panel access ledger")
    if set(panels) != {"fit", *HOLDOUT_PANELS}:
        raise ValueError("panel access ledger is incomplete")
    for panel in ("fit", *HOLDOUT_PANELS):
        record = _mapping(panels[panel], context=f"{panel} access")
        authorized = panel == "fit" or holdouts_authorized
        expected = EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
        expected_fields = (
            {
                "authorized",
                "artifact_hash_passes",
                "image_hash_byte_open_events",
                "shard_hash_byte_open_events",
            }
            if panel == "fit"
            else {
                "authorized",
                "artifact_hash_passes",
                "image_hash_byte_open_events",
                "shard_hash_byte_open_events",
                "dataset_access",
            }
            if authorized
            else {
                "authorized",
                "artifact_hash_passes",
                "image_hash_byte_open_events",
                "shard_hash_byte_open_events",
                "model_output_frames",
            }
        )
        if set(record) != expected_fields:
            raise ValueError(f"{panel} access record fields drift")
        if record.get("authorized") is not authorized:
            raise ValueError(f"{panel} access authorization drift")
        if authorized:
            if (
                _strict_int(
                    record.get("artifact_hash_passes"),
                    context=f"{panel} artifact_hash_passes",
                )
                != 2
                or _strict_int(
                    record.get("image_hash_byte_open_events"),
                    context=f"{panel} image_hash_byte_open_events",
                )
                != 2 * expected["images"]
                or _strict_int(
                    record.get("shard_hash_byte_open_events"),
                    context=f"{panel} shard_hash_byte_open_events",
                )
                != 2 * expected["shards"]
            ):
                raise ValueError(f"{panel} payload verification access drift")
        else:
            for name in (
                "artifact_hash_passes",
                "image_hash_byte_open_events",
                "shard_hash_byte_open_events",
                "model_output_frames",
            ):
                if _strict_int(record.get(name), context=f"{panel} {name}") != 0:
                    raise ValueError(f"{panel} unauthorized payload access exists")
            nested = record.get("dataset_access")
            if nested is not None:
                nested_record = _mapping(
                    nested,
                    context=f"{panel} unauthorized dataset access",
                )
                if any(
                    _strict_int(value, context=f"{panel} dataset_access/{name}") != 0
                    for name, value in nested_record.items()
                ):
                    raise ValueError(f"{panel} unauthorized dataset access exists")
        if panel in HOLDOUT_PANELS and authorized:
            dataset = _mapping(record.get("dataset_access"), context=f"{panel} dataset access")
            if set(dataset) != {
                "image_requests",
                "target_requests",
                "image_decode_events",
                "label_shard_npz_open_events",
                "model_calls",
                "model_output_frames",
            }:
                raise ValueError(f"{panel} dataset access fields drift")
            if (
                _strict_int(
                    dataset.get("image_decode_events"),
                    context=f"{panel} image_decode_events",
                )
                != 320
                or _strict_int(
                    dataset.get("label_shard_npz_open_events"),
                    context=f"{panel} label_shard_npz_open_events",
                )
                != expected["shards"]
                or _strict_int(
                    dataset.get("image_requests"),
                    context=f"{panel} image_requests",
                )
                != 960
                or _strict_int(
                    dataset.get("target_requests"),
                    context=f"{panel} target_requests",
                )
                != 320
                or _strict_int(dataset.get("model_calls"), context=f"{panel} model_calls")
                != 80
                or _strict_int(
                    dataset.get("model_output_frames"),
                    context=f"{panel} model_output_frames",
                )
                != 960
            ):
                raise ValueError(f"{panel} model/data access does not reconcile")
    totals = _mapping(access.get("fit_dataset_totals"), context="fit dataset totals")
    if set(totals) != {
        "image_requests",
        "target_requests",
        "image_decode_events",
        "label_shard_npz_open_events",
    }:
        raise ValueError("fit dataset total fields drift")
    if (
        _strict_int(
            totals.get("image_decode_events"),
            context="fit totals image_decode_events",
        )
        != 320
        or _strict_int(
            totals.get("label_shard_npz_open_events"),
            context="fit totals label_shard_npz_open_events",
        )
        != 20
    ):
        raise ValueError("fit payload access does not reconcile")
    for role in ("checkpoint_selection", "probability_calibration", "g2_evaluation"):
        _validate_zero_contact(access.get(role), context=f"{role} ledger")
    for name in (
        "non_train_image_opens",
        "non_train_label_shard_opens",
        "non_train_model_outputs",
    ):
        if _strict_int(access.get(name), context=f"access ledger/{name}") != 0:
            raise ValueError("N32 result records forbidden non-train access")


def _validate_authoritative_result(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if expected_seed not in EXPECTED_SEEDS:
        raise ValueError("N32 finalizer accepts exactly the two registered seeds")
    if set(artifact) != TOP_LEVEL_FIELDS:
        raise ValueError("N32 result top-level schema fields drifted")
    if artifact.get("schema") == SMOKE_RESULT_SCHEMA:
        raise ValueError("N32 finalizer rejects smoke artifacts")
    if (
        artifact.get("schema") != RESULT_SCHEMA
        or artifact.get("authoritative") is not True
        or artifact.get("aggregation_eligible") is not True
        or artifact.get("promotion_eligible") is not False
        or _strict_int(artifact.get("seed"), context="N32 result seed")
        != expected_seed
        or artifact.get("categorical_radial_full_train_candidate_licensed")
        is not False
    ):
        raise ValueError("N32 result is not authoritative and aggregation eligible")
    core = dict(artifact)
    declared_content_hash = core.pop("content_sha256", None)
    if (
        not _is_sha256(declared_content_hash)
        or _canonical_json_sha256(core) != declared_content_hash
    ):
        raise ValueError("N32 result content hash mismatch")
    if (
        not isinstance(artifact.get("created_at_utc"), str)
        or not artifact["created_at_utc"]
        or not isinstance(artifact.get("completed_at_utc"), str)
        or not artifact["completed_at_utc"]
    ):
        raise ValueError("N32 result timestamps are missing")
    invocation = artifact.get("invocation")
    if (
        not isinstance(invocation, list)
        or not invocation
        or any(not isinstance(item, str) for item in invocation)
    ):
        raise ValueError("N32 invocation provenance is missing")
    execution = _mapping(artifact.get("execution"), context="N32 execution")
    if (
        _strict_int(
            execution.get("batch_size_frames"),
            context="N32 execution batch_size_frames",
        )
        != 4
        or _strict_int(
            execution.get("evaluation_interval"),
            context="N32 execution evaluation_interval",
        )
        != 100
        or execution.get("branches") != BRANCH_CONFIGS
        or execution.get("fp32_no_autocast_amp_compile_or_quantization") is not True
    ):
        raise ValueError("N32 execution contract drift")
    branches = _mapping(execution.get("branches"), context="N32 execution branches")
    for name in BRANCH_CONFIGS:
        branch = _mapping(branches.get(name), context=f"N32 execution branch {name}")
        _strict_int(branch.get("updates"), context=f"N32 {name} updates", minimum=1)
        for field in ("learning_rate", "weight_decay"):
            _finite_number(
                branch.get(field),
                context=f"N32 {name} {field}",
            )
    determinism = _mapping(execution.get("determinism"), context="N32 determinism")
    expected_determinism = {
        "seed": expected_seed,
        "requested": "strict_deterministic_algorithms",
        "effective": "strict_where_supported_warn_on_unsupported",
        "warn_only": True,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    _strict_int(determinism.get("seed"), context="N32 determinism seed")
    for name in (
        "warn_only",
        "torch_deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ):
        _strict_bool(determinism.get(name), context=f"N32 determinism/{name}")
    if determinism != expected_determinism:
        raise ValueError("N32 execution seed drift")
    contract = _mapping(artifact.get("contract"), context="N32 execution binding")
    if (
        Path(str(contract.get("path", ""))).resolve() != CONTRACT_PATH.resolve()
        or contract.get("sha256") != EXECUTION_BINDING_SHA256
    ):
        raise ValueError("N32 execution binding drift")
    inputs = _validate_inputs(artifact.get("inputs"), seed=expected_seed)
    sources = _validate_source_hashes(artifact.get("source_hashes"))
    _mapping(artifact.get("git"), context="N32 git provenance")
    model = _mapping(artifact.get("model"), context="N32 model")
    initial_state = model.get("initial_state_sha256")
    if (
        model.get("class") != "CategoricalRadialPerceptionFullRay"
        or _strict_int(
            model.get("parameter_count"),
            context="N32 model parameter_count",
            minimum=1,
        )
        != REGISTERED_PARAMETER_COUNT
        or not _is_sha256(initial_state)
        or model.get("all_invoked_branches_restart_same_initial_state") is not True
        or initial_state
        != {
            20260710: EXPECTED_SEED10_INITIAL_STATE_SHA256,
            20260711: EXPECTED_SEED11_INITIAL_STATE_SHA256,
        }[expected_seed]
    ):
        raise ValueError("N32 model mechanism or initialization drift")
    reference = _validate_patch7_reference(artifact.get("patch7_reference"))
    stages = _mapping(artifact.get("stages"), context="N32 stages")
    if set(stages) != set(BRANCH_CONFIGS):
        raise ValueError("N32 stage structure is incomplete")
    faithful, faithful_summary = _validate_stage(
        stages["production_faithful"],
        stage_name="production_faithful",
        seed=expected_seed,
        expected_initial_state=initial_state,
        expected_controls=(
            None if expected_controls is None else expected_controls["fit"]
        ),
    )
    ceiling_raw = stages["ceiling_optimizer"]
    if faithful_summary["passes"]:
        if ceiling_raw is not None:
            raise ValueError("ceiling is forbidden after a faithful fit pass")
        ceiling = None
        ceiling_summary = None
    else:
        ceiling, ceiling_summary = _validate_stage(
            ceiling_raw,
            stage_name="ceiling_optimizer",
            seed=expected_seed,
            expected_initial_state=initial_state,
            expected_controls=(
                None if expected_controls is None else expected_controls["fit"]
            ),
        )
        if ceiling["minibatch_indices"][:2000] != faithful["minibatch_indices"]:
            raise ValueError("faithful/ceiling minibatch prefixes differ")
    qualifying_stage = (
        "production_faithful"
        if faithful_summary["passes"]
        else "ceiling_optimizer"
        if ceiling_summary is not None and ceiling_summary["passes"]
        else None
    )
    holdouts = artifact.get("holdouts")
    stored_checks = artifact.get("holdout_checks")
    if qualifying_stage is None:
        if holdouts is not None or stored_checks is not None:
            raise ValueError("holdouts are forbidden when both fit branches fail")
        recomputed_checks = None
    else:
        holdout_map = _mapping(holdouts, context="N32 holdouts")
        check_map = _mapping(stored_checks, context="N32 holdout checks")
        if set(holdout_map) != set(HOLDOUT_PANELS) or set(check_map) != set(
            HOLDOUT_PANELS
        ):
            raise ValueError("authorized holdout payload is incomplete")
        recomputed_checks = {}
        for panel in HOLDOUT_PANELS:
            candidate = _validate_panel_report(
                holdout_map[panel],
                panel=panel,
                seed=expected_seed,
                require_fit_gate=False,
                expected_controls=(
                    None if expected_controls is None else expected_controls[panel]
                ),
            )
            recomputed_checks[panel] = categorical_holdout_checks(
                candidate,
                reference["panels"][panel],
            )
        if check_map != recomputed_checks:
            raise ValueError("stored categorical holdout comparison does not recompute")
        for panel in HOLDOUT_PANELS:
            _validate_gate_boolean_types(
                check_map[panel],
                context=f"{panel} holdout checks",
            )
            for name in (
                "strictly_favorable_family_count",
                "strictly_favorable_family_requirement",
            ):
                _strict_int(
                    check_map[panel].get(name),
                    context=f"{panel} holdout checks/{name}",
                )
    for stage_name, stage in (
        ("production_faithful", faithful),
        ("ceiling_optimizer", ceiling),
    ):
        if stage is None:
            continue
        if stage.get("holdouts_evaluated") is not (stage_name == qualifying_stage):
            raise ValueError("conditional stage holdout flag drift")
    recomputed_decision = per_seed_decision(faithful, ceiling, recomputed_checks)
    stored_decision = _mapping(artifact.get("decision"), context="stored N32 decision")
    for name in (
        "production_faithful_fit_passes",
        "ceiling_optimizer_invoked",
        "favorable",
        "aggregation_eligible",
        "categorical_radial_full_train_candidate_licensed",
        "promotion_licensed",
    ):
        _strict_bool(stored_decision.get(name), context=f"stored N32 decision/{name}")
    ceiling_pass = stored_decision.get("ceiling_optimizer_fit_passes")
    if ceiling_pass is not None:
        _strict_bool(
            ceiling_pass,
            context="stored N32 decision/ceiling_optimizer_fit_passes",
        )
    holdout_passes = stored_decision.get("holdout_passes")
    if holdout_passes is not None:
        holdout_pass_map = _mapping(
            holdout_passes,
            context="stored N32 decision/holdout_passes",
        )
        for panel in HOLDOUT_PANELS:
            _strict_bool(
                holdout_pass_map.get(panel),
                context=f"stored N32 decision/holdout_passes/{panel}",
            )
    if stored_decision != recomputed_decision:
        raise ValueError("stored N32 seed decision does not recompute")
    _validate_artifact_verification(artifact.get("artifact_verification"))
    _validate_panel_access(
        artifact.get("access_ledger"),
        holdouts_authorized=qualifying_stage is not None,
    )
    access = _mapping(artifact.get("access_ledger"), context="access ledger")
    totals = _mapping(access.get("fit_dataset_totals"), context="fit dataset totals")
    invoked_stages = [faithful, *([] if ceiling is None else [ceiling])]
    faithful_training = faithful["training_access"]
    faithful_evaluation = faithful["fit_evaluation_access"]
    if (
        _strict_int(
            faithful_training["image_decode_events"],
            context="faithful training image_decode_events",
        )
        != 320
        or _strict_int(
            faithful_training["label_shard_npz_open_events"],
            context="faithful training label_shard_npz_open_events",
        )
        != 20
        or _strict_int(
            faithful_evaluation["image_decode_events"],
            context="faithful evaluation image_decode_events",
        )
        != 0
        or _strict_int(
            faithful_evaluation["label_shard_npz_open_events"],
            context="faithful evaluation label_shard_npz_open_events",
        )
        != 0
    ):
        raise ValueError("faithful fit cache chronology does not reconcile")
    if ceiling is not None:
        if any(
            _strict_int(
                stage_access[name],
                context=f"ceiling {name}",
            )
            != 0
            for stage_access in (
                ceiling["training_access"],
                ceiling["fit_evaluation_access"],
            )
            for name in ("image_decode_events", "label_shard_npz_open_events")
        ):
            raise ValueError("ceiling reopened cached fit payloads")
    expected_image_requests = sum(
        _strict_int(stage["completed_steps"], context="stage completed_steps") * 4
        + len(stage["learning_curve"]) * 960
        for stage in invoked_stages
    )
    expected_target_requests = sum(
        _strict_int(stage["completed_steps"], context="stage completed_steps") * 4
        + len(stage["learning_curve"]) * 320
        for stage in invoked_stages
    )
    if (
        _strict_int(totals.get("image_requests"), context="fit totals image_requests")
        != expected_image_requests
        or _strict_int(
            totals.get("target_requests"), context="fit totals target_requests"
        )
        != expected_target_requests
        or _strict_int(
            totals.get("image_decode_events"),
            context="fit totals image_decode_events",
        )
        != sum(
            _strict_int(
                stage[access_name]["image_decode_events"],
                context=f"{access_name} image_decode_events",
            )
            for stage in invoked_stages
            for access_name in ("training_access", "fit_evaluation_access")
        )
        or _strict_int(
            totals.get("label_shard_npz_open_events"),
            context="fit totals label_shard_npz_open_events",
        )
        != sum(
            _strict_int(
                stage[access_name]["label_shard_npz_open_events"],
                context=f"{access_name} label_shard_npz_open_events",
            )
            for stage in invoked_stages
            for access_name in ("training_access", "fit_evaluation_access")
        )
    ):
        raise ValueError("fit request access does not reconcile across invoked stages")
    return {
        "content_sha256": declared_content_hash,
        "inputs": inputs,
        "source_hashes": sources,
        "contract": contract,
        "model_mechanism": {
            "class": model["class"],
            "parameter_count": model["parameter_count"],
        },
        "patch7_reference": reference,
        "decision": recomputed_decision,
        "qualifying_stage": qualifying_stage,
    }


def _common_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {name: inputs[name] for name in EXPECTED_INPUTS}


def _aggregate_validated_results(
    primary: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> dict[str, Any]:
    primary_decision = _mapping(primary["decision"], context="primary decision")
    replication_decision = _mapping(
        replication["decision"],
        context="replication decision",
    )
    both_favorable = bool(primary_decision["favorable"]) and bool(
        replication_decision["favorable"]
    )
    primary_stage = primary_decision.get("qualifying_optimizer_stage")
    replication_stage = replication_decision.get("qualifying_optimizer_stage")
    same_stage = primary_stage is not None and primary_stage == replication_stage
    licensed = both_favorable and same_stage
    if licensed:
        classification = "two_seed_favorable_same_branch"
    elif both_favorable:
        classification = "two_seed_inconclusive"
    else:
        classification = "two_seed_replication_failed"
    return {
        "classification": classification,
        "seeds": list(EXPECTED_SEEDS),
        "both_seeds_favorable": both_favorable,
        "same_qualifying_optimizer_stage": same_stage,
        "qualifying_optimizer_stage": primary_stage if licensed else None,
        "seed_decisions": {
            str(EXPECTED_SEEDS[0]): dict(primary_decision),
            str(EXPECTED_SEEDS[1]): dict(replication_decision),
        },
        "categorical_radial_full_train_candidate_licensed": licensed,
        "promotion_licensed": False,
        "g2_licensed": False,
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "execution_binding": RUNNER_SOURCE_BINDINGS["n32_contract"],
        "finalizer": Path(__file__).resolve(),
        "n32_pure": RUNNER_SOURCE_BINDINGS["n32_pure"],
        "runner": RUNNER_SOURCE_BINDINGS["runner"],
    }
    result = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    result.update(
        {
            f"runner_bound_{name}": record
            for name, record in _runner_source_hashes().items()
        }
    )
    return dict(sorted(result.items()))


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-20260710-result", type=Path, required=True)
    parser.add_argument(
        "--expected-seed-20260710-result-sha256",
        required=True,
    )
    parser.add_argument("--seed-20260711-result", type=Path, required=True)
    parser.add_argument(
        "--expected-seed-20260711-result-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; N32 finalization artifacts are immutable")
    if args.seed_20260710_result.resolve() == args.seed_20260711_result.resolve():
        parser.error("the two seed inputs must be distinct files")
    for seed, path in (
        (20260710, args.seed_20260710_result),
        (20260711, args.seed_20260711_result),
    ):
        if path.resolve() != CANONICAL_RESULT_PATHS[seed].resolve():
            parser.error(f"seed {seed} result path is not canonical")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bound_evidence = _load_bound_evidence()
    controls = {
        seed: _expected_controls(bound_evidence["panels"], seed=seed)
        for seed in EXPECTED_SEEDS
    }
    source_start = _source_hashes()
    paths = {
        20260710: args.seed_20260710_result.resolve(),
        20260711: args.seed_20260711_result.resolve(),
    }
    expected_hashes = {
        20260710: str(args.expected_seed_20260710_result_sha256),
        20260711: str(args.expected_seed_20260711_result_sha256),
    }
    payloads: dict[int, dict[str, Any]] = {}
    ledgers: dict[int, dict[str, Any]] = {}
    validated: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        payloads[seed], ledgers[seed] = _load_expected_json(
            paths[seed],
            expected_sha256=expected_hashes[seed],
        )
        validated[seed] = _validate_authoritative_result(
            payloads[seed],
            expected_seed=seed,
            expected_controls=controls[seed],
        )
        if validated[seed]["patch7_reference"] != bound_evidence["patch7_reference"]:
            raise ValueError("N32 seed embeds a drifted bound patch7 reference")
    for field in ("source_hashes", "contract", "model_mechanism", "patch7_reference"):
        if _canonical_json(validated[20260710][field]) != _canonical_json(
            validated[20260711][field]
        ):
            raise ValueError(f"two N32 seeds disagree on common {field}")
    if _canonical_json(_common_inputs(validated[20260710]["inputs"])) != _canonical_json(
        _common_inputs(validated[20260711]["inputs"])
    ):
        raise ValueError("two N32 seeds disagree on common immutable inputs")
    authorization = _mapping(
        validated[20260711]["inputs"]["seed_20260710_authorization"],
        context="seed 20260710 authorization",
    )
    if authorization.get("sha256") != expected_hashes[20260710]:
        raise ValueError("seed 20260711 authorization does not bind the primary input")
    if not bool(validated[20260710]["decision"]["favorable"]):
        raise ValueError("seed 20260711 exists without a favorable primary authorization")
    aggregation = _aggregate_validated_results(
        validated[20260710],
        validated[20260711],
    )
    for seed in EXPECTED_SEEDS:
        final_hash = _sha256_file(paths[seed])
        if final_hash != ledgers[seed]["pre_deserialization_sha256"]:
            raise RuntimeError(f"seed {seed} result changed during finalization")
        ledgers[seed]["post_validation_sha256"] = final_hash
        ledgers[seed]["post_validation_unchanged"] = True
        ledgers[seed]["content_sha256"] = validated[seed]["content_sha256"]
        ledgers[seed]["decision_recomputed_exactly"] = True
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("N32 finalizer sources changed during finalization")
    evidence_end = {
        str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE
    }
    if evidence_end != bound_evidence["pre_hashes"]:
        raise RuntimeError("bound N32 evidence changed during finalization")
    core = {
        "schema": TWO_SEED_RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_inputs_only": True,
        "aggregation_eligible_inputs_only": True,
        "input_hash_verification": [ledgers[seed] for seed in EXPECTED_SEEDS],
        "bound_evidence_hash_verification": {
            "pre_deserialization": bound_evidence["pre_hashes"],
            "post_deserialization": bound_evidence["post_parse_hashes"],
            "post_finalization": evidence_end,
            "unchanged": True,
        },
        "common_provenance_validated": {
            "immutable_inputs": True,
            "source_hashes": True,
            "execution_binding": True,
            "model_mechanism": True,
            "patch7_reference": True,
        },
        "stored_seed_decisions_recomputed_from_raw_metrics": True,
        "aggregation": aggregation,
        "source_hashes": source_end,
        "categorical_radial_full_train_candidate_licensed": aggregation[
            "categorical_radial_full_train_candidate_licensed"
        ],
    }
    result = {**core, "content_sha256": _canonical_json_sha256(core)}
    _atomic_write_json_exclusive(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_sha256": result["content_sha256"],
                "classification": aggregation["classification"],
                "categorical_radial_full_train_candidate_licensed": result[
                    "categorical_radial_full_train_candidate_licensed"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
