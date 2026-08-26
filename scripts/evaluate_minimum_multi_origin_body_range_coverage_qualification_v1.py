#!/usr/bin/env python3
"""Execute the development-only minimum multi-origin range qualification.

The evaluator reuses the frozen BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1
geometry and decision reducers.  It never steps Genesis, imports a learned
predictor, trains a model, or changes a corpus/action/contact identity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import gzip
import hashlib
from functools import lru_cache
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import minimum_multi_origin_body_range_coverage_qualification_v1_contract as CONTRACT
from scripts import evaluate_body_centric_range_coverage_qualification_v1 as BASE


EXPERIMENT = "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1"
EXPECTED_BRANCH = "jepa-spatial-world-model-nav"
PREDECESSOR_RESULT_COMMIT = "d9748abe0fad0a25face56801f6b0c5e699db92f"
PREDECESSOR_FREEZE_COMMIT = "6fb55dec810b8fb8337d4519096f17a294c78425"
PREDECESSOR_RESULT_CONTENT_DIGEST = "98699ac43046a4d1f425998217d5527637209ca233a872cf667e484123a967ac"
PREDECESSOR_RESULT_FILE_SHA256 = "8844c0ee5a8bcdd28f505d64a670b5dee595933af05a00290f327cb8f3702019"
PREDECESSOR_MATERIALIZATION_INDEX_SHA256 = "4e35e813d6f1bc2593c6a889a24e3de952b2797ce48f2c28224cfe16e5af398d"
PREDECESSOR_THRESHOLD_FREEZE_SHA256 = "8bda91c4127ec2a43e877f64ce58cde39379682bd674dc1354166290ec61b75e"
PREDECESSOR_PERSISTENCE_RECEIPT_SHA256 = "12bdb2e6e0e7b54a9ad1947126715ac85d367afcbfd403ca1c80ebd593f51a43"
OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
TRACKED_PREREG = ROOT / CONTRACT.TRACKED_PREREGISTRATION_PATH
TRACKED_CONTRACT = ROOT / CONTRACT.TRACKED_CONTRACT_RECEIPT_PATH
TRACKED_SCHEMA = ROOT / CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH
TRACKED_MOUNTS = ROOT / CONTRACT.TRACKED_MOUNT_LIBRARY_PATH
TRACKED_CLOSURE = ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH
TRACKED_FIXTURE = ROOT / "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_fixture_2026-08-26.json"
RESULT_JSON = ROOT / "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_result_2026-08-26.json"
RESULT_MD = ROOT / "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_result_2026-08-26.md"
PREDECESSOR_RESULT = ROOT / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_result_2026-08-25.json"
PREDECESSOR_OUTPUT = Path("/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/body_centric_range_coverage_qualification_v1")

EXPECTED = {
    "corpus_logical_digest": "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223",
    "corpus_index_sha256": "c1055724ebffd71c67b5424b4e447d223a828d3bc4da01369486bff73ad265f0",
    "action_contract_sha256": "cf8df092e8eff61d04348ebfe22b5e6a0cd31b5f39a4e05e45242752b3e5dc06",
    "repaired_row_ledger_sha256": "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94",
    "states": 176,
    "training_states": 128,
    "calibration_states": 24,
    "heldout_states": 24,
    "transitions": 29_470,
    "physics_frames": 1_473_500,
    "protected_links": 13,
    "action_representatives": 13_385,
    "geometry_representatives": 13_584,
}
STEPS = 50
LINKS = 13
MODES = tuple(CONTRACT.EVIDENCE_MODE_IDS)
MULTI_CONDITIONS = tuple(
    list(CONTRACT.DUAL_CONDITION_IDS) + list(CONTRACT.THREE_CONDITION_IDS)
)
REGRESSION_CONDITIONS = tuple(CONTRACT.REGRESSION_CONDITION_IDS)
DIAGNOSTIC_CONDITION = tuple(CONTRACT.DIAGNOSTIC_CONDITION_IDS)[0]
STORED_FLOAT_EVIDENCE_FIELDS = (
    "clearance_m",
    "point_age_s",
    "support_point_age_s",
    "obstacle_direction_body_rad",
)
STORED_BOOL_EVIDENCE_FIELDS = (
    "support",
    "event_time_support",
    "nominal_fov",
    "direct_visibility",
    "self_occluded",
    "near_blind",
    "finite_scan_support_inherited",
)
STORED_INT16_EVIDENCE_FIELDS = (
    "responsible_object_index",
    "point_support_count",
)


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def canonical_bytes(value: Any) -> bytes:
    return BASE.canonical_bytes(value)


def content_digest(value: Any) -> str:
    return BASE.content_digest(value)


def atomic_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    BASE.atomic_json(path, value, canonical=canonical)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    BASE.atomic_npz(path, **arrays)


def git(*args: str) -> str:
    return BASE.git(*args)


def filesystem_receipt(path: Path) -> dict[str, Any]:
    return BASE.filesystem_receipt(path)


def _directory_allocated_bytes(path: Path) -> int:
    return BASE._directory_allocated_bytes(path)


@lru_cache(maxsize=1)
def _assert_predecessor_authority() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != PREDECESSOR_RESULT_COMMIT and subprocess.call(
        ["git", "merge-base", "--is-ancestor", PREDECESSOR_RESULT_COMMIT, "HEAD"],
        cwd=ROOT,
    ):
        raise RuntimeError("the required predecessor result is not an ancestor")
    if sha256_file(PREDECESSOR_RESULT) != PREDECESSOR_RESULT_FILE_SHA256:
        raise RuntimeError("tracked predecessor result file SHA-256 drift")
    result = json.loads(PREDECESSOR_RESULT.read_text())
    result_core = dict(result)
    declared_result_digest = result_core.pop("result_content_sha256", None)
    if declared_result_digest != content_digest(result_core):
        raise RuntimeError("predecessor result self-digest drift")
    if result.get("result_content_sha256") != PREDECESSOR_RESULT_CONTENT_DIGEST:
        raise RuntimeError("predecessor result content digest drift")
    if result.get("source_freeze_commit") != PREDECESSOR_FREEZE_COMMIT:
        raise RuntimeError("predecessor source-freeze binding drift")
    if result.get("primary_classification") != "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO":
        raise RuntimeError("predecessor single-origin classification drift")
    result_path = PREDECESSOR_OUTPUT / "result.json"
    materialization_path = PREDECESSOR_OUTPUT / "materialization_index.json"
    threshold_path = PREDECESSOR_OUTPUT / "calibration/thresholds_frozen.json"
    persistence_path = PREDECESSOR_OUTPUT / "persistence_receipt.json"
    for path in (result_path, materialization_path, threshold_path, persistence_path):
        if not path.is_file():
            raise RuntimeError(f"predecessor authority is missing: {path}")
    if result_path.read_bytes() != PREDECESSOR_RESULT.read_bytes():
        raise RuntimeError("tracked/external predecessor result mismatch")
    if sha256_file(materialization_path) != PREDECESSOR_MATERIALIZATION_INDEX_SHA256:
        raise RuntimeError("predecessor materialization-index SHA-256 drift")
    if sha256_file(threshold_path) != PREDECESSOR_THRESHOLD_FREEZE_SHA256:
        raise RuntimeError("predecessor threshold-freeze SHA-256 drift")
    if sha256_file(persistence_path) != PREDECESSOR_PERSISTENCE_RECEIPT_SHA256:
        raise RuntimeError("predecessor persistence-receipt SHA-256 drift")
    materialization = json.loads(materialization_path.read_text())
    materialization_core = dict(materialization)
    declared_materialization_digest = materialization_core.pop(
        "content_digest", None
    )
    if (
        materialization.get("status") != "PASS"
        or declared_materialization_digest != content_digest(materialization_core)
        or int(materialization.get("states", -1)) != EXPECTED["states"]
        or int(materialization.get("transitions", -1)) != EXPECTED["transitions"]
    ):
        raise RuntimeError("predecessor materialization-index custody drift")
    persistence = json.loads(persistence_path.read_text())
    persistence_core = dict(persistence)
    declared_persistence_digest = persistence_core.pop("content_digest", None)
    if (
        persistence.get("pass") is not True
        or declared_persistence_digest != content_digest(persistence_core)
        or persistence.get("materialization_index", {}).get("sha256")
        != PREDECESSOR_MATERIALIZATION_INDEX_SHA256
        or persistence.get("threshold_freeze_receipt", {}).get("sha256")
        != PREDECESSOR_THRESHOLD_FREEZE_SHA256
        or persistence.get("result", {}).get("sha256")
        != PREDECESSOR_RESULT_FILE_SHA256
    ):
        raise RuntimeError("predecessor persistence custody drift")
    threshold = json.loads(threshold_path.read_text())
    threshold_core = dict(threshold)
    declared_threshold_digest = threshold_core.pop("content_digest", None)
    if (
        threshold.get("pass") is not True
        or declared_threshold_digest != content_digest(threshold_core)
        or threshold.get("materialization_index_sha256")
        != PREDECESSOR_MATERIALIZATION_INDEX_SHA256
    ):
        raise RuntimeError("predecessor threshold-freeze custody drift")
    required_arrays = {
        "target_points_world_m": (STEPS, LINKS, 3),
        "target_object_index": (STEPS, LINKS),
        "target_geom_index": (STEPS, LINKS),
        "oracle_clearance_m": (STEPS, LINKS),
        "action_representative_transition": (),
        "geometry_representative_transition": (),
    }
    for record in materialization["records"]:
        state_id = str(record["state_id"])
        shard = Path(record["shard_path"])
        if (
            not shard.is_file()
            or shard.stat().st_size != int(record["storage_bytes"])
            or sha256_file(shard) != record["shard_sha256"]
        ):
            raise RuntimeError(f"predecessor state shard custody drift: {state_id}")
        state_receipt_path = PREDECESSOR_OUTPUT / "states" / f"{state_id}.json"
        state_receipt = json.loads(state_receipt_path.read_text())
        state_core = dict(state_receipt)
        declared_state_digest = state_core.pop("content_digest", None)
        if (
            state_receipt.get("status") != "PASS"
            or declared_state_digest != content_digest(state_core)
            or state_receipt.get("shard_sha256") != record["shard_sha256"]
        ):
            raise RuntimeError(f"predecessor state receipt custody drift: {state_id}")
        transitions = int(record["transitions"])
        with np.load(shard, allow_pickle=False) as archive:
            for field, suffix in required_arrays.items():
                if field not in archive:
                    raise RuntimeError(
                        f"predecessor state shard missing {field}: {state_id}"
                    )
                expected_shape = (transitions, *suffix)
                if archive[field].shape != expected_shape:
                    raise RuntimeError(
                        f"predecessor {state_id}/{field} shape drift"
                    )
            action_map = np.asarray(
                archive["action_representative_transition"], np.int32
            )
            geometry_map = np.asarray(
                archive["geometry_representative_transition"], np.int32
            )
            if (
                np.any(action_map < 0)
                or np.any(action_map >= transitions)
                or np.any(geometry_map < 0)
                or np.any(geometry_map >= transitions)
                or not np.array_equal(action_map[action_map], action_map)
                or not np.array_equal(geometry_map[geometry_map], geometry_map)
            ):
                raise RuntimeError(
                    f"predecessor representative-map custody drift: {state_id}"
                )
    result["predecessor_materialization_index_sha256"] = (
        PREDECESSOR_MATERIALIZATION_INDEX_SHA256
    )
    result["predecessor_persistence_receipt_sha256"] = (
        PREDECESSOR_PERSISTENCE_RECEIPT_SHA256
    )
    return result


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "README.md",
        ROOT / "config/go2_platform_manifest.yaml",
        ROOT / "lewm_genesis/lewm_genesis/rollout.py",
        ROOT / "lewm/__init__.py",
        ROOT / "lewm/safety/__init__.py",
        ROOT / "lewm/safety/body_centric_range_coverage_qualification_v1_contract.py",
        ROOT / "lewm/safety/body_centric_range_coverage_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_metrics_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_analysis_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_corpus_v1.py",
        ROOT / "lewm/safety/body_centric_range_coverage_reporting_v1.py",
        ROOT / "lewm/safety/minimum_multi_origin_body_range_coverage_v1.py",
        ROOT / "lewm/safety/minimum_multi_origin_body_range_coverage_metrics_v1.py",
        ROOT / "lewm/safety/minimum_multi_origin_body_range_coverage_qualification_v1_contract.py",
        ROOT / "scripts/evaluate_body_centric_range_coverage_qualification_v1.py",
        Path(__file__).resolve(),
        ROOT / "lewm/tests/test_minimum_multi_origin_body_range_coverage_v1.py",
        ROOT / "lewm/tests/test_minimum_multi_origin_body_range_coverage_metrics_v1.py",
        ROOT / "lewm/tests/test_minimum_multi_origin_body_range_coverage_qualification_v1_contract.py",
        ROOT / "lewm/tests/test_evaluate_minimum_multi_origin_body_range_coverage_qualification_v1.py",
        TRACKED_PREREG,
        TRACKED_CONTRACT,
        TRACKED_SCHEMA,
        TRACKED_MOUNTS,
        TRACKED_FIXTURE,
    )


def build_source_closure() -> dict[str, Any]:
    rows = []
    for path in _source_paths():
        if not path.is_file():
            raise RuntimeError(f"source closure member missing: {path}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    value = {
        "schema": "minimum_multi_origin_body_range_coverage_source_closure_v1",
        "experiment": EXPERIMENT,
        "files": rows,
        "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
        "excludes": [
            "model training",
            "fresh state collection",
            "JEPA predictor",
            "G2 evaluation",
            "memory and navigation systems",
        ],
        "source_freeze_commit": "RECORDED_BY_POST_COMMIT_PREFLIGHT",
    }
    value["content_digest"] = content_digest(value)
    return value


def _combined_fixture_receipt() -> dict[str, Any]:
    from lewm.safety import minimum_multi_origin_body_range_coverage_v1 as core
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    core_receipt = core.run_fixtures()
    metrics_receipt = metrics.run_fixtures()
    value = {
        "schema": "minimum_multi_origin_body_range_coverage_fixture_v1",
        "experiment": EXPERIMENT,
        "core": core_receipt,
        "metrics": metrics_receipt,
        "pass": bool(core_receipt.get("pass") and metrics_receipt.get("pass")),
    }
    value["content_digest"] = content_digest(value)
    return value


def write_freeze_receipts() -> dict[str, Any]:
    from lewm.safety import minimum_multi_origin_body_range_coverage_v1 as core

    mount_receipt = core.build_mount_library_receipt()
    if mount_receipt.get("pass") is not True:
        raise RuntimeError("static label-free mount/orientation selection failed")
    atomic_json(TRACKED_MOUNTS, mount_receipt, canonical=True)
    CONTRACT.write_contract(TRACKED_CONTRACT)
    CONTRACT.write_output_schema(TRACKED_SCHEMA)
    CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    CONTRACT.load_and_validate_mount_library(TRACKED_MOUNTS)
    fixture = _combined_fixture_receipt()
    if fixture.get("pass") is not True:
        raise RuntimeError("deterministic multi-origin fixture gate failed")
    atomic_json(TRACKED_FIXTURE, fixture, canonical=True)
    closure = build_source_closure()
    atomic_json(TRACKED_CLOSURE, closure, canonical=True)
    return {
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "schema_sha256": sha256_file(TRACKED_SCHEMA),
        "mount_library_sha256": sha256_file(TRACKED_MOUNTS),
        "fixture_sha256": sha256_file(TRACKED_FIXTURE),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_closure_content_digest": closure["content_digest"],
    }


def validate_source_closure() -> dict[str, Any]:
    closure = json.loads(TRACKED_CLOSURE.read_text())
    core = dict(closure)
    declared = core.pop("content_digest", None)
    if declared != content_digest(core):
        raise RuntimeError("source closure self-digest mismatch")
    for row in closure["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source closure drift: {path}")
    return closure


def _augment_environment_receipt(environment: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact direct import/source closure of this new evaluator."""

    value = dict(environment)
    value.pop("content_digest", None)
    new_sources = {
        "lewm.safety.minimum_multi_origin_body_range_coverage_v1": (
            ROOT / "lewm/safety/minimum_multi_origin_body_range_coverage_v1.py"
        ),
        "lewm.safety.minimum_multi_origin_body_range_coverage_metrics_v1": (
            ROOT / "lewm/safety/minimum_multi_origin_body_range_coverage_metrics_v1.py"
        ),
        "lewm.safety.minimum_multi_origin_body_range_coverage_qualification_v1_contract": (
            ROOT
            / "lewm/safety/minimum_multi_origin_body_range_coverage_qualification_v1_contract.py"
        ),
        "scripts.evaluate_body_centric_range_coverage_qualification_v1": (
            ROOT / "scripts/evaluate_body_centric_range_coverage_qualification_v1.py"
        ),
        "scripts.evaluate_minimum_multi_origin_body_range_coverage_qualification_v1": (
            Path(__file__).resolve()
        ),
    }
    source_rows = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in new_sources.items()
    }
    first_party = list(value["import_closure"]["first_party"])
    for name in ("scripts", *new_sources):
        if name not in first_party:
            first_party.append(name)
    stdlib = list(value["import_closure"]["stdlib"])
    for name in ("functools", "zipfile"):
        if name not in stdlib:
            stdlib.append(name)
    value["import_closure"] = {
        **value["import_closure"],
        "first_party": first_party,
        "stdlib": sorted(stdlib),
    }
    value["first_party_sources"] = {
        **value["first_party_sources"],
        **source_rows,
    }
    value["experiment_import_closure"] = {
        "first_party": first_party,
        "first_party_sources": source_rows,
        "namespace_packages": {
            "scripts": str(ROOT / "scripts"),
        },
        "stdlib": sorted(stdlib),
        "third_party": list(value["import_closure"]["third_party"]),
        "not_imported": list(value["import_closure"]["not_imported"]),
        "entrypoint_runtime_name": "__main__",
    }
    required_loaded = set(new_sources) - {
        "scripts.evaluate_minimum_multi_origin_body_range_coverage_qualification_v1"
    }
    missing_loaded = sorted(required_loaded - set(sys.modules))
    if missing_loaded:
        raise RuntimeError(
            f"experiment first-party import closure is incomplete: {missing_loaded}"
        )
    value["content_digest"] = content_digest(value)
    return value


def preflight() -> dict[str, Any]:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("preflight requires the committed clean freeze")
    head = git("rev-parse", "HEAD")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("wrong execution branch")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", PREDECESSOR_RESULT_COMMIT, head],
        cwd=ROOT,
    ):
        raise RuntimeError("predecessor result commit is not an ancestor")
    active_processes = BASE.running_scientific_processes()
    if active_processes:
        raise RuntimeError(f"scientific process already active: {active_processes}")
    if OUTPUT_ROOT.exists() and any(path.is_file() for path in OUTPUT_ROOT.rglob("*")):
        raise RuntimeError("canonical multi-origin output namespace is not empty")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = filesystem_receipt(ROOT)
    output = filesystem_receipt(OUTPUT_ROOT)
    if workspace["device"] == output["device"] or output["fstype"] != "ext4":
        raise RuntimeError("output is not the independent frozen ext4 target")
    if workspace["free_bytes"] < 20_000_000_000:
        raise RuntimeError("workspace has less than 20 GB free")
    if output["free_bytes"] < 100_000_000_000:
        raise RuntimeError("output filesystem has less than 100 GB free")
    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    mounts = CONTRACT.load_and_validate_mount_library(TRACKED_MOUNTS)
    closure = validate_source_closure()
    fixture = _combined_fixture_receipt()
    if TRACKED_FIXTURE.read_bytes() != canonical_bytes(fixture) or not fixture["pass"]:
        raise RuntimeError("fixture receipt does not regenerate byte-identically")
    predecessor = _assert_predecessor_authority()
    inputs = BASE.validate_frozen_inputs(full_shards=True)
    from lewm.safety import body_centric_range_coverage_analysis_v1 as _analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import body_centric_range_coverage_reporting_v1 as _reporting

    # These imports are part of the predecessor-proven exact first-party
    # closure required by BASE.environment_receipt().
    del _analysis, _reporting

    context = corpus_adapter.load_corpus_context(ROOT)
    geometry_audit = BASE.exact_geometry_materialization_corpus_audit(context, corpus_adapter)
    environment = _augment_environment_receipt(BASE.environment_receipt())
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered the no-training import closure")
    receipt = {
        "schema": "minimum_multi_origin_body_range_coverage_preexecution_v1",
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head": head,
        "branch": EXPECTED_BRANCH,
        "worktree_clean": True,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "contract_content_digest": contract["contract_sha256"],
        "output_schema_sha256": sha256_file(TRACKED_SCHEMA),
        "output_schema_content_digest": schema["output_schema_sha256"],
        "mount_library_sha256": sha256_file(TRACKED_MOUNTS),
        "mount_library_receipt_sha256": sha256_file(TRACKED_MOUNTS),
        "mount_library_content_digest": mounts["content_digest"],
        "fixture_sha256": sha256_file(TRACKED_FIXTURE),
        "fixture_content_digest": fixture["content_digest"],
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_closure_content_digest": closure["content_digest"],
        "predecessor_result_content_digest": predecessor["result_content_sha256"],
        "predecessor_bindings": {
            "result_commit": PREDECESSOR_RESULT_COMMIT,
            "contract_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
            "result_content_digest": PREDECESSOR_RESULT_CONTENT_DIGEST,
            "external_result_sha256": sha256_file(
                PREDECESSOR_OUTPUT / "result.json"
            ),
            "materialization_index_sha256": (
                PREDECESSOR_MATERIALIZATION_INDEX_SHA256
            ),
            "threshold_freeze_sha256": PREDECESSOR_THRESHOLD_FREEZE_SHA256,
            "persistence_receipt_sha256": (
                PREDECESSOR_PERSISTENCE_RECEIPT_SHA256
            ),
            "state_shards_validated": EXPECTED["states"],
        },
        "inputs": inputs,
        "exact_geometry_materialization_preflight": geometry_audit,
        "environment": environment,
        "workspace_filesystem": workspace,
        "output_filesystem": output,
        "temporary_storage_ceiling_bytes": 50_000_000_000,
        "final_storage_ceiling_bytes": 25_000_000_000,
        "predicted_temporary_storage_bytes": 35_000_000_000,
        "predicted_final_storage_bytes": 23_000_000_000,
        "training_authorized": False,
        "fresh_panel_authorized": False,
        "jepa_predictor_access_authorized": False,
        "g2_access_authorized": False,
        "no_active_scientific_process": True,
        "scientific_processes_before": active_processes,
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(OUTPUT_ROOT / "receipts/environment_receipt.json", environment)
    shutil.copy2(TRACKED_MOUNTS, OUTPUT_ROOT / "receipts/mount_library.json")
    atomic_json(OUTPUT_ROOT / "preexecution_receipt.json", receipt)
    return receipt


def validate_preflight() -> dict[str, Any]:
    path = OUTPUT_ROOT / "preexecution_receipt.json"
    if not path.is_file():
        raise RuntimeError("post-freeze preexecution receipt is missing")
    receipt = json.loads(path.read_text())
    core = dict(receipt)
    declared = core.pop("content_digest", None)
    if receipt.get("pass") is not True or declared != content_digest(core):
        raise RuntimeError("preexecution receipt is invalid")
    if receipt["head"] != git("rev-parse", "HEAD"):
        raise RuntimeError("preexecution source-freeze commit drift")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("scientific execution requires a clean worktree")
    if receipt["contract_sha256"] != sha256_file(TRACKED_CONTRACT):
        raise RuntimeError("preexecution contract drift")
    if receipt["source_closure_sha256"] != sha256_file(TRACKED_CLOSURE):
        raise RuntimeError("preexecution source closure drift")
    if receipt["inputs"].get("geometry_shard_hashes_checked") != EXPECTED["states"]:
        raise RuntimeError("full 176-shard input audit is required")
    workspace = filesystem_receipt(ROOT)
    output = filesystem_receipt(OUTPUT_ROOT)
    if workspace["free_bytes"] < 20_000_000_000 or output["free_bytes"] < 100_000_000_000:
        raise RuntimeError("execution storage headroom gate no longer passes")
    return receipt


@lru_cache(maxsize=1)
def _mount_entries() -> dict[str, dict[str, Any]]:
    receipt = CONTRACT.load_and_validate_mount_library(TRACKED_MOUNTS)
    source = receipt.get("mounts", receipt.get("mount_library"))
    if source is None and "mount_candidates" in receipt:
        candidates = {
            str(row["mount_id"]): dict(row)
            for row in receipt["mount_candidates"]
        }
        selections = {
            str(row["mount_id"]): dict(row["selected_pose"])
            for row in receipt["orientation_selections"]
        }
        source = {
            mount_id: {
                **candidate,
                "selected_orientation": selections[mount_id],
            }
            for mount_id, candidate in candidates.items()
        }
    if isinstance(source, list):
        entries = {str(row["mount_id"]): dict(row) for row in source}
    elif isinstance(source, dict):
        entries = {str(key): dict(value) for key, value in source.items()}
    else:
        raise RuntimeError("mount-library receipt has no mount entries")
    if tuple(entries) != tuple(CONTRACT.MOUNT_IDS):
        raise RuntimeError("mount-library identity/order drift")
    return entries


@lru_cache(maxsize=8)
def _mount_transform(mount_id: str) -> tuple[np.ndarray, np.ndarray]:
    entry = _mount_entries()[mount_id]
    translation = entry.get(
        "translation_m",
        entry.get("translation_xyz_m", entry.get("translation_body_xyz_m")),
    )
    orientation = entry.get("selected_orientation", entry)
    rotation = orientation.get(
        "rotation_rpy_rad", orientation.get("rpy_rad", orientation.get("rpy_body_rad"))
    )
    if translation is None or rotation is None:
        raise RuntimeError(f"mount transform is incomplete: {mount_id}")
    return np.asarray(translation, np.float64), np.asarray(rotation, np.float64)


@lru_cache(maxsize=1)
def _contract_content_digest() -> str:
    return str(
        CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)["contract_sha256"]
    )


def _layout_mounts(layout_id: str) -> tuple[str, ...]:
    mounts = tuple(str(value) for value in layout_id.split("__"))
    if not mounts or mounts[0] != "HEAD_STOCK":
        raise RuntimeError(f"layout lacks the frozen head origin: {layout_id}")
    if any(value not in CONTRACT.MOUNT_IDS for value in mounts):
        raise RuntimeError(f"unknown mount in layout: {layout_id}")
    return mounts


def _append_other_sensor_housings(
    *,
    emitting_mount_id: str,
    installed_mount_ids: tuple[str, ...],
    specs: Sequence[Any],
    positions: np.ndarray,
    quaternions: np.ndarray,
    base_positions: np.ndarray,
    base_quaternions: np.ndarray,
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Add every *other* supplemental L2 housing as a ray-only occluder.

    The emitting sensor's own housing is prospectively aperture-exempt.  The
    stock head assembly remains represented by the frozen Go2 robot shapes;
    supplemental housings are 75 x 75 x 65 mm rigid trunk-frame OBBs and do
    not enter protected/contact geometry or clearance reduction.
    """

    from lewm.safety import body_centric_range_coverage_v1 as geometry

    output_specs = list(specs)
    output_positions = np.asarray(positions, np.float64)
    output_quaternions = np.asarray(quaternions, np.float64)
    base_position = np.asarray(base_positions, np.float64)
    base_quaternion = np.asarray(base_quaternions, np.float64)
    if base_position.shape != (len(output_positions), 3):
        raise RuntimeError("sensor-housing base-position alignment drift")
    if base_quaternion.shape != (len(output_positions), 4):
        raise RuntimeError("sensor-housing base-quaternion alignment drift")
    supplemental = tuple(
        mount_id
        for mount_id in installed_mount_ids
        if mount_id != "HEAD_STOCK" and mount_id != emitting_mount_id
    )
    for mount_id in supplemental:
        translation, rotation = _mount_transform(mount_id)
        housing_position, _optical_quaternion = BASE.mounted_sensor_poses(
            base_position,
            base_quaternion,
            translation_m=translation,
            rotation_rpy_rad=rotation,
        )
        # The prospectively frozen mechanical/occlusion envelope is aligned
        # to the trunk frame.  Optical-ray orientation is independent because
        # rotating the coarse 75x75x65 mm box at the frozen flank centres would
        # violate their exact 10 mm body-envelope clearance construction.
        housing_quaternion = np.asarray(base_quaternion, np.float64)
        output_specs.append(
            geometry.RobotPrimitiveSpec(
                identity=f"installed_sensor_housing:{mount_id}",
                kind="box",
                data=(0.075, 0.075, 0.065),
                local_position_xyz_m=(0.0, 0.0, 0.0),
                local_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                geom_index=100 + CONTRACT.MOUNT_IDS.index(mount_id),
                link_index=0,
                link_name=f"sensor_housing:{mount_id}",
            )
        )
        output_positions = np.concatenate(
            (output_positions, np.asarray(housing_position)[:, None, :]), axis=1
        )
        output_quaternions = np.concatenate(
            (output_quaternions, np.asarray(housing_quaternion)[:, None, :]), axis=1
        )
    return output_specs, output_positions, output_quaternions


def _base_pose_rows_for_robot_pose_rows(
    robot_positions: np.ndarray,
    robot_quaternions: np.ndarray,
    *,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the acquisition base pose aligned to repeated raycast rows."""

    candidate_qpos = np.concatenate(
        (np.asarray(boundary_qpos, np.float64)[None], np.asarray(qpos, np.float64)),
        axis=0,
    )
    candidate_geom = np.concatenate(
        (
            np.asarray(boundary_geom_transform, np.float64)[None],
            np.asarray(geom_transform, np.float64),
        ),
        axis=0,
    )
    lookup: dict[bytes, int] = {}
    for index in range(len(candidate_qpos)):
        key = np.concatenate((candidate_geom[index, 0, :3], candidate_geom[index, 0, 3:])).tobytes()
        previous = lookup.setdefault(key, index)
        if previous != index and not np.array_equal(candidate_qpos[previous, :7], candidate_qpos[index, :7]):
            raise RuntimeError("ambiguous base pose for identical frozen base geometry")
    row_positions = np.asarray(robot_positions, np.float64)
    row_quaternions = np.asarray(robot_quaternions, np.float64)
    row_pose = np.ascontiguousarray(
        np.concatenate((row_positions[:, 0], row_quaternions[:, 0]), axis=1)
    )
    row_keys = row_pose.view(np.dtype((np.void, row_pose.dtype.itemsize * 7))).reshape(-1)
    unique_keys, inverse = np.unique(row_keys, return_inverse=True)
    unique_indices = np.empty(len(unique_keys), np.int16)
    for index, key in enumerate(unique_keys):
        key_bytes = key.tobytes()
        if key_bytes not in lookup:
            raise RuntimeError("dense ray row does not match a frozen acquisition pose")
        unique_indices[index] = lookup[key_bytes]
    indices = unique_indices[inverse]
    return candidate_qpos[indices, :3], candidate_qpos[indices, 3:7]


def _origin_dense_evidence(
    *,
    mount_id: str,
    spherical: bool,
    mode_id: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
    targets: dict[str, np.ndarray],
    installed_mount_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Run the predecessor analytic witness reducer at one frozen origin."""

    translation, rotation = _mount_transform(mount_id)
    old_sensor = BASE.sensor_pose_for_condition
    old_occlusion = BASE.self_occlusion_geometry_for_condition

    def sensor_pose(_condition: str, base_position: np.ndarray, base_quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return BASE.mounted_sensor_poses(
            base_position,
            base_quaternion,
            translation_m=translation,
            rotation_rpy_rad=rotation,
        )

    def occlusion(
        _condition: str,
        specs: list[Any],
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> tuple[list[Any], np.ndarray, np.ndarray]:
        base_position, base_quaternion = _base_pose_rows_for_robot_pose_rows(
            positions,
            quaternions,
            boundary_qpos=boundary_qpos,
            boundary_geom_transform=boundary_geom_transform,
            qpos=qpos,
            geom_transform=geom_transform,
        )
        if mount_id == "HEAD_STOCK":
            filtered_specs, filtered_positions, filtered_quaternions = old_occlusion(
                "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
                specs,
                positions,
                quaternions,
            )
        else:
            filtered_specs = list(specs)
            filtered_positions = np.asarray(positions)
            filtered_quaternions = np.asarray(quaternions)
        return _append_other_sensor_housings(
            emitting_mount_id=mount_id,
            installed_mount_ids=installed_mount_ids,
            specs=filtered_specs,
            positions=filtered_positions,
            quaternions=filtered_quaternions,
            base_positions=base_position,
            base_quaternions=base_quaternion,
        )

    BASE.sensor_pose_for_condition = sensor_pose
    BASE.self_occlusion_geometry_for_condition = occlusion
    try:
        return BASE.dense_per_link_evidence(
            condition_id=(
                "DENSE_BODY_CENTRIC_SINGLE_ORIGIN"
                if spherical
                else "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT"
            ),
            mode_id=mode_id,
            boundary_qpos=boundary_qpos,
            boundary_geom_transform=boundary_geom_transform,
            qpos=qpos,
            geom_transform=geom_transform,
            environment_boxes=environment_boxes,
            robot_specs=robot_specs,
            targets=targets,
        )
    finally:
        BASE.sensor_pose_for_condition = old_sensor
        BASE.self_occlusion_geometry_for_condition = old_occlusion


def _render_realistic_origin(
    *,
    mount_id: str,
    mode_id: str,
    transition_uid: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
    installed_mount_ids: tuple[str, ...],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_v1 as geometry

    contract_digest_sha256 = _contract_content_digest()
    phase = CONTRACT.derive_multi_origin_scan_phases(
        contract_digest_sha256=contract_digest_sha256,
        transition_uid=transition_uid,
        mount_id=mount_id,
    )
    horizontal_phase = float(phase["horizontal_phase_cycles"])
    vertical_phase = float(phase["vertical_phase_cycles"])
    causal = mode_id == "PLANNING_TIME_CAUSAL_CLOUD"
    if causal:
        horizontal_phase -= 5.55 * 0.1
        vertical_phase -= 216.0 * 0.1
    pattern = geometry.generate_l2_scan_pattern(
        horizontal_phase_cycles=horizontal_phase,
        vertical_phase_cycles=vertical_phase,
    )
    if pattern.ray_count != 6400:
        raise RuntimeError("replicated-L2 scan does not contain 6,400 rays")
    if causal:
        query = np.zeros(pattern.ray_count, np.float64)
        point_time = np.asarray(pattern.timestamps_s, np.float64) - 0.1
        base_position = np.repeat(boundary_qpos[None, :3], pattern.ray_count, axis=0)
        base_quaternion = np.repeat(boundary_qpos[None, 3:7], pattern.ray_count, axis=0)
        robot_position = np.repeat(
            boundary_geom_transform[None, :, :3], pattern.ray_count, axis=0
        )
        robot_quaternion = np.repeat(
            boundary_geom_transform[None, :, 3:], pattern.ray_count, axis=0
        )
    else:
        query = np.asarray(pattern.timestamps_s, np.float64)
        point_time = query.copy()
        base_position, base_quaternion, robot_position, robot_quaternion = (
            BASE.interpolate_transition_poses(
                boundary_qpos,
                boundary_geom_transform,
                qpos,
                geom_transform,
                query,
            )
        )
    translation, rotation = _mount_transform(mount_id)
    sensor_position, sensor_quaternion = BASE.mounted_sensor_poses(
        base_position,
        base_quaternion,
        translation_m=translation,
        rotation_rpy_rad=rotation,
    )
    directions = BASE.rotate_vectors(
        sensor_quaternion,
        np.asarray(pattern.directions_sensor_fru, np.float64),
    )
    if mount_id == "HEAD_STOCK":
        occluder_specs, occluder_position, occluder_quaternion = (
            BASE.self_occlusion_geometry_for_condition(
                "REALISTIC_PLATFORM_SCAN",
                robot_specs,
                robot_position,
                robot_quaternion,
            )
        )
    else:
        occluder_specs = list(robot_specs)
        occluder_position = robot_position
        occluder_quaternion = robot_quaternion
    occluder_specs, occluder_position, occluder_quaternion = (
        _append_other_sensor_housings(
            emitting_mount_id=mount_id,
            installed_mount_ids=installed_mount_ids,
            specs=occluder_specs,
            positions=occluder_position,
            quaternions=occluder_quaternion,
            base_positions=base_position,
            base_quaternions=base_quaternion,
        )
    )
    cast = geometry.raycast_moving_scene(
        sensor_position,
        directions,
        0.05,
        30.0,
        environment_boxes,
        occluder_specs,
        occluder_position,
        occluder_quaternion,
        ground=True,
    )
    environment = cast.valid_return & (cast.hit_kind == "ENVIRONMENT")
    self_return = np.asarray(cast.self_return, bool)
    return {
        "points": np.asarray(cast.point_world_xyz_m[environment], np.float64),
        "object_index": np.asarray(cast.object_index[environment], np.int16),
        "ray_index": np.flatnonzero(environment).astype(np.int32),
        "point_time_s": np.asarray(point_time[environment], np.float64),
        "point_range_m": np.asarray(cast.distance_m[environment], np.float64),
        "self_ray_index": np.flatnonzero(self_return).astype(np.int32),
        "self_geom_index": np.asarray(cast.geom_index[self_return], np.int16),
        "self_link_index": np.asarray(cast.link_index[self_return], np.int16),
        "self_distance_m": np.asarray(cast.raw_distance_m[self_return], np.float64),
        "pattern": pattern.to_serializable(include_rays=False),
        "phase": phase,
        "ray_count": pattern.ray_count,
        "environment_return_count": int(environment.sum()),
        "self_return_count": int(self_return.sum()),
        "near_blind_count": int(cast.near_blind.sum()),
        "ground_return_count": int(
            (cast.valid_return & (cast.hit_kind == "GROUND")).sum()
        ),
        "no_hit_count": int(cast.no_hit.sum()),
        "transition_identity": transition_uid,
        "mount_id": mount_id,
    }


def _origin_realistic_evidence(
    *,
    mount_id: str,
    mode_id: str,
    transition_uid: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
    targets: dict[str, np.ndarray],
    installed_mount_ids: tuple[str, ...],
    matched_dense_l2_evidence: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray], int]:
    cloud = _render_realistic_origin(
        mount_id=mount_id,
        mode_id=mode_id,
        transition_uid=transition_uid,
        boundary_qpos=boundary_qpos,
        boundary_geom_transform=boundary_geom_transform,
        qpos=qpos,
        geom_transform=geom_transform,
        environment_boxes=environment_boxes,
        robot_specs=robot_specs,
        installed_mount_ids=installed_mount_ids,
    )
    values = BASE.sparse_per_link_evidence(
        cloud=cloud,
        robot_specs=robot_specs,
        geom_transform=geom_transform,
        qpos=qpos,
        targets=targets,
        evaluation_time_s=np.arange(1, STEPS + 1, dtype=np.float64) * 0.002,
    )
    values["responsible_geom_index"] = np.asarray(targets["geom_index"], np.int16)
    dense = _copy_evidence(matched_dense_l2_evidence)
    _enrich_sparse_with_dense_attribution(values, dense)
    inherited = _inherit_supported_evidence(
        dense,
        values,
        source_is_finite_scan=True,
    )
    if np.any(np.asarray(values["support"], bool) & ~np.asarray(dense["support"], bool)):
        raise RuntimeError("realistic per-origin support is not contained by dense L2 FOV")
    return values, cloud, dense, inherited


def _copy_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in evidence.items()
    }


def _enrich_sparse_with_dense_attribution(
    sparse: dict[str, np.ndarray], dense: dict[str, np.ndarray]
) -> None:
    """Attach target-visibility diagnostics without replacing sparse evidence."""

    for field in (
        "nominal_fov",
        "horizontal_fov",
        "vertical_fov",
        "direct_visibility",
        "self_occluded",
        "environment_occluded",
        "near_blind",
        "self_occluder_geom_index",
        "self_occluder_link_index",
        "target_azimuth_deg",
        "target_elevation_deg",
        "target_range_m",
        "acquisition_times_s",
        "acquisition_support",
        "acquisition_nominal_fov",
        "acquisition_self_occluded",
        "acquisition_self_geom_index",
        "acquisition_self_link_index",
        "acquisition_environment_occluded",
    ):
        if field in dense:
            sparse[field] = np.array(dense[field], copy=True)


def _inherit_supported_evidence(
    target: dict[str, np.ndarray],
    source: dict[str, np.ndarray],
    *,
    source_is_finite_scan: bool,
) -> int:
    """Union a lower representation into its prospectively stronger bound."""

    source_support = np.asarray(source["support"], bool)
    target_support = np.asarray(target["support"], bool)
    inherited = source_support & ~target_support
    source_clearance = np.asarray(source["clearance_m"], np.float64)
    target_clearance = np.asarray(target["clearance_m"], np.float64)
    replace = source_support & (
        ~target_support | (source_clearance < target_clearance)
    )
    if replace.any():
        selectable = tuple(BASE.FLOAT_EVIDENCE_FIELDS) + tuple(
            BASE.INT16_EVIDENCE_FIELDS
        ) + (
            "nearest_ray_index",
            "support_nearest_ray_index",
            "responsible_geom_index",
            "obstacle_direction_body_rad",
        )
        for field in dict.fromkeys(selectable):
            if field in target and field in source:
                target[field][replace] = np.asarray(source[field])[replace]
        target.setdefault(
            "finite_scan_support_inherited",
            np.zeros_like(target_support, bool),
        )
        source_finite = np.asarray(
            source.get(
                "finite_scan_support_inherited",
                np.full_like(source_support, source_is_finite_scan, bool),
            ),
            bool,
        )
        if source_is_finite_scan:
            source_finite = np.ones_like(source_support, bool)
        target["finite_scan_support_inherited"][replace] |= source_finite[replace]
    # Support and event-causal support are set unions even when the target
    # already had an analytic observation.  Clearance/provenance above is a
    # minimum union: an overlapping finite observation replaces a larger
    # analytic clearance so the stronger representation remains at least as
    # conservative as its subset.
    target["support"] |= source_support
    target["event_time_support"] |= np.asarray(
        source["event_time_support"], bool
    )
    for field in (
        "nominal_fov",
        "horizontal_fov",
        "vertical_fov",
        "direct_visibility",
    ):
        if field in target:
            target[field] |= np.asarray(
                source.get(field, source_support), bool
            )
    for field in (
        "self_occluded",
        "environment_occluded",
        "near_blind",
    ):
        if field in target:
            target[field][source_support] = False
    for field in ("self_occluder_geom_index", "self_occluder_link_index"):
        if field in target:
            target[field][source_support] = -1
    if np.any(source_support & ~np.asarray(target["support"], bool)):
        raise RuntimeError("support hierarchy inheritance failed")
    return int(inherited.sum())


def _union_dense_uid_evidence(
    accumulated: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> None:
    """Union UID-specific finite inheritance while retaining minimum clearance."""

    accumulated_support = np.asarray(accumulated["support"], bool)
    candidate_support = np.asarray(candidate["support"], bool)
    accumulated_clearance = np.asarray(accumulated["clearance_m"], np.float64)
    candidate_clearance = np.asarray(candidate["clearance_m"], np.float64)
    replace = candidate_support & (
        ~accumulated_support | (candidate_clearance < accumulated_clearance)
    )
    selectable = tuple(BASE.FLOAT_EVIDENCE_FIELDS) + tuple(
        BASE.INT16_EVIDENCE_FIELDS
    ) + (
        "nearest_ray_index",
        "support_nearest_ray_index",
        "responsible_geom_index",
        "obstacle_direction_body_rad",
    )
    for field in dict.fromkeys(selectable):
        if field in accumulated and field in candidate:
            accumulated[field][replace] = np.asarray(candidate[field])[replace]
    accumulated["support"] |= candidate_support
    accumulated["event_time_support"] |= np.asarray(
        candidate["event_time_support"], bool
    )
    for field in (
        "nominal_fov",
        "horizontal_fov",
        "vertical_fov",
        "direct_visibility",
    ):
        if field in accumulated and field in candidate:
            accumulated[field] |= np.asarray(candidate[field], bool)
    accumulated["self_occluded"][accumulated["support"]] = False
    accumulated["environment_occluded"][accumulated["support"]] = False
    accumulated["near_blind"][accumulated["support"]] = False
    accumulated.setdefault(
        "finite_scan_support_inherited",
        np.zeros_like(accumulated_support, bool),
    )
    accumulated["finite_scan_support_inherited"] |= np.asarray(
        candidate.get(
            "finite_scan_support_inherited", np.zeros_like(candidate_support, bool)
        ),
        bool,
    )


def _realistic_layout_evidence(
    *,
    condition_id: str,
    layout_id: str,
    mounts: tuple[str, ...],
    mode_id: str,
    transition_index: int,
    action_representative_transition_index: int,
    geometry_representative_transition_index: int,
    transition_uid: str,
    boundary_qpos: np.ndarray,
    boundary_geom_transform: np.ndarray,
    qpos: np.ndarray,
    geom_transform: np.ndarray,
    environment_boxes: list[Any],
    robot_specs: list[Any],
    targets: dict[str, np.ndarray],
    matched_dense_l2_by_mount: dict[str, dict[str, np.ndarray]],
) -> tuple[
    dict[str, np.ndarray],
    list[tuple[str, dict[str, np.ndarray]]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, dict[str, np.ndarray]]],
]:
    """Materialize one transition-identity-bound replicated-L2 layout."""

    per_origin: list[tuple[str, dict[str, np.ndarray]]] = []
    per_origin_dense_l2: list[tuple[str, dict[str, np.ndarray]]] = []
    clouds: dict[str, dict[str, Any]] = {}
    scan_rows: list[dict[str, Any]] = []
    for mount_id in mounts:
        if mount_id not in matched_dense_l2_by_mount:
            raise RuntimeError(f"matched dense L2 evidence missing for {mount_id}")
        values, cloud, matched_dense, inherited = _origin_realistic_evidence(
            mount_id=mount_id,
            mode_id=mode_id,
            transition_uid=transition_uid,
            boundary_qpos=boundary_qpos,
            boundary_geom_transform=boundary_geom_transform,
            qpos=qpos,
            geom_transform=geom_transform,
            environment_boxes=environment_boxes,
            robot_specs=robot_specs,
            targets=targets,
            installed_mount_ids=mounts,
            matched_dense_l2_evidence=matched_dense_l2_by_mount[mount_id],
        )
        clouds[mount_id] = cloud
        per_origin.append((mount_id, values))
        per_origin_dense_l2.append((mount_id, matched_dense))
        scan_rows.append(
            {
                "transition_index": transition_index,
                "action_representative_transition_index": (
                    action_representative_transition_index
                ),
                "geometry_representative_transition_index": (
                    geometry_representative_transition_index
                ),
                "transition_uid": transition_uid,
                "condition_id": condition_id,
                "evidence_mode": mode_id,
                "layout_id": layout_id,
                "mount_id": mount_id,
                "ray_count": int(cloud["ray_count"]),
                "environment_return_count": int(cloud["environment_return_count"]),
                "self_return_count": int(cloud["self_return_count"]),
                "near_blind_count": int(cloud["near_blind_count"]),
                "ground_return_count": int(cloud["ground_return_count"]),
                "phase": cloud["phase"],
                "render_cache_reused": False,
                "dense_l2_inherited_support_witnesses": inherited,
                "dense_spherical_inherited_support_witnesses": 0,
            }
        )
    combined = _merge_origin_evidence(per_origin)
    dense_combined = _merge_origin_evidence(per_origin_dense_l2)
    if np.any(
        np.asarray(combined["support"], bool)
        & ~np.asarray(dense_combined["support"], bool)
    ):
        raise RuntimeError("realistic fused support is not contained by dense L2 FOV")
    return combined, per_origin, clouds, scan_rows, per_origin_dense_l2


def _merge_origin_evidence(
    per_origin: Sequence[tuple[str, dict[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    if not per_origin:
        raise ValueError("a multi-origin layout must contain at least one origin")
    origin_ids = [row[0] for row in per_origin]
    values = [row[1] for row in per_origin]
    shape = (STEPS, LINKS)
    support_stack = np.stack([np.asarray(row["support"], bool) for row in values])
    clearance_stack = np.stack(
        [np.asarray(row["clearance_m"], np.float64) for row in values]
    )
    effective = np.where(support_stack, clearance_stack, np.inf)
    chosen = np.argmin(effective, axis=0)
    supported = np.any(support_stack, axis=0)
    chosen = np.where(supported, chosen, 0)
    row_index, link_index = np.indices(shape)

    def selected(field: str, dtype: Any | None = None) -> np.ndarray:
        stack = np.stack([np.asarray(row[field]) for row in values])
        output = stack[chosen, row_index, link_index]
        return output.astype(dtype) if dtype is not None else output.copy()

    combined: dict[str, np.ndarray] = {}
    for field in BASE.FLOAT_EVIDENCE_FIELDS:
        if all(field in row for row in values):
            combined[field] = selected(field, np.float64)
    for field in BASE.INT16_EVIDENCE_FIELDS:
        if all(field in row for row in values):
            combined[field] = selected(field, np.int16)
    for field in ("nearest_ray_index", "support_nearest_ray_index"):
        if all(field in row for row in values):
            combined[field] = selected(field, np.int32)
    if all("responsible_geom_index" in row for row in values):
        combined["responsible_geom_index"] = selected(
            "responsible_geom_index", np.int16
        )
    combined["clearance_m"] = np.min(effective, axis=0)
    combined["support"] = supported
    combined["event_time_support"] = np.any(
        np.stack([np.asarray(row["event_time_support"], bool) for row in values]),
        axis=0,
    )
    combined["point_support_count"] = np.sum(
        np.stack([np.asarray(row["point_support_count"], np.int32) for row in values]),
        axis=0,
    ).clip(0, np.iinfo(np.int16).max).astype(np.int16)
    for field in ("nominal_fov", "horizontal_fov", "vertical_fov", "direct_visibility"):
        combined[field] = np.any(
            np.stack([np.asarray(row[field], bool) for row in values]), axis=0
        )
    nominal_stack = np.stack([np.asarray(row["nominal_fov"], bool) for row in values])
    self_stack = np.stack([np.asarray(row["self_occluded"], bool) for row in values])
    env_stack = np.stack([np.asarray(row["environment_occluded"], bool) for row in values])
    combined["self_occluded"] = (
        ~supported
        & np.any(nominal_stack, axis=0)
        & np.all(~nominal_stack | self_stack, axis=0)
    )
    combined["environment_occluded"] = (
        ~supported
        & ~combined["self_occluded"]
        & np.any(env_stack, axis=0)
    )
    combined["near_blind"] = (
        ~supported
        & np.any(
            np.stack([np.asarray(row["near_blind"], bool) for row in values]),
            axis=0,
        )
    )
    combined["finite_scan_support_inherited"] = np.any(
        np.stack(
            [
                np.asarray(
                    row.get("finite_scan_support_inherited", np.zeros(shape, bool)),
                    bool,
                )
                for row in values
            ]
        ),
        axis=0,
    )
    bitmask = np.zeros(shape, np.uint8)
    for index, row in enumerate(support_stack):
        bitmask |= row.astype(np.uint8) << index
    combined["supporting_origin_bitmask"] = bitmask
    combined["supporting_origin_count"] = support_stack.sum(axis=0).astype(np.uint8)
    combined["responsible_origin_index"] = np.where(supported, chosen, -1).astype(np.int8)
    combined["origin_ids"] = np.asarray(origin_ids)
    combined["per_origin_support"] = support_stack
    combined["per_origin_event_time_support"] = np.stack(
        [np.asarray(row["event_time_support"], bool) for row in values]
    )
    combined["per_origin_nominal_fov"] = nominal_stack
    combined["per_origin_direct_visibility"] = np.stack(
        [np.asarray(row["direct_visibility"], bool) for row in values]
    )
    combined["per_origin_self_occluded"] = self_stack
    combined["per_origin_finite_scan_support_inherited"] = np.stack(
        [
            np.asarray(
                row.get("finite_scan_support_inherited", np.zeros(shape, bool)),
                bool,
            )
            for row in values
        ]
    )
    combined["per_origin_point_support_count"] = np.stack(
        [np.asarray(row["point_support_count"], np.int16) for row in values]
    )
    combined["per_origin_support_acquisition_index"] = np.stack(
        [np.asarray(row["support_acquisition_index"], np.int16) for row in values]
    )
    combined["per_origin_nearest_ray_index"] = np.stack(
        [np.asarray(row["support_nearest_ray_index"], np.int32) for row in values]
    )
    combined["per_origin_point_age_s"] = np.stack(
        [np.asarray(row["support_point_age_s"], np.float64) for row in values]
    )
    combined["per_origin_clearance_m"] = clearance_stack
    combined["per_origin_object_index"] = np.stack(
        [np.asarray(row["support_object_index"], np.int16) for row in values]
    )
    return combined


def _region_link_masks() -> dict[str, np.ndarray]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus

    names = tuple(corpus.PROTECTED_LINK_NAMES)
    return {
        "TRUNK": np.asarray([name == "base" for name in names], bool),
        "FRONT_LIMBS": np.asarray([name.startswith(("FL_", "FR_")) for name in names], bool),
        "REAR_LIMBS": np.asarray([name.startswith(("RL_", "RR_")) for name in names], bool),
        "HIPS_AND_THIGHS": np.asarray(
            [name.endswith(("_hip", "_thigh")) for name in names], bool
        ),
        "CALVES": np.asarray([name.endswith("_calf") for name in names], bool),
    }


LABEL_FREE_LAYOUT_REUSE_FIELDS = (
    "qpos",
    "link_transform",
    "geom_transform",
)


class _LabelFreeLayoutStateView:
    """Outcome-free view exposed to the layout-selection objective."""

    __slots__ = (
        "state_id",
        "family",
        "role",
        "scene_boxes",
        "geometry_contract",
        "boundaries",
        "transition_rows",
        "action_copy_map",
        "arrays",
        "transition_count",
    )

    def __init__(
        self,
        *,
        state_id: str,
        family: str,
        role: str,
        scene_boxes: Any,
        geometry_contract: Sequence[dict[str, Any]],
        boundaries: Any,
        transition_rows: Sequence[dict[str, Any]],
        action_copy_map: Any,
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.state_id = state_id
        self.family = family
        self.role = role
        self.scene_boxes = scene_boxes
        self.geometry_contract = tuple(geometry_contract)
        self.boundaries = boundaries
        self.transition_rows = tuple(transition_rows)
        self.action_copy_map = action_copy_map
        self.arrays = arrays
        self.transition_count = len(self.transition_rows)


def _load_label_free_layout_state(
    context: Any, state_id: str, corpus_adapter: Any
) -> _LabelFreeLayoutStateView:
    """Load only geometry/action fields exposed to layout scoring.

    The full corpus has already passed the separate preexecution custody
    audit.  This loader deliberately avoids ``load_state`` and
    ``transition_identity_rows`` because those convenience APIs materialize
    contact, viability, and route outcomes that are forbidden inputs to the
    layout-selection objective.
    """

    geometry_receipt = context.geometry_record(state_id)
    source_receipt = context.corpus_record(state_id)
    shard_path = corpus_adapter._resolve_frozen_path(  # noqa: SLF001 - frozen adapter
        context.root, geometry_receipt["shard_path"]
    )
    if sha256_file(shard_path) != geometry_receipt["shard_sha256"]:
        raise RuntimeError(f"{state_id}: label-free geometry shard SHA drift")
    with np.load(shard_path, allow_pickle=False) as archive:
        missing = [field for field in LABEL_FREE_LAYOUT_REUSE_FIELDS if field not in archive]
        if missing:
            raise RuntimeError(f"{state_id}: label-free shard fields missing: {missing}")
        arrays = {
            field: np.asarray(archive[field]) for field in LABEL_FREE_LAYOUT_REUSE_FIELDS
        }
    transitions = int(geometry_receipt["transitions"])
    expected_shapes = {
        "qpos": (transitions, STEPS, 19),
        "link_transform": (transitions, STEPS, LINKS, 7),
        "geom_transform": (transitions, STEPS, 27, 7),
    }
    for field, expected_shape in expected_shapes.items():
        if arrays[field].shape != expected_shape:
            raise RuntimeError(
                f"{state_id}: label-free {field} shape {arrays[field].shape} != {expected_shape}"
            )
        arrays[field].setflags(write=False)

    transition_rows = tuple(
        {
            "transition_index": int(row["transition_index"]),
            "level": str(row["level"]),
            "current_action_index": int(row["current_action_index"]),
            "action_index": int(row["action_index"]),
        }
        for row in geometry_receipt["transition_rows"]
    )

    def stripped_action(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "action_index": int(row["action_index"]),
            "controller": str(row["controller"]),
            "applied_action": [float(value) for value in row["applied_action"]],
        }

    stripped_source = {
        "state_id": str(source_receipt["state_id"]),
        "current_rows": [
            stripped_action(dict(row)) for row in source_receipt["current_rows"]
        ],
        "successor_rows": [
            {
                "current_action_index": int(row["current_action_index"]),
                "next_actions": [
                    stripped_action(dict(candidate))
                    for candidate in row["next_actions"]
                ],
            }
            for row in source_receipt["successor_rows"]
        ],
    }
    stripped_geometry = {
        "transition_rows": list(transition_rows),
    }
    action_copy_map = corpus_adapter.build_applied_action_copy_map(
        stripped_source, stripped_geometry
    )
    return _LabelFreeLayoutStateView(
        state_id=str(state_id),
        family=str(geometry_receipt["family"]),
        role=str(context.role_for_state(state_id)),
        scene_boxes=context.scene_obbs(state_id),
        geometry_contract=context.collision_shape_contract(state_id),
        boundaries=context.load_boundary_transforms(state_id),
        transition_rows=transition_rows,
        action_copy_map=action_copy_map,
        arrays=arrays,
    )


def _build_label_free_layout_reuse_groups(
    shard: Any,
    action_copy_map: Any,
    boundary_snapshot_digests: Sequence[str],
) -> dict[int, tuple[int, ...]]:
    """Group byte-identical dense-visibility traces without outcome fields.

    The deployable-action map is used only as a conservative partition.  A
    layout-selection render may be reused inside that partition only when the
    boundary snapshot and all three geometric trajectory arrays are exactly
    equal.  Contact labels, contact arrays, safe-action counts, and route
    outcomes are deliberately neither required nor inspected here.
    """

    transition_count = int(shard.transition_count)
    digests = tuple(str(value) for value in boundary_snapshot_digests)
    if len(digests) != transition_count:
        raise RuntimeError("label-free layout boundary-digest cardinality drift")
    arrays = shard.arrays
    missing = [field for field in LABEL_FREE_LAYOUT_REUSE_FIELDS if field not in arrays]
    if missing:
        raise RuntimeError(f"label-free layout geometry fields missing: {missing}")

    groups: dict[int, list[int]] = {}
    assigned: set[int] = set()
    for _action_representative, action_copies in sorted(
        action_copy_map.copies_by_representative.items()
    ):
        geometric_representatives: list[int] = []
        for transition in action_copies:
            transition = int(transition)
            selected: int | None = None
            for representative in geometric_representatives:
                if digests[representative] != digests[transition]:
                    continue
                if all(
                    np.array_equal(
                        arrays[field][representative], arrays[field][transition]
                    )
                    for field in LABEL_FREE_LAYOUT_REUSE_FIELDS
                ):
                    selected = representative
                    break
            if selected is None:
                selected = transition
                geometric_representatives.append(selected)
            groups.setdefault(selected, []).append(transition)
            assigned.add(transition)
    if assigned != set(range(transition_count)):
        raise RuntimeError("label-free layout geometry reuse map is incomplete")
    return {key: tuple(value) for key, value in sorted(groups.items())}


def _layout_selection_state_worker(state_id: str) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    state = _load_label_free_layout_state(context, state_id, corpus_adapter)
    if state.role != "training":
        raise RuntimeError("layout-selection worker received a non-training state")
    boxes, specs = BASE.runtime_geometry_contract(state)
    boundary_digests = tuple(
        BASE.transition_boundary(state, dict(row))[2] for row in state.transition_rows
    )
    geometry_groups = _build_label_free_layout_reuse_groups(
        state, state.action_copy_map, boundary_digests
    )
    layout_ids = tuple(CONTRACT.PAIR_LAYOUT_IDS) + tuple(CONTRACT.THREE_LAYOUT_IDS)
    region_masks = _region_link_masks()
    protected_link_names = tuple(corpus_adapter.PROTECTED_LINK_NAMES)
    accumulators = {
        layout_id: {
            "transition_support": np.zeros(state.transition_count, np.float32),
            "transition_supported_count": np.zeros(state.transition_count, np.int16),
            "transition_total_count": np.zeros(state.transition_count, np.int16),
            "transition_nominal_count": np.zeros(state.transition_count, np.int16),
            "transition_self_occluded_count": np.zeros(
                state.transition_count, np.int16
            ),
            "transition_region_supported_count": {
                key: np.zeros(state.transition_count, np.int16)
                for key in region_masks
            },
            "transition_region_total_count": {
                key: np.zeros(state.transition_count, np.int16)
                for key in region_masks
            },
            "transition_link_supported_count": np.zeros(
                (state.transition_count, LINKS), np.int16
            ),
            "transition_link_total_count": np.zeros(
                (state.transition_count, LINKS), np.int16
            ),
            "link_supported": np.zeros(LINKS, np.int64),
            "link_total": np.zeros(LINKS, np.int64),
            "region_supported": {key: 0 for key in region_masks},
            "region_total": {key: 0 for key in region_masks},
            "supported": 0,
            "total": 0,
            "nominal": 0,
            "self_occluded": 0,
        }
        for layout_id in layout_ids
    }
    for representative, copies in geometry_groups.items():
        row = dict(state.transition_rows[representative])
        boundary_qpos, boundary_geom, _ = BASE.transition_boundary(state, row)
        qpos = np.asarray(state.arrays["qpos"][representative], np.float64)
        geom = np.asarray(state.arrays["geom_transform"][representative], np.float64)
        targets = BASE.closest_scene_targets(
            geom[:, :, :3], geom[:, :, 3:], specs, boxes
        )
        per_layout_mount: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for layout_id in layout_ids:
            mounts = _layout_mounts(layout_id)
            for mount_id in mounts:
                per_layout_mount[(layout_id, mount_id)] = _origin_dense_evidence(
                    mount_id=mount_id,
                    spherical=False,
                    mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
                    boundary_qpos=boundary_qpos,
                    boundary_geom_transform=boundary_geom,
                    qpos=qpos,
                    geom_transform=geom,
                    environment_boxes=boxes,
                    robot_specs=specs,
                    targets=targets,
                    installed_mount_ids=mounts,
                )
            support = np.any(
                np.stack(
                    [
                        np.asarray(per_layout_mount[(layout_id, mount)]["support"], bool)
                        for mount in mounts
                    ]
                ),
                axis=0,
            )
            nominal = np.any(
                np.stack(
                    [
                        np.asarray(
                            per_layout_mount[(layout_id, mount)]["nominal_fov"], bool
                        )
                        for mount in mounts
                    ]
                ),
                axis=0,
            )
            self_occluded = (
                ~support
                & nominal
                & np.all(
                    np.stack(
                        [
                            ~np.asarray(
                                per_layout_mount[(layout_id, mount)]["nominal_fov"],
                                bool,
                            )
                            | np.asarray(
                                per_layout_mount[(layout_id, mount)]["self_occluded"],
                                bool,
                            )
                            for mount in mounts
                        ]
                    ),
                    axis=0,
                )
            )
            acc = accumulators[layout_id]
            fraction = float(support.mean())
            for copy in copies:
                acc["transition_support"][copy] = fraction
                acc["transition_supported_count"][copy] = int(support.sum())
                acc["transition_total_count"][copy] = int(support.size)
                acc["transition_nominal_count"][copy] = int(nominal.sum())
                acc["transition_self_occluded_count"][copy] = int(
                    self_occluded.sum()
                )
                for region, mask in region_masks.items():
                    acc["transition_region_supported_count"][region][copy] = int(
                        support[:, mask].sum()
                    )
                    acc["transition_region_total_count"][region][copy] = int(
                        support[:, mask].size
                    )
                acc["transition_link_supported_count"][copy] = support.sum(
                    axis=0
                )
                acc["transition_link_total_count"][copy] = STEPS
            copy_count = len(copies)
            acc["supported"] += int(support.sum()) * copy_count
            acc["total"] += int(support.size) * copy_count
            acc["nominal"] += int(nominal.sum()) * copy_count
            acc["self_occluded"] += int(self_occluded.sum()) * copy_count
            acc["link_supported"] += support.sum(axis=0) * copy_count
            acc["link_total"] += STEPS * copy_count
            for region, mask in region_masks.items():
                acc["region_supported"][region] += int(support[:, mask].sum()) * copy_count
                acc["region_total"][region] += int(support[:, mask].size) * copy_count
    return {
        "state_id": state_id,
        "family": state.family,
        "label_free_geometry_reuse_fields": list(LABEL_FREE_LAYOUT_REUSE_FIELDS),
        "label_free_geometry_representatives": len(geometry_groups),
        "transition_rows": [dict(row) for row in state.transition_rows],
        "protected_link_names": list(protected_link_names),
        "layouts": {
            layout_id: {
                "transition_support": accumulator["transition_support"].tolist(),
                "transition_supported_count": accumulator[
                    "transition_supported_count"
                ].tolist(),
                "transition_total_count": accumulator[
                    "transition_total_count"
                ].tolist(),
                "transition_nominal_count": accumulator[
                    "transition_nominal_count"
                ].tolist(),
                "transition_self_occluded_count": accumulator[
                    "transition_self_occluded_count"
                ].tolist(),
                "transition_region_supported_count": {
                    region: values.tolist()
                    for region, values in accumulator[
                        "transition_region_supported_count"
                    ].items()
                },
                "transition_region_total_count": {
                    region: values.tolist()
                    for region, values in accumulator[
                        "transition_region_total_count"
                    ].items()
                },
                "transition_link_supported_count": accumulator[
                    "transition_link_supported_count"
                ].tolist(),
                "transition_link_total_count": accumulator[
                    "transition_link_total_count"
                ].tolist(),
                "link_supported": accumulator["link_supported"].tolist(),
                "link_total": accumulator["link_total"].tolist(),
                "region_supported": accumulator["region_supported"],
                "region_total": accumulator["region_total"],
                "supported": accumulator["supported"],
                "total": accumulator["total"],
                "nominal": accumulator["nominal"],
                "self_occluded": accumulator["self_occluded"],
            }
            for layout_id, accumulator in accumulators.items()
        },
    }


def _validate_training_layout_evidence_binding(binding: dict[str, Any]) -> None:
    core = dict(binding)
    declared = core.pop("content_digest", None)
    path = Path(binding["path"])
    if (
        declared != content_digest(core)
        or binding.get("schema")
        != "minimum_multi_origin_training_layout_evidence_v1"
        or tuple(binding.get("layout_ids", ()))
        != tuple(CONTRACT.PAIR_LAYOUT_IDS) + tuple(CONTRACT.THREE_LAYOUT_IDS)
        or tuple(binding.get("body_region_ids", ()))
        != tuple(CONTRACT.BODY_REGION_IDS)
        or binding.get("role") != "development_training"
        or binding.get("outcome_fields_used_by_layout_objective") != []
        or int(binding.get("rows", -1)) <= 0
        or not path.is_file()
        or path.stat().st_size != int(binding.get("bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise RuntimeError("training layout row-evidence binding drift")


def _validate_training_layout_evidence_rows(
    binding: dict[str, Any], selection: dict[str, Any]
) -> None:
    """Validate every label-free row and regenerate all six candidate scores."""

    _validate_training_layout_evidence_binding(binding)
    schema = CONTRACT.build_output_schema()["files"]["training_layout_evidence"]
    required_row = set(schema["required_keys"])
    required_metric = set(schema["layout_metric_required_keys"])
    layout_ids = tuple(CONTRACT.PAIR_LAYOUT_IDS) + tuple(CONTRACT.THREE_LAYOUT_IDS)
    region_ids = tuple(schema["body_region_ids"])
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    link_names = tuple(corpus_adapter.PROTECTED_LINK_NAMES)
    aggregate: dict[str, dict[str, Any]] = {
        layout_id: {
            "transition_support": [],
            "supported": 0,
            "total": 0,
            "nominal": 0,
            "self_occluded": 0,
            "regions": {
                region: {"supported": 0, "total": 0} for region in region_ids
            },
            "links": {
                link: {"supported": 0, "total": 0} for link in link_names
            },
            "families": defaultdict(lambda: {"supported": 0, "total": 0}),
        }
        for layout_id in layout_ids
    }
    seen: set[tuple[str, int]] = set()
    states: set[str] = set()
    rows = 0
    with gzip.open(Path(binding["path"]), "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise RuntimeError("training layout JSONL row lacks LF")
            row = json.loads(line)
            if not required_row.issubset(row):
                raise RuntimeError(
                    f"training layout evidence row {line_number} key drift"
                )
            if (
                row.get("schema")
                != "minimum_multi_origin_training_layout_transition_evidence_v1"
                or row.get("role") != "development_training"
                or row.get("contact_label") is not None
                or row.get("safe_action_count") is not None
                or row.get("route_outcome") is not None
                or set(row.get("layout_metrics", {})) != set(layout_ids)
            ):
                raise RuntimeError("training layout evidence outcome/layout drift")
            identity = (str(row["state_id"]), int(row["transition_index"]))
            if identity in seen:
                raise RuntimeError("duplicate training layout transition evidence")
            seen.add(identity)
            states.add(identity[0])
            family = str(row["family"])
            for layout_id in layout_ids:
                metric = row["layout_metrics"][layout_id]
                if not required_metric.issubset(metric):
                    raise RuntimeError("training layout nested metric key drift")
                supported = int(metric["supported_witnesses"])
                total = int(metric["total_witnesses"])
                nominal = int(metric["nominally_eligible_witnesses"])
                self_occluded = int(
                    metric["all_origin_self_occluded_witnesses"]
                )
                if (
                    total != STEPS * LINKS
                    or not 0 <= supported <= total
                    or not 0 <= self_occluded <= nominal <= total
                    or not math.isclose(
                        float(metric["support_fraction"]),
                        supported / total,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        float(metric["unsupported_swept_volume_fraction"]),
                        1.0 - supported / total,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ):
                    raise RuntimeError("training layout witness-count drift")
                expected_self_fraction = (
                    None if nominal == 0 else self_occluded / nominal
                )
                if metric["self_occluded_fraction_of_nominal"] != expected_self_fraction:
                    raise RuntimeError("training layout self-occlusion fraction drift")
                if set(metric["body_region_counts"]) != set(region_ids):
                    raise RuntimeError("training layout body-region drift")
                if set(metric["body_region_support"]) != set(region_ids):
                    raise RuntimeError("training layout body-region support drift")
                if set(metric["protected_link_counts"]) != set(link_names):
                    raise RuntimeError("training layout protected-link drift")
                if set(metric["protected_link_support"]) != set(link_names):
                    raise RuntimeError("training layout protected-link support drift")
                target = aggregate[layout_id]
                target["transition_support"].append(supported / total)
                target["supported"] += supported
                target["total"] += total
                target["nominal"] += nominal
                target["self_occluded"] += self_occluded
                target["families"][family]["supported"] += supported
                target["families"][family]["total"] += total
                for region in region_ids:
                    counts = metric["body_region_counts"][region]
                    region_supported = int(counts["supported_witnesses"])
                    region_total = int(counts["total_witnesses"])
                    if not 0 <= region_supported <= region_total:
                        raise RuntimeError("training layout region-count drift")
                    if not math.isclose(
                        float(metric["body_region_support"][region]),
                        region_supported / region_total,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise RuntimeError("training layout region fraction drift")
                    target["regions"][region]["supported"] += region_supported
                    target["regions"][region]["total"] += region_total
                for link in link_names:
                    counts = metric["protected_link_counts"][link]
                    link_supported = int(counts["supported_witnesses"])
                    link_total = int(counts["total_witnesses"])
                    if link_total != STEPS or not 0 <= link_supported <= link_total:
                        raise RuntimeError("training layout link-count drift")
                    if not math.isclose(
                        float(metric["protected_link_support"][link]),
                        link_supported / link_total,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise RuntimeError("training layout link fraction drift")
                    target["links"][link]["supported"] += link_supported
                    target["links"][link]["total"] += link_total
            rows += 1
    expected_rows = {int(selection["scores"][layout]["transition_count"]) for layout in layout_ids}
    if (
        rows != int(binding["rows"])
        or expected_rows != {rows}
        or len(states) != EXPECTED["training_states"]
    ):
        raise RuntimeError("training layout evidence cardinality drift")

    def same(actual: float, expected: float) -> bool:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15)

    for layout_id in layout_ids:
        target = aggregate[layout_id]
        score = selection["scores"][layout_id]
        region_support = {
            region: values["supported"] / values["total"]
            for region, values in target["regions"].items()
        }
        link_support = {
            link: values["supported"] / values["total"]
            for link, values in target["links"].items()
        }
        family_support = {
            family: values["supported"] / values["total"]
            for family, values in sorted(target["families"].items())
        }
        expected_scalars = {
            "minimum_body_region_support": min(region_support.values()),
            "transition_support_p05": float(
                np.percentile(
                    np.asarray(target["transition_support"], np.float64),
                    5,
                    method="linear",
                )
            ),
            "rear_limb_support": region_support["REAR_LIMBS"],
            "calf_support": region_support["CALVES"],
            "overall_mean_support": target["supported"] / target["total"],
            "self_occluded_fraction": target["self_occluded"] / target["nominal"],
            "minimum_family_support": min(family_support.values()),
        }
        if (
            any(not same(score[key], value) for key, value in expected_scalars.items())
            or any(not same(score["region_support"][key], value) for key, value in region_support.items())
            or any(not same(score["per_link_support"][key], value) for key, value in link_support.items())
            or any(not same(score["per_family_support"][key], value) for key, value in family_support.items())
            or {
                key: {
                    "supported_witnesses": int(value["supported"]),
                    "total_witnesses": int(value["total"]),
                }
                for key, value in target["links"].items()
            }
            != score["per_link_counts"]
        ):
            raise RuntimeError("training layout aggregate score drift")

    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    selected_dual = metrics.select_training_layout(
        [selection["scores"][value] for value in CONTRACT.PAIR_LAYOUT_IDS],
        expected_layout_ids=tuple(CONTRACT.PAIR_LAYOUT_IDS),
    )["layout_id"]
    selected_three = metrics.select_training_layout(
        [selection["scores"][value] for value in CONTRACT.THREE_LAYOUT_IDS],
        expected_layout_ids=tuple(CONTRACT.THREE_LAYOUT_IDS),
    )["layout_id"]
    if (
        selected_dual != selection["selected_dual_layout"]
        or selected_three != selection["selected_three_origin_layout"]
    ):
        raise RuntimeError("training layout selection recomputation drift")


def _persist_training_layout_evidence(
    records: dict[str, dict[str, Any]],
    state_ids: Sequence[str],
    layout_ids: Sequence[str],
) -> dict[str, Any]:
    """Persist every label-free training transition/layout scoring input."""

    path = OUTPUT_ROOT / "layout_selection/training_layout_evidence.jsonl.gz"
    temporary, raw, stream = _gzip_jsonl_writer(path)
    rows_written = 0
    try:
        for state_id in state_ids:
            state = records[state_id]
            transition_rows = tuple(state["transition_rows"])
            for transition_index, transition in enumerate(transition_rows):
                if int(transition["transition_index"]) != transition_index:
                    raise RuntimeError(
                        f"{state_id}: label-free transition order drift"
                    )
                layout_metrics: dict[str, Any] = {}
                protected_link_names = tuple(state["protected_link_names"])
                if len(protected_link_names) != LINKS:
                    raise RuntimeError("layout ledger protected-link cardinality drift")
                for layout_id in layout_ids:
                    values = state["layouts"][layout_id]
                    supported = int(
                        values["transition_supported_count"][transition_index]
                    )
                    total = int(values["transition_total_count"][transition_index])
                    nominal = int(
                        values["transition_nominal_count"][transition_index]
                    )
                    self_occluded = int(
                        values["transition_self_occluded_count"][transition_index]
                    )
                    if total != STEPS * LINKS or supported < 0 or supported > total:
                        raise RuntimeError("layout transition support cardinality drift")
                    if nominal < 0 or nominal > total or self_occluded > nominal:
                        raise RuntimeError("layout transition occlusion cardinality drift")
                    region_counts = {
                        region: {
                            "supported_witnesses": int(
                                values["transition_region_supported_count"][region][
                                    transition_index
                                ]
                            ),
                            "total_witnesses": int(
                                values["transition_region_total_count"][region][
                                    transition_index
                                ]
                            ),
                        }
                        for region in _region_link_masks()
                    }
                    link_counts = {
                        link_name: {
                            "supported_witnesses": int(
                                values["transition_link_supported_count"][
                                    transition_index
                                ][link_index]
                            ),
                            "total_witnesses": int(
                                values["transition_link_total_count"][
                                    transition_index
                                ][link_index]
                            ),
                        }
                        for link_index, link_name in enumerate(
                            protected_link_names
                        )
                    }
                    layout_metrics[layout_id] = {
                        "mount_ids": list(_layout_mounts(layout_id)),
                        "supported_witnesses": supported,
                        "total_witnesses": total,
                        "support_fraction": supported / total,
                        "unsupported_swept_volume_fraction": 1.0
                        - supported / total,
                        "nominally_eligible_witnesses": nominal,
                        "all_origin_self_occluded_witnesses": self_occluded,
                        "self_occluded_fraction_of_nominal": (
                            None if nominal == 0 else self_occluded / nominal
                        ),
                        "body_region_counts": region_counts,
                        "body_region_support": {
                            region: row["supported_witnesses"]
                            / row["total_witnesses"]
                            for region, row in region_counts.items()
                        },
                        "protected_link_counts": link_counts,
                        "protected_link_support": {
                            link_name: row["supported_witnesses"]
                            / row["total_witnesses"]
                            for link_name, row in link_counts.items()
                        },
                    }
                row = {
                    "schema": "minimum_multi_origin_training_layout_transition_evidence_v1",
                    "state_id": state_id,
                    "family": state["family"],
                    "role": "development_training",
                    "transition_index": transition_index,
                    "transition_key": f"{state_id}:transition:{transition_index}",
                    "transition_kind": str(transition["level"]),
                    "current_action_index": int(
                        transition["current_action_index"]
                    ),
                    "action_index": int(transition["action_index"]),
                    "layout_metrics": layout_metrics,
                    "contact_label": None,
                    "safe_action_count": None,
                    "route_outcome": None,
                }
                stream.write(canonical_bytes(row))
                rows_written += 1
        _finalize_gzip_jsonl(path, temporary, raw, stream)
    except BaseException:
        try:
            stream.close()
        finally:
            raw.close()
            temporary.unlink(missing_ok=True)
        raise
    binding = {
        "schema": "minimum_multi_origin_training_layout_evidence_v1",
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows_written,
        "role": "development_training",
        "layout_ids": list(layout_ids),
        "body_region_ids": list(_region_link_masks()),
        "outcome_fields_used_by_layout_objective": [],
    }
    binding["content_digest"] = content_digest(binding)
    _validate_training_layout_evidence_binding(binding)
    return binding


def select_layouts() -> dict[str, Any]:
    preexecution = validate_preflight()
    existing = OUTPUT_ROOT / "layout_selection/selected_layouts.json"
    if existing.is_file():
        receipt = json.loads(existing.read_text())
        core = dict(receipt)
        declared = core.pop("content_digest", None)
        if (
            declared != content_digest(core)
            or receipt.get("pass") is not True
            or receipt.get("source_freeze_commit") != preexecution["head"]
            or receipt.get("contract_sha256") != preexecution["contract_sha256"]
        ):
            raise RuntimeError("existing layout-selection receipt is not reusable")
        _validate_training_layout_evidence_binding(
            receipt.get("training_layout_evidence", {})
        )
        _validate_training_layout_evidence_rows(
            receipt["training_layout_evidence"], receipt
        )
        return receipt
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    context = corpus_adapter.load_corpus_context(ROOT)
    state_ids = tuple(context.state_ids_for_role("training"))
    if len(state_ids) != EXPECTED["training_states"]:
        raise RuntimeError("training-role state cardinality drift")
    records: dict[str, dict[str, Any]] = {}
    worker_count = 12
    os.environ.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    process_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count, mp_context=process_context
    ) as executor:
        futures = {
            executor.submit(_layout_selection_state_worker, state_id): state_id
            for state_id in state_ids
        }
        for ordinal, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            records[row["state_id"]] = row
            print(
                json.dumps(
                    {
                        "layout_selection_states": ordinal,
                        "total": len(state_ids),
                        "state_id": row["state_id"],
                    }
                ),
                flush=True,
            )
    layout_ids = tuple(CONTRACT.PAIR_LAYOUT_IDS) + tuple(CONTRACT.THREE_LAYOUT_IDS)
    scores: dict[str, Any] = {}
    for layout_id in layout_ids:
        rows = [records[state_id]["layouts"][layout_id] for state_id in state_ids]
        transition_support = np.concatenate(
            [np.asarray(row["transition_support"], np.float64) for row in rows]
        )
        region_supported = {
            region: sum(int(row["region_supported"][region]) for row in rows)
            for region in _region_link_masks()
        }
        region_total = {
            region: sum(int(row["region_total"][region]) for row in rows)
            for region in _region_link_masks()
        }
        link_supported = np.sum(
            np.asarray([row["link_supported"] for row in rows], np.int64),
            axis=0,
        )
        link_total = np.sum(
            np.asarray([row["link_total"] for row in rows], np.int64), axis=0
        )
        if len(link_supported) != LINKS or np.any(link_total <= 0):
            raise RuntimeError("layout-selection per-link denominator drift")
        per_family: dict[str, dict[str, int]] = defaultdict(
            lambda: {"supported": 0, "total": 0}
        )
        for state_id in state_ids:
            family = records[state_id]["family"]
            row = records[state_id]["layouts"][layout_id]
            per_family[family]["supported"] += int(row["supported"])
            per_family[family]["total"] += int(row["total"])
        scores[layout_id] = {
            "layout_id": layout_id,
            "mount_ids": list(_layout_mounts(layout_id)),
            "region_support": {
                region: region_supported[region] / region_total[region]
                for region in region_supported
            },
            "per_link_support": {
                link_name: float(link_supported[index] / link_total[index])
                for index, link_name in enumerate(corpus_adapter.PROTECTED_LINK_NAMES)
            },
            "per_link_counts": {
                link_name: {
                    "supported_witnesses": int(link_supported[index]),
                    "total_witnesses": int(link_total[index]),
                }
                for index, link_name in enumerate(corpus_adapter.PROTECTED_LINK_NAMES)
            },
            "minimum_body_region_support": min(
                region_supported[region] / region_total[region]
                for region in region_supported
            ),
            "transition_support_p05": float(
                np.percentile(transition_support, 5, method="linear")
            ),
            "rear_limb_support": region_supported["REAR_LIMBS"]
            / region_total["REAR_LIMBS"],
            "calf_support": region_supported["CALVES"] / region_total["CALVES"],
            "overall_mean_support": sum(int(row["supported"]) for row in rows)
            / sum(int(row["total"]) for row in rows),
            "self_occluded_fraction": sum(
                int(row["self_occluded"]) for row in rows
            )
            / sum(int(row["nominal"]) for row in rows),
            "minimum_family_support": min(
                value["supported"] / value["total"] for value in per_family.values()
            ),
            "per_family_support": {
                family: value["supported"] / value["total"]
                for family, value in sorted(per_family.items())
            },
            "transition_count": len(transition_support),
        }
    selected_dual = metrics.select_training_layout(
        [scores[value] for value in CONTRACT.PAIR_LAYOUT_IDS],
        expected_layout_ids=tuple(CONTRACT.PAIR_LAYOUT_IDS),
    )
    selected_three = metrics.select_training_layout(
        [scores[value] for value in CONTRACT.THREE_LAYOUT_IDS],
        expected_layout_ids=tuple(CONTRACT.THREE_LAYOUT_IDS),
    )
    training_layout_evidence = _persist_training_layout_evidence(
        records, state_ids, layout_ids
    )
    receipt = {
        "schema": "minimum_multi_origin_label_free_layout_selection_v1",
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_freeze_commit": preexecution["head"],
        "contract_sha256": preexecution["contract_sha256"],
        "mount_library_sha256": preexecution["mount_library_sha256"],
        "role": "development_training",
        "states": len(state_ids),
        "training_state_count": len(state_ids),
        "contact_labels_used_for_layout_selection": False,
        "frozen_outcomes_read_only_for_corpus_custody_validation": True,
        "outcome_fields_read_by_layout_objective": [],
        "outcome_fields_used_by_layout_objective": [],
        "safe_action_counts_used_by_layout_objective": 0,
        "route_outcomes_used_by_layout_objective": 0,
        "calibration_rows_read": 0,
        "heldout_rows_read": 0,
        "frozen_before_calibration": True,
        "label_free_geometry_reuse": {
            "fields": list(LABEL_FREE_LAYOUT_REUSE_FIELDS),
            "boundary_snapshot_digest_required": True,
            "numeric_tolerance": 0.0,
            "action_group_partition_only": True,
            "outcome_fields_accessed": [],
            "representatives": sum(
                int(records[state_id]["label_free_geometry_representatives"])
                for state_id in state_ids
            ),
        },
        "training_layout_evidence": training_layout_evidence,
        "scores": scores,
        "candidate_metrics": scores,
        "pair_candidates": [scores[value] for value in CONTRACT.PAIR_LAYOUT_IDS],
        "three_candidates": [scores[value] for value in CONTRACT.THREE_LAYOUT_IDS],
        "selected_dual_layout": selected_dual["layout_id"],
        "selected_three_origin_layout": selected_three["layout_id"],
        "selected_pair_layout_id": selected_dual["layout_id"],
        "selected_three_layout_id": selected_three["layout_id"],
        "selection_metric_order": [
            "greatest_minimum_body_region_support",
            "greatest_transition_support_p05",
            "greatest_rear_limb_support",
            "greatest_calf_support",
            "greatest_overall_mean_support",
            "lowest_self_occluded_fraction",
            "fixed_layout_identifier",
        ],
        "selection_keys": {
            "dual": selected_dual["selection_key"],
            "three": selected_three["selection_key"],
        },
        "worker_count": worker_count,
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    _validate_training_layout_evidence_rows(training_layout_evidence, receipt)
    atomic_json(existing, receipt)
    return receipt


def _empty_condition_evidence(
    transitions: int, origin_count: int
) -> dict[str, np.ndarray]:
    shape = (transitions, len(MODES), STEPS, LINKS)
    arrays: dict[str, np.ndarray] = {}
    for field in STORED_FLOAT_EVIDENCE_FIELDS:
        arrays[field] = np.full(
            shape, np.inf if field == "clearance_m" else np.nan, np.float32
        )
    for field in STORED_BOOL_EVIDENCE_FIELDS:
        arrays[field] = np.zeros(shape, np.uint8)
    for field in STORED_INT16_EVIDENCE_FIELDS:
        fill = -1 if field in {
            "responsible_object_index",
            "support_object_index",
            "responsible_acquisition_index",
            "support_acquisition_index",
            "self_occluder_geom_index",
            "self_occluder_link_index",
        } else 0
        arrays[field] = np.full(shape, fill, np.int16)
    arrays["nearest_ray_index"] = np.full(shape, -1, np.int32)
    arrays["support_nearest_ray_index"] = np.full(shape, -1, np.int32)
    arrays["supporting_origin_bitmask"] = np.zeros(shape, np.uint8)
    arrays["supporting_origin_count"] = np.zeros(shape, np.uint8)
    arrays["responsible_origin_index"] = np.full(shape, -1, np.int8)
    origin_shape = (transitions, len(MODES), origin_count, STEPS, LINKS)
    for field in (
        "per_origin_support",
        "per_origin_event_time_support",
        "per_origin_nominal_fov",
        "per_origin_direct_visibility",
        "per_origin_self_occluded",
        "per_origin_finite_scan_support_inherited",
    ):
        arrays[field] = np.zeros(origin_shape, np.uint8)
    arrays["per_origin_point_support_count"] = np.zeros(origin_shape, np.int16)
    arrays["per_origin_support_acquisition_index"] = np.full(
        origin_shape, -1, np.int16
    )
    arrays["per_origin_nearest_ray_index"] = np.full(origin_shape, -1, np.int32)
    arrays["per_origin_point_age_s"] = np.full(origin_shape, np.nan, np.float32)
    return arrays


def _store_condition_evidence(
    arrays: dict[str, np.ndarray],
    transition: int,
    mode_index: int,
    evidence: dict[str, np.ndarray],
) -> None:
    for field in STORED_FLOAT_EVIDENCE_FIELDS:
        if field in evidence:
            arrays[field][transition, mode_index] = np.asarray(
                evidence[field], np.float32
            )
    for field in STORED_BOOL_EVIDENCE_FIELDS:
        arrays[field][transition, mode_index] = np.asarray(
            evidence[field], np.uint8
        )
    for field in STORED_INT16_EVIDENCE_FIELDS:
        arrays[field][transition, mode_index] = np.asarray(
            evidence[field], np.int16
        )
    for field, dtype in (
        ("nearest_ray_index", np.int32),
        ("support_nearest_ray_index", np.int32),
        ("supporting_origin_bitmask", np.uint8),
        ("supporting_origin_count", np.uint8),
        ("responsible_origin_index", np.int8),
    ):
        if field in evidence:
            arrays[field][transition, mode_index] = np.asarray(evidence[field], dtype)
    origin_count = len(np.asarray(evidence.get("origin_ids", ())))
    if origin_count:
        for field, dtype in (
            ("per_origin_support", np.uint8),
            ("per_origin_event_time_support", np.uint8),
            ("per_origin_nominal_fov", np.uint8),
            ("per_origin_direct_visibility", np.uint8),
            ("per_origin_self_occluded", np.uint8),
            ("per_origin_finite_scan_support_inherited", np.uint8),
            ("per_origin_point_support_count", np.int16),
            ("per_origin_support_acquisition_index", np.int16),
            ("per_origin_nearest_ray_index", np.int32),
            ("per_origin_point_age_s", np.float32),
        ):
            arrays[field][transition, mode_index] = np.asarray(
                evidence[field], dtype
            )


def _copy_condition_evidence(
    arrays: dict[str, np.ndarray], representative: int, copy: int
) -> None:
    for value in arrays.values():
        value[copy] = value[representative]


def _condition_spec(
    condition_id: str, selection: dict[str, Any]
) -> tuple[str, tuple[str, ...], str]:
    if condition_id in CONTRACT.DUAL_CONDITION_IDS:
        layout_id = str(selection["selected_dual_layout"])
    elif condition_id in CONTRACT.THREE_CONDITION_IDS:
        layout_id = str(selection["selected_three_origin_layout"])
    elif condition_id == DIAGNOSTIC_CONDITION:
        layout_id = "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK"
    else:
        raise ValueError(condition_id)
    if "REALISTIC" in condition_id:
        kind = "REALISTIC"
    elif "SPHERICAL" in condition_id:
        kind = "DENSE_SPHERICAL"
    else:
        kind = "DENSE_L2_FOV"
    return layout_id, _layout_mounts(layout_id), kind


def _condition_allows_geometry_evidence_copy(
    condition_id: str, selection: dict[str, Any]
) -> bool:
    """Dense analytic evidence is geometry-copyable; UID-phased scans are not."""

    return _condition_spec(condition_id, selection)[2] != "REALISTIC"


def _condition_support_hierarchy(
    realistic_condition_id: str,
) -> tuple[str, str]:
    mapping = {
        "DUAL_REALISTIC_L2_SCAN": (
            "DUAL_DENSE_L2_FOV_UPPER_BOUND",
            "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
        ),
        "THREE_REALISTIC_L2_SCAN": (
            "THREE_DENSE_L2_FOV_UPPER_BOUND",
            "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
        ),
    }
    try:
        return mapping[realistic_condition_id]
    except KeyError as error:
        raise ValueError(realistic_condition_id) from error


def _support_hierarchy_key(realistic_condition_id: str) -> str:
    dense_l2, dense_spherical = _condition_support_hierarchy(
        realistic_condition_id
    )
    return f"{realistic_condition_id}<={dense_l2}<={dense_spherical}"


def _validate_support_dominance_shape(
    value: dict[str, Any], condition_ids: Sequence[str]
) -> None:
    realistic = tuple(
        condition_id for condition_id in condition_ids if "REALISTIC" in condition_id
    )
    if not realistic:
        if value != {}:
            raise RuntimeError("dense-only phase has unexpected support-dominance rows")
        return
    expected_chains = {_support_hierarchy_key(condition_id) for condition_id in realistic}
    if set(value) != expected_chains:
        raise RuntimeError("support-dominance condition-chain cardinality drift")
    required = set(
        CONTRACT.build_output_schema()["files"]["materialization_index"][
            "support_dominance_required_keys"
        ]
    )
    for chain in expected_chains:
        if set(value[chain]) != set(MODES):
            raise RuntimeError(f"{chain}: support-dominance evidence-mode drift")
        for mode_id in MODES:
            row = value[chain][mode_id]
            inherited = row.get("inherited_support_witnesses_by_origin", {})
            expected_origins = 2 if chain.startswith("DUAL_") else 3
            if (
                not required.issubset(row)
                or row.get("condition_chain") != chain
                or row.get("evidence_mode") != mode_id
                or len(inherited) != expected_origins
                or any(mount_id not in CONTRACT.MOUNT_IDS for mount_id in inherited)
                or any(
                    set(counts) != {"dense_l2", "dense_spherical"}
                    or int(counts["dense_l2"]) < 0
                    or int(counts["dense_spherical"]) < 0
                    for counts in inherited.values()
                )
                or row.get("pass") is not True
                or int(row.get("per_origin_queries_checked", 0)) <= 0
                or int(row.get("fused_queries_checked", 0)) <= 0
                or int(row.get("clearance_monotonicity_pairs_checked", 0)) <= 0
                or int(row.get("per_origin_subset_violations", -1)) != 0
                or int(row.get("fused_subset_violations", -1)) != 0
                or int(row.get("clearance_monotonicity_violations", -1)) != 0
                or float(row.get("maximum_clearance_excess_m", math.inf)) > 1e-9
            ):
                raise RuntimeError(f"{chain}/{mode_id}: support dominance failed")


def _audit_selection() -> tuple[dict[str, list[str]], set[str]]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    by_role = BASE.frozen_audit_subset(context)
    return by_role, {
        identity for values in by_role.values() for identity in values
    }


def _write_multi_origin_raw_audit(
    *,
    transition_uid: str,
    role: str,
    family: str,
    level: str,
    action_representative_uid: str,
    geometry_representative_uid: str,
    boundary_digest: str,
    condition_id: str,
    mode_id: str,
    layout_id: str,
    targets: dict[str, np.ndarray],
    per_origin: Sequence[tuple[str, dict[str, np.ndarray]]],
    clouds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    safe = hashlib.sha256(transition_uid.encode()).hexdigest()[:16]
    relative = Path("raw_audit") / safe / f"{condition_id}__{mode_id}.npz"
    path = OUTPUT_ROOT / relative
    arrays: dict[str, np.ndarray] = {
        "target_world_xyz_m": np.asarray(targets["target_points"], np.float32),
        "target_object_index": np.asarray(targets["object_index"], np.int16),
    }
    for ordinal, (mount_id, evidence) in enumerate(per_origin):
        prefix = f"origin_{ordinal}_{mount_id.lower()}"
        arrays[f"{prefix}_support"] = np.asarray(evidence["support"], np.uint8)
        arrays[f"{prefix}_event_time_support"] = np.asarray(
            evidence["event_time_support"], np.uint8
        )
        arrays[f"{prefix}_self_occluded"] = np.asarray(
            evidence["self_occluded"], np.uint8
        )
        arrays[f"{prefix}_nominal_fov"] = np.asarray(
            evidence["nominal_fov"], np.uint8
        )
        for field, dtype in (
            ("acquisition_times_s", np.float32),
            ("acquisition_support", np.uint8),
            ("acquisition_nominal_fov", np.uint8),
            ("acquisition_self_occluded", np.uint8),
            ("acquisition_self_geom_index", np.int16),
            ("acquisition_self_link_index", np.int16),
            ("acquisition_environment_occluded", np.uint8),
        ):
            if field in evidence:
                arrays[f"{prefix}_{field}"] = np.asarray(evidence[field], dtype)
        if mount_id in clouds:
            cloud = clouds[mount_id]
            arrays[f"{prefix}_point_world_xyz_m"] = np.asarray(
                cloud["points"], np.float32
            )
            arrays[f"{prefix}_object_index"] = np.asarray(
                cloud["object_index"], np.int16
            )
            arrays[f"{prefix}_ray_index"] = np.asarray(cloud["ray_index"], np.int32)
            arrays[f"{prefix}_point_time_s"] = np.asarray(
                cloud["point_time_s"], np.float32
            )
            arrays[f"{prefix}_point_range_m"] = np.asarray(
                cloud["point_range_m"], np.float32
            )
            arrays[f"{prefix}_self_ray_index"] = np.asarray(
                cloud["self_ray_index"], np.int32
            )
            arrays[f"{prefix}_self_geom_index"] = np.asarray(
                cloud["self_geom_index"], np.int16
            )
    atomic_npz(path, **arrays)
    row = {
        "transition_uid": transition_uid,
        "role": role,
        "family": family,
        "transition_kind": level,
        "source_action_representative_transition_uid": action_representative_uid,
        "source_geometry_representative_transition_uid": geometry_representative_uid,
        "boundary_snapshot_digest": boundary_digest,
        "condition_id": condition_id,
        "evidence_mode": mode_id,
        "layout_id": layout_id,
        "mount_ids": [row[0] for row in per_origin],
        "mount_id": "MULTI_ORIGIN_UNION",
        "phase_digest_sha256_by_mount": {
            mount_id: (
                None
                if mount_id not in clouds
                else clouds[mount_id].get("phase", {}).get(
                    "phase_digest_sha256"
                )
            )
            for mount_id, _evidence in per_origin
        },
        "selection_sha256": BASE.audit_selection_sha256(
            transition_uid=transition_uid,
            role=role,
            family=family,
            transition_kind=level,
        ),
        "artifact_relative_path": str(relative),
        "artifact_sha256": sha256_file(path),
        "point_or_witness_count": int(
            sum(len(clouds[mount_id]["points"]) for mount_id in clouds)
            if clouds
            else np.asarray(targets["target_points"]).size // 3
        ),
        "bytes": path.stat().st_size,
    }
    row["phase_digest_sha256"] = content_digest(
        row["phase_digest_sha256_by_mount"]
    )
    return row


def _finalize_support_hierarchy_for_geometry_group(
    *,
    realistic_condition_id: str,
    mode_id: str,
    copies: Sequence[int],
    realistic_records: dict[
        tuple[str, str, int],
        tuple[
            list[tuple[str, dict[str, np.ndarray]]],
            list[tuple[str, dict[str, np.ndarray]]],
            list[dict[str, Any]],
        ],
    ],
    dense_l2_uid_union: list[tuple[str, dict[str, np.ndarray]]],
    dense_spherical_base: list[tuple[str, dict[str, np.ndarray]]],
) -> tuple[
    list[tuple[str, dict[str, np.ndarray]]],
    list[tuple[str, dict[str, np.ndarray]]],
    dict[str, Any],
]:
    """Enforce REALISTIC <= DENSE_L2 <= SPHERICAL for one exact trace group."""

    dense_l2 = [
        (mount_id, _copy_evidence(values))
        for mount_id, values in dense_l2_uid_union
    ]
    dense_spherical = [
        (mount_id, _copy_evidence(values))
        for mount_id, values in dense_spherical_base
    ]
    spherical_inherited: dict[str, int] = {}
    for (l2_mount, l2_values), (sphere_mount, sphere_values) in zip(
        dense_l2, dense_spherical, strict=True
    ):
        if l2_mount != sphere_mount:
            raise RuntimeError("dense hierarchy mount-order drift")
        spherical_inherited[l2_mount] = _inherit_supported_evidence(
            sphere_values,
            l2_values,
            source_is_finite_scan=False,
        )

    l2_fused = _merge_origin_evidence(dense_l2)
    sphere_fused = _merge_origin_evidence(dense_spherical)
    per_origin_queries = 0
    per_origin_violations = 0
    fused_queries = 0
    fused_violations = 0
    clearance_pairs = 0
    clearance_violations = 0
    maximum_excess = 0.0
    inherited_by_origin = {
        mount_id: {"dense_l2": 0, "dense_spherical": 0}
        for mount_id, _values in dense_l2
    }

    def compare(lower: dict[str, np.ndarray], upper: dict[str, np.ndarray]) -> tuple[int, int, int, int, float]:
        lower_support = np.asarray(lower["support"], bool)
        upper_support = np.asarray(upper["support"], bool)
        subset_violations = int((lower_support & ~upper_support).sum())
        pairs = lower_support & upper_support
        lower_clearance = np.asarray(lower["clearance_m"], np.float64)
        upper_clearance = np.asarray(upper["clearance_m"], np.float64)
        excess = np.full(lower_clearance.shape, -np.inf, np.float64)
        excess[pairs] = upper_clearance[pairs] - lower_clearance[pairs]
        finite_excess = excess[np.isfinite(excess)]
        maximum = float(np.max(finite_excess, initial=0.0))
        monotonic_violations = int((pairs & (excess > 1e-9)).sum())
        return (
            int(lower_support.size),
            subset_violations,
            int(pairs.sum()),
            monotonic_violations,
            maximum,
        )

    for copy in copies:
        per_realistic, per_candidate_l2, scan_receipts = realistic_records[
            (realistic_condition_id, mode_id, int(copy))
        ]
        if len(per_realistic) != len(dense_l2):
            raise RuntimeError("realistic/dense origin cardinality drift")
        for origin, (
            (mount_id, realistic_values),
            (union_mount, l2_values),
            (sphere_mount, sphere_values),
            (_candidate_mount, candidate_l2),
        ) in enumerate(
            zip(
                per_realistic,
                dense_l2,
                dense_spherical,
                per_candidate_l2,
                strict=True,
            )
        ):
            if mount_id != union_mount or mount_id != sphere_mount:
                raise RuntimeError("support hierarchy origin identity drift")
            first = compare(realistic_values, l2_values)
            second = compare(l2_values, sphere_values)
            per_origin_queries += first[0] + second[0]
            per_origin_violations += first[1] + second[1]
            clearance_pairs += first[2] + second[2]
            clearance_violations += first[3] + second[3]
            maximum_excess = max(maximum_excess, first[4], second[4])
            inherited_by_origin[mount_id]["dense_l2"] += int(
                scan_receipts[origin]["dense_l2_inherited_support_witnesses"]
            )
            immediate_spherical = int(
                (
                    np.asarray(candidate_l2["support"], bool)
                    & ~np.asarray(dense_spherical_base[origin][1]["support"], bool)
                ).sum()
            )
            scan_receipts[origin][
                "dense_spherical_inherited_support_witnesses"
            ] = immediate_spherical
            inherited_by_origin[mount_id]["dense_spherical"] += immediate_spherical

        realistic_fused = _merge_origin_evidence(per_realistic)
        first_fused = compare(realistic_fused, l2_fused)
        second_fused = compare(l2_fused, sphere_fused)
        fused_queries += first_fused[0] + second_fused[0]
        fused_violations += first_fused[1] + second_fused[1]
        clearance_pairs += first_fused[2] + second_fused[2]
        clearance_violations += first_fused[3] + second_fused[3]
        maximum_excess = max(maximum_excess, first_fused[4], second_fused[4])

    passed = (
        per_origin_violations == 0
        and fused_violations == 0
        and clearance_violations == 0
    )
    summary = {
        "condition_chain": _support_hierarchy_key(realistic_condition_id),
        "evidence_mode": mode_id,
        "per_origin_queries_checked": per_origin_queries,
        "per_origin_subset_violations": per_origin_violations,
        "fused_queries_checked": fused_queries,
        "fused_subset_violations": fused_violations,
        "clearance_monotonicity_pairs_checked": clearance_pairs,
        "clearance_monotonicity_violations": clearance_violations,
        "maximum_clearance_excess_m": maximum_excess,
        "inherited_support_witnesses_by_origin": inherited_by_origin,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(
            f"{realistic_condition_id}/{mode_id}: support hierarchy validation failed"
        )
    return dense_l2, dense_spherical, summary


def _materialize_state_conditions(
    state_id: str,
    condition_ids: tuple[str, ...],
    audit_identities: tuple[str, ...],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    started = time.time()
    context = corpus_adapter.load_corpus_context(ROOT)
    state = corpus_adapter.load_state(context, state_id, shard_fields=())
    selection = json.loads(
        (OUTPUT_ROOT / "layout_selection/selected_layouts.json").read_text()
    )
    predecessor_path = PREDECESSOR_OUTPUT / "states" / f"{state_id}.npz"
    if not predecessor_path.is_file():
        raise RuntimeError(f"predecessor state evidence missing: {state_id}")
    with np.load(predecessor_path, allow_pickle=False) as predecessor:
        target_points = np.asarray(predecessor["target_points_world_m"], np.float64)
        target_object = np.asarray(predecessor["target_object_index"], np.int16)
        target_geom = np.asarray(predecessor["target_geom_index"], np.int16)
        oracle_clearance = np.asarray(predecessor["oracle_clearance_m"], np.float64)
        action_representative = np.asarray(
            predecessor["action_representative_transition"], np.int32
        )
        geometry_representative = np.asarray(
            predecessor["geometry_representative_transition"], np.int32
        )
    if len(target_points) != state.shard.transition_count:
        raise RuntimeError("predecessor/new transition alignment drift")
    geometry_groups: dict[int, list[int]] = defaultdict(list)
    for copy, representative in enumerate(geometry_representative):
        geometry_groups[int(representative)].append(copy)
    if len(geometry_groups) != state.action_copy_map.representative_count + sum(
        int(value != action_representative[index])
        for index, value in enumerate(geometry_representative)
        if int(value) == index
    ):
        # A stricter exact total check follows through the receipt; this guard
        # catches malformed, non-idempotent provenance arrays immediately.
        if any(geometry_representative[int(value)] != value for value in geometry_representative):
            raise RuntimeError("geometry representative mapping is not idempotent")
    boxes, specs = BASE.runtime_geometry_contract(state)
    arrays_by_condition = {
        condition_id: _empty_condition_evidence(
            state.shard.transition_count,
            len(_condition_spec(condition_id, selection)[1]),
        )
        for condition_id in condition_ids
    }
    scan_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    support_dominance: dict[str, dict[str, dict[str, Any]]] = {}
    audit_set = set(audit_identities)
    for ordinal, representative in enumerate(sorted(geometry_groups), 1):
        dense_origins_by_condition_mode: dict[
            tuple[str, str], list[tuple[str, dict[str, np.ndarray]]]
        ] = {}
        realistic_records: dict[
            tuple[str, str, int],
            tuple[
                list[tuple[str, dict[str, np.ndarray]]],
                list[tuple[str, dict[str, np.ndarray]]],
                list[dict[str, Any]],
            ],
        ] = {}
        dense_l2_uid_unions: dict[
            tuple[str, str], list[tuple[str, dict[str, np.ndarray]]]
        ] = {}
        identity_row = dict(state.transition_rows[representative])
        transition_uid = str(identity_row["identity"])
        boundary_qpos, boundary_geom, boundary_digest = BASE.transition_boundary(
            state, identity_row
        )
        qpos = np.asarray(state.shard.arrays["qpos"][representative], np.float64)
        geom = np.asarray(state.shard.arrays["geom_transform"][representative], np.float64)
        targets = {
            "target_points": target_points[representative],
            "object_index": target_object[representative],
            "geom_index": target_geom[representative],
            "clearance_m": oracle_clearance[representative],
        }
        for condition_id in condition_ids:
            layout_id, mounts, kind = _condition_spec(condition_id, selection)
            condition_modes = (
                ("TRUE_FUTURE_OBSERVABILITY_CLOUD",)
                if condition_id == DIAGNOSTIC_CONDITION
                else MODES
            )
            for mode_id in condition_modes:
                mode_index = MODES.index(mode_id)
                if kind == "REALISTIC":
                    dense_l2_condition, _dense_spherical_condition = (
                        _condition_support_hierarchy(condition_id)
                    )
                    dense_rows = dense_origins_by_condition_mode.get(
                        (dense_l2_condition, mode_id)
                    )
                    if dense_rows is None:
                        raise RuntimeError(
                            f"{condition_id}/{mode_id}: matched dense L2 must materialize first"
                        )
                    combined, per_origin, clouds, transition_scans, matched_dense = (
                        _realistic_layout_evidence(
                            condition_id=condition_id,
                            layout_id=layout_id,
                            mounts=mounts,
                            mode_id=mode_id,
                            transition_index=representative,
                            action_representative_transition_index=int(
                                action_representative[representative]
                            ),
                            geometry_representative_transition_index=representative,
                            transition_uid=transition_uid,
                            boundary_qpos=boundary_qpos,
                            boundary_geom_transform=boundary_geom,
                            qpos=qpos,
                            geom_transform=geom,
                            environment_boxes=boxes,
                            robot_specs=specs,
                            targets=targets,
                            matched_dense_l2_by_mount=dict(dense_rows),
                        )
                    )
                    scan_rows.extend(transition_scans)
                    realistic_records[(condition_id, mode_id, representative)] = (
                        per_origin,
                        matched_dense,
                        transition_scans,
                    )
                    dense_l2_uid_unions[(condition_id, mode_id)] = [
                        (mount_id, _copy_evidence(values))
                        for mount_id, values in matched_dense
                    ]
                else:
                    per_origin = []
                    clouds = {}
                    for mount_id in mounts:
                        values = _origin_dense_evidence(
                            mount_id=mount_id,
                            spherical=kind == "DENSE_SPHERICAL",
                            mode_id=mode_id,
                            boundary_qpos=boundary_qpos,
                            boundary_geom_transform=boundary_geom,
                            qpos=qpos,
                            geom_transform=geom,
                            environment_boxes=boxes,
                            robot_specs=specs,
                            targets=targets,
                            installed_mount_ids=mounts,
                        )
                        values["responsible_geom_index"] = target_geom[representative]
                        per_origin.append((mount_id, values))
                    combined = _merge_origin_evidence(per_origin)
                    dense_origins_by_condition_mode[(condition_id, mode_id)] = [
                        (mount_id, _copy_evidence(values))
                        for mount_id, values in per_origin
                    ]
                _store_condition_evidence(
                    arrays_by_condition[condition_id],
                    representative,
                    mode_index,
                    combined,
                )
                selected_copy_indices = [
                    copy
                    for copy in geometry_groups[representative]
                    if _condition_allows_geometry_evidence_copy(
                        condition_id, selection
                    )
                    or copy == representative
                    if str(state.transition_rows[copy]["identity"]) in audit_set
                ]
                for selected_copy in selected_copy_indices:
                    selected_row = dict(state.transition_rows[selected_copy])
                    selected_uid = str(selected_row["identity"])
                    action_index = int(action_representative[selected_copy])
                    raw_rows.append(
                        _write_multi_origin_raw_audit(
                            transition_uid=selected_uid,
                            role=state.role,
                            family=state.family,
                            level=str(selected_row["level"]),
                            action_representative_uid=str(
                                state.transition_rows[action_index]["identity"]
                            ),
                            geometry_representative_uid=transition_uid,
                            boundary_digest=boundary_digest,
                            condition_id=condition_id,
                            mode_id=mode_id,
                            layout_id=layout_id,
                            targets=targets,
                            per_origin=per_origin,
                            clouds=clouds,
                        )
                    )
        for copy in geometry_groups[representative]:
            if copy == representative:
                continue
            for condition_id, arrays in arrays_by_condition.items():
                layout_id, mounts, kind = _condition_spec(condition_id, selection)
                if _condition_allows_geometry_evidence_copy(condition_id, selection):
                    _copy_condition_evidence(arrays, representative, copy)
                    continue
                copy_row = dict(state.transition_rows[copy])
                copy_uid = str(copy_row["identity"])
                copy_boundary_qpos, copy_boundary_geom, copy_boundary_digest = (
                    BASE.transition_boundary(state, copy_row)
                )
                copy_qpos = np.asarray(state.shard.arrays["qpos"][copy], np.float64)
                copy_geom = np.asarray(
                    state.shard.arrays["geom_transform"][copy], np.float64
                )
                copy_targets = {
                    "target_points": target_points[copy],
                    "object_index": target_object[copy],
                    "geom_index": target_geom[copy],
                    "clearance_m": oracle_clearance[copy],
                }
                for mode_id in MODES:
                    mode_index = MODES.index(mode_id)
                    dense_l2_condition, _dense_spherical_condition = (
                        _condition_support_hierarchy(condition_id)
                    )
                    dense_rows = dense_origins_by_condition_mode.get(
                        (dense_l2_condition, mode_id)
                    )
                    if dense_rows is None:
                        raise RuntimeError(
                            f"{condition_id}/{mode_id}: matched dense L2 evidence missing"
                        )
                    combined, per_origin, clouds, transition_scans, matched_dense = (
                        _realistic_layout_evidence(
                            condition_id=condition_id,
                            layout_id=layout_id,
                            mounts=mounts,
                            mode_id=mode_id,
                            transition_index=copy,
                            action_representative_transition_index=int(
                                action_representative[copy]
                            ),
                            geometry_representative_transition_index=representative,
                            transition_uid=copy_uid,
                            boundary_qpos=copy_boundary_qpos,
                            boundary_geom_transform=copy_boundary_geom,
                            qpos=copy_qpos,
                            geom_transform=copy_geom,
                            environment_boxes=boxes,
                            robot_specs=specs,
                            targets=copy_targets,
                            matched_dense_l2_by_mount=dict(dense_rows),
                        )
                    )
                    _store_condition_evidence(arrays, copy, mode_index, combined)
                    scan_rows.extend(transition_scans)
                    realistic_records[(condition_id, mode_id, copy)] = (
                        per_origin,
                        matched_dense,
                        transition_scans,
                    )
                    union_rows = dense_l2_uid_unions[(condition_id, mode_id)]
                    for (union_mount, union_values), (
                        candidate_mount,
                        candidate_values,
                    ) in zip(union_rows, matched_dense, strict=True):
                        if union_mount != candidate_mount:
                            raise RuntimeError("matched dense mount-order drift")
                        _union_dense_uid_evidence(union_values, candidate_values)
                    if copy_uid in audit_set:
                        action_index = int(action_representative[copy])
                        raw_rows.append(
                            _write_multi_origin_raw_audit(
                                transition_uid=copy_uid,
                                role=state.role,
                                family=state.family,
                                level=str(copy_row["level"]),
                                action_representative_uid=str(
                                    state.transition_rows[action_index]["identity"]
                                ),
                                geometry_representative_uid=transition_uid,
                                boundary_digest=copy_boundary_digest,
                                condition_id=condition_id,
                                mode_id=mode_id,
                                layout_id=layout_id,
                                targets=copy_targets,
                                per_origin=per_origin,
                                clouds=clouds,
                            )
                        )
        realistic_conditions = tuple(
            condition_id
            for condition_id in condition_ids
            if _condition_spec(condition_id, selection)[2] == "REALISTIC"
        )
        for realistic_condition_id in realistic_conditions:
            dense_l2_condition, dense_spherical_condition = (
                _condition_support_hierarchy(realistic_condition_id)
            )
            layout_id, _mounts, _kind = _condition_spec(
                realistic_condition_id, selection
            )
            for mode_id in MODES:
                dense_l2, dense_spherical, summary = (
                    _finalize_support_hierarchy_for_geometry_group(
                        realistic_condition_id=realistic_condition_id,
                        mode_id=mode_id,
                        copies=geometry_groups[representative],
                        realistic_records=realistic_records,
                        dense_l2_uid_union=dense_l2_uid_unions[
                            (realistic_condition_id, mode_id)
                        ],
                        dense_spherical_base=dense_origins_by_condition_mode[
                            (dense_spherical_condition, mode_id)
                        ],
                    )
                )
                mode_index = MODES.index(mode_id)
                dense_l2_combined = _merge_origin_evidence(dense_l2)
                dense_spherical_combined = _merge_origin_evidence(dense_spherical)
                for copy in geometry_groups[representative]:
                    _store_condition_evidence(
                        arrays_by_condition[dense_l2_condition],
                        copy,
                        mode_index,
                        dense_l2_combined,
                    )
                    _store_condition_evidence(
                        arrays_by_condition[dense_spherical_condition],
                        copy,
                        mode_index,
                        dense_spherical_combined,
                    )

                chain = str(summary["condition_chain"])
                target = support_dominance.setdefault(chain, {}).setdefault(
                    mode_id,
                    {
                        "condition_chain": chain,
                        "evidence_mode": mode_id,
                        "per_origin_queries_checked": 0,
                        "per_origin_subset_violations": 0,
                        "fused_queries_checked": 0,
                        "fused_subset_violations": 0,
                        "clearance_monotonicity_pairs_checked": 0,
                        "clearance_monotonicity_violations": 0,
                        "maximum_clearance_excess_m": 0.0,
                        "inherited_support_witnesses_by_origin": {},
                        "pass": True,
                    },
                )
                for field in (
                    "per_origin_queries_checked",
                    "per_origin_subset_violations",
                    "fused_queries_checked",
                    "fused_subset_violations",
                    "clearance_monotonicity_pairs_checked",
                    "clearance_monotonicity_violations",
                ):
                    target[field] += int(summary[field])
                target["maximum_clearance_excess_m"] = max(
                    float(target["maximum_clearance_excess_m"]),
                    float(summary["maximum_clearance_excess_m"]),
                )
                for mount_id, inherited in summary[
                    "inherited_support_witnesses_by_origin"
                ].items():
                    mount_target = target[
                        "inherited_support_witnesses_by_origin"
                    ].setdefault(
                        mount_id, {"dense_l2": 0, "dense_spherical": 0}
                    )
                    mount_target["dense_l2"] += int(inherited["dense_l2"])
                    mount_target["dense_spherical"] += int(
                        inherited["dense_spherical"]
                    )
                target["pass"] = bool(target["pass"] and summary["pass"])

                group_uids = {
                    str(state.transition_rows[copy]["identity"])
                    for copy in geometry_groups[representative]
                }
                raw_rows = [
                    row
                    for row in raw_rows
                    if not (
                        row["transition_uid"] in group_uids
                        and row["condition_id"]
                        in (dense_l2_condition, dense_spherical_condition)
                        and row["evidence_mode"] == mode_id
                    )
                ]
                for copy in geometry_groups[representative]:
                    copy_row = dict(state.transition_rows[copy])
                    copy_uid = str(copy_row["identity"])
                    if copy_uid not in audit_set:
                        continue
                    _copy_boundary_qpos, _copy_boundary_geom, copy_boundary_digest = (
                        BASE.transition_boundary(state, copy_row)
                    )
                    copy_targets = {
                        "target_points": target_points[copy],
                        "object_index": target_object[copy],
                        "geom_index": target_geom[copy],
                        "clearance_m": oracle_clearance[copy],
                    }
                    action_index = int(action_representative[copy])
                    for dense_condition_id, dense_per_origin in (
                        (dense_l2_condition, dense_l2),
                        (dense_spherical_condition, dense_spherical),
                    ):
                        raw_rows.append(
                            _write_multi_origin_raw_audit(
                                transition_uid=copy_uid,
                                role=state.role,
                                family=state.family,
                                level=str(copy_row["level"]),
                                action_representative_uid=str(
                                    state.transition_rows[action_index]["identity"]
                                ),
                                geometry_representative_uid=transition_uid,
                                boundary_digest=copy_boundary_digest,
                                condition_id=dense_condition_id,
                                mode_id=mode_id,
                                layout_id=layout_id,
                                targets=copy_targets,
                                per_origin=dense_per_origin,
                                clouds={},
                            )
                        )
        if ordinal % 25 == 0:
            print(
                json.dumps(
                    {
                        "state_id": state_id,
                        "geometry_representatives_done": ordinal,
                        "geometry_representatives_total": len(geometry_groups),
                        "conditions": list(condition_ids),
                    }
                ),
                flush=True,
            )
    condition_records: list[dict[str, Any]] = []
    _validate_support_dominance_shape(support_dominance, condition_ids)
    for condition_id, arrays in arrays_by_condition.items():
        path = OUTPUT_ROOT / "states" / condition_id / f"{state_id}.npz"
        atomic_npz(
            path,
            **arrays,
            action_representative_transition=action_representative,
            geometry_representative_transition=geometry_representative,
        )
        condition_records.append(
            {
                "condition_id": condition_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    receipt = {
        "schema": "minimum_multi_origin_body_range_coverage_state_phase_v1",
        "status": "PASS",
        "state_id": state_id,
        "scene_id": state.scene_id,
        "family": state.family,
        "role": state.role,
        "conditions": list(condition_ids),
        "transitions": state.shard.transition_count,
        "transition_uid_by_index": [
            str(row["identity"]) for row in state.transition_rows
        ],
        "transition_uid_by_index_sha256": content_digest(
            [str(row["identity"]) for row in state.transition_rows]
        ),
        "action_representatives": len(np.unique(action_representative)),
        "geometry_representatives": len(np.unique(geometry_representative)),
        "condition_records": condition_records,
        "scan_receipts": scan_rows,
        "support_dominance": support_dominance,
        "raw_audit": raw_rows,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "source_freeze_commit": git("rev-parse", "HEAD"),
        "predecessor_state_sha256": sha256_file(predecessor_path),
        "runtime_s": time.time() - started,
        "simulator_steps": 0,
        "training_steps": 0,
        "jepa_predictor_opens": 0,
    }
    receipt["content_digest"] = content_digest(receipt)
    phase = "dual" if tuple(condition_ids) == tuple(CONTRACT.DUAL_CONDITION_IDS) else (
        "three" if tuple(condition_ids) == tuple(CONTRACT.THREE_CONDITION_IDS) else "diagnostic"
    )
    receipt_path = OUTPUT_ROOT / "state_receipts" / phase / f"{state_id}.json"
    atomic_json(receipt_path, receipt)
    return receipt


def _materialize_state_worker(
    payload: tuple[str, tuple[str, ...], tuple[str, ...]]
) -> dict[str, Any]:
    return _materialize_state_conditions(*payload)


def _phase_name(condition_ids: tuple[str, ...]) -> str:
    if condition_ids == tuple(CONTRACT.DUAL_CONDITION_IDS):
        return "dual"
    if condition_ids == tuple(CONTRACT.THREE_CONDITION_IDS):
        return "three"
    if condition_ids == (DIAGNOSTIC_CONDITION,):
        return "diagnostic"
    raise ValueError(condition_ids)


def _validate_uid_bound_scan_counts(
    scan_counts: dict[str, Any],
    condition_ids: tuple[str, ...],
    selection: dict[str, Any],
) -> None:
    realistic_conditions = {
        condition_id for condition_id in condition_ids if "REALISTIC" in condition_id
    }
    if set(scan_counts) != realistic_conditions:
        raise RuntimeError("realistic scan-condition cardinality drift")
    for condition_id in realistic_conditions:
        expected_scans = EXPECTED["transitions"] * len(
            _condition_spec(condition_id, selection)[1]
        )
        if set(scan_counts[condition_id]) != set(MODES):
            raise RuntimeError(f"{condition_id} realistic evidence-mode drift")
        for mode_id in MODES:
            values = scan_counts[condition_id][mode_id]
            if (
                int(values["sensor_scans"]) != expected_scans
                or int(values["unique_rendered_scans"]) != expected_scans
                or int(values["rays"]) != expected_scans * 6_400
            ):
                raise RuntimeError(
                    f"{condition_id}/{mode_id} UID-bound scan cardinality drift"
                )


@lru_cache(maxsize=None)
def _frozen_transition_uids(state_id: str) -> tuple[str, ...]:
    """Read the ordered UID authority without exposing outcomes to selection."""

    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    return tuple(
        str(row["identity"])
        for row in context.geometry_record(state_id)["transition_rows"]
    )


def _npz_array_metadata(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Read NPY headers inside an NPZ without materializing tensor payloads."""

    output: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                else:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
            output[Path(member).stem] = (tuple(shape), np.dtype(dtype))
    return output


def _validate_condition_npz_schema(
    record: dict[str, Any], selection: dict[str, Any]
) -> None:
    path = Path(record["shard_path"])
    metadata = _npz_array_metadata(path)
    required = tuple(
        CONTRACT.build_output_schema()["files"]["materialization_index"][
            "npz_required_arrays"
        ]
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise RuntimeError(f"condition shard required arrays missing: {path}: {missing}")
    transitions = int(record["transitions"])
    condition_id = str(record["condition_id"])
    origin_count = len(_condition_spec(condition_id, selection)[1])
    fused_shape = (transitions, len(MODES), STEPS, LINKS)
    origin_shape = (transitions, len(MODES), origin_count, STEPS, LINKS)
    expected: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        "action_representative_transition": ((transitions,), np.dtype(np.int32)),
        "geometry_representative_transition": ((transitions,), np.dtype(np.int32)),
        "support": (fused_shape, np.dtype(np.uint8)),
        "finite_scan_support_inherited": (fused_shape, np.dtype(np.uint8)),
        "per_origin_support": (origin_shape, np.dtype(np.uint8)),
        "per_origin_event_time_support": (origin_shape, np.dtype(np.uint8)),
        "per_origin_nominal_fov": (origin_shape, np.dtype(np.uint8)),
        "per_origin_direct_visibility": (origin_shape, np.dtype(np.uint8)),
        "per_origin_self_occluded": (origin_shape, np.dtype(np.uint8)),
        "per_origin_finite_scan_support_inherited": (
            origin_shape,
            np.dtype(np.uint8),
        ),
        "per_origin_point_support_count": (origin_shape, np.dtype(np.int16)),
        "per_origin_support_acquisition_index": (
            origin_shape,
            np.dtype(np.int16),
        ),
        "per_origin_nearest_ray_index": (origin_shape, np.dtype(np.int32)),
        "per_origin_point_age_s": (origin_shape, np.dtype(np.float32)),
    }
    for field in required:
        if metadata[field] != expected[field]:
            raise RuntimeError(
                f"condition shard {field} shape/dtype drift: {path}: "
                f"{metadata[field]} != {expected[field]}"
            )


def _validate_state_scan_receipts(
    receipt: dict[str, Any],
    condition_ids: tuple[str, ...],
    selection: dict[str, Any],
) -> set[tuple[str, str, str, str]]:
    """Validate every UID/mount phase row in one state receipt."""

    realistic_conditions = tuple(
        value for value in condition_ids if "REALISTIC" in value
    )
    rows = tuple(receipt.get("scan_receipts", ()))
    scan_required = set(
        CONTRACT.build_output_schema()["files"]["materialization_index"][
            "scan_receipt_required_keys"
        ]
    )
    transitions = int(receipt["transitions"])
    frozen_uids = tuple(str(value) for value in receipt.get("transition_uid_by_index", ()))
    authoritative_uids = _frozen_transition_uids(str(receipt["state_id"]))
    if (
        len(frozen_uids) != transitions
        or len(set(frozen_uids)) != transitions
        or frozen_uids != authoritative_uids
        or receipt.get("transition_uid_by_index_sha256")
        != content_digest(list(frozen_uids))
    ):
        raise RuntimeError("state transition UID binding drift")
    if not realistic_conditions:
        if rows:
            raise RuntimeError("dense-only phase has unexpected finite scan receipts")
        return set()
    expected_count = sum(
        transitions * len(MODES) * len(_condition_spec(condition, selection)[1])
        for condition in realistic_conditions
    )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"{receipt['state_id']}: UID-bound scan-row cardinality drift"
        )
    expected_phase_contract = _contract_content_digest()
    keys: set[tuple[str, str, str, str]] = set()
    index_uids: dict[int, str] = {}
    for row in rows:
        if not scan_required.issubset(row):
            raise RuntimeError("finite scan receipt key drift")
        condition_id = str(row["condition_id"])
        mode_id = str(row["evidence_mode"])
        transition_index = int(row["transition_index"])
        transition_uid = str(row["transition_uid"])
        mount_id = str(row["mount_id"])
        if condition_id not in realistic_conditions or mode_id not in MODES:
            raise RuntimeError("unexpected finite scan receipt condition/mode")
        layout_id, mounts, kind = _condition_spec(condition_id, selection)
        if (
            kind != "REALISTIC"
            or row.get("layout_id") != layout_id
            or mount_id not in mounts
            or not 0 <= transition_index < transitions
            or not transition_uid
            or int(row.get("ray_count", -1)) != 6_400
            or row.get("render_cache_reused") is not False
            or int(row.get("dense_l2_inherited_support_witnesses", -1)) < 0
            or int(row.get("dense_spherical_inherited_support_witnesses", -1)) < 0
        ):
            raise RuntimeError("malformed UID-bound scan receipt")
        previous_uid = index_uids.setdefault(transition_index, transition_uid)
        if previous_uid != transition_uid:
            raise RuntimeError("transition index has inconsistent scan UID")
        if frozen_uids[transition_index] != transition_uid:
            raise RuntimeError("scan UID does not match frozen transition identity")
        key = (condition_id, mode_id, transition_uid, mount_id)
        if key in keys:
            raise RuntimeError("duplicate UID/mount scan receipt")
        keys.add(key)
        expected_phase = CONTRACT.derive_multi_origin_scan_phases(
            contract_digest_sha256=expected_phase_contract,
            transition_uid=transition_uid,
            mount_id=mount_id,
        )
        if row.get("phase") != expected_phase:
            raise RuntimeError("UID/mount scan phase derivation drift")
    if set(index_uids) != set(range(transitions)):
        raise RuntimeError("finite scan receipts omit a transition index")
    return keys


def _validate_bound_state_phase_receipt(
    record: dict[str, Any],
    condition_ids: tuple[str, ...],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Validate a phase state receipt, including scientific cardinalities."""

    receipt_path = Path(record["state_receipt_path"])
    if (
        not receipt_path.is_file()
        or receipt_path.stat().st_size != int(record["state_receipt_bytes"])
        or sha256_file(receipt_path) != record["state_receipt_sha256"]
    ):
        raise RuntimeError(
            f"materialization state receipt drift: {receipt_path}"
        )
    receipt = json.loads(receipt_path.read_text())
    state_required = set(
        CONTRACT.build_output_schema()["files"]["materialization_index"][
            "state_phase_receipt_required_keys"
        ]
    )
    if not state_required.issubset(receipt):
        raise RuntimeError(f"materialization state receipt key drift: {receipt_path}")
    core = dict(receipt)
    declared = core.pop("content_digest", None)
    if (
        declared != content_digest(core)
        or receipt.get("status") != "PASS"
        or tuple(receipt.get("conditions", ())) != condition_ids
        or receipt.get("state_id") != record.get("state_id")
        or int(receipt.get("transitions", -1)) != int(record.get("transitions", -2))
    ):
        raise RuntimeError(f"invalid materialization state receipt: {receipt_path}")
    _validate_support_dominance_shape(
        receipt.get("support_dominance", {}), condition_ids
    )
    _validate_state_scan_receipts(receipt, condition_ids, selection)
    return receipt


def _validate_all_state_scan_receipts(
    receipts: Sequence[dict[str, Any]],
    condition_ids: tuple[str, ...],
    selection: dict[str, Any],
) -> None:
    all_keys: set[tuple[str, str, str, str]] = set()
    transition_uids: set[str] = set()
    state_required = set(
        CONTRACT.build_output_schema()["files"]["materialization_index"][
            "state_phase_receipt_required_keys"
        ]
    )
    for receipt in receipts:
        if (
            not state_required.issubset(receipt)
            or tuple(receipt.get("conditions", ())) != condition_ids
        ):
            raise RuntimeError("materialization state receipt key drift")
        _validate_support_dominance_shape(
            receipt.get("support_dominance", {}), condition_ids
        )
        state_keys = _validate_state_scan_receipts(
            receipt, condition_ids, selection
        )
        if all_keys & state_keys:
            raise RuntimeError("UID/mount scan receipt repeats across states")
        all_keys |= state_keys
        transition_uids |= {key[2] for key in state_keys}
    realistic = any("REALISTIC" in value for value in condition_ids)
    expected_uids = EXPECTED["transitions"] if realistic else 0
    if len(transition_uids) != expected_uids:
        raise RuntimeError("global finite scan transition-UID cardinality drift")


def materialize_conditions(condition_ids: tuple[str, ...]) -> dict[str, Any]:
    preexecution = validate_preflight()
    selection = select_layouts()
    phase = _phase_name(condition_ids)
    index_path = OUTPUT_ROOT / "materialization" / f"{phase}_index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        core = dict(index)
        declared = core.pop("content_digest", None)
        if (
            declared != content_digest(core)
            or index.get("status") != "PASS"
            or index.get("source_freeze_commit") != preexecution["head"]
            or tuple(index.get("conditions", ())) != condition_ids
            or index.get("states") != EXPECTED["states"]
            or index.get("transitions") != EXPECTED["transitions"]
        ):
            raise RuntimeError(f"existing {phase} materialization index is invalid")
        for record in index.get("records", ()):
            shard = Path(record["shard_path"])
            if (
                not shard.is_file()
                or shard.stat().st_size != int(record["shard_bytes"])
                or sha256_file(shard) != record["shard_sha256"]
            ):
                raise RuntimeError(f"existing materialization shard drift: {shard}")
            _validate_condition_npz_schema(record, selection)
        for record in index.get("state_records", ()):
            _validate_bound_state_phase_receipt(record, condition_ids, selection)
        manifest = Path(index["raw_audit_manifest_path"])
        if not manifest.is_file() or sha256_file(manifest) != index["raw_audit_manifest_sha256"]:
            raise RuntimeError(f"existing {phase} raw-audit manifest drift")
        _validate_support_dominance_shape(
            index.get("support_dominance", {}), condition_ids
        )
        _validate_uid_bound_scan_counts(
            index.get("scan_counts", {}), condition_ids, selection
        )
        _validate_all_state_scan_receipts(
            [
                json.loads(Path(record["state_receipt_path"]).read_text())
                for record in index.get("state_records", ())
            ],
            condition_ids,
            selection,
        )
        return index
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    audit_by_role, audit_identities = _audit_selection()
    marker = OUTPUT_ROOT / f"MATERIALIZATION_{phase.upper()}_RUNNING.json"
    atomic_json(
        marker,
        {
            "schema": "minimum_multi_origin_materialization_running_v1",
            "phase": phase,
            "pid": os.getpid(),
            "head": preexecution["head"],
            "conditions": list(condition_ids),
        },
    )
    started = time.time()
    worker_count = 12
    records_by_state: dict[str, dict[str, Any]] = {}
    try:
        os.environ.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        process_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count, mp_context=process_context
        ) as executor:
            futures = {
                executor.submit(
                    _materialize_state_worker,
                    (state_id, condition_ids, tuple(sorted(audit_identities))),
                ): state_id
                for state_id in context.state_ids
            }
            for ordinal, future in enumerate(concurrent.futures.as_completed(futures), 1):
                receipt = future.result()
                records_by_state[receipt["state_id"]] = receipt
                allocated = _directory_allocated_bytes(OUTPUT_ROOT)
                if allocated > 25_000_000_000:
                    raise RuntimeError("final storage ceiling exceeded during materialization")
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "materialized_states": ordinal,
                            "total_states": len(context.state_ids),
                            "state_id": receipt["state_id"],
                            "allocated_bytes": allocated,
                        }
                    ),
                    flush=True,
                )
        records = [records_by_state[state_id] for state_id in context.state_ids]
        _validate_all_state_scan_receipts(records, condition_ids, selection)
        raw_rows = [row for record in records for row in record["raw_audit"]]
        manifest = OUTPUT_ROOT / "raw_audit" / f"{phase}_manifest.jsonl"
        BASE.atomic_bytes(manifest, b"".join(canonical_bytes(row) for row in raw_rows))
        scan_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for record in records:
            for row in record["scan_receipts"]:
                target = scan_counts[row["condition_id"]][row["evidence_mode"]]
                target["sensor_scans"] += 1
                target["unique_rendered_scans"] += int(
                    not row["render_cache_reused"]
                )
                target["rays"] += int(row["ray_count"])
                target["environment_returns"] += int(row["environment_return_count"])
                target["robot_self_returns"] += int(row["self_return_count"])
                target["near_blind_first_hits"] += int(row["near_blind_count"])
                target["ground_returns"] += int(row["ground_return_count"])
        support_dominance: dict[str, dict[str, dict[str, Any]]] = {}
        for record in records:
            for chain, mode_rows in record.get("support_dominance", {}).items():
                for mode_id, row in mode_rows.items():
                    target = support_dominance.setdefault(chain, {}).setdefault(
                        mode_id,
                        {
                            "condition_chain": chain,
                            "evidence_mode": mode_id,
                            "per_origin_queries_checked": 0,
                            "per_origin_subset_violations": 0,
                            "fused_queries_checked": 0,
                            "fused_subset_violations": 0,
                            "clearance_monotonicity_pairs_checked": 0,
                            "clearance_monotonicity_violations": 0,
                            "maximum_clearance_excess_m": 0.0,
                            "inherited_support_witnesses_by_origin": {},
                            "pass": True,
                        },
                    )
                    for field in (
                        "per_origin_queries_checked",
                        "per_origin_subset_violations",
                        "fused_queries_checked",
                        "fused_subset_violations",
                        "clearance_monotonicity_pairs_checked",
                        "clearance_monotonicity_violations",
                    ):
                        target[field] += int(row[field])
                    target["maximum_clearance_excess_m"] = max(
                        float(target["maximum_clearance_excess_m"]),
                        float(row["maximum_clearance_excess_m"]),
                    )
                    for mount_id, inherited in row[
                        "inherited_support_witnesses_by_origin"
                    ].items():
                        mount_target = target[
                            "inherited_support_witnesses_by_origin"
                        ].setdefault(
                            mount_id, {"dense_l2": 0, "dense_spherical": 0}
                        )
                        for level in ("dense_l2", "dense_spherical"):
                            mount_target[level] += int(inherited[level])
                    target["pass"] = bool(target["pass"] and row["pass"])
        _validate_support_dominance_shape(support_dominance, condition_ids)
        index = {
            "schema": "minimum_multi_origin_body_range_coverage_materialization_phase_v1",
            "status": "PASS",
            "phase": phase,
            "conditions": list(condition_ids),
            "source_freeze_commit": preexecution["head"],
            "contract_sha256": preexecution["contract_sha256"],
            "source_closure_sha256": preexecution["source_closure_sha256"],
            "layout_selection_sha256": sha256_file(
                OUTPUT_ROOT / "layout_selection/selected_layouts.json"
            ),
            "selected_dual_layout": selection["selected_dual_layout"],
            "selected_three_origin_layout": selection[
                "selected_three_origin_layout"
            ],
            "states": len(records),
            "transitions": sum(int(row["transitions"]) for row in records),
            "action_representatives": sum(
                int(row["action_representatives"]) for row in records
            ),
            "geometry_representatives": sum(
                int(row["geometry_representatives"]) for row in records
            ),
            "physics_frames": EXPECTED["physics_frames"],
            "protected_links": LINKS,
            "raw_audit_artifacts": len(raw_rows),
            "raw_audit_manifest_path": str(manifest),
            "raw_audit_manifest_sha256": sha256_file(manifest),
            "scan_counts": {
                condition: {
                    mode: dict(values)
                    for mode, values in mode_rows.items()
                }
                for condition, mode_rows in scan_counts.items()
            },
            "support_dominance": support_dominance,
            "dense_analytic_target_query_counts": {
                condition_id: {
                    mode_id: (
                        EXPECTED["geometry_representatives"]
                        * len(_condition_spec(condition_id, selection)[1])
                        * LINKS
                        * STEPS
                        * (
                            51
                            if mode_id == "TRUE_FUTURE_OBSERVABILITY_CLOUD"
                            else 1
                        )
                    )
                    for mode_id in (
                        ("TRUE_FUTURE_OBSERVABILITY_CLOUD",)
                        if condition_id == DIAGNOSTIC_CONDITION
                        else MODES
                    )
                }
                for condition_id in condition_ids
                if "REALISTIC" not in condition_id
            },
            "audit_selection_by_role": audit_by_role,
            "state_records": [
                {
                    **{
                        key: row[key]
                        for key in (
                            "state_id",
                            "scene_id",
                            "family",
                            "role",
                            "transitions",
                            "action_representatives",
                            "geometry_representatives",
                            "condition_records",
                            "runtime_s",
                        )
                    },
                    "state_receipt_path": str(
                        OUTPUT_ROOT
                        / "state_receipts"
                        / phase
                        / f"{row['state_id']}.json"
                    ),
                    "state_receipt_sha256": sha256_file(
                        OUTPUT_ROOT
                        / "state_receipts"
                        / phase
                        / f"{row['state_id']}.json"
                    ),
                    "state_receipt_bytes": (
                        OUTPUT_ROOT
                        / "state_receipts"
                        / phase
                        / f"{row['state_id']}.json"
                    ).stat().st_size,
                }
                for row in records
            ],
            "records": [
                {
                    "state_id": row["state_id"],
                    "scene_id": row["scene_id"],
                    "family": row["family"],
                    "role": row["role"],
                    "condition_id": condition["condition_id"],
                    "transitions": row["transitions"],
                    "shard_path": condition["path"],
                    "shard_sha256": condition["sha256"],
                    "shard_bytes": condition["bytes"],
                }
                for row in records
                for condition in row["condition_records"]
            ],
            "worker_count": worker_count,
            "runtime_s": time.time() - started,
            "storage_bytes": _directory_allocated_bytes(OUTPUT_ROOT),
            "simulator_steps": 0,
            "training_steps": 0,
            "fresh_panel_rows": 0,
            "jepa_predictor_opens": 0,
        }
        for key, expected in (
            ("states", EXPECTED["states"]),
            ("transitions", EXPECTED["transitions"]),
            ("action_representatives", EXPECTED["action_representatives"]),
            ("geometry_representatives", EXPECTED["geometry_representatives"]),
        ):
            if index[key] != expected:
                raise RuntimeError(f"{phase} materialization {key} drift")
        for record in index["records"]:
            _validate_condition_npz_schema(record, selection)
        _validate_uid_bound_scan_counts(index["scan_counts"], condition_ids, selection)
        index["content_digest"] = content_digest(index)
        atomic_json(index_path, index)
        return index
    finally:
        marker.unlink(missing_ok=True)


def materialize() -> dict[str, Any]:
    return materialize_conditions(tuple(CONTRACT.DUAL_CONDITION_IDS))


def _condition_state_path(condition_id: str, state_id: str) -> Path:
    return OUTPUT_ROOT / "states" / condition_id / f"{state_id}.npz"


def _load_condition_records(
    context: Any,
    *,
    condition_id: str,
    mode_id: str,
    roles: tuple[str, ...],
) -> list[dict[str, Any]]:
    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    mode_index = MODES.index(mode_id)
    output: list[dict[str, Any]] = []
    for role in roles:
        for state_id in context.state_ids_for_role(role):
            identities = BASE._identity_rows_with_oracle(context, state_id)
            condition_path = _condition_state_path(condition_id, state_id)
            predecessor_path = PREDECESSOR_OUTPUT / "states" / f"{state_id}.npz"
            if not condition_path.is_file():
                raise RuntimeError(f"condition state shard missing: {condition_path}")
            with np.load(condition_path, allow_pickle=False) as archive:
                clearance = np.asarray(
                    archive["clearance_m"][:, mode_index], np.float64
                )
                support = np.asarray(archive["support"][:, mode_index], bool)
                nominal = np.asarray(archive["nominal_fov"][:, mode_index], bool)
                direct = np.asarray(
                    archive["direct_visibility"][:, mode_index], bool
                )
                point_count = np.asarray(
                    archive["point_support_count"][:, mode_index], np.float64
                )
            with np.load(predecessor_path, allow_pickle=False) as predecessor:
                oracle_link = np.asarray(
                    predecessor["oracle_contact_link"], np.int16
                )
                oracle_step = np.asarray(
                    predecessor["oracle_contact_step"], np.int16
                )
            unsupported = ~np.all(support, axis=(1, 2))
            global_clearance = np.min(clearance, axis=(1, 2))
            link_contact = np.zeros(clearance.shape, bool)
            for transition, (step, link) in enumerate(
                zip(oracle_step, oracle_link, strict=True)
            ):
                if step >= 0 and link >= 0:
                    link_contact[transition, int(step), int(link)] = True
            records = analysis.transition_records_from_arrays(
                identities,
                clearance_m=global_clearance,
                unsupported=unsupported,
                link_names=context.protected_link_names(state_id),
                per_link_clearance_m=clearance,
                per_link_support=support,
                per_link_contact=link_contact,
                per_link_nominal_fov=nominal,
                per_link_direct_visibility=direct,
                per_link_point_support_count=point_count,
                body_region_by_link=corpus_adapter.BODY_REGION_BY_LINK,
            )
            output.extend(records)
    return output


def _persist_threshold_frontier(
    phase: str,
    condition_id: str,
    mode_id: str,
    frontier: dict[str, Any],
) -> dict[str, Any]:
    relative = Path("calibration") / phase / f"{condition_id}__{mode_id}.json.gz"
    path = OUTPUT_ROOT / relative
    payload = canonical_bytes(frontier)
    BASE.atomic_bytes(path, gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "condition_id": condition_id,
        "evidence_mode": mode_id,
        "selected_threshold_m": float(frontier["selected_threshold_m"]),
        "frontier_points": int(frontier["frontier_points"]),
        "eligible_points": int(frontier["eligible_points"]),
        "selection_key": list(frontier["selection_key"]),
        "selected": frontier["selected"],
        "artifact_relative_path": str(relative),
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "uncompressed_content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def analyze_condition_phase(
    condition_ids: tuple[str, ...],
    *,
    modes_by_condition: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    phase = _phase_name(condition_ids)
    result_path = OUTPUT_ROOT / "evaluation" / f"{phase}_metrics.json"
    if result_path.is_file():
        receipt = json.loads(result_path.read_text())
        core = dict(receipt)
        declared = core.pop("content_digest", None)
        freeze_path = OUTPUT_ROOT / "calibration" / f"{phase}_thresholds_frozen.json"
        if (
            declared != content_digest(core)
            or receipt.get("pass") is not True
            or tuple(receipt.get("conditions", ())) != condition_ids
            or not freeze_path.is_file()
            or receipt.get("threshold_freeze_sha256") != sha256_file(freeze_path)
        ):
            raise RuntimeError(f"existing {phase} evaluation receipt is invalid")
        return receipt
    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    context = corpus_adapter.load_corpus_context(ROOT)
    thresholds: dict[str, Any] = {}
    allowed_modes = {
        condition_id: (
            modes_by_condition[condition_id]
            if modes_by_condition is not None
            else MODES
        )
        for condition_id in condition_ids
    }
    for condition_id in condition_ids:
        thresholds[condition_id] = {}
        for mode_id in allowed_modes[condition_id]:
            calibration = _load_condition_records(
                context,
                condition_id=condition_id,
                mode_id=mode_id,
                roles=("calibration",),
            )
            frontier = analysis.calibrate_threshold(calibration, role="calibration")
            thresholds[condition_id][mode_id] = _persist_threshold_frontier(
                phase, condition_id, mode_id, frontier
            )
    phase_index = OUTPUT_ROOT / "materialization" / f"{phase}_index.json"
    freeze = {
        "schema": "minimum_multi_origin_threshold_freeze_phase_v1",
        "phase": phase,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_freeze_commit": git("rev-parse", "HEAD"),
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "materialization_index_sha256": sha256_file(phase_index),
        "calibration_role": "internal_calibration",
        "calibration_state_count": EXPECTED["calibration_states"],
        "heldout_rows_used_by_selection": 0,
        "thresholds": thresholds,
        "pass": True,
    }
    freeze["content_digest"] = content_digest(freeze)
    freeze_path = OUTPUT_ROOT / "calibration" / f"{phase}_thresholds_frozen.json"
    atomic_json(freeze_path, freeze)
    # Development-held-out data is loaded only after every threshold in this
    # phase is atomically persisted and its materialization binding verified.
    reloaded = json.loads(freeze_path.read_text())
    freeze_core = dict(reloaded)
    declared = freeze_core.pop("content_digest", None)
    if declared != content_digest(freeze_core) or not reloaded["pass"]:
        raise RuntimeError("phase threshold-freeze receipt failed validation")
    if reloaded["materialization_index_sha256"] != sha256_file(phase_index):
        raise RuntimeError("phase threshold/materialization binding drift")
    condition_metrics: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    expected_progress = float(
        json.loads(PREDECESSOR_RESULT.read_text())["condition_metrics"]
        ["REALISTIC_PLATFORM_SCAN"]
        ["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
        ["viability"]
        ["oracle_h3_route_progress_m"]
    )
    for condition_id in condition_ids:
        condition_metrics[condition_id] = {}
        for mode_id in allowed_modes[condition_id]:
            heldout = _load_condition_records(
                context,
                condition_id=condition_id,
                mode_id=mode_id,
                roles=("heldout",),
            )
            evaluation = analysis.summarize_condition_mode(
                heldout,
                None,
                float(thresholds[condition_id][mode_id]["selected_threshold_m"]),
                role="heldout",
            )
            if not math.isclose(
                float(evaluation["viability"]["oracle_h3_route_progress_m"]),
                expected_progress,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("exact-geometry held-out H3 authority drift")
            condition_metrics[condition_id][mode_id] = evaluation
        future = condition_metrics[condition_id]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
        gates[condition_id] = metrics.evaluate_true_future_gate(future)
    receipt = {
        "schema": "minimum_multi_origin_condition_phase_evaluation_v1",
        "phase": phase,
        "conditions": list(condition_ids),
        "threshold_freeze_sha256": sha256_file(freeze_path),
        "thresholds": thresholds,
        "condition_metrics": condition_metrics,
        "true_future_gates": gates,
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(result_path, receipt)
    return receipt


def _predecessor_regression_payload() -> dict[str, Any]:
    result = _assert_predecessor_authority()
    receipt = {
        "schema": "minimum_multi_origin_single_origin_regression_v1",
        "experiment_id": EXPERIMENT,
        "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
        "predecessor_result_content_sha256": result["result_content_sha256"],
        "predecessor_result_file_sha256": sha256_file(PREDECESSOR_RESULT),
        "condition_ids": list(REGRESSION_CONDITIONS),
        "evidence_modes": list(MODES),
        "source_result_path": str(PREDECESSOR_RESULT),
        "source_result_sha256": sha256_file(PREDECESSOR_RESULT),
        "source_result_content_digest": result["result_content_sha256"],
        "condition_metrics": {
            condition: result["condition_metrics"][condition]
            for condition in REGRESSION_CONDITIONS
        },
        "calibration_thresholds": {
            condition: result["calibration_thresholds"][condition]
            for condition in REGRESSION_CONDITIONS
        },
        "referenced_thresholds": {
            condition: result["calibration_thresholds"][condition]
            for condition in REGRESSION_CONDITIONS
        },
        "gate_results": {
            "true_future": {
                condition: result["gate_results"]["true_future"][condition]
                for condition in REGRESSION_CONDITIONS
            },
            "planning_time_diagnostic": {
                condition: result["gate_results"]["planning_time_diagnostic"][condition]
                for condition in REGRESSION_CONDITIONS
            },
        },
        "referenced_metrics": {
            condition: result["condition_metrics"][condition]
            for condition in REGRESSION_CONDITIONS
        },
        "row_evidence_reused_by_reference": True,
        "rematerialized_rows": 0,
        "reinterpretation": False,
        "reused_without_rematerialization": True,
        "source_compatibility_validated": True,
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    return receipt


def _gzip_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    raw = temporary.open("wb")
    stream = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0)
    return temporary, raw, stream


def _finalize_gzip_jsonl(
    path: Path, temporary: Path, raw: Any, stream: Any
) -> None:
    stream.close()
    raw.close()
    os.replace(temporary, path)


def _decision_lookup(condition_metrics: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for condition_id, mode_rows in condition_metrics.items():
        for mode_id, metrics in mode_rows.items():
            for row in metrics.get("per_state_decisions", []):
                output[(condition_id, mode_id, str(row["state_id"]))] = row
    return output


def _origin_names(mask: int, mounts: tuple[str, ...]) -> list[str]:
    return [mount for index, mount in enumerate(mounts) if int(mask) & (1 << index)]


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def persist_row_level_evidence(
    executed_conditions: tuple[str, ...],
    thresholds: dict[str, Any],
    condition_metrics: dict[str, Any],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    selection = json.loads(
        (OUTPUT_ROOT / "layout_selection/selected_layouts.json").read_text()
    )
    decisions = _decision_lookup(condition_metrics)
    transition_path = OUTPUT_ROOT / "evidence/transition_evidence.jsonl.gz"
    link_path = OUTPUT_ROOT / "evidence/per_link_evidence.jsonl.gz"
    transition_tmp, transition_raw, transition_stream = _gzip_jsonl_writer(
        transition_path
    )
    link_tmp, link_raw, link_stream = _gzip_jsonl_writer(link_path)
    transition_rows = 0
    link_rows = 0
    try:
        for state_id in context.state_ids:
            identities = [dict(row) for row in context.transition_identity_rows(state_id)]
            predecessor_path = PREDECESSOR_OUTPUT / "states" / f"{state_id}.npz"
            with np.load(predecessor_path, allow_pickle=False) as predecessor:
                oracle_contact_link = np.asarray(
                    predecessor["oracle_contact_link"], np.int16
                )
                oracle_contact_step = np.asarray(
                    predecessor["oracle_contact_step"], np.int16
                )
                oracle_clearance = np.asarray(
                    predecessor["oracle_clearance_m"], np.float64
                )
            link_names = tuple(context.protected_link_names(state_id))
            for condition_id in executed_conditions:
                layout_id, mounts, _kind = _condition_spec(condition_id, selection)
                with np.load(
                    _condition_state_path(condition_id, state_id), allow_pickle=False
                ) as archive:
                    action_rep = np.asarray(
                        archive["action_representative_transition"], np.int32
                    )
                    geometry_rep = np.asarray(
                        archive["geometry_representative_transition"], np.int32
                    )
                    for mode_index, mode_id in enumerate(MODES):
                        if mode_id not in condition_metrics[condition_id]:
                            continue
                        threshold = float(
                            thresholds[condition_id][mode_id]["selected_threshold_m"]
                        )
                        clearance = np.asarray(
                            archive["clearance_m"][:, mode_index], np.float64
                        )
                        support = np.asarray(
                            archive["support"][:, mode_index], bool
                        )
                        self_occluded = np.asarray(
                            archive["self_occluded"][:, mode_index], bool
                        )
                        origin_mask = np.asarray(
                            archive["supporting_origin_bitmask"][:, mode_index],
                            np.uint8,
                        )
                        responsible_origin = np.asarray(
                            archive["responsible_origin_index"][:, mode_index],
                            np.int8,
                        )
                        direction = np.asarray(
                            archive["obstacle_direction_body_rad"][:, mode_index],
                            np.float64,
                        )
                        nearest_ray = np.asarray(
                            archive["nearest_ray_index"][:, mode_index], np.int32
                        )
                        point_age = np.asarray(
                            archive["point_age_s"][:, mode_index], np.float64
                        )
                        responsible_object = np.asarray(
                            archive["responsible_object_index"][:, mode_index],
                            np.int16,
                        )
                        per_origin_nominal = np.asarray(
                            archive["per_origin_nominal_fov"][:, mode_index], bool
                        )
                        per_origin_direct = np.asarray(
                            archive["per_origin_direct_visibility"][:, mode_index], bool
                        )
                        per_origin_point_count = np.asarray(
                            archive["per_origin_point_support_count"][:, mode_index],
                            np.int16,
                        )
                        per_origin_acquisition = np.asarray(
                            archive["per_origin_support_acquisition_index"][:, mode_index],
                            np.int16,
                        )
                        per_origin_ray = np.asarray(
                            archive["per_origin_nearest_ray_index"][:, mode_index],
                            np.int32,
                        )
                        per_origin_age = np.asarray(
                            archive["per_origin_point_age_s"][:, mode_index],
                            np.float64,
                        )
                        per_origin_support = np.asarray(
                            archive["per_origin_support"][:, mode_index], bool
                        )
                        per_origin_self = np.asarray(
                            archive["per_origin_self_occluded"][:, mode_index], bool
                        )
                        per_origin_inherited = np.asarray(
                            archive[
                                "per_origin_finite_scan_support_inherited"
                            ][:, mode_index],
                            bool,
                        )
                        fused_inherited = np.asarray(
                            archive["finite_scan_support_inherited"][:, mode_index],
                            bool,
                        )
                        global_clearance = np.min(clearance, axis=(1, 2))
                        unsupported = ~np.all(support, axis=(1, 2))
                        predicted = unsupported | (global_clearance <= threshold)
                        state_decision = decisions.get(
                            (condition_id, mode_id, state_id)
                        )
                        for transition, identity in enumerate(identities):
                            representative = int(action_rep[transition])
                            representative_action = int(
                                identities[representative]["action_index"]
                            )
                            decision_fields: dict[str, Any] = {}
                            if (
                                state_decision is not None
                                and identity["level"] == "current"
                            ):
                                prediction_counts = state_decision.get(
                                    "predicted_safe_counts", {}
                                )
                                key: int | str = representative_action
                                if key not in prediction_counts:
                                    key = str(representative_action)
                                successor_available = key in state_decision.get(
                                    "predicted_safe_counts", {}
                                )
                                decision_fields = {
                                    "successor_set_available": successor_available,
                                    "predicted_safe_next_action_count": (
                                        int(
                                            state_decision["predicted_safe_counts"][key]
                                        )
                                        if successor_available
                                        else None
                                    ),
                                    "oracle_safe_next_action_count": (
                                        int(state_decision["true_safe_counts"][key])
                                        if successor_available
                                        else None
                                    ),
                                    "admitted": (
                                        bool(state_decision["admitted"][key])
                                        if successor_available
                                        else None
                                    ),
                                    "selected": bool(
                                        state_decision.get("selected")
                                        == representative_action
                                    ),
                                }
                            if not decision_fields:
                                decision_fields = {
                                    "successor_set_available": None,
                                    "predicted_safe_next_action_count": None,
                                    "oracle_safe_next_action_count": None,
                                    "admitted": None,
                                    "selected": None,
                                }
                            row = {
                                "transition_uid": str(identity["identity"]),
                                "state_id": state_id,
                                "family": str(identity["family"]),
                                "role": str(identity["role"]),
                                "transition_level": str(identity["level"]),
                                "transition_index": transition,
                                "current_action_index": int(
                                    identity["current_action_index"]
                                ),
                                "action_index": int(identity["action_index"]),
                                "action_representative_transition_index": representative,
                                "geometry_representative_transition_index": int(
                                    geometry_rep[transition]
                                ),
                                "condition_id": condition_id,
                                "evidence_mode": mode_id,
                                "layout_id": layout_id,
                                "mount_ids": list(mounts),
                                "origin_mount_ids": list(mounts),
                                "threshold_m": threshold,
                                "sensor_global_minimum_clearance_m": _finite_or_none(
                                    global_clearance[transition]
                                ),
                                "unsupported_protected_sweep": bool(
                                    unsupported[transition]
                                ),
                                "unsupported_risk": bool(unsupported[transition]),
                                "predicted_contact": bool(predicted[transition]),
                                "oracle_contact": bool(identity["frozen_contact"]),
                                "per_link_row_count": LINKS,
                                **decision_fields,
                            }
                            transition_stream.write(canonical_bytes(row))
                            transition_rows += 1
                            for link, link_name in enumerate(link_names):
                                link_clearance = clearance[transition, :, link]
                                link_support = support[transition, :, link]
                                finite_supported = link_support & np.isfinite(
                                    link_clearance
                                )
                                first_crossing = np.flatnonzero(
                                    finite_supported
                                    & (link_clearance <= threshold)
                                )
                                unsupported_steps = np.flatnonzero(~link_support)
                                if finite_supported.any():
                                    finite = np.where(
                                        finite_supported,
                                        link_clearance,
                                        np.inf,
                                    )
                                    minimum_step: int | None = int(np.argmin(finite))
                                    minimum: float | None = float(finite[minimum_step])
                                else:
                                    minimum_step = None
                                    minimum = None
                                union_mask = int(
                                    np.bitwise_or.reduce(
                                        origin_mask[transition, :, link], initial=0
                                    )
                                )
                                responsible_index = (
                                    int(
                                        responsible_origin[
                                            transition, minimum_step, link
                                        ]
                                    )
                                    if minimum_step is not None
                                    else -1
                                )
                                collision_region = (
                                    "TRUNK"
                                    if link_name == "base"
                                    else (
                                        "HIP"
                                        if link_name.endswith("_hip")
                                        else (
                                            "THIGH"
                                            if link_name.endswith("_thigh")
                                            else "CALF"
                                        )
                                    )
                                )
                                crossing_step = (
                                    int(first_crossing[0])
                                    if len(first_crossing)
                                    else None
                                )
                                origin_nominal = [
                                    bool(per_origin_nominal[transition, origin, :, link].any())
                                    for origin in range(len(mounts))
                                ]
                                origin_direct = [
                                    bool(per_origin_direct[transition, origin, :, link].any())
                                    for origin in range(len(mounts))
                                ]
                                origin_counts = [
                                    int(per_origin_point_count[transition, origin, :, link].sum())
                                    for origin in range(len(mounts))
                                ]
                                origin_rays = [
                                    (
                                        int(per_origin_ray[transition, origin, minimum_step, link])
                                        if minimum_step is not None
                                        and per_origin_ray[transition, origin, minimum_step, link] >= 0
                                        else None
                                    )
                                    for origin in range(len(mounts))
                                ]
                                origin_ages = [
                                    _finite_or_none(
                                        per_origin_age[
                                            transition, origin, minimum_step, link
                                        ]
                                    )
                                    if minimum_step is not None
                                    else None
                                    for origin in range(len(mounts))
                                ]
                                origin_acquisition_times = [
                                    (
                                        _finite_or_none(
                                            0.002 * (minimum_step + 1) - age
                                        )
                                        if age is not None
                                        else None
                                    )
                                    for age in origin_ages
                                ]
                                link_row = {
                                    "transition_uid": str(identity["identity"]),
                                    "state_id": state_id,
                                    "family": str(identity["family"]),
                                    "role": str(identity["role"]),
                                    "transition_level": str(identity["level"]),
                                    "transition_index": transition,
                                    "current_action_index": int(
                                        identity["current_action_index"]
                                    ),
                                    "action_index": int(identity["action_index"]),
                                    "action_representative_transition_index": representative,
                                    "geometry_representative_transition_index": int(
                                        geometry_rep[transition]
                                    ),
                                    "condition_id": condition_id,
                                    "evidence_mode": mode_id,
                                    "layout_id": layout_id,
                                    "link_index": link,
                                    "link_name": link_name,
                                    "protected_link": link_name,
                                    "collision_region": collision_region,
                                    "minimum_observed_clearance_m": _finite_or_none(
                                        minimum
                                    ) if minimum is not None else None,
                                    "minimum_observed_environment_clearance_m": _finite_or_none(
                                        minimum
                                    ) if minimum is not None else None,
                                    "clearance_observation_status": (
                                        "OBSERVED_FINITE"
                                        if minimum_step is not None
                                        else "UNSUPPORTED_NO_FINITE_CLEARANCE"
                                    ),
                                    "time_to_minimum_clearance_s": (
                                        0.002 * (minimum_step + 1)
                                        if minimum_step is not None
                                        else None
                                    ),
                                    "first_threshold_crossing_step": (
                                        crossing_step
                                    ),
                                    "first_threshold_crossing_time_s": (
                                        0.002 * (crossing_step + 1)
                                        if crossing_step is not None
                                        else None
                                    ),
                                    "unsupported_risk_first_step": (
                                        int(unsupported_steps[0])
                                        if len(unsupported_steps)
                                        else None
                                    ),
                                    "unsupported_risk_first_time_s": (
                                        0.002 * (int(unsupported_steps[0]) + 1)
                                        if len(unsupported_steps)
                                        else None
                                    ),
                                    "obstacle_direction_body_rad": _finite_or_none(
                                        direction[transition, minimum_step, link]
                                    ) if minimum_step is not None else None,
                                    "obstacle_sector": (
                                        BASE._obstacle_sector(
                                            direction[transition, minimum_step, link]
                                        )
                                        if minimum_step is not None
                                        else None
                                    ),
                                    "supporting_origin_bitmask": union_mask,
                                    "supporting_origins": _origin_names(
                                        union_mask, mounts
                                    ),
                                    "supporting_origin_ids": _origin_names(
                                        union_mask, mounts
                                    ),
                                    "responsible_origin": (
                                        mounts[responsible_index]
                                        if 0 <= responsible_index < len(mounts)
                                        else None
                                    ),
                                    "responsible_origin_id": (
                                        mounts[responsible_index]
                                        if 0 <= responsible_index < len(mounts)
                                        else None
                                    ),
                                    "observation_support": bool(
                                        link_support.all()
                                    ),
                                    "unsupported_swept_volume_fraction": float(
                                        1.0 - link_support.mean()
                                    ),
                                    "robot_self_occlusion_fraction": float(
                                        self_occluded[transition, :, link].mean()
                                    ),
                                    "nearest_sampled_ray": (
                                        int(
                                            nearest_ray[
                                                transition, minimum_step, link
                                            ]
                                        )
                                        if minimum_step is not None
                                        and nearest_ray[
                                            transition, minimum_step, link
                                        ]
                                        >= 0
                                        else None
                                    ),
                                    "point_age_s": _finite_or_none(
                                        point_age[
                                            transition, minimum_step, link
                                        ]
                                    ) if minimum_step is not None else None,
                                    "responsible_environment_object_index": (
                                        int(
                                            responsible_object[
                                                transition, minimum_step, link
                                            ]
                                        )
                                        if minimum_step is not None
                                        and responsible_object[
                                            transition, minimum_step, link
                                        ]
                                        >= 0
                                        else None
                                    ),
                                    "responsible_environment_object": (
                                        int(
                                            responsible_object[
                                                transition, minimum_step, link
                                            ]
                                        )
                                        if minimum_step is not None
                                        and responsible_object[
                                            transition, minimum_step, link
                                        ]
                                        >= 0
                                        else None
                                    ),
                                    "nominal_fov_inclusion_by_origin": dict(
                                        zip(mounts, origin_nominal, strict=True)
                                    ),
                                    "direct_visibility_by_origin": dict(
                                        zip(mounts, origin_direct, strict=True)
                                    ),
                                    "point_support_count_by_origin": dict(
                                        zip(mounts, origin_counts, strict=True)
                                    ),
                                    "support_acquisition_times_s_by_origin": dict(
                                        zip(mounts, origin_acquisition_times, strict=True)
                                    ),
                                    "nearest_ray_or_point_by_origin": dict(
                                        zip(mounts, origin_rays, strict=True)
                                    ),
                                    "point_age_s_by_origin": dict(
                                        zip(mounts, origin_ages, strict=True)
                                    ),
                                    "observation_support_by_origin": {
                                        mount: bool(
                                            per_origin_support[
                                                transition, origin, :, link
                                            ].all()
                                        )
                                        for origin, mount in enumerate(mounts)
                                    },
                                    "self_occlusion_by_origin": {
                                        mount: float(
                                            per_origin_self[
                                                transition, origin, :, link
                                            ].mean()
                                        )
                                        for origin, mount in enumerate(mounts)
                                    },
                                    "finite_scan_support_inherited_by_origin": {
                                        mount: int(
                                            per_origin_inherited[
                                                transition, origin, :, link
                                            ].sum()
                                        )
                                        for origin, mount in enumerate(mounts)
                                    },
                                    "finite_scan_support_inherited_count": int(
                                        fused_inherited[transition, :, link].sum()
                                    ),
                                    "responsible_acquisition_time_s": (
                                        origin_acquisition_times[responsible_index]
                                        if 0 <= responsible_index < len(mounts)
                                        else None
                                    ),
                                    "oracle_contact_link": bool(
                                        oracle_contact_link[transition] == link
                                    ),
                                    "oracle_contact_step": (
                                        int(oracle_contact_step[transition])
                                        if oracle_contact_step[transition] >= 0
                                        else None
                                    ),
                                    "exact_oracle_minimum_clearance_m": _finite_or_none(
                                        np.min(oracle_clearance[transition, :, link])
                                    ),
                                }
                                link_stream.write(canonical_bytes(link_row))
                                link_rows += 1
        _finalize_gzip_jsonl(
            transition_path, transition_tmp, transition_raw, transition_stream
        )
        _finalize_gzip_jsonl(link_path, link_tmp, link_raw, link_stream)
    except BaseException:
        transition_stream.close()
        transition_raw.close()
        link_stream.close()
        link_raw.close()
        transition_tmp.unlink(missing_ok=True)
        link_tmp.unlink(missing_ok=True)
        raise
    expected_transition_rows = EXPECTED["transitions"] * sum(
        len(condition_metrics[condition]) for condition in executed_conditions
    )
    if transition_rows != expected_transition_rows or link_rows != transition_rows * LINKS:
        raise RuntimeError("row-level evidence cardinality mismatch")
    receipt = {
        "schema": "minimum_multi_origin_row_level_evidence_v1",
        "transition_evidence": {
            "path": str(transition_path),
            "sha256": sha256_file(transition_path),
            "bytes": transition_path.stat().st_size,
            "rows": transition_rows,
            "compression": "gzip-mtime-zero",
        },
        "per_link_evidence": {
            "path": str(link_path),
            "sha256": sha256_file(link_path),
            "bytes": link_path.stat().st_size,
            "rows": link_rows,
            "compression": "gzip-mtime-zero",
        },
        "regression_evidence_authority": str(
            PREDECESSOR_OUTPUT / "persistence_receipt.json"
        ),
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(OUTPUT_ROOT / "evidence/row_level_evidence_receipt.json", receipt)
    return receipt


def _report_region_masks(link_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        "trunk": np.asarray([name == "base" for name in link_names], bool),
        "front_left_limb": np.asarray([name.startswith("FL_") for name in link_names], bool),
        "front_right_limb": np.asarray([name.startswith("FR_") for name in link_names], bool),
        "rear_left_limb": np.asarray([name.startswith("RL_") for name in link_names], bool),
        "rear_right_limb": np.asarray([name.startswith("RR_") for name in link_names], bool),
        "hips": np.asarray([name.endswith("_hip") for name in link_names], bool),
        "thighs": np.asarray([name.endswith("_thigh") for name in link_names], bool),
        "calves": np.asarray([name.endswith("_calf") for name in link_names], bool),
    }


def coverage_support_summary(
    executed_conditions: tuple[str, ...],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter

    context = corpus_adapter.load_corpus_context(ROOT)
    state_ids = tuple(context.state_ids_for_role("heldout"))
    link_names = tuple(context.protected_link_names(state_ids[0]))
    region_masks = _report_region_masks(link_names)
    selection = json.loads(
        (OUTPUT_ROOT / "layout_selection/selected_layouts.json").read_text()
    )
    output: dict[str, Any] = {}
    for condition_id in executed_conditions:
        _layout_id, mount_ids, _kind = _condition_spec(condition_id, selection)
        region_counts = {
            region: {"supported": 0, "total": 0, "self_occluded": 0}
            for region in region_masks
        }
        link_counts = {
            name: {"supported": 0, "total": 0, "self_occluded": 0}
            for name in link_names
        }
        family_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"supported": 0, "total": 0, "self_occluded": 0}
        )
        origin_counts = {
            mount_id: {
                "supported": 0,
                "event_time_supported": 0,
                "nominal": 0,
                "direct": 0,
                "self_occluded": 0,
                "total": 0,
            }
            for mount_id in mount_ids
        }
        for state_id in state_ids:
            family = str(context.geometry_record(state_id)["family"])
            with np.load(
                _condition_state_path(condition_id, state_id), allow_pickle=False
            ) as archive:
                mode = MODES.index("TRUE_FUTURE_OBSERVABILITY_CLOUD")
                support = np.asarray(archive["support"][:, mode], bool)
                self_occluded = np.asarray(
                    archive["self_occluded"][:, mode], bool
                )
                per_origin_support = np.asarray(
                    archive["per_origin_support"][:, mode], bool
                )
                per_origin_event = np.asarray(
                    archive["per_origin_event_time_support"][:, mode], bool
                )
                per_origin_nominal = np.asarray(
                    archive["per_origin_nominal_fov"][:, mode], bool
                )
                per_origin_direct = np.asarray(
                    archive["per_origin_direct_visibility"][:, mode], bool
                )
                per_origin_self = np.asarray(
                    archive["per_origin_self_occluded"][:, mode], bool
                )
            if per_origin_support.shape[1] != len(mount_ids):
                raise RuntimeError(
                    f"{condition_id}/{state_id}: per-origin cardinality drift"
                )
            for origin, mount_id in enumerate(mount_ids):
                counts = origin_counts[mount_id]
                counts["supported"] += int(per_origin_support[:, origin].sum())
                counts["event_time_supported"] += int(
                    per_origin_event[:, origin].sum()
                )
                counts["nominal"] += int(per_origin_nominal[:, origin].sum())
                counts["direct"] += int(per_origin_direct[:, origin].sum())
                counts["self_occluded"] += int(per_origin_self[:, origin].sum())
                counts["total"] += int(per_origin_support[:, origin].size)
            family_counts[family]["supported"] += int(support.sum())
            family_counts[family]["total"] += int(support.size)
            family_counts[family]["self_occluded"] += int(self_occluded.sum())
            for link, name in enumerate(link_names):
                link_counts[name]["supported"] += int(support[:, :, link].sum())
                link_counts[name]["total"] += int(support[:, :, link].size)
                link_counts[name]["self_occluded"] += int(
                    self_occluded[:, :, link].sum()
                )
            for region, mask in region_masks.items():
                region_counts[region]["supported"] += int(support[:, :, mask].sum())
                region_counts[region]["total"] += int(support[:, :, mask].size)
                region_counts[region]["self_occluded"] += int(
                    self_occluded[:, :, mask].sum()
                )

        def normalize(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
            return {
                key: {
                    **value,
                    "support_fraction": value["supported"] / value["total"],
                    "unsupported_fraction": 1.0
                    - value["supported"] / value["total"],
                    "self_occluded_fraction": value["self_occluded"]
                    / value["total"],
                }
                for key, value in sorted(rows.items())
            }

        output[condition_id] = {
            "mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
            "per_region": normalize(region_counts),
            "per_link": normalize(link_counts),
            "per_family": normalize(dict(family_counts)),
            "per_origin": {
                mount_id: {
                    **counts,
                    "support_fraction": counts["supported"] / counts["total"],
                    "event_time_support_fraction": counts["event_time_supported"]
                    / counts["total"],
                    "nominal_fraction": counts["nominal"] / counts["total"],
                    "direct_visibility_fraction": counts["direct"]
                    / counts["total"],
                    "self_occluded_fraction_of_all_queries": counts["self_occluded"]
                    / counts["total"],
                    "self_occluded_fraction_of_nominal_queries": (
                        counts["self_occluded"] / counts["nominal"]
                        if counts["nominal"]
                        else 1.0
                    ),
                }
                for mount_id, counts in origin_counts.items()
            },
        }
    return output


def persist_coverage_errors(
    executed_conditions: tuple[str, ...],
    thresholds: dict[str, Any],
    condition_metrics: dict[str, Any],
) -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    context = corpus_adapter.load_corpus_context(ROOT)
    selection = json.loads(
        (OUTPUT_ROOT / "layout_selection/selected_layouts.json").read_text()
    )
    path = OUTPUT_ROOT / "evidence/coverage_errors.jsonl.gz"
    temporary, raw, stream = _gzip_jsonl_writer(path)
    counts = {error_class: 0 for error_class in CONTRACT.COVERAGE_ERROR_CLASSES}
    scope_counts: dict[str, int] = defaultdict(int)
    decision_failure_counts: dict[str, int] = defaultdict(int)
    decisions = _decision_lookup(condition_metrics)
    rows = 0
    try:
        for state_id in context.state_ids_for_role("heldout"):
            identities = [dict(row) for row in context.transition_identity_rows(state_id)]
            predecessor_path = PREDECESSOR_OUTPUT / "states" / f"{state_id}.npz"
            with np.load(predecessor_path, allow_pickle=False) as predecessor:
                contact_step = np.asarray(
                    predecessor["oracle_contact_step"], np.int16
                )
                contact_link = np.asarray(
                    predecessor["oracle_contact_link"], np.int16
                )
                critical_step = np.asarray(predecessor["critical_step"], np.int16)
                critical_link = np.asarray(predecessor["critical_link"], np.int16)
                oracle_clearance = np.asarray(
                    predecessor["oracle_clearance_m"], np.float64
                )
                action_representative = np.asarray(
                    predecessor["action_representative_transition"], np.int32
                )
            archives = {
                condition_id: np.load(
                    _condition_state_path(condition_id, state_id), allow_pickle=False
                )
                for condition_id in executed_conditions
            }
            try:
                for condition_id in executed_conditions:
                    layout_id, mounts, kind = _condition_spec(condition_id, selection)
                    for mode_index, mode_id in enumerate(MODES):
                        if mode_id not in thresholds[condition_id]:
                            continue
                        archive = archives[condition_id]
                        threshold = float(
                            thresholds[condition_id][mode_id]["selected_threshold_m"]
                        )
                        clearance = np.asarray(
                            archive["clearance_m"][:, mode_index], np.float64
                        )
                        support = np.asarray(
                            archive["support"][:, mode_index], bool
                        )
                        global_clearance = np.min(clearance, axis=(1, 2))
                        unsupported = ~np.all(support, axis=(1, 2))
                        labels = np.asarray(
                            [bool(row["frozen_contact"]) for row in identities], bool
                        )
                        predicted = unsupported | (global_clearance <= threshold)

                        def build_error_row(
                            transition: int,
                            *,
                            error_scope: str,
                            decision_failures: Sequence[str] = (),
                        ) -> dict[str, Any]:
                            oracle_positive = bool(
                                identities[int(transition)]["frozen_contact"]
                            )
                            if (
                                oracle_positive
                                and contact_step[transition] >= 0
                                and contact_link[transition] >= 0
                            ):
                                step = int(contact_step[transition])
                                link = int(contact_link[transition])
                            elif unsupported[transition]:
                                witness = np.argwhere(~support[transition])
                                step, link = (int(value) for value in witness[0])
                            else:
                                flat = int(np.argmin(clearance[transition]))
                                step, link = divmod(flat, LINKS)
                            current_support = bool(support[transition, step, link])
                            near = bool(
                                archive["near_blind"][
                                    transition, mode_index, step, link
                                ]
                            )
                            self_blocked = bool(
                                archive["self_occluded"][
                                    transition, mode_index, step, link
                                ]
                            )
                            event_support = bool(
                                archive["event_time_support"]
                                [transition, mode_index, step, link]
                            )
                            sibling_prefix = (
                                "DUAL" if condition_id.startswith("DUAL_") else "THREE"
                            )
                            sphere_id = f"{sibling_prefix}_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"
                            fov_id = f"{sibling_prefix}_DENSE_L2_FOV_UPPER_BOUND"
                            realistic_id = f"{sibling_prefix}_REALISTIC_L2_SCAN"

                            def sibling_support(sibling: str) -> bool:
                                return bool(
                                    sibling in archives
                                    and archives[sibling]["support"]
                                    [transition, mode_index, step, link]
                                )

                            larger_support = False
                            if sibling_prefix == "DUAL":
                                larger_available = (
                                    "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"
                                    in archives
                                )
                                larger_support = bool(
                                    larger_available
                                    and sibling_support(
                                        "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"
                                    )
                                )
                            elif (
                                condition_id != DIAGNOSTIC_CONDITION
                                and DIAGNOSTIC_CONDITION in archives
                            ):
                                larger_available = True
                                larger_support = sibling_support(DIAGNOSTIC_CONDITION)
                            else:
                                larger_available = False
                            flags = {
                                "point_fusion_error": bool(
                                    int(
                                        archive["supporting_origin_bitmask"]
                                        [transition, mode_index, step, link]
                                    )
                                    and not current_support
                                ),
                                "near_blind_region": near,
                                "robot_self_occlusion": self_blocked,
                                "vertical_fov_limitation": bool(
                                    kind != "DENSE_SPHERICAL"
                                    and sphere_id in archives
                                    and sibling_support(sphere_id)
                                    and not sibling_support(fov_id)
                                ),
                                "scan_timing_limitation": bool(
                                    condition_id == realistic_id
                                    and current_support
                                    and not event_support
                                ),
                                "scan_pattern_sparsity": bool(
                                    condition_id == realistic_id
                                    and sibling_support(fov_id)
                                    and not current_support
                                ),
                                "insufficient_origin_count": bool(
                                    kind == "DENSE_SPHERICAL"
                                    and not current_support
                                    and larger_support
                                ),
                                "mount_position_limitation": bool(
                                    kind == "DENSE_SPHERICAL"
                                    and not current_support
                                    and larger_available
                                    and not larger_support
                                ),
                            }
                            classification = metrics.classify_multi_origin_error(flags)
                            error_class = str(classification["error_class"])
                            mask = int(
                                archive["supporting_origin_bitmask"]
                                [transition, mode_index, step, link]
                            )
                            nearest_ray = int(
                                archive["support_nearest_ray_index"]
                                [transition, mode_index, step, link]
                            )
                            age = float(
                                archive["support_point_age_s"]
                                [transition, mode_index, step, link]
                            )
                            mount_visibility: dict[str, Any] = {}
                            mount_self: dict[str, bool] = {}
                            mount_points: dict[str, int] = {}
                            mount_rays: dict[str, int | None] = {}
                            mount_ages: dict[str, float | None] = {}
                            for origin_index, mount_id in enumerate(mounts):
                                origin_support = bool(
                                    archive["per_origin_support"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_nominal = bool(
                                    archive["per_origin_nominal_fov"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_direct = bool(
                                    archive["per_origin_direct_visibility"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_event = bool(
                                    archive["per_origin_event_time_support"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_self = bool(
                                    archive["per_origin_self_occluded"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_point_count = int(
                                    archive["per_origin_point_support_count"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_ray = int(
                                    archive["per_origin_nearest_ray_index"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                origin_age = float(
                                    archive["per_origin_point_age_s"]
                                    [transition, mode_index, origin_index, step, link]
                                )
                                mount_visibility[mount_id] = {
                                    "nominal_fov": origin_nominal,
                                    "direct_visibility": origin_direct,
                                    "observation_support": origin_support,
                                    "event_time_support": origin_event,
                                }
                                mount_self[mount_id] = origin_self
                                mount_points[mount_id] = origin_point_count
                                mount_rays[mount_id] = (
                                    origin_ray if origin_ray >= 0 else None
                                )
                                mount_ages[mount_id] = _finite_or_none(origin_age)
                            link_name = context.protected_link_names(state_id)[link]
                            supported_origins = _origin_names(mask, mounts)
                            row = {
                                "transition_uid": str(
                                    identities[int(transition)]["identity"]
                                ),
                                "state_id": state_id,
                                "family": str(
                                    identities[int(transition)]["family"]
                                ),
                                "transition_index": int(transition),
                                "transition_level": str(
                                    identities[int(transition)]["level"]
                                ),
                                "current_action_index": int(
                                    identities[int(transition)][
                                        "current_action_index"
                                    ]
                                ),
                                "action_index": int(
                                    identities[int(transition)]["action_index"]
                                ),
                                "condition_id": condition_id,
                                "evidence_mode": mode_id,
                                "layout_id": layout_id,
                                "error_scope": error_scope,
                                "decision_failure_types": list(decision_failures),
                                "robot_link": link_name,
                                "robot_link_or_body_region": link_name,
                                "link_index": link,
                                "physics_step": step,
                                "contact_or_minimum_clearance_physics_step": step,
                                "supporting_origins": supported_origins,
                                "supporting_origin_ids": supported_origins,
                                "failed_origins": [
                                    mount
                                    for mount in mounts
                                    if mount not in supported_origins
                                ],
                                "nominal_fov": bool(
                                    archive["nominal_fov"]
                                    [transition, mode_index, step, link]
                                ),
                                "robot_self_occlusion": self_blocked,
                                "nearest_sampled_ray": (
                                    nearest_ray if nearest_ray >= 0 else None
                                ),
                                "point_age_s": _finite_or_none(age),
                                "origin_visibility": mount_visibility,
                                "origin_self_occlusion": mount_self,
                                "origin_scan_point_availability": mount_points,
                                "nearest_ray_or_point_by_origin": mount_rays,
                                "point_age_s_by_origin": mount_ages,
                                "exact_clearance_m": _finite_or_none(
                                    oracle_clearance[transition, step, link]
                                ),
                                "observed_clearance_m": _finite_or_none(
                                    clearance[transition, step, link]
                                ),
                                "sensor_derived_clearance_m": _finite_or_none(
                                    clearance[transition, step, link]
                                ),
                                "oracle_contact": oracle_positive,
                                "predicted_contact": bool(predicted[transition]),
                                "error_class": error_class,
                                "error_attribution": classification,
                            }
                            return row

                        contact_errors = np.flatnonzero(predicted != labels)
                        for transition in contact_errors:
                            row = build_error_row(
                                int(transition), error_scope="CONTACT_CLASSIFICATION"
                            )
                            counts[row["error_class"]] += 1
                            scope_counts[row["error_scope"]] += 1
                            stream.write(canonical_bytes(row))
                            rows += 1

                        state_decision = decisions.get(
                            (condition_id, mode_id, state_id)
                        )
                        if state_decision is None:
                            raise RuntimeError(
                                "heldout decision result missing from error persistence"
                            )
                        failures: list[str] = []
                        for field in (
                            "false_abstention",
                            "unsafe_movement",
                            "selected_immediate_contact",
                            "selected_nonviable_successor",
                        ):
                            if bool(state_decision.get(field)):
                                failures.append(field.upper())
                        if bool(state_decision.get("oracle_viable")) and not bool(
                            state_decision.get("retained")
                        ):
                            failures.append("ORACLE_VIABLE_ACTION_NOT_RETAINED")
                        if int(state_decision.get("falsely_viable_candidates", 0)) > 0:
                            failures.append("FALSELY_VIABLE_CANDIDATES")
                        if failures:
                            preferred_action = state_decision.get("selected")
                            if preferred_action is None:
                                preferred_action = state_decision.get("oracle_selected")
                            candidate_transitions = [
                                index
                                for index, identity in enumerate(identities)
                                if identity["level"] == "current"
                                and int(identity["action_index"])
                                == int(preferred_action)
                                and int(action_representative[index]) == index
                            ] if preferred_action is not None else []
                            cause = (
                                candidate_transitions[0]
                                if candidate_transitions
                                else next(
                                    index
                                    for index, identity in enumerate(identities)
                                    if identity["level"] == "current"
                                    and int(action_representative[index]) == index
                                )
                            )
                            # Prefer the exact transition-level contact error
                            # that caused the set-level decision whenever it is
                            # available: current first, then this action's
                            # successor set.
                            if not bool(predicted[cause] != labels[cause]):
                                related = [
                                    index
                                    for index, identity in enumerate(identities)
                                    if identity["level"] == "successor"
                                    and preferred_action is not None
                                    and int(identity["current_action_index"])
                                    == int(preferred_action)
                                    and bool(predicted[index] != labels[index])
                                ]
                                if related:
                                    cause = related[0]
                            row = build_error_row(
                                int(cause),
                                error_scope="DECISION_LEVEL",
                                decision_failures=failures,
                            )
                            counts[row["error_class"]] += 1
                            scope_counts[row["error_scope"]] += 1
                            for failure in failures:
                                decision_failure_counts[failure] += 1
                            stream.write(canonical_bytes(row))
                            rows += 1
            finally:
                for archive in archives.values():
                    archive.close()
        _finalize_gzip_jsonl(path, temporary, raw, stream)
    except BaseException:
        stream.close()
        raw.close()
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "schema": "minimum_multi_origin_coverage_error_receipt_v1",
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "counts": counts,
        "scope_counts": dict(sorted(scope_counts.items())),
        "decision_failure_counts": dict(sorted(decision_failure_counts.items())),
        "compression": "gzip-mtime-zero",
        "pass": True,
    }
    receipt["content_digest"] = content_digest(receipt)
    atomic_json(OUTPUT_ROOT / "evidence/coverage_error_receipt.json", receipt)
    return receipt


def _benchmark_worker() -> dict[str, Any]:
    from lewm.safety import body_centric_range_coverage_analysis_v1 as analysis
    from lewm.safety import body_centric_range_coverage_corpus_v1 as corpus_adapter
    from lewm.safety import body_centric_range_coverage_metrics_v1 as base_metrics
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    request = json.loads((OUTPUT_ROOT / "benchmark_request.json").read_text())
    condition_id = str(request["condition_id"])
    threshold = float(request["threshold_m"])
    context = corpus_adapter.load_corpus_context(ROOT)
    all_heldout = tuple(context.state_ids_for_role("heldout"))
    by_family: dict[str, str] = {}
    for state_id in all_heldout:
        family = str(context.geometry_record(state_id)["family"])
        by_family.setdefault(family, state_id)
    selected_state_ids = tuple(by_family[key] for key in sorted(by_family))
    entries: list[dict[str, Any]] = []
    mode_index = MODES.index("TRUE_FUTURE_OBSERVABILITY_CLOUD")
    for state_id in selected_state_ids:
        records = _load_condition_records(
            context,
            condition_id=condition_id,
            mode_id="TRUE_FUTURE_OBSERVABILITY_CLOUD",
            roles=("heldout",),
        )
        records = [row for row in records if str(row["state_id"]) == state_id]
        with np.load(
            _condition_state_path(condition_id, state_id), allow_pickle=False
        ) as archive:
            clearance = np.asarray(
                archive["clearance_m"][:, mode_index], np.float32
            )
            support = np.asarray(archive["support"][:, mode_index], bool)
        if len(records) != len(clearance):
            raise RuntimeError("benchmark state record/tensor alignment drift")
        entries.append(
            {
                "state_id": state_id,
                "family": str(context.geometry_record(state_id)["family"]),
                "records": records,
                "clearance": clearance,
                "support": support,
            }
        )

    def timed_reducer(entry: dict[str, Any]) -> dict[str, Any]:
        # The archive holds one materialized clearance/support witness for
        # every physics-step/protected-link query.  Keep the frozen two-stage
        # set reduction explicit inside the timed scope: point/step witnesses
        # -> per-link state -> transition decision scalar.
        point_clearance = np.asarray(entry["clearance"], np.float32)
        point_support = np.asarray(entry["support"], bool)
        per_link_clearance = np.min(point_clearance, axis=1)
        per_link_supported = np.all(point_support, axis=1)
        global_clearance = np.min(per_link_clearance, axis=1)
        unsupported = ~np.all(per_link_supported, axis=1)
        records = [
            {
                **row,
                "clearance_m": float(global_clearance[index]),
                "unsupported": bool(unsupported[index]),
            }
            for index, row in enumerate(entry["records"])
        ]
        templates = analysis.build_two_ply_states(records)
        predicted = analysis.inject_contact_predictions(templates, records, threshold)
        return base_metrics.reduce_two_ply_state(predicted[0])

    benchmark = metrics.benchmark_complete_and_per_state_decisions(
        entries,
        per_state_reducer=timed_reducer,
        warmups=30,
        iterations=1000,
        required_families=tuple(sorted(by_family)),
    )
    benchmark.update(
        {
            "condition_id": condition_id,
            "evidence_mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
            "threshold_m": threshold,
            "representative_state_ids": list(selected_state_ids),
            "representative_state_policy": "first frozen heldout state per family",
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_vram_bytes": 0,
        }
    )
    return benchmark


def launch_benchmark(
    condition_id: str, threshold_m: float
) -> dict[str, Any]:
    request = {
        "condition_id": condition_id,
        "threshold_m": float(threshold_m),
        "source_freeze_commit": git("rev-parse", "HEAD"),
    }
    atomic_json(OUTPUT_ROOT / "benchmark_request.json", request)
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "benchmark-worker"],
        cwd=ROOT,
        env=environment,
        text=True,
    )
    result = json.loads(output)
    atomic_json(OUTPUT_ROOT / "compute_benchmark.json", result)
    return result


def _hardware_feasibility_summary(
    classification: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    primary = str(classification["primary_classification"])
    use_dual = primary.startswith("DUAL_")
    layout_id = (
        str(selection["selected_dual_layout"])
        if use_dual
        else str(selection["selected_three_origin_layout"])
    )
    sensor_count = len(_layout_mounts(layout_id))
    structured_point_bytes = 16
    structured_bytes_per_second = 64_000 * sensor_count * structured_point_bytes
    return {
        "layout_id": layout_id,
        "mount_ids": list(_layout_mounts(layout_id)),
        "sensor_count": sensor_count,
        "assumed_sensor_class": "ASSUMED_GO2_HEAD_LIDAR_L2",
        "contract_class": "APPROXIMATED_REALISTIC_PLATFORM_SCAN",
        "effective_points_per_second_per_sensor": 64_000,
        "aggregate_point_rate_hz": 64_000 * sensor_count,
        "aggregate_effective_points_per_second": 64_000 * sensor_count,
        "ranging_samples_per_second_per_sensor": 128_000,
        "aggregate_ranging_samples_per_second": 128_000 * sensor_count,
        "rays_per_100ms_per_sensor": 6_400,
        "aggregate_rays_per_100ms": 6_400 * sensor_count,
        "mass_g_per_sensor": 230,
        "aggregate_sensor_mass_g": 230 * sensor_count,
        "supplemental_sensor_mass_over_stock_g": 230 * (sensor_count - 1),
        "approximate_payload_duplication_g": 230 * (sensor_count - 1),
        "typical_power_w_per_sensor": 10,
        "aggregate_typical_power_w": 10 * sensor_count,
        "peak_power_w_per_sensor": 13,
        "aggregate_peak_power_w": 13 * sensor_count,
        "housing_envelope_mm_per_sensor": [75, 75, 65],
        "interface": "Ethernet UDP or TTL UART",
        "structured_scan_payload_assumption_bytes_per_effective_point": (
            structured_point_bytes
        ),
        "aggregate_structured_scan_payload_bytes_per_second": (
            structured_bytes_per_second
        ),
        "aggregate_structured_scan_payload_bandwidth_mbps": (
            structured_bytes_per_second * 8 / 1_000_000
        ),
        "wire_bandwidth": "PROTOCOL_AND_PACKET_FORMAT_DEPENDENT",
        "scan_bandwidth_status": (
            "EXPLICIT_16_BYTE_PER_EFFECTIVE_POINT_DEVELOPMENT_PAYLOAD_ASSUMPTION; "
            "EXCLUDES_WIRE_PROTOCOL_OVERHEAD"
        ),
        "cable_constraints": (
            "supplemental rigid-trunk power/data routing, strain relief, ingress "
            "protection, and leg-sweep clearance require mechanical review"
        ),
        "payload_power_bandwidth_cost_approved": False,
        "approved_bom": False,
        "physical_go2_design_status": "DEVELOPMENT_GEOMETRY_ONLY_REQUIRES_FEASIBILITY_REVIEW",
        "source_urls": [
            "https://www.unitree.com/L2/",
            "https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf",
        ],
    }


def _next_decision(primary: str, realistic_pass: bool) -> dict[str, Any]:
    if realistic_pass:
        return {
            "decision": "SPECIFY_MULTI_ORIGIN_PER_LINK_CLEARANCE_PREDICTOR_V1_WITHOUT_TRAINING",
            "architecture": "MULTI_ORIGIN_PER_LINK_CLEARANCE_PREDICTOR_V1",
            "training_authorized_now": False,
            "required_predictions": [
                "per-link minimum clearance during the next tick",
                "time to first clearance violation",
                "obstacle and body sector",
                "supporting sensor origin",
                "observation support",
                "lower confidence bound or uncertainty interval",
            ],
            "inputs": [
                "planning-time multi-origin range observations",
                "articulated embodied state",
                "one-tick candidate action",
                "control history",
            ],
            "deterministic_derivations": [
                "contact",
                "successor viability",
            ],
            "forbidden_scalar_only_outputs": [
                "binary contact",
                "binary nonviability",
                "utility score",
            ],
            "requires_before_training": [
                "final physical sensor choice",
                "final mount design",
                "payload/power/bandwidth review",
                "fresh scene-disjoint claim-bearing panel",
            ],
        }
    if primary.endswith("SCAN_OR_FOV_BOTTLENECK"):
        return {
            "decision": "SPECIFY_RANGE_SCAN_CONTRACT_SUCCESSOR_V1",
            "architecture": "RANGE_SCAN_CONTRACT_SUCCESSOR_V1",
            "training_authorized_now": False,
            "must_address": [
                "vertical field of view",
                "scan density",
                "scan timing",
                "motion compensation",
                "hardware scan pattern",
            ],
        }
    return {
        "decision": "SPECIFY_PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1",
        "architecture": "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1",
        "training_authorized_now": False,
        "retrospective_label_change_authorized": False,
        "review_scope": [
            "operational hard-contact constraints",
            "recoverable contact monitoring",
            "platform body protection",
            "task-performance and recovery requirements",
        ],
    }


def _validate_execution_plan_receipt(
    receipt: dict[str, Any],
    gate_results: dict[str, Any],
    selection: dict[str, Any],
    *,
    expected_source_freeze_commit: str,
) -> None:
    core = dict(receipt)
    declared = core.pop("content_digest", None)
    dual_pass = bool(gate_results["DUAL_REALISTIC_L2_SCAN"]["pass"])
    three_required = not dual_pass
    diagnostic_required = bool(
        three_required
        and not gate_results["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"][
            "pass"
        ]
    )
    expected_phases = ["dual"]
    expected_conditions = list(CONTRACT.DUAL_CONDITION_IDS)
    if three_required:
        expected_phases.append("three")
        expected_conditions.extend(CONTRACT.THREE_CONDITION_IDS)
    if diagnostic_required:
        expected_phases.append("diagnostic")
        expected_conditions.append(DIAGNOSTIC_CONDITION)
    expected_modes = {
        condition_id: (
            ["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
            if condition_id == DIAGNOSTIC_CONDITION
            else list(MODES)
        )
        for condition_id in expected_conditions
    }
    expected_skipped = {
        **{
            condition_id: list(MODES)
            for condition_id in MULTI_CONDITIONS + (DIAGNOSTIC_CONDITION,)
            if condition_id not in expected_conditions
        },
        **(
            {DIAGNOSTIC_CONDITION: ["PLANNING_TIME_CAUSAL_CLOUD"]}
            if diagnostic_required
            else {}
        ),
    }
    if (
        declared != content_digest(core)
        or receipt.get("pass") is not True
        or receipt.get("conditional_flow_valid") is not True
        or receipt.get("source_freeze_commit") != expected_source_freeze_commit
        or receipt.get("selected_dual_layout_id")
        != selection["selected_dual_layout"]
        or receipt.get("selected_three_layout_id")
        != selection["selected_three_origin_layout"]
        or receipt.get("dual_realistic_pass") is not dual_pass
        or receipt.get("dual_realistic_gate_pass") is not dual_pass
        or receipt.get("three_required") is not three_required
        or receipt.get("three_origin_executed") is not three_required
        or receipt.get("three_complete") is not three_required
        or receipt.get("all_four_diagnostic_required") is not diagnostic_required
        or receipt.get("all_four_diagnostic_executed") is not diagnostic_required
        or receipt.get("all_four_diagnostic_complete") is not diagnostic_required
        or receipt.get("phase_order") != expected_phases
        or receipt.get("executed_condition_ids") != expected_conditions
        or receipt.get("executed_condition_modes") != expected_modes
        or receipt.get("skipped_condition_modes") != expected_skipped
        or set(receipt.get("materialization_indices", {})) != set(expected_phases)
        or set(receipt.get("threshold_freezes", {})) != set(expected_phases)
    ):
        raise RuntimeError("conditional execution-plan receipt drift")
    for group in ("materialization_indices", "threshold_freezes"):
        for phase, artifact in receipt[group].items():
            path = Path(artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(
                    f"conditional execution-plan {group}/{phase} artifact drift"
                )


def _render_markdown(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "null" if value is None else f"{float(value):.6f}"

    lines = [
        "# Minimum Multi-Origin Body Range Coverage Qualification V1",
        "",
        f"Source freeze: `{result['source_freeze_commit']}`",
        f"Primary classification: `{result['primary_classification']}`",
        "",
        "This was a development-only, non-claim-bearing, no-training qualification. "
        "True-future observations are a geometric observability upper bound, not "
        "information available before action execution and not evidence that future "
        "geometry is predictable.",
        "",
        "## Frozen scientific authority",
        "",
        "> Exact Genesis per-link geometry reconstructs immediate contact, successor "
        "contact, zero-versus-nonzero safe-action availability and the complete "
        "two-ply viability decision.",
        "",
        "> Under the tested full-body protected-contact scope, one range-sensor "
        "origin does not observe enough of the trunk, hip, thigh and calf swept "
        "volume. Vertical coverage and robot self-occlusion dominate; additional "
        "scan density at the same origin does not solve the problem.",
        "",
        "## Label-free mounts and layouts",
        "",
        f"Selected dual: `{result['layout_selection']['selected_dual_layout']}`",
        f"Selected triple: `{result['layout_selection']['selected_three_origin_layout']}`",
        "",
        "Orientations and every candidate score are frozen in the mount-library and "
        "layout-selection receipts. No contact label, safe-action count, route outcome, "
        "calibration state, or held-out state entered their selection.",
        "",
        "| Mount | Translation body XYZ (m) | RPY body (rad) | Selected orientation |",
        "|---|---|---|---|",
    ]
    for orientation in result["mount_library"]["orientation_selections"]:
        pose = orientation["selected_pose"]
        lines.append(
            f"| {orientation['mount_id']} | `{pose['translation_body_xyz_m']}` | "
            f"`{pose['rpy_body_rad']}` | {orientation['selected_orientation_id']} |"
        )
    lines.extend(
        [
            "",
            "### Training-role label-free layout coverage",
            "",
            "| Layout | Minimum region | Transition P05 | Rear limb | Calf | Mean | Self-occluded |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for layout_id, score in result["layout_selection"]["candidate_metrics"].items():
        lines.append(
            f"| {layout_id} | {fmt(score['minimum_body_region_support'])} | "
            f"{fmt(score['transition_support_p05'])} | {fmt(score['rear_limb_support'])} | "
            f"{fmt(score['calf_support'])} | {fmt(score['overall_mean_support'])} | "
            f"{fmt(score['self_occluded_fraction'])} |"
        )
    lines.extend(
        [
        "",
        "## Held-out condition summary",
        "",
        "| Condition | Mode | Current AUC | Successor AUC | Recall | Safe zero/nonzero | Viable retained | Progress fraction | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_id, mode_rows in result["condition_metrics"].items():
        for mode_id, metrics in mode_rows.items():
            gate = result["gate_results"].get(condition_id, {})
            gate_text = (
                str(bool(gate.get("pass"))).lower()
                if mode_id == "TRUE_FUTURE_OBSERVABILITY_CLOUD"
                else "n/a causal diagnostic"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        condition_id,
                        mode_id,
                        f"{metrics['current_contact']['auc']:.6f}",
                        f"{metrics['successor_contact']['auc']:.6f}",
                        f"{metrics['combined_contact']['recall']:.6f}",
                        f"{metrics['safe_action_count']['zero_vs_nonzero_accuracy']:.6f}",
                        f"{metrics['viability']['states_retaining_admitted_action']}/{metrics['viability']['oracle_viable_states']}",
                        f"{metrics['viability']['oracle_progress_fraction']:.6f}",
                        gate_text,
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "### Complete contact and set-decision metrics",
            "",
            "| Condition / mode | Threshold | Current AP/retention | Successor AP/retention | Count MAE/Spearman/exact/FZ/FNZ | Viable retained | Nonviable abstain | Selected contact/nonviable | Progress/regret/top1/top3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_id, mode_rows in result["condition_metrics"].items():
        for mode_id, metrics in mode_rows.items():
            current = metrics["current_contact"]
            successor = metrics["successor_contact"]
            count = metrics["safe_action_count"]
            viability = metrics["viability"]
            lines.append(
                f"| {condition_id} / {mode_id} | {fmt(metrics['threshold_m'])} | "
                f"{fmt(current['average_precision'])}/{fmt(current['negative_retention'])} | "
                f"{fmt(successor['average_precision'])}/{fmt(successor['negative_retention'])} | "
                f"{fmt(count['mae'])}/{fmt(count['spearman'])}/{fmt(count['exact_count_accuracy'])}/"
                f"{fmt(count['false_zero_rate'])}/{fmt(count['false_nonzero_rate'])} | "
                f"{viability['states_retaining_admitted_action']}/{viability['oracle_viable_states']} | "
                f"{viability['correct_abstentions']}/{viability['oracle_nonviable_states']} | "
                f"{viability['selected_immediate_contacts']}/{viability['selected_oracle_nonviable_successors']} | "
                f"{fmt(viability['oracle_progress_fraction'])}/{fmt(viability['normalized_regret'])}/"
                f"{fmt(viability['best_admissible_top1'])}/{fmt(viability['best_admissible_top3'])} |"
            )
    lines.extend(
        [
            "",
            "### True-future support by body region",
            "",
            "| Condition | Region | Support | Self-occluded |",
            "|---|---|---:|---:|",
        ]
    )
    for condition_id, coverage in result["coverage_support"].items():
        for region, values in coverage["per_region"].items():
            lines.append(
                f"| {condition_id} | {region} | {fmt(values['support_fraction'])} | "
                f"{fmt(values['self_occluded_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "Per-link and per-family true-future support is preserved in full under "
            "`coverage_support.per_link` and `coverage_support.per_family` in result.json.",
            "",
            "### Error attribution",
            "",
        ]
    )
    lines.append(
        ", ".join(
            f"`{key}`={value}"
            for key, value in result["coverage_error_counts"].items()
        )
    )
    benchmark = result["compute_benchmark"]
    pooled = benchmark["pooled_complete_state_samples"]
    lines.extend(
        [
            "",
            "## Coverage, errors, and compute",
            "",
            "Complete per-link, per-region, per-family, safe-count, viability and error "
            "evidence is in the machine-readable result and gzip JSONL ledgers.",
            "",
            f"Benchmark condition: `{benchmark['condition_id']}`; P50 "
            f"{pooled['p50_ms']:.3f} ms, P90 {pooled['p90_ms']:.3f} ms, "
            f"P95 {pooled['p95_ms']:.3f} ms, P99 {pooled['p99_ms']:.3f} ms, "
            f"max {pooled['max_ms']:.3f} ms; classification "
            f"`{benchmark['classification']}`.",
            "",
            "## Materialisation and storage",
            "",
            f"Executed {result['materialisation_counts']['executed_multi_origin_conditions']} "
            "multi-origin conditions over 176 states, 29,470 transitions, 1,473,500 "
            "physics frames and 13 protected links. Dense evidence reused 13,584 exact "
            "geometry representatives; realistic scans were phase-bound to every "
            "transition identity.",
            "",
            f"Allocated before terminal result: {result['storage_actual']['allocated_bytes_before_result']} "
            f"bytes; final ceiling: {result['storage_preflight']['final_ceiling_bytes']} bytes.",
            "",
            "## Hardware feasibility",
            "",
            f"Strongest/relevant layout `{result['hardware_accounting']['layout_id']}` uses "
            f"{result['hardware_accounting']['sensor_count']} assumed L2-equivalent sensors, "
            f"{result['hardware_accounting']['aggregate_point_rate_hz']} effective points/s, "
            f"{result['hardware_accounting']['aggregate_rays_per_100ms']} rays/100 ms, "
            f"an estimated {result['hardware_accounting']['aggregate_structured_scan_payload_bandwidth_mbps']:.3f} "
            "Mbit/s structured payload under the explicit 16-byte/point assumption, "
            f"{result['hardware_accounting']['aggregate_sensor_mass_g']} g nominal sensor mass, "
            f"{result['hardware_accounting']['supplemental_sensor_mass_over_stock_g']} g supplemental "
            "sensor mass relative to the stock unit, "
            f"and {result['hardware_accounting']['aggregate_typical_power_w']} W typical sensor power. "
            "Wire-protocol overhead, payload integration, cable routing and mechanical "
            "approval remain unresolved. This is not an approved BOM.",
            "",
            "## Decision boundary",
            "",
            f"Next decision: `{result['next_decision']['decision']}`.",
            "",
            "Replicated assumed L2 scans are a geometric development contract, not an "
            "approved BOM or purchase recommendation. No model was trained; no fresh "
            "panel, JEPA predictor, G2 evaluation, memory, novelty, learned navigation, "
            "routing, or beacon system was opened or executed.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(*, execution_started: float | None = None) -> dict[str, Any]:
    evaluation_started = time.time()
    total_started = evaluation_started if execution_started is None else execution_started
    preexecution = validate_preflight()
    selection = select_layouts()
    dual_index = materialize_conditions(tuple(CONTRACT.DUAL_CONDITION_IDS))
    dual = analyze_condition_phase(tuple(CONTRACT.DUAL_CONDITION_IDS))
    executed_conditions = list(CONTRACT.DUAL_CONDITION_IDS)
    thresholds: dict[str, Any] = dict(dual["thresholds"])
    condition_metrics: dict[str, Any] = dict(dual["condition_metrics"])
    gates: dict[str, Any] = dict(dual["true_future_gates"])
    indices = {"dual": dual_index}
    phases = {"dual": dual}
    dual_realistic_pass = bool(gates["DUAL_REALISTIC_L2_SCAN"]["pass"])
    if not dual_realistic_pass:
        three_index = materialize_conditions(tuple(CONTRACT.THREE_CONDITION_IDS))
        three = analyze_condition_phase(tuple(CONTRACT.THREE_CONDITION_IDS))
        indices["three"] = three_index
        phases["three"] = three
        executed_conditions.extend(CONTRACT.THREE_CONDITION_IDS)
        thresholds.update(three["thresholds"])
        condition_metrics.update(three["condition_metrics"])
        gates.update(three["true_future_gates"])
        if not bool(gates["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"]["pass"]):
            diagnostic_index = materialize_conditions((DIAGNOSTIC_CONDITION,))
            diagnostic = analyze_condition_phase(
                (DIAGNOSTIC_CONDITION,),
                modes_by_condition={
                    DIAGNOSTIC_CONDITION: (
                        "TRUE_FUTURE_OBSERVABILITY_CLOUD",
                    )
                },
            )
            indices["diagnostic"] = diagnostic_index
            phases["diagnostic"] = diagnostic
            executed_conditions.append(DIAGNOSTIC_CONDITION)
            thresholds.update(diagnostic["thresholds"])
            condition_metrics.update(diagnostic["condition_metrics"])
            gates.update(diagnostic["true_future_gates"])
    executed = tuple(executed_conditions)
    regression = _predecessor_regression_payload()
    atomic_json(OUTPUT_ROOT / "receipts/predecessor_regression.json", regression)
    all_gate_results = {
        **regression["gate_results"]["true_future"],
        **gates,
    }
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    classification = metrics.classify_multi_origin_conditions(all_gate_results)
    row_receipt = persist_row_level_evidence(executed, thresholds, condition_metrics)
    coverage = coverage_support_summary(executed)
    error_receipt = persist_coverage_errors(executed, thresholds, condition_metrics)
    strongest = (
        "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"
        if "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND" in executed
        else "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"
    )
    benchmark = launch_benchmark(
        strongest,
        float(
            thresholds[strongest]["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
            ["selected_threshold_m"]
        ),
    )
    realistic_pass = bool(
        gates.get("DUAL_REALISTIC_L2_SCAN", {}).get("pass")
        or gates.get("THREE_REALISTIC_L2_SCAN", {}).get("pass")
    )
    contract = CONTRACT.load_and_validate_contract(TRACKED_CONTRACT)
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    mounts = CONTRACT.load_and_validate_mount_library(TRACKED_MOUNTS)
    fixture = json.loads(TRACKED_FIXTURE.read_text())
    storage_before = _directory_allocated_bytes(OUTPUT_ROOT)
    planning_reference_condition = next(
        (
            condition_id
            for condition_id in (
                "THREE_REALISTIC_L2_SCAN",
                "DUAL_REALISTIC_L2_SCAN",
                "THREE_DENSE_L2_FOV_UPPER_BOUND",
                "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
                "DUAL_DENSE_L2_FOV_UPPER_BOUND",
                "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
            )
            if condition_id in condition_metrics
            and gates.get(condition_id, {}).get("pass") is True
        ),
        None,
    )
    planning_time_observability_limitation = False
    if planning_reference_condition is not None:
        causal = condition_metrics[planning_reference_condition].get(
            "PLANNING_TIME_CAUSAL_CLOUD"
        )
        planning_time_observability_limitation = bool(
            causal is None
            or not metrics.evaluate_true_future_gate(causal)["pass"]
        )
    secondary = metrics.derive_secondary_classifications(
        conditional_classification=classification,
        planning_time_observability_limitation=(
            planning_time_observability_limitation
        ),
        coverage_error_counts=error_receipt["counts"],
        assumed_sensor_contract=True,
    )
    for value in (
        "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO",
        "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO",
        "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL",
        "DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED",
        "STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL",
        "REPLANNING_INTERFACE_UNRESOLVED",
        "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
        "ASSUMED_SENSOR_CONTRACT",
        benchmark["classification"],
    ):
        if value not in secondary:
            secondary.append(value)
    aggregate_manifest = OUTPUT_ROOT / "raw_audit/manifest.jsonl"
    aggregate_payload = b"".join(
        (OUTPUT_ROOT / "raw_audit" / f"{phase}_manifest.jsonl").read_bytes()
        for phase in indices
    )
    BASE.atomic_bytes(aggregate_manifest, aggregate_payload)
    raw_manifest_receipt = {
        "schema": "minimum_multi_origin_raw_audit_manifest_v1",
        "path": str(aggregate_manifest),
        "sha256": sha256_file(aggregate_manifest),
        "bytes": aggregate_manifest.stat().st_size,
        "rows": sum(int(index["raw_audit_artifacts"]) for index in indices.values()),
        "phase_manifests": {
            phase: {
                "path": index["raw_audit_manifest_path"],
                "sha256": index["raw_audit_manifest_sha256"],
            }
            for phase, index in indices.items()
        },
        "pass": True,
    }
    raw_manifest_receipt["content_digest"] = content_digest(raw_manifest_receipt)
    atomic_json(OUTPUT_ROOT / "raw_audit/manifest_receipt.json", raw_manifest_receipt)
    execution_plan = {
        "schema": "minimum_multi_origin_conditional_execution_plan_v1",
        "experiment_id": EXPERIMENT,
        "source_freeze_commit": preexecution["head"],
        "selected_dual_layout_id": selection["selected_dual_layout"],
        "selected_three_layout_id": selection["selected_three_origin_layout"],
        "dual_realistic_pass": dual_realistic_pass,
        "regression_complete": True,
        "dual_complete": True,
        "dual_realistic_gate_pass": dual_realistic_pass,
        "three_required": not dual_realistic_pass,
        "three_origin_executed": "three" in phases,
        "three_complete": "three" in phases,
        "all_four_diagnostic_required": bool(
            "three" in phases
            and not gates["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"]["pass"]
        ),
        "all_four_diagnostic_executed": "diagnostic" in phases,
        "all_four_diagnostic_complete": "diagnostic" in phases,
        "executed_condition_ids": list(executed),
        "phase_order": list(indices),
        "executed_condition_modes": {
            condition_id: (
                ["TRUE_FUTURE_OBSERVABILITY_CLOUD"]
                if condition_id == DIAGNOSTIC_CONDITION
                else list(MODES)
            )
            for condition_id in executed
        },
        "skipped_condition_modes": {
            **{
                condition_id: list(MODES)
                for condition_id in MULTI_CONDITIONS + (DIAGNOSTIC_CONDITION,)
                if condition_id not in executed
            },
            **(
                {DIAGNOSTIC_CONDITION: ["PLANNING_TIME_CAUSAL_CLOUD"]}
                if DIAGNOSTIC_CONDITION in executed
                else {}
            ),
        },
        "materialization_indices": {
            phase: {
                "path": str(OUTPUT_ROOT / "materialization" / f"{phase}_index.json"),
                "sha256": sha256_file(
                    OUTPUT_ROOT / "materialization" / f"{phase}_index.json"
                ),
            }
            for phase in indices
        },
        "threshold_freezes": {
            phase: {
                "path": str(
                    OUTPUT_ROOT / "calibration" / f"{phase}_thresholds_frozen.json"
                ),
                "sha256": sha256_file(
                    OUTPUT_ROOT / "calibration" / f"{phase}_thresholds_frozen.json"
                ),
            }
            for phase in indices
        },
        "conditional_rule": (
            "stop after dual only iff dual realistic passes; otherwise execute all "
            "three-origin conditions; execute all-four dense spherical true-future "
            "diagnostic iff three-origin dense spherical fails"
        ),
        "conditional_flow_valid": True,
        "pass": True,
    }
    execution_plan["content_digest"] = content_digest(execution_plan)
    _validate_execution_plan_receipt(
        execution_plan,
        gates,
        selection,
        expected_source_freeze_commit=preexecution["head"],
    )
    atomic_json(OUTPUT_ROOT / "receipts/execution_plan.json", execution_plan)
    planning_comparison = {
        condition: {
            "planning_time": condition_metrics[condition].get(
                "PLANNING_TIME_CAUSAL_CLOUD"
            ),
            "true_future": condition_metrics[condition][
                "TRUE_FUTURE_OBSERVABILITY_CLOUD"
            ],
            "future_points_available_before_execution": False,
            "true_future_is_observability_upper_bound_not_prediction": True,
        }
        for condition in executed
    }
    hardware = _hardware_feasibility_summary(classification, selection)
    storage_summary = {
        "preflight": {
            "workspace_filesystem": preexecution["workspace_filesystem"],
            "output_filesystem": preexecution["output_filesystem"],
            "temporary_ceiling_bytes": 50_000_000_000,
            "final_ceiling_bytes": 25_000_000_000,
            "predicted_temporary_bytes": preexecution[
                "predicted_temporary_storage_bytes"
            ],
            "predicted_final_bytes": preexecution["predicted_final_storage_bytes"],
        },
        "actual": {
            "allocated_bytes_before_result": storage_before,
            "ceiling_pass": storage_before <= 25_000_000_000,
        },
    }
    result = {
        "schema_version": "minimum_multi_origin_body_range_coverage_qualification_v1.result.v1",
        "experiment_id": EXPERIMENT,
        "development_only": True,
        "claim_bearing": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_freeze_commit": preexecution["head"],
        "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
        "predecessor_contract_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "contract_content_digest": contract["contract_sha256"],
        "output_schema_sha256": sha256_file(TRACKED_SCHEMA),
        "output_schema_content_digest": schema["output_schema_sha256"],
        "mount_library_sha256": sha256_file(TRACKED_MOUNTS),
        "mount_library_receipt_sha256": sha256_file(TRACKED_MOUNTS),
        "mount_library_content_digest": mounts["content_digest"],
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "corpus_bindings": {
            **EXPECTED,
            "result_commit": PREDECESSOR_RESULT_COMMIT,
            "contract_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
            "body_centric_result_content_digest": PREDECESSOR_RESULT_CONTENT_DIGEST,
        },
        "environment_receipt": preexecution["environment"],
        "storage_preflight": {
            "workspace_filesystem": preexecution["workspace_filesystem"],
            "output_filesystem": preexecution["output_filesystem"],
            "temporary_ceiling_bytes": 50_000_000_000,
            "final_ceiling_bytes": 25_000_000_000,
            "predicted_temporary_bytes": preexecution[
                "predicted_temporary_storage_bytes"
            ],
            "predicted_final_bytes": preexecution[
                "predicted_final_storage_bytes"
            ],
        },
        "storage_actual": {
            "allocated_bytes_before_result": storage_before,
            "ceiling_pass": storage_before <= 25_000_000_000,
        },
        "storage": storage_summary,
        "mount_library": mounts,
        "layout_selection": selection,
        "executed_conditions": list(executed),
        "conditional_execution": {
            "dual_realistic_pass": dual_realistic_pass,
            "three_origin_executed": "three" in phases,
            "all_four_diagnostic_executed": "diagnostic" in phases,
        },
        "execution_plan": execution_plan,
        "single_origin_regression": regression,
        "materialisation_indices": indices,
        "support_dominance": {
            phase: index["support_dominance"] for phase, index in indices.items()
        },
        "materialisation_counts": {
            "states_per_condition": EXPECTED["states"],
            "transitions_per_condition": EXPECTED["transitions"],
            "action_representatives_per_condition": EXPECTED[
                "action_representatives"
            ],
            "geometry_representatives_per_condition": EXPECTED[
                "geometry_representatives"
            ],
            "physics_frames_per_condition": EXPECTED["physics_frames"],
            "protected_links": LINKS,
            "executed_multi_origin_conditions": len(executed),
            "raw_audit_artifacts": sum(
                int(index["raw_audit_artifacts"]) for index in indices.values()
            ),
            "realistic_transition_identities_per_executed_realistic_condition": EXPECTED[
                "transitions"
            ],
            "realistic_scan_phase_reuse_across_transition_identities": 0,
            "scan_materialization_counts": {
                phase: index["scan_counts"] for phase, index in indices.items()
            },
            "dense_analytic_target_query_counts": {
                phase: index["dense_analytic_target_query_counts"]
                for phase, index in indices.items()
            },
            "runtime_s": sum(float(index["runtime_s"]) for index in indices.values()),
        },
        "fixture_results": fixture,
        "calibration_thresholds": thresholds,
        "condition_metrics": {
            **regression["condition_metrics"],
            **condition_metrics,
        },
        "gate_results": all_gate_results,
        "causal_vs_true_future": planning_comparison,
        "planning_time_true_future_comparison": planning_comparison,
        "coverage_support": coverage,
        "planning_time_observability_limitation_reference_condition": (
            planning_reference_condition
        ),
        "coverage_errors": error_receipt,
        "coverage_error_counts": error_receipt["counts"],
        "row_level_evidence": row_receipt,
        "compute_benchmark": benchmark,
        "compute_classification": benchmark["classification"],
        "primary_classification": classification["primary_classification"],
        "secondary_classifications": secondary,
        "classification_receipt": classification,
        "hardware_feasibility": hardware,
        "hardware_accounting": hardware,
        "next_decision": _next_decision(
            classification["primary_classification"], realistic_pass
        ),
        "preserved_findings": {
            "exact_geometry_upper_bound": (
                "Exact Genesis per-link geometry reconstructs immediate contact, "
                "successor contact, zero-versus-nonzero safe-action availability and "
                "the complete two-ply viability decision."
            ),
            "single_origin_finding": (
                "Under the tested full-body protected-contact scope, one range-sensor "
                "origin does not observe enough of the trunk, hip, thigh and calf swept "
                "volume. Vertical coverage and robot self-occlusion dominate; additional "
                "scan density at the same origin does not solve the problem."
            ),
        },
        "custody": {
            "training_steps": 0,
            "fresh_panel_rows": 0,
            "simulator_steps": 0,
            "jepa_predictor_opens": 0,
            "g2_opens": 0,
            "memory_novelty_navigation_routing_beacon_executions": 0,
            "contact_label_changes": 0,
            "state_transition_action_identity_changes": 0,
            "protected_link_changes": 0,
        },
        "raw_audit_manifest": raw_manifest_receipt,
        "runtime_s": time.time() - total_started,
        "analysis_and_conditional_execution_runtime_s": (
            time.time() - evaluation_started
        ),
    }
    if not result["storage_actual"]["ceiling_pass"]:
        raise RuntimeError("final storage ceiling exceeded before result persistence")
    result_digest = content_digest(result)
    result["result_content_digest"] = result_digest
    result["result_content_sha256"] = result_digest
    result_payload = (
        json.dumps(BASE.json_ready(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    report_payload = _render_markdown(result).encode()
    BASE.atomic_bytes(OUTPUT_ROOT / "result.json", result_payload)
    BASE.atomic_bytes(OUTPUT_ROOT / "report.md", report_payload)
    BASE.atomic_bytes(RESULT_JSON, result_payload)
    BASE.atomic_bytes(RESULT_MD, report_payload)
    persistence = {
        "schema": "minimum_multi_origin_body_range_coverage_persistence_v1",
        "experiment_id": EXPERIMENT,
        "contract_sha256": sha256_file(TRACKED_CONTRACT),
        "mount_library_receipt_sha256": sha256_file(TRACKED_MOUNTS),
        "source_closure_sha256": sha256_file(TRACKED_CLOSURE),
        "result": {
            "path": str(OUTPUT_ROOT / "result.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "result.json"),
            "bytes": (OUTPUT_ROOT / "result.json").stat().st_size,
        },
        "report": {
            "path": str(OUTPUT_ROOT / "report.md"),
            "sha256": sha256_file(OUTPUT_ROOT / "report.md"),
            "bytes": (OUTPUT_ROOT / "report.md").stat().st_size,
        },
        "row_level_evidence": row_receipt,
        "coverage_errors": error_receipt,
        "layout_selection_receipt": {
            "path": str(OUTPUT_ROOT / "layout_selection/selected_layouts.json"),
            "sha256": sha256_file(
                OUTPUT_ROOT / "layout_selection/selected_layouts.json"
            ),
        },
        "execution_plan_receipt": {
            "path": str(OUTPUT_ROOT / "receipts/execution_plan.json"),
            "sha256": sha256_file(OUTPUT_ROOT / "receipts/execution_plan.json"),
        },
        "materialization_indices": {
            phase: {
                "path": str(OUTPUT_ROOT / "materialization" / f"{phase}_index.json"),
                "sha256": sha256_file(
                    OUTPUT_ROOT / "materialization" / f"{phase}_index.json"
                ),
            }
            for phase in indices
        },
        "executed_materialization_indices": execution_plan[
            "materialization_indices"
        ],
        "executed_threshold_freeze_receipts": execution_plan["threshold_freezes"],
        "raw_audit_manifests": raw_manifest_receipt,
        "allocated_bytes_final": 0,
        "final_storage_ceiling_bytes": 25_000_000_000,
        "pass": True,
    }
    persistence_path = OUTPUT_ROOT / "persistence_receipt.json"
    for _attempt in range(4):
        persistence.pop("content_digest", None)
        persistence["pass"] = (
            int(persistence["allocated_bytes_final"])
            <= int(persistence["final_storage_ceiling_bytes"])
        )
        persistence["content_digest"] = content_digest(persistence)
        atomic_json(persistence_path, persistence)
        allocated = _directory_allocated_bytes(OUTPUT_ROOT)
        if allocated == int(persistence["allocated_bytes_final"]):
            break
        persistence["allocated_bytes_final"] = allocated
    else:  # pragma: no cover - fixed-size receipt should converge immediately
        raise RuntimeError("final allocated-byte receipt did not converge")
    if not persistence["pass"]:
        raise RuntimeError("final persistence storage ceiling exceeded")
    return result


def execute() -> dict[str, Any]:
    execution_started = time.time()
    if not (OUTPUT_ROOT / "preexecution_receipt.json").is_file():
        preflight()
    validate_preflight()
    select_layouts()
    materialize()
    return evaluate(execution_started=execution_started)


def _require_schema_keys(
    value: dict[str, Any], required: Sequence[str], artifact: str
) -> None:
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(f"{artifact} is missing frozen schema keys: {missing}")


def _validate_jsonl_schema(
    path: Path,
    required: Sequence[str],
    *,
    compressed: bool,
    expected_rows: int | None = None,
) -> tuple[int, dict[str, int]]:
    opener = gzip.open if compressed else open
    count = 0
    error_counts: dict[str, int] = defaultdict(int)
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"JSONL row lacks LF: {path}:{line_number}")
            row = json.loads(line)
            _require_schema_keys(row, required, f"{path}:{line_number}")
            if "error_class" in row:
                error_counts[str(row["error_class"])] += 1
            count += 1
    if expected_rows is not None and count != expected_rows:
        raise RuntimeError(
            f"JSONL cardinality drift for {path}: {count} != {expected_rows}"
        )
    return count, dict(error_counts)


def check_result() -> dict[str, Any]:
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("wrong result branch")
    result_path = OUTPUT_ROOT / "result.json"
    persistence_path = OUTPUT_ROOT / "persistence_receipt.json"
    if not result_path.is_file() or not persistence_path.is_file():
        raise RuntimeError("terminal result/persistence receipt is missing")
    result = json.loads(result_path.read_text())
    schema = CONTRACT.load_and_validate_output_schema(TRACKED_SCHEMA)
    files = schema["files"]
    _require_schema_keys(result, files["result"]["required_keys"], "result")
    core = dict(result)
    declared = core.pop("result_content_digest", None)
    declared_alias = core.pop("result_content_sha256", None)
    if declared != declared_alias or declared != content_digest(core):
        raise RuntimeError("result self-digest mismatch")
    if result_path.read_bytes() != RESULT_JSON.read_bytes():
        raise RuntimeError("tracked/external result mismatch")
    if (OUTPUT_ROOT / "report.md").read_bytes() != RESULT_MD.read_bytes():
        raise RuntimeError("tracked/external report mismatch")
    persistence = json.loads(persistence_path.read_text())
    _require_schema_keys(
        persistence,
        files["persistence_receipt"]["required_keys"],
        "persistence receipt",
    )
    persistence_core = dict(persistence)
    persistence_digest = persistence_core.pop("content_digest", None)
    if persistence_digest != content_digest(persistence_core) or not persistence["pass"]:
        raise RuntimeError("persistence receipt invalid")
    for artifact in (persistence["result"], persistence["report"]):
        path = Path(artifact["path"])
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_file(path) != artifact["sha256"]
        ):
            raise RuntimeError(f"terminal artifact drift: {path}")
    for phase, entry in persistence["materialization_indices"].items():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"materialization index drift: {phase}")
        index = json.loads(path.read_text())
        _require_schema_keys(
            index, files["materialization_index"]["required_keys"], path.name
        )
        for record in index["records"]:
            _require_schema_keys(
                record,
                files["materialization_index"]["record_required_keys"],
                f"{path.name} record",
            )
            shard = Path(record["shard_path"])
            if (
                shard.stat().st_size != record["shard_bytes"]
                or sha256_file(shard) != record["shard_sha256"]
            ):
                raise RuntimeError(f"state condition shard drift: {shard}")
    for receipt in (persistence["row_level_evidence"], persistence["coverage_errors"]):
        if "path" in receipt:
            artifacts = (receipt,)
        else:
            artifacts = (
                receipt["transition_evidence"],
                receipt["per_link_evidence"],
            )
        for artifact in artifacts:
            path = Path(artifact["path"])
            if sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"row evidence drift: {path}")
    artifact_values = {
        "preexecution_receipt": json.loads(
            (OUTPUT_ROOT / "preexecution_receipt.json").read_text()
        ),
        "environment_receipt": json.loads(
            (OUTPUT_ROOT / "receipts/environment_receipt.json").read_text()
        ),
        "mount_library_receipt": json.loads(
            (OUTPUT_ROOT / "receipts/mount_library.json").read_text()
        ),
        "layout_selection_receipt": json.loads(
            (OUTPUT_ROOT / "layout_selection/selected_layouts.json").read_text()
        ),
        "predecessor_regression_receipt": json.loads(
            (OUTPUT_ROOT / "receipts/predecessor_regression.json").read_text()
        ),
        "execution_plan_receipt": json.loads(
            (OUTPUT_ROOT / "receipts/execution_plan.json").read_text()
        ),
        "row_level_evidence_receipt": persistence["row_level_evidence"],
        "coverage_error_receipt": persistence["coverage_errors"],
        "compute_benchmark": result["compute_benchmark"],
    }
    for name, value in artifact_values.items():
        _require_schema_keys(value, files[name]["required_keys"], name)
    _validate_training_layout_evidence_rows(
        artifact_values["layout_selection_receipt"]["training_layout_evidence"],
        artifact_values["layout_selection_receipt"],
    )
    _validate_execution_plan_receipt(
        artifact_values["execution_plan_receipt"],
        result["gate_results"],
        artifact_values["layout_selection_receipt"],
        expected_source_freeze_commit=result["source_freeze_commit"],
    )
    if (OUTPUT_ROOT / "receipts/mount_library.json").read_bytes() != TRACKED_MOUNTS.read_bytes():
        raise RuntimeError("tracked/output mount-library receipt mismatch")
    fixture = _combined_fixture_receipt()
    if TRACKED_FIXTURE.read_bytes() != canonical_bytes(fixture) or not fixture["pass"]:
        raise RuntimeError("terminal fixture regeneration mismatch")
    validate_source_closure()
    mode_condition_count = sum(
        1 if condition == DIAGNOSTIC_CONDITION else len(MODES)
        for condition in result["executed_conditions"]
    )
    transition_rows_expected = EXPECTED["transitions"] * mode_condition_count
    transition_path = Path(
        persistence["row_level_evidence"]["transition_evidence"]["path"]
    )
    per_link_path = Path(
        persistence["row_level_evidence"]["per_link_evidence"]["path"]
    )
    _validate_jsonl_schema(
        transition_path,
        files["transition_evidence"]["required_keys"],
        compressed=True,
        expected_rows=transition_rows_expected,
    )
    _validate_jsonl_schema(
        per_link_path,
        files["per_link_evidence"]["required_keys"],
        compressed=True,
        expected_rows=transition_rows_expected * LINKS,
    )
    coverage_path = Path(persistence["coverage_errors"]["path"])
    coverage_rows, coverage_counts = _validate_jsonl_schema(
        coverage_path,
        files["coverage_errors"]["required_keys"],
        compressed=True,
        expected_rows=int(persistence["coverage_errors"]["rows"]),
    )
    if coverage_rows != sum(coverage_counts.values()):
        raise RuntimeError("coverage-error class cardinality drift")
    if {
        key: coverage_counts.get(key, 0) for key in CONTRACT.COVERAGE_ERROR_CLASSES
    } != persistence["coverage_errors"]["counts"]:
        raise RuntimeError("coverage-error class receipt drift")
    terminal_support_dominance: dict[str, Any] = {}
    for phase, entry in persistence["executed_materialization_indices"].items():
        index_path = Path(entry["path"])
        if sha256_file(index_path) != entry["sha256"]:
            raise RuntimeError(f"materialization-index drift: {phase}")
        index = json.loads(index_path.read_text())
        index_core = dict(index)
        declared_index_digest = index_core.pop("content_digest", None)
        if declared_index_digest != content_digest(index_core):
            raise RuntimeError(f"materialization-index self-digest drift: {phase}")
        condition_ids = tuple(index.get("conditions", ()))
        _validate_support_dominance_shape(
            index.get("support_dominance", {}), condition_ids
        )
        terminal_support_dominance[phase] = index["support_dominance"]
        _validate_uid_bound_scan_counts(
            index.get("scan_counts", {}),
            condition_ids,
            artifact_values["layout_selection_receipt"],
        )
        for record in index["records"]:
            shard = Path(record["shard_path"])
            if (
                shard.stat().st_size != int(record["shard_bytes"])
                or sha256_file(shard) != record["shard_sha256"]
            ):
                raise RuntimeError(f"materialization shard drift: {shard}")
            _validate_condition_npz_schema(
                record, artifact_values["layout_selection_receipt"]
            )
        validated_state_receipts = [
            _validate_bound_state_phase_receipt(
                record,
                condition_ids,
                artifact_values["layout_selection_receipt"],
            )
            for record in index["state_records"]
        ]
        _validate_all_state_scan_receipts(
            validated_state_receipts,
            condition_ids,
            artifact_values["layout_selection_receipt"],
        )
        manifest = Path(index["raw_audit_manifest_path"])
        _validate_jsonl_schema(
            manifest,
            files["raw_audit_manifest"]["required_keys"],
            compressed=False,
            expected_rows=int(index["raw_audit_artifacts"]),
        )
        with manifest.open("rt", encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                artifact = OUTPUT_ROOT / row["artifact_relative_path"]
                if sha256_file(artifact) != row["artifact_sha256"]:
                    raise RuntimeError(f"raw-audit artifact drift: {artifact}")
    if result.get("support_dominance") != terminal_support_dominance:
        raise RuntimeError("result/materialization support-dominance drift")
    for phase, entry in persistence["executed_threshold_freeze_receipts"].items():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"threshold-freeze drift: {phase}")
        freeze = json.loads(path.read_text())
        _require_schema_keys(
            freeze, files["calibration_thresholds"]["required_keys"], path.name
        )
        for mode_rows in freeze["thresholds"].values():
            for threshold in mode_rows.values():
                artifact = OUTPUT_ROOT / threshold["artifact_relative_path"]
                if sha256_file(artifact) != threshold["artifact_sha256"]:
                    raise RuntimeError(f"threshold frontier drift: {artifact}")
    from lewm.safety import minimum_multi_origin_body_range_coverage_metrics_v1 as metrics

    recomputed = metrics.classify_multi_origin_conditions(result["gate_results"])
    if recomputed["primary_classification"] != result["primary_classification"]:
        raise RuntimeError("primary classification recomputation drift")
    if any(OUTPUT_ROOT.glob("MATERIALIZATION_*_RUNNING.json")):
        raise RuntimeError("materialization marker remains")
    processes = BASE.running_scientific_processes()
    if processes:
        raise RuntimeError(f"scientific process remains active: {processes}")
    if _directory_allocated_bytes(OUTPUT_ROOT) > 25_000_000_000:
        raise RuntimeError("final output exceeds storage ceiling")
    return {
        "status": "PASS",
        "result_content_digest": result["result_content_digest"],
        "primary_classification": result["primary_classification"],
        "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(OUTPUT_ROOT / "report.md"),
        "persistence_sha256": sha256_file(persistence_path),
        "running_processes": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "preflight",
            "select-layouts",
            "materialize",
            "evaluate",
            "execute",
            "check",
            "benchmark-worker",
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        value = write_freeze_receipts()
    elif args.command == "preflight":
        value = preflight()
    elif args.command == "select-layouts":
        value = select_layouts()
    elif args.command == "materialize":
        value = materialize()
    elif args.command == "evaluate":
        value = evaluate()
    elif args.command == "execute":
        value = execute()
    elif args.command == "benchmark-worker":
        value = _benchmark_worker()
    else:
        value = check_result()
    print(json.dumps(BASE.json_ready(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
