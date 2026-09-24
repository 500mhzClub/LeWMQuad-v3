#!/usr/bin/env python3
"""Publish the fixed V4 canonical-claim oracle regression, fail closed.

This module intentionally imports only the Python standard library at module
load. Project imports happen only after the reviewed implementation manifest
and its complete source map have been verified byte-for-byte.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_canonical_physical_claim_oracle_regression_implementation_manifest_2026-07-11.json"
)
OUTPUT_PATH = (
    ROOT
    / ".generated/oracle_positive_control/go2_generalization_v4_development"
    / "canonical_physical_claim_v1_report.json"
)
DEVELOPMENT_MANIFEST_PATH = ROOT / "config/go2_generalization_v4/development.json"
SCENE_CORPUS_PATH = ROOT / ".generated/scene_corpus/go2_generalization_v4"
MATERIALIZATION_PATH = SCENE_CORPUS_PATH / "materialization_both.json"
GEOMETRY_PATH = ROOT / "config/go2_generalization_geometry_v2.json"
PRIMITIVE_REGISTRY_PATH = ROOT / "config/go2_primitive_registry.yaml"
DIRECTIONAL_POLICY_PATH = (
    ROOT
    / "config/go2_geometry_v2_artifacts"
    / "go2_directional_footprint_policy_v1_c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc.json"
)
PRIOR_COMPARATOR_PATH = (
    ROOT
    / ".generated/oracle_positive_control/go2_generalization_v4_development/report.json"
)

IMPLEMENTATION_MANIFEST_SCHEMA = (
    "lewm_go2_canonical_physical_claim_oracle_implementation_manifest_v1"
)
RESULT_SCHEMA = "lewm_go2_canonical_physical_claim_oracle_regression_v1"
EXPECTED_BINDING_SHA256 = (
    "2de4ff20cff2901ab07b681f042c231f1a1e06f95a77d8c4ae2c20c9e2bb8112"
)
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "563f240a023309af42a05a9a8f29008f02a0629dee9f77f03568f779d1166d41"
)
EXPECTED_MATERIALIZATION_SHA256 = (
    "a52bd82cb501481707d518d1fffd86e5475b440332f7d226586ebda47e6b1415"
)
EXPECTED_GEOMETRY_FILE_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
EXPECTED_GEOMETRY_SEMANTIC_SHA256 = (
    "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
)
EXPECTED_PRIMITIVE_REGISTRY_SHA256 = (
    "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8"
)
EXPECTED_DIRECTIONAL_POLICY_FILE_SHA256 = (
    "750d8afe47ee3edd5988cdea443f19703efad7a3266218932671b9fdfbe43828"
)
EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256 = (
    "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc"
)
EXPECTED_PRIOR_COMPARATOR_SHA256 = (
    "7c0a63bb0548fee81918df22b227adec43d4bdc824875ef447793ef4f99d97a5"
)

SOURCE_PATHS: Mapping[str, str] = {
    "binding": "docs/lewm_go2_canonical_physical_claim_evaluator_binding_2026-07-11.md",
    "integration_record": "docs/lewm_go2_canonical_physical_claim_integration_2026-07-11.md",
    "physical_claim_evaluator": "lewm/benchmarks/go2_physical_claim_evaluator.py",
    "physical_claim_canonical": "lewm/benchmarks/go2_physical_claim_canonical.py",
    "physical_claim_trace": "lewm/benchmarks/go2_physical_claim_trace.py",
    "physical_claim_observer": "lewm/benchmarks/go2_physical_claim_observer.py",
    "physical_claim_result": "lewm/benchmarks/go2_physical_claim_result.py",
    "physical_claim_runtime_finalizer": "lewm/benchmarks/go2_physical_claim_finalizer.py",
    "generalization_protocol": "lewm/benchmarks/generalization_protocol.py",
    "strict_result_scorer": "lewm/benchmarks/strict_result_scorer.py",
    "oracle": "lewm/benchmarks/go2_oracle_positive_control.py",
    "physical_eligibility": "lewm/benchmarks/go2_physical_eligibility.py",
    "runtime": "scripts/benchmark_go2_memory_closed_loop.py",
    "batch_scorer": "scripts/score_go2_result_batch.py",
    "generalized_suite_checker": "scripts/check_go2_generalized_suite.py",
    "fully_learned_checker": "scripts/check_go2_fully_learned_demo.py",
    "teacher_checker": "scripts/check_go2_teacher_dataset.py",
    "clean_demo_checker": "scripts/check_go2_clean_demo_candidate.py",
    "wall_aware_checker": "scripts/check_go2_wallaware_closed_loop_gate.py",
    "generalized_suite_wrapper": "scripts/run_go2_generalized_learned_local_suite.sh",
    "generalized_teacher_collection_wrapper": "scripts/run_go2_generalized_teacher_collection.sh",
    "fully_learned_demo_wrapper": "scripts/run_go2_fully_learned_demo.sh",
    "replay_diagnostic": "scripts/render_go2_closed_loop_result_replay.py",
    "review_video_diagnostic": "scripts/compose_go2_physical_review_ui_video.py",
    "test_physical_claim_evaluator": "lewm/tests/test_go2_physical_claim_evaluator.py",
    "test_physical_claim_canonical": "lewm/tests/test_go2_physical_claim_canonical.py",
    "test_physical_claim_trace": "lewm/tests/test_go2_physical_claim_trace.py",
    "test_physical_claim_observer": "lewm/tests/test_go2_physical_claim_observer.py",
    "test_physical_claim_result": "lewm/tests/test_go2_physical_claim_result.py",
    "test_physical_claim_runtime_finalizer": "lewm/tests/test_go2_physical_claim_finalizer.py",
    "test_generalization_protocol": "lewm/tests/test_generalization_protocol.py",
    "test_strict_result_scorer": "lewm/tests/test_strict_result_scorer.py",
    "test_oracle": "lewm/tests/test_go2_oracle_positive_control.py",
    "test_physical_eligibility": "lewm/tests/test_go2_physical_eligibility.py",
    "test_generalized_suite_checker": "lewm/tests/test_check_go2_generalized_suite_claims.py",
    "test_fully_learned_checker": "lewm/tests/test_check_go2_fully_learned_demo.py",
    "test_clean_demo_checker": "lewm/tests/test_check_go2_clean_demo_candidate.py",
    "test_teacher_checker": "lewm/tests/test_check_go2_teacher_dataset.py",
    "test_wall_aware_checker": "lewm/tests/test_check_go2_wallaware_closed_loop_gate.py",
    "test_claim_checker_manifest_binding": "lewm/tests/test_go2_claim_checker_manifest_binding.py",
    "test_checker_wrapper_manifest_plumbing": "lewm/tests/test_go2_claim_checker_wrapper_manifest_plumbing.py",
    "oracle_suite_finalizer": "lewm/benchmarks/go2_canonical_physical_claim_oracle_finalizer.py",
    "test_oracle_suite_finalizer": "lewm/tests/test_go2_canonical_physical_claim_oracle_finalizer.py",
    "oracle_regression_runner": "scripts/run_go2_canonical_physical_claim_oracle_regression.py",
    "test_oracle_regression_runner": "lewm/tests/test_run_go2_canonical_physical_claim_oracle_regression.py",
    "geometry_contract": "lewm/planning/geometry_contract.py",
    "exact_occupancy_adapter": "lewm/planning/exact_occupancy_belief_adapter.py",
    "online_belief_map": "lewm/planning/online_belief_map.py",
    "oriented_footprint": "lewm/planning/oriented_footprint.py",
    "world_manifest": "lewm_worlds/lewm_worlds/manifest.py",
    "world_scene_graph": "lewm_worlds/lewm_worlds/scene_graph.py",
    "world_planning_grid": "lewm_worlds/lewm_worlds/planning_grid.py",
    "genesis_contract": "lewm_genesis/lewm_genesis/lewm_contract.py",
}

INPUT_PATHS: Mapping[str, str] = {
    "development_manifest": "config/go2_generalization_v4/development.json",
    "materialization": ".generated/scene_corpus/go2_generalization_v4/materialization_both.json",
    "geometry": "config/go2_generalization_geometry_v2.json",
    "primitive_registry": "config/go2_primitive_registry.yaml",
    "directional_policy": (
        "config/go2_geometry_v2_artifacts/"
        "go2_directional_footprint_policy_v1_"
        "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc.json"
    ),
    "prior_comparator": (
        ".generated/oracle_positive_control/go2_generalization_v4_development/report.json"
    ),
}

COMMAND_IDENTITY_TEMPLATE = {
    "cwd": str(ROOT),
    "executable": "/usr/bin/python3",
    "script": "scripts/run_go2_canonical_physical_claim_oracle_regression.py",
    "argument": "--implementation-manifest-sha256",
    "pythonpath": [str(ROOT), str(ROOT / "lewm_worlds"), str(ROOT / "lewm_genesis")],
    "worker_pool": {
        "kind": "spawn_process",
        "worker_count": 6,
        "threads_per_worker": 1,
        "merge_order": "development_manifest_index",
        "worker_runtime_input_file_access": False,
    },
}

ZERO_EVALUATOR_ACCESS_LEDGER = {
    "evaluator_output_reads_by_controller": 0,
    "evaluator_callbacks_into_controller": 0,
    "evaluator_derived_termination_signals": 0,
}
WORKER_COUNT = 6
CPU_THREAD_CAP_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class ReadinessError(ValueError):
    """The frozen authoritative regression contract is not satisfied."""


@dataclass(frozen=True)
class VerifiedImplementation:
    sha256: str
    payload: Mapping[str, Any]
    source_map: Mapping[str, Any]
    input_bindings: Mapping[str, Any]
    command_identity: Mapping[str, Any]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_equal(left: object, right: object) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (OverflowError, TypeError, ValueError):
        return False


def _content_sha256(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("content_sha256", None)
    return _canonical_sha256(content)


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_loads(payload: bytes, *, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReadinessError(f"{label} contains non-finite number {value}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"{label} is not strict UTF-8 JSON") from exc


def _exact_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ReadinessError(f"{label} path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ReadinessError(f"{label} path must be an exact repository-relative path")
    return value


def _verify_source_map(
    source_map: object,
    *,
    root: Path = ROOT,
    file_hasher: Callable[[Path], str] = _hash_file,
) -> Mapping[str, Any]:
    if type(source_map) is not dict or set(source_map) != {
        "entries",
        "source_map_sha256",
    }:
        raise ReadinessError("source_map must contain exact entries/hash keys")
    entries = source_map["entries"]
    if type(entries) is not list or len(entries) != len(SOURCE_PATHS):
        raise ReadinessError("source_map entry count differs from the exact role graph")
    expected_roles = set(SOURCE_PATHS)
    observed: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"role", "path", "sha256"}:
            raise ReadinessError(f"source_map entry {index} has the wrong key set")
        role = entry["role"]
        if type(role) is not str or role not in expected_roles or role in observed:
            raise ReadinessError(f"source_map entry {index} has an invalid/duplicate role")
        path = _exact_relative_path(entry["path"], label=f"source role {role}")
        if path != SOURCE_PATHS[role]:
            raise ReadinessError(f"source role {role} path changed")
        if not _is_sha256(entry["sha256"]):
            raise ReadinessError(f"source role {role} SHA-256 is malformed")
        source_path = root / path
        if not source_path.is_file():
            raise ReadinessError(f"source role {role} is missing")
        if file_hasher(source_path) != entry["sha256"]:
            raise ReadinessError(f"source role {role} SHA-256 mismatch")
        observed[role] = entry
    if set(observed) != expected_roles:
        raise ReadinessError("source_map role set is incomplete")
    if observed["binding"]["sha256"] != EXPECTED_BINDING_SHA256:
        raise ReadinessError("source_map binding source differs from the frozen contract")
    if entries != sorted(entries, key=lambda item: item["role"].encode("utf-8")):
        raise ReadinessError("source_map entries are not in exact role order")
    if not _is_sha256(source_map["source_map_sha256"]):
        raise ReadinessError("source_map SHA-256 is malformed")
    if _canonical_sha256(entries) != source_map["source_map_sha256"]:
        raise ReadinessError("source_map canonical SHA-256 mismatch")
    return source_map


def _verify_input_bindings(input_bindings: object) -> Mapping[str, Any]:
    if type(input_bindings) is not dict or set(input_bindings) != set(INPUT_PATHS):
        raise ReadinessError("input binding role set differs from the frozen graph")
    for role, expected_path in INPUT_PATHS.items():
        record = input_bindings[role]
        expected_keys = {"path", "sha256", "access_mode"}
        if role == "directional_policy":
            expected_keys.add("content_sha256")
        if type(record) is not dict or set(record) != expected_keys:
            raise ReadinessError(f"input binding {role} has the wrong key set")
        if _exact_relative_path(record["path"], label=f"input {role}") != expected_path:
            raise ReadinessError(f"input binding {role} path changed")
        if not _is_sha256(record["sha256"]):
            raise ReadinessError(f"input binding {role} SHA-256 is malformed")
        expected_mode = (
            "identity_only_do_not_open"
            if role == "prior_comparator"
            else "hash_then_parse"
            if role in {"development_manifest", "geometry", "primitive_registry", "directional_policy"}
            else "hash_only"
        )
        if record["access_mode"] != expected_mode:
            raise ReadinessError(f"input binding {role} access mode changed")
    fixed_hashes = {
        "development_manifest": EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "materialization": EXPECTED_MATERIALIZATION_SHA256,
        "geometry": EXPECTED_GEOMETRY_FILE_SHA256,
        "primitive_registry": EXPECTED_PRIMITIVE_REGISTRY_SHA256,
        "directional_policy": EXPECTED_DIRECTIONAL_POLICY_FILE_SHA256,
        "prior_comparator": EXPECTED_PRIOR_COMPARATOR_SHA256,
    }
    for role, expected_hash in fixed_hashes.items():
        if input_bindings[role]["sha256"] != expected_hash:
            raise ReadinessError(f"input binding {role} differs from the frozen binding")
    if (
        input_bindings["directional_policy"]["content_sha256"]
        != EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256
    ):
        raise ReadinessError("directional-policy content identity changed")
    return input_bindings


def _verify_command_identity(command: object) -> Mapping[str, Any]:
    if not _canonical_equal(command, COMMAND_IDENTITY_TEMPLATE):
        raise ReadinessError("authoritative command identity changed")
    return command


def _verify_implementation_payload(
    payload: object,
    *,
    manifest_sha256: str,
    root: Path = ROOT,
    file_hasher: Callable[[Path], str] = _hash_file,
) -> VerifiedImplementation:
    expected_keys = {
        "schema",
        "binding_sha256",
        "source_map",
        "input_bindings",
        "command_identity",
        "oracle_config",
        "eligibility_config",
        "exclusive_output",
        "content_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ReadinessError("implementation manifest has the wrong top-level key set")
    if payload["schema"] != IMPLEMENTATION_MANIFEST_SCHEMA:
        raise ReadinessError("implementation manifest schema changed")
    if payload["binding_sha256"] != EXPECTED_BINDING_SHA256:
        raise ReadinessError("implementation manifest binding SHA-256 changed")
    if not _is_sha256(manifest_sha256):
        raise ReadinessError("implementation manifest file SHA-256 is malformed")
    if not _is_sha256(payload["content_sha256"]):
        raise ReadinessError("implementation manifest content SHA-256 is malformed")
    if _content_sha256(payload) != payload["content_sha256"]:
        raise ReadinessError("implementation manifest content SHA-256 mismatch")
    source_map = _verify_source_map(
        payload["source_map"], root=root, file_hasher=file_hasher
    )
    inputs = _verify_input_bindings(payload["input_bindings"])
    command = _verify_command_identity(payload["command_identity"])
    if type(payload["oracle_config"]) is not dict or not payload["oracle_config"]:
        raise ReadinessError("implementation manifest oracle_config is empty")
    if type(payload["eligibility_config"]) is not dict or not payload["eligibility_config"]:
        raise ReadinessError("implementation manifest eligibility_config is empty")
    if not _canonical_equal(
        payload["exclusive_output"],
        {
            "path": str(OUTPUT_PATH.relative_to(ROOT)),
            "schema": RESULT_SCHEMA,
            "atomic_no_replace": True,
        },
    ):
        raise ReadinessError("implementation manifest exclusive-output contract changed")
    return VerifiedImplementation(
        sha256=manifest_sha256,
        payload=payload,
        source_map=source_map,
        input_bindings=inputs,
        command_identity=command,
    )


def load_and_verify_implementation_manifest(
    expected_sha256: str,
    *,
    manifest_path: Path = IMPLEMENTATION_MANIFEST_PATH,
    root: Path = ROOT,
    file_hasher: Callable[[Path], str] = _hash_file,
) -> VerifiedImplementation:
    if not _is_sha256(expected_sha256):
        raise ReadinessError("--implementation-manifest-sha256 must be lowercase SHA-256")
    payload_bytes = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ReadinessError("implementation manifest file SHA-256 mismatch")
    payload = _strict_json_loads(payload_bytes, label="implementation manifest")
    return _verify_implementation_payload(
        payload,
        manifest_sha256=actual_sha256,
        root=root,
        file_hasher=file_hasher,
    )


def verify_runtime_input_files(
    implementation: VerifiedImplementation,
    *,
    root: Path = ROOT,
    file_hasher: Callable[[Path], str] = _hash_file,
) -> dict[str, int]:
    ledger = {f"{role}_hash_reads": 0 for role in INPUT_PATHS}
    for role, record in implementation.input_bindings.items():
        if role == "prior_comparator":
            continue
        path = root / str(record["path"])
        if not path.is_file():
            raise ReadinessError(f"authorized input {role} is missing")
        if file_hasher(path) != record["sha256"]:
            raise ReadinessError(f"authorized input {role} SHA-256 mismatch")
        ledger[f"{role}_hash_reads"] += 1
    if ledger["prior_comparator_hash_reads"] != 0:
        raise AssertionError("prior comparator must remain unopened")
    return ledger


def _resolved_command_identity(implementation_sha256: str) -> dict[str, Any]:
    return {
        **COMMAND_IDENTITY_TEMPLATE,
        "argv": [
            COMMAND_IDENTITY_TEMPLATE["executable"],
            COMMAND_IDENTITY_TEMPLATE["script"],
            COMMAND_IDENTITY_TEMPLATE["argument"],
            implementation_sha256,
        ],
    }


def configure_single_thread_cpu_worker() -> dict[str, str | None]:
    """Cap nested native pools before any project/numpy import."""

    previous = {name: os.environ.get(name) for name in CPU_THREAD_CAP_ENV}
    for name in CPU_THREAD_CAP_ENV:
        os.environ[name] = "1"
    return previous


def restore_cpu_thread_environment(previous: Mapping[str, str | None]) -> None:
    for name in CPU_THREAD_CAP_ENV:
        value = previous.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _run_indexed_eligibility_job(job: tuple[int, Any, Any, Any]) -> tuple[int, dict]:
    """Worker boundary over already-loaded, verified in-memory objects."""

    index, manifest, policy, config = job
    from lewm.benchmarks.go2_physical_eligibility import (
        audit_physical_scene_eligibility,
    )

    return index, audit_physical_scene_eligibility(
        manifest,
        policy=policy,
        config=config,
    ).to_dict()


def merge_indexed_payloads(
    indexed_payloads: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    expected_scene_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Restore exact manifest order independent of worker completion order."""

    by_index: dict[int, dict[str, Any]] = {}
    for index, payload in indexed_payloads:
        if type(index) is not int or not 0 <= index < len(expected_scene_ids):
            raise ReadinessError("worker result index is outside the fixed scene panel")
        if index in by_index:
            raise ReadinessError("worker result index is duplicated")
        if not isinstance(payload, Mapping):
            raise ReadinessError("worker result payload is not a mapping")
        if payload.get("scene_id") != expected_scene_ids[index]:
            raise ReadinessError("worker result scene identity changed")
        by_index[index] = dict(payload)
    if set(by_index) != set(range(len(expected_scene_ids))):
        raise ReadinessError("worker result panel is incomplete")
    return [by_index[index] for index in range(len(expected_scene_ids))]


def run_parallel_eligibility_jobs(
    *,
    scene_ids: Sequence[str],
    scene_manifests: Mapping[str, Any],
    policy: Any,
    config: Any,
    workers: int = WORKER_COUNT,
) -> dict[str, dict[str, Any]]:
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ReadinessError("eligibility workers must be an exact integer in [1, 8]")
    jobs = [
        (index, scene_manifests[scene_id], policy, config)
        for index, scene_id in enumerate(scene_ids)
    ]
    if workers == 1:
        indexed = [_run_indexed_eligibility_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=configure_single_thread_cpu_worker,
        ) as executor:
            futures = [executor.submit(_run_indexed_eligibility_job, job) for job in jobs]
            indexed = [future.result() for future in as_completed(futures)]
    ordered = merge_indexed_payloads(indexed, expected_scene_ids=scene_ids)
    return {
        scene_id: report for scene_id, report in zip(scene_ids, ordered, strict=True)
    }


def _load_project_api() -> SimpleNamespace:
    for source_root in (ROOT, ROOT / "lewm_worlds", ROOT / "lewm_genesis"):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    from lewm.benchmarks.go2_canonical_physical_claim_oracle_finalizer import (
        finalize_canonical_physical_claim_oracle_regression,
    )
    from lewm.benchmarks.go2_oracle_positive_control import (
        OracleConfig,
        run_development_suite,
    )
    from lewm.benchmarks.go2_physical_eligibility import (
        audit_physical_scene_eligibility,
        physical_config_from_geometry_contract,
        policy_from_geometry_contract,
    )
    from lewm.planning.geometry_contract import load_geometry_contract
    from lewm_genesis.lewm_contract import PrimitiveRegistry
    from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict

    return SimpleNamespace(
        OracleConfig=OracleConfig,
        PrimitiveRegistry=PrimitiveRegistry,
        audit_physical_scene_eligibility=audit_physical_scene_eligibility,
        finalize_suite=finalize_canonical_physical_claim_oracle_regression,
        load_geometry_contract=load_geometry_contract,
        manifest_sha256=manifest_sha256,
        parse_scene_manifest_dict=parse_scene_manifest_dict,
        physical_config_from_geometry_contract=physical_config_from_geometry_contract,
        policy_from_geometry_contract=policy_from_geometry_contract,
        run_development_suite=run_development_suite,
    )


def _load_development_protocol() -> Mapping[str, Any]:
    payload = _strict_json_loads(
        DEVELOPMENT_MANIFEST_PATH.read_bytes(), label="V4 development manifest"
    )
    if type(payload) is not dict or payload.get("schema") != "lewm_navigation_development_manifest_v0":
        raise ReadinessError("V4 development manifest schema changed")
    if payload.get("geometry_contract_sha256") != EXPECTED_GEOMETRY_SEMANTIC_SHA256:
        raise ReadinessError("V4 development manifest geometry identity changed")
    records = payload.get("validation_scenes")
    if type(records) is not list or len(records) != 24:
        raise ReadinessError("V4 development manifest must contain exactly 24 validation scenes")
    scene_ids: list[str] = []
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ReadinessError(f"development scene record {index} is not an object")
        scene_id = record.get("scene_id")
        if type(scene_id) is not str or not scene_id or scene_id in scene_ids:
            raise ReadinessError("development scene IDs must be exact, nonempty, and unique")
        if record.get("beacon_count") != 4:
            raise ReadinessError(f"development scene {scene_id} does not bind four objects")
        if record.get("fully_reachable") is not True or str(record.get("failure_reason", "")):
            raise ReadinessError(f"development scene {scene_id} is not fully eligible")
        scene_ids.append(scene_id)
    return payload


def _load_scene_manifests(
    api: SimpleNamespace,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for record in records:
        scene_id = str(record["scene_id"])
        family = str(record["family"])
        path = SCENE_CORPUS_PATH / "development" / family / scene_id / "manifest.json"
        manifest = api.parse_scene_manifest_dict(
            _strict_json_loads(path.read_bytes(), label=f"scene manifest {scene_id}")
        )
        if api.manifest_sha256(manifest) != str(record["manifest_sha256"]):
            raise ReadinessError(f"scene manifest {scene_id} SHA-256 mismatch")
        manifests[scene_id] = manifest
    return manifests


def _bound_input_identity(
    implementation: VerifiedImplementation,
) -> dict[str, Any]:
    inputs = json.loads(_canonical_bytes(implementation.input_bindings).decode("utf-8"))
    inputs["geometry_contract_sha256"] = EXPECTED_GEOMETRY_SEMANTIC_SHA256
    inputs["directional_policy_content_sha256"] = (
        EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256
    )
    inputs["oracle_config"] = implementation.payload["oracle_config"]
    inputs["physical_eligibility_config"] = implementation.payload[
        "eligibility_config"
    ]
    return inputs


def publish_json_atomic_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON artifact without a replacement race."""

    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    nonce = hashlib.sha256(
        f"{os.getpid()}:{destination}:{len(encoded)}".encode("utf-8")
    ).hexdigest()[:16]
    temporary = destination.parent / f".{destination.name}.{nonce}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"immutable canonical output already exists: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_finalized_result(finalization: object, *, path: Path = OUTPUT_PATH) -> None:
    """Publish only an independently passed, materialized finalization."""

    if (
        getattr(finalization, "passed", None) is not True
        or getattr(finalization, "finalized_payload", None) is None
    ):
        errors = getattr(finalization, "errors", ())
        raise ReadinessError("canonical oracle finalization failed: " + ";".join(errors))
    publish_json_atomic_no_replace(path, finalization.finalized_payload)


def run_authoritative_regression(implementation_sha256: str) -> Mapping[str, Any]:
    implementation = load_and_verify_implementation_manifest(implementation_sha256)
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"immutable canonical output already exists: {OUTPUT_PATH}")
    preflight_input_reads = verify_runtime_input_files(implementation)

    if any(os.environ.get(name) != "1" for name in CPU_THREAD_CAP_ENV):
        raise ReadinessError("parent CPU thread caps were not installed before imports")
    api = _load_project_api()
    protocol = _load_development_protocol()
    records = list(protocol["validation_scenes"])
    scene_ids = [str(record["scene_id"]) for record in records]
    scene_families = {
        str(record["scene_id"]): str(record["family"]) for record in records
    }
    expected_manifest_sha256 = {
        str(record["scene_id"]): str(record["manifest_sha256"])
        for record in records
    }
    expected_beacon_counts = {
        str(record["scene_id"]): int(record["beacon_count"]) for record in records
    }

    geometry = api.load_geometry_contract(GEOMETRY_PATH, repository_root=ROOT)
    config = api.OracleConfig.from_geometry_contract(geometry)
    if not _canonical_equal(asdict(config), implementation.payload["oracle_config"]):
        raise ReadinessError("effective oracle config differs from the reviewed manifest")
    registry = api.PrimitiveRegistry.from_yaml(PRIMITIVE_REGISTRY_PATH)
    scene_manifests = _load_scene_manifests(api, records)
    policy = api.policy_from_geometry_contract(geometry, repository_root=ROOT)
    if policy.content_sha256 != EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256:
        raise ReadinessError("loaded directional-policy content identity changed")
    oracle_suite = api.run_development_suite(
        scene_corpus=SCENE_CORPUS_PATH,
        split="development",
        family=None,
        scene_ids=scene_ids,
        scene_families=scene_families,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_beacon_counts=expected_beacon_counts,
        development_manifest=DEVELOPMENT_MANIFEST_PATH,
        registry=registry,
        config=config,
        geometry_contract=geometry,
        progress=None,
        workers=WORKER_COUNT,
        preloaded_scene_manifests=scene_manifests,
        preloaded_directional_policy=policy,
    )

    eligibility_config = api.physical_config_from_geometry_contract(geometry)
    if not _canonical_equal(
        asdict(eligibility_config), implementation.payload["eligibility_config"]
    ):
        raise ReadinessError(
            "effective physical eligibility config differs from the reviewed manifest"
        )
    eligibility_reports = run_parallel_eligibility_jobs(
        scene_ids=scene_ids,
        scene_manifests=scene_manifests,
        policy=policy,
        config=eligibility_config,
        workers=WORKER_COUNT,
    )
    _verify_source_map(implementation.source_map)
    post_execution_input_reads = verify_runtime_input_files(implementation)
    input_access_ledger = {
        "preflight_hash_reads": preflight_input_reads,
        "post_execution_hash_reads": post_execution_input_reads,
        "development_manifest_parse_calls": 1,
        "development_scene_manifest_parse_calls_by_parent": 24,
        "geometry_load_calls": 1,
        "primitive_registry_load_calls": 1,
        "directional_policy_load_calls": 1,
        "worker_runtime_input_file_opens": 0,
        "prior_comparator_payload_opens": 0,
        "heldout_payload_opens": 0,
        "sealed_payload_opens": 0,
        "g2_payload_opens": 0,
        "label_payload_opens": 0,
        "image_payload_opens": 0,
        "model_output_opens": 0,
    }
    bound_inputs = _bound_input_identity(implementation)
    command = _resolved_command_identity(implementation.sha256)
    candidate = {
        "schema": RESULT_SCHEMA,
        "binding_sha256": EXPECTED_BINDING_SHA256,
        "implementation_manifest_sha256": implementation.sha256,
        "source_map": implementation.source_map,
        "input_bindings": bound_inputs,
        "command": command,
        "evaluator_access_ledger": dict(ZERO_EVALUATOR_ACCESS_LEDGER),
        "input_access_ledger": input_access_ledger,
        "oracle_report": oracle_suite,
        "physical_eligibility_reports": eligibility_reports,
    }
    finalized = api.finalize_suite(
        candidate,
        scene_manifests=scene_manifests,
        expected_scene_ids=scene_ids,
        expected_binding_sha256=EXPECTED_BINDING_SHA256,
        expected_implementation_manifest_sha256=implementation.sha256,
        expected_source_map=implementation.source_map,
        expected_input_bindings=bound_inputs,
        expected_command=command,
        expected_directional_policy_content_sha256=(
            EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256
        ),
    )
    publish_finalized_result(finalized)
    return finalized.finalized_payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--implementation-manifest-sha256", required=True)
    raw = list(sys.argv[1:] if argv is None else argv)
    parsed = parser.parse_args(raw)
    if len(raw) != 2 or raw[0] != "--implementation-manifest-sha256":
        parser.error("authoritative runner requires the exact frozen argument form")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ReadinessError(f"authoritative runner must execute from {ROOT}")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3").resolve():
        raise ReadinessError("authoritative runner requires /usr/bin/python3")
    expected_pythonpath = os.pathsep.join(COMMAND_IDENTITY_TEMPLATE["pythonpath"])
    if os.environ.get("PYTHONPATH") != expected_pythonpath:
        raise ReadinessError("authoritative runner PYTHONPATH differs from frozen command")
    previous_thread_environment = configure_single_thread_cpu_worker()
    try:
        run_authoritative_regression(str(args.implementation_manifest_sha256))
    finally:
        restore_cpu_thread_environment(previous_thread_environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
