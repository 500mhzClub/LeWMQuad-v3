#!/usr/bin/env python3
"""Prepare, run, and bind the model-free projective-support label preflight.

The three explicit phases keep custody simple:

``prepare`` atomically reserves the one label root before development metadata is
opened and writes the exact 88-scene source binding; ``run`` materializes labels
without RGB/model/GPU access and executes the frozen oracle metric check; and
``authorize`` writes the separate one-attempt training binding.  Every output is
write-once and every phase fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.benchmarks import (  # noqa: E402
    go2_post_action_projective_support_corridor_contract_v1 as contract,
)
from lewm.benchmarks import (  # noqa: E402
    go2_post_action_projective_support_labels_v1 as labels,
)
from lewm.benchmarks import (  # noqa: E402
    go2_post_action_projective_support_metrics_v1 as metrics,
)
from lewm.benchmarks import (  # noqa: E402
    go2_post_action_projective_support_source_authority_v1 as authority,
)
from scripts.build_go2_post_action_projective_support_labels_v1 import (  # noqa: E402
    build_from_binding,
)


SOURCE_PURPOSES = (
    "source_frames_jsonl",
    "render_summary",
    "source_scene_manifest",
)
LABEL_PREFLIGHT_FAILURE_RELATIVE_PATH = (
    ".generated/go2_post_action_projective_support_labels_v3_preflight_failure.json"
)


def _rooted(root: Path, relative: str) -> Path:
    return (Path(root) / relative).absolute()


def _read_regular(path: Path, *, name: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{name} is not a regular non-symlink file")
    return path.read_bytes()


def _write_new(path: Path, raw: bytes) -> None:
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _source_custody(
    repository_root: Path,
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).absolute()
    manifest_raw = _read_regular(
        _rooted(root, authority.SOURCE_MANIFEST_RELATIVE_PATH),
        name="source manifest",
    )
    review_raw = _read_regular(
        _rooted(root, authority.SOURCE_REVIEW_RELATIVE_PATH),
        name="source review",
    )
    authority.validate_source_manifest(manifest_raw, root=root)
    authority.validate_source_review_receipt(
        review_raw,
        manifest_raw,
        root=root,
    )
    return (
        manifest_raw,
        review_raw,
        authority.source_manifest_binding(manifest_raw),
        authority.source_review_binding(review_raw),
    )


def _regular_input_binding(
    repository_root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    relative = str(expected["path"])
    path = _rooted(repository_root, relative)
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"bound preflight input is absent: {relative}")
    size = path.stat().st_size
    expected_size = expected.get("byte_count")
    if expected_size is not None and size != expected_size:
        raise PermissionError(f"bound preflight input byte count changed: {relative}")
    return {**dict(expected), "path": relative, "byte_count": size}


def _preflight_inputs(repository_root: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    runtime_paths = {
        "raw_manifest": contract.RAW_MANIFEST_RELATIVE_PATH,
        "raw_pairs": contract.RAW_PAIRS_RELATIVE_PATH,
        "raw_endpoints": contract.RAW_ENDPOINTS_RELATIVE_PATH,
        "raw_audit": contract.RAW_AUDIT_RELATIVE_PATH,
        "schedule": contract.SCHEDULE_RELATIVE_PATH,
    }
    geometry_names = (
        "geometry_contract",
        "directional_policy",
        "primitive_registry",
    )
    bindings = {
        name: _regular_input_binding(
            repository_root,
            contract.RUNTIME_BINDINGS[path],
        )
        for name, path in runtime_paths.items()
    }
    bindings.update(
        {
            name: _regular_input_binding(
                repository_root,
                contract.GEOMETRY_BINDINGS[name],
            )
            for name in geometry_names
        }
    )
    paths = {
        name: _rooted(repository_root, str(binding["path"]))
        for name, binding in bindings.items()
    }
    return paths, bindings


def source_records_from_raw_indexes_v1(raw_indexes: Any) -> list[dict[str, Any]]:
    provenance = raw_indexes.manifest.get("input_provenance")
    inventory = (
        provenance.get("source_payload_inventory")
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(inventory, list):
        raise PermissionError("raw source payload inventory is absent")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in inventory:
        if not isinstance(record, Mapping):
            raise PermissionError("raw source inventory record is malformed")
        scene = record.get("scene_id")
        purpose = record.get("purpose")
        if isinstance(scene, str) and purpose in SOURCE_PURPOSES:
            key = (scene, str(purpose))
            if key in by_key:
                raise PermissionError("raw source inventory repeats a scene/purpose")
            by_key[key] = record
    result: list[dict[str, Any]] = []
    for scene, shard in sorted(
        raw_indexes.shard_by_scene.items(),
        key=lambda item: (
            contract.ROLE_ORDER.index(str(item[1]["dataset_role"])),
            item[0],
        ),
    ):
        for purpose in SOURCE_PURPOSES:
            record = by_key.get((scene, purpose))
            if record is None:
                raise PermissionError("raw inventory is missing a bound scene source")
            result.append(
                {
                    "path": str(record["path"]),
                    "byte_count": int(record["byte_count"]),
                    "file_sha256": str(record["file_sha256"]),
                    "purpose": purpose,
                    "scene_id": scene,
                    "dataset_role": str(shard["dataset_role"]),
                    "family": str(shard["family"]),
                }
            )
    if len(result) != 264:
        raise PermissionError("label binding must contain exactly 264 source records")
    return result


def _label_preflight_authority() -> dict[str, Any]:
    return {
        "development_label_preflight_authorized": True,
        "rgb_decode_authorized": False,
        "tensor_or_gpu_authorized": False,
        "training_authorized": False,
        "checkpoint_authorized": False,
        "runtime_output_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "production_authorized": False,
        "promotion_authorized": False,
        "retry_or_resume_authorized": False,
    }


def prepare_label_execution_binding_v1(
    *,
    repository_root: Path = ROOT,
) -> Mapping[str, Any]:
    """Reserve once, validate metadata-only inputs, and write the exact binding."""

    root = Path(repository_root).absolute()
    contract.validate_label_v1_terminal_predecessor(root=root)
    contract.validate_label_v2_terminal_predecessor(root=root)
    _, _, source_manifest, source_review = _source_custody(root)
    labels.reserve_label_root_v1(
        root,
        source_manifest=source_manifest,
        independent_source_review=source_review,
    )
    access_ledger = labels.new_access_ledger_v1()
    schedule_prefix_sha256: str | None = None
    try:
        paths, input_bindings = _preflight_inputs(root)
        raw_indexes = labels.load_and_validate_raw_indexes(
            paths["raw_manifest"],
            paths["raw_pairs"],
            paths["raw_endpoints"],
            access_ledger=access_ledger,
        )
        labels.validate_raw_audit_v1(
            paths["raw_audit"],
            access_ledger=access_ledger,
        )
        schedule = labels.load_schedule_indices_v1(
            paths["schedule"],
            raw_indexes=raw_indexes,
            access_ledger=access_ledger,
        )
        schedule_prefix_sha256 = labels.canonical_json_sha256(list(schedule))
        if schedule_prefix_sha256 != contract.SCHEDULE_PREFIX_SHA256:
            raise PermissionError("frozen 16,000-presentation prefix changed")
        labels.load_geometry_inputs_v1(
            repository_root=root,
            geometry_path=paths["geometry_contract"],
            directional_policy_path=paths["directional_policy"],
            primitive_registry_path=paths["primitive_registry"],
            access_ledger=access_ledger,
        )
        binding = labels.with_content_sha256(
            {
                "schema": labels.EXECUTION_BINDING_SCHEMA,
                "status": "AUTHORIZED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
                "preregistration_commit": contract.PREREGISTRATION_COMMIT,
                "integrity_adapter_amendment": (
                    contract.integrity_adapter_amendment_binding()
                ),
                "schedule_schema_adapter_amendment": (
                    contract.schedule_schema_adapter_amendment_binding()
                ),
                "label_v1_terminal_predecessor_bindings": {
                    name: dict(binding)
                    for name, binding in (
                        contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS.items()
                    )
                },
                "label_v2_terminal_predecessor_bindings": {
                    name: dict(binding)
                    for name, binding in (
                        contract.LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS.items()
                    )
                },
                "source_manifest": source_manifest,
                "independent_source_review": source_review,
                "inputs": input_bindings,
                "output_directory": contract.LABEL_ROOT_RELATIVE_PATH,
                "source_records": source_records_from_raw_indexes_v1(raw_indexes),
                "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
                "preparation_access_ledger": dict(access_ledger),
                "authority": _label_preflight_authority(),
            }
        )
        labels.validate_execution_binding_v1(binding, raw_indexes=raw_indexes)
        binding_path = _rooted(root, labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH)
        _write_new(
            binding_path,
            labels.canonical_json_bytes(binding) + b"\n",
        )
        return binding
    except BaseException as error:
        writer = getattr(labels, "write_label_failure_v1", None)
        if callable(writer):
            writer(
                root,
                phase="prepare_execution_binding",
                error=error,
                source_manifest=source_manifest,
                independent_source_review=source_review,
                schedule_prefix_sha256=schedule_prefix_sha256,
                access_ledger=access_ledger,
            )
        raise


def _role_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(rows) == 0 or len(rows) % len(contract.ACTION_VOCABULARY) != 0:
        raise ValueError("role labels are not complete state/action groups")
    groups = tuple(
        rows[offset : offset + len(contract.ACTION_VOCABULARY)]
        for offset in range(0, len(rows), len(contract.ACTION_VOCABULARY))
    )
    station_safe = np.asarray(
        [[row["station_safe"] for row in group] for group in groups],
        dtype=bool,
    )
    immediate = np.asarray(
        [
            [bool(row["immediate_primitive"]["feasible"]) for row in group]
            for group in groups
        ],
        dtype=bool,
    )
    blind = np.asarray(
        [
            [bool(row["blind_bridge"]["feasible"]) for row in group]
            for group in groups
        ],
        dtype=bool,
    )
    scenes = tuple(str(group[0]["scene_id"]) for group in groups)
    families = tuple(str(group[0]["family"]) for group in groups)
    endpoints = tuple(str(group[0]["current_endpoint_sha256"]) for group in groups)
    if station_safe.shape != (len(groups), 9, 11):
        raise ValueError("role station labels changed shape")
    return station_safe, immediate, blind, scenes, families, endpoints


def wrong_rgb_mapping_binding_v1(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    pairs_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    enforce_frozen_counts: bool = True,
) -> dict[str, Any]:
    endpoints_by_role: dict[str, tuple[tuple[str, str, str], ...]] = {}
    per_role: dict[str, dict[str, Any]] = {}
    all_endpoints: list[tuple[str, str, str]] = []
    paired_next_collisions: list[list[Any]] = []
    for role in contract.ROLE_ORDER:
        _, _, _, scenes, _, endpoints = _role_arrays(rows_by_role[role])
        values = tuple((role, scene, endpoint) for scene, endpoint in zip(scenes, endpoints))
        mapping = metrics.wrong_rgb_endpoint_mapping(values)
        pairs = tuple(pairs_by_role[role])
        if len(pairs) != len(values):
            raise PermissionError(f"{role} wrong-RGB pair population changed")
        for state_index, (identity, pair) in enumerate(
            zip(values, pairs, strict=True)
        ):
            pair_identity = (
                pair.get("dataset_role"),
                pair.get("scene_id"),
                pair.get("current_endpoint_sha256"),
            )
            if pair_identity != identity:
                raise PermissionError(
                    f"{role} wrong-RGB pair order changed at state {state_index}"
                )
            mapped_endpoint = mapping.by_endpoint[identity]
            if mapped_endpoint == pair.get("next_endpoint_sha256"):
                paired_next_collisions.append([
                    role,
                    state_index,
                    identity[1],
                    identity[2],
                    str(pair.get("next_endpoint_sha256")),
                    mapped_endpoint,
                ])
        endpoints_by_role[role] = values
        all_endpoints.extend(values)
        per_role[role] = {
            "row_count": len(mapping.rows),
            "mapping_sha256": mapping.mapping_sha256,
        }
    combined = metrics.wrong_rgb_endpoint_mapping(tuple(all_endpoints))
    if paired_next_collisions:
        failure_mapping = {
            "algorithm": "role_scene_local_lexicographic_cyclic_derangement_v1",
            "roles": list(contract.ROLE_ORDER),
            "row_count": len(combined.rows),
            "mapping_sha256": combined.mapping_sha256,
            "per_role": per_role,
            "paired_next_collision_count": len(paired_next_collisions),
            "paired_next_collision_rows_sha256": (
                contract.canonical_json_sha256(paired_next_collisions)
            ),
            "mapped_endpoint_is_never_paired_next": False,
        }
        error = PermissionError(
            "wrong-RGB mapping selected a paired future endpoint"
        )
        error.wrong_rgb_mapping = failure_mapping  # type: ignore[attr-defined]
        raise error
    result = {
        "algorithm": "role_scene_local_lexicographic_cyclic_derangement_v1",
        "roles": list(contract.ROLE_ORDER),
        "row_count": len(combined.rows),
        "mapping_sha256": combined.mapping_sha256,
        "per_role": per_role,
        "paired_next_collision_count": 0,
        "paired_next_collision_rows_sha256": contract.canonical_json_sha256([]),
        "mapped_endpoint_is_never_paired_next": True,
    }
    expected_counts = {
        role: contract.ROLE_COUNTS[role]["states"] for role in contract.ROLE_ORDER
    }
    if enforce_frozen_counts and (
        result["row_count"] != contract.TOTAL_STATES
        or {role: item["row_count"] for role, item in per_role.items()}
        != expected_counts
    ):
        raise PermissionError("wrong-RGB endpoint mapping population changed")
    return result


def action_prior_binding_v1(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    enforce_frozen_count: bool = True,
) -> dict[str, Any]:
    station_safe, _, _, _, _, _ = _role_arrays(train_rows)
    probabilities = metrics.action_prior_probabilities(station_safe).tolist()
    if enforce_frozen_count and station_safe.shape[0] != contract.ROLE_COUNTS["train"]["states"]:
        raise PermissionError("action-prior train population changed")
    return {
        "source_role": "train",
        "source_roles": ["train"],
        "source_state_count": int(station_safe.shape[0]),
        "action_order": list(contract.ACTION_VOCABULARY),
        "station_count": contract.STATION_COUNT,
        "shape": [9, 11],
        "probabilities": probabilities,
        "probabilities_sha256": contract.canonical_json_sha256(probabilities),
    }


def oracle_metric_mapping_v1(result: Any) -> dict[str, Any]:
    return {
        "status": "PASS" if bool(result.passed) else "STOP",
        "passed": bool(result.passed),
        "failed_checks": list(result.failed_checks),
        "checks": dict(result.checks),
    }


def _label_file_bindings(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    by_name = {str(record["path"]): record for record in manifest["files"]}
    result: dict[str, dict[str, Any]] = {}
    for relative in authority.LABEL_FILE_PATHS:
        name = Path(relative).name
        record = by_name[name]
        result[relative] = {
            "path": relative,
            "file_sha256": str(record["file_sha256"]),
            "byte_count": int(record["byte_count"]),
        }
    return result


def run_label_preflight_v1(
    *,
    repository_root: Path = ROOT,
) -> Mapping[str, Any]:
    """Materialize once, run the oracle pipeline, and write the PASS receipt."""

    root = Path(repository_root).absolute()
    _, _, source_manifest_binding, source_review_binding = _source_custody(root)
    binding_path = _rooted(root, labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH)
    phase = "open_label_builder_execution_binding"
    binding_raw: bytes | None = None
    manifest: Mapping[str, Any] | None = None
    manifest_raw: bytes | None = None
    label_files: Mapping[str, Any] | None = None
    oracle_mapping: Mapping[str, Any] | None = None
    wrong_rgb: Mapping[str, Any] | None = None
    action_prior: Mapping[str, Any] | None = None
    preflight_access_ledger = labels.new_access_ledger_v1()
    try:
        binding_raw = _read_regular(binding_path, name="label execution binding")
        phase = "materialize_label_bundle"
        manifest = build_from_binding(binding_path, repository_root=root)
        phase = "validate_materialized_label_bundle"
        manifest_path = _rooted(root, contract.LABEL_MANIFEST_RELATIVE_PATH)
        manifest_raw = _read_regular(manifest_path, name="label manifest")
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        label_files = _label_file_bindings(manifest)
        role_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
        by_name = {str(record["path"]): record for record in manifest["files"]}
        for role in contract.ROLE_ORDER:
            name = f"{role}.jsonl"
            role_rows[role] = labels.load_role_labels_v1(
                _rooted(root, contract.LABEL_ROLE_RELATIVE_PATHS[role]),
                role=role,
                expected_file_sha256=str(by_name[name]["file_sha256"]),
            )
        paths, _ = _preflight_inputs(root)
        raw_indexes = labels.load_and_validate_raw_indexes(
            paths["raw_manifest"],
            paths["raw_pairs"],
            paths["raw_endpoints"],
            access_ledger=preflight_access_ledger,
        )
        pairs_by_role = {
            role: tuple(
                pair
                for pair in raw_indexes.pairs
                if pair["dataset_role"] == role
            )
            for role in contract.ROLE_ORDER
        }
        calibration = _role_arrays(role_rows["probability_calibration"])
        selection = _role_arrays(role_rows["checkpoint_selection"])
        phase = "oracle_metric_pipeline"
        oracle = metrics.oracle_metric_pipeline_preflight(
            calibration[0],
            selection[0],
            selection[3],
            selection[4],
            selection[1],
            selection[2],
            calibration_family_ids=calibration[4],
        )
        oracle_mapping = oracle_metric_mapping_v1(oracle)
        if not oracle.passed:
            raise PermissionError(
                "oracle metric-pipeline preflight STOP: "
                + ",".join(oracle.failed_checks)
            )
        phase = "wrong_rgb_paired_future_validation"
        wrong_rgb = wrong_rgb_mapping_binding_v1(role_rows, pairs_by_role)
        phase = "train_only_action_prior"
        action_prior = action_prior_binding_v1(role_rows["train"])
        phase = "publish_label_preflight_receipt"
        receipt = authority.build_label_preflight_receipt(
            binding_raw,
            manifest_raw,
            label_files,
            oracle_metric_pipeline=oracle_mapping,
            wrong_rgb_mapping=wrong_rgb,
            action_prior=action_prior,
        )
        receipt_path = _rooted(root, authority.LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH)
        _write_new(receipt_path, authority.canonical_document_bytes(receipt))
        if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
            raise PermissionError("label manifest changed during preflight finalization")
        return receipt
    except BaseException as error:
        path = _rooted(root, LABEL_PREFLIGHT_FAILURE_RELATIVE_PATH)
        if not path.exists():
            if wrong_rgb is None:
                candidate = getattr(error, "wrong_rgb_mapping", None)
                if isinstance(candidate, Mapping):
                    wrong_rgb = dict(candidate)
            builder_binding = None
            if binding_raw is not None:
                builder_binding = {
                    "path": labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH,
                    "file_sha256": hashlib.sha256(binding_raw).hexdigest(),
                    "byte_count": len(binding_raw),
                }
            label_manifest_binding = None
            if manifest_raw is not None:
                label_manifest_binding = {
                    "path": contract.LABEL_MANIFEST_RELATIVE_PATH,
                    "file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                    "byte_count": len(manifest_raw),
                    "content_sha256": (
                        manifest.get("content_sha256")
                        if isinstance(manifest, Mapping)
                        else None
                    ),
                }
            protected_keys = (
                "rgb_opens",
                "checkpoint_opens",
                "runtime_output_opens",
                "g2_opens",
                "navigation_opens",
                "heldout_opens",
                "sealed_opens",
                "production_opens",
            )
            failure = contract.with_content_sha256(
                {
                    "schema": f"{contract.SCHEMA_PREFIX}_label_preflight_failure_v1",
                    "status": "TERMINAL_LABEL_PREFLIGHT_STOP",
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "source_manifest": source_manifest_binding,
                    "independent_source_review": source_review_binding,
                    "label_builder_execution_binding": builder_binding,
                    "label_manifest": label_manifest_binding,
                    "label_files": label_files,
                    "oracle_metric_pipeline": oracle_mapping,
                    "wrong_rgb_mapping": wrong_rgb,
                    "action_prior": action_prior,
                    "preflight_access_ledger": dict(preflight_access_ledger),
                    "protected_access_counts": {
                        key: int(preflight_access_ledger[key])
                        for key in protected_keys
                    },
                    "retry": False,
                    "resume": False,
                    "training_authorized": False,
                    "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
                }
            )
            _write_new(path, contract.canonical_json_bytes(failure) + b"\n")
        raise


def authorize_training_attempt_v1(
    *,
    authorizer: str,
    repository_root: Path = ROOT,
) -> Mapping[str, Any]:
    """Bind the reviewed source and PASS label receipt without opening RGB/model."""

    root = Path(repository_root).absolute()
    source_manifest_raw, source_review_raw, _, _ = _source_custody(root)
    label_manifest_path = _rooted(root, contract.LABEL_MANIFEST_RELATIVE_PATH)
    label_manifest_raw = _read_regular(label_manifest_path, name="label manifest")
    label_manifest = contract.parse_canonical_json(
        label_manifest_raw,
        name="label manifest",
    )
    label_bindings = _label_file_bindings(label_manifest)
    label_binding_raw = _read_regular(
        _rooted(root, labels.LABEL_EXECUTION_BINDING_RELATIVE_PATH),
        name="label execution binding",
    )
    receipt_raw = _read_regular(
        _rooted(root, authority.LABEL_PREFLIGHT_RECEIPT_RELATIVE_PATH),
        name="label preflight receipt",
    )
    execution = authority.build_execution_binding(
        source_manifest_raw,
        source_review_raw,
        label_manifest_raw,
        label_bindings,
        authorizer=authorizer,
        label_builder_execution_binding_raw=label_binding_raw,
        label_preflight_receipt_raw=receipt_raw,
        root=root,
    )
    output = _rooted(root, authority.EXECUTION_BINDING_RELATIVE_PATH)
    _write_new(output, authority.canonical_document_bytes(execution))
    return execution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("run")
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--authorizer", required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        result = prepare_label_execution_binding_v1(
            repository_root=args.repository_root,
        )
    elif args.phase == "run":
        result = run_label_preflight_v1(repository_root=args.repository_root)
    else:
        result = authorize_training_attempt_v1(
            authorizer=args.authorizer,
            repository_root=args.repository_root,
        )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "status": result["status"],
                "content_sha256": result["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
