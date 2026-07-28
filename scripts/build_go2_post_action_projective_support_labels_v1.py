#!/usr/bin/env python3
"""Materialize the frozen development-only projective-support labels once."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.benchmarks.go2_post_action_projective_support_labels_v1 import (  # noqa: E402
    LabelContractError,
    LABEL_EXECUTION_BINDING_RELATIVE_PATH,
    LABEL_OUTPUT_RELATIVE_PATH,
    SOURCE_MANIFEST_RELATIVE_PATH,
    SOURCE_REVIEW_RELATIVE_PATH,
    SCHEDULE_PREFIX_SHA256,
    canonical_json_sha256,
    claim_label_builder_v1,
    load_and_validate_raw_indexes,
    load_execution_binding_file_v1,
    load_geometry_inputs_v1,
    load_label_reservation_v1,
    load_schedule_indices_v1,
    materialize_role_labels_v1,
    new_access_ledger_v1,
    validate_execution_binding_envelope_v1,
    validate_execution_binding_v1,
    validate_raw_audit_v1,
    write_label_failure_v1,
)


def _path(value: object, *, repository_root: Path) -> Path:
    candidate = Path(str(value))
    return (candidate if candidate.is_absolute() else repository_root / candidate).absolute()


def _bound_inputs(
    binding: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Path]:
    inputs = binding.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LabelContractError("execution binding has no input map")
    result: dict[str, Path] = {}
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise LabelContractError(f"execution input {name} is malformed")
        result[str(name)] = _path(record.get("path"), repository_root=repository_root)
    return result


def _validate_opened_byte_counts(
    binding: Mapping[str, Any],
    inputs: Mapping[str, Path],
    *names: str,
) -> None:
    records = binding["inputs"]
    for name in names:
        if inputs[name].stat().st_size != records[name]["byte_count"]:
            raise LabelContractError(f"execution input {name} byte count changed")


def _read_authority_artifact(
    record: Mapping[str, Any],
    *,
    repository_root: Path,
    access_ledger: dict[str, int],
    access_key: str,
) -> bytes:
    path = _path(record["path"], repository_root=repository_root)
    if path.is_symlink() or not path.is_file():
        raise LabelContractError(f"authority artifact is not a regular non-symlink: {path}")
    access_ledger[access_key] += 1
    raw = path.read_bytes()
    if (
        len(raw) != record["byte_count"]
        or hashlib.sha256(raw).hexdigest() != record["file_sha256"]
    ):
        raise LabelContractError(f"authority artifact bytes changed: {record['path']}")
    return raw


def _validate_source_authority_artifacts(
    binding: Mapping[str, Any],
    *,
    repository_root: Path,
    access_ledger: dict[str, int],
) -> None:
    """Recursively validate the frozen source closure/review before raw opens."""

    source_record = binding["source_manifest"]
    review_record = binding["independent_source_review"]
    if (
        source_record["path"] != SOURCE_MANIFEST_RELATIVE_PATH
        or review_record["path"] != SOURCE_REVIEW_RELATIVE_PATH
    ):
        raise LabelContractError("label source-authority paths changed")
    source_raw = _read_authority_artifact(
        source_record,
        repository_root=repository_root,
        access_ledger=access_ledger,
        access_key="source_manifest_opens",
    )
    review_raw = _read_authority_artifact(
        review_record,
        repository_root=repository_root,
        access_ledger=access_ledger,
        access_key="independent_source_review_opens",
    )
    access_ledger["source_authority_validation_calls"] += 1
    authority = importlib.import_module(
        "lewm.benchmarks.go2_post_action_projective_support_source_authority_v1"
    )
    source = authority.validate_source_manifest(source_raw, root=repository_root)
    review = authority.validate_source_review_receipt(
        review_raw,
        source_raw,
        root=repository_root,
    )
    if (
        authority.source_manifest_binding(source_raw) != dict(source_record)
        or authority.source_review_binding(review_raw) != dict(review_record)
        or review.get("source_manifest") != dict(source_record)
        or source.get("preregistration", {}).get("commit")
        != authority.contract.PREREGISTRATION_COMMIT
    ):
        raise PermissionError("label source manifest/review recursive identity changed")


def build_from_binding(binding_path: Path, *, repository_root: Path) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    binding_path = Path(binding_path).absolute()
    expected_binding_path = repository_root / LABEL_EXECUTION_BINDING_RELATIVE_PATH
    reservation = load_label_reservation_v1(repository_root)
    output = repository_root / LABEL_OUTPUT_RELATIVE_PATH
    if any(
        (output / name).exists()
        for name in ("builder_claim.json", "failure.json", "manifest.json")
    ):
        raise PermissionError("label builder invocation was already consumed")

    access_ledger = new_access_ledger_v1()
    phase = "load_execution_binding"
    binding: Mapping[str, Any] | None = None
    schedule_prefix_sha256: str | None = None
    claimed = False
    try:
        if binding_path != expected_binding_path:
            raise PermissionError(
                "builder requires the exact label execution-binding path"
            )
        binding = load_execution_binding_file_v1(
            binding_path,
            repository_root=repository_root,
            access_ledger=access_ledger,
        )
        phase = "validate_execution_binding_envelope"
        validate_execution_binding_envelope_v1(binding)
        if (
            binding["source_manifest"] != reservation["source_manifest"]
            or binding["independent_source_review"]
            != reservation["independent_source_review"]
        ):
            raise PermissionError("execution binding escaped its label reservation")
        claim_label_builder_v1(
            repository_root,
            execution_binding_content_sha256=str(binding["content_sha256"]),
        )
        claimed = True

        phase = "validate_source_manifest_and_review"
        _validate_source_authority_artifacts(
            binding,
            repository_root=repository_root,
            access_ledger=access_ledger,
        )
        inputs = _bound_inputs(binding, repository_root=repository_root)

        phase = "load_raw_indexes"
        raw = load_and_validate_raw_indexes(
            inputs["raw_manifest"],
            inputs["raw_pairs"],
            inputs["raw_endpoints"],
            access_ledger=access_ledger,
        )
        _validate_opened_byte_counts(
            binding, inputs, "raw_manifest", "raw_pairs", "raw_endpoints"
        )
        phase = "validate_raw_audit_and_source_records"
        validate_raw_audit_v1(inputs["raw_audit"], access_ledger=access_ledger)
        _validate_opened_byte_counts(binding, inputs, "raw_audit")
        source_records = validate_execution_binding_v1(binding, raw_indexes=raw)

        phase = "load_geometry"
        geometry = load_geometry_inputs_v1(
            repository_root=repository_root,
            geometry_path=inputs["geometry_contract"],
            directional_policy_path=inputs["directional_policy"],
            primitive_registry_path=inputs["primitive_registry"],
            access_ledger=access_ledger,
        )
        _validate_opened_byte_counts(
            binding,
            inputs,
            "geometry_contract",
            "directional_policy",
            "primitive_registry",
        )
        phase = "load_schedule"
        schedule_indices = load_schedule_indices_v1(
            inputs["schedule"],
            raw_indexes=raw,
            access_ledger=access_ledger,
        )
        _validate_opened_byte_counts(binding, inputs, "schedule")
        schedule_prefix_sha256 = canonical_json_sha256(list(schedule_indices))
        if schedule_prefix_sha256 != SCHEDULE_PREFIX_SHA256:
            raise LabelContractError("frozen schedule-prefix SHA-256 changed")

        phase = "materialize_and_publish_manifest_last"
        return materialize_role_labels_v1(
            raw_indexes=raw,
            geometry_inputs=geometry,
            source_records_by_scene=source_records,
            execution_binding=binding,
            repository_root=repository_root,
            output_directory=output,
            schedule_indices=schedule_indices,
            access_ledger=access_ledger,
        )
    except BaseException as error:
        # A racing second claimant must not write into the active first attempt.
        if claimed or not (output / "builder_claim.json").exists():
            try:
                write_label_failure_v1(
                    repository_root,
                    phase=phase,
                    error=error,
                    source_manifest=reservation["source_manifest"],
                    independent_source_review=reservation[
                        "independent_source_review"
                    ],
                    binding_content_sha256=(
                        str(binding["content_sha256"])
                        if binding is not None
                        and isinstance(binding.get("content_sha256"), str)
                        else None
                    ),
                    schedule_prefix_sha256=schedule_prefix_sha256,
                    access_ledger=access_ledger,
                )
            except FileExistsError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-binding", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    manifest = build_from_binding(
        args.execution_binding.absolute(),
        repository_root=args.repository_root.absolute(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "state_count": manifest["state_count"],
                "action_row_count": manifest["action_row_count"],
                "station_label_count": manifest["station_label_count"],
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
